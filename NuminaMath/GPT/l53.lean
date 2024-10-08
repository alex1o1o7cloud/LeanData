import Mathlib

namespace exam_total_boys_l53_53711

theorem exam_total_boys (T F : ℕ) (avg_total avg_passed avg_failed : ℕ) 
    (H1 : avg_total = 40) (H2 : avg_passed = 39) (H3 : avg_failed = 15) (H4 : 125 > 0) (H5 : 125 * avg_passed + (T - 125) * avg_failed = T * avg_total) : T = 120 :=
by
  sorry

end exam_total_boys_l53_53711


namespace measure_of_angle_A_l53_53202

-- Define the given conditions
variables (A B : ℝ)
axiom supplementary : A + B = 180
axiom measure_rel : A = 7 * B

-- The theorem statement to prove
theorem measure_of_angle_A : A = 157.5 :=
by
  -- proof steps would go here, but are omitted
  sorry

end measure_of_angle_A_l53_53202


namespace man_total_pay_l53_53708

def regular_rate : ℕ := 3
def regular_hours : ℕ := 40
def overtime_hours : ℕ := 13

def regular_pay : ℕ := regular_rate * regular_hours
def overtime_rate : ℕ := 2 * regular_rate
def overtime_pay : ℕ := overtime_rate * overtime_hours

def total_pay : ℕ := regular_pay + overtime_pay

theorem man_total_pay : total_pay = 198 := by
  sorry

end man_total_pay_l53_53708


namespace abc_not_less_than_two_l53_53589

theorem abc_not_less_than_two (a b c : ℝ) (h : a + b + c = 6) : a ≥ 2 ∨ b ≥ 2 ∨ c ≥ 2 :=
sorry

end abc_not_less_than_two_l53_53589


namespace range_of_a_if_intersection_empty_range_of_a_if_union_equal_B_l53_53250

-- Definitions for the sets A and B
def setA (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < a + 1}
def setB : Set ℝ := {x : ℝ | x < -1 ∨ x > 2}

-- Question (1): Proof statement for A ∩ B = ∅ implying 0 ≤ a ≤ 1
theorem range_of_a_if_intersection_empty (a : ℝ) :
  (setA a ∩ setB = ∅) → (0 ≤ a ∧ a ≤ 1) := 
sorry

-- Question (2): Proof statement for A ∪ B = B implying a ≤ -2 or a ≥ 3
theorem range_of_a_if_union_equal_B (a : ℝ) :
  (setA a ∪ setB = setB) → (a ≤ -2 ∨ 3 ≤ a) := 
sorry

end range_of_a_if_intersection_empty_range_of_a_if_union_equal_B_l53_53250


namespace find_six_quotients_l53_53911

def is_5twos_3ones (n: ℕ) : Prop :=
  n.digits 10 = [2, 2, 2, 2, 2, 1, 1, 1]

def divides_by_7 (n: ℕ) : Prop :=
  n % 7 = 0

theorem find_six_quotients:
  ∃ n₁ n₂ n₃ n₄ n₅: ℕ, 
    n₁ ≠ n₂ ∧ n₁ ≠ n₃ ∧ n₂ ≠ n₃ ∧ n₁ ≠ n₄ ∧ n₂ ≠ n₄ ∧ n₃ ≠ n₄ ∧ n₁ ≠ n₅ ∧ n₂ ≠ n₅ ∧ n₃ ≠ n₅ ∧ n₄ ≠ n₅ ∧
    is_5twos_3ones n₁ ∧ is_5twos_3ones n₂ ∧ is_5twos_3ones n₃ ∧ is_5twos_3ones n₄ ∧ is_5twos_3ones n₅ ∧
    divides_by_7 n₁ ∧ divides_by_7 n₂ ∧ divides_by_7 n₃ ∧ divides_by_7 n₄ ∧ divides_by_7 n₅ ∧
    n₁ / 7 = 1744603 ∧ n₂ / 7 = 3031603 ∧ n₃ / 7 = 3160303 ∧ n₄ / 7 = 3017446 ∧ n₅ / 7 = 3030316 :=
sorry

end find_six_quotients_l53_53911


namespace loads_of_laundry_l53_53713

theorem loads_of_laundry (families : ℕ) (days : ℕ) (adults_per_family : ℕ) (children_per_family : ℕ)
  (adult_towels_per_day : ℕ) (child_towels_per_day : ℕ) (initial_capacity : ℕ) (reduced_capacity : ℕ)
  (initial_days : ℕ) (remaining_days : ℕ) : 
  families = 7 → days = 12 → adults_per_family = 2 → children_per_family = 4 → 
  adult_towels_per_day = 2 → child_towels_per_day = 1 → initial_capacity = 8 → 
  reduced_capacity = 6 → initial_days = 6 → remaining_days = 6 → 
  (families * (adults_per_family * adult_towels_per_day + children_per_family * child_towels_per_day) * initial_days / initial_capacity) +
  (families * (adults_per_family * adult_towels_per_day + children_per_family * child_towels_per_day) * remaining_days / reduced_capacity) = 98 :=
by 
  intros _ _ _ _ _ _ _ _ _ _
  sorry

end loads_of_laundry_l53_53713


namespace solve_inequalities_l53_53962

-- Define the interval [-1, 1]
def interval := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

-- State the problem
theorem solve_inequalities :
  {x : ℝ | 3 * x^2 + 2 * x - 9 ≤ 0 ∧ x ≥ -1} = interval := 
sorry

end solve_inequalities_l53_53962


namespace handmade_ornaments_l53_53138

noncomputable def handmade_more_than_1_sixth(O : ℕ) (h1 : (1 / 3 : ℚ) * O = 20) (h2 : (1 / 2 : ℚ) * (handmade : ℕ) = 20) : Prop :=
  handmade - (1 / 6 * O) = 20

theorem handmade_ornaments (O handmade : ℕ) (h1 : (1 / 3 : ℚ) * O = 20) (h2 : (1 / 2 : ℚ) * handmade = 20) :
  handmade_more_than_1_sixth O h1 h2 :=
by
  sorry

end handmade_ornaments_l53_53138


namespace projection_of_AB_on_AC_l53_53328

noncomputable def A : ℝ × ℝ := (-1, 1)
noncomputable def B : ℝ × ℝ := (0, 3)
noncomputable def C : ℝ × ℝ := (3, 4)

noncomputable def vectorAB := (B.1 - A.1, B.2 - A.2)
noncomputable def vectorAC := (C.1 - A.1, C.2 - A.2)

noncomputable def dotProduct (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem projection_of_AB_on_AC :
  (dotProduct vectorAB vectorAC) / (magnitude vectorAC) = 2 :=
  sorry

end projection_of_AB_on_AC_l53_53328


namespace general_term_arithmetic_sequence_sum_terms_sequence_l53_53865

noncomputable def a_n (n : ℕ) : ℤ := 
  2 * (n : ℤ) - 1

theorem general_term_arithmetic_sequence :
  ∀ n : ℕ, a_n n = 2 * (n : ℤ) - 1 :=
by sorry

noncomputable def c (n : ℕ) : ℚ := 
  1 / ((2 * (n : ℤ) - 1) * (2 * (n + 1) - 1))

noncomputable def T_n (n : ℕ) : ℚ :=
  (1 / 2 : ℚ) * (1 - (1 / (2 * (n : ℤ) + 1)))

theorem sum_terms_sequence :
  ∀ n : ℕ, T_n n = (n : ℚ) / (2 * (n : ℤ) + 1) :=
by sorry

end general_term_arithmetic_sequence_sum_terms_sequence_l53_53865


namespace trigonometric_identity_proof_l53_53283

variable (α : ℝ)

theorem trigonometric_identity_proof :
  3 + 4 * (Real.sin (4 * α + (3 / 2) * Real.pi)) +
  Real.sin (8 * α + (5 / 2) * Real.pi) = 
  8 * (Real.sin (2 * α))^4 :=
sorry

end trigonometric_identity_proof_l53_53283


namespace area_of_region_l53_53704

theorem area_of_region : 
    ∃ (area : ℝ), 
    (∀ (x y : ℝ), (x^2 + y^2 + 6 * x - 10 * y + 5 = 0) → 
    area = 29 * Real.pi) := 
by
  use 29 * Real.pi
  intros x y h
  sorry

end area_of_region_l53_53704


namespace solve_quadratic_inequality_l53_53682

theorem solve_quadratic_inequality (x : ℝ) : (-x^2 - 2 * x + 3 < 0) ↔ (x < -3 ∨ x > 1) := 
sorry

end solve_quadratic_inequality_l53_53682


namespace largest_cube_edge_length_l53_53939

theorem largest_cube_edge_length (a : ℕ) : 
  (6 * a ^ 2 ≤ 1500) ∧
  (a * 15 ≤ 60) ∧
  (a * 15 ≤ 25) →
  a ≤ 15 :=
by
  sorry

end largest_cube_edge_length_l53_53939


namespace Jesse_remaining_money_l53_53195

-- Define the conditions
def initial_money := 50
def novel_cost := 7
def lunch_cost := 2 * novel_cost
def total_spent := novel_cost + lunch_cost

-- Define the remaining money after spending
def remaining_money := initial_money - total_spent

-- Prove that the remaining money is $29
theorem Jesse_remaining_money : remaining_money = 29 := 
by
  sorry

end Jesse_remaining_money_l53_53195


namespace gold_tetrahedron_volume_l53_53454

theorem gold_tetrahedron_volume (side_length : ℝ) (h : side_length = 8) : 
  volume_of_tetrahedron_with_gold_vertices = 170.67 := 
by 
  sorry

end gold_tetrahedron_volume_l53_53454


namespace incorrect_value_at_x5_l53_53988

theorem incorrect_value_at_x5 
  (f : ℕ → ℕ) 
  (provided_values : List ℕ) 
  (h_f : ∀ x, f x = 2 * x ^ 2 + 3 * x + 5)
  (h_provided_values : provided_values = [10, 18, 29, 44, 63, 84, 111, 140]) : 
  ¬ (f 5 = provided_values.get! 4) := 
by
  sorry

end incorrect_value_at_x5_l53_53988


namespace nth_equation_l53_53912

theorem nth_equation (n : ℕ) : (2 * n + 2) ^ 2 - (2 * n) ^ 2 = 4 * (2 * n + 1) :=
by
  sorry

end nth_equation_l53_53912


namespace sector_arc_length_120_degrees_radius_3_l53_53540

noncomputable def arc_length (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 360) * 2 * Real.pi * r

theorem sector_arc_length_120_degrees_radius_3 :
  arc_length 120 3 = 2 * Real.pi :=
by
  sorry

end sector_arc_length_120_degrees_radius_3_l53_53540


namespace flower_beds_fraction_l53_53022

noncomputable def area_triangle (leg: ℝ) : ℝ := (leg * leg) / 2
noncomputable def area_rectangle (length width: ℝ) : ℝ := length * width
noncomputable def area_trapezoid (a b height: ℝ) : ℝ := ((a + b) * height) / 2

theorem flower_beds_fraction : 
  ∀ (leg len width a b height total_length: ℝ),
    a = 30 →
    b = 40 →
    height = 6 →
    total_length = 60 →
    leg = 5 →
    len = 20 →
    width = 5 →
    (area_rectangle len width + 2 * area_triangle leg) / (area_trapezoid a b height + area_rectangle len width) = 125 / 310 :=
by
  intros
  sorry

end flower_beds_fraction_l53_53022


namespace cyclic_inequality_l53_53889

theorem cyclic_inequality
    (x1 x2 x3 x4 x5 : ℝ)
    (h1 : 0 < x1)
    (h2 : 0 < x2)
    (h3 : 0 < x3)
    (h4 : 0 < x4)
    (h5 : 0 < x5) :
    (x1 + x2 + x3 + x4 + x5)^2 > 4 * (x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x1) :=
by
  sorry

end cyclic_inequality_l53_53889


namespace geometric_sequence_product_bound_l53_53085

theorem geometric_sequence_product_bound {a1 a2 a3 m q : ℝ} (h_sum : a1 + a2 + a3 = 3 * m) (h_m_pos : 0 < m) (h_q_pos : 0 < q) (h_geom : a1 = a2 / q ∧ a3 = a2 * q) : 
  0 < a1 * a2 * a3 ∧ a1 * a2 * a3 ≤ m^3 := 
sorry

end geometric_sequence_product_bound_l53_53085


namespace contrapositive_proof_l53_53835

theorem contrapositive_proof (a b : ℝ) : 
  (a > b → a - 1 > b - 1) ↔ (a - 1 ≤ b - 1 → a ≤ b) :=
sorry

end contrapositive_proof_l53_53835


namespace find_two_digit_number_l53_53457

def digit_eq_square_of_units (n x : ℤ) : Prop :=
  10 * (x - 3) + x = n ∧ n = x * x

def units_digit_3_larger_than_tens (x : ℤ) : Prop :=
  x - 3 >= 1 ∧ x - 3 < 10 ∧ x >= 3 ∧ x < 10

theorem find_two_digit_number (n x : ℤ) (h1 : digit_eq_square_of_units n x)
  (h2 : units_digit_3_larger_than_tens x) : n = 25 ∨ n = 36 :=
by sorry

end find_two_digit_number_l53_53457


namespace rectangle_perimeter_l53_53315

theorem rectangle_perimeter (u v : ℝ) (π : ℝ) (major minor : ℝ) (area_rect area_ellipse : ℝ) 
  (inscribed : area_ellipse = 4032 * π ∧ area_rect = 4032 ∧ major = 2 * (u + v)) :
  2 * (u + v) = 128 := by
  -- Given: the area of the rectangle, the conditions of the inscribed ellipse, and the major axis constraint.
  sorry

end rectangle_perimeter_l53_53315


namespace passengers_in_each_car_l53_53347

theorem passengers_in_each_car (P : ℕ) (h1 : 20 * (P + 2) = 80) : P = 2 := 
by
  sorry

end passengers_in_each_car_l53_53347


namespace value_of_M_l53_53598

theorem value_of_M (M : ℕ) : (32^3) * (16^3) = 2^M → M = 27 :=
by
  sorry

end value_of_M_l53_53598


namespace height_of_right_triangle_l53_53219

theorem height_of_right_triangle (a b c : ℝ) (h : ℝ) (h_right : a^2 + b^2 = c^2) (h_area : h = (a * b) / c) : h = (a * b) / c := 
by
  sorry

end height_of_right_triangle_l53_53219


namespace avg_salary_increase_l53_53688

theorem avg_salary_increase (A1 : ℝ) (M : ℝ) (n : ℕ) (N : ℕ) 
  (h1 : n = 20) (h2 : A1 = 1500) (h3 : M = 4650) (h4 : N = n + 1) :
  (20 * A1 + M) / N - A1 = 150 :=
by
  -- proof goes here
  sorry

end avg_salary_increase_l53_53688


namespace ladder_length_l53_53039

variable (x y : ℝ)

theorem ladder_length :
  (x^2 = 15^2 + y^2) ∧ (x^2 = 24^2 + (y - 13)^2) → x = 25 := by
  sorry

end ladder_length_l53_53039


namespace sum_sequence_eq_l53_53886

noncomputable def S (n : ℕ) : ℝ := Real.log (1 + n) / Real.log 0.1

theorem sum_sequence_eq :
  (S 99 - S 9) = -1 := by
  sorry

end sum_sequence_eq_l53_53886


namespace terminating_decimals_count_l53_53121

noncomputable def int_counts_terminating_decimals : ℕ :=
  let n_limit := 500
  let denominator := 2100
  Nat.floor (n_limit / 21)

theorem terminating_decimals_count :
  int_counts_terminating_decimals = 23 :=
by
  /- Proof will be here eventually -/
  sorry

end terminating_decimals_count_l53_53121


namespace max_b_for_integer_solutions_l53_53973

theorem max_b_for_integer_solutions (b : ℕ) (h : ∃ x : ℤ, x^2 + b * x = -21) : b ≤ 22 :=
sorry

end max_b_for_integer_solutions_l53_53973


namespace katie_total_earnings_l53_53094

-- Define the conditions
def bead_necklaces := 4
def gem_necklaces := 3
def price_per_necklace := 3

-- The total money earned
def total_money_earned := bead_necklaces + gem_necklaces * price_per_necklace = 21

-- The statement to prove
theorem katie_total_earnings : total_money_earned :=
by
  sorry

end katie_total_earnings_l53_53094


namespace sum_of_coefficients_of_expansion_l53_53514

theorem sum_of_coefficients_of_expansion (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (2 * x - 1)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_1 + a_2 + a_3 + a_4 + a_5 = 2 :=
by
  intro h
  have h0 := h 0
  have h1 := h 1
  sorry

end sum_of_coefficients_of_expansion_l53_53514


namespace find_TU2_l53_53752

-- Define the structure of the square, distances, and points
structure square (P Q R S T U : Type) :=
(PQ : ℝ)
(PT QU QT RU TU2 : ℝ)
(h1 : PQ = 15)
(h2 : PT = 7)
(h3 : QU = 7)
(h4 : QT = 17)
(h5 : RU = 17)
(h6 : TU2 = TU^2)
(h7 : TU2 = 1073)

-- The main proof statement
theorem find_TU2 {P Q R S T U : Type} (sq : square P Q R S T U) : sq.TU2 = 1073 := by
  sorry

end find_TU2_l53_53752


namespace mila_hours_to_match_agnes_monthly_earnings_l53_53086

-- Definitions based on given conditions
def hourly_rate_mila : ℕ := 10
def hourly_rate_agnes : ℕ := 15
def weekly_hours_agnes : ℕ := 8
def weeks_in_month : ℕ := 4

-- Target statement to prove: Mila needs to work 48 hours to earn as much as Agnes in a month
theorem mila_hours_to_match_agnes_monthly_earnings :
  ∃ (h : ℕ), h = 48 ∧ (h * hourly_rate_mila) = (hourly_rate_agnes * weekly_hours_agnes * weeks_in_month) :=
by
  sorry

end mila_hours_to_match_agnes_monthly_earnings_l53_53086


namespace pow_addition_l53_53288

theorem pow_addition : (-2)^2 + 2^2 = 8 :=
by
  sorry

end pow_addition_l53_53288


namespace kim_hard_correct_l53_53556

-- Definitions
def points_per_easy := 2
def points_per_average := 3
def points_per_hard := 5
def easy_correct := 6
def average_correct := 2
def total_points := 38

-- Kim's correct answers in the hard round is 4
theorem kim_hard_correct : (total_points - (easy_correct * points_per_easy + average_correct * points_per_average)) / points_per_hard = 4 :=
by
  sorry

end kim_hard_correct_l53_53556


namespace find_f3_l53_53423

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem find_f3 
  (hf : is_odd f) 
  (hg : is_even g) 
  (h : ∀ x, f x + g x = 1 / (x - 1)) : 
  f 3 = 3 / 8 :=
by 
  sorry

end find_f3_l53_53423


namespace hardcover_volumes_l53_53526

theorem hardcover_volumes (h p : ℕ) (h1 : h + p = 10) (h2 : 25 * h + 15 * p = 220) : h = 7 :=
by sorry

end hardcover_volumes_l53_53526


namespace joan_gave_mike_seashells_l53_53774

-- Definitions based on the conditions
def original_seashells : ℕ := 79
def remaining_seashells : ℕ := 16
def given_seashells := original_seashells - remaining_seashells

-- The theorem we want to prove
theorem joan_gave_mike_seashells : given_seashells = 63 := by
  sorry

end joan_gave_mike_seashells_l53_53774


namespace simplify_fraction_l53_53309

theorem simplify_fraction (n : ℕ) : 
  (3 ^ (n + 3) - 3 * (3 ^ n)) / (3 * 3 ^ (n + 2)) = 8 / 9 :=
by sorry

end simplify_fraction_l53_53309


namespace min_area_after_fold_l53_53662

theorem min_area_after_fold (A : ℝ) (h_A : A = 1) (c : ℝ) (h_c : 0 ≤ c ∧ c ≤ 1) : 
  ∃ (m : ℝ), m = min_area ∧ m = 2 / 3 :=
by
  sorry

end min_area_after_fold_l53_53662


namespace evaluate_expression_l53_53788

def cyclical_i (z : ℂ) : Prop := z^4 = 1

theorem evaluate_expression (i : ℂ) (h : cyclical_i i) : i^15 + i^20 + i^25 + i^30 + i^35 = -i :=
by
  sorry

end evaluate_expression_l53_53788


namespace clever_calculation_part1_clever_calculation_part2_clever_calculation_part3_l53_53061

-- Prove that 46.3 * 0.56 + 5.37 * 5.6 + 1 * 0.056 equals 56.056
theorem clever_calculation_part1 : 46.3 * 0.56 + 5.37 * 5.6 + 1 * 0.056 = 56.056 :=
by
sorry

-- Prove that 101 * 92 - 92 equals 9200
theorem clever_calculation_part2 : 101 * 92 - 92 = 9200 :=
by
sorry

-- Prove that 36000 / 125 / 8 equals 36
theorem clever_calculation_part3 : 36000 / 125 / 8 = 36 :=
by
sorry

end clever_calculation_part1_clever_calculation_part2_clever_calculation_part3_l53_53061


namespace problem1_l53_53997

theorem problem1 (a b : ℝ) : (a - b)^3 + 3 * a * b * (a - b) + b^3 - a^3 = 0 :=
sorry

end problem1_l53_53997


namespace muffin_expense_l53_53919

theorem muffin_expense (B D : ℝ) 
    (h1 : D = 0.90 * B) 
    (h2 : B = D + 15) : 
    B + D = 285 := 
    sorry

end muffin_expense_l53_53919


namespace function_properties_l53_53373

noncomputable def f (x : ℝ) : ℝ := x^2

theorem function_properties :
  (∀ x1 x2 : ℝ, f (x1 * x2) = f x1 * f x2) ∧
  (∀ x : ℝ, 0 < x → deriv f x > 0) ∧
  (∀ x : ℝ, deriv f (-x) = -deriv f x) :=
by
  sorry

end function_properties_l53_53373


namespace initial_percentage_proof_l53_53150

noncomputable def initialPercentageAntifreeze (P : ℝ) : Prop :=
  let initial_fluid : ℝ := 4
  let drained_fluid : ℝ := 2.2857
  let added_antifreeze_fluid : ℝ := 2.2857 * 0.8
  let final_percentage : ℝ := 0.5
  let final_fluid : ℝ := 4
  
  let initial_antifreeze : ℝ := initial_fluid * P
  let drained_antifreeze : ℝ := drained_fluid * P
  let total_antifreeze_after_replacement : ℝ := initial_antifreeze - drained_antifreeze + added_antifreeze_fluid
  
  total_antifreeze_after_replacement = final_fluid * final_percentage

-- Prove that the initial percentage is 0.1
theorem initial_percentage_proof : initialPercentageAntifreeze 0.1 :=
by
  dsimp [initialPercentageAntifreeze]
  simp
  exact sorry

end initial_percentage_proof_l53_53150


namespace wizard_elixir_combinations_l53_53107

def roots : ℕ := 4
def minerals : ℕ := 5
def incompatible_pairs : ℕ := 3
def total_combinations : ℕ := roots * minerals
def valid_combinations : ℕ := total_combinations - incompatible_pairs

theorem wizard_elixir_combinations : valid_combinations = 17 := by
  sorry

end wizard_elixir_combinations_l53_53107


namespace quadratic_inequality_empty_solution_set_l53_53964

theorem quadratic_inequality_empty_solution_set
  (a b c : ℝ)
  (h₁ : a > 0)
  (h₂ : ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0) :
  {x : ℝ | a * x^2 + b * x + c < 0} = ∅ := 
by sorry

end quadratic_inequality_empty_solution_set_l53_53964


namespace robinson_crusoe_sees_multiple_colors_l53_53758

def chameleons_multiple_colors (r b v : ℕ) : Prop :=
  let d1 := (r - b) % 3
  let d2 := (b - v) % 3
  let d3 := (r - v) % 3
  -- Given initial counts and rules.
  (r = 155) ∧ (b = 49) ∧ (v = 96) ∧
  -- Translate specific steps and conditions into properties
  (d1 = 1 % 3) ∧ (d2 = 1 % 3) ∧ (d3 = 2 % 3)

noncomputable def will_see_multiple_colors : Prop :=
  chameleons_multiple_colors 155 49 96 →
  ∃ (r b v : ℕ), r + b + v = 300 ∧
  ((r % 3 = 0 ∧ b % 3 ≠ 0 ∧ v % 3 ≠ 0) ∨
   (r % 3 ≠ 0 ∧ b % 3 = 0 ∧ v % 3 ≠ 0) ∨
   (r % 3 ≠ 0 ∧ b % 3 ≠ 0 ∧ v % 3 = 0))

theorem robinson_crusoe_sees_multiple_colors : will_see_multiple_colors :=
sorry

end robinson_crusoe_sees_multiple_colors_l53_53758


namespace inequality_ge_9_l53_53464

theorem inequality_ge_9 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : 
  (2 / a + 1 / b) ≥ 9 :=
sorry

end inequality_ge_9_l53_53464


namespace nat_implies_int_incorrect_reasoning_due_to_minor_premise_l53_53731

-- Definitions for conditions
def is_integer (x : ℚ) : Prop := ∃ (n : ℤ), x = n
def is_natural (x : ℚ) : Prop := ∃ (n : ℕ), x = n

-- Major premise: Natural numbers are integers
theorem nat_implies_int (n : ℕ) : is_integer n := 
  ⟨n, rfl⟩

-- Minor premise: 1 / 3 is a natural number
def one_div_three_is_natural : Prop := is_natural (1 / 3)

-- Conclusion: 1 / 3 is an integer
def one_div_three_is_integer : Prop := is_integer (1 / 3)

-- The proof problem
theorem incorrect_reasoning_due_to_minor_premise :
  ¬one_div_three_is_natural :=
sorry

end nat_implies_int_incorrect_reasoning_due_to_minor_premise_l53_53731


namespace geometric_sequence_condition_l53_53732

-- Define the condition ac = b^2
def condition (a b c : ℝ) : Prop := a * c = b ^ 2

-- Define what it means for a, b, c to form a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop := 
  (b ≠ 0 → a / b = b / c) ∧ (a = 0 → b = 0 ∧ c = 0)

-- The goal is to prove the necessary but not sufficient condition
theorem geometric_sequence_condition (a b c : ℝ) :
  condition a b c ↔ (geometric_sequence a b c → condition a b c) ∧ (¬ (geometric_sequence a b c) → condition a b c ∧ ¬ (geometric_sequence (2 : ℝ) (0 : ℝ) (0 : ℝ))) :=
by
  sorry

end geometric_sequence_condition_l53_53732


namespace common_difference_range_l53_53819

theorem common_difference_range (a : ℕ → ℝ) (d : ℝ) (h : a 3 = 2) (h_pos : ∀ n, a n > 0) (h_arith : ∀ n, a (n + 1) = a n + d) : 0 ≤ d ∧ d < 1 :=
by
  sorry

end common_difference_range_l53_53819


namespace percent_calculation_l53_53157

theorem percent_calculation (Part Whole : ℝ) (h1 : Part = 120) (h2 : Whole = 80) :
  (Part / Whole) * 100 = 150 :=
by
  sorry

end percent_calculation_l53_53157


namespace find_b_l53_53409

theorem find_b (a b c : ℝ) (h1 : a = 6) (h2 : c = 3) (h3 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) : b = 15 :=
by
  rw [h1, h2] at h3
  sorry

end find_b_l53_53409


namespace avg_weekly_income_500_l53_53410

theorem avg_weekly_income_500 :
  let base_salary := 350
  let income_past_5_weeks := [406, 413, 420, 436, 495]
  let commission_next_2_weeks_avg := 315
  let total_income_past_5_weeks := income_past_5_weeks.sum
  let total_base_salary_next_2_weeks := base_salary * 2
  let total_commission_next_2_weeks := commission_next_2_weeks_avg * 2
  let total_income := total_income_past_5_weeks + total_base_salary_next_2_weeks + total_commission_next_2_weeks
  let avg_weekly_income := total_income / 7
  avg_weekly_income = 500 := by
{
  sorry
}

end avg_weekly_income_500_l53_53410


namespace retirement_savings_l53_53207

/-- Define the initial deposit amount -/
def P : ℕ := 800000

/-- Define the annual interest rate as a rational number -/
def r : ℚ := 7/100

/-- Define the number of years the money is invested for -/
def t : ℕ := 15

/-- Simple interest formula to calculate the accumulated amount -/
noncomputable def A : ℚ := P * (1 + r * t)

theorem retirement_savings :
  A = 1640000 := 
by
  sorry

end retirement_savings_l53_53207


namespace simplify_expression_l53_53692

theorem simplify_expression : 4 * (15 / 5) * (24 / -60) = - (24 / 5) := 
by
  sorry

end simplify_expression_l53_53692


namespace axis_of_symmetry_parabola_l53_53890

theorem axis_of_symmetry_parabola (a b : ℝ) (h₁ : a = -3) (h₂ : b = 6) :
  -b / (2 * a) = 1 :=
by
  sorry

end axis_of_symmetry_parabola_l53_53890


namespace tangent_line_relation_l53_53448

noncomputable def proof_problem (x1 x2 : ℝ) : Prop :=
  ((∃ (P Q : ℝ × ℝ),
    P = (x1, Real.log x1) ∧
    Q = (x2, Real.exp x2) ∧
    ∀ k : ℝ, Real.exp x2 = k ↔ k * (x2 - x1) = Real.log x1 - Real.exp x2) →
    (((x1 * Real.exp x2 = 1) ∧ ((x1 + 1) / (x1 - 1) + x2 = 0))))


theorem tangent_line_relation (x1 x2 : ℝ) (h : proof_problem x1 x2) : 
  (x1 * Real.exp x2 = 1) ∧ ((x1 + 1) / (x1 - 1) + x2 = 0) :=
sorry

end tangent_line_relation_l53_53448


namespace shorter_piece_length_l53_53554

theorem shorter_piece_length : ∃ (x : ℕ), (x + (x + 2) = 30) ∧ x = 14 :=
by {
  sorry
}

end shorter_piece_length_l53_53554


namespace line_equation_l53_53822

theorem line_equation (x y : ℝ) : 
  ((y = 1 → x = 2) ∧ ((x,y) = (1,1) ∨ (x,y) = (3,5)))
  → (2 * x - y - 3 = 0) ∨ (x = 2) :=
sorry

end line_equation_l53_53822


namespace explicit_form_of_f_l53_53319

noncomputable def f (x : ℝ) : ℝ := sorry

theorem explicit_form_of_f :
  (∀ x : ℝ, f x + f (x + 3) = 0) →
  (∀ x : ℝ, -1 < x ∧ x ≤ 1 → f x = 2 * x - 3) →
  (∀ x : ℝ, 2 < x ∧ x ≤ 4 → f x = -2 * x + 9) :=
by
  intros h1 h2
  sorry

end explicit_form_of_f_l53_53319


namespace bill_due_in_9_months_l53_53106

-- Define the conditions
def true_discount : ℝ := 240
def face_value : ℝ := 2240
def interest_rate : ℝ := 0.16

-- Define the present value calculated from the true discount and face value
def present_value := face_value - true_discount

-- Define the time in months required to match the conditions
noncomputable def time_in_months : ℝ := 12 * ((face_value / present_value - 1) / interest_rate)

-- State the theorem that the bill is due in 9 months
theorem bill_due_in_9_months : time_in_months = 9 :=
by
  sorry

end bill_due_in_9_months_l53_53106


namespace number_of_solutions_l53_53541

theorem number_of_solutions :
  ∃ (solutions : Finset (ℝ × ℝ)), 
  (∀ (x y : ℝ), (x, y) ∈ solutions ↔ (x + 2 * y = 2 ∧ abs (abs x - 2 * abs y) = 1)) ∧ 
  solutions.card = 2 :=
by
  sorry

end number_of_solutions_l53_53541


namespace existence_of_inf_polynomials_l53_53132

noncomputable def P_xy_defined (P : ℝ→ℝ) (x y z : ℝ) :=
  P x ^ 2 + P y ^ 2 + P z ^ 2 + 2 * P x * P y * P z = 1

theorem existence_of_inf_polynomials (x y z : ℝ) (P : ℕ → ℝ → ℝ) :
  (x^2 + y^2 + z^2 + 2 * x * y * z = 1) →
  (∀ n, P (n+1) = P n ∘ P n) →
  P_xy_defined (P 0) x y z →
  ∀ n, P_xy_defined (P n) x y z :=
by
  intros h1 h2 h3
  sorry

end existence_of_inf_polynomials_l53_53132


namespace Gwen_still_has_money_in_usd_l53_53063

open Real

noncomputable def exchange_rate : ℝ := 0.85
noncomputable def usd_gift : ℝ := 5.00
noncomputable def eur_gift : ℝ := 20.00
noncomputable def usd_spent_on_candy : ℝ := 3.25
noncomputable def eur_spent_on_toy : ℝ := 5.50

theorem Gwen_still_has_money_in_usd :
  let eur_conversion_to_usd := eur_gift / exchange_rate
  let total_usd_received := usd_gift + eur_conversion_to_usd
  let usd_spent_on_toy := eur_spent_on_toy / exchange_rate
  let total_usd_spent := usd_spent_on_candy + usd_spent_on_toy
  total_usd_received - total_usd_spent = 18.81 :=
by
  sorry

end Gwen_still_has_money_in_usd_l53_53063


namespace eight_machines_produce_ninety_six_bottles_in_three_minutes_l53_53482

-- Define the initial conditions
def rate_per_machine: ℕ := 16 / 4 -- bottles per minute per machine

def total_bottles_8_machines_3_minutes: ℕ := 8 * rate_per_machine * 3

-- Prove the question
theorem eight_machines_produce_ninety_six_bottles_in_three_minutes:
  total_bottles_8_machines_3_minutes = 96 :=
by
  sorry

end eight_machines_produce_ninety_six_bottles_in_three_minutes_l53_53482


namespace intersection_point_polar_coords_l53_53149

open Real

def curve_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 = 2

def curve_C2 (t x y : ℝ) : Prop :=
  (x = 2 - t) ∧ (y = t)

theorem intersection_point_polar_coords :
  ∃ (ρ θ : ℝ), (ρ = sqrt 2) ∧ (θ = π / 4) ∧
  ∃ (x y t : ℝ), curve_C2 t x y ∧ curve_C1 x y ∧
  (ρ = sqrt (x^2 + y^2)) ∧ (tan θ = y / x) :=
by
  sorry

end intersection_point_polar_coords_l53_53149


namespace find_y_l53_53748

theorem find_y (t : ℝ) (x : ℝ) (y : ℝ) (h1 : x = 3 - t) (h2 : y = 3 * t + 6) (h3 : x = -6) : y = 33 := by
  sorry

end find_y_l53_53748


namespace journey_time_l53_53401

variables (d1 d2 : ℝ) (T : ℝ)

theorem journey_time :
  (d1 / 30 + (150 - d1) / 4 = T) ∧
  (d1 / 30 + d2 / 30 + (150 - (d1 + d2)) / 4 = T) ∧
  (d2 / 4 + (150 - (d1 + d2)) / 4 = T) ∧
  (d1 = 3 / 2 * d2) 
  → T = 18 :=
by
  sorry

end journey_time_l53_53401


namespace prob1_prob2_l53_53922

-- Define the polynomial function
def polynomial (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Problem 1: Prove |b| ≤ 1, given conditions
theorem prob1 (a b c : ℝ) (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |polynomial a b c x| ≤ 1) : |b| ≤ 1 :=
sorry

-- Problem 2: Find a = 2, given conditions
theorem prob2 (a b c : ℝ) 
  (h1 : polynomial a b c 0 = -1) 
  (h2 : polynomial a b c 1 = 1) 
  (h3 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |polynomial a b c x| ≤ 1) : 
  a = 2 :=
sorry

end prob1_prob2_l53_53922


namespace four_digit_numbers_gt_3000_l53_53551

theorem four_digit_numbers_gt_3000 (d1 d2 d3 d4 : ℕ) (h_digits : (d1, d2, d3, d4) = (2, 0, 5, 5)) (h_distinct_4digit : (d1 * 1000 + d2 * 100 + d3 * 10 + d4) > 3000) :
  ∃ count, count = 3 := sorry

end four_digit_numbers_gt_3000_l53_53551


namespace vasim_share_l53_53367

theorem vasim_share (x : ℕ) (F V R : ℕ) (h1 : F = 3 * x) (h2 : V = 5 * x) (h3 : R = 11 * x) (h4 : R - F = 2400) : V = 1500 :=
by sorry

end vasim_share_l53_53367


namespace find_general_term_a_l53_53226

-- Define the sequence and conditions
noncomputable def S (n : ℕ) : ℚ :=
  if n = 0 then 0 else (n - 1) / (n * (n + 1))

-- General term to prove
def a (n : ℕ) : ℚ := 1 / (2^n) - 1 / (n * (n + 1))

theorem find_general_term_a :
  ∀ n : ℕ, n > 0 → S n + a n = (n - 1) / (n * (n + 1)) :=
by
  intro n hn
  sorry -- Proof omitted

end find_general_term_a_l53_53226


namespace parabola_symmetric_points_l53_53134

theorem parabola_symmetric_points (a : ℝ) (x1 y1 x2 y2 m : ℝ) 
  (h_parabola : ∀ x, y = a * x^2)
  (h_a_pos : a > 0)
  (h_focus_directrix : 1 / (2 * a) = 1 / 4)
  (h_symmetric : y1 = a * x1^2 ∧ y2 = a * x2^2 ∧ ∃ m, y1 = m + (x1 - m))
  (h_product : x1 * x2 = -1 / 2) :
  m = 3 / 2 := 
sorry

end parabola_symmetric_points_l53_53134


namespace sin_double_angle_tan_double_angle_l53_53641

-- Step 1: Define the first problem in Lean 4.
theorem sin_double_angle (α : ℝ) (h1 : Real.sin α = 12 / 13) (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  Real.sin (2 * α) = -120 / 169 := 
sorry

-- Step 2: Define the second problem in Lean 4.
theorem tan_double_angle (α : ℝ) (h1 : Real.tan α = 1 / 2) :
  Real.tan (2 * α) = 4 / 3 := 
sorry

end sin_double_angle_tan_double_angle_l53_53641


namespace unit_digit_seven_power_500_l53_53241

def unit_digit (x : ℕ) : ℕ := x % 10

theorem unit_digit_seven_power_500 :
  unit_digit (7 ^ 500) = 1 := 
by
  sorry

end unit_digit_seven_power_500_l53_53241


namespace find_fruit_juice_amount_l53_53179

def total_punch : ℕ := 14 * 10
def mountain_dew : ℕ := 6 * 12
def ice : ℕ := 28
def fruit_juice : ℕ := total_punch - mountain_dew - ice

theorem find_fruit_juice_amount : fruit_juice = 40 := by
  sorry

end find_fruit_juice_amount_l53_53179


namespace largest_constant_l53_53655

theorem largest_constant (x y z : ℝ) : (x^2 + y^2 + z^2 + 3 ≥ 2 * (x + y + z)) :=
by
  sorry

end largest_constant_l53_53655


namespace find_x_solution_l53_53534

theorem find_x_solution (b x : ℝ) (hb : b > 1) (hx : x > 0) 
    (h_eq : (4 * x)^(Real.log 4 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) : 
    x = 1 / 5 :=
by
  sorry

end find_x_solution_l53_53534


namespace explain_education_policy_l53_53516

theorem explain_education_policy :
  ∃ (reason1 reason2 : String), reason1 ≠ reason2 ∧
    (reason1 = "International Agreements: Favorable foreign credit terms or reciprocal educational benefits" ∧
     reason2 = "Addressing Demographic Changes: Attracting educated youth for future economic contributions")
    ∨
    (reason2 = "International Agreements: Favorable foreign credit terms or reciprocal educational benefits" ∧
     reason1 = "Addressing Demographic Changes: Attracting educated youth for future economic contributions") :=
by
  sorry

end explain_education_policy_l53_53516


namespace email_scam_check_l53_53499

-- Define the condition for receiving an email about winning a car
def received_email (info: String) : Prop :=
  info = "You received an email informing you that you have won a car. You are asked to provide your mobile phone number for contact and to transfer 150 rubles to a bank card to cover the postage fee for sending the invitation letter."

-- Define what indicates a scam
def is_scam (info: String) : Prop :=
  info = "Request for mobile number already known to the sender and an upfront payment."

-- Proving that the information in the email implies it is a scam
theorem email_scam_check (info: String) (h1: received_email info) : is_scam info :=
by
  sorry

end email_scam_check_l53_53499


namespace department_store_earnings_l53_53948

theorem department_store_earnings :
  let original_price : ℝ := 1000000
  let discount_rate : ℝ := 0.1
  let prizes := [ (5, 1000), (10, 500), (20, 200), (40, 100), (5000, 10) ]
  let A_earnings := original_price * (1 - discount_rate)
  let total_prizes := prizes.foldl (fun sum (count, amount) => sum + count * amount) 0
  let B_earnings := original_price - total_prizes
  (B_earnings - A_earnings) >= 32000 := by
  sorry

end department_store_earnings_l53_53948


namespace intersection_M_N_l53_53742

def M := {p : ℝ × ℝ | p.snd = 2 - p.fst}
def N := {p : ℝ × ℝ | p.fst - p.snd = 4}
def intersection := {p : ℝ × ℝ | p = (3, -1)}

theorem intersection_M_N : M ∩ N = intersection := 
by sorry

end intersection_M_N_l53_53742


namespace household_savings_regression_l53_53444

-- Define the problem conditions in Lean
def n := 10
def sum_x := 80
def sum_y := 20
def sum_xy := 184
def sum_x2 := 720

-- Define the averages
def x_bar := sum_x / n
def y_bar := sum_y / n

-- Define the lxx and lxy as per the solution
def lxx := sum_x2 - n * x_bar^2
def lxy := sum_xy - n * x_bar * y_bar

-- Define the regression coefficients
def b_hat := lxy / lxx
def a_hat := y_bar - b_hat * x_bar

-- State the theorem to be proved
theorem household_savings_regression :
  (∀ (x: ℝ), y = b_hat * x + a_hat) :=
by
  sorry -- skip the proof

end household_savings_regression_l53_53444


namespace percentage_correct_l53_53650

noncomputable def part : ℝ := 172.8
noncomputable def whole : ℝ := 450.0
noncomputable def percentage (part whole : ℝ) := (part / whole) * 100

theorem percentage_correct : percentage part whole = 38.4 := by
  sorry

end percentage_correct_l53_53650


namespace number_of_dogs_l53_53307

theorem number_of_dogs
    (total_animals : ℕ)
    (dogs_ratio : ℕ) (bunnies_ratio : ℕ) (birds_ratio : ℕ)
    (h_total : total_animals = 816)
    (h_ratio : dogs_ratio = 3 ∧ bunnies_ratio = 9 ∧ birds_ratio = 11) :
    (total_animals / (dogs_ratio + bunnies_ratio + birds_ratio) * dogs_ratio = 105) :=
by
    sorry

end number_of_dogs_l53_53307


namespace initial_bananas_on_tree_l53_53523

-- Definitions of given conditions
def bananas_left_on_tree : ℕ := 100
def bananas_eaten : ℕ := 70
def bananas_in_basket : ℕ := 2 * bananas_eaten

-- Statement to prove the initial number of bananas on the tree
theorem initial_bananas_on_tree : bananas_left_on_tree + (bananas_in_basket + bananas_eaten) = 310 :=
by
  sorry

end initial_bananas_on_tree_l53_53523


namespace intersection_property_l53_53582

theorem intersection_property (x_0 : ℝ) (h1 : x_0 > 0) (h2 : -x_0 = Real.tan x_0) :
  (x_0^2 + 1) * (Real.cos (2 * x_0) + 1) = 2 :=
sorry

end intersection_property_l53_53582


namespace first_guinea_pig_food_l53_53236

theorem first_guinea_pig_food (x : ℕ) (h1 : ∃ x : ℕ, R = x + 2 * x + (2 * x + 3)) (hp : 13 = x + 2 * x + (2 * x + 3)) : x = 2 :=
by
  sorry

end first_guinea_pig_food_l53_53236


namespace initial_men_employed_l53_53566

theorem initial_men_employed (M : ℕ) 
  (h1 : ∀ m d, m * d = 2 * 10)
  (h2 : ∀ m t, (m + 30) * t = 10 * 30) : 
  M = 75 :=
by
  sorry

end initial_men_employed_l53_53566


namespace geom_seq_ratio_l53_53563

variable {a_1 r : ℚ}
variable {S : ℕ → ℚ}

-- The sum of the first n terms of a geometric sequence
def geom_sum (a_1 r : ℚ) (n : ℕ) : ℚ := a_1 * (1 - r^n) / (1 - r)

-- Given conditions
axiom Sn_def : ∀ n, S n = geom_sum a_1 r n
axiom condition : S 10 / S 5 = 1 / 2

-- Theorem to prove
theorem geom_seq_ratio (h : r ≠ 1) : S 15 / S 5 = 3 / 4 :=
by
  -- proof omitted
  sorry

end geom_seq_ratio_l53_53563


namespace triangle_not_always_obtuse_l53_53147

def is_acute_triangle (A B C : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧ A < 90 ∧ B < 90 ∧ C < 90

theorem triangle_not_always_obtuse : ∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧ is_acute_triangle A B C :=
by
  -- Exact proof here.
  sorry

end triangle_not_always_obtuse_l53_53147


namespace john_total_spent_l53_53190

-- Defining the conditions from part a)
def vacuum_cleaner_original_price : ℝ := 250
def vacuum_cleaner_discount_rate : ℝ := 0.20
def dishwasher_price : ℝ := 450
def special_offer_discount : ℝ := 75
def sales_tax_rate : ℝ := 0.07

-- The adesso to formalize part c noncomputably.
noncomputable def total_amount_spent : ℝ :=
  let vacuum_cleaner_discount := vacuum_cleaner_original_price * vacuum_cleaner_discount_rate
  let vacuum_cleaner_final_price := vacuum_cleaner_original_price - vacuum_cleaner_discount
  let total_before_special_offer := vacuum_cleaner_final_price + dishwasher_price
  let total_after_special_offer := total_before_special_offer - special_offer_discount
  let sales_tax := total_after_special_offer * sales_tax_rate
  total_after_special_offer + sales_tax

-- The proof statement
theorem john_total_spent : total_amount_spent = 615.25 := by
  sorry

end john_total_spent_l53_53190


namespace inv_sum_mod_l53_53351

theorem inv_sum_mod 
  : (∃ (x y : ℤ), (3 * x ≡ 1 [ZMOD 25]) ∧ (3^2 * y ≡ 1 [ZMOD 25]) ∧ (x + y ≡ 6 [ZMOD 25])) :=
sorry

end inv_sum_mod_l53_53351


namespace solve_system_eqns_l53_53904

theorem solve_system_eqns (x y z a : ℝ)
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = a^2)
  (h3 : x^3 + y^3 + z^3 = a^3) :
  (x = a ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = a ∧ z = 0) ∨
  (x = 0 ∧ y = 0 ∧ z = a) := 
by
  sorry

end solve_system_eqns_l53_53904


namespace smallest_int_a_for_inequality_l53_53440

theorem smallest_int_a_for_inequality (a : ℤ) : 
  (∀ x : ℝ, (0 < x ∧ x < Real.pi / 2) → 
  Real.exp x - x * Real.cos x + Real.cos x * Real.log (Real.cos x) + a * x^2 ≥ 1) → 
  a = 1 := 
sorry

end smallest_int_a_for_inequality_l53_53440


namespace robin_earns_30_percent_more_than_erica_l53_53603

variable (E R C : ℝ)

theorem robin_earns_30_percent_more_than_erica
  (h1 : C = 1.60 * E)
  (h2 : C = 1.23076923076923077 * R) :
  R = 1.30 * E :=
by
  sorry

end robin_earns_30_percent_more_than_erica_l53_53603


namespace max_x_minus_2y_l53_53604

open Real

theorem max_x_minus_2y (x y : ℝ) (h : (x^2) / 16 + (y^2) / 9 = 1) : 
  ∃ t : ℝ, t = 2 * sqrt 13 ∧ x - 2 * y = t := 
sorry

end max_x_minus_2y_l53_53604


namespace infinitely_many_good_pairs_l53_53986

def is_triangular (t : ℕ) : Prop :=
  ∃ n : ℕ, t = n * (n + 1) / 2

theorem infinitely_many_good_pairs :
  ∃ (a b : ℕ), (0 < a) ∧ (0 < b) ∧ 
  ∀ t : ℕ, is_triangular t ↔ is_triangular (a * t + b) :=
sorry

end infinitely_many_good_pairs_l53_53986


namespace g_neg_3_eq_neg_9_l53_53979

-- Define even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Given functions and values
variables (f g : ℝ → ℝ) (h_even : is_even_function f) (h_f_g : ∀ x, f x = g x - 2 * x)
variables (h_g3 : g 3 = 3)

-- Goal: Prove that g (-3) = -9
theorem g_neg_3_eq_neg_9 : g (-3) = -9 :=
sorry

end g_neg_3_eq_neg_9_l53_53979


namespace fraction_of_students_with_buddy_l53_53013

theorem fraction_of_students_with_buddy (t s : ℕ) (h1 : (t / 4) = (3 * s / 5)) :
  (t / 4 + 3 * s / 5) / (t + s) = 6 / 17 :=
by
  sorry

end fraction_of_students_with_buddy_l53_53013


namespace product_of_fractions_l53_53034

-- Define the fractions
def one_fourth : ℚ := 1 / 4
def one_half : ℚ := 1 / 2
def one_eighth : ℚ := 1 / 8

-- State the theorem we are proving
theorem product_of_fractions :
  one_fourth * one_half = one_eighth :=
by
  sorry

end product_of_fractions_l53_53034


namespace log_property_l53_53611

theorem log_property (x : ℝ) (h₁ : Real.log x > 0) (h₂ : x > 1) : x > Real.exp 1 := by 
  sorry

end log_property_l53_53611


namespace triangle_area_l53_53114

theorem triangle_area (a b c : ℕ) (h : a = 12) (i : b = 16) (j : c = 20) (hc : c * c = a * a + b * b) :
  ∃ (area : ℕ), area = 96 :=
by
  sorry

end triangle_area_l53_53114


namespace value_of_x_l53_53706

theorem value_of_x (x : ℝ) (h : 3 * x + 15 = (1/3) * (7 * x + 45)) : x = 0 :=
by
  sorry

end value_of_x_l53_53706


namespace truthful_dwarfs_count_l53_53156

def number_of_dwarfs := 10
def vanilla_ice_cream := number_of_dwarfs
def chocolate_ice_cream := number_of_dwarfs / 2
def fruit_ice_cream := 1

theorem truthful_dwarfs_count (T L : ℕ) (h1 : T + L = 10)
  (h2 : vanilla_ice_cream = T + (L * 2))
  (h3 : chocolate_ice_cream = T / 2 + (L / 2 * 2))
  (h4 : fruit_ice_cream = 1)
  : T = 4 :=
sorry

end truthful_dwarfs_count_l53_53156


namespace seven_in_M_l53_53590

-- Define the universal set U
def U : Set ℕ := {1, 3, 5, 7, 9}

-- Define the set M complement with respect to U
def compl_U_M : Set ℕ := {1, 3, 5}

-- Define the set M
def M : Set ℕ := U \ compl_U_M

-- Prove that 7 is an element of M
theorem seven_in_M : 7 ∈ M :=
by {
  sorry
}

end seven_in_M_l53_53590


namespace rate_of_fuel_consumption_l53_53546

-- Define the necessary conditions
def total_fuel : ℝ := 100
def total_hours : ℝ := 175

-- Prove the rate of fuel consumption per hour
theorem rate_of_fuel_consumption : (total_fuel / total_hours) = 100 / 175 := 
by 
  sorry

end rate_of_fuel_consumption_l53_53546


namespace general_term_arithmetic_sequence_l53_53634

variable {α : Type*}
variables (a_n a : ℕ → ℕ) (d a_1 a_2 a_3 a_4 n : ℕ)

-- Define the arithmetic sequence condition
def arithmetic_sequence (a_n : ℕ → ℕ) (d : ℕ) :=
  ∀ n, a_n (n + 1) = a_n n + d

-- Define the inequality solution condition 
def inequality_solution_set (a_1 a_2 : ℕ) (x : ℕ) :=
  a_1 ≤ x ∧ x ≤ a_2

theorem general_term_arithmetic_sequence :
  arithmetic_sequence a_n d ∧ (d ≠ 0) ∧ 
  (∀ x, x^2 - a_3 * x + a_4 ≤ 0 ↔ inequality_solution_set a_1 a_2 x) →
  a_n = 2 * n :=
by
  sorry

end general_term_arithmetic_sequence_l53_53634


namespace jim_catches_bob_in_20_minutes_l53_53251

theorem jim_catches_bob_in_20_minutes
  (bob_speed : ℝ)
  (jim_speed : ℝ)
  (bob_head_start : ℝ)
  (bob_speed_mph : bob_speed = 6)
  (jim_speed_mph : jim_speed = 9)
  (bob_headstart_miles : bob_head_start = 1) :
  ∃ (m : ℝ), m = 20 := 
by
  sorry

end jim_catches_bob_in_20_minutes_l53_53251


namespace no_integer_coeff_trinomials_with_integer_roots_l53_53959

theorem no_integer_coeff_trinomials_with_integer_roots :
  ¬ ∃ (a b c : ℤ),
    (∀ x : ℤ, a * x^2 + b * x + c = 0 → (∃ x1 x2 : ℤ, a = 0 ∧ x = x1 ∨ a ≠ 0 ∧ x = x1 ∨ x = x2 ∨ x = x1 ∧ x = x2)) ∧
    (∀ x : ℤ, (a + 1) * x^2 + (b + 1) * x + (c + 1) = 0 → (∃ x1 x2 : ℤ, (a + 1) = 0 ∧ x = x1 ∨ (a + 1) ≠ 0 ∧ x = x1 ∨ x = x2 ∨ x = x1 ∧ x = x2)) :=
by
  sorry

end no_integer_coeff_trinomials_with_integer_roots_l53_53959


namespace amy_total_equals_bob_total_l53_53378

def original_price : ℝ := 120.00
def sales_tax_rate : ℝ := 0.08
def discount_rate : ℝ := 0.25
def additional_discount : ℝ := 0.10
def num_sweaters : ℕ := 4

def calculate_amy_total (original_price : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) (additional_discount : ℝ) (num_sweaters : ℕ) : ℝ :=
  let price_with_tax := original_price * (1.0 + sales_tax_rate)
  let discounted_price := price_with_tax * (1.0 - discount_rate)
  let final_price := discounted_price * (1.0 - additional_discount)
  final_price * (num_sweaters : ℝ)
  
def calculate_bob_total (original_price : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) (additional_discount : ℝ) (num_sweaters : ℕ) : ℝ :=
  let discounted_price := original_price * (1.0 - discount_rate)
  let further_discounted_price := discounted_price * (1.0 - additional_discount)
  let price_with_tax := further_discounted_price * (1.0 + sales_tax_rate)
  price_with_tax * (num_sweaters : ℝ)

theorem amy_total_equals_bob_total :
  calculate_amy_total original_price sales_tax_rate discount_rate additional_discount num_sweaters =
  calculate_bob_total original_price sales_tax_rate discount_rate additional_discount num_sweaters :=
by
  sorry

end amy_total_equals_bob_total_l53_53378


namespace even_function_solution_l53_53082

theorem even_function_solution :
  ∀ (m : ℝ), (∀ x : ℝ, (m+1) * x^2 + (m-2) * x = (m+1) * x^2 - (m-2) * x) → (m = 2 ∧ ∀ x : ℝ, (2+1) * x^2 + (2-2) * x = 3 * x^2) :=
by
  sorry

end even_function_solution_l53_53082


namespace intersection_of_M_and_N_is_0_and_2_l53_53687

open Set

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_of_M_and_N_is_0_and_2 : M ∩ N = {0, 2} :=
by
  sorry

end intersection_of_M_and_N_is_0_and_2_l53_53687


namespace ratio_of_perimeters_l53_53779

theorem ratio_of_perimeters (s : ℝ) (hs : s > 0) :
  let small_triangle_perimeter := s + (s / 2) + (s / 2)
  let large_rectangle_perimeter := 2 * (s + (s / 2))
  small_triangle_perimeter / large_rectangle_perimeter = 2 / 3 :=
by
  sorry

end ratio_of_perimeters_l53_53779


namespace total_surface_area_hemisphere_l53_53621

theorem total_surface_area_hemisphere (A : ℝ) (r : ℝ) : (A = 100 * π) → (r = 10) → (2 * π * r^2 + A = 300 * π) :=
by
  intro hA hr
  sorry

end total_surface_area_hemisphere_l53_53621


namespace quadratic_roots_solution_l53_53000

theorem quadratic_roots_solution (x : ℝ) (h : x > 0) (h_roots : 7 * x^2 - 8 * x - 6 = 0) : (x = 6 / 7) ∨ (x = 1) :=
sorry

end quadratic_roots_solution_l53_53000


namespace speed_of_man_l53_53370

theorem speed_of_man (v_m v_s : ℝ) 
    (h1 : (v_m + v_s) * 4 = 32) 
    (h2 : (v_m - v_s) * 4 = 24) : v_m = 7 := 
by
  sorry

end speed_of_man_l53_53370


namespace function_nonnegative_l53_53460

noncomputable def f (x : ℝ) := (x - 10*x^2 + 35*x^3) / (9 - x^3)

theorem function_nonnegative (x : ℝ) : 
  (f x ≥ 0) ↔ (0 ≤ x ∧ x ≤ (1 / 7)) ∨ (3 ≤ x) :=
sorry

end function_nonnegative_l53_53460


namespace most_economical_speed_and_cost_l53_53966

open Real

theorem most_economical_speed_and_cost :
  ∀ (x : ℝ),
  (120:ℝ) / x * 36 + (120:ℝ) / x * 6 * (4 + x^2 / 360) = ((7200:ℝ) / x) + 2 * x → 
  50 ≤ x ∧ x ≤ 100 → 
  (∀ v : ℝ, (50 ≤ v ∧ v ≤ 100) → 
  (120 / v * 36 + 120 / v * 6 * (4 + v^2 / 360) ≤ 120 / x * 36 + 120 / x * 6 * (4 + x^2 / 360)) ) → 
  x = 60 → 
  (120 / x * 36 + 120 / x * 6 * (4 + x^2 / 360) = 240) :=
by
  intros x hx bounds min_cost opt_speed
  sorry

end most_economical_speed_and_cost_l53_53966


namespace total_population_l53_53326

-- Defining the populations of Springfield and the difference in population
def springfield_population : ℕ := 482653
def population_difference : ℕ := 119666

-- The definition of Greenville's population in terms of Springfield's population
def greenville_population : ℕ := springfield_population - population_difference

-- The statement that we want to prove: the total population of Springfield and Greenville
theorem total_population :
  springfield_population + greenville_population = 845640 := by
  sorry

end total_population_l53_53326


namespace percentage_increase_l53_53950

variable (presentIncome : ℝ) (newIncome : ℝ)

theorem percentage_increase (h1 : presentIncome = 12000) (h2 : newIncome = 12240) :
  ((newIncome - presentIncome) / presentIncome) * 100 = 2 := by
  sorry

end percentage_increase_l53_53950


namespace correct_calculation_l53_53703

theorem correct_calculation (a b : ℝ) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by
  sorry

end correct_calculation_l53_53703


namespace domain_of_function_l53_53925

def valid_domain (x : ℝ) : Prop :=
  (2 - x ≥ 0) ∧ (x > 0) ∧ (x ≠ 2)

theorem domain_of_function :
  {x : ℝ | ∃ (y : ℝ), y = x ∧ valid_domain x} = {x | 0 < x ∧ x < 2} :=
by
  sorry

end domain_of_function_l53_53925


namespace circumscribed_circle_area_l53_53369

theorem circumscribed_circle_area (side_length : ℝ) (h : side_length = 12) :
  ∃ (A : ℝ), A = 48 * π :=
by
  sorry

end circumscribed_circle_area_l53_53369


namespace students_prefer_windows_to_mac_l53_53705

-- Define the conditions
def total_students : ℕ := 210
def students_prefer_mac : ℕ := 60
def students_equally_prefer_both : ℕ := 20
def students_no_preference : ℕ := 90

-- The proof problem
theorem students_prefer_windows_to_mac :
  total_students - students_prefer_mac - students_equally_prefer_both - students_no_preference = 40 :=
by sorry

end students_prefer_windows_to_mac_l53_53705


namespace find_a_2b_3c_value_l53_53627

-- Problem statement and conditions
theorem find_a_2b_3c_value (a b c : ℝ)
  (h : ∀ x : ℝ, (x < -1 ∨ abs (x - 10) ≤ 2) ↔ (x - a) * (x - b) / (x - c) ≤ 0)
  (h_ab : a < b) : a + 2 * b + 3 * c = 29 := 
sorry

end find_a_2b_3c_value_l53_53627


namespace ratio_of_money_spent_l53_53145

theorem ratio_of_money_spent (h : ∀(a b c : ℕ), a + b + c = 75) : 
  (25 / 75 = 1 / 3) ∧ 
  (40 / 75 = 4 / 3) ∧ 
  (10 / 75 = 2 / 15) :=
by
  sorry

end ratio_of_money_spent_l53_53145


namespace perpendicular_bisector_l53_53877

theorem perpendicular_bisector (x y : ℝ) (h : -1 ≤ x ∧ x ≤ 3) (h_line : x - 2 * y + 1 = 0) : 
  2 * x - y - 1 = 0 :=
sorry

end perpendicular_bisector_l53_53877


namespace arithmetic_seq_proof_l53_53744

noncomputable def arithmetic_sequence : Type := ℕ → ℝ

variables (a : ℕ → ℝ) (d : ℝ)

def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

variables (a₁ a₂ a₃ a₄ : ℝ)
variables (h1 : a 1 + a 2 = 10)
variables (h2 : a 4 = a 3 + 2)
variables (h3 : is_arithmetic_seq a d)

theorem arithmetic_seq_proof :
  a 3 + a 4 = 18 :=
sorry

end arithmetic_seq_proof_l53_53744


namespace change_in_total_berries_l53_53400

theorem change_in_total_berries (B S : ℕ) (hB : B = 20) (hS : S + B = 50) : (S - B) = 10 := by
  sorry

end change_in_total_berries_l53_53400


namespace solution_set_inequality_l53_53261

theorem solution_set_inequality (x : ℝ) : (x + 1) * (2 - x) < 0 ↔ x > 2 ∨ x < -1 :=
sorry

end solution_set_inequality_l53_53261


namespace points_on_quadratic_l53_53978

theorem points_on_quadratic (c y₁ y₂ : ℝ) 
  (hA : y₁ = (-1)^2 - 6*(-1) + c) 
  (hB : y₂ = 2^2 - 6*2 + c) : y₁ > y₂ := 
  sorry

end points_on_quadratic_l53_53978


namespace largest_of_three_consecutive_integers_sum_90_is_31_l53_53920

theorem largest_of_three_consecutive_integers_sum_90_is_31 :
  ∃ (a b c : ℤ), (a + b + c = 90) ∧ (b = a + 1) ∧ (c = b + 1) ∧ (c = 31) :=
by
  sorry

end largest_of_three_consecutive_integers_sum_90_is_31_l53_53920


namespace parking_space_area_l53_53951

theorem parking_space_area (L W : ℕ) (h1 : L = 9) (h2 : 2 * W + L = 37) : L * W = 126 :=
by
  -- Proof omitted.
  sorry

end parking_space_area_l53_53951


namespace expression_value_l53_53215

noncomputable def givenExpression : ℝ :=
  -2^2 + Real.sqrt 8 - 3 + 1/3

theorem expression_value : givenExpression = -20/3 + 2 * Real.sqrt 2 := 
by
  sorry

end expression_value_l53_53215


namespace min_le_one_fourth_sum_max_ge_four_ninths_sum_l53_53037

variable (a b c : ℝ)

theorem min_le_one_fourth_sum
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_roots : b^2 - 4 * a * c ≥ 0) :
  min a (min b c) ≤ 1 / 4 * (a + b + c) :=
sorry

theorem max_ge_four_ninths_sum
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_roots : b^2 - 4 * a * c ≥ 0) :
  max a (max b c) ≥ 4 / 9 * (a + b + c) :=
sorry

end min_le_one_fourth_sum_max_ge_four_ninths_sum_l53_53037


namespace abs_has_min_at_zero_l53_53035

def f (x : ℝ) : ℝ := abs x

theorem abs_has_min_at_zero : ∃ m, (∀ x : ℝ, f x ≥ m) ∧ f 0 = m := by
  sorry

end abs_has_min_at_zero_l53_53035


namespace shortest_is_Bob_l53_53649

variable {Person : Type}
variable [LinearOrder Person]

variable (Amy Bob Carla Dan Eric : Person)

-- Conditions
variable (h1 : Amy > Carla)
variable (h2 : Dan < Eric)
variable (h3 : Dan > Bob)
variable (h4 : Eric < Carla)

theorem shortest_is_Bob : ∀ p : Person, p = Bob :=
by
  intro p
  sorry

end shortest_is_Bob_l53_53649


namespace mowers_mow_l53_53494

theorem mowers_mow (mowers hectares days mowers_new days_new : ℕ)
  (h1 : 3 * 3 * days = 3 * hectares)
  (h2 : 5 * days_new = 5 * (days_new * hectares / days)) :
  5 * days_new * (hectares / (3 * days)) = 25 / 3 :=
sorry

end mowers_mow_l53_53494


namespace intersection_A_B_l53_53735

def setA (x : ℝ) : Prop := x^2 - 2 * x > 0
def setB (x : ℝ) : Prop := abs (x + 1) < 2

theorem intersection_A_B :
  {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -3 < x ∧ x < 0} :=
by
  sorry

end intersection_A_B_l53_53735


namespace fraction_d_can_be_zero_l53_53117

theorem fraction_d_can_be_zero :
  ∃ x : ℝ, (x + 1) / (x - 1) = 0 :=
by {
  sorry
}

end fraction_d_can_be_zero_l53_53117


namespace walking_time_l53_53032

theorem walking_time (v : ℕ) (d : ℕ) (h1 : v = 10) (h2 : d = 4) : 
    ∃ (T : ℕ), T = 24 := 
by
  sorry

end walking_time_l53_53032


namespace unique_zero_of_f_l53_53092

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2*x + a * (Real.exp (x - 1) + Real.exp (-x + 1))

theorem unique_zero_of_f (a : ℝ) : (∃! x, f x a = 0) ↔ a = 1 / 2 := sorry

end unique_zero_of_f_l53_53092


namespace parallelepiped_diagonal_relationship_l53_53764

theorem parallelepiped_diagonal_relationship {a b c d e f g : ℝ} 
  (h1 : c = d) 
  (h2 : e = e) 
  (h3 : f = f) 
  (h4 : g = g) 
  : a^2 + b^2 + c^2 + g^2 = d^2 + e^2 + f^2 :=
by
  sorry

end parallelepiped_diagonal_relationship_l53_53764


namespace quadratic_coeffs_l53_53200

theorem quadratic_coeffs (x : ℝ) :
  (x - 1)^2 = 3 * x - 2 → ∃ b c, (x^2 + b * x + c = 0 ∧ b = -5 ∧ c = 3) :=
by
  sorry

end quadratic_coeffs_l53_53200


namespace find_non_integer_solution_l53_53232

noncomputable def q (x y : ℝ) (b : Fin 10 → ℝ) : ℝ :=
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 +
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3

theorem find_non_integer_solution (b : Fin 10 → ℝ)
  (h0 : q 0 0 b = 0)
  (h1 : q 1 0 b = 0)
  (h2 : q (-1) 0 b = 0)
  (h3 : q 0 1 b = 0)
  (h4 : q 0 (-1) b = 0)
  (h5 : q 1 1 b = 0)
  (h6 : q 1 (-1) b = 0)
  (h7 : q (-1) 1 b = 0)
  (h8 : q (-1) (-1) b = 0) :
  ∃ r s : ℝ, q r s b = 0 ∧ ¬ (∃ n : ℤ, r = n) ∧ ¬ (∃ n : ℤ, s = n) :=
sorry

end find_non_integer_solution_l53_53232


namespace polynomial_not_factorizable_l53_53152

theorem polynomial_not_factorizable
  (n m : ℕ)
  (hnm : n > m)
  (hm1 : m > 1)
  (hn_odd : n % 2 = 1)
  (hm_odd : m % 2 = 1) :
  ¬ ∃ (g h : Polynomial ℤ), g.degree > 0 ∧ h.degree > 0 ∧ (x^n + x^m + x + 1 = g * h) :=
by
  sorry

end polynomial_not_factorizable_l53_53152


namespace taylor_class_more_girls_l53_53446

theorem taylor_class_more_girls (b g : ℕ) (total : b + g = 42) (ratio : b / g = 3 / 4) : g - b = 6 := by
  sorry

end taylor_class_more_girls_l53_53446


namespace intersection_is_correct_l53_53470

noncomputable def setA : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
noncomputable def setB : Set ℝ := {x | Real.log x / Real.log 2 ≤ 2}

theorem intersection_is_correct : setA ∩ setB = {x : ℝ | 0 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_is_correct_l53_53470


namespace find_x_l53_53515

def F (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) : ℕ := a^b + c * d

theorem find_x (x : ℕ) : F 3 x 5 9 = 500 → x = 6 := 
by 
  sorry

end find_x_l53_53515


namespace polynomial_integer_roots_k_zero_l53_53629

theorem polynomial_integer_roots_k_zero :
  (∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∀ x : ℤ, (x - a) * (x - b) * (x - c) = x^3 - x + 0) ∨
  (∀ x : ℤ, (x - a) * (x - b) * (x - c) = x^3 - x + k)) →
  k = 0 :=
sorry

end polynomial_integer_roots_k_zero_l53_53629


namespace circle_second_x_intercept_l53_53786

theorem circle_second_x_intercept :
  ∀ (circle : ℝ × ℝ → Prop), (∀ (x y : ℝ), circle (x, y) ↔ (x - 5) ^ 2 + y ^ 2 = 25) →
    ∃ x : ℝ, (x ≠ 0 ∧ circle (x, 0) ∧ x = 10) :=
by {
  sorry
}

end circle_second_x_intercept_l53_53786


namespace proof_ac_plus_bd_l53_53192

theorem proof_ac_plus_bd (a b c d : ℝ)
  (h1 : a + b + c = 10)
  (h2 : a + b + d = -6)
  (h3 : a + c + d = 0)
  (h4 : b + c + d = 15) :
  ac + bd = -130.111 := 
by
  sorry

end proof_ac_plus_bd_l53_53192


namespace Vins_total_miles_l53_53782

theorem Vins_total_miles : 
  let dist_library_one_way := 6
  let dist_school_one_way := 5
  let dist_friend_one_way := 8
  let extra_miles := 1
  let shortcut_miles := 2
  let days_per_week := 7
  let weeks := 4

  -- Calculate weekly miles
  let library_round_trip := (dist_library_one_way + dist_library_one_way + extra_miles)
  let total_library_weekly := library_round_trip * 3

  let school_round_trip := (dist_school_one_way + dist_school_one_way + extra_miles)
  let total_school_weekly := school_round_trip * 2

  let friend_round_trip := dist_friend_one_way + (dist_friend_one_way - shortcut_miles)
  let total_friend_weekly := friend_round_trip / 2 -- Every two weeks

  let total_weekly := total_library_weekly + total_school_weekly + total_friend_weekly

  -- Calculate total miles over the weeks
  let total_miles := total_weekly * weeks

  total_miles = 272 := sorry

end Vins_total_miles_l53_53782


namespace james_new_friends_l53_53769

-- Definitions and assumptions based on the conditions provided
def initial_friends := 20
def lost_friends := 2
def friends_after_loss : ℕ := initial_friends - lost_friends
def friends_upon_arrival := 19

-- Definition of new friends made
def new_friends : ℕ := friends_upon_arrival - friends_after_loss

-- Statement to prove
theorem james_new_friends :
  new_friends = 1 :=
by
  -- Solution proof would be inserted here
  sorry

end james_new_friends_l53_53769


namespace seating_arrangement_l53_53967

theorem seating_arrangement (x y : ℕ) (h : x + y ≤ 8) (h1 : 9 * x + 6 * y = 57) : x = 5 := 
by
  sorry

end seating_arrangement_l53_53967


namespace part1_part2_l53_53725

theorem part1 : (π - 3)^0 + (-1)^(2023) - Real.sqrt 8 = -2 * Real.sqrt 2 := sorry

theorem part2 (x : ℝ) : (4 * x - 3 > 9) ∧ (2 + x ≥ 0) ↔ x > 3 := sorry

end part1_part2_l53_53725


namespace second_hand_travel_distance_l53_53625

theorem second_hand_travel_distance (radius : ℝ) (time_minutes : ℕ) (C : ℝ) (distance : ℝ) 
    (h1 : radius = 8) (h2 : time_minutes = 45) 
    (h3 : C = 2 * Real.pi * radius) 
    (h4 : distance = time_minutes * C)
    : distance = 720 * Real.pi := 
by 
  rw [h1, h2, h3] at *
  sorry

end second_hand_travel_distance_l53_53625


namespace train_truck_load_l53_53637

variables (x y : ℕ)

def transport_equations (x y : ℕ) : Prop :=
  (2 * x + 5 * y = 120) ∧ (8 * x + 10 * y = 440)

def tonnage (x y : ℕ) : ℕ :=
  5 * x + 8 * y

theorem train_truck_load
  (x y : ℕ)
  (h : transport_equations x y) :
  tonnage x y = 282 :=
sorry

end train_truck_load_l53_53637


namespace valid_sequences_l53_53949

-- Define the transformation function for a ten-digit number
noncomputable def transform (n : ℕ) : ℕ := sorry

-- Given sequences
def seq1 := 1101111111
def seq2 := 1201201020
def seq3 := 1021021020
def seq4 := 0112102011

-- The proof problem statement
theorem valid_sequences :
  (transform 1101111111 = seq1) ∧
  (transform 1021021020 = seq3) ∧
  (transform 0112102011 = seq4) :=
sorry

end valid_sequences_l53_53949


namespace stone_radius_l53_53230

theorem stone_radius (hole_diameter hole_depth : ℝ) (r : ℝ) :
  hole_diameter = 30 → hole_depth = 10 → (r - 10)^2 + 15^2 = r^2 → r = 16.25 :=
by
  intros h_diam h_depth hyp_eq
  sorry

end stone_radius_l53_53230


namespace original_average_age_l53_53314

theorem original_average_age (N : ℕ) (A : ℝ) (h1 : A = 50) (h2 : 12 * 32 + N * 50 = (N + 12) * (A - 4)) : A = 50 := by
  sorry 

end original_average_age_l53_53314


namespace total_animals_for_sale_l53_53280

theorem total_animals_for_sale (dogs cats birds fish : ℕ) 
  (h1 : dogs = 6)
  (h2 : cats = dogs / 2)
  (h3 : birds = dogs * 2)
  (h4 : fish = dogs * 3) :
  dogs + cats + birds + fish = 39 := 
by
  sorry

end total_animals_for_sale_l53_53280


namespace completing_the_square_l53_53618

theorem completing_the_square (x : ℝ) : x^2 + 8 * x + 9 = 0 → (x + 4)^2 = 7 :=
by 
  intro h
  sorry

end completing_the_square_l53_53618


namespace liquid_levels_proof_l53_53854

noncomputable def liquid_levels (H : ℝ) : ℝ × ℝ :=
  let ρ_water := 1000
  let ρ_gasoline := 600
  -- x = level drop in the left vessel
  let x := (3 / 14) * H
  let h_left := 0.9 * H - x
  let h_right := H
  (h_left, h_right)

theorem liquid_levels_proof (H : ℝ) (h : ℝ) :
  H > 0 →
  h = 0.9 * H →
  liquid_levels H = (0.69 * H, H) :=
by
  intros
  sorry

end liquid_levels_proof_l53_53854


namespace function_g_l53_53159

theorem function_g (g : ℝ → ℝ) (t : ℝ) :
  (∀ t, (20 * t - 14) = 2 * (g t) - 40) → (g t = 10 * t + 13) :=
by
  intro h
  have h1 : 20 * t - 14 = 2 * (g t) - 40 := h t
  sorry

end function_g_l53_53159


namespace series_sum_eq_four_ninths_l53_53426

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l53_53426


namespace length_of_string_for_circle_l53_53498

theorem length_of_string_for_circle (A : ℝ) (pi_approx : ℝ) (extra_length : ℝ) (hA : A = 616) (hpi : pi_approx = 22 / 7) (hextra : extra_length = 5) :
  ∃ (length : ℝ), length = 93 :=
by {
  sorry
}

end length_of_string_for_circle_l53_53498


namespace p_necessary_not_sufficient_q_l53_53030

theorem p_necessary_not_sufficient_q (x : ℝ) : (|x| = 2) → (x = 2) → (|x| = 2 ∧ (x ≠ 2 ∨ x = -2)) := by
  intros h_p h_q
  sorry

end p_necessary_not_sufficient_q_l53_53030


namespace find_f_inv_8_l53_53204

variable (f : ℝ → ℝ)

-- Given conditions
axiom h1 : f 5 = 1
axiom h2 : ∀ x, f (2 * x) = 2 * f x

-- Theorem to prove
theorem find_f_inv_8 : f ⁻¹' {8} = {40} :=
by sorry

end find_f_inv_8_l53_53204


namespace watch_cost_l53_53376

theorem watch_cost (number_of_dimes : ℕ) (value_of_dime : ℝ) (h : number_of_dimes = 50) (hv : value_of_dime = 0.10) :
  number_of_dimes * value_of_dime = 5.00 :=
by
  sorry

end watch_cost_l53_53376


namespace water_tank_capacity_l53_53576

theorem water_tank_capacity (C : ℝ) :
  (0.40 * C - 0.25 * C = 36) → C = 240 :=
  sorry

end water_tank_capacity_l53_53576


namespace sum_first_9000_terms_l53_53972

noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ :=
a * ((1 - r^n) / (1 - r))

theorem sum_first_9000_terms (a r : ℝ) (h1 : geom_sum a r 3000 = 1000) 
                              (h2 : geom_sum a r 6000 = 1900) : 
                              geom_sum a r 9000 = 2710 := 
by sorry

end sum_first_9000_terms_l53_53972


namespace division_of_floats_l53_53198

theorem division_of_floats : 4.036 / 0.04 = 100.9 :=
by
  sorry

end division_of_floats_l53_53198


namespace intersection_M_N_l53_53327

def M (x : ℝ) : Prop := Real.log x / Real.log 2 ≥ 0
def N (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x | N x} = {x | 1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l53_53327


namespace smallest_piece_length_l53_53334

theorem smallest_piece_length (x : ℕ) :
  (9 - x) + (14 - x) ≤ (16 - x) → x ≥ 7 :=
by
  sorry

end smallest_piece_length_l53_53334


namespace max_good_diagonals_l53_53467

def is_good_diagonal (n : ℕ) (d : ℕ) : Prop := ∀ (P : Fin n → Prop), ∃! (i j : Fin n), P i ∧ P j ∧ (d = i + j)

theorem max_good_diagonals (n : ℕ) (h : 2 ≤ n) :
  (∃ (m : ℕ), is_good_diagonal n m ∧ (m = n - 2 ↔ Even n) ∧ (m = n - 3 ↔ Odd n)) :=
by
  sorry

end max_good_diagonals_l53_53467


namespace find_x_l53_53864

theorem find_x (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y + x * y = 152) : x = 16 := 
by
  sorry

end find_x_l53_53864


namespace calculate_expression_l53_53804

theorem calculate_expression : 14 - (-12) + (-25) - 17 = -16 := by
  -- definitions from conditions are understood and used here implicitly
  sorry

end calculate_expression_l53_53804


namespace cookie_cost_l53_53151

theorem cookie_cost 
    (initial_amount : ℝ := 100)
    (latte_cost : ℝ := 3.75)
    (croissant_cost : ℝ := 3.50)
    (days : ℕ := 7)
    (num_cookies : ℕ := 5)
    (remaining_amount : ℝ := 43) :
    (initial_amount - remaining_amount - (days * (latte_cost + croissant_cost))) / num_cookies = 1.25 := 
by
  sorry

end cookie_cost_l53_53151


namespace jean_business_hours_l53_53208

-- Definitions of the conditions
def weekday_hours : ℕ := 10 - 16 -- from 4 pm to 10 pm
def weekend_hours : ℕ := 10 - 18 -- from 6 pm to 10 pm
def weekdays : ℕ := 5 -- Monday through Friday
def weekends : ℕ := 2 -- Saturday and Sunday

-- Total weekly hours
def total_weekly_hours : ℕ :=
  (weekday_hours * weekdays) + (weekend_hours * weekends)

-- Proof statement
theorem jean_business_hours : total_weekly_hours = 38 :=
by
  sorry

end jean_business_hours_l53_53208


namespace min_value_a_plus_2b_l53_53636

theorem min_value_a_plus_2b (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 2 * b + 2 * a * b = 8) :
  a + 2 * b ≥ 4 :=
sorry

end min_value_a_plus_2b_l53_53636


namespace percentage_of_water_in_mixture_is_17_14_l53_53581

def Liquid_A_water_percentage : ℝ := 0.10
def Liquid_B_water_percentage : ℝ := 0.15
def Liquid_C_water_percentage : ℝ := 0.25
def Liquid_D_water_percentage : ℝ := 0.35

def parts_A : ℝ := 3
def parts_B : ℝ := 2
def parts_C : ℝ := 1
def parts_D : ℝ := 1

def part_unit : ℝ := 100

noncomputable def total_units : ℝ := 
  parts_A * part_unit + parts_B * part_unit + parts_C * part_unit + parts_D * part_unit

noncomputable def total_water_units : ℝ :=
  parts_A * part_unit * Liquid_A_water_percentage +
  parts_B * part_unit * Liquid_B_water_percentage +
  parts_C * part_unit * Liquid_C_water_percentage +
  parts_D * part_unit * Liquid_D_water_percentage

noncomputable def percentage_water : ℝ := (total_water_units / total_units) * 100

theorem percentage_of_water_in_mixture_is_17_14 :
  percentage_water = 17.14 := sorry

end percentage_of_water_in_mixture_is_17_14_l53_53581


namespace Kelly_weight_is_M_l53_53243

variable (M : ℝ) -- Megan's weight
variable (K : ℝ) -- Kelly's weight
variable (Mike : ℝ) -- Mike's weight

-- Conditions based on the problem statement
def Kelly_less_than_Megan (M K : ℝ) : Prop := K = 0.85 * M
def Mike_greater_than_Megan (M Mike : ℝ) : Prop := Mike = M + 5
def Total_weight_exceeds_bridge (total_weight : ℝ) : Prop := total_weight = 100 + 19
def Total_weight_of_children (M K Mike total_weight : ℝ) : Prop := total_weight = M + K + Mike

theorem Kelly_weight_is_M : (M = 40) → (Total_weight_exceeds_bridge 119) → (Kelly_less_than_Megan M K) → (Mike_greater_than_Megan M Mike) → K = 34 :=
by
  -- Insert proof here
  sorry

end Kelly_weight_is_M_l53_53243


namespace sampling_method_is_stratified_l53_53038

/-- There are 500 boys and 400 girls in the high school senior year.
The total population consists of 900 students.
A random sample of 25 boys and 20 girls was taken.
Prove that the sampling method used is stratified sampling method. -/
theorem sampling_method_is_stratified :
    let boys := 500
    let girls := 400
    let total_students := 900
    let sample_boys := 25
    let sample_girls := 20
    let sampling_method := "Stratified sampling"
    sample_boys < boys ∧ sample_girls < girls → sampling_method = "Stratified sampling"
:=
sorry

end sampling_method_is_stratified_l53_53038


namespace no_integer_solutions_3a2_eq_b2_plus_1_l53_53746

theorem no_integer_solutions_3a2_eq_b2_plus_1 :
  ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 :=
by
  sorry

end no_integer_solutions_3a2_eq_b2_plus_1_l53_53746


namespace ratio_of_areas_l53_53794

theorem ratio_of_areas (C1 C2 : ℝ) (h : (60 / 360) * C1 = (30 / 360) * C2) :
  (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 1 / 4 :=
by
  sorry

end ratio_of_areas_l53_53794


namespace xinxin_nights_at_seaside_l53_53301

-- Definitions from conditions
def arrival_day : ℕ := 30
def may_days : ℕ := 31
def departure_day : ℕ := 4
def nights_spent : ℕ := (departure_day + (may_days - arrival_day))

-- Theorem to prove the number of nights spent
theorem xinxin_nights_at_seaside : nights_spent = 5 := 
by
  -- Include proof steps here in actual Lean proof
  sorry

end xinxin_nights_at_seaside_l53_53301


namespace B_div_A_75_l53_53565

noncomputable def find_ratio (A B : ℝ) (x : ℝ) :=
  (A / (x + 3) + B / (x * (x - 9)) = (x^2 - 3*x + 15) / (x * (x + 3) * (x - 9)))

theorem B_div_A_75 :
  ∀ (A B : ℝ), (∀ (x : ℝ), x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 9 → find_ratio A B x) → 
  B/A = 7.5 :=
by
  sorry

end B_div_A_75_l53_53565


namespace two_f_eq_eight_over_four_plus_x_l53_53472

noncomputable def f : ℝ → ℝ := sorry

theorem two_f_eq_eight_over_four_plus_x (f_def : ∀ x > 0, f (2 * x) = 2 / (2 + x)) :
  ∀ x > 0, 2 * f x = 8 / (4 + x) :=
by
  sorry

end two_f_eq_eight_over_four_plus_x_l53_53472


namespace decomposition_sum_of_cubes_l53_53372

theorem decomposition_sum_of_cubes 
  (a b c d e : ℤ) 
  (h : (512 : ℤ) * x ^ 3 + 27 = (a * x + b) * (c * x ^ 2 + d * x + e)) :
  a + b + c + d + e = 60 := 
sorry

end decomposition_sum_of_cubes_l53_53372


namespace equation_of_line_containing_chord_l53_53694

theorem equation_of_line_containing_chord (x y : ℝ) : 
  (y^2 = -8 * x) ∧ ((-1, 1) = ((x + x) / 2, (y + y) / 2)) →
  4 * x + y + 3 = 0 :=
by 
  sorry

end equation_of_line_containing_chord_l53_53694


namespace double_angle_cosine_l53_53010

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l53_53010


namespace candles_on_rituprts_cake_l53_53689

theorem candles_on_rituprts_cake (peter_candles : ℕ) (rupert_factor : ℝ) 
  (h_peter : peter_candles = 10) (h_rupert : rupert_factor = 3.5) : 
  ∃ rupert_candles : ℕ, rupert_candles = 35 :=
by
  sorry

end candles_on_rituprts_cake_l53_53689


namespace side_length_of_square_base_l53_53888

theorem side_length_of_square_base (area slant_height : ℝ) (h_area : area = 120) (h_slant_height : slant_height = 40) :
  ∃ s : ℝ, area = 0.5 * s * slant_height ∧ s = 6 :=
by
  sorry

end side_length_of_square_base_l53_53888


namespace intersection_area_two_circles_l53_53095

theorem intersection_area_two_circles :
  let r : ℝ := 3
  let center1 : ℝ × ℝ := (3, 0)
  let center2 : ℝ × ℝ := (0, 3)
  let intersection_area := (9 * Real.pi - 18) / 2
  (∃ x y : ℝ, (x - center1.1)^2 + y^2 = r^2 ∧ x^2 + (y - center2.2)^2 = r^2) →
  (∃ (a : ℝ), a = intersection_area) :=
by
  sorry

end intersection_area_two_circles_l53_53095


namespace convex_100gon_distinct_numbers_l53_53048

theorem convex_100gon_distinct_numbers :
  ∀ (vertices : Fin 100 → (ℕ × ℕ)),
  (∀ i, (vertices i).1 ≠ (vertices i).2) →
  ∃ (erase_one_number : ∀ (i : Fin 100), ℕ),
  (∀ i, erase_one_number i = (vertices i).1 ∨ erase_one_number i = (vertices i).2) ∧
  (∀ i j, i ≠ j → (i = j + 1 ∨ (i = 0 ∧ j = 99)) → erase_one_number i ≠ erase_one_number j) :=
by sorry

end convex_100gon_distinct_numbers_l53_53048


namespace quadratic_inequality_solution_l53_53081

theorem quadratic_inequality_solution (x : ℝ) : (2 * x^2 - 5 * x - 3 < 0) ↔ (-1/2 < x ∧ x < 3) :=
by
  sorry

end quadratic_inequality_solution_l53_53081


namespace range_of_p_l53_53374

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - 10 * x

-- A = { x | f'(x) ≤ 0 }
def A : Set ℝ := { x | deriv f x ≤ 0 }

-- B = { x | p + 1 ≤ x ≤ 2p - 1 }
def B (p : ℝ) : Set ℝ := { x | p + 1 ≤ x ∧ x ≤ 2 * p - 1 }

-- Given that A ∪ B = A, prove the range of values for p is ≤ 3.
theorem range_of_p (p : ℝ) : (A ∪ B p = A) → p ≤ 3 := sorry

end range_of_p_l53_53374


namespace find_roots_l53_53537

theorem find_roots (x : ℝ) :
  5 * x^4 - 28 * x^3 + 46 * x^2 - 28 * x + 5 = 0 → x = 3.2 ∨ x = 0.8 ∨ x = 1 :=
by
  intro h
  sorry

end find_roots_l53_53537


namespace polynomial_behavior_l53_53206

noncomputable def Q (x : ℝ) : ℝ := x^6 - 6 * x^5 + 10 * x^4 - x^3 - x + 12

theorem polynomial_behavior : 
  (∀ x : ℝ, x < 0 → Q x > 0) ∧ (∃ x : ℝ, x > 0 ∧ Q x = 0) := 
by 
  sorry

end polynomial_behavior_l53_53206


namespace greatest_difference_l53_53559

theorem greatest_difference (n m : ℕ) (hn : 1023 = 17 * n + m) (hn_pos : 0 < n) (hm_pos : 0 < m) : n - m = 57 :=
sorry

end greatest_difference_l53_53559


namespace abc_inequality_l53_53141

theorem abc_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

end abc_inequality_l53_53141


namespace ice_cream_initial_amount_l53_53749

noncomputable def initial_ice_cream (milkshake_count : ℕ) : ℕ :=
  12 * milkshake_count

theorem ice_cream_initial_amount (m_i m_f : ℕ) (milkshake_count : ℕ) (I_f : ℕ) :
  m_i = 72 →
  m_f = 8 →
  milkshake_count = (m_i - m_f) / 4 →
  I_f = initial_ice_cream milkshake_count →
  I_f = 192 :=
by
  intros hmi hmf hcount hIc
  sorry

end ice_cream_initial_amount_l53_53749


namespace range_of_a_l53_53851

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ (a < -4 ∨ a > 4) := by
  sorry

end range_of_a_l53_53851


namespace base_b_three_digit_count_l53_53902

-- Define the condition that counts the valid three-digit numbers in base b
def num_three_digit_numbers (b : ℕ) : ℕ :=
  (b - 1) ^ 2 * b

-- Define the specific problem statement
theorem base_b_three_digit_count :
  num_three_digit_numbers 4 = 72 :=
by
  -- Proof skipped as per the instruction
  sorry

end base_b_three_digit_count_l53_53902


namespace mode_is_3_5_of_salaries_l53_53322

def salaries : List ℚ := [30, 14, 9, 6, 4, 3.5, 3]
def frequencies : List ℕ := [1, 2, 3, 4, 5, 6, 4]

noncomputable def mode_of_salaries (salaries : List ℚ) (frequencies : List ℕ) : ℚ :=
by
  sorry

theorem mode_is_3_5_of_salaries :
  mode_of_salaries salaries frequencies = 3.5 :=
by
  sorry

end mode_is_3_5_of_salaries_l53_53322


namespace circumscribed_circle_radius_l53_53036

theorem circumscribed_circle_radius (h8 h15 h17 : ℝ) (h_triangle : h8 = 8 ∧ h15 = 15 ∧ h17 = 17) : 
  ∃ R : ℝ, R = 17 := 
sorry

end circumscribed_circle_radius_l53_53036


namespace sum_of_roots_of_quadratic_l53_53349

theorem sum_of_roots_of_quadratic :
  ∀ x1 x2 : ℝ, (∃ a b c, a = -1 ∧ b = 2 ∧ c = 4 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) → (x1 + x2 = 2) :=
by
  sorry

end sum_of_roots_of_quadratic_l53_53349


namespace senior_ticket_cost_l53_53741

variable (tickets_total : ℕ)
variable (adult_ticket_price senior_ticket_price : ℕ)
variable (total_receipts : ℕ)
variable (senior_tickets_sold : ℕ)

theorem senior_ticket_cost (h1 : tickets_total = 529) 
                           (h2 : adult_ticket_price = 25)
                           (h3 : total_receipts = 9745)
                           (h4 : senior_tickets_sold = 348) 
                           (h5 : senior_ticket_price * 348 + 25 * (529 - 348) = 9745) : 
                           senior_ticket_price = 15 := by
  sorry

end senior_ticket_cost_l53_53741


namespace cubic_inequality_l53_53026

theorem cubic_inequality (a b : ℝ) (ha_pos : a > 0) (hb_pos : b > 0) (hne : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := 
sorry

end cubic_inequality_l53_53026


namespace little_john_friends_share_l53_53183

-- Noncomputable definition for dealing with reals
noncomputable def amount_given_to_each_friend :=
  let total_initial := 7.10
  let total_left := 4.05
  let spent_on_sweets := 1.05
  let total_given_away := total_initial - total_left
  let total_given_to_friends := total_given_away - spent_on_sweets
  total_given_to_friends / 2

-- The theorem stating the result
theorem little_john_friends_share :
  amount_given_to_each_friend = 1.00 :=
by
  sorry

end little_john_friends_share_l53_53183


namespace simplify_polynomial_l53_53520

variable (y : ℤ)

theorem simplify_polynomial :
  (3 * y - 2) * (5 * y^12 + 3 * y^11 + 6 * y^10 + 2 * y^9 + 4) = 
  15 * y^13 - y^12 + 12 * y^11 - 6 * y^10 - 4 * y^9 + 12 * y - 8 :=
by
  sorry

end simplify_polynomial_l53_53520


namespace total_lives_after_third_level_l53_53002

def initial_lives : ℕ := 2

def extra_lives_first_level : ℕ := 6
def modifier_first_level (lives : ℕ) : ℕ := lives / 2

def extra_lives_second_level : ℕ := 11
def challenge_second_level (lives : ℕ) : ℕ := lives - 3

def reward_third_level (lives_first_two_levels : ℕ) : ℕ := 2 * lives_first_two_levels

theorem total_lives_after_third_level :
  let lives_first_level := modifier_first_level extra_lives_first_level
  let lives_after_first_level := initial_lives + lives_first_level
  let lives_second_level := challenge_second_level extra_lives_second_level
  let lives_after_second_level := lives_after_first_level + lives_second_level
  let total_gained_lives_first_two_levels := lives_first_level + lives_second_level
  let third_level_reward := reward_third_level total_gained_lives_first_two_levels
  lives_after_second_level + third_level_reward = 35 :=
by
  sorry

end total_lives_after_third_level_l53_53002


namespace lines_intersecting_sum_a_b_l53_53129

theorem lines_intersecting_sum_a_b 
  (a b : ℝ) 
  (hx : ∃ (x y : ℝ), x = 4 ∧ y = 1 ∧ x = 3 * y + a)
  (hy : ∃ (x y : ℝ), x = 4 ∧ y = 1 ∧ y = 3 * x + b)
  : a + b = -10 :=
by
  sorry

end lines_intersecting_sum_a_b_l53_53129


namespace find_sum_A_B_l53_53265

-- Definitions based on conditions
def A : ℤ := -3 - (-5)
def B : ℤ := 2 + (-2)

-- Theorem statement matching the problem
theorem find_sum_A_B : A + B = 2 :=
sorry

end find_sum_A_B_l53_53265


namespace gas_usage_correct_l53_53840

def starting_gas : ℝ := 0.5
def ending_gas : ℝ := 0.16666666666666666

theorem gas_usage_correct : starting_gas - ending_gas = 0.33333333333333334 := by
  sorry

end gas_usage_correct_l53_53840


namespace certain_number_mod_l53_53430

theorem certain_number_mod (n : ℤ) : (73 * n) % 8 = 7 → n % 8 = 7 := 
by sorry

end certain_number_mod_l53_53430


namespace sum_eq_two_l53_53174

theorem sum_eq_two (x y : ℝ) (h : x^2 + y^2 = 10 * x - 6 * y - 34) : x + y = 2 :=
by
  sorry

end sum_eq_two_l53_53174


namespace fourth_root_of_expression_l53_53690

theorem fourth_root_of_expression (x : ℝ) (h : 0 < x) : Real.sqrt (x^3 * Real.sqrt (x^2)) ^ (1 / 4) = x := sorry

end fourth_root_of_expression_l53_53690


namespace price_adjustment_l53_53404

theorem price_adjustment (P : ℝ) (x : ℝ) (hx : P * (1 - (x / 100)^2) = 0.75 * P) : 
  x = 50 :=
by
  -- skipping the proof with sorry
  sorry

end price_adjustment_l53_53404


namespace ellipse_area_l53_53879

/-- 
In a certain ellipse, the endpoints of the major axis are (1, 6) and (21, 6). 
Also, the ellipse passes through the point (19, 9). Prove that the area of the ellipse is 50π. 
-/
theorem ellipse_area : 
  let a := 10
  let b := 5 
  let center := (11, 6)
  let endpoints_major := [(1, 6), (21, 6)]
  let point_on_ellipse := (19, 9)
  ∀ x y, ((x - 11)^2 / a^2) + ((y - 6)^2 / b^2) = 1 → 
    (x, y) = (19, 9) →  -- given point on the ellipse
    (endpoints_major = [(1, 6), (21, 6)]) →  -- given endpoints of the major axis
    50 * Real.pi = π * a * b := 
by
  sorry

end ellipse_area_l53_53879


namespace total_wristbands_proof_l53_53987

-- Definitions from the conditions
def wristbands_per_person : ℕ := 2
def total_wristbands : ℕ := 125

-- Theorem statement to be proved
theorem total_wristbands_proof : total_wristbands = 125 :=
by
  sorry

end total_wristbands_proof_l53_53987


namespace expression_value_zero_l53_53295

theorem expression_value_zero (a b c : ℝ) (h : a^2 + b = b^2 + c ∧ b^2 + c = c^2 + a) : 
  a * (a^2 - b^2) + b * (b^2 - c^2) + c * (c^2 - a^2) = 0 := by
  sorry

end expression_value_zero_l53_53295


namespace A_share_of_profit_l53_53059

theorem A_share_of_profit
  (A_investment : ℤ) (B_investment : ℤ) (C_investment : ℤ)
  (A_profit_share : ℚ) (B_profit_share : ℚ) (C_profit_share : ℚ)
  (total_profit : ℤ) :
  A_investment = 6300 ∧ B_investment = 4200 ∧ C_investment = 10500 ∧
  A_profit_share = 0.45 ∧ B_profit_share = 0.3 ∧ C_profit_share = 0.25 ∧ 
  total_profit = 12200 →
  A_profit_share * total_profit = 5490 :=
by sorry

end A_share_of_profit_l53_53059


namespace value_of_a_l53_53388

variable (a : ℤ)
def U : Set ℤ := {2, 4, 3 - a^2}
def P : Set ℤ := {2, a^2 + 2 - a}

theorem value_of_a (h : (U a) \ (P a) = {-1}) : a = 2 :=
sorry

end value_of_a_l53_53388


namespace unique_pair_l53_53348

theorem unique_pair (m n : ℕ) (h1 : m < n) (h2 : n ∣ m^2 + 1) (h3 : m ∣ n^2 + 1) : (m, n) = (1, 1) :=
sorry

end unique_pair_l53_53348


namespace total_outcomes_l53_53162

-- Define the number of students
def num_students : ℕ := 5

-- Define the number of events
def num_events : ℕ := 3

-- Theorem statement: asserting the total number of different outcomes
theorem total_outcomes : num_students ^ num_events = 125 :=
by
  sorry

end total_outcomes_l53_53162


namespace solve_for_y_l53_53542

noncomputable def find_angle_y : Prop :=
  let AB_CD_are_straight_lines : Prop := True
  let angle_AXB : ℕ := 70
  let angle_BXD : ℕ := 40
  let angle_CYX : ℕ := 100
  let angle_YXZ := 180 - angle_AXB - angle_BXD
  let angle_XYZ := 180 - angle_CYX
  let y := 180 - angle_YXZ - angle_XYZ
  y = 30

theorem solve_for_y : find_angle_y :=
by
  trivial

end solve_for_y_l53_53542


namespace max_dominoes_in_grid_l53_53820

-- Definitions representing the conditions
def total_squares (rows cols : ℕ) : ℕ := rows * cols
def domino_squares : ℕ := 3
def max_dominoes (total domino : ℕ) : ℕ := total / domino

-- Statement of the problem
theorem max_dominoes_in_grid : max_dominoes (total_squares 20 19) domino_squares = 126 :=
by
  -- placeholders for the actual proof
  sorry

end max_dominoes_in_grid_l53_53820


namespace solution_set_correct_l53_53956

theorem solution_set_correct (a b : ℝ) :
  (∀ x : ℝ, - 1 / 2 < x ∧ x < 1 / 3 → ax^2 + bx + 2 > 0) →
  (a - b = -10) :=
by
  sorry

end solution_set_correct_l53_53956


namespace tips_fraction_of_income_l53_53492

theorem tips_fraction_of_income
  (S T : ℝ)
  (h1 : T = (2 / 4) * S) :
  T / (S + T) = 1 / 3 :=
by
  -- Proof goes here
  sorry

end tips_fraction_of_income_l53_53492


namespace calc_hash_80_l53_53762

def hash (N : ℝ) : ℝ := 0.4 * N * 1.5

theorem calc_hash_80 : hash (hash (hash 80)) = 17.28 :=
by 
  sorry

end calc_hash_80_l53_53762


namespace evaluate_at_2_l53_53933

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem evaluate_at_2 : f 2 = 62 := 
by
  sorry

end evaluate_at_2_l53_53933


namespace everyone_can_cross_l53_53595

-- Define each agent
inductive Agent
| C   -- Princess Sonya
| K (i : Fin 8) -- Knights numbered 1 to 7

open Agent

-- Define friendships
def friends (a b : Agent) : Prop :=
  match a, b with
  | C, (K 4) => False
  | (K 4), C => False
  | _, _ => (∃ i : Fin 8, a = K i ∧ b = K (i+1)) ∨ (∃ i : Fin 7, a = K (i+1) ∧ b = K i) ∨ a = C ∨ b = C

-- Define the crossing conditions
def boatCanCarry : List Agent → Prop
| [a, b] => friends a b
| [a, b, c] => friends a b ∧ friends b c ∧ friends a c
| _ => False

-- The main statement to prove
theorem everyone_can_cross (agents : List Agent) (steps : List (List Agent)) :
  agents = [C, K 0, K 1, K 2, K 3, K 4, K 5, K 6, K 7] →
  (∀ step ∈ steps, boatCanCarry step) →
  (∃ final_state : List (List Agent), final_state = [[C, K 0, K 1, K 2, K 3, K 4, K 5, K 6, K 7]]) :=
by 
  -- The proof is omitted.
  sorry

end everyone_can_cross_l53_53595


namespace g_4_minus_g_7_l53_53483

theorem g_4_minus_g_7 (g : ℝ → ℝ) (h_linear : ∀ x y : ℝ, g (x + y) = g x + g y)
  (h_diff : ∀ k : ℝ, g (k + 1) - g k = 5) : g 4 - g 7 = -15 :=
by
  sorry

end g_4_minus_g_7_l53_53483


namespace compare_y1_y2_l53_53060

def parabola (x : ℝ) (c : ℝ) : ℝ := -x^2 + 4 * x + c

theorem compare_y1_y2 (c y1 y2 : ℝ) :
  parabola (-1) c = y1 →
  parabola 1 c = y2 →
  y1 < y2 :=
by
  intro h1 h2
  sorry

end compare_y1_y2_l53_53060


namespace loss_percentage_l53_53131

-- Definitions of cost price (C) and selling price (S)
def cost_price : ℤ := sorry
def selling_price : ℤ := sorry

-- Given condition: Cost price of 40 articles equals selling price of 25 articles
axiom condition : 40 * cost_price = 25 * selling_price

-- Statement to prove: The merchant made a loss of 20%
theorem loss_percentage (C S : ℤ) (h : 40 * C = 25 * S) : 
  ((S - C) * 100) / C = -20 := 
sorry

end loss_percentage_l53_53131


namespace find_angle_B_find_a_plus_c_l53_53983

variable (A B C a b c S : Real)

-- Conditions
axiom h1 : a = (1 / 2) * c + b * Real.cos C
axiom h2 : S = Real.sqrt 3
axiom h3 : b = Real.sqrt 13

-- Questions (Proving the answers from the problem)
theorem find_angle_B (hA : A = Real.pi - (B + C)) : 
  B = Real.pi / 3 := by
  sorry

theorem find_a_plus_c (hac : (1 / 2) * a * c * Real.sin (Real.pi / 3) = Real.sqrt 3) : 
  a + c = 5 := by
  sorry

end find_angle_B_find_a_plus_c_l53_53983


namespace light_flashes_in_three_quarters_hour_l53_53302

theorem light_flashes_in_three_quarters_hour (flash_interval seconds_in_three_quarters_hour : ℕ) 
  (h1 : flash_interval = 15) (h2 : seconds_in_three_quarters_hour = 2700) : 
  (seconds_in_three_quarters_hour / flash_interval = 180) :=
by
  sorry

end light_flashes_in_three_quarters_hour_l53_53302


namespace minewaska_state_park_l53_53803

variable (B H : Nat)

theorem minewaska_state_park (hikers_bike_riders_sum : H + B = 676) (hikers_more_than_bike_riders : H = B + 178) : H = 427 :=
sorry

end minewaska_state_park_l53_53803


namespace stella_profit_l53_53853

def price_of_doll := 5
def price_of_clock := 15
def price_of_glass := 4

def number_of_dolls := 3
def number_of_clocks := 2
def number_of_glasses := 5

def cost := 40

def dolls_sales := number_of_dolls * price_of_doll
def clocks_sales := number_of_clocks * price_of_clock
def glasses_sales := number_of_glasses * price_of_glass

def total_sales := dolls_sales + clocks_sales + glasses_sales

def profit := total_sales - cost

theorem stella_profit : profit = 25 :=
by 
  sorry

end stella_profit_l53_53853


namespace paper_cups_pallets_l53_53821

theorem paper_cups_pallets (total_pallets : ℕ) (paper_towels_fraction tissues_fraction paper_plates_fraction : ℚ) :
  total_pallets = 20 → paper_towels_fraction = 1 / 2 → tissues_fraction = 1 / 4 → paper_plates_fraction = 1 / 5 →
  total_pallets - (total_pallets * paper_towels_fraction + total_pallets * tissues_fraction + total_pallets * paper_plates_fraction) = 1 :=
by sorry

end paper_cups_pallets_l53_53821


namespace max_value_of_trig_expression_l53_53403

open Real

theorem max_value_of_trig_expression : ∀ x : ℝ, 3 * cos x + 4 * sin x ≤ 5 :=
sorry

end max_value_of_trig_expression_l53_53403


namespace proof_problem_l53_53238

open Real

-- Definitions of curves and transformations
def C1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }
def C2 := { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1 }

-- Parametric equation of C2
def parametric_C2 := ∃ α : ℝ, (0 ≤ α ∧ α ≤ 2*π) ∧
  (C2 = { p : ℝ × ℝ | p.1 = 2 * cos α ∧ p.2 = (1/2) * sin α })

-- Equation of line l1 maximizing the perimeter of ABCD
def line_l1 (p : ℝ × ℝ): Prop :=
  p.2 = (1/4) * p.1

theorem proof_problem : parametric_C2 ∧
  ∀ (A B C D : ℝ × ℝ),
    (A ∈ C2 ∧ B ∈ C2 ∧ C ∈ C2 ∧ D ∈ C2) →
    (line_l1 A ∧ line_l1 B) → 
    (line_l1 A ∧ line_l1 B) ∧
    (line_l1 C ∧ line_l1 D) →
    y = (1 / 4) * x :=
sorry

end proof_problem_l53_53238


namespace problem_l53_53220

noncomputable def fx (a b c : ℝ) (x : ℝ) : ℝ := a * x + b / x + c

theorem problem 
  (a b c : ℝ) 
  (h_odd : ∀ x, fx a b c x = -fx a b c (-x))
  (h_f1 : fx a b c 1 = 5 / 2)
  (h_f2 : fx a b c 2 = 17 / 4) :
  (a = 2) ∧ (b = 1 / 2) ∧ (c = 0) ∧ (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 / 2 → fx a b c x₁ > fx a b c x₂) := 
sorry

end problem_l53_53220


namespace incorrect_statement_B_is_wrong_l53_53858

variable (number_of_students : ℕ) (sample_size : ℕ) (population : Set ℕ) (sample : Set ℕ)

-- Conditions
def school_population_is_4000 := number_of_students = 4000
def sample_selected_is_400 := sample_size = 400
def valid_population := population = { x | x < 4000 }
def valid_sample := sample = { x | x < 400 }

-- Incorrect statement (as per given solution)
def incorrect_statement_B := ¬(∀ student ∈ population, true)

theorem incorrect_statement_B_is_wrong 
  (h1 : school_population_is_4000 number_of_students)
  (h2 : sample_selected_is_400 sample_size)
  (h3 : valid_population population)
  (h4 : valid_sample sample)
  : incorrect_statement_B population :=
sorry

end incorrect_statement_B_is_wrong_l53_53858


namespace work_ratio_l53_53475

theorem work_ratio 
  (m b : ℝ) 
  (h : 7 * m + 2 * b = 6 * (m + b)) : 
  m / b = 4 := 
sorry

end work_ratio_l53_53475


namespace sum_of_repeating_decimals_l53_53633

/-- Definitions of the repeating decimals as real numbers. --/
def x : ℝ := 0.3 -- This actually represents 0.\overline{3} in Lean
def y : ℝ := 0.04 -- This actually represents 0.\overline{04} in Lean
def z : ℝ := 0.005 -- This actually represents 0.\overline{005} in Lean

/-- The theorem stating that the sum of these repeating decimals is a specific fraction. --/
theorem sum_of_repeating_decimals : x + y + z = (14 : ℝ) / 37 := 
by 
  sorry -- Placeholder for the proof

end sum_of_repeating_decimals_l53_53633


namespace trapezoid_PQRS_perimeter_l53_53913

noncomputable def trapezoid_perimeter (PQ RS : ℝ) (height : ℝ) (PS QR : ℝ) : ℝ :=
  PQ + RS + PS + QR

theorem trapezoid_PQRS_perimeter :
  ∀ (PQ RS : ℝ) (height : ℝ)
  (PS QR : ℝ),
  PQ = 6 →
  RS = 10 →
  height = 5 →
  PS = Real.sqrt (5^2 + 4^2) →
  QR = Real.sqrt (5^2 + 4^2) →
  trapezoid_perimeter PQ RS height PS QR = 16 + 2 * Real.sqrt 41 :=
by
  intros
  sorry

end trapezoid_PQRS_perimeter_l53_53913


namespace cost_difference_per_square_inch_l53_53025

theorem cost_difference_per_square_inch (width1 height1 width2 height2 : ℕ) (cost1 cost2 : ℕ)
  (h_size1 : width1 = 24 ∧ height1 = 16)
  (h_cost1 : cost1 = 672)
  (h_size2 : width2 = 48 ∧ height2 = 32)
  (h_cost2 : cost2 = 1152) :
  (cost1 / (width1 * height1) : ℚ) - (cost2 / (width2 * height2) : ℚ) = 1 := 
by
  sorry

end cost_difference_per_square_inch_l53_53025


namespace quadratic_inequality_solution_set_l53_53335

theorem quadratic_inequality_solution_set (a b c : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ (b^2 - 4 * a * c) < 0) :=
by sorry

end quadratic_inequality_solution_set_l53_53335


namespace days_of_harvest_l53_53062

-- Conditions
def ripeOrangesPerDay : ℕ := 82
def totalRipeOranges : ℕ := 2050

-- Problem statement: Prove the number of days of harvest
theorem days_of_harvest : (totalRipeOranges / ripeOrangesPerDay) = 25 :=
by
  sorry

end days_of_harvest_l53_53062


namespace find_m_value_l53_53868

theorem find_m_value :
  let x_values := [8, 9.5, m, 10.5, 12]
  let y_values := [16, 10, 8, 6, 5]
  let regression_eq (x : ℝ) := -3.5 * x + 44
  let avg (l : List ℝ) := l.sum / l.length
  avg y_values = 9 →
  avg x_values = (40 + m) / 5 →
  9 = regression_eq (avg x_values) →
  m = 10 :=
by
  sorry

end find_m_value_l53_53868


namespace max_f_value_range_of_a_l53_53509

noncomputable def f (x a : ℝ) : ℝ := |x + 1| - |x - 4| - a

theorem max_f_value (a : ℝ) : ∃ x, f x a = 5 - a :=
sorry

theorem range_of_a (a : ℝ) : (∃ x, f x a ≥ (4 / a) + 1) ↔ (a = 2 ∨ a < 0) :=
sorry

end max_f_value_range_of_a_l53_53509


namespace initial_candy_bobby_l53_53602

-- Definitions given conditions
def initial_candy (x : ℕ) : Prop :=
  (x + 42 = 70)

-- Theorem statement
theorem initial_candy_bobby : ∃ x : ℕ, initial_candy x ∧ x = 28 :=
by {
  sorry
}

end initial_candy_bobby_l53_53602


namespace remainder_of_3056_div_78_l53_53727

-- Define the necessary conditions and the statement
theorem remainder_of_3056_div_78 : (3056 % 78) = 14 :=
by
  sorry

end remainder_of_3056_div_78_l53_53727


namespace profit_percentage_l53_53699

theorem profit_percentage (cost_price selling_price marked_price : ℝ)
  (h1 : cost_price = 47.50)
  (h2 : selling_price = 0.90 * marked_price)
  (h3 : selling_price = 65.97) :
  ((selling_price - cost_price) / cost_price) * 100 = 38.88 := 
by
  sorry

end profit_percentage_l53_53699


namespace power_difference_divisible_by_10000_l53_53930

theorem power_difference_divisible_by_10000 (a b : ℤ) (m : ℤ) (h : a - b = 100 * m) : ∃ k : ℤ, a^100 - b^100 = 10000 * k := by
  sorry

end power_difference_divisible_by_10000_l53_53930


namespace wax_he_has_l53_53921

def total_wax : ℕ := 353
def additional_wax : ℕ := 22

theorem wax_he_has : total_wax - additional_wax = 331 := by
  sorry

end wax_he_has_l53_53921


namespace completing_the_square_l53_53571

theorem completing_the_square (x : ℝ) (h : x^2 - 6 * x + 7 = 0) : (x - 3)^2 - 2 = 0 := 
by sorry

end completing_the_square_l53_53571


namespace paint_left_after_two_coats_l53_53041

theorem paint_left_after_two_coats :
  let initial_paint := 3 -- liters
  let first_coat_paint := initial_paint / 2
  let paint_after_first_coat := initial_paint - first_coat_paint
  let second_coat_paint := (2 / 3) * paint_after_first_coat
  let paint_after_second_coat := paint_after_first_coat - second_coat_paint
  (paint_after_second_coat * 1000) = 500 := by
  sorry

end paint_left_after_two_coats_l53_53041


namespace solve_inequalities_l53_53800

theorem solve_inequalities (x : ℝ) (h₁ : 5 * x - 8 > 12 - 2 * x) (h₂ : |x - 1| ≤ 3) : 
  (20 / 7) < x ∧ x ≤ 4 :=
by
  sorry

end solve_inequalities_l53_53800


namespace average_value_of_items_in_loot_box_l53_53654

-- Definitions as per the given conditions
def cost_per_loot_box : ℝ := 5
def total_spent : ℝ := 40
def total_loss : ℝ := 12

-- Proving the average value of items inside each loot box
theorem average_value_of_items_in_loot_box :
  (total_spent - total_loss) / (total_spent / cost_per_loot_box) = 3.50 := by
  sorry

end average_value_of_items_in_loot_box_l53_53654


namespace problem_1_problem_2_l53_53812

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x - 1|

-- Problem 1: When a = -1, prove the solution set for f(x) ≤ 2 is [-1/2, 1/2].
theorem problem_1 (x : ℝ) : (f x (-1) ≤ 2) ↔ (-1/2 ≤ x ∧ x ≤ 1/2) := 
sorry

-- Problem 2: If the solution set of f(x) ≤ |2x + 1| contains the interval [1/2, 1], find the range of a.
theorem problem_2 (a : ℝ) : (∀ x, (1/2 ≤ x ∧ x ≤ 1) → f x a ≤ |2 * x + 1|) ↔ (0 ≤ a ∧ a ≤ 3) :=
sorry

end problem_1_problem_2_l53_53812


namespace candy_remaining_l53_53087

theorem candy_remaining
  (initial_candies : ℕ)
  (talitha_took : ℕ)
  (solomon_took : ℕ)
  (h_initial : initial_candies = 349)
  (h_talitha : talitha_took = 108)
  (h_solomon : solomon_took = 153) :
  initial_candies - (talitha_took + solomon_took) = 88 :=
by
  sorry

end candy_remaining_l53_53087


namespace magnitude_difference_l53_53248

open Complex

noncomputable def c1 : ℂ := 18 - 5 * I
noncomputable def c2 : ℂ := 14 + 6 * I
noncomputable def c3 : ℂ := 3 - 12 * I
noncomputable def c4 : ℂ := 4 + 9 * I

theorem magnitude_difference : 
  Complex.abs ((c1 * c2) - (c3 * c4)) = Real.sqrt 146365 :=
by
  sorry

end magnitude_difference_l53_53248


namespace total_shells_correct_l53_53447

def morning_shells : ℕ := 292
def afternoon_shells : ℕ := 324

theorem total_shells_correct : morning_shells + afternoon_shells = 616 := by
  sorry

end total_shells_correct_l53_53447


namespace power_equiv_l53_53965

theorem power_equiv (x_0 : ℝ) (h : x_0 ^ 11 + x_0 ^ 7 + x_0 ^ 3 = 1) : x_0 ^ 4 + x_0 ^ 3 - 1 = x_0 ^ 15 :=
by
  -- the proof goes here
  sorry

end power_equiv_l53_53965


namespace div_expression_l53_53406

theorem div_expression : 180 / (12 + 13 * 2) = 90 / 19 := 
  sorry

end div_expression_l53_53406


namespace noRepeatedDigitsFourDigit_noRepeatedDigitsFiveDigitDiv5_noRepeatedDigitsFourDigitGreaterThan1325_l53_53839

-- Problem 1: Four-digit numbers with no repeated digits
theorem noRepeatedDigitsFourDigit :
  ∃ (n : ℕ), (n = 120) := sorry

-- Problem 2: Five-digit numbers with no repeated digits and divisible by 5
theorem noRepeatedDigitsFiveDigitDiv5 :
  ∃ (n : ℕ), (n = 216) := sorry

-- Problem 3: Four-digit numbers with no repeated digits and greater than 1325
theorem noRepeatedDigitsFourDigitGreaterThan1325 :
  ∃ (n : ℕ), (n = 181) := sorry

end noRepeatedDigitsFourDigit_noRepeatedDigitsFiveDigitDiv5_noRepeatedDigitsFourDigitGreaterThan1325_l53_53839


namespace number_of_graphic_novels_l53_53438

theorem number_of_graphic_novels (total_books novels_percent comics_percent : ℝ) 
  (h_total : total_books = 120) 
  (h_novels_percent : novels_percent = 0.65) 
  (h_comics_percent : comics_percent = 0.20) :
  total_books - (novels_percent * total_books + comics_percent * total_books) = 18 :=
by
  sorry

end number_of_graphic_novels_l53_53438


namespace transformed_line_l53_53292

-- Define the original line equation
def original_line (x y : ℝ) : Prop := (x - 2 * y = 2)

-- Define the transformation
def transformation (x y x' y' : ℝ) : Prop :=
  (x' = x) ∧ (y' = 2 * y)

-- Prove that the transformed line equation holds
theorem transformed_line (x y x' y' : ℝ) (h₁ : original_line x y) (h₂ : transformation x y x' y') :
  x' - y' = 2 :=
sorry

end transformed_line_l53_53292


namespace slope_angle_at_point_l53_53065

def f (x : ℝ) : ℝ := 2 * x^3 - 7 * x + 2

theorem slope_angle_at_point :
  let deriv_f := fun x : ℝ => 6 * x^2 - 7
  let slope := deriv_f 1
  let angle := Real.arctan slope
  angle = (3 * Real.pi) / 4 :=
by
  sorry

end slope_angle_at_point_l53_53065


namespace simplify_and_evaluate_l53_53180

theorem simplify_and_evaluate (x : ℝ) (h : x = 3) : ((x - 2) / (x - 1)) / ((x + 1) - (3 / (x - 1))) = 1 / 5 :=
by
  sorry

end simplify_and_evaluate_l53_53180


namespace percentage_of_men_l53_53110

variable (M : ℝ)

theorem percentage_of_men (h1 : 0.20 * M + 0.40 * (1 - M) = 0.33) : 
  M = 0.35 :=
sorry

end percentage_of_men_l53_53110


namespace crayons_at_the_end_of_thursday_l53_53594

-- Definitions for each day's changes
def monday_crayons : ℕ := 7
def tuesday_crayons (initial : ℕ) := initial + 3
def wednesday_crayons (initial : ℕ) := initial - 5 + 4
def thursday_crayons (initial : ℕ) := initial + 6 - 2

-- Proof statement to show the number of crayons at the end of Thursday
theorem crayons_at_the_end_of_thursday : thursday_crayons (wednesday_crayons (tuesday_crayons monday_crayons)) = 13 :=
by
  sorry

end crayons_at_the_end_of_thursday_l53_53594


namespace percent_of_dollar_in_pocket_l53_53917

def value_of_penny : ℕ := 1  -- value of one penny in cents
def value_of_nickel : ℕ := 5  -- value of one nickel in cents
def value_of_half_dollar : ℕ := 50 -- value of one half-dollar in cents

def pennies : ℕ := 3  -- number of pennies
def nickels : ℕ := 2  -- number of nickels
def half_dollars : ℕ := 1  -- number of half-dollars

def total_value_in_cents : ℕ :=
  (pennies * value_of_penny) + (nickels * value_of_nickel) + (half_dollars * value_of_half_dollar)

def value_of_dollar_in_cents : ℕ := 100

def percent_of_dollar (value : ℕ) (total : ℕ) : ℚ := (value / total) * 100

theorem percent_of_dollar_in_pocket : percent_of_dollar total_value_in_cents value_of_dollar_in_cents = 63 :=
by
  sorry

end percent_of_dollar_in_pocket_l53_53917


namespace irr_sqrt6_l53_53007

open Real

theorem irr_sqrt6 : ¬ ∃ (q : ℚ), (↑q : ℝ) = sqrt 6 := by
  sorry

end irr_sqrt6_l53_53007


namespace monomial_2024_l53_53909

def monomial (n : ℕ) : ℤ × ℕ := ((-1)^(n + 1) * (2 * n - 1), n)

theorem monomial_2024 :
  monomial 2024 = (-4047, 2024) :=
sorry

end monomial_2024_l53_53909


namespace find_mini_cupcakes_l53_53414

-- Definitions of the conditions
def number_of_donut_holes := 12
def number_of_students := 13
def desserts_per_student := 2

-- Statement of the theorem to prove the number of mini-cupcakes is 14
theorem find_mini_cupcakes :
  let D := number_of_donut_holes
  let N := number_of_students
  let total_desserts := N * desserts_per_student
  let C := total_desserts - D
  C = 14 :=
by
  sorry

end find_mini_cupcakes_l53_53414


namespace sum_series_equals_three_fourths_l53_53384

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l53_53384


namespace distance_per_interval_l53_53772

-- Definitions for the conditions
def total_distance : ℕ := 3  -- miles
def total_time : ℕ := 45  -- minutes
def interval_time : ℕ := 15  -- minutes per interval

-- Mathematical problem statement
theorem distance_per_interval :
  (total_distance / (total_time / interval_time) = 1) :=
by 
  sorry

end distance_per_interval_l53_53772


namespace somu_one_fifth_age_back_l53_53459

theorem somu_one_fifth_age_back {S F Y : ℕ}
  (h1 : S = 16)
  (h2 : S = F / 3)
  (h3 : S - Y = (F - Y) / 5) :
  Y = 8 :=
by
  sorry

end somu_one_fifth_age_back_l53_53459


namespace andrea_fewer_apples_l53_53512

theorem andrea_fewer_apples {total_apples given_to_zenny kept_by_yanna given_to_andrea : ℕ} 
  (h1 : total_apples = 60) 
  (h2 : given_to_zenny = 18) 
  (h3 : kept_by_yanna = 36) 
  (h4 : given_to_andrea = total_apples - kept_by_yanna - given_to_zenny) : 
  (given_to_andrea + 12 = given_to_zenny) := 
sorry

end andrea_fewer_apples_l53_53512


namespace complete_the_square_l53_53929

theorem complete_the_square :
  ∀ (x : ℝ), (x^2 + 14 * x + 24 = 0) → (∃ c d : ℝ, (x + c)^2 = d ∧ d = 25) :=
by
  intro x h
  sorry

end complete_the_square_l53_53929


namespace top_four_cards_probability_l53_53377

def num_cards : ℕ := 52

def num_hearts : ℕ := 13

def num_diamonds : ℕ := 13

def num_clubs : ℕ := 13

def prob_first_heart := (num_hearts : ℚ) / num_cards
def prob_second_heart := (num_hearts - 1 : ℚ) / (num_cards - 1)
def prob_third_diamond := (num_diamonds : ℚ) / (num_cards - 2)
def prob_fourth_club := (num_clubs : ℚ) / (num_cards - 3)

def combined_prob :=
  prob_first_heart * prob_second_heart * prob_third_diamond * prob_fourth_club

theorem top_four_cards_probability :
  combined_prob = 39 / 63875 := by
  sorry

end top_four_cards_probability_l53_53377


namespace max_area_rectangle_l53_53799

theorem max_area_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 40) : x * y ≤ 100 :=
by
  sorry

end max_area_rectangle_l53_53799


namespace first_car_gas_consumed_l53_53739

theorem first_car_gas_consumed 
    (sum_avg_mpg : ℝ) (g2_gallons : ℝ) (total_miles : ℝ) 
    (avg_mpg_car1 : ℝ) (avg_mpg_car2 : ℝ) (g1_gallons : ℝ) :
    sum_avg_mpg = avg_mpg_car1 + avg_mpg_car2 →
    g2_gallons = 35 →
    total_miles = 2275 →
    avg_mpg_car1 = 40 →
    avg_mpg_car2 = 35 →
    g1_gallons = (total_miles - (avg_mpg_car2 * g2_gallons)) / avg_mpg_car1 →
    g1_gallons = 26.25 :=
by
  intros h_sum_avg_mpg h_g2_gallons h_total_miles h_avg_mpg_car1 h_avg_mpg_car2 h_g1_gallons
  sorry

end first_car_gas_consumed_l53_53739


namespace smallest_number_condition_l53_53870

def smallest_number := 1621432330
def primes := [29, 53, 37, 41, 47, 61]
def lcm_of_primes := primes.prod

theorem smallest_number_condition :
  ∃ k : ℕ, 5 * (smallest_number + 11) = k * lcm_of_primes ∧
          (∀ y, (∃ m : ℕ, 5 * (y + 11) = m * lcm_of_primes) → smallest_number ≤ y) :=
by
  -- The proof goes here
  sorry

#print smallest_number_condition

end smallest_number_condition_l53_53870


namespace product_of_roots_of_quadratic_l53_53313

   -- Definition of the quadratic equation used in the condition
   def quadratic (x : ℝ) : ℝ := x^2 - 2 * x - 8

   -- Problem statement: Prove that the product of the roots of the given quadratic equation is -8.
   theorem product_of_roots_of_quadratic : 
     (∀ x : ℝ, quadratic x = 0 → (x = 4 ∨ x = -2)) → (4 * -2 = -8) :=
   by
     sorry
   
end product_of_roots_of_quadratic_l53_53313


namespace rainfall_mondays_l53_53724

theorem rainfall_mondays
  (M : ℕ)
  (rain_monday : ℝ)
  (rain_tuesday : ℝ)
  (num_tuesdays : ℕ)
  (extra_rain_tuesdays : ℝ)
  (h1 : rain_monday = 1.5)
  (h2 : rain_tuesday = 2.5)
  (h3 : num_tuesdays = 9)
  (h4 : num_tuesdays * rain_tuesday = rain_monday * M + extra_rain_tuesdays)
  (h5 : extra_rain_tuesdays = 12) :
  M = 7 := 
sorry

end rainfall_mondays_l53_53724


namespace youseff_blocks_l53_53928

-- Definition of the conditions
def time_to_walk (x : ℕ) : ℕ := x
def time_to_ride (x : ℕ) : ℕ := (20 * x) / 60
def extra_time (x : ℕ) : ℕ := time_to_walk x - time_to_ride x

-- Statement of the problem in Lean
theorem youseff_blocks : ∃ x : ℕ, extra_time x = 6 ∧ x = 9 :=
by {
  sorry
}

end youseff_blocks_l53_53928


namespace no_digit_c_make_2C4_multiple_of_5_l53_53813

theorem no_digit_c_make_2C4_multiple_of_5 : ∀ C, ¬ (C ≥ 0 ∧ C ≤ 9 ∧ (20 * 10 + C * 10 + 4) % 5 = 0) :=
by intros C hC; sorry

end no_digit_c_make_2C4_multiple_of_5_l53_53813


namespace rectangles_in_cube_l53_53756

/-- Number of rectangles that can be formed by the vertices of a cube is 12. -/
theorem rectangles_in_cube : 
  ∃ (n : ℕ), (n = 12) := by
  -- The cube has vertices, and squares are a subset of rectangles.
  -- We need to count rectangles including squares among vertices of the cube.
  sorry

end rectangles_in_cube_l53_53756


namespace volume_of_bag_l53_53324

-- Define the dimensions of the cuboid
def width : ℕ := 9
def length : ℕ := 4
def height : ℕ := 7

-- Define the volume calculation function for a cuboid
def volume (l w h : ℕ) : ℕ :=
  l * w * h

-- Provide the theorem to prove the volume is 252 cubic centimeters
theorem volume_of_bag : volume length width height = 252 := by
  -- Since the proof is not requested, insert sorry to complete the statement.
  sorry

end volume_of_bag_l53_53324


namespace minimal_ab_l53_53054

theorem minimal_ab (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
(h : 1 / (a : ℝ) + 1 / (3 * b : ℝ) = 1 / 9) : a * b = 60 :=
sorry

end minimal_ab_l53_53054


namespace dolls_total_l53_53466

theorem dolls_total (dina_dolls ivy_dolls casey_dolls : ℕ) 
  (h1 : dina_dolls = 2 * ivy_dolls)
  (h2 : (2 / 3 : ℚ) * ivy_dolls = 20)
  (h3 : casey_dolls = 5 * 20) :
  dina_dolls + ivy_dolls + casey_dolls = 190 :=
by sorry

end dolls_total_l53_53466


namespace arithmetic_evaluation_l53_53881

theorem arithmetic_evaluation : 6 * 2 - 3 = 9 := by
  sorry

end arithmetic_evaluation_l53_53881


namespace route_Y_is_quicker_l53_53624

noncomputable def route_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

def route_X_distance : ℝ := 8
def route_X_speed : ℝ := 40

def route_Y_total_distance : ℝ := 7
def route_Y_construction_distance : ℝ := 1
def route_Y_construction_speed : ℝ := 20
def route_Y_regular_speed_distance : ℝ := 6
def route_Y_regular_speed : ℝ := 50

noncomputable def route_X_time : ℝ :=
  route_time route_X_distance route_X_speed * 60  -- converting to minutes

noncomputable def route_Y_time : ℝ :=
  (route_time route_Y_regular_speed_distance route_Y_regular_speed +
  route_time route_Y_construction_distance route_Y_construction_speed) * 60 -- converting to minutes

theorem route_Y_is_quicker : route_X_time - route_Y_time = 1.8 :=
  by
    sorry

end route_Y_is_quicker_l53_53624


namespace union_M_N_eq_M_l53_53555

-- Define set M
def M : Set ℝ := { y | ∃ x : ℝ, y = 2^x }

-- Define set N
def N : Set ℝ := { y | ∃ x : ℝ, y = Real.log (x - 1) }

-- Statement to prove that M ∪ N = M
theorem union_M_N_eq_M : M ∪ N = M := by
  sorry

end union_M_N_eq_M_l53_53555


namespace ratio_new_radius_l53_53600

theorem ratio_new_radius (r R h : ℝ) (h₀ : π * r^2 * h = 6) (h₁ : π * R^2 * h = 186) : R / r = Real.sqrt 31 :=
by
  sorry

end ratio_new_radius_l53_53600


namespace find_triangle_with_properties_l53_53253

-- Define the angles forming an arithmetic progression
def angles_arithmetic_progression (α β γ : ℝ) : Prop :=
  β - α = γ - β

-- Define the sides forming an arithmetic progression
def sides_arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

-- Define the sides forming a geometric progression
def sides_geometric_progression (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Define the sum of angles in a triangle
def sum_of_angles (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- The problem statement:
theorem find_triangle_with_properties 
    (α β γ a b c : ℝ)
    (h1 : angles_arithmetic_progression α β γ)
    (h2 : sum_of_angles α β γ)
    (h3 : sides_arithmetic_progression a b c ∨ sides_geometric_progression a b c) :
  α = 60 ∧ β = 60 ∧ γ = 60 :=
by 
  sorry

end find_triangle_with_properties_l53_53253


namespace first_number_percentage_of_second_l53_53100

theorem first_number_percentage_of_second {X : ℝ} (H1 : ℝ) (H2 : ℝ) 
  (H1_def : H1 = 0.05 * X) (H2_def : H2 = 0.25 * X) : 
  (H1 / H2) * 100 = 20 :=
by
  sorry

end first_number_percentage_of_second_l53_53100


namespace sufficient_not_necessary_condition_l53_53070

noncomputable def setA (x : ℝ) : Prop := 
  (Real.log x / Real.log 2 - 1) * (Real.log x / Real.log 2 - 3) ≤ 0

noncomputable def setB (x : ℝ) (a : ℝ) : Prop := 
  (2 * x - a) / (x + 1) > 1

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, setA x → setB x a) ∧ (¬ ∀ x, setB x a → setA x) ↔ 
  -2 < a ∧ a < 1 := 
  sorry

end sufficient_not_necessary_condition_l53_53070


namespace projection_of_sum_on_vec_a_l53_53399

open Real

noncomputable def vector_projection (a b : ℝ) (angle : ℝ) : ℝ := 
  (cos angle) * (a * b) / a

theorem projection_of_sum_on_vec_a (a b : EuclideanSpace ℝ (Fin 3)) 
  (h₁ : ‖a‖ = 2) 
  (h₂ : ‖b‖ = 2) 
  (h₃ : inner a b = (2 * 2) * (cos (π / 3))):
  (inner (a + b) a) / ‖a‖ = 3 := 
by
  sorry

end projection_of_sum_on_vec_a_l53_53399


namespace compare_fractions_compare_integers_l53_53366

-- First comparison: Prove -4/7 > -2/3
theorem compare_fractions : - (4 : ℚ) / 7 > - (2 : ℚ) / 3 := 
by sorry

-- Second comparison: Prove -(-7) > -| -7 |
theorem compare_integers : -(-7) > -abs (-7) := 
by sorry

end compare_fractions_compare_integers_l53_53366


namespace vertex_and_segment_condition_g_monotonically_increasing_g_minimum_value_l53_53926

def f (x : ℝ) : ℝ := -x^2 + 2 * x + 15
def g (x a : ℝ) : ℝ := (2 - 2 * a) * x - f x

theorem vertex_and_segment_condition : 
  (f 1 = 16) ∧ ∃ x1 x2 : ℝ, (f x1 = 0) ∧ (f x2 = 0) ∧ (x2 - x1 = 8) := 
sorry

theorem g_monotonically_increasing (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2 → g x1 a ≤ g x2 a) ↔ a ≤ 0 :=
sorry

theorem g_minimum_value (a : ℝ) :
  (0 < a ∧ g 2 a = -4 * a - 11) ∨ (a < 0 ∧ g 0 a = -15) ∨ (0 ≤ a ∧ a ≤ 2 ∧ g a a = -a^2 - 15) :=
sorry

end vertex_and_segment_condition_g_monotonically_increasing_g_minimum_value_l53_53926


namespace polygon_with_150_degree_interior_angles_has_12_sides_l53_53710

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l53_53710


namespace infinitely_many_perfect_squares_of_form_l53_53944

theorem infinitely_many_perfect_squares_of_form (k : ℕ) (h : k > 0) : 
  ∃ (n : ℕ), ∃ m : ℕ, n * 2^k - 7 = m^2 :=
by
  sorry

end infinitely_many_perfect_squares_of_form_l53_53944


namespace average_yield_per_tree_l53_53083

theorem average_yield_per_tree :
  let t1 := 3
  let t2 := 2
  let t3 := 1
  let nuts1 := 60
  let nuts2 := 120
  let nuts3 := 180
  let total_nuts := t1 * nuts1 + t2 * nuts2 + t3 * nuts3
  let total_trees := t1 + t2 + t3
  let average_yield := total_nuts / total_trees
  average_yield = 100 := 
by
  sorry

end average_yield_per_tree_l53_53083


namespace range_of_b_l53_53558

theorem range_of_b (b : ℝ) :
  (∀ x : ℝ, |3 * x - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) → (5 < b ∧ b < 7) :=
sorry

end range_of_b_l53_53558


namespace typing_pages_l53_53695

theorem typing_pages (typists : ℕ) (pages min : ℕ) 
  (h_typists_can_type_two_pages_in_two_minutes : typists * 2 / min = pages / min) 
  (h_10_typists_type_25_pages_in_5_minutes : 10 * 25 / 5 = pages / min) :
  pages / min = 2 := 
sorry

end typing_pages_l53_53695


namespace rahul_share_l53_53575

theorem rahul_share :
  let total_payment := 370
  let bonus := 30
  let remaining_payment := total_payment - bonus
  let rahul_work_per_day := 1 / 3
  let rajesh_work_per_day := 1 / 2
  let ramesh_work_per_day := 1 / 4
  
  let total_work_per_day := rahul_work_per_day + rajesh_work_per_day + ramesh_work_per_day
  let rahul_share_of_work := rahul_work_per_day / total_work_per_day
  let rahul_payment := rahul_share_of_work * remaining_payment

  rahul_payment = 80 :=
by {
  sorry
}

end rahul_share_l53_53575


namespace sample_size_calculation_l53_53768

theorem sample_size_calculation 
    (total_teachers : ℕ) (total_male_students : ℕ) (total_female_students : ℕ) 
    (sample_size_female_students : ℕ) 
    (H1 : total_teachers = 100) (H2 : total_male_students = 600) 
    (H3 : total_female_students = 500) (H4 : sample_size_female_students = 40)
    : (sample_size_female_students * (total_teachers + total_male_students + total_female_students) / total_female_students) = 96 := 
by
  /- sorry, proof omitted -/
  sorry
  
end sample_size_calculation_l53_53768


namespace max_area_of_rectangle_l53_53170

-- Question: Prove the largest possible area of a rectangle given the conditions
theorem max_area_of_rectangle :
  ∀ (x : ℝ), (2 * x + 2 * (x + 5) = 60) → x * (x + 5) ≤ 218.75 :=
by
  sorry

end max_area_of_rectangle_l53_53170


namespace quadratic_eq_coeff_m_l53_53810

theorem quadratic_eq_coeff_m (m : ℤ) : 
  (|m| = 2 ∧ m + 2 ≠ 0) → m = 2 := 
by
  intro h
  sorry

end quadratic_eq_coeff_m_l53_53810


namespace parabola_focus_l53_53163

theorem parabola_focus (x y : ℝ) (p : ℝ) (h_eq : x^2 = 8 * y) (h_form : x^2 = 4 * p * y) : 
  p = 2 ∧ y = (x^2 / 8) ∧ (0, p) = (0, 2) :=
by
  sorry

end parabola_focus_l53_53163


namespace positive_solution_of_x_l53_53614

theorem positive_solution_of_x :
  ∃ x y z : ℝ, (x * y = 6 - 2 * x - 3 * y) ∧ (y * z = 6 - 4 * y - 2 * z) ∧ (x * z = 30 - 4 * x - 3 * z) ∧ x > 0 ∧ x = 3 :=
by
  sorry

end positive_solution_of_x_l53_53614


namespace probability_of_selecting_GEARS_letter_l53_53648

def bag : List Char := ['A', 'L', 'G', 'E', 'B', 'R', 'A', 'S']
def target_word : List Char := ['G', 'E', 'A', 'R', 'S']

theorem probability_of_selecting_GEARS_letter :
  (6 : ℚ) / 8 = 3 / 4 :=
by
  sorry

end probability_of_selecting_GEARS_letter_l53_53648


namespace jon_initial_fastball_speed_l53_53199

theorem jon_initial_fastball_speed 
  (S : ℝ) -- Condition: Jon's initial fastball speed \( S \)
  (h1 : ∀ t : ℕ, t = 4 * 4)  -- Condition: Training time is 4 times for 4 weeks each
  (h2 : ∀ w : ℕ, w = 16)  -- Condition: Total weeks of training (4*4=16)
  (h3 : ∀ g : ℝ, g = 1)  -- Condition: Gains 1 mph per week
  (h4 : ∃ S_new : ℝ, S_new = (S + 16) ∧ S_new = 1.2 * S) -- Condition: Speed increases by 20%
  : S = 80 := 
sorry

end jon_initial_fastball_speed_l53_53199


namespace average_salary_rest_workers_l53_53525

-- Define the conditions
def total_workers : Nat := 21
def average_salary_all_workers : ℝ := 8000
def number_of_technicians : Nat := 7
def average_salary_technicians : ℝ := 12000

-- Define the task
theorem average_salary_rest_workers :
  let number_of_rest := total_workers - number_of_technicians
  let total_salary_all := average_salary_all_workers * total_workers
  let total_salary_technicians := average_salary_technicians * number_of_technicians
  let total_salary_rest := total_salary_all - total_salary_technicians
  let average_salary_rest := total_salary_rest / number_of_rest
  average_salary_rest = 6000 :=
by
  sorry

end average_salary_rest_workers_l53_53525


namespace find_y_intersection_of_tangents_l53_53360

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define the tangent slope at a point on the parabola
def tangent_slope (x : ℝ) : ℝ := 2 * (x - 1)

-- Define the perpendicular condition for tangents at points A and B
def perpendicular_condition (a b : ℝ) : Prop := (a - 1) * (b - 1) = -1 / 4

-- Define the y-coordinate of the intersection point P of the tangents at A and B
def y_coordinate_of_intersection (a b : ℝ) : ℝ := a * b - a - b + 2

-- Theorem to be proved
theorem find_y_intersection_of_tangents (a b : ℝ) 
  (ha : parabola a = a ^ 2 - 2 * a - 3) 
  (hb : parabola b = b ^ 2 - 2 * b - 3) 
  (hp : perpendicular_condition a b) :
  y_coordinate_of_intersection a b = -1 / 4 :=
sorry

end find_y_intersection_of_tangents_l53_53360


namespace probability_A_not_losing_l53_53560

theorem probability_A_not_losing (P_draw : ℚ) (P_win_A : ℚ) (h1 : P_draw = 1/2) (h2 : P_win_A = 1/3) : 
  P_draw + P_win_A = 5/6 :=
by
  rw [h1, h2]
  norm_num

end probability_A_not_losing_l53_53560


namespace graduating_class_total_l53_53880

theorem graduating_class_total (boys girls : ℕ) 
  (h_boys : boys = 138)
  (h_more_girls : girls = boys + 69) :
  boys + girls = 345 :=
sorry

end graduating_class_total_l53_53880


namespace elena_meeting_percentage_l53_53795

noncomputable def workday_hours : ℕ := 10
noncomputable def first_meeting_duration_minutes : ℕ := 60
noncomputable def second_meeting_duration_minutes : ℕ := 3 * first_meeting_duration_minutes
noncomputable def total_workday_minutes := workday_hours * 60
noncomputable def total_meeting_minutes := first_meeting_duration_minutes + second_meeting_duration_minutes
noncomputable def percent_time_in_meetings := (total_meeting_minutes * 100) / total_workday_minutes

theorem elena_meeting_percentage : percent_time_in_meetings = 40 := by 
  sorry

end elena_meeting_percentage_l53_53795


namespace calculate_value_l53_53597

def f (x : ℝ) : ℝ := 9 - x
def g (x : ℝ) : ℝ := x - 9

theorem calculate_value : g (f 15) = -15 := by
  sorry

end calculate_value_l53_53597


namespace calculate_angle_C_l53_53478

variable (A B C : ℝ)

theorem calculate_angle_C (h1 : A = C - 40) (h2 : B = 2 * A) (h3 : A + B + C = 180) :
  C = 75 :=
by
  sorry

end calculate_angle_C_l53_53478


namespace no_sol_for_eq_xn_minus_yn_eq_2k_l53_53681

theorem no_sol_for_eq_xn_minus_yn_eq_2k (k n : ℕ) (h_pos_k : k > 0) (h_pos_n : n > 0) (h_n : n > 2) :
  ¬ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^n - y^n = 2^k := 
sorry

end no_sol_for_eq_xn_minus_yn_eq_2k_l53_53681


namespace pool_one_quarter_capacity_in_six_hours_l53_53792

theorem pool_one_quarter_capacity_in_six_hours (d : ℕ → ℕ) :
  (∀ n : ℕ, d (n + 1) = 2 * d n) → d 8 = 2^8 →
  d 6 = 2^6 :=
by
  intros h1 h2
  sorry

end pool_one_quarter_capacity_in_six_hours_l53_53792


namespace monopoly_favor_durable_machine_competitive_market_prefer_durable_l53_53785

-- Define the conditions
def consumer_valuation : ℕ := 10
def durable_cost : ℕ := 6

-- Define the monopoly decision problem: prove C > 3
theorem monopoly_favor_durable_machine (C : ℕ) : 
  consumer_valuation * 2 - durable_cost > 2 * (consumer_valuation - C) → C > 3 := 
by 
  sorry

-- Define the competitive market decision problem: prove C > 3
theorem competitive_market_prefer_durable (C : ℕ) :
  2 * consumer_valuation - durable_cost > 2 * (consumer_valuation - C) → C > 3 := 
by 
  sorry

end monopoly_favor_durable_machine_competitive_market_prefer_durable_l53_53785


namespace value_of_a_squared_plus_b_squared_l53_53184

variable (a b : ℝ)

theorem value_of_a_squared_plus_b_squared (h1 : a - b = 10) (h2 : a * b = 55) : a^2 + b^2 = 210 := 
by 
sorry

end value_of_a_squared_plus_b_squared_l53_53184


namespace gratuity_percentage_l53_53619

open Real

theorem gratuity_percentage (num_bankers num_clients : ℕ) (total_bill per_person_cost : ℝ) 
    (h1 : num_bankers = 4) (h2 : num_clients = 5) (h3 : total_bill = 756) 
    (h4 : per_person_cost = 70) : 
    ((total_bill - (num_bankers + num_clients) * per_person_cost) / 
     ((num_bankers + num_clients) * per_person_cost)) = 0.2 :=
by 
  sorry

end gratuity_percentage_l53_53619


namespace equation1_solution_equation2_solution_l53_53532

-- Equation 1: x^2 + 2x - 8 = 0 has solutions x = -4 and x = 2.
theorem equation1_solution (x : ℝ) : x^2 + 2 * x - 8 = 0 ↔ x = -4 ∨ x = 2 := by
  sorry

-- Equation 2: 2(x+3)^2 = x(x+3) has solutions x = -3 and x = -6.
theorem equation2_solution (x : ℝ) : 2 * (x + 3)^2 = x * (x + 3) ↔ x = -3 ∨ x = -6 := by
  sorry

end equation1_solution_equation2_solution_l53_53532


namespace range_of_a_l53_53753

variable {x a : ℝ}

def p (x : ℝ) := x^2 - 8 * x - 20 > 0
def q (a : ℝ) (x : ℝ) := x^2 - 2 * x + 1 - a^2 > 0

theorem range_of_a (h₀ : ∀ x, p x → q a x) (h₁ : a > 0) : 0 < a ∧ a ≤ 3 := 
by 
  sorry

end range_of_a_l53_53753


namespace cone_volume_l53_53527

theorem cone_volume (r h : ℝ) (π : ℝ) (V : ℝ) :
    r = 3 → h = 4 → π = Real.pi → V = (1/3) * π * r^2 * h → V = 37.68 :=
by
  sorry

end cone_volume_l53_53527


namespace possible_values_f_one_l53_53278

noncomputable def f (x : ℝ) : ℝ := sorry

variables (a b : ℝ)
axiom f_equation : ∀ x y : ℝ, 
  f ((x - y) ^ 2) = a * (f x)^2 - 2 * x * f y + b * y^2

theorem possible_values_f_one : f 1 = 1 ∨ f 1 = 2 :=
sorry

end possible_values_f_one_l53_53278


namespace math_problem_modulo_l53_53656

theorem math_problem_modulo :
    (245 * 15 - 20 * 8 + 5) % 17 = 1 := 
by
  sorry

end math_problem_modulo_l53_53656


namespace slope_of_line_l53_53726

theorem slope_of_line (x y : ℝ) (h : 2 * y = -3 * x + 6) : (∃ m b : ℝ, y = m * x + b) ∧  (m = -3 / 2) :=
by 
  sorry

end slope_of_line_l53_53726


namespace general_term_a_sum_of_bn_l53_53955

-- Define sequences a_n and b_n
noncomputable def a (n : ℕ) : ℕ := 2 * n + 1
noncomputable def b (n : ℕ) : ℚ := 1 / ((2 * n + 1) * (2 * n + 3))

-- Conditions
lemma condition_1 (n : ℕ) : a n > 0 := by sorry
lemma condition_2 (n : ℕ) : (a n)^2 + 2 * (a n) = 4 * (n * (n + 1)) + 3 := 
  by sorry

-- Theorem for question 1
theorem general_term_a (n : ℕ) : a n = 2 * n + 1 := by sorry

-- Theorem for question 2
theorem sum_of_bn (n : ℕ) : 
  (Finset.range n).sum b = (n : ℚ) / (6 * n + 9) := by sorry

end general_term_a_sum_of_bn_l53_53955


namespace problem_1_problem_2_l53_53461

variables (α : ℝ) (h : Real.tan α = 3)

theorem problem_1 : (Real.sin α + 3 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = 6 / 11 :=
by
  -- Proof is skipped
  sorry

theorem problem_2 : Real.sin α * Real.sin α + Real.sin α * Real.cos α + 3 * Real.cos α * Real.cos α = 3 / 2 :=
by
  -- Proof is skipped
  sorry

end problem_1_problem_2_l53_53461


namespace inequality_solution_l53_53242

theorem inequality_solution (x : ℝ) : 
  x^3 - 10 * x^2 + 28 * x > 0 ↔ (0 < x ∧ x < 4) ∨ (6 < x)
:= sorry

end inequality_solution_l53_53242


namespace student_factor_l53_53067

theorem student_factor (x : ℤ) : (121 * x - 138 = 104) → x = 2 :=
by
  intro h
  sorry

end student_factor_l53_53067


namespace number_of_pairs_l53_53441

theorem number_of_pairs : 
  (∃ (m n : ℤ), m + n = mn - 3) → ∃! (count : ℕ), count = 6 := by
  sorry

end number_of_pairs_l53_53441


namespace reasoning_is_wrong_l53_53130

-- Definitions of the conditions
def some_rationals_are_proper_fractions := ∃ q : ℚ, ∃ f : ℚ, q = f ∧ f.den ≠ 1
def integers_are_rationals := ∀ z : ℤ, ∃ q : ℚ, q = z

-- Proof that the form of reasoning is wrong given the conditions
theorem reasoning_is_wrong 
  (h₁ : some_rationals_are_proper_fractions) 
  (h₂ : integers_are_rationals) :
  ¬ (∀ z : ℤ, ∃ f : ℚ, z = f ∧ f.den ≠ 1) := 
sorry

end reasoning_is_wrong_l53_53130


namespace x_eq_3_minus_2t_and_y_eq_3t_plus_6_l53_53234

theorem x_eq_3_minus_2t_and_y_eq_3t_plus_6 (t : ℝ) (x : ℝ) (y : ℝ) : x = 3 - 2 * t → y = 3 * t + 6 → x = 0 → y = 10.5 :=
by
  sorry

end x_eq_3_minus_2t_and_y_eq_3t_plus_6_l53_53234


namespace plates_per_meal_l53_53469

theorem plates_per_meal 
  (people : ℕ) (meals_per_day : ℕ) (total_days : ℕ) (total_plates : ℕ) 
  (h_people : people = 6) 
  (h_meals : meals_per_day = 3) 
  (h_days : total_days = 4) 
  (h_plates : total_plates = 144) 
  : (total_plates / (people * meals_per_day * total_days)) = 2 := 
  sorry

end plates_per_meal_l53_53469


namespace geometric_sequence_a5_value_l53_53716

theorem geometric_sequence_a5_value :
  ∃ (a : ℕ → ℝ) (r : ℝ), (a 3)^2 - 4 * a 3 + 3 = 0 ∧ 
                         (a 7)^2 - 4 * a 7 + 3 = 0 ∧ 
                         (a 3) * (a 7) = 3 ∧ 
                         (a 3) + (a 7) = 4 ∧ 
                         a 5 = (a 3 * a 7).sqrt :=
sorry

end geometric_sequence_a5_value_l53_53716


namespace nickels_count_l53_53340

theorem nickels_count (original_nickels : ℕ) (additional_nickels : ℕ) 
                        (h₁ : original_nickels = 7) 
                        (h₂ : additional_nickels = 5) : 
    original_nickels + additional_nickels = 12 := 
by sorry

end nickels_count_l53_53340


namespace find_f_of_7_l53_53970

variable {f : ℝ → ℝ}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem find_f_of_7 (h1 : is_odd_function f)
                    (h2 : is_periodic_function f 4)
                    (h3 : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) :
  f 7 = -2 := 
by
  sorry

end find_f_of_7_l53_53970


namespace intersection_M_N_l53_53144

def M : Set ℝ := { x | (x - 1)^2 < 4 }
def N : Set ℝ := { -1, 0, 1, 2, 3 }

theorem intersection_M_N : M ∩ N = {0, 1, 2} := 
by
  sorry

end intersection_M_N_l53_53144


namespace Joan_balloons_l53_53733

variable (J : ℕ) -- Joan's blue balloons

theorem Joan_balloons (h : J + 41 = 81) : J = 40 :=
by
  sorry

end Joan_balloons_l53_53733


namespace length_of_EC_l53_53089

variable (AC : ℝ) (AB : ℝ) (CD : ℝ) (EC : ℝ)

def is_trapezoid (AB CD : ℝ) : Prop := AB = 3 * CD
def perimeter (AB CD AC : ℝ) : Prop := AB + CD + AC + (AC / 3) = 36

theorem length_of_EC
  (h1 : is_trapezoid AB CD)
  (h2 : AC = 18)
  (h3 : perimeter AB CD AC) :
  EC = 9 / 2 :=
  sorry

end length_of_EC_l53_53089


namespace blue_tiles_in_45th_row_l53_53194

theorem blue_tiles_in_45th_row :
  ∀ (n : ℕ), n = 45 → (∃ r b : ℕ, (r + b = 2 * n - 1) ∧ (r > b) ∧ (r - 1 = b)) → b = 44 :=
by
  -- Skipping the proof with sorry to adhere to instruction
  sorry

end blue_tiles_in_45th_row_l53_53194


namespace lcm_of_8_9_5_10_l53_53115

theorem lcm_of_8_9_5_10 : Nat.lcm (Nat.lcm 8 9) (Nat.lcm 5 10) = 360 := by
  sorry

end lcm_of_8_9_5_10_l53_53115


namespace calculate_sum_l53_53895

theorem calculate_sum : (2 / 20) + (3 / 50 * 5 / 100) + (4 / 1000) + (6 / 10000) = 0.1076 := 
by
  sorry

end calculate_sum_l53_53895


namespace arithmetic_geometric_sequence_k4_l53_53304

theorem arithmetic_geometric_sequence_k4 (a : ℕ → ℝ) (d : ℝ) (h_d_ne_zero : d ≠ 0)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_geo_seq : ∃ k : ℕ → ℕ, k 0 = 1 ∧ k 1 = 2 ∧ k 2 = 6 ∧ ∀ i, a (k i + 1) / a (k i) = a (k i + 2) / a (k i + 1)) :
  ∃ k4 : ℕ, k4 = 22 := 
by
  sorry

end arithmetic_geometric_sequence_k4_l53_53304


namespace find_B_max_f_A_l53_53468

namespace ProofProblem

-- Definitions
variables {A B C a b c : ℝ} -- Angles and sides in the triangle
noncomputable def givenCondition (A B C a b c : ℝ) : Prop :=
  2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x * Real.sin (x + Real.pi / 3) - Real.sqrt 3 / 4

-- Problem Statements (to be proved)
theorem find_B (h : givenCondition A B C a b c) : B = Real.pi / 6 := sorry

theorem max_f_A (A : ℝ) (B : ℝ) (h1 : 0 < A) (h2 : A < 5 * Real.pi / 6) (h3 : B = Real.pi / 6) : (∃ (x : ℝ), f x = 1 / 2) := sorry

end ProofProblem

end find_B_max_f_A_l53_53468


namespace sufficient_but_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3_sufficient_but_not_necessary_0_lt_x_lt_5_for_abs_x_minus_2_lt_3_l53_53382

variable (x : ℝ)

theorem sufficient_but_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3 :
        (0 < x ∧ x < 5) → (|x - 2| < 3) :=
by sorry

theorem not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3 :
        (|x - 2| < 3) → (0 < x ∧ x < 5) :=
by sorry

theorem sufficient_but_not_necessary_0_lt_x_lt_5_for_abs_x_minus_2_lt_3 :
        (0 < x ∧ x < 5) ↔ (|x - 2| < 3) → false :=
by sorry

end sufficient_but_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3_sufficient_but_not_necessary_0_lt_x_lt_5_for_abs_x_minus_2_lt_3_l53_53382


namespace election_result_l53_53167

theorem election_result (Vx Vy Vz : ℝ) (Pz : ℝ)
  (h1 : Vx = 3 * (Vx / 3)) (h2 : Vy = 2 * (Vy / 2)) (h3 : Vz = 1 * (Vz / 1))
  (h4 : 0.63 * (Vx + Vy + Vz) = 0.74 * Vx + 0.67 * Vy + Pz * Vz) :
  Pz = 0.22 :=
by
  -- proof steps would go here
  -- sorry to keep the proof incomplete
  sorry

end election_result_l53_53167


namespace remainder_of_sum_l53_53021

theorem remainder_of_sum (h1 : 9375 % 5 = 0) (h2 : 9376 % 5 = 1) (h3 : 9377 % 5 = 2) (h4 : 9378 % 5 = 3) :
  (9375 + 9376 + 9377 + 9378) % 5 = 1 :=
by
  sorry

end remainder_of_sum_l53_53021


namespace watermelons_with_seeds_l53_53424

def ripe_watermelons : ℕ := 11
def unripe_watermelons : ℕ := 13
def seedless_watermelons : ℕ := 15
def total_watermelons := ripe_watermelons + unripe_watermelons

theorem watermelons_with_seeds :
  total_watermelons - seedless_watermelons = 9 :=
by
  sorry

end watermelons_with_seeds_l53_53424


namespace dilation_image_l53_53991

open Complex

theorem dilation_image (z₀ : ℂ) (c : ℂ) (k : ℝ) (z : ℂ)
    (h₀ : z₀ = 0 - 2*I) (h₁ : c = 1 + 2*I) (h₂ : k = 2) :
    z = -1 - 6*I :=
by
  sorry

end dilation_image_l53_53991


namespace calculate_total_cost_l53_53173

noncomputable def sandwich_cost : ℕ := 4
noncomputable def soda_cost : ℕ := 3
noncomputable def num_sandwiches : ℕ := 7
noncomputable def num_sodas : ℕ := 8
noncomputable def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem calculate_total_cost : total_cost = 52 := by
  sorry

end calculate_total_cost_l53_53173


namespace gel_pen_ratio_l53_53832

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end gel_pen_ratio_l53_53832


namespace find_ordered_pairs_l53_53137

theorem find_ordered_pairs (a b : ℕ) (h1 : 2 * a + 1 ∣ 3 * b - 1) (h2 : 2 * b + 1 ∣ 3 * a - 1) : 
  (a = 2 ∧ b = 2) ∨ (a = 12 ∧ b = 17) ∨ (a = 17 ∧ b = 12) :=
by {
  sorry -- proof omitted
}

end find_ordered_pairs_l53_53137


namespace prime_solution_l53_53646

theorem prime_solution (p : ℕ) (hp : Nat.Prime p) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ (p = 2 ∨ p = 3 ∨ p = 7) :=
by
  sorry

end prime_solution_l53_53646


namespace non_adjacent_divisibility_l53_53591

theorem non_adjacent_divisibility (a : Fin 7 → ℕ) (h : ∀ i, a i ∣ a ((i + 1) % 7) ∨ a ((i + 1) % 7) ∣ a i) :
  ∃ i j : Fin 7, i ≠ j ∧ (¬(i + 1)%7 = j) ∧ (a i ∣ a j ∨ a j ∣ a i) :=
by
  sorry

end non_adjacent_divisibility_l53_53591


namespace find_base_l53_53119

theorem find_base (b : ℕ) (h : (3 * b + 2) ^ 2 = b ^ 3 + b + 4) : b = 8 :=
sorry

end find_base_l53_53119


namespace person_birth_year_and_age_l53_53934

theorem person_birth_year_and_age (x y: ℕ) (h1: x ≤ 9) (h2: y ≤ 9) (hy: y = (88 - 10 * x) / (x + 1)):
  1988 - (1900 + 10 * x + y) = x * y → 1900 + 10 * x + y = 1964 ∧ 1988 - (1900 + 10 * x + y) = 24 :=
by
  sorry

end person_birth_year_and_age_l53_53934


namespace cos_neg245_l53_53050

-- Define the given condition and declare the theorem to prove the required equality
variable (a : ℝ)
def cos_25_eq_a : Prop := (Real.cos 25 * Real.pi / 180 = a)

theorem cos_neg245 :
  cos_25_eq_a a → Real.cos (-245 * Real.pi / 180) = -Real.sqrt (1 - a^2) :=
by
  intro h
  sorry

end cos_neg245_l53_53050


namespace find_tangency_segments_equal_l53_53109

-- Conditions of the problem as a theorem statement
theorem find_tangency_segments_equal (AB BC CD DA : ℝ) (x y : ℝ)
    (h1 : AB = 80)
    (h2 : BC = 140)
    (h3 : CD = 100)
    (h4 : DA = 120)
    (h5 : x + y = CD)
    (tangency_property : |x - y| = 0) :
  |x - y| = 0 :=
sorry

end find_tangency_segments_equal_l53_53109


namespace find_initial_investment_l53_53355

-- Define the necessary parameters for the problem
variables (P r : ℝ)

-- Given conditions
def condition1 : Prop := P * (1 + r * 3) = 240
def condition2 : Prop := 150 * (1 + r * 6) = 210

-- The statement to be proved
theorem find_initial_investment (h1 : condition1 P r) (h2 : condition2 r) : P = 200 :=
sorry

end find_initial_investment_l53_53355


namespace arithmetic_sequence_subtract_l53_53178

theorem arithmetic_sequence_subtract (a : ℕ → ℝ) (d : ℝ) :
  (a 4 + a 6 + a 8 + a 10 + a 12 = 120) →
  (a 9 - (1 / 3) * a 11 = 16) :=
by
  sorry

end arithmetic_sequence_subtract_l53_53178


namespace annual_interest_rate_is_6_percent_l53_53221

-- Definitions from the conditions
def principal : ℕ := 150
def total_amount_paid : ℕ := 159
def interest := total_amount_paid - principal
def interest_rate := (interest * 100) / principal

-- The theorem to prove
theorem annual_interest_rate_is_6_percent :
  interest_rate = 6 := by sorry

end annual_interest_rate_is_6_percent_l53_53221


namespace folding_positions_l53_53164

theorem folding_positions (positions : Finset ℕ) (h_conditions: positions = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}) : 
  ∃ valid_positions : Finset ℕ, valid_positions = {1, 2, 3, 4, 9, 10, 11, 12} ∧ valid_positions.card = 8 :=
by
  sorry

end folding_positions_l53_53164


namespace journey_total_distance_l53_53874

-- Define the conditions
def miles_already_driven : ℕ := 642
def miles_to_drive : ℕ := 558

-- The total distance of the journey
def total_distance : ℕ := miles_already_driven + miles_to_drive

-- Prove that the total distance of the journey equals 1200 miles
theorem journey_total_distance : total_distance = 1200 := 
by
  -- here the proof would go
  sorry

end journey_total_distance_l53_53874


namespace ivar_total_water_needed_l53_53044

-- Define the initial number of horses
def initial_horses : ℕ := 3

-- Define the added horses
def added_horses : ℕ := 5

-- Define the total number of horses
def total_horses : ℕ := initial_horses + added_horses

-- Define water consumption per horse per day for drinking
def water_consumption_drinking : ℕ := 5

-- Define water consumption per horse per day for bathing
def water_consumption_bathing : ℕ := 2

-- Define total water consumption per horse per day
def total_water_consumption_per_horse_per_day : ℕ := 
    water_consumption_drinking + water_consumption_bathing

-- Define total daily water consumption for all horses
def daily_water_consumption_all_horses : ℕ := 
    total_horses * total_water_consumption_per_horse_per_day

-- Define total water consumption over 28 days
def total_water_consumption_28_days : ℕ := 
    daily_water_consumption_all_horses * 28

-- State the theorem
theorem ivar_total_water_needed : 
    total_water_consumption_28_days = 1568 := 
by
  sorry

end ivar_total_water_needed_l53_53044


namespace solution_inequality_set_l53_53015

-- Define the inequality condition
def inequality (x : ℝ) : Prop := x^2 - 3 * x - 10 ≤ 0

-- Define the interval solution set
def solution_set := Set.Icc (-2 : ℝ) 5

-- The statement that we want to prove
theorem solution_inequality_set : {x : ℝ | inequality x} = solution_set :=
  sorry

end solution_inequality_set_l53_53015


namespace incorrect_calculation_l53_53557

theorem incorrect_calculation (a : ℝ) : (2 * a) ^ 3 ≠ 6 * a ^ 3 :=
by {
  sorry
}

end incorrect_calculation_l53_53557


namespace slope_condition_l53_53489

theorem slope_condition {m : ℝ} : 
  (4 - m) / (m + 2) = 1 → m = 1 :=
by
  sorry

end slope_condition_l53_53489


namespace cube_root_floor_equality_l53_53659

theorem cube_root_floor_equality (n : ℕ) : 
  (⌊(n : ℝ)^(1/3) + (n+1 : ℝ)^(1/3)⌋ : ℝ) = ⌊(8*n + 3 : ℝ)^(1/3)⌋ :=
sorry

end cube_root_floor_equality_l53_53659


namespace train_passing_through_tunnel_l53_53550

theorem train_passing_through_tunnel :
  let train_length : ℝ := 300
  let tunnel_length : ℝ := 1200
  let speed_in_kmh : ℝ := 54
  let speed_in_mps : ℝ := speed_in_kmh * (1000 / 3600)
  let total_distance : ℝ := train_length + tunnel_length
  let time : ℝ := total_distance / speed_in_mps
  time = 100 :=
by
  sorry

end train_passing_through_tunnel_l53_53550


namespace charlie_ride_distance_l53_53723

-- Define the known values
def oscar_ride : ℝ := 0.75
def difference : ℝ := 0.5

-- Define Charlie's bus ride distance
def charlie_ride : ℝ := oscar_ride - difference

-- The theorem to be proven
theorem charlie_ride_distance : charlie_ride = 0.25 := 
by sorry

end charlie_ride_distance_l53_53723


namespace value_of_8x_minus_5_squared_l53_53496

theorem value_of_8x_minus_5_squared (x : ℝ) (h : 8 * x ^ 2 + 7 = 12 * x + 17) : (8 * x - 5) ^ 2 = 465 := 
sorry

end value_of_8x_minus_5_squared_l53_53496


namespace find_c_l53_53952

open Function

noncomputable def g (x : ℝ) : ℝ :=
  (x - 4) * (x - 2) * x * (x + 2) * (x + 4) / 255 - 5

theorem find_c (c : ℤ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, g x₁ = c ∧ g x₂ = c ∧ g x₃ = c ∧ g x₄ = c ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) →
  ∀ k : ℤ, k < c → ¬ ∃ x₁ x₂ x₃ x₄ : ℝ, g x₁ = k ∧ g x₂ = k ∧ g x₃ = k ∧ g x₄ = k ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ :=
sorry

end find_c_l53_53952


namespace arithmetic_sequence_value_l53_53518

theorem arithmetic_sequence_value (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)) -- definition of arithmetic sequence
  (h2 : a 2 + a 10 = -12) -- given that a_2 + a_{10} = -12
  (h3 : a_2 = -6) -- given that a_6 is the average of a_2 and a_{10}
  : a 6 = -6 :=
sorry

end arithmetic_sequence_value_l53_53518


namespace integral_evaluation_l53_53697

noncomputable def definite_integral (a b : ℝ) (f : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, f x

theorem integral_evaluation : 
  definite_integral 1 2 (fun x => 1 / x + x) = Real.log 2 + 3 / 2 :=
  sorry

end integral_evaluation_l53_53697


namespace maximum_area_l53_53005

variable {l w : ℝ}

theorem maximum_area (h1 : l + w = 200) (h2 : l ≥ 90) (h3 : w ≥ 50) (h4 : l ≤ 2 * w) : l * w ≤ 10000 :=
sorry

end maximum_area_l53_53005


namespace probability_of_picking_letter_in_mathematics_l53_53586

def unique_letters_in_mathematics : List Char := ['M', 'A', 'T', 'H', 'E', 'I', 'C', 'S']

def number_of_unique_letters_in_word : ℕ := unique_letters_in_mathematics.length

def total_letters_in_alphabet : ℕ := 26

theorem probability_of_picking_letter_in_mathematics :
  (number_of_unique_letters_in_word : ℚ) / total_letters_in_alphabet = 4 / 13 :=
by
  sorry

end probability_of_picking_letter_in_mathematics_l53_53586


namespace Brians_trip_distance_l53_53974

theorem Brians_trip_distance (miles_per_gallon : ℕ) (gallons_used : ℕ) (distance_traveled : ℕ) 
  (h1 : miles_per_gallon = 20) (h2 : gallons_used = 3) : 
  distance_traveled = 60 :=
by
  sorry

end Brians_trip_distance_l53_53974


namespace opposite_of_neg2023_l53_53272

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l53_53272


namespace p_sufficient_not_necessary_l53_53068

theorem p_sufficient_not_necessary:
  (∀ a b : ℝ, a > b ∧ b > 0 → (1 / a^2 < 1 / b^2)) ∧ 
  (∃ a b : ℝ, (1 / a^2 < 1 / b^2) ∧ ¬ (a > b ∧ b > 0)) :=
sorry

end p_sufficient_not_necessary_l53_53068


namespace opposite_of_2023_l53_53417

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l53_53417


namespace john_saves_water_l53_53530

-- Define the conditions
def old_water_per_flush : ℕ := 5
def num_flushes_per_day : ℕ := 15
def reduction_percentage : ℕ := 80
def days_in_june : ℕ := 30

-- Define the savings calculation
def water_saved_in_june : ℕ :=
  let old_daily_usage := old_water_per_flush * num_flushes_per_day
  let old_june_usage := old_daily_usage * days_in_june
  let new_water_per_flush := old_water_per_flush * (100 - reduction_percentage) / 100
  let new_daily_usage := new_water_per_flush * num_flushes_per_day
  let new_june_usage := new_daily_usage * days_in_june
  old_june_usage - new_june_usage

-- The proof problem statement
theorem john_saves_water : water_saved_in_june = 1800 := 
by
  -- Proof would go here
  sorry

end john_saves_water_l53_53530


namespace payment_to_Y_is_227_27_l53_53016

-- Define the conditions
def total_payment_per_week (x y : ℝ) : Prop :=
  x + y = 500

def x_payment_is_120_percent_of_y (x y : ℝ) : Prop :=
  x = 1.2 * y

-- Formulate the problem as a theorem to be proven
theorem payment_to_Y_is_227_27 (Y : ℝ) (X : ℝ) 
  (h1 : total_payment_per_week X Y) 
  (h2 : x_payment_is_120_percent_of_y X Y) : 
  Y = 227.27 :=
by
  sorry

end payment_to_Y_is_227_27_l53_53016


namespace problem_statement_l53_53342

variable (f : ℝ → ℝ) 

def prop1 (f : ℝ → ℝ) : Prop := ∃T > 0, T ≠ 3 / 2 ∧ ∀ x, f (x + T) = f x
def prop2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 3 / 4) = f (-x + 3 / 4)
def prop3 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def prop4 (f : ℝ → ℝ) : Prop := Monotone f

theorem problem_statement (h₁ : ∀ x, f (x + 3 / 2) = -f x)
                          (h₂ : ∀ x, f (x - 3 / 4) = -f (-x - 3 / 4)) : 
                          (¬prop1 f) ∧ (prop2 f) ∧ (prop3 f) ∧ (¬prop4 f) :=
by
  sorry

end problem_statement_l53_53342


namespace total_dogs_at_center_l53_53632

structure PawsitiveTrainingCenter :=
  (sit : Nat)
  (stay : Nat)
  (fetch : Nat)
  (roll_over : Nat)
  (sit_stay : Nat)
  (sit_fetch : Nat)
  (sit_roll_over : Nat)
  (stay_fetch : Nat)
  (stay_roll_over : Nat)
  (fetch_roll_over : Nat)
  (sit_stay_fetch : Nat)
  (sit_stay_roll_over : Nat)
  (sit_fetch_roll_over : Nat)
  (stay_fetch_roll_over : Nat)
  (all_four : Nat)
  (none : Nat)

def PawsitiveTrainingCenter.total_dogs (p : PawsitiveTrainingCenter) : Nat :=
  p.sit + p.stay + p.fetch + p.roll_over
  - p.sit_stay - p.sit_fetch - p.sit_roll_over - p.stay_fetch - p.stay_roll_over - p.fetch_roll_over
  + p.sit_stay_fetch + p.sit_stay_roll_over + p.sit_fetch_roll_over + p.stay_fetch_roll_over
  - p.all_four + p.none

theorem total_dogs_at_center (p : PawsitiveTrainingCenter) (h : 
  p.sit = 60 ∧
  p.stay = 35 ∧
  p.fetch = 45 ∧
  p.roll_over = 40 ∧
  p.sit_stay = 20 ∧
  p.sit_fetch = 15 ∧
  p.sit_roll_over = 10 ∧
  p.stay_fetch = 5 ∧
  p.stay_roll_over = 8 ∧
  p.fetch_roll_over = 6 ∧
  p.sit_stay_fetch = 4 ∧
  p.sit_stay_roll_over = 3 ∧
  p.sit_fetch_roll_over = 2 ∧
  p.stay_fetch_roll_over = 1 ∧
  p.all_four = 2 ∧
  p.none = 12
) : PawsitiveTrainingCenter.total_dogs p = 135 := by
  sorry

end total_dogs_at_center_l53_53632


namespace FGH_supermarkets_total_l53_53564

theorem FGH_supermarkets_total 
  (us_supermarkets : ℕ)
  (ca_supermarkets : ℕ)
  (h1 : us_supermarkets = 41)
  (h2 : us_supermarkets = ca_supermarkets + 22) :
  us_supermarkets + ca_supermarkets = 60 :=
by
  sorry

end FGH_supermarkets_total_l53_53564


namespace lydia_candy_problem_l53_53845

theorem lydia_candy_problem :
  ∃ m: ℕ, (∀ k: ℕ, (k * 24 = Nat.lcm (Nat.lcm 16 18) 20) → k ≥ m) ∧ 24 * m = Nat.lcm (Nat.lcm 16 18) 20 ∧ m = 30 :=
by
  sorry

end lydia_candy_problem_l53_53845


namespace largest_positive_integer_l53_53413

def binary_op (n : ℕ) : ℤ := n - (n * 5)

theorem largest_positive_integer (n : ℕ) (h : binary_op n < 21) : n ≤ 1 := 
sorry

end largest_positive_integer_l53_53413


namespace Isabella_paint_area_l53_53665

def bedroom1_length : ℕ := 14
def bedroom1_width : ℕ := 11
def bedroom1_height : ℕ := 9

def bedroom2_length : ℕ := 13
def bedroom2_width : ℕ := 12
def bedroom2_height : ℕ := 9

def unpaintable_area_per_bedroom : ℕ := 70

theorem Isabella_paint_area :
  let wall_area (length width height : ℕ) := 2 * (length * height) + 2 * (width * height)
  let paintable_area (length width height : ℕ) := wall_area length width height - unpaintable_area_per_bedroom
  paintable_area bedroom1_length bedroom1_width bedroom1_height +
  paintable_area bedroom1_length bedroom1_width bedroom1_height +
  paintable_area bedroom2_length bedroom2_width bedroom2_height +
  paintable_area bedroom2_length bedroom2_width bedroom2_height =
  1520 := 
by
  sorry

end Isabella_paint_area_l53_53665


namespace find_n_l53_53371

open Classical

theorem find_n (n : ℕ) (h : (8 * Nat.choose n 3) = 8 * (2 * Nat.choose n 1)) : n = 5 := by
  sorry

end find_n_l53_53371


namespace column_of_2023_l53_53182

theorem column_of_2023 : 
  let columns := ["G", "H", "I", "J", "K", "L", "M"]
  let pattern := ["H", "I", "J", "K", "L", "M", "L", "K", "J", "I", "H", "G"]
  let n := 2023
  (pattern.get! ((n - 2) % 12)) = "I" :=
by
  -- Sorry is a placeholder for the proof
  sorry

end column_of_2023_l53_53182


namespace factor_example_solve_equation_example_l53_53664

-- Factorization proof problem
theorem factor_example (m a b : ℝ) : 
  (m * a ^ 2 - 4 * m * b ^ 2) = m * (a + 2 * b) * (a - 2 * b) :=
sorry

-- Solving the equation proof problem
theorem solve_equation_example (x : ℝ) (hx1: x ≠ 2) (hx2: x ≠ 0) : 
  (1 / (x - 2) = 3 / x) ↔ x = 3 :=
sorry

end factor_example_solve_equation_example_l53_53664


namespace range_of_m_l53_53818

theorem range_of_m (a m x : ℝ) (p q : Prop) :
  (p ↔ ∃ (a : ℝ) (m : ℝ), ∀ (x : ℝ), 4 * x^2 - 2 * a * x + 2 * a + 5 = 0) →
  (q ↔ 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0) →
  (¬ p → ¬ q) →
  (∀ a, -2 ≤ a ∧ a ≤ 10) →
  (1 - m ≤ -2) ∧ (1 + m ≥ 10) →
  m ≥ 9 :=
by
  intros hp hq npnq ha hm
  sorry  -- Proof omitted

end range_of_m_l53_53818


namespace requiredSheetsOfPaper_l53_53479

-- Define the conditions
def englishAlphabetLetters : ℕ := 26
def timesWrittenPerLetter : ℕ := 3
def sheetsOfPaperPerLetter (letters : ℕ) (times : ℕ) : ℕ := letters * times

-- State the theorem equivalent to the original math problem
theorem requiredSheetsOfPaper : sheetsOfPaperPerLetter englishAlphabetLetters timesWrittenPerLetter = 78 := by
  sorry

end requiredSheetsOfPaper_l53_53479


namespace find_length_of_sheet_l53_53651

noncomputable section

-- Axioms regarding the conditions
def width_of_sheet : ℝ := 36       -- The width of the metallic sheet is 36 meters
def side_of_square : ℝ := 7        -- The side length of the square cut off from each corner is 7 meters
def volume_of_box : ℝ := 5236      -- The volume of the resulting box is 5236 cubic meters

-- Define the length of the metallic sheet as L
def length_of_sheet (L : ℝ) : Prop :=
  let new_length := L - 2 * side_of_square
  let new_width := width_of_sheet - 2 * side_of_square
  let height := side_of_square
  volume_of_box = new_length * new_width * height

-- The condition to prove
theorem find_length_of_sheet : ∃ L : ℝ, length_of_sheet L ∧ L = 48 :=
by
  sorry

end find_length_of_sheet_l53_53651


namespace parallel_condition_sufficient_not_necessary_l53_53533

noncomputable def a (x : ℝ) : ℝ × ℝ := (1, x - 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x + 1, 3)

theorem parallel_condition_sufficient_not_necessary (x : ℝ) :
  (x = 2) → (a x = b x) ∨ (a (-2) = b (-2)) :=
by sorry

end parallel_condition_sufficient_not_necessary_l53_53533


namespace min_value_l53_53047

theorem min_value : ∀ (a b : ℝ), a + b^2 = 2 → (∀ x y : ℝ, x = a^2 + 6 * y^2 → y = b) → (∃ c : ℝ, c = 3) :=
by
  intros a b h₁ h₂
  sorry

end min_value_l53_53047


namespace johns_total_earnings_per_week_l53_53431

def small_crab_baskets_monday := 3
def medium_crab_baskets_monday := 2
def large_crab_baskets_thursday := 4
def jumbo_crab_baskets_thursday := 1

def crabs_per_small_basket := 4
def crabs_per_medium_basket := 3
def crabs_per_large_basket := 5
def crabs_per_jumbo_basket := 2

def price_per_small_crab := 3
def price_per_medium_crab := 4
def price_per_large_crab := 5
def price_per_jumbo_crab := 7

def total_weekly_earnings :=
  (small_crab_baskets_monday * crabs_per_small_basket * price_per_small_crab) +
  (medium_crab_baskets_monday * crabs_per_medium_basket * price_per_medium_crab) +
  (large_crab_baskets_thursday * crabs_per_large_basket * price_per_large_crab) +
  (jumbo_crab_baskets_thursday * crabs_per_jumbo_basket * price_per_jumbo_crab)

theorem johns_total_earnings_per_week : total_weekly_earnings = 174 :=
by sorry

end johns_total_earnings_per_week_l53_53431


namespace predicted_yield_of_rice_l53_53072

theorem predicted_yield_of_rice (x : ℝ) (h : x = 80) : 5 * x + 250 = 650 :=
by {
  sorry -- proof will be given later
}

end predicted_yield_of_rice_l53_53072


namespace area_of_rhombus_perimeter_of_rhombus_l53_53290

-- Definitions and conditions for the area of the rhombus
def d1 : ℕ := 18
def d2 : ℕ := 16

-- Definition for the side length of the rhombus
def side_length : ℕ := 10

-- Statement for the area of the rhombus
theorem area_of_rhombus : (d1 * d2) / 2 = 144 := by
  sorry

-- Statement for the perimeter of the rhombus
theorem perimeter_of_rhombus : 4 * side_length = 40 := by
  sorry

end area_of_rhombus_perimeter_of_rhombus_l53_53290


namespace number_of_dimes_l53_53267

theorem number_of_dimes (k : ℕ) (dimes quarters : ℕ) (value : ℕ)
  (h1 : 3 * k = dimes)
  (h2 : 2 * k = quarters)
  (h3 : value = (10 * dimes) + (25 * quarters))
  (h4 : value = 400) :
  dimes = 15 :=
by {
  sorry
}

end number_of_dimes_l53_53267


namespace sum_first_n_terms_of_arithmetic_sequence_l53_53042

def arithmetic_sequence_sum (a1 d n: ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem sum_first_n_terms_of_arithmetic_sequence :
  arithmetic_sequence_sum 2 2 n = n * (n + 1) / 2 :=
by sorry

end sum_first_n_terms_of_arithmetic_sequence_l53_53042


namespace sum_on_simple_interest_is_1750_l53_53279

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r)^t - P

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

theorem sum_on_simple_interest_is_1750 :
  let P_ci := 4000
  let r_ci := 0.10
  let t_ci := 2
  let r_si := 0.08
  let t_si := 3
  let CI := compound_interest P_ci r_ci t_ci
  let SI := CI / 2
  let P_si := SI / (r_si * t_si)
  P_si = 1750 :=
by
  sorry

end sum_on_simple_interest_is_1750_l53_53279


namespace average_water_per_day_l53_53133

variable (day1 : ℕ)
variable (day2 : ℕ)
variable (day3 : ℕ)

def total_water_over_three_days (d1 d2 d3 : ℕ) := d1 + d2 + d3

theorem average_water_per_day :
  day1 = 215 ->
  day2 = 215 + 76 ->
  day3 = 291 - 53 ->
  (total_water_over_three_days day1 day2 day3) / 3 = 248 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end average_water_per_day_l53_53133


namespace find_total_buffaloes_l53_53770

-- Define the problem parameters.
def number_of_ducks : ℕ := sorry
def number_of_cows : ℕ := 8

-- Define the conditions.
def duck_legs : ℕ := 2 * number_of_ducks
def cow_legs : ℕ := 4 * number_of_cows
def total_heads : ℕ := number_of_ducks + number_of_cows

-- The given equation as a condition.
def total_legs : ℕ := duck_legs + cow_legs

-- Translate condition from the problem:
def condition : Prop := total_legs = 2 * total_heads + 16

-- The proof statement.
theorem find_total_buffaloes : number_of_cows = 8 :=
by
  -- Place the placeholder proof here.
  sorry

end find_total_buffaloes_l53_53770


namespace distribution_value_l53_53437

def standard_deviation := 2
def mean := 51

theorem distribution_value (x : ℝ) (hx : x < 45) : (mean - 3 * standard_deviation) > x :=
by
  -- Provide the statement without proof
  sorry

end distribution_value_l53_53437


namespace find_a_l53_53449

noncomputable def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem find_a (a : ℝ) (h : ∃ x, f x a = 3) : a = 1 ∨ a = 7 := 
sorry

end find_a_l53_53449


namespace cylinder_volume_ratio_l53_53893

theorem cylinder_volume_ratio (s : ℝ) :
  let r := s / 2
  let h := s
  let V_cylinder := π * r^2 * h
  let V_cube := s^3
  V_cylinder / V_cube = π / 4 :=
by
  sorry

end cylinder_volume_ratio_l53_53893


namespace apples_minimum_count_l53_53358

theorem apples_minimum_count :
  ∃ n : ℕ, n ≡ 2 [MOD 3] ∧ n ≡ 2 [MOD 4] ∧ n ≡ 2 [MOD 5] ∧ n = 62 := by
sorry

end apples_minimum_count_l53_53358


namespace shaded_square_percentage_l53_53669

theorem shaded_square_percentage (total_squares shaded_squares : ℕ) (h_total: total_squares = 25) (h_shaded: shaded_squares = 13) : 
(shaded_squares * 100) / total_squares = 52 := 
by
  sorry

end shaded_square_percentage_l53_53669


namespace car_speed_first_hour_l53_53004

theorem car_speed_first_hour (speed1 speed2 avg_speed : ℕ) (h1 : speed2 = 70) (h2 : avg_speed = 95) :
  (2 * avg_speed) = speed1 + speed2 → speed1 = 120 :=
by
  sorry

end car_speed_first_hour_l53_53004


namespace exists_prime_among_15_numbers_l53_53392

theorem exists_prime_among_15_numbers 
    (integers : Fin 15 → ℕ)
    (h1 : ∀ i, 1 < integers i)
    (h2 : ∀ i, integers i < 1998)
    (h3 : ∀ i j, i ≠ j → Nat.gcd (integers i) (integers j) = 1) :
    ∃ i, Nat.Prime (integers i) :=
by
  sorry

end exists_prime_among_15_numbers_l53_53392


namespace age_group_caloric_allowance_l53_53143

theorem age_group_caloric_allowance
  (average_daily_allowance : ℕ)
  (daily_reduction : ℕ)
  (reduced_weekly_allowance : ℕ)
  (week_days : ℕ)
  (h1 : daily_reduction = 500)
  (h2 : week_days = 7)
  (h3 : reduced_weekly_allowance = 10500)
  (h4 : reduced_weekly_allowance = (average_daily_allowance - daily_reduction) * week_days) :
  average_daily_allowance = 2000 :=
sorry

end age_group_caloric_allowance_l53_53143


namespace arithmetic_sequence_50th_term_l53_53217

theorem arithmetic_sequence_50th_term :
  let a_1 := 3
  let d := 5
  let n := 50
  let a_n := a_1 + (n - 1) * d
  a_n = 248 :=
by
  let a_1 := 3
  let d := 5
  let n := 50
  let a_n := a_1 + (n - 1) * d
  sorry

end arithmetic_sequence_50th_term_l53_53217


namespace keith_picked_0_pears_l53_53463

structure Conditions where
  apples_total : ℕ
  apples_mike : ℕ
  apples_nancy : ℕ
  apples_keith : ℕ
  pears_keith : ℕ

theorem keith_picked_0_pears (c : Conditions) (h_total : c.apples_total = 16)
 (h_mike : c.apples_mike = 7) (h_nancy : c.apples_nancy = 3)
 (h_keith : c.apples_keith = 6) : c.pears_keith = 0 :=
by
  sorry

end keith_picked_0_pears_l53_53463


namespace total_amount_raised_l53_53228

-- Definitions based on conditions
def PancakeCost : ℕ := 4
def BaconCost : ℕ := 2
def NumPancakesSold : ℕ := 60
def NumBaconSold : ℕ := 90

-- Lean statement proving that the total amount raised is $420
theorem total_amount_raised : (NumPancakesSold * PancakeCost) + (NumBaconSold * BaconCost) = 420 := by
  -- Since we are not required to prove, we use sorry here
  sorry

end total_amount_raised_l53_53228


namespace initial_beavers_l53_53915

theorem initial_beavers (B C : ℕ) (h1 : C = 40) (h2 : B + C + 2 * B + (C - 10) = 130) : B = 20 :=
by
  sorry

end initial_beavers_l53_53915


namespace perpendicular_lines_a_value_l53_53259

theorem perpendicular_lines_a_value (a : ℝ) :
  (∃ m1 m2 : ℝ, (m1 = -a / 2 ∧ m2 = -1 / (a * (a + 1)) ∧ m1 * m2 = -1) ∨
   (a = 0 ∧ ax + 2 * y + 6 = 0 ∧ x + a * (a + 1) * y + (a^2 - 1) = 0)) →
  (a = -3 / 2 ∨ a = 0) :=
by
  sorry

end perpendicular_lines_a_value_l53_53259


namespace tan_alpha_minus_pi_over_4_l53_53393

variable (α β : ℝ)

-- Given conditions
axiom h1 : Real.tan (α + β) = 2 / 5
axiom h2 : Real.tan β = 1 / 3

-- The goal to prove
theorem tan_alpha_minus_pi_over_4: 
  Real.tan (α - π / 4) = -8 / 9 := by
  sorry

end tan_alpha_minus_pi_over_4_l53_53393


namespace find_point_coordinates_l53_53684

open Real

-- Define circles C1 and C2
def circle_C1 (x y : ℝ) : Prop := (x + 4)^2 + (y - 2)^2 = 9
def circle_C2 (x y : ℝ) : Prop := (x - 5)^2 + (y - 6)^2 = 9

-- Define mutually perpendicular lines passing through point P
def line_l1 (P : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop := y - P.2 = k * (x - P.1)
def line_l2 (P : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop := y - P.2 = -1/k * (x - P.1)

-- Define the condition that chord lengths intercepted by lines on respective circles are equal
def equal_chord_lengths (P : ℝ × ℝ) (k : ℝ) : Prop :=
  abs (-4 * k - 2 + P.2 - k * P.1) / sqrt ((k^2) + 1) = abs (5 + 6 * k - k * P.2 - P.1) / sqrt ((k^2) + 1)

-- Main statement to be proved
theorem find_point_coordinates :
  ∃ (P : ℝ × ℝ), 
  circle_C1 (P.1) (P.2) ∧
  circle_C2 (P.1) (P.2) ∧
  (∀ k : ℝ, k ≠ 0 → equal_chord_lengths P k) ∧
  (P = (-3/2, 17/2) ∨ P = (5/2, -1/2)) :=
sorry

end find_point_coordinates_l53_53684


namespace hancho_milk_consumption_l53_53481

theorem hancho_milk_consumption :
  ∀ (initial_yeseul_consumption gayoung_bonus liters_left initial_milk consumption_yeseul consumption_gayoung consumption_total), 
  initial_yeseul_consumption = 0.1 →
  gayoung_bonus = 0.2 →
  liters_left = 0.3 →
  initial_milk = 1 →
  consumption_yeseul = initial_yeseul_consumption →
  consumption_gayoung = initial_yeseul_consumption + gayoung_bonus →
  consumption_total = consumption_yeseul + consumption_gayoung →
  (initial_milk - (consumption_total + liters_left)) = 0.3 :=
by sorry

end hancho_milk_consumption_l53_53481


namespace harrison_annual_croissant_expenditure_l53_53617

-- Define the different costs and frequency of croissants.
def cost_regular_croissant : ℝ := 3.50
def cost_almond_croissant : ℝ := 5.50
def cost_chocolate_croissant : ℝ := 4.50
def cost_ham_cheese_croissant : ℝ := 6.00

def frequency_regular_croissant : ℕ := 52
def frequency_almond_croissant : ℕ := 52
def frequency_chocolate_croissant : ℕ := 52
def frequency_ham_cheese_croissant : ℕ := 26

-- Calculate annual expenditure for each type of croissant.
def annual_expenditure (cost : ℝ) (frequency : ℕ) : ℝ :=
  cost * frequency

-- Total annual expenditure on croissants.
def total_annual_expenditure : ℝ :=
  annual_expenditure cost_regular_croissant frequency_regular_croissant +
  annual_expenditure cost_almond_croissant frequency_almond_croissant +
  annual_expenditure cost_chocolate_croissant frequency_chocolate_croissant +
  annual_expenditure cost_ham_cheese_croissant frequency_ham_cheese_croissant

-- The theorem to prove.
theorem harrison_annual_croissant_expenditure :
  total_annual_expenditure = 858 := by
  sorry

end harrison_annual_croissant_expenditure_l53_53617


namespace inequality_not_always_true_l53_53346

theorem inequality_not_always_true
  (x y w : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x^2 > y^2) (hw : w ≠ 0) :
  ∃ w, w ≠ 0 ∧ x^2 * w ≤ y^2 * w :=
sorry

end inequality_not_always_true_l53_53346


namespace average_age_combined_l53_53720

theorem average_age_combined (fifth_graders_count : ℕ) (fifth_graders_avg_age : ℚ)
                             (parents_count : ℕ) (parents_avg_age : ℚ)
                             (grandparents_count : ℕ) (grandparents_avg_age : ℚ) :
  fifth_graders_count = 40 →
  fifth_graders_avg_age = 10 →
  parents_count = 60 →
  parents_avg_age = 35 →
  grandparents_count = 20 →
  grandparents_avg_age = 65 →
  (fifth_graders_count * fifth_graders_avg_age + 
   parents_count * parents_avg_age + 
   grandparents_count * grandparents_avg_age) / 
  (fifth_graders_count + parents_count + grandparents_count) = 95 / 3 := sorry

end average_age_combined_l53_53720


namespace cos_75_eq_l53_53750

theorem cos_75_eq : Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_75_eq_l53_53750


namespace remainder_sum_div7_l53_53020

theorem remainder_sum_div7 (a b c : ℕ) (h1 : a * b * c ≡ 2 [MOD 7])
  (h2 : 3 * c ≡ 4 [MOD 7])
  (h3 : 4 * b ≡ 2 + b [MOD 7]) :
  (a + b + c) % 7 = 6 := by
  sorry

end remainder_sum_div7_l53_53020


namespace odd_function_has_specific_a_l53_53165

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = - f x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
x / ((2 * x + 1) * (x - a))

theorem odd_function_has_specific_a :
  ∀ a, is_odd (f a) → a = 1 / 2 :=
by sorry

end odd_function_has_specific_a_l53_53165


namespace total_nails_needed_l53_53923

-- Given conditions
def nails_per_plank : ℕ := 2
def number_of_planks : ℕ := 16

-- Prove the total number of nails required
theorem total_nails_needed : nails_per_plank * number_of_planks = 32 :=
by
  sorry

end total_nails_needed_l53_53923


namespace no_solution_system_iff_n_eq_neg_cbrt_four_l53_53787

variable (n : ℝ)

theorem no_solution_system_iff_n_eq_neg_cbrt_four :
    (∀ x y z : ℝ, ¬ (2 * n * x + 3 * y = 2 ∧ 3 * n * y + 4 * z = 3 ∧ 4 * x + 2 * n * z = 4)) ↔
    n = - (4 : ℝ)^(1/3) := 
by
  sorry

end no_solution_system_iff_n_eq_neg_cbrt_four_l53_53787


namespace students_and_swimmers_l53_53096

theorem students_and_swimmers (N : ℕ) (x : ℕ) 
  (h1 : x = N / 4) 
  (h2 : x / 2 = 4) : 
  N = 32 ∧ N - x = 24 := 
by 
  sorry

end students_and_swimmers_l53_53096


namespace half_angle_in_second_quadrant_l53_53153

theorem half_angle_in_second_quadrant 
  {θ : ℝ} (k : ℤ)
  (hθ_quadrant4 : 2 * k * Real.pi + (3 / 2) * Real.pi ≤ θ ∧ θ ≤ 2 * k * Real.pi + 2 * Real.pi)
  (hcos : abs (Real.cos (θ / 2)) = - Real.cos (θ / 2)) : 
  ∃ m : ℤ, (m * Real.pi + (Real.pi / 2) ≤ θ / 2 ∧ θ / 2 ≤ m * Real.pi + Real.pi) :=
sorry

end half_angle_in_second_quadrant_l53_53153


namespace tan_sum_l53_53937

theorem tan_sum (α : ℝ) (h : Real.cos (π / 2 + α) = 2 * Real.cos α) : 
  Real.tan α + Real.tan (2 * α) = -2 / 3 :=
by
  sorry

end tan_sum_l53_53937


namespace greatest_possible_x_max_possible_x_l53_53872

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end greatest_possible_x_max_possible_x_l53_53872


namespace cousins_room_distributions_l53_53353

theorem cousins_room_distributions : 
  let cousins := 5
  let rooms := 4
  let possible_distributions := (1 + 5 + 10 + 10 + 15 + 10 : ℕ)
  possible_distributions = 51 :=
by
  sorry

end cousins_room_distributions_l53_53353


namespace infinite_sum_converges_to_3_l53_53299

theorem infinite_sum_converges_to_3 :
  (∑' k : ℕ, (7 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))) = 3 :=
by
  sorry

end infinite_sum_converges_to_3_l53_53299


namespace total_liquid_consumption_l53_53338

-- Define the given conditions
def elijah_drink_pints : ℝ := 8.5
def emilio_drink_pints : ℝ := 9.5
def isabella_drink_liters : ℝ := 3
def xavier_drink_gallons : ℝ := 2
def pint_to_cups : ℝ := 2
def liter_to_cups : ℝ := 4.22675
def gallon_to_cups : ℝ := 16
def xavier_soda_fraction : ℝ := 0.60
def xavier_fruit_punch_fraction : ℝ := 0.40

-- Define the converted amounts
def elijah_cups := elijah_drink_pints * pint_to_cups
def emilio_cups := emilio_drink_pints * pint_to_cups
def isabella_cups := isabella_drink_liters * liter_to_cups
def xavier_total_cups := xavier_drink_gallons * gallon_to_cups
def xavier_soda_cups := xavier_soda_fraction * xavier_total_cups
def xavier_fruit_punch_cups := xavier_fruit_punch_fraction * xavier_total_cups

-- Total amount calculation
def total_cups := elijah_cups + emilio_cups + isabella_cups + xavier_soda_cups + xavier_fruit_punch_cups

-- Proof statement
theorem total_liquid_consumption : total_cups = 80.68025 := by
  sorry

end total_liquid_consumption_l53_53338


namespace value_of_k_l53_53601

theorem value_of_k (m n k : ℝ) (h1 : 3 ^ m = k) (h2 : 5 ^ n = k) (h3 : 1 / m + 1 / n = 2) : k = Real.sqrt 15 :=
  sorry

end value_of_k_l53_53601


namespace proof_problem_l53_53245

theorem proof_problem (a b : ℝ) (n : ℕ) 
  (P1 P2 : ℝ × ℝ)
  (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_n_gt_1 : n > 1)
  (h_P1_on_curve : P1.1 ^ n = a * P1.2 ^ n + b)
  (h_P2_on_curve : P2.1 ^ n = a * P2.2 ^ n + b)
  (h_y1_lt_y2 : P1.2 < P2.2)
  (A : ℝ) (h_A : A = (1/2) * |P1.1 * P2.2 - P2.1 * P1.2|) :
  b * P2.2 > 2 * n * P1.2 ^ (n - 1) * a ^ (1 - (1 / n)) * A :=
sorry

end proof_problem_l53_53245


namespace point_in_fourth_quadrant_l53_53676

theorem point_in_fourth_quadrant (x y : Real) (hx : x = 2) (hy : y = Real.tan 300) : 
  (0 < x) → (y < 0) → (x = 2 ∧ y = -Real.sqrt 3) :=
by
  intro hx_trans hy_trans
  -- Here you will provide statements or tactics to assist the proof if you were completing it
  sorry

end point_in_fourth_quadrant_l53_53676


namespace total_cost_is_15_l53_53814

def toast_cost : ℕ := 1
def egg_cost : ℕ := 3

def dale_toast : ℕ := 2
def dale_eggs : ℕ := 2

def andrew_toast : ℕ := 1
def andrew_eggs : ℕ := 2

def dale_breakfast_cost := dale_toast * toast_cost + dale_eggs * egg_cost
def andrew_breakfast_cost := andrew_toast * toast_cost + andrew_eggs * egg_cost

def total_breakfast_cost := dale_breakfast_cost + andrew_breakfast_cost

theorem total_cost_is_15 : total_breakfast_cost = 15 := by
  sorry

end total_cost_is_15_l53_53814


namespace pow_mod_eq_l53_53231

theorem pow_mod_eq :
  (13 ^ 7) % 11 = 7 :=
by
  sorry

end pow_mod_eq_l53_53231


namespace range_of_a_l53_53958

theorem range_of_a (a : ℝ) (x : ℝ) : (x > a ∧ x > 1) → (x > 1) → (a ≤ 1) :=
by 
  intros hsol hx
  sorry

end range_of_a_l53_53958


namespace solve_linear_system_l53_53247

-- Given conditions
def matrix : Matrix (Fin 2) (Fin 3) ℚ :=
  ![![1, -1, 1], ![1, 1, 3]]

def system_of_equations (x y : ℚ) : Prop :=
  (x - y = 1) ∧ (x + y = 3)

-- Desired solution
def solution (x y : ℚ) : Prop :=
  x = 2 ∧ y = 1

-- Proof problem statement
theorem solve_linear_system : ∃ x y : ℚ, system_of_equations x y ∧ solution x y := by
  sorry

end solve_linear_system_l53_53247


namespace triangle_centroid_altitude_l53_53274

/-- In triangle XYZ with side lengths XY = 7, XZ = 24, and YZ = 25, the length of GQ where Q 
    is the foot of the altitude from the centroid G to the side YZ is 56/25. -/
theorem triangle_centroid_altitude :
  let XY := 7
  let XZ := 24
  let YZ := 25
  let GQ := 56 / 25
  GQ = (56 : ℝ) / 25 :=
by
  -- proof goes here
  sorry

end triangle_centroid_altitude_l53_53274


namespace hyperbola_center_coordinates_l53_53224

-- Defining the equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  (3 * y + 6)^2 / 16 - (2 * x - 1)^2 / 9 = 1

-- Stating the theorem to verify the center of the hyperbola
theorem hyperbola_center_coordinates :
  ∃ (h k : ℝ), (h = 1/2) ∧ (k = -2) ∧ 
    ∀ x y, hyperbola_eq x y ↔ ((y + 2)^2 / (4 / 3)^2 - (x - 1/2)^2 / (3 / 2)^2 = 1) :=
by sorry

end hyperbola_center_coordinates_l53_53224


namespace simplify_expression_l53_53673

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l53_53673


namespace equation1_unique_solutions_equation2_unique_solutions_l53_53123

noncomputable def solve_equation1 : ℝ → Prop :=
fun x => x ^ 2 - 4 * x + 1 = 0

noncomputable def solve_equation2 : ℝ → Prop :=
fun x => 2 * x ^ 2 - 3 * x + 1 = 0

theorem equation1_unique_solutions :
  ∀ x, solve_equation1 x ↔ (x = 2 + Real.sqrt 3) ∨ (x = 2 - Real.sqrt 3) := by
  sorry

theorem equation2_unique_solutions :
  ∀ x, solve_equation2 x ↔ (x = 1) ∨ (x = 1 / 2) := by
  sorry

end equation1_unique_solutions_equation2_unique_solutions_l53_53123


namespace sum_of_all_possible_x_l53_53524

theorem sum_of_all_possible_x : 
  (∀ x : ℝ, |x - 5| - 4 = -1 → (x = 8 ∨ x = 2)) → ( ∃ (x1 x2 : ℝ), (x1 = 8) ∧ (x2 = 2) ∧ (x1 + x2 = 10) ) :=
by
  admit

end sum_of_all_possible_x_l53_53524


namespace triangle_angle_B_eq_60_l53_53616

theorem triangle_angle_B_eq_60 {A B C : ℝ} (h1 : B = 2 * A) (h2 : C = 3 * A) (h3 : A + B + C = 180) : B = 60 :=
by sorry

end triangle_angle_B_eq_60_l53_53616


namespace train_john_arrival_probability_l53_53898

-- Define the probability of independent uniform distributions on the interval [0, 120]
noncomputable def probability_train_present_when_john_arrives : ℝ :=
  let total_square_area := (120 : ℝ) * 120
  let triangle_area := (1 / 2) * 90 * 30
  let trapezoid_area := (1 / 2) * (30 + 0) * 30
  let total_shaded_area := triangle_area + trapezoid_area
  total_shaded_area / total_square_area

theorem train_john_arrival_probability :
  probability_train_present_when_john_arrives = 1 / 8 :=
by {
  sorry
}

end train_john_arrival_probability_l53_53898


namespace vasya_is_not_mistaken_l53_53386

theorem vasya_is_not_mistaken (X Y N A B : ℤ)
  (h_sum : X + Y = N)
  (h_tanya : A * X + B * Y ≡ 0 [ZMOD N]) :
  B * X + A * Y ≡ 0 [ZMOD N] :=
sorry

end vasya_is_not_mistaken_l53_53386


namespace employees_age_distribution_l53_53780

-- Define the total number of employees
def totalEmployees : ℕ := 15000

-- Define the percentages
def malePercentage : ℝ := 0.58
def femalePercentage : ℝ := 0.42

-- Define the age distribution percentages for male employees
def maleBelow30Percentage : ℝ := 0.25
def male30To50Percentage : ℝ := 0.40
def maleAbove50Percentage : ℝ := 0.35

-- Define the percentage of female employees below 30
def femaleBelow30Percentage : ℝ := 0.30

-- Define the number of male employees
def numMaleEmployees : ℝ := malePercentage * totalEmployees

-- Calculate the number of male employees in each age group
def numMaleBelow30 : ℝ := maleBelow30Percentage * numMaleEmployees
def numMale30To50 : ℝ := male30To50Percentage * numMaleEmployees
def numMaleAbove50 : ℝ := maleAbove50Percentage * numMaleEmployees

-- Define the number of female employees
def numFemaleEmployees : ℝ := femalePercentage * totalEmployees

-- Calculate the number of female employees below 30
def numFemaleBelow30 : ℝ := femaleBelow30Percentage * numFemaleEmployees

-- Calculate the total number of employees below 30
def totalBelow30 : ℝ := numMaleBelow30 + numFemaleBelow30

-- We now state our theorem to prove
theorem employees_age_distribution :
  numMaleBelow30 = 2175 ∧
  numMale30To50 = 3480 ∧
  numMaleAbove50 = 3045 ∧
  totalBelow30 = 4065 := by
    sorry

end employees_age_distribution_l53_53780


namespace dan_destroyed_l53_53088

def balloons_initial (fred: ℝ) (sam: ℝ) : ℝ := fred + sam

theorem dan_destroyed (fred: ℝ) (sam: ℝ) (final_balloons: ℝ) (destroyed_balloons: ℝ) :
  fred = 10.0 →
  sam = 46.0 →
  final_balloons = 40.0 →
  destroyed_balloons = (balloons_initial fred sam) - final_balloons →
  destroyed_balloons = 16.0 := by
  intros h1 h2 h3 h4
  sorry

end dan_destroyed_l53_53088


namespace total_amount_invested_l53_53897

-- Define the problem details: given conditions
def interest_rate_share1 : ℚ := 9 / 100
def interest_rate_share2 : ℚ := 11 / 100
def total_interest_rate : ℚ := 39 / 400
def amount_invested_share2 : ℚ := 3750

-- Define the total amount invested (A), the amount invested at the 9% share (x)
variable (A x : ℚ)

-- Conditions
axiom condition1 : x + amount_invested_share2 = A
axiom condition2 : interest_rate_share1 * x + interest_rate_share2 * amount_invested_share2 = total_interest_rate * A

-- Prove that the total amount invested in both types of shares is Rs. 10,000
theorem total_amount_invested : A = 10000 :=
by {
  -- proof goes here
  sorry
}

end total_amount_invested_l53_53897


namespace smallest_number_jungkook_l53_53229

theorem smallest_number_jungkook (jungkook yoongi yuna : ℕ) 
  (hj : jungkook = 6 - 3) (hy : yoongi = 4) (hu : yuna = 5) : 
  jungkook < yoongi ∧ jungkook < yuna :=
by
  sorry

end smallest_number_jungkook_l53_53229


namespace solve_inequality_l53_53883

-- Define the inequality
def inequality (x : ℝ) : Prop :=
  -3 * (x^2 - 4 * x + 16) * (x^2 + 6 * x + 8) / ((x^3 + 64) * (Real.sqrt (x^2 + 4 * x + 4))) ≤ x^2 + x - 3

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  x ∈ Set.Iic (-4) ∪ {x : ℝ | -4 < x ∧ x ≤ -3} ∪ {x : ℝ | -2 < x ∧ x ≤ -1} ∪ Set.Ici 0

-- The theorem statement, which we need to prove
theorem solve_inequality : ∀ x : ℝ, inequality x ↔ solution_set x :=
by
  intro x
  sorry

end solve_inequality_l53_53883


namespace exponent_fraction_simplification_l53_53297

theorem exponent_fraction_simplification :
  (2 ^ 2020 + 2 ^ 2016) / (2 ^ 2020 - 2 ^ 2016) = 17 / 15 :=
by
  sorry

end exponent_fraction_simplification_l53_53297


namespace triangle_angle_contradiction_l53_53668

theorem triangle_angle_contradiction :
  ∀ (α β γ : ℝ), (α + β + γ = 180) →
  (α > 60) ∧ (β > 60) ∧ (γ > 60) →
  false :=
by
  intros α β γ h_sum h_angles
  sorry

end triangle_angle_contradiction_l53_53668


namespace tyler_saltwater_animals_l53_53892

/-- Tyler had 56 aquariums for saltwater animals and each aquarium has 39 animals in it. 
    We need to prove that the total number of saltwater animals Tyler has is 2184. --/
theorem tyler_saltwater_animals : (56 * 39) = 2184 := by
  sorry

end tyler_saltwater_animals_l53_53892


namespace triangle_height_and_segments_l53_53969

-- Define the sides of the triangle
noncomputable def a : ℝ := 13
noncomputable def b : ℝ := 14
noncomputable def c : ℝ := 15

-- Define the height h and the segments m and 15 - m
noncomputable def m : ℝ := 6.6
noncomputable def h : ℝ := 11.2
noncomputable def base_segment_left : ℝ := m
noncomputable def base_segment_right : ℝ := c - m

-- The height and segments calculation theorem
theorem triangle_height_and_segments :
  h = 11.2 ∧ m = 6.6 ∧ (c - m) = 8.4 :=
by {
  sorry
}

end triangle_height_and_segments_l53_53969


namespace ratio_apps_optimal_l53_53887

theorem ratio_apps_optimal (max_apps : ℕ) (recommended_apps : ℕ) (apps_to_delete : ℕ) (current_apps : ℕ)
  (h_max_apps : max_apps = 50)
  (h_recommended_apps : recommended_apps = 35)
  (h_apps_to_delete : apps_to_delete = 20)
  (h_current_apps : current_apps = max_apps + apps_to_delete) :
  current_apps / recommended_apps = 2 :=
by {
  sorry
}

end ratio_apps_optimal_l53_53887


namespace total_cable_cost_neighborhood_l53_53501

-- Define the number of east-west streets and their length
def ew_streets : ℕ := 18
def ew_length_per_street : ℕ := 2

-- Define the number of north-south streets and their length
def ns_streets : ℕ := 10
def ns_length_per_street : ℕ := 4

-- Define the cable requirements and cost
def cable_per_mile_of_street : ℕ := 5
def cable_cost_per_mile : ℕ := 2000

-- Calculate total length of east-west streets
def ew_total_length : ℕ := ew_streets * ew_length_per_street

-- Calculate total length of north-south streets
def ns_total_length : ℕ := ns_streets * ns_length_per_street

-- Calculate total length of all streets
def total_street_length : ℕ := ew_total_length + ns_total_length

-- Calculate total length of cable required
def total_cable_length : ℕ := total_street_length * cable_per_mile_of_street

-- Calculate total cost of the cable
def total_cost : ℕ := total_cable_length * cable_cost_per_mile

-- The statement to prove
theorem total_cable_cost_neighborhood : total_cost = 760000 :=
by
  sorry

end total_cable_cost_neighborhood_l53_53501


namespace greatest_root_of_f_one_is_root_of_f_l53_53606

def f (x : ℝ) : ℝ := 16 * x^6 - 15 * x^4 + 4 * x^2 - 1

theorem greatest_root_of_f :
  ∀ x : ℝ, f x = 0 → x ≤ 1 :=
sorry

theorem one_is_root_of_f :
  f 1 = 0 :=
sorry

end greatest_root_of_f_one_is_root_of_f_l53_53606


namespace largest_odd_integer_satisfying_inequality_l53_53831

theorem largest_odd_integer_satisfying_inequality : 
  ∃ (x : ℤ), (x % 2 = 1) ∧ (1 / 4 < x / 6) ∧ (x / 6 < 7 / 9) ∧ (∀ y : ℤ, (y % 2 = 1) ∧ (1 / 4 < y / 6) ∧ (y / 6 < 7 / 9) → y ≤ x) :=
sorry

end largest_odd_integer_satisfying_inequality_l53_53831


namespace area_of_field_l53_53536

theorem area_of_field (L W A : ℕ) (h₁ : L = 20) (h₂ : L + 2 * W = 80) : A = 600 :=
by
  sorry

end area_of_field_l53_53536


namespace convert_base_8_to_7_l53_53008

def convert_base_8_to_10 (n : Nat) : Nat :=
  let d2 := n / 100 % 10
  let d1 := n / 10 % 10
  let d0 := n % 10
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

def convert_base_10_to_7 (n : Nat) : List Nat :=
  if n = 0 then [0]
  else 
    let rec helper (n : Nat) (acc : List Nat) : List Nat :=
      if n = 0 then acc
      else helper (n / 7) ((n % 7) :: acc)
    helper n []

def represent_in_base_7 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem convert_base_8_to_7 :
  represent_in_base_7 (convert_base_10_to_7 (convert_base_8_to_10 653)) = 1150 :=
by
  sorry

end convert_base_8_to_7_l53_53008


namespace range_of_a_for_three_tangents_curve_through_point_l53_53363

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := x^3 + 3 * x^2 + a * x + a - 2

noncomputable def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 6 * x + a

theorem range_of_a_for_three_tangents_curve_through_point :
  ∀ (a : ℝ), (∀ x0 : ℝ, 2 * x0^3 + 3 * x0^2 + 4 - a = 0 → 
    ((2 * -1^3 + 3 * -1^2 + 4 - a > 0) ∧ (2 * 0^3 + 3 * 0^2 + 4 - a < 0))) ↔ (4 < a ∧ a < 5) :=
by
  sorry

end range_of_a_for_three_tangents_curve_through_point_l53_53363


namespace length_YW_l53_53488

-- Definitions of the sides of the triangle
def XY := 6
def YZ := 8
def XZ := 10

-- The total perimeter of triangle XYZ
def perimeter : ℕ := XY + YZ + XZ

-- Each ant travels half the perimeter
def halfPerimeter : ℕ := perimeter / 2

-- Distance one ant travels from X to W through Y
def distanceXtoW : ℕ := XY + 6

-- Prove that the distance segment YW is 6
theorem length_YW : distanceXtoW = halfPerimeter := by sorry

end length_YW_l53_53488


namespace train_length_approx_500_l53_53028

noncomputable def length_of_train (speed_km_per_hr : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600
  speed_m_per_s * time_sec

theorem train_length_approx_500 :
  length_of_train 120 15 = 500 :=
by
  sorry

end train_length_approx_500_l53_53028


namespace find_distance_between_PQ_l53_53473

-- Defining distances and speeds
def distance_by_first_train (t : ℝ) : ℝ := 50 * t
def distance_by_second_train (t : ℝ) : ℝ := 40 * t
def distance_between_PQ (t : ℝ) : ℝ := distance_by_first_train t + (distance_by_first_train t - 100)

-- Main theorem stating the problem
theorem find_distance_between_PQ : ∃ t : ℝ, distance_by_first_train t - distance_by_second_train t = 100 ∧ distance_between_PQ t = 900 := 
sorry

end find_distance_between_PQ_l53_53473


namespace solve_fractions_l53_53079

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l53_53079


namespace original_solution_sugar_percentage_l53_53862

theorem original_solution_sugar_percentage :
  ∃ x : ℚ, (∀ (y : ℚ), (y = 14) → (∃ (z : ℚ), (z = 26) → (3 / 4 * x + 1 / 4 * z = y))) → x = 10 := 
  sorry

end original_solution_sugar_percentage_l53_53862


namespace continuity_at_4_l53_53277

def f (x : ℝ) : ℝ := -2 * x^2 + 9

theorem continuity_at_4 : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 4| < δ → |f x + 23| < ε := by
  sorry

end continuity_at_4_l53_53277


namespace emptying_rate_l53_53298

theorem emptying_rate (fill_time1 : ℝ) (total_fill_time : ℝ) (T : ℝ) 
  (h1 : fill_time1 = 4) 
  (h2 : total_fill_time = 20) 
  (h3 : 1 / fill_time1 - 1 / T = 1 / total_fill_time) :
  T = 5 :=
by
  sorry

end emptying_rate_l53_53298


namespace parabola_chord_midpoint_l53_53647

/-- 
If the point (3, 1) is the midpoint of a chord of the parabola y^2 = 2px, 
and the slope of the line containing this chord is 2, then p = 2. 
-/
theorem parabola_chord_midpoint (p : ℝ) :
    (∃ (m : ℝ), (m = 2) ∧ ∀ (x y : ℝ), y = 2 * x - 5 → y^2 = 2 * p * x → 
        ((x1 = 0 ∧ y1 = 0 ∧ x2 = 6 ∧ y2 = 6) → 
            (x1 + x2 = 6) ∧ (y1 + y2 = 2) ∧ (p = 2))) :=
sorry

end parabola_chord_midpoint_l53_53647


namespace apples_distribution_l53_53738

variable (x : ℕ)

theorem apples_distribution :
  0 ≤ 5 * x + 12 - 8 * (x - 1) ∧ 5 * x + 12 - 8 * (x - 1) < 8 :=
sorry

end apples_distribution_l53_53738


namespace find_f_neg_a_l53_53270

noncomputable def f (x : ℝ) : ℝ := x^3 * Real.cos x + 1

variable (a : ℝ)

-- Given condition
axiom h_fa : f a = 11

-- Statement to prove
theorem find_f_neg_a : f (-a) = -9 :=
by
  sorry

end find_f_neg_a_l53_53270


namespace factor_expression_l53_53214

theorem factor_expression (x y z : ℝ) :
  x^3 * (y^2 - z^2) - y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x * y + z^2 - z * x) :=
by
  sorry

end factor_expression_l53_53214


namespace problem_f_g_comp_sum_l53_53572

-- Define the functions
def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 9) / (x^2 - 2 * x + 5)
def g (x : ℚ) : ℚ := x - 2

-- Define the statement we want to prove
theorem problem_f_g_comp_sum (x : ℚ) (h : x = 2) : f (g x) + g (f x) = 36 / 5 := by
  sorry

end problem_f_g_comp_sum_l53_53572


namespace total_soda_bottles_l53_53443

def regular_soda : ℕ := 57
def diet_soda : ℕ := 26
def lite_soda : ℕ := 27

theorem total_soda_bottles : regular_soda + diet_soda + lite_soda = 110 := by
  sorry

end total_soda_bottles_l53_53443


namespace evaluate_f_of_f_of_3_l53_53896

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 2

theorem evaluate_f_of_f_of_3 :
  f (f 3) = 2943 :=
by
  sorry

end evaluate_f_of_f_of_3_l53_53896


namespace total_cantaloupes_l53_53691

theorem total_cantaloupes (fred_cantaloupes : ℕ) (tim_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : tim_cantaloupes = 44) : fred_cantaloupes + tim_cantaloupes = 82 :=
by
  sorry

end total_cantaloupes_l53_53691


namespace infinitely_many_solutions_implies_b_eq_neg6_l53_53760

theorem infinitely_many_solutions_implies_b_eq_neg6 (b : ℤ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 8)) → b = -6 :=
  sorry

end infinitely_many_solutions_implies_b_eq_neg6_l53_53760


namespace polygon_sides_l53_53411

theorem polygon_sides (n : ℕ) 
  (h : 3240 = 180 * (n - 2) - (360)) : n = 22 := 
by 
  sorry

end polygon_sides_l53_53411


namespace complete_the_square_l53_53105

theorem complete_the_square (x : ℝ) (h : x^2 + 7 * x - 5 = 0) : (x + 7 / 2) ^ 2 = 69 / 4 :=
sorry

end complete_the_square_l53_53105


namespace trader_profit_percentage_l53_53362

-- Definitions for the conditions
def trader_buys_weight (indicated_weight: ℝ) : ℝ :=
  1.10 * indicated_weight

def trader_claimed_weight_to_customer (actual_weight: ℝ) : ℝ :=
  1.30 * actual_weight

-- Main theorem statement
theorem trader_profit_percentage (indicated_weight: ℝ) (actual_weight: ℝ) (claimed_weight: ℝ) :
  trader_buys_weight 1000 = 1100 →
  trader_claimed_weight_to_customer actual_weight = claimed_weight →
  claimed_weight = 1000 →
  (1000 - actual_weight) / actual_weight * 100 = 30 :=
by
  intros h1 h2 h3
  sorry

end trader_profit_percentage_l53_53362


namespace percentage_decrease_l53_53610

theorem percentage_decrease (original_salary new_salary decreased_salary : ℝ) (p : ℝ) (D : ℝ) : 
  original_salary = 4000.0000000000005 →
  p = 10 →
  new_salary = original_salary * (1 + p/100) →
  decreased_salary = 4180 →
  decreased_salary = new_salary * (1 - D / 100) →
  D = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end percentage_decrease_l53_53610


namespace leonardo_needs_more_money_l53_53255

-- Defining the problem
def cost_of_chocolate : ℕ := 500 -- 5 dollars in cents
def leonardo_own_money : ℕ := 400 -- 4 dollars in cents
def borrowed_money : ℕ := 59 -- borrowed cents

-- Prove that Leonardo needs 41 more cents
theorem leonardo_needs_more_money : (cost_of_chocolate - (leonardo_own_money + borrowed_money) = 41) :=
by
  sorry

end leonardo_needs_more_money_l53_53255


namespace factorization_of_cubic_polynomial_l53_53885

theorem factorization_of_cubic_polynomial (x y z : ℝ) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = (x + y + z) * (x^2 + y^2 + z^2 - x * y - y * z - z * x) := 
by sorry

end factorization_of_cubic_polynomial_l53_53885


namespace crayons_slightly_used_l53_53777

theorem crayons_slightly_used (total_crayons : ℕ) (new_fraction : ℚ) (broken_fraction : ℚ) 
  (htotal : total_crayons = 120) (hnew : new_fraction = 1 / 3) (hbroken : broken_fraction = 20 / 100) :
  let new_crayons := total_crayons * new_fraction
  let broken_crayons := total_crayons * broken_fraction
  let slightly_used_crayons := total_crayons - new_crayons - broken_crayons
  slightly_used_crayons = 56 := 
by
  -- This is where the proof would go
  sorry

end crayons_slightly_used_l53_53777


namespace min_value_expression_l53_53049

open Real

theorem min_value_expression 
  (a : ℝ) 
  (b : ℝ) 
  (hb : 0 < b) 
  (e : ℝ) 
  (he : e = 2.718281828459045) :
  ∃ x : ℝ, 
  (x = 2 * (1 - log 2)^2) ∧
  ∀ a b, 
    0 < b → 
    ((1 / 2) * exp a - log (2 * b))^2 + (a - b)^2 ≥ x :=
sorry

end min_value_expression_l53_53049


namespace download_time_l53_53947

theorem download_time (speed : ℕ) (file1 file2 file3 : ℕ) (total_time : ℕ) (hours : ℕ) :
  speed = 2 ∧ file1 = 80 ∧ file2 = 90 ∧ file3 = 70 ∧ total_time = file1 / speed + file2 / speed + file3 / speed ∧
  hours = total_time / 60 → hours = 2 := 
by
  sorry

end download_time_l53_53947


namespace proof1_proof2a_proof2b_l53_53613

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (S : ℝ)

-- Given conditions for Question 1
def condition1 := (a = 3 * Real.cos C ∧ b = 1)

-- Proof statement for Question 1
theorem proof1 : condition1 a b C → Real.tan C = 2 * Real.tan B :=
by sorry

-- Given conditions for Question 2a
def condition2a := (S = 1 / 2 * a * b * Real.sin C ∧ S = 1 / 2 * 3 * Real.cos C * 1 * Real.sin C)

-- Proof statement for Question 2a
theorem proof2a : condition2a a b C S → Real.cos (2 * B) = 3 / 5 :=
by sorry

-- Given conditions for Question 2b
def condition2b := (c = Real.sqrt 10 / 2)

-- Proof statement for Question 2b
theorem proof2b : condition1 a b C → condition2b c → Real.cos (2 * B) = 3 / 5 :=
by sorry

end proof1_proof2a_proof2b_l53_53613


namespace flagpole_height_in_inches_l53_53802

theorem flagpole_height_in_inches
  (height_lamppost shadow_lamppost : ℚ)
  (height_flagpole shadow_flagpole : ℚ)
  (h₁ : height_lamppost = 50)
  (h₂ : shadow_lamppost = 12)
  (h₃ : shadow_flagpole = 18 / 12) :
  height_flagpole * 12 = 75 :=
by
  -- Note: To keep the theorem concise, proof steps are omitted
  sorry

end flagpole_height_in_inches_l53_53802


namespace remainder_of_sum_of_squares_mod_n_l53_53671

theorem remainder_of_sum_of_squares_mod_n (a b n : ℤ) (hn : n > 1) 
  (ha : a * a ≡ 1 [ZMOD n]) (hb : b * b ≡ 1 [ZMOD n]) : 
  (a^2 + b^2) % n = 2 := 
by 
  sorry

end remainder_of_sum_of_squares_mod_n_l53_53671


namespace find_n_l53_53056

theorem find_n (n k : ℕ) (a b : ℝ) (h_pos : k > 0) (h_n : n ≥ 2) (h_ab_neq : a ≠ 0 ∧ b ≠ 0) (h_a : a = (k + 1) * b) : n = 2 * k + 2 :=
by sorry

end find_n_l53_53056


namespace lana_extra_flowers_l53_53285

theorem lana_extra_flowers :
  ∀ (tulips roses used total_extra : ℕ),
    tulips = 36 →
    roses = 37 →
    used = 70 →
    total_extra = (tulips + roses - used) →
    total_extra = 3 :=
by
  intros tulips roses used total_extra ht hr hu hte
  rw [ht, hr, hu] at hte
  sorry

end lana_extra_flowers_l53_53285


namespace score_order_l53_53935

theorem score_order (a b c d : ℕ) 
  (h1 : b + d = a + c)
  (h2 : a + b > c + d)
  (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c := 
by
  sorry

end score_order_l53_53935


namespace donuts_percentage_missing_l53_53698

noncomputable def missing_donuts_percentage (initial_donuts : ℕ) (remaining_donuts : ℕ) : ℝ :=
  ((initial_donuts - remaining_donuts : ℕ) : ℝ) / initial_donuts * 100

theorem donuts_percentage_missing
  (h_initial : ℕ := 30)
  (h_remaining : ℕ := 9) :
  missing_donuts_percentage h_initial h_remaining = 70 :=
by
  sorry

end donuts_percentage_missing_l53_53698


namespace route_comparison_l53_53398

-- Definitions based on given conditions

def time_uphill : ℕ := 6
def time_path : ℕ := 2 * time_uphill
def total_first_two_stages : ℕ := time_uphill + time_path
def time_final_stage : ℕ := total_first_two_stages / 3
def total_time_first_route : ℕ := total_first_two_stages + time_final_stage

def time_flat_path : ℕ := 14
def time_second_stage : ℕ := 2 * time_flat_path
def total_time_second_route : ℕ := time_flat_path + time_second_stage

-- Statement we want to prove
theorem route_comparison : 
  total_time_second_route - total_time_first_route = 18 := by
  sorry

end route_comparison_l53_53398


namespace quadratic_range_and_value_l53_53543

theorem quadratic_range_and_value (k : ℝ) :
  (∃ x1 x2 : ℝ, (x1^2 + (2 * k - 1) * x1 + k^2 - 1 = 0) ∧
  (x2^2 + (2 * k - 1) * x2 + k^2 - 1 = 0)) →
  k ≤ 5 / 4 ∧ (∀ x1 x2 : ℝ, (x1^2 + (2 * k - 1) * x1 + k^2 - 1 = 0) ∧
  (x2^2 + (2 * k - 1) * x2 + k^2 - 1 = 0) ∧ (x1^2 + x2^2 = 16 + x1 * x2)) → k = -2 :=
by sorry

end quadratic_range_and_value_l53_53543


namespace rectangular_solid_surface_area_l53_53186

theorem rectangular_solid_surface_area (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (h_volume : a * b * c = 1001) :
  2 * (a * b + b * c + c * a) = 622 :=
by
  sorry

end rectangular_solid_surface_area_l53_53186


namespace natalie_needs_10_bushes_l53_53064

-- Definitions based on the conditions
def bushes_to_containers (bushes : ℕ) := bushes * 10
def containers_to_zucchinis (containers : ℕ) := (containers * 3) / 4

-- The proof statement
theorem natalie_needs_10_bushes :
  ∃ bushes : ℕ, containers_to_zucchinis (bushes_to_containers bushes) ≥ 72 ∧ bushes = 10 :=
sorry

end natalie_needs_10_bushes_l53_53064


namespace abs_ineq_range_m_l53_53055

theorem abs_ineq_range_m :
  ∀ m : ℝ, (∀ x : ℝ, |x - 1| + |x + 2| ≥ m) ↔ m ≤ 3 :=
by
  sorry

end abs_ineq_range_m_l53_53055


namespace min_a2_b2_c2_l53_53528

theorem min_a2_b2_c2 (a b c : ℕ) (h : a + 2 * b + 3 * c = 73) : a^2 + b^2 + c^2 ≥ 381 :=
by sorry

end min_a2_b2_c2_l53_53528


namespace composite_A_l53_53761

def A : ℕ := 10^1962 + 1

theorem composite_A : ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ A = p * q :=
  sorry

end composite_A_l53_53761


namespace mean_height_is_approx_correct_l53_53507

def heights : List ℕ := [120, 123, 127, 132, 133, 135, 140, 142, 145, 148, 152, 155, 158, 160]

def mean_height : ℚ := heights.sum / heights.length

theorem mean_height_is_approx_correct : 
  abs (mean_height - 140.71) < 0.01 := 
by
  sorry

end mean_height_is_approx_correct_l53_53507


namespace minimize_sum_of_legs_l53_53737

noncomputable def area_of_right_angle_triangle (a b : ℝ) : Prop :=
  1/2 * a * b = 50

theorem minimize_sum_of_legs (a b : ℝ) (h : area_of_right_angle_triangle a b) :
  a + b = 20 ↔ a = 10 ∧ b = 10 :=
by
  sorry

end minimize_sum_of_legs_l53_53737


namespace inverse_B_squared_l53_53160

variable (B : Matrix (Fin 2) (Fin 2) ℝ)

def B_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -3, 2],
    ![  1, -1 ]]

theorem inverse_B_squared :
  B⁻¹ = B_inv →
  (B^2)⁻¹ = B_inv * B_inv :=
by sorry

end inverse_B_squared_l53_53160


namespace original_wire_length_l53_53395

theorem original_wire_length (S L : ℝ) (h1: S = 30) (h2: S = (3 / 5) * L) : S + L = 80 := by 
  sorry

end original_wire_length_l53_53395


namespace original_selling_price_l53_53963

theorem original_selling_price (P SP1 SP2 : ℝ) (h1 : SP1 = 1.10 * P)
    (h2 : SP2 = 1.17 * P) (h3 : SP2 - SP1 = 35) : SP1 = 550 :=
by
  sorry

end original_selling_price_l53_53963


namespace triangle_acute_of_angles_sum_gt_90_l53_53754

theorem triangle_acute_of_angles_sum_gt_90 
  (α β γ : ℝ) 
  (h₁ : α + β + γ = 180) 
  (h₂ : α + β > 90) 
  (h₃ : α + γ > 90) 
  (h₄ : β + γ > 90) 
  : α < 90 ∧ β < 90 ∧ γ < 90 :=
sorry

end triangle_acute_of_angles_sum_gt_90_l53_53754


namespace other_root_of_quadratic_l53_53643

theorem other_root_of_quadratic (k : ℝ) (h : -2 * 1 = -2) (h_eq : x^2 + k * x - 2 = 0) :
  1 * -2 = -2 :=
by
  sorry

end other_root_of_quadratic_l53_53643


namespace total_letters_received_l53_53657

theorem total_letters_received 
  (Brother_received Greta_received Mother_received : ℕ) 
  (h1 : Greta_received = Brother_received + 10)
  (h2 : Brother_received = 40)
  (h3 : Mother_received = 2 * (Greta_received + Brother_received)) :
  Brother_received + Greta_received + Mother_received = 270 := 
sorry

end total_letters_received_l53_53657


namespace find_triples_l53_53562

theorem find_triples (a b c : ℝ) 
  (h1 : a = (b + c) ^ 2) 
  (h2 : b = (a + c) ^ 2) 
  (h3 : c = (a + b) ^ 2) : 
  (a = 0 ∧ b = 0 ∧ c = 0) 
  ∨ 
  (a = 1/4 ∧ b = 1/4 ∧ c = 1/4) :=
  sorry

end find_triples_l53_53562


namespace lisa_matching_pair_probability_l53_53815

theorem lisa_matching_pair_probability :
  let total_socks := 22
  let gray_socks := 12
  let white_socks := 10
  let total_pairs := total_socks * (total_socks - 1) / 2
  let gray_pairs := gray_socks * (gray_socks - 1) / 2
  let white_pairs := white_socks * (white_socks - 1) / 2
  let matching_pairs := gray_pairs + white_pairs
  let probability := matching_pairs / total_pairs
  probability = (111 / 231) :=
by
  sorry

end lisa_matching_pair_probability_l53_53815


namespace functional_expression_value_at_x_equals_zero_l53_53914

-- Define the basic properties
def y_inversely_proportional_to_x_plus_2 (y x : ℝ) : Prop :=
  ∃ k : ℝ, y = k / (x + 2)

-- Given condition: y = 3 when x = -1
def condition (y x : ℝ) : Prop :=
  y = 3 ∧ x = -1

-- Theorems to prove
theorem functional_expression (y x : ℝ) :
  y_inversely_proportional_to_x_plus_2 y x ∧ condition y x → y = 3 / (x + 2) :=
by
  sorry

theorem value_at_x_equals_zero (y x : ℝ) :
  y_inversely_proportional_to_x_plus_2 y x ∧ condition y x → (y = 3 / (x + 2) ∧ x = 0 → y = 3 / 2) :=
by
  sorry

end functional_expression_value_at_x_equals_zero_l53_53914


namespace robot_Y_reaches_B_after_B_reaches_A_l53_53980

-- Definitions for the setup of the problem
def time_J_to_B (t_J_to_B : ℕ) := t_J_to_B = 12
def time_J_catch_up_B (t_J_catch_up_B : ℕ) := t_J_catch_up_B = 9

-- Main theorem to be proved
theorem robot_Y_reaches_B_after_B_reaches_A : 
  ∀ t_J_to_B t_J_catch_up_B, 
    (time_J_to_B t_J_to_B) → 
    (time_J_catch_up_B t_J_catch_up_B) →
    ∃ t : ℕ, t = 56 :=
by 
  sorry

end robot_Y_reaches_B_after_B_reaches_A_l53_53980


namespace irene_overtime_pay_per_hour_l53_53368

def irene_base_pay : ℝ := 500
def irene_base_hours : ℕ := 40
def irene_total_hours_last_week : ℕ := 50
def irene_total_income_last_week : ℝ := 700

theorem irene_overtime_pay_per_hour :
  (irene_total_income_last_week - irene_base_pay) / (irene_total_hours_last_week - irene_base_hours) = 20 := 
by
  sorry

end irene_overtime_pay_per_hour_l53_53368


namespace grains_in_batch_l53_53828

-- Define the given constants from the problem
def total_rice_shi : ℕ := 1680
def sample_total_grains : ℕ := 250
def sample_containing_grains : ℕ := 25

-- Define the statement to be proven
theorem grains_in_batch : (total_rice_shi * (sample_containing_grains / sample_total_grains)) = 168 := by
  -- Proof steps will go here
  sorry

end grains_in_batch_l53_53828


namespace factor_exp_l53_53264

theorem factor_exp (k : ℕ) : 3^1999 - 3^1998 - 3^1997 + 3^1996 = k * 3^1996 → k = 16 :=
by
  intro h
  sorry

end factor_exp_l53_53264


namespace log_diff_l53_53237

theorem log_diff : (Real.log (12:ℝ) / Real.log (2:ℝ)) - (Real.log (3:ℝ) / Real.log (2:ℝ)) = 2 := 
by
  sorry

end log_diff_l53_53237


namespace base_conversion_l53_53014

theorem base_conversion (b2_to_b10_step : 101101 = 1 * 2 ^ 5 + 0 * 2 ^ 4 + 1 * 2 ^ 3 + 1 * 2 ^ 2 + 0 * 2 + 1)
  (b10_to_b7_step1 : 45 / 7 = 6) (b10_to_b7_step2 : 45 % 7 = 3) (b10_to_b7_step3 : 6 / 7 = 0) (b10_to_b7_step4 : 6 % 7 = 6) :
  101101 = 45 ∧ 45 = 63 :=
by {
  -- Conversion steps from the proof will be filled in here
  sorry
}

end base_conversion_l53_53014


namespace avg_words_per_hour_l53_53628

theorem avg_words_per_hour (words hours : ℝ) (h_words : words = 40000) (h_hours : hours = 80) :
  words / hours = 500 :=
by
  rw [h_words, h_hours]
  norm_num
  done

end avg_words_per_hour_l53_53628


namespace walking_running_ratio_l53_53075

theorem walking_running_ratio (d_w d_r : ℝ) (h1 : d_w / 4 + d_r / 8 = 3) (h2 : d_w + d_r = 16) :
  d_w / d_r = 1 := by
  sorry

end walking_running_ratio_l53_53075


namespace length_of_platform_is_correct_l53_53801

-- Given conditions:
def length_of_train : ℕ := 250
def speed_of_train_kmph : ℕ := 72
def time_to_cross_platform : ℕ := 20

-- Convert speed from kmph to m/s
def speed_of_train_mps : ℕ := speed_of_train_kmph * 1000 / 3600

-- Distance covered in 20 seconds
def distance_covered : ℕ := speed_of_train_mps * time_to_cross_platform

-- Length of the platform
def length_of_platform : ℕ := distance_covered - length_of_train

-- The proof statement
theorem length_of_platform_is_correct :
  length_of_platform = 150 := by
  -- This proof would involve the detailed calculations and verifications as laid out in the solution steps.
  sorry

end length_of_platform_is_correct_l53_53801


namespace g_1_5_l53_53480

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_defined (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : g x ≠ 0

axiom g_zero : g 0 = 0

axiom g_mono (x y : ℝ) (hx : 0 ≤ x ∧ x < y ∧ y ≤ 1) : g x ≤ g y

axiom g_symmetry (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : g (1 - x) = 1 - g x

axiom g_scaling (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : g (x/4) = g x / 2

theorem g_1_5 : g (1 / 5) = 1 / 4 := 
sorry

end g_1_5_l53_53480


namespace union_A_B_intersection_complement_A_B_l53_53058

def A := {x : ℝ | 3 ≤ x ∧ x < 7}
def B := {x : ℝ | 4 < x ∧ x < 10}

theorem union_A_B :
  A ∪ B = {x : ℝ | 3 ≤ x ∧ x < 10} :=
sorry

def complement_A := {x : ℝ | x < 3 ∨ x ≥ 7}

theorem intersection_complement_A_B :
  (complement_A ∩ B) = {x : ℝ | 7 ≤ x ∧ x < 10} :=
sorry

end union_A_B_intersection_complement_A_B_l53_53058


namespace simple_interest_rate_l53_53640

variables (P R T SI : ℝ)

theorem simple_interest_rate :
  T = 10 →
  SI = (2 / 5) * P →
  SI = (P * R * T) / 100 →
  R = 4 :=
by
  intros hT hSI hFormula
  sorry

end simple_interest_rate_l53_53640


namespace total_cost_l53_53276

-- Define conditions as variables
def n_b : ℕ := 3    -- number of bedroom doors
def n_o : ℕ := 2    -- number of outside doors
def c_o : ℕ := 20   -- cost per outside door
def c_b : ℕ := c_o / 2  -- cost per bedroom door

-- Define the total cost using the conditions
def c_total : ℕ := (n_o * c_o) + (n_b * c_b)

-- State the theorem to be proven
theorem total_cost :
  c_total = 70 :=
by
  sorry

end total_cost_l53_53276


namespace find_point_A_equidistant_l53_53246

theorem find_point_A_equidistant :
  ∃ (x : ℝ), (∃ A : ℝ × ℝ × ℝ, A = (x, 0, 0)) ∧
              (∃ B : ℝ × ℝ × ℝ, B = (4, 0, 5)) ∧
              (∃ C : ℝ × ℝ × ℝ, C = (5, 4, 2)) ∧
              (dist (x, 0, 0) (4, 0, 5) = dist (x, 0, 0) (5, 4, 2)) ∧ 
              (x = 2) :=
by
  sorry

end find_point_A_equidistant_l53_53246


namespace minimum_value_of_f_range_of_t_l53_53843

def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 4)

theorem minimum_value_of_f : ∀ x, f x ≥ 6 ∧ ∃ x0 : ℝ, f x0 = 6 := 
by sorry

theorem range_of_t (t : ℝ) : (t ≤ -2 ∨ t ≥ 3) ↔ ∃ x : ℝ, -3 ≤ x ∧ x ≤ 5 ∧ f x ≤ t^2 - t :=
by sorry

end minimum_value_of_f_range_of_t_l53_53843


namespace min_attempts_to_pair_keys_suitcases_l53_53233

theorem min_attempts_to_pair_keys_suitcases (n : ℕ) : ∃ p : ℕ, (∀ (keyOpen : Fin n → Fin n), ∃ f : (Fin n × Fin n) → Bool, ∀ (i j : Fin n), i ≠ j → (keyOpen i = j ↔ f (i, j) = tt)) ∧ p = Nat.choose n 2 := by
  sorry

end min_attempts_to_pair_keys_suitcases_l53_53233


namespace lowest_temperature_l53_53798

theorem lowest_temperature 
  (temps : Fin 5 → ℝ) 
  (avg_temp : (temps 0 + temps 1 + temps 2 + temps 3 + temps 4) / 5 = 60)
  (max_range : ∀ i j, temps i - temps j ≤ 75) : 
  ∃ L : ℝ, L = 0 ∧ ∃ i, temps i = L :=
by 
  sorry

end lowest_temperature_l53_53798


namespace range_of_a_l53_53605

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 := 
sorry

end range_of_a_l53_53605


namespace even_integer_squares_l53_53508

noncomputable def Q (x : ℤ) : ℤ := x^4 + 6 * x^3 + 11 * x^2 + 3 * x + 25

theorem even_integer_squares (x : ℤ) (hx : x % 2 = 0) :
  (∃ (a : ℤ), Q x = a ^ 2) → x = 8 :=
by
  sorry

end even_integer_squares_l53_53508


namespace find_number_l53_53740

theorem find_number (x : ℝ) (h : -200 * x = 1600) : x = -8 :=
by sorry

end find_number_l53_53740


namespace odometer_trip_l53_53539

variables (d e f : ℕ) (x : ℕ)

-- Define the conditions
def start_odometer (d e f : ℕ) : ℕ := 100 * d + 10 * e + f
def end_odometer (d e f : ℕ) : ℕ := 100 * f + 10 * e + d
def distance_travelled (x : ℕ) : ℕ := 65 * x
def valid_trip (d e f x : ℕ) : Prop := 
  d ≥ 1 ∧ d + e + f ≤ 9 ∧ 
  end_odometer d e f - start_odometer d e f = distance_travelled x

-- The final statement to prove
theorem odometer_trip (h : valid_trip d e f x) : d^2 + e^2 + f^2 = 41 := 
sorry

end odometer_trip_l53_53539


namespace box_volume_l53_53680

-- Definitions for the dimensions of the box: Length (L), Width (W), and Height (H)
variables (L W H : ℝ)

-- Condition 1: Area of the front face is half the area of the top face
def condition1 := L * W = 0.5 * (L * H)

-- Condition 2: Area of the top face is 1.5 times the area of the side face
def condition2 := L * H = 1.5 * (W * H)

-- Condition 3: Area of the side face is 200
def condition3 := W * H = 200

-- Theorem stating the volume of the box is 3000 given the above conditions
theorem box_volume : condition1 L W H ∧ condition2 L W H ∧ condition3 W H → L * W * H = 3000 :=
by sorry

end box_volume_l53_53680


namespace nitrogen_mass_percentage_in_ammonium_phosphate_l53_53218

def nitrogen_mass_percentage
  (molar_mass_N : ℚ)
  (molar_mass_H : ℚ)
  (molar_mass_P : ℚ)
  (molar_mass_O : ℚ)
  : ℚ :=
  let molar_mass_NH4 := molar_mass_N + 4 * molar_mass_H
  let molar_mass_PO4 := molar_mass_P + 4 * molar_mass_O
  let molar_mass_NH4_3_PO4 := 3 * molar_mass_NH4 + molar_mass_PO4
  let mass_N_in_NH4_3_PO4 := 3 * molar_mass_N
  (mass_N_in_NH4_3_PO4 / molar_mass_NH4_3_PO4) * 100

theorem nitrogen_mass_percentage_in_ammonium_phosphate
  (molar_mass_N : ℚ := 14.01)
  (molar_mass_H : ℚ := 1.01)
  (molar_mass_P : ℚ := 30.97)
  (molar_mass_O : ℚ := 16.00)
  : nitrogen_mass_percentage molar_mass_N molar_mass_H molar_mass_P molar_mass_O = 28.19 :=
by
  sorry

end nitrogen_mass_percentage_in_ammonium_phosphate_l53_53218


namespace find_g_of_2_l53_53910

theorem find_g_of_2 {g : ℝ → ℝ} (h : ∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) : g 2 = 14 :=
sorry

end find_g_of_2_l53_53910


namespace snake_length_difference_l53_53166

theorem snake_length_difference :
  ∀ (jake_len penny_len : ℕ), 
    jake_len > penny_len →
    jake_len + penny_len = 70 →
    jake_len = 41 →
    jake_len - penny_len = 12 :=
by
  intros jake_len penny_len h1 h2 h3
  sorry

end snake_length_difference_l53_53166


namespace benny_money_l53_53807

-- Conditions
def cost_per_apple (cost : ℕ) := cost = 4
def apples_needed (apples : ℕ) := apples = 5 * 18

-- The proof problem
theorem benny_money (cost : ℕ) (apples : ℕ) (total_money : ℕ) :
  cost_per_apple cost → apples_needed apples → total_money = apples * cost → total_money = 360 :=
by
  intros h_cost h_apples h_total
  rw [h_cost, h_apples] at h_total
  exact h_total

end benny_money_l53_53807


namespace slices_needed_l53_53344

def number_of_sandwiches : ℕ := 5
def slices_per_sandwich : ℕ := 3
def total_slices_required (n : ℕ) (s : ℕ) : ℕ := n * s

theorem slices_needed : total_slices_required number_of_sandwiches slices_per_sandwich = 15 :=
by
  sorry

end slices_needed_l53_53344


namespace total_amount_spent_is_300_l53_53396

-- Definitions of conditions
def S : ℕ := 97
def H : ℕ := 2 * S + 9

-- The total amount spent
def total_spent : ℕ := S + H

-- Proof statement
theorem total_amount_spent_is_300 : total_spent = 300 :=
by
  sorry

end total_amount_spent_is_300_l53_53396


namespace cos_theta_sub_pi_div_3_value_l53_53538

open Real

noncomputable def problem_statement (θ : ℝ) : Prop :=
  sin (3 * π - θ) = (sqrt 5 / 2) * sin (π / 2 + θ)

theorem cos_theta_sub_pi_div_3_value (θ : ℝ) (hθ : problem_statement θ) :
  cos (θ - π / 3) = 1 / 3 + sqrt 15 / 6 ∨ cos (θ - π / 3) = - (1 / 3 + sqrt 15 / 6) :=
sorry

end cos_theta_sub_pi_div_3_value_l53_53538


namespace pond_eye_count_l53_53869

def total_animal_eyes (snakes alligators spiders snails : ℕ) 
    (snake_eyes alligator_eyes spider_eyes snail_eyes: ℕ) : ℕ :=
  snakes * snake_eyes + alligators * alligator_eyes + spiders * spider_eyes + snails * snail_eyes

theorem pond_eye_count : total_animal_eyes 18 10 5 15 2 2 8 2 = 126 := 
by
  sorry

end pond_eye_count_l53_53869


namespace reading_time_difference_l53_53308

theorem reading_time_difference
  (tristan_speed : ℕ := 120)
  (ella_speed : ℕ := 40)
  (book_pages : ℕ := 360) :
  let tristan_time := book_pages / tristan_speed
  let ella_time := book_pages / ella_speed
  let time_difference_hours := ella_time - tristan_time
  let time_difference_minutes := time_difference_hours * 60
  time_difference_minutes = 360 :=
by
  sorry

end reading_time_difference_l53_53308


namespace solve_triplet_l53_53626

theorem solve_triplet (x y z : ℕ) (h : 2^x * 3^y + 1 = 7^z) :
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 4 ∧ y = 1 ∧ z = 2) :=
 by sorry

end solve_triplet_l53_53626


namespace homework_total_l53_53505

theorem homework_total :
  let math_pages := 20
  let reading_pages := math_pages - (30 * math_pages / 100)
  let science_pages := 2 * reading_pages
  math_pages + reading_pages + science_pages = 62 :=
by
  let math_pages := 20
  let reading_pages := math_pages - (30 * math_pages / 100)
  let science_pages := 2 * reading_pages
  show math_pages + reading_pages + science_pages = 62
  sorry

end homework_total_l53_53505


namespace time_spent_moving_l53_53856

noncomputable def time_per_trip_filling : ℝ := 15
noncomputable def time_per_trip_driving : ℝ := 30
noncomputable def time_per_trip_unloading : ℝ := 20
noncomputable def number_of_trips : ℕ := 10

theorem time_spent_moving :
  10.83 = (time_per_trip_filling + time_per_trip_driving + time_per_trip_unloading) * number_of_trips / 60 :=
by
  sorry

end time_spent_moving_l53_53856


namespace avg_weight_b_c_l53_53225

theorem avg_weight_b_c
  (a b c : ℝ)
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : b = 31) :
  (b + c) / 2 = 43 := 
by {
  sorry
}

end avg_weight_b_c_l53_53225


namespace y_intercept_of_line_l53_53857

theorem y_intercept_of_line : 
  (∃ t : ℝ, 4 - 4 * t = 0) → (∃ y : ℝ, y = -2 + 3 * 1) := 
by
  sorry

end y_intercept_of_line_l53_53857


namespace water_parts_in_solution_l53_53545

def lemonade_syrup_parts : ℝ := 7
def target_percentage : ℝ := 0.30
def adjusted_parts : ℝ := 2.1428571428571423

-- Original equation: L = 0.30 * (L + W)
-- Substitute L = 7 for the particular instance.
-- Therefore, 7 = 0.30 * (7 + W)

theorem water_parts_in_solution (W : ℝ) : 
  (7 = 0.30 * (7 + W)) → 
  W = 16.333333333333332 := 
by
  sorry

end water_parts_in_solution_l53_53545


namespace minimal_surface_area_l53_53578

-- Definitions based on the conditions in the problem.
def unit_cube (a b c : ℕ) : Prop := a * b * c = 25
def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + a * c + b * c)

-- The proof problem statement.
theorem minimal_surface_area : ∃ (a b c : ℕ), unit_cube a b c ∧ surface_area a b c = 54 := 
sorry

end minimal_surface_area_l53_53578


namespace initial_capacity_l53_53244

theorem initial_capacity (x : ℝ) (h1 : 0.9 * x = 198) : x = 220 :=
by
  sorry

end initial_capacity_l53_53244


namespace miki_sandcastle_height_correct_l53_53320

namespace SandcastleHeight

def sister_sandcastle_height := 0.5
def difference_in_height := 0.3333333333333333
def miki_sandcastle_height := sister_sandcastle_height + difference_in_height

theorem miki_sandcastle_height_correct : miki_sandcastle_height = 0.8333333333333333 := by
  unfold miki_sandcastle_height sister_sandcastle_height difference_in_height
  simp
  sorry

end SandcastleHeight

end miki_sandcastle_height_correct_l53_53320


namespace cost_price_of_one_ball_l53_53318

theorem cost_price_of_one_ball (x : ℝ) (h : 11 * x - 720 = 5 * x) : x = 120 :=
sorry

end cost_price_of_one_ball_l53_53318


namespace range_of_a_l53_53658

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → (x^2 + 2*x + a) / x > 0) ↔ a > -3 :=
by
  sorry

end range_of_a_l53_53658


namespace longer_train_length_l53_53841

def length_of_longer_train
  (speed_train1 : ℝ) (speed_train2 : ℝ)
  (length_shorter_train : ℝ) (time_to_clear : ℝ)
  (relative_speed : ℝ := (speed_train1 + speed_train2) * 1000 / 3600)
  (total_distance : ℝ := relative_speed * time_to_clear) : ℝ :=
  total_distance - length_shorter_train

theorem longer_train_length :
  length_of_longer_train 80 55 121 7.626056582140095 = 164.9771230827526 :=
by
  unfold length_of_longer_train
  norm_num
  sorry  -- This placeholder is used to avoid writing out the full proof.

end longer_train_length_l53_53841


namespace angle_relationship_l53_53239

-- Define the angles and the relationship
def larger_angle : ℝ := 99
def smaller_angle : ℝ := 81

-- State the problem as a theorem
theorem angle_relationship : larger_angle - smaller_angle = 18 := 
by
  -- The proof would be here
  sorry

end angle_relationship_l53_53239


namespace find_f_neg2_l53_53522

-- Condition (1): f is an even function on ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Condition (2): f(x) = x^2 + 1 for x > 0
def function_defined_for_positive_x {f : ℝ → ℝ} (h_even : even_function f): Prop :=
  ∀ x : ℝ, x > 0 → f x = x^2 + 1

-- Proof problem: prove that given the conditions, f(-2) = 5
theorem find_f_neg2 (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_pos : function_defined_for_positive_x h_even) : 
  f (-2) = 5 := 
sorry

end find_f_neg2_l53_53522


namespace sum_transformed_roots_l53_53074

theorem sum_transformed_roots :
  ∀ (a b c : ℝ),
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) →
  (45 * a^3 - 75 * a^2 + 33 * a - 2 = 0) →
  (45 * b^3 - 75 * b^2 + 33 * b - 2 = 0) →
  (45 * c^3 - 75 * c^2 + 33 * c - 2 = 0) →
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 60) :=
by
  intros a b c h_bounds h_poly_a h_poly_b h_poly_c h_distinct
  sorry

end sum_transformed_roots_l53_53074


namespace marble_ratio_l53_53271

theorem marble_ratio (total_marbles red_marbles dark_blue_marbles : ℕ) (h_total : total_marbles = 63) (h_red : red_marbles = 38) (h_blue : dark_blue_marbles = 6) :
  (total_marbles - red_marbles - dark_blue_marbles) / red_marbles = 1 / 2 := by
  sorry

end marble_ratio_l53_53271


namespace cos_sum_eq_neg_ratio_l53_53434

theorem cos_sum_eq_neg_ratio (γ δ : ℝ) 
  (hγ: Complex.exp (Complex.I * γ) = 4 / 5 + 3 / 5 * Complex.I) 
  (hδ: Complex.exp (Complex.I * δ) = -5 / 13 + 12 / 13 * Complex.I) :
  Real.cos (γ + δ) = -56 / 65 :=
  sorry

end cos_sum_eq_neg_ratio_l53_53434


namespace max_value_f_min_value_a_l53_53848

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - 4 * Real.pi / 3) + 2 * (Real.cos x)^2

theorem max_value_f :
  ∀ x, f x ≤ 2 ∧ (∃ k : ℤ, x = k * Real.pi - Real.pi / 6) → f x = 2 :=
by { sorry }

variables {A B C a b c : ℝ}

noncomputable def f' (x : ℝ) : ℝ := Real.cos (2 * x +  Real.pi / 3) + 1

theorem min_value_a
  (h1 : f' (B + C) = 3/2)
  (h2 : b + c = 2)
  (h3 : A + B + C = Real.pi)
  (h4 : Real.cos A = 1/2) :
  ∃ a, ∀ b c, a^2 = b^2 + c^2 - 2 * b * c * Real.cos A ∧ a ≥ 1 :=
by { sorry }

end max_value_f_min_value_a_l53_53848


namespace convex_hexagon_possibilities_l53_53139

noncomputable def hexagon_side_lengths : List ℕ := [1, 2, 3, 4, 5, 6]

theorem convex_hexagon_possibilities : 
  ∃ (hexagons : List (List ℕ)), 
    (∀ h ∈ hexagons, 
      (h.length = 6) ∧ 
      (∀ a ∈ h, a ∈ hexagon_side_lengths)) ∧ 
      (hexagons.length = 3) := 
sorry

end convex_hexagon_possibilities_l53_53139


namespace find_x_l53_53329

theorem find_x (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 40) : x = 32 :=
by
  sorry

end find_x_l53_53329


namespace positive_integer_solutions_l53_53260

theorem positive_integer_solutions
  (x : ℤ) :
  (5 + 3 * x < 13) ∧ ((x + 2) / 3 - (x - 1) / 2 <= 2) →
  (x = 1 ∨ x = 2) :=
by
  sorry

end positive_integer_solutions_l53_53260


namespace chocolate_bars_cost_l53_53375

variable (n : ℕ) (c : ℕ)

-- Jessica's purchase details
def gummy_bears_packs := 10
def gummy_bears_cost_per_pack := 2
def chocolate_chips_bags := 20
def chocolate_chips_cost_per_bag := 5

-- Calculated costs
def total_gummy_bears_cost := gummy_bears_packs * gummy_bears_cost_per_pack
def total_chocolate_chips_cost := chocolate_chips_bags * chocolate_chips_cost_per_bag

-- Total cost
def total_cost := 150

-- Remaining cost for chocolate bars
def remaining_cost_for_chocolate_bars := total_cost - (total_gummy_bears_cost + total_chocolate_chips_cost)

theorem chocolate_bars_cost (h : remaining_cost_for_chocolate_bars = n * c) : remaining_cost_for_chocolate_bars = 30 :=
by
  sorry

end chocolate_bars_cost_l53_53375


namespace initial_sale_price_percent_l53_53719

theorem initial_sale_price_percent (P S : ℝ) (h1 : S * 0.90 = 0.63 * P) :
  S = 0.70 * P :=
by
  sorry

end initial_sale_price_percent_l53_53719


namespace real_part_is_neg4_l53_53811

def real_part_of_z (z : ℂ) : ℝ :=
  z.re

theorem real_part_is_neg4 (i : ℂ) (h : i^2 = -1) :
  real_part_of_z ((3 + 4 * i) * i) = -4 := by
  sorry

end real_part_is_neg4_l53_53811


namespace max_girls_with_five_boys_l53_53938

theorem max_girls_with_five_boys : 
  ∃ n : ℕ, n = 20 ∧ ∀ (boys : Fin 5 → ℝ × ℝ), 
  (∃ (girls : Fin n → ℝ × ℝ),
  (∀ i : Fin n, ∃ j k : Fin 5, j ≠ k ∧ dist (girls i) (boys j) = 5 ∧ dist (girls i) (boys k) = 5)) :=
sorry

end max_girls_with_five_boys_l53_53938


namespace total_right_handed_players_l53_53306

theorem total_right_handed_players
  (total_players : ℕ)
  (total_throwers : ℕ)
  (left_handed_throwers_perc : ℕ)
  (right_handed_thrower_runs : ℕ)
  (left_handed_thrower_runs : ℕ)
  (total_runs : ℕ)
  (batsmen_to_allrounders_run_ratio : ℕ)
  (proportion_left_right_non_throwers : ℕ)
  (left_handed_non_thrower_runs : ℕ)
  (left_handed_batsmen_eq_allrounders : Prop)
  (left_handed_throwers : ℕ)
  (right_handed_throwers : ℕ)
  (total_right_handed_thrower_runs : ℕ)
  (total_left_handed_thrower_runs : ℕ)
  (total_throwers_runs : ℕ)
  (total_non_thrower_runs : ℕ)
  (allrounder_runs : ℕ)
  (batsmen_runs : ℕ)
  (left_handed_batsmen : ℕ)
  (left_handed_allrounders : ℕ)
  (total_left_handed_non_throwers : ℕ)
  (right_handed_non_throwers : ℕ)
  (total_right_handed_players : ℕ) :
  total_players = 120 →
  total_throwers = 55 →
  left_handed_throwers_perc = 20 →
  right_handed_thrower_runs = 25 →
  left_handed_thrower_runs = 30 →
  total_runs = 3620 →
  batsmen_to_allrounders_run_ratio = 2 →
  proportion_left_right_non_throwers = 5 →
  left_handed_non_thrower_runs = 720 →
  left_handed_batsmen_eq_allrounders →
  left_handed_throwers = total_throwers * left_handed_throwers_perc / 100 →
  right_handed_throwers = total_throwers - left_handed_throwers →
  total_right_handed_thrower_runs = right_handed_throwers * right_handed_thrower_runs →
  total_left_handed_thrower_runs = left_handed_throwers * left_handed_thrower_runs →
  total_throwers_runs = total_right_handed_thrower_runs + total_left_handed_thrower_runs →
  total_non_thrower_runs = total_runs - total_throwers_runs →
  allrounder_runs = total_non_thrower_runs / (batsmen_to_allrounders_run_ratio + 1) →
  batsmen_runs = batsmen_to_allrounders_run_ratio * allrounder_runs →
  left_handed_batsmen = left_handed_non_thrower_runs / (left_handed_thrower_runs * 2) →
  left_handed_allrounders = left_handed_non_thrower_runs / (left_handed_thrower_runs * 2) →
  total_left_handed_non_throwers = left_handed_batsmen + left_handed_allrounders →
  right_handed_non_throwers = total_left_handed_non_throwers * proportion_left_right_non_throwers →
  total_right_handed_players = right_handed_throwers + right_handed_non_throwers →
  total_right_handed_players = 164 :=
by sorry

end total_right_handed_players_l53_53306


namespace bananas_per_chimp_per_day_l53_53960

theorem bananas_per_chimp_per_day (total_chimps total_bananas : ℝ) (h_chimps : total_chimps = 45) (h_bananas : total_bananas = 72) :
  total_bananas / total_chimps = 1.6 :=
by
  rw [h_chimps, h_bananas]
  norm_num

end bananas_per_chimp_per_day_l53_53960


namespace total_sales_l53_53678

theorem total_sales (T : ℝ) (h1 : (2 / 5) * T = (2 / 5) * T) (h2 : (3 / 5) * T = 48) : T = 80 :=
by
  -- added sorry to skip proofs as per the requirement
  sorry

end total_sales_l53_53678


namespace correct_graph_is_C_l53_53323

-- Define the years and corresponding remote work percentages
def percentages : List (ℕ × ℝ) := [
  (1960, 0.1),
  (1970, 0.15),
  (1980, 0.12),
  (1990, 0.25),
  (2000, 0.4)
]

-- Define the property of the graph trend
def isCorrectGraph (p : List (ℕ × ℝ)) : Prop :=
  p = [
    (1960, 0.1),
    (1970, 0.15),
    (1980, 0.12),
    (1990, 0.25),
    (2000, 0.4)
  ]

-- State the theorem
theorem correct_graph_is_C : isCorrectGraph percentages = True :=
  sorry

end correct_graph_is_C_l53_53323


namespace desk_height_l53_53715

variables (h l w : ℝ)

theorem desk_height
  (h_eq_2l_50 : h + 2 * l = 50)
  (h_eq_2w_40 : h + 2 * w = 40)
  (l_minus_w_eq_5 : l - w = 5) :
  h = 30 :=
by {
  sorry
}

end desk_height_l53_53715


namespace tan_neg_five_pi_over_four_l53_53679

theorem tan_neg_five_pi_over_four : Real.tan (-5 * Real.pi / 4) = -1 :=
  sorry

end tan_neg_five_pi_over_four_l53_53679


namespace total_amount_spent_l53_53622

theorem total_amount_spent (tax_paid : ℝ) (tax_rate : ℝ) (tax_free_cost : ℝ) (total_spent : ℝ) :
  tax_paid = 30 → tax_rate = 0.06 → tax_free_cost = 19.7 →
  total_spent = 30 / 0.06 + 19.7 :=
by
  -- Definitions for assumptions
  intro h1 h2 h3
  -- Skip the proof here
  sorry

end total_amount_spent_l53_53622


namespace rook_placements_5x5_l53_53407

/-- The number of ways to place five distinct rooks on a 
  5x5 chess board such that each column and row of the 
  board contains exactly one rook is 120. -/
theorem rook_placements_5x5 : 
  ∃! (f : Fin 5 → Fin 5), Function.Bijective f :=
by
  sorry

end rook_placements_5x5_l53_53407


namespace sugar_theft_problem_l53_53116

-- Define the statements by Gercoginya and the Cook
def gercoginya_statement := "The cook did not steal the sugar"
def cook_statement := "The sugar was stolen by Gercoginya"

-- Define the thief and truth/lie conditions
def thief_lies (x: String) : Prop := x = "The cook stole the sugar"
def other_truth_or_lie (x y: String) : Prop := x = "The sugar was stolen by Gercoginya" ∨ x = "The sugar was not stolen by Gercoginya"

-- The main proof problem to be solved
theorem sugar_theft_problem : 
  ∃ thief : String, 
    (thief = "cook" ∧ thief_lies gercoginya_statement ∧ other_truth_or_lie cook_statement gercoginya_statement) ∨ 
    (thief = "gercoginya" ∧ thief_lies cook_statement ∧ other_truth_or_lie gercoginya_statement cook_statement) :=
sorry

end sugar_theft_problem_l53_53116


namespace shaded_region_area_l53_53645

theorem shaded_region_area (r : ℝ) (n : ℕ) (shaded_area : ℝ) (h_r : r = 3) (h_n : n = 6) :
  shaded_area = 27 * Real.pi - 54 := by
  sorry

end shaded_region_area_l53_53645


namespace X_is_N_l53_53052

theorem X_is_N (X : Set ℕ) (h_nonempty : ∃ x, x ∈ X)
  (h_condition1 : ∀ x ∈ X, 4 * x ∈ X)
  (h_condition2 : ∀ x ∈ X, Nat.floor (Real.sqrt x) ∈ X) : 
  X = Set.univ := 
sorry

end X_is_N_l53_53052


namespace snowballs_made_by_brother_l53_53982

/-- Janet makes 50 snowballs and her brother makes the remaining snowballs. Janet made 25% of the total snowballs. 
    Prove that her brother made 150 snowballs. -/
theorem snowballs_made_by_brother (total_snowballs : ℕ) (janet_snowballs : ℕ) (fraction_janet : ℚ)
  (h1 : janet_snowballs = 50) (h2 : fraction_janet = 25 / 100) (h3 : janet_snowballs = fraction_janet * total_snowballs) :
  total_snowballs - janet_snowballs = 150 :=
by
  sorry

end snowballs_made_by_brother_l53_53982


namespace amanda_car_round_trip_time_l53_53757

theorem amanda_car_round_trip_time :
  let bus_time := 40
  let bus_distance := 120
  let detour := 15
  let reduced_time := 5
  let amanda_trip_one_way_time := bus_time - reduced_time
  let amanda_round_trip_distance := (bus_distance * 2) + (detour * 2)
  let required_time := amanda_round_trip_distance * amanda_trip_one_way_time / bus_distance
  required_time = 79 :=
by
  sorry

end amanda_car_round_trip_time_l53_53757


namespace find_x_for_equation_l53_53561

def f (x : ℝ) : ℝ := 2 * x - 3

theorem find_x_for_equation : (2 * f x - 21 = f (x - 4)) ↔ (x = 8) :=
by
  sorry

end find_x_for_equation_l53_53561


namespace find_a4_b4_c4_l53_53766

variables {a b c : ℝ}

theorem find_a4_b4_c4 (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 0.1) : a^4 + b^4 + c^4 = 0.005 :=
sorry

end find_a4_b4_c4_l53_53766


namespace wxyz_sum_l53_53484

noncomputable def wxyz (w x y z : ℕ) := 2^w * 3^x * 5^y * 7^z

theorem wxyz_sum (w x y z : ℕ) (h : wxyz w x y z = 1260) : w + 2 * x + 3 * y + 4 * z = 13 :=
sorry

end wxyz_sum_l53_53484


namespace matrix_power_101_l53_53999

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![1, 0, 0],
  ![0, 0, 1],
  ![0, 1, 0]
]

theorem matrix_power_101 :
  B ^ (101 : ℕ) = B := sorry

end matrix_power_101_l53_53999


namespace magnitude_correct_l53_53240

open Real

noncomputable def magnitude_of_vector_addition
  (a b : ℝ × ℝ)
  (theta : ℝ)
  (ha : a = (1, 1))
  (hb : ‖b‖ = 2)
  (h_angle : theta = π / 4) : ℝ :=
  ‖3 • a + b‖

theorem magnitude_correct 
  (a b : ℝ × ℝ)
  (theta : ℝ)
  (ha : a = (1, 1))
  (hb : ‖b‖ = 2)
  (h_angle : theta = π / 4) :
  magnitude_of_vector_addition a b theta ha hb h_angle = sqrt 34 :=
sorry

end magnitude_correct_l53_53240


namespace find_num_large_envelopes_l53_53644

def numLettersInSmallEnvelopes : Nat := 20
def totalLetters : Nat := 150
def totalLettersInMediumLargeEnvelopes := totalLetters - numLettersInSmallEnvelopes -- 130
def lettersPerLargeEnvelope : Nat := 5
def lettersPerMediumEnvelope : Nat := 3
def numLargeEnvelopes (L : Nat) : Prop := 5 * L + 6 * L = totalLettersInMediumLargeEnvelopes

theorem find_num_large_envelopes : ∃ L : Nat, numLargeEnvelopes L ∧ L = 11 := by
  sorry

end find_num_large_envelopes_l53_53644


namespace ironed_clothing_l53_53860

theorem ironed_clothing (shirts_rate pants_rate shirts_hours pants_hours : ℕ)
    (h1 : shirts_rate = 4)
    (h2 : pants_rate = 3)
    (h3 : shirts_hours = 3)
    (h4 : pants_hours = 5) :
    shirts_rate * shirts_hours + pants_rate * pants_hours = 27 := by
  sorry

end ironed_clothing_l53_53860


namespace incorrect_variance_l53_53940

noncomputable def normal_pdf (x : ℝ) : ℝ :=
  (1 / Real.sqrt (2 * Real.pi)) * Real.exp (- (x - 1)^2 / 2)

theorem incorrect_variance :
  (∫ x, normal_pdf x * x^2) - (∫ x, normal_pdf x * x)^2 ≠ 2 := 
sorry

end incorrect_variance_l53_53940


namespace max_quotient_l53_53759

theorem max_quotient (a b : ℕ) 
  (h1 : 400 ≤ a) (h2 : a ≤ 800) 
  (h3 : 400 ≤ b) (h4 : b ≤ 1600) 
  (h5 : a + b ≤ 2000) 
  : b / a ≤ 4 := 
sorry

end max_quotient_l53_53759


namespace du_chin_fraction_of_sales_l53_53402

theorem du_chin_fraction_of_sales :
  let pies := 200
  let price_per_pie := 20
  let remaining_money := 1600
  let total_sales := pies * price_per_pie
  let used_for_ingredients := total_sales - remaining_money
  let fraction_used_for_ingredients := used_for_ingredients / total_sales
  fraction_used_for_ingredients = (3 / 5) := by
    sorry

end du_chin_fraction_of_sales_l53_53402


namespace solve_equation_solve_inequality_l53_53718

-- Defining the first problem
theorem solve_equation (x : ℝ) : 3 * (x - 2) - (1 - 2 * x) = 3 ↔ x = 2 := 
by
  sorry

-- Defining the second problem
theorem solve_inequality (x : ℝ) : (2 * x - 1 < 4 * x + 3) ↔ (x > -2) :=
by
  sorry

end solve_equation_solve_inequality_l53_53718


namespace probability_of_perfect_square_sum_l53_53427

def is_perfect_square (n : ℕ) : Prop :=
  n = 1*1 ∨ n = 2*2 ∨ n = 3*3 ∨ n = 4*4

theorem probability_of_perfect_square_sum :
  let total_outcomes := 64
  let perfect_square_sums := 12
  (perfect_square_sums / total_outcomes : ℚ) = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_l53_53427


namespace nearest_integer_to_sum_l53_53161

theorem nearest_integer_to_sum (x y : ℝ) (h1 : |x| - y = 1) (h2 : |x| * y + x^2 = 2) : Int.ceil (x + y) = 2 :=
sorry

end nearest_integer_to_sum_l53_53161


namespace proof_probability_at_least_one_makes_both_shots_l53_53391

-- Define the shooting percentages for Player A and Player B
def shooting_percentage_A : ℝ := 0.4
def shooting_percentage_B : ℝ := 0.5

-- Define the probability that Player A makes both shots
def prob_A_makes_both_shots : ℝ := shooting_percentage_A * shooting_percentage_A

-- Define the probability that Player B makes both shots
def prob_B_makes_both_shots : ℝ := shooting_percentage_B * shooting_percentage_B

-- Define the probability that neither makes both shots
def prob_neither_makes_both_shots : ℝ := (1 - prob_A_makes_both_shots) * (1 - prob_B_makes_both_shots)

-- Define the probability that at least one of them makes both shots
def prob_at_least_one_makes_both_shots : ℝ := 1 - prob_neither_makes_both_shots

-- Prove that the probability that at least one of them makes both shots is 0.37
theorem proof_probability_at_least_one_makes_both_shots :
  prob_at_least_one_makes_both_shots = 0.37 :=
sorry

end proof_probability_at_least_one_makes_both_shots_l53_53391


namespace correct_equation_l53_53361

-- Definitions of the conditions
def january_turnover (T : ℝ) : Prop := T = 36
def march_turnover (T : ℝ) : Prop := T = 48
def average_monthly_growth_rate (x : ℝ) : Prop := True

-- The goal to be proved
theorem correct_equation (T_jan T_mar : ℝ) (x : ℝ) 
  (h_jan : january_turnover T_jan) 
  (h_mar : march_turnover T_mar) 
  (h_growth : average_monthly_growth_rate x) : 
  36 * (1 + x)^2 = 48 :=
sorry

end correct_equation_l53_53361


namespace households_with_dvd_player_l53_53312

noncomputable def numHouseholds : ℕ := 100
noncomputable def numWithCellPhone : ℕ := 90
noncomputable def numWithMP3Player : ℕ := 55
noncomputable def greatestWithAllThree : ℕ := 55 -- maximum x
noncomputable def differenceX_Y : ℕ := 25 -- x - y = 25

def numberOfDVDHouseholds : ℕ := 15

theorem households_with_dvd_player : ∀ (D : ℕ),
  D + 25 - D = 55 - 20 →
  D = numberOfDVDHouseholds :=
by
  intro D h
  sorry

end households_with_dvd_player_l53_53312


namespace stones_on_one_side_l53_53666

theorem stones_on_one_side (total_perimeter_stones : ℕ) (h : total_perimeter_stones = 84) :
  ∃ s : ℕ, 4 * s - 4 = total_perimeter_stones ∧ s = 22 :=
by
  use 22
  sorry

end stones_on_one_side_l53_53666


namespace sam_spent_136_96_l53_53867

def glove_original : Real := 35
def glove_discount : Real := 0.20
def baseball_price : Real := 15
def bat_original : Real := 50
def bat_discount : Real := 0.10
def cleats_price : Real := 30
def cap_price : Real := 10
def tax_rate : Real := 0.07

def total_spent (glove_original : Real) (glove_discount : Real) (baseball_price : Real) (bat_original : Real) (bat_discount : Real) (cleats_price : Real) (cap_price : Real) (tax_rate : Real) : Real :=
  let glove_price := glove_original - (glove_discount * glove_original)
  let bat_price := bat_original - (bat_discount * bat_original)
  let total_before_tax := glove_price + baseball_price + bat_price + cleats_price + cap_price
  let tax_amount := total_before_tax * tax_rate
  total_before_tax + tax_amount

theorem sam_spent_136_96 :
  total_spent glove_original glove_discount baseball_price bat_original bat_discount cleats_price cap_price tax_rate = 136.96 :=
sorry

end sam_spent_136_96_l53_53867


namespace perfect_square_expression_l53_53568

theorem perfect_square_expression (p : ℝ) (h : p = 0.28) : 
  (12.86 * 12.86 + 12.86 * p + 0.14 * 0.14) = (12.86 + 0.14) * (12.86 + 0.14) :=
by 
  -- proof goes here
  sorry

end perfect_square_expression_l53_53568


namespace percentage_decrease_l53_53158

theorem percentage_decrease (x y z : ℝ) (h1 : x = 1.30 * y) (h2 : x = 0.65 * z) : 
  ((z - y) / z) * 100 = 50 :=
by
  sorry

end percentage_decrease_l53_53158


namespace simplify_expression_l53_53931

theorem simplify_expression (x : ℝ) (h1 : x ≠ 0) (h2 : 1 - x ≠ 0) :
  (1 - x) / x / ((1 - x) / x^2) = x := 
by 
  sorry

end simplify_expression_l53_53931


namespace ex3_solutions_abs_eq_l53_53763

theorem ex3_solutions_abs_eq (a : ℝ) : (∃ x1 x2 x3 x4 : ℝ, 
        2 * abs (abs (x1 - 1) - 3) = a ∧ 
        2 * abs (abs (x2 - 1) - 3) = a ∧ 
        2 * abs (abs (x3 - 1) - 3) = a ∧ 
        2 * abs (abs (x4 - 1) - 3) = a ∧ 
        x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ (x1 = x4 ∨ x2 = x4 ∨ x3 = x4)) ↔ a = 6 :=
by
    sorry

end ex3_solutions_abs_eq_l53_53763


namespace train_speed_correct_l53_53871

/-- Define the length of the train in meters -/
def length_train : ℝ := 120

/-- Define the length of the bridge in meters -/
def length_bridge : ℝ := 160

/-- Define the time taken to pass the bridge in seconds -/
def time_taken : ℝ := 25.2

/-- Define the expected speed of the train in meters per second -/
def expected_speed : ℝ := 11.1111

/-- Prove that the speed of the train is 11.1111 meters per second given conditions -/
theorem train_speed_correct :
  (length_train + length_bridge) / time_taken = expected_speed :=
by
  sorry

end train_speed_correct_l53_53871


namespace max_area_of_rectangular_pen_l53_53793

-- Define the perimeter and derive the formula for the area
def perimeter := 60
def half_perimeter := perimeter / 2
def area (x : ℝ) := x * (half_perimeter - x)

-- Statement of the problem: prove the maximum area is 225 square feet
theorem max_area_of_rectangular_pen : ∃ x : ℝ, 0 ≤ x ∧ x ≤ half_perimeter ∧ area x = 225 := 
sorry

end max_area_of_rectangular_pen_l53_53793


namespace integer_divisibility_l53_53006

theorem integer_divisibility (m n : ℕ) (hm : m > 1) (hn : n > 1) (h1 : n ∣ 4^m - 1) (h2 : 2^m ∣ n - 1) : n = 2^m + 1 :=
by sorry

end integer_divisibility_l53_53006


namespace math_proof_problem_l53_53350

noncomputable def M : ℝ :=
  let x := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / (Real.sqrt (Real.sqrt 7 + 2))
  let y := Real.sqrt (5 - 2 * Real.sqrt 6)
  x - y

theorem math_proof_problem :
  M = (Real.sqrt 57 - 6 * Real.sqrt 6 + 4) / 3 :=
by
  sorry

end math_proof_problem_l53_53350


namespace apples_for_juice_is_correct_l53_53269

noncomputable def apples_per_year : ℝ := 8 -- 8 million tons
noncomputable def percentage_mixed : ℝ := 0.30 -- 30%
noncomputable def remaining_apples := apples_per_year * (1 - percentage_mixed) -- Apples after mixed
noncomputable def percentage_for_juice : ℝ := 0.60 -- 60%
noncomputable def apples_for_juice := remaining_apples * percentage_for_juice -- Apples for juice

theorem apples_for_juice_is_correct :
  apples_for_juice = 3.36 :=
by
  sorry

end apples_for_juice_is_correct_l53_53269


namespace inequality_holds_l53_53702

theorem inequality_holds (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) : 
  ((2 + x)/(1 + x))^2 + ((2 + y)/(1 + y))^2 ≥ 9/2 := 
sorry

end inequality_holds_l53_53702


namespace average_interest_rate_l53_53809

theorem average_interest_rate
  (x : ℝ)
  (h₀ : 0 ≤ x)
  (h₁ : x ≤ 5000)
  (h₂ : 0.05 * x = 0.03 * (5000 - x)) :
  (0.05 * x + 0.03 * (5000 - x)) / 5000 = 0.0375 :=
by
  sorry

end average_interest_rate_l53_53809


namespace triangle_area_l53_53357

-- Define the points P, Q, R and the conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def PQR_right_triangle (P Q R : Point) : Prop := 
  (P.x - R.x)^2 + (P.y - R.y)^2 = 24^2 ∧  -- Length PR
  (Q.x - R.x)^2 + (Q.y - R.y)^2 = 73^2 ∧  -- Length RQ
  (P.x - Q.x)^2 + (P.y - Q.y)^2 = 75^2 ∧  -- Hypotenuse PQ
  (P.y = 3 * P.x + 4) ∧                   -- Median through P
  (Q.y = -Q.x + 5)                        -- Median through Q


noncomputable def area (P Q R : Point) : ℝ := 
  0.5 * abs (P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y))

theorem triangle_area (P Q R : Point) (h : PQR_right_triangle P Q R) : 
  area P Q R = 876 :=
sorry

end triangle_area_l53_53357


namespace present_condition_l53_53943

variable {α : Type} [Finite α]

-- We will represent children as members of a type α and assume there are precisely 3n children.
variable (n : ℕ) (h_odd : odd n) [h : Fintype α] (card_3n : Fintype.card α = 3 * n)

noncomputable def makes_present_to (A B : α) : α := sorry -- Create a function that maps pairs of children to exactly one child.

theorem present_condition : ∀ (A B C : α), makes_present_to A B = C → makes_present_to A C = B :=
sorry

end present_condition_l53_53943


namespace number_of_impossible_d_vals_is_infinite_l53_53154

theorem number_of_impossible_d_vals_is_infinite
  (t_1 t_2 s d : ℕ)
  (h1 : 2 * t_1 + t_2 - 4 * s = 4041)
  (h2 : t_1 = s + 2 * d)
  (h3 : t_2 = s + d)
  (h4 : 4 * s > 0) :
  ∀ n : ℕ, n ≠ 808 * 5 ↔ ∃ d, d > 0 ∧ d ≠ n :=
sorry

end number_of_impossible_d_vals_is_infinite_l53_53154


namespace sum_last_two_digits_15_pow_25_plus_5_pow_25_mod_100_l53_53090

theorem sum_last_two_digits_15_pow_25_plus_5_pow_25_mod_100 : 
  (15^25 + 5^25) % 100 = 0 := 
by
  sorry

end sum_last_two_digits_15_pow_25_plus_5_pow_25_mod_100_l53_53090


namespace function_property_l53_53827

theorem function_property 
  (f : ℝ → ℝ) 
  (hf : ∀ x, x ≠ 0 → f (1 - 2 * x) = (1 - x^2) / (x^2)) 
  : 
  (f (1 / 2) = 15) ∧
  (∀ x, x ≠ 1 → f (x) = 4 / (x - 1)^2 - 1) ∧
  (∀ x, x ≠ 0 → x ≠ 1 → f (1 / x) = 4 * x^2 / (x - 1)^2 - 1) :=
by {
  sorry
}

end function_property_l53_53827


namespace sufficient_not_necessary_condition_l53_53168

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x = -1 → x^2 = 1) ∧ ¬(x^2 = 1 → x = -1) :=
by
  sorry

end sufficient_not_necessary_condition_l53_53168


namespace simplify_fraction_120_1800_l53_53078

theorem simplify_fraction_120_1800 :
  (120 : ℚ) / 1800 = (1 : ℚ) / 15 := by
  sorry

end simplify_fraction_120_1800_l53_53078


namespace tomTotalWeightMoved_is_525_l53_53992

-- Tom's weight
def tomWeight : ℝ := 150

-- Weight in each hand
def weightInEachHand : ℝ := 1.5 * tomWeight

-- Weight vest
def weightVest : ℝ := 0.5 * tomWeight

-- Total weight moved
def totalWeightMoved : ℝ := (weightInEachHand * 2) + weightVest

theorem tomTotalWeightMoved_is_525 : totalWeightMoved = 525 := by
  sorry

end tomTotalWeightMoved_is_525_l53_53992


namespace symmetric_about_line_5pi12_l53_53040

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem symmetric_about_line_5pi12 :
  ∀ x : ℝ, f (5 * Real.pi / 12 - x) = f (5 * Real.pi / 12 + x) :=
by
  intros x
  sorry

end symmetric_about_line_5pi12_l53_53040


namespace competition_results_l53_53639

variables (x : ℝ) (freq1 freq3 freq4 freq5 freq2 : ℝ)

/-- Axiom: Given frequencies of groups and total frequency, determine the total number of participants and the probability of an excellent score -/
theorem competition_results :
  freq1 = 0.30 ∧
  freq3 = 0.15 ∧
  freq4 = 0.10 ∧
  freq5 = 0.05 ∧
  freq2 = 40 / x ∧
  (freq1 + freq2 + freq3 + freq4 + freq5 = 1) ∧
  (x * freq2 = 40) →
  x = 100 ∧ (freq4 + freq5 = 0.15) := sorry

end competition_results_l53_53639


namespace mary_income_percentage_more_than_tim_l53_53945

variables (J T M : ℝ)
-- Define the conditions
def condition1 := T = 0.5 * J -- Tim's income is 50% less than Juan's
def condition2 := M = 0.8 * J -- Mary's income is 80% of Juan's

-- Define the theorem stating the question and the correct answer
theorem mary_income_percentage_more_than_tim (J T M : ℝ) 
  (h1 : T = 0.5 * J) 
  (h2 : M = 0.8 * J) : 
  (M - T) / T * 100 = 60 := 
  by sorry

end mary_income_percentage_more_than_tim_l53_53945


namespace triangle_has_angle_45_l53_53544

theorem triangle_has_angle_45
  (A B C : ℝ)
  (h1 : A + B + C = 180)
  (h2 : B + C = 3 * A) :
  A = 45 :=
by
  sorry

end triangle_has_angle_45_l53_53544


namespace minimum_value_expression_l53_53826

theorem minimum_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ m, (∀ x y, x > 0 ∧ y > 0 → (x + y) * (1/x + 4/y) ≥ m) ∧ m = 9 :=
sorry

end minimum_value_expression_l53_53826


namespace perpendicular_lines_b_eq_neg_six_l53_53341

theorem perpendicular_lines_b_eq_neg_six
    (b : ℝ) :
    (∀ x y : ℝ, 3 * y + 2 * x - 4 = 0 → y = (-2/3) * x + 4/3) →
    (∀ x y : ℝ, 4 * y + b * x - 6 = 0 → y = (-b/4) * x + 3/2) →
    - (2/3) * (-b/4) = -1 →
    b = -6 := 
sorry

end perpendicular_lines_b_eq_neg_six_l53_53341


namespace number_of_boys_in_second_grade_l53_53927

-- conditions definition
variables (B : ℕ) (G2 : ℕ := 11) (G3 : ℕ := 2 * (B + G2)) (total : ℕ := B + G2 + G3)

-- mathematical statement to be proved
theorem number_of_boys_in_second_grade : total = 93 → B = 20 :=
by
  -- omitting the proof
  intro h_total
  sorry

end number_of_boys_in_second_grade_l53_53927


namespace total_pears_l53_53908

noncomputable def Jason_pears : ℝ := 46
noncomputable def Keith_pears : ℝ := 47
noncomputable def Mike_pears : ℝ := 12
noncomputable def Sarah_pears : ℝ := 32.5
noncomputable def Emma_pears : ℝ := (2 / 3) * Mike_pears
noncomputable def James_pears : ℝ := (2 * Sarah_pears) - 3

theorem total_pears :
  Jason_pears + Keith_pears + Mike_pears + Sarah_pears + Emma_pears + James_pears = 207.5 :=
by
  sorry

end total_pears_l53_53908


namespace correct_calculation_l53_53985

theorem correct_calculation (x a : Real) :
  (3 * x^2 - x^2 ≠ 3) → 
  (-3 * a^2 - 2 * a^2 ≠ -a^2) →
  (x^3 / x ≠ 3) → 
  ((-x)^3 = -x^3) → 
  true :=
by
  intros _ _ _ _
  trivial

end correct_calculation_l53_53985


namespace veronica_flashlight_distance_l53_53816

theorem veronica_flashlight_distance (V F Vel : ℕ) 
  (h1 : F = 3 * V)
  (h2 : Vel = 5 * F - 2000)
  (h3 : Vel = V + 12000) : 
  V = 1000 := 
by {
  sorry 
}

end veronica_flashlight_distance_l53_53816


namespace problem1_l53_53552

theorem problem1 (k : ℝ) : (∃ x : ℝ, k*x^2 + (2*k + 1)*x + (k - 1) = 0) → k ≥ -1/8 := 
sorry

end problem1_l53_53552


namespace rashmi_bus_stop_distance_l53_53797

theorem rashmi_bus_stop_distance
  (T D : ℝ)
  (h1 : 5 * (T + 10/60) = D)
  (h2 : 6 * (T - 10/60) = D) :
  D = 5 :=
by
  sorry

end rashmi_bus_stop_distance_l53_53797


namespace yi_reads_more_than_jia_by_9_pages_l53_53721

-- Define the number of pages in the book
def total_pages : ℕ := 120

-- Define number of pages read per day by Jia and Yi
def pages_per_day_jia : ℕ := 8
def pages_per_day_yi : ℕ := 13

-- Define the number of days in the period
def total_days : ℕ := 7

-- Calculate total pages read by Jia in the given period
def pages_read_by_jia : ℕ := total_days * pages_per_day_jia

-- Calculate the number of reading days by Yi in the given period
def reading_days_yi : ℕ := (total_days / 3) * 2 + (total_days % 3).min 2

-- Calculate total pages read by Yi in the given period
def pages_read_by_yi : ℕ := reading_days_yi * pages_per_day_yi

-- Given all conditions, prove that Yi reads 9 pages more than Jia over the 7-day period
theorem yi_reads_more_than_jia_by_9_pages :
  pages_read_by_yi - pages_read_by_jia = 9 :=
by
  sorry

end yi_reads_more_than_jia_by_9_pages_l53_53721


namespace vasya_birthday_was_thursday_l53_53189

variable (today : String)
variable (tomorrow : String)
variable (day_after_tomorrow : String)
variable (birthday : String)

-- Conditions given in the problem
axiom birthday_not_sunday : birthday ≠ "Sunday"
axiom sunday_day_after_tomorrow : day_after_tomorrow = "Sunday"

-- From the conditions, we need to prove that Vasya's birthday was on Thursday.
theorem vasya_birthday_was_thursday : birthday = "Thursday" :=
by
  -- Fill in the proof here
  sorry

end vasya_birthday_was_thursday_l53_53189


namespace no_prime_divisor_of_form_8k_minus_1_l53_53805

theorem no_prime_divisor_of_form_8k_minus_1 (n : ℕ) (h : 0 < n) :
  ¬ ∃ p k : ℕ, Nat.Prime p ∧ p = 8 * k - 1 ∧ p ∣ (2^n + 1) :=
by
  sorry

end no_prime_divisor_of_form_8k_minus_1_l53_53805


namespace four_digit_numbers_with_property_l53_53790

theorem four_digit_numbers_with_property :
  (∃ N a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                   N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) ∧
  ∀ (N : ℕ), (∃ a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                           N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) →
             ∃ n : ℕ, n = 9 :=
by
  sorry

end four_digit_numbers_with_property_l53_53790


namespace who_threw_at_third_child_l53_53852

-- Definitions based on conditions
def children_count : ℕ := 43

def threw_snowball (i j : ℕ) : Prop :=
∃ k, i = (k % children_count).succ ∧ j = ((k + 1) % children_count).succ

-- Conditions
axiom cond_1 : threw_snowball 1 (1 + 1) -- child 1 threw a snowball at the child who threw a snowball at child 2
axiom cond_2 : threw_snowball 2 (2 + 1) -- child 2 threw a snowball at the child who threw a snowball at child 3
axiom cond_3 : threw_snowball 43 1 -- child 43 threw a snowball at the child who threw a snowball at the first child

-- Question to prove
theorem who_threw_at_third_child : threw_snowball 24 3 :=
sorry

end who_threw_at_third_child_l53_53852


namespace parallel_lines_m_condition_l53_53846

theorem parallel_lines_m_condition (m : ℝ) : 
  (∀ (x y : ℝ), (2 * x - m * y - 1 = 0) ↔ ((m - 1) * x - y + 1 = 0)) → m = 2 :=
by
  sorry

end parallel_lines_m_condition_l53_53846


namespace alloy_gold_content_l53_53364

theorem alloy_gold_content (x : ℝ) (w : ℝ) (p0 p1 : ℝ) (h_w : w = 16)
  (h_p0 : p0 = 0.50) (h_p1 : p1 = 0.80) (h_alloy : x = 24) :
  (p0 * w + x) / (w + x) = p1 :=
by sorry

end alloy_gold_content_l53_53364


namespace rectangle_perimeter_l53_53847

theorem rectangle_perimeter (s : ℝ) (h1 : 4 * s = 180) :
    let length := s
    let width := s / 3
    2 * (length + width) = 120 := 
by
  sorry

end rectangle_perimeter_l53_53847


namespace only_k_equal_1_works_l53_53451

-- Define the first k prime numbers product
def prime_prod (k : ℕ) : ℕ :=
  Nat.recOn k 1 (fun n prod => prod * (Nat.factorial (n + 1) - Nat.factorial n))

-- Define a predicate for being the sum of two positive cubes
def is_sum_of_two_cubes (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ n = a^3 + b^3

-- The theorem statement
theorem only_k_equal_1_works :
  ∀ k : ℕ, (prime_prod k = 2 ↔ k = 1) :=
by
  sorry

end only_k_equal_1_works_l53_53451


namespace polynomial_simplification_l53_53043

theorem polynomial_simplification (x : ℝ) :
    (3 * x - 2) * (5 * x^12 - 3 * x^11 + 4 * x^9 - 2 * x^8)
    = 15 * x^13 - 19 * x^12 + 6 * x^11 + 12 * x^10 - 14 * x^9 - 4 * x^8 := by
  sorry

end polynomial_simplification_l53_53043


namespace value_of_a_l53_53223

theorem value_of_a {a : ℝ} (h : ∀ x y : ℝ, (a * x^2 + 2 * x + 1 = 0 ∧ a * y^2 + 2 * y + 1 = 0) → x = y) : a = 0 ∨ a = 1 := 
  sorry

end value_of_a_l53_53223


namespace expression_value_l53_53379

theorem expression_value
  (x y z : ℝ)
  (hx : x = -5 / 4)
  (hy : y = -3 / 2)
  (hz : z = Real.sqrt 2) :
  -2 * x ^ 3 - y ^ 2 + Real.sin z = 53 / 32 + Real.sin (Real.sqrt 2) :=
by
  rw [hx, hy, hz]
  sorry

end expression_value_l53_53379


namespace correct_set_of_equations_l53_53477

-- Define the digits x and y as integers
def digits (x y : ℕ) := x + y = 8

-- Conditions
def condition_1 (x y : ℕ) := 10*y + x + 18 = 10*x + y

theorem correct_set_of_equations : 
  ∃ (x y : ℕ), digits x y ∧ condition_1 x y :=
sorry

end correct_set_of_equations_l53_53477


namespace AM_GM_problem_l53_53462

theorem AM_GM_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := 
sorry

end AM_GM_problem_l53_53462


namespace mean_of_squares_eq_l53_53620

noncomputable def sum_of_squares (n : ℕ) : ℚ := (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def arithmetic_mean_of_squares (n : ℕ) : ℚ := sum_of_squares n / n

theorem mean_of_squares_eq (n : ℕ) (h : n ≠ 0) : arithmetic_mean_of_squares n = ((n + 1) * (2 * n + 1)) / 6 :=
by
  sorry

end mean_of_squares_eq_l53_53620


namespace min_colors_to_distinguish_keys_l53_53521

def min_colors_needed (n : Nat) : Nat :=
  if n <= 2 then n
  else if n >= 6 then 2
  else 3

theorem min_colors_to_distinguish_keys (n : Nat) :
  (n ≤ 2 → min_colors_needed n = n) ∧
  (3 ≤ n ∧ n ≤ 5 → min_colors_needed n = 3) ∧
  (n ≥ 6 → min_colors_needed n = 2) :=
by
  sorry

end min_colors_to_distinguish_keys_l53_53521


namespace original_price_of_shoes_l53_53975

noncomputable def original_price (final_price : ℝ) (sales_tax : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  final_price / sales_tax / (discount1 * discount2)

theorem original_price_of_shoes :
  original_price 51 1.07 0.40 0.85 = 140.18 := by
    have h_pre_tax_price : 47.66 = 51 / 1.07 := sorry
    have h_price_relation : 47.66 = 0.85 * 0.40 * 140.18 := sorry
    sorry

end original_price_of_shoes_l53_53975


namespace percent_defective_units_shipped_l53_53984

variable (P : Real)
variable (h1 : 0.07 * P = d)
variable (h2 : 0.0035 * P = s)

theorem percent_defective_units_shipped (h1 : 0.07 * P = d) (h2 : 0.0035 * P = s) : 
  (s / d) * 100 = 5 := sorry

end percent_defective_units_shipped_l53_53984


namespace bill_new_profit_percentage_l53_53031

theorem bill_new_profit_percentage 
  (original_SP : ℝ)
  (profit_percent : ℝ)
  (increment : ℝ)
  (CP : ℝ)
  (CP_new : ℝ)
  (SP_new : ℝ)
  (Profit_new : ℝ)
  (new_profit_percent : ℝ) :
  original_SP = 439.99999999999966 →
  profit_percent = 0.10 →
  increment = 28 →
  CP = original_SP / (1 + profit_percent) →
  CP_new = CP * (1 - profit_percent) →
  SP_new = original_SP + increment →
  Profit_new = SP_new - CP_new →
  new_profit_percent = (Profit_new / CP_new) * 100 →
  new_profit_percent = 30 :=
by
  -- sorry to skip the proof
  sorry

end bill_new_profit_percentage_l53_53031


namespace gcd_gx_x_l53_53422

def g (x : ℕ) : ℕ := (5 * x + 3) * (11 * x + 2) * (17 * x + 7) * (3 * x + 8)

theorem gcd_gx_x (x : ℕ) (h : 27720 ∣ x) : Nat.gcd (g x) x = 168 := by
  sorry

end gcd_gx_x_l53_53422


namespace electric_sharpens_more_l53_53747

noncomputable def number_of_pencils_hand_crank : ℕ := 360 / 45
noncomputable def number_of_pencils_electric : ℕ := 360 / 20

theorem electric_sharpens_more : number_of_pencils_electric - number_of_pencils_hand_crank = 10 := by
  sorry

end electric_sharpens_more_l53_53747


namespace sum_at_simple_interest_l53_53428

theorem sum_at_simple_interest (P R : ℝ) (h1 : P * R * 3 / 100 - P * (R + 3) * 3 / 100 = -90) : P = 1000 :=
sorry

end sum_at_simple_interest_l53_53428


namespace greatest_ribbon_length_l53_53257

-- Define lengths of ribbons
def ribbon_lengths : List ℕ := [8, 16, 20, 28]

-- Condition ensures gcd and prime check
def gcd_is_prime (n : ℕ) : Prop :=
  ∃ d : ℕ, (∀ l ∈ ribbon_lengths, d ∣ l) ∧ Prime d ∧ n = d

-- Prove the greatest length that can make the ribbon pieces, with no ribbon left over, is 2
theorem greatest_ribbon_length : ∃ d, gcd_is_prime d ∧ ∀ m, gcd_is_prime m → m ≤ 2 := 
sorry

end greatest_ribbon_length_l53_53257


namespace carrots_not_used_l53_53667

theorem carrots_not_used :
  let total_carrots := 300
  let carrots_before_lunch := (2 / 5) * total_carrots
  let remaining_after_lunch := total_carrots - carrots_before_lunch
  let carrots_by_end_of_day := (3 / 5) * remaining_after_lunch
  remaining_after_lunch - carrots_by_end_of_day = 72
:= by
  sorry

end carrots_not_used_l53_53667


namespace number_of_ways_to_select_books_l53_53531

theorem number_of_ways_to_select_books :
  let bag1 := 4
  let bag2 := 5
  bag1 * bag2 = 20 :=
by
  sorry

end number_of_ways_to_select_books_l53_53531


namespace find_abc_l53_53693

theorem find_abc (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
  (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8) :=
by 
  sorry

end find_abc_l53_53693


namespace pieces_left_l53_53282

def pieces_initial : ℕ := 900
def pieces_used : ℕ := 156

theorem pieces_left : pieces_initial - pieces_used = 744 := by
  sorry

end pieces_left_l53_53282


namespace consecutive_integers_average_and_product_l53_53916

theorem consecutive_integers_average_and_product (n m : ℤ) (hnm : n ≤ m) 
  (h1 : (n + m) / 2 = 20) 
  (h2 : n * m = 391) :  m - n + 1 = 7 :=
  sorry

end consecutive_integers_average_and_product_l53_53916


namespace graph_does_not_pass_first_quadrant_l53_53418

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - 2

theorem graph_does_not_pass_first_quadrant :
  ¬ ∃ x > 0, f x > 0 := by
sorry

end graph_does_not_pass_first_quadrant_l53_53418


namespace clock_angle_7_35_l53_53510

theorem clock_angle_7_35 : 
  let minute_hand_angle := (35 / 60) * 360
  let hour_hand_angle := 7 * 30 + (35 / 60) * 30
  let angle_between := hour_hand_angle - minute_hand_angle
  angle_between = 17.5 := by
sorry

end clock_angle_7_35_l53_53510


namespace truck_distance_l53_53850

theorem truck_distance :
  let a1 := 8
  let d := 9
  let n := 40
  let an := a1 + (n - 1) * d
  let S_n := n / 2 * (a1 + an)
  S_n = 7340 :=
by
  sorry

end truck_distance_l53_53850


namespace min_empty_squares_eq_nine_l53_53588

-- Definition of the problem conditions
def chessboard_size : ℕ := 9
def total_squares : ℕ := chessboard_size * chessboard_size
def number_of_white_squares : ℕ := 4 * chessboard_size
def number_of_black_squares : ℕ := 5 * chessboard_size
def minimum_number_of_empty_squares : ℕ := number_of_black_squares - number_of_white_squares

-- Theorem to prove minimum number of empty squares
theorem min_empty_squares_eq_nine :
  minimum_number_of_empty_squares = 9 :=
by
  -- Placeholder for the proof
  sorry

end min_empty_squares_eq_nine_l53_53588


namespace avg_one_fourth_class_l53_53842

variable (N : ℕ) -- Total number of students

-- Define the average grade for the entire class
def avg_entire_class : ℝ := 84

-- Define the average grade of three fourths of the class
def avg_three_fourths_class : ℝ := 80

-- Statement to prove
theorem avg_one_fourth_class (A : ℝ) (h1 : 1/4 * A + 3/4 * avg_three_fourths_class = avg_entire_class) : 
  A = 96 := 
sorry

end avg_one_fourth_class_l53_53842


namespace rebecca_soda_left_l53_53381

-- Definitions of the conditions
def total_bottles_purchased : ℕ := 3 * 6
def days_in_four_weeks : ℕ := 4 * 7
def total_half_bottles_drinks : ℕ := days_in_four_weeks
def total_whole_bottles_drinks : ℕ := total_half_bottles_drinks / 2

-- The final statement we aim to prove
theorem rebecca_soda_left : 
  total_bottles_purchased - total_whole_bottles_drinks = 4 := 
by
  -- proof is not required as per the guidelines
  sorry

end rebecca_soda_left_l53_53381


namespace sum_smallest_largest_even_integers_l53_53486

theorem sum_smallest_largest_even_integers (n : ℕ) (h_odd : n % 2 = 1) (b z : ℤ)
  (h_mean : z = b + n - 1) : (b + (b + 2 * (n - 1))) = 2 * z :=
by
  sorry

end sum_smallest_largest_even_integers_l53_53486


namespace flour_more_than_sugar_l53_53661

-- Define the conditions.
def sugar_needed : ℕ := 9
def total_flour_needed : ℕ := 14
def salt_needed : ℕ := 40
def flour_already_added : ℕ := 4

-- Define the target proof statement.
theorem flour_more_than_sugar :
  (total_flour_needed - flour_already_added) - sugar_needed = 1 :=
by
  -- sorry is used here to skip the proof.
  sorry

end flour_more_than_sugar_l53_53661


namespace store_sales_correct_l53_53953

def price_eraser_pencil : ℝ := 0.8
def price_regular_pencil : ℝ := 0.5
def price_short_pencil : ℝ := 0.4
def price_mechanical_pencil : ℝ := 1.2
def price_novelty_pencil : ℝ := 1.5

def quantity_eraser_pencil : ℕ := 200
def quantity_regular_pencil : ℕ := 40
def quantity_short_pencil : ℕ := 35
def quantity_mechanical_pencil : ℕ := 25
def quantity_novelty_pencil : ℕ := 15

def total_sales : ℝ :=
  (quantity_eraser_pencil * price_eraser_pencil) +
  (quantity_regular_pencil * price_regular_pencil) +
  (quantity_short_pencil * price_short_pencil) +
  (quantity_mechanical_pencil * price_mechanical_pencil) +
  (quantity_novelty_pencil * price_novelty_pencil)

theorem store_sales_correct : total_sales = 246.5 :=
by sorry

end store_sales_correct_l53_53953


namespace range_of_f_l53_53638

noncomputable def f (x : ℝ) : ℝ := 3^(x - 2)

theorem range_of_f : Set.Icc 1 9 = {y : ℝ | ∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ f x = y} :=
by
  sorry

end range_of_f_l53_53638


namespace compute_ab_l53_53570

namespace MathProof

variable {a b : ℝ}

theorem compute_ab (h1 : a + b = 8) (h2 : a^3 + b^3 = 152) : a * b = 15 := 
by
  sorry

end MathProof

end compute_ab_l53_53570


namespace ratio_of_surface_areas_l53_53289

theorem ratio_of_surface_areas {r R : ℝ} 
  (h : (4/3) * Real.pi * r^3 / ((4/3) * Real.pi * R^3) = 1 / 8) :
  (4 * Real.pi * r^2) / (4 * Real.pi * R^2) = 1 / 4 := 
sorry

end ratio_of_surface_areas_l53_53289


namespace coin_order_correct_l53_53330

-- Define the coins
inductive Coin
| A | B | C | D | E
deriving DecidableEq

open Coin

-- Define the conditions
def covers (x y : Coin) : Prop :=
  (x = A ∧ y = B) ∨
  (x = C ∧ (y = A ∨ y = D)) ∨
  (x = D ∧ y = B) ∨
  (y = E ∧ x = C)

-- Define the order of coins from top to bottom as a list
def coinOrder : List Coin := [C, E, A, D, B]

-- Prove that the order is correct
theorem coin_order_correct :
  ∀ c₁ c₂ : Coin, c₁ ≠ c₂ → List.indexOf c₁ coinOrder < List.indexOf c₂ coinOrder ↔ covers c₁ c₂ :=
by
  sorry

end coin_order_correct_l53_53330


namespace doubling_profit_condition_l53_53957

-- Definitions
def purchase_price : ℝ := 210
def initial_selling_price : ℝ := 270
def initial_items_sold : ℝ := 30
def profit_per_item (selling_price : ℝ) : ℝ := selling_price - purchase_price
def daily_profit (selling_price : ℝ) (items_sold : ℝ) : ℝ := profit_per_item selling_price * items_sold
def increase_in_items_sold_per_yuan (reduction : ℝ) : ℝ := 3 * reduction

-- Condition: Initial daily profit
def initial_daily_profit : ℝ := daily_profit initial_selling_price initial_items_sold

-- Proof problem
theorem doubling_profit_condition (reduction : ℝ) :
  daily_profit (initial_selling_price - reduction) (initial_items_sold + increase_in_items_sold_per_yuan reduction) = 2 * initial_daily_profit :=
sorry

end doubling_profit_condition_l53_53957


namespace sum_of_x_and_y_l53_53111

theorem sum_of_x_and_y (x y : ℚ) (h1 : 1/x + 1/y = 3) (h2 : 1/x - 1/y = -7) : x + y = -3/10 :=
by
  sorry

end sum_of_x_and_y_l53_53111


namespace graph1_higher_than_graph2_l53_53863

theorem graph1_higher_than_graph2 :
  ∀ (x : ℝ), (-x^2 + 2 * x + 3) ≥ (x^2 - 2 * x + 3) :=
by
  intros x
  sorry

end graph1_higher_than_graph2_l53_53863


namespace slope_range_PA2_l53_53599

-- Define the given conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

def A1 : ℝ × ℝ := (-2, 0)
def A2 : ℝ × ℝ := (2, 0)
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.fst P.snd

-- Define the range of the slope of line PA1
def slope_range_PA1 (k_PA1 : ℝ) : Prop := -2 ≤ k_PA1 ∧ k_PA1 ≤ -1

-- Main theorem
theorem slope_range_PA2 (x0 y0 k_PA1 k_PA2 : ℝ) (h1 : on_ellipse (x0, y0)) (h2 : slope_range_PA1 k_PA1) :
  k_PA1 = (y0 / (x0 + 2)) →
  k_PA2 = (y0 / (x0 - 2)) →
  - (3 / 4) = k_PA1 * k_PA2 →
  (3 / 8) ≤ k_PA2 ∧ k_PA2 ≤ (3 / 4) :=
by
  sorry

end slope_range_PA2_l53_53599


namespace find_larger_number_l53_53077

theorem find_larger_number (x y : ℕ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : y = 33 :=
by
  sorry

end find_larger_number_l53_53077


namespace entrepreneurs_not_attending_any_session_l53_53412

theorem entrepreneurs_not_attending_any_session 
  (total_entrepreneurs : ℕ) 
  (digital_marketing_attendees : ℕ) 
  (e_commerce_attendees : ℕ) 
  (both_sessions_attendees : ℕ)
  (h1 : total_entrepreneurs = 40)
  (h2 : digital_marketing_attendees = 22) 
  (h3 : e_commerce_attendees = 18) 
  (h4 : both_sessions_attendees = 8) : 
  total_entrepreneurs - (digital_marketing_attendees + e_commerce_attendees - both_sessions_attendees) = 8 :=
by sorry

end entrepreneurs_not_attending_any_session_l53_53412


namespace pow_mul_eq_add_l53_53169

variable (a : ℝ)

theorem pow_mul_eq_add : a^2 * a^3 = a^5 := 
by 
  sorry

end pow_mul_eq_add_l53_53169


namespace largest_possible_value_of_s_l53_53275

theorem largest_possible_value_of_s (p q r s : ℝ)
  (h₁ : p + q + r + s = 12)
  (h₂ : pq + pr + ps + qr + qs + rs = 24) : 
  s ≤ 3 + 3 * Real.sqrt 5 :=
sorry

end largest_possible_value_of_s_l53_53275


namespace tom_hockey_games_l53_53714

def tom_hockey_games_last_year (games_this_year missed_this_year total_games : Nat) : Nat :=
  total_games - games_this_year

theorem tom_hockey_games :
  ∀ (games_this_year missed_this_year total_games : Nat),
    games_this_year = 4 →
    missed_this_year = 7 →
    total_games = 13 →
    tom_hockey_games_last_year games_this_year total_games = 9 := by
  intros games_this_year missed_this_year total_games h1 h2 h3
  -- The proof steps would go here
  sorry

end tom_hockey_games_l53_53714


namespace expected_left_handed_l53_53736

theorem expected_left_handed (p : ℚ) (n : ℕ) (h : p = 1/6) (hs : n = 300) : n * p = 50 :=
by 
  -- Proof goes here
  sorry

end expected_left_handed_l53_53736


namespace units_digit_base8_l53_53146

theorem units_digit_base8 (a b : ℕ) (h_a : a = 505) (h_b : b = 71) : 
  ((a * b) % 8) = 7 := 
by
  sorry

end units_digit_base8_l53_53146


namespace solve_equation_l53_53855

-- Define the equation to be proven
def equation (x : ℚ) : Prop :=
  (x + 4) / (x - 3) = (x - 2) / (x + 2)

-- State the theorem
theorem solve_equation : equation (-2 / 11) :=
by
  -- Introduce the equation and the solution to be proven
  unfold equation

  -- Simplify the equation to verify the solution
  sorry


end solve_equation_l53_53855


namespace junior_high_ten_total_games_l53_53051

theorem junior_high_ten_total_games :
  let teams := 10
  let conference_games_per_team := 3
  let non_conference_games_per_team := 5
  let pairs_of_teams := Nat.choose teams 2
  let total_conference_games := pairs_of_teams * conference_games_per_team
  let total_non_conference_games := teams * non_conference_games_per_team
  let total_games := total_conference_games + total_non_conference_games
  total_games = 185 :=
by
  sorry

end junior_high_ten_total_games_l53_53051


namespace geometric_sequence_problem_l53_53989

noncomputable def geometric_sum (a q : ℕ) (n : ℕ) : ℕ :=
  a * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_problem (a : ℕ) (q : ℕ) (n : ℕ) (h_q : q = 2) (h_n : n = 4) :
  (geometric_sum a q 4) / (a * q) = 15 / 2 :=
by
  sorry

end geometric_sequence_problem_l53_53989


namespace smallest_n_l53_53024

theorem smallest_n (n : ℕ) (h : 0 < n) (h1 : 813 * n % 30 = 1224 * n % 30) : n = 10 := 
sorry

end smallest_n_l53_53024


namespace triangle_shading_probability_l53_53596

theorem triangle_shading_probability (n_triangles: ℕ) (n_shaded: ℕ) (h1: n_triangles > 4) (h2: n_shaded = 4) (h3: n_triangles = 10) :
  (n_shaded / n_triangles) = 2 / 5 := 
by
  sorry

end triangle_shading_probability_l53_53596


namespace equilateral_triangle_l53_53405

noncomputable def angles_arithmetic_seq (A B C : ℝ) : Prop := B - A = C - B

noncomputable def sides_geometric_seq (a b c : ℝ) : Prop := b / a = c / b

theorem equilateral_triangle 
  (A B C a b c : ℝ) 
  (h_angles : angles_arithmetic_seq A B C) 
  (h_sides : sides_geometric_seq a b c) 
  (h_triangle : A + B + C = π) 
  (h_pos_sides : a > 0 ∧ b > 0 ∧ c > 0) :
  (A = B ∧ B = C) ∧ (a = b ∧ b = c) :=
sorry

end equilateral_triangle_l53_53405


namespace lights_on_bottom_layer_l53_53311

theorem lights_on_bottom_layer
  (a₁ : ℕ)
  (q : ℕ := 3)
  (S₅ : ℕ := 242)
  (n : ℕ := 5)
  (sum_formula : S₅ = (a₁ * (q^n - 1)) / (q - 1)) :
  (a₁ * q^(n-1) = 162) :=
by
  sorry

end lights_on_bottom_layer_l53_53311


namespace john_gym_hours_l53_53653

theorem john_gym_hours :
  (2 * (1 + 1/3)) + (2 * (1 + 1/2)) + (1.5 + 3/4) = 7.92 :=
by
  sorry

end john_gym_hours_l53_53653


namespace area_of_PQRS_l53_53873

noncomputable def length_EF := 6
noncomputable def width_EF := 4

noncomputable def area_PQRS := (length_EF + 6 * Real.sqrt 3) * (width_EF + 4 * Real.sqrt 3)

theorem area_of_PQRS :
  area_PQRS = 60 + 48 * Real.sqrt 3 := by
  sorry

end area_of_PQRS_l53_53873


namespace rachel_took_money_l53_53213

theorem rachel_took_money (x y : ℕ) (h₁ : x = 5) (h₂ : y = 3) : x - y = 2 :=
by {
  sorry
}

end rachel_took_money_l53_53213


namespace coefficient_c_nonzero_l53_53995

-- We are going to define the given polynomial and its conditions
def P (x : ℝ) (a b c d e : ℝ) : ℝ :=
  x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + e

-- Given conditions
def five_x_intercepts (P : ℝ → ℝ) (x1 x2 x3 x4 x5 : ℝ) : Prop :=
  P x1 = 0 ∧ P x2 = 0 ∧ P x3 = 0 ∧ P x4 = 0 ∧ P x5 = 0

def double_root_at_zero (P : ℝ → ℝ) : Prop :=
  P 0 = 0 ∧ deriv P 0 = 0

-- Equivalent proof problem
theorem coefficient_c_nonzero (a b c d e : ℝ)
  (h1 : P 0 a b c d e = 0)
  (h2 : deriv (P · a b c d e) 0 = 0)
  (h3 : ∀ x, P x a b c d e = x^2 * (x - 1) * (x - 2) * (x - 3))
  (h4 : ∀ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0) : 
  c ≠ 0 := 
sorry

end coefficient_c_nonzero_l53_53995


namespace number_of_skew_line_pairs_in_cube_l53_53458

theorem number_of_skew_line_pairs_in_cube : 
  let vertices := 8
  let total_lines := 28
  let sets_of_4_points := Nat.choose 8 4 - 12
  let skew_pairs_per_set := 3
  let number_of_skew_pairs := sets_of_4_points * skew_pairs_per_set
  number_of_skew_pairs = 174 := sorry

end number_of_skew_line_pairs_in_cube_l53_53458


namespace find_equation_of_line_midpoint_find_equation_of_line_vector_l53_53981

-- Definition for Problem 1
def equation_of_line_midpoint (x y : ℝ) : Prop :=
  ∃ l : ℝ → ℝ, (l x = 0 ∧ l 0 = y ∧ (x / (-6) + y / 2 = 1) ∧ l (-3) = 1)

-- Proof Statement for Problem 1
theorem find_equation_of_line_midpoint : equation_of_line_midpoint (-6) 2 :=
sorry

-- Definition for Problem 2
def equation_of_line_vector (x y : ℝ) : Prop :=
  ∃ l : ℝ → ℝ, (l x = 0 ∧ l 0 = y ∧ (y - 1) / (-1) = (x + 3) / (-6) ∧ l (-3) = 1)

-- Proof Statement for Problem 2
theorem find_equation_of_line_vector : equation_of_line_vector (-9) (3 / 2) :=
sorry

end find_equation_of_line_midpoint_find_equation_of_line_vector_l53_53981


namespace trumpet_cost_l53_53623

/-
  Conditions:
  1. Cost of the music tool: $9.98
  2. Cost of the song book: $4.14
  3. Total amount Joan spent at the music store: $163.28

  Prove that the cost of the trumpet is $149.16
-/

theorem trumpet_cost :
  let c_mt := 9.98
  let c_sb := 4.14
  let t_sp := 163.28
  let c_trumpet := t_sp - (c_mt + c_sb)
  c_trumpet = 149.16 :=
by
  sorry

end trumpet_cost_l53_53623


namespace grace_earnings_l53_53076

noncomputable def weekly_charge : ℕ := 300
noncomputable def payment_interval : ℕ := 2
noncomputable def target_weeks : ℕ := 6
noncomputable def target_amount : ℕ := 1800

theorem grace_earnings :
  (target_weeks * weekly_charge = target_amount) → 
  (target_weeks / payment_interval) * (payment_interval * weekly_charge) = target_amount :=
by
  sorry

end grace_earnings_l53_53076


namespace range_of_a_l53_53491

noncomputable def A := {x : ℝ | x^2 - 2*x - 8 < 0}
noncomputable def B := {x : ℝ | x^2 + 2*x - 3 > 0}
noncomputable def C (a : ℝ) := {x : ℝ | x^2 - 3*a*x + 2*a^2 < 0}

theorem range_of_a (a : ℝ) :
  (C a ⊆ A ∩ B) ↔ (1 ≤ a ∧ a ≤ 2 ∨ a = 0) :=
sorry

end range_of_a_l53_53491


namespace parabola_ratio_l53_53057

-- Define the conditions and question as a theorem statement
theorem parabola_ratio
  (V₁ V₃ : ℝ × ℝ)
  (F₁ F₃ : ℝ × ℝ)
  (hV₁ : V₁ = (0, 0))
  (hF₁ : F₁ = (0, 1/8))
  (hV₃ : V₃ = (0, -1/2))
  (hF₃ : F₃ = (0, -1/4)) :
  dist F₁ F₃ / dist V₁ V₃ = 3 / 4 :=
  by
  sorry

end parabola_ratio_l53_53057


namespace partiallyFilledBoxes_l53_53579

/-- Define the number of cards Joe collected -/
def numPokemonCards : Nat := 65
def numMagicCards : Nat := 55
def numYuGiOhCards : Nat := 40

/-- Define the number of cards each full box can hold -/
def pokemonBoxCapacity : Nat := 8
def magicBoxCapacity : Nat := 10
def yuGiOhBoxCapacity : Nat := 12

/-- Define the partially filled boxes for each type -/
def pokemonPartialBox : Nat := numPokemonCards % pokemonBoxCapacity
def magicPartialBox : Nat := numMagicCards % magicBoxCapacity
def yuGiOhPartialBox : Nat := numYuGiOhCards % yuGiOhBoxCapacity

/-- Theorem to prove number of cards in each partially filled box -/
theorem partiallyFilledBoxes :
  pokemonPartialBox = 1 ∧
  magicPartialBox = 5 ∧
  yuGiOhPartialBox = 4 :=
by
  -- proof goes here
  sorry

end partiallyFilledBoxes_l53_53579


namespace correct_statements_l53_53266

/-- The line (3+m)x+4y-3+3m=0 (m ∈ ℝ) always passes through the fixed point (-3, 3) -/
def statement1 (m : ℝ) : Prop :=
  ∀ x y : ℝ, (3 + m) * x + 4 * y - 3 + 3 * m = 0 → (x = -3 ∧ y = 3)

/-- For segment AB with endpoint B at (3,4) and A moving on the circle x²+y²=4,
    the trajectory equation of the midpoint M of segment AB is (x - 3/2)²+(y - 2)²=1 -/
def statement2 : Prop :=
  ∀ x y x1 y1 : ℝ, ((x1, y1) : ℝ × ℝ) ∈ {p | p.1^2 + p.2^2 = 4} → x = (x1 + 3) / 2 → y = (y1 + 4) / 2 → 
    (x - 3 / 2)^2 + (y - 2)^2 = 1

/-- Given M = {(x, y) | y = √(1 - x²)} and N = {(x, y) | y = x + b},
    if M ∩ N ≠ ∅, then b ∈ [-√2, √2] -/
def statement3 (b : ℝ) : Prop :=
  ∃ x y : ℝ, y = Real.sqrt (1 - x^2) ∧ y = x + b → b ∈ [-Real.sqrt 2, Real.sqrt 2]

/-- Given the circle C: (x - b)² + (y - c)² = a² (a > 0, b > 0, c > 0) intersects the x-axis and is
    separate from the y-axis, then the intersection point of the line ax + by + c = 0 and the line
    x + y + 1 = 0 is in the second quadrant -/
def statement4 (a b c : ℝ) : Prop :=
  a > 0 → b > 0 → c > 0 → b > a → a > c →
  ∃ x y : ℝ, (a * x + b * y + c = 0 ∧ x + y + 1 = 0) ∧ x < 0 ∧ y > 0

/-- Among the statements, the correct ones are 1, 2, and 4 -/
theorem correct_statements : 
  (∀ m : ℝ, statement1 m) ∧ statement2 ∧ (∀ b : ℝ, ¬ statement3 b) ∧ 
  (∀ a b c : ℝ, statement4 a b c) :=
by sorry

end correct_statements_l53_53266


namespace harry_weekly_earnings_l53_53497

def dogs_walked_per_day : Nat → Nat
| 1 => 7  -- Monday
| 2 => 12 -- Tuesday
| 3 => 7  -- Wednesday
| 4 => 9  -- Thursday
| 5 => 7  -- Friday
| _ => 0  -- Other days (not relevant for this problem)

def payment_per_dog : Nat := 5

def daily_earnings (day : Nat) : Nat :=
  dogs_walked_per_day day * payment_per_dog

def total_weekly_earnings : Nat :=
  (daily_earnings 1) + (daily_earnings 2) + (daily_earnings 3) +
  (daily_earnings 4) + (daily_earnings 5)

theorem harry_weekly_earnings : total_weekly_earnings = 210 :=
by
  sorry

end harry_weekly_earnings_l53_53497


namespace man_born_in_1936_l53_53291

noncomputable def year_of_birth (x : ℕ) : ℕ :=
  x^2 - 2 * x

theorem man_born_in_1936 :
  ∃ x : ℕ, x < 50 ∧ year_of_birth x < 1950 ∧ year_of_birth x = 1892 :=
by
  sorry

end man_born_in_1936_l53_53291


namespace ratio_of_X_to_Y_l53_53630

theorem ratio_of_X_to_Y (total_respondents : ℕ) (preferred_X : ℕ)
    (h_total : total_respondents = 250)
    (h_X : preferred_X = 200) :
    preferred_X / (total_respondents - preferred_X) = 4 := by
  sorry

end ratio_of_X_to_Y_l53_53630


namespace range_of_a_l53_53093

noncomputable def P (a : ℝ) : Prop :=
∀ x : ℝ, a * x^2 + a * x + 1 > 0

noncomputable def Q (a : ℝ) : Prop :=
(∃ (x y : ℝ), (x^2 / a + y^2 / (a - 3) = 1)) ∧ ∀ (x y : ℝ), (x^2 / a + y^2 / (a - 3) = 1) → (a * (a - 3) < 0)

theorem range_of_a (a : ℝ) (h1 : P a ∨ Q a) (h2 : ¬ (P a ∧ Q a)) : a = 0 ∨ (3 ≤ a ∧ a < 4) := 
sorry

end range_of_a_l53_53093


namespace range_of_a_l53_53103

noncomputable def f (a x : ℝ) : ℝ :=
  x^3 - 3 * a * x^2 + (2 * a + 1) * x

noncomputable def f' (a x : ℝ) : ℝ :=
  3 * x^2 - 6 * a * x + (2 * a + 1)

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f' a x = 0 ∧ ∀ y : ℝ, f' a y ≠ 0) →
  (a > 1 ∨ a < -1 / 3) :=
sorry

end range_of_a_l53_53103


namespace sum_of_squares_inequality_l53_53211

theorem sum_of_squares_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ (1/3)*(a + b + c)^2 := sorry

end sum_of_squares_inequality_l53_53211


namespace binomial_inequality_l53_53029

theorem binomial_inequality (n : ℕ) (x : ℝ) (h : x ≥ -1) : (1 + x)^n ≥ 1 + n * x :=
by sorry

end binomial_inequality_l53_53029


namespace total_simple_interest_l53_53707

theorem total_simple_interest (P R T : ℝ) (hP : P = 6178.846153846154) (hR : R = 0.13) (hT : T = 5) :
    P * R * T = 4011.245192307691 := by
  rw [hP, hR, hT]
  norm_num
  sorry

end total_simple_interest_l53_53707


namespace total_kids_in_camp_l53_53303

-- Definitions from the conditions
variables (X : ℕ)
def kids_going_to_soccer_camp := X / 2
def kids_going_to_soccer_camp_morning := kids_going_to_soccer_camp / 4
def kids_going_to_soccer_camp_afternoon := kids_going_to_soccer_camp - kids_going_to_soccer_camp_morning

-- Given condition that 750 kids are going to soccer camp in the afternoon
axiom h : kids_going_to_soccer_camp_afternoon X = 750

-- The statement to prove that X = 2000
theorem total_kids_in_camp : X = 2000 :=
sorry

end total_kids_in_camp_l53_53303


namespace negation_of_proposition_l53_53135

theorem negation_of_proposition :
  (∀ x : ℝ, 2^x + x^2 > 0) → (∃ x0 : ℝ, 2^x0 + x0^2 ≤ 0) :=
sorry

end negation_of_proposition_l53_53135


namespace arithmetic_sequence_l53_53584

theorem arithmetic_sequence (S : ℕ → ℕ) (h : ∀ n, S n = 3 * n * n) :
  (∃ a d : ℕ, ∀ n : ℕ, S n - S (n - 1) = a + (n - 1) * d) ∧
  (∀ n, S n - S (n - 1) = 6 * n - 3) :=
by
  sorry

end arithmetic_sequence_l53_53584


namespace range_of_a_monotonically_decreasing_l53_53946

noncomputable def f (x a : ℝ) := x^3 - a * x^2 + 1

theorem range_of_a_monotonically_decreasing (a : ℝ) :
  (∀ x y : ℝ, (0 < x ∧ x < 2) ∧ (0 < y ∧ y < 2) → x < y → f x a ≥ f y a) → (a ≥ 3) :=
by
  sorry

end range_of_a_monotonically_decreasing_l53_53946


namespace number_of_men_l53_53354

variable (W M : ℝ)
variable (N_women N_men : ℕ)

theorem number_of_men (h1 : M = 2 * W)
  (h2 : N_women * W * 30 = 21600) :
  (N_men * M * 20 = 14400) → N_men = N_women / 3 :=
by
  sorry

end number_of_men_l53_53354


namespace initial_time_for_train_l53_53608

theorem initial_time_for_train (S : ℝ)
  (length_initial : ℝ := 12 * 15)
  (length_detached : ℝ := 11 * 15)
  (time_detached : ℝ := 16.5)
  (speed_constant : S = length_detached / time_detached) :
  (length_initial / S = 18) :=
by
  sorry

end initial_time_for_train_l53_53608


namespace jake_and_luke_items_l53_53677

theorem jake_and_luke_items :
  ∃ (p j : ℕ), 6 * p + 2 * j ≤ 50 ∧ (∀ (p' : ℕ), 6 * p' + 2 * j ≤ 50 → p' ≤ p) ∧ p + j = 9 :=
by
  sorry

end jake_and_luke_items_l53_53677


namespace find_constant_k_l53_53256

theorem find_constant_k (k : ℝ) :
    -x^2 - (k + 9) * x - 8 = -(x - 2) * (x - 4) → k = -15 := by
  sorry

end find_constant_k_l53_53256


namespace number_of_sides_of_regular_polygon_l53_53433

theorem number_of_sides_of_regular_polygon (P s n : ℕ) (hP : P = 150) (hs : s = 15) (hP_formula : P = n * s) : n = 10 :=
  by {
    -- proof goes here
    sorry
  }

end number_of_sides_of_regular_polygon_l53_53433


namespace binom_60_3_l53_53844

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_l53_53844


namespace red_card_events_l53_53203

-- Definitions based on the conditions
inductive Person
| A | B | C | D

inductive Card
| Red | Black | Blue | White

-- Definition of the events
def event_A_receives_red (distribution : Person → Card) : Prop :=
  distribution Person.A = Card.Red

def event_B_receives_red (distribution : Person → Card) : Prop :=
  distribution Person.B = Card.Red

-- The relationship between the two events
def mutually_exclusive_but_not_opposite (distribution : Person → Card) : Prop :=
  (event_A_receives_red distribution → ¬ event_B_receives_red distribution) ∧
  (event_B_receives_red distribution → ¬ event_A_receives_red distribution)

-- The formal theorem statement
theorem red_card_events (distribution : Person → Card) :
  mutually_exclusive_but_not_opposite distribution :=
sorry

end red_card_events_l53_53203


namespace min_value_expr_l53_53587

theorem min_value_expr (x y z w : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) (hz : -1 < z ∧ z < 1) (hw : -2 < w ∧ w < 2) :
  2 ≤ (1 / ((1 - x) * (1 - y) * (1 - z) * (1 - w / 2)) + 1 / ((1 + x) * (1 + y) * (1 + z) * (1 + w / 2))) :=
sorry

end min_value_expr_l53_53587


namespace greatest_a_no_integral_solution_l53_53273

theorem greatest_a_no_integral_solution (a : ℤ) :
  (∀ x : ℤ, |x + 1| ≥ a - 3 / 2) → a = 1 :=
by
  sorry

end greatest_a_no_integral_solution_l53_53273


namespace total_visitors_three_days_l53_53796

def V_Rachel := 92
def V_prev_day := 419
def V_day_before_prev := 103

theorem total_visitors_three_days : V_Rachel + V_prev_day + V_day_before_prev = 614 := 
by sorry

end total_visitors_three_days_l53_53796


namespace rate_per_kg_grapes_is_70_l53_53249

-- Let G be the rate per kg for the grapes
def rate_per_kg_grapes (G : ℕ) := G

-- Bruce purchased 8 kg of grapes at rate G per kg
def grapes_cost (G : ℕ) := 8 * G

-- Bruce purchased 11 kg of mangoes at the rate of 55 per kg
def mangoes_cost := 11 * 55

-- Bruce paid a total of 1165 to the shopkeeper
def total_paid := 1165

-- The problem: Prove that the rate per kg for the grapes is 70
theorem rate_per_kg_grapes_is_70 : rate_per_kg_grapes 70 = 70 ∧ grapes_cost 70 + mangoes_cost = total_paid := by
  sorry

end rate_per_kg_grapes_is_70_l53_53249


namespace simplify_fraction_l53_53465

-- Define the given variables and their assigned values.
variable (b : ℕ)
variable (b_eq : b = 2)

-- State the theorem we want to prove
theorem simplify_fraction (b : ℕ) (h : b = 2) : 
  15 * b ^ 4 / (75 * b ^ 3) = 2 / 5 :=
by
  -- sorry indicates where the proof would be written.
  sorry

end simplify_fraction_l53_53465


namespace lottery_prob_correct_l53_53755

def possibleMegaBalls : ℕ := 30
def possibleWinnerBalls : ℕ := 49
def drawnWinnerBalls : ℕ := 6

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def winningProbability : ℚ :=
  (1 : ℚ) / possibleMegaBalls * (1 : ℚ) / combination possibleWinnerBalls drawnWinnerBalls

theorem lottery_prob_correct :
  winningProbability = 1 / 419514480 := by
  sorry

end lottery_prob_correct_l53_53755


namespace domain_of_g_eq_l53_53113

noncomputable def g (x : ℝ) : ℝ := (x + 2) / (Real.sqrt (x^2 - 5 * x + 6))

theorem domain_of_g_eq : 
  {x : ℝ | 0 < x^2 - 5 * x + 6} = {x : ℝ | x < 2} ∪ {x : ℝ | 3 < x} :=
by
  sorry

end domain_of_g_eq_l53_53113


namespace solve_for_x_l53_53712

-- Defining the given conditions
def y : ℕ := 6
def lhs (x : ℕ) : ℕ := Nat.pow x y
def rhs : ℕ := Nat.pow 3 12

-- Theorem statement to prove
theorem solve_for_x (x : ℕ) (hypothesis : lhs x = rhs) : x = 9 :=
by sorry

end solve_for_x_l53_53712


namespace negative_square_inequality_l53_53549

theorem negative_square_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 :=
sorry

end negative_square_inequality_l53_53549


namespace max_k_solution_l53_53485

theorem max_k_solution
  (k x y : ℝ)
  (h_pos: 0 < k ∧ 0 < x ∧ 0 < y)
  (h_eq: 5 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) :
  ∃ k, 8*k^3 - 8*k^2 - 7*k = 0 := 
sorry

end max_k_solution_l53_53485


namespace reciprocal_of_neg_2023_l53_53502

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l53_53502


namespace infinite_pairs_natural_numbers_l53_53383

theorem infinite_pairs_natural_numbers :
  ∃ (infinite_pairs : ℕ × ℕ → Prop), (∀ a b : ℕ, infinite_pairs (a, b) ↔ (b ∣ (a^2 + 1) ∧ a ∣ (b^2 + 1))) ∧
    ∀ n : ℕ, ∃ (a b : ℕ), infinite_pairs (a, b) :=
sorry

end infinite_pairs_natural_numbers_l53_53383


namespace solve_for_x_l53_53954

theorem solve_for_x (x : ℝ) (h : 0.009 / x = 0.1) : x = 0.09 :=
sorry

end solve_for_x_l53_53954


namespace interest_rate_proof_l53_53631
noncomputable def interest_rate_B (P : ℝ) (rA : ℝ) (t : ℝ) (gain_B : ℝ) : ℝ := 
  (P * rA * t + gain_B) / (P * t)

theorem interest_rate_proof
  (P : ℝ := 3500)
  (rA : ℝ := 0.10)
  (t : ℝ := 3)
  (gain_B : ℝ := 210) :
  interest_rate_B P rA t gain_B = 0.12 :=
sorry

end interest_rate_proof_l53_53631


namespace bogatyrs_truthful_count_l53_53830

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end bogatyrs_truthful_count_l53_53830


namespace expected_profit_l53_53806

namespace DailyLottery

/-- Definitions for the problem -/

def ticket_cost : ℝ := 2
def first_prize : ℝ := 100
def second_prize : ℝ := 10
def prob_first_prize : ℝ := 0.001
def prob_second_prize : ℝ := 0.1
def prob_no_prize : ℝ := 1 - prob_first_prize - prob_second_prize

/-- Expected profit calculation as a theorem -/

theorem expected_profit :
  (first_prize * prob_first_prize + second_prize * prob_second_prize + 0 * prob_no_prize) - ticket_cost = -0.9 :=
by
  sorry

end DailyLottery

end expected_profit_l53_53806


namespace gage_needs_to_skate_l53_53339

noncomputable def gage_average_skating_time (d1 d2: ℕ) (t1 t2 t8: ℕ) : ℕ :=
  let total_time := (d1 * t1) + (d2 * t2) + t8
  (total_time / (d1 + d2 + 1))

theorem gage_needs_to_skate (t1 t2: ℕ) (d1 d2: ℕ) (avg: ℕ) 
  (t1_minutes: t1 = 80) (t2_minutes: t2 = 105) 
  (days1: d1 = 4) (days2: d2 = 3) (avg_goal: avg = 95) :
  gage_average_skating_time d1 d2 t1 t2 125 = avg :=
by
  sorry

end gage_needs_to_skate_l53_53339


namespace time_to_walk_against_walkway_150_l53_53891

def v_p := 4 / 3
def v_w := 2 - v_p
def distance := 100
def time_against_walkway := distance / (v_p - v_w)

theorem time_to_walk_against_walkway_150 :
  time_against_walkway = 150 := by
  -- Note: Proof goes here (not required)
  sorry

end time_to_walk_against_walkway_150_l53_53891


namespace minimum_value_of_a_l53_53287

-- Define the given condition
axiom a_pos : ℝ → Prop
axiom positive : ∀ (x : ℝ), 0 < x

-- Definition of the equation
def equation (x y a : ℝ) : Prop :=
  (2 * x - y / Real.exp 1) * Real.log (y / x) = x / (a * Real.exp 1)

-- The mathematical statement we need to prove
theorem minimum_value_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) (h_eq : equation x y a) : 
  a ≥ 1 / Real.exp 1 :=
sorry

end minimum_value_of_a_l53_53287


namespace subtract_045_from_3425_l53_53425

theorem subtract_045_from_3425 : 34.25 - 0.45 = 33.8 :=
by sorry

end subtract_045_from_3425_l53_53425


namespace problem1_problem2_l53_53781

-- Definitions of the sets A, B, and C based on conditions given
def setA : Set ℝ := {x | x^2 + 2*x - 8 ≥ 0}
def setB : Set ℝ := {x | Real.sqrt (9 - 3*x) ≤ Real.sqrt (2*x + 19)}
def setC (a : ℝ) : Set ℝ := {x | x^2 + 2*a*x + 2 ≤ 0}

-- Problem (1): Prove values of b and c
theorem problem1 (b c : ℝ) :
  (∀ x, x ∈ (setA ∩ setB) ↔ b*x^2 + 10*x + c ≥ 0) → b = -2 ∧ c = -12 := sorry

-- Universal set definition and its complement
def universalSet : Set ℝ := {x | True}
def complementA : Set ℝ := {x | (x ∉ setA)}

-- Problem (2): Range of a
theorem problem2 (a : ℝ) :
  (setC a ⊆ setB ∪ complementA) → a ∈ Set.Icc (-11/6) (9/4) := sorry

end problem1_problem2_l53_53781


namespace k_range_l53_53201

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  (Real.log x) - x - x * Real.exp (-x) - k

theorem k_range (k : ℝ) : (∀ x > 0, ∃ x > 0, f x k = 0) ↔ k ≤ -1 - (1 / Real.exp 1) :=
sorry

end k_range_l53_53201


namespace find_cos_alpha_l53_53722

variable (α β : ℝ)

-- Conditions
def acute_angles (α β : ℝ) : Prop := 0 < α ∧ α < (Real.pi / 2) ∧ 0 < β ∧ β < (Real.pi / 2)
def cos_alpha_beta : Prop := Real.cos (α + β) = 12 / 13
def cos_2alpha_beta : Prop := Real.cos (2 * α + β) = 3 / 5

-- Main theorem
theorem find_cos_alpha (h1 : acute_angles α β) (h2 : cos_alpha_beta α β) (h3 : cos_2alpha_beta α β) : 
  Real.cos α = 56 / 65 :=
sorry

end find_cos_alpha_l53_53722


namespace bus_rent_proof_l53_53663

theorem bus_rent_proof (r1 r2 : ℝ) (r1_rent_eq : r1 + 2 * r2 = 2800) (r2_mult : r2 = 1.25 * r1) :
  r1 = 800 ∧ r2 = 1000 := 
by
  sorry

end bus_rent_proof_l53_53663


namespace sector_angle_l53_53343

-- Define the conditions
def perimeter (r l : ℝ) : ℝ := 2 * r + l
def arc_length (α r : ℝ) : ℝ := α * r

-- Define the problem statement
theorem sector_angle (perimeter_eq : perimeter 1 l = 4) (arc_length_eq : arc_length α 1 = l) : α = 2 := 
by 
  -- remainder of the proof can be added here 
  sorry

end sector_angle_l53_53343


namespace problem1_problem2_problem3_general_conjecture_l53_53187

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

-- Prove f(0) + f(1) = sqrt(2) / 2
theorem problem1 : f 0 + f 1 = Real.sqrt 2 / 2 := by
  sorry

-- Prove f(-1) + f(2) = sqrt(2) / 2
theorem problem2 : f (-1) + f 2 = Real.sqrt 2 / 2 := by
  sorry

-- Prove f(-2) + f(3) = sqrt(2) / 2
theorem problem3 : f (-2) + f 3 = Real.sqrt 2 / 2 := by
  sorry

-- Prove ∀ x, f(-x) + f(x+1) = sqrt(2) / 2
theorem general_conjecture (x : ℝ) : f (-x) + f (x + 1) = Real.sqrt 2 / 2 := by
  sorry

end problem1_problem2_problem3_general_conjecture_l53_53187


namespace find_ratio_of_b1_b2_l53_53176

variable (a b k a1 a2 b1 b2 : ℝ)
variable (h1 : a1 ≠ 0) (h2 : a2 ≠ 0) (hb1 : b1 ≠ 0) (hb2 : b2 ≠ 0)

noncomputable def inversely_proportional_condition := a1 * b1 = a2 * b2
noncomputable def ratio_condition := a1 / a2 = 3 / 4
noncomputable def difference_condition := b1 - b2 = 5

theorem find_ratio_of_b1_b2 
  (h_inv : inversely_proportional_condition a1 a2 b1 b2)
  (h_rat : ratio_condition a1 a2)
  (h_diff : difference_condition b1 b2) :
  b1 / b2 = 4 / 3 :=
sorry

end find_ratio_of_b1_b2_l53_53176


namespace remainder_of_addition_and_division_l53_53696

theorem remainder_of_addition_and_division :
  (3452179 + 50) % 7 = 4 :=
by
  sorry

end remainder_of_addition_and_division_l53_53696


namespace max_2ab_plus_2bc_sqrt2_l53_53345

theorem max_2ab_plus_2bc_sqrt2 (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a^2 + b^2 + c^2 = 1) :
  2 * a * b + 2 * b * c * Real.sqrt 2 ≤ Real.sqrt 3 :=
sorry

end max_2ab_plus_2bc_sqrt2_l53_53345


namespace conditions_for_unique_solution_l53_53197

noncomputable def is_solution (n p x y z : ℕ) : Prop :=
x + p * y = n ∧ x + y = p^z

def unique_positive_integer_solution (n p : ℕ) : Prop :=
∃! (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ is_solution n p x y z

theorem conditions_for_unique_solution {n p : ℕ} :
  (1 < p) ∧ ((n - 1) % (p - 1) = 0) ∧ ∀ k : ℕ, n ≠ p^k ↔ unique_positive_integer_solution n p :=
sorry

end conditions_for_unique_solution_l53_53197


namespace integer_solution_unique_l53_53593

theorem integer_solution_unique (x y z : ℤ) : x^3 - 2*y^3 - 4*z^3 = 0 → x = 0 ∧ y = 0 ∧ z = 0 :=
by 
  sorry

end integer_solution_unique_l53_53593


namespace total_ticket_cost_is_14_l53_53569

-- Definitions of the ticket costs
def ticket_cost_hat : ℕ := 2
def ticket_cost_stuffed_animal : ℕ := 10
def ticket_cost_yoyo : ℕ := 2

-- Definition of the total ticket cost
def total_ticket_cost : ℕ := ticket_cost_hat + ticket_cost_stuffed_animal + ticket_cost_yoyo

-- Theorem stating the total ticket cost is 14
theorem total_ticket_cost_is_14 : total_ticket_cost = 14 := by
  -- Proof would go here, but sorry is used to skip it
  sorry

end total_ticket_cost_is_14_l53_53569


namespace new_computer_lasts_l53_53128

theorem new_computer_lasts (x : ℕ) 
  (h1 : 600 = 400 + 200)
  (h2 : ∀ y : ℕ, (2 * 200 = 400) → (2 * 3 = 6) → y = 6)
  (h3 : 200 = 600 - 400) :
  x = 6 :=
by
  sorry

end new_computer_lasts_l53_53128


namespace find_b_c_l53_53765

-- Definitions and the problem statement
theorem find_b_c (b c : ℝ) (x1 x2 : ℝ) (h1 : x1 = 1) (h2 : x2 = -2) 
  (h_eq : ∀ x, x^2 - b * x + c = (x - x1) * (x - x2)) :
  b = -1 ∧ c = -2 :=
by
  sorry

end find_b_c_l53_53765


namespace prime_numbers_count_and_sum_l53_53838

-- Definition of prime numbers less than or equal to 20
def prime_numbers_leq_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Proposition stating the number of prime numbers and their sum within 20
theorem prime_numbers_count_and_sum :
  (prime_numbers_leq_20.length = 8) ∧ (prime_numbers_leq_20.sum = 77) := by
  sorry

end prime_numbers_count_and_sum_l53_53838


namespace tan_sub_sin_eq_sq3_div2_l53_53439

noncomputable def tan_60 := Real.tan (Real.pi / 3)
noncomputable def sin_60 := Real.sin (Real.pi / 3)
noncomputable def result := (tan_60 - sin_60)

theorem tan_sub_sin_eq_sq3_div2 : result = Real.sqrt 3 / 2 := 
by
  -- Proof might go here
  sorry

end tan_sub_sin_eq_sq3_div2_l53_53439


namespace unique_combined_friends_count_l53_53001

theorem unique_combined_friends_count 
  (james_friends : ℕ)
  (susan_friends : ℕ)
  (john_multiplier : ℕ)
  (shared_friends : ℕ)
  (maria_shared_friends : ℕ)
  (maria_friends : ℕ)
  (h_james : james_friends = 90)
  (h_susan : susan_friends = 50)
  (h_john : ∃ (john_friends : ℕ), john_friends = john_multiplier * susan_friends ∧ john_multiplier = 4)
  (h_shared : shared_friends = 35)
  (h_maria_shared : maria_shared_friends = 10)
  (h_maria : maria_friends = 80) :
  ∃ (total_unique_friends : ℕ), total_unique_friends = 325 :=
by
  -- Proof is omitted
  sorry

end unique_combined_friends_count_l53_53001


namespace ratio_Smax_Smin_l53_53672

-- Define the area of a cube's diagonal cross-section through BD1
def cross_section_area (a : ℝ) : ℝ := sorry

theorem ratio_Smax_Smin (a : ℝ) (S S_min S_max : ℝ) :
  cross_section_area a = S →
  S_min = (a^2 * Real.sqrt 6) / 2 →
  S_max = a^2 * Real.sqrt 6 →
  S_max / S_min = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end ratio_Smax_Smin_l53_53672


namespace total_shaded_area_of_square_carpet_l53_53717

theorem total_shaded_area_of_square_carpet :
  ∀ (S T : ℝ),
    (9 / S = 3) →
    (S / T = 3) →
    (8 * T^2 + S^2 = 17) :=
by
  intros S T h1 h2
  sorry

end total_shaded_area_of_square_carpet_l53_53717


namespace solution_set_a_eq_1_no_positive_a_for_all_x_l53_53091

-- Define the original inequality for a given a.
def inequality (a x : ℝ) : Prop := |a * x - 1| + |a * x - a| ≥ 2

-- Part 1: For a = 1
theorem solution_set_a_eq_1 :
  {x : ℝ | inequality 1 x } = {x : ℝ | x ≤ 0 ∨ x ≥ 2} :=
sorry

-- Part 2: There is no positive a such that the inequality holds for all x ∈ ℝ
theorem no_positive_a_for_all_x :
  ¬ ∃ a > 0, ∀ x : ℝ, inequality a x :=
sorry

end solution_set_a_eq_1_no_positive_a_for_all_x_l53_53091


namespace discrim_of_quadratic_eqn_l53_53003

theorem discrim_of_quadratic_eqn : 
  let a := 3
  let b := -2
  let c := -1
  b^2 - 4 * a * c = 16 := 
by
  sorry

end discrim_of_quadratic_eqn_l53_53003


namespace remainder_when_sum_divided_by_7_l53_53878

theorem remainder_when_sum_divided_by_7 (a b c : ℕ) (ha : a < 7) (hb : b < 7) (hc : c < 7)
  (h1 : a * b * c ≡ 1 [MOD 7])
  (h2 : 4 * c ≡ 3 [MOD 7])
  (h3 : 5 * b ≡ 4 + b [MOD 7]) :
  (a + b + c) % 7 = 6 := by
  sorry

end remainder_when_sum_divided_by_7_l53_53878


namespace number_of_white_dogs_l53_53321

noncomputable def number_of_brown_dogs : ℕ := 20
noncomputable def number_of_black_dogs : ℕ := 15
noncomputable def total_number_of_dogs : ℕ := 45

theorem number_of_white_dogs : total_number_of_dogs - (number_of_brown_dogs + number_of_black_dogs) = 10 := by
  sorry

end number_of_white_dogs_l53_53321


namespace a_power_2018_plus_b_power_2018_eq_2_l53_53932

noncomputable def f (x a b : ℝ) : ℝ := (x + a) / (x + b)

theorem a_power_2018_plus_b_power_2018_eq_2 (a b : ℝ) :
  (∀ x : ℝ, f x a b + f (1 / x) a b = 0) → a^2018 + b^2018 = 2 :=
by 
  sorry

end a_power_2018_plus_b_power_2018_eq_2_l53_53932


namespace luggage_max_length_l53_53903

theorem luggage_max_length
  (l w h : ℕ)
  (h_eq : h = 30)
  (ratio_l_w : l = 3 * w / 2)
  (sum_leq : l + w + h ≤ 160) :
  l ≤ 78 := sorry

end luggage_max_length_l53_53903


namespace heartsuit_example_l53_53359

def heartsuit (a b : ℤ) : ℤ := a * b^3 - 2 * b + 3

theorem heartsuit_example : heartsuit 2 3 = 51 :=
by
  sorry

end heartsuit_example_l53_53359


namespace best_store_is_A_l53_53876

/-- Problem conditions -/
def price_per_ball : Nat := 25
def balls_to_buy : Nat := 58

/-- Store A conditions -/
def balls_bought_per_offer_A : Nat := 10
def balls_free_per_offer_A : Nat := 3

/-- Store B conditions -/
def discount_per_ball_B : Nat := 5

/-- Store C conditions -/
def cashback_rate_C : Nat := 40
def cashback_threshold_C : Nat := 200

/-- Cost calculations -/
def cost_store_A (total_balls : Nat) (price : Nat) : Nat :=
  let full_offers := total_balls / balls_bought_per_offer_A
  let remaining_balls := total_balls % balls_bought_per_offer_A
  let balls_paid_for := full_offers * (balls_bought_per_offer_A - balls_free_per_offer_A) + remaining_balls
  balls_paid_for * price

def cost_store_B (total_balls : Nat) (price : Nat) (discount : Nat) : Nat :=
  total_balls * (price - discount)

def cost_store_C (total_balls : Nat) (price : Nat) (cashback_rate : Nat) (threshold : Nat) : Nat :=
  let cost_before_cashback := total_balls * price
  let full_cashbacks := cost_before_cashback / threshold
  let cashback_amount := full_cashbacks * cashback_rate
  cost_before_cashback - cashback_amount

theorem best_store_is_A :
  cost_store_A balls_to_buy price_per_ball = 1075 ∧
  cost_store_B balls_to_buy price_per_ball discount_per_ball_B = 1160 ∧
  cost_store_C balls_to_buy price_per_ball cashback_rate_C cashback_threshold_C = 1170 ∧
  cost_store_A balls_to_buy price_per_ball < cost_store_B balls_to_buy price_per_ball discount_per_ball_B ∧
  cost_store_A balls_to_buy price_per_ball < cost_store_C balls_to_buy price_per_ball cashback_rate_C cashback_threshold_C :=
by {
  -- placeholder for the proof
  sorry
}

end best_store_is_A_l53_53876


namespace translate_parabola_up_one_unit_l53_53432

theorem translate_parabola_up_one_unit (x : ℝ) :
  let y := 3 * x^2
  (y + 1) = 3 * x^2 + 1 :=
by
  -- Proof omitted
  sorry

end translate_parabola_up_one_unit_l53_53432


namespace no_integer_solutions_for_square_polynomial_l53_53513

theorem no_integer_solutions_for_square_polynomial :
  (∀ x : ℤ, ∃ k : ℤ, k^2 = x^4 + 5*x^3 + 10*x^2 + 5*x + 25 → false) :=
by
  sorry

end no_integer_solutions_for_square_polynomial_l53_53513


namespace find_two_digit_number_l53_53924

def product_of_digits (n : ℕ) : ℕ := 
-- Implementation that calculates the product of the digits of n
sorry

def sum_of_digits (n : ℕ) : ℕ := 
-- Implementation that calculates the sum of the digits of n
sorry

theorem find_two_digit_number (M : ℕ) (h1 : 10 ≤ M ∧ M < 100) (h2 : M = product_of_digits M + sum_of_digits M + 1) : M = 18 :=
by
  sorry

end find_two_digit_number_l53_53924


namespace parabola_focus_distance_l53_53196

theorem parabola_focus_distance (C : Set (ℝ × ℝ))
  (hC : ∀ x y, (y^2 = x) → (x, y) ∈ C)
  (F : ℝ × ℝ)
  (hF : F = (1/4, 0))
  (A : ℝ × ℝ)
  (hA : A = (x0, y0) ∧ (y0^2 = x0 ∧ (x0, y0) ∈ C))
  (hAF : dist A F = (5/4) * x0) :
  x0 = 1 :=
sorry

end parabola_focus_distance_l53_53196


namespace find_m_prove_inequality_l53_53171

-- Using noncomputable to handle real numbers where needed
noncomputable def f (x m : ℝ) := m - |x - 1|

-- First proof: Find m given conditions on f(x)
theorem find_m (m : ℝ) :
  (∀ x, f (x + 2) m + f (x - 2) m ≥ 0 ↔ -2 ≤ x ∧ x ≤ 4) → m = 3 :=
sorry

-- Second proof: Prove the inequality given m = 3
theorem prove_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / a + 1 / (2 * b) + 1 / (3 * c) = 3) → a + 2 * b + 3 * c ≥ 3 :=
sorry

end find_m_prove_inequality_l53_53171


namespace sum_of_areas_l53_53281

theorem sum_of_areas (r s t : ℝ)
  (h1 : r + s = 13)
  (h2 : s + t = 5)
  (h3 : r + t = 12)
  (h4 : t = r / 2) : 
  π * (r ^ 2 + s ^ 2 + t ^ 2) = 105 * π := 
by
  sorry

end sum_of_areas_l53_53281


namespace inverse_proposition_of_divisibility_by_5_l53_53776

theorem inverse_proposition_of_divisibility_by_5 (n : ℕ) :
  (n % 10 = 5 → n % 5 = 0) → (n % 5 = 0 → n % 10 = 5) :=
sorry

end inverse_proposition_of_divisibility_by_5_l53_53776


namespace smallest_n_produces_terminating_decimal_l53_53941

noncomputable def smallest_n := 12

theorem smallest_n_produces_terminating_decimal (n : ℕ) (h_pos: 0 < n) : 
    (∀ m : ℕ, m > 113 → (n = m - 113 → (∃ k : ℕ, 1 ≤ k ∧ (m = 2^k ∨ m = 5^k)))) :=
by
  sorry

end smallest_n_produces_terminating_decimal_l53_53941


namespace AndrewAge_l53_53612

variable (a f g : ℚ)
axiom h1 : f = 8 * a
axiom h2 : g = 3 * f
axiom h3 : g - a = 72

theorem AndrewAge : a = 72 / 23 :=
by
  sorry

end AndrewAge_l53_53612


namespace total_earrings_after_one_year_l53_53503

theorem total_earrings_after_one_year :
  let bella_earrings := 10
  let monica_earrings := 10 / 0.25
  let rachel_earrings := monica_earrings / 2
  let initial_total := bella_earrings + monica_earrings + rachel_earrings
  let olivia_earrings_initial := initial_total + 5
  let olivia_earrings_after := olivia_earrings_initial * 1.2
  let total_earrings := bella_earrings + monica_earrings + rachel_earrings + olivia_earrings_after
  total_earrings = 160 :=
by
  sorry

end total_earrings_after_one_year_l53_53503


namespace percent_sold_second_day_l53_53901

-- Defining the problem conditions
def initial_pears (x : ℕ) : ℕ := x
def pears_sold_first_day (x : ℕ) : ℕ := (20 * x) / 100
def pears_remaining_after_first_sale (x : ℕ) : ℕ := x - pears_sold_first_day x
def pears_thrown_away_first_day (x : ℕ) : ℕ := (50 * pears_remaining_after_first_sale x) / 100
def pears_remaining_after_first_day (x : ℕ) : ℕ := pears_remaining_after_first_sale x - pears_thrown_away_first_day x
def total_pears_thrown_away (x : ℕ) : ℕ := (72 * x) / 100
def pears_thrown_away_second_day (x : ℕ) : ℕ := total_pears_thrown_away x - pears_thrown_away_first_day x
def pears_remaining_after_second_day (x : ℕ) : ℕ := pears_remaining_after_first_day x - pears_thrown_away_second_day x

-- Prove that the vendor sold 20% of the remaining pears on the second day
theorem percent_sold_second_day (x : ℕ) (h : x > 0) :
  ((pears_remaining_after_second_day x * 100) / pears_remaining_after_first_day x) = 20 :=
by 
  sorry

end percent_sold_second_day_l53_53901


namespace sqrt_expression_evaluation_l53_53495

theorem sqrt_expression_evaluation :
  (Real.sqrt 48 - 6 * Real.sqrt (1 / 3) - Real.sqrt 18 / Real.sqrt 6) = Real.sqrt 3 :=
by
  sorry

end sqrt_expression_evaluation_l53_53495


namespace total_investment_is_10000_l53_53397

open Real

-- Definitions of conditions
def interest_rate_8 : Real := 0.08
def interest_rate_9 : Real := 0.09
def combined_interest : Real := 840
def investment_8 : Real := 6000
def total_interest (x : Real) : Real := (interest_rate_8 * investment_8 + interest_rate_9 * x)
def investment_9 : Real := 4000

-- Theorem stating the problem
theorem total_investment_is_10000 :
    (∀ x : Real,
        total_interest x = combined_interest → x = investment_9) →
    investment_8 + investment_9 = 10000 := 
by
    intros
    sorry

end total_investment_is_10000_l53_53397


namespace baseball_card_total_percent_decrease_l53_53529

theorem baseball_card_total_percent_decrease :
  ∀ (original_value first_year_decrease second_year_decrease : ℝ),
  first_year_decrease = 0.60 →
  second_year_decrease = 0.10 →
  original_value > 0 →
  (original_value - original_value * first_year_decrease - (original_value * (1 - first_year_decrease)) * second_year_decrease) =
  original_value * (1 - 0.64) :=
by
  intros original_value first_year_decrease second_year_decrease h_first_year h_second_year h_original_pos
  sorry

end baseball_card_total_percent_decrease_l53_53529


namespace x_intercept_of_line_l53_53836

variables (x₁ y₁ x₂ y₂ : ℝ) (m : ℝ)

/-- The line passing through the points (-1, 1) and (3, 9) has an x-intercept of -3/2. -/
theorem x_intercept_of_line : 
  let x₁ := -1
  let y₁ := 1
  let x₂ := 3
  let y₂ := 9
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (0 : ℝ) = m * (x : ℝ) + b → x = (-3 / 2) := 
by 
  sorry

end x_intercept_of_line_l53_53836


namespace find_a_b_sum_pos_solution_l53_53834

theorem find_a_b_sum_pos_solution :
  ∃ (a b : ℕ), (∃ (x : ℝ), x^2 + 16 * x = 100 ∧ x = Real.sqrt a - b) ∧ a + b = 172 :=
by
  sorry

end find_a_b_sum_pos_solution_l53_53834


namespace find_ratio_l53_53066

noncomputable def complex_numbers_are_non_zero (x y z : ℂ) : Prop :=
x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0

noncomputable def sum_is_30 (x y z : ℂ) : Prop :=
x + y + z = 30

noncomputable def expanded_equality (x y z : ℂ) : Prop :=
((x - y)^2 + (x - z)^2 + (y - z)^2) * (x + y + z) = x * y * z

theorem find_ratio (x y z : ℂ)
  (h1 : complex_numbers_are_non_zero x y z)
  (h2 : sum_is_30 x y z)
  (h3 : expanded_equality x y z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3.5 :=
sorry

end find_ratio_l53_53066


namespace cost_per_gallon_l53_53310

theorem cost_per_gallon (weekly_spend : ℝ) (two_week_usage : ℝ) (weekly_spend_eq : weekly_spend = 36) (two_week_usage_eq : two_week_usage = 24) : 
  (2 * weekly_spend / two_week_usage) = 3 :=
by sorry

end cost_per_gallon_l53_53310


namespace combination_recurrence_l53_53577

variable {n r : ℕ}
variable (C : ℕ → ℕ → ℕ)

theorem combination_recurrence (hn : n > 0) (hr : r > 0) (h : n > r)
  (h2 : ∀ (k : ℕ), k = 1 → C 2 1 = C 1 1 + C 1) 
  (h3 : ∀ (k : ℕ), k = 1 → C 3 1 = C 2 1 + C 2) 
  (h4 : ∀ (k : ℕ), k = 2 → C 3 2 = C 2 2 + C 2 1)
  (h5 : ∀ (k : ℕ), k = 1 → C 4 1 = C 3 1 + C 3) 
  (h6 : ∀ (k : ℕ), k = 2 → C 4 2 = C 3 2 + C 3 1)
  (h7 : ∀ (k : ℕ), k = 3 → C 4 3 = C 3 3 + C 3 2)
  (h8 : ∀ n r : ℕ, (n > r) → C n r = C (n-1) r + C (n-1) (r-1)) :
  C n r = C (n-1) r + C (n-1) (r-1) :=
sorry

end combination_recurrence_l53_53577


namespace range_of_sqrt_x_minus_1_meaningful_l53_53900

theorem range_of_sqrt_x_minus_1_meaningful (x : ℝ) (h : 0 ≤ x - 1) : 1 ≤ x := 
sorry

end range_of_sqrt_x_minus_1_meaningful_l53_53900


namespace total_price_of_purchases_l53_53471

def price_of_refrigerator := 4275
def price_difference := 1490
def price_of_washing_machine := price_of_refrigerator - price_difference
def total_price := price_of_refrigerator + price_of_washing_machine

theorem total_price_of_purchases : total_price = 7060 :=
by
  rfl  -- This is just a placeholder; you need to solve the proof.

end total_price_of_purchases_l53_53471


namespace math_problem_l53_53099

theorem math_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (a + 2) * (b + 2) = 18) :
  (∀ x, (x = 3 / (a + 2) + 3 / (b + 2)) → x ≥ Real.sqrt 2) ∧
  ¬(∃ y, (y = a * b) ∧ y ≤ 11 - 6 * Real.sqrt 2) ∧
  (∀ z, (z = 2 * a + b) → z ≥ 6) ∧
  (∀ w, (w = (a + 1) * b) → w ≤ 8) :=
sorry

end math_problem_l53_53099


namespace proof_problem_l53_53332

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 4 * x^2) - 2 * x) + 3

theorem proof_problem : f (Real.log 2) + f (Real.log (1 / 2)) = 6 := 
by 
  sorry

end proof_problem_l53_53332


namespace baker_number_of_eggs_l53_53652

theorem baker_number_of_eggs (flour cups eggs : ℕ) (h1 : eggs = 3 * (flour / 2)) (h2 : flour = 6) : eggs = 9 :=
by
  sorry

end baker_number_of_eggs_l53_53652


namespace q_is_false_of_pq_false_and_notp_false_l53_53185

variables (p q : Prop)

theorem q_is_false_of_pq_false_and_notp_false (hpq_false : ¬(p ∧ q)) (hnotp_false : ¬(¬p)) : ¬q := 
by 
  sorry

end q_is_false_of_pq_false_and_notp_false_l53_53185


namespace quadratic_roots_l53_53907

theorem quadratic_roots (m x1 x2 : ℝ) 
  (h1 : 2*x1^2 + 4*m*x1 + m = 0)
  (h2 : 2*x2^2 + 4*m*x2 + m = 0)
  (h3 : x1 ≠ x2)
  (h4 : x1^2 + x2^2 = 3/16) : 
  m = -1/8 := 
sorry

end quadratic_roots_l53_53907


namespace max_value_fraction_l53_53385

theorem max_value_fraction (e a b : ℝ) (h : ∀ x : ℝ, (e - a) * Real.exp x + x + b + 1 ≤ 0) : 
  (b + 1) / a ≤ 1 / e :=
sorry

end max_value_fraction_l53_53385


namespace arithmetic_common_difference_l53_53017

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l53_53017


namespace probability_of_selecting_double_l53_53609

-- Define the conditions and the question
def total_integers : ℕ := 13

def number_of_doubles : ℕ := total_integers

def total_pairings : ℕ := 
  (total_integers * (total_integers + 1)) / 2

def probability_double : ℚ := 
  number_of_doubles / total_pairings

-- Statement to be proved 
theorem probability_of_selecting_double : 
  probability_double = 1/7 := 
sorry

end probability_of_selecting_double_l53_53609


namespace geometric_sum_l53_53181

def S10 : ℕ := 36
def S20 : ℕ := 48

theorem geometric_sum (S30 : ℕ) (h1 : S10 = 36) (h2 : S20 = 48) : S30 = 52 :=
by
  have h3 : (S20 - S10) ^ 2 = S10 * (S30 - S20) :=
    sorry -- This is based on the properties of the geometric sequence
  sorry  -- Solve the equation to show S30 = 52

end geometric_sum_l53_53181


namespace length_more_than_breadth_l53_53773

theorem length_more_than_breadth
  (b x : ℝ)
  (h1 : b + x = 60)
  (h2 : 4 * b + 2 * x = 200) :
  x = 20 :=
by
  sorry

end length_more_than_breadth_l53_53773


namespace candle_ratio_l53_53783

theorem candle_ratio (r b : ℕ) (h1: r = 45) (h2: b = 27) : r / Nat.gcd r b = 5 ∧ b / Nat.gcd r b = 3 := 
by
  sorry

end candle_ratio_l53_53783


namespace earnings_difference_l53_53547

theorem earnings_difference :
  let lower_tasks := 400
  let lower_rate := 0.25
  let higher_tasks := 5
  let higher_rate := 2.00
  let lower_earnings := lower_tasks * lower_rate
  let higher_earnings := higher_tasks * higher_rate
  lower_earnings - higher_earnings = 90 := by
  sorry

end earnings_difference_l53_53547


namespace trail_mix_total_weight_l53_53455

theorem trail_mix_total_weight :
  let peanuts := 0.16666666666666666
  let chocolate_chips := 0.16666666666666666
  let raisins := 0.08333333333333333
  let almonds := 0.14583333333333331
  let cashews := (1 / 8 : Real)
  let dried_cranberries := (3 / 32 : Real)
  (peanuts + chocolate_chips + raisins + almonds + cashews + dried_cranberries) = 0.78125 :=
by
  sorry

end trail_mix_total_weight_l53_53455


namespace simplify_and_evaluate_division_l53_53775

theorem simplify_and_evaluate_division (a : ℝ) (h : a = 3) :
  (a + 2 + 4 / (a - 2)) / (a ^ 3 / (a ^ 2 - 4 * a + 4)) = 1 / 3 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_division_l53_53775


namespace no_rectangle_with_five_distinct_squares_no_rectangle_with_six_distinct_squares_l53_53102

-- Part (a): Prove that it is impossible to arrange five distinct-sized squares to form a rectangle.
theorem no_rectangle_with_five_distinct_squares (s1 s2 s3 s4 s5 : ℕ) 
  (dist : s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s1 ≠ s5 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s2 ≠ s5 ∧ s3 ≠ s4 ∧ s3 ≠ s5 ∧ s4 ≠ s5) :
  ¬ ∃ (l w : ℕ), (s1 ≤ l ∧ s1 ≤ w) ∧ (s2 ≤ l ∧ s2 ≤ w) ∧ (s3 ≤ l ∧ s3 ≤ w) ∧ (s4 ≤ l ∧ s4 ≤ w) ∧ (s5 ≤ l ∧ s5 ≤ w) ∧
  (l * w = (s1 + s2 + s3 + s4 + s5)) :=
by
  -- Proof placeholder
  sorry

-- Part (b): Prove that it is impossible to arrange six distinct-sized squares to form a rectangle.
theorem no_rectangle_with_six_distinct_squares (s1 s2 s3 s4 s5 s6 : ℕ) 
  (dist : s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s1 ≠ s5 ∧ s1 ≠ s6 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s2 ≠ s5 ∧ s2 ≠ s6 ∧ s3 ≠ s4 ∧ s3 ≠ s5 ∧ s3 ≠ s6 ∧ s4 ≠ s5 ∧ s4 ≠ s6 ∧ s5 ≠ s6) :
  ¬ ∃ (l w : ℕ), (s1 ≤ l ∧ s1 ≤ w) ∧ (s2 ≤ l ∧ s2 ≤ w) ∧ (s3 ≤ l ∧ s3 ≤ w) ∧ (s4 ≤ l ∧ s4 ≤ w) ∧ (s5 ≤ l ∧ s5 ≤ w) ∧ (s6 ≤ l ∧ s6 ≤ w) ∧
  (l * w = (s1 + s2 + s3 + s4 + s5 + s6)) :=
by
  -- Proof placeholder
  sorry

end no_rectangle_with_five_distinct_squares_no_rectangle_with_six_distinct_squares_l53_53102


namespace triangles_with_positive_area_l53_53875

-- Define the set of points in the coordinate grid
def points := { p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 4 ∧ 1 ≤ p.2 ∧ p.2 ≤ 4 }

-- Number of ways to choose 3 points from the grid
def total_triples := Nat.choose 16 3

-- Number of collinear triples
def collinear_triples := 32 + 8 + 4

-- Number of triangles with positive area
theorem triangles_with_positive_area :
  (total_triples - collinear_triples) = 516 :=
by
  -- Definitions for total_triples and collinear_triples.
  -- Proof steps would go here.
  sorry

end triangles_with_positive_area_l53_53875


namespace multiples_of_3_ending_number_l53_53583

theorem multiples_of_3_ending_number :
  ∃ n, ∃ k, k = 93 ∧ (∀ m, 81 + 3 * m = n → 0 ≤ m ∧ m < k) ∧ n = 357 := 
by
  sorry

end multiples_of_3_ending_number_l53_53583


namespace at_least_one_genuine_product_l53_53683

-- Definitions of the problem conditions
structure Products :=
  (total : ℕ)
  (genuine : ℕ)
  (defective : ℕ)

def products : Products := { total := 12, genuine := 10, defective := 2 }

-- Definition of the event
def certain_event (p : Products) (selected : ℕ) : Prop :=
  selected > p.defective

-- The theorem stating that there is at least one genuine product among the selected ones
theorem at_least_one_genuine_product : certain_event products 3 :=
by
  sorry

end at_least_one_genuine_product_l53_53683


namespace total_visitors_count_l53_53262

def initial_morning_visitors : ℕ := 500
def noon_departures : ℕ := 119
def additional_afternoon_arrivals : ℕ := 138

def afternoon_arrivals : ℕ := noon_departures + additional_afternoon_arrivals
def total_visitors : ℕ := initial_morning_visitors + afternoon_arrivals

theorem total_visitors_count : total_visitors = 757 := 
by sorry

end total_visitors_count_l53_53262


namespace car_speed_l53_53205

theorem car_speed 
  (d : ℝ) (t : ℝ) 
  (hd : d = 520) (ht : t = 8) : 
  d / t = 65 := 
by 
  sorry

end car_speed_l53_53205


namespace euler_totient_inequality_l53_53675

open Int

def is_power_of_prime (m : ℕ) : Prop :=
  ∃ p k : ℕ, (Nat.Prime p) ∧ (k ≥ 1) ∧ (m = p^k)

def φ (n m : ℕ) (h : m ≠ 1) : ℕ := -- This is a placeholder, you would need an actual implementation for φ
  sorry

theorem euler_totient_inequality (m : ℕ) (h : m ≠ 1) :
  (is_power_of_prime m) ↔ (∀ n > 0, (φ n m h) / n ≥ (φ m m h) / m) :=
sorry

end euler_totient_inequality_l53_53675


namespace max_value_expression_l53_53325

theorem max_value_expression : 
  ∃ x_max : ℝ, 
    (∀ x : ℝ, -3 * x^2 + 15 * x + 9 ≤ -3 * x_max^2 + 15 * x_max + 9) ∧
    (-3 * x_max^2 + 15 * x_max + 9 = 111 / 4) :=
by
  sorry

end max_value_expression_l53_53325


namespace product_xyz_is_minus_one_l53_53286

-- Definitions of the variables and equations
variables (x y z : ℝ)

-- Assumptions based on the given conditions
def condition1 : Prop := x + (1 / y) = 2
def condition2 : Prop := y + (1 / z) = 2
def condition3 : Prop := z + (1 / x) = 2

-- The theorem stating the conclusion to be proved
theorem product_xyz_is_minus_one (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z x) : x * y * z = -1 :=
by sorry

end product_xyz_is_minus_one_l53_53286


namespace domain_fraction_function_l53_53743

theorem domain_fraction_function (f : ℝ → ℝ):
  (∀ x : ℝ, -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 0) →
  (∀ x : ℝ, x ≠ 0 → -2 ≤ x ∧ x < 0) →
  (∀ x, (2^x - 1) ≠ 0) →
  true := sorry

end domain_fraction_function_l53_53743


namespace min_radius_cylinder_proof_l53_53227

-- Defining the radius of the hemisphere
def radius_hemisphere : ℝ := 10

-- Defining the angle alpha which is less than or equal to 30 degrees
def angle_alpha_leq_30 (α : ℝ) : Prop := α ≤ 30 * Real.pi / 180

-- Minimum radius of the cylinder given alpha <= 30 degrees
noncomputable def min_radius_cylinder : ℝ :=
  10 * (2 / Real.sqrt 3 - 1)

theorem min_radius_cylinder_proof (α : ℝ) (hα : angle_alpha_leq_30 α) :
  min_radius_cylinder = 10 * (2 / Real.sqrt 3 - 1) :=
by
  -- Here would go the detailed proof steps
  sorry

end min_radius_cylinder_proof_l53_53227


namespace regression_eq_change_in_y_l53_53728

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 2 - 1.5 * x

-- Define the statement to be proved
theorem regression_eq_change_in_y (x : ℝ) :
  regression_eq (x + 1) = regression_eq x - 1.5 :=
by sorry

end regression_eq_change_in_y_l53_53728


namespace worker_late_by_10_minutes_l53_53235

def usual_time : ℕ := 40
def speed_ratio : ℚ := 4 / 5
def time_new := (usual_time : ℚ) * (5 / 4) -- This is the equation derived from solving

theorem worker_late_by_10_minutes : 
  ((time_new : ℚ) - usual_time) = 10 :=
by
  sorry -- proof is skipped

end worker_late_by_10_minutes_l53_53235


namespace min_colors_correctness_l53_53009

noncomputable def min_colors_no_monochromatic_cycle (n : ℕ) : ℕ :=
if n ≤ 2 then 1 else 2

theorem min_colors_correctness (n : ℕ) (h₀ : n > 0) :
  (min_colors_no_monochromatic_cycle n = 1 ∧ n ≤ 2) ∨
  (min_colors_no_monochromatic_cycle n = 2 ∧ n ≥ 3) :=
by
  sorry

end min_colors_correctness_l53_53009


namespace factorize_1_factorize_2_l53_53998

-- Define the variables involved
variables (a x y : ℝ)

-- Problem (1): 18a^2 - 32 = 2 * (3a + 4) * (3a - 4)
theorem factorize_1 (a : ℝ) : 
  18 * a^2 - 32 = 2 * (3 * a + 4) * (3 * a - 4) :=
sorry

-- Problem (2): y - 6xy + 9x^2y = y * (1 - 3x) ^ 2
theorem factorize_2 (x y : ℝ) : 
  y - 6 * x * y + 9 * x^2 * y = y * (1 - 3 * x) ^ 2 :=
sorry

end factorize_1_factorize_2_l53_53998


namespace geometric_common_ratio_of_arithmetic_seq_l53_53961

theorem geometric_common_ratio_of_arithmetic_seq 
  (a : ℕ → ℝ) (d q : ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_a1 : a 1 = 2)
  (h_nonzero_diff : d ≠ 0)
  (h_geo_seq : a 1 = 2 ∧ a 3 = 2 * q ∧ a 11 = 2 * q^2) : 
  q = 4 := 
by
  sorry

end geometric_common_ratio_of_arithmetic_seq_l53_53961


namespace order_of_A_B_C_D_l53_53476

def A := Nat.factorial 8 ^ Nat.factorial 8
def B := 8 ^ (8 ^ 8)
def C := 8 ^ 88
def D := 8 ^ 64

theorem order_of_A_B_C_D : D < C ∧ C < B ∧ B < A := by
  sorry

end order_of_A_B_C_D_l53_53476


namespace range_of_m_l53_53416

theorem range_of_m (m : ℝ) : (1^2 + 2*1 - m ≤ 0) ∧ (2^2 + 2*2 - m > 0) → 3 ≤ m ∧ m < 8 := by
  sorry

end range_of_m_l53_53416


namespace complex_division_l53_53642

theorem complex_division (i : ℂ) (h : i * i = -1) : 3 / (1 - i) ^ 2 = (3 / 2) * i :=
by
  sorry

end complex_division_l53_53642


namespace money_total_l53_53592

theorem money_total (s j m : ℝ) (h1 : 3 * s = 80) (h2 : j / 2 = 70) (h3 : 2.5 * m = 100) :
  s + j + m = 206.67 :=
sorry

end money_total_l53_53592


namespace fraction_power_seven_l53_53905

theorem fraction_power_seven : (5 / 3 : ℚ) ^ 7 = 78125 / 2187 := 
by
  sorry

end fraction_power_seven_l53_53905


namespace total_fare_for_20km_l53_53882

def base_fare : ℝ := 8
def fare_per_km_from_3_to_10 : ℝ := 1.5
def fare_per_km_beyond_10 : ℝ := 0.8

def fare_for_first_3km : ℝ := base_fare
def fare_for_3_to_10_km : ℝ := 7 * fare_per_km_from_3_to_10
def fare_for_beyond_10_km : ℝ := 10 * fare_per_km_beyond_10

theorem total_fare_for_20km : fare_for_first_3km + fare_for_3_to_10_km + fare_for_beyond_10_km = 26.5 :=
by
  sorry

end total_fare_for_20km_l53_53882


namespace weight_ratio_l53_53053

-- Conditions
def initial_weight : ℕ := 99
def initial_loss : ℕ := 12
def weight_added_back (x : ℕ) : Prop := x = 81 + 30 - initial_weight
def times_lost : ℕ := 3 * initial_loss
def final_gain : ℕ := 6
def final_weight : ℕ := 81

-- Question
theorem weight_ratio (x : ℕ)
  (H1 : weight_added_back x)
  (H2 : initial_weight - initial_loss + x - times_lost + final_gain = final_weight) :
  x / initial_loss = 2 := by
  sorry

end weight_ratio_l53_53053


namespace perpendicular_lines_condition_l53_53685

theorem perpendicular_lines_condition (m : ℝ) :
    (m = 1 → (∀ (x y : ℝ), (∀ (c d : ℝ), c * (m * x + y - 1) = 0 → d * (x - m * y - 1) = 0 → (c * m + d / m) ^ 2 = 1))) ∧ (∀ (m' : ℝ), m' ≠ 1 → ¬ (∀ (x y : ℝ), (∀ (c d : ℝ), c * (m' * x + y - 1) = 0 → d * (x - m' * y - 1) = 0 → (c * m' + d / m') ^ 2 = 1))) :=
by
  sorry

end perpendicular_lines_condition_l53_53685


namespace cost_of_black_and_white_drawing_l53_53829

-- Given the cost of the color drawing is 1.5 times the cost of the black and white drawing
-- and John paid $240 for the color drawing, we need to prove the cost of the black and white drawing is $160.

theorem cost_of_black_and_white_drawing (C : ℝ) (h : 1.5 * C = 240) : C = 160 :=
by
  sorry

end cost_of_black_and_white_drawing_l53_53829


namespace bananas_eaten_l53_53316

variable (initial_bananas : ℕ) (remaining_bananas : ℕ)

theorem bananas_eaten (initial_bananas remaining_bananas : ℕ) (h_initial : initial_bananas = 12) (h_remaining : remaining_bananas = 10) : initial_bananas - remaining_bananas = 2 := by
  -- Proof goes here
  sorry

end bananas_eaten_l53_53316


namespace solve_triangle_l53_53140

theorem solve_triangle (a b m₁ m₂ k₃ : ℝ) (h1 : a = m₂ / Real.sin γ) (h2 : b = m₁ / Real.sin γ) : 
  a = m₂ / Real.sin γ ∧ b = m₁ / Real.sin γ := 
  by 
  sorry

end solve_triangle_l53_53140


namespace greatest_common_multiple_of_9_and_15_less_than_120_l53_53968

-- Definition of LCM.
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- The main theorem to be proved.
theorem greatest_common_multiple_of_9_and_15_less_than_120 : ∃ x, x = 90 ∧ x < 120 ∧ x % 9 = 0 ∧ x % 15 = 0 :=
by
  -- Proof goes here.
  sorry

end greatest_common_multiple_of_9_and_15_less_than_120_l53_53968


namespace italian_clock_hand_coincidence_l53_53317

theorem italian_clock_hand_coincidence :
  let hour_hand_rotation := 1 / 24
  let minute_hand_rotation := 1
  ∃ (t : ℕ), 0 ≤ t ∧ t < 24 ∧ (t * hour_hand_rotation) % 1 = (t * minute_hand_rotation) % 1
:= sorry

end italian_clock_hand_coincidence_l53_53317


namespace dots_not_visible_l53_53296

def total_dots (n_dice : ℕ) : ℕ := n_dice * 21

def sum_visible_dots (visible : List ℕ) : ℕ := visible.foldl (· + ·) 0

theorem dots_not_visible (visible : List ℕ) (h : visible = [1, 1, 2, 3, 4, 5, 5, 6]) :
  total_dots 4 - sum_visible_dots visible = 57 :=
by
  rw [total_dots, sum_visible_dots]
  simp
  sorry

end dots_not_visible_l53_53296


namespace math_problem_l53_53331

theorem math_problem (x y : ℝ) :
  let A := x^3 + 3*x^2*y + y^3 - 3*x*y^2
  let B := x^2*y - x*y^2
  A - 3*B = x^3 + y^3 := by
  sorry

end math_problem_l53_53331


namespace candies_total_l53_53976

theorem candies_total (N a S : ℕ) (h1 : S = 2 * a + 7) (h2 : S = N * a) (h3 : a > 1) (h4 : N = 3) : S = 21 := 
sorry

end candies_total_l53_53976


namespace common_chord_of_circles_is_x_eq_y_l53_53487

theorem common_chord_of_circles_is_x_eq_y :
  ∀ x y : ℝ, (x^2 + y^2 - 4 * x - 3 = 0) ∧ (x^2 + y^2 - 4 * y - 3 = 0) → (x = y) :=
by
  sorry

end common_chord_of_circles_is_x_eq_y_l53_53487


namespace currency_conversion_l53_53155

variable (a : ℚ)

theorem currency_conversion
  (h1 : (0.5 / 100) * a = 75 / 100) -- 0.5% of 'a' = 75 paise
  (rate_usd : ℚ := 0.012)          -- Conversion rate (USD/INR)
  (rate_eur : ℚ := 0.010)          -- Conversion rate (EUR/INR)
  (rate_gbp : ℚ := 0.009)          -- Conversion rate (GBP/INR)
  (paise_to_rupees : ℚ := 1 / 100) -- 1 Rupee = 100 paise
  : (a * paise_to_rupees * rate_usd = 1.8) ∧
    (a * paise_to_rupees * rate_eur = 1.5) ∧
    (a * paise_to_rupees * rate_gbp = 1.35) :=
by
  sorry

end currency_conversion_l53_53155


namespace jacob_hours_l53_53222

theorem jacob_hours (J : ℕ) (H1 : ∃ (G P : ℕ),
    G = J - 6 ∧
    P = 2 * G - 4 ∧
    J + G + P = 50) : J = 18 :=
by
  sorry

end jacob_hours_l53_53222


namespace income_day_3_is_750_l53_53012

-- Define the given incomes for the specific days
def income_day_1 : ℝ := 250
def income_day_2 : ℝ := 400
def income_day_4 : ℝ := 400
def income_day_5 : ℝ := 500

-- Define the total number of days and the average income over these days
def total_days : ℝ := 5
def average_income : ℝ := 460

-- Define the total income based on the average
def total_income : ℝ := total_days * average_income

-- Define the income on the third day
def income_day_3 : ℝ := total_income - (income_day_1 + income_day_2 + income_day_4 + income_day_5)

-- Claim: The income on the third day is $750
theorem income_day_3_is_750 : income_day_3 = 750 := by
  sorry

end income_day_3_is_750_l53_53012


namespace find_N_l53_53293

theorem find_N (a b c N : ℝ) (h1 : a + b + c = 120) (h2 : a - 10 = N) 
               (h3 : b + 10 = N) (h4 : 7 * c = N): N = 56 :=
by
  sorry

end find_N_l53_53293


namespace geometric_sequence_property_l53_53421

theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ)
  (H_geo : ∀ n, a (n + 1) = a n * q)
  (H_cond1 : a 5 * a 7 = 2)
  (H_cond2 : a 2 + a 10 = 3) :
  (a 12 / a 4 = 2) ∨ (a 12 / a 4 = 1/2) :=
sorry

end geometric_sequence_property_l53_53421


namespace find_max_marks_l53_53142

theorem find_max_marks (M : ℝ) (h1 : 0.60 * M = 80 + 100) : M = 300 := 
by
  sorry

end find_max_marks_l53_53142


namespace smallest_n_positive_odd_integer_l53_53824

theorem smallest_n_positive_odd_integer (n : ℕ) (h1 : n % 2 = 1) (h2 : 3 ^ ((n + 1)^2 / 5) > 500) : n = 6 := sorry

end smallest_n_positive_odd_integer_l53_53824


namespace cos_sum_identity_l53_53084

theorem cos_sum_identity :
  (Real.cos (75 * Real.pi / 180)) ^ 2 + (Real.cos (15 * Real.pi / 180)) ^ 2 + 
  (Real.cos (75 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 5 / 4 := 
by
  sorry

end cos_sum_identity_l53_53084


namespace solve_for_y_l53_53808

-- Define the given condition as a Lean definition
def equation (y : ℝ) : Prop :=
  (2 / y) + ((3 / y) / (6 / y)) = 1.2

-- Theorem statement proving the solution given the condition
theorem solve_for_y (y : ℝ) (h : equation y) : y = 20 / 7 := by
  sorry

-- Example usage to instantiate and make use of the definition
example : equation (20 / 7) := by
  unfold equation
  sorry

end solve_for_y_l53_53808


namespace loom_weaving_rate_l53_53942

theorem loom_weaving_rate (total_cloth : ℝ) (total_time : ℝ) (rate : ℝ) 
  (h1 : total_cloth = 26) (h2 : total_time = 203.125) : rate = total_cloth / total_time := by
  sorry

#check loom_weaving_rate

end loom_weaving_rate_l53_53942


namespace residue_of_11_pow_2048_mod_19_l53_53027

theorem residue_of_11_pow_2048_mod_19 :
  (11 ^ 2048) % 19 = 16 := 
by
  sorry

end residue_of_11_pow_2048_mod_19_l53_53027


namespace differential_savings_l53_53729

theorem differential_savings (income : ℕ) (tax_rate_before : ℝ) (tax_rate_after : ℝ) : 
  income = 36000 → tax_rate_before = 0.46 → tax_rate_after = 0.32 →
  ((income * tax_rate_before) - (income * tax_rate_after)) = 5040 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end differential_savings_l53_53729


namespace calculate_r_when_n_is_3_l53_53394

theorem calculate_r_when_n_is_3 : 
  ∀ (r s n : ℕ), r = 4^s - s → s = 3^n + 2 → n = 3 → r = 4^29 - 29 :=
by 
  intros r s n h1 h2 h3
  sorry

end calculate_r_when_n_is_3_l53_53394


namespace find_base_of_denominator_l53_53046

theorem find_base_of_denominator 
  (some_base : ℕ)
  (h1 : (1/2)^16 * (1/81)^8 = 1 / some_base^16) : 
  some_base = 18 :=
sorry

end find_base_of_denominator_l53_53046


namespace total_cows_l53_53674

theorem total_cows (n : ℕ) 
  (h₁ : n / 3 + n / 6 + n / 9 + 8 = n) : n = 144 :=
by sorry

end total_cows_l53_53674


namespace convert_base_10_to_base_6_l53_53709

theorem convert_base_10_to_base_6 : 
  ∃ (digits : List ℕ), (digits.length = 4 ∧
    List.foldr (λ (x : ℕ) (acc : ℕ) => acc * 6 + x) 0 digits = 314 ∧
    digits = [1, 2, 4, 2]) := by
  sorry

end convert_base_10_to_base_6_l53_53709


namespace balls_in_boxes_l53_53380

theorem balls_in_boxes : 
  ∀ (n k : ℕ), n = 6 ∧ k = 3 ∧ ∀ i, i < k → 1 ≤ i → 
             ( ∃ ways : ℕ, ways = Nat.choose ((n - k) + k - 1) (k - 1) ∧ ways = 10 ) :=
by
  sorry

end balls_in_boxes_l53_53380


namespace problem1_problem2_l53_53580

noncomputable def f (x a b : ℝ) : ℝ := |x - a| - |x + b|
noncomputable def g (x a b : ℝ) : ℝ := -x^2 - a*x - b

-- Problem 1: Prove that a + b = 3
theorem problem1 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : ∀ x, f x a b ≤ 3) : a + b = 3 := 
sorry

-- Problem 2: Prove that 1/2 < a < 3
theorem problem2 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 3) 
  (h₃ : ∀ x, x ≥ a → g x a b < f x a b) : 1/2 < a ∧ a < 3 := 
sorry

end problem1_problem2_l53_53580


namespace math_problem_l53_53387

open Real

-- Conditions extracted from the problem
def cond1 (a b : ℝ) : Prop := -|2 - a| + b = 5
def cond2 (a b : ℝ) : Prop := -|8 - a| + b = 3
def cond3 (c d : ℝ) : Prop := |2 - c| + d = 5
def cond4 (c d : ℝ) : Prop := |8 - c| + d = 3
def cond5 (a c : ℝ) : Prop := 2 < a ∧ a < 8
def cond6 (a c : ℝ) : Prop := 2 < c ∧ c < 8

-- Proof problem: Given the conditions, prove that a + c = 10
theorem math_problem (a b c d : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) (h3 : cond3 c d) (h4 : cond4 c d)
  (h5 : cond5 a c) (h6 : cond6 a c) : a + c = 10 := 
by
  sorry

end math_problem_l53_53387


namespace unique_solution_3_pow_x_minus_2_pow_y_eq_7_l53_53701

theorem unique_solution_3_pow_x_minus_2_pow_y_eq_7 :
  ∀ x y : ℕ, (1 ≤ x) → (1 ≤ y) → (3 ^ x - 2 ^ y = 7) → (x = 2 ∧ y = 1) :=
by
  intros x y hx hy hxy
  sorry

end unique_solution_3_pow_x_minus_2_pow_y_eq_7_l53_53701


namespace square_of_other_leg_l53_53837

-- Conditions
variable (a b c : ℝ)
variable (h₁ : c = a + 2)
variable (h₂ : a^2 + b^2 = c^2)

-- The theorem statement
theorem square_of_other_leg (a b c : ℝ) (h₁ : c = a + 2) (h₂ : a^2 + b^2 = c^2) : b^2 = 4 * a + 4 :=
by
  sorry

end square_of_other_leg_l53_53837


namespace remainder_and_division_l53_53023

theorem remainder_and_division (n : ℕ) (h1 : n = 1680) (h2 : n % 9 = 0) : 
  1680 % 1677 = 3 :=
by {
  sorry
}

end remainder_and_division_l53_53023


namespace solve_for_y_l53_53607

theorem solve_for_y (y : ℝ) (h : y + 81 / (y - 3) = -12) : y = -6 ∨ y = -3 :=
sorry

end solve_for_y_l53_53607


namespace sixth_number_is_811_l53_53825

noncomputable def sixth_number_in_21st_row : ℕ := 
  let n := 21 
  let k := 6
  let total_numbers_up_to_previous_row := n * n
  let position_in_row := total_numbers_up_to_previous_row + k
  2 * position_in_row - 1

theorem sixth_number_is_811 : sixth_number_in_21st_row = 811 := by
  sorry

end sixth_number_is_811_l53_53825


namespace number_of_n_such_that_n_div_25_minus_n_is_square_l53_53172

theorem number_of_n_such_that_n_div_25_minus_n_is_square :
  ∃! n1 n2 : ℤ, ∀ n : ℤ, (n = n1 ∨ n = n2) ↔ ∃ k : ℤ, k^2 = n / (25 - n) :=
sorry

end number_of_n_such_that_n_div_25_minus_n_is_square_l53_53172


namespace firstDiscountIsTenPercent_l53_53408

def listPrice : ℝ := 70
def finalPrice : ℝ := 56.16
def secondDiscount : ℝ := 10.857142857142863

theorem firstDiscountIsTenPercent (x : ℝ) : 
    finalPrice = listPrice * (1 - x / 100) * (1 - secondDiscount / 100) ↔ x = 10 := 
by
  sorry

end firstDiscountIsTenPercent_l53_53408


namespace winter_melon_ratio_l53_53268

theorem winter_melon_ratio (T Ok_sales Choc_sales : ℕ) (hT : T = 50) 
  (hOk : Ok_sales = 3 * T / 10) (hChoc : Choc_sales = 15) :
  (T - (Ok_sales + Choc_sales)) / T = 2 / 5 :=
by
  sorry

end winter_melon_ratio_l53_53268


namespace possible_values_of_m_l53_53284

theorem possible_values_of_m (m : ℝ) (h1 : |m| = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
by
  sorry

end possible_values_of_m_l53_53284


namespace sum_of_digits_9ab_l53_53493

def a : ℕ := 999
def b : ℕ := 666

theorem sum_of_digits_9ab : 
  let n := 9 * a * b
  (n.digits 10).sum = 36 := 
by
  sorry

end sum_of_digits_9ab_l53_53493


namespace simplify_power_expression_l53_53188

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^5 = 243 * x^20 :=
by
  sorry

end simplify_power_expression_l53_53188


namespace square_placement_conditions_l53_53504

-- Definitions for natural numbers at vertices and center
def top_left := 14
def top_right := 6
def bottom_right := 15
def bottom_left := 35
def center := 210

theorem square_placement_conditions :
  (∃ gcd1 > 1, gcd1 = Nat.gcd top_left top_right) ∧
  (∃ gcd2 > 1, gcd2 = Nat.gcd top_right bottom_right) ∧
  (∃ gcd3 > 1, gcd3 = Nat.gcd bottom_right bottom_left) ∧
  (∃ gcd4 > 1, gcd4 = Nat.gcd bottom_left top_left) ∧
  (Nat.gcd top_left bottom_right = 1) ∧
  (Nat.gcd top_right bottom_left = 1) ∧
  (Nat.gcd top_left center > 1) ∧
  (Nat.gcd top_right center > 1) ∧
  (Nat.gcd bottom_right center > 1) ∧
  (Nat.gcd bottom_left center > 1) 
 := by
sorry

end square_placement_conditions_l53_53504


namespace wire_cut_min_area_l53_53686

theorem wire_cut_min_area :
  ∃ x : ℝ, 0 < x ∧ x < 100 ∧ S = π * (x / (2 * π))^2 + ((100 - x) / 4)^2 ∧ 
  (∀ y : ℝ, 0 < y ∧ y < 100 → (π * (y / (2 * π))^2 + ((100 - y) / 4)^2 ≥ S)) ∧
  x = 100 * π / (16 + π) :=
sorry

end wire_cut_min_area_l53_53686


namespace chandra_valid_pairings_l53_53071

def valid_pairings (num_bowls : ℕ) (num_glasses : ℕ) : ℕ :=
  num_bowls * num_glasses

theorem chandra_valid_pairings : valid_pairings 6 6 = 36 :=
  by sorry

end chandra_valid_pairings_l53_53071


namespace arrange_abc_l53_53866

theorem arrange_abc (a b c : ℝ) (h1 : a = Real.log 2 / Real.log 2)
                               (h2 : b = Real.sqrt 2)
                               (h3 : c = Real.cos ((3 / 4) * Real.pi)) :
  c < a ∧ a < b :=
by
  sorry

end arrange_abc_l53_53866


namespace sin_double_angle_identity_l53_53389

theorem sin_double_angle_identity (alpha : ℝ) (h : Real.cos (Real.pi / 4 - alpha) = -4 / 5) : 
  Real.sin (2 * alpha) = 7 / 25 :=
by
  sorry

end sin_double_angle_identity_l53_53389


namespace seated_ways_alice_between_bob_and_carol_l53_53337

-- Define the necessary entities and conditions for the problem.
def num_people : Nat := 7
def alice := "Alice"
def bob := "Bob"
def carol := "Carol"

-- The main theorem
theorem seated_ways_alice_between_bob_and_carol :
  ∃ (ways : Nat), ways = 48 := by
  sorry

end seated_ways_alice_between_bob_and_carol_l53_53337


namespace mrs_franklin_initial_valentines_l53_53745

theorem mrs_franklin_initial_valentines (v g l : ℕ) (h1 : g = 42) (h2 : l = 16) (h3 : v = g + l) : v = 58 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end mrs_franklin_initial_valentines_l53_53745


namespace square_of_99_is_9801_l53_53535

theorem square_of_99_is_9801 : 99 ^ 2 = 9801 := 
by
  sorry

end square_of_99_is_9801_l53_53535


namespace quadratic_function_is_explicit_form_l53_53352

-- Conditions
variable {f : ℝ → ℝ}
variable (H1 : f (-1) = 0)
variable (H2 : ∀ x : ℝ, x ≤ f x ∧ f x ≤ (x^2 + 1) / 2)

-- The quadratic function we aim to prove
def quadratic_function_form_proof (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = (1/4) * x^2 + (1/2) * x + (1/4)

-- Main theorem statement
theorem quadratic_function_is_explicit_form : quadratic_function_form_proof f :=
by
  -- Placeholder for the proof
  sorry

end quadratic_function_is_explicit_form_l53_53352


namespace sugar_cups_l53_53906

theorem sugar_cups (S : ℕ) (h1 : 21 = S + 8) : S = 13 := 
by { sorry }

end sugar_cups_l53_53906


namespace initial_mean_corrected_l53_53097

theorem initial_mean_corrected
  (M : ℝ)
  (h : 30 * M + 10 = 30 * 140.33333333333334) :
  M = 140 :=
by
  sorry

end initial_mean_corrected_l53_53097


namespace log_minus_one_has_one_zero_l53_53419

theorem log_minus_one_has_one_zero : ∃! x : ℝ, x > 0 ∧ (Real.log x - 1 = 0) :=
sorry

end log_minus_one_has_one_zero_l53_53419


namespace solve_y_l53_53660

theorem solve_y : ∀ y : ℚ, (9 * y^2 + 8 * y - 2 = 0) ∧ (27 * y^2 + 62 * y - 8 = 0) → y = 1 / 9 :=
by
  intro y h
  cases h
  sorry

end solve_y_l53_53660


namespace tangent_line_at_neg_ln_2_range_of_a_inequality_range_of_a_zero_point_l53_53125

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

/-- Problem 1 -/
theorem tangent_line_at_neg_ln_2 :
  let x := -Real.log 2
  let y := f x
  ∃ k b : ℝ, (y - b) = k * (x - (-Real.log 2)) ∧ k = (Real.exp x - 1) ∧ b = Real.log 2 + 1/2 :=
sorry

/-- Problem 2 -/
theorem range_of_a_inequality :
  ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x > a * x) ↔ a ∈ Set.Iio (Real.exp 1 - 1) :=
sorry

/-- Problem 3 -/
theorem range_of_a_zero_point :
  ∀ a : ℝ, (∃! x : ℝ, f x - a * x = 0) ↔ a ∈ (Set.Iio (-1) ∪ Set.Ioi (Real.exp 1 - 1)) :=
sorry

end tangent_line_at_neg_ln_2_range_of_a_inequality_range_of_a_zero_point_l53_53125


namespace decimal_representation_of_7_div_12_l53_53993

theorem decimal_representation_of_7_div_12 : (7 / 12 : ℚ) = 0.58333333 := 
sorry

end decimal_representation_of_7_div_12_l53_53993


namespace probability_of_each_suit_in_five_draws_with_replacement_l53_53452

theorem probability_of_each_suit_in_five_draws_with_replacement :
  let deck_size := 52
  let num_cards := 5
  let num_suits := 4
  let prob_each_suit := 1/4
  let target_probability := 9/16
  prob_each_suit * (3/4) * (1/2) * (1/4) * 24 = target_probability :=
by sorry

end probability_of_each_suit_in_five_draws_with_replacement_l53_53452


namespace jason_picked_7_pears_l53_53212

def pears_picked_by_jason (total_pears mike_pears : ℕ) : ℕ :=
  total_pears - mike_pears

theorem jason_picked_7_pears :
  pears_picked_by_jason 15 8 = 7 :=
by
  -- Proof is required but we can insert sorry here to skip it for now
  sorry

end jason_picked_7_pears_l53_53212


namespace problem_l53_53080

variable (a b : ℝ)

theorem problem (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 :=
by
  sorry

end problem_l53_53080


namespace domain_of_f_i_l53_53615

variable (f : ℝ → ℝ)

theorem domain_of_f_i (h : ∀ x, -1 ≤ x + 1 ∧ x + 1 ≤ 1) : ∀ x, -2 ≤ x ∧ x ≤ 0 :=
by
  intro x
  specialize h x
  sorry

end domain_of_f_i_l53_53615


namespace fred_added_nine_l53_53175

def onions_in_basket (initial_onions : ℕ) (added_by_sara : ℕ) (taken_by_sally : ℕ) (added_by_fred : ℕ) : ℕ :=
  initial_onions + added_by_sara - taken_by_sally + added_by_fred

theorem fred_added_nine : ∀ (S F : ℕ), onions_in_basket S 4 5 F = S + 8 → F = 9 :=
by
  intros S F h
  sorry

end fred_added_nine_l53_53175


namespace option_C_is_correct_l53_53585

-- Define the conditions as propositions
def condition_A := |-2| = 2
def condition_B := (-1)^2 = 1
def condition_C := -7 + 3 = -4
def condition_D := 6 / (-2) = -3

-- The statement that option C is correct
theorem option_C_is_correct : condition_C := by
  sorry

end option_C_is_correct_l53_53585


namespace temperature_on_fifth_day_l53_53429

theorem temperature_on_fifth_day (T : ℕ → ℝ) (x : ℝ)
  (h1 : (T 1 + T 2 + T 3 + T 4) / 4 = 58)
  (h2 : (T 2 + T 3 + T 4 + T 5) / 4 = 59)
  (h3 : T 1 / T 5 = 7 / 8) :
  T 5 = 32 := 
sorry

end temperature_on_fifth_day_l53_53429


namespace find_x_l53_53574

theorem find_x (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 :=
by
  sorry

end find_x_l53_53574


namespace second_place_jump_l53_53191

theorem second_place_jump : 
  ∀ (Kyungsoo Younghee Jinju Chanho : ℝ), 
    Kyungsoo = 2.3 → 
    Younghee = 0.9 → 
    Jinju = 1.8 → 
    Chanho = 2.5 → 
    ((Kyungsoo < Chanho) ∧ (Kyungsoo > Jinju) ∧ (Kyungsoo > Younghee)) :=
by 
  sorry

end second_place_jump_l53_53191


namespace minimum_value_l53_53258

theorem minimum_value :
  ∀ (m n : ℝ), m > 0 → n > 0 → (3 * m + n = 1) → (3 / m + 1 / n) ≥ 16 :=
by
  intros m n hm hn hline
  sorry

end minimum_value_l53_53258


namespace h_oplus_h_op_h_equals_h_l53_53136

def op (x y : ℝ) : ℝ := x^3 - y

theorem h_oplus_h_op_h_equals_h (h : ℝ) : op h (op h h) = h := by
  sorry

end h_oplus_h_op_h_equals_h_l53_53136


namespace sum_mod_9_equal_6_l53_53126

theorem sum_mod_9_equal_6 :
  ((1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888) % 9) = 6 :=
by
  sorry

end sum_mod_9_equal_6_l53_53126


namespace pizza_slices_per_pizza_l53_53734

theorem pizza_slices_per_pizza (num_coworkers slices_per_person num_pizzas : ℕ) (h1 : num_coworkers = 12) (h2 : slices_per_person = 2) (h3 : num_pizzas = 3) :
  (num_coworkers * slices_per_person) / num_pizzas = 8 :=
by
  sorry

end pizza_slices_per_pizza_l53_53734


namespace minimum_BC_length_l53_53791

theorem minimum_BC_length (AB AC DC BD BC : ℕ)
  (h₁ : AB = 5) (h₂ : AC = 12) (h₃ : DC = 8) (h₄ : BD = 20) (h₅ : BC > 12) : BC = 13 :=
by
  sorry

end minimum_BC_length_l53_53791


namespace sum_of_perimeters_correct_l53_53300

noncomputable def sum_of_perimeters (s w : ℝ) : ℝ :=
  let l := 2 * w
  let square_area := s^2
  let rectangle_area := l * w
  let sq_perimeter := 4 * s
  let rect_perimeter := 2 * l + 2 * w
  sq_perimeter + rect_perimeter

theorem sum_of_perimeters_correct (s w : ℝ) (h1 : s^2 + 2 * w^2 = 130) (h2 : s^2 - 2 * w^2 = 50) :
  sum_of_perimeters s w = 12 * Real.sqrt 10 + 12 * Real.sqrt 5 :=
by sorry

end sum_of_perimeters_correct_l53_53300


namespace find_a_given_integer_roots_l53_53861

-- Given polynomial equation and the condition of integer roots
theorem find_a_given_integer_roots (a : ℤ) :
    (∃ x y : ℤ, x ≠ y ∧ (x^2 - (a+8)*x + 8*a - 1 = 0) ∧ (y^2 - (a+8)*y + 8*a - 1 = 0)) → 
    a = 8 := 
by
  sorry

end find_a_given_integer_roots_l53_53861


namespace problem_statement_l53_53767

-- Proposition p: For any x ∈ ℝ, 2^x > x^2
def p : Prop := ∀ x : ℝ, 2 ^ x > x ^ 2

-- Proposition q: "ab > 4" is a sufficient but not necessary condition for "a > 2 and b > 2"
def q : Prop := (∀ a b : ℝ, (a > 2 ∧ b > 2) → (a * b > 4)) ∧ ¬ (∀ a b : ℝ, (a * b > 4) → (a > 2 ∧ b > 2))

-- Problem statement: Determine that the true statement is ¬p ∧ ¬q
theorem problem_statement : ¬p ∧ ¬q := by
  sorry

end problem_statement_l53_53767


namespace simplify_and_rationalize_l53_53789

theorem simplify_and_rationalize :
  ( (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 11) * (Real.sqrt 9 / Real.sqrt 13) = 
    (3 * Real.sqrt 15015) / 1001 ) :=
by
  sorry

end simplify_and_rationalize_l53_53789


namespace train_pass_platform_time_l53_53567

theorem train_pass_platform_time :
  ∀ (length_train length_platform speed_time_cross_tree speed_train pass_time : ℕ), 
  length_train = 1200 →
  length_platform = 300 →
  speed_time_cross_tree = 120 →
  speed_train = length_train / speed_time_cross_tree →
  pass_time = (length_train + length_platform) / speed_train →
  pass_time = 150 :=
by
  intros
  sorry

end train_pass_platform_time_l53_53567


namespace three_layers_coverage_l53_53445

/--
Three table runners have a combined area of 208 square inches. 
By overlapping the runners to cover 80% of a table of area 175 square inches, 
the area that is covered by exactly two layers of runner is 24 square inches. 
Prove that the area of the table that is covered with three layers of runner is 22 square inches.
--/
theorem three_layers_coverage :
  ∀ (A T two_layers total_table_coverage : ℝ),
  A = 208 ∧ total_table_coverage = 0.8 * 175 ∧ two_layers = 24 →
  A = (total_table_coverage - two_layers - T) + 2 * two_layers + 3 * T →
  T = 22 :=
by
  intros A T two_layers total_table_coverage h1 h2
  sorry

end three_layers_coverage_l53_53445


namespace ratio_of_new_time_to_previous_time_l53_53450

noncomputable def distance : ℝ := 420
noncomputable def previous_time : ℝ := 7
noncomputable def speed_increase : ℝ := 40

-- Original speed
noncomputable def original_speed : ℝ := distance / previous_time

-- New speed
noncomputable def new_speed : ℝ := original_speed + speed_increase

-- New time taken to cover the same distance at the new speed
noncomputable def new_time : ℝ := distance / new_speed

-- Ratio of new time to previous time
noncomputable def time_ratio : ℝ := new_time / previous_time

theorem ratio_of_new_time_to_previous_time :
  time_ratio = 0.6 :=
by sorry

end ratio_of_new_time_to_previous_time_l53_53450


namespace sum_of_two_numbers_l53_53548

theorem sum_of_two_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x * y = 9) (h2 : (1 / x) = 4 * (1 / y)) : x + y = 15 / 2 :=
  sorry

end sum_of_two_numbers_l53_53548


namespace find_side_length_l53_53435

theorem find_side_length
  (X : ℕ)
  (h1 : 3 + 2 + X + 4 = 12) :
  X = 3 :=
by
  sorry

end find_side_length_l53_53435


namespace trapezium_area_l53_53101

-- Definitions based on the problem conditions
def length_side_a : ℝ := 20
def length_side_b : ℝ := 18
def distance_between_sides : ℝ := 15

-- Statement of the proof problem
theorem trapezium_area :
  (1 / 2 * (length_side_a + length_side_b) * distance_between_sides) = 285 := by
  sorry

end trapezium_area_l53_53101


namespace tangent_lines_to_circle_l53_53045

theorem tangent_lines_to_circle 
  (x y : ℝ) 
  (circle : (x - 2) ^ 2 + (y + 1) ^ 2 = 1) 
  (point : x = 3 ∧ y = 3) : 
  (x = 3 ∨ 15 * x - 8 * y - 21 = 0) :=
sorry

end tangent_lines_to_circle_l53_53045


namespace distance_between_towns_proof_l53_53506

noncomputable def distance_between_towns : ℕ :=
  let distance := 300
  let time_after_departure := 2
  let remaining_distance := 40
  let speed_difference := 10
  let total_distance_covered := distance - remaining_distance
  let speed_slower_train := 60
  let speed_faster_train := speed_slower_train + speed_difference
  let relative_speed := speed_slower_train + speed_faster_train
  distance

theorem distance_between_towns_proof 
  (distance : ℕ) 
  (time_after_departure : ℕ) 
  (remaining_distance : ℕ) 
  (speed_difference : ℕ) 
  (h1 : distance = 300) 
  (h2 : time_after_departure = 2) 
  (h3 : remaining_distance = 40) 
  (h4 : speed_difference = 10) 
  (speed_slower_train speed_faster_train relative_speed : ℕ)
  (h_speed_faster : speed_faster_train = speed_slower_train + speed_difference)
  (h_relative_speed : relative_speed = speed_slower_train + speed_faster_train) :
  distance = 300 :=
by {
  sorry
}

end distance_between_towns_proof_l53_53506


namespace darkest_cell_product_l53_53294

theorem darkest_cell_product (a b c d : ℕ)
  (h1 : a > 1) (h2 : b > 1) (h3 : c = a * b)
  (h4 : d = c * (9 * 5) * (9 * 11)) :
  d = 245025 :=
by
  sorry

end darkest_cell_product_l53_53294


namespace range_of_m_l53_53193

def is_ellipse (m : ℝ) : Prop :=
  ∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

theorem range_of_m (m : ℝ) (h : is_ellipse m) : m > 5 :=
sorry

end range_of_m_l53_53193


namespace blue_crayons_l53_53018

variables (B G : ℕ)

theorem blue_crayons (h1 : 24 = 8 + B + G + 6) (h2 : G = (2 / 3) * B) : B = 6 :=
by 
-- This is where the proof would go
sorry

end blue_crayons_l53_53018


namespace total_photos_newspaper_l53_53356

-- Define the conditions
def num_pages1 := 12
def photos_per_page1 := 2
def num_pages2 := 9
def photos_per_page2 := 3

-- Define the total number of photos
def total_photos := (num_pages1 * photos_per_page1) + (num_pages2 * photos_per_page2)

-- Prove that the total number of photos is 51
theorem total_photos_newspaper : total_photos = 51 := by
  -- We will fill the proof later
  sorry

end total_photos_newspaper_l53_53356


namespace total_female_officers_l53_53700

theorem total_female_officers
  (percent_female_on_duty : ℝ)
  (total_on_duty : ℝ)
  (half_of_total_on_duty : ℝ)
  (num_females_on_duty : ℝ) :
  percent_female_on_duty = 0.10 →
  total_on_duty = 200 →
  half_of_total_on_duty = total_on_duty / 2 →
  num_females_on_duty = half_of_total_on_duty →
  num_females_on_duty = percent_female_on_duty * (1000 : ℝ) :=
by
  intros h1 h2 h3 h4
  sorry

end total_female_officers_l53_53700


namespace john_total_spending_l53_53033

def t_shirt_price : ℝ := 20
def num_t_shirts : ℝ := 3
def t_shirt_offer_discount : ℝ := 0.50
def t_shirt_total_cost : ℝ := (2 * t_shirt_price) + (t_shirt_price * t_shirt_offer_discount)

def pants_price : ℝ := 50
def num_pants : ℝ := 2
def pants_total_cost : ℝ := pants_price * num_pants

def jacket_original_price : ℝ := 80
def jacket_discount : ℝ := 0.25
def jacket_total_cost : ℝ := jacket_original_price * (1 - jacket_discount)

def hat_price : ℝ := 15

def shoes_original_price : ℝ := 60
def shoes_discount : ℝ := 0.10
def shoes_total_cost : ℝ := shoes_original_price * (1 - shoes_discount)

def clothes_tax_rate : ℝ := 0.05
def shoes_tax_rate : ℝ := 0.08

def clothes_total_cost : ℝ := t_shirt_total_cost + pants_total_cost + jacket_total_cost + hat_price
def total_cost_before_tax : ℝ := clothes_total_cost + shoes_total_cost

def clothes_tax : ℝ := clothes_total_cost * clothes_tax_rate
def shoes_tax : ℝ := shoes_total_cost * shoes_tax_rate

def total_cost_including_tax : ℝ := total_cost_before_tax + clothes_tax + shoes_tax

theorem john_total_spending :
  total_cost_including_tax = 294.57 := by
  sorry

end john_total_spending_l53_53033


namespace ducks_in_garden_l53_53210

theorem ducks_in_garden (num_rabbits : ℕ) (num_ducks : ℕ) 
  (total_legs : ℕ)
  (rabbit_legs : ℕ) (duck_legs : ℕ) 
  (H1 : num_rabbits = 9)
  (H2 : rabbit_legs = 4)
  (H3 : duck_legs = 2)
  (H4 : total_legs = 48)
  (H5 : num_rabbits * rabbit_legs + num_ducks * duck_legs = total_legs) :
  num_ducks = 6 := 
by {
  sorry
}

end ducks_in_garden_l53_53210


namespace greatest_value_of_n_l53_53456

theorem greatest_value_of_n : ∀ (n : ℤ), 102 * n^2 ≤ 8100 → n ≤ 8 :=
by 
  sorry

end greatest_value_of_n_l53_53456


namespace geometric_series_common_ratio_l53_53127

theorem geometric_series_common_ratio (a r : ℝ) (h : a / (1 - r) = 64 * (a * r^4 / (1 - r))) : r = 1/2 :=
by {
  sorry
}

end geometric_series_common_ratio_l53_53127


namespace ellipse_minor_axis_length_l53_53490

noncomputable def minor_axis_length (a b : ℝ) (eccentricity : ℝ) (sum_distances : ℝ) :=
  if (a > b ∧ b > 0 ∧ eccentricity = (Real.sqrt 5) / 3 ∧ sum_distances = 12) then
    2 * b
  else
    0

theorem ellipse_minor_axis_length (a b : ℝ) (eccentricity : ℝ) (sum_distances : ℝ)
  (h1 : a > b) (h2 : b > 0) (h3 : eccentricity = (Real.sqrt 5) / 3) (h4 : sum_distances = 12) :
  minor_axis_length a b eccentricity sum_distances = 8 :=
sorry

end ellipse_minor_axis_length_l53_53490


namespace max_quarters_l53_53263

theorem max_quarters (a b c : ℕ) (h1 : a + b + c = 120) (h2 : 5 * a + 10 * b + 25 * c = 1000) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c) : c ≤ 19 :=
sorry

example : ∃ a b c : ℕ, a + b + c = 120 ∧ 5 * a + 10 * b + 25 * c = 1000 ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ c = 19 :=
sorry

end max_quarters_l53_53263


namespace line_intersects_axes_l53_53254

theorem line_intersects_axes (a b : ℝ) (x1 y1 x2 y2 : ℝ) (h_points : (x1, y1) = (8, 2) ∧ (x2, y2) = (4, 6)) :
  (∃ x_intercept : ℝ, (x_intercept, 0) = (10, 0)) ∧ (∃ y_intercept : ℝ, (0, y_intercept) = (0, 10)) :=
by
  sorry

end line_intersects_axes_l53_53254


namespace rectangle_width_solution_l53_53390

noncomputable def solve_rectangle_width (W L w l : ℝ) :=
  L = 2 * W ∧ 3 * w = W ∧ 2 * l = L ∧ 6 * l * w = 5400

theorem rectangle_width_solution (W L w l : ℝ) :
  solve_rectangle_width W L w l → w = 10 * Real.sqrt 3 :=
by
  sorry

end rectangle_width_solution_l53_53390


namespace ratio_of_segments_l53_53553

theorem ratio_of_segments (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end ratio_of_segments_l53_53553


namespace population_net_increase_l53_53442

def birth_rate : ℕ := 8
def birth_time : ℕ := 2
def death_rate : ℕ := 6
def death_time : ℕ := 2
def seconds_per_minute : ℕ := 60
def minutes_per_hour : ℕ := 60
def hours_per_day : ℕ := 24

theorem population_net_increase :
  (birth_rate / birth_time - death_rate / death_time) * (seconds_per_minute * minutes_per_hour * hours_per_day) = 86400 :=
by
  sorry

end population_net_increase_l53_53442


namespace only_n1_makes_n4_plus4_prime_l53_53098

theorem only_n1_makes_n4_plus4_prime (n : ℕ) (h : n > 0) : (n = 1) ↔ Prime (n^4 + 4) :=
sorry

end only_n1_makes_n4_plus4_prime_l53_53098


namespace prism_volume_l53_53104

theorem prism_volume (x : ℝ) (L W H : ℝ) (hL : L = 2 * x) (hW : W = x) (hH : H = 1.5 * x) 
  (hedges_sum : 4 * L + 4 * W + 4 * H = 72) : 
  L * W * H = 192 := 
by
  sorry

end prism_volume_l53_53104


namespace alice_score_record_l53_53519

def total_points : ℝ := 72
def average_points_others : ℝ := 4.7
def others_count : ℕ := 7

def total_points_others : ℝ := others_count * average_points_others
def alice_points : ℝ := total_points - total_points_others

theorem alice_score_record : alice_points = 39.1 :=
by {
  -- Proof should be inserted here
  sorry
}

end alice_score_record_l53_53519


namespace pete_ate_percentage_l53_53511

-- Definitions of the conditions
def total_slices : ℕ := 2 * 12
def stephen_ate_slices : ℕ := (25 * total_slices) / 100
def remaining_slices_after_stephen : ℕ := total_slices - stephen_ate_slices
def slices_left_after_pete : ℕ := 9

-- The statement to be proved
theorem pete_ate_percentage (h1 : total_slices = 24)
                            (h2 : stephen_ate_slices = 6)
                            (h3 : remaining_slices_after_stephen = 18)
                            (h4 : slices_left_after_pete = 9) :
  ((remaining_slices_after_stephen - slices_left_after_pete) * 100 / remaining_slices_after_stephen) = 50 :=
sorry

end pete_ate_percentage_l53_53511


namespace pass_rate_eq_l53_53305

theorem pass_rate_eq (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : (1 - a) * (1 - b) = ab - a - b + 1 :=
by
  sorry

end pass_rate_eq_l53_53305


namespace tan_45_eq_one_l53_53500

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l53_53500


namespace turtle_distance_during_rabbit_rest_l53_53670

theorem turtle_distance_during_rabbit_rest
  (D : ℕ)
  (vr vt : ℕ)
  (rabbit_speed_multiple : vr = 15 * vt)
  (rabbit_remaining_distance : D - 100 = 900)
  (turtle_finish_time : true)
  (rabbit_to_be_break : true)
  (turtle_finish_during_rabbit_rest : true) :
  (D - (900 / 15) = 940) :=
by
  sorry

end turtle_distance_during_rabbit_rest_l53_53670


namespace adult_tickets_sold_l53_53333

theorem adult_tickets_sold (A C : ℕ) (h1 : A + C = 85) (h2 : 5 * A + 2 * C = 275) : A = 35 := by
  sorry

end adult_tickets_sold_l53_53333


namespace value_of_a_l53_53216

/--
Given that x = 3 is a solution to the equation 3x - 2a = 5,
prove that a = 2.
-/
theorem value_of_a (x a : ℤ) (h : 3 * x - 2 * a = 5) (hx : x = 3) : a = 2 :=
by
  sorry

end value_of_a_l53_53216


namespace range_of_first_term_l53_53112

-- Define the arithmetic sequence and its common difference.
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Define the sum of the first n terms of the sequence.
def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a + (n - 1) * d)) / 2

-- Prove the range of the first term a1 given the conditions.
theorem range_of_first_term (a d : ℤ) (S : ℕ → ℤ) (h1 : d = -2)
  (h2 : ∀ n, S n = sum_of_first_n_terms a d n)
  (h3 : S 7 = S 7)
  (h4 : ∀ n, n ≠ 7 → S n < S 7) :
  12 < a ∧ a < 14 :=
by
  sorry

end range_of_first_term_l53_53112


namespace cone_curved_surface_area_l53_53771

def radius (r : ℝ) := r = 3
def slantHeight (l : ℝ) := l = 15
def curvedSurfaceArea (csa : ℝ) := csa = 45 * Real.pi

theorem cone_curved_surface_area 
  (r l csa : ℝ) 
  (hr : radius r) 
  (hl : slantHeight l) 
  : curvedSurfaceArea (Real.pi * r * l) 
  := by
  unfold radius at hr
  unfold slantHeight at hl
  unfold curvedSurfaceArea
  rw [hr, hl]
  norm_num
  sorry

end cone_curved_surface_area_l53_53771


namespace subset_definition_l53_53884

variable {α : Type} {A B : Set α}

theorem subset_definition :
  A ⊆ B ↔ ∀ a ∈ A, a ∈ B :=
by sorry

end subset_definition_l53_53884


namespace min_value_fraction_sum_l53_53635

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (h_collinear : 3 * a + 2 * b = 1)

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_collinear : 3 * a + 2 * b = 1) : 
  (3 / a + 1 / b) = 11 + 6 * Real.sqrt 2 :=
by
  sorry

end min_value_fraction_sum_l53_53635


namespace find_sale4_l53_53859

variable (sale1 sale2 sale3 sale5 sale6 avg : ℕ)
variable (total_sales : ℕ := 6 * avg)
variable (known_sales : ℕ := sale1 + sale2 + sale3 + sale5 + sale6)
variable (sale4 : ℕ := total_sales - known_sales)

theorem find_sale4 (h1 : sale1 = 6235) (h2 : sale2 = 6927) (h3 : sale3 = 6855)
                   (h5 : sale5 = 6562) (h6 : sale6 = 5191) (h_avg : avg = 6500) :
  sale4 = 7225 :=
by 
  sorry

end find_sale4_l53_53859


namespace total_animals_count_l53_53894

theorem total_animals_count (a m : ℕ) (h1 : a = 35) (h2 : a + 7 = m) : a + m = 77 :=
by
  sorry

end total_animals_count_l53_53894


namespace range_of_m_l53_53996

def p (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0 ∧ m > 0 
def neg_q_sufficient_for_neg_p (m : ℝ) : Prop :=
  ∀ x : ℝ, p x → q x m

theorem range_of_m (m : ℝ) : neg_q_sufficient_for_neg_p m → m ≥ 9 :=
by
  sorry

end range_of_m_l53_53996


namespace speed_of_stream_l53_53120

theorem speed_of_stream (v : ℝ) :
  (∀ s : ℝ, s = 3 → (3 + v) / (3 - v) = 2) → v = 1 :=
by 
  intro h
  sorry

end speed_of_stream_l53_53120


namespace figure_50_unit_squares_l53_53252

-- Definitions reflecting the conditions from step A
def f (n : ℕ) := (1/2 : ℚ) * n^3 + (7/2 : ℚ) * n + 1

theorem figure_50_unit_squares : f 50 = 62676 := by
  sorry

end figure_50_unit_squares_l53_53252


namespace total_gumballs_l53_53971

-- Define the count of red, blue, and green gumballs
def red_gumballs := 16
def blue_gumballs := red_gumballs / 2
def green_gumballs := blue_gumballs * 4

-- Prove that the total number of gumballs is 56
theorem total_gumballs : red_gumballs + blue_gumballs + green_gumballs = 56 := by
  sorry

end total_gumballs_l53_53971


namespace temperature_max_time_l53_53118

theorem temperature_max_time (t : ℝ) (h : 0 ≤ t) : 
  (-t^2 + 10 * t + 60 = 85) → t = 15 := 
sorry

end temperature_max_time_l53_53118


namespace factorize_expression_l53_53122

theorem factorize_expression (m n : ℤ) : 
  4 * m^2 * n - 4 * n^3 = 4 * n * (m + n) * (m - n) :=
by
  sorry

end factorize_expression_l53_53122


namespace arithmetic_geometric_common_ratio_l53_53474

theorem arithmetic_geometric_common_ratio (a₁ r : ℝ) 
  (h₁ : a₁ + a₁ * r^2 = 10) 
  (h₂ : a₁ * (1 + r + r^2 + r^3) = 15) : 
  r = 1/2 ∨ r = -1/2 :=
by {
  sorry
}

end arithmetic_geometric_common_ratio_l53_53474


namespace find_line_equation_l53_53436

-- define the condition of passing through the point (-3, -1)
def passes_through (x y : ℝ) (a b : ℝ) := (a = -3) ∧ (b = -1)

-- define the condition of being parallel to the line x - 3y - 1 = 0
def is_parallel (m n c : ℝ) := (m = 1) ∧ (n = -3)

-- theorem statement
theorem find_line_equation (a b : ℝ) (c : ℝ) :
  passes_through a b (-3) (-1) →
  is_parallel 1 (-3) c →
  (a - 3 * b + c = 0) :=
sorry

end find_line_equation_l53_53436


namespace number_of_boys_in_school_l53_53918

theorem number_of_boys_in_school (x g : ℕ) (h1 : x + g = 400) (h2 : g = (x * 400) / 100) : x = 80 :=
by
  sorry

end number_of_boys_in_school_l53_53918


namespace apples_in_box_l53_53453

theorem apples_in_box :
  (∀ (o p a : ℕ), 
    (o = 1 / 4 * 56) ∧ 
    (p = 1 / 2 * o) ∧ 
    (a = 5 * p) → 
    a = 35) :=
  by sorry

end apples_in_box_l53_53453


namespace find_initial_amount_l53_53990

-- defining conditions
def compound_interest (A P : ℝ) (r : ℝ) (n t : ℕ) : ℝ :=
  A - P

-- main theorem to prove the principal amount
theorem find_initial_amount 
  (A P : ℝ) (r : ℝ)
  (n t : ℕ)
  (h_P : A = P * (1 + r / n)^t)
  (compound_interest_eq : A - P = 1785.98)
  (r_eq : r = 0.20)
  (n_eq : n = 1)
  (t_eq : t = 5) :
  P = 1200 :=
by
  sorry

end find_initial_amount_l53_53990


namespace necessary_but_not_sufficient_condition_l53_53209

open Classical

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (a > 1 ∧ b > 3) → (a + b > 4) ∧ ¬((a + b > 4) → (a > 1 ∧ b > 3)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l53_53209


namespace find_f_neg_2_l53_53751

def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 4

variable (a b : ℝ)

theorem find_f_neg_2 (h1 : f a b 2 = 6) : f a b (-2) = -14 :=
by
  sorry

end find_f_neg_2_l53_53751


namespace taehyung_math_score_l53_53823

theorem taehyung_math_score
  (avg_before : ℝ)
  (drop_in_avg : ℝ)
  (num_subjects_before : ℕ)
  (num_subjects_after : ℕ)
  (avg_after : ℝ)
  (total_before : ℝ)
  (total_after : ℝ)
  (math_score : ℝ) :
  avg_before = 95 →
  drop_in_avg = 3 →
  num_subjects_before = 3 →
  num_subjects_after = 4 →
  avg_after = avg_before - drop_in_avg →
  total_before = avg_before * num_subjects_before →
  total_after = avg_after * num_subjects_after →
  math_score = total_after - total_before →
  math_score = 83 :=
by
  intros
  sorry

end taehyung_math_score_l53_53823


namespace circle_intersection_range_l53_53011

theorem circle_intersection_range (a : ℝ) :
  (0 < a ∧ a < 2 * Real.sqrt 2) ∨ (-2 * Real.sqrt 2 < a ∧ a < 0) ↔
  (let C := { p : ℝ × ℝ | (p.1 - a) ^ 2 + (p.2 - a) ^ 2 = 4 };
   let O := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 4 };
   ∀ p, p ∈ C → p ∈ O) :=
sorry

end circle_intersection_range_l53_53011


namespace waiters_dropped_out_l53_53936

theorem waiters_dropped_out (initial_chefs initial_waiters chefs_dropped remaining_staff : ℕ)
  (h1 : initial_chefs = 16) 
  (h2 : initial_waiters = 16) 
  (h3 : chefs_dropped = 6) 
  (h4 : remaining_staff = 23) : 
  initial_waiters - (remaining_staff - (initial_chefs - chefs_dropped)) = 3 := 
by 
  sorry

end waiters_dropped_out_l53_53936


namespace irrational_number_problem_l53_53899

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_number_problem :
  ∀ x ∈ ({(0.4 : ℝ), (2 / 3 : ℝ), (2 : ℝ), - (Real.sqrt 5)} : Set ℝ), 
  is_irrational x ↔ x = - (Real.sqrt 5) :=
by
  intros x hx
  -- Other proof steps can go here
  sorry

end irrational_number_problem_l53_53899


namespace all_equal_l53_53994

theorem all_equal (n : ℕ) (a : ℕ → ℝ) (h1 : 3 < n)
  (h2 : ∀ k : ℕ, k < n -> (a k)^3 = (a (k + 1 % n))^2 + (a (k + 2 % n))^2 + (a (k + 3 % n))^2) : 
  ∀ i j : ℕ, i < n -> j < n -> a i = a j :=
by
  sorry

end all_equal_l53_53994


namespace infinite_expressible_terms_l53_53019

theorem infinite_expressible_terms
  (a : ℕ → ℕ)
  (h1 : ∀ n, a n < a (n + 1)) :
  ∃ f : ℕ → ℕ, (∀ n, a (f n) = (f n).succ * a 1 + (f n).succ.succ * a 2) ∧
    ∀ i j, i ≠ j → f i ≠ f j :=
by
  sorry

end infinite_expressible_terms_l53_53019


namespace base_length_of_triangle_l53_53778

theorem base_length_of_triangle (height area : ℕ) (h1 : height = 8) (h2 : area = 24) : 
  ∃ base : ℕ, (1/2 : ℚ) * base * height = area ∧ base = 6 := by
  sorry

end base_length_of_triangle_l53_53778


namespace abs_x_minus_one_iff_x_in_interval_l53_53148

theorem abs_x_minus_one_iff_x_in_interval (x : ℝ) :
  |x - 1| < 2 ↔ (x + 1) * (x - 3) < 0 :=
by
  sorry

end abs_x_minus_one_iff_x_in_interval_l53_53148


namespace inequality_holds_equality_condition_l53_53849

variables {x y z : ℝ}
-- Assuming positive real numbers and the given condition
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom pos_z : 0 < z
axiom h : x * y + y * z + z * x = x + y + z

theorem inequality_holds : 
  (1 / (x^2 + y + 1)) + (1 / (y^2 + z + 1)) + (1 / (z^2 + x + 1)) ≤ 1 :=
by
  sorry

theorem equality_condition : 
  (1 / (x^2 + y + 1)) + (1 / (y^2 + z + 1)) + (1 / (z^2 + x + 1)) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end inequality_holds_equality_condition_l53_53849


namespace rate_percent_correct_l53_53817

noncomputable def findRatePercent (P A T : ℕ) : ℚ :=
  let SI := A - P
  (SI * 100 : ℚ) / (P * T)

theorem rate_percent_correct :
  findRatePercent 12000 19500 7 = 8.93 := by
  sorry

end rate_percent_correct_l53_53817


namespace shortest_distance_parabola_to_line_l53_53977

open Real

theorem shortest_distance_parabola_to_line :
  ∃ (d : ℝ), 
    (∀ (P : ℝ × ℝ), (P.1 = (P.2^2) / 8) → 
      ((2 * P.1 - P.2 - 4) / sqrt 5 ≥ d)) ∧ 
    (d = 3 * sqrt 5 / 5) :=
sorry

end shortest_distance_parabola_to_line_l53_53977


namespace temperature_reaches_100_at_5_hours_past_noon_l53_53784

theorem temperature_reaches_100_at_5_hours_past_noon :
  ∃ t : ℝ, (-2 * t^2 + 16 * t + 40 = 100) ∧ ∀ t' : ℝ, (-2 * t'^2 + 16 * t' + 40 = 100) → 5 ≤ t' :=
by
  -- We skip the proof and assume the theorem is true.
  sorry

end temperature_reaches_100_at_5_hours_past_noon_l53_53784


namespace find_pairs_l53_53730

theorem find_pairs (m n : ℕ) : 
  (20^m - 10 * m^2 + 1 = 19^n ↔ (m = 0 ∧ n = 0) ∨ (m = 2 ∧ n = 2)) :=
by
  sorry

end find_pairs_l53_53730


namespace greening_task_equation_l53_53420

variable (x : ℝ)

theorem greening_task_equation (h1 : 600000 = 600 * 1000)
    (h2 : ∀ a b : ℝ, a * 1.25 = b -> b = a * (1 + 25 / 100)) :
  (60 * (1 + 25 / 100)) / x - 60 / x = 30 := by
  sorry

end greening_task_equation_l53_53420


namespace exponential_fixed_point_l53_53336

theorem exponential_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : (a^(4-4) + 5 = 6) :=
sorry

end exponential_fixed_point_l53_53336


namespace min_c_value_l53_53415

def y_eq_abs_sum (x a b c : ℝ) : ℝ := |x - a| + |x - b| + |x - c|
def y_eq_line (x : ℝ) : ℝ := -2 * x + 2023

theorem min_c_value (a b c : ℕ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (order : a ≤ b ∧ b < c)
  (unique_sol : ∃! x : ℝ, y_eq_abs_sum x a b c = y_eq_line x) :
  c = 2022 := sorry

end min_c_value_l53_53415


namespace find_x_l53_53573

theorem find_x (x : ℕ) (h1 : x ≥ 10) (h2 : x > 8) : x = 9 := by
  sorry

end find_x_l53_53573


namespace faster_pipe_rate_l53_53365

-- Set up our variables and the condition
variable (F S : ℝ)
variable (n : ℕ)

-- Given conditions
axiom S_rate : S = 1 / 180
axiom combined_rate : F + S = 1 / 36
axiom faster_rate : F = n * S

-- Theorem to prove
theorem faster_pipe_rate : n = 4 := by
  sorry

end faster_pipe_rate_l53_53365


namespace apples_per_pie_l53_53517

theorem apples_per_pie (total_apples : ℕ) (unripe_apples : ℕ) (pies : ℕ) (ripe_apples : ℕ)
  (H1 : total_apples = 34)
  (H2 : unripe_apples = 6)
  (H3 : pies = 7)
  (H4 : ripe_apples = total_apples - unripe_apples) :
  ripe_apples / pies = 4 := by
  sorry

end apples_per_pie_l53_53517


namespace range_of_m_l53_53073

theorem range_of_m (m : ℝ) : ((m + 3 > 0) ∧ (m - 1 < 0)) ↔ (-3 < m ∧ m < 1) :=
by
  sorry

end range_of_m_l53_53073


namespace correct_result_l53_53069

-- Define the original number
def original_number := 51 + 6

-- Define the correct calculation using multiplication
def correct_calculation (x : ℕ) : ℕ := x * 6

-- Theorem to prove the correct calculation
theorem correct_result : correct_calculation original_number = 342 := by
  -- Skip the actual proof steps
  sorry

end correct_result_l53_53069


namespace turtle_population_2002_l53_53108

theorem turtle_population_2002 (k : ℝ) (y : ℝ)
  (h1 : 58 + k * 92 = y)
  (h2 : 179 - 92 = k * y) 
  : y = 123 :=
by
  sorry

end turtle_population_2002_l53_53108


namespace rob_has_24_cards_l53_53177

theorem rob_has_24_cards 
  (r : ℕ) -- total number of baseball cards Rob has
  (dr : ℕ) -- number of doubles Rob has
  (hj: dr = 1 / 3 * r) -- one third of Rob's cards are doubles
  (jess_doubles : ℕ) -- number of doubles Jess has
  (hj_mult : jess_doubles = 5 * dr) -- Jess has 5 times as many doubles as Rob
  (jess_doubles_40 : jess_doubles = 40) -- Jess has 40 doubles baseball cards
: r = 24 :=
by
  sorry

end rob_has_24_cards_l53_53177


namespace upstream_distance_is_48_l53_53833

variables (distance_downstream time_downstream time_upstream speed_stream : ℝ)
variables (speed_boat distance_upstream : ℝ)

-- Given conditions
axiom h1 : distance_downstream = 84
axiom h2 : time_downstream = 2
axiom h3 : time_upstream = 2
axiom h4 : speed_stream = 9

-- Define the effective speeds
def speed_downstream (speed_boat speed_stream : ℝ) := speed_boat + speed_stream
def speed_upstream (speed_boat speed_stream : ℝ) := speed_boat - speed_stream

-- Equations based on travel times and distances
axiom eq1 : distance_downstream = (speed_downstream speed_boat speed_stream) * time_downstream
axiom eq2 : distance_upstream = (speed_upstream speed_boat speed_stream) * time_upstream

-- Theorem to prove the distance rowed upstream is 48 km
theorem upstream_distance_is_48 :
  distance_upstream = 48 :=
by
  sorry

end upstream_distance_is_48_l53_53833


namespace vojta_correct_sum_l53_53124

theorem vojta_correct_sum (S A B C : ℕ)
  (h1 : S + (10 * B + C) = 2224)
  (h2 : S + (10 * A + B) = 2198)
  (h3 : S + (10 * A + C) = 2204)
  (A_digit : 0 ≤ A ∧ A < 10)
  (B_digit : 0 ≤ B ∧ B < 10)
  (C_digit : 0 ≤ C ∧ C < 10) :
  S + 100 * A + 10 * B + C = 2324 := 
sorry

end vojta_correct_sum_l53_53124
