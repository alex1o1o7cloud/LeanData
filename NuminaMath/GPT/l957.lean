import Mathlib

namespace NUMINAMATH_GPT_square_root_value_l957_95724

-- Define the problem conditions
def x : ℝ := 5

-- Prove the solution
theorem square_root_value : (Real.sqrt (x - 3)) = Real.sqrt 2 :=
by
  -- Proof steps skipped
  sorry

end NUMINAMATH_GPT_square_root_value_l957_95724


namespace NUMINAMATH_GPT_stratified_sampling_height_group_selection_l957_95780

theorem stratified_sampling_height_group_selection :
  let total_students := 100
  let group1 := 20
  let group2 := 50
  let group3 := 30
  let total_selected := 18
  group1 + group2 + group3 = total_students →
  (group3 : ℝ) / total_students * total_selected = 5.4 →
  round ((group3 : ℝ) / total_students * total_selected) = 3 :=
by
  intros total_students group1 group2 group3 total_selected h1 h2
  sorry

end NUMINAMATH_GPT_stratified_sampling_height_group_selection_l957_95780


namespace NUMINAMATH_GPT_triple_overlap_area_correct_l957_95746

-- Define the dimensions of the auditorium and carpets
def auditorium_dim : ℕ × ℕ := (10, 10)
def carpet1_dim : ℕ × ℕ := (6, 8)
def carpet2_dim : ℕ × ℕ := (6, 6)
def carpet3_dim : ℕ × ℕ := (5, 7)

-- The coordinates and dimensions of the overlap regions are derived based on the given positions
-- Here we assume derivations as described in the solution steps without recalculating them

-- Overlap area of the second and third carpets
def overlap23 : ℕ × ℕ := (5, 3)

-- Intersection of this overlap with the first carpet
def overlap_all : ℕ × ℕ := (2, 3)

-- Calculate the area of the region where all three carpets overlap
def triple_overlap_area : ℕ :=
  (overlap_all.1 * overlap_all.2)

theorem triple_overlap_area_correct :
  triple_overlap_area = 6 := by
  -- Expected result should be 6 square meters
  sorry

end NUMINAMATH_GPT_triple_overlap_area_correct_l957_95746


namespace NUMINAMATH_GPT_solve_for_x_l957_95737

theorem solve_for_x (x : ℚ) (h : (3 - x)/(x + 2) + (3 * x - 6)/(3 - x) = 2) : x = -7/6 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l957_95737


namespace NUMINAMATH_GPT_value_of_expression_l957_95722

-- Definitions of the variables x and y along with their assigned values
def x : ℕ := 20
def y : ℕ := 8

-- The theorem that asserts the value of (x - y) * (x + y) equals 336
theorem value_of_expression : (x - y) * (x + y) = 336 := by 
  -- Skipping proof
  sorry

end NUMINAMATH_GPT_value_of_expression_l957_95722


namespace NUMINAMATH_GPT_train_crossing_time_l957_95790

noncomputable def time_to_cross_bridge (length_train : ℝ) (length_bridge : ℝ) (speed_kmh : ℝ) : ℝ :=
  let total_distance := length_train + length_bridge
  let speed_ms := (speed_kmh * 1000) / 3600
  total_distance / speed_ms

theorem train_crossing_time :
  time_to_cross_bridge 100 145 65 = 13.57 :=
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l957_95790


namespace NUMINAMATH_GPT_cos_of_angle_through_point_l957_95714

-- Define the point P and the angle α
def P : ℝ × ℝ := (4, 3)
def α : ℝ := sorry  -- α is an angle such that its terminal side passes through P

-- Define the squared distance from the origin to the point P
noncomputable def distance_squared : ℝ := P.1^2 + P.2^2

-- Define cos α
noncomputable def cosα : ℝ := P.1 / (Real.sqrt distance_squared)

-- State the theorem
theorem cos_of_angle_through_point : cosα = 4 / 5 := 
by sorry

end NUMINAMATH_GPT_cos_of_angle_through_point_l957_95714


namespace NUMINAMATH_GPT_arithmetic_sequence_201_is_61_l957_95794

def is_arithmetic_sequence_term (a_5 a_45 : ℤ) (n : ℤ) (a_n : ℤ) : Prop :=
  ∃ d a_1, a_1 + 4 * d = a_5 ∧ a_1 + 44 * d = a_45 ∧ a_1 + (n - 1) * d = a_n

theorem arithmetic_sequence_201_is_61 : is_arithmetic_sequence_term 33 153 61 201 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_201_is_61_l957_95794


namespace NUMINAMATH_GPT_hyperbola_standard_equation_l957_95736

def ellipse_equation (x y : ℝ) : Prop :=
  (y^2) / 16 + (x^2) / 12 = 1

def hyperbola_equation (x y : ℝ) : Prop :=
  (y^2) / 2 - (x^2) / 2 = 1

def passes_through_point (x y : ℝ) : Prop :=
  x = 1 ∧ y = Real.sqrt 3

theorem hyperbola_standard_equation (x y : ℝ) (hx : passes_through_point x y)
  (ellipse_foci_shared : ∀ x y : ℝ, ellipse_equation x y → ellipse_equation x y)
  : hyperbola_equation x y := 
sorry

end NUMINAMATH_GPT_hyperbola_standard_equation_l957_95736


namespace NUMINAMATH_GPT_tagged_fish_in_second_catch_l957_95729

theorem tagged_fish_in_second_catch 
  (total_fish : ℕ := 3200) 
  (initial_tagged : ℕ := 80) 
  (second_catch : ℕ := 80) 
  (T : ℕ) 
  (h : (T : ℚ) / second_catch = initial_tagged / total_fish) :
  T = 2 :=
by 
  sorry

end NUMINAMATH_GPT_tagged_fish_in_second_catch_l957_95729


namespace NUMINAMATH_GPT_find_polynomial_value_l957_95792

theorem find_polynomial_value
  (x y : ℝ)
  (h1 : 3 * x + y = 5)
  (h2 : x + 3 * y = 6) :
  5 * x^2 + 8 * x * y + 5 * y^2 = 61 := 
by {
  -- The proof part is omitted here
  sorry
}

end NUMINAMATH_GPT_find_polynomial_value_l957_95792


namespace NUMINAMATH_GPT_product_ab_l957_95791

noncomputable def a : ℝ := 1           -- From the condition 1 = a * tan(π / 4)
noncomputable def b : ℝ := 2           -- From the condition π / b = π / 2

theorem product_ab (a b : ℝ)
  (ha : a > 0) (hb : b > 0)
  (period_condition : (π / b = π / 2))
  (point_condition : a * Real.tan ((π / 8) * b) = 1) :
  a * b = 2 := sorry

end NUMINAMATH_GPT_product_ab_l957_95791


namespace NUMINAMATH_GPT_sheela_monthly_income_l957_95717

-- Definitions from the conditions
def deposited_amount : ℝ := 5000
def percentage_of_income : ℝ := 0.20

-- The theorem to be proven
theorem sheela_monthly_income : (deposited_amount / percentage_of_income) = 25000 := by
  sorry

end NUMINAMATH_GPT_sheela_monthly_income_l957_95717


namespace NUMINAMATH_GPT_solve_for_x_l957_95703

theorem solve_for_x : ∃ x k l : ℕ, (3 * 22 = k) ∧ (66 + l = 90) ∧ (160 * 3 / 4 = x - l) → x = 144 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l957_95703


namespace NUMINAMATH_GPT_product_of_possible_x_l957_95721

theorem product_of_possible_x : 
  (∀ x : ℚ, abs ((18 / x) + 4) = 3 → x = -18 ∨ x = -18 / 7) → 
  ((-18) * (-18 / 7) = 324 / 7) :=
by
  sorry

end NUMINAMATH_GPT_product_of_possible_x_l957_95721


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l957_95782

def point : Type := ℝ × ℝ

def A : point := (2, 1)
def B : point := (1, 4)
def on_line (C : point) : Prop := C.1 + C.2 = 9
def area_triangle (A B C : point) : ℝ := 0.5 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - B.1 * A.2 - C.1 * B.2 - A.1 * C.2)

theorem area_of_triangle_ABC :
  ∃ C : point, on_line C ∧ area_triangle A B C = 2 :=
sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l957_95782


namespace NUMINAMATH_GPT_distance_between_red_lights_l957_95725

def position_of_nth_red (n : ℕ) : ℕ :=
  7 * (n - 1) / 3 + n

def in_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem distance_between_red_lights :
  in_feet ((position_of_nth_red 30 - position_of_nth_red 5) * 8) = 41 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_red_lights_l957_95725


namespace NUMINAMATH_GPT_solve_x_l957_95774

theorem solve_x : ∃ (x : ℚ), (3*x - 17) / 4 = (x + 9) / 6 ∧ x = 69 / 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_x_l957_95774


namespace NUMINAMATH_GPT_intersection_A_B_l957_95743

-- Define sets A and B
def A : Set ℝ := { x | -2 < x ∧ x < 4 }
def B : Set ℝ := { 2, 3, 4, 5 }

-- State the theorem about the intersection A ∩ B
theorem intersection_A_B : A ∩ B = { 2, 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l957_95743


namespace NUMINAMATH_GPT_smallest_circle_covering_region_l957_95775

/-- 
Given the conditions describing the plane region:
1. x ≥ 0
2. y ≥ 0
3. x + 2y - 4 ≤ 0

Prove that the equation of the smallest circle covering this region is (x - 2)² + (y - 1)² = 5.
-/
theorem smallest_circle_covering_region :
  (∀ (x y : ℝ), (x ≥ 0 ∧ y ≥ 0 ∧ x + 2 * y - 4 ≤ 0) → (x - 2)^2 + (y - 1)^2 ≤ 5) :=
sorry

end NUMINAMATH_GPT_smallest_circle_covering_region_l957_95775


namespace NUMINAMATH_GPT_total_students_l957_95797

theorem total_students (teams students_per_team : ℕ) (h1 : teams = 9) (h2 : students_per_team = 18) :
  teams * students_per_team = 162 := by
  sorry

end NUMINAMATH_GPT_total_students_l957_95797


namespace NUMINAMATH_GPT_rectangle_perimeter_l957_95779

theorem rectangle_perimeter
  (w l P : ℝ)
  (h₁ : l = 2 * w)
  (h₂ : l * w = 400) :
  P = 60 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l957_95779


namespace NUMINAMATH_GPT_find_missing_term_l957_95719

theorem find_missing_term (a b : ℕ) : ∃ x, (2 * a - b) * x = 4 * a^2 - b^2 :=
by
  use (2 * a + b)
  sorry

end NUMINAMATH_GPT_find_missing_term_l957_95719


namespace NUMINAMATH_GPT_simplify_expr_1_l957_95789

theorem simplify_expr_1 (a : ℝ) : (2 * a - 3) ^ 2 + (2 * a + 3) * (2 * a - 3) = 8 * a ^ 2 - 12 * a :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_1_l957_95789


namespace NUMINAMATH_GPT_influenza_probability_l957_95735

theorem influenza_probability :
  let flu_rate_A := 0.06
  let flu_rate_B := 0.05
  let flu_rate_C := 0.04
  let population_ratio_A := 6
  let population_ratio_B := 5
  let population_ratio_C := 4
  (population_ratio_A * flu_rate_A + population_ratio_B * flu_rate_B + population_ratio_C * flu_rate_C) / 
  (population_ratio_A + population_ratio_B + population_ratio_C) = 77 / 1500 :=
by
  sorry

end NUMINAMATH_GPT_influenza_probability_l957_95735


namespace NUMINAMATH_GPT_problem_statement_l957_95756

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) :=
  ∀ x y, x ≤ y → f x ≤ f y

noncomputable def isOddFunction (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

noncomputable def isArithmeticSeq (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem problem_statement (f : ℝ → ℝ) (a : ℕ → ℝ) (d : ℝ) (a3 : ℝ):
  isMonotonicIncreasing f →
  isOddFunction f →
  isArithmeticSeq a →
  a 3 = a3 →
  a3 > 0 →
  f (a 1) + f (a 3) + f (a 5) > 0 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_problem_statement_l957_95756


namespace NUMINAMATH_GPT_simplify_inverse_expression_l957_95798

theorem simplify_inverse_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ - y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (y * z - x * z + x * y) :=
by
  sorry

end NUMINAMATH_GPT_simplify_inverse_expression_l957_95798


namespace NUMINAMATH_GPT_perpendicular_line_eq_l957_95795

theorem perpendicular_line_eq (a b : ℝ) (ha : 2 * a - 5 * b + 3 = 0) (hpt : a = 2 ∧ b = -1) : 
    ∃ c : ℝ, c = 5 * a + 2 * b - 8 := 
sorry

end NUMINAMATH_GPT_perpendicular_line_eq_l957_95795


namespace NUMINAMATH_GPT_relationship_of_points_l957_95742

variable (y k b x : ℝ)
variable (y1 y2 : ℝ)

noncomputable def linear_func (x : ℝ) : ℝ := k * x - b

theorem relationship_of_points
  (h_pos_k : k > 0)
  (h_point1 : linear_func k b (-1) = y1)
  (h_point2 : linear_func k b 2 = y2):
  y1 < y2 := 
sorry

end NUMINAMATH_GPT_relationship_of_points_l957_95742


namespace NUMINAMATH_GPT_seating_arrangement_l957_95770

-- We define the conditions under which we will prove our theorem.
def chairs : ℕ := 7
def people : ℕ := 5

/-- Prove that there are exactly 1800 ways to seat five people in seven chairs such that the first person cannot sit in the first or last chair. -/
theorem seating_arrangement : (5 * 6 * 5 * 4 * 3) = 1800 :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangement_l957_95770


namespace NUMINAMATH_GPT_correct_log_values_l957_95752

theorem correct_log_values (a b c : ℝ)
                          (log_027 : ℝ) (log_21 : ℝ) (log_1_5 : ℝ) (log_2_8 : ℝ)
                          (log_3 : ℝ) (log_5 : ℝ) (log_6 : ℝ) (log_7 : ℝ)
                          (log_8 : ℝ) (log_9 : ℝ) (log_14 : ℝ) :
  (log_3 = 2 * a - b) →
  (log_5 = a + c) →
  (log_6 = 1 + a - b - c) →
  (log_7 = 2 * (b + c)) →
  (log_9 = 4 * a - 2 * b) →
  (log_1_5 = 3 * a - b + c) →
  (log_14 = 1 - c + 2 * b) →
  (log_1_5 = 3 * a - b + c - 1) ∧ (log_7 = 2 * b + c) := sorry

end NUMINAMATH_GPT_correct_log_values_l957_95752


namespace NUMINAMATH_GPT_scientific_notation_of_153000_l957_95793

theorem scientific_notation_of_153000 :
  ∃ (a : ℝ) (n : ℤ), 153000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 1.53 ∧ n = 5 := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_153000_l957_95793


namespace NUMINAMATH_GPT_polynomial_m_n_values_l957_95711

theorem polynomial_m_n_values :
  ∀ (m n : ℝ), ((x - 1) * (x + m) = x^2 - n * x - 6) → (m = 6 ∧ n = -5) := 
by
  intros m n h
  sorry

end NUMINAMATH_GPT_polynomial_m_n_values_l957_95711


namespace NUMINAMATH_GPT_find_B_plus_C_l957_95771

-- Define the arithmetic translations for base 8 numbers
def base8_to_dec (a b c : ℕ) : ℕ := 8^2 * a + 8 * b + c

def condition1 (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 1 ≤ A ∧ A ≤ 7 ∧ 1 ≤ B ∧ B ≤ 7 ∧ 1 ≤ C ∧ C ≤ 7

-- Define the main condition in the problem
def condition2 (A B C : ℕ) : Prop :=
  base8_to_dec A B C + base8_to_dec B C A + base8_to_dec C A B = 8^3 * A + 8^2 * A + 8 * A

-- The main statement to be proven
theorem find_B_plus_C (A B C : ℕ) (h1 : condition1 A B C) (h2 : condition2 A B C) : B + C = 7 :=
sorry

end NUMINAMATH_GPT_find_B_plus_C_l957_95771


namespace NUMINAMATH_GPT_y1_lt_y2_of_linear_graph_l957_95754

/-- In the plane rectangular coordinate system xOy, if points A(2, y1) and B(5, y2) 
    lie on the graph of a linear function y = x + b (where b is a constant), then y1 < y2. -/
theorem y1_lt_y2_of_linear_graph (y1 y2 b : ℝ) (hA : y1 = 2 + b) (hB : y2 = 5 + b) : y1 < y2 :=
by
  sorry

end NUMINAMATH_GPT_y1_lt_y2_of_linear_graph_l957_95754


namespace NUMINAMATH_GPT_garden_roller_length_l957_95748

noncomputable def length_of_garden_roller (d : ℝ) (A : ℝ) (revolutions : ℕ) (π : ℝ) : ℝ :=
  let r := d / 2
  let area_in_one_revolution := A / revolutions
  let L := area_in_one_revolution / (2 * π * r)
  L

theorem garden_roller_length :
  length_of_garden_roller 1.2 37.714285714285715 5 (22 / 7) = 2 := by
  sorry

end NUMINAMATH_GPT_garden_roller_length_l957_95748


namespace NUMINAMATH_GPT_find_absolute_difference_l957_95739

def condition_avg_sum (m n : ℝ) : Prop :=
  m + n + 5 + 6 + 4 = 25

def condition_variance (m n : ℝ) : Prop :=
  (m - 5) ^ 2 + (n - 5) ^ 2 = 8

theorem find_absolute_difference (m n : ℝ) (h1 : condition_avg_sum m n) (h2 : condition_variance m n) : |m - n| = 4 :=
sorry

end NUMINAMATH_GPT_find_absolute_difference_l957_95739


namespace NUMINAMATH_GPT_fraction_exponentiation_and_multiplication_l957_95788

theorem fraction_exponentiation_and_multiplication :
  ( (2 : ℚ) / 3 ) ^ 3 * (1 / 4) = 2 / 27 :=
by
  sorry

end NUMINAMATH_GPT_fraction_exponentiation_and_multiplication_l957_95788


namespace NUMINAMATH_GPT_route_one_speed_is_50_l957_95799

noncomputable def speed_route_one (x : ℝ) : Prop :=
  let time_route_one := 75 / x
  let time_route_two := 90 / (1.8 * x)
  time_route_one = time_route_two + 1/2

theorem route_one_speed_is_50 :
  ∃ x : ℝ, speed_route_one x ∧ x = 50 :=
by
  sorry

end NUMINAMATH_GPT_route_one_speed_is_50_l957_95799


namespace NUMINAMATH_GPT_geometric_mean_of_4_and_9_l957_95783

theorem geometric_mean_of_4_and_9 :
  ∃ b : ℝ, (4 * 9 = b^2) ∧ (b = 6 ∨ b = -6) :=
by
  sorry

end NUMINAMATH_GPT_geometric_mean_of_4_and_9_l957_95783


namespace NUMINAMATH_GPT_sum_of_fractions_equals_l957_95732

theorem sum_of_fractions_equals :
  (1 / 15 + 2 / 25 + 3 / 35 + 4 / 45 : ℚ) = 0.32127 :=
  sorry

end NUMINAMATH_GPT_sum_of_fractions_equals_l957_95732


namespace NUMINAMATH_GPT_min_f_l957_95705

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x then (x + 1) * Real.log x
else 2 * x + 3

noncomputable def f' (x : ℝ) : ℝ :=
if 0 < x then Real.log x + (x + 1) / x
else 2

theorem min_f'_for_x_pos : ∃ (c : ℝ), c = 2 ∧ ∀ x > 0, f' x ≥ c := 
  sorry

end NUMINAMATH_GPT_min_f_l957_95705


namespace NUMINAMATH_GPT_negation_of_at_most_one_obtuse_l957_95700

-- Defining a predicate to express the concept of an obtuse angle
def is_obtuse (θ : ℝ) : Prop := θ > 90

-- Defining a triangle with three interior angles α, β, and γ
structure Triangle :=
  (α β γ : ℝ)
  (sum_angles : α + β + γ = 180)

-- Defining the condition that "At most, only one interior angle of a triangle is obtuse"
def at_most_one_obtuse (T : Triangle) : Prop :=
  (is_obtuse T.α ∧ ¬ is_obtuse T.β ∧ ¬ is_obtuse T.γ) ∨
  (¬ is_obtuse T.α ∧ is_obtuse T.β ∧ ¬ is_obtuse T.γ) ∨
  (¬ is_obtuse T.α ∧ ¬ is_obtuse T.β ∧ is_obtuse T.γ) ∨
  (¬ is_obtuse T.α ∧ ¬ is_obtuse T.β ∧ ¬ is_obtuse T.γ)

-- The theorem we want to prove: Negation of "At most one obtuse angle" is "At least two obtuse angles"
theorem negation_of_at_most_one_obtuse (T : Triangle) :
  ¬ at_most_one_obtuse T ↔ (is_obtuse T.α ∧ is_obtuse T.β) ∨ (is_obtuse T.α ∧ is_obtuse T.γ) ∨ (is_obtuse T.β ∧ is_obtuse T.γ) := by
  sorry

end NUMINAMATH_GPT_negation_of_at_most_one_obtuse_l957_95700


namespace NUMINAMATH_GPT_problem_statement_l957_95744

def diamond (a b : ℚ) : ℚ := a - (1 / b)

theorem problem_statement : ((diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4))) = -29 / 132 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l957_95744


namespace NUMINAMATH_GPT_actual_average_height_l957_95720

theorem actual_average_height (average_height : ℝ) (num_students : ℕ)
  (incorrect_heights actual_heights : Fin 3 → ℝ)
  (h_avg : average_height = 165)
  (h_num : num_students = 50)
  (h_incorrect : incorrect_heights 0 = 150 ∧ incorrect_heights 1 = 175 ∧ incorrect_heights 2 = 190)
  (h_actual : actual_heights 0 = 135 ∧ actual_heights 1 = 170 ∧ actual_heights 2 = 185) :
  (average_height * num_students 
   - (incorrect_heights 0 + incorrect_heights 1 + incorrect_heights 2) 
   + (actual_heights 0 + actual_heights 1 + actual_heights 2))
   / num_students = 164.5 :=
by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_actual_average_height_l957_95720


namespace NUMINAMATH_GPT_smallest_k_l957_95777

theorem smallest_k (k : ℕ) 
  (h1 : 201 % 24 = 9 % 24) 
  (h2 : (201 + k) % (24 + k) = (9 + k) % (24 + k)) : 
  k = 8 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_k_l957_95777


namespace NUMINAMATH_GPT_arithmetic_sequence_property_l957_95766

-- Define the arithmetic sequence {an}
variable {α : Type*} [LinearOrderedField α]

def is_arith_seq (a : ℕ → α) := ∃ (d : α), ∀ (n : ℕ), a (n+1) = a n + d

-- Define the condition
def given_condition (a : ℕ → α) : Prop := a 5 / a 3 = 5 / 9

-- Main theorem statement
theorem arithmetic_sequence_property (a : ℕ → α) (h : is_arith_seq a) 
  (h_condition : given_condition a) : 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_property_l957_95766


namespace NUMINAMATH_GPT_max_value_f_zero_points_range_k_l957_95704

noncomputable def f (x k : ℝ) : ℝ := 3 * x^2 + 2 * (k - 1) * x + (k + 5)

theorem max_value_f (k : ℝ) (h : k < -7/2 ∨ k ≥ -7/2) :
  ∃ max_val : ℝ, max_val = if k < -7/2 then k + 5 else 7 * k + 26 :=
sorry

theorem zero_points_range_k :
  ∀ k : ℝ, (f 0 k) * (f 3 k) ≤ 0 ↔ (-5 ≤ k ∧ k ≤ -2) :=
sorry

end NUMINAMATH_GPT_max_value_f_zero_points_range_k_l957_95704


namespace NUMINAMATH_GPT_fermats_little_theorem_l957_95768

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (ha : ¬ p ∣ a) :
  a^(p-1) ≡ 1 [MOD p] :=
sorry

end NUMINAMATH_GPT_fermats_little_theorem_l957_95768


namespace NUMINAMATH_GPT_guilty_prob_l957_95751

-- Defining suspects
inductive Suspect
| A
| B
| C

open Suspect

-- Constants for the problem
def looks_alike (x y : Suspect) : Prop :=
(x = A ∧ y = B) ∨ (x = B ∧ y = A)

def timid (x : Suspect) : Prop :=
x = A ∨ x = B

def bold (x : Suspect) : Prop :=
x = C

def alibi_dover (x : Suspect) : Prop :=
x = A ∨ x = B

def needs_accomplice (x : Suspect) : Prop :=
timid x

def works_alone (x : Suspect) : Prop :=
bold x

def in_bar_during_robbery (x : Suspect) : Prop :=
x = A ∨ x = B

-- Theorem to be proved
theorem guilty_prob :
  ∃ x : Suspect, (x = B) ∧ ∀ y : Suspect, y ≠ B → 
    ((y = A ∧ timid y ∧ needs_accomplice y ∧ in_bar_during_robbery y) ∨
    (y = C ∧ bold y ∧ works_alone y)) :=
by
  sorry

end NUMINAMATH_GPT_guilty_prob_l957_95751


namespace NUMINAMATH_GPT_minimum_occupied_seats_l957_95776

theorem minimum_occupied_seats (total_seats : ℕ) (min_empty_seats : ℕ) (occupied_seats : ℕ)
  (h1 : total_seats = 150)
  (h2 : min_empty_seats = 2)
  (h3 : occupied_seats = 2 * (total_seats / (occupied_seats + min_empty_seats + min_empty_seats)))
  : occupied_seats = 74 := by
  sorry

end NUMINAMATH_GPT_minimum_occupied_seats_l957_95776


namespace NUMINAMATH_GPT_danny_marks_in_math_l957_95761

theorem danny_marks_in_math
  (english_marks : ℕ := 76)
  (physics_marks : ℕ := 82)
  (chemistry_marks : ℕ := 67)
  (biology_marks : ℕ := 75)
  (average_marks : ℕ := 73)
  (num_subjects : ℕ := 5) :
  ∃ (math_marks : ℕ), math_marks = 65 :=
by
  let total_marks := average_marks * num_subjects
  let other_subjects_marks := english_marks + physics_marks + chemistry_marks + biology_marks
  have math_marks := total_marks - other_subjects_marks
  use math_marks
  sorry

end NUMINAMATH_GPT_danny_marks_in_math_l957_95761


namespace NUMINAMATH_GPT_number_of_triangles_l957_95749

/-!
# Problem Statement
Given a square with 20 interior points connected such that the lines do not intersect and divide the square into triangles,
prove that the number of triangles formed is 42.
-/

theorem number_of_triangles (V E F : ℕ) (hV : V = 24) (hE : E = (3 * F + 1) / 2) (hF : V - E + F = 2) :
  (F - 1) = 42 :=
by
  sorry

end NUMINAMATH_GPT_number_of_triangles_l957_95749


namespace NUMINAMATH_GPT_inequality_solutions_l957_95718

theorem inequality_solutions (n : ℕ) (h : n > 0) : n^3 - n < n! ↔ (n = 1 ∨ n ≥ 6) := 
by
  sorry

end NUMINAMATH_GPT_inequality_solutions_l957_95718


namespace NUMINAMATH_GPT_find_k_value_l957_95710

theorem find_k_value (k : ℝ) (h₁ : ∀ x, k * x^2 - 5 * x - 12 = 0 → (x = 3 ∨ x = -4 / 3)) : k = 3 :=
sorry

end NUMINAMATH_GPT_find_k_value_l957_95710


namespace NUMINAMATH_GPT_maximum_xyz_l957_95709

theorem maximum_xyz {x y z : ℝ} (hx: 0 < x) (hy: 0 < y) (hz: 0 < z) 
  (h : (x * y) + z = (x + z) * (y + z)) : xyz ≤ (1 / 27) :=
by
  sorry

end NUMINAMATH_GPT_maximum_xyz_l957_95709


namespace NUMINAMATH_GPT_money_left_l957_95781

noncomputable def initial_amount : ℝ := 10.10
noncomputable def spent_on_sweets : ℝ := 3.25
noncomputable def amount_per_friend : ℝ := 2.20
noncomputable def remaining_amount : ℝ := initial_amount - spent_on_sweets - 2 * amount_per_friend

theorem money_left : remaining_amount = 2.45 :=
by
  sorry

end NUMINAMATH_GPT_money_left_l957_95781


namespace NUMINAMATH_GPT_cricket_bat_selling_price_l957_95769

theorem cricket_bat_selling_price (profit : ℝ) (profit_percentage : ℝ) (C : ℝ) (selling_price : ℝ) 
  (h1 : profit = 150) 
  (h2 : profit_percentage = 20) 
  (h3 : profit = (profit_percentage / 100) * C) 
  (h4 : selling_price = C + profit) : 
  selling_price = 900 := 
sorry

end NUMINAMATH_GPT_cricket_bat_selling_price_l957_95769


namespace NUMINAMATH_GPT_triangle_base_value_l957_95778

variable (L R B : ℕ)

theorem triangle_base_value
    (h1 : L = 12)
    (h2 : R = L + 2)
    (h3 : L + R + B = 50) :
    B = 24 := 
sorry

end NUMINAMATH_GPT_triangle_base_value_l957_95778


namespace NUMINAMATH_GPT_range_of_a_l957_95764

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * Real.sin x - (Real.cos x)^2 ≤ 3) : -3 ≤ a ∧ a ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l957_95764


namespace NUMINAMATH_GPT_tessa_still_owes_greg_l957_95723

def initial_debt : ℝ := 40
def first_repayment : ℝ := 0.25 * initial_debt
def debt_after_first_repayment : ℝ := initial_debt - first_repayment
def second_borrowing : ℝ := 25
def debt_after_second_borrowing : ℝ := debt_after_first_repayment + second_borrowing
def second_repayment : ℝ := 0.5 * debt_after_second_borrowing
def debt_after_second_repayment : ℝ := debt_after_second_borrowing - second_repayment
def third_borrowing : ℝ := 30
def debt_after_third_borrowing : ℝ := debt_after_second_repayment + third_borrowing
def third_repayment : ℝ := 0.1 * debt_after_third_borrowing
def final_debt : ℝ := debt_after_third_borrowing - third_repayment

theorem tessa_still_owes_greg : final_debt = 51.75 := by
  sorry

end NUMINAMATH_GPT_tessa_still_owes_greg_l957_95723


namespace NUMINAMATH_GPT_third_side_length_l957_95745

theorem third_side_length (a b x : ℝ) (h₁ : a = 3) (h₂ : b = 8) (h₃ : 5 < x) (h₄ : x < 11) : x = 6 :=
sorry

end NUMINAMATH_GPT_third_side_length_l957_95745


namespace NUMINAMATH_GPT_bob_time_improvement_l957_95767

def time_improvement_percent (bob_time sister_time improvement_time : ℕ) : ℕ :=
  ((improvement_time * 100) / bob_time)

theorem bob_time_improvement : 
  ∀ (bob_time sister_time : ℕ), bob_time = 640 → sister_time = 608 → 
  time_improvement_percent bob_time sister_time (bob_time - sister_time) = 5 :=
by
  intros bob_time sister_time h_bob h_sister
  rw [h_bob, h_sister]
  sorry

end NUMINAMATH_GPT_bob_time_improvement_l957_95767


namespace NUMINAMATH_GPT_correct_choice_is_C_l957_95727

def first_quadrant_positive_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

def right_angle_is_axial (θ : ℝ) : Prop :=
  θ = 90

def obtuse_angle_second_quadrant (θ : ℝ) : Prop :=
  90 < θ ∧ θ < 180

def terminal_side_initial_side_same (θ : ℝ) : Prop :=
  θ = 0 ∨ θ = 360

theorem correct_choice_is_C : obtuse_angle_second_quadrant 120 :=
by
  sorry

end NUMINAMATH_GPT_correct_choice_is_C_l957_95727


namespace NUMINAMATH_GPT_clock_rings_eight_times_in_a_day_l957_95772

theorem clock_rings_eight_times_in_a_day : 
  ∀ t : ℕ, t % 3 = 1 → 0 ≤ t ∧ t < 24 → ∃ n : ℕ, n = 8 := 
by 
  sorry

end NUMINAMATH_GPT_clock_rings_eight_times_in_a_day_l957_95772


namespace NUMINAMATH_GPT_number_of_remaining_grandchildren_l957_95747

-- Defining the given values and conditions
def total_amount : ℕ := 124600
def half_amount : ℕ := total_amount / 2
def amount_per_remaining_grandchild : ℕ := 6230

-- Defining the goal to prove the number of remaining grandchildren
theorem number_of_remaining_grandchildren : (half_amount / amount_per_remaining_grandchild) = 10 := by
  sorry

end NUMINAMATH_GPT_number_of_remaining_grandchildren_l957_95747


namespace NUMINAMATH_GPT_solve_quadratic_l957_95765

theorem solve_quadratic (x : ℝ) (h₁ : x > 0) (h₂ : 3 * x^2 - 7 * x - 6 = 0) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l957_95765


namespace NUMINAMATH_GPT_people_who_cannot_do_either_l957_95750

def people_total : ℕ := 120
def can_dance : ℕ := 88
def can_write_calligraphy : ℕ := 32
def can_do_both : ℕ := 18

theorem people_who_cannot_do_either : 
  people_total - (can_dance + can_write_calligraphy - can_do_both) = 18 := 
by
  sorry

end NUMINAMATH_GPT_people_who_cannot_do_either_l957_95750


namespace NUMINAMATH_GPT_number_of_boys_l957_95762

theorem number_of_boys {total_students : ℕ} (h1 : total_students = 49)
  (ratio_girls_boys : ℕ → ℕ → Prop)
  (h2 : ratio_girls_boys 4 3) :
  ∃ boys : ℕ, boys = 21 := by
  sorry

end NUMINAMATH_GPT_number_of_boys_l957_95762


namespace NUMINAMATH_GPT_coeff_x2_expansion_sqrt_x_plus_1_over_x_pow_10_eq_45_l957_95731

theorem coeff_x2_expansion_sqrt_x_plus_1_over_x_pow_10_eq_45 :
  let general_term (r : ℕ) := (Nat.choose 10 r) * (x^(10 - 3 * r)/2)
  ∃ r : ℕ, (general_term r) = 2 ∧ (Nat.choose 10 r) = 45 :=
by
  sorry

end NUMINAMATH_GPT_coeff_x2_expansion_sqrt_x_plus_1_over_x_pow_10_eq_45_l957_95731


namespace NUMINAMATH_GPT_k_value_five_l957_95706

theorem k_value_five (a b k : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (a^2 + b^2) / (a * b - 1) = k) : k = 5 := 
sorry

end NUMINAMATH_GPT_k_value_five_l957_95706


namespace NUMINAMATH_GPT_equivalent_expression_l957_95787

theorem equivalent_expression (a : ℝ) (h1 : a ≠ -2) (h2 : a ≠ -1) :
  ( (a^2 + a - 2) / (a^2 + 3*a + 2) * 5 * (a + 1)^2 = 5*a^2 - 5 ) :=
by {
  sorry
}

end NUMINAMATH_GPT_equivalent_expression_l957_95787


namespace NUMINAMATH_GPT_first_diamond_second_spade_prob_l957_95759

/--
Given a standard deck of 52 cards, there are 13 cards of each suit.
What is the probability that the first card dealt is a diamond (♦) 
and the second card dealt is a spade (♠)?
-/
theorem first_diamond_second_spade_prob : 
  let total_cards := 52
  let diamonds := 13
  let spades := 13
  let first_diamond_prob := diamonds / total_cards
  let second_spade_prob_after_diamond := spades / (total_cards - 1)
  let combined_prob := first_diamond_prob * second_spade_prob_after_diamond
  combined_prob = 13 / 204 := 
by
  sorry

end NUMINAMATH_GPT_first_diamond_second_spade_prob_l957_95759


namespace NUMINAMATH_GPT_number_of_solutions_l957_95773

open Nat

-- Definitions arising from the conditions
def is_solution (x y : ℕ) : Prop := 3 * x + 5 * y = 501

-- Statement of the problem
theorem number_of_solutions :
  (∃ k : ℕ, k ≥ 0 ∧ k < 33 ∧ ∀ (x y : ℕ), x = 5 * k + 2 ∧ y = 99 - 3 * k → is_solution x y) :=
  sorry

end NUMINAMATH_GPT_number_of_solutions_l957_95773


namespace NUMINAMATH_GPT_minimum_value_expression_l957_95726

theorem minimum_value_expression (F M N : ℝ × ℝ) (x y : ℝ) (a : ℝ) (k : ℝ) :
  (y ^ 2 = 16 * x ∧ F = (4, 0) ∧ l = (k * (x - 4), y) ∧ (M = (x₁, y₁) ∧ N = (x₂, y₂)) ∧
  0 ≤ x₁ ∧ y₁ ^ 2 = 16 * x₁ ∧ 0 ≤ x₂ ∧ y₂ ^ 2 = 16 * x₂) →
  (abs (dist F N) / 9 - 4 / abs (dist F M) ≥ 1 / 3) :=
sorry -- proof will be provided

end NUMINAMATH_GPT_minimum_value_expression_l957_95726


namespace NUMINAMATH_GPT_train_length_l957_95760

theorem train_length :
  ∃ L : ℝ, 
    (∀ V : ℝ, V = L / 24 ∧ V = (L + 650) / 89) → 
    L = 240 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l957_95760


namespace NUMINAMATH_GPT_original_class_strength_l957_95738

variable (x : ℕ)

/-- The average age of an adult class is 40 years.
  18 new students with an average age of 32 years join the class, 
  therefore decreasing the average by 4 years.
  Find the original strength of the class.
-/
theorem original_class_strength (h1 : 40 * x + 18 * 32 = (x + 18) * 36) : x = 18 := 
by sorry

end NUMINAMATH_GPT_original_class_strength_l957_95738


namespace NUMINAMATH_GPT_xiaoming_xiaoqiang_common_visit_l957_95708

-- Define the initial visit dates and subsequent visit intervals
def xiaoming_initial_visit : ℕ := 3 -- The first Wednesday of January
def xiaoming_interval : ℕ := 4

def xiaoqiang_initial_visit : ℕ := 4 -- The first Thursday of January
def xiaoqiang_interval : ℕ := 3

-- Prove that the only common visit date is January 7
theorem xiaoming_xiaoqiang_common_visit : 
  ∃! d, (d < 32) ∧ ∃ n m, d = xiaoming_initial_visit + n * xiaoming_interval ∧ d = xiaoqiang_initial_visit + m * xiaoqiang_interval :=
  sorry

end NUMINAMATH_GPT_xiaoming_xiaoqiang_common_visit_l957_95708


namespace NUMINAMATH_GPT_ratio_H_G_l957_95707

theorem ratio_H_G (G H : ℤ) (h : ∀ x : ℝ, x ≠ -3 → x ≠ 0 → x ≠ 5 → 
    (G / (x + 3) + H / (x * (x - 5)) = (x^2 - 3 * x + 8) / (x^3 + x^2 - 15 * x))) :
    H / G = 64 :=
sorry

end NUMINAMATH_GPT_ratio_H_G_l957_95707


namespace NUMINAMATH_GPT_max_height_reached_l957_95758

def h (t : ℝ) : ℝ := -20 * t ^ 2 + 120 * t + 36

theorem max_height_reached :
  ∃ t : ℝ, h t = 216 ∧ t = 3 :=
sorry

end NUMINAMATH_GPT_max_height_reached_l957_95758


namespace NUMINAMATH_GPT_two_lines_perpendicular_to_same_plane_are_parallel_l957_95755

variables {Plane Line : Type} 
variables (perp : Line → Plane → Prop) (parallel : Line → Line → Prop)

theorem two_lines_perpendicular_to_same_plane_are_parallel
  (a b : Line) (α : Plane) (ha : perp a α) (hb : perp b α) : parallel a b :=
sorry

end NUMINAMATH_GPT_two_lines_perpendicular_to_same_plane_are_parallel_l957_95755


namespace NUMINAMATH_GPT_inconsistent_equation_system_l957_95753

variables {a x c : ℝ}

theorem inconsistent_equation_system (h1 : (a + x) / 2 = 110) (h2 : (x + c) / 2 = 170) (h3 : a - c = 120) : false :=
by
  sorry

end NUMINAMATH_GPT_inconsistent_equation_system_l957_95753


namespace NUMINAMATH_GPT_combined_area_of_three_walls_l957_95740

theorem combined_area_of_three_walls (A : ℝ) :
  (A - 2 * 30 - 3 * 45 = 180) → (A = 375) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_combined_area_of_three_walls_l957_95740


namespace NUMINAMATH_GPT_sammy_offer_l957_95734

-- Declaring the given constants and assumptions
def peggy_records : ℕ := 200
def bryan_interested_records : ℕ := 100
def bryan_uninterested_records : ℕ := 100
def bryan_interested_offer : ℕ := 6
def bryan_uninterested_offer : ℕ := 1
def sammy_offer_diff : ℕ := 100

-- The problem to be proved
theorem sammy_offer:
    ∃ S : ℝ, 
    (200 * S) - 
    (bryan_interested_records * bryan_interested_offer +
    bryan_uninterested_records * bryan_uninterested_offer) = sammy_offer_diff → 
    S = 4 :=
sorry

end NUMINAMATH_GPT_sammy_offer_l957_95734


namespace NUMINAMATH_GPT_cyclic_sum_ineq_l957_95786

theorem cyclic_sum_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (a^2 + a * b + b^2) + b^3 / (b^2 + b * c + c^2) + c^3 / (c^2 + c * a + a^2)) 
  ≥ (1 / 3) * (a + b + c) :=
by
  sorry

end NUMINAMATH_GPT_cyclic_sum_ineq_l957_95786


namespace NUMINAMATH_GPT_num_blue_balls_l957_95702

theorem num_blue_balls (total_balls blue_balls : ℕ) 
  (prob_all_blue : ℚ)
  (h_total : total_balls = 12)
  (h_prob : prob_all_blue = 1 / 55)
  (h_prob_eq : (blue_balls / 12) * ((blue_balls - 1) / 11) * ((blue_balls - 2) / 10) = prob_all_blue) :
  blue_balls = 4 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_num_blue_balls_l957_95702


namespace NUMINAMATH_GPT_slope_of_line_through_focus_of_parabola_l957_95733

theorem slope_of_line_through_focus_of_parabola
  (C : (x y : ℝ) → y^2 = 4 * x)
  (F : (ℝ × ℝ) := (1, 0))
  (A B : (ℝ × ℝ))
  (l : ℝ → ℝ)
  (intersects : (x : ℝ) → (l x) ^ 2 = 4 * x)
  (passes_through_focus : l 1 = 0)
  (distance_condition : ∀ (d1 d2 : ℝ), d1 = 4 * d2 → dist F A = d1 ∧ dist F B = d2) :
  ∃ k : ℝ, (∀ (x : ℝ), l x = k * (x - 1)) ∧ (k = 4 / 3 ∨ k = -4 / 3) :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_through_focus_of_parabola_l957_95733


namespace NUMINAMATH_GPT_parts_processed_per_hour_l957_95712

theorem parts_processed_per_hour (x : ℕ) (y : ℕ) (h1 : y = x + 10) (h2 : 150 / y = 120 / x) :
  x = 40 ∧ y = 50 :=
by {
  sorry
}

end NUMINAMATH_GPT_parts_processed_per_hour_l957_95712


namespace NUMINAMATH_GPT_sum_arith_seq_l957_95728

theorem sum_arith_seq (d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ)
    (h₁ : ∀ n, S n = n * a 1 + (n * (n - 1)) * d / 2)
    (h₂ : S 10 = S 20)
    (h₃ : d > 0) :
    a 10 + a 22 > 0 := 
sorry

end NUMINAMATH_GPT_sum_arith_seq_l957_95728


namespace NUMINAMATH_GPT_geom_seq_sum_3000_l957_95741

noncomputable
def sum_geom_seq (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then a * n
  else a * (1 - r ^ n) / (1 - r)

theorem geom_seq_sum_3000 (a r : ℝ) (h1: sum_geom_seq a r 1000 = 300) (h2: sum_geom_seq a r 2000 = 570) :
  sum_geom_seq a r 3000 = 813 :=
sorry

end NUMINAMATH_GPT_geom_seq_sum_3000_l957_95741


namespace NUMINAMATH_GPT_sale_in_second_month_l957_95701

theorem sale_in_second_month
  (sale1 sale3 sale4 sale5 sale6 : ℕ)
  (average_sale : ℕ)
  (total_months : ℕ)
  (h_sale1 : sale1 = 5420)
  (h_sale3 : sale3 = 6200)
  (h_sale4 : sale4 = 6350)
  (h_sale5 : sale5 = 6500)
  (h_sale6 : sale6 = 6470)
  (h_average_sale : average_sale = 6100)
  (h_total_months : total_months = 6) :
  ∃ sale2 : ℕ, sale2 = 5660 := 
by
  sorry

end NUMINAMATH_GPT_sale_in_second_month_l957_95701


namespace NUMINAMATH_GPT_tanner_savings_in_october_l957_95796

theorem tanner_savings_in_october 
    (sept_savings : ℕ := 17) 
    (nov_savings : ℕ := 25)
    (spent : ℕ := 49) 
    (left : ℕ := 41) 
    (X : ℕ) 
    (h : sept_savings + X + nov_savings - spent = left) 
    : X = 48 :=
by
  sorry

end NUMINAMATH_GPT_tanner_savings_in_october_l957_95796


namespace NUMINAMATH_GPT_primes_p_p2_p4_l957_95713

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem primes_p_p2_p4 (p : ℕ) (hp : is_prime p) (hp2 : is_prime (p + 2)) (hp4 : is_prime (p + 4)) :
  p = 3 :=
sorry

end NUMINAMATH_GPT_primes_p_p2_p4_l957_95713


namespace NUMINAMATH_GPT_largest_possible_cupcakes_without_any_ingredients_is_zero_l957_95784

-- Definitions of properties of the cupcakes
def total_cupcakes : ℕ := 60
def blueberries (n : ℕ) : Prop := n = total_cupcakes / 3
def sprinkles (n : ℕ) : Prop := n = total_cupcakes / 4
def frosting (n : ℕ) : Prop := n = total_cupcakes / 2
def pecans (n : ℕ) : Prop := n = total_cupcakes / 5

-- Theorem statement
theorem largest_possible_cupcakes_without_any_ingredients_is_zero :
  ∃ n, blueberries n ∧ sprinkles n ∧ frosting n ∧ pecans n → n = 0 := 
sorry

end NUMINAMATH_GPT_largest_possible_cupcakes_without_any_ingredients_is_zero_l957_95784


namespace NUMINAMATH_GPT_difference_in_perimeter_is_50_cm_l957_95715

-- Define the lengths of the four ribbons
def ribbon_lengths (x : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  (x, x + 25, x + 50, x + 75)

-- Define the perimeter of the first shape
def perimeter_first_shape (x : ℕ) : ℕ :=
  2 * x + 230

-- Define the perimeter of the second shape
def perimeter_second_shape (x : ℕ) : ℕ :=
  2 * x + 280

-- Define the main theorem that the difference in perimeter is 50 cm
theorem difference_in_perimeter_is_50_cm (x : ℕ) :
  perimeter_second_shape x - perimeter_first_shape x = 50 := by
  sorry

end NUMINAMATH_GPT_difference_in_perimeter_is_50_cm_l957_95715


namespace NUMINAMATH_GPT_greatest_possible_perimeter_l957_95757

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end NUMINAMATH_GPT_greatest_possible_perimeter_l957_95757


namespace NUMINAMATH_GPT_max_closable_companies_l957_95785

def number_of_planets : ℕ := 10 ^ 2015
def number_of_companies : ℕ := 2015

theorem max_closable_companies (k : ℕ) : k = 1007 :=
sorry

end NUMINAMATH_GPT_max_closable_companies_l957_95785


namespace NUMINAMATH_GPT_moving_circle_trajectory_is_ellipse_l957_95716

noncomputable def trajectory_of_center (x y : ℝ) : Prop :=
  let ellipse_eq := x^2 / 4 + y^2 / 3 = 1 
  ellipse_eq ∧ x ≠ -2

theorem moving_circle_trajectory_is_ellipse
  (M_1 M_2 center : ℝ × ℝ)
  (r1 r2 R : ℝ)
  (h1 : M_1 = (-1, 0))
  (h2 : M_2 = (1, 0))
  (h3 : r1 = 1)
  (h4 : r2 = 3)
  (h5 : (center.1 + 1)^2 + center.2^2 = (1 + R)^2)
  (h6 : (center.1 - 1)^2 + center.2^2 = (3 - R)^2) :
  trajectory_of_center center.1 center.2 :=
by sorry

end NUMINAMATH_GPT_moving_circle_trajectory_is_ellipse_l957_95716


namespace NUMINAMATH_GPT_initial_files_count_l957_95763

theorem initial_files_count (deleted_files folders files_per_folder total_files initial_files : ℕ)
    (h1 : deleted_files = 21)
    (h2 : folders = 9)
    (h3 : files_per_folder = 8)
    (h4 : total_files = folders * files_per_folder)
    (h5 : initial_files = total_files + deleted_files) :
    initial_files = 93 :=
by
  sorry

end NUMINAMATH_GPT_initial_files_count_l957_95763


namespace NUMINAMATH_GPT_expected_value_coin_flip_l957_95730

def probability_heads : ℚ := 2 / 3
def probability_tails : ℚ := 1 / 3
def gain_heads : ℤ := 5
def loss_tails : ℤ := -9

theorem expected_value_coin_flip : (2 / 3 : ℚ) * 5 + (1 / 3 : ℚ) * (-9) = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_expected_value_coin_flip_l957_95730
