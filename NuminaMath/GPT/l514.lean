import Mathlib

namespace NUMINAMATH_GPT_system_solutions_a_l514_51439

theorem system_solutions_a (x y z : ℝ) :
  (2 * x = (y + z) ^ 2) ∧ (2 * y = (z + x) ^ 2) ∧ (2 * z = (x + y) ^ 2) ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end NUMINAMATH_GPT_system_solutions_a_l514_51439


namespace NUMINAMATH_GPT_inequality_proof_l514_51414

theorem inequality_proof (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) :
  (1 - 2 * x) / Real.sqrt (x * (1 - x)) + 
  (1 - 2 * y) / Real.sqrt (y * (1 - y)) + 
  (1 - 2 * z) / Real.sqrt (z * (1 - z)) ≥ 
  Real.sqrt (x / (1 - x)) + 
  Real.sqrt (y / (1 - y)) + 
  Real.sqrt (z / (1 - z)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l514_51414


namespace NUMINAMATH_GPT_probability_is_two_thirds_l514_51424

-- Define the general framework and conditions
def total_students : ℕ := 4
def students_from_first_grade : ℕ := 2
def students_from_second_grade : ℕ := 2

-- Define the combinations for selecting 2 students out of 4
def total_ways_to_select_2_students : ℕ := Nat.choose total_students 2

-- Define the combinations for selecting 1 student from each grade
def ways_to_select_1_from_first : ℕ := Nat.choose students_from_first_grade 1
def ways_to_select_1_from_second : ℕ := Nat.choose students_from_second_grade 1
def favorable_ways : ℕ := ways_to_select_1_from_first * ways_to_select_1_from_second

-- The target probability calculation
noncomputable def probability_of_different_grades : ℚ :=
  favorable_ways / total_ways_to_select_2_students

-- The statement and proof requirement (proof is deferred with sorry)
theorem probability_is_two_thirds :
  probability_of_different_grades = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_probability_is_two_thirds_l514_51424


namespace NUMINAMATH_GPT_smallest_integer_ends_in_3_divisible_by_11_correct_l514_51416

def ends_in_3 (n : ℕ) : Prop :=
  n % 10 = 3

def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def smallest_ends_in_3_divisible_by_11 : ℕ :=
  33

theorem smallest_integer_ends_in_3_divisible_by_11_correct :
  smallest_ends_in_3_divisible_by_11 = 33 ∧ ends_in_3 smallest_ends_in_3_divisible_by_11 ∧ divisible_by_11 smallest_ends_in_3_divisible_by_11 := 
by
  sorry

end NUMINAMATH_GPT_smallest_integer_ends_in_3_divisible_by_11_correct_l514_51416


namespace NUMINAMATH_GPT_inequality_solution_l514_51430

theorem inequality_solution (x : ℝ) :
  2 * (2 * x - 1) > 3 * x - 1 → x > 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l514_51430


namespace NUMINAMATH_GPT_triangle_RS_length_l514_51411

theorem triangle_RS_length (PQ QR PS QS RS : ℝ)
  (h1 : PQ = 8) (h2 : QR = 8) (h3 : PS = 10) (h4 : QS = 5) :
  RS = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_RS_length_l514_51411


namespace NUMINAMATH_GPT_cal_fraction_of_anthony_l514_51400

theorem cal_fraction_of_anthony (mabel_transactions anthony_transactions cal_transactions jade_transactions : ℕ)
  (h_mabel : mabel_transactions = 90)
  (h_anthony : anthony_transactions = mabel_transactions + mabel_transactions / 10)
  (h_jade : jade_transactions = 82)
  (h_jade_cal : jade_transactions = cal_transactions + 16) :
  (cal_transactions : ℚ) / (anthony_transactions : ℚ) = 2 / 3 :=
by
  -- The proof would be here, but it is omitted as per the requirement.
  sorry

end NUMINAMATH_GPT_cal_fraction_of_anthony_l514_51400


namespace NUMINAMATH_GPT_point_on_opposite_sides_l514_51460

theorem point_on_opposite_sides (y_0 : ℝ) :
  (2 - 2 * 3 + 5 > 0) ∧ (6 - 2 * y_0 < 0) → y_0 > 3 :=
by
  sorry

end NUMINAMATH_GPT_point_on_opposite_sides_l514_51460


namespace NUMINAMATH_GPT_total_legs_in_household_l514_51426

def number_of_legs (humans children dogs cats : ℕ) (human_legs child_legs dog_legs cat_legs : ℕ) : ℕ :=
  humans * human_legs + children * child_legs + dogs * dog_legs + cats * cat_legs

theorem total_legs_in_household : number_of_legs 2 3 2 1 2 2 4 4 = 22 :=
  by
    -- The statement ensures the total number of legs is 22, given the defined conditions.
    sorry

end NUMINAMATH_GPT_total_legs_in_household_l514_51426


namespace NUMINAMATH_GPT_inequality_one_inequality_two_l514_51435

variable (a b c : ℝ)

-- First Inequality Proof Statement
theorem inequality_one (h_pos : a > 0 ∧ b > 0 ∧ c > 0) : 
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ 3 / 2 := 
sorry

-- Second Inequality Proof Statement
theorem inequality_two (h_pos : a > 0 ∧ b > 0 ∧ c > 0) : 
  (a^3 + b^3 + c^3 + 1/a + 1/b + 1/c) ≥ 2 * (a + b + c) := 
sorry

end NUMINAMATH_GPT_inequality_one_inequality_two_l514_51435


namespace NUMINAMATH_GPT_arithmetic_mean_of_p_and_q_l514_51473

variable (p q r : ℝ)

theorem arithmetic_mean_of_p_and_q
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 22)
  (h3 : r - p = 24) :
  (p + q) / 2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_p_and_q_l514_51473


namespace NUMINAMATH_GPT_five_n_plus_3_composite_l514_51459

theorem five_n_plus_3_composite (n : ℕ)
  (h1 : ∃ k : ℤ, 2 * n + 1 = k^2)
  (h2 : ∃ m : ℤ, 3 * n + 1 = m^2) :
  ¬ Prime (5 * n + 3) :=
by
  sorry

end NUMINAMATH_GPT_five_n_plus_3_composite_l514_51459


namespace NUMINAMATH_GPT_simplify_expression_l514_51476

variable (a b : ℤ)

theorem simplify_expression : (a - b) - (3 * (a + b)) - b = a - 8 * b := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l514_51476


namespace NUMINAMATH_GPT_find_numbers_l514_51490

theorem find_numbers (a b c d : ℕ)
  (h1 : a + b + c = 21)
  (h2 : a + b + d = 28)
  (h3 : a + c + d = 29)
  (h4 : b + c + d = 30) : 
  a = 6 ∧ b = 7 ∧ c = 8 ∧ d = 15 :=
sorry

end NUMINAMATH_GPT_find_numbers_l514_51490


namespace NUMINAMATH_GPT_bobby_last_10_throws_successful_l514_51442

theorem bobby_last_10_throws_successful :
    let initial_successful := 18 -- Bobby makes 18 successful throws out of his initial 30 throws.
    let total_throws := 30 + 10 -- Bobby makes a total of 40 throws.
    let final_successful := 0.64 * total_throws -- Bobby needs to make 64% of 40 throws to achieve a 64% success rate.
    let required_successful := 26 -- Adjusted to the nearest whole number.
    -- Bobby makes 8 successful throws in his last 10 attempts.
    required_successful - initial_successful = 8 := by
  sorry

end NUMINAMATH_GPT_bobby_last_10_throws_successful_l514_51442


namespace NUMINAMATH_GPT_simplify_fraction_150_div_225_l514_51429

theorem simplify_fraction_150_div_225 :
  let a := 150
  let b := 225
  let gcd_ab := Nat.gcd a b
  let num_fact := 2 * 3 * 5^2
  let den_fact := 3^2 * 5^2
  gcd_ab = 75 →
  num_fact = a →
  den_fact = b →
  (a / gcd_ab) / (b / gcd_ab) = (2 / 3) :=
  by
    intros 
    sorry

end NUMINAMATH_GPT_simplify_fraction_150_div_225_l514_51429


namespace NUMINAMATH_GPT_sum_of_incircle_areas_l514_51404

variables {a b c : ℝ} (ABC : Triangle ℝ) (s K r : ℝ)
  (hs : s = (a + b + c) / 2)
  (hK : K = Real.sqrt (s * (s - a) * (s - b) * (s - c)))
  (hr : r = K / s)

theorem sum_of_incircle_areas :
  let larger_circle_area := π * r^2
  let smaller_circle_area := π * (r / 2)^2
  larger_circle_area + 3 * smaller_circle_area = 7 * π * r^2 / 4 :=
sorry

end NUMINAMATH_GPT_sum_of_incircle_areas_l514_51404


namespace NUMINAMATH_GPT_construct_angle_approx_l514_51450
-- Use a broader import to bring in the entirety of the necessary library

-- Define the problem 
theorem construct_angle_approx (α : ℝ) (m : ℕ) (h : ∃ l : ℕ, (l : ℝ) / 2^m * 90 ≤ α ∧ α ≤ ((l+1) : ℝ) / 2^m * 90) :
  ∃ β : ℝ, β ∈ { β | ∃ l : ℕ, β = (l : ℝ) / 2^m * 90} ∧ |α - β| ≤ 90 / 2^m :=
sorry

end NUMINAMATH_GPT_construct_angle_approx_l514_51450


namespace NUMINAMATH_GPT_bus_journey_distance_l514_51447

theorem bus_journey_distance (x : ℝ) (h1 : 0 ≤ x)
  (h2 : 0 ≤ 250 - x)
  (h3 : x / 40 + (250 - x) / 60 = 5.2) :
  x = 124 :=
sorry

end NUMINAMATH_GPT_bus_journey_distance_l514_51447


namespace NUMINAMATH_GPT_find_function_l514_51428

variable (R : Type) [LinearOrderedField R]

theorem find_function
  (f : R → R)
  (h : ∀ x y : R, f (x + y) + y ≤ f (f (f x))) :
  ∃ c : R, ∀ x : R, f x = c - x :=
sorry

end NUMINAMATH_GPT_find_function_l514_51428


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l514_51451

-- Definition of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := (x^2) / 16 - (y^2) / 9 = 1

-- Definition of the equations of the asymptotes
def asymptote_eq (x y : ℝ) : Prop := y = (3/4)*x ∨ y = -(3/4)*x

-- Theorem statement
theorem asymptotes_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_eq x y → asymptote_eq x y :=
sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l514_51451


namespace NUMINAMATH_GPT_correct_statement_2_l514_51452

-- Definitions of parallel and perpendicular relationships
variables (a b : line) (α β : plane)

-- Conditions
def parallel (x y : plane) : Prop := sorry -- definition not provided
def perpendicular (x y : plane) : Prop := sorry -- definition not provided
def line_parallel_plane (l : line) (p : plane) : Prop := sorry -- definition not provided
def line_perpendicular_plane (l : line) (p : plane) : Prop := sorry -- definition not provided
def line_perpendicular (l1 l2 : line) : Prop := sorry -- definition not provided

-- Proof of the correct statement among the choices
theorem correct_statement_2 :
  line_perpendicular a b → line_perpendicular_plane a α → line_perpendicular_plane b β → perpendicular α β :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_correct_statement_2_l514_51452


namespace NUMINAMATH_GPT_other_solution_quadratic_l514_51468

theorem other_solution_quadratic (h : (49 : ℚ) * (5 / 7)^2 - 88 * (5 / 7) + 40 = 0) : 
  ∃ x : ℚ, x ≠ 5 / 7 ∧ (49 * x^2 - 88 * x + 40 = 0) ∧ x = 8 / 7 :=
by
  sorry

end NUMINAMATH_GPT_other_solution_quadratic_l514_51468


namespace NUMINAMATH_GPT_students_in_school_at_least_225_l514_51478

-- Conditions as definitions
def students_in_band := 85
def students_in_sports := 200
def students_in_both := 60
def students_in_either := 225

-- The proof statement
theorem students_in_school_at_least_225 :
  students_in_band + students_in_sports - students_in_both = students_in_either :=
by
  -- This statement will just assert the correctness as per given information in the problem
  sorry

end NUMINAMATH_GPT_students_in_school_at_least_225_l514_51478


namespace NUMINAMATH_GPT_min_value_expression_l514_51433

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a^x + b

theorem min_value_expression (a b : ℝ) (h1 : b > 0) (h2 : f a b 1 = 3) :
  ∃ x, x = (4 / (a - 1) + 1 / b) ∧ x = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l514_51433


namespace NUMINAMATH_GPT_carpet_shaded_area_is_correct_l514_51413

def total_shaded_area (carpet_side_length : ℝ) (large_square_side : ℝ) (small_square_side : ℝ) : ℝ :=
  let large_shaded_area := large_square_side * large_square_side
  let small_shaded_area := small_square_side * small_square_side
  large_shaded_area + 12 * small_shaded_area

theorem carpet_shaded_area_is_correct :
  ∀ (S T : ℝ), 
  12 / S = 4 →
  S / T = 4 →
  total_shaded_area 12 S T = 15.75 :=
by
  intros S T h1 h2
  sorry

end NUMINAMATH_GPT_carpet_shaded_area_is_correct_l514_51413


namespace NUMINAMATH_GPT_num_of_dogs_l514_51480

theorem num_of_dogs (num_puppies : ℕ) (dog_food_per_meal : ℕ) (dog_meals_per_day : ℕ) (total_food : ℕ)
  (h1 : num_puppies = 4)
  (h2 : dog_food_per_meal = 4)
  (h3 : dog_meals_per_day = 3)
  (h4 : total_food = 108)
  : ∃ (D : ℕ), num_puppies * (dog_food_per_meal / 2) * (dog_meals_per_day * 3) + D * (dog_food_per_meal * dog_meals_per_day) = total_food ∧ D = 3 :=
by
  sorry

end NUMINAMATH_GPT_num_of_dogs_l514_51480


namespace NUMINAMATH_GPT_correct_system_of_equations_l514_51444

theorem correct_system_of_equations (x y : ℝ) :
  (5 * x + 6 * y = 16) ∧ (4 * x + y = x + 5 * y) :=
sorry

end NUMINAMATH_GPT_correct_system_of_equations_l514_51444


namespace NUMINAMATH_GPT_parabola_problem_l514_51432

theorem parabola_problem (a x1 x2 y1 y2 : ℝ)
  (h1 : y1^2 = a * x1)
  (h2 : y2^2 = a * x2)
  (h3 : x1 + x2 = 8)
  (h4 : (x2 - x1)^2 + (y2 - y1)^2 = 144) : 
  a = 8 := 
sorry

end NUMINAMATH_GPT_parabola_problem_l514_51432


namespace NUMINAMATH_GPT_max_xyz_l514_51463

theorem max_xyz (x y z : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : 5 * x + 8 * y + 3 * z = 90) : xyz ≤ 225 :=
by
  sorry

end NUMINAMATH_GPT_max_xyz_l514_51463


namespace NUMINAMATH_GPT_proposition_a_is_true_l514_51410

-- Define a quadrilateral
structure Quadrilateral (α : Type*) [Ring α] :=
(a b c d : α)

-- Define properties of a Quadrilateral
def parallel_and_equal_opposite_sides (Q : Quadrilateral ℝ) : Prop := sorry  -- Assumes parallel and equal opposite sides
def is_parallelogram (Q : Quadrilateral ℝ) : Prop := sorry  -- Defines a parallelogram

-- The theorem we need to prove
theorem proposition_a_is_true (Q : Quadrilateral ℝ) (h : parallel_and_equal_opposite_sides Q) : is_parallelogram Q :=
sorry

end NUMINAMATH_GPT_proposition_a_is_true_l514_51410


namespace NUMINAMATH_GPT_women_with_fair_hair_percentage_l514_51469

-- Define the conditions
variables {E : ℝ} (hE : E > 0)

def percent_factor : ℝ := 100

def employees_have_fair_hair (E : ℝ) : ℝ := 0.80 * E
def fair_hair_women (E : ℝ) : ℝ := 0.40 * (employees_have_fair_hair E)

-- Define the target proof statement
theorem women_with_fair_hair_percentage
  (h1 : E > 0)
  (h2 : employees_have_fair_hair E = 0.80 * E)
  (h3 : fair_hair_women E = 0.40 * (employees_have_fair_hair E)):
  (fair_hair_women E / E) * percent_factor = 32 := 
sorry

end NUMINAMATH_GPT_women_with_fair_hair_percentage_l514_51469


namespace NUMINAMATH_GPT_intersection_sets_l514_51458

theorem intersection_sets :
  let M := {x : ℝ | (x + 3) * (x - 2) < 0 }
  let N := {x : ℝ | 1 ≤ x ∧ x ≤ 3 }
  M ∩ N = {x : ℝ | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_sets_l514_51458


namespace NUMINAMATH_GPT_frog_escape_l514_51446

theorem frog_escape (wellDepth dayClimb nightSlide escapeDays : ℕ)
  (h_depth : wellDepth = 30)
  (h_dayClimb : dayClimb = 3)
  (h_nightSlide : nightSlide = 2)
  (h_escape : escapeDays = 28) :
  ∃ n, n = escapeDays ∧
       ((wellDepth ≤ (n - 1) * (dayClimb - nightSlide) + dayClimb)) :=
by
  sorry

end NUMINAMATH_GPT_frog_escape_l514_51446


namespace NUMINAMATH_GPT_prove_if_alpha_parallel_beta_and_a_perpendicular_beta_then_a_perpendicular_alpha_l514_51466

-- Definitions of the entities involved
variables {L : Type} -- All lines
variables {P : Type} -- All planes

-- Relations
variables (perpendicular : L → P → Prop)
variables (parallel : P → P → Prop)

-- Conditions
variables (a b : L)
variables (α β : P)

-- Statements we want to prove
theorem prove_if_alpha_parallel_beta_and_a_perpendicular_beta_then_a_perpendicular_alpha
  (H1 : parallel α β) 
  (H2 : perpendicular a β) : 
  perpendicular a α :=
  sorry

end NUMINAMATH_GPT_prove_if_alpha_parallel_beta_and_a_perpendicular_beta_then_a_perpendicular_alpha_l514_51466


namespace NUMINAMATH_GPT_sam_gave_joan_seashells_l514_51443

variable (original_seashells : ℕ) (total_seashells : ℕ)

theorem sam_gave_joan_seashells (h1 : original_seashells = 70) (h2 : total_seashells = 97) :
  total_seashells - original_seashells = 27 :=
by
  sorry

end NUMINAMATH_GPT_sam_gave_joan_seashells_l514_51443


namespace NUMINAMATH_GPT_find_A_l514_51479

theorem find_A (A : ℝ) (h : (12 + 3) * (12 - A) = 120) : A = 4 :=
by sorry

end NUMINAMATH_GPT_find_A_l514_51479


namespace NUMINAMATH_GPT_part1_l514_51425

theorem part1 (a b : ℝ) : 3*(a - b)^2 - 6*(a - b)^2 + 2*(a - b)^2 = - (a - b)^2 :=
by
  sorry

end NUMINAMATH_GPT_part1_l514_51425


namespace NUMINAMATH_GPT_notebook_ratio_l514_51497

theorem notebook_ratio (C N : ℕ) (h1 : ∀ k, N = k / C)
  (h2 : ∃ k, N = k / (C / 2) ∧ 16 = k / (C / 2))
  (h3 : C * N = 512) : (N : ℚ) / C = 1 / 8 := 
by
  sorry

end NUMINAMATH_GPT_notebook_ratio_l514_51497


namespace NUMINAMATH_GPT_regular_polygons_from_cube_intersection_l514_51405

noncomputable def cube : Type := sorry  -- Define a 3D cube type
noncomputable def plane : Type := sorry  -- Define a plane type

-- Define what it means for a polygon to be regular (equilateral and equiangular)
def is_regular_polygon (polygon : Type) : Prop := sorry

-- Define a function that describes the intersection of a plane with a cube,
-- resulting in a polygon
noncomputable def intersection (c : cube) (p : plane) : Type := sorry

-- Define predicates for the specific regular polygons: triangle, quadrilateral, and hexagon
def is_triangle (polygon : Type) : Prop := sorry
def is_quadrilateral (polygon : Type) : Prop := sorry
def is_hexagon (polygon : Type) : Prop := sorry

-- Ensure these predicates imply regular polygons
axiom triangle_is_regular : ∀ (t : Type), is_triangle t → is_regular_polygon t
axiom quadrilateral_is_regular : ∀ (q : Type), is_quadrilateral q → is_regular_polygon q
axiom hexagon_is_regular : ∀ (h : Type), is_hexagon h → is_regular_polygon h

-- The main theorem statement
theorem regular_polygons_from_cube_intersection (c : cube) (p : plane) :
  is_regular_polygon (intersection c p) →
  is_triangle (intersection c p) ∨ is_quadrilateral (intersection c p) ∨ is_hexagon (intersection c p) :=
sorry

end NUMINAMATH_GPT_regular_polygons_from_cube_intersection_l514_51405


namespace NUMINAMATH_GPT_solve_quadratic_l514_51484

theorem solve_quadratic (x : ℝ) : x^2 - 2*x = 0 ↔ (x = 0 ∨ x = 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l514_51484


namespace NUMINAMATH_GPT_choose_3_of_9_colors_l514_51483

-- Define the combination function
noncomputable def combination (n k : ℕ) := n.choose k

-- Noncomputable because factorial and combination require division.
noncomputable def combination_9_3 := combination 9 3

-- State the theorem we are proving
theorem choose_3_of_9_colors : combination_9_3 = 84 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_choose_3_of_9_colors_l514_51483


namespace NUMINAMATH_GPT_convert_base_9A3_16_to_4_l514_51418

theorem convert_base_9A3_16_to_4 :
  let h₁ := 9
  let h₂ := 10 -- A in hexadecimal
  let h₃ := 3
  let b₁ := 21 -- h₁ converted to base 4
  let b₂ := 22 -- h₂ converted to base 4
  let b₃ := 3  -- h₃ converted to base 4
  9 * 16^2 + 10 * 16^1 + 3 * 16^0 = 2 * 4^5 + 1 * 4^4 + 2 * 4^3 + 2 * 4^2 + 0 * 4^1 + 3 * 4^0 :=
by
  sorry

end NUMINAMATH_GPT_convert_base_9A3_16_to_4_l514_51418


namespace NUMINAMATH_GPT_greatest_integer_y_l514_51455

theorem greatest_integer_y (y : ℤ) : (8 : ℚ) / 11 > y / 17 ↔ y ≤ 12 := 
sorry

end NUMINAMATH_GPT_greatest_integer_y_l514_51455


namespace NUMINAMATH_GPT_eq_4_double_prime_l514_51477

-- Define the function f such that f(q) = 3q - 3
def f (q : ℕ) : ℕ := 3 * q - 3

-- Theorem statement to show that f(f(4)) = 24
theorem eq_4_double_prime : f (f 4) = 24 := by
  sorry

end NUMINAMATH_GPT_eq_4_double_prime_l514_51477


namespace NUMINAMATH_GPT_shaded_region_is_correct_l514_51474

noncomputable def area_shaded_region : ℝ :=
  let r_small := (3 : ℝ) / 2
  let r_large := (15 : ℝ) / 2
  let area_small := (1 / 2) * Real.pi * r_small^2
  let area_large := (1 / 2) * Real.pi * r_large^2
  (area_large - 2 * area_small + 3 * area_small)

theorem shaded_region_is_correct :
  area_shaded_region = (117 / 4) * Real.pi :=
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_shaded_region_is_correct_l514_51474


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l514_51467

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x^2 + y^2 ≤ 1) → ((x - 1)^2 + y^2 ≤ 4) ∧ ¬ ((x - 1)^2 + y^2 ≤ 4 → x^2 + y^2 ≤ 1) :=
by sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l514_51467


namespace NUMINAMATH_GPT_water_added_is_five_l514_51431

theorem water_added_is_five :
  ∃ W x : ℝ, (4 / 3 = 10 / W) ∧ (4 / 5 = 10 / (W + x)) ∧ x = 5 := by
  sorry

end NUMINAMATH_GPT_water_added_is_five_l514_51431


namespace NUMINAMATH_GPT_find_hyperbola_focus_l514_51470

theorem find_hyperbola_focus : ∃ (x y : ℝ), 
  2 * x ^ 2 - 3 * y ^ 2 + 8 * x - 12 * y - 8 = 0 
  → (x, y) = (-2 + (Real.sqrt 30)/3, -2) :=
by
  sorry

end NUMINAMATH_GPT_find_hyperbola_focus_l514_51470


namespace NUMINAMATH_GPT_tension_limit_l514_51401

theorem tension_limit (M m g : ℝ) (hM : 0 < M) (hg : 0 < g) :
  (∀ T, (T = Mg ↔ m = 0) → (∀ ε, 0 < ε → ∃ m₀, m > m₀ → |T - 2 * M * g| < ε)) :=
by 
  sorry

end NUMINAMATH_GPT_tension_limit_l514_51401


namespace NUMINAMATH_GPT_compare_abc_l514_51475

noncomputable def a : ℝ := (1 / 6) ^ (1 / 2)
noncomputable def b : ℝ := Real.log 1 / 3 / Real.log 6
noncomputable def c : ℝ := Real.log 1 / 7 / Real.log (1 / 6)

theorem compare_abc : c > a ∧ a > b := by
  sorry

end NUMINAMATH_GPT_compare_abc_l514_51475


namespace NUMINAMATH_GPT_terrell_lifting_l514_51493

theorem terrell_lifting :
  (3 * 25 * 10 = 3 * 20 * 12.5) :=
by
  sorry

end NUMINAMATH_GPT_terrell_lifting_l514_51493


namespace NUMINAMATH_GPT_ratio_is_three_l514_51409

-- Define the conditions
def area_of_garden : ℕ := 588
def width_of_garden : ℕ := 14
def length_of_garden : ℕ := area_of_garden / width_of_garden

-- Define the ratio
def ratio_length_to_width := length_of_garden / width_of_garden

-- The proof statement
theorem ratio_is_three : ratio_length_to_width = 3 := 
by sorry

end NUMINAMATH_GPT_ratio_is_three_l514_51409


namespace NUMINAMATH_GPT_line_tangent_to_parabola_l514_51472

theorem line_tangent_to_parabola (k : ℝ) :
  (∀ (x y : ℝ), y^2 = 16 * x → 4 * x + 3 * y + k = 0) → k = 9 :=
by
  sorry

end NUMINAMATH_GPT_line_tangent_to_parabola_l514_51472


namespace NUMINAMATH_GPT_minimum_value_of_a_l514_51422

variable (a x y : ℝ)

-- Condition
def condition (x y : ℝ) (a : ℝ) : Prop := 
  (x + y) * ((1/x) + (a/y)) ≥ 9

-- Main statement
theorem minimum_value_of_a : (∀ x > 0, ∀ y > 0, condition x y a) → a ≥ 4 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_a_l514_51422


namespace NUMINAMATH_GPT_ratio_of_costs_l514_51498

-- Definitions based on conditions
def old_car_cost : ℕ := 1800
def new_car_cost : ℕ := 1800 + 2000

-- Theorem stating the desired proof
theorem ratio_of_costs :
  (new_car_cost / old_car_cost : ℚ) = 19 / 9 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_costs_l514_51498


namespace NUMINAMATH_GPT_dave_hourly_wage_l514_51454

theorem dave_hourly_wage :
  ∀ (hours_monday hours_tuesday total_money : ℝ),
  hours_monday = 6 → hours_tuesday = 2 → total_money = 48 →
  (total_money / (hours_monday + hours_tuesday) = 6) :=
by
  intros hours_monday hours_tuesday total_money h_monday h_tuesday h_money
  sorry

end NUMINAMATH_GPT_dave_hourly_wage_l514_51454


namespace NUMINAMATH_GPT_sqrt_fraction_l514_51427

theorem sqrt_fraction {a b c : ℝ}
  (h1 : a = Real.sqrt 27)
  (h2 : b = Real.sqrt 243)
  (h3 : c = Real.sqrt 48) :
  (a + b) / c = 3 := by
  sorry

end NUMINAMATH_GPT_sqrt_fraction_l514_51427


namespace NUMINAMATH_GPT_triangle_angle_C_l514_51408

theorem triangle_angle_C (A B C : ℝ) (h : A + B = 80) : C = 100 :=
sorry

end NUMINAMATH_GPT_triangle_angle_C_l514_51408


namespace NUMINAMATH_GPT_triangle_side_b_length_l514_51457

noncomputable def length_of_side_b (A B C a b c : ℝ) (h1 : a = 1)
  (h2 : Real.cos A = 4/5) (h3 : Real.cos C = 5/13) : Prop :=
  b = 21 / 13

theorem triangle_side_b_length (A B C a b c : ℝ) (h1 : a = 1)
  (h2 : Real.cos A = 4/5) (h3 : Real.cos C = 5/13) :
  length_of_side_b A B C a b c h1 h2 h3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_b_length_l514_51457


namespace NUMINAMATH_GPT_mutually_exclusive_one_two_odd_l514_51464

-- Define the event that describes rolling a fair die
def is_odd (n : ℕ) : Prop := n % 2 = 1

/-- Event: Exactly one die shows an odd number -/
def exactly_one_odd (d1 d2 : ℕ) : Prop :=
  (is_odd d1 ∧ ¬ is_odd d2) ∨ (¬ is_odd d1 ∧ is_odd d2)

/-- Event: Exactly two dice show odd numbers -/
def exactly_two_odd (d1 d2 : ℕ) : Prop :=
  is_odd d1 ∧ is_odd d2

/-- Main theorem: Exactly one odd number and exactly two odd numbers are mutually exclusive but not converse-/
theorem mutually_exclusive_one_two_odd (d1 d2 : ℕ) :
  (exactly_one_odd d1 d2 ∧ ¬ exactly_two_odd d1 d2) ∧
  (¬ exactly_one_odd d1 d2 ∧ exactly_two_odd d1 d2) ∧
  (exactly_one_odd d1 d2 ∨ exactly_two_odd d1 d2) :=
by
  sorry

end NUMINAMATH_GPT_mutually_exclusive_one_two_odd_l514_51464


namespace NUMINAMATH_GPT_quadrants_cos_sin_identity_l514_51487

theorem quadrants_cos_sin_identity (α : ℝ) 
  (h1 : π < α ∧ α < 2 * π)  -- α in the fourth quadrant
  (h2 : Real.cos α = 3 / 5) :
  (1 + Real.sqrt 2 * Real.cos (2 * α - π / 4)) / 
  (Real.sin (α + π / 2)) = -2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_quadrants_cos_sin_identity_l514_51487


namespace NUMINAMATH_GPT_Tim_Linda_Mow_Lawn_l514_51465

theorem Tim_Linda_Mow_Lawn :
  let tim_time := 1.5
  let linda_time := 2
  let tim_rate := 1 / tim_time
  let linda_rate := 1 / linda_time
  let combined_rate := tim_rate + linda_rate
  let combined_time_hours := 1 / combined_rate
  let combined_time_minutes := combined_time_hours * 60
  combined_time_minutes = 51.43 := 
by
    sorry

end NUMINAMATH_GPT_Tim_Linda_Mow_Lawn_l514_51465


namespace NUMINAMATH_GPT_is_divisible_by_7_l514_51489

theorem is_divisible_by_7 : ∃ k : ℕ, 42 = 7 * k := by
  sorry

end NUMINAMATH_GPT_is_divisible_by_7_l514_51489


namespace NUMINAMATH_GPT_parking_average_cost_l514_51456

noncomputable def parking_cost_per_hour := 
  let cost_two_hours : ℝ := 20.00
  let cost_per_excess_hour : ℝ := 1.75
  let weekend_surcharge : ℝ := 5.00
  let discount_rate : ℝ := 0.10
  let total_hours : ℝ := 9.00
  let excess_hours : ℝ := total_hours - 2.00
  let remaining_cost := cost_per_excess_hour * excess_hours
  let total_cost_before_discount := cost_two_hours + remaining_cost + weekend_surcharge
  let discount := discount_rate * total_cost_before_discount
  let discounted_total_cost := total_cost_before_discount - discount
  let average_cost_per_hour := discounted_total_cost / total_hours
  average_cost_per_hour

theorem parking_average_cost :
  parking_cost_per_hour = 3.725 := 
by
  sorry

end NUMINAMATH_GPT_parking_average_cost_l514_51456


namespace NUMINAMATH_GPT_math_proof_problem_l514_51440

theorem math_proof_problem
  (n m k l : ℕ)
  (hpos_n : n > 0)
  (hpos_m : m > 0)
  (hpos_k : k > 0)
  (hpos_l : l > 0)
  (hneq_n : n ≠ 1)
  (hdiv : n^k + m*n^l + 1 ∣ n^(k+l) - 1) :
  (m = 1 ∧ l = 2*k) ∨ (l ∣ k ∧ m = (n^(k-l) - 1) / (n^l - 1)) :=
by 
  sorry

end NUMINAMATH_GPT_math_proof_problem_l514_51440


namespace NUMINAMATH_GPT_geometric_sequence_sixth_term_l514_51434

variable (a r : ℝ) 

theorem geometric_sequence_sixth_term (h1 : a * (1 + r + r^2 + r^3) = 40)
                                    (h2 : a * r^4 = 32) :
  a * r^5 = 1280 / 15 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_sixth_term_l514_51434


namespace NUMINAMATH_GPT_find_number_l514_51492

theorem find_number (x : ℝ) (h : 54 / 2 + 3 * x = 75) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l514_51492


namespace NUMINAMATH_GPT_height_of_building_l514_51403

-- Define the conditions
def height_flagpole : ℝ := 18
def shadow_flagpole : ℝ := 45
def shadow_building : ℝ := 55

-- State the theorem to prove the height of the building
theorem height_of_building (h : ℝ) : (height_flagpole / shadow_flagpole) = (h / shadow_building) → h = 22 :=
by
  sorry

end NUMINAMATH_GPT_height_of_building_l514_51403


namespace NUMINAMATH_GPT_fractional_part_zero_l514_51441

noncomputable def fractional_part (z : ℝ) : ℝ := z - (⌊z⌋ : ℝ)

theorem fractional_part_zero (x : ℝ) :
  fractional_part (1 / 3 * (1 / 3 * (1 / 3 * x - 3) - 3) - 3) = 0 ↔ 
  ∃ k : ℤ, 27 * k + 9 ≤ x ∧ x < 27 * k + 18 :=
by
  sorry

end NUMINAMATH_GPT_fractional_part_zero_l514_51441


namespace NUMINAMATH_GPT_solve_for_xy_l514_51402

-- The conditions given in the problem
variables (x y : ℝ)
axiom cond1 : 1 / 2 * x - y = 5
axiom cond2 : y - 1 / 3 * x = 2

-- The theorem we need to prove
theorem solve_for_xy (x y : ℝ) (cond1 : 1 / 2 * x - y = 5) (cond2 : y - 1 / 3 * x = 2) : 
  x = 42 ∧ y = 16 := sorry

end NUMINAMATH_GPT_solve_for_xy_l514_51402


namespace NUMINAMATH_GPT_sequence_twice_square_l514_51436

theorem sequence_twice_square (n : ℕ) (a : ℕ → ℕ) :
    (∀ i : ℕ, a i = 0) →
    (∀ m : ℕ, 1 ≤ m ∧ m ≤ n → 
        ∀ i : ℕ, i % (2 * m) = 0 → 
            a i = if a i = 0 then 1 else 0) →
    (∀ i : ℕ, a i = 1 ↔ ∃ k : ℕ, i = 2 * k^2) :=
by
  sorry

end NUMINAMATH_GPT_sequence_twice_square_l514_51436


namespace NUMINAMATH_GPT_Jan_is_6_inches_taller_than_Bill_l514_51499

theorem Jan_is_6_inches_taller_than_Bill :
  ∀ (Cary Bill Jan : ℕ),
    Cary = 72 →
    Bill = Cary / 2 →
    Jan = 42 →
    Jan - Bill = 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Jan_is_6_inches_taller_than_Bill_l514_51499


namespace NUMINAMATH_GPT_clock_correct_after_240_days_l514_51445

theorem clock_correct_after_240_days (days : ℕ) (minutes_fast_per_day : ℕ) (hours_to_be_correct : ℕ) 
  (h1 : minutes_fast_per_day = 3) (h2 : hours_to_be_correct = 12) : 
  (days * minutes_fast_per_day) % (hours_to_be_correct * 60) = 0 :=
by 
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_clock_correct_after_240_days_l514_51445


namespace NUMINAMATH_GPT_petya_vasya_problem_l514_51438

theorem petya_vasya_problem :
  ∀ n : ℕ, (∀ x : ℕ, x = 12320 * 10 ^ (10 * n + 1) - 1 →
    (∃ p q : ℕ, (p ≠ q ∧ ∀ r : ℕ, (r ∣ x → (r = p ∨ r = q))))) → n = 0 :=
by
  sorry

end NUMINAMATH_GPT_petya_vasya_problem_l514_51438


namespace NUMINAMATH_GPT_identity_holds_for_all_real_numbers_l514_51419

theorem identity_holds_for_all_real_numbers (a b : ℝ) : 
  a^2 + b^2 + 2 * a * b = (a + b)^2 := 
by sorry

end NUMINAMATH_GPT_identity_holds_for_all_real_numbers_l514_51419


namespace NUMINAMATH_GPT_union_of_A_B_l514_51423

def A (p q : ℝ) : Set ℝ := {x | x^2 + p * x + q = 0}
def B (p q : ℝ) : Set ℝ := {x | x^2 - p * x - 2 * q = 0}

theorem union_of_A_B (p q : ℝ)
  (h1 : A p q ∩ B p q = {-1}) :
  A p q ∪ B p q = {-1, -2, 4} := by
sorry

end NUMINAMATH_GPT_union_of_A_B_l514_51423


namespace NUMINAMATH_GPT_solve_for_x_y_l514_51488

noncomputable def x_y_2018_sum (x y : ℝ) : ℝ := x^2018 + y^2018

theorem solve_for_x_y (A B : Set ℝ) (x y : ℝ)
  (hA : A = {x, x * y, x + y})
  (hB : B = {0, |x|, y}) 
  (h : A = B) :
  x_y_2018_sum x y = 2 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_y_l514_51488


namespace NUMINAMATH_GPT_largest_integer_in_mean_set_l514_51421

theorem largest_integer_in_mean_set :
  ∃ (A B C D : ℕ), 
    A < B ∧ B < C ∧ C < D ∧
    (A + B + C + D) = 4 * 68 ∧
    A ≥ 5 ∧
    D = 254 :=
sorry

end NUMINAMATH_GPT_largest_integer_in_mean_set_l514_51421


namespace NUMINAMATH_GPT_combined_6th_grade_percent_is_15_l514_51481

-- Definitions
def annville_students := 100
def cleona_students := 200

def percent_6th_annville := 11
def percent_6th_cleona := 17

def total_students := annville_students + cleona_students
def total_6th_students := (percent_6th_annville * annville_students / 100) + (percent_6th_cleona * cleona_students / 100)

def percent_6th_combined := (total_6th_students * 100) / total_students

-- Theorem statement
theorem combined_6th_grade_percent_is_15 : percent_6th_combined = 15 :=
by
  sorry

end NUMINAMATH_GPT_combined_6th_grade_percent_is_15_l514_51481


namespace NUMINAMATH_GPT_probability_top_two_same_suit_l514_51406

theorem probability_top_two_same_suit :
  let deck_size := 52
  let suits := 4
  let cards_per_suit := 13
  let first_card_prob := (13 / 52 : ℚ)
  let remaining_cards := 51
  let second_card_same_suit_prob := (12 / 51 : ℚ)
  first_card_prob * second_card_same_suit_prob = (1 / 17 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_probability_top_two_same_suit_l514_51406


namespace NUMINAMATH_GPT_union_is_faction_l514_51407

variable {D : Type} (is_faction : Set D → Prop)
variable (A B : Set D)

-- Define the complement
def complement (S : Set D) : Set D := {x | x ∉ S}

-- State the given condition
axiom faction_complement_union (A B : Set D) : 
  is_faction A → is_faction B → is_faction (complement (A ∪ B))

-- The theorem to prove
theorem union_is_faction (A B : Set D) :
  is_faction A → is_faction B → is_faction (A ∪ B) := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_union_is_faction_l514_51407


namespace NUMINAMATH_GPT_convert_spherical_coords_l514_51412

theorem convert_spherical_coords (ρ θ φ : ℝ) (hρ : ρ > 0) (hθ : 0 ≤ θ ∧ θ < 2 * π) (hφ : 0 ≤ φ ∧ φ ≤ π) :
  (ρ = 4 ∧ θ = 4 * π / 3 ∧ φ = π / 4) ↔ (ρ, θ, φ) = (4, 4 * π / 3, π / 4) :=
by { sorry }

end NUMINAMATH_GPT_convert_spherical_coords_l514_51412


namespace NUMINAMATH_GPT_bob_total_miles_l514_51494

def total_miles_day1 (T : ℝ) := 0.20 * T
def remaining_miles_day1 (T : ℝ) := T - total_miles_day1 T
def total_miles_day2 (T : ℝ) := 0.50 * remaining_miles_day1 T
def remaining_miles_day2 (T : ℝ) := remaining_miles_day1 T - total_miles_day2 T
def total_miles_day3 (T : ℝ) := 28

theorem bob_total_miles (T : ℝ) (h : total_miles_day3 T = remaining_miles_day2 T) : T = 70 :=
by
  sorry

end NUMINAMATH_GPT_bob_total_miles_l514_51494


namespace NUMINAMATH_GPT_number_of_possible_values_of_r_eq_894_l514_51482

noncomputable def r_possible_values : ℕ :=
  let lower_bound := 0.3125
  let upper_bound := 0.4018
  let min_r := 3125  -- equivalent to the lowest four-digit decimal ≥ 0.3125
  let max_r := 4018  -- equivalent to the highest four-digit decimal ≤ 0.4018
  1 + max_r - min_r  -- total number of possible values

theorem number_of_possible_values_of_r_eq_894 :
  r_possible_values = 894 :=
by
  sorry

end NUMINAMATH_GPT_number_of_possible_values_of_r_eq_894_l514_51482


namespace NUMINAMATH_GPT_second_car_speed_correct_l514_51449

noncomputable def first_car_speed : ℝ := 90

noncomputable def time_elapsed (h : ℕ) (m : ℕ) : ℝ := h + m / 60

noncomputable def distance_travelled (speed : ℝ) (time : ℝ) : ℝ := speed * time

def distance_ratio_at_832 (dist1 dist2 : ℝ) : Prop := dist1 = 1.2 * dist2
def distance_ratio_at_920 (dist1 dist2 : ℝ) : Prop := dist1 = 2 * dist2

noncomputable def time_first_car_832 : ℝ := time_elapsed 0 24
noncomputable def dist_first_car_832 : ℝ := distance_travelled first_car_speed time_first_car_832

noncomputable def dist_second_car_832 : ℝ := dist_first_car_832 / 1.2

noncomputable def time_first_car_920 : ℝ := time_elapsed 1 12
noncomputable def dist_first_car_920 : ℝ := distance_travelled first_car_speed time_first_car_920

noncomputable def dist_second_car_920 : ℝ := dist_first_car_920 / 2

noncomputable def time_second_car_travel : ℝ := time_elapsed 0 42

noncomputable def second_car_speed : ℝ := (dist_second_car_920 - dist_second_car_832) / time_second_car_travel

theorem second_car_speed_correct :
  second_car_speed = 34.2857 := by
  sorry

end NUMINAMATH_GPT_second_car_speed_correct_l514_51449


namespace NUMINAMATH_GPT_math_problem_l514_51415

theorem math_problem : ((3.6 * 0.3) / 0.6 = 1.8) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l514_51415


namespace NUMINAMATH_GPT_alpha_less_than_60_degrees_l514_51471

theorem alpha_less_than_60_degrees
  (R r : ℝ)
  (b c : ℝ)
  (α : ℝ)
  (h1 : b * c = 8 * R * r) :
  α < 60 := sorry

end NUMINAMATH_GPT_alpha_less_than_60_degrees_l514_51471


namespace NUMINAMATH_GPT_function_equality_l514_51461

theorem function_equality (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 / x) = x / (1 - x)) :
  ∀ x : ℝ, f x = 1 / (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_function_equality_l514_51461


namespace NUMINAMATH_GPT_average_after_adding_ten_l514_51486

theorem average_after_adding_ten (avg initial_sum new_mean : ℕ) (n : ℕ) (h1 : n = 15) (h2 : avg = 40) (h3 : initial_sum = n * avg) (h4 : new_mean = (initial_sum + n * 10) / n) : new_mean = 50 := 
by
  sorry

end NUMINAMATH_GPT_average_after_adding_ten_l514_51486


namespace NUMINAMATH_GPT_ratio_neha_mother_age_12_years_ago_l514_51462

variables (N : ℕ) (M : ℕ) (X : ℕ)

theorem ratio_neha_mother_age_12_years_ago 
  (hM : M = 60)
  (h_future : M + 12 = 2 * (N + 12)) :
  (12 : ℕ) * (M - 12) = (48 : ℕ) * (N - 12) :=
by
  sorry

end NUMINAMATH_GPT_ratio_neha_mother_age_12_years_ago_l514_51462


namespace NUMINAMATH_GPT_largest_solution_l514_51437

-- Define the largest solution to the equation |5x - 3| = 28 as 31/5.
theorem largest_solution (x : ℝ) (h : |5 * x - 3| = 28) : x ≤ 31 / 5 := 
  sorry

end NUMINAMATH_GPT_largest_solution_l514_51437


namespace NUMINAMATH_GPT_num_of_friends_donated_same_l514_51485

def total_clothing_donated_by_adam (pants jumpers pajama_sets t_shirts : ℕ) : ℕ :=
  pants + jumpers + 2 * pajama_sets + t_shirts

def clothing_kept_by_adam (initial_donation : ℕ) : ℕ :=
  initial_donation / 2

def clothing_donated_by_friends (total_donated keeping friends_donation : ℕ) : ℕ :=
  total_donated - keeping

def num_friends (friends_donation adam_initial_donation : ℕ) : ℕ :=
  friends_donation / adam_initial_donation

theorem num_of_friends_donated_same (pants jumpers pajama_sets t_shirts total_donated : ℕ)
  (initial_donation := total_clothing_donated_by_adam pants jumpers pajama_sets t_shirts)
  (keeping := clothing_kept_by_adam initial_donation)
  (friends_donation := clothing_donated_by_friends total_donated keeping initial_donation)
  (friends := num_friends friends_donation initial_donation)
  (hp : pants = 4)
  (hj : jumpers = 4)
  (hps : pajama_sets = 4)
  (ht : t_shirts = 20)
  (htotal : total_donated = 126) :
  friends = 3 :=
by
  sorry

end NUMINAMATH_GPT_num_of_friends_donated_same_l514_51485


namespace NUMINAMATH_GPT_solve_abs_inequality_l514_51420

theorem solve_abs_inequality (x : ℝ) (h : abs ((8 - x) / 4) < 3) : -4 < x ∧ x < 20 := 
  sorry

end NUMINAMATH_GPT_solve_abs_inequality_l514_51420


namespace NUMINAMATH_GPT_smallest_integer_to_multiply_y_to_make_perfect_square_l514_51417

noncomputable def y : ℕ :=
  3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_integer_to_multiply_y_to_make_perfect_square :
  ∃ k : ℕ, k > 0 ∧ (∃ m : ℕ, (k * y) = m^2) ∧ k = 3 := by
  sorry

end NUMINAMATH_GPT_smallest_integer_to_multiply_y_to_make_perfect_square_l514_51417


namespace NUMINAMATH_GPT_sum_of_A_B_in_B_l514_51453

def A : Set ℤ := { x | ∃ k : ℤ, x = 2 * k }
def B : Set ℤ := { x | ∃ k : ℤ, x = 2 * k + 1 }
def C : Set ℤ := { x | ∃ k : ℤ, x = 4 * k + 1 }

theorem sum_of_A_B_in_B (a b : ℤ) (ha : a ∈ A) (hb : b ∈ B) : a + b ∈ B := by
  sorry

end NUMINAMATH_GPT_sum_of_A_B_in_B_l514_51453


namespace NUMINAMATH_GPT_sum_of_squares_l514_51448

theorem sum_of_squares (x y z : ℕ) (h1 : x + y + z = 30)
  (h2 : Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 12) :
  x^2 + y^2 + z^2 = 504 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l514_51448


namespace NUMINAMATH_GPT_arithmetic_progression_conditions_l514_51496

theorem arithmetic_progression_conditions (a d : ℝ) :
  let x := a
  let y := a + d
  let z := a + 2 * d
  (y^2 = (x^2 * z^2)^(1/2)) ↔ (d = 0 ∨ d = a * (-2 + Real.sqrt 2) ∨ d = a * (-2 - Real.sqrt 2)) :=
by
  intros
  sorry

end NUMINAMATH_GPT_arithmetic_progression_conditions_l514_51496


namespace NUMINAMATH_GPT_simplify_fraction_expression_l514_51495

variable (d : ℝ)

theorem simplify_fraction_expression :
  (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_expression_l514_51495


namespace NUMINAMATH_GPT_ellipse_properties_l514_51491

theorem ellipse_properties :
  (∀ x y: ℝ, (x^2)/100 + (y^2)/36 = 1) →
  ∃ a b c e : ℝ, 
  a = 10 ∧ 
  b = 6 ∧ 
  c = 8 ∧ 
  2 * a = 20 ∧ 
  e = 4 / 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ellipse_properties_l514_51491
