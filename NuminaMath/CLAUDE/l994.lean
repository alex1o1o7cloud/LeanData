import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l994_99405

-- Define the hyperbola C
def C (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the focal distance
def focal_distance (c : ℝ) : Prop := c = 2

-- Define a point on the hyperbola
def point_on_hyperbola (a b : ℝ) : Prop :=
  C a b 2 3

-- Theorem statement
theorem hyperbola_eccentricity (a b c : ℝ) :
  C a b 2 3 → focal_distance c → c / a = 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l994_99405


namespace NUMINAMATH_CALUDE_range_of_m_l994_99435

theorem range_of_m (m : ℝ) : 
  (¬((m + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m*x + 1 > 0))) → 
  m ∈ Set.Iic (-2) ∪ Set.Ioi (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l994_99435


namespace NUMINAMATH_CALUDE_simplified_expression_l994_99417

theorem simplified_expression (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (3 * x^2 - 2 * x - 5) / ((x - 3) * (x + 2)) - (5 * x - 6) / ((x - 3) * (x + 2)) =
  3 * (x - (7 + Real.sqrt 37) / 6) * (x - (7 - Real.sqrt 37) / 6) / ((x - 3) * (x + 2)) := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_l994_99417


namespace NUMINAMATH_CALUDE_base8_digit_product_8654_l994_99428

/-- Convert a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculate the product of a list of natural numbers --/
def listProduct (l : List ℕ) : ℕ :=
  sorry

/-- The product of the digits in the base 8 representation of 8654₁₀ is 0 --/
theorem base8_digit_product_8654 :
  listProduct (toBase8 8654) = 0 :=
sorry

end NUMINAMATH_CALUDE_base8_digit_product_8654_l994_99428


namespace NUMINAMATH_CALUDE_constant_revenue_increase_l994_99466

def revenue : Fin 14 → ℕ
  | 0  => 150000  -- January (year 1)
  | 1  => 180000  -- February (year 1)
  | 2  => 210000  -- March (year 1)
  | 3  => 240000  -- April (year 1)
  | 4  => 270000  -- May (year 1)
  | 5  => 300000  -- June (year 1)
  | 6  => 330000  -- July (year 1)
  | 7  => 300000  -- August (year 1)
  | 8  => 270000  -- September (year 1)
  | 9  => 300000  -- October (year 1)
  | 10 => 330000  -- November (year 1)
  | 11 => 360000  -- December (year 1)
  | 12 => 390000  -- January (year 2)
  | 13 => 420000  -- February (year 2)

theorem constant_revenue_increase :
  ∀ i : Fin 13, i.val ≠ 6 ∧ i.val ≠ 7 →
    revenue (i + 1) - revenue i = 30000 :=
by sorry

end NUMINAMATH_CALUDE_constant_revenue_increase_l994_99466


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l994_99441

/-- The distance between the foci of an ellipse given by the equation
    9x^2 - 36x + 4y^2 + 16y + 16 = 0 is 2√5 -/
theorem ellipse_foci_distance : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, 9*x^2 - 36*x + 4*y^2 + 16*y + 16 = 0 ↔ 
      (x - 2)^2 / a^2 + (y + 2)^2 / b^2 = 1) ∧
    a^2 - b^2 = c^2 ∧
    2 * c = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l994_99441


namespace NUMINAMATH_CALUDE_main_result_l994_99459

/-- Average of two numbers -/
def avg (a b : ℚ) : ℚ := (a + b) / 2

/-- Weighted average of four numbers with weights 1:2:1:2 -/
def wavg (a b c d : ℚ) : ℚ := (a + 2*b + c + 2*d) / 6

/-- The main theorem to prove -/
theorem main_result : wavg (wavg 2 2 1 1) (avg 1 2) 0 2 = 17/12 := by sorry

end NUMINAMATH_CALUDE_main_result_l994_99459


namespace NUMINAMATH_CALUDE_f_1_equals_5_l994_99426

-- Define the quadratic polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom quad_f : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
axiom quad_g : ∃ a b c : ℝ, ∀ x, g x = a * x^2 + b * x + c
axiom f_2_3 : f 2 = 2 ∧ f 3 = 2
axiom g_2_3 : g 2 = 2 ∧ g 3 = 2
axiom g_1 : g 1 = 2
axiom f_5 : f 5 = 7
axiom g_5 : g 5 = 2

-- State the theorem
theorem f_1_equals_5 : f 1 = 5 := by sorry

end NUMINAMATH_CALUDE_f_1_equals_5_l994_99426


namespace NUMINAMATH_CALUDE_l_shaped_field_area_l994_99469

theorem l_shaped_field_area :
  let field_length : ℕ := 10
  let field_width : ℕ := 7
  let removed_length_diff : ℕ := 3
  let removed_width_diff : ℕ := 2
  let removed_length : ℕ := field_length - removed_length_diff
  let removed_width : ℕ := field_width - removed_width_diff
  let total_area : ℕ := field_length * field_width
  let removed_area : ℕ := removed_length * removed_width
  let l_shaped_area : ℕ := total_area - removed_area
  l_shaped_area = 35 := by sorry

end NUMINAMATH_CALUDE_l_shaped_field_area_l994_99469


namespace NUMINAMATH_CALUDE_traffic_class_total_l994_99442

/-- The number of drunk drivers in the traffic class -/
def drunk_drivers : ℕ := 6

/-- The number of speeders in the traffic class -/
def speeders : ℕ := 7 * drunk_drivers - 3

/-- The total number of students in the traffic class -/
def total_students : ℕ := drunk_drivers + speeders

/-- Theorem stating that the total number of students in the traffic class is 45 -/
theorem traffic_class_total : total_students = 45 := by sorry

end NUMINAMATH_CALUDE_traffic_class_total_l994_99442


namespace NUMINAMATH_CALUDE_tomatoes_left_theorem_l994_99454

/-- Calculates the number of tomatoes left after processing -/
def tomatoes_left (plants : ℕ) (tomatoes_per_plant : ℕ) : ℕ :=
  let total := plants * tomatoes_per_plant
  let dried := total / 2
  let remaining := total - dried
  let marinara := remaining / 3
  remaining - marinara

/-- Theorem: Given 18 plants with 7 tomatoes each, after processing, 42 tomatoes are left -/
theorem tomatoes_left_theorem : tomatoes_left 18 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_theorem_l994_99454


namespace NUMINAMATH_CALUDE_right_triangle_from_angle_condition_l994_99419

theorem right_triangle_from_angle_condition (A B C : Real) :
  -- Triangle condition
  A + B + C = 180 →
  -- Given angle condition
  A = B ∧ A = (1/2) * C →
  -- Conclusion: C is a right angle
  C = 90 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_from_angle_condition_l994_99419


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l994_99489

/-- Given a cubic polynomial x^3 - 3x - 2 = 0 with roots a, b, and c,
    prove that a(b + c)^2 + b(c + a)^2 + c(a + b)^2 = 6 -/
theorem cubic_root_sum_squares (a b c : ℝ) : 
  a^3 - 3*a - 2 = 0 → 
  b^3 - 3*b - 2 = 0 → 
  c^3 - 3*c - 2 = 0 → 
  a*(b + c)^2 + b*(c + a)^2 + c*(a + b)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l994_99489


namespace NUMINAMATH_CALUDE_sin_sum_angles_l994_99487

/-- Given a point A(1, 2) on the terminal side of angle α in the Cartesian plane,
    and angle β formed by rotating α's terminal side counterclockwise by π/2,
    prove that sin(α + β) = -3/5 -/
theorem sin_sum_angles (α β : Real) : 
  (∃ A : ℝ × ℝ, A = (1, 2) ∧ A.1 = Real.cos α * Real.sqrt (A.1^2 + A.2^2) ∧ 
                   A.2 = Real.sin α * Real.sqrt (A.1^2 + A.2^2)) →
  β = α + π/2 →
  Real.sin (α + β) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_angles_l994_99487


namespace NUMINAMATH_CALUDE_quadratic_rational_root_parity_l994_99437

theorem quadratic_rational_root_parity (a b c : ℤ) (x : ℚ) : 
  a ≠ 0 → 
  a * x^2 + b * x + c = 0 → 
  ¬(Odd a ∧ Odd b ∧ Odd c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_parity_l994_99437


namespace NUMINAMATH_CALUDE_difference_105th_100th_term_l994_99415

def arithmeticSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem difference_105th_100th_term :
  let a₁ := 3
  let d := 5
  (arithmeticSequence a₁ d 105) - (arithmeticSequence a₁ d 100) = 25 := by
  sorry

end NUMINAMATH_CALUDE_difference_105th_100th_term_l994_99415


namespace NUMINAMATH_CALUDE_no_solutions_for_star_equation_l994_99492

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y

-- Theorem statement
theorem no_solutions_for_star_equation :
  ¬ ∃ y : ℝ, star 2 y = 20 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_star_equation_l994_99492


namespace NUMINAMATH_CALUDE_constant_product_l994_99496

-- Define the circle and points
variable (Circle : Type) (A B C D : Point)
variable (diameter : Circle → Point → Point → Prop)
variable (tangent : Circle → Point → Prop)
variable (on_circle : Circle → Point → Prop)
variable (on_tangent : Circle → Point → Prop)
variable (distance : Point → Point → ℝ)

-- State the theorem
theorem constant_product 
  (circle : Circle)
  (h1 : diameter circle A B)
  (h2 : tangent circle B)
  (h3 : on_circle circle C)
  (h4 : on_tangent circle D)
  : distance A C * distance A D = distance A B * distance A B :=
sorry

end NUMINAMATH_CALUDE_constant_product_l994_99496


namespace NUMINAMATH_CALUDE_intersection_distance_squared_l994_99480

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y + Real.sqrt 3 = 0

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Theorem statement
theorem intersection_distance_squared :
  ∀ (A B M : ℝ × ℝ),
    curve_C A.1 A.2 →
    curve_C B.1 B.2 →
    line_l A.1 A.2 →
    line_l B.1 B.2 →
    line_l M.1 M.2 →
    y_axis M.1 →
    (Real.sqrt ((A.1 - M.1)^2 + (A.2 - M.2)^2) + Real.sqrt ((B.1 - M.1)^2 + (B.2 - M.2)^2))^2 = 16 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_l994_99480


namespace NUMINAMATH_CALUDE_initial_cells_eq_one_l994_99436

/-- Represents the doubling time of the bacteria in minutes -/
def doubling_time : ℕ := 20

/-- Represents the growth time in hours -/
def growth_time : ℕ := 4

/-- Represents the final number of bacterial cells -/
def final_cells : ℕ := 4096

/-- Calculates the number of doublings that occurred during the growth period -/
def num_doublings : ℕ := growth_time * 60 / doubling_time

/-- Represents the initial number of bacterial cells -/
def initial_cells : ℕ := final_cells / (2^num_doublings)

/-- Proves that the initial number of cells was 1 -/
theorem initial_cells_eq_one : initial_cells = 1 := by
  sorry

end NUMINAMATH_CALUDE_initial_cells_eq_one_l994_99436


namespace NUMINAMATH_CALUDE_stating_professor_seating_arrangements_l994_99498

/-- Represents the number of chairs in a row -/
def num_chairs : ℕ := 10

/-- Represents the number of students -/
def num_students : ℕ := 6

/-- Represents the number of professors -/
def num_professors : ℕ := 3

/-- Represents the effective number of chair positions professors can choose from -/
def effective_chairs : ℕ := 4

/-- 
Theorem stating that the number of ways professors can choose their chairs
under the given conditions is 24.
-/
theorem professor_seating_arrangements :
  (effective_chairs.choose num_professors) * num_professors.factorial = 24 :=
by sorry

end NUMINAMATH_CALUDE_stating_professor_seating_arrangements_l994_99498


namespace NUMINAMATH_CALUDE_money_distribution_l994_99402

theorem money_distribution (A B C : ℝ) 
  (total : A + B + C = 450)
  (ac_sum : A + C = 200)
  (bc_sum : B + C = 350) :
  C = 100 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l994_99402


namespace NUMINAMATH_CALUDE_smallest_x_value_exists_solution_l994_99494

theorem smallest_x_value (x : ℝ) : 
  ((5*x - 20)/(4*x - 5))^2 + (5*x - 20)/(4*x - 5) = 20 → x ≥ 0 :=
by sorry

theorem exists_solution : 
  ∃ x : ℝ, ((5*x - 20)/(4*x - 5))^2 + (5*x - 20)/(4*x - 5) = 20 ∧ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_exists_solution_l994_99494


namespace NUMINAMATH_CALUDE_delores_money_left_l994_99460

/-- Calculates the money left after purchases given initial amount and costs --/
def money_left (initial_amount : ℕ) (computer_cost : ℕ) (printer_cost : ℕ) : ℕ :=
  initial_amount - (computer_cost + printer_cost)

theorem delores_money_left :
  money_left 450 400 40 = 10 := by
  sorry

end NUMINAMATH_CALUDE_delores_money_left_l994_99460


namespace NUMINAMATH_CALUDE_luke_fillets_l994_99431

/-- Calculates the total number of fish fillets Luke has after fishing for a given number of days. -/
def total_fillets (fish_per_day : ℕ) (days : ℕ) (fillets_per_fish : ℕ) : ℕ :=
  fish_per_day * days * fillets_per_fish

/-- Proves that Luke has 120 fish fillets after fishing for 30 days. -/
theorem luke_fillets : total_fillets 2 30 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_luke_fillets_l994_99431


namespace NUMINAMATH_CALUDE_water_tank_capacity_l994_99451

theorem water_tank_capacity (c : ℝ) (h1 : c > 0) : 
  (c / 3 : ℝ) / c = 1 / 3 ∧ 
  (c / 3 + 5 : ℝ) / c = 1 / 2 → 
  c = 30 := by
sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l994_99451


namespace NUMINAMATH_CALUDE_remaining_area_in_square_l994_99439

theorem remaining_area_in_square : 
  let large_square_side : ℝ := 3.5
  let small_square_side : ℝ := 2
  let rectangle_length : ℝ := 2
  let rectangle_width : ℝ := 1.5
  let triangle_leg : ℝ := 1
  let large_square_area := large_square_side ^ 2
  let small_square_area := small_square_side ^ 2
  let rectangle_area := rectangle_length * rectangle_width
  let triangle_area := 0.5 * triangle_leg * triangle_leg
  let occupied_area := small_square_area + rectangle_area + triangle_area
  large_square_area - occupied_area = 4.75 := by
sorry

end NUMINAMATH_CALUDE_remaining_area_in_square_l994_99439


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l994_99421

theorem min_value_of_expression (x y : ℝ) : (x * y - 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

theorem min_value_achievable : ∃ x y : ℝ, (x * y - 1)^2 + (x + y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l994_99421


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l994_99495

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def M : Set Int := {-1, 0, 1, 3}
def N : Set Int := {-2, 0, 2, 3}

theorem complement_intersection_theorem :
  (Set.compl M ∩ N) = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l994_99495


namespace NUMINAMATH_CALUDE_bucket_fill_theorem_l994_99478

/-- Given two buckets P and Q, where P has thrice the capacity of Q,
    and P alone takes 60 turns to fill a drum, prove that P and Q together
    take 45 turns to fill the same drum. -/
theorem bucket_fill_theorem (p q : ℕ) (drum : ℕ) : 
  p = 3 * q →  -- Bucket P has thrice the capacity of bucket Q
  60 * p = drum →  -- It takes 60 turns for bucket P to fill the drum
  45 * (p + q) = drum :=  -- It takes 45 turns for both buckets to fill the drum
by sorry

end NUMINAMATH_CALUDE_bucket_fill_theorem_l994_99478


namespace NUMINAMATH_CALUDE_symmetric_complex_product_l994_99468

theorem symmetric_complex_product (z₁ z₂ : ℂ) : 
  (z₁.re = 1 ∧ z₁.im = 1) → 
  (z₂.re = -z₁.re ∧ z₂.im = z₁.im) → 
  z₁ * z₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_product_l994_99468


namespace NUMINAMATH_CALUDE_mr_green_garden_yield_l994_99499

/-- Represents the dimensions and expected yield of a rectangular garden -/
structure Garden where
  length_paces : ℕ
  width_paces : ℕ
  feet_per_pace : ℕ
  yield_per_sqft : ℚ

/-- Calculates the expected potato yield from a garden in pounds -/
def expected_yield (g : Garden) : ℚ :=
  (g.length_paces * g.feet_per_pace) *
  (g.width_paces * g.feet_per_pace) *
  g.yield_per_sqft

/-- Theorem stating the expected yield for Mr. Green's garden -/
theorem mr_green_garden_yield :
  let g : Garden := {
    length_paces := 18,
    width_paces := 25,
    feet_per_pace := 3,
    yield_per_sqft := 3/4
  }
  expected_yield g = 3037.5 := by sorry

end NUMINAMATH_CALUDE_mr_green_garden_yield_l994_99499


namespace NUMINAMATH_CALUDE_unit_square_max_distance_l994_99447

theorem unit_square_max_distance (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  min (min (min (Real.sqrt ((x - 0)^2 + (y - 0)^2))
                (Real.sqrt ((x - 1)^2 + (y - 0)^2)))
           (Real.sqrt ((x - 1)^2 + (y - 1)^2)))
      (Real.sqrt ((x - 0)^2 + (y - 1)^2))
  ≤ Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_unit_square_max_distance_l994_99447


namespace NUMINAMATH_CALUDE_bob_cleaning_time_l994_99491

def alice_time : ℝ := 30

theorem bob_cleaning_time :
  let bob_time := (3 / 4 : ℝ) * alice_time
  bob_time = 22.5 := by sorry

end NUMINAMATH_CALUDE_bob_cleaning_time_l994_99491


namespace NUMINAMATH_CALUDE_parabola_directrix_l994_99424

/-- Given a parabola y² = 2px where p > 0, if a point M(1, m) on the parabola
    is at a distance of 5 from the focus, then the directrix is x = -4 -/
theorem parabola_directrix (p : ℝ) (m : ℝ) (h1 : p > 0) (h2 : m^2 = 2*p) 
  (h3 : (1 - p/2)^2 + m^2 = 5^2) : 
  ∃ (x : ℝ), x = -4 ∧ ∀ (y : ℝ), (x + p/2)^2 = (1 - x)^2 + m^2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l994_99424


namespace NUMINAMATH_CALUDE_max_player_salary_l994_99477

theorem max_player_salary (n : ℕ) (min_salary max_total : ℝ) :
  n = 25 →
  min_salary = 18000 →
  max_total = 1000000 →
  let max_single_salary := max_total - (n - 1) * min_salary
  max_single_salary = 568000 :=
by sorry

end NUMINAMATH_CALUDE_max_player_salary_l994_99477


namespace NUMINAMATH_CALUDE_constant_covered_area_l994_99457

/-- Represents a square in 2D space -/
structure Square where
  side_length : ℝ
  center : ℝ × ℝ

/-- Represents the configuration of two squares as described in the problem -/
structure TwoSquaresConfig where
  bottom_square : Square
  top_square : Square
  rotation_angle : ℝ

/-- Calculates the total area covered by two squares in the given configuration -/
noncomputable def total_covered_area (config : TwoSquaresConfig) : ℝ :=
  sorry

/-- Theorem: The total covered area is constant regardless of the rotation angle -/
theorem constant_covered_area
  (bottom_square : Square)
  (top_square : Square)
  (h_identical : bottom_square.side_length = top_square.side_length)
  (h_diagonal_intersection : top_square.center = (bottom_square.center.1 + bottom_square.side_length / 2, bottom_square.center.2 + bottom_square.side_length / 2)) :
  ∀ θ₁ θ₂ : ℝ,
    total_covered_area { bottom_square := bottom_square, top_square := top_square, rotation_angle := θ₁ } =
    total_covered_area { bottom_square := bottom_square, top_square := top_square, rotation_angle := θ₂ } :=
  sorry

end NUMINAMATH_CALUDE_constant_covered_area_l994_99457


namespace NUMINAMATH_CALUDE_square_of_999999999_has_8_zeros_l994_99427

theorem square_of_999999999_has_8_zeros :
  let n : ℕ := 999999999
  ∃ m : ℕ, n^2 = m * 10^8 ∧ m % 10 ≠ 0 ∧ m ≥ 10^9 ∧ m < 10^10 :=
by sorry

end NUMINAMATH_CALUDE_square_of_999999999_has_8_zeros_l994_99427


namespace NUMINAMATH_CALUDE_square_roots_problem_l994_99479

theorem square_roots_problem (x a : ℝ) (hx : x > 0) :
  ((-a + 2)^2 = x ∧ (2*a - 1)^2 = x) → (a = -1 ∧ x = 9) := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l994_99479


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l994_99443

theorem unique_two_digit_integer (u : ℕ) : 
  (10 ≤ u ∧ u < 100) ∧ (13 * u) % 100 = 52 ↔ u = 4 := by sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l994_99443


namespace NUMINAMATH_CALUDE_intersection_point_in_interval_l994_99464

open Real

theorem intersection_point_in_interval (f g : ℝ → ℝ) (x₀ : ℝ) :
  (∀ x, f x = x^3) →
  (∀ x, g x = 2^x + 1) →
  f x₀ = g x₀ →
  1 < x₀ ∧ x₀ < 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_in_interval_l994_99464


namespace NUMINAMATH_CALUDE_scientific_notation_32000000_l994_99403

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_32000000 :
  toScientificNotation 32000000 = ScientificNotation.mk 3.2 7 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_32000000_l994_99403


namespace NUMINAMATH_CALUDE_transformed_expr_at_one_l994_99449

-- Define the original expression
def original_expr (x : ℚ) : ℚ := (x + 2) / (x - 3)

-- Define the transformed expression
def transformed_expr (x : ℚ) : ℚ := 
  (original_expr x + 2) / (original_expr x - 3)

-- Theorem statement
theorem transformed_expr_at_one :
  transformed_expr 1 = -1/9 := by sorry

end NUMINAMATH_CALUDE_transformed_expr_at_one_l994_99449


namespace NUMINAMATH_CALUDE_new_person_weight_l994_99418

/-- Given a group of 7 people, if replacing one person weighing 95 kg with a new person
    increases the average weight by 12.3 kg, then the weight of the new person is 181.1 kg. -/
theorem new_person_weight (group_size : ℕ) (weight_increase : ℝ) (old_weight : ℝ) :
  group_size = 7 →
  weight_increase = 12.3 →
  old_weight = 95 →
  (group_size : ℝ) * weight_increase + old_weight = 181.1 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l994_99418


namespace NUMINAMATH_CALUDE_factors_of_1320_l994_99413

theorem factors_of_1320 : Finset.card (Nat.divisors 1320) = 24 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_1320_l994_99413


namespace NUMINAMATH_CALUDE_prob_A_and_B_l994_99486

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.75

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.60

/-- The theorem stating that the probability of both A and B occurring is 0.45 -/
theorem prob_A_and_B : prob_A * prob_B = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_and_B_l994_99486


namespace NUMINAMATH_CALUDE_sqrt_a_plus_one_range_l994_99448

theorem sqrt_a_plus_one_range :
  ∀ a : ℝ, (∃ x : ℝ, x^2 = a + 1) ↔ a ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_a_plus_one_range_l994_99448


namespace NUMINAMATH_CALUDE_sales_tax_percentage_l994_99420

theorem sales_tax_percentage (total_allowed : ℝ) (food_cost : ℝ) (tip_percentage : ℝ) :
  total_allowed = 75 →
  food_cost = 61.48 →
  tip_percentage = 15 →
  ∃ (sales_tax_percentage : ℝ),
    sales_tax_percentage ≤ 6.95 ∧
    food_cost * (1 + sales_tax_percentage / 100 + tip_percentage / 100) ≤ total_allowed :=
by sorry

end NUMINAMATH_CALUDE_sales_tax_percentage_l994_99420


namespace NUMINAMATH_CALUDE_max_area_rectangle_d_l994_99430

/-- Given a rectangle divided into four smaller rectangles A, B, C, and D,
    where the perimeters of A, B, and C are known, 
    prove that the maximum possible area of rectangle D is 16 cm². -/
theorem max_area_rectangle_d (perim_A perim_B perim_C : ℝ) 
  (h_perim_A : perim_A = 10)
  (h_perim_B : perim_B = 12)
  (h_perim_C : perim_C = 14) :
  ∃ (area_D : ℝ), area_D ≤ 16 ∧ 
  ∀ (other_area : ℝ), (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2*(a+b) = perim_B + perim_C - perim_A ∧ other_area = a*b) 
  → other_area ≤ area_D := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_d_l994_99430


namespace NUMINAMATH_CALUDE_mileage_reimbursement_rate_calculation_l994_99488

/-- Calculates the mileage reimbursement rate given daily mileages and total reimbursement -/
def mileage_reimbursement_rate (monday_miles tuesday_miles wednesday_miles thursday_miles friday_miles total_reimbursement : ℚ) : ℚ :=
  total_reimbursement / (monday_miles + tuesday_miles + wednesday_miles + thursday_miles + friday_miles)

theorem mileage_reimbursement_rate_calculation 
  (monday_miles tuesday_miles wednesday_miles thursday_miles friday_miles total_reimbursement : ℚ) :
  mileage_reimbursement_rate monday_miles tuesday_miles wednesday_miles thursday_miles friday_miles total_reimbursement =
  total_reimbursement / (monday_miles + tuesday_miles + wednesday_miles + thursday_miles + friday_miles) :=
by sorry

end NUMINAMATH_CALUDE_mileage_reimbursement_rate_calculation_l994_99488


namespace NUMINAMATH_CALUDE_simplify_expression_l994_99450

theorem simplify_expression : (625 : ℝ) ^ (1/4) * (400 : ℝ) ^ (1/2) = 100 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l994_99450


namespace NUMINAMATH_CALUDE_parabola_opens_downwards_l994_99425

/-- A parabola opens downwards if its quadratic coefficient is negative -/
def opens_downwards (a b c : ℝ) : Prop :=
  a < 0

/-- The theorem states that for a = -3, the parabola y = ax^2 + bx + c opens downwards -/
theorem parabola_opens_downwards :
  let a : ℝ := -3
  opens_downwards a b c := by sorry

end NUMINAMATH_CALUDE_parabola_opens_downwards_l994_99425


namespace NUMINAMATH_CALUDE_expression_properties_l994_99472

def expression_result (signs : List Bool) : Int :=
  let nums := List.range 9
  List.foldl (λ acc (n, sign) => if sign then acc + (n + 1) else acc - (n + 1)) 0 (List.zip nums signs)

theorem expression_properties :
  (∀ signs : List Bool, expression_result signs ≠ 0) ∧
  (∃ signs : List Bool, expression_result signs = 1) ∧
  (∀ n : Int, (n % 2 = 1 ∧ -45 ≤ n ∧ n ≤ 45) ↔ ∃ signs : List Bool, expression_result signs = n) := by
  sorry

end NUMINAMATH_CALUDE_expression_properties_l994_99472


namespace NUMINAMATH_CALUDE_oil_price_rollback_l994_99473

def current_price : ℝ := 1.4
def liters_today : ℝ := 10
def liters_friday : ℝ := 25
def total_liters : ℝ := liters_today + liters_friday
def total_spend : ℝ := 39

theorem oil_price_rollback :
  let friday_price := (total_spend - current_price * liters_today) / liters_friday
  current_price - friday_price = 0.4 := by sorry

end NUMINAMATH_CALUDE_oil_price_rollback_l994_99473


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l994_99401

theorem product_of_sum_and_sum_of_cubes (c d : ℝ) 
  (h1 : c + d = 10) 
  (h2 : c^3 + d^3 = 370) : 
  c * d = 21 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l994_99401


namespace NUMINAMATH_CALUDE_path_count_l994_99429

/-- The number of paths between two points -/
def num_paths (start finish : Point) : ℕ := sorry

/-- The set of points in the problem -/
inductive Point
| A
| B
| C
| D

/-- The total number of paths from A to C -/
def total_paths : ℕ := sorry

theorem path_count :
  (num_paths Point.A Point.B = 2) →
  (num_paths Point.B Point.D = 2) →
  (num_paths Point.D Point.C = 2) →
  (num_paths Point.A Point.C = 1 + num_paths Point.A Point.B * num_paths Point.B Point.D * num_paths Point.D Point.C) →
  (total_paths = 9) :=
by sorry

end NUMINAMATH_CALUDE_path_count_l994_99429


namespace NUMINAMATH_CALUDE_fraction_reducibility_implies_determinant_divisibility_l994_99476

theorem fraction_reducibility_implies_determinant_divisibility
  (a b c d l k : ℤ) 
  (h : ∃ (m n : ℤ), a * l + b = k * m ∧ c * l + d = k * n) :
  k ∣ (a * d - b * c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_reducibility_implies_determinant_divisibility_l994_99476


namespace NUMINAMATH_CALUDE_gerbil_weight_difference_gerbil_weight_difference_proof_l994_99438

/-- The weight difference between Scruffy and Muffy given the conditions of the gerbil problem -/
theorem gerbil_weight_difference : ℝ → Prop :=
  fun weight_difference =>
    ∃ (muffy_weight : ℝ),
      let puffy_weight := muffy_weight + 5
      let scruffy_weight := 12
      puffy_weight + muffy_weight = 23 ∧
      weight_difference = scruffy_weight - muffy_weight ∧
      weight_difference = 3

/-- Proof of the gerbil weight difference theorem -/
theorem gerbil_weight_difference_proof : gerbil_weight_difference 3 := by
  sorry

end NUMINAMATH_CALUDE_gerbil_weight_difference_gerbil_weight_difference_proof_l994_99438


namespace NUMINAMATH_CALUDE_maria_has_four_dimes_l994_99445

/-- Represents the number of coins of each type in Maria's piggy bank -/
structure CoinCount where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ

/-- Calculates the total value in cents given a CoinCount -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.dimes * 10 + coins.quarters * 25 + coins.nickels * 5

/-- Theorem stating that Maria has 4 dimes -/
theorem maria_has_four_dimes :
  ∃ (initial : CoinCount),
    initial.quarters = 4 ∧
    initial.nickels = 7 ∧
    totalValue { dimes := initial.dimes,
                 quarters := initial.quarters + 5,
                 nickels := initial.nickels } = 300 ∧
    initial.dimes = 4 := by
  sorry

end NUMINAMATH_CALUDE_maria_has_four_dimes_l994_99445


namespace NUMINAMATH_CALUDE_circle_series_area_sum_l994_99433

/-- The sum of the areas of an infinite series of circles, where the first circle has a radius of 2 inches
    and each subsequent circle's radius is half of the previous one, is equal to 16π/3. -/
theorem circle_series_area_sum : 
  let radius : ℕ → ℝ := fun n => 2 / (2 ^ (n - 1))
  let area : ℕ → ℝ := fun n => π * (radius n)^2
  (∑' n, area n) = 16 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_series_area_sum_l994_99433


namespace NUMINAMATH_CALUDE_junk_items_after_transactions_l994_99483

/-- Represents the composition of items in the attic -/
structure AtticComposition where
  useful : Rat
  valuable : Rat
  junk : Rat

/-- Represents the number of items in each category -/
structure AtticItems where
  useful : ℕ
  valuable : ℕ
  junk : ℕ

/-- The theorem to prove -/
theorem junk_items_after_transactions 
  (initial_composition : AtticComposition)
  (initial_items : AtticItems)
  (items_removed : AtticItems)
  (final_composition : AtticComposition)
  (final_useful_items : ℕ) :
  (initial_composition.useful = 1/5) →
  (initial_composition.valuable = 1/10) →
  (initial_composition.junk = 7/10) →
  (items_removed.useful = 4) →
  (items_removed.valuable = 3) →
  (final_composition.useful = 1/4) →
  (final_composition.valuable = 3/20) →
  (final_composition.junk = 3/5) →
  (final_useful_items = 20) →
  ∃ (final_items : AtticItems), final_items.junk = 48 := by
  sorry

end NUMINAMATH_CALUDE_junk_items_after_transactions_l994_99483


namespace NUMINAMATH_CALUDE_equation_solution_l994_99481

theorem equation_solution :
  ∃! y : ℚ, y + 4/5 = 2/3 + y/6 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_equation_solution_l994_99481


namespace NUMINAMATH_CALUDE_max_min_sum_equals_22_5_l994_99444

/-- Given real numbers x, y, and z satisfying 5(x + y + z) = x^2 + y^2 + z^2,
    the maximum value of xy + xz + yz plus 5 times the minimum value of xy + xz + yz equals 22.5 -/
theorem max_min_sum_equals_22_5 :
  ∃ (N n : ℝ),
    (∀ (x y z : ℝ), 5 * (x + y + z) = x^2 + y^2 + z^2 →
      x * y + x * z + y * z ≤ N ∧
      n ≤ x * y + x * z + y * z) ∧
    (∃ (x y z : ℝ), 5 * (x + y + z) = x^2 + y^2 + z^2 ∧ x * y + x * z + y * z = N) ∧
    (∃ (x y z : ℝ), 5 * (x + y + z) = x^2 + y^2 + z^2 ∧ x * y + x * z + y * z = n) ∧
    N + 5 * n = 22.5 :=
by sorry

end NUMINAMATH_CALUDE_max_min_sum_equals_22_5_l994_99444


namespace NUMINAMATH_CALUDE_inequality_equivalence_l994_99474

theorem inequality_equivalence (x : ℝ) :
  (x^2 - 9) / (x^2 - 4) > 0 ↔ x < -3 ∨ x > 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l994_99474


namespace NUMINAMATH_CALUDE_unique_solution_system_l994_99467

theorem unique_solution_system : 
  ∃! (x y : ℝ), x + y = 3 ∧ x^4 - y^4 = 8*x - y ∧ x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l994_99467


namespace NUMINAMATH_CALUDE_pencil_box_calculation_l994_99411

/-- Given a total number of pencils and pencils per box, calculate the number of filled boxes -/
def filled_boxes (total_pencils : ℕ) (pencils_per_box : ℕ) : ℕ :=
  total_pencils / pencils_per_box

/-- Theorem: Given 648 pencils and 4 pencils per box, the number of filled boxes is 162 -/
theorem pencil_box_calculation :
  filled_boxes 648 4 = 162 := by
  sorry

end NUMINAMATH_CALUDE_pencil_box_calculation_l994_99411


namespace NUMINAMATH_CALUDE_alicia_science_books_l994_99422

/-- Represents the number of science books Alicia bought -/
def science_books : ℕ := sorry

/-- Represents the cost of a math book -/
def math_book_cost : ℕ := 3

/-- Represents the cost of a science book -/
def science_book_cost : ℕ := 3

/-- Represents the cost of an art book -/
def art_book_cost : ℕ := 2

/-- Represents the number of math books Alicia bought -/
def math_books : ℕ := 2

/-- Represents the number of art books Alicia bought -/
def art_books : ℕ := 3

/-- Represents the total cost of all books -/
def total_cost : ℕ := 30

/-- Theorem stating that Alicia bought 6 science books -/
theorem alicia_science_books : 
  math_books * math_book_cost + art_books * art_book_cost + science_books * science_book_cost = total_cost → 
  science_books = 6 := by
  sorry

end NUMINAMATH_CALUDE_alicia_science_books_l994_99422


namespace NUMINAMATH_CALUDE_peach_distribution_l994_99416

/-- Proves that given 60 peaches distributed among two equal-sized containers and one smaller container,
    where the smaller container holds half as many peaches as each of the equal-sized containers,
    the number of peaches in the smaller container is 12. -/
theorem peach_distribution (total_peaches : ℕ) (cloth_bag : ℕ) (knapsack : ℕ) : 
  total_peaches = 60 →
  2 * cloth_bag + knapsack = total_peaches →
  knapsack = cloth_bag / 2 →
  knapsack = 12 := by
sorry

end NUMINAMATH_CALUDE_peach_distribution_l994_99416


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l994_99432

theorem smallest_common_multiple_of_8_and_6 : 
  ∃ (n : ℕ), n > 0 ∧ 8 ∣ n ∧ 6 ∣ n ∧ ∀ (m : ℕ), m > 0 ∧ 8 ∣ m ∧ 6 ∣ m → n ≤ m :=
by
  use 24
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l994_99432


namespace NUMINAMATH_CALUDE_yulia_profit_is_44_l994_99400

/-- Calculates Yulia's profit given her revenues and expenses -/
def yulia_profit (lemonade_revenue babysitting_revenue lemonade_expenses : ℕ) : ℕ :=
  (lemonade_revenue + babysitting_revenue) - lemonade_expenses

/-- Proves that Yulia's profit is $44 given the provided revenues and expenses -/
theorem yulia_profit_is_44 :
  yulia_profit 47 31 34 = 44 := by
  sorry

end NUMINAMATH_CALUDE_yulia_profit_is_44_l994_99400


namespace NUMINAMATH_CALUDE_max_salary_proof_l994_99456

/-- The number of players in a team -/
def team_size : ℕ := 25

/-- The minimum salary for a player -/
def min_salary : ℕ := 15000

/-- The total salary cap for a team -/
def salary_cap : ℕ := 850000

/-- The maximum possible salary for a single player -/
def max_player_salary : ℕ := 490000

theorem max_salary_proof :
  (team_size - 1) * min_salary + max_player_salary = salary_cap ∧
  ∀ (x : ℕ), x > max_player_salary →
    (team_size - 1) * min_salary + x > salary_cap :=
by sorry

end NUMINAMATH_CALUDE_max_salary_proof_l994_99456


namespace NUMINAMATH_CALUDE_circle_arrangement_theorem_l994_99410

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def circleA : Circle := { center := { x := 0, y := -1 }, radius := 1 }
def circleB : Circle := { center := { x := 5, y := 3 }, radius := 3 }
def circleC : Circle := { center := { x := 8, y := -4 }, radius := 4 }

def line_l : Line := { a := 0, b := 1, c := 0 }

def is_below (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c < 0

def is_above (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c > 0

def is_tangent (c : Circle) (l : Line) : Prop :=
  abs (l.a * c.center.x + l.b * c.center.y + l.c) = c.radius * (l.a^2 + l.b^2).sqrt

def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let dx := c1.center.x - c2.center.x
  let dy := c1.center.y - c2.center.y
  (dx^2 + dy^2).sqrt = c1.radius + c2.radius

def triangle_area (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem circle_arrangement_theorem :
  is_below circleA.center line_l ∧
  is_below circleC.center line_l ∧
  is_above circleB.center line_l ∧
  is_tangent circleA line_l ∧
  is_tangent circleB line_l ∧
  is_tangent circleC line_l ∧
  are_externally_tangent circleB circleA ∧
  are_externally_tangent circleB circleC →
  triangle_area circleA.center circleB.center circleC.center = 23.5 := by
  sorry

end NUMINAMATH_CALUDE_circle_arrangement_theorem_l994_99410


namespace NUMINAMATH_CALUDE_battery_factory_robots_l994_99463

/-- The number of robots working simultaneously in a battery factory -/
def num_robots : ℕ :=
  let time_per_battery : ℕ := 15  -- 6 minutes for materials + 9 minutes for creation
  let total_time : ℕ := 300       -- 5 hours * 60 minutes
  let total_batteries : ℕ := 200
  total_batteries * time_per_battery / total_time

theorem battery_factory_robots :
  num_robots = 10 :=
sorry

end NUMINAMATH_CALUDE_battery_factory_robots_l994_99463


namespace NUMINAMATH_CALUDE_cubic_expression_value_l994_99434

theorem cubic_expression_value (m n : ℝ) 
  (h1 : m^2 = n + 2) 
  (h2 : n^2 = m + 2) 
  (h3 : m ≠ n) : 
  m^3 - 2*m*n + n^3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l994_99434


namespace NUMINAMATH_CALUDE_percent_relation_l994_99452

theorem percent_relation (x y z : ℝ) 
  (h1 : 0.45 * z = 0.96 * y) 
  (h2 : y = 0.75 * x) : 
  z = 1.6 * x := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l994_99452


namespace NUMINAMATH_CALUDE_fixed_point_of_function_l994_99475

theorem fixed_point_of_function (n : ℤ) (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => x^n + a^(x-1)
  f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_function_l994_99475


namespace NUMINAMATH_CALUDE_pauls_garage_sale_l994_99497

/-- The number of books Paul sold in the garage sale -/
def books_sold (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : ℕ :=
  initial - given_away - remaining

/-- Proof that Paul sold 27 books in the garage sale -/
theorem pauls_garage_sale : books_sold 134 39 68 = 27 := by
  sorry

end NUMINAMATH_CALUDE_pauls_garage_sale_l994_99497


namespace NUMINAMATH_CALUDE_otimes_inequality_solutions_l994_99414

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a * (a - b) + 1

-- Define the set of non-negative integers
def NonNegIntegers : Set ℤ := {x : ℤ | x ≥ 0}

-- Theorem statement
theorem otimes_inequality_solutions :
  {x ∈ NonNegIntegers | otimes 2 x ≥ 3} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_otimes_inequality_solutions_l994_99414


namespace NUMINAMATH_CALUDE_max_trip_weight_l994_99412

def max_crates : ℕ := 5
def min_crate_weight : ℕ := 1250

theorem max_trip_weight :
  max_crates * min_crate_weight = 6250 :=
by sorry

end NUMINAMATH_CALUDE_max_trip_weight_l994_99412


namespace NUMINAMATH_CALUDE_max_value_is_nine_l994_99404

-- Define the set of possible values
def S : Finset ℕ := {1, 2, 4, 5}

-- Define the expression to be maximized
def f (x y z w : ℕ) : ℤ := x * y - y * z + z * w - w * x

-- Theorem statement
theorem max_value_is_nine :
  ∃ (x y z w : ℕ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ w ∈ S ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
  f x y z w = 9 ∧
  ∀ (a b c d : ℕ), a ∈ S → b ∈ S → c ∈ S → d ∈ S →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  f a b c d ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_value_is_nine_l994_99404


namespace NUMINAMATH_CALUDE_hcf_of_three_numbers_l994_99409

theorem hcf_of_three_numbers (a b c : ℕ+) : 
  (a + b + c : ℝ) = 60 →
  Nat.lcm (a : ℕ) (Nat.lcm b c) = 180 →
  (1 / (a : ℝ) + 1 / (b : ℝ) + 1 / (c : ℝ)) = 11 / 120 →
  (a * b * c : ℕ) = 900 →
  Nat.gcd (a : ℕ) (Nat.gcd b c) = 5 := by
sorry


end NUMINAMATH_CALUDE_hcf_of_three_numbers_l994_99409


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l994_99485

theorem angle_in_fourth_quadrant (θ : Real) 
  (h1 : Real.sin θ < Real.cos θ) 
  (h2 : Real.sin θ * Real.cos θ < 0) : 
  0 < θ ∧ θ < Real.pi / 2 ∧ Real.sin θ < 0 ∧ Real.cos θ > 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l994_99485


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l994_99461

-- Define set A
def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}

-- Define set B
def B : Set ℝ := {x | Real.rpow 2022 x > Real.sqrt 2022}

-- Theorem statement
theorem A_intersect_B_eq_open_interval :
  A ∩ B = Set.Ioo (1/2 : ℝ) 6 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l994_99461


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l994_99482

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, (x - 1) * (x - 2) ≤ 0 → x^2 - 3*x ≤ 0) ∧ 
  (∃ x : ℝ, x^2 - 3*x ≤ 0 ∧ (x - 1) * (x - 2) > 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l994_99482


namespace NUMINAMATH_CALUDE_problem_solution_l994_99453

/-- A geometric sequence with the given property -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

/-- The property of the sequence given in the problem -/
def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n, a n + a (n + 1) = 3 * (1/2)^n

theorem problem_solution (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : sequence_property a) : 
  a 5 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l994_99453


namespace NUMINAMATH_CALUDE_quadratic_function_value_l994_99407

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_value (a b c : ℝ) :
  f a b c 1 = 3 → f a b c 2 = 12 → f a b c 3 = 27 → f a b c 4 = 48 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l994_99407


namespace NUMINAMATH_CALUDE_money_sharing_problem_l994_99446

/-- Represents the ratio of money shared among three people -/
structure MoneyRatio :=
  (a b c : ℕ)

/-- Calculates the total amount of money given a ratio and the first person's share -/
def totalAmount (ratio : MoneyRatio) (firstShare : ℕ) : ℕ :=
  firstShare * (ratio.a + ratio.b + ratio.c)

/-- Theorem: Given a money ratio of 1:2:7 and the first person's share of $20, 
    the total amount shared is $200 -/
theorem money_sharing_problem (ratio : MoneyRatio) (firstShare : ℕ) :
  ratio.a = 1 → ratio.b = 2 → ratio.c = 7 → firstShare = 20 →
  totalAmount ratio firstShare = 200 := by
  sorry

#eval totalAmount ⟨1, 2, 7⟩ 20

end NUMINAMATH_CALUDE_money_sharing_problem_l994_99446


namespace NUMINAMATH_CALUDE_train_journey_theorem_l994_99484

/-- Represents the properties of a train journey -/
structure TrainJourney where
  reducedSpeed : ℝ  -- Speed at which the train actually travels
  speedFraction : ℝ  -- Fraction of the train's own speed at which it travels
  time : ℝ  -- Time taken for the journey
  distance : ℝ  -- Distance traveled

/-- The problem setup -/
def trainProblem (trainA trainB : TrainJourney) : Prop :=
  trainA.speedFraction = 2/3 ∧
  trainA.time = 12 ∧
  trainA.distance = 360 ∧
  trainB.speedFraction = 1/2 ∧
  trainB.time = 8 ∧
  trainB.distance = trainA.distance

/-- The theorem to be proved -/
theorem train_journey_theorem (trainA trainB : TrainJourney) 
  (h : trainProblem trainA trainB) : 
  (trainA.time * (1 - trainA.speedFraction) + trainB.time * (1 - trainB.speedFraction) = 8) ∧
  (trainB.distance = 360) := by
  sorry

end NUMINAMATH_CALUDE_train_journey_theorem_l994_99484


namespace NUMINAMATH_CALUDE_eight_digit_increasing_remainder_l994_99406

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) n

/-- The number of 8-digit positive integers with digits in increasing order -/
def M : ℕ := 9 * stars_and_bars 7 10

theorem eight_digit_increasing_remainder :
  M % 1000 = 960 := by sorry

end NUMINAMATH_CALUDE_eight_digit_increasing_remainder_l994_99406


namespace NUMINAMATH_CALUDE_basketball_conference_games_l994_99423

/-- The number of teams in the basketball conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team in the conference -/
def games_per_pair : ℕ := 3

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 2

/-- The total number of games in a season for the basketball conference -/
def total_games : ℕ := (num_teams.choose 2 * games_per_pair) + (num_teams * non_conference_games)

theorem basketball_conference_games :
  total_games = 155 := by sorry

end NUMINAMATH_CALUDE_basketball_conference_games_l994_99423


namespace NUMINAMATH_CALUDE_pages_per_day_l994_99458

/-- Given a book with 576 pages read over 72 days, prove that the number of pages read per day is 8 -/
theorem pages_per_day (total_pages : ℕ) (total_days : ℕ) (h1 : total_pages = 576) (h2 : total_days = 72) :
  total_pages / total_days = 8 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_day_l994_99458


namespace NUMINAMATH_CALUDE_symmetric_points_line_equation_l994_99490

/-- Given two points are symmetric about a line, prove the equation of the line -/
theorem symmetric_points_line_equation (O A : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  O = (0, 0) →
  A = (-4, 2) →
  (∀ p : ℝ × ℝ, p ∈ l ↔ (p.1 - O.1) * (A.1 - O.1) + (p.2 - O.2) * (A.2 - O.2) = 0) →
  (∀ x y : ℝ, (x, y) ∈ l ↔ 2*x - y + 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_line_equation_l994_99490


namespace NUMINAMATH_CALUDE_remaining_income_calculation_l994_99440

def remaining_income (food_percent : ℝ) (education_percent : ℝ) (rent_percent : ℝ) 
  (utilities_percent : ℝ) (transportation_percent : ℝ) (insurance_percent : ℝ) 
  (emergency_fund_percent : ℝ) : ℝ :=
  let initial_remaining := 1 - (food_percent + education_percent + transportation_percent)
  let rent_amount := rent_percent * initial_remaining
  let post_rent_remaining := initial_remaining - rent_amount
  let utilities_amount := utilities_percent * rent_amount
  let post_utilities_remaining := post_rent_remaining - utilities_amount
  let insurance_amount := insurance_percent * post_utilities_remaining
  let pre_emergency_remaining := post_utilities_remaining - insurance_amount
  let emergency_fund_amount := emergency_fund_percent * pre_emergency_remaining
  pre_emergency_remaining - emergency_fund_amount

theorem remaining_income_calculation :
  remaining_income 0.42 0.18 0.30 0.25 0.12 0.15 0.06 = 0.139825 := by
  sorry

#eval remaining_income 0.42 0.18 0.30 0.25 0.12 0.15 0.06

end NUMINAMATH_CALUDE_remaining_income_calculation_l994_99440


namespace NUMINAMATH_CALUDE_original_mango_price_l994_99465

/-- Represents the price increase rate -/
def price_increase_rate : ℝ := 0.15

/-- Represents the original price of an orange -/
def original_orange_price : ℝ := 40

/-- Represents the total cost of 10 oranges and 10 mangoes after price increase -/
def total_cost : ℝ := 1035

/-- Represents the quantity of each fruit -/
def quantity : ℕ := 10

/-- Calculates the new price after applying the price increase -/
def new_price (original_price : ℝ) : ℝ :=
  original_price * (1 + price_increase_rate)

/-- Theorem stating that the original price of a mango was $50 -/
theorem original_mango_price :
  ∃ (original_mango_price : ℝ),
    original_mango_price = 50 ∧
    (quantity : ℝ) * new_price original_orange_price +
    (quantity : ℝ) * new_price original_mango_price = total_cost := by
  sorry

end NUMINAMATH_CALUDE_original_mango_price_l994_99465


namespace NUMINAMATH_CALUDE_line_AB_not_through_point_B_l994_99493

-- Define the circles C and M
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4
def circle_M (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = a^2 + b^2

-- Define the condition that (a, b) is on circle C
def M_on_C (a b : ℝ) : Prop := circle_C a b

-- Define the line AB
def line_AB (a b x y : ℝ) : Prop := (2*a - 2)*x + 2*b*y - 3 = 0

-- Theorem statement
theorem line_AB_not_through_point_B (a b : ℝ) (h : M_on_C a b) :
  ¬ line_AB a b (1/2) (1/2) :=
sorry

end NUMINAMATH_CALUDE_line_AB_not_through_point_B_l994_99493


namespace NUMINAMATH_CALUDE_truncated_hexahedron_property_l994_99455

-- Define the structure of our polyhedron
structure Polyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  H : ℕ  -- number of hexagonal faces
  T : ℕ  -- number of triangular faces

-- Define the properties of our specific polyhedron
def truncated_hexahedron : Polyhedron where
  V := 20
  E := 36
  F := 18
  H := 6
  T := 12

-- Theorem statement
theorem truncated_hexahedron_property (p : Polyhedron) 
  (euler : p.V - p.E + p.F = 2)
  (faces : p.F = 18)
  (hex_tri : p.H + p.T = p.F)
  (vertex_config : 2 * p.V = 3 * p.T + 6 * p.H) :
  100 * 2 + 10 * 2 + p.V = 240 := by
  sorry

#check truncated_hexahedron_property

end NUMINAMATH_CALUDE_truncated_hexahedron_property_l994_99455


namespace NUMINAMATH_CALUDE_red_bellied_minnows_count_l994_99470

/-- Represents the number of minnows in a pond with different belly colors. -/
structure MinnowPond where
  total : ℕ
  red_percent : ℚ
  green_percent : ℚ
  white_count : ℕ

/-- Theorem stating the number of red-bellied minnows in the pond. -/
theorem red_bellied_minnows_count (pond : MinnowPond)
  (h1 : pond.red_percent = 2/5)
  (h2 : pond.green_percent = 3/10)
  (h3 : pond.white_count = 15)
  (h4 : pond.total * (1 - pond.red_percent - pond.green_percent) = pond.white_count) :
  pond.total * pond.red_percent = 20 := by
  sorry

end NUMINAMATH_CALUDE_red_bellied_minnows_count_l994_99470


namespace NUMINAMATH_CALUDE_mail_order_cost_l994_99471

/-- The total cost of mail ordering books with a shipping fee -/
def total_cost (unit_price : ℝ) (shipping_rate : ℝ) (num_books : ℝ) : ℝ :=
  unit_price * num_books * (1 + shipping_rate)

/-- Theorem: The total cost of mail ordering 'a' books with a unit price of 8 yuan and a 10% shipping fee is 8(1+10%)a yuan -/
theorem mail_order_cost (a : ℝ) : 
  total_cost 8 0.1 a = 8 * (1 + 0.1) * a := by
  sorry

end NUMINAMATH_CALUDE_mail_order_cost_l994_99471


namespace NUMINAMATH_CALUDE_tom_time_ratio_l994_99462

/-- The duration of the BS program in years -/
def bs_duration : ℕ := 3

/-- The duration of the Ph.D. program in years -/
def phd_duration : ℕ := 5

/-- Tom's total time to complete both programs in years -/
def tom_total_time : ℕ := 6

/-- The normal time to complete both programs -/
def normal_time : ℕ := bs_duration + phd_duration

theorem tom_time_ratio :
  (tom_total_time : ℚ) / (normal_time : ℚ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tom_time_ratio_l994_99462


namespace NUMINAMATH_CALUDE_deepak_current_age_l994_99408

/-- Represents the ages of Rahul and Deepak -/
structure Ages where
  rahul : ℕ
  deepak : ℕ

/-- The ratio between Rahul and Deepak's ages -/
def age_ratio (ages : Ages) : ℚ :=
  ages.rahul / ages.deepak

/-- Rahul's age after 6 years -/
def rahul_future_age (ages : Ages) : ℕ :=
  ages.rahul + 6

theorem deepak_current_age (ages : Ages) :
  age_ratio ages = 4/3 →
  rahul_future_age ages = 42 →
  ages.deepak = 27 := by
sorry

end NUMINAMATH_CALUDE_deepak_current_age_l994_99408
