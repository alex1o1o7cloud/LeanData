import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_degree_l95_9503

/-- The degree of the polynomial (3x^4+4x^3+2x-7)(3x^10-9x^7+9x^4+30)-(x^2+5)^8 is 16 -/
theorem polynomial_degree : 
  let p₁ : Polynomial ℝ := 3 * X^4 + 4 * X^3 + 2 * X - 7
  let p₂ : Polynomial ℝ := 3 * X^10 - 9 * X^7 + 9 * X^4 + 30
  let p₃ : Polynomial ℝ := (X^2 + 5)^8
  (p₁ * p₂ - p₃).degree = 16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_l95_9503


namespace NUMINAMATH_CALUDE_apollo_total_cost_l95_9559

/-- Represents the cost structure for a blacksmith --/
structure BlacksmithCost where
  monthly_rates : List ℕ
  installation_fee : ℕ
  installation_frequency : ℕ

/-- Calculates the total cost for a blacksmith for a year --/
def calculate_blacksmith_cost (cost : BlacksmithCost) : ℕ :=
  (cost.monthly_rates.sum) + 
  (12 / cost.installation_frequency * cost.installation_fee)

/-- Hephaestus's cost structure --/
def hephaestus_cost : BlacksmithCost := {
  monthly_rates := [3, 3, 3, 3, 6, 6, 6, 6, 9, 9, 9, 9],
  installation_fee := 2,
  installation_frequency := 1
}

/-- Athena's cost structure --/
def athena_cost : BlacksmithCost := {
  monthly_rates := [5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7],
  installation_fee := 10,
  installation_frequency := 12
}

/-- Ares's cost structure --/
def ares_cost : BlacksmithCost := {
  monthly_rates := [4, 4, 4, 6, 6, 6, 6, 6, 6, 8, 8, 8],
  installation_fee := 3,
  installation_frequency := 3
}

/-- The total cost for Apollo's chariot wheels for a year --/
theorem apollo_total_cost : 
  calculate_blacksmith_cost hephaestus_cost + 
  calculate_blacksmith_cost athena_cost + 
  calculate_blacksmith_cost ares_cost = 265 := by
  sorry

end NUMINAMATH_CALUDE_apollo_total_cost_l95_9559


namespace NUMINAMATH_CALUDE_line_properties_l95_9505

-- Define the line equation
def line_equation (x y m : ℝ) : Prop := x - m * y + 2 = 0

-- Theorem statement
theorem line_properties (m : ℝ) :
  (∀ y, line_equation (-2) y m) ∧
  (∃ x, x ≠ 0 ∧ line_equation x 0 m) :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l95_9505


namespace NUMINAMATH_CALUDE_shifted_line_not_in_third_quadrant_l95_9596

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shifts a line horizontally -/
def shift_line (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.slope * shift + l.intercept }

/-- Checks if a line passes through the third quadrant -/
def passes_through_third_quadrant (l : Line) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ y = l.slope * x + l.intercept

/-- The original line y = -2x - 1 -/
def original_line : Line :=
  { slope := -2, intercept := -1 }

/-- The amount of right shift -/
def shift_amount : ℝ := 3

theorem shifted_line_not_in_third_quadrant :
  ¬ passes_through_third_quadrant (shift_line original_line shift_amount) := by
  sorry

end NUMINAMATH_CALUDE_shifted_line_not_in_third_quadrant_l95_9596


namespace NUMINAMATH_CALUDE_min_angular_frequency_l95_9591

/-- Given a cosine function with specific properties, prove that the minimum angular frequency is 2 -/
theorem min_angular_frequency (ω φ : ℝ) : 
  ω > 0 → 
  (∃ k : ℤ, ω * (π / 3) + φ = k * π) →
  1/2 * Real.cos (ω * (π / 12) + φ) + 1 = 1 →
  (∀ ω' > 0, 
    (∃ k : ℤ, ω' * (π / 3) + φ = k * π) →
    1/2 * Real.cos (ω' * (π / 12) + φ) + 1 = 1 →
    ω' ≥ ω) →
  ω = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_angular_frequency_l95_9591


namespace NUMINAMATH_CALUDE_train_length_proof_l95_9548

/-- Proves that a train with given speed passing a platform of known length in a certain time has a specific length -/
theorem train_length_proof (train_speed : ℝ) (platform_length : ℝ) (passing_time : ℝ) :
  train_speed = 45 * 1000 / 3600 →
  platform_length = 240 →
  passing_time = 48 →
  train_speed * passing_time - platform_length = 360 :=
by sorry

end NUMINAMATH_CALUDE_train_length_proof_l95_9548


namespace NUMINAMATH_CALUDE_range_of_m_l95_9521

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x > 1 → (x^2 + 3) / (x - 1) > m^2 + 1) → 
  -Real.sqrt 5 < m ∧ m < Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l95_9521


namespace NUMINAMATH_CALUDE_tangent_angle_range_l95_9527

open Real

noncomputable def curve (x : ℝ) : ℝ := 4 / (exp x + 1)

theorem tangent_angle_range :
  ∀ (x : ℝ), 
  let y := curve x
  let α := Real.arctan (deriv curve x)
  3 * π / 4 ≤ α ∧ α < π :=
by sorry

end NUMINAMATH_CALUDE_tangent_angle_range_l95_9527


namespace NUMINAMATH_CALUDE_solutions_for_20_l95_9561

/-- The number of distinct integer solutions (x, y) for |x| + |y| = n -/
def num_solutions (n : ℕ) : ℕ := sorry

/-- The theorem stating the number of solutions for n = 20 -/
theorem solutions_for_20 :
  num_solutions 1 = 4 →
  num_solutions 2 = 8 →
  num_solutions 3 = 12 →
  num_solutions 20 = 80 := by sorry

end NUMINAMATH_CALUDE_solutions_for_20_l95_9561


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l95_9568

theorem min_value_squared_sum (a b t p : ℝ) (h1 : a + b = t) (h2 : a * b = p) :
  a^2 + a*b + b^2 ≥ (3/4) * t^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l95_9568


namespace NUMINAMATH_CALUDE_circle_line_intersection_k_range_l95_9555

/-- Given a circle and a line, if there exists a point on the line such that a circle 
    with this point as its center and radius 1 has a common point with the given circle, 
    then k is within a specific range. -/
theorem circle_line_intersection_k_range :
  ∀ (k : ℝ),
  (∃ (x y : ℝ), x^2 + y^2 + 4*x + 3 = 0 ∧ y = k*x - 1 ∧
   ∃ (x₀ y₀ : ℝ), y₀ = k*x₀ - 1 ∧ 
   ∃ (x₁ y₁ : ℝ), (x₁ - x₀)^2 + (y₁ - y₀)^2 = 1 ∧ x₁^2 + y₁^2 + 4*x₁ + 3 = 0) →
  -4/3 ≤ k ∧ k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_k_range_l95_9555


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_R_l95_9536

/-- The solution set of a quadratic inequality is R iff a < 0 and discriminant < 0 -/
theorem quadratic_inequality_solution_set_R 
  (a b c : ℝ) (h : a ≠ 0) : 
  (∀ x, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_R_l95_9536


namespace NUMINAMATH_CALUDE_cosine_is_even_l95_9577

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem cosine_is_even : IsEven Real.cos := by
  sorry

end NUMINAMATH_CALUDE_cosine_is_even_l95_9577


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l95_9549

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the different relation for planes and lines
variable (different : Plane → Plane → Prop)
variable (different_line : Line → Line → Prop)

-- State the theorem
theorem perpendicular_transitivity
  (α β γ : Plane) (m n l : Line)
  (h_diff_planes : different α β ∧ different α γ ∧ different β γ)
  (h_diff_lines : different_line m n ∧ different_line m l ∧ different_line n l)
  (h_n_perp_α : perp n α)
  (h_n_perp_β : perp n β)
  (h_m_perp_α : perp m α) :
  perp m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l95_9549


namespace NUMINAMATH_CALUDE_parking_arrangement_l95_9597

/-- The number of ways to park cars in a row with empty spaces -/
def park_cars (total_spaces : ℕ) (cars : ℕ) (empty_spaces : ℕ) : ℕ :=
  (total_spaces - empty_spaces + 1) * (cars.factorial)

theorem parking_arrangement :
  park_cars 8 4 4 = 120 :=
by sorry

end NUMINAMATH_CALUDE_parking_arrangement_l95_9597


namespace NUMINAMATH_CALUDE_complement_union_theorem_l95_9557

-- Define the universal set U
def U : Finset ℕ := {0, 1, 2, 3, 4}

-- Define set M
def M : Finset ℕ := {1, 2, 4}

-- Define set N
def N : Finset ℕ := {2, 3}

-- Theorem statement
theorem complement_union_theorem : 
  (U \ M) ∪ N = {0, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l95_9557


namespace NUMINAMATH_CALUDE_neg_cos_double_angle_range_l95_9522

theorem neg_cos_double_angle_range (θ : Real) : -1 ≤ -Real.cos (2 * θ) ∧ -Real.cos (2 * θ) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_neg_cos_double_angle_range_l95_9522


namespace NUMINAMATH_CALUDE_max_intersection_points_circle_rectangle_l95_9532

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A rectangle in a plane -/
structure Rectangle where
  corners : Fin 4 → ℝ × ℝ

/-- The number of intersection points between a circle and a line segment -/
def intersectionPointsCircleLine (c : Circle) (p1 p2 : ℝ × ℝ) : ℕ := sorry

/-- The maximum number of intersection points between a circle and a rectangle -/
def maxIntersectionPoints (c : Circle) (r : Rectangle) : ℕ :=
  (intersectionPointsCircleLine c (r.corners 0) (r.corners 1)) +
  (intersectionPointsCircleLine c (r.corners 1) (r.corners 2)) +
  (intersectionPointsCircleLine c (r.corners 2) (r.corners 3)) +
  (intersectionPointsCircleLine c (r.corners 3) (r.corners 0))

/-- Theorem: The maximum number of intersection points between a circle and a rectangle is 8 -/
theorem max_intersection_points_circle_rectangle :
  ∀ (c : Circle) (r : Rectangle), maxIntersectionPoints c r ≤ 8 ∧
  ∃ (c : Circle) (r : Rectangle), maxIntersectionPoints c r = 8 :=
sorry

end NUMINAMATH_CALUDE_max_intersection_points_circle_rectangle_l95_9532


namespace NUMINAMATH_CALUDE_involutive_function_theorem_l95_9510

/-- A function f is involutive if f(f(x)) = x for all x in its domain -/
def Involutive (f : ℝ → ℝ) : Prop :=
  ∀ x, f (f x) = x

/-- The main theorem -/
theorem involutive_function_theorem (a b c d : ℝ) 
    (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) :
    let f := fun x => (2 * a * x + b) / (3 * c * x + 2 * d)
    Involutive f → 2 * a + 2 * d = 0 := by
  sorry


end NUMINAMATH_CALUDE_involutive_function_theorem_l95_9510


namespace NUMINAMATH_CALUDE_bill_difference_l95_9570

/-- The number of $20 bills Mandy has -/
def mandy_twenty_bills : ℕ := 3

/-- The number of $50 bills Manny has -/
def manny_fifty_bills : ℕ := 2

/-- The value of a $20 bill -/
def twenty_bill_value : ℕ := 20

/-- The value of a $50 bill -/
def fifty_bill_value : ℕ := 50

/-- The value of a $10 bill -/
def ten_bill_value : ℕ := 10

/-- Theorem stating the difference in $10 bills between Manny and Mandy -/
theorem bill_difference :
  (manny_fifty_bills * fifty_bill_value) / ten_bill_value -
  (mandy_twenty_bills * twenty_bill_value) / ten_bill_value = 4 := by
  sorry

end NUMINAMATH_CALUDE_bill_difference_l95_9570


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l95_9592

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l95_9592


namespace NUMINAMATH_CALUDE_minimum_buses_required_l95_9551

theorem minimum_buses_required (total_students : ℕ) (bus_capacity : ℕ) (h1 : total_students = 325) (h2 : bus_capacity = 45) : 
  ∃ (n : ℕ), n * bus_capacity ≥ total_students ∧ ∀ (m : ℕ), m * bus_capacity ≥ total_students → m ≥ n ∧ n = 8 :=
by sorry

end NUMINAMATH_CALUDE_minimum_buses_required_l95_9551


namespace NUMINAMATH_CALUDE_path_count_theorem_l95_9560

/-- The number of paths on a grid from point C to point D, where D is 6 units right and 2 units up from C, and the path consists of exactly 8 steps. -/
def number_of_paths : ℕ := 28

/-- The horizontal distance between points C and D on the grid. -/
def horizontal_distance : ℕ := 6

/-- The vertical distance between points C and D on the grid. -/
def vertical_distance : ℕ := 2

/-- The total number of steps in the path. -/
def total_steps : ℕ := 8

theorem path_count_theorem :
  number_of_paths = Nat.choose total_steps vertical_distance :=
by sorry

end NUMINAMATH_CALUDE_path_count_theorem_l95_9560


namespace NUMINAMATH_CALUDE_sum_of_first_15_natural_numbers_mod_11_l95_9508

theorem sum_of_first_15_natural_numbers_mod_11 :
  (List.range 16).sum % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_15_natural_numbers_mod_11_l95_9508


namespace NUMINAMATH_CALUDE_num_ways_to_sum_eq_two_pow_n_minus_one_l95_9593

/-- The number of ways to express a natural number as a sum of one or more natural numbers, considering the order of the terms. -/
def numWaysToSum (n : ℕ) : ℕ := 2^(n-1)

/-- Theorem: For any natural number n, the number of ways to express n as a sum of one or more natural numbers, considering the order of the terms, is equal to 2^(n-1). -/
theorem num_ways_to_sum_eq_two_pow_n_minus_one (n : ℕ) : 
  numWaysToSum n = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_num_ways_to_sum_eq_two_pow_n_minus_one_l95_9593


namespace NUMINAMATH_CALUDE_farther_from_theorem_l95_9588

theorem farther_from_theorem :
  -- Part 1
  ∀ x : ℝ, |x^2 - 1| > 1 ↔ x < -Real.sqrt 2 ∨ x > Real.sqrt 2

  -- Part 2
  ∧ ∀ a b : ℝ, a > 0 → b > 0 → a ≠ b →
    |a^3 + b^3 - (a^2*b + a*b^2)| > |2*a*b*Real.sqrt (a*b) - (a^2*b + a*b^2)| :=
by sorry

end NUMINAMATH_CALUDE_farther_from_theorem_l95_9588


namespace NUMINAMATH_CALUDE_max_value_of_expression_l95_9511

/-- The set of digits to be used -/
def Digits : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The expression to be maximized -/
def expression (a b c d e f : ℕ) : ℚ :=
  a / b + c / d + e / f

/-- The theorem stating the maximum value of the expression -/
theorem max_value_of_expression :
  ∃ (a b c d e f : ℕ),
    a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧ e ∈ Digits ∧ f ∈ Digits ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    expression a b c d e f = 59 / 6 ∧
    ∀ (x y z w u v : ℕ),
      x ∈ Digits → y ∈ Digits → z ∈ Digits → w ∈ Digits → u ∈ Digits → v ∈ Digits →
      x ≠ y → x ≠ z → x ≠ w → x ≠ u → x ≠ v →
      y ≠ z → y ≠ w → y ≠ u → y ≠ v →
      z ≠ w → z ≠ u → z ≠ v →
      w ≠ u → w ≠ v →
      u ≠ v →
      expression x y z w u v ≤ 59 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l95_9511


namespace NUMINAMATH_CALUDE_circle_area_above_line_l95_9517

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 18*y + 61 = 0

-- Define the line equation
def line_equation (y : ℝ) : Prop :=
  y = 4

-- Theorem statement
theorem circle_area_above_line : 
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    (center_y > 4) ∧
    (center_y - radius > 4) ∧
    (radius = 1) ∧
    (Real.pi * radius^2 = Real.pi) :=
sorry

end NUMINAMATH_CALUDE_circle_area_above_line_l95_9517


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l95_9524

theorem complex_sum_theorem (A B C D : ℂ) : 
  A = 3 + 2*I → B = -5 → C = 1 - 2*I → D = 3 + 5*I → 
  A + B + C + D = 2 + 5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l95_9524


namespace NUMINAMATH_CALUDE_expression_value_l95_9585

theorem expression_value (x y z : ℝ) 
  (eq1 : 4*x - 6*y - 2*z = 0)
  (eq2 : x + 2*y - 10*z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 - x*y) / (y^2 + z^2) = 26/25 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l95_9585


namespace NUMINAMATH_CALUDE_planting_methods_result_l95_9562

/-- The number of rows in the field -/
def total_rows : ℕ := 10

/-- The minimum required interval between crops A and B -/
def min_interval : ℕ := 6

/-- The number of crops to be planted -/
def num_crops : ℕ := 2

/-- Calculates the number of ways to plant two crops with the given constraints -/
def planting_methods (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  -- n: total rows
  -- k: number of crops
  -- m: minimum interval
  sorry

theorem planting_methods_result : planting_methods total_rows num_crops min_interval = 12 := by
  sorry

end NUMINAMATH_CALUDE_planting_methods_result_l95_9562


namespace NUMINAMATH_CALUDE_lcm_problem_l95_9554

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 36 m = 180) (h2 : Nat.lcm m 50 = 300) : m = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l95_9554


namespace NUMINAMATH_CALUDE_surface_area_unchanged_surface_area_4x4x4_with_corners_removed_l95_9543

/-- Represents a cube with given side length -/
structure Cube where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℝ := 6 * c.side_length ^ 2

/-- Represents the process of removing corner cubes from a larger cube -/
structure CornerCubeRemoval where
  original_cube : Cube
  corner_cube : Cube
  corner_cube_fits : corner_cube.side_length ≤ original_cube.side_length / 2

/-- Theorem stating that removing corner cubes does not change the surface area -/
theorem surface_area_unchanged (removal : CornerCubeRemoval) :
  surface_area removal.original_cube = surface_area
    { side_length := removal.original_cube.side_length,
      side_length_pos := removal.original_cube.side_length_pos } := by
  sorry

/-- The main theorem proving that a 4x4x4 cube with 2x2x2 corner cubes removed has the same surface area -/
theorem surface_area_4x4x4_with_corners_removed :
  let original_cube : Cube := ⟨4, by norm_num⟩
  let corner_cube : Cube := ⟨2, by norm_num⟩
  let removal : CornerCubeRemoval := ⟨original_cube, corner_cube, by norm_num⟩
  surface_area original_cube = 96 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_surface_area_4x4x4_with_corners_removed_l95_9543


namespace NUMINAMATH_CALUDE_monday_water_usage_l95_9507

/-- Represents the relationship between rainfall and water usage -/
structure RainfallWaterUsage where
  rainfall : ℝ
  water_used : ℝ

/-- The constant of inverse proportionality between rainfall and water usage -/
def inverse_proportionality_constant (day : RainfallWaterUsage) : ℝ :=
  day.rainfall * day.water_used

theorem monday_water_usage 
  (sunday : RainfallWaterUsage)
  (monday_rainfall : ℝ)
  (h_sunday_rainfall : sunday.rainfall = 3)
  (h_sunday_water : sunday.water_used = 10)
  (h_monday_rainfall : monday_rainfall = 5)
  (h_inverse_prop : ∀ (day1 day2 : RainfallWaterUsage), 
    inverse_proportionality_constant day1 = inverse_proportionality_constant day2) :
  ∃ (monday : RainfallWaterUsage), 
    monday.rainfall = monday_rainfall ∧ 
    monday.water_used = 6 :=
sorry

end NUMINAMATH_CALUDE_monday_water_usage_l95_9507


namespace NUMINAMATH_CALUDE_inequality_solution_set_l95_9553

theorem inequality_solution_set (x : ℝ) : 
  (2 * x - 4 < 6) ↔ (x < 5) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l95_9553


namespace NUMINAMATH_CALUDE_sat_markings_count_l95_9594

/-- The number of ways to mark a single question on the SAT answer sheet -/
def markings_per_question : ℕ := 32

/-- The number of questions to be marked -/
def num_questions : ℕ := 10

/-- Function to calculate the number of valid sequences of length n with no consecutive 1s -/
def f : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n + 2) => f (n + 1) + f n

/-- The number of letters in the SAT answer sheet -/
def num_letters : ℕ := 5

/-- Theorem stating the total number of ways to mark the SAT answer sheet -/
theorem sat_markings_count :
  (f num_questions) ^ num_letters = 2^20 * 3^10 := by sorry

end NUMINAMATH_CALUDE_sat_markings_count_l95_9594


namespace NUMINAMATH_CALUDE_abs_sum_minimum_l95_9537

theorem abs_sum_minimum (x : ℝ) : 
  |x + 1| + |x + 3| + |x + 6| ≥ 5 ∧ ∃ y : ℝ, |y + 1| + |y + 3| + |y + 6| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_minimum_l95_9537


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_segment_ratio_l95_9558

theorem right_triangle_hypotenuse_segment_ratio 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_ratio : a / b = 3 / 4) 
  (d : ℝ) 
  (h_d : d * c = a * b) : 
  (c - d) / d = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_segment_ratio_l95_9558


namespace NUMINAMATH_CALUDE_arithmetic_triangle_b_range_l95_9599

/-- A triangle with side lengths forming an arithmetic sequence --/
structure ArithmeticTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_arithmetic : ∃ d : ℝ, a = b - d ∧ c = b + d
  sum_of_squares : a^2 + b^2 + c^2 = 21
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The range of possible values for the middle term of the arithmetic sequence --/
theorem arithmetic_triangle_b_range (t : ArithmeticTriangle) :
  t.b ∈ Set.Ioo (Real.sqrt 6) (Real.sqrt 7) ∪ {Real.sqrt 7} :=
sorry

end NUMINAMATH_CALUDE_arithmetic_triangle_b_range_l95_9599


namespace NUMINAMATH_CALUDE_probability_four_ones_in_five_rolls_l95_9528

-- Define a fair six-sided die
def fair_six_sided_die : Finset Nat := Finset.range 6

-- Define the probability of rolling a 1
def prob_roll_one : ℚ := 1 / 6

-- Define the number of rolls
def num_rolls : Nat := 5

-- Define the number of times we want to see 1
def target_ones : Nat := 4

-- Theorem statement
theorem probability_four_ones_in_five_rolls :
  (Nat.choose num_rolls target_ones : ℚ) * prob_roll_one^target_ones * (1 - prob_roll_one)^(num_rolls - target_ones) = 25 / 7776 :=
sorry

end NUMINAMATH_CALUDE_probability_four_ones_in_five_rolls_l95_9528


namespace NUMINAMATH_CALUDE_largest_common_divisor_36_60_l95_9565

theorem largest_common_divisor_36_60 : 
  ∃ (n : ℕ), n > 0 ∧ n ∣ 36 ∧ n ∣ 60 ∧ ∀ (m : ℕ), m > 0 ∧ m ∣ 36 ∧ m ∣ 60 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_36_60_l95_9565


namespace NUMINAMATH_CALUDE_sam_initial_dimes_l95_9582

/-- Represents the number of cents in a coin -/
def cents_in_coin (coin : String) : ℕ :=
  match coin with
  | "dime" => 10
  | "quarter" => 25
  | _ => 0

/-- Represents Sam's initial coin counts and purchases -/
structure SamsPurchase where
  initial_quarters : ℕ
  candy_bars : ℕ
  candy_bar_price : ℕ
  lollipops : ℕ
  lollipop_price : ℕ
  cents_left : ℕ

/-- Theorem stating that Sam had 19 dimes initially -/
theorem sam_initial_dimes (purchase : SamsPurchase)
  (h1 : purchase.initial_quarters = 6)
  (h2 : purchase.candy_bars = 4)
  (h3 : purchase.candy_bar_price = 3)
  (h4 : purchase.lollipops = 1)
  (h5 : purchase.lollipop_price = 1)
  (h6 : purchase.cents_left = 195) :
  (purchase.cents_left +
   purchase.candy_bars * purchase.candy_bar_price * cents_in_coin "dime" +
   purchase.lollipops * cents_in_coin "quarter" -
   purchase.initial_quarters * cents_in_coin "quarter") / cents_in_coin "dime" = 19 := by
  sorry

#eval cents_in_coin "dime"  -- Should output 10
#eval cents_in_coin "quarter"  -- Should output 25

end NUMINAMATH_CALUDE_sam_initial_dimes_l95_9582


namespace NUMINAMATH_CALUDE_tangency_triangle_area_for_given_radii_l95_9581

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents three mutually externally tangent circles -/
structure TangentCircles where
  c1 : Circle
  c2 : Circle
  c3 : Circle

/-- The area of the triangle formed by the points of tangency of three mutually externally tangent circles -/
def tangencyTriangleArea (tc : TangentCircles) : ℝ := sorry

/-- Theorem stating that for the given radii, the area of the tangency triangle is 120/25 -/
theorem tangency_triangle_area_for_given_radii :
  ∀ tc : TangentCircles,
    tc.c1.radius = 5 ∧ tc.c2.radius = 12 ∧ tc.c3.radius = 13 →
    tangencyTriangleArea tc = 120 / 25 := by
  sorry

end NUMINAMATH_CALUDE_tangency_triangle_area_for_given_radii_l95_9581


namespace NUMINAMATH_CALUDE_inequality_counterexample_l95_9564

theorem inequality_counterexample : 
  (∀ a b : ℝ, a > 0 → b > 0 → a + b ≥ 2 * Real.sqrt (a * b)) → 
  ¬(∀ x : ℝ, x + 1/x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_counterexample_l95_9564


namespace NUMINAMATH_CALUDE_plywood_length_l95_9566

/-- The length of a rectangular piece of plywood with given area and width -/
theorem plywood_length (area width : ℝ) (h1 : area = 24) (h2 : width = 6) :
  area / width = 4 := by
  sorry

end NUMINAMATH_CALUDE_plywood_length_l95_9566


namespace NUMINAMATH_CALUDE_city_B_sand_amount_l95_9573

def total_sand : ℝ := 95
def city_A_sand : ℝ := 16.5
def city_C_sand : ℝ := 24.5
def city_D_sand : ℝ := 28

theorem city_B_sand_amount : 
  total_sand - city_A_sand - city_C_sand - city_D_sand = 26 := by
  sorry

end NUMINAMATH_CALUDE_city_B_sand_amount_l95_9573


namespace NUMINAMATH_CALUDE_cheaper_to_buy_more_count_l95_9580

def C (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 15 then 15 * n + 20
  else if 16 ≤ n ∧ n ≤ 30 then 13 * n
  else if 31 ≤ n ∧ n ≤ 45 then 11 * n + 50
  else 9 * n

theorem cheaper_to_buy_more_count :
  (∃ s : Finset ℕ, s.card = 4 ∧ ∀ n ∈ s, C (n + 1) < C n) ∧
  ¬(∃ s : Finset ℕ, s.card > 4 ∧ ∀ n ∈ s, C (n + 1) < C n) :=
by sorry

end NUMINAMATH_CALUDE_cheaper_to_buy_more_count_l95_9580


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l95_9516

theorem greatest_integer_inequality : ∀ y : ℤ, (5 : ℚ) / 8 > (y : ℚ) / 17 ↔ y ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l95_9516


namespace NUMINAMATH_CALUDE_special_sequence_max_length_l95_9567

/-- A finite sequence of real numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (∀ i, i + 2 < n → a i + a (i + 1) + a (i + 2) < 0) ∧
  (∀ i, i + 3 < n → a i + a (i + 1) + a (i + 2) + a (i + 3) > 0)

/-- The maximum length of a SpecialSequence is 5 -/
theorem special_sequence_max_length :
  ∀ n : ℕ, ∀ a : ℕ → ℝ, SpecialSequence a n → n ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_special_sequence_max_length_l95_9567


namespace NUMINAMATH_CALUDE_complement_of_A_l95_9519

def U : Set Nat := {1, 2, 3}
def A : Set Nat := {1, 3}

theorem complement_of_A : (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l95_9519


namespace NUMINAMATH_CALUDE_circle_properties_l95_9520

-- Define the circle C
def C : Set (ℝ × ℝ) := sorry

-- Define the endpoints of the diameter
def endpoint1 : ℝ × ℝ := (3, -2)
def endpoint2 : ℝ × ℝ := (-9, 4)

-- Define the center of the circle
def center : ℝ × ℝ := ((-3 : ℝ), (1 : ℝ))

-- Define the point to be checked
def point : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem circle_properties :
  (center = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2)) ∧
  (point ∉ C) := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l95_9520


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l95_9574

/-- Given that in the expansion of (2x + a/x^2)^5, the coefficient of x^(-4) is 320, prove that a = 2 -/
theorem binomial_expansion_coefficient (a : ℝ) : 
  (∃ (c : ℝ), c = (Nat.choose 5 3) * 2^2 * a^3 ∧ c = 320) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l95_9574


namespace NUMINAMATH_CALUDE_symmetry_of_point_l95_9569

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Origin point (0,0) -/
def origin : Point := ⟨0, 0⟩

/-- Symmetry of a point about the origin -/
def symmetric_about_origin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

theorem symmetry_of_point :
  let A : Point := ⟨2, -1⟩
  let B : Point := symmetric_about_origin A
  B = ⟨-2, 1⟩ := by sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l95_9569


namespace NUMINAMATH_CALUDE_slope_angle_vertical_line_l95_9541

/-- Given two points A(2, 1) and B(2, 3), prove that the slope angle of the line AB is 90 degrees. -/
theorem slope_angle_vertical_line : 
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (2, 3)
  let slope_angle := Real.arctan ((B.2 - A.2) / (B.1 - A.1)) * (180 / Real.pi)
  slope_angle = 90 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_vertical_line_l95_9541


namespace NUMINAMATH_CALUDE_no_equal_coin_exchange_l95_9579

theorem no_equal_coin_exchange : ¬ ∃ (n : ℕ), n > 0 ∧ n * (1 + 2 + 3 + 5) = 500 := by
  sorry

end NUMINAMATH_CALUDE_no_equal_coin_exchange_l95_9579


namespace NUMINAMATH_CALUDE_product_101_squared_l95_9587

theorem product_101_squared : 101 * 101 = 10201 := by
  sorry

end NUMINAMATH_CALUDE_product_101_squared_l95_9587


namespace NUMINAMATH_CALUDE_both_selected_l95_9595

-- Define the probabilities of selection for Ram and Ravi
def prob_ram : ℚ := 1/7
def prob_ravi : ℚ := 1/5

-- Define the probability of both being selected
def prob_both : ℚ := prob_ram * prob_ravi

-- Theorem: The probability of both Ram and Ravi being selected is 1/35
theorem both_selected : prob_both = 1/35 := by
  sorry

end NUMINAMATH_CALUDE_both_selected_l95_9595


namespace NUMINAMATH_CALUDE_work_completion_time_l95_9542

/-- The number of days it takes A to complete the work alone -/
def days_A : ℝ := 6

/-- The total payment for the work -/
def total_payment : ℝ := 4000

/-- The number of days it takes A, B, and C to complete the work together -/
def days_ABC : ℝ := 3

/-- The payment to C -/
def payment_C : ℝ := 500.0000000000002

/-- The number of days it takes B to complete the work alone -/
def days_B : ℝ := 8

theorem work_completion_time :
  (1 / days_A + 1 / days_B + payment_C / total_payment * (1 / days_ABC) = 1 / days_ABC) ∧
  days_B = 8 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l95_9542


namespace NUMINAMATH_CALUDE_nilpotent_matrices_l95_9530

open Matrix

theorem nilpotent_matrices (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℝ) 
  (t : Fin (n+1) → ℝ) (h_distinct : ∀ i j, i ≠ j → t i ≠ t j) :
  (∀ i : Fin (n+1), ∃ k : ℕ, (A + t i • B) ^ k = 0) →
  (∃ k₁ : ℕ, A ^ k₁ = 0) ∧ (∃ k₂ : ℕ, B ^ k₂ = 0) := by
  sorry

end NUMINAMATH_CALUDE_nilpotent_matrices_l95_9530


namespace NUMINAMATH_CALUDE_infinite_pairs_divisibility_property_l95_9502

theorem infinite_pairs_divisibility_property (x : ℤ) (h : x ≥ 2) :
  let y := x * (x^8 - x^2 - 1)
  1 < x ∧ x < y ∧ (x^3 + y) ∣ (x + y^3) :=
by sorry

end NUMINAMATH_CALUDE_infinite_pairs_divisibility_property_l95_9502


namespace NUMINAMATH_CALUDE_initial_birds_count_l95_9535

theorem initial_birds_count (storks : ℕ) (additional_birds : ℕ) (bird_stork_difference : ℕ) :
  storks = 5 →
  additional_birds = 4 →
  bird_stork_difference = 2 →
  ∃ initial_birds : ℕ, 
    initial_birds + additional_birds = storks + bird_stork_difference ∧
    initial_birds = 3 :=
by sorry

end NUMINAMATH_CALUDE_initial_birds_count_l95_9535


namespace NUMINAMATH_CALUDE_gala_trees_l95_9512

/-- Represents the orchard with Fuji and Gala apple trees -/
structure Orchard where
  total : ℕ
  fuji : ℕ
  gala : ℕ
  crossPollinated : ℕ

/-- Conditions of the orchard -/
def validOrchard (o : Orchard) : Prop :=
  o.crossPollinated = o.total / 10 ∧
  o.fuji + o.crossPollinated = 170 ∧
  o.fuji = 3 * o.total / 4 ∧
  o.total = o.fuji + o.gala + o.crossPollinated

theorem gala_trees (o : Orchard) (h : validOrchard o) : o.gala = 50 := by
  sorry

end NUMINAMATH_CALUDE_gala_trees_l95_9512


namespace NUMINAMATH_CALUDE_train_length_l95_9539

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 54 → time = 7 → speed * time * (1000 / 3600) = 105 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l95_9539


namespace NUMINAMATH_CALUDE_annual_phone_bill_l95_9531

def original_bill : ℚ := 50
def increase_rate : ℚ := 0.1
def months_per_year : ℕ := 12

theorem annual_phone_bill :
  (original_bill * (1 + increase_rate)) * months_per_year = 660 := by
  sorry

end NUMINAMATH_CALUDE_annual_phone_bill_l95_9531


namespace NUMINAMATH_CALUDE_equation_solutions_l95_9545

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 9 = 0 ↔ x = 3/2 ∨ x = -3/2) ∧
  (∀ x : ℝ, 64 * (x - 2)^3 - 1 = 0 ↔ x = 9/4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l95_9545


namespace NUMINAMATH_CALUDE_area_of_triangle_OAB_is_one_l95_9513

/-- Given vector a and b in ℝ², prove that the area of triangle OAB is 1 -/
theorem area_of_triangle_OAB_is_one 
  (a b : ℝ × ℝ)
  (h_a : a = (-1/2, Real.sqrt 3/2))
  (h_OA : (a.1 - b.1, a.2 - b.2) = (a.1 - b.1, a.2 - b.2))
  (h_OB : (a.1 + b.1, a.2 + b.2) = (a.1 + b.1, a.2 + b.2))
  (h_isosceles : ‖(a.1 - b.1, a.2 - b.2)‖ = ‖(a.1 + b.1, a.2 + b.2)‖)
  (h_right_angle : (a.1 - b.1, a.2 - b.2) • (a.1 + b.1, a.2 + b.2) = 0) :
  (1/2) * ‖(a.1 - b.1, a.2 - b.2)‖ * ‖(a.1 + b.1, a.2 + b.2)‖ = 1 :=
by sorry


end NUMINAMATH_CALUDE_area_of_triangle_OAB_is_one_l95_9513


namespace NUMINAMATH_CALUDE_greatest_a_value_l95_9504

theorem greatest_a_value (x a : ℤ) : 
  (∃ x : ℤ, x^2 + a*x = 18) ∧ (a > 0) → a ≤ 17 := by
  sorry

end NUMINAMATH_CALUDE_greatest_a_value_l95_9504


namespace NUMINAMATH_CALUDE_ellipse_other_x_intercept_l95_9590

/-- Definition of an ellipse with given foci and one x-intercept -/
def Ellipse (f1 f2 x1 : ℝ × ℝ) : Prop :=
  let d1 (x y : ℝ) := Real.sqrt ((x - f1.1)^2 + (y - f1.2)^2)
  let d2 (x y : ℝ) := Real.sqrt ((x - f2.1)^2 + (y - f2.2)^2)
  ∀ x y : ℝ, d1 x y + d2 x y = d1 x1.1 x1.2 + d2 x1.1 x1.2

/-- The main theorem -/
theorem ellipse_other_x_intercept :
  let f1 : ℝ × ℝ := (0, 3)
  let f2 : ℝ × ℝ := (4, 0)
  let x1 : ℝ × ℝ := (1, 0)
  let x2 : ℝ × ℝ := ((13 - 14 * Real.sqrt 10) / (2 * Real.sqrt 10 + 14), 0)
  Ellipse f1 f2 x1 → x2.1 ≠ x1.1 → Ellipse f1 f2 x2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_other_x_intercept_l95_9590


namespace NUMINAMATH_CALUDE_circle_area_equality_l95_9578

theorem circle_area_equality (r₁ r₂ : ℝ) (h₁ : r₁ = 24) (h₂ : r₂ = 34) :
  ∃ r : ℝ, π * r^2 = π * (r₂^2 - r₁^2) ∧ r = 2 * Real.sqrt 145 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equality_l95_9578


namespace NUMINAMATH_CALUDE_power_of_negative_square_l95_9515

theorem power_of_negative_square (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_square_l95_9515


namespace NUMINAMATH_CALUDE_trig_identity_1_trig_identity_2_l95_9500

-- Part 1
theorem trig_identity_1 (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3/22 := by
  sorry

-- Part 2
theorem trig_identity_2 (α β : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos (α + β) = Real.sqrt 5 / 5)
  (h4 : Real.sin (α - β) = Real.sqrt 10 / 10) :
  2 * β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_1_trig_identity_2_l95_9500


namespace NUMINAMATH_CALUDE_triangle_sum_of_squares_l95_9563

-- Define an equilateral triangle ABC with side length 10
def Triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = 10 ∧ dist B C = 10 ∧ dist C A = 10

-- Define points P and Q on AB and AC respectively
def PointP (A B P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B ∧ dist A P = 2

def PointQ (A C Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • A + t • C ∧ dist A Q = 2

-- Theorem statement
theorem triangle_sum_of_squares 
  (A B C P Q : ℝ × ℝ) 
  (h_triangle : Triangle A B C) 
  (h_P : PointP A B P) 
  (h_Q : PointQ A C Q) : 
  (dist C P)^2 + (dist C Q)^2 = 168 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_of_squares_l95_9563


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l95_9552

/-- The trajectory of the midpoint of a segment between a fixed point and a point on a circle -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ (px py : ℝ), px^2 + py^2 = 16 ∧ x = (px + 12) / 2 ∧ y = py / 2) → 
  (x - 6)^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l95_9552


namespace NUMINAMATH_CALUDE_invalid_deduction_from_false_premise_l95_9526

-- Define the concept of a premise
def Premise : Type := Prop

-- Define the concept of a conclusion
def Conclusion : Type := Prop

-- Define the concept of a deduction
def Deduction := Premise → Conclusion

-- Define what it means for a premise to be false
def IsFalsePremise (p : Premise) : Prop := ¬p

-- Define what it means for a conclusion to be valid
def IsValidConclusion (c : Conclusion) : Prop := c

-- Theorem: Logical deductions based on false premises cannot lead to valid conclusions
theorem invalid_deduction_from_false_premise :
  ∀ (p : Premise) (d : Deduction),
    IsFalsePremise p → ¬(IsValidConclusion (d p)) :=
by sorry

end NUMINAMATH_CALUDE_invalid_deduction_from_false_premise_l95_9526


namespace NUMINAMATH_CALUDE_class_size_class_size_is_60_l95_9540

theorem class_size (cafeteria_students : ℕ) (no_lunch_students : ℕ) : ℕ :=
  let bring_lunch_students := 3 * cafeteria_students
  let total_lunch_students := cafeteria_students + bring_lunch_students
  let total_students := total_lunch_students + no_lunch_students
  total_students

theorem class_size_is_60 : 
  class_size 10 20 = 60 := by sorry

end NUMINAMATH_CALUDE_class_size_class_size_is_60_l95_9540


namespace NUMINAMATH_CALUDE_abs_neg_eleven_l95_9529

theorem abs_neg_eleven : |(-11 : ℤ)| = 11 := by sorry

end NUMINAMATH_CALUDE_abs_neg_eleven_l95_9529


namespace NUMINAMATH_CALUDE_union_complement_equality_l95_9572

def U : Finset Nat := {0, 1, 2, 4, 6, 8}
def M : Finset Nat := {0, 4, 6}
def N : Finset Nat := {0, 1, 6}

theorem union_complement_equality : M ∪ (U \ N) = {0, 2, 4, 6, 8} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equality_l95_9572


namespace NUMINAMATH_CALUDE_baker_cakes_l95_9584

theorem baker_cakes (cakes_sold : ℕ) (cakes_remaining : ℕ) (initial_cakes : ℕ) : 
  cakes_sold = 10 → cakes_remaining = 139 → initial_cakes = cakes_sold + cakes_remaining → initial_cakes = 149 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_l95_9584


namespace NUMINAMATH_CALUDE_reciprocal_problem_l95_9538

theorem reciprocal_problem (x : ℚ) (h : 8 * x = 10) : 50 * (1 / x) = 40 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l95_9538


namespace NUMINAMATH_CALUDE_complex_equation_solution_l95_9550

theorem complex_equation_solution (z : ℂ) : z * (2 - I) = 11 + 7 * I → z = 3 + 5 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l95_9550


namespace NUMINAMATH_CALUDE_investment_schemes_count_l95_9501

/-- The number of projects to be invested -/
def num_projects : ℕ := 4

/-- The number of candidate cities -/
def num_cities : ℕ := 4

/-- The maximum number of projects that can be invested in a single city -/
def max_projects_per_city : ℕ := 2

/-- A function that calculates the number of ways to distribute projects among cities -/
def investment_schemes (projects : ℕ) (cities : ℕ) (max_per_city : ℕ) : ℕ := sorry

/-- Theorem stating that the number of investment schemes is 240 -/
theorem investment_schemes_count : 
  investment_schemes num_projects num_cities max_projects_per_city = 240 := by sorry

end NUMINAMATH_CALUDE_investment_schemes_count_l95_9501


namespace NUMINAMATH_CALUDE_inverse_arcsin_function_l95_9576

theorem inverse_arcsin_function (f : ℝ → ℝ) (h : ∀ x, f x = Real.arcsin (2 * x + 1)) :
  f⁻¹ (π / 6) = -1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_arcsin_function_l95_9576


namespace NUMINAMATH_CALUDE_b_work_fraction_proof_l95_9509

/-- The fraction of a day that b works --/
def b_work_fraction : ℚ := 1 / 5

/-- The time it takes a and b together to complete the work (in days) --/
def together_time : ℚ := 12

/-- The time it takes a alone to complete the work (in days) --/
def a_alone_time : ℚ := 20

/-- The time it takes a and b together to complete the work when b works a fraction of a day (in days) --/
def partial_together_time : ℚ := 15

theorem b_work_fraction_proof :
  (1 / a_alone_time + b_work_fraction * (1 / together_time) = 1 / partial_together_time) ∧
  (b_work_fraction > 0) ∧ (b_work_fraction < 1) := by
  sorry

end NUMINAMATH_CALUDE_b_work_fraction_proof_l95_9509


namespace NUMINAMATH_CALUDE_joe_original_cans_l95_9598

/-- Represents the number of rooms that can be painted with a given number of paint cans -/
def rooms_paintable (cans : ℕ) : ℕ := sorry

/-- The number of rooms Joe could initially paint -/
def initial_rooms : ℕ := 40

/-- The number of rooms Joe could paint after losing cans -/
def remaining_rooms : ℕ := 32

/-- The number of cans Joe lost -/
def lost_cans : ℕ := 2

theorem joe_original_cans :
  ∃ (original_cans : ℕ),
    rooms_paintable original_cans = initial_rooms ∧
    rooms_paintable (original_cans - lost_cans) = remaining_rooms ∧
    original_cans = 10 := by sorry

end NUMINAMATH_CALUDE_joe_original_cans_l95_9598


namespace NUMINAMATH_CALUDE_fixed_point_of_parabola_l95_9518

theorem fixed_point_of_parabola (s : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 + s * x - 3 * s
  f 3 = 36 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_parabola_l95_9518


namespace NUMINAMATH_CALUDE_apples_problem_l95_9547

/-- The number of apples Adam and Jackie have together -/
def total_apples (adam : ℕ) (jackie : ℕ) : ℕ := adam + jackie

/-- Adam has 9 more apples than the total -/
def adam_more_than_total (adam : ℕ) (jackie : ℕ) : Prop :=
  adam = total_apples adam jackie + 9

/-- Adam has 8 more apples than Jackie -/
def adam_more_than_jackie (adam : ℕ) (jackie : ℕ) : Prop :=
  adam = jackie + 8

theorem apples_problem (adam jackie : ℕ) 
  (h1 : adam_more_than_total adam jackie)
  (h2 : adam_more_than_jackie adam jackie)
  (h3 : adam = 21) : 
  total_apples adam jackie = 34 := by
  sorry

end NUMINAMATH_CALUDE_apples_problem_l95_9547


namespace NUMINAMATH_CALUDE_polynomial_factorization_l95_9583

theorem polynomial_factorization (x : ℝ) : 
  x^4 - 4*x^3 + 6*x^2 - 4*x + 1 = (x - 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l95_9583


namespace NUMINAMATH_CALUDE_sum_in_base4_l95_9506

/-- Converts a base 4 number to base 10 -/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4^i)) 0

/-- Converts a base 10 number to base 4 -/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem sum_in_base4 :
  let a := [2, 1, 3]  -- 312₄ in reverse order
  let b := [1, 3]     -- 31₄ in reverse order
  let c := [3]        -- 3₄
  let sum := [2, 1, 0, 1]  -- 1012₄ in reverse order
  base10ToBase4 (base4ToBase10 a + base4ToBase10 b + base4ToBase10 c) = sum := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base4_l95_9506


namespace NUMINAMATH_CALUDE_max_value_on_curve_l95_9571

-- Define the curve
def on_curve (x y b : ℝ) : Prop := x^2/4 + y^2/b^2 = 1

-- Define the function to maximize
def f (x y : ℝ) : ℝ := x^2 + 2*y

-- State the theorem
theorem max_value_on_curve (b : ℝ) (h : b > 0) :
  (∃ (x y : ℝ), on_curve x y b ∧ 
    ∀ (x' y' : ℝ), on_curve x' y' b → f x y ≥ f x' y') →
  ((0 < b ∧ b ≤ 4 → ∃ (x y : ℝ), on_curve x y b ∧ f x y = b^2/4 + 4) ∧
   (b > 4 → ∃ (x y : ℝ), on_curve x y b ∧ f x y = 2*b)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_curve_l95_9571


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l95_9533

theorem complex_modulus_problem (z : ℂ) (i : ℂ) : 
  i^2 = -1 → z = (1 - i) / (1 + i) → Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l95_9533


namespace NUMINAMATH_CALUDE_no_additional_salt_needed_l95_9525

/-- Represents the problem of mixing salt to achieve a specific profit -/
def SaltMixtureProblem (initialSaltWeight : ℝ) (initialSaltCost : ℝ) (mixtureSalePrice : ℝ) (desiredProfit : ℝ) :=
  ∃ (additionalSaltWeight : ℝ),
    additionalSaltWeight ≥ 0 ∧
    let totalCost := initialSaltWeight * initialSaltCost + additionalSaltWeight * 0.5
    let totalWeight := initialSaltWeight + additionalSaltWeight
    let totalRevenue := totalWeight * mixtureSalePrice
    totalRevenue = (1 + desiredProfit) * totalCost

/-- The main theorem stating that no additional salt is needed for the given problem -/
theorem no_additional_salt_needed :
  SaltMixtureProblem 40 0.35 0.48 0.2 → ∃ (x : ℝ), x = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_additional_salt_needed_l95_9525


namespace NUMINAMATH_CALUDE_percentage_error_calculation_l95_9514

theorem percentage_error_calculation (N : ℝ) (h : N > 0) : 
  let correct := N * 5
  let incorrect := N / 10
  let absolute_error := |correct - incorrect|
  let percentage_error := (absolute_error / correct) * 100
  percentage_error = 98 := by
sorry

end NUMINAMATH_CALUDE_percentage_error_calculation_l95_9514


namespace NUMINAMATH_CALUDE_value_of_X_l95_9534

theorem value_of_X : ∀ M N X : ℕ,
  M = 2023 / 3 →
  N = M / 3 →
  X = M - N →
  X = 449 := by
sorry

end NUMINAMATH_CALUDE_value_of_X_l95_9534


namespace NUMINAMATH_CALUDE_energy_saving_product_analysis_l95_9589

/-- Represents the sales volume in ten thousand items -/
def y (x : ℝ) : ℝ := -x + 120

/-- Represents the profit in ten thousand dollars -/
def W (x : ℝ) : ℝ := -(x - 100)^2 - 80

/-- Represents the profit in the second year considering donations -/
def W2 (x : ℝ) : ℝ := (x - 82) * (-x + 120)

theorem energy_saving_product_analysis :
  (∀ x, 90 ≤ x → x ≤ 110 → y x = -x + 120) ∧
  (∀ x, 90 ≤ x ∧ x ≤ 110 → W x ≤ 0) ∧
  (∃ x, 90 ≤ x ∧ x ≤ 110 ∧ W x = -80) ∧
  (∃ x, 92 ≤ x ∧ x ≤ 110 ∧ W2 x ≥ 280 ∧
    ∀ x', 92 ≤ x' ∧ x' ≤ 110 → W2 x' ≤ W2 x) :=
by sorry

end NUMINAMATH_CALUDE_energy_saving_product_analysis_l95_9589


namespace NUMINAMATH_CALUDE_candy_distribution_l95_9523

theorem candy_distribution (n : ℕ) (total_candies : ℕ) : 
  total_candies = 120 →
  total_candies = 2 * n →
  n = 60 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l95_9523


namespace NUMINAMATH_CALUDE_cos_shift_l95_9544

theorem cos_shift (x : ℝ) : 
  Real.cos (x / 2 - π / 3) = Real.cos ((x - 2 * π / 3) / 2) := by sorry

end NUMINAMATH_CALUDE_cos_shift_l95_9544


namespace NUMINAMATH_CALUDE_complex_equation_solution_l95_9556

theorem complex_equation_solution (i z : ℂ) (h1 : i * i = -1) (h2 : i * z = 1 - i) :
  z = -i - 1 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l95_9556


namespace NUMINAMATH_CALUDE_largest_k_for_inequality_l95_9575

theorem largest_k_for_inequality : 
  (∃ (k : ℝ), ∀ (x : ℝ), (1 + Real.sin x) / (2 + Real.cos x) ≥ k) ∧ 
  (∀ (k : ℝ), k > 4/3 → ¬(∃ (x : ℝ), (1 + Real.sin x) / (2 + Real.cos x) ≥ k)) :=
sorry

end NUMINAMATH_CALUDE_largest_k_for_inequality_l95_9575


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l95_9546

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l95_9546


namespace NUMINAMATH_CALUDE_cubic_gt_27_implies_abs_gt_3_but_not_conversely_l95_9586

theorem cubic_gt_27_implies_abs_gt_3_but_not_conversely :
  (∀ x : ℝ, x^3 > 27 → |x| > 3) ∧
  (∃ x : ℝ, |x| > 3 ∧ x^3 ≤ 27) :=
by sorry

end NUMINAMATH_CALUDE_cubic_gt_27_implies_abs_gt_3_but_not_conversely_l95_9586
