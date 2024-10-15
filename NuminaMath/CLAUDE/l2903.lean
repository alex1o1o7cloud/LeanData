import Mathlib

namespace NUMINAMATH_CALUDE_square_ending_in_five_l2903_290343

theorem square_ending_in_five (a : ℕ) :
  let n : ℕ := 10 * a + 5
  ∃ (m : ℕ), n^2 = m^2 → a % 10 = 2 :=
by sorry

end NUMINAMATH_CALUDE_square_ending_in_five_l2903_290343


namespace NUMINAMATH_CALUDE_fraction_sum_l2903_290353

theorem fraction_sum (a b : ℕ+) (h1 : (a : ℚ) / b = 9 / 16) 
  (h2 : ∀ d : ℕ, d > 1 → d ∣ a → d ∣ b → False) : 
  (a : ℕ) + b = 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2903_290353


namespace NUMINAMATH_CALUDE_sine_function_expression_l2903_290324

theorem sine_function_expression 
  (y : ℝ → ℝ) 
  (A ω : ℝ) 
  (h1 : A > 0)
  (h2 : ω > 0)
  (h3 : ∀ x, y x = A * Real.sin (ω * x + φ))
  (h4 : A = 2)
  (h5 : 2 * Real.pi / ω = Real.pi / 2)
  (h6 : φ = -3) :
  ∀ x, y x = 2 * Real.sin (4 * x - 3) := by
sorry

end NUMINAMATH_CALUDE_sine_function_expression_l2903_290324


namespace NUMINAMATH_CALUDE_square_difference_l2903_290373

theorem square_difference (m n : ℕ+) 
  (h : (2001 : ℕ) * m ^ 2 + m = (2002 : ℕ) * n ^ 2 + n) :
  ∃ k : ℕ, (m : ℤ) - (n : ℤ) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2903_290373


namespace NUMINAMATH_CALUDE_value_of_x_when_y_is_two_l2903_290304

theorem value_of_x_when_y_is_two (x y : ℚ) : 
  y = 1 / (5 * x + 2) → y = 2 → x = -3 / 10 := by sorry

end NUMINAMATH_CALUDE_value_of_x_when_y_is_two_l2903_290304


namespace NUMINAMATH_CALUDE_remainder_after_adding_4500_l2903_290322

theorem remainder_after_adding_4500 (n : ℤ) (h : n % 6 = 1) : (n + 4500) % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_4500_l2903_290322


namespace NUMINAMATH_CALUDE_greatest_integer_of_a_l2903_290315

def a : ℕ → ℚ
  | 0 => 1994
  | n + 1 => 1994^2 / (a n + 1)

theorem greatest_integer_of_a (n : ℕ) (h : n ≤ 998) :
  ⌊a n⌋ = 1994 - n := by sorry

end NUMINAMATH_CALUDE_greatest_integer_of_a_l2903_290315


namespace NUMINAMATH_CALUDE_shortest_side_of_octagon_l2903_290327

theorem shortest_side_of_octagon (x : ℝ) : 
  x > 0 →                             -- x is positive
  x^2 = 100 →                         -- combined area of cut-off triangles
  20 - x = 10 :=                      -- shortest side of octagon
by sorry

end NUMINAMATH_CALUDE_shortest_side_of_octagon_l2903_290327


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2903_290359

/-- Given a triangle with inradius 2.5 cm and area 45 cm², its perimeter is 36 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 45 → A = r * (p / 2) → p = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2903_290359


namespace NUMINAMATH_CALUDE_min_value_theorem_l2903_290318

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log (2^x) + Real.log (8^y) = Real.log 2) : 
  1/x + 1/(3*y) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2903_290318


namespace NUMINAMATH_CALUDE_g_of_neg_four_l2903_290316

/-- Given a function g(x) = 5x - 2, prove that g(-4) = -22 -/
theorem g_of_neg_four (g : ℝ → ℝ) (h : ∀ x, g x = 5 * x - 2) : g (-4) = -22 := by
  sorry

end NUMINAMATH_CALUDE_g_of_neg_four_l2903_290316


namespace NUMINAMATH_CALUDE_min_dot_product_l2903_290375

/-- A line with direction vector (4, -4) passing through (0, -4) -/
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (t, -t - 4)}

/-- Two points on line_l -/
def point_on_line (M N : ℝ × ℝ) : Prop :=
  M ∈ line_l ∧ N ∈ line_l

/-- Distance between two points is 4 -/
def distance_is_4 (M N : ℝ × ℝ) : Prop :=
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 16

/-- Dot product of OM and ON -/
def dot_product (M N : ℝ × ℝ) : ℝ :=
  M.1 * N.1 + M.2 * N.2

theorem min_dot_product (M N : ℝ × ℝ) 
  (h1 : point_on_line M N) 
  (h2 : distance_is_4 M N) : 
  ∃ min_val : ℝ, min_val = 4 ∧ ∀ M' N' : ℝ × ℝ, 
    point_on_line M' N' → distance_is_4 M' N' → 
    dot_product M' N' ≥ min_val :=
  sorry

end NUMINAMATH_CALUDE_min_dot_product_l2903_290375


namespace NUMINAMATH_CALUDE_solve_for_y_l2903_290371

theorem solve_for_y (x y : ℝ) (h : 2 * x - 7 * y = 8) : y = (2 * x - 8) / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2903_290371


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2903_290302

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 2 * x + 15 = 0 ↔ x = (a : ℂ) + b * I ∨ x = (a : ℂ) - b * I) →
  a + b^2 = 79/25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2903_290302


namespace NUMINAMATH_CALUDE_arithmetic_is_linear_geometric_is_exponential_l2903_290323

/-- Definition of an arithmetic sequence -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Definition of a geometric sequence -/
def geometric_sequence (a₁ r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

/-- Theorem: The n-th term of an arithmetic sequence is a linear function of n -/
theorem arithmetic_is_linear (a₁ d : ℝ) :
  ∃ m b : ℝ, ∀ n : ℕ, arithmetic_sequence a₁ d n = m * n + b :=
sorry

/-- Theorem: The n-th term of a geometric sequence is an exponential function of n -/
theorem geometric_is_exponential (a₁ r : ℝ) :
  ∃ A B : ℝ, ∀ n : ℕ, geometric_sequence a₁ r n = A * B^n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_is_linear_geometric_is_exponential_l2903_290323


namespace NUMINAMATH_CALUDE_equation_properties_l2903_290306

-- Define the equation
def equation (x p : ℝ) : ℝ := (x - 3) * (x - 2) - p^2

-- Define the property of having two distinct real roots
def has_two_distinct_real_roots (p : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ p = 0 ∧ equation x₂ p = 0

-- Define the condition for the roots
def roots_condition (x₁ x₂ : ℝ) : Prop :=
  x₁^2 + x₂^2 = 3 * x₁ * x₂

-- Theorem statement
theorem equation_properties :
  (∀ p : ℝ, has_two_distinct_real_roots p) ∧
  (∀ p x₁ x₂ : ℝ, equation x₁ p = 0 → equation x₂ p = 0 → 
    roots_condition x₁ x₂ → p = 1 ∨ p = -1) :=
sorry

end NUMINAMATH_CALUDE_equation_properties_l2903_290306


namespace NUMINAMATH_CALUDE_stability_comparison_l2903_290366

/-- Represents a student's math exam scores -/
structure StudentScores where
  mean : ℝ
  variance : ℝ
  exam_count : ℕ

/-- Defines the concept of stability for exam scores -/
def more_stable (a b : StudentScores) : Prop :=
  a.variance < b.variance

theorem stability_comparison 
  (student_A student_B : StudentScores)
  (h1 : student_A.mean = student_B.mean)
  (h2 : student_A.exam_count = student_B.exam_count)
  (h3 : student_A.exam_count = 5)
  (h4 : student_A.mean = 102)
  (h5 : student_A.variance = 38)
  (h6 : student_B.variance = 15) :
  more_stable student_B student_A :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_l2903_290366


namespace NUMINAMATH_CALUDE_train_pass_man_time_l2903_290386

def train_speed : Real := 36 -- km/hr
def platform_length : Real := 180 -- meters
def time_pass_platform : Real := 30 -- seconds

theorem train_pass_man_time : 
  ∃ (train_length : Real),
    (train_speed * 1000 / 3600 * time_pass_platform = train_length + platform_length) ∧
    (train_length / (train_speed * 1000 / 3600) = 12) :=
by sorry

end NUMINAMATH_CALUDE_train_pass_man_time_l2903_290386


namespace NUMINAMATH_CALUDE_binomial_expansion_constant_term_l2903_290320

/-- 
Given that the third term of the expansion of (3x - 2/x)^n is a constant term,
prove that n = 8.
-/
theorem binomial_expansion_constant_term (n : ℕ) : 
  (∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → 
    (Nat.choose n 2 * (3 * x - 2 / x)^(n - 2) * (-2 / x)^2 = c)) → 
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_constant_term_l2903_290320


namespace NUMINAMATH_CALUDE_bisecting_angle_tangent_l2903_290321

/-- A triangle with side lengths 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 5
  hb : b = 12
  hc : c = 13

/-- A line that bisects both the perimeter and area of the triangle -/
structure BisectingLine where
  x : ℝ
  y : ℝ

/-- The two bisecting lines of the triangle -/
def bisecting_lines (t : RightTriangle) : Prod BisectingLine BisectingLine :=
  ⟨⟨10, -5⟩, ⟨7.5, -7.5⟩⟩

/-- The acute angle between the two bisecting lines -/
def bisecting_angle (t : RightTriangle) : ℝ := sorry

theorem bisecting_angle_tangent (t : RightTriangle) :
  let lines := bisecting_lines t
  let φ := bisecting_angle t
  Real.tan φ = 
    let v1 := lines.1
    let v2 := lines.2
    let dot_product := v1.x * v2.x + v1.y * v2.y
    let mag1 := Real.sqrt (v1.x^2 + v1.y^2)
    let mag2 := Real.sqrt (v2.x^2 + v2.y^2)
    let cos_φ := dot_product / (mag1 * mag2)
    Real.sqrt (1 - cos_φ^2) / cos_φ := by sorry

end NUMINAMATH_CALUDE_bisecting_angle_tangent_l2903_290321


namespace NUMINAMATH_CALUDE_container_capacity_proof_l2903_290352

/-- The capacity of a container in liters -/
def container_capacity : ℝ := 100

/-- The initial fill level of the container as a percentage -/
def initial_fill : ℝ := 30

/-- The final fill level of the container as a percentage -/
def final_fill : ℝ := 75

/-- The amount of water added to the container in liters -/
def water_added : ℝ := 45

theorem container_capacity_proof :
  (final_fill / 100 * container_capacity) - (initial_fill / 100 * container_capacity) = water_added :=
sorry

end NUMINAMATH_CALUDE_container_capacity_proof_l2903_290352


namespace NUMINAMATH_CALUDE_largest_n_proof_l2903_290388

/-- Binary operation @ defined as n @ n = n - (n * 5) -/
def binary_op (n : ℤ) : ℤ := n - (n * 5)

/-- The largest positive integer n such that n @ n < 21 -/
def largest_n : ℕ := 1

theorem largest_n_proof :
  (∀ (m : ℕ), m > largest_n → binary_op m ≥ 21) ∧
  binary_op largest_n < 21 :=
sorry

end NUMINAMATH_CALUDE_largest_n_proof_l2903_290388


namespace NUMINAMATH_CALUDE_product_inspection_theorem_l2903_290364

/-- Represents a collection of products -/
structure ProductCollection where
  total : ℕ
  selected : ℕ

/-- Defines the concept of a population in statistics -/
def population (pc : ProductCollection) : ℕ := pc.total

/-- Defines the concept of a sample in statistics -/
def sample (pc : ProductCollection) : ℕ := pc.selected

/-- Defines the concept of sample size in statistics -/
def sampleSize (pc : ProductCollection) : ℕ := pc.selected

theorem product_inspection_theorem (pc : ProductCollection) 
  (h1 : pc.total = 80) 
  (h2 : pc.selected = 10) 
  (h3 : pc.selected ≤ pc.total) : 
  population pc = 80 ∧ 
  sampleSize pc = 10 ∧ 
  sample pc ≤ population pc := by
  sorry


end NUMINAMATH_CALUDE_product_inspection_theorem_l2903_290364


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_system_l2903_290362

theorem unique_solution_quadratic_system (x : ℚ) :
  (6 * x^2 + 19 * x - 7 = 0) ∧ (18 * x^2 + 47 * x - 21 = 0) → x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_system_l2903_290362


namespace NUMINAMATH_CALUDE_randys_fathers_biscuits_l2903_290349

/-- Proves that Randy's father gave him 13 biscuits given the initial conditions and final result. -/
theorem randys_fathers_biscuits :
  ∀ (initial mother_gave brother_ate final father_gave : ℕ),
  initial = 32 →
  mother_gave = 15 →
  brother_ate = 20 →
  final = 40 →
  initial + mother_gave + father_gave - brother_ate = final →
  father_gave = 13 := by
  sorry

end NUMINAMATH_CALUDE_randys_fathers_biscuits_l2903_290349


namespace NUMINAMATH_CALUDE_quadratic_equation_k_value_l2903_290369

theorem quadratic_equation_k_value (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - 402*x₁ + k = 0 ∧ 
                x₂^2 - 402*x₂ + k = 0 ∧ 
                x₁ + 3 = 80 * x₂) → 
  k = 1985 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_value_l2903_290369


namespace NUMINAMATH_CALUDE_car_final_velocity_l2903_290384

/-- Calculates the final velocity of a car parallel to the ground after accelerating on an inclined slope. -/
theorem car_final_velocity (u : Real) (a : Real) (t : Real) (θ : Real) :
  u = 10 ∧ a = 2 ∧ t = 3 ∧ θ = 15 * π / 180 →
  ∃ v : Real, abs (v - (u + a * t) * Real.cos θ) < 0.0001 ∧ abs (v - 15.4544) < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_car_final_velocity_l2903_290384


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l2903_290325

/-- Given that the binomial coefficients of the third and seventh terms
    in the expansion of (x+2)^n are equal, prove that n = 8 and
    the coefficient of the (k+1)th term is maximum when k = 5 or k = 6 -/
theorem binomial_expansion_properties (n : ℕ) :
  (Nat.choose n 2 = Nat.choose n 6) →
  (n = 8 ∧ 
   (∀ j : ℕ, j ≠ 5 ∧ j ≠ 6 → 
     Nat.choose 8 5 * 2^5 ≥ Nat.choose 8 j * 2^j ∧
     Nat.choose 8 6 * 2^6 ≥ Nat.choose 8 j * 2^j)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l2903_290325


namespace NUMINAMATH_CALUDE_min_value_theorem_l2903_290328

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : 2*x + 2*y + 3*z = 3) : 
  (2*(x + y)) / (x*y*z) ≥ 14.2222 := by
sorry

#eval (8 : ℚ) / (9 : ℚ) * 16

end NUMINAMATH_CALUDE_min_value_theorem_l2903_290328


namespace NUMINAMATH_CALUDE_initial_flow_rate_is_two_l2903_290330

/-- Represents the flow rate of cleaner through a pipe over time -/
structure FlowRate where
  initial : ℝ
  after15min : ℝ
  after25min : ℝ

/-- Calculates the total amount of cleaner used given a flow rate profile -/
def totalCleanerUsed (flow : FlowRate) : ℝ :=
  15 * flow.initial + 10 * flow.after15min + 5 * flow.after25min

/-- Theorem stating that the initial flow rate is 2 ounces per minute -/
theorem initial_flow_rate_is_two :
  ∃ (flow : FlowRate),
    flow.after15min = 3 ∧
    flow.after25min = 4 ∧
    totalCleanerUsed flow = 80 ∧
    flow.initial = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_flow_rate_is_two_l2903_290330


namespace NUMINAMATH_CALUDE_monotonic_cubic_range_l2903_290303

/-- The function f(x) = -x^3 + ax^2 - x - 1 is monotonic on ℝ -/
def is_monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

/-- The main theorem: if f(x) = -x^3 + ax^2 - x - 1 is monotonic on ℝ, then a ∈ [-√3, √3] -/
theorem monotonic_cubic_range (a : ℝ) :
  is_monotonic (fun x => -x^3 + a*x^2 - x - 1) → a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_range_l2903_290303


namespace NUMINAMATH_CALUDE_canoe_downstream_speed_l2903_290374

/-- The speed of a canoe downstream given its upstream speed and the stream speed -/
theorem canoe_downstream_speed (upstream_speed stream_speed : ℝ) :
  upstream_speed = 3 →
  stream_speed = 4.5 →
  upstream_speed + 2 * stream_speed = 12 :=
by sorry

end NUMINAMATH_CALUDE_canoe_downstream_speed_l2903_290374


namespace NUMINAMATH_CALUDE_base_six_addition_l2903_290379

/-- Given a base 6 addition problem 3XY_6 + 23_6 = 41X_6, prove that X + Y = 7 in base 10 -/
theorem base_six_addition (X Y : ℕ) : 
  (3 * 6^2 + X * 6 + Y) + (2 * 6 + 3) = 4 * 6^2 + X * 6 → X + Y = 7 :=
by sorry

end NUMINAMATH_CALUDE_base_six_addition_l2903_290379


namespace NUMINAMATH_CALUDE_square_root_one_ninth_l2903_290313

theorem square_root_one_ninth : Real.sqrt (1/9) = 1/3 ∨ Real.sqrt (1/9) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_one_ninth_l2903_290313


namespace NUMINAMATH_CALUDE_zhang_san_not_losing_probability_l2903_290336

theorem zhang_san_not_losing_probability
  (p_win : ℚ) (p_draw : ℚ)
  (h_win : p_win = 1 / 3)
  (h_draw : p_draw = 1 / 4) :
  p_win + p_draw = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_zhang_san_not_losing_probability_l2903_290336


namespace NUMINAMATH_CALUDE_negative_sqrt_product_l2903_290368

theorem negative_sqrt_product (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  -Real.sqrt a * Real.sqrt b = -Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_product_l2903_290368


namespace NUMINAMATH_CALUDE_max_uncovered_sections_specific_case_l2903_290307

/-- Represents a corridor with carpet strips -/
structure CarpetedCorridor where
  corridorLength : ℕ
  numStrips : ℕ
  totalStripLength : ℕ

/-- Calculates the maximum number of uncovered sections in a carpeted corridor -/
def maxUncoveredSections (c : CarpetedCorridor) : ℕ :=
  sorry

/-- Theorem stating the maximum number of uncovered sections for the given problem -/
theorem max_uncovered_sections_specific_case :
  let c : CarpetedCorridor := {
    corridorLength := 100,
    numStrips := 20,
    totalStripLength := 1000
  }
  maxUncoveredSections c = 11 :=
by sorry

end NUMINAMATH_CALUDE_max_uncovered_sections_specific_case_l2903_290307


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l2903_290350

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (not_subset : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular_parallel
  (m n l : Line) (α : Plane)
  (distinct : m ≠ n ∧ m ≠ l ∧ n ≠ l)
  (perp_lm : perpendicular l m)
  (not_in_plane : not_subset m α)
  (perp_lα : perpendicular_plane l α) :
  parallel_plane m α :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l2903_290350


namespace NUMINAMATH_CALUDE_quadratic_through_origin_l2903_290300

/-- If the graph of the quadratic function y = mx^2 + x + m(m-3) passes through the origin, then m = 3 -/
theorem quadratic_through_origin (m : ℝ) : 
  (∀ x y : ℝ, y = m * x^2 + x + m * (m - 3)) → 
  (0 = m * 0^2 + 0 + m * (m - 3)) → 
  m = 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_through_origin_l2903_290300


namespace NUMINAMATH_CALUDE_correct_calculation_l2903_290396

theorem correct_calculation (a : ℝ) : 4 * a - (-7 * a) = 11 * a := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2903_290396


namespace NUMINAMATH_CALUDE_diamonds_in_G_15_l2903_290354

-- Define the sequence G
def G : ℕ → ℕ
| 0 => 1  -- G_1 has 1 diamond
| n + 1 => G n + 4 * (n + 2)  -- G_{n+1} adds 4 sides with (n+2) more diamonds each

-- Theorem statement
theorem diamonds_in_G_15 : G 14 = 1849 := by
  sorry

end NUMINAMATH_CALUDE_diamonds_in_G_15_l2903_290354


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l2903_290309

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 3) : x^2 + (1 / x^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l2903_290309


namespace NUMINAMATH_CALUDE_lines_skew_and_parallel_l2903_290361

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_skew_and_parallel (a b c : Line) 
  (h1 : skew a b) (h2 : parallel c a) : skew c b := by
  sorry

end NUMINAMATH_CALUDE_lines_skew_and_parallel_l2903_290361


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_condition_l2903_290331

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem complex_pure_imaginary_condition (a : ℝ) :
  IsPureImaginary ((a^2 - 1) + (a - 1) * Complex.I) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_condition_l2903_290331


namespace NUMINAMATH_CALUDE_max_k_value_l2903_290342

theorem max_k_value (k : ℝ) : (∀ x : ℝ, Real.exp x ≥ k + x) → k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_k_value_l2903_290342


namespace NUMINAMATH_CALUDE_trig_identity_l2903_290346

theorem trig_identity (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin (5*π/6 - x) + (Real.sin (π/3 - x))^2 = 19/16 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2903_290346


namespace NUMINAMATH_CALUDE_patricia_candy_count_l2903_290301

theorem patricia_candy_count (initial_candy : ℕ) (taken_candy : ℕ) : 
  initial_candy = 76 → taken_candy = 5 → initial_candy - taken_candy = 71 := by
  sorry

end NUMINAMATH_CALUDE_patricia_candy_count_l2903_290301


namespace NUMINAMATH_CALUDE_min_value_theorem_l2903_290332

theorem min_value_theorem (x : ℝ) (h : x > 2) : x + 4 / (x - 2) ≥ 6 ∧ (x + 4 / (x - 2) = 6 ↔ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2903_290332


namespace NUMINAMATH_CALUDE_evaluate_expression_l2903_290348

theorem evaluate_expression : (2^2003 * 3^2005) / 6^2004 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2903_290348


namespace NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l2903_290357

/-- The area of a right triangle with base 15 and height 10 is 75 -/
theorem right_triangle_area : Real → Real → Real → Prop :=
  fun base height area =>
    base = 15 ∧ height = 10 ∧ area = (base * height) / 2 → area = 75

theorem right_triangle_area_proof : right_triangle_area 15 10 75 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l2903_290357


namespace NUMINAMATH_CALUDE_total_cash_is_correct_l2903_290392

/-- Calculates the total cash realized from three stocks after accounting for brokerage fees. -/
def total_cash_realized (stock_a_proceeds stock_b_proceeds stock_c_proceeds : ℝ)
                        (stock_a_brokerage_rate stock_b_brokerage_rate stock_c_brokerage_rate : ℝ) : ℝ :=
  let stock_a_cash := stock_a_proceeds * (1 - stock_a_brokerage_rate)
  let stock_b_cash := stock_b_proceeds * (1 - stock_b_brokerage_rate)
  let stock_c_cash := stock_c_proceeds * (1 - stock_c_brokerage_rate)
  stock_a_cash + stock_b_cash + stock_c_cash

/-- Theorem stating that the total cash realized from the given stock sales is equal to 463.578625. -/
theorem total_cash_is_correct : 
  total_cash_realized 107.25 155.40 203.50 (1/400) (1/200) (3/400) = 463.578625 := by
  sorry

end NUMINAMATH_CALUDE_total_cash_is_correct_l2903_290392


namespace NUMINAMATH_CALUDE_melissa_shoe_repair_time_l2903_290377

/-- The time Melissa spends repairing her shoes -/
theorem melissa_shoe_repair_time :
  ∀ (buckle_time heel_time : ℕ) (num_shoes : ℕ),
  buckle_time = 5 →
  heel_time = 10 →
  num_shoes = 2 →
  buckle_time * num_shoes + heel_time * num_shoes = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_melissa_shoe_repair_time_l2903_290377


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2903_290367

theorem algebraic_expression_value (a b : ℝ) (h : a - 3*b = -3) :
  5 - a + 3*b = 8 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2903_290367


namespace NUMINAMATH_CALUDE_tom_weeds_earnings_l2903_290385

/-- Tom's lawn mowing business -/
def tom_lawn_business (weeds_earnings : ℕ) : Prop :=
  let lawns_mowed : ℕ := 3
  let charge_per_lawn : ℕ := 12
  let gas_cost : ℕ := 17
  let total_profit : ℕ := 29
  let mowing_profit : ℕ := lawns_mowed * charge_per_lawn - gas_cost
  weeds_earnings = total_profit - mowing_profit

theorem tom_weeds_earnings : 
  ∃ (weeds_earnings : ℕ), tom_lawn_business weeds_earnings ∧ weeds_earnings = 10 :=
sorry

end NUMINAMATH_CALUDE_tom_weeds_earnings_l2903_290385


namespace NUMINAMATH_CALUDE_stream_speed_l2903_290394

/-- Given a boat that travels at 14 km/hr in still water and covers 72 km downstream in 3.6 hours,
    prove that the speed of the stream is 6 km/hr. -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_speed : ℝ) : 
  boat_speed = 14 →
  distance = 72 →
  time = 3.6 →
  stream_speed = (distance / time) - boat_speed →
  stream_speed = 6 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l2903_290394


namespace NUMINAMATH_CALUDE_max_bouquet_size_l2903_290319

/-- Represents the cost of a yellow tulip in rubles -/
def yellow_cost : ℕ := 50

/-- Represents the cost of a red tulip in rubles -/
def red_cost : ℕ := 31

/-- Represents the maximum budget in rubles -/
def max_budget : ℕ := 600

/-- Represents a valid bouquet of tulips -/
structure Bouquet where
  yellow : ℕ
  red : ℕ
  odd_total : Odd (yellow + red)
  color_diff : (yellow = red + 1) ∨ (red = yellow + 1)
  within_budget : yellow * yellow_cost + red * red_cost ≤ max_budget

/-- The maximum number of tulips in a bouquet -/
def max_tulips : ℕ := 15

/-- Theorem stating that the maximum number of tulips in a valid bouquet is 15 -/
theorem max_bouquet_size :
  ∀ b : Bouquet, b.yellow + b.red ≤ max_tulips ∧
  ∃ b' : Bouquet, b'.yellow + b'.red = max_tulips :=
sorry

end NUMINAMATH_CALUDE_max_bouquet_size_l2903_290319


namespace NUMINAMATH_CALUDE_M_binary_op_result_l2903_290335

def M : Set ℕ := {2, 3}

def binary_op (A : Set ℕ) : Set ℕ := {x | ∃ a b, a ∈ A ∧ b ∈ A ∧ x = a + b}

theorem M_binary_op_result : binary_op M = {4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_M_binary_op_result_l2903_290335


namespace NUMINAMATH_CALUDE_tv_selection_combinations_l2903_290345

def num_type_a : ℕ := 4
def num_type_b : ℕ := 5
def num_to_choose : ℕ := 3

def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem tv_selection_combinations : 
  (combinations num_type_a 2 * combinations num_type_b 1) + 
  (combinations num_type_a 1 * combinations num_type_b 2) = 70 := by
  sorry

end NUMINAMATH_CALUDE_tv_selection_combinations_l2903_290345


namespace NUMINAMATH_CALUDE_train_distance_problem_l2903_290326

theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 16) (h2 : v2 = 21) 
  (h3 : v1 > 0) (h4 : v2 > 0) (h5 : d > 0) : 
  (∃ (t : ℝ), t > 0 ∧ v1 * t + v2 * t = v1 * t + d + v2 * t) → 
  v1 * t + v2 * t = 444 :=
sorry

end NUMINAMATH_CALUDE_train_distance_problem_l2903_290326


namespace NUMINAMATH_CALUDE_day_300_is_tuesday_l2903_290344

/-- If the 26th day of a 366-day year falls on a Monday, then the 300th day of that year falls on a Tuesday. -/
theorem day_300_is_tuesday (year_length : ℕ) (day_26_weekday : ℕ) :
  year_length = 366 →
  day_26_weekday = 1 →
  (300 - 26) % 7 + day_26_weekday ≡ 2 [MOD 7] :=
by sorry

end NUMINAMATH_CALUDE_day_300_is_tuesday_l2903_290344


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l2903_290390

/-- Given a shopkeeper who sells 15 articles at the cost price of 20 articles, 
    prove that the profit percentage is 1/3. -/
theorem shopkeeper_profit_percentage 
  (cost_price : ℝ) (cost_price_positive : cost_price > 0) : 
  let selling_price := 20 * cost_price
  let total_cost := 15 * cost_price
  let profit := selling_price - total_cost
  let profit_percentage := profit / total_cost
  profit_percentage = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l2903_290390


namespace NUMINAMATH_CALUDE_triangle_angle_not_all_less_than_60_l2903_290338

theorem triangle_angle_not_all_less_than_60 : 
  ¬ (∀ (a b c : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) → 
    (a + b + c = 180) → 
    (a < 60 ∧ b < 60 ∧ c < 60)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_not_all_less_than_60_l2903_290338


namespace NUMINAMATH_CALUDE_water_amount_from_reaction_l2903_290380

-- Define the chemical species
inductive ChemicalSpecies
| NaOH
| HClO4
| NaClO4
| H2O

-- Define the reaction equation
def reactionEquation : List (ChemicalSpecies × ℕ) := 
  [(ChemicalSpecies.NaOH, 1), (ChemicalSpecies.HClO4, 1), 
   (ChemicalSpecies.NaClO4, 1), (ChemicalSpecies.H2O, 1)]

-- Define the molar mass of water
def molarMassWater : ℝ := 18.015

-- Define the amount of reactants
def amountNaOH : ℝ := 1
def amountHClO4 : ℝ := 1

-- Theorem statement
theorem water_amount_from_reaction :
  let waterFormed := amountNaOH * molarMassWater
  waterFormed = 18.015 := by sorry

end NUMINAMATH_CALUDE_water_amount_from_reaction_l2903_290380


namespace NUMINAMATH_CALUDE_choir_size_proof_choir_size_minimum_l2903_290391

theorem choir_size_proof : 
  ∀ n : ℕ, (n % 9 = 0 ∧ n % 11 = 0 ∧ n % 13 = 0 ∧ n % 10 = 0) → n ≥ 12870 :=
by
  sorry

theorem choir_size_minimum : 
  12870 % 9 = 0 ∧ 12870 % 11 = 0 ∧ 12870 % 13 = 0 ∧ 12870 % 10 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_choir_size_proof_choir_size_minimum_l2903_290391


namespace NUMINAMATH_CALUDE_davids_math_marks_l2903_290356

def english_marks : ℝ := 90
def physics_marks : ℝ := 85
def chemistry_marks : ℝ := 87
def biology_marks : ℝ := 85
def average_marks : ℝ := 87.8
def total_subjects : ℕ := 5

theorem davids_math_marks :
  ∃ (math_marks : ℝ),
    (english_marks + physics_marks + chemistry_marks + biology_marks + math_marks) / total_subjects = average_marks ∧
    math_marks = 92 := by
  sorry

end NUMINAMATH_CALUDE_davids_math_marks_l2903_290356


namespace NUMINAMATH_CALUDE_evaluate_expression_l2903_290305

theorem evaluate_expression (x : ℝ) (hx : x ≠ 0) :
  (20 * x^3) * (8 * x^2) * (1 / (4*x)^3) = (5/2) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2903_290305


namespace NUMINAMATH_CALUDE_pet_food_difference_l2903_290337

theorem pet_food_difference (dog_food cat_food : ℕ) 
  (h1 : dog_food = 600) (h2 : cat_food = 327) : 
  dog_food - cat_food = 273 := by
  sorry

end NUMINAMATH_CALUDE_pet_food_difference_l2903_290337


namespace NUMINAMATH_CALUDE_current_rate_calculation_l2903_290347

theorem current_rate_calculation (boat_speed : ℝ) (distance : ℝ) (time : ℝ) :
  boat_speed = 21 →
  distance = 6.283333333333333 →
  time = 13 / 60 →
  ∃ current_rate : ℝ, 
    distance = (boat_speed + current_rate) * time ∧
    current_rate = 8 := by
  sorry

end NUMINAMATH_CALUDE_current_rate_calculation_l2903_290347


namespace NUMINAMATH_CALUDE_platform_length_l2903_290372

/-- Given a train of length 300 meters that crosses a platform in 45 seconds
    and crosses a signal pole in 18 seconds, prove that the platform length is 450 meters. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 45)
  (h3 : pole_crossing_time = 18) :
  let train_speed := train_length / pole_crossing_time
  let total_distance := train_speed * platform_crossing_time
  train_length + (total_distance - train_length) = 450 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l2903_290372


namespace NUMINAMATH_CALUDE_ratio_problem_l2903_290339

theorem ratio_problem (first_number second_number : ℚ) : 
  (first_number / second_number = 15) → 
  (first_number = 150) → 
  (second_number = 10) := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2903_290339


namespace NUMINAMATH_CALUDE_women_not_french_approx_97_14_percent_l2903_290334

/-- Represents the composition of employees in a company -/
structure Company where
  total_employees : ℕ
  men_percentage : ℚ
  men_french_percentage : ℚ
  total_french_percentage : ℚ

/-- Calculates the percentage of women who do not speak French in the company -/
def women_not_french_percentage (c : Company) : ℚ :=
  let women_percentage := 1 - c.men_percentage
  let men_french := c.men_percentage * c.men_french_percentage
  let women_french := c.total_french_percentage - men_french
  let women_not_french := women_percentage - women_french
  women_not_french / women_percentage

/-- Theorem stating that for a company with the given percentages,
    the percentage of women who do not speak French is approximately 97.14% -/
theorem women_not_french_approx_97_14_percent 
  (c : Company) 
  (h1 : c.men_percentage = 65/100)
  (h2 : c.men_french_percentage = 60/100)
  (h3 : c.total_french_percentage = 40/100) :
  ∃ ε > 0, |women_not_french_percentage c - 9714/10000| < ε :=
sorry

end NUMINAMATH_CALUDE_women_not_french_approx_97_14_percent_l2903_290334


namespace NUMINAMATH_CALUDE_find_k_value_l2903_290395

/-- Given two functions f and g, prove that k = -15.8 when f(5) - g(5) = 15 -/
theorem find_k_value (f g : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = 5 * x^2 - 3 * x + 6) → 
  (∀ x, g x = 2 * x^2 - k * x + 2) → 
  f 5 - g 5 = 15 → 
  k = -15.8 := by
sorry

end NUMINAMATH_CALUDE_find_k_value_l2903_290395


namespace NUMINAMATH_CALUDE_equal_debt_after_calculated_days_l2903_290312

/-- The number of days until Darren and Fergie owe the same amount -/
def days_until_equal_debt : ℝ := 53.75

/-- Darren's initial borrowed amount -/
def darren_initial_borrowed : ℝ := 200

/-- Fergie's initial borrowed amount -/
def fergie_initial_borrowed : ℝ := 300

/-- Darren's initial daily interest rate -/
def darren_initial_rate : ℝ := 0.08

/-- Darren's reduced daily interest rate after 10 days -/
def darren_reduced_rate : ℝ := 0.06

/-- Fergie's daily interest rate -/
def fergie_rate : ℝ := 0.04

/-- The number of days after which Darren's interest rate changes -/
def rate_change_days : ℝ := 10

/-- Theorem stating that Darren and Fergie owe the same amount after the calculated number of days -/
theorem equal_debt_after_calculated_days :
  let darren_debt := if days_until_equal_debt ≤ rate_change_days
    then darren_initial_borrowed * (1 + darren_initial_rate * days_until_equal_debt)
    else darren_initial_borrowed * (1 + darren_initial_rate * rate_change_days) *
      (1 + darren_reduced_rate * (days_until_equal_debt - rate_change_days))
  let fergie_debt := fergie_initial_borrowed * (1 + fergie_rate * days_until_equal_debt)
  darren_debt = fergie_debt := by sorry


end NUMINAMATH_CALUDE_equal_debt_after_calculated_days_l2903_290312


namespace NUMINAMATH_CALUDE_expression_value_l2903_290351

theorem expression_value (x y : ℝ) (h : x - 2*y = 3) : 1 - 2*x + 4*y = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2903_290351


namespace NUMINAMATH_CALUDE_adam_purchases_cost_l2903_290389

/-- Represents the cost of Adam's purchases -/
def total_cost (nuts_quantity : ℝ) (dried_fruits_quantity : ℝ) (nuts_price : ℝ) (dried_fruits_price : ℝ) : ℝ :=
  nuts_quantity * nuts_price + dried_fruits_quantity * dried_fruits_price

/-- Theorem stating that Adam's purchases cost $56 -/
theorem adam_purchases_cost :
  total_cost 3 2.5 12 8 = 56 := by
  sorry

end NUMINAMATH_CALUDE_adam_purchases_cost_l2903_290389


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_20_l2903_290382

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem largest_four_digit_sum_20 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 20 → n ≤ 9920 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_20_l2903_290382


namespace NUMINAMATH_CALUDE_beef_weight_is_fifteen_l2903_290329

/-- The number of ounces in a pound -/
def ounces_per_pound : ℕ := 16

/-- The weight of each steak in ounces -/
def steak_weight : ℕ := 12

/-- The number of steaks Matt gets from the beef -/
def number_of_steaks : ℕ := 20

/-- The total weight of beef in pounds -/
def total_beef_weight : ℚ :=
  (steak_weight * number_of_steaks : ℚ) / ounces_per_pound

theorem beef_weight_is_fifteen :
  total_beef_weight = 15 := by sorry

end NUMINAMATH_CALUDE_beef_weight_is_fifteen_l2903_290329


namespace NUMINAMATH_CALUDE_new_supervisor_salary_range_l2903_290387

theorem new_supervisor_salary_range (
  old_average : ℝ) 
  (old_supervisor_salary : ℝ) 
  (new_average : ℝ) 
  (min_worker_salary : ℝ) 
  (max_worker_salary : ℝ) 
  (min_supervisor_salary : ℝ) 
  (max_supervisor_salary : ℝ) :
  old_average = 430 →
  old_supervisor_salary = 870 →
  new_average = 410 →
  min_worker_salary = 300 →
  max_worker_salary = 500 →
  min_supervisor_salary = 800 →
  max_supervisor_salary = 1100 →
  ∃ (new_supervisor_salary : ℝ),
    min_supervisor_salary ≤ new_supervisor_salary ∧
    new_supervisor_salary ≤ max_supervisor_salary ∧
    (9 * new_average - 8 * old_average + old_supervisor_salary = new_supervisor_salary) :=
by sorry

end NUMINAMATH_CALUDE_new_supervisor_salary_range_l2903_290387


namespace NUMINAMATH_CALUDE_expression_evaluation_l2903_290340

theorem expression_evaluation : (3 * 4 * 6) * (1/3 + 1/4 + 1/6) = 54 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2903_290340


namespace NUMINAMATH_CALUDE_new_oranges_added_l2903_290360

/-- Calculates the number of new oranges added to a bin -/
def new_oranges (initial : ℕ) (thrown_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial - thrown_away)

/-- Proves that the number of new oranges added is 28 -/
theorem new_oranges_added : new_oranges 5 2 31 = 28 := by
  sorry

end NUMINAMATH_CALUDE_new_oranges_added_l2903_290360


namespace NUMINAMATH_CALUDE_two_thirds_squared_l2903_290317

theorem two_thirds_squared : (2 / 3 : ℚ) ^ 2 = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_squared_l2903_290317


namespace NUMINAMATH_CALUDE_det_of_specific_matrix_l2903_290311

/-- The determinant of the matrix [5 -2; 4 3] is 23. -/
theorem det_of_specific_matrix :
  let M : Matrix (Fin 2) (Fin 2) ℤ := !![5, -2; 4, 3]
  Matrix.det M = 23 := by
  sorry

end NUMINAMATH_CALUDE_det_of_specific_matrix_l2903_290311


namespace NUMINAMATH_CALUDE_good_numbers_characterization_l2903_290341

/-- A number n > 3 is 'good' if the set of weights {1, 2, 3, ..., n} can be divided into three piles of equal mass -/
def is_good (n : ℕ) : Prop :=
  n > 3 ∧ ∃ (a b c : Finset ℕ), a ∪ b ∪ c = Finset.range n ∧ 
    a ∩ b = ∅ ∧ a ∩ c = ∅ ∧ b ∩ c = ∅ ∧
    a.sum id = b.sum id ∧ b.sum id = c.sum id

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem good_numbers_characterization (n : ℕ) :
  is_good n ↔ (∃ k : ℕ, k ≥ 1 ∧ (n = 3 * k ∨ n = 3 * k + 2)) :=
sorry

end NUMINAMATH_CALUDE_good_numbers_characterization_l2903_290341


namespace NUMINAMATH_CALUDE_sin_210_degrees_l2903_290398

theorem sin_210_degrees :
  Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l2903_290398


namespace NUMINAMATH_CALUDE_cloth_cost_price_l2903_290363

def cloth_problem (total_meters : ℝ) (total_price : ℝ) (loss_per_meter : ℝ) (discount_rate : ℝ) : Prop :=
  let selling_price_per_meter : ℝ := total_price / total_meters
  let discounted_price_per_meter : ℝ := selling_price_per_meter * (1 - discount_rate)
  let cost_price_per_meter : ℝ := discounted_price_per_meter + loss_per_meter
  cost_price_per_meter = 130

theorem cloth_cost_price :
  cloth_problem 450 45000 40 0.1 :=
by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l2903_290363


namespace NUMINAMATH_CALUDE_kevin_cards_total_l2903_290381

/-- Given that Kevin starts with 7 cards and finds 47 more, prove that he ends up with 54 cards in total. -/
theorem kevin_cards_total : 
  let initial_cards : ℕ := 7
  let found_cards : ℕ := 47
  initial_cards + found_cards = 54 := by sorry

end NUMINAMATH_CALUDE_kevin_cards_total_l2903_290381


namespace NUMINAMATH_CALUDE_steve_final_marbles_l2903_290383

/- Define the initial number of marbles for each person -/
def sam_initial : ℕ := 14
def steve_initial : ℕ := 7
def sally_initial : ℕ := 9

/- Define the number of marbles Sam gives away -/
def marbles_given : ℕ := 3

/- Define Sam's final number of marbles -/
def sam_final : ℕ := 8

/- Theorem to prove -/
theorem steve_final_marbles :
  /- Conditions -/
  (sam_initial = 2 * steve_initial) →
  (sally_initial = sam_initial - 5) →
  (sam_final = sam_initial - 2 * marbles_given) →
  /- Conclusion -/
  (steve_initial + marbles_given = 10) := by
sorry

end NUMINAMATH_CALUDE_steve_final_marbles_l2903_290383


namespace NUMINAMATH_CALUDE_triangle_theorem_l2903_290310

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c - t.b = 2 * t.b * Real.cos t.A ∧
  Real.cos t.B = 3/4 ∧
  t.c = 5

-- Theorem to prove
theorem triangle_theorem (t : Triangle) 
  (h : triangle_conditions t) : 
  t.A = 2 * t.B ∧ 
  (1/2 * t.b * t.c * Real.sin t.A) = (15/4) * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2903_290310


namespace NUMINAMATH_CALUDE_class_height_ratio_l2903_290397

theorem class_height_ratio :
  ∀ (x y : ℕ),
  x > 0 → y > 0 →
  149 * x + 144 * y = 147 * (x + y) →
  (x : ℚ) / y = 3 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_class_height_ratio_l2903_290397


namespace NUMINAMATH_CALUDE_ellipse_equation_l2903_290399

/-- An ellipse with center at the origin, foci on the x-axis, eccentricity 1/2, 
    and the perimeter of triangle PF₁F₂ equal to 12 -/
structure Ellipse where
  /-- The eccentricity of the ellipse -/
  e : ℝ
  /-- The perimeter of triangle PF₁F₂ -/
  perimeter : ℝ
  /-- The eccentricity is 1/2 -/
  h_e : e = 1/2
  /-- The perimeter is 12 -/
  h_perimeter : perimeter = 12

/-- The standard equation of the ellipse -/
def standardEquation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2/16 + y^2/12 = 1

/-- Theorem stating that the given ellipse satisfies the standard equation -/
theorem ellipse_equation (E : Ellipse) (x y : ℝ) : 
  standardEquation E x y := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2903_290399


namespace NUMINAMATH_CALUDE_tan_difference_l2903_290393

theorem tan_difference (α β : Real) 
  (h1 : Real.tan (α + β) = 1/2) 
  (h2 : Real.tan (α + Real.pi/4) = -1/3) : 
  Real.tan (β - Real.pi/4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_l2903_290393


namespace NUMINAMATH_CALUDE_black_ants_count_l2903_290314

theorem black_ants_count (total_ants red_ants : ℕ) 
  (h1 : total_ants = 900) 
  (h2 : red_ants = 413) : 
  total_ants - red_ants = 487 := by
  sorry

end NUMINAMATH_CALUDE_black_ants_count_l2903_290314


namespace NUMINAMATH_CALUDE_range_of_m_solution_set_correct_l2903_290378

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (2*m + 1) * x + 2

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  (∃ x y, x > 1 ∧ y < 1 ∧ f m x = 0 ∧ f m y = 0) → -1 < m ∧ m < 0 :=
sorry

-- Define the solution set for f(x) ≤ 0
def solution_set (m : ℝ) : Set ℝ :=
  if m = 0 then { x | x ≤ -2 }
  else if m < 0 then { x | -2 ≤ x ∧ x ≤ -1/m }
  else if 0 < m ∧ m < 1/2 then { x | -1/m ≤ x ∧ x ≤ -2 }
  else if m = 1/2 then { -2 }
  else { x | -2 ≤ x ∧ x ≤ -1/m }

-- Theorem for the solution set
theorem solution_set_correct (m : ℝ) (x : ℝ) : 
  x ∈ solution_set m ↔ f m x ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_solution_set_correct_l2903_290378


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2903_290365

theorem quadratic_inequality_solution_set (a : ℝ) :
  {x : ℝ | x^2 - (2*a + 1)*x + a^2 + a < 0} = Set.Ioo a (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2903_290365


namespace NUMINAMATH_CALUDE_number_problem_l2903_290333

theorem number_problem (x : ℝ) : 
  (0.25 * x = 0.20 * 650 + 190) → x = 1280 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l2903_290333


namespace NUMINAMATH_CALUDE_correct_regression_coefficients_l2903_290358

-- Define the linear regression equation
def linear_regression (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- Define positive correlation
def positively_correlated (a : ℝ) : Prop := a > 0

-- Define the sample means
def x_mean : ℝ := 3
def y_mean : ℝ := 3.5

-- Theorem statement
theorem correct_regression_coefficients (a b : ℝ) :
  positively_correlated a ∧
  linear_regression a b x_mean = y_mean →
  a = 0.4 ∧ b = 2.3 :=
by sorry

end NUMINAMATH_CALUDE_correct_regression_coefficients_l2903_290358


namespace NUMINAMATH_CALUDE_parabola_directrix_l2903_290308

/-- The equation of the directrix of the parabola y = -1/8 * x^2 is y = 2 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = -1/8 * x^2) → (∃ p : ℝ, p = 4 ∧ y = p/2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2903_290308


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2903_290370

def P : Set ℝ := {x | x^2 + x - 6 = 0}
def Q (a : ℝ) : Set ℝ := {x | a*x + 1 = 0}

theorem possible_values_of_a :
  {a : ℝ | Q a ⊆ P} = {0, 1/3, -1/2} := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2903_290370


namespace NUMINAMATH_CALUDE_distance_A_to_y_axis_l2903_290376

/-- The distance from a point to the y-axis is the absolute value of its x-coordinate -/
def distanceToYAxis (x : ℝ) (y : ℝ) : ℝ := |x|

/-- Point A has coordinates (2, -3) -/
def pointA : ℝ × ℝ := (2, -3)

/-- Theorem: The distance from point A(2, -3) to the y-axis is 2 -/
theorem distance_A_to_y_axis :
  distanceToYAxis pointA.1 pointA.2 = 2 := by sorry

end NUMINAMATH_CALUDE_distance_A_to_y_axis_l2903_290376


namespace NUMINAMATH_CALUDE_no_real_roots_l2903_290355

theorem no_real_roots : ∀ x : ℝ, (x + 1) * |x + 1| - x * |x| + 1 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l2903_290355
