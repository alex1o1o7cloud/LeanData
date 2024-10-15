import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l2749_274905

/-- The range of values for the real number a such that at least one of the given quadratic equations has real roots -/
theorem quadratic_real_roots_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + a^2 = 0 ∨ x^2 + 2*a*x - 2*a = 0) ↔ 
  a ≤ -2 ∨ (-1/3 ≤ a ∧ a < 1) ∨ 0 ≤ a := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l2749_274905


namespace NUMINAMATH_CALUDE_division_problem_l2749_274927

theorem division_problem (number : ℕ) : 
  (number / 179 = 89) ∧ (number % 179 = 37) → number = 15968 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2749_274927


namespace NUMINAMATH_CALUDE_y_intercept_of_specific_line_l2749_274976

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ := sorry

/-- Given a line with slope 3 and x-intercept (-3, 0), its y-intercept is (0, 9). -/
theorem y_intercept_of_specific_line :
  let l : Line := { slope := 3, x_intercept := (-3, 0) }
  y_intercept l = (0, 9) := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_specific_line_l2749_274976


namespace NUMINAMATH_CALUDE_polynomial_coefficient_property_l2749_274949

theorem polynomial_coefficient_property (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_property_l2749_274949


namespace NUMINAMATH_CALUDE_rectangle_area_l2749_274982

/-- Given a rectangle with perimeter 28 cm and width 6 cm, its area is 48 square centimeters. -/
theorem rectangle_area (perimeter width : ℝ) (h_perimeter : perimeter = 28) (h_width : width = 6) :
  let length := (perimeter - 2 * width) / 2
  width * length = 48 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2749_274982


namespace NUMINAMATH_CALUDE_numerica_base_l2749_274916

/-- Convert a number from base r to base 10 -/
def to_base_10 (digits : List Nat) (r : Nat) : Nat :=
  digits.foldr (fun d acc => d + r * acc) 0

/-- The base r used in Numerica -/
def r : Nat := sorry

/-- The price of the gadget in base r -/
def price : List Nat := [5, 3, 0]

/-- The payment made in base r -/
def payment : List Nat := [1, 1, 0, 0]

/-- The change received in base r -/
def change : List Nat := [4, 6, 0]

theorem numerica_base :
  (to_base_10 price r + to_base_10 change r = to_base_10 payment r) ↔ r = 9 := by
  sorry

end NUMINAMATH_CALUDE_numerica_base_l2749_274916


namespace NUMINAMATH_CALUDE_max_min_x2_xy_y2_l2749_274945

theorem max_min_x2_xy_y2 (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (A_min A_max : ℝ), A_min = 2 ∧ A_max = 6 ∧
  ∀ A, A = x^2 + x*y + y^2 → A_min ≤ A ∧ A ≤ A_max :=
by sorry

end NUMINAMATH_CALUDE_max_min_x2_xy_y2_l2749_274945


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l2749_274973

/-- A cuboid with three distinct side areas -/
structure Cuboid where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ

/-- The total surface area of a cuboid -/
def surface_area (c : Cuboid) : ℝ := 2 * (c.area1 + c.area2 + c.area3)

/-- Theorem: The surface area of a cuboid with side areas 4, 3, and 6 is 26 -/
theorem cuboid_surface_area :
  let c : Cuboid := { area1 := 4, area2 := 3, area3 := 6 }
  surface_area c = 26 := by
  sorry

#check cuboid_surface_area

end NUMINAMATH_CALUDE_cuboid_surface_area_l2749_274973


namespace NUMINAMATH_CALUDE_find_adult_ticket_cost_l2749_274922

def adult_ticket_cost (total_cost children_cost : ℕ) : Prop :=
  ∃ (adult_cost : ℕ), adult_cost + 6 * children_cost = total_cost

theorem find_adult_ticket_cost :
  adult_ticket_cost 155 20 → ∃ (adult_cost : ℕ), adult_cost = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_find_adult_ticket_cost_l2749_274922


namespace NUMINAMATH_CALUDE_classroom_setup_l2749_274962

/-- Represents the number of desks in a classroom setup for an exam. -/
def num_desks : ℕ := 33

/-- Represents the number of chairs per desk. -/
def chairs_per_desk : ℕ := 4

/-- Represents the number of legs per chair. -/
def legs_per_chair : ℕ := 4

/-- Represents the number of legs per desk. -/
def legs_per_desk : ℕ := 6

/-- Represents the total number of legs from all desks and chairs. -/
def total_legs : ℕ := 728

theorem classroom_setup :
  num_desks * chairs_per_desk * legs_per_chair + num_desks * legs_per_desk = total_legs :=
by sorry

end NUMINAMATH_CALUDE_classroom_setup_l2749_274962


namespace NUMINAMATH_CALUDE_cashier_money_value_l2749_274920

def total_bills : ℕ := 30
def ten_dollar_bills : ℕ := 27
def twenty_dollar_bills : ℕ := 3
def ten_dollar_value : ℕ := 10
def twenty_dollar_value : ℕ := 20

theorem cashier_money_value :
  ten_dollar_bills + twenty_dollar_bills = total_bills →
  ten_dollar_bills * ten_dollar_value + twenty_dollar_bills * twenty_dollar_value = 330 :=
by
  sorry

end NUMINAMATH_CALUDE_cashier_money_value_l2749_274920


namespace NUMINAMATH_CALUDE_locus_is_circle_l2749_274917

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  s : ℝ
  a : Point
  b : Point
  c : Point

/-- The sum of squares of distances from a point to the vertices of a triangle -/
def sumOfSquaredDistances (p : Point) (t : IsoscelesRightTriangle) : ℝ :=
  (p.x - t.a.x)^2 + (p.y - t.a.y)^2 +
  (p.x - t.b.x)^2 + (p.y - t.b.y)^2 +
  (p.x - t.c.x)^2 + (p.y - t.c.y)^2

/-- The locus of points P such that the sum of squares of distances from P to the vertices is less than 2s^2 -/
def locus (t : IsoscelesRightTriangle) : Set Point :=
  {p : Point | sumOfSquaredDistances p t < 2 * t.s^2}

theorem locus_is_circle (t : IsoscelesRightTriangle) :
  locus t = {p : Point | (p.x - t.s/3)^2 + (p.y - t.s/3)^2 < (2*t.s/3)^2} :=
sorry

end NUMINAMATH_CALUDE_locus_is_circle_l2749_274917


namespace NUMINAMATH_CALUDE_complementary_event_equivalence_l2749_274969

/-- The number of products in the sample -/
def sample_size : ℕ := 10

/-- Event A: at least 2 defective products -/
def event_A (defective : ℕ) : Prop := defective ≥ 2

/-- Complementary event of A -/
def comp_A (defective : ℕ) : Prop := ¬(event_A defective)

/-- At most 1 defective product -/
def at_most_one_defective (defective : ℕ) : Prop := defective ≤ 1

/-- At least 2 non-defective products -/
def at_least_two_non_defective (defective : ℕ) : Prop := sample_size - defective ≥ 2

theorem complementary_event_equivalence :
  ∀ defective : ℕ, defective ≤ sample_size →
    (comp_A defective ↔ at_most_one_defective defective) ∧
    (comp_A defective ↔ at_least_two_non_defective defective) :=
by sorry

end NUMINAMATH_CALUDE_complementary_event_equivalence_l2749_274969


namespace NUMINAMATH_CALUDE_parallelogram_height_relation_crosswalk_problem_l2749_274933

/-- Given a parallelogram with sides a, b, height h_a perpendicular to side a,
    prove that the height h_b perpendicular to side b is (a * h_a) / b -/
theorem parallelogram_height_relation (a b h_a h_b : ℝ) 
    (ha : a > 0) (hb : b > 0) (hha : h_a > 0) (hhb : h_b > 0) :
  a * h_a = b * h_b :=
by sorry

/-- Prove that for a parallelogram with a = 25, b = 50, and h_a = 60,
    the height h_b perpendicular to side b is 30 -/
theorem crosswalk_problem (a b h_a h_b : ℝ) 
    (ha : a = 25) (hb : b = 50) (hha : h_a = 60) :
  h_b = 30 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_height_relation_crosswalk_problem_l2749_274933


namespace NUMINAMATH_CALUDE_quartic_root_ratio_l2749_274952

theorem quartic_root_ratio (a b c d e : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) →
  d / e = -25 / 12 := by
sorry

end NUMINAMATH_CALUDE_quartic_root_ratio_l2749_274952


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2749_274908

theorem complex_equation_solution (z : ℂ) : z + Complex.abs z = 1 + Complex.I → z = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2749_274908


namespace NUMINAMATH_CALUDE_square_root_of_one_fourth_l2749_274981

theorem square_root_of_one_fourth : 
  {x : ℝ | x^2 = (1/4 : ℝ)} = {-(1/2 : ℝ), (1/2 : ℝ)} := by sorry

end NUMINAMATH_CALUDE_square_root_of_one_fourth_l2749_274981


namespace NUMINAMATH_CALUDE_sin_product_equals_sqrt5_minus_1_over_32_sin_cos_ratio_equals_neg_sqrt2_l2749_274932

-- Part 1
theorem sin_product_equals_sqrt5_minus_1_over_32 :
  Real.sin (6 * π / 180) * Real.sin (42 * π / 180) * Real.sin (66 * π / 180) * Real.sin (78 * π / 180) = 
  (Real.sqrt 5 - 1) / 32 := by sorry

-- Part 2
theorem sin_cos_ratio_equals_neg_sqrt2 (α : Real) 
  (h1 : π / 2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : Real.sin α = Real.sqrt 15 / 4) :
  (Real.sin (α + π / 4)) / (Real.sin (2 * α) + Real.cos (2 * α) + 1) = -Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sin_product_equals_sqrt5_minus_1_over_32_sin_cos_ratio_equals_neg_sqrt2_l2749_274932


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l2749_274958

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 11

-- Define the point of tangency
def P : ℝ × ℝ := (1, 12)

-- Theorem statement
theorem tangent_line_y_intercept :
  let df (x : ℝ) := 3 * x^2  -- Derivative of f
  let m : ℝ := df P.1        -- Slope of the tangent line
  let b : ℝ := P.2 - m * P.1 -- y-intercept of the tangent line
  b = 9 := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l2749_274958


namespace NUMINAMATH_CALUDE_fraction_simplification_l2749_274964

theorem fraction_simplification : 
  let numerator := (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400)
  let denominator := (6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400)
  ∀ x : ℕ, x^4 + 400 = (x^2 - 10*x + 20) * (x^2 + 10*x + 20) →
  numerator / denominator = 995 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2749_274964


namespace NUMINAMATH_CALUDE_right_triangle_line_equation_l2749_274971

/-- Given a right triangle in the first quadrant with vertices at (0, 0), (a, 0), and (0, b),
    where the area of the triangle is T, prove that the equation of the line passing through
    (0, b) and (a, 0) in its standard form is 2Tx - a²y + 2Ta = 0. -/
theorem right_triangle_line_equation (a b T : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : T = (1/2) * a * b) :
  ∃ (A B C : ℝ), A * a + B * b + C = 0 ∧ 
                 (∀ x y : ℝ, A * x + B * y + C = 0 ↔ 2 * T * x - a^2 * y + 2 * T * a = 0) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_line_equation_l2749_274971


namespace NUMINAMATH_CALUDE_hexagonal_prism_volume_l2749_274940

-- Define the hexagonal prism
structure HexagonalPrism where
  sideEdgeLength : ℝ
  lateralSurfaceAreaQuadPrism : ℝ

-- Define the theorem
theorem hexagonal_prism_volume 
  (prism : HexagonalPrism)
  (h1 : prism.sideEdgeLength = 3)
  (h2 : prism.lateralSurfaceAreaQuadPrism = 30) :
  ∃ (volume : ℝ), volume = 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_prism_volume_l2749_274940


namespace NUMINAMATH_CALUDE_root_sum_ratio_l2749_274918

theorem root_sum_ratio (m₁ m₂ : ℝ) : 
  (∃ p q : ℝ, 
    (∀ m : ℝ, m * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m * (q^2 - 3*q) + 2*q + 7 = 0) ∧
    p / q + q / p = 2 ∧
    (m₁ * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m₁ * (q^2 - 3*q) + 2*q + 7 = 0) ∧
    (m₂ * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m₂ * (q^2 - 3*q) + 2*q + 7 = 0)) →
  m₁ / m₂ + m₂ / m₁ = 136 / 9 := by
sorry

end NUMINAMATH_CALUDE_root_sum_ratio_l2749_274918


namespace NUMINAMATH_CALUDE_expression_values_l2749_274914

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  expr = 5 ∨ expr = 1 ∨ expr = -3 :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l2749_274914


namespace NUMINAMATH_CALUDE_cubic_polynomial_problem_l2749_274906

-- Define the cubic polynomial whose roots are a, b, c
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 2*x + 3

-- Define the properties of P
def is_valid_P (P : ℝ → ℝ) (a b c : ℝ) : Prop :=
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  P a = b + c ∧ P b = a + c ∧ P c = a + b ∧
  P (a + b + c) = -20

-- The theorem to prove
theorem cubic_polynomial_problem :
  ∃ (P : ℝ → ℝ) (a b c : ℝ),
    is_valid_P P a b c ∧
    (∀ x, P x = -17/3 * x^3 + 68/3 * x^2 - 31/3 * x - 18) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_problem_l2749_274906


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l2749_274957

theorem inequality_proof (a b c d : ℝ) 
  (ha : 1 ≤ a ∧ a ≤ 2) 
  (hb : 1 ≤ b ∧ b ≤ 2) 
  (hc : 1 ≤ c ∧ c ≤ 2) 
  (hd : 1 ≤ d ∧ d ≤ 2) : 
  (a + b) / (b + c) + (c + d) / (d + a) ≤ 4 * (a + c) / (b + d) :=
sorry

theorem equality_condition (a b c d : ℝ) 
  (ha : 1 ≤ a ∧ a ≤ 2) 
  (hb : 1 ≤ b ∧ b ≤ 2) 
  (hc : 1 ≤ c ∧ c ≤ 2) 
  (hd : 1 ≤ d ∧ d ≤ 2) :
  (a + b) / (b + c) + (c + d) / (d + a) = 4 * (a + c) / (b + d) ↔ a = b ∧ b = c ∧ c = d :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l2749_274957


namespace NUMINAMATH_CALUDE_sixtieth_term_of_arithmetic_sequence_l2749_274963

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence as a function from ℕ to ℚ
  d : ℚ      -- The common difference
  h : ∀ n, a (n + 1) = a n + d

/-- Given an arithmetic sequence with a₁ = 7 and a₁₅ = 37,
    prove that a₆₀ = 134.5 -/
theorem sixtieth_term_of_arithmetic_sequence
  (seq : ArithmeticSequence)
  (h1 : seq.a 1 = 7)
  (h15 : seq.a 15 = 37) :
  seq.a 60 = 134.5 := by
  sorry

end NUMINAMATH_CALUDE_sixtieth_term_of_arithmetic_sequence_l2749_274963


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l2749_274944

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 1024 ways to distribute 5 distinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes :
  distribute 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l2749_274944


namespace NUMINAMATH_CALUDE_unique_quadratic_root_l2749_274910

theorem unique_quadratic_root (a : ℝ) : 
  (∃! x : ℝ, (a^2 - 1) * x^2 + (a + 1) * x + 1 = 0) ↔ (a = 1 ∨ a = 5/3) := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_root_l2749_274910


namespace NUMINAMATH_CALUDE_sum_product_ratio_l2749_274995

theorem sum_product_ratio (x y z : ℝ) (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) (hsum : x + y + z = 1) :
  (x*y + y*z + z*x) / (x^2 + y^2 + z^2) = (1 - (x^2 + y^2 + z^2)) / (2*(x^2 + y^2 + z^2)) := by
  sorry

end NUMINAMATH_CALUDE_sum_product_ratio_l2749_274995


namespace NUMINAMATH_CALUDE_sum_of_roots_l2749_274999

theorem sum_of_roots (h : ℝ) (x₁ x₂ : ℝ) :
  x₁ ≠ x₂ →
  (6 * x₁^2 - 5 * h * x₁ - 4 * h = 0) →
  (6 * x₂^2 - 5 * h * x₂ - 4 * h = 0) →
  x₁ + x₂ = 5 * h / 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2749_274999


namespace NUMINAMATH_CALUDE_divisor_problem_l2749_274942

theorem divisor_problem (k : ℕ) : 12^k ∣ 856736 → k = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l2749_274942


namespace NUMINAMATH_CALUDE_opposite_numbers_l2749_274911

theorem opposite_numbers : -3 = -(Real.sqrt ((-3)^2)) := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_l2749_274911


namespace NUMINAMATH_CALUDE_triangle_inequality_l2749_274915

theorem triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b + c = 1) : a^2 * c + b^2 * a + c^2 * b < 1/8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2749_274915


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l2749_274967

theorem complex_number_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - 2 * w) = 30)
  (h2 : Complex.abs (z + 2 * w) = 5)
  (h3 : Complex.abs (z + w) = 2) :
  Complex.abs z = Real.sqrt (19 / 8) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l2749_274967


namespace NUMINAMATH_CALUDE_original_bacteria_count_l2749_274983

theorem original_bacteria_count (current : ℕ) (increase : ℕ) (original : ℕ)
  (h1 : current = 8917)
  (h2 : increase = 8317)
  (h3 : current = original + increase) :
  original = 600 := by
  sorry

end NUMINAMATH_CALUDE_original_bacteria_count_l2749_274983


namespace NUMINAMATH_CALUDE_triangle_projection_inequality_l2749_274989

/-- Given a triangle ABC with sides a, b, c and projections satisfying certain conditions,
    prove that a specific inequality holds. -/
theorem triangle_projection_inequality 
  (a b c : ℝ) 
  (t r μ : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_t : ∃ (A C₁ : ℝ), A > 0 ∧ C₁ > 0 ∧ C₁ = 2 * t * c) 
  (h_r : ∃ (B A₁ : ℝ), B > 0 ∧ A₁ > 0 ∧ A₁ = 2 * r * a) 
  (h_μ : ∃ (C B₁ : ℝ), C > 0 ∧ B₁ > 0 ∧ B₁ = 2 * μ * b) :
  (a^2 / b^2) * (t / (1 - 2*t))^2 + 
  (b^2 / c^2) * (r / (1 - 2*r))^2 + 
  (c^2 / a^2) * (μ / (1 - 2*μ))^2 + 
  16 * t * r * μ ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_projection_inequality_l2749_274989


namespace NUMINAMATH_CALUDE_unique_base_solution_l2749_274938

/-- Converts a base-10 number to base-a representation --/
def toBaseA (n : ℕ) (a : ℕ) : List ℕ :=
  sorry

/-- Converts a base-a number (represented as a list of digits) to base-10 --/
def fromBaseA (digits : List ℕ) (a : ℕ) : ℕ :=
  sorry

/-- Checks if the equation 452_a + 127_a = 5B0_a holds for a given base a --/
def checkEquation (a : ℕ) : Prop :=
  fromBaseA (toBaseA 452 a) a + fromBaseA (toBaseA 127 a) a = 
  fromBaseA ([5, 11, 0]) a

theorem unique_base_solution :
  ∃! a : ℕ, a > 11 ∧ checkEquation a ∧ fromBaseA ([11]) a = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_base_solution_l2749_274938


namespace NUMINAMATH_CALUDE_robin_water_bottles_l2749_274951

theorem robin_water_bottles (morning_bottles : ℕ) (afternoon_bottles : ℕ) 
  (h1 : morning_bottles = 7) 
  (h2 : afternoon_bottles = 7) : 
  morning_bottles + afternoon_bottles = 14 := by
  sorry

end NUMINAMATH_CALUDE_robin_water_bottles_l2749_274951


namespace NUMINAMATH_CALUDE_point_division_and_linear_combination_l2749_274993

/-- Given a line segment AB and a point P on it, prove that P divides AB in the ratio 4:1 
    and can be expressed as a linear combination of A and B -/
theorem point_division_and_linear_combination (A B P : ℝ × ℝ) : 
  A = (1, 2) →
  B = (4, 3) →
  (P.1 - A.1) / (B.1 - P.1) = 4 →
  (P.2 - A.2) / (B.2 - P.2) = 4 →
  ∃ (t u : ℝ), P = (t * A.1 + u * B.1, t * A.2 + u * B.2) ∧ t = 1/5 ∧ u = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_point_division_and_linear_combination_l2749_274993


namespace NUMINAMATH_CALUDE_geometric_progression_condition_l2749_274948

/-- 
Given a, b, c are real numbers and k, n, p are integers,
if a, b, c are the k-th, n-th, and p-th terms respectively of a geometric progression,
then (a/b)^(k-p) = (a/c)^(k-n)
-/
theorem geometric_progression_condition 
  (a b c : ℝ) (k n p : ℤ) 
  (hk : k ≠ n) (hn : n ≠ p) (hp : p ≠ k)
  (hgp : ∃ (r : ℝ), r ≠ 0 ∧ b = a * r^(n-k) ∧ c = a * r^(p-k)) :
  (a/b)^(k-p) = (a/c)^(k-n) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_condition_l2749_274948


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2749_274966

theorem trigonometric_simplification :
  (Real.sin (11 * π / 180) * Real.cos (15 * π / 180) + 
   Real.sin (15 * π / 180) * Real.cos (11 * π / 180)) / 
  (Real.sin (18 * π / 180) * Real.cos (12 * π / 180) + 
   Real.sin (12 * π / 180) * Real.cos (18 * π / 180)) = 
  2 * Real.sin (26 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2749_274966


namespace NUMINAMATH_CALUDE_cos_pi_half_minus_two_alpha_l2749_274943

theorem cos_pi_half_minus_two_alpha (α : ℝ) (h : Real.sin (π/4 + α) = 1/3) :
  Real.cos (π/2 - 2*α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_half_minus_two_alpha_l2749_274943


namespace NUMINAMATH_CALUDE_store_products_theorem_l2749_274998

-- Define the universe of products
variable (Product : Type)

-- Define a predicate for products displayed in the store
variable (displayed : Product → Prop)

-- Define a predicate for products that are for sale
variable (for_sale : Product → Prop)

-- Theorem stating that if not all displayed products are for sale,
-- then some displayed products are not for sale and not all displayed products are for sale
theorem store_products_theorem (h : ¬∀ (p : Product), displayed p → for_sale p) :
  (∃ (p : Product), displayed p ∧ ¬for_sale p) ∧
  (¬∀ (p : Product), displayed p → for_sale p) :=
by sorry

end NUMINAMATH_CALUDE_store_products_theorem_l2749_274998


namespace NUMINAMATH_CALUDE_micrometer_conversion_l2749_274974

-- Define the conversion factor from micrometers to meters
def micrometer_to_meter : ℝ := 1e-6

-- State the theorem
theorem micrometer_conversion :
  0.01 * micrometer_to_meter = 1e-8 := by
  sorry

end NUMINAMATH_CALUDE_micrometer_conversion_l2749_274974


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2749_274959

theorem absolute_value_inequality_solution_set : 
  {x : ℝ | |x - 2| ≤ 1} = Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2749_274959


namespace NUMINAMATH_CALUDE_grade12_sample_size_l2749_274970

/-- Represents the number of grade 12 students in a stratified sample -/
def grade12InSample (totalStudents gradeStudents sampleSize : ℕ) : ℚ :=
  (sampleSize : ℚ) * (gradeStudents : ℚ) / (totalStudents : ℚ)

/-- Theorem: The number of grade 12 students in the sample is 140 -/
theorem grade12_sample_size :
  grade12InSample 2000 700 400 = 140 := by sorry

end NUMINAMATH_CALUDE_grade12_sample_size_l2749_274970


namespace NUMINAMATH_CALUDE_first_year_after_2021_with_sum_15_l2749_274996

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_first_year_after_2021_with_sum_15 (year : ℕ) : Prop :=
  year > 2021 ∧
  sum_of_digits year = 15 ∧
  ∀ y : ℕ, 2021 < y ∧ y < year → sum_of_digits y ≠ 15

theorem first_year_after_2021_with_sum_15 :
  is_first_year_after_2021_with_sum_15 2049 := by
  sorry

end NUMINAMATH_CALUDE_first_year_after_2021_with_sum_15_l2749_274996


namespace NUMINAMATH_CALUDE_max_salary_theorem_l2749_274901

/-- Represents a basketball team in a semipro league. -/
structure BasketballTeam where
  num_players : ℕ
  min_salary : ℕ
  max_total_salary : ℕ

/-- Calculates the maximum possible salary for a single player in a basketball team. -/
def max_single_player_salary (team : BasketballTeam) : ℕ :=
  team.max_total_salary - (team.num_players - 1) * team.min_salary

/-- Theorem stating the maximum possible salary for a single player in the given conditions. -/
theorem max_salary_theorem (team : BasketballTeam) 
    (h1 : team.num_players = 21)
    (h2 : team.min_salary = 20000)
    (h3 : team.max_total_salary = 900000) : 
  max_single_player_salary team = 500000 := by
  sorry

#eval max_single_player_salary { num_players := 21, min_salary := 20000, max_total_salary := 900000 }

end NUMINAMATH_CALUDE_max_salary_theorem_l2749_274901


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l2749_274924

theorem quadratic_equation_proof (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    x1^2 - 4*x1 - 2*m + 5 = 0 ∧ 
    x2^2 - 4*x2 - 2*m + 5 = 0 ∧
    x1*x2 + x1 + x2 = m^2 + 6) →
  m = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l2749_274924


namespace NUMINAMATH_CALUDE_xy_value_l2749_274929

theorem xy_value (x y : ℝ) 
  (h1 : (8 : ℝ)^x / (4 : ℝ)^(x+y) = 16)
  (h2 : (27 : ℝ)^(x+y) / (9 : ℝ)^(4*y) = 729) : 
  x * y = 48 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l2749_274929


namespace NUMINAMATH_CALUDE_linear_function_quadrant_l2749_274979

theorem linear_function_quadrant (m : ℤ) : 
  (∀ x y : ℝ, y = (m + 4) * x + (m + 2) → ¬(x < 0 ∧ y > 0)) →
  (m = -3 ∨ m = -2) :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrant_l2749_274979


namespace NUMINAMATH_CALUDE_total_accessories_is_712_l2749_274946

/-- Calculates the total number of accessories used by Jane and Emily for their dresses -/
def total_accessories : ℕ :=
  let jane_dresses := 4 * 10
  let emily_dresses := 3 * 8
  let jane_accessories_per_dress := 3 + 2 + 1 + 4
  let emily_accessories_per_dress := 2 + 3 + 2 + 5 + 1
  jane_dresses * jane_accessories_per_dress + emily_dresses * emily_accessories_per_dress

/-- Theorem stating that the total number of accessories is 712 -/
theorem total_accessories_is_712 : total_accessories = 712 := by
  sorry

end NUMINAMATH_CALUDE_total_accessories_is_712_l2749_274946


namespace NUMINAMATH_CALUDE_minimum_donut_cost_minimum_donut_cost_proof_l2749_274956

/-- The minimum cost to buy at least 550 donuts, given that they are sold in dozens at $7.49 per dozen -/
theorem minimum_donut_cost : ℝ → Prop :=
  fun cost =>
    ∀ n : ℕ,
      (12 * n ≥ 550) →
      (cost ≤ n * 7.49) ∧
      (∃ m : ℕ, (12 * m ≥ 550) ∧ (cost = m * 7.49)) →
      cost = 344.54

/-- Proof of the minimum_donut_cost theorem -/
theorem minimum_donut_cost_proof : minimum_donut_cost 344.54 := by
  sorry

end NUMINAMATH_CALUDE_minimum_donut_cost_minimum_donut_cost_proof_l2749_274956


namespace NUMINAMATH_CALUDE_regular_polygon_angle_relation_l2749_274934

theorem regular_polygon_angle_relation : 
  ∀ n : ℕ, 
  n ≥ 3 →
  (360 / n : ℚ) = (120 / 5 : ℚ) →
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_angle_relation_l2749_274934


namespace NUMINAMATH_CALUDE_rectangular_park_length_l2749_274923

theorem rectangular_park_length 
  (perimeter : ℝ) 
  (breadth : ℝ) 
  (h1 : perimeter = 1000) 
  (h2 : breadth = 200) : 
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter ∧ 
  perimeter / 2 - breadth = 300 := by
sorry

end NUMINAMATH_CALUDE_rectangular_park_length_l2749_274923


namespace NUMINAMATH_CALUDE_remainder_three_to_seventeen_mod_five_l2749_274997

theorem remainder_three_to_seventeen_mod_five : 3^17 % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_to_seventeen_mod_five_l2749_274997


namespace NUMINAMATH_CALUDE_initial_short_trees_count_l2749_274902

/-- The number of short trees in the park after planting -/
def final_short_trees : ℕ := 95

/-- The number of short trees planted today -/
def planted_short_trees : ℕ := 64

/-- The initial number of short trees in the park -/
def initial_short_trees : ℕ := final_short_trees - planted_short_trees

theorem initial_short_trees_count : initial_short_trees = 31 := by
  sorry

end NUMINAMATH_CALUDE_initial_short_trees_count_l2749_274902


namespace NUMINAMATH_CALUDE_prove_d_value_l2749_274990

def floor_d : ℤ := -9

def frac_d : ℚ := 2/5

theorem prove_d_value :
  let d : ℚ := floor_d + frac_d
  (3 * floor_d^2 + 14 * floor_d - 45 = 0) ∧
  (5 * frac_d^2 - 18 * frac_d + 8 = 0) ∧
  (0 ≤ frac_d ∧ frac_d < 1) →
  d = -43/5 := by sorry

end NUMINAMATH_CALUDE_prove_d_value_l2749_274990


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2749_274936

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 2) = 7 → x = 47 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2749_274936


namespace NUMINAMATH_CALUDE_sequence_sum_theorem_l2749_274919

def sequence_term (n : ℕ+) : ℚ := 1 / (n * (n + 1))

def sum_of_terms (n : ℕ+) : ℚ := n / (n + 1)

theorem sequence_sum_theorem (n : ℕ+) :
  (∀ k : ℕ+, k ≤ n → sequence_term k = 1 / (k * (k + 1))) →
  sum_of_terms n = 10 / 11 →
  n = 10 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_theorem_l2749_274919


namespace NUMINAMATH_CALUDE_units_digit_of_p_plus_two_l2749_274935

def is_positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

def has_positive_units_digit (n : ℕ) : Prop := n % 10 > 0

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_p_plus_two (p : ℕ) 
  (h1 : is_positive_even p)
  (h2 : has_positive_units_digit p)
  (h3 : units_digit (p^3) - units_digit (p^2) = 0) :
  units_digit (p + 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_p_plus_two_l2749_274935


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l2749_274928

theorem regular_polygon_exterior_angle (n : ℕ) : 
  (n > 2) → (360 / n = 72) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l2749_274928


namespace NUMINAMATH_CALUDE_f_odd_f_increasing_f_odd_and_increasing_l2749_274972

/-- The function f(x) = x|x| -/
def f (x : ℝ) : ℝ := x * abs x

/-- f is an odd function -/
theorem f_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

/-- f is an increasing function -/
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by sorry

/-- f is both odd and increasing -/
theorem f_odd_and_increasing : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) := by sorry

end NUMINAMATH_CALUDE_f_odd_f_increasing_f_odd_and_increasing_l2749_274972


namespace NUMINAMATH_CALUDE_cookie_theorem_l2749_274909

def cookie_problem (initial_cookies eaten_cookies given_cookies : ℕ) : Prop :=
  initial_cookies = eaten_cookies + given_cookies ∧
  eaten_cookies - given_cookies = 11

theorem cookie_theorem :
  cookie_problem 17 14 3 := by
  sorry

end NUMINAMATH_CALUDE_cookie_theorem_l2749_274909


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l2749_274992

theorem largest_triangle_perimeter (x : ℤ) : 
  (7 : ℝ) + 11 > (x : ℝ) → 
  (7 : ℝ) + (x : ℝ) > 11 → 
  11 + (x : ℝ) > 7 → 
  (∃ (y : ℤ), (7 : ℝ) + 11 + (y : ℝ) ≥ 7 + 11 + (x : ℝ)) ∧ 
  (7 : ℝ) + 11 + (y : ℝ) ≤ 35 :=
by sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l2749_274992


namespace NUMINAMATH_CALUDE_sum_of_absolute_values_l2749_274954

def S (n : ℕ+) : ℤ := n^2 + 6*n + 1

def a (n : ℕ+) : ℤ := S n - S (n-1)

theorem sum_of_absolute_values : |a 1| + |a 2| + |a 3| + |a 4| = 41 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_values_l2749_274954


namespace NUMINAMATH_CALUDE_round_1647_to_hundredth_l2749_274926

def round_to_hundredth (x : ℚ) : ℚ :=
  (⌊x * 100 + 0.5⌋ : ℚ) / 100

theorem round_1647_to_hundredth :
  round_to_hundredth (1647 / 1000) = 165 / 100 := by
  sorry

end NUMINAMATH_CALUDE_round_1647_to_hundredth_l2749_274926


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2749_274925

theorem absolute_value_inequality (x : ℝ) : |2*x - 1| < 3 ↔ -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2749_274925


namespace NUMINAMATH_CALUDE_racecourse_length_l2749_274987

/-- Racecourse problem -/
theorem racecourse_length
  (speed_a speed_b : ℝ)
  (head_start : ℝ)
  (h1 : speed_a = 2 * speed_b)
  (h2 : head_start = 64)
  (h3 : speed_a > 0)
  (h4 : speed_b > 0) :
  ∃ (length : ℝ), 
    length > 0 ∧
    length / speed_a = (length - head_start) / speed_b ∧
    length = 128 := by
  sorry

end NUMINAMATH_CALUDE_racecourse_length_l2749_274987


namespace NUMINAMATH_CALUDE_function_derivative_equality_l2749_274988

/-- Given a function f(x) = x(2017 + ln x), prove that if f'(x₀) = 2018, then x₀ = 1 -/
theorem function_derivative_equality (x₀ : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x * (2017 + Real.log x)
  (deriv f x₀ = 2018) → x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_derivative_equality_l2749_274988


namespace NUMINAMATH_CALUDE_mean_inequality_for_close_numbers_l2749_274913

theorem mean_inequality_for_close_numbers
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (hxy : x ≠ y)
  (hyz : y ≠ z)
  (hxz : x ≠ z)
  (hclose : ∃ (ε δ : ℝ), ε > 0 ∧ δ > 0 ∧ ε < 1 ∧ δ < 1 ∧ x = y + ε ∧ z = y - δ) :
  (x + y) / 2 > Real.sqrt (x * y) ∧ Real.sqrt (x * y) > 2 * y * z / (y + z) :=
sorry

end NUMINAMATH_CALUDE_mean_inequality_for_close_numbers_l2749_274913


namespace NUMINAMATH_CALUDE_log_sum_inequality_l2749_274978

theorem log_sum_inequality (a b : ℝ) (h1 : 2^a = Real.pi) (h2 : 5^b = Real.pi) :
  1/a + 1/b > 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_inequality_l2749_274978


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l2749_274960

theorem final_sum_after_operations (S a b : ℝ) (h : a + b = S) :
  3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l2749_274960


namespace NUMINAMATH_CALUDE_exponent_division_l2749_274939

theorem exponent_division (x : ℝ) (hx : x ≠ 0) : 2 * x^4 / x^3 = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2749_274939


namespace NUMINAMATH_CALUDE_sqrt_simplification_l2749_274986

theorem sqrt_simplification (x : ℝ) :
  1 + x ≥ 0 → -1 - x ≥ 0 → Real.sqrt (1 + x) - Real.sqrt (-1 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l2749_274986


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l2749_274912

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^23 + (i^105 * i^17) = -i - 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l2749_274912


namespace NUMINAMATH_CALUDE_haley_facebook_pictures_l2749_274904

/-- The number of pictures Haley uploaded to Facebook -/
def total_pictures : ℕ := 65

/-- The number of pictures in the first album -/
def first_album_pictures : ℕ := 17

/-- The number of additional albums -/
def additional_albums : ℕ := 6

/-- The number of pictures in each additional album -/
def pictures_per_additional_album : ℕ := 8

/-- Theorem stating the total number of pictures uploaded to Facebook -/
theorem haley_facebook_pictures :
  total_pictures = first_album_pictures + additional_albums * pictures_per_additional_album :=
by sorry

end NUMINAMATH_CALUDE_haley_facebook_pictures_l2749_274904


namespace NUMINAMATH_CALUDE_airline_capacity_proof_l2749_274900

/-- Calculates the number of passengers an airline can accommodate daily -/
def airline_capacity (num_airplanes : ℕ) (rows_per_airplane : ℕ) (seats_per_row : ℕ) (flights_per_day : ℕ) : ℕ :=
  num_airplanes * rows_per_airplane * seats_per_row * flights_per_day

/-- Proves that the airline company can accommodate 1400 passengers daily -/
theorem airline_capacity_proof :
  airline_capacity 5 20 7 2 = 1400 := by
  sorry

#eval airline_capacity 5 20 7 2

end NUMINAMATH_CALUDE_airline_capacity_proof_l2749_274900


namespace NUMINAMATH_CALUDE_probability_same_tribe_l2749_274937

def total_participants : ℕ := 18
def tribe_size : ℕ := 9
def num_quitters : ℕ := 3

theorem probability_same_tribe :
  (Nat.choose tribe_size num_quitters * 2 : ℚ) / Nat.choose total_participants num_quitters = 7 / 34 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_tribe_l2749_274937


namespace NUMINAMATH_CALUDE_arrangements_with_A_B_at_ends_arrangements_with_A_B_not_adjacent_adjustment_methods_l2749_274980

-- Define the number of instructors and students
def num_instructors : ℕ := 3
def num_students : ℕ := 7

-- Define the theorem for part (1)
theorem arrangements_with_A_B_at_ends :
  (2 * Nat.factorial 5 * Nat.factorial num_instructors : ℕ) = 1440 := by sorry

-- Define the theorem for part (2)
theorem arrangements_with_A_B_not_adjacent :
  (Nat.factorial 5 * Nat.choose 6 2 * 2 * Nat.factorial num_instructors : ℕ) = 21600 := by sorry

-- Define the theorem for part (3)
theorem adjustment_methods :
  (Nat.choose num_students 2 * (Nat.factorial 5 / Nat.factorial 3) : ℕ) = 420 := by sorry

end NUMINAMATH_CALUDE_arrangements_with_A_B_at_ends_arrangements_with_A_B_not_adjacent_adjustment_methods_l2749_274980


namespace NUMINAMATH_CALUDE_max_prob_two_unqualified_expected_cost_min_compensation_fee_l2749_274907

-- Define the probability of a fruit being unqualified
variable (p : ℝ) (hp : 0 < p ∧ p < 1)

-- Define the number of fruits in a box and sample size
def box_size : ℕ := 80
def sample_size : ℕ := 10

-- Define the inspection cost per fruit
def inspection_cost : ℝ := 1.5

-- Define the compensation fee per unqualified fruit
variable (a : ℕ) (ha : a > 0)

-- Function to calculate the probability of exactly k unqualified fruits in a sample of n
def binomial_prob (n k : ℕ) : ℝ → ℝ :=
  λ p => (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

-- Statement 1: Probability that maximizes likelihood of 2 unqualified fruits in 10
theorem max_prob_two_unqualified :
  ∃ p₀, 0 < p₀ ∧ p₀ < 1 ∧
  ∀ p, 0 < p ∧ p < 1 → binomial_prob sample_size 2 p ≤ binomial_prob sample_size 2 p₀ ∧
  p₀ = 0.2 := sorry

-- Statement 2: Expected cost given p = 0.2
theorem expected_cost (p₀ : ℝ) (hp₀ : p₀ = 0.2) :
  (sample_size : ℝ) * inspection_cost + a * (box_size - sample_size : ℝ) * p₀ = 15 + 14 * a := sorry

-- Statement 3: Minimum compensation fee for full inspection
theorem min_compensation_fee :
  ∃ a_min : ℕ, a_min > 0 ∧
  ∀ a : ℕ, a ≥ a_min →
    (box_size : ℝ) * inspection_cost < (sample_size : ℝ) * inspection_cost + a * (box_size - sample_size : ℝ) * 0.2 ∧
  a_min = 8 := sorry

end NUMINAMATH_CALUDE_max_prob_two_unqualified_expected_cost_min_compensation_fee_l2749_274907


namespace NUMINAMATH_CALUDE_triangle_side_length_l2749_274947

theorem triangle_side_length (A B C : Real) (R : Real) (a b c : Real) :
  R = 5/6 →
  Real.cos B = 3/5 →
  Real.cos A = 12/13 →
  c = 21/13 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2749_274947


namespace NUMINAMATH_CALUDE_fraction_meaningful_l2749_274950

theorem fraction_meaningful (x : ℝ) : 
  IsRegular (4 / (x + 2)) ↔ x ≠ -2 :=
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l2749_274950


namespace NUMINAMATH_CALUDE_solution_set_rational_inequality_l2749_274941

theorem solution_set_rational_inequality :
  {x : ℝ | (x - 2) / (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_rational_inequality_l2749_274941


namespace NUMINAMATH_CALUDE_lily_petals_l2749_274931

theorem lily_petals (num_lilies : ℕ) (num_tulips : ℕ) (tulip_petals : ℕ) (total_petals : ℕ) :
  num_lilies = 8 →
  num_tulips = 5 →
  tulip_petals = 3 →
  total_petals = 63 →
  ∃ (lily_petals : ℕ), lily_petals * num_lilies + tulip_petals * num_tulips = total_petals ∧ lily_petals = 6 :=
by sorry

end NUMINAMATH_CALUDE_lily_petals_l2749_274931


namespace NUMINAMATH_CALUDE_max_q_minus_r_l2749_274977

theorem max_q_minus_r (q r : ℕ+) (h : 961 = 23 * q + r) : q - r ≤ 23 := by
  sorry

end NUMINAMATH_CALUDE_max_q_minus_r_l2749_274977


namespace NUMINAMATH_CALUDE_empty_set_condition_single_element_condition_single_element_values_l2749_274991

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 3 * x + 2 = 0}

-- Theorem for the empty set condition
theorem empty_set_condition (a : ℝ) : A a = ∅ ↔ a > 9/8 := by sorry

-- Theorem for the single element condition
theorem single_element_condition (a : ℝ) : 
  (∃! x, x ∈ A a) ↔ (a = 0 ∨ a = 9/8) := by sorry

-- Theorem for the specific elements when A has a single element
theorem single_element_values (a : ℝ) :
  (∃! x, x ∈ A a) → 
  ((a = 0 → A a = {2/3}) ∧ (a = 9/8 → A a = {4/3})) := by sorry

end NUMINAMATH_CALUDE_empty_set_condition_single_element_condition_single_element_values_l2749_274991


namespace NUMINAMATH_CALUDE_line_exists_l2749_274985

/-- The parabola y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The line l1: x + 5y - 5 = 0 -/
def line_l1 (x y : ℝ) : Prop := x + 5*y - 5 = 0

/-- The line l: 25x - 5y - 21 = 0 -/
def line_l (x y : ℝ) : Prop := 25*x - 5*y - 21 = 0

/-- Two points are distinct -/
def distinct (x1 y1 x2 y2 : ℝ) : Prop := x1 ≠ x2 ∨ y1 ≠ y2

/-- A line perpendicularly bisects a segment -/
def perpendicularly_bisects (x1 y1 x2 y2 : ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (xm ym : ℝ), line xm ym ∧ 
    xm = (x1 + x2) / 2 ∧ 
    ym = (y1 + y2) / 2 ∧
    (y2 - y1) * (x2 - xm) = (x2 - x1) * (y2 - ym)

theorem line_exists : ∃ (x1 y1 x2 y2 : ℝ),
  parabola x1 y1 ∧ parabola x2 y2 ∧
  line_l x1 y1 ∧ line_l x2 y2 ∧
  distinct x1 y1 x2 y2 ∧
  perpendicularly_bisects x1 y1 x2 y2 line_l1 :=
sorry

end NUMINAMATH_CALUDE_line_exists_l2749_274985


namespace NUMINAMATH_CALUDE_math_books_in_same_box_l2749_274984

def total_textbooks : ℕ := 15
def math_textbooks : ℕ := 3
def box_capacities : List ℕ := [3, 5, 7]

def probability_all_math_in_same_box : ℚ := 25 / 242

theorem math_books_in_same_box :
  let total_arrangements := (total_textbooks.choose box_capacities[0]!) *
    ((total_textbooks - box_capacities[0]!).choose box_capacities[1]!) *
    ((total_textbooks - box_capacities[0]! - box_capacities[1]!).choose box_capacities[2]!)
  let favorable_outcomes := 
    (total_textbooks - math_textbooks).choose box_capacities[0]! +
    ((total_textbooks - math_textbooks).choose (box_capacities[1]! - math_textbooks)) * 
      ((total_textbooks - box_capacities[1]!).choose box_capacities[0]!) +
    ((total_textbooks - math_textbooks).choose (box_capacities[2]! - math_textbooks)) * 
      ((total_textbooks - box_capacities[2]!).choose box_capacities[0]!)
  probability_all_math_in_same_box = favorable_outcomes / total_arrangements :=
by sorry

end NUMINAMATH_CALUDE_math_books_in_same_box_l2749_274984


namespace NUMINAMATH_CALUDE_common_root_not_implies_equal_coefficients_l2749_274903

theorem common_root_not_implies_equal_coefficients
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (x : ℝ), (a * x^2 + b * x + c = 0 ∧ c * x^2 + b * x + a = 0) → ¬(a = c) :=
sorry

end NUMINAMATH_CALUDE_common_root_not_implies_equal_coefficients_l2749_274903


namespace NUMINAMATH_CALUDE_nine_digit_divisible_by_101_l2749_274975

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ | 100 ≤ n ∧ n < 1000 }

/-- Converts a three-digit number to a nine-digit number by repeating it three times -/
def toNineDigitNumber (n : ThreeDigitNumber) : ℕ :=
  1000000 * n + 1000 * n + n

/-- Theorem: Any nine-digit number formed by repeating a three-digit number three times is divisible by 101 -/
theorem nine_digit_divisible_by_101 (n : ThreeDigitNumber) :
  ∃ k : ℕ, toNineDigitNumber n = 101 * k := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_divisible_by_101_l2749_274975


namespace NUMINAMATH_CALUDE_karls_clothing_store_l2749_274961

/-- Karl's clothing store problem -/
theorem karls_clothing_store (tshirt_price : ℝ) (pants_price : ℝ) (skirt_price : ℝ) :
  tshirt_price = 5 →
  pants_price = 4 →
  (2 * tshirt_price + pants_price + 4 * skirt_price + 6 * (tshirt_price / 2) = 53) →
  skirt_price = 6 := by
sorry

end NUMINAMATH_CALUDE_karls_clothing_store_l2749_274961


namespace NUMINAMATH_CALUDE_linear_regression_intercept_l2749_274994

/-- Linear regression model parameters -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Mean values of x and y -/
structure MeanValues where
  x_mean : ℝ
  y_mean : ℝ

/-- Theorem: Given a linear regression model and mean values, prove the intercept -/
theorem linear_regression_intercept 
  (model : LinearRegression) 
  (means : MeanValues) 
  (h_slope : model.slope = -12/5) 
  (h_x_mean : means.x_mean = -4) 
  (h_y_mean : means.y_mean = 25) : 
  model.intercept = 77/5 := by
  sorry

#check linear_regression_intercept

end NUMINAMATH_CALUDE_linear_regression_intercept_l2749_274994


namespace NUMINAMATH_CALUDE_ellipse_equation_l2749_274953

/-- Given an ellipse with focal distance 4 passing through (√2, √3), prove its equation is x²/8 + y²/4 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ c : ℝ, c = 2 ∧ a^2 - b^2 = c^2) → 
  (2 / a^2 + 3 / b^2 = 1) → 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 8 + y^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2749_274953


namespace NUMINAMATH_CALUDE_book_selection_l2749_274921

theorem book_selection (n m k : ℕ) (hn : n = 8) (hm : m = 5) (hk : k = 1) :
  (Nat.choose (n - k) (m - k)) = 35 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_l2749_274921


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2749_274968

theorem diophantine_equation_solution (x y : ℤ) :
  x^2 = 2 + 6*y^2 + y^4 ↔ (x = 3 ∧ y = 1) ∨ (x = -3 ∧ y = 1) ∨ (x = 3 ∧ y = -1) ∨ (x = -3 ∧ y = -1) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2749_274968


namespace NUMINAMATH_CALUDE_first_brand_price_l2749_274965

/-- The regular price of pony jeans -/
def pony_price : ℝ := 18

/-- The total savings on 5 pairs of jeans -/
def total_savings : ℝ := 8.55

/-- The sum of the two discount rates -/
def sum_discount_rates : ℝ := 0.22

/-- The discount rate on pony jeans -/
def pony_discount_rate : ℝ := 0.15

/-- The number of pairs of the first brand of jeans -/
def num_first_brand : ℕ := 3

/-- The number of pairs of pony jeans -/
def num_pony : ℕ := 2

/-- Theorem stating that the regular price of the first brand of jeans is $15 -/
theorem first_brand_price : ∃ (price : ℝ),
  price = 15 ∧
  (price * num_first_brand * (sum_discount_rates - pony_discount_rate) +
   pony_price * num_pony * pony_discount_rate = total_savings) :=
sorry

end NUMINAMATH_CALUDE_first_brand_price_l2749_274965


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2749_274955

theorem inscribed_circle_radius 
  (R : ℝ) 
  (r : ℝ) 
  (h1 : R = 18) 
  (h2 : r = 9) 
  (h3 : r = R / 2) : 
  ∃ x : ℝ, x = 8 ∧ 
    (R - x)^2 - x^2 = (r + x)^2 - x^2 ∧ 
    x > 0 ∧ 
    x < R ∧ 
    x < r := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2749_274955


namespace NUMINAMATH_CALUDE_range_of_m_l2749_274930

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 2 then 2^(x - m) else (m * x) / (4 * x^2 + 16)

theorem range_of_m (m : ℝ) :
  (∀ x₁ ≥ 2, ∃ x₂ ≤ 2, f m x₁ = f m x₂) →
  m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2749_274930
