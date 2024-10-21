import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_sufficient_not_necessary_for_positive_sine_l982_98297

-- Define what it means for an angle to be acute
def is_acute (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2

-- Define the theorem
theorem acute_sufficient_not_necessary_for_positive_sine :
  (∀ α, is_acute α → Real.sin α > 0) ∧
  ∃ α, Real.sin α > 0 ∧ ¬is_acute α :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_sufficient_not_necessary_for_positive_sine_l982_98297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_circumference_l982_98244

/-- The circumference of a circular track where two objects meet under specific conditions -/
theorem track_circumference
  (first_meeting : ℝ)
  (second_meeting : ℝ)
  (full_lap : ℝ)
  (circumference : ℝ) :
  first_meeting = 150 ∧
  second_meeting = full_lap - 90 ∧
  circumference = 2 * first_meeting + 2 * (full_lap - second_meeting) →
  circumference = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_circumference_l982_98244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_for_checkerboard_l982_98283

/-- Represents the color of a cell -/
inductive Color where
  | White
  | Black

/-- Represents a move on the grid -/
structure Move where
  row_or_column : Bool  -- true for row, false for column
  index : Fin 100
  cells : Fin 99 → Fin 100

/-- Represents the state of the grid -/
def Grid := Fin 100 → Fin 100 → Color

/-- Checks if a grid has a checkerboard pattern -/
def is_checkerboard (g : Grid) : Prop :=
  ∀ i j : Fin 100, g i j = if (i.val + j.val) % 2 = 0 then Color.White else Color.Black

/-- Applies a move to a grid -/
def apply_move (g : Grid) (m : Move) : Grid :=
  λ i j => sorry

/-- The initial all-white grid -/
def initial_grid : Grid :=
  λ _ _ => Color.White

/-- Theorem: The minimum number of moves to achieve a checkerboard pattern is 100 -/
theorem min_moves_for_checkerboard :
  ∃ (moves : List Move),
    moves.length = 100 ∧
    is_checkerboard (moves.foldl apply_move initial_grid) ∧
    ∀ (moves' : List Move),
      is_checkerboard (moves'.foldl apply_move initial_grid) →
      moves'.length ≥ 100 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_for_checkerboard_l982_98283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_li_ming_boards_first_probability_l982_98285

-- Define the arrival intervals for the buses
noncomputable def bus1_interval : ℝ := 8
noncomputable def bus2_interval : ℝ := 10

-- Define the probability function
noncomputable def probability_li_ming_first : ℝ := 
  (bus1_interval * bus2_interval - (1/2) * bus1_interval * bus1_interval) / (bus1_interval * bus2_interval)

-- Theorem statement
theorem li_ming_boards_first_probability :
  probability_li_ming_first = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_li_ming_boards_first_probability_l982_98285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l982_98248

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2)^(a*x)
noncomputable def g (x : ℝ) : ℝ := 4^(-x) - 2

-- State the theorem
theorem function_equality :
  ∃ (a : ℝ), 
    (f a (-1) = 2) ∧ 
    (∃ (x : ℝ), g x = f a x ∧ x = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l982_98248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cross_section_area_l982_98267

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
structure Vector3D where
  i : ℝ
  j : ℝ
  k : ℝ

/-- Represents a rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  base_center : Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculate the cross product of two 3D vectors -/
def cross_product (v1 v2 : Vector3D) : Vector3D :=
  { i := v1.j * v2.k - v1.k * v2.j,
    j := v1.k * v2.i - v1.i * v2.k,
    k := v1.i * v2.j - v1.j * v2.i }

/-- Calculate the magnitude of a 3D vector -/
noncomputable def magnitude (v : Vector3D) : ℝ :=
  Real.sqrt (v.i^2 + v.j^2 + v.k^2)

/-- Calculate the area of the cross-sectional cut -/
noncomputable def cross_section_area (prism : RectangularPrism) (plane : Plane) : ℝ :=
  let v1 : Vector3D := { i := -prism.length, j := 0, k := plane.a * prism.length / (-plane.c) }
  let v2 : Vector3D := { i := 0, j := -prism.width, k := plane.b * prism.width / (-plane.c) }
  (1/2) * magnitude (cross_product v1 v2)

theorem max_cross_section_area (prism : RectangularPrism) (plane : Plane) :
  cross_section_area prism plane = (1/2) * Real.sqrt 56016 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cross_section_area_l982_98267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_equation_circle_C_polar_equation_circle_C_line_intersects_circle_chord_length_l982_98259

-- Define the circle C
noncomputable def circle_C (θ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos θ, 2 * Real.sin θ)

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + (4/5) * t, (3/5) * t)

-- Standard equation of circle C
theorem standard_equation_circle_C :
  ∀ x y : ℝ, (∃ θ : ℝ, circle_C θ = (x, y)) ↔ (x - 2)^2 + y^2 = 4 := by sorry

-- Polar equation of circle C
theorem polar_equation_circle_C :
  ∀ ρ θ : ℝ, (∃ φ : ℝ, circle_C φ = (ρ * Real.cos θ, ρ * Real.sin θ)) ↔ ρ = 4 * Real.cos θ := by sorry

-- Line l intersects circle C
theorem line_intersects_circle :
  ∃ t θ : ℝ, circle_C θ = line_l t := by sorry

-- Length of the chord
theorem chord_length :
  let intersection_points := {p : ℝ × ℝ | ∃ t θ : ℝ, circle_C θ = p ∧ line_l t = p}
  ∃ p1 p2 : ℝ × ℝ, p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_equation_circle_C_polar_equation_circle_C_line_intersects_circle_chord_length_l982_98259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_f_approaches_one_f_approaches_two_f_range_l982_98245

/-- The function f defined on positive real numbers -/
noncomputable def f (x y z : ℝ) : ℝ := x / (x + y) + y / (y + z) + z / (z + x)

/-- The theorem stating that f(x,y,z) is always between 1 and 2 for positive real x, y, z -/
theorem f_bounds (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  1 < f x y z ∧ f x y z < 2 := by
  sorry

/-- The theorem stating that f(x,y,z) can be arbitrarily close to 1 -/
theorem f_approaches_one :
  ∀ ε > (0 : ℝ), ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ |f x y z - 1| < ε := by
  sorry

/-- The theorem stating that f(x,y,z) can be arbitrarily close to 2 -/
theorem f_approaches_two :
  ∀ ε > (0 : ℝ), ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ |f x y z - 2| < ε := by
  sorry

/-- The main theorem characterizing the range of f -/
theorem f_range :
  ∀ w ∈ Set.Ioo 1 2, ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ f x y z = w := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_f_approaches_one_f_approaches_two_f_range_l982_98245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angleBisectorTrianglePerimeter_specific_l982_98298

/-- Given a right triangle with legs a and b, returns the perimeter of the triangle
    formed by the intersections of angle bisectors with opposite sides. -/
noncomputable def angleBisectorTrianglePerimeter (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  let x := (a * b) / (a + c)
  let y := (a * b) / (b + c)
  let z := (b * c) / (a + b)
  Real.sqrt (x^2 + y^2) + Real.sqrt (y^2 + z^2 - 2 * y * z * (a / c)) +
    Real.sqrt (x^2 + z^2 - 2 * x * z * (b / c))

theorem angleBisectorTrianglePerimeter_specific :
  angleBisectorTrianglePerimeter 126 168 = Real.sqrt 7105 + Real.sqrt 5440 + Real.sqrt 5265 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angleBisectorTrianglePerimeter_specific_l982_98298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_ratio_is_17_7_l982_98238

/-- Right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 5
  hb : b = 12
  hc : c = 13
  right_angle : a^2 + b^2 = c^2

/-- Square inscribed with one vertex at the right angle -/
noncomputable def square_at_vertex (t : RightTriangle) : ℝ :=
  60 / 7

/-- Square inscribed with one side on the hypotenuse -/
noncomputable def square_on_hypotenuse (t : RightTriangle) : ℝ :=
  60 / 17

/-- The ratio of the two square side lengths -/
noncomputable def square_ratio (t : RightTriangle) : ℝ :=
  (square_at_vertex t) / (square_on_hypotenuse t)

theorem square_ratio_is_17_7 (t : RightTriangle) :
  square_ratio t = 17 / 7 := by
  unfold square_ratio square_at_vertex square_on_hypotenuse
  simp [div_div_eq_mul_div]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_ratio_is_17_7_l982_98238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_y_l982_98224

theorem min_value_of_y (x : ℝ) (h : x > 2) : 
  (∀ y, y = x + 4 / (x - 2) → y ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_y_l982_98224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l982_98205

/-- The curve C in the Cartesian coordinate system -/
noncomputable def curve_C (x : ℝ) : ℝ := x^2 / 8 - 1

/-- The line l in the Cartesian coordinate system -/
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

/-- The distance function from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x - y + 3| / Real.sqrt 2

theorem max_distance_curve_to_line :
  ∃ (max_dist : ℝ), max_dist = 3 * Real.sqrt 2 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-4 : ℝ) 4 →
    distance_to_line x (curve_C x) ≤ max_dist ∧
    ∃ (x₀ : ℝ), x₀ ∈ Set.Icc (-4 : ℝ) 4 ∧
      distance_to_line x₀ (curve_C x₀) = max_dist := by
  sorry

#check max_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l982_98205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_is_one_point_five_l982_98226

-- Define the points in 3D space
noncomputable def A : Fin 3 → ℝ := ![0, 0, 0]
noncomputable def B : Fin 3 → ℝ := ![0, 3, 0]
noncomputable def C : Fin 3 → ℝ := ![4, 3, 0]
noncomputable def D : Fin 3 → ℝ := ![4, 0, 0]

-- Define the heights of the extended points
def height_A' : ℝ := 12
def height_B' : ℝ := 9
def height_C' : ℝ := 15
def height_D' : ℝ := 21

-- Define the extended points
noncomputable def A' : Fin 3 → ℝ := ![A 0, A 1, height_A']
noncomputable def B' : Fin 3 → ℝ := ![B 0, B 1, height_B']
noncomputable def C' : Fin 3 → ℝ := ![C 0, C 1, height_C']
noncomputable def D' : Fin 3 → ℝ := ![D 0, D 1, height_D']

-- Define midpoints M and N
noncomputable def M : Fin 3 → ℝ := ![
  (A' 0 + C' 0) / 2,
  (A' 1 + C' 1) / 2,
  (A' 2 + C' 2) / 2
]

noncomputable def N : Fin 3 → ℝ := ![
  (B' 0 + D' 0) / 2,
  (B' 1 + D' 1) / 2,
  (B' 2 + D' 2) / 2
]

-- Theorem statement
theorem length_MN_is_one_point_five :
  ‖M - N‖ = 1.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_is_one_point_five_l982_98226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_four_representation_of_150_l982_98261

/-- Represents a number in base 4 as a list of digits (least significant first) -/
def BaseFour : Type := List Nat

/-- Converts a base 10 number to base 4 -/
def toBaseFour (n : Nat) : BaseFour :=
  sorry

/-- Converts a base 4 number to base 10 -/
def fromBaseFour (b : BaseFour) : Nat :=
  sorry

theorem base_four_representation_of_150 :
  toBaseFour 150 = [2, 1, 1, 2] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_four_representation_of_150_l982_98261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l982_98281

/-- Circle C with center (2, 0) and radius 2 -/
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

/-- Unit circle -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Line passing through M(m, n) -/
def line_l (m n x y : ℝ) : Prop := m * x + n * y = 1

/-- Point M is on circle C -/
def M_on_C (m n : ℝ) : Prop := circle_C m n

/-- Distance from origin to line l -/
noncomputable def distance_origin_to_line (m n : ℝ) : ℝ := 1 / Real.sqrt (m^2 + n^2)

/-- Area of triangle OAB -/
noncomputable def area_OAB (d : ℝ) : ℝ := Real.sqrt (d^2 - d^4)

theorem max_area_triangle :
  ∃ m n : ℝ,
    M_on_C m n ∧
    distance_origin_to_line m n < 1 ∧
    area_OAB (distance_origin_to_line m n) = 1/2 ∧
    (∀ m' n' : ℝ, M_on_C m' n' →
      distance_origin_to_line m' n' < 1 →
      area_OAB (distance_origin_to_line m' n') ≤ 1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l982_98281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_root_of_g_between_pi_over_two_and_pi_l982_98242

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin x + 3 * Real.cos x + 4 * (Real.cos x / Real.sin x)

theorem smallest_positive_root_of_g_between_pi_over_two_and_pi :
  ∃ s : ℝ, π/2 < s ∧ s < π ∧ g s = 0 ∧ ∀ x, 0 < x ∧ x < s → g x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_root_of_g_between_pi_over_two_and_pi_l982_98242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l982_98236

/-- Represents the number of boys -/
def m : ℕ := sorry

/-- Represents the number of girls -/
def n : ℕ := sorry

/-- The condition that m and n satisfy the given inequality -/
axiom m_n_bounds : 10 ≥ m ∧ m > n ∧ n ≥ 4

/-- The probability of selecting two people of the same gender -/
noncomputable def P_A : ℚ := (Nat.choose m 2 + Nat.choose n 2) / Nat.choose (m + n) 2

/-- The probability of selecting two people of different genders -/
noncomputable def P_B : ℚ := (m * n) / Nat.choose (m + n) 2

/-- The condition that the probabilities of events A and B are equal -/
axiom prob_equal : P_A = P_B

/-- The theorem stating that the only solution for (m, n) is (10, 6) -/
theorem unique_solution : m = 10 ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l982_98236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_contribution_is_15750_l982_98240

/-- Represents the investment scenario with two partners A and B --/
structure Investment where
  a_initial : ℚ  -- A's initial investment
  a_months : ℚ   -- Number of months A invests
  b_months : ℚ   -- Number of months B invests
  ratio_a : ℚ    -- Profit ratio for A
  ratio_b : ℚ    -- Profit ratio for B

/-- Calculates B's contribution based on the given investment scenario --/
def calculate_b_contribution (inv : Investment) : ℚ :=
  (inv.a_initial * inv.a_months * inv.ratio_b) / (inv.b_months * inv.ratio_a)

/-- Theorem stating that B's contribution is 15750 given the specific scenario --/
theorem b_contribution_is_15750 :
  let scenario : Investment := {
    a_initial := 3500,
    a_months := 12,
    b_months := 4,
    ratio_a := 2,
    ratio_b := 3
  }
  calculate_b_contribution scenario = 15750 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_contribution_is_15750_l982_98240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l982_98203

noncomputable def f (x : ℝ) : ℝ := (2 - Real.sqrt 2 * Real.sin (Real.pi / 4 * x)) / (x^2 + 4*x + 5)

theorem max_value_of_f :
  ∃ (M : ℝ), M = 2 + Real.sqrt 2 ∧
  ∀ (x : ℝ), -4 ≤ x ∧ x ≤ 0 → f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l982_98203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l982_98256

noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 - x + 3/2

noncomputable def f_deriv (x : ℝ) : ℝ := -x - 1

theorem solution_set_of_inequality (x : ℝ) :
  (f (10^x) > 0) ↔ (x < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l982_98256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_and_area_theorem_l982_98253

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the dot product condition
def dot_product_condition (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = -2

-- Define the perpendicular line l1
def line_l1 (k : ℝ) (x y : ℝ) : Prop := y = -1/k * x + 1

-- Define the area of quadrilateral PMQN
noncomputable def area_PMQN (k : ℝ) : ℝ :=
  2 * Real.sqrt (12 + 1 / (k^2 + 2 + 1/k^2))

-- Main theorem
theorem circle_intersection_and_area_theorem :
  ∀ (k : ℝ) (x1 y1 x2 y2 : ℝ),
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    line_l k x1 y1 ∧ line_l k x2 y2 ∧
    dot_product_condition x1 y1 x2 y2 →
    k = 0 ∧
    ∀ (x : ℝ), area_PMQN x ≤ 7 ∧ (∃ k', area_PMQN k' = 7) :=
by
  sorry

-- Lemma for the maximum value of area_PMQN
lemma area_PMQN_max :
  ∀ (k : ℝ), area_PMQN k ≤ 7 ∧ (∃ k', area_PMQN k' = 7) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_and_area_theorem_l982_98253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_characterization_l982_98265

def is_valid_number (n : ℕ) : Prop :=
  (10^6 ≤ n) ∧ (n < 10^7) ∧ (n % 21 = 0) ∧ 
  (∀ d, d ∈ n.digits 10 → d = 3 ∨ d = 7)

def valid_numbers : Set ℕ := {3373377, 7373373, 7733733, 3733737, 7337337, 3777333}

theorem valid_numbers_characterization : 
  ∀ n : ℕ, is_valid_number n ↔ n ∈ valid_numbers := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_characterization_l982_98265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_koala_diet_l982_98280

-- Define the absorption rates
noncomputable def fiber_absorption_rate : ℝ := 0.40
noncomputable def protein_absorption_rate : ℝ := 0.20

-- Define the absorbed amounts
noncomputable def absorbed_fiber : ℝ := 12
noncomputable def protein_absorption : ℝ := 2

-- Define the function to calculate total amount eaten
noncomputable def total_eaten (absorbed : ℝ) (rate : ℝ) : ℝ := absorbed / rate

-- Theorem statement
theorem koala_diet :
  total_eaten absorbed_fiber fiber_absorption_rate = 30 ∧
  total_eaten protein_absorption protein_absorption_rate = 10 :=
by
  -- Split the conjunction
  constructor
  -- Prove the first part
  · simp [total_eaten, absorbed_fiber, fiber_absorption_rate]
    norm_num
  -- Prove the second part
  · simp [total_eaten, protein_absorption, protein_absorption_rate]
    norm_num

#check koala_diet

end NUMINAMATH_CALUDE_ERRORFEEDBACK_koala_diet_l982_98280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_b_l982_98295

theorem a_greater_than_b (m n : ℕ) (x : ℝ) 
  (h1 : m > n) (h2 : n > 0) (h3 : x > 1) : 
  (Real.log x)^m + (Real.log x)^(-m : ℤ) > (Real.log x)^n + (Real.log x)^(-n : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_b_l982_98295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_properties_l982_98220

-- Define the sector
structure Sector where
  r : ℝ
  θ : ℝ
  r_pos : r > 0
  θ_pos : θ > 0

-- Define the circumference of the sector
noncomputable def circumference (s : Sector) : ℝ :=
  s.r * s.θ + 2 * s.r

-- Define the area of the sector
noncomputable def area (s : Sector) : ℝ :=
  1/2 * s.r^2 * s.θ

-- Theorem statement
theorem sector_properties (s : Sector) 
  (h_circ : circumference s = 44) 
  (h_angle : s.θ = 2) : 
  s.r = 11 ∧ area s = 121 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_properties_l982_98220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_connection_theorem_l982_98251

/-- A regular octahedron -/
structure RegularOctahedron where
  edgeLength : ℝ
  edgeLength_pos : edgeLength > 0

/-- The result of connecting midpoints of faces of a regular octahedron -/
structure MidpointConnectedSolid where
  original : RegularOctahedron
  resultingShape : Type
  resultingEdgeLength : ℝ

/-- Definition of a cube for completeness -/
inductive Cube

/-- Theorem stating that connecting midpoints of faces of a regular octahedron results in a cube -/
theorem midpoint_connection_theorem (o : RegularOctahedron) :
  ∃ (s : MidpointConnectedSolid),
    s.original = o ∧
    s.resultingShape = Cube ∧
    s.resultingEdgeLength = (Real.sqrt 2 / 3) * o.edgeLength := by
  sorry

/-- Helper function to create a MidpointConnectedSolid -/
noncomputable def connect_midpoints (o : RegularOctahedron) : MidpointConnectedSolid where
  original := o
  resultingShape := Cube
  resultingEdgeLength := (Real.sqrt 2 / 3) * o.edgeLength


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_connection_theorem_l982_98251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_roots_l982_98234

noncomputable def root_count (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  Nat.card {x : ℝ | a ≤ x ∧ x ≤ b ∧ f x = 0}

theorem periodic_function_roots
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (3 + x) = f (3 - x))
  (h2 : ∀ x, f (8 + x) = f (8 - x))
  (h3 : f 0 = 0) :
  root_count f (-950) 950 ≥ 267 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_roots_l982_98234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_to_line_parallel_l982_98214

/-- A line parameterized by t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Check if a vector is on a parameterized line -/
def vector_on_line (v : ℝ × ℝ) (l : ParametricLine) : Prop :=
  ∃ t : ℝ, v.1 = l.x t ∧ v.2 = l.y t

/-- Check if two vectors are parallel -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_to_line_parallel (l : ParametricLine) (v : ℝ × ℝ) :
  (∀ t : ℝ, l.x t = 5 * t + 1) →
  (∀ t : ℝ, l.y t = 2 * t + 3) →
  vector_on_line v l →
  parallel v (3, 1) →
  v = (-39, -13) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_to_line_parallel_l982_98214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gaussian_function_properties_l982_98229

-- Define the Gaussian function
noncomputable def f (x : ℝ) : ℤ := ⌊x⌋

-- Theorem statement
theorem gaussian_function_properties :
  (f (-3) = -3) ∧
  (∀ a b : ℝ, f a = f b → |a - b| < 1) ∧
  (∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x ≤ y → x * (f x : ℝ) ≤ y * (f y : ℝ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gaussian_function_properties_l982_98229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_equal_distances_l982_98294

/-- The distance from a point (x₀, y₀) to a line ax + by + c = 0 -/
noncomputable def distance_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  (|a * x₀ + b * y₀ + c|) / Real.sqrt (a^2 + b^2)

/-- The theorem statement -/
theorem unique_m_for_equal_distances : ∃! m : ℝ, m ≠ 0 ∧
  distance_to_line 0 0 m (m^2) 6 = distance_to_line 4 (-1) m (m^2) 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_equal_distances_l982_98294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_order_l982_98269

theorem abc_order : 
  let a : ℝ := (-2/3)^(-2 : ℤ)
  let b : ℝ := (-1)^(-1 : ℤ)
  let c : ℝ := (-Real.pi/2)^(0 : ℤ)
  b < c ∧ c < a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_order_l982_98269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_formation_l982_98217

/-- A structure representing a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A structure representing a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Function to check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop := l1.slope * l2.slope = -1

/-- Function to find the intersection point of two lines -/
noncomputable def intersection (l1 l2 : Line) : Point :=
  { x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope),
    y := l1.slope * (l2.intercept - l1.intercept) / (l1.slope - l2.slope) + l1.intercept }

/-- Function to check if four points form a square -/
def is_square (p1 p2 p3 p4 : Point) : Prop :=
  let d12 := (p1.x - p2.x)^2 + (p1.y - p2.y)^2
  let d23 := (p2.x - p3.x)^2 + (p2.y - p3.y)^2
  let d34 := (p3.x - p4.x)^2 + (p3.y - p4.y)^2
  let d41 := (p4.x - p1.x)^2 + (p4.y - p1.y)^2
  d12 = d23 ∧ d23 = d34 ∧ d34 = d41

/-- Function to check if a point is on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

theorem square_formation (m : ℝ) :
  let l1 : Line := { slope := m, intercept := 1 }
  let l2 : Line := { slope := m, intercept := m }
  let l3 : Line := { slope := -1/m, intercept := 1/m }
  let l4 : Line := { slope := -1/m, intercept := 0 }
  parallel l1 l2 ∧ perpendicular l1 l3 ∧ perpendicular l1 l4 ∧
  point_on_line ⟨0, 1⟩ l1 ∧ point_on_line ⟨-1, 0⟩ l2 ∧
  point_on_line ⟨1, 0⟩ l3 ∧ point_on_line ⟨0, 0⟩ l4 →
  is_square (intersection l1 l3) (intersection l1 l4) (intersection l2 l3) (intersection l2 l4) ↔
  m = 0 ∨ m = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_formation_l982_98217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_x_plus_y_l982_98210

-- Define the function y in terms of x
noncomputable def y (x : ℝ) : ℝ := Real.sqrt (x - 3) + Real.sqrt (3 - x) + 1

-- Theorem statement
theorem sqrt_x_plus_y (x : ℝ) : (Real.sqrt (x + y x) = 2) ∨ (Real.sqrt (x + y x) = -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_x_plus_y_l982_98210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_discount_l982_98276

theorem merchant_discount (cost_price : ℝ) (markup_percentage : ℝ) (profit_percentage : ℝ) : 
  markup_percentage = 75 →
  profit_percentage = 57.5 →
  (let marked_price := cost_price * (1 + markup_percentage / 100)
   let selling_price := cost_price * (1 + profit_percentage / 100)
   let discount := marked_price - selling_price
   let discount_percentage := (discount / marked_price) * 100
   discount_percentage) = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_discount_l982_98276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_of_inverse_x_in_expression_l982_98293

/-- The coefficient of 1/x in the expansion of ((1-x²)⁴(x+1)/x)⁵) -/
def coeff_of_inverse_x : ℤ := -29

/-- The expression ((1-x²)⁴(x+1)/x)⁵) -/
noncomputable def expression (x : ℝ) : ℝ := ((1 - x^2)^4 * (x + 1) / x)^5

/-- Function to extract the coefficient of 1/x in a power series -/
noncomputable def coefficient_of_inverse_x (f : ℝ → ℝ) : ℤ := sorry

/-- Theorem stating that the coefficient of 1/x in the expression equals coeff_of_inverse_x -/
theorem coeff_of_inverse_x_in_expression :
  coefficient_of_inverse_x expression = coeff_of_inverse_x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_of_inverse_x_in_expression_l982_98293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_axis_l982_98207

/-- The function f(x) defined as cos(x + π/2) * cos(x + π/4) -/
noncomputable def f (x : ℝ) : ℝ := Real.cos (x + Real.pi/2) * Real.cos (x + Real.pi/4)

/-- Theorem stating that f(x) has an axis of symmetry at x = 5π/8 -/
theorem f_symmetry_axis :
  ∀ (x : ℝ), f (5 * Real.pi / 8 + x) = f (5 * Real.pi / 8 - x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_axis_l982_98207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l982_98201

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (π / 4 - α) * Real.sin (π / 4 + α) = -3 / 10) : 
  Real.tan α = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l982_98201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octahedron_volume_l982_98291

/-- The volume of a regular octahedron with edge length a -/
noncomputable def octahedron_volume (a : ℝ) : ℝ := (a^3 * Real.sqrt 2) / 3

/-- Theorem: The volume of a regular octahedron with edge length a is (a³√2) / 3 -/
theorem regular_octahedron_volume (a : ℝ) (h : a > 0) :
  octahedron_volume a = (a^3 * Real.sqrt 2) / 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octahedron_volume_l982_98291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l982_98264

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x * e^x) / (e^(ax) - 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (x * Real.exp x) / (Real.exp (a * x) - 1)

/-- If f(x) = (x * e^x) / (e^(ax) - 1) is an even function, then a = 2 -/
theorem even_function_implies_a_equals_two :
  ∃ a : ℝ, IsEven (f a) → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l982_98264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l982_98296

-- Define the points A, B, C, and D
def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (9, -40)
def C : ℝ × ℝ := (-5, -12)
def D : ℝ × ℝ := (-7, 24)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem min_distance_sum :
  ∀ P : ℝ × ℝ, distance P A + distance P B + distance P C + distance P D ≥ 16 * Real.sqrt 5 + 8 * Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l982_98296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_cubic_equation_l982_98208

noncomputable def h (x : ℝ) : ℝ := Real.rpow ((2 * x + 5) / 5) (1/3)

theorem solve_cubic_equation (x : ℝ) :
  h (3 * x) = 3 * h x ↔ x = -65/24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_cubic_equation_l982_98208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_sum_l982_98232

/-- Given a triangle ABC with point D on side BC, prove that if CD = 2DB and AD = rAB + sAC, then r + s = 1 -/
theorem triangle_vector_sum (A B C D : EuclideanSpace ℝ (Fin 2)) (r s : ℝ) : 
  (C - D) = 2 • (D - B) → 
  (A - D) = r • (A - B) + s • (A - C) → 
  r + s = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_sum_l982_98232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l982_98260

/-- Represents a point in the Euclidean plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral in the Euclidean plane -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- IsoscelesTrapezoid represents an isosceles trapezoid -/
def IsoscelesTrapezoid (ABCD : Quadrilateral) : Prop :=
  sorry

/-- Midline represents the midline of a quadrilateral -/
def Midline (ABCD : Quadrilateral) : ℝ :=
  sorry

/-- AC represents one diagonal of a quadrilateral -/
def AC (ABCD : Quadrilateral) : Point × Point :=
  (ABCD.A, ABCD.C)

/-- BD represents the other diagonal of a quadrilateral -/
def BD (ABCD : Quadrilateral) : Point × Point :=
  (ABCD.B, ABCD.D)

/-- Perpendicular represents that two line segments are perpendicular -/
def Perpendicular (l1 l2 : Point × Point) : Prop :=
  sorry

/-- Area represents the area of a quadrilateral -/
def Area (ABCD : Quadrilateral) : ℝ :=
  sorry

/-- An isosceles trapezoid with midline m and perpendicular diagonals has area m^2 -/
theorem isosceles_trapezoid_area (m : ℝ) (h : m > 0) :
  ∃ (ABCD : Quadrilateral),
    IsoscelesTrapezoid ABCD ∧
    Midline ABCD = m ∧
    Perpendicular (AC ABCD) (BD ABCD) →
    Area ABCD = m^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l982_98260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_equality_l982_98252

theorem ratio_equality (x y z w : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0) 
  (h_eq : y / (x - z) = (x + y + w) / (z + w) ∧ (x + y + w) / (z + w) = x / (y + w)) :
  x / (y + w) = 2 := by
  sorry

#check ratio_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_equality_l982_98252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_in_right_square_pyramid_l982_98212

-- Define a right square pyramid
structure RightSquarePyramid where
  V : EuclideanSpace ℝ (Fin 3)  -- apex
  A : EuclideanSpace ℝ (Fin 3)  -- base vertices
  B : EuclideanSpace ℝ (Fin 3)
  C : EuclideanSpace ℝ (Fin 3)
  D : EuclideanSpace ℝ (Fin 3)
  is_square_base : sorry  -- Placeholder for IsSquare condition
  is_right_pyramid : sorry  -- Placeholder for IsRightPyramid condition

-- Define a point inside the base
def PointInsideBase (pyramid : RightSquarePyramid) (P : EuclideanSpace ℝ (Fin 3)) : Prop :=
  sorry  -- Placeholder for inside condition

-- Define the sum of distances to triangular faces
noncomputable def SumDistancesToFaces (pyramid : RightSquarePyramid) (P : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  sorry  -- Placeholder for sum of distances to faces

-- Define the sum of distances to base sides
noncomputable def SumDistancesToBaseSides (pyramid : RightSquarePyramid) (P : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  sorry  -- Placeholder for sum of distances to base sides

-- The main theorem
theorem distance_ratio_in_right_square_pyramid (pyramid : RightSquarePyramid) (P : EuclideanSpace ℝ (Fin 3)) 
  (h_inside : PointInsideBase pyramid P) :
  SumDistancesToFaces pyramid P / SumDistancesToBaseSides pyramid P = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_in_right_square_pyramid_l982_98212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuckoo_clock_chimes_l982_98227

/-- Represents the number of chimes for a given hour on a cuckoo clock -/
def chimes (hour : Nat) : Nat :=
  if hour ≤ 12 then hour else hour - 12

/-- The sum of chimes over a period of 7 hours, starting from the 10th hour -/
def totalChimes : Nat :=
  (List.range 7).map (fun i => chimes (i + 10)) |>.sum

theorem cuckoo_clock_chimes : totalChimes = 43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuckoo_clock_chimes_l982_98227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magpie_polygon_theorem_l982_98282

/-- A regular n-gon with magpies on its vertices -/
structure MagpiePolygon where
  n : ℕ
  is_regular : n ≥ 3

/-- Represents the condition for a triangle to have all acute angles -/
def all_acute_angles (p : MagpiePolygon) (i j k : Fin p.n) : Prop := sorry

/-- Represents the condition for a triangle to have all right angles -/
def all_right_angles (p : MagpiePolygon) (i j k : Fin p.n) : Prop := sorry

/-- Represents the condition for a triangle to have all obtuse angles -/
def all_obtuse_angles (p : MagpiePolygon) (i j k : Fin p.n) : Prop := sorry

/-- The condition that three magpies form a specific type of triangle -/
def satisfies_triangle_condition (p : MagpiePolygon) : Prop :=
  ∀ (initial_positions final_positions : Fin p.n → Fin p.n),
    Function.Bijective initial_positions → Function.Bijective final_positions →
    ∃ (i j k : Fin p.n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
      (all_acute_angles p i j k ∨ all_right_angles p i j k ∨ all_obtuse_angles p i j k)

/-- The main theorem -/
theorem magpie_polygon_theorem (p : MagpiePolygon) :
  satisfies_triangle_condition p ↔ p.n ≥ 3 ∧ p.n ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magpie_polygon_theorem_l982_98282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_products_eq_n_l982_98230

open BigOperators Finset

def sum_reciprocal_products (n : ℕ) : ℚ :=
  ∑ k in range n.succ, ∑ s in powerset (range n),
    if s.card == k then (∏ i in s, (↑i + 1 : ℚ))⁻¹ else 0

theorem sum_reciprocal_products_eq_n (n : ℕ) :
  sum_reciprocal_products n = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_products_eq_n_l982_98230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_l982_98275

open Real

-- Define the function f
noncomputable def f (x : ℝ) := log ((1 + x) / (1 - x))

-- State the theorem
theorem f_composition (x : ℝ) (h : -1 < x ∧ x < 1) :
  f ((2*x + x^2) / (1 + 2*x^2)) = 2 * f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_l982_98275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gabe_in_seat_two_l982_98270

-- Define the people and seats
inductive Person : Type
  | Ella | Fiona | Gabe | Harry | Ivan

inductive Seat : Type
  | one | two | three | four | five

-- Define the seating arrangement
variable (seating : Person → Seat)

-- Define what it means to be next to someone
def next_to (p1 p2 : Person) : Prop :=
  (seating p1 = Seat.one ∧ seating p2 = Seat.two) ∨
  (seating p1 = Seat.two ∧ seating p2 = Seat.three) ∨
  (seating p1 = Seat.three ∧ seating p2 = Seat.four) ∨
  (seating p1 = Seat.four ∧ seating p2 = Seat.five) ∨
  (seating p2 = Seat.one ∧ seating p1 = Seat.two) ∨
  (seating p2 = Seat.two ∧ seating p1 = Seat.three) ∨
  (seating p2 = Seat.three ∧ seating p1 = Seat.four) ∨
  (seating p2 = Seat.four ∧ seating p1 = Seat.five)

-- Define what it means to be between two people
def between (p1 p2 p3 : Person) : Prop :=
  (seating p2 = Seat.two ∧ seating p1 = Seat.one ∧ seating p3 = Seat.three) ∨
  (seating p2 = Seat.three ∧ seating p1 = Seat.two ∧ seating p3 = Seat.four) ∨
  (seating p2 = Seat.four ∧ seating p1 = Seat.three ∧ seating p3 = Seat.five) ∨
  (seating p2 = Seat.two ∧ seating p3 = Seat.one ∧ seating p1 = Seat.three) ∨
  (seating p2 = Seat.three ∧ seating p3 = Seat.two ∧ seating p1 = Seat.four) ∨
  (seating p2 = Seat.four ∧ seating p3 = Seat.three ∧ seating p1 = Seat.five)

theorem gabe_in_seat_two :
  ∀ (seating : Person → Seat),
  (next_to seating Person.Fiona Person.Gabe) ∧
  ¬(between seating Person.Fiona Person.Ella Person.Gabe) ∧
  (seating Person.Fiona = Seat.three) →
  (seating Person.Gabe = Seat.two) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gabe_in_seat_two_l982_98270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_town_distance_theorem_l982_98213

/-- Represents a town with x and y coordinates -/
structure Town where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two towns -/
noncomputable def distance (t1 t2 : Town) : ℝ :=
  Real.sqrt ((t1.x - t2.x)^2 + (t1.y - t2.y)^2)

/-- Theorem about the distance between three towns -/
theorem town_distance_theorem (A B C : Town) 
  (h1 : distance A B = 8) 
  (h2 : distance B C = 10) : 
  ∃ (x : ℝ), distance A C = x ∧ 2 ≤ x ∧ x ≤ 18 ∧ 
  ¬∃! (y : ℝ), distance A C = y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_town_distance_theorem_l982_98213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l982_98202

-- Define the curves in polar coordinates
noncomputable def curve1 (θ : ℝ) : ℝ × ℝ := (2 / Real.sin θ, θ)
noncomputable def curve2 (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, θ)

-- Convert polar coordinates to Cartesian coordinates
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Calculate distance between two points in Cartesian coordinates
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_between_curves :
  ∃ (c : ℝ), c = 1 ∧ 
  ∀ (θ1 θ2 : ℝ), 
    distance (polar_to_cartesian (curve1 θ1).1 (curve1 θ1).2)
             (polar_to_cartesian (curve2 θ2).1 (curve2 θ2).2) ≥ c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l982_98202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l982_98266

/-- The original function before transformation -/
noncomputable def original_function (x : ℝ) : ℝ := 2^(x-1) + 3

/-- The transformation applied to the graph -/
def transform (x y : ℝ) : ℝ × ℝ := (x + 1, y + 2)

/-- The resulting function after transformation -/
noncomputable def resulting_function (x : ℝ) : ℝ := 2^x + 1

/-- Theorem stating that the transformation of the original function results in the resulting function -/
theorem graph_transformation (x : ℝ) : 
  (transform x (original_function x)).2 = resulting_function ((transform x (original_function x)).1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l982_98266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_5_l982_98211

/-- The equation of motion for a body -/
noncomputable def s (t : ℝ) : ℝ := (1/4) * t^4 - 3

/-- The velocity function derived from the equation of motion -/
noncomputable def v (t : ℝ) : ℝ := deriv s t

/-- Theorem: The instantaneous velocity at t = 5 is 125 -/
theorem instantaneous_velocity_at_5 : v 5 = 125 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_5_l982_98211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_l982_98278

/-- Given a cube with edge length a, the total surface area of the pyramid formed by
    connecting the center of the upper face to the vertices of the base is a^2(1+√5). -/
theorem pyramid_surface_area (a : ℝ) (h : a > 0) :
  a^2 * (1 + Real.sqrt 5) = a^2 * (1 + Real.sqrt 5) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_l982_98278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mona_age_l982_98200

def guesses : List Nat := [16, 25, 27, 32, 36, 40, 42, 49, 64, 81]

def is_perfect_square (n : Nat) : Prop := ∃ m : Nat, n = m * m

def count_lower_guesses (age : Nat) (guesses : List Nat) : Nat :=
  (guesses.filter (fun g => g < age)).length

def off_by_one_count (age : Nat) (guesses : List Nat) : Nat :=
  (guesses.filter (fun n => n = age - 1 ∨ n = age + 1)).length

theorem mona_age : ∃ (age : Nat),
  age ∈ guesses ∧
  is_perfect_square age ∧
  count_lower_guesses age guesses ≥ guesses.length / 2 ∧
  off_by_one_count age guesses = 2 ∧
  age = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mona_age_l982_98200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_upper_bound_l982_98218

def sequenceProperty (x : ℕ → ℝ) :=
  x 1 = 0 ∧ ∀ i, x (i + 1) = x i + (1 / 30000) * Real.sqrt (1 - x i ^ 2)

theorem sequence_upper_bound (x : ℕ → ℝ) (n : ℕ) (h : sequenceProperty x) :
  x n < 1 → n < 50000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_upper_bound_l982_98218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l982_98225

noncomputable def a : ℝ := ∫ x in (0:ℝ)..(2:ℝ), (1 - 3 * x^2) + 4

def n : ℕ := 6

theorem sum_of_coefficients :
  (1 + 1 / (-2:ℝ))^n = 1/64 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l982_98225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_greater_equal_04_l982_98290

def count_greater_equal (numbers : List ℚ) (threshold : ℚ) : ℕ :=
  (numbers.filter (λ x => x ≥ threshold)).length

theorem count_greater_equal_04 : count_greater_equal [8/10, 1/2, 9/10] (4/10) = 3 := by
  rfl

#eval count_greater_equal [8/10, 1/2, 9/10] (4/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_greater_equal_04_l982_98290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_business_dinner_max_cost_l982_98223

/-- The maximum cost of food given total allowable spending, sales tax rate, and tip rate -/
noncomputable def max_food_cost (total_allowable : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  total_allowable / (1 + sales_tax_rate + tip_rate)

/-- Theorem stating the maximum food cost for the given conditions -/
theorem business_dinner_max_cost :
  let total_allowable : ℝ := 50
  let sales_tax_rate : ℝ := 0.07
  let tip_rate : ℝ := 0.15
  let result := max_food_cost total_allowable sales_tax_rate tip_rate
  ∃ ε > 0, |result - 40.98| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_business_dinner_max_cost_l982_98223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_object_displacement_l982_98257

-- Define the velocity function
def v (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 3

-- Define the displacement function
noncomputable def displacement (a b : ℝ) : ℝ := ∫ t in a..b, v t

-- Theorem statement
theorem object_displacement : displacement 0 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_object_displacement_l982_98257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_terms_count_l982_98249

theorem expansion_terms_count (a b : ℝ) : 
  (∃ c₁ c₂ c₃ c₄ c₅ c₆ c₇ : ℝ, 
    ((a + 3*b)^3 * (a - 3*b)^3)^2 = 
      c₁*a^12 + c₂*a^10*b^2 + c₃*a^8*b^4 + c₄*a^6*b^6 + 
      c₅*a^4*b^8 + c₆*a^2*b^10 + c₇*b^12) ∧
  (∀ d₁ d₂ d₃ d₄ d₅ d₆ d₇ d₈ : ℝ, ∀ n m : ℕ, 
    ((a + 3*b)^3 * (a - 3*b)^3)^2 ≠ 
      d₁*a^12 + d₂*a^10*b^2 + d₃*a^8*b^4 + d₄*a^6*b^6 + 
      d₅*a^4*b^8 + d₆*a^2*b^10 + d₇*b^12 + d₈*a^n*b^m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_terms_count_l982_98249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_squared_l982_98243

-- Define the circles
def circle1 : ℝ × ℝ → Prop := λ p => (p.1 - 2)^2 + p.2^2 = 25
def circle2 : ℝ × ℝ → Prop := λ p => (p.1 - 7)^2 + p.2^2 = 4

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {p | circle1 p ∧ circle2 p}

-- Theorem statement
theorem intersection_distance_squared :
  ∃ p q : ℝ × ℝ, p ∈ intersection_points ∧ q ∈ intersection_points ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = 15.3664 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_squared_l982_98243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_system_equations_l982_98228

/-- A system of two cylinders connected by a weightless string with a hanging mass -/
structure CylinderSystem where
  g : ℝ  -- gravitational acceleration
  m : ℝ  -- hanging mass
  m₁ : ℝ  -- mass of first cylinder
  r₁ : ℝ  -- radius of first cylinder
  m₂ : ℝ  -- mass of second cylinder
  r₂ : ℝ  -- radius of second cylinder

/-- The equations for angular accelerations and tensions when the rope doesn't slip -/
noncomputable def no_slip_equations (s : CylinderSystem) :
  (ℝ × ℝ × ℝ × ℝ) :=
  let β₁ := (s.g / s.r₁) * (2 * s.m / (2 * s.m + s.m₁ + s.m₂))
  let β₂ := (s.g / s.r₂) * (2 * s.m / (2 * s.m + s.m₁ + s.m₂))
  let K₁ := s.m₁ * s.g * (s.m / (2 * s.m + s.m₁ + s.m₂))
  let K₂ := s.m * s.g * ((s.m₁ + s.m₂) / (2 * s.m + s.m₁ + s.m₂))
  (β₁, β₂, K₁, K₂)

theorem cylinder_system_equations (s : CylinderSystem) :
  let (β₁, β₂, K₁, K₂) := no_slip_equations s
  (β₁ = (s.g / s.r₁) * (2 * s.m / (2 * s.m + s.m₁ + s.m₂))) ∧
  (β₂ = (s.g / s.r₂) * (2 * s.m / (2 * s.m + s.m₁ + s.m₂))) ∧
  (K₁ = s.m₁ * s.g * (s.m / (2 * s.m + s.m₁ + s.m₂))) ∧
  (K₂ = s.m * s.g * ((s.m₁ + s.m₂) / (2 * s.m + s.m₁ + s.m₂))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_system_equations_l982_98228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_adjacent_uncovered_l982_98287

/-- Represents a 6x6 chessboard with some dominoes placed -/
structure Chessboard :=
  (uncovered : Finset (Fin 6 × Fin 6))
  (h_uncovered : uncovered.card = 14)

/-- Two squares are adjacent if they share an edge -/
def adjacent (a b : Fin 6 × Fin 6) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ b.2 = a.2 + 1)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ b.1 = a.1 + 1))

/-- Main theorem: There exists a pair of adjacent uncovered squares -/
theorem exists_adjacent_uncovered (board : Chessboard) :
  ∃ (a b : Fin 6 × Fin 6), a ∈ board.uncovered ∧ b ∈ board.uncovered ∧ adjacent a b :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_adjacent_uncovered_l982_98287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_13_5_l982_98254

/-- The line equation y = 9 - 3x --/
def line_equation (x : ℝ) : ℝ := 9 - 3 * x

/-- The x-intercept of the line --/
noncomputable def x_intercept : ℝ := 3

/-- The y-intercept of the line --/
noncomputable def y_intercept : ℝ := 9

/-- The area of the triangle --/
noncomputable def triangle_area : ℝ := (1 / 2) * x_intercept * y_intercept

/-- Theorem: The area of the triangle bounded by y = 9 - 3x and the coordinate axes is 13.5 --/
theorem triangle_area_is_13_5 : triangle_area = 13.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_13_5_l982_98254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_main_purpose_l982_98277

/-- Represents a categorical variable -/
structure CategoricalVariable where
  name : String

/-- Represents an independence test -/
structure IndependenceTest where
  var1 : CategoricalVariable
  var2 : CategoricalVariable

/-- The purpose of an independence test -/
def purpose_of_independence_test (test : IndependenceTest) : Prop :=
  ∃ (reliability : Prop), reliability

/-- Theorem stating the main purpose of an independence test -/
theorem independence_test_main_purpose (test : IndependenceTest) :
  purpose_of_independence_test test := by
  sorry

#check independence_test_main_purpose

end NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_main_purpose_l982_98277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_25pi_div_6_sin_alpha_value_l982_98209

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x

-- Theorem for part I
theorem f_at_25pi_div_6 : f (25 * Real.pi / 6) = 0 := by sorry

-- Theorem for part II
theorem sin_alpha_value (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi) 
  (h3 : f (α / 2) = 1 / 4 - Real.sqrt 3 / 2) : 
  Real.sin α = (1 + 3 * Real.sqrt 5) / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_25pi_div_6_sin_alpha_value_l982_98209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l982_98247

theorem trigonometric_identities (x : ℝ) 
    (h1 : -1 < x) (h2 : x < 0) (h3 : Real.sin x + Real.cos x = Real.sqrt 2) : 
    (Real.sin x - Real.cos x = -Real.sqrt (2 - Real.sqrt 2)) ∧ 
    (Real.sin x ^ 2 + Real.cos x ^ 2 = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l982_98247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l982_98235

/-- The distance between the foci of an ellipse -/
noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 45 + y^2 / 9 = 6

theorem ellipse_foci_distance :
  distance_between_foci (Real.sqrt 7.5) (Real.sqrt 1.5) = 2 * Real.sqrt 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l982_98235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_translation_l982_98246

-- Define a generic function g
variable (g : ℝ → ℝ)

-- Define the translated function
def g_translated (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + 2

-- Theorem statement
theorem vertical_translation (x : ℝ) :
  g_translated g x = g x + 2 :=
by
  -- Unfold the definition of g_translated
  unfold g_translated
  -- The equality now holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_translation_l982_98246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_orders_l982_98221

theorem ice_cream_orders (total : ℕ) (vanilla_percent : ℚ) : 
  total = 220 → vanilla_percent = 1/5 → 
  ∃ (chocolate vanilla : ℕ), 
    chocolate + vanilla = total ∧ 
    vanilla = 2 * chocolate ∧
    vanilla = (vanilla_percent * ↑total).floor ∧
    chocolate = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_orders_l982_98221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l982_98284

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 - x + Real.sin x

theorem tangent_line_at_zero (x : ℝ) :
  let y : ℝ → ℝ := λ t => (deriv f) 0 * t + f 0
  ∀ t, y t = t + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l982_98284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_l982_98215

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- represents ax + by + c = 0

/-- The number of intersection points between a circle and a line --/
def circleLineIntersections (c : Circle) (l : Line) : ℕ := 
  sorry

/-- The number of intersection points between two lines --/
def lineLineIntersections (l1 l2 : Line) : ℕ := 
  sorry

/-- Theorem: The maximum number of intersection points between one circle and three distinct lines in a plane is 9 --/
theorem max_intersections (c : Circle) (l1 l2 l3 : Line) :
  (∃ (n : ℕ), n ≤ 9 ∧
    n = circleLineIntersections c l1 +
        circleLineIntersections c l2 +
        circleLineIntersections c l3 +
        lineLineIntersections l1 l2 +
        lineLineIntersections l2 l3 +
        lineLineIntersections l1 l3) ∧
  (∀ (m : ℕ), m > 9 →
    m > circleLineIntersections c l1 +
        circleLineIntersections c l2 +
        circleLineIntersections c l3 +
        lineLineIntersections l1 l2 +
        lineLineIntersections l2 l3 +
        lineLineIntersections l1 l3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_l982_98215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rubiks_cube_diagonal_path_impossible_l982_98241

/-- Represents the surface of a Rubik's cube -/
structure RubiksCubeSurface where
  facelets : Nat
  vertices : Nat
  vertex_connections : Nat → Nat

/-- Defines the properties of a standard Rubik's cube surface -/
def standard_rubiks_cube : RubiksCubeSurface :=
  { facelets := 54,
    vertices := 56,
    vertex_connections := λ n => if n = 3 ∨ n = 4 then n else 0 }

/-- Represents a path on the surface of a Rubik's cube -/
def CubePath := List Nat

/-- Checks if a path is non-self-intersecting -/
def is_non_self_intersecting (p : CubePath) : Prop :=
  ∀ i j, i ≠ j → p.get! i ≠ p.get! j

/-- Theorem stating the impossibility of drawing a non-self-intersecting path 
    through all facelets of a Rubik's cube -/
theorem rubiks_cube_diagonal_path_impossible (cube : RubiksCubeSurface) 
  (h_cube : cube = standard_rubiks_cube) :
  ¬∃ (p : CubePath), p.length = cube.facelets ∧ is_non_self_intersecting p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rubiks_cube_diagonal_path_impossible_l982_98241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_lucky_correct_expected_lucky_pairs_correct_expected_lucky_pairs_gt_half_l982_98219

/-- Represents the number of pairs of socks -/
def n : ℕ := sorry

/-- Assumption that n is positive -/
axiom hn : n > 0

/-- Probability that all pairs are lucky -/
noncomputable def prob_all_lucky (n : ℕ) : ℚ :=
  (2^n * n.factorial) / ((2*n).factorial)

/-- Expected number of lucky pairs -/
noncomputable def expected_lucky_pairs (n : ℕ) : ℚ :=
  n / (2*n - 1)

/-- Theorem stating the probability of all pairs being lucky -/
theorem prob_all_lucky_correct :
  prob_all_lucky n = (2^n * n.factorial) / ((2*n).factorial) := by
  sorry

/-- Theorem stating the expected number of lucky pairs -/
theorem expected_lucky_pairs_correct :
  expected_lucky_pairs n = n / (2*n - 1) := by
  sorry

/-- Theorem proving that the expected number of lucky pairs is greater than 0.5 -/
theorem expected_lucky_pairs_gt_half :
  expected_lucky_pairs n > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_lucky_correct_expected_lucky_pairs_correct_expected_lucky_pairs_gt_half_l982_98219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_first_quadrant_l982_98271

/-- Predicate to check if a point is on the terminal side of an angle -/
def is_on_terminal_side (P : ℝ × ℝ) (α : ℝ) : Prop :=
  sorry

/-- Predicate to check if an angle is in the first quadrant -/
def is_in_first_quadrant (α : ℝ) : Prop :=
  0 < α ∧ α < Real.pi / 2

/-- Given a point P(√3/2, 1/2) on the terminal side of angle α, prove that α is in the first quadrant. -/
theorem angle_in_first_quadrant (α : ℝ) (P : ℝ × ℝ) :
  P.1 = Real.sqrt 3 / 2 ∧ P.2 = 1 / 2 ∧ is_on_terminal_side P α → is_in_first_quadrant α :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_first_quadrant_l982_98271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l982_98274

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  25 * x^2 + 100 * x + 4 * y^2 + 8 * y + 9 = 0

/-- The distance between the foci of the ellipse -/
noncomputable def foci_distance : ℝ := 4 * Real.sqrt 5

/-- Theorem: The distance between the foci of the ellipse given by the equation
    25x^2 + 100x + 4y^2 + 8y + 9 = 0 is 4√5 -/
theorem ellipse_foci_distance :
  ∀ x y : ℝ, ellipse_equation x y → foci_distance = 4 * Real.sqrt 5 :=
by
  sorry

#check ellipse_foci_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l982_98274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_form_ratio_2018_l982_98250

/-- Represents the set of digits on the cards -/
def Digits : Finset Nat := Finset.range 10

/-- The sum of all digits on the original 20 cards -/
def TotalSum : Nat := 2 * (Finset.sum Digits id)

/-- The sum of digits on the remaining 19 cards -/
def RemainingSum : Nat := TotalSum - 1

/-- Checks if a number is divisible by 3 -/
def DivisibleBy3 (n : Nat) : Prop := n % 3 = 0

theorem cannot_form_ratio_2018 : 
  ¬∃ (a b : Nat), a ≠ 0 ∧ b ≠ 0 ∧ 
  (Nat.digits 10 a).length + (Nat.digits 10 b).length = 19 ∧
  (Nat.digits 10 a).sum + (Nat.digits 10 b).sum = RemainingSum ∧
  (a : ℚ) / b = 2018 := by
  sorry

#eval TotalSum
#eval RemainingSum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_form_ratio_2018_l982_98250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l982_98233

noncomputable section

-- Define the function f
def f (p q x : ℝ) : ℝ := p * x + q / x

-- State the theorem
theorem function_properties :
  ∃ p q : ℝ,
  (f p q 1 = 5/2) ∧
  (f p q 2 = 17/4) ∧
  (∀ x : ℝ, x > 0 → f p q x = 2 * x + 1 / (2 * x)) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1/2 → f p q x₁ > f p q x₂) ∧
  (∀ m : ℝ, (∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → f p q x ≥ 2 - m) ↔ m ≥ 0) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l982_98233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_l982_98273

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular box -/
noncomputable def boxVolume (d : BoxDimensions) : ℝ := d.length * d.width * d.height

/-- Calculates the volume of a cube -/
noncomputable def cubeVolume (side : ℝ) : ℝ := side ^ 3

/-- Calculates the percentage of volume removed -/
noncomputable def percentageVolumeRemoved (boxDim : BoxDimensions) (cubeSide : ℝ) : ℝ :=
  (8 * cubeVolume cubeSide) / boxVolume boxDim * 100

/-- The main theorem stating the percentage of volume removed -/
theorem volume_removed_percentage :
  let boxDim : BoxDimensions := ⟨20, 12, 10⟩
  let cubeSide : ℝ := 4
  abs (percentageVolumeRemoved boxDim cubeSide - 21.33) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_l982_98273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_hyperbola_line_intersection_l982_98299

/-- Eccentricity range for a hyperbola intersecting a line at two points -/
theorem eccentricity_range_hyperbola_line_intersection (a : ℝ) (h_a_pos : a > 0) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧
    x₁ + y₁ = 1 ∧
    x₂ + y₂ = 1 ∧
    x₁^2 / a^2 - y₁^2 = 1 ∧
    x₂^2 / a^2 - y₂^2 = 1) →
  let e := Real.sqrt (1 + 1 / a^2)
  e ∈ Set.Ioo (Real.sqrt 6 / 2) (Real.sqrt 2) ∪ Set.Ioi (Real.sqrt 2) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_hyperbola_line_intersection_l982_98299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pretzels_sold_is_36_l982_98237

/-- Represents the candy store's sales and revenue --/
structure CandyStore where
  fudge_pounds : ℕ
  fudge_price : ℚ
  truffle_dozens : ℕ
  truffle_price : ℚ
  pretzel_price : ℚ
  total_revenue : ℚ

/-- Calculates the number of dozens of chocolate-covered pretzels sold --/
def pretzels_sold (store : CandyStore) : ℕ :=
  let fudge_revenue := store.fudge_pounds * store.fudge_price
  let truffle_revenue := (store.truffle_dozens * 12) * store.truffle_price
  let pretzel_revenue := store.total_revenue - fudge_revenue - truffle_revenue
  (pretzel_revenue / store.pretzel_price).floor.toNat

/-- Theorem stating the number of dozens of chocolate-covered pretzels sold --/
theorem pretzels_sold_is_36 (store : CandyStore) 
  (h1 : store.fudge_pounds = 20)
  (h2 : store.fudge_price = 5/2)
  (h3 : store.truffle_dozens = 5)
  (h4 : store.truffle_price = 3/2)
  (h5 : store.pretzel_price = 2)
  (h6 : store.total_revenue = 212) :
  pretzels_sold store = 36 := by
  sorry

#eval pretzels_sold { fudge_pounds := 20, fudge_price := 5/2, truffle_dozens := 5, truffle_price := 3/2, pretzel_price := 2, total_revenue := 212 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pretzels_sold_is_36_l982_98237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonals_equal_rhombus_not_necessarily_l982_98268

-- Define a parallelogram
structure Parallelogram :=
  (diagonals_bisect : Bool)

-- Define a rectangle as a parallelogram with right angles and equal diagonals
structure Rectangle extends Parallelogram :=
  (right_angles : Bool)
  (diagonals_equal : Bool)

-- Define a rhombus as a parallelogram with all sides equal
structure Rhombus extends Parallelogram :=
  (all_sides_equal : Bool)

-- Axiom: Not all rhombuses have equal diagonals
axiom not_all_rhombuses_equal_diagonals :
  ¬∀ (rh : Rhombus), ∃ (diagonals_equal : Bool), diagonals_equal = true

-- Theorem statement
theorem rectangle_diagonals_equal_rhombus_not_necessarily :
  ∃ (r : Rectangle), r.diagonals_equal = true ∧
  ¬∀ (rh : Rhombus), ∃ (diagonals_equal : Bool), diagonals_equal = true :=
by
  -- We construct a rectangle with equal diagonals
  let r : Rectangle := ⟨⟨true⟩, true, true⟩
  
  -- We use the axiom that not all rhombuses have equal diagonals
  have h : ¬∀ (rh : Rhombus), ∃ (diagonals_equal : Bool), diagonals_equal = true :=
    not_all_rhombuses_equal_diagonals
  
  -- We combine these facts to prove our theorem
  exact ⟨r, ⟨rfl, h⟩⟩

-- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonals_equal_rhombus_not_necessarily_l982_98268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_zero_digits_in_base_representation_l982_98255

/-- Returns the number of non-zero digits in the base b representation of n -/
def num_non_zero_digits_base (b : ℕ) (n : ℕ) : ℕ := by
  sorry

/-- Given positive integers a, b, and n, where b > 1 and b^n - 1 divides a,
    the number of non-zero digits in the base b representation of a is greater than or equal to n. -/
theorem non_zero_digits_in_base_representation (a b n : ℕ) 
  (h1 : b > 1) 
  (h2 : (b^n - 1) ∣ a) : 
  (num_non_zero_digits_base b a) ≥ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_zero_digits_in_base_representation_l982_98255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_a_had_most_balls_l982_98262

/-- Represents the number of balls in each box -/
structure BoxState where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

/-- Transfers balls from one box to others, doubling the balls in the source box -/
def transfer (state : BoxState) (source : Fin 4) : BoxState :=
  match source with
  | 0 => ⟨2 * state.a, state.b + state.a, state.c + state.a, state.d + state.a⟩
  | 1 => ⟨state.a + state.b, 2 * state.b, state.c + state.b, state.d + state.b⟩
  | 2 => ⟨state.a + state.c, state.b + state.c, 2 * state.c, state.d + state.c⟩
  | 3 => ⟨state.a + state.d, state.b + state.d, state.c + state.d, 2 * state.d⟩

/-- The final state after all transfers -/
def finalState (initial : BoxState) : BoxState :=
  transfer (transfer (transfer (transfer initial 0) 1) 2) 3

/-- Helper function to get the number of balls in a box given its index -/
def getBoxValue (state : BoxState) (index : Fin 4) : ℕ :=
  match index with
  | 0 => state.a
  | 1 => state.b
  | 2 => state.c
  | 3 => state.d

theorem box_a_had_most_balls :
  ∃ (initial : BoxState),
    finalState initial = ⟨16, 16, 16, 16⟩ ∧
    initial.a = 33 ∧
    ∀ (x : Fin 4), initial.a ≥ getBoxValue initial x :=
by sorry

#check box_a_had_most_balls

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_a_had_most_balls_l982_98262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solvability_l982_98222

theorem equation_solvability (x y a : ℝ) : 
  (∃ x y, (Real.cos x)^2 - 2 * (Real.cos x) * (Real.cos y) * (Real.cos (x + y)) + (Real.cos (x + y))^2 = a) ↔ 0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solvability_l982_98222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_both_ends_probability_l982_98289

/-- The number of people standing in a row -/
def n : ℕ := 5

/-- The probability that A and B do not stand at the two ends simultaneously -/
def prob : ℚ := 9/10

/-- Theorem stating the probability of A and B not standing at the two ends simultaneously -/
theorem not_both_ends_probability :
  (n.factorial - 2 * (n - 2).factorial) / n.factorial = prob :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_both_ends_probability_l982_98289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_proof_l982_98216

-- Define the constant speed of the car
noncomputable def car_speed : ℝ := 73.5

-- Define the reference speed (80 km/hr)
noncomputable def reference_speed : ℝ := 80

-- Define the distance traveled
noncomputable def distance : ℝ := 2

-- Define the additional time taken
noncomputable def additional_time : ℝ := 8 / 3600

-- Theorem statement
theorem car_speed_proof :
  distance / car_speed = distance / reference_speed + additional_time := by
  sorry

#eval IO.println "Theorem stated successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_proof_l982_98216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_triangle_ac_ratio_range_l982_98263

open Real

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to A
  b : ℝ  -- Side opposite to B
  c : ℝ  -- Side opposite to C
  angle_sum : A + B + C = π
  arithmetic_seq : 2 * B = A + C
  sine_law : a / (sin A) = b / (sin B)
  side_ratio : a / b = (a + b) / (a + b + c)

-- Theorem 1: If a/b = (a+b)/(a+b+c), then angle C = π/2
theorem right_angled_triangle (t : Triangle) : t.C = π / 2 := by
  sorry

-- Theorem 2: If the triangle is not obtuse, then 1/2 ≤ a/c ≤ 2
theorem ac_ratio_range (t : Triangle) (h : t.A ≤ π / 2 ∧ t.B ≤ π / 2 ∧ t.C ≤ π / 2) :
  1 / 2 ≤ t.a / t.c ∧ t.a / t.c ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_triangle_ac_ratio_range_l982_98263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_given_tan_l982_98292

theorem cos_value_given_tan (α : Real) : 
  0 < α ∧ α < Real.pi/2 →  -- α is in the first quadrant
  Real.tan α = 1/2 → 
  Real.cos α = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_given_tan_l982_98292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_3025th_term_l982_98288

def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d => d * d) |>.sum

def sequence_term (n : ℕ) : ℕ → ℕ
  | 0 => 3025
  | m + 1 => sum_of_squares_of_digits (sequence_term n m)

theorem sequence_3025th_term :
  sequence_term 0 3024 = 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_3025th_term_l982_98288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l982_98206

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x < 0) ↔ (∀ x : ℝ, x > 0 → Real.log x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l982_98206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rainfall_calculation_l982_98279

/-- The average rainfall per hour in Mawsynram, India, during February 2020 -/
noncomputable def average_rainfall_per_hour : ℝ :=
  280 / (29 * 24)

/-- The total rainfall in Mawsynram, India, during February 2020 -/
def total_rainfall : ℝ := 280

/-- The number of days in February 2020 (a leap year) -/
def days_in_february_2020 : ℕ := 29

/-- The number of hours in a day -/
def hours_in_day : ℕ := 24

theorem average_rainfall_calculation :
  average_rainfall_per_hour = total_rainfall / (days_in_february_2020 * hours_in_day) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rainfall_calculation_l982_98279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sine_intersections_l982_98231

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the sine function
noncomputable def sine_func (x : ℝ) : ℝ := Real.sin x

-- Define the x-value range
def x_range (x : ℝ) : Prop := -2 * Real.pi ≤ x ∧ x ≤ 2 * Real.pi

-- Define an intersection point
def intersection_point (x : ℝ) : Prop :=
  x_range x ∧ circle_eq x (sine_func x)

-- Theorem statement
theorem circle_sine_intersections :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x, x ∈ s ↔ intersection_point x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sine_intersections_l982_98231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclical_3subset_bounds_l982_98204

/-- Represents a tournament with 2n+1 teams --/
structure Tournament (n : ℕ) where
  teams : Fin (2*n + 1) → Type
  plays_once : ∀ (i j : Fin (2*n + 1)), i ≠ j → Prop
  has_winner : ∀ (i j : Fin (2*n + 1)) (h : i ≠ j), plays_once i j h → Prop

/-- A cyclical 3-subset in the tournament --/
def cyclical_3subset (n : ℕ) (t : Tournament n) (a b c : Fin (2*n + 1)) : Prop :=
  ∃ (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a),
    ∃ (p_ab : t.plays_once a b hab)
      (p_bc : t.plays_once b c hbc)
      (p_ca : t.plays_once c a hca),
    t.has_winner a b hab p_ab ∧
    t.has_winner b c hbc p_bc ∧
    t.has_winner c a hca p_ca

/-- The number of cyclical 3-subsets in a tournament --/
def num_cyclical_3subsets (n : ℕ) (t : Tournament n) : ℕ :=
  sorry

/-- Theorem stating the minimum and maximum number of cyclical 3-subsets --/
theorem cyclical_3subset_bounds (n : ℕ) :
  (∃ t : Tournament n, num_cyclical_3subsets n t = 0) ∧
  (∀ t : Tournament n, num_cyclical_3subsets n t ≤ (6*n^3 + n^2 - n) / 6) ∧
  (∃ t : Tournament n, num_cyclical_3subsets n t = (6*n^3 + n^2 - n) / 6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclical_3subset_bounds_l982_98204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_properties_l982_98286

/-- Represents a triangle with given side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Represents a parallelepiped with given edge lengths -/
structure Parallelepiped where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ

/-- Creates a parallelepiped from a triangle by folding along medians -/
noncomputable def createParallelepiped (t : Triangle) : Parallelepiped :=
  { edge1 := t.side1 / 2
  , edge2 := t.side2 / 2
  , edge3 := t.side3 / 2 }

/-- Checks if a parallelepiped is rectangular -/
def isRectangular (p : Parallelepiped) (t : Triangle) : Prop :=
  p.edge1^2 + p.edge2^2 = (t.side1 / 2)^2 ∧
  p.edge1^2 + p.edge3^2 = (t.side2 / 2)^2 ∧
  p.edge2^2 + p.edge3^2 = (t.side3 / 2)^2

/-- Calculates the volume of a parallelepiped -/
noncomputable def volume (p : Parallelepiped) : ℝ :=
  p.edge1 * p.edge2 * p.edge3

/-- Main theorem -/
theorem parallelepiped_properties (t : Triangle) 
    (h1 : t.side1 = 34) 
    (h2 : t.side2 = 30) 
    (h3 : t.side3 = 8 * Real.sqrt 13) : 
  let p := createParallelepiped t
  isRectangular p t ∧ volume p = 1224 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_properties_l982_98286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_price_yields_more_profit_l982_98239

/-- Represents the dried shrimp sales problem -/
structure ShrimpSales where
  total_weight : ℝ
  purchase_price : ℝ
  min_selling_price : ℝ
  max_selling_price : ℝ
  base_sales : ℝ
  sales_increase_rate : ℝ
  daily_expense : ℝ

/-- Calculate daily sales volume based on selling price -/
noncomputable def daily_sales_volume (s : ShrimpSales) (price : ℝ) : ℝ :=
  s.base_sales + s.sales_increase_rate * (s.max_selling_price - price)

/-- Calculate daily sales revenue based on selling price -/
noncomputable def daily_sales_revenue (s : ShrimpSales) (price : ℝ) : ℝ :=
  price * (daily_sales_volume s price)

/-- Calculate daily profit based on selling price -/
noncomputable def daily_profit (s : ShrimpSales) (price : ℝ) : ℝ :=
  (price - s.purchase_price) * (daily_sales_volume s price) - s.daily_expense

/-- Calculate total profit for a given selling price -/
noncomputable def total_profit (s : ShrimpSales) (price : ℝ) : ℝ :=
  (s.total_weight / daily_sales_volume s price) * daily_profit s price

/-- The main theorem: Highest price yields more total profit -/
theorem highest_price_yields_more_profit (s : ShrimpSales) :
  s.total_weight = 2000 ∧
  s.purchase_price = 20 ∧
  s.min_selling_price = 20 ∧
  s.max_selling_price = 50 ∧
  s.base_sales = 30 ∧
  s.sales_increase_rate = 2 ∧
  s.daily_expense = 400 →
  total_profit s s.max_selling_price > total_profit s (85/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_price_yields_more_profit_l982_98239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l982_98272

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  f1 : Point  -- First focus
  f2 : Point  -- Second focus
  p : Point   -- Point on the hyperbola

/-- Calculate the dot product of two vectors -/
def dot_product (p1 p2 p3 p4 : Point) : ℝ :=
  (p3.x - p1.x) * (p4.x - p2.x) + (p3.y - p1.y) * (p4.y - p2.y)

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Check if the given hyperbola satisfies the conditions -/
def is_valid_hyperbola (h : Hyperbola) : Prop :=
  h.f1.x = -Real.sqrt 5 ∧ h.f1.y = 0 ∧
  h.f2.x = Real.sqrt 5 ∧ h.f2.y = 0 ∧
  dot_product h.p h.f1 h.p h.f2 = 0 ∧
  distance h.p h.f1 * distance h.p h.f2 = 2

/-- The standard equation of the hyperbola -/
def standard_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 = 1

theorem hyperbola_standard_equation (h : Hyperbola) 
  (h_valid : is_valid_hyperbola h) : 
  ∀ x y, x = h.p.x ∧ y = h.p.y → standard_equation x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l982_98272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tammy_total_distance_l982_98258

/-- Represents a segment of Tammy's journey --/
structure Segment where
  speed : ℝ  -- Speed in miles per hour
  time : ℝ   -- Time driven in hours

/-- Calculates the distance traveled in a segment --/
def distance (s : Segment) : ℝ := s.speed * s.time

/-- Tammy's journey segments --/
def journey : List Segment := [
  ⟨70, 2⟩,
  ⟨60, 3⟩,
  ⟨55, 2⟩,
  ⟨65, 4⟩
]

/-- Theorem: The total distance Tammy drove is 690 miles --/
theorem tammy_total_distance : 
  (journey.map distance).sum = 690 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tammy_total_distance_l982_98258
