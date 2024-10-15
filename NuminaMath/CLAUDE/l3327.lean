import Mathlib

namespace NUMINAMATH_CALUDE_square_product_equality_l3327_332744

theorem square_product_equality : (15 : ℕ)^2 * 9^2 * 356 = 6489300 := by
  sorry

end NUMINAMATH_CALUDE_square_product_equality_l3327_332744


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3327_332783

-- Define the universal set I
def I : Set (ℝ × ℝ) := Set.univ

-- Define set A
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 - 3 = p.1 - 2 ∧ p.1 ≠ 2}

-- Define set B
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 1}

-- State the theorem
theorem complement_A_intersect_B : (I \ A) ∩ B = {(2, 3)} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3327_332783


namespace NUMINAMATH_CALUDE_min_dihedral_angle_cube_l3327_332772

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A unit cube with vertices ABCD-A₁B₁C₁D₁ -/
structure UnitCube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- A point P on edge AB of the cube -/
def P (cube : UnitCube) (t : ℝ) : Point3D :=
  { x := cube.A.x + t * (cube.B.x - cube.A.x),
    y := cube.A.y + t * (cube.B.y - cube.A.y),
    z := cube.A.z + t * (cube.B.z - cube.A.z) }

/-- The dihedral angle between two planes -/
def dihedralAngle (plane1 : Plane3D) (plane2 : Plane3D) : ℝ := sorry

/-- The plane PDB₁ -/
def planePDB₁ (cube : UnitCube) (p : Point3D) : Plane3D := sorry

/-- The plane ADD₁A₁ -/
def planeADD₁A₁ (cube : UnitCube) : Plane3D := sorry

theorem min_dihedral_angle_cube (cube : UnitCube) :
  ∃ (t : ℝ), t ∈ Set.Icc 0 1 ∧
    ∀ (s : ℝ), s ∈ Set.Icc 0 1 →
      dihedralAngle (planePDB₁ cube (P cube t)) (planeADD₁A₁ cube) ≤
      dihedralAngle (planePDB₁ cube (P cube s)) (planeADD₁A₁ cube) ∧
    dihedralAngle (planePDB₁ cube (P cube t)) (planeADD₁A₁ cube) = Real.arctan (Real.sqrt 2 / 2) := by
  sorry


end NUMINAMATH_CALUDE_min_dihedral_angle_cube_l3327_332772


namespace NUMINAMATH_CALUDE_correct_answers_for_given_score_l3327_332775

/-- Represents a test with a scoring system and a student's performance. -/
structure Test where
  total_questions : ℕ
  correct_answers : ℕ
  score : ℤ

/-- Calculates the score based on correct and incorrect answers. -/
def calculate_score (test : Test) : ℤ :=
  (test.correct_answers : ℤ) - 2 * ((test.total_questions - test.correct_answers) : ℤ)

theorem correct_answers_for_given_score (test : Test) :
  test.total_questions = 100 ∧
  test.score = 64 ∧
  calculate_score test = test.score →
  test.correct_answers = 88 := by
  sorry

end NUMINAMATH_CALUDE_correct_answers_for_given_score_l3327_332775


namespace NUMINAMATH_CALUDE_parabola_properties_l3327_332755

/-- A parabola is defined by the equation y = -(x-3)^2 --/
def parabola (x y : ℝ) : Prop := y = -(x-3)^2

/-- The axis of symmetry of the parabola --/
def axis_of_symmetry : ℝ := 3

/-- Theorem: The parabola opens downwards and has its axis of symmetry at x=3 --/
theorem parabola_properties :
  (∀ x y : ℝ, parabola x y → y ≤ 0) ∧ 
  (∀ y : ℝ, ∃ x₁ x₂ : ℝ, x₁ < axis_of_symmetry ∧ axis_of_symmetry < x₂ ∧ parabola x₁ y ∧ parabola x₂ y) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l3327_332755


namespace NUMINAMATH_CALUDE_intersection_empty_iff_t_leq_neg_one_l3327_332735

-- Define sets A and B
def A : Set ℝ := {x | |x - 2| ≤ 3}
def B (t : ℝ) : Set ℝ := {x | x < t}

-- State the theorem
theorem intersection_empty_iff_t_leq_neg_one (t : ℝ) :
  A ∩ B t = ∅ ↔ t ≤ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_t_leq_neg_one_l3327_332735


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l3327_332743

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  monday_hours : ℕ
  wednesday_hours : ℕ
  friday_hours : ℕ
  tuesday_hours : ℕ
  thursday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculate Sheila's hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  let total_hours := 
    3 * schedule.monday_hours + 
    2 * schedule.tuesday_hours
  schedule.weekly_earnings / total_hours

/-- Sheila's actual work schedule --/
def sheila_schedule : WorkSchedule := {
  monday_hours := 8
  wednesday_hours := 8
  friday_hours := 8
  tuesday_hours := 6
  thursday_hours := 6
  weekly_earnings := 504
}

/-- Theorem: Sheila's hourly wage is $14 --/
theorem sheila_hourly_wage : 
  hourly_wage sheila_schedule = 14 := by
  sorry

end NUMINAMATH_CALUDE_sheila_hourly_wage_l3327_332743


namespace NUMINAMATH_CALUDE_sixteen_black_squares_with_odd_numbers_l3327_332776

/-- Represents a square on the chessboard -/
structure Square where
  row : Nat
  col : Nat
  number : Nat
  isBlack : Bool

/-- Represents a chessboard -/
def Chessboard := List Square

/-- Creates a standard 8x8 chessboard with alternating black and white squares,
    numbered from 1 to 64 left to right and top to bottom, with 1 on a black square -/
def createStandardChessboard : Chessboard := sorry

/-- Counts the number of black squares containing odd numbers on the chessboard -/
def countBlackSquaresWithOddNumbers (board : Chessboard) : Nat := sorry

/-- Theorem stating that there are exactly 16 black squares containing odd numbers
    on a standard 8x8 chessboard -/
theorem sixteen_black_squares_with_odd_numbers :
  ∀ (board : Chessboard),
    board = createStandardChessboard →
    countBlackSquaresWithOddNumbers board = 16 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_black_squares_with_odd_numbers_l3327_332776


namespace NUMINAMATH_CALUDE_ponderosa_price_calculation_l3327_332746

/-- The price of each ponderosa pine tree -/
def ponderosa_price : ℕ := 225

/-- The total number of trees -/
def total_trees : ℕ := 850

/-- The number of trees bought of one kind -/
def trees_of_one_kind : ℕ := 350

/-- The price of each Douglas fir tree -/
def douglas_price : ℕ := 300

/-- The total amount paid for all trees -/
def total_paid : ℕ := 217500

theorem ponderosa_price_calculation :
  ponderosa_price = 225 ∧
  total_trees = 850 ∧
  trees_of_one_kind = 350 ∧
  douglas_price = 300 ∧
  total_paid = 217500 →
  ∃ (douglas_count ponderosa_count : ℕ),
    douglas_count + ponderosa_count = total_trees ∧
    (douglas_count = trees_of_one_kind ∨ ponderosa_count = trees_of_one_kind) ∧
    douglas_count * douglas_price + ponderosa_count * ponderosa_price = total_paid :=
by sorry

end NUMINAMATH_CALUDE_ponderosa_price_calculation_l3327_332746


namespace NUMINAMATH_CALUDE_min_distance_and_slope_l3327_332751

-- Define the circle F
def circle_F (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the curve W (trajectory)
def curve_W (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l passing through F(1,0)
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the intersection points
def point_A (k : ℝ) : ℝ × ℝ := sorry
def point_D (k : ℝ) : ℝ × ℝ := sorry
def point_B (k : ℝ) : ℝ × ℝ := sorry
def point_C (k : ℝ) : ℝ × ℝ := sorry

-- Define the distances
def dist_AB (k : ℝ) : ℝ := sorry
def dist_CD (k : ℝ) : ℝ := sorry

-- State the theorem
theorem min_distance_and_slope :
  ∃ (k : ℝ), 
    (∀ (k' : ℝ), dist_AB k + 4 * dist_CD k ≤ dist_AB k' + 4 * dist_CD k') ∧
    dist_AB k + 4 * dist_CD k = 4 ∧
    (k = 2 * Real.sqrt 2 ∨ k = -2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_and_slope_l3327_332751


namespace NUMINAMATH_CALUDE_final_mixture_percentage_l3327_332718

/-- Percentage of material A in solution X -/
def x_percentage : ℝ := 0.20

/-- Percentage of material A in solution Y -/
def y_percentage : ℝ := 0.30

/-- Percentage of solution X in the final mixture -/
def x_mixture_percentage : ℝ := 0.80

/-- Calculate the percentage of material A in the final mixture -/
def final_percentage : ℝ := x_percentage * x_mixture_percentage + y_percentage * (1 - x_mixture_percentage)

/-- Theorem stating that the percentage of material A in the final mixture is 22% -/
theorem final_mixture_percentage : final_percentage = 0.22 := by
  sorry

end NUMINAMATH_CALUDE_final_mixture_percentage_l3327_332718


namespace NUMINAMATH_CALUDE_ellipse_symmetric_points_m_bound_l3327_332724

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The line equation -/
def line (x y m : ℝ) : Prop := y = 4 * x + m

/-- Two points are symmetric with respect to a line -/
def symmetric_points (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2 ∧ y₂ - y₁ = -4 * (x₂ - x₁)

/-- The theorem statement -/
theorem ellipse_symmetric_points_m_bound :
  ∀ m : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse x₁ y₁ ∧
    ellipse x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    (∃ x₀ y₀ : ℝ,
      line x₀ y₀ m ∧
      symmetric_points x₁ y₁ x₂ y₂ x₀ y₀)) →
  -2 * Real.sqrt 3 / 13 < m ∧ m < 2 * Real.sqrt 3 / 13 :=
sorry

end NUMINAMATH_CALUDE_ellipse_symmetric_points_m_bound_l3327_332724


namespace NUMINAMATH_CALUDE_parabola_equation_l3327_332792

-- Define a parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define the properties of the parabola
def has_vertex_at_origin (p : Parabola) : Prop :=
  p.equation 0 0

def focus_on_coordinate_axis (p : Parabola) : Prop :=
  ∃ (k : ℝ), (p.equation k 0 ∨ p.equation 0 k) ∧ k ≠ 0

def passes_through_point (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

-- Theorem statement
theorem parabola_equation :
  ∀ (p : Parabola),
    has_vertex_at_origin p →
    focus_on_coordinate_axis p →
    passes_through_point p (-2) 4 →
    (∀ (x y : ℝ), p.equation x y ↔ x^2 = y) ∨
    (∀ (x y : ℝ), p.equation x y ↔ y^2 = -8*x) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3327_332792


namespace NUMINAMATH_CALUDE_circles_intersect_l3327_332728

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x-2)^2 + y^2 = 9

-- Define the distance between the centers
def distance_between_centers : ℝ := 2

-- Define the radii of the circles
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  distance_between_centers > |radius1 - radius2| ∧
  distance_between_centers < radius1 + radius2 :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l3327_332728


namespace NUMINAMATH_CALUDE_pension_calculation_l3327_332742

/-- Represents the pension calculation problem -/
theorem pension_calculation
  (c d r s y : ℝ)
  (h_cd : c ≠ d)
  (h_c : ∃ (t : ℝ), ∀ (x : ℝ), t * Real.sqrt (x + c - y) = t * Real.sqrt (x - y) + r)
  (h_d : ∃ (t : ℝ), ∀ (x : ℝ), t * Real.sqrt (x + d - y) = t * Real.sqrt (x - y) + s) :
  ∃ (t : ℝ), ∀ (x : ℝ), t * Real.sqrt (x - y) = (c * s^2 - d * r^2) / (2 * (d * r - c * s)) :=
sorry

end NUMINAMATH_CALUDE_pension_calculation_l3327_332742


namespace NUMINAMATH_CALUDE_line_through_circle_center_perpendicular_to_given_line_l3327_332773

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0

-- Define the given line equation
def given_line (x y : ℝ) : Prop := x + y = 0

-- Define the equation of the line we want to prove
def target_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem statement
theorem line_through_circle_center_perpendicular_to_given_line :
  ∃ (cx cy : ℝ),
    (∀ x y, circle_equation x y ↔ (x - cx)^2 + (y - cy)^2 = (-cx)^2 + (-cy)^2) ∧
    target_line cx cy ∧
    (∀ x y, target_line x y → given_line x y → (x - cx) * (x - cx) + (y - cy) * (y - cy) = 0) :=
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_perpendicular_to_given_line_l3327_332773


namespace NUMINAMATH_CALUDE_only_one_four_cell_piece_l3327_332766

/-- Represents a piece on the board -/
structure Piece where
  size : Nat
  deriving Repr

/-- Represents the board configuration -/
structure Board where
  size : Nat
  pieces : List Piece
  deriving Repr

/-- Checks if a board configuration is valid -/
def isValidBoard (b : Board) : Prop :=
  b.size = 7 ∧ 
  b.pieces.all (λ p => p.size = 4) ∧
  b.pieces.length ≤ 3 ∧
  (b.pieces.map (λ p => p.size)).sum = b.size * b.size

/-- Theorem: Only one four-cell piece can be used in a valid 7x7 board configuration -/
theorem only_one_four_cell_piece (b : Board) :
  isValidBoard b → (b.pieces.filter (λ p => p.size = 4)).length = 1 := by
  sorry

#check only_one_four_cell_piece

end NUMINAMATH_CALUDE_only_one_four_cell_piece_l3327_332766


namespace NUMINAMATH_CALUDE_largest_consecutive_even_integer_l3327_332731

theorem largest_consecutive_even_integer (a b c d : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- positive integers
  Even a ∧ Even b ∧ Even c ∧ Even d →  -- even integers
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2 →  -- consecutive
  a * b * c * d = 5040 →  -- product is 5040
  d = 20 :=  -- largest is 20
by sorry

end NUMINAMATH_CALUDE_largest_consecutive_even_integer_l3327_332731


namespace NUMINAMATH_CALUDE_equilateral_cone_central_angle_l3327_332749

/-- An equilateral cone is a cone whose cross-section is an equilateral triangle -/
structure EquilateralCone where
  radius : ℝ
  slant_height : ℝ
  slant_height_eq : slant_height = 2 * radius

/-- The central angle of the sector of an equilateral cone is π radians -/
theorem equilateral_cone_central_angle (cone : EquilateralCone) :
  (2 * π * cone.radius) / cone.slant_height = π :=
sorry

end NUMINAMATH_CALUDE_equilateral_cone_central_angle_l3327_332749


namespace NUMINAMATH_CALUDE_bankers_gain_calculation_l3327_332752

/-- Banker's gain calculation -/
theorem bankers_gain_calculation 
  (time : ℝ) 
  (rate : ℝ) 
  (true_discount : ℝ) 
  (ε : ℝ) 
  (h1 : time = 1) 
  (h2 : rate = 12) 
  (h3 : true_discount = 55) 
  (h4 : ε > 0) : 
  ∃ (bankers_gain : ℝ), 
    abs (bankers_gain - 6.60) < ε ∧ 
    bankers_gain = 
      (((true_discount * 100) / (rate * time) + true_discount) * rate * time) / 100 - 
      true_discount :=
sorry

end NUMINAMATH_CALUDE_bankers_gain_calculation_l3327_332752


namespace NUMINAMATH_CALUDE_melanie_dimes_l3327_332762

/-- The number of dimes Melanie has after receiving dimes from her parents -/
def total_dimes (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Theorem: Melanie has 19 dimes after receiving dimes from her parents -/
theorem melanie_dimes : total_dimes 7 8 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_l3327_332762


namespace NUMINAMATH_CALUDE_eighth_root_of_390625000000000_l3327_332726

theorem eighth_root_of_390625000000000 : (390625000000000 : ℝ) ^ (1/8 : ℝ) = 101 := by
  sorry

end NUMINAMATH_CALUDE_eighth_root_of_390625000000000_l3327_332726


namespace NUMINAMATH_CALUDE_sqrt_180_simplified_l3327_332733

theorem sqrt_180_simplified : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_180_simplified_l3327_332733


namespace NUMINAMATH_CALUDE_potato_yield_difference_l3327_332780

/-- Represents the yield difference between varietal and non-varietal potatoes -/
def yield_difference (
  non_varietal_area : ℝ
  ) (varietal_area : ℝ
  ) (yield_difference : ℝ
  ) : Prop :=
  let total_area := non_varietal_area + varietal_area
  let x := non_varietal_area
  let y := varietal_area
  ∃ (non_varietal_yield varietal_yield : ℝ),
    (non_varietal_yield * x + varietal_yield * y) / total_area = 
    non_varietal_yield + yield_difference ∧
    varietal_yield - non_varietal_yield = yield_difference

/-- Theorem stating the yield difference between varietal and non-varietal potatoes -/
theorem potato_yield_difference :
  yield_difference 14 4 90 := by
  sorry

end NUMINAMATH_CALUDE_potato_yield_difference_l3327_332780


namespace NUMINAMATH_CALUDE_complex_angle_proof_l3327_332771

theorem complex_angle_proof (z : ℂ) : z = -1 - Real.sqrt 3 * I → ∃ r θ : ℝ, z = r * Complex.exp (θ * I) ∧ θ = (4 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_angle_proof_l3327_332771


namespace NUMINAMATH_CALUDE_deck_width_is_four_feet_l3327_332789

/-- Given a rectangular pool and a surrounding deck, this theorem proves
    that the deck width is 4 feet under specific conditions. -/
theorem deck_width_is_four_feet 
  (pool_length : ℝ) 
  (pool_width : ℝ) 
  (total_area : ℝ) 
  (h1 : pool_length = 10)
  (h2 : pool_width = 12)
  (h3 : total_area = 360)
  (w : ℝ) -- deck width
  (h4 : (pool_length + 2 * w) * (pool_width + 2 * w) = total_area) :
  w = 4 := by
  sorry

#check deck_width_is_four_feet

end NUMINAMATH_CALUDE_deck_width_is_four_feet_l3327_332789


namespace NUMINAMATH_CALUDE_max_value_expression_l3327_332785

theorem max_value_expression (x y : ℝ) : 
  (Real.sqrt (3 - Real.sqrt 2) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 1) * 
  (3 + 2 * Real.sqrt (7 - Real.sqrt 2) * Real.cos y - Real.cos (2 * y)) ≤ 9.5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3327_332785


namespace NUMINAMATH_CALUDE_tangent_line_to_curve_l3327_332758

/-- A line y = x - 2a is tangent to the curve y = x ln x - x if and only if a = e/2 -/
theorem tangent_line_to_curve (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (x₀ - 2*a = x₀ * Real.log x₀ - x₀) ∧ 
    (1 = Real.log x₀)) ↔ 
  a = Real.exp 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_curve_l3327_332758


namespace NUMINAMATH_CALUDE_race_track_cost_l3327_332721

theorem race_track_cost (initial_amount : ℚ) (num_cars : ℕ) (car_cost : ℚ) (remaining : ℚ) : 
  initial_amount = 17.80 ∧ 
  num_cars = 4 ∧ 
  car_cost = 0.95 ∧ 
  remaining = 8 → 
  initial_amount - (↑num_cars * car_cost) - remaining = 6 := by
sorry

end NUMINAMATH_CALUDE_race_track_cost_l3327_332721


namespace NUMINAMATH_CALUDE_fraction_value_l3327_332704

theorem fraction_value (a b : ℚ) (h1 : a = 7) (h2 : b = 2) : 3 / (a + b) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3327_332704


namespace NUMINAMATH_CALUDE_shortest_path_length_l3327_332754

/-- Represents a frustum of a right circular cone -/
structure ConeFrustum where
  lower_circumference : ℝ
  upper_circumference : ℝ
  inclination_angle : ℝ

/-- The shortest path from a point on the lower base to the upper base and back -/
def shortest_return_path (cf : ConeFrustum) : ℝ := sorry

theorem shortest_path_length (cf : ConeFrustum) 
  (h1 : cf.lower_circumference = 8)
  (h2 : cf.upper_circumference = 6)
  (h3 : cf.inclination_angle = π / 3) :
  shortest_return_path cf = 4 * Real.sqrt 3 / π := by sorry

end NUMINAMATH_CALUDE_shortest_path_length_l3327_332754


namespace NUMINAMATH_CALUDE_cubic_foot_to_cubic_inches_l3327_332714

/-- Conversion factor from feet to inches -/
def inches_per_foot : ℕ := 12

/-- Cubic inches in one cubic foot -/
def cubic_inches_per_cubic_foot : ℕ := inches_per_foot ^ 3

theorem cubic_foot_to_cubic_inches :
  cubic_inches_per_cubic_foot = 1728 :=
sorry

end NUMINAMATH_CALUDE_cubic_foot_to_cubic_inches_l3327_332714


namespace NUMINAMATH_CALUDE_sticker_distribution_solution_l3327_332741

/-- Represents the sticker distribution problem --/
structure StickerDistribution where
  space : ℕ := 120
  cat : ℕ := 80
  dinosaur : ℕ := 150
  superhero : ℕ := 45
  space_given : ℕ := 25
  cat_given : ℕ := 13
  dinosaur_given : ℕ := 33
  superhero_given : ℕ := 29

/-- Calculates the total number of stickers left after initial distribution --/
def remaining_stickers (sd : StickerDistribution) : ℕ :=
  (sd.space - sd.space_given) + (sd.cat - sd.cat_given) + 
  (sd.dinosaur - sd.dinosaur_given) + (sd.superhero - sd.superhero_given)

/-- Theorem stating the solution to the sticker distribution problem --/
theorem sticker_distribution_solution (sd : StickerDistribution) :
  ∃ (X : ℕ), X = 3 ∧ (remaining_stickers sd - X) / 4 = 73 := by
  sorry


end NUMINAMATH_CALUDE_sticker_distribution_solution_l3327_332741


namespace NUMINAMATH_CALUDE_pat_calculation_l3327_332797

theorem pat_calculation (x : ℝ) : (x / 7 + 10 = 20) → (x * 7 - 10 = 480) := by
  sorry

end NUMINAMATH_CALUDE_pat_calculation_l3327_332797


namespace NUMINAMATH_CALUDE_daisy_taller_than_reese_l3327_332708

/-- The heights of three people and their relationships -/
structure Heights where
  daisy : ℝ
  parker : ℝ
  reese : ℝ
  parker_shorter : parker = daisy - 4
  reese_height : reese = 60
  average_height : (daisy + parker + reese) / 3 = 64

/-- Daisy is 8 inches taller than Reese -/
theorem daisy_taller_than_reese (h : Heights) : h.daisy - h.reese = 8 := by
  sorry

end NUMINAMATH_CALUDE_daisy_taller_than_reese_l3327_332708


namespace NUMINAMATH_CALUDE_man_speed_l3327_332781

/-- The speed of a man running opposite to a train, given the train's length, speed, and time to pass the man. -/
theorem man_speed (train_length : Real) (train_speed : Real) (time_to_pass : Real) :
  train_length = 110 ∧ 
  train_speed = 40 ∧ 
  time_to_pass = 9 →
  ∃ (man_speed : Real),
    man_speed > 0 ∧ 
    man_speed < train_speed ∧
    abs (man_speed - train_speed) * time_to_pass / 3600 = train_length / 1000 ∧
    abs (man_speed - 3.992) < 0.001 :=
sorry

end NUMINAMATH_CALUDE_man_speed_l3327_332781


namespace NUMINAMATH_CALUDE_special_prime_sum_of_squares_l3327_332737

theorem special_prime_sum_of_squares (n : ℕ) : 
  (∃ a b : ℤ, n = a^2 + b^2 ∧ Int.gcd a b = 1) →
  (∀ p : ℕ, Nat.Prime p → p ≤ Nat.sqrt n → ∃ k : ℤ, k * p = a * b) →
  n = 5 ∨ n = 13 := by sorry

end NUMINAMATH_CALUDE_special_prime_sum_of_squares_l3327_332737


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l3327_332774

theorem cube_edge_ratio (a b : ℝ) (h : a ^ 3 / b ^ 3 = 8 / 1) : a / b = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l3327_332774


namespace NUMINAMATH_CALUDE_locus_equidistant_point_line_l3327_332791

/-- The locus of points equidistant from a point and a line is a parabola -/
theorem locus_equidistant_point_line (x y : ℝ) : 
  let F : ℝ × ℝ := (0, -3)
  let line_eq : ℝ → ℝ → Prop := λ x y => y + 5 = 0
  let distance_to_point : ℝ × ℝ → ℝ := λ p => Real.sqrt ((p.1 - F.1)^2 + (p.2 - F.2)^2)
  let distance_to_line : ℝ × ℝ → ℝ := λ p => |p.2 + 5|
  distance_to_point (x, y) = distance_to_line (x, y) ↔ y = (1/4) * x^2 - 4 := by
sorry

end NUMINAMATH_CALUDE_locus_equidistant_point_line_l3327_332791


namespace NUMINAMATH_CALUDE_quadrilateral_rod_count_quadrilateral_rod_count_is_17_l3327_332727

theorem quadrilateral_rod_count : ℕ → Prop :=
  fun n =>
    let rods : Finset ℕ := Finset.range 30
    let used_rods : Finset ℕ := {3, 7, 15}
    let valid_rods : Finset ℕ := 
      rods.filter (fun x => 
        x > 5 ∧ x < 25 ∧ x ∉ used_rods)
    n = valid_rods.card

theorem quadrilateral_rod_count_is_17 :
  quadrilateral_rod_count 17 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_rod_count_quadrilateral_rod_count_is_17_l3327_332727


namespace NUMINAMATH_CALUDE_kayak_rental_cost_l3327_332701

theorem kayak_rental_cost 
  (canoe_cost : ℝ) 
  (canoe_kayak_ratio : ℚ) 
  (total_revenue : ℝ) 
  (canoe_kayak_difference : ℕ) :
  canoe_cost = 12 →
  canoe_kayak_ratio = 3 / 2 →
  total_revenue = 504 →
  canoe_kayak_difference = 7 →
  ∃ (kayak_cost : ℝ) (num_canoes num_kayaks : ℕ),
    num_canoes = num_kayaks + canoe_kayak_difference ∧
    (num_canoes : ℚ) / num_kayaks = canoe_kayak_ratio ∧
    total_revenue = canoe_cost * num_canoes + kayak_cost * num_kayaks ∧
    kayak_cost = 18 :=
by sorry

end NUMINAMATH_CALUDE_kayak_rental_cost_l3327_332701


namespace NUMINAMATH_CALUDE_tan_product_values_l3327_332770

theorem tan_product_values (a b : Real) :
  3 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b - 1) = 0 →
  Real.tan (a / 2) * Real.tan (b / 2) = 1 / 2 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_values_l3327_332770


namespace NUMINAMATH_CALUDE_john_typing_duration_l3327_332713

/-- The time John typed before Jack took over -/
def john_typing_time (
  john_total_time : ℝ)
  (jack_rate_ratio : ℝ)
  (jack_completion_time : ℝ) : ℝ :=
  3

/-- Theorem stating that John typed for 3 hours before Jack took over -/
theorem john_typing_duration :
  john_typing_time 5 (2/5) 4.999999999999999 = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_typing_duration_l3327_332713


namespace NUMINAMATH_CALUDE_ashok_pyarelal_capital_ratio_l3327_332712

/-- Given a total loss and Pyarelal's loss, calculate the ratio of Ashok's capital to Pyarelal's capital -/
theorem ashok_pyarelal_capital_ratio 
  (total_loss : ℕ) 
  (pyarelal_loss : ℕ) 
  (h1 : total_loss = 1600) 
  (h2 : pyarelal_loss = 1440) : 
  ∃ (a p : ℕ), a ≠ 0 ∧ p ≠ 0 ∧ a / p = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ashok_pyarelal_capital_ratio_l3327_332712


namespace NUMINAMATH_CALUDE_combinations_equal_twenty_l3327_332725

/-- The number of available colors -/
def num_colors : ℕ := 5

/-- The number of available painting methods -/
def num_methods : ℕ := 4

/-- The total number of combinations of colors and painting methods -/
def total_combinations : ℕ := num_colors * num_methods

/-- Theorem stating that the total number of combinations is 20 -/
theorem combinations_equal_twenty : total_combinations = 20 := by
  sorry

end NUMINAMATH_CALUDE_combinations_equal_twenty_l3327_332725


namespace NUMINAMATH_CALUDE_hyperbola_distance_theorem_l3327_332745

/-- A hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- A point is on a hyperbola if the absolute difference of its distances to the foci is constant -/
def IsOnHyperbola (h : Hyperbola) (p : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), |distance p h.F₁ - distance p h.F₂| = k

theorem hyperbola_distance_theorem (h : Hyperbola) (p : ℝ × ℝ) :
  IsOnHyperbola h p → distance p h.F₁ = 12 →
  distance p h.F₂ = 22 ∨ distance p h.F₂ = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_distance_theorem_l3327_332745


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3327_332764

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

/-- Definition of the line passing through (0, 2) with slope 1 -/
def line (x y : ℝ) : Prop :=
  y = x + 2

/-- Intersection points of the ellipse and the line -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧ line A.1 A.2 ∧ line B.1 B.2

theorem ellipse_and_line_intersection :
  ∃ (A B : ℝ × ℝ),
    intersection_points A B ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (6 * Real.sqrt 3 / 5)^2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3327_332764


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l3327_332761

theorem sphere_hemisphere_volume_ratio (p : ℝ) (hp : p > 0) :
  (4 / 3 * Real.pi * p^3) / (1 / 2 * 4 / 3 * Real.pi * (2*p)^3) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l3327_332761


namespace NUMINAMATH_CALUDE_smallest_positive_integer_x_l3327_332769

theorem smallest_positive_integer_x (x : ℕ+) : (2 * (x : ℝ)^2 < 50) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_x_l3327_332769


namespace NUMINAMATH_CALUDE_sister_packs_l3327_332788

def total_packs : ℕ := 13
def emily_packs : ℕ := 6

theorem sister_packs : total_packs - emily_packs = 7 := by
  sorry

end NUMINAMATH_CALUDE_sister_packs_l3327_332788


namespace NUMINAMATH_CALUDE_problem_statement_l3327_332717

theorem problem_statement (w x y : ℝ) 
  (h1 : 7 / w + 7 / x = 7 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) :
  x = 0.5 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3327_332717


namespace NUMINAMATH_CALUDE_simplification_value_at_3_value_at_negative_3_even_function_l3327_332705

-- Define the original expression
def original_expression (x : ℝ) : ℝ :=
  6 * x^2 + 4 * x - 2 * (x^2 - 1) - 2 * (2 * x + x^2)

-- Define the simplified expression
def simplified_expression (x : ℝ) : ℝ :=
  2 * x^2 + 2

-- Theorem stating that the original expression simplifies to the simplified expression
theorem simplification : 
  ∀ x : ℝ, original_expression x = simplified_expression x :=
sorry

-- Theorem stating that the simplified expression equals 20 when x = 3
theorem value_at_3 : simplified_expression 3 = 20 :=
sorry

-- Theorem stating that the simplified expression equals 20 when x = -3
theorem value_at_negative_3 : simplified_expression (-3) = 20 :=
sorry

-- Theorem stating that the simplified expression is an even function
theorem even_function :
  ∀ x : ℝ, simplified_expression x = simplified_expression (-x) :=
sorry

end NUMINAMATH_CALUDE_simplification_value_at_3_value_at_negative_3_even_function_l3327_332705


namespace NUMINAMATH_CALUDE_white_ball_count_l3327_332759

theorem white_ball_count (total : ℕ) (white blue red : ℕ) : 
  total = 1000 →
  blue = white + 14 →
  red = 3 * (blue - white) →
  total = white + blue + red →
  white = 472 := by
  sorry

end NUMINAMATH_CALUDE_white_ball_count_l3327_332759


namespace NUMINAMATH_CALUDE_range_of_a_l3327_332738

def sequence_a (a : ℝ) (n : ℕ) : ℝ := a * n^2 + n

theorem range_of_a (a : ℝ) :
  (∀ n, sequence_a a n < sequence_a a (n + 1)) ↔ a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3327_332738


namespace NUMINAMATH_CALUDE_binomial_60_3_l3327_332734

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by sorry

end NUMINAMATH_CALUDE_binomial_60_3_l3327_332734


namespace NUMINAMATH_CALUDE_min_ice_cost_l3327_332765

/-- Represents the ice purchasing options --/
inductive IcePackType
  | OnePound
  | FivePound

/-- Calculates the cost of ice for a given pack type and number of packs --/
def calculateCost (packType : IcePackType) (numPacks : ℕ) : ℚ :=
  match packType with
  | IcePackType.OnePound => 
      if numPacks > 20 
      then (6 * numPacks : ℚ) * 0.9
      else 6 * numPacks
  | IcePackType.FivePound => 
      if numPacks > 20 
      then (25 * numPacks : ℚ) * 0.9
      else 25 * numPacks

/-- Calculates the number of packs needed for a given pack type and total ice needed --/
def calculatePacks (packType : IcePackType) (totalIce : ℕ) : ℕ :=
  match packType with
  | IcePackType.OnePound => (totalIce + 9) / 10
  | IcePackType.FivePound => (totalIce + 49) / 50

/-- Theorem: The minimum cost for ice is $100.00 --/
theorem min_ice_cost : 
  let totalPeople : ℕ := 50
  let icePerPerson : ℕ := 4
  let totalIce : ℕ := totalPeople * icePerPerson
  let onePoundCost := calculateCost IcePackType.OnePound (calculatePacks IcePackType.OnePound totalIce)
  let fivePoundCost := calculateCost IcePackType.FivePound (calculatePacks IcePackType.FivePound totalIce)
  min onePoundCost fivePoundCost = 100 := by
  sorry

end NUMINAMATH_CALUDE_min_ice_cost_l3327_332765


namespace NUMINAMATH_CALUDE_inequality_proof_l3327_332707

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a * b + b * c + c * a ≤ 3 * a * b * c) :
  Real.sqrt ((a^2 + b^2) / (a + b)) + Real.sqrt ((b^2 + c^2) / (b + c)) +
  Real.sqrt ((c^2 + a^2) / (c + a)) + 3 ≤
  Real.sqrt 2 * (Real.sqrt (a + b) + Real.sqrt (b + c) + Real.sqrt (c + a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3327_332707


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3327_332756

theorem division_remainder_problem 
  (P D Q R D' Q' R' C : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R')
  (h3 : R < D)
  (h4 : R' < D') :
  P % ((D + C) * D') = D' * C * R' + D * R' + C * R' + R := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3327_332756


namespace NUMINAMATH_CALUDE_xy_value_l3327_332787

theorem xy_value (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = -9 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3327_332787


namespace NUMINAMATH_CALUDE_tree_branches_after_eight_weeks_l3327_332793

def branch_growth (g : ℕ → ℕ) : Prop :=
  g 2 = 1 ∧
  g 3 = 2 ∧
  (∀ n ≥ 3, g (n + 1) = g n + g (n - 1)) ∧
  g 5 = 5

theorem tree_branches_after_eight_weeks (g : ℕ → ℕ) 
  (h : branch_growth g) : g 8 = 21 := by
  sorry

end NUMINAMATH_CALUDE_tree_branches_after_eight_weeks_l3327_332793


namespace NUMINAMATH_CALUDE_largest_y_value_l3327_332702

theorem largest_y_value (y : ℝ) : 
  (y / 7 + 2 / (3 * y) = 3) → y ≤ (63 + Real.sqrt 3801) / 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_y_value_l3327_332702


namespace NUMINAMATH_CALUDE_odd_function_properties_l3327_332730

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_properties (f : ℝ → ℝ) (h : is_odd f) :
  (f 0 = 0) ∧
  (∀ a > 0, (∀ x > 0, f x ≥ a) → (∀ y < 0, f y ≤ -a)) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l3327_332730


namespace NUMINAMATH_CALUDE_M_values_l3327_332732

theorem M_values (a b : ℚ) (hab : a * b ≠ 0) :
  let M := (2 * abs a) / a + (3 * b) / abs b
  M = 1 ∨ M = -1 ∨ M = 5 ∨ M = -5 := by sorry

end NUMINAMATH_CALUDE_M_values_l3327_332732


namespace NUMINAMATH_CALUDE_cos_three_halves_lt_sin_one_tenth_l3327_332740

theorem cos_three_halves_lt_sin_one_tenth :
  Real.cos (3/2) < Real.sin (1/10) := by
  sorry

end NUMINAMATH_CALUDE_cos_three_halves_lt_sin_one_tenth_l3327_332740


namespace NUMINAMATH_CALUDE_probability_cos_geq_half_is_two_thirds_l3327_332716

noncomputable def probability_cos_geq_half : ℝ := by sorry

theorem probability_cos_geq_half_is_two_thirds :
  probability_cos_geq_half = 2/3 := by sorry

end NUMINAMATH_CALUDE_probability_cos_geq_half_is_two_thirds_l3327_332716


namespace NUMINAMATH_CALUDE_fleet_capacity_l3327_332729

theorem fleet_capacity (num_vans : ℕ) (large_capacity : ℕ) 
  (h_num_vans : num_vans = 6)
  (h_large_capacity : large_capacity = 8000)
  (h_small_capacity : ∃ small_capacity : ℕ, small_capacity = large_capacity - (large_capacity * 30 / 100))
  (h_very_large_capacity : ∃ very_large_capacity : ℕ, very_large_capacity = large_capacity + (large_capacity * 50 / 100))
  (h_num_large : ∃ num_large : ℕ, num_large = 2)
  (h_num_small : ∃ num_small : ℕ, num_small = 1)
  (h_num_very_large : ∃ num_very_large : ℕ, num_very_large = num_vans - 2 - 1) :
  ∃ total_capacity : ℕ, total_capacity = 57600 ∧
    total_capacity = (2 * large_capacity) + 
                     (large_capacity - (large_capacity * 30 / 100)) + 
                     (3 * (large_capacity + (large_capacity * 50 / 100))) :=
by
  sorry

end NUMINAMATH_CALUDE_fleet_capacity_l3327_332729


namespace NUMINAMATH_CALUDE_greatest_divisor_of_fourth_power_difference_l3327_332748

/-- The function that reverses the digits of a positive integer -/
noncomputable def reverse_digits (n : ℕ+) : ℕ+ := sorry

/-- Theorem stating that 99 is the greatest integer that always divides n^4 - f(n)^4 -/
theorem greatest_divisor_of_fourth_power_difference (n : ℕ+) : 
  (∃ (k : ℕ), k > 99 ∧ ∀ (m : ℕ+), k ∣ (m^4 - (reverse_digits m)^4)) → False :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_of_fourth_power_difference_l3327_332748


namespace NUMINAMATH_CALUDE_max_tetrahedron_volume_l3327_332763

noncomputable def square_pyramid_volume (base_side : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_side^2 * height

theorem max_tetrahedron_volume
  (base_side : ℝ)
  (m_distance : ℝ)
  (h : base_side = 6)
  (d : m_distance = 10) :
  ∃ (max_vol : ℝ),
    max_vol = 24 ∧
    ∀ (vol : ℝ),
      ∃ (height : ℝ),
        vol = square_pyramid_volume base_side height →
        vol ≤ max_vol :=
by sorry

end NUMINAMATH_CALUDE_max_tetrahedron_volume_l3327_332763


namespace NUMINAMATH_CALUDE_valleyball_hockey_league_players_l3327_332711

/-- The cost of a pair of gloves in dollars -/
def glove_cost : ℕ := 7

/-- The additional cost of a helmet compared to gloves in dollars -/
def helmet_additional_cost : ℕ := 8

/-- The total cost to equip all players in the league in dollars -/
def total_league_cost : ℕ := 3570

/-- The number of sets of equipment each player needs -/
def sets_per_player : ℕ := 2

/-- The number of players in the league -/
def num_players : ℕ := 81

theorem valleyball_hockey_league_players :
  num_players * sets_per_player * (glove_cost + (glove_cost + helmet_additional_cost)) = total_league_cost :=
sorry

end NUMINAMATH_CALUDE_valleyball_hockey_league_players_l3327_332711


namespace NUMINAMATH_CALUDE_max_value_of_y_l3327_332782

-- Define the function y
def y (x a : ℝ) : ℝ := |x - a| + |x + 19| + |x - a - 96|

-- State the theorem
theorem max_value_of_y (a : ℝ) (h1 : 19 < a) (h2 : a < 96) :
  ∃ (max_y : ℝ), max_y = 211 ∧ ∀ x, a ≤ x → x ≤ 96 → y x a ≤ max_y :=
sorry

end NUMINAMATH_CALUDE_max_value_of_y_l3327_332782


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_conditions_l3327_332790

theorem greatest_integer_satisfying_conditions :
  ∃ (n : ℕ), n < 150 ∧
  (∃ (a : ℕ), n = 9 * a - 2) ∧
  (∃ (b : ℕ), n = 11 * b - 4) ∧
  (∃ (c : ℕ), n = 5 * c + 1) ∧
  (∀ (m : ℕ), m < 150 →
    (∃ (a' : ℕ), m = 9 * a' - 2) →
    (∃ (b' : ℕ), m = 11 * b' - 4) →
    (∃ (c' : ℕ), m = 5 * c' + 1) →
    m ≤ n) ∧
  n = 142 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_conditions_l3327_332790


namespace NUMINAMATH_CALUDE_fish_fraction_removed_on_day_five_l3327_332709

/-- Represents the number of fish in Jason's aquarium on a given day -/
def fish (day : ℕ) : ℚ :=
  match day with
  | 0 => 6
  | 1 => 12
  | 2 => 16
  | 3 => 32
  | 4 => 64
  | 5 => 128
  | 6 => 256
  | _ => 0

/-- The fraction of fish removed on day 5 -/
def f : ℚ := 1/4

theorem fish_fraction_removed_on_day_five :
  fish 6 - 4 * f * fish 4 + 15 = 207 :=
sorry

end NUMINAMATH_CALUDE_fish_fraction_removed_on_day_five_l3327_332709


namespace NUMINAMATH_CALUDE_tax_deduction_for_jacob_l3327_332777

/-- Calculates the local tax deduction in cents given an hourly wage in dollars and a tax rate percentage. -/
def localTaxDeduction (hourlyWage : ℚ) (taxRate : ℚ) : ℚ :=
  hourlyWage * 100 * (taxRate / 100)

/-- Theorem stating that for an hourly wage of $25 and a 2% tax rate, the local tax deduction is 50 cents. -/
theorem tax_deduction_for_jacob :
  localTaxDeduction 25 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_tax_deduction_for_jacob_l3327_332777


namespace NUMINAMATH_CALUDE_negation_equivalence_l3327_332768

theorem negation_equivalence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3327_332768


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3327_332739

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x + 2 > 0) ↔ (1 ≤ m ∧ m < 9) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3327_332739


namespace NUMINAMATH_CALUDE_digit_reversal_l3327_332778

theorem digit_reversal (n : ℕ) : 
  let B := n^2 + 1
  (n^2 * (n^2 + 2)^2 = 1 * B^3 + 0 * B^2 + (B - 2) * B + (B - 1)) ∧
  (n^4 * (n^2 + 2)^2 = 1 * B^3 + (B - 2) * B^2 + 0 * B + (B - 1)) := by
sorry

end NUMINAMATH_CALUDE_digit_reversal_l3327_332778


namespace NUMINAMATH_CALUDE_transformed_graph_equivalence_l3327_332795

-- Define the original function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := f (2 * x + 1)

-- Define the horizontal shift transformation
noncomputable def shift (h : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - h)

-- Define the horizontal compression transformation
noncomputable def compress (k : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f (k * x)

-- Theorem statement
theorem transformed_graph_equivalence :
  ∀ x : ℝ, g x = (compress (1/2) (shift (-1/2) f)) x := by sorry

end NUMINAMATH_CALUDE_transformed_graph_equivalence_l3327_332795


namespace NUMINAMATH_CALUDE_inequality_solution_l3327_332710

def solution_set : Set ℝ := Set.union (Set.Icc 2 3) (Set.Ioc 3 48)

theorem inequality_solution (x : ℝ) : 
  x ∈ solution_set ↔ (x ≠ 3 ∧ (x * (x + 2)) / ((x - 3)^2) ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3327_332710


namespace NUMINAMATH_CALUDE_quadratic_equation_with_specific_discriminant_l3327_332760

/-- Represents a quadratic equation of the form ax² + bx + c = 0 -/
structure QuadraticEquation (α : Type*) [Field α] where
  a : α
  b : α
  c : α

/-- Calculates the discriminant of a quadratic equation -/
def discriminant {α : Type*} [Field α] (eq : QuadraticEquation α) : α :=
  eq.b ^ 2 - 4 * eq.a * eq.c

/-- Checks if the roots of a quadratic equation are real and unequal -/
def has_real_unequal_roots {α : Type*} [LinearOrderedField α] (eq : QuadraticEquation α) : Prop :=
  discriminant eq > 0

theorem quadratic_equation_with_specific_discriminant 
  (d : ℝ) (eq : QuadraticEquation ℝ) 
  (h1 : eq.a = 3)
  (h2 : eq.b = -6 * Real.sqrt 3)
  (h3 : eq.c = d)
  (h4 : discriminant eq = 12) :
  d = 8 ∧ has_real_unequal_roots eq :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_specific_discriminant_l3327_332760


namespace NUMINAMATH_CALUDE_lindas_lunchbox_total_cost_l3327_332750

/-- The cost of a sandwich at Linda's Lunchbox -/
def sandwich_cost : ℕ := 4

/-- The cost of a soda at Linda's Lunchbox -/
def soda_cost : ℕ := 2

/-- The cost of a cookie at Linda's Lunchbox -/
def cookie_cost : ℕ := 1

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 7

/-- The number of sodas purchased -/
def num_sodas : ℕ := 6

/-- The number of cookies purchased -/
def num_cookies : ℕ := 4

/-- The total cost of the purchase at Linda's Lunchbox -/
def total_cost : ℕ := num_sandwiches * sandwich_cost + num_sodas * soda_cost + num_cookies * cookie_cost

theorem lindas_lunchbox_total_cost : total_cost = 44 := by
  sorry

end NUMINAMATH_CALUDE_lindas_lunchbox_total_cost_l3327_332750


namespace NUMINAMATH_CALUDE_midpoint_square_sum_l3327_332719

/-- Given that C = (5, 3) is the midpoint of line segment AB, where A = (3, -3) and B = (x, y),
    prove that x^2 + y^2 = 130. -/
theorem midpoint_square_sum (x y : ℝ) : 
  (5 : ℝ) = (3 + x) / 2 ∧ (3 : ℝ) = (-3 + y) / 2 → x^2 + y^2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_square_sum_l3327_332719


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l3327_332720

/-- A polynomial is a perfect square trinomial if it can be expressed as (x + a)^2 for some real number a. -/
def IsPerfectSquareTrinomial (p : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, p x = (x + a)^2

/-- Given that x^2 + (m-2)x + 9 is a perfect square trinomial, prove that m = 8 or m = -4. -/
theorem perfect_square_trinomial_m_values (m : ℝ) :
  IsPerfectSquareTrinomial (fun x ↦ x^2 + (m-2)*x + 9) → m = 8 ∨ m = -4 := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l3327_332720


namespace NUMINAMATH_CALUDE_house_transaction_result_l3327_332706

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : ℝ
  hasHouse : Bool

/-- Represents the state of the house -/
structure HouseState where
  value : ℝ
  owner : String

def initial_mr_a : FinancialState := { cash := 15000, hasHouse := true }
def initial_mr_b : FinancialState := { cash := 20000, hasHouse := false }
def initial_house : HouseState := { value := 15000, owner := "A" }

def house_sale_price : ℝ := 20000
def depreciation_rate : ℝ := 0.15

theorem house_transaction_result :
  let first_transaction_mr_a : FinancialState :=
    { cash := initial_mr_a.cash + house_sale_price, hasHouse := false }
  let first_transaction_mr_b : FinancialState :=
    { cash := initial_mr_b.cash - house_sale_price, hasHouse := true }
  let depreciated_house_value : ℝ := initial_house.value * (1 - depreciation_rate)
  let final_mr_a : FinancialState :=
    { cash := first_transaction_mr_a.cash - depreciated_house_value, hasHouse := true }
  let final_mr_b : FinancialState :=
    { cash := first_transaction_mr_b.cash + depreciated_house_value, hasHouse := false }
  let mr_a_net_gain : ℝ := final_mr_a.cash + depreciated_house_value - (initial_mr_a.cash + initial_house.value)
  let mr_b_net_gain : ℝ := final_mr_b.cash - initial_mr_b.cash
  mr_a_net_gain = 5000 ∧ mr_b_net_gain = -7250 := by
  sorry

end NUMINAMATH_CALUDE_house_transaction_result_l3327_332706


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_proof_smallest_resolvable_debt_achievable_l3327_332799

/-- The smallest positive debt that can be resolved using chairs and tables -/
def smallest_resolvable_debt : ℕ := 60

/-- The value of a chair in dollars -/
def chair_value : ℕ := 240

/-- The value of a table in dollars -/
def table_value : ℕ := 180

theorem smallest_resolvable_debt_proof :
  ∀ (d : ℕ), d > 0 →
  (∃ (c t : ℤ), d = chair_value * c + table_value * t) →
  d ≥ smallest_resolvable_debt := by
sorry

theorem smallest_resolvable_debt_achievable :
  ∃ (c t : ℤ), smallest_resolvable_debt = chair_value * c + table_value * t := by
sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_proof_smallest_resolvable_debt_achievable_l3327_332799


namespace NUMINAMATH_CALUDE_floor_plus_s_eq_15_4_l3327_332779

theorem floor_plus_s_eq_15_4 (s : ℝ) : 
  (⌊s⌋ : ℝ) + s = 15.4 → s = 7.4 := by
sorry

end NUMINAMATH_CALUDE_floor_plus_s_eq_15_4_l3327_332779


namespace NUMINAMATH_CALUDE_function_properties_l3327_332796

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
    (∀ x, f a b x ≤ f a b (-2)) ∧
    (f a b (-2) = 4) ∧
    (a = 3 ∧ b = 0) ∧
    (∀ x ∈ Set.Icc (-3) 1, f a b x ≤ 4) ∧
    (∃ x ∈ Set.Icc (-3) 1, f a b x = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_function_properties_l3327_332796


namespace NUMINAMATH_CALUDE_azure_valley_skirts_l3327_332798

/-- The number of skirts in Purple Valley -/
def purple_skirts : ℕ := 10

/-- The ratio of skirts in Purple Valley to Seafoam Valley -/
def purple_to_seafoam_ratio : ℚ := 1 / 4

/-- The ratio of skirts in Seafoam Valley to Azure Valley -/
def seafoam_to_azure_ratio : ℚ := 2 / 3

/-- The number of skirts in Azure Valley -/
def azure_skirts : ℕ := 60

theorem azure_valley_skirts :
  azure_skirts = (purple_skirts : ℚ) / (purple_to_seafoam_ratio * seafoam_to_azure_ratio) := by
  sorry

end NUMINAMATH_CALUDE_azure_valley_skirts_l3327_332798


namespace NUMINAMATH_CALUDE_similar_triangles_dimensions_l3327_332794

theorem similar_triangles_dimensions (h₁ base₁ h₂ base₂ : ℝ) : 
  h₁ > 0 → base₁ > 0 → h₂ > 0 → base₂ > 0 →
  (h₁ * base₁) / (h₂ * base₂) = 1 / 9 →
  h₁ = 5 → base₁ = 6 →
  h₂ = 15 ∧ base₂ = 18 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_dimensions_l3327_332794


namespace NUMINAMATH_CALUDE_area_equals_half_radius_times_pedal_perimeter_l3327_332703

open Real

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents the pedal triangle of a given triangle -/
def pedalTriangle (T : Triangle) : Triangle := sorry

/-- The area of a triangle -/
def area (T : Triangle) : ℝ := sorry

/-- The perimeter of a triangle -/
def perimeter (T : Triangle) : ℝ := sorry

/-- The circumradius of a triangle -/
def circumradius (T : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is acute-angled -/
def isAcute (T : Triangle) : Prop := sorry

theorem area_equals_half_radius_times_pedal_perimeter (T : Triangle) 
  (h : isAcute T) : 
  area T = (circumradius T / 2) * perimeter (pedalTriangle T) := by
  sorry

end NUMINAMATH_CALUDE_area_equals_half_radius_times_pedal_perimeter_l3327_332703


namespace NUMINAMATH_CALUDE_prob_two_unmarked_correct_l3327_332786

/-- The probability of selecting two unmarked items from a set of 10 items where 3 are marked -/
def prob_two_unmarked (total : Nat) (marked : Nat) (select : Nat) : Rat :=
  if total = 10 ∧ marked = 3 ∧ select = 2 then
    7 / 15
  else
    0

theorem prob_two_unmarked_correct :
  prob_two_unmarked 10 3 2 = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_unmarked_correct_l3327_332786


namespace NUMINAMATH_CALUDE_jana_walking_distance_l3327_332723

/-- Jana's walking pattern and distance traveled -/
theorem jana_walking_distance :
  let usual_pace : ℚ := 1 / 30  -- miles per minute
  let half_pace : ℚ := usual_pace / 2
  let double_pace : ℚ := usual_pace * 2
  let first_15_min_distance : ℚ := half_pace * 15
  let next_5_min_distance : ℚ := double_pace * 5
  first_15_min_distance + next_5_min_distance = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_jana_walking_distance_l3327_332723


namespace NUMINAMATH_CALUDE_quadratic_expression_rewrite_l3327_332757

theorem quadratic_expression_rewrite :
  ∃ (c p q : ℚ),
    (∀ k, 8 * k^2 - 12 * k + 20 = c * (k + p)^2 + q) ∧
    q / p = -142 / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_expression_rewrite_l3327_332757


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3327_332722

theorem quadratic_factorization (x : ℝ) : x^2 - 6*x + 9 = (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3327_332722


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3327_332736

-- Define variables
variable (x y : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 :
  3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2) = 5 * x^2 - 3 * x - 3 := by sorry

-- Theorem for the second expression
theorem simplify_expression_2 :
  3 * x^2 * y - (2 * x * y - 2 * (x * y - 3/2 * x^2 * y) + x^2 * y^2) = -x^2 * y^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3327_332736


namespace NUMINAMATH_CALUDE_pig_count_l3327_332715

theorem pig_count (P1 P2 : ℕ) (h1 : P1 = 64) (h2 : P1 + P2 = 86) : P2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_pig_count_l3327_332715


namespace NUMINAMATH_CALUDE_population_is_all_scores_l3327_332700

/-- Represents a math exam with participants and their scores -/
structure MathExam where
  participants : ℕ
  scores : Finset ℝ

/-- Represents a statistical analysis of a math exam -/
structure StatisticalAnalysis where
  exam : MathExam
  sample_size : ℕ

/-- The definition of population in the context of this statistical analysis -/
def population (analysis : StatisticalAnalysis) : Finset ℝ :=
  analysis.exam.scores

/-- Theorem stating that the population in this statistical analysis
    is the set of all participants' scores -/
theorem population_is_all_scores
  (exam : MathExam)
  (analysis : StatisticalAnalysis)
  (h1 : exam.participants = 40000)
  (h2 : analysis.sample_size = 400)
  (h3 : analysis.exam = exam)
  (h4 : exam.scores.card = exam.participants) :
  population analysis = exam.scores :=
sorry

end NUMINAMATH_CALUDE_population_is_all_scores_l3327_332700


namespace NUMINAMATH_CALUDE_total_fish_l3327_332767

def gold_fish : ℕ := 15
def blue_fish : ℕ := 7

theorem total_fish : gold_fish + blue_fish = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_l3327_332767


namespace NUMINAMATH_CALUDE_inequality_of_powers_l3327_332753

theorem inequality_of_powers (a n k : ℕ) (ha : a > 1) (hnk : 0 < n ∧ n < k) :
  (a^n - 1) / n < (a^k - 1) / k := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_powers_l3327_332753


namespace NUMINAMATH_CALUDE_min_b_over_a_is_one_minus_e_l3327_332747

/-- Given two real functions f and g, if f(x) ≤ g(x) for all x > 0,
    then the minimum value of b/a is 1 - e. -/
theorem min_b_over_a_is_one_minus_e (a b : ℝ)
    (f : ℝ → ℝ) (g : ℝ → ℝ)
    (hf : ∀ x, x > 0 → f x = Real.log x + a)
    (hg : ∀ x, g x = a * x + b + 1)
    (h_le : ∀ x, x > 0 → f x ≤ g x) :
    ∃ m, m = 1 - Real.exp 1 ∧ ∀ k, (b / a ≥ k → k ≥ m) :=
  sorry

end NUMINAMATH_CALUDE_min_b_over_a_is_one_minus_e_l3327_332747


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l3327_332784

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2 * 2^3) = 5760 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l3327_332784
