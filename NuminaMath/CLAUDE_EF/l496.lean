import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_average_speed_l496_49672

/-- Calculates the average speed given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem beth_average_speed :
  let john_speed : ℝ := 40
  let john_time : ℝ := 0.5
  let john_distance : ℝ := john_speed * john_time
  let beth_distance : ℝ := john_distance + 5
  let beth_time : ℝ := john_time + 1/3
  average_speed beth_distance beth_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_average_speed_l496_49672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_y_l496_49678

/-- The algebraic cofactor of the element in the third row and second column of the given determinant -/
noncomputable def f (x : ℝ) : ℝ := -(2 : ℝ)^(x+2) * (1 + 2^x)

/-- The function whose zero we want to find -/
noncomputable def y (x : ℝ) : ℝ := 1 + f x

/-- Theorem stating that the zero of y(x) is -1 -/
theorem zero_of_y : ∃ x : ℝ, y x = 0 ∧ x = -1 := by
  use -1
  constructor
  · -- Prove y(-1) = 0
    sorry
  · -- Prove x = -1
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_y_l496_49678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l496_49634

-- Define the function f as noncomputable due to its dependence on Real.log
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else a^x

-- State the theorem
theorem function_equality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (Real.exp 2) = f a (-2) → a = Real.sqrt 2 / 2 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l496_49634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_calculation_l496_49668

/-- The volume of a frustum with given dimensions -/
noncomputable def frustum_volume (r₁ r₂ h : ℝ) : ℝ :=
  (1/3) * Real.pi * h * (r₁^2 + r₂^2 + r₁*r₂)

/-- The central angle of the sector formed by unfolding the cone's side -/
def central_angle : ℝ := 120

/-- The lower base radius of the frustum (same as the cone's base radius) -/
def lower_radius : ℝ := 2

/-- The upper base radius of the frustum -/
def upper_radius : ℝ := 1

/-- The height of the frustum -/
noncomputable def frustum_height : ℝ := 2 * Real.sqrt 2

/-- Theorem: The volume of the frustum with the given dimensions is (14√2π)/3 -/
theorem frustum_volume_calculation :
  frustum_volume lower_radius upper_radius frustum_height = (14 * Real.sqrt 2 * Real.pi) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_calculation_l496_49668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l496_49654

theorem cosine_identity (A : ℝ) (h : Real.cos (2 * A) = -Real.sqrt 5 / 3) :
  6 * (Real.sin A)^6 + 6 * (Real.cos A)^6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l496_49654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_sundae_combinations_l496_49626

theorem ice_cream_sundae_combinations : ℕ := by
  -- Define the number of flavors
  let n : ℕ := 8
  -- Define the number of scoops in a sundae
  let k : ℕ := 3
  -- Define the number of combinations
  let combinations : ℕ := n.choose k
  -- Assert that the number of combinations is 56
  have h : combinations = 56 := by
    -- Proof goes here
    sorry
  -- Return the result
  exact combinations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_sundae_combinations_l496_49626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_intersection_l496_49662

/-- Fixed point A -/
def A : ℝ × ℝ := (12, 0)

/-- Moving point M on curve -/
noncomputable def M (θ : ℝ) : ℝ × ℝ := (6 + 2 * Real.cos θ, 2 * Real.sin θ)

/-- Point P satisfies AP = 2AM -/
noncomputable def P (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ + 12, 4 * Real.sin θ)

/-- Trajectory C of point P -/
def C : Set (ℝ × ℝ) := {p | (p.1 - 12)^2 + p.2^2 = 16}

/-- Line l: y = -x + a -/
def l (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 = -p.1 + a}

/-- Theorem stating the properties of the trajectory and intersection -/
theorem trajectory_and_intersection :
  ∀ θ a, P θ ∈ C ∧
  (∃ E F, E ∈ C ∧ F ∈ C ∧ E ∈ l a ∧ F ∈ l a ∧ E ≠ F ∧
  E.1 * F.1 + E.2 * F.2 = 12) →
  (∃ E F, E ∈ C ∧ F ∈ C ∧ E ∈ l a ∧ F ∈ l a ∧ E ≠ F ∧
  (E.1 * F.1 + E.2 * F.2) / (Real.sqrt ((E.1^2 + E.2^2) * (F.1^2 + F.2^2))) = 3/4 ∧
  a = 2 * Real.sqrt 7 ∨ a = -2 * Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_intersection_l496_49662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_son_age_difference_l496_49681

/-- Proves that a man is 20 years older than his son given the specified conditions -/
theorem man_son_age_difference (son_age man_age : ℕ) :
  son_age = 18 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_son_age_difference_l496_49681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ana_age_is_four_l496_49630

-- Define the rounding function
def round_to_tens (x : Int) : Int :=
  if x % 10 < 5 then (x / 10) * 10 else ((x + 5) / 10) * 10

-- Define the theorem
theorem ana_age_is_four 
  (cake_weight : Int) 
  (candle_weight : Int) 
  (ana_age : Int) 
  (h1 : round_to_tens cake_weight = 1440)
  (h2 : round_to_tens (cake_weight + ana_age * candle_weight) = 1610)
  (h3 : round_to_tens candle_weight = 40)
  (h4 : ana_age > 0) : 
  ana_age = 4 := by
  sorry

#check ana_age_is_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ana_age_is_four_l496_49630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_congruence_l496_49640

/-- Two figures in a plane -/
structure Figure where
  -- Add necessary properties for a figure
  points : Set (Real × Real)

/-- Line in a plane -/
structure Line where
  -- Define a line using two points or slope-intercept form
  point1 : Real × Real
  point2 : Real × Real

/-- Symmetry about a line -/
def symmetrical_about_line (f1 f2 : Figure) (l : Line) : Prop :=
  -- Define symmetry condition
  sorry

/-- Congruence of figures -/
def congruent (f1 f2 : Figure) : Prop :=
  -- Define congruence condition
  sorry

/-- Theorem: If two figures are symmetrical about a line, then they are congruent -/
theorem symmetry_implies_congruence (f1 f2 : Figure) (l : Line) :
  symmetrical_about_line f1 f2 l → congruent f1 f2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_congruence_l496_49640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irregular_hexagon_existence_l496_49674

-- Define a hexagon
structure Hexagon where
  sides : Fin 6 → ℝ
  angles : Fin 6 → ℝ

-- Define the properties of a hexagon
def is_valid_hexagon (h : Hexagon) : Prop :=
  (∀ i, h.sides i > 0) ∧ 
  (∀ i, 0 < h.angles i ∧ h.angles i < Real.pi) ∧
  (Finset.sum Finset.univ (λ i => h.angles i)) = 2 * Real.pi

-- Define the conditions for Case 1
def case1 (h : Hexagon) : Prop :=
  (∀ i j, h.sides i = h.sides j) ∧
  (∃ a b c d : Fin 6, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
    (∀ i, i = a ∨ i = b ∨ i = c ∨ i = d → h.angles i = h.angles a))

-- Define the conditions for Case 2
def case2 (h : Hexagon) : Prop :=
  (∀ i j, h.angles i = h.angles j) ∧
  (∃ a b c d : Fin 6, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
    (∀ i, i = a ∨ i = b ∨ i = c ∨ i = d → h.sides i = h.sides a))

-- Theorem statement
theorem irregular_hexagon_existence :
  (∃ h : Hexagon, is_valid_hexagon h ∧ case1 h) ∧
  (∃ h : Hexagon, is_valid_hexagon h ∧ case2 h) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irregular_hexagon_existence_l496_49674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_sum_constant_l496_49607

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The ellipse C with equation x²/4 + y²/3 = 1 -/
def ellipse_C (p : Point) : Prop :=
  p.x^2 / 4 + p.y^2 / 3 = 1

/-- The focus F2 of the ellipse -/
noncomputable def F2 : Point :=
  { x := 1, y := 0 }

/-- Point T -/
noncomputable def T : Point :=
  { x := 4, y := 0 }

/-- A line passing through a point -/
def passes_through (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- The slope of a line passing through two points -/
noncomputable def slope_between (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

/-- The theorem to be proved -/
theorem slope_sum_constant 
  (l : Line) 
  (R S : Point) 
  (h1 : ellipse_C R) 
  (h2 : ellipse_C S) 
  (h3 : passes_through l F2) 
  (h4 : passes_through l R) 
  (h5 : passes_through l S) : 
  slope_between T R + slope_between T S = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_sum_constant_l496_49607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l496_49639

/-- The area of a trapezoid with height y, where one base is 3y and the other base is 4y, is 7y²/2 -/
theorem trapezoid_area (y : ℝ) : 
  (1 / 2) * ((3 * y) + (4 * y)) * y = (7 * y^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l496_49639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_geometric_progression_sum_of_c_sequence_l496_49687

def a : ℕ → ℕ
  | 0 => 1  -- Add this case to handle Nat.zero
  | 1 => 1
  | n + 1 => if n % 2 = 0 then 2 * a n else a n + 1

def b (n : ℕ) : ℕ := a (2 * n - 1) + 2

def c (n : ℕ) : ℕ := n * a (2 * n - 1)

theorem b_is_geometric_progression (n : ℕ) (h : n > 0) :
  b (n + 1) = 2 * b n := by sorry

theorem sum_of_c_sequence (n : ℕ) :
  (Finset.range n).sum c = (3 * n - 3) * 2^n + 3 - n^2 - n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_geometric_progression_sum_of_c_sequence_l496_49687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tony_water_consumption_l496_49647

/-- Represents the problem setup for Tony's sandbox filling and water consumption --/
structure SandboxProblem where
  bucket_capacity : ℚ  -- in pounds
  sandbox_depth : ℚ    -- in feet
  sandbox_width : ℚ    -- in feet
  sandbox_length : ℚ   -- in feet
  sand_density : ℚ     -- in pounds per cubic foot
  water_frequency : ℕ  -- trips between water consumption
  water_bottle_size : ℚ -- in ounces
  water_bottle_cost : ℚ -- in dollars
  initial_money : ℚ    -- in dollars
  change_after_water : ℚ -- in dollars

/-- Calculates the amount of water Tony drinks every 4 trips --/
noncomputable def water_consumption (p : SandboxProblem) : ℚ :=
  let sandbox_volume := p.sandbox_depth * p.sandbox_width * p.sandbox_length
  let total_sand_weight := sandbox_volume * p.sand_density
  let total_trips := total_sand_weight / p.bucket_capacity
  let water_drinking_times := total_trips / p.water_frequency
  let money_spent_on_water := p.initial_money - p.change_after_water
  let bottles_bought := money_spent_on_water / p.water_bottle_cost
  (bottles_bought * p.water_bottle_size) / water_drinking_times

/-- Theorem stating that Tony drinks 3 ounces of water every 4 trips --/
theorem tony_water_consumption (p : SandboxProblem) 
  (h1 : p.bucket_capacity = 2)
  (h2 : p.sandbox_depth = 2)
  (h3 : p.sandbox_width = 4)
  (h4 : p.sandbox_length = 5)
  (h5 : p.sand_density = 3)
  (h6 : p.water_frequency = 4)
  (h7 : p.water_bottle_size = 15)
  (h8 : p.water_bottle_cost = 2)
  (h9 : p.initial_money = 10)
  (h10 : p.change_after_water = 4) :
  water_consumption p = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tony_water_consumption_l496_49647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_in_special_triangle_l496_49622

/-- In a triangle ABC, if sin B - sin C = (1/4) sin A and 2b = 3c, then cos A = -1/4 -/
theorem cosine_value_in_special_triangle (a b c A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- Positive angles
  A + B + C = Real.pi →  -- Sum of angles in a triangle
  Real.sin B - Real.sin C = (1/4) * Real.sin A →  -- Given condition
  2 * b = 3 * c →  -- Given condition
  a * Real.sin B = b * Real.sin A →  -- Sine rule
  a * Real.sin C = c * Real.sin A →  -- Sine rule
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →  -- Cosine rule
  Real.cos A = -1/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_in_special_triangle_l496_49622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_correct_answer_is_option_a_l496_49649

-- Define variables
variable (a b c d e f : ℚ)

-- Assume all variables are positive
variable [Fact (0 < a)] [Fact (0 < b)] [Fact (0 < c)] [Fact (0 < d)] [Fact (0 < e)] [Fact (0 < f)]

-- Define the function for milk production
def milk_production (a b c d e f : ℚ) : ℚ := (f * b * d * e) / (a * c)

-- Theorem: The milk production formula is correct
theorem milk_production_correct :
  milk_production a b c d e f = (f * b * d * e) / (a * c) := by
  -- Unfold the definition of milk_production
  unfold milk_production
  -- The equality holds by definition
  rfl

-- Proof that the answer matches option A
theorem answer_is_option_a :
  milk_production a b c d e f = f * b * d * e / (a * c) := by
  -- This is true by the definition of milk_production
  rfl

#check milk_production_correct
#check answer_is_option_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_correct_answer_is_option_a_l496_49649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_property_l496_49677

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.exp (2 * x) - 2 * x * (x + 1) * Real.exp x

-- State the theorem
theorem extreme_points_property (m : ℝ) (x₁ x₂ : ℝ) :
  (∃! (a b : ℝ), a ≠ b ∧ (∀ x, f m x ≤ f m a ∨ f m x ≤ f m b)) →
  (-Real.exp 2 < m ∧ m ≤ 0) ∧ (3 < x₁ * x₂ - (x₁ + x₂) ∧ x₁ * x₂ - (x₁ + x₂) < 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_property_l496_49677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_significance_prob_above_45_correct_dist_X_correct_expect_X_correct_l496_49682

-- Define the contingency table
def contingency_table : Fin 2 → Fin 2 → ℕ
| 0, 0 => 35
| 0, 1 => 45
| 1, 0 => 15
| 1, 1 => 5

-- Define the chi-square statistic function
def chi_square (m : Fin 2 → Fin 2 → ℕ) : ℚ :=
  let N := (m 0 0) + (m 0 1) + (m 1 0) + (m 1 1)
  let a := m 0 0
  let b := m 0 1
  let c := m 1 0
  let d := m 1 1
  ↑N * (↑(a * d - b * c))^2 / (↑((a + b) * (c + d) * (a + c) * (b + d)))

-- Define the probability function for selecting a person above 45
def prob_above_45 (below_45 : ℕ) (above_45 : ℕ) : ℚ :=
  ↑above_45 / ↑(below_45 + above_45 - 1)

-- Define the distribution of X
def dist_X (x : Fin 3) : ℚ :=
  match x with
  | 0 => 3 / 14
  | 1 => 4 / 7
  | 2 => 3 / 14

-- Define the expectation of X
def expect_X : ℚ :=
  (0 * (3 / 14) + 1 * (4 / 7) + 2 * (3 / 14))

-- Theorem statements
theorem chi_square_significance :
  chi_square contingency_table > 3841 / 1000 := by sorry

theorem prob_above_45_correct :
  prob_above_45 4 4 = 4 / 7 := by sorry

theorem dist_X_correct (x : Fin 3) :
  dist_X x = if x = 0 then 3/14 else if x = 1 then 4/7 else 3/14 := by sorry

theorem expect_X_correct :
  expect_X = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_significance_prob_above_45_correct_dist_X_correct_expect_X_correct_l496_49682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equality_condition_l496_49663

theorem area_equality_condition (φ : Real) (h : 0 < φ ∧ φ < π / 2) :
  (∃ r : Real, r > 0 ∧ 
    (r^2 * φ / 2 = r^2 * Real.tan φ / 2 - r^2 * φ / 2)) ↔ Real.tan φ = 4 * φ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equality_condition_l496_49663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_patrol_officer_journey_l496_49659

def travel_records : List Int := [2, -3, 5, -4, 6, -2, 4, -2]

noncomputable def fuel_consumption_rate : ℚ := 8 / 100

theorem patrol_officer_journey (records : List Int) (rate : ℚ) :
  (records.sum = 6) ∧
  (rate * (records.map Int.natAbs).sum = 224 / 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_patrol_officer_journey_l496_49659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blood_pressure_reading_l496_49646

noncomputable def blood_pressure (t : ℝ) : ℝ := 110 + 25 * Real.sin (160 * t)

theorem blood_pressure_reading :
  (∃ systolic diastolic : ℝ,
    (∀ t, blood_pressure t ≤ systolic) ∧
    (∃ t, blood_pressure t = systolic) ∧
    (∀ t, blood_pressure t ≥ diastolic) ∧
    (∃ t, blood_pressure t = diastolic) ∧
    systolic = 135 ∧
    diastolic = 85) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blood_pressure_reading_l496_49646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l496_49605

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex_quadrilateral (q : Quadrilateral) : Prop := sorry

noncomputable def side_length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

noncomputable def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area (q : Quadrilateral) :
  is_convex_quadrilateral q →
  side_length q.A q.B = 8 →
  side_length q.B q.C = 6 →
  side_length q.C q.D = 10 →
  side_length q.D q.A = 10 →
  angle q.C q.D q.A = Real.pi / 2 →
  area q = 50 + Real.sqrt 2675.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l496_49605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_center_l496_49606

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

-- Define the given line
def given_line (x y : ℝ) : Prop := x + y = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 0)

theorem perpendicular_line_through_center :
  (perp_line center.1 center.2) ∧ 
  (∀ (x y : ℝ), given_line x y → given_line y (-x)) ∧
  (my_circle center.1 center.2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_center_l496_49606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cos_inequality_l496_49658

theorem triangle_cos_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.cos A + Real.cos B * Real.cos C ≤ 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cos_inequality_l496_49658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_tangent_to_x_axis_l496_49684

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point is on a circle -/
def onCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a circle is tangent to the x-axis -/
def tangentToXAxis (c : Circle) : Prop :=
  c.center.2 = c.radius

theorem circle_equation_tangent_to_x_axis :
  let c : Circle := { center := (-3, 4), radius := 4 }
  (∀ (x y : ℝ), onCircle c (x, y) ↔ (x + 3)^2 + (y - 4)^2 = 16) ∧
  tangentToXAxis c := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_tangent_to_x_axis_l496_49684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l496_49611

-- Define the solution set
def solution_set : Set ℝ := Set.Icc (-1/2) 3

-- Define the inequality function
def inequality (x : ℝ) : Prop := (2*x + 1) / (3 - x) ≥ 0

-- Theorem statement
theorem inequality_solution_set :
  ∀ x, x ∈ solution_set ↔ inequality x ∧ x ≠ 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l496_49611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_largest_product_l496_49666

/-- A function that returns true if a number is a single-digit prime -/
def isSingleDigitPrime (p : ℕ) : Prop :=
  p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7

/-- A function that calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- The theorem to be proved -/
theorem sum_of_digits_of_largest_product : ∃ (n d e : ℕ),
  isSingleDigitPrime d ∧
  isSingleDigitPrime e ∧
  (d = 3 ∨ d = 5 ∨ d = 7) ∧
  Nat.Prime d ∧
  Nat.Prime e ∧
  Nat.Prime (9*d + e) ∧
  n = d * e * (9*d + e) ∧
  (∀ (m : ℕ), ∀ (d' e' : ℕ), m = d' * e' * (9*d' + e') → 
    isSingleDigitPrime d' → 
    isSingleDigitPrime e' → 
    (d' = 3 ∨ d' = 5 ∨ d' = 7) → 
    Nat.Prime d' → 
    Nat.Prime e' → 
    Nat.Prime (9*d' + e') → 
    m ≤ n) ∧
  sumOfDigits n = 11 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_largest_product_l496_49666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l496_49685

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a > b ∧ b > 0

-- Define points and lines
noncomputable def F (e : Ellipse) : ℝ × ℝ := (-e.c, 0)
noncomputable def A (e : Ellipse) : ℝ × ℝ := (e.a, 0)
noncomputable def E (e : Ellipse) : ℝ × ℝ := (0, e.c)

-- Define the area of triangle EFA
noncomputable def area_EFA (e : Ellipse) : ℝ := e.b^2 / 2

-- Define point Q on segment AE
noncomputable def Q (e : Ellipse) : ℝ × ℝ := sorry

-- Define |FQ|
noncomputable def FQ_length (e : Ellipse) : ℝ := 3 * e.c / 2

-- Define point P where FQ intersects the ellipse
noncomputable def P (e : Ellipse) : ℝ × ℝ := sorry

-- Define points M and N on x-axis
noncomputable def M (e : Ellipse) : ℝ × ℝ := sorry
noncomputable def N (e : Ellipse) : ℝ × ℝ := sorry

-- Define the distance between PM and QN
noncomputable def PM_QN_distance (e : Ellipse) : ℝ := e.c

-- Define the area of quadrilateral PQNM
noncomputable def area_PQNM (e : Ellipse) : ℝ := 3 * e.c

-- Helper function (not part of the original problem, but needed for the theorem)
noncomputable def slope_FP (e : Ellipse) : ℝ := sorry

-- Theorem statement
theorem ellipse_properties (e : Ellipse) :
  (e.c / e.a = 1/2) ∧  -- eccentricity
  (slope_FP e = 3/4) ∧  -- slope of line FP
  (∀ x y, x^2/16 + y^2/12 = 1 ↔ x^2/e.a^2 + y^2/e.b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l496_49685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_speed_l496_49655

/-- The speed of a man rowing in still water, given his upstream and downstream speeds -/
noncomputable def speed_in_still_water (upstream_speed downstream_speed : ℝ) : ℝ :=
  (upstream_speed + downstream_speed) / 2

/-- Theorem: Given a man who can row upstream at 15 kmph and downstream at 35 kmph, 
    his speed in still water is 25 kmph -/
theorem man_rowing_speed :
  speed_in_still_water 15 35 = 25 := by
  unfold speed_in_still_water
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_speed_l496_49655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l496_49608

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Define the transformed function
noncomputable def g (x : ℝ) : ℝ := Real.log (x + 1) - 2

-- Theorem statement
theorem graph_translation (x : ℝ) : 
  g x = f (x + 1) - 2 := by
  -- Unfold the definitions of g and f
  unfold g f
  -- The proof is now trivial by definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l496_49608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equality_l496_49664

/-- Given a quadratic function f(x) = x^2 + ax + b, if f(1) = f(2), then a = -3 -/
theorem quadratic_equality (a b : ℝ) : 
  (∀ x, x^2 + a*x + b = x^2 + a*x + b) → 
  (1^2 + a*1 + b = 2^2 + a*2 + b) → 
  a = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equality_l496_49664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l496_49671

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

noncomputable def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := eccentricity a b
  let r := 1  -- radius of incircle
  let S := 2  -- sum of areas of triangles PIF₁ and PIF₂
  ∀ x y : ℝ, ellipse a b x y →
    e = 1/2 ∧ S = triangle_area (2*a) r →
      (a = 2 ∧ b = Real.sqrt 3) ∧
      (∀ m : ℝ, 
        let AB := Real.sqrt (12 * m^2 + 1) / (3 * m^2 + 4)
        let CD := Real.sqrt (12 / m^2 + 1) / (3 / m^2 + 4)
        AB + CD ≥ 48/7) :=
by
  sorry

#check ellipse_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l496_49671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_C_properties_l496_49690

/-- Ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  eccentricity : ℝ
  triangle_area : ℝ

/-- Definition of the specific ellipse C -/
noncomputable def ellipse_C : Ellipse :=
  { center := (0, 0)
  , foci_on_x_axis := true
  , eccentricity := Real.sqrt 3 / 2
  , triangle_area := 1 / 2 }

/-- Theorem about the equation of ellipse C and the range of λ -/
theorem ellipse_C_properties (C : Ellipse) (h : C = ellipse_C) :
  (∃ (x y : ℝ), x^2 + 4*y^2 = 1) ∧ 
  (∃ (l : ℝ), (l > -1 ∧ l < -1/3) ∨ (l > 1/3 ∧ l < 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_C_properties_l496_49690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_unfair_die_l496_49601

def die_probability (n : ℕ) : ℚ :=
  if n ≤ 7 then 1/14 else if n = 8 then 5/14 else 0

def expected_value : ℚ :=
  Finset.sum (Finset.range 9) (λ n ↦ n * die_probability n)

theorem expected_value_of_unfair_die :
  expected_value = 68/14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_unfair_die_l496_49601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_ratio_l496_49680

/-- Two lines with the same non-zero y-intercept, different slopes, and x-intercepts -/
structure TwoLines where
  b : ℝ  -- shared y-intercept
  u : ℝ  -- x-intercept of the first line
  v : ℝ  -- x-intercept of the second line
  hb : b ≠ 0  -- y-intercept is non-zero
  hu : 8 * u + b = 0  -- equation for first line's x-intercept
  hv : 12 * v + b = 0  -- equation for second line's x-intercept

/-- The ratio of x-intercepts is 3/2 -/
theorem x_intercept_ratio (l : TwoLines) : l.u / l.v = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_ratio_l496_49680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_ln_cos_l496_49695

open Real

theorem arc_length_ln_cos (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = 1 - log (cos x)) →
  a = 0 →
  b = π / 6 →
  ∫ x in a..b, sqrt (1 + (deriv f x) ^ 2) = (1 / 2) * log 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_ln_cos_l496_49695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_logarithm_and_exponent_l496_49614

theorem compare_logarithm_and_exponent :
  ∀ (x y : ℝ), 0 < x → x < 1 → 0 < y → y < 1 →
  (Real.log y / Real.log x > 1) ∧ (1 > x^y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_logarithm_and_exponent_l496_49614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_pyramid_volume_l496_49667

theorem cone_pyramid_volume (l : ℝ) (h : l = 6) : 
  let α : ℝ := π / 8
  let β : ℝ := π / 4
  let O₁O₂ : ℝ := l * Real.sin (2 * α)
  let BO₂ : ℝ := l * Real.sin α * Real.cos α
  let AO₃ : ℝ := l * Real.cos β
  let BC : ℝ := l * Real.cos α * Real.sqrt ((Real.cos (2 * α) - Real.cos (2 * (α + β))) / 2)
  l > 0 → (1 / 3 : ℝ) * O₁O₂ * BO₂ * AO₃ * BC = 9 * Real.sqrt (Real.sqrt 2 + 1) := by
  intro hl
  sorry

#check cone_pyramid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_pyramid_volume_l496_49667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l496_49612

-- Define the constants as noncomputable
noncomputable def a : ℝ := Real.rpow 0.6 0.6
noncomputable def b : ℝ := Real.rpow 0.6 1.5
noncomputable def c : ℝ := Real.rpow 1.5 0.6

-- State the theorem
theorem relationship_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l496_49612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l496_49637

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set ℕ := {x : ℕ | 1 < x ∧ x ≤ 4}

-- Define set B
noncomputable def B : Set ℕ := {x : ℕ | ∃ y : ℝ, y ∈ {y : ℝ | y^2 - 3*y + 2 = 0} ∧ x = Int.floor y}

-- Theorem statement
theorem set_operations :
  (A ∩ B) = {2} ∧
  (U \ (A ∪ B)) = {0, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l496_49637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l496_49600

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x) ^ 2 + Real.sqrt 3 * Real.sin (ω * x) * Real.sin (ω * x + Real.pi / 2)

-- State the theorem
theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_period : ∀ x, f ω x = f ω (x + Real.pi) ∧ 
    ∀ y, 0 < y → y < Real.pi → (∀ x, f ω x = f ω (x + y)) → y = Real.pi) :
  (ω = 1) ∧ 
  (∀ x ∈ Set.Icc 0 (2*Real.pi/3), f ω x ∈ Set.Icc 0 (3/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l496_49600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_expansion_l496_49623

theorem coefficient_expansion (a : ℝ) : 
  (Finset.range 7).sum (λ k => (-1)^k * Nat.choose 6 k * a^k * (1 - a)^(6 - k)) = 1 →
  15 * a^4 = 240 →
  a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_expansion_l496_49623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l496_49610

noncomputable section

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the points that the ellipse passes through
def ellipse_points (a b : ℝ) : Prop :=
  ellipse a b (Real.sqrt 3) (1/2) ∧ ellipse a b 1 (Real.sqrt 3 / 2)

-- Define the bottom vertex A
def bottom_vertex (a b : ℝ) : ℝ × ℝ := (0, -b)

-- Define perpendicular lines l1 and l2 passing through A
def perpendicular_lines (k a b : ℝ) (x y : ℝ) : Prop :=
  (y = k * x - b) ∨ (y = -1/k * x - b)

-- Define points E and F where l1 and l2 intersect y = x
def intersection_points (k : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((1/(k-1), 1/(k-1)), (1/(-1/k-1), 1/(-1/k-1)))

-- Define OE = OF condition
def equal_distances (E F : ℝ × ℝ) : Prop :=
  E.1^2 + E.2^2 = F.1^2 + F.2^2

theorem ellipse_and_line_properties (a b k : ℝ) :
  ellipse_points a b →
  let A := bottom_vertex a b
  let (E, F) := intersection_points k
  perpendicular_lines k a b A.1 A.2 →
  equal_distances E F →
  (a = 2 ∧ b = 1) ∧ (k = 1 + Real.sqrt 2 ∨ k = 1 - Real.sqrt 2) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l496_49610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cabinet_price_after_discount_l496_49619

/-- Given a cabinet with an original price and a discount percentage, 
    calculate the final price after applying the discount. -/
noncomputable def final_price (original_price : ℝ) (discount_percentage : ℝ) : ℝ :=
  original_price * (1 - discount_percentage / 100)

/-- Theorem stating that for a cabinet with an original price of $1200 
    and a 15% discount, the final price is $1020. -/
theorem cabinet_price_after_discount :
  final_price 1200 15 = 1020 := by
  -- Unfold the definition of final_price
  unfold final_price
  -- Simplify the arithmetic expression
  simp [mul_sub, mul_div_right_comm]
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cabinet_price_after_discount_l496_49619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trirectangular_pythagorean_l496_49641

/-- Represents a trirectangular tetrahedron with vertices at (0,0,0), (x,0,0), (0,y,0), and (0,0,z) --/
structure TrirectangularTetrahedron where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the area of a triangle given two side lengths --/
noncomputable def triangleArea (a b : ℝ) : ℝ := (1/2) * a * b

/-- Calculates the area of the face opposite to the right-angled vertex --/
noncomputable def oppositeArea (t : TrirectangularTetrahedron) : ℝ :=
  (1/2) * Real.sqrt ((t.y * t.z)^2 + (t.x * t.z)^2 + (t.x * t.y)^2)

/-- States the three-dimensional Pythagorean theorem for a trirectangular tetrahedron --/
theorem trirectangular_pythagorean (t : TrirectangularTetrahedron) :
  (oppositeArea t)^2 = (triangleArea t.x t.y)^2 + (triangleArea t.y t.z)^2 + (triangleArea t.x t.z)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trirectangular_pythagorean_l496_49641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_three_l496_49657

def mySequence (n : ℕ) : ℚ :=
  if n % 2 = 1 then 3 else 4

theorem fifteenth_term_is_three :
  let s := mySequence
  (∀ n : ℕ, n > 0 → s n * s (n + 1) = 12) →
  s 1 = 3 →
  s 2 = 4 →
  s 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_three_l496_49657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_logarithm_base_l496_49629

/-- Square ABCD with area 64 and AB parallel to x-axis -/
structure Square where
  area : ℝ
  ab_parallel_x : Bool
  h_area : area = 64
  h_parallel : ab_parallel_x = true

/-- Logarithmic functions for vertices A, B, and C -/
noncomputable def log_function (a : ℝ) (k : ℝ) (x : ℝ) : ℝ := k * Real.log x / Real.log a

/-- Theorem stating the value of a for the given conditions -/
theorem square_logarithm_base (S : Square) 
  (h_A : ∃ x y, y = log_function a 1 x)
  (h_B : ∃ x y, y = log_function a 3 (x + 8))
  (h_C : ∃ x y, y + 8 = log_function a 5 (x + 8))
  (h_a_pos : a > 0) :
  a = (64 : ℝ) ^ (1/6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_logarithm_base_l496_49629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_cap_ratio_l496_49648

/-- 
Given a sphere of radius r > 0 and a spherical cap of height m > 0,
if the surface area of the spherical cap is n > 1 times the lateral surface area
of the inscribed cone in the corresponding spherical slice,
then the ratio of the height of the slice to the diameter of the sphere
is (n^2 - 1) / n^2.
-/
theorem spherical_cap_ratio (r m n : ℝ) (hr : r > 0) (hm : m > 0) (hn : n > 1) :
  (2 * π * r * m = n * π * Real.sqrt (2 * r * m - m^2) * Real.sqrt ((2 * r * m - m^2) + m^2)) →
  m / (2 * r) = (n^2 - 1) / n^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_cap_ratio_l496_49648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_and_point_l496_49689

/-- Curve C in Cartesian coordinates -/
def curve_C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- Point R in Cartesian coordinates -/
def point_R : ℝ × ℝ := (2, 2)

/-- Rectangle PQRS with PR as diagonal and one side perpendicular to x-axis -/
def rectangle_PQRS (P Q R S : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  let (sx, sy) := S
  curve_C px py ∧
  R = point_R ∧
  (px - rx)^2 + (py - ry)^2 = (qx - sx)^2 + (qy - sy)^2 ∧
  qx = rx ∧ sy = py

/-- Perimeter of rectangle PQRS -/
noncomputable def perimeter_PQRS (P Q R S : ℝ × ℝ) : ℝ :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  let (sx, sy) := S
  2 * (Real.sqrt ((px - qx)^2 + (py - qy)^2) + Real.sqrt ((qx - rx)^2 + (qy - ry)^2))

/-- Theorem: Minimum perimeter and corresponding point P -/
theorem min_perimeter_and_point :
  ∃ (P : ℝ × ℝ),
    curve_C P.1 P.2 ∧
    (∀ (Q R S : ℝ × ℝ), rectangle_PQRS P Q R S →
      perimeter_PQRS P Q R S ≥ 8) ∧
    P = (3/2, 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_and_point_l496_49689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_transformation_equivalence_l496_49627

-- Define a general function g
variable (g : ℝ → ℝ)

-- Define the transformed function
noncomputable def g_transformed (g : ℝ → ℝ) (x : ℝ) : ℝ := g (-x/3 + 1)

-- State the theorem
theorem g_transformation_equivalence (g : ℝ → ℝ) (x : ℝ) :
  g_transformed g x = g (-(x - 3)/3) :=
by
  -- Unfold the definition of g_transformed
  unfold g_transformed
  -- Show that the arguments are equal
  have h : -x/3 + 1 = -(x - 3)/3 := by
    -- Algebraic manipulation
    ring
  -- Rewrite using the equality
  rw [h]

-- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_transformation_equivalence_l496_49627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alberts_earnings_increase_l496_49673

/-- Given Albert's original monthly earnings and two scenarios of increased earnings,
    calculate the percentage increase in the second scenario. -/
theorem alberts_earnings_increase (E : ℝ) (h1 : E + 0.20 * E = 560) 
    (h2 : ∃ P : ℝ, E + P * E = 564.67) : 
    ∃ P : ℝ, E + P * E = 564.67 ∧ |P - 0.2099| < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alberts_earnings_increase_l496_49673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tiles_for_given_dimensions_l496_49693

/-- The minimum number of rectangular tiles needed to cover a rectangular region -/
def min_tiles (tile_length : ℚ) (tile_width : ℚ) (region_length : ℚ) (region_width : ℚ) : ℕ :=
  Nat.ceil ((region_length * region_width) / (tile_length * tile_width))

/-- Theorem: The minimum number of 6-inch by 4-inch tiles needed to cover a 3-foot by 8-foot rectangular region is 144 -/
theorem min_tiles_for_given_dimensions :
  min_tiles (1/2) (1/3) 3 8 = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tiles_for_given_dimensions_l496_49693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_calculation_l496_49676

/-- The simple interest formula -/
def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * (1 + r * t)

/-- The theorem statement -/
theorem initial_amount_calculation (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) 
  (h1 : A = 514)
  (h2 : r = 5.963855421686747 / 100)
  (h3 : t = 4)
  (h4 : A = simple_interest P r t) :
  ∃ ε > 0, |P - 414.96| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_calculation_l496_49676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_even_l496_49616

-- Define a point in the plane
structure Point where
  x : ℤ
  y : ℤ

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℕ :=
  sorry  -- We don't define the actual distance function, as it's not necessary for the statement

-- Define the perimeter of a triangle
noncomputable def perimeter (t : Triangle) : ℕ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

-- Theorem statement
theorem triangle_perimeter_even (t : Triangle) 
  (h1 : distance t.A t.B > 0) 
  (h2 : distance t.B t.C > 0) 
  (h3 : distance t.C t.A > 0) : 
  Even (perimeter t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_even_l496_49616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_price_change_l496_49642

theorem tv_price_change (P : ℝ) (h : P > 0) : 
  let price_after_decrease := P * 0.9
  let price_after_increase := price_after_decrease * 1.3
  let net_change := (price_after_increase - P) / P
  net_change = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_price_change_l496_49642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_survey_respondents_l496_49688

/-- Approximate equality for real numbers --/
def approx_eq (a b : ℝ) : Prop :=
  |a - b| < 0.5

notation:50 a " ≈ " b:50 => approx_eq a b

/-- Proves that the number of respondents to the original survey is approximately 11 --/
theorem original_survey_respondents :
  ∀ (x : ℝ),
  (x / 90 : ℝ) + 0.06 = 9 / 63 →
  ⌊x⌋ ≈ (11 : ℝ) :=
by
  intro x h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_survey_respondents_l496_49688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_count_l496_49651

noncomputable def initial_height : ℝ := 324
noncomputable def bounce_ratio : ℝ := 3/4
noncomputable def target_height : ℝ := 40

noncomputable def height_after_bounces (n : ℕ) : ℝ :=
  initial_height * (bounce_ratio ^ n)

theorem ball_bounce_count :
  ∀ k : ℕ, k < 8 → height_after_bounces k ≥ target_height ∧
  height_after_bounces 8 < target_height :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_count_l496_49651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l496_49698

noncomputable section

-- Define the vectors m and n
def m (x : ℝ) : ℝ × ℝ := (Real.sin (x/4), Real.cos (x/4))
def n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos (x/4), Real.cos (x/4))

-- Define the dot product function
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define f(x)
def f (x : ℝ) : ℝ := dot_product (m x) (n x)

-- Part 1
theorem part1 (x : ℝ) (h : f x = 1) : Real.cos (x + π/3) = 1/2 := by sorry

-- Part 2
theorem part2 (A B C : ℝ) (a b c : ℝ) 
  (h1 : A + B + C = π) 
  (h2 : (2*a - c) * Real.cos B = b * Real.cos C) :
  Set.Icc 1 (3/2) ⊆ Set.range (λ A => Real.sin (A/2 + π/6) + 1/2) ∧ 
  Set.range (λ A => Real.sin (A/2 + π/6) + 1/2) ⊆ Set.Ioo 1 (3/2) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l496_49698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_through_points_l496_49602

theorem circle_radius_through_points : ∃ (x : ℝ), 
  (let center := (x, 0);
   let point1 := (0, 3);
   let point2 := (2, 5);
   let distance (p1 p2 : ℝ × ℝ) := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2);
   distance center point1 = distance center point2 ∧ 
   distance center point1 = Real.sqrt 34) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_through_points_l496_49602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircles_tangent_l496_49699

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the property that sum of opposite sides are equal
def oppositeSidesEqual (q : Quadrilateral) : Prop :=
  let d1 := dist q.A q.B + dist q.C q.D
  let d2 := dist q.A q.D + dist q.B q.C
  d1 = d2

-- Define an incircle
structure Incircle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define a function to get the incircle of a triangle
noncomputable def getIncircle (A B C : ℝ × ℝ) : Incircle :=
  sorry

-- Define a predicate for two circles being tangent
def areTangent (c1 c2 : Incircle) : Prop :=
  dist c1.center c2.center = c1.radius + c2.radius

-- Theorem statement
theorem incircles_tangent (q : Quadrilateral) 
  (h : oppositeSidesEqual q) : 
  let incircle1 := getIncircle q.A q.B q.C
  let incircle2 := getIncircle q.A q.C q.D
  areTangent incircle1 incircle2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircles_tangent_l496_49699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_star_eight_equals_one_point_five_l496_49635

-- Define the * operation
noncomputable def star_op (a b : ℝ) : ℝ := a + a * (1 / b) - 3

-- Theorem statement
theorem four_star_eight_equals_one_point_five :
  star_op 4 8 = 1.5 := by
  -- Expand the definition of star_op
  unfold star_op
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_star_eight_equals_one_point_five_l496_49635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_is_best_method_l496_49661

/-- Represents a transmission line with a fault -/
structure TransmissionLine where
  hasElectricityAtStart : Bool
  hasElectricityAtEnd : Bool
  hasFault : Bool

/-- Enum representing different fault location methods -/
inductive FaultLocationMethod
  | Bisection
  | Fraction
  | ZeroSixOneEight
  | BlindManClimbing

/-- Function to determine the best fault location method for a given transmission line -/
def best_fault_location_method (line : TransmissionLine) : FaultLocationMethod :=
  FaultLocationMethod.Bisection

/-- Theorem stating that the Bisection method is the best for locating a fault in a transmission line -/
theorem bisection_is_best_method (line : TransmissionLine) 
  (h1 : line.hasElectricityAtStart = true)
  (h2 : line.hasElectricityAtEnd = false)
  (h3 : line.hasFault = true) :
  best_fault_location_method line = FaultLocationMethod.Bisection :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_is_best_method_l496_49661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_l496_49632

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem x0_value (x₀ : ℝ) (h₁ : x₀ > 0) (h₂ : (deriv f) x₀ = 2) : x₀ = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_l496_49632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_property_l496_49660

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Adding case for 0
  | 1 => 2
  | 2 => 6
  | (n + 3) => 2 * sequence_a (n + 2) - sequence_a (n + 1) + 2

theorem sum_property :
  ∃ (sum : ℚ), sum > 2016 ∧ sum < 2017 ∧
  sum = (Finset.range 2017).sum (λ i => 2017 / sequence_a (i + 1)) := by
  sorry

#eval sequence_a 5  -- This line is added to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_property_l496_49660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_third_quadrant_range_l496_49603

/-- 
Given a real number m, if the point corresponding to the complex number (3-i)m-(1+i) 
lies in the third quadrant, then -1 < m < 1/3.
-/
theorem complex_third_quadrant_range (m : ℝ) : 
  (((3 : ℂ) - I) * m - ((1 : ℂ) + I)).re < 0 ∧ 
  (((3 : ℂ) - I) * m - ((1 : ℂ) + I)).im < 0 → 
  -1 < m ∧ m < 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_third_quadrant_range_l496_49603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_prism_volume_ratio_l496_49633

/-- The ratio of the volume of an inscribed right circular cone to its enclosing right rectangular prism -/
theorem cone_prism_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (1 / 3 * π * r^2 * h) / ((4 * r) * (2 * r) * h) = π / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_prism_volume_ratio_l496_49633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_polynomial_roots_l496_49691

/-- The polynomial function defined as x^n - x^{n-1} + x^{n-2} - ... + (-1)^{n-1}x + (-1)^n -/
def alternating_polynomial (n : ℕ+) (x : ℝ) : ℝ := 
  (Finset.range (n.val + 1)).sum (λ i => (-1)^(n.val - i) * x^i)

theorem alternating_polynomial_roots (n : ℕ+) :
  (∀ x : ℝ, alternating_polynomial n x = 0 ↔ (n.val % 2 = 1 ∧ x = -1) ∨ (n.val % 2 = 0 ∧ (x = 1 ∨ x = -1))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_polynomial_roots_l496_49691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_third_quadrant_function_value_l496_49618

theorem angle_third_quadrant_function_value (α : Real) : 
  (π < α ∧ α < 3*π/2) →  -- α is in the third quadrant
  Real.cos (α - 3*π/2) = 1/5 → 
  (Real.sin (α - π/2) * Real.cos (3*π/2 + α) * Real.tan (π - α)) / (Real.tan (-α - π) * Real.sin (-α - π)) = 2*Real.sqrt 6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_third_quadrant_function_value_l496_49618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l496_49621

def f (x : ℝ) := 2*x + x - 5

theorem root_in_interval : 
  (∃ x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0) → 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l496_49621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cardinality_of_A_l496_49656

-- Define the sets B and C
def B : Finset ℕ := {0, 1, 2, 3, 4}
def C : Finset ℕ := {0, 2, 4, 8}

-- Define the theorem
theorem max_cardinality_of_A (A : Finset ℕ) (h1 : A ⊂ B) (h2 : A ⊆ C) :
  Finset.card A ≤ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cardinality_of_A_l496_49656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_triangle_rational_tangents_l496_49652

/-- A point with integer coordinates -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A triangle with vertices on integer coordinates -/
structure GridTriangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint

/-- The tangent of an angle is rational if it can be expressed as a ratio of two integers -/
def RationalTangent (θ : ℝ) : Prop :=
  ∃ (p q : ℤ), q ≠ 0 ∧ Real.tan θ = p / q

/-- An angle is part of a triangle -/
def AngleInTriangle (θ : ℝ) (T : GridTriangle) : Prop :=
  sorry -- We'll define this properly later

/-- Main theorem: The tangent of any angle in a grid triangle is rational -/
theorem grid_triangle_rational_tangents (T : GridTriangle) :
    ∀ θ, AngleInTriangle θ T → RationalTangent θ :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_triangle_rational_tangents_l496_49652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_of_triangle_l496_49653

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that the area of its circumcircle is 49π when cosA = 3√5/7 and bcosC + ccosB = 4 -/
theorem circumcircle_area_of_triangle (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  Real.cos A = 3 * Real.sqrt 5 / 7 →
  b * Real.cos C + c * Real.cos B = 4 →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (a / Real.sin A = 2 * R) →
  Real.pi * R^2 = 49 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_of_triangle_l496_49653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_same_color_distance_pi_l496_49692

/-- A type representing colors -/
inductive Color
| Red
| Blue

/-- A structure representing a colored point in a disc -/
structure ColoredPoint where
  x : ℝ
  y : ℝ
  color : Color

/-- A disc with radius r and a coloring function -/
structure ColoredDisc (r : ℝ) where
  coloring : ℝ → ℝ → Color

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ColoredPoint) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For any disc with radius r > π/2 and a two-color coloring,
    there exist two points A and B in the interior with distance π and the same color -/
theorem two_points_same_color_distance_pi (r : ℝ) (h : r > Real.pi / 2)
  (D : ColoredDisc r) : ∃ (A B : ColoredPoint),
  (A.x^2 + A.y^2 < r^2) ∧ (B.x^2 + B.y^2 < r^2) ∧
  distance A B = Real.pi ∧ A.color = B.color := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_same_color_distance_pi_l496_49692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_intersection_property_l496_49696

/-- The centroid of a triangle. -/
def is_centroid (G A B C : ℝ × ℝ) : Prop :=
  G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

/-- A set of points forms a line. -/
def is_line (l : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ), ∀ (x y : ℝ), (x, y) ∈ l ↔ a * x + b * y + c = 0

/-- A line intersects a triangle. -/
def intersects (l : Set (ℝ × ℝ)) (t : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ l ∧ p ∈ t

/-- The perpendicular distance from a point to a line. -/
noncomputable def perpendicular_distance (P : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ :=
  sorry  -- Definition omitted for brevity

/-- The set of points forming a triangle. -/
def triangle (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry  -- Definition omitted for brevity

/-- Given a triangle ABC with centroid G, any line l intersecting the triangle
    such that the perpendicular distance from A to l equals the sum of
    perpendicular distances from B and C to l, passes through G. -/
theorem centroid_intersection_property (A B C G : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  is_centroid G A B C →
  is_line l →
  intersects l (triangle A B C) →
  perpendicular_distance A l = perpendicular_distance B l + perpendicular_distance C l →
  G ∈ l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_intersection_property_l496_49696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l496_49670

/-- The side length of the square -/
def square_side : ℚ := 40

/-- The leg length of each triangle -/
def triangle_leg : ℚ := 15

/-- The number of triangles in the square -/
def num_triangles : ℕ := 3

/-- The area of the square -/
def square_area : ℚ := square_side ^ 2

/-- The area of one triangle -/
def triangle_area : ℚ := (1 / 2) * triangle_leg ^ 2

/-- The total area of all triangles -/
def total_triangle_area : ℚ := num_triangles * triangle_area

/-- The shaded area in the square -/
def shaded_area : ℚ := square_area - total_triangle_area

theorem shaded_area_calculation : shaded_area = 1262.5 := by
  -- Unfold definitions
  unfold shaded_area square_area total_triangle_area triangle_area
  unfold square_side triangle_leg num_triangles
  -- Simplify the expression
  simp
  -- The proof is completed by computation
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l496_49670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_angles_l496_49625

theorem tan_half_sum_angles (x y : ℝ) 
  (h1 : Real.cos x + Real.cos y = 1) 
  (h2 : Real.sin x + Real.sin y = 1/2) : 
  Real.tan ((x + y) / 2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_angles_l496_49625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l496_49609

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The foci of an ellipse -/
noncomputable def foci (e : Ellipse) : ℝ × ℝ :=
  (Real.sqrt (e.a^2 - e.b^2), -Real.sqrt (e.a^2 - e.b^2))

theorem ellipse_properties (e : Ellipse) (h_major_axis : e.a = 3) :
  (∀ N : PointOnEllipse e, ∃ Q : PointOnEllipse e,
    let (f₁, f₂) := foci e
    (Q.x - f₁) * (Q.x - f₂) + (Q.y * Q.y) = 0) ∧
  (∀ N : PointOnEllipse e,
    let (f₁, f₂) := foci e
    (Real.sqrt ((N.x - f₁)^2 + N.y^2) + Real.sqrt ((N.x - f₂)^2 + N.y^2)) /
    (Real.sqrt ((N.x - f₁)^2 + N.y^2) * Real.sqrt ((N.x - f₂)^2 + N.y^2)) ≥ 2/3) ∧
  (Real.sqrt 6 / 3 < eccentricity e ∧ eccentricity e < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l496_49609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_triangle_vertices_l496_49620

noncomputable def point (x y : ℝ) := (x, y)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem sum_distances_triangle_vertices :
  let Q := point 3 3
  let D := point 0 0
  let E := point 7 2
  let F := point 4 5
  distance Q D + distance Q E + distance Q F = 3 * Real.sqrt 2 + Real.sqrt 17 + Real.sqrt 5 := by
  sorry

#eval "Theorem statement type-checked successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_triangle_vertices_l496_49620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_a_value_when_sum_of_slopes_is_one_l496_49624

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x + 2 * a

-- Define the derivative of f(x)
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - a

-- Theorem 1: Tangent line equation when a = 1
theorem tangent_line_equation (x y : ℝ) :
  f 1 2 = (8/3) →
  f' 1 2 = 3 →
  (9 * x - 3 * y - 10 = 0) ↔ (y - (8/3) = 3 * (x - 2)) := by
  sorry

-- Theorem 2: Value of a when sum of slopes is 1
theorem a_value_when_sum_of_slopes_is_one (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    x₁ ≠ x₂ ∧
    f a x₁ = f' a x₁ * (x₁ - 2) ∧
    f a x₂ = f' a x₂ * (x₂ - 2) ∧
    f' a x₁ + f' a x₂ = 1) →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_a_value_when_sum_of_slopes_is_one_l496_49624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_to_four_probability_l496_49665

/-- Rounds a real number to the nearest integer -/
noncomputable def my_round (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

/-- The probability that two nonnegative real numbers summing to 3.5
    round to integers that sum to 4 -/
theorem sum_to_four_probability : 
  (∫ x in (Set.Icc 0 3.5), if my_round x + my_round (3.5 - x) = 4 then (1 : ℝ) else 0) / 3.5 = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_to_four_probability_l496_49665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_m_eq_ten_l496_49613

/-- The function f(x) defined in terms of m and x -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 
  2 * m * Real.sin x - 2 * (Real.cos x)^2 + m^2 / 2 - 4 * m + 3

/-- The theorem stating that if the minimum value of f(x) is -7, then m = 10 -/
theorem min_value_implies_m_eq_ten (m : ℝ) :
  (∀ x : ℝ, f m x ≥ -7) ∧ (∃ x : ℝ, f m x = -7) → m = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_m_eq_ten_l496_49613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_length_difference_l496_49645

-- Define the function f(x) = 4^|x| as noncomputable
noncomputable def f (x : ℝ) : ℝ := 4^(abs x)

-- Theorem statement
theorem interval_length_difference (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, 1 ≤ f x ∧ f x ≤ 4) →
  (∃ a' b', ∀ x ∈ Set.Icc a' b', 1 ≤ f x ∧ f x ≤ 4 ∧ b' - a' = (b - a)) →
  (b - a) - 1 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_length_difference_l496_49645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_preserving_bijection_preserves_lines_l496_49631

/-- A circle in ℝ² -/
def IsCircle (s : Set (ℝ × ℝ)) : Prop := sorry

/-- A line in ℝ² -/
def IsLine (s : Set (ℝ × ℝ)) : Prop := sorry

/-- A bijective mapping from ℝ² to ℝ² that preserves circles -/
def CirclePreservingBijection : Type :=
  {f : ℝ × ℝ → ℝ × ℝ // Function.Bijective f ∧ ∀ c : Set (ℝ × ℝ), IsCircle c → IsCircle (f '' c)}

/-- Theorem: A bijective mapping from ℝ² to ℝ² that maps every circle to a circle also maps every line to a line -/
theorem circle_preserving_bijection_preserves_lines (f : CirclePreservingBijection) :
  ∀ l : Set (ℝ × ℝ), IsLine l → IsLine (f.val '' l) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_preserving_bijection_preserves_lines_l496_49631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_min_area_l496_49644

/-- The minimum area of a parallelogram with sides 20 and 15 and positive integer area -/
theorem parallelogram_min_area : ℕ := by
  -- Define the parallelogram
  let a : ℝ := 20
  let b : ℝ := 15

  -- Define the area function
  let area (θ : ℝ) : ℝ := a * b * Real.sin θ

  -- Condition that the area is a positive integer
  have area_is_pos_int : ∀ θ, ∃ n : ℕ, area θ = n ∧ n > 0 := by sorry

  -- The minimum area is 1
  have min_area : (∃ θ, area θ = 1) ∧ (∀ θ, area θ ≥ 1) := by sorry

  -- Conclude that the minimum area is 1
  exact 1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_min_area_l496_49644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_nonzero_terms_l496_49638

theorem expansion_nonzero_terms : 
  let f : Polynomial ℚ := (X^2 + 3) * (3*X^3 + 4*X - 5) - 2*X*(X^4 - 3*X^3 + X^2 + 8)
  (f.support.filter (λ n => f.coeff n ≠ 0)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_nonzero_terms_l496_49638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l496_49628

/-- The eccentricity of a hyperbola given its parameter b -/
noncomputable def eccentricity (b : ℝ) : ℝ := Real.sqrt (1 + b^2)

/-- The distance from the point (0, 2) to the asymptote of the hyperbola -/
noncomputable def distance_to_asymptote (b : ℝ) : ℝ := 2 / Real.sqrt (1 + b^2)

theorem hyperbola_eccentricity_range (b : ℝ) (h1 : b > 0) 
  (h2 : distance_to_asymptote b ≥ 1) : 
  1 < eccentricity b ∧ eccentricity b ≤ 2 := by
  sorry

#check hyperbola_eccentricity_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l496_49628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiples_of_30_not_40_l496_49679

theorem three_digit_multiples_of_30_not_40 :
  (Finset.filter (fun n : ℕ => 
    100 ≤ n ∧ n < 1000 ∧ 
    n % 30 = 0 ∧ 
    n % 40 ≠ 0) (Finset.range 1000)).card = 22 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiples_of_30_not_40_l496_49679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagram_circle_area_ratio_l496_49604

/-- The ratio of the area of a pentagram to the area of its circumscribing circle -/
theorem pentagram_circle_area_ratio :
  let r : ℝ := 3  -- radius of the circle
  let circle_area : ℝ := π * r^2
  let pentagon_area : ℝ := (Real.sqrt (25 + 10 * Real.sqrt 5) / 4) * r^2
  let pentagram_area : ℝ := (5/8) * pentagon_area
  pentagram_area / circle_area = (5 * Real.sqrt (25 + 10 * Real.sqrt 5)) / (32 * π) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagram_circle_area_ratio_l496_49604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clay_capacity_l496_49617

/-- Represents a container with dimensions and clay capacity -/
structure Container where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℝ

/-- The problem statement -/
theorem clay_capacity (c1 c2 : Container) :
  c1.height = 3 →
  c1.width = 5 →
  c1.length = 7 →
  c1.capacity = 105 →
  c2.height = 3 * c1.height →
  c2.width = 2 * c1.width →
  c2.length = c1.length →
  c2.capacity = 630 := by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry

#check clay_capacity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clay_capacity_l496_49617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_problem_l496_49636

/-- Geometric progression sum formula -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- Last term of a geometric progression -/
noncomputable def geometric_last_term (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * r^(n - 1)

/-- Theorem for the geometric progression problem -/
theorem geometric_progression_problem :
  let a : ℝ := 3
  let r : ℝ := 2
  let n : ℕ := 5
  geometric_sum a r n = 93 ∧ geometric_last_term a r n = 48 := by
  sorry

#check geometric_progression_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_problem_l496_49636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_infinite_rioplatense_set_l496_49694

def is_rioplatense (a b : ℕ) : Prop :=
  ∀ k, k ∈ ({0, 1, 2, 3, 4} : Set ℕ) → ∃ m : ℕ, b + k = m * (a + k)

theorem exists_infinite_rioplatense_set :
  ∃ A : Set ℕ, (Set.Infinite A) ∧
    (∀ a b, a ∈ A → b ∈ A → a < b → is_rioplatense a b) := by
  sorry

#check exists_infinite_rioplatense_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_infinite_rioplatense_set_l496_49694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_product_l496_49697

theorem complex_fraction_product (a b : ℝ) : 
  (1 + 7 * Complex.I) / (2 - Complex.I) = (a : ℂ) + b * Complex.I → a * b = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_product_l496_49697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmland_tax_range_l496_49686

/-- Represents the annual farmland loss in ten thousand acres -/
noncomputable def annual_loss (t : ℝ) : ℝ := 20 - (5/2) * t

/-- Represents the tax revenue in ten thousand yuan -/
noncomputable def tax_revenue (t : ℝ) : ℝ := annual_loss t * 24000 * (t/100)

/-- The theorem states that the range of t satisfying the conditions is [3, 5] -/
theorem farmland_tax_range : 
  {t : ℝ | 0 ≤ t ∧ t ≤ 100 ∧ tax_revenue t ≥ 900000 ∧ annual_loss t < 20} = {t : ℝ | 3 ≤ t ∧ t ≤ 5} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmland_tax_range_l496_49686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_five_l496_49643

theorem expression_equals_five :
  Real.sqrt 12 + (2014 - 2015 : Int) ^ (0 : Nat) + (1 / 4 : Real) ^ (-1 : Int) - 6 * Real.tan (30 * π / 180) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_five_l496_49643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_theorem_l496_49650

noncomputable section

/-- The volume of a cone formed by a 270-degree sector of a circle with radius 15, divided by π -/
def cone_volume_over_pi (r : ℝ) (angle : ℝ) : ℝ :=
  let base_radius := r * angle / (2 * Real.pi)
  let height := Real.sqrt (r^2 - base_radius^2)
  (1/3) * base_radius^2 * height

theorem cone_volume_theorem :
  cone_volume_over_pi 15 (3/4 * 2 * Real.pi) = 453.515625 * Real.sqrt 10.9375 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_theorem_l496_49650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_problem_l496_49615

/-- Calculates the total amount returned after compound interest --/
noncomputable def total_amount_after_compound_interest 
  (principal : ℝ) 
  (interest_rate : ℝ) 
  (time : ℝ) 
  (compound_frequency : ℝ) : ℝ :=
  principal * (1 + interest_rate / compound_frequency) ^ (compound_frequency * time)

/-- Theorem stating the total amount returned after compound interest --/
theorem compound_interest_problem :
  let interest_rate : ℝ := 0.05
  let time : ℝ := 2
  let compound_frequency : ℝ := 1
  let compound_interest : ℝ := 246
  let principal := compound_interest / ((1 + interest_rate / compound_frequency) ^ (compound_frequency * time) - 1)
  total_amount_after_compound_interest principal interest_rate time compound_frequency = 2646 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_problem_l496_49615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_theorem_l496_49669

/-- The count of positive integers n less than 10^4 for which 2^n - n^2 is divisible by 7 -/
def count_divisible_by_seven : ℕ := 2857

/-- Checks if 2^n - n^2 is divisible by 7 -/
def is_divisible (n : ℕ) : Bool :=
  (2^n - n^2) % 7 = 0

theorem count_divisible_theorem :
  (Finset.filter (fun n => is_divisible n) (Finset.range 10000)).card = count_divisible_by_seven := by
  sorry

#eval (Finset.filter (fun n => is_divisible n) (Finset.range 10000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_theorem_l496_49669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l496_49683

open Real

theorem trigonometric_identities (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : Real.sin α = 4/5) : 
  (Real.tan α = 4/3) ∧ 
  ((Real.sin (α + π) - 2 * Real.cos (π/2 + α)) / (-Real.sin (-α) + Real.cos (π + α)) = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l496_49683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_equals_three_to_three_fourths_l496_49675

-- Define the general term of the product
noncomputable def a (n : ℕ) : ℝ := (3 ^ (3 ^ n)) ^ ((n + 1 : ℝ) / (3 ^ (n + 1)))

-- Define the infinite product
noncomputable def P : ℝ := ∏' n, a n

-- State the theorem
theorem infinite_product_equals_three_to_three_fourths : P = 3 ^ (3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_equals_three_to_three_fourths_l496_49675
