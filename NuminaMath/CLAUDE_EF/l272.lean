import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_and_coefficient_sum_l272_27206

/-- Represents the grid and circle configuration --/
structure GridConfig where
  grid_size : ℕ
  square_size : ℝ
  small_circle_radius : ℝ
  small_circle_count : ℕ
  large_circle_radius : ℝ
  large_circle_count : ℕ

/-- Calculates the shaded area for a given grid configuration --/
noncomputable def shaded_area (config : GridConfig) : ℝ :=
  let total_area := (config.grid_size * config.grid_size * config.square_size * config.square_size : ℝ)
  let small_circle_area := (config.small_circle_count : ℝ) * Real.pi * config.small_circle_radius * config.small_circle_radius
  let large_circle_area := (config.large_circle_count : ℝ) * Real.pi * config.large_circle_radius * config.large_circle_radius
  total_area - small_circle_area - large_circle_area

/-- Theorem stating the shaded area and coefficient sum for the given configuration --/
theorem shaded_area_and_coefficient_sum :
  let config : GridConfig := {
    grid_size := 6,
    square_size := 2,
    small_circle_radius := 1.5,
    small_circle_count := 4,
    large_circle_radius := 2,
    large_circle_count := 2
  }
  let area := shaded_area config
  ∃ (A B : ℝ), area = A - B * Real.pi ∧ A = 144 ∧ B = 17 ∧ A + B = 161 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_and_coefficient_sum_l272_27206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_iff_m_in_range_l272_27258

/-- A circle in the xy-plane with equation x^2 + y^2 + 4mx - 2y + 5m = 0 -/
def circle_equation (m : ℝ) (x y : ℝ) : ℝ :=
  x^2 + y^2 + 4*m*x - 2*y + 5*m

/-- A point is outside the circle if the left side of the equation is positive -/
def outside_circle (m : ℝ) (x y : ℝ) : Prop :=
  circle_equation m x y > 0

/-- The theorem stating the range of m for which the point (-1, -1) is outside the circle -/
theorem point_outside_circle_iff_m_in_range :
  ∀ m : ℝ, outside_circle m (-1) (-1) ↔ (m > -4 ∧ m < 1/4) ∨ m > 1 :=
by
  intro m
  simp [outside_circle, circle_equation]
  sorry

/-- The range of m for which the point (-1, -1) is outside the circle -/
def m_range : Set ℝ :=
  {m : ℝ | (m > -4 ∧ m < 1/4) ∨ m > 1}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_iff_m_in_range_l272_27258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l272_27248

noncomputable section

/-- Definition of an ellipse -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- Definition of eccentricity -/
def Eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

/-- Definition of a line passing through a point -/
def Line (m : ℝ) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.2 - p.2 = m * (q.1 - p.1)}

/-- Definition of triangle area -/
def TriangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

theorem ellipse_and_line_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : (1^2 / a^2) + ((3/2)^2 / b^2) = 1)
  (h4 : Eccentricity a b = 1/2)
  (h5 : ∃ f1 f2 : ℝ × ℝ, f1.1 ∈ Set.Icc (-2) 0 ∧ f2.1 ∈ Set.Icc (-2) 0 ∧
    f1 ∈ Ellipse a b ∧ f2 ∈ Ellipse a b)
  (h6 : ∃ A B : ℝ × ℝ, A ∈ Ellipse a b ∧ B ∈ Ellipse a b ∧
    ∃ m : ℝ, A ∈ Line m f1 ∧ B ∈ Line m f1)
  (h7 : ∃ F2 : ℝ × ℝ, F2 ∈ Ellipse a b ∧ F2 ≠ f1 ∧
    TriangleArea F2 A B = 12 * Real.sqrt 2 / 7) :
  (Ellipse a b = Ellipse 2 (Real.sqrt 3)) ∧
  (∃ m : ℝ, Line m f1 = {p : ℝ × ℝ | p.1 - p.2 + 1 = 0} ∨
            Line m f1 = {p : ℝ × ℝ | p.1 + p.2 + 1 = 0}) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l272_27248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l272_27268

theorem trigonometric_identities (α : Real) :
  (Real.cos α = -4/5 ∧ π/2 < α ∧ α < π) →
    (Real.sin α = 3/5 ∧ Real.tan α = -3/4) ∧
  (Real.tan α = -2) →
    (Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l272_27268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l272_27257

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) + Real.log (2 - x) / Real.log 10

-- State the theorem
theorem domain_of_f :
  {x : ℝ | x - 1 ≥ 0 ∧ 2 - x > 0} = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l272_27257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l272_27239

noncomputable def f (x : ℝ) := Real.sin x - 2 * Real.sqrt 3 * (Real.sin (x / 2))^2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (x y : ℝ), π/6 ≤ x ∧ x < y ∧ y ≤ 7*π/6 → f y < f x) ∧
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2*π/3 → f x ≥ -Real.sqrt 3) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2*π/3 ∧ f x = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l272_27239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l272_27299

/-- Represents the time (in hours) it takes to fill a cistern when both filling and emptying taps are opened simultaneously. -/
noncomputable def simultaneous_fill_time (fill_time empty_time : ℝ) : ℝ :=
  (fill_time * empty_time) / (empty_time - fill_time)

/-- Theorem stating that for a cistern that can be filled in 4 hours and emptied in 5 hours,
    it will take 20 hours to fill when both taps are opened simultaneously. -/
theorem cistern_fill_time :
  simultaneous_fill_time 4 5 = 20 := by
  -- Unfold the definition of simultaneous_fill_time
  unfold simultaneous_fill_time
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l272_27299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l272_27243

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_a : a > 1
  h_b : b > 1

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Check if a point is on an ellipse -/
def on_ellipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Check if a point is on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.c

/-- The perimeter of a quadrilateral given its four vertices -/
noncomputable def quadrilateral_perimeter (p q r s : Point) : ℝ :=
  distance p q + distance q r + distance r s + distance s p

/-- The area of a quadrilateral given its four vertices -/
noncomputable def quadrilateral_area (p q r s : Point) : ℝ :=
  (1/2) * abs ((p.x*q.y + q.x*r.y + r.x*s.y + s.x*p.y) -
               (q.x*p.y + r.x*q.y + s.x*r.y + p.x*s.y))

/-- The distance from a point to a line -/
noncomputable def point_to_line_distance (p : Point) (l : Line) : ℝ :=
  abs (l.m * p.x - p.y + l.c) / Real.sqrt (l.m^2 + 1)

theorem ellipse_properties (e : Ellipse) (l : Line) (A B F₁ F₂ : Point)
    (h_l_slope : l.m = 1)
    (h_l_origin : on_line origin l)
    (h_A_ellipse : on_ellipse A e)
    (h_B_ellipse : on_ellipse B e)
    (h_A_line : on_line A l)
    (h_B_line : on_line B l)
    (h_perimeter : quadrilateral_perimeter F₁ A F₂ B = 8)
    (h_area : quadrilateral_area F₁ A F₂ B = 4 * Real.sqrt 21 / 7) :
  (e.a = 2 ∧ e.b = Real.sqrt 3) ∧
  (∀ l' : Line, ∀ C D : Point,
    on_ellipse C e → on_ellipse D e → on_line C l' → on_line D l' →
    point_to_line_distance origin l' = 2 * Real.sqrt 21 / 7) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l272_27243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_project_solution_l272_27225

/-- Represents the daily construction output and cost for road building teams -/
structure ConstructionTeam where
  dailyOutput : ℝ
  dailyCost : ℝ

/-- Represents the road construction project -/
structure RoadProject where
  totalLength : ℝ
  teamA : ConstructionTeam
  teamB : ConstructionTeam
  daysAlone : ℝ
  daysExtra : ℝ
  daysTogether : ℝ
  maxTotalCost : ℝ

theorem road_project_solution (project : RoadProject) 
  (h1 : project.totalLength = 600)
  (h2 : project.daysAlone = 10)
  (h3 : project.daysExtra = 15)
  (h4 : project.daysTogether = 12)
  (h5 : project.teamA.dailyCost = 0.6)
  (h6 : project.daysAlone * project.teamA.dailyOutput + 
        project.daysExtra * project.teamB.dailyOutput = project.totalLength)
  (h7 : project.daysTogether * (project.teamA.dailyOutput + project.teamB.dailyOutput) = project.totalLength)
  (h8 : project.maxTotalCost = 12)
  (h9 : project.daysAlone * (project.teamA.dailyCost + project.teamB.dailyCost) + 
        (project.totalLength - project.daysAlone * (project.teamA.dailyOutput + project.teamB.dailyOutput)) / 
        project.teamB.dailyOutput * project.teamB.dailyCost ≤ project.maxTotalCost) :
  project.teamA.dailyOutput = 30 ∧ 
  project.teamB.dailyOutput = 20 ∧ 
  project.teamB.dailyCost ≤ 0.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_project_solution_l272_27225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_T_l272_27283

noncomputable def T (n : ℕ+) : ℝ :=
  Real.sqrt (n^4 + 4*n^3 + 2*n^2 - 4*n + 529)

theorem unique_integer_T (n : ℕ+) :
  (∃ (m : ℕ), T n = m) ↔ n = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_T_l272_27283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_sum_less_than_10_and_odd_l272_27295

def roll_8_sided_die : Finset ℕ := Finset.range 8

def sum_of_two_dice (roll1 roll2 : ℕ) : ℕ := roll1 + roll2 + 2

def is_less_than_10_and_odd (n : ℕ) : Bool := n < 10 ∧ n % 2 = 1

def favorable_outcomes : Finset (ℕ × ℕ) :=
  Finset.filter (λ p : ℕ × ℕ => is_less_than_10_and_odd (sum_of_two_dice p.1 p.2))
    (Finset.product roll_8_sided_die roll_8_sided_die)

theorem probability_of_sum_less_than_10_and_odd :
  (Finset.card favorable_outcomes : ℚ) / (Finset.card (Finset.product roll_8_sided_die roll_8_sided_die) : ℚ) = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_sum_less_than_10_and_odd_l272_27295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_congruent_l272_27210

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def congruent (a b : V) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ ∃ θ : ℝ, 0 < θ ∧ θ < Real.pi/2 ∧ ‖a‖ / ‖b‖ = Real.cos θ

theorem projection_of_congruent (a b : V) (h : congruent b a) :
  inner a (a - b) / ‖a‖ = (‖a‖^2 - ‖b‖^2) / ‖a‖ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_congruent_l272_27210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l272_27284

def a : ℕ → ℚ
  | 0 => 1/2  -- Add this case for n = 0
  | 1 => 1/2
  | (n+2) => a (n+1) + 1 / ((n+2)^2 + 3*(n+2) + 2)

theorem a_general_term (n : ℕ) : a n = n / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l272_27284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l272_27277

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  center : Point

/-- Checks if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

/-- Calculates the vector from one point to another -/
def vector (p1 p2 : Point) : Point :=
  ⟨p2.x - p1.x, p2.y - p1.y⟩

/-- Checks if two vectors are perpendicular -/
def isPerpendicular (v1 v2 : Point) : Prop :=
  v1.x * v2.x + v1.y * v2.y = 0

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)) / 2

theorem ellipse_property (e : Ellipse) (p f1 f2 : Point) :
  e.a > e.b ∧ e.b > 0 ∧
  isOnEllipse e p ∧
  isPerpendicular (vector p f1) (vector p f2) ∧
  triangleArea p f1 f2 = 9 →
  e.b = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l272_27277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_has_winning_strategy_l272_27272

/-- Represents a point on the hexagonal lattice grid -/
structure HexPoint where
  x : ℤ
  y : ℤ

/-- Represents the color of a point -/
inductive Color
  | Red
  | Blue
  | Uncolored

/-- Represents the game state -/
structure GameState where
  grid : HexPoint → Color
  turn : ℕ

/-- Checks if three points form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : HexPoint) : Prop :=
  sorry

/-- Checks if all points in the grid are colored -/
def allPointsColored (state : GameState) : Prop :=
  sorry

/-- Represents a player's move -/
structure Move where
  points : List HexPoint
  color : Color

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if a move is valid for the current player -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  sorry

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Prop :=
  sorry

/-- Checks if Player A (Alzim) has won -/
def playerAWins (state : GameState) : Prop :=
  sorry

/-- Represents a strategy for Player A -/
def Strategy := GameState → Move

/-- Helper function to apply a strategy multiple times -/
def applyStrategyNTimes (s : Strategy) (n : ℕ) (initialState : GameState) : GameState :=
  match n with
  | 0 => initialState
  | n+1 => applyMove (applyStrategyNTimes s n initialState) (s (applyStrategyNTimes s n initialState))

/-- The main theorem stating that Player A has a winning strategy -/
theorem player_a_has_winning_strategy :
  ∃ (s : Strategy), ∀ (initialState : GameState),
    (∀ (state : GameState), isValidMove state (s state)) →
    ∃ (n : ℕ), playerAWins (applyStrategyNTimes s n initialState) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_has_winning_strategy_l272_27272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_intercept_l272_27265

/-- A line intersecting a unit circle forms an equilateral triangle if and only if its y-intercept is ±√6/2 -/
theorem equilateral_triangle_intercept (m : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.1 - A.2 + m = 0 ∧ A.1^2 + A.2^2 = 1) ∧ 
    (B.1 - B.2 + m = 0 ∧ B.1^2 + B.2^2 = 1) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 1 ∧
    (A.1)^2 + (A.2)^2 = 1 ∧
    (B.1)^2 + (B.2)^2 = 1) ↔
  m = Real.sqrt 6 / 2 ∨ m = -(Real.sqrt 6 / 2) := by
  sorry

#check equilateral_triangle_intercept

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_intercept_l272_27265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_sums_l272_27244

noncomputable def z : ℂ := (1 - Complex.I * Real.sqrt 3) / 2

def sum_odd_powers (w : ℂ) : ℂ := w + w^3 + w^5 + w^7 + w^9 + w^11 + w^13 + w^15

theorem product_of_sums : sum_odd_powers z * sum_odd_powers (1 / z) = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_sums_l272_27244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_volume_greater_than_half_l272_27287

/-- Regular tetrahedron in 3D space -/
structure RegularTetrahedron where
  sideLength : ℝ
  center : ℝ × ℝ × ℝ

/-- The volume of the common part of two regular tetrahedra -/
noncomputable def commonVolume (t1 t2 : RegularTetrahedron) : ℝ := sorry

/-- Two regular tetrahedra with side length √6 and coincident centers -/
noncomputable def tetrahedra : RegularTetrahedron × RegularTetrahedron :=
  let side := Real.sqrt 6
  let center := (0, 0, 0)
  (⟨side, center⟩, ⟨side, center⟩)

/-- Theorem: The volume of the common part of two identical regular tetrahedra 
    with side length √6 and coincident centers is greater than 1/2 -/
theorem common_volume_greater_than_half :
  commonVolume tetrahedra.fst tetrahedra.snd > 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_volume_greater_than_half_l272_27287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_emptying_time_l272_27222

/-- Represents a right circular cone -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents the state of water in the cone at a given time -/
structure WaterState where
  height : ℝ
  time : ℝ

/-- The flow rate of water out of the cone -/
def flowRate (w : WaterState) : ℝ := w.height

/-- The volume of water in the cone given its current height -/
noncomputable def waterVolume (c : Cone) (w : WaterState) : ℝ :=
  (1/3) * Real.pi * (w.height * c.baseRadius / c.height)^2 * w.height

/-- The time it takes for all water to flow out of the cone -/
noncomputable def emptyingTime (c : Cone) : ℝ :=
  (9 * Real.pi) / 2

theorem cone_emptying_time (c : Cone) (h : c.height = 12 ∧ c.baseRadius = 3) :
  emptyingTime c = (9 * Real.pi) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_emptying_time_l272_27222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l272_27296

noncomputable def f (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

theorem min_value_of_f (x y : ℝ) (hx : 0.4 ≤ x ∧ x ≤ 0.6) (hy : 0.3 ≤ y ∧ y ≤ 0.4) :
  ∃ (m : ℝ), m = 0.5 ∧ ∀ (a b : ℝ), 0.4 ≤ a ∧ a ≤ 0.6 → 0.3 ≤ b ∧ b ≤ 0.4 → m ≤ f a b := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l272_27296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beam_equation_l272_27205

/-- The number of beams that can be bought with 6210 wen -/
def num_beams (x : ℕ) : Prop := x > 1

/-- The total price of the beams in wen -/
def total_price : ℕ := 6210

/-- The freight cost per beam in wen -/
def freight_per_beam : ℕ := 3

/-- The price of a single beam -/
def beam_price (x : ℕ) : ℚ := total_price / x

theorem beam_equation (x : ℕ) (h1 : num_beams x) :
  3 * (x - 1) = total_price / x := by
  sorry

#check beam_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beam_equation_l272_27205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_occurrence_l272_27261

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1)^2 / (x * (x^2 - 1))

def max_occurrence_values : Set ℝ := {a | a < -4 ∨ a > 4}

theorem f_max_occurrence :
  ∀ a : ℝ, (∃ x : ℝ, f x = a) →
    (∀ b : ℝ, (∃ x : ℝ, f x = b) →
      (∃ y : ℝ, f y = a ∧ y ≠ x) →
        (∃ z : ℝ, f z = b ∧ z ≠ x ∧ z ≠ y) →
          a ∈ max_occurrence_values) :=
by
  sorry

#check f_max_occurrence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_occurrence_l272_27261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_pyramid_l272_27226

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The four given points -/
def A : Point3D := ⟨4, 1, 1⟩
def B : Point3D := ⟨4, -2, -1⟩
def C : Point3D := ⟨-2, -2, -1⟩
def D : Point3D := ⟨-2, 1, -1⟩

/-- The sphere M passing through the four points -/
noncomputable def M : Sphere := sorry

/-- Volume of a sphere -/
noncomputable def sphereVolume (s : Sphere) : ℝ := (4 / 3) * Real.pi * s.radius ^ 3

/-- Volume of the triangular pyramid A-BCD -/
def pyramidVolume : ℝ := 6

/-- Theorem stating the probability of a random point inside sphere M being inside the pyramid A-BCD -/
theorem probability_in_pyramid : 
  (pyramidVolume / sphereVolume M) = 36 / (343 * Real.pi) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_pyramid_l272_27226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l272_27214

/-- The time taken for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  total_distance / train_speed_ms

/-- Theorem: The time taken for a train of length 110 m, traveling at 72 km/hr, 
    to cross a bridge of length 134 m is approximately 12.2 seconds -/
theorem train_crossing_bridge : 
  |train_crossing_time 110 72 134 - 12.2| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l272_27214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_coords_l272_27271

/-- Given a point P in a 3D rectangular coordinate system, 
    returns the point symmetric to P about the x-axis -/
def symmetry_about_x_axis (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (P.1, -P.2.1, -P.2.2)

/-- The point P in the rectangular coordinate system -/
def P : ℝ × ℝ × ℝ := (3, -2, 1)

/-- Theorem: The point symmetric to P(3,-2,1) about the x-axis 
    has coordinates (3, 2, -1) -/
theorem symmetric_point_coords : 
  symmetry_about_x_axis P = (3, 2, -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_coords_l272_27271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_length_l272_27219

/-- Represents a sphere with a given radius and center position -/
structure Sphere where
  radius : ℝ
  center : ℝ × ℝ × ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- Checks if two spheres are tangent -/
def are_tangent (s1 s2 : Sphere) : Prop :=
  let (x1, y1, z1) := s1.center
  let (x2, y2, z2) := s2.center
  (x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2 = (s1.radius + s2.radius)^2

/-- Checks if a tetrahedron circumscribes a set of spheres -/
def circumscribes (t : Tetrahedron) (spheres : Fin 4 → Sphere) : Prop :=
  ∀ (i : Fin 4), ∀ (j : Fin 4), i ≠ j →
    let (xi, yi, zi) := t.vertices i
    let (xj, yj, zj) := t.vertices j
    (xi - xj)^2 + (yi - yj)^2 + (zi - zj)^2 = 16

/-- The main theorem -/
theorem tetrahedron_edge_length 
  (spheres : Fin 4 → Sphere)
  (t : Tetrahedron)
  (h1 : ∀ i, (spheres i).radius = 2)
  (h2 : ∀ i j, i ≠ j → are_tangent (spheres i) (spheres j))
  (h3 : ∃ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ k ≠ i ∧
        (spheres i).center.2.2 = 0 ∧ (spheres j).center.2.2 = 0 ∧ (spheres k).center.2.2 = 0)
  (h4 : circumscribes t spheres) :
  ∀ (i j : Fin 4), i ≠ j →
    let (xi, yi, zi) := t.vertices i
    let (xj, yj, zj) := t.vertices j
    (xi - xj)^2 + (yi - yj)^2 + (zi - zj)^2 = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_length_l272_27219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_market_model_results_l272_27215

/-- Market model with linear demand and supply functions, and a per-unit tax -/
structure MarketModel where
  -- Demand function: Qd = a - bP
  a : ℝ
  b : ℝ
  -- Supply function: Qs = cP + d
  c : ℝ
  d : ℝ
  -- Per-unit tax
  t : ℝ

/-- Market equilibrium without tax -/
noncomputable def equilibrium (m : MarketModel) : ℝ × ℝ :=
  let p := (m.a - m.d) / (m.b + m.c)
  let q := m.a - m.b * p
  (p, q)

/-- Price elasticity of demand at equilibrium -/
noncomputable def demand_elasticity (m : MarketModel) : ℝ :=
  let (p, q) := equilibrium m
  m.b * p / q

/-- Price elasticity of supply at equilibrium -/
noncomputable def supply_elasticity (m : MarketModel) : ℝ :=
  let (p, q) := equilibrium m
  m.c * p / q

/-- Tax revenue -/
noncomputable def tax_revenue (m : MarketModel) : ℝ :=
  let p_producer := (m.a - m.t * m.b - m.d) / (m.b + m.c)
  let q := m.c * p_producer + m.d
  m.t * q

/-- Theorem stating the main results of the market model -/
theorem market_model_results (m : MarketModel) 
  (h1 : m.a = 688 ∧ m.b = 4)
  (h2 : supply_elasticity m = 1.5 * abs (demand_elasticity m))
  (h3 : (m.a - m.t * m.b - m.d) / (m.b + m.c) = 64)
  (h4 : m.t = 90) :
  m.c = 6 ∧ 
  m.d = -312 ∧ 
  tax_revenue m = 6480 ∧ 
  (∃ t_opt : ℝ, ∀ t : ℝ, tax_revenue {m with t := t} ≤ tax_revenue {m with t := t_opt} ∧ t_opt = 60) ∧
  (∃ max_revenue : ℝ, max_revenue = 8640 ∧ ∀ t : ℝ, tax_revenue {m with t := t} ≤ max_revenue) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_market_model_results_l272_27215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_distance_max_l272_27249

/-- Given a rhombus ABCD with side length a and ∠A = π/3, when folded along its diagonals
    to form a dihedral angle θ where θ ∈ [π/3, 2π/3], the maximum distance between
    the diagonals is 3/4 * a. -/
theorem rhombus_diagonal_distance_max (a : ℝ) (θ : ℝ) :
  (0 < a) →
  (θ ≥ π / 3) →
  (θ ≤ 2 * π / 3) →
  ∃ (d : ℝ), d ≤ 3 / 4 * a ∧
    ∀ (d' : ℝ), (∃ (θ' : ℝ), θ' ≥ π / 3 ∧ θ' ≤ 2 * π / 3 ∧
      d' = max (a * Real.sqrt 3 / 2 * Real.cos (θ' / 2)) (a / 2 * Real.cos (θ' / 2))) →
    d' ≤ d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_distance_max_l272_27249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_temperature_l272_27274

/-- Represents the temperature for each day of the week --/
structure WeekTemperatures where
  monday : ℚ
  tuesday : ℚ
  wednesday : ℚ
  thursday : ℚ
  friday : ℚ

/-- The average temperature of four consecutive days --/
def average4days (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

/-- Theorem: Given the conditions, the temperature on Friday is 33 degrees --/
theorem friday_temperature (w : WeekTemperatures) 
  (h1 : average4days w.monday w.tuesday w.wednesday w.thursday = 48)
  (h2 : average4days w.tuesday w.wednesday w.thursday w.friday = 46)
  (h3 : w.monday = 41) :
  w.friday = 33 := by
  sorry

#check friday_temperature

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_temperature_l272_27274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_bounds_l272_27221

open Set
open MeasureTheory
open Interval
open Real

-- Define the function type
def DiffFunction := {f : ℝ → ℝ // Differentiable ℝ f}

-- Define the property of the function
def SatisfiesConditions (f : DiffFunction) : Prop :=
  f.val 0 = 0 ∧ f.val 1 = 1 ∧ ∀ x : ℝ, |deriv f.val x| ≤ 2

-- Define the set of all possible integral values
noncomputable def IntegralValues (f : DiffFunction) : Set ℝ :=
  {y : ℝ | ∃ g : DiffFunction, SatisfiesConditions g ∧ y = ∫ x in (0:ℝ)..1, g.val x}

-- State the theorem
theorem integral_bounds (f : DiffFunction) (h : SatisfiesConditions f) :
  ∃ a b : ℝ, IntegralValues f = Ioo a b ∧ b - a = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_bounds_l272_27221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_point_correct_l272_27280

/-- The point on the line y = 2x from which segment AB is seen at the largest angle -/
def largest_angle_point : ℝ × ℝ := (2, 4)

/-- The line y = 2x -/
def line_c (x : ℝ) : ℝ := 2 * x

/-- Angle between three points -/
noncomputable def angle (P Q R : ℝ × ℝ) : ℝ := sorry

theorem largest_angle_point_correct :
  let A := (2, 2)
  let B := (6, 2)
  ∀ x : ℝ, (x, line_c x) ≠ largest_angle_point →
    angle (x, line_c x) A B < angle largest_angle_point A B :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_point_correct_l272_27280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_odd_function_iff_zero_at_origin_l272_27297

theorem sin_odd_function_iff_zero_at_origin (ω φ : ℝ) :
  (∀ x, Real.sin (ω * x + φ) = -Real.sin (ω * (-x) + φ)) ↔ Real.sin φ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_odd_function_iff_zero_at_origin_l272_27297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_counterfeit_coins_l272_27232

/-- A coin can be either genuine or counterfeit -/
inductive CoinType
| Genuine
| Counterfeit
deriving Repr, DecidableEq

/-- A device that tests two coins and returns the number of counterfeit coins -/
def testDevice (c1 c2 : CoinType) : Nat :=
  match c1, c2 with
  | CoinType.Genuine, CoinType.Genuine => 0
  | CoinType.Counterfeit, CoinType.Genuine => 1
  | CoinType.Genuine, CoinType.Counterfeit => 1
  | CoinType.Counterfeit, CoinType.Counterfeit => 2

/-- A set of 25 coins with exactly 2 counterfeit coins -/
def CoinSet := { s : Finset CoinType // s.card = 25 ∧ (s.filter (· = CoinType.Counterfeit)).card = 2 }

/-- The main theorem stating that it's possible to identify both counterfeit coins in at most 13 tests -/
theorem identify_counterfeit_coins (cs : CoinSet) :
  ∃ (strategy : Nat → CoinType × CoinType),
    (∀ n, n < 13 → (strategy n).1 ∈ cs.1 ∧ (strategy n).2 ∈ cs.1) ∧
    (∃ (i j : Nat), i < 13 ∧ j < 13 ∧ i ≠ j ∧
      (strategy i).1 = CoinType.Counterfeit ∧
      (strategy j).2 = CoinType.Counterfeit) :=
by
  sorry

#check identify_counterfeit_coins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_counterfeit_coins_l272_27232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sqrt_calculation_l272_27289

theorem correct_sqrt_calculation :
  (∀ x y : ℝ, x > 0 ∧ y > 0 → Real.sqrt (x * y) = Real.sqrt x * Real.sqrt y) ∧
  (Real.sqrt 20 ≠ 2 * Real.sqrt 10) ∧
  (Real.sqrt (3 * 5) = Real.sqrt 15) ∧
  (2 * Real.sqrt 2 * Real.sqrt 3 ≠ Real.sqrt 6) ∧
  (Real.sqrt ((-3)^2) ≠ -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sqrt_calculation_l272_27289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_trapezoid_midline_l272_27254

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithInscribedCircle where
  /-- The distance from the center of the inscribed circle to one end of a leg -/
  d1 : ℝ
  /-- The distance from the center of the inscribed circle to the other end of the same leg -/
  d2 : ℝ

/-- The midline of a trapezoid -/
noncomputable def midline (t : RightTrapezoidWithInscribedCircle) : ℝ :=
  18 * Real.sqrt 5 / 5

/-- Theorem: The midline of a right trapezoid with an inscribed circle,
    where the center of the circle is at distances of 8 cm and 4 cm
    from the ends of one of the legs, is (18 * √5) / 5 cm -/
theorem right_trapezoid_midline
  (t : RightTrapezoidWithInscribedCircle)
  (h1 : t.d1 = 8)
  (h2 : t.d2 = 4) :
  midline t = 18 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_trapezoid_midline_l272_27254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lukas_games_played_l272_27291

/-- Given a basketball player's average points per game and total points scored,
    calculate the number of games played. -/
noncomputable def games_played (avg_points_per_game : ℚ) (total_points : ℚ) : ℚ :=
  total_points / avg_points_per_game

/-- Theorem: Lukas played 5 games given his average points and total score. -/
theorem lukas_games_played :
  let avg_points : ℚ := 12
  let total_points : ℚ := 60
  games_played avg_points total_points = 5 := by
  -- Unfold the definition of games_played
  unfold games_played
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lukas_games_played_l272_27291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_silo_storage_ratio_l272_27240

/-- The diameter of Bryan's silo in cm -/
noncomputable def bryan_diameter : ℝ := 8

/-- The height of Bryan's silo in cm -/
noncomputable def bryan_height : ℝ := 20

/-- The fill percentage of Bryan's silo -/
noncomputable def bryan_fill_percentage : ℝ := 0.9

/-- The diameter of Sara's silo in cm -/
noncomputable def sara_diameter : ℝ := 10

/-- The height of Sara's silo in cm -/
noncomputable def sara_height : ℝ := 16

/-- The fill percentage of Sara's silo -/
noncomputable def sara_fill_percentage : ℝ := 0.85

/-- The effective storage ratio of Bryan's silo to Sara's silo -/
noncomputable def effective_storage_ratio : ℝ := 18 / 17

theorem silo_storage_ratio :
  let bryan_volume := π * (bryan_diameter / 2) ^ 2 * bryan_height
  let sara_volume := π * (sara_diameter / 2) ^ 2 * sara_height
  let bryan_effective_storage := bryan_volume * bryan_fill_percentage
  let sara_effective_storage := sara_volume * sara_fill_percentage
  bryan_effective_storage / sara_effective_storage = effective_storage_ratio := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_silo_storage_ratio_l272_27240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_4_2_l272_27256

-- Define a power function that passes through (4, 2)
noncomputable def f : ℝ → ℝ := fun x => x ^ (Real.log 2 / Real.log 4)

-- Theorem statement
theorem power_function_through_4_2 : f 16 = 4 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp [Real.rpow_def]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_4_2_l272_27256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_triple_solution_l272_27202

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def digits_distinct (n : ℕ) : Prop :=
  let digits := List.range 6 |>.map (fun i => (n / (10^i)) % 10)
  digits.Nodup

def first_digit_nonzero (n : ℕ) : Prop := n / 100000 ≠ 0

def fifth_digit_nonzero (n : ℕ) : Prop := (n / 10) % 10 ≠ 0

def rotated_number (n : ℕ) : ℕ := (n % 100) * 10000 + n / 100

theorem six_digit_triple_solution :
  ∃! (s : Finset ℕ), s.card = 3 ∧
  (∀ n ∈ s, is_six_digit n ∧ digits_distinct n ∧
            first_digit_nonzero n ∧ fifth_digit_nonzero n ∧
            3 * n = rotated_number n) := by
  sorry

#check six_digit_triple_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_triple_solution_l272_27202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wage_proof_l272_27250

/-- The weekly wage of employee Y in rupees -/
noncomputable def wage_Y : ℝ := 254.55

/-- The percentage of Y's wage that X is paid -/
def percentage_X : ℝ := 120

/-- The weekly wage of employee X in rupees -/
noncomputable def wage_X : ℝ := (percentage_X / 100) * wage_Y

/-- The total amount paid to both employees per week in rupees -/
noncomputable def total_wage : ℝ := wage_X + wage_Y

/-- Theorem stating the total wage paid to both employees -/
theorem total_wage_proof : total_wage = 560.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wage_proof_l272_27250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_100_l272_27252

/-- The distance covered by Sourav given his initial and increased speeds -/
noncomputable def distance_covered (initial_speed increased_speed : ℝ) : ℝ :=
  let usual_time := 100 / initial_speed
  let reduced_time := usual_time - 1
  increased_speed * reduced_time

/-- Theorem stating that the distance covered is 100 km -/
theorem distance_is_100 (initial_speed increased_speed : ℝ) 
  (h1 : initial_speed = 20)
  (h2 : increased_speed = 25)
  (h3 : distance_covered initial_speed increased_speed = increased_speed * (100 / initial_speed - 1)) :
  distance_covered initial_speed increased_speed = 100 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval distance_covered 20 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_100_l272_27252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_percent_error_circle_square_l272_27241

noncomputable def actual_length : ℝ := 30
noncomputable def error_rate : ℝ := 0.3

noncomputable def min_measurement : ℝ := actual_length * (1 - error_rate)
noncomputable def max_measurement : ℝ := actual_length * (1 + error_rate)

noncomputable def circle_area (d : ℝ) : ℝ := Real.pi * (d / 2) ^ 2
noncomputable def square_area (s : ℝ) : ℝ := s ^ 2

noncomputable def percent_error (actual area : ℝ) : ℝ := |area - actual| / actual * 100

theorem max_percent_error_circle_square :
  let actual_circle_area := circle_area actual_length
  let actual_square_area := square_area actual_length
  let max_circle_area := circle_area max_measurement
  let max_square_area := square_area max_measurement
  ∀ d s, min_measurement ≤ d ∧ d ≤ max_measurement →
         min_measurement ≤ s ∧ s ≤ max_measurement →
         percent_error actual_circle_area (circle_area d) ≤ 69 ∧
         percent_error actual_square_area (square_area s) ≤ 69 ∧
         (percent_error actual_circle_area max_circle_area = 69 ∨
          percent_error actual_square_area max_square_area = 69) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_percent_error_circle_square_l272_27241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airport_distance_l272_27227

/-- The distance from David's home to the airport -/
def distance : ℝ := sorry

/-- The time David would have taken if he continued at the initial speed -/
def initial_time : ℝ := sorry

/-- David's initial speed in miles per hour -/
def initial_speed : ℝ := 40

/-- David's increased speed in miles per hour -/
def increased_speed : ℝ := 60

/-- The time David actually spent traveling -/
def actual_time : ℝ := sorry

theorem airport_distance :
  (distance = initial_speed * (initial_time + 1.5)) ∧
  (distance - initial_speed * 1 = increased_speed * (actual_time - 1.75)) ∧
  (initial_time + 1.5 = actual_time + 0.75) →
  distance = 310 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_airport_distance_l272_27227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_quadratic_roots_pairs_l272_27290

theorem infinite_quadratic_roots_pairs : 
  ∃ (S : Set (ℤ × ℤ)), Set.Infinite S ∧ 
  (∀ (p : ℤ × ℤ), p ∈ S → 
    let (b, c) := p
    c < 2000 ∧ 
    ∀ (x : ℂ), x^2 - b*x + c = 0 → Complex.abs (Complex.re x) > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_quadratic_roots_pairs_l272_27290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_properties_l272_27270

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.cos x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := x + Real.sin x

-- Theorem statement
theorem f_derivative_properties :
  (∀ x ∈ Set.Icc (-1) 1, f' x = x + Real.sin x) ∧
  (∀ x, f' (-x) = -(f' x)) ∧
  (∃ (m M : ℝ), ∀ x ∈ Set.Icc (-1) 1, m ≤ f' x ∧ f' x ≤ M) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_properties_l272_27270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_max_area_l272_27288

/-- Ellipse C with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1^2 / a^2 + p.2^2 / b^2 = 1}

/-- Circle O centered at the origin with radius r -/
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = r^2}

/-- Line l: y = x + 2 -/
def Line : Set (ℝ × ℝ) :=
  {p | p.2 = p.1 + 2}

/-- Curve |y| = kx for k > 0 -/
def Curve (k : ℝ) : Set (ℝ × ℝ) :=
  {p | |p.2| = k * p.1}

/-- Predicate to check if a line is tangent to a circle -/
def IsTangentTo (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop := sorry

/-- Function to calculate the area of a triangle formed by the origin and two points -/
noncomputable def AreaOfTriangle (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem ellipse_equation_and_max_area 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : (a^2 - b^2) / a^2 = 3/9) -- eccentricity squared
  (h4 : IsTangentTo Line (Circle b)) :
  (Ellipse a b = Ellipse (Real.sqrt 3) (Real.sqrt 2)) ∧
  (∃ (k : ℝ), k > 0 ∧ 
    (∀ (k' : ℝ), k' > 0 → 
      AreaOfTriangle (Ellipse a b ∩ Curve k') ≤ Real.sqrt 6 / 2) ∧
    AreaOfTriangle (Ellipse a b ∩ Curve k) = Real.sqrt 6 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_max_area_l272_27288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_POQ_l272_27286

/-- Circle C with equation x² + y² = 1 -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Point A lies on the circle C -/
def PointA (x y : ℝ) : Prop := Circle x y

/-- Point B is on the y-axis -/
def PointB (_ : ℝ) : Prop := True

/-- Point P satisfies BP = 2BA -/
def PointP (x y ax ay : ℝ) : Prop := x = 2 * ax ∧ y = ay

/-- Point Q is on the line x = 3 -/
def PointQ (_ : ℝ) : Prop := True

/-- O is the origin (0, 0) -/
def Origin : ℝ × ℝ := (0, 0)

/-- OP is perpendicular to OQ -/
def Perpendicular (px py qx qy : ℝ) : Prop := px * qx + py * qy = 0

/-- Area of triangle POQ -/
noncomputable def AreaPOQ (px py qy : ℝ) : ℝ := (1/2) * px * (3 - 0) * (qy - py)

/-- The minimum area of triangle POQ is 3/2 -/
theorem min_area_POQ :
  ∀ (ax ay px py qy : ℝ),
  PointA ax ay →
  PointP px py ax ay →
  PointQ qy →
  Perpendicular px py 3 qy →
  ∀ (area : ℝ), area = AreaPOQ px py qy →
  area ≥ (3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_POQ_l272_27286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_value_of_f_l272_27237

/-- The function f(x) = 5x^4 - 12x^3 + 30x^2 - 12x + 5 -/
def f (x : ℝ) : ℝ := 5*x^4 - 12*x^3 + 30*x^2 - 12*x + 5

/-- x₁ is a non-negative integer -/
def x₁ : ℕ := sorry

/-- p is a prime number -/
def p : ℕ := sorry

/-- p is the result of f(x₁) -/
axiom h_fp : f (x₁ : ℝ) = p

/-- p is prime -/
axiom h_p_prime : Nat.Prime p

theorem max_prime_value_of_f :
  ∀ q : ℕ, Nat.Prime q → (∃ y : ℕ, f (y : ℝ) = q) → q ≤ 5 :=
by
  sorry

#check max_prime_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_value_of_f_l272_27237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_rises_left_to_right_l272_27269

/-- The inverse proportion function -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

/-- A function rises from left to right if its derivative is positive -/
def rises_left_to_right (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → (deriv f) x > 0

/-- Theorem: The inverse proportion function with k = -2 rises from left to right -/
theorem inverse_proportion_rises_left_to_right :
  rises_left_to_right (inverse_proportion (-2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_rises_left_to_right_l272_27269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l272_27260

def Student := Fin 6

structure Arrangement where
  day1 : Fin 2 → Student
  day2 : Fin 2 → Student
  day3 : Fin 2 → Student

def is_valid_arrangement (arr : Arrangement) (student_A student_B student_C : Student) : Prop :=
  (student_A ∉ Set.range arr.day1) ∧
  (student_B ∉ Set.range arr.day2 ∨ student_C ∉ Set.range arr.day2) ∧
  (student_B ∉ Set.range arr.day3 ∨ student_C ∉ Set.range arr.day3) ∧
  (∀ (i j : Fin 2), i ≠ j → arr.day1 i ≠ arr.day1 j) ∧
  (∀ (i j : Fin 2), i ≠ j → arr.day2 i ≠ arr.day2 j) ∧
  (∀ (i j : Fin 2), i ≠ j → arr.day3 i ≠ arr.day3 j) ∧
  (∀ s : Student, s ∈ Set.range arr.day1 ∪ Set.range arr.day2 ∪ Set.range arr.day3)

-- Add this instance to provide a finite type for valid arrangements
instance (student_A student_B student_C : Student) : 
  Fintype { arr : Arrangement // is_valid_arrangement arr student_A student_B student_C } :=
  sorry

theorem arrangement_count (student_A student_B student_C : Student) 
  (h_distinct : student_A ≠ student_B ∧ student_A ≠ student_C ∧ student_B ≠ student_C) :
  Fintype.card { arr : Arrangement // is_valid_arrangement arr student_A student_B student_C } = 48 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l272_27260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_scalar_multiple_l272_27220

/-- Given a 2x2 matrix B with elements [[3, 4], [7, e]], 
    prove that if B^(-1) = m * B, then e = -3 and m = 1/19 -/
theorem inverse_scalar_multiple (e m : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 7, e]
  B⁻¹ = m • B → e = -3 ∧ m = 1/19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_scalar_multiple_l272_27220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_l272_27251

open Real

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := sqrt 2 * cos (3 * x - π / 4)

/-- The reference function -/
noncomputable def g (x : ℝ) : ℝ := sqrt 2 * cos (3 * x)

/-- The shifted reference function -/
noncomputable def h (x : ℝ) : ℝ := g (x - π / 12)

theorem graph_shift :
  ∀ x, f x = h x :=
by
  intro x
  simp [f, h, g]
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_l272_27251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_journey_l272_27292

/-- A hiker's journey with resupply --/
theorem hiker_journey (supplies_per_mile : ℝ) (initial_pack : ℝ) (resupply_ratio : ℝ) 
  (hiking_speed : ℝ) (hiking_days : ℕ) (h1 : supplies_per_mile = 0.5) 
  (h2 : initial_pack = 40) (h3 : resupply_ratio = 0.25) (h4 : hiking_speed = 2.5) 
  (h5 : hiking_days = 5) : 
  (initial_pack / supplies_per_mile + initial_pack * resupply_ratio / supplies_per_mile) / 
  (hiking_days : ℝ) / hiking_speed = 8 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_journey_l272_27292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_polynomial_l272_27276

/-- The polynomial for which we want to find the roots -/
def f (x : ℝ) : ℝ := x^3 - x^2 - 6*x + 8

/-- The first root of the polynomial -/
def root1 : ℝ := 2

/-- The second root of the polynomial -/
noncomputable def root2 : ℝ := (-1 + Real.sqrt 17) / 2

/-- The third root of the polynomial -/
noncomputable def root3 : ℝ := (-1 - Real.sqrt 17) / 2

/-- Theorem stating that root1, root2, and root3 are the roots of the polynomial f -/
theorem roots_of_polynomial :
  f root1 = 0 ∧ f root2 = 0 ∧ f root3 = 0 ∧
  ∀ x : ℝ, f x = 0 → x = root1 ∨ x = root2 ∨ x = root3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_polynomial_l272_27276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_jars_same_coins_l272_27259

/-- Represents the number of coins in each jar -/
def CoinDistribution := Fin 2017 → ℕ

/-- Represents the process of adding coins to 10 consecutive jars -/
def add_coins (d : CoinDistribution) (start : Fin 2017) : CoinDistribution :=
  λ i ↦ if start ≤ i ∧ i < start + 10 then d i + 1 else d i

/-- Checks if a given number of jars have the same number of coins -/
def same_coins (d : CoinDistribution) (n : ℕ) : Prop :=
  ∃ (coin_count : ℕ) (jars : Finset (Fin 2017)),
    jars.card = n ∧ ∀ j ∈ jars, d j = coin_count

/-- The main theorem stating the maximum number of jars with the same number of coins -/
theorem max_jars_same_coins :
  ∃ (final_distribution : CoinDistribution),
    (∀ d : CoinDistribution, ∀ n : ℕ, same_coins d n → n ≤ 2014) ∧
    same_coins final_distribution 2014 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_jars_same_coins_l272_27259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_number_ordering_l272_27253

/-- Given 4 distinct real numbers, prove that at most 5 pairwise comparisons
    are needed to determine their descending order. -/
theorem four_number_ordering (a b c d : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  ∃ (p : Fin 4 → ℝ), 
    (∀ i j : Fin 4, i < j → p i > p j) ∧
    (∃ (comparisons : List (Fin 4 × Fin 4)),
      comparisons.length ≤ 5 ∧
      ∀ i j : Fin 4, i < j → 
        (i, j) ∈ comparisons ∨ (j, i) ∈ comparisons) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_number_ordering_l272_27253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunny_lead_in_second_race_l272_27255

/-- Represents the result of a race between Sunny and Windy -/
structure RaceResult where
  distance : ℝ  -- Race distance
  sunny_lead : ℝ  -- How far ahead (or behind) Sunny is when finishing

/-- Given the race parameters, calculates the result of the second race -/
noncomputable def second_race_result (h d : ℝ) : RaceResult :=
  { distance := h^2 / 100,
    sunny_lead := 400 * d^2 / h^2 }

/-- Theorem stating that given the race conditions, Sunny finishes the second race 400d²/h² meters ahead of Windy -/
theorem sunny_lead_in_second_race (h d : ℝ) (h_pos : h > 0) (d_pos : d > 0) :
  let first_race := RaceResult.mk (h^2 / 100) (2 * d)
  let second_race := second_race_result h d
  second_race.sunny_lead = 400 * d^2 / h^2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunny_lead_in_second_race_l272_27255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_l272_27242

theorem least_subtraction (n : ℕ) : 
  (∀ d ∈ ({9, 11, 13} : Set ℕ), (2590 - n) % d = 6) ∧ 
  (∀ m < n, ∃ d ∈ ({9, 11, 13} : Set ℕ), (2590 - m) % d ≠ 6) → 
  n = 28 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_l272_27242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l272_27200

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

def r1 : ℝ := 4
def r2 : ℝ := 6
def r3 : ℝ := 8

theorem snowman_volume :
  sphere_volume r1 + sphere_volume r2 + sphere_volume r3 = (3168 / 3) * Real.pi :=
by
  -- Expand the definition of sphere_volume
  unfold sphere_volume
  -- Simplify the expressions
  simp [r1, r2, r3]
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l272_27200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_specific_parabola_vertex_l272_27267

/-- The vertex of a parabola y = ax² + bx + c is at (-b/(2a), f(-b/(2a))) where f(x) = ax² + bx + c -/
theorem parabola_vertex (a b c : ℝ) (ha : a ≠ 0) :
  let f := λ x : ℝ => a * x^2 + b * x + c
  let x₀ := -b / (2 * a)
  (x₀, f x₀) = (-b / (2 * a), -(b^2 - 4 * a * c) / (4 * a)) := by sorry

/-- The vertex of the parabola y = x² - 12x + 9 is at (6, -27) -/
theorem specific_parabola_vertex :
  let f := λ x : ℝ => x^2 - 12 * x + 9
  (6, f 6) = (6, -27) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_specific_parabola_vertex_l272_27267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l272_27204

/-- The distance between the vertices of a hyperbola with equation x^2/121 - y^2/49 = 1 is 22 -/
theorem hyperbola_vertex_distance :
  let a : ℝ := Real.sqrt 121
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 121 - y^2 / 49 = 1
  2 * a = 22 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l272_27204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ray_tom_combined_mpg_l272_27212

/-- The combined rate of miles per gallon for two cars -/
noncomputable def combined_mpg (mpg1 : ℝ) (mpg2 : ℝ) (distance : ℝ) : ℝ :=
  (2 * distance) / (distance / mpg1 + distance / mpg2)

/-- Theorem: The combined rate of miles per gallon for Ray and Tom's trip -/
theorem ray_tom_combined_mpg :
  let ray_mpg : ℝ := 50
  let tom_mpg : ℝ := 20
  let distance : ℝ := 100
  abs (combined_mpg ray_mpg tom_mpg distance - 200/7) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ray_tom_combined_mpg_l272_27212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_earnings_l272_27233

/-- Calculates the total earnings from selling apples and oranges in a cafeteria --/
theorem cafeteria_earnings :
  let initial_apples : ℕ := 50
  let initial_oranges : ℕ := 40
  let apple_price : ℚ := 4/5
  let orange_price : ℚ := 1/2
  let remaining_apples : ℕ := 10
  let remaining_oranges : ℕ := 6
  (initial_apples - remaining_apples : ℚ) * apple_price + 
  (initial_oranges - remaining_oranges : ℚ) * orange_price = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_earnings_l272_27233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l272_27263

-- Define the custom operation
noncomputable def custom_op (x y : ℝ) : ℝ :=
  if x ≤ y then x else y

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (custom_op (|m - 1|) m = |m - 1|) ↔ m ∈ Set.Ici (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l272_27263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_sum_implies_m_eq_two_l272_27293

/-- Given a real number m, we define the function f. -/
noncomputable def f (m : ℝ) : ℝ → ℝ := λ x ↦ Real.log (x + m)

/-- Given a real number m, we define the function g. -/
noncomputable def g (m : ℝ) : ℝ → ℝ := λ x ↦ m - Real.exp (-x)

/-- The main theorem stating the value of m given the symmetry and sum condition. -/
theorem symmetry_and_sum_implies_m_eq_two (m : ℝ) :
  (∀ x y : ℝ, f m x = y ↔ g m (-y) = -x) →
  g m 0 + g m (-Real.log 2) = 1 →
  m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_sum_implies_m_eq_two_l272_27293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_fourth_power_l272_27230

theorem det_B_fourth_power {n : Type*} [Fintype n] [DecidableEq n] 
  (B : Matrix n n ℝ) (h : Matrix.det B = -3) : Matrix.det (B^4) = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_fourth_power_l272_27230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_with_m_zeros_l272_27201

theorem prime_with_m_zeros (m : ℕ) : ∃ p : ℕ, Nat.Prime p ∧ (∃ k : ℕ, p = 10^(m.succ + 1) * k + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_with_m_zeros_l272_27201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l272_27218

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 3 then 7 * x + 20 else 3 * x - 18

-- Theorem statement
theorem sum_of_solutions :
  ∃ x₁ x₂, f x₁ = 5 ∧ f x₂ = 5 ∧ x₁ + x₂ = 116 / 21 := by
  -- We'll use -15/7 and 23/3 as our solutions
  let x₁ : ℝ := -15/7
  let x₂ : ℝ := 23/3
  
  have h₁ : f x₁ = 5 := by
    -- Proof for f(-15/7) = 5
    sorry
  
  have h₂ : f x₂ = 5 := by
    -- Proof for f(23/3) = 5
    sorry
  
  have h₃ : x₁ + x₂ = 116 / 21 := by
    -- Proof for -15/7 + 23/3 = 116/21
    sorry
  
  exact ⟨x₁, x₂, h₁, h₂, h₃⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l272_27218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_term_l272_27247

/-- 
Given the expansion of (2x - 1/2)^6, prove that the term with the largest 
binomial coefficient is -20x^3.
-/
theorem largest_coefficient_term (x : ℝ) : 
  ∃ k : ℕ, 0 ≤ k ∧ k ≤ 6 ∧ 
    (∀ j : ℕ, 0 ≤ j ∧ j ≤ 6 → 
      (Nat.choose 6 k) * (2*x)^k * (-1/2)^(6-k) ≥ (Nat.choose 6 j) * (2*x)^j * (-1/2)^(6-j)) ∧
    (Nat.choose 6 k) * (2*x)^k * (-1/2)^(6-k) = -20 * x^3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_term_l272_27247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_sum_condition_l272_27211

/-- Two geometric progressions with first terms a and b, and common ratios p and q respectively -/
def geometric_progression_1 (a p : ℝ) (n : ℕ) : ℕ → ℝ := λ k ↦ a * p^k
def geometric_progression_2 (b q : ℝ) (n : ℕ) : ℕ → ℝ := λ k ↦ b * q^k

/-- The series formed by the sums of corresponding terms of the two geometric progressions -/
def sum_series (a b p q : ℝ) (n : ℕ) : ℕ → ℝ := 
  λ k ↦ geometric_progression_1 a p n k + geometric_progression_2 b q n k

/-- A sequence is a geometric progression if the ratio of successive terms is constant -/
def is_geometric_progression (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ k : ℕ, s (k + 1) = r * s k

theorem geometric_progression_sum_condition (a b p q : ℝ) (n : ℕ) :
  is_geometric_progression (sum_series a b p q n) ↔ p = q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_sum_condition_l272_27211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_30_l272_27279

theorem probability_factor_of_30 : 
  (Finset.filter (λ n : ℕ => n > 0 ∧ n ≤ 30 ∧ 30 % n = 0) (Finset.range 31)).card / 30 = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_30_l272_27279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_good_set_element_l272_27245

/-- The set A = {1, 2, ..., 2016} -/
def A : Finset ℕ := Finset.range 2016

/-- A subset X of A is a "good set" if there exist x, y ∈ X such that x < y and x | y -/
def is_good_set (X : Finset ℕ) : Prop :=
  ∃ x y, x ∈ X ∧ y ∈ X ∧ x < y ∧ x ∣ y

/-- The theorem stating that 671 is the largest positive integer with the given property -/
theorem largest_good_set_element : 
  ∀ a ∈ A, a ≤ 671 ↔ 
    ∀ X : Finset ℕ, X ⊆ A → X.card = 1008 → a ∈ X → is_good_set X :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_good_set_element_l272_27245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_element_is_fifteen_l272_27217

def is_valid_set (T : Finset Nat) : Prop :=
  T.card = 5 ∧
  (∀ x, x ∈ T → 2 ≤ x ∧ x ≤ 15) ∧
  (∀ c d, c ∈ T → d ∈ T → c < d → ¬(d ∣ c))

theorem max_element_is_fifteen :
  ∃ T : Finset Nat, is_valid_set T ∧ 
  (∀ x, x ∈ T → x ≤ 15) ∧
  (∃ y, y ∈ T ∧ y = 15) ∧
  (∀ S : Finset Nat, is_valid_set S → ∀ z, z ∈ S → z ≤ 15) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_element_is_fifteen_l272_27217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_ant_fall_off_time_l272_27234

/-- Represents the state of an ant on the checkerboard -/
structure Ant where
  x : ℝ
  y : ℝ
  direction : ℕ  -- 0: up, 1: right, 2: down, 3: left

/-- Represents the checkerboard and its rules -/
class Checkerboard (m : ℕ) where
  size : m > 0
  board : Set Ant
  move : Ant → ℝ → Ant
  collision : List Ant → List Ant

/-- The latest time an ant can be at position (x, y) moving upward -/
noncomputable def f (cb : Checkerboard m) (x y : ℝ) : ℝ := sorry

/-- The theorem stating the latest possible moment for the last ant to fall off -/
theorem last_ant_fall_off_time (m : ℕ) (cb : Checkerboard m) :
  (∀ (x y : ℝ), x ∈ Set.Icc (1 : ℝ) m → y ∈ Set.Icc (1 : ℝ) m → 
    f cb x y ≤ y - 1 + min (x - 1) (m - x)) →
  (∃ (t : ℝ), t = (3 / 2 : ℝ) * m - 1 ∧
    ∀ (a : Ant), a ∈ cb.board → 
      cb.move a t ∉ { a : Ant | a.x ∈ Set.Icc (1 : ℝ) m ∧ a.y ∈ Set.Icc (1 : ℝ) m }) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_ant_fall_off_time_l272_27234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_theorem_l272_27208

/-- The inclination angle of a line with equation x*sin(θ) + √3*y + 2 = 0 -/
noncomputable def inclination_angle (θ : ℝ) : ℝ :=
  Real.arctan (-Real.sin θ / Real.sqrt 3)

/-- The range of the inclination angle -/
def inclination_angle_range : Set ℝ :=
  Set.Icc 0 (Real.pi / 6) ∪ Set.Ico (5 * Real.pi / 6) Real.pi

/-- Theorem stating that the inclination angle is always within the specified range -/
theorem inclination_angle_theorem :
  ∀ θ, inclination_angle θ ∈ inclination_angle_range :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_theorem_l272_27208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_distances_l272_27231

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := 2 * Real.sqrt x

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance function
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem ratio_of_distances 
  (P Q F : Point)
  (h1 : P.y = curve P.x)
  (h2 : Q.y = curve Q.x)
  (h3 : F.x = 1 ∧ F.y = 0)
  (h4 : Q.x = 3 * P.x + 2)
  : distance Q F / distance P F = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_distances_l272_27231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coord_for_y_five_l272_27228

/-- A line in the coordinate plane passing through the origin -/
structure Line where
  slope : ℚ

/-- A point in the coordinate plane -/
structure Point where
  x : ℚ
  y : ℚ

/-- Check if a point is on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x

/-- The x-coordinate of a point on a line with a given y-coordinate -/
noncomputable def x_coord_at_y (l : Line) (y : ℚ) : ℚ :=
  y / l.slope

theorem x_coord_for_y_five (k : Line) (h : k.slope = 1/5) :
  on_line ⟨5, 5⟩ k → x_coord_at_y k 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coord_for_y_five_l272_27228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l272_27229

/-- The line equation: kx - y - 4k + 3 = 0 --/
def line_equation (k x y : ℝ) : Prop := k * x - y - 4 * k + 3 = 0

/-- The circle equation: x^2 + y^2 - 6x - 8y + 21 = 0 --/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6 * x - 8 * y + 21 = 0

/-- The minimum chord length theorem --/
theorem min_chord_length :
  ∃ (k : ℝ), ∀ (x y : ℝ),
    line_equation k x y ∧ circle_equation x y →
    ∃ (x' y' : ℝ), line_equation k x' y' ∧ circle_equation x' y' ∧
      ∀ (x'' y'' : ℝ), line_equation k x'' y'' ∧ circle_equation x'' y'' →
        (x - x')^2 + (y - y')^2 ≤ (x - x'')^2 + (y - y'')^2 ∧
        (x - x')^2 + (y - y')^2 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l272_27229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_cannot_catch_nut_l272_27236

/-- The initial distance between Gavrila and the squirrel in meters -/
noncomputable def initial_distance : ℝ := 3.75

/-- The initial horizontal velocity of the thrown nut in m/s -/
noncomputable def initial_velocity : ℝ := 5

/-- The maximum distance the squirrel can jump in meters -/
noncomputable def squirrel_jump_distance : ℝ := 1.7

/-- The acceleration due to gravity in m/s² -/
noncomputable def gravity : ℝ := 10

/-- The square of the distance between the nut and the squirrel at time t -/
noncomputable def distance_squared (t : ℝ) : ℝ :=
  (initial_velocity * t - initial_distance)^2 + (1/2 * gravity * t^2)^2

/-- The minimum distance between the nut and the squirrel -/
noncomputable def min_distance : ℝ := 5 * Real.sqrt 2 / 4

theorem squirrel_cannot_catch_nut :
  min_distance > squirrel_jump_distance := by
  sorry

#eval "Theorem stated and proof skipped with sorry."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_cannot_catch_nut_l272_27236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_S_non_negative_l272_27282

noncomputable def A : Finset ℕ := {0, 1, 2, 3}
noncomputable def B : Finset ℕ := {0, 1, 2}

noncomputable def S (a b : ℕ) : ℝ := Real.sin ((a - b : ℝ) / 3 * Real.pi)

noncomputable def favorable_outcomes : Finset (ℕ × ℕ) :=
  (A.product B).filter (fun (a, b) => S a b ≥ 0)

theorem probability_S_non_negative :
  (favorable_outcomes.card : ℝ) / ((A.card * B.card) : ℝ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_S_non_negative_l272_27282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_two_equals_three_l272_27213

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of y being an even function
def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- State the theorem
theorem f_minus_two_equals_three
  (h1 : is_even (λ x ↦ f (2 * x) + x))
  (h2 : f 2 = 1) :
  f (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_two_equals_three_l272_27213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_union_sets_l272_27275

/-- Represents a set of people with their total age and count. -/
structure PeopleSet where
  totalAge : ℝ
  count : ℝ

/-- Calculates the average age of a PeopleSet. -/
noncomputable def averageAge (s : PeopleSet) : ℝ := s.totalAge / s.count

theorem average_age_union_sets (A B C : PeopleSet)
  (hDisjoint : A.count + B.count + C.count > 0)
  (hAvgA : averageAge A = 30)
  (hAvgB : averageAge B = 25)
  (hAvgC : averageAge C = 50)
  (hAvgAB : averageAge ⟨A.totalAge + B.totalAge, A.count + B.count⟩ = 27)
  (hAvgAC : averageAge ⟨A.totalAge + C.totalAge, A.count + C.count⟩ = 42)
  (hAvgBC : averageAge ⟨B.totalAge + C.totalAge, B.count + C.count⟩ = 35) :
  averageAge ⟨A.totalAge + B.totalAge + C.totalAge, A.count + B.count + C.count⟩ = 35.625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_union_sets_l272_27275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_order_l272_27207

noncomputable section

/-- Inclination angle of a line -/
def inclination_angle (m : ℝ) : ℝ := Real.arctan m

/-- The line l₁: x - y = 0 -/
def l₁ (x y : ℝ) : Prop := x - y = 0

/-- The line l₂: x + 2y = 0 -/
def l₂ (x y : ℝ) : Prop := x + 2*y = 0

/-- The line l₃: x + 3y = 0 -/
def l₃ (x y : ℝ) : Prop := x + 3*y = 0

/-- The slope of l₁ -/
def m₁ : ℝ := 1

/-- The slope of l₂ -/
def m₂ : ℝ := -1/2

/-- The slope of l₃ -/
def m₃ : ℝ := -1/3

/-- The inclination angle of l₁ -/
def α₁ : ℝ := inclination_angle m₁

/-- The inclination angle of l₂ -/
def α₂ : ℝ := inclination_angle m₂

/-- The inclination angle of l₃ -/
def α₃ : ℝ := inclination_angle m₃

theorem inclination_angle_order : α₁ < α₂ ∧ α₂ < α₃ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_order_l272_27207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_fuel_efficiency_l272_27262

/-- Represents the fuel efficiency of a car in miles per gallon -/
structure FuelEfficiency where
  highway : ℝ
  city : ℝ

/-- Calculates the amount of gasoline used for a given distance -/
noncomputable def gasolineUsed (efficiency : FuelEfficiency) (distance : ℝ) (isHighway : Bool) : ℝ :=
  if isHighway then distance / efficiency.highway else distance / efficiency.city

/-- The main theorem to prove -/
theorem city_fuel_efficiency (efficiency : FuelEfficiency) :
  efficiency.highway = 34 →
  gasolineUsed efficiency 4 true + gasolineUsed efficiency 4 false =
  gasolineUsed efficiency 8 true * (1 + 0.34999999999999999) →
  efficiency.city = 10 := by
  sorry

#check city_fuel_efficiency

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_fuel_efficiency_l272_27262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l272_27298

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = -14)
  (h_a5 : a 5 = -5) :
  (∀ n : ℕ, a n = 3 * n - 20) ∧
  (∃ n : ℕ, ∀ k : ℕ, k > 0 → (Finset.range k).sum a ≥ (Finset.range n).sum a) ∧
  ((Finset.range 6).sum a = -57) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l272_27298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cube_roots_l272_27273

/-- Given positive integers p, q, and r satisfying the equation
    2√(³√7 - ³√3) = ³√p + ³√q - ³√r, prove that p + q + r = 63 -/
theorem sum_of_cube_roots (p q r : ℕ+) 
  (h : 2 * Real.sqrt (Real.rpow 7 (1/3) - Real.rpow 3 (1/3)) = 
       Real.rpow p.val (1/3) + Real.rpow q.val (1/3) - Real.rpow r.val (1/3)) : 
  p.val + q.val + r.val = 63 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cube_roots_l272_27273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_symmetric_l272_27281

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 2 - x)

theorem f_is_even_and_symmetric : 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x : ℝ, f x = f (-x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_symmetric_l272_27281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_function_properties_l272_27246

/-- A sinusoidal function with given properties -/
noncomputable def SinFunction (A ω φ : ℝ) : ℝ → ℝ := λ x ↦ A * Real.sin (ω * x + φ)

theorem sin_function_properties (A ω φ : ℝ) (h_A : A > 0) (h_ω : ω > 0) :
  (∀ x ∈ Set.Ioo 0 (7 * Real.pi), SinFunction A ω φ x ≤ 3) →
  (∀ x ∈ Set.Ioo 0 (7 * Real.pi), SinFunction A ω φ x ≥ -3) →
  (SinFunction A ω φ Real.pi = 3) →
  (SinFunction A ω φ (6 * Real.pi) = -3) →
  (A = 3 ∧ ω = 1/5 ∧ φ = 3 * Real.pi / 10) ∧
  (∀ m : ℝ, (1/2 < m ∧ m ≤ 2) ↔ 
    A * Real.sin (ω * Real.sqrt (-m^2 + 2*m + 3) + φ) > 
    A * Real.sin (ω * Real.sqrt (-m^2 + 4) + φ)) :=
by sorry

#check sin_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_function_properties_l272_27246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_distance_l272_27223

/-- Represents the smartphone battery capacity in arbitrary units -/
noncomputable def BatteryCapacity : ℝ := 1

/-- Represents the discharge rate for video playback in units per hour -/
noncomputable def VideoDischargeRate : ℝ := 1 / 3

/-- Represents the discharge rate for playing Tetris in units per hour -/
noncomputable def TetrisDischargeRate : ℝ := 1 / 5

/-- Represents the speed of the train for the first half of the journey in km/h -/
noncomputable def Speed1 : ℝ := 80

/-- Represents the speed of the train for the second half of the journey in km/h -/
noncomputable def Speed2 : ℝ := 60

/-- Theorem stating that the distance traveled is approximately 257 km -/
theorem train_journey_distance : 
  ∃ (t : ℝ), 
    (VideoDischargeRate * (t / 2) + TetrisDischargeRate * (t / 2) = BatteryCapacity) ∧ 
    (∃ (d : ℝ), d / (2 * Speed1) + d / (2 * Speed2) = t ∧ 
      (Int.floor d = 257 ∨ Int.ceil d = 257)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_distance_l272_27223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fans_with_all_items_l272_27285

theorem fans_with_all_items (capacity : ℕ) (tshirt_interval : ℕ) (cap_interval : ℕ) (socks_interval : ℕ) :
  capacity = 5000 ∧ tshirt_interval = 100 ∧ cap_interval = 45 ∧ socks_interval = 60 →
  (Finset.filter (λ n : ℕ ↦ n ≤ capacity ∧ n % tshirt_interval = 0 ∧ n % cap_interval = 0 ∧ n % socks_interval = 0) (Finset.range (capacity + 1))).card = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fans_with_all_items_l272_27285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_equivalence_l272_27278

theorem proposition_equivalence (p q : Prop) : 
  (p ↔ ∀ x : ℝ, (1/10 : ℝ)^x > 0) → 
  ((¬p ∧ q) ↔ False) → 
  (q ↔ (Real.log 2 + Real.log 3 = Real.log 5)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_equivalence_l272_27278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_sixty_degree_triangle_l272_27238

/-- A triangle ABC is isosceles if at least two of its sides are equal -/
def IsIsosceles (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A B = dist A C ∨ dist A B = dist B C ∨ dist A C = dist B C

/-- The measure of an angle in degrees -/
noncomputable def AngleMeasure (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

theorem isosceles_sixty_degree_triangle
  (A B C : EuclideanSpace ℝ (Fin 2)) (h_isosceles : IsIsosceles A B C) (h_angle_ABC : AngleMeasure A B C = 60) :
  AngleMeasure A C B = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_sixty_degree_triangle_l272_27238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_solution_nested_sqrt_solution_exists_l272_27209

/-- The number of square roots in the equation -/
def n : ℕ := 1998

/-- Recursive function defining the nested square root expression -/
noncomputable def f (x : ℝ) : ℕ → ℝ
  | 0 => x
  | m + 1 => Real.sqrt (x + f x m)

/-- The theorem stating that the only integer solution to the equation is (0, 0) -/
theorem nested_sqrt_solution (x y : ℤ) : f (x : ℝ) n = y → x = 0 ∧ y = 0 := by
  sorry

/-- Existence of the solution -/
theorem nested_sqrt_solution_exists : ∃ x y : ℤ, f (x : ℝ) n = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_solution_nested_sqrt_solution_exists_l272_27209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_problem_l272_27216

theorem circle_tangency_problem :
  let X := 120
  let is_valid_radius (r : ℕ) := Nat.Prime r ∧ r < X ∧ X % r = 0
  (∃! (n : ℕ), n = (Finset.filter is_valid_radius (Finset.range X)).card) ∧
  (∀ (n : ℕ), n = (Finset.filter is_valid_radius (Finset.range X)).card → n = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_problem_l272_27216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_solution_l272_27264

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def frac (x : ℝ) : ℝ := x - ↑(floor x)

theorem absolute_difference_of_solution (x y : ℝ) : 
  (↑(floor x) + frac y = 3.6) → 
  (frac x + ↑(floor y) = 4.5) → 
  |x - y| = 1.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_solution_l272_27264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_is_10_l272_27224

/-- The coefficient of x^4 in the expansion of x(1+x)(1+x^2)^10 -/
def coefficient_x4 : ℕ := 10

/-- The polynomial x(1+x)(1+x^2)^10 -/
def polynomial (x : ℝ) : ℝ := x * (1 + x) * (1 + x^2)^10

/-- The coefficient of x^n in the expansion of polynomial p -/
noncomputable def coeff (p : ℝ → ℝ) (n : ℕ) : ℝ :=
  (deriv^[n] p 0) / n.factorial

theorem coefficient_x4_is_10 : 
  coeff polynomial 4 = coefficient_x4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_is_10_l272_27224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l272_27294

noncomputable def projection (v : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * u.1 + v.2 * u.2
  let norm_squared := u.1 * u.1 + u.2 * u.2
  (dot_product / norm_squared * u.1, dot_product / norm_squared * u.2)

def is_projection (p : (ℝ × ℝ) → (ℝ × ℝ)) : Prop :=
  ∃ u : ℝ × ℝ, ∀ v : ℝ × ℝ, p v = projection v u

theorem projection_problem :
  ∀ p : (ℝ × ℝ) → (ℝ × ℝ),
  is_projection p →
  p (3, -1) = (6, -2) →
  p (2, -6) = (3.6, -1.2) := by
  sorry

#check projection_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l272_27294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_changes_orientation_l272_27235

/-- A triangle in a 2D plane --/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- The orientation of a triangle --/
def orientation (t : Triangle) : ℝ :=
  (t.b.1 - t.a.1) * (t.c.2 - t.a.2) - (t.c.1 - t.a.1) * (t.b.2 - t.a.2)

/-- A reflection of a triangle --/
def reflect (t : Triangle) (line : (ℝ × ℝ) → (ℝ × ℝ)) : Triangle :=
  Triangle.mk
    (line t.a)
    (line t.b)
    (line t.c)

/-- Theorem: A triangle and its reflection cannot have the same orientation --/
theorem reflection_changes_orientation (t : Triangle) (line : (ℝ × ℝ) → (ℝ × ℝ)) :
  orientation t ≠ orientation (reflect t line) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_changes_orientation_l272_27235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_1_part_2_l272_27266

-- Define the parabola C: y² = 4x
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l: x = -1
def l (x : ℝ) : Prop := x = -1

-- Define the distance function
noncomputable def dist (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Define the point Q
noncomputable def Q : ℝ × ℝ := (1/2, Real.sqrt 2)

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Theorem for part 1
theorem part_1 :
  C Q.1 Q.2 ∧ 
  dist Q.1 Q.2 (-1) 0 = dist Q.1 Q.2 0 0 :=
by sorry

-- Define a tangent line to C
def tangent_line (k t : ℝ) (x y : ℝ) : Prop :=
  y - t = k * (x + 1) ∧ k^2 + k*t - 1 = 0

-- Theorem for part 2
theorem part_2 (P : ℝ × ℝ) (k₁ k₂ t : ℝ) :
  l P.1 →
  tangent_line k₁ t P.1 P.2 →
  tangent_line k₂ t P.1 P.2 →
  k₁ ≠ k₂ →
  ∃ (A B : ℝ × ℝ),
    C A.1 A.2 ∧ C B.1 B.2 ∧
    tangent_line k₁ t A.1 A.2 ∧
    tangent_line k₂ t B.1 B.2 ∧
    (B.2 - A.2) * (F.1 - A.1) = (F.2 - A.2) * (B.1 - A.1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_1_part_2_l272_27266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_x_minus_y_l272_27203

theorem closest_integer_to_x_minus_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : |x| + y = 3) (h2 : |x| * y + x^3 = 0) :
  ∃ (n : ℤ), n = -3 ∧ ∀ (m : ℤ), |↑m - (x - y)| ≥ |↑n - (x - y)| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_x_minus_y_l272_27203
