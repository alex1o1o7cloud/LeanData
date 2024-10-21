import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_proof_l178_17828

def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ := !![7/25, -24/25; -24/25, -7/25]

def direction_vector : Fin 2 → ℚ := ![4, -3]

theorem direction_vector_proof :
  -- The direction vector is an eigenvector of the reflection matrix
  reflection_matrix.mulVec direction_vector = direction_vector ∧
  -- The first component is positive
  direction_vector 0 > 0 ∧
  -- The components are integers
  (∃ (a b : ℤ), direction_vector 0 = a ∧ direction_vector 1 = b) ∧
  -- The GCD of the absolute values of the components is 1
  Int.gcd (Int.natAbs (Int.floor (direction_vector 0)))
          (Int.natAbs (Int.floor (direction_vector 1))) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_proof_l178_17828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_circumcircle_diameter_l178_17829

/-- The diameter of the circumscribed circle of a right triangle with sides 16 and 12 -/
theorem right_triangle_circumcircle_diameter :
  ∃ (a b c d : ℝ),
    a = 16 ∧ b = 12 ∧
    c^2 = a^2 + b^2 ∧
    (d = a ∨ d = c) ∧
    d = 16 ∨ d = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_circumcircle_diameter_l178_17829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_odd_functions_f_is_odd_h_is_odd_g_is_not_odd_l178_17819

-- Define the three functions
noncomputable def f (x : ℝ) : ℝ := 1 / x
noncomputable def g (x : ℝ) : ℝ := 10^(Real.log x)
noncomputable def h (x : ℝ) : ℝ := -x^3

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem exactly_two_odd_functions :
  (is_odd f ∧ is_odd h ∧ ¬is_odd g) :=
by
  sorry

-- Prove that f is odd
theorem f_is_odd : is_odd f :=
by
  sorry

-- Prove that h is odd
theorem h_is_odd : is_odd h :=
by
  sorry

-- Prove that g is not odd
theorem g_is_not_odd : ¬is_odd g :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_odd_functions_f_is_odd_h_is_odd_g_is_not_odd_l178_17819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_abs_is_even_and_increasing_l178_17800

-- Define the functions
def f (x : ℝ) := x^3
noncomputable def g (x : ℝ) := Real.log (abs x)
noncomputable def h (x : ℝ) := Real.cos x

-- Define evenness
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define monotonically increasing on (0, +∞)
def is_increasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

theorem ln_abs_is_even_and_increasing :
  is_even g ∧ is_increasing_on_positive g ∧
  ¬(is_even f ∧ is_increasing_on_positive f) ∧
  ¬(is_even h ∧ is_increasing_on_positive h) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_abs_is_even_and_increasing_l178_17800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l178_17858

noncomputable def f (x : ℝ) : ℝ := x / (1 + abs x)

theorem f_properties :
  (∀ x : ℝ, f (-x) + f x = 0) ∧
  (Set.range f = Set.Ioo (-1) 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ ≠ f x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l178_17858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_range_l178_17881

theorem real_part_range (z : ℂ) (h : Complex.abs (z - 1) = 1) :
  ∃ x : ℝ, x = (1 / (z - 1)).re ∧ -1 ≤ x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_range_l178_17881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_guaranteed_boxes_l178_17890

/-- Represents the game state -/
structure GameState where
  totalBalls : ℕ
  totalBoxes : ℕ
  ballDistribution : List ℕ
  deriving Repr

/-- Represents a player's move -/
structure Move where
  boxIndex : ℕ
  deriving Repr

/-- Represents the game rules and logic -/
class Game where
  initialState : GameState
  isValidMove : GameState → Move → Bool
  applyMove : GameState → Move → GameState
  isGameOver : GameState → Bool
  scoreFirstPlayer : GameState → ℕ

/-- Helper function to simulate playing the game -/
def playGame (game : Game) (strategy1 strategy2 : GameState → Move) : GameState :=
  sorry

/-- Theorem stating the maximum guaranteed boxes for the first player -/
theorem max_guaranteed_boxes
  (game : Game)
  (h1 : game.initialState.totalBalls = 1000)
  (h2 : game.initialState.totalBoxes = 30)
  (h3 : game.initialState.ballDistribution.length = 30)
  (h4 : game.initialState.ballDistribution.sum = 1000) :
  ∃ (strategy : GameState → Move),
    ∀ (opponentStrategy : GameState → Move),
      let finalState := playGame game strategy opponentStrategy
      game.scoreFirstPlayer finalState ≥ 15 ∧
      ¬∃ (betterStrategy : GameState → Move),
        ∀ (opponentStrategy : GameState → Move),
          let betterFinalState := playGame game betterStrategy opponentStrategy
          game.scoreFirstPlayer betterFinalState > 15 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_guaranteed_boxes_l178_17890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_equation_m_value_l178_17810

-- Define the circle C
def circle_eq (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + x - 6*y + m = 0

-- Define the line L
def line_eq (x y : ℝ) : Prop :=
  x + 2*y - 3 = 0

-- Define the dot product of two 2D vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ :=
  x1 * x2 + y1 * y2

-- Part 1: Reflected ray equation
theorem reflected_ray_equation :
  ∃ (k : ℝ), k = (-4 - Real.sqrt 7) / 3 ∧
  ∀ (x y : ℝ),
    circle_eq (1/4) x y →
    (y + 3 = k * (x - 11/2) → 
      ∃ (t : ℝ), x = 11/2 * (1 - t) + x * t ∧
                 y = -3 * (1 - t) + y * t ∧
                 0 ≤ t ∧ t ≤ 1) :=
sorry

-- Part 2: Value of m
theorem m_value :
  ∀ (m : ℝ),
    (∃ (x1 y1 x2 y2 : ℝ),
      circle_eq m x1 y1 ∧ circle_eq m x2 y2 ∧
      line_eq x1 y1 ∧ line_eq x2 y2 ∧
      dot_product x1 y1 x2 y2 = 0) →
    m = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_equation_m_value_l178_17810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_identities_l178_17873

theorem triangle_identities
  (α β γ r r_a p a b R : ℝ)
  (h1 : r * Real.cos (α/2) * Real.sin (α/2) = p - a)
  (h2 : Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2) = r/(4*R))
  (h3 : r_a * Real.tan (γ/2) = p - b)
  (h4 : p - b = r * (Real.cos (β/2) / Real.sin (β/2))) :
  (Real.cos (α/2) * Real.sin (β/2) * Real.sin (γ/2) = (p-a)/(4*R)) ∧
  (Real.sin (α/2) * Real.cos (β/2) * Real.cos (γ/2) = r_a/(4*R)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_identities_l178_17873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l178_17893

/-- The area of a triangle with vertices at (2, 1), (8, -3), and (2, 7) is 18 square units. -/
theorem triangle_area : ∃ (area : ℝ), area = 18 := by
  -- Define the vertices of the triangle
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (8, -3)
  let C : ℝ × ℝ := (2, 7)

  -- Calculate the area using the formula: |1/2 * ((x2-x1)(y3-y1) - (x3-x1)(y2-y1))|
  let area := (1/2 : ℝ) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

  -- Prove that this area exists and is equal to 18
  use area
  sorry  -- Skip the actual proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l178_17893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rounds_is_three_l178_17892

/-- Represents the data for a group of golfers and their rounds played. -/
structure GolferGroup where
  rounds : Nat
  golfers : Nat

/-- Calculates the average number of rounds played by all golfers. -/
def averageRounds (groups : List GolferGroup) : Rat :=
  let totalRounds := (groups.map (λ g => g.rounds * g.golfers)).sum
  let totalGolfers := (groups.map (λ g => g.golfers)).sum
  totalRounds / totalGolfers

/-- Rounds a rational number to the nearest whole number. -/
def roundToNearest (x : Rat) : Int :=
  (x + 1/2).floor

/-- The main theorem about the average number of rounds played. -/
theorem average_rounds_is_three (groups : List GolferGroup) 
  (h1 : groups = [
    { rounds := 1, golfers := 3 },
    { rounds := 2, golfers := 5 },
    { rounds := 3, golfers := 7 },
    { rounds := 6, golfers := 2 },
    { rounds := 8, golfers := 1 }
  ]) : roundToNearest (averageRounds groups) = 3 := by
  sorry

#eval roundToNearest (averageRounds [
  { rounds := 1, golfers := 3 },
  { rounds := 2, golfers := 5 },
  { rounds := 3, golfers := 7 },
  { rounds := 6, golfers := 2 },
  { rounds := 8, golfers := 1 }
])

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rounds_is_three_l178_17892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l178_17864

theorem triangle_angle_theorem (α β γ : Real) (h_triangle : α + β + γ = Real.pi) 
  (h_equation : Real.sin α + Real.sin β = (Real.cos α + Real.cos β) * Real.sin γ) : γ = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l178_17864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_height_variance_l178_17859

/-- Calculates the variance of heights for a class of students given the statistics for male and female subgroups. -/
theorem class_height_variance
  (total_students : ℕ)
  (male_students : ℕ)
  (female_students : ℕ)
  (male_avg_height : ℝ)
  (female_avg_height : ℝ)
  (male_variance : ℝ)
  (female_variance : ℝ)
  (h_total : total_students = 40)
  (h_male : male_students = 22)
  (h_female : female_students = 18)
  (h_total_sum : total_students = male_students + female_students)
  (h_male_avg : male_avg_height = 173)
  (h_female_avg : female_avg_height = 163)
  (h_male_var : male_variance = 28)
  (h_female_var : female_variance = 32)
  : ∃ (class_variance : ℝ), abs (class_variance - 54.5875) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_height_variance_l178_17859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_climb_eight_stairs_l178_17850

/-- The number of ways to climb n stairs, where each step can be 1, 2, 4, or 5 stairs at a time. -/
def climbStairs : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => climbStairs (n + 2) + climbStairs (n + 1) + climbStairs (n - 1) + 
             if n ≥ 2 then climbStairs (n - 2) else 0

/-- Theorem: The number of ways to climb 8 stairs, where each step can be 1, 2, 4, or 5 stairs at a time, is equal to 52. -/
theorem climb_eight_stairs : climbStairs 8 = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_climb_eight_stairs_l178_17850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l178_17849

/-- Variable cost function -/
noncomputable def W (x : ℝ) : ℝ :=
  if x < 8 then (1/3) * x^2 + x
  else 6 * x + 100 / x - 38

/-- Annual profit function -/
noncomputable def L (x : ℝ) : ℝ := 5 * x - W x - 3

/-- Theorem stating the maximum profit and corresponding production quantity -/
theorem max_profit :
  ∃ (max_x : ℝ), max_x = 6 ∧
  ∀ (x : ℝ), x > 0 → L x ≤ L max_x ∧ L max_x = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l178_17849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z2_value_l178_17811

noncomputable def complex_midpoint (z₁ z₂ : ℂ) : ℂ := (z₁ + z₂) / 2

theorem z2_value (z₁ z₂ : ℂ) (h₁ : z₁ = 4 + I) (h₂ : complex_midpoint z₁ z₂ = 1 + 2*I) : 
  z₂ = -2 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z2_value_l178_17811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_normal_intersection_l178_17802

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + x

/-- The derivative of the parabola function -/
def f' (x : ℝ) : ℝ := 2*x + 1

/-- Point A on the parabola -/
def A : ℝ × ℝ := (1, 2)

/-- The slope of the normal line at A -/
noncomputable def m : ℝ := -1 / (f' A.1)

/-- The normal line function passing through A -/
noncomputable def g (x : ℝ) : ℝ := m * (x - A.1) + A.2

/-- The x-coordinate of point B -/
noncomputable def B_x : ℝ := (-4 - 4 * Real.sqrt 7) / 6

/-- The y-coordinate of point B -/
noncomputable def B_y : ℝ := (B_x^2 + B_x)

/-- Point B -/
noncomputable def B : ℝ × ℝ := (B_x, B_y)

theorem parabola_normal_intersection :
  f B.1 = B.2 ∧ g B.1 = B.2 ∧ B ≠ A := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_normal_intersection_l178_17802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_cos_plus_ax_implies_a_ge_one_l178_17831

/-- If the function f(x) = cos x + ax is increasing on [-π/2, π/2], then a ≥ 1 -/
theorem increasing_cos_plus_ax_implies_a_ge_one (a : ℝ) :
  (∀ x ∈ Set.Icc (-Real.pi/2) (Real.pi/2), 
    ∀ y ∈ Set.Icc (-Real.pi/2) (Real.pi/2), 
    x < y → (Real.cos x + a * x) < (Real.cos y + a * y)) →
  a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_cos_plus_ax_implies_a_ge_one_l178_17831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l178_17820

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * x - 2 * (1 + Real.log x) + a
noncomputable def g (x : ℝ) : ℝ := Real.exp 1 * x / Real.exp x

-- Theorem for the tangent line
theorem tangent_line_at_one :
  let f₁ : ℝ → ℝ := λ x => x - 2 * (1 + Real.log x) + 1
  ∃ m b : ℝ, m = -1 ∧ b = 1 ∧
    ∀ x : ℝ, (deriv f₁ 1) * (x - 1) + f₁ 1 = m * x + b := by
  sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ a : ℝ,
    (∀ x₀ : ℝ, 0 < x₀ ∧ x₀ ≤ Real.exp 1 →
      ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ (Real.exp 1)^2 ∧
        f a x₁ = g x₀ ∧ f a x₂ = g x₀) ↔
    a ≤ 2 - 5 / ((Real.exp 1)^2 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l178_17820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athul_downstream_distance_l178_17861

/-- The distance Athul rows upstream in kilometers -/
def upstream_distance : ℚ := 16

/-- The time Athul spends rowing upstream in hours -/
def upstream_time : ℚ := 4

/-- The time Athul spends rowing downstream in hours -/
def downstream_time : ℚ := 4

/-- The speed of the stream in km/h -/
def stream_speed : ℚ := 1

/-- Athul's speed in still water in km/h -/
noncomputable def still_water_speed : ℚ := (upstream_distance / upstream_time) + stream_speed

/-- The distance Athul rows downstream in kilometers -/
noncomputable def downstream_distance : ℚ := (still_water_speed + stream_speed) * downstream_time

theorem athul_downstream_distance :
  downstream_distance = 24 := by
  -- Unfold definitions
  unfold downstream_distance
  unfold still_water_speed
  -- Simplify the expression
  simp [upstream_distance, upstream_time, downstream_time, stream_speed]
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_athul_downstream_distance_l178_17861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_endpoint_l178_17853

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line segment in 2D space -/
structure LineSegment where
  start : Point2D
  finish : Point2D

/-- Check if a line segment is parallel to the x-axis -/
def isParallelToXAxis (seg : LineSegment) : Prop :=
  seg.start.y = seg.finish.y

/-- Calculate the length of a line segment -/
noncomputable def length (seg : LineSegment) : ℝ :=
  Real.sqrt ((seg.finish.x - seg.start.x)^2 + (seg.finish.y - seg.start.y)^2)

theorem line_segment_endpoint (A : Point2D) (h1 : A.x = 2 ∧ A.y = 1) :
  ∃ B : Point2D, ∃ seg : LineSegment,
    seg.start = A ∧
    seg.finish = B ∧
    isParallelToXAxis seg ∧
    length seg = 5 ∧
    ((B.x = -3 ∧ B.y = 1) ∨ (B.x = 7 ∧ B.y = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_endpoint_l178_17853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ab_is_10km_l178_17843

/-- Represents the cat's journey between points A and B -/
structure CatJourney where
  uphill_speed : ℚ
  downhill_speed : ℚ
  time_ab : ℚ
  time_ba : ℚ

/-- Calculates the total distance of the journey -/
def total_distance (j : CatJourney) : ℚ :=
  let uphill_distance := j.uphill_speed * (j.time_ab / 2)
  let downhill_distance := j.downhill_speed * (j.time_ab / 2)
  uphill_distance + downhill_distance

/-- Theorem stating that the distance from A to B is 10 km -/
theorem distance_ab_is_10km (j : CatJourney) 
  (h1 : j.uphill_speed = 4)
  (h2 : j.downhill_speed = 5)
  (h3 : j.time_ab = 11/5)  -- 2 hours and 12 minutes as a fraction
  (h4 : j.time_ba = j.time_ab + 1/10)  -- 6 minutes longer as a fraction
  : total_distance j = 10 := by
  sorry

#eval total_distance { uphill_speed := 4, downhill_speed := 5, time_ab := 11/5, time_ba := 23/10 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ab_is_10km_l178_17843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_telegraph_post_l178_17821

/-- Calculates the time (in seconds) for a train to pass a stationary point -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  train_length / train_speed_ms

theorem train_passing_telegraph_post :
  train_passing_time 60 54 = 4 := by
  -- Unfold the definition of train_passing_time
  unfold train_passing_time
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_telegraph_post_l178_17821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_iff_a_ge_one_l178_17872

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  1/2 * (Real.cos x - Real.sin x) * (Real.cos x + Real.sin x) + 3*a*(Real.sin x - Real.cos x) + (4*a - 1)*x

-- Define the interval
def I : Set ℝ := Set.Icc (-Real.pi/2) 0

-- State the theorem
theorem f_strictly_increasing_iff_a_ge_one :
  ∀ a : ℝ, (∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f a x < f a y) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_iff_a_ge_one_l178_17872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_rationalize_l178_17863

theorem simplify_and_rationalize :
  1 / (1 + 1 / (Real.sqrt 5 + 2)) = (Real.sqrt 5 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_rationalize_l178_17863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joes_hvac_cost_per_vent_l178_17896

/-- Represents an HVAC system -/
structure HVAC where
  total_cost : ℚ
  num_zones : ℕ
  vents_per_zone : ℕ

/-- Calculates the cost per vent for an HVAC system -/
def cost_per_vent (system : HVAC) : ℚ :=
  system.total_cost / (system.num_zones * system.vents_per_zone)

/-- Theorem: The cost per vent for Joe's HVAC system is $2000 -/
theorem joes_hvac_cost_per_vent :
  let joe_system : HVAC := ⟨20000, 2, 5⟩
  cost_per_vent joe_system = 2000 := by
  sorry

#eval cost_per_vent ⟨20000, 2, 5⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joes_hvac_cost_per_vent_l178_17896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_population_growth_l178_17840

/-- The final population of bacteria after exponential growth -/
noncomputable def final_population (initial_population : ℕ) (growth_rate : ℝ) (time : ℝ) : ℝ :=
  (initial_population : ℝ) * growth_rate ^ time

/-- Theorem stating the approximate final population of bacteria -/
theorem bacteria_population_growth :
  let initial_population := 1000
  let growth_rate := (2 : ℝ)
  let time := 8.965784284662087
  ⌊final_population initial_population growth_rate time⌋ = 495033 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_population_growth_l178_17840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percentage_is_negative_19_33_percent_l178_17817

noncomputable section

def original_price : ℝ := 1500
def discount_rate : ℝ := 0.05
def sales_tax_rate : ℝ := 0.10
def shipping_fee : ℝ := 300
def selling_price : ℝ := 1620
def commission_rate : ℝ := 0.05
def donation_rate : ℝ := 0.02

def discounted_price : ℝ := original_price * (1 - discount_rate)
def price_with_tax : ℝ := discounted_price * (1 + sales_tax_rate)
def total_cost_price : ℝ := price_with_tax + shipping_fee

def commission : ℝ := selling_price * commission_rate
def donation : ℝ := selling_price * donation_rate
def actual_selling_price : ℝ := selling_price - commission - donation

def gain : ℝ := actual_selling_price - total_cost_price
def gain_percentage : ℝ := gain / total_cost_price * 100

theorem gain_percentage_is_negative_19_33_percent :
  ∃ ε > 0, |gain_percentage + 19.33| < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percentage_is_negative_19_33_percent_l178_17817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_collinear_l178_17842

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define point L on BC that bisects angle BAC
variable (L : ℝ × ℝ)

-- Define R on AC and S on AB
variable (R S : ℝ × ℝ)

-- Define M on LR such that BM is perpendicular to AL
variable (M : ℝ × ℝ)

-- Define D as the midpoint of BC
noncomputable def D (A B C : ℝ × ℝ) : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the conditions
variable (h1 : L.1 = B.1 + (C.1 - B.1) * (A.2 / (A.2 + A.1)))
variable (h2 : L.2 = B.2 + (C.2 - B.2) * (A.2 / (A.2 + A.1)))
variable (h3 : R.2 - A.2 = (C.2 - A.2) * ((R.1 - A.1) / (C.1 - A.1)))
variable (h4 : S.1 - A.1 = (B.1 - A.1) * ((S.2 - A.2) / (B.2 - A.2)))
variable (h5 : (M.1 - B.1) * (L.1 - A.1) + (M.2 - B.2) * (L.2 - A.2) = 0)
variable (h6 : M.2 - L.2 = (R.2 - L.2) * ((M.1 - L.1) / (R.1 - L.1)))

-- Theorem statement
theorem points_collinear (A B C L R S M : ℝ × ℝ) (h1 h2 h3 h4 h5 h6 : Prop) :
  ∃ (k : ℝ), M = k • ((D A B C) - A) + A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_collinear_l178_17842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_line_max_distance_l178_17862

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem ellipse_tangent_line_max_distance (Γ : Ellipse) (E : Point) (O : Ellipse) :
  Γ.a = 2 ∧ Γ.b = 1 →
  E.x = Real.sqrt 3 ∧ E.y = 1/2 →
  (E.x^2 / Γ.a^2 + E.y^2 / Γ.b^2 = 1) →
  (Real.sqrt (Γ.a^2 - Γ.b^2) / Γ.a = Real.sqrt 3 / 2) →
  O.a = 1 ∧ O.b = 1 →
  ∀ (l : Line) (M A B : Point),
    (M.x^2 + M.y^2 = O.b^2) →
    (A.x^2 / Γ.a^2 + A.y^2 / Γ.b^2 = 1) →
    (B.x^2 / Γ.a^2 + B.y^2 / Γ.b^2 = 1) →
    (A.y = l.slope * A.x + l.intercept) →
    (B.y = l.slope * B.x + l.intercept) →
    (M.y = l.slope * M.x + l.intercept) →
    distance A B ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_line_max_distance_l178_17862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basket_game_theorem_l178_17833

/-- A two-player basket shooting game -/
structure BasketGame where
  player_a_percentage : ℝ
  player_b_percentage : ℝ
  first_player_probability : ℝ

/-- The probability that the second player takes the second shot -/
noncomputable def second_shot_probability (game : BasketGame) : ℝ :=
  game.first_player_probability * (1 - game.player_a_percentage) +
  (1 - game.first_player_probability) * game.player_b_percentage

/-- The probability that the first player takes the i-th shot -/
noncomputable def ith_shot_probability (game : BasketGame) (i : ℕ) : ℝ :=
  1/3 + 1/6 * (2/5)^(i-1)

/-- The expected number of times the first player shoots in the first n shots -/
noncomputable def expected_shots (game : BasketGame) (n : ℕ) : ℝ :=
  5/18 * (1 - (2/5)^n) + n/3

/-- Main theorem for the basket shooting game -/
theorem basket_game_theorem (game : BasketGame) :
  game.player_a_percentage = 0.6 →
  game.player_b_percentage = 0.8 →
  game.first_player_probability = 0.5 →
  (second_shot_probability game = 0.6) ∧
  (∀ i : ℕ, i > 0 → ith_shot_probability game i = 1/3 + 1/6 * (2/5)^(i-1)) ∧
  (∀ n : ℕ, n > 0 → expected_shots game n = 5/18 * (1 - (2/5)^n) + n/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basket_game_theorem_l178_17833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_divisibility_l178_17808

theorem odd_prime_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  (∃ n : ℤ, (p : ℤ) ∣ n * (n + 1) * (n + 2) * (n + 3) + 1) ↔
  (∃ m : ℤ, (p : ℤ) ∣ m^2 - 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_divisibility_l178_17808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l178_17869

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 6 → ∃ (length : ℝ), abs (length - 100.02) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l178_17869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_m_bounds_l178_17889

noncomputable def f (x : ℝ) : ℝ := 2 + Real.log (x^2)

noncomputable def g (x : ℝ) : ℝ := (x * f x) / (x - 2)

theorem f_m_bounds (h1 : ∀ x : ℝ, f x ≤ 2 * |x|) 
  (h2 : ∃ m : ℝ, m > 2 ∧ ∀ x > 2, g m ≤ g x) : 
  ∃ m : ℝ, 6 < f m ∧ f m < 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_m_bounds_l178_17889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_rate_of_change_l178_17883

noncomputable section

/-- The volume function of the solution in milliliters -/
def V (t : ℝ) : ℝ := Real.pi * t^3 + 2 * Real.pi * t^2

/-- The radius of the cup's bottom in centimeters -/
def r : ℝ := 4

/-- The area of the cup's bottom in square centimeters -/
def S : ℝ := Real.pi * r^2

/-- The height function of the solution in centimeters -/
def h (t : ℝ) : ℝ := V t / S

/-- The derivative of the height function -/
def h' (t : ℝ) : ℝ := (3 * t^2 + 4 * t) / 16

theorem instantaneous_rate_of_change : h' 4 = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_rate_of_change_l178_17883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_eq_imag_implies_a_eq_neg_six_l178_17891

-- Define the complex number as a function of a
noncomputable def z (a : ℝ) : ℂ := (2 + a * Complex.I) / (1 + 2 * Complex.I)

-- State the theorem
theorem real_eq_imag_implies_a_eq_neg_six :
  ∀ a : ℝ, Complex.re (z a) = Complex.im (z a) → a = -6 := by
  intro a h
  sorry

#check real_eq_imag_implies_a_eq_neg_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_eq_imag_implies_a_eq_neg_six_l178_17891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_replace_tile_l178_17854

/-- Represents a tile type -/
inductive TileType
| Square  : TileType  -- 2x2 tile
| Rectangle : TileType  -- 1x4 tile

/-- Represents a color in the 4-coloring scheme -/
inductive Color
| One : Color
| Two : Color
| Three : Color
| Four : Color

/-- Represents a rectangular floor -/
structure Floor where
  width : ℕ
  height : ℕ
  tiles : List TileType

/-- Represents the coloring of a cell -/
def colorCell (x y : ℕ) : Color :=
  match (x % 2, y % 2) with
  | (0, 0) => Color.One
  | (1, 0) => Color.Two
  | (0, 1) => Color.Three
  | (1, 1) => Color.Four
  | _ => Color.One  -- Default case to handle all other possibilities

/-- Helper function to count the number of cells of a given color covered by a list of tiles -/
def count_color (tiles : List TileType) (c : Color) : ℕ :=
  sorry  -- Implementation details omitted for brevity

/-- Theorem: It's impossible to replace a broken tile with a tile of the other type -/
theorem cannot_replace_tile (f : Floor) : 
  ∀ (broken_tile replacement_tile : TileType), 
    broken_tile ≠ replacement_tile → 
    ¬ (∃ (new_tiles : List TileType), 
      new_tiles.length = f.tiles.length ∧ 
      (∀ c : Color, (count_color f.tiles c) % 2 = (count_color new_tiles c) % 2)) :=
by
  sorry  -- Proof details omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_replace_tile_l178_17854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_z_l178_17846

theorem min_value_z (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  ∃ (min_z : ℝ), min_z = Real.sqrt 2 - 1 ∧
    ∀ z', z' > 0 → (z' + 1)^2 / (x * y * z') ≥ (min_z + 1)^2 / (x * y * min_z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_z_l178_17846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_up_theorem_l178_17882

/-- The number of ways to arrange five people in a line with restrictions -/
def lineUpWays : ℕ := 72

/-- The total number of people -/
def totalPeople : ℕ := 5

/-- The number of positions the youngest person can occupy -/
def youngestPositions : ℕ := totalPeople - 2

/-- The number of ways to arrange the remaining people after placing the youngest -/
def remainingArrangements : ℕ := Nat.factorial (totalPeople - 1)

theorem line_up_theorem :
  lineUpWays = youngestPositions * remainingArrangements := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_up_theorem_l178_17882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_through_point_l178_17832

/-- Theorem: If the terminal side of angle α passes through the point (2, -1), then sin α = -√5/5 -/
theorem sine_of_angle_through_point (α : ℝ) :
  (∃ (t : ℝ), t > 0 ∧ t * Real.cos α = 2 ∧ t * Real.sin α = -1) →
  Real.sin α = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_through_point_l178_17832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_and_min_sum_squares_l178_17847

-- Define the functions f and g
noncomputable def f (x : ℝ) (θ : ℝ) : ℝ := |x + Real.sin θ^2|
noncomputable def g (x : ℝ) (θ : ℝ) : ℝ := 2 * |x - Real.cos θ^2|

-- Define the theorem
theorem max_a_and_min_sum_squares :
  (∃ m : ℝ,
    (∀ a : ℝ, (∀ x θ : ℝ, θ ∈ Set.Icc 0 (2 * Real.pi) → 2 * f x θ ≥ a - g x θ) → a ≤ m) ∧
    m = 2) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + 2*b + 3*c = 4 →
    a^2 + b^2 + c^2 ≥ 8/7) ∧
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + 2*b + 3*c = 4 ∧ a^2 + b^2 + c^2 = 8/7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_and_min_sum_squares_l178_17847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_function_theorem_l178_17814

def NatStar := {n : ℕ // n > 0}

theorem identity_function_theorem (f : NatStar → NatStar) 
  (h : ∀ (m n : NatStar), f ⟨m.val ^ 2 + (f n).val, sorry⟩ = ⟨(f m).val ^ 2 + n.val, sorry⟩) :
  ∀ (n : NatStar), f n = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_function_theorem_l178_17814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_minute_hand_angle_l178_17830

/-- The angle covered by the minute hand of a clock in radians during a given time period. -/
noncomputable def minuteHandAngle (minutes : ℝ) : ℝ := -2 * Real.pi * (minutes / 60)

/-- Theorem stating that the angle covered by the minute hand during a 45-minute class is -3π/2 radians. -/
theorem class_minute_hand_angle :
  minuteHandAngle 45 = -(3 / 2) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_minute_hand_angle_l178_17830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_items_count_l178_17827

theorem exam_items_count (lyssa_incorrect_percentage : ℚ) 
                         (precious_mistakes : ℕ) 
                         (lyssa_correct_difference : ℕ) 
                         (h1 : lyssa_incorrect_percentage = 20)
                         (h2 : precious_mistakes = 12)
                         (h3 : lyssa_correct_difference = 3) :
  ∃ (total_items : ℕ), 
    (total_items : ℚ) * (1 - lyssa_incorrect_percentage / 100) = 
    (total_items - precious_mistakes : ℚ) + lyssa_correct_difference ∧
    total_items = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_items_count_l178_17827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_can_ensure_last_two_from_same_box_l178_17815

/-- Represents a box of candies -/
structure Box where
  candies : ℕ

/-- Represents the game state -/
structure GameState where
  boxes : List Box
  turn : Bool  -- true for girl's turn, false for boy's turn

/-- A strategy for the boy to choose candies -/
def BoyStrategy := GameState → ℕ  -- Returns the index of the box to choose from

theorem boy_can_ensure_last_two_from_same_box :
  ∀ (n : ℕ), n > 0 →
  ∀ (initial_state : GameState),
  (initial_state.boxes.length = n) →
  (initial_state.boxes.map Box.candies).sum = 2 * n →
  initial_state.turn = true →
  ∃ (strategy : BoyStrategy),
  ∀ (final_state : GameState),
  (final_state.boxes.map Box.candies).sum = 2 →
  ∃ (i : ℕ), i < final_state.boxes.length ∧ final_state.boxes.get? i = some ⟨2⟩ :=
by
  intros n hn initial_state h_length h_sum h_turn
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_can_ensure_last_two_from_same_box_l178_17815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l178_17870

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2*x - x^2)

-- State the theorem
theorem monotonic_decreasing_interval_of_f :
  ∃ (a b : ℝ), a = 1 ∧ b = 2 ∧
  (∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x ≤ y → f y ≤ f x) ∧
  (∀ c d, c < a ∨ b < d →
    ¬(∀ x y, x ∈ Set.Icc c d → y ∈ Set.Icc c d → x ≤ y → f y ≤ f x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l178_17870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_point_l178_17855

-- Define the function f(x)
noncomputable def f (a b x : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2 + b

-- Theorem statement
theorem unique_zero_point (a b : ℝ) :
  ((1/2 < a ∧ a ≤ Real.exp 2 / 2 ∧ b > 2*a) ∨
   (0 < a ∧ a < 1/2 ∧ b ≤ 2*a)) →
  ∃! x, f a b x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_point_l178_17855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_covers_set_l178_17897

/-- A function satisfying the given conditions -/
def SpecialFunction (n : ℕ) (hodd : Odd n) :=
  { f : Fin n → Fin n → Fin n //
    (∀ r s, f r s = f s r) ∧
    (∀ r, Set.range (f r) = Set.univ) }

/-- The main theorem -/
theorem diagonal_covers_set {n : ℕ} (hodd : Odd n) (f : SpecialFunction n hodd) :
  Set.range (λ r : Fin n ↦ f.val r r) = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_covers_set_l178_17897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_recurrence_l178_17822

/-- Represents the number of distinct partitions for a collection of 2n letters -/
def u : ℕ → ℕ := sorry

/-- The recurrence relation for u_n -/
theorem u_recurrence (n : ℕ) :
  u (n + 1) = (n + 1) * u n - (n * (n - 1) / 2) * u (n - 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_recurrence_l178_17822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l178_17807

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

-- Define the interval [1, 3]
def I : Set ℝ := Set.Icc 1 3

-- Theorem statement
theorem f_properties :
  (∀ (x y : ℝ), x ∈ I → y ∈ I → x < y → f x < f y) ∧
  (∀ x, x ∈ I → f x ≥ 0) ∧
  (∀ x, x ∈ I → f x ≤ 1/2) ∧
  (f 1 = 0) ∧
  (f 3 = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l178_17807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l178_17845

theorem trigonometric_problem (α β : Real) 
  (h1 : π < α ∧ α < 3 * π / 2)
  (h2 : π < β ∧ β < 3 * π / 2)
  (h3 : Real.sin α = -Real.sqrt 5 / 5)
  (h4 : Real.cos β = -Real.sqrt 10 / 10) :
  (α - β = -π / 4) ∧ (Real.tan (2 * α - β) = -1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l178_17845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_greater_than_one_l178_17874

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x * Real.log x) / (x - 1) - a

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := (x^2 - x) * f a x

theorem root_sum_greater_than_one (a m : ℝ) (x₁ x₂ : ℝ) :
  a < 0 →
  x₁ ≠ x₂ →
  h a x₁ = m →
  h a x₂ = m →
  x₁ + x₂ > 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_greater_than_one_l178_17874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l178_17886

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x)) / Real.log 2

-- Theorem statement
theorem f_properties :
  -- 1. Domain of f is (-1, 1)
  (∀ x, -1 < x ∧ x < 1 ↔ (1 - x) / (1 + x) > 0) ∧
  -- 2. f is decreasing on its domain
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  -- 3. Solution to f(2x-1) < 0 is 1/2 < x < 1
  (∀ x, f (2*x - 1) < 0 ↔ 1/2 < x ∧ x < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l178_17886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_solution_dilution_l178_17898

-- Define the variables
variable (m n x y : ℝ)

-- Define the conditions
variable (h1 : m > 0)
variable (h2 : n > 0)
variable (h3 : m > 3*n)
variable (h4 : x > 0)
variable (h5 : y ≥ 0)

-- Define the initial and final acid amounts
noncomputable def initial_acid (m : ℝ) := m^2 / 100
noncomputable def final_acid (m n x y : ℝ) := (m - n) * (m + x + y) / 100

-- State the theorem
theorem acid_solution_dilution :
  initial_acid m = final_acid m n x y →
  x = n * m / (m + n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_solution_dilution_l178_17898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_probability_l178_17878

/-- An exponential distribution with parameter α -/
structure ExponentialDistribution (α : ℝ) where
  α_pos : α > 0

/-- The cumulative distribution function (CDF) for the exponential distribution -/
noncomputable def cdf (α : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 0 else 1 - Real.exp (-α * x)

/-- The probability that a random variable X with exponential distribution
    falls within the interval (a, b) -/
theorem exponential_probability (α : ℝ) (a b : ℝ) (h : ExponentialDistribution α) (hab : a < b) :
  (cdf α b - cdf α a) = Real.exp (-α * a) - Real.exp (-α * b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_probability_l178_17878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l178_17888

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem monotonic_increasing_interval :
  ∃ (a b : ℝ), a = -Real.pi/3 ∧ b = Real.pi/6 ∧
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l178_17888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_formula_l178_17856

/-- Sequence defined by the recurrence relation -/
def a : ℕ → ℤ
  | 0 => 5  -- Adding this case to cover Nat.zero
  | 1 => 5
  | (n + 2) => a (n + 1) + 6 * a n

/-- Theorem stating the general formula for the sequence -/
theorem a_general_formula (n : ℕ) : a n = 3^n - (-2)^n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_formula_l178_17856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hua_luogeng_optimal_selection_method_uses_golden_ratio_l178_17805

/-- The optimal selection method popularized by Hua Luogeng -/
structure OptimalSelectionMethod where
  concept : ℝ

/-- Hua Luogeng's optimal selection method uses the golden ratio -/
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  ∃ (method : OptimalSelectionMethod), method.concept = (1 + Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hua_luogeng_optimal_selection_method_uses_golden_ratio_l178_17805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equivalent_shape_l178_17806

/-- Represents a geometric shape with an area -/
structure Shape where
  area : ℝ

/-- Represents the diagram with two squares -/
structure TwoSquares where
  large_square : Shape
  small_square : Shape

/-- The shaded area in the original diagram -/
def shaded_area (ts : TwoSquares) : ℝ :=
  ts.large_square.area - ts.small_square.area

/-- A theorem stating that we can find a shape with the same area as the shaded region -/
theorem exists_equivalent_shape (ts : TwoSquares) :
    ∃ (s : Shape), s.area = shaded_area ts := by
  sorry  -- The actual construction would be done visually, not proved formally

#check exists_equivalent_shape

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equivalent_shape_l178_17806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_symmetry_parabola_intersection_l178_17868

-- Define the hyperbola
def on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the symmetry line
def symmetric_about_line (x₁ y₁ x₂ y₂ b : ℝ) : Prop :=
  y₁ = -x₁ + b ∧ y₂ = -x₂ + b ∧ x₁ + x₂ = y₁ + y₂

-- Define the midpoint
def midpoint_of (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

-- Define the parabola
def on_parabola (x y : ℝ) : Prop := y^2 = 8*x

theorem hyperbola_symmetry_parabola_intersection (x₁ y₁ x₂ y₂ x₀ y₀ b : ℝ) :
  on_hyperbola x₁ y₁ ∧
  on_hyperbola x₂ y₂ ∧
  symmetric_about_line x₁ y₁ x₂ y₂ b ∧
  midpoint_of x₁ y₁ x₂ y₂ x₀ y₀ ∧
  on_parabola x₀ y₀ →
  b = 0 ∨ b = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_symmetry_parabola_intersection_l178_17868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_value_l178_17834

def b : ℕ → ℚ
  | 0 => 2  -- Adding this case to cover Nat.zero
  | 1 => 2
  | 2 => 1
  | n+3 => (2 - b (n+2)) / (3 * b (n+1))

theorem b_100_value : b 100 = 101 / 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_value_l178_17834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_example_l178_17899

/-- The height of a trapezium given its parallel sides and area -/
noncomputable def trapezium_height (a b : ℝ) (area : ℝ) : ℝ :=
  (2 * area) / (a + b)

/-- Theorem: The height of a trapezium with parallel sides 20 cm and 18 cm, 
    and area 228 sq cm, is 12 cm -/
theorem trapezium_height_example : 
  trapezium_height 20 18 228 = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_example_l178_17899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_negative_max_value_is_e_not_monotone_increasing_two_zeros_condition_l178_17813

-- Define the function f(x) = (2-x)e^x
noncomputable def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

-- Statement 1: The slope of the tangent line at x=2 is negative
theorem tangent_slope_negative : 
  ∃ (δ : ℝ), δ > 0 ∧ ∀ (h : ℝ), 0 < |h| ∧ |h| < δ → (f (2 + h) - f 2) / h < 0 := by
  sorry

-- Statement 2: The maximum value of f(x) is e
theorem max_value_is_e : 
  ∃ (x : ℝ), f x = Real.exp 1 ∧ ∀ (y : ℝ), f y ≤ Real.exp 1 := by
  sorry

-- Statement 3: f(x) is not monotonically increasing on (1, +∞)
theorem not_monotone_increasing : 
  ¬ (∀ (x y : ℝ), 1 < x ∧ x < y → f x < f y) := by
  sorry

-- Statement 4: If f(x) = a has two zeros, then 0 < a < e
theorem two_zeros_condition : 
  ∀ (a : ℝ), (∃ (x y : ℝ), x ≠ y ∧ f x = a ∧ f y = a) → 0 < a ∧ a < Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_negative_max_value_is_e_not_monotone_increasing_two_zeros_condition_l178_17813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_waste_for_3x4_rectangle_l178_17823

/-- The total amount of metal wasted when cutting a maximum-sized circular piece
    from a 3m × 4m rectangular piece of metal, and then cutting a maximum-sized
    square piece from that circular piece. -/
noncomputable def metal_waste (rectangle_length : ℝ) (rectangle_width : ℝ) : ℝ :=
  let circle_radius := min rectangle_length rectangle_width / 2
  let circle_area := Real.pi * circle_radius ^ 2
  let square_side := circle_radius * Real.sqrt 2
  let square_area := square_side ^ 2
  rectangle_length * rectangle_width - square_area

/-- Theorem stating that the metal waste for a 3m × 4m rectangle is 7.5m² -/
theorem metal_waste_for_3x4_rectangle :
  metal_waste 3 4 = 7.5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_waste_for_3x4_rectangle_l178_17823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_member_in_four_committees_l178_17877

/-- Represents a club with committees and members -/
structure Club where
  committees : Finset (Finset Nat)
  members : Finset Nat

/-- Conditions for the club structure -/
def ClubConditions (c : Club) : Prop :=
  (c.committees.card = 11) ∧
  (∀ comm ∈ c.committees, comm.card = 5) ∧
  (∀ comm1 comm2, comm1 ∈ c.committees → comm2 ∈ c.committees → comm1 ≠ comm2 → (comm1 ∩ comm2).Nonempty)

/-- Theorem stating the existence of a member in at least 4 committees -/
theorem member_in_four_committees (c : Club) (h : ClubConditions c) :
  ∃ m ∈ c.members, (c.committees.filter (λ comm => m ∈ comm)).card ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_member_in_four_committees_l178_17877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_cone_volume_l178_17803

/-- The angle (in radians) that maximizes the volume of a cone formed from a circular sheet --/
noncomputable def optimal_angle : ℝ := Real.pi * 65 / 180

/-- The volume of the cone as a function of the cut-out angle θ --/
noncomputable def cone_volume (R : ℝ) (θ : ℝ) : ℝ :=
  (1 / 3) * Real.pi * (R * (2 * Real.pi - θ) / (2 * Real.pi))^2 * 
  Real.sqrt (R^2 * (1 - (2 * Real.pi - θ)^2 / (4 * Real.pi^2)))

theorem optimal_cone_volume (R : ℝ) (h : R > 0) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  ∀ (θ : ℝ), θ ≥ 0 → θ ≤ 2 * Real.pi →
    cone_volume R θ ≤ cone_volume R optimal_angle + ε := by
  sorry

#check optimal_cone_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_cone_volume_l178_17803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_negative_one_l178_17841

-- Define the set M
def M : Set ℝ := {-3, -2, -1}

-- Define the set N
def N : Set ℝ := {x | (x + 2) * (x - 3) < 0}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Theorem statement
theorem intersection_equals_negative_one : M_intersect_N = {-1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_negative_one_l178_17841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_values_l178_17825

/-- Triangle ABC with side lengths AB and BC -/
structure Triangle :=
  (AB : ℝ)
  (BC : ℝ)

/-- The maximum possible value of tan A in a triangle -/
noncomputable def max_tan_A (t : Triangle) : ℝ := 4/3

/-- The maximum possible value of tan C in a triangle -/
noncomputable def max_tan_C (t : Triangle) : ℝ := 5/Real.sqrt 41

/-- Theorem stating the maximum values of tan A and tan C in the given triangle -/
theorem max_tan_values (t : Triangle) 
  (h1 : t.AB = 25) 
  (h2 : t.BC = 20) : 
  max_tan_A t = 4/3 ∧ max_tan_C t = 5/Real.sqrt 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_values_l178_17825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_prism_volume_l178_17837

/-- Volume of a regular triangular prism given its base side length and inscribed sphere radius -/
noncomputable def volume_regular_triangular_prism (base_side_length inscribed_sphere_radius : ℝ) : ℝ :=
sorry

/-- A regular triangular prism with given base side length and inscribed sphere radius has volume √3 -/
theorem regular_triangular_prism_volume 
  (base_side_length : ℝ) 
  (inscribed_sphere_radius : ℝ) 
  (h_base_side : base_side_length = 2 * Real.sqrt 3)
  (h_sphere_radius : inscribed_sphere_radius = Real.sqrt 2 - 1) :
  volume_regular_triangular_prism base_side_length inscribed_sphere_radius = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_prism_volume_l178_17837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l178_17895

/-- Two-dimensional point -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A curve in two-dimensional space -/
structure Curve where
  f : ℝ → ℝ → ℝ

/-- Proposition A: The curves F(x, y) = 0 and G(x, y) = 0 intersect at point P -/
def proposition_A (F G : Curve) (P : Point2D) : Prop :=
  F.f P.x P.y = 0 ∧ G.f P.x P.y = 0

/-- Proposition B: The curve F(x, y) + λG(x, y) = 0 passes through point P -/
def proposition_B (F G : Curve) (P : Point2D) (lambda : ℝ) : Prop :=
  F.f P.x P.y + lambda * G.f P.x P.y = 0

/-- Theorem: A is a sufficient but not necessary condition for B -/
theorem sufficient_not_necessary (F G : Curve) (P : Point2D) :
  (∃ lambda : ℝ, proposition_A F G P → proposition_B F G P lambda) ∧
  ¬(∀ lambda : ℝ, proposition_B F G P lambda → proposition_A F G P) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l178_17895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_acquaintances_in_berezovka_l178_17804

/-- Represents a village with residents and their acquaintances -/
structure Village where
  residents : Finset ℕ
  acquaintances : Finset (ℕ × ℕ)

/-- Checks if a given set of residents can be seated at a round table -/
def can_be_seated_at_round_table (v : Village) (seated : Finset ℕ) : Prop :=
  seated.card = 6 ∧
  ∀ i j k, i ∈ seated → j ∈ seated → k ∈ seated → 
    (i, j) ∈ v.acquaintances ∧ (j, k) ∈ v.acquaintances

/-- The main theorem about the minimum number of acquaintances in Berezovka -/
theorem min_acquaintances_in_berezovka (v : Village) :
  v.residents.card = 200 →
  (∀ seated : Finset ℕ, seated ⊆ v.residents → can_be_seated_at_round_table v seated) →
  v.acquaintances.card ≥ 19600 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_acquaintances_in_berezovka_l178_17804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bookseller_original_cost_l178_17844

/-- Calculates the original price of a book given its selling price with a 10% markup -/
noncomputable def originalPrice (sellingPrice : ℝ) : ℝ := sellingPrice / 1.1

/-- The problem statement -/
theorem bookseller_original_cost (price1 price2 price3 : ℝ) 
  (h1 : price1 = 11)
  (h2 : price2 = 16.5)
  (h3 : price3 = 24.2) :
  originalPrice price1 + originalPrice price2 + originalPrice price3 = 47 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bookseller_original_cost_l178_17844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boole_transformation_preserves_lebesgue_measure_l178_17860

open MeasureTheory

theorem boole_transformation_preserves_lebesgue_measure 
  (l : ℝ) (hl : l > 0) (f : ℝ → ℝ) (hf : Integrable f volume) :
  ∫ x, f x = ∫ x, f (x - l / x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boole_transformation_preserves_lebesgue_measure_l178_17860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_intersection_sum_l178_17824

/-- Triangle with side lengths 13, 14, and 15 -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 13
  hb : b = 14
  hc : c = 15

/-- Centroid of a triangle -/
class HasCentroid (T : Type) where
  centroid : T → ℝ × ℝ

/-- Intersection points A', B', C' -/
structure IntersectionPoints (T : Type) [HasCentroid T] where
  triangle : T
  A' : ℝ × ℝ
  B' : ℝ × ℝ
  C' : ℝ × ℝ

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Sum of distances AA', BB', CC' -/
def sum_distances (T : Type) [HasCentroid T] (t : T) (i : IntersectionPoints T) : ℝ := sorry

/-- Instance of HasCentroid for Triangle -/
instance : HasCentroid Triangle where
  centroid := λ _ => (0, 0)  -- Placeholder implementation

theorem centroid_intersection_sum (t : Triangle) 
  (i : IntersectionPoints Triangle) : sum_distances Triangle t i = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_intersection_sum_l178_17824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_roots_l178_17866

/-- Given f(x) = x^3 - 2x^2 - 3x + 4, if there exist distinct real numbers a, b, and c
    such that f(a) = f(b) = f(c) and a < b < c, then a^2 + b^2 + c^2 = 10 -/
theorem sum_of_squares_of_roots (f : ℝ → ℝ) (a b c : ℝ) : 
  (∀ x, f x = x^3 - 2*x^2 - 3*x + 4) →
  f a = f b → f b = f c →
  a < b → b < c →
  a^2 + b^2 + c^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_roots_l178_17866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_groceries_percentage_l178_17884

noncomputable def salary : ℝ := 4000
noncomputable def fixed_deposit_rate : ℝ := 0.15
noncomputable def cash_in_hand : ℝ := 2380

noncomputable def remaining_amount : ℝ := salary * (1 - fixed_deposit_rate)

noncomputable def amount_spent_on_groceries : ℝ := remaining_amount - cash_in_hand

noncomputable def percentage_spent_on_groceries : ℝ := (amount_spent_on_groceries / remaining_amount) * 100

theorem groceries_percentage :
  percentage_spent_on_groceries = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_groceries_percentage_l178_17884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_l178_17879

/-- The area of a regular hexagon with side length 3 -/
noncomputable def hexagon_area : ℝ := 27 * Real.sqrt 3 / 2

/-- Theorem: The area of a regular hexagon with side length 3 is 27√3/2 -/
theorem regular_hexagon_area :
  let side_length : ℝ := 3
  let num_triangles : ℕ := 6
  let triangle_area : ℝ → ℝ := λ s => Real.sqrt 3 / 4 * s^2
  num_triangles * triangle_area side_length = hexagon_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_l178_17879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l178_17851

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 1 - a * Real.log x

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ,
  (a < 0) →
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ioc 0 1 → x₂ ∈ Set.Ioc 0 1 →
    |f a x₁ - f a x₂| ≤ 4 * |1 / x₁ - 1 / x₂|) →
  a ∈ Set.Ico (-3) 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l178_17851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l178_17836

/-- Represents the decimal 3.17171717... as a rational number -/
def repeating_decimal : ℚ := 314 / 99

/-- The sum of the numerator and denominator of the fraction representing 3.17171717... in its lowest terms -/
def sum_of_parts : ℕ := 413

theorem repeating_decimal_sum : 
  (Int.natAbs repeating_decimal.num) + repeating_decimal.den = sum_of_parts := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l178_17836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_zero_to_one_max_value_f_min_value_f_l178_17857

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^3 + 12*x

-- Theorem for the definite integral
theorem integral_f_zero_to_one :
  ∫ x in (0:ℝ)..(1:ℝ), f x = 23/4 := by sorry

-- Theorem for the maximum value
theorem max_value_f :
  ∃ x ∈ Set.Icc (-3 : ℝ) (1 : ℝ), ∀ y ∈ Set.Icc (-3 : ℝ) (1 : ℝ), f y ≤ f x ∧ f x = 11 := by sorry

-- Theorem for the minimum value
theorem min_value_f :
  ∃ x ∈ Set.Icc (-3 : ℝ) (1 : ℝ), ∀ y ∈ Set.Icc (-3 : ℝ) (1 : ℝ), f x ≤ f y ∧ f x = -16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_zero_to_one_max_value_f_min_value_f_l178_17857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_fractions_l178_17885

/-- A function that checks if a rational number is a terminating decimal -/
def is_terminating_decimal (q : ℚ) : Prop :=
  ∃ (a b : ℕ), q = a / b ∧ ∀ (p : ℕ), p.Prime → p ∣ b → p = 2 ∨ p = 5

/-- A function that checks if a rational number has a non-zero thousandths digit -/
def has_nonzero_thousandths (q : ℚ) : Prop :=
  ∃ (k : ℕ), 1 ≤ k ∧ k < 10 ∧ (q * 1000).num % (q * 1000).den ≠ 0 ∧
    ((q * 1000).num % (q * 1000).den) * 10 / (q * 1000).den = k

/-- The main theorem stating that there are exactly 4 positive integers satisfying the conditions -/
theorem count_special_fractions :
  ∃! (S : Finset ℕ), 
    S.card = 4 ∧ 
    (∀ n ∈ S, is_terminating_decimal (1 / n) ∧ has_nonzero_thousandths (1 / n)) ∧
    (∀ n ∉ S, ¬(is_terminating_decimal (1 / n) ∧ has_nonzero_thousandths (1 / n))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_fractions_l178_17885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_bisect_in_all_shapes_l178_17835

-- Define the shapes
structure Rectangle where
  length : ℝ
  width : ℝ

structure Rhombus where
  side : ℝ
  angle : ℝ

structure Square where
  side : ℝ

-- Define the diagonal property
def diagonalsBisectEachOther (shape : Type) : Prop :=
  ∀ (s : shape), ∃ (d1 d2 : ℝ × ℝ), d1.1 = d2.1 ∧ d1.2 = d2.2

-- Theorem statement
theorem diagonals_bisect_in_all_shapes :
  (diagonalsBisectEachOther Rectangle) ∧
  (diagonalsBisectEachOther Rhombus) ∧
  (diagonalsBisectEachOther Square) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_bisect_in_all_shapes_l178_17835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l178_17871

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a point on the ellipse -/
structure EllipsePoint (e : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- Predicate to check if a given radius is the incircle radius of a triangle -/
def is_incircle_radius (r : ℝ) (A B F₂ : ℝ × ℝ) : Prop :=
  sorry  -- Definition of incircle radius property

/-- Main theorem statement -/
theorem ellipse_eccentricity_special_case (e : Ellipse) 
  (A B : EllipsePoint e) (F₁ F₂ : ℝ × ℝ) :
  (∃ (l : ℝ × ℝ → ℝ × ℝ → Prop), l F₁ (A.x, A.y) ∧ l F₁ (B.x, B.y)) →  -- Line through F₁ intersects ellipse at A and B
  (abs (A.y - B.y) = 3) →                            -- |y₁ - y₂| = 3
  (abs (F₁.1 - F₂.1) = 2) →                          -- |F₁F₂| = 2
  (∃ (r : ℝ), r = 1 ∧ is_incircle_radius r (A.x, A.y) (B.x, B.y) F₂) → -- Radius of incircle of ABF₂ is 1
  eccentricity e = 2/3 :=
by
  sorry  -- Proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l178_17871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olivias_pool_fill_time_l178_17894

/-- Represents the pool filling scenario -/
structure PoolFilling where
  capacity : ℚ  -- Pool capacity in gallons
  hoses : ℕ     -- Number of hoses
  flow_rate : ℚ -- Flow rate per hose in gallons per minute

/-- Calculates the time needed to fill the pool in hours, rounded to the nearest hour -/
def fill_time (p : PoolFilling) : ℕ :=
  let total_flow_per_hour := p.hoses * p.flow_rate * 60
  (p.capacity / total_flow_per_hour + 1/2).floor.toNat

/-- The specific pool filling scenario described in the problem -/
def olivias_pool : PoolFilling :=
  { capacity := 30000
    hoses := 5
    flow_rate := 3 }

/-- Theorem stating that it takes 34 hours to fill Olivia's pool -/
theorem olivias_pool_fill_time :
  fill_time olivias_pool = 34 := by
  sorry

#eval fill_time olivias_pool

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olivias_pool_fill_time_l178_17894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_B_distance_A_C_eq_two_sum_distances_eq_AB_distance_D_P_eq_three_l178_17865

-- Define the distance function on a number line
def distance (x y : ℝ) : ℝ := |x - y|

-- Define points A, B, C, D, E, F
def A : ℝ := -1
def B : ℝ := 2
def D : ℝ := -2
def E : ℝ := 4
def F : ℝ := 6

-- Theorem 1
theorem distance_A_B : distance A B = 3 := by sorry

-- Theorem 2
theorem distance_A_C_eq_two (x : ℝ) :
  distance A x = 2 → x = 1 ∨ x = -3 := by sorry

-- Theorem 3
theorem sum_distances_eq_AB (x : ℤ) :
  distance A x + distance B x = distance A B →
  x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 := by sorry

-- Define the position of P as a function of time
noncomputable def P (t : ℝ) : ℝ :=
  if t ≤ 4 then D + 2 * t else F - 2 * (t - 4)

-- Theorem 4
theorem distance_D_P_eq_three :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
  distance D (P t₁) = 3 ∧
  distance D (P t₂) = 3 ∧
  t₁ = 1.5 ∧ t₂ = 6.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_B_distance_A_C_eq_two_sum_distances_eq_AB_distance_D_P_eq_three_l178_17865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l178_17876

noncomputable def x (t : ℝ) : ℝ := 2 * (2 * Real.cos t - Real.cos (2 * t))
noncomputable def y (t : ℝ) : ℝ := 2 * (2 * Real.sin t - Real.sin (2 * t))

noncomputable def arc_length (f g : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ t in a..b, Real.sqrt ((deriv f t) ^ 2 + (deriv g t) ^ 2)

theorem arc_length_of_curve :
  arc_length x y 0 (Real.pi / 3) = 8 * (2 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l178_17876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sigmoid_properties_l178_17867

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

theorem sigmoid_properties :
  (∀ x, 0 ≤ f x ∧ f x ≤ 1) ∧
  (∀ x y, x < y → f x < f y) ∧
  (∀ x y, f (x + y) ≤ f x + f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sigmoid_properties_l178_17867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_door_length_l178_17801

/-- Proves that a door with area 3 m² and width 150 cm has length 200 cm -/
theorem door_length (area : ℝ) (width_cm : ℝ) (length_cm : ℝ) : 
  area = 3 → 
  width_cm = 150 → 
  area = (width_cm / 100) * (length_cm / 100) → 
  length_cm = 200 := by
  intros h1 h2 h3
  -- The proof steps would go here
  sorry

#check door_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_door_length_l178_17801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_l178_17848

/-- Given points P and Q in the xy-plane, find the value of m that minimizes PR + RQ for point R -/
theorem minimize_distance (P Q : ℝ × ℝ) (m : ℝ) : 
  P = (-2, -4) →
  Q = (5, 3) →
  (∀ m' : ℝ, dist P (2, m) + dist (2, m) Q ≤ dist P (2, m') + dist (2, m') Q) →
  m = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_l178_17848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_5Tn_n_minus_1_is_1_l178_17826

def triangular_number (n : ℕ+) : ℕ := (n.val * (n.val + 1)) / 2

theorem gcd_5Tn_n_minus_1_is_1 (n : ℕ+) : 
  Nat.gcd (5 * triangular_number n) (n.val - 1) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_5Tn_n_minus_1_is_1_l178_17826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_negative_two_l178_17839

theorem expression_equals_negative_two :
  (Real.sqrt 3 - 1) ^ 0 + (-1/3 : ℝ) ^ (-1 : ℤ) - 2 * Real.cos (30 * π / 180) + Real.sqrt (1/2) * Real.sqrt 6 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_negative_two_l178_17839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_specific_l178_17852

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (n : ℕ) (a₁ : ℝ) (d : ℝ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem: Sum of first n terms of a specific arithmetic sequence -/
theorem arithmetic_sum_specific (n : ℕ) : 
  let d : ℝ := 2
  let a₄ : ℝ := 8
  let a₁ : ℝ := a₄ - 3 * d
  arithmetic_sum n a₁ d = n * (n + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_specific_l178_17852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_diagonal_ratio_value_l178_17887

/-- An isosceles trapezoid with specific proportions -/
structure IsoscelesTrapezoid where
  /-- Length of the smaller base -/
  smaller_base : ℝ
  /-- Length of the larger base -/
  larger_base : ℝ
  /-- Length of the altitude -/
  altitude : ℝ
  /-- Length of the diagonal -/
  diagonal : ℝ
  /-- The larger base is twice the smaller base -/
  larger_base_prop : larger_base = 2 * smaller_base
  /-- The altitude is half the smaller base -/
  altitude_prop : altitude = smaller_base / 2
  /-- The diagonal is properly defined based on the Pythagorean theorem -/
  diagonal_prop : diagonal^2 = smaller_base^2 + altitude^2

/-- The ratio of the smaller base to the diagonal in the specific isosceles trapezoid -/
noncomputable def base_diagonal_ratio (t : IsoscelesTrapezoid) : ℝ :=
  t.smaller_base / t.diagonal

/-- Theorem stating the ratio of the smaller base to the diagonal -/
theorem base_diagonal_ratio_value (t : IsoscelesTrapezoid) :
    base_diagonal_ratio t = 2 * Real.sqrt 5 / 5 := by
  sorry

#check base_diagonal_ratio_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_diagonal_ratio_value_l178_17887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_antiderivative_of_f_l178_17818

open Real

/-- The integrand function --/
noncomputable def f (x : ℝ) : ℝ := (((1 + x^(2/3))^3)^(1/4)) / (x^2 * x^(1/6))

/-- The antiderivative function --/
noncomputable def F (x : ℝ) : ℝ := -((6 * ((1 + x^(2/3))^7)^(1/4)) / (7 * x^(7/6)))

/-- Theorem stating that F is an antiderivative of f --/
theorem F_is_antiderivative_of_f : 
  ∀ x : ℝ, x > 0 → HasDerivAt F (f x) x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_antiderivative_of_f_l178_17818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l178_17838

open Real

-- Define the differential equation
def diff_eq (y : ℝ → ℝ) (x : ℝ) : Prop :=
  (1 + x^2) * ((deriv (deriv y)) x) + 2 * x * ((deriv y) x) = 12 * x^3

-- Define the general solution
noncomputable def general_solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  C₁ * arctan x + x^3 - 3 * x + C₂

-- Theorem statement
theorem general_solution_satisfies_diff_eq (C₁ C₂ : ℝ) :
  ∀ x, diff_eq (general_solution C₁ C₂) x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l178_17838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_symmetry_and_equality_l178_17875

-- Define the necessary functions
noncomputable def f (x : ℝ) := Real.cos (-x)
noncomputable def g (x : ℝ) := Real.cos (abs x)
noncomputable def h (x : ℝ) := Real.cos x

-- State the theorem
theorem cos_symmetry_and_equality :
  (∀ x, f x = g x) ∧ 
  (∀ x, h x = h (-x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_symmetry_and_equality_l178_17875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_inequality_l178_17816

noncomputable def f (a x : ℝ) : ℝ := (a * x^2 + x + a) * Real.exp (-x)

theorem max_value_and_inequality :
  (∀ a : ℝ, (a ≥ 0 ∧ (∀ x : ℝ, f a x ≤ 3 / Real.exp 1) ∧ (∃ x : ℝ, f a x = 3 / Real.exp 1)) ↔ a = 1) ∧
  (∀ b : ℝ, (∀ a : ℝ, a ≤ 0 → (∀ x : ℝ, x ≥ 0 → f a x ≤ b * Real.log (x + 1))) ↔ b ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_inequality_l178_17816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_properties_l178_17812

-- Define the types for points and distances
variable {Point : Type}
variable (distance : Point → Point → ℝ)

-- Define collinearity
def collinear (distance : Point → Point → ℝ) (A B C : Point) : Prop := 
  ∃ t : ℝ, distance A C = t * distance A B ∨ distance B C = t * distance A B

-- Define midpoint
def is_midpoint (distance : Point → Point → ℝ) (M A B : Point) : Prop :=
  distance M A = distance M B ∧ collinear distance A M B

-- Define symmetry with respect to a point
def symmetric_wrt (distance : Point → Point → ℝ) (A O A' : Point) : Prop :=
  collinear distance A O A' ∧ distance O A = distance O A' ∧ is_midpoint distance O A A'

-- Theorem statement
theorem symmetry_properties {Point : Type} (distance : Point → Point → ℝ) (A O A' : Point) :
  symmetric_wrt distance A O A' → 
  (collinear distance A O A' ∧ distance O A = distance O A' ∧ is_midpoint distance O A A') :=
by
  intro h
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_properties_l178_17812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_x_squared_cos_x_l178_17880

theorem derivative_x_squared_cos_x (x : ℝ) :
  deriv (λ x => x^2 * Real.cos x) x = 2*x*Real.cos x - x^2*Real.sin x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_x_squared_cos_x_l178_17880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_l178_17809

/-- The height of a square-based pyramid with specific properties -/
theorem pyramid_height (base_edge : ℝ) (angle : ℝ) 
  (h_base : base_edge = 2)
  (h_angle : angle = 30 * π / 180) :
  ∃ (height : ℝ), height = (Real.sqrt 15 + Real.sqrt 3) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_l178_17809
