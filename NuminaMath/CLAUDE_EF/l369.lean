import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_ends_6_rounds_prob_prob_dist_x_correct_expected_value_x_correct_l369_36989

-- Define the game parameters
def win_points : ℕ := 2
def lose_points : ℕ := 0
def win_threshold : ℕ := 8
def team_a_win_prob : ℚ := 2/3

-- Define the function to calculate the probability of game ending after exactly 6 rounds
noncomputable def prob_game_ends_6_rounds : ℚ := 200/729

-- Define the type for the number of rounds needed to end the game
def rounds_to_end : Type := Fin 3

-- Define the probability distribution of X when Team A is leading 4:2
noncomputable def prob_dist_x : rounds_to_end → ℚ
| ⟨0, _⟩ => 4/9
| ⟨1, _⟩ => 1/3
| ⟨2, _⟩ => 2/9

-- Define the expected value of X when Team A is leading 4:2
noncomputable def expected_value_x : ℚ := 25/9

-- Theorem statements
theorem game_ends_6_rounds_prob :
  prob_game_ends_6_rounds = 200/729 := by sorry

theorem prob_dist_x_correct :
  (prob_dist_x ⟨0, by norm_num⟩ = 4/9) ∧
  (prob_dist_x ⟨1, by norm_num⟩ = 1/3) ∧
  (prob_dist_x ⟨2, by norm_num⟩ = 2/9) := by sorry

theorem expected_value_x_correct :
  expected_value_x = 25/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_ends_6_rounds_prob_prob_dist_x_correct_expected_value_x_correct_l369_36989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l369_36984

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def GeometricSequence (t : Triangle) : Prop :=
  t.b ^ 2 = t.a * t.c

def SideRelation (t : Triangle) : Prop :=
  t.a ^ 2 - t.c ^ 2 = t.a * t.c - t.b * t.c

def CosineRule (t : Triangle) : Prop :=
  Real.cos t.A = (t.b ^ 2 + t.c ^ 2 - t.a ^ 2) / (2 * t.b * t.c)

def SineRule (t : Triangle) : Prop :=
  Real.sin t.B / t.a = Real.sin t.A / t.b

-- Main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : GeometricSequence t) 
  (h2 : SideRelation t) 
  (h3 : CosineRule t) 
  (h4 : SineRule t) : 
  Real.cos t.A = 1 / 2 ∧ t.b * Real.sin t.B / t.c = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l369_36984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l369_36992

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3

noncomputable def h (x : ℝ) : ℝ := Real.log (3 * x - 3) / Real.log 3

theorem graph_translation (x : ℝ) :
  f (x - 1) + 1 = h x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l369_36992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AME_l369_36939

/-- Rectangle ABCD with given sides and points M and E -/
structure Rectangle where
  AB : ℝ
  BC : ℝ
  M : ℝ × ℝ  -- Coordinates of M
  E : ℝ      -- Position of E on AB

/-- Properties of the rectangle and points -/
def rectangle_properties (rect : Rectangle) : Prop :=
  rect.AB = 12 ∧ 
  rect.BC = 10 ∧
  rect.M = ((rect.AB / 2), (rect.BC / 2)) ∧  -- M is midpoint of AC
  rect.E = rect.AB / 3 ∧
  (rect.M.1 - rect.E) * (rect.AB) + (rect.M.2 * rect.BC) = 0  -- ME ⟂ AC

/-- The area of triangle AME -/
noncomputable def triangle_AME_area (rect : Rectangle) : ℝ :=
  (1 / 2) * rect.E * (((rect.M.1 - rect.E)^2 + rect.M.2^2).sqrt)

/-- Theorem stating the area of triangle AME -/
theorem area_of_triangle_AME (rect : Rectangle) 
  (h : rectangle_properties rect) : 
  triangle_AME_area rect = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AME_l369_36939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2alpha_value_l369_36993

theorem tan_2alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo π (3*π/2)) 
  (h2 : (1 + Real.sin (2*α)) / (1 + Real.cos (2*α)) = 9/2) : 
  Real.tan (2*α) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2alpha_value_l369_36993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_length_l369_36926

/-- Line l in parametric form -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t / 2, 1 - (Real.sqrt 3 / 2) * t)

/-- Circle C in polar form -/
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.sin θ

/-- Theorem stating the number of intersection points and the length of the segment -/
theorem intersection_and_length :
  ∃ A B : ℝ × ℝ,
    (∃ t : ℝ, line_l t = A) ∧
    (∃ t : ℝ, line_l t = B) ∧
    (∃ θ : ℝ, (circle_C θ * Real.cos θ, circle_C θ * Real.sin θ) = A) ∧
    (∃ θ : ℝ, (circle_C θ * Real.cos θ, circle_C θ * Real.sin θ) = B) ∧
    (∀ P : ℝ × ℝ, (∃ t : ℝ, line_l t = P) ∧ (∃ θ : ℝ, (circle_C θ * Real.cos θ, circle_C θ * Real.sin θ) = P) → P = A ∨ P = B) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_length_l369_36926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_special_vectors_l369_36969

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def angle_between_vectors (a b : V) : ℝ :=
  Real.arccos ((inner a b) / (norm a * norm b))

theorem angle_between_special_vectors (a b : V) 
  (h_nonzero_a : a ≠ 0)
  (h_nonzero_b : b ≠ 0)
  (h_norm_a : norm a = 2 * norm b)
  (h_norm_sum : norm (a + b) = 2 * norm b) :
  angle_between_vectors a b = Real.arccos (-1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_special_vectors_l369_36969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_transportation_cost_correct_l369_36904

/-- Calculate the total transportation cost for a one-week event --/
def total_transportation_cost
  (off_peak_rate : ℝ)
  (peak_rate : ℝ)
  (distance : ℝ)
  (toll_fee : ℝ)
  (tax_rate : ℝ)
  (train_cost : ℝ)
  (carpool_discount : ℝ)
  : ℝ :=
let daily_cab_fare := (off_peak_rate + peak_rate) * distance
let discounted_fare := daily_cab_fare * (1 - carpool_discount)
let total_with_toll := discounted_fare + toll_fee
let total_with_tax := total_with_toll * (1 + tax_rate)
let cab_days_cost := total_with_tax * 5
let train_days_cost := train_cost * 2
cab_days_cost + train_days_cost

theorem total_transportation_cost_correct
  (off_peak_rate : ℝ)
  (peak_rate : ℝ)
  (distance : ℝ)
  (toll_fee : ℝ)
  (tax_rate : ℝ)
  (train_cost : ℝ)
  (carpool_discount : ℝ)
  (h_off_peak : off_peak_rate = 2.5)
  (h_peak : peak_rate = 3.5)
  (h_distance : distance = 200)
  (h_toll : toll_fee = 10)
  (h_tax : tax_rate = 0.05)
  (h_train : train_cost = 70)
  (h_discount : carpool_discount = 0.2)
  : total_transportation_cost off_peak_rate peak_rate distance toll_fee tax_rate train_cost carpool_discount = 5232.5 :=
by
  sorry

#eval total_transportation_cost 2.5 3.5 200 10 0.05 70 0.2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_transportation_cost_correct_l369_36904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_cone_apex_angle_l369_36998

/-- The apex angle of a cone -/
def ApexAngle : Type := ℝ

/-- The plane that all cones touch -/
def TouchingPlane : Type := Unit

/-- A cone with a given apex angle -/
structure Cone :=
  (angle : ApexAngle)
  (touchesPlane : TouchingPlane → Prop)

/-- Three cones with a common vertex -/
structure ThreeCones :=
  (cone1 : Cone)
  (cone2 : Cone)
  (cone3 : Cone)
  (commonVertex : Unit)
  (touchExternally : Prop)
  (onOneSide : TouchingPlane → Prop)

/-- The configuration of cones as described in the problem -/
noncomputable def problemSetup (π : ℝ) : ThreeCones :=
  { cone1 := { angle := π/3, touchesPlane := λ _ => True },
    cone2 := { angle := π/3, touchesPlane := λ _ => True },
    cone3 := { angle := 2 * Real.arctan (1 / (2 * (Real.sqrt 3 + Real.sqrt 2))), touchesPlane := λ _ => True },
    commonVertex := (),
    touchExternally := True,
    onOneSide := λ _ => True }

/-- The theorem stating the apex angle of the third cone -/
theorem third_cone_apex_angle (π : ℝ) :
  ∃ (setup : ThreeCones), setup = problemSetup π ∧
  (setup.cone3.angle = 2 * Real.arctan (1 / (2 * (Real.sqrt 3 + Real.sqrt 2))) ∨
   setup.cone3.angle = 2 * Real.arctan (1 / (2 * (Real.sqrt 3 - Real.sqrt 2)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_cone_apex_angle_l369_36998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_bounds_certain_event_probability_medical_treatment_probability_lottery_probability_l369_36971

-- Define a probability measure
structure ProbabilityMeasure (Ω : Type) where
  measure : Set Ω → ℝ
  nonneg : ∀ A, 0 ≤ measure A
  total : measure (Set.univ : Set Ω) = 1

-- 1. Probability of an event is between 0 and 1
theorem probability_bounds {Ω : Type} (P : ProbabilityMeasure Ω) (A : Set Ω) :
  0 ≤ P.measure A ∧ P.measure A ≤ 1 := by
  sorry

-- 2. Probability of a certain event is 1
theorem certain_event_probability {Ω : Type} (P : ProbabilityMeasure Ω) :
  P.measure (Set.univ : Set Ω) = 1 := by
  sorry

-- 3. Estimated probability from sample data
noncomputable def estimated_probability (successes trials : ℕ) : ℝ :=
  (successes : ℝ) / (trials : ℝ)

theorem medical_treatment_probability :
  estimated_probability 380 500 = 0.76 := by
  sorry

-- 4. Probability of exactly 5 successes in 10 independent trials
noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

theorem lottery_probability :
  binomial_probability 10 5 0.5 ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_bounds_certain_event_probability_medical_treatment_probability_lottery_probability_l369_36971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_triangle_area_l369_36915

/-- Represents a side of a triangle -/
inductive Side
  | A
  | B
  | C

/-- Represents a triangle -/
structure Triangle where
  /-- The area of the triangle -/
  area : ℝ
  /-- Function to get the length of a side -/
  length : Side → ℝ

/-- Given a triangle whose dimensions are doubled to form a new triangle with an area of 72 square feet, 
    prove that the area of the original triangle was 18 square feet. -/
theorem original_triangle_area (original new : Triangle) : 
  (∀ (side : Side), new.length side = 2 * original.length side) →
  new.area = 72 →
  original.area = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_triangle_area_l369_36915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_M_cubed_l369_36959

open Matrix

theorem det_M_cubed {n : Type*} [Fintype n] [DecidableEq n] 
  (M : Matrix n n ℝ) (h : det M = 3) : det (M^3) = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_M_cubed_l369_36959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_eight_relation_l369_36936

theorem power_eight_relation (y : ℝ) (h : (8 : ℝ)^(3*y) = 512) : (8 : ℝ)^(3*y - 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_eight_relation_l369_36936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_complete_arc_l369_36982

-- Define the circle as a set of points in polar coordinates
def circle_set : Set (ℝ × ℝ) := {p | p.1 = Real.sin p.2}

-- Define the arc of the circle from 0 to t
def arc (t : ℝ) : Set (ℝ × ℝ) := {p | p ∈ circle_set ∧ 0 ≤ p.2 ∧ p.2 ≤ t}

-- Theorem statement
theorem smallest_complete_arc :
  ∀ t : ℝ, t > 0 → (arc t = circle_set → t ≥ π) ∧ (arc π = circle_set) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_complete_arc_l369_36982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_transitive_through_parallel_l369_36966

-- Define the types for lines and planes
structure Line where

structure Plane where

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry

def parallel_planes (p1 p2 : Plane) : Prop := sorry

def parallel_lines (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_transitive_through_parallel
  (a b : Line) (α β : Plane)
  (h1 : perpendicular a α)
  (h2 : parallel_planes α β)
  (h3 : parallel_lines a b) :
  perpendicular b β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_transitive_through_parallel_l369_36966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_2x_plus_5_when_x_is_4_l369_36943

theorem square_of_2x_plus_5_when_x_is_4 :
  (fun x : ℝ => (2 * x + 5) ^ 2) 4 = 169 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_2x_plus_5_when_x_is_4_l369_36943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_approx_69_3_l369_36923

/-- Represents the properties of the rectangle arrangement -/
structure RectangleArrangement where
  num_rectangles : ℕ
  area_PQRS : ℝ
  length_width_ratio : ℝ

/-- Calculates the length of each rectangle in the arrangement -/
noncomputable def rectangle_length (arr : RectangleArrangement) : ℝ :=
  let total_width := (5 / 2 : ℝ) * Real.sqrt (arr.area_PQRS / (6 * 5 / 2 : ℝ))
  arr.length_width_ratio * (total_width / 5)

/-- The main theorem stating the approximate length of each rectangle -/
theorem rectangle_length_approx_69_3 (arr : RectangleArrangement) 
  (h1 : arr.num_rectangles = 10)
  (h2 : arr.area_PQRS = 8000)
  (h3 : arr.length_width_ratio = 3) :
  ∃ ε > 0, |rectangle_length arr - 69.3| < ε := by
  sorry

#eval Float.round ((3 : Float) * Float.sqrt (8000 / 15))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_approx_69_3_l369_36923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_one_l369_36960

/-- Point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The area of a triangle given two points in polar coordinates and the pole -/
noncomputable def triangleArea (a b : PolarPoint) : ℝ :=
  (1/2) * a.r * b.r * |Real.sin (b.θ - a.θ)|

/-- Theorem: The area of triangle AOB is 1 -/
theorem triangle_area_is_one :
  let a : PolarPoint := ⟨1, π/6⟩
  let b : PolarPoint := ⟨2, 2*π/3⟩
  triangleArea a b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_one_l369_36960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_convergence_l369_36920

def Grid := Matrix (Fin 3) (Fin 3) Int

def adjacent (i j : Fin 3) : List (Fin 3 × Fin 3) :=
  [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
  |> List.filter (λ (x, y) => x.val < 3 ∧ y.val < 3)

def step (g : Grid) : Grid :=
  Matrix.of (λ i j =>
    (adjacent i j).map (λ (x, y) => g x y)
    |> List.prod)

def isAllOnes (g : Grid) : Prop :=
  ∀ i j, g i j = 1

theorem grid_convergence (g : Grid) (h : ∀ i j, g i j = 1 ∨ g i j = -1) :
  isAllOnes (step (step (step (step g)))) := by
  sorry

#check grid_convergence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_convergence_l369_36920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_product_in_S_l369_36951

def S : Finset Int := {-10, -4, 0, 1, 2}

theorem smallest_product_in_S : 
  (∀ a b : Int, a ∈ S → b ∈ S → a * b ≥ -20) ∧ 
  (∃ x y : Int, x ∈ S ∧ y ∈ S ∧ x * y = -20) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_product_in_S_l369_36951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_promotion_savings_l369_36911

/-- Calculates the total savings from a pizza promotion -/
theorem pizza_promotion_savings 
  (regular_price : ℕ) 
  (promo_price : ℕ) 
  (num_pizzas : ℕ) 
  (h1 : regular_price = 18) 
  (h2 : promo_price = 5) 
  (h3 : num_pizzas = 3) : 
  (regular_price - promo_price) * num_pizzas = 39 := by
  sorry

#check pizza_promotion_savings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_promotion_savings_l369_36911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l369_36927

/-- Calculates the time for a train to cross a platform given its speed -/
noncomputable def time_to_cross (train_length : ℝ) (platform_length : ℝ) (speed : ℝ) : ℝ :=
  (train_length + platform_length) / speed

theorem train_crossing_time
  (train_length : ℝ)
  (first_platform_length : ℝ)
  (first_crossing_time : ℝ)
  (second_platform_length : ℝ)
  (h1 : train_length = 70)
  (h2 : first_platform_length = 170)
  (h3 : first_crossing_time = 15)
  (h4 : second_platform_length = 250) :
  time_to_cross train_length second_platform_length
    ((train_length + first_platform_length) / first_crossing_time) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l369_36927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l369_36962

-- Define the original hyperbola
def original_hyperbola (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

-- Define the new hyperbola
def new_hyperbola (x y : ℝ) : Prop :=
  4 * x^2 / 9 - y^2 / 4 = 1

-- Define the point that the new hyperbola should pass through
noncomputable def point : ℝ × ℝ := (-3, 2 * Real.sqrt 3)

-- Define a function to get the asymptotes of a hyperbola
def asymptotes (a b : ℝ) : Set (ℝ → ℝ) :=
  {f | ∃ (sign : ℝ), (sign = 1 ∨ sign = -1) ∧ f = λ x => sign * (b / a) * x}

-- Theorem statement
theorem hyperbola_properties :
  (asymptotes 3 4 = asymptotes 3 4) ∧
  new_hyperbola point.fst point.snd := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l369_36962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_distance_minimum_tangent_line_distance_l369_36910

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y

-- Define the circle O
def circleO (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the focus of the parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p/2)

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem 1
theorem intersection_point_distance (p : ℝ) (x_A y_A : ℝ) :
  p > 0 →
  parabola p x_A y_A →
  circleO x_A y_A →
  circleO 0 (p/2) →
  distance x_A y_A 0 (p/2) = Real.sqrt 5 - 1 := by
  sorry

-- Theorem 2
theorem minimum_tangent_line_distance (p : ℝ) :
  p > 0 →
  (∃ x_M y_M x_N y_N : ℝ,
    parabola p x_M y_M ∧
    circleO x_N y_N ∧
    -- Additional conditions for tangency would be needed here
    ∀ q : ℝ, q > 0 → ∀ x_M' y_M' x_N' y_N' : ℝ,
      parabola q x_M' y_M' ∧
      circleO x_N' y_N' →
      distance x_M y_M x_N y_N ≤ distance x_M' y_M' x_N' y_N') →
  ∃ x_M y_M x_N y_N : ℝ,
    distance x_M y_M x_N y_N = 2 * Real.sqrt 2 ∧
    p = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_distance_minimum_tangent_line_distance_l369_36910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scout_earnings_l369_36916

/-- Scout's earnings calculation --/
theorem scout_earnings (base_pay hour_rate tip_rate : ℕ)
  (saturday_hours saturday_deliveries : ℕ)
  (sunday_hours sunday_deliveries : ℕ)
  (h1 : base_pay = 10)
  (h2 : tip_rate = 5)
  (h3 : saturday_hours = 4)
  (h4 : saturday_deliveries = 5)
  (h5 : sunday_hours = 5)
  (h6 : sunday_deliveries = 8) :
  base_pay * (saturday_hours + sunday_hours) + 
  tip_rate * (saturday_deliveries + sunday_deliveries) = 155 := by
  sorry

#check scout_earnings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scout_earnings_l369_36916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_l369_36967

/-- Regular triangular prism with base side length a -/
structure RegularTriangularPrism (a : ℝ) where
  base_side_length : a > 0

/-- Cross-section of the prism -/
structure CrossSection (a : ℝ) (P : RegularTriangularPrism a) where
  parallel_to_BC : Prop
  perpendicular_to_PBC : Prop
  angle_with_base : ℝ
  area : ℝ

/-- The theorem statement -/
theorem cross_section_area (a : ℝ) (P : RegularTriangularPrism a) 
  (S : CrossSection a P)
  (h₁ : S.parallel_to_BC)
  (h₂ : S.perpendicular_to_PBC)
  (h₃ : S.angle_with_base = 30 * π / 180) -- Angle with base is 30°
  : S.area = (3 * Real.sqrt 3 / 64) * a^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_l369_36967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solutions_l369_36922

/-- The number of real solutions to the equation x^3 - 3x = a -/
noncomputable def num_solutions (a : ℝ) : ℕ :=
  if a > 2 ∨ a < -2 then 1
  else if a = 2 ∨ a = -2 then 2
  else 3

theorem cubic_equation_solutions (a : ℝ) :
  (∃ x : ℝ, x^3 - 3*x = a) ∧
  (num_solutions a = 1 ↔ (a > 2 ∨ a < -2)) ∧
  (num_solutions a = 2 ↔ (a = 2 ∨ a = -2)) ∧
  (num_solutions a = 3 ↔ (-2 < a ∧ a < 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solutions_l369_36922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l369_36985

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x / 2 + Real.pi / 6) + 3

def increasingIntervals (k : ℤ) : Set ℝ := Set.Icc (-4 * Real.pi / 3 + 4 * ↑k * Real.pi) (2 * Real.pi / 3 + 4 * ↑k * Real.pi)

theorem f_properties :
  (∀ k, StrictMonoOn f (increasingIntervals k)) ∧
  (∀ x ∈ Set.Icc (Real.pi / 3) (4 * Real.pi / 3), f x ≥ 9 / 2) ∧
  (∀ x ∈ Set.Icc (Real.pi / 3) (4 * Real.pi / 3), f x ≤ 6) ∧
  (f (4 * Real.pi / 3) = 9 / 2) ∧
  (f (2 * Real.pi / 3) = 6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l369_36985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l369_36944

/-- Given a triangle ABC with cos A = 7/8, c - a = 2, and b = 3, prove that a = 1 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < a → 0 < b → 0 < c →  -- Ensure positive side lengths
  Real.cos A = 7/8 →
  c - a = 2 →
  b = 3 →
  a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l369_36944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_implies_relation_l369_36932

/-- If (x + 3) is a factor of x^3 - mx^2 + nx - 15, then m = -14/3 - n/3 --/
theorem factor_implies_relation (m n : ℚ) : 
  (∀ x : ℚ, (x + 3) * (x^2 - (m + 3)*x + (n + m*3 + 9)) = x^3 - m*x^2 + n*x - 15) →
  m = -14/3 - n/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_implies_relation_l369_36932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_day_challenge_l369_36957

/-- Represents a class with girls and boys -/
structure MyClass where
  girls : Nat
  boys : Nat

/-- Represents a grade with three classes -/
structure Grade where
  class1 : MyClass
  class2 : MyClass
  class3 : MyClass

def third_grade : Grade := {
  class1 := { girls := 10, boys := 14 }
  class2 := { girls := 12, boys := 10 }
  class3 := { girls := 11, boys := 9 }
}

def fourth_grade : Grade := {
  class1 := { girls := 12, boys := 13 }
  class2 := { girls := 15, boys := 11 }
  class3 := { girls := 14, boys := 12 }
}

def fifth_grade : Grade := {
  class1 := { girls := 9, boys := 13 }
  class2 := { girls := 10, boys := 11 }
  class3 := { girls := 11, boys := 14 }
}

def total_girls (g : Grade) : Nat :=
  g.class1.girls + g.class2.girls + g.class3.girls

def total_boys (g : Grade) : Nat :=
  g.class1.boys + g.class2.boys + g.class3.boys

theorem field_day_challenge :
  (total_boys third_grade + total_boys fourth_grade + total_boys fifth_grade) -
  (total_girls third_grade + total_girls fourth_grade + total_girls fifth_grade) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_day_challenge_l369_36957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_cubes_12x12x12_l369_36964

/-- Represents a cube with side length n -/
structure Cube where
  side_length : ℕ

/-- Calculates the number of visible unit cubes from a single point for a cube -/
def visible_unit_cubes (c : Cube) : ℕ :=
  3 * c.side_length^2 - 3 * (c.side_length - 1) + 1

/-- Theorem stating that for a 12x12x12 cube, the number of visible unit cubes is 400 -/
theorem visible_cubes_12x12x12 :
  visible_unit_cubes { side_length := 12 } = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_cubes_12x12x12_l369_36964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l369_36906

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = -3

-- Define the area of the region
noncomputable def region_area : ℝ := 10 * Real.pi

-- Theorem statement
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ (x y : ℝ), region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Provide the center coordinates and radius
  let center_x := 2
  let center_y := 3
  let radius := Real.sqrt 10
  
  -- Assert the existence of these values
  use center_x, center_y, radius
  
  constructor
  
  -- Prove the equivalence of the equations
  · intro x y
    constructor
    · intro h
      -- Expand the definition and simplify
      sorry
    · intro h
      -- Expand the definition and simplify
      sorry
  
  -- Prove the area equality
  · simp [region_area]
    -- Simplify and prove the equality
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l369_36906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l369_36908

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 2

-- Define the derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := x^2 - 4

-- Theorem statement
theorem decreasing_interval_of_f :
  ∀ x : ℝ, (x > -2 ∧ x < 2) ↔ (f' x < 0) :=
by
  sorry

#check decreasing_interval_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l369_36908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l369_36901

theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (l : ℝ) :
  a = (2, 4) →
  b = (1, 1) →
  (b.1 * (a.1 + l * b.1) + b.2 * (a.2 + l * b.2) = 0) →
  l = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l369_36901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l369_36991

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y : ℝ, y = m * (x - 1) + f 1 ↔ y = f x + f' 1 * (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l369_36991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_equals_80_l369_36907

-- Define the points
variable (A B C D E F : ℝ × ℝ)

-- Define the conditions
axiom C_midpoint_AB : C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
axiom D_midpoint_BC : D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
axiom E_midpoint_CD : E = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
axiom F_midpoint_DE : F = ((D.1 + E.1) / 2, (D.2 + E.2) / 2)
axiom EF_length : Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) = 5

-- Define the length of AB
noncomputable def AB_length (A B : ℝ × ℝ) : ℝ := 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem to prove
theorem AB_equals_80 : AB_length A B = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_equals_80_l369_36907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_garage_full_spots_l369_36929

theorem parking_garage_full_spots :
  let total_levels : ℕ := 4
  let spots_per_level : ℕ := 100
  let open_spots : List ℕ := [58, 60, 65, 31]
  let full_spots : List ℕ := open_spots.map (λ x => spots_per_level - x)
  full_spots.sum = 186 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_garage_full_spots_l369_36929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equivalences_l369_36934

theorem set_equivalences :
  (∀ (s1 s2 : Set ℤ) (s3 : Set ℕ) (s4_pairs : Set (ℕ × ℕ)) (s5 : Set ℤ),
    s1 = {x : ℤ | |x| ≤ 2} →
    s3 = {x : ℕ | x < 10 ∧ ∃ k : ℕ, x = 3 * k} →
    s2 = {x : ℤ | x = |x| ∧ x < 5} →
    s4_pairs = {p : ℕ × ℕ | p.1 + p.2 = 6 ∧ p.1 ≠ 0 ∧ p.2 ≠ 0} →
    s5 = {-3, -1, 1, 3, 5} →
    s1 = {-2, -1, 0, 1, 2} ∧
    s3 = {3, 6, 9} ∧
    s2 = {0, 1, 2, 3, 4} ∧
    s4_pairs = {(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)} ∧
    s5 = {x : ℤ | ∃ k : ℤ, x = 2 * k - 1 ∧ -1 ≤ k ∧ k ≤ 3}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equivalences_l369_36934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_time_approximately_58_hours_l369_36937

noncomputable def planet_radius : ℝ := 5000
noncomputable def jet_avg_speed : ℝ := 550
noncomputable def wind_effect : ℝ := 50

noncomputable def circumference : ℝ := 2 * Real.pi * planet_radius

noncomputable def first_half_speed : ℝ := jet_avg_speed + wind_effect
noncomputable def second_half_speed : ℝ := jet_avg_speed - wind_effect

noncomputable def first_half_time : ℝ := (circumference / 2) / first_half_speed
noncomputable def second_half_time : ℝ := (circumference / 2) / second_half_speed

noncomputable def total_flight_time : ℝ := first_half_time + second_half_time

theorem flight_time_approximately_58_hours :
  ∃ ε > 0, |total_flight_time - 58| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_time_approximately_58_hours_l369_36937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_triangle_triangle_condition_l369_36968

/-- Vector in R² -/
structure Vec2 where
  x : ℝ
  y : ℝ

/-- Define the dot product of two Vec2 -/
def dot (v w : Vec2) : ℝ := v.x * w.x + v.y * w.y

/-- Define vector subtraction -/
def sub (v w : Vec2) : Vec2 := Vec2.mk (v.x - w.x) (v.y - w.y)

/-- Given vectors -/
def OA : Vec2 := Vec2.mk 2 (-3)
def OB : Vec2 := Vec2.mk (-5) 4
def OC (lambda : ℝ) : Vec2 := Vec2.mk (1 - lambda) (3 * lambda + 2)

theorem right_angled_triangle (lambda : ℝ) : 
  dot (sub OA OB) (sub (OC lambda) OB) = 0 → lambda = 2 := by sorry

theorem triangle_condition (lambda : ℝ) : 
  (∃ (v : Vec2), v ≠ OA ∧ v ≠ OB ∧ v ≠ OC lambda) ↔ lambda ≠ -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_triangle_triangle_condition_l369_36968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_A_worked_alone_is_60_l369_36917

/-- Represents the time (in days) it takes for A and B to complete the work together -/
def total_time_together : ℚ := 40

/-- Represents the time (in days) it takes for A to complete the work alone -/
def total_time_A : ℚ := 80

/-- Represents the time (in days) A and B worked together before B left -/
def time_worked_together : ℚ := 10

/-- Calculates the number of days A worked alone to finish the remaining work -/
noncomputable def days_A_worked_alone : ℚ :=
  let work_rate_together := 1 / total_time_together
  let work_rate_A := 1 / total_time_A
  let work_done_together := work_rate_together * time_worked_together
  let remaining_work := 1 - work_done_together
  remaining_work / work_rate_A

theorem days_A_worked_alone_is_60 : days_A_worked_alone = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_A_worked_alone_is_60_l369_36917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_iff_m_range_l369_36976

/-- The function f(x) defined by a parameter m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (m * x^2 + (m - 3) * x + 1)

/-- The theorem stating the relationship between the range of f and the values of m -/
theorem f_range_iff_m_range :
  ∀ m : ℝ, (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, f m x = y) ↔ (m ∈ Set.Icc 0 1 ∪ Set.Ici 9) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_iff_m_range_l369_36976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_j_l369_36955

noncomputable def j (x : ℝ) : ℝ := 1 / (x + 8) + 1 / (x^2 + 8) + 1 / (x^3 + 8)

theorem domain_of_j :
  {x : ℝ | ∃ y, j x = y} = {x | x < -8 ∨ (-8 < x ∧ x < -2) ∨ -2 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_j_l369_36955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_22_5_percent_l369_36979

/-- Given a principal amount, simple interest, and time, calculate the interest rate. -/
noncomputable def calculate_interest_rate (principal : ℝ) (simple_interest : ℝ) (time : ℝ) : ℝ :=
  (simple_interest * 100) / (principal * time)

/-- Theorem: The interest rate is 22.5% given the specified conditions. -/
theorem interest_rate_is_22_5_percent 
  (principal : ℝ) 
  (simple_interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 400)
  (h2 : simple_interest = 180)
  (h3 : time = 2) :
  calculate_interest_rate principal simple_interest time = 22.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_interest_rate 400 180 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_22_5_percent_l369_36979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_cosine_theorem_l369_36961

theorem chord_cosine_theorem (r : ℝ) (γ δ : ℝ) :
  γ + δ < π →
  8^2 = 2 * r^2 * (1 - Real.cos γ) →
  15^2 = 2 * r^2 * (1 - Real.cos δ) →
  17^2 = 2 * r^2 * (1 - Real.cos (γ + δ)) →
  0 < Real.cos γ →
  ∃ (a b : ℕ), Real.cos γ = a / b ∧ Nat.Coprime a b →
  Real.cos γ = 15 / 17 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_cosine_theorem_l369_36961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_third_point_l369_36994

/-- Golden ratio constant --/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Golden section search method parameters --/
structure GoldenSectionParams where
  lower : ℝ
  upper : ℝ
  x₁ : ℝ
  x₂ : ℝ

/-- Conditions for the golden section search problem --/
def GoldenSectionConditions (p : GoldenSectionParams) : Prop :=
  p.lower = 10 ∧
  p.upper = 110 ∧
  p.x₁ = p.lower + (1 - 1/φ) * (p.upper - p.lower) ∧
  p.x₂ = p.lower + p.upper - p.x₁ ∧
  p.x₁ > p.x₂

/-- Third test point in the golden section search method --/
def x₃ (p : GoldenSectionParams) : ℝ := p.lower + p.x₁ - p.x₂

/-- Theorem stating that the third test point equals 33.6 mL --/
theorem golden_section_third_point (p : GoldenSectionParams) 
  (h : GoldenSectionConditions p) : x₃ p = 33.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_third_point_l369_36994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_price_increase_l369_36946

/-- Calculates the new price of an item after a percentage increase -/
noncomputable def new_price (original_price : ℝ) (percentage_increase : ℝ) : ℝ :=
  original_price * (1 + percentage_increase / 100)

/-- Theorem: The new price of a bicycle that originally cost $220 after a 15% increase is $253 -/
theorem bicycle_price_increase : new_price 220 15 = 253 := by
  -- Unfold the definition of new_price
  unfold new_price
  -- Simplify the arithmetic
  simp [mul_add, mul_div_right_comm]
  -- Check that the result is equal to 253
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_price_increase_l369_36946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_transformations_l369_36931

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the two transformations
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - 1)
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-x + 1)

-- Define symmetry with respect to the line x = 1
def symmetric_about_x_equals_1 (g h : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (1 + (1 - x)) = h x

-- State the theorem
theorem symmetry_of_transformations (f : ℝ → ℝ) :
  symmetric_about_x_equals_1 (g f) (h f) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_transformations_l369_36931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_sum_l369_36935

theorem quadratic_root_difference_sum (a b c : ℝ) (m n : ℕ) :
  a = 2 ∧ b = -10 ∧ c = 3 ∧
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ∧
  (∃ r : ℝ, r > 0 ∧ r = abs (x - y) ∧ r^2 = m ∧ r * n = Real.sqrt m) ∧
  (∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ m)) →
  m + n = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_sum_l369_36935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l369_36974

/-- Calculates the time (in seconds) for a train to cross a pole given its speed and length -/
noncomputable def time_to_cross_pole (speed_kmh : ℝ) (length_m : ℝ) : ℝ :=
  length_m / (speed_kmh * 1000 / 3600)

/-- Theorem stating that a train with speed 50 km/h and length 250 m takes approximately 18 seconds to cross a pole -/
theorem train_crossing_time :
  let speed := (50 : ℝ)
  let length := (250 : ℝ)
  let time := time_to_cross_pole speed length
  ∃ ε > 0, |time - 18| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l369_36974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_product_palindrome_l369_36986

/-- Represents a number consisting of n ones (e.g., 111...1) -/
def ones (n : ℕ) : ℕ := (10^n - 1) / 9

/-- Checks if a natural number is a palindrome -/
def isPalindrome (n : ℕ) : Prop := sorry

theorem ones_product_palindrome (m n : ℕ) :
  isPalindrome (ones m * ones n) ↔ m = n ∧ m ≤ 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_product_palindrome_l369_36986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_m_l369_36947

def m : ℕ := 2^5 * 3^6 * 5^7

theorem number_of_factors_of_m :
  (Finset.filter (· ∣ m) (Finset.range (m + 1))).card = 336 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_m_l369_36947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_C_subset_B_iff_l369_36930

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 9*x + 18 ≥ 0}
noncomputable def B : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (x + 2) + Real.log (9 - x)}

-- Define the set C
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Statement 1
theorem complement_A_intersect_B : 
  (Set.univ \ A) ∩ B = {x : ℝ | 3 < x ∧ x < 6} := by sorry

-- Statement 2
theorem C_subset_B_iff (a : ℝ) : 
  C a ⊆ B ↔ -2 ≤ a ∧ a ≤ 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_C_subset_B_iff_l369_36930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l369_36975

/-- Given point A(1, -2), if vector AB is in the same direction as a=(2, 3) 
    and |AB| = 2√13, then the coordinates of point B are (5, 4). -/
theorem point_B_coordinates :
  let A : ℝ × ℝ := (1, -2)
  let a : ℝ × ℝ := (2, 3)
  ∀ (B : ℝ × ℝ),
    (∃ (k : ℝ), k > 0 ∧ (B.1 - A.1, B.2 - A.2) = (k * a.1, k * a.2)) →
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = (2 * Real.sqrt 13)^2 →
    B = (5, 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l369_36975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_charrua_with_more_digits_l369_36900

/-- Helper function to get the digits of a natural number -/
def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

/-- Helper function to count the number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ :=
  (digits n).length

/-- Definition of a charrua number -/
def is_charrua (n : ℕ) : Prop :=
  (∀ d, d ∈ digits n → d > 1) ∧
  (∀ a b c d, a ∈ digits n → b ∈ digits n → c ∈ digits n → d ∈ digits n →
    (a * b * c * d) ∣ n)

/-- The main theorem -/
theorem exists_charrua_with_more_digits (k : ℕ) :
  ∃ n : ℕ, is_charrua n ∧ num_digits n > k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_charrua_with_more_digits_l369_36900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l369_36997

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.log (5 + x)) / Real.sqrt (2 - x)

-- State the theorem
theorem f_defined_iff (x : ℝ) : 
  (∃ y, f x = y) ↔ -5 < x ∧ x < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l369_36997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coffee_tea_intersection_l369_36965

theorem min_coffee_tea_intersection (coffee_drinkers tea_drinkers : ℝ) 
  (h1 : coffee_drinkers = 0.9)
  (h2 : tea_drinkers = 0.8) : 
  max 0 (coffee_drinkers + tea_drinkers - 1) = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coffee_tea_intersection_l369_36965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l369_36949

theorem problem_statement (x y : ℝ) 
  (h1 : (2 : ℝ)^x + (2 : ℝ)^(y+1) = 1) 
  (m : ℝ) (h2 : m = x + y)
  (n : ℝ) (h3 : n = (1/2 : ℝ)^x + (1/2 : ℝ)^(y-1)) :
  (x < 0 ∧ y < -1) ∧ 
  (∀ z w : ℝ, (2 : ℝ)^z + (2 : ℝ)^(w+1) = 1 → z + w ≤ m) ∧
  n * (2 : ℝ)^m < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l369_36949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l369_36996

/-- Circle with center (2,2) and radius 2 -/
def Circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 4

/-- Line passing through (3,1) with slope k -/
def Line (x y k : ℝ) : Prop := y - 1 = k * (x - 3)

/-- The shortest chord length cut by the line from the circle -/
noncomputable def ShortestChordLength (k : ℝ) : ℝ := 2 * Real.sqrt 2

theorem shortest_chord_length :
  ∀ k : ℝ, (∃ x y : ℝ, Circle x y ∧ Line x y k) →
    (∀ x y : ℝ, Circle x y ∧ Line x y k →
      ∃ x₁ y₁ x₂ y₂ : ℝ, Circle x₁ y₁ ∧ Circle x₂ y₂ ∧ Line x₁ y₁ k ∧ Line x₂ y₂ k ∧
        ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2 : ℝ) ≥ ShortestChordLength k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l369_36996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_cleaning_time_l369_36988

/-- Represents the time needed to clean the remaining sections of a wall -/
def time_to_clean_wall (total_sections : ℕ) (cleaned_sections : ℕ) (time_for_cleaned : ℕ) : ℕ :=
  let remaining_sections := total_sections - cleaned_sections
  let time_per_section := time_for_cleaned / cleaned_sections
  remaining_sections * time_per_section

/-- Theorem stating that the time to clean the remaining wall sections is approximately 170 minutes -/
theorem wall_cleaning_time :
  let total_sections : ℕ := 15
  let cleaned_sections : ℕ := 3
  let time_for_cleaned : ℕ := 34
  (time_to_clean_wall total_sections cleaned_sections time_for_cleaned) = 170 := by
  sorry

#check wall_cleaning_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_cleaning_time_l369_36988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_fourth_quadrant_l369_36958

theorem sin_alpha_fourth_quadrant (α : Real) : 
  (α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) → -- α is in the fourth quadrant
  (Real.tan α = -5/12) →                          -- tan(α) = -5/12
  (Real.sin α = -5/13) :=                         -- sin(α) = -5/13
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_fourth_quadrant_l369_36958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C1_C2_l369_36980

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2/3
def C2 (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- State the theorem
theorem min_distance_C1_C2 :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 6 / 3 ∧
  ∀ (x1 y1 x2 y2 : ℝ), C1 x1 y1 → C2 x2 y2 →
    distance x1 y1 x2 y2 ≥ min_dist :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C1_C2_l369_36980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_student_position_l369_36972

/-- Represents the game with students in a circle -/
def StudentGame (n : ℕ) : ℕ → ℕ
| 0 => n  -- Initial number of students
| (k+1) => ((StudentGame n k) + 2) / 3  -- Number of students after k+1 rounds

/-- The position of the last remaining student -/
noncomputable def LastStudent (n : ℕ) : ℕ :=
  (3^(Nat.log 3 n).succ - 1) / 2

/-- Theorem stating that for 1991 students, the last remaining student was at position 1093 -/
theorem last_student_position :
  LastStudent 1991 = 1093 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_student_position_l369_36972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_products_l369_36933

noncomputable section

variable (a b : ℝ × ℝ)

noncomputable def angle_between (u v : ℝ × ℝ) : ℝ := 
  Real.arccos ((u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1^2 + u.2^2) * Real.sqrt (v.1^2 + v.2^2)))

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem vector_products (h1 : angle_between a b = 2 * Real.pi / 3)
                        (h2 : magnitude a = 4)
                        (h3 : magnitude b = 2) :
  (a.1 * b.1 + a.2 * b.2 = -4) ∧
  ((a.1 + b.1) * (a.1 - 2*b.1) + (a.2 + b.2) * (a.2 - 2*b.2) = 12) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_products_l369_36933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skateboard_price_is_1500_l369_36987

/-- The price of a skateboard given its partial payment -/
noncomputable def skateboard_price (partial_payment : ℚ) (payment_percentage : ℚ) : ℚ :=
  partial_payment / (payment_percentage / 100)

/-- Theorem: The price of the skateboard is $1500 -/
theorem skateboard_price_is_1500 :
  skateboard_price 300 20 = 1500 := by
  -- Unfold the definition of skateboard_price
  unfold skateboard_price
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_skateboard_price_is_1500_l369_36987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l369_36918

/-- The function f(x) defined as ka^x - a^(-x) --/
noncomputable def f (k a x : ℝ) : ℝ := k * (a^x) - a^(-x)

/-- The theorem stating the properties of f and its solution set --/
theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x, f 1 a x = -f 1 a (-x)) : 
  (∃ k : ℝ, k = 1 ∧ ∀ x, f k a x = f 1 a x) ∧
  (f 1 a 1 > 0 → ∀ x, f 1 a (x^2 + 2*x) + f 1 a (x - 4) > 0 ↔ x > 1 ∨ x < -4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l369_36918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_for_divisibility_by_10_l369_36941

theorem least_subtraction_for_divisibility_by_10 (n : ℕ) (hn : n = 724946) :
  ∃ (k : ℕ), k = 6 ∧ 
  (∀ (m : ℕ), m < k → ¬(10 ∣ (n - m))) ∧
  (10 ∣ (n - k)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_for_divisibility_by_10_l369_36941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_2x_fourier_series_hermite_l369_36990

-- Define the Hermite polynomials
noncomputable def hermite_polynomial (n : ℕ) : ℝ → ℝ :=
  λ x => (-1)^n * Real.exp (x^2) * (deriv^[n] (λ y => Real.exp (-y^2))) x

-- Define the scalar product for the orthogonal basis
noncomputable def scalar_product (u v : ℝ → ℝ) : ℝ :=
  ∫ (x : ℝ), u x * v x * Real.exp (-x^2)

-- Define the Fourier series expansion
noncomputable def fourier_series (f : ℝ → ℝ) (basis : ℕ → ℝ → ℝ) (coeff : ℕ → ℝ) : ℝ → ℝ :=
  λ x => ∑' n, coeff n * basis n x

-- State the theorem
theorem exp_2x_fourier_series_hermite :
  ∃ (coeff : ℕ → ℝ), 
    (∀ x, fourier_series (λ x => Real.exp (2*x)) hermite_polynomial coeff x = 
           Real.exp 1 * ∑' n, (1 / Nat.factorial n) * hermite_polynomial n x) ∧
    (∀ n, coeff n = Real.exp 1 / Nat.factorial n) := by
  sorry

-- Additional lemma to show the convergence of the integral
lemma integral_convergence :
  ∃ C, ∫ (x : ℝ), Real.exp (4*x - x^2) < C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_2x_fourier_series_hermite_l369_36990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_intercept_l369_36952

theorem line_slope_intercept (m n : ℝ) : 
  (∀ x y : ℝ, m * x + n * y + 3 = 0 → (x = 0 → y = -1)) →
  (∀ x y : ℝ, m * x + n * y + 3 = 0 → -m / n = 2 * (1 / 2)) →
  m = -4 ∧ n = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_intercept_l369_36952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l369_36948

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define what it means for P to be the midpoint of AB
def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Define what it means for A and B to be on the circle
def on_my_circle (point : ℝ × ℝ) : Prop := my_circle point.1 point.2

-- Theorem statement
theorem chord_equation (A B : ℝ × ℝ) 
  (h1 : is_midpoint P A B) 
  (h2 : on_my_circle A) 
  (h3 : on_my_circle B) : 
  ∃ (k : ℝ), ∀ (x y : ℝ), (x - y - 3 = k) ↔ (∃ t : ℝ, x = A.1 + t * (B.1 - A.1) ∧ y = A.2 + t * (B.2 - A.2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l369_36948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_prize_class1_l369_36925

/-- Represents the number of students in each class -/
def class_sizes : Fin 4 → ℕ
| 0 => 30
| 1 => 40
| 2 => 20
| 3 => 10

/-- Total number of students -/
def total_students : ℕ := (Finset.range 4).sum (fun i => class_sizes i)

/-- Number of students selected for the competition -/
def selected_students : ℕ := 10

/-- Number of preset questions -/
def total_questions : ℕ := 10

/-- Number of questions each student answers -/
def questions_answered : ℕ := 4

/-- Minimum number of correct answers to receive a prize -/
def min_correct_for_prize : ℕ := 3

/-- Probability of a Class 1 student answering a question correctly -/
def prob_correct_class1 : ℚ := 1/3

/-- Number of students selected from Class 1 -/
def class1_selected : ℕ := 3

/-- Theorem: Probability that at least one student from Class 1 will receive a prize -/
theorem prob_at_least_one_prize_class1 : 
  (1 : ℚ) - (8/9)^3 = 217/729 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_prize_class1_l369_36925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_per_year_l369_36902

/-- Calculates the compound interest --/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (frequency : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

/-- Represents the financial transaction --/
structure Transaction where
  principal : ℝ
  time : ℝ
  borrowRate : ℝ
  lendRate : ℝ
  borrowFrequency : ℝ
  lendFrequency : ℝ

/-- Calculates the gain per year for a given transaction --/
noncomputable def gainPerYear (t : Transaction) : ℝ :=
  let borrowed := compoundInterest t.principal t.borrowRate t.borrowFrequency t.time
  let lent := compoundInterest t.principal t.lendRate t.lendFrequency t.time
  (lent - borrowed) / t.time

theorem transaction_gain_per_year :
  let t : Transaction := {
    principal := 7000,
    time := 2,
    borrowRate := 0.04,
    lendRate := 0.06,
    borrowFrequency := 1,
    lendFrequency := 2
  }
  ∃ (ε : ℝ), abs (gainPerYear t - 153.65) < ε ∧ ε < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_per_year_l369_36902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_neighbor_difference_l369_36950

/-- Represents a chessboard with numbers placed on it -/
def Chessboard (n : ℕ+) := Fin n → Fin n → Fin (n^2)

/-- Two positions on the chessboard are neighbors if they differ by 1 in exactly one coordinate -/
def are_neighbors {n : ℕ+} (p q : Fin n × Fin n) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ p.2.val = q.2.val + 1)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ p.1.val = q.1.val + 1))

/-- Main theorem: For any n × n chessboard with numbers 1 to n^2,
    there exist two neighboring squares whose numbers differ by at least n -/
theorem chessboard_neighbor_difference {n : ℕ+} (board : Chessboard n) :
  ∃ (p q : Fin n × Fin n), are_neighbors p q ∧ 
    (n : ℕ) ≤ (board p.1 p.2).val - (board q.1 q.2).val ∨ 
    (n : ℕ) ≤ (board q.1 q.2).val - (board p.1 p.2).val :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_neighbor_difference_l369_36950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_line_l369_36903

-- Define the curve and line
noncomputable def curve (x : ℝ) : ℝ := Real.log x
def line (x : ℝ) : ℝ := 2 * x + 6

-- Define the distance function between a point on the curve and a point on the line
noncomputable def distance (x y : ℝ) : ℝ :=
  |curve x - line y|

-- State the theorem
theorem min_distance_curve_line :
  ∃ (d : ℝ), d = (7 + Real.log 2) * Real.sqrt 5 / 5 ∧
  ∀ (x y : ℝ), x > 0 → distance x y ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_line_l369_36903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l369_36973

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 + f y) = x * f x + y) : 
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l369_36973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_quadrant_sufficient_not_necessary_l369_36956

open Real

-- Define the property of being in the third quadrant
def in_third_quadrant (θ : ℝ) : Prop :=
  Real.sin θ < 0 ∧ Real.cos θ < 0

-- Define the property of sinθtanθ < 0
def sin_tan_product_negative (θ : ℝ) : Prop :=
  Real.sin θ * Real.tan θ < 0

-- Theorem statement
theorem third_quadrant_sufficient_not_necessary :
  (∀ θ : ℝ, in_third_quadrant θ → sin_tan_product_negative θ) ∧
  (∃ θ : ℝ, ¬in_third_quadrant θ ∧ sin_tan_product_negative θ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_quadrant_sufficient_not_necessary_l369_36956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_one_three_two_l369_36995

-- Define the nabla operation
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- State the theorem
theorem nabla_one_three_two :
  nabla (nabla 1 3) 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_one_three_two_l369_36995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bullet_train_crossing_time_l369_36940

/-- The time it takes for two trains to cross each other -/
noncomputable def crossingTime (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / ((speed1 + speed2) * 1000 / 3600)

/-- Theorem stating the crossing time for the given problem -/
theorem bullet_train_crossing_time :
  let length1 : ℝ := 140
  let length2 : ℝ := 200
  let speed1 : ℝ := 60
  let speed2 : ℝ := 40
  ∃ ε > 0, |crossingTime length1 length2 speed1 speed2 - 12.24| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bullet_train_crossing_time_l369_36940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_C_to_line_l_l369_36963

noncomputable section

/-- The distance between a point and a line in 2D space -/
def distance_point_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- The parametric equations of line l -/
def line_l (t : ℝ) : ℝ × ℝ := (t, t + 1)

/-- The parametric equations of circle C -/
def circle_C (θ : ℝ) : ℝ × ℝ := (Real.cos θ + 1, Real.sin θ)

/-- The center of circle C -/
def center_C : ℝ × ℝ := (1, 0)

theorem distance_center_C_to_line_l :
  distance_point_line (center_C.1) (center_C.2) 1 (-1) 1 = Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_C_to_line_l_l369_36963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l369_36970

open Real

-- Define variables
variable (x₀ y₀ x₁ y₁ x y : ℝ)

-- Define the two points
def M₀ (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, y₀)
def M₁ (x₁ y₁ : ℝ) : ℝ × ℝ := (x₁, y₁)

-- Define the condition that the points are distinct
def distinct_points (x₀ y₀ x₁ y₁ : ℝ) : Prop := x₁ ≠ x₀ ∧ y₁ ≠ y₀

-- Define the equation of the line
def line_equation (x₀ y₀ x₁ y₁ x y : ℝ) : Prop := 
  (y - y₀) / (y₁ - y₀) = (x - x₀) / (x₁ - x₀)

-- Theorem statement
theorem line_through_points (x₀ y₀ x₁ y₁ x y : ℝ) 
  (h : distinct_points x₀ y₀ x₁ y₁) : 
  line_equation x₀ y₀ x₁ y₁ x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l369_36970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_captain_total_coins_l369_36999

-- Define the number of pirates
def num_pirates : ℕ := 15

-- Define the initial captain's take
def initial_take : ℕ := 120

-- Define the captain's final take
def final_take : ℕ := 45

-- Define the function to calculate the remaining coins after each pirate's turn
def remaining_coins (initial_coins : ℕ) (pirate : ℕ) : ℚ :=
  (initial_coins - initial_take : ℚ) * (Finset.prod (Finset.range pirate) (fun k => (num_pirates - k - 1 : ℚ) / num_pirates))

-- Define the function to calculate the smallest initial number of coins
noncomputable def smallest_initial_coins : ℕ :=
  Nat.lcm (15^14) (Nat.factorial 14) + initial_take

-- Theorem statement
theorem captain_total_coins :
  ⌊remaining_coins smallest_initial_coins (num_pirates - 1)⌋ + initial_take + final_take = 3850 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_captain_total_coins_l369_36999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_plaza_diameter_l369_36983

/-- The diameter of the outer edge of a circular plaza -/
noncomputable def outerDiameter (walkingPathWidth : ℝ) (gardenWidth : ℝ) (fountainDiameter : ℝ) : ℝ :=
  2 * (fountainDiameter / 2 + gardenWidth + walkingPathWidth)

/-- Theorem stating the diameter of the outer edge of the circular plaza -/
theorem circular_plaza_diameter :
  outerDiameter 10 12 14 = 58 := by
  -- Unfold the definition of outerDiameter
  unfold outerDiameter
  -- Simplify the arithmetic
  simp [add_assoc, mul_add, mul_comm]
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_plaza_diameter_l369_36983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisibility_l369_36945

def sequence_a : ℕ → ℤ
  | 0 => 2
  | 1 => 4
  | (n + 2) => (sequence_a n * sequence_a (n + 1)) / 2 + sequence_a n + sequence_a (n + 1)

theorem prime_divisibility (p : ℕ) (hp : p.Prime) (hp2 : p > 2) :
  ∃ m : ℕ, m > 0 ∧ (p : ℤ) ∣ (sequence_a m - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisibility_l369_36945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_formula_l369_36981

noncomputable def f (x : ℝ) : ℝ := x / (1 + x)

noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 => f  -- Adding case for 0 to fix missing case error
| 1 => f
| n + 1 => λ x => f (f_n n x)

theorem f_n_formula (n : ℕ) (x : ℝ) (h : x ≥ 0) (h_n : n > 0) :
  f_n n x = x / (1 + n * x) := by
  sorry

#check f_n_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_formula_l369_36981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_l369_36978

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem sine_symmetry (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < π) 
  (h3 : is_symmetric_about (λ x ↦ Real.sin (2 * x + φ)) (π / 3)) :
  φ = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_l369_36978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l369_36921

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 - 5*x

-- Define the proposed inverse function g
noncomputable def g (x : ℝ) : ℝ := (4 - x) / 5

-- Theorem statement
theorem f_inverse_is_g : 
  (∀ x, f (g x) = x) ∧ (∀ x, g (f x) = x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l369_36921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_center_and_P_line_perpendicular_to_PC_l369_36954

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define point P
def P : ℝ × ℝ := (2, 2)

-- Define the center of the circle
def center : ℝ × ℝ := (1, 0)

-- Define a line passing through two points
def line_equation (p1 p2 : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p1.2) * (p2.1 - p1.1) = (x - p1.1) * (p2.2 - p1.2)

-- Theorem 1: Line passing through center and P
theorem line_through_center_and_P :
  ∀ x y : ℝ, line_equation center P x y ↔ 2*x - y - 2 = 0 :=
by sorry

-- Theorem 2: Line perpendicular to PC and passing through P
theorem line_perpendicular_to_PC :
  ∀ x y : ℝ, (line_equation P (x, y) x y ∧ 
    (x - P.1) * (center.1 - P.1) + (y - P.2) * (center.2 - P.2) = 0) 
    ↔ x + 2*y - 6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_center_and_P_line_perpendicular_to_PC_l369_36954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_implies_a_eq_one_l369_36938

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = ln(x + √(a + x^2)) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (x + Real.sqrt (a + x^2))

/-- If f(x) = ln(x + √(a + x^2)) is an odd function, then a = 1 -/
theorem f_odd_implies_a_eq_one (a : ℝ) :
  IsOdd (f a) → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_implies_a_eq_one_l369_36938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colors_l369_36919

/-- Represents a coloring of a 4x4 grid -/
def Coloring := Fin 4 → Fin 4 → ℕ

/-- Checks if a 1x3 rectangle contains at least two cells of the same color -/
def valid_1x3 (c : Coloring) (i j : Fin 4) : Prop :=
  ∃ k l : Fin 4, k ≠ l ∧ c i k = c i l ∧ 
  (k.val < j.val + 3 ∧ l.val < j.val + 3) ∧ 
  (k.val ≥ j.val ∧ l.val ≥ j.val)

/-- A coloring is valid if all 1x3 rectangles contain at least two cells of the same color -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ i j : Fin 4, valid_1x3 c i j

/-- The number of distinct colors used in a coloring -/
def num_colors (c : Coloring) : ℕ :=
  Finset.card (Finset.image (fun p => c p.1 p.2) (Finset.univ.product Finset.univ))

/-- The maximum number of colors that can be used in a valid coloring is 9 -/
theorem max_colors :
  ∀ c : Coloring, valid_coloring c → num_colors c ≤ 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colors_l369_36919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l369_36924

-- Define the complex number z
noncomputable def z : ℂ := Complex.abs (Complex.I * Real.sqrt 3 - 1) + 1 / (1 + Complex.I)

-- Theorem statement
theorem imaginary_part_of_z :
  Complex.im z = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l369_36924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l369_36912

noncomputable def g (A : ℝ) : ℝ :=
  (Real.sin A * (5 * Real.cos A ^ 2 + Real.tan A ^ 2 + 2 * Real.sin A ^ 2 + Real.sin A ^ 2 * Real.cos A ^ 2)) /
  (Real.tan A * (1 / Real.sin A - Real.sin A * Real.tan A))

theorem g_range :
  ∀ A : ℝ, (∀ n : ℤ, A ≠ n * Real.pi / 2) →
  ∃ y ∈ Set.Ioo 3 8, g A = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l369_36912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l369_36953

theorem beta_value (α β : Real) 
  (h1 : Real.cos α = 1/7)
  (h2 : Real.cos (α + β) = -11/14)
  (h3 : 0 < α ∧ α < Real.pi/2)
  (h4 : Real.pi/2 < α + β ∧ α + β < Real.pi) : 
  β = Real.pi/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l369_36953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rs_value_l369_36909

theorem rs_value (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h1 : r^2 + s^2 = 1) (h2 : r^4 + s^4 = 15/16) : r * s = Real.sqrt 2 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rs_value_l369_36909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l369_36977

open Real

-- Define the function f(x)
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := x + b / x

-- Define the derivative of f(x)
noncomputable def f_deriv (b : ℝ) (x : ℝ) : ℝ := 1 - b / (x^2)

theorem f_monotone_increasing (b : ℝ) :
  (∃ x ∈ Set.Ioo 1 2, f_deriv b x = 0) →
  StrictMonoOn (f b) (Set.Ioi 2) := by
  sorry

#check f_monotone_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l369_36977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_monotonic_condition_zeros_of_g_l369_36914

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x + Real.log x

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 1 - a / (x^2) + 1 / x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f' a x - x

-- Theorem for part (I)
theorem extremum_condition (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x > 0 ∧ |x - 1| < ε → f a x ≥ f a 1) ↔ a = 2 :=
sorry

-- Theorem for part (II)
theorem monotonic_condition (a : ℝ) :
  (∀ (x y : ℝ), 1 < x ∧ x < y ∧ y < 2 → f a x < f a y) ↔ a ≤ 2 :=
sorry

-- Theorem for part (III)
theorem zeros_of_g (a : ℝ) :
  (a > 1 → ¬∃ (x : ℝ), x > 0 ∧ g a x = 0) ∧
  ((a = 1 ∨ a ≤ 0) → ∃! (x : ℝ), x > 0 ∧ g a x = 0) ∧
  (0 < a ∧ a < 1 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≠ y ∧ g a x = 0 ∧ g a y = 0 ∧
    ∀ (z : ℝ), z > 0 ∧ g a z = 0 → z = x ∨ z = y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_monotonic_condition_zeros_of_g_l369_36914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_diagonals_l369_36913

-- Define the pentagon PQRST
structure Pentagon where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  T : ℝ × ℝ

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the conditions for the pentagon
def inscribed_pentagon (pent : Pentagon) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    distance center pent.P = radius ∧
    distance center pent.Q = radius ∧
    distance center pent.R = radius ∧
    distance center pent.S = radius ∧
    distance center pent.T = radius

-- Define the side length conditions
def side_conditions (pent : Pentagon) : Prop :=
  distance pent.P pent.Q = 4 ∧
  distance pent.R pent.S = 4 ∧
  distance pent.Q pent.R = 11 ∧
  distance pent.S pent.T = 11 ∧
  distance pent.P pent.T = 15

-- Theorem statement
theorem sum_of_diagonals (pent : Pentagon) 
  (h1 : inscribed_pentagon pent) 
  (h2 : side_conditions pent) : 
  distance pent.P pent.R + distance pent.P pent.S + 
  distance pent.Q pent.S + distance pent.Q pent.T + 
  distance pent.R pent.T = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_diagonals_l369_36913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l369_36905

theorem solution_set (a : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ x₅ : ℝ, 
    x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧ x₅ ≥ 0 ∧
    (1 * x₁ + 2 * x₂ + 3 * x₃ + 4 * x₄ + 5 * x₅ = a) ∧
    (1^3 * x₁ + 2^3 * x₂ + 3^3 * x₃ + 4^3 * x₄ + 5^3 * x₅ = a^2) ∧
    (1^5 * x₁ + 2^5 * x₂ + 3^5 * x₃ + 4^5 * x₄ + 5^5 * x₅ = a^3)) →
  a ∈ ({0, 1, 4, 9, 16, 25} : Set ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l369_36905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hours_to_work_correct_l369_36928

/-- Calculates the number of hours James needs to work to pay for the food fight damages -/
def hours_to_work (minimum_wage : ℚ) (meat_price meat_amount : ℚ) (fruit_price fruit_amount : ℚ)
  (cheese_price cheese_amount : ℚ) (bread_price bread_amount : ℚ) (milk_price milk_amount : ℚ)
  (juice_price juice_amount : ℚ) (tax_rate : ℚ) (cleaning_cost : ℚ) (interest_rate : ℚ)
  (penalty : ℚ) (janitor_rate : ℚ) (janitor_hours : ℚ) : ℕ :=
  let food_cost := meat_price * meat_amount + fruit_price * fruit_amount +
    cheese_price * cheese_amount + bread_price * bread_amount +
    milk_price * milk_amount + juice_price * juice_amount
  let taxed_food_cost := food_cost * (1 + tax_rate)
  let subtotal := taxed_food_cost + cleaning_cost
  let with_interest := subtotal * (1 + interest_rate)
  let with_penalty := with_interest + penalty
  let janitor_pay := janitor_rate * janitor_hours * (3/2)
  let total_cost := with_penalty + janitor_pay
  (total_cost / minimum_wage).ceil.toNat

theorem hours_to_work_correct : 
  hours_to_work 8 5 20 4 15 (7/2) 25 (3/2) 60 2 20 6 5 (7/100) 15 (1/20) 50 10 10 = 85 := by
  sorry

#eval hours_to_work 8 5 20 4 15 (7/2) 25 (3/2) 60 2 20 6 5 (7/100) 15 (1/20) 50 10 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hours_to_work_correct_l369_36928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_upper_bound_l369_36942

open Real

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := a * (x^2 + 1) + log x

-- State the theorem
theorem m_upper_bound 
  (h_a : ∀ a, a ∈ Set.Ioo (-4) (-2) → 
    ∀ x, x ∈ Set.Icc 1 3 → 
      ∀ m : ℝ, m * a - f a x > a^2) : 
  ∀ m : ℝ, (∀ a, a ∈ Set.Ioo (-4) (-2) → 
    ∀ x, x ∈ Set.Icc 1 3 → 
      m * a - f a x > a^2) → 
  m ≤ -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_upper_bound_l369_36942
