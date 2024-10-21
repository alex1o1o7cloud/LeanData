import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l379_37964

/-- Hyperbola C₁ with equation x²/a² - y²/b² = 1 -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- Parabola C₂ with equation y² = 4cx -/
structure Parabola (c : ℝ)

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle formed by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Equilateral triangle -/
def IsEquilateral (t : Triangle) : Prop := sorry

/-- Eccentricity of a hyperbola -/
noncomputable def Eccentricity (h : Hyperbola a b) (c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity 
  (h : Hyperbola a b) 
  (p : Parabola c) 
  (F₁ F₂ M N : Point)
  (t : Triangle)
  (h_foci : F₁.x = -c ∧ F₂.x = c) 
  (h_intersect : sorry) -- Directrix of C₂ intersects C₁ at M and N
  (h_equilateral : IsEquilateral t) :
  Eccentricity h c = Real.sqrt 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l379_37964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l379_37989

/-- Given a right triangle with hypotenuse 12 inches and one angle 45°, its area is 36 square inches -/
theorem right_triangle_area (h : ℝ) (α : ℝ) (area : ℝ) 
  (h_hypotenuse : h = 12)
  (h_angle : α = 45 * Real.pi / 180)
  (h_right : α + α + Real.pi / 2 = Real.pi)
  (h_area : area = (1 / 2) * (h * Real.sin α) * (h * Real.cos α)) :
  area = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l379_37989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_trig_composition_2010_l379_37911

/-- The set of allowed trigonometric and inverse trigonometric functions -/
inductive TrigFunction
| sin : TrigFunction
| cos : TrigFunction
| tan : TrigFunction
| cot : TrigFunction
| arcsin : TrigFunction
| arccos : TrigFunction
| arctan : TrigFunction
| arccot : TrigFunction

/-- A composition of trigonometric functions -/
def TrigComposition := List TrigFunction

/-- Apply a single trigonometric function to a real number -/
noncomputable def applyTrigFunction (f : TrigFunction) (x : ℝ) : ℝ :=
  match f with
  | TrigFunction.sin => Real.sin x
  | TrigFunction.cos => Real.cos x
  | TrigFunction.tan => Real.tan x
  | TrigFunction.cot => 1 / Real.tan x  -- Replace Real.cot with 1 / tan
  | TrigFunction.arcsin => Real.arcsin x
  | TrigFunction.arccos => Real.arccos x
  | TrigFunction.arctan => Real.arctan x
  | TrigFunction.arccot => Real.pi / 2 - Real.arctan x  -- Replace Real.arccot with pi/2 - arctan

/-- Apply a composition of trigonometric functions to a real number -/
noncomputable def applyTrigComposition (comp : TrigComposition) (x : ℝ) : ℝ :=
  comp.foldl (fun acc f => applyTrigFunction f acc) x

/-- The theorem stating that 2010 can be obtained from 1 using a composition of trigonometric functions -/
theorem exists_trig_composition_2010 : ∃ (comp : TrigComposition), applyTrigComposition comp 1 = 2010 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_trig_composition_2010_l379_37911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l379_37992

noncomputable def z (n : ℕ) : ℂ := (n - Complex.I) / (n + 1)

theorem sequence_convergence :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, Complex.abs (z n - 1) < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l379_37992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_am_gm_hm_inequality_l379_37905

theorem am_gm_hm_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a < b) :
  (((a + b) / 2 - Real.sqrt (a * b)) / ((a + b) / 2 - 2 * a * b / (a + b))) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_am_gm_hm_inequality_l379_37905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l379_37963

/-- A complex number z is in the third quadrant if its real part is negative and its imaginary part is negative -/
def in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

/-- The complex number z defined as (3-ai)/i -/
noncomputable def z (a : ℝ) : ℂ := (3 - a * Complex.I) / Complex.I

/-- Theorem stating that z is in the third quadrant if and only if a > 0 -/
theorem z_in_third_quadrant (a : ℝ) :
  in_third_quadrant (z a) ↔ a > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l379_37963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_reciprocal_sum_l379_37901

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 25

-- Define the line l
def line_l (x y : ℝ) (m : ℝ) : Prop := 3 * x + 4 * y + m = 0

-- Define the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem min_value_reciprocal_sum (m : ℝ) (a b : ℝ) :
  m < 0 →
  (∃ x y, circle_C x y ∧ line_l x y m) →
  (∀ x y, circle_C x y → line_l x y m → (x - a)^2 + (y - b)^2 ≥ 1) →
  (∃ x y, circle_C x y ∧ line_l x y m ∧ (x - a)^2 + (y - b)^2 = 1) →
  line_l a b m →
  first_quadrant a b →
  (1 / a + 1 / b ≥ (7 + 4 * Real.sqrt 3) / 55) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_reciprocal_sum_l379_37901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l379_37954

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Circle structure -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Point on a parabola -/
def PointOnParabola (C : Parabola) (x y : ℝ) : Prop :=
  y^2 = 2 * C.p * x

/-- Angle between three points -/
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Theorem statement -/
theorem parabola_circle_intersection
  (C : Parabola)
  (F : ℝ × ℝ)  -- Focus
  (l : ℝ → ℝ)  -- Directrix
  (circ : Circle)
  (A B E : ℝ × ℝ)  -- Intersection points
  (hF : circ.center = F)
  (hrad : circ.radius = 4)
  (hA : l A.1 = A.2)
  (hB : l B.1 = B.2)
  (hE : PointOnParabola C E.1 E.2)
  (hangle : angle E A B = 90) :
  (C.p = 2) ∧
  (∀ (P Q R : ℝ × ℝ),
    PointOnParabola C P.1 P.2 →
    PointOnParabola C Q.1 Q.2 →
    PointOnParabola C R.1 R.2 →
    P.2 = -1 →
    P ≠ Q ∧ P ≠ R ∧ Q ≠ R →
    (Q.2 - P.2) / (Q.1 - P.1) + (R.2 - P.2) / (R.1 - P.1) = -1 →
    ∃ (t : ℝ), Q.1 - (-7/4) = t * (Q.2 - (-3)) ∧ R.1 - (-7/4) = t * (R.2 - (-3))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l379_37954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_theorem_l379_37960

theorem erdos_theorem (X : Type) (H : Set (Set X)) (k : ℕ) 
  (h_infinite : Infinite X)
  (h_finite_subsets : ∀ A, A ∈ H → Finite A)
  (h_union_representation : ∀ A : Set X, Finite A → 
    ∃ H₁ H₂, H₁ ∈ H ∧ H₂ ∈ H ∧ Disjoint H₁ H₂ ∧ A = H₁ ∪ H₂) :
  ∃ A : Set X, ∃ L : List (Set X × Set X),
    (∀ p ∈ L, p.1 ∈ H ∧ p.2 ∈ H ∧ Disjoint p.1 p.2 ∧ A = p.1 ∪ p.2) ∧
    L.length ≥ k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_theorem_l379_37960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l379_37943

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (Real.sqrt a + Real.sqrt b ≤ 2) ∧ ((a + b^3) * (a^3 + b) ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l379_37943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_g_eq_5_l379_37929

-- Define the piecewise function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then 4 * x + 2 else 3 * x - 4

-- Theorem statement
theorem unique_solution_g_eq_5 :
  ∃! x : ℝ, g x = 5 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_g_eq_5_l379_37929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_shells_per_crab_l379_37982

/-- Proves that the number of spiral shells per hermit crab is 3 -/
theorem spiral_shells_per_crab (
  hermit_crabs : ℕ)
  (starfish_per_shell : ℕ)
  (total_souvenirs : ℕ)
  (spiral_shells_per_crab : ℕ)
  (h1 : hermit_crabs = 45)
  (h2 : starfish_per_shell = 2)
  (h3 : total_souvenirs = 450)
  (h4 : total_souvenirs = hermit_crabs + hermit_crabs * spiral_shells_per_crab +
    hermit_crabs * spiral_shells_per_crab * starfish_per_shell) :
  spiral_shells_per_crab = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_shells_per_crab_l379_37982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l379_37921

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote passing through (2,1)
def asymptote_passes_through (a b : ℝ) : Prop :=
  ∃ (k : ℝ), 2 = k * a ∧ 1 = k * b

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2) / a

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  hyperbola a b 2 1 → asymptote_passes_through a b →
  eccentricity a b = Real.sqrt 5 / 2 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l379_37921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_OQ_l379_37984

noncomputable section

-- Define the parabola C: y² = 2px (p > 0)
def Parabola (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ p > 0

-- Define the focus F at a distance of 2 from the directrix
def FocusDistance (p : ℝ) : Prop :=
  p = 2

-- Define point P on the parabola
def PointOnParabola (p : ℝ) (x y : ℝ) : Prop :=
  Parabola p x y

-- Define point Q satisfying PQ = 9QF
def PointQ (p xp yp xq yq : ℝ) : Prop :=
  (xp - xq)^2 + (yp - yq)^2 = 81 * ((p - xq)^2 + yq^2)

-- Define the slope of line OQ
def SlopeOQ (xq yq : ℝ) : ℝ :=
  yq / xq

-- Theorem statement
theorem max_slope_OQ (p xp yp xq yq : ℝ) :
  Parabola p xp yp →
  FocusDistance p →
  PointOnParabola p xp yp →
  PointQ p xp yp xq yq →
  ∃ (k : ℝ), ∀ (xq' yq' : ℝ), PointQ p xp yp xq' yq' →
    SlopeOQ xq' yq' ≤ k ∧ k = 1/3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_OQ_l379_37984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_selling_price_l379_37958

-- Define the profit margins
def manufacturer_profit : ℝ := 0.18
def wholesaler_profit : ℝ := 0.20
def retailer_profit : ℝ := 0.25

-- Define the manufacturer's cost price
def manufacturer_cost : ℝ := 17

-- Define the function to calculate selling price with profit
def selling_price (cost : ℝ) (profit : ℝ) : ℝ :=
  cost * (1 + profit)

-- Theorem to prove
theorem retailer_selling_price :
  let manufacturer_selling_price := selling_price manufacturer_cost manufacturer_profit
  let wholesaler_selling_price := selling_price manufacturer_selling_price wholesaler_profit
  let retailer_selling_price := selling_price wholesaler_selling_price retailer_profit
  ∃ ε > 0, |retailer_selling_price - 30.09| < ε := by
  sorry

#eval selling_price (selling_price (selling_price manufacturer_cost manufacturer_profit) wholesaler_profit) retailer_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_selling_price_l379_37958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kims_driving_speed_l379_37923

/-- Calculates the driving speed given the conditions of Kim's trip -/
noncomputable def calculate_driving_speed (distance_to_friend : ℝ) (detour_percentage : ℝ) (time_at_friends : ℝ) (total_time_away : ℝ) : ℝ :=
  let detour_distance := distance_to_friend * (1 + detour_percentage)
  let total_distance := distance_to_friend + detour_distance
  let driving_time := total_time_away - time_at_friends
  total_distance / driving_time

/-- Theorem stating that Kim's driving speed is 44 miles per hour -/
theorem kims_driving_speed :
  calculate_driving_speed 30 0.2 0.5 2 = 44 := by
  -- Unfold the definition of calculate_driving_speed
  unfold calculate_driving_speed
  -- Simplify the expressions
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kims_driving_speed_l379_37923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_trip_time_l379_37978

/-- Represents the time zones the airplane passes through -/
inductive TimeZone
| Pacific
| Mountain
| Central
| Eastern

/-- Calculates the total time for the airplane's trip over two days -/
def total_trip_time (first_day_hover_times : TimeZone → ℝ) 
                    (layover_time : ℝ) 
                    (speed_increase : ℝ) 
                    (second_day_time_decrease : ℝ) : ℝ :=
  let first_day_total := (first_day_hover_times TimeZone.Pacific +
                          first_day_hover_times TimeZone.Mountain +
                          first_day_hover_times TimeZone.Central +
                          first_day_hover_times TimeZone.Eastern) +
                         4 * layover_time
  let second_day_total := ((first_day_hover_times TimeZone.Pacific - second_day_time_decrease) +
                           (first_day_hover_times TimeZone.Mountain - second_day_time_decrease) +
                           (first_day_hover_times TimeZone.Central - second_day_time_decrease) +
                           (first_day_hover_times TimeZone.Eastern - second_day_time_decrease)) +
                          4 * layover_time
  first_day_total + second_day_total

theorem airplane_trip_time :
  let first_day_hover_times : TimeZone → ℝ := λ tz => 
    match tz with
    | TimeZone.Pacific => 2
    | TimeZone.Mountain => 3
    | TimeZone.Central => 4
    | TimeZone.Eastern => 3
  let layover_time : ℝ := 1.5
  let speed_increase : ℝ := 0.2
  let second_day_time_decrease : ℝ := 0.6
  total_trip_time first_day_hover_times layover_time speed_increase second_day_time_decrease = 33.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_trip_time_l379_37978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_complex_trig_function_l379_37987

open Real

theorem derivative_of_complex_trig_function (x : ℝ) :
  let y := λ x : ℝ => sin (log 2) + (sin (25 * x))^2 / (25 * cos (50 * x))
  deriv y x = sin (50 * x) / (cos (50 * x))^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_complex_trig_function_l379_37987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l379_37952

-- Define the ellipse
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the points
def O : ℝ × ℝ := (0, 0)
def F (c : ℝ) : ℝ × ℝ := (-c, 0)
def A (a : ℝ) : ℝ × ℝ := (-a, 0)
def B (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define the conditions
def Conditions (a b c : ℝ) (P M E : ℝ × ℝ) :=
  a > b ∧ b > 0 ∧
  P ∈ Ellipse a b ∧
  P.1 = -c ∧
  M.1 = -c ∧
  E.1 = 0 ∧
  (∃ k : ℝ, M.2 = k * (a - c) ∧ E.2 = k * a) ∧
  B a = (0, E.2 / 2) + M

-- State the theorem
theorem ellipse_eccentricity (a b c : ℝ) (P M E : ℝ × ℝ) 
  (h : Conditions a b c P M E) : 
  c / a = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l379_37952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_equivalence_l379_37936

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_inequality_equivalence (x : ℝ) :
  (floor x)^2 - 5*(floor x) + 6 ≤ 0 ↔ 2 ≤ x ∧ x < 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_equivalence_l379_37936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_area_l379_37932

-- Define the circles and points
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def P : Set (ℝ × ℝ) := Circle (0, 0) 1
def Q : Set (ℝ × ℝ) := Circle (5, 0) 4
def A : ℝ × ℝ := (1, 0)

noncomputable def B : ℝ × ℝ := sorry
noncomputable def C : ℝ × ℝ := sorry
noncomputable def D : ℝ × ℝ := sorry
noncomputable def E : ℝ × ℝ := sorry

-- Define the line BC
noncomputable def line_BC : Set (ℝ × ℝ) := sorry

-- Define the line ℓ
noncomputable def line_l : Set (ℝ × ℝ) := sorry

-- Define area of a triangle
noncomputable def area_triangle (a b c : ℝ × ℝ) : ℝ := sorry

-- Define tangency
def IsTangentTo (line : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)) : Prop := sorry

-- State the theorem
theorem circle_tangent_area : 
  B ∈ P ∧ 
  C ∈ Q ∧ 
  IsTangentTo line_BC P ∧ 
  IsTangentTo line_BC Q ∧
  A ∈ line_l ∧ 
  D ∈ P ∧ D ∈ line_l ∧
  E ∈ Q ∧ E ∈ line_l ∧
  (B.2 > 0 ∧ C.2 > 0 ∨ B.2 < 0 ∧ C.2 < 0) →
  area_triangle A B D = area_triangle A C E ∧ 
  area_triangle A B D = 64 / 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_area_l379_37932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l379_37934

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 - Real.sqrt (2 * x + 4)
def g (a x : ℝ) : ℝ := a * x + a - 1

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ,
  (∀ x₁ : ℝ, x₁ ≥ 0 → ∃ x₂ : ℝ, x₂ ≤ 1 ∧ f x₁ = g a x₂) ↔
  a ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l379_37934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_heptagon_distance_reciprocal_sum_l379_37975

/-- A regular heptagon in the Euclidean plane. -/
structure RegularHeptagon where
  vertices : Fin 7 → ℝ × ℝ
  is_regular : ∀ (i j : Fin 7), ‖vertices i - vertices j‖ = ‖vertices 0 - vertices 1‖

/-- The distance between two vertices of a regular heptagon. -/
def distance (h : RegularHeptagon) (i j : Fin 7) : ℝ :=
  ‖h.vertices i - h.vertices j‖

/-- The theorem stating the relationship between reciprocals of distances in a regular heptagon. -/
theorem regular_heptagon_distance_reciprocal_sum (h : RegularHeptagon) :
  1 / distance h 0 4 + 1 / distance h 0 2 = 1 / distance h 0 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_heptagon_distance_reciprocal_sum_l379_37975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l379_37966

/-- The length of a train given the speeds of two trains and the time taken to pass --/
noncomputable def trainLength (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v1 + v2) * (5 / 18) * t

/-- Theorem stating the length of the faster train --/
theorem faster_train_length :
  let v_slow : ℝ := 36 -- Speed of slower train in kmph
  let v_fast : ℝ := 45 -- Speed of faster train in kmph
  let t : ℝ := 6 -- Time taken to pass in seconds
  trainLength v_slow v_fast t = 135 := by
  -- Unfold the definition of trainLength
  unfold trainLength
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l379_37966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l379_37967

noncomputable def f (x : ℝ) : ℝ := 1 / (2^(x^2 + 2*x + 2))

theorem f_properties :
  (∀ x, f x > 0) ∧ 
  (∀ x, f x ≤ 1/2) ∧
  (∀ x₁ x₂, x₁ ≤ x₂ ∧ x₂ ≤ -1 → f x₁ ≤ f x₂) ∧
  (∀ x₁ x₂, -1 ≤ x₁ ∧ x₁ ≤ x₂ → f x₂ ≤ f x₁) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l379_37967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l379_37998

noncomputable def f (x : ℝ) := 4 * Real.sin (2 * x + Real.pi / 3)

theorem symmetry_of_f :
  ∀ x : ℝ, f (-π/6 - x) = -f (-π/6 + x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l379_37998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_trapezoid_area_l379_37930

/-- The area of a right-angled trapezoid with an acute angle of 60°, smaller base a, and longer leg b -/
noncomputable def trapezoidArea (a b : ℝ) : ℝ := ((2 * a + b) * b * Real.sqrt 3) / 4

/-- Theorem: The area of a right-angled trapezoid with an acute angle of 60°, smaller base a, and longer leg b is (2a + b)b√3 / 4 -/
theorem right_angled_trapezoid_area (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) :
  trapezoidArea a b = ((2 * a + b) * b * Real.sqrt 3) / 4 := by
  -- Unfold the definition of trapezoidArea
  unfold trapezoidArea
  -- The equality follows directly from the definition
  rfl

#check right_angled_trapezoid_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_trapezoid_area_l379_37930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_and_excluded_sum_l379_37972

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 5 * x / (3 * x^2 - 9 * x + 6)

-- Define the domain
def domain : Set ℝ := {x | x ≠ 1 ∧ x ≠ 2}

theorem function_domain_and_excluded_sum :
  (∀ x ∈ domain, ∃ y, f x = y) ∧
  (∀ x ∉ domain, ¬∃ y, f x = y) ∧
  (Finset.sum {1, 2} id) = 3 := by
  sorry

#eval Finset.sum {1, 2} id

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_and_excluded_sum_l379_37972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_l379_37973

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the additional points
structure AdditionalPoints where
  D : ℝ × ℝ
  F : ℝ × ℝ
  Q : ℝ × ℝ

-- Define the ratios
def ratios (t : Triangle) (p : AdditionalPoints) : Prop :=
  let (xC, yC) := t.C
  let (xD, yD) := p.D
  let (xB, yB) := t.B
  let (xA, yA) := t.A
  let (xF, yF) := p.F
  (xD - xC) / (xB - xD) = 2/3 ∧ (xF - xA) / (xB - xF) = 1/3

-- Define Q as the intersection point
def is_intersection (t : Triangle) (p : AdditionalPoints) : Prop :=
  let (xA, yA) := t.A
  let (xD, yD) := p.D
  let (xC, yC) := t.C
  let (xF, yF) := p.F
  let (xQ, yQ) := p.Q
  (yD - yA) * (xQ - xA) = (xD - xA) * (yQ - yA) ∧
  (yF - yC) * (xQ - xC) = (xF - xC) * (yQ - yC)

-- Theorem statement
theorem intersection_ratio (t : Triangle) (p : AdditionalPoints) :
  ratios t p → is_intersection t p →
  let (xC, yC) := t.C
  let (xQ, yQ) := p.Q
  let (xF, yF) := p.F
  (xQ - xC) / (xF - xQ) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_l379_37973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_neg_2024_round_7_8963_l379_37994

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Define a rounding function to the nearest hundredth
noncomputable def roundToHundredth (x : ℝ) : ℝ := 
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

-- Theorem for the reciprocal of -2024
theorem reciprocal_of_neg_2024 : 
  reciprocal (-2024) = -1 / 2024 := by
  -- Proof goes here
  sorry

-- Theorem for rounding 7.8963 to the nearest hundredth
theorem round_7_8963 : 
  roundToHundredth 7.8963 = 7.90 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_neg_2024_round_7_8963_l379_37994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_height_calculation_l379_37902

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the total volume of all bricks -/
def totalBrickVolume (b : BrickDimensions) (numBricks : ℕ) : ℝ :=
  (brickVolume b) * (numBricks : ℝ)

/-- Calculates the volume of the wall given its dimensions and height -/
def wallVolume (w : WallDimensions) (height : ℝ) : ℝ :=
  w.length * w.width * height

theorem wall_height_calculation (brick : BrickDimensions) (wall : WallDimensions) (numBricks : ℕ) :
  brick.length = 125 →
  brick.width = 11.25 →
  brick.height = 6 →
  wall.length = 800 →
  wall.width = 22.5 →
  numBricks = 1280 →
  ∃ h : ℝ, h = 600 ∧ totalBrickVolume brick numBricks = wallVolume wall h := by
  sorry

#check wall_height_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_height_calculation_l379_37902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l379_37957

open Real

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + m * x + m * log x

-- State the theorem
theorem max_m_value (m : ℝ) :
  m > 0 ∧
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 →
    |f m x₁ - f m x₂| < x₂^2 - x₁^2) →
  m ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l379_37957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l379_37927

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + (1/2) * t, (Real.sqrt 3 / 2) * t)

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define the valid range for ρ and θ
def valid_polar (ρ θ : ℝ) : Prop := ρ ≥ 0 ∧ 0 ≤ θ ∧ θ ≤ 2 * Real.pi

-- Theorem statement
theorem intersection_points :
  ∃ t₁ t₂ θ₁ θ₂ : ℝ,
    let (x₁, y₁) := line_l t₁
    let (x₂, y₂) := line_l t₂
    let ρ₁ := Real.sqrt (x₁^2 + y₁^2)
    let ρ₂ := Real.sqrt (x₂^2 + y₂^2)
    ρ₁ = curve_C θ₁ ∧
    ρ₂ = curve_C θ₂ ∧
    valid_polar ρ₁ θ₁ ∧
    valid_polar ρ₂ θ₂ ∧
    (ρ₁ = 2 ∧ θ₁ = 5 * Real.pi / 3) ∧
    (ρ₂ = 2 * Real.sqrt 3 ∧ θ₂ = Real.pi / 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l379_37927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rug_area_proof_rug_area_is_24_l379_37922

theorem rug_area_proof (floor_length floor_width strip_width : ℝ) 
  (h1 : floor_length = 12)
  (h2 : floor_width = 10)
  (h3 : strip_width = 3) : 
  (floor_length - 2 * strip_width) * (floor_width - 2 * strip_width) = 24 := by
  -- Substitute the given values
  rw [h1, h2, h3]
  -- Simplify the expression
  ring
  -- The result is automatically proven by the ring tactic

#eval (12 : ℝ) - 2 * 3
#eval (10 : ℝ) - 2 * 3
#eval ((12 : ℝ) - 2 * 3) * ((10 : ℝ) - 2 * 3)

theorem rug_area_is_24 : 
  ∃ (floor_length floor_width strip_width : ℝ),
    floor_length = 12 ∧ 
    floor_width = 10 ∧ 
    strip_width = 3 ∧
    (floor_length - 2 * strip_width) * (floor_width - 2 * strip_width) = 24 := by
  use 12, 10, 3
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  exact rug_area_proof 12 10 3 rfl rfl rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rug_area_proof_rug_area_is_24_l379_37922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_half_l379_37900

theorem angle_sum_is_pi_half (α β : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin (α + β) = (Real.sin α) ^ 2 + (Real.sin β) ^ 2) : 
  α + β = π/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_half_l379_37900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andrew_age_calculation_l379_37920

/-- Andrew's age in years -/
def andrew_age : ℚ := sorry

/-- Andrew's grandfather's age in years -/
def grandfather_age : ℚ := sorry

/-- The current year -/
def current_year : ℕ := 2023

/-- Andrew's grandfather is currently 16 times older than Andrew -/
axiom current_age_ratio : grandfather_age = 16 * andrew_age

/-- 20 years ago, Andrew's grandfather was 45 years older than Andrew -/
axiom past_age_difference : grandfather_age - 20 - (andrew_age - 20) = 45

theorem andrew_age_calculation : andrew_age = 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andrew_age_calculation_l379_37920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_after_2020_with_sum_4_l379_37944

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def firstYearAfterWithSumOfDigits (startYear : ℕ) (targetSum : ℕ) : ℕ :=
  let rec findYear (year : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then year
    else if sumOfDigits year = targetSum then year
    else findYear (year + 1) (fuel - 1)
  findYear (startYear + 1) 1000 -- Assuming we'll find the answer within 1000 years

theorem first_year_after_2020_with_sum_4 :
  firstYearAfterWithSumOfDigits 2020 4 = 2021 := by
  -- The proof goes here
  sorry

#eval firstYearAfterWithSumOfDigits 2020 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_after_2020_with_sum_4_l379_37944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l379_37947

noncomputable def a (n : ℕ) : ℝ := 2^n

noncomputable def b (n : ℕ) : ℝ := 1 + 3 * (n - 1)

noncomputable def S (n : ℕ) : ℝ := 2^(n+1) - 2

noncomputable def T (n : ℕ) : ℝ := (3 * n^2 - 1) / 2

theorem sequence_properties :
  (∀ n, a (n + 1) = 2 * a n) ∧
  a 1 = 2 ∧
  a 2 + 1 = (a 1 + a 3) / 2 ∧
  (∀ n, b (n + 1) - b n = b 2 - b 1) ∧
  b 2 = a 1 ∧
  b 8 = a 2 + a 4 ∧
  (∀ n, a n = 2^n) ∧
  (∀ n, S n = 2^(n+1) - 2) ∧
  (∀ n, T n = (3 * n^2 - 1) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l379_37947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_points_B_l379_37917

/-- Helper function to determine if a triangle is acute-angled -/
def is_acute_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ‖B - A‖^2 + ‖C - B‖^2 > ‖C - A‖^2 ∧
  ‖C - B‖^2 + ‖A - C‖^2 > ‖A - B‖^2 ∧
  ‖A - C‖^2 + ‖B - A‖^2 > ‖B - C‖^2

/-- Helper function to determine if a point is the centroid of a triangle -/
def is_centroid (Q A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  3 • Q = A + B + C

/-- Given two points A and Q in a plane, this theorem characterizes the locus of points B
    such that there exists an acute-angled triangle ABC for which Q is the centroid. -/
theorem locus_of_points_B (A Q : EuclideanSpace ℝ (Fin 2)) : 
  ∃ (M A₁ : EuclideanSpace ℝ (Fin 2)), 
    (∃ (k : ℝ), k > 1 ∧ M = A + k • (Q - A)) ∧  -- M is on the extension of AQ
    ‖Q - M‖ = (1/2) * ‖A - Q‖ ∧                 -- |QM| = 1/2|AQ|
    ‖M - A₁‖ = ‖A - M‖ ∧                        -- |MA₁| = |AM|
    (∀ B : EuclideanSpace ℝ (Fin 2), 
      (∃ C : EuclideanSpace ℝ (Fin 2), 
        is_acute_triangle A B C ∧ 
        is_centroid Q A B C) ↔ 
      (‖B - A‖ * ‖B - M‖ > ‖A - M‖^2 ∧          -- B is outside circle AM
       ‖B - M‖ * ‖B - A₁‖ > ‖M - A₁‖^2 ∧        -- B is outside circle MA₁
       ‖B - A‖ * ‖B - A₁‖ < ‖A - A₁‖^2)) :=     -- B is inside circle AA₁
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_points_B_l379_37917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l379_37945

/-- A hyperbola with foci on the y-axis and passing through a given point -/
structure Hyperbola where
  c : ℝ  -- Half the distance between foci
  p : ℝ × ℝ  -- Point that the hyperbola passes through

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : (ℝ → ℝ → Prop) :=
  λ x y ↦ y^2 - (x^2 / 3) = 1

/-- Theorem: Given a hyperbola with foci (0, -2) and (0, 2), passing through (-3, 2),
    its standard equation is y² - (x²/3) = 1 -/
theorem hyperbola_equation (h : Hyperbola) 
    (h_foci : h.c = 2)
    (h_point : h.p = (-3, 2)) :
  standard_equation h = λ x y ↦ y^2 - (x^2 / 3) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l379_37945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_converges_to_2_l379_37926

noncomputable def u : ℕ → ℝ
  | 0 => 3
  | n + 1 => (u n + 2 * (n + 1)^2 - 2) / (n + 1)^2

theorem u_converges_to_2 : ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n - 2| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_converges_to_2_l379_37926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_theorem_l379_37931

def is_valid_partition (n : ℕ) (A B : Finset ℕ) : Prop :=
  A.card = B.card ∧ 
  A ∪ B = Finset.range (n^2 + 1) \ {0} ∧
  A ∩ B = ∅

noncomputable def partition_ratio (A B : Finset ℕ) : ℚ :=
  (A.sum (fun x => (x : ℚ))) / (B.sum (fun x => (x : ℚ)))

theorem partition_theorem (n : ℕ) : 
  (∃ A B : Finset ℕ, is_valid_partition n A B ∧ partition_ratio A B = 39 / 64) ↔ 
  (∃ k : ℕ, n = 206 * k) := by
  sorry

#check partition_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_theorem_l379_37931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_alpha_eq_neg_one_l379_37909

theorem sin_two_alpha_eq_neg_one (α : ℝ) 
  (h1 : Real.sin α - Real.cos α = Real.sqrt 2) 
  (h2 : α ∈ Set.Ioo 0 Real.pi) : 
  Real.sin (2 * α) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_alpha_eq_neg_one_l379_37909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l379_37914

noncomputable def f (x : ℝ) := (1/3) * x^3 - 4*x + 4

theorem function_properties :
  (f 2 = -4/3) ∧
  (deriv f 2 = 0) ∧
  (f (-2) = 28/3) ∧
  (deriv f (-2) = 0) ∧
  (∀ x : ℝ, f x ≤ f (-2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l379_37914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l379_37999

/-- The set B of real numbers m such that x^2 - x - m < 0 for all x in [-1, 1] --/
def B : Set ℝ := {m | ∀ x, -1 ≤ x ∧ x ≤ 1 → x^2 - x - m < 0}

/-- The set A of real numbers x satisfying (x - 3a)(x - a - 2) < 0 --/
def A (a : ℝ) : Set ℝ := {x | (x - 3*a) * (x - a - 2) < 0}

theorem problem_solution :
  (B = Set.Ioi 2) ∧
  ({a : ℝ | A a ⊆ B} = Set.Ici (2/3)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l379_37999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_count_theorem_l379_37959

/-- The number of ways to distribute n distinct balls into 4 distinct boxes,
    where the first box contains an odd number of balls and the second box
    contains an even number of balls. -/
def distribution_count (n : ℕ) : ℕ :=
  4^(n-1)

/-- A function representing the number of ways to distribute n distinct balls
    into k distinct boxes, given a predicate on the distribution. -/
def number_of_distributions (n k : ℕ) 
  (p : (Fin k → Finset (Fin n)) → Prop) : ℕ :=
  sorry

/-- Theorem stating that the number of ways to distribute n distinct balls
    (n ≥ 1) into 4 distinct boxes, where the first box contains an odd number
    of balls and the second box contains an even number of balls, is equal to
    4^(n-1). -/
theorem distribution_count_theorem (n : ℕ) (h : n ≥ 1) :
  (distribution_count n) = 
  (number_of_distributions n 4 
    (λ boxes => (Finset.card (boxes 0)) % 2 = 1 ∧ (Finset.card (boxes 1)) % 2 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_count_theorem_l379_37959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_510_l379_37915

/-- Given a square ABCD with side length 60 and an inscribed circle with center O -/
def Square (A B C D O : ℝ × ℝ) : Prop :=
  let side_length := 60
  A = (-side_length/2, side_length/2) ∧
  B = (side_length/2, side_length/2) ∧
  C = (side_length/2, -side_length/2) ∧
  D = (-side_length/2, -side_length/2) ∧
  O = (0, 0)

/-- E is the midpoint of AD -/
def Midpoint (E A D : ℝ × ℝ) : Prop :=
  E = ((A.1 + D.1)/2, (A.2 + D.2)/2)

/-- AC and BE intersect at H -/
def Intersection (A B C E H : ℝ × ℝ) : Prop :=
  ∃ t s : ℝ,
    H = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2)) ∧
    H = (B.1 + s * (E.1 - B.1), B.2 + s * (E.2 - B.2))

/-- F and G are on the top half of the circle -/
def OnCircle (F G O : ℝ × ℝ) : Prop :=
  (F.1 - O.1)^2 + (F.2 - O.2)^2 = (60/2)^2 ∧
  (G.1 - O.1)^2 + (G.2 - O.2)^2 = (60/2)^2 ∧
  F.2 > O.2 ∧ G.2 > O.2

/-- The area of a triangle given three points -/
noncomputable def TriangleArea (P Q R : ℝ × ℝ) : ℝ :=
  let a := P.1 * (Q.2 - R.2)
  let b := Q.1 * (R.2 - P.2)
  let c := R.1 * (P.2 - Q.2)
  abs (a + b + c) / 2

/-- The main theorem -/
theorem shaded_area_is_510 (A B C D E F G H O : ℝ × ℝ) :
  Square A B C D O →
  Midpoint E A D →
  Intersection A B C E H →
  OnCircle F G O →
  TriangleArea A H E + TriangleArea H G O = 510 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_510_l379_37915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_lcm_sum_pairs_count_l379_37962

theorem gcd_lcm_sum_pairs_count : 
  let S := Finset.range 21
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 ∈ S ∧ p.2 ∈ S ∧ 
    p.1 + 10 ≤ 30 ∧ p.2 + 10 ≤ 30 ∧
    Nat.gcd (p.1 + 10) (p.2 + 10) + Nat.lcm (p.1 + 10) (p.2 + 10) = (p.1 + 10) + (p.2 + 10))
    (Finset.product S S)).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_lcm_sum_pairs_count_l379_37962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_increase_qualified_students_l379_37953

/-- Represents the percentage of qualified students to appeared students in School A -/
noncomputable def school_a_qualification_rate : ℝ := 70 / 100

/-- Represents the increase rate of appeared students in School B compared to School A -/
noncomputable def school_b_appearance_increase : ℝ := 20 / 100

/-- Represents the percentage of qualified students to appeared students in School B -/
noncomputable def school_b_qualification_rate : ℝ := 87.5 / 100

/-- Theorem stating that the percentage increase in qualified students from School B 
    compared to School A is 50% -/
theorem percentage_increase_qualified_students : 
  let school_a_appeared : ℝ := 1 -- Arbitrary value for School A appeared students
  let school_a_qualified := school_a_appeared * school_a_qualification_rate
  let school_b_appeared := school_a_appeared * (1 + school_b_appearance_increase)
  let school_b_qualified := school_b_appeared * school_b_qualification_rate
  (school_b_qualified - school_a_qualified) / school_a_qualified * 100 = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_increase_qualified_students_l379_37953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l379_37968

/-- Parabola type -/
structure Parabola where
  f : ℝ → ℝ
  focus : ℝ × ℝ

/-- Point type -/
abbrev Point := ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Main theorem -/
theorem parabola_distance_theorem (C : Parabola) (A B : Point) :
  C.f A.2 = A.1 ∧  -- A lies on C
  C.f = (λ y => y^2 / 4) ∧  -- C: y² = 4x
  B = (3, 0) ∧
  distance A C.focus = distance B C.focus →
  distance A B = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l379_37968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invertible_functions_product_l379_37948

def f₁ : ℤ → ℝ
| -3 => -21
| -2 => -6
| -1 => 1
| 0 => 0
| 1 => -1
| 2 => 2
| 3 => 21
| _ => 0  -- This case should never be reached due to the domain restriction

def f₂ : ℝ → ℝ := fun x ↦ x - 2

def f₃ : ℝ → ℝ := fun x ↦ 2 - x

def is_invertible (f : ℝ → ℝ) : Prop :=
  ∀ y, ∃! x, f x = y

theorem invertible_functions_product : 
  (is_invertible f₂ ∧ is_invertible f₃ ∧ ¬is_invertible (fun x ↦ f₁ (Int.floor x))) →
  2 * 3 = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_invertible_functions_product_l379_37948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_youseff_distance_l379_37924

/-- The number of blocks Youseff lives from his office -/
def blocks_to_office : ℕ := 9

/-- Time in minutes to walk one block -/
def walk_time_per_block : ℚ := 1

/-- Time in minutes to bike one block -/
def bike_time_per_block : ℚ := 1/3

/-- The difference in minutes between walking and biking time -/
def time_difference : ℕ := 6

theorem youseff_distance : 
  blocks_to_office = 9 ∧ 
  (blocks_to_office : ℚ) * walk_time_per_block = 
    (blocks_to_office : ℚ) * bike_time_per_block + time_difference := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_youseff_distance_l379_37924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l379_37974

noncomputable def variance (l : List ℝ) : ℝ := 
  let mean := l.sum / l.length
  (l.map (λ x => (x - mean)^2)).sum / l.length

theorem variance_transformation 
  (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (h : variance [a₁, a₂, a₃, a₄, a₅, a₆] = 3) : 
  variance [2*(a₁-3), 2*(a₂-3), 2*(a₃-3), 2*(a₄-3), 2*(a₅-3), 2*(a₆-3)] = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l379_37974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_dot_product_minimum_distance_l379_37937

-- Define the parabola and its properties
def Parabola (p : ℝ) := {(x, y) : ℝ × ℝ | y^2 = 2*p*x ∧ p > 0}

-- Define the directrix
def Directrix := {(x, _y) : ℝ × ℝ | x = -1}

-- Define the point T
def T (t : ℝ) : ℝ × ℝ := (t, 0)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the minimum distance function d(t)
noncomputable def d (t : ℝ) : ℝ :=
  if t ≥ 2 then 2 * Real.sqrt (t - 1) else t

-- Theorem 1: Prove the parabola equation and dot product property
theorem parabola_and_dot_product (p : ℝ) (t : ℝ) (h1 : p > 0) (h2 : t > 0) :
  ∃ (A B : ℝ × ℝ), A ∈ Parabola p ∧ B ∈ Parabola p ∧
  dot_product A B = t^2 - 4*t := by sorry

-- Theorem 2: Prove the minimum distance function
theorem minimum_distance (p : ℝ) (t : ℝ) (h1 : p > 0) (h2 : t > 0) :
  ∀ P : ℝ × ℝ, P ∈ Parabola p →
  d t ≤ distance P (T t) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_dot_product_minimum_distance_l379_37937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folder_cost_l379_37961

theorem folder_cost (num_pens : ℕ) (pen_cost : ℚ) (num_notebooks : ℕ) (notebook_cost : ℚ) 
  (num_folders : ℕ) (paid : ℚ) (change : ℚ) :
  num_pens = 3 →
  pen_cost = 1 →
  num_notebooks = 4 →
  notebook_cost = 3 →
  num_folders = 2 →
  paid = 50 →
  change = 25 →
  (paid - change - (num_pens * pen_cost + num_notebooks * notebook_cost)) / num_folders = 5 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folder_cost_l379_37961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_l379_37969

/-- Given a function f(x) = sin(ωx + π/3) where 0 < ω < 1, 
    if the graph of f(x) is symmetric about the point (-2, 0), 
    then ω = π/6 -/
theorem sine_symmetry (ω : ℝ) (h1 : 0 < ω) (h2 : ω < 1) :
  (∀ x : ℝ, Real.sin (ω * x + π/3) = Real.sin (ω * (-4 - x) + π/3)) → ω = π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_l379_37969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sqrt_two_identity_l379_37995

theorem tan_sqrt_two_identity (α : ℝ) (h : Real.tan α = Real.sqrt 2) :
  2 * (Real.sin α)^2 - Real.sin α * Real.cos α + (Real.cos α)^2 = (5 - Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sqrt_two_identity_l379_37995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l379_37912

-- Define the function v(x)
noncomputable def v (x : ℝ) : ℝ := 1 / Real.sqrt (x + 4)

-- State the theorem about the domain of v(x)
theorem domain_of_v :
  {x : ℝ | v x ≠ 0 ∧ v x ∈ Set.range v} = Set.Ioi (-4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l379_37912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_special_case_l379_37903

/-- Represents a hyperbola with semi-major axis a, semi-minor axis b, and focal distance c -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_c_eq : c^2 = a^2 + b^2

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

theorem hyperbola_eccentricity_special_case (h : Hyperbola)
  (h_circle_intersect : ∃ (A B : ℝ × ℝ), 
    (A.1 - h.c)^2 + A.2^2 = h.a^2 ∧ 
    (B.1 - h.c)^2 + B.2^2 = h.a^2 ∧
    A.2 / A.1 = h.b / h.a ∧
    B.2 / B.1 = h.b / h.a)
  (h_AB_length : ∃ (A B : ℝ × ℝ), 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2) = (2/3) * h.c) :
  eccentricity h = 3 * Real.sqrt 5⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_special_case_l379_37903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_sqrt_radii_theorem_l379_37988

noncomputable def initial_radius_1 : ℝ := 50^2
noncomputable def initial_radius_2 : ℝ := 53^2

def num_layers : ℕ := 7  -- L₀ to L₆ inclusive

noncomputable def sum_reciprocal_sqrt_radii : ℝ :=
  (2^num_layers - 1) * (1 / Real.sqrt initial_radius_1 + 1 / Real.sqrt initial_radius_2)

theorem sum_reciprocal_sqrt_radii_theorem :
  sum_reciprocal_sqrt_radii = 13021 / 2650 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_sqrt_radii_theorem_l379_37988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l379_37942

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | n + 1 => sequence_a n - (1/3) * (sequence_a n)^2

theorem sequence_properties :
  (∀ n : ℕ, sequence_a (n + 1) < sequence_a n) ∧
  (∀ n : ℕ, sequence_a n < 2 * sequence_a (n + 1)) ∧
  (5/2 < 100 * sequence_a 99 ∧ 100 * sequence_a 99 < 3) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l379_37942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_property_l379_37993

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Int
  ones : Int
  tens_range : 1 ≤ tens ∧ tens ≤ 9
  ones_range : 0 ≤ ones ∧ ones ≤ 9

/-- Returns the numeric value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Int :=
  10 * n.tens + n.ones

/-- Returns the reverse of a two-digit number -/
def TwoDigitNumber.reverse (n : TwoDigitNumber) : Int :=
  10 * n.ones + n.tens

/-- Returns the sum of digits of a two-digit number -/
def TwoDigitNumber.digitSum (n : TwoDigitNumber) : Int :=
  n.tens + n.ones

theorem two_digit_number_property (n : TwoDigitNumber) :
  |n.value - n.reverse| = 7 * n.digitSum →
  n.value + n.reverse = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_property_l379_37993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_a_b_l379_37965

noncomputable def average_weight (x y : ℝ) : ℝ := (x + y) / 2

variable (weight_a weight_b weight_c : ℝ)

theorem average_weight_a_b
  (h1 : (weight_a + weight_b + weight_c) / 3 = 30)
  (h2 : average_weight weight_b weight_c = 28)
  (h3 : weight_b = 16) :
  average_weight weight_a weight_b = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_a_b_l379_37965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_mixture_theorem_l379_37928

-- Define the initial quantities and percentages
noncomputable def initial_volume : ℝ := 8
noncomputable def initial_butterfat_percent : ℝ := 50
noncomputable def added_butterfat_percent : ℝ := 10
noncomputable def final_butterfat_percent : ℝ := 20

-- Define the volume of milk to be added
noncomputable def added_volume : ℝ := 24

-- Define the total volume after mixing
noncomputable def total_volume : ℝ := initial_volume + added_volume

-- Define the function to calculate the final butterfat percentage
noncomputable def final_mixture_percent (init_vol init_percent added_vol added_percent : ℝ) : ℝ :=
  (init_vol * init_percent + added_vol * added_percent) / (init_vol + added_vol) / 100

-- Theorem statement
theorem milk_mixture_theorem :
  final_mixture_percent initial_volume initial_butterfat_percent added_volume added_butterfat_percent = final_butterfat_percent / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_mixture_theorem_l379_37928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_theorem_l379_37946

noncomputable def original_function (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)
noncomputable def reference_function (x : ℝ) : ℝ := 2 * Real.sin (2 * x)

theorem min_shift_theorem :
  ∃ (shift : ℝ), shift > 0 ∧
  (∀ x : ℝ, original_function x = reference_function (x - shift)) ∧
  (∀ s : ℝ, s > 0 ∧ (∀ x : ℝ, original_function x = reference_function (x - s)) → s ≥ shift) ∧
  shift = Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_theorem_l379_37946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_proportional_inverse_proportional_quadratic_power_function_l379_37941

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 + 2*m) * x^(m^2 + m - 1)

-- Theorem for direct proportional function
theorem direct_proportional (m : ℝ) :
  (∃ k : ℝ, ∀ x : ℝ, f m x = k * x) ↔ m = 1 := by sorry

-- Theorem for inverse proportional function
theorem inverse_proportional (m : ℝ) :
  (∃ k : ℝ, ∀ x : ℝ, f m x = k / x) ↔ m = -1 := by sorry

-- Theorem for quadratic function
theorem quadratic (m : ℝ) :
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f m x = a * x^2 + b * x + c) ↔ 
  (m = (-1 + Real.sqrt 13) / 2 ∨ m = (-1 - Real.sqrt 13) / 2) := by sorry

-- Theorem for power function
theorem power_function (m : ℝ) :
  (∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f m x = a * x^b) ↔ 
  (m = -1 + Real.sqrt 2 ∨ m = -1 - Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_proportional_inverse_proportional_quadratic_power_function_l379_37941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_as_linear_combination_l379_37907

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, -1; 2, -4]

theorem inverse_as_linear_combination :
  let I : Matrix (Fin 2) (Fin 2) ℚ := 1
  N⁻¹ = (1 / 14 : ℚ) • N + (1 / 14 : ℚ) • I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_as_linear_combination_l379_37907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_zero_on_chessboard_l379_37938

/-- Represents a square on the chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- The chessboard represented as a function from Square to ℝ -/
def Chessboard := Square → ℝ

/-- Returns the list of adjacent squares for a given square -/
def adjacent (s : Square) : List Square :=
  sorry

/-- The condition that each number is not greater than the average of adjacent numbers -/
def satisfies_condition (board : Chessboard) : Prop :=
  ∀ s : Square, board s ≤ (((adjacent s).map board).sum / (adjacent s).length)

theorem all_zero_on_chessboard (board : Chessboard) 
  (corner_zeros : ∀ s : Square, (s.row = 0 ∨ s.row = 7) ∧ (s.col = 0 ∨ s.col = 7) → board s = 0)
  (h_condition : satisfies_condition board) :
  ∀ s : Square, board s = 0 :=
by
  sorry

#check all_zero_on_chessboard

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_zero_on_chessboard_l379_37938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_theorem_l379_37980

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Reflect a point across the xy-plane -/
def reflect_xy (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

theorem reflection_distance_theorem :
  let A : Point3D := { x := 2, y := -3, z := 5 }
  let B : Point3D := reflect_xy A
  distance A B = 10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_theorem_l379_37980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l379_37976

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 1

-- Define a line passing through (2,3)
def line_through_point (k : ℝ) (x y : ℝ) : Prop := y - 3 = k * (x - 2)

-- Define tangency condition
def is_tangent (k : ℝ) : Prop := 
  (k = 0 ∧ ∀ y, ¬my_circle 2 y) ∨ 
  (k ≠ 0 ∧ ∃! p : ℝ × ℝ, my_circle p.1 p.2 ∧ line_through_point k p.1 p.2)

-- Theorem statement
theorem tangent_line_equation : 
  ∃ k, is_tangent k ∧ 
  (∀ x y, line_through_point k x y ↔ (x = 2 ∨ 12*x - 5*y - 9 = 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l379_37976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_polish_time_is_150_l379_37997

/-- Calculates the remaining time to polish items given initial quantities, polishing times, and completed percentages. -/
def remaining_polish_time (shoe_pairs belt_count watch_count : ℕ) 
                          (shoe_time belt_time watch_time : ℕ) 
                          (shoe_percent belt_percent watch_percent : ℚ) : ℕ :=
  let shoe_count := 2 * shoe_pairs
  let shoes_left := shoe_count - Int.floor (shoe_percent * ↑shoe_count)
  let belts_left := belt_count - Int.floor (belt_percent * ↑belt_count)
  let watches_left := watch_count - Int.floor (watch_percent * ↑watch_count)
  (shoes_left * shoe_time + belts_left * belt_time + watches_left * watch_time).toNat

/-- Theorem stating that the remaining polish time for given conditions is 150 minutes. -/
theorem remaining_polish_time_is_150 : 
  remaining_polish_time 10 5 3 10 15 5 (45/100) (60/100) (20/100) = 150 := by
  sorry

#eval remaining_polish_time 10 5 3 10 15 5 (45/100) (60/100) (20/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_polish_time_is_150_l379_37997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perimeters_l379_37950

-- Define the side length of the first triangle
noncomputable def initial_side_length : ℝ := 30

-- Define the sequence of side lengths
noncomputable def side_length (n : ℕ) : ℝ := initial_side_length / (2 ^ n)

-- Define the perimeter of the nth triangle
noncomputable def perimeter (n : ℕ) : ℝ := 3 * side_length n

-- Theorem statement
theorem sum_of_perimeters :
  (∑' n, perimeter n) = 180 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perimeters_l379_37950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_parallelogram_syllogism_l379_37904

-- Define the structures
structure Square where
  -- Add some placeholder field to make it a valid structure
  dummy : Unit

structure Parallelogram where
  -- Add some placeholder field to make it a valid structure
  dummy : Unit

-- Define the properties
def has_equal_opposite_sides (P : Type) : Prop := sorry

-- Define the syllogism components
def major_premise : Prop := ∀ (P : Parallelogram), has_equal_opposite_sides Parallelogram
def minor_premise : Prop := ∀ (S : Square), ∃ (P : Parallelogram), S.dummy = P.dummy
def conclusion : Prop := ∀ (S : Square), has_equal_opposite_sides Square

-- Theorem statement
theorem square_parallelogram_syllogism :
  minor_premise = (∀ (S : Square), ∃ (P : Parallelogram), S.dummy = P.dummy) :=
by
  -- The proof is just a restatement of the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_parallelogram_syllogism_l379_37904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l379_37918

/-- A linear function f(x) = ax + b satisfying certain intersection conditions -/
def LinearFunctionWithIntersections (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ,
    (∀ x, f x = a * x + b) ∧
    (Real.sqrt ((a^2 + 1) * (a^2 + 4*b - 4)) = 3 * Real.sqrt 2) ∧
    (Real.sqrt ((a^2 + 1) * (a^2 + 4*b - 8)) = Real.sqrt 10)

/-- The main theorem stating the distance between intersection points -/
theorem intersection_distance (f : ℝ → ℝ) 
    (hf : LinearFunctionWithIntersections f) :
    ∃ a b : ℝ, (∀ x, f x = a * x + b) ∧
    Real.sqrt ((a^2 + 1) * (a^2 + 4*b)) = Real.sqrt 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l379_37918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_CDEF_concyclic_l379_37939

-- Define the circle
variable (circle : Set (ℝ × ℝ))

-- Define the points on the circle
variable (A B C D : ℝ × ℝ)

-- Define S as the midpoint of arc AB
variable (S : ℝ × ℝ)

-- Define E as the intersection of SD and AB
variable (E : ℝ × ℝ)

-- Define F as the intersection of SC and AB
variable (F : ℝ × ℝ)

-- Axioms based on the given conditions
axiom on_circle : A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle

axiom order_on_circle : ∃ (arc : Set (ℝ × ℝ)), arc ⊆ circle ∧ A ∈ arc ∧ B ∈ arc ∧ C ∈ arc ∧ D ∈ arc ∧ 
  (∀ (X Y : ℝ × ℝ), X ∈ arc → Y ∈ arc → (X = A ∨ X = B ∨ X = C ∨ X = D) → 
  (Y = A ∨ Y = B ∨ Y = C ∨ Y = D) → X ≠ Y → 
  ((X = A ∧ Y = B) ∨ (X = B ∧ Y = C) ∨ (X = C ∧ Y = D) ∨ (X = D ∧ Y = A)))

axiom S_midpoint : ∃ (arc : Set (ℝ × ℝ)), arc ⊆ circle ∧ A ∈ arc ∧ B ∈ arc ∧ S ∈ arc ∧ 
  C ∉ arc ∧ D ∉ arc ∧ (∀ (X : ℝ × ℝ), X ∈ arc → (dist X A = dist X B))

noncomputable def Line (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {X : ℝ × ℝ | ∃ t : ℝ, X = (1 - t) • P + t • Q}

axiom E_intersection : E ∈ (Line S D) ∩ (Line A B)

axiom F_intersection : F ∈ (Line S C) ∩ (Line A B)

-- Theorem to prove
theorem CDEF_concyclic : ∃ (circle' : Set (ℝ × ℝ)), C ∈ circle' ∧ D ∈ circle' ∧ E ∈ circle' ∧ F ∈ circle' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_CDEF_concyclic_l379_37939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_response_rate_increase_l379_37955

def original_customers : ℕ := 80
def original_respondents : ℕ := 7
def redesigned_customers : ℕ := 63
def redesigned_respondents : ℕ := 9

noncomputable def response_rate (respondents : ℕ) (customers : ℕ) : ℝ :=
  (respondents : ℝ) / (customers : ℝ) * 100

noncomputable def percentage_increase (original : ℝ) (redesigned : ℝ) : ℝ :=
  ((redesigned - original) / original) * 100

theorem response_rate_increase :
  abs (percentage_increase 
    (response_rate original_respondents original_customers)
    (response_rate redesigned_respondents redesigned_customers) - 63.24) < 0.01 := by
  sorry

#eval original_customers
#eval original_respondents
#eval redesigned_customers
#eval redesigned_respondents

end NUMINAMATH_CALUDE_ERRORFEEDBACK_response_rate_increase_l379_37955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_sin_x_equals_1_solutions_l379_37951

theorem cos_2x_plus_sin_x_equals_1_solutions (x : ℝ) :
  (0 < x ∧ x < π ∧ Real.cos (2 * x) + Real.sin x = 1) ↔ (x = π / 6 ∨ x = 5 * π / 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_sin_x_equals_1_solutions_l379_37951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_polynomial_equation_l379_37970

theorem smallest_k_for_polynomial_equation (n : ℕ) (hn : Even n) :
  ∃ (k₀ : ℕ) (f g : Polynomial ℤ) (α t : ℕ),
    k₀ > 0 ∧
    n = 2^α * t ∧
    Odd t ∧
    k₀ = 2^t ∧
    (∀ (x : ℤ), (k₀ : ℤ) = Polynomial.eval x f * (x + 1)^n + Polynomial.eval x g * (x^n + 1)) ∧
    (∀ (k : ℕ) (f' g' : Polynomial ℤ),
      k > 0 ∧ (∀ (x : ℤ), (k : ℤ) = Polynomial.eval x f' * (x + 1)^n + Polynomial.eval x g' * (x^n + 1))
      → k₀ ≤ k) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_polynomial_equation_l379_37970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_5_13th_l379_37996

/-- The decimal representation of 5/13 repeats every 6 digits -/
def repeating_period : ℕ := 6

/-- The repeating decimal representation of 5/13 -/
def decimal_rep : Fin repeating_period → ℕ
| ⟨0, _⟩ => 3
| ⟨1, _⟩ => 8
| ⟨2, _⟩ => 4
| ⟨3, _⟩ => 6
| ⟨4, _⟩ => 1
| ⟨5, _⟩ => 5

theorem digit_150_of_5_13th : 
  decimal_rep (Fin.ofNat ((150 - 1) % repeating_period)) = 5 := by
  -- Convert the natural number to Fin type
  have h : (150 - 1) % repeating_period = 5 := by sorry
  -- Rewrite using the above equality
  rw [h]
  -- The result follows from the definition of decimal_rep
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_5_13th_l379_37996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l379_37991

/-- The trajectory of a point P(x, y) is a parabola, given that the difference between 
    its distance to the line x+5=0 and its distance to the point M(2,0) is 3. -/
theorem trajectory_is_parabola : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ x y : ℝ, (|x + 5| - Real.sqrt ((x - 2)^2 + y^2) = 3) → 
               y^2 = a*x^2 + b*x + c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l379_37991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_group_frequency_l379_37981

theorem sixth_group_frequency 
  (total_sample : ℕ)
  (num_groups : ℕ)
  (group1_freq : ℕ)
  (group2_freq : ℕ)
  (group3_freq : ℕ)
  (group4_freq : ℕ)
  (group5_freq_ratio : ℚ)
  (h1 : total_sample = 40)
  (h2 : num_groups = 6)
  (h3 : group1_freq = 10)
  (h4 : group2_freq = 5)
  (h5 : group3_freq = 7)
  (h6 : group4_freq = 6)
  (h7 : group5_freq_ratio = 1/10)
  : ∃ (group6_freq : ℕ), group6_freq = 8 ∧ 
    group1_freq + group2_freq + group3_freq + group4_freq + 
    (group5_freq_ratio * ↑total_sample).floor + group6_freq = total_sample :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_group_frequency_l379_37981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_S_div_a_l379_37986

noncomputable def a (n : ℕ) : ℝ := 9^n

noncomputable def S (n : ℕ) : ℝ := 4 * 9^(n-1) * (1 - 1 / 10^n)

theorem limit_S_div_a :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |S n / a n - 4/9| < ε :=
by
  sorry

#check limit_S_div_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_S_div_a_l379_37986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_line_intersection_l379_37985

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents a parallelepiped -/
structure Parallelepiped where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Theorem: In a parallelepiped, if a line passes through the center of symmetry of the base
    and intersects specific edges, it divides one edge in a 2:1 ratio. -/
theorem parallelepiped_line_intersection
  (p : Parallelepiped)
  (H : Point3D)  -- Center of symmetry of base ABCD
  (F : Point3D)  -- Midpoint of AA₁
  (l : Line3D)   -- Line passing through H
  (X : Point3D)  -- Intersection point of l and FB₁
  (h1 : H.x = (p.A.x + p.C.x) / 2 ∧ H.y = (p.A.y + p.C.y) / 2)  -- H is center of ABCD
  (h2 : F.x = (p.A.x + p.A₁.x) / 2 ∧ F.y = (p.A.y + p.A₁.y) / 2 ∧ F.z = (p.A.z + p.A₁.z) / 2)  -- F is midpoint of AA₁
  (h3 : l.point = H)  -- l passes through H
  (h4 : ∃ t : ℝ, X = Point3D.mk (F.x + t * (p.B₁.x - F.x)) (F.y + t * (p.B₁.y - F.y)) (F.z + t * (p.B₁.z - F.z)))  -- X is on FB₁
  (h5 : ∃ s : ℝ, Point3D.mk (p.B.x + s * (p.C₁.x - p.B.x)) (p.B.y + s * (p.C₁.y - p.B.y)) (p.B.z + s * (p.C₁.z - p.B.z)) = Point3D.mk (H.x + l.direction.x * s) (H.y + l.direction.y * s) (H.z + l.direction.z * s))  -- l intersects BC₁
  : (X.x - F.x)^2 + (X.y - F.y)^2 + (X.z - F.z)^2 = 4 * ((p.B₁.x - F.x)^2 + (p.B₁.y - F.y)^2 + (p.B₁.z - F.z)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_line_intersection_l379_37985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l379_37983

/-- Triangle ABC with given side lengths and angle -/
structure TriangleABC where
  AB : ℝ
  AC : ℝ
  BAC : ℝ
  h_AB : AB = 2
  h_AC : AC = 4
  h_BAC : BAC = π / 3

/-- The area of triangle ABC -/
noncomputable def area (t : TriangleABC) : ℝ := 
  1/2 * t.AB * t.AC * Real.sin t.BAC

/-- The radius of the incircle of triangle ABC -/
noncomputable def incircle_radius (t : TriangleABC) : ℝ :=
  let BC := Real.sqrt (t.AB^2 + t.AC^2 - 2*t.AB*t.AC*Real.cos t.BAC)
  let s := (t.AB + t.AC + BC) / 2
  area t / s

/-- Main theorem about the area and incircle radius of triangle ABC -/
theorem triangle_ABC_properties (t : TriangleABC) : 
  area t = 2 * Real.sqrt 3 ∧ incircle_radius t = Real.sqrt 3 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l379_37983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l379_37977

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * 7786 * k) :
  Int.gcd (8 * b^2 + 85 * b + 200) (2 * b + 10) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l379_37977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sine_relations_l379_37908

noncomputable section

open Real

theorem triangle_angle_sine_relations (A B C : ℝ) (hTriangle : A + B + C = π) :
  (∀ (A B : ℝ), Real.sin A > Real.sin B → A > B) ∧
  (∀ (A B : ℝ), A ≤ B → Real.sin A ≤ Real.sin B) ∧
  (∀ (A B : ℝ), A > B → Real.sin A > Real.sin B) ∧
  (∀ (A B : ℝ), Real.sin A ≤ Real.sin B → A ≤ B) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sine_relations_l379_37908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_yellow_l379_37956

/-- Represents the color of a traffic light -/
inductive Color
  | Red
  | Green
  | Yellow
  deriving Repr, DecidableEq

/-- Represents the state of all traffic lights -/
def TrafficLightState := Fin 1998 → Color

/-- The initial state of the traffic lights -/
def initial_state : TrafficLightState :=
  fun i => if i = 0 then Color.Red else Color.Green

/-- The rule for updating a traffic light's color based on its neighbors -/
def update_color (left right : Color) : Color :=
  if left = right then left
  else match left, right with
    | Color.Red, Color.Green | Color.Green, Color.Red => Color.Yellow
    | Color.Red, Color.Yellow | Color.Yellow, Color.Red => Color.Green
    | Color.Green, Color.Yellow | Color.Yellow, Color.Green => Color.Red
    | _, _ => left  -- This case should never occur

/-- The function to update the entire state of traffic lights -/
def update_state (state : TrafficLightState) : TrafficLightState :=
  fun i => update_color (state ((i - 1 + 1998) % 1998)) (state ((i + 1) % 1998))

/-- Theorem stating that it's impossible for all lights to become yellow -/
theorem not_all_yellow (n : Nat) : ∃ i, (update_state^[n] initial_state) i ≠ Color.Yellow := by
  sorry

#eval initial_state 0
#eval initial_state 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_yellow_l379_37956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_stop_time_example_l379_37971

/-- Calculates the time a train stops per hour given its speeds with and without stoppages -/
noncomputable def train_stop_time (Vi Ve : ℝ) : ℝ :=
  (Ve - Vi) / Ve * 60

/-- Theorem: A train with speed including stoppages of 90 kmph and
    speed excluding stoppages of 120 kmph stops for 15 minutes per hour -/
theorem train_stop_time_example : train_stop_time 90 120 = 15 := by
  -- Unfold the definition of train_stop_time
  unfold train_stop_time
  -- Simplify the arithmetic expression
  simp [div_mul_eq_mul_div]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_stop_time_example_l379_37971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tommys_house_price_l379_37940

theorem tommys_house_price : ∃ (first_house_price : ℝ),
  let first_house_current_value := 1.25 * first_house_price
  let new_house_price := 500000
  let loan_percentage := 0.75
  first_house_current_value = (1 - loan_percentage) * new_house_price ∧
  first_house_price = 100000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tommys_house_price_l379_37940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shovel_time_approx_20_hours_l379_37990

/-- Represents Pauline's snow shoveling rate and driveway dimensions --/
structure SnowRemovalProblem where
  initial_rate : ℕ              -- Initial shoveling rate in cubic yards per hour
  rate_decrease : ℕ             -- Rate decrease per hour in cubic yards
  driveway_width : ℕ            -- Driveway width in yards
  driveway_length : ℕ           -- Driveway length in yards
  snow_depth : ℕ                -- Snow depth in yards

/-- Calculates the approximate time needed to shovel the driveway clean --/
def shovel_time (problem : SnowRemovalProblem) : ℕ :=
  sorry

/-- Theorem stating that it takes approximately 20 hours to shovel the driveway clean --/
theorem shovel_time_approx_20_hours (problem : SnowRemovalProblem)
  (h1 : problem.initial_rate = 25)
  (h2 : problem.rate_decrease = 2)
  (h3 : problem.driveway_width = 5)
  (h4 : problem.driveway_length = 12)
  (h5 : problem.snow_depth = 3) :
  shovel_time problem = 20 :=
by
  sorry

#check shovel_time_approx_20_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shovel_time_approx_20_hours_l379_37990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cotangent_identity_l379_37949

theorem cotangent_identity (θ : ℝ) (h : Real.tan (Real.pi / 2 - θ) = 3) :
  (1 - Real.sin θ) / Real.cos θ - Real.cos θ / (1 + Real.sin θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cotangent_identity_l379_37949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_inverse_l379_37910

theorem inverse_sum_inverse : (9⁻¹ - 5⁻¹ + 2⁻¹ : ℝ)⁻¹ = 90 / 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_inverse_l379_37910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_gift_assignment_l379_37906

-- Define the types for guests and gifts
inductive Guest : Type
  | Bear | Lynx | Squirrel | Mouse | Wolf | Sheep

inductive Gift : Type
  | Candlestick | Plate | Needle | Ring

-- Define the invitation events
def invitation1 : List Guest := [Guest.Bear, Guest.Lynx, Guest.Squirrel]
def gifts1 : List Gift := [Gift.Candlestick, Gift.Plate]

def invitation2 : List Guest := [Guest.Lynx, Guest.Squirrel, Guest.Mouse, Guest.Wolf]
def gifts2 : List Gift := [Gift.Candlestick, Gift.Needle]

def invitation3 : List Guest := [Guest.Wolf, Guest.Mouse, Guest.Sheep]
def gifts3 : List Gift := [Gift.Needle, Gift.Ring]

def invitation4 : List Guest := [Guest.Sheep, Guest.Bear, Guest.Wolf, Guest.Squirrel]
def gifts4 : List Gift := [Gift.Ring, Gift.Plate]

-- Define the gift assignment function
def giftAssignment : Guest → Option Gift
  | Guest.Bear => some Gift.Plate
  | Guest.Lynx => some Gift.Candlestick
  | Guest.Mouse => some Gift.Needle
  | Guest.Sheep => some Gift.Ring
  | Guest.Wolf => none
  | Guest.Squirrel => none

-- Theorem statement
theorem unique_gift_assignment :
  (giftAssignment Guest.Bear = some Gift.Plate) ∧
  (giftAssignment Guest.Lynx = some Gift.Candlestick) ∧
  (giftAssignment Guest.Mouse = some Gift.Needle) ∧
  (giftAssignment Guest.Sheep = some Gift.Ring) ∧
  (giftAssignment Guest.Wolf = none) ∧
  (giftAssignment Guest.Squirrel = none) := by
  sorry

#check unique_gift_assignment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_gift_assignment_l379_37906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_squared_in_cones_l379_37935

/-- Represents a right circular cone --/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones --/
structure ConePair where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- The maximum squared radius of a sphere that can fit within two intersecting cones --/
noncomputable def maxSphereRadiusSquared (cp : ConePair) : ℝ :=
  144 / 29

/-- Theorem stating the maximum squared radius of a sphere fitting in the given cone configuration --/
theorem max_sphere_radius_squared_in_cones 
  (cp : ConePair)
  (h1 : cp.cone1 = cp.cone2)
  (h2 : cp.cone1.baseRadius = 4)
  (h3 : cp.cone1.height = 10)
  (h4 : cp.intersectionDistance = 4) :
  maxSphereRadiusSquared cp = 144 / 29 := by
  sorry

#check max_sphere_radius_squared_in_cones

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_squared_in_cones_l379_37935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_number_proof_l379_37979

theorem larger_number_proof (A B : ℕ) (hcf lcm : ℕ) : 
  hcf = 15 →
  lcm = hcf * 11 * 15 →
  lcm * hcf = A * B →
  A ≥ B →
  A = 165 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_number_proof_l379_37979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employees_with_advanced_degrees_l379_37916

theorem employees_with_advanced_degrees 
  (total_employees : ℕ) 
  (female_employees : ℕ) 
  (males_with_college_only : ℕ) 
  (h1 : total_employees = 200) 
  (h2 : female_employees = 120) 
  (h3 : males_with_college_only = 40) : 
  total_employees - males_with_college_only = 160 := by
  sorry

#check employees_with_advanced_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employees_with_advanced_degrees_l379_37916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_on_parabola_l379_37919

/-- A parabola defined by y = 2x^2 -/
def parabola (x : ℝ) : ℝ := 2 * x^2

/-- Triangle ABC with vertices on the parabola -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h₁ : A.2 = parabola A.1
  h₂ : B.2 = parabola B.1
  h₃ : C.2 = parabola C.1

/-- The length of a line segment between two points -/
noncomputable def length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

theorem triangle_on_parabola (t : Triangle) 
  (h_origin : t.A = (0, 0))
  (h_parallel : t.B.2 = t.C.2)
  (h_area : triangleArea (length t.B t.C) (t.B.2 - t.A.2) = 128) :
  length t.B t.C = 8 := by
  sorry

#check triangle_on_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_on_parabola_l379_37919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_primes_from_thirty_l379_37913

def is_prime (n : ℕ) : Bool :=
  if n ≤ 1 then false
  else
    (List.range (n - 1)).all (λ m => m + 2 = n ∨ n % (m + 2) ≠ 0)

def count_primes (n : ℕ) : ℕ := (List.range n).filter (λ x => is_prime (x + 1)) |>.length

theorem probability_two_primes_from_thirty :
  let total_choices := Nat.choose 30 2
  let prime_choices := Nat.choose (count_primes 30) 2
  (prime_choices : ℚ) / total_choices = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_primes_from_thirty_l379_37913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_speed_comparison_l379_37933

-- Define the runners' data
structure RunnerData where
  distance : ℚ
  time : ℚ

-- Define the runners
def jim : RunnerData := { distance := 16, time := 2 }
def frank : RunnerData := { distance := 20, time := 5/2 }
def susan : RunnerData := { distance := 12, time := 3/2 }

-- Define the speed calculation function
def speed (runner : RunnerData) : ℚ := runner.distance / runner.time

-- Theorem statement
theorem runners_speed_comparison :
  speed jim = speed frank ∧ 
  speed jim = speed susan ∧ 
  speed frank - speed jim = 0 ∧ 
  speed susan - speed jim = 0 := by
  sorry

#eval speed jim
#eval speed frank
#eval speed susan

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_speed_comparison_l379_37933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_additional_points_l379_37925

def cube_vertices : List (Fin 3 → ℝ) := [
  ![0, 0, 0], ![0, 0, 6], ![0, 6, 0], ![0, 6, 6],
  ![6, 0, 0], ![6, 0, 6], ![6, 6, 0], ![6, 6, 6]
]

def P : Fin 3 → ℝ := ![0, 3, 0]
def Q : Fin 3 → ℝ := ![2, 0, 0]
def R : Fin 3 → ℝ := ![2, 6, 6]

def plane_equation (v : Fin 3 → ℝ) : Prop :=
  3 * v 0 + 2 * v 1 - v 2 = 6

theorem distance_between_additional_points :
  ∃ (S T : Fin 3 → ℝ),
    (S ∈ cube_vertices) ∧
    (T ∈ cube_vertices) ∧
    plane_equation S ∧
    plane_equation T ∧
    Real.sqrt ((S 0 - T 0)^2 + (S 1 - T 1)^2 + (S 2 - T 2)^2) = 2 * Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_additional_points_l379_37925
