import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1295_129519

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_properties (ω φ : ℝ) :
  ω > 0 →
  0 ≤ φ ∧ φ ≤ π →
  (∀ x, f ω φ x = f ω φ (-x)) →
  (∀ x, f ω φ (3*π/4 - x) = -f ω φ (3*π/4 + x)) →
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π/2 → f ω φ x ≥ f ω φ y ∨ f ω φ x ≤ f ω φ y) →
  (φ = π/2 ∧ (ω = 2/3 ∨ ω = 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1295_129519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_derivative_at_one_l1295_129500

noncomputable def f (x : ℝ) : ℝ := Real.cos (x^3 - 4*x^2 + 5*x - 2)

theorem tenth_derivative_at_one (fact_10 : Nat.factorial 10 = 3628800) :
  (deriv^[10] f) 1 = 907200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_derivative_at_one_l1295_129500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_trigonometric_expression_l1295_129535

theorem range_of_trigonometric_expression (α β : ℝ) 
  (h : Real.sin α + 2 * Real.cos β = 2) :
  ∃ (y : ℝ), y = Real.sin (α + π/4) + 2 * Real.sin (β + π/4) ∧ 
  Real.sqrt 2 - Real.sqrt 10 / 2 ≤ y ∧ y ≤ Real.sqrt 2 + Real.sqrt 10 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_trigonometric_expression_l1295_129535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_probability_l1295_129527

/-- Represents a jump in 2D space -/
structure Jump where
  length : ℝ
  direction : ℝ × ℝ

/-- Represents the frog's movement -/
def FrogMovement :=
  (start : ℝ × ℝ) → (jumps : List Jump) → ℝ × ℝ

/-- Calculates the distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The probability of the frog's final position being within 2 meters of the start -/
noncomputable def probability_within_2m (m : FrogMovement) : ℝ :=
  sorry

theorem frog_jump_probability :
  ∀ (m : FrogMovement) (start : ℝ × ℝ),
    let jumps := [
      ⟨1, (sorry : ℝ × ℝ)⟩,
      ⟨2, (sorry : ℝ × ℝ)⟩,
      ⟨2, (sorry : ℝ × ℝ)⟩,
      ⟨3, (sorry : ℝ × ℝ)⟩
    ]
    let end_pos := m start jumps
    probability_within_2m m = 1/5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_probability_l1295_129527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_sin_l1295_129578

/-- If the terminal side of angle α passes through the point (2cos 120°, √2sin 225°), then sin α = -√2/2 -/
theorem angle_terminal_side_sin (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 2 * Real.cos (120 * π / 180) ∧ 
                  r * Real.sin α = Real.sqrt 2 * Real.sin (225 * π / 180)) → 
  Real.sin α = -(Real.sqrt 2)/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_sin_l1295_129578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_l1295_129592

-- Define the points and circles
variable (A B C D P Q R : EuclideanSpace ℝ (Fin 2))
variable (circleA circleB circleC circleD : Set (EuclideanSpace ℝ (Fin 2)))

-- Define the radii
variable (rA rB rC rD : ℝ)

-- Define the conditions
axiom distinct_circles : circleA ≠ circleB ∧ circleA ≠ circleC ∧ circleA ≠ circleD ∧ 
                         circleB ≠ circleC ∧ circleB ≠ circleD ∧ circleC ≠ circleD

axiom P_on_circles : P ∈ circleA ∧ P ∈ circleB ∧ P ∈ circleC ∧ P ∈ circleD
axiom Q_on_circles : Q ∈ circleA ∧ Q ∈ circleB ∧ Q ∈ circleC ∧ Q ∈ circleD

axiom radius_relation1 : rA = (5/8) * rC
axiom radius_relation2 : rB = (5/8) * rD

axiom distance_AC : dist A C = 50
axiom distance_BD : dist B D = 50
axiom distance_PQ : dist P Q = 52

axiom R_midpoint : R = (1/2) • (P + Q)

-- The theorem to prove
theorem sum_of_distances : dist A R + dist B R + dist C R + dist D R = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_l1295_129592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eiffel_tower_scale_l1295_129548

/-- Represents the scale relationship between a tower and its model --/
structure TowerScale where
  tower_height : ℝ  -- Height of the tower in meters
  model_height : ℝ  -- Height of the model in centimeters

/-- Calculates the scale factor between the tower and its model --/
noncomputable def scale_factor (ts : TowerScale) : ℝ :=
  (ts.tower_height * 100) / ts.model_height

/-- Theorem: For the Eiffel Tower (324m) and its model (50cm),
    1 cm of the model represents 6.48m of the actual tower --/
theorem eiffel_tower_scale :
  let ts : TowerScale := ⟨324, 50⟩
  scale_factor ts = 6.48 := by
  sorry

#eval Float.toString ((324 * 100) / 50)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eiffel_tower_scale_l1295_129548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_theorem_l1295_129565

-- Define the trapezium ABCD
structure Trapezium where
  A : Real × Real
  B : Real × Real
  C : Real × Real
  D : Real × Real
  O : Real × Real
  ab_parallel_dc : (B.1 - A.1) * (D.2 - C.2) = (D.1 - C.1) * (B.2 - A.2)
  ac_bd_intersect : ∃ t s, O = (1 - t) • A + t • C ∧ O = (1 - s) • B + s • D

-- Define area function (simplified for this example)
def Area (p q r : Real × Real) : Real :=
  let (x1, y1) := p
  let (x2, y2) := q
  let (x3, y3) := r
  0.5 * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

-- Define the theorem
theorem trapezium_area_theorem (ABCD : Trapezium) 
  (area_AOB : Area ABCD.A ABCD.O ABCD.B = 16)
  (area_COD : Area ABCD.C ABCD.O ABCD.D = 25) :
  Area ABCD.A ABCD.B ABCD.C + Area ABCD.A ABCD.C ABCD.D = 81 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_theorem_l1295_129565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_I_l1295_129522

noncomputable def cubic_function (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d

noncomputable def integral_condition (b c d : ℝ) : Prop :=
  ∫ x in (-1)..1, (b * x^2 + c * x + d) = 1

noncomputable def second_derivative (a b : ℝ) : ℝ → ℝ := λ x ↦ 6 * a * x + 2 * b

noncomputable def I (a b : ℝ) : ℝ := ∫ x in (-1)..(1/2), (second_derivative a b x)^2

theorem min_value_I (a b c d : ℝ) :
  cubic_function a b c d 1 = 1 →
  cubic_function a b c d (-1) = -1 →
  integral_condition b c d →
  ∃ (min_I : ℝ), min_I = 81/128 ∧ ∀ (a' b' c' d' : ℝ),
    cubic_function a' b' c' d' 1 = 1 →
    cubic_function a' b' c' d' (-1) = -1 →
    integral_condition b' c' d' →
    I a' b' ≥ min_I := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_I_l1295_129522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_product_l1295_129531

noncomputable def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 = 1}

noncomputable def LeftFocus : ℝ × ℝ := (-Real.sqrt 2, 0)

noncomputable def RightFocus : ℝ × ℝ := (Real.sqrt 2, 0)

noncomputable def FocalAngle (p : ℝ × ℝ) : ℝ :=
  Real.arccos ((p.1 - LeftFocus.1)*(p.1 - RightFocus.1) + p.2^2) /
    (Real.sqrt ((p.1 - LeftFocus.1)^2 + p.2^2) * Real.sqrt ((p.1 - RightFocus.1)^2 + p.2^2))

theorem hyperbola_focal_product (p : ℝ × ℝ) 
    (h1 : p ∈ Hyperbola) 
    (h2 : FocalAngle p = π / 3) : 
  Real.sqrt ((p.1 - LeftFocus.1)^2 + p.2^2) * Real.sqrt ((p.1 - RightFocus.1)^2 + p.2^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_product_l1295_129531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_ellipse_with_specific_foci_l1295_129503

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parametric curve in 2D space -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ
  b : ℝ

/-- Checks if a given parametric curve is an ellipse -/
def isEllipse (curve : ParametricCurve) (e : Ellipse) : Prop :=
  ∀ θ : ℝ, (curve.x θ - e.center.x)^2 / e.a^2 + (curve.y θ - e.center.y)^2 / e.b^2 = 1

/-- Calculates the foci of an ellipse -/
noncomputable def ellipseFoci (e : Ellipse) : (Point × Point) :=
  let c := Real.sqrt (e.a^2 - e.b^2)
  ({ x := e.center.x - c, y := e.center.y }, { x := e.center.x + c, y := e.center.y })

/-- The main theorem to prove -/
theorem curve_is_ellipse_with_specific_foci :
  let curve : ParametricCurve := { x := λ θ => 4 * Real.cos θ, y := λ θ => 3 * Real.sin θ }
  let e : Ellipse := { center := { x := 0, y := 0 }, a := 4, b := 3 }
  isEllipse curve e ∧ ellipseFoci e = ({ x := -Real.sqrt 7, y := 0 }, { x := Real.sqrt 7, y := 0 }) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_ellipse_with_specific_foci_l1295_129503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_butchers_shop_cost_l1295_129508

/-- Calculates the total cost at the butcher's shop given specific conditions --/
theorem butchers_shop_cost : 
  ∃ (total_after_tax : ℚ),
  let steak_weight : ℚ := 3/2;
  let steak_price : ℚ := 15;
  let chicken_weight : ℚ := 3/2;
  let chicken_price : ℚ := 8;
  let sausage_weight : ℚ := 2;
  let sausage_price : ℚ := 13/2;
  let pork_weight : ℚ := 7/2;
  let pork_price : ℚ := 10;
  let bacon_weight : ℚ := 1/2;
  let bacon_price : ℚ := 9;
  let salmon_weight : ℚ := 1/4;
  let salmon_price : ℚ := 30;
  let salmon_discount : ℚ := 1/10;
  let pork_discount_weight : ℚ := 1;
  let sales_tax : ℚ := 6/100;

  let steak_cost := steak_weight * steak_price;
  let chicken_cost := chicken_weight * chicken_price;
  let sausage_cost := sausage_weight * sausage_price;
  let pork_cost := pork_weight * pork_price - pork_discount_weight * pork_price;
  let bacon_cost := bacon_weight * bacon_price;
  let salmon_cost := salmon_weight * salmon_price * (1 - salmon_discount);

  let total_before_tax := steak_cost + chicken_cost + sausage_cost + pork_cost + bacon_cost + salmon_cost;
  total_after_tax = total_before_tax * (1 + sales_tax) ∧ total_after_tax = 8878/100 := by
  sorry

#eval (8878:ℚ)/100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_butchers_shop_cost_l1295_129508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_condition_l1295_129587

/-- O'A' is a function of α and ρ, but we don't need to define it explicitly for the statement -/
noncomputable def OPrimeAPrime (α : Real) (ρ : Real) : Real :=
  sorry

/-- IsTriangle predicate -/
def IsTriangle (A B C : Real × Real) : Prop :=
  sorry

/-- AngleAt function -/
def AngleAt (A B C : Real × Real) : Real :=
  sorry

/-- AltitudeLength function -/
def AltitudeLength (A B C : Real × Real) : Real :=
  sorry

/-- InscribedCircleRadius function -/
def InscribedCircleRadius (A B C : Real × Real) : Real :=
  sorry

/-- Statement: A triangle with given angle, altitude, and inscribed circle radius exists
    if and only if certain conditions are met -/
theorem triangle_existence_condition (α m ρ : Real) :
  (∃ (A B C : Real × Real), 
    IsTriangle A B C ∧ 
    AngleAt A B C = α ∧
    AltitudeLength A B C = m ∧
    InscribedCircleRadius A B C = ρ) ↔ 
  (m > 2 * ρ ∧ OPrimeAPrime α ρ > ρ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_condition_l1295_129587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probabilities_l1295_129558

def XiaoLiCards : Finset ℕ := {1, 2, 3, 5}
def BrotherCards : Finset ℕ := {4, 6, 7, 8}

def isEven (n : ℕ) : Bool := n % 2 = 0

def XiaoLiWins (x : ℕ) (y : ℕ) : Bool := isEven (x + y)

theorem game_probabilities :
  let totalOutcomes := (XiaoLiCards.card * BrotherCards.card : ℚ)
  let xiaoLiWinningOutcomes := Finset.card (XiaoLiCards.filter (fun x => 
    Finset.card (BrotherCards.filter (fun y => XiaoLiWins x y)) > 0))
  (xiaoLiWinningOutcomes : ℚ) / totalOutcomes = 3 / 8 ∧
  (totalOutcomes - xiaoLiWinningOutcomes) / totalOutcomes = 5 / 8 := by
  sorry

#eval XiaoLiCards.card * BrotherCards.card
#eval Finset.card (XiaoLiCards.filter (fun x => 
  Finset.card (BrotherCards.filter (fun y => XiaoLiWins x y)) > 0))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probabilities_l1295_129558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_is_isosceles_triangle_l1295_129550

-- Define the three lines
def line1 (x : ℝ) : ℝ := 2 * x + 3
def line2 (x : ℝ) : ℝ := -2 * x + 3
def line3 : ℝ := -3

-- Define the intersection points
def point1 : ℝ × ℝ := (0, 3)
def point2 : ℝ × ℝ := (-3, -3)
def point3 : ℝ × ℝ := (3, -3)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem polygon_is_isosceles_triangle :
  let side1 := distance point1 point2
  let side2 := distance point1 point3
  let side3 := distance point2 point3
  (side1 = side2 ∧ side1 ≠ side3) ∨ (side1 = side3 ∧ side1 ≠ side2) ∨ (side2 = side3 ∧ side2 ≠ side1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_is_isosceles_triangle_l1295_129550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_fraction_of_jar_x_l1295_129524

/-- Represents the capacity of a jar --/
structure Capacity where
  volume : ℝ
  positive : volume > 0

/-- Represents a jar with water --/
structure Jar where
  capacity : Capacity
  water : ℝ
  water_nonneg : water ≥ 0
  water_le_capacity : water ≤ capacity.volume

/-- The fraction of a jar that is filled with water --/
noncomputable def fraction_filled (j : Jar) : ℝ :=
  j.water / j.capacity.volume

theorem initial_fraction_of_jar_x (jar_x jar_y : Jar) : 
  jar_y.capacity.volume = jar_x.capacity.volume / 2 →
  fraction_filled jar_y = 1/2 →
  fraction_filled { capacity := jar_x.capacity,
                    water := jar_x.water + jar_y.water,
                    water_nonneg := by
                      apply add_nonneg
                      exact jar_x.water_nonneg
                      exact jar_y.water_nonneg,
                    water_le_capacity := by sorry } = 3/4 →
  fraction_filled jar_x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_fraction_of_jar_x_l1295_129524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l1295_129582

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * x^2 - b * Real.log x

-- Define the function g
noncomputable def g (a b x : ℝ) : ℝ := f a b x - x^2 + (x - 1)

-- Main theorem
theorem tangent_line_and_monotonicity 
  (a b : ℝ) 
  (h1 : f a b 1 = 1) 
  (h2 : (deriv (f a b)) 1 = 0) :
  (a = 2 ∧ b = 4) ∧ 
  (∀ x, x ∈ Set.Ioc 0 1 → 
    (MonotoneOn (g a b) (Set.Ioc 0 1) ↔ a ≤ 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l1295_129582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drink_becomes_harmful_after_4_5_hours_l1295_129514

/-- Represents the volume of the drink in milliliters -/
def drink_volume : ℝ := 200

/-- Represents the initial concentration of bacteria A per milliliter -/
def initial_concentration : ℝ := 20

/-- Represents the initial number of bacteria A in the drink -/
def initial_bacteria : ℝ := drink_volume * initial_concentration

/-- Represents the harmful threshold for the number of bacteria A -/
def harmful_threshold : ℝ := 90000

/-- Represents the number of bacteria A after x hours -/
noncomputable def bacteria_count (x : ℝ) : ℝ := initial_bacteria * (2 : ℝ) ^ x

/-- Theorem stating that the drink becomes harmful after 4.5 hours -/
theorem drink_becomes_harmful_after_4_5_hours :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ 4.5 ∧ bacteria_count x < harmful_threshold ∧
  ∀ y : ℝ, y > 4.5 → bacteria_count y > harmful_threshold := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_drink_becomes_harmful_after_4_5_hours_l1295_129514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_k_values_l1295_129547

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the line on which point A lies
def line_A (x y k : ℝ) : Prop := y = k*x - 3

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define the condition for non-intersection
def non_intersect (k : ℤ) : Prop :=
  ∀ x y : ℝ, line_A x y (k : ℝ) → distance x y 1 0 ≥ 2

-- State the theorem
theorem possible_k_values :
  ∀ k : ℤ, non_intersect k ↔ k = -2 ∨ k = -1 ∨ k = 0 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_k_values_l1295_129547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1295_129557

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2*x - 3) + 1 / (x - 3)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 3/2 ∧ x ≠ 3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1295_129557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_positivity_l1295_129588

noncomputable def f (a x : ℝ) := a^2 * x^2 + a*x - 3*Real.log x + 1

noncomputable def f' (a x : ℝ) := 2*a^2*x + a - 3/x

theorem f_monotonicity_and_positivity (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/a), HasDerivAt (f a) (f' a x) x ∧ f' a x < 0) ∧
  (∀ x ∈ Set.Ioi (1/a), HasDerivAt (f a) (f' a x) x ∧ f' a x > 0) ∧
  (∀ x > 0, f a x > 0 ↔ a > 1/Real.exp 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_positivity_l1295_129588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1295_129518

/-- The inclination angle of a line with slope k -/
noncomputable def inclinationAngle (k : ℝ) : ℝ := Real.arctan k * (180 / Real.pi)

/-- The slope of the line x + y + 1 = 0 -/
def lineSlope : ℝ := -1

theorem line_inclination_angle :
  inclinationAngle lineSlope = 135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1295_129518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l1295_129520

noncomputable def f (a : ℝ) (x : ℕ+) : ℝ := (x.val^2 + a * x.val + 11) / (x.val + 1)

theorem min_a_value (a : ℝ) (h : ∀ x : ℕ+, f a x ≥ 3) : a ≥ -8/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l1295_129520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_power_12_sum_not_neg_one_l1295_129523

/-- ω is a complex number defined as (-1 + i√3) / 2 -/
noncomputable def ω : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2

/-- ω² is a complex number defined as (-1 - i√3) / 2 -/
noncomputable def ω_squared : ℂ := (-1 - Complex.I * Real.sqrt 3) / 2

/-- The sum of ω¹² and (ω²)¹² is not equal to -1 -/
theorem omega_power_12_sum_not_neg_one : ω^12 + ω_squared^12 ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_power_12_sum_not_neg_one_l1295_129523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1295_129512

/-- Hyperbola C with foci F₁ and F₂, and point P on its right branch -/
structure HyperbolaC where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  heq : c^2 = a^2 + b^2
  F₁ : ℝ × ℝ := (-c, 0)
  F₂ : ℝ × ℝ := (c, 0)
  P : ℝ × ℝ
  hP : P.1^2 / a^2 - P.2^2 / b^2 = 1
  hPright : P.1 > 0

/-- Inscribed circle of triangle PF₁F₂ -/
structure InscribedCircle (h : HyperbolaC) where
  M : ℝ × ℝ  -- Center of the inscribed circle
  r : ℝ      -- Radius of the inscribed circle
  hr : r = h.a

/-- Centroid of triangle PF₁F₂ -/
noncomputable def centroid (h : HyperbolaC) : ℝ × ℝ :=
  ((h.F₁.1 + h.F₂.1 + h.P.1) / 3, (h.F₁.2 + h.F₂.2 + h.P.2) / 3)

/-- The eccentricity of a hyperbola is the ratio of the distance between the foci to the distance between the vertices -/
noncomputable def eccentricity (h : HyperbolaC) : ℝ := h.c / h.a

/-- Main theorem: The eccentricity of hyperbola C is 2 -/
theorem hyperbola_eccentricity (h : HyperbolaC) (i : InscribedCircle h) 
    (hMG : (i.M.1 - (centroid h).1) / (i.M.2 - (centroid h).2) = (h.F₂.1 - h.F₁.1) / (h.F₂.2 - h.F₁.2)) :
    eccentricity h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1295_129512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_of_roots_l1295_129507

theorem sine_sum_of_roots (x₁ x₂ m : ℝ) : 
  x₁ ∈ Set.Icc 0 π → 
  x₂ ∈ Set.Icc 0 π → 
  2 * Real.sin x₁ + Real.cos x₁ = m → 
  2 * Real.sin x₂ + Real.cos x₂ = m → 
  Real.sin (x₁ + x₂) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_of_roots_l1295_129507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_percentages_l1295_129545

theorem difference_of_percentages : ℝ := by
  -- Define the given number
  let number : ℝ := 800

  -- Define the percentage calculations
  let forty_percent_of_number : ℝ := 0.4 * number
  let twenty_percent_of_650 : ℝ := 0.2 * 650

  -- Calculate the difference
  let difference : ℝ := forty_percent_of_number - twenty_percent_of_650

  -- State and prove the theorem
  have : difference = 190 := by
    -- Proof steps would go here
    sorry

  exact difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_percentages_l1295_129545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_1998_terms_l1295_129597

def sequenceA : ℕ → ℕ
  | 0 => 1
  | n + 1 => sorry

theorem sum_of_first_1998_terms : 
  (∀ n, sequenceA n = 1 ∨ sequenceA n = 2) →
  sequenceA 0 = 1 →
  (∀ k, ∃ m, 
    (∀ i < 2^(k-1), sequenceA (m + i) = 2) ∧
    sequenceA (m + 2^(k-1)) = 1) →
  (Finset.range 1998).sum (fun i => sequenceA i) = 3985 := by
  sorry

#check sum_of_first_1998_terms

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_1998_terms_l1295_129597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_tangent_circle_l1295_129555

/-- The line l: x + y - 2 = 0 -/
def line (x y : ℝ) : Prop := x + y - 2 = 0

/-- The curve: x^2 + y^2 - 12x - 12y + 54 = 0 -/
def curve (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 12*y + 54 = 0

/-- The circle with center (2, 2) and radius √2 -/
def target_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 2

/-- A predicate to check if a circle is tangent to the line -/
def tangent_to_line (c : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, c x y ∧ line x y

/-- A predicate to check if a circle is tangent to the curve -/
def tangent_to_curve (c : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, c x y ∧ curve x y

/-- The main theorem stating that the given circle is the unique smallest circle
    tangent to both the line and the curve -/
theorem smallest_tangent_circle :
  (tangent_to_line target_circle) ∧
  (tangent_to_curve target_circle) ∧
  (∀ c : ℝ → ℝ → Prop, c ≠ target_circle →
    (tangent_to_line c ∧ tangent_to_curve c) →
    ∃ r₁ r₂, (∀ x y, c x y → (x - 2)^2 + (y - 2)^2 = r₁^2) ∧
             (∀ x y, target_circle x y → (x - 2)^2 + (y - 2)^2 = r₂^2) ∧
             r₁ > r₂) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_tangent_circle_l1295_129555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l1295_129502

/-- The parabola on which point P moves -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- The circle on which point P1 moves -/
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 1

/-- The distance from a point to the x-axis -/
def dist_to_x_axis (y : ℝ) : ℝ := |y|

/-- The distance between two points -/
noncomputable def dist_between_points (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The main theorem -/
theorem min_distance_point (x y x1 y1 : ℝ) :
  parabola x y →
  circle_eq x1 y1 →
  (∀ x' y' x1' y1', parabola x' y' → circle_eq x1' y1' →
    dist_to_x_axis y + dist_between_points x y x1 y1 ≤
    dist_to_x_axis y' + dist_between_points x' y' x1' y1') →
  x = 2 * Real.sqrt 2 - 2 ∧ y = 3 - 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l1295_129502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_products_nonzero_l1295_129586

/-- Represents a 2007x2007 array of 1 and -1 -/
def Array2007 := Fin 2007 → Fin 2007 → Int

/-- Condition that each entry in the array is either 1 or -1 -/
def validArray (arr : Array2007) : Prop :=
  ∀ i j, arr i j = 1 ∨ arr i j = -1

/-- Product of entries in the i-th row -/
def rowProduct (arr : Array2007) (i : Fin 2007) : Int :=
  (Finset.univ.prod fun j => arr i j)

/-- Product of entries in the j-th column -/
def colProduct (arr : Array2007) (j : Fin 2007) : Int :=
  (Finset.univ.prod fun i => arr i j)

/-- Sum of all row products and column products -/
def sumProducts (arr : Array2007) : Int :=
  (Finset.univ.sum fun i => rowProduct arr i) + (Finset.univ.sum fun j => colProduct arr j)

/-- Theorem: The sum of all row products and column products is not zero -/
theorem sum_products_nonzero (arr : Array2007) (h : validArray arr) :
  sumProducts arr ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_products_nonzero_l1295_129586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midway_distance_is_6_5_l1295_129515

/-- Represents the properties of an elliptical orbit -/
structure EllipticalOrbit where
  perigee : ℝ
  apogee : ℝ

/-- Calculates the distance from the focus to a point midway along the orbit -/
noncomputable def midwayDistance (orbit : EllipticalOrbit) : ℝ :=
  (orbit.apogee + orbit.perigee) / 2

/-- Theorem stating that for the given orbit, the midway distance is 6.5 AU -/
theorem midway_distance_is_6_5 (orbit : EllipticalOrbit) 
  (h1 : orbit.perigee = 3) 
  (h2 : orbit.apogee = 10) : 
  midwayDistance orbit = 6.5 := by
  -- Unfold the definition of midwayDistance
  unfold midwayDistance
  -- Substitute the given values
  rw [h1, h2]
  -- Simplify the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midway_distance_is_6_5_l1295_129515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_monotone_increasing_l1295_129554

-- Define the function f(x) = log₁₀(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_monotone_increasing :
  MonotoneOn f (Set.Ioi 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_monotone_increasing_l1295_129554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_array_sum_theorem_remainder_theorem_l1295_129511

/-- Represents the sum of all terms in a 1/p-array -/
def arraySum (p : ℕ) : ℚ :=
  2 * p^2 / ((2*p - 1) * (p^2 - 1))

/-- Represents the numerator and denominator of the simplified fraction form of arraySum -/
def simplifiedFraction (p : ℕ) : ℤ × ℤ :=
  let sum := arraySum p
  (sum.num, sum.den)

theorem array_sum_theorem (p : ℕ) (hp : p > 1) :
  arraySum p = ∑' (r : ℕ), ∑' (c : ℕ), (1 / (2*p)^r) * (1 / p^(2*c)) :=
sorry

theorem remainder_theorem :
  let (m, n) := simplifiedFraction 2008
  (m + n).natAbs % 2008 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_array_sum_theorem_remainder_theorem_l1295_129511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_shaded_region_l1295_129559

/-- Line passing through two points -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- Rectangle with fixed width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The combined shaded region -/
def ShadedRegion (l₁ l₂ : Line) (rect : Rectangle) : Set (ℝ × ℝ) := sorry

/-- Measure of a set in ℝ² -/
noncomputable def areaOfSet (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem area_of_shaded_region :
  let l₁ : Line := { x₁ := 0, y₁ := 5, x₂ := 10, y₂ := 2 }
  let l₂ : Line := { x₁ := 2, y₁ := 6, x₂ := 9, y₂ := 1 }
  let rect : Rectangle := { width := 2, height := 5 }
  areaOfSet (ShadedRegion l₁ l₂ rect) = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_shaded_region_l1295_129559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_difference_l1295_129530

theorem set_equality_implies_difference (a b : ℝ) : 
  ({a, 1} : Set ℝ) = ({0, a + b} : Set ℝ) → b - a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_difference_l1295_129530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_without_nine_bound_l1295_129540

/-- A function that checks if a natural number contains the digit 9 in its decimal representation -/
def contains_nine (n : ℕ) : Bool :=
  sorry

/-- The sum of reciprocals of numbers from 1 to n that don't contain 9 in their decimal representation -/
noncomputable def sum_without_nine (n : ℕ) : ℝ :=
  (Finset.range (n + 1)).sum (λ i => if ¬(contains_nine i) ∧ i ≠ 0 then (1 : ℝ) / i else 0)

/-- Theorem stating that the sum of reciprocals without 9 is less than 80 for any n -/
theorem sum_without_nine_bound {n : ℕ} : sum_without_nine n < 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_without_nine_bound_l1295_129540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1295_129567

theorem equation_solution : ∀ y : ℝ, y > 2 → (Real.sqrt (8 * y) / Real.sqrt (2 * (y - 2)) = 3) → y = 18 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1295_129567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_tint_percentage_after_addition_l1295_129579

/-- Represents a paint mixture -/
structure PaintMixture where
  volume : ℝ
  redTint : ℝ
  yellowTint : ℝ
  water : ℝ

/-- Calculates the percentage of yellow tint in a paint mixture -/
noncomputable def yellowTintPercentage (mixture : PaintMixture) : ℝ :=
  (mixture.yellowTint / mixture.volume) * 100

/-- Theorem: The percentage of yellow tint in the new mixture is 32% -/
theorem yellow_tint_percentage_after_addition 
  (initialMixture : PaintMixture)
  (addedYellowTint : ℝ) :
  initialMixture.volume = 40 ∧
  initialMixture.redTint = 0.35 * initialMixture.volume ∧
  initialMixture.yellowTint = 0.15 * initialMixture.volume ∧
  initialMixture.water = 0.50 * initialMixture.volume ∧
  addedYellowTint = 10 →
  let newMixture : PaintMixture := {
    volume := initialMixture.volume + addedYellowTint,
    redTint := initialMixture.redTint,
    yellowTint := initialMixture.yellowTint + addedYellowTint,
    water := initialMixture.water
  }
  yellowTintPercentage newMixture = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_tint_percentage_after_addition_l1295_129579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bandi_jan_1_tie_l1295_129534

-- Define the friends
inductive Friend
| Aladar
| Bandi
| Feri
| Pista
| Geza

-- Define the tie colors
inductive Color
| Blue
| Red
| Green
| Yellow
| White

-- Define a function to represent the number of ties each friend has
def numTies : Friend → Nat := sorry

-- Define a function to represent the tie color worn by a friend on a given day
def tieWorn : Friend → Nat → Color := sorry

-- Axioms representing the conditions
axiom tie_count_range (f : Friend) : 2 ≤ numTies f ∧ numTies f < 12

axiom unique_tie_count (f1 f2 : Friend) : f1 ≠ f2 → numTies f1 ≠ numTies f2

axiom cycle_ties (f : Friend) (d : Nat) : 
  tieWorn f (d + numTies f) = tieWorn f d

axiom dec_1_ties : 
  tieWorn Friend.Aladar 1 = Color.Blue ∧
  tieWorn Friend.Bandi 1 = Color.Red ∧
  tieWorn Friend.Feri 1 = Color.Red ∧
  tieWorn Friend.Pista 1 = Color.Green ∧
  tieWorn Friend.Geza 1 = Color.Yellow

axiom dec_19_ties :
  tieWorn Friend.Pista 19 = Color.Green ∧
  tieWorn Friend.Geza 19 = Color.Yellow ∧
  tieWorn Friend.Feri 19 = Color.Blue ∧
  (tieWorn Friend.Aladar 19 = Color.Red ∨ tieWorn Friend.Bandi 19 = Color.Red)

axiom pista_ties :
  tieWorn Friend.Pista 23 = Color.White ∧
  tieWorn Friend.Pista 26 = Color.Yellow

axiom dec_11_ties :
  ∃ (f1 f2 f3 f4 f5 : Friend),
    f1 ≠ f2 ∧ f1 ≠ f3 ∧ f1 ≠ f4 ∧ f1 ≠ f5 ∧
    f2 ≠ f3 ∧ f2 ≠ f4 ∧ f2 ≠ f5 ∧
    f3 ≠ f4 ∧ f3 ≠ f5 ∧
    f4 ≠ f5 ∧
    tieWorn f1 11 = Color.Yellow ∧
    tieWorn f2 11 = Color.Red ∧
    tieWorn f3 11 = Color.Blue ∧
    tieWorn f4 11 = Color.Green ∧
    tieWorn f5 11 = Color.White

axiom dec_31_ties (f : Friend) : tieWorn f 31 = tieWorn f 1

-- Theorem to prove
theorem bandi_jan_1_tie : tieWorn Friend.Bandi 32 = Color.Green := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bandi_jan_1_tie_l1295_129534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_g_l1295_129562

noncomputable def f (x : ℝ) : ℝ := 2^x

def is_valid_g (g : ℝ → ℝ) : Prop :=
  (∃ a b : ℝ, ∀ x, g x = a * x + b) ∧  -- g is linear
  (f (g 2) = 2) ∧                      -- (2, 2) lies on f[g(x)]
  (g (f 2) = 5)                        -- (2, 5) lies on g[f(x)]

theorem unique_g : 
  ∃! g : ℝ → ℝ, is_valid_g g ∧ ∀ x, g x = 2 * x - 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_g_l1295_129562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l1295_129594

open Set

def U : Set ℝ := univ

def M : Set ℝ := {x : ℝ | x^2 + x - 2 > 0}

def N : Set ℝ := {x : ℝ | (1/2 : ℝ)^(x-1) ≥ 2}

theorem set_intersection_theorem :
  (U \ M) ∩ N = Icc (-2) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l1295_129594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l1295_129599

theorem sin_beta_value (α β : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π/2)
  (h4 : Real.sin α = 3/5)
  (h5 : Real.cos (β - α) = 12/13) : 
  Real.sin β = 56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l1295_129599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crank_slider_properties_l1295_129551

/-- Crank-slider mechanism -/
structure CrankSlider where
  oa : ℝ  -- Length of OA
  ab : ℝ  -- Length of AB
  am : ℝ  -- Length of AM
  ω : ℝ   -- Angular velocity

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

noncomputable def CrankSlider.motion (cs : CrankSlider) (t : ℝ) : Point :=
  { x := 45 * (1 + Real.cos (cs.ω * t))
  , y := 45 * Real.sin (cs.ω * t) }

def CrankSlider.trajectory (cs : CrankSlider) (p : Point) : Prop :=
  (p.y / 45) ^ 2 + ((p.x - 45) / 45) ^ 2 = 1

noncomputable def CrankSlider.velocity (cs : CrankSlider) (t : ℝ) : Vector2D :=
  { x := -450 * Real.sin (cs.ω * t)
  , y := 450 * Real.cos (cs.ω * t) }

theorem crank_slider_properties (cs : CrankSlider) 
    (h1 : cs.oa = 90) (h2 : cs.ab = 90) (h3 : cs.am = cs.ab / 2) (h4 : cs.ω = 10) :
    (∀ t, cs.motion t = cs.motion t) ∧ 
    (∀ p, cs.trajectory p ↔ cs.trajectory p) ∧
    (∀ t, cs.velocity t = cs.velocity t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crank_slider_properties_l1295_129551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_consecutive_sides_double_angle_l1295_129556

theorem unique_triangle_consecutive_sides_double_angle :
  ∃! (a b c : ℕ), 
    (b = a + 1) ∧ 
    (c = b + 1) ∧ 
    (a > 0) ∧
    (∃ (A B C : ℝ), 
      (A > 0) ∧ (B > 0) ∧ (C > 0) ∧
      (A + B + C = Real.pi) ∧
      (c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)) ∧
      ((A = 2*B) ∨ (B = 2*C) ∨ (C = 2*A))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_consecutive_sides_double_angle_l1295_129556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_l1295_129549

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

variable (i j k : V)
variable (h_noncoplanar : ¬(∃ (α β γ : ℝ), α • i + β • j + γ • k = 0 ∧ (α ≠ 0 ∨ β ≠ 0 ∨ γ ≠ 0)))

noncomputable def a (V : Type*) [AddCommGroup V] [Module ℝ V] (i j k : V) : V := (1/2 : ℝ) • i - j + k
noncomputable def b (V : Type*) [AddCommGroup V] [Module ℝ V] (i j k : V) : V := 5 • i - 2 • j - k

theorem vector_equation (V : Type*) [AddCommGroup V] [Module ℝ V] (i j k : V) 
  (h_noncoplanar : ¬(∃ (α β γ : ℝ), α • i + β • j + γ • k = 0 ∧ (α ≠ 0 ∨ β ≠ 0 ∨ γ ≠ 0))) :
  4 • a V i j k - 3 • b V i j k = -13 • i + 2 • j + 7 • k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_l1295_129549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1295_129564

theorem inequality_proof (n : ℕ) (a : Fin n → ℝ) 
  (h_pos : ∀ i, a i > 0) : 
  let s := (Finset.univ.sum a)
  let p := (Finset.univ.prod a)
  2^n * Real.sqrt p ≤ (Finset.range (n+1)).sum (λ k ↦ s^k / k.factorial) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1295_129564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_result_l1295_129544

/-- The total number of participants in the survey -/
def total_participants : ℕ := 150 * 2

/-- The number of participants who chose option A -/
def option_A_participants : ℕ := 150

/-- The number of participants who chose option B -/
def option_B_participants : ℕ := 90

/-- The percentage of participants who chose option A -/
def option_A_percentage : ℚ := 1/2

/-- The percentage of participants who chose option B -/
def option_B_percentage : ℚ := 3/10

theorem survey_result : 
  option_B_participants = (option_B_percentage / option_A_percentage * option_A_participants).floor :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_result_l1295_129544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptarithm_solution_l1295_129516

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a mapping of letters to digits -/
def LetterMapping := Char → Digit

/-- Converts a string to a number using the given letter mapping -/
def stringToNumber (s : String) (mapping : LetterMapping) : ℕ :=
  s.foldl (fun acc c => 10 * acc + (mapping c).val) 0

/-- Checks if a letter mapping is valid for the given cryptarithm -/
def isValidMapping (mapping : LetterMapping) : Prop :=
  let mimimi := stringToNumber "MIMIMI" mapping
  let nyanya := stringToNumber "NYANYA" mapping
  let olalaoy := stringToNumber "OLALAOY" mapping
  mimimi + nyanya = olalaoy ∧
  mapping 'M' ≠ mapping 'N' ∧ mapping 'M' ≠ mapping 'Y' ∧ mapping 'M' ≠ mapping 'O' ∧
  mapping 'M' ≠ mapping 'L' ∧ mapping 'M' ≠ mapping 'A' ∧ mapping 'M' ≠ mapping 'I' ∧
  mapping 'N' ≠ mapping 'Y' ∧ mapping 'N' ≠ mapping 'O' ∧ mapping 'N' ≠ mapping 'L' ∧
  mapping 'N' ≠ mapping 'A' ∧ mapping 'N' ≠ mapping 'I' ∧ mapping 'Y' ≠ mapping 'O' ∧
  mapping 'Y' ≠ mapping 'L' ∧ mapping 'Y' ≠ mapping 'A' ∧ mapping 'Y' ≠ mapping 'I' ∧
  mapping 'O' ≠ mapping 'L' ∧ mapping 'O' ≠ mapping 'A' ∧ mapping 'O' ≠ mapping 'I' ∧
  mapping 'L' ≠ mapping 'A' ∧ mapping 'L' ≠ mapping 'I' ∧ mapping 'A' ≠ mapping 'I'

theorem cryptarithm_solution :
  ∃ (mapping : LetterMapping), isValidMapping mapping ∧
  stringToNumber "MI" mapping + stringToNumber "NY" mapping = 119 := by
  sorry

#check cryptarithm_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptarithm_solution_l1295_129516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_lines_intersection_count_l1295_129525

/-- A type representing a line in a plane -/
structure Line where
  id : ℕ

/-- A type representing an intersection point of two lines -/
structure IntersectionPoint where
  line1 : Line
  line2 : Line

/-- A set of five distinct lines in a plane -/
def five_lines : Finset Line := sorry

/-- Predicate to check if three lines are concurrent -/
def are_concurrent (l1 l2 l3 : Line) : Prop := sorry

/-- The set of all intersection points where exactly two lines intersect -/
def intersection_points (lines : Finset Line) : Finset IntersectionPoint := sorry

theorem five_lines_intersection_count :
  (∀ l1 l2 l3, l1 ∈ five_lines → l2 ∈ five_lines → l3 ∈ five_lines →
    l1 ≠ l2 → l2 ≠ l3 → l1 ≠ l3 → ¬are_concurrent l1 l2 l3) →
  Finset.card (intersection_points five_lines) = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_lines_intersection_count_l1295_129525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_solutions_grow_indefinitely_l1295_129568

theorem diophantine_equation_solutions (n : ℕ) :
  n ≥ 8 ∧ n ∉ ({9, 10, 12, 14} : Finset ℕ) →
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 5 * y = n :=
sorry

theorem solutions_grow_indefinitely :
  ∀ m : ℕ, ∃ N : ℕ, ∀ n : ℕ, n > N →
    ∃ S : Finset (ℕ × ℕ),
      S.card ≥ m ∧
      ∀ p : ℕ × ℕ, p ∈ S → p.1 > 0 ∧ p.2 > 0 ∧ 3 * p.1 + 5 * p.2 = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_solutions_grow_indefinitely_l1295_129568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_sum_ratio_l1295_129513

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a 1 + (n - 1 : ℝ) * (a 2 - a 1)) / 2

theorem min_value_of_sequence_sum_ratio
  (a : ℕ → ℝ) (d : ℝ) (n : ℕ) :
  arithmetic_sequence a d →
  d ≠ 0 →
  geometric_sequence (a 2) (a 3) (a 6) →
  a 10 = -17 →
  ∃ k : ℕ, ∀ m : ℕ, m ≥ 1 → 
    sum_of_arithmetic_sequence a k / (2 : ℝ)^k ≤ sum_of_arithmetic_sequence a m / (2 : ℝ)^m ∧
    sum_of_arithmetic_sequence a k / (2 : ℝ)^k = (-1 : ℝ) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_sum_ratio_l1295_129513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_distance_l1295_129526

/-- The speed of sound in feet per second -/
def speed_of_sound : ℝ := 1100

/-- The time delay between lightning and thunder in seconds -/
def time_delay : ℝ := 8

/-- The number of feet in a mile -/
def feet_per_mile : ℝ := 5280

/-- Rounds a number to the nearest half -/
noncomputable def round_to_nearest_half (x : ℝ) : ℝ :=
  ⌊x * 2 + 0.5⌋ / 2

/-- The theorem stating that the distance to the lightning strike is 1.5 miles -/
theorem lightning_distance : 
  round_to_nearest_half ((speed_of_sound * time_delay) / feet_per_mile) = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_distance_l1295_129526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_l1295_129528

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 0
  else if x = 0 then Real.pi
  else Real.pi^2 + 1

theorem f_composition : f (f (f (-1))) = Real.pi := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_l1295_129528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_unique_equation_l1295_129589

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ := (l.y₂ - l.y₁) / (l.x₂ - l.x₁)

/-- The equation of a line in the form ax + by = c -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Convert a Line to its equation -/
noncomputable def Line.toEquation (l : Line) : LineEquation :=
  let m := l.slope
  let b := l.y₁ - m * l.x₁
  { a := m, b := -1, c := b }

/-- Check if a point (x, y) lies on a line given by its equation -/
def LineEquation.containsPoint (eq : LineEquation) (x y : ℝ) : Prop :=
  eq.a * x + eq.b * y = eq.c

theorem line_through_points_unique_equation :
  ∀ (l : Line),
    l.x₁ = 2 ∧ l.y₁ = -3 ∧ l.x₂ = 0 ∧ l.y₂ = 0 →
    ∃ (eq : LineEquation),
      eq.a = 3 ∧ eq.b = 2 ∧ eq.c = 0 ∧
      eq.containsPoint l.x₁ l.y₁ ∧
      eq.containsPoint l.x₂ l.y₂ :=
by
  sorry

#check line_through_points_unique_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_unique_equation_l1295_129589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_cake_division_l1295_129585

theorem birthday_cake_division (total_pieces : ℕ) (eaten_percentage : ℚ) (num_sisters : ℕ) :
  total_pieces = 240 →
  eaten_percentage = 60 / 100 →
  num_sisters = 3 →
  (total_pieces - (eaten_percentage * ↑total_pieces).floor) / num_sisters = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_cake_division_l1295_129585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_shift_right_g_shift_is_translation_l1295_129506

-- Define the original piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -x - 2
  else if 0 < x ∧ x ≤ 2 then -Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 < x ∧ x ≤ 3 then 1.5 * (x - 2) + 2
  else 0  -- Define a default value for x outside the given ranges

-- State the theorem about the rightward shift
theorem g_shift_right (x : ℝ) : g (x - 2) = 
  if -1 ≤ x ∧ x ≤ 2 then -x
  else if 2 < x ∧ x ≤ 4 then -Real.sqrt (4 - (x - 4)^2) - 2
  else if 4 < x ∧ x ≤ 5 then 1.5 * (x - 4) + 2
  else 0 := by
  sorry

-- State that this shift represents a translation of 2 units to the right
theorem g_shift_is_translation : 
  ∀ x : ℝ, g (x - 2) = g x → x ∈ {y : ℝ | y = x + 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_shift_right_g_shift_is_translation_l1295_129506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_distance_l1295_129510

/-- Two lines intersecting at a point with given slopes have a specific distance between their x-intercepts -/
theorem x_intercept_distance (slope1 slope2 x0 y0 : ℝ) : 
  slope1 = 2 →
  slope2 = 6 →
  x0 = 40 →
  y0 = 30 →
  let line1 := λ x ↦ slope1 * (x - x0) + y0
  let line2 := λ x ↦ slope2 * (x - x0) + y0
  let x_intercept1 := (y0 - line1 0) / slope1 + x0
  let x_intercept2 := (y0 - line2 0) / slope2 + x0
  |x_intercept2 - x_intercept1| = 10 := by
  sorry

#check x_intercept_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_distance_l1295_129510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_l1295_129561

/-- The ellipse equation -/
noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 3)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 25

/-- The foci of the ellipse -/
def focus1 : ℝ × ℝ := (4, 3)
def focus2 : ℝ × ℝ := (-6, 9)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The distance between the foci of the ellipse is 4√34 -/
theorem distance_between_foci :
  distance focus1 focus2 = 4 * Real.sqrt 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_l1295_129561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_18pi_l1295_129517

/-- A room shaped like the region between two semicircles -/
structure Room where
  R : ℝ -- radius of the outer semicircle
  r : ℝ -- radius of the inner semicircle
  h : R > r -- outer radius is larger than inner radius

/-- The farthest distance between two points with a clear line of sight -/
noncomputable def maxDistance (room : Room) : ℝ := 2 * Real.sqrt (room.R^2 - room.r^2)

/-- The area of the room -/
noncomputable def roomArea (room : Room) : ℝ := Real.pi / 2 * (room.R^2 - room.r^2)

/-- Theorem: If the max distance is 12, then the area is 18π -/
theorem area_is_18pi (room : Room) (h : maxDistance room = 12) : roomArea room = 18 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_18pi_l1295_129517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l1295_129576

/-- The area of a square inscribed in the ellipse x²/3 + y²/6 = 1, with its sides parallel to the coordinate axes, is 8. -/
theorem inscribed_square_area : 
  ∃ (s : Set (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ s → (p.1)^2/3 + (p.2)^2/6 = 1) ∧ 
    (∃ (t : ℝ), s = {p : ℝ × ℝ | p.1 = t ∨ p.1 = -t ∨ p.2 = t ∨ p.2 = -t}) ∧
    (MeasureTheory.volume s = 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l1295_129576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l1295_129596

theorem vector_magnitude (a b : ℝ × ℝ) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) 
  (hangle : a.1 * b.1 + a.2 * b.2 = Real.cos (π / 3)) : 
  ‖(2 * a.1 - b.1, 2 * a.2 - b.2)‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l1295_129596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_l1295_129584

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x^2 + 2*x else (-x)^2 - 2*x

-- State the theorem
theorem solution_set_of_f (x : ℝ) :
  (∀ y : ℝ, f y = f (-y)) ∧  -- f is even
  (∀ y : ℝ, y ≤ 0 → f y = y^2 + 2*y) →  -- definition for y ≤ 0
  (f (x + 2) < 3 ↔ -5 < x ∧ x < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_l1295_129584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_observed_price_theorem_l1295_129575

/-- Calculates the observed price in local currency for an international buyer --/
noncomputable def observed_price_local_currency (
  producer_cost : ℝ)
  (shipping_cost : ℝ)
  (online_store_commission : ℝ)
  (tax_rate : ℝ)
  (profit_rate : ℝ)
  (exchange_rate : ℝ) : ℝ :=
  let base_cost := producer_cost + shipping_cost
  let tax_amount := tax_rate * base_cost
  let total_cost := base_cost + tax_amount
  let profit_amount := profit_rate * total_cost
  let price_before_commission := total_cost + profit_amount
  let distributor_price := price_before_commission / (1 - online_store_commission)
  distributor_price * exchange_rate

/-- Theorem stating that the observed price in local currency is 35.64 --/
theorem observed_price_theorem :
  observed_price_local_currency 19 5 0.20 0.10 0.20 0.90 = 35.64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_observed_price_theorem_l1295_129575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_four_solutions_l1295_129581

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the equation
def equation (x : ℝ) : Prop :=
  4 * x^2 - 40 * (floor x) + 51 = 0

-- Theorem statement
theorem equation_has_four_solutions :
  ∃ (s : Finset ℝ), s.card = 4 ∧ (∀ x ∈ s, equation x) ∧
    (∀ y : ℝ, equation y → y ∈ s) := by
  sorry

#check equation_has_four_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_four_solutions_l1295_129581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matt_peanut_plantation_revenue_l1295_129583

/-- Calculates the revenue from a square peanut plantation -/
noncomputable def peanut_plantation_revenue (side_length : ℝ) (peanut_yield : ℝ) (peanut_to_butter_ratio : ℝ) (butter_price : ℝ) : ℝ :=
  let plantation_area := side_length * side_length
  let total_peanuts := plantation_area * peanut_yield
  let total_butter := total_peanuts * peanut_to_butter_ratio
  let total_butter_kg := total_butter / 1000
  total_butter_kg * butter_price

/-- Proves that the revenue from Matt's peanut plantation is $31,250 -/
theorem matt_peanut_plantation_revenue :
  peanut_plantation_revenue 500 50 (5 / 20) 10 = 31250 := by
  unfold peanut_plantation_revenue
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matt_peanut_plantation_revenue_l1295_129583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_l1295_129542

/-- The function f(x) = 2 + x + 1 -/
def f (x : ℝ) : ℝ := 2 + x + 1

/-- The maximum value of |f(x)| on the interval (0, 1] -/
noncomputable def y_max (a : ℝ) : ℝ :=
  if -1 < a ∧ a < 1 then a + 2
  else if 3 ≤ a ∧ a < -1 then 1
  else a - 2

theorem max_value_f (a : ℝ) :
  ∀ x ∈ Set.Ioo 0 1, |f x| ≤ y_max a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_l1295_129542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_product_l1295_129560

theorem simplify_sqrt_product : Real.sqrt (5 * 3) * Real.sqrt (3^5 * 5^5) = 3375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_product_l1295_129560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_l1295_129569

/-- The operation performed at each step -/
def combine (a b : ℝ) : ℝ := a + b + a * b

/-- The set of initial numbers -/
def initial_set (n : ℕ) : Set ℝ := {x | ∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ x = 1 / k}

/-- The theorem stating the final result of the process -/
theorem final_number (n : ℕ) :
  ∃ (process : ℕ → Set ℝ),
    process 0 = initial_set n ∧
    (∀ k, process (k + 1) ⊆ process k) ∧
    (∀ k, ∃ a b, a ∈ process k ∧ b ∈ process k ∧ a ≠ b ∧
      process (k + 1) = (process k \ {a, b}) ∪ {combine a b}) ∧
    (∃ m, ∀ k > m, process k = process m ∧ ∃! x, x ∈ process m) ∧
    (∃ m, ∀ x, (∃! y, y ∈ process m) → x = n) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_l1295_129569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1295_129580

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle O (renamed to avoid conflict)
def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 2

-- Define the eccentricity of an ellipse
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Define a point on the ellipse
def point_on_ellipse (a b : ℝ) : Prop :=
  ellipse a b 2 1

-- Define the right focus of the ellipse
noncomputable def right_focus (a b : ℝ) : ℝ × ℝ :=
  (Real.sqrt (a^2 - b^2), 0)

-- Define a line tangent to the circle and passing through a point
def tangent_line (x y : ℝ) : Set (ℝ × ℝ) :=
  {(p, q) | ∃ (k m : ℝ), q = k * (p - x) + y ∧ m^2 = 2 * k^2 + 2}

-- Main theorem
theorem ellipse_properties
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : eccentricity a b = Real.sqrt 2 / 2)
  (h4 : point_on_ellipse a b)
  (P Q : ℝ × ℝ)
  (h5 : ellipse a b P.1 P.2)
  (h6 : ellipse a b Q.1 Q.2)
  (h7 : ∃ (l : Set (ℝ × ℝ)), l = tangent_line (right_focus a b).1 (right_focus a b).2 ∧ P ∈ l ∧ Q ∈ l) :
  a^2 = 6 ∧ b^2 = 3 ∧
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) * Real.sqrt 2 / 2 = 6 * Real.sqrt 3 / 5 ∧
  (P.1 * Q.1 + P.2 * Q.2 = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1295_129580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l1295_129572

/-- The time (in seconds) it takes for a train to pass a bridge -/
noncomputable def train_passing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

theorem train_bridge_passing_time :
  train_passing_time 310 140 45 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l1295_129572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_in_pascal_l1295_129566

/-- Pascal's triangle, represented as a function from row and column indices to natural numbers -/
def pascal_triangle : ℕ → ℕ → ℕ := sorry

/-- Assertion that Pascal's triangle contains all positive integers -/
axiom pascal_contains_all_positive : ∀ n : ℕ, n > 0 → ∃ i j : ℕ, pascal_triangle i j = n

/-- The smallest four-digit number -/
def smallest_four_digit : ℕ := 1000

/-- Theorem stating that 1000 is the smallest four-digit number in Pascal's triangle -/
theorem smallest_four_digit_in_pascal : 
  ∃ i j : ℕ, pascal_triangle i j = smallest_four_digit ∧ 
  ∀ k l : ℕ, pascal_triangle k l < smallest_four_digit → pascal_triangle k l < 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_in_pascal_l1295_129566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_glass_bowl_rate_l1295_129537

/-- The rate at which the person bought each glass bowl -/
noncomputable def R : ℝ := sorry

/-- The number of glass bowls bought -/
def bowls_bought : ℕ := 115

/-- The number of glass bowls sold -/
def bowls_sold : ℕ := 104

/-- The selling price of each bowl -/
def selling_price : ℝ := 20

/-- The percentage gain -/
def percentage_gain : ℝ := 0.4830917874396135

theorem glass_bowl_rate :
  ∃ (ε : ℝ), ε > 0 ∧ |R - 18| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_glass_bowl_rate_l1295_129537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_min_value_l1295_129573

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x else 1/x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  a * f x - |x - 2|

-- Part I
theorem part_one : ∀ b : ℝ, b ≥ -1 → ∀ x : ℝ, x > 0 → g 0 x ≤ |x - 1| + b := by
  sorry

-- Part II
theorem part_two : ∀ x : ℝ, x > 0 → g 1 x ≥ 0 := by
  sorry

theorem min_value : ∃ x : ℝ, x > 0 ∧ g 1 x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_min_value_l1295_129573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l1295_129536

theorem sin_beta_value (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2)
  (h3 : -π/2 < β) (h4 : β < 0)
  (h5 : Real.cos (α - β) = -5/13)
  (h6 : Real.sin α = 4/5) : 
  Real.sin β = -56/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l1295_129536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_f_l1295_129533

/-- The function f(x) = sin(2x + π/3) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

/-- Theorem: The axis of symmetry for the graph of f(x) = sin(2x + π/3) is x = -5π/12 -/
theorem axis_of_symmetry_f :
  ∃ (k : ℤ), (Real.pi / 12 + k * Real.pi / 2 : ℝ) = -5 * Real.pi / 12 ∧
  ∀ (x : ℝ), f (Real.pi / 12 + k * Real.pi / 2 + x) = f (Real.pi / 12 + k * Real.pi / 2 - x) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_f_l1295_129533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_second_number_l1295_129501

theorem multiple_of_second_number (x y : ℤ) (m : ℚ) : 
  x + y = 50 →
  y = Int.floor (m * x) - 43 →
  max x y = 31 →
  Int.floor m = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_second_number_l1295_129501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_composition_is_translation_l1295_129574

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rotation in the xy-plane -/
structure Rotation where
  angle : ℝ
  center : Point

/-- The composition of rotations -/
noncomputable def compose_rotations (rotations : List Rotation) (p : Point) : Point :=
  sorry

/-- Theorem statement -/
theorem rotation_composition_is_translation 
  (n : ℕ) 
  (h_n : n ≥ 2) 
  (θ : ℝ) 
  (h_θ : θ = 2 * π / n) 
  (rotations : List Rotation) 
  (h_rotations : rotations = (List.range n).map (λ k => 
    Rotation.mk θ (Point.mk (k : ℝ) 0)))
  (p : Point) : 
  compose_rotations rotations p = Point.mk (p.x + n) p.y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_composition_is_translation_l1295_129574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_image_eq_base3NoTwo_l1295_129593

-- Define the function f on natural numbers
def f : ℕ → ℕ
| 0 => 1
| n + 1 => if (n + 1) % 2 = 0 then 3 * f ((n + 1) / 2) else f n + 1

-- Define the set of numbers that can be written in base 3 without the digit 2
def base3NoTwo : Set ℕ :=
  {m : ℕ | ∃ (digits : List (Fin 2)), m = digits.reverse.foldl (fun acc d => acc * 3 + d.val) 0}

-- Statement of the theorem
theorem f_image_eq_base3NoTwo :
  {m : ℕ | ∃ n : ℕ, f n = m} = base3NoTwo :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_image_eq_base3NoTwo_l1295_129593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_neg_one_increasing_function_implies_a_nonpositive_l1295_129529

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

-- Theorem 1: If f is odd, then a = -1
theorem odd_function_implies_a_eq_neg_one (a : ℝ) :
  (∀ x, f a x = -f a (-x)) → a = -1 := by
  sorry

-- Theorem 2: If f is increasing on ℝ, then a ∈ (-∞, 0]
theorem increasing_function_implies_a_nonpositive (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_neg_one_increasing_function_implies_a_nonpositive_l1295_129529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_sum_sixty_consecutive_even_sum_l1295_129541

theorem consecutive_even_sum (n : Nat) : n ∈ ({32, 80, 104, 200} : Set Nat) →
  ¬∃ (m : Nat), Even m ∧ n = 4 * m + 12 :=
by sorry

theorem sixty_consecutive_even_sum :
  ∃ (m : Nat), Even m ∧ 60 = 4 * m + 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_sum_sixty_consecutive_even_sum_l1295_129541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_four_integers_l1295_129521

theorem product_of_four_integers (A B C D : ℕ) : 
  A + B + C + D = 72 →
  A + 3 = B - 3 ∧ A + 3 = C * 3 ∧ A + 3 = D / 2 →
  A * B * C * D = 68040 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_four_integers_l1295_129521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_player_wins_l1295_129505

/-- Represents a heap of candies -/
structure Heap where
  candies : ℕ

/-- Represents the game state with two heaps -/
structure GameState where
  heap1 : Heap
  heap2 : Heap

/-- Checks if a heap has a winning number of candies (5k ± 2) -/
def is_winning_heap (h : Heap) : Prop :=
  ∃ k : ℕ, h.candies = 5 * k + 2 ∨ h.candies = 5 * k - 2

/-- Represents a valid move in the game -/
inductive Move : GameState → GameState → Prop where
  | split_first (g : GameState) (a b : ℕ) :
      a + b = g.heap1.candies →
      Move g ⟨Heap.mk a, Heap.mk b⟩
  | split_second (g : GameState) (a b : ℕ) :
      a + b = g.heap2.candies →
      Move g ⟨Heap.mk a, Heap.mk b⟩

/-- Checks if a game state is winning for the current player -/
def is_winning_state (g : GameState) : Prop :=
  g.heap1.candies = 1 ∨ g.heap2.candies = 1

/-- Represents the initial game state -/
def initial_state : GameState :=
  ⟨Heap.mk 33, Heap.mk 35⟩

/-- Theorem stating that the starting player has a winning strategy -/
theorem starting_player_wins :
  ∃ (strategy : GameState → GameState),
    (Move initial_state (strategy initial_state)) ∧
    ∀ (opponent_move : GameState),
      Move (strategy initial_state) opponent_move →
      is_winning_state opponent_move ∨
      ∃ (next_move : GameState),
        Move opponent_move next_move ∧ is_winning_state next_move :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_player_wins_l1295_129505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1295_129552

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 / (x + 2)

-- State the theorem about the domain of f
theorem domain_of_f : 
  {x : ℝ | f x ≠ 0} = {x : ℝ | x ≠ -2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1295_129552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_focal_lengths_l1295_129546

-- Define the ellipses
def C1 (x y : ℝ) : Prop := x^2 / 12 + y^2 / 4 = 1
def C2 (x y : ℝ) : Prop := x^2 / 16 + y^2 / 8 = 1

-- Define the focal length of an ellipse
noncomputable def focal_length (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

-- Theorem: The focal lengths of C1 and C2 are equal
theorem equal_focal_lengths :
  focal_length (Real.sqrt 12) 2 = focal_length 4 (Real.sqrt 8) := by
  -- Expand the definition of focal_length
  unfold focal_length
  -- Simplify the expressions
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_focal_lengths_l1295_129546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1295_129571

-- Define the sets M and N
def M : Set ℝ := { x | x < 2 }
def N : Set ℝ := { x | Real.rpow 3 x > 1/3 }

-- State the theorem
theorem intersection_M_N :
  M ∩ N = { x : ℝ | -1 < x ∧ x < 2 } := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1295_129571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_midsegment_length_trapezoid_midsegment_length_explicit_l1295_129590

/-- 
In a trapezoid with bases a and c, a segment parallel to the bases that divides 
the area in half has length √((a² + c²)/2).
-/
theorem trapezoid_midsegment_length 
  (a c : ℝ) (h : 0 < a ∧ 0 < c) : 
  ∃ (x : ℝ), x > 0 ∧ x^2 = (a^2 + c^2) / 2 := by
  -- We need to prove the existence of a positive real number x 
  -- such that x^2 = (a^2 + c^2) / 2
  sorry

/-- The length of the midsegment is √((a² + c²)/2) -/
theorem trapezoid_midsegment_length_explicit 
  (a c : ℝ) (h : 0 < a ∧ 0 < c) : 
  Real.sqrt ((a^2 + c^2) / 2) > 0 ∧ 
  (Real.sqrt ((a^2 + c^2) / 2))^2 = (a^2 + c^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_midsegment_length_trapezoid_midsegment_length_explicit_l1295_129590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dan_picked_nine_l1295_129563

/-- The number of plums picked by each person and in total -/
structure PlumPicking where
  melanie : ℕ
  sally : ℕ
  total : ℕ
  dan : ℕ

/-- The conditions of the plum picking scenario -/
def plum_conditions (p : PlumPicking) : Prop :=
  p.melanie = 4 ∧ p.sally = 3 ∧ p.total = 16 ∧ p.total = p.melanie + p.sally + p.dan

/-- Theorem stating that Dan picked 9 plums -/
theorem dan_picked_nine (p : PlumPicking) (h : plum_conditions p) : p.dan = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dan_picked_nine_l1295_129563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l1295_129577

-- Define the parameters
noncomputable def train1_length : ℝ := 200
noncomputable def train2_length : ℝ := 150
noncomputable def bridge_length : ℝ := 135
noncomputable def train1_speed_kmh : ℝ := 95
noncomputable def train2_speed_kmh : ℝ := 105

-- Convert speeds from km/h to m/s
noncomputable def train1_speed_ms : ℝ := train1_speed_kmh * 1000 / 3600
noncomputable def train2_speed_ms : ℝ := train2_speed_kmh * 1000 / 3600

-- Calculate total distance and relative speed
noncomputable def total_distance : ℝ := train1_length + bridge_length + train2_length
noncomputable def relative_speed : ℝ := train1_speed_ms + train2_speed_ms

-- Define the theorem
theorem train_crossing_time_approx (ε : ℝ) (h : ε > 0) :
  ∃ (t : ℝ), t > 0 ∧ |t - total_distance / relative_speed| < ε ∧ |t - 8.73| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l1295_129577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arbitrarily_large_circle_exists_countable_lines_intersect_all_circles_l1295_129595

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for lines in a plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a type for circles in a plane
structure Circle where
  center : Point
  radius : ℝ

-- Define a function to check if a circle intersects a line
def circleIntersectsLine (c : Circle) (l : Line) : Prop :=
  sorry

-- Theorem 1: For any finite set of lines, there exists an arbitrarily large circle that doesn't intersect any line
theorem arbitrarily_large_circle_exists (lines : Finset Line) :
  ∀ r : ℝ, ∃ c : Circle, c.radius > r ∧ ∀ l ∈ lines, ¬circleIntersectsLine c l := by
  sorry

-- Theorem 2: There exists a countable set of lines such that any circle with positive radius intersects at least one line
theorem countable_lines_intersect_all_circles :
  ∃ lines : Set Line, (Countable lines) ∧
  (∀ c : Circle, c.radius > 0 → ∃ l ∈ lines, circleIntersectsLine c l) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arbitrarily_large_circle_exists_countable_lines_intersect_all_circles_l1295_129595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_sum_reciprocal_l1295_129504

theorem exponent_sum_reciprocal (x y : ℝ) (h1 : (4 : ℝ)^x = 6) (h2 : (9 : ℝ)^y = 6) : 
  1/x + 1/y = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_sum_reciprocal_l1295_129504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_and_customer_benefit_l1295_129570

/-- Represents the profit function for a product sale --/
structure ProfitFunction where
  cost : ℝ  -- Cost price
  initial_price : ℝ  -- Initial selling price
  initial_units : ℝ  -- Initial units sold
  price_increase_effect : ℝ  -- Units decreased per 1 yuan increase
  price_decrease_effect : ℝ  -- Units increased per 1 yuan decrease

/-- Calculates the profit based on the selling price --/
noncomputable def profit (pf : ProfitFunction) (x : ℝ) : ℝ :=
  if x > pf.initial_price then
    -pf.price_increase_effect * (x - pf.initial_price)^2 + 
    (pf.initial_units * (pf.initial_price - pf.cost))
  else if x > pf.cost then
    pf.price_decrease_effect * (pf.initial_price - x)^2 + 
    (pf.initial_units * (pf.initial_price - pf.cost))
  else 0

theorem profit_maximization_and_customer_benefit 
  (pf : ProfitFunction) 
  (h_cost : pf.cost = 100)
  (h_initial_price : pf.initial_price = 120)
  (h_initial_units : pf.initial_units = 300)
  (h_price_increase : pf.price_increase_effect = 10)
  (h_price_decrease : pf.price_decrease_effect = 30) :
  ∃ (max_price customer_price : ℝ),
    (∀ x, profit pf x ≤ profit pf max_price) ∧
    (max_price = 115) ∧
    (profit pf customer_price = profit pf pf.initial_price) ∧
    (customer_price = pf.initial_price) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_and_customer_benefit_l1295_129570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_line_intersection_distance_curve_C_points_distance_from_line_l1295_129553

/-- Line l in polar coordinates -/
def line_l (ρ θ : ℝ) : Prop :=
  ρ * (3 * Real.cos θ - 4 * Real.sin θ) = 2

/-- Curve C in polar coordinates -/
def curve_C (ρ m : ℝ) : Prop :=
  ρ = m ∧ m > 0

/-- Distance from a point (x, y) to line ax + by + c = 0 -/
noncomputable def distance_to_line (x y a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

theorem polar_line_intersection_distance :
  ∃ (ρ : ℝ), line_l ρ 0 ∧ ρ = 2/3 := by sorry

theorem curve_C_points_distance_from_line (m : ℝ) :
  (∃! (p q : ℝ × ℝ), 
    curve_C (Real.sqrt (p.1^2 + p.2^2)) m ∧
    curve_C (Real.sqrt (q.1^2 + q.2^2)) m ∧
    distance_to_line p.1 p.2 3 (-4) (-2) = 1/5 ∧
    distance_to_line q.1 q.2 3 (-4) (-2) = 1/5) ↔
  1/5 < m ∧ m < 3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_line_intersection_distance_curve_C_points_distance_from_line_l1295_129553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_problem_l1295_129598

/-- A function representing the inverse square relationship between x and y -/
noncomputable def inverse_square_relation (k : ℝ) (y : ℝ) : ℝ := k / (y ^ 2)

/-- Theorem stating that given the inverse square relationship between x and y,
    if y = 2 when x = 1, then x = 1/9 when y = 6 -/
theorem inverse_square_problem (k : ℝ) :
  inverse_square_relation k 2 = 1 →
  inverse_square_relation k 6 = 1/9 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_problem_l1295_129598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_transformation_l1295_129543

/-- Represents a point in 3D space with rectangular coordinates -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents spherical coordinates -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Conversion from spherical to rectangular coordinates -/
noncomputable def sphericalToRectangular (s : SphericalCoord) : Point3D :=
  { x := s.ρ * Real.sin s.φ * Real.cos s.θ
  , y := s.ρ * Real.sin s.φ * Real.sin s.θ
  , z := s.ρ * Real.cos s.φ }

/-- The main theorem to prove -/
theorem coordinate_transformation (p : Point3D) (s : SphericalCoord) :
  p = { x := 3, y := -3, z := 2 * Real.sqrt 2 } →
  p = sphericalToRectangular s →
  sphericalToRectangular { ρ := s.ρ, θ := s.θ + π / 2, φ := -s.φ } = { x := -3, y := -3, z := 2 * Real.sqrt 2 } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_transformation_l1295_129543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_for_factorial_product_l1295_129591

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem smallest_difference_for_factorial_product (x y z : ℕ+) : 
  (x : ℕ) * y * z = factorial 7 → x < y → y < z → 
  (∀ a b c : ℕ+, (a : ℕ) * b * c = factorial 7 → a < b → b < c → z - x ≤ c - a) →
  z - x = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_for_factorial_product_l1295_129591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_circumscribed_circle_l1295_129532

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to find the x-intercept of a line -/
noncomputable def xIntercept (l : Line) : ℝ := -l.c / l.a

/-- Function to find the y-intercept of a line -/
noncomputable def yIntercept (l : Line) : ℝ := -l.c / l.b

/-- Function to check if four points form a tangential quadrilateral -/
def isTangentialQuadrilateral (p1 p2 p3 p4 : Point) : Prop :=
  abs (p1.x - p2.x) + abs (p3.x - p4.x) = abs (p1.y - p4.y) + abs (p2.y - p3.y)

theorem quadrilateral_circumscribed_circle (k : ℝ) : 
  let line1 : Line := { a := 1, b := 3, c := -7 }
  let line2 : Line := { a := k, b := -1, c := -2 }
  let p1 : Point := { x := xIntercept line1, y := 0 }
  let p2 : Point := { x := 0, y := yIntercept line1 }
  let p3 : Point := { x := xIntercept line2, y := 0 }
  let p4 : Point := { x := 0, y := yIntercept line2 }
  isTangentialQuadrilateral p1 p2 p3 p4 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_circumscribed_circle_l1295_129532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_cosine_over_one_minus_cosine_cubed_l1295_129509

theorem integral_cosine_over_one_minus_cosine_cubed : 
  ∫ x in Set.Icc (2 * Real.arctan (1/2)) (π/2), (Real.cos x) / ((1 - Real.cos x)^3) = 21/20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_cosine_over_one_minus_cosine_cubed_l1295_129509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_darwin_expenses_l1295_129539

theorem darwin_expenses (initial_amount : ℝ) (gas_fraction food_fraction clothing_fraction : ℝ) 
  (h_initial : initial_amount = 9000)
  (h_gas : gas_fraction = 2/5)
  (h_food : food_fraction = 1/3)
  (h_clothing : clothing_fraction = 1/4) : 
  let remaining_after_gas := initial_amount - gas_fraction * initial_amount
  let remaining_after_food := remaining_after_gas - food_fraction * remaining_after_gas
  let final_amount := remaining_after_food - clothing_fraction * remaining_after_food
  final_amount = 2700 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_darwin_expenses_l1295_129539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_nine_is_45_l1295_129538

/-- An arithmetic sequence with a given property -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  property : a 3 + a 5 + a 7 = 15

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The sum of the first 9 terms is 45 -/
theorem sum_of_nine_is_45 (seq : ArithmeticSequence) : S seq 9 = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_nine_is_45_l1295_129538
