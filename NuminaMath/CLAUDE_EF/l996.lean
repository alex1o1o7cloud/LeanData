import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_stack_height_for_12cm_diameter_l996_99695

/-- The height of a pile of 4 identical cylindrical pipes stacked in a triangular pyramid -/
noncomputable def pipeStackHeight (diameter : ℝ) : ℝ :=
  diameter / 2 + (diameter / 2) * Real.sqrt 3

theorem pipe_stack_height_for_12cm_diameter :
  pipeStackHeight 12 = 6 + 6 * Real.sqrt 3 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_stack_height_for_12cm_diameter_l996_99695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l996_99602

-- Define the function f(x) = x + sin(x)
noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

-- Theorem stating the properties of f(x)
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is an odd function
  (∀ x y, x < y → f x < f y) ∧  -- f is monotonically increasing
  (∀ y, ∃ x, f x = y) ∧  -- the range of f is ℝ
  (¬ ∃ T, T > 0 ∧ ∀ x, f (x + T) = f x)  -- f is not periodic
  := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l996_99602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_l996_99697

/-- The scaling transformation φ that maps (x, y) to (x', y') -/
noncomputable def φ (x y : ℝ) : ℝ × ℝ :=
  (3 * x, y / 2)

/-- The original hyperbola C -/
def C (x y : ℝ) : Prop :=
  x^2 - y^2 / 64 = 1

/-- The transformed curve C' -/
def C' (x' y' : ℝ) : Prop :=
  x'^2 / 9 - y'^2 / 16 = 1

/-- Theorem stating that the scaling transformation φ maps C to C' -/
theorem scaling_transformation (x y : ℝ) :
  C x y ↔ C' (φ x y).1 (φ x y).2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_l996_99697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_systems_imply_equivalent_combinations_l996_99669

/-- A linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A system of two linear equations -/
structure LinearSystem where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- The solution of a linear system -/
structure Solution where
  x : ℝ
  y : ℝ

/-- Two linear systems are equivalent if they have the same solution -/
def equivalent (sys1 sys2 : LinearSystem) : Prop :=
  ∀ (sol : Solution), 
    (sol.x * sys1.eq1.a + sol.y * sys1.eq1.b = sys1.eq1.c ∧
     sol.x * sys1.eq2.a + sol.y * sys1.eq2.b = sys1.eq2.c) ↔
    (sol.x * sys2.eq1.a + sol.y * sys2.eq1.b = sys2.eq1.c ∧
     sol.x * sys2.eq2.a + sol.y * sys2.eq2.b = sys2.eq2.c)

theorem equivalent_systems_imply_equivalent_combinations 
  (sys12 sys34 : LinearSystem) 
  (h : equivalent sys12 sys34) : 
  let sys13 := LinearSystem.mk sys12.eq1 sys34.eq1
  let sys14 := LinearSystem.mk sys12.eq1 sys34.eq2
  let sys23 := LinearSystem.mk sys12.eq2 sys34.eq1
  let sys24 := LinearSystem.mk sys12.eq2 sys34.eq2
  equivalent sys12 sys13 ∧ 
  equivalent sys12 sys14 ∧ 
  equivalent sys12 sys23 ∧ 
  equivalent sys12 sys24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_systems_imply_equivalent_combinations_l996_99669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_natasha_avg_speed_up_l996_99678

/-- Represents Natasha's hill climbing journey -/
structure HillClimb where
  upTime : ℝ  -- Time taken to climb up (in hours)
  downTime : ℝ  -- Time taken to descend (in hours)
  avgSpeed : ℝ  -- Average speed for the whole journey (in km/h)

/-- Calculates the average speed while climbing up the hill -/
noncomputable def avgSpeedUp (h : HillClimb) : ℝ :=
  let totalTime := h.upTime + h.downTime
  let totalDistance := h.avgSpeed * totalTime
  totalDistance / (2 * h.upTime)

/-- Theorem stating that Natasha's average speed while climbing up is 2.5 km/h -/
theorem natasha_avg_speed_up :
  let h : HillClimb := { upTime := 3, downTime := 2, avgSpeed := 3 }
  avgSpeedUp h = 2.5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_natasha_avg_speed_up_l996_99678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_distance_ratio_l996_99647

/-- Represents the triathlon event with given parameters -/
structure Triathlon where
  total_time : ℚ
  walk_time : ℚ
  walk_speed : ℚ
  jog_time : ℚ
  jog_speed : ℚ
  cycle_speed : ℚ

/-- Calculates the ratio of (jogging + cycling distance) to walking distance -/
noncomputable def distance_ratio (t : Triathlon) : ℚ :=
  let walk_dist := t.walk_time / 60 * t.walk_speed
  let jog_dist := t.jog_time / 60 * t.jog_speed
  let cycle_time := t.total_time - t.walk_time - t.jog_time
  let cycle_dist := cycle_time / 60 * t.cycle_speed
  (jog_dist + cycle_dist) / walk_dist

/-- The main theorem stating the ratio is approximately 8 -/
theorem triathlon_distance_ratio :
  let t : Triathlon := {
    total_time := 65
    walk_time := 15
    walk_speed := 5
    jog_time := 25
    jog_speed := 8
    cycle_speed := 16
  }
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1/10) ∧ |distance_ratio t - 8| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_distance_ratio_l996_99647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_dot_product_l996_99641

-- Define the curve
def curve (x : ℝ) : ℝ := x^2 - 6*x + 1

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 9

-- Define point A
def A : ℝ × ℝ := (3, 0)

-- Define a point B on the circle
noncomputable def B (θ : ℝ) : ℝ × ℝ := (3 + 3 * Real.cos θ, 1 + 3 * Real.sin θ)

-- Define the dot product of OA and OB
noncomputable def dot_product (θ : ℝ) : ℝ := A.1 * (B θ).1 + A.2 * (B θ).2

-- Theorem statement
theorem circle_and_dot_product :
  -- The circle passes through the intersection points of the curve with the axes
  (∃ x, x^2 - 6*x + 1 = 0 ∧ circle_eq x 0) ∧
  (circle_eq 0 (curve 0)) ∧
  -- The maximum value of OA · OB is 18
  (∀ θ, dot_product θ ≤ 18) ∧
  (∃ θ, dot_product θ = 18) ∧
  -- The length of the chord when OA · OB is maximum
  (∃ θ, dot_product θ = 18 ∧
    let O : ℝ × ℝ := (0, 0)
    let C : ℝ × ℝ := (3, 1)
    let d := abs (((B θ).2 - O.2) * (C.1 - O.1) - ((B θ).1 - O.1) * (C.2 - O.2)) /
              Real.sqrt ((B θ).2 - O.2)^2 + ((B θ).1 - O.1)^2
    2 * Real.sqrt (3^2 - d^2) = 36 * Real.sqrt 37 / 37) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_dot_product_l996_99641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l996_99690

/-- Simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Problem statement -/
theorem interest_rate_calculation (principal time interest : ℝ) 
  (h_principal : principal = 800)
  (h_time : time = 4)
  (h_interest : interest = 176) :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l996_99690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_sports_equipment_l996_99670

theorem min_cost_sports_equipment (basketball_price badminton_price : ℕ) 
  (h1 : basketball_price = badminton_price + 50)
  (h2 : 2 * basketball_price + 3 * badminton_price = 250)
  (total_items : ℕ) (h3 : total_items = 200)
  (min_basketballs : ℕ) (h4 : min_basketballs = 80) :
  let cost := fun a ↦ a * basketball_price + (total_items - a) * badminton_price
  ∃ (a : ℕ), a ≥ min_basketballs ∧ 
    (∀ (b : ℕ), b ≥ min_basketballs → cost a ≤ cost b) ∧
    cost a = 10000 ∧ a = 80 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_sports_equipment_l996_99670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hilary_meal_cost_l996_99635

/-- Calculates the total cost of a meal including tip and tax --/
def meal_cost (samosa_price : ℚ) (samosa_count : ℕ)
              (pakora_price : ℚ) (pakora_count : ℕ)
              (lassi_price : ℚ) (lassi_count : ℕ)
              (biryani_price : ℚ) (biryani_count : ℕ)
              (naan_price : ℚ) (naan_count : ℕ)
              (tip_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let subtotal := samosa_price * samosa_count +
                  pakora_price * pakora_count +
                  lassi_price * lassi_count +
                  biryani_price * biryani_count +
                  naan_price * naan_count
  let tax := (tax_rate * subtotal).floor / 100
  let total_before_tip := subtotal + tax
  let tip := (tip_rate * total_before_tip).floor / 100
  total_before_tip + tip

/-- Theorem stating the total cost of Hilary's meal --/
theorem hilary_meal_cost :
  meal_cost 2 3 3 4 2 1 (11/2) 2 (3/2) 1 (18/100) (7/100) = 4104/100 := by
  sorry

#eval meal_cost 2 3 3 4 2 1 (11/2) 2 (3/2) 1 (18/100) (7/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hilary_meal_cost_l996_99635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l996_99621

/-- Parabola type -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type -/
structure Line where
  m : ℝ
  b : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_properties 
  (C : Parabola)
  (O A B P Q : Point)
  (h1 : C.p > 0)
  (h2 : C.eq = fun x y => x^2 = 2 * C.p * y)
  (h3 : O.x = 0 ∧ O.y = 0)
  (h4 : A.x = 1 ∧ A.y = 1)
  (h5 : C.eq A.x A.y)
  (h6 : B.x = 0 ∧ B.y = -1)
  (h7 : ∃ (l : Line), l.m * P.x + l.b = P.y ∧ l.m * Q.x + l.b = Q.y ∧ l.m * B.x + l.b = B.y)
  (h8 : C.eq P.x P.y ∧ C.eq Q.x Q.y) :
  (∃ (l : Line), l.m * A.x + l.b = A.y ∧ l.m * B.x + l.b = B.y ∧ 
    ∀ (x y : ℝ), C.eq x y → l.m * x + l.b ≤ y) ∧ 
  (distance O P * distance O Q > distance O A ^ 2) ∧
  (distance B P * distance B Q > distance B A ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l996_99621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l996_99606

theorem inequality_theorem (n : ℕ) (a : Fin n → ℕ) 
  (h_pos : ∀ i, a i > 0)
  (h_increasing : ∀ i j, i < j → a i < a j)
  (h_divides : ∀ i j, i ≠ j → (a j - a i) ∣ a i) :
  ∀ (i j : Fin n), i < j → (i.val : ℕ) * a j ≤ (j.val : ℕ) * a i :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l996_99606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_range_l996_99661

-- Define the line l
def line_l (a t : ℝ) : ℝ × ℝ := (a - 2*t, -4*t)

-- Define the circle C
noncomputable def circle_C (θ : ℝ) : ℝ × ℝ := (4*Real.cos θ, 4*Real.sin θ)

-- Theorem statement
theorem line_circle_intersection_range (a : ℝ) :
  (∃ t θ : ℝ, line_l a t = circle_C θ) ↔ -2 * Real.sqrt 5 ≤ a ∧ a ≤ 2 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_range_l996_99661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_decomposition_l996_99682

theorem partial_fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 → (B * x - 17) / (x^2 - 9*x + 20) = A / (x - 4) + 5 / (x - 5)) →
  A = -3/20 ∧ B = 22/5 →
  A + B = 17/4 := by
  intros h1 h2
  sorry

#check partial_fraction_decomposition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_decomposition_l996_99682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_sum_l996_99638

theorem least_integer_sum (x y z : ℕ+) (h : 2 * x.val = 5 * y.val ∧ 5 * y.val = 6 * z.val) :
  ∃ a : ℕ, a + y.val + z.val = 26 ∧ ∀ b : ℕ, b + y.val + z.val = 26 → a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_sum_l996_99638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_y_is_negative_85_l996_99659

/-- The y-coordinate of the intersection point of two lines -/
noncomputable def intersection_y_coordinate (m1 b1 a2 b2 c2 : ℝ) : ℝ :=
  let x := (c2 + 2*b1) / (2*m1 + a2)
  m1 * x + b1

/-- The two lines intersect at a point with y-coordinate -85 -/
theorem intersection_y_is_negative_85 :
  intersection_y_coordinate 3 5 5 (-2) 20 = -85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_y_is_negative_85_l996_99659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_and_projection_l996_99674

noncomputable def a : ℝ × ℝ := (-1, 1)
noncomputable def b : ℝ × ℝ := (4, 3)
noncomputable def c : ℝ × ℝ := (5, -2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem cosine_and_projection :
  (dot_product a b / (magnitude a * magnitude b) = -Real.sqrt 2 / 10) ∧
  (dot_product a c / magnitude a = -7/2 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_and_projection_l996_99674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_fraction_l996_99663

/-- Given a square ABCD with side length s, point P located at (s/3, 0),
    and point Q located at (s, 2s/3), the area of the region inside the square
    but outside the triangle APQ is 8/9 of the total area of the square. -/
theorem shaded_area_fraction (s : ℝ) (h : s > 0) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (s, 0)
  let C : ℝ × ℝ := (s, s)
  let D : ℝ × ℝ := (0, s)
  let P : ℝ × ℝ := (s / 3, 0)
  let Q : ℝ × ℝ := (s, 2 * s / 3)
  let square_area : ℝ := s^2
  let triangle_area : ℝ := (s * (2 * s / 3)) / 2
  let shaded_area : ℝ := square_area - triangle_area
  shaded_area / square_area = 8 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_fraction_l996_99663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_outputs_minimum_l996_99652

noncomputable def algorithm (a b c d : ℝ) : ℝ :=
  let m₁ := min a b
  let m₂ := min m₁ c
  min m₂ d

theorem algorithm_outputs_minimum (a b c d : ℝ) :
  algorithm a b c d = min (min (min a b) c) d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_outputs_minimum_l996_99652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_condition_l996_99631

/-- The function f(x) = -1/2x^2 + 4x - 2alnx has two distinct extreme points iff 0 < a < 2 -/
theorem extreme_points_condition (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧
    (∀ x : ℝ, x > 0 → 
      (deriv (λ x => -1/2 * x^2 + 4*x - 2*a * Real.log x)) x = 0 ↔ (x = x₁ ∨ x = x₂)))
  ↔ 0 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_condition_l996_99631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_existence_l996_99613

noncomputable def f (x : ℝ) := Real.log x / Real.log 2 - 1 / x

theorem zero_point_existence :
  Continuous f ∧ 
  StrictMono f ∧
  (∀ x, x > 0 → f x ∈ Set.univ) ∧
  f 1 < 0 ∧
  f 2 > 0 →
  ∃ c ∈ Set.Ioo 1 2, f c = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_existence_l996_99613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_arrives_later_l996_99620

/-- A worker's journey to the office -/
structure Journey where
  usual_speed : ℚ
  distance : ℚ
  usual_time : ℚ
  slower_speed : ℚ

/-- Properties of the worker's journey -/
def journey_properties (j : Journey) : Prop :=
  j.usual_time = 24 ∧
  j.distance = j.usual_speed * j.usual_time ∧
  j.slower_speed = 3/4 * j.usual_speed

/-- The time difference between the usual journey and the slower journey -/
noncomputable def time_difference (j : Journey) : ℚ :=
  j.distance / j.slower_speed - j.usual_time

/-- Theorem stating that the worker arrives 8 minutes later when walking at 3/4 of her normal speed -/
theorem worker_arrives_later (j : Journey) (h : journey_properties j) : 
  time_difference j = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_arrives_later_l996_99620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_factors_count_l996_99672

def n : ℕ := 2^3 * 3^2 * 5 * 7^3

theorem even_factors_count : 
  (Finset.filter (λ x : ℕ => x ∣ n ∧ Even x) (Finset.range (n + 1))).card = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_factors_count_l996_99672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rate_of_change_l996_99628

-- Define the temperature function
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + 8

-- Define the derivative of the temperature function
noncomputable def f' (x : ℝ) : ℝ := x^2 - 2*x

-- Theorem statement
theorem min_rate_of_change :
  ∀ x ∈ Set.Icc 0 5, f' x ≥ -1 ∧ ∃ y ∈ Set.Icc 0 5, f' y = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rate_of_change_l996_99628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_to_hexagon_l996_99694

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle --/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Represents a hexagon --/
structure Hexagon where
  vertices : Finset Point

/-- Predicate to check if a triangle is equilateral --/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  sorry

/-- Represents the folding operation --/
def fold (t : Triangle) : Triangle :=
  sorry

/-- Represents the cutting operation --/
def cut (t : Triangle) : Triangle :=
  sorry

/-- Represents the unfolding operation --/
def unfold (t : Triangle) : Hexagon :=
  sorry

/-- Main theorem: Folding an equilateral triangle three times, cutting a corner, and unfolding results in a hexagon --/
theorem equilateral_triangle_to_hexagon (t : Triangle) (h : t.isEquilateral) :
  ∃ (hex : Hexagon), unfold (cut (fold (fold (fold t)))) = hex :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_to_hexagon_l996_99694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_circle_distance_l996_99618

/-- The number of boys on the circle -/
def num_boys : ℕ := 8

/-- The radius of the circle in feet -/
noncomputable def radius : ℝ := 50

/-- The angle between adjacent boys in radians -/
noncomputable def angle_between : ℝ := 2 * Real.pi / num_boys

/-- The number of non-adjacent boys for each boy -/
def non_adjacent_boys : ℕ := num_boys - 3

/-- The total distance traveled by all boys -/
noncomputable def total_distance : ℝ := num_boys * non_adjacent_boys * (2 * radius + 4 * radius * Real.sqrt 2)

theorem boys_circle_distance :
  total_distance = 800 + 1600 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_circle_distance_l996_99618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l996_99683

-- Define the function to be minimized
noncomputable def f (a b : ℝ) : ℝ := 1/a + 2/b + 2/(a*b)

-- State the theorem
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.log ((2-b)/a) = 2*a + 2*b - 4) : 
  f a b ≥ (5 + 2*Real.sqrt 6) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l996_99683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_ratio_l996_99610

/-- The ratio of the volume of a cube with edge length 4 inches to the volume of a cube with edge length 2 feet -/
theorem cube_volume_ratio : 
  (4^3 : ℚ) / (24^3) = 1 / 216 := by
  -- Convert integers to rationals
  have h1 : (4^3 : ℚ) = 64 := by norm_num
  have h2 : (24^3 : ℚ) = 13824 := by norm_num
  
  -- Rewrite the left side of the equation
  rw [h1, h2]
  
  -- Simplify the fraction
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_ratio_l996_99610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_intersection_sum_l996_99668

-- Define the points
def E : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (2, 4)
def G : ℝ × ℝ := (6, 2)
def H : ℝ × ℝ := (7, 0)

-- Define the quadrilateral
def EFGH : Set (ℝ × ℝ) := {E, F, G, H}

-- Define a function to represent a line through E
def line_through_E (m : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y = m * x}

-- Define the condition that the line divides the quadrilateral into two equal areas
def divides_equally (l : Set (ℝ × ℝ)) (q : Set (ℝ × ℝ)) : Prop :=
  ∃ (A₁ A₂ : ℝ), A₁ = A₂ ∧ A₁ + A₂ = 19 -- We use 19 as the total area

-- Define the intersection point
noncomputable def intersection_point (u v w z : ℤ) : ℝ × ℝ := (u / v, w / z)

-- State the theorem
theorem equal_area_intersection_sum :
  ∀ (u v w z : ℤ),
  (∃ (m : ℝ), divides_equally (line_through_E m) EFGH) →
  (∃ (x y : ℝ), (x, y) ∈ line_through_E m ∧ (x, y) ∈ Set.Icc G H ∧ (x, y) = intersection_point u v w z) →
  (Int.gcd u v = 1 ∧ Int.gcd w z = 1) →
  u + v + w + z = 68 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_intersection_sum_l996_99668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_is_ln2_l996_99675

noncomputable def f (x : ℝ) : ℝ := Real.log (x / 2) + 1 / 2

noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 2)

theorem min_difference_is_ln2 :
  ∃ (min : ℝ), min = Real.log 2 ∧
  (∀ (a : ℝ), ∃ (b : ℝ), b > 0 ∧ g a = f b ∧ b - a ≥ min) ∧
  (∃ (a : ℝ) (b : ℝ), b > 0 ∧ g a = f b ∧ b - a = min) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_is_ln2_l996_99675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_plum_alignment_l996_99632

/-- Represents a position on the grid -/
structure Position where
  x : Nat
  y : Nat

/-- Represents the state of the grid -/
structure GridState where
  n : Nat
  strawberries : List Position
  plums : List Position

/-- Checks if a move is valid -/
def isValidMove (state : GridState) (pos1 pos2 : Position) : Prop := sorry

/-- Applies a move to the grid state -/
def applyMove (state : GridState) (pos1 pos2 : Position) : GridState := sorry

/-- Checks if the goal state is reached -/
def isGoalState (state : GridState) : Prop := sorry

/-- Helper function to count plums in a rectangle -/
def count_plums (state : GridState) (top_left bottom_right : Position) : Nat := sorry

/-- Helper function to count strawberries in a rectangle -/
def count_strawberries (state : GridState) (top_left bottom_right : Position) : Nat := sorry

/-- Main theorem -/
theorem strawberry_plum_alignment
  (n : Nat)
  (initial_state : GridState)
  (h1 : initial_state.n = n)
  (h2 : initial_state.strawberries.length = n)
  (h3 : initial_state.plums.length = 1)
  (h4 : ∀ i j, i < n → j < n → (∃ s ∈ initial_state.strawberries, s.x = i) ∧ (∃ s ∈ initial_state.strawberries, s.y = j))
  (h5 : ∀ x y, x < n → y < n → (count_plums initial_state ⟨0, 0⟩ ⟨x, y⟩ >
                                count_strawberries initial_state ⟨0, 0⟩ ⟨x, y⟩)) :
  ∃ final_state : GridState,
    ∃ moves : List (Position × Position),
      final_state = moves.foldl (λ s m ↦ applyMove s m.1 m.2) initial_state ∧
      (∀ m ∈ moves, isValidMove initial_state m.1 m.2) ∧
      isGoalState final_state := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_plum_alignment_l996_99632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_produce_l996_99615

/-- Represents the total produce of mangoes in kg -/
def M : ℕ := sorry

/-- The price per kg of fruit in dollars -/
def price_per_kg : ℕ := 50

/-- The total revenue from selling all fruits in dollars -/
def total_revenue : ℕ := 90000

/-- The total produce of apples in kg -/
def apples : ℕ := 2 * M

/-- The total produce of oranges in kg -/
def oranges : ℕ := M + 200

/-- The theorem stating that the total produce of mangoes is 400 kg -/
theorem mango_produce : M = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_produce_l996_99615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l996_99614

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cuboid -/
structure Cuboid where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Predicate to check if a point is on the surface of the top face of the cuboid -/
def surface (A₁ B₁ C₁ D₁ : Point3D) (P : Point3D) : Prop :=
  sorry -- Definition of the surface would go here

/-- Theorem: Minimum sum of distances in a cuboid -/
theorem min_sum_distances (cuboid : Cuboid) 
  (h_AB : distance cuboid.A cuboid.B = 3)
  (h_AD : distance cuboid.A cuboid.D = 4)
  (h_AA₁ : distance cuboid.A cuboid.A₁ = 5) :
  ∃ (min_val : ℝ), 
    (∀ (P : Point3D), (surface cuboid.A₁ cuboid.B₁ cuboid.C₁ cuboid.D₁ P) → 
      (distance P cuboid.A + distance P cuboid.C ≥ min_val)) ∧
    min_val = 5 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l996_99614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l996_99679

/-- Triangle ABC with given properties -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  side_a : a = 1
  side_b : b = 2
  angle_C : Real.cos C = 1/4

/-- Properties of the triangle to be proven -/
theorem triangle_properties (t : Triangle) :
  -- Perimeter
  t.a + t.b + t.c = 5 ∧
  -- Area
  (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 15/4 ∧
  -- Cosine of sum of angles
  Real.cos (t.A + t.C) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l996_99679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_lines_l996_99640

/-- The distance between two parallel lines -/
theorem distance_between_parallel_lines (a : ℝ) :
  let l₁ : ℝ × ℝ → Prop := fun p => 3 * p.1 + 4 * p.2 - 4 = 0
  let l₂ : ℝ × ℝ → Prop := fun p => a * p.1 + 8 * p.2 + 2 = 0
  (∀ p q, l₁ p ∧ l₂ q → (p.1 - q.1) * 4 = (p.2 - q.2) * 3) →
  (∃ d, ∀ p q, l₁ p ∧ l₂ q → d = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) →
  (∃ d, ∀ p q, l₁ p ∧ l₂ q → d = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ∧ d = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_lines_l996_99640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_k_values_l996_99612

/-- An ellipse with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b
  h_e_range : 0 ≤ e ∧ e < 1
  h_ab_relation : b^2 = a^2 * (1 - e^2)

/-- The equation of an ellipse in the form x^2/A + y^2/B = 1 -/
def ellipse_equation (A B : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x^2 / A + y^2 / B = 1

theorem ellipse_eccentricity_k_values 
  (k : ℝ) 
  (h_equation : ∀ x y, ellipse_equation (k + 8) 9 x y)
  (h_eccentricity : ∃ (E : Ellipse), E.e = 1/2 ∧ 
    ((E.a = Real.sqrt (k + 8) ∧ E.b = 3) ∨ 
     (E.a = 3 ∧ E.b = Real.sqrt (k + 8)))) :
  k = 4 ∨ k = -5/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_k_values_l996_99612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_removal_for_no_products_l996_99644

theorem minimum_removal_for_no_products (n : ℕ) (h : n = 1982) : 
  ∃ (S : Finset ℕ), 
    (∀ x, x ∈ S → x ≤ n) ∧ 
    (∀ a b c, a ∈ S → b ∈ S → c ∈ S → a * b ≠ c) ∧
    (∀ T : Finset ℕ, (∀ x, x ∈ T → x ≤ n) ∧ (∀ a b c, a ∈ T → b ∈ T → c ∈ T → a * b ≠ c) → T.card ≤ S.card) ∧
    S.card = n - 43 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_removal_for_no_products_l996_99644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_exists_in_interval_l996_99645

-- Define the function f(x) = |x-2| - ln x
noncomputable def f (x : ℝ) : ℝ := |x - 2| - Real.log x

-- Theorem statement
theorem zero_point_exists_in_interval :
  ∃ x : ℝ, 3 < x ∧ x < 4 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_exists_in_interval_l996_99645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l996_99649

-- Define the ellipse equation
noncomputable def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / 4 + y^2 / m = 1

-- Define the eccentricity
noncomputable def eccentricity (m : ℝ) : ℝ :=
  1 / 2

-- Theorem statement
theorem ellipse_m_values (m : ℝ) :
  (∀ x y, ellipse_equation x y m) → eccentricity m = 1/2 → m = 3 ∨ m = 16/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l996_99649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_bound_l996_99617

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a * Real.log x

theorem f_derivative_bound (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 2) →
  a ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_bound_l996_99617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_special_matrix_l996_99607

theorem determinant_special_matrix (α β : ℝ) : 
  Matrix.det !![0, Real.cos α, Real.sin α;
                Real.sin α, 0, Real.cos β;
                -Real.cos α, -Real.sin β, 0] = 
  -(Real.cos β * (Real.cos α)^2 + Real.sin β * (Real.sin α)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_special_matrix_l996_99607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l996_99603

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := 
  1 / 2 * t.a * t.c * Real.sin t.B

/-- The theorem stating the properties of the triangle ABC -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.b * Real.sin t.A + t.a * Real.cos t.B = 0)
  (h2 : t.b = 2) :
  t.B = 3 * Real.pi / 4 ∧ 
  (∀ (s : Triangle), s.b = 2 → area s ≤ Real.sqrt 2 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l996_99603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l996_99630

/-- A function that satisfies the given conditions -/
def f_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧  -- even function
  (∀ x, f (x + 2) = f x + f 1) ∧  -- functional equation
  (∀ x ∈ Set.Icc 0 1, MonotoneOn f (Set.Icc 0 1))  -- increasing on [0, 1]

/-- Main theorem stating the properties of the function -/
theorem f_properties (f : ℝ → ℝ) (hf : f_conditions f) :
  (∀ x, f (1 + x) = f (1 - x)) ∧  -- symmetry about x = 1
  (∀ x, f (x + 2) = f x) ∧  -- period 2
  (∀ x, Even x → IsLocalMin f x) :=  -- even x-coordinates are local minima
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l996_99630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_is_nineteen_l996_99605

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if four digits are distinct -/
def distinct (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Converts a three-digit number represented by digits to a natural number -/
def toNat3 (x y z : Digit) : ℕ :=
  100 * x.val + 10 * y.val + z.val

theorem digit_sum_is_nineteen
  (a b c d : Digit)
  (h_distinct : distinct a b c d)
  (h_equation : toNat3 a b d + toNat3 c d b = 1000) :
  a.val + b.val + c.val + d.val = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_is_nineteen_l996_99605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C₂_to_l_l996_99666

noncomputable section

/-- The curve C₂ -/
def C₂ (θ : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos θ, 2 * Real.sin θ)

/-- The line l -/
def l (x y : ℝ) : Prop := 2 * x - y - 6 = 0

/-- The distance from a point to the line l -/
def distance_to_l (p : ℝ × ℝ) : ℝ :=
  let (x, y) := p
  abs (2 * x - y - 6) / Real.sqrt 5

theorem max_distance_C₂_to_l :
  ∃ (d : ℝ), d = 2 * Real.sqrt 5 ∧
  ∀ (θ : ℝ), distance_to_l (C₂ θ) ≤ d ∧
  ∃ (θ₀ : ℝ), distance_to_l (C₂ θ₀) = d := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C₂_to_l_l996_99666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_third_not_third_quadrant_l996_99646

-- Define the type for quadrants
inductive Quadrant
| I
| II
| III
| IV

-- Define a function to determine the quadrant of an angle
noncomputable def angle_quadrant (θ : ℝ) : Quadrant :=
  if 0 ≤ θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi / 2 then Quadrant.I
  else if Real.pi / 2 ≤ θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi then Quadrant.II
  else if Real.pi ≤ θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < 3 * Real.pi / 2 then Quadrant.III
  else Quadrant.IV

-- Theorem statement
theorem alpha_third_not_third_quadrant (α : ℝ) 
  (h : angle_quadrant α = Quadrant.II) : 
  angle_quadrant (α / 3) ≠ Quadrant.III := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_third_not_third_quadrant_l996_99646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_find_k_range_l996_99660

-- Define the power function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1)^2 * x^(m^2 - 4*m + 2)

-- Define the function g
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := 2^x - k

-- Theorem for part (Ⅰ)
theorem find_m :
  (∀ x y, 0 < x ∧ x < y → f 0 x < f 0 y) →
  (∀ m : ℝ, (∀ x y, 0 < x ∧ x < y → f m x < f m y) → m = 0) :=
by sorry

-- Theorem for part (Ⅱ)
theorem find_k_range :
  let A : Set ℝ := { y | ∃ x ∈ Set.Icc 1 2, y = f 0 x }
  let B : Set ℝ := { y | ∃ x ∈ Set.Icc 1 2, ∃ k, y = g k x }
  ∀ k, (A ∪ B = A) ↔ k ∈ Set.Icc 0 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_find_k_range_l996_99660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_trailing_zeroes_l996_99686

/-- A type representing a nine-digit number formed using digits 1 to 9 exactly once -/
def NineDigitNumber : Type := Fin 9 → Fin 9

/-- The sum of nine NineDigitNumbers -/
def sum_nine_numbers (numbers : Fin 9 → NineDigitNumber) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin 9)) (λ i =>
    Finset.sum (Finset.univ : Finset (Fin 9)) (λ j => (numbers i j).val + 1))

/-- The number of trailing zeroes in a natural number -/
def trailing_zeroes (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log 10 (Nat.gcd n (10^Nat.log 10 n))

/-- Theorem stating the maximum number of trailing zeroes -/
theorem max_trailing_zeroes :
  ∃ (numbers : Fin 9 → NineDigitNumber),
    trailing_zeroes (sum_nine_numbers numbers) = 8 ∧
    ∀ (other_numbers : Fin 9 → NineDigitNumber),
      trailing_zeroes (sum_nine_numbers other_numbers) ≤ 8 := by
  sorry

#check max_trailing_zeroes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_trailing_zeroes_l996_99686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_distance_theorem_l996_99627

/-- The distance traveled by a light ray reflecting off the x-axis -/
noncomputable def lightRayDistance (a b : ℝ × ℝ) : ℝ :=
  let a' := (a.1, -a.2)  -- Reflection of A across x-axis
  Real.sqrt ((b.1 - a'.1)^2 + (b.2 - a'.2)^2)

/-- Theorem stating that the light ray distance from A to B is 13 -/
theorem light_ray_distance_theorem :
  lightRayDistance (-3, 5) (2, 7) = 13 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval lightRayDistance (-3, 5) (2, 7)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_distance_theorem_l996_99627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knave_of_hearts_tarts_l996_99643

noncomputable def tarts_remaining_after_hearts (T : ℝ) : ℝ :=
  T / 2 - 1/2

noncomputable def tarts_remaining_after_diamonds (T : ℝ) : ℝ :=
  (tarts_remaining_after_hearts T) / 2 - 1/2

noncomputable def tarts_remaining_after_clubs (T : ℝ) : ℝ :=
  (tarts_remaining_after_diamonds T) / 2 - 1/2

theorem knave_of_hearts_tarts :
  ∃ T : ℝ, T > 0 ∧ (tarts_remaining_after_clubs T = 1) ∧ T = 15 := by
  -- We'll use 15 as our witness
  use 15
  -- Split the goal into three parts
  apply And.intro
  · -- Prove T > 0
    norm_num
  · apply And.intro
    · -- Prove tarts_remaining_after_clubs 15 = 1
      -- Expand definitions and simplify
      unfold tarts_remaining_after_clubs tarts_remaining_after_diamonds tarts_remaining_after_hearts
      norm_num
    · -- Prove T = 15
      rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_knave_of_hearts_tarts_l996_99643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_M_two_not_in_M_l996_99685

-- Define the set M
def M : Set Int := sorry

-- Define the properties of M
axiom M_has_positive : ∃ x : Int, x > 0 ∧ x ∈ M
axiom M_has_negative : ∃ x : Int, x < 0 ∧ x ∈ M
axiom M_has_odd : ∃ x : Int, x % 2 = 1 ∧ x ∈ M
axiom M_has_even : ∃ x : Int, x % 2 = 0 ∧ x ∈ M
axiom M_not_contains_neg_one : -1 ∉ M
axiom M_closed_under_addition : ∀ x y : Int, x ∈ M → y ∈ M → (x + y) ∈ M

-- Theorem to prove
theorem zero_in_M_two_not_in_M : 0 ∈ M ∧ 2 ∉ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_M_two_not_in_M_l996_99685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_points_l996_99623

/-- The area of a triangle formed by three points in a 2D coordinate system -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

theorem triangle_area_specific_points :
  triangleArea (0, 0) (8, 15) (8, 0) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_points_l996_99623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l996_99664

open Real

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
axiom f_domain (x : ℝ) : x > 0 → f x ≠ 0

-- Define f(1) = e
axiom f_one : f 1 = Real.exp 1

-- Define the inequality condition for f
axiom f_inequality (x₁ x₂ : ℝ) : x₁ > 0 → x₂ > 0 → x₂ > x₁ → 
  (f x₁ - f x₂) / (x₁ * x₂) > exp x₂ / x₁ - exp x₁ / x₂

-- State the theorem
theorem a_range (a : ℝ) : f (log a) > 2 * exp 1 - a * log a → 1 < a ∧ a < exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l996_99664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_condition_l996_99637

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  !![4, -Real.sqrt 5; 2 * Real.sqrt 5, -3]

theorem matrix_power_condition (m n : ℤ) :
  n ≥ 1 ∧ abs m ≤ n ∧
  (∀ i j, i < 2 ∧ j < 2 →
    ∃ k : ℤ, (A^n - (m + n^2 : ℝ) • A) i j = ↑k) →
  (m = 0 ∧ n = 1) ∨ (m = -6 ∧ n = 7) := by
  sorry

#check matrix_power_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_condition_l996_99637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l996_99699

theorem sin_2alpha_value (α β : ℝ) 
  (h1 : π / 2 < β) 
  (h2 : β < α) 
  (h3 : α < 3 * π / 4)
  (h4 : Real.cos (α - β) = 12 / 13)
  (h5 : Real.sin (α + β) = -3 / 5) : 
  Real.sin (2 * α) = -56 / 65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l996_99699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_binary_divisible_by_16_l996_99673

/-- A positive integer composed only of 0s and 1s -/
def BinaryInteger (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ Nat.digits 10 n → d = 0 ∨ d = 1

theorem smallest_binary_divisible_by_16 :
  ∃ (T : ℕ),
    T > 0 ∧
    BinaryInteger T ∧
    (∃ (X : ℕ), X > 0 ∧ T = 16 * X) ∧
    (∀ (S : ℕ),
      S > 0 →
      BinaryInteger S →
      (∃ (Y : ℕ), Y > 0 ∧ S = 16 * Y) →
      69444375 ≤ Y) :=
by sorry

#eval 16 * 69444375  -- To verify that this equals 1111110000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_binary_divisible_by_16_l996_99673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_divides_product_implies_inequality_l996_99671

theorem factorial_sum_divides_product_implies_inequality (a b : ℕ) 
  (ha : a > 0) (hb : b > 0)
  (h : (Nat.factorial a + Nat.factorial b) ∣ (Nat.factorial a * Nat.factorial b)) :
  3 * a ≥ 2 * b + 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_divides_product_implies_inequality_l996_99671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l996_99693

noncomputable def f (x : ℝ) := Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 4 - 1

noncomputable def smallest_positive_period : ℝ := Real.pi

noncomputable def axis_of_symmetry (k : ℤ) : ℝ := 5 * Real.pi / 12 + k * Real.pi

noncomputable def max_value : ℝ := -3/4
noncomputable def max_point : ℝ := Real.pi/4

noncomputable def min_value : ℝ := -3/2
noncomputable def min_point : ℝ := -Real.pi/12

theorem f_properties :
  (∀ x : ℝ, f (x + smallest_positive_period) = f x) ∧
  (∀ k : ℤ, ∀ x : ℝ, f (axis_of_symmetry k + x) = f (axis_of_symmetry k - x)) ∧
  (∀ x : ℝ, -Real.pi/4 ≤ x ∧ x ≤ Real.pi/4 → f x ≤ max_value) ∧
  (f max_point = max_value) ∧
  (∀ x : ℝ, -Real.pi/4 ≤ x ∧ x ≤ Real.pi/4 → f x ≥ min_value) ∧
  (f min_point = min_value) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l996_99693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_energy_minimum_l996_99691

/-- Represents the remaining energy of an athlete during a marathon --/
noncomputable def Q (t : ℝ) : ℝ :=
  if t ≤ 1 then 10000 - 3600 * t else 400 + 1200 * t + 4800 / t

/-- The time at which the minimum energy occurs --/
def t_min : ℝ := 2

/-- The minimum energy level --/
def E_min : ℝ := 5200

theorem marathon_energy_minimum :
  ∀ t ∈ Set.Icc (0 : ℝ) 4, Q t ≥ E_min ∧ Q t_min = E_min := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_energy_minimum_l996_99691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l996_99653

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with vertex at the origin and focus on the positive x-axis -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- The distance between two points -/
noncomputable def distance (A B : Point) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

/-- A point lies on the parabola -/
def onParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- A point lies on the line x + y - 1 = 0 -/
def onLine (point : Point) : Prop :=
  point.x + point.y - 1 = 0

theorem parabola_intersection_theorem :
  ∀ (parabola : Parabola) (A B : Point),
    onParabola A parabola →
    onParabola B parabola →
    onLine A →
    onLine B →
    distance A B = 8 * Real.sqrt 6 / 11 →
    (parabola.p = 2/11 ∧
     ¬∃ (C : Point), C.y = 0 ∧ distance A C = distance B C ∧ distance A C = distance A B) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l996_99653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l996_99680

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2^(x+1)) / (2^x + 1)

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ (0 < y ∧ y < 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l996_99680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flights_climbed_is_five_l996_99698

/-- Represents the climbing problem with stairs, rope, and ladder. -/
structure ClimbingProblem where
  flight_height : ℝ  -- Height of each flight of stairs
  rope_height : ℝ    -- Height of the rope
  ladder_height : ℝ  -- Height of the ladder
  total_height : ℝ   -- Total height climbed

/-- Creates a climbing problem with the given parameters. -/
noncomputable def create_climbing_problem (flight_height : ℝ) : ClimbingProblem :=
  { flight_height := flight_height,
    rope_height := flight_height / 2,
    ladder_height := flight_height / 2 + 10,
    total_height := 70 }

/-- Calculates the number of flights climbed given a climbing problem. -/
noncomputable def flights_climbed (problem : ClimbingProblem) : ℝ :=
  (problem.total_height - problem.rope_height - problem.ladder_height) / problem.flight_height

/-- Theorem stating that for a climbing problem with 10-foot flights,
    the number of flights climbed is 5. -/
theorem flights_climbed_is_five :
  flights_climbed (create_climbing_problem 10) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flights_climbed_is_five_l996_99698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_m_3_range_of_m_l996_99616

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - m|

-- Part 1: Solution set when m = 3
theorem solution_set_m_3 :
  {x : ℝ | f 3 x ≥ 5} = Set.Iic (-1/2) ∪ Set.Ici (9/2) :=
sorry

-- Part 2: Range of m for which f(x) ≥ 2m - 1 holds for all x
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, f m x ≥ 2*m - 1) ↔ m ≤ 2/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_m_3_range_of_m_l996_99616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satellite_lift_work_l996_99622

/-- Work done to lift a satellite -/
noncomputable def work_done (m : ℝ) (H : ℝ) (R₃ : ℝ) (g : ℝ) : ℝ :=
  m * g * R₃^2 * (1 / R₃ - 1 / (R₃ + H))

/-- Theorem stating the work done to lift a satellite -/
theorem satellite_lift_work :
  let m : ℝ := 6.0 * 1000  -- mass in kg
  let H : ℝ := 350 * 1000  -- height in m
  let R₃ : ℝ := 6380 * 1000  -- Earth's radius in m
  let g : ℝ := 10  -- gravity in m/s²
  ∃ ε > 0, |work_done m H R₃ g - 19911420| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_satellite_lift_work_l996_99622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_b_values_l996_99642

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 4)

noncomputable def f_inv (b : ℝ) (y : ℝ) : ℝ := (b / y + 4) / 3

theorem product_of_b_values (b : ℝ) : 
  f b 3 = f_inv b (b + 2) → 
  ∃ b₁ b₂ : ℝ, b₁ * b₂ = -40/3 ∧ (b = b₁ ∨ b = b₂) := by
  sorry

#check product_of_b_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_b_values_l996_99642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_distance_theorem_l996_99684

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A rectangle with vertices P, Q, R, S -/
structure Rectangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem rectangle_distance_theorem (t : Triangle) (r1 r2 : Rectangle) :
  t.A = (0, 0) →
  t.B = (2 * Real.sqrt 2, 0) →
  t.C = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  distance t.A t.B = 2 * Real.sqrt 2 →
  distance t.A t.C = 1 →
  (t.B.1 - t.A.1) * (t.C.2 - t.A.2) = (t.C.1 - t.A.1) * (t.B.2 - t.A.2) →
  r1.P = t.A →
  r1.Q = (0, 2) →
  r1.R = t.B →
  r1.S = (2 * Real.sqrt 2, 2) →
  r2.P = t.A →
  r2.Q = (1, 0) →
  r2.R = t.C →
  r2.S = (1 / Real.sqrt 2, 1 / Real.sqrt 2 + 1) →
  distance r1.S r2.S = 3 * Real.sqrt (2 + 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_distance_theorem_l996_99684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_ratio_l996_99651

/-- Given a right triangle with sides 5, 12, and 13 -/
def RightTriangle : Set ℝ × Set ℝ × Set ℝ :=
  ({5}, {12}, {13})

/-- Square inscribed with one vertex at the right angle -/
def SquareAtRightAngle (a : ℝ) : Prop :=
  a > 0 ∧ a * (17 / 5) = 12

/-- Square inscribed with one side on the hypotenuse -/
def SquareOnHypotenuse (b : ℝ) : Prop :=
  b > 0 ∧ b * (169 / 60) = 13

/-- The ratio of the side lengths of the two inscribed squares -/
noncomputable def SideRatio (a b : ℝ) : ℝ :=
  a / b

theorem inscribed_squares_ratio :
  ∀ a b : ℝ, SquareAtRightAngle a → SquareOnHypotenuse b →
  SideRatio a b = 39 / 51 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_ratio_l996_99651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_circle_center_correct_answer_l996_99626

/-- The equation of the circle in polar coordinates -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = Real.sqrt 2 * (Real.cos θ + Real.sin θ)

/-- The center of the circle in polar coordinates -/
noncomputable def circle_center : ℝ × ℝ := (1, Real.pi / 4)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center : 
  ∀ ρ θ, circle_equation ρ θ → 
  ∃ r, r * (Real.cos (θ - circle_center.2)) = ρ * Real.cos θ - circle_center.1 ∧ 
       r * (Real.sin (θ - circle_center.2)) = ρ * Real.sin θ - circle_center.1 :=
by
  sorry

/-- The correct answer is option A -/
theorem correct_answer : 
  circle_center = (1, Real.pi / 4) :=
by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_circle_center_correct_answer_l996_99626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bells_toll_together_l996_99604

def bell_intervals : List ℕ := [2, 4, 6, 8, 10, 12]
def time_period : ℕ := 30 * 60  -- 30 minutes in seconds

theorem bells_toll_together (intervals : List ℕ) (period : ℕ) :
  intervals = bell_intervals →
  period = time_period →
  (period / (intervals.foldl Nat.lcm 1) + 1 : ℕ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bells_toll_together_l996_99604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_implies_a_range_l996_99654

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (1 - 2*a)^x else Real.log x / Real.log a + 1/3

theorem function_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  0 < a ∧ a ≤ 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_implies_a_range_l996_99654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_length_different_turns_l996_99600

/-- Represents a point in the triangular grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a direction in the triangular grid --/
inductive Direction
  | Up
  | UpRight
  | DownRight
  | Down
  | DownLeft
  | UpLeft

/-- Represents a path in the triangular grid --/
structure GridPath where
  points : List GridPoint
  turns : ℕ

/-- The triangular grid city --/
class TriangularCity where
  grid : Set GridPoint

/-- Function to make a left turn --/
def leftTurn (d : Direction) : Direction :=
  match d with
  | Direction.Up => Direction.UpLeft
  | Direction.UpRight => Direction.Up
  | Direction.DownRight => Direction.UpRight
  | Direction.Down => Direction.DownRight
  | Direction.DownLeft => Direction.Down
  | Direction.UpLeft => Direction.DownLeft

/-- Function to calculate the length of a path --/
def pathLength (p : GridPath) : ℕ :=
  p.points.length - 1

theorem equal_length_different_turns (city : TriangularCity) :
  ∃ (p1 p2 : GridPath),
    p1.turns = 4 ∧
    p2.turns = 1 ∧
    pathLength p1 = pathLength p2 ∧
    pathLength p1 = 8 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_length_different_turns_l996_99600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l996_99696

theorem trigonometric_identity (α : ℝ) 
  (h1 : Real.tan (α + π/4) = 1/2) 
  (h2 : -π/2 < α) 
  (h3 : α < 0) : 
  (2 * Real.sin α ^ 2 + Real.sin (2 * α)) / Real.cos (α - π/4) = -2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l996_99696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_area_ratio_l996_99648

/-- The ratio of the area of a square inscribed in a right-angled isosceles triangle
    with hypotenuse h to the area of a square inscribed in a semicircle with radius h -/
theorem inscribed_squares_area_ratio (h : ℝ) (h_pos : h > 0) :
  (h^2 / 2) / ((4 / 5) * h^2) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_area_ratio_l996_99648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l996_99656

-- Define the sequence a_n
def a : ℕ → ℤ
  | 0 => 1  -- We define a(0) = 1 to match the given a(1) = 1
  | n + 1 => 2 * a n + (n + 1) - 2

-- State the theorem
theorem a_general_term (n : ℕ) : a n = 2^n - n :=
  by
    -- The proof goes here
    sorry

-- Optionally, we can add some examples to check our definition
#eval a 1  -- Should output 1
#eval a 2  -- Should output 2
#eval a 3  -- Should output 5
#eval a 4  -- Should output 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l996_99656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_and_function_range_l996_99611

noncomputable def α : ℝ := Real.pi / 3

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.cos (x - α)

-- Theorem statement
theorem alpha_and_function_range :
  (Real.sin α * Real.tan α = 3/2) →
  (0 < α) →
  (α < Real.pi) →
  (α = Real.pi/3) ∧
  (Set.Icc 2 3 = Set.image f (Set.Icc 0 (Real.pi/4))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_and_function_range_l996_99611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_l₁_and_l₂_l996_99639

/-- The distance between two parallel lines in 2D space -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- Line l₁: x + 2y + 2 = 0 -/
def l₁ (x y : ℝ) : Prop := x + 2*y + 2 = 0

/-- Line l₂: 2x + 4y - 1 = 0 -/
def l₂ (x y : ℝ) : Prop := 2*x + 4*y - 1 = 0

theorem distance_between_l₁_and_l₂ :
  distance_between_parallel_lines 2 4 4 (-1) = Real.sqrt 5 / 2 := by
  sorry

#check distance_between_l₁_and_l₂

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_l₁_and_l₂_l996_99639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_y_axis_intersection_l996_99692

/-- Circle with center at (-8, 4) and radius 12 -/
def circle_C (x y : ℝ) : Prop := (x + 8)^2 + (y - 4)^2 = 144

/-- Points on the y-axis -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- Distance between a point and the origin -/
noncomputable def distance_to_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

theorem circle_y_axis_intersection :
  ∃ y₁ y₂ : ℝ,
    circle_C 0 y₁ ∧ circle_C 0 y₂ ∧
    on_y_axis 0 y₁ ∧ on_y_axis 0 y₂ ∧
    y₁ + y₂ = 8 ∧
    distance_to_origin 0 y₁ = 4 * Real.sqrt 5 ∧
    distance_to_origin 0 y₂ = 4 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_y_axis_intersection_l996_99692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_invariance_l996_99677

def ElementaryOperation (M : Matrix (Fin 5) (Fin 5) ℤ) : Type :=
  (Fin 5 → Matrix (Fin 5) (Fin 5) ℤ) ⊕ 
  (Fin 5 → Matrix (Fin 5) (Fin 5) ℤ) ⊕ 
  (Fin 5 → Fin 5 → Matrix (Fin 5) (Fin 5) ℤ) ⊕ 
  (Fin 5 → Fin 5 → Matrix (Fin 5) (Fin 5) ℤ)

def ApplyOperation (M : Matrix (Fin 5) (Fin 5) ℤ) (op : ElementaryOperation M) : Matrix (Fin 5) (Fin 5) ℤ :=
  match op with
  | Sum.inl f => f 0  -- Change signs in row
  | Sum.inr (Sum.inl f) => f 0  -- Change signs in column
  | Sum.inr (Sum.inr (Sum.inl f)) => f 0 0  -- Interchange rows
  | Sum.inr (Sum.inr (Sum.inr f)) => f 0 0  -- Interchange columns

def CanTransform (A B : Matrix (Fin 5) (Fin 5) ℤ) : Prop :=
  ∃ (ops : List (ElementaryOperation A)), B = ops.foldl ApplyOperation A

theorem determinant_invariance 
  (A B : Matrix (Fin 5) (Fin 5) ℤ)
  (h1 : ∀ i j, A i j = 1 ∨ A i j = -1)
  (h2 : ∀ i j, B i j = 1 ∨ B i j = -1)
  (h3 : |Matrix.det A| ≠ |Matrix.det B|) :
  ¬(CanTransform A B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_invariance_l996_99677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roses_to_cut_l996_99657

/-- The number of roses from other sources in the vase -/
def other_roses : ℕ := 3

/-- The ratio of Alyssa's roses to other roses -/
def ratio : ℚ := 5 / 1

/-- The number of roses Alyssa needs to cut from her garden -/
def alyssas_roses : ℕ := (other_roses * ratio.num).toNat

theorem roses_to_cut : alyssas_roses = 15 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roses_to_cut_l996_99657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_taking_german_l996_99625

theorem students_taking_german (total : ℕ) (french : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 87 → french = 41 → both = 9 → neither = 33 → 
  ∃ (german : ℕ), german = 22 ∧ 
    total = french + german - both + neither :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_taking_german_l996_99625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_value_l996_99633

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -Real.sqrt 3 * Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 + 1

theorem sin_2theta_value (θ : ℝ) (h1 : f θ = 5/6) (h2 : θ > π/3) (h3 : θ < 2*π/3) :
  Real.sin (2*θ) = (2*Real.sqrt 3 - Real.sqrt 5) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_value_l996_99633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l996_99608

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 40 ∧ B.2 = 0 ∧ C.1 = 0 ∧ C.2 = 30

-- Define point D as the midpoint of AB
def point_D (A B D : ℝ × ℝ) : Prop :=
  D.1 = (A.1 + B.1) / 2 ∧ D.2 = (A.2 + B.2) / 2

-- Helper functions
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_properties (A B C D : ℝ × ℝ) 
  (h_triangle : triangle_ABC A B C) 
  (h_midpoint : point_D A B D) : 
  (area_triangle A B C = 600) ∧ 
  (distance A D = 20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l996_99608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adapted_bowling_ball_volume_l996_99650

-- Define the sphere's diameter and radius
noncomputable def sphere_diameter : ℝ := 40
noncomputable def sphere_radius : ℝ := sphere_diameter / 2

-- Define the holes' properties
noncomputable def hole_depth : ℝ := 10
noncomputable def hole_diameters : List ℝ := [2, 3, 4]

-- Define the volume of a sphere
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Define the volume of a cylinder
noncomputable def cylinder_volume (r : ℝ) (h : ℝ) : ℝ := Real.pi * r^2 * h

-- Calculate the volumes of the holes
noncomputable def hole_volumes : List ℝ := hole_diameters.map (fun d => cylinder_volume (d / 2) hole_depth)

-- Calculate the total volume removed by the holes
noncomputable def total_hole_volume : ℝ := hole_volumes.sum

-- State the theorem
theorem adapted_bowling_ball_volume :
  sphere_volume sphere_radius - total_hole_volume = 10594.17 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adapted_bowling_ball_volume_l996_99650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l996_99689

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : Real.sin α = -5/13) 
  (h2 : α ∈ Set.Icc (3*Real.pi/2) (2*Real.pi)) : 
  Real.tan (α + Real.pi/4) = 7/17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l996_99689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_b_l996_99687

theorem least_possible_b (a b : ℕ+) 
  (ha : (Finset.filter (λ x : ℕ ↦ a.val % x = 0) (Finset.range (a.val + 1))).card = 4)
  (hb : (Finset.filter (λ x : ℕ ↦ b.val % x = 0) (Finset.range (b.val + 1))).card = a.val)
  (hdiv : b.val % a.val = 0) :
  ∀ c : ℕ+, 
    (Finset.filter (λ x : ℕ ↦ c.val % x = 0) (Finset.range (c.val + 1))).card = a.val → 
    c.val % a.val = 0 → 
    b.val ≤ c.val :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_b_l996_99687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l996_99609

-- Define the function f(x)
def f (a c : ℝ) (x : ℝ) : ℝ := a * x^2 + x + c

-- Define the function g(x)
def g (a c k : ℝ) (x : ℝ) : ℝ := f a c x + (4*k + 3) * x

-- Define the minimum value function h(k)
noncomputable def h (k : ℝ) : ℝ :=
  if k ≤ -7 then 12*k + 27
  else if k < -2 then -2*k^2 - 4*k - 5
  else 4*k + 3

-- Theorem statement
theorem quadratic_function_properties (a c : ℝ) (h1 : a > 0) :
  (∀ x, f a c (x - 1/4) = f a c (-(x - 1/4))) →
  (∃ m, m < 1 ∧ ∀ x, f a c x < 0 ↔ m < x ∧ x < 1) →
  (∀ x, f a c x = 2*x^2 + x - 3) ∧
  (∀ k, ∀ x ∈ Set.Icc 1 3, g a c k x ≥ h k) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l996_99609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_increase_l996_99634

theorem rectangle_perimeter_increase (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let original_perimeter := 2 * (a + b)
  let new_perimeter := 2 * (1.1 * a + 1.2 * b)
  (∀ (x y : ℝ), x > 0 → y > 0 → 2 * (1.1 * x + 1.2 * y) / (2 * (x + y)) ≤ 1.2) ∧
  (new_perimeter / original_perimeter = 1.18 → a / b = 1 / 4) := by
  sorry

#check rectangle_perimeter_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_increase_l996_99634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_simplification_and_sum_l996_99667

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := (x^3 + 10*x^2 + 31*x + 30) / (x + 2)

/-- The simplified function g(x) -/
def g (x : ℝ) : ℝ := x^2 + 8*x + 15

/-- The point where f(x) is undefined -/
def D : ℝ := -2

theorem function_simplification_and_sum :
  (∀ x : ℝ, x ≠ D → f x = g x) ∧
  (1 + 8 + 15 + D = 22) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_simplification_and_sum_l996_99667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_charge_calculation_l996_99665

/-- Calculates the total charge for a taxi trip with different rates for peak and off-peak hours -/
theorem taxi_charge_calculation (initial_fee : ℚ) (normal_rate : ℚ) (peak_increase : ℚ) 
  (off_peak_decrease : ℚ) (total_distance : ℚ) (peak_distance : ℚ) (off_peak_distance : ℚ) :
  let peak_rate := normal_rate * (1 + peak_increase)
  let off_peak_rate := normal_rate * (1 - off_peak_decrease)
  let distance_unit := 2 / 5
  let peak_increments := peak_distance / distance_unit
  let off_peak_increments := off_peak_distance / distance_unit
  let total_charge := initial_fee + peak_rate * peak_increments + off_peak_rate * off_peak_increments
  (initial_fee = 9/4) →
  (normal_rate = 3/10) →
  (peak_increase = 1/5) →
  (off_peak_decrease = 3/20) →
  (total_distance = 18/5) →
  (peak_distance = 9/5) →
  (off_peak_distance = 9/5) →
  (peak_distance + off_peak_distance = total_distance) →
  (⌊total_charge * 100 + 1/2⌋ / 100 = 502/100) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_charge_calculation_l996_99665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sum_of_square_roots_l996_99655

theorem integer_sum_of_square_roots (n : ℕ) :
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    Real.sqrt x + Real.sqrt y + Real.sqrt z = 1 ∧
    ∃ (m : ℤ), Real.sqrt (x + n) + Real.sqrt (y + n) + Real.sqrt (z + n) = m) →
  ∃ (k : ℤ), (k % 9 = 1 ∨ k % 9 = -1) ∧ n = (k^2 - 1) / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sum_of_square_roots_l996_99655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_difference_length_l996_99629

/-- Given a circle with radius 1 and two chords, this theorem states the length of the chord
    that subtends the arc equal to the difference of the arcs of the given chords. -/
theorem chord_difference_length
  (a c b d : ℝ) -- Half-lengths of chords and their distances from center
  (ha : 0 < a) (hc : 0 < c) (hb : 0 ≤ b) (hd : 0 ≤ d)
  (hb_lt : b < 1) (hd_lt : d < 1) :
  ∃ (e f : ℝ), 
    0 < e ∧ 0 < f ∧
    (e - f)^2 = 4 * (a * d - b * c)^2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_difference_length_l996_99629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l996_99624

-- Define the angle type
def Angle := ℝ

-- Define a function to determine if an angle is in the first or third quadrant
def isFirstOrThirdQuadrant (θ : ℝ) : Prop :=
  (0 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi / 2) ∨
  (Real.pi < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < 3 * Real.pi / 2)

-- Define the condition that α has the same terminal side as a 150° angle
def hasSameTerminalSideAs150 (α : ℝ) : Prop :=
  ∃ k : ℤ, α = 150 * Real.pi / 180 + k * 2 * Real.pi

-- State the theorem
theorem half_angle_quadrant (α : ℝ) :
  hasSameTerminalSideAs150 α → isFirstOrThirdQuadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l996_99624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_distribution_theorem_l996_99676

/-- Represents a cell in the circular pool -/
structure Cell where
  id : Nat
  neighbors : Fin 3 → Nat

/-- Represents the state of the pool -/
structure PoolState where
  n : Nat
  cells : Fin (2 * n) → Cell
  frog_count : Fin (2 * n) → Nat

/-- Defines a valid initial pool state -/
def valid_initial_state (state : PoolState) : Prop :=
  state.n ≥ 5 ∧
  (∀ i : Fin (2 * state.n), 
    (state.cells i).neighbors 0 ≠ (state.cells i).neighbors 1 ∧
    (state.cells i).neighbors 1 ≠ (state.cells i).neighbors 2 ∧
    (state.cells i).neighbors 2 ≠ (state.cells i).neighbors 0) ∧
  (Finset.sum (Finset.range (2 * state.n)) (λ i ↦ state.frog_count ⟨i, by sorry⟩)) = 4 * state.n + 1

/-- Defines the frog hopping rule -/
def frog_hop (state : PoolState) (i : Fin (2 * state.n)) : PoolState :=
  if state.frog_count i ≥ 3 then
    { state with 
      frog_count := λ j ↦ 
        if j = i then state.frog_count j - 3
        else if (state.cells i).neighbors 0 = j.val ∨ 
                (state.cells i).neighbors 1 = j.val ∨ 
                (state.cells i).neighbors 2 = j.val 
        then state.frog_count j + 1
        else state.frog_count j }
  else state

/-- Defines a roughly even distribution of frogs -/
def roughly_even_distribution (state : PoolState) : Prop :=
  ∀ i : Fin (2 * state.n), 
    state.frog_count i > 0 ∨ 
    (∀ j : Fin 3, state.frog_count ⟨(state.cells i).neighbors j, by sorry⟩ > 0)

/-- The main theorem to prove -/
theorem frog_distribution_theorem (initial_state : PoolState) 
  (h : valid_initial_state initial_state) :
  ∃ final_state : PoolState, roughly_even_distribution final_state :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_distribution_theorem_l996_99676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_monotonic_range_of_a_l996_99662

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * a * x^3 + (1/2) * (a + 1) * x^2 - (a + 2) * x + 6

-- Theorem 1: Minimum value of f(x)
theorem min_value_of_f (a : ℝ) :
  (∃ (x : ℝ), IsLocalMax (f a) x) →
  (∃ (y : ℝ), ∀ (z : ℝ), f a z ≥ f a y ∧ f a y = 13/3) := by
  sorry

-- Theorem 2: Range of a for monotonicity
theorem monotonic_range_of_a :
  ∀ (a : ℝ), (∀ (x y : ℝ), x < y → (f a x < f a y) ∨ (∀ (x y : ℝ), x < y → f a x > f a y)) ↔
  (-5 - 2 * Real.sqrt 5) / 5 ≤ a ∧ a ≤ (-5 + 2 * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_monotonic_range_of_a_l996_99662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_increase_percentage_l996_99681

/-- The percentage increase from an initial value to a new value. -/
noncomputable def percentageIncrease (initial : ℝ) (new : ℝ) : ℝ :=
  ((new - initial) / initial) * 100

/-- Prove that the percentage increase from $60 to $100 is approximately 66.67%. -/
theorem wage_increase_percentage : 
  ∃ ε > 0, |percentageIncrease 60 100 - 66.67| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_increase_percentage_l996_99681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l996_99601

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y = 0

-- Define the point P on the circle
def P : ℝ × ℝ := (1, 1)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

-- Theorem statement
theorem tangent_line_at_P :
  my_circle P.1 P.2 →
  ∀ x y : ℝ, tangent_line x y ↔ 
    (∃ t : ℝ, x = P.1 + t ∧ y = P.2 + t/2 ∧
      ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
        ((x' - P.1)^2 + (y' - P.2)^2 < δ^2 ∧ my_circle x' y') →
        ((x' - x)^2 + (y' - y)^2 < ε^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l996_99601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crucian_bream_comparison_l996_99658

/-- Represents the weight of a fish -/
structure FishWeight where
  weight : ℝ

/-- Crucian weight -/
def crucian : FishWeight := ⟨1⟩

/-- Bream weight -/
def bream : FishWeight := ⟨1⟩

/-- Perch weight -/
def perch : FishWeight := ⟨1⟩

/-- Comparison of fish weights -/
def heavier (a b : FishWeight) : Prop := a.weight > b.weight

/-- Multiply a fish weight by a natural number -/
def mul_weight (n : ℕ) (f : FishWeight) : FishWeight :=
  ⟨n * f.weight⟩

theorem crucian_bream_comparison :
  (heavier (mul_weight 6 crucian) (mul_weight 10 bream)) →
  (heavier (mul_weight 5 perch) (mul_weight 6 crucian)) →
  (heavier (mul_weight 10 crucian) (mul_weight 8 perch)) →
  (heavier (mul_weight 2 crucian) (mul_weight 3 bream)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crucian_bream_comparison_l996_99658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_change_approx_l996_99619

/-- Represents the yearly population change factor -/
def population_change (year : Fin 5) : ℝ :=
  match year with
  | 0 => 1.3  -- 30% increase
  | 1 => 0.7  -- 30% decrease
  | 2 => 1.3  -- 30% increase
  | 3 => 0.85 -- 15% decrease
  | 4 => 0.8  -- 20% decrease

/-- Calculates the net population change over 5 years -/
noncomputable def net_change : ℝ :=
  (Finset.prod (Finset.range 5) (fun i => population_change i)) - 1

theorem population_change_approx :
  abs (net_change + 0.37) < 0.005 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_change_approx_l996_99619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_external_tangent_circles_l996_99636

/-- Given two circles with radii R and r that are externally tangent,
    and a quadrilateral ABCD formed by their common external tangents,
    the radius of the inscribed circle in ABCD is (2rR)/(r+R). -/
theorem inscribed_circle_radius_external_tangent_circles 
  (R r : ℝ) (h : 0 < r ∧ r < R) :
  ∃ (radius : ℝ), radius = (2 * r * R) / (r + R) ∧ 
  radius > 0 ∧ 
  radius < r ∧
  radius < R := by
  -- Let radius be our proposed formula
  let radius := (2 * r * R) / (r + R)
  
  -- We'll prove that this radius satisfies the required properties
  have h_radius_formula : radius = (2 * r * R) / (r + R) := by rfl
  
  -- Prove radius is positive
  have h_radius_pos : radius > 0 := by
    -- This follows from r and R being positive
    sorry
  
  -- Prove radius is less than r
  have h_radius_lt_r : radius < r := by
    -- This can be shown algebraically
    sorry
  
  -- Prove radius is less than R
  have h_radius_lt_R : radius < R := by
    -- This also follows from algebraic manipulation
    sorry
  
  -- Conclude by providing the radius and its properties
  exact ⟨radius, h_radius_formula, h_radius_pos, h_radius_lt_r, h_radius_lt_R⟩

#check inscribed_circle_radius_external_tangent_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_external_tangent_circles_l996_99636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l996_99688

theorem remainder_theorem :
  ∃ q r : Polynomial ℤ,
    X^2030 + 1 = (X^12 - X^9 + X^6 - X^3 + 1) * q + r ∧
    r.degree < (X^12 - X^9 + X^6 - X^3 + 1).degree ∧
    r = -X^20 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l996_99688
