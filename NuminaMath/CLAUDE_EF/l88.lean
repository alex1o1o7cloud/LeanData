import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l88_8830

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 - sqrt 3 * cos x * cos (x + π / 2)

-- State the theorem
theorem f_monotone_increasing :
  ∀ x y, x ∈ Set.Icc (0 : ℝ) (π / 3) →
         y ∈ Set.Icc (0 : ℝ) (π / 3) →
         x ≤ y → f x ≤ f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l88_8830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_sum_number_characterization_l88_8835

def is_pythagorean_sum_number (M : ℕ) : Prop :=
  let a := M / 1000
  let b := (M / 100) % 10
  let c := (M / 10) % 10
  let d := M % 10
  10 * a + b = c^2 + d^2

def G (M : ℕ) : ℚ :=
  let c := (M / 10) % 10
  let d := M % 10
  (c + d : ℚ) / 9

def P (M : ℕ) : ℚ :=
  let a := M / 1000
  let b := (M / 100) % 10
  let c := (M / 10) % 10
  let d := M % 10
  (|10 * (a - c) + (b - d)| : ℚ) / 3

theorem pythagorean_sum_number_characterization :
  ∀ M : ℕ, 1000 ≤ M ∧ M < 10000 →
    (is_pythagorean_sum_number M ∧ 
     (G M).isInt ∧ 
     (P M).isInt) ↔ 
    (M = 8109 ∨ M = 8190 ∨ M = 4536 ∨ M = 4563) := by
  sorry

#check pythagorean_sum_number_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_sum_number_characterization_l88_8835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_flow_maximum_l88_8824

-- Define the traffic speed function
noncomputable def v (x : ℝ) : ℝ :=
  if x ≤ 30 then 50
  else if x > 30 ∧ x ≤ 280 then -0.2 * x + 56
  else 0

-- Define the traffic flow function
noncomputable def f (x : ℝ) : ℝ := x * v x

-- Theorem statement
theorem traffic_flow_maximum :
  ∃ (x_max : ℝ), x_max = 140 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 280 → f x ≤ f x_max ∧
  f x_max = 3920 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_flow_maximum_l88_8824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_in_middle_pile_after_redistribution_l88_8810

/-- Represents the number of cards in each pile -/
structure PileState where
  left : ℕ
  middle : ℕ
  right : ℕ

/-- The game rules for redistributing cards -/
def redistribute (initial : PileState) : PileState :=
  let step1 := initial
  let step2 : PileState := ⟨step1.left - 2, step1.middle + 2, step1.right⟩
  let step3 : PileState := ⟨step2.left, step2.middle + 1, step2.right - 1⟩
  ⟨step3.left + step3.left, step3.middle - step3.left, step3.right⟩

theorem cards_in_middle_pile_after_redistribution 
  (initial : PileState) 
  (h_initial : initial.left = initial.middle ∧ initial.middle = initial.right)
  (h_min_cards : initial.left ≥ 2) :
  (redistribute initial).middle = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_in_middle_pile_after_redistribution_l88_8810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_condition_l88_8802

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- The fixed points A and B -/
def A : Point3D := ⟨2, 3, 0⟩
def B : Point3D := ⟨5, 1, 0⟩

/-- The theorem stating the condition for point P -/
theorem equidistant_condition (P : Point3D) :
  distance P A = distance P B ↔ 6 * P.x - 4 * P.y - 13 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_condition_l88_8802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_center_exists_l88_8832

-- Define the given circle
structure GivenCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the points
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define the given circle
noncomputable def givenCircle : GivenCircle := sorry

-- Assert that B is on the circumference of the given circle
axiom B_on_circle : (B.1 - givenCircle.center.1)^2 + (B.2 - givenCircle.center.2)^2 = givenCircle.radius^2

-- Assert that A is outside the given circle
axiom A_outside_circle : (A.1 - givenCircle.center.1)^2 + (A.2 - givenCircle.center.2)^2 > givenCircle.radius^2

-- Define the perpendicular bisector of AB
def perp_bisector (p q : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ x ↦ (x.1 - (p.1 + q.1)/2)^2 + (x.2 - (p.2 + q.2)/2)^2 = ((p.1 - q.1)^2 + (p.2 - q.2)^2)/4

-- Define the line OB
def line_OB (x : ℝ × ℝ) : Prop :=
  (x.2 - givenCircle.center.2) * (B.1 - givenCircle.center.1) = 
  (x.1 - givenCircle.center.1) * (B.2 - givenCircle.center.2)

-- Theorem statement
theorem tangent_circle_center_exists :
  ∃ O₁ : ℝ × ℝ, perp_bisector A B O₁ ∧ line_OB O₁ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_center_exists_l88_8832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_has_18_terms_l88_8858

/-- An arithmetic sequence with first term a, last term l, and common difference d has n terms -/
def arithmetic_sequence_length (a l d : ℤ) : ℕ :=
  Int.toNat ((l - a) / d + 1)

/-- The number of terms in the specific arithmetic sequence -/
def sequence_length : ℕ := arithmetic_sequence_length (-48) 72 7

theorem sequence_has_18_terms : sequence_length = 18 := by
  -- Proof goes here
  sorry

#eval sequence_length  -- This will evaluate and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_has_18_terms_l88_8858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l88_8897

theorem sqrt_equation_solution : ∃ x : ℝ, (Real.sqrt x + Real.sqrt 567) / Real.sqrt 175 = 2.6 ∧ 
  |x - 112| < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l88_8897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_other_x_intercept_l88_8885

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse defined by its foci and a point on the curve -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  pointOnCurve : Point

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Constant sum of distances from any point on the ellipse to its foci -/
noncomputable def ellipseConstant (e : Ellipse) : ℝ :=
  distance e.pointOnCurve e.focus1 + distance e.pointOnCurve e.focus2

/-- Theorem: For the given ellipse, the other x-intercept is at (91/20, 0) -/
theorem ellipse_other_x_intercept (e : Ellipse)
  (h1 : e.focus1 = ⟨0, 3⟩)
  (h2 : e.focus2 = ⟨4, 0⟩)
  (h3 : e.pointOnCurve = ⟨5, 0⟩)
  : ∃ (x : ℝ), x = 91/20 ∧ 
    ellipseConstant e = distance ⟨x, 0⟩ e.focus1 + distance ⟨x, 0⟩ e.focus2 := by
  sorry

#check ellipse_other_x_intercept

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_other_x_intercept_l88_8885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_layer_rug_area_l88_8879

/-- Proves that the area covered by four layers of rug is 12 square meters given the specified conditions -/
theorem four_layer_rug_area
  (total_area : ℝ)
  (floor_area : ℝ)
  (two_layer_area : ℝ)
  (three_layer_area : ℝ)
  (four_layer_area : ℝ)
  (h1 : total_area = 280)
  (h2 : floor_area = 180)
  (h3 : two_layer_area = 36)
  (h4 : three_layer_area = 16)
  (h5 : floor_area = (floor_area - two_layer_area - three_layer_area - four_layer_area) +
        two_layer_area + three_layer_area + four_layer_area)
  (h6 : total_area = (floor_area - two_layer_area - three_layer_area - four_layer_area) +
        2 * two_layer_area + 3 * three_layer_area + 4 * four_layer_area)
  : four_layer_area = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_layer_rug_area_l88_8879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_problem_l88_8846

/-- The slope of the acute angle bisector between two lines with slopes m₁ and m₂ -/
noncomputable def acute_angle_bisector_slope (m₁ m₂ : ℝ) : ℝ :=
  (m₁ + m₂ + Real.sqrt ((m₁ - m₂)^2 + 4)) / 2

/-- The problem statement -/
theorem angle_bisector_slope_problem :
  let m₁ : ℝ := 2
  let m₂ : ℝ := -1
  acute_angle_bisector_slope m₁ m₂ = (1 + Real.sqrt 13) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_problem_l88_8846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_difference_l88_8895

theorem set_equality_implies_difference (a b : ℝ) : 
  let A : Set ℝ := {1, a + b, a}
  let B : Set ℝ := {0, b / a, b}
  A = B → b - a = -2 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_difference_l88_8895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l88_8876

-- Define the function p
def p (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4) * (x - a)

-- Define the condition for p being monotonically increasing
def p_monotonic (a : ℝ) : Prop :=
  ∀ x, (x < -2 ∨ x > 2) → StrictMono (fun x => p a x)

-- Define the integral function for q
def q_integral (x : ℝ) : ℝ := x^2 - 2*x

-- Define the condition for q
def q_condition (a : ℝ) : Prop :=
  ∀ x, q_integral x > a

-- Define the theorem
theorem a_range :
  ∀ a : ℝ, (p_monotonic a ∧ ¬q_condition a) ∨ (¬p_monotonic a ∧ q_condition a) →
    a ∈ Set.Iic (-2) ∪ Set.Icc (-1) 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l88_8876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_length_proof_l88_8890

/-- Converts centimeters to kilometers -/
noncomputable def cm_to_km (cm : ℝ) : ℝ := cm / 100000

/-- Calculates the actual length given the map length and scale -/
noncomputable def actual_length (map_length : ℝ) (scale : ℝ) : ℝ := map_length * scale

theorem road_length_proof (map_length : ℝ) (scale : ℝ) (h1 : map_length = 6) (h2 : scale = 2500000) :
  cm_to_km (actual_length map_length scale) = 150 := by
  sorry

#check road_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_length_proof_l88_8890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alkali_concentration_theorem_l88_8864

-- Define the concentration function
noncomputable def concentration (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then -16 / (x + 2) - x + 8
  else if 2 < x ∧ x ≤ 4 then 4 - x
  else 0

-- Define the effective inhibition threshold
def effective_inhibition_threshold : ℝ := 1

-- Define the time of second introduction
def second_introduction_time : ℝ := 2

-- Define the concentration function after second introduction
noncomputable def concentration_after_second (x : ℝ) : ℝ :=
  if 2 < x ∧ x ≤ 4 then
    14 - (2 * x + 16 / x)
  else
    concentration x

-- State the theorem
theorem alkali_concentration_theorem :
  -- Part 1: Time interval for effective inhibition
  (∀ x : ℝ, (5 - Real.sqrt 17) / 2 ≤ x ∧ x ≤ 3 → concentration x ≥ effective_inhibition_threshold) ∧
  (∀ x : ℝ, x < (5 - Real.sqrt 17) / 2 ∨ x > 3 → concentration x < effective_inhibition_threshold) ∧
  -- Part 2: Maximum concentration after second introduction
  (∀ x : ℝ, 2 < x ∧ x ≤ 4 → concentration_after_second x ≤ 14 - 8 * Real.sqrt 2) ∧
  (concentration_after_second (2 * Real.sqrt 2) = 14 - 8 * Real.sqrt 2) := by
  sorry

#check alkali_concentration_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alkali_concentration_theorem_l88_8864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_alpha_l88_8825

theorem parallel_vectors_tan_alpha (α : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 3]
  let b : Fin 2 → ℝ := ![Real.sin α, Real.cos α]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, a i = k * b i)) → Real.tan α = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_alpha_l88_8825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rare_card_percentage_approx_l88_8840

/-- The number of rare cards in the pack -/
def rare_cards : ℕ := 2

/-- The ratio of regular cards to rare cards -/
def regular_to_rare_ratio : ℕ := 5

/-- The total number of cards in the pack -/
def total_cards : ℕ := rare_cards + regular_to_rare_ratio * rare_cards

/-- The percentage of rare cards in the pack -/
def rare_card_percentage : ℚ := (rare_cards : ℚ) / (total_cards : ℚ) * 100

theorem rare_card_percentage_approx :
  abs (rare_card_percentage - 16.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rare_card_percentage_approx_l88_8840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_opens_downwards_l88_8893

/-- A function f(x) = ax^n is quadratic if n = 2 -/
def is_quadratic (a n : ℝ) : Prop := n = 2

/-- A quadratic function f(x) = ax^2 opens downwards if a < 0 -/
def opens_downwards (a : ℝ) : Prop := a < 0

/-- The given function f(x) = (m-1)x^(m^2+1) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * (x ^ (m^2 + 1))

theorem quadratic_function_opens_downwards (m : ℝ) : 
  is_quadratic (m - 1) (m^2 + 1) ∧ opens_downwards (m - 1) → m = -1 := by
  sorry

#check quadratic_function_opens_downwards

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_opens_downwards_l88_8893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l88_8898

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : Real.sqrt (1 - b^2 / a^2) = 1 / 3

/-- The vertices of the ellipse -/
def Ellipse.vertices (e : Ellipse) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ :=
  (-e.a, 0, e.a, 0, 0, e.b)

/-- The dot product condition for the vertices -/
def Ellipse.vertex_condition (e : Ellipse) : Prop :=
  let (x₁, y₁, x₂, y₂, x₃, y₃) := e.vertices
  (x₁ - x₃) * (x₂ - x₃) + (y₁ - y₃) * (y₂ - y₃) = -1

/-- The theorem stating the specific equation of the ellipse -/
theorem ellipse_equation (e : Ellipse) (h : e.vertex_condition) :
    e.a = 3 ∧ e.b = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l88_8898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cp_pa_ratio_l88_8836

/- Define the triangle and its properties -/
structure Triangle (A B C : ℝ × ℝ) :=
  (ab_length : dist A B = 24)
  (ac_length : dist A C = 15)

/- Define the angle bisector and midpoint -/
def AngleBisector (A B C D : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ D = (1 - t) • B + t • C

def Midpoint (M A D : ℝ × ℝ) : Prop :=
  M = (1/2 : ℝ) • (A + D)

/- Define the intersection point -/
def Intersect (P A C B M : ℝ × ℝ) : Prop :=
  ∃ (t s : ℝ), 0 < t ∧ t < 1 ∧ 0 < s ∧ s < 1 ∧ 
    P = (1 - t) • A + t • C ∧
    P = (1 - s) • B + s • M

/- Main theorem -/
theorem cp_pa_ratio 
  (A B C D M P : ℝ × ℝ) 
  (tri : Triangle A B C) 
  (bisector : AngleBisector A B C D) 
  (mid : Midpoint M A D) 
  (intersect : Intersect P A C B M) : 
  dist C P / dist P A = 13/8 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cp_pa_ratio_l88_8836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barn_paint_area_l88_8878

/-- Calculate the total area to be painted in a barn with given dimensions and conditions. -/
theorem barn_paint_area (width length height door_width door_height : ℝ) 
  (h_width : width = 12)
  (h_length : length = 15)
  (h_height : height = 8)
  (h_door_width : door_width = 4)
  (h_door_height : door_height = 3) :
  let wall_area_1 := 2 * (width * height - door_width * door_height)
  let wall_area_2 := 2 * length * height
  let ceiling_area := width * length
  let total_area := 2 * (wall_area_1 + wall_area_2) / 2 + ceiling_area
  total_area = 1020 := by
  sorry

#check barn_paint_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barn_paint_area_l88_8878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zeros_omega_range_l88_8814

theorem function_zeros_omega_range (ω : ℝ) (h_ω_pos : ω > 0) :
  (∃! (z₁ z₂ : ℝ), 0 < z₁ ∧ z₁ < z₂ ∧ z₂ < π ∧
    Real.sqrt 2 * Real.sin (ω * z₁ + π/4) = 0 ∧
    Real.sqrt 2 * Real.sin (ω * z₂ + π/4) = 0 ∧
    (∀ z, 0 < z ∧ z < π ∧ Real.sqrt 2 * Real.sin (ω * z + π/4) = 0 → z = z₁ ∨ z = z₂)) →
  7/4 < ω ∧ ω ≤ 11/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zeros_omega_range_l88_8814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_length_is_576_l88_8882

/-- Represents a road with trees planted on both sides -/
structure TreeLinedRoad where
  length : ℕ  -- Length of the road in meters

/-- Calculates the number of trees that can be planted given a planting interval -/
def trees_planted (road : TreeLinedRoad) (interval : ℕ) : ℤ :=
  (road.length / interval : ℤ) + 1

/-- Theorem stating the conditions and the expected road length -/
theorem road_length_is_576 (road : TreeLinedRoad) :
  trees_planted road 8 + 8 = trees_planted road 9 - 8 →
  road.length = 576 := by
  intro h
  -- The proof goes here
  sorry

/-- Example usage -/
def example_road : TreeLinedRoad := ⟨576⟩

#eval trees_planted example_road 8
#eval trees_planted example_road 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_length_is_576_l88_8882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l88_8886

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

/-- The theorem stating the eccentricity of the given hyperbola -/
theorem hyperbola_eccentricity (C : Hyperbola) (Q : Circle) :
  (Q.h = 2 ∧ Q.k = -3 ∧ Q.r^2 = 13) →  -- Center and radius of circle Q
  (C.b / C.a = 3 / 2) →                -- Condition from asymptote passing through circle center
  eccentricity C = Real.sqrt 13 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l88_8886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_score_is_91_l88_8833

noncomputable def average (scores : List ℝ) : ℝ :=
  scores.sum / scores.length

theorem third_score_is_91 (scores : List ℝ) :
  scores.length = 4 →
  scores.get! 0 = 55 →
  scores.get! 1 = 67 →
  scores.get! 3 = 55 →
  average scores = 67 →
  scores.get! 2 = 91 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_score_is_91_l88_8833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_equipment_installation_l88_8827

-- Define the number of units that can be installed per day
def units_per_day (skilled_workers new_workers : ℕ) : ℕ := 
  4 * skilled_workers + 2 * new_workers

-- Define the total units that need to be installed
def total_units : ℕ := 360

-- Define the number of days available
def available_days : ℕ := 15

-- Define the theorem
theorem sports_equipment_installation :
  -- Part 1: Skilled worker installs 4 units/day, new worker installs 2 units/day
  (units_per_day 2 1 = 10 ∧ units_per_day 3 2 = 16) →
  -- Part 2: There are 4 recruitment plans
  (∃ (plans : List (ℕ × ℕ)), 
    plans.length = 4 ∧
    (∀ (pair : ℕ × ℕ), pair ∈ plans → 
      let (n, m) := pair
      1 < n ∧ n < 8 ∧ 
      units_per_day n m * available_days = total_units)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_equipment_installation_l88_8827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_rent_approximation_l88_8819

/-- Calculates the monthly rent per square meter required to recoup the entire investment -/
noncomputable def calculate_monthly_rent (construction_cost : ℝ) (useful_life : ℕ) (annual_interest_rate : ℝ) : ℝ :=
  let annual_rent := construction_cost * (((1 + annual_interest_rate) ^ useful_life * annual_interest_rate) / ((1 + annual_interest_rate) ^ useful_life - 1))
  annual_rent / 12

/-- Theorem stating that the calculated monthly rent is approximately 1.14 yuan -/
theorem monthly_rent_approximation (construction_cost : ℝ) (useful_life : ℕ) (annual_interest_rate : ℝ)
  (h1 : construction_cost = 250)
  (h2 : useful_life = 50)
  (h3 : annual_interest_rate = 0.05) :
  |calculate_monthly_rent construction_cost useful_life annual_interest_rate - 1.14| < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_monthly_rent 250 50 0.05

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_rent_approximation_l88_8819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l88_8855

theorem largest_expression : 
  let a := Real.sqrt (Real.rpow 7 (1/3) * Real.rpow 8 (1/3))
  let b := Real.sqrt (8 * Real.rpow 7 (1/3))
  let c := Real.sqrt (7 * Real.rpow 8 (1/3))
  let d := Real.rpow (7 * Real.sqrt 8) (1/3)
  let e := Real.rpow (8 * Real.sqrt 7) (1/3)
  b = max a (max b (max c (max d e))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l88_8855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_specific_line_equation_l88_8815

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 5 = 0

-- Define the line l
def line_l (x y a : ℝ) : Prop := x + a*y - a - 2 = 0

-- Theorem 1: The line l always intersects the circle C
theorem line_intersects_circle :
  ∀ a : ℝ, ∃ x y : ℝ, circle_C x y ∧ line_l x y a :=
by sorry

-- Theorem 2: When AB = 2√2, the equation of line l is x - y = 1
theorem specific_line_equation :
  ∃ a : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_l x₁ y₁ a ∧ line_l x₂ y₂ a ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) →
  (∀ x y : ℝ, line_l x y a ↔ x - y = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_specific_line_equation_l88_8815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_formula_l88_8851

theorem tan_sum_formula (x y : Real) 
  (h1 : Real.sin x + Real.sin y = 15/17)
  (h2 : Real.cos x + Real.cos y = 8/17)
  (h3 : Real.sin (x - y) = 1/5) :
  Real.tan x + Real.tan y = 195 * Real.sqrt 6 / 328 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_formula_l88_8851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_abscissa_l88_8881

open Real

-- Define the curve function
noncomputable def f (x : ℝ) : ℝ := x^2 / 2 - 2 * log x + 1

-- Define the derivative of the curve function
noncomputable def f' (x : ℝ) : ℝ := x - 2 / x

-- Theorem statement
theorem tangent_point_abscissa :
  ∀ x : ℝ, x > 0 → f' x = 1 → x = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_abscissa_l88_8881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_probabilities_l88_8867

noncomputable def prob_A_hit : ℝ := 2/3
noncomputable def prob_B_hit : ℝ := 3/4

noncomputable def both_hit : ℝ := prob_A_hit * prob_B_hit
noncomputable def A_three_of_four : ℝ := 
  prob_A_hit^3 * (1 - prob_A_hit) + (1 - prob_A_hit) * prob_A_hit^3
noncomputable def B_terminated_four : ℝ := 
  prob_B_hit^2 * (1 - prob_B_hit)^2 + (1 - prob_B_hit) * prob_B_hit * (1 - prob_B_hit)^2

theorem shooting_probabilities :
  both_hit = 1/2 ∧ 
  A_three_of_four = 16/81 ∧ 
  B_terminated_four = 3/64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_probabilities_l88_8867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_ionization_approx_l88_8848

/-- The ionization constant for HCN -/
def K_HCN : ℝ := 7.2e-10

/-- The concentration in mol/L -/
def C : ℝ := 0.1

/-- The degree of ionization -/
noncomputable def α : ℝ := Real.sqrt (K_HCN * C)

/-- Theorem stating that the calculated degree of ionization is approximately 8.5e-5 -/
theorem degree_of_ionization_approx :
  ∃ ε > 0, |α - 8.5e-5| < ε ∧ ε < 1e-7 := by
  sorry

#eval K_HCN
#eval C
-- Note: We can't use #eval for α since it's noncomputable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_ionization_approx_l88_8848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l88_8875

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- An ellipse with axes parallel to the coordinate axes, tangent to the x-axis at (6, 0) and to the y-axis at (0, 3) -/
structure Ellipse where
  axes_parallel : Bool
  tangent_x : Point := ⟨6, 0⟩
  tangent_y : Point := ⟨0, 3⟩

/-- The distance between the foci of the ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ :=
  6 * Real.sqrt 3

/-- Theorem: The distance between the foci of the given ellipse is 6√3 -/
theorem ellipse_focal_distance (e : Ellipse) (h1 : e.axes_parallel = true) 
  (h2 : e.tangent_x = ⟨6, 0⟩) (h3 : e.tangent_y = ⟨0, 3⟩) : 
  focal_distance e = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l88_8875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equality_trigonometric_expressions_with_given_tan_l88_8854

-- Part 1
theorem trigonometric_expression_equality :
  (Real.sqrt 3 * Real.sin (-20 * Real.pi / 3)) / Real.tan (11 * Real.pi / 3) -
  Real.cos (13 * Real.pi / 4) * Real.tan (-37 * Real.pi / 4) =
  (3 * Real.sqrt 3 - Real.sqrt 2) / 2 := by sorry

-- Part 2
theorem trigonometric_expressions_with_given_tan (α : Real) (h : Real.tan α = 4/3) :
  ((Real.sin (2*α) + 2 * Real.sin α * Real.cos α) / (2 * Real.cos (2*α) - Real.sin (2*α)) = -24/19) ∧
  (Real.sin α * Real.cos α = 12/25) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equality_trigonometric_expressions_with_given_tan_l88_8854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_distances_l88_8862

/-- The parabola x² = 4y -/
def Parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- The line 4x - 3y - 7 = 0 -/
def Line1 (x y : ℝ) : Prop := 4*x - 3*y - 7 = 0

/-- The line y + 2 = 0 -/
def Line2 (y : ℝ) : Prop := y + 2 = 0

/-- Distance from a point (x, y) to Line1 -/
noncomputable def distToLine1 (x y : ℝ) : ℝ := |4*x - 3*y - 7| / 5

/-- Distance from a point (x, y) to Line2 -/
noncomputable def distToLine2 (y : ℝ) : ℝ := |y + 2|

/-- Sum of distances from a point on the parabola to Line1 and Line2 -/
noncomputable def sumOfDistances (a : ℝ) : ℝ := distToLine1 a (a^2/4) + distToLine2 (a^2/4)

theorem min_sum_of_distances :
  ∃ (min : ℝ), min = 3 ∧ ∀ (a : ℝ), sumOfDistances a ≥ min := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_distances_l88_8862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approximation_l88_8829

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  distance t.A t.B = distance t.B t.C ∧ distance t.B t.C = distance t.C t.A

/-- Calculate the area of an equilateral triangle given its side length -/
noncomputable def areaEquilateral (side : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * side^2

/-- Main theorem -/
theorem triangle_area_approximation (A B C P : Point) :
  let t := Triangle.mk A B C
  isEquilateral t →
  distance P A = 7 →
  distance P B = 7 →
  distance P C = 10 →
  ∃ (side : ℝ), distance A B = side ∧ 
    Int.floor (areaEquilateral side) = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approximation_l88_8829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_minus_sqrt3_cos_l88_8861

/-- The function f(x) = sin x - √3 cos x has a maximum value of 2. -/
theorem max_value_sin_minus_sqrt3_cos : 
  ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), Real.sin x - Real.sqrt 3 * Real.cos x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_minus_sqrt3_cos_l88_8861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_property_l88_8820

def a : ℕ → ℚ
  | 0 => 1/3  -- Added case for 0
  | 1 => 1/3
  | n+2 => let a_n_minus_2 := a n
           let a_n_minus_1 := a (n+1)
           ((1 - 2*a_n_minus_2) * a_n_minus_1^2) / (2*a_n_minus_1^2 - 4*a_n_minus_2^2 * a_n_minus_1^2 + a_n_minus_2)

theorem a_n_property (n : ℕ) : ∃ (k : ℕ), (1 / a n - 2 : ℚ) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_property_l88_8820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hill_climbing_average_speed_l88_8891

/-- Calculates the average speed for a round trip journey up and down a hill -/
theorem hill_climbing_average_speed 
  (time_up : ℝ) 
  (time_down : ℝ) 
  (speed_up : ℝ) 
  (h1 : time_up = 4) 
  (h2 : time_down = 2) 
  (h3 : speed_up = 2.625) : 
  (2 * time_up * speed_up) / (time_up + time_down) = 3.5 := by
  -- Replace all occurrences of variables with their actual values
  rw [h1, h2, h3]
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num
  -- QED
  done

#check hill_climbing_average_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hill_climbing_average_speed_l88_8891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_exists_x_for_a_less_than_one_power_less_than_e_l88_8883

-- Define the function f(x) = ln(1+x) / x
noncomputable def f (x : ℝ) : ℝ := (Real.log (1 + x)) / x

-- Define the derivative of f
noncomputable def f_deriv (x : ℝ) : ℝ := (1 / (1 + x) - Real.log (1 + x)) / x^2

-- Statement 1: f(x) is decreasing on (0, ∞)
theorem f_decreasing : ∀ x > 0, HasDerivAt f (f_deriv x) x ∧ f_deriv x < 0 := by sorry

-- Statement 2: For all a < 1, there exists an x > 0 such that ln(1+x) ≥ ax
theorem exists_x_for_a_less_than_one : ∀ a < 1, ∃ x > 0, Real.log (1 + x) ≥ a * x := by sorry

-- Statement 3: For all n ∈ N*, (1 + 1/n)^n < e
theorem power_less_than_e : ∀ n : ℕ+, (1 + 1 / (n : ℝ)) ^ (n : ℝ) < Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_exists_x_for_a_less_than_one_power_less_than_e_l88_8883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_gap_ultra_even_second_shortest_gap_ultra_even_l88_8863

-- Define what an ultra-even year is
def is_ultra_even (year : ℕ) : Prop :=
  ∀ d, d ∈ (Nat.digits 10 year) → d % 2 = 0

-- Define the range of years we're considering
def year_range : Set ℕ := {y | 1 ≤ y ∧ y ≤ 10000}

-- Define the gap between two years
def gap (year1 year2 : ℕ) : ℕ := Int.natAbs (year1 - year2)

-- Theorem for the longest gap
theorem longest_gap_ultra_even :
  ∃ y1 y2 : ℕ, y1 ∈ year_range ∧ y2 ∈ year_range ∧ 
  is_ultra_even y1 ∧ is_ultra_even y2 ∧
  gap y1 y2 = 1112 ∧
  (∀ y : ℕ, y ∈ year_range → is_ultra_even y → y ≤ y1 ∨ y ≥ y2) ∧
  (∀ a b : ℕ, a ∈ year_range → b ∈ year_range → 
   is_ultra_even a → is_ultra_even b → gap a b ≤ 1112) :=
by sorry

-- Theorem for the second-shortest gap
theorem second_shortest_gap_ultra_even :
  ∃ y1 y2 : ℕ, y1 ∈ year_range ∧ y2 ∈ year_range ∧ 
  is_ultra_even y1 ∧ is_ultra_even y2 ∧
  gap y1 y2 = 12 ∧
  (∀ a b : ℕ, a ∈ year_range → b ∈ year_range → 
   is_ultra_even a → is_ultra_even b → a ≠ b → 
   gap a b = 2 ∨ gap a b ≥ 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_gap_ultra_even_second_shortest_gap_ultra_even_l88_8863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_three_l88_8807

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) + 1

-- Define the point of interest
def point : ℝ × ℝ := (0, 2)

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := 2 * x + 2

-- Define the lines y = 0 and y = x
def line_y_equals_0 : ℝ → ℝ := λ _ ↦ 0
def line_y_equals_x : ℝ → ℝ := λ x ↦ x

-- Theorem statement
theorem triangle_area_is_three :
  ∃ x₁ x₂ : ℝ, tangent_line x₁ = line_y_equals_0 x₁ ∧
             tangent_line x₂ = line_y_equals_x x₂ ∧
             (1/2) * (x₂ - x₁) * (line_y_equals_x x₂) = 3 := by
  sorry

#check triangle_area_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_three_l88_8807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_condition_range_of_f_l88_8838

-- Define the quadratic function
def quadratic (a c : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x + c

-- Define the solution set condition
def solution_set (a c : ℝ) : Prop :=
  ∀ x, quadratic a c x < 0 ↔ (x < -1 ∨ x > 2)

-- Theorem for statement B
theorem quadratic_condition (a c : ℝ) (h : solution_set a c) : a + c = 2 := by
  sorry

-- Define the function for statement C
noncomputable def f (x : ℝ) : ℝ := 2 * x + Real.sqrt (1 - x)

-- Theorem for statement C
theorem range_of_f : Set.range f = Set.Iic (17/8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_condition_range_of_f_l88_8838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_inequality_l88_8839

-- Define a circumscribed quadrilateral
structure CircumscribedQuadrilateral (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C D : V)
  (is_circumscribed : sorry) -- Placeholder for the actual condition

-- Define the inradius of a polygon
noncomputable def inradius {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (vertices : List V) : ℝ := sorry

-- Theorem statement
theorem inradius_inequality 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (ABCD : CircumscribedQuadrilateral V) : 
  inradius [ABCD.A, ABCD.B, ABCD.C, ABCD.D] < 
  inradius [ABCD.A, ABCD.B, ABCD.C] + inradius [ABCD.A, ABCD.C, ABCD.D] :=
by
  sorry -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_inequality_l88_8839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combine_with_2sqrt3_l88_8811

theorem combine_with_2sqrt3 :
  (∃ a : ℝ, Real.sqrt 27 = a * Real.sqrt 3) ∧
  (∀ b : ℝ, Real.sqrt 8 ≠ b * Real.sqrt 3) ∧
  (∀ c : ℝ, Real.sqrt 18 ≠ c * Real.sqrt 3) ∧
  (∀ d : ℝ, Real.sqrt 9 ≠ d * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combine_with_2sqrt3_l88_8811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_trapezoid_angles_l88_8856

/-- A right trapezoid with specific diagonal and base ratios -/
structure RightTrapezoid where
  -- The shorter base
  base₁ : ℝ
  -- The longer base
  base₂ : ℝ
  -- The shorter diagonal
  diag₁ : ℝ
  -- The longer diagonal
  diag₂ : ℝ
  -- The base ratio condition
  base_ratio : base₂ = 4 * base₁
  -- The diagonal ratio condition
  diag_ratio : diag₂ = 2 * diag₁
  -- Positive base and diagonal lengths
  base₁_pos : base₁ > 0
  diag₁_pos : diag₁ > 0

/-- The angles of a right trapezoid with the given conditions -/
noncomputable def trapezoid_angles (t : RightTrapezoid) : Fin 4 → ℝ
| 0 => Real.pi / 2
| 1 => Real.pi / 2
| 2 => Real.arctan (2 / 3)
| 3 => Real.pi - Real.arctan (2 / 3)

/-- Theorem stating that the angles of the right trapezoid with the given conditions are as specified -/
theorem right_trapezoid_angles (t : RightTrapezoid) :
  trapezoid_angles t = λ i => match i with
    | 0 => Real.pi / 2
    | 1 => Real.pi / 2
    | 2 => Real.arctan (2 / 3)
    | 3 => Real.pi - Real.arctan (2 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_trapezoid_angles_l88_8856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_equivalence_l88_8826

theorem sin_shift_equivalence (x : ℝ) :
  Real.sin (2*x - π/4) = Real.sin (2*(x - π/8)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_equivalence_l88_8826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olympic_card_game_expected_draws_l88_8880

/-- Represents the Olympic card game -/
structure OlympicCardGame where
  total_cards : ℕ
  emblem_cards : ℕ
  mascot_cards : ℕ
  players : ℕ

/-- Calculates the probability of drawing k cards of a certain type from n total cards -/
def prob_draw (n k : ℕ) (total : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (Nat.choose total k : ℚ)

/-- Calculates the probability of a specific draw sequence -/
def prob_sequence (game : OlympicCardGame) : ℕ → ℚ
  | 0 => 0
  | 1 => prob_draw game.mascot_cards 2 game.total_cards
  | n + 1 =>
    (1 - prob_draw game.mascot_cards 2 game.total_cards) *
    prob_sequence { game with
      total_cards := game.total_cards - 2
      mascot_cards := game.mascot_cards - 1
    } n

/-- Calculates the expected number of draws until the game ends -/
noncomputable def expected_draws (game : OlympicCardGame) : ℚ :=
  let probs := List.range game.players |>.map (λ i => ((i + 1 : ℕ), prob_sequence game (i + 1)))
  probs.foldl (λ acc (i, p) => acc + (i : ℚ) * p) 0

/-- The main theorem stating the expected number of draws in the Olympic card game -/
theorem olympic_card_game_expected_draws :
  let game : OlympicCardGame := {
    total_cards := 8
    emblem_cards := 3
    mascot_cards := 5
    players := 4
  }
  expected_draws game = 15 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_olympic_card_game_expected_draws_l88_8880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_f_on_interval_l88_8843

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 - (3/2) * x^2 + 5

-- Define the interval
def interval : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem max_min_f_on_interval :
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y) ∧
  (∃ x ∈ interval, f x = 7) ∧
  (∃ x ∈ interval, f x = -9) := by
  sorry

#check max_min_f_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_f_on_interval_l88_8843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_statement_true_l88_8896

open Real

theorem exactly_one_statement_true : ∃! n : Fin 3, 
  match n with
  | 0 => (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ¬(∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0)
  | 1 => ∀ a b c : ℝ, (b = Real.sqrt (a * c)) ↔ (∃ r : ℝ, b = a * r ∧ c = b * r)
  | 2 => ∀ m : ℝ, (m = -1) ↔ 
         (∀ x y : ℝ, (m * x + (2 * m - 1) * y + 1 = 0) → 
         (3 * x + m * y + 2 = 0) → 
         (m * 3 + m * (2 * m - 1) = 0))
  := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_statement_true_l88_8896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_parallel_to_line_through_points_outcomes_l88_8837

-- Define the basic geometric objects
variable (L : Type) -- The given line type
variable (P Q : Type) -- The point type

-- Define a predicate for a point being on a line
variable (IsOn : P → L → Prop)

-- Define the condition that P and Q are not on L
variable (l : L)
variable (p q : P)
axiom P_not_on_L : ¬ IsOn p l
axiom Q_not_on_L : ¬ IsOn q l

-- Define the possible outcomes for the plane
inductive PlaneOutcome
  | UniqueConstruction
  | InfiniteConstructions
  | NoExistence

-- Theorem stating that all outcomes are possible
theorem plane_parallel_to_line_through_points_outcomes :
  ∃ (outcome : PlaneOutcome), 
    (outcome = PlaneOutcome.UniqueConstruction) ∨
    (outcome = PlaneOutcome.InfiniteConstructions) ∨
    (outcome = PlaneOutcome.NoExistence) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_parallel_to_line_through_points_outcomes_l88_8837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_EC_dot_ED_equals_three_l88_8822

/-- Square ABCD with side length 2 and E as midpoint of AB -/
structure SquareABCD where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  is_square : A = (0, 0) ∧ B = (2, 0) ∧ C = (2, 2) ∧ D = (0, 2)
  E_midpoint : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

/-- Vector from point P to point Q -/
def vector (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Theorem: EC · ED = 3 in the given square ABCD -/
theorem EC_dot_ED_equals_three (square : SquareABCD) :
  dot_product (vector square.E square.C) (vector square.E square.D) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_EC_dot_ED_equals_three_l88_8822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_implies_a_neg_one_f_inequality_implies_m_bound_l88_8842

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 - a * x) / (x - 1) / Real.log (1 / 2)

-- Theorem 1: If f is odd, then a = -1
theorem f_odd_implies_a_neg_one (a : ℝ) :
  (∀ x, f a x = -f a (-x)) → a = -1 := by sorry

-- Theorem 2: If f(x) > (1/2)ˣ + m for all x ∈ [3, 4], then m < -9/8
theorem f_inequality_implies_m_bound (a : ℝ) (m : ℝ) :
  (a = -1) →
  (∀ x ∈ Set.Icc 3 4, f a x > (1/2)^x + m) →
  m < -9/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_implies_a_neg_one_f_inequality_implies_m_bound_l88_8842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l88_8857

/-- Given a train crossing a platform and a signal pole, calculate the platform length -/
theorem platform_length_calculation
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 40)
  (h3 : time_cross_pole = 18) :
  ∃ (platform_length : ℝ),
    (abs (platform_length - 366.7) < 0.1) ∧
    (train_length + platform_length) / time_cross_platform = train_length / time_cross_pole :=
by
  sorry

#check platform_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l88_8857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_and_range_l88_8801

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem f_sum_and_range :
  (f 1 + f 2 + f 3 + f (1/2) + f (1/3) = 5/2) ∧
  (∀ x : ℝ, 0 < f x ∧ f x ≤ 1) ∧
  (∀ y : ℝ, 0 < y → y ≤ 1 → ∃ x : ℝ, f x = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_and_range_l88_8801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_power_theorem_l88_8892

/-- An infinite arithmetic progression of positive integers. -/
structure ArithmeticProgression where
  first : ℕ+  -- First term
  diff : ℕ+   -- Common difference
  seq : ℕ → ℕ+
  seq_def : ∀ n, seq n = first + n * diff

/-- A number is a perfect power with exponent k -/
def IsPerfectPower (n : ℕ+) (k : ℕ+) : Prop :=
  ∃ m : ℕ+, n = m ^ k.val

theorem arithmetic_progression_power_theorem (ap : ArithmeticProgression) :
  (∃ n : ℕ, IsPerfectPower (ap.seq n) 2) →  -- Contains a square
  (∃ m : ℕ, IsPerfectPower (ap.seq m) 3) →  -- Contains a cube
  (∃ k : ℕ, IsPerfectPower (ap.seq k) 6) :=  -- Contains a sixth power
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_power_theorem_l88_8892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_l88_8847

theorem sine_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < π) (h₃ : 0 < x₂) (h₄ : x₂ < π) (h₅ : x₁ ≠ x₂) :
  (Real.sin x₁ + Real.sin x₂) / 2 < Real.sin ((x₁ + x₂) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_l88_8847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_max_l88_8869

noncomputable def f (x : ℝ) := Real.sin x * Real.cos x - Real.sqrt 3 * Real.cos (x + Real.pi) * Real.cos x

noncomputable def g (x : ℝ) := f (x - Real.pi / 4) + Real.sqrt 3 / 2

theorem f_period_and_g_max :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∃ (M : ℝ), M = (3 * Real.sqrt 3) / 2 ∧
    (∀ (x : ℝ), x ≥ 0 → x ≤ Real.pi / 4 → g x ≤ M) ∧
    (∃ (x : ℝ), x ≥ 0 ∧ x ≤ Real.pi / 4 ∧ g x = M)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_max_l88_8869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_time_difference_l88_8884

/-- Represents the walking data for a single day -/
structure DayWalk where
  distance : ℚ
  speed : ℚ

/-- Calculates the time taken to walk given distance at given speed -/
def walkTime (d : DayWalk) : ℚ := d.distance / d.speed

/-- Calculates the total walking time for multiple days -/
def totalWalkTime (days : List DayWalk) : ℚ :=
  days.map walkTime |>.sum

/-- Calculates the total distance walked over multiple days -/
def totalDistance (days : List DayWalk) : ℚ :=
  days.map (fun d => d.distance) |>.sum

theorem walking_time_difference : 
  let actual_walks := [
    DayWalk.mk 3 6,  -- Monday
    DayWalk.mk 4 4,  -- Wednesday
    DayWalk.mk 5 5   -- Friday
  ]
  let constant_speed := 5
  let time_diff := totalWalkTime actual_walks - totalDistance actual_walks / constant_speed
  time_diff * 60 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_time_difference_l88_8884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_x_value_l88_8852

/-- Represents the state of the program at each iteration -/
structure ProgramState where
  x : ℕ
  S : ℕ

/-- Updates the program state according to the given rules -/
def updateState (state : ProgramState) : ProgramState :=
  let newX := state.x + 2
  { x := newX, S := state.S + newX }

/-- Calculates the final state of the program -/
def finalState : ProgramState :=
  let rec loop (state : ProgramState) (fuel : ℕ) : ProgramState :=
    if fuel = 0 then state
    else if state.S ≥ 10000 then state
    else loop (updateState state) (fuel - 1)
  loop { x := 3, S := 0 } 1000  -- Use a sufficiently large fuel value

/-- The theorem stating the final value of x -/
theorem final_x_value : finalState.x = 201 := by
  sorry

#eval finalState.x  -- This will print the actual result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_x_value_l88_8852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_theorem_l88_8874

/-- Represents a quadrilateral with side lengths and an angle sum property -/
structure Quadrilateral where
  PI : ℝ
  IN : ℝ
  NE : ℝ
  EP : ℝ
  angle_sum : ℝ

/-- The area of a quadrilateral with the given properties -/
noncomputable def quadrilateral_area (q : Quadrilateral) : ℝ :=
  (100 * Real.sqrt 3) / 3

/-- Theorem stating that a quadrilateral with the given properties has the specified area -/
theorem quadrilateral_area_theorem (q : Quadrilateral) 
  (h1 : q.PI = 6)
  (h2 : q.IN = 15)
  (h3 : q.NE = 6)
  (h4 : q.EP = 25)
  (h5 : q.angle_sum = 60) :
  quadrilateral_area q = (100 * Real.sqrt 3) / 3 := by
  sorry

#check quadrilateral_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_theorem_l88_8874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abby_in_seat_three_l88_8806

-- Define the set of people
inductive Person : Type
| Abby : Person
| Bret : Person
| Carl : Person
| Dana : Person
| Erin : Person

-- Define the seating arrangement
def Seat := Fin 5

-- Define the seating function
variable (seating : Person → Seat)

-- Bret is in seat #4
axiom bret_seat : seating Person.Bret = ⟨3, by norm_num⟩

-- Bret is not next to Dana
axiom bret_not_next_to_dana :
  ∀ (s : Seat), seating Person.Dana ≠ s → 
  (seating Person.Bret).val ≠ s.val + 1 ∧ 
  (seating Person.Bret).val ≠ s.val - 1

-- Erin is not between Bret and Carl
axiom erin_not_between_bret_carl :
  (seating Person.Erin).val < (seating Person.Bret).val ∨
  (seating Person.Erin).val > (seating Person.Carl).val

-- Each person sits in a unique seat
axiom unique_seats : ∀ (p1 p2 : Person), p1 ≠ p2 → seating p1 ≠ seating p2

-- Theorem to prove
theorem abby_in_seat_three : seating Person.Abby = ⟨2, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abby_in_seat_three_l88_8806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_R_in_special_triangle_l88_8828

/-- Given a triangle PQR with specific cotangent relationships, prove that tan R equals 4 + √7 -/
theorem tan_R_in_special_triangle (P Q R : ℝ) (h_triangle : P + Q + R = Real.pi) 
  (h_cot_PR : Real.tan P⁻¹ * Real.tan R⁻¹ = 1) 
  (h_cot_QR : Real.tan Q⁻¹ * Real.tan R⁻¹ = 1/8) : 
  Real.tan R = 4 + Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_R_in_special_triangle_l88_8828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_is_correct_l88_8870

/-- The smallest positive integer b for which x^2 + bx + 1764 factors into two integer binomials -/
def smallest_b : ℕ := 84

/-- Predicate to check if a quadratic expression factors into two integer binomials -/
def factors_into_integer_binomials (b : ℤ) : Prop :=
  ∃ (r s : ℤ), ∀ (x : ℤ), x^2 + b*x + 1764 = (x + r) * (x + s)

theorem smallest_b_is_correct :
  (∀ b : ℕ, b < smallest_b → ¬(factors_into_integer_binomials (b : ℤ))) ∧
  factors_into_integer_binomials smallest_b :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_is_correct_l88_8870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_in_cube_l88_8868

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Theorem: Light path length in a cube -/
theorem light_path_length_in_cube (cube : Cube) 
  (A : Point3D) (P : Point3D) (C : Point3D) :
  cube.sideLength = 10 →
  P.x = 3 →
  P.y = 6 →
  P.z = 10 →
  A.x = 0 →
  A.y = 0 →
  A.z = 0 →
  C.x = 10 →
  C.y = 10 →
  C.z = 10 →
  distance A P + distance P C = 2 * Real.sqrt 145 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_in_cube_l88_8868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_roots_product_l88_8816

theorem trigonometric_roots_product (α β a b c d : ℝ) : 
  (∃ x y : ℝ, x^2 - a*x + b = 0 ∧ y^2 - a*y + b = 0 ∧ x = Real.sin α ^ 2 ∧ y = Real.sin β ^ 2) →
  (∃ x y : ℝ, x^2 - c*x + d = 0 ∧ y^2 - c*y + d = 0 ∧ x = Real.cos α ^ 2 ∧ y = Real.cos β ^ 2) →
  c * d = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_roots_product_l88_8816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l88_8873

-- Define the ellipse C
def Ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def Line (k m x y : ℝ) : Prop := y = k*x + m

-- Define the condition for points A and B
def ConditionAB (xA yA xB yB : ℝ) : Prop :=
  (xA^2 + yA^2 + 4*xB^2 + 4*yB^2 + 4*xA*xB + 4*yA*yB) =
  (xA^2 + yA^2 + 4*xB^2 + 4*yB^2 - 4*xA*xB - 4*yA*yB)

-- Main theorem
theorem ellipse_intersection_theorem :
  ∃ (k m : ℝ), ∀ (xA yA xB yB : ℝ),
    Ellipse xA yA ∧ Ellipse xB yB ∧
    Line k m xA yA ∧ Line k m xB yB ∧
    ConditionAB xA yA xB yB →
    m ≤ -2*(21:ℝ).sqrt/7 ∨ m ≥ 2*(21:ℝ).sqrt/7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l88_8873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_youseff_distance_l88_8841

/-- The number of blocks between Youseff's home and office -/
noncomputable def x : ℝ := sorry

/-- The time it takes Youseff to walk to work -/
noncomputable def walkingTime : ℝ := x

/-- The time it takes Youseff to bike to work -/
noncomputable def bikingTime : ℝ := (20/60) * x

/-- The relationship between walking time and biking time -/
axiom time_difference : walkingTime = bikingTime + 4

theorem youseff_distance : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_youseff_distance_l88_8841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l88_8803

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola with equation x²/a² - y² = 1 -/
structure Hyperbola where
  a : ℝ
  h : a > 0

/-- Checks if a point is on the upper branch of the hyperbola -/
def isOnUpperBranch (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 = 1 ∧ p.y ≥ 0

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Checks if two points are the foci of the hyperbola -/
def isFocus (h : Hyperbola) (f1 f2 : Point) : Prop := sorry

/-- Calculates the area of a triangle given its three vertices -/
noncomputable def areaTriangle (p1 p2 p3 : Point) : ℝ := sorry

/-- Theorem: Area of triangle F₁PF₂ is √3/3 -/
theorem area_of_triangle (h : Hyperbola) (p f1 f2 : Point)
    (h_upper : isOnUpperBranch h p)
    (h_angle : angle f1 p f2 = 2 * Real.pi / 3)
    (h_foci : isFocus h f1 f2) : 
    areaTriangle f1 p f2 = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l88_8803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_equation_and_min_distance_point_l88_8818

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 4*x ∧ x ≥ 0

-- Define the distance function from a point to F(1,0)
noncomputable def dist_to_F (x y : ℝ) : ℝ := Real.sqrt ((x - 1)^2 + y^2)

-- Define the line x+y+4=0
def line (x y : ℝ) : Prop := x + y + 4 = 0

-- Define the distance function from a point to the line x+y+4=0
noncomputable def dist_to_line (x y : ℝ) : ℝ := |x + y + 4| / Real.sqrt 2

theorem curve_C_equation_and_min_distance_point :
  (∀ x y : ℝ, curve_C x y → x + 1 = dist_to_F x y) →
  (∀ x y : ℝ, curve_C x y → 
    ∀ x' y' : ℝ, curve_C x' y' → dist_to_line x y ≤ dist_to_line x' y') →
  (curve_C 1 (-2) ∧ 
   ∀ x y : ℝ, curve_C x y → y^2 = 4*x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_equation_and_min_distance_point_l88_8818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_represents_basketball_quantity_l88_8889

/-- Represents the quantity of basketballs -/
def x : ℕ := sorry

/-- Represents the quantity of soccer balls -/
def soccer_balls : ℕ := 2 * x

/-- Represents the total cost of soccer balls in yuan -/
def soccer_cost : ℕ := 5000

/-- Represents the total cost of basketballs in yuan -/
def basketball_cost : ℕ := 4000

/-- Represents the difference in unit price between basketballs and soccer balls in yuan -/
def price_difference : ℕ := 30

/-- Theorem stating that x represents the quantity of basketballs -/
theorem x_represents_basketball_quantity :
  (soccer_cost : ℚ) / soccer_balls = (basketball_cost : ℚ) / x - price_difference :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_represents_basketball_quantity_l88_8889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l88_8887

theorem problem_solution : 
  (-1)^2022 + |1 - Real.sqrt 2| + Real.sqrt 4 - (8 : ℝ)^(1/3) = Real.sqrt 2 - 2 ∧
  (Real.sqrt 5 + Real.sqrt 3) * (Real.sqrt 5 - Real.sqrt 3) = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l88_8887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_63_l88_8813

-- Define the vertices of the rhombus
def v1 : ℝ × ℝ := (0, 3.5)
def v2 : ℝ × ℝ := (9, 0)
def v3 : ℝ × ℝ := (0, -3.5)
def v4 : ℝ × ℝ := (-9, 0)

-- Define the rhombus area calculation function
noncomputable def rhombusArea (a b c d : ℝ × ℝ) : ℝ :=
  let d1 := |a.2 - c.2|
  let d2 := |b.1 - d.1|
  (d1 * d2) / 2

-- Theorem statement
theorem rhombus_area_is_63 :
  rhombusArea v1 v2 v3 v4 = 63 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_63_l88_8813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_eq_neg_sin_l88_8823

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => Real.sin
  | (n + 1) => deriv (f n)

-- State the theorem
theorem f_2010_eq_neg_sin : f 2010 = fun x => -Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_eq_neg_sin_l88_8823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_inscribed_angles_24_arcs_l88_8871

/-- Represents an inscribed angle in a circle -/
structure InscribedAngle where
  arcs : ℕ  -- number of arcs spanned by the angle

/-- Calculates the measure of an inscribed angle in degrees -/
noncomputable def inscribedAngleMeasure (angle : InscribedAngle) (totalArcs : ℕ) : ℝ :=
  (360 : ℝ) * (angle.arcs : ℝ) / (2 * totalArcs : ℝ)

theorem sum_of_inscribed_angles_24_arcs (x y : InscribedAngle) 
    (hx : x.arcs = 4) (hy : y.arcs = 6) : 
    inscribedAngleMeasure x 24 + inscribedAngleMeasure y 24 = 75 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_inscribed_angles_24_arcs_l88_8871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cotangent_inequality_l88_8866

theorem triangle_cotangent_inequality (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 → A + B + C = π →
  (1 / Real.tan (A / 2)) * (1 / Real.tan (B / 2)) * (1 / Real.tan (C / 2)) ≥ 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cotangent_inequality_l88_8866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l88_8812

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

theorem smallest_positive_period (p : ℝ) :
  (∀ x, f (x + p) = f x) ∧ 
  (∀ q, 0 < q → q < p → ∃ x, f (x + q) ≠ f x) →
  p = Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l88_8812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_average_difference_l88_8809

noncomputable def janet_semester1_grades : List ℚ := [90, 80, 70, 100]
def janet_semester2_average : ℚ := 82

noncomputable def calculate_average (grades : List ℚ) : ℚ :=
  (grades.sum) / grades.length

theorem janet_average_difference :
  calculate_average janet_semester1_grades - janet_semester2_average = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_average_difference_l88_8809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_good_2023_not_good_2022_l88_8804

def is_good (k n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a^3 + b^k = c^3 + d^k ∧ a * b * c * d = n

def x : ℕ := 3^1011

theorem infinitely_many_good_2023_not_good_2022 :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ n ∈ S, ∃ p : ℕ, Nat.Prime p ∧
      n = 3 * x * p^4052 * (9 * x^3 + 1) * (3 * x^3 + 1) ∧
      is_good 2023 n ∧
      ¬is_good 2022 n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_good_2023_not_good_2022_l88_8804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_3_minutes_l88_8860

/-- The distance between two vehicles moving at different speeds after a given time -/
noncomputable def distance_between_vehicles (v1 v2 t : ℝ) : ℝ :=
  (v2 - v1) * t

/-- Conversion from minutes to hours -/
noncomputable def minutes_to_hours (m : ℝ) : ℝ :=
  m / 60

theorem distance_after_3_minutes (truck_speed car_speed : ℝ) 
  (h1 : truck_speed = 65)
  (h2 : car_speed = 85) :
  distance_between_vehicles truck_speed car_speed (minutes_to_hours 3) = 1 := by
  sorry

#check distance_after_3_minutes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_3_minutes_l88_8860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vicious_circle_fallacy_l88_8872

/-- Represents a logical proposition --/
inductive Proposition where
  | Atom : String → Proposition
  | And : Proposition → Proposition → Proposition
  | Or : Proposition → Proposition → Proposition
  | Implies : Proposition → Proposition → Proposition
  deriving Repr, DecidableEq

/-- Represents a logical argument --/
structure Argument where
  premises : List Proposition
  conclusion : Proposition

/-- Checks if a proposition is contained within a list of propositions --/
def containsProposition (props : List Proposition) (p : Proposition) : Prop :=
  p ∈ props

/-- Defines a vicious circle (circulus vitiosus) in an argument --/
def isViciousCircle (arg : Argument) : Prop :=
  containsProposition arg.premises arg.conclusion

/-- Validity of an argument --/
def Valid (arg : Argument) : Prop :=
  ∀ (interpretation : Proposition → Bool),
    (arg.premises.all (fun p => interpretation p = true)) →
    (interpretation arg.conclusion = true)

/-- Theorem stating that an argument with a vicious circle is fallacious --/
theorem vicious_circle_fallacy (arg : Argument) :
  isViciousCircle arg → ¬ Valid arg :=
by
  intro h
  intro validArg
  -- The proof is omitted for brevity
  sorry

#check vicious_circle_fallacy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vicious_circle_fallacy_l88_8872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_specific_l88_8888

theorem sin_double_angle_specific (α : Real) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.sin α = 12 / 13) :
  Real.sin (2 * α) = -120 / 169 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_specific_l88_8888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_height_30_feet_l88_8805

/-- The height of a projectile as a function of time -/
def projectile_height (t : ℝ) : ℝ := 60 - 9*t - 5*t^2

/-- Theorem stating that the projectile reaches a height of 30 feet at time 6/5 seconds -/
theorem projectile_height_30_feet :
  ∃ t : ℝ, projectile_height t = 30 ∧ t = 6/5 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_height_30_feet_l88_8805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vaccination_gender_independence_l88_8834

-- Define the sample data
def sample_size : Nat := 500
def not_vaccinated_males : Nat := 20
def not_vaccinated_females : Nat := 10
def vaccinated_males : Nat := 230
def vaccinated_females : Nat := 240

-- Define the K^2 formula
noncomputable def k_squared (a b c d n : Nat) : ℝ :=
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99% confidence
def critical_value : ℝ := 6.635

-- Theorem statement
theorem vaccination_gender_independence : 
  k_squared not_vaccinated_males not_vaccinated_females vaccinated_males vaccinated_females sample_size < critical_value := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vaccination_gender_independence_l88_8834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_three_zeros_l88_8865

-- Define the piecewise function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then 2^x - m else -x^2 - 2*m*x

-- Define function g
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m x - m

-- State the theorem
theorem m_range_for_three_zeros (m : ℝ) :
  (∃! (z₁ z₂ z₃ : ℝ), g m z₁ = 0 ∧ g m z₂ = 0 ∧ g m z₃ = 0 ∧ z₁ ≠ z₂ ∧ z₂ ≠ z₃ ∧ z₁ ≠ z₃) →
  m > 1 ∧ ∀ k > 1, ∃ (z₁ z₂ z₃ : ℝ), g k z₁ = 0 ∧ g k z₂ = 0 ∧ g k z₃ = 0 ∧ z₁ ≠ z₂ ∧ z₂ ≠ z₃ ∧ z₁ ≠ z₃ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_three_zeros_l88_8865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l88_8859

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * (a^x - a^(-x))

-- State the theorem
theorem f_properties (a : ℝ) (h : 0 < a ∧ a < 1) :
  -- 1. f is odd
  (∀ x : ℝ, f a (-x) = -f a x) ∧
  -- 2. f is increasing
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) ∧
  -- 3. Equivalence of the inequality and range of a
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1/2 → f a (2*a*t^2 - a^2 - a) + f a (6*a*t - 1) ≤ 0) ↔
  (0 < a ∧ a ≤ 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l88_8859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_equation_l88_8853

theorem tan_triple_angle_equation (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π/6) 
  (h3 : Real.tan θ + Real.tan (2*θ) + Real.tan (4*θ) = 0) : 
  Real.tan θ = Real.sqrt ((7 - Real.sqrt 13) / 6) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_equation_l88_8853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whitewashing_cost_theorem_l88_8849

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a rectangular opening (door or window) -/
structure OpeningDimensions where
  width : ℝ
  height : ℝ

/-- Calculate the total cost of white washing a room -/
def whitewashingCost (room : RoomDimensions) (costPerSqFt : ℝ) (door : OpeningDimensions) (window : OpeningDimensions) (numWindows : ℕ) : ℝ :=
  let wallArea := 2 * (room.length * room.height + room.width * room.height)
  let doorArea := door.width * door.height
  let windowArea := window.width * window.height * (numWindows : ℝ)
  let netArea := wallArea - doorArea - windowArea
  netArea * costPerSqFt

/-- The main theorem stating the cost of white washing the room -/
theorem whitewashing_cost_theorem (room : RoomDimensions) (costPerSqFt : ℝ) (door : OpeningDimensions) (window : OpeningDimensions) (numWindows : ℕ) :
  room.length = 25 ∧ room.width = 15 ∧ room.height = 12 ∧
  costPerSqFt = 3 ∧
  door.width = 6 ∧ door.height = 3 ∧
  window.width = 4 ∧ window.height = 3 ∧
  numWindows = 3 →
  whitewashingCost room costPerSqFt door window numWindows = 2718 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_whitewashing_cost_theorem_l88_8849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_50_mod_7_l88_8894

/-- Sequence T defined recursively -/
def T : ℕ → ℕ
  | 0 => 9  -- Add base case for 0
  | n + 1 => 9^(T n)

/-- The 50th term of sequence T is congruent to 4 modulo 7 -/
theorem T_50_mod_7 : T 50 ≡ 4 [MOD 7] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_50_mod_7_l88_8894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l88_8845

noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n - 1)

noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_ratio 
  (a₁ : ℝ) (q : ℝ) (hq : q ≠ 0) (ha₁ : a₁ ≠ 0) :
  (geometric_sequence a₁ q 3 = 2 * geometric_sum a₁ q 2 + 1) →
  (geometric_sequence a₁ q 4 = 2 * geometric_sum a₁ q 3 + 1) →
  q = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l88_8845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_marked_cells_for_full_subgrid_l88_8808

/-- Represents a 10x10 grid where some cells are marked -/
def Grid := Fin 10 → Fin 10 → Bool

/-- Returns true if the 2x2 subgrid starting at (i,j) is fully marked -/
def isFullSubgrid (g : Grid) (i j : Fin 10) : Prop :=
  g i j ∧ g i (j.succ) ∧ g (i.succ) j ∧ g (i.succ) (j.succ)

/-- Returns true if there exists a fully marked 2x2 subgrid in the grid -/
def existsFullSubgrid (g : Grid) : Prop :=
  ∃ i j, isFullSubgrid g i j

/-- Returns the number of marked cells in the grid -/
def countMarkedCells (g : Grid) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 10)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 10)) fun j =>
      if g i j then 1 else 0)

/-- The main theorem: 76 is the smallest number of marked cells that guarantees a full 2x2 subgrid -/
theorem smallest_marked_cells_for_full_subgrid :
  (∀ g : Grid, countMarkedCells g ≥ 76 → existsFullSubgrid g) ∧
  (∃ g : Grid, countMarkedCells g = 75 ∧ ¬existsFullSubgrid g) := by
  sorry

#check smallest_marked_cells_for_full_subgrid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_marked_cells_for_full_subgrid_l88_8808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l88_8877

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  sine_law : a / sin A = b / sin B
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*cos C

/-- Given equation in the triangle -/
def given_equation (t : Triangle) : Prop :=
  (2 * t.a - t.c) / t.b = cos t.C / cos t.B

/-- Definition of z -/
noncomputable def z (t : Triangle) : ℝ :=
  (cos (t.A - t.C) + 2) / (sin t.A + sin t.C)

/-- Main theorem -/
theorem triangle_problem (t : Triangle) (h : given_equation t) :
  t.B = π/3 ∧ ∃ (l u : ℝ), l = 2*Real.sqrt 6/3 ∧ u = Real.sqrt 3 ∧ ∀ x, z t = x → l ≤ x ∧ x ≤ u := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l88_8877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_upper_bound_l88_8800

theorem inequality_implies_upper_bound (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → a ≤ -x^2 + 2*x) →
  a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_upper_bound_l88_8800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_l88_8844

theorem junior_score (total_students : ℕ) (junior_percentage senior_percentage : ℚ)
  (class_average senior_average : ℚ) (h1 : junior_percentage = 1/5)
  (h2 : senior_percentage = 4/5) (h3 : junior_percentage + senior_percentage = 1)
  (h4 : class_average = 82) (h5 : senior_average = 80) :
  let junior_count := (junior_percentage * total_students).floor
  let senior_count := (senior_percentage * total_students).floor
  let total_score := class_average * total_students
  let senior_total_score := senior_average * senior_count
  let junior_total_score := total_score - senior_total_score
  junior_total_score / junior_count = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_l88_8844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l88_8821

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0

-- Define the set of a values
def A : Set ℝ := {a | a ∈ Set.Icc (-2/3) 0 ∨ a ∈ Set.Iic (-4)}

-- State the theorem
theorem range_of_a :
  (∀ x a, p x a → q x) ∧
  (∃ x a, q x ∧ ¬(p x a)) ∧
  (∀ a, a ∈ A ↔ (∃ x, p x a) ∧ a < 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l88_8821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nut_division_l88_8817

theorem nut_division (seq : List ℤ) (h1 : seq.length = 64) 
  (h2 : ∀ i, i < seq.length - 1 → |seq.get! (i+1) - seq.get! i| = 1) :
  ∃ (subseq1 subseq2 : List ℤ),
    subseq1.length = subseq2.length ∧
    subseq1.sum = subseq2.sum ∧
    seq = subseq1 ++ subseq2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nut_division_l88_8817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_difference_l88_8899

/-- Given an arithmetic sequence with first term 4 and common difference 6,
    the positive difference between the 150th and 155th terms is 30. -/
theorem arithmetic_sequence_difference :
  let a₁ : ℕ := 4  -- first term
  let d : ℕ := 6   -- common difference
  (a₁ + 154 * d) - (a₁ + 149 * d) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_difference_l88_8899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_B_l88_8850

-- Define A and B as noncomputable
noncomputable def A : ℝ := Real.sqrt 5 + Real.sqrt (22 + 2 * Real.sqrt 5)
noncomputable def B : ℝ := Real.sqrt (11 + 2 * Real.sqrt 29) + Real.sqrt (16 - 2 * Real.sqrt 29 + 2 * Real.sqrt (55 - 10 * Real.sqrt 29))

-- Theorem statement
theorem A_equals_B : A = B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_B_l88_8850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_sum_is_96_l88_8831

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  /-- Dimensions of the solid -/
  a : ℝ
  r : ℝ
  /-- Volume is 216 cm³ -/
  volume_eq : a^3 = 216
  /-- Surface area is 288 cm² -/
  surface_area_eq : 2 * (a^2 / r + a^2 + a^2 * r) = 288
  /-- Dimensions are in geometric progression -/
  geometric_progression : r > 0

/-- The sum of the lengths of all edges of the rectangular solid -/
noncomputable def edge_sum (solid : RectangularSolid) : ℝ :=
  4 * (solid.a / solid.r + solid.a + solid.a * solid.r)

/-- Theorem: The sum of the lengths of all edges is 96 cm -/
theorem edge_sum_is_96 (solid : RectangularSolid) : edge_sum solid = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_sum_is_96_l88_8831
