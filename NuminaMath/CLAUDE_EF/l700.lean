import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_points_inequality_l700_70035

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2

theorem critical_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 ∧ x₂ > 0 ∧ x₂ > 2 * x₁ ∧
  (∀ x > 0, (deriv (f a)) x = 0 → x = x₁ ∨ x = x₂) →
  x₁^2 * x₂^3 > Real.sqrt (Real.exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_points_inequality_l700_70035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l700_70044

theorem problem_statement (p q : ℕ) 
  (hp : p > 0)
  (hq : q > 0)
  (h1 : p + 5 < q) 
  (h2 : (p + (p + 2) + (p + 5) + q + (q + 1) + (2*q - 1)) / 6 = q) 
  (h3 : (p + 5 + q) / 2 = q) : 
  p + q = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l700_70044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_sum_l700_70059

theorem min_value_exponential_sum (x y : ℝ) (h : x + 2*y = 4) :
  ∃ (min_val : ℝ), min_val = 8 ∧ ∀ (a b : ℝ), a + 2*b = 4 → (2:ℝ)^x + (4:ℝ)^y ≤ (2:ℝ)^a + (4:ℝ)^b :=
by
  sorry

#check min_value_exponential_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_sum_l700_70059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_members_age_l700_70066

theorem new_members_age (initial_size : ℕ) (initial_avg : ℚ) (years_passed : ℕ) (new_size : ℕ) :
  initial_size = 6 →
  initial_avg = 19 →
  years_passed = 3 →
  new_size = 8 →
  (initial_size * initial_avg + initial_size * (years_passed : ℚ)) / (new_size : ℚ) = initial_avg →
  new_size * initial_avg - (initial_size * initial_avg + initial_size * (years_passed : ℚ)) = 20 := by
  sorry

#check new_members_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_members_age_l700_70066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_intersections_is_100_l700_70061

/-- Represents a regular polygon inscribed in a circle -/
structure RegularPolygon where
  sides : ℕ

/-- Calculates the number of intersections between two polygons -/
def intersections (p1 p2 : RegularPolygon) : ℕ :=
  2 * min p1.sides p2.sides

/-- The set of regular polygons inscribed in the circle -/
def polygons : List RegularPolygon :=
  [⟨7⟩, ⟨9⟩, ⟨11⟩, ⟨13⟩]

/-- Theorem: The total number of intersections is 100 -/
theorem total_intersections_is_100 :
  (List.sum (do
    let p1 ← polygons
    let p2 ← polygons
    guard (p1.sides < p2.sides)
    pure (intersections p1 p2))) = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_intersections_is_100_l700_70061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candles_fit_on_cake_l700_70065

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a point is inside or on the circle -/
def isInside (c : Circle) (p : Point) : Prop :=
  distance c.center p ≤ c.radius

theorem candles_fit_on_cake (cake_diameter : ℝ) (num_candles : ℕ) (min_distance : ℝ)
    (h_diameter : cake_diameter = 36)
    (h_num_candles : num_candles = 13)
    (h_min_distance : min_distance = 10) :
    ∃ (points : List Point) (cake : Circle),
      cake.radius = cake_diameter / 2 ∧
      points.length = num_candles ∧
      (∀ p, p ∈ points → isInside cake p) ∧
      (∀ p1 p2, p1 ∈ points → p2 ∈ points → p1 ≠ p2 → distance p1 p2 ≥ min_distance) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candles_fit_on_cake_l700_70065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l700_70009

def sequence_a : ℕ → ℚ
| 0 => 2  -- Define the base case for 0 (representing the first element)
| 1 => 2  -- Keep the original base case for 1
| (n + 2) => 1 / (1 - sequence_a (n + 1))

theorem a_5_value : sequence_a 5 = -1 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l700_70009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_coordinates_l700_70063

noncomputable def A : ℝ × ℝ := (3, -4)
noncomputable def B : ℝ × ℝ := (-9, 2)

noncomputable def vector_length (v : ℝ × ℝ) : ℝ :=
  Real.sqrt ((v.1)^2 + (v.2)^2)

def line_segment (A B P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * B.1 + (1 - t) * A.1, t * B.2 + (1 - t) * A.2)

theorem point_P_coordinates :
  ∀ P : ℝ × ℝ,
  line_segment A B P →
  vector_length (P.1 - A.1, P.2 - A.2) = (1/3) * vector_length (B.1 - A.1, B.2 - A.2) →
  P = (-1, -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_coordinates_l700_70063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l700_70008

theorem sequence_sum_theorem : ∃ n : ℕ, 
  (-2 : ℤ)^n + ((-2 : ℤ)^n + 2) + (1/2 * (-2 : ℤ)^n) = -1278 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l700_70008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_equals_6_sqrt_3_l700_70041

-- Define a right triangle with an acute angle β
structure RightTriangle where
  β : Real
  is_acute : 0 < β ∧ β < Real.pi / 2
  is_right_triangle : Real.tan (β / 2) = Real.sqrt 3

-- Define φ as the angle between the median and angle bisector
noncomputable def phi (t : RightTriangle) : Real :=
  -- The actual definition of phi is not provided in the problem,
  -- so we leave it as an opaque definition
  sorry

-- State the theorem
theorem tan_phi_equals_6_sqrt_3 (t : RightTriangle) :
  Real.tan (phi t) = 6 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_equals_6_sqrt_3_l700_70041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_identity_l700_70084

-- Define the left-hand side of the equation
noncomputable def lhs (x : ℝ) : ℝ := (6 * x + 1) / (6 * x^2 + 19 * x + 15)

-- Define the right-hand side of the equation
noncomputable def rhs (x : ℝ) : ℝ := -1 / (x + 3/4) + 2 / (x + 5/3)

-- Theorem statement
theorem fraction_identity : ∀ x : ℝ, 
  x ≠ -3/4 ∧ x ≠ -5/3 → lhs x = rhs x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_identity_l700_70084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l700_70021

noncomputable def star (a b : ℝ) : ℝ := Real.sqrt (a + b) / Real.sqrt (a - b)

theorem star_equation_solution (y : ℝ) (h : star y 15 = 5) : y = 65 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l700_70021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_unit_circle_l700_70074

theorem angle_on_unit_circle (α : ℝ) :
  -- Conditions
  (∃ P : ℝ × ℝ, P.1 = -3/5 ∧ P.2 = 4/5 ∧ P.1^2 + P.2^2 = 1) →
  -- Conclusions
  Real.sin α = 4/5 ∧ (Real.sin (2*α) + Real.cos (2*α) + 1) / (1 + Real.tan α) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_unit_circle_l700_70074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_calculation_l700_70047

theorem total_cost_calculation (sandwich_price soda_price : ℚ) 
  (sandwich_quantity soda_quantity : ℕ) 
  (h1 : sandwich_price = 149/100)
  (h2 : soda_price = 87/100)
  (h3 : sandwich_quantity = 2)
  (h4 : soda_quantity = 4)
  : sandwich_quantity * sandwich_price + soda_quantity * soda_price = 646/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_calculation_l700_70047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_y_cube_root_inverse_negative_l700_70086

theorem negative_y_cube_root_inverse_negative (y : ℝ) (h : y < 0) :
  -(y^(-(1/3 : ℝ))) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_y_cube_root_inverse_negative_l700_70086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_2_pow_2018_l700_70025

def last_digit (n : ℕ) : ℕ := n % 10

def power_of_two_last_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | _ => 8

theorem last_digit_of_2_pow_2018 :
  last_digit (2^2018) = power_of_two_last_digit 2018 :=
by
  sorry

#eval power_of_two_last_digit 2018

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_2_pow_2018_l700_70025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_solution_l700_70089

theorem sequence_solution (a : Fin 1375 → ℝ) 
  (h1 : ∀ n : Fin 1374, 2 * Real.sqrt (a n - (n.val : ℝ) + 1) ≥ a (n.succ) - (n.val : ℝ) + 1)
  (h2 : 2 * Real.sqrt (a ⟨1374, by norm_num⟩ - 1374) ≥ a ⟨0, by norm_num⟩ + 1) :
  ∀ n : Fin 1375, a n = n.val := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_solution_l700_70089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contemporary_probability_theorem_l700_70013

/-- The number of years in the birth period -/
def birth_period : ℕ := 400

/-- The lifespan of each historian -/
def lifespan : ℕ := 80

/-- The probability that two historians lived as contemporaries -/
def contemporary_probability : ℚ := 9 / 25

/-- 
Theorem: Given two historians born within a 400-year period, each living for 80 years, 
and with birth years uniformly distributed, the probability that they lived as 
contemporaries for any length of time is 9/25.
-/
theorem contemporary_probability_theorem :
  (birth_period * birth_period - 2 * (birth_period - lifespan) * (birth_period - lifespan) / 2 : ℚ) /
  (birth_period * birth_period) = contemporary_probability := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_contemporary_probability_theorem_l700_70013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l700_70076

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions of the problem
def satisfiesConditions (t : Triangle) : Prop :=
  t.b * (t.b - Real.sqrt 3 * t.c) = (t.a - t.c) * (t.a + t.c) ∧
  Real.pi / 2 < t.B ∧ t.B < Real.pi

-- Define the theorem
theorem triangle_theorem (t : Triangle) (h : satisfiesConditions t) :
  t.A = Real.pi / 6 ∧
  (t.a = 1 / 2 → -1 / 2 < t.b - Real.sqrt 3 * t.c ∧ t.b - Real.sqrt 3 * t.c < 1 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l700_70076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_cube_surface_area_l700_70018

theorem small_cube_surface_area
  (large_surface_area : ℝ)
  (num_small_cubes : ℕ)
  (h1 : large_surface_area = 600)
  (h2 : num_small_cubes = 125) :
  ∃ (small_surface_area : ℝ),
    small_surface_area = 24 ∧
    (large_surface_area / 6) ^ (3/2) = num_small_cubes * (small_surface_area / 6) ^ (3/2) := by
  sorry

#check small_cube_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_cube_surface_area_l700_70018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_OQ_l700_70038

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the parabola -/
def on_parabola (C : Parabola) (P : Point) : Prop :=
  P.y^2 = 2 * C.p * P.x

/-- Definition of the focus -/
def focus (C : Parabola) : Point :=
  ⟨C.p, 0⟩

/-- Definition of the relation between P and Q -/
def PQ_relation (C : Parabola) (P Q : Point) : Prop :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2 = 81 * ((Q.x - C.p)^2 + Q.y^2)

/-- Slope of line OQ -/
noncomputable def slope_OQ (Q : Point) : ℝ :=
  Q.y / Q.x

/-- The main theorem -/
theorem max_slope_OQ (C : Parabola) :
  C.p = 2 →
  ∃ (max_slope : ℝ),
    max_slope = 1/3 ∧
    ∀ (P Q : Point),
      on_parabola C P →
      PQ_relation C P Q →
      |slope_OQ Q| ≤ max_slope := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_OQ_l700_70038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wendy_meal_cost_l700_70053

/-- Represents the cost of a meal for a group of friends -/
structure MealCost where
  num_friends : ℕ
  item1_cost : ℚ
  item1_quantity : ℕ
  item2_cost : ℚ
  item2_quantity : ℕ
  item3_cost : ℚ
  item3_quantity : ℕ
  item4_cost : ℚ
  item4_quantity : ℕ

/-- Calculates the cost per person when splitting the bill equally -/
def cost_per_person (meal : MealCost) : ℚ :=
  (meal.item1_cost * meal.item1_quantity +
   meal.item2_cost * meal.item2_quantity +
   meal.item3_cost * meal.item3_quantity +
   meal.item4_cost * meal.item4_quantity) / meal.num_friends

/-- Theorem stating that for the given meal, each person pays $11 -/
theorem wendy_meal_cost :
  let meal := MealCost.mk 5 10 1 5 5 (5/2) 4 2 5
  cost_per_person meal = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wendy_meal_cost_l700_70053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_complex_function_l700_70007

theorem range_of_complex_function (z : ℂ) :
  Complex.abs (z + 2 - Complex.I) = 1 →
  ∃ (w : ℝ), Complex.abs (2*z - 1) = w ∧ Real.sqrt 29 - 2 ≤ w ∧ w ≤ Real.sqrt 29 + 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_complex_function_l700_70007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_and_q_true_l700_70092

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x * y ≠ 6 → x ≠ 2 ∨ y ≠ 3

-- Define proposition q
def q : Prop := ∀ a x : ℝ, a ∈ Set.Ioc (-1) 5 → |2 - x| + |3 + x| ≥ a^2 - 4*a

-- Theorem to prove
theorem p_and_q_true : p ∧ q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_and_q_true_l700_70092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_coefficients_l700_70091

/-- A parabola with a vertical axis of symmetry -/
structure VerticalParabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The parabola passes through a given point -/
def passes_through (p : VerticalParabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- The vertex of the parabola -/
noncomputable def vertex (p : VerticalParabola) : ℝ × ℝ :=
  (- p.b / (2 * p.a), p.c - p.b^2 / (4 * p.a))

theorem parabola_coefficients :
  ∃ (p : VerticalParabola),
    vertex p = (5, -3) ∧
    passes_through p 3 7 ∧
    p.a = 5/2 ∧
    p.b = -25 ∧
    p.c = 119/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_coefficients_l700_70091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_inverse_squares_l700_70016

/-- A plane intersecting the coordinate axes -/
structure IntersectingPlane where
  /-- Distance from the origin to the plane -/
  distance : ℝ
  /-- x-coordinate of the intersection with the x-axis -/
  a : ℝ
  /-- y-coordinate of the intersection with the y-axis -/
  b : ℝ
  /-- z-coordinate of the intersection with the z-axis -/
  c : ℝ
  /-- The points are distinct from the origin -/
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0
  /-- The plane equation holds -/
  plane_eq : 1 / a + 1 / b + 1 / c = 1 / distance

/-- The centroid of a triangle formed by the intersections -/
noncomputable def centroid (plane : IntersectingPlane) : ℝ × ℝ × ℝ :=
  (plane.a / 3, plane.b / 3, plane.c / 3)

/-- The main theorem -/
theorem centroid_sum_inverse_squares (plane : IntersectingPlane) 
    (h : plane.distance = 2) : 
    let (p, q, r) := centroid plane
    1 / p^2 + 1 / q^2 + 1 / r^2 = 2.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_inverse_squares_l700_70016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_proof_l700_70057

-- Define the basic structures
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define the given conditions
def touches_internally (c1 c2 : Circle) (N : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def is_chord (c : Circle) (A B : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def touches_circle (c : Circle) (l : Set (EuclideanSpace ℝ (Fin 2))) (P : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def is_midpoint_of_arc (c : Circle) (M A B : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def on_circumcircle (A B C P : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def is_parallelogram (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- State the theorem
theorem parallelogram_proof 
  (outer inner : Circle) 
  (N A B C K M Q P B₁ : EuclideanSpace ℝ (Fin 2)) :
  touches_internally outer inner N →
  is_chord outer A B →
  is_chord outer B C →
  touches_circle inner {x : EuclideanSpace ℝ (Fin 2) | ∃ t, x = (1 - t) • A + t • B} K →
  touches_circle inner {x : EuclideanSpace ℝ (Fin 2) | ∃ t, x = (1 - t) • B + t • C} M →
  is_midpoint_of_arc outer Q A B →
  is_midpoint_of_arc outer P B C →
  on_circumcircle B Q K B₁ →
  on_circumcircle B P M B₁ →
  is_parallelogram B P B₁ Q := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_proof_l700_70057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_l700_70080

/-- Given that m is the normal vector of plane a and n is the direction vector of line l -/
def m (x : ℝ) : Fin 3 → ℝ := λ i => match i with
  | 0 => 2
  | 1 => -4*x
  | 2 => 1

def n (y : ℝ) : Fin 3 → ℝ := λ i => match i with
  | 0 => 6
  | 1 => 12
  | 2 => -3*y

/-- Line l is perpendicular to plane a -/
def perpendicular (x y : ℝ) : Prop := m x = n y

theorem parallel_vectors_sum (x y : ℝ) (h : perpendicular x y) : x + y = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_l700_70080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_implies_t_values_sufficient_condition_implies_m_range_l700_70075

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 * (Real.pi/4 + x) - Real.sqrt 3 * Real.cos (2*x) - 1

-- Define the function h
noncomputable def h (t : ℝ) (x : ℝ) : ℝ := f (x + t)

-- Theorem 1
theorem symmetric_point_implies_t_values (t : ℝ) :
  t ∈ Set.Ioo 0 Real.pi →
  (∀ x, h t x = h t (-Real.pi/3 - x)) →
  t = Real.pi/3 ∨ t = 5*Real.pi/6 := by
  sorry

-- Theorem 2
theorem sufficient_condition_implies_m_range (m : ℝ) :
  (∀ x, x ∈ Set.Icc (Real.pi/4) (Real.pi/2) → |f x - m| ≤ 3) →
  m ∈ Set.Icc (-1) 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_implies_t_values_sufficient_condition_implies_m_range_l700_70075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_factorial_to_add_l700_70033

/-- Count the number of trailing zeroes in a factorial -/
def trailingZeroes (n : Nat) : Nat := sorry

/-- Check if a number's factorial can be added to 70! without increasing trailing zeroes -/
def canAddWithoutIncreasingZeroes (k : Nat) : Prop :=
  trailingZeroes (Nat.factorial 70 + Nat.factorial k) = trailingZeroes (Nat.factorial 70)

theorem largest_factorial_to_add : 
  (∀ m : Nat, m > 4 → ¬(canAddWithoutIncreasingZeroes m)) ∧ 
  canAddWithoutIncreasingZeroes 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_factorial_to_add_l700_70033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_exponential_function_l700_70037

theorem tangent_line_exponential_function (a : ℝ) :
  (∃ x₀ : ℝ, (-(1/a) * Real.exp x₀ = -x₀ + 1) ∧ 
  (-(1/a) * Real.exp x₀ = deriv (λ x => -(1/a) * Real.exp x) x₀)) →
  a = Real.exp 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_exponential_function_l700_70037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_eleven_is_sixtysix_l700_70097

/-- An arithmetic sequence with the given property -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  property : a 9 = (1/2) * a 12 + 3

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The main theorem stating that S_11 = 66 for the given arithmetic sequence -/
theorem sum_eleven_is_sixtysix (seq : ArithmeticSequence) : sum_n seq 11 = 66 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_eleven_is_sixtysix_l700_70097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l700_70090

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (winner_percent : ℚ)
  (h_total : total_votes = 7500)
  (h_invalid : invalid_percent = 1/5)
  (h_winner : winner_percent = 11/20) :
  (↑total_votes - ↑total_votes * (invalid_percent : ℝ)) * (1 - (winner_percent : ℝ)) = 2700 :=
by
  -- Convert total_votes to ℝ
  have total_votes_real : ℝ := ↑total_votes

  -- Calculate valid votes
  let valid_votes : ℝ := total_votes_real * (1 - (invalid_percent : ℝ))

  -- Calculate other candidate's votes
  let other_votes : ℝ := valid_votes * (1 - (winner_percent : ℝ))

  -- Prove the equality
  calc
    (↑total_votes - ↑total_votes * (invalid_percent : ℝ)) * (1 - (winner_percent : ℝ))
      = valid_votes * (1 - (winner_percent : ℝ)) := by sorry
    _ = other_votes := by rfl
    _ = 2700 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l700_70090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_equals_729_l700_70070

def mySequence : List ℚ := [1/3, 9/1, 1/27, 81/1, 1/243, 729/1, 1/2187, 6561/1]

theorem sequence_product_equals_729 : 
  mySequence.prod = 729 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_equals_729_l700_70070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_tangent_implies_a_positive_l700_70056

/-- The function f(x) = ax^2 - ln(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - Real.log x

/-- The derivative of f(x) --/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2 * a * x - 1 / x

theorem vertical_tangent_implies_a_positive 
  (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ f_deriv a x = 0) → a > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_tangent_implies_a_positive_l700_70056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gate_perimeter_with_cutout_l700_70055

/-- Calculates the perimeter of a circular gate with a sector cutout -/
noncomputable def gatePerimeter (radius : ℝ) (cutoutAngle : ℝ) : ℝ :=
  let remainingArcLength := (2 * Real.pi - cutoutAngle / (2 * Real.pi)) * (2 * Real.pi * radius)
  let straightEdgesLength := 2 * radius
  remainingArcLength + straightEdgesLength

/-- Theorem: The perimeter of a circular gate with radius 2 cm and a 90° sector cutout is 3π + 4 cm -/
theorem gate_perimeter_with_cutout :
  gatePerimeter 2 (Real.pi / 2) = 3 * Real.pi + 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gate_perimeter_with_cutout_l700_70055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_squares_line_equation_no_three_equal_area_triangles_l700_70048

-- Define the ellipse
noncomputable def on_ellipse (x y : ℝ) : Prop := x^2 + y^2/2 = 1

-- Define the area of a triangle
noncomputable def triangle_area (x₁ y₁ x₂ y₂ : ℝ) : ℝ := abs (x₁ * y₂ - x₂ * y₁) / 2

-- Theorem statement
theorem constant_sum_of_squares (x₁ y₁ x₂ y₂ : ℝ) :
  on_ellipse x₁ y₁ → on_ellipse x₂ y₂ → 
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
  triangle_area x₁ y₁ x₂ y₂ = Real.sqrt 2 / 2 →
  x₁^2 + x₂^2 = 1 ∧ y₁^2 + y₂^2 = 2 := by
  sorry

-- Part 1: Equation of the line l
theorem line_equation :
  ∃ (m : ℝ), m^2 = 1/2 ∧ 
  (∀ x y : ℝ, on_ellipse x y → x = m ∨ x = -m) := by
  sorry

-- Part 3: Non-existence of points D, E, G
theorem no_three_equal_area_triangles :
  ¬∃ (u v u₁ v₁ u₂ v₂ : ℝ),
    on_ellipse u v ∧ on_ellipse u₁ v₁ ∧ on_ellipse u₂ v₂ ∧
    triangle_area u v u₁ v₁ = Real.sqrt 2 / 2 ∧
    triangle_area u v u₂ v₂ = Real.sqrt 2 / 2 ∧
    triangle_area u₁ v₁ u₂ v₂ = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_squares_line_equation_no_three_equal_area_triangles_l700_70048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_truths_l700_70071

open Real

theorem proposition_truths : ∃ (count : Nat), count = 2 ∧
  count = (Finset.filter (λ p => p = True) {
    (∀ x, sin x = 0 → cos x = 1),
    (∀ x, cos x = 1 → sin x = 0),
    (∀ x, sin x ≠ 0 → cos x ≠ 1),
    (∀ x, cos x ≠ 1 → sin x ≠ 0)
  }).card := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_truths_l700_70071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l700_70085

noncomputable def f (x : ℝ) : ℝ := -x + 2/x

theorem f_properties :
  (∀ x, x > 0 → f x = -x + 2/x) ∧
  f 1 = 1 ∧
  f 2 = -1 ∧
  (∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x ∈ Set.Icc (1/4 : ℝ) 1, 1 ≤ f x ∧ f x ≤ 31/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l700_70085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_quantity_problem_l700_70001

/-- Represents the initial quantity of milk in container A -/
def A : ℝ := sorry

/-- The quantity of milk in container B after initial pouring -/
def B : ℝ := 0.375 * A

/-- The quantity of milk in container C after initial pouring -/
def C : ℝ := 0.625 * A

/-- The amount transferred from C to B -/
def transfer : ℝ := 156

theorem milk_quantity_problem :
  (B + transfer = C - transfer) → A = 1248 := by
  intro h
  sorry

#check milk_quantity_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_quantity_problem_l700_70001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l700_70078

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, -1/2)

noncomputable def f (x : ℝ) : ℝ := ((a x).1 + (b x).1) * (a x).1 + ((a x).2 + (b x).2) * (a x).2 - 2

theorem triangle_area (A B C : ℝ) (hA : 0 < A ∧ A < Real.pi / 2) 
  (ha : Real.sqrt ((2 * Real.sqrt 3) ^ 2) = 2 * Real.sqrt 3) 
  (hc : C = 4) (hf : f A = 1) : 
  (1/2) * 4 * (Real.sqrt ((2 * Real.sqrt 3) ^ 2 + 4^2 - 2 * 2 * Real.sqrt 3 * 4 * Real.cos A)) * Real.sin A = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l700_70078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l700_70098

noncomputable def f (t : ℝ) : ℝ := (2^t - 3*t) * t / 4^t

theorem max_value_of_f :
  ∃ (t : ℝ), f t = 1/12 ∧ ∀ (s : ℝ), f s ≤ 1/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l700_70098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opponent_total_score_l700_70011

def TeamScores : List ℕ := [2, 3, 4, 5, 6, 8, 10, 12]

structure Game where
  team_score : ℕ
  opponent_score : ℕ

def is_loss_by_one (g : Game) : Bool :=
  g.opponent_score = g.team_score + 1

def is_triple_score (g : Game) : Bool :=
  g.team_score = 3 * g.opponent_score

theorem opponent_total_score : 
  ∃ (games : List Game), 
    (games.length = 8) ∧ 
    (games.map (λ g => g.team_score) = TeamScores) ∧
    (games.filter is_loss_by_one).length = 4 ∧
    (∀ g ∈ games, is_loss_by_one g = true ∨ is_triple_score g = true) ∧
    (games.map (λ g => g.opponent_score)).sum = 38 :=
by sorry

#check opponent_total_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opponent_total_score_l700_70011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_teaching_method_more_effective_l700_70069

-- Define the score ranges
inductive ScoreRange
| Range1 : ScoreRange -- 0 < x ≤ 5
| Range2 : ScoreRange -- 5 < x ≤ 10
| Range3 : ScoreRange -- 10 < x ≤ 15
| Range4 : ScoreRange -- 15 < x ≤ 20
| Range5 : ScoreRange -- 20 < x ≤ 25

-- Define the class data structure
structure ClassData where
  preTest : ScoreRange → Nat
  postTest : ScoreRange → Nat

-- Define the experiment data
def experimentData : (ClassData × ClassData) := sorry

-- Calculate the number of students in a class
def numStudents (data : ClassData) : Nat :=
  (data.preTest ScoreRange.Range1) + (data.preTest ScoreRange.Range2) +
  (data.preTest ScoreRange.Range3) + (data.preTest ScoreRange.Range4) +
  (data.preTest ScoreRange.Range5)

-- Calculate the mean score for a class
noncomputable def meanScore (data : ClassData) (isPostTest : Bool) : Real := sorry

-- Define the effectiveness measure
def isMoreEffective (controlClassPostTestMean controlClassPreTestMean : Real)
                    (experimentClassPostTestMean experimentClassPreTestMean : Real) : Prop :=
  (experimentClassPostTestMean - experimentClassPreTestMean) >
  (controlClassPostTestMean - controlClassPreTestMean)

-- Theorem statement
theorem new_teaching_method_more_effective :
  let (classA, classB) := experimentData
  let classAPreMean := meanScore classA false
  let classAPostMean := meanScore classA true
  let classBPreMean := meanScore classB false
  let classBPostMean := meanScore classB true
  isMoreEffective classAPostMean classAPreMean classBPostMean classBPreMean :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_teaching_method_more_effective_l700_70069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l700_70015

/-- The inclination angle of a line with equation x*tan(π/3) + y + 2 = 0 is 2π/3 -/
theorem line_inclination_angle (x y : ℝ) :
  x * Real.tan (π / 3) + y + 2 = 0 →
  Real.arctan (-Real.tan (π / 3)) = 2 * π / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l700_70015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mean_difference_l700_70093

/-- Represents the distribution of scores in a math test -/
structure ScoreDistribution where
  score60 : ℝ
  score75 : ℝ
  score85 : ℝ
  score95 : ℝ
  sum_to_one : score60 + score75 + score85 + score95 = 1

/-- Calculates the mean score given a score distribution -/
def mean (sd : ScoreDistribution) : ℝ :=
  60 * sd.score60 + 75 * sd.score75 + 85 * sd.score85 + 95 * sd.score95

/-- Determines the median score given a score distribution -/
noncomputable def median (sd : ScoreDistribution) : ℝ :=
  if sd.score60 + sd.score75 ≤ 0.5 ∧ sd.score60 + sd.score75 + sd.score85 > 0.5 then 85 else 0

/-- Theorem stating the difference between median and mean scores -/
theorem median_mean_difference (sd : ScoreDistribution) 
  (h1 : sd.score60 = 0.15)
  (h2 : sd.score75 = 0.20)
  (h3 : sd.score85 = 0.40) :
  median sd - mean sd = 3.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mean_difference_l700_70093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_and_optimal_cost_l700_70096

noncomputable def sales_volume (x : ℝ) : ℝ := 5 - 12 / (x + 3)

noncomputable def profit (x : ℝ) : ℝ := 20 - 4 / x - x

noncomputable def optimal_cost (a : ℝ) : ℝ :=
  if a ≥ 2 then 2 else a

theorem profit_and_optimal_cost (a : ℝ) (h : a > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ a → profit x = 20 - 4 / x - x) ∧
  (optimal_cost a = if a ≥ 2 then 2 else a) ∧
  (∀ x, 0 ≤ x ∧ x ≤ a → profit x ≤ profit (optimal_cost a)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_and_optimal_cost_l700_70096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l700_70020

/-- Calculates the length of a platform given train length and crossing times -/
noncomputable def platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) : ℝ :=
  let train_speed := train_length / pole_time
  train_speed * platform_time - train_length

/-- Theorem stating the length of the platform given the problem conditions -/
theorem platform_length_calculation :
  let train_length : ℝ := 300
  let platform_time : ℝ := 48
  let pole_time : ℝ := 18
  ∃ ε > 0, |platform_length train_length platform_time pole_time - 500.16| < ε :=
by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions
-- #eval platform_length 300 48 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l700_70020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_equivalence_l700_70052

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem function_inequality_equivalence (m : ℝ) :
  (∃ x ∈ Set.Icc (Real.exp 1) (Real.exp 2), f x - m * x - 1/2 + m ≤ 0) ↔ m ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_equivalence_l700_70052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l700_70073

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a+2)*x + a * Real.log x

-- State the theorem
theorem intersection_range (a : ℝ) (h : a > 0) :
  a = 4 →
  ∃ (m_min m_max : ℝ),
    m_min = 4 * Real.log 2 - 8 ∧
    m_max = -5 ∧
    ∀ m : ℝ, m_min < m ∧ m < m_max →
      ∃ (x₁ x₂ x₃ : ℝ),
        x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
        f a x₁ = m ∧ f a x₂ = m ∧ f a x₃ = m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l700_70073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tweedledee_exists_l700_70029

-- Define the characters
inductive Character : Type
| Tweedledum : Character
| Tweedledee : Character

-- Define a day type
inductive Day : Type
| TruthDay : Day
| LyingDay : Day

-- Define the statement made by the character
structure Statement where
  identity : Character → Prop
  day_type : Day → Prop

-- Define the conditions of the problem
axiom met_brother : ∃ (c : Character), True
axiom brother_statement : Statement
axiom identity_claim : brother_statement.identity = λ c ↦ c = Character.Tweedledum ∨ c = Character.Tweedledee
axiom lying_day_claim : brother_statement.day_type = λ d ↦ d = Day.LyingDay

-- The theorem to prove
theorem tweedledee_exists : ∃ (c : Character), c = Character.Tweedledee := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tweedledee_exists_l700_70029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_angle_specific_vectors_l700_70039

/-- The sine of the angle between two 2D vectors -/
noncomputable def sine_angle (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt (1 - (((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))^2))

/-- Theorem: The sine of the angle between vectors (-1, 1) and (3, -4) is √2/10 -/
theorem sine_angle_specific_vectors :
  sine_angle (-1, 1) (3, -4) = Real.sqrt 2 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_angle_specific_vectors_l700_70039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_bottles_count_l700_70046

/-- The number of plastic bottles needed to make one new bottle -/
def bottles_per_new : ℕ := 5

/-- The initial number of plastic bottles -/
def initial_bottles : ℕ := 3125

/-- The sum of a geometric series with first term a, common ratio r, and n terms -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The total number of new bottles that can be made -/
def total_new_bottles : ℕ :=
  (geometric_sum (initial_bottles / bottles_per_new : ℚ) (1 / bottles_per_new : ℚ) 5).floor.toNat

theorem new_bottles_count :
  total_new_bottles = 156 := by
  sorry

#eval total_new_bottles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_bottles_count_l700_70046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l700_70060

/-- The line represented by the equation x + y - a = 0 -/
def line (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 - a = 0}

/-- The circle represented by the equation (x+1)² + (y+1)² = 1 -/
def circle_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 + 1)^2 = 1}

/-- The condition for the line and circle to intersect at two points -/
def intersectAtTwoPoints (a : ℝ) : Prop :=
  ∃ p q : ℝ × ℝ, p ≠ q ∧ p ∈ line a ∧ p ∈ circle_set ∧ q ∈ line a ∧ q ∈ circle_set

/-- The theorem stating the range of 'a' for which the line and circle intersect at two points -/
theorem intersection_range :
  ∀ a : ℝ, intersectAtTwoPoints a ↔ -2 - Real.sqrt 2 < a ∧ a < -2 + Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l700_70060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l700_70082

/-- Parabola structure -/
structure Parabola where
  t : ℝ
  x : ℝ := 4 * t^2
  y : ℝ := 4 * t

/-- Line passing through focus and intersecting parabola -/
structure IntersectingLine where
  slope : ℝ
  passesThroghFocus : Prop
  intersectionPoints : Fin 2 → Parabola

/-- Theorem: Length of line segment AB on parabola -/
theorem parabola_intersection_length 
  (p : Parabola) 
  (l : IntersectingLine) 
  (h1 : l.slope = 1) 
  (h2 : l.passesThroghFocus) : 
  ∃ A B : Parabola, (A ∈ Set.range l.intersectionPoints) ∧ 
                    (B ∈ Set.range l.intersectionPoints) ∧ 
                    ‖(A.x, A.y) - (B.x, B.y)‖ = 8 := by
  sorry

#check parabola_intersection_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l700_70082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l700_70027

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3) / Real.log (1/2)

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := x > 3 ∨ x < -1

-- Theorem statement
theorem f_monotone_increasing :
  ∀ x₁ x₂ : ℝ, domain x₁ → domain x₂ → x₁ < x₂ → x₂ ≤ -1 → f x₁ < f x₂ :=
by
  sorry

#check f_monotone_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l700_70027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l700_70067

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2
  let y_line := 2 * b
  let right_focus := (Real.sqrt (a^2 + b^2), 0)
  let intersection_points := {p : ℝ × ℝ | f p.1 p.2 = 1 ∧ p.2 = y_line}
  ∃ B C : ℝ × ℝ, B ∈ intersection_points ∧ C ∈ intersection_points ∧
    (B.1 - right_focus.1) * (C.1 - right_focus.1) +
    (B.2 - right_focus.2) * (C.2 - right_focus.2) = 0 →
  Real.sqrt (a^2 + b^2) / a = 3 * Real.sqrt 5 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l700_70067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_implies_greater_l700_70081

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Add this case for 0
  | 1 => 2
  | (n + 2) => (sequence_a (n + 1))^3 - sequence_a (n + 1) + 1

theorem prime_divides_implies_greater
  (p n : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  p ∣ sequence_a n → p > n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_implies_greater_l700_70081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circles_distance_bounds_l700_70024

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

-- Define the left circle
def left_circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

-- Define the right circle
def right_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- State the theorem
theorem ellipse_circles_distance_bounds :
  ∀ (px py mx my nx ny : ℝ),
    ellipse px py →
    left_circle mx my →
    right_circle nx ny →
    4 ≤ distance px py mx my + distance px py nx ny ∧
    distance px py mx my + distance px py nx ny ≤ 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circles_distance_bounds_l700_70024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_O1O2_is_two_l700_70049

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the right triangle ABC with given side lengths
def rightTriangle : Triangle :=
  { A := (0, 4),
    B := (3, 0),
    C := (0, 0) }

-- Define the altitude CD
def altitudeCD : ℝ × ℝ :=
  (1.92, 1.44)

-- Define the incenter of a triangle
noncomputable def incenter (A B C : ℝ × ℝ) : ℝ × ℝ :=
  sorry  -- The actual calculation of the incenter is omitted

-- Define O1 as the incenter of ACD
noncomputable def O1 (t : Triangle) : ℝ × ℝ :=
  incenter t.A altitudeCD t.C

-- Define O2 as the incenter of BCD
noncomputable def O2 (t : Triangle) : ℝ × ℝ :=
  incenter t.B altitudeCD t.C

-- Calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem length_O1O2_is_two :
  distance (O1 rightTriangle) (O2 rightTriangle) = 2 := by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_O1O2_is_two_l700_70049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_increasing_l700_70006

/-- The inverse proportion function -/
noncomputable def inverse_proportion (m : ℝ) (x : ℝ) : ℝ := (m - 5) / x

/-- y increases as x increases -/
def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

/-- Theorem stating the condition for the inverse proportion function to be increasing -/
theorem inverse_proportion_increasing (m : ℝ) :
  increasing_function (inverse_proportion m) ↔ m < 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_increasing_l700_70006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clearing_time_approx_l700_70002

/-- The time (in seconds) for two trains to be completely clear of each other -/
noncomputable def clearingTime (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / ((speed1 + speed2) * 1000 / 3600)

/-- Theorem stating that the clearing time for the given trains is approximately 7.82 seconds -/
theorem train_clearing_time_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |clearingTime 150 165 80 65 - 7.82| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clearing_time_approx_l700_70002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_sales_amount_l700_70043

/-- Calculates the amount of first sales given profit information and ratio decrease --/
noncomputable def calculate_first_sales (profit_first : ℝ) (profit_next : ℝ) (sales_next : ℝ) (ratio_decrease : ℝ) : ℝ :=
  (profit_first * sales_next) / (profit_next * (1 - ratio_decrease / 100))

/-- Theorem stating the calculation of first sales amount --/
theorem first_sales_amount :
  let profit_first : ℝ := 125000
  let profit_next : ℝ := 80000
  let sales_next : ℝ := 2000000
  let ratio_decrease : ℝ := 200
  abs (calculate_first_sales profit_first profit_next sales_next ratio_decrease - 1041666.67) < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_sales_amount_l700_70043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_plane_unit_distance_l700_70058

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define the distance function between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Theorem statement
theorem two_color_plane_unit_distance :
  ∃ (p q : Point), coloring p = coloring q ∧ distance p q = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_plane_unit_distance_l700_70058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_adjustment_l700_70040

theorem salary_adjustment (S : ℝ) (h : S > 0) : 
  let reduced_salary := S * (1 - 0.14)
  let taxed_salary := reduced_salary * (1 - 0.28)
  let required_increase := (S / taxed_salary) - 1
  abs (required_increase - 0.6154) < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_adjustment_l700_70040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_picnic_recyclables_l700_70051

/-- Given the conditions of a summer picnic, proves that the total number of recyclable cans and bottles collected is 115. -/
theorem summer_picnic_recyclables (total_people : ℕ) (soda_cans : ℕ) (water_bottles : ℕ) (juice_bottles : ℕ)
  (h_total : total_people = 90)
  (h_soda : soda_cans = 50)
  (h_water : water_bottles = 50)
  (h_juice : juice_bottles = 50)
  (h_soda_drinkers : total_people / 2 = 45)
  (h_water_drinkers : total_people / 3 = 30)
  (h_juice_consumed : juice_bottles * 4 / 5 = 40) :
  45 + 30 + 40 = 115 := by
  -- The proof goes here
  sorry

#check summer_picnic_recyclables

end NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_picnic_recyclables_l700_70051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_not_determinable_l700_70017

/-- Represents a triangular prism with an equilateral triangle base -/
structure TriangularPrism where
  base_side : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  volume : ℝ

/-- Theorem stating the impossibility of expressing the prism volume solely in terms of the circumscribed sphere's volume -/
theorem prism_volume_not_determinable 
  (prism : TriangularPrism) 
  (sphere : Sphere) 
  (h1 : prism.height > 0)
  (h2 : prism.base_side > 0)
  (h3 : sphere.volume > 0)
  (h4 : ∃ (r : ℝ), sphere.volume = (4/3) * Real.pi * r^3)
  (h5 : ∃ (r : ℝ), prism.height = (3/2) * r) :
  ¬∃ (f : ℝ → ℝ), ∀ (s : Sphere), 
    (prism.base_side * prism.base_side * prism.height * Real.sqrt 3 / 4) = f s.volume :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_not_determinable_l700_70017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_k_for_integer_b_l700_70064

noncomputable def b : ℕ → ℝ
  | 0 => 2
  | (n + 1) => b n + (Real.log 3)⁻¹ * Real.log ((4 * (n + 1) + 3) / (4 * (n + 1) + 2))

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem least_integer_k_for_integer_b :
  ∃ k : ℕ, k > 1 ∧ is_integer (b k) ∧ ∀ m : ℕ, 1 < m ∧ m < k → ¬is_integer (b m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_k_for_integer_b_l700_70064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_eight_terms_l700_70087

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- The first, second, and fifth terms form a geometric sequence -/
def geometric_subsequence (a : ℕ → ℝ) : Prop :=
  (a 2) ^ 2 = a 1 * a 5

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a 1 + (n - 1) * 2)

theorem sum_of_eight_terms (a : ℕ → ℝ) :
  arithmetic_sequence a → geometric_subsequence a → arithmetic_sum a 8 = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_eight_terms_l700_70087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_problem_l700_70014

/-- The cost price of an item given its retail price, discounts, and profit margin. -/
noncomputable def cost_price (retail_price : ℝ) (percent_reduction : ℝ) (additional_discount : ℝ) (profit_margin : ℝ) : ℝ :=
  let reduced_price := retail_price * (1 - percent_reduction)
  let final_price := reduced_price - additional_discount
  final_price / (1 + profit_margin)

/-- Theorem stating that the cost price is 635 yuan given the problem conditions. -/
theorem cost_price_problem : cost_price 900 0.1 48 0.2 = 635 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_problem_l700_70014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l700_70004

/-- Given an ellipse C with specified properties, prove its equation and the range of t --/
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let e := Real.sqrt 2 / 2  -- eccentricity
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = b^2}
  let tangent_line := {(x, y) : ℝ × ℝ | x - y + Real.sqrt 2 = 0}
  ∀ (x y : ℝ), (x, y) ∈ C → 
    (∀ (p : ℝ × ℝ), p ∈ circle → p ∉ tangent_line) ∧
    (∃ (p : ℝ × ℝ), p ∈ circle ∧ p ∈ tangent_line) →
      (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 / 2 + y^2 = 1) ∧
      (∀ (t : ℝ) (A B P : ℝ × ℝ),
        A ∈ C ∧ B ∈ C ∧ P ∈ C ∧
        (∃ (k : ℝ), A.2 = k * (A.1 - 2) ∧ B.2 = k * (B.1 - 2)) ∧
        A + B = t • P ∧
        ‖A - P‖ - ‖B - P‖ < 2 * Real.sqrt 5 / 3 →
          t ∈ Set.Ioo (-2 : ℝ) (-2 * Real.sqrt 6 / 3) ∪ Set.Ioo (2 * Real.sqrt 6 / 3) 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l700_70004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_like_equation_solution_l700_70031

theorem fermat_like_equation_solution :
  ∀ n : ℕ, (∃ x y k : ℕ, k ≥ 2 ∧ Nat.Coprime x y ∧ 3^n = x^k + y^k) ↔ n = 0 ∨ n = 1 ∨ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_like_equation_solution_l700_70031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_negative_one_l700_70026

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.exp x + a * Real.exp (-x))

-- State the theorem
theorem even_function_implies_a_equals_negative_one :
  (∀ x : ℝ, f a x = f a (-x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_negative_one_l700_70026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_minus_alpha_l700_70032

theorem cos_pi_half_minus_alpha (α : ℝ) (h1 : π/2 < α) (h2 : α < π) (h3 : Real.sin α = 3/4) :
  Real.cos (π/2 - α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_minus_alpha_l700_70032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_shirt_cost_is_five_l700_70019

/-- The cost of shirts for field day -/
structure FieldDayShirts where
  kindergarten_count : ℕ
  kindergarten_cost : ℚ
  first_grade_count : ℕ
  second_grade_count : ℕ
  second_grade_cost : ℚ
  third_grade_count : ℕ
  third_grade_cost : ℚ
  total_spent : ℚ
  kindergarten_condition : kindergarten_count = 101 ∧ kindergarten_cost = 29/5
  first_grade_condition : first_grade_count = 113
  second_grade_condition : second_grade_count = 107 ∧ second_grade_cost = 28/5
  third_grade_condition : third_grade_count = 108 ∧ third_grade_cost = 21/4
  total_spent_condition : total_spent = 2317

/-- The cost of each yellow shirt -/
def yellow_shirt_cost (s : FieldDayShirts) : ℚ :=
  (s.total_spent - (s.kindergarten_count * s.kindergarten_cost + 
                    s.second_grade_count * s.second_grade_cost + 
                    s.third_grade_count * s.third_grade_cost)) / s.first_grade_count

/-- Theorem stating that the cost of each yellow shirt is $5.00 -/
theorem yellow_shirt_cost_is_five (s : FieldDayShirts) : 
  yellow_shirt_cost s = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_shirt_cost_is_five_l700_70019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_relation_l700_70079

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  first_term : a 1 = 2
  common_difference : ∀ n : ℕ, a (n + 1) = a n + d

/-- Represents a geometric sequence -/
structure GeometricSequence where
  b : ℕ → ℚ
  r : ℚ
  geometric_property : ∀ n : ℕ, b (n + 1) = b n * r

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic_sequence (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_geometric_relation 
  (arith : ArithmeticSequence) 
  (geo : GeometricSequence) 
  (sum_condition : sum_arithmetic_sequence arith 5 = 40)
  (relation1 : geo.b 3 = arith.a 3)
  (relation2 : geo.b 4 = arith.a 1 + arith.a 5) :
  arith.a 43 = geo.b 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_relation_l700_70079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trent_walked_blocks_l700_70077

/-- Represents the number of blocks Trent walked from his house to the bus stop. -/
def blocks_walked : ℕ → Prop := sorry

/-- Represents the total number of blocks Trent traveled. -/
def total_blocks : ℕ := sorry

/-- Represents the number of blocks Trent rode on the bus (one way). -/
def bus_ride_blocks : ℕ := sorry

/-- The theorem stating the number of blocks Trent walked from his house to the bus stop. -/
theorem trent_walked_blocks :
  total_blocks = 22 ∧ bus_ride_blocks = 7 →
  blocks_walked 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trent_walked_blocks_l700_70077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_probability_l700_70068

noncomputable def k_set : Finset ℝ := {-3, -1/2, Real.sqrt 3, 1, 6}

theorem inverse_proportion_quadrants_probability : 
  (Finset.filter (λ x => x < 0) k_set).card / k_set.card = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_probability_l700_70068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_example_l700_70062

/-- Given a triangle with inradius r and area A, calculate its perimeter p -/
noncomputable def triangle_perimeter (r : ℝ) (A : ℝ) : ℝ :=
  2 * A / r

theorem triangle_perimeter_example : 
  triangle_perimeter 2.5 60 = 48 := by
  -- Unfold the definition of triangle_perimeter
  unfold triangle_perimeter
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_example_l700_70062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_conditions_count_is_four_l700_70028

/-- The number of integer pairs (a, b) satisfying the field conditions --/
noncomputable def field_conditions_count : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    let a := p.1
    let b := p.2
    b > a ∧
    (a - 4) * (b - 4) = 2 * (a * b) / 3 ∧
    a ≥ 7 ∧
    b ≥ 7
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- Theorem stating that there are exactly 4 pairs satisfying the field conditions --/
theorem field_conditions_count_is_four :
  field_conditions_count = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_conditions_count_is_four_l700_70028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_proportion_trapezoid_segment_proportion_trapezoid_segment_lengths_l700_70030

-- Part 1
theorem similar_triangles_proportion (DE BC EC : ℝ) (h1 : DE = 6) (h2 : BC = 10) (h3 : EC = 3) :
  ∃ AE : ℝ, AE = 9/2 := by sorry

-- Part 2
theorem trapezoid_segment_proportion (WX ZY WM MZ XN NY : ℝ) 
  (h1 : WX / ZY = 3/4) (h2 : WM / MZ = 2/3) (h3 : XN / NY = 2/3) :
  WX / (WX * (5/17)) = 15/17 := by sorry

-- Part 3
theorem trapezoid_segment_lengths (WX ZY MN : ℕ) 
  (h1 : (WX : ℚ) / ZY = 3/4) 
  (h2 : ∃ (k : ℕ), k > 0 ∧ (MZ : ℚ) / WM = k ∧ (NY : ℚ) / XN = k) 
  (h3 : WX + MN + ZY = 2541) :
  MN ∈ ({763, 770, 777, 847} : Set ℕ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_proportion_trapezoid_segment_proportion_trapezoid_segment_lengths_l700_70030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_profit_percentage_l700_70042

/-- Represents the profit percentage of a sale -/
noncomputable def ProfitPercentage (costPrice sellingPrice : ℝ) : ℝ :=
  (sellingPrice - costPrice) / costPrice * 100

/-- Represents the selling price after applying a profit percentage -/
noncomputable def SellingPriceWithProfit (costPrice : ℝ) (profitPercentage : ℝ) : ℝ :=
  costPrice * (1 + profitPercentage / 100)

theorem bicycle_profit_percentage :
  let costPriceA : ℝ := 114.94
  let finalSellingPrice : ℝ := 225
  let profitPercentageB : ℝ := 45

  let sellingPriceB : ℝ := finalSellingPrice
  let costPriceB : ℝ := sellingPriceB / (1 + profitPercentageB / 100)
  let sellingPriceA : ℝ := costPriceB

  let profitPercentageA : ℝ := ProfitPercentage costPriceA sellingPriceA

  ∃ ε > 0, |profitPercentageA - 35.01| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_profit_percentage_l700_70042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_convergence_l700_70034

def y : ℕ → ℕ
  | 0 => 128  -- Add this case to cover all natural numbers
  | 1 => 128
  | (k+2) => y (k+1) * y (k+1) - y (k+1)

theorem sequence_sum_convergence :
  let series := fun n => 1 / ((y n - 1 : ℚ))
  (∑' n, series n) = 1 / 127 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_convergence_l700_70034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_theorem_l700_70095

-- Define the quadratic function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- Define the piecewise function F
noncomputable def F (a b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then f a b x else -f a b x

-- Define the function g
def g (a b k : ℝ) (x : ℝ) : ℝ := f a b x - k * x

-- State the theorem
theorem quadratic_functions_theorem (a b : ℝ) (h1 : a > 0) (h2 : f a b (-1) = 0) 
  (h3 : ∀ x, f a b x ≥ 0) :
  (∀ x, F a b x = if x > 0 then (x + 1)^2 else -(x + 1)^2) ∧
  (∀ k, (∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → StrictMono (g a b k)) ↔ (k ≥ 6 ∨ k ≤ -2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_theorem_l700_70095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_percentage_l700_70010

theorem price_decrease_percentage (initial_price : ℝ) (h : initial_price > 0) :
  let usd_to_eur := 0.85
  let eur_to_usd := 1.18
  let first_increase := 1.20
  let second_increase := 1.12
  let tax := 1.05
  let final_price := initial_price * first_increase * usd_to_eur * second_increase * tax * eur_to_usd
  let decrease_percentage := 1 - (1 / (first_increase * usd_to_eur * second_increase * tax * eur_to_usd))
  initial_price = final_price * (1 - decrease_percentage) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_percentage_l700_70010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_condition_l700_70072

-- Define IsHyperbola as a predicate on real numbers
def IsHyperbola (a b c d : ℝ) : Prop :=
  a * d < 0 ∧ b * c < 0

theorem hyperbola_condition (m : ℝ) :
  (∀ x y : ℝ, x^2 / (m - 2) + y^2 / (m + 3) = 1 → IsHyperbola (m - 2) (m + 3) 1 1) ↔ -3 < m ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_condition_l700_70072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_factorial_ratio_l700_70083

def factorial (n : ℕ) : ℕ := Nat.factorial n

def factorial_sum (n : ℕ) : ℕ := Finset.sum (Finset.range n) (λ i => factorial (i + 1))

theorem floor_factorial_ratio :
  ⌊(factorial 2002 : ℝ) / (factorial_sum 2001 : ℝ)⌋ = 2000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_factorial_ratio_l700_70083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_iff_last_digit_odd_perfect_ten_odd_probability_l700_70005

/-- A perfect ten three-digit number -/
def PerfectTenNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  (n / 100 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10) ∧
  (n / 100 + (n / 10) % 10 + n % 10 = 10)

/-- The set of all perfect ten three-digit numbers -/
def PerfectTenSet : Set ℕ :=
  {n : ℕ | PerfectTenNumber n}

/-- A number is odd if and only if its last digit is odd -/
theorem odd_iff_last_digit_odd (n : ℕ) : Odd n ↔ Odd (n % 10) := by sorry

/-- The main theorem: probability of a perfect ten three-digit number being odd is 1/2 -/
theorem perfect_ten_odd_probability :
  ∃ (s : Finset ℕ), s.Nonempty ∧ (∀ n ∈ s, PerfectTenNumber n) ∧
  (s.filter (λ n => Odd n)).card / s.card = 1 / 2 := by
  -- We will construct the set explicitly
  let s : Finset ℕ := {109, 190, 901, 910, 127, 172, 271, 217, 721, 712,
                       136, 163, 316, 361, 613, 631, 145, 154, 451, 415,
                       514, 541, 208, 280, 802, 820, 235, 253, 352, 325,
                       523, 532, 307, 370, 703, 730, 406, 460, 604, 640}
  use s
  sorry -- The proof details are omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_iff_last_digit_odd_perfect_ten_odd_probability_l700_70005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_times_rest_divisible_by_six_l700_70094

theorem last_digit_times_rest_divisible_by_six (k : ℕ) (h : k > 3) :
  let N := 2^k
  let a := N % 10
  let A := N / 10
  (a * A) % 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_times_rest_divisible_by_six_l700_70094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_3400_l700_70000

def binary_representation (n : ℕ) : List ℕ :=
  sorry

theorem sum_of_exponents_3400 :
  let binary_rep := binary_representation 3400
  binary_rep.sum = 38 ∧ 
  (∀ i j, i < binary_rep.length → j < binary_rep.length → i ≠ j → binary_rep.get ⟨i, by sorry⟩ ≠ binary_rep.get ⟨j, by sorry⟩) ∧
  3400 = (binary_rep.map (λ x => 2^x)).sum :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_3400_l700_70000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_exact_visitation_l700_70099

/-- Represents a chessboard with two pieces -/
structure Chessboard where
  total_squares : Nat
  total_placements : Nat
  monochromatic_placements : Nat
  polychromatic_placements : Nat

/-- Defines the properties of our specific chessboard -/
def our_chessboard : Chessboard :=
  { total_squares := 64,
    total_placements := 2016,
    monochromatic_placements := 992,
    polychromatic_placements := 1024 }

/-- Theorem stating that it's impossible to visit all placements exactly once -/
theorem impossibility_of_exact_visitation (cb : Chessboard) : 
  cb.total_squares = 64 →
  cb.total_placements = 2016 →
  cb.monochromatic_placements = 992 →
  cb.polychromatic_placements = 1024 →
  cb.monochromatic_placements + cb.polychromatic_placements = cb.total_placements →
  ¬ ∃ (sequence : List Nat), 
    (sequence.length = cb.total_placements) ∧ 
    (sequence.toFinset.card = cb.total_placements) ∧
    (∀ i, i < sequence.length - 1 → 
      (sequence.get ⟨i, by sorry⟩ % 2 = 0 → sequence.get ⟨i+1, by sorry⟩ % 2 = 1) ∧
      (sequence.get ⟨i, by sorry⟩ % 2 = 1 → sequence.get ⟨i+1, by sorry⟩ % 2 = 0)) :=
by sorry

#check impossibility_of_exact_visitation our_chessboard

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_exact_visitation_l700_70099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_right_angle_triangle_area_l700_70023

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola with equation x²/9 - y²/16 = 1 -/
def Hyperbola := {p : Point | p.x^2 / 9 - p.y^2 / 16 = 1}

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- Theorem: For the given hyperbola, if there's a point P on it forming a right angle with the foci,
    then the area of the triangle formed by P and the foci is 16 -/
theorem hyperbola_right_angle_triangle_area 
  (F1 F2 : Point) -- The foci of the hyperbola
  (P : Point) -- A point on the hyperbola
  (h1 : P ∈ Hyperbola) -- P is on the hyperbola
  (h2 : distance F1 P^2 + distance F2 P^2 = distance F1 F2^2) -- Right angle at P
  : triangleArea F1 F2 P = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_right_angle_triangle_area_l700_70023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_winner_strongest_l700_70003

-- Define the type for teams
variable {Team : Type}

-- Define the relation for one team defeating another
variable (defeats : Team → Team → Prop)

-- Define the property of a team being stronger than another
def stronger (A B : Team) : Prop :=
  defeats A B ∨ ∃ C, defeats A C ∧ defeats C B

-- Define the property of a team winning the tournament
def tournament_winner (A : Team) : Prop :=
  ∀ B : Team, B ≠ A → ∃ n : ℕ, n > 0 ∧ 
    (∃ (C : Fin (n + 1) → Team), C 0 = A ∧ C (Fin.last n) = B ∧
      ∀ i : Fin n, defeats (C i) (C i.succ))

-- Theorem statement
theorem tournament_winner_strongest {A : Team} :
  tournament_winner defeats A → ∀ B : Team, B ≠ A → stronger defeats A B :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_winner_strongest_l700_70003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l700_70022

noncomputable section

/-- The focal length of a hyperbola -/
def focal_length (a : ℝ) : ℝ := 4 * a

/-- The equation of a hyperbola -/
def hyperbola_equation (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The eccentricity of a hyperbola -/
def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_focal_length 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (point_on_hyperbola : hyperbola_equation 2 3 a b)
  (e : eccentricity a (a * 2) = 2) :
  focal_length a = 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l700_70022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_is_twelve_l700_70054

/-- A rectangle with a given perimeter and width -/
structure Rectangle where
  perimeter : ℝ
  width : ℝ

/-- The length of a rectangle given its perimeter and width -/
noncomputable def Rectangle.length (r : Rectangle) : ℝ :=
  (r.perimeter / 2) - r.width

/-- Theorem: For a rectangle with perimeter 40 and width 8, the length is 12 -/
theorem rectangle_length_is_twelve (r : Rectangle) 
  (h_perimeter : r.perimeter = 40)
  (h_width : r.width = 8) : 
  r.length = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_is_twelve_l700_70054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_center_sum_l700_70088

noncomputable section

structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

def is_in_first_quadrant (s : Square) : Prop :=
  s.A.1 ≥ 0 ∧ s.A.2 ≥ 0 ∧
  s.B.1 ≥ 0 ∧ s.B.2 ≥ 0 ∧
  s.C.1 ≥ 0 ∧ s.C.2 ≥ 0 ∧
  s.D.1 ≥ 0 ∧ s.D.2 ≥ 0

def point_on_line (p : ℝ × ℝ) (l1 : ℝ × ℝ) (l2 : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p = (l1.1 + t * (l2.1 - l1.1), l1.2 + t * (l2.2 - l1.2))

def center (s : Square) : ℝ × ℝ :=
  ((s.A.1 + s.C.1) / 2, (s.A.2 + s.C.2) / 2)

theorem square_center_sum (s : Square) 
  (h1 : is_in_first_quadrant s)
  (h2 : point_on_line (9, 0) s.A s.D)
  (h3 : point_on_line (4, 0) s.B s.C)
  (h4 : point_on_line (0, 3) s.C s.D) :
  (center s).1 + (center s).2 = 8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_center_sum_l700_70088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l700_70045

def M : ℕ := 2^4 * 3^3 * 7^2

theorem number_of_factors_of_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l700_70045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lines_equal_angles_l700_70050

/-- A type representing a line in 3D space -/
def Line3D : Type := Unit

/-- A type representing a plane in 3D space -/
def Plane : Type := Unit

/-- Given a line and a plane, this function returns the angle between them -/
noncomputable def angle_with_plane (l : Line3D) (p : Plane) : ℝ := sorry

/-- This function checks if all lines in a list form equal angles with a given plane -/
def all_equal_angles (lines : List Line3D) (p : Plane) : Prop :=
  ∀ l1 l2, l1 ∈ lines → l2 ∈ lines → angle_with_plane l1 p = angle_with_plane l2 p

/-- This function checks if there exists a plane such that all given lines form equal angles with it -/
def exists_equal_angle_plane (lines : List Line3D) : Prop :=
  ∃ p : Plane, all_equal_angles lines p

/-- The main theorem: the maximum number of arbitrary lines that always satisfy the equal angle condition is 3 -/
theorem max_lines_equal_angles :
  (∀ (lines : List Line3D), lines.length ≤ 3 → exists_equal_angle_plane lines) ∧
  (∃ (lines : List Line3D), lines.length = 4 ∧ ¬exists_equal_angle_plane lines) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lines_equal_angles_l700_70050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_placement_theorem_l700_70036

/-- The number of radars --/
def n : ℕ := 5

/-- The radius of each radar's coverage area in km --/
noncomputable def r : ℝ := 25

/-- The width of the coverage ring in km --/
noncomputable def w : ℝ := 14

/-- The angle between adjacent radars in radians --/
noncomputable def θ : ℝ := 2 * Real.pi / n

/-- The maximum distance from the center to each radar --/
noncomputable def max_distance : ℝ := 24 / Real.sin (θ / 2)

/-- The area of the coverage ring --/
noncomputable def coverage_area : ℝ := 672 * Real.pi / Real.tan (θ / 2)

theorem radar_placement_theorem :
  (n = 5 ∧ r = 25 ∧ w = 14) →
  (max_distance = 24 / Real.sin (Real.pi / 5) ∧
   coverage_area = 672 * Real.pi / Real.tan (Real.pi / 5)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_placement_theorem_l700_70036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l700_70012

noncomputable def g (D E F : ℤ) (x : ℝ) : ℝ := x^2 / (D * x^2 + E * x + F)

theorem asymptote_sum (D E F : ℤ) :
  (∀ x : ℝ, x > 5 → g D E F x > 0.5) →
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 2 → g D E F x ≠ 0) →
  g D E F 0 = 1/4 →
  D + E + F = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l700_70012
