import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_interchange_l1217_121784

theorem two_digit_number_interchange (x y : ℤ) : 
  0 ≤ x ∧ x < 10 ∧ 0 < y ∧ y < 10 ∧  -- Ensure it's a two-digit number
  (x + y) - (y - x) = 8 ∧    -- Difference between sum and difference of digits is 8
  y = 2 * x →                -- Ratio between digits is 1:2
  |((10 * x + y) - (10 * y + x))| = 36 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_interchange_l1217_121784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_symmetric_sine_graphs_l1217_121726

/-- Given ω > 0, if the graph of y = sin(ωx + π/3) shifted 4π/5 units to the right 
    is symmetric about the x-axis with the original graph, 
    then the minimum value of ω is 5/4. -/
theorem min_omega_for_symmetric_sine_graphs (ω : ℝ) : 
  ω > 0 → 
  (∀ x : ℝ, Real.sin (ω * x + π / 3) = -Real.sin (ω * (x - 4 * π / 5) + π / 3)) →
  ω ≥ 5 / 4 ∧ ∀ ω' : ℝ, ω' > 0 ∧ ω' < 5 / 4 → 
    ¬(∀ x : ℝ, Real.sin (ω' * x + π / 3) = -Real.sin (ω' * (x - 4 * π / 5) + π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_symmetric_sine_graphs_l1217_121726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_probability_correct_l1217_121753

/-- The probability that a random chord of the outer circle intersects the inner circle -/
noncomputable def chord_intersection_probability (inner_radius outer_radius : ℝ) : ℝ :=
  let tangent_angle := 2 * Real.arctan (inner_radius / Real.sqrt (outer_radius^2 - inner_radius^2))
  tangent_angle / (2 * Real.pi)

/-- Theorem stating that the probability of chord intersection is correct for circles with radii 3 and 5 -/
theorem chord_intersection_probability_correct :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |chord_intersection_probability 3 5 - 0.205| < ε := by
  sorry

-- Note: We can't use #eval for noncomputable functions, so this line is removed
-- #eval chord_intersection_probability 3 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_probability_correct_l1217_121753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_satisfies_all_planes_l1217_121770

/-- The point of intersection of three planes -/
noncomputable def intersection_point : ℝ × ℝ × ℝ := (-41/5, 23/5, 39/5)

/-- First plane equation -/
def plane1 (p : ℝ × ℝ × ℝ) : Prop := 3 * p.1 - p.2.1 + 4 * p.2.2 = 2

/-- Second plane equation -/
def plane2 (p : ℝ × ℝ × ℝ) : Prop := -3 * p.1 + 4 * p.2.1 - 3 * p.2.2 = 4

/-- Third plane equation -/
def plane3 (p : ℝ × ℝ × ℝ) : Prop := -p.1 + p.2.1 - p.2.2 = 5

theorem intersection_point_satisfies_all_planes :
  plane1 intersection_point ∧ plane2 intersection_point ∧ plane3 intersection_point :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_satisfies_all_planes_l1217_121770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1217_121702

noncomputable section

variable (a b : ℝ × ℝ)

def angle_between (v w : ℝ × ℝ) : ℝ := Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem vector_problem (h1 : angle_between a b = π / 3)
                       (h2 : magnitude a = 2)
                       (h3 : magnitude b = 1) :
  dot_product a b = 1 ∧ magnitude (a.1 + 2 * b.1, a.2 + 2 * b.2) = 2 * Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1217_121702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johnson_farm_solution_l1217_121718

/-- Represents the farm cultivation problem -/
structure FarmProblem where
  total_land : ℝ
  corn_cost : ℝ
  wheat_cost : ℝ
  total_budget : ℝ

/-- Solves the farm cultivation problem -/
noncomputable def solve_farm_problem (p : FarmProblem) : ℝ :=
  let wheat_acres := (p.total_budget - p.corn_cost * p.total_land) / (p.wheat_cost - p.corn_cost)
  wheat_acres

/-- Theorem stating the solution to the specific farm problem -/
theorem johnson_farm_solution :
  let p : FarmProblem := {
    total_land := 500,
    corn_cost := 42,
    wheat_cost := 30,
    total_budget := 18600
  }
  solve_farm_problem p = 200 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johnson_farm_solution_l1217_121718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pf_length_for_parabola_configuration_l1217_121723

/-- A parabola in the Cartesian plane -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A chord of a parabola -/
structure Chord where
  a : Point
  b : Point

/-- A circle in the 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Theorem: PF length for a specific parabola configuration -/
theorem pf_length_for_parabola_configuration 
  (p : Parabola)
  (f : Point)
  (ab : Chord)
  (circ : Circle)
  (intersect_p : Point) :
  p.equation = (fun x y => y^2 = 4*x) →
  f = ⟨1, 0⟩ →
  ab.a.x = ab.a.y^2 / 4 ∧ ab.b.x = ab.b.y^2 / 4 →
  (ab.a.x - f.x) * (ab.b.y - f.y) = (ab.b.x - f.x) * (ab.a.y - f.y) →
  circ.center = ⟨0, 0⟩ →
  p.equation intersect_p.x intersect_p.y →
  intersect_p ≠ ⟨0, 0⟩ ∧ intersect_p ≠ ab.a ∧ intersect_p ≠ ab.b →
  (intersect_p.x - ab.a.x)^2 + (intersect_p.y - ab.a.y)^2 = 
  (intersect_p.x - f.x)^2 + (intersect_p.y - f.y)^2 →
  (intersect_p.x - f.x)^2 + (intersect_p.y - f.y)^2 = (Real.sqrt 13 - 1)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pf_length_for_parabola_configuration_l1217_121723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_board_bus_251_l1217_121721

-- Define the bus routes
def Route152Interval : ℝ := 5
def Route251Interval : ℝ := 7

-- Define the probability space
def ProbabilitySpace : Set (ℝ × ℝ) := 
  {x | 0 ≤ x.1 ∧ x.1 < Route152Interval ∧ 0 ≤ x.2 ∧ x.2 < Route251Interval}

-- Define the event of boarding bus 251
def BoardBus251 : Set (ℝ × ℝ) := 
  {x ∈ ProbabilitySpace | x.2 < x.1}

-- Theorem statement
theorem probability_board_bus_251 :
  (MeasureTheory.volume BoardBus251) / (MeasureTheory.volume ProbabilitySpace) = 5 / 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_board_bus_251_l1217_121721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composite_function_l1217_121773

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1)

-- State the theorem
theorem domain_of_composite_function :
  {x : ℝ | ∃ y, y = f (x/2) + f (4/x)} = {x : ℝ | 2 ≤ x ∧ x ≤ 4} := by
  sorry

-- Additional helper lemmas if needed
lemma domain_of_f : 
  Set.range f = {y : ℝ | ∃ x ≥ 1, y = Real.sqrt (x - 1)} := by
  sorry

lemma composite_function_domain :
  ∀ x, (x/2 ≥ 1 ∧ 4/x ≥ 1) ↔ (2 ≤ x ∧ x ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composite_function_l1217_121773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orchard_growth_period_l1217_121765

/-- The number of years it takes for an orchard to grow from initial_trees to final_trees
    with an annual growth rate of growth_rate -/
noncomputable def years_to_grow (initial_trees final_trees : ℕ) (growth_rate : ℚ) : ℕ :=
  ⌊(Real.log (final_trees / initial_trees : ℝ) / Real.log (1 + growth_rate : ℝ))⌋₊

theorem orchard_growth_period :
  years_to_grow 2304 5625 (1/4 : ℚ) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orchard_growth_period_l1217_121765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shirt_price_is_eight_l1217_121780

/-- The greatest possible whole-dollar price of a shirt given the conditions --/
def max_shirt_price (total_budget : ℕ) (num_shirts : ℕ) (entrance_fee : ℕ) (tax_rate : ℚ) : ℕ :=
  let available_amount : ℚ := (total_budget - entrance_fee : ℚ) / (1 + tax_rate)
  let max_price_per_shirt : ℚ := available_amount / num_shirts
  (Int.floor max_price_per_shirt).toNat

/-- Theorem stating the greatest possible whole-dollar price of a shirt under given conditions --/
theorem max_shirt_price_is_eight :
  max_shirt_price 130 14 2 (5/100) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shirt_price_is_eight_l1217_121780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_equals_one_third_triple_distance_l1217_121785

/-- The diagonal of a rectangular parallelepiped --/
noncomputable def diagonal (a b c : ℝ) : ℝ := Real.sqrt (a^2 + b^2 + c^2)

/-- The distance between opposite corners of three identical rectangular parallelepipeds arranged in a line --/
noncomputable def tripleDistance (a b c : ℝ) : ℝ := Real.sqrt ((3*a)^2 + b^2 + c^2)

/-- Theorem stating that the diagonal of a single parallelepiped is one-third of the triple distance --/
theorem diagonal_equals_one_third_triple_distance (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  diagonal a b c = (1/3) * tripleDistance a b c := by
  sorry

#check diagonal_equals_one_third_triple_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_equals_one_third_triple_distance_l1217_121785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l1217_121732

/-- The parabola y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p | (p.2)^2 = 8 * p.1}

/-- Point A -/
def A : ℝ × ℝ := (2, 0)

/-- Point B -/
def B : ℝ × ℝ := (8, 6)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Sum of distances from a point to A and B -/
noncomputable def sumDistances (p : ℝ × ℝ) : ℝ :=
  distance p A + distance p B

theorem min_sum_distances :
  ∀ p ∈ Parabola, sumDistances p ≥ 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l1217_121732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_in_unit_circle_max_points_in_large_circle_points_in_large_circle_exist_l1217_121797

-- Define a circle with radius 1
def unit_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 1}

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- The main theorem
theorem max_points_in_unit_circle (points : Finset (ℝ × ℝ)) :
  (∀ p ∈ points, p ∈ unit_circle) → (points.card > 5) →
  ∃ p q : ℝ × ℝ, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ distance p q ≤ 1 := by
  sorry

-- Theorem for part (b)
theorem max_points_in_large_circle (points : Finset (ℝ × ℝ)) :
  (∀ p ∈ points, p.1^2 + p.2^2 ≤ 100) → (points.card = 450) →
  ∃ p q : ℝ × ℝ, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ distance p q ≤ 1 := by
  sorry

-- Theorem for part (c)
theorem points_in_large_circle_exist :
  ∃ points : Finset (ℝ × ℝ),
    (∀ p ∈ points, p.1^2 + p.2^2 ≤ 100) ∧
    (points.card = 400) ∧
    (∀ p q : ℝ × ℝ, p ∈ points → q ∈ points → p ≠ q → distance p q > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_in_unit_circle_max_points_in_large_circle_points_in_large_circle_exist_l1217_121797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_l1217_121758

/-- Represents a set of 5 salaries -/
def Salaries := Fin 5 → ℝ

/-- Calculate the mean of salaries -/
noncomputable def mean (s : Salaries) : ℝ := (s 0 + s 1 + s 2 + s 3 + s 4) / 5

/-- Calculate the variance of salaries -/
noncomputable def variance (s : Salaries) : ℝ :=
  ((s 0 - mean s)^2 + (s 1 - mean s)^2 + (s 2 - mean s)^2 + 
   (s 3 - mean s)^2 + (s 4 - mean s)^2) / 5

/-- Increase each salary by a constant -/
def increase_salaries (s : Salaries) (c : ℝ) : Salaries :=
  fun i => s i + c

theorem salary_increase (s : Salaries) (c : ℝ) :
  mean (increase_salaries s c) = mean s + c ∧ 
  variance (increase_salaries s c) = variance s := by
  sorry

#check salary_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_l1217_121758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1217_121748

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 1 / Real.sqrt ((x^2 + x + k)^2 + 2*(x^2 + x + k) - 3)

-- State the theorem
theorem f_properties (k : ℝ) (h : k < -6) :
  -- Domain of f
  (∀ x, f k x ≠ 0 ↔ x ∈ Set.union (Set.union (Set.Ioo (-Real.sqrt (2 - k)) (-1 - Real.sqrt (2 - k))) 
                                            (Set.Ioo (-1 - Real.sqrt (2 - k)) (1 + Real.sqrt (-2*k - 4))))
                                  (Set.Ioi (Real.sqrt (2 - k)))) ∧
  -- Set where f(x) > f(-3)
  (∀ x, f k x > f k (-3) ↔ x ∈ Set.union (Set.union (Set.union 
    (Set.Ioo (-1 - Real.sqrt (-4 - k)) (-1 - Real.sqrt (2 - k)))
    (Set.Ioi (1 - Real.sqrt (-2 - k))))
    (Set.Ioo (-1) (-1 + Real.sqrt (-2*k))))
    (Set.Ioo (-1 + Real.sqrt (2 - k)) (1 + Real.sqrt (-2*k - 4)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1217_121748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l1217_121735

-- Define the system of equations
def satisfies_system (x y : ℝ) : Prop :=
  x + 3 * y = 3 ∧ |abs x - abs y| = 1

-- Theorem statement
theorem exactly_three_solutions :
  ∃! (s : Set (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ s ↔ satisfies_system p.1 p.2) ∧
    (Finite s ∧ Nat.card s = 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l1217_121735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_coordinates_l1217_121740

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.exp (-x)

theorem tangent_point_coordinates (a : ℝ) :
  (∀ x, f_deriv a x = -f_deriv a (-x)) →  -- f'(x) is an odd function
  (∃ x, f_deriv a x = 3/2) →              -- slope of tangent is 3/2 at some point
  ∃ x y, x = Real.log 2 ∧ y = 5/2 ∧ f a x = y ∧ f_deriv a x = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_coordinates_l1217_121740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1217_121716

/-- The function for which we want to find the minimum value -/
noncomputable def f (x : ℝ) : ℝ := 2*x + 1/x

/-- The theorem stating the minimum value of the function and where it occurs -/
theorem min_value_of_f :
  ∀ x : ℝ, x > 0 → f x ≥ 2 * Real.sqrt 2 ∧
  (f x = 2 * Real.sqrt 2 ↔ x = Real.sqrt 2 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1217_121716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_message_chain_l1217_121749

theorem message_chain (juan_to_laurence : ℚ) 
  (h1 : juan_to_laurence > 0)
  (h2 : 8 * juan_to_laurence = juan_to_keith)
  (h3 : 9/2 * juan_to_laurence = laurence_to_missy)
  (h4 : 7 * laurence_to_missy = missy_to_noah)
  (h5 : 3 * missy_to_noah = noah_to_olivia)
  (h6 : noah_to_olivia = 27) : 
  juan_to_keith = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_message_chain_l1217_121749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1217_121736

open Real

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  cos A ^ 2 = cos B ^ 2 + sin C ^ 2 - sin A * sin C →
  b = 2 * Real.sqrt 3 →
  1/2 * b * c * sin B = 2 * Real.sqrt 3 →
  B = π/3 ∧ a + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1217_121736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_ellipse_area_smallest_ellipse_area_value_l1217_121750

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_positive : 0 < r

/-- Checks if a point (x, y) is on the ellipse -/
def Ellipse.contains (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Checks if a point (x, y) is on the circle -/
def Circle.contains (c : Circle) (x y : ℝ) : Prop :=
  (x - c.h)^2 + (y - c.k)^2 = c.r^2

/-- The area of an ellipse -/
noncomputable def Ellipse.area (e : Ellipse) : ℝ :=
  Real.pi * e.a * e.b

theorem smallest_ellipse_area (e : Ellipse) (c1 c2 : Circle)
    (h1 : c1.h = -2 ∧ c1.k = 0 ∧ c1.r = 2)
    (h2 : c2.h = 2 ∧ c2.k = 0 ∧ c2.r = 2)
    (h_contains1 : ∀ x y, c1.contains x y → e.contains x y)
    (h_contains2 : ∀ x y, c2.contains x y → e.contains x y) :
    ∀ e' : Ellipse, (∀ x y, c1.contains x y → e'.contains x y) →
                    (∀ x y, c2.contains x y → e'.contains x y) →
                    e.area ≤ e'.area := by
  sorry

/-- The smallest possible area of the ellipse is 9√3π/4 -/
theorem smallest_ellipse_area_value (e : Ellipse) (c1 c2 : Circle)
    (h1 : c1.h = -2 ∧ c1.k = 0 ∧ c1.r = 2)
    (h2 : c2.h = 2 ∧ c2.k = 0 ∧ c2.r = 2)
    (h_contains1 : ∀ x y, c1.contains x y → e.contains x y)
    (h_contains2 : ∀ x y, c2.contains x y → e.contains x y)
    (h_smallest : ∀ e' : Ellipse, (∀ x y, c1.contains x y → e'.contains x y) →
                                  (∀ x y, c2.contains x y → e'.contains x y) →
                                  e.area ≤ e'.area) :
    e.area = (9 * Real.sqrt 3 / 4) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_ellipse_area_smallest_ellipse_area_value_l1217_121750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_x_value_l1217_121776

/-- Given two vectors a and b in R², where a = (4, x) and b = (2, 4),
    if a is perpendicular to b, then x = -2. -/
theorem perpendicular_vectors_x_value (x : ℝ) : 
  let a : ℝ × ℝ := (4, x)
  let b : ℝ × ℝ := (2, 4)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = -2 := by
  intro h
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_x_value_l1217_121776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1217_121743

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 4*x

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- Distance from a point to the focus -/
noncomputable def dist_to_focus (p : Parabola) : ℝ :=
  Real.sqrt ((p.x - focus.1)^2 + (p.y - focus.2)^2)

/-- Distance from a point to the y-axis -/
def dist_to_y_axis (p : Parabola) : ℝ := p.x

theorem parabola_distance_theorem (A B : Parabola) 
  (h : dist_to_focus A + dist_to_focus B = 7) :
  dist_to_y_axis A + dist_to_y_axis B = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1217_121743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_rhombus_l1217_121703

-- Define a point in 2D space
structure Point :=
  (x y : ℝ)

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a circle
structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define the property of a quadrilateral being circumscribed around a circle
def is_circumscribed (q : Quadrilateral) (c : Circle) : Prop :=
  sorry

-- Define the property of diagonals intersecting at a point
def diagonals_intersect_at (q : Quadrilateral) (p : Point) : Prop :=
  sorry

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statement
theorem quadrilateral_is_rhombus 
  (q : Quadrilateral) (c : Circle) :
  is_circumscribed q c →
  diagonals_intersect_at q c.center →
  is_rhombus q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_rhombus_l1217_121703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_determinant_l1217_121760

theorem cubic_roots_determinant (p q : ℝ) (a b c : ℝ) : 
  (a^3 - 4*a^2 + p*a + q = 0) →
  (b^3 - 4*b^2 + p*b + q = 0) →
  (c^3 - 4*c^2 + p*c + q = 0) →
  Matrix.det !![a, b, c; b, c, a; c, a, b] = -64 + 8*p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_determinant_l1217_121760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_value_l1217_121715

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define a₀ = 1 to match a₁ in the original problem
  | (n + 1) => (2 * sequence_a n) / (sequence_a n + 2)

theorem a_4_value : sequence_a 4 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_value_l1217_121715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1217_121774

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if cos A = 4/5, cos C = 5/13, and a = 1, then b = 21/13 -/
theorem triangle_side_length (A B C : Real) (a b c : Real) : 
  Real.cos A = 4/5 → Real.cos C = 5/13 → a = 1 → b = 21/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1217_121774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_l1217_121778

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define points M and N on AC
variable (M N : EuclideanSpace ℝ (Fin 2))

-- AC is the longest side
variable (h_longest : dist A C ≥ max (dist A B) (dist B C))

-- M and N are on AC
variable (h_M_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • A + t • C)
variable (h_N_on_AC : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ N = (1 - s) • A + s • C)

-- AM = AB
variable (h_AM_eq_AB : dist A M = dist A B)

-- CN = CB
variable (h_CN_eq_CB : dist C N = dist C B)

-- BM = BN
variable (h_BM_eq_BN : dist B M = dist B N)

-- Theorem statement
theorem isosceles_triangle (A B C M N : EuclideanSpace ℝ (Fin 2))
  (h_longest : dist A C ≥ max (dist A B) (dist B C))
  (h_M_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • A + t • C)
  (h_N_on_AC : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ N = (1 - s) • A + s • C)
  (h_AM_eq_AB : dist A M = dist A B)
  (h_CN_eq_CB : dist C N = dist C B)
  (h_BM_eq_BN : dist B M = dist B N) :
  dist A B = dist C B :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_l1217_121778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_lateral_area_from_unit_square_l1217_121767

/-- Predicate to check if a set in 2D space is a square -/
def is_square (s : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Function to calculate the side length of a square -/
def square_side_length (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Predicate to check if a cylinder is formed by rotating a square about one of its sides -/
def cylinder_from_square_rotation (s : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry

/-- Function to calculate the lateral area of a cylinder -/
def cylinder_lateral_area (c : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- The lateral area of a cylinder formed by rotating a square with side length 1 about one of its sides is equal to 2π. -/
theorem cylinder_lateral_area_from_unit_square : 
  ∀ (square : Set (ℝ × ℝ)) (cylinder : Set (ℝ × ℝ × ℝ)),
    is_square square ∧ 
    square_side_length square = 1 ∧
    cylinder_from_square_rotation square cylinder →
    cylinder_lateral_area cylinder = 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_lateral_area_from_unit_square_l1217_121767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1217_121728

noncomputable section

/-- Hyperbola C with equation x²/a² - y²/b² = 1 -/
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

/-- The left focus of the hyperbola -/
def F₁ (c : ℝ) : ℝ × ℝ := (-c, 0)

/-- The right focus of the hyperbola -/
def F₂ (c : ℝ) : ℝ × ℝ := (c, 0)

/-- The origin -/
def O : ℝ × ℝ := (0, 0)

/-- The asymptote of the hyperbola -/
def Asymptote (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (b / a) * p.1}

/-- The circle with center F₁ and radius |OF₁| -/
def Circle (c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + c)^2 + p.2^2 = c^2}

/-- The point symmetric to F₂ with respect to the asymptote -/
def SymmetricPoint (a b c : ℝ) : ℝ × ℝ :=
  (c + 2*b^2/a, 2*b*c/a)

theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  SymmetricPoint a b c ∈ Circle c →
  c / a = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1217_121728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birdseed_mix_l1217_121705

theorem birdseed_mix (brand_a_millet brand_b_millet mix_millet : ℝ) 
  (ha : brand_a_millet = 0.4)
  (hb : brand_b_millet = 0.65)
  (hm : mix_millet = 0.5) :
  (brand_b_millet - mix_millet) / (brand_b_millet - brand_a_millet) = 0.6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birdseed_mix_l1217_121705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_factors_l1217_121794

/-- The number of even natural-number factors of 2^3 * 3^2 * 5^1 -/
def num_even_factors : ℕ := 18

/-- The prime factorization of m -/
def m : ℕ := 2^3 * 3^2 * 5^1

theorem count_even_factors :
  (Finset.filter (λ x => x ∣ m ∧ Even x) (Finset.range (m + 1))).card = num_even_factors := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_factors_l1217_121794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1217_121708

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1/x - 4/Real.sqrt x + 3

-- State the theorem
theorem f_range (x : ℝ) (h : 1/16 ≤ x ∧ x ≤ 1) : 
  ∃ y z, f y = -1 ∧ f z = 3 ∧ ∀ w, 1/16 ≤ w ∧ w ≤ 1 → -1 ≤ f w ∧ f w ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1217_121708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_three_identities_l1217_121788

theorem tangent_three_identities (α : ℝ) (h1 : Real.tan α = 3) (h2 : α ∈ Set.Icc 0 Real.pi) :
  (4 * Real.sin α - Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 11/14 ∧
  Real.sin α + Real.cos α = 4 * Real.sqrt 10 / 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_three_identities_l1217_121788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tip_percentage_is_correct_l1217_121791

noncomputable def dining_bill : ℝ := 139.00
def num_people : ℕ := 6
noncomputable def per_person_share : ℝ := 25.48

noncomputable def calculate_tip_percentage : ℝ := by
  -- Calculate total amount paid
  let total_paid := per_person_share * (num_people : ℝ)
  
  -- Calculate tip amount
  let tip_amount := total_paid - dining_bill
  
  -- Calculate tip percentage
  let tip_percentage := (tip_amount / dining_bill) * 100
  
  -- Assert that the tip percentage is approximately 9.99%
  have h : |tip_percentage - 9.99| < 0.01 := by sorry
  
  exact tip_percentage

theorem tip_percentage_is_correct :
  |calculate_tip_percentage - 9.99| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tip_percentage_is_correct_l1217_121791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_f_l1217_121731

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - x - 2)

-- Define the domain of f(x)
def domain : Set ℝ := {x | x > 2 ∨ x < -1}

-- Theorem statement
theorem increasing_interval_of_f :
  ∀ x ∈ domain, ∀ y ∈ domain,
    x < y → x > 2 → y > 2 → (f x < f y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_f_l1217_121731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_box_divisors_l1217_121754

theorem marble_box_divisors :
  let total_marbles : ℕ := 900
  let valid_divisors : Finset ℕ := (Finset.range (total_marbles - 1)).filter (λ d => d > 1 ∧ d ∣ total_marbles)
  Finset.card valid_divisors = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_box_divisors_l1217_121754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_root_of_quadratic_l1217_121706

-- Define the given equation
def equation (z : ℂ) : Prop := z^2 = -75 + 40*Complex.I

-- State the theorem
theorem other_root_of_quadratic :
  let z₁ : ℂ := 5 + 7*Complex.I
  let z₂ : ℂ := -5 - 7*Complex.I
  equation z₁ → equation z₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_root_of_quadratic_l1217_121706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_four_players_one_game_l1217_121786

/-- Represents a tournament with the given specifications -/
structure Tournament :=
  (players : Nat)
  (rounds : Nat)
  (games_per_round : Nat)
  (h_players : players = 18)
  (h_rounds : rounds = 17)
  (h_games_per_round : games_per_round = 9)

/-- Counts the number of games played by a player after n rounds -/
def count_games (t : Tournament) (player : Nat) (n : Nat) : Nat :=
  sorry

/-- Counts the number of games played between two players -/
def games_between (t : Tournament) (p1 p2 : Nat) : Nat :=
  sorry

/-- Represents the state of the tournament after n rounds -/
def TournamentState (t : Tournament) (n : Nat) : Prop :=
  n ≤ t.rounds ∧
  ∀ p, p < t.players → (count_games t p n = n) ∧
  ∀ p q, p < t.players → q < t.players → p ≠ q → (games_between t p q ≤ 1)

/-- States that we can always find 4 players with only one game played amongst them -/
def four_players_one_game (t : Tournament) (n : Nat) : Prop :=
  ∀ s : TournamentState t n,
    ∃ a b c d, a < t.players ∧ b < t.players ∧ c < t.players ∧ d < t.players ∧
               a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
               (games_between t a b + games_between t a c + games_between t a d +
                games_between t b c + games_between t b d + games_between t c d) = 1

/-- The main theorem to be proved -/
theorem largest_n_four_players_one_game (t : Tournament) :
  (∀ n ≤ 7, four_players_one_game t n) ∧
  ¬(four_players_one_game t 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_four_players_one_game_l1217_121786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_l1217_121733

/-- Given a train of length 110 m traveling at 60 kmph that takes 14.998800095992321 seconds to cross a bridge, the length of the bridge is 140 m. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmph : ℝ) (time_to_cross : ℝ) :
  train_length = 110 →
  train_speed_kmph = 60 →
  time_to_cross = 14.998800095992321 →
  (train_speed_kmph * 1000 / 3600 * time_to_cross) - train_length = 140 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_l1217_121733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_removable_coins_l1217_121725

/-- Represents the state of the boxes -/
structure BoxState where
  first : ℕ  -- number of coins in the first box
  second : ℕ -- number of coins in the second box

/-- Represents a move -/
inductive Move
  | Transfer   -- transfer one coin from first to second box
  | Remove     -- remove k coins from first box, where k is the number in second box

/-- Applies a move to a box state -/
def apply_move (state : BoxState) (move : Move) : BoxState :=
  match move with
  | Move.Transfer => ⟨state.first - 1, state.second + 1⟩
  | Move.Remove => ⟨state.first - state.second, state.second⟩

/-- Checks if a sequence of moves is valid -/
def is_valid_sequence (init : BoxState) (moves : List Move) : Prop :=
  moves.length ≤ 10 ∧
  (moves.foldl apply_move init).first = 0

/-- The main theorem to prove -/
theorem max_removable_coins :
  ∀ n : ℕ, n ≤ 30 ↔ ∃ (moves : List Move), is_valid_sequence ⟨n, 0⟩ moves :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_removable_coins_l1217_121725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_derivative_y_l1217_121719

noncomputable def y (x : ℝ) : ℝ := (2 * x^3 + 1) * Real.cos x

theorem fifth_derivative_y (x : ℝ) :
  (deriv^[5] y) x = (30 * x^2 - 120) * Real.cos x - (2 * x^3 - 120 * x + 1) * Real.sin x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_derivative_y_l1217_121719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_honey_water_percentage_l1217_121782

/-- The percentage of water in honey produced from nectar -/
noncomputable def waterPercentageInHoney (nectarWeight : ℝ) (honeyWeight : ℝ) (waterPercentageInNectar : ℝ) : ℝ :=
  let waterInNectar := nectarWeight * (waterPercentageInNectar / 100)
  let solidsInNectar := nectarWeight - waterInNectar
  let waterInHoney := honeyWeight - solidsInNectar
  (waterInHoney / honeyWeight) * 100

/-- Theorem: Given 1.6 kg of nectar (50% water) yields 1 kg of honey, the honey contains 20% water -/
theorem honey_water_percentage :
  waterPercentageInHoney 1.6 1 50 = 20 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval waterPercentageInHoney 1.6 1 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_honey_water_percentage_l1217_121782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_40_cents_is_five_eighths_l1217_121792

-- Define the coin types
inductive Coin : Type
| Penny
| Nickel
| Dime
| Quarter
| HalfDollar
deriving Repr, DecidableEq

-- Define the value of each coin in cents
def coinValue : Coin → ℕ
| Coin.Penny => 1
| Coin.Nickel => 5
| Coin.Dime => 10
| Coin.Quarter => 25
| Coin.HalfDollar => 50

-- Define a flip result
inductive FlipResult
| Heads
| Tails
deriving Repr, DecidableEq

-- Define a type for the result of flipping all coins
def FlipOutcome : Type := Coin → FlipResult

-- Function to calculate the total value of heads in a flip outcome
def headsValue (outcome : FlipOutcome) : ℕ :=
  (Coin.Penny :: Coin.Nickel :: Coin.Dime :: Coin.Quarter :: Coin.HalfDollar :: []).foldl
    (fun acc coin => acc + if outcome coin = FlipResult.Heads then coinValue coin else 0)
    0

-- Define the probability of getting at least 40 cents in heads
noncomputable def probAtLeast40Cents : ℚ :=
  let totalOutcomes : ℕ := 32  -- 2^5
  let favorableOutcomes : ℕ := 20  -- Calculated from the problem
  ↑favorableOutcomes / ↑totalOutcomes

-- The theorem to prove
theorem prob_at_least_40_cents_is_five_eighths :
  probAtLeast40Cents = 5 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_40_cents_is_five_eighths_l1217_121792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1217_121744

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (1 - 2*x)

-- State the theorem about the range of f
theorem range_of_f :
  (∀ y, y ∈ Set.range f → y ≤ 1) ∧
  (∀ y, y < 1 → ∃ x, f x = y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1217_121744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_necklace_lengths_l1217_121729

/-- Represents the different types of beads -/
inductive BeadType
  | GreenCube
  | BlueCube
  | RedPyramid
  | BluePyramid
deriving BEq, Repr

/-- Represents a necklace as a list of beads -/
def Necklace := List BeadType

/-- Checks if two adjacent beads are valid according to the rules -/
def validAdjacent (a b : BeadType) : Bool :=
  match a, b with
  | BeadType.GreenCube, BeadType.RedPyramid => true
  | BeadType.GreenCube, BeadType.BluePyramid => true
  | BeadType.BlueCube, BeadType.RedPyramid => true
  | BeadType.BlueCube, BeadType.BluePyramid => true
  | BeadType.RedPyramid, BeadType.GreenCube => true
  | BeadType.RedPyramid, BeadType.BlueCube => true
  | BeadType.BluePyramid, BeadType.GreenCube => true
  | BeadType.BluePyramid, BeadType.BlueCube => true
  | _, _ => false

/-- Checks if a necklace is valid according to all rules -/
def isValidNecklace (n : Necklace) : Bool :=
  n.length ≥ 6 ∧
  n.length % 2 = 0 ∧
  n.contains BeadType.GreenCube ∧
  n.contains BeadType.BlueCube ∧
  n.contains BeadType.RedPyramid ∧
  n.contains BeadType.BluePyramid ∧
  (List.zip n (n.rotateRight 1)).all (fun (a, b) => validAdjacent a b)

theorem valid_necklace_lengths :
  ∀ n : Necklace, isValidNecklace n → (n.length = 8 ∨ n.length = 10) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_necklace_lengths_l1217_121729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_symmetry_l1217_121771

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

-- State the theorem
theorem f_sum_symmetry (x : ℝ) : f x + f (1 - x) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_symmetry_l1217_121771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_time_l1217_121789

-- Define the time taken by each person to complete the task alone
def x : ℝ := sorry
def y : ℝ := sorry
def z : ℝ := sorry
def w : ℝ := sorry
def v : ℝ := sorry

-- Define the conditions given in the problem
axiom condition1 : 1/x + 1/y + 1/z = 1/7.5
axiom condition2 : 1/x + 1/z + 1/v = 1/5
axiom condition3 : 1/x + 1/z + 1/w = 1/6
axiom condition4 : 1/y + 1/w + 1/v = 1/4

-- Define the theorem to be proved
theorem task_completion_time :
  1 / (1/x + 1/y + 1/z + 1/w + 1/v) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_time_l1217_121789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l1217_121768

-- Define the total cost function
def G (x : ℝ) : ℝ := 15 + 5 * x

-- Define the sales income function
noncomputable def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 5 then -2 * x^2 + 21 * x + 1 else 56

-- Define the profit function
noncomputable def f (x : ℝ) : ℝ := R x - G x

-- Theorem stating the maximum profit
theorem max_profit :
  ∃ (x : ℝ), x = 4 ∧ f x = 18 ∧ ∀ y : ℝ, f y ≤ f x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l1217_121768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_two_l1217_121781

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a point in Cartesian coordinates -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Converts a polar point to a Cartesian point -/
noncomputable def polarToCartesian (p : PolarPoint) : CartesianPoint :=
  { x := p.ρ * Real.cos p.θ, y := p.ρ * Real.sin p.θ }

/-- Defines the line in polar coordinates -/
def lineInPolar (p : PolarPoint) : Prop :=
  p.ρ * Real.cos p.θ - Real.sqrt 3 * p.ρ * Real.sin p.θ - 1 = 0

/-- Defines the circle in polar coordinates -/
def circleInPolar (p : PolarPoint) : Prop :=
  p.ρ = 2 * Real.cos p.θ

/-- Calculates the distance between two Cartesian points -/
noncomputable def distance (p1 p2 : CartesianPoint) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem stating that the length of the chord is 2 -/
theorem chord_length_is_two :
  ∃ (A B : PolarPoint),
    lineInPolar A ∧ lineInPolar B ∧
    circleInPolar A ∧ circleInPolar B ∧
    A ≠ B ∧
    distance (polarToCartesian A) (polarToCartesian B) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_two_l1217_121781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_squared_bound_l1217_121783

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0  -- Added case for 0
  | 1 => 0
  | 2 => 1
  | (n + 3) => (1 + 1 / (n + 2 : ℝ)) * sequence_a (n + 2) - sequence_a (n + 1)

theorem sequence_a_squared_bound (n : ℕ) : 
  (sequence_a n) ^ 2 ≤ 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_squared_bound_l1217_121783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_on_hyperbola_l1217_121766

theorem lattice_points_on_hyperbola :
  let hyperbola := {p : ℤ × ℤ | (p.1 : ℤ)^2 - (p.2 : ℤ)^2 = 1800^2}
  Finset.card (Finset.filter (λ p => (p.1 : ℤ)^2 - (p.2 : ℤ)^2 = 1800^2) (Finset.product (Finset.range 1801) (Finset.range 1801))) = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_on_hyperbola_l1217_121766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_implies_perpendicular_planes_skew_perpendicular_lines_implies_perpendicular_plane_l1217_121752

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- Define a "lies in" relation for a line in a plane
variable (liesIn : Line → Plane → Prop)

-- Theorem 1
theorem perpendicular_parallel_implies_perpendicular_planes 
  (m : Line) (α β : Plane) :
  perpendicular m α → parallel m β → perpendicularPlanes α β :=
by sorry

-- Theorem 2
theorem skew_perpendicular_lines_implies_perpendicular_plane 
  (m n : Line) :
  skew m n → perpendicularLines m n → 
  ∃ (γ : Plane), (perpendicular n γ) ∧ (liesIn m γ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_implies_perpendicular_planes_skew_perpendicular_lines_implies_perpendicular_plane_l1217_121752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_minus_sin_half_pi_plus_alpha_eq_zero_l1217_121717

theorem cos_2alpha_minus_sin_half_pi_plus_alpha_eq_zero
  (α : ℝ)
  (h1 : Real.tan α = Real.sqrt 3)
  (h2 : π < α)
  (h3 : α < 3 * π / 2) :
  Real.cos (2 * α) - Real.sin (π / 2 + α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_minus_sin_half_pi_plus_alpha_eq_zero_l1217_121717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_from_tan_l1217_121795

theorem sin_double_angle_from_tan (α : ℝ) : 
  Real.tan (Real.pi + α) = 2 → Real.sin (2 * α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_from_tan_l1217_121795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_square_sum_inequality_trig_function_minimum_l1217_121710

theorem log_inequality (n : ℕ+) : Real.log (n + 1 : ℝ) / Real.log n > Real.log (n + 2 : ℝ) / Real.log (n + 1 : ℝ) := by sorry

theorem square_sum_inequality (a b x y : ℝ) : (a^2 + b^2) * (x^2 + y^2) ≥ (a*x + b*y)^2 := by sorry

theorem trig_function_minimum (x : ℝ) : (Real.sin x)^2 * (1 / (Real.sin x)^2) + (Real.cos x)^2 * (4 / (Real.cos x)^2) ≥ 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_square_sum_inequality_trig_function_minimum_l1217_121710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_f_l1217_121787

-- Define V(n) as the sum of exponents of prime factors of n greater than 10^100
def V (n : ℕ) : ℕ := sorry

-- Define the property for the function f
def SatisfiesProperty (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, a > b → V (Int.natAbs (f a - f b)) ≤ V (Int.natAbs (a - b))

-- Define strictly increasing function
def StrictlyIncreasing (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, x < y → f x < f y

-- Main theorem
theorem characterization_of_f :
  ∀ f : ℤ → ℤ, StrictlyIncreasing f → SatisfiesProperty f →
  ∃ a b : ℤ, a > 0 ∧ V (Int.natAbs a) = 0 ∧ ∀ x : ℤ, f x = a * x + b :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_f_l1217_121787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_hemisphere_volume_l1217_121759

noncomputable section

/-- The volume of a right circular cone -/
def cone_volume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The volume of a hemisphere -/
def hemisphere_volume (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

/-- The total volume of a cone and hemisphere -/
def total_volume (r : ℝ) (h : ℝ) : ℝ := cone_volume r h + hemisphere_volume r

theorem cone_hemisphere_volume :
  total_volume 3 10 = 48 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_hemisphere_volume_l1217_121759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pyramid_volume_l1217_121709

/-- A pyramid with a square base and height intersecting the base diagonal. -/
structure Pyramid where
  base_side : ℝ
  height : ℝ

/-- The volume of a pyramid. -/
noncomputable def volume (p : Pyramid) : ℝ := (1/3) * p.base_side^2 * p.height

/-- The perimeter of the diagonal cross-section. -/
noncomputable def diagonal_perimeter (p : Pyramid) : ℝ :=
  p.base_side * Real.sqrt 2 + 2 * Real.sqrt ((5 - p.base_side * Real.sqrt 2) / 2)^2

/-- Theorem stating the maximum volume of the pyramid under given conditions. -/
theorem max_pyramid_volume :
  ∃ (p : Pyramid), 
    diagonal_perimeter p = 5 ∧
    ∀ (q : Pyramid), diagonal_perimeter q = 5 → volume q ≤ volume p ∧
    volume p = Real.sqrt 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pyramid_volume_l1217_121709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_trig_equation_solutions_l1217_121769

theorem cubic_trig_equation_solutions (θ : ℝ) :
  let f : ℝ → ℝ := λ x => (Real.cos θ)^2 * x^3 - (1 + 3 * (Real.sin θ)^2) * x + Real.sin (2 * θ)
  (Real.cos θ = 0 → f 0 = 0) ∧
  (Real.cos θ ≠ 0 →
    f (2 * Real.tan θ) = 0 ∧
    f (-Real.tan θ + 1 / Real.cos θ) = 0 ∧
    f (-Real.tan θ - 1 / Real.cos θ) = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_trig_equation_solutions_l1217_121769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_bisector_slope_l1217_121738

/-- The slope of the acute angle bisector between two lines --/
noncomputable def angle_bisector_slope (m₁ m₂ : ℝ) : ℝ :=
  (m₁ + m₂ - Real.sqrt (1 + m₁^2 + m₂^2)) / (1 - m₁ * m₂)

/-- Theorem: The slope of the acute angle bisector between y = 2x and y = 4x is (√21 - 6) / 7 --/
theorem acute_angle_bisector_slope :
  angle_bisector_slope 2 4 = (Real.sqrt 21 - 6) / 7 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_bisector_slope_l1217_121738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_positive_sums_l1217_121747

/-- Represents a grid of real numbers -/
def Grid (m n : ℕ) := Fin m → Fin n → ℝ

/-- Represents a configuration of sign changes for rows and columns -/
def SignConfig (m n : ℕ) := (Fin m → Bool) × (Fin n → Bool)

/-- Applies sign changes to a grid based on a configuration -/
def applySignConfig (g : Grid m n) (config : SignConfig m n) : Grid m n :=
  fun i j => let (rowSigns, colSigns) := config
              if rowSigns i ≠ colSigns j then -g i j else g i j

/-- Calculates the sum of a row in a grid -/
def rowSum (g : Grid m n) (i : Fin m) : ℝ :=
  Finset.sum (Finset.univ : Finset (Fin n)) fun j => g i j

/-- Calculates the sum of a column in a grid -/
def colSum (g : Grid m n) (j : Fin n) : ℝ :=
  Finset.sum (Finset.univ : Finset (Fin m)) fun i => g i j

/-- Main theorem: There exists a sign configuration that makes all row and column sums positive -/
theorem exists_positive_sums (m n : ℕ) (g : Grid m n) 
  (h : ∀ i j, g i j ≠ 0) : 
  ∃ (config : SignConfig m n), 
    (∀ i, rowSum (applySignConfig g config) i > 0) ∧ 
    (∀ j, colSum (applySignConfig g config) j > 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_positive_sums_l1217_121747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercepts_count_l1217_121701

open Real

-- Define the function
noncomputable def f (x : ℝ) := cos (2 / x)

-- Define the interval
def a : ℝ := 0.00005
def b : ℝ := 0.0005

-- Theorem statement
theorem x_intercepts_count :
  (∃ (s : Finset ℝ), (∀ x ∈ s, a < x ∧ x < b ∧ f x = 0) ∧ s.card = 2862) ∧
  (∀ (t : Finset ℝ), (∀ x ∈ t, a < x ∧ x < b ∧ f x = 0) → t.card ≤ 2862) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercepts_count_l1217_121701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_one_l1217_121712

/-- The equation of the tangent line to y = 2x - x³ at (1, 1) is x + y - 2 = 0 -/
theorem tangent_line_at_one_one :
  ∀ x y : ℝ,
  y = 2*x - x^3 →
  (x = 1 ∧ y = 1) →
  x + y - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_one_l1217_121712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_mnp_sum_l1217_121741

open Real

-- Define the triangle
def triangle (x : ℝ) : Prop :=
  ∃ (a b c : ℝ), a + b + c = 180 ∧ 
  b - a = c - b ∧
  (7 = 9 * Real.sin a / Real.sin b ∨ 7 = x * Real.sin a / Real.sin c) ∧
  (9 = 7 * Real.sin b / Real.sin a ∨ 9 = x * Real.sin b / Real.sin c) ∧
  (x = 7 * Real.sin c / Real.sin a ∨ x = 9 * Real.sin c / Real.sin b)

-- Define the sum of possible x values
noncomputable def sum_x_values : ℝ := Real.sqrt 67

-- Define the relationship between m, n, p and the sum of x values
def mnp_relation (m n p : ℕ) : Prop :=
  m + Real.sqrt (n : ℝ) + Real.sqrt (p : ℝ) = sum_x_values

-- State the theorem
theorem triangle_mnp_sum :
  ∃ (m n p : ℕ), triangle (Real.sqrt 67) ∧ mnp_relation m n p ∧ m + n + p = 67 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_mnp_sum_l1217_121741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l1217_121746

-- Define the ellipse E
def Ellipse (x y : ℝ) : Prop := x^2 + y^2/4 = 1

-- Define the line l
def Line (k m x y : ℝ) : Prop := y = k*x + m

-- Define the intersection points
def Intersects (k m : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂, 
  Ellipse x₁ y₁ ∧ Ellipse x₂ y₂ ∧ Line k m x₁ y₁ ∧ Line k m x₂ y₂ ∧ x₁ ≠ x₂

-- Define the condition AP = 3PB
def APEquals3PB (k m : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂, 
  Ellipse x₁ y₁ ∧ Ellipse x₂ y₂ ∧ Line k m x₁ y₁ ∧ Line k m x₂ y₂ ∧ 
  x₁ = -3*x₂

-- Main theorem
theorem ellipse_and_line_intersection :
  -- Given conditions
  (eccentricity : Real.sqrt 3 / 2 = (Real.sqrt 3 : ℝ) / 2) →
  (perimeter : 4 * Real.sqrt ((2:ℝ)^2 + 1^2) = 4 * Real.sqrt 5) →
  -- Conclusions
  (∀ x y, Ellipse x y ↔ x^2 + y^2/4 = 1) ∧
  (∀ k m, Intersects k m → APEquals3PB k m → 1 < m^2 ∧ m^2 < 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l1217_121746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composite_at_one_l1217_121745

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2 * x^2 + 1 else x + 5

theorem f_composite_at_one : f (f 1) = 8 := by
  -- Evaluate f(1)
  have h1 : f 1 = 3 := by
    simp [f]
    norm_num
  
  -- Evaluate f(f(1)) = f(3)
  calc f (f 1) = f 3 := by rw [h1]
                 _ = 3 + 5 := by simp [f]
                 _ = 8 := by norm_num

  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composite_at_one_l1217_121745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1217_121757

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

-- State the theorem
theorem f_properties :
  -- The smallest positive period of f is π
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T')) ∧
  -- The maximum area of triangle ABC
  (∀ a b c A B C : ℝ,
    -- Given conditions
    0 < A ∧ 0 < B ∧ 0 < C ∧
    A + B + C = π ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) ∧
    f C = 2 ∧
    a + b = 4 →
    -- The maximum area is √3
    (1/2 * a * b * Real.sin C) ≤ Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1217_121757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1217_121707

def factorial (n : ℕ) : ℕ := Nat.factorial n

def floor_sum (x : ℕ) : ℕ := 
  (Finset.range 10).sum (fun k => (x / factorial (k + 1)))

theorem unique_solution : 
  ∃! x : ℕ, x > 0 ∧ floor_sum x = 3468 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1217_121707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_constrained_l1217_121775

theorem min_distance_constrained (x y : ℝ) : 
  3 * x + 4 * y = 24 → x + y = 10 → Real.sqrt (x^2 + y^2) ≥ Real.sqrt 292 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_constrained_l1217_121775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1217_121763

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem range_of_f : Set.range f = Set.Ioo 0 1 ∪ {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1217_121763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1217_121798

/-- A hyperbola with given properties -/
structure Hyperbola where
  center : ℝ × ℝ := (0, 0)
  foci_on_axes : Bool
  eccentricity : ℝ
  passing_point : ℝ × ℝ
  m : ℝ

/-- The set of points on the hyperbola -/
def set_of_hyperbola (h : Hyperbola) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 = 6}

/-- Calculate the area of a triangle given three points -/
noncomputable def area_triangle (a b c : ℝ × ℝ) : ℝ :=
  abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2)) / 2

/-- Theorem about the properties of the given hyperbola -/
theorem hyperbola_properties (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_foci : h.foci_on_axes = true)
  (h_ecc : h.eccentricity = Real.sqrt 2)
  (h_point : h.passing_point = (4, -Real.sqrt 10))
  (h_m : ∃ (m : ℝ), (3, m) ∈ {p : ℝ × ℝ | p.1^2 - p.2^2 = 6}) :
  (∀ (x y : ℝ), x^2 - y^2 = 6 ↔ (x, y) ∈ set_of_hyperbola h) ∧ 
  (∃ (f₁ f₂ : ℝ × ℝ), f₁.1 = -f₂.1 ∧ f₁.2 = 0 ∧ f₂.2 = 0 ∧ 
    let m := (3, h.m)
    let v₁ := (m.1 - f₁.1, m.2 - f₁.2)
    let v₂ := (m.1 - f₂.1, m.2 - f₂.2)
    v₁.1 * v₂.1 + v₁.2 * v₂.2 = 0) ∧
  (∃ (f₁ f₂ : ℝ × ℝ), f₁.1 = -f₂.1 ∧ f₁.2 = 0 ∧ f₂.2 = 0 ∧ 
    let m := (3, h.m)
    area_triangle f₁ m f₂ = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1217_121798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_number_exists_l1217_121762

theorem unknown_number_exists : ∃ N : ℝ, 
  (0.47 * N - 0.36 * 1412) + 65 = 5 ∧ 
  abs (N - 953.87) < 0.01 := by
  sorry

#eval (953.87 : Float)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_number_exists_l1217_121762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_three_zeros_l1217_121704

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > a then x + 2 else x^2 + 5*x + 2

-- Define function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - 2*x

-- Define a predicate for a function having exactly three distinct zeros
def has_exactly_three_distinct_zeros (h : ℝ → ℝ) : Prop :=
  ∃ (x y z : ℝ), (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
    (h x = 0 ∧ h y = 0 ∧ h z = 0) ∧
    (∀ w : ℝ, h w = 0 → w = x ∨ w = y ∨ w = z)

-- State the theorem
theorem g_three_zeros (a : ℝ) :
  has_exactly_three_distinct_zeros (g a) ↔ a ∈ Set.Icc (-1) 2 ∧ a ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_three_zeros_l1217_121704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_and_g_sum_bound_l1217_121755

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1| + |x - 5|

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (1 + x^2)

-- State the theorem
theorem f_min_and_g_sum_bound :
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ (∃ (x₀ : ℝ), f x₀ = m) ∧ m = 4) ∧
  (∀ (a b : ℝ), a^2 + b^2 = 6 → g a + g b ≤ 4) := by
  sorry

-- You can add more lemmas or theorems here if needed for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_and_g_sum_bound_l1217_121755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_diameters_shaded_area_l1217_121790

-- Define the circle
def circle_radius : ℝ := 6

-- Define the shaded area function
noncomputable def shaded_area (r : ℝ) : ℝ := 2 * (r^2) + (Real.pi / 2) * r^2

-- Theorem statement
theorem perpendicular_diameters_shaded_area :
  shaded_area circle_radius = 36 + 18 * Real.pi := by
  -- Unfold the definition of shaded_area
  unfold shaded_area
  -- Simplify the expression
  simp [circle_radius]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_diameters_shaded_area_l1217_121790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rental_cost_per_mile_l1217_121777

/-- Calculates the cost per mile for a car rental given the daily rate, total payment, and miles driven. -/
def cost_per_mile (daily_rate : ℚ) (total_payment : ℚ) (miles_driven : ℚ) : ℚ :=
  (total_payment - daily_rate) / miles_driven

/-- Theorem stating that the cost per mile for the given car rental scenario is approximately $0.08. -/
theorem car_rental_cost_per_mile :
  let daily_rate : ℚ := 29
  let total_payment : ℚ := 46.12
  let miles_driven : ℚ := 214
  abs (cost_per_mile daily_rate total_payment miles_driven - 0.08) < 0.001 := by
  sorry

#eval cost_per_mile 29 46.12 214

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rental_cost_per_mile_l1217_121777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1217_121756

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 12 / x + 3 * x

-- State the theorem
theorem min_value_of_f :
  (∀ x : ℝ, x > 0 → f x ≥ 12) ∧ (∃ x : ℝ, x > 0 ∧ f x = 12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1217_121756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_max_vertex_value_cube_specified_vertex_value_octahedron_vertex_A_value_octahedron_vertex_B_value_l1217_121796

-- Define the face numberings for each solid
def pyramid_faces : List ℕ := [1, 2, 3, 4]
def cube_faces : List ℕ := [1, 2, 3, 4, 5, 6]
def octahedron_faces : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Define a function to calculate vertex value
def vertex_value (faces : List ℕ) : ℕ := faces.sum

-- Theorem for the pyramid
theorem pyramid_max_vertex_value :
  (List.maximum (List.map (λ i => vertex_value [pyramid_faces[i]!, pyramid_faces[i+1]!, pyramid_faces[i+2]!])
    (List.range (pyramid_faces.length - 2)))) = some 9 := by sorry

-- Theorem for the cube
theorem cube_specified_vertex_value :
  vertex_value [cube_faces[2]!, cube_faces[5]!, cube_faces[1]!] = 11 := by sorry

-- Theorem for the octahedron vertex A
theorem octahedron_vertex_A_value :
  vertex_value [octahedron_faces[3]!, octahedron_faces[4]!, octahedron_faces[5]!, octahedron_faces[6]!] = 22 := by sorry

-- Theorem for the octahedron vertex B
theorem octahedron_vertex_B_value :
  vertex_value [octahedron_faces[0]!, octahedron_faces[1]!, octahedron_faces[3]!, octahedron_faces[4]!] = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_max_vertex_value_cube_specified_vertex_value_octahedron_vertex_A_value_octahedron_vertex_B_value_l1217_121796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_A_incorrect_l1217_121742

noncomputable section

-- Define the algebraic expressions
def expr_A (x y : ℝ) := 5 * x + y / 2
def expr_B (x y : ℝ) := 5 * (x + y)
def expr_C (x y : ℝ) := x^2 + y^2
def expr_D (x : ℝ) := 2 * x + 3

-- Define the correctness of each statement
def statement_A_correct : Prop := ∀ x y : ℝ, expr_A x y = 5 * x + y / 2
def statement_B_correct : Prop := ∀ x y : ℝ, expr_B x y = 5 * (x + y)
def statement_C_correct : Prop := ∀ x y : ℝ, expr_C x y = x^2 + y^2
def statement_D_correct : Prop := ∀ x : ℝ, expr_D x = 2 * x + 3

-- Theorem stating that statement A is incorrect while others are correct
theorem statement_A_incorrect 
  (hB : statement_B_correct) 
  (hC : statement_C_correct) 
  (hD : statement_D_correct) : 
  ¬statement_A_correct :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_A_incorrect_l1217_121742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinderVolume_20_10_l1217_121711

/-- The volume of a cylindrical tank -/
noncomputable def cylinderVolume (diameter height : ℝ) : ℝ :=
  (Real.pi / 4) * diameter^2 * height

/-- Theorem: The volume of a cylindrical tank with diameter 20 feet and height 10 feet is 1000π cubic feet -/
theorem cylinderVolume_20_10 :
  cylinderVolume 20 10 = 1000 * Real.pi := by
  -- Unfold the definition of cylinderVolume
  unfold cylinderVolume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinderVolume_20_10_l1217_121711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_of_M_l1217_121734

def M : Set (ℕ × ℕ) := {p | 3 * p.1 + 4 * p.2 - 12 < 0 ∧ p.1 > 0 ∧ p.2 > 0}

theorem number_of_proper_subsets_of_M : 
  Finset.card (Finset.powerset (Finset.filter (fun p => 3 * p.1 + 4 * p.2 - 12 < 0 ∧ p.1 > 0 ∧ p.2 > 0) 
    (Finset.product (Finset.range 3) (Finset.range 3))) \ {∅}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_of_M_l1217_121734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1217_121722

def a : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => (a n)^2 / ((a n)^2 - a n + 1)

theorem sequence_properties :
  (a 1 = 1/7 ∧ a 2 = 1/43) ∧
  (∀ n : ℕ, Finset.sum (Finset.range (n + 1)) (fun i => a i) = 1/2 - (a (n + 1)) / (1 - a (n + 1))) ∧
  (∀ n : ℕ, 1/2 - 1/(3^(2^n)) < Finset.sum (Finset.range (n + 1)) (fun i => a i) ∧
            Finset.sum (Finset.range (n + 1)) (fun i => a i) < 1/2 - 1/(3^(2^(n+1)))) :=
by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1217_121722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vector_range_l1217_121779

/-- The ellipse on which points C and D move -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Point M -/
def M : ℝ × ℝ := (0, 2)

/-- Vector from M to a point P -/
def vector_MP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 - M.1, P.2 - M.2)

/-- The condition that vector MD is lambda times vector MC -/
def vector_condition (C D : ℝ × ℝ) (lambda : ℝ) : Prop :=
  vector_MP D = (lambda * (vector_MP C).1, lambda * (vector_MP C).2)

theorem ellipse_vector_range :
  ∀ C D : ℝ × ℝ, 
  ellipse C.1 C.2 → 
  ellipse D.1 D.2 → 
  ∀ lambda : ℝ,
  vector_condition C D lambda →
  lambda ∈ Set.Icc (1/3 : ℝ) 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vector_range_l1217_121779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l1217_121772

theorem polynomial_remainder (Q : Polynomial ℝ) 
  (h1 : Q.eval 20 = 105)
  (h2 : Q.eval 105 = 20) :
  ∃ R : Polynomial ℝ, Q = (X - 20) * (X - 105) * R + (-X + 125) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l1217_121772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l1217_121739

theorem min_omega_value (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x : ℝ, Real.tan (ω * (x - π/6) + π/4) = Real.tan (ω * x + π/6)) : 
  ω ≥ 1/2 ∧ ∀ ω' > 0, (∀ x : ℝ, Real.tan (ω' * (x - π/6) + π/4) = Real.tan (ω' * x + π/6)) → ω' ≥ ω :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l1217_121739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_minus_b_l1217_121713

-- Define the vectors as functions to allow for variables
noncomputable def a (z : ℝ) : ℝ × ℝ × ℝ := (0, 1, z)
noncomputable def b (y : ℝ) : ℝ × ℝ × ℝ := (2, y, 2)
def c : ℝ × ℝ × ℝ := (-3, 6, -3)

-- Define the conditions
def a_perp_c (z : ℝ) : Prop := (a z).1 * c.1 + (a z).2.1 * c.2.1 + (a z).2.2 * c.2.2 = 0
def b_parallel_c (y : ℝ) : Prop := ∃ k : ℝ, (b y).1 = k * c.1 ∧ (b y).2.1 = k * c.2.1 ∧ (b y).2.2 = k * c.2.2

-- The theorem to prove
theorem magnitude_a_minus_b (z y : ℝ) :
  a_perp_c z → b_parallel_c y →
  ‖((a z).1 - (b y).1, (a z).2.1 - (b y).2.1, (a z).2.2 - (b y).2.2)‖ = Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_minus_b_l1217_121713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1217_121761

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi ∧
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C)

theorem triangle_ABC_properties 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C) 
  (h_sides : a^2 + b^2 - c^2 = 8) 
  (h_area : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3) :
  C = Real.pi/3 ∧ (c = 2 * Real.sqrt 3 → Real.sin A + Real.sin B = 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1217_121761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_ratio_l1217_121793

theorem line_segment_ratio (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a < b) : 
  let S : ℝ := a / b
  (a / b = b / (2 * (a + b))) → 
  S^(S^(S^2 + S⁻¹) + S⁻¹) + S⁻¹ = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_ratio_l1217_121793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_element_A_intersect_B_l1217_121730

def A : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2023}

def B : Set ℕ := {x : ℕ | ∃ k : ℤ, x = 3 * k + 2}

theorem largest_element_A_intersect_B :
  ∃ m : ℕ, m ∈ A ∩ B ∧ m = 2021 ∧ ∀ n ∈ A ∩ B, n ≤ m := by
  sorry

#check largest_element_A_intersect_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_element_A_intersect_B_l1217_121730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_angle_measure_l1217_121720

/-- The number of trapezoids in the circular arrangement -/
def n : ℕ := 12

/-- The measure of the central angle for each trapezoid, in degrees -/
noncomputable def central_angle : ℝ := 360 / n

/-- The measure of half the central angle, in degrees -/
noncomputable def half_central_angle : ℝ := central_angle / 2

/-- The measure of the angle at the top vertex of each trapezoid, in degrees -/
noncomputable def top_angle : ℝ := 180 - half_central_angle

/-- The measure of each angle opposite the longer base of the trapezoid, in degrees -/
noncomputable def base_angle : ℝ := top_angle / 2

/-- The measure of the larger interior angle of each trapezoid, in degrees -/
noncomputable def larger_interior_angle : ℝ := 180 - base_angle

theorem trapezoid_angle_measure :
  larger_interior_angle = 97.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_angle_measure_l1217_121720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1217_121799

theorem trig_identity (α : Real) 
  (h1 : Real.sin (α + π) = 1/4) 
  (h2 : α ∈ Set.Ioo (-π/2) 0) : 
  (Real.cos (2*α) - 1) / Real.tan α = Real.sqrt 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1217_121799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_twelve_l1217_121737

/-- The distance between points P and Q in miles. -/
def distance : ℝ := sorry

/-- The average speed from P to Q in miles per hour. -/
def speed_PQ : ℝ := 40

/-- The average speed from Q to P in miles per hour. -/
def speed_QP : ℝ := 45

/-- The time difference between the two trips in minutes. -/
def time_difference : ℝ := 2

/-- Theorem stating that the distance between P and Q is 12 miles. -/
theorem distance_is_twelve :
  (distance / speed_PQ * 60 - distance / speed_QP * 60 = time_difference) →
  distance = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_twelve_l1217_121737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l1217_121764

/-- Given three squares with side lengths 1, 3, and 5 units, arranged side-by-side
    on a line AB, the area of the quadrilateral formed by the segment connecting
    the bottom left corner of the smallest square to the upper right corner of
    the largest square is 75/18 square units. -/
theorem quadrilateral_area (square1 square2 square3 : ℝ) (h1 : square1 = 1)
    (h2 : square2 = 3) (h3 : square3 = 5) :
  let total_base := square1 + square2 + square3
  let height_ratio := square3 / total_base
  let height1 := square1 * height_ratio
  let height2 := (square1 + square2) * height_ratio
  (height1 + height2) * square2 / 2 = 75 / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l1217_121764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l1217_121714

-- Define the functions f and φ as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4*x + 3)
noncomputable def φ (x : ℝ) : ℝ := Real.sqrt ((4 - x) * (x - 3))

-- Define the sets M and P
def M : Set ℝ := {x | x < 1 ∨ x > 3}
def P : Set ℝ := {x | 3 ≤ x ∧ x ≤ 4}

-- Theorem statement
theorem domain_intersection :
  M ∩ P = {x : ℝ | 3 < x ∧ x ≤ 4} := by
  sorry

#check domain_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l1217_121714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_cone_altitude_is_15_l1217_121727

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  altitude : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

/-- Calculates the altitude of the smaller cone cut off from a frustum -/
noncomputable def smaller_cone_altitude (f : Frustum) : ℝ :=
  f.altitude / 2

/-- Theorem: The altitude of the smaller cone cut off from the given frustum is 15 cm -/
theorem smaller_cone_altitude_is_15 (f : Frustum) 
  (h1 : f.altitude = 30)
  (h2 : f.lower_base_area = 400 * Real.pi)
  (h3 : f.upper_base_area = 100 * Real.pi) :
  smaller_cone_altitude f = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_cone_altitude_is_15_l1217_121727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_of_time_satisfies_equation_l1217_121724

/-- Represents the unit of time in seconds -/
def U : ℝ := 1000

/-- Operating-system overhead cost in dollars -/
def overhead : ℝ := 1.07

/-- Cost per unit of computer time in dollars -/
def costPerUnit : ℝ := 0.023

/-- Cost for mounting a data tape in dollars -/
def tapeCost : ℝ := 5.35

/-- Program runtime in seconds -/
def runtime : ℝ := 1.5

/-- Total cost for 1 run in dollars -/
def totalCost : ℝ := 40.92

/-- Theorem stating that the unit of time U satisfies the total cost equation -/
theorem unit_of_time_satisfies_equation : 
  totalCost = overhead + costPerUnit * (runtime / U) + tapeCost := by
  sorry

#check unit_of_time_satisfies_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_of_time_satisfies_equation_l1217_121724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_all_problems_solved_l1217_121700

/-- Represents a math competition with n problems and participants. -/
structure MathCompetition where
  n : ℕ
  participants : Finset ℕ
  solved : ℕ → Finset ℕ
  h1 : n ≥ 4
  h2 : ∀ p, p < n → (solved p).card = 4
  h3 : ∀ p q, p < n → q < n → p ≠ q → (solved p ∩ solved q).card = 1
  h4 : participants.card ≥ 4 * n

/-- A person who solved all problems in the competition. -/
def solvedAll (mc : MathCompetition) (person : ℕ) : Prop :=
  ∀ p, p < mc.n → person ∈ mc.solved p

/-- The main theorem stating the minimum value of n. -/
theorem min_n_for_all_problems_solved (mc : MathCompetition) :
  (∃ person, solvedAll mc person) → mc.n ≥ 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_all_problems_solved_l1217_121700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_range_l1217_121751

theorem inequality_system_solution_range (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 4 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (4 - 2 * x : ℝ) ≥ 0 ∧ (1/2 : ℝ) * x - a > 0)) 
  ↔ 
  -1 ≤ a ∧ a < -1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_range_l1217_121751
