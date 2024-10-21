import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_good_intersection_point_l774_77456

/-- A line in a plane --/
structure Line where
  -- Add necessary fields for a line
  a : ℝ
  b : ℝ
  c : ℝ

/-- An intersection point of lines --/
structure IntersectionPoint where
  point : ℝ × ℝ
  lines : Finset Line
  intersection_condition : lines.card ≥ 2

/-- A good intersection point belongs to exactly two lines --/
def GoodIntersectionPoint (p : IntersectionPoint) : Prop :=
  p.lines.card = 2

/-- The main theorem --/
theorem exists_good_intersection_point
  (lines : Finset Line)
  (intersection_points : Finset IntersectionPoint)
  (h_two_intersections : intersection_points.card ≥ 2) :
  ∃ p : IntersectionPoint, p ∈ intersection_points ∧ GoodIntersectionPoint p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_good_intersection_point_l774_77456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_k_values_l774_77488

/-- Two lines are parallel if their slopes are equal -/
noncomputable def are_parallel (a1 b1 a2 b2 : ℝ) : Prop := a1 * b2 = a2 * b1

/-- The slope of the first line -/
noncomputable def slope1 (k : ℝ) : ℝ := (k - 3) / (k - 4)

/-- The slope of the second line -/
noncomputable def slope2 (k : ℝ) : ℝ := (2 * (k - 3)) / 2

theorem parallel_lines_k_values (k : ℝ) :
  are_parallel (k - 3) (4 - k) (2 * (k - 3)) (-2) → k = 3 ∨ k = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_k_values_l774_77488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_angle_equality_l774_77423

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary functions and relations
variable (center : Circle → Point)
variable (on_circle : Point → Circle → Prop)
variable (intersect : Circle → Circle → Point → Prop)
variable (tangent_point : Circle → Point → Prop)
variable (midpoint : Point → Point → Point)
variable (angle : Point → Point → Point → ℝ)
variable (coplanar : Circle → Circle → Prop)

-- State the theorem
theorem tangent_circles_angle_equality
  (C₁ C₂ : Circle)
  (O₁ O₂ A P₁ P₂ Q₁ Q₂ M₁ M₂ : Point)
  (h_distinct : C₁ ≠ C₂)
  (h_coplanar : coplanar C₁ C₂)
  (h_center₁ : center C₁ = O₁)
  (h_center₂ : center C₂ = O₂)
  (h_intersect : intersect C₁ C₂ A)
  (h_tangent₁_P : tangent_point C₁ P₁)
  (h_tangent₁_Q : tangent_point C₁ Q₁)
  (h_tangent₂_P : tangent_point C₂ P₂)
  (h_tangent₂_Q : tangent_point C₂ Q₂)
  (h_midpoint₁ : M₁ = midpoint P₁ Q₁)
  (h_midpoint₂ : M₂ = midpoint P₂ Q₂) :
  angle O₁ A O₂ = angle M₁ A M₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_angle_equality_l774_77423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_march_walking_theorem_problem_solution_match_l774_77475

/-- Calculates the total miles walked given the number of days in a month,
    days not walked, miles per hour, and hours per walk. -/
def total_miles_walked (days_in_month : ℕ) (days_not_walked : ℕ) 
                       (miles_per_hour : ℝ) (hours_per_walk : ℝ) : ℝ :=
  (days_in_month - days_not_walked : ℝ) * miles_per_hour * hours_per_walk

theorem march_walking_theorem : 
  total_miles_walked 31 4 4 1 = 108 := by
  -- Unfold the definition of total_miles_walked
  unfold total_miles_walked
  -- Simplify the arithmetic
  simp [sub_mul, mul_assoc]
  -- Check that 27 * 4 = 108
  norm_num

#eval total_miles_walked 31 4 4 1

/-- Proof that our calculation matches the problem statement -/
theorem problem_solution_match : 
  total_miles_walked 31 4 4 1 = 108 := by
  -- Apply our previously proven theorem
  exact march_walking_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_march_walking_theorem_problem_solution_match_l774_77475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_parallel_to_axis_l774_77432

-- Define the polar coordinate system
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

-- Define the concept of a line in polar coordinates
structure PolarLine where
  equation : PolarPoint → Prop

-- Define the polar axis
def polarAxis : PolarLine := { equation := fun p => p.θ = 0 }

-- Define what it means for two lines to be parallel in polar coordinates
def parallel (l1 l2 : PolarLine) : Prop := sorry

-- Define the given point
noncomputable def givenPoint : PolarPoint := { ρ := 2, θ := Real.pi / 2 }

-- Define the line we're interested in
def ourLine : PolarLine := { equation := fun p => p.ρ * Real.sin p.θ = 2 }

-- State the theorem
theorem line_through_point_parallel_to_axis :
  ourLine.equation givenPoint ∧ parallel ourLine polarAxis := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_parallel_to_axis_l774_77432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_g_l774_77448

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 3*x else -(((-x)^2) - 3*(-x))

noncomputable def g (x : ℝ) : ℝ := f x - x + 3

theorem zeros_of_g :
  ∀ x : ℝ, g x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_g_l774_77448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_sweets_fraction_l774_77438

theorem mother_sweets_fraction (total_sweets : ℕ) (eldest_sweets : ℕ) (second_sweets : ℕ) :
  total_sweets = 27 →
  eldest_sweets = 8 →
  second_sweets = 6 →
  let youngest_sweets := eldest_sweets / 2;
  let children_sweets := eldest_sweets + youngest_sweets + second_sweets;
  let mother_sweets := total_sweets - children_sweets;
  (mother_sweets : ℚ) / total_sweets = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_sweets_fraction_l774_77438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_point_sum_l774_77422

theorem exponential_function_point_sum (a m n : ℝ) : 
  a > 0 → a ≠ 1 → (fun x ↦ a^x) m = n → m + n = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_point_sum_l774_77422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_intersects_cube_l774_77426

-- Define the cube
def Cube : Set (Fin 3 → ℝ) := {p | ∀ i, 0 ≤ p i ∧ p i ≤ 30}

-- Define points on the cube's edges
noncomputable def P : Fin 3 → ℝ := ![6, 0, 0]
noncomputable def Q : Fin 3 → ℝ := ![30, 0, 17]
noncomputable def R : Fin 3 → ℝ := ![30, 3, 30]

-- Define the theorem
theorem plane_intersects_cube :
  ∃ (a b c d : ℝ), 
    (a * P 0 + b * P 1 + c * P 2 = d) ∧ 
    (a * Q 0 + b * Q 1 + c * Q 2 = d) ∧ 
    (a * R 0 + b * R 1 + c * R 2 = d) ∧
    (∃ p : Fin 3 → ℝ, p ∈ Cube ∧ a * p 0 + b * p 1 + c * p 2 = d) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_intersects_cube_l774_77426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_l774_77464

theorem expansion_coefficient (n : ℕ) : 
  (∃ r : ℕ, (n.choose r) * ((-1 : ℤ)^r) = 15 ∧ 3*r = n) →
  (∃ r' : ℕ, 3*r' - n = 3 ∧ (-(n.choose r' : ℤ)) = -20) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_l774_77464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_problem_l774_77495

/-- Represents the number of small semicircles -/
def N : ℕ := sorry

/-- Represents the radius of each small semicircle -/
def r : ℝ := sorry

/-- The area of all small semicircles -/
noncomputable def area_small_semicircles (N : ℕ) (r : ℝ) : ℝ := N * (Real.pi * r^2 / 2)

/-- The area of the large semicircle -/
noncomputable def area_large_semicircle (N : ℕ) (r : ℝ) : ℝ := Real.pi * (N * r)^2 / 2

/-- The area between the large semicircle and small semicircles -/
noncomputable def area_between (N : ℕ) (r : ℝ) : ℝ :=
  area_large_semicircle N r - area_small_semicircles N r

/-- The theorem stating that when the ratio of areas is 1:12, N must be 13 -/
theorem semicircle_problem (N : ℕ) (r : ℝ) (h : r > 0) :
  area_small_semicircles N r / area_between N r = 1 / 12 → N = 13 := by
  sorry

#check semicircle_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_problem_l774_77495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l774_77454

-- Define the function g(x) as noncomputable
noncomputable def g (x c d : ℝ) : ℝ := (x - 3) / (x^2 + c*x + d)

-- State the theorem
theorem asymptote_sum (c d : ℝ) :
  (∀ x, x ≠ 2 ∧ x ≠ -1 → g x c d ≠ 0) ∧
  (∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ ∀ x, 0 < |x - 2| ∧ |x - 2| < δ → |g x c d| > 1/ε) ∧
  (∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ ∀ x, 0 < |x + 1| ∧ |x + 1| < δ → |g x c d| > 1/ε) →
  c + d = -3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l774_77454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l774_77460

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := |x + 8/m| + |x - 2*m|

theorem f_properties (m : ℝ) (h : m > 0) :
  (∀ x, f m x ≥ 8) ∧
  (Set.Ioo 0 1 ∪ Set.Ioi 4 : Set ℝ) = {m | f m 1 > 10} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l774_77460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_measure_8_minutes_l774_77465

/-- Represents an hourglass with a specific duration --/
structure Hourglass where
  duration : ℕ
  sand_top : ℝ
  sand_bottom : ℝ

/-- Represents the state of the timing system --/
structure TimingSystem where
  hourglass5 : Hourglass
  hourglass2 : Hourglass
  elapsed_time : ℝ

/-- Flips an hourglass, exchanging the sand between top and bottom --/
def flipHourglass (h : Hourglass) : Hourglass :=
  { h with sand_top := h.sand_bottom, sand_bottom := h.sand_top }

/-- Advances the timing system by a given amount of time --/
noncomputable def advance (s : TimingSystem) (t : ℝ) : TimingSystem :=
  { s with 
    hourglass5 := { s.hourglass5 with 
      sand_top := max 0 (s.hourglass5.sand_top - t),
      sand_bottom := min s.hourglass5.duration (s.hourglass5.sand_bottom + t) },
    hourglass2 := { s.hourglass2 with 
      sand_top := max 0 (s.hourglass2.sand_top - t),
      sand_bottom := min s.hourglass2.duration (s.hourglass2.sand_bottom + t) },
    elapsed_time := s.elapsed_time + t }

/-- The theorem stating that it's possible to measure 8 minutes --/
theorem can_measure_8_minutes :
  ∃ (initial_sand_top : ℝ),
  ∃ (sequence : List (TimingSystem → TimingSystem)),
  let initial_state : TimingSystem := {
    hourglass5 := { duration := 5, sand_top := 0, sand_bottom := 5 },
    hourglass2 := { duration := 2, sand_top := initial_sand_top, sand_bottom := 2 - initial_sand_top },
    elapsed_time := 0
  }
  let final_state := sequence.foldl (fun s f => f s) initial_state
  final_state.elapsed_time = 8 ∧ 
  (∀ (s : TimingSystem), s ∈ sequence.scanl (fun s f => f s) initial_state → 
    s.hourglass5.sand_top + s.hourglass5.sand_bottom = 5 ∧
    s.hourglass2.sand_top + s.hourglass2.sand_bottom = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_measure_8_minutes_l774_77465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_to_x_axis_l774_77486

-- Define a function f and its derivative f'
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)
variable (x₀ : ℝ)

-- Define the condition that f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Define the given condition that f'(x) = 0 for all x
axiom f'_is_zero : ∀ x, f' x = 0

-- Define TangentLine and xAxis
def TangentLine (f : ℝ → ℝ) (x₀ : ℝ) : Set ℝ := sorry
def xAxis : Set ℝ := {x : ℝ | x ∈ Set.univ}

-- Define parallel relation for sets
def Parallel (A B : Set ℝ) : Prop := sorry

infix:50 " ∥ " => Parallel

-- Theorem: If f'(x) = 0, then the tangent line at (x₀, f(x₀)) is parallel to the x-axis
theorem tangent_line_parallel_to_x_axis :
  (∀ x, f' x = 0) → TangentLine f x₀ ∥ xAxis :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_to_x_axis_l774_77486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_concentration_of_first_solution_l774_77466

/-- Represents a salt solution with a given volume and salt concentration -/
structure SaltSolution where
  volume : ℝ
  concentration : ℝ

/-- Represents a mixture of two salt solutions -/
noncomputable def mixSolutions (s1 s2 : SaltSolution) : SaltSolution :=
  { volume := s1.volume + s2.volume,
    concentration := (s1.volume * s1.concentration + s2.volume * s2.concentration) / (s1.volume + s2.volume) }

theorem salt_concentration_of_first_solution 
  (solution1 solution2 finalMixture : SaltSolution)
  (h1 : solution2.concentration = 0.12)
  (h2 : solution1.volume = 600)
  (h3 : finalMixture.volume = 1000)
  (h4 : finalMixture.concentration = 0.084)
  (h5 : finalMixture = mixSolutions solution1 solution2) :
  solution1.concentration = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_concentration_of_first_solution_l774_77466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pole_height_is_four_meters_l774_77444

/-- Represents the geometry of a telephone pole supported by a cable -/
structure TelephonePoleSetup where
  /-- The horizontal distance from the pole to the cable attachment point -/
  cable_ground_distance : ℝ
  /-- The distance from the pole where a person stands -/
  person_distance : ℝ
  /-- The height of the person -/
  person_height : ℝ

/-- Calculates the height of the telephone pole given the setup -/
noncomputable def pole_height (setup : TelephonePoleSetup) : ℝ :=
  (setup.person_height * setup.cable_ground_distance) / (setup.cable_ground_distance - setup.person_distance)

/-- Theorem stating that for the given measurements, the pole height is 4 meters -/
theorem pole_height_is_four_meters :
  let setup : TelephonePoleSetup := {
    cable_ground_distance := 5,
    person_distance := 3,
    person_height := 1.6
  }
  pole_height setup = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pole_height_is_four_meters_l774_77444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_card_probability_l774_77478

theorem yellow_card_probability (total_cards : ℕ) (green_green : ℕ) (green_yellow : ℕ) (yellow_yellow : ℕ)
  (h_total : total_cards = green_green + green_yellow + yellow_yellow)
  (h_gg : green_green = 4)
  (h_gy : green_yellow = 2)
  (h_yy : yellow_yellow = 2) :
  (2 * yellow_yellow : ℚ) / (green_yellow + 2 * yellow_yellow) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_card_probability_l774_77478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slipper_cost_l774_77497

/-- Calculates the total cost of slippers with embroidery and shipping --/
theorem slipper_cost (original_price discount_percent embroidery_cost_per_shoe shipping_cost : ℝ) :
  original_price = 50 →
  discount_percent = 10 →
  embroidery_cost_per_shoe = 5.5 →
  shipping_cost = 10 →
  let discounted_price := original_price * (1 - discount_percent / 100)
  let total_embroidery_cost := embroidery_cost_per_shoe * 2
  let total_cost := discounted_price + total_embroidery_cost + shipping_cost
  total_cost = 66 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slipper_cost_l774_77497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l774_77433

/-- Given a parabola y² = 2px (p > 0) and a line passing through its focus
    intersecting the parabola at points A and B with |AB| = 4,
    prove that the range of p is (0, 2). -/
theorem parabola_chord_length (p : ℝ) (A B : ℝ × ℝ) :
  p > 0 →
  (∀ y : ℝ, y^2 = 2*p*A.1) →  -- Parabola equation for point A
  (∀ y : ℝ, y^2 = 2*p*B.1) →  -- Parabola equation for point B
  (∃ k : ℝ, A.2 = k*(A.1 - p/2) ∧ B.2 = k*(B.1 - p/2)) →  -- Line passes through focus
  ‖(A.1, A.2) - (B.1, B.2)‖ = 4 →  -- Distance between A and B is 4
  0 < p ∧ p < 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l774_77433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_circle_l774_77441

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define point P
def P : ℝ × ℝ := (-2, -3)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_to_circle :
  ∃ (min_dist : ℝ), ∀ (Q : ℝ × ℝ), circle_eq Q.1 Q.2 →
    distance P Q ≥ min_dist ∧
    ∃ (Q_min : ℝ × ℝ), circle_eq Q_min.1 Q_min.2 ∧ distance P Q_min = min_dist ∧
    min_dist = 3 * Real.sqrt 2 - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_circle_l774_77441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sum_implies_integer_l774_77434

theorem floor_sum_implies_integer (a b c : ℝ) :
  (∀ n : ℕ, ⌊n * a⌋ + ⌊n * b⌋ = ⌊n * c⌋) →
  (Int.floor a = a ∨ Int.floor b = b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sum_implies_integer_l774_77434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_congruence_l774_77402

def arithmetic_sequence (a₁ n d : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

theorem product_congruence (n : ℕ) :
  (arithmetic_sequence 7 20 10).prod % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_congruence_l774_77402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_golden_ratio_l774_77414

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The left focus of an ellipse -/
noncomputable def left_focus (e : Ellipse) : ℝ × ℝ :=
  (-Real.sqrt (e.a^2 - e.b^2), 0)

/-- The left vertex of an ellipse -/
def left_vertex (e : Ellipse) : ℝ × ℝ :=
  (-e.a, 0)

/-- The upper vertex of an ellipse -/
def upper_vertex (e : Ellipse) : ℝ × ℝ :=
  (0, e.b)

/-- The angle between three points -/
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem ellipse_eccentricity_golden_ratio (e : Ellipse) :
  let f := left_focus e
  let m := left_vertex e
  let n := upper_vertex e
  angle f m n + π/2 = angle m f n →
  eccentricity e = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_golden_ratio_l774_77414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonzero_terms_count_l774_77461

/-- The number of nonzero terms in the expansion of (2x^3 + 5x - 2)(4x^2 - x + 1) - 4(x^3 - 3x^2 + 2) -/
theorem nonzero_terms_count : ∃ (p : Polynomial ℤ), 
  p = (2 * X^3 + 5 * X - 2) * (4 * X^2 - X + 1) - 4 * (X^3 - 3 * X^2 + 2) ∧ 
  (Finset.filter (fun i => p.coeff i ≠ 0) (Finset.range (p.natDegree + 1))).card = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonzero_terms_count_l774_77461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l774_77467

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  xFocused : Bool

/-- Checks if a point (x, y) lies on the hyperbola -/
def Hyperbola.containsPoint (h : Hyperbola) (x y : ℝ) : Prop :=
  if h.xFocused then
    x^2 / h.a^2 - y^2 / h.b^2 = 1
  else
    y^2 / h.a^2 - x^2 / h.b^2 = 1

/-- The focal length of a hyperbola -/
noncomputable def Hyperbola.focalLength (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2)

theorem hyperbola_equation (c1 c2 : Hyperbola) : 
  c2.a^2 = 7 ∧ c2.b^2 = 1 ∧ c2.xFocused = true →
  c1.focalLength = c2.focalLength →
  c1.containsPoint 3 1 →
  (c1.a^2 = 6 ∧ c1.b^2 = 2 ∧ c1.xFocused = true) ∨
  (c1.a^2 = 9 - Real.sqrt 73 ∧ c1.b^2 = Real.sqrt 73 - 1 ∧ c1.xFocused = false) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l774_77467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l774_77459

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * 2^x + a - 2) / (2^x + 1)

theorem f_properties (a : ℝ) :
  (∀ x : ℝ, f a (-x) + f a x = 0) →
  (a = 1 ∧ 
   f a 3 = 7/9 ∧ 
   ∀ x y : ℝ, x < y → f a x < f a y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l774_77459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l774_77420

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the foci
noncomputable def F₁ : ℝ × ℝ := sorry
noncomputable def F₂ : ℝ × ℝ := sorry

-- Define a point on the hyperbola
noncomputable def P : ℝ × ℝ := sorry

-- State that P is on the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2

-- Define the angle F₁PF₂
noncomputable def angle_F₁PF₂ : ℝ := sorry

-- State that the angle is 60°
axiom angle_is_60 : angle_F₁PF₂ = Real.pi / 3

-- Define the area of triangle F₁PF₂
noncomputable def area_F₁PF₂ : ℝ := sorry

-- Theorem to prove
theorem area_of_triangle : area_F₁PF₂ = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l774_77420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_and_intersection_l774_77462

/-- A line passing through the origin -/
structure OriginLine where
  slope : ℝ

/-- A circle with equation x^2 + y^2 - 8x + 12 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 - 8*p.1 + 12 = 0}

/-- The trajectory of the midpoint of a chord of the circle -/
def MidpointTrajectory : Set (ℝ × ℝ) :=
  {p | (p.1 - 2)^2 + p.2^2 = 4 ∧ 3 < p.1 ∧ p.1 ≤ 4}

/-- The set of k values for which y = k(x-5) intersects the midpoint trajectory at exactly one point -/
def ValidKSet : Set ℝ :=
  {k | k ∈ Set.Icc (-Real.sqrt 3 / 2) (Real.sqrt 3 / 2) ∨ k = -2*Real.sqrt 5 / 5 ∨ k = 2*Real.sqrt 5 / 5}

theorem midpoint_trajectory_and_intersection (l : OriginLine) :
  ∃ A B : ℝ × ℝ, A ∈ Circle ∧ B ∈ Circle ∧ A ≠ B ∧
  (∀ P : ℝ × ℝ, P ∈ MidpointTrajectory ↔ P = ((A.1 + B.1)/2, (A.2 + B.2)/2)) ∧
  (∀ k : ℝ, (∃! P : ℝ × ℝ, P ∈ MidpointTrajectory ∧ P.2 = k*(P.1 - 5)) ↔ k ∈ ValidKSet) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_and_intersection_l774_77462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_symmetry_axes_l774_77463

/-- Helper function to represent the concept of distance between adjacent symmetry axes. -/
noncomputable def distance_between_adjacent_symmetry_axes (f : ℝ → ℝ) : ℝ := 
  sorry

/-- Given a function f(x) = sin(ωx + π/3) where the distance between
    two adjacent symmetry axes is π, prove that ω = 1. -/
theorem sine_function_symmetry_axes (ω : ℝ) : 
  (∃ f : ℝ → ℝ, f = λ x => Real.sin (ω * x + π / 3)) → 
  (∃ d : ℝ, d = π ∧ d = distance_between_adjacent_symmetry_axes (λ x => Real.sin (ω * x + π / 3))) →
  ω = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_symmetry_axes_l774_77463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_property_characterization_l774_77452

/-- An integer n is good if |n| is not the square of an integer -/
def IsGood (n : ℤ) : Prop :=
  ∀ k : ℤ, |n| ≠ k^2

/-- The property that m can be represented in infinitely many ways as a sum of three distinct good integers whose product is the square of an odd integer -/
def HasProperty (m : ℤ) : Prop :=
  ∃ f : ℕ → (ℤ × ℤ × ℤ),
    (∀ i : ℕ, let (u, v, w) := f i;
      m = u + v + w ∧
      IsGood u ∧ IsGood v ∧ IsGood w ∧
      u ≠ v ∧ u ≠ w ∧ v ≠ w ∧
      ∃ k : ℤ, u * v * w = (2 * k + 1)^2) ∧
    (∀ i j : ℕ, i ≠ j → f i ≠ f j)

theorem property_characterization (m : ℤ) :
  HasProperty m ↔ ∃ k : ℤ, m = 3 + 4 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_property_characterization_l774_77452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_phase_shift_l774_77482

/-- The phase shift of a sinusoidal function y = sin(bx - c) is c/b -/
noncomputable def phase_shift (b c : ℝ) : ℝ := c / b

/-- The sine function we're analyzing -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (5 * x - 2 * Real.pi)

theorem sine_phase_shift :
  phase_shift 5 (2 * Real.pi) = (2 * Real.pi) / 5 :=
by
  -- Unfold the definition of phase_shift
  unfold phase_shift
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_phase_shift_l774_77482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_equation_solution_oplus_equation_range_l774_77421

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ := 2 * a - 3/2 * (a + b)

-- Part 1
theorem oplus_equation_solution :
  ∀ x : ℝ, oplus x 4 = 0 → x = 12 := by
  intro x h
  -- Proof steps would go here
  sorry

-- Part 2
theorem oplus_equation_range :
  ∀ x m : ℝ, x ≥ 0 → oplus x m = oplus (-2) (x + 4) → m ≥ 14/3 := by
  intro x m hx h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_equation_solution_oplus_equation_range_l774_77421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_from_ellipse_l774_77469

-- Define the ellipse
noncomputable def ellipse_equation (x y : ℝ) : Prop := 4 * x^2 + y^2 = 64

-- Define the eccentricity of the ellipse
noncomputable def ellipse_eccentricity : ℝ := Real.sqrt 3 / 2

-- Define the common focus
noncomputable def common_focus : ℝ × ℝ := (0, 4 * Real.sqrt 3)

-- Define the hyperbola
def hyperbola_equation (x y : ℝ) : Prop := y^2 - 3 * x^2 = 36

-- Define the theorem
theorem hyperbola_from_ellipse :
  ∀ x y : ℝ,
  (∃ f : ℝ × ℝ, f = common_focus) →
  (∃ e : ℝ, e * ellipse_eccentricity = 1) →
  hyperbola_equation x y :=
by
  sorry

#check hyperbola_from_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_from_ellipse_l774_77469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_axis_distance_l774_77404

-- Define the parabola
noncomputable def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- Define the axis of symmetry
noncomputable def axis_of_symmetry (a : ℝ) : ℝ := -1 / (4 * a)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x y : ℝ) (a : ℝ) : ℝ := |x + axis_of_symmetry a|

-- Theorem statement
theorem parabola_axis_distance (a : ℝ) : 
  distance_to_line 1 1 a = 2 ↔ a = 1/4 ∨ a = -1/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_axis_distance_l774_77404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_marks_l774_77407

/-- Calculates the average marks for a class given specific score distributions -/
theorem class_average_marks (total_students : ℕ) 
  (high_score_students : ℕ) (high_score : ℕ) 
  (mid_score_students : ℕ) (mid_score_diff : ℕ)
  (low_score : ℕ) : 
  total_students = 50 →
  high_score_students = 10 →
  mid_score_students = 15 →
  high_score = 90 →
  mid_score_diff = 10 →
  low_score = 60 →
  (high_score_students * high_score + 
   mid_score_students * (high_score - mid_score_diff) + 
   (total_students - high_score_students - mid_score_students) * low_score) / total_students = 72 := by
  intro h1 h2 h3 h4 h5 h6
  sorry

#check class_average_marks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_marks_l774_77407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l774_77439

theorem trigonometric_identity (α : ℝ) : 
  Real.sin α ^ 2 + Real.cos (Real.pi + α) * Real.cos (Real.pi - α) + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l774_77439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_is_3_l774_77490

/-- The decimal representation of 1/13 -/
def decimal_rep : ℚ := 1 / 13

/-- The repeating block in the decimal representation of 1/13 -/
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

/-- The length of the repeating block -/
def block_length : ℕ := List.length repeating_block

/-- The position we're interested in (150th digit after the decimal point) -/
def target_position : ℕ := 150

/-- The function that returns the nth digit after the decimal point -/
def nth_digit (n : ℕ) : ℕ := 
  repeating_block[((n - 1) % block_length)]'(by {
    simp [block_length]
    apply Nat.mod_lt
    exact Nat.zero_lt_succ _
  })

theorem digit_150_is_3 : 
  nth_digit target_position = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_is_3_l774_77490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_minus_1_equals_sin_2x_plus_pi_over_2_minus_1_l774_77445

theorem cos_2x_minus_1_equals_sin_2x_plus_pi_over_2_minus_1 :
  ∀ x : ℝ, Real.cos (2 * x) - 1 = Real.sin (2 * (x + Real.pi / 4)) - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_minus_1_equals_sin_2x_plus_pi_over_2_minus_1_l774_77445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l774_77446

/-- The speed of a train in km/h given its length and time to pass a stationary point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem stating that a 100-meter long train crossing a point in 5 seconds has a speed of 72 km/h -/
theorem train_speed_calculation :
  train_speed 100 5 = 72 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  simp [mul_div_assoc, mul_comm]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l774_77446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_midpoints_constant_sum_x_coordinates_q3_l774_77400

/-- Given a list of n real numbers, returns a new list of n real numbers
    where each element is the average of two adjacent elements from the original list,
    wrapping around at the end. -/
noncomputable def midpoints (xs : List ℝ) : List ℝ :=
  let n := xs.length
  List.range n |>.map (fun i => (xs[i]! + xs[(i+1) % n]!) / 2)

/-- The sum of x-coordinates remains constant when taking midpoints of a polygon -/
theorem sum_midpoints_constant (n : ℕ) (xs : List ℝ) (hn : xs.length = n) :
  xs.sum = (midpoints xs).sum := by sorry

theorem sum_x_coordinates_q3 (xs : List ℝ) (h120 : xs.length = 120) (hsum : xs.sum = 2500) :
  ((midpoints (midpoints xs)).sum = 2500) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_midpoints_constant_sum_x_coordinates_q3_l774_77400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equality_l774_77418

theorem trigonometric_expression_equality : 
  2 * |1 - Real.sin (60 * π / 180)| + Real.tan (45 * π / 180) / (1 / Real.tan (30 * π / 180) - 2 * Real.cos (45 * π / 180)) = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equality_l774_77418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mechanism_unique_config_l774_77411

/-- Represents a mechanism with large and small parts. -/
structure Mechanism where
  total : ℕ
  large : ℕ
  small : ℕ
  total_eq : total = large + small
  small_in_12 : ∀ (selection : Finset ℕ), selection.card = 12 → ∃ x ∈ selection, x < small
  large_in_20 : ∀ (selection : Finset ℕ), selection.card = 20 → ∃ x ∈ selection, x ≥ small

/-- The only valid configuration for the mechanism is 11 large parts and 19 small parts. -/
theorem mechanism_unique_config :
  ∀ (m : Mechanism), m.total = 30 → m.large = 11 ∧ m.small = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mechanism_unique_config_l774_77411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_equality_l774_77491

/-- Set of all ways to insert 3 pairs of parentheses into a1a2a3a4a5 -/
def A : Finset (List ℕ) :=
  sorry

/-- Set of all ways to partition a convex hexagon into 4 triangles -/
def B : Finset (List (List ℕ)) :=
  sorry

/-- Set of all ways to arrange 4 black and 4 white balls in a row
    with white balls never less than black balls at any position -/
def C : Finset (List Bool) :=
  sorry

/-- The cardinalities of sets A, B, and C are equal -/
theorem cardinality_equality : Finset.card A = Finset.card B ∧ Finset.card B = Finset.card C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_equality_l774_77491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_properties_l774_77440

/-- Sample data type -/
structure SampleData where
  x : ℝ
  y : ℝ

/-- Linear regression model -/
structure LinearRegression where
  samples : List SampleData
  b_hat : ℝ
  a_hat : ℝ
  r_squared : ℝ
  r : ℝ

/-- Mean of x values -/
noncomputable def x_mean (model : LinearRegression) : ℝ := sorry

/-- Mean of y values -/
noncomputable def y_mean (model : LinearRegression) : ℝ := sorry

/-- Predicted y value for a given x -/
def y_hat (model : LinearRegression) (x : ℝ) : ℝ :=
  model.b_hat * x + model.a_hat

/-- Sum of squared residuals -/
noncomputable def sum_squared_residuals (model : LinearRegression) : ℝ := sorry

/-- Predicate for better fit -/
def fits_better (m₁ m₂ : LinearRegression) : Prop := sorry

/-- Predicate for stronger linear correlation -/
def stronger_linear_correlation (m₁ m₂ : LinearRegression) : Prop := sorry

/-- Linear regression properties -/
theorem linear_regression_properties (model : LinearRegression) :
  (y_hat model (x_mean model) = y_mean model) ∧
  (∀ m₁ m₂ : LinearRegression, sum_squared_residuals m₁ < sum_squared_residuals m₂ →
    fits_better m₁ m₂) ∧
  (∀ m₁ m₂ : LinearRegression, m₁.r_squared > m₂.r_squared →
    fits_better m₁ m₂) ∧
  (∀ m₁ m₂ : LinearRegression, |m₁.r| > |m₂.r| →
    stronger_linear_correlation m₁ m₂) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_properties_l774_77440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_sales_calc_l774_77499

def initial_stock : ℕ := 900
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def unsold_percentage : ℚ := 55333333333333336 / 100000000000000000

theorem friday_sales_calc : 
  ∃ (friday_sales : ℕ), 
    friday_sales = initial_stock - 
      (Int.toNat ⌊(unsold_percentage * initial_stock)⌋) - 
      (monday_sales + tuesday_sales + wednesday_sales + thursday_sales) ∧
    friday_sales = 135 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_sales_calc_l774_77499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_iff_equal_tangents_l774_77492

/-- A closed convex curve in the plane. -/
structure ClosedConvexCurve where
  /-- The set of points on the curve. -/
  points : Set (ℝ × ℝ)
  /-- Proof that the curve is closed. -/
  closed : IsClosed points
  /-- Proof that the curve is convex. -/
  convex : Convex ℝ points

/-- A tangent line to a curve at a point. -/
noncomputable def TangentLine (curve : ClosedConvexCurve) (point : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The length of a tangent line segment from an external point to the curve. -/
noncomputable def TangentLength (curve : ClosedConvexCurve) (externalPoint : ℝ × ℝ) : ℝ := sorry

/-- Predicate stating that all tangent lengths from an external point are equal. -/
def AllTangentLengthsEqual (curve : ClosedConvexCurve) : Prop :=
  ∀ (p : ℝ × ℝ), p ∉ curve.points →
    ∀ (q r : ℝ × ℝ), q ∈ curve.points → r ∈ curve.points →
      TangentLength curve p = TangentLength curve p

/-- Definition of a circle in the plane. -/
def IsCircle (curve : ClosedConvexCurve) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ∀ (p : ℝ × ℝ), p ∈ curve.points ↔ dist p center = radius

/-- The main theorem: a closed convex curve is a circle if and only if
    all tangent lengths from any external point are equal. -/
theorem circle_iff_equal_tangents (curve : ClosedConvexCurve) :
  IsCircle curve ↔ AllTangentLengthsEqual curve := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_iff_equal_tangents_l774_77492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_equals_2_l774_77443

-- Define the sequence a_n
def a : ℕ → ℚ
  | 0 => -1  -- Define for 0 to cover all natural numbers
  | 1 => -1
  | n+2 => 1 - 1 / a (n+1)

-- State the theorem
theorem a_2018_equals_2 : a 2018 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_equals_2_l774_77443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_sum_digits_l774_77451

/-- A number is a palindrome if it reads the same backwards as forwards. -/
def isPalindrome (n : Nat) : Prop := n.digits 10 = (n.digits 10).reverse

/-- The sum of digits of a natural number. -/
def digitSum (n : Nat) : Nat := n.digits 10 |>.sum

/-- Theorem: If x is a three-digit palindrome and x + 32 is a four-digit palindrome, 
    then the sum of the digits of x is 24. -/
theorem palindrome_sum_digits : 
  ∀ x : Nat, 
  (100 ≤ x ∧ x < 1000) → 
  isPalindrome x → 
  isPalindrome (x + 32) → 
  (1000 ≤ x + 32 ∧ x + 32 < 10000) → 
  digitSum x = 24 := by
  sorry

#eval digitSum 969  -- This should output 24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_sum_digits_l774_77451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cut_theorem_l774_77470

/-- Given a square with side length a, prove that it can be cut into 4 parts
    that can form two squares with side lengths a/2 and a/√2. -/
theorem square_cut_theorem (a : ℝ) (h : a > 0) :
  ∃ (s₁ s₂ s₃ s₄ : Set (ℝ × ℝ)),
    let original := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ a ∧ 0 ≤ p.2 ∧ p.2 ≤ a}
    let result₁ := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ a/2 ∧ 0 ≤ p.2 ∧ p.2 ≤ a/2}
    let result₂ := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ a/Real.sqrt 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ a/Real.sqrt 2}
    (s₁ ∪ s₂ ∪ s₃ ∪ s₄ = original) ∧
    (s₁ ∩ s₂ = ∅) ∧ (s₁ ∩ s₃ = ∅) ∧ (s₁ ∩ s₄ = ∅) ∧
    (s₂ ∩ s₃ = ∅) ∧ (s₂ ∩ s₄ = ∅) ∧ (s₃ ∩ s₄ = ∅) ∧
    ∃ (f : (ℝ × ℝ) → (ℝ × ℝ)), Function.Bijective f ∧
      f '' (s₁ ∪ s₂) = result₁ ∧
      f '' (s₃ ∪ s₄) = result₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cut_theorem_l774_77470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_length_l774_77429

theorem right_triangle_length (AC : ℝ) (slope : ℝ) (h1 : AC = 25) (h2 : slope = 4/3) :
  ∃ AB : ℝ, AB^2 + (slope * AB)^2 = AC^2 ∧ AB = 15 := by
  use 15
  constructor
  · calc
      15^2 + (slope * 15)^2 = 15^2 + ((4/3) * 15)^2 := by rw [h2]
      _ = 225 + 400 := by ring
      _ = 625 := by ring
      _ = 25^2 := by ring
      _ = AC^2 := by rw [h1]
  · rfl

#check right_triangle_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_length_l774_77429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_corresponding_angles_congruence_false_l774_77413

-- Define a structure for Triangle if it's not already defined in the imported library
structure Triangle where
  -- You might need to add appropriate fields here, e.g., vertices or angles
  -- For now, we'll leave it empty as a placeholder

-- Define an angle function for Triangle if it's not already defined
def angle (T : Triangle) (i : Fin 3) : Real := sorry

-- Define congruence for triangles
def congruent (T1 T2 : Triangle) : Prop := sorry

infix:50 " ≅ " => congruent

theorem inverse_corresponding_angles_congruence_false :
  ¬(∀ (T1 T2 : Triangle), 
    (∀ (i : Fin 3), angle T1 i = angle T2 i) → 
    T1 ≅ T2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_corresponding_angles_congruence_false_l774_77413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_calculation_l774_77401

/-- The volume of a cone with slant height 1 and central angle 120° in its lateral surface development -/
noncomputable def coneVolume : ℝ := (2 * Real.sqrt 2 / 81) * Real.pi

/-- Theorem stating that the volume of a cone with given properties is equal to the calculated value -/
theorem cone_volume_calculation (slantHeight : ℝ) (centralAngle : ℝ) :
  slantHeight = 1 →
  centralAngle = 2 * Real.pi / 3 →
  coneVolume = (1 / 3) * Real.pi * (1 / 3)^2 * (2 * Real.sqrt 2 / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_calculation_l774_77401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jims_purchase_cost_is_36_l774_77480

/-- The cost of Jim's purchase at Kara's Kafe -/
def jimsPurchaseCost (sandwichPrice sodaPrice : ℚ) (sandwichCount sodaCount : ℕ) 
  (discountRate : ℚ) (discountThreshold : ℚ) : ℚ :=
  let initialTotal := sandwichPrice * sandwichCount + sodaPrice * sodaCount
  if initialTotal > discountThreshold then
    initialTotal * (1 - discountRate)
  else
    initialTotal

/-- Theorem stating that Jim's purchase cost is $36 -/
theorem jims_purchase_cost_is_36 :
  jimsPurchaseCost 4 3 7 4 (1/10) 30 = 36 := by
  -- Unfold the definition of jimsPurchaseCost
  unfold jimsPurchaseCost
  -- Simplify the arithmetic expressions
  simp
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jims_purchase_cost_is_36_l774_77480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_definitive_conclusion_l774_77412

/-- Two angles in a plane -/
structure AnglePair where
  angle1 : Real
  angle2 : Real

/-- Predicate to check if two angles are congruent -/
def are_congruent (ap : AnglePair) : Prop :=
  ap.angle1 = ap.angle2

/-- Representation of a line in 2D space -/
structure Line where
  slope : Real
  intercept : Real

/-- Predicate to check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Predicate to check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- Predicate to check if one pair of corresponding sides is parallel -/
def one_pair_parallel (ap : AnglePair) : Prop :=
  ∃ (side1 side2 : Line), parallel side1 side2

/-- Theorem stating that no definitive conclusion can be made about the other pair of sides -/
theorem no_definitive_conclusion (ap : AnglePair) 
  (h1 : are_congruent ap) 
  (h2 : one_pair_parallel ap) : 
  ¬ (∀ (side1 side2 : Line), parallel side1 side2 ∨ ¬parallel side1 side2 ∨ perpendicular side1 side2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_definitive_conclusion_l774_77412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_old_machine_rate_proof_l774_77473

/-- The rate of the new machine in bolts per hour -/
noncomputable def new_machine_rate : ℝ := 150

/-- The time both machines work together in hours -/
noncomputable def work_time : ℝ := 108 / 60

/-- The total number of bolts produced by both machines -/
noncomputable def total_bolts : ℝ := 450

/-- The rate of the old machine in bolts per hour -/
noncomputable def old_machine_rate : ℝ := 100

/-- Proof that the combined rate of both machines multiplied by the work time equals the total bolts produced -/
theorem old_machine_rate_proof :
  (old_machine_rate + new_machine_rate) * work_time = total_bolts :=
by
  -- We use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_old_machine_rate_proof_l774_77473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logo_shaded_area_l774_77458

/-- The shaded area of a logo consisting of a square with side length 24 inches
    and five circles (one at each corner and one in the center), where each circle
    is tangent to two sides of the square and adjacent circles. -/
noncomputable def shaded_area : ℝ := 576 - 180 * Real.pi

/-- Theorem stating that the shaded area of the logo is 576 - 180π square inches. -/
theorem logo_shaded_area :
  let square_side : ℝ := 24
  let num_circles : ℕ := 5
  let circle_radius : ℝ := square_side / 4
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := Real.pi * circle_radius ^ 2
  square_area - num_circles * circle_area = shaded_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logo_shaded_area_l774_77458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_coordinates_l774_77479

/-- Given an ellipse with equation x²/16 + y²/9 = 1, its foci are located at (-√7, 0) and (√7, 0) -/
theorem ellipse_foci_coordinates :
  let ellipse := {p : ℝ × ℝ | p.1^2/16 + p.2^2/9 = 1}
  ∃ (f₁ f₂ : ℝ × ℝ), f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧ 
    f₁ = (-Real.sqrt 7, 0) ∧ f₂ = (Real.sqrt 7, 0) ∧
    ∀ (p : ℝ × ℝ), p ∈ ellipse → 
      dist p f₁ + dist p f₂ = 2 * 4 :=
by
  sorry

where
  dist (p₁ p₂ : ℝ × ℝ) : ℝ := 
    Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_coordinates_l774_77479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_average_speed_l774_77450

noncomputable section

-- Define the motion function
def s (t : ℝ) : ℝ := t^2 + 3

-- Define the average speed function
def average_speed (t1 t2 : ℝ) : ℝ := (s t2 - s t1) / (t2 - t1)

-- Theorem statement
theorem particle_average_speed (Δt : ℝ) (h : Δt ≠ 0) : 
  average_speed 3 (3 + Δt) = 6 + Δt := by
  -- Expand the definition of average_speed
  unfold average_speed
  -- Expand the definition of s
  unfold s
  -- Simplify the expression
  simp [h]
  -- The proof is complete
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_average_speed_l774_77450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chips_required_l774_77424

/-- Represents a hexagonal cell in the grid -/
structure HexCell where
  id : Nat
  neighbors : Finset Nat

/-- Represents the hexagonal grid -/
structure HexGrid where
  cells : Finset HexCell
  total_cells : Nat
  special_cells : Nat

/-- Represents a configuration of chips on the grid -/
def ChipConfiguration := Finset Nat

/-- Checks if a given configuration satisfies all number constraints -/
def satisfies_constraints (grid : HexGrid) (config : ChipConfiguration) : Prop :=
  ∀ c ∈ grid.cells, 
    (c.neighbors ∩ config).card ≥ 2 → 
    (c.neighbors ∩ config).card = c.id

/-- The main theorem to prove -/
theorem min_chips_required (grid : HexGrid) : 
  grid.total_cells = 37 ∧ 
  grid.special_cells = 7 ∧ 
  (∀ c ∈ grid.cells, c.neighbors.card ∈ ({3, 4, 6} : Set Nat)) →
  ∃ (config : ChipConfiguration), 
    satisfies_constraints grid config ∧ 
    config.card = 8 ∧
    (∀ (other_config : ChipConfiguration), 
      satisfies_constraints grid other_config → 
      other_config.card ≥ 8) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chips_required_l774_77424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_characterization_tangent_characterization_unique_tangent_at_two_two_tangents_above_two_l774_77403

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1 / x

-- Define the property of having extremum values
def has_extremum (a : ℝ) : Prop :=
  ∃ x₀ > 0, ∀ x > 0, f a x₀ ≤ f a x ∨ f a x₀ ≥ f a x

-- Define the property of having a tangent line through the origin
def has_tangent_through_origin (a : ℝ) : Prop :=
  ∃ x₀ > 0, (f a x₀) / x₀ = (a / x₀ - 1 / x₀^2)

-- Theorem 1: Characterization of extremum values
theorem extremum_characterization :
  ∀ a : ℝ, has_extremum a ↔ a > 0 :=
by sorry

-- Theorem 2: Characterization of tangent lines through origin
theorem tangent_characterization :
  ∀ a : ℝ, has_tangent_through_origin a ↔ a ≥ 2 :=
by sorry

-- Theorem 3: Unique tangent line when a = 2
theorem unique_tangent_at_two :
  ∃! x₀, x₀ > 0 ∧ (f 2 x₀) / x₀ = (2 / x₀ - 1 / x₀^2) :=
by sorry

-- Theorem 4: Two tangent lines when a > 2
theorem two_tangents_above_two :
  ∀ a, a > 2 → ∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    (f a x₁) / x₁ = (a / x₁ - 1 / x₁^2) ∧
    (f a x₂) / x₂ = (a / x₂ - 1 / x₂^2) :=
by sorry

end


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_characterization_tangent_characterization_unique_tangent_at_two_two_tangents_above_two_l774_77403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_two_digit_numbers_l774_77476

def digits : List Nat := [2, 3, 7]

def twoDigitNumbers : List Nat :=
  List.map (λ (d1, d2) => 10 * d1 + d2) (List.product digits digits)

theorem sum_of_two_digit_numbers :
  (twoDigitNumbers.sum) = 396 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_two_digit_numbers_l774_77476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_property_a_sum_bound_l774_77427

noncomputable def a : ℕ → ℝ
| 0 => 1
| n + 1 => 2 * a n + Real.sqrt (3 * (a n)^2 + 1)

theorem a_property (n : ℕ) (hn : n > 1) :
  a (n + 1) + a (n - 1) = 4 * a n := by
  sorry

theorem a_sum_bound (n : ℕ) :
  (Finset.range n).sum (λ k => 1 / a k) < (1 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_property_a_sum_bound_l774_77427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_factorial_sum_l774_77455

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_factorials : ℕ := List.range 14
  |>.map (fun i => factorial (7 * (i + 1)))
  |>.sum

theorem last_two_digits_of_factorial_sum :
  last_two_digits sum_factorials = last_two_digits (factorial 7) := by
  sorry

#eval last_two_digits sum_factorials
#eval last_two_digits (factorial 7)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_factorial_sum_l774_77455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_analysis_l774_77487

-- Define the propositions
def p : Prop := ∃ (T : ℝ), T > 0 ∧ T = Real.pi/2 ∧ ∀ (x : ℝ), Real.sin (2*x + T) = Real.sin (2*x)
def q : Prop := ∀ (x : ℝ), Real.cos x = Real.cos (Real.pi - x)

-- State the theorem
theorem proposition_analysis :
  (¬p) ∧ (¬q) ∧ (¬q) ∧ ¬(p ∧ q) ∧ ¬(p ∨ q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_analysis_l774_77487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_on_interval_l774_77425

def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 14

theorem max_min_values_on_interval :
  (∃ x ∈ Set.Icc (-3 : ℝ) 4, ∀ y ∈ Set.Icc (-3 : ℝ) 4, f y ≤ f x) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 4, f x = 142) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 4, ∀ y ∈ Set.Icc (-3 : ℝ) 4, f x ≤ f y) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 4, f x = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_on_interval_l774_77425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_determinant_l774_77468

theorem cubic_roots_determinant (a b c : ℂ) : 
  (a^3 - 2*a^2 + 4*a - 5 = 0) →
  (b^3 - 2*b^2 + 4*b - 5 = 0) →
  (c^3 - 2*c^2 + 4*c - 5 = 0) →
  Matrix.det ![![a, b, c], ![b, c, a], ![c, a, b]] = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_determinant_l774_77468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_l774_77472

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x ∈ {x : ℝ | x ≠ 1 ∧ x ≠ 1 + 1/a}, f a b (-x) = -(f a b x)) →
  a = -1/2 ∧ b = Real.log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_l774_77472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_equals_5pi_l774_77415

-- Define the bounding functions
noncomputable def f (x : ℝ) : ℝ := 1 - x^2
noncomputable def g (y : ℝ) : ℝ := Real.sqrt (y - 2)

-- Define the region of integration
def region : Set (ℝ × ℝ) :=
  {(x, y) | 0 ≤ x ∧ x ≤ 1 ∧ g y ≤ x ∧ y ≤ f x}

-- Define the volume of revolution
noncomputable def volume_of_revolution : ℝ :=
  ∫ x in Set.Icc 0 1, Real.pi * ((f x)^2 - (g (f x))^2)

-- State the theorem
theorem volume_equals_5pi : volume_of_revolution = 5 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_equals_5pi_l774_77415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_difference_l774_77428

/-- Given two monomials are like terms, prove that m - n = 4 -/
theorem like_terms_difference (m n : ℤ) : 
  (∃ (a b : ℚ) (h : ℚ → ℚ → ℚ), 
    a * h 1 1 ^ 3 * h 1 1 ^ (n + 1) = b * h 1 1 ^ (m - 2) * h 1 1 ^ 2) → 
  m - n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_difference_l774_77428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_monotonic_increase_intervals_l774_77493

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.sqrt 3 * Real.sin (2 * x) + a

-- Theorem 1: If the maximum value of f(x) is 2, then a = -1
theorem max_value_implies_a (a : ℝ) : 
  (∀ x : ℝ, f a x ≤ 2) ∧ (∃ x : ℝ, f a x = 2) → a = -1 :=
by sorry

-- Theorem 2: Intervals of monotonic increase
theorem monotonic_increase_intervals (a : ℝ) (k : ℤ) :
  StrictMonoOn (f a) (Set.Icc (-(Real.pi/3) + k * Real.pi) (Real.pi/6 + k * Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_monotonic_increase_intervals_l774_77493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segments_equivalent_circles_equivalent_l774_77410

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for line segments
structure LineSegment where
  start : Point
  finish : Point  -- Changed 'end' to 'finish' to avoid keyword conflict

-- Define a type for circles
structure Circle where
  center : Point
  radius : ℝ

-- Define equivalence between subsets of the plane
def Equivalent (A B : Set Point) :=
  ∃ f : A → B, Function.Bijective f

-- Theorem for line segments
theorem line_segments_equivalent (l1 l2 : LineSegment) :
  Equivalent 
    {p : Point | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p.x = (1 - t) * l1.start.x + t * l1.finish.x ∧ 
                                        p.y = (1 - t) * l1.start.y + t * l1.finish.y}
    {p : Point | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p.x = (1 - t) * l2.start.x + t * l2.finish.x ∧ 
                                        p.y = (1 - t) * l2.start.y + t * l2.finish.y} :=
by
  sorry

-- Theorem for circles
theorem circles_equivalent (c1 c2 : Circle) :
  Equivalent 
    {p : Point | (p.x - c1.center.x)^2 + (p.y - c1.center.y)^2 = c1.radius^2}
    {p : Point | (p.x - c2.center.x)^2 + (p.y - c2.center.y)^2 = c2.radius^2} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segments_equivalent_circles_equivalent_l774_77410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_l774_77409

-- Define the sets A and B
def A : Finset ℕ := {1, 2}
def B : Finset ℕ := {2, 3, 4}

-- Theorem statement
theorem union_cardinality : Finset.card (A ∪ B) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_l774_77409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l774_77494

theorem relationship_abc (a b c : ℝ) : 
  a = Real.sqrt 0.3 → b = (2 : ℝ)^(0.3 : ℝ) → c = (0.3 : ℝ)^(0.2 : ℝ) → b > a ∧ a > c :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l774_77494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_not_non_perfect_increasing_condition_for_non_perfect_increasing_l774_77405

/-- A function is a "non-perfect increasing function" on an interval I if it's increasing on I
    and F(x) = f(x)/x is decreasing on I. -/
def NonPerfectIncreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  (∀ x y, x ∈ I → y ∈ I → x < y → f x < f y) ∧
  (∀ x y, x ∈ I → y ∈ I → x < y → f x / x > f y / y)

theorem ln_not_non_perfect_increasing :
    ¬NonPerfectIncreasing Real.log (Set.Ioc 0 1) := by sorry

theorem condition_for_non_perfect_increasing (a : ℝ) :
    NonPerfectIncreasing (fun x ↦ 2*x + 2/x + a * Real.log x) (Set.Ici 1) ↔ 0 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_not_non_perfect_increasing_condition_for_non_perfect_increasing_l774_77405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_internal_environment_components_l774_77437

-- Define the types
inductive Component
| sodium
| antibodies
| plasmaProteins
| hemoglobin
| oxygen
| glucose
| carbonDioxide
| insulin
| hormones
| neurotransmitterVesicles
| aminoAcids

-- Define the internal environment
def internalEnvironment : Set Component := sorry

-- Define the location of hemoglobin
def hemoglobinLocation : String := "red blood cells"

-- Define the location of neurotransmitter vesicles
def neurotransmitterVesiclesLocation : String := "synaptosomes"

-- Theorem statement
theorem internal_environment_components :
  hemoglobinLocation = "red blood cells" →
  neurotransmitterVesiclesLocation = "synaptosomes" →
  Component.sodium ∈ internalEnvironment ∧
  Component.antibodies ∈ internalEnvironment ∧
  Component.plasmaProteins ∈ internalEnvironment ∧
  Component.glucose ∈ internalEnvironment ∧
  Component.carbonDioxide ∈ internalEnvironment ∧
  Component.insulin ∈ internalEnvironment :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_internal_environment_components_l774_77437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_association_prob_visit_B_third_l774_77436

-- Define the restaurants
inductive Restaurant
| A
| B

-- Define the satisfaction status
inductive Satisfaction
| Satisfied
| Dissatisfied

-- Define the survey results
def survey_results : List (Restaurant × Satisfaction) := sorry

-- Define the chi-square formula
noncomputable def chi_square (a b c d : ℕ) : ℝ :=
  let n : ℕ := a + b + c + d
  (n : ℝ) * ((a * d - b * c) ^ 2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d) : ℝ)

-- Define the critical value for α = 0.005
noncomputable def chi_square_critical : ℝ := 7.879

-- Define the transition probabilities
def transition_prob (prev next : Restaurant) : ℚ :=
  match prev, next with
  | Restaurant.A, Restaurant.A => 1/4
  | Restaurant.A, Restaurant.B => 3/4
  | Restaurant.B, Restaurant.A => 1/2
  | Restaurant.B, Restaurant.B => 1/2

-- Theorem for the chi-square test
theorem no_association : 
  chi_square 15 52 6 63 < chi_square_critical := by sorry

-- Theorem for the probability of visiting restaurant B on the third visit
theorem prob_visit_B_third : 
  (1/2 * transition_prob Restaurant.A Restaurant.B + 
   1/2 * transition_prob Restaurant.B Restaurant.B) * transition_prob Restaurant.A Restaurant.B +
  (1/2 * transition_prob Restaurant.A Restaurant.A + 
   1/2 * transition_prob Restaurant.B Restaurant.A) * transition_prob Restaurant.B Restaurant.B
  = 19/32 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_association_prob_visit_B_third_l774_77436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_after_two_years_l774_77419

/-- The amount after n years given an initial amount and a rate of increase -/
noncomputable def amountAfterYears (initialAmount : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initialAmount * (1 + rate) ^ years

/-- Theorem stating that an amount of 65000 increasing by 1/8th each year will be 82265.625 after two years -/
theorem amount_after_two_years :
  amountAfterYears 65000 (1/8) 2 = 82265.625 := by
  -- Proof steps would go here
  sorry

-- Use #eval only for computable expressions
#eval (65000 : Float) * (1 + 1/8) ^ (2 : Float)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_after_two_years_l774_77419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_average_speed_l774_77477

/-- Calculates the average speed of a trip with three segments -/
noncomputable def average_speed (total_distance : ℝ) (d1 d2 d3 : ℝ) (v1 v2 v3 : ℝ) : ℝ :=
  total_distance / ((d1 / v1) + (d2 / v2) + (d3 / v3))

/-- Theorem: The average speed for Tom's trip is approximately 29.67 mph -/
theorem toms_average_speed :
  let total_distance : ℝ := 180
  let d1 : ℝ := 60
  let d2 : ℝ := 50
  let d3 : ℝ := 70
  let v1 : ℝ := 20
  let v2 : ℝ := 30
  let v3 : ℝ := 50
  abs (average_speed total_distance d1 d2 d3 v1 v2 v3 - 29.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_average_speed_l774_77477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_triangles_bound_l774_77483

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle formed by three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- A set of points in a plane -/
def PointSet := Finset Point

/-- A property that no three points in a set are collinear -/
def noThreeCollinear (s : PointSet) : Prop :=
  sorry

/-- The set of all triangles with the largest area formed by points in a set -/
noncomputable def goodTriangles (s : PointSet) : Finset (Point × Point × Point) :=
  sorry

theorem good_triangles_bound (s : PointSet) (h1 : s.card = 2017) (h2 : noThreeCollinear s) :
  (goodTriangles s).card ≤ 2017 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_triangles_bound_l774_77483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l774_77471

theorem cos_minus_sin_value (θ : Real) 
  (h1 : θ > π/4 ∧ θ < π/2) 
  (h2 : Real.sin (2*θ) = 1/16) : 
  Real.cos θ - Real.sin θ = -Real.sqrt 15 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l774_77471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_length_pieces_l774_77496

-- Define the lengths of the cords as noncomputable
noncomputable def cord1_length : ℝ := Real.sqrt 20
noncomputable def cord2_length : ℝ := Real.sqrt 50
noncomputable def cord3_length : ℝ := Real.sqrt 98

-- Define the function to find the greatest common divisor of three real numbers
noncomputable def gcd_three_reals (a b c : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_equal_length_pieces :
  gcd_three_reals cord1_length cord2_length cord3_length = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_length_pieces_l774_77496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_triangle_properties_l774_77498

/-- A right triangle with altitude h to the hypotenuse -/
structure RightTriangle where
  h : ℝ
  h_pos : h > 0

/-- The triangle formed by the legs and the line through inscribed circle centers -/
structure CenterTriangle where
  angles : Fin 3 → ℝ
  area : ℝ

/-- The theorem statement -/
theorem center_triangle_properties (rt : RightTriangle) :
  ∃ (ct : CenterTriangle),
    ct.angles 0 = π/4 ∧ ct.angles 1 = π/4 ∧ ct.angles 2 = π/2 ∧
    ct.area = rt.h^2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_triangle_properties_l774_77498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l774_77442

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x + 1) / (2 * x^2 - x - 1)

-- Define a predicate for valid inputs
def IsValidInput (f : ℝ → ℝ) (x : ℝ) : Prop :=
  (2 * x + 1 ≥ 0) ∧ (2 * x^2 - x - 1 ≠ 0)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | IsValidInput f x} = {x : ℝ | x > -1/2 ∧ x ≠ 1} :=
by sorry

-- Note: We've defined IsValidInput explicitly and made f noncomputable.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l774_77442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domains_and_ranges_l774_77417

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (-(x - 2*a) * (x - a)) / Real.log a

def A (a : ℝ) : Set ℝ := Set.Ioo a (2*a)

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (9 - 3^x)

def B : Set ℝ := Set.Ici 2

def C : Set ℝ := Set.Ioc 0 3

theorem function_domains_and_ranges 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : A a ∪ C = C) : 
  A a = Set.Ioo a (2*a) ∧ 
  B = Set.Ici 2 ∧ 
  C = Set.Ioc 0 3 ∧ 
  0 < a ∧ a ≤ 3/2 ∧ a ≠ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domains_and_ranges_l774_77417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_problem_l774_77457

/-- The speed of an ant walking along a parabola -/
noncomputable def ant_speed (start_x : ℝ) (end_x : ℝ) (midpoint_speed : ℝ) : ℝ :=
  let v_x := 3 - Real.sqrt 3
  Real.sqrt 3 * v_x

/-- The problem statement -/
theorem ant_problem :
  let left_start : ℝ × ℝ := (-1, 1)
  let right_start : ℝ × ℝ := (1, 1)
  let parabola := fun x : ℝ ↦ x^2
  let midpoint_line := fun _ : ℝ ↦ (1 : ℝ)
  let midpoint_speed := 1
  let left_end_y := (1/2 : ℝ)
  ant_speed left_start.1 (-Real.sqrt (1/2)) midpoint_speed = 3 * Real.sqrt 3 - 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_problem_l774_77457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_expansion_no_pre_period_l774_77430

-- Define a function to represent the decimal expansion of 1/m
def decimal_expansion (m : ℕ) : ℕ → ℕ :=
  sorry -- Implementation details omitted for brevity

-- Define what it means for a decimal expansion to have no pre-period
def has_no_pre_period (expansion : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, ∃ n : ℕ, ∀ i : ℕ, expansion (k + i) = expansion i

-- State the theorem
theorem decimal_expansion_no_pre_period (m : ℕ) (h : Nat.Coprime m 10) :
  has_no_pre_period (decimal_expansion m) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_expansion_no_pre_period_l774_77430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_increase_l774_77485

def big_sale_commission : ℕ := 1300
def new_average_commission : ℕ := 400
def total_sales : ℕ := 6

theorem commission_increase : 
  (new_average_commission * total_sales - big_sale_commission) / (total_sales - 1) = 220 ∧
  new_average_commission - 220 = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_increase_l774_77485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_two_fifths_l774_77484

-- Define the sequence b_n
def b : ℕ → ℚ
  | 0 => 2  -- Add this case to cover all natural numbers
  | 1 => 2
  | 2 => 2
  | (n + 3) => b (n + 2) + b (n + 1)

-- Define the series
noncomputable def series_sum : ℚ := ∑' n, b n / 3^(n + 1)

-- Theorem statement
theorem series_sum_equals_two_fifths : series_sum = 2/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_two_fifths_l774_77484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elderly_arrangement_count_l774_77406

def num_volunteers : ℕ := 4
def num_elderly : ℕ := 2

theorem elderly_arrangement_count :
  (Nat.factorial (num_volunteers + num_elderly)) / 2 -
  2 * (Nat.factorial (num_volunteers - 1)) * 2 = 144 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elderly_arrangement_count_l774_77406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l774_77489

noncomputable def C (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x + 1

noncomputable def tangentSlope (x : ℝ) : ℝ := -Real.sqrt 3 * Real.sin x

noncomputable def slopeAngle (x : ℝ) : ℝ := Real.arctan (tangentSlope x)

theorem slope_angle_range :
  ∀ x : ℝ, slopeAngle x ∈ Set.union (Set.Icc 0 (Real.pi / 3)) (Set.Ico (2 * Real.pi / 3) Real.pi) :=
by
  sorry

#check slope_angle_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l774_77489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_in_rectangle_l774_77416

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  O : Point
  A : Point
  B : Point
  C : Point

/-- Inverse proportionality function -/
noncomputable def inverseProp (k : ℝ) (x : ℝ) : ℝ := k / x

/-- Area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Main theorem -/
theorem inverse_prop_in_rectangle (rect : Rectangle) (k : ℝ) (E F : Point) :
  rect.O.x = 0 ∧ rect.O.y = 0 ∧
  rect.A.x = 6 ∧ rect.A.y = 0 ∧
  rect.C.x = 0 ∧ rect.C.y = 5 ∧
  E.x < rect.A.x ∧ E.y = inverseProp k E.x ∧
  F.x = 0 ∧ F.y < rect.C.y ∧
  rect.C.y - F.y < F.y ∧
  triangleArea rect.O E F - triangleArea rect.B F E = 5 + 11/30 →
  k = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_in_rectangle_l774_77416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_with_equidistant_point_l774_77408

-- Define the points M and N
noncomputable def M : ℝ × ℝ := (1, 5/4)
noncomputable def N : ℝ × ℝ := (-4, -5/4)

-- Define the curves
def curve1 (x y : ℝ) : Prop := 4*x + 2*y - 1 = 0
def curve2 (x y : ℝ) : Prop := x^2 + y^2 = 3
def curve3 (x y : ℝ) : Prop := x^2 + y^2/4 = 1
def curve4 (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define the perpendicular bisector of MN
def perp_bisector (x y : ℝ) : Prop := 2*x + y + 3 = 0

-- Theorem statement
theorem curves_with_equidistant_point :
  (∃ x y : ℝ, perp_bisector x y ∧ curve2 x y) ∧
  (∃ x y : ℝ, perp_bisector x y ∧ curve4 x y) ∧
  ¬(∃ x y : ℝ, perp_bisector x y ∧ curve1 x y) ∧
  ¬(∃ x y : ℝ, perp_bisector x y ∧ curve3 x y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_with_equidistant_point_l774_77408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_circles_l774_77449

-- Define the rectangle dimensions
def rectangle_width : ℝ := 4
def rectangle_height : ℝ := 6

-- Define the radii of the circles
def radius1 : ℝ := 2
def radius2 : ℝ := 3
def radius3 : ℝ := 1

-- Define π as a real number (we'll use this instead of the exact π for simplicity)
noncomputable def π : ℝ := Real.pi

-- Theorem statement
theorem area_outside_circles : 
  abs ((rectangle_width * rectangle_height) - 
       ((radius1^2 + radius2^2 + radius3^2) * π / 4) - 13) < 0.1 := by 
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_circles_l774_77449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotonic_odd_function_l774_77481

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x > 0 then Real.exp x + a
  else if x < 0 then -(Real.exp (-x) + a)
  else a

-- State the theorem
theorem min_a_for_monotonic_odd_function (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) →  -- f is odd
  (∀ x y, x < y → f a x < f a y) →  -- f is strictly increasing (monotonic)
  a > -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotonic_odd_function_l774_77481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l774_77435

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else a^x

theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) ↔ a ∈ Set.Icc (1/6) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l774_77435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l774_77431

def sequence_a : ℕ → ℕ
  | 0 => 5  -- Add this case to handle Nat.zero
  | 1 => 5
  | 2 => 13
  | (n + 3) => 5 * sequence_a (n + 2) - 6 * sequence_a (n + 1)

theorem sequence_a_properties :
  (∀ n : ℕ, n ≥ 1 → Nat.gcd (sequence_a n) (sequence_a (n + 1)) = 1) ∧
  (∀ k : ℕ, ∀ p : ℕ, Nat.Prime p → p ∣ sequence_a (2^k) → 2^(k+1) ∣ (p - 1)) := by
  sorry

#eval sequence_a 0  -- This will output 5
#eval sequence_a 1  -- This will output 5
#eval sequence_a 2  -- This will output 13
#eval sequence_a 3  -- This will output 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l774_77431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_g_2010_l774_77453

def g (n : ℕ+) : ℕ+ :=
  2^n.val * 5^n.val * 3

theorem divisors_of_g_2010 :
  (Finset.card (Nat.divisors (g 2010).val)) = 8084442 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_g_2010_l774_77453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_range_l774_77474

-- Define an acute triangle
def AcuteTriangle (A B C : ℝ) := 
  0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2 ∧ A + B + C = Real.pi

-- Define the theorem
theorem side_b_range (A B C a b c : ℝ) 
  (h_acute : AcuteTriangle A B C) 
  (h_a : a = 1) 
  (h_B : B = Real.pi/3) 
  (h_law_of_sines : a / Real.sin A = b / Real.sin B) :
  Real.sqrt 3 / 2 < b ∧ b < Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_range_l774_77474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_locus_sin_xy_surface_area_equality_l774_77447

-- Part 1: Geometric locus
theorem geometric_locus_sin_xy (x y : ℝ) :
  Real.sin (x + y) = 0 ↔ ∃ k : ℤ, x + y = k * Real.pi :=
sorry

-- Part 2: Surface area comparison
-- Let's define some types and functions to represent the problem
def Cube := Unit -- Placeholder for a cube type
noncomputable def surface_area : Cube → ℝ := sorry -- Function to calculate surface area

def large_cube : Cube := sorry -- The large cube made of 27 smaller cubes
def modified_cube : Cube := sorry -- The cube with corner cubes removed

theorem surface_area_equality :
  surface_area large_cube = surface_area modified_cube :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_locus_sin_xy_surface_area_equality_l774_77447
