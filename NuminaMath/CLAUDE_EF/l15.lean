import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birds_optimal_speed_l15_1509

/-- Represents the travel scenario for Mr. Earl E. Bird --/
structure TravelScenario where
  distance : ℝ  -- Distance to work in miles
  target_time : ℝ  -- Time to arrive at work in hours

/-- Calculates the travel time given speed and distance --/
noncomputable def travel_time (speed : ℝ) (distance : ℝ) : ℝ := distance / speed

/-- The main theorem to prove --/
theorem birds_optimal_speed (scenario : TravelScenario) 
  (h1 : travel_time 45 scenario.distance = scenario.target_time + 1/15)
  (h2 : travel_time 55 scenario.distance = scenario.target_time - 1/30) :
  ∃ speed : ℝ, 50 < speed ∧ speed < 52 ∧ 
    travel_time speed scenario.distance = scenario.target_time := by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birds_optimal_speed_l15_1509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_x_squared_inequality_l15_1530

theorem ln_x_squared_inequality (x : ℝ) (h : 1 < x ∧ x < 2) :
  (Real.log x / x)^2 < Real.log x / x ∧ Real.log x / x < Real.log (x^2) / x^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_x_squared_inequality_l15_1530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_specific_l15_1532

/-- The volume of a pyramid with a triangular base and specified dimensions -/
noncomputable def pyramid_volume (base_side1 base_side2 base_side3 edge_length : ℝ) : ℝ :=
  let base_area := Real.sqrt ((base_side1 + base_side2 + base_side3) * 
                              (base_side1 + base_side2 - base_side3) * 
                              (base_side1 - base_side2 + base_side3) * 
                              (-base_side1 + base_side2 + base_side3)) / 4
  let height := Real.sqrt (edge_length^2 - (base_area / (base_side1 * base_side2 * base_side3)))
  (1/3) * base_area * height

theorem pyramid_volume_specific : 
  pyramid_volume 20 20 24 25 = 800 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_specific_l15_1532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_some_expression_value_l15_1564

theorem some_expression_value (x y : ℝ) (h : x * y = 1) :
  ∃ some_expression : ℝ, 
    (4 : ℝ)^some_expression^2 / (4 : ℝ)^(x-y)^2 = 256 ∧ 
    some_expression = 2 + (x - 1/x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_some_expression_value_l15_1564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_slope_range_l15_1519

noncomputable section

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 + y^2/4 = 1

/-- Point A -/
def A : ℝ × ℝ := (1/2, 1/2)

/-- Point B -/
def B : ℝ × ℝ := (1/2, 1)

/-- A point is between two other points if its y-coordinate is between their y-coordinates -/
def is_between (p q r : ℝ × ℝ) : Prop :=
  p.2 ≤ q.2 ∧ q.2 ≤ r.2

/-- The midpoint of two points -/
def midpoint_of (p q : ℝ × ℝ) : ℝ × ℝ := ((p.1 + q.1)/2, (p.2 + q.2)/2)

/-- The slope of a line through two points -/
def slope_of (p q : ℝ × ℝ) : ℝ := (q.2 - p.2) / (q.1 - p.1)

/-- The main theorem -/
theorem chord_slope_range :
  ∀ (p q : ℝ × ℝ),
  is_on_ellipse p.1 p.2 →
  is_on_ellipse q.1 q.2 →
  is_between A (midpoint_of p q) B →
  -4 ≤ slope_of p q ∧ slope_of p q ≤ -2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_slope_range_l15_1519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_chalk_length_l15_1556

/-- Represents the length of a chalk stick in centimeters -/
def ChalkLength := ℝ

/-- The range of tested chalk lengths -/
def TestedRange : Set ℝ := {x : ℝ | 10 ≤ x ∧ x ≤ 15}

/-- Predicate to check if a length is optimal based on trials -/
def IsOptimalLength (l : ℝ) : Prop := 
  l ∈ TestedRange ∧ ∀ x ∈ TestedRange, l ≤ x → x ≤ l

/-- The optimal length found through trials -/
def OptimalLength : ℝ := 12

/-- Theorem stating that 12 cm is the optimal length for a chalk stick -/
theorem optimal_chalk_length : 
  IsOptimalLength OptimalLength := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_chalk_length_l15_1556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_solid_volume_l15_1573

/-- A parallelepiped in 3D space -/
structure Parallelepiped where
  volume : ℝ
  surfaceArea : ℝ
  edgeLengthSum : ℝ

/-- The solid formed by extending a parallelepiped by distance t -/
noncomputable def ExtendedSolid (P : Parallelepiped) (t : ℝ) : ℝ :=
  P.volume + P.surfaceArea * t + (Real.pi / 4) * P.edgeLengthSum * t^2 + (4 * Real.pi / 3) * t^3

/-- Theorem stating the volume of the extended solid -/
theorem extended_solid_volume (P : Parallelepiped) (t : ℝ) (h : t ≥ 0) :
  ExtendedSolid P t = P.volume + P.surfaceArea * t + (Real.pi / 4) * P.edgeLengthSum * t^2 + (4 * Real.pi / 3) * t^3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_solid_volume_l15_1573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l15_1553

/-- Theorem: Equation of a specific hyperbola

Given a hyperbola C: (x²/a²) - (y²/b²) = 1 with a > 0, b > 0, right focus at F, origin at O,
and a circle with OF as diameter intersecting C and its asymptote at O and A(3/2, √3/2),
the equation of C is x²/3 - y² = 1.
-/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (C : Set (ℝ × ℝ))
  (hC : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x, y) ∈ C)
  (F : ℝ × ℝ)
  (hF : F.1 > 0 ∧ F.2 = 0)  -- Right focus
  (hO : (0, 0) ∈ C)  -- Origin on hyperbola
  (hCircle : ∃ (circle : Set (ℝ × ℝ)), F ∈ circle ∧ (0, 0) ∈ circle ∧ (3/2, Real.sqrt 3/2) ∈ circle)
  (hAsymptote : ∃ (asymptote : Set (ℝ × ℝ)), (0, 0) ∈ asymptote ∧ (3/2, Real.sqrt 3/2) ∈ asymptote)
  : ∀ x y : ℝ, (x, y) ∈ C ↔ x^2 / 3 - y^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l15_1553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_undefined_l15_1547

theorem fraction_undefined (x : ℝ) : (x - 2) / (x + 1) = 0/0 ↔ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_undefined_l15_1547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cost_at_18_minutes_l15_1539

/-- Represents the duration of a phone call in minutes -/
def call_duration : ℝ := 18

/-- Calculates the cost of a call under Plan A -/
noncomputable def cost_plan_a (duration : ℝ) : ℝ :=
  if duration ≤ 4 then 0.60 else 0.60 + 0.06 * (duration - 4)

/-- Calculates the cost of a call under Plan B -/
def cost_plan_b (duration : ℝ) : ℝ := 0.08 * duration

/-- Theorem stating that the costs of Plan A and Plan B are equal at 18 minutes -/
theorem equal_cost_at_18_minutes :
  cost_plan_a call_duration = cost_plan_b call_duration := by
  -- Unfold definitions
  unfold cost_plan_a cost_plan_b call_duration
  -- Simplify the if-then-else expression
  simp
  -- Perform numerical calculations
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cost_at_18_minutes_l15_1539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_sequence_l15_1503

theorem perfect_square_sequence (k : ℕ) : ∃ m : ℕ, 
  (10^(2*k) - 1)/9 + 4*(10^k - 1)/9 + 1 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_sequence_l15_1503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_circle_equation_fixed_point_property_l15_1524

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line passing through M(m,0) with slope k
def line (x y m k : ℝ) : Prop := y = k*(x - m)

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem parabola_line_intersection (m : ℝ) :
  (∀ k : ℝ, ∃ A B : ℝ × ℝ,
    parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
    line A.1 A.2 m k ∧ line B.1 B.2 m k) :=
by sorry

theorem circle_equation :
  let m := 1
  let k := 1
  ∃ A B : ℝ × ℝ,
    parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
    line A.1 A.2 m k ∧ line B.1 B.2 m k ∧
    ∀ x y : ℝ, (x - 3)^2 + (y - 2)^2 = 16 ↔
      distance x y A.1 A.2 = distance x y B.1 B.2 :=
by sorry

theorem fixed_point_property :
  let m := 2
  ∀ k : ℝ, ∃ A B : ℝ × ℝ,
    parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
    line A.1 A.2 m k ∧ line B.1 B.2 m k ∧
    1 / (distance A.1 A.2 m 0)^2 + 1 / (distance B.1 B.2 m 0)^2 = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_circle_equation_fixed_point_property_l15_1524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_2_sqrt_2_l15_1504

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 12 = 0

/-- The length of the common chord between the two circles -/
noncomputable def commonChordLength : ℝ := 2 * Real.sqrt 2

theorem common_chord_length_is_2_sqrt_2 :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y ∧
  commonChordLength = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_2_sqrt_2_l15_1504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_S_n_l15_1510

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  fun n ↦ a₁ + (n - 1) * d

-- Define the sum of the first n terms
noncomputable def S_n (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- The main theorem
theorem min_positive_S_n (a₁ : ℝ) (d : ℝ) :
  d < 0 →
  (arithmetic_sequence a₁ d 11) / (arithmetic_sequence a₁ d 10) < -1 →
  (∃ k : ℕ, ∀ n : ℕ, S_n a₁ d n ≤ S_n a₁ d k) →
  (∃! n : ℕ, n > 0 ∧ S_n a₁ d n > 0 ∧ 
    ∀ m : ℕ, m > 0 ∧ m ≠ n → S_n a₁ d m > S_n a₁ d n → S_n a₁ d m > S_n a₁ d n) →
  n = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_S_n_l15_1510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_skew_iff_b_neq_one_fifteenth_l15_1568

/-- Two lines in ℝ³ represented by their parametric equations -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Check if two lines in ℝ³ are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ∀ (t u : ℝ), 
    (l1.point.1 + t * l1.direction.1 ≠ l2.point.1 + u * l2.direction.1) ∨
    (l1.point.2.1 + t * l1.direction.2.1 ≠ l2.point.2.1 + u * l2.direction.2.1) ∨
    (l1.point.2.2 + t * l1.direction.2.2 ≠ l2.point.2.2 + u * l2.direction.2.2)

theorem lines_skew_iff_b_neq_one_fifteenth (b : ℝ) :
  let l1 : Line3D := ⟨(2, 3, b), (3, 4, 5)⟩
  let l2 : Line3D := ⟨(3, 4, 1), (6, 3, 2)⟩
  are_skew l1 l2 ↔ b ≠ 1/15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_skew_iff_b_neq_one_fifteenth_l15_1568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alchemy_value_l15_1513

def alphabet_value (n : ℕ) : ℤ :=
  match n % 13 with
  | 1 => 3
  | 2 => 2
  | 3 => 1
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -3
  | 8 => -2
  | 9 => -1
  | 10 => 0
  | 11 => 1
  | 12 => 2
  | 0 => 3
  | _ => 0  -- This case should never occur

def letter_to_position (c : Char) : ℕ :=
  match c with
  | 'A' => 1
  | 'L' => 12
  | 'C' => 3
  | 'H' => 8
  | 'E' => 5
  | 'M' => 13
  | 'Y' => 25
  | _ => 0  -- This case should never occur

def word_value (word : String) : ℤ :=
  List.sum (word.toList.map (fun c => alphabet_value (letter_to_position c)))

theorem alchemy_value : word_value "ALCHEMY" = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alchemy_value_l15_1513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l15_1577

-- Define the circle
def circle' (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + y^2 = 1

-- Define the line
def line' (x y : ℝ) : Prop := y = x

-- Define the third quadrant
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Theorem statement
theorem circle_tangent_to_line :
  ∃ (a : ℝ), ∃ (x y : ℝ),
    circle' a x y ∧ 
    line' x y ∧ 
    third_quadrant x y ∧ 
    a = -Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l15_1577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_no_fixed_point_theorem_l15_1592

noncomputable section

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define a line passing through two points
noncomputable def line_through (p q : ℝ × ℝ) (x : ℝ) : ℝ := 
  (q.2 - p.2) / (q.1 - p.1) * (x - p.1) + p.2

theorem fixed_point_theorem (A B : ℝ × ℝ) 
  (hA : parabola A ∧ A ≠ origin) 
  (hB : parabola B ∧ B ≠ origin) 
  (h_orthogonal : dot_product A B = 0) :
  ∃ (y : ℝ), line_through A B 4 = y ∧ y = 0 := by sorry

theorem no_fixed_point_theorem (A B : ℝ × ℝ) 
  (hA : parabola A ∧ A ≠ origin) 
  (hB : parabola B ∧ B ≠ origin) 
  (h_dot_product : dot_product A B = -2) :
  ¬∃ (p : ℝ × ℝ), ∀ (A' B' : ℝ × ℝ), 
    parabola A' ∧ A' ≠ origin → 
    parabola B' ∧ B' ≠ origin → 
    dot_product A' B' = -2 → 
    line_through A' B' p.1 = p.2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_no_fixed_point_theorem_l15_1592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_iterations_for_decay_l15_1552

/-- The exponential decay model for learning rate -/
noncomputable def learning_rate (L₀ D G G₀ : ℝ) : ℝ := L₀ * D^(G / G₀)

/-- The minimum number of iterations for learning rate to decay below 0.1 -/
theorem min_iterations_for_decay :
  let L₀ : ℝ := 0.5
  let G₀ : ℝ := 18
  let D : ℝ := 4/5
  ∀ G : ℕ, (G ≥ 130 ↔ learning_rate L₀ D (G : ℝ) G₀ < 0.1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_iterations_for_decay_l15_1552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l15_1512

/-- The sine function with angular frequency ω and phase shift π/3 -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 3)

/-- Theorem stating that under given conditions, ω must equal 4 -/
theorem omega_value (m n ω : ℝ) (h1 : |n| ≠ 1) 
  (h2 : f ω m = n) 
  (h3 : f ω (m + Real.pi) = n)
  (h4 : ∃ (x1 x2 x3 : ℝ), m < x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < m + Real.pi ∧ 
        f ω x1 = n ∧ f ω x2 = n ∧ f ω x3 = n) : 
  ω = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l15_1512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_jogging_time_l15_1583

/-- Jogging schedule for Mr. John --/
structure JoggingSchedule where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ
  saturday : ℝ
  sunday : ℝ

/-- Calculate total jogging time for two weeks --/
def totalJoggingTime (schedule : JoggingSchedule) : ℝ :=
  let firstWeek := schedule.monday + schedule.tuesday + schedule.wednesday + 
                   schedule.thursday + schedule.friday + schedule.saturday + schedule.sunday
  let secondWeek := schedule.monday + schedule.tuesday + 
                    schedule.thursday + schedule.friday + schedule.saturday + schedule.sunday
  firstWeek + secondWeek

/-- Mr. John's actual jogging schedule --/
def johnSchedule : JoggingSchedule :=
  { monday := 1.25
    tuesday := 1.75
    wednesday := 1.5
    thursday := 1.333
    friday := 2
    saturday := 1
    sunday := 0 }

/-- Theorem: Mr. John's total jogging time for two weeks is approximately 16.166 hours --/
theorem john_jogging_time : 
  abs (totalJoggingTime johnSchedule - 16.166) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_jogging_time_l15_1583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_poly_representation_l15_1527

-- Define the quadratic equation
def quadratic (a b c x : ℚ) : Prop := a * x^2 + b * x + c = 0

-- Define the existence of polynomials p and q
def exists_poly_pq (x a b c : ℚ) : Prop :=
  ∀ n : ℕ, ∃ (p q : ℚ → ℚ → ℚ → ℚ → ℚ),
    x = p (x^n) a b c / q (x^n) a b c

-- Define the specific polynomials r and s
def poly_rs (x : ℚ) : Prop :=
  x = (x^3 + (x + 1/x)) / ((x + 1/x)^2 - 1)

theorem quadratic_poly_representation (a b c x : ℚ) :
  quadratic a b c x → exists_poly_pq x a b c ∧ poly_rs x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_poly_representation_l15_1527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_cube_from_tetrahedrons_l15_1529

-- Define a regular tetrahedron
structure RegularTetrahedron where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

-- Define a cube
structure Cube where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

-- Define the property of constructing a cube from tetrahedrons
def CanConstructCube (n : ℕ) (t : RegularTetrahedron) (c : Cube) : Prop :=
  ∃ (k : ℕ), k > 0 ∧
  n * (Real.sqrt 2 / 12) * t.sideLength^3 = c.sideLength^3 ∧
  k * (Real.sqrt 3 / 4) * t.sideLength^2 = c.sideLength^2

-- Theorem statement
theorem impossible_cube_from_tetrahedrons :
  ∀ (n : ℕ) (t : RegularTetrahedron),
  ¬ CanConstructCube n t (Cube.mk 1 (by norm_num)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_cube_from_tetrahedrons_l15_1529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_connecting_edges_values_l15_1516

/-- A simple graph with 4038 vertices -/
structure Graph :=
  (vertices : Finset (Fin 4038))
  (edges : Finset (Fin 4038 × Fin 4038))
  (simple : ∀ e ∈ edges, e.1 ≠ e.2)

/-- A partition of the vertices into two groups of 2019 each -/
structure Partition (G : Graph) :=
  (group1 : Finset (Fin 4038))
  (group2 : Finset (Fin 4038))
  (partition_complete : group1 ∪ group2 = G.vertices)
  (partition_disjoint : group1 ∩ group2 = ∅)
  (group1_size : group1.card = 2019)
  (group2_size : group2.card = 2019)

/-- The number of edges connecting the two groups -/
def connecting_edges (G : Graph) (P : Partition G) : ℕ :=
  (G.edges.filter (fun e => (e.1 ∈ P.group1 ∧ e.2 ∈ P.group2) ∨ 
                            (e.1 ∈ P.group2 ∧ e.2 ∈ P.group1))).card

/-- The theorem stating the possible values of connecting edges -/
theorem connecting_edges_values (G : Graph) (P : Partition G) :
  connecting_edges G P ∈ ({2019, 2019^2 - 2019, 2019^2} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_connecting_edges_values_l15_1516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l15_1562

/-- The angle of inclination of a line with slope m is the angle θ such that tan θ = m -/
noncomputable def angle_of_inclination (m : ℝ) : ℝ := Real.arctan m

theorem line_inclination (x y : ℝ) : 
  y = -Real.sqrt 3 * x + 2 * Real.sqrt 3 → angle_of_inclination (-Real.sqrt 3) = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l15_1562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_and_perpendicular_unit_vector_l15_1593

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, -4)

theorem vector_operations_and_perpendicular_unit_vector :
  ∃ (x y : ℝ),
  let c := (x, y)
  (c.1 * (a.1 - b.1) + c.2 * (a.2 - b.2) = 0) ∧
  (c.1^2 + c.2^2 = 1) ∧
  (2 * a.1 + 3 * b.1 = -5 ∧ 2 * a.2 + 3 * b.2 = -10) ∧
  ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2 = 145) ∧
  ((x = Real.sqrt 2 / 2 ∧ y = -Real.sqrt 2 / 2) ∨ 
   (x = -Real.sqrt 2 / 2 ∧ y = Real.sqrt 2 / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_and_perpendicular_unit_vector_l15_1593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_m_coordinate_l15_1511

/-- Given points A and B in 3D space and M on the y-axis, prove M's y-coordinate -/
theorem point_m_coordinate (A B M : ℝ × ℝ × ℝ) : 
  A = (1, 0, 2) →
  B = (1, -3, 1) →
  M.1 = 0 ∧ M.2.2 = 0 →
  (M.1 - A.1)^2 + (M.2.1 - A.2.1)^2 + (M.2.2 - A.2.2)^2 = 
  (M.1 - B.1)^2 + (M.2.1 - B.2.1)^2 + (M.2.2 - B.2.2)^2 →
  M.2.1 = -1 := by
  sorry

#check point_m_coordinate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_m_coordinate_l15_1511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_implication_l15_1589

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → (2 : ℝ)^a > (2 : ℝ)^b) ↔ (a ≤ b → (2 : ℝ)^a ≤ (2 : ℝ)^b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_implication_l15_1589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stones_for_hall_l15_1500

/-- The number of stones required to pave a rectangular hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℕ :=
  (hall_length * 10 * hall_width * 10 / (stone_length * stone_width)).floor.toNat

/-- Theorem stating that 2700 stones are required to pave the given hall -/
theorem stones_for_hall : stones_required 36 15 4 5 = 2700 := by
  -- Unfold the definition of stones_required
  unfold stones_required
  -- Evaluate the expression
  norm_num
  -- QED
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stones_for_hall_l15_1500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_l15_1502

theorem sequence_existence : ∃ (a : Fin 2016 → ℕ),
  (∀ (r s : Fin 2016), r ≤ s → (Finset.range (s - r + 1)).sum (λ i => a (r + i)) > 1 → 
    ¬ Nat.Prime ((Finset.range (s - r + 1)).sum (λ i => a (r + i)))) ∧
  (∀ i : Fin 2015, Nat.gcd (a i) (a (i + 1)) = 1) ∧
  (∀ i : Fin 2014, Nat.gcd (a i) (a (i + 2)) = 1) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_l15_1502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l15_1554

-- Define the two functions
noncomputable def f (x : ℝ) := Real.sqrt x
def g (x : ℝ) := x^3

-- Define the area between the curves
noncomputable def area : ℝ := ∫ x in Set.Icc 0 1, f x - g x

-- Theorem statement
theorem area_between_curves : area = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l15_1554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l15_1533

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x)

def x₀ : ℝ := 0

def m : ℝ := 2

def b : ℝ := 1

theorem tangent_line_at_zero :
  (fun x => m * x + b) = (fun x => (deriv f x₀) * (x - x₀) + f x₀) := by
  sorry

#check tangent_line_at_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l15_1533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_ten_l15_1541

def next_term (n : ℕ) : ℕ :=
  if n ≤ 5 then n * 8
  else if n % 2 = 0 then n / 2
  else n + 3

def sequence_term (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => next_term (sequence_term start n)

theorem tenth_term_is_ten :
  sequence_term 100 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_ten_l15_1541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_unique_circles_l15_1561

-- Define a rhombus
structure Rhombus where
  vertices : Finset (ℝ × ℝ)
  is_rhombus : vertices.card = 4 ∧ 
               ∀ v1 v2, v1 ∈ vertices → v2 ∈ vertices → v1 ≠ v2 → 
               ∃ v3 v4, v3 ∈ vertices ∧ v4 ∈ vertices ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v2 ≠ v3 ∧ v2 ≠ v4 ∧
               (v1.1 - v2.1)^2 + (v1.2 - v2.2)^2 = (v3.1 - v4.1)^2 + (v3.2 - v4.2)^2

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to check if a circle's diameter has endpoints at vertices of the rhombus
def is_valid_circle (r : Rhombus) (c : Circle) : Prop :=
  ∃ v1 v2, v1 ∈ r.vertices ∧ v2 ∈ r.vertices ∧ v1 ≠ v2 ∧
  (v1.1 - v2.1)^2 + (v1.2 - v2.2)^2 = 4 * c.radius^2 ∧
  c.center = ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)

-- Theorem statement
theorem rhombus_unique_circles (r : Rhombus) :
  ∃! (s : Finset Circle), s.card = 2 ∧ ∀ c ∈ s, is_valid_circle r c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_unique_circles_l15_1561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sum_inequality_l15_1574

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 * b^3 * (a * b - a * c - b * c + c^2) +
  b^3 * c^3 * (b * c - b * a - c * a + a^2) +
  c^3 * a^3 * (c * a - c * b - a * b + b^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sum_inequality_l15_1574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_percentage_after_addition_l15_1586

/-- Represents the composition of a kola solution -/
structure KolaSolution where
  total_volume : ℝ
  water_volume : ℝ
  kola_volume : ℝ
  sugar_volume : ℝ

/-- Calculates the percentage of a component in the solution -/
noncomputable def percentage (component_volume : ℝ) (total_volume : ℝ) : ℝ :=
  (component_volume / total_volume) * 100

/-- Theorem stating the final sugar percentage in the kola solution -/
theorem sugar_percentage_after_addition 
  (initial : KolaSolution)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_kola : ℝ)
  (h1 : initial.total_volume = 340)
  (h2 : initial.water_volume = initial.total_volume * 0.64)
  (h3 : initial.kola_volume = initial.total_volume * 0.09)
  (h4 : initial.sugar_volume = initial.total_volume - initial.water_volume - initial.kola_volume)
  (h5 : added_sugar = 3.2)
  (h6 : added_water = 8)
  (h7 : added_kola = 6.8) :
  let final_solution : KolaSolution := {
    total_volume := initial.total_volume + added_sugar + added_water + added_kola,
    water_volume := initial.water_volume + added_water,
    kola_volume := initial.kola_volume + added_kola,
    sugar_volume := initial.sugar_volume + added_sugar
  }
  ∃ ε > 0, |percentage final_solution.sugar_volume final_solution.total_volume - 26.54| < ε := by
  sorry

#eval "Theorem statement completed. Proof is omitted (sorry)."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_percentage_after_addition_l15_1586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_increase_l15_1514

-- Define the slope of the line
def line_slope : ℚ := 10 / 4

-- Define the change in x
def delta_x : ℚ := 12

-- Theorem statement
theorem y_increase : line_slope * delta_x = 30 := by
  -- Unfold the definitions
  unfold line_slope delta_x
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_increase_l15_1514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_value_is_2015_l15_1535

/-- Represents the configuration of nine integers in circles --/
structure CircleConfig where
  integers : List Int
  sums : List Int

/-- The problem setup --/
def problem_setup : CircleConfig → Prop := fun config =>
  -- Nine consecutive integers starting from 2012
  config.integers.length = 9 ∧
  config.integers.head! = 2012 ∧
  (List.zip config.integers (List.tail config.integers)).all (fun (a, b) => b = a + 1) ∧
  -- Four lines with equal sums
  config.sums.length = 4 ∧
  config.sums.all (fun s => s = config.sums.head!)

/-- The sum is as small as possible --/
def min_sum (config : CircleConfig) : Prop :=
  ∀ (other : CircleConfig), problem_setup other → config.sums.head! ≤ other.sums.head!

/-- The theorem to prove --/
theorem middle_value_is_2015 (config : CircleConfig) :
  problem_setup config → min_sum config → config.integers[4]? = some 2015 := by
  intro h_setup h_min_sum
  sorry

#check middle_value_is_2015

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_value_is_2015_l15_1535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_spheres_volume_l15_1549

/-- Volume of a sphere with radius r -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The problem of concentric spheres -/
theorem concentric_spheres_volume (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ = 4) (h₂ : r₂ = 7) (h₃ : r₃ = 10) :
  sphereVolume r₃ - sphereVolume r₂ = (2628 / 3) * Real.pi ∧ 
  sphereVolume r₂ - sphereVolume r₁ = (1116 / 3) * Real.pi := by
  sorry

#check concentric_spheres_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_spheres_volume_l15_1549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_of_factorial_divided_l15_1501

theorem factorial_of_factorial_divided (n : ℕ) : (n.factorial.factorial) / n.factorial = (n.factorial - 1).factorial :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_of_factorial_divided_l15_1501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_relations_l15_1566

theorem set_relations :
  ({0} : Set ℕ) ≠ (∅ : Set ℕ) ∧
  ({2} : Set ℕ) ⊆ ({2, 4, 6} : Set ℕ) ∧
  2 ∉ {x : ℝ | x^2 - 3*x + 2 = 0} ∧
  (0 : ℕ) ∈ ({0} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_relations_l15_1566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_minus_two_alpha_l15_1557

theorem sin_pi_half_minus_two_alpha (α : ℝ) (h : Real.cos α = 1/3) : 
  Real.sin (π/2 - 2*α) = -7/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_minus_two_alpha_l15_1557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_zeros_f_three_tangents_l15_1581

-- Define the functions
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x + 1

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := k * x + 1 - Real.log x

-- Define the piecewise function h
noncomputable def h (k : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then f x else g k x

-- Part 1: Number of zeros of h(x) when k < 0
theorem h_zeros (k : ℝ) (h_k : k < 0) :
  (∃! x, h k x = 0 ∧ k < -1) ∨
  (∃! x₁ x₂, x₁ ≠ x₂ ∧ h k x₁ = 0 ∧ h k x₂ = 0 ∧ -1 ≤ k) :=
sorry

-- Part 2: Range of a for exactly three tangents
theorem f_three_tangents (a : ℝ) :
  (∃! t₁ t₂ t₃, t₁ ≠ t₂ ∧ t₁ ≠ t₃ ∧ t₂ ≠ t₃ ∧
    f t₁ - (-4) = (6 * t₁^2 - 6 * t₁) * (t₁ - a) ∧
    f t₂ - (-4) = (6 * t₂^2 - 6 * t₂) * (t₂ - a) ∧
    f t₃ - (-4) = (6 * t₃^2 - 6 * t₃) * (t₃ - a)) ↔
  (a > 7/2 ∨ a < -1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_zeros_f_three_tangents_l15_1581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_simplification_l15_1537

theorem trigonometric_expression_simplification (α : ℝ) :
  (Real.sin (2*α + 2*Real.pi) + 2*Real.sin (4*α - Real.pi) + Real.sin (6*α + 4*Real.pi)) /
  (Real.cos (6*Real.pi - 2*α) + 2*Real.cos (4*α - Real.pi) + Real.cos (6*α - 4*Real.pi)) = Real.tan (4*α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_simplification_l15_1537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pelican_speed_is_33_l15_1584

/-- The speed of the eagle in miles per hour -/
def eagle_speed : ℚ := 15

/-- The speed of the falcon in miles per hour -/
def falcon_speed : ℚ := 46

/-- The speed of the hummingbird in miles per hour -/
def hummingbird_speed : ℚ := 30

/-- The time all birds flew in hours -/
def flight_time : ℚ := 2

/-- The total distance covered by all birds in miles -/
def total_distance : ℚ := 248

/-- The speed of the pelican in miles per hour -/
def pelican_speed : ℚ := (total_distance - (eagle_speed + falcon_speed + hummingbird_speed) * flight_time) / flight_time

theorem pelican_speed_is_33 : pelican_speed = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pelican_speed_is_33_l15_1584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dana_fewest_cookies_l15_1540

-- Define the volunteers
inductive Volunteer
| Alex
| Beth
| Carl
| Dana

-- Define the cookie shapes
inductive Shape
| Pentagon
| Square
| Rhombus
| Circle

-- Define the function to get the area of each shape
noncomputable def shapeArea (s : Shape) : ℝ :=
  match s with
  | Shape.Pentagon => 15
  | Shape.Square => 9
  | Shape.Rhombus => 12
  | Shape.Circle => 18

-- Define the function to get the shape for each volunteer
def volunteerShape (v : Volunteer) : Shape :=
  match v with
  | Volunteer.Alex => Shape.Pentagon
  | Volunteer.Beth => Shape.Square
  | Volunteer.Carl => Shape.Rhombus
  | Volunteer.Dana => Shape.Circle

-- Define the function to calculate the number of cookies for a given volunteer
noncomputable def cookieCount (v : Volunteer) (totalDough : ℝ) : ℝ :=
  totalDough / shapeArea (volunteerShape v)

-- Theorem: Dana makes the fewest cookies
theorem dana_fewest_cookies (totalDough : ℝ) (h : totalDough > 0) :
  ∀ v : Volunteer, cookieCount Volunteer.Dana totalDough ≤ cookieCount v totalDough :=
by
  sorry

#check dana_fewest_cookies

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dana_fewest_cookies_l15_1540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dima_has_winning_strategy_l15_1580

/-- Represents a player in the game -/
inductive Player : Type
| Gosha : Player
| Dima : Player

/-- Represents a cell on the game board -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents the game state -/
structure GameState where
  n : Nat
  board : List Cell
  currentPlayer : Player

/-- Checks if a sequence of 7 consecutive cells is formed -/
def hasWinningSequence (board : List Cell) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Cell

/-- Simulates the game given the strategies of both players -/
def playGame (n : Nat) (gosha_strategy : Strategy) (dima_strategy : Strategy) : GameState :=
  sorry

/-- Theorem stating that Dima (second player) has a winning strategy -/
theorem dima_has_winning_strategy :
  ∀ n : Nat, ∃ (strategy : Strategy),
    ∀ (gosha_strategy : Strategy),
      hasWinningSequence (playGame n gosha_strategy strategy).board ∧
      (playGame n gosha_strategy strategy).currentPlayer = Player.Dima :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dima_has_winning_strategy_l15_1580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_decrease_approx_l15_1538

/-- Represents the percent decrease in area of a square when its side length is reduced by 15% --/
noncomputable def square_area_percent_decrease : ℝ :=
  let original_area : ℝ := 27
  let new_side_length : ℝ := 3 * Real.sqrt 3 * (1 - 0.15)
  let new_area : ℝ := new_side_length ^ 2
  ((original_area - new_area) / original_area) * 100

/-- Approximate equality for real numbers with a small tolerance --/
def approx (x y : ℝ) : Prop := abs (x - y) < 0.01

/-- Notation for approximate equality --/
notation:50 x " ≈ " y => approx x y

/-- The percent decrease in area of the square is approximately 27.74% --/
theorem square_area_decrease_approx :
  square_area_percent_decrease ≈ 27.74 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_decrease_approx_l15_1538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_sum_l15_1582

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  horizontal_asymptote : ∀ x, x ≠ 0 → |p x / q x| ≤ 1 / |x|
  vertical_asymptote : ∀ ε > 0, ∃ δ > 0, ∀ x, |x + 2| < δ → |q x| < ε
  q_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  p_2 : p 2 = 4
  q_2 : q 2 = 8
  hole : p 3 = 0 ∧ q 3 = 0

/-- The sum of p and q for a rational function with specific properties -/
theorem rational_function_sum (f : RationalFunction) : 
  ∀ x, f.p x + f.q x = -2*x^2 - 2*x + 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_sum_l15_1582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_equals_sqrt3_over_3_pi_l15_1534

noncomputable section

-- Define the radius of the semi-circle
def semicircle_radius : ℝ := 2

-- Define the volume of the cone
noncomputable def cone_volume (r : ℝ) : ℝ := (1/3) * Real.pi * r^2 * (semicircle_radius^2 - r^2).sqrt

-- Theorem statement
theorem cone_volume_equals_sqrt3_over_3_pi :
  ∃ (r : ℝ), r > 0 ∧ r < semicircle_radius ∧ cone_volume r = (Real.sqrt 3 / 3) * Real.pi :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_equals_sqrt3_over_3_pi_l15_1534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l15_1521

/-- The given function f(x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.sin (x + Real.pi / 2)

/-- The function g(x) obtained by translating f(x) -/
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 6) + 2

/-- Theorem stating the properties of f and g -/
theorem f_and_g_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
  ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 7 * Real.pi / 12 → 
  3 - Real.sqrt 3 ≤ g x ∧ g x ≤ 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l15_1521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_when_m_is_two_subset_iff_m_geq_three_l15_1523

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 2 * m + 1}
def B : Set ℝ := {x | (1 / 9 : ℝ) ≤ (3 : ℝ)^x ∧ (3 : ℝ)^x ≤ 81}

-- Theorem 1: When m = 2, A ∪ B = {x | -2 ≤ x ≤ 5}
theorem union_when_m_is_two : 
  A 2 ∪ B = {x : ℝ | -2 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2: B ⊆ A if and only if m ≥ 3
theorem subset_iff_m_geq_three (m : ℝ) : 
  B ⊆ A m ↔ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_when_m_is_two_subset_iff_m_geq_three_l15_1523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marlene_shopping_cost_l15_1542

noncomputable def calculate_discounted_price (regular_price : ℝ) (quantity : ℕ) (discount_percentage : ℝ) : ℝ :=
  let total_price := regular_price * (quantity : ℝ)
  let discount_amount := total_price * (discount_percentage / 100)
  total_price - discount_amount

theorem marlene_shopping_cost :
  let shirt_price : ℝ := 50
  let pants_price : ℝ := 40
  let shoes_price : ℝ := 60
  let shirt_quantity : ℕ := 6
  let pants_quantity : ℕ := 4
  let shoes_quantity : ℕ := 3
  let shirt_discount : ℝ := 20
  let pants_discount : ℝ := 15
  let shoes_discount : ℝ := 25
  
  let shirts_cost := calculate_discounted_price shirt_price shirt_quantity shirt_discount
  let pants_cost := calculate_discounted_price pants_price pants_quantity pants_discount
  let shoes_cost := calculate_discounted_price shoes_price shoes_quantity shoes_discount
  
  shirts_cost + pants_cost + shoes_cost = 511 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marlene_shopping_cost_l15_1542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l15_1596

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_measure (t : Triangle) : 
  (t.a * Real.sin t.A + t.b * Real.sin t.B - t.c * Real.sin t.C) / (t.a * Real.sin t.B) = 2 * Real.sqrt 3 * Real.sin t.C →
  t.C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l15_1596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_max_in_triangle_l15_1599

-- Define a convex function on an interval
def ConvexOnInterval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y t : ℝ, a ≤ x → x ≤ b → a ≤ y → y ≤ b → 0 ≤ t → t ≤ 1 →
    f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y

-- Theorem statement
theorem sin_sum_max_in_triangle :
  ConvexOnInterval Real.sin 0 Real.pi →
  ∀ A B C : ℝ,
    0 < A ∧ A < Real.pi →
    0 < B ∧ B < Real.pi →
    0 < C ∧ C < Real.pi →
    A + B + C = Real.pi →
    Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_max_in_triangle_l15_1599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_8_equals_8_l15_1526

-- Define the sum of digits function
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Define the function f
def f (n : Nat) : Nat :=
  sumOfDigits (n^2 + 1)

-- Define the recursive function fₖ
def fₖ : Nat → Nat → Nat
  | 0, n => n  -- Base case for k = 0
  | 1, n => f n
  | k+1, n => f (fₖ k n)

theorem f_2016_8_equals_8 : fₖ 2016 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_8_equals_8_l15_1526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_l15_1551

/-- The number of hours it takes the first group to complete the work -/
noncomputable def first_group_hours : ℝ := 15

/-- The number of men in the second group -/
noncomputable def second_group_men : ℝ := 15

/-- The number of hours it takes the second group to complete the work -/
noncomputable def second_group_hours : ℝ := 36

/-- The work rate (portion of work done per man per hour) -/
noncomputable def work_rate (men : ℝ) (hours : ℝ) : ℝ := 1 / (men * hours)

/-- Theorem stating that the number of men in the first group is 36 -/
theorem first_group_size :
  ∃ (men : ℝ), work_rate men first_group_hours = work_rate second_group_men second_group_hours ∧ men = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_l15_1551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_f_l15_1555

noncomputable def f (x : ℝ) : ℝ := max (Real.sin x) (max (Real.cos x) ((Real.sin x + Real.cos x) / Real.sqrt 2))

theorem sum_of_max_and_min_f :
  (⨆ x, f x) + (⨅ x, f x) = 1 - Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_f_l15_1555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_function_correct_l15_1578

noncomputable def T (t : ℝ) : ℝ :=
  if 0 ≤ t ∧ t ≤ 3 then 20 * t + 40
  else if 3 < t ∧ t ≤ 7 then 100
  else 0  -- undefined for t < 0 or t > 7

theorem temperature_function_correct :
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 7 →
    (0 ≤ t ∧ t ≤ 3 → T t = 20 * t + 40) ∧
    (3 < t ∧ t ≤ 7 → T t = 100) := by
  sorry

#check temperature_function_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_function_correct_l15_1578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_root_simplification_l15_1548

theorem nested_root_simplification :
  Real.sqrt (Real.sqrt (Real.sqrt (1 / 256))) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_root_simplification_l15_1548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_well_defined_implies_a_bound_l15_1576

-- Define the function f
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log ((1 + 2^x + 4^x * a) / 3)

-- Theorem statement
theorem f_well_defined_implies_a_bound (a : ℝ) : 
  (∀ x ∈ Set.Iic 1, (1 + 2^x + 4^x * a) / 3 > 0) → a > -3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_well_defined_implies_a_bound_l15_1576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_sailing_rate_is_10_5_l15_1543

/-- Represents the ship's situation --/
structure ShipSituation where
  distance_to_shore : ℝ
  water_intake_rate : ℝ
  water_intake_time : ℝ
  sinking_capacity : ℝ
  pump_rate : ℝ

/-- Calculates the average sailing rate needed for the ship to reach shore just as it begins to sink --/
noncomputable def average_sailing_rate (s : ShipSituation) : ℝ :=
  let water_intake_per_hour := s.water_intake_rate * (60 / s.water_intake_time)
  let net_water_intake_rate := water_intake_per_hour - s.pump_rate
  let time_to_sink := s.sinking_capacity / net_water_intake_rate
  s.distance_to_shore / time_to_sink

/-- Theorem stating that the average sailing rate for the given situation is 10.5 km/hour --/
theorem ship_sailing_rate_is_10_5 :
  let s : ShipSituation := {
    distance_to_shore := 77,
    water_intake_rate := 9/4,
    water_intake_time := 11/2,
    sinking_capacity := 92,
    pump_rate := 12
  }
  average_sailing_rate s = 10.5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_sailing_rate_is_10_5_l15_1543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_no_intersection_l15_1595

-- Define the line
def line (a : ℝ) (x y : ℝ) : Prop := x + a * y - 2 = 0

-- Define the segment
def line_segment (x y : ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = 3 - 2*t ∧ y = 1 + t

-- Define the condition of no common points
def no_common_points (a : ℝ) : Prop :=
  ∀ x y : ℝ, ¬(line a x y ∧ line_segment x y)

-- The theorem to prove
theorem line_segment_no_intersection (a : ℝ) :
  no_common_points a → (a < -1 ∨ a > 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_no_intersection_l15_1595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_solution_for_a_3_solution_for_a_neg_1_l15_1518

/-- The system of inequalities has a unique solution if and only if a = 3 or a = -1 -/
theorem unique_solution_condition (a : ℝ) :
  (∃! p : ℝ × ℝ, p.1^2 + p.2^2 + 2*p.1 ≤ 1 ∧ p.1 - p.2 + a = 0) ↔ 
  (a = 3 ∨ a = -1) :=
sorry

/-- When a = 3, the unique solution is (-2, 1) -/
theorem solution_for_a_3 :
  (∃! p : ℝ × ℝ, p.1^2 + p.2^2 + 2*p.1 ≤ 1 ∧ p.1 - p.2 + 3 = 0) ∧
  ((-2 : ℝ)^2 + (1 : ℝ)^2 + 2*(-2 : ℝ) ≤ 1) ∧
  ((-2 : ℝ) - (1 : ℝ) + 3 = 0) :=
sorry

/-- When a = -1, the unique solution is (0, -1) -/
theorem solution_for_a_neg_1 :
  (∃! p : ℝ × ℝ, p.1^2 + p.2^2 + 2*p.1 ≤ 1 ∧ p.1 - p.2 + (-1) = 0) ∧
  ((0 : ℝ)^2 + (-1 : ℝ)^2 + 2*(0 : ℝ) ≤ 1) ∧
  ((0 : ℝ) - (-1 : ℝ) + (-1) = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_solution_for_a_3_solution_for_a_neg_1_l15_1518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l15_1558

open Real

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := (x - 2) * exp x - a * x^2 + 2 * a * x - 2 * a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧
    (∀ x : ℝ, (deriv (f a)) x = 0 → x = x₁ ∨ x = x₂) ∧
    (∀ x : ℝ, 0 < x → x < x₂ → f a x < -2 * a)) →
  a ∈ Set.Ioo 0 (exp 1 / 2) ∪ Set.Ioo (exp 1 / 2) (exp 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l15_1558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_points_l15_1591

/-- Given a parabola y = x^2 + (2k + 1)x - k^2 + k, where k is a real number,
    this theorem proves that the parabola always has two distinct intersection points with the x-axis,
    and finds the value of k when x_1^2 + x_2^2 = -2k^2 + 2k + 1,
    where x_1 and x_2 are the x-coordinates of these intersection points. -/
theorem parabola_intersection_points (k : ℝ) :
  let f : ℝ → ℝ := fun x ↦ x^2 + (2*k + 1)*x - k^2 + k
  let Δ := (2*k + 1)^2 - 4*(-k^2 + k)
  Δ > 0 ∧ (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + x₂^2 = -2*k^2 + 2*k + 1) → k = 0 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_points_l15_1591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l15_1569

theorem empty_subset_singleton_zero :
  ∅ ⊆ ({0} : Set ℕ) :=
by
  intro x
  intro h
  contradiction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l15_1569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_6_equals_sqrt_5_pow_6_b_6_l15_1546

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => 7/4 * a n + 9/4 * Real.sqrt (5^n - (a n)^2)

noncomputable def b : ℕ → ℝ
  | 0 => 1
  | n + 1 => 7/4 * b n + 9/4 * Real.sqrt (1 - (b n)^2)

theorem a_6_equals_sqrt_5_pow_6_b_6 :
  a 6 = (Real.sqrt 5)^6 * b 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_6_equals_sqrt_5_pow_6_b_6_l15_1546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l15_1587

theorem vector_equation_solution :
  ∃ (x y : ℝ), 
    x = -20/19 ∧ 
    y = 11/38 ∧ 
    (⟨3, 2⟩ : ℝ × ℝ) + x • (⟨5, -6⟩ : ℝ × ℝ) = 2 • ((⟨-2, 3⟩ : ℝ × ℝ) + y • (⟨3, 4⟩ : ℝ × ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l15_1587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moon_stats_l15_1522

/-- Represents the number of moons for each celestial body --/
def moon_counts : List Nat := [0, 0, 0, 1, 1, 2, 2, 5, 15, 18, 25]

/-- Calculates the median of a sorted list --/
def median (l : List Nat) : Option Nat :=
  let n := l.length
  if n = 0 then
    none
  else if n % 2 = 0 then
    (l.get? (n / 2 - 1)).bind (fun a =>
      (l.get? (n / 2)).map (fun b =>
        (a + b) / 2))
  else
    l.get? (n / 2)

/-- Calculates the mode of a list --/
def mode (l : List Nat) : Option Nat :=
  l.foldl (fun acc x => 
    match acc with
    | none => some x
    | some y => if l.count x > l.count y then some x else some y
  ) none

/-- Theorem stating that the median of moon_counts is 2 and the mode is 0 --/
theorem moon_stats :
  median moon_counts = some 2 ∧ mode moon_counts = some 0 := by
  sorry

#eval median moon_counts
#eval mode moon_counts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moon_stats_l15_1522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l15_1525

-- Define the ellipse parameters
noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 2

-- Define the eccentricity
noncomputable def e : ℝ := Real.sqrt 2 / 2

-- Define the vertex
def A : ℝ × ℝ := (2, 0)

-- Define the line parameter
noncomputable def k : ℝ := Real.sqrt 2

-- Define the area of triangle AMN
noncomputable def S : ℝ := 4 * Real.sqrt 2 / 5

-- Helper function for triangle area (not part of the problem, but needed for the statement)
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_problem :
  (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ↔ x^2/4 + y^2/2 = 1) ∧
  (∀ (x y : ℝ), x^2/4 + y^2/2 = 1 ∧ y = k*(x-1) →
    ∃ (M N : ℝ × ℝ), M ≠ N ∧
    (M.1^2/4 + M.2^2/2 = 1) ∧ (N.1^2/4 + N.2^2/2 = 1) ∧
    M.2 = k*(M.1-1) ∧ N.2 = k*(N.1-1) ∧
    area_triangle A M N = S) ∧
  (a > b ∧ b > 0 ∧ A = (2, 0) ∧ e = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l15_1525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_catch_up_time_is_one_hour_l15_1560

/-- Represents the time taken for the naval vessel to catch up with the fishing boat -/
def catch_up_time (t : ℝ) : Prop :=
  ∃ (x y : ℝ),
    -- Distance equation
    (18 * t)^2 = (6 * Real.sqrt 3)^2 + (6 * Real.sqrt 3 * t)^2 - 2 * (6 * Real.sqrt 3) * (6 * Real.sqrt 3 * t) * Real.cos (2 * Real.pi / 3) ∧
    -- Non-negative time
    t ≥ 0

/-- The theorem stating that the catch-up time is 1 hour -/
theorem catch_up_time_is_one_hour : catch_up_time 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_catch_up_time_is_one_hour_l15_1560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_parabola_180_l15_1505

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rotates a parabola 180° around its vertex -/
noncomputable def rotate180AroundVertex (p : Parabola) : Parabola :=
  { a := -p.a
    b := p.b
    c := p.c - 2 * p.a * (p.b / (2 * p.a))^2 }

theorem rotate_parabola_180 :
  let original := Parabola.mk 2 (-12) 16
  let rotated := rotate180AroundVertex original
  rotated = Parabola.mk (-2) 12 (-20) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_parabola_180_l15_1505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_time_period_l15_1567

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Compound interest calculation -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

theorem simple_interest_time_period : 
  ∃ (t : ℝ), 
    let simple_principal := 1272.000000000001
    let simple_rate := 0.10
    let compound_principal := 5000
    let compound_rate := 0.12
    let compound_time := 2
    simple_interest simple_principal simple_rate t = 
      (1/2) * compound_interest compound_principal compound_rate compound_time ∧ 
    t = 5 := by
  sorry

#check simple_interest_time_period

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_time_period_l15_1567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_theorem_l15_1590

/-- Calculates the total round trip time given the distance to work and average speeds -/
noncomputable def round_trip_time (distance_to_work : ℝ) (speed_to_work : ℝ) (speed_return : ℝ) : ℝ :=
  distance_to_work / speed_to_work + distance_to_work / speed_return

theorem round_trip_theorem (distance_to_work : ℝ) (speed_to_work : ℝ) (speed_return : ℝ) 
  (h1 : distance_to_work = 72)
  (h2 : speed_to_work = 60)
  (h3 : speed_return = 90) :
  round_trip_time distance_to_work speed_to_work speed_return = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_theorem_l15_1590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factor_implies_values_l15_1571

theorem polynomial_factor_implies_values (a c : ℤ) : 
  (∃ (p : Polynomial ℤ), a • X^19 + c • X^18 + (1 : Polynomial ℤ) = (X^2 - X - 1) * p) → 
  (a = 1597 ∧ c = -2584) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factor_implies_values_l15_1571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l15_1545

noncomputable def f (x : ℝ) := 4 * Real.sin (x - Real.pi / 3) * Real.cos x + Real.sqrt 3

theorem f_properties :
  -- The smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- The maximum value on [-π/4, π/3] is √3 at x = π/3
  (∀ (x : ℝ), -Real.pi/4 ≤ x ∧ x ≤ Real.pi/3 → f x ≤ Real.sqrt 3) ∧
  (f (Real.pi/3) = Real.sqrt 3) ∧
  -- The minimum value on [-π/4, π/3] is -2 at x = -π/12
  (∀ (x : ℝ), -Real.pi/4 ≤ x ∧ x ≤ Real.pi/3 → -2 ≤ f x) ∧
  (f (-Real.pi/12) = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l15_1545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_l15_1585

noncomputable def sequenceLimit (n : ℝ) : ℝ := 
  (4 * n^2 - n^(3/4)) / (n^2 * (1 + 1/n^3 + 1/n^6)^(1/3) - 5*n)

theorem sequence_limit : 
  ∀ ε > 0, ∃ N : ℝ, ∀ n ≥ N, |sequenceLimit n - 4| < ε :=
by sorry

#check sequence_limit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_l15_1585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_range_m_range_l15_1507

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x) + Real.sqrt (1 - x)

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := f x / (2 * Real.sqrt (1 - x^2) + 6)

-- Theorem for the domain and range of f
theorem f_domain_and_range :
  (∀ x ∈ Set.Icc (-1) 1, f x ∈ Set.Icc (Real.sqrt 2) 2) ∧
  (∀ y ∈ Set.Icc (Real.sqrt 2) 2, ∃ x ∈ Set.Icc (-1) 1, f x = y) :=
sorry

-- Theorem for the range of m
theorem m_range (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, ∀ a ∈ Set.Icc (-1) 1,
    h x ≤ 3/4 * m^2 - 1/2 * a * m) →
  m ∈ Set.Iic (-1) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_range_m_range_l15_1507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_luke_bought_two_games_from_friend_l15_1598

/-- The number of games Luke bought from his friend -/
def games_from_friend : ℕ := 2

/-- The number of games Luke bought at the garage sale -/
def games_from_garage_sale : ℕ := 2

/-- The number of games that didn't work -/
def broken_games : ℕ := 2

/-- The number of good games Luke ended up with -/
def good_games : ℕ := 2

/-- Theorem stating that Luke bought 2 games from his friend -/
theorem luke_bought_two_games_from_friend :
  games_from_friend = 2 :=
by
  -- The equation from the problem
  have h : games_from_friend + games_from_garage_sale - broken_games = good_games := by sorry
  -- Substituting known values
  have h2 : games_from_friend + 2 - 2 = 2 := by sorry
  -- Simplifying
  have h3 : games_from_friend = 2 := by sorry
  -- Concluding the proof
  exact h3

#check luke_bought_two_games_from_friend

end NUMINAMATH_CALUDE_ERRORFEEDBACK_luke_bought_two_games_from_friend_l15_1598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_derivative_of_f_l15_1528

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 3) / x^2

-- State the theorem
theorem fourth_derivative_of_f (x : ℝ) (h : x > 0) :
  (deriv^[4] f) x = (-154 + 120 * Real.log x) / (x^6 * Real.log 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_derivative_of_f_l15_1528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_x_intercept_l15_1531

noncomputable section

/-- The function f(x) = x^3 + 4x + 5 -/
def f (x : ℝ) : ℝ := x^3 + 4*x + 5

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 4

/-- The point of tangency -/
def a : ℝ := 1

/-- The slope of the tangent line at x = 1 -/
def m : ℝ := f' a

/-- The y-coordinate of the point of tangency -/
def b : ℝ := f a

/-- The x-intercept of the tangent line -/
def x_intercept : ℝ := -3/7

theorem tangent_line_x_intercept :
  ∃ (y : ℝ), y = m * (x_intercept - a) + b ∧ y = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_x_intercept_l15_1531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_is_101_l15_1559

/-- Represents a set of four different non-zero digits -/
structure FourDigits where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  d1_nonzero : d1 ≠ 0
  d2_nonzero : d2 ≠ 0
  d3_nonzero : d3 ≠ 0
  d4_nonzero : d4 ≠ 0
  d1_lt_10 : d1 < 10
  d2_lt_10 : d2 < 10
  d3_lt_10 : d3 < 10
  d4_lt_10 : d4 < 10
  all_different : d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4

/-- The sum of all 24 four-digit numbers formed using the given digits -/
def sum_of_numbers (digits : FourDigits) : Nat :=
  6666 * (digits.d1 + digits.d2 + digits.d3 + digits.d4)

/-- Theorem stating that the largest prime factor of the sum is 101 -/
theorem largest_prime_factor_is_101 (digits : FourDigits) :
  (Nat.factors (sum_of_numbers digits)).maximum? = some 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_is_101_l15_1559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_probability_l15_1563

-- Define the isosceles right triangle
def IsoscelesRightTriangle (A B C : ℝ × ℝ) : Prop :=
  let d := Real.sqrt 2
  ‖B - A‖ = d ∧ ‖C - A‖ = 1 ∧ ‖C - B‖ = 1

-- Define the probability function
noncomputable def Probability (A B : ℝ × ℝ) : ℝ :=
  let d := Real.sqrt 2
  1 / d

-- Theorem statement
theorem isosceles_right_triangle_probability 
  (A B C : ℝ × ℝ) (h : IsoscelesRightTriangle A B C) : 
  Probability A B = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_probability_l15_1563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_state_is_white_l15_1594

/-- Represents the color of a chameleon -/
inductive Color
  | Blue
  | White
  | Red

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  blue : Nat
  white : Nat
  red : Nat

/-- The initial state of chameleons on the island -/
def initialState : ChameleonState :=
  { blue := 800, white := 1000, red := 1220 }

/-- The total number of chameleons -/
def totalChameleons : Nat := 2020

/-- Function to model the color change when two chameleons of different colors meet -/
def colorChange (c1 c2 : Color) : Color :=
  match c1, c2 with
  | Color.Blue, Color.White => Color.Red
  | Color.White, Color.Blue => Color.Red
  | Color.Blue, Color.Red => Color.White
  | Color.Red, Color.Blue => Color.White
  | Color.White, Color.Red => Color.Blue
  | Color.Red, Color.White => Color.Blue
  | _, _ => c1  -- No change if colors are the same

/-- Predicate to check if all chameleons are the same color -/
def allSameColor (state : ChameleonState) : Prop :=
  (state.blue = totalChameleons ∧ state.white = 0 ∧ state.red = 0) ∨
  (state.blue = 0 ∧ state.white = totalChameleons ∧ state.red = 0) ∨
  (state.blue = 0 ∧ state.white = 0 ∧ state.red = totalChameleons)

/-- Main theorem: The only possible final state where all chameleons are the same color is when they are all white -/
theorem final_state_is_white :
  ∀ (finalState : ChameleonState),
    (∃ (states : List ChameleonState),
      states.head? = some initialState ∧
      states.getLast? = some finalState ∧
      (∀ (i : Nat) (s1 s2 : ChameleonState),
        i < states.length - 1 →
        states.get? i = some s1 →
        states.get? (i + 1) = some s2 →
        ∃ (c1 c2 : Color),
          colorChange c1 c2 ≠ c1 ∧
          colorChange c1 c2 ≠ c2 ∧
          s2.blue + s2.white + s2.red = totalChameleons)) →
    allSameColor finalState →
    finalState.white = totalChameleons ∧ finalState.blue = 0 ∧ finalState.red = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_state_is_white_l15_1594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_muffin_probability_l15_1572

theorem cake_muffin_probability (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) :
  total = 100 →
  cake = 50 →
  muffin = 40 →
  both = 15 →
  (total - (cake + muffin - both)) / total = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_muffin_probability_l15_1572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_polygon_perimeter_l15_1550

/-- A polygon that is part of a hexagon -/
structure PartialHexagonPolygon where
  hexagon_side_length : ℝ
  occupied_sides : ℕ
  congruent_sides : ℕ

/-- Calculate the perimeter of the polygon -/
def polygon_perimeter (p : PartialHexagonPolygon) : ℝ :=
  p.hexagon_side_length * (p.occupied_sides + p.congruent_sides : ℝ)

/-- Theorem: The perimeter of the specific polygon is 32 units -/
theorem specific_polygon_perimeter :
  ∃ (p : PartialHexagonPolygon),
    p.hexagon_side_length = 8 ∧
    p.occupied_sides = 3 ∧
    p.congruent_sides = 1 ∧
    polygon_perimeter p = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_polygon_perimeter_l15_1550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_theorem_l15_1506

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  let vecBA := (t.A.1 - t.B.1, t.A.2 - t.B.2)
  let vecAC := (t.C.1 - t.A.1, t.C.2 - t.A.2)
  (vecBA.1 * vecAC.1 + vecBA.2 * vecAC.2 = 6) ∧
  (t.b - t.c = 2) ∧
  (Real.tan (Real.arctan ((t.C.2 - t.A.2) / (t.C.1 - t.A.1))) = -Real.sqrt 15)

-- Define the altitude
noncomputable def altitude (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  2 * Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c)) / t.a

-- State the theorem
theorem altitude_theorem (t : Triangle) :
  satisfies_conditions t → altitude t = 3 * Real.sqrt 15 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_theorem_l15_1506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l15_1508

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the line 3x + 4y + a = 0
def line (a : ℝ) (x y : ℝ) : Prop := 3*x + 4*y + a = 0

-- Define the circle x^2 + y^2 = 1
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (a > 0) →  -- a is positive
  (a ≠ 1) →  -- a is not equal to 1
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) →  -- f is increasing on ℝ
  (¬ ∃ x y : ℝ, line a x y ∧ unit_circle x y) →  -- line doesn't intersect circle
  a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l15_1508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_in_special_triangle_l15_1536

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where a = 1, b = 1, and c = √2, prove that sin A = √2 / 2 -/
theorem sin_A_in_special_triangle (A B C : ℝ) (a b c : ℝ) :
  a = 1 →
  b = 1 →
  c = Real.sqrt 2 →
  Real.sin A = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_in_special_triangle_l15_1536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_is_periodic_l15_1517

-- Define the concept of a periodic function
def IsPeriodic (f : ℝ → ℝ) : Prop := 
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x, f (x + T) = f x

-- Define the concept of a trigonometric function
def IsTrigonometric (f : ℝ → ℝ) : Prop :=
  f = Real.cos ∨ f = Real.sin ∨ f = Real.tan

-- State the theorem
theorem cos_is_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →  -- All trigonometric functions are periodic
  IsTrigonometric Real.cos →                         -- cos is a trigonometric function
  IsPeriodic Real.cos                                -- Therefore, cos is periodic
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_is_periodic_l15_1517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_outside_circle_l15_1579

-- Define the line l: ax + by = 1
def line (a b x y : ℝ) : Prop := a * x + b * y = 1

-- Define the circle C: x² + y² = 1
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point P(a, b)
def point_P (a b : ℝ) : ℝ × ℝ := (a, b)

-- Define what it means for a point to be outside a circle
def outside_circle (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + y^2 > 1

-- Theorem statement
theorem point_P_outside_circle (a b : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ 
    line a b x₁ y₁ ∧ circle_C x₁ y₁ ∧
    line a b x₂ y₂ ∧ circle_C x₂ y₂) →
  outside_circle (point_P a b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_outside_circle_l15_1579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_perfect_square_product_l15_1565

/-- The set of primes less than 30 -/
def primes_lt_30 : Finset Nat := Finset.filter (fun p => p.Prime ∧ p < 30) (Finset.range 30)

/-- A set of 2022 strictly positive integers with prime divisors less than 30 -/
def A : Finset Nat :=
  Finset.filter (fun n => n > 0 ∧ n.factorization.support ⊆ primes_lt_30) (Finset.range 10000)

/-- Main theorem -/
theorem exist_perfect_square_product (h : A.card = 2022) :
  ∃ a b c d : Nat, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ k : Nat, a * b * c * d = k ^ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_perfect_square_product_l15_1565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_phase_shift_l15_1544

/-- The function f(x) = 2 cos(2x + π/3) + 1 -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x + Real.pi / 3) + 1

/-- The phase shift of a cosine function -/
def phase_shift (f : ℝ → ℝ) : ℝ := sorry

/-- The phase shift of f is -π/6 -/
theorem f_phase_shift : phase_shift f = -Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_phase_shift_l15_1544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_l15_1597

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (age_increase : ℝ) (teacher_age : ℝ) : 
  num_students = 25 →
  student_avg_age = 12 →
  age_increase = 1.5 →
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = student_avg_age + age_increase →
  teacher_age = 51 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_l15_1597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invisible_square_exists_l15_1520

theorem invisible_square_exists (n : ℕ) : 
  ∃ (a b : ℤ), ∀ (i j : ℕ), i < n ∧ j < n → Int.gcd (a + i) (b + j) > 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_invisible_square_exists_l15_1520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_identity_l15_1570

theorem cosine_sum_identity (α β : Real) :
  (Real.cos α * Real.cos (β/2) / Real.cos (α - β/2) + Real.cos β * Real.cos (α/2) / Real.cos (β - α/2) = 1) →
  (Real.cos α + Real.cos β = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_identity_l15_1570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l15_1575

noncomputable def f (x y : ℝ) := (x * y) / (x^2 + y^2)

theorem min_value_of_f :
  ∀ x y : ℝ, 1/3 ≤ x ∧ x ≤ 3/5 ∧ 1/4 ≤ y ∧ y ≤ 2/5 →
  f x y ≥ (2/15) * (225/61) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l15_1575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l15_1515

/-- The constant term in the expansion of (x^2 + 2/x^3)^5 is 40 -/
theorem constant_term_expansion : ∃ c : ℕ, c = 40 ∧ 
  ∀ x : ℝ, x ≠ 0 → ∃ p : ℝ → ℝ, (x^2 + 2/x^3)^5 = c + x * (p x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l15_1515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l15_1588

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 8) * x - a - a * b

-- State the theorem
theorem quadratic_function_properties :
  ∃ (a b : ℝ),
    (∀ x, x ∈ Set.Ioo (-3) 2 → f a b x > 0) ∧
    (∀ x, x ∈ Set.Iic (-3) ∪ Set.Ici 2 → f a b x < 0) →
    (∃ (A B C : ℝ), ∀ x, f a b x = A * x^2 + B * x + C ∧ A = -3 ∧ B = 5 ∧ C = 18) ∧
    (∀ c, (∀ x, a * x^2 + b * x + c ≤ 0) ↔ c ≤ -25/12) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l15_1588
