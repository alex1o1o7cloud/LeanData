import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_trace_area_l956_95620

/-- The area traced on a sphere given its radius and the area traced on another sphere -/
noncomputable def areaTraced (r₁ : ℝ) (r₂ : ℝ) (a₁ : ℝ) : ℝ :=
  a₁ * (r₂ * r₂) / (r₁ * r₁)

theorem sphere_trace_area (r_inner r_outer a_inner : ℝ) 
  (h_r_inner : r_inner = 4)
  (h_r_outer : r_outer = 6)
  (h_a_inner : a_inner = 17) :
  areaTraced r_inner r_outer a_inner = 38.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_trace_area_l956_95620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_productivity_difference_l956_95613

/-- Given Alex's work conditions over two days, prove the difference in pages produced. -/
theorem alex_productivity_difference (h : ℝ) : 
  (3 * h - 5) * (h + 3) - (3 * h * h) = 4 * h - 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_productivity_difference_l956_95613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_ratio_l956_95656

/-- Given a square with side length s₁ and an equilateral triangle with side length s₂,
    if their areas are equal, then the ratio of s₂ to s₁ is 2 * (3^(1/4)) -/
theorem equal_area_ratio (s₁ s₂ : ℝ) (h : s₁ > 0) (h' : s₂ > 0) :
  s₁^2 = (s₂^2 * Real.sqrt 3) / 4 → s₂ / s₁ = 2 * (3^(1/4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_ratio_l956_95656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_curve_length_l956_95633

/-- The length of a parametric curve described by (x, y) = (3 sin t, 3 cos t) from t = 0 to t = 2π -/
noncomputable def parametricCurveLength : ℝ := 6 * Real.pi

/-- The parametric equations of the curve -/
noncomputable def curve (t : ℝ) : ℝ × ℝ := (3 * Real.sin t, 3 * Real.cos t)

theorem parametric_curve_length :
  ∫ t in (0)..(2 * Real.pi), Real.sqrt ((3 * Real.cos t)^2 + (- 3 * Real.sin t)^2) = parametricCurveLength := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_curve_length_l956_95633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_same_suit_face_card_l956_95636

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (h_size : cards.card = 52)
  (h_valid : ∀ c ∈ cards, c.1 ∈ Finset.range 4 ∧ c.2 ∈ Finset.range 13)

/-- Represents a deck after one card has been removed -/
def RemovedDeck (d : Deck) := { c : Deck | c.cards.card = 51 ∧ c.cards ⊆ d.cards }

/-- Checks if a card is a face card (Jack, Queen, or King) -/
def isFaceCard (card : Nat × Nat) : Bool := card.2 ≥ 10

/-- Checks if two cards have the same suit -/
def sameSuit (card1 card2 : Nat × Nat) : Bool := card1.1 = card2.1

/-- The main theorem -/
theorem probability_same_suit_face_card (d : Deck) :
  ∀ rd ∈ RemovedDeck d,
  (Finset.filter (λ (c1, c2) => sameSuit c1 c2 ∧ isFaceCard c2) (rd.cards.product rd.cards)).card /
  (rd.cards.card * (rd.cards.card - 1)) = 3 / 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_same_suit_face_card_l956_95636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l956_95694

-- Define sets A and B
def A : Set ℝ := {x | |x - 3| < 2}
def B : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Set.Icc (-1) 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l956_95694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_count_l956_95641

/-- Represents a plane in 3D space -/
structure Plane where

/-- Represents the intersection of two planes -/
def intersection (p q : Plane) : Set Plane :=
  sorry

/-- Counts the number of distinct intersection lines among three planes -/
def countIntersectionLines (α β γ : Plane) : Nat :=
  sorry

/-- Theorem: Given three planes where one intersects the other two, 
    the number of intersection lines can be 1, 2, or 3 -/
theorem intersection_line_count 
  (α β γ : Plane) 
  (h1 : intersection α β ≠ ∅) 
  (h2 : intersection α γ ≠ ∅) : 
  countIntersectionLines α β γ ∈ ({1, 2, 3} : Set Nat) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_count_l956_95641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l956_95699

-- Define the necessary structures
structure Line : Type where

structure Plane : Type where

-- Define the relationships
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

def perpendicular_plane_plane (p1 : Plane) (p2 : Plane) : Prop := sorry

def parallel (l : Line) (p : Plane) : Prop := sorry

def contained_in (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem line_plane_relationship (l : Line) (α β : Plane) :
  perpendicular_line_plane l β → perpendicular_plane_plane α β →
  (parallel l α ∨ contained_in l α) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l956_95699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_average_problem_l956_95673

theorem exam_average_problem (total_students : ℕ) (group_a : ℕ) (group_b : ℕ) 
  (avg_a : ℚ) (overall_avg : ℚ) :
  total_students = group_a + group_b →
  group_a = 15 →
  group_b = 10 →
  avg_a * group_a + 90 * group_b = overall_avg * total_students →
  overall_avg = 78 →
  avg_a = 70 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_average_problem_l956_95673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_is_2_sqrt_5_l956_95624

/-- A particle moves so that it is at (2t + 7, 4t - 13) at time t. 
    This function represents the position of the particle at time t. -/
def particlePosition (t : ℝ) : ℝ × ℝ := (2 * t + 7, 4 * t - 13)

/-- The speed of a particle is defined as the magnitude of its velocity vector. -/
noncomputable def particleSpeed (pos : ℝ → ℝ × ℝ) : ℝ :=
  let velocity t := ((pos (t + 1)).1 - (pos t).1, (pos (t + 1)).2 - (pos t).2)
  let v := velocity 0  -- Velocity is constant, so we can evaluate at any time
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

/-- Theorem stating that the speed of the particle is 2√5. -/
theorem particle_speed_is_2_sqrt_5 : 
  particleSpeed particlePosition = 2 * Real.sqrt 5 := by
  -- Unfold definitions
  unfold particleSpeed particlePosition
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_is_2_sqrt_5_l956_95624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_false_propositions_l956_95664

-- Define the propositions
def proposition1 : Prop := ∀ x > 0, (x + 1) * Real.exp x > 1

def proposition2 : Prop := ∀ x, 3 * Real.sin (2 * (x + Real.pi / 6)) = 3 * Real.sin (2 * x)

def proposition3 : Prop := ∀ total sample sampleA producedB,
  total = 4800 ∧ sample = 80 ∧ sampleA = 50 ∧ producedB = 1800 →
  (sampleA / sample : ℝ) = ((total - producedB) / total : ℝ)

def proposition4 : Prop := ∀ a b,
  (a + b = 2 → ∃ x y, x + y = 0 ∧ (x - a)^2 + (y - b)^2 = 2) ∧
  (∃ x y, x + y = 0 ∧ (x - a)^2 + (y - b)^2 = 2 → a + b = 2)

-- Theorem statement
theorem exactly_two_false_propositions :
  (¬ proposition1 ∧ proposition2 ∧ proposition3 ∧ ¬ proposition4) ∨
  (¬ proposition1 ∧ proposition2 ∧ ¬ proposition3 ∧ proposition4) ∨
  (¬ proposition1 ∧ ¬ proposition2 ∧ proposition3 ∧ proposition4) ∨
  (proposition1 ∧ ¬ proposition2 ∧ ¬ proposition3 ∧ proposition4) ∨
  (proposition1 ∧ ¬ proposition2 ∧ proposition3 ∧ ¬ proposition4) ∨
  (proposition1 ∧ proposition2 ∧ ¬ proposition3 ∧ ¬ proposition4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_false_propositions_l956_95664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l956_95616

/-- The speed of a train given its length and time to cross a fixed point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem: The speed of a train is approximately 13.33 m/s -/
theorem train_speed_approx :
  let length : ℝ := 80
  let time : ℝ := 6
  abs (train_speed length time - 13.33) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l956_95616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinates_l956_95691

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its three vertices -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem third_vertex_coordinates (x : ℝ) :
  x > 6 →
  triangleArea ⟨6, 4⟩ ⟨0, 0⟩ ⟨x, 0⟩ = 48 →
  x = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinates_l956_95691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l956_95634

theorem triangle_area_ratio (A B C D E F : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  let CF := Real.sqrt ((C.1 - F.1)^2 + (C.2 - F.2)^2)
  let AEF := abs ((A.1 * (E.2 - F.2) + E.1 * (F.2 - A.2) + F.1 * (A.2 - E.2)) / 2)
  let DBE := abs ((D.1 * (B.2 - E.2) + B.1 * (E.2 - D.2) + E.1 * (D.2 - B.2)) / 2)
  AB = 100 ∧ AC = 100 ∧ AD = 25 ∧ CF = 60 → AEF / DBE = 10.24 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l956_95634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_solution_l956_95632

noncomputable def nested_sqrt (n : ℝ) (depth : ℕ) : ℝ :=
  match depth with
  | 0 => 0
  | k + 1 => Real.sqrt (n + nested_sqrt n k)

theorem unique_integer_solution :
  ∀ n m : ℕ, nested_sqrt (n : ℝ) 1964 = m → n = 0 ∧ m = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_solution_l956_95632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l956_95695

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_ratio 
  (a b : ArithmeticSequence) 
  (h : ∀ n, sum_n a n / sum_n b n = 2 * n / (3 * n + 1)) :
  (a.a 4 + a.a 6) / (b.a 3 + b.a 7) = 9 / 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l956_95695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eternal_life_conditions_l956_95690

/-- Represents the state of a cell (alive or dead) -/
inductive CellState
| Alive
| Dead

/-- Represents a grid of cells -/
def Grid (m n : ℕ) := Fin m → Fin n → CellState

/-- The next state of a cell given its current state and number of live neighbors -/
def nextState (current : CellState) (liveNeighbors : ℕ) : CellState :=
  match current, liveNeighbors with
  | CellState.Alive, _ => CellState.Dead
  | CellState.Dead, n => if n % 2 = 1 then CellState.Alive else CellState.Dead

/-- Evolves the grid t times from the initial state -/
def evolve (m n : ℕ) (initial : Grid m n) : ℕ → Grid m n
| 0 => initial
| t + 1 => λ i j =>
    let liveNeighbors := sorry -- Count live neighbors
    nextState (evolve m n initial t i j) liveNeighbors

/-- Determines if a configuration leads to eternal life -/
def hasEternalLife (m n : ℕ) : Prop :=
  ∃ (initial : Grid m n), ∀ (t : ℕ), ∃ (i : Fin m) (j : Fin n),
    (evolve m n initial t i j) = CellState.Alive

/-- Main theorem: Characterizes the conditions for eternal life in an m × n grid -/
theorem eternal_life_conditions (m n : ℕ) :
  hasEternalLife m n ↔ (n = 2 ∨ n ≥ 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eternal_life_conditions_l956_95690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spotlight_configuration_exists_l956_95635

-- Define Point as a type alias for ℝ × ℝ
def Point := ℝ × ℝ

/-- A spotlight illuminates a convex shape in the arena -/
structure Spotlight where
  illuminated_area : Set Point

/-- The arena is a set of points -/
def Arena : Type := Set Point

/-- A configuration of spotlights in the arena -/
structure SpotlightConfiguration where
  n : ℕ
  spotlights : Fin n → Spotlight
  arena : Arena

/-- The area illuminated by a subset of spotlights -/
def illuminated_area (config : SpotlightConfiguration) (subset : Set (Fin config.n)) : Set Point :=
  ⋃ i ∈ subset, (config.spotlights i).illuminated_area

/-- A configuration is valid if it satisfies the given conditions -/
def is_valid_configuration (config : SpotlightConfiguration) : Prop :=
  (config.n ≥ 2) ∧
  (∀ i : Fin config.n, illuminated_area config {j | j ≠ i} = config.arena) ∧
  (∀ i j : Fin config.n, i ≠ j →
    illuminated_area config {k | k ≠ i ∧ k ≠ j} ≠ config.arena)

theorem spotlight_configuration_exists :
  ∀ n : ℕ, n ≥ 2 → ∃ config : SpotlightConfiguration, config.n = n ∧ is_valid_configuration config := by
  sorry

#check spotlight_configuration_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spotlight_configuration_exists_l956_95635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l956_95614

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.sin x ≠ x - 1) ↔ (∃ x : ℝ, Real.sin x = x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l956_95614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_triangle_properties_l956_95646

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) * (Real.cos x) + Real.sqrt 3 * Real.sin (2 * x)

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem function_and_triangle_properties :
  ∃ (T : ℝ) (abc : Triangle),
    (∀ x : ℝ, f (x + T) = f x) ∧  -- f has period T
    (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧  -- T is the smallest positive period
    T = Real.pi ∧  -- The period is π
    f abc.A = 2 ∧  -- f(A) = 2
    abc.a = Real.sqrt 3 ∧  -- a = √3
    abc.b + abc.c = 3 ∧  -- b + c = 3
    abc.b > abc.c ∧  -- b > c
    abc.b = 2 ∧  -- b = 2
    abc.c = 1 :=  -- c = 1
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_triangle_properties_l956_95646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eddy_travel_time_is_three_hours_l956_95618

/-- Represents the travel time and distances for Eddy and Freddy -/
structure TravelData where
  distanceAB : ℝ  -- Distance from A to B
  distanceAC : ℝ  -- Distance from A to C
  freddyTime : ℝ  -- Freddy's travel time
  speedRatio : ℝ  -- Ratio of Eddy's speed to Freddy's speed

/-- Calculates Eddy's travel time given the travel data -/
noncomputable def eddyTravelTime (data : TravelData) : ℝ :=
  data.distanceAB / (data.speedRatio * (data.distanceAC / data.freddyTime))

/-- Theorem stating that Eddy's travel time is 3 hours given the specific conditions -/
theorem eddy_travel_time_is_three_hours :
  let data : TravelData := {
    distanceAB := 600,
    distanceAC := 460,
    freddyTime := 4,
    speedRatio := 1.7391304347826086
  }
  eddyTravelTime data = 3 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eddy_travel_time_is_three_hours_l956_95618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l956_95639

-- Define set A
def A : Set ℝ := {x : ℝ | |x - 1| < 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 4*x + 3 < 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l956_95639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l956_95602

/-- A quadrilateral with specific properties -/
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)
  (right_angle_F : (F.1 - E.1) * (G.1 - F.1) + (F.2 - E.2) * (G.2 - F.2) = 0)
  (right_angle_H : (H.1 - G.1) * (E.1 - H.1) + (H.2 - G.2) * (E.2 - H.2) = 0)
  (diagonal_length : Real.sqrt ((E.1 - G.1)^2 + (E.2 - G.2)^2) = 5)
  (distinct_integer_sides : ∃ (a b : ℕ), a ≠ b ∧ 
    (Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) = a ∨
     Real.sqrt ((F.1 - G.1)^2 + (F.2 - G.2)^2) = a ∨
     Real.sqrt ((G.1 - H.1)^2 + (G.2 - H.2)^2) = a ∨
     Real.sqrt ((H.1 - E.1)^2 + (H.2 - E.2)^2) = a) ∧
    (Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) = b ∨
     Real.sqrt ((F.1 - G.1)^2 + (F.2 - G.2)^2) = b ∨
     Real.sqrt ((G.1 - H.1)^2 + (G.2 - H.2)^2) = b ∨
     Real.sqrt ((H.1 - E.1)^2 + (H.2 - E.2)^2) = b))

/-- The area of a quadrilateral with the given properties is 12 -/
theorem quadrilateral_area (q : Quadrilateral) : 
  abs ((q.E.1 - q.G.1) * (q.F.2 - q.H.2) - (q.E.2 - q.G.2) * (q.F.1 - q.H.1)) / 2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l956_95602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_36_l956_95680

theorem probability_factor_of_36 : 
  (Finset.filter (λ n : ℕ => n ∣ 36) (Finset.range 37)).card / 36 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_36_l956_95680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_theorem_l956_95685

open Real

def circle_radius_proof (A₁ A₂ : ℝ) : Prop :=
  let larger_radius : ℝ := 4
  let larger_area : ℝ := A₁ + A₂
  let smaller_area : ℝ := A₁
  (larger_area = Real.pi * larger_radius^2) ∧
  (A₂ = (larger_area - smaller_area) / 2) ∧
  (∃ (smaller_radius : ℝ), smaller_area = Real.pi * smaller_radius^2 ∧ smaller_radius = 4 * sqrt 3 / 3)

theorem circle_radius_theorem (A₁ A₂ : ℝ) :
  circle_radius_proof A₁ A₂ := by
  sorry

#check circle_radius_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_theorem_l956_95685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_l956_95622

/-- Given a function f(x) = (1/2)x^2 - 2ax + b*ln(x) + 2a^2 that achieves an extremum at x = 1 with value 1/2, 
    prove that a + b = -1 -/
theorem extremum_condition (a b : ℝ) : 
  let f := fun x : ℝ => (1/2) * x^2 - 2*a*x + b * Real.log x + 2*a^2
  (f 1 = 1/2) → (deriv f 1 = 0) → (a + b = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_l956_95622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_from_series_l956_95681

theorem cos_double_angle_from_series (θ : ℝ) :
  (∑' (n : ℕ), (Real.cos θ)^(2*n) = 5) → Real.cos (2*θ) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_from_series_l956_95681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paulson_savings_increase_l956_95689

/-- Represents the percentage increase in savings given changes in income and expenditure -/
noncomputable def savings_increase (initial_spending_ratio : ℝ) (income_increase : ℝ) (expenditure_increase : ℝ) : ℝ :=
  let initial_savings_ratio := 1 - initial_spending_ratio
  let new_income_ratio := 1 + income_increase
  let new_expenditure_ratio := initial_spending_ratio * (1 + expenditure_increase)
  let new_savings_ratio := new_income_ratio - new_expenditure_ratio
  (new_savings_ratio / initial_savings_ratio - 1) * 100

/-- Theorem stating that under the given conditions, the savings increase is 50% -/
theorem paulson_savings_increase :
  savings_increase 0.75 0.20 0.10 = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paulson_savings_increase_l956_95689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airport_travel_ratio_l956_95638

/-- Represents the time components of Mary's business trip -/
structure TripTime where
  uberToHouse : ℕ
  checkBag : ℕ
  waitForBoarding : ℕ
  airportTravel : ℕ

/-- Calculates the total trip time in minutes -/
def totalTripTime (t : TripTime) : ℕ :=
  t.uberToHouse + t.checkBag + 3 * t.checkBag + t.waitForBoarding + 2 * t.waitForBoarding + t.airportTravel

/-- The ratio of airport travel time to Uber-to-house time is 5:1 -/
theorem airport_travel_ratio (t : TripTime) : 
  t.uberToHouse = 10 ∧ 
  t.checkBag = 15 ∧ 
  t.waitForBoarding = 20 ∧ 
  totalTripTime t = 180 → 
  t.airportTravel / t.uberToHouse = 5 := by
  sorry

-- Remove the #eval line as it's not necessary and was causing an error

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airport_travel_ratio_l956_95638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_transformation_properties_l956_95659

-- Define the transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)

-- Define the triangle
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

-- Define the centroid
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.D.1 + t.E.1 + t.F.1) / 3, (t.D.2 + t.E.2 + t.F.2) / 3)

-- Define the area of a triangle
noncomputable def area (t : Triangle) : ℝ :=
  abs ((t.D.1 * (t.E.2 - t.F.2) + t.E.1 * (t.F.2 - t.D.2) + t.F.1 * (t.D.2 - t.E.2)) / 2)

-- Define the slope of a line
noncomputable def slopeLine (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

theorem triangle_transformation_properties
  (t : Triangle)
  (h1 : t.D.1 > 0 ∧ t.D.2 > 0)
  (h2 : t.E.1 > 0 ∧ t.E.2 > 0)
  (h3 : t.F.1 > 0 ∧ t.F.2 > 0) :
  let t' := Triangle.mk (transform t.D) (transform t.E) (transform t.F)
  (centroid t').2 < 0 ∧
  area t = area t' ∧
  slopeLine t.D (transform t.D) = slopeLine t.F (transform t.F) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_transformation_properties_l956_95659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_slope_l956_95676

-- Define the parabola
def parabola (x y : ℝ) : Prop := y ^ 2 = 4 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def on_parabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2

-- Define a point in the first quadrant
def in_first_quadrant (P : ℝ × ℝ) : Prop := P.1 > 0 ∧ P.2 > 0

-- Define the vector relationship
def vector_relationship (P Q : ℝ × ℝ) : Prop :=
  3 * (P.1 - focus.1, P.2 - focus.2) = (Q.1 - focus.1, Q.2 - focus.2)

-- Define the slope of a line given two points
noncomputable def line_slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

-- The main theorem
theorem parabola_intersection_slope (P Q : ℝ × ℝ) :
  on_parabola P → on_parabola Q →
  in_first_quadrant Q →
  vector_relationship P Q →
  line_slope P Q = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_slope_l956_95676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_minimized_at_median_sum_distances_minimum_only_at_median_l956_95630

/-- Represents a point on a line -/
structure Point where
  x : ℝ

/-- Represents a line with 9 ordered points -/
structure Line where
  points : Fin 9 → Point
  ordered : ∀ i j, i < j → (points i).x < (points j).x

/-- The sum of distances from a point Q to all points on the line -/
def sumOfDistances (l : Line) (Q : Point) : ℝ :=
  Finset.sum Finset.univ (fun i => |Q.x - (l.points i).x|)

/-- Theorem: The sum of distances is minimized when Q is at the median (Q₅) -/
theorem sum_distances_minimized_at_median (l : Line) :
  ∀ Q, sumOfDistances l Q ≥ sumOfDistances l (l.points 4) := by
  sorry

/-- Theorem: The minimum occurs only at the median (Q₅) -/
theorem sum_distances_minimum_only_at_median (l : Line) :
  ∀ Q, Q ≠ l.points 4 → sumOfDistances l Q > sumOfDistances l (l.points 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_minimized_at_median_sum_distances_minimum_only_at_median_l956_95630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_speed_approx_l956_95666

/-- The speed of a man running opposite to a train, given the train's characteristics and passing time --/
noncomputable def mans_speed (train_length : ℝ) (train_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed * 1000 / 3600
  let relative_speed := train_length / passing_time
  let mans_speed_mps := relative_speed - train_speed_mps
  mans_speed_mps * 3600 / 1000

/-- Theorem stating the man's speed given the problem conditions --/
theorem mans_speed_approx (train_length : ℝ) (train_speed : ℝ) (passing_time : ℝ)
    (h1 : train_length = 275)
    (h2 : train_speed = 60)
    (h3 : passing_time = 15) :
    ∃ ε > 0, |mans_speed train_length train_speed passing_time - 5.98| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_speed_approx_l956_95666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_equality_l956_95697

theorem trigonometric_sum_equality : 
  Real.sin (π/8)^4 + Real.cos (3*π/8)^4 + Real.sin (5*π/8)^4 + Real.cos (7*π/8)^4 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_equality_l956_95697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_network_resistance_l956_95671

/-- The equivalent resistance of a network of 13 identical resistors -/
noncomputable def equivalent_resistance (R₀ : ℝ) : ℝ :=
  4 * R₀ / 10

/-- Theorem stating that the equivalent resistance of the network is 4 * R₀ / 10 -/
theorem network_resistance (R₀ : ℝ) (h : R₀ > 0) :
  equivalent_resistance R₀ = 4 * R₀ / 10 := by
  -- Unfold the definition of equivalent_resistance
  unfold equivalent_resistance
  -- The equality holds by definition
  rfl

#check network_resistance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_network_resistance_l956_95671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l956_95645

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
  (h_arithmetic : ∃ d : ℝ, ∀ n, a (n + 1) = a n + d) 
  (h_sub_sequence : 2 * a 5 = a 3 + a 6) :
  (a 3 + a 5) / (a 4 + a 6) = 1 ∨ (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l956_95645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_third_term_l956_95648

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_third_term
  (seq : ArithmeticSequence)
  (h1 : seq.a 4 + seq.a 5 = 24)
  (h2 : S seq 6 = 48) :
  seq.a 3 = 6 := by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_third_term_l956_95648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l956_95640

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- One of the bases of the trapezoid -/
  a : ℝ
  /-- The side length of the trapezoid -/
  l : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The trapezoid is circumscribed around a circle -/
  isCircumscribed : True
  /-- The base and side length are positive -/
  a_pos : a > 0
  l_pos : l > 0
  /-- The base is not longer than the perimeter -/
  a_le_2l : a ≤ 2 * l

/-- The area of an isosceles trapezoid circumscribed around a circle -/
noncomputable def area (t : IsoscelesTrapezoid) : ℝ :=
  t.l * Real.sqrt (t.a * (2 * t.l - t.a))

/-- Theorem: The area of an isosceles trapezoid circumscribed around a circle
    is equal to l * sqrt(a(2l - a)) -/
theorem isosceles_trapezoid_area (t : IsoscelesTrapezoid) :
  area t = t.l * Real.sqrt (t.a * (2 * t.l - t.a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l956_95640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l956_95660

/-- The function f(x) = x² for x > 0 -/
noncomputable def f (x : ℝ) : ℝ := x^2

/-- The derivative of f(x) -/
noncomputable def f_derivative (x : ℝ) : ℝ := 2 * x

/-- The equation of the tangent line at point (a, f(a)) -/
def tangent_line (a : ℝ) (x y : ℝ) : Prop :=
  2 * a * x - y - a^2 = 0

/-- The x-intercept of the tangent line -/
noncomputable def x_intercept (a : ℝ) : ℝ := a / 2

/-- The y-intercept of the tangent line -/
noncomputable def y_intercept (a : ℝ) : ℝ := -a^2

/-- The area of the triangle formed by the tangent line and the coordinate axes -/
noncomputable def triangle_area (a : ℝ) : ℝ :=
  1 / 2 * x_intercept a * (abs (y_intercept a))

theorem tangent_triangle_area (a : ℝ) :
  a > 0 → triangle_area a = 2 → a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l956_95660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l956_95621

theorem boat_speed_in_still_water 
  (a c : ℝ) 
  (h1 : a > 0) 
  (h2 : c > 0) :
  ∃ x : ℝ, 
    x > c ∧ 
    a / x + a / (2 * (x - c)) = 1 ∧
    x = (3 * a + 2 * c + Real.sqrt (9 * a^2 - 4 * a * c + 4 * c^2)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l956_95621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_can_form_square_l956_95643

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  smallBase : ℝ
  largeBase : ℝ
  height : ℝ

/-- Calculate the area of an isosceles trapezoid -/
noncomputable def trapezoidArea (t : IsoscelesTrapezoid) : ℝ :=
  (t.smallBase + t.largeBase) * t.height / 2

/-- The side length of a square with the same area as the trapezoid -/
noncomputable def equivalentSquareSide (t : IsoscelesTrapezoid) : ℝ :=
  Real.sqrt (trapezoidArea t)

theorem trapezoid_can_form_square (t : IsoscelesTrapezoid)
    (h1 : t.smallBase = 4)
    (h2 : t.largeBase = 12)
    (h3 : t.height = 4) :
    trapezoidArea t = equivalentSquareSide t ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_can_form_square_l956_95643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mirror_symmetric_equal_volume_l956_95669

/-- A polyhedron in 3D space -/
structure Polyhedron where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Mirror symmetry between two polyhedra -/
def mirror_symmetric (p1 p2 : Polyhedron) : Prop :=
  sorry -- Definition of mirror symmetry

/-- Volume of a polyhedron -/
def volume (p : Polyhedron) : ℝ :=
  sorry -- Definition of volume

/-- Theorem: Mirror-symmetric polyhedra have equal volumes -/
theorem mirror_symmetric_equal_volume (p1 p2 : Polyhedron) :
  mirror_symmetric p1 p2 → volume p1 = volume p2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mirror_symmetric_equal_volume_l956_95669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_squared_l956_95623

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → Point
  isEquilateral : ∀ i j : Fin 3, i ≠ j → 
    (vertices i).x^2 + (vertices i).y^2 + (vertices j).x^2 + (vertices j).y^2 - 
    2*((vertices i).x*(vertices j).x + (vertices i).y*(vertices j).y) = 
    (vertices 0).x^2 + (vertices 0).y^2 + (vertices 1).x^2 + (vertices 1).y^2 - 
    2*((vertices 0).x*(vertices 1).x + (vertices 0).y*(vertices 1).y)

/-- Calculates the centroid of a triangle -/
noncomputable def centroid (t : EquilateralTriangle) : Point :=
  { x := ((t.vertices 0).x + (t.vertices 1).x + (t.vertices 2).x) / 3,
    y := ((t.vertices 0).y + (t.vertices 1).y + (t.vertices 2).y) / 3 }

/-- Checks if a point lies on the hyperbola xy = 4 -/
def onHyperbola (p : Point) : Prop :=
  p.x * p.y = 4

/-- The main theorem -/
theorem equilateral_triangle_area_squared 
  (t : EquilateralTriangle) 
  (h1 : ∀ i : Fin 3, onHyperbola (t.vertices i))
  (h2 : centroid t = Point.mk 2 2)
  (h3 : onHyperbola (Point.mk 2 2)) : 
  (3 * Real.sqrt 3 / 4 * ((t.vertices 0).x - (t.vertices 1).x)^2)^2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_squared_l956_95623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_focus_l956_95626

/-- The parabola defined by y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (2, 0)

/-- The distance from a point (x, y) to the focus -/
noncomputable def distance_to_focus (x y : ℝ) : ℝ := Real.sqrt ((x - 2)^2 + y^2)

/-- Theorem stating the minimum distance from a point on the parabola to its focus -/
theorem min_distance_to_focus :
  ∃ (d : ℝ), d = 2 ∧ 
  (∀ (x y : ℝ), parabola x y → distance_to_focus x y ≥ d) ∧
  (∃ (x y : ℝ), parabola x y ∧ distance_to_focus x y = d) := by
  sorry

#check min_distance_to_focus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_focus_l956_95626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l956_95619

noncomputable def A : ℂ := Complex.ofReal 3 - Complex.I
noncomputable def B : ℂ := Complex.ofReal 2 - 2 * Complex.I
noncomputable def C : ℂ := Complex.ofReal 1 + 5 * Complex.I

noncomputable def triangle_area (z₁ z₂ z₃ : ℂ) : ℝ :=
  1/2 * abs ((z₁.re * (z₂.im - z₃.im) + z₂.re * (z₃.im - z₁.im) + z₃.re * (z₁.im - z₂.im)))

theorem area_of_triangle_ABC : triangle_area A B C = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l956_95619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_property_unique_solution_base_10_no_solution_base_12_l956_95692

-- Define the base of the number system
def g : ℕ := 10

-- Define the arithmetic mean function
def arithmetic_mean (x y : ℕ) : ℚ := (x + y) / 2

-- Define the geometric mean function
noncomputable def geometric_mean (x y : ℕ) : ℝ := Real.sqrt (x * y)

-- Define a function to check if a number is two-digit in base g
def is_two_digit (n : ℕ) : Prop := n ≥ g ∧ n < g^2

-- Define a function to interchange digits of a two-digit number in base g
def interchange_digits (n : ℕ) : ℕ :=
  let units := n % g
  let tens := n / g
  units * g + tens

-- Main theorem
theorem arithmetic_geometric_mean_property :
  ∃ (x y : ℕ), x ≠ y ∧ x > 0 ∧ y > 0 ∧
  is_two_digit (Int.toNat ⌊arithmetic_mean x y⌋) ∧
  (Int.toNat ⌊geometric_mean x y⌋ = interchange_digits (Int.toNat ⌊arithmetic_mean x y⌋)) ∧
  ((x = 98 ∧ y = 32) ∨ (x = 32 ∧ y = 98)) :=
sorry

-- Uniqueness in base 10
theorem unique_solution_base_10 :
  ∀ (x y : ℕ), x ≠ y ∧ x > 0 ∧ y > 0 ∧
  is_two_digit (Int.toNat ⌊arithmetic_mean x y⌋) ∧
  (Int.toNat ⌊geometric_mean x y⌋ = interchange_digits (Int.toNat ⌊arithmetic_mean x y⌋)) →
  ((x = 98 ∧ y = 32) ∨ (x = 32 ∧ y = 98)) :=
sorry

-- No solution in base 12
theorem no_solution_base_12 :
  ¬∃ (x y : ℕ), x ≠ y ∧ x > 0 ∧ y > 0 ∧
  (let g := 12; is_two_digit (Int.toNat ⌊arithmetic_mean x y⌋)) ∧
  (Int.toNat ⌊geometric_mean x y⌋ = interchange_digits (Int.toNat ⌊arithmetic_mean x y⌋)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_property_unique_solution_base_10_no_solution_base_12_l956_95692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_four_times_negative_four_equals_one_l956_95688

theorem power_four_times_negative_four_equals_one (a : ℚ) (ha : a ≠ 0) :
  a^4 * a^(-4 : ℤ) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_four_times_negative_four_equals_one_l956_95688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l956_95678

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x y : ℝ) (a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- The focus of the parabola y^2 = 8x -/
def focus : ℝ × ℝ := (2, 0)

/-- The asymptote of the hyperbola y^2/4 - x^2 = 1 -/
def asymptote_slope : ℝ := 2
def asymptote_intercept : ℝ := 0

theorem distance_focus_to_asymptote :
  distance_point_to_line focus.1 focus.2 asymptote_slope (-1) asymptote_intercept = 4 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l956_95678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_y_coord_l956_95611

/-- The y-coordinate of the point on the y-axis equidistant from A(3, 0) and B(2, 5) is 2 -/
theorem equidistant_point_y_coord : 
  ∃ y : ℝ, (Real.sqrt ((3 - 0)^2 + (0 - y)^2) = Real.sqrt ((2 - 0)^2 + (5 - y)^2)) ∧ y = 2 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_y_coord_l956_95611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_C_l956_95684

/-- A pentagon with a vertical line of symmetry -/
structure SymmetricPentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  symmetry : A.1 + E.1 = B.1 + D.1 ∧ C.1 = (A.1 + E.1) / 2

/-- The length of a line segment between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The area of a pentagon -/
noncomputable def pentagonArea (p : SymmetricPentagon) : ℝ :=
  sorry

theorem y_coordinate_of_C (p : SymmetricPentagon) 
  (h1 : distance p.A p.E = 5)
  (h2 : distance p.B p.D = 5)
  (h3 : pentagonArea p = 50)
  (h4 : p.A = (0, 0))
  (h5 : p.B = (0, 5))
  (h6 : p.D = (5, 5))
  (h7 : p.E = (5, 0)) :
  p.C.2 = 15 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_C_l956_95684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_y_product_bounds_l956_95661

noncomputable def x : ℕ → ℝ
  | 0 => Real.sqrt 3
  | n + 1 => x n + Real.sqrt (1 + x n ^ 2)

noncomputable def y : ℕ → ℝ
  | 0 => Real.sqrt 3
  | n + 1 => y n / (1 + Real.sqrt (1 + y n ^ 2))

theorem x_y_product_bounds (n : ℕ) : 2 < x n * y n ∧ x n * y n < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_y_product_bounds_l956_95661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l956_95650

-- Define the function f(x) = ln x - x
noncomputable def f (x : ℝ) : ℝ := Real.log x - x

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Ioo 0 (Real.exp 1) ∧
  (∀ x ∈ Set.Ioo 0 (Real.exp 1), f x ≤ f c) ∧
  f c = -1 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l956_95650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_munificence_monic_cubic_l956_95682

/-- The munificence of a polynomial on [-1, 1] -/
noncomputable def munificence (q : ℝ → ℝ) : ℝ :=
  ⨆ (x : ℝ) (h : x ∈ Set.Icc (-1) 1), |q x|

/-- A monic cubic polynomial -/
def monicCubic (b c d : ℝ) (x : ℝ) : ℝ :=
  x^3 + b*x^2 + c*x + d

theorem smallest_munificence_monic_cubic :
  ∃ (b c d : ℝ), munificence (monicCubic b c d) = 0 ∧
  ∀ (b' c' d' : ℝ), munificence (monicCubic b' c' d') ≥ 0 := by
  sorry

#check smallest_munificence_monic_cubic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_munificence_monic_cubic_l956_95682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l956_95674

/-- The function for which we want to find the vertical asymptote -/
noncomputable def f (x : ℝ) : ℝ := (2 * x - 3) / (6 * x + 4)

/-- Theorem stating that the vertical asymptote of f occurs at x = -2/3 -/
theorem vertical_asymptote_of_f :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x + 2/3| ∧ |x + 2/3| < δ → |f x| > (1 : ℝ) / ε :=
by
  sorry

#check vertical_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l956_95674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_8_l956_95637

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i => a (i + 1))

theorem a_4_equals_8 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 2) →
  (∀ n : ℕ, n > 0 → S n = a (n + 1)) →
  (∀ n : ℕ, n > 0 → S n = sequence_sum a n) →
  a 4 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_8_l956_95637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_percent_men_speak_french_l956_95665

/-- Represents the percentage of men who speak French in a company -/
noncomputable def percentage_men_speak_french (total_employees : ℝ) (men_employees : ℝ) (french_speakers : ℝ) (women_not_french : ℝ) : ℝ :=
  let women_employees := total_employees - men_employees
  let women_french_speakers := women_employees * (1 - women_not_french)
  let men_french_speakers := french_speakers - women_french_speakers
  (men_french_speakers / men_employees) * 100

/-- Theorem stating that 60% of men speak French given the company statistics -/
theorem sixty_percent_men_speak_french :
  ∀ (total_employees : ℝ),
  total_employees > 0 →
  percentage_men_speak_french
    total_employees
    (0.65 * total_employees)
    (0.40 * total_employees)
    0.9714285714285714
  = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_percent_men_speak_french_l956_95665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ba_approx_l956_95652

/-- The molar mass of Barium in g/mol -/
noncomputable def molar_mass_Ba : ℝ := 137.33

/-- The molar mass of Fluorine in g/mol -/
noncomputable def molar_mass_F : ℝ := 19.00

/-- The number of Barium atoms in BaF2 -/
def num_Ba_atoms : ℕ := 1

/-- The number of Fluorine atoms in BaF2 -/
def num_F_atoms : ℕ := 2

/-- The molar mass of BaF2 in g/mol -/
noncomputable def molar_mass_BaF2 : ℝ := molar_mass_Ba + num_F_atoms * molar_mass_F

/-- The mass percentage of Barium in BaF2 -/
noncomputable def mass_percentage_Ba : ℝ := (molar_mass_Ba / molar_mass_BaF2) * 100

theorem mass_percentage_Ba_approx :
  |mass_percentage_Ba - 78.35| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ba_approx_l956_95652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_complex_on_unit_circle_l956_95654

theorem square_of_complex_on_unit_circle (G : ℂ) : 
  Complex.abs G = 1 →  -- G is on the unit circle
  G = Complex.exp (π / 6 * Complex.I) →  -- G is at a 30° angle to the positive real axis
  G^2 = Complex.exp (π / 3 * Complex.I) := by  -- G^2 = e^(iπ/3)
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_complex_on_unit_circle_l956_95654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l956_95607

/-- The sum of the infinite geometric series with first term 1/2 and common ratio 1/2 -/
noncomputable def geometric_series_sum : ℝ := 1

/-- The first term of the geometric series -/
noncomputable def a : ℝ := 1/2

/-- The common ratio of the geometric series -/
noncomputable def r : ℝ := 1/2

/-- The nth term of the geometric series -/
noncomputable def nth_term (n : ℕ) : ℝ := a * r^n

theorem infinite_geometric_series_sum :
  Summable nth_term ∧ (∑' n, nth_term n) = geometric_series_sum := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l956_95607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_8_20_l956_95663

/-- Calculates the angle of the minute hand from 12 o'clock position at a given minute -/
noncomputable def minute_hand_angle (minute : ℕ) : ℝ :=
  (minute : ℝ) / 60 * 360

/-- Calculates the angle of the hour hand from 12 o'clock position at a given hour and minute -/
noncomputable def hour_hand_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  (hour % 12 : ℝ) * 30 + (minute : ℝ) / 60 * 30

/-- Calculates the smaller angle between two given angles (in degrees) -/
noncomputable def smaller_angle (angle1 : ℝ) (angle2 : ℝ) : ℝ :=
  min (abs (angle1 - angle2)) (360 - abs (angle1 - angle2))

/-- Theorem: At 8:20, the smaller angle between the hour-hand and minute-hand of a clock is 130° -/
theorem clock_angle_at_8_20 :
  smaller_angle (hour_hand_angle 8 20) (minute_hand_angle 20) = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_8_20_l956_95663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l956_95657

/-- Definition of a quadratic equation in standard form -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The first equation: 2x^2 = 5x - 1 -/
noncomputable def f₁ : ℝ → ℝ := λ x ↦ 2 * x^2 - 5 * x + 1

/-- The second equation: x + 1/x = 2 -/
noncomputable def f₂ : ℝ → ℝ := λ x ↦ x + 1/x - 2

/-- The third equation: (x-3)(x+1) = x^2 - 5 -/
noncomputable def f₃ : ℝ → ℝ := λ x ↦ (x - 3) * (x + 1) - (x^2 - 5)

/-- The fourth equation: 3x - y = 5 -/
noncomputable def f₄ : ℝ × ℝ → ℝ := λ (x, y) ↦ 3 * x - y - 5

theorem quadratic_equation_identification :
  is_quadratic_equation f₁ ∧
  ¬is_quadratic_equation f₂ ∧
  ¬is_quadratic_equation f₃ ∧
  ¬∃ f : ℝ → ℝ, (∀ x y, f x = f₄ (x, y)) ∧ is_quadratic_equation f :=
by sorry

#check quadratic_equation_identification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l956_95657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l956_95649

/-- Calculates the speed of a train given the parameters of two trains passing each other. -/
noncomputable def calculate_train_speed (train1_length : ℝ) (train1_speed : ℝ) (train2_length : ℝ) (crossing_time : ℝ) : ℝ :=
  let total_length := train1_length + train2_length
  let relative_speed := total_length / crossing_time
  let train1_speed_ms := train1_speed * (1000 / 3600)
  let train2_speed_ms := relative_speed - train1_speed_ms
  train2_speed_ms * (3600 / 1000)

/-- Theorem stating that the calculated speed of Train 2 is approximately 79.83 km/h -/
theorem train_speed_calculation :
  let train1_length : ℝ := 150
  let train1_speed : ℝ := 120
  let train2_length : ℝ := 350.04
  let crossing_time : ℝ := 9
  let calculated_speed := calculate_train_speed train1_length train1_speed train2_length crossing_time
  ∃ (ε : ℝ), ε > 0 ∧ |calculated_speed - 79.83| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l956_95649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minutes_per_hour_correct_l956_95668

def minutes_per_hour : ℕ :=
  let hours_of_sleep : ℕ := 8
  let minutes_of_sleep : ℕ := 480
  minutes_of_sleep / hours_of_sleep

theorem minutes_per_hour_correct : minutes_per_hour = 60 := by
  unfold minutes_per_hour
  norm_num

#eval minutes_per_hour

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minutes_per_hour_correct_l956_95668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_tangent_line_l956_95625

noncomputable section

-- Define the line and parabola
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1
def parabola (x : ℝ) : ℝ := x^2 / 4

-- Define the intersection points A and B
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x, p.1 = x ∧ p.2 = line k x ∧ p.2 = parabola x}

-- Define the property of being an obtuse triangle
def is_obtuse_triangle (A B : ℝ × ℝ) : Prop :=
  (A.1 * B.1 + A.2 * B.2) < 0

-- Define the tangent line parallel to AB
def tangent_line (k : ℝ) (x : ℝ) : ℝ := k * x + (parabola (2*k) - k * (2*k))

-- Define the area of triangle PAB
def triangle_area (k : ℝ) : ℝ :=
  (1/2) * (Real.sqrt (1 + k^2)) * (4*k^2 + 4)

-- State the theorem
theorem intersection_and_tangent_line (k : ℝ) :
  (∃ A B, A ∈ intersection_points k ∧ B ∈ intersection_points k ∧ is_obtuse_triangle A B) ∧
  (triangle_area k = 16 → ∃ c, c = Real.sqrt 3 ∨ c = -Real.sqrt 3 ∧ tangent_line c = line c) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_tangent_line_l956_95625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_solution_l956_95644

open Real

theorem alpha_beta_solution : ∃ (α β : ℝ),
  α ∈ Set.Ioo (-π/2) (π/2) ∧
  β ∈ Set.Ioo 0 π ∧
  sin (3*π - α) = sqrt 2 * cos (π/2 - β) ∧
  sqrt 3 * cos (-α) = -sqrt 2 * cos (π + β) ∧
  α = π/4 ∧
  β = π/6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_solution_l956_95644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_equilateral_triangle_in_rectangle_l956_95627

/-- The maximum area of an equilateral triangle inscribed in a 12 by 13 rectangle --/
theorem max_area_equilateral_triangle_in_rectangle : ∃ (A : ℝ),
  (∀ (s : ℝ), s > 0 → s^2 * Real.sqrt 3 / 4 ≤ A) ∧ 
  (∃ (s : ℝ), s > 0 ∧ s^2 * Real.sqrt 3 / 4 = A) ∧
  (A = 313 * Real.sqrt 3 - 468) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_equilateral_triangle_in_rectangle_l956_95627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_concentrate_water_ratio_l956_95696

/-- The ratio of concentrate to water in orange juice preparation -/
def orange_juice_ratio (concentrate : ℕ) (water : ℕ) : Prop :=
  concentrate * 3 = water

theorem orange_juice_concentrate_water_ratio :
  ∀ (concentrate : ℕ) (water : ℕ),
  orange_juice_ratio concentrate water →
  (concentrate : ℚ) / (water : ℚ) = 1 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_concentrate_water_ratio_l956_95696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_not_proportional_l956_95653

/-- A linear equation of the form ax + by = c -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0

/-- Direct proportionality between x and y -/
def DirectlyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t, y t = k * x t

/-- Inverse proportionality between x and y -/
def InverselyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t, x t * y t = k

theorem linear_equation_not_proportional (eq : LinearEquation) :
  ¬(DirectlyProportional (λ t ↦ t) (λ t ↦ (eq.c - eq.a * t) / eq.b)) ∧
  ¬(InverselyProportional (λ t ↦ t) (λ t ↦ (eq.c - eq.a * t) / eq.b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_not_proportional_l956_95653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_equals_eccentricity_l956_95677

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b
  b_le_a : b ≤ a

/-- A point on an ellipse -/
structure PointOnEllipse (a b : ℝ) extends Ellipse a b where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / a^2 + y^2 / b^2 = 1
  not_vertex : x ≠ a ∧ x ≠ -a

/-- The foci of an ellipse -/
noncomputable def foci (e : Ellipse a b) : Prod (ℝ × ℝ) (ℝ × ℝ) :=
  let c := Real.sqrt (a^2 - b^2)
  ((-c, 0), (c, 0))

/-- The incenter of a triangle -/
noncomputable def incenter (p₁ p₂ p₃ : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The intersection of a line through two points with another line segment -/
noncomputable def lineIntersection (p₁ p₂ q₁ q₂ : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The distance between two points -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse a b) : ℝ := Real.sqrt (1 - b^2 / a^2)

theorem ellipse_ratio_equals_eccentricity
  (e : Ellipse a b) (p : PointOnEllipse a b) :
  let (f₁, f₂) := foci e
  let i := incenter (p.x, p.y) f₁ f₂
  let t := lineIntersection (p.x, p.y) i f₁ f₂
  distance t i / distance i (p.x, p.y) = eccentricity e := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_equals_eccentricity_l956_95677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l956_95667

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

-- Define the theorem
theorem cubic_function_properties (a b : ℝ) :
  -- Condition 1: f(-1) = -2
  f a b (-1) = -2 →
  -- Condition 2: Tangent line at x = -1 is perpendicular to 3x - y = 0
  (3 * (3 * a * (-1)^2 + 2 * b * (-1)) = -1) →
  -- Conclusion 1: Values of a and b
  (a = -1 ∧ b = -3) ∧
  -- Conclusion 2: Range of m for monotonicity
  ∃ m : ℝ, m ∈ Set.Icc (-2) 0 ∧
    ∀ x y, x ∈ Set.Icc 0 m → y ∈ Set.Icc 0 m → x < y → f a b x < f a b y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l956_95667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l956_95651

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : Matrix.det (B^2 - 3 • B) = 88 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l956_95651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_2013_eq_neg_one_l956_95655

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => 1 - 1 / sequenceA n

def productA (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => sequenceA n * productA n

theorem product_2013_eq_neg_one : productA 2013 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_2013_eq_neg_one_l956_95655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_greater_than_function_implies_inequality_l956_95609

open Real

theorem derivative_greater_than_function_implies_inequality 
  (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, HasDerivAt f (f' x) x) 
  (h' : ∀ x, f' x > f x) : 
  f 2011 > Real.exp 2011 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_greater_than_function_implies_inequality_l956_95609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_singer_arrangements_l956_95679

theorem singer_arrangements (n : ℕ) (cant_be_first : ℕ) (must_be_last : ℕ) : 
  n = 5 → cant_be_first ≠ must_be_last → cant_be_first ≤ n → must_be_last ≤ n →
  (Nat.factorial (n - 2)) * (n - 2) = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_singer_arrangements_l956_95679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l956_95693

noncomputable def sequence_a (lambda : ℝ) : ℕ → ℝ
  | 0 => 1/2
  | n + 1 => sequence_a lambda n + lambda * (sequence_a lambda n)^2

theorem sequence_properties (lambda : ℝ) (h : lambda > 0) :
  (∀ n, sequence_a lambda n > 0) ∧
  (lambda = 1 / sequence_a lambda 1 →
    (∀ n, sequence_a lambda (n + 1) / sequence_a lambda n = (1 + Real.sqrt 5) / 2) ∧
    (∀ n, sequence_a lambda n = (1/2) * ((1 + Real.sqrt 5) / 2)^(n - 1))) ∧
  (lambda = 1 / 2016 →
    (∃ n, sequence_a lambda n > 1) ∧
    (∀ m, m < 2018 → sequence_a lambda m ≤ 1) ∧
    (sequence_a lambda 2018 > 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l956_95693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_on_line_l956_95615

noncomputable def terminal_side (α : Real) : Set (Real × Real) :=
  {(x, y) | x = Real.cos α ∧ y = Real.sin α}

theorem tan_alpha_on_line (α : Real) : 
  (∀ x y : Real, x + y = 0 → (x, y) ∈ terminal_side α) → 
  Real.tan α = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_on_line_l956_95615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_right_l956_95628

variable (a b c : ℝ)

-- Definition of the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*(a+b)*x + c^2 + 2*a*b

-- Theorem statement
theorem triangle_is_right 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)  -- Positive side lengths
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)  -- Triangle inequality
  (h7 : ∃ x, f a b c x = 0 ∧ (∀ y, f a b c y ≥ f a b c x)) :  -- Vertex on x-axis
  c^2 = a^2 + b^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_right_l956_95628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_proportional_to_distance_l956_95670

/-- Two intersecting circles -/
structure IntersectingCircles where
  circle1 : Set (ℝ × ℝ)
  circle2 : Set (ℝ × ℝ)
  intersect : (circle1 ∩ circle2).Nonempty

/-- A line passing through the intersection points of two circles -/
noncomputable def IntersectionLine (ic : IntersectingCircles) : Set (ℝ × ℝ) :=
  sorry

/-- Distance from a point to a line -/
noncomputable def distanceToLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Length of a tangent segment from a point to a circle -/
noncomputable def tangentLength (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: The square of the tangent length is proportional to the distance from the point to the intersection line -/
theorem tangent_proportional_to_distance (ic : IntersectingCircles) :
  ∃ k : ℝ, ∀ p ∈ ic.circle1,
    (tangentLength p ic.circle2)^2 = k * distanceToLine p (IntersectionLine ic) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_proportional_to_distance_l956_95670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_permutations_l956_95675

/-- A permutation of the numbers 1 to 6 satisfying the given conditions -/
def ValidPermutation : Type :=
  { p : Fin 6 → Fin 6 // Function.Bijective p ∧
    p 0 ≠ 1 ∧ p 2 ≠ 3 ∧ p 4 ≠ 5 ∧ p 0 < p 2 ∧ p 2 < p 4 }

instance : Fintype ValidPermutation :=
  sorry

/-- The number of valid permutations is 30 -/
theorem count_valid_permutations :
  Fintype.card ValidPermutation = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_permutations_l956_95675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stretch_transformation_properties_l956_95658

noncomputable section

-- Define the stretch transformation
def φ (x y : ℝ) : ℝ × ℝ := (2 * x, 3 * y)

-- Define the resulting function after transformation
noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin (x + Real.pi / 6)

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem stretch_transformation_properties :
  -- 1. The smallest positive period of f is π
  (∀ x, f (x + Real.pi) = f x) ∧ (∀ p, 0 < p → p < Real.pi → ∃ x, f (x + p) ≠ f x) ∧
  -- 2. f achieves its maximum when x = π/6 + kπ, k ∈ ℤ
  (∀ x, f x ≤ 1 ∧ (∃ k : ℤ, x = Real.pi/6 + k * Real.pi → f x = 1)) ∧
  -- 3. f achieves its minimum when x = -π/3 + kπ, k ∈ ℤ
  (∀ x, -1 ≤ f x ∧ (∃ k : ℤ, x = -Real.pi/3 + k * Real.pi → f x = -1)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stretch_transformation_properties_l956_95658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_area_correct_l956_95672

/-- The area of the region outside a circle of radius 2 that is internally tangent to two circles of radius 3, where the points of tangency form a diameter of the smaller circle. -/
noncomputable def tangent_circles_area : ℝ :=
  let r₁ : ℝ := 2  -- radius of smaller circle
  let r₂ : ℝ := 3  -- radius of larger circles
  let area_sector : ℝ := Real.pi * r₂^2 / 3  -- area of 60° sector of larger circle
  let angle : ℝ := Real.arccos (1/3)  -- angle between radii at tangent point
  let area_triangle : ℝ := r₁ * Real.sqrt 5 / 2  -- area of right triangle formed by radii
  3 * Real.pi - 3 * angle * Real.pi - 2 * Real.sqrt 5

/-- Theorem stating that the calculated area is correct. -/
theorem tangent_circles_area_correct : 
  tangent_circles_area = 3 * Real.pi - 3 * Real.arccos (1/3) * Real.pi - 2 * Real.sqrt 5 :=
by
  -- The proof goes here
  sorry

-- We can't use #eval for noncomputable definitions, so we'll use #check instead
#check tangent_circles_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_area_correct_l956_95672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_minimum_value_l956_95612

theorem vector_minimum_value (a b : ℝ × ℝ) : 
  (‖a‖ = 2) →
  (‖b‖ = Real.sqrt 3) →
  (∀ x : ℝ, ‖x • a + (1 - 2*x) • b‖ ≥ Real.sqrt 3 / 2) →
  (∃ x : ℝ, ‖x • a + (1 - 2*x) • b‖ = Real.sqrt 3 / 2) →
  (∃ y : ℝ, ‖a + y • b‖ = 1 ∨ ‖a + y • b‖ = 2) ∧
  (∀ y : ℝ, ‖a + y • b‖ ≥ 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_minimum_value_l956_95612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_score_average_l956_95698

/-- The average score of Class A -/
noncomputable def class_a_average : ℚ := 84

/-- The average score of Class B -/
noncomputable def class_b_average : ℚ := 70

/-- The ratio of students in Class A to Class B -/
noncomputable def class_a_to_b_ratio : ℚ := 3/4

/-- The average score of all students in both classes -/
noncomputable def total_average : ℚ := 76

theorem test_score_average : 
  (class_a_average * class_a_to_b_ratio + class_b_average) / (1 + class_a_to_b_ratio) = total_average := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_score_average_l956_95698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l956_95610

theorem problem_solution (a b c : ℝ) : 
  (Real.sqrt a = 3) → ((b + 1) ^ (1/3 : ℝ) = 2) → (Real.sqrt c = 0) →
  (a = 9 ∧ b = 7 ∧ c = 0) ∧ 
  (Real.sqrt (a * b - c + 1) = 8 ∨ Real.sqrt (a * b - c + 1) = -8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l956_95610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_point_d_l956_95642

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a vector in 2D space -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Calculate the vector between two points -/
def vectorBetween (p1 p2 : Point) : Vec2D :=
  { x := p2.x - p1.x, y := p2.y - p1.y }

/-- Calculate the length of a vector -/
noncomputable def vectorLength (v : Vec2D) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2)

/-- Check if two vectors are parallel -/
def areParallel (v1 v2 : Vec2D) : Prop :=
  v1.x * v2.y = v1.y * v2.x

/-- Definition of an isosceles trapezoid -/
def isIsoscelesTrapezoid (a b c d : Point) : Prop :=
  let ab := vectorBetween a b
  let cd := vectorBetween c d
  let ac := vectorBetween a c
  let bd := vectorBetween b d
  areParallel ab cd ∧ vectorLength ac = vectorLength bd

/-- The main theorem -/
theorem isosceles_trapezoid_point_d :
  let a : Point := { x := 2, y := 1 }
  let b : Point := { x := 3, y := -1 }
  let c : Point := { x := -4, y := 0 }
  let d : Point := { x := -1.4, y := -5.2 }
  let k : ℝ := vectorLength (vectorBetween a b) / vectorLength (vectorBetween c d)
  isIsoscelesTrapezoid a b c d ∧ 
  vectorLength (vectorBetween a b) = k * vectorLength (vectorBetween c d) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_point_d_l956_95642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_period_l956_95617

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 3)

def is_periodic (h : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, h (x + p) = h x

def smallest_positive_period (h : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic h p ∧ p > 0 ∧ ∀ q, 0 < q ∧ q < p → ¬ is_periodic h q

theorem g_period :
  smallest_positive_period g Real.pi := by
  sorry

#check g_period

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_period_l956_95617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_triangle_area_l956_95604

-- Define the hyperbola
def Hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the foci
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 5, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 5, 0)

-- Define the real axis length
def realAxisLength : ℝ := 4

-- Theorem for the standard equation of the hyperbola
theorem hyperbola_equation :
  ∀ x y : ℝ, Hyperbola x y ↔ x^2 / 4 - y^2 = 1 := by
  sorry

-- Theorem for the area of triangle PF₁F₂
theorem triangle_area (P : ℝ × ℝ) :
  Hyperbola P.1 P.2 →
  (P.1 - F₁.1) * (P.2 - F₁.2) + (P.1 - F₂.1) * (P.2 - F₂.2) = 0 →
  (1/2) * Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * 
         Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) * 
         Real.sin (Real.arccos ((P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2)) / 
                   (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * 
                    Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_triangle_area_l956_95604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_OA_OB_k_value_for_area_sqrt10_l956_95601

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = -x
def line (k x y : ℝ) : Prop := y = k * (x + 1)

-- Define the intersection points
def intersection_points (k : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    parabola x1 y1 ∧ line k x1 y1 ∧
    parabola x2 y2 ∧ line k x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2)

-- Statement 1: OA ⊥ OB for any k ≠ 0
theorem perpendicular_OA_OB (k : ℝ) (h : k ≠ 0) :
  intersection_points k → 
  ∃ (x1 y1 x2 y2 : ℝ), x1 * x2 + y1 * y2 = 0 :=
by sorry

-- Define the area of triangle OAB
noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (1/2) * Real.sqrt ((y1 + y2)^2 - 4*y1*y2)

-- Statement 2: k = ±1/6 when area of triangle OAB is √10
theorem k_value_for_area_sqrt10 :
  ∃ (k : ℝ), (∀ (x1 y1 x2 y2 : ℝ), 
    intersection_points k → 
    triangle_area x1 y1 x2 y2 = Real.sqrt 10) → 
    (k = 1/6 ∨ k = -1/6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_OA_OB_k_value_for_area_sqrt10_l956_95601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_to_product_l956_95631

theorem cos_sum_to_product (x : ℝ) : 
  Real.cos (2 * x) + Real.cos (4 * x) + Real.cos (8 * x) + Real.cos (10 * x) = 
  4 * Real.cos (6 * x) * Real.cos (3 * x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_to_product_l956_95631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l956_95647

/-- The function f(t) = (2^t - 4t^2)t / 8^t -/
noncomputable def f (t : ℝ) : ℝ := (2^t - 4*t^2)*t / 8^t

/-- The maximum value of f(t) is √3/9 -/
theorem f_max_value : ∃ (M : ℝ), M = Real.sqrt 3 / 9 ∧ ∀ (t : ℝ), f t ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l956_95647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_sum_base_nine_l956_95629

/-- Represents a digit in base 9 -/
def BaseNineDigit : Type := Fin 9

/-- Represents a number in base 9 -/
def BaseNineNumber : Type := List BaseNineDigit

/-- Get the units digit of a base 9 number -/
def unitsDigit (n : BaseNineNumber) : BaseNineDigit :=
  match n with
  | [] => ⟨0, by norm_num⟩
  | d :: _ => d

/-- Add two base 9 numbers -/
def addBaseNine (a b : BaseNineNumber) : BaseNineNumber :=
  sorry

/-- Convert a natural number to its base 9 representation -/
def toBaseNine (n : ℕ) : BaseNineNumber :=
  sorry

theorem units_digit_sum_base_nine :
  let a : BaseNineNumber := toBaseNine 85
  let b : BaseNineNumber := toBaseNine 37
  unitsDigit (addBaseNine a b) = ⟨3, by norm_num⟩ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_sum_base_nine_l956_95629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_region_T_l956_95603

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The configuration of circles in the problem -/
def CircleConfiguration : Type :=
  {c : List Circle // c.length = 5 ∧ (c.map Circle.radius).toFinset = {2, 4, 6, 8, 10}}

/-- A point is inside exactly one circle of the configuration -/
def InsideExactlyOne (p : ℝ × ℝ) (config : CircleConfiguration) : Prop :=
  ((config.val.filter (fun c => (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2)).length = 1)

/-- The region T -/
def RegionT (config : CircleConfiguration) : Set (ℝ × ℝ) :=
  {p | InsideExactlyOne p config}

/-- The area of a set in ℝ² -/
noncomputable def Area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating the maximum area of region T -/
theorem max_area_region_T :
  ∃ (config : CircleConfiguration),
    ∀ (other_config : CircleConfiguration),
      Area (RegionT config) ≥ Area (RegionT other_config) ∧
      Area (RegionT config) = 220 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_region_T_l956_95603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_without_two_l956_95608

def S : Finset ℕ := {1, 2, 3, 4, 5}

theorem subsets_without_two (S : Finset ℕ) : 
  (S.powerset.filter (fun A => 2 ∉ A)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_without_two_l956_95608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_perpendicular_lines_min_ab_l956_95605

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := x + a^2 * y + 1 = 0
def l₂ (a b : ℝ) (x y : ℝ) : Prop := (a^2 + 1) * x - b * y + 3 = 0

-- Part 1
theorem parallel_lines_a_value (a : ℝ) :
  (∃ x y : ℝ, l₁ a x y ∧ l₂ a (-12) x y) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ → l₂ a (-12) x₂ y₂ → 
    (x₁ - x₂) * (-12) = (y₁ - y₂) * (a^2 + 1)) →
  a = Real.sqrt 3 ∨ a = -Real.sqrt 3 :=
by sorry

-- Part 2
theorem perpendicular_lines_min_ab (a b : ℝ) :
  (∃ x y : ℝ, l₁ a x y ∧ l₂ a b x y) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ → l₂ a b x₂ y₂ → 
    (x₁ - x₂) * (x₁ - x₂) + (y₁ - y₂) * (y₁ - y₂) ≠ 0 →
    ((x₁ - x₂) * 1 + (y₁ - y₂) * a^2) * ((x₁ - x₂) * (a^2 + 1) + (y₁ - y₂) * (-b)) = 0) →
  (∀ a₀ b₀ : ℝ, |a₀ * b₀| ≥ 2) →
  ∃ a₀ b₀ : ℝ, |a₀ * b₀| = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_perpendicular_lines_min_ab_l956_95605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_heights_sum_l956_95687

/-- A sequence of student heights forming an arithmetic progression -/
def StudentHeights (a d : ℝ) (i : ℕ) : ℝ := a + d * (i - 1)

/-- The sum of the first k terms of an arithmetic sequence -/
noncomputable def ArithmeticSum (a d : ℝ) (k : ℕ) : ℝ :=
  k * (2 * a + (k - 1) * d) / 2

theorem student_heights_sum 
  (a d : ℝ) 
  (h_sum10 : ArithmeticSum a d 10 = 12.5)
  (h_sum20 : ArithmeticSum a d 20 = 26.5) :
  ArithmeticSum a d 30 = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_heights_sum_l956_95687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_calculation_l956_95662

noncomputable def closest_value (x : ℝ) (values : List ℝ) : ℝ :=
  match values.argmin (fun v => |x - v|) with
  | some v => v
  | none => 0  -- Default value if the list is empty

theorem closest_to_calculation : 
  closest_value ((0.00056 * 5210362) / 2) [1500, 1600, 1700, 1800] = 1600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_calculation_l956_95662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_increase_rate_l956_95683

/-- Given a population increase of 220 persons in 55 minutes, 
    prove that the rate of population increase in seconds 
    is approximately 0.0667 persons per second. -/
theorem population_increase_rate 
  (population_increase : ℕ) 
  (time_minutes : ℕ) 
  (h1 : population_increase = 220)
  (h2 : time_minutes = 55) :
  ∃ (rate_per_second : ℚ), 
    rate_per_second = population_increase / (time_minutes * 60) ∧ 
    abs (rate_per_second - 0.0667) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_increase_rate_l956_95683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_markup_percentage_l956_95686

/-- Calculates the optimal marked price as a percentage of the list price for a merchant -/
theorem optimal_markup_percentage 
  (list_price : ℝ) 
  (purchase_discount : ℝ) 
  (sale_discount : ℝ) 
  (profit_margin : ℝ) 
  (h1 : list_price > 0)
  (h2 : purchase_discount = 0.25)
  (h3 : sale_discount = 0.25)
  (h4 : profit_margin = 0.30) :
  let purchase_price := list_price * (1 - purchase_discount)
  let markup_factor := (purchase_price / (1 - sale_discount - profit_margin)) / list_price
  ∃ ε > 0, |markup_factor - 1.4286| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_markup_percentage_l956_95686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_2x_l956_95600

-- Define the function to be integrated
noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x)

-- State the theorem
theorem integral_exp_2x : 
  ∫ x in (0)..(1/2), f x = (1/2) * (Real.exp 1 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_2x_l956_95600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l956_95606

theorem triangle_property (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 + c^2 = 4 * a * c →
  S = (Real.sqrt 3 / 2) * a * c * Real.cos B →
  Real.sin A * Real.sin C = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l956_95606
