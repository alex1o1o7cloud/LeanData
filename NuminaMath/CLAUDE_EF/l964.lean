import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_l964_96450

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := Real.sinh t ^ 2
noncomputable def y (t : ℝ) : ℝ := 1 / (Real.cosh t ^ 2)

-- State the theorem
theorem second_derivative_parametric (t : ℝ) :
  let x' := deriv x t
  let y' := deriv y t
  let y'_x := y' / x'
  let y''_xx := deriv (fun t => y' / x') t / x'
  y''_xx = 2 / (Real.cosh t ^ 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_l964_96450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_c_values_l964_96470

-- Define the line before translation
def original_line (x y c : ℝ) : Prop := 2 * x - y + c = 0

-- Define the translation vector
def translation_vector : ℝ × ℝ := (1, -1)

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the condition of tangency
def is_tangent (c : ℝ) : Prop :=
  ∃ x y : ℝ, 
    2 * (x - translation_vector.1) - (y + translation_vector.2) + c = 0 ∧
    circle_equation x y

-- The main theorem
theorem tangent_line_c_values :
  ∀ c : ℝ, is_tangent c → (c = -2 ∨ c = 8) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_c_values_l964_96470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_intersect_T_equals_S_l964_96425

def U : Set ℕ := Set.univ

def S : Set ℕ := {x : ℕ | x^2 - x = 0}

def T : Set ℕ := {x : ℕ | ∃ (k : ℤ), 6 = k * (x - 2)}

theorem S_intersect_T_equals_S : S ∩ T = S := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_intersect_T_equals_S_l964_96425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_symmetry_l964_96490

noncomputable def g (x φ : ℝ) : ℝ := Real.sin (2 * x - 2 * φ + Real.pi / 3)

theorem min_phi_for_symmetry :
  ∃ (φ : ℝ), φ > 0 ∧
  (∀ (x : ℝ), g x φ = g (-x) φ) ∧
  (∀ (ψ : ℝ), ψ > 0 ∧ (∀ (x : ℝ), g x ψ = g (-x) ψ) → φ ≤ ψ) ∧
  φ = 5 * Real.pi / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_symmetry_l964_96490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_codomain_l964_96478

-- Define the function as noncomputable due to its dependency on Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x^2)

-- State the theorem
theorem f_codomain : ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 := by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_codomain_l964_96478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jessie_scott_awards_ratio_l964_96419

theorem jessie_scott_awards_ratio (scott_awards best_athlete_awards jessie_ratio : ℕ) :
  (scott_awards * jessie_ratio * 2 = best_athlete_awards) →
  (scott_awards = 4) →
  (best_athlete_awards = 24) →
  jessie_ratio = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jessie_scott_awards_ratio_l964_96419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_quadrant_l964_96442

-- Define a function to represent the quadrant of an angle
noncomputable def quadrant (θ : ℝ) : ℕ :=
  if 0 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi / 2 then 1
  else if Real.pi / 2 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi then 2
  else if Real.pi < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < 3 * Real.pi / 2 then 3
  else 4

-- Theorem statement
theorem angle_bisector_quadrant (α : ℝ) :
  quadrant α = 4 → quadrant (α / 2) = 2 ∨ quadrant (α / 2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_quadrant_l964_96442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_through_cube_hole_l964_96468

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with unit side length -/
structure UnitCube where
  vertices : Finset Point3D
  edge_length : ℝ
  is_unit : edge_length = 1

/-- Represents a spatial hexagon within a cube -/
structure SpatialHexagon where
  vertices : Finset Point3D
  in_cube : UnitCube

/-- Predicate to check if a cube can pass through a spatial hexagon -/
def can_pass_through (h : SpatialHexagon) (c : UnitCube) : Prop := sorry

/-- Theorem: There exists a spatial hexagon within a unit cube that allows another unit cube to pass through it -/
theorem cube_through_cube_hole (c : UnitCube) : 
  ∃ (h : SpatialHexagon), ∃ (passing_cube : UnitCube), 
    h.in_cube = c ∧ can_pass_through h passing_cube := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_through_cube_hole_l964_96468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_heads_probability_l964_96429

/-- The probability of getting heads on a single coin toss -/
noncomputable def prob_heads : ℚ := 1/2

/-- The number of coins Keiko tosses -/
def keiko_coins : ℕ := 1

/-- The number of coins Ephraim tosses -/
def ephraim_coins : ℕ := 3

/-- The probability that Ephraim gets the same number of heads as Keiko -/
noncomputable def prob_same_heads : ℚ := 1/4

theorem same_heads_probability : 
  prob_same_heads = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_heads_probability_l964_96429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_minimum_l964_96497

noncomputable def M (a b : ℝ) : ℝ := max (3 * a^2 + 2 * b) (3 * b^2 + 2 * a)

theorem M_minimum :
  ∀ a b : ℝ, M a b ≥ -1/3 ∧
  (M a b = -1/3 ↔ a = -1/3 ∧ b = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_minimum_l964_96497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_equation_l964_96445

/-- Given a hyperbola and a parabola with specific properties, prove that the directrix of the parabola has the equation x = -2. -/
theorem parabola_directrix_equation (a : ℝ) (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ)
  (h_a_pos : a > 0)
  (h_hyperbola : 3 * P.1^2 - P.2^2 = 3 * a^2)
  (h_parabola : P.2^2 = 8 * a * P.1)
  (h_foci : F₁.1 < F₂.1 ∧ 3 * F₁.1^2 - F₁.2^2 = 3 * a^2 ∧ 3 * F₂.1^2 - F₂.2^2 = 3 * a^2)
  (h_distance : dist P F₁ + dist P F₂ = 12)
  : {x : ℝ × ℝ | x.1 = -2} = {x : ℝ × ℝ | x.1 = -2 * a} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_equation_l964_96445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_trip_time_l964_96403

/-- Represents Maria's trip details -/
structure Trip where
  highway_distance : ℚ
  mountain_distance : ℚ
  highway_speed_ratio : ℚ
  mountain_time : ℚ
  break_time : ℚ

/-- Calculates the total time of the trip in minutes -/
noncomputable def total_trip_time (t : Trip) : ℚ :=
  let mountain_speed := t.mountain_distance / t.mountain_time
  let highway_speed := mountain_speed * t.highway_speed_ratio
  let highway_time := t.highway_distance / highway_speed
  t.mountain_time + highway_time + t.break_time

/-- Theorem stating that Maria's trip took 105 minutes -/
theorem maria_trip_time :
  let t : Trip := {
    highway_distance := 100,
    mountain_distance := 20,
    highway_speed_ratio := 4,
    mountain_time := 40,
    break_time := 15
  }
  total_trip_time t = 105 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_trip_time_l964_96403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_masters_possible_eight_or_more_masters_impossible_l964_96414

/-- Represents a chess tournament with the given rules --/
structure ChessTournament where
  num_players : Nat
  master_threshold : Rat
  deriving Repr

/-- Defines the conditions for the specific tournament in the problem --/
def tournament : ChessTournament :=
  { num_players := 12
  , master_threshold := 7/10 }

/-- Represents the number of players who could potentially earn the master title --/
def potential_masters (t : ChessTournament) : Nat := 0  -- Default value, to be implemented

/-- Theorem stating that exactly 7 players can earn the master title --/
theorem seven_masters_possible (t : ChessTournament) :
  t = tournament → potential_masters t = 7 := by sorry

/-- Theorem stating that it's impossible for 8 or more players to earn the master title --/
theorem eight_or_more_masters_impossible (t : ChessTournament) :
  t = tournament → potential_masters t < 8 := by sorry

#check seven_masters_possible
#check eight_or_more_masters_impossible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_masters_possible_eight_or_more_masters_impossible_l964_96414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_effectiveness_of_slabs_l964_96424

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions and cost of a slab -/
structure Slab where
  length : ℝ
  width : ℝ
  cost_per_sqm : ℝ

noncomputable def room_area (longer_part shorter_part : RoomDimensions) : ℝ :=
  longer_part.length * longer_part.width + shorter_part.length * shorter_part.width

noncomputable def slab_area (slab : Slab) : ℝ :=
  slab.length * slab.width

noncomputable def num_slabs_needed (room_area slab_area : ℝ) : ℝ :=
  room_area / slab_area

noncomputable def total_cost (num_slabs cost_per_sqm : ℝ) : ℝ :=
  num_slabs * cost_per_sqm

theorem cost_effectiveness_of_slabs
  (longer_part : RoomDimensions)
  (shorter_part : RoomDimensions)
  (slab_a : Slab)
  (slab_b : Slab)
  (h1 : longer_part.length = 7 ∧ longer_part.width = 4)
  (h2 : shorter_part.length = 5 ∧ shorter_part.width = 3)
  (h3 : slab_a.length = 1 ∧ slab_a.width = 0.5 ∧ slab_a.cost_per_sqm = 900)
  (h4 : slab_b.length = 0.5 ∧ slab_b.width = 0.5 ∧ slab_b.cost_per_sqm = 950) :
  total_cost (num_slabs_needed (room_area longer_part shorter_part) (slab_area slab_a)) slab_a.cost_per_sqm <
  total_cost (num_slabs_needed (room_area longer_part shorter_part) (slab_area slab_b)) slab_b.cost_per_sqm :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_effectiveness_of_slabs_l964_96424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l964_96471

/-- The eccentricity of a hyperbola with asymptotes tangent to a specific circle -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →
  (∃ x y : ℝ, x^2 + y^2 - 6*x + 5 = 0) →
  (∃ m₁ m₂ : ℝ, ∀ x y : ℝ, (y = m₁*x ∨ y = m₂*x) → x^2 + y^2 - 6*x + 5 = 0) →
  let c := Real.sqrt (a^2 + b^2)
  c / a = 3 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l964_96471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_20_value_l964_96479

/-- Sequence c defined recursively -/
def c : ℕ → ℕ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 3
  | (n + 3) => c (n + 2) * c (n + 1)

/-- The 20th term of sequence c equals 3^4181 -/
theorem c_20_value : c 20 = 3^4181 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_20_value_l964_96479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_close_points_exist_l964_96437

-- Define the rectangle
def Rectangle : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem close_points_exist (points : Finset (ℝ × ℝ)) 
  (h_in_rectangle : ∀ p ∈ points, p ∈ Rectangle)
  (h_card : points.card = 6) :
  ∃ p q, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ distance p q ≤ Real.sqrt 5 := by
  sorry

#check close_points_exist

end NUMINAMATH_CALUDE_ERRORFEEDBACK_close_points_exist_l964_96437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l964_96486

theorem cosine_identity (θ : ℝ) :
  Real.cos (π / 6 + θ) = Real.sqrt 3 / 3 →
  Real.cos (5 * π / 6 - θ) = -Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l964_96486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l964_96482

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  shortest_distance_B_to_AC : ℝ
  sin_angle_C : ℝ
  side_AC : ℝ

/-- Helper function to calculate distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem stating the possible side lengths of the triangle -/
theorem triangle_side_lengths (t : Triangle)
  (h1 : t.shortest_distance_B_to_AC = 12)
  (h2 : t.sin_angle_C = Real.sqrt 3 / 2)
  (h3 : t.side_AC = 5) :
  (distance t.B t.C = 12 ∧ distance t.A t.B = Real.sqrt 229) ∨
  (distance t.B t.C = (5 + Real.sqrt 501) / 2 ∧ distance t.A t.B = 12) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l964_96482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l964_96421

-- Define the function
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

-- State the theorem
theorem derivative_of_f :
  deriv f = λ x ↦ 3*x^2 + 6*x + 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l964_96421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_license_plate_count_l964_96467

/-- The number of English letters -/
def num_letters : ℕ := 26

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The total number of positions in the license plate -/
def total_positions : ℕ := 5

/-- The number of letter positions in the license plate -/
def letter_positions : ℕ := 2

/-- The number of digit positions in the license plate -/
def digit_positions : ℕ := 3

/-- Theorem: The number of different license plate numbers -/
theorem license_plate_count : 
  (num_letters.choose letter_positions) * 
  (total_positions.choose letter_positions) * 
  (num_digits ^ digit_positions) = 
  (num_letters.choose letter_positions) * 
  (total_positions.choose letter_positions) * 
  (10 ^ 3) := by
  sorry

#eval (num_letters.choose letter_positions) * 
       (total_positions.choose letter_positions) * 
       (num_digits ^ digit_positions)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_license_plate_count_l964_96467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_to_cone_volume_l964_96410

/-- The volume of a cone formed by folding a sector -/
theorem sector_to_cone_volume (r θ : ℝ) : 
  r = 3 →
  θ = 120 * π / 180 →
  let arc_length := r * θ
  let base_radius := arc_length / (2 * π)
  let height := Real.sqrt (r^2 - base_radius^2)
  (1/3) * π * base_radius^2 * height = (2 * Real.sqrt 2 / 3) * π := by
  sorry

#check sector_to_cone_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_to_cone_volume_l964_96410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_center_l964_96494

/-- Square vertices -/
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (1, -1)
def D : ℝ × ℝ := (-1, -1)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The locus condition -/
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  max (distance P A) (distance P C) = (distance P B + distance P D) / 2

/-- The theorem statement -/
theorem locus_is_center :
  ∀ P : ℝ × ℝ, satisfies_condition P ↔ P = (0, 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_center_l964_96494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_equation_l964_96483

/-- The projection of vector v onto vector u -/
noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_squared := u.1 * u.1 + u.2 * u.2
  (dot_product / norm_squared * u.1, dot_product / norm_squared * u.2)

theorem projection_line_equation :
  ∀ (v : ℝ × ℝ), proj (7, 3) v = (-7, -3) →
  v.2 = -7/3 * v.1 - 58/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_equation_l964_96483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_point_min_sum_distances_l964_96476

/-- Line l: y = 2x + 1 -/
def line_l (x y : ℝ) : Prop := y = 2 * x + 1

/-- Point A -/
def point_A : ℝ × ℝ := (-2, 3)

/-- Point B -/
def point_B : ℝ × ℝ := (1, 6)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem for part (1) -/
theorem equal_distance_point (P : ℝ × ℝ) :
  line_l P.1 P.2 →
  distance P point_A = distance P point_B →
  P = (1, 3) := by sorry

/-- Theorem for part (2) -/
theorem min_sum_distances :
  ∀ P : ℝ × ℝ, line_l P.1 P.2 →
  distance P point_A + distance P point_B ≥ 11 * Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_point_min_sum_distances_l964_96476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_textbooks_in_same_box_probability_l964_96498

def total_textbooks : ℕ := 18
def math_textbooks : ℕ := 4
def box_capacities : List ℕ := [5, 6, 3, 4]

def probability_all_math_in_same_box : ℚ :=
  213227 / 14692368

theorem math_textbooks_in_same_box_probability :
  probability_all_math_in_same_box = 
    (Nat.choose total_textbooks math_textbooks * 
     (Nat.choose (total_textbooks - math_textbooks) (box_capacities.get! 0 - math_textbooks) +
      Nat.choose (total_textbooks - math_textbooks) (box_capacities.get! 1 - math_textbooks) +
      Nat.choose (total_textbooks - math_textbooks) (box_capacities.get! 3 - math_textbooks))) /
    (Nat.choose total_textbooks (box_capacities.get! 0) * 
     Nat.choose (total_textbooks - box_capacities.get! 0) (box_capacities.get! 1) *
     Nat.choose (total_textbooks - box_capacities.get! 0 - box_capacities.get! 1) (box_capacities.get! 2) *
     Nat.choose (total_textbooks - box_capacities.get! 0 - box_capacities.get! 1 - box_capacities.get! 2) (box_capacities.get! 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_textbooks_in_same_box_probability_l964_96498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_line_l964_96492

/-- The parabola function -/
noncomputable def parabola (x : ℝ) : ℝ := x^2 - 4*x + 4

/-- The line function -/
noncomputable def line (x : ℝ) : ℝ := 2*x - 3

/-- The distance function between a point (x, parabola(x)) and the line -/
noncomputable def distance (x : ℝ) : ℝ := |2*x - (parabola x) - 3| / Real.sqrt 5

/-- The theorem stating the minimum distance between the parabola and the line -/
theorem min_distance_parabola_line :
  (∃ (x : ℝ), ∀ (y : ℝ), distance x ≤ distance y) ∧
  (∀ (x : ℝ), distance x ≥ Real.sqrt 10 / 5) ∧
  (∃ (x : ℝ), distance x = Real.sqrt 10 / 5) := by
  sorry

#check min_distance_parabola_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_line_l964_96492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l964_96493

-- Define the rectangular parallelepiped
structure Parallelepiped where
  AB : ℝ
  BC : ℝ
  CG : ℝ

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the theorem
theorem pyramid_volume (p : Parallelepiped) (E G M : Point3D) : 
  p.AB = 4 → 
  p.BC = 1 → 
  p.CG = 2 → 
  E = ⟨0, 0, 0⟩ → 
  G = ⟨0, 1, 2⟩ → 
  M = ⟨0, 1/3, 4/3⟩ → 
  (1/3 : ℝ) * Real.sqrt 18 * (4/3 : ℝ) = (4 * Real.sqrt 2) / 3 :=
by
  sorry

#check pyramid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l964_96493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_usable_pieces_value_l964_96409

/-- The length of the pencil lead in cm -/
noncomputable def lead_length : ℝ := 7

/-- A piece is considered unusable if its length is less than or equal to this value in cm -/
noncomputable def unusable_length : ℝ := 2

/-- The probability that a piece of the broken lead is usable -/
noncomputable def prob_usable : ℝ := (lead_length - unusable_length) / lead_length

/-- The expected number of usable pieces after breaking the lead -/
noncomputable def expected_usable_pieces : ℝ := 2 * prob_usable

theorem expected_usable_pieces_value : expected_usable_pieces = 10 / 7 := by
  sorry

#eval (100 * 10 + 7 : ℕ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_usable_pieces_value_l964_96409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_min_distance_l964_96460

noncomputable section

/-- Ellipse parameters -/
structure EllipseParams where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a > b
  k : b > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the ellipse -/
def is_on_ellipse (p : Point) (e : EllipseParams) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Eccentricity of the ellipse -/
def eccentricity (e : EllipseParams) : ℝ :=
  e.c / e.a

/-- Perpendicularity of two vectors -/
def perpendicular (p1 p2 : Point) : Prop :=
  p1.x * p2.x + p1.y * p2.y = 0

/-- Theorem about the specific ellipse and the minimum distance -/
theorem ellipse_and_min_distance (e : EllipseParams) 
  (h1 : is_on_ellipse ⟨0, Real.sqrt 5⟩ e)
  (h2 : eccentricity e = 2/3) :
  (∃ (A B : Point), 
    A.x = 4 ∧ 
    is_on_ellipse B e ∧ 
    perpendicular A B ∧
    (∀ (A' B' : Point), 
      A'.x = 4 → 
      is_on_ellipse B' e → 
      perpendicular A' B' → 
      (A.x - B.x)^2 + (A.y - B.y)^2 ≤ (A'.x - B'.x)^2 + (A'.y - B'.y)^2)) →
  (e.a = 3 ∧ e.b = Real.sqrt 5) ∧
  (∃ (A B : Point), 
    A.x = 4 ∧ 
    is_on_ellipse B e ∧ 
    perpendicular A B ∧
    (A.x - B.x)^2 + (A.y - B.y)^2 = 21) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_min_distance_l964_96460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l964_96434

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

noncomputable def shifted_function (x : ℝ) : ℝ := original_function (x + Real.pi / 3)

noncomputable def final_function (x : ℝ) : ℝ := shifted_function (x / 2)

theorem function_transformation :
  ∀ x : ℝ, final_function x = Real.sin (x / 2 + Real.pi / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l964_96434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_numbers_are_correct_check_special_numbers_check_non_special_number_l964_96436

/-- A function that reverses a two-digit number -/
def reverse (n : Nat) : Nat :=
  10 * (n % 10) + (n / 10)

/-- The set of two-digit numbers that satisfy the property -/
def specialNumbers : Set Nat :=
  {n | 10 ≤ n ∧ n ≤ 99 ∧ ∃ k : Nat, n + reverse n = k * k}

/-- The theorem stating the correct set of numbers -/
theorem special_numbers_are_correct : 
  specialNumbers = {29, 38, 47, 56, 65, 74, 83, 92} := by
  sorry

-- Remove the #eval statement as it's causing issues
-- #eval specialNumbers

-- Instead, we can use a theorem to check if specific numbers are in the set
theorem check_special_numbers :
  29 ∈ specialNumbers ∧ 38 ∈ specialNumbers ∧ 47 ∈ specialNumbers ∧
  56 ∈ specialNumbers ∧ 65 ∈ specialNumbers ∧ 74 ∈ specialNumbers ∧
  83 ∈ specialNumbers ∧ 92 ∈ specialNumbers := by
  sorry

-- We can also check that a number not in our solution is not in the set
theorem check_non_special_number :
  30 ∉ specialNumbers := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_numbers_are_correct_check_special_numbers_check_non_special_number_l964_96436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l964_96456

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  area : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  4 * Real.sin t.A - t.b * Real.sin t.B = t.c * Real.sin (t.A - t.B) ∧
  t.area = (Real.sqrt 3 * (t.b^2 + t.c^2 - t.a^2)) / 4

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.a = 4 ∧ t.a + t.b + t.c ≤ 12 := by
  sorry

#check triangle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l964_96456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gallons_needed_formula_l964_96439

/-- Represents the fuel efficiency of a car in kilometers per gallon -/
noncomputable def fuel_efficiency : ℝ := 40

/-- Represents the number of gallons needed to travel a given distance -/
noncomputable def gallons_needed (distance : ℝ) : ℝ := distance / fuel_efficiency

/-- Theorem stating that the number of gallons needed is the distance divided by the fuel efficiency -/
theorem gallons_needed_formula (distance : ℝ) : 
  gallons_needed distance = distance / fuel_efficiency := by
  -- Unfold the definition of gallons_needed
  unfold gallons_needed
  -- The equation is now trivially true
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gallons_needed_formula_l964_96439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_approx_l964_96496

/-- The rate of simple interest given principal, amount, and time -/
noncomputable def simple_interest_rate (principal amount : ℝ) (time : ℝ) : ℝ :=
  ((amount - principal) * 100) / (principal * time)

/-- Theorem: The rate of simple interest is approximately 8.93% -/
theorem simple_interest_rate_approx :
  let principal := (12000 : ℝ)
  let amount := (19500 : ℝ)
  let time := (7 : ℝ)
  abs (simple_interest_rate principal amount time - 8.93) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_approx_l964_96496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_length_is_320_l964_96433

/-- Represents a circular track with two runners -/
structure CircularTrack where
  length : ℝ
  runner1_speed : ℝ
  runner2_speed : ℝ

/-- Calculates the distance traveled by runner 2 at first meeting -/
noncomputable def first_meeting_distance (track : CircularTrack) : ℝ :=
  track.length / 2 - 80

/-- Calculates the total distance traveled by runner 1 at second meeting -/
noncomputable def runner1_second_meeting_distance (track : CircularTrack) : ℝ :=
  3 * track.length / 2 - 120

/-- Calculates the total distance traveled by runner 2 at second meeting -/
noncomputable def runner2_second_meeting_distance (track : CircularTrack) : ℝ :=
  track.length / 2 + 40

/-- The main theorem stating the length of the track -/
theorem track_length_is_320 (track : CircularTrack) 
  (h1 : track.runner1_speed > 0)
  (h2 : track.runner2_speed > 0)
  (h3 : 80 / (first_meeting_distance track) = 
        (runner2_second_meeting_distance track - first_meeting_distance track) / 120) :
  track.length = 320 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_length_is_320_l964_96433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_same_area_perimeter_l964_96426

noncomputable def triangle_U_perimeter : ℝ := 24
noncomputable def triangle_U_area : ℝ := 10 * Real.sqrt 6

theorem isosceles_triangles_same_area_perimeter 
  (c d : ℝ) 
  (h_perimeter : 2 * c + d = triangle_U_perimeter)
  (h_area : (1/2) * d * Real.sqrt (c^2 - (d/2)^2) = triangle_U_area)
  : d = 6 ∨ d = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_same_area_perimeter_l964_96426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_P_and_Q_P_subset_Q_iff_a_in_range_l964_96452

-- Define the function
noncomputable def f (x : ℝ) := Real.sqrt (x + 2) * Real.sqrt (5 - x)

-- Define the domain Q
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Define the set P
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a + 1}

-- Theorem 1
theorem intersection_complement_P_and_Q (a : ℝ) (h : a = 3) :
  (Set.univ \ P a) ∩ Q = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

-- Theorem 2
theorem P_subset_Q_iff_a_in_range :
  ∀ a : ℝ, P a ⊆ Q ↔ a ∈ Set.Iic 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_P_and_Q_P_subset_Q_iff_a_in_range_l964_96452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_l964_96462

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_squared := v.1^2 + v.2^2
  (dot_product / norm_squared) • v

theorem find_c : ∃ c : ℝ, projection (4, c) (-3, 2) = (10/13 : ℝ) • (-3, 2) ∧ c = 11 := by
  use 11
  apply And.intro
  · simp [projection]
    norm_num
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_l964_96462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_l964_96447

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^2 + a * Real.log x

noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := (2/3) * x + a / x

theorem tangent_line_values (a b : ℝ) :
  (f_deriv a 2 = 1) ∧ 
  (f a 2 = 2 + b) →
  (a = -2/3) ∧ (b = -(2/3) * (Real.log 2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_l964_96447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_range_of_m_l964_96484

-- Define the propositions
def p (a : ℝ) : Prop := ∃ x : ℝ, a * x^2 + a * x + 1 = 0

def q (a m : ℝ) : Prop := m > 0 ∧ ∀ x : ℝ, x > -1 → x - a + m / (x + 1) ≥ 0

-- Define the set of a values for which p is true
def A : Set ℝ := {a | p a}

-- Define the set of a values for which q is true
def B (m : ℝ) : Set ℝ := {a | q a m}

-- Theorem 1: Range of a for which p is true
theorem range_of_a : A = Set.Iic 0 ∪ Set.Ici 4 := by
  sorry

-- Theorem 2: Range of m given the condition
theorem range_of_m : 
  {m : ℝ | m > 0 ∧ A ⊂ B m ∧ A ≠ B m} = Set.Ioo 0 (1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_range_of_m_l964_96484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_m_range_l964_96431

theorem subset_implies_m_range (A B : Set ℝ) (m : ℝ) : 
  A = {x : ℝ | x ≤ -2} →
  B = {x : ℝ | x < m} →
  B ⊆ A →
  m ∈ Set.Iic (-2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_m_range_l964_96431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_domain_theorem_l964_96422

open Set Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := tan (π/4 - 2*x)

-- Define the domain
def domain : Set ℝ := {x | ∀ k : ℤ, x ≠ k * π/2 + 3*π/8}

-- Theorem statement
theorem tan_domain_theorem : 
  {x : ℝ | ∃ y, f x = y} = domain := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_domain_theorem_l964_96422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_cost_per_kg_l964_96489

/-- Proves that the cost of meat per kilogram is $8 -/
theorem meat_cost_per_kg (cheese_amount : Real) (meat_amount : Real) 
  (cheese_cost_per_kg : Real) (total_cost : Real) 
  (h1 : cheese_amount = 1.5)
  (h2 : meat_amount = 0.5)
  (h3 : cheese_cost_per_kg = 6)
  (h4 : total_cost = 13)
  : (total_cost - cheese_amount * cheese_cost_per_kg) / meat_amount = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_cost_per_kg_l964_96489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l964_96407

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

theorem quadratic_function_properties (a b : ℝ) :
  (∀ x, f a b x ≥ f a b (-1)) ∧ f a b (-1) = 0 →
  (f a b = λ x ↦ x^2 + 2*x + 1) ∧
  (∀ x ≤ -1, ∀ y ≥ x, f a b x ≤ f a b y) ∧
  (∀ x ≥ -1, ∀ y ≥ x, f a b x ≤ f a b y) ∧
  (∀ k, (∀ x ∈ Set.Icc (-3) (-1), f a b x > x + k) → k < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l964_96407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inverse_inequality_l964_96448

-- Define the logarithmic function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the inverse proportion function g(x)
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := k / x

-- Define the set of real numbers greater than 0 and not equal to 1
def A : Set ℝ := {x : ℝ | x > 0 ∧ x ≠ 1}

-- State the theorem
theorem log_inverse_inequality (a : ℝ) (k : ℝ) (h1 : a ∈ A) (h2 : f a 2 = 1/2) (h3 : g k 2 = 1/2) :
  {x : ℝ | g k (f a x) < 2} = Set.Ioo 0 1 ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inverse_inequality_l964_96448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_decomposition_l964_96400

theorem fraction_decomposition : ∃ (a b c x y z : ℕ), 
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  (a * b * c = 385) ∧ 
  (x * b * c + y * a * c + z * a * b = 674 * a * b * c) ∧ 
  (x + y + z = (String.toList (toString a)).foldr (λ d acc => (Char.toNat d - 48) + acc) 0 + 
               (String.toList (toString b)).foldr (λ d acc => (Char.toNat d - 48) + acc) 0 + 
               (String.toList (toString c)).foldr (λ d acc => (Char.toNat d - 48) + acc) 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_decomposition_l964_96400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l964_96477

noncomputable def f (x : ℝ) := Real.exp (x + 1) - Real.log (x + 2)

theorem f_monotonicity :
  let f' := fun x => Real.exp (x + 1) - 1 / (x + 2)
  (∀ x ∈ Set.Ioo (-2 : ℝ) (-1), f' x < 0) ∧
  (∀ x ∈ Set.Ioi (-1 : ℝ), f' x > 0) ∧
  (f' (-1) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l964_96477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_products_l964_96455

theorem max_sum_of_products (f g h j : ℕ) : 
  f ∈ ({3, 4, 5, 6} : Set ℕ) → g ∈ ({3, 4, 5, 6} : Set ℕ) → 
  h ∈ ({3, 4, 5, 6} : Set ℕ) → j ∈ ({3, 4, 5, 6} : Set ℕ) →
  f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j →
  f * g + g * h + h * j + f * j ≤ 81 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_products_l964_96455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_result_l964_96461

noncomputable def dilation (z₀ : ℂ) (k : ℝ) (z : ℂ) : ℂ := z₀ + k • (z - z₀)

theorem dilation_result :
  let z₀ : ℂ := 2 - 3*I
  let k : ℝ := -2
  let z : ℂ := -1 + 2*I
  dilation z₀ k z = 8 - 13*I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_result_l964_96461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_formula_l964_96481

/-- A rectangular parallelepiped with a plane passing through its diagonal -/
structure RectangularParallelepiped where
  /-- The distance from the plane to the diagonal of the base -/
  l : ℝ
  /-- The plane passes through the parallelepiped's diagonal -/
  plane_through_diagonal : True
  /-- The plane forms a 45° angle with one side of the base -/
  angle_45 : True
  /-- The plane forms a 30° angle with another side of the base -/
  angle_30 : True
  /-- The plane is parallel to the diagonal of the base -/
  parallel_to_base_diagonal : True

/-- The surface area of a sphere circumscribed around the rectangular parallelepiped -/
noncomputable def sphere_surface_area (p : RectangularParallelepiped) : ℝ := 16 * Real.pi * p.l^2

/-- Theorem stating that the surface area of the circumscribed sphere is 16πl² -/
theorem sphere_surface_area_formula (p : RectangularParallelepiped) :
  sphere_surface_area p = 16 * Real.pi * p.l^2 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_formula_l964_96481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_equals_20_l964_96499

-- Define the function
noncomputable def f (a b x : ℝ) : ℝ := a + b / x

-- State the theorem
theorem a_plus_b_equals_20 (a b : ℝ) :
  (f a b (-2) = 2) → (f a b (-6) = 6) → (a + b = 20) := by
  intro h1 h2
  -- Proof steps would go here
  sorry

#check a_plus_b_equals_20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_equals_20_l964_96499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_c_range_l964_96430

open Real

noncomputable def f (x : ℝ) : ℝ := 1 / exp x - 1

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f x + a * x

theorem f_inequality_implies_c_range :
  ∀ c : ℝ, (∀ x : ℝ, exp x * f x ≤ c * (x - 1) + 1) → c ∈ Set.Icc (-exp 2) 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_c_range_l964_96430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_and_magnitude_l964_96466

def OA : Fin 3 → ℝ := ![1, 1, -2]
def OB : Fin 3 → ℝ := ![0, 2, 3]

def AB : Fin 3 → ℝ := ![OB 0 - OA 0, OB 1 - OA 1, OB 2 - OA 2]

theorem vector_subtraction_and_magnitude :
  AB = ![-1, 1, 5] ∧ 
  Real.sqrt ((AB 0)^2 + (AB 1)^2 + (AB 2)^2) = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_and_magnitude_l964_96466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l964_96404

-- Define the circle
def circle_eq (x y a : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + a = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 5 = 0

-- Define the midpoint of a line segment
def midpoint_of (x1 y1 x2 y2 xm ym : ℝ) : Prop := xm = (x1 + x2) / 2 ∧ ym = (y1 + y2) / 2

-- Theorem statement
theorem line_equation (a : ℝ) (xa ya xb yb : ℝ) :
  a < 3 →
  circle_eq xa ya a →
  circle_eq xb yb a →
  midpoint_of xa ya xb yb (-2) 3 →
  ∀ x y, line_l x y ↔ (x - xa) * (yb - ya) = (y - ya) * (xb - xa) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l964_96404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_iff_parallel_l964_96475

open EuclideanGeometry

-- Define the points A, B, C, D
variable (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the convex quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the circle with diameter AB
def circle_AB (A B : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the circle with diameter CD
def circle_CD (C D : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define a line being tangent to a circle
def is_tangent_to (line : Set (EuclideanSpace ℝ (Fin 2))) (circle : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

-- Define two lines being parallel
def are_parallel (line1 line2 : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

-- Define a line through two points
def line_through (P Q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- The main theorem
theorem tangent_iff_parallel (h_convex : is_convex_quadrilateral A B C D) 
  (h_CD_tangent : is_tangent_to (line_through C D) (circle_AB A B)) :
  is_tangent_to (line_through A B) (circle_CD C D) ↔ are_parallel (line_through B C) (line_through A D) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_iff_parallel_l964_96475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l964_96491

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 4 / (a n * a (n + 1))

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℚ := n / (n + 1)

theorem arithmetic_sequence_proof :
  (a 1 + a 4 = 10) ∧ (a 5 = 10) →
  (∀ n : ℕ, a n = 2 * n) ∧
  (∀ n : ℕ, n ≠ 0 → Finset.sum (Finset.range n) b = S n) :=
by
  intro h
  constructor
  · intro n
    rfl
  · intro n hn
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l964_96491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_difference_proof_l964_96473

/-- Calculates the average speed given distance and time -/
noncomputable def averageSpeed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- The problem statement -/
theorem speed_difference_proof (distance : ℝ) (time_heavy : ℝ) (time_no : ℝ) 
  (h1 : distance = 200)
  (h2 : time_heavy = 5)
  (h3 : time_no = 4) :
  averageSpeed distance time_no - averageSpeed distance time_heavy = 10 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_difference_proof_l964_96473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_sqrt_6_l964_96415

def A : Fin 3 → ℝ := ![2, 3, 5]
def B : Fin 3 → ℝ := ![3, 1, 4]

theorem distance_AB_is_sqrt_6 :
  Real.sqrt (((A 0) - (B 0))^2 + ((A 1) - (B 1))^2 + ((A 2) - (B 2))^2) = Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_sqrt_6_l964_96415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_classes_is_five_l964_96453

/-- Represents the number of classes in a building block given certain conditions -/
def number_of_classes (whiteboards_per_class : ℕ) (ink_per_whiteboard : ℚ) 
  (ink_cost_per_ml : ℚ) (total_daily_cost : ℚ) : ℕ :=
  ((total_daily_cost / ink_cost_per_ml / ink_per_whiteboard) / whiteboards_per_class).floor.toNat

/-- Theorem stating that the number of classes is 5 given the specific conditions -/
theorem number_of_classes_is_five : 
  number_of_classes 2 20 (1/2) 100 = 5 := by
  sorry

#eval number_of_classes 2 20 (1/2) 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_classes_is_five_l964_96453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_team_total_points_l964_96458

/-- The total points scored by the winning team in a basketball game -/
def winning_team_points : ℕ → ℕ := sorry

/-- The points of the losing team at the end of the first quarter -/
def losing_team_first_quarter : ℕ := 10

/-- The points of the winning team at the end of each quarter -/
def winning_team_quarter_points : ℕ → ℕ := sorry

theorem winning_team_total_points :
  (winning_team_quarter_points 1 = 2 * losing_team_first_quarter) ∧
  (winning_team_quarter_points 2 = winning_team_quarter_points 1 + 10) ∧
  (winning_team_quarter_points 3 = winning_team_quarter_points 2 + 20) ∧
  (winning_team_points 4 = winning_team_quarter_points 3) →
  winning_team_points 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_team_total_points_l964_96458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l964_96480

/-- A function satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x < 0, Differentiable ℝ f ∧ 2 * f x + x * deriv f x > x^2

/-- The main theorem -/
theorem solution_set
  (f : ℝ → ℝ)
  (h : SatisfiesCondition f) :
  {x : ℝ | (x + 2016)^2 * f (x + 2016) - 9 * f (-3) < 0} = Set.Ioo (-2019) (-2016) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l964_96480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_pi_approximation_l964_96408

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- The statement that rounding 3.1415926 to the nearest hundredth equals 3.14 -/
theorem round_pi_approximation :
  roundToHundredth 3.1415926 = 3.14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_pi_approximation_l964_96408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slant_height_approx_l964_96446

/-- The slant height of a cone given its base radius and curved surface area. -/
noncomputable def slant_height (radius : ℝ) (curved_surface_area : ℝ) : ℝ :=
  curved_surface_area / (Real.pi * radius)

/-- Theorem stating that for a cone with given dimensions, the slant height is approximately 21 cm. -/
theorem cone_slant_height_approx :
  let radius : ℝ := 10
  let curved_surface_area : ℝ := 659.7344572538566
  let computed_slant_height := slant_height radius curved_surface_area
  ∃ ε > 0, abs (computed_slant_height - 21) < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slant_height_approx_l964_96446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_positive_x_axis_l964_96417

theorem point_on_positive_x_axis (m : ℝ) : 
  let point : ℝ × ℝ := (m^2 + Real.pi, 0)
  point.1 > 0 ∧ point.2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_positive_x_axis_l964_96417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_R_l964_96416

-- Define the region R in three-dimensional space
def R : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   (|x| + |y| + 2*|z| ≤ 2) ∧ (|x| + |y| + 2*|z-1| ≤ 2)}

-- State the theorem about the volume of R
theorem volume_of_R : MeasureTheory.volume R = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_R_l964_96416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l964_96423

theorem solve_exponential_equation (x : ℝ) : (8 : ℝ)^x^3 * (8 : ℝ)^x^3 = (64 : ℝ)^6 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l964_96423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_a_l964_96405

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a) / (x + 1)

/-- The derivative of f(x) -/
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 
  (2*x*(x+1) - (x^2 + a)) / (x + 1)^2

theorem tangent_slope_implies_a (a : ℝ) : 
  f_prime a 1 = -1 → a = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_a_l964_96405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_ride_time_l964_96427

-- Define the variables
noncomputable def walking_time_stationary : ℝ := 80
noncomputable def walking_time_moving : ℝ := 30
noncomputable def delay_time : ℝ := 5

-- Define Clea's walking speed and the escalator's length
noncomputable def walking_speed : ℝ := walking_time_stationary⁻¹

-- Define the escalator's speed
noncomputable def escalator_speed : ℝ := walking_speed * (walking_time_stationary / walking_time_moving - 1)

-- Define the time taken to ride the escalator without walking
noncomputable def riding_time : ℝ := walking_time_stationary * escalator_speed⁻¹

-- Theorem statement
theorem escalator_ride_time :
  riding_time + delay_time = 53 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_ride_time_l964_96427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_eccentricity_l964_96463

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  p : ℝ

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

theorem hyperbola_parabola_eccentricity
  (h : Hyperbola) (p : Parabola) (F P : Point)
  (h_eq : ∀ x y : ℝ, x^2 / h.a^2 - y^2 / h.b^2 = 1 ↔ Point.mk x y ∈ {p : Point | p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1})
  (p_eq : ∀ x y : ℝ, y^2 = 8*x ↔ Point.mk x y ∈ {p : Point | p.y^2 = 8*p.x})
  (common_focus : F ∈ {p : Point | p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1} ∩ {p : Point | p.y^2 = 8*p.x})
  (intersection : P ∈ {p : Point | p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1} ∩ {p : Point | p.y^2 = 8*p.x})
  (dist_PF : distance P F = 5) :
  eccentricity h = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_eccentricity_l964_96463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_existence_l964_96413

/-- A line in a plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Angle between two vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ := sorry

/-- Triangle construction problem -/
theorem triangle_construction_existence
  (e : Line) 
  (P Q : ℝ × ℝ) 
  (a : ℝ) 
  (α : ℝ) 
  (h_a : a > 0) 
  (h_α : 0 < α ∧ α < Real.pi) :
  ∃ (A B C : ℝ × ℝ),
    (∃ t : ℝ, A + t • (P - A) = P) ∧  -- AP is a side
    (∃ s : ℝ, A + s • (Q - A) = Q) ∧  -- AQ is a side
    (∃ r : ℝ, B + r • e.direction = C) ∧  -- BC lies on line e
    ‖B - C‖ = a ∧  -- |BC| = a
    angle (B - A) (C - A) = α  -- ∠BAC = α
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_existence_l964_96413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_g_tetrahedron_l964_96472

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Function g for a tetrahedron ABCD and a point X -/
noncomputable def g (A B C D X : Point3D) : ℝ :=
  distance A X + distance B X + distance C X + distance D X

/-- Theorem stating the minimum value of g for a specific tetrahedron -/
theorem min_g_tetrahedron (A B C D : Point3D) 
    (h1 : distance A D = 24) (h2 : distance B C = 24)
    (h3 : distance A C = 40) (h4 : distance B D = 40)
    (h5 : distance A B = 48) (h6 : distance C D = 48) :
    ∃ (min_val : ℝ), min_val = 4 * Real.sqrt 578 ∧ 
    ∀ (X : Point3D), g A B C D X ≥ min_val := by
  sorry

#check min_g_tetrahedron

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_g_tetrahedron_l964_96472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l964_96432

-- Define the sequence a_n
def a (n : ℕ+) : ℚ := 3 * (-2 : ℚ) ^ (n.val - 1)

-- Define S_n as the sum of the first n terms of a_n
def S (n : ℕ+) : ℚ := (2 / 3) * a n + 1

-- Define T_n as the sum of the first n terms of n|a_n|
def T (n : ℕ+) : ℚ := 3 + 3 * n.val * (2 : ℚ)^n.val - 3 * (2 : ℚ)^n.val

theorem sequence_properties :
  ∀ (n : ℕ+),
    (S n = (2 / 3) * a n + 1) ∧
    (T n = 3 + 3 * n.val * (2 : ℚ)^n.val - 3 * (2 : ℚ)^n.val) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l964_96432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_B_equals_one_l964_96495

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the arithmetic sequence property for logarithms
def isArithmeticSequence (a c b : ℝ) : Prop :=
  ∃ d : ℝ, d < 0 ∧ Real.log c - Real.log a = d ∧ Real.log b - Real.log c = d

-- State the theorem
theorem sin_B_equals_one (t : Triangle) (h : isArithmeticSequence t.a t.c t.b) :
  Real.sin t.B = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_B_equals_one_l964_96495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l964_96420

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin (2 * x) + Real.cos (2 * x)

theorem function_properties (a : ℝ) :
  (f a (π / 3) = (Real.sqrt 3 - 1) / 2) →
  (a = 1 ∧
   (∀ x, f a x ≤ Real.sqrt 2) ∧
   (∀ k : ℤ, ∀ x, k * π + π / 8 ≤ x ∧ x ≤ k * π + 5 * π / 8 →
     ∀ y, x < y → f a y < f a x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l964_96420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_sum_three_wheels_l964_96485

noncomputable def probability_even_sum (p1 p2 p3 : ℝ) : ℝ :=
  let p_odd1 := 1 - p1
  let p_odd2 := 1 - p2
  let p_odd3 := 1 - p3
  p1 * p2 * p3 + 
  p_odd1 * p_odd2 * p3 + 
  p_odd1 * p2 * p_odd3 + 
  p1 * p_odd2 * p_odd3

theorem prob_even_sum_three_wheels :
  probability_even_sum (3/4 : ℝ) (1/2 : ℝ) (1/4 : ℝ) = 1/2 := by
  sorry

-- Remove the #eval statement as it's causing issues with compilation
-- #eval probability_even_sum (3/4 : ℝ) (1/2 : ℝ) (1/4 : ℝ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_sum_three_wheels_l964_96485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_without_rulers_l964_96440

/-- Proves that the number of boys without rulers is 15 in Dr. Lee's math class -/
theorem boys_without_rulers (total_boys : ℕ) (students_with_rulers : ℕ) 
  (girls_with_rulers : ℕ) (non_binary_students : ℕ) (non_binary_with_rulers : ℕ) :
  total_boys = 24 →
  students_with_rulers = 30 →
  girls_with_rulers = 18 →
  non_binary_students = 5 →
  non_binary_with_rulers = 3 →
  total_boys - (students_with_rulers - (girls_with_rulers + non_binary_with_rulers)) = 15 := by
  intro h1 h2 h3 h4 h5
  -- The proof steps would go here, but we'll use sorry to skip the proof
  sorry

#check boys_without_rulers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_without_rulers_l964_96440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bell_rings_for_geography_l964_96488

-- Define the class schedule
inductive ClassType
| Maths
| History
| Geography
| Science
| Music

-- Define the structure of a school day
structure SchoolDay where
  classes : List ClassType
  breakAfterFirst : Nat
  breakAfterSecond : Nat
  breakAfterThird : Nat
  breakAfterFourth : Nat
  canceledClass : Option ClassType

-- Function to count bell rings
def countBellRings (day : SchoolDay) (currentClass : ClassType) : Nat :=
  sorry

-- Theorem statement
theorem bell_rings_for_geography (day : SchoolDay) :
  day.classes = [ClassType.Maths, ClassType.History, ClassType.Geography, ClassType.Science, ClassType.Music] →
  day.breakAfterFirst = 15 →
  day.breakAfterSecond = 10 →
  day.breakAfterThird = 5 →
  day.breakAfterFourth = 20 →
  day.canceledClass = some ClassType.History →
  countBellRings day ClassType.Geography = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bell_rings_for_geography_l964_96488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dotProductRange_l964_96402

/-- The circle C -/
def myCircle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

/-- The ellipse E -/
def myEllipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The dot product of PA and PB -/
def dotProduct (x y : ℝ) : ℝ := x^2 + y^2 - 2*y

theorem dotProductRange :
  ∀ x y : ℝ, myEllipse x y →
  ∃ a b c d : ℝ, myCircle a b ∧ myCircle c d ∧
  (a - c)^2 + (b - d)^2 = 4 →
  -1 ≤ dotProduct x y ∧ dotProduct x y ≤ 13/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dotProductRange_l964_96402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l964_96443

noncomputable def is_solution (x : ℝ) : Prop :=
  (3 - x) ^ (1/3 : ℝ) + Real.sqrt (x - 2) = 1

theorem equation_solutions :
  {x : ℝ | is_solution x} = {2, 3, 11} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l964_96443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_ratio_l964_96454

/-- Triangle ABC with special points and ratios -/
structure SpecialTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  M : ℝ × ℝ
  E : ℝ × ℝ
  H : ℝ × ℝ
  G : ℝ × ℝ
  h : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 15
  ac_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 24
  e_on_ac : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))
  h_on_ab : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ H = (A.1 + s * (B.1 - A.1), A.2 + s * (B.2 - A.2))
  g_intersection : ∃ u v : ℝ, 
    G = (A.1 + u * (M.1 - A.1), A.2 + u * (M.2 - A.2)) ∧
    G = (E.1 + v * (H.1 - E.1), E.2 + v * (H.2 - E.2))
  ae_3ah : Real.sqrt ((A.1 - E.1)^2 + (A.2 - E.2)^2) = 3 * Real.sqrt ((A.1 - H.1)^2 + (A.2 - H.2)^2)

/-- The main theorem -/
theorem special_triangle_ratio (T : SpecialTriangle) : 
  let eg := Real.sqrt ((T.E.1 - T.G.1)^2 + (T.E.2 - T.G.2)^2)
  let gh := Real.sqrt ((T.G.1 - T.H.1)^2 + (T.G.2 - T.H.2)^2)
  eg / gh = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_ratio_l964_96454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_parallel_line_5_units_away_l964_96438

/-- Given a line with equation y = mx + b, this function returns the y-intercept of a parallel line that is d units away from the original line. -/
def parallelLineYIntercept (m : ℝ) (b : ℝ) (d : ℝ) : Set ℝ :=
  {y | ∃ (sign : ℝ), (sign = 1 ∨ sign = -1) ∧ y = b + sign * d * Real.sqrt (m^2 + 1)}

theorem parallel_line_equation (m b d : ℝ) :
  let original_line := fun (x : ℝ) => m * x + b
  let parallel_line := fun (x y : ℝ) => ∃ (c : ℝ), c ∈ parallelLineYIntercept m b d ∧ y = m * x + c
  ∀ (x y : ℝ), parallel_line x y →
    (y = m * x + (b + d * Real.sqrt (m^2 + 1)) ∨
     y = m * x + (b - d * Real.sqrt (m^2 + 1))) :=
by sorry

/-- The main theorem that proves the equation of the parallel line -/
theorem parallel_line_5_units_away :
  let original_line := fun (x : ℝ) => (1/2) * x + 3
  let parallel_line := fun (x y : ℝ) => ∃ (c : ℝ), c ∈ parallelLineYIntercept (1/2) 3 5 ∧ y = (1/2) * x + c
  ∀ (x y : ℝ), parallel_line x y →
    (y = (1/2) * x + (3 + (5 * Real.sqrt 5) / 2) ∨
     y = (1/2) * x + (3 - (5 * Real.sqrt 5) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_parallel_line_5_units_away_l964_96438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bc_length_cos_a_minus_c_l964_96441

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  BC : Real
  AC : Real

-- Define the given triangle
noncomputable def givenTriangle : Triangle where
  A := Real.pi / 3
  AB := 2
  AC := 3
  B := 0 -- Placeholder value, not given
  C := 0 -- Placeholder value, not given
  BC := Real.sqrt 7 -- This is what we need to prove

-- Theorem 1: Prove that BC = √7
theorem bc_length (t : Triangle) (h1 : t = givenTriangle) : t.BC = Real.sqrt 7 := by
  sorry

-- Theorem 2: Prove that cos(A-C) = (5√7)/14
theorem cos_a_minus_c (t : Triangle) (h1 : t = givenTriangle) : 
  Real.cos (t.A - t.C) = (5 * Real.sqrt 7) / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bc_length_cos_a_minus_c_l964_96441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l964_96451

-- Define the parabola
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  a_pos : a > 0
  gcd_one : Int.gcd a (Int.gcd b (Int.gcd c (Int.gcd d (Int.gcd e f)))) = 1

-- Define the properties of the parabola
def passes_through (p : Parabola) (x y : ℤ) : Prop :=
  p.a * x^2 + p.b * x * y + p.c * y^2 + p.d * x + p.e * y + p.f = 0

def focus_x (p : Parabola) (x : ℤ) : Prop :=
  x = 3

def axis_parallel_y (p : Parabola) : Prop :=
  p.b = 0 ∧ p.c = 0

def vertex_on_x_axis (p : Parabola) : Prop :=
  ∃ x : ℤ, p.a * x^2 + p.d * x + p.f = 0

-- Theorem statement
theorem parabola_equation (p : Parabola) 
  (h1 : passes_through p 5 1)
  (h2 : focus_x p 3)
  (h3 : axis_parallel_y p)
  (h4 : vertex_on_x_axis p) :
  p.a = 1 ∧ p.b = 0 ∧ p.c = 0 ∧ p.d = -6 ∧ p.e = -4 ∧ p.f = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l964_96451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_15_l964_96406

theorem remainder_sum_mod_15 (a b c d : ℕ) 
  (ha : a % 15 = 11)
  (hb : b % 15 = 13)
  (hc : c % 15 = 14)
  (hd : d % 15 = 9) : 
  (a + b + c + d) % 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_15_l964_96406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_origin_l964_96487

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem max_distance_to_origin :
  ∃ (max_dist : ℝ), max_dist = 6 ∧
  ∀ (x y : ℝ), circleC x y →
    distance origin (x, y) ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_origin_l964_96487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_20_degrees_l964_96428

/-- The area of a figure formed by rotating a semicircle around one of its ends -/
noncomputable def rotated_semicircle_area (R : ℝ) (α : ℝ) : ℝ := (2 * Real.pi * R^2 * α) / (2 * Real.pi)

/-- Theorem: The area of a figure formed by rotating a semicircle of radius R
    around one of its ends by an angle of 20° is equal to (2πR²)/9 -/
theorem rotated_semicircle_area_20_degrees (R : ℝ) (h : R > 0) :
  rotated_semicircle_area R (20 * Real.pi / 180) = (2 * Real.pi * R^2) / 9 := by
  sorry

#check rotated_semicircle_area_20_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_20_degrees_l964_96428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l964_96469

theorem vector_equation_solution : ∃ (a b : ℚ),
  a = 19/22 ∧ b = 13/22 ∧
  a • (![3, 4] : Fin 2 → ℚ) + b • (![(-1), 6] : Fin 2 → ℚ) = ![2, 7] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l964_96469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_medicine_discount_l964_96401

/-- The discounted price of a medicine given its list price and discount percentage. -/
noncomputable def discountedPrice (listPrice : ℝ) (discountPercentage : ℝ) : ℝ :=
  listPrice * (1 - discountPercentage / 100)

/-- Theorem stating that a 30% discount on a medicine with a list price of 120 results in a final price of 84. -/
theorem medicine_discount : discountedPrice 120 30 = 84 := by
  -- Unfold the definition of discountedPrice
  unfold discountedPrice
  -- Simplify the arithmetic
  simp [mul_sub, mul_div_cancel']
  -- The proof is completed
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_medicine_discount_l964_96401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_between_trig_functions_l964_96418

open Real

theorem relationship_between_trig_functions (x : ℝ) (h : x ∈ Set.Ioo (-1/2) 0) :
  cos ((x + 1) * π) < sin (cos (x * π)) ∧ sin (cos (x * π)) < cos (sin (x * π)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_between_trig_functions_l964_96418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_equiv_l964_96464

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- Define the horizontal shift transformation
def shift (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := λ x ↦ f (x - h)

-- Theorem statement
theorem horizontal_shift_equiv (x : ℝ) :
  (shift g 3) x = g (x - 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_equiv_l964_96464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OMN_l964_96474

-- Define the circle C1
def C1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 20

-- Define the lines C2 and C3 in polar coordinates
def C2 (ρ θ : ℝ) : Prop := θ = Real.pi/3
def C3 (ρ θ : ℝ) : Prop := θ = Real.pi/6

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the intersection points M and N
noncomputable def M : ℝ × ℝ := sorry
noncomputable def N : ℝ × ℝ := sorry

-- State the theorem
theorem area_of_triangle_OMN :
  let areaOMN := (1/2) * ‖M - O‖ * ‖N - O‖ * Real.sin (Real.pi/6)
  areaOMN = 8 + 5 * Real.sqrt 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OMN_l964_96474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_surface_area_volume_l964_96444

theorem prism_surface_area_volume (x : ℝ) :
  x > 0 →
  let a := Real.log x / Real.log 5
  let b := Real.log x / Real.log 6
  let c := Real.log x / Real.log 7
  2 * (a * b + a * c + b * c) = 2 * (a * b * c) →
  x = 210 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_surface_area_volume_l964_96444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_B_completion_time_l964_96459

-- Define the constants
noncomputable def pipe_A_full_time : ℝ := 12
noncomputable def pipe_A_work_time : ℝ := 8
noncomputable def pipe_B_rate_ratio : ℝ := 1/3

-- Define the theorem
theorem pipe_B_completion_time :
  let pipe_A_portion := pipe_A_work_time / pipe_A_full_time
  let remaining_portion := 1 - pipe_A_portion
  let pipe_B_full_time := pipe_A_full_time / pipe_B_rate_ratio
  pipe_B_full_time * remaining_portion = 12 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_B_completion_time_l964_96459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_even_multiple_of_5_l964_96457

/-- The number of toys in the bag -/
def n : ℕ := 50

/-- A toy is even and a multiple of 5 -/
def is_even_multiple_of_5 (x : ℕ) : Prop := x % 2 = 0 ∧ x % 5 = 0

/-- Decidability instance for is_even_multiple_of_5 -/
instance (x : ℕ) : Decidable (is_even_multiple_of_5 x) :=
  show Decidable (x % 2 = 0 ∧ x % 5 = 0) from inferInstance

/-- The set of toys that are even and multiples of 5 -/
def even_multiple_of_5_toys : Finset ℕ := Finset.filter is_even_multiple_of_5 (Finset.range n)

/-- The number of toys that are even and multiples of 5 -/
def k : ℕ := even_multiple_of_5_toys.card

theorem probability_two_even_multiple_of_5 :
  (k : ℚ) * (k - 1) / (n * (n - 1)) = 2 / 245 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_even_multiple_of_5_l964_96457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_imply_a_range_l964_96411

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a * Real.log x

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := x + a / x

theorem perpendicular_tangents_imply_a_range :
  ∀ a : ℝ,
  (∃ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < 2 ∧ 1 < x₂ ∧ x₂ < 2 ∧
    x₁ ≠ x₂ ∧
    f_derivative a x₁ * f_derivative a x₂ = -1) →
  -3 < a ∧ a < -2 :=
by
  sorry

#check perpendicular_tangents_imply_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_imply_a_range_l964_96411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l964_96435

/-- The curve C defined by the product of distances from any point on C to F₁(-1,0) and F₂(1,0) being equal to 4 -/
def curve_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ((p.1 + 1)^2 + p.2^2) * ((p.1 - 1)^2 + p.2^2) = 16}

/-- Fixed points F₁ and F₂ -/
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

/-- The ellipse defined by x²/4 + y²/3 = 1 -/
def ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (a b c : ℝ × ℝ) : ℝ := sorry

theorem curve_C_properties :
  (∀ p : ℝ × ℝ, p ∈ curve_C → (-p.1, -p.2) ∈ curve_C) ∧
  (∀ p : ℝ × ℝ, p ∈ curve_C → area_triangle F₁ p F₂ ≤ 2) ∧
  (∃! p₁ p₂ : ℝ × ℝ, p₁ ∈ curve_C ∧ p₁ ∈ ellipse ∧ p₂ ∈ curve_C ∧ p₂ ∈ ellipse ∧ p₁ ≠ p₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l964_96435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suit_price_calculation_l964_96449

def original_price : ℚ := 200
def increase_rate : ℚ := 1/4
def discount_rate : ℚ := 1/4

theorem suit_price_calculation : 
  (original_price * (1 + increase_rate)) * (1 - discount_rate) = 187.5 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suit_price_calculation_l964_96449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_production_l964_96465

/-- Revenue function -/
noncomputable def R (x : ℝ) : ℝ := 5 * x - (1/2) * x^2

/-- Fixed cost -/
def fixed_cost : ℝ := 0.5

/-- Variable cost per 100 units -/
def variable_cost_per_100 : ℝ := 0.25

/-- Profit function -/
noncomputable def G (x : ℝ) : ℝ := R x - (fixed_cost + variable_cost_per_100 * x)

/-- Annual demand in hundreds of units -/
def annual_demand : ℝ := 5

theorem max_profit_production (x : ℝ) 
  (h1 : 0 ≤ x) (h2 : x ≤ 5) :
  ∃ (x_max : ℝ), G x_max = max (G x) (G x_max) ∧ x_max * 100 = 475 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_production_l964_96465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_eight_percent_l964_96412

/-- Calculates the rate of interest per annum given the principal, time, and simple interest -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (simple_interest : ℝ) : ℝ :=
  (simple_interest * 100) / (principal * time)

/-- Theorem stating that given the specific values, the interest rate is 8% -/
theorem interest_rate_is_eight_percent :
  let principal : ℝ := 2323
  let time : ℝ := 5
  let simple_interest : ℝ := 929.20
  calculate_interest_rate principal time simple_interest = 8 := by
  -- Unfold the definition and simplify
  unfold calculate_interest_rate
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_eight_percent_l964_96412
