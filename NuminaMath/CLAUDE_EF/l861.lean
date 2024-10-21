import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_and_general_term_l861_86190

noncomputable def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

noncomputable def sequence_a (x y : ℝ) : ℕ → ℝ
  | 0 => x
  | 1 => y
  | (n + 2) => (sequence_a x y (n + 1) * sequence_a x y n + 1) / (sequence_a x y (n + 1) + sequence_a x y n)

theorem sequence_convergence_and_general_term (x y : ℝ) :
  (∃ (n₀ : ℕ), ∀ (n : ℕ), n ≥ n₀ → sequence_a x y n = sequence_a x y n₀) ↔ 
  (abs x = 1 ∧ y ≠ -x) ∧
  (∀ (n : ℕ), (sequence_a x y n - 1) / (sequence_a x y n + 1) = 
    ((y - 1) / (y + 1)) ^ (fibonacci (n - 1)) * ((x - 1) / (x + 1)) ^ (fibonacci (n - 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_and_general_term_l861_86190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_rotation_l861_86199

/-- Rotates a list of 2008 digits by a given amount and interprets it as a number. -/
def rotateNumber (α : Fin 2008 → Fin 9) (k : Fin 2008) : ℕ :=
  (List.range 2008).foldl
    (fun acc i => acc * 10 + (α ((i + k) % 2008)).val + 1)
    0

/-- Given a list of 2008 integers between 1 and 9, if rotating the list by some amount
    produces a number divisible by 101, then rotating by any amount produces a number
    divisible by 101. -/
theorem divisibility_rotation (α : Fin 2008 → Fin 9) :
  (∃ k : Fin 2008, 101 ∣ rotateNumber α k) →
  ∀ i : Fin 2008, 101 ∣ rotateNumber α i :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_rotation_l861_86199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gecko_edge_probability_l861_86146

/-- Represents a position on the 4x4 grid -/
structure Position where
  x : Fin 4
  y : Fin 4

/-- Represents a direction of movement -/
inductive Direction where
  | Up
  | Down
  | Left
  | Right

/-- Defines whether a position is on the edge of the grid -/
def isEdge (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Defines whether a position is in the center of the grid -/
def isCenter (p : Position) : Bool :=
  (p.x = 1 || p.x = 2) && (p.y = 1 || p.y = 2)

/-- Defines a valid move based on the given rules -/
def isValidMove (prev : Position) (curr : Position) (lastDir : Option Direction) : Bool :=
  let dx := (curr.x - prev.x + 4) % 4
  let dy := (curr.y - prev.y + 4) % 4
  match (dx, dy, lastDir) with
  | (1, 0, some Direction.Left) => false
  | (3, 0, some Direction.Right) => false
  | (0, 1, some Direction.Down) => false
  | (0, 3, some Direction.Up) => false
  | (1, 0, _) => true
  | (3, 0, _) => true
  | (0, 1, _) => true
  | (0, 3, _) => true
  | _ => false

/-- Placeholder for the probability calculation function -/
noncomputable def probabilityReachEdge (start : Position) (hops : Nat) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem gecko_edge_probability :
  ∃ (p : ℚ), p = 27 / 64 ∧
  (∀ (start : Position),
    isCenter start →
    p = probabilityReachEdge start 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gecko_edge_probability_l861_86146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_increasing_iff_a_gt_one_l861_86147

-- Define the exponential function as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem exponential_increasing_iff_a_gt_one (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a > 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_increasing_iff_a_gt_one_l861_86147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_inscribed_in_parallelogram_l861_86101

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Represents a square -/
structure Square where
  K : Point
  L : Point
  M : Point
  N : Point

/-- The center of a parallelogram -/
noncomputable def center (p : Parallelogram) : Point :=
  { x := (p.A.x + p.C.x) / 2,
    y := (p.A.y + p.C.y) / 2 }

/-- Checks if a point is on a line segment -/
def isOnSegment (P : Point) (A : Point) (B : Point) : Prop :=
  (P.x - A.x) * (B.y - A.y) = (P.y - A.y) * (B.x - A.x) ∧
  min A.x B.x ≤ P.x ∧ P.x ≤ max A.x B.x ∧
  min A.y B.y ≤ P.y ∧ P.y ≤ max A.y B.y

/-- Checks if a square is inscribed in a parallelogram -/
def isInscribed (s : Square) (p : Parallelogram) : Prop :=
  isOnSegment s.K p.A p.B ∧
  isOnSegment s.L p.B p.C ∧
  isOnSegment s.M p.C p.D ∧
  isOnSegment s.N p.D p.A

/-- Theorem: Under certain conditions, it is possible to inscribe a square in a given parallelogram -/
theorem square_inscribed_in_parallelogram (p : Parallelogram) : 
  ∃ (s : Square), isInscribed s p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_inscribed_in_parallelogram_l861_86101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_arccos_arccot_inequality_l861_86142

theorem arcsin_arccos_arccot_inequality (x : ℝ) :
  Real.arcsin x < Real.arccos x ∧ Real.arccos x < Real.arctan (1 / x) →
  0 < x ∧ x < Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_arccos_arccot_inequality_l861_86142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hired_workers_constraint_l861_86171

/-- Represents the wage for a carpenter in yuan -/
def carpenter_wage : ℕ := 50

/-- Represents the wage for a bricklayer in yuan -/
def bricklayer_wage : ℕ := 40

/-- Represents the total wage for workers in yuan -/
def total_wage : ℕ := 2000

/-- Theorem stating the constraint condition for hired workers -/
theorem hired_workers_constraint (x y : ℕ) : 
  carpenter_wage * x + bricklayer_wage * y = total_wage ↔ 5 * x + 4 * y = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hired_workers_constraint_l861_86171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_properties_l861_86187

/-- A regular octagon with side length 1 -/
structure RegularOctagon where
  side_length : ℝ
  side_length_eq_one : side_length = 1

/-- Predicate to check if a given angle is an interior angle of the octagon -/
def is_interior_angle (o : RegularOctagon) (angle : ℝ) : Prop :=
  sorry

/-- Predicate to check if a given length is a diagonal of the octagon -/
def is_diagonal (o : RegularOctagon) (length : ℝ) : Prop :=
  sorry

theorem regular_octagon_properties (o : RegularOctagon) :
  (∀ angle : ℝ, is_interior_angle o angle → angle = 135 * (π / 180)) ∧
  (∃ diagonal : ℝ, is_diagonal o diagonal ∧ diagonal = Real.sqrt (2 + Real.sqrt 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_properties_l861_86187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l861_86191

-- Define the vectors
def AB : Fin 3 → ℝ := ![2, -1, -4]
def AD : Fin 3 → ℝ := ![4, 2, 0]
def AP : Fin 3 → ℝ := ![-1, 2, -1]

-- Define dot product
def dot_product (v1 v2 : Fin 3 → ℝ) : ℝ :=
  (v1 0) * (v2 0) + (v1 1) * (v2 1) + (v1 2) * (v2 2)

-- Define perpendicularity
def is_perpendicular (v1 v2 : Fin 3 → ℝ) : Prop :=
  dot_product v1 v2 = 0

-- Define the theorem
theorem vector_properties :
  is_perpendicular AP AB ∧
  is_perpendicular AP AD ∧
  (is_perpendicular AP AB ∧ is_perpendicular AP AD → AP 0 ≠ 0 ∨ AP 1 ≠ 0 ∨ AP 2 ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l861_86191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_g_l861_86175

/-- Given a linear function f with a zero at x = 2, prove that the zeros of g are 0 and -1/2 --/
theorem zeros_of_g (a b : ℝ) : 
  (∃ f : ℝ → ℝ, f = λ x ↦ a * x + b) → 
  (a * 2 + b = 0) →
  (∃ g : ℝ → ℝ, g = λ x ↦ b * x^2 - a * x) →
  (∃ x : ℝ, b * x^2 - a * x = 0) ∧ 
  (b * 0^2 - a * 0 = 0) ∧ 
  (b * (-1/2)^2 - a * (-1/2) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_g_l861_86175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_champion_is_class_three_l861_86119

-- Define the possible classes
inductive ClassType where
  | Three : ClassType
  | Four : ClassType
  | Five : ClassType
deriving BEq, Repr

-- Define a judge's prediction
structure Prediction where
  notChampion : List ClassType
  isChampion : Option ClassType

-- Define the correctness of a judge's prediction
def isPredictionCorrect (p : Prediction) (champion : ClassType) : Bool :=
  !p.notChampion.contains champion && (p.isChampion.map (· == champion)).getD true

-- Define the judges' predictions
def judgeA : Prediction := ⟨[ClassType.Three, ClassType.Four], none⟩
def judgeB : Prediction := ⟨[ClassType.Three], some ClassType.Five⟩
def judgeC : Prediction := ⟨[ClassType.Five], some ClassType.Three⟩

-- Define the theorem
theorem champion_is_class_three :
  ∃ (champion : ClassType),
    (champion = ClassType.Three) ∧
    (∃! (j : Fin 3), (isPredictionCorrect (match j with
      | 0 => judgeA
      | 1 => judgeB
      | 2 => judgeC) champion) = true) ∧
    (∃! (j : Fin 3), (isPredictionCorrect (match j with
      | 0 => judgeA
      | 1 => judgeB
      | 2 => judgeC) champion) = false) ∧
    (∃! (j : Fin 3), 
      let p := match j with
        | 0 => judgeA
        | 1 => judgeB
        | 2 => judgeC
      (isPredictionCorrect p champion) ≠ (isPredictionCorrect p champion)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_champion_is_class_three_l861_86119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_permutations_count_l861_86118

def Sequence : List Nat := [2, 4, 5, 7, 8]

def IsValidPermutation (perm : List Nat) : Bool :=
  perm.length = 5 &&
  perm.toFinset = Sequence.toFinset &&
  (List.range (perm.length - 2)).all (fun i => 
    !(perm[i]! < perm[i+1]! && perm[i+1]! < perm[i+2]!)) &&
  (List.range (perm.length - 2)).all (fun i => 
    !(perm[i]! > perm[i+1]! && perm[i+1]! > perm[i+2]!)) &&
  (List.range (perm.length - 1)).all (fun i => 
    perm[i]!.mod 2 ≠ perm[i+1]!.mod 2)

theorem valid_permutations_count :
  (Sequence.permutations.filter IsValidPermutation).length = 4 := by
  sorry

#eval Sequence.permutations.filter IsValidPermutation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_permutations_count_l861_86118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vector_length_l861_86121

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ

/-- Point on a plane -/
def Point := ℝ × ℝ

/-- Vector between two points -/
def VectorBetween (A B : Point) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

/-- Length of a vector -/
noncomputable def vectorLength (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

/-- Main theorem -/
theorem parabola_vector_length 
  (C : Parabola) 
  (A B : Point) 
  (hC : C.equation = fun x y ↦ y^2 = 4*x) 
  (hF : C.focus = (1, 0)) 
  (hl : C.directrix = fun x ↦ -1) 
  (hA : A.2 = C.directrix A.1) 
  (hB : C.equation B.1 B.2) 
  (hAFB : VectorBetween C.focus B = (1/3 : ℝ) • VectorBetween C.focus A) : 
  vectorLength (VectorBetween A C.focus) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vector_length_l861_86121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_fourth_element_l861_86179

/-- Systematic sampling function -/
def systematic_sample (total : ℕ) (sample_size : ℕ) (start : ℕ) : List ℕ :=
  List.range sample_size |>.map (λ i => start + i * (total / sample_size))

theorem systematic_sampling_fourth_element
  (total : ℕ)
  (sample_size : ℕ)
  (start : ℕ)
  (h1 : total = 64)
  (h2 : sample_size = 4)
  (h3 : start = 8)
  (h4 : systematic_sample total sample_size start = [8, 24, 56, 40]) :
  (systematic_sample total sample_size start).get! 3 = 40 := by
  rw [h4]
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_fourth_element_l861_86179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_return_probability_l861_86139

structure TruncatedPyramid where
  top_vertex : Fin 1
  larger_base : Fin 6
  smaller_base : Fin 6

noncomputable def ant_walk (pyramid : TruncatedPyramid) : ℝ :=
  (1 : ℝ) / 6 * (1 : ℝ) / 6 * 6

theorem ant_return_probability (pyramid : TruncatedPyramid) :
  ant_walk pyramid = (1 : ℝ) / 6 := by
  sorry

#eval (1 : ℚ) / 6 -- This line is added to check if the fraction can be evaluated

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_return_probability_l861_86139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_ratio_l861_86143

-- Define a cone
structure Cone where
  height : ℝ
  radius : ℝ

-- Define the volume of a cone
noncomputable def volume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

-- Define the volume of water in the cone when filled to 2/3 of its height
noncomputable def water_volume (c : Cone) : ℝ := 
  (1/3) * Real.pi * ((2/3) * c.radius)^2 * ((2/3) * c.height)

-- Theorem statement
theorem water_volume_ratio (c : Cone) : 
  water_volume c = (8/27) * volume c := by
  -- Expand the definitions of water_volume and volume
  unfold water_volume volume
  -- Simplify the expressions
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_ratio_l861_86143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_trip_time_is_five_hours_l861_86197

/-- Represents the trip details between A-ville and B-town -/
structure TripDetails where
  speed_to_b : ℚ  -- Speed from A-ville to B-town in km/h
  speed_from_b : ℚ  -- Speed from B-town to A-ville in km/h
  time_to_b : ℚ  -- Time from A-ville to B-town in minutes

/-- Calculates the total time for a round trip given the trip details -/
def total_trip_time (trip : TripDetails) : ℚ :=
  let time_to_b_hours := trip.time_to_b / 60
  let distance := trip.speed_to_b * time_to_b_hours
  let time_from_b_hours := distance / trip.speed_from_b
  time_to_b_hours + time_from_b_hours

/-- Theorem stating that the total trip time is 5 hours given the specific conditions -/
theorem total_trip_time_is_five_hours (trip : TripDetails) 
    (h1 : trip.speed_to_b = 95)
    (h2 : trip.speed_from_b = 155)
    (h3 : trip.time_to_b = 186) : 
  total_trip_time trip = 5 := by
  sorry

#eval total_trip_time { speed_to_b := 95, speed_from_b := 155, time_to_b := 186 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_trip_time_is_five_hours_l861_86197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_diagonals_l861_86156

/-- A hexagon inscribed in a circle -/
structure InscribedHexagon where
  -- Side lengths
  sideAB : ℝ
  sideBC : ℝ
  sideCD : ℝ
  sideDE : ℝ
  sideEF : ℝ
  sideFA : ℝ
  -- Diagonal lengths
  diagAC : ℝ
  diagAD : ℝ
  diagAE : ℝ
  -- Condition that the hexagon is inscribed in a circle
  inscribed : Prop

/-- The specific hexagon from the problem -/
def problemHexagon : InscribedHexagon where
  sideAB := 26
  sideBC := 73
  sideCD := 73
  sideDE := 73
  sideEF := 58
  sideFA := 58
  diagAC := 89  -- These values are not given in the problem,
  diagAD := 136 -- but we include them for completeness
  diagAE := 127
  inscribed := True

/-- The theorem to be proved -/
theorem sum_of_diagonals (h : InscribedHexagon) 
  (h_sides : h.sideAB = 26 ∧ h.sideBC = 73 ∧ h.sideCD = 73 ∧ h.sideDE = 73 ∧ h.sideEF = 58 ∧ h.sideFA = 58) :
  h.diagAC + h.diagAD + h.diagAE = 352 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_diagonals_l861_86156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l861_86102

def a : ℝ × ℝ × ℝ := (2, 5, -1)
def b : ℝ × ℝ × ℝ := (1, -1, -3)

theorem vector_properties :
  (‖a‖ = Real.sqrt 30) ∧
  (‖b‖ = Real.sqrt 11) ∧
  (a • b = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l861_86102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_triangle_rational_tangent_l861_86133

-- Define a point on the grid
structure GridPoint where
  x : Int
  y : Int

-- Define a triangle on the grid
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

-- Define the tangent of an angle in the triangle
def angle_tangent (t : GridTriangle) (θ : ℚ) : ℚ := sorry

-- Theorem statement
theorem grid_triangle_rational_tangent (t : GridTriangle) :
  ∀ θ : ℚ, ∃ q : ℚ, angle_tangent t θ = q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_triangle_rational_tangent_l861_86133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l861_86173

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem range_of_x (x : ℝ) :
  (∀ t, HasDerivAt f (5 + Real.cos t) t) →  -- f'(x) = 5 + cos x
  f 0 = 0 →                                 -- f(0) = 0
  f (1 - x) + f (1 - x^2) < 0 →             -- Condition
  x < -2 ∨ x > 1 :=                         -- Conclusion
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l861_86173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doll_cost_is_400_l861_86132

/-- The cost of one doll in rubles -/
def doll_cost : ℕ → Prop := sorry

/-- The cost of one robot in rubles -/
def robot_cost : ℕ → Prop := sorry

/-- Four dolls and five robots cost 4100 rubles -/
axiom condition1 : ∃ d r : ℕ, doll_cost d ∧ robot_cost r ∧ 4 * d + 5 * r = 4100

/-- Five dolls and four robots cost 4000 rubles -/
axiom condition2 : ∃ d r : ℕ, doll_cost d ∧ robot_cost r ∧ 5 * d + 4 * r = 4000

/-- The cost of one doll is 400 rubles -/
theorem doll_cost_is_400 : doll_cost 400 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doll_cost_is_400_l861_86132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_area_l861_86189

theorem sector_max_area (perimeter : Real) (h : perimeter = 4) :
  ∃ (α : Real), 
    (∀ r l, r > 0 ∧ l > 0 ∧ 2 * r + l = perimeter →
      r * l / 2 ≤ 1) ∧
    (∃ r l, r > 0 ∧ l > 0 ∧ 2 * r + l = perimeter ∧
      r * l / 2 = 1) ∧
    α = 2 := by
  sorry

#check sector_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_area_l861_86189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_l861_86137

/-- The curve function -/
noncomputable def f (a x : ℝ) : ℝ := a * x^2 - Real.log x

/-- The derivative of the curve function -/
noncomputable def f_derivative (a x : ℝ) : ℝ := 2 * a * x - 1 / x

theorem tangent_parallel_to_x_axis (a : ℝ) :
  f_derivative a 1 = 0 ↔ a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_l861_86137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l861_86162

theorem triangle_angle_proof (A B C : Real) (a b c : Real) :
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  Real.sqrt 2 * a = 2 * b * Real.sin A →
  B = π/4 ∨ B = 3*π/4 := by
  sorry

#check triangle_angle_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l861_86162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_extrema_l861_86160

theorem sin_cos_extrema (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  let μ := Real.sin y + Real.cos x ^ 2
  (∃ (x' y' : ℝ), Real.sin x' + Real.sin y' = 1/3 ∧ μ ≤ 19/12) ∧
  (∃ (x'' y'' : ℝ), Real.sin x'' + Real.sin y'' = 1/3 ∧ μ ≥ -2/3) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), Real.sin x₁ + Real.sin y₁ = 1/3 ∧ Real.sin x₂ + Real.sin y₂ = 1/3 ∧ 
    Real.sin y₁ + Real.cos x₁ ^ 2 = 19/12 ∧ Real.sin y₂ + Real.cos x₂ ^ 2 = -2/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_extrema_l861_86160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bottles_correct_l861_86111

noncomputable def min_bottles (a b : ℕ) : ℕ :=
  Nat.ceil ((a : ℝ) - (a : ℝ) / (b : ℝ))

theorem min_bottles_correct (a b : ℕ) (ha : a > 1) (hb : b > 1) (hab : a > b) :
  min_bottles a b = Nat.ceil ((a : ℝ) - (a : ℝ) / (b : ℝ)) ∧
  ∀ x : ℕ, x < min_bottles a b → (b : ℝ) * (x : ℝ) / ((b : ℝ) - 1) < (a : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bottles_correct_l861_86111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l861_86149

/-- Given a triangle ABC where AB = 2 and AC = √3 * BC, 
    the maximum area of the triangle is √3. -/
theorem triangle_max_area : 
  ∀ (A B C : ℝ × ℝ),
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let area := Real.sqrt (((AB + BC + AC) / 2) * 
               (((AB + BC + AC) / 2) - AB) * 
               (((AB + BC + AC) / 2) - BC) * 
               (((AB + BC + AC) / 2) - AC))
  AB = 2 ∧ AC = Real.sqrt 3 * BC → 
  area ≤ Real.sqrt 3 := by
  sorry

/-- Helper lemma: For any triangle satisfying the conditions, 
    its area is less than or equal to √3. -/
lemma area_bound (A B C : ℝ × ℝ) 
  (hAB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2)
  (hAC : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 
         Real.sqrt 3 * Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)) :
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let area := Real.sqrt (((AB + BC + AC) / 2) * 
               (((AB + BC + AC) / 2) - AB) * 
               (((AB + BC + AC) / 2) - BC) * 
               (((AB + BC + AC) / 2) - AC))
  area ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l861_86149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_cosine_l861_86103

/-- Given vectors a and b with an angle θ between them, 
    if a = (3, -1) and b - a = (-1, 1), then cos θ = 3√10/10 -/
theorem vector_angle_cosine (a b : ℝ × ℝ) (θ : ℝ) :
  a = (3, -1) → b - a = (-1, 1) → Real.cos θ = (3 * Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_cosine_l861_86103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_triangle_l861_86167

theorem longest_side_of_triangle (A B C : ℝ) (a b c : ℝ) : 
  B = 2 * Real.pi / 3 →
  C = Real.pi / 6 →
  a = 5 →
  A + B + C = Real.pi →
  Real.sin A * b = Real.sin B * a →
  Real.sin B * c = Real.sin C * b →
  Real.sin C * a = Real.sin A * c →
  b ≥ a ∧ b ≥ c →
  b = 5 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_triangle_l861_86167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_c_value_l861_86176

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := 3 * Real.cos (5 * x + c) - 1

theorem min_c_value (c : ℝ) :
  (∀ x, f c 0 ≤ f c x) → c ≥ π ∧ ∃ c', c' ≥ π ∧ ∀ x, f c' 0 ≤ f c' x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_c_value_l861_86176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_sixth_plus_alpha_l861_86194

theorem sin_pi_sixth_plus_alpha (α : Real) 
  (h1 : Real.sin (π/3 - α) = 1/3) 
  (h2 : 0 < α) 
  (h3 : α < π/2) : 
  Real.sin (π/6 + α) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_sixth_plus_alpha_l861_86194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_shaded_area_l861_86126

/-- The area of a triangle with base 9 cm and height 12 cm is 54 square centimeters. -/
theorem triangle_area (base height : ℝ) 
  (h1 : base = 9) (h2 : height = 12) :
  (1/2) * base * height = 54 := by
  sorry

/-- The area of a triangle that is part of a larger right triangle 
    with base 16 cm and height 12 cm, where one side of the inner triangle 
    is 12 cm and the other is 9 cm, is 54 square centimeters. -/
theorem shaded_area (large_base large_height inner_side1 inner_side2 : ℝ) 
  (h1 : large_base = 16) (h2 : large_height = 12)
  (h3 : inner_side1 = 12) (h4 : inner_side2 = 9) :
  (1/2) * inner_side1 * inner_side2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_shaded_area_l861_86126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l861_86115

theorem count_integer_pairs : ∃! n : ℕ, 
  n = (Finset.filter (fun p : ℕ × ℕ => 
    let a := p.1
    let b := p.2
    a > 0 ∧ b > 0 ∧ 
    a + b ≤ 150 ∧ 
    (a : ℚ) + (1 : ℚ) / b = 17 * ((1 : ℚ) / a + (b : ℚ))) (Finset.product (Finset.range 151) (Finset.range 151))).card ∧
  n = 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l861_86115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_D_E_l861_86164

-- Define the region D
noncomputable def region_D (x y : ℝ) : Prop :=
  x ≥ 2 ∧ x + y ≤ 0 ∧ x - y - 10 ≤ 0

-- Define the line y = 2x
def line_y_2x (x y : ℝ) : Prop :=
  y = 2 * x

-- Define the reflection of a point across the line y = 2x
noncomputable def reflect_point (x y : ℝ) : ℝ × ℝ :=
  let t := (2 * x + y) / 3
  (t, 2 * t)

-- Define the region E as the reflection of D
noncomputable def region_E (x y : ℝ) : Prop :=
  ∃ (x' y' : ℝ), region_D x' y' ∧ reflect_point x' y' = (x, y)

-- State the theorem
theorem min_distance_D_E :
  ∃ (d : ℝ), d = 12 * Real.sqrt 5 / 5 ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ),
    region_D x₁ y₁ → region_E x₂ y₂ →
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≥ d) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    region_D x₁ y₁ ∧ region_E x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_D_E_l861_86164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l861_86131

noncomputable section

-- Define the function f(x) = ln x
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Define the points A, B, C, and E
noncomputable def A : ℝ × ℝ := (2, f 2)
noncomputable def B : ℝ × ℝ := (32, f 32)

-- Define C as the point that divides AB in a 1:3 ratio
noncomputable def C : ℝ × ℝ := (
  (1/4) * A.fst + (3/4) * B.fst,
  (1/4) * A.snd + (3/4) * B.snd
)

-- Define E as the point where the horizontal line through C intersects f(x)
noncomputable def E : ℝ × ℝ := (Real.exp C.snd, C.snd)

-- Theorem statement
theorem intersection_point :
  E.fst = 16 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l861_86131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l861_86134

/-- Definition of a Triangle -/
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b

/-- Triangle inequality theorem -/
theorem triangle_inequality (a b c : ℝ) (h : Triangle a b c) : 
  a + b > c ∧ b + c > a ∧ a + c > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l861_86134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_any_line_parallel_implies_planes_parallel_perpendicular_lines_parallel_implies_planes_parallel_l861_86108

namespace ParallelPlanes

-- Define the concept of a plane
structure Plane where
  -- We'll leave this empty for now, as we don't need specific fields for this problem

-- Define the concept of a line
structure Line where
  -- We'll leave this empty for now, as we don't need specific fields for this problem

-- Define parallel relation between a line and a plane
def line_parallel_to_plane (l : Line) (p : Plane) : Prop :=
  sorry

-- Define parallel relation between two planes
def planes_parallel (p1 p2 : Plane) : Prop :=
  sorry

-- Define perpendicular relation between a line and a plane
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop :=
  sorry

-- Define parallel relation between two lines
def lines_parallel (l1 l2 : Line) : Prop :=
  sorry

-- Define a membership relation for a line in a plane
def line_in_plane (l : Line) (p : Plane) : Prop :=
  sorry

-- Theorem 1: If any line in plane α is parallel to plane β, then α and β are parallel
theorem any_line_parallel_implies_planes_parallel (α β : Plane) :
  (∀ l : Line, line_in_plane l α → line_parallel_to_plane l β) →
  planes_parallel α β :=
by sorry

-- Theorem 2: If a ⊥ α, b ⊥ β, and a ∥ b, then α and β are parallel
theorem perpendicular_lines_parallel_implies_planes_parallel 
  (α β : Plane) (a b : Line) :
  line_perpendicular_to_plane a α →
  line_perpendicular_to_plane b β →
  lines_parallel a b →
  planes_parallel α β :=
by sorry

end ParallelPlanes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_any_line_parallel_implies_planes_parallel_perpendicular_lines_parallel_implies_planes_parallel_l861_86108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_f_l861_86150

open Real

/-- The function f(x) = 2sin(3x + π/6) -/
noncomputable def f (x : ℝ) : ℝ := 2 * sin (3 * x + π / 6)

/-- The minimum positive period of f(x) is 2π/3 -/
theorem min_positive_period_f : 
  ∃ (T : ℝ), T > 0 ∧ T = 2 * π / 3 ∧ 
  (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, T' < T → ∃ x, f (x + T') ≠ f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_f_l861_86150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_prime_difference_in_sequence_l861_86122

-- Define the sequence
def our_sequence (k : ℕ) : ℕ := 4 + 10 * k

-- Define primality
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem no_prime_difference_in_sequence :
  ∀ k : ℕ, ¬∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p - q = our_sequence k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_prime_difference_in_sequence_l861_86122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_in_set_l861_86174

def is_valid_set (M : Set ℕ) : Prop :=
  Set.Infinite M ∧
  ∀ a b, a ∈ M → b ∈ M → a ≠ b →
    (a^b + 2 ∈ M ∨ a^b - 2 ∈ M)

theorem composite_in_set (M : Set ℕ) (h : is_valid_set M) :
  ∃ n, n ∈ M ∧ ¬ Nat.Prime n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_in_set_l861_86174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_probabilities_l861_86140

/-- Represents the number of intersections in the car's journey -/
def num_intersections : ℕ := 4

/-- Probability of encountering a green light at each intersection -/
def p_green : ℚ := 3/4

/-- Probability of encountering a red light at each intersection -/
def p_red : ℚ := 1/4

/-- ξ represents the number of intersections passed when the car stops -/
def ξ : ℕ → ℚ := sorry

/-- The expectation of ξ -/
def E_ξ : ℚ := 525/256

/-- The probability that at most 3 intersections have been passed when the car stops -/
def P_ξ_le_3 : ℚ := 175/256

/-- Theorem stating the expectation of ξ and the probability of ξ ≤ 3 -/
theorem car_journey_probabilities :
  E_ξ = 525/256 ∧ P_ξ_le_3 = 175/256 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_probabilities_l861_86140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_gcd_lcm_l861_86128

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem count_pairs_gcd_lcm : 
  let n := 50
  let gcd_value := factorial n
  let lcm_value := gcd_value ^ 2
  Finset.card (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ 
    Nat.gcd p.1 p.2 = gcd_value ∧ 
    Nat.lcm p.1 p.2 = lcm_value
  ) (Finset.product (Finset.range (lcm_value + 1)) (Finset.range (lcm_value + 1)))) = 2^15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_gcd_lcm_l861_86128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_pi_plus_alpha_l861_86105

theorem cos_negative_pi_plus_alpha (α : ℝ) :
  (∃ (P : ℝ × ℝ), P = (-3, 4) ∧ P.1 = -3 * Real.cos α ∧ P.2 = 3 * Real.sin α) →
  Real.cos (-Real.pi - α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_pi_plus_alpha_l861_86105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_minimum_triangle_area_line_equation_at_minimum_area_l861_86157

/-- A line parameterized by k -/
noncomputable def line (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0

/-- The area of triangle AOB formed by the line and coordinate axes -/
noncomputable def triangle_area (k : ℝ) : ℝ := 
  (2 + 1 / k) * (2 * k + 1) / 2

theorem line_passes_through_fixed_point :
  ∀ k : ℝ, line k (-2) 1 := by sorry

theorem minimum_triangle_area :
  ∃ min_area : ℝ, min_area = 4 ∧ ∀ k : ℝ, k > 0 → triangle_area k ≥ min_area := by sorry

theorem line_equation_at_minimum_area :
  ∃ k : ℝ, k > 0 ∧ triangle_area k = 4 ∧ (∀ x y, line k x y ↔ x - 2 * y + 4 = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_minimum_triangle_area_line_equation_at_minimum_area_l861_86157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_altitude_largest_integer_altitude_l861_86114

-- Define a scalene triangle
structure ScaleneTriangle where
  -- We don't need to define the vertices explicitly
  altitude1 : ℝ
  altitude2 : ℝ
  altitude3 : ℝ
  is_scalene : altitude1 ≠ altitude2 ∧ altitude2 ≠ altitude3 ∧ altitude3 ≠ altitude1

-- Define our specific triangle
def our_triangle : ScaleneTriangle where
  altitude1 := 6
  altitude2 := 18
  altitude3 := 0  -- We'll prove this can be at most 9
  is_scalene := by sorry  -- We assume this is true based on the problem statement

-- Theorem statement
theorem max_third_altitude (t : ScaleneTriangle) (h1 : t.altitude1 = 6) (h2 : t.altitude2 = 18) 
    (h3 : ∃ (n : ℕ), t.altitude3 = ↑n) : t.altitude3 ≤ 9 := by
  sorry

-- The largest possible integer value is indeed 9
theorem largest_integer_altitude : ∃ (t : ScaleneTriangle), 
    t.altitude1 = 6 ∧ t.altitude2 = 18 ∧ t.altitude3 = 9 ∧ ∃ (n : ℕ), t.altitude3 = ↑n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_altitude_largest_integer_altitude_l861_86114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_squares_l861_86183

theorem cosine_difference_squares : 
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_squares_l861_86183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_4501st_and_4052nd_digits_l861_86112

/-- The sequence where each positive integer n is repeated n times in increasing order -/
def customSequence (n : ℕ) : ℕ :=
  sorry

/-- The sum of the digits at positions 4501 and 4052 in the sequence -/
def sumOfSpecificDigits : ℕ := customSequence 4501 + customSequence 4052

/-- Theorem stating that the sum of the 4501st and 4052nd digits is 9 -/
theorem sum_of_4501st_and_4052nd_digits :
  sumOfSpecificDigits = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_4501st_and_4052nd_digits_l861_86112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_male_difference_l861_86166

/-- Represents the number of people at a family reunion -/
structure FamilyReunion where
  male_adults : ℕ
  female_adults : ℕ
  children : ℕ

/-- The conditions of the family reunion problem -/
def reunion_conditions (r : FamilyReunion) : Prop :=
  r.male_adults = 100 ∧
  r.children = 2 * (r.male_adults + r.female_adults) ∧
  r.male_adults + r.female_adults + r.children = 750

theorem female_male_difference (r : FamilyReunion) 
  (h : reunion_conditions r) : 
  r.female_adults - r.male_adults = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_male_difference_l861_86166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_equation_l861_86158

theorem mod_equation (m : ℕ) (h1 : 0 ≤ m) (h2 : m < 37) (h3 : (7 * m) % 37 = 1) :
  (3^m)^4 % 37 - 3 % 37 = 13 % 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_equation_l861_86158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_black_correct_l861_86172

/-- Represents a 4x4 grid of squares -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Probability of a square being initially black -/
noncomputable def initial_black_prob : ℝ := 1 / 2

/-- Rotates the grid 90 degrees clockwise -/
def rotate (g : Grid) : Grid :=
  λ i j => g (3 - j) i

/-- Applies the repainting rule after rotation -/
def repaint (original : Grid) (rotated : Grid) : Grid :=
  λ i j => rotated i j || original i j

/-- Probability that the entire grid is black after two rotations -/
noncomputable def prob_all_black_after_two_rotations : ℝ :=
  (1 / 2) ^ 16

theorem prob_all_black_correct :
  prob_all_black_after_two_rotations = 1 / 65536 := by
  sorry

#eval "Proof completed."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_black_correct_l861_86172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carpeting_cost_l861_86104

noncomputable def room_length : ℝ := 13
noncomputable def room_width : ℝ := 9
noncomputable def carpet_width_cm : ℝ := 75
noncomputable def rate_per_sqm : ℝ := 12

noncomputable def carpet_width : ℝ := carpet_width_cm / 100

theorem carpeting_cost : 
  room_length * room_width * rate_per_sqm = 1404 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carpeting_cost_l861_86104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_plus_alpha_l861_86163

theorem tan_pi_fourth_plus_alpha (α : Real) (h_acute : 0 < α ∧ α < π / 2) (h_cos : Real.cos α = Real.sqrt 5 / 5) : 
  Real.tan (π / 4 + α) = -3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_plus_alpha_l861_86163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_functions_existence_l861_86152

/-- Definition of a family function -/
def is_family_function (f : ℝ → ℝ) (D₁ D₂ : Set ℝ) : Prop :=
  D₁ ≠ D₂ ∧ f '' D₁ = f '' D₂

/-- The absolute value function shifted by 3 -/
def f (x : ℝ) : ℝ := |x - 3|

/-- The identity function -/
def g (x : ℝ) : ℝ := x

/-- The exponential function with base 2 -/
noncomputable def h (x : ℝ) : ℝ := 2^x

/-- The logarithm function with base 1/2 -/
noncomputable def k (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

theorem family_functions_existence :
  (∃ D₁ D₂ : Set ℝ, is_family_function f D₁ D₂) ∧
  (¬ ∃ D₁ D₂ : Set ℝ, is_family_function g D₁ D₂) ∧
  (¬ ∃ D₁ D₂ : Set ℝ, is_family_function h D₁ D₂) ∧
  (¬ ∃ D₁ D₂ : Set ℝ, is_family_function k D₁ D₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_functions_existence_l861_86152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escher_prints_consecutive_probability_l861_86109

/-- The probability of placing n special items consecutively in a row of m total items -/
def consecutive_probability (m n : ℕ) : ℚ :=
  if n > m then 0
  else Nat.factorial (m - n + 1) * Nat.factorial n / Nat.factorial m

/-- The number of pieces of art -/
def total_pieces : ℕ := 12

/-- The number of Escher prints -/
def escher_prints : ℕ := 4

/-- The probability of all Escher prints being placed consecutively -/
theorem escher_prints_consecutive_probability :
  consecutive_probability total_pieces escher_prints = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escher_prints_consecutive_probability_l861_86109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thirds_pi_in_degrees_pi_in_degrees_l861_86107

-- Define the conversion factor from radians to degrees
noncomputable def rad_to_deg : ℝ := 180 / Real.pi

-- Theorem statement
theorem two_thirds_pi_in_degrees :
  (2 / 3 : ℝ) * Real.pi * rad_to_deg = 120 := by
  -- Proof steps would go here
  sorry

-- Additional helper lemma to demonstrate the relationship
theorem pi_in_degrees :
  Real.pi * rad_to_deg = 180 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thirds_pi_in_degrees_pi_in_degrees_l861_86107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_equals_secant_l861_86148

theorem tan_difference_equals_secant (α β : Real) : 
  0 < α ∧ α < π/2 → 0 < β ∧ β < π/2 → Real.tan α - Real.tan β = 1 / Real.cos β → 2 * α - β = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_equals_secant_l861_86148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_worked_ten_days_l861_86141

-- Define the total work
noncomputable def total_work : ℝ := 1

-- Define the work rates of x and y
noncomputable def x_rate : ℝ := total_work / 18
noncomputable def y_rate : ℝ := total_work / 15

-- Define the time x needed to finish the remaining work
noncomputable def x_remaining_time : ℝ := 6

-- Define the function to calculate y's working days
noncomputable def y_working_days : ℝ → ℝ := λ d => 
  d * y_rate + x_remaining_time * x_rate

theorem y_worked_ten_days : 
  y_working_days 10 = total_work :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_worked_ten_days_l861_86141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l861_86100

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle (renamed to avoid conflict)
def circleEq (x y r : ℝ) : Prop := (x - 5)^2 + y^2 = r^2

-- Define the line
def line (k b x y : ℝ) : Prop := x = k * y + b

-- Define the midpoint condition
def is_midpoint (m_x m_y a_x a_y b_x b_y : ℝ) : Prop :=
  m_x = (a_x + b_x) / 2 ∧ m_y = (a_y + b_y) / 2

-- Define the main theorem
theorem parabola_circle_intersection 
  (l_k l_b : ℝ) -- Parameters of the line
  (a_x a_y b_x b_y : ℝ) -- Coordinates of points A and B
  (m_x m_y : ℝ) -- Coordinates of point M
  (r : ℝ) -- Radius of the circle
  (h_r : r > 0) -- r is positive
  (h_parabola_a : parabola a_x a_y) -- A is on the parabola
  (h_parabola_b : parabola b_x b_y) -- B is on the parabola
  (h_line_a : line l_k l_b a_x a_y) -- A is on the line
  (h_line_b : line l_k l_b b_x b_y) -- B is on the line
  (h_circle : circleEq m_x m_y r) -- M is on the circle
  (h_tangent : ∀ x y, line l_k l_b x y → circleEq x y r → x = m_x ∧ y = m_y) -- l is tangent to circle at M
  (h_midpoint : is_midpoint m_x m_y a_x a_y b_x b_y) -- M is midpoint of AB
  : 
  -- (1) If triangle AOB is equilateral, its side length is 8√3
  (∀ (side : ℝ), a_x^2 + a_y^2 = side^2 ∧ b_x^2 + b_y^2 = side^2 ∧ (a_x - b_x)^2 + (a_y - b_y)^2 = side^2 → side = 8 * Real.sqrt 3) ∧
  -- (2) When r = 4, the only possible equations for line l are x = 1 and x = 9
  (r = 4 → (l_k = 0 ∧ (l_b = 1 ∨ l_b = 9))) ∧
  -- (3) Number of lines for different r values
  (2 < r ∧ r < 4 → ∃ (n : ℕ), n = 4 ∧ ∃ (l1 l2 l3 l4 : ℝ × ℝ), l1 ≠ l2 ∧ l2 ≠ l3 ∧ l3 ≠ l4 ∧ l4 ≠ l1 ∧ 
    (∀ (k b : ℝ), (k, b) = l1 ∨ (k, b) = l2 ∨ (k, b) = l3 ∨ (k, b) = l4 ↔ 
      ∃ (a_x a_y b_x b_y m_x m_y : ℝ), 
        parabola a_x a_y ∧ parabola b_x b_y ∧
        line k b a_x a_y ∧ line k b b_x b_y ∧
        circleEq m_x m_y r ∧
        (∀ x y, line k b x y → circleEq x y r → x = m_x ∧ y = m_y) ∧
        is_midpoint m_x m_y a_x a_y b_x b_y)) ∧
  ((r > 0 ∧ r ≤ 2) ∨ (r ≥ 4 ∧ r < 5) → ∃ (n : ℕ), n = 2 ∧ ∃ (l1 l2 : ℝ × ℝ), l1 ≠ l2 ∧ 
    (∀ (k b : ℝ), (k, b) = l1 ∨ (k, b) = l2 ↔ 
      ∃ (a_x a_y b_x b_y m_x m_y : ℝ), 
        parabola a_x a_y ∧ parabola b_x b_y ∧
        line k b a_x a_y ∧ line k b b_x b_y ∧
        circleEq m_x m_y r ∧
        (∀ x y, line k b x y → circleEq x y r → x = m_x ∧ y = m_y) ∧
        is_midpoint m_x m_y a_x a_y b_x b_y)) ∧
  (r ≥ 5 → ∃ (n : ℕ), n = 1 ∧ ∃ (l1 : ℝ × ℝ), 
    (∀ (k b : ℝ), (k, b) = l1 ↔ 
      ∃ (a_x a_y b_x b_y m_x m_y : ℝ), 
        parabola a_x a_y ∧ parabola b_x b_y ∧
        line k b a_x a_y ∧ line k b b_x b_y ∧
        circleEq m_x m_y r ∧
        (∀ x y, line k b x y → circleEq x y r → x = m_x ∧ y = m_y) ∧
        is_midpoint m_x m_y a_x a_y b_x b_y))
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l861_86100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_performances_is_14_l861_86110

/-- Represents a performance schedule for an art festival. -/
structure PerformanceSchedule where
  singers : Nat
  performances : Nat
  singersPerPerformance : Nat
  pairAppearances : Nat

/-- Helper function to count the number of performances two singers perform together. -/
def number_of_performances_together (s : PerformanceSchedule) (p q : Nat) : Nat :=
  sorry

/-- Checks if the performance schedule satisfies the given conditions. -/
def isValidSchedule (s : PerformanceSchedule) : Prop :=
  s.singers = 108 ∧
  s.singersPerPerformance = 4 ∧
  ∀ (group : Finset Nat) (pair : Finset Nat),
    group.card = 8 → pair.card = 2 → pair ⊆ group →
    ∃ (r : Nat), ∀ (p q : Nat), p ∈ pair → q ∈ pair → p ≠ q →
      (number_of_performances_together s p q) = r

/-- The minimum number of performances needed. -/
def minPerformances : Nat := 14

/-- Theorem stating that 14 is the minimum number of performances needed. -/
theorem min_performances_is_14 :
  ∀ (s : PerformanceSchedule), isValidSchedule s →
    s.performances ≥ minPerformances ∧
    ∃ (s' : PerformanceSchedule), isValidSchedule s' ∧ s'.performances = minPerformances :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_performances_is_14_l861_86110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_center_locus_l861_86123

/-- Represents a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Represents the incircle of a triangle -/
structure Incircle where
  center : Point
  radius : ℝ

/-- Represents a strip in a plane -/
structure Strip where
  lower : Line
  upper : Line

/-- Checks if three lines are parallel -/
def areParallel (l1 l2 l3 : Line) : Prop :=
  sorry

/-- Checks if a triangle has vertices on three given lines -/
def hasVerticesOnLines (t : Triangle) (l1 l2 l3 : Line) : Prop :=
  sorry

/-- Checks if a strip's boundaries are parallel to given lines and halfway between them -/
def isHalfwayParallelStrip (s : Strip) (l1 l2 l3 : Line) : Prop :=
  sorry

/-- Checks if a point is within a strip -/
def isInStrip (p : Point) (s : Strip) : Prop :=
  sorry

/-- The main theorem -/
theorem incircle_center_locus 
  (l1 l2 l3 : Line) 
  (h : areParallel l1 l2 l3) :
  ∃ (s : Strip), 
    isHalfwayParallelStrip s l1 l2 l3 ∧ 
    ∀ (t : Triangle) (i : Incircle), 
      hasVerticesOnLines t l1 l2 l3 → 
      isInStrip i.center s :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_center_locus_l861_86123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_complement_of_B_l861_86178

-- Define the universe U as ℝ
def U : Type := ℝ

-- Define the set A
def A : Set ℝ := {t | ∃ a b : ℝ, a ≠ 0 ∧ a > b ∧
  (∀ x : ℝ, x ≠ -1/a → a*x^2 + 2*x + b > 0) ∧
  t = (a^2 + b^2)/(a - b)}

-- Define the set B
def B : Set ℝ := {m | ∀ x : ℝ, |x + 1| - |x - 3| ≤ m^2 - 3*m}

-- State the theorem
theorem intersection_of_A_and_complement_of_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 2 * Real.sqrt 2 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_complement_of_B_l861_86178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_propositions_false_l861_86180

-- Define proposition p
def proposition_p : Prop := 
  ∀ f : ℝ → ℝ, (¬ (∀ x : ℝ, f x = f (-x))) → (∀ x : ℝ, f (-x) ≠ f x)

-- Define function f for proposition q
noncomputable def f (x : ℝ) : ℝ := x * abs x

-- Define proposition q
def proposition_q : Prop :=
  (∀ x y : ℝ, x < y ∧ x < 0 ∧ y < 0 → f x > f y) ∧
  (∀ x y : ℝ, 0 < x ∧ x < y → f x < f y)

-- Theorem stating both propositions are false
theorem both_propositions_false : ¬ proposition_p ∧ ¬ proposition_q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_propositions_false_l861_86180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_equals_21_l861_86113

noncomputable def angle_sequence : List ℝ := 
  List.range 44 |> List.map (fun n => (5 + 4 * n) * Real.pi / 180)

theorem cosine_sum_equals_21 :
  (angle_sequence.map (fun θ => Real.cos θ ^ 2)).sum = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_equals_21_l861_86113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l861_86196

-- Part I
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

noncomputable def F (a b c : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then f a b c x else -f a b c x

theorem part_one (a b : ℝ) (ha : a > 0) :
  f a b 1 (-1) = 0 → F a b 1 2 + F a b 1 (-2) = 8 := by sorry

-- Part II
def g (b x : ℝ) : ℝ := x^2 + b * x

theorem part_two (b : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, |g b x| ≤ 1) → -2 ≤ b ∧ b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l861_86196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_value_l861_86144

def sequence_a : ℕ → ℚ
| 0 => 1
| n + 1 => sequence_a n / (2 * sequence_a n + 3)

theorem a_4_value : sequence_a 3 = 1 / 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_value_l861_86144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_horizontal_asymptote_l861_86106

/-- The function for which we want to find the horizontal asymptote -/
noncomputable def f (x : ℝ) : ℝ := (15 * x^4 + 6 * x^3 + 5 * x^2 + 2 * x + 7) / (5 * x^4 + 3 * x^3 + 4 * x^2 + 2 * x + 1)

/-- The horizontal asymptote of the function f -/
def horizontal_asymptote : ℝ := 3

/-- Theorem stating that the horizontal asymptote of f is 3 -/
theorem f_horizontal_asymptote :
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - horizontal_asymptote| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_horizontal_asymptote_l861_86106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_and_remainder_l861_86170

theorem largest_power_and_remainder : ∃ n : ℕ,
  (∀ k : ℕ, 3^k ∣ (2014^(Nat.factorial 100) - 2011^(Nat.factorial 100)) → k ≤ n) ∧
  (3^n ∣ (2014^(Nat.factorial 100) - 2011^(Nat.factorial 100))) ∧
  n = 49 ∧
  3^49 % 1000 = 83 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_and_remainder_l861_86170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_trick_always_works_l861_86117

/-- Given a circle with 13 positions, this function checks if two positions
    are within the set {k+1, k+2, k+5, k+7} (modulo 13) for some k. -/
def are_positions_covered (p1 p2 : Fin 13) : Prop :=
  ∃ k : Fin 13, (p1 ∈ ({k+1, k+2, k+5, k+7} : Finset (Fin 13)) ∧
                 p2 ∈ ({k+1, k+2, k+5, k+7} : Finset (Fin 13)))

/-- Theorem stating that for any two distinct positions in a circle of 13 positions,
    there exists a position k such that both original positions are in the set
    {k+1, k+2, k+5, k+7} (modulo 13). -/
theorem magic_trick_always_works :
  ∀ p1 p2 : Fin 13, p1 ≠ p2 → are_positions_covered p1 p2 :=
by
  sorry

#check magic_trick_always_works

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_trick_always_works_l861_86117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_equals_3_power_a_c_10_value_l861_86135

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => a (n + 1) + a n + 1

def c : ℕ → ℕ
  | 0 => 3
  | 1 => 9
  | (n + 2) => c (n + 1) * c n

theorem c_equals_3_power_a (n : ℕ) : c n = 3^(a n) := by
  sorry

theorem c_10_value : c 10 = 3^143 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_equals_3_power_a_c_10_value_l861_86135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l861_86161

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Theorem: The distance between points (4, 3) and (7, -1) is 5 -/
theorem distance_between_specific_points :
  distance 4 3 7 (-1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l861_86161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_max_ab_value_max_ab_value_achieved_l861_86168

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.log (a * x + b) + x^2

-- Part I
theorem tangent_line_condition (a b : ℝ) (h : a ≠ 0) :
  (∀ x, (deriv (f a b)) 1 * (x - 1) + f a b 1 = x) →
  a = -1 ∧ b = 2 := by sorry

-- Part II
theorem max_ab_value (a b : ℝ) (h : a ≠ 0) :
  (∀ x, f a b x ≤ x^2 + x) →
  a * b ≤ Real.exp 1 / 2 := by sorry

theorem max_ab_value_achieved :
  ∃ a b, a ≠ 0 ∧ (∀ x, f a b x ≤ x^2 + x) ∧ a * b = Real.exp 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_max_ab_value_max_ab_value_achieved_l861_86168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_circumcircle_l861_86159

-- Define the points
def E : ℝ × ℝ := (1, 0)
def K : ℝ × ℝ := (-1, 0)

-- Define the moving point P
variable (P : ℝ × ℝ)

-- Define vectors
def PE (P : ℝ × ℝ) : ℝ × ℝ := (E.1 - P.1, E.2 - P.2)
def PK (P : ℝ × ℝ) : ℝ × ℝ := (K.1 - P.1, K.2 - P.2)
def EK : ℝ × ℝ := (K.1 - E.1, K.2 - E.2)

-- Define dot product
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector magnitude
noncomputable def mag (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the condition |PE| * |KE| = PK · EK
def condition (P : ℝ × ℝ) : Prop :=
  mag (PE P) * mag EK = dot (PK P) EK

-- Define the trajectory C
def C (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1

-- Define line l passing through K
def l (m : ℝ) (y : ℝ) : ℝ := m * y - 1

-- Define points A, B, and D
variable (A B : ℝ × ℝ)
def D (A : ℝ × ℝ) : ℝ × ℝ := (A.1, -A.2)

-- Define the condition EA · EB = -8
def conditionEAEB (A B : ℝ × ℝ) : Prop :=
  dot (PE A) (PE B) = -8

-- Define the circumcircle of triangle ABD
def circumcircle (x y : ℝ) : Prop := (x - 9)^2 + y^2 = 40

-- Theorem statement
theorem trajectory_and_circumcircle 
  (hP : condition P) 
  (hC : C P) 
  (hA : C A ∧ A.2 > 0) 
  (hB : C B) 
  (hl : ∃ m, A.1 = l m A.2 ∧ B.1 = l m B.2) 
  (hEAEB : conditionEAEB A B) :
  (∀ x y, C (x, y) ↔ y^2 = 4*x) ∧
  (∀ x y, (x, y) = D A ∨ (x, y) = A ∨ (x, y) = B → circumcircle x y) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_circumcircle_l861_86159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l861_86188

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the properties of the triangle
def RightAngled (t : Triangle) : Prop :=
  t.A = 90

def SideBC (t : Triangle) : ℝ :=
  30

def TanCRelation (t : Triangle) : Prop :=
  Real.tan t.C = 3 * Real.cos t.C

-- Define the length of side AB
noncomputable def SideAB (t : Triangle) : ℝ :=
  Real.sqrt (SideBC t ^ 2 - (SideBC t * Real.cos t.C) ^ 2)

-- State the theorem
theorem triangle_side_length (t : Triangle) 
  (h1 : RightAngled t) 
  (h2 : TanCRelation t) : 
  SideAB t = 26 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l861_86188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irregular_shingle_area_l861_86129

/-- The area of an irregular-shaped roof shingle -/
theorem irregular_shingle_area : ℝ := by
  let rectangle_length : ℝ := 10
  let rectangle_width : ℝ := 7
  let trapezoid_height : ℝ := 2
  let trapezoid_base1 : ℝ := 6
  let trapezoid_base2 : ℝ := rectangle_length - trapezoid_base1
  let rectangle_area := rectangle_length * rectangle_width
  let trapezoid_area := (trapezoid_base1 + trapezoid_base2) * trapezoid_height / 2
  have h : rectangle_area - trapezoid_area = 60 := by
    -- Proof steps would go here
    sorry
  exact 60


end NUMINAMATH_CALUDE_ERRORFEEDBACK_irregular_shingle_area_l861_86129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_power_functions_l861_86186

noncomputable def f₁ (x : ℝ) : ℝ := x⁻¹
noncomputable def f₂ (x : ℝ) : ℝ := Real.sqrt x
def f₃ (x : ℝ) : ℝ := x
def f₄ (x : ℝ) : ℝ := x^2
def f₅ (x : ℝ) : ℝ := x^3

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem increasing_power_functions :
  (¬is_increasing f₁) ∧
  (is_increasing f₂) ∧
  (is_increasing f₃) ∧
  (¬is_increasing f₄) ∧
  (is_increasing f₅) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_power_functions_l861_86186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_break_height_l861_86165

/-- Represents the height at which a flagpole breaks -/
noncomputable def breaking_height (total_height tip_height angle : ℝ) : ℝ :=
  total_height - (total_height - tip_height) / 2

/-- Theorem stating the breaking height of a flagpole under specific conditions -/
theorem flagpole_break_height :
  let total_height : ℝ := 18
  let tip_height : ℝ := 3
  let angle : ℝ := 30 * (Real.pi / 180)  -- Convert degrees to radians
  breaking_height total_height tip_height angle = 7.5 := by
  sorry

-- #eval breaking_height 18 3 (30 * (Real.pi / 180))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_break_height_l861_86165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tens_digit_of_9_pow_2023_l861_86195

theorem tens_digit_of_9_pow_2023 : ∃ k : ℕ, 9^2023 ≡ 20 + k [MOD 100] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tens_digit_of_9_pow_2023_l861_86195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l861_86154

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2

theorem f_monotone_increasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l861_86154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_approximation_l861_86130

def f (x : ℝ) : ℝ := x

def c : ℝ := 9.237333333333334

theorem solution_approximation :
  ∃ x : ℝ, (abs ((f (69.28 * x) / 0.03) - c) < 0.000001) ∧ 
           (abs (x - 0.004) < 0.000001) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_approximation_l861_86130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_is_half_radian_l861_86124

-- Define the sector
structure CircularSector where
  radius : ℝ
  angle : ℝ

-- Define the perimeter and area functions
noncomputable def perimeter (s : CircularSector) : ℝ := 2 * s.radius + s.radius * s.angle
noncomputable def area (s : CircularSector) : ℝ := 1/2 * s.radius^2 * s.angle

-- Theorem statement
theorem sector_angle_is_half_radian (s : CircularSector) :
  perimeter s = 5 ∧ area s = 1 → s.angle = 1/2 := by
  sorry

#check sector_angle_is_half_radian

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_is_half_radian_l861_86124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_dot_product_bounds_l861_86155

/-- Given a parallelogram ABCD with specific properties, prove the bounds of the dot product PA · PB -/
theorem parallelogram_dot_product_bounds (A B C D P : ℝ × ℝ) : 
  (∀ (X Y : ℝ × ℝ), (X - Y) = (D - A) ↔ (X - Y) = (C - B)) →  -- ABCD is a parallelogram
  ‖B - A‖ = 4 →  -- AB = 4
  ‖D - A‖ = 2 →  -- AD = 2
  (B - A) • (D - A) = 4 →  -- ⃗AB · ⃗AD = 4
  (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = C + t • (D - C)) →  -- P is on side CD
  -1 ≤ (P - A) • (P - B) ∧ (P - A) • (P - B) ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_dot_product_bounds_l861_86155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equality_l861_86151

noncomputable def circle_area (diameter : ℝ) : ℝ := Real.pi * (diameter / 2) ^ 2

noncomputable def v (d1 d2 d3 : ℝ) : ℝ :=
  circle_area d1 + circle_area d2 + circle_area d3 - Real.pi * (6 / 2) ^ 2

noncomputable def w (d : ℝ) : ℝ := circle_area d

theorem area_equality (d1 d2 d3 d4 : ℝ) 
  (h1 : d1 = 4) (h2 : d2 = 4) (h3 : d3 = 2) (h4 : d4 = 6) :
  v d1 d2 d3 = w d4 := by
  sorry

#eval "Theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equality_l861_86151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l861_86145

/-- Calculates the length of a train given its speed, the speed of a person moving in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length_calculation (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 84 →
  person_speed = 6 →
  passing_time = 4.399648028157747 →
  ∃ (train_length : ℝ), abs (train_length - 110.991201) < 0.000001 := by
  intro h_train_speed h_person_speed h_passing_time
  -- The proof steps would go here
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l861_86145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_point_with_tangents_l861_86125

/-- The trajectory of a point P with tangents to a unit circle -/
theorem trajectory_of_point_with_tangents (x y : ℝ) :
  -- Given a circle with equation x^2 + y^2 = 1
  -- And a point P(x, y) outside the circle
  -- Such that tangents PA and PB can be drawn to the circle
  -- Where A and B are the tangent points
  -- And the angle APB is 60 degrees
  -- Then the trajectory of P satisfies x^2 + y^2 = 4
  (∃ (A B : ℝ × ℝ), 
    -- A and B are on the unit circle
    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧
    -- PA and PB are tangent to the circle
    (x - A.1) * A.1 + (y - A.2) * A.2 = 0 ∧
    (x - B.1) * B.1 + (y - B.2) * B.2 = 0 ∧
    -- Angle APB is 60 degrees
    Real.arccos ((x - A.1) * (x - B.1) + (y - A.2) * (y - B.2)) / 
      (((x - A.1)^2 + (y - A.2)^2).sqrt * ((x - B.1)^2 + (y - B.2)^2).sqrt) = π / 3) →
  x^2 + y^2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_point_with_tangents_l861_86125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_count_l861_86192

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 - 4*x + y^2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 3 = 0

-- Define the number of common tangent lines
def num_common_tangents : ℕ := 4

-- Theorem statement
theorem common_tangents_count :
  ∃ (n : ℕ), n = num_common_tangents ∧ 
  n = (4 : ℕ) := by
  use 4
  constructor
  . rfl
  . rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_count_l861_86192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_comparison_indeterminate_l861_86181

theorem sine_comparison_indeterminate 
  (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : α > β) : 
  ¬ (∀ x y : Real, (x = Real.sin α ∧ y = Real.sin β) → (x > y ∨ x < y ∨ x = y)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_comparison_indeterminate_l861_86181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_log_sum_l861_86120

/-- A power function that passes through the point (1/2, √2/2) -/
noncomputable def f (x : ℝ) : ℝ := x^(1/2)

/-- The property that f passes through (1/2, √2/2) -/
axiom f_property : f (1/2) = Real.sqrt 2 / 2

/-- The theorem to be proved -/
theorem power_function_log_sum :
  Real.log (f 2) + Real.log (f 5) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_log_sum_l861_86120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_location_l861_86153

/-- The profit function for a company's promotional campaign -/
noncomputable def profit (x : ℝ) : ℝ := 26 - 4 / (x + 1) - x

/-- The derivative of the profit function -/
noncomputable def profit_derivative (x : ℝ) : ℝ := -4 / ((x + 1) ^ 2) - 1

/-- Theorem: The maximum profit occurs at x = 1 if a ≥ 1, and at x = a if a < 1 -/
theorem max_profit_location (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Icc 0 a, profit x ≤ profit (min 1 a)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_location_l861_86153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_profit_percentage_l861_86138

/-- Calculates the overall profit percentage for a merchant selling sugar. -/
theorem sugar_profit_percentage
  (total_sugar : ℝ)
  (sugar_18_percent : ℝ)
  (profit_8_percent : ℝ)
  (profit_18_percent : ℝ)
  (h1 : total_sugar = 1000)
  (h2 : sugar_18_percent = 600)
  (h3 : profit_8_percent = 8)
  (h4 : profit_18_percent = 18) :
  (let sugar_8_percent := total_sugar - sugar_18_percent
   let profit_8 := (profit_8_percent / 100) * sugar_8_percent
   let profit_18 := (profit_18_percent / 100) * sugar_18_percent
   let total_profit := profit_8 + profit_18
   let overall_profit_percentage := (total_profit / total_sugar) * 100
   overall_profit_percentage) = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_profit_percentage_l861_86138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yearly_cost_ratio_is_one_to_one_l861_86185

/-- The cost of raising a child for John --/
structure ChildRaisingCost where
  yearly_cost_first_8: ℝ
  years_until_university: ℕ
  university_tuition: ℝ
  total_cost: ℝ

/-- Theorem about the ratio of yearly costs --/
theorem yearly_cost_ratio_is_one_to_one (cost: ChildRaisingCost)
  (h1: cost.yearly_cost_first_8 = 10000)
  (h2: cost.years_until_university = 18)
  (h3: cost.university_tuition = 250000)
  (h4: cost.total_cost = 265000) :
  (cost.total_cost - cost.yearly_cost_first_8 * 8 / 2 - cost.university_tuition / 2) / 10 =
  cost.yearly_cost_first_8 := by
  sorry

#check yearly_cost_ratio_is_one_to_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yearly_cost_ratio_is_one_to_one_l861_86185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_ratio_l861_86136

def a (n : ℕ) : ℝ :=
  n * (n - 1) + 12

theorem min_value_of_sequence_ratio :
  ∀ n : ℕ, n ≥ 1 → a n / n ≥ 6 ∧ ∃ m : ℕ, m ≥ 1 ∧ a m / m = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_ratio_l861_86136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_count_l861_86169

structure Rectangle where
  vertices : Fin 4 → Point

structure Segment where
  endpoints : Fin 2 → Point

def divides_into_congruent_rectangles (r : Rectangle) (s : Segment) : Prop :=
  sorry

def is_right_triangle (p q r : Point) : Prop :=
  sorry

def count_right_triangles (points : List Point) : Nat :=
  sorry

theorem right_triangles_count
  (efgh : Rectangle)
  (rs : Segment)
  (e f g h r s : Point)
  (h1 : divides_into_congruent_rectangles efgh rs)
  (h2 : efgh.vertices 0 = e)
  (h3 : efgh.vertices 1 = f)
  (h4 : efgh.vertices 2 = g)
  (h5 : efgh.vertices 3 = h)
  (h6 : rs.endpoints 0 = r)
  (h7 : rs.endpoints 1 = s)
  : count_right_triangles [e, r, f, g, s, h] = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_count_l861_86169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_domain_l861_86198

-- Define h as a function from ℝ to ℝ
noncomputable def h : ℝ → ℝ := sorry

-- Define the domain of h
def h_domain (x : ℝ) : Prop := -10 ≤ x ∧ x ≤ 6

-- Define function p
noncomputable def p (x : ℝ) : ℝ := h (-3 * x + 1)

-- Theorem statement
theorem p_domain :
  {x : ℝ | h_domain (-3 * x + 1)} = {x : ℝ | -5/3 ≤ x ∧ x ≤ 11/3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_domain_l861_86198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_proposition_l861_86116

open Set Real

theorem negation_of_existential_proposition :
  (¬ ∃ x₀ : ℝ, x₀ ∈ Set.Ioi 0 ∧ x₀^2 ≤ x₀ + 2) ↔
  (∀ x : ℝ, x ∈ Set.Ioi 0 → x^2 > x + 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_proposition_l861_86116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_one_third_l861_86193

theorem reciprocal_of_negative_one_third :
  (1 : ℚ) / (-1/3 : ℚ) = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_one_third_l861_86193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_2s_l861_86127

/-- The displacement function for an object's motion --/
noncomputable def displacement (t : ℝ) : ℝ := 10 * t - t^2

/-- The velocity function derived from the displacement function --/
noncomputable def velocity (t : ℝ) : ℝ := 
  (displacement (t + 0.00001) - displacement t) / 0.00001

theorem velocity_at_2s : velocity 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_2s_l861_86127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_circular_sector_l861_86182

/-- The height of a cone formed from a sector of a circular sheet --/
theorem cone_height_from_circular_sector (r : ℝ) (h : r = 10) :
  let sector_arc_length := (2 * Real.pi * r) / 4
  let base_radius := sector_arc_length / (2 * Real.pi)
  let slant_height := r
  Real.sqrt (slant_height^2 - base_radius^2) = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_circular_sector_l861_86182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_defeat_probability_l861_86184

-- Define the probabilities
noncomputable def p_two_heads : ℝ := 1/4
noncomputable def p_one_head : ℝ := 1/3
noncomputable def p_no_heads : ℝ := 5/12

-- Define the expected change in number of heads after one strike
noncomputable def expected_change : ℝ := 2 * p_two_heads + p_one_head + 0 * p_no_heads - 1

-- Define the probability of defeating the dragon
def defeat_probability : ℝ := 1

-- Theorem statement
theorem dragon_defeat_probability :
  p_two_heads + p_one_head + p_no_heads = 1 ∧
  expected_change < 0 →
  defeat_probability = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_defeat_probability_l861_86184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_some_triangles_right_angled_inequality_solution_range_function_property_implies_sin_2theta_zero_l861_86177

-- Statement 1
theorem negation_some_triangles_right_angled :
  (∃ t : Type, ∃ (isRightAngled : t → Prop), ∃ x : t, isRightAngled x) →
  ¬(∀ t : Type, ∀ (isRightAngled : t → Prop), ∀ x : t, ¬isRightAngled x) :=
sorry

-- Statement 2
theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, x ≥ 1 ∧ a * x^2 - 2*x - 1 < 0) → a < 3 :=
sorry

-- Statement 3
theorem function_property_implies_sin_2theta_zero (θ : ℝ) :
  (∀ x : ℝ, Real.sin (2*(Real.pi/2 - x) + θ) = -Real.sin (2*x + θ)) → Real.sin (2*θ) = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_some_triangles_right_angled_inequality_solution_range_function_property_implies_sin_2theta_zero_l861_86177
