import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_battery_tries_theorem_l1234_123450

/-- A graph representing batteries and tries -/
structure BatteryGraph where
  n : ℕ
  vertices : Finset (Fin (2 * n))
  edges : Finset (Fin (2 * n) × Fin (2 * n))

/-- An independent set in the graph -/
def IndependentSet (G : BatteryGraph) (S : Finset (Fin (2 * G.n))) : Prop :=
  ∀ u v, u ∈ S → v ∈ S → u ≠ v → (u, v) ∉ G.edges ∧ (v, u) ∉ G.edges

/-- The main theorem -/
theorem battery_tries_theorem (G : BatteryGraph) (hn : G.n ≥ 4) 
    (hedge : G.edges.card ≤ G.n + 2) :
    ∃ S : Finset (Fin (2 * G.n)), IndependentSet G S ∧ S.card = G.n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_battery_tries_theorem_l1234_123450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l1234_123462

noncomputable section

/-- The lateral surface area of a frustum of a right circular cone. -/
def lateralSurfaceArea (r₁ r₂ h : ℝ) : ℝ :=
  let s := Real.sqrt (h^2 + (r₂ - r₁)^2)
  Real.pi * (r₁ + r₂) * s

/-- Theorem: The lateral surface area of a frustum of a right circular cone
    with upper base radius 5 inches, lower base radius 8 inches,
    and vertical height 9 inches is equal to 39π√10 square inches. -/
theorem frustum_lateral_surface_area :
  lateralSurfaceArea 5 8 9 = 39 * Real.pi * Real.sqrt 10 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l1234_123462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_figure_line_intersection_l1234_123447

/-- A convex figure in a 2D space -/
structure ConvexFigure where
  -- Add necessary fields and properties to define a convex figure
  is_convex : Bool

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  -- Add necessary fields to define a line
  dummy : Unit

/-- Predicate to check if a point is interior to a convex figure -/
def is_interior (p : Point) (f : ConvexFigure) : Prop := sorry

/-- Predicate to check if a point is on the boundary of a convex figure -/
def is_boundary (p : Point) (f : ConvexFigure) : Prop := sorry

/-- Predicate to check if a convex figure is bounded -/
def is_bounded (f : ConvexFigure) : Prop := sorry

/-- Function to count intersection points of a line with the boundary of a convex figure -/
def count_boundary_intersections (l : Line) (f : ConvexFigure) : ℕ := sorry

/-- Predicate to check if a point is on a line -/
def point_on_line (p : Point) (l : Line) : Prop := sorry

theorem convex_figure_line_intersection 
  (f : ConvexFigure) 
  (l : Line) 
  (p : Point) 
  (h1 : f.is_convex = true) 
  (h2 : is_interior p f) 
  (h3 : point_on_line p l) : 
  (count_boundary_intersections l f ≤ 2) ∧ 
  (is_bounded f → count_boundary_intersections l f = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_figure_line_intersection_l1234_123447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_16_l1234_123451

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

noncomputable def a : ℤ := floor ((Real.sqrt 3 - Real.sqrt 2) ^ 2009) + 16

theorem a_equals_16 : a = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_16_l1234_123451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_N_l1234_123495

open Nat

def has_more_ones_than_zeros (n : ℕ) : Bool :=
  let binary := n.digits 2
  binary.count 1 > binary.count 0

def N : ℕ := (Finset.range 1051).filter (λ n => has_more_ones_than_zeros n) |>.card

theorem remainder_of_N : N % 1000 = 737 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_N_l1234_123495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_at_noon_l1234_123423

/-- The time when the first train started from station A -/
def start_time : ℝ := 7

/-- The distance between stations A and B in kilometers -/
def distance : ℝ := 200

/-- The speed of the first train in km/h -/
def speed_train1 : ℝ := 20

/-- The speed of the second train in km/h -/
def speed_train2 : ℝ := 25

/-- The time when the second train started from station B -/
def start_time_train2 : ℝ := 8

/-- The time when the trains meet -/
def meet_time : ℝ := 12

theorem trains_meet_at_noon :
  start_time = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_at_noon_l1234_123423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_f_of_f_eq_zero_l1234_123484

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 then -0.5 * x^2 + x + 3
  else if x < 1 then 3.5 - x
  else 0.5 * x^2 - x + 1.5

-- Define the domain of f
def domain : Set ℝ := Set.Icc (-5) 5

-- State the theorem
theorem no_solutions_for_f_of_f_eq_zero :
  ∀ x ∈ domain, f (f x) ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_f_of_f_eq_zero_l1234_123484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_trailing_zeros_is_eight_l1234_123497

/-- A type representing a nine-digit number formed from digits 1 to 9 --/
def NineDigitNumber := Fin 9 → Fin 9

/-- The set of all valid nine-digit numbers --/
def ValidNumbers : Set NineDigitNumber :=
  {n | ∀ i j, i ≠ j → n i ≠ n j}

/-- The sum of nine valid nine-digit numbers --/
def SumOfNineNumbers (nums : Fin 9 → NineDigitNumber) : ℕ :=
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => (nums i j).val + 1) * 10^(8 - (nums i 0).val))

/-- The number of trailing zeros in a natural number --/
def trailingZeros (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log 10 (n.gcd (10^(Nat.log 10 n + 1)))

/-- The theorem stating the maximum number of trailing zeros --/
theorem max_trailing_zeros_is_eight :
  ∃ (nums : Fin 9 → NineDigitNumber),
    (∀ i, nums i ∈ ValidNumbers) ∧
    trailingZeros (SumOfNineNumbers nums) = 8 ∧
    ∀ (other_nums : Fin 9 → NineDigitNumber),
      (∀ i, other_nums i ∈ ValidNumbers) →
      trailingZeros (SumOfNineNumbers other_nums) ≤ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_trailing_zeros_is_eight_l1234_123497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_limit_l1234_123421

/-- The infinite nested radical expression -/
noncomputable def nestedRadical : ℝ → ℝ := fun x ↦ Real.sqrt (18 + x)

/-- The limit of the nested radical sequence -/
noncomputable def limitNestedRadical : ℝ := (1 + Real.sqrt 73) / 2

/-- Theorem stating that the limit of the nested radical is a fixed point of the function
    and is equal to the calculated value -/
theorem nested_radical_limit :
  ∃ (x : ℝ), x = nestedRadical x ∧ x = limitNestedRadical :=
by
  -- We'll use the calculated value as our witness
  use limitNestedRadical
  constructor
  · -- Prove that limitNestedRadical is a fixed point of nestedRadical
    sorry
  · -- Prove that limitNestedRadical equals itself (trivial)
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_limit_l1234_123421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1234_123432

/-- Given a hyperbola with equation x^2 - 2y^2 = 1, its asymptotes are y = ±(√2/2)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 - 2*y^2 = 1) →
  (∃ (k : ℝ), k = Real.sqrt 2 / 2 ∧ (y = k*x ∨ y = -k*x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1234_123432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_from_sines_l1234_123437

theorem angle_sum_from_sines (α β : ℝ) (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2) (h3 : Real.sin α = Real.sqrt 5/5) (h4 : Real.sin β = Real.sqrt 10/10) :
  α + β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_from_sines_l1234_123437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_x_value_l1234_123408

def sequenceList : List ℕ := [2, 4, 8, 14, 32]

def differences (s : List ℕ) : List ℕ :=
  List.zipWith (·-·) s.tail s

theorem sequence_x_value (x : ℕ) :
  differences (sequenceList.insertNth 4 x) = [2, 4, 6, 8, 18 - x] →
  x = 22 := by
  sorry

#eval sequenceList
#eval differences sequenceList

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_x_value_l1234_123408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_microscope_magnification_l1234_123483

/-- The magnification factor of an electron microscope. -/
noncomputable def magnification_factor (magnified_diameter actual_diameter : ℝ) : ℝ :=
  magnified_diameter / actual_diameter

/-- Theorem: The magnification factor is 1000 given the specified diameters. -/
theorem microscope_magnification :
  let magnified_diameter : ℝ := 1
  let actual_diameter : ℝ := 0.001
  magnification_factor magnified_diameter actual_diameter = 1000 := by
  -- Unfold the definition of magnification_factor
  unfold magnification_factor
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_microscope_magnification_l1234_123483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_present_weight_l1234_123428

-- Define Jake's and Kendra's weights
variable (jake_weight : ℝ)
variable (kendra_weight : ℝ)

-- Condition 1: If Jake loses 8 pounds, he will weigh twice as much as Kendra
axiom condition1 : jake_weight - 8 = 2 * kendra_weight

-- Condition 2: Together, Jake and Kendra weigh 293 pounds
axiom condition2 : jake_weight + kendra_weight = 293

-- Theorem to prove
theorem jake_present_weight :
  jake_weight = 198 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_present_weight_l1234_123428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base5_division_proof_l1234_123459

/-- Represents a number in base 5 --/
structure Base5 where
  value : Nat

/-- Converts a base 5 number to its decimal representation --/
def to_decimal (n : Base5) : Nat := sorry

/-- Converts a decimal number to its base 5 representation --/
def to_base5 (n : Nat) : Base5 := sorry

/-- Performs division in base 5 --/
def base5_div (a b : Base5) : Base5 := sorry

theorem base5_division_proof :
  base5_div (Base5.mk 24342) (Base5.mk 23) = Base5.mk 43 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base5_division_proof_l1234_123459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1234_123443

/-- A parabola with equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- Point B -/
def B : ℝ × ℝ := (3, 0)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_theorem (A : ℝ × ℝ) 
  (h1 : A ∈ Parabola) 
  (h2 : distance A Focus = distance B Focus) : 
  distance A B = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1234_123443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_complex_l1234_123438

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  ∃ w : ℂ, Complex.abs w = 3 ∧ Complex.abs ((2 + 3*Complex.I)*(w^4) - w^6) = 81 * Real.sqrt 34 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_complex_l1234_123438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_l1234_123426

noncomputable def terminal_point : ℝ × ℝ := (Real.sqrt 3 / 2, -1 / 2)

theorem smallest_positive_angle (α : ℝ) :
  (Real.cos α = terminal_point.fst ∧ Real.sin α = terminal_point.snd) →
  ∃ k : ℤ, α = 11 * Real.pi / 6 + 2 * Real.pi * ↑k ∧ 
  ∀ m : ℤ, 0 < 11 * Real.pi / 6 + 2 * Real.pi * ↑m → 
  11 * Real.pi / 6 + 2 * Real.pi * ↑k ≤ 11 * Real.pi / 6 + 2 * Real.pi * ↑m :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_l1234_123426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_properties_l1234_123475

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total edge length of a cuboid -/
def totalEdgeLength (c : Cuboid) : ℝ :=
  4 * (c.length + c.width + c.height)

/-- Calculates the surface area of a cuboid -/
def surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ :=
  c.length * c.width * c.height

/-- Theorem about a specific cuboid's properties -/
theorem cuboid_properties :
  ∃ c : Cuboid,
    totalEdgeLength c = 72 ∧
    c.length = (3 : ℝ) * c.height ∧
    c.width = (2 : ℝ) * c.height ∧
    surfaceArea c = 198 ∧
    volume c = 162 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_properties_l1234_123475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_number_existence_l1234_123410

-- Define an approximation relation for natural numbers and reals
def approx (n : ℕ) (x : ℝ) := abs (x - n) < 0.5

notation:50 n " ≈ " x => approx n x

theorem base_number_existence : ∃ (x : ℕ) (k : ℝ), 
  k > 0 ∧ 
  (x : ℝ) ^ k = 4 ∧ 
  (x : ℝ) ^ (2 * k + 3) = 3456 ∧ 
  x ≈ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_number_existence_l1234_123410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_return_to_initial_config_l1234_123465

-- Define the strip as a function from integers to natural numbers (number of stones on each square)
def Strip := ℤ → ℕ

-- Define a valid move
def validMove (s₁ s₂ : Strip) : Prop :=
  ∃ i : ℤ, s₁ i ≥ 2 ∧
    s₂ i = s₁ i - 2 ∧
    s₂ (i + 1) = s₁ (i + 1) + 1 ∧
    s₂ (i - 1) = s₁ (i - 1) + 1 ∧
    ∀ j : ℤ, j ≠ i ∧ j ≠ i + 1 ∧ j ≠ i - 1 → s₂ j = s₁ j

-- Define a sequence of valid moves
def validMoveSequence (s₁ s₂ : Strip) : Prop :=
  ∃ n : ℕ, ∃ seq : Fin (n + 1) → Strip,
    seq ⟨0, Nat.zero_lt_succ n⟩ = s₁ ∧
    seq ⟨n, Nat.lt_succ_self n⟩ = s₂ ∧
    ∀ i : Fin n, validMove (seq i) (seq i.succ)

-- Define the quantity q for a given strip configuration
noncomputable def q (s : Strip) : ℝ := ∑' i, (s i : ℝ) * (2 : ℝ) ^ (i : ℝ)

-- State the theorem
theorem no_return_to_initial_config (s : Strip) :
  ¬∃ s' : Strip, s' ≠ s ∧ validMoveSequence s s' ∧ validMoveSequence s' s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_return_to_initial_config_l1234_123465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_greater_of_function_less_l1234_123479

open Real Set

theorem square_greater_of_function_less (f : ℝ → ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x ∈ Icc (-π/4) (π/4) → f x = sin x^4 + cos x^4) →
  x₁ ∈ Icc (-π/4) (π/4) →
  x₂ ∈ Icc (-π/4) (π/4) →
  f x₁ < f x₂ →
  x₁^2 > x₂^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_greater_of_function_less_l1234_123479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_and_ratio_l1234_123433

noncomputable def equation (x : ℝ) : Prop := (7 * x) / 5 + 2 = 4 / x

noncomputable def x_form (a b c d : ℤ) : ℝ := (a + b * Real.sqrt c) / d

theorem largest_solution_and_ratio :
  ∃ (a b c d : ℤ),
    equation (x_form a b c d) ∧
    (∀ x, equation x → x ≤ x_form a b c d) ∧
    x_form a b c d = (-5 + 5 * Real.sqrt 66) / 7 ∧
    (a * c * d) / b = -462 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_and_ratio_l1234_123433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_negative_one_l1234_123487

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add this case to handle Nat.zero
  | 1 => 2
  | n + 1 => 1 - 1 / sequence_a n

theorem a_2016_equals_negative_one : sequence_a 2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_negative_one_l1234_123487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_weight_approx_l1234_123430

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 27

/-- The atomic weight of Chlorine in g/mol -/
def atomic_weight_Cl : ℝ := 35.5

/-- The number of Aluminum atoms in the compound -/
def num_Al : ℕ := 1

/-- The number of Chlorine atoms in the compound -/
def num_Cl : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 132

/-- Theorem stating that the molecular weight of the compound is approximately 132 g/mol -/
theorem compound_weight_approx : 
  |((num_Al : ℝ) * atomic_weight_Al + (num_Cl : ℝ) * atomic_weight_Cl) - molecular_weight| < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_weight_approx_l1234_123430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_configuration_iff_odd_l1234_123453

/-- A sequence of natural numbers forms a valid configuration if:
    1. It has length greater than 2
    2. Not all numbers are equal
    3. The products form an arithmetic progression with nonzero common difference -/
def ValidConfiguration (a : List ℕ) : Prop :=
  a.length > 2 ∧
  ∃ i j, i < a.length ∧ j < a.length ∧ a[i]? ≠ a[j]? ∧
  ∃ d : ℤ, d ≠ 0 ∧ ∀ i < a.length - 1,
    (a[i]?.getD 1 * a[i+1]?.getD 1 : ℤ) - (a[i-1]?.getD 1 * a[i]?.getD 1) = d

theorem valid_configuration_iff_odd (n : ℕ) :
  (∃ a : List ℕ, a.length = n ∧ ValidConfiguration a) ↔ Odd n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_configuration_iff_odd_l1234_123453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_points_is_120_degrees_l1234_123494

noncomputable section

/-- The angle between two points on a unit sphere --/
def sphereAngle (lat1 : Real) (long1 : Real) (lat2 : Real) (long2 : Real) : Real :=
  let x1 := Real.cos lat1 * Real.cos long1
  let y1 := Real.cos lat1 * Real.sin long1
  let z1 := Real.sin lat1
  let x2 := Real.cos lat2 * Real.cos long2
  let y2 := Real.cos lat2 * Real.sin long2
  let z2 := Real.sin lat2
  Real.arccos (x1 * x2 + y1 * y2 + z1 * z2)

/-- Convert degrees to radians --/
def degToRad (deg : Real) : Real :=
  deg * Real.pi / 180

theorem angle_between_points_is_120_degrees :
  sphereAngle (degToRad 0) (degToRad 110) (degToRad 45) (degToRad (-115)) = degToRad 120 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_points_is_120_degrees_l1234_123494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_treasure_determination_possible_l1234_123414

-- Define the types of people on the island
inductive PersonType
  | Knight
  | Liar
  | Normal
deriving BEq, Repr

-- Define a subject as a person with a type
structure Subject where
  name : String
  type : PersonType
deriving BEq, Repr

-- Define the island
structure Island where
  subjects : List Subject
  hasTreasure : Bool

-- Define a question as a function that takes a subject and returns a boolean
def Question := Subject → Bool

-- Theorem statement
theorem treasure_determination_possible 
  (island : Island) 
  (A B C : Subject) :
  (island.subjects = [A, B, C]) →
  (∀ s : Subject, s ∈ island.subjects → s.type ∈ [PersonType.Knight, PersonType.Liar, PersonType.Normal]) →
  (List.count PersonType.Normal (List.map Subject.type island.subjects) ≤ 1) →
  (∃ (q1 q2 : Question), ∃ (f : Bool → Bool → Bool), 
    f (q1 A) (q2 (if q1 A then C else B)) = island.hasTreasure) :=
by sorry

#check treasure_determination_possible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_treasure_determination_possible_l1234_123414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_15_degrees_l1234_123476

noncomputable def sectorArea (r : ℝ) (θ : ℝ) : ℝ :=
  1/2 * r^2 * (θ * Real.pi / 180)

theorem sector_area_15_degrees (r : ℝ) (h : r = 6) :
  sectorArea r 15 = 3 * Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_15_degrees_l1234_123476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_calculation_l1234_123490

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The given complex expression -/
noncomputable def complex_expr : ℂ := (1 - i)^3 / i

theorem complex_calculation : complex_expr = -2 + 2*i := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_calculation_l1234_123490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_l1234_123486

/-- Given a line segment AB with midpoints as described, prove its length --/
theorem segment_length (A B C D E F G : EuclideanSpace ℝ (Fin 3)) : 
  (C = (1/2 : ℝ) • (A + B)) →  -- C is midpoint of AB
  (D = (1/2 : ℝ) • (A + C)) →  -- D is midpoint of AC
  (E = (1/2 : ℝ) • (A + D)) →  -- E is midpoint of AD
  (F = (1/2 : ℝ) • (A + E)) →  -- F is midpoint of AE
  (G = (1/2 : ℝ) • (A + F)) →  -- G is midpoint of AF
  (dist A G = 4) →     -- AG = 4
  (dist A B = 128) :=  -- AB = 128
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_l1234_123486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1234_123464

/-- The function f(x) after translation and symmetry adjustment -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x + 2 * Real.pi / 3)

/-- The theorem stating the minimum value of f(x) in the given interval -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = -Real.sqrt 3 ∧
  ∀ x, x ∈ Set.Icc (-Real.pi / 2) 0 → f x ≥ min ∧
  ∃ x₀, x₀ ∈ Set.Icc (-Real.pi / 2) 0 ∧ f x₀ = min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1234_123464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_candidate_percentage_l1234_123473

def candidate1_votes : ℕ := 5136
def candidate2_votes : ℕ := 7636
def candidate3_votes : ℕ := 11628

def total_votes : ℕ := candidate1_votes + candidate2_votes + candidate3_votes

def winning_votes : ℕ := max candidate1_votes (max candidate2_votes candidate3_votes)

noncomputable def winning_percentage : ℚ := (winning_votes : ℚ) / (total_votes : ℚ) * 100

theorem winning_candidate_percentage :
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 : ℚ) / 100 ∧ |winning_percentage - (4766 : ℚ) / 100| < ε :=
by
  sorry

#eval winning_votes
#eval total_votes
#eval (winning_votes : ℚ) / (total_votes : ℚ) * 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_candidate_percentage_l1234_123473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_extrema_line_origin_l1234_123477

/-- A cubic function with coefficients a, b, c, and d. -/
def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

/-- Condition for the existence of two extrema. -/
def has_two_extrema (a b c : ℝ) : Prop :=
  b^2 - 3*a*c > 0

/-- The x-coordinates of the extrema. -/
def extrema_x (a b c : ℝ) : Set ℝ :=
  {x : ℝ | 3*a*x^2 + 2*b*x + c = 0}

/-- The line connecting the extrema passes through the origin. -/
def extrema_line_through_origin (a b c d : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ extrema_x a b c → x₂ ∈ extrema_x a b c → x₁ ≠ x₂ →
    (cubic_function a b c d x₁) * x₂ = (cubic_function a b c d x₂) * x₁

theorem cubic_extrema_line_origin (a b c d : ℝ) :
  has_two_extrema a b c →
  extrema_line_through_origin a b c d →
  9*a*d = b*c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_extrema_line_origin_l1234_123477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_desiree_age_l1234_123452

/-- Desiree's current age -/
def D : ℝ := sorry

/-- Desiree's cousin's current age -/
def C : ℝ := sorry

/-- Grandmother's current age -/
def G : ℝ := sorry

/-- Grandfather's current age -/
def Gr : ℝ := sorry

/-- Desiree is twice as old as her cousin now -/
axiom h1 : D = 2 * C

/-- In 30 years, Desiree's age will be 14 years more than two-thirds of her cousin's age -/
axiom h2 : D + 30 = (2/3) * (C + 30) + 14

/-- Their grandmother's age is currently the sum of Desiree's and her cousin's ages -/
axiom h3 : G = D + C

/-- In 20 years, their grandmother's age will be three times the difference between Desiree's and her cousin's ages at that time -/
axiom h4 : G + 20 = 3 * (D - C)

/-- Their grandfather's current age is equal to half the product of the ages of Desiree and her cousin in 10 years -/
axiom h5 : Gr = (1/2) * (D + 10) * (C + 10)

theorem desiree_age : D = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_desiree_age_l1234_123452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_sqrt_2021_l1234_123425

/-- Represents a straight line in 2D Euclidean space. -/
def Line : Type := EuclideanSpace ℝ (Fin 2) → Prop

/-- Represents a circle in 2D Euclidean space. -/
def Circle : Type := EuclideanSpace ℝ (Fin 2) → Prop

/-- Predicate to determine if points are constructible using given constructions. -/
def constructible (A B C D : EuclideanSpace ℝ (Fin 2)) 
  (constructions : List (Line ⊕ Circle)) : Prop :=
sorry

/-- Given two points with distance 1, it is possible to construct two points
    with distance √2021 using at most 10 straight lines and circles. -/
theorem construct_sqrt_2021 (A B : EuclideanSpace ℝ (Fin 2)) 
  (h : dist A B = 1) : 
  ∃ (C D : EuclideanSpace ℝ (Fin 2)) (constructions : List (Line ⊕ Circle)),
    dist C D = Real.sqrt 2021 ∧ 
    constructions.length ≤ 10 ∧
    constructible A B C D constructions :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_sqrt_2021_l1234_123425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_a_theorem_l1234_123422

-- Define the propositions p and q
def p (x a : ℝ) : Prop := |x - a| > 3
def q (x : ℝ) : Prop := (x + 1) * (2 * x - 1) ≥ 0

-- Define the range of a
def range_of_a : Set ℝ := Set.Iic (-4) ∪ Set.Ici (7/2)

-- State the theorem
theorem range_a_theorem : 
  (∀ x a : ℝ, (¬(p x a) → q x) ∧ ∃ x : ℝ, q x ∧ p x a) → 
  ∀ a : ℝ, a ∈ range_of_a :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_a_theorem_l1234_123422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l1234_123467

/-- The line l is defined by the equation (1+2m)x-(m+1)y-4m-3=0, where m is a real number -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  (1 + 2*m)*x - (m + 1)*y - 4*m - 3 = 0

/-- Point P is defined as (-5, 0) -/
def point_P : ℝ × ℝ := (-5, 0)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem stating that the maximum distance from P to any line l is 2√10 -/
theorem max_distance_to_line :
  ∀ m : ℝ, ∀ p : ℝ × ℝ, line_l m p.1 p.2 →
    distance point_P p ≤ 2 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l1234_123467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_decomposition_l1234_123412

/-- Definition of set A: numbers with even-indexed digits set to zero -/
def A : Set ℕ := {n | ∀ i : ℕ, i % 2 = 0 → (n / 10^i) % 10 = 0}

/-- Definition of set B: numbers with odd-indexed digits set to zero -/
def B : Set ℕ := {n | ∀ i : ℕ, i % 2 = 1 → (n / 10^i) % 10 = 0}

/-- Theorem stating the existence of unique decomposition for all natural numbers -/
theorem unique_decomposition :
  (Set.Infinite A) ∧ 
  (Set.Infinite B) ∧ 
  (∀ n : ℕ, ∃! (a b : ℕ), a ∈ A ∧ b ∈ B ∧ n = a + b) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_decomposition_l1234_123412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_formula_l1234_123404

def mySequence (n : ℕ) : ℤ :=
  match n with
  | 0 => 1
  | 1 => 2
  | n + 2 => 3 * mySequence (n + 1) - 2 * mySequence n + 1

theorem mySequence_formula (n : ℕ) : mySequence n = 2^n - n := by
  sorry

#eval mySequence 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_formula_l1234_123404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_term_125_l1234_123469

def sequence_a (n : ℕ) : ℤ := 2 * n^2 - 3

theorem sequence_term_125 :
  ∃ n : ℕ, sequence_a n = 125 ∧ n = 8 :=
by
  use 8
  constructor
  · -- Prove sequence_a 8 = 125
    rw [sequence_a]
    norm_num
  · -- Prove n = 8
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_term_125_l1234_123469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_balls_count_l1234_123468

theorem red_balls_count (total_balls : ℕ) (prob_red : ℚ) (h_total : total_balls = 50) (h_prob : prob_red = 7/10) :
  Int.floor (total_balls * prob_red) = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_balls_count_l1234_123468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scheherazade_nights_l1234_123493

-- Define the Circle type
structure Circle :=
  (points : ℕ)

-- Define the cut operation
def cut (c : Circle) : Circle :=
  ⟨c.points - 1⟩

-- Define the property of being able to make a valid cut
def can_make_valid_cut (c : Circle) : Prop :=
  c.points > 3

-- Define the iterate function
def iterate (f : α → α) : ℕ → α → α
  | 0, x => x
  | n + 1, x => iterate f n (f x)

-- Theorem statement
theorem scheherazade_nights (initial_circle : Circle) :
  initial_circle.points = 1001 →
  ¬(can_make_valid_cut (iterate cut 1998 initial_circle)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scheherazade_nights_l1234_123493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_minimum_l1234_123409

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.log ((x - 1) / 32)

-- Define the integral as a function of a
noncomputable def I (a : ℝ) : ℝ := ∫ x in a..a^2, f x

-- State the theorem
theorem integral_minimum (a : ℝ) (h : a > 1) :
  ∀ b > 1, I a ≤ I b ↔ a = 3 := by
  sorry

#check integral_minimum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_minimum_l1234_123409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l1234_123405

-- Define the triangle
noncomputable def triangle_side_1 : ℝ := 3
noncomputable def triangle_side_2 : ℝ := 3
noncomputable def triangle_side_3 : ℝ := Real.sqrt 15 - Real.sqrt 3

-- Theorem statement
theorem triangle_angles :
  let angle1 := Real.arccos (Real.sqrt 5 / 3)
  let angle2 := (Real.pi - angle1) / 2
  let angle3 := angle2
  (Real.sin angle1 / triangle_side_1 = Real.sin angle2 / triangle_side_2) ∧
  (Real.sin angle2 / triangle_side_2 = Real.sin angle3 / triangle_side_3) ∧
  (angle1 + angle2 + angle3 = Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l1234_123405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_cone_configuration_l1234_123441

/-- Represents a cone with base radius and apex angle -/
structure Cone where
  baseRadius : ℝ
  apexAngle : ℝ

/-- Represents the configuration of cones and sphere -/
structure ConeSphereProblem where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  sphereTouchesCones : Bool
  sphereCenterEquidistant : Bool

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Function to check if a sphere touches the cones -/
def Sphere.touchesCones (s : Sphere) (c1 c2 c3 : Cone) : Prop := sorry

/-- Function to check if the sphere's center is equidistant from cone bases -/
def Sphere.centerEquidistantFromConeBases (s : Sphere) (c1 c2 c3 : Cone) : Prop := sorry

/-- The main theorem stating the radius of the sphere -/
theorem sphere_radius_in_cone_configuration 
  (problem : ConeSphereProblem)
  (h1 : problem.cone1 = ⟨32, π/3⟩)
  (h2 : problem.cone2 = ⟨48, 2*π/3⟩)
  (h3 : problem.cone3 = ⟨48, 2*π/3⟩)
  (h4 : problem.sphereTouchesCones = true)
  (h5 : problem.sphereCenterEquidistant = true) :
  ∃ (r : ℝ), r = 13*(Real.sqrt 3 + 1) ∧ 
  (∃ (sphere : Sphere), sphere.radius = r ∧ 
    sphere.touchesCones problem.cone1 problem.cone2 problem.cone3 ∧
    sphere.centerEquidistantFromConeBases problem.cone1 problem.cone2 problem.cone3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_cone_configuration_l1234_123441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_ones_correct_probability_two_ones_rounded_correct_l1234_123478

/-- The probability of exactly two dice showing a 1 when 12 standard 6-sided dice are rolled -/
def probability_two_ones : ℚ :=
  (11 * 5^10) / 6^11

/-- Theorem stating that the probability of exactly two dice showing a 1
    when 12 standard 6-sided dice are rolled is equal to (11 * 5^10) / (6^11) -/
theorem probability_two_ones_correct :
  probability_two_ones = (11 * 5^10) / 6^11 := by
  rfl

/-- The probability rounded to the nearest thousandth -/
def probability_two_ones_rounded : ℚ :=
  ((probability_two_ones * 1000).floor + 
   if (probability_two_ones * 1000 - (probability_two_ones * 1000).floor) ≥ 1/2 then 1 else 0) / 1000

/-- Theorem stating that the probability rounded to the nearest thousandth is 0.298 -/
theorem probability_two_ones_rounded_correct :
  probability_two_ones_rounded = 298 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_ones_correct_probability_two_ones_rounded_correct_l1234_123478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_top_sphere_height_l1234_123460

/-- The radius of each sphere -/
noncomputable def r : ℝ := 22 - 11 * Real.sqrt 2

/-- The side length of the square formed by the centers of the four base spheres -/
noncomputable def side_length : ℝ := 2 * r

/-- The height of the pyramid formed by the centers of the five spheres -/
noncomputable def pyramid_height : ℝ := side_length

/-- The theorem stating that the distance from the highest point of the top sphere to the plane is 22 -/
theorem top_sphere_height : 
  pyramid_height + 2 * r = 22 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_top_sphere_height_l1234_123460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_secant_equality_l1234_123400

/-- Plane type -/
def Plane : Type := ℝ × ℝ

/-- Point in a plane -/
def Point (Plane : Type) := Plane

/-- Circle in a plane -/
structure Circle (Plane : Type) where
  center : Point Plane
  radius : ℝ

/-- Line in a plane -/
structure Line (Plane : Type)

/-- Distance between two points -/
def Point.dist (p q : Point Plane) : ℝ := sorry

/-- Intersection of two circles -/
def Circle.intersectsWith (c₁ c₂ : Circle Plane) (p : Point Plane) : Prop :=
  p.dist c₁.center = c₁.radius ∧ p.dist c₂.center = c₂.radius

/-- Tangent line to a circle at a point -/
def Circle.tangentAt (c : Circle Plane) (p : Point Plane) : Line Plane := sorry

/-- Point belongs to a line -/
def Point.on (p : Point Plane) (l : Line Plane) : Prop := sorry

/-- Intersection of a line with a circle -/
def Line.intersectsWith (l : Line Plane) (c : Circle Plane) (p : Point Plane) : Prop :=
  p.on l ∧ p.dist c.center = c.radius

/-- Given two circles k₁ and k₂ that intersect at points A and B, 
    with tangents as described, prove that the product of the square 
    of one tangent segment and the other tangent segment equals the 
    product of the square of the other tangent segment and the first 
    tangent segment. -/
theorem tangent_secant_equality 
  (k₁ k₂ : Circle Plane) 
  (A B C D : Point Plane) : 
  (k₁.intersectsWith k₂ A) → 
  (k₁.intersectsWith k₂ B) → 
  (k₁.tangentAt A).intersectsWith k₂ C → 
  (k₂.tangentAt B).intersectsWith k₁ D → 
  (B.dist D)^2 * (B.dist C) = (A.dist C)^2 * (A.dist D) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_secant_equality_l1234_123400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_l1234_123420

/-- The polynomial we're working with -/
def p (x : ℝ) : ℝ := x^4 - 3*x^3 + x^2 + 3*x - 2

/-- The set of real roots of the polynomial -/
def roots : Set ℝ := {1, 2, -1}

/-- Theorem stating that the roots of the polynomial are correct -/
theorem polynomial_roots :
  ∀ x : ℝ, x ∈ roots ↔ p x = 0 ∧ (x = 1 → ∃ ε > 0, ∀ y, |y - x| < ε → y ≠ x → p y ≠ 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_l1234_123420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_range_l1234_123407

-- Define the line equation
def line_equation (x y : ℝ) (α : ℝ) : Prop :=
  x * Real.sin α + y + 2 = 0

-- Define the inclination angle of a line
noncomputable def inclination_angle (k : ℝ) : ℝ := Real.arctan k

-- Define the range of inclination angle
def inclination_angle_range (α : ℝ) : Prop :=
  α ∈ Set.Icc 0 (Real.pi/4) ∪ Set.Ico (3*Real.pi/4) Real.pi

-- Theorem statement
theorem line_inclination_range :
  ∀ α : ℝ, (∃ x y : ℝ, line_equation x y α) →
  inclination_angle_range (inclination_angle (-Real.sin α)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_range_l1234_123407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_ratio_l1234_123448

/-- Given a triangle ABC with points D on AC and P on BD, prove that if CD = DA and 
    AP = λ * AB + (1/6) * AC, then λ = 2/3 -/
theorem triangle_vector_ratio (A B C D P : EuclideanSpace ℝ (Fin 2)) (lambda : ℝ) : 
  (D - A : EuclideanSpace ℝ (Fin 2)) = (C - D) →  -- CD = DA
  (P - A : EuclideanSpace ℝ (Fin 2)) = lambda • (B - A) + (1/6) • (C - A) →  -- AP = λ * AB + (1/6) * AC
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • B + t • D) →  -- P is on segment BD
  lambda = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_ratio_l1234_123448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1234_123417

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sin x) + Real.sqrt (49 - x^2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | x ∈ Set.Ioo (-2*π) (-π) ∪ Set.Ioo 0 π ∪ Set.Ioc (2*π) 7} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1234_123417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1234_123485

/-- The distance between the foci of an ellipse -/
noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

/-- Theorem about the distance between foci and eccentricity of a specific ellipse -/
theorem ellipse_properties :
  let a := Real.sqrt 2
  let b := 3 / 2 * Real.sqrt (2/3)
  distance_between_foci a b = Real.sqrt 14 ∧
  eccentricity a b = Real.sqrt 7 / 4 := by
  sorry

#check ellipse_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1234_123485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_properties_l1234_123419

noncomputable def f (x : ℝ) := Real.sin x + Real.cos x
noncomputable def g (x : ℝ) := 2 * Real.sqrt 2 * Real.sin x * Real.cos x

theorem functions_properties :
  (∀ x y, -π/4 < x ∧ x < y ∧ y < π/4 → f x < f y ∧ g x < g y) ∧
  (∃ m : ℝ, (∀ x, f x ≤ m) ∧ (∀ x, g x ≤ m) ∧ (∃ x₁ x₂, f x₁ = m ∧ g x₂ = m)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_properties_l1234_123419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_participant_partition_l1234_123401

/-- A graph representing participants in the competition -/
structure ParticipantGraph where
  /-- The set of vertices (participants) -/
  V : Type
  /-- The number of vertices -/
  num_vertices : Nat
  /-- The set of edges (knowledge between participants) -/
  E : V → V → Prop
  /-- The pairing function (representing countries) -/
  country : V → V
  /-- There are 200 participants -/
  vertex_count : num_vertices = 200
  /-- Each participant is paired with exactly one other (compatriot) -/
  compatriot_unique : ∀ v : V, country (country v) = v ∧ country v ≠ v
  /-- Each participant knows their compatriot -/
  knows_compatriot : ∀ v : V, E v (country v)
  /-- Each participant knows exactly one other participant besides their compatriot -/
  knows_one_other : ∀ v : V, ∃! w : V, E v w ∧ w ≠ country v
  /-- Knowledge is mutual -/
  mutual_knowledge : ∀ v w : V, E v w → E w v

/-- The main theorem stating that the participants can be divided into two groups -/
theorem participant_partition (G : ParticipantGraph) :
  ∃ (A B : Set G.V), A ∪ B = Set.univ ∧ A ∩ B = ∅ ∧
  (∀ v w, v ∈ A → w ∈ A → ¬G.E v w ∧ G.country v ≠ w) ∧
  (∀ v w, v ∈ B → w ∈ B → ¬G.E v w ∧ G.country v ≠ w) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_participant_partition_l1234_123401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1234_123454

def sequenceA (n : ℕ) : ℤ :=
  match n with
  | 1 => -1
  | 2 => 4
  | 3 => -16
  | 4 => 64
  | 5 => -256
  | _ => 0  -- placeholder for other terms

theorem sequence_general_term (n : ℕ) (h : n > 0) : 
  sequenceA n = -(-4 : ℤ)^(n-1) := by
  sorry

#eval sequenceA 1
#eval sequenceA 2
#eval sequenceA 3
#eval sequenceA 4
#eval sequenceA 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1234_123454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_conditions_l1234_123457

-- Define the function f as noncomputable
noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

-- State the theorem
theorem odd_function_conditions (a b : ℝ) :
  (∀ x, x ≠ 1 → f a b (-x) = -(f a b x)) → a = -1/2 ∧ b = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_conditions_l1234_123457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_power_equals_128y_l1234_123470

theorem product_power_equals_128y (a b : ℤ) (n : ℕ) (h : (a * b) ^ n = 128 * 8) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_power_equals_128y_l1234_123470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_triangles_in_regular_ngon_l1234_123481

/-- The number of distinct triangles in a regular n-gon -/
def num_triangles (n : ℕ) : ℕ :=
  Int.toNat (round ((n^2 : ℚ) / 12))

/-- Theorem stating that the number of distinct triangles in a regular n-gon
    is equal to the integer nearest to n^2 / 12 -/
theorem num_triangles_in_regular_ngon (n : ℕ) :
  num_triangles n = Int.toNat (round ((n^2 : ℚ) / 12)) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_triangles_in_regular_ngon_l1234_123481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l1234_123455

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 + (Real.log x) / (Real.log 3)

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := (f x)^2 + f (x^2)

-- State the theorem
theorem max_value_of_y :
  ∃ (x : ℝ), x ∈ Set.Icc 1 9 ∧ 
  ∀ (z : ℝ), z ∈ Set.Icc 1 9 → y z ≤ y x ∧
  y x = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l1234_123455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1234_123418

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area : 
  let AB : ℝ × ℝ := (Real.cos (32 * π / 180), Real.cos (58 * π / 180))
  let BC : ℝ × ℝ := (Real.sin (60 * π / 180) * Real.sin (118 * π / 180), 
                     Real.sin (120 * π / 180) * Real.sin (208 * π / 180))
  area_triangle (0, 0) AB (AB.1 + BC.1, AB.2 + BC.2) = 3/8
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1234_123418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_line_l1234_123474

def S : Set ℂ := {z : ℂ | ∃ (r : ℝ), (1 + 2*Complex.I)*z = r}

theorem S_is_line : ∃ (a b : ℝ), S = {z : ℂ | z.im = a * z.re + b} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_line_l1234_123474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_law_of_cosines_l1234_123472

/-- The law of cosines for a triangle --/
theorem law_of_cosines (a b c γ : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : c < a + b ∧ a < b + c ∧ b < c + a)
  (h_angle : 0 ≤ γ ∧ γ ≤ Real.pi)
  (h_opposite : γ = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) :
  c^2 = a^2 + b^2 - 2*a*b*Real.cos γ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_law_of_cosines_l1234_123472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_l1234_123429

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the inverse function of f
def f_inv : ℝ → ℝ := sorry

-- Axiom: The inverse function of y = f(x+1) is y = f^(-1)(x+1)
axiom inverse_relation (x : ℝ) : f (f_inv (x + 1)) = x + 1

-- Given: f(1) = 3997
axiom f_1 : f 1 = 3997

-- Theorem to prove
theorem f_2012 : f 2012 = 1986 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_l1234_123429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_product_l1234_123444

theorem power_of_three_product : (3^12 * 3^18 : ℕ) = 243^6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_product_l1234_123444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l1234_123416

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  coords : ℝ × ℝ

-- Helper functions (declared but not implemented)
def are_externally_tangent (c₁ c₂ : Circle) : Prop := sorry
def point_on_circle (p : Point) (c : Circle) : Prop := sorry
def distance (p₁ p₂ : Point) : ℝ := sorry
def is_tangent_line (p₁ p₂ : Point) (c : Circle) : Prop := sorry
def triangle_area (p₁ p₂ p₃ : Point) : ℝ := sorry

-- Define the problem setup
def problem_setup (ω₁ ω₂ ω₃ : Circle) (P₁ P₂ P₃ : Point) : Prop :=
  ω₁.radius = 3 ∧ ω₂.radius = 3 ∧ ω₃.radius = 3 ∧
  (are_externally_tangent ω₁ ω₂) ∧ 
  (are_externally_tangent ω₂ ω₃) ∧ 
  (are_externally_tangent ω₃ ω₁) ∧
  (point_on_circle P₁ ω₁) ∧ 
  (point_on_circle P₂ ω₂) ∧ 
  (point_on_circle P₃ ω₃) ∧
  (distance P₁ P₂ = distance P₂ P₃) ∧ 
  (distance P₂ P₃ = distance P₃ P₁) ∧
  (is_tangent_line P₁ P₂ ω₁) ∧ 
  (is_tangent_line P₂ P₃ ω₂) ∧ 
  (is_tangent_line P₃ P₁ ω₃)

-- Define the theorem
theorem triangle_area_theorem 
  (ω₁ ω₂ ω₃ : Circle) (P₁ P₂ P₃ : Point) 
  (h : problem_setup ω₁ ω₂ ω₃ P₁ P₂ P₃) : 
  ∃ (a b : ℕ), 
    triangle_area P₁ P₂ P₃ = Real.sqrt a + Real.sqrt b ∧ 
    a + b = 597 ∧ 
    a = 507 ∧ 
    b = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l1234_123416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_proposition_l1234_123499

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, (2 : ℝ)^x ≥ 0) ↔ (∀ x : ℝ, (2 : ℝ)^x < 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_proposition_l1234_123499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gaussian_guardians_total_points_l1234_123436

/-- Represents a player in the Gaussian Guardians team -/
inductive Player : Type
  | Daniel
  | Curtis
  | Sid
  | Emily
  | Kalyn
  | Hyojeong
  | Ty
  | Winston

/-- Returns the number of points scored by a given player -/
def points (p : Player) : ℕ :=
  match p with
  | Player.Daniel => 7
  | Player.Curtis => 8
  | Player.Sid => 2
  | Player.Emily => 11
  | Player.Kalyn => 6
  | Player.Hyojeong => 12
  | Player.Ty => 1
  | Player.Winston => 7

/-- The theorem stating that the total points scored by the Gaussian Guardians is 54 -/
theorem gaussian_guardians_total_points :
  points Player.Daniel + points Player.Curtis + points Player.Sid +
  points Player.Emily + points Player.Kalyn + points Player.Hyojeong +
  points Player.Ty + points Player.Winston = 54 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gaussian_guardians_total_points_l1234_123436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1234_123449

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (Nat.choose 6 3) * a^3 * b^3 = 5/2) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ (Nat.choose 6 3) * x^3 * y^3 = 5/2 → a + 2*b ≤ x + 2*y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1234_123449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_town_x_employment_l1234_123445

/-- Represents the employment data for town X -/
structure TownEmployment where
  total_employed : ℚ
  employed_males : ℚ
  male_tech : ℚ
  male_health : ℚ
  female_education : ℚ

/-- The conditions of the problem -/
def town_x : TownEmployment where
  total_employed := 64/100
  employed_males := 55/100
  male_tech := 30/100
  male_health := 40/100
  female_education := 60/100

/-- Calculate the percentage of employed people who are females -/
def female_employment_percentage (t : TownEmployment) : ℚ :=
  ((t.total_employed - t.employed_males) / t.total_employed) * 100

/-- Check if the majority of employed females work in Education -/
def majority_females_in_education (t : TownEmployment) : Prop :=
  t.female_education > 1/2

/-- The main theorem to prove -/
theorem town_x_employment :
  female_employment_percentage town_x = 14.0625 ∧
  majority_females_in_education town_x := by
  sorry

#eval female_employment_percentage town_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_town_x_employment_l1234_123445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scott_running_distance_l1234_123442

/-- The distance Scott runs every Monday through Wednesday -/
def daily_distance : ℝ := sorry

/-- The total distance Scott runs in a week -/
def weekly_distance : ℝ := 3 * daily_distance + 2 * (2 * daily_distance)

/-- The total distance Scott runs in a month (4 weeks) -/
def monthly_distance : ℝ := 4 * weekly_distance

theorem scott_running_distance : daily_distance = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scott_running_distance_l1234_123442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_sequence_57th_term_l1234_123482

/-- Represents the sequence of red numbers according to the given rules -/
def redSequence : ℕ → ℕ := sorry

/-- The number of terms in the nth group of the sequence -/
def groupSize (n : ℕ) : ℕ := n + 1

/-- Indicates whether the nth group consists of even or odd numbers -/
def isEvenGroup (n : ℕ) : Bool := n % 2 = 1

/-- The first term of the nth group in the sequence -/
def groupStart (n : ℕ) : ℕ := sorry

/-- The 57th term of the redSequence is 103 -/
theorem red_sequence_57th_term : redSequence 57 = 103 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_sequence_57th_term_l1234_123482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_is_hyperbola_l1234_123446

/-- Represents a conic section equation in the form ax^2 + by^2 + cx + dy + e = 0 -/
structure ConicSection where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- Determines if a conic section is a hyperbola -/
def is_hyperbola (conic : ConicSection) : Prop :=
  ∃ h k a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
    ∀ x y : ℝ, conic.a * (x - h)^2 + conic.b * (y - k)^2 = a * b

theorem equation_is_hyperbola :
  let conic : ConicSection := ⟨9, -16, -18, 32, 64⟩
  is_hyperbola conic :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_is_hyperbola_l1234_123446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_l1234_123480

theorem triangle_side_ratio (A B C : ℝ) (a b c : ℝ) :
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A + B + C = π ∧
  A = B ∧ C = 4 * A ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  (a : ℝ) / 1 = (b : ℝ) / 1 ∧ (c : ℝ) / (b : ℝ) = Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_l1234_123480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_l1234_123431

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 + Real.log x - a * x

-- State the theorem
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x > 0, Monotone (f a)) → a ∈ Set.Iic 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_l1234_123431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hidden_sum_is_24_l1234_123439

/-- Represents a regular six-sided die -/
structure RegularDie where
  faces : Fin 6 → Nat
  opposite_sum : ∀ (i : Fin 6), faces i + faces (5 - i) = 7

/-- Represents the stack of four dice -/
structure DiceStack where
  dice : Fin 4 → RegularDie
  bottom_visible : (dice 0).faces 0 = 2 ∧ (dice 0).faces 1 = 4
  top_visible : (dice 3).faces 5 = 3

/-- The sum of hidden dots between the dice -/
def hidden_sum (stack : DiceStack) : Nat :=
  (stack.dice 0).faces 5 +
  (stack.dice 1).faces 0 + (stack.dice 1).faces 5 +
  (stack.dice 2).faces 0 + (stack.dice 2).faces 5 +
  (stack.dice 3).faces 0

/-- Theorem stating that the sum of hidden dots is 24 -/
theorem hidden_sum_is_24 (stack : DiceStack) : hidden_sum stack = 24 := by
  sorry

#check hidden_sum_is_24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hidden_sum_is_24_l1234_123439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_hyperbola_l1234_123498

/-- The x-coordinate of a point on the curve as a function of t -/
noncomputable def x (t : ℝ) : ℝ := 4 * Real.exp t + 4 * Real.exp (-t)

/-- The y-coordinate of a point on the curve as a function of t -/
noncomputable def y (t : ℝ) : ℝ := 5 * (Real.exp t - Real.exp (-t))

/-- Constants for the standard form of a hyperbola equation -/
def a : ℝ := 8
def b : ℝ := 5

/-- Theorem stating that the points (x(t), y(t)) lie on a hyperbola -/
theorem points_on_hyperbola :
  ∀ t : ℝ, (x t)^2 / a^2 - (y t)^2 / b^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_hyperbola_l1234_123498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_inverse_x_expression_impossibility_without_f₁_impossibility_without_f₂_impossibility_without_f₃_l1234_123413

-- Define the three functions
noncomputable def f₁ (x : ℝ) : ℝ := x + 1/x
noncomputable def f₂ (x : ℝ) : ℝ := x^2
noncomputable def f₃ (x : ℝ) : ℝ := (x - 1)^2

-- Define a type for the set of allowed operations
inductive AllowedOp
| Add : AllowedOp → AllowedOp → AllowedOp
| Sub : AllowedOp → AllowedOp → AllowedOp
| Mul : AllowedOp → AllowedOp → AllowedOp
| Pow : AllowedOp → ℕ → AllowedOp
| ScalarMul : ℝ → AllowedOp → AllowedOp
| ScalarAdd : ℝ → AllowedOp → AllowedOp
| F₁ : AllowedOp
| F₂ : AllowedOp
| F₃ : AllowedOp

-- Define a function to evaluate an AllowedOp
noncomputable def evaluate : AllowedOp → ℝ → ℝ
| AllowedOp.Add a b, x => evaluate a x + evaluate b x
| AllowedOp.Sub a b, x => evaluate a x - evaluate b x
| AllowedOp.Mul a b, x => evaluate a x * evaluate b x
| AllowedOp.Pow a n, x => (evaluate a x) ^ n
| AllowedOp.ScalarMul c a, x => c * evaluate a x
| AllowedOp.ScalarAdd c a, x => c + evaluate a x
| AllowedOp.F₁, x => f₁ x
| AllowedOp.F₂, x => f₂ x
| AllowedOp.F₃, x => f₃ x

-- Theorem statement
theorem exists_inverse_x_expression :
  ∃ (op : AllowedOp), ∀ (x : ℝ), x ≠ 0 → evaluate op x = 1/x := by
  sorry

-- Helper function to check if an AllowedOp contains a specific operation
def containsOp (op : AllowedOp) (target : AllowedOp) : Prop :=
  match op with
  | AllowedOp.Add a b => containsOp a target ∨ containsOp b target
  | AllowedOp.Sub a b => containsOp a target ∨ containsOp b target
  | AllowedOp.Mul a b => containsOp a target ∨ containsOp b target
  | AllowedOp.Pow a _ => containsOp a target
  | AllowedOp.ScalarMul _ a => containsOp a target
  | AllowedOp.ScalarAdd _ a => containsOp a target
  | _ => op = target

-- Theorem statements for impossibility when one function is removed
theorem impossibility_without_f₁ :
  ¬∃ (op : AllowedOp), (∀ (x : ℝ), x ≠ 0 → evaluate op x = 1/x) ∧
  ¬(containsOp op AllowedOp.F₁) := by
  sorry

theorem impossibility_without_f₂ :
  ¬∃ (op : AllowedOp), (∀ (x : ℝ), x ≠ 0 → evaluate op x = 1/x) ∧
  ¬(containsOp op AllowedOp.F₂) := by
  sorry

theorem impossibility_without_f₃ :
  ¬∃ (op : AllowedOp), (∀ (x : ℝ), x ≠ 0 → evaluate op x = 1/x) ∧
  ¬(containsOp op AllowedOp.F₃) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_inverse_x_expression_impossibility_without_f₁_impossibility_without_f₂_impossibility_without_f₃_l1234_123413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_amount_l1234_123402

/-- The initial amount of water in a glass, given evaporation conditions -/
theorem initial_water_amount (daily_evaporation : ℚ) (days : ℕ) (evaporation_percentage : ℚ) 
  (h1 : daily_evaporation = 4/100)
  (h2 : days = 10)
  (h3 : evaporation_percentage = 8/5) :
  (daily_evaporation * days) / (evaporation_percentage / 100) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_amount_l1234_123402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_hyperbola_equation_l1234_123403

/-- An equilateral hyperbola with foci at a distance √2 from the asymptotes -/
def EquilateralHyperbola (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) ∧
  a^2 = b^2 ∧
  (let c := Real.sqrt (a^2 + b^2)
   let focus_distance := Real.sqrt 2
   c / Real.sqrt 2 = focus_distance)

/-- The equation of the hyperbola is x^2 - y^2 = 2 -/
theorem equilateral_hyperbola_equation (a b : ℝ) (h : EquilateralHyperbola a b) :
  ∀ x y : ℝ, x^2 - y^2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_hyperbola_equation_l1234_123403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l1234_123434

/-- The number of people in the room -/
def num_people : ℕ := 52

/-- The number of months in a year -/
def num_months : ℕ := 12

/-- The largest value of n such that at least n people always share a birthday month -/
noncomputable def largest_shared_birthday (p : ℕ) (m : ℕ) : ℕ :=
  Int.toNat ⌈(p : ℚ) / m⌉

theorem birthday_problem :
  largest_shared_birthday num_people num_months = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l1234_123434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_identity_l1234_123488

theorem cube_root_identity (t : ℝ) : t = 1 / (1 - (2 : ℝ)^(1/3)) → t = (1 + (2 : ℝ)^(1/3)) * (1 + (4 : ℝ)^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_identity_l1234_123488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_implies_m_equals_one_l1234_123471

/-- The quadratic function representing the left side of the inequality -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2*x + m*x

/-- The condition that the solution set is (0, 2) -/
def solution_set (m : ℝ) : Prop :=
  ∀ x : ℝ, f m x < 0 ↔ 0 < x ∧ x < 2

theorem inequality_solution_implies_m_equals_one :
  ∀ m : ℝ, solution_set m → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_implies_m_equals_one_l1234_123471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_line_l1234_123424

-- Define the points
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 6)
def C : ℝ × ℝ := (-1, 3)

-- Define a line passing through point C with slope k
noncomputable def line (k : ℝ) : ℝ → ℝ := λ x ↦ k * (x + 1) + 3

-- Define the distance from a point to the line
noncomputable def distance_to_line (p : ℝ × ℝ) (k : ℝ) : ℝ :=
  |k * p.1 - p.2 + 3 + k| / Real.sqrt (k^2 + 1)

-- Theorem statement
theorem equidistant_line :
  ∃ k₁ k₂ : ℝ, k₁ ≠ k₂ ∧
  distance_to_line A k₁ = distance_to_line B k₁ ∧
  distance_to_line A k₂ = distance_to_line B k₂ ∧
  (∀ x y : ℝ, y = line k₁ x ↔ 2*x - y + 5 = 0) ∧
  (∀ x y : ℝ, y = line k₂ x ↔ x - 3*y + 10 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_line_l1234_123424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1234_123466

/-- A parabola is defined by its coefficients a, b, and c in the form ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
noncomputable def vertex (p : Parabola) : ℝ × ℝ :=
  (- p.b / (2 * p.a), - (p.b^2 - 4 * p.a * p.c) / (4 * p.a))

/-- Check if a point lies on the parabola -/
def lies_on (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- The axis of symmetry is vertical if the coefficient of x^2 is non-zero -/
def vertical_axis_of_symmetry (p : Parabola) : Prop :=
  p.a ≠ 0

theorem parabola_properties (p : Parabola) :
  p.a = 3 ∧ p.b = -18 ∧ p.c = 25 →
  vertex p = (3, -2) ∧
  vertical_axis_of_symmetry p ∧
  lies_on p 4 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1234_123466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_a_l1234_123427

theorem sum_of_valid_a : ∃ S : Finset ℤ, 
  (∀ a ∈ S, (∀ x : ℝ, (((5*x - a) / 3 - x < 3) ∧ (3*x < 2*x + 1)) ↔ x < 1) ∧ 
  (∃ y : ℕ, y > 1 ∧ (3*y + a) / (y - 1) - 1 = (2*a) / (1 - y))) ∧
  S.sum id = -15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_a_l1234_123427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wave_largeness_l1234_123406

/-- A graph with vertices of type α -/
structure Graph (α : Type) where
  vertices : Set α
  edges : Set (α × α)

/-- A wave in a graph from set A to set B -/
structure Wave (G : Graph α) (A B : Set α) where
  paths : Set (List α)
  end_vertices : Set α

/-- A wave is proper if it's not empty -/
def proper_wave {α : Type} (G : Graph α) (A B : Set α) (w : Wave G A B) : Prop :=
  w.paths.Nonempty

/-- A wave is large if it includes all vertices in B -/
def large_wave {α : Type} (G : Graph α) (A B : Set α) (w : Wave G A B) : Prop :=
  w.end_vertices = B

/-- The graph G with vertex x removed -/
def remove_vertex {α : Type} (G : Graph α) (x : α) : Graph α :=
  { vertices := G.vertices \ {x},
    edges := { e ∈ G.edges | e.1 ≠ x ∧ e.2 ≠ x } }

theorem wave_largeness
  {α : Type} (G : Graph α) (A B : Set α) (x : α)
  (h1 : x ∈ G.vertices \ A)
  (h2 : ¬ ∃ w : Wave G A B, proper_wave G A B w)
  (h3 : ∃ w : Wave (remove_vertex G x) A B, proper_wave (remove_vertex G x) A B w) :
  ∀ w : Wave (remove_vertex G x) A B, large_wave (remove_vertex G x) A B w :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wave_largeness_l1234_123406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_approx_l1234_123496

/-- The molar mass of aluminum in g/mol -/
noncomputable def molar_mass_Al : ℝ := 26.98

/-- The molar mass of oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The number of aluminum atoms in Al2O3 -/
def num_Al : ℕ := 2

/-- The number of oxygen atoms in Al2O3 -/
def num_O : ℕ := 3

/-- The molar mass of Al2O3 in g/mol -/
noncomputable def molar_mass_Al2O3 : ℝ := num_Al * molar_mass_Al + num_O * molar_mass_O

/-- The mass percentage of oxygen in Al2O3 -/
noncomputable def mass_percentage_O : ℝ := (num_O * molar_mass_O) / molar_mass_Al2O3 * 100

/-- Theorem stating that the mass percentage of oxygen in Al2O3 is approximately 47.07% -/
theorem mass_percentage_O_approx :
  abs (mass_percentage_O - 47.07) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_approx_l1234_123496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1234_123491

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

theorem hyperbola_eccentricity 
    (h : Hyperbola) 
    (F1 F2 A B : Point)
    (h_foci : F1.x < 0 ∧ F2.x > 0 ∧ F1.y = 0 ∧ F2.y = 0)
    (h_on_hyperbola : on_hyperbola h A ∧ on_hyperbola h B)
    (h_symmetric : A.x = -B.x ∧ A.y = -B.y)
    (h_ptolemy : distance A B * distance F1 F2 = 
                 distance A F1 * distance B F2 + distance A F2 * distance B F1)
    (h_angle : Real.cos (π/6) = (distance A F1)^2 + (distance A F2)^2 - (distance F1 F2)^2 
                                 / (2 * distance A F1 * distance A F2)) :
    eccentricity h = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1234_123491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_intervals_f_max_on_interval_f_min_on_interval_l1234_123435

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos (x + Real.pi/3) + Real.sqrt 3 / 2

-- Theorem for decreasing intervals
theorem f_decreasing_intervals (k : ℤ) :
  ∀ x ∈ Set.Icc (k * Real.pi + Real.pi/12) (k * Real.pi + 7*Real.pi/12),
  ∀ y ∈ Set.Icc (k * Real.pi + Real.pi/12) (k * Real.pi + 7*Real.pi/12),
  x ≤ y → f y ≤ f x := by
  sorry

-- Theorem for maximum value on [0, π/2]
theorem f_max_on_interval :
  ∃ x ∈ Set.Icc 0 (Real.pi/2), ∀ y ∈ Set.Icc 0 (Real.pi/2), f y ≤ f x ∧ f x = 1 := by
  sorry

-- Theorem for minimum value on [0, π/2]
theorem f_min_on_interval :
  ∃ x ∈ Set.Icc 0 (Real.pi/2), ∀ y ∈ Set.Icc 0 (Real.pi/2), f y ≥ f x ∧ f x = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_intervals_f_max_on_interval_f_min_on_interval_l1234_123435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_data_theorem_l1234_123456

/-- Definition of the first set of sample data -/
def first_set : Finset ℝ := sorry

/-- Definition of the second set of sample data -/
def second_set : Finset ℝ := sorry

/-- The average of the first set -/
noncomputable def m : ℝ := sorry

/-- The standard deviation of the first set -/
def std_dev_first : ℝ := 3

/-- The average of the second set -/
noncomputable def x_bar : ℝ := sorry

/-- The standard deviation of the second set -/
noncomputable def s : ℝ := sorry

theorem sample_data_theorem :
  (first_set.card = 8) →
  (second_set.card = 9) →
  (second_set = first_set ∪ {m}) →
  (Finset.sum first_set id = 8 * m) →
  (x_bar = m) ∧ (s < std_dev_first) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_data_theorem_l1234_123456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_l1234_123440

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- represents ax + by + c = 0

/-- The number of intersection points between a circle and a line --/
def circleLineIntersections (c : Circle) (l : Line) : ℕ :=
  sorry

/-- The number of intersection points between two lines --/
def lineLineIntersections (l1 l2 : Line) : ℕ :=
  sorry

/-- Three lines are distinct if no two of them are the same --/
def distinctLines (l1 l2 l3 : Line) : Prop :=
  l1 ≠ l2 ∧ l1 ≠ l3 ∧ l2 ≠ l3

theorem max_intersections (c : Circle) (l1 l2 l3 : Line) 
  (h : distinctLines l1 l2 l3) : 
  ∃ (n : ℕ), n ≤ 9 ∧ 
    ∀ (m : ℕ), m = circleLineIntersections c l1 + 
                   circleLineIntersections c l2 + 
                   circleLineIntersections c l3 + 
                   lineLineIntersections l1 l2 + 
                   lineLineIntersections l1 l3 + 
                   lineLineIntersections l2 l3 → 
    m ≤ n :=
  by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_l1234_123440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_boxes_l1234_123463

-- Define the function that calculates the number of ways
def number_of_ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ :=
  k^n

theorem balls_in_boxes (n k : ℕ) : 
  n = 5 → k = 4 → number_of_ways_to_put_balls_in_boxes n k = 1024 :=
by
  intros hn hk
  rw [number_of_ways_to_put_balls_in_boxes]
  rw [hn, hk]
  norm_num

#eval number_of_ways_to_put_balls_in_boxes 5 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_boxes_l1234_123463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1234_123492

noncomputable def f (x φ : ℝ) := 2 * Real.sin (2 * x + φ)

theorem function_properties
  (φ : ℝ)
  (h1 : 0 < φ ∧ φ < π)
  (h2 : ∀ x : ℝ, f (π/6 + x) φ = f (-x) φ) :
  (∃ S : Set ℝ, S = {x | ∃ k : ℤ, x = k * π + π/12} ∧
    ∀ x : ℝ, f x φ ≤ f (π/12) φ ∧ (x ∈ S ↔ f x φ = f (π/12) φ)) ∧
  (∃ m : ℝ, 11*π/24 ≤ m ∧ m < 17*π/24 ∧
    ∃! x₀ : ℝ, x₀ ∈ Set.Icc 0 m ∧ f x₀ φ = -Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1234_123492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_distance_l1234_123461

-- Define the equilateral triangle ABC
def triangle_side_length : ℝ := 500

-- Define the dihedral angle
def dihedral_angle : ℝ := 150

-- Define the point O and its distance d from A, B, C, P, and Q
def distance_d : ℝ := 353.55

-- Define points A, B, C, P, Q, and O
variable (A B C P Q O : EuclideanSpace ℝ (Fin 3))

-- State the conditions of the problem
axiom equilateral_triangle : 
  norm (A - B) = triangle_side_length ∧ 
  norm (B - C) = triangle_side_length ∧ 
  norm (C - A) = triangle_side_length

axiom P_equidistant : 
  norm (P - A) = norm (P - B) ∧ 
  norm (P - B) = norm (P - C)

axiom Q_equidistant : 
  norm (Q - A) = norm (Q - B) ∧ 
  norm (Q - B) = norm (Q - C)

-- For the dihedral angle, we'll use a placeholder axiom
axiom dihedral_angle_PAB_QAB : 
  ∃ (angle : ℝ), angle = dihedral_angle

axiom O_equidistant : 
  norm (O - A) = norm (O - B) ∧ 
  norm (O - B) = norm (O - C) ∧ 
  norm (O - C) = norm (O - P) ∧ 
  norm (O - P) = norm (O - Q)

-- State the theorem to be proved
theorem equidistant_point_distance : norm (O - A) = distance_d :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_distance_l1234_123461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_rotations_l1234_123489

/-- A configuration of 8 circular disks where 7 are fixed forming a regular hexagon -/
structure DiskConfiguration where
  r : ℝ  -- radius of each disk
  fixed_disks : Fin 7 → ℝ × ℝ  -- positions of the 7 fixed disks
  is_hexagon : ∀ i : Fin 6, 
    dist (fixed_disks i) (fixed_disks ((i + 1) % 6)) = 2 * r
  center_position : fixed_disks 6 = (0, 0)  -- center disk at origin

/-- The path of the 8th disk rolling around the fixed disks -/
noncomputable def rolling_path (config : DiskConfiguration) : ℝ → ℝ × ℝ := 
  sorry

/-- The angle of rotation of the 8th disk about its own center -/
noncomputable def rotation_angle (config : DiskConfiguration) : ℝ → ℝ :=
  sorry

theorem four_rotations (config : DiskConfiguration) :
  rotation_angle config (2 * Real.pi) = 8 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_rotations_l1234_123489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_53_l1234_123415

theorem sum_53 (A : Finset ℕ) (h1 : A.card = 53) (h2 : A.sum id ≤ 1990) :
  ∃ a b, a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ a + b = 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_53_l1234_123415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_five_l1234_123411

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then 3 * x + 4 else x^2 - 4 * x + 3

-- State the theorem
theorem g_of_negative_five : g (-5) = -11 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the if-then-else expression
  simp [show (-5 : ℝ) < 0 from by norm_num]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_five_l1234_123411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_win_probability_l1234_123458

/-- The probability of the first player winning in a turn-based game -/
noncomputable def first_player_win_prob (p1 p2 : ℝ) : ℝ :=
  p1 / (1 - (1 - p1) * (1 - p2))

/-- Theorem: In a game where the first player has 2/3 probability of success
    and the second player has 1/3 probability of success, 
    the probability of the first player winning is 3/4 -/
theorem game_win_probability :
  first_player_win_prob (2/3) (1/3) = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_win_probability_l1234_123458
