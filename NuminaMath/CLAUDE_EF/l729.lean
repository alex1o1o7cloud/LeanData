import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_walking_time_l729_72947

/-- The time taken for Martin to walk from his house to Lawrence's house -/
noncomputable def walking_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Proof that Martin's walking time is 6 hours -/
theorem martin_walking_time :
  let distance : ℝ := 12 -- miles
  let speed : ℝ := 2 -- miles per hour
  walking_time distance speed = 6 -- hours
:= by
  -- Unfold the definition of walking_time
  unfold walking_time
  -- Simplify the expression
  simp
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_walking_time_l729_72947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_B_catches_C_problem_solution_l729_72900

/-- Represents the time it takes for one marble to catch up with another -/
structure CatchUpTime where
  marble1 : Nat
  marble2 : Nat
  time : Nat

/-- Represents the circular track and marbles -/
structure CircularTrack where
  length : ℝ
  marbleA_speed : ℝ
  marbleB_speed : ℝ
  marbleC_speed : ℝ

/-- The main theorem stating the time for Marble B to catch up with Marble C -/
theorem marble_B_catches_C (track : CircularTrack) 
  (catchup1 : CatchUpTime) 
  (catchup2 : CatchUpTime) 
  (catchup3 : CatchUpTime) 
  (catchup4 : CatchUpTime) : ℝ :=
by
  sorry

#check marble_B_catches_C

/-- The specific instance of the problem -/
noncomputable def problem_instance : ℝ :=
  let track : CircularTrack := { 
    length := 1,  -- Arbitrary length, as it doesn't affect the result
    marbleA_speed := 1,  -- Arbitrary speed, as relative speeds matter
    marbleB_speed := 0.9,  -- Arbitrary, but slower than A
    marbleC_speed := 0.8  -- Arbitrary, but slower than B
  }
  let catchup1 : CatchUpTime := { marble1 := 1, marble2 := 2, time := 10 }
  let catchup2 : CatchUpTime := { marble1 := 1, marble2 := 3, time := 30 }
  let catchup3 : CatchUpTime := { marble1 := 1, marble2 := 2, time := 60 }
  let catchup4 : CatchUpTime := { marble1 := 1, marble2 := 3, time := 70 }
  marble_B_catches_C track catchup1 catchup2 catchup3 catchup4

-- This is a placeholder for the actual computation
def compute_result : ℝ := 110

theorem problem_solution : problem_instance = compute_result := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_B_catches_C_problem_solution_l729_72900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_medians_parallel_to_sides_l729_72903

-- Define the triangle type
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define a function to get the sides of a triangle
noncomputable def sides (t : Triangle) : Fin 3 → ℝ × ℝ := 
  λ i => t.vertices ((i + 1) % 3) - t.vertices i

-- Define a function to get the medians of a triangle
noncomputable def medians (t : Triangle) : Fin 3 → ℝ × ℝ := 
  λ i => (t.vertices ((i + 1) % 3) + t.vertices ((i + 2) % 3)) / 2 - t.vertices i

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = k • w ∨ w = k • v

theorem medians_parallel_to_sides (T T₁ : Triangle) 
  (h : ∀ (i : Fin 3), parallel (sides T i) (medians T₁ i)) :
  ∀ (i : Fin 3), parallel (medians T i) (sides T₁ i) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_medians_parallel_to_sides_l729_72903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l729_72941

/-- The circle with center (2, 2) and radius √2 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 2)^2 = 2}

/-- The line x - y - 4 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 - 4 = 0}

/-- The distance function from a point to the line -/
noncomputable def distToLine (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 - 4| / Real.sqrt 2

theorem max_distance_circle_to_line :
  (⨆ p ∈ Circle, distToLine p) = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l729_72941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l729_72995

theorem solve_exponential_equation :
  ∃ x : ℝ, (2 : ℝ)^x + 8 = 4 * (2 : ℝ)^x - 40 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l729_72995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l729_72943

theorem intersection_distance_difference (p q : ℕ) (hp : p > 0) (hq : q > 0) : 
  (∃ (C D : ℝ × ℝ), 
    (C.2 = 5 ∧ C.2 = 5 * C.1^2 + 2 * C.1 - 2) ∧ 
    (D.2 = 5 ∧ D.2 = 5 * D.1^2 + 2 * D.1 - 2) ∧
    (C ≠ D) ∧
    (Real.sqrt p / q : ℝ) = Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) ∧
    Nat.Coprime p q) →
  p - q = 476 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l729_72943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_removable_columns_l729_72929

/-- Represents a rectangular board with black and white squares -/
def Board (m n : ℕ) := Fin m → Fin n → Bool

/-- Predicate to check if all rows in a board are unique -/
def hasUniqueRows (b : Board m n) : Prop :=
  ∀ i j, i ≠ j → (∃ k, b i k ≠ b j k)

/-- Predicate to check if k columns can be removed while maintaining unique rows -/
def canRemoveColumns (b : Board m n) (k : ℕ) : Prop :=
  ∃ (cols : Fin k → Fin n), hasUniqueRows (λ i j ↦ b i j ∧ (∀ l, j ≠ cols l))

/-- The main theorem -/
theorem max_removable_columns
  (m n : ℕ) (h : m ≤ n) (b : Board m n) (hb : hasUniqueRows b) :
  (∀ k, k > n - m + 1 → ¬ canRemoveColumns b k) ∧
  canRemoveColumns b (n - m + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_removable_columns_l729_72929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l729_72926

/-- The inradius of a tetrahedron given its volume and face areas. -/
def inradius_tetrahedron (V S₁ S₂ S₃ S₄ : ℝ) : ℝ := 
  sorry

/-- The area of a face of a tetrahedron given its volume and face index. -/
def face_area_tetrahedron (V : ℝ) (i : Fin 4) : ℝ := 
  sorry

/-- For a tetrahedron with inradius R and face areas S₁, S₂, S₃, and S₄,
    the volume V is equal to R(S₁ + S₂ + S₃ + S₄). -/
theorem tetrahedron_volume (R S₁ S₂ S₃ S₄ V : ℝ) 
  (h_positive : R > 0 ∧ S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0)
  (h_inradius : R = inradius_tetrahedron V S₁ S₂ S₃ S₄)
  (h_face_areas : S₁ = face_area_tetrahedron V 0 ∧ 
                  S₂ = face_area_tetrahedron V 1 ∧
                  S₃ = face_area_tetrahedron V 2 ∧
                  S₄ = face_area_tetrahedron V 3) :
  V = R * (S₁ + S₂ + S₃ + S₄) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l729_72926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_fish_population_l729_72945

/-- Estimates the total number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population (initially_marked later_caught marked_in_later_catch : ℕ) : 
  initially_marked > 0 → later_caught > 0 → marked_in_later_catch > 0 →
  initially_marked * later_caught / marked_in_later_catch = 1050 → 
  (initially_marked = 50 ∧ later_caught = 168 ∧ marked_in_later_catch = 8) →
  1050 = initially_marked * later_caught / marked_in_later_catch := by
  sorry

#check estimate_fish_population

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_fish_population_l729_72945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_properties_l729_72980

/-- Represents an isosceles trapezoid WXYZ -/
structure IsoscelesTrapezoid where
  WZ : ℝ
  WY : ℝ
  XZ : ℝ
  XY : ℝ
  altitude : ℝ

/-- Calculates the area of an isosceles trapezoid -/
noncomputable def area (t : IsoscelesTrapezoid) : ℝ :=
  (t.WY + t.XZ) * t.altitude / 2

/-- Calculates the length of the diagonal WX in an isosceles trapezoid -/
noncomputable def diagonalWX (t : IsoscelesTrapezoid) : ℝ :=
  Real.sqrt ((t.WY - t.XZ) ^ 2 / 4 + t.altitude ^ 2)

theorem isosceles_trapezoid_properties (t : IsoscelesTrapezoid) 
    (h1 : t.WZ = 18) (h2 : t.WY = 60) (h3 : t.XZ = 30) (h4 : t.XY = 28) (h5 : t.altitude = 15) :
    area t = 675 ∧ diagonalWX t = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_properties_l729_72980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_side_length_is_180_l729_72982

/-- A rectangular pasture with one side against a barn and three sides fenced --/
structure Pasture where
  barn_length : ℝ
  fence_length : ℝ

/-- The length of the side parallel to the barn --/
noncomputable def parallel_side_length (p : Pasture) : ℝ :=
  p.fence_length - 2 * ((p.fence_length - p.barn_length) / 2)

/-- Theorem stating the length of the side parallel to the barn --/
theorem parallel_side_length_is_180 (p : Pasture) 
  (h1 : p.barn_length = 400)
  (h2 : p.fence_length = 300) : 
  parallel_side_length p = 180 := by
  sorry

-- Use a 'def' instead of '#eval' for a computable function
def compute_parallel_side_length : ℚ :=
  300 - 2 * ((300 - 400) / 2)

#eval compute_parallel_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_side_length_is_180_l729_72982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminating_decimal_count_l729_72923

theorem terminating_decimal_count : 
  ∃ (S : Finset ℕ), 
    (∀ m ∈ S, 1 ≤ m ∧ m ≤ 500) ∧ 
    (∀ m ∈ S, ∃ (n a b : ℕ), m / 980 = n / (2^a * 5^b)) ∧
    S.card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminating_decimal_count_l729_72923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_concurrent_lines_theorem_l729_72973

-- Define the triangle and points
def Triangle (A B C : ℝ × ℝ) : Prop := True
def OnSide (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := True
def Concurrent (L1 L2 L3 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := True

-- Define the theorem
theorem triangle_concurrent_lines_theorem 
  (A B C D E F P : ℝ × ℝ) 
  (h_triangle : Triangle A B C)
  (h_D_on_BC : OnSide D B C)
  (h_E_on_CA : OnSide E C A)
  (h_F_on_AB : OnSide F A B)
  (h_concurrent : Concurrent (A, D) (B, E) (C, F))
  (h_sum : (|((A.1 - P.1) / (P.1 - D.1))| + 
            |((B.1 - P.1) / (P.1 - E.1))| + 
            |((C.1 - P.1) / (P.1 - F.1))|) = 88)
  (h_product : |((A.1 - P.1) / (P.1 - D.1))| * 
               |((B.1 - P.1) / (P.1 - E.1))| = 32) :
  |((A.1 - P.1) / (P.1 - D.1))| * 
  |((B.1 - P.1) / (P.1 - E.1))| * 
  |((C.1 - P.1) / (P.1 - F.1))| = 1792 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_concurrent_lines_theorem_l729_72973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sequence_lengths_l729_72989

-- Define the triangle structure
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  E : ℝ × ℝ

-- Define the right-angled property
def isRightAngled (t : Triangle) : Prop :=
  sorry

-- Define the angle measurement
def angle (t : Triangle) (v : ℝ × ℝ) : ℝ :=
  sorry

-- Define the length of a side
def length (a b : ℝ × ℝ) : ℝ :=
  sorry

theorem triangle_sequence_lengths 
  (ABE BCE CDE : Triangle)
  (h1 : isRightAngled ABE)
  (h2 : isRightAngled BCE)
  (h3 : isRightAngled CDE)
  (h4 : angle ABE ABE.B = 60)
  (h5 : angle BCE BCE.B = 60)
  (h6 : angle CDE CDE.B = 60)
  (h7 : length ABE.A ABE.E = 30) :
  length BCE.B BCE.E = 7.5 ∧ length CDE.B CDE.E = 3.75 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sequence_lengths_l729_72989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l729_72957

/-- Circle with equation x^2 + y^2 - 4x - 5 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 - 4*p.1 - 5 = 0}

/-- Midpoint of chord AB -/
def P : ℝ × ℝ := (3, 1)

/-- The equation of line AB given the circle and midpoint P -/
theorem chord_equation (A B : ℝ × ℝ) (hA : A ∈ Circle) (hB : B ∈ Circle)
    (hP : P = ((A.1 + B.1)/2, (A.2 + B.2)/2)) :
  ∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ ({p : ℝ × ℝ | ∃ t, p = (1-t)•A + t•B}) ↔ x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l729_72957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mono_triangle_prob_approx_l729_72913

/-- A regular hexagon with colored edges -/
structure ColoredHexagon :=
  (edges : Fin 15 → Bool)

/-- The probability of an edge being red (or blue) -/
noncomputable def edge_prob : ℝ := 1 / 2

/-- The number of possible triangles in a hexagon -/
def num_triangles : ℕ := 20

/-- The probability of a specific triangle not being monochromatic -/
noncomputable def non_mono_triangle_prob : ℝ := 3 / 4

/-- The probability of at least one monochromatic triangle -/
noncomputable def mono_triangle_prob : ℝ := 1 - non_mono_triangle_prob ^ num_triangles

/-- Theorem: The probability of at least one monochromatic triangle is approximately 0.995 -/
theorem mono_triangle_prob_approx :
  ∃ ε > 0, abs (mono_triangle_prob - 0.995) < ε ∧ ε < 0.001 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mono_triangle_prob_approx_l729_72913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_jersey_price_l729_72902

theorem lowest_jersey_price (list_prices : List ℝ)
  (min_discount : ℝ) (max_discount : ℝ) (additional_discount : ℝ)
  (shipping_fee : ℝ) (sales_tax_rate : ℝ) :
  min_discount = 0.3 →
  max_discount = 0.5 →
  additional_discount = 0.2 →
  shipping_fee = 10 →
  sales_tax_rate = 0.1 →
  list_prices = [100, 120, 150] →
  ∃ (final_price : ℝ),
    final_price = 55 ∧
    ∀ (price : ℝ),
      price ∈ list_prices →
      final_price ≤ (price * (1 - max_discount) * (1 - additional_discount) + shipping_fee) * (1 + sales_tax_rate) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_jersey_price_l729_72902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_f_strictly_increasing_all_intervals_l729_72979

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

-- State the theorem
theorem f_strictly_increasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 8) (k * Real.pi + 3 * Real.pi / 8)) := by
  sorry

-- The main theorem stating that f is strictly increasing on all such intervals
theorem f_strictly_increasing_all_intervals :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 8) (k * Real.pi + 3 * Real.pi / 8)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_f_strictly_increasing_all_intervals_l729_72979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_power_4_in_expansion_l729_72914

def binomial_coefficient (n k : ℕ) : ℕ := sorry

noncomputable def coefficient_x_power_4 (a b : ℝ) (n : ℕ) : ℝ :=
  (binomial_coefficient n 3 : ℝ) * (1 / (2^3))

theorem coefficient_x_power_4_in_expansion :
  coefficient_x_power_4 1 (1 / (2 * 3)) 8 = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_power_4_in_expansion_l729_72914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lloyds_hourly_rate_l729_72972

/-- Lloyd's work scenario --/
structure WorkScenario where
  regularHours : ℚ
  overtimeRate : ℚ
  totalHours : ℚ
  totalEarnings : ℚ

/-- Calculate Lloyd's regular hourly rate --/
noncomputable def calculateHourlyRate (scenario : WorkScenario) : ℚ :=
  scenario.totalEarnings / (scenario.regularHours + scenario.overtimeRate * (scenario.totalHours - scenario.regularHours))

/-- Theorem: Lloyd's regular hourly rate is $4 --/
theorem lloyds_hourly_rate :
  let scenario : WorkScenario := {
    regularHours := 15/2,
    overtimeRate := 3/2,
    totalHours := 21/2,
    totalEarnings := 48
  }
  calculateHourlyRate scenario = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lloyds_hourly_rate_l729_72972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_with_remainder_l729_72986

theorem unique_divisor_with_remainder : ∃! n : ℕ, 
  n > 5 ∧
  (200 % n = 5) ∧ 
  (395 % n = 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_with_remainder_l729_72986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_lawn_mowing_fraction_mowed_correct_l729_72959

/-- If Tom can mow a lawn in 60 minutes, he can mow 1/4 of the lawn in 15 minutes. -/
theorem tom_lawn_mowing (total_time : ℝ) (part_time : ℝ) 
    (h1 : total_time = 60) 
    (h2 : part_time = 15) : 
    part_time / total_time = 1 / 4 := by
  rw [h1, h2]
  norm_num

/-- The fraction of the lawn Tom can mow in 15 minutes. -/
def fraction_mowed : ℚ := 1 / 4

theorem fraction_mowed_correct : 
    fraction_mowed = 1 / 4 := by rfl

#eval fraction_mowed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_lawn_mowing_fraction_mowed_correct_l729_72959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_right_isosceles_triangle_l729_72942

structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  exterior_angle3 : ℝ

theorem exterior_angle_right_isosceles_triangle (t : Triangle) 
  (h1 : t.angle1 = 45) 
  (h2 : t.angle2 = 45) 
  (h3 : t.angle3 = 90) : 
  t.exterior_angle3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_right_isosceles_triangle_l729_72942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_at_two_pi_thirds_l729_72919

-- Define the curve y = -sin x
noncomputable def curve (x : ℝ) : ℝ := -Real.sin x

-- Define the line x - 2y - 6 = 0
def line (x y : ℝ) : Prop := x - 2*y - 6 = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem min_distance_at_two_pi_thirds :
  ∀ (x : ℝ) (y : ℝ) (x_q : ℝ) (y_q : ℝ),
    0 ≤ x ∧ x ≤ Real.pi →
    y = curve x →
    line x_q y_q →
    (∀ (x' : ℝ) (y' : ℝ) (x_q' : ℝ) (y_q' : ℝ),
      0 ≤ x' ∧ x' ≤ Real.pi →
      y' = curve x' →
      line x_q' y_q' →
      distance x y x_q y_q ≤ distance x' y' x_q' y_q') →
    x = 2*Real.pi/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_at_two_pi_thirds_l729_72919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiling_rearrangement_impossible_l729_72990

/-- Represents the types of tiles used in the bathroom -/
inductive TileType
  | Two_by_Two
  | One_by_Four
  deriving BEq, Repr

/-- Represents a tiling of a bathroom floor -/
def Tiling := List TileType

/-- Checks if a tiling is valid (covers the entire floor without gaps or overlaps) -/
def is_valid_tiling : Tiling → Bool := sorry

/-- Theorem stating that it's impossible to rearrange a valid tiling by replacing one tile type with another -/
theorem tiling_rearrangement_impossible (t : Tiling) (h : is_valid_tiling t = true) :
  ¬∃ (t' : Tiling), (is_valid_tiling t' = true) ∧ 
    ((t.count TileType.Two_by_Two = t'.count TileType.Two_by_Two + 1 ∧
      t.count TileType.One_by_Four = t'.count TileType.One_by_Four - 1) ∨
     (t.count TileType.Two_by_Two = t'.count TileType.Two_by_Two - 1 ∧
      t.count TileType.One_by_Four = t'.count TileType.One_by_Four + 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiling_rearrangement_impossible_l729_72990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_property_l729_72983

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Represents that three points form an equilateral triangle -/
def IsEquilateral (A B C : Point) : Prop := sorry

/-- Represents that one circle is tangent internally to another -/
def IsTangentInternally (c1 c2 : Circle) : Prop := sorry

/-- Represents that a point is on a circle -/
def IsOnCircle (P : Point) (c : Circle) : Prop := sorry

/-- Represents that a length is the tangent length from a point to a circle -/
def IsTangentLength (length : ℝ) (P : Point) (c : Circle) : Prop := sorry

/-- Given three circles a, b, c centered at vertices of an equilateral triangle,
    and a circle d tangent to a, b, c internally, 
    for any point P on d, the sum of lengths of any two tangent segments 
    from P to a, b, c equals the length of the third segment -/
theorem tangent_sum_property 
  (a b c d : Circle) 
  (A B C : Point) 
  (is_equilateral : IsEquilateral A B C) 
  (centers_match : a.center = A ∧ b.center = B ∧ c.center = C) 
  (d_tangent_internally : IsTangentInternally d a ∧ IsTangentInternally d b ∧ IsTangentInternally d c) 
  (P : Point) 
  (P_on_d : IsOnCircle P d) 
  (PA PB PC : ℝ) 
  (PA_tangent : IsTangentLength PA P a) 
  (PB_tangent : IsTangentLength PB P b) 
  (PC_tangent : IsTangentLength PC P c) : 
  PA + PB = PC ∧ PB + PC = PA ∧ PC + PA = PB :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_property_l729_72983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_fifth_term_l729_72905

def mySequence : Fin 6 → ℤ
  | 0 => 2
  | 1 => 5
  | 2 => 11
  | 3 => 2
  | 5 => 47
  | _ => 0  -- placeholder for a₅

theorem sequence_fifth_term : mySequence 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_fifth_term_l729_72905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_current_age_l729_72904

-- Define variables for current ages
variable (my_age : ℕ)
variable (brother_age : ℕ)

-- Define the conditions
def condition1 (my_age brother_age : ℕ) : Prop := brother_age - 5 = 2 * (my_age - 5)
def condition2 (my_age brother_age : ℕ) : Prop := (my_age + 8) + (brother_age + 8) = 50

-- Theorem to prove
theorem my_current_age :
  ∀ (my_age brother_age : ℕ),
  condition1 my_age brother_age →
  condition2 my_age brother_age →
  my_age = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_current_age_l729_72904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_ball_probability_l729_72968

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from the given containers -/
def probability_green (containers : List Container) : ℚ :=
  let total_containers := containers.length
  let prob_per_container : ℚ := 1 / total_containers
  let green_probs := containers.map (λ c => (c.green : ℚ) / ((c.red : ℚ) + (c.green : ℚ)))
  (green_probs.map (· * prob_per_container)).sum

/-- The main theorem stating the probability of selecting a green ball -/
theorem green_ball_probability :
  let containers : List Container := [
    { red := 5, green := 5 },  -- Container I
    { red := 3, green := 3 },  -- Container II
    { red := 4, green := 2 }   -- Container III
  ]
  probability_green containers = 4/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_ball_probability_l729_72968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composition_l729_72961

-- Define the function f with domain [-1, 1]
def f : Set ℝ := Set.Icc (-1) 1

-- State the theorem
theorem domain_of_composition :
  {x : ℝ | x^2 - 1 ∈ f} = Set.Icc (-Real.sqrt 2) (Real.sqrt 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composition_l729_72961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l729_72978

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  16 * y^2 - 32 * y - 4 * x^2 - 24 * x + 84 = 0

/-- The distance between the vertices of the hyperbola -/
noncomputable def vertex_distance : ℝ := 2 * Real.sqrt 6.5

/-- Theorem stating that the distance between the vertices of the hyperbola
    given by the equation is equal to 2√6.5 -/
theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_equation x y → vertex_distance = 2 * Real.sqrt 6.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l729_72978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stationary_tank_radius_proof_l729_72936

/-- The volume of a right circular cylinder. -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The radius of the stationary tank. -/
def stationaryTankRadius : ℝ := 100

theorem stationary_tank_radius_proof :
  let truckTankRadius : ℝ := 7
  let truckTankHeight : ℝ := 10
  let stationaryTankHeight : ℝ := 25
  let oilLevelDrop : ℝ := 0.049
  cylinderVolume truckTankRadius truckTankHeight = 
    cylinderVolume stationaryTankRadius oilLevelDrop →
  stationaryTankRadius = 100 := by
  sorry

#check stationary_tank_radius_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stationary_tank_radius_proof_l729_72936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_height_is_five_l729_72946

/-- Represents a triangle with its three heights -/
structure Triangle where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ

/-- The maximum integer height of an unequal-sided triangle with two given heights -/
noncomputable def max_third_height (h₁ h₂ : ℝ) : ℕ :=
  sorry

theorem max_third_height_is_five :
  ∀ t : Triangle,
    t.h₁ = 4 →
    t.h₂ = 12 →
    t.h₁ ≠ t.h₂ →
    t.h₁ ≠ t.h₃ →
    t.h₂ ≠ t.h₃ →
    (∃ n : ℕ, t.h₃ = n) →
    max_third_height t.h₁ t.h₂ = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_height_is_five_l729_72946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_cone_cylinder_l729_72999

noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

theorem volume_ratio_cone_cylinder (r_cyl h_cyl r_cone h_cone : ℝ) 
  (h_cyl_pos : h_cyl > 0) (h_cone_pos : h_cone > 0) (r_cyl_pos : r_cyl > 0) (r_cone_pos : r_cone > 0) :
  r_cyl = 5 → h_cyl = 10 → r_cone = 5 → h_cone = 5 → 
  (cone_volume r_cone h_cone) / (cylinder_volume r_cyl h_cyl) = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_cone_cylinder_l729_72999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_l729_72910

/-- Surface area of a rectangular box with volume 32m³ and height 2m -/
noncomputable def surface_area (x : ℝ) : ℝ := 4 * x + 64 / x + 32

/-- Theorem stating the minimum surface area and optimal base length -/
theorem min_surface_area :
  ∃ (x : ℝ), x > 0 ∧ 
    (∀ y > 0, surface_area x ≤ surface_area y) ∧
    x = 4 ∧ surface_area x = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_l729_72910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l729_72911

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

noncomputable def sequence_a (a : ℝ) (n : ℕ+) : ℝ := f a n

def is_increasing (a : ℝ) : Prop :=
  ∀ n m : ℕ+, n < m → sequence_a a n < sequence_a a m

theorem range_of_a :
  ∀ a : ℝ, is_increasing a ↔ 2 < a ∧ a < 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l729_72911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_fixed_points_l729_72908

theorem circle_fixed_points (m : ℝ) :
  let circle := λ (x y : ℝ) => x^2 + y^2 + 2*m*x - m*y - 25 = 0
  circle (Real.sqrt 5) (2 * Real.sqrt 5) ∧ circle (-Real.sqrt 5) (-2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_fixed_points_l729_72908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_left_handed_people_l729_72933

theorem expected_left_handed_people (sample_size : ℕ) (probability : ℚ) 
  (h1 : sample_size = 300) 
  (h2 : probability = 1/6) : 
  (probability : ℝ) * (sample_size : ℝ) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_left_handed_people_l729_72933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_spending_percentage_l729_72977

/-- Proof that B spends 85% of his salary given the conditions --/
theorem b_spending_percentage
  (total_salary : ℝ)
  (a_salary : ℝ)
  (a_spending_percentage : ℝ)
  (b_spending_percentage : ℝ)  -- Add this line to declare b_spending_percentage
  (h_total : total_salary = 4000)
  (h_a_salary : a_salary = 3000)
  (h_a_spending : a_spending_percentage = 95)
  (h_equal_savings : (1 - a_spending_percentage / 100) * a_salary = 
                     (1 - (b_spending_percentage / 100)) * (total_salary - a_salary)) :
  b_spending_percentage = 85 := by
  sorry

#check b_spending_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_spending_percentage_l729_72977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_at_zero_and_four_l729_72966

/-- A polynomial of degree 4 with real coefficients -/
def polynomial (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_sum_at_zero_and_four
  (a b c d : ℝ)
  (h1 : polynomial a b c d 1 = 2)
  (h2 : polynomial a b c d 2 = 4)
  (h3 : polynomial a b c d 3 = 6) :
  polynomial a b c d 0 + polynomial a b c d 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_at_zero_and_four_l729_72966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l729_72912

-- Define the differential equation
def differential_equation (x : ℝ) (y : ℝ → ℝ) : Prop :=
  2 * y x * (deriv y x) = 3 * x^2

-- Define the proposed solution
noncomputable def proposed_solution (C : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt (x^3 + C)

-- Theorem statement
theorem solution_satisfies_equation (C : ℝ) :
  ∀ x : ℝ, differential_equation x (proposed_solution C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l729_72912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_to_polar_l729_72955

theorem cartesian_to_polar :
  ∃ (ρ θ : ℝ), ρ = 2 ∧ θ = 11 * Real.pi / 6 ∧
  Real.sqrt 3 = ρ * Real.cos θ ∧ -1 = ρ * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_to_polar_l729_72955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l729_72964

noncomputable def g (x : ℝ) : ℝ := Real.sin (4 * x + Real.pi / 6) + 1 / 2

theorem g_range :
  ∀ y : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 4) ∧ g x = y) ↔ y ∈ Set.Icc 0 (3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l729_72964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_quadratic_l729_72949

/-- Definition of a quadratic function -/
def is_quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function y = x²/3 -/
noncomputable def f (x : ℝ) : ℝ := x^2 / 3

/-- Theorem: f is a quadratic function -/
theorem f_is_quadratic : is_quadratic_function f := by
  use (1/3 : ℝ), (0 : ℝ), (0 : ℝ)
  constructor
  · -- Prove a ≠ 0
    exact one_div_ne_zero (by norm_num)
  · -- Prove ∀ x, f x = a * x^2 + b * x + c
    intro x
    simp [f]
    ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_quadratic_l729_72949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_specific_vectors_l729_72915

def vector_sum (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.1 + b.1, a.2.1 + b.2.1, a.2.2 + b.2.2)

def vector_diff (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.1 - b.1, a.2.1 - b.2.1, a.2.2 - b.2.2)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2

noncomputable def vector_magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (a.1^2 + a.2.1^2 + a.2.2^2)

noncomputable def cos_angle (a b : ℝ × ℝ × ℝ) : ℝ :=
  (dot_product a b) / ((vector_magnitude a) * (vector_magnitude b))

theorem cos_angle_specific_vectors :
  ∃ (a b : ℝ × ℝ × ℝ),
    vector_sum a b = (0, Real.sqrt 2, 0) ∧
    vector_diff a b = (2, Real.sqrt 2, -2 * Real.sqrt 3) ∧
    cos_angle a b = -Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_specific_vectors_l729_72915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_trig_function_l729_72931

theorem range_of_trig_function :
  ∀ x : ℝ, -3 ≤ Real.cos (2 * x) + 2 * Real.sin x ∧ Real.cos (2 * x) + 2 * Real.sin x ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_trig_function_l729_72931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_four_l729_72991

/-- The line passing through the origin with slope 1 -/
def my_line (x y : ℝ) : Prop := x - y = 0

/-- The circle centered at the origin with radius 2 -/
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The length of the chord intercepted by the line on the circle -/
def chord_length : ℝ := 4

/-- Theorem stating that the length of the chord intercepted by the line on the circle is 4 -/
theorem chord_length_is_four :
  ∀ x y : ℝ, my_line x y → my_circle x y → chord_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_four_l729_72991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l729_72970

/-- The integral equation φ(x) = ∫₀¹ x t² φ(t) dt + 1 -/
def integral_equation (φ : ℝ → ℝ) : Prop :=
  ∀ x, φ x = (∫ t in Set.Icc 0 1, x * t^2 * φ t) + 1

/-- The proposed solution to the integral equation -/
noncomputable def proposed_solution (x : ℝ) : ℝ := 1 + (4/9) * x

/-- Theorem stating that the proposed solution satisfies the integral equation -/
theorem solution_satisfies_equation :
  integral_equation proposed_solution := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l729_72970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_count_theorem_l729_72981

/-- Given a sample of size 1000 divided into several groups, 
    if the frequency of a certain group is 0.4, 
    then the frequency count of that group is 400. -/
theorem frequency_count_theorem (sample_size : ℕ) (frequency : ℚ) (group_count : ℕ) : 
  sample_size = 1000 → 
  frequency = 0.4 → 
  (frequency * (sample_size : ℚ)) = 400 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_count_theorem_l729_72981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tape_length_calculation_l729_72921

/-- Calculate the total length of a continuous tape made from multiple sheets with overlap -/
theorem tape_length_calculation (num_sheets : ℕ) (sheet_length : ℝ) (overlap : ℝ) : 
  num_sheets > 0 → 
  (num_sheets * sheet_length - (num_sheets - 1) * overlap) / 100 = 3.68 ∧
  num_sheets = 15 ∧ sheet_length = 25 ∧ overlap = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tape_length_calculation_l729_72921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_greater_than_f_one_l729_72963

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4*x + 6 else x + 6

-- Theorem statement
theorem solution_set_of_f_greater_than_f_one :
  {x : ℝ | f x > f 1} = {x : ℝ | -3 < x ∧ x < 1 ∨ x > 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_greater_than_f_one_l729_72963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_equals_one_l729_72928

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the properties of f and g
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x

-- State the relationship between f and g
axiom f_g_sum : ∀ x : ℝ, f x + g x = x^3 - x^2 + 1

-- Theorem to prove
theorem f_one_equals_one : f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_equals_one_l729_72928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_determines_m_coordinate_l729_72944

/-- Given two points P and Q on a line with slope 2, prove that the m-coordinate satisfies m = 4/3 -/
theorem slope_determines_m_coordinate (m : ℝ) : 
  (((m + 2) / (3 - m)) = 2) → m = 4/3 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_determines_m_coordinate_l729_72944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l729_72917

/-- Calculates the rate of interest per annum given the principal, time, and simple interest -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (simple_interest : ℝ) : ℝ :=
  (simple_interest * 100) / (principal * time)

/-- Theorem: Given the specified conditions, the interest rate is 10% -/
theorem interest_rate_is_ten_percent
  (principal : ℝ)
  (time : ℝ)
  (simple_interest : ℝ)
  (h1 : principal = 8032.5)
  (h2 : time = 5)
  (h3 : simple_interest = 4016.25) :
  calculate_interest_rate principal time simple_interest = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l729_72917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l729_72960

noncomputable def f (x m : ℝ) : ℝ := Real.cos x * (Real.sqrt 3 * Real.sin x - Real.cos x) + m

noncomputable def g (x m : ℝ) : ℝ := f (x + Real.pi/6) m

theorem problem_solution :
  ∃ (m : ℝ), 
    (∀ x ∈ Set.Icc (Real.pi/4) (Real.pi/3), g x m ≥ Real.sqrt 3 / 2) ∧
    (∃ x ∈ Set.Icc (Real.pi/4) (Real.pi/3), g x m = Real.sqrt 3 / 2) ∧
    (∀ A B C : ℝ, 0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2 ∧ A + B + C = Real.pi →
      g (C/2) m = -1/2 + Real.sqrt 3 →
        Real.sqrt 3 / 2 < Real.sin A + Real.cos B ∧ Real.sin A + Real.cos B < 3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l729_72960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_ratio_ge_prime_plus_one_l729_72920

theorem exists_ratio_ge_prime_plus_one (p : ℕ) (M : Finset ℕ) 
  (hp : Prime p) (hM : M.card = p + 1) (hM_pos : ∀ m ∈ M, m > 0) :
  ∃ a b, a ∈ M ∧ b ∈ M ∧ a > b ∧ (a : ℚ) / gcd a b ≥ p + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_ratio_ge_prime_plus_one_l729_72920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_ratio_l729_72998

theorem sphere_volume_ratio (r₁ r₂ r₃ : ℝ) (h : r₁ / r₂ = 1 / 2 ∧ r₂ / r₃ = 2 / 3) :
  (4 / 3 * Real.pi * r₁^3) / (4 / 3 * Real.pi * r₂^3) = 1 / 8 ∧
  (4 / 3 * Real.pi * r₂^3) / (4 / 3 * Real.pi * r₃^3) = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_ratio_l729_72998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l729_72988

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-15 * x^2 - 20 * x + 24)

theorem domain_of_f :
  Set.Icc (-12/5 : ℝ) (2/3 : ℝ) = {x : ℝ | ∃ y : ℝ, f x = y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l729_72988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_room_l729_72951

/-- The area of a region between two semicircles with the same center and parallel diameters -/
noncomputable def area_between_semicircles (R r : ℝ) : ℝ := (Real.pi / 2) * (R^2 - r^2)

/-- The farthest distance between two points with a clear line of sight in the region -/
noncomputable def farthest_distance (R r : ℝ) : ℝ := Real.sqrt ((R - r)^2 + (2*R - 2*r)^2)

theorem area_of_room (R r : ℝ) (h1 : R > r) (h2 : farthest_distance R r = 12) :
  area_between_semicircles R r = 18 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_room_l729_72951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_diagonals_not_implies_rectangle_l729_72907

-- Define a quadrilateral
structure Quadrilateral :=
(vertices : Fin 4 → ℝ × ℝ)

-- Define diagonals of a quadrilateral
def diagonals (q : Quadrilateral) : (ℝ × ℝ) × (ℝ × ℝ) :=
  (q.vertices 0, q.vertices 2)

-- Define equality of diagonals
def equal_diagonals (q : Quadrilateral) : Prop :=
  let (d1, d2) := diagonals q
  (d1.1 - d2.1)^2 + (d1.2 - d2.2)^2 = (q.vertices 1 - q.vertices 3).1^2 + (q.vertices 1 - q.vertices 3).2^2

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (q.vertices 0 = (0, 0)) ∧
    (q.vertices 1 = (a, 0)) ∧
    (q.vertices 2 = (a, b)) ∧
    (q.vertices 3 = (0, b))

-- The theorem to be proved
theorem equal_diagonals_not_implies_rectangle :
  ¬(∀ q : Quadrilateral, equal_diagonals q → is_rectangle q) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_diagonals_not_implies_rectangle_l729_72907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_theorem_l729_72993

-- Define the cost per kg for each item
def mango_cost : ℝ := sorry
def rice_cost : ℝ := sorry
def flour_cost : ℝ := sorry

-- State the problem conditions
axiom mango_rice_relation : 10 * mango_cost = 24 * rice_cost
axiom flour_rice_relation : 6 * flour_cost = 2 * rice_cost
axiom flour_price : flour_cost = 20.50

-- Define the total cost function
def total_cost (mango_kg rice_kg flour_kg : ℝ) : ℝ :=
  mango_cost * mango_kg + rice_cost * rice_kg + flour_cost * flour_kg

-- State the theorem
theorem total_cost_theorem :
  total_cost 4 3 5 = 877.40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_theorem_l729_72993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l729_72901

/-- The domain of the function f(x) = 1/((x-3) + (x-9)) is (-∞, 6) ∪ (6, ∞) -/
theorem domain_of_f : 
  let f : ℝ → ℝ := λ x ↦ 1 / ((x - 3) + (x - 9))
  Set.range (Set.univ.restrict f) = {y | ∃ x : ℝ, x ≠ 6 ∧ y = f x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l729_72901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_binomial_expansion_l729_72962

theorem coefficient_x_cubed_in_binomial_expansion : 
  Finset.sum (Finset.range 7) (fun k => (Nat.choose 6 k) * (2^k) * (if k = 3 then 1 else 0)) = 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_binomial_expansion_l729_72962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_time_proof_l729_72975

/-- The number of days A and B take together to complete the work -/
noncomputable def total_days : ℚ := 18

/-- The ratio of A's work rate to B's work rate -/
noncomputable def work_rate_ratio : ℚ := 1 / 2

/-- The ratio of time A works compared to B -/
noncomputable def time_ratio : ℚ := 3 / 4

/-- The time B takes to complete the work alone -/
noncomputable def b_time : ℚ := 24

theorem b_time_proof :
  b_time = total_days * (1 + 1 / (work_rate_ratio / time_ratio)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_time_proof_l729_72975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_l729_72906

def is_prime (n : ℕ) : Bool :=
  if n ≤ 1 then false
  else
    (List.range (n - 1)).all (fun d => 
      if d + 2 ≤ n ∧ d + 2 > 1 then n % (d + 2) ≠ 0
      else true)

def spinner_sections : ℕ := 8

def spinner_labels : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

def prime_count : ℕ := (spinner_labels.filter is_prime).length

theorem spinner_prime_probability : 
  (prime_count : ℚ) / spinner_sections = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_l729_72906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_intersection_l729_72974

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the intersection point X
noncomputable def X : ℝ × ℝ := sorry

-- Define distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define line function
def line (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define circle_boundary function
def circle_boundary (c : Circle) : Set (ℝ × ℝ) := sorry

-- Define the theorem
theorem triangle_circle_intersection
  (triangle : Triangle)
  (circle : Circle)
  (h1 : circle.center = triangle.A)
  (h2 : circle.radius = distance triangle.A triangle.B)
  (h3 : distance triangle.A triangle.B = 86)
  (h4 : distance triangle.A triangle.C = 97)
  (h5 : X ∈ line triangle.B triangle.C)
  (h6 : X ∈ circle_boundary circle)
  (h7 : ∃ n : ℕ, distance triangle.B X = n)
  (h8 : ∃ m : ℕ, distance X triangle.C = m)
  : distance triangle.B triangle.C = 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_intersection_l729_72974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_beta_l729_72935

theorem find_beta (α β : Real) (h1 : Real.cos α = 1/7) (h2 : Real.cos (α - β) = 13/14)
  (h3 : 0 < β) (h4 : β < α) (h5 : α < π/2) : β = π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_beta_l729_72935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_sum_equals_target_l729_72940

/-- Represents an equilateral triangle with side length s -/
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

/-- Calculates the area of an equilateral triangle -/
noncomputable def area (t : EquilateralTriangle) : ℝ :=
  (Real.sqrt 3 / 4) * t.side ^ 2

/-- Represents the shading process in the triangle -/
structure ShadingProcess where
  initial_triangle : EquilateralTriangle
  shaded_ratio : ℝ
  shaded_ratio_valid : shaded_ratio = 1 / 9

/-- Calculates the sum of the shaded areas after infinite iterations -/
noncomputable def sum_shaded_areas (p : ShadingProcess) : ℝ :=
  (area p.initial_triangle * p.shaded_ratio) / (1 - p.shaded_ratio)

/-- The theorem to be proved -/
theorem shaded_area_sum_equals_target (t : EquilateralTriangle) 
    (h : t.side = 12) : 
    ∃ (p : ShadingProcess), 
      p.initial_triangle = t ∧ 
      sum_shaded_areas p = 4.5 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_sum_equals_target_l729_72940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_is_cube_l729_72932

theorem product_is_cube (nums : Finset ℕ) : 
  (Finset.card nums = 175) →
  (∀ n ∈ nums, n > 0 ∧ ∀ p : ℕ, p.Prime → p ∣ n → p ≤ 10) →
  ∃ a b c, a ∈ nums ∧ b ∈ nums ∧ c ∈ nums ∧ 
           a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
           ∃ m : ℕ, a * b * c = m^3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_is_cube_l729_72932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shallow_side_is_one_meter_l729_72925

/-- Represents a trapezoidal prism-shaped swimming pool -/
structure SwimmingPool where
  width : ℝ
  length : ℝ
  deep_side_depth : ℝ
  volume : ℝ

/-- Calculates the shallow side depth of a swimming pool -/
noncomputable def shallow_side_depth (pool : SwimmingPool) : ℝ :=
  2 * pool.volume / (pool.width * pool.length) - pool.deep_side_depth

/-- Theorem stating that the shallow side depth of the given swimming pool is 1 meter -/
theorem shallow_side_is_one_meter (pool : SwimmingPool) 
  (h_width : pool.width = 9)
  (h_length : pool.length = 12)
  (h_deep_side : pool.deep_side_depth = 4)
  (h_volume : pool.volume = 270) :
  shallow_side_depth pool = 1 := by
  sorry

#check shallow_side_is_one_meter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shallow_side_is_one_meter_l729_72925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_no_zeros_l729_72927

/-- The function f(x) given in the problem -/
noncomputable def f (a x : ℝ) : ℝ := a / x + x / a - (a - 1 / a) * Real.log x

/-- Theorem stating that f(x) has no zeros when a ∈ [1/2, 2] -/
theorem f_no_zeros (a x : ℝ) (ha : 1/2 ≤ a ∧ a ≤ 2) (hx : x > 0) : f a x > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_no_zeros_l729_72927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_chords_distance_l729_72958

/-- Given a semicircle with two parallel chords perpendicular to its diameter,
    where the first chord is 24 units long and the second chord is 10 units long,
    the distance between these parallel chords is 6 11/12 units. -/
theorem parallel_chords_distance (r d : ℝ) : 
  r > 0 → d > 0 → 
  r^2 = d^2 + 12^2 →
  r^2 = (d + (r - d))^2 + 5^2 →
  d = 83/12 := by
  sorry

#eval (83 : ℚ) / 12  -- This will evaluate to 6 11/12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_chords_distance_l729_72958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_monotonically_increasing_intervals_l729_72922

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x - 6

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := x^2 - 1

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ (m b : ℝ), m * 3 + b = f 3 ∧
  ∀ x, m * x + b = f_derivative 3 * (x - 3) + f 3 := by
  sorry

-- Theorem for monotonically increasing intervals
theorem monotonically_increasing_intervals :
  (∀ x, x < -1 → f_derivative x > 0) ∧
  (∀ x, x > 1 → f_derivative x > 0) ∧
  (∀ x, x > -1 ∧ x < 1 → f_derivative x < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_monotonically_increasing_intervals_l729_72922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l729_72965

/-- Calculates the speed of a train given its length and time to cross a fixed point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem: A train 300 meters long that takes 20 seconds to cross an electric pole has a speed of 15 meters per second -/
theorem train_speed_calculation :
  train_speed 300 20 = 15 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l729_72965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_tan_difference_l729_72969

theorem arctan_tan_difference (θ : Real) : 
  θ ∈ Set.Icc 0 180 → 
  Real.arctan (Real.tan (75 * π / 180) - 2 * Real.tan (30 * π / 180)) * 180 / π = θ → 
  θ = 75 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_tan_difference_l729_72969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_max_value_achieved_l729_72948

noncomputable def f (x y z : ℝ) : ℝ := Real.sqrt (x * y + 5) + Real.sqrt (y * z + 5) + Real.sqrt (z * x + 5)

theorem max_value_of_f (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_constraint : x * y + y * z + z * x = 1) :
  f x y z ≤ 4 * Real.sqrt 3 := by
  sorry

theorem max_value_achieved (ε : ℝ) (h_pos : ε > 0) :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y + y * z + z * x = 1 ∧ f x y z > 4 * Real.sqrt 3 - ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_max_value_achieved_l729_72948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_l729_72976

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (3 * x^2 + 4 * x - 7) / (-7 * x^2 + 4 * x + a)

/-- The domain of f(x) is all real numbers iff a < -4/7 -/
theorem domain_condition (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ a < -4/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_l729_72976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_of_paraboloid_surface_l729_72930

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the paraboloid surface y^2 + z^2 = 10x -/
def Paraboloid (p : Point3D) : Prop :=
  p.y^2 + p.z^2 = 10 * p.x

/-- The cutting plane x = 10 -/
def CuttingPlane (p : Point3D) : Prop :=
  p.x = 10

/-- The centroid of the surface -/
noncomputable def centroid : Point3D :=
  { x := (25 * Real.sqrt 5 + 1) / (5 * Real.sqrt 5 - 1),
    y := 0,
    z := 0 }

/-- Theorem stating that the calculated centroid is correct for the given surface -/
theorem centroid_of_paraboloid_surface :
  ∀ (p : Point3D), Paraboloid p → CuttingPlane p →
  ∃ (c : Point3D), c = centroid ∧
    (∀ (q : Point3D), Paraboloid q → CuttingPlane q →
      (q.x - c.x)^2 + (q.y - c.y)^2 + (q.z - c.z)^2 ≤
      (q.x - p.x)^2 + (q.y - p.y)^2 + (q.z - p.z)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_of_paraboloid_surface_l729_72930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l729_72954

-- Definition of a neighboring root equation
noncomputable def is_neighboring_root_equation (a b c : ℝ) : Prop :=
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  abs (x₁ - x₂) = 1

-- Part 1
theorem part1 : is_neighboring_root_equation 2 (-2*Real.sqrt 5) 2 := by sorry

-- Part 2
theorem part2 (m : ℝ) : is_neighboring_root_equation 1 (2-m) (-2*m) ↔ m = -1 ∨ m = -3 := by sorry

-- Part 3
theorem part3 : 
  ∃ (t : ℝ), t = 18 ∧ 
  ∀ (a b : ℝ), a < 0 → is_neighboring_root_equation a b 2 → 
  2 - b^2 ≤ t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l729_72954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l729_72909

-- Define the arithmetic sequence
noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the sum of the first n terms
noncomputable def S_n (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_problem (a₁ : ℝ) (d : ℝ) :
  (arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 3 + arithmetic_sequence a₁ d 5 = 105) →
  (arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 4 + arithmetic_sequence a₁ d 6 = 99) →
  (d = -2 ∧ ∀ n : ℕ, n > 20 → S_n a₁ d n ≤ S_n a₁ d 20) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l729_72909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_for_pencils_l729_72996

theorem no_integer_solution_for_pencils :
  ∀ (p : ℕ),
    let q := p + 5
    let m := (7 * p) / 5
    (5 * p = 6 * q - 30) →
    (3 * p + 2 * q + 4 * m = 156) →
    (∃ (k : ℕ), 5 * k = p ∧ 6 * k = q ∧ 7 * k = m) →
    False :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_for_pencils_l729_72996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l729_72950

-- Define the function f(x) = xe^x
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (y = m * (x - 1) + f 1) ↔ (2 * Real.exp 1 * x - y - Real.exp 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l729_72950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_theorem_l729_72956

/-- Two circles in a plane -/
structure TwoCircles where
  ω₁ : Set (ℝ × ℝ)
  ω₂ : Set (ℝ × ℝ)

/-- Configuration of points for the problem -/
structure Configuration (tc : TwoCircles) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ  -- The other intersection point of ω₁ and ω₂
  A_in_intersection : A ∈ tc.ω₁ ∩ tc.ω₂
  F_in_intersection : F ∈ tc.ω₁ ∩ tc.ω₂
  B_in_ω₁ : B ∈ tc.ω₁
  AB_tangent_to_ω₂ : ∀ X ∈ tc.ω₂, (X - A) • (B - A) = 0 → X = A
  CD_on_tangent_at_B : (∀ X ∈ tc.ω₁, (X - B) • (C - B) = 0) ∧ (∀ X ∈ tc.ω₁, (X - B) • (D - B) = 0)
  C_D_in_ω₂ : C ∈ tc.ω₂ ∧ D ∈ tc.ω₂
  D_closer_to_B : ‖B - D‖ < ‖B - C‖
  E_on_AD_and_ω₁ : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • A + t • D ∧ E ∈ tc.ω₁
  BD_length : ‖B - D‖ = 3
  CD_length : ‖C - D‖ = 13

/-- The main theorem to be proved -/
theorem ratio_theorem (tc : TwoCircles) (cfg : Configuration tc) :
  ‖cfg.E - cfg.B‖ / ‖cfg.E - cfg.D‖ = (4 * Real.sqrt 3) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_theorem_l729_72956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_45_degrees_l729_72953

-- Define a triangle ABC with sides a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the area of the triangle
noncomputable def area (t : Triangle) : ℝ := (t.a^2 + t.b^2 - t.c^2) / 4

-- Define the measure of angle C in radians
noncomputable def angle_C (t : Triangle) : ℝ := Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

-- Theorem statement
theorem angle_C_is_45_degrees (t : Triangle) (h : area t = (t.a^2 + t.b^2 - t.c^2) / 4) :
  angle_C t = π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_45_degrees_l729_72953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l729_72938

variable (M : Matrix (Fin 2) (Fin 2) ℝ)

def v1 : Fin 2 → ℝ := ![1, -2]
def v2 : Fin 2 → ℝ := ![-4, 6]
def v : Fin 2 → ℝ := ![7, -1]

def w1 : Fin 2 → ℝ := ![2, 1]
def w2 : Fin 2 → ℝ := ![0, -2]
def w : Fin 2 → ℝ := ![-38, -6]

theorem matrix_transformation (h1 : M.mulVec v1 = w1) (h2 : M.mulVec v2 = w2) :
  M.mulVec v = w := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l729_72938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emmy_had_14_ipods_emmy_had_14_ipods_proof_l729_72937

-- Define the number of iPods Emmy originally had
def emmy_original : ℕ → Prop := fun e => True

-- Define the number of iPods Rosa has
def rosa : ℕ → Prop := fun r => True

-- Emmy loses 6 iPods and still has twice as many as Rosa
axiom condition1 {e r : ℕ} : emmy_original e → rosa r → e - 6 = 2 * r

-- Emmy and Rosa together have 12 iPods after Emmy loses 6
axiom condition2 {e r : ℕ} : emmy_original e → rosa r → (e - 6) + r = 12

-- Theorem: Emmy originally had 14 iPods
theorem emmy_had_14_ipods : ∃ e, emmy_original e ∧ e = 14 := by
  -- We'll use 14 as our witness for e
  use 14
  apply And.intro
  · -- Prove emmy_original 14
    trivial
  · -- Prove 14 = 14
    rfl

-- The actual proof would go here, but we'll use sorry for now
theorem emmy_had_14_ipods_proof : ∃ e, emmy_original e ∧ e = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emmy_had_14_ipods_emmy_had_14_ipods_proof_l729_72937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l729_72934

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := 10^(log10 (x - 1))
noncomputable def g (x : ℝ) : ℝ := ((x - 1) / Real.sqrt (x - 1))^2

-- Theorem statement
theorem f_equals_g (x : ℝ) (h : x > 1) : f x = g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l729_72934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l729_72916

-- Define the function f(x) = ln x + 1/x
noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l729_72916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_point_with_double_inclination_line_equation_maximizing_distance_l729_72939

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Angle of inclination of a line -/
noncomputable def angleOfInclination (l : Line) : ℝ := Real.arctan (-l.a / l.b)

/-- Distance from a point to a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

theorem line_equation_through_point_with_double_inclination
  (P : Point)
  (h₁ : P.x = -1 ∧ P.y = -3)
  (l₁ : Line)
  (h₂ : l₁.a = 1 ∧ l₁.b = -3 ∧ l₁.c = 0)
  (l₂ : Line)
  (h₃ : angleOfInclination l₂ = 2 * angleOfInclination l₁)
  (h₄ : l₂.a * P.x + l₂.b * P.y + l₂.c = 0) :
  l₂.a = 3 ∧ l₂.b = 4 ∧ l₂.c = 15 :=
by sorry

theorem line_equation_maximizing_distance
  (A B : Point)
  (h₁ : A.x = -1 ∧ A.y = 1)
  (h₂ : B.x = 2 ∧ B.y = -1)
  (l : Line)
  (h₃ : l.a * A.x + l.b * A.y + l.c = 0)
  (h₄ : ∀ l' : Line, l'.a * A.x + l'.b * A.y + l'.c = 0 →
        distancePointToLine B l ≥ distancePointToLine B l') :
  l.a = 3 ∧ l.b = -2 ∧ l.c = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_point_with_double_inclination_line_equation_maximizing_distance_l729_72939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_area_ratio_l729_72992

/-- For a right triangle with hypotenuse h and an inscribed circle of radius r,
    the ratio of the area of the inscribed circle to the area of the triangle is πr / (h + 2r) -/
theorem inscribed_circle_area_ratio (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let triangle := {p : ℝ × ℝ | p.1^2 + p.2^2 = h^2 ∧ p.1 > 0 ∧ p.2 > 0}
  ∀ p ∈ triangle,
  (π * r^2) / ((1/2) * p.1 * p.2) = π * r / (h + 2*r) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_area_ratio_l729_72992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_fixed_point_l729_72971

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the tangent line
def tangentLine (x : ℝ) : Prop := x = -1

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The theorem
theorem circle_passes_through_fixed_point :
  ∀ (center : ℝ × ℝ),
    parabola center →
    (∃ (r : ℝ), r > 0 ∧ distance center (-1, center.2) = r) →
    distance center (1, 0) = distance center (-1, center.2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_fixed_point_l729_72971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_larger_angle_larger_sin_squared_sum_implies_obtuse_l729_72924

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  -- Add triangle inequality constraints
  ab_sum : a + b > c
  bc_sum : b + c > a
  ca_sum : c + a > b
  -- Angles sum to pi
  angle_sum : A + B + C = Real.pi

-- Theorem for Option A
theorem sin_larger_angle_larger (t : Triangle) (h : t.A > t.B) : 
  Real.sin t.A > Real.sin t.B :=
sorry

-- Theorem for Option D
theorem sin_squared_sum_implies_obtuse (t : Triangle) 
  (h : Real.sin t.C ^ 2 > Real.sin t.A ^ 2 + Real.sin t.B ^ 2) : 
  t.C > Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_larger_angle_larger_sin_squared_sum_implies_obtuse_l729_72924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_bullets_theorem_probability_all_five_bullets_l729_72994

/-- The probability of requiring all bullets in a shooting practice -/
def probability_all_bullets (n : ℕ) (p : ℝ) : ℝ :=
  (1 - p) ^ (n - 1)

/-- Theorem stating the probability of requiring all bullets in a shooting practice -/
theorem probability_all_bullets_theorem (n : ℕ) (p : ℝ) 
  (hn : n > 0) (hp_pos : 0 < p) (hp_lt_one : p < 1) : 
  probability_all_bullets n p = (1 - p) ^ (n - 1) := by
  sorry

/-- The specific case for 5 bullets and 2/3 hit probability -/
theorem probability_all_five_bullets : 
  probability_all_bullets 5 (2/3) = (1/3)^4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_bullets_theorem_probability_all_five_bullets_l729_72994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_properties_l729_72984

/-- A circle passing through two points with its center on a given line -/
structure SpecialCircle where
  -- Center of the circle
  center : ℝ × ℝ
  -- Radius of the circle
  radius : ℝ
  -- The circle passes through points A(-1, 2) and B(3, 4)
  passes_through_A : (center.1 + 1)^2 + (center.2 - 2)^2 = radius^2
  passes_through_B : (center.1 - 3)^2 + (center.2 - 4)^2 = radius^2
  -- The center lies on the line x + 3y - 15 = 0
  center_on_line : center.1 + 3 * center.2 - 15 = 0

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (P Q R : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))

/-- The main theorem about the special circle and the maximum area of triangle PAB -/
theorem special_circle_properties (c : SpecialCircle) :
  -- The equation of the circle is x² + (y - 5)² = 10
  (c.center = (0, 5) ∧ c.radius^2 = 10) ∧
  -- The maximum area of triangle PAB is 5 + 5√2
  (∃ (max_area : ℝ), max_area = 5 + 5 * Real.sqrt 2 ∧
    ∀ (P : ℝ × ℝ), (P.1 - c.center.1)^2 + (P.2 - c.center.2)^2 = c.radius^2 →
      area_triangle P (-1, 2) (3, 4) ≤ max_area) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_properties_l729_72984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_l729_72997

theorem subset_count : 
  let S : Finset Nat := {1, 2, 3, 4, 5, 6}
  let T : Finset Nat := {1, 2}
  (Finset.filter (fun X => T ⊆ X ∧ X ⊆ S) (Finset.powerset S)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_l729_72997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_optimal_price_l729_72985

/-- Represents the hand sanitizer pricing and sales model -/
structure HandSanitizerModel where
  costPrice : ℝ
  initialSellingPrice : ℝ
  initialDailySales : ℝ
  priceDecrease : ℝ
  salesIncrease : ℝ

/-- Calculate daily sales volume based on selling price -/
noncomputable def dailySalesVolume (model : HandSanitizerModel) (sellingPrice : ℝ) : ℝ :=
  model.initialDailySales + (model.salesIncrease / model.priceDecrease) * (model.initialSellingPrice - sellingPrice)

/-- Calculate daily profit based on selling price -/
noncomputable def dailyProfit (model : HandSanitizerModel) (sellingPrice : ℝ) : ℝ :=
  (sellingPrice - model.costPrice) * (dailySalesVolume model sellingPrice)

/-- Theorem stating the maximum daily profit and optimal selling price -/
theorem max_profit_at_optimal_price (model : HandSanitizerModel) 
  (h1 : model.costPrice = 16)
  (h2 : model.initialSellingPrice = 20)
  (h3 : model.initialDailySales = 80)
  (h4 : model.priceDecrease = 0.5)
  (h5 : model.salesIncrease = 20) :
  ∃ (optimalPrice : ℝ), 
    optimalPrice = 19 ∧ 
    dailyProfit model optimalPrice = 360 ∧
    ∀ (price : ℝ), model.costPrice ≤ price → dailyProfit model price ≤ dailyProfit model optimalPrice := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_optimal_price_l729_72985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_range_l729_72967

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (1/3) * Real.sin (2*x) + a * Real.sin x

theorem monotone_increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  -1/3 ≤ a ∧ a ≤ 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_range_l729_72967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_points_unit_distance_existence_l729_72918

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Predicate to check if two points are unit distance apart -/
def isUnitApart (p q : Point) : Prop :=
  distance p q = 1

/-- Main theorem: Existence of seven points with the desired property -/
theorem seven_points_unit_distance_existence :
  ∃ (points : Finset Point),
    Finset.card points = 7 ∧
    ∀ (subset : Finset Point),
      subset ⊆ points →
      Finset.card subset = 3 →
      ∃ (p q : Point), p ∈ subset ∧ q ∈ subset ∧ p ≠ q ∧ isUnitApart p q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_points_unit_distance_existence_l729_72918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l729_72952

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  property1 : a 3 + a 4 = 15
  property2 : a 2 * a 5 = 54
  property3 : d < 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence -/
def S_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = 11 - n) ∧
  (∃ n_max : ℕ, n_max = 11 ∧ ∀ n, S_n seq n ≤ S_n seq n_max) ∧
  S_n seq 11 = 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l729_72952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l729_72987

-- Define the function f(x) = 3^x + 5x
noncomputable def f (x : ℝ) : ℝ := Real.exp (Real.log 3 * x) + 5 * x

-- Theorem statement
theorem zero_in_interval :
  ∃ x ∈ Set.Ioo (-1/5 : ℝ) 0, f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l729_72987
