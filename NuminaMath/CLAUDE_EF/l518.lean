import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_series_positive_l518_51822

open Real BigOperators

theorem sine_series_positive (x : ℝ) (h1 : 0 < x) (h2 : x < π) :
  ∀ n : ℕ+, (∑ k in Finset.range n, sin ((2 * ↑k + 1) * x) / (2 * ↑k + 1)) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_series_positive_l518_51822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_drain_rate_value_l518_51843

-- Define the tank and its properties
def tank_capacity : ℝ := 1000
def initial_water_level : ℝ := 500
def fill_rate : ℝ := 500
def first_drain_rate : ℝ := 250
def fill_time : ℝ := 6

-- Define the second drain rate as a function
def second_drain_rate (x : ℝ) : Prop :=
  (fill_rate - first_drain_rate - x) * fill_time = tank_capacity - initial_water_level

-- Theorem statement
theorem second_drain_rate_value :
  ∃ x, second_drain_rate x ∧ x = 1000 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_drain_rate_value_l518_51843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_in_cone_l518_51895

/-- Given a right cone with base radius r and height m, and a sphere centered at the apex of the cone
    that divides the cone into two equal volumes, this theorem states that the volume of the sphere
    is equal to (4/3) * π * (294 * 25). -/
theorem sphere_volume_in_cone (r m : ℝ) (hr : r = 7) (hm : m = 24) :
  let cone_volume := (1/3) * Real.pi * r^2 * m
  let sphere_radius := (294 * 25 : ℝ) ^ (1/3)
  let sphere_volume := (4/3) * Real.pi * sphere_radius^3
  2 * ((1/3) * Real.pi * sphere_radius^2 * (sphere_radius / 25)) = cone_volume / 2 →
  sphere_volume = (4/3) * Real.pi * (294 * 25) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_in_cone_l518_51895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_difference_l518_51851

theorem angle_difference (α β : Real) : 
  α ∈ Set.Ioo 0 (Real.pi / 2) →
  β ∈ Set.Ioo 0 (Real.pi / 2) →
  Real.sin α = Real.sqrt 5 / 5 →
  Real.cos β = 1 / Real.sqrt 10 →
  α - β = -Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_difference_l518_51851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_interval_is_80_to_84_l518_51869

/-- Represents a histogram interval with its lower bound and frequency -/
structure HistInterval where
  lower_bound : ℕ
  frequency : ℕ

/-- The problem statement -/
theorem median_interval_is_80_to_84 
  (total_students : ℕ)
  (intervals : List HistInterval)
  (h_total : total_students = 100)
  (h_intervals : intervals = [
    ⟨90, 15⟩, 
    ⟨85, 20⟩, 
    ⟨80, 25⟩, 
    ⟨75, 18⟩, 
    ⟨70, 12⟩, 
    ⟨65, 10⟩
  ])
  : ∃ i ∈ intervals, i.lower_bound = 80 ∧ 
    (List.sum (List.map HistInterval.frequency (List.takeWhile (fun x => x.lower_bound ≥ i.lower_bound) intervals)) ≥ total_students / 2) ∧
    (List.sum (List.map HistInterval.frequency (List.takeWhile (fun x => x.lower_bound > i.lower_bound) intervals)) < total_students / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_interval_is_80_to_84_l518_51869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_values_of_a_l518_51815

theorem integer_values_of_a (a b : ℝ) (h1 : a + 8*b - 2*b^2 = 7) (h2 : 1 ≤ b) (h3 : b ≤ 4) :
  Finset.card (Finset.Icc (-1 : ℤ) 7) = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_values_of_a_l518_51815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sqrt_three_others_rational_l518_51896

-- Define the numbers given in the problem
noncomputable def a : ℝ := 0
noncomputable def b : ℝ := Real.sqrt 3
noncomputable def c : ℝ := -2
noncomputable def d : ℝ := 3 / 7

-- Theorem stating that √3 is irrational while the others are rational
theorem irrational_sqrt_three_others_rational :
  ¬ (∃ (p q : ℤ), b = ↑p / ↑q) ∧
  (∃ (p q : ℤ), a = ↑p / ↑q) ∧
  (∃ (p q : ℤ), c = ↑p / ↑q) ∧
  (∃ (p q : ℤ), d = ↑p / ↑q) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sqrt_three_others_rational_l518_51896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_center_lines_l518_51842

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a 3D plane
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Predicate to check if four lines are pairwise skew -/
def PairwiseSkew (l₁ l₂ l₃ l₄ : Line3D) : Prop := sorry

/-- Predicate to check if no three lines are parallel to the same plane -/
def NoThreeParallelToSamePlane (l₁ l₂ l₃ l₄ : Line3D) : Prop := sorry

/-- Represents a set of four pairwise skew lines in 3D space -/
structure SkewLines where
  l₁ : Line3D
  l₂ : Line3D
  l₃ : Line3D
  l₄ : Line3D
  are_pairwise_skew : PairwiseSkew l₁ l₂ l₃ l₄
  no_three_parallel_to_same_plane : NoThreeParallelToSamePlane l₁ l₂ l₃ l₄

/-- Function to find the number of lines traced by the centers of parallelograms -/
def numCenterLines (lines : SkewLines) : ℕ := sorry

/-- Theorem stating that the number of lines traced by the centers of parallelograms is exactly 3 -/
theorem parallelogram_center_lines (lines : SkewLines) : numCenterLines lines = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_center_lines_l518_51842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_equals_fraction_l518_51860

open BigOperators Finset Real

noncomputable def binomial_sum (n : ℕ) : ℝ :=
  ∑ k in range (n + 1), ((-1)^k * (n.choose k : ℝ)) / ((k^3 : ℝ) + 9*k^2 + 26*k + 24)

theorem binomial_sum_equals_fraction (n : ℕ) :
  binomial_sum n = 1 / (2 * (n + 3) * (n + 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_equals_fraction_l518_51860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_5k_divisible_by_5_l518_51891

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_5k_divisible_by_5 (k : ℕ) (h : k ≥ 1) :
  ∃ m : ℕ, fibonacci (5 * k) = 5 * m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_5k_divisible_by_5_l518_51891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_price_increase_l518_51866

/-- Given two successive price increases in gas, prove that the second increase is 20%
    when the first increase is 30% and a 35.89743589743589% reduction in consumption
    keeps the expenditure constant. -/
theorem gas_price_increase (P C : ℝ) (hP : P > 0) (hC : C > 0) : 
  let first_increase := 0.30
  let consumption_reduction := 0.3589743589743589
  let second_increase := 0.20
  (P * (1 + first_increase) * (1 + second_increase)) * (C * (1 - consumption_reduction)) = P * C :=
by
  -- Introduce the local definitions
  have first_increase := 0.30
  have consumption_reduction := 0.3589743589743589
  have second_increase := 0.20

  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_price_increase_l518_51866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_a_eq_half_l518_51864

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^3 - 1 else 2^x

-- State the theorem
theorem f_of_a_eq_half :
  ∃ a : ℝ, f 0 = a ∧ f a = 1/2 := by
  -- Prove that a = -1
  let a := -1
  
  -- Show that f(0) = a
  have h1 : f 0 = a := by
    simp [f]
    -- 0^3 - 1 = -1
    norm_num
  
  -- Show that f(a) = 1/2
  have h2 : f a = 1/2 := by
    simp [f, a]
    -- 2^(-1) = 1/2
    norm_num
  
  -- Conclude the proof
  exact ⟨a, h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_a_eq_half_l518_51864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_cylinder_volume_ratio_is_256_135_l518_51898

/-- The ratio of the volume of a sphere to the volume of a cylinder -/
noncomputable def sphere_to_cylinder_volume_ratio (q : ℝ) : ℝ :=
  let sphere_volume := (4 / 3) * Real.pi * (4 * q) ^ 3
  let cylinder_volume := Real.pi * (3 * q) ^ 2 * (5 * q)
  sphere_volume / cylinder_volume

/-- Theorem: The ratio of the volume of a sphere with radius 4q to the volume of a cylinder 
    with radius 3q and height 5q is equal to 256/135 -/
theorem sphere_to_cylinder_volume_ratio_is_256_135 (q : ℝ) :
  sphere_to_cylinder_volume_ratio q = 256 / 135 := by
  -- Unfold the definition of sphere_to_cylinder_volume_ratio
  unfold sphere_to_cylinder_volume_ratio
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_cylinder_volume_ratio_is_256_135_l518_51898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l518_51805

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

/-- The center of the ellipse -/
def O : ℝ × ℝ := (0, 0)

/-- The left focus of the ellipse -/
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 7, 0)

/-- The right focus of the ellipse -/
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 7, 0)

/-- Distance between two points -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem ellipse_property (x y : ℝ) :
  is_on_ellipse x y →
  x ≠ 4 ∧ x ≠ -4 →
  distance (x, y) F₁ * distance (x, y) F₂ + (distance (x, y) O)^2 = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l518_51805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_product_factorials_l518_51826

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- The sum of trailing zeros for factorials up to n! -/
def sumTrailingZeros (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun i => trailingZeros (i + 1))

/-- The main theorem: The number of trailing zeros in the product 1!2!3!4!⋯49!50! modulo 1000 is 702 -/
theorem trailing_zeros_product_factorials :
  sumTrailingZeros 50 % 1000 = 702 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_product_factorials_l518_51826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_handshake_triangle_l518_51801

/-- Represents a meeting of representatives. -/
structure Meeting (m : ℕ) where
  handshakes : Fin (3 * m) → Fin (3 * m) → Bool
  symm : ∀ i j, handshakes i j = handshakes j i
  no_self_handshake : ∀ i, handshakes i i = false

/-- A meeting is n-interesting if there exist n representatives with handshake counts 1, 2, ..., n. -/
def is_n_interesting (m n : ℕ) (meeting : Meeting m) : Prop :=
  ∃ (reps : Fin n → Fin (3 * m)), 
    (∀ i j, i ≠ j → reps i ≠ reps j) ∧ 
    (∀ i : Fin n, (Finset.univ.filter (fun j ↦ meeting.handshakes (reps i) j)).card = i.val + 1)

/-- Three representatives have all shaken hands with each other. -/
def has_handshake_triangle (m : ℕ) (meeting : Meeting m) : Prop :=
  ∃ i j k : Fin (3 * m), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    meeting.handshakes i j ∧ meeting.handshakes j k ∧ meeting.handshakes i k

/-- The main theorem stating that for any m ≥ 2, the smallest n such that every n-interesting 
    meeting guarantees a handshake triangle is 3. -/
theorem smallest_n_for_handshake_triangle (m : ℕ) (h : m ≥ 2) :
  (∀ n : ℕ, n ≥ 3 → ∀ meeting : Meeting m, is_n_interesting m n meeting → has_handshake_triangle m meeting) ∧
  (∃ meeting : Meeting m, is_n_interesting m 2 meeting ∧ ¬has_handshake_triangle m meeting) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_handshake_triangle_l518_51801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_range_l518_51855

open Real

theorem alpha_range (α : ℝ) (h_α : α ∈ Set.Ioo 0 (π / 2)) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioo 0 1 ∧
    (fun x => 1 / x) x₀ = (fun x => log x + tan α) x₀) →
  α ∈ Set.Ioo (π / 4) (π / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_range_l518_51855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l518_51868

noncomputable def f (x : Real) : Real := 5 * Real.cos (x + Real.pi / 3)

theorem phase_shift_of_f :
  ∃ (shift : Real), shift = -Real.pi / 3 ∧
  ∀ (x : Real), f x = 5 * Real.cos (x - shift) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l518_51868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_randy_gave_sally_verify_problem_instance_l518_51873

/-- Calculates the amount Randy gave Sally based on his initial money, 
    the gift from Smith, and the amount Randy kept. -/
theorem randy_gave_sally (initial : ℕ) (smith_gift : ℕ) (randy_kept : ℕ) : 
  initial + smith_gift - randy_kept = 1200 :=
by
  sorry

/-- Verifies the specific problem instance -/
theorem verify_problem_instance : 
  3000 + 200 - 2000 = 1200 :=
by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_randy_gave_sally_verify_problem_instance_l518_51873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_l518_51808

-- Define the quadrilateral ABCD
def A : ℝ × ℝ := (-2, 1)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (3, -1)
def D : ℝ × ℝ := (0, -2)

-- Define the angles in degrees
def angle_DAB : ℝ := 30
def angle_ABC : ℝ := 85
def angle_BCD : ℝ := 60
def angle_CDA : ℝ := 55

-- Define the segments
noncomputable def AB : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
noncomputable def BC : ℝ := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
noncomputable def CD : ℝ := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
noncomputable def DA : ℝ := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
noncomputable def BD : ℝ := Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2)

-- Theorem statement
theorem longest_segment :
  CD > AB ∧ CD > BC ∧ CD > DA ∧ CD > BD := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_l518_51808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l518_51818

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  4 * x^2 - 16 * x - y^2 + 6 * y - 11 = 0

/-- The distance between vertices of the hyperbola -/
noncomputable def vertex_distance : ℝ := 2 * Real.sqrt 4.5

/-- Theorem: The distance between the vertices of the hyperbola
    4x^2 - 16x - y^2 + 6y - 11 = 0 is equal to 2√(4.5) -/
theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_eq x y → 
  ∃ x₁ x₂ y₀ : ℝ, (x₁ ≠ x₂) ∧ 
  hyperbola_eq x₁ y₀ ∧ hyperbola_eq x₂ y₀ ∧
  |x₁ - x₂| = vertex_distance := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l518_51818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_example_l518_51876

theorem complex_magnitude_example : Complex.abs (7/4 - 3*Complex.I) = Real.sqrt 193 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_example_l518_51876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_altitude_segments_l518_51887

-- Define the necessary concepts
def IsAcute (a b c : ℝ) : Prop := sorry
def IsAltitude (a b c d : ℝ) : Prop := sorry
def SegmentLength (a b : ℝ) : ℝ := sorry

-- The main theorem
theorem acute_triangle_altitude_segments 
  (a b c : ℝ) 
  (h_acute : IsAcute a b c) 
  (h_alt1 : ∃ (d : ℝ), IsAltitude a b c d ∧ SegmentLength b d = 7 ∧ SegmentLength d c = 10)
  (h_alt2 : ∃ (e y : ℝ), IsAltitude b a c e ∧ SegmentLength a e = 3 ∧ SegmentLength e c = y) :
  ∃ (y : ℝ), y = 30 / 7 :=
by
  sorry -- The proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_altitude_segments_l518_51887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axes_intersect_at_centroid_l518_51877

/-- A polygon with uniform mass distribution at vertices -/
structure UniformPolygon where
  vertices : Set (ℝ × ℝ)
  is_polygon : Prop -- We'll leave this as a proposition to be defined later

/-- An axis of symmetry for a polygon -/
structure AxisOfSymmetry (P : UniformPolygon) where
  line : Set (ℝ × ℝ) -- Representing a line as a set of points
  is_symmetry_axis : Prop -- We'll leave this as a proposition to be defined later

/-- The centroid (center of mass) of a polygon -/
noncomputable def centroid (P : UniformPolygon) : ℝ × ℝ := sorry

/-- Theorem: All axes of symmetry of a polygon with uniform mass distribution intersect at its centroid -/
theorem axes_intersect_at_centroid (P : UniformPolygon) 
  (h : ∃ (A B : AxisOfSymmetry P), A ≠ B) :
  ∀ (axis : AxisOfSymmetry P), (centroid P) ∈ axis.line := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axes_intersect_at_centroid_l518_51877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_30_degrees_l518_51813

/-- Triangle ABC with given properties -/
structure TriangleABC where
  angleA : Real
  a : Real
  b : Real
  angleA_eq : angleA = 2 * Real.pi / 3  -- 120° in radians
  a_eq : a = 2
  b_eq : b = 2 * Real.sqrt 3 / 3

/-- The measure of angle B in the triangle -/
noncomputable def angleB (t : TriangleABC) : Real := Real.pi / 6  -- 30° in radians

/-- Theorem: In triangle ABC, if angle A = 120°, a = 2, and b = 2√3/3, then angle B = 30° -/
theorem angle_B_is_30_degrees (t : TriangleABC) : angleB t = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_30_degrees_l518_51813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_can_escape_l518_51897

/-- Represents a point in the Euclidean plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The game state after n moves -/
structure GameState where
  n : ℕ
  rabbit_pos : Point
  hunter_pos : Point

/-- A strategy for the rabbit -/
def RabbitStrategy := GameState → Point

/-- A strategy for the hunter -/
def HunterStrategy := GameState → Point → Point

/-- The game simulation function -/
noncomputable def simulate (rabbit_strategy : RabbitStrategy) (hunter_strategy : HunterStrategy) (n : ℕ) : GameState :=
  sorry

/-- The theorem statement -/
theorem rabbit_can_escape :
  ∃ (rabbit_strategy : RabbitStrategy),
    ∀ (hunter_strategy : HunterStrategy),
      ∃ (n : ℕ),
        n < 10^9 ∧
        let final_state := simulate rabbit_strategy hunter_strategy n
        distance final_state.rabbit_pos final_state.hunter_pos > 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_can_escape_l518_51897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_gcd_product_l518_51821

theorem square_of_gcd_product (x y z : ℕ+) (h : (1 : ℚ) / x.val - (1 : ℚ) / y.val = (1 : ℚ) / z.val) :
  ∃ k : ℕ, Nat.gcd x.val (Nat.gcd y.val z.val) * x.val * y.val * z.val = k ^ 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_gcd_product_l518_51821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l518_51823

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) + 1 / (x - 2)

-- Define the domain
def domain : Set ℝ := {x | x ≥ 1 ∧ x ≠ 2}

-- Theorem statement
theorem f_domain : 
  {x : ℝ | ∃ y, f x = y} = domain := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l518_51823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_identity_l518_51862

theorem triangle_trigonometric_identity 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  c - a = h ∧
  h = c * Real.sin B →
  Real.sin ((C - A) / 2) + Real.cos ((C + A) / 2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_identity_l518_51862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l518_51844

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) = f (f x - y) + 4 * f x * y) : 
  (∀ x : ℝ, f x = x^2) ∨ (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l518_51844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt10_parts_sqrt6_sqrt13_sum_opposite_x_minus_y_l518_51854

-- Define the integer and decimal parts of a real number
noncomputable def intPart (x : ℝ) : ℤ := Int.floor x
noncomputable def decPart (x : ℝ) : ℝ := x - Int.floor x

-- Statement 1
theorem sqrt10_parts : intPart (Real.sqrt 10) = 3 ∧ decPart (Real.sqrt 10) = Real.sqrt 10 - 3 := by
  sorry

-- Statement 2
theorem sqrt6_sqrt13_sum :
  let a := decPart (Real.sqrt 6)
  let b := intPart (Real.sqrt 13)
  a + b - Real.sqrt 6 = 1 := by
  sorry

-- Statement 3
theorem opposite_x_minus_y :
  ∃ (x : ℤ) (y : ℝ), 12 + Real.sqrt 3 = x + y ∧ 0 < y ∧ y < 1 ∧
  -(x - y) = Real.sqrt 3 - 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt10_parts_sqrt6_sqrt13_sum_opposite_x_minus_y_l518_51854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_sum_l518_51820

def is_valid_polynomial (P : ℕ → ℕ) : Prop :=
  ∀ i, P i > 0 ∧ (i > 699 → P i = 0)

def polynomial_sum (P : ℕ → ℕ) : ℕ :=
  (Finset.range 700).sum P

def consecutive_sum (P : ℕ → ℕ) (start : ℕ) (len : ℕ) : ℕ :=
  (Finset.range len).sum (λ i => P (start + i))

theorem polynomial_coefficient_sum
  (P : ℕ → ℕ)
  (h_valid : is_valid_polynomial P)
  (h_sum : polynomial_sum P ≤ 2022) :
  ∃ start len, (len > 0) ∧
    (consecutive_sum P start len ∈ ({22, 55, 77} : Finset ℕ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_sum_l518_51820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_matches_exp_l518_51861

-- Define the function f
noncomputable def f (x : ℝ) := Real.log (x + 1)

-- Define the translation of f to the right by one unit
noncomputable def f_translated (x : ℝ) := f (x - 1)

-- State the theorem
theorem f_matches_exp :
  ∀ x : ℝ, f_translated x = Real.exp x ∨ Real.exp (f_translated x) = x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_matches_exp_l518_51861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_prop_true_l518_51852

-- Define the basic structures
structure Line
structure Plane

-- Define the relationships
axiom parallel : Line → Plane → Prop
axiom parallel_lines : Line → Line → Prop
axiom line_in_plane : Line → Plane → Prop
axiom line_outside_plane : Line → Plane → Prop

-- Define the propositions
def prop1 (l : Line) (α : Plane) : Prop :=
  (∃ (S : Set Line), (∀ m ∈ S, parallel_lines l m ∧ line_in_plane m α) ∧ Set.Infinite S) →
  parallel l α

def prop2 (a : Line) (α : Plane) : Prop :=
  line_outside_plane a α → parallel a α

def prop3 (a b : Line) (α : Plane) : Prop :=
  parallel_lines a b ∧ line_in_plane b α → parallel a α

def prop4 (a b : Line) (α : Plane) : Prop :=
  parallel_lines a b ∧ line_in_plane b α →
  ∃ (S : Set Line), (∀ m ∈ S, parallel_lines a m ∧ line_in_plane m α) ∧ Set.Infinite S

-- Theorem stating that only the fourth proposition is true
theorem only_fourth_prop_true :
  (∃ l : Line, ∃ α : Plane, ¬prop1 l α) ∧
  (∃ a : Line, ∃ α : Plane, ¬prop2 a α) ∧
  (∃ a b : Line, ∃ α : Plane, ¬prop3 a b α) ∧
  (∀ a b : Line, ∀ α : Plane, prop4 a b α) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_prop_true_l518_51852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_single_asymptote_l518_51884

/-- The rational function g(x) with parameter d -/
noncomputable def g (d : ℝ) (x : ℝ) : ℝ := (x^2 - 3*x + d) / ((x - 4) * (x + 2))

/-- A vertical asymptote occurs at x if the denominator is zero but the numerator is not -/
def has_vertical_asymptote (d : ℝ) (x : ℝ) : Prop :=
  (x - 4) * (x + 2) = 0 ∧ x^2 - 3*x + d ≠ 0

/-- The theorem stating that there are no values of d for which g(x) has exactly one vertical asymptote -/
theorem no_single_asymptote : ¬∃d : ℝ, ∃!x : ℝ, has_vertical_asymptote d x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_single_asymptote_l518_51884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mow_time_is_40_l518_51878

/-- Represents the time (in minutes) it takes Max to mow the lawn. -/
def mow_time : ℕ := sorry

/-- Represents the time (in minutes) it takes Max to fertilize the lawn. -/
def fertilize_time : ℕ := sorry

/-- The time to fertilize is twice the time to mow. -/
axiom fertilize_double_mow : fertilize_time = 2 * mow_time

/-- The total time to mow and fertilize is 120 minutes. -/
axiom total_time : mow_time + fertilize_time = 120

/-- Proves that it takes Max 40 minutes to mow the lawn. -/
theorem mow_time_is_40 : mow_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mow_time_is_40_l518_51878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_worker_loading_time_specific_loading_time_l518_51802

/-- The time taken for two workers to load a truck together, given their individual loading times -/
theorem two_worker_loading_time (t1 t2 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) :
  (1 / (1 / t1 + 1 / t2)) = (t1 * t2) / (t1 + t2) := by
  sorry

/-- The specific case for the given problem -/
theorem specific_loading_time :
  (6 * 5) / (6 + 5) = 30 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_worker_loading_time_specific_loading_time_l518_51802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clara_height_cm_l518_51807

-- Define Clara's height in inches
def clara_height_inches : ℝ := 54

-- Define the conversion factor from inches to centimeters
def inch_to_cm : ℝ := 2.54

-- Function to convert inches to centimeters
def inches_to_cm (inches : ℝ) : ℝ := inches * inch_to_cm

-- Function to round to the nearest tenth
noncomputable def round_to_tenth (x : ℝ) : ℝ := 
  ⌊x * 10 + 0.5⌋ / 10

-- Theorem statement
theorem clara_height_cm :
  round_to_tenth (inches_to_cm clara_height_inches) = 137.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clara_height_cm_l518_51807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l518_51837

-- Define the parabola C
def C (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus F
noncomputable def F (p : ℝ) : ℝ × ℝ := (Real.sqrt p, 0)

-- Define the circle with diameter PF
def circleEq (p : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point of intersection A
noncomputable def A (p : ℝ) : ℝ × ℝ := (2 * Real.sqrt 5 - 4, 4 * Real.sqrt (2 * Real.sqrt 5 - 4))

-- Define the angle OAF
noncomputable def angleOAF (p : ℝ) : ℝ := Real.arccos ((Real.sqrt 5 - 1) / 2)

-- Theorem statement
theorem parabola_properties (p : ℝ) :
  (∀ x y, C p x y ↔ y^2 = 8*x) ∧
  (Real.cos (angleOAF p) = (Real.sqrt 5 - 1) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l518_51837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_quadratic_sum_l518_51888

-- Define the original function
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- Define the shifted function
def g (x : ℝ) : ℝ := f (x + 6)

-- Theorem statement
theorem shifted_quadratic_sum :
  ∃ (a b c : ℝ), (∀ x, g x = a * x^2 + b * x + c) ∧ (a + b + c = 156) := by
  -- Introduce the existential variables
  use 3, 38, 115
  
  -- Split the goal into two parts
  constructor
  
  -- Prove that g x = 3x^2 + 38x + 115 for all x
  · intro x
    simp [g, f]
    ring
  
  -- Prove that 3 + 38 + 115 = 156
  · ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_quadratic_sum_l518_51888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_perimeter_l518_51893

/-- The minimum perimeter of a triangle with two sides of 24 and 37 units, and the third side being an integer -/
theorem min_triangle_perimeter : ℕ := by
  -- Define the triangle sides
  let a : ℕ := 24
  let b : ℕ := 37
  -- Define the third side as a variable
  let c : ℕ := 14  -- We know the minimum value is 14
  -- Define the perimeter
  let perimeter : ℕ := a + b + c
  -- Assert that c satisfies the triangle inequality
  have h1 : c + a > b := by
    norm_num
  have h2 : c + b > a := by
    norm_num
  have h3 : a + b > c := by
    norm_num
  -- The minimum perimeter
  have h_perimeter : perimeter = 75 := by
    rfl
  exact 75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_perimeter_l518_51893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_parallel_iff_a_eq_neg_one_l518_51879

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_iff_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- Definition of line l1 -/
def l1 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x + a * y + 6 = 0

/-- Definition of line l2 -/
def l2 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (a - 2) * x + 3 * y + 2 * a = 0

/-- Theorem: l1 and l2 are parallel if and only if a = -1 -/
theorem lines_parallel_iff_a_eq_neg_one :
  ∀ a : ℝ, (∀ x y : ℝ, l1 a x y ↔ l2 a x y) ↔ a = -1 := by
  sorry

#check lines_parallel_iff_a_eq_neg_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_parallel_iff_a_eq_neg_one_l518_51879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_questionnaire_c_count_l518_51889

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  first_group_number : ℕ
  (population_positive : population > 0)
  (sample_size_positive : sample_size > 0)
  (sample_size_le_population : sample_size ≤ population)
  (first_group_number_valid : first_group_number > 0 ∧ first_group_number ≤ population / sample_size)

/-- Calculates the number of sampled items in a given range -/
def count_in_range (s : SystematicSample) (start : ℕ) (end_ : ℕ) : ℕ :=
  sorry

theorem questionnaire_c_count (s : SystematicSample) 
  (h1 : s.population = 1000)
  (h2 : s.sample_size = 50)
  (h3 : s.first_group_number = 8) :
  count_in_range s 751 1000 = 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_questionnaire_c_count_l518_51889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_tangential_quadrilateral_properties_l518_51810

/-- A quadrilateral that is both cyclic and tangential -/
structure CyclicTangentialQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  cyclic : Bool
  tangential : Bool

/-- Properties of a cyclic tangential quadrilateral -/
def CyclicTangentialQuadrilateralProps (q : CyclicTangentialQuadrilateral) : Prop :=
  q.cyclic ∧ q.tangential ∧ q.a > 0 ∧ q.b > 0 ∧ q.c > 0 ∧ q.d > 0

theorem cyclic_tangential_quadrilateral_properties 
  (q : CyclicTangentialQuadrilateral) 
  (h : CyclicTangentialQuadrilateralProps q) :
  let t := Real.sqrt (q.a * q.b * q.c * q.d)
  let r := t / (q.a + q.c)
  let tan_half_angle := Real.sqrt ((q.c * q.d) / (q.a * q.b))
  (∀ s, s = q.a + q.c → s = q.b + q.d) ∧  -- semi-perimeter property
  t = Real.sqrt (q.a * q.b * q.c * q.d) ∧  -- area
  r = t / (q.a + q.c) ∧ r = t / (q.b + q.d) ∧  -- radius of inscribed circle
  tan_half_angle = Real.sqrt ((q.c * q.d) / (q.a * q.b))  -- tangent of half-angle
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_tangential_quadrilateral_properties_l518_51810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_number_l518_51882

theorem smallest_divisible_number : ∃ (N : ℕ), 
  (∀ k : ℕ, k ∈ Finset.range 9 → (N + k + 2) % (k + 2) = 0) ∧ 
  (∀ M : ℕ, M < N → ∃ k : ℕ, k ∈ Finset.range 9 ∧ (M + k + 2) % (k + 2) ≠ 0) ∧ 
  N = 2520 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_number_l518_51882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_radius_is_circle_l518_51894

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The set of points satisfying r = 5 in polar coordinates -/
def ConstantRadiusSet : Set PolarPoint :=
  {p : PolarPoint | p.r = 5}

/-- A circle with center (0,0) and radius 5 in Cartesian coordinates -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 25}

/-- Conversion from polar to Cartesian coordinates -/
noncomputable def polarToCartesian (p : PolarPoint) : ℝ × ℝ :=
  (p.r * Real.cos p.θ, p.r * Real.sin p.θ)

theorem constant_radius_is_circle :
  {polarToCartesian p | p ∈ ConstantRadiusSet} = Circle := by
  sorry

#check constant_radius_is_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_radius_is_circle_l518_51894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_proof_l518_51849

/-- The radius of a circle inscribed in a square with side length 12, 
    touching two congruent equilateral triangles at one point each. -/
noncomputable def inscribed_circle_radius : ℝ := 6 - 3 * Real.sqrt 2

/-- The side length of the square -/
def square_side_length : ℝ := 12

/-- The side length of the equilateral triangles -/
noncomputable def triangle_side_length : ℝ := 4 * Real.sqrt 6

theorem inscribed_circle_radius_proof :
  ∃ (r : ℝ),
    r = inscribed_circle_radius ∧
    r > 0 ∧
    r * 2 < square_side_length ∧
    ∃ (h : ℝ),
      h = Real.sqrt (triangle_side_length ^ 2 - (triangle_side_length / 2) ^ 2) ∧
      r = (square_side_length - h) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_proof_l518_51849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_domain_and_inequality_imply_b_range_l518_51838

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * 2^x + a - 2) / (2^x + 1)

-- Theorem for part (1)
theorem odd_function_implies_a_equals_one (a : ℝ) :
  (∀ x, f a x = -f a (-x)) → a = 1 := by sorry

-- Theorem for part (2)
theorem domain_and_inequality_imply_b_range (a b : ℝ) :
  (∀ x ≥ -4, f a x ≠ 0) →
  (∀ x, f a (Real.cos x + b + 1/4) ≥ f a (Real.sin x ^ 2 - b - 3)) →
  b ∈ Set.Icc (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_domain_and_inequality_imply_b_range_l518_51838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_position_proof_l518_51856

/-- Represents the direction of the dog's movement -/
inductive Direction
| Left
| Right

/-- Represents a single run of the dog -/
structure Run where
  distance : ℕ
  direction : Direction

/-- The state of the dog's position after a run -/
structure DogState where
  position : Int  -- Position relative to the midpoint M
  lastDirection : Direction

def pathLength : ℕ := 28

def midpointDistance : ℕ := pathLength / 2

def runPattern : List Run := [
  {distance := 10, direction := Direction.Left},
  {distance := 14, direction := Direction.Right}
]

def numRuns : ℕ := 20

def finalPosition : Int := midpointDistance - 1

def initialPosition : Int := 7 - midpointDistance

/-- Function to update the dog's state after a run -/
def updateState (state : DogState) (run : Run) : DogState :=
  sorry

/-- Function to simulate all runs -/
def simulateRuns (initialState : DogState) : DogState :=
  sorry

theorem dog_position_proof :
  let initialState : DogState := {position := initialPosition, lastDirection := Direction.Left}
  let finalState := simulateRuns initialState
  finalState.position = finalPosition := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_position_proof_l518_51856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l518_51865

theorem cubic_equation_roots (α β γ : ℝ) : 
  let P (x : ℝ) := (1+2*α)*(β-γ)*x^3 + (2+α)*(γ-1)*x^2 + (3-α)*(1-β)*x + (4+α)*(β-γ)
  let Q (x : ℝ) := (2+α)*x^2 + (3-α)*x + (4+α)
  P 1 = 0 → ∃ (a b : ℝ), (∀ x, P x = (x - 1) * Q x) ∧ Q a = 0 ∧ Q b = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l518_51865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_7_l518_51827

-- Define the functions t and f
noncomputable def t (x : ℝ) : ℝ := Real.sqrt (4 * x^2 + 1)
noncomputable def f (x : ℝ) : ℝ := 7 - t x

-- State the theorem
theorem t_of_f_7 : t (f 7) = Real.sqrt (985 - 56 * Real.sqrt 197) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_7_l518_51827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l518_51806

theorem function_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) = 2 * x + f (-f (f x) + f y)) :
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l518_51806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_parts_l518_51800

-- Define the function f implicitly
noncomputable def f (x : ℝ) : Set ℝ :=
  {y | 4 * y - x = Real.sqrt (x^2 - 2*x + 10*y - 4*x*y - 1)}

-- State the theorem
theorem function_parts :
  (∀ x ∈ Set.Iic 2, (1/2 : ℝ) ∈ f x) ∧
  (∀ x : ℝ, (1/4 * x + 1/8 : ℝ) ∈ f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_parts_l518_51800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookies_collected_proof_l518_51847

/-- The number of cookies in each box -/
def cookies_per_box : ℕ := 48

/-- The number of boxes Abigail collected -/
def abigail_boxes : ℕ := 2

/-- The fraction of a box Grayson collected -/
def grayson_fraction : ℚ := 3/4

/-- The number of boxes Olivia collected -/
def olivia_boxes : ℕ := 3

/-- The total number of cookies collected by Abigail, Grayson, and Olivia -/
def total_cookies : ℕ := 276

/-- Theorem stating that the total number of cookies collected is 276 -/
theorem cookies_collected_proof : 
  (abigail_boxes * cookies_per_box) + 
  (grayson_fraction * ↑cookies_per_box).floor + 
  (olivia_boxes * cookies_per_box) = total_cookies := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookies_collected_proof_l518_51847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_obtuse_triangle_with_center_l518_51863

-- Define a circle
def Circle : Type := { p : ℝ × ℝ // p.1^2 + p.2^2 = 1 }

-- Define a function to check if two points form an obtuse triangle with the center
def isObtuseAngleWithCenter (a b : Circle) : Prop :=
  (a.val.1 * b.val.1 + a.val.2 * b.val.2 < 0)

-- Define a function to check if three points form an obtuse triangle with the center
def isObtuseTriangleWithCenter (p1 p2 p3 : Circle) : Prop :=
  isObtuseAngleWithCenter p1 p2 ∨ isObtuseAngleWithCenter p2 p3 ∨ isObtuseAngleWithCenter p3 p1

-- Define the probability space
def Ω : Type := Circle × Circle × Circle

-- Define the probability measure (noncomputable as it involves real numbers)
noncomputable def P : Set Ω → ℝ := sorry

-- State the theorem
theorem probability_no_obtuse_triangle_with_center :
  P {ω : Ω | ¬isObtuseTriangleWithCenter ω.1 ω.2.1 ω.2.2} = 3/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_obtuse_triangle_with_center_l518_51863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_on_time_condition_late_condition_journey_distance_l518_51858

/-- Represents the total distance of the journey in kilometers. -/
noncomputable def total_distance : ℝ := 70

/-- Represents the time taken to reach the destination on time in hours. -/
noncomputable def time_on_time : ℝ := total_distance / 40

/-- The car reaches its destination on time with an average speed of 40 km/hr. -/
theorem on_time_condition : total_distance = 40 * time_on_time := by sorry

/-- The car is late by 15 minutes with an average speed of 35 km/hr. -/
theorem late_condition : total_distance = 35 * (time_on_time + 0.25) := by sorry

/-- The total distance of the journey is 70 kilometers. -/
theorem journey_distance : total_distance = 70 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_on_time_condition_late_condition_journey_distance_l518_51858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_MOI_l518_51832

/-- Predicate to check if a point is the circumcenter of a triangle -/
def is_circumcenter (O A B C : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a point is the incenter of a triangle -/
def is_incenter (I A B C : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a point is the center of a circle tangent to two sides of a triangle and its circumcircle -/
def is_tangent_circle (M A B C : ℝ × ℝ) : Prop := sorry

/-- Function to calculate the area of a triangle given its vertices -/
def area_triangle (P Q R : ℝ × ℝ) : ℝ := sorry

/-- Given a triangle ABC with side lengths, prove that the area of triangle MOI is 1/4 -/
theorem area_triangle_MOI (A B C O I M : ℝ × ℝ) : 
  let d := (λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))
  d A B = 15 ∧ d A C = 8 ∧ d B C = 7 →
  is_circumcenter O A B C →
  is_incenter I A B C →
  is_tangent_circle M A B C →
  area_triangle M O I = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_MOI_l518_51832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_reflection_distance_p_to_p_reflected_l518_51880

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection (x y : ℝ) : 
  let p : ℝ × ℝ := (x, y)
  let p_reflected : ℝ × ℝ := (x, -y)
  Real.sqrt ((p.1 - p_reflected.1)^2 + (p.2 - p_reflected.2)^2) = 2 * |y| :=
by sorry

/-- The specific case for P(2, 5) --/
theorem distance_p_to_p_reflected : 
  let p : ℝ × ℝ := (2, 5)
  let p_reflected : ℝ × ℝ := (2, -5)
  Real.sqrt ((p.1 - p_reflected.1)^2 + (p.2 - p_reflected.2)^2) = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_reflection_distance_p_to_p_reflected_l518_51880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_l518_51804

-- Define the triangle and its properties
def Triangle (D E F : ℝ × ℝ) : Prop :=
  ‖E - F‖ = 5 ∧ ‖D - E‖ = 5 ∧ ‖D - F‖ = 5

-- Define the folded point D'
def FoldedPoint (D' : ℝ × ℝ) (E F : ℝ × ℝ) : Prop :=
  D' ∈ Set.Icc E F ∧ ‖E - D'‖ = 2 ∧ ‖D' - F‖ = 3

-- Define the crease points R and S
def CreasePoints (R S : ℝ × ℝ) (D D' E F : ℝ × ℝ) : Prop :=
  R ∈ Set.Icc E D ∧ S ∈ Set.Icc F D ∧
  ‖R - D'‖ = ‖R - D‖ ∧ ‖S - D'‖ = ‖S - D‖

-- State the theorem
theorem crease_length 
  (D E F D' R S : ℝ × ℝ) 
  (h1 : Triangle D E F) 
  (h2 : FoldedPoint D' E F) 
  (h3 : CreasePoints R S D D' E F) : 
  ‖R - S‖ = Real.sqrt 301 / 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_l518_51804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l518_51886

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

/-- The derivative of the curve function -/
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

/-- The point of tangency -/
noncomputable def point_of_tangency : ℝ := Real.exp 1

theorem tangent_line_equation :
  ∃ (m b : ℝ), ∀ x y : ℝ,
    y = m * x + b ↔
    y - f point_of_tangency = f' point_of_tangency * (x - point_of_tangency) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l518_51886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_implies_fraction_value_l518_51819

theorem sin_equation_implies_fraction_value (α : ℝ) :
  Real.sin (3 * π + α) = 2 * Real.sin ((3 * π) / 2 + α) →
  (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_implies_fraction_value_l518_51819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thabo_books_l518_51848

/-- The number of books Thabo owns -/
def total_books : ℕ := 180

/-- The number of hardcover nonfiction books Thabo owns -/
def hardcover_nonfiction : ℕ := 30

/-- The number of paperback nonfiction books Thabo owns -/
def paperback_nonfiction : ℕ := hardcover_nonfiction + 20

/-- The number of paperback fiction books Thabo owns -/
def paperback_fiction : ℕ := 2 * paperback_nonfiction

theorem thabo_books : 
  hardcover_nonfiction + paperback_nonfiction + paperback_fiction = total_books := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thabo_books_l518_51848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_T_shaped_is_nine_sixteenths_l518_51890

/-- The area of the T-shaped region in a square with side length s -/
noncomputable def area_T_shaped (s : ℝ) : ℝ :=
  let square_WXYZ := s^2
  let square_1 := (s/2)^2
  let square_2 := (s/4)^2
  let rectangle := (s/2) * (s/4)
  square_WXYZ - square_1 - square_2 - rectangle

/-- Theorem stating that the area of the T-shaped region is 9s²/16 -/
theorem area_T_shaped_is_nine_sixteenths (s : ℝ) (h : s > 0) :
  area_T_shaped s = (9 * s^2) / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_T_shaped_is_nine_sixteenths_l518_51890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_digit_of_power_sum_2011_l518_51857

/-- The ones digit of a number -/
def onesDigit (n : Nat) : Nat := n % 10

/-- The sum of nth powers of numbers from 1 to m -/
def powerSum (m n : Nat) : Nat :=
  (List.range m).map (fun i => (i + 1) ^ n) |>.sum

/-- The main theorem -/
theorem ones_digit_of_power_sum_2011 :
  onesDigit (powerSum 2011 2011) = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_digit_of_power_sum_2011_l518_51857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l518_51803

/-- Defines an ellipse with foci at (-c, 0) and (c, 0), and semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_pos : 0 < b ∧ b < a
  h_c : c^2 = a^2 - b^2

/-- A point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ := E.c / E.a

/-- Theorem stating the range of eccentricity for an ellipse with the given properties -/
theorem eccentricity_range (E : Ellipse) (P : PointOnEllipse E)
  (h_dot_product : (P.x + E.c)^2 + P.y^2 * ((P.x - E.c)^2 + P.y^2) = E.c^2) :
  Real.sqrt 3 / 3 ≤ eccentricity E ∧ eccentricity E ≤ Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l518_51803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l518_51835

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x

-- State the theorem
theorem f_inequality : f (-3) < f 2 ∧ f 2 < f (Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l518_51835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_sum_l518_51830

theorem tan_ratio_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 1)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 6) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 124/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_sum_l518_51830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_graph_condition_l518_51809

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2 * Real.sqrt 5 * x + 3 * m - 1

def in_first_second_third_quadrant (x y : ℝ) : Prop :=
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)

def graph_in_first_second_third_quadrants (m : ℝ) : Prop :=
  ∀ x : ℝ, in_first_second_third_quadrant x (f m x)

theorem quadratic_graph_condition (m : ℝ) :
  graph_in_first_second_third_quadrants m ↔ 1/3 ≤ m ∧ m < 2 := by
  sorry

#check quadratic_graph_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_graph_condition_l518_51809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vincent_wins_iff_l518_51870

/-- Given a prime p > 3 and k ≥ 2, Vincent can always determine the permutation
    chosen by Théo if and only if 3 ≤ k ≤ p-2. -/
theorem vincent_wins_iff (p k : ℕ) (hp : p.Prime) (hp3 : p > 3) (hk : k ≥ 2) :
  (∀ σ : Equiv.Perm (Fin k), 
    ∀ f : Fin k → Fin k → Fin p,
    (∀ i j : Fin k, i ≠ j → f i j = ((σ i).val * (σ j).val : ℕ) % p) →
    ∃! τ : Equiv.Perm (Fin k), ∀ i j : Fin k, i ≠ j → f i j = ((τ i).val * (τ j).val : ℕ) % p) ↔
  3 ≤ k ∧ k ≤ p - 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vincent_wins_iff_l518_51870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l518_51885

theorem log_inequality (a : ℝ) (m n p : ℝ) 
  (h1 : 0 < a) (h2 : a < 1)
  (hm : m = Real.log (a^2 + 1) / Real.log a)
  (hn : n = Real.log (a + 1) / Real.log a)
  (hp : p = Real.log (2*a) / Real.log a) :
  p > m ∧ m > n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l518_51885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_inequality_system_solution_l518_51839

-- Problem 1
theorem calculation_proof :
  (3 - Real.pi) ^ 0 - Real.sqrt 4 + 4 * Real.sin (60 * π / 180) + |Real.sqrt 3 - 3| = 2 + Real.sqrt 3 := by sorry

-- Problem 2
theorem inequality_system_solution (x : ℝ) :
  (5 * (x + 3) > 4 * x + 8 ∧ x / 6 - 1 < (x - 2) / 3) ↔ x > -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_inequality_system_solution_l518_51839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l518_51872

-- Define the circle C in polar coordinates
noncomputable def circle_C (θ : ℝ) : ℝ := 4 * Real.sin θ

-- Define the line l in parametric form
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-Real.sqrt 3 / 2 * t, 2 + t / 2)

-- Define the intersection points A and B
def A : ℝ := 2
def B : ℝ := -2

-- Define point P
def P : ℝ := -4

-- Theorem statement
theorem circle_and_line_properties :
  -- The center of circle C has polar coordinates (2, π/2)
  (∃ (ρ θ : ℝ), ρ = 2 ∧ θ = Real.pi / 2 ∧ 
    ∀ (φ : ℝ), circle_C φ = ρ * Real.sin (φ - θ)) ∧
  -- |PA| + |PB| = 8
  |A - P| + |B - P| = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l518_51872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_little_theorem_inverse_l518_51899

theorem fermat_little_theorem_inverse (p : ℕ) (h_prime : Nat.Prime p) :
  ∃ a : ℕ, a > 0 ∧ ∀ k : ℤ, k ≠ 0 → (k : ZMod p) ≠ 0 → (k : ZMod p)^a = (k : ZMod p)⁻¹ :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_little_theorem_inverse_l518_51899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_angles_equal_converse_right_angles_equal_common_factor_polynomials_correct_statements_l518_51833

-- Define IsVerticalAngle and IsRightAngle as predicates
def IsVerticalAngle (α β : Real) : Prop := sorry
def IsRightAngle (α : Real) : Prop := sorry

-- Statement 1: Vertical angles are equal
theorem vertical_angles_equal (α β : Real) (h : IsVerticalAngle α β) : α = β := by sorry

-- Statement 2: Converse of "all right angles are equal"
theorem converse_right_angles_equal (α β : Real) (h : α = β) : IsRightAngle α ∨ IsRightAngle β := by sorry

-- Statement 3: Common factor of polynomials
theorem common_factor_polynomials (a : Real) : 
  ∃ (f g : Polynomial Real), (X^2 - 4 : Polynomial Real) = f * (X - 2) ∧ 
                             (X^2 - 4*X + 4 : Polynomial Real) = g * (X - 2) := by sorry

-- Main theorem combining all correct statements
theorem correct_statements : 
  (∀ α β : Real, IsVerticalAngle α β → α = β) ∧
  (∀ α β : Real, α = β → IsRightAngle α ∨ IsRightAngle β) ∧
  (∃ f g : Polynomial Real, ∀ a : Real, 
    (X^2 - 4 : Polynomial Real).eval a = f.eval a * (a - 2) ∧
    (X^2 - 4*X + 4 : Polynomial Real).eval a = g.eval a * (a - 2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_angles_equal_converse_right_angles_equal_common_factor_polynomials_correct_statements_l518_51833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l518_51871

-- Define the vectors a and b
noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.cos x ^ 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, 2)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- State the theorem
theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ 3) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ 0) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 3) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l518_51871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_dogwood_count_l518_51881

def dogwood_tree_count (initial planted_today planted_tomorrow : ℕ) : ℕ :=
  initial + planted_today + planted_tomorrow

theorem final_dogwood_count :
  dogwood_tree_count 39 41 20 = 100 := by
  unfold dogwood_tree_count
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_dogwood_count_l518_51881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_special_case_eccentricity_l518_51816

/-- Represents a hyperbola with semi-axes a and b, and focal distance c -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance from a focus to an asymptote of a hyperbola -/
noncomputable def focus_to_asymptote_distance (h : Hyperbola) : ℝ := 
  h.b * h.c / Real.sqrt (h.a^2 + h.b^2)

/-- Theorem: If the distance from the foci to the asymptotes equals the real semi-axis length,
    then the eccentricity is √2 -/
theorem hyperbola_special_case_eccentricity (h : Hyperbola) 
    (h_condition : focus_to_asymptote_distance h = h.a) : 
    eccentricity h = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_special_case_eccentricity_l518_51816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sign_sum_theorem_l518_51883

theorem sign_sum_theorem (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (a / abs a + b / abs b + c / abs c + d / abs d + (a * b * c * d) / abs (a * b * c * d)) ∈ ({5, 1, -3} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sign_sum_theorem_l518_51883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_value_l518_51824

/-- The smallest positive real number such that there exists a positive real
    number b where all roots of x^3 - ax^2 + bx - a^2 are real -/
noncomputable def a : ℝ := 9

/-- The unique positive real number b for the above a -/
noncomputable def b : ℝ := 27

theorem unique_b_value (x : ℝ) :
  (∀ r : ℝ, (x^3 - a*x^2 + b*x - a^2 = 0) → r ∈ Set.univ) ∧
  (∀ b' : ℝ, b' > 0 → b' ≠ b →
    ∃ r : ℂ, r.im ≠ 0 ∧ r^3 - a*r^2 + b'*r - a^2 = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_value_l518_51824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_measures_stability_variance_stability_correspondence_l518_51811

/-- A type representing a student's exam scores -/
def ExamScores := List Float

/-- A function to calculate the variance of a list of scores -/
noncomputable def variance (scores : ExamScores) : Float := sorry

/-- A function to determine if one set of scores is more stable than another -/
def isMoreStable (scoresA scoresB : ExamScores) : Prop :=
  variance scoresA < variance scoresB

/-- Theorem stating that variance is the appropriate measure for score stability -/
theorem variance_measures_stability (scoresA scoresB : ExamScores) :
  isMoreStable scoresA scoresB ↔ (variance scoresA < variance scoresB) := by sorry

/-- Proposition that student A has more stable scores than student B -/
def a_more_stable_than_b (scoresA scoresB : ExamScores) : Prop :=
  isMoreStable scoresA scoresB

/-- Theorem linking the mathematical definition to the problem statement -/
theorem variance_stability_correspondence (scoresA scoresB : ExamScores) :
  a_more_stable_than_b scoresA scoresB ↔ isMoreStable scoresA scoresB := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_measures_stability_variance_stability_correspondence_l518_51811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_parabola_l518_51875

/-- The line y = px + q is tangent to the parabola y = x^2 + px + q at the point (0, q) -/
theorem line_tangent_to_parabola (p q : ℝ) :
  let parabola := fun x : ℝ => x^2 + p*x + q
  let line := fun x : ℝ => p*x + q
  let tangent_point := (0 : ℝ)
  (∀ x : ℝ, parabola x ≥ line x) ∧
  (parabola tangent_point = line tangent_point) ∧
  (deriv parabola tangent_point = p) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_parabola_l518_51875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l518_51817

-- Define the center of the circle
noncomputable def center (k : ℝ) : ℝ × ℝ := (0, k)

-- Define the equations of the lines
def line1 (x : ℝ) : ℝ := x
def line2 (x : ℝ) : ℝ := -x
def line3 : ℝ := -6

-- Define the radius of the circle
noncomputable def radius : ℝ := 6 * Real.sqrt 2 + 6

-- Theorem statement
theorem circle_radius (k : ℝ) (h1 : k < -6) 
  (h2 : ∃ (x1 y1 : ℝ), (x1 - 0)^2 + (y1 - k)^2 = radius^2 ∧ y1 = line1 x1)
  (h3 : ∃ (x2 y2 : ℝ), (x2 - 0)^2 + (y2 - k)^2 = radius^2 ∧ y2 = line2 x2)
  (h4 : ∃ (x3 : ℝ), (x3 - 0)^2 + (line3 - k)^2 = radius^2) :
  radius = 6 * Real.sqrt 2 + 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l518_51817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_perimeter_ratio_l518_51841

/-- Two triangles are similar -/
structure SimilarTriangles (ABC DEF : Type) where
  similar : ABC → DEF → Prop

/-- The area of a triangle -/
noncomputable def area (T : Type) : ℝ := sorry

/-- The perimeter of a triangle -/
noncomputable def perimeter (T : Type) : ℝ := sorry

/-- Theorem: For similar triangles with area ratio 1:9, the perimeter ratio is 1:3 -/
theorem similar_triangles_perimeter_ratio 
  (ABC DEF : Type) 
  (h : SimilarTriangles ABC DEF) 
  (area_ratio : area ABC / area DEF = 1 / 9) : 
  perimeter ABC / perimeter DEF = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_perimeter_ratio_l518_51841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_distribution_l518_51829

/-- Represents an exam score distribution --/
structure ExamDistribution where
  mean : ℝ
  stdDev : ℝ

/-- Calculates the number of standard deviations a score is from the mean --/
noncomputable def standardDeviations (d : ExamDistribution) (score : ℝ) : ℝ :=
  (score - d.mean) / d.stdDev

theorem exam_score_distribution (d : ExamDistribution) :
  d.mean = 74 ∧ 
  standardDeviations d 58 = -2 →
  standardDeviations d 98 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_distribution_l518_51829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_sphere_radii_l518_51840

/-- Regular pentagonal pyramid with specific properties -/
structure RegularPentagonalPyramid where
  -- Base edges are of unit length
  base_edge : ℝ
  base_edge_unit : base_edge = 1
  -- When lateral faces are unfolded, they form a regular star pentagon
  forms_star_pentagon : Bool

/-- Calculate the radius of the inscribed sphere -/
noncomputable def inscribed_sphere_radius (p : RegularPentagonalPyramid) : ℝ :=
  Real.sqrt ((5 + Real.sqrt 5) / 40)

/-- Calculate the radius of the circumscribed sphere -/
noncomputable def circumscribed_sphere_radius (p : RegularPentagonalPyramid) : ℝ :=
  (1 / 4) * Real.sqrt (10 + 2 * Real.sqrt 5)

/-- Theorem stating the radii of inscribed and circumscribed spheres -/
theorem pyramid_sphere_radii (p : RegularPentagonalPyramid) :
  inscribed_sphere_radius p = Real.sqrt ((5 + Real.sqrt 5) / 40) ∧
  circumscribed_sphere_radius p = (1 / 4) * Real.sqrt (10 + 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_sphere_radii_l518_51840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ribbon_length_is_18_sqrt_5_l518_51892

/-- The length of a ribbon spiraling around a cylindrical post -/
noncomputable def ribbon_length (post_circumference : ℝ) (post_height : ℝ) (num_loops : ℕ) : ℝ :=
  let vertical_rise := post_height / num_loops
  let horizontal_distance := post_circumference
  let loop_length := Real.sqrt (vertical_rise ^ 2 + horizontal_distance ^ 2)
  num_loops * loop_length

/-- Theorem: The length of the ribbon is 18√5 feet -/
theorem ribbon_length_is_18_sqrt_5 :
  ribbon_length 6 18 6 = 18 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ribbon_length_is_18_sqrt_5_l518_51892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_missed_questions_l518_51828

theorem max_missed_questions (total_questions : ℕ) (passing_percentage : ℚ) : 
  total_questions = 50 → 
  passing_percentage = 3/4 → 
  (total_questions - Int.toNat (Nat.ceil (passing_percentage * total_questions))) = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_missed_questions_l518_51828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_left_load_velocity_l518_51874

/-- Represents a pulley system with two loads. -/
structure PulleySystem where
  v : ℝ  -- Velocity of the right load (m/s)
  inextensible : Bool  -- Whether the strings are inextensible
  weightless : Bool  -- Whether the strings are weightless
  rigid : Bool  -- Whether the lever is rigid

/-- Calculates the velocity of the left load given a pulley system. -/
noncomputable def leftLoadVelocity (system : PulleySystem) : ℝ :=
  if system.inextensible ∧ system.weightless ∧ system.rigid then
    -(7/2) * system.v
  else
    0  -- undefined for other cases

/-- Theorem stating the velocity of the left load in the given system. -/
theorem left_load_velocity (system : PulleySystem) 
  (h1 : system.v = 1)
  (h2 : system.inextensible)
  (h3 : system.weightless)
  (h4 : system.rigid) :
  leftLoadVelocity system = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_left_load_velocity_l518_51874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_feet_perpendiculars_l518_51825

noncomputable section

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the foci of the ellipse
def foci (a b : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let c := Real.sqrt (a^2 - b^2)
  (-c, 0, c, 0)

-- Define the locus of the feet of perpendiculars
def locus (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = a^2

-- Main theorem
theorem locus_of_feet_perpendiculars (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a > b) :
  ∀ x y : ℝ, ellipse a b x y →
  ∃ x' y' : ℝ, locus a x' y' := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_feet_perpendiculars_l518_51825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_l518_51867

noncomputable def f (α : ℝ) : ℝ := (Real.cos (Real.pi / 2 + α) * Real.cos (Real.pi - α)) / Real.sin (Real.pi + α)

theorem f_simplification (α : ℝ) (h : Real.sin α ≠ 0) : f α = -Real.cos α := by sorry

theorem f_specific_value (α : ℝ) 
  (h1 : Real.pi < α ∧ α < 3 * Real.pi / 2)  -- α is in the third quadrant
  (h2 : Real.cos (α - 3 * Real.pi / 2) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_l518_51867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_over_angle_decreasing_l518_51845

theorem sin_over_angle_decreasing {α β : ℝ} (h1 : 0 < α) (h2 : α < β) (h3 : β < π/2) :
  Real.sin α / α > Real.sin β / β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_over_angle_decreasing_l518_51845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_cube_l518_51853

theorem det_cube {n : Type*} [Fintype n] [DecidableEq n] (M : Matrix n n ℝ) (h : Matrix.det M = 3) : 
  Matrix.det (M^3) = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_cube_l518_51853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_tangent_l518_51836

/-- Given a line ax - by - 3 = 0 and a function f(x) = xe^x, 
    if the line is perpendicular to the tangent line of f at (1, e),
    then a/b = -1/(2e) -/
theorem perpendicular_line_tangent (a b : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = x * Real.exp x) →
  (∀ x y, a * x - b * y - 3 = 0) →
  (∃ k : ℝ, ∀ x, k * (x - 1) + Real.exp 1 = f x + (deriv f 1) * (x - 1)) →
  (a / b * (deriv f 1) = -1) →
  a / b = -1 / (2 * Real.exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_tangent_l518_51836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_interest_rate_proof_l518_51859

noncomputable def simple_interest_rate (principal amount : ℝ) (time : ℝ) : ℝ :=
  (amount - principal) / (principal * time) * 100

noncomputable def simple_interest_amount (principal rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time / 100)

theorem additional_interest_rate_proof (initial_deposit : ℝ) (initial_amount : ℝ) (desired_amount : ℝ) (time : ℝ) :
  initial_deposit = 8000 →
  initial_amount = 9200 →
  desired_amount = 9800 →
  time = 3 →
  let initial_rate := simple_interest_rate initial_deposit initial_amount time
  let additional_rate := simple_interest_rate initial_deposit desired_amount time - initial_rate
  additional_rate = 2.5 :=
by
  sorry

#check additional_interest_rate_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_interest_rate_proof_l518_51859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_coordinates_l518_51846

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the triangle ABC -/
def triangle_ABC (C : Point) :=
  let A : Point := ⟨3, 2⟩
  let B : Point := ⟨-1, 5⟩
  (C.x, C.y, A, B)

/-- Condition: Point C lies on the line 3x - y + 3 = 0 -/
def C_on_line (C : Point) : Prop :=
  3 * C.x - C.y + 3 = 0

/-- Calculates the area of a triangle given three points -/
noncomputable def triangle_area (A B C : Point) : ℝ :=
  (1/2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

/-- Main theorem: Given the conditions, C has one of two specific coordinate pairs -/
theorem C_coordinates :
  ∀ C : Point,
    C_on_line C →
    triangle_area ⟨3, 2⟩ ⟨-1, 5⟩ C = 10 →
    (C = ⟨-1, 0⟩ ∨ C = ⟨5/3, 8⟩) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_coordinates_l518_51846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_sum_l518_51850

theorem triangle_tangent_sum (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- angles are positive
  A + B + C = π ∧  -- sum of angles in a triangle
  2 * B = A + C →  -- given condition
  Real.tan (A/2) + Real.tan (C/2) + Real.sqrt 3 * Real.tan (A/2) * Real.tan (C/2) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_sum_l518_51850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_consecutive_integers_sum_digits_multiple_of_8_l518_51814

/-- The sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ :=
  sorry

/-- A function that checks if there exists a number in a list of n consecutive positive integers
    whose sum of digits is a multiple of 8 -/
def exists_multiple_of_8 (start : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k < n ∧ (sum_of_digits (start + k) % 8 = 0)

/-- The main theorem stating that 15 is the minimum value of n for which the property holds -/
theorem min_consecutive_integers_sum_digits_multiple_of_8 :
  (∀ start : ℕ, exists_multiple_of_8 start 15) ∧
  (∀ n : ℕ, n < 15 → ∃ start : ℕ, ¬exists_multiple_of_8 start n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_consecutive_integers_sum_digits_multiple_of_8_l518_51814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_theorem_l518_51812

noncomputable section

variable (x : ℝ)

/-- The number of cows needed to produce a certain amount of milk in a given time -/
def cows_needed (initial_cows : ℝ) (initial_milk : ℝ) (initial_days : ℝ) 
                (target_milk : ℝ) (target_days : ℝ) : ℝ :=
  (initial_cows * initial_days * target_milk) / (initial_milk * target_days)

theorem milk_production_theorem (x : ℝ) :
  cows_needed x (x + 2) (x + 1) (x + 4) (x + 3) = (x * (x + 1) * (x + 4)) / ((x + 2) * (x + 3)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_theorem_l518_51812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l518_51834

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 2 * Real.sqrt 3 ∧
  (2 * Real.sqrt 3 + t.b) * (Real.sin t.A - Real.sin t.B) = (t.c - t.b) * Real.sin t.C

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.A = π / 3 ∧ 
  (∀ s : Triangle, triangle_conditions s → s.a * s.b * Real.sin s.C / 2 ≤ 3 * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l518_51834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_64_sqrt_is_plus_minus_2_l518_51831

theorem cube_64_sqrt_is_plus_minus_2 (x : ℝ) : x^3 = 64 → Real.sqrt x = 2 ∨ Real.sqrt x = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_64_sqrt_is_plus_minus_2_l518_51831
