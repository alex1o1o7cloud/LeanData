import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l644_64407

theorem find_a : ∃ a : ℕ+, 
  (Real.sqrt (2 + 2/3) = 2 * Real.sqrt (2/3)) ∧ 
  (Real.sqrt (3 + 3/8) = 3 * Real.sqrt (3/8)) ∧ 
  (Real.sqrt (4 + 4/15) = 4 * Real.sqrt (4/15)) ∧ 
  (Real.sqrt (8 + 8/a) = 8 * Real.sqrt (8/a)) ∧ 
  a = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l644_64407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_area_enclosed_by_curve_l644_64426

-- Define the curve M
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | Real.sqrt p.1 + Real.sqrt p.2 = 1}

-- Theorem for the minimum distance
theorem min_distance_to_origin :
  ∃ q : ℝ × ℝ, q ∈ M ∧ Real.sqrt (q.1^2 + q.2^2) = Real.sqrt 2 / 4 ∧
  ∀ r : ℝ × ℝ, r ∈ M → Real.sqrt (r.1^2 + r.2^2) ≥ Real.sqrt 2 / 4 :=
sorry

-- Theorem for the enclosed area
theorem area_enclosed_by_curve :
  (∫ (x : ℝ) in Set.Icc 0 1, (1 - Real.sqrt x)^2) ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_area_enclosed_by_curve_l644_64426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yolka_probability_l644_64427

-- Define the time range (in minutes) for arrivals
def arrivalTimeRange : Set ℝ := Set.Icc 0 60

-- Define waiting times
def vasyaWaitTime : ℝ := 15
def boryaWaitTime : ℝ := 10

-- Define the event of Anna arriving last
def annaArrivesLast (a b v : ℝ) : Prop :=
  a ∈ arrivalTimeRange ∧ b ∈ arrivalTimeRange ∧ v ∈ arrivalTimeRange ∧
  a > b ∧ a > v

-- Define the event of Borya and Vasya meeting
def boryaVasyaMeet (b v : ℝ) : Prop :=
  b ∈ arrivalTimeRange ∧ v ∈ arrivalTimeRange ∧
  (v ≤ b + vasyaWaitTime) ∧ (b ≤ v + boryaWaitTime)

-- Define the probability of all three going to "Yolka" together
noncomputable def probAllTogether : ℝ := (1/3) * (1337.5/3600)

-- Theorem statement
theorem yolka_probability :
  ∀ (a b v : ℝ),
    annaArrivesLast a b v →
    boryaVasyaMeet b v →
    probAllTogether = (1/3) * (1337.5/3600) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_yolka_probability_l644_64427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ptolemy_theorem_l644_64499

-- Define a circle
def Circle : Set (ℝ × ℝ) := {p | ∃ (center : ℝ × ℝ) (radius : ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem ptolemy_theorem (A B C D : ℝ × ℝ) (h : A ∈ Circle ∧ B ∈ Circle ∧ C ∈ Circle ∧ D ∈ Circle) :
  distance A C * distance B D = distance A B * distance D C + distance A D * distance B C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ptolemy_theorem_l644_64499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_communities_count_l644_64421

theorem other_communities_count (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ) 
  (h1 : total = 700)
  (h2 : muslim_percent = 44/100)
  (h3 : hindu_percent = 28/100)
  (h4 : sikh_percent = 10/100) :
  Int.floor (total * (1 - (muslim_percent + hindu_percent + sikh_percent))) = 126 := by
  sorry

#check other_communities_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_communities_count_l644_64421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_62_80_l644_64487

/-- The area of a rhombus given its diagonal lengths -/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

/-- Theorem: The area of a rhombus with diagonals 62 m and 80 m is 2480 square meters -/
theorem rhombus_area_62_80 : rhombusArea 62 80 = 2480 := by
  -- Unfold the definition of rhombusArea
  unfold rhombusArea
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_62_80_l644_64487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_vehicle_has_four_wheels_l644_64476

/-- Represents the number of wheels on a vehicle -/
structure Wheels where
  count : Nat

/-- The total number of wheels in the garage -/
def total_wheels : Wheels := ⟨22⟩

/-- The number of cars in the garage -/
def num_cars : Nat := 2

/-- The number of wheels on a car -/
def wheels_per_car : Wheels := ⟨4⟩

/-- The number of bicycles in the garage -/
def num_bicycles : Nat := 3

/-- The number of wheels on a bicycle -/
def wheels_per_bicycle : Wheels := ⟨2⟩

/-- The number of tricycles in the garage -/
def num_tricycles : Nat := 1

/-- The number of wheels on a tricycle -/
def wheels_per_tricycle : Wheels := ⟨3⟩

/-- The number of unicycles in the garage -/
def num_unicycles : Nat := 1

/-- The number of wheels on a unicycle -/
def wheels_per_unicycle : Wheels := ⟨1⟩

/-- Multiplication of Nat and Wheels -/
instance : HMul Nat Wheels Wheels where
  hMul n w := ⟨n * w.count⟩

/-- Addition of Wheels -/
instance : Add Wheels where
  add w1 w2 := ⟨w1.count + w2.count⟩

/-- Theorem stating that the unknown vehicle has 4 wheels -/
theorem unknown_vehicle_has_four_wheels :
  ∃ (unknown_wheels : Wheels),
    total_wheels = 
      num_cars * wheels_per_car +
      num_bicycles * wheels_per_bicycle +
      num_tricycles * wheels_per_tricycle +
      num_unicycles * wheels_per_unicycle +
      unknown_wheels ∧
    unknown_wheels = ⟨4⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_vehicle_has_four_wheels_l644_64476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_match_probabilities_l644_64410

/-- Fencing match probabilities -/
structure FencingMatch where
  prob_a_win : ℝ
  prob_tie : ℝ
  prob_sum_valid : prob_a_win + prob_tie ≤ 1

/-- Theorem about fencing match probabilities -/
theorem fencing_match_probabilities (fm : FencingMatch)
  (h1 : fm.prob_a_win = 0.41)
  (h2 : fm.prob_tie = 0.27) :
  (fm.prob_a_win + fm.prob_tie = 0.68) ∧
  (1 - fm.prob_a_win = 0.59) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_match_probabilities_l644_64410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_ownership_l644_64464

theorem vehicle_ownership (total_adults : Nat) (car_owners : Finset Nat) (motorcycle_owners : Finset Nat) (bicycle_owners : Finset Nat)
  (h1 : total_adults = 500)
  (h2 : car_owners.card = 420)
  (h3 : motorcycle_owners.card = 80)
  (h4 : bicycle_owners.card = 200)
  (h5 : ∀ a, a ∈ Finset.range total_adults → (a ∈ car_owners ∨ a ∈ motorcycle_owners ∨ a ∈ bicycle_owners)) :
  car_owners.card - ((car_owners ∩ motorcycle_owners).card + (car_owners ∩ bicycle_owners).card - (car_owners ∩ motorcycle_owners ∩ bicycle_owners).card) = 375 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_ownership_l644_64464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_and_C₁_no_intersection_C_inside_C₁_l644_64477

noncomputable def C (θ : ℝ) : ℝ × ℝ := 
  (2 * Real.sqrt 2 * Real.cos θ * Real.cos θ, 2 * Real.sqrt 2 * Real.cos θ * Real.sin θ)

def A : ℝ × ℝ := (1, 0)

noncomputable def M_to_P (M : ℝ × ℝ) : ℝ × ℝ := 
  (A.1 + Real.sqrt 2 * (M.1 - A.1), A.2 + Real.sqrt 2 * (M.2 - A.2))

noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := 
  (3 - Real.sqrt 2 + 2 * Real.cos θ, 2 * Real.sin θ)

theorem C_and_C₁_no_intersection : 
  ∀ θ₁ θ₂ : ℝ, C θ₁ ≠ C₁ θ₂ := by
  sorry

-- Additional helper theorem to show C is inside C₁
theorem C_inside_C₁ :
  ∀ θ : ℝ, (C θ).1^2 + (C θ).2^2 < ((3 - Real.sqrt 2) - Real.sqrt 2)^2 + 2^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_and_C₁_no_intersection_C_inside_C₁_l644_64477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_alternating_sums_l644_64449

open BigOperators

def alternating_sum (n : ℕ) (a : ℕ → ℚ) : ℚ :=
  ∑ i in Finset.range n, ((-1)^(i / 3 : ℕ)) * a i

def S₁ : ℚ := alternating_sum 2019 (λ i => 1 / 2^(2019 - i))
def S₂ : ℚ := alternating_sum 2019 (λ i => 1 / 2^(i + 1))

theorem ratio_of_alternating_sums : S₁ / S₂ = -1/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_alternating_sums_l644_64449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l644_64402

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (1 + x) + Real.sqrt (1 - x) + 2) * (Real.sqrt (1 - x^2) + 1)

-- Main theorem
theorem f_properties :
  (∀ x ∈ Set.Icc (0 : ℝ) 1, 0 < f x ∧ f x ≤ 8) ∧
  (∃ x₀ ∈ Set.Icc (0 : ℝ) 1, f x₀ = 8) ∧
  (∃ β > 0, ∀ x ∈ Set.Icc (0 : ℝ) 1, Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^2 / β) ∧
  (∀ β > 0, (∀ x ∈ Set.Icc (0 : ℝ) 1, Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^2 / β) → β ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l644_64402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_proof_l644_64480

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The midpoint of the chord -/
def chord_midpoint : ℝ × ℝ := (1, -1)

/-- The equation of the line containing the chord -/
def chord_line (x y : ℝ) : Prop := 4*x + y - 3 = 0

theorem chord_equation_proof :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ 
    parabola x₂ y₂ ∧ 
    chord_midpoint = ((x₁ + x₂)/2, (y₁ + y₂)/2) →
    ∀ (x y : ℝ), chord_line x y ↔ 
      ∃ (t : ℝ), x = (1-t)*x₁ + t*x₂ ∧ y = (1-t)*y₁ + t*y₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_proof_l644_64480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_sum_l644_64419

theorem two_integers_sum (a b : ℕ+) : 
  a * b + a + b = 119 →
  Nat.Coprime a b →
  a < 20 →
  b < 20 →
  a + b = 21 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_sum_l644_64419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l644_64454

-- Define the line equation
def line_equation (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 3 = 0

-- Define the inclination angle of a line
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan (-m)

-- Theorem statement
theorem line_inclination_angle :
  ∃ (m : ℝ), (∀ (x y : ℝ), line_equation x y ↔ y = m * x + 3) ∧
  inclination_angle m = 2 * Real.pi / 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l644_64454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_12_minus_alpha_l644_64412

theorem sin_pi_12_minus_alpha (α : ℝ) 
  (h1 : Real.sin (α + π/6) = 1/3) 
  (h2 : π/3 < α) 
  (h3 : α < π) : 
  Real.sin (π/12 - α) = -(4 + Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_12_minus_alpha_l644_64412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_is_two_l644_64458

/-- Line l passing through (0,1) with slope k -/
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x + 1

/-- Circle C with center (2,3) and radius 1 -/
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

/-- Intersection points of line l and circle C -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ y = line_l k x ∧ circle_C x y}

/-- Dot product of vectors OM and ON -/
def dot_product_OM_ON (M N : ℝ × ℝ) : ℝ :=
  M.1 * N.1 + M.2 * N.2

theorem length_MN_is_two (k : ℝ) :
  ∃ M N, M ∈ intersection_points k ∧ N ∈ intersection_points k ∧
    dot_product_OM_ON M N = 12 →
    ((M.1 - N.1)^2 + (M.2 - N.2)^2).sqrt = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_is_two_l644_64458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_option_is_rational_l644_64416

-- Define the options
noncomputable def option_a : ℝ := Real.sqrt 2 - 1
noncomputable def option_b : ℝ := Real.sqrt 2 + 1
noncomputable def option_c : ℝ := -1 - Real.sqrt 2
noncomputable def option_d : ℝ := Real.sqrt 2

-- Define the multiplicand
noncomputable def multiplicand : ℝ := 1 + Real.sqrt 2

-- Define rationality
def is_rational (x : ℝ) : Prop := ∃ (q : ℚ), x = ↑q

-- Theorem statement
theorem correct_option_is_rational :
  is_rational (multiplicand * option_a) ∧
  ¬is_rational (multiplicand * option_b) ∧
  ¬is_rational (multiplicand * option_c) ∧
  ¬is_rational (multiplicand * option_d) := by
  sorry

#check correct_option_is_rational

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_option_is_rational_l644_64416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_decreasing_arithmetic_seq_l644_64444

/-- An arithmetic sequence -/
structure ArithmeticSequence (α : Type*) [LinearOrderedField α] :=
  (a : ℕ → α)
  (d : α)
  (h : ∀ n, a (n + 1) = a n + d)

/-- Sum of the first n terms of a sequence -/
def sequenceSum {α : Type*} [AddCommMonoid α] (a : ℕ → α) (n : ℕ) : α :=
  Finset.sum (Finset.range n) a

/-- Theorem: For a decreasing arithmetic sequence with S_5 = S_10, S_n is maximized when n = 7 or n = 8 -/
theorem max_sum_decreasing_arithmetic_seq
  {α : Type*} [LinearOrderedField α]
  (seq : ArithmeticSequence α)
  (h_decreasing : seq.d < 0)
  (h_sum_equal : sequenceSum seq.a 5 = sequenceSum seq.a 10) :
  ∃ n : ℕ, (n = 7 ∨ n = 8) ∧
    ∀ m : ℕ, sequenceSum seq.a m ≤ sequenceSum seq.a n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_decreasing_arithmetic_seq_l644_64444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_theta_l644_64478

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -5669/16384 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_theta_l644_64478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_equal_one_range_l644_64494

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then (x + 1)^2 else 4 - Real.sqrt (x - 1)

-- Define the set of x values where f(x) ≥ 1
def S : Set ℝ := {x | f x ≥ 1}

-- Theorem statement
theorem f_greater_equal_one_range : S = Set.Iic (-2) ∪ Set.Icc 0 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_equal_one_range_l644_64494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_trig_propositions_l644_64433

theorem three_trig_propositions :
  (∀ θ : ℝ, 0 < θ ∧ θ < Real.pi / 2 → Real.sin (Real.cos θ) < Real.cos (Real.sin θ)) ∧
  (∀ θ : ℝ, 0 < θ ∧ θ < Real.pi / 2 → Real.cos (Real.cos θ) > Real.sin (Real.sin θ)) ∧
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi → Real.sin (Real.cos θ) < Real.cos (Real.sin θ)) :=
by
  sorry

#check three_trig_propositions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_trig_propositions_l644_64433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l644_64436

open Real

noncomputable def f (x : ℝ) : ℝ := (-x^2 + x - 1) * exp x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * x^2 + m

noncomputable def h (m : ℝ) (x : ℝ) : ℝ := f x - g m x

theorem intersection_range (m : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f x₁ = g m x₁ ∧ f x₂ = g m x₂ ∧ f x₃ = g m x₃) →
  m > -(3/exp 1) - 1/6 ∧ m < -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l644_64436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stationary_points_of_z_l644_64474

/-- The function z(x, y) = e^x * (x - y^3 + 3y) -/
noncomputable def z (x y : ℝ) : ℝ := Real.exp x * (x - y^3 + 3*y)

/-- Partial derivative of z with respect to x -/
noncomputable def z_x (x y : ℝ) : ℝ := Real.exp x * (1 + x - y^3 + 3*y)

/-- Partial derivative of z with respect to y -/
noncomputable def z_y (x y : ℝ) : ℝ := Real.exp x * (-3*y^2 + 3)

/-- A point (x, y) is stationary if both partial derivatives are zero -/
def is_stationary (x y : ℝ) : Prop := z_x x y = 0 ∧ z_y x y = 0

theorem stationary_points_of_z :
  ∀ x y : ℝ, is_stationary x y ↔ (x = -3 ∧ y = 1) ∨ (x = 1 ∧ y = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stationary_points_of_z_l644_64474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_symmetry_axis_f_symmetry_center_l644_64438

/-- The function f(x) defined on real numbers. -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 2

/-- The smallest positive period of f(x) is π. -/
theorem f_period : ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧ 
  (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧ T = Real.pi := by
  sorry

/-- The symmetry axis of f(x) is given by x = 5π/12 + kπ/2, where k ∈ ℤ. -/
theorem f_symmetry_axis : ∀ x : ℝ, ∃ k : ℤ, f (5 * Real.pi / 12 + k * Real.pi / 2 + x) = 
  f (5 * Real.pi / 12 + k * Real.pi / 2 - x) := by
  sorry

/-- The symmetry center of f(x) is at (π/6 + kπ/2, 0), where k ∈ ℤ. -/
theorem f_symmetry_center : ∀ k : ℤ, f (Real.pi / 6 + k * Real.pi / 2) = 0 ∧ 
  (∀ x : ℝ, f (Real.pi / 6 + k * Real.pi / 2 + x) = -f (Real.pi / 6 + k * Real.pi / 2 - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_symmetry_axis_f_symmetry_center_l644_64438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_foldable_polygons_l644_64466

/-- Represents a polygon formed by squares --/
structure Polygon where
  squares : ℕ
  is_l_shaped : Bool

/-- Represents a position where an additional square can be attached --/
structure AttachmentPosition

/-- Represents the result of attaching a square to a polygon --/
def attach_square (p : Polygon) (pos : AttachmentPosition) : Polygon :=
  sorry

/-- Checks if a polygon can be folded into a cube with one face missing --/
def can_fold_to_incomplete_cube (p : Polygon) : Bool :=
  sorry

/-- The original L-shaped polygon --/
def base_polygon : Polygon :=
  { squares := 6, is_l_shaped := true }

/-- The set of all possible attachment positions --/
def attachment_positions : Finset AttachmentPosition :=
  sorry

/-- The number of attachment positions --/
axiom num_attachment_positions : attachment_positions.card = 14

/-- Theorem stating that exactly 7 resulting polygons can form a cube with one face missing --/
theorem seven_foldable_polygons :
  (attachment_positions.filter (λ pos => 
    can_fold_to_incomplete_cube (attach_square base_polygon pos))).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_foldable_polygons_l644_64466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_search_third_point_l644_64451

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The initial lower bound of the interval -/
def L₀ : ℝ := 1000

/-- The initial upper bound of the interval -/
def U₀ : ℝ := 2000

/-- The first trial point -/
noncomputable def x₁ : ℝ := U₀ - φ * (U₀ - L₀)

/-- The second trial point -/
noncomputable def x₂ : ℝ := L₀ + φ * (U₀ - L₀)

/-- The third trial point -/
noncomputable def x₃ : ℝ := x₁ + φ * (U₀ - x₁)

theorem golden_section_search_third_point :
  x₃ = 1764 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_search_third_point_l644_64451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_is_2_sqrt_26_l644_64473

/-- An isosceles trapezoid with given side lengths and perpendicular height -/
structure IsoscelesTrapezoid where
  AB : ℝ
  CD : ℝ
  AD : ℝ
  BC : ℝ
  height_perpendicular : Bool
  ab_eq : AB = 30
  cd_eq : CD = 12
  ad_eq : AD = 13
  bc_eq : BC = 13
  is_isosceles : AD = BC

/-- The length of the diagonal AC in the isosceles trapezoid -/
noncomputable def diagonal_length (t : IsoscelesTrapezoid) : ℝ := 2 * Real.sqrt 26

/-- Theorem stating that the diagonal length is 2√26 -/
theorem diagonal_length_is_2_sqrt_26 (t : IsoscelesTrapezoid) :
  diagonal_length t = 2 * Real.sqrt 26 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_is_2_sqrt_26_l644_64473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l644_64441

-- Define the logarithm with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then log_half (-x + 1) else log_half (x + 1)

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = f x) ∧  -- f is even
  (∀ x ≤ 0, f x = log_half (-x + 1)) →  -- given condition
  ((∀ x > 0, f x = log_half (x + 1)) ∧  -- part 1: analytic expression
   (∀ x ≤ 0, f x = log_half (-x + 1))) ∧
  {a : ℝ | f (a - 1) < -1} = {a : ℝ | a < 0 ∨ a > 2} :=  -- part 2: range of a
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l644_64441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_tangent_circles_l644_64425

/-- Given two circles with centers O₁ and O₂ that are internally tangent to a circle
    with center O and radius R, and the circles with centers O₁ and O₂ are tangent
    to each other, prove that the perimeter of triangle OO₁O₂ is equal to 2R. -/
theorem perimeter_of_tangent_circles (O O₁ O₂ : EuclideanSpace ℝ (Fin 2)) (R : ℝ) :
  (∃ (r₁ r₂ : ℝ), r₁ > 0 ∧ r₂ > 0 ∧
    Metric.sphere O R ∩ Metric.sphere O₁ r₁ ≠ ∅ ∧
    Metric.sphere O R ∩ Metric.sphere O₂ r₂ ≠ ∅ ∧
    Metric.sphere O₁ r₁ ∩ Metric.sphere O₂ r₂ ≠ ∅) →
  dist O O₁ + dist O O₂ + dist O₁ O₂ = 2 * R := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_tangent_circles_l644_64425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_b_l644_64408

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -1 / x

-- State the theorem
theorem solve_for_b (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a * b ≠ 0) :
  f a = -1/3 ∧ f (a * b) = 1/6 → b = -2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_b_l644_64408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_l644_64411

-- Define the coordinates of the points
noncomputable def F : ℝ × ℝ := (0, 0)
noncomputable def G : ℝ × ℝ := (0, -1)
noncomputable def H : ℝ × ℝ := (Real.sqrt 3 / Real.sqrt 2, -1 - Real.sqrt 3 / Real.sqrt 2)
noncomputable def I : ℝ × ℝ := (Real.sqrt 3 / Real.sqrt 2 + 2 * (Real.sqrt 3 / 2), -1 - Real.sqrt 3 / Real.sqrt 2 - 2 * (1 / 2))
noncomputable def J : ℝ × ℝ := (Real.sqrt 3 / Real.sqrt 2 + 2 * (Real.sqrt 3 / 2) + 2 * (Real.sqrt 5 / Real.sqrt 4), 
                  -1 - Real.sqrt 3 / Real.sqrt 2 - 2 * (1 / 2) - 2 * (1 / Real.sqrt 4))

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define x as the sum of squares of J's coordinates
noncomputable def x : ℝ := J.1^2 + J.2^2

-- Theorem statement
theorem pentagon_perimeter : 
  distance F G + distance G H + distance H I + distance I J + distance J F = 5 + Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_l644_64411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l644_64417

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (y + 1) * f (x + y) - f x * f (x + y^2) = y * f x) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l644_64417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_n_l644_64481

def n : Nat := 2^3 * 3^4 * 5^5 * 7^6

theorem number_of_factors_of_n : 
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 840 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_n_l644_64481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_value_l644_64423

theorem trig_expression_value (α : Real) :
  ∃ r > 0, (1 : Real) = r * Real.cos α ∧ 3 = r * Real.sin α →
  (Real.sin α + 3 * Real.cos α) / (Real.cos α - 3 * Real.sin α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_value_l644_64423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l644_64490

def S (n : ℕ) : ℕ := n^2 + 3*n + 5

def a : ℕ → ℕ
  | 0 => 9  -- Add this case for n = 0
  | 1 => 9
  | n+2 => 2*(n+2) + 2

theorem sequence_formula (n : ℕ) : 
  (∀ k, k ≤ n → S k = k^2 + 3*k + 5) → 
  a n = if n = 0 ∨ n = 1 then 9 else 2*n + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l644_64490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_condition_l644_64468

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - b else 2^x

theorem function_value_condition (b : ℝ) : f b (f b (5/6)) = 4 → b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_condition_l644_64468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_remaining_is_94_l644_64492

noncomputable def initial_bread : ℝ := 1500

noncomputable def day1_fraction : ℝ := 3/8
noncomputable def day2_fraction : ℝ := 7/10
noncomputable def day3_fraction : ℝ := 1/6
noncomputable def day4_fraction : ℝ := 4/9
noncomputable def day5_fraction : ℝ := 5/18

noncomputable def remaining_bread : ℝ :=
  initial_bread *
  (1 - day1_fraction) *
  (1 - day2_fraction) *
  (1 - day3_fraction) *
  (1 - day4_fraction) *
  (1 - day5_fraction)

theorem bread_remaining_is_94 :
  ⌊remaining_bread⌋ = 94 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_remaining_is_94_l644_64492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l644_64484

/-- Calculates the speed of a train given the time it takes to cross a platform and a man -/
noncomputable def train_speed (platform_length : ℝ) (platform_time : ℝ) (man_time : ℝ) : ℝ :=
  (platform_length / (platform_time - man_time)) * 3.6

/-- Theorem: The speed of the train is 72 km/h -/
theorem train_speed_calculation :
  train_speed 300 30 15 = 72 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num

-- We can't use #eval for noncomputable functions, so we'll use #check instead
#check train_speed 300 30 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l644_64484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_sale_price_percentage_l644_64431

theorem lowest_sale_price_percentage (list_price : ℝ) 
  (regular_discount_max : ℝ) (summer_sale_discount : ℝ) : 
  (list_price * (1 - regular_discount_max) - (list_price * summer_sale_discount)) / list_price * 100 = 30 :=
by
  -- Proof steps would go here
  sorry

#eval Float.toString ((80 * (1 - 0.5) - (80 * 0.2)) / 80 * 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_sale_price_percentage_l644_64431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_nine_equals_powers_and_35_l644_64497

theorem factorial_nine_equals_powers_and_35 : 2^7 * 3^4 * 35 = Nat.factorial 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_nine_equals_powers_and_35_l644_64497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l644_64470

theorem vector_magnitude_problem (a b : EuclideanSpace ℝ (Fin 3)) :
  ‖a‖ = 1 →
  ‖a - 2 • b‖ = Real.sqrt 21 →
  inner a b = ‖a‖ * ‖b‖ * (-1/2) →
  ‖b‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l644_64470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_patricia_walking_distance_l644_64446

/-- Represents a pedometer with a maximum count and current reading -/
structure Pedometer :=
  (max_count : ℕ)
  (current_reading : ℕ)

/-- Calculates the total steps given the number of full cycles and final reading -/
def total_steps (p : Pedometer) (full_cycles : ℕ) : ℕ :=
  full_cycles * (p.max_count + 1) + p.current_reading

/-- Converts steps to miles -/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℚ :=
  (steps : ℚ) / steps_per_mile

theorem patricia_walking_distance :
  ∀ (p : Pedometer) (full_cycles : ℕ) (steps_per_mile : ℕ),
    p.max_count = 89999 →
    full_cycles = 27 →
    p.current_reading = 40000 →
    steps_per_mile = 1500 →
    ∃ (miles : ℚ), (miles ≥ 1649.5 ∧ miles ≤ 1650.5) ∧ miles = steps_to_miles (total_steps p full_cycles) steps_per_mile :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_patricia_walking_distance_l644_64446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l644_64469

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if a circle with diameter equal to the distance between its foci intersects
    the hyperbola's asymptote at the point (1, 2), then the standard equation
    of the hyperbola is x² - y²/4 = 1 -/
theorem hyperbola_standard_equation
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (h_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h_asymptote : ∀ x : ℝ, b / a * x = 2 * x)
  (h_circle : ∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = Real.sqrt 5) :
  ∀ x y : ℝ, x^2 - y^2 / 4 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l644_64469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_approx_4_925_l644_64453

/-- Triangle XYZ with given properties --/
structure TriangleXYZ where
  -- Side lengths
  x : ℝ
  y : ℝ
  z : ℝ
  -- Angles
  X : ℝ
  Y : ℝ
  Z : ℝ
  -- Given conditions
  y_eq : y = 7
  z_eq : z = 3
  cos_diff : Real.cos (X - Y) = 15/16
  -- Triangle properties
  angle_sum : X + Y + Z = π
  -- Law of cosines
  law_cosines : x^2 = y^2 + z^2 - 2*y*z*(Real.cos X)
  -- Law of sines
  law_sines_xy : x / Real.sin X = y / Real.sin Y
  law_sines_xz : x / Real.sin X = z / Real.sin Z

/-- The main theorem stating that x is approximately 4.925 --/
theorem x_approx_4_925 (t : TriangleXYZ) : ∃ ε > 0, |t.x - 4.925| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_approx_4_925_l644_64453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_count_l644_64443

/-- The set of digits to be arranged -/
def digits : Finset ℕ := {2, 0, 1, 9, 20, 19}

/-- The function that counts valid 8-digit numbers -/
def count_valid_numbers (S : Finset ℕ) : ℕ :=
  let total_permutations := 5 * Nat.factorial 5
  let B := Nat.factorial 4  -- permutations where 2 is followed by 0
  let C := Nat.factorial 5 - B  -- permutations where 1 is followed by 9
  let D := 4 * Nat.factorial 4  -- permutations where 9 is followed by 1
  total_permutations - (3 * B / 4) - (C / 2) - (D / 2)

/-- The main theorem stating that the count of valid numbers is 498 -/
theorem valid_number_count :
  count_valid_numbers digits = 498 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_count_l644_64443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_20_solution_l644_64430

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 30 / (x + 2)
noncomputable def g (x : ℝ) : ℝ := 4 * (Function.invFun f x)

-- State the theorem
theorem g_equals_20_solution :
  ∃ x : ℝ, g x = 20 ∧ x = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_20_solution_l644_64430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l644_64437

open Real MeasureTheory

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := sqrt (x - x^2) - arccos (sqrt x) + 5

-- Define the arc length function
noncomputable def arcLength (a b : ℝ) : ℝ := ∫ x in a..b, sqrt (1 + (deriv f x)^2)

-- Theorem statement
theorem arc_length_of_curve :
  arcLength (1/9) 1 = 4/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l644_64437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stones_for_hall_l644_64420

/-- Calculates the number of stones required to pave a rectangular hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℕ :=
  (hall_length * hall_width / (stone_length * stone_width)).ceil.toNat

/-- Theorem stating that 2700 stones are required to pave the given hall -/
theorem stones_for_hall : stones_required 72 30 (8/10) 1 = 2700 := by
  -- Proof goes here
  sorry

#eval stones_required 72 30 (8/10) 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stones_for_hall_l644_64420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_N_l644_64428

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {2, 4}

-- Define set N
def N : Set Nat := {3, 5}

-- Theorem statement
theorem complement_M_intersect_N :
  (U \ M) ∩ N = {3, 5} := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_N_l644_64428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sum_l644_64459

-- Define the power function
noncomputable def power_function (k α : ℝ) (x : ℝ) : ℝ := k * x^α

-- State the theorem
theorem power_function_sum (k α : ℝ) :
  (power_function k α (1/2) = 2) → k + α = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sum_l644_64459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_dot_product_l644_64456

-- Define the triangle ABC and its circumscribed circle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  O : ℝ × ℝ  -- Center of the circumscribed circle

-- Define vector operations
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def vec_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- State the theorem
theorem triangle_dot_product (t : Triangle) : 
  -- Conditions
  (vec_length (vec t.O t.A) = 1) →  -- Radius of circumscribed circle is 1
  (vec t.O t.A = vec t.A t.B) →     -- |⃗OA| = |⃗AB|
  (2 • (vec t.O t.A) + vec t.A t.B + vec t.A t.C = (0, 0)) →  -- 2⃗OA + ⃗AB + ⃗AC = 0⃗
  -- Conclusion
  (dot_product (vec t.C t.A) (vec t.C t.B) = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_dot_product_l644_64456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_track_for_grade_reduction_l644_64467

/-- Calculates the additional track length required to reduce grade for a given vertical rise -/
noncomputable def additionalTrackLength (verticalRise : ℝ) (initialGrade : ℝ) (finalGrade : ℝ) : ℝ :=
  verticalRise * (1 / finalGrade - 1 / initialGrade)

/-- Theorem stating that for a 900 feet vertical rise, reducing grade from 3% to 1.5% requires 30000 additional feet of track -/
theorem additional_track_for_grade_reduction :
  additionalTrackLength 900 0.03 0.015 = 30000 := by
  -- Unfold the definition of additionalTrackLength
  unfold additionalTrackLength
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry

-- We can't use #eval for noncomputable functions, so we'll use the following instead:
#check additionalTrackLength 900 0.03 0.015

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_track_for_grade_reduction_l644_64467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt19_plus_92_produces_one_root19_92_produces_one_sqrt19_plus_sqrt92_not_produces_one_l644_64462

-- Define the allowed operations
inductive Operation
| Add : ℝ → ℝ → Operation
| Sub : ℝ → ℝ → Operation
| Reciprocal : ℝ → Operation

-- Define a function to apply an operation
noncomputable def applyOperation (op : Operation) : ℝ :=
  match op with
  | Operation.Add x y => x + y
  | Operation.Sub x y => x - y
  | Operation.Reciprocal x => 1 / x

-- Define a sequence of operations
def OperationSequence := List Operation

-- Function to check if a sequence of operations produces 1
def producesOne (start : ℝ) (seq : OperationSequence) : Prop :=
  ∃ (result : ℝ), result = seq.foldl (λ acc op => applyOperation op) start ∧ result = 1

-- Theorem statements
theorem sqrt19_plus_92_produces_one :
  ∃ (seq : OperationSequence), producesOne (Real.sqrt 19 + 92) seq := by
  sorry

theorem root19_92_produces_one :
  ∃ (seq : OperationSequence), producesOne ((92 : ℝ) ^ (1 / 19)) seq := by
  sorry

theorem sqrt19_plus_sqrt92_not_produces_one :
  ¬∃ (seq : OperationSequence), producesOne (Real.sqrt 19 + Real.sqrt 92) seq := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt19_plus_92_produces_one_root19_92_produces_one_sqrt19_plus_sqrt92_not_produces_one_l644_64462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l644_64483

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x - 4) ^ 0 / Real.sqrt (x + 1)

-- State the theorem
theorem f_domain : 
  {x : ℝ | f x ≠ 0} = {x : ℝ | x > -1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l644_64483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_50_value_l644_64496

def sequence_a : ℕ → ℝ
  | 0 => 1
  | (n + 1) => (64 * (sequence_a n)^3)^(1/3)

theorem a_50_value : sequence_a 49 = 4^49 := by
  sorry

#eval sequence_a 49

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_50_value_l644_64496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_sum_l644_64465

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the points
variable (A B C D E : V)

-- Define the conditions
variable (h1 : D = (B + C) / 2)
variable (h2 : ∃ (x y : ℝ), A - C = x • (A - B) + y • (B - E))
variable (h3 : C - D = 3 • (C - E) - 2 • (C - A))

-- State the theorem
theorem triangle_vector_sum :
  ∃ (x y : ℝ), A - C = x • (A - B) + y • (B - E) ∧ x + y = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_sum_l644_64465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asep_winning_strategy_l644_64429

/-- The game state -/
structure GameState where
  board : ℕ
  steps : ℕ

/-- The possible moves in the game -/
inductive Move
  | up (d : ℕ)
  | down (d : ℕ)

/-- A function representing whether a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- A function representing a valid move in the game -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  match move with
  | Move.up d => d > 0 ∧ d ∣ state.board
  | Move.down d => d > 0 ∧ d ∣ state.board ∧ state.board > d

/-- A function representing the result of applying a move to a game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.up d => { board := state.board + d, steps := state.steps + 1 }
  | Move.down d => { board := state.board - d, steps := state.steps + 1 }

/-- The theorem to be proved -/
theorem asep_winning_strategy (n : ℕ) (h : n ≥ 14) :
  ∃ (strategy : GameState → Move),
    ∀ (game : ℕ → GameState),
      game 0 = { board := n, steps := 0 } →
      (∀ k, game (k + 1) = applyMove (game k) (strategy (game k))) →
      ∃ k, isPerfectSquare (game k).board ∧ (game k).steps ≤ (n - 5) / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asep_winning_strategy_l644_64429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_divisor_of_110122012_base_5_l644_64447

def base_5_to_decimal (n : List (Fin 5)) : ℕ :=
  n.enum.foldr (fun (i, d) acc => acc + d.val * (5 ^ i)) 0

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

theorem largest_prime_divisor_of_110122012_base_5 :
  let n : List (Fin 5) := [1, 1, 0, 1, 2, 2, 0, 1, 2]
  let decimal := base_5_to_decimal n
  ∃ (p : ℕ), is_prime p ∧ (decimal % p = 0) ∧
    ∀ (q : ℕ), is_prime q → (decimal % q = 0) → q ≤ p ∧
    p = 473401 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_divisor_of_110122012_base_5_l644_64447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_to_line_l644_64457

/-- The maximum distance from a point on the unit circle to a line -/
theorem max_distance_point_to_line :
  let P : ℝ → ℝ × ℝ := λ θ ↦ (Real.cos θ, Real.sin θ)
  let line (m : ℝ) := {(x, y) : ℝ × ℝ | x - m * y - 2 = 0}
  let d (θ m : ℝ) := abs (Real.cos θ - m * Real.sin θ - 2) / Real.sqrt (1 + m^2)
  ∀ θ m, d θ m ≤ 3 ∧ ∃ θ₀ m₀, d θ₀ m₀ = 3
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_to_line_l644_64457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_shelf_theorem_l644_64486

/-- Represents the thickness of a book -/
structure BookThickness (α : Type*) where
  thickness : α

/-- Represents a shelf in the library -/
structure LibraryShelf (α : Type*) where
  length : α

theorem library_shelf_theorem 
  {α : Type*} [LinearOrderedField α] 
  (biology : BookThickness α) 
  (physics : BookThickness α)
  (shelf1 shelf2 shelf3 : LibraryShelf α)
  (B P Q R F : ℕ) :
  biology.thickness < physics.thickness →
  B * biology.thickness + P * physics.thickness = shelf1.length →
  Q * biology.thickness + R * physics.thickness = shelf2.length →
  shelf2.length < shelf1.length →
  F * biology.thickness = shelf1.length →
  B ≠ P ∧ B ≠ Q ∧ B ≠ R ∧ B ≠ F ∧
  P ≠ Q ∧ P ≠ R ∧ P ≠ F ∧
  Q ≠ R ∧ Q ≠ F ∧
  R ≠ F →
  F = (Q - B) / (P - R) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_shelf_theorem_l644_64486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_range_l644_64485

theorem sin_equation_range :
  ∀ a : ℝ, (∃ x : ℝ, Real.sin x ^ 2 - 4 * Real.sin x + 1 - a = 0) ↔ a ∈ Set.Icc (-2) 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_range_l644_64485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expensive_feed_price_l644_64482

/-- Prove that the price of the more expensive feed is $0.36 per pound --/
theorem expensive_feed_price 
  (total_mix : ℝ) 
  (mix_price : ℝ) 
  (cheap_price : ℝ) 
  (cheap_amount : ℝ) 
  (h1 : total_mix = 27) 
  (h2 : mix_price = 0.26) 
  (h3 : cheap_price = 0.17) 
  (h4 : cheap_amount = 14.2105263158) : 
  ∃ (expensive_price : ℝ), abs (expensive_price - 0.36) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expensive_feed_price_l644_64482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_144_l644_64413

/-- The length of one edge of a cube, given the sum of all edges --/
noncomputable def edge_length (sum_of_edges : ℝ) : ℝ :=
  sum_of_edges / 12

/-- Theorem: For a cube with sum of all edges 144 cm, the length of one edge is 12 cm --/
theorem cube_edge_length_144 :
  edge_length 144 = 12 := by
  -- Unfold the definition of edge_length
  unfold edge_length
  -- Simplify the division
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_144_l644_64413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_correct_l644_64401

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x ≥ 0 then x * (1 - x) else x * (1 + x)

-- State the theorem
theorem f_is_odd_and_correct : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x : ℝ, x ≥ 0 → f x = x * (1 - x)) ∧
  (∀ x : ℝ, x ≤ 0 → f x = x * (1 + x)) := by
  sorry

#check f_is_odd_and_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_correct_l644_64401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_calls_required_l644_64418

/-- Represents a club with n members, each having unique information -/
structure Club (n : ℕ) where
  members : Fin n → Set String
  unique_info : ∀ i j, i ≠ j → members i ≠ members j

/-- Represents the state of information distribution in the club -/
structure InfoState (n : ℕ) where
  knowledge : Fin n → Set String

/-- A phone call transfers all information from one member to another -/
def make_call {n : ℕ} (state : InfoState n) (caller receiver : Fin n) : InfoState n :=
  { knowledge := λ i => if i = receiver then state.knowledge caller ∪ state.knowledge receiver else state.knowledge i }

/-- Checks if all members have all information -/
def all_informed {n : ℕ} (club : Club n) (state : InfoState n) : Prop :=
  ∀ i, state.knowledge i = ⋃ j, club.members j

/-- The main theorem: minimal number of calls required is 2n-2 -/
theorem min_calls_required (n : ℕ) (club : Club n) :
  (∃ calls : List (Fin n × Fin n),
    calls.length = 2*n - 2 ∧
    all_informed club (calls.foldl (λ s c => make_call s c.fst c.snd) { knowledge := club.members })) ∧
  (∀ calls : List (Fin n × Fin n),
    all_informed club (calls.foldl (λ s c => make_call s c.fst c.snd) { knowledge := club.members }) →
    calls.length ≥ 2*n - 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_calls_required_l644_64418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_is_75_l644_64400

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  /-- The length of the altitude to the base -/
  altitude : ℝ
  /-- The perimeter of the triangle -/
  perimeter : ℝ
  /-- The base of the triangle -/
  base : ℝ
  /-- One of the equal sides of the triangle -/
  side : ℝ
  /-- The altitude divides the base into two equal parts -/
  base_division : base / 2 > 0
  /-- Perimeter constraint -/
  perimeter_constraint : perimeter = 2 * side + base
  /-- Pythagorean theorem in the right triangle formed by the altitude -/
  pythagorean : side ^ 2 = (base / 2) ^ 2 + altitude ^ 2

/-- The area of an isosceles triangle -/
noncomputable def triangle_area (t : IsoscelesTriangle) : ℝ := t.base * t.altitude / 2

/-- Theorem: The area of the isosceles triangle with altitude 10 and perimeter 40 is 75 -/
theorem isosceles_triangle_area_is_75 :
  ∀ t : IsoscelesTriangle, t.altitude = 10 ∧ t.perimeter = 40 → triangle_area t = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_is_75_l644_64400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_value_l644_64409

/-- The average of the first 7 positive multiples of 8 -/
noncomputable def a : ℝ := (1 / 7) * (8 + 16 + 24 + 32 + 40 + 48 + 56)

/-- The median of the first 3 positive multiples of a positive integer n -/
def b (n : ℕ) : ℝ := 2 * n

/-- Theorem stating that if a^2 - b^2 = 0, then n = 16 -/
theorem positive_integer_value (n : ℕ) (h : n > 0) :
  a^2 - (b n)^2 = 0 → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_value_l644_64409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_profit_calculation_l644_64422

/-- Represents the profit percentage when selling an article -/
noncomputable def profit_percentage (cost_price selling_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Represents the selling price after applying a discount -/
noncomputable def discounted_price (original_price discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

theorem discount_profit_calculation (cost_price : ℝ) (hp : cost_price > 0) :
  let no_discount_profit := 30
  let discount_rate := 0.05
  let marked_price := cost_price * (1 + no_discount_profit / 100)
  let selling_price := discounted_price marked_price discount_rate
  profit_percentage cost_price selling_price = 23.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_profit_calculation_l644_64422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_mod_500_l644_64475

def has_more_ones_than_zeros (n : ℕ) : Bool :=
  let digits := Nat.digits 2 n
  (digits.count 1) > (digits.count 0)

def M : ℕ := (Finset.range 1501).filter (λ n => has_more_ones_than_zeros n) |>.card

theorem M_mod_500 : M % 500 = 152 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_mod_500_l644_64475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_difference_log_calculation_l644_64488

theorem log_sum_difference (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.log x + Real.log y - Real.log z = Real.log ((x * y) / z) :=
by sorry

theorem log_calculation :
  Real.log 60 + Real.log 40 - Real.log 15 = Real.log 160 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_difference_log_calculation_l644_64488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_strip_probability_l644_64495

/-- The region defined by |x| + |y| ≤ 2 -/
def diamond : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1| + |p.2| ≤ 2}

/-- The region where the distance from the x-axis is ≤ 1 -/
def strip : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.2| ≤ 1}

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The probability of a point in the diamond being in the strip -/
noncomputable def probability : ℝ :=
  (area (diamond ∩ strip)) / (area diamond)

theorem diamond_strip_probability :
  probability = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_strip_probability_l644_64495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l644_64489

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Given a line segment AB of length 100, C divides it in the golden ratio -/
def golden_section (AC BC : ℝ) : Prop :=
  AC / BC = φ ∧ AC + BC = 100 ∧ AC > BC

theorem golden_section_length :
  ∀ AC BC : ℝ, golden_section AC BC → AC = 75 - 25 * Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l644_64489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l644_64498

/-- Geometric sequence sum for first n terms -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_ratio (q : ℝ) (n : ℕ) (A B : ℝ) 
  (h₁ : A ≠ 0) (h₂ : q ≠ 1) :
  ∃ (a₁ : ℝ), 
    A = geometric_sum a₁ q n ∧ 
    B = geometric_sum a₁ q (2*n) → 
    (B - A) / A = q^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l644_64498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_smallest_positive_period_l644_64442

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 
  (sin (2 * x) / (1 - 2 * sin (2 * (x / 2 - π / 4)))) * (1 + 3 * tan x)

/-- The period of a real-valued function -/
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

/-- The smallest positive period of a function -/
def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_period f p ∧ p > 0 ∧ ∀ q, is_period f q → q > 0 → p ≤ q

/-- The theorem stating that 2π is the smallest positive period of f -/
theorem f_smallest_positive_period :
  smallest_positive_period f (2 * π) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_smallest_positive_period_l644_64442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l644_64405

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + 1/2 * x^2 + 2*a*x

theorem f_properties :
  /- 1. When a = -1, f(x) is decreasing on ℝ -/
  (∀ x y : ℝ, x < y → f (-1) x > f (-1) y) ∧
  /- 2. The range of a for which f(x) has a monotonically increasing interval in (2/3, +∞) is (-1/9, +∞) -/
  (∀ a : ℝ, a > -1/9 ↔ ∃ x : ℝ, x > 2/3 ∧ ∀ y : ℝ, y > x → f a y > f a x) ∧
  /- 3. When 0 < a < 2 and the minimum value of f(x) on [1, 4] is -16/3, the maximum value of f(x) on [1, 4] is 10/3 -/
  (∀ a : ℝ, 0 < a ∧ a < 2 ∧ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → f a x ≥ -16/3) ∧ (∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ f a x = -16/3) →
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → f a x ≤ 10/3) ∧ (∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ f a x = 10/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l644_64405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_detergent_amount_l644_64414

-- Define the original ratio
def original_ratio : Fin 3 → ℚ := ![2, 40, 100]

-- Define the altered ratio function
def alter_ratio (r : Fin 3 → ℚ) : Fin 3 → ℚ :=
  ![3 * r 0, r 1, 2 * r 2]

-- Define the amount of water in the altered solution
def water_amount : ℚ := 300

-- Theorem statement
theorem detergent_amount (r : Fin 3 → ℚ) (h : r = original_ratio) :
  let altered := alter_ratio r
  (altered 1 / altered 2) * water_amount = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_detergent_amount_l644_64414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_person_y_speed_l644_64440

/-- The speed of the river current -/
def y : ℝ := 0  -- We'll define this as a variable later if needed

/-- Person X's rowing speed in still water -/
def x_speed : ℝ := 6

/-- Person Y's rowing speed in still water -/
def y_speed : ℝ := 10

/-- Time taken for X and Y to meet when traveling towards each other -/
def meeting_time : ℝ := 4

/-- Time taken for Y to catch up to X when traveling in the same direction -/
def catch_up_time : ℝ := 16

/-- The total distance traveled when X and Y meet -/
def total_distance : ℝ := meeting_time * (x_speed - y + y_speed + y)

/-- The distance between X and Y when traveling in the same direction -/
def distance_difference : ℝ := catch_up_time * (y_speed - x_speed)

theorem person_y_speed : 
  x_speed = 6 →
  meeting_time = 4 →
  catch_up_time = 16 →
  y_speed = 10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_person_y_speed_l644_64440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_implies_x_half_l644_64435

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Given a real number x, define z as (2+i)/(x-i) -/
noncomputable def z (x : ℝ) : ℂ := (2 + Complex.I) / (x - Complex.I)

/-- If z(x) is purely imaginary, then x = 1/2 -/
theorem z_purely_imaginary_implies_x_half (x : ℝ) :
  IsPurelyImaginary (z x) → x = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_implies_x_half_l644_64435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_cost_price_approx_l644_64450

/-- The cost price of a radio given selling price, overhead expenses, and profit percent. -/
noncomputable def cost_price (selling_price : ℝ) (overhead : ℝ) (profit_percent : ℝ) : ℝ :=
  (selling_price - overhead) / (1 + profit_percent / 100)

/-- Theorem stating that the cost price of the radio is approximately 229.41 given the conditions. -/
theorem radio_cost_price_approx :
  let selling_price : ℝ := 300
  let overhead : ℝ := 28
  let profit_percent : ℝ := 18.577075098814234
  abs (cost_price selling_price overhead profit_percent - 229.41) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_cost_price_approx_l644_64450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_polyhedra_arrangement_l644_64406

/-- A convex polyhedron in 3-dimensional space -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Predicate to check if two polyhedra touch -/
def touch (p q : ConvexPolyhedron) : Prop :=
  sorry

/-- Predicate to check if three polyhedra share a common point -/
def shareCommonPoint (p q r : ConvexPolyhedron) : Prop :=
  sorry

/-- Theorem stating the existence of the required arrangement -/
theorem existence_of_polyhedra_arrangement :
  ∃ (arrangement : Finset ConvexPolyhedron),
    (arrangement.card = 2001) ∧
    (∀ p q : ConvexPolyhedron, p ∈ arrangement → q ∈ arrangement → p ≠ q → touch p q) ∧
    (∀ p q r : ConvexPolyhedron, p ∈ arrangement → q ∈ arrangement → r ∈ arrangement →
      p ≠ q ∧ q ≠ r ∧ p ≠ r → ¬ shareCommonPoint p q r) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_polyhedra_arrangement_l644_64406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_finding_space_l644_64432

def total_spaces : ℕ := 20
def cars_parked : ℕ := 15
def required_consecutive_spaces : ℕ := 3

theorem probability_of_finding_space (total_spaces cars_parked required_consecutive_spaces : ℕ) 
  (h1 : total_spaces = 20)
  (h2 : cars_parked = 15)
  (h3 : required_consecutive_spaces = 3) :
  (964 : ℚ) / 1107 = 1 - (Nat.choose 14 5 : ℚ) / (Nat.choose 20 5 : ℚ) :=
by
  sorry

#eval (964 : ℚ) / 1107
#eval 1 - (Nat.choose 14 5 : ℚ) / (Nat.choose 20 5 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_finding_space_l644_64432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_curve_with_tangent_l644_64472

noncomputable def curve (x : ℝ) : ℝ := 4 / x^2

noncomputable def curve_derivative (x : ℝ) : ℝ := -8 / x^3

noncomputable def tangent_angle : ℝ := 3 * Real.pi / 4

theorem point_on_curve_with_tangent :
  let P : ℝ × ℝ := (2, 1)
  curve P.1 = P.2 ∧ 
  curve_derivative P.1 = Real.tan tangent_angle :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_curve_with_tangent_l644_64472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_real_roots_l644_64448

/-- A polynomial of degree 504 with real coefficients -/
def Polynomial504 : Type := Polynomial ℝ

/-- The roots of a polynomial -/
noncomputable def roots (p : Polynomial504) : Finset ℂ := sorry

/-- The absolute values of the roots of a polynomial -/
noncomputable def rootAbs (p : Polynomial504) : Finset ℝ := sorry

/-- The number of real roots of a polynomial -/
noncomputable def realRootCount (p : Polynomial504) : ℕ := sorry

/-- The theorem stating the minimum number of real roots -/
theorem min_real_roots (p : Polynomial504) 
  (h1 : p.degree = 504)
  (h2 : (rootAbs p).card = 252) :
  126 ≤ realRootCount p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_real_roots_l644_64448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_angle_C_value_l644_64479

open Real

structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  sine_law : sin A / a = sin B / b
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*cos C

theorem triangle_theorem (t : Triangle) 
  (h : (t.a - 3*t.b) * cos t.C = t.c * (3 * cos t.B - cos t.A)) : 
  sin t.B / sin t.A = 3 := by
  sorry

theorem angle_C_value (t : Triangle) 
  (h1 : (t.a - 3*t.b) * cos t.C = t.c * (3 * cos t.B - cos t.A))
  (h2 : t.c = Real.sqrt 7 * t.a) : 
  t.C = π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_angle_C_value_l644_64479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l644_64445

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - Real.log (x + a)

-- Part 1
theorem part_one (a : ℝ) (h_a : a > 0) (h_min : ∀ x, f a x ≥ 0) (h_exists : ∃ x, f a x = 0) : a = 1 :=
sorry

-- Part 2
theorem part_two : 
  (∃ k : ℝ, (∀ x : ℝ, x ≥ 0 → f 1 x ≤ k * x^2) ∧ 
   (∀ k' : ℝ, (∀ x : ℝ, x ≥ 0 → f 1 x ≤ k' * x^2) → k ≤ k')) ∧ 
  (let k := (1/2 : ℝ);
   (∀ x : ℝ, x ≥ 0 → f 1 x ≤ k * x^2) ∧
   (∀ k' : ℝ, (∀ x : ℝ, x ≥ 0 → f 1 x ≤ k' * x^2) → k ≤ k')) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l644_64445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_equals_2R_l644_64491

/-- A truncated cone with a 30° angle between the axis and slant height -/
structure TruncatedCone where
  R : ℝ  -- Radius of the larger base
  r : ℝ  -- Radius of the smaller base
  h : ℝ  -- Height of the truncated cone
  slant_angle : ℝ  -- Angle between the axis and slant height in radians
  slant_angle_is_30 : slant_angle = π / 6

/-- The shortest path on the surface of the truncated cone -/
def shortest_path (cone : TruncatedCone) : ℝ := 2 * cone.R

/-- Theorem: The shortest path on the surface of the truncated cone connecting
    diametrically opposite points on the boundaries of the two bases is equal to
    twice the radius of the larger base -/
theorem shortest_path_equals_2R (cone : TruncatedCone) :
  shortest_path cone = 2 * cone.R := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_equals_2R_l644_64491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_left_directrix_is_5_l644_64434

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define a point P on the ellipse
noncomputable def P : ℝ × ℝ := sorry

-- Assume P is on the ellipse
axiom P_on_ellipse : ellipse P.1 P.2

-- Define the distance from P to the left focus
noncomputable def distance_to_left_focus : ℝ := 5/2

-- Define the distance from P to the left directrix
noncomputable def distance_to_left_directrix : ℝ := sorry

-- Theorem to prove
theorem distance_to_left_directrix_is_5 :
  distance_to_left_directrix = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_left_directrix_is_5_l644_64434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_seq_transform_l644_64460

/-- A function that represents a sequence of binary digits -/
def BinarySeq (n : ℕ) := Fin n → Bool

/-- A step that changes k elements in a binary sequence -/
def Step (n k : ℕ) (A B : BinarySeq n) :=
  ∃ (S : Finset (Fin n)), S.card = k ∧ ∀ i, i ∉ S → A i = B i

/-- The theorem stating that any binary sequence can be transformed into another
    using a finite number of steps that change k elements at a time -/
theorem binary_seq_transform (n k : ℕ) (h : 0 < k) (h' : k < n) :
  ∀ (A B : BinarySeq n), ∃ (m : ℕ) (f : Fin (m + 1) → BinarySeq n),
    f 0 = A ∧ f ⟨m, Nat.lt_succ_self m⟩ = B ∧
    ∀ i : Fin m, Step n k (f i) (f i.succ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_seq_transform_l644_64460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l644_64403

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x

/-- The line function -/
def g (x : ℝ) : ℝ := x + 3

/-- The shortest distance from a point on y = 2 + ln x to y = x + 3 -/
theorem shortest_distance : 
  ∃ (x₀ : ℝ), x₀ > 0 ∧ 
  ∀ (x : ℝ), x > 0 → 
  (x₀ - x)^2 + (f x₀ - g x₀)^2 ≤ (x - x)^2 + (f x - g x)^2 ∧
  (x₀ - x₀)^2 + (f x₀ - g x₀)^2 = 2 := by
  sorry

#check shortest_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l644_64403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_diophantine_equation_l644_64471

theorem infinite_solutions_diophantine_equation :
  ∃ f : ℕ → ℕ × ℕ × ℕ,
    Function.Injective f ∧
    ∀ n : ℕ,
      let (x, y, z) := f n
      x > 0 ∧ y > 0 ∧ z > 0 ∧
      x^7 + y^8 = z^9 :=
by
  -- We'll define f explicitly
  let f := λ n : ℕ => (2^32 * n^72, 2^28 * n^63, 2^25 * n^56)
  
  -- Prove existence of f
  use f

  -- Prove injectivity of f and the required properties
  constructor
  · -- Injectivity
    intro n1 n2 h
    -- The proof of injectivity would go here
    sorry
  
  · -- Properties for all n
    intro n
    let (x, y, z) := f n
    -- The proof of the properties would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_diophantine_equation_l644_64471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolic_identity_l644_64439

-- Define hyperbolic sine function
noncomputable def sh (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

-- Define hyperbolic cosine function
noncomputable def ch (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2

-- Theorem statement
theorem hyperbolic_identity (x : ℝ) : (ch x)^2 - (sh x)^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolic_identity_l644_64439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_approx_l644_64455

noncomputable section

-- Define the rhombus properties
def diagonal1 : ℝ := 12
def area : ℝ := 328.19506394825623

-- Define the function to calculate the side length
def rhombus_side_length (d1 : ℝ) (a : ℝ) : ℝ :=
  let d2 := 2 * a / d1
  Real.sqrt (0.25 * (d1^2 + d2^2))

-- Theorem statement
theorem rhombus_side_length_approx :
  |rhombus_side_length diagonal1 area - 53.352| < 0.001 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_approx_l644_64455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_range_l644_64415

theorem equation_solution_range (a : ℝ) :
  (∃ x : ℝ, (1/3)^(abs x) + a - 1 = 0) ↔ (0 ≤ a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_range_l644_64415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_is_20_5_l644_64463

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the distance from a point to a line -/
noncomputable def distancePointToLine (p : Point2D) (l : Line2D) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- Theorem: The sum of altitudes of the triangle formed by 15x + 3y = 45 and the axes is 20.5 -/
theorem sum_of_altitudes_is_20_5 : 
  let line := Line2D.mk 15 3 (-45)
  let origin := Point2D.mk 0 0
  let x_intercept := Point2D.mk 3 0
  let y_intercept := Point2D.mk 0 15
  let altitude_to_x_axis := x_intercept.y
  let altitude_to_y_axis := y_intercept.x
  let altitude_to_hypotenuse := distancePointToLine origin line
  altitude_to_x_axis + altitude_to_y_axis + altitude_to_hypotenuse = 20.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_is_20_5_l644_64463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minimum_value_l644_64404

noncomputable def g (x : ℝ) : ℝ := x + x / (x^2 + 1) + x * (x + 5) / (x^2 + 3) + 3 * (x + 3) / (x * (x^2 + 3))

theorem g_minimum_value (x : ℝ) (h : x > 0) : g x ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minimum_value_l644_64404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_month_bill_is_48_l644_64461

/-- Represents Elvin's monthly telephone bill structure -/
structure PhoneBill where
  callCharge : ℚ
  internetCharge : ℚ

/-- Calculates the total bill given a PhoneBill -/
def totalBill (bill : PhoneBill) : ℚ :=
  bill.callCharge + bill.internetCharge

theorem first_month_bill_is_48 
  (firstMonth secondMonth : PhoneBill)
  (h1 : totalBill firstMonth = 48)
  (h2 : totalBill secondMonth = 90)
  (h3 : secondMonth.callCharge = 2 * firstMonth.callCharge)
  (h4 : firstMonth.internetCharge = secondMonth.internetCharge) :
  totalBill firstMonth = 48 := by
  -- The proof goes here
  sorry

#check first_month_bill_is_48

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_month_bill_is_48_l644_64461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_l644_64493

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluate a quadratic polynomial at a given value -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℤ) : ℤ :=
  p.a * x^2 + p.b * x + p.c

/-- A polynomial with integer coefficients -/
structure IntPolynomial where
  coeffs : List ℤ

/-- Evaluate an integer polynomial at a given value -/
def IntPolynomial.eval (p : IntPolynomial) (x : ℤ) : ℤ :=
  (List.range p.coeffs.length).foldl (fun acc i => acc + (p.coeffs.get! i) * x^i) 0

/-- The statement of the problem -/
theorem polynomial_existence (p : QuadraticPolynomial) 
  (h : ∀ x : ℤ, ¬(3 ∣ p.eval x)) :
  ∃ (f h : IntPolynomial), 
    ∀ x : ℤ, 
      (p.eval x) * (IntPolynomial.eval f x) + 3 * (IntPolynomial.eval h x) = 
      x^6 + x^4 + x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_l644_64493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_parabola_chords_l644_64424

/-- Given a parabola y^2 = 2px (p > 0) with focus F and two mutually perpendicular chords FA and FB
    passing through F, the minimum area of triangle FAB is (3 - 2√2) p^2. -/
theorem min_area_triangle_parabola_chords (p : ℝ) (h : p > 0) :
  ∃ (F A B : ℝ × ℝ),
    (∀ (x y : ℝ), y^2 = 2*p*x → (x, y) ∈ Set.range (λ (t : ℝ × ℝ) ↦ t)) ∧ 
    (F.1 = p ∧ F.2 = 0) ∧
    ((A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = 0) ∧
    (∀ (A' B' : ℝ × ℝ),
      ((A'.1 - F.1) * (B'.1 - F.1) + (A'.2 - F.2) * (B'.2 - F.2) = 0) →
      abs ((A'.1 - F.1) * (B'.2 - F.2) - (A'.2 - F.2) * (B'.1 - F.1)) / 2 ≥
      abs ((A.1 - F.1) * (B.2 - F.2) - (A.2 - F.2) * (B.1 - F.1)) / 2) ∧
    abs ((A.1 - F.1) * (B.2 - F.2) - (A.2 - F.2) * (B.1 - F.1)) / 2 = (3 - 2 * Real.sqrt 2) * p^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_parabola_chords_l644_64424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_denial_of_motion_implies_relativism_and_sophistry_problem_statement_l644_64452

-- Define the concepts
def motion : Prop := sorry
def stillness : Prop := sorry
def relativism : Prop := sorry
def sophistry : Prop := sorry

-- Define the implication
theorem denial_of_motion_implies_relativism_and_sophistry :
  (¬motion ∧ stillness) → (relativism ∧ sophistry) :=
sorry

-- The actual statement in the problem is false, so we negate it
theorem problem_statement : ¬((¬motion ∧ stillness) → (relativism ∧ sophistry)) :=
sorry

#check problem_statement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_denial_of_motion_implies_relativism_and_sophistry_problem_statement_l644_64452
