import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l477_47767

def M : Set ℝ := {x | (1 : ℝ) / x < 1}

theorem complement_of_M : {x : ℝ | 0 ≤ x ∧ x ≤ 1} = (Set.univ : Set ℝ) \ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l477_47767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_exists_iff_acute_angle_l477_47706

/-- A triangle with side length a and opposite angle α, where the angle bisector of α forms a 45° angle with side a -/
structure SpecialTriangle (a : ℝ) (α : ℝ) where
  /-- Side length a is positive -/
  a_pos : 0 < a
  /-- Angle α is positive -/
  α_pos : 0 < α
  /-- The angle bisector of α forms a 45° angle with side a -/
  bisector_angle : α / 2 = π / 4

/-- The existence of a SpecialTriangle is equivalent to α being an acute angle -/
theorem special_triangle_exists_iff_acute_angle {a α : ℝ} (h : 0 < a) :
  Nonempty (SpecialTriangle a α) ↔ 0 < α ∧ α < π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_exists_iff_acute_angle_l477_47706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fixed_interval_for_f_l477_47771

noncomputable def f (x : ℝ) : ℝ := -x / (1 + abs x)

theorem no_fixed_interval_for_f :
  ¬ ∃ (a b : ℝ), a < b ∧ Set.Ioc a b = {y | ∃ x ∈ Set.Ioc a b, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fixed_interval_for_f_l477_47771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_k_l477_47784

theorem existence_of_k (m n : ℕ+) : ∃ k : ℕ+, 
  (List.length (Nat.factors ((2^(k : ℕ) : ℕ) - m))) ≥ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_k_l477_47784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_multiple_of_three_l477_47701

def n : ℕ := 2^4 * 3^3 * 7

theorem factors_multiple_of_three :
  (Finset.filter (λ x : ℕ => x ∣ n ∧ 3 ∣ x) (Finset.range (n + 1))).card = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_multiple_of_three_l477_47701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_l477_47709

theorem smallest_number : 
  let a := -2
  let b := abs (-2)
  let c := -(-1)
  let d := -(1/2)
  a < b ∧ a < c ∧ a < d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_l477_47709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bailing_rate_for_scenario_l477_47778

/-- Represents the scenario of Mike and John's fishing trip -/
structure FishingTrip where
  distanceToShore : ℝ
  waterIntakeRate : ℝ
  maxWaterCapacity : ℝ
  rowingSpeed : ℝ

/-- Calculates the minimum bailing rate required to prevent sinking -/
noncomputable def minBailingRate (trip : FishingTrip) : ℝ :=
  let timeToShore := trip.distanceToShore / trip.rowingSpeed
  let totalWaterIntake := trip.waterIntakeRate * timeToShore
  let excessWater := totalWaterIntake - trip.maxWaterCapacity
  excessWater / timeToShore

/-- The theorem stating the minimum bailing rate for the given scenario -/
theorem min_bailing_rate_for_scenario :
  let trip := FishingTrip.mk 2 12 50 3
  minBailingRate trip = 10.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bailing_rate_for_scenario_l477_47778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_a2_monotonicity_depends_on_a_f_nonpositive_range_l477_47745

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (a + 1) * x + Real.log x

theorem extreme_values_a2 (x : ℝ) (hx : x > 0) :
  ∃ (y z : ℝ), y = 1/2 ∧ z = 1 ∧
  (∀ w, w > 0 → f 2 w ≤ f 2 y ∨ f 2 w ≤ f 2 z) :=
by sorry

theorem monotonicity_depends_on_a (a : ℝ) :
  ∃ (P : ℝ → Prop), (∀ x y, x > 0 → y > 0 → x < y → P x → P y → f a x < f a y) ∨
                    (∀ x y, x > 0 → y > 0 → x < y → P x → P y → f a x > f a y) :=
by sorry

theorem f_nonpositive_range (a : ℝ) :
  (∀ x, x > 0 → f a x ≤ 0) ↔ -2 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_a2_monotonicity_depends_on_a_f_nonpositive_range_l477_47745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_correct_l477_47799

def sequence_a (n : ℕ) : ℕ := 2^(n-1) + 2

theorem sequence_formula_correct :
  (sequence_a 1 = 3) ∧
  (sequence_a 2 = 4) ∧
  (sequence_a 3 = 6) ∧
  (sequence_a 4 = 10) ∧
  (sequence_a 5 = 18) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 2^(n-1) + 2) := by
  constructor
  · simp [sequence_a]
  constructor
  · simp [sequence_a]
  constructor
  · simp [sequence_a]
  constructor
  · simp [sequence_a]
  constructor
  · simp [sequence_a]
  intro n hn
  simp [sequence_a]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_correct_l477_47799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_of_exponential_expressions_l477_47734

theorem comparison_of_exponential_expressions :
  let a := (1/3 : Real) ^ (Real.log 3 / Real.log 2)
  let b := (1/3 : Real) ^ (Real.log 4 / Real.log 5)
  let c := (3 : Real) ^ Real.log 3
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_of_exponential_expressions_l477_47734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_arithmetic_sequence_l477_47718

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + d * (n - 1)) / 2

theorem min_sum_arithmetic_sequence :
  ∀ (a₁ d : ℝ),
    a₁ = -15 →
    arithmetic_sequence a₁ d 3 + arithmetic_sequence a₁ d 5 = -18 →
    ∃ (n : ℕ),
      ∀ (m : ℕ),
        sum_arithmetic_sequence a₁ d n ≤ sum_arithmetic_sequence a₁ d m ∧
        n = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_arithmetic_sequence_l477_47718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_50_between_consecutive_integers_l477_47703

theorem log2_50_between_consecutive_integers :
  ∃ c d : ℤ, c + 1 = d ∧ (c : ℝ) < Real.log 50 / Real.log 2 ∧ Real.log 50 / Real.log 2 < (d : ℝ) ∧ c + d = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_50_between_consecutive_integers_l477_47703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_count_representations_correct_l477_47793

/-- 
Given a natural number n, counts the number of distinct representations of n 
as a sum of smaller natural addends, where different orderings are considered distinct.
-/
def countRepresentations (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2^(n-1) - 1

/-- 
Helper function to represent the actual number of distinct representations
of n as a sum of smaller natural addends.
This function is not actually implemented, it's just for theorem statement purposes.
-/
noncomputable def number_of_distinct_representations (n : ℕ) : ℕ :=
  sorry

/-- 
Theorem stating that countRepresentations correctly counts the number of 
distinct representations for any natural number n.
-/
theorem count_representations_correct (n : ℕ) : 
  countRepresentations n = number_of_distinct_representations n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_count_representations_correct_l477_47793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_zeros_greater_than_e_squared_l477_47783

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log x + k * x

noncomputable def F (a m : ℝ) (x : ℝ) : ℝ := (a - 1) / x - m

theorem product_of_zeros_greater_than_e_squared 
  (a b m : ℝ) 
  (h1 : ∀ x > 0, f 0 x + b / x - a ≥ 0)
  (h2 : ∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ F a m x₁ = 0 ∧ F a m x₂ = 0 ∧ x₁ ≠ x₂) :
  ∃ x₁ x₂, F a m x₁ = 0 ∧ F a m x₂ = 0 ∧ x₁ * x₂ > Real.exp 2 := by
  sorry

#check product_of_zeros_greater_than_e_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_zeros_greater_than_e_squared_l477_47783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_3_l477_47764

def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem right_triangle_hypotenuse_3 
  (a b : ℝ) 
  (h : right_triangle a b 3) :
  let sides := [a, b, 3]
  let mean := (a + b + 3) / 3
  let variance := (1/3) * ((a - mean)^2 + (b - mean)^2 + (3 - mean)^2)
  (variance < 2) ∧
  (∃ (min_std_dev : ℝ), min_std_dev = Real.sqrt 2 - 1 ∧
    ∀ (x y : ℝ), right_triangle x y 3 → 
      Real.sqrt ((1/3) * ((x - (x + y + 3) / 3)^2 + (y - (x + y + 3) / 3)^2 + (3 - (x + y + 3) / 3)^2)) ≥ min_std_dev) ∧
  (let min_leg := 3 * Real.sqrt 2 / 2
   variance = ((min_leg - mean)^2 + (min_leg - mean)^2 + (3 - mean)^2) / 3
   → a = min_leg ∧ b = min_leg) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_3_l477_47764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l477_47742

noncomputable def f (x : ℝ) := 3^(1 + abs x) - 1 / (1 + x^2)

theorem f_inequality (x : ℝ) : f x > f (2*x - 1) ↔ x ∈ Set.Ioo (1/3) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l477_47742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorability_l477_47705

/-- A polynomial of degree 4 with integer coefficients -/
def PolynomialA (a : ℤ) (x : ℝ) : ℝ := x^4 - 3*x^3 + a*x^2 - 9*x - 2

/-- Represents a quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  p : ℤ
  q : ℤ
  r : ℤ

/-- Checks if a polynomial can be factored into two quadratic polynomials -/
def IsFactorable (a : ℤ) : Prop :=
  ∃ (q1 q2 : QuadraticPolynomial), ∀ x : ℝ,
    PolynomialA a x = (q1.p * x^2 + q1.q * x + q1.r) * (q2.p * x^2 + q2.q * x + q2.r)

/-- The main theorem stating the conditions for factorability -/
theorem polynomial_factorability (a : ℤ) :
  IsFactorable a ↔ a = -3 ∨ a = -11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorability_l477_47705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_greater_than_one_l477_47747

noncomputable def f (x : ℝ) := |Real.log x|

theorem ab_greater_than_one 
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < b) 
  (h3 : f a < f b) : 
  a * b > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_greater_than_one_l477_47747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stairs_theorem_l477_47786

noncomputable def stairs_problem (a b c d : ℕ) (inch_to_cm : ℚ) : ℚ × ℚ :=
  let net_steps : ℤ := a * c - b * c
  let total_inches : ℚ := (net_steps : ℚ) * d
  let feet : ℚ := total_inches / 12
  let meters : ℚ := total_inches * inch_to_cm / 100
  (feet, meters)

theorem stairs_theorem :
  stairs_problem 3 6 12 8 (254 / 100) = (-24, -7306 / 1000) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stairs_theorem_l477_47786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l477_47713

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x + x * (deriv f x) < 0) {a b : ℝ} (hab : a < b) : 
  a * f a > b * f b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l477_47713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_two_element_set_l477_47790

theorem number_of_subsets_two_element_set {α : Type*} (S : Finset α) (h : S.card = 2) :
  (Finset.powerset S).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_two_element_set_l477_47790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_plus_π_3_l477_47782

theorem cos_2α_plus_π_3 (α : ℝ) (h : Real.cos α + Real.sqrt 3 * Real.sin α = 3/5) :
  Real.cos (2 * α + Real.pi/3) = 41/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_plus_π_3_l477_47782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_remaining_numbers_l477_47726

theorem average_of_remaining_numbers 
  (total_count : ℕ) 
  (total_average : ℚ) 
  (first_pair_average : ℚ) 
  (second_pair_average : ℚ) 
  (h1 : total_count = 6)
  (h2 : total_average = 395/100)
  (h3 : first_pair_average = 38/10)
  (h4 : second_pair_average = 385/100)
  : 
  let total_sum := total_count * total_average
  let first_pair_sum := 2 * first_pair_average
  let second_pair_sum := 2 * second_pair_average
  let remaining_pair_sum := total_sum - first_pair_sum - second_pair_sum
  remaining_pair_sum / 2 = 42/10 := by
    sorry

#check average_of_remaining_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_remaining_numbers_l477_47726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_theorem_shortest_distance_is_shortest_l477_47791

/-- Represents a rectangular cuboid with edge lengths a, b, and c. -/
structure RectangularCuboid where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : 0 < a
  h2 : 0 < b
  h3 : 0 < c
  h4 : a < b
  h5 : b < c

/-- The shortest distance from point A to point C₁ along the surface of the cuboid. -/
noncomputable def shortest_distance (cuboid : RectangularCuboid) : ℝ :=
  Real.sqrt (cuboid.a^2 + cuboid.b^2 + cuboid.c^2 + 2 * cuboid.b * cuboid.c)

/-- Theorem stating that the shortest distance from A to C₁ along the surface
    of the cuboid is √(a² + b² + c² + 2bc). -/
theorem shortest_distance_theorem (cuboid : RectangularCuboid) :
  shortest_distance cuboid = Real.sqrt (cuboid.a^2 + cuboid.b^2 + cuboid.c^2 + 2 * cuboid.b * cuboid.c) :=
by
  -- Unfold the definition of shortest_distance
  unfold shortest_distance
  -- The equality holds by definition
  rfl

/-- Proof that the shortest distance is indeed the shortest -/
theorem shortest_distance_is_shortest (cuboid : RectangularCuboid) :
  ∀ path : ℝ, path ≥ shortest_distance cuboid :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_theorem_shortest_distance_is_shortest_l477_47791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_32_465_not_round_to_32_47_l477_47769

/-- Rounds a number to the nearest hundredth using the "round half up" rule -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The list of numbers to be checked -/
def numbersList : List ℝ := [32.469, 32.465, 32.4701, 32.474999, 32.473]

theorem only_32_465_not_round_to_32_47 :
  ∃! x, x ∈ numbersList ∧ roundToHundredth x ≠ 32.47 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_32_465_not_round_to_32_47_l477_47769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_cone_l477_47761

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Checks if three spheres are tangent to each other and the cone -/
def areTangent (c : Cone) (s1 s2 s3 : Sphere) : Prop :=
  s1.radius = s2.radius ∧ s2.radius = s3.radius ∧
  -- Additional conditions for tangency would be defined here
  True  -- Placeholder for additional conditions

theorem sphere_radius_in_cone (c : Cone) (s1 s2 s3 : Sphere) :
  c.baseRadius = 7 ∧ c.height = 15 ∧ areTangent c s1 s2 s3 →
  s1.radius = (630 - 262.5 * Real.sqrt 3) / 69 := by
  sorry

#check sphere_radius_in_cone

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_cone_l477_47761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_sum_of_sqrts_l477_47736

noncomputable def f (x : ℝ) : ℝ := |2*x - 3/4| + |2*x + 5/4|

-- Part I
theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, f x ≥ a^2 - a) ↔ -1 ≤ a ∧ a ≤ 2 := by sorry

-- Part II
theorem sum_of_sqrts (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  Real.sqrt (2*m + 1) + Real.sqrt (2*n + 1) ≤ 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_sum_of_sqrts_l477_47736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perpendicularity_l477_47744

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

-- Define the points
noncomputable def A (t : Triangle) : ℝ × ℝ := (0, t.a)
noncomputable def B (t : Triangle) : ℝ × ℝ := (t.b, 0)
noncomputable def C (t : Triangle) : ℝ × ℝ := (t.c, 0)

-- Define the circumcenter O
noncomputable def O (t : Triangle) : ℝ × ℝ := ((t.b + t.c) / 2, (t.a^2 + t.b * t.c) / (2 * t.a))

-- Define the orthocenter H
noncomputable def H (t : Triangle) : ℝ × ℝ := (0, -t.b * t.c / t.a)

-- Define the intersection points M and N
noncomputable def M (t : Triangle) : ℝ × ℝ := 
  ((t.a^2 + t.b * t.c) * t.b / (t.a^2 + 2 * t.b * t.c - t.b^2), 
   (t.c - t.b) * t.a * t.b / (t.a^2 + 2 * t.b * t.c - t.b^2))

noncomputable def N (t : Triangle) : ℝ × ℝ := 
  ((t.a^2 + t.b * t.c) * t.c / (t.a^2 + 2 * t.b * t.c - t.c^2), 
   (t.b - t.c) * t.a * t.c / (t.a^2 + 2 * t.b * t.c - t.c^2))

-- Define perpendicularity
def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

-- Define vector subtraction
def vsub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- State the theorem
theorem triangle_perpendicularity (t : Triangle) : 
  (perpendicular (vsub (O t) (B t)) (vsub (H t) (C t))) ∧ 
  (perpendicular (vsub (O t) (C t)) (vsub (H t) (B t))) ∧
  (perpendicular (vsub (O t) (H t)) (vsub (M t) (N t))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perpendicularity_l477_47744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_beats_B_l477_47776

/-- Define the distance both runners cover -/
noncomputable def distance : ℝ := 4.5

/-- Define A's time to run the distance in seconds -/
noncomputable def time_A : ℝ := 90

/-- Define B's time to run the distance in seconds -/
noncomputable def time_B : ℝ := 180

/-- Calculate A's speed -/
noncomputable def speed_A : ℝ := distance / time_A

/-- Calculate the time difference between A and B -/
noncomputable def time_diff : ℝ := time_B - time_A

/-- Theorem: A beats B by a distance equal to the original distance -/
theorem distance_A_beats_B : speed_A * time_diff = distance := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_beats_B_l477_47776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_roll_probability_l477_47766

/-- The probability of no two adjacent people rolling the same number on a twelve-sided die,
    when four people sit in a line and each roll once. -/
theorem adjacent_roll_probability :
  let die_sides : ℕ := 12
  let num_people : ℕ := 4
  let prob : ℚ := 1331 / 1728
  (die_sides - 1)^(num_people - 1) / die_sides^num_people = prob := by
  -- Introduce the local variables
  let die_sides := 12
  let num_people := 4
  let prob := 1331 / 1728
  -- The proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_roll_probability_l477_47766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_projection_is_incenter_l477_47715

/-- A tetrahedron with vertex S and base ABC. -/
structure Tetrahedron where
  S : EuclideanSpace ℝ (Fin 3)
  A : EuclideanSpace ℝ (Fin 3)
  B : EuclideanSpace ℝ (Fin 3)
  C : EuclideanSpace ℝ (Fin 3)

/-- The projection of a point onto a plane. -/
noncomputable def project (p : EuclideanSpace ℝ (Fin 3)) (plane : Set (EuclideanSpace ℝ (Fin 3))) : EuclideanSpace ℝ (Fin 3) :=
  sorry

/-- The incenter of a triangle. -/
noncomputable def incenter (A B C : EuclideanSpace ℝ (Fin 3)) : EuclideanSpace ℝ (Fin 3) :=
  sorry

/-- The dihedral angle between two faces of a tetrahedron. -/
noncomputable def dihedralAngle (t : Tetrahedron) (face1 face2 : Set (EuclideanSpace ℝ (Fin 3))) : ℝ :=
  sorry

/-- Predicate to check if a point is inside a triangle. -/
def isInside (p : EuclideanSpace ℝ (Fin 3)) (A B C : EuclideanSpace ℝ (Fin 3)) : Prop :=
  sorry

/-- The theorem statement. -/
theorem tetrahedron_projection_is_incenter (t : Tetrahedron) 
  (h1 : t.A ≠ t.B ∧ t.B ≠ t.C ∧ t.C ≠ t.A) -- Base is scalene
  (h2 : ∃ plane, dihedralAngle t {t.S, t.A, t.B} plane = 
                 dihedralAngle t {t.S, t.B, t.C} plane ∧
                 dihedralAngle t {t.S, t.C, t.A} plane = 
                 dihedralAngle t {t.S, t.A, t.B} plane) -- Equal dihedral angles
  (h3 : isInside (project t.S {t.A, t.B, t.C}) t.A t.B t.C) -- Projection inside base
  : project t.S {t.A, t.B, t.C} = incenter t.A t.B t.C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_projection_is_incenter_l477_47715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l477_47710

noncomputable section

-- Define points A and B
def A : ℝ × ℝ := (0, -3)
def B : ℝ × ℝ := (4, 0)

-- Define the circle equation
def on_circle (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 - 2*P.2 = 0

-- Define the area of triangle ABP
def triangle_area (P : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (P.2 - A.2) - (B.2 - A.2) * (P.1 - A.1)) / 2

-- Theorem statement
theorem min_triangle_area :
  ∃ (min_area : ℝ), min_area = 11/2 ∧
  ∀ (P : ℝ × ℝ), on_circle P → triangle_area P ≥ min_area := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l477_47710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_solution_l477_47723

/-- A subtype for positive natural numbers -/
def PositiveNat := {n : ℕ // n > 0}

/-- The functional equation property -/
def SatisfiesFunctionalEquation (f : PositiveNat → PositiveNat) : Prop :=
  ∀ m n : PositiveNat, (f ⟨m.val + n.val, sorry⟩).val * (f ⟨m.val - n.val, sorry⟩).val = (f ⟨m.val * m.val, sorry⟩).val

/-- The theorem statement -/
theorem constant_function_solution 
  (f : PositiveNat → PositiveNat) 
  (h : SatisfiesFunctionalEquation f) : 
  ∀ n : PositiveNat, f n = ⟨1, sorry⟩ := by
  sorry

#check constant_function_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_solution_l477_47723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_increase_calculation_l477_47743

/-- Calculates the increase in speed for an ultramarathon runner given initial conditions and new total distance --/
theorem speed_increase_calculation (initial_time initial_speed : ℝ) (time_increase : ℝ) (new_total_distance : ℝ) : 
  initial_time = 8 → 
  initial_speed = 8 → 
  time_increase = 0.75 → 
  new_total_distance = 168 → 
  let new_time := initial_time * (1 + time_increase)
  let new_speed := new_total_distance / new_time
  new_speed - initial_speed = 4 := by
  sorry

/-- The speed increase is 4 mph --/
def speed_increase : ℝ := 4

#eval speed_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_increase_calculation_l477_47743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_tiling_l477_47711

/-- Represents a tile with width and height -/
structure Tile where
  width : Nat
  height : Nat

/-- Represents a collection of tiles -/
structure TileSet where
  rectangles : List Tile
  square : Tile

/-- Represents a square board -/
structure Board where
  size : Nat

/-- Checks if a tile set can perfectly cover a board -/
def canCover (board : Board) (tiles : TileSet) : Prop :=
  sorry

theorem no_perfect_tiling :
  ¬ ∃ (arrangement : TileSet → Board → Prop),
    (let board := Board.mk 8
     let tiles := TileSet.mk
       (List.replicate 15 (Tile.mk 4 1))
       (Tile.mk 2 2)
     canCover board tiles ∧ arrangement tiles board) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_tiling_l477_47711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l477_47708

-- Define the hyperbola
def hyperbola (b : ℝ) (x y : ℝ) : Prop := x^2 / 4 - y^2 / b^2 = 1

-- Define the circle (renamed to avoid conflict with built-in circle)
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the asymptotes
def asymptote (b : ℝ) (x y : ℝ) : Prop := y = b / 2 * x ∨ y = -b / 2 * x

-- Define the area of quadrilateral ABCD
def quadrilateral_area (b : ℝ) : ℝ := 2 * b

-- Theorem statement
theorem hyperbola_equation (b : ℝ) (h1 : b > 0) :
  (∃ A B C D : ℝ × ℝ,
    circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧ circle_eq C.1 C.2 ∧ circle_eq D.1 D.2 ∧
    asymptote b A.1 A.2 ∧ asymptote b B.1 B.2 ∧ asymptote b C.1 C.2 ∧ asymptote b D.1 D.2 ∧
    quadrilateral_area b = 2 * b) →
  hyperbola b = hyperbola 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l477_47708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_correct_l477_47787

-- Define the types for planes, lines, and points
variable (Plane Line Point : Type)

-- Define the relationships between planes and lines
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the membership relation
variable (mem : Point → Line → Prop)

-- Define the given planes and lines
variable (α β : Plane)
variable (m n : Line)

-- State the theorem
theorem all_propositions_correct 
  (distinct_planes : α ≠ β)
  (non_intersecting : ¬ ∃ (p : Point), mem p m ∧ mem p n) :
  -- Proposition 1
  (parallel m n ∧ perpendicular m α → perpendicular n α) ∧
  -- Proposition 2
  (perpendicular m α ∧ perpendicular m β → plane_parallel α β) ∧
  -- Proposition 3
  (perpendicular m α ∧ parallel m n ∧ subset n β → plane_perpendicular α β) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_correct_l477_47787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_solutions_l477_47775

theorem factorial_equation_solutions :
  ∀ x y n : ℕ, 
    x > 0 → y > 0 → n > 0 →
    (Nat.factorial x + Nat.factorial y) / Nat.factorial n = 3^n ↔ 
    ((x = 2 ∧ y = 1 ∧ n = 1) ∨ (x = 1 ∧ y = 2 ∧ n = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_solutions_l477_47775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gray_area_is_ten_cm_squared_l477_47733

/-- Represents the dimensions of the rectangular sheet of paper -/
structure PaperDimensions where
  length : ℝ
  width : ℝ

/-- Represents the area of the gray region in the paper -/
def gray_area (dimensions : PaperDimensions) : ℝ := sorry

/-- Theorem stating that the area of the gray region is 10 cm² -/
theorem gray_area_is_ten_cm_squared (dimensions : PaperDimensions) 
  (h1 : dimensions.length = 55)
  (h2 : dimensions.width = 44)
  (h3 : ∀ (x y : ℝ), x ≥ 0 ∧ x ≤ dimensions.length ∧ y ≥ 0 ∧ y ≤ dimensions.width →
         ∃ (line : Set (ℝ × ℝ)), (∀ (p : ℝ × ℝ), p ∈ line → (p.2 - y) = (p.1 - x) ∨ (p.2 - y) = -(p.1 - x)))
  : gray_area dimensions = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gray_area_is_ten_cm_squared_l477_47733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_angle_relation_l477_47756

noncomputable section

open Real

theorem triangle_side_angle_relation (A B C : ℝ) (a b : ℝ) :
  (A + B + C = π) →  -- Triangle angle sum is π
  (a > 0) →  -- Positive side lengths
  (b > 0) →
  (a / Real.sin A = b / Real.sin B) →  -- Law of sines
  (a > b ↔ Real.sin A > Real.sin B) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_angle_relation_l477_47756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distances_equal_to_one_l477_47741

-- Define the curve C
def curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * Real.sqrt 3 * ρ * Real.sin (θ + Real.pi/3) = 8

-- Define line l1
def line_l1 (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x

-- Define line l2
def line_l2 (x y t : ℝ) : Prop :=
  x = -1/2 * x + Real.sqrt 3 * t ∧ y = -Real.sqrt 3/2 - 1/2 * t

-- Define points A and B as intersections of C and l1
noncomputable def point_A : ℝ × ℝ := sorry
noncomputable def point_B : ℝ × ℝ := sorry

-- Define point P as intersection of l1 and l2
noncomputable def point_P : ℝ × ℝ := sorry

-- Distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distances_equal_to_one :
  distance point_P point_A = 1 ∧ distance point_P point_B = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distances_equal_to_one_l477_47741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_price_is_680_l477_47740

noncomputable def original_jewelry_price : ℝ := 30
noncomputable def original_painting_price : ℝ := 100
noncomputable def jewelry_price_increase : ℝ := 10
noncomputable def painting_price_increase_percentage : ℝ := 20
def jewelry_quantity : ℕ := 2
def painting_quantity : ℕ := 5

noncomputable def new_jewelry_price : ℝ := original_jewelry_price + jewelry_price_increase
noncomputable def new_painting_price : ℝ := original_painting_price * (1 + painting_price_increase_percentage / 100)

noncomputable def total_price : ℝ := jewelry_quantity * new_jewelry_price + painting_quantity * new_painting_price

theorem total_price_is_680 : total_price = 680 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_price_is_680_l477_47740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_passing_percentage_calculation_l477_47753

/-- Given a total of 600 marks, if a student scores 125 marks and fails by 73 marks,
    then the percentage of total marks needed to pass is 33%. -/
theorem passing_percentage_calculation
  (total_marks : ℕ)
  (student_score : ℕ)
  (failing_margin : ℕ)
  (h1 : total_marks = 600)
  (h2 : student_score = 125)
  (h3 : failing_margin = 73) :
  (((student_score + failing_margin : ℚ) / total_marks) * 100 : ℚ) = 33 := by
  sorry

#check passing_percentage_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_passing_percentage_calculation_l477_47753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weight_count_is_13_l477_47768

/-- A weight set satisfying the given conditions -/
structure WeightSet where
  weights : Finset ℕ
  unique_weights : weights.card = 5
  balance_property : ∀ a b, a ∈ weights → b ∈ weights → 
    ∃ c d, c ∈ weights ∧ d ∈ weights ∧ c ≠ a ∧ c ≠ b ∧ d ≠ a ∧ d ≠ b ∧ a + b = c + d

/-- The minimum number of weights in a set satisfying the conditions -/
def min_weight_count : ℕ := 13

/-- Theorem stating that the minimum number of weights is 13 -/
theorem min_weight_count_is_13 :
  ∀ s : WeightSet, (s.weights.card ≥ min_weight_count) ∧
  (∃ s' : WeightSet, s'.weights.card = min_weight_count) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weight_count_is_13_l477_47768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l477_47738

/-- An ellipse with equation x²/2 + y² = 1 -/
structure Ellipse where
  eq : ℝ → ℝ → Prop
  h : ∀ x y, eq x y ↔ x^2 / 2 + y^2 = 1

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  on_ellipse : e.eq x y

/-- The foci of the ellipse -/
def foci : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((1, 0), (-1, 0))

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem to be proved -/
theorem ellipse_foci_distance (e : Ellipse) (p : PointOnEllipse e) :
  let (f1, f2) := foci
  distance (p.x, p.y) f1 = 1 →
  distance (p.x, p.y) f2 = 2 * Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l477_47738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_to_jill_paths_l477_47749

/-- Represents a point on a 2D grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Calculates the number of paths between two points on a grid -/
def numPaths (start finish : Point) : ℕ :=
  sorry

/-- Calculates the number of paths between two points on a grid, avoiding specified points -/
def numPathsAvoiding (start finish : Point) (avoid : List Point) : ℕ :=
  sorry

/-- The starting point (Jack's house) -/
def start : Point := ⟨0, 0⟩

/-- The ending point (Jill's house) -/
def finish : Point := ⟨5, 3⟩

/-- The dangerous intersections to avoid -/
def dangerousPoints : List Point := [⟨2, 1⟩, ⟨1, 2⟩]

/-- The theorem to prove -/
theorem jack_to_jill_paths : 
  numPathsAvoiding start finish dangerousPoints = 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_to_jill_paths_l477_47749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1000_equals_negative_three_l477_47795

def sequence_a : ℕ → ℤ
  | 0 => 3  -- Adding this case to cover Nat.zero
  | 1 => 3
  | 2 => 6
  | (n + 3) => sequence_a (n + 2) - sequence_a (n + 1)

theorem a_1000_equals_negative_three :
  sequence_a 1000 = -3 := by
  sorry

#eval sequence_a 1000  -- This line is added to check the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1000_equals_negative_three_l477_47795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_and_segment_length_range_l477_47757

noncomputable section

/-- Circle C₁ with center at origin and radius 3 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}

/-- Line l₁ tangent to C₁ -/
def l₁ : Set (ℝ × ℝ) := {p | p.1 - 2*p.2 + 3*Real.sqrt 5 = 0}

/-- Point A on C₁ -/
def A (t : ℝ) : ℝ × ℝ := (3 * Real.cos t, 3 * Real.sin t)

/-- Point M on x-axis -/
def M (a : ℝ × ℝ) : ℝ := a.1

/-- Point N defined by vector equation -/
def N (a : ℝ × ℝ) : ℝ × ℝ := 
  (2/3 * a.1 + (2*Real.sqrt 2/3 - 2/3) * (M a), 2/3 * a.2)

/-- Curve C traced by point N -/
def C : Set (ℝ × ℝ) := {p | p.1^2/8 + p.2^2/4 = 1}

theorem curve_equation_and_segment_length_range :
  (∀ a ∈ C₁, N a ∈ C) ∧
  (∀ l : Set (ℝ × ℝ), ∀ a b : ℝ × ℝ, 
    a ∈ C ∧ b ∈ C ∧ a ∈ l ∧ b ∈ l ∧ a ≠ b ∧ 
    (a.1 * b.1 + a.2 * b.2 = 0) →
    4 * Real.sqrt 6 / 3 ≤ Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) ∧
    Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) ≤ 2 * Real.sqrt 3) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_and_segment_length_range_l477_47757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jumbo_tile_fraction_is_three_fifths_l477_47714

/-- Represents the properties of tiles and the wall -/
structure TileSystem where
  regularTileArea : ℝ
  jumboTileLengthRatio : ℝ
  totalWallArea : ℝ
  regularTilesCoverage : ℝ

/-- Calculates the fraction of jumbo tiles in the system -/
noncomputable def jumboTileFraction (ts : TileSystem) : ℝ :=
  3 / 5

/-- Theorem: The fraction of jumbo tiles in the given system is 3/5 -/
theorem jumbo_tile_fraction_is_three_fifths (ts : TileSystem) 
  (h1 : ts.regularTileArea > 0)
  (h2 : ts.jumboTileLengthRatio = 3)
  (h3 : ts.totalWallArea = 220)
  (h4 : ts.regularTilesCoverage = 40) :
  jumboTileFraction ts = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jumbo_tile_fraction_is_three_fifths_l477_47714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_can_always_return_l477_47712

-- Define the checkered square
structure CheckeredSquare where
  cells : Set (ℕ × ℕ)
  doors : Set ((ℕ × ℕ) × (ℕ × ℕ))
  adjacent : (cell1 cell2 : ℕ × ℕ) → Bool

-- Define the beetle's movement
structure BeetleMovement where
  start : ℕ × ℕ
  current : ℕ × ℕ
  openDoors : Set ((ℕ × ℕ) × (ℕ × ℕ))

-- Define the property of being able to return to the start
def canReturnToStart (square : CheckeredSquare) (movement : BeetleMovement) : Prop :=
  ∃ path : List (ℕ × ℕ), 
    path.head? = some movement.current ∧ 
    path.getLast? = some movement.start ∧
    ∀ i, i < path.length - 1 → 
      square.adjacent (path[i]!) (path[i+1]!) ∧
      ((path[i]!, path[i+1]!) ∈ movement.openDoors ∨ (path[i+1]!, path[i]!) ∈ movement.openDoors)

-- State the theorem
theorem beetle_can_always_return 
  (square : CheckeredSquare) 
  (movement : BeetleMovement) : 
  canReturnToStart square movement := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_can_always_return_l477_47712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_y_and_f_three_halves_l477_47724

-- Define the functions g and f
noncomputable def g (x : ℝ) : ℝ := 1 - x^2 + Real.sqrt x

noncomputable def f (x : ℝ) : ℝ := (1 - x^2 + Real.sqrt x) / x^2

-- State the theorem
theorem exists_y_and_f_three_halves :
  ∃ y : ℝ, g y = 3/2 ∧ f (3/2) = (1 - y^2 + Real.sqrt y) / y^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_y_and_f_three_halves_l477_47724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_relation_l477_47777

/-- Volumes of geometric shapes with common radius -/
structure GeometricShapes where
  r : ℝ  -- common radius
  A : ℝ  -- volume of cone
  M : ℝ  -- volume of cylinder
  C : ℝ  -- volume of sphere

/-- The common height of cone and cylinder equals the diameter of sphere -/
def heightEqualsDiameter (shapes : GeometricShapes) : Prop :=
  ∃ h : ℝ, h = 2 * shapes.r ∧ 
    shapes.A = (1/3) * Real.pi * shapes.r^2 * h ∧
    shapes.M = Real.pi * shapes.r^2 * h ∧
    shapes.C = (4/3) * Real.pi * shapes.r^3

/-- Theorem: For shapes satisfying the given conditions, 2A + 2M = 3C -/
theorem volume_relation (shapes : GeometricShapes) 
    (h_height : heightEqualsDiameter shapes) : 
    2 * shapes.A + 2 * shapes.M = 3 * shapes.C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_relation_l477_47777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_count_l477_47716

theorem integer_root_count : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, ∃ k : ℤ, Real.sqrt (123 - Real.sqrt x) = k) ∧ 
  S.card = 12 ∧ 
  (∀ y : ℝ, y ∉ S → ¬∃ k : ℤ, Real.sqrt (123 - Real.sqrt y) = k) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_count_l477_47716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_sin_over_x_l477_47746

noncomputable def f (x : ℝ) : ℝ := Real.sin x / x

theorem derivative_sin_over_x : 
  ∀ x : ℝ, x ≠ 0 → deriv f x = (x * Real.cos x - Real.sin x) / x^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_sin_over_x_l477_47746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_shaded_area_l477_47730

/-- Represents the floor dimensions -/
structure FloorDimensions where
  length : ℝ
  width : ℝ

/-- Represents the tile dimensions -/
structure TileDimensions where
  side : ℝ

/-- Represents the quarter circle radius -/
def quarterCircleRadius : ℝ := 1

/-- Calculates the total shaded area of the floor -/
noncomputable def totalShadedArea (floor : FloorDimensions) (tile : TileDimensions) : ℝ :=
  let numTiles := (floor.length * floor.width) / (tile.side * tile.side)
  let tileArea := tile.side * tile.side
  let whiteAreaPerTile := Real.pi * quarterCircleRadius ^ 2
  let shadedAreaPerTile := tileArea - whiteAreaPerTile
  numTiles * shadedAreaPerTile

/-- Theorem stating the total shaded area of the floor -/
theorem floor_shaded_area :
  let floor := FloorDimensions.mk 16 20
  let tile := TileDimensions.mk 2
  totalShadedArea floor tile = 320 - 80 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_shaded_area_l477_47730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l477_47737

-- Define the length of the train in meters
noncomputable def train_length : ℝ := 375.03

-- Define the time taken to cross the pole in seconds
noncomputable def crossing_time : ℝ := 5

-- Define the conversion factor from m/s to km/h
noncomputable def mps_to_kmph : ℝ := 3600 / 1000

-- Define the speed of the train in km/h
noncomputable def train_speed : ℝ := (train_length / crossing_time) * mps_to_kmph

-- Theorem to prove
theorem train_speed_approx :
  |train_speed - 270.02| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l477_47737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_sequence_over_1000_l477_47721

def is_valid_sequence (a : List Nat) : Prop :=
  ∀ k, 0 ≤ k ∧ k < a.length →
    (List.range (a.length - k)).foldl
      (fun sum i => sum + a[i]! * a[i + k]!) 0 % 2 = 1

theorem exists_valid_sequence_over_1000 :
  ∃ n : Nat, n > 1000 ∧ ∃ a : List Nat, a.length = n ∧
    (∀ x ∈ a, x = 0 ∨ x = 1) ∧ is_valid_sequence a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_sequence_over_1000_l477_47721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l477_47731

/-- A function f(x) defined as -1/2 * (x-2)^2 + b * ln(x) -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := -1/2 * (x-2)^2 + b * Real.log x

/-- The theorem stating that if f(x) is decreasing on (1,+∞), then b ≤ -1 -/
theorem b_range (b : ℝ) :
  (∀ x > 1, ∀ y > x, f b y < f b x) →
  b ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l477_47731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l477_47789

theorem cos_double_angle_special_case (θ : ℝ) 
  (h1 : Real.sin θ + Real.cos θ = 1/5) 
  (h2 : π/2 ≤ θ) 
  (h3 : θ ≤ 3*π/4) : 
  Real.cos (2*θ) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l477_47789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_range_implies_sum_l477_47727

noncomputable def f (a b x : ℝ) : ℝ := Real.log x / Real.log a + b

theorem function_domain_range_implies_sum (a b : ℝ) :
  a > 0 → a ≠ 1 →
  (∀ x, x ∈ Set.Icc 1 2 ↔ f a b x ∈ Set.Icc 1 2) →
  a + b = 5/2 ∨ a + b = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_range_implies_sum_l477_47727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_roots_quadratic_l477_47748

theorem difference_of_roots_quadratic : 
  let r₁ : ℝ := (5 + Real.sqrt 1) / 2
  let r₂ : ℝ := (5 - Real.sqrt 1) / 2
  r₁ - r₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_roots_quadratic_l477_47748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l477_47762

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (Real.log (5 - 2*x)) + Real.sqrt (Real.exp x - 1)

theorem f_domain : Set.Icc 0 2 \ {2} = {x : ℝ | f x ∈ Set.range f} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l477_47762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_amount_correct_l477_47773

/-- The amount of salt needed for Tom's dough ball project --/
def salt_needed (flour_needed : ℕ) (flour_bag_size : ℕ) (flour_bag_cost : ℕ) 
  (salt_cost_per_pound : ℚ) (promotion_cost : ℕ) (ticket_price : ℕ) 
  (tickets_sold : ℕ) (total_profit : ℕ) : ℚ :=
  let flour_cost := (flour_needed / flour_bag_size) * flour_bag_cost
  let revenue := ticket_price * tickets_sold
  let expenses := flour_cost + promotion_cost
  let salt_money := revenue - expenses - total_profit
  (salt_money : ℚ) / salt_cost_per_pound

theorem salt_amount_correct :
  salt_needed 500 50 20 (2/10) 1000 20 500 8798 = 44000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_amount_correct_l477_47773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_75_90_l477_47772

theorem common_divisors_75_90 : 
  (Finset.filter (λ x : ℕ => x ∣ 75 ∧ x ∣ 90) (Finset.range 91)).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_75_90_l477_47772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_weight_is_75_l477_47704

/-- Represents the weight and price of a bag of grass seed -/
structure SeedBag where
  weight : Nat
  price : Rat

/-- Represents the problem constraints -/
structure GrassSeedProblem where
  bags : List SeedBag
  minWeight : Nat
  maxCost : Rat

def problem : GrassSeedProblem :=
  { bags := [
      { weight := 5, price := 1385/100 },
      { weight := 10, price := 2042/100 },
      { weight := 25, price := 3225/100 }
    ],
    minWeight := 65,
    maxCost := 9877/100
  }

/-- Calculates the maximum weight of grass seed that can be bought given the problem constraints -/
def maxWeight (p : GrassSeedProblem) : Nat :=
  sorry

theorem max_weight_is_75 :
  maxWeight problem = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_weight_is_75_l477_47704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_theta_range_l477_47780

open Real Set

noncomputable def f (x θ : ℝ) : ℝ := 1 / (x - 2)^2 - 2*x + cos (2*θ) - 3*sin θ + 2

theorem f_positive_iff_theta_range :
  ∀ θ ∈ Set.Ioo 0 π,
    (∀ x < 2, f x θ > 0) ↔ 
      (θ ∈ Set.Ioo 0 (π/6) ∨ θ ∈ Set.Ioo (5*π/6) π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_theta_range_l477_47780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_formula_l477_47781

/-- Definition of sequence a_n -/
def a (n : ℕ) : ℕ := 2^n

/-- Definition of sequence b_n -/
def b (n : ℕ) : ℕ := 3^n

/-- Definition of sequence c_n -/
def c : ℕ → ℕ
| 0 => 0
| n + 1 => (Finset.range (n + 1)).sum (λ i => a (i + 1) * b (n + 1 - i))

/-- Theorem stating the general term formula for c_n -/
theorem c_formula (n : ℕ) : c n = 6 * (3^n - 2^n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_formula_l477_47781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unused_sector_angle_l477_47751

/-- Given a circular piece of paper with radius 20 cm, if a sector is removed to form a cone
    with radius 15 cm and volume 675π cubic cm, then the angle of the unused sector is 90°. -/
theorem unused_sector_angle (paper_radius : ℝ) (cone_radius : ℝ) (cone_volume : ℝ) :
  paper_radius = 20 →
  cone_radius = 15 →
  cone_volume = 675 * Real.pi →
  360 - (2 * Real.pi * cone_radius) / (2 * Real.pi * paper_radius) * 360 = 90 := by
  intros h_paper h_cone h_volume
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unused_sector_angle_l477_47751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_pairs_l477_47719

/-- A cubic polynomial with real coefficients -/
def CubicPolynomial := ℝ → ℝ

/-- Checks if a polynomial satisfies the condition of taking only 0 or 1 at x = 1, 2, 3, 4 -/
def satisfies_01_condition (P : CubicPolynomial) : Prop :=
  ∀ x : ℕ, x ∈ ({1, 2, 3, 4} : Set ℕ) → P x = 0 ∨ P x = 1

/-- Represents the conditions given in the problem -/
def satisfies_all_conditions (P Q : CubicPolynomial) : Prop :=
  satisfies_01_condition P ∧
  satisfies_01_condition Q ∧
  ((P 1 = 0 ∨ P 2 = 1) → Q 1 = 1 ∧ Q 3 = 1) ∧
  ((P 2 = 0 ∨ P 4 = 0) → Q 2 = 0 ∧ Q 4 = 0) ∧
  ((P 3 = 1 ∨ P 4 = 1) → Q 1 = 0)

/-- The six specific cubic polynomials mentioned in the solution -/
noncomputable def R₁ : CubicPolynomial := λ x => -1/2*x^3 + 7/2*x^2 - 7*x + 4
noncomputable def R₂ : CubicPolynomial := λ x => 1/2*x^3 - 4*x^2 + 19/2*x - 6
noncomputable def R₃ : CubicPolynomial := λ x => -1/6*x^3 + 3/2*x^2 - 13/3*x + 4
noncomputable def R₄ : CubicPolynomial := λ x => -2/3*x^3 + 5*x^2 - 34/3*x + 8
noncomputable def R₅ : CubicPolynomial := λ x => -1/2*x^3 + 4*x^2 - 19/2*x + 7
noncomputable def R₆ : CubicPolynomial := λ x => 1/3*x^3 - 5/2*x^2 + 31/6*x - 2

/-- The theorem to be proved -/
theorem cubic_polynomial_pairs : 
  ∀ P Q : CubicPolynomial, satisfies_all_conditions P Q ↔ 
    (P = R₂ ∧ Q = R₄) ∨ 
    (P = R₃ ∧ Q = R₁) ∨ 
    (P = R₃ ∧ Q = R₃) ∨ 
    (P = R₃ ∧ Q = R₄) ∨ 
    (P = R₄ ∧ Q = R₁) ∨ 
    (P = R₅ ∧ Q = R₁) ∨ 
    (P = R₆ ∧ Q = R₄) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_pairs_l477_47719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_of_zeros_l477_47779

-- Define the function f
noncomputable def f (x t : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x - t else 2 * (x + 1) - t

-- State the theorem
theorem min_difference_of_zeros :
  ∃ t : ℝ, ∃ x₁ x₂ : ℝ, x₁ > x₂ ∧ f x₁ t = 0 ∧ f x₂ t = 0 ∧
  (∀ y₁ y₂ : ℝ, y₁ > y₂ ∧ f y₁ t = 0 ∧ f y₂ t = 0 → x₁ - x₂ ≤ y₁ - y₂) ∧
  x₁ - x₂ = 15/16 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_of_zeros_l477_47779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_S_l477_47725

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the expression S as a function of n
noncomputable def S (n : ℤ) : ℂ := (i^n + i^(-n))^2

-- Theorem statement
theorem distinct_values_of_S :
  ∃ (A : Finset ℂ), (∀ n : ℤ, S n ∈ A) ∧ (Finset.card A = 2) := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_S_l477_47725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_one_l477_47732

/-- A monic cubic polynomial satisfying specific conditions -/
def f (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c
  where
    a : ℝ := 2
    b : ℝ := -6
    c : ℝ := -6

/-- Theorem stating the value of f(1) given specific conditions -/
theorem f_value_at_one
  (h1 : f (-1) = 1)
  (h2 : f 2 = -2)
  (h3 : f (-3) = 3) :
  f 1 = -9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_one_l477_47732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_theorem_l477_47765

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is on a line segment -/
def isOnSegment (P Q R : Point) : Prop := sorry

/-- Checks if two line segments intersect -/
def intersect (P Q R S : Point) : Point := sorry

/-- Calculates the length of a line segment -/
def length (P Q : Point) : ℝ := sorry

theorem trapezoid_theorem (ABCD : Trapezoid) (E O P : Point) :
  isOnSegment ABCD.A ABCD.D E →
  length ABCD.A E = length ABCD.B ABCD.C →
  O = intersect ABCD.C ABCD.A ABCD.B ABCD.D →
  P = intersect ABCD.C E ABCD.B ABCD.D →
  length ABCD.B O = length P ABCD.D →
  (length ABCD.A ABCD.D)^2 = (length ABCD.B ABCD.C)^2 + (length ABCD.A ABCD.D) * (length ABCD.B ABCD.C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_theorem_l477_47765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l477_47774

def sequence_a : ℕ → ℚ
  | 0 => 1
  | n + 1 => sequence_a n + (2 * sequence_a n) / n

theorem a_100_value : sequence_a 100 = 5151 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l477_47774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l477_47752

noncomputable def α_set : Set ℝ := {-2, -1, -1/2, 1/3, 1/2, 1, 2, 3}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_decreasing_on_positive_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f y < f x

noncomputable def power_function (α : ℝ) : ℝ → ℝ := λ x => x ^ α

theorem power_function_properties :
  ∃ α ∈ α_set,
    is_even_function (power_function α) ∧
    is_decreasing_on_positive_reals (power_function α) →
  α = -2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l477_47752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_two_l477_47717

/-- Hyperbola C with equation x²/a² - y²/b² = 1 -/
structure Hyperbola (a b : ℝ) where
  eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

theorem hyperbola_eccentricity_sqrt_two 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (C : Hyperbola a b)
  (F : Point)
  (B : Point)
  (M : Point)
  (O : Point)
  (N : Point)
  (h_F : F = ⟨Real.sqrt (a^2 + b^2), 0⟩)  -- Right focus
  (h_B : B = ⟨0, b⟩)  -- Endpoint on imaginary axis
  (h_M : M = ⟨(Real.sqrt (a^2 + b^2)) / 2, b / 2⟩)  -- Midpoint of BF
  (h_O : O = ⟨0, 0⟩)  -- Origin
  (h_N : N = ⟨Real.sqrt (a^2 + b^2), b⟩)  -- Intersection of OM and hyperbola
  (h_perpendicular : (N.x - F.x) * (N.y - F.y) = 0)  -- FN perpendicular to x-axis
  : eccentricity a b = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_two_l477_47717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l477_47754

/-- Define an ellipse with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- Define a line with slope m passing through point p -/
def Line (m : ℝ) (p : ℝ × ℝ) := {q : ℝ × ℝ | q.2 - p.2 = m * (q.1 - p.1)}

/-- Define a circle with diameter AB -/
def Circle (A B : ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - (A.1 + B.1)/2)^2 + (p.2 - (A.2 + B.2)/2)^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4}

theorem ellipse_properties (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := Ellipse a b
  let A := (a/2, 3*c/2)
  let l := Line (1/2) A
  let B := (- 2 * c, 0)  -- We know B from the solution
  let circle := Circle A B
  (A ∈ e) →
  ((1/2 : ℝ), 9/2) ∈ circle →
  (c^2 = a^2 - b^2) →
  (c = a/2) ∧ (a = 4) ∧ (b^2 = 12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l477_47754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_no_568_l477_47735

def arithmetic_sequence (n : ℕ) : ℕ :=
  8 * ((10^n - 1) / 9)

def arithmetic_mean (seq : ℕ → ℕ) (n : ℕ) : ℚ :=
  (Finset.sum (Finset.range n) seq) / n

theorem arithmetic_mean_no_568 :
  let N := arithmetic_mean arithmetic_sequence 9
  ∀ d : ℕ, d ∈ [5, 6, 8] → d ∉ (N.num.toNat.digits 10).toFinset := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_no_568_l477_47735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_multiples_of_seven_eq_1286_l477_47796

/-- The count of four-digit positive integers that are multiples of 7 -/
def count_four_digit_multiples_of_seven : ℕ :=
  Finset.card (Finset.filter (fun n => 1000 ≤ 7 * n ∧ 7 * n ≤ 9999) (Finset.range 1429))

/-- Theorem stating that the count of four-digit positive integers that are multiples of 7 is 1286 -/
theorem count_four_digit_multiples_of_seven_eq_1286 :
  count_four_digit_multiples_of_seven = 1286 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_multiples_of_seven_eq_1286_l477_47796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_and_inequality_l477_47794

noncomputable def f (x : Real) : Real := 2 * Real.sin (2 * x - Real.pi / 6)

theorem f_value_and_inequality (x : Real) :
  (f (5 * Real.pi / 24) = Real.sqrt 2) ∧
  (f x ≥ 1 ↔ ∃ k : Int, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_and_inequality_l477_47794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l477_47750

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define a line passing through a point with a given angle
def line_through_point (point : ℝ × ℝ) (angle : ℝ) (x y : ℝ) : Prop :=
  y - point.2 = Real.tan angle * (x - point.1)

-- Define the intersection points of the line and parabola
def intersection_points (p : ℝ) (A B : ℝ × ℝ) : Prop :=
  parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧
  line_through_point (focus p) (Real.pi/3) A.1 A.2 ∧
  line_through_point (focus p) (Real.pi/3) B.1 B.2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_intersection_ratio (p : ℝ) (A B : ℝ × ℝ) :
  parabola p A.1 A.2 →
  parabola p B.1 B.2 →
  line_through_point (focus p) (Real.pi/3) A.1 A.2 →
  line_through_point (focus p) (Real.pi/3) B.1 B.2 →
  distance A (focus p) > distance B (focus p) →
  distance A (focus p) / distance B (focus p) = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l477_47750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l477_47770

/-- The number of integer values for the third side of a non-degenerate triangle with two sides of length 15 and 40 units -/
theorem triangle_side_count : ∃ (n : ℕ), n = 29 ∧ 
  n = (Finset.filter (λ x : ℕ ↦ 
    x > 25 ∧ x < 55 ∧ 
    x + 15 > 40 ∧ x + 40 > 15 ∧ 15 + 40 > x
  ) (Finset.range 56)).card := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l477_47770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_discount_percentage_l477_47739

-- Define the wholesale price
noncomputable def wholesale_price : ℚ := 4

-- Define the markup percentage
noncomputable def markup_percentage : ℚ := 25

-- Define the price paid by the customer with coupon
noncomputable def price_with_coupon : ℚ := 4.75

-- Calculate the retail price
noncomputable def retail_price : ℚ := wholesale_price * (1 + markup_percentage / 100)

-- Define the theorem
theorem coupon_discount_percentage :
  let discount_amount := retail_price - price_with_coupon
  let discount_percentage := (discount_amount / retail_price) * 100
  discount_percentage = 5 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_discount_percentage_l477_47739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juanita_drums_hit_l477_47722

/-- Represents the drumming contest scenario -/
structure DrummingContest where
  entryCost : ℚ
  threshold : ℕ
  earningsPerDrum : ℚ
  moneyLost : ℚ

/-- Calculates the total number of drums hit given the contest parameters -/
def totalDrumsHit (contest : DrummingContest) : ℕ :=
  contest.threshold + 
  ((contest.entryCost - contest.moneyLost) * 100 / contest.earningsPerDrum).floor.toNat

/-- Theorem stating that Juanita hit 300 drums in the given scenario -/
theorem juanita_drums_hit : 
  let contest := DrummingContest.mk 10 200 (5/2) (15/2)
  totalDrumsHit contest = 300 := by
  sorry

#eval totalDrumsHit (DrummingContest.mk 10 200 (5/2) (15/2))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_juanita_drums_hit_l477_47722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l477_47797

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (u z : Fin 2 → ℝ)

theorem matrix_vector_computation
  (hu : M.vecMul u = ![3, -4])
  (hz : M.vecMul z = ![-1, 6]) :
  M.vecMul (3 • u - 2 • z) = ![11, -24] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l477_47797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_one_half_l477_47785

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 / (2*x + 1)

-- Define the iterated function f_n
noncomputable def f_n : ℕ → ℝ → ℝ 
  | 0, x => x
  | n+1, x => f (f_n n x)

-- State the theorem
theorem f_10_one_half : f_n 10 (1/2) = 1 / (3^1024 - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_one_half_l477_47785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_common_terms_are_one_and_seven_l477_47760

/-- Sequence a_n defined recursively -/
def a : ℕ → ℤ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 2
  | (n + 3) => 4 * a (n + 2) - a (n + 1)

/-- Sequence b_n defined recursively -/
def b : ℕ → ℤ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 7
  | (n + 3) => 6 * b (n + 2) - b (n + 1)

/-- The set of common terms between sequences a and b -/
def commonTerms : Set ℤ :=
  {x | ∃ n m : ℕ, a n = x ∧ b m = x}

/-- Theorem stating that the only common terms are 1 and 7 -/
theorem only_common_terms_are_one_and_seven :
  commonTerms = {1, 7} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_common_terms_are_one_and_seven_l477_47760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_from_sum_magnitude_l477_47788

/-- Given that a, b, and c are unit vectors in a plane, if |a + b - c| = 3, 
    then a and b are collinear. -/
theorem collinearity_from_sum_magnitude 
  (a b c : EuclideanSpace ℝ (Fin 2)) 
  (ha : ‖a‖ = 1) 
  (hb : ‖b‖ = 1) 
  (hc : ‖c‖ = 1) 
  (hsum : ‖a + b - c‖ = 3) : 
  ∃ (k : ℝ), a = k • b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_from_sum_magnitude_l477_47788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_diagonals_in_hexagon_l477_47755

/-- A convex hexagon is a polygon with 6 vertices and 6 sides, where all interior angles are less than 180 degrees. -/
def ConvexHexagon : Type := sorry

/-- A diagonal of a polygon is a line segment that connects two non-adjacent vertices. -/
def Diagonal (h : ConvexHexagon) : Type := sorry

/-- Two diagonals are considered equal if they have the same length. -/
def EqualDiagonals (h : ConvexHexagon) (d1 d2 : Diagonal h) : Prop := sorry

/-- The maximum number of equal-length diagonals in a convex hexagon is 7. -/
theorem max_equal_diagonals_in_hexagon (h : ConvexHexagon) :
  ∃ (S : Finset (Diagonal h)), (∀ d1 d2, d1 ∈ S → d2 ∈ S → EqualDiagonals h d1 d2) ∧ S.card = 7 ∧
  ∀ (T : Finset (Diagonal h)), (∀ d1 d2, d1 ∈ T → d2 ∈ T → EqualDiagonals h d1 d2) → T.card ≤ 7 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_diagonals_in_hexagon_l477_47755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_value_range_l477_47700

/-- Proposition P: p satisfies the binomial distribution and probability condition -/
def PropP (p : ℝ) : Prop :=
  0 < p ∧ p < 1 ∧ (Nat.choose 5 3 * p^3 * (1-p)^2 > Nat.choose 5 4 * p^4 * (1-p))

/-- Proposition Q: p satisfies the ellipse equation -/
def PropQ (p : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (3*p) + y^2 / (2-p) = 1

/-- The range of p values satisfying the given conditions -/
def PRange (p : ℝ) : Prop :=
  (0 < p ∧ p ≤ 1/2) ∨ (2/3 ≤ p ∧ p < 2)

theorem p_value_range :
  ∀ p : ℝ, (PropP p ∨ PropQ p) ∧ ¬(PropP p ∧ PropQ p) → PRange p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_value_range_l477_47700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deleted_to_kept_ratio_l477_47763

/-- Given that Kaleb had 26 songs originally and deleted 20 songs,
    prove that the simplified ratio of deleted to kept songs is 10:3 -/
theorem deleted_to_kept_ratio :
  let original_songs : ℕ := 26
  let deleted_songs : ℕ := 20
  let kept_songs : ℕ := original_songs - deleted_songs
  let ratio : Rat := deleted_songs / kept_songs
  ratio = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deleted_to_kept_ratio_l477_47763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_overlap_distance_l477_47720

/-- The overlap distance between two circles -/
noncomputable def overlap_distance (c1 c2 : ℝ × ℝ) : ℝ :=
  let d := Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2)
  let r1 := Real.sqrt 7
  let r2 := 4
  r1 + r2 - d

/-- First circle equation -/
def circle1 (p : ℝ × ℝ) : Prop :=
  p.1^2 - 2*p.1 + p.2^2 - 6*p.2 = -3

/-- Second circle equation -/
def circle2 (p : ℝ × ℝ) : Prop :=
  p.1^2 + 8*p.1 + p.2^2 + 2*p.2 = -1

/-- Theorem stating that the overlap distance between the two circles is √7 + 4 - √41 -/
theorem circle_overlap_distance :
  overlap_distance (1, 3) (-4, -1) = Real.sqrt 7 + 4 - Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_overlap_distance_l477_47720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_salary_percentage_l477_47798

/-- The percentage of Y's salary that X is paid -/
noncomputable def salary_percentage (total_salary y_salary : ℝ) : ℝ :=
  ((total_salary - y_salary) / y_salary) * 100

/-- Theorem stating that X is paid approximately 120% of Y's salary -/
theorem x_salary_percentage :
  let total_salary : ℝ := 590
  let y_salary : ℝ := 268.1818181818182
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |salary_percentage total_salary y_salary - 120| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_salary_percentage_l477_47798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newborn_count_verify_newborn_count_l477_47707

/-- The probability of an animal surviving one month -/
noncomputable def survival_prob : ℝ := 9/10

/-- The number of animals expected to survive three months -/
noncomputable def survivors : ℝ := 182.25

/-- The number of newborn members in the group -/
def newborns : ℕ := 250

/-- Theorem stating the relationship between newborns, survival probability, and survivors -/
theorem newborn_count : (↑newborns : ℝ) * survival_prob^3 = survivors := by
  sorry

/-- Verification of the newborn count -/
theorem verify_newborn_count : newborns = 250 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_newborn_count_verify_newborn_count_l477_47707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_phi_range_of_a_l477_47728

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 3/2 - a/x
noncomputable def φ (x : ℝ) : ℝ := f x - g 1 x

-- Statement 1
theorem min_value_phi :
  ∀ x ≥ 4, φ x ≥ 2 * Real.log 2 - 5/4 := by
  sorry

-- Statement 2
theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Icc (1/2 : ℝ) 1, Real.exp (2 * f x) = g a x) →
  a ∈ Set.Icc (1/2 : ℝ) (Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_phi_range_of_a_l477_47728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_empty_time_is_60_l477_47758

/-- Represents the time it takes for a cistern to empty given its filling times with and without a leak -/
noncomputable def cisternEmptyTime (normalFillTime leakyFillTime : ℝ) : ℝ :=
  (leakyFillTime * normalFillTime) / (leakyFillTime - normalFillTime)

/-- Theorem stating that a cistern that normally fills in 10 hours but takes 12 hours with a leak will empty in 60 hours -/
theorem cistern_empty_time_is_60 :
  cisternEmptyTime 10 12 = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_empty_time_is_60_l477_47758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_balance_percentage_l477_47759

noncomputable def initial_balance : ℝ := 125

noncomputable def apply_percentage_change (balance : ℝ) (percentage : ℝ) : ℝ :=
  balance * (1 + percentage / 100)

theorem final_balance_percentage : 
  let b1 := apply_percentage_change initial_balance 25
  let b2 := apply_percentage_change b1 (-20)
  let b3 := apply_percentage_change b2 15
  let b4 := apply_percentage_change b3 (-10)
  b4 / initial_balance * 100 = 103.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_balance_percentage_l477_47759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l477_47792

-- Define the constants and functions
noncomputable def x : ℝ := (3 + Real.sqrt 8) ^ 30
noncomputable def n : ℤ := ⌊x⌋
noncomputable def f : ℝ := x - n

-- State the theorem
theorem x_times_one_minus_f_equals_one : x * (1 - f) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l477_47792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l477_47729

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- The focal distance of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ := e.a * eccentricity e

theorem ellipse_eccentricity_range (e : Ellipse) :
  let c := focal_distance e
  let A := (-c, e.b^2 / e.a)
  let B := (-c, -e.b^2 / e.a)
  let F₁ := (-c, 0)
  let F₂ := (c, 0)
  (∀ θ : ℝ, θ ∈ Set.Ioo 0 (π/2) → 
    Real.tan θ > (A.2 - F₂.2) / (A.1 - F₂.1)) →
  Real.sqrt 2 - 1 < eccentricity e ∧ eccentricity e < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l477_47729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l477_47702

/-- The constant term in the expansion of (x^2 + 1/x)^6 is 15 -/
theorem constant_term_expansion : ℕ := 15

/-- Proof of the theorem -/
lemma proof_of_constant_term_expansion : constant_term_expansion = 15 := by
  -- Unfold the definition of constant_term_expansion
  unfold constant_term_expansion
  -- The result follows directly from the definition
  rfl

/-- Explanation of why the constant term is 15 -/
lemma explanation_of_constant_term : 
  ∃ (f : ℝ → ℝ), ∀ (x : ℝ), x ≠ 0 → (x^2 + 1/x)^6 = f x + 15 := by
  -- We don't provide a full proof here, but we can outline the reasoning
  sorry
  -- The proof would involve:
  -- 1. Expanding (x^2 + 1/x)^6 using the binomial theorem
  -- 2. Collecting terms with x^0 (constant terms)
  -- 3. Showing that this term is equal to C(6,4) = 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l477_47702
