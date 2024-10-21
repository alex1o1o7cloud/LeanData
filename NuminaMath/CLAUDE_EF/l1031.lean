import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l1031_103129

noncomputable def series_term (n : ℕ) : ℝ := (2 * n - 1) * (1 / 2000) ^ (n - 1)

noncomputable def series_sum : ℝ := ∑' n, series_term n

theorem series_sum_value : series_sum = 2003000 / 3996001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l1031_103129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_intersection_l1031_103104

-- Define the function f(x) = √(x^2 - 1)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 1)

-- Define the domain of f
def domain : Set ℝ := {x | x^2 - 1 ≥ 0}

-- Define the range of f
def range : Set ℝ := {y | ∃ x, f x = y}

-- Theorem stating that the intersection of domain and range is [1, +∞)
theorem domain_range_intersection :
  {x ∈ domain | f x ∈ range} = Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_intersection_l1031_103104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_dot_product_ellipse_intersection_l1031_103197

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a vector in 2D space -/
def Vector2D := Point

/-- Calculates the dot product of two vectors -/
def dot_product (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

/-- Theorem: Constant dot product for specific ellipse intersection -/
theorem constant_dot_product_ellipse_intersection (k : ℝ) (A B : Point)
    (h1 : A.x^2 / 8 + A.y^2 / 4 = 1)
    (h2 : B.x^2 / 8 + B.y^2 / 4 = 1)
    (h3 : A.y = k * (A.x - 1))
    (h4 : B.y = k * (B.x - 1))
    : dot_product
        { x := A.x - 11/4, y := A.y }
        { x := B.x - 11/4, y := B.y }
      = -7/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_dot_product_ellipse_intersection_l1031_103197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_even_odd_dice_probability_l1031_103122

theorem equal_even_odd_dice_probability :
  let n : ℕ := 8  -- Total number of dice
  let p : ℚ := 1/2  -- Probability of rolling an even number on a single die
  let Prob : ℚ := (n.choose (n/2)) * p^n  -- Probability of rolling equal number of even and odd numbers
  Prob = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_even_odd_dice_probability_l1031_103122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l1031_103141

/-- The length of a train that passes a stationary man in 8 seconds
    and crosses a 270-metre platform in 20 seconds is 180 metres. -/
theorem train_length (L : ℝ) : 
  (∃ (S : ℝ), S = L / 8 ∧ S = (L + 270) / 20) → L = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l1031_103141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1031_103196

-- Define the function f(x) = 2/x - x
noncomputable def f (x : ℝ) : ℝ := 2/x - x

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 5, x^2 + a*x - 2 > 0) ↔ a > -23/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1031_103196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_dark_for_10_seconds_l1031_103180

/-- Represents the number of revolutions per minute of the searchlight -/
noncomputable def revolutions_per_minute : ℚ := 2

/-- Represents the duration in seconds for which we want to calculate the probability of staying in the dark -/
noncomputable def dark_duration : ℚ := 10

/-- Calculates the probability of staying in the dark for at least the given duration -/
noncomputable def probability_in_dark (rpm : ℚ) (duration : ℚ) : ℚ :=
  duration / (60 / rpm)

/-- Theorem stating that the probability of staying in the dark for at least 10 seconds
    given a searchlight making 2 revolutions per minute is 1/3 -/
theorem probability_in_dark_for_10_seconds :
  probability_in_dark revolutions_per_minute dark_duration = 1/3 := by
  -- Unfold definitions
  unfold probability_in_dark
  unfold revolutions_per_minute
  unfold dark_duration
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_dark_for_10_seconds_l1031_103180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_m_between_zero_and_three_two_satisfies_conditions_l1031_103185

-- Define the inverse proportion function
noncomputable def f (x : ℝ) : ℝ := 6 / x

-- Define the theorem
theorem exists_m_between_zero_and_three :
  ∃ m : ℝ, 0 < m ∧ m < 3 ∧
  ∀ y₁ y₂ : ℝ, (f 3 = y₁ ∧ f m = y₂ ∧ y₁ < y₂) → m ∈ Set.Ioo 0 3 := by
  -- Proof goes here
  sorry

-- Example of a specific value satisfying the conditions
theorem two_satisfies_conditions :
  ∃ y₁ y₂ : ℝ, f 3 = y₁ ∧ f 2 = y₂ ∧ y₁ < y₂ := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_m_between_zero_and_three_two_satisfies_conditions_l1031_103185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l1031_103195

noncomputable def f (x : ℝ) := Real.exp x * Real.cos x

theorem tangent_line_at_zero :
  let f' := λ x => Real.exp x * (Real.cos x - Real.sin x)
  let slope := f' 0
  let y_intercept := f 0
  (λ x y => x - y + 1 = 0) = (λ x y => y - y_intercept = slope * (x - 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l1031_103195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1031_103173

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 2016 + 1 / (x - 2016)) / 2

-- State the theorem
theorem f_minimum_value :
  ∀ x > 2016, f x ≥ 1 ∧ ∃ x₀ > 2016, f x₀ = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1031_103173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_distance_range_l1031_103175

/-- Given two vectors AB and AC in a Euclidean space, with |AB| = 18 and |AC| = 5,
    prove that the minimum value of |BC| is 13 and the maximum value is 23. -/
theorem vector_distance_range (AB AC : EuclideanSpace ℝ (Fin 3)) 
    (h1 : ‖AB‖ = 18) (h2 : ‖AC‖ = 5) : 
    (∃ (BC : EuclideanSpace ℝ (Fin 3)), ‖BC‖ = 13) ∧ 
    (∃ (BC : EuclideanSpace ℝ (Fin 3)), ‖BC‖ = 23) ∧
    (∀ (BC : EuclideanSpace ℝ (Fin 3)), 13 ≤ ‖BC‖ ∧ ‖BC‖ ≤ 23) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_distance_range_l1031_103175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_portions_theorem_l1031_103184

/-- The volume of the milk container in liters -/
def container_volume : ℚ := 2

/-- The volume of one portion in milliliters -/
def portion_volume : ℚ := 200

/-- The number of portions that can be poured from the container -/
def num_portions : ℕ := (container_volume * 1000 / portion_volume).floor.toNat

theorem milk_portions_theorem : num_portions = 10 := by
  rw [num_portions]
  rw [container_volume, portion_volume]
  norm_num
  rfl

#eval num_portions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_portions_theorem_l1031_103184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1031_103105

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Ioo 0 1 ∪ {1} :=
by
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1031_103105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeroes_2023_factorial_l1031_103116

theorem trailing_zeroes_2023_factorial : 
  (63 : ℕ) = ((Nat.floor (2023 / 17 : ℚ) + 
               Nat.floor (2023 / (17^2) : ℚ) + 
               Nat.floor (2023 / (17^3) : ℚ)) / 2 : ℕ) := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

#eval ((Nat.floor (2023 / 17 : ℚ) + 
        Nat.floor (2023 / (17^2) : ℚ) + 
        Nat.floor (2023 / (17^3) : ℚ)) / 2 : ℕ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeroes_2023_factorial_l1031_103116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_b_value_l1031_103188

-- Define the hyperbola
def hyperbola (x y b : ℝ) : Prop := x^2 / 4 - y^2 / b^2 = 1

-- Define the line passing through left focus with slope 2
def line_through_focus (x y : ℝ) : Prop := ∃ c : ℝ, y = 2*x + c

-- Define points A and B on both the hyperbola and the line
def point_A (x₁ y₁ b : ℝ) : Prop := hyperbola x₁ y₁ b ∧ line_through_focus x₁ y₁
def point_B (x₂ y₂ b : ℝ) : Prop := hyperbola x₂ y₂ b ∧ line_through_focus x₂ y₂

-- Define midpoint P
def midpoint_P (x₁ y₁ x₂ y₂ xp yp : ℝ) : Prop := xp = (x₁ + x₂) / 2 ∧ yp = (y₁ + y₂) / 2

-- Define slope of line OP
def slope_OP (xp yp : ℝ) : Prop := yp / xp = 1 / 4

theorem hyperbola_b_value (b x₁ y₁ x₂ y₂ xp yp : ℝ) :
  hyperbola x₁ y₁ b →
  hyperbola x₂ y₂ b →
  line_through_focus x₁ y₁ →
  line_through_focus x₂ y₂ →
  midpoint_P x₁ y₁ x₂ y₂ xp yp →
  slope_OP xp yp →
  b = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_b_value_l1031_103188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_a_l1031_103102

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x + a else -x - 2 * a

theorem solve_for_a (a : ℝ) (h1 : a < 0) (h2 : f a (1 - a) = f a (1 + a)) : a = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_a_l1031_103102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_for_candidate_A_l1031_103181

theorem election_votes_for_candidate_A :
  ∀ (total_votes : ℕ) (invalid_percentage valid_percentage candidate_A_percentage : ℚ),
    total_votes = 1280000 →
    invalid_percentage = 25 / 100 →
    valid_percentage = 1 - invalid_percentage →
    candidate_A_percentage = 60 / 100 →
    ∃ (valid_votes : ℕ) (votes_for_A : ℕ),
      valid_votes = (valid_percentage * ↑total_votes).floor ∧
      votes_for_A = (candidate_A_percentage * ↑valid_votes).floor ∧
      votes_for_A = 576000 :=
by
  intro total_votes invalid_percentage valid_percentage candidate_A_percentage
  intro h_total h_invalid h_valid h_candidate_A
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_for_candidate_A_l1031_103181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_a_l1031_103136

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x^2) - 3 * x) + 1

-- State the theorem
theorem f_negative_a (a : ℝ) (h : f a = 1/3) : f (-a) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_a_l1031_103136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_germination_experiment_l1031_103170

/-- Represents a plot with the number of seeds planted and the germination rate -/
structure Plot where
  seeds : ℕ
  germination_rate : ℚ

/-- Calculates the average germination rate across multiple plots -/
def average_germination_rate (plots : List Plot) : ℚ :=
  let total_germinated := plots.map (fun p => (p.seeds : ℚ) * p.germination_rate) |>.sum
  let total_planted := plots.map (fun p => p.seeds) |>.sum
  total_germinated / (total_planted : ℚ)

/-- The main theorem stating that the average germination rate for the given plots is 28.25% -/
theorem average_germination_experiment :
  let plots : List Plot := [
    { seeds := 300, germination_rate := 25/100 },
    { seeds := 200, germination_rate := 40/100 },
    { seeds := 500, germination_rate := 30/100 },
    { seeds := 400, germination_rate := 35/100 },
    { seeds := 600, germination_rate := 20/100 }
  ]
  average_germination_rate plots = 2825/10000 := by
  sorry

#eval average_germination_rate [
  { seeds := 300, germination_rate := 25/100 },
  { seeds := 200, germination_rate := 40/100 },
  { seeds := 500, germination_rate := 30/100 },
  { seeds := 400, germination_rate := 35/100 },
  { seeds := 600, germination_rate := 20/100 }
]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_germination_experiment_l1031_103170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outbound_journey_time_outbound_journey_minutes_l1031_103106

/-- Calculates the time taken for the outbound journey in a round trip -/
noncomputable def outbound_time (outbound_speed inbound_speed : ℝ) (total_time : ℝ) : ℝ :=
  (total_time * outbound_speed * inbound_speed) / (outbound_speed + inbound_speed)

/-- Theorem stating that under the given conditions, the outbound journey takes 108 minutes -/
theorem outbound_journey_time :
  outbound_time 80 120 3 = 1.8 := by
  sorry

/-- Corollary converting the outbound time to minutes -/
theorem outbound_journey_minutes :
  outbound_time 80 120 3 * 60 = 108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_outbound_journey_time_outbound_journey_minutes_l1031_103106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l1031_103135

noncomputable section

-- Define the points
def P : ℝ × ℝ := (1, 2)
def A : ℝ × ℝ := (-1, -1)

-- Define the reference line
def reference_line (x : ℝ) : ℝ := (1/2) * x

-- Define the property of equal intercepts
def has_equal_intercepts (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ a * c = b * c

-- Define the property of passing through a point
def passes_through (a b c : ℝ) (p : ℝ × ℝ) : Prop :=
  a * p.1 + b * p.2 + c = 0

-- Define the property of having double angle of inclination
def has_double_inclination (a b : ℝ) (ref : ℝ → ℝ) : Prop :=
  (a / b) = (2 * (ref 1 - ref 0)) / (1 - (ref 1 - ref 0)^2)

theorem line_equations :
  (∃ (a b c : ℝ), (passes_through a b c P ∧ has_equal_intercepts a b c) ∧
    ((a = 2 ∧ b = -1 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -3))) ∧
  (∃ (a b c : ℝ), passes_through a b c A ∧ has_double_inclination a b reference_line ∧
    a = 4 ∧ b = -3 ∧ c = 1) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l1031_103135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_existence_l1031_103151

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define properties
def isIsoscelesRight (t : Triangle) : Prop :=
  (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 = (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2

noncomputable def rotatePoint (p : Point) (center : Point) (angle : ℝ) : Point :=
  { x := center.x + (p.x - center.x) * Real.cos angle - (p.y - center.y) * Real.sin angle,
    y := center.y + (p.x - center.x) * Real.sin angle + (p.y - center.y) * Real.cos angle }

noncomputable def rotateTriangle (t : Triangle) (center : Point) (angle : ℝ) : Triangle :=
  { A := center,  -- Assuming A is the center of rotation
    B := rotatePoint t.B center angle,
    C := rotatePoint t.C center angle }

-- Main theorem
theorem isosceles_right_triangle_existence 
  (ABC : Triangle) (ADE : Triangle) (h1 : isIsoscelesRight ABC) (h2 : isIsoscelesRight ADE) 
  (h3 : ABC ≠ ADE) (angle : ℝ) :
  ∃ (M : Point), 
    let rotatedADE := rotateTriangle ADE ABC.A angle
    let B := ABC.B
    let D := rotatedADE.B
    let E := rotatedADE.C
    let C := ABC.C
    M.x ≥ min C.x E.x ∧ M.x ≤ max C.x E.x ∧
    M.y ≥ min C.y E.y ∧ M.y ≤ max C.y E.y ∧
    isIsoscelesRight { A := B, B := M, C := D } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_existence_l1031_103151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_pentagon_l1031_103163

def Pentagon (A B C D E : ℝ) : Prop :=
  A + B + C + D + E = 540

theorem largest_angle_in_pentagon (A B C D E : ℝ) 
  (h_pentagon : Pentagon A B C D E)
  (h_A : A = 75)
  (h_B : B = 95)
  (h_D : D = C + 10)
  (h_E : E = 2 * C + 20) :
  max A (max B (max C (max D E))) = 190 := by
  sorry

#check largest_angle_in_pentagon

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_pentagon_l1031_103163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_square_sum_of_cubes_l1031_103138

theorem unique_prime_square_sum_of_cubes : 
  ∃! p : ℕ, Nat.Prime p ∧ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ p^2 = a^3 + b^3 :=
by
  -- The unique prime p is 3
  use 3
  constructor
  · -- Prove that 3 satisfies the conditions
    constructor
    · exact Nat.prime_three -- 3 is prime
    · use 1, 2 -- a = 1, b = 2
      constructor
      · exact Nat.zero_lt_one -- 1 > 0
      constructor
      · exact Nat.zero_lt_two -- 2 > 0
      · -- Prove 3^2 = 1^3 + 2^3
        ring
  · -- Prove that no other prime satisfies the conditions
    intro q hq
    -- Assume q is another prime satisfying the conditions
    rcases hq with ⟨prime_q, a, b, pos_a, pos_b, eq⟩
    -- Proof that q = 3
    sorry

#check unique_prime_square_sum_of_cubes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_square_sum_of_cubes_l1031_103138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_domain_of_f_l1031_103178

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def DomainF : Set ℝ := {x : ℝ | ∃ y, f x = y}

-- State the properties of f
axiom prop1 : ∀ x ∈ DomainF, (1 / x) ∈ DomainF
axiom prop2 : ∀ x ∈ DomainF, f x + f (1 / x) = x

-- Theorem: The largest possible domain of f is {-1, 1}
theorem largest_domain_of_f : 
  DomainF = {x : ℝ | x = -1 ∨ x = 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_domain_of_f_l1031_103178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_and_periodicity_l1031_103128

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ)

theorem function_symmetry_and_periodicity 
  (ω : ℝ) (φ : ℝ) (h1 : ω > 0) (h2 : -π/2 ≤ φ ∧ φ < π/2) 
  (h3 : ∀ x, f ω φ (x - π/3) = f ω φ (π/3 - x)) 
  (h4 : ∀ x, f ω φ (x + π) = f ω φ x) 
  (α : ℝ) (h5 : π/6 < α ∧ α < 2*π/3) 
  (h6 : f ω φ (α/2) = 4*Real.sqrt 3/5) : 
  ω = 2 ∧ φ = -π/6 ∧ Real.sin α = (4*Real.sqrt 3 + 3)/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_and_periodicity_l1031_103128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_v_at_specific_points_l1031_103149

noncomputable def v (x : ℝ) : ℝ := -x^2 + 3 * Real.sin (x * Real.pi / 3)

theorem sum_of_v_at_specific_points :
  v (-2.5) + v (-1) + v 1 + v 2.5 = -14.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_v_at_specific_points_l1031_103149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_arithmetic_sequence_ratio_l1031_103131

/-- The sum of the first n terms of a geometric progression with first term a and common ratio q -/
noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- Theorem: If S_{n+1}, S_n, and S_{n+2} form an arithmetic sequence for a geometric progression 
    with common ratio q, then q = -2 -/
theorem geometric_progression_arithmetic_sequence_ratio 
  (a : ℝ) (q : ℝ) (n : ℕ) (hq : q ≠ 1) :
  (2 * geometric_sum a q n = geometric_sum a q (n + 1) + geometric_sum a q (n + 2)) →
  q = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_arithmetic_sequence_ratio_l1031_103131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l1031_103130

theorem diophantine_equation_solutions (x y : ℤ) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_eq : x^2 - 3*x*y + (p : ℤ)^2*y^2 = 12*(p : ℤ)) :
  ((p = 3 ∧ x = 6 ∧ y = 0) ∨
   (p = 3 ∧ x = -6 ∧ y = 0) ∨
   (p = 3 ∧ x = 4 ∧ y = 2) ∨
   (p = 3 ∧ x = -2 ∧ y = 2) ∨
   (p = 3 ∧ x = 2 ∧ y = -2) ∨
   (p = 3 ∧ x = -4 ∧ y = -2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l1031_103130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_when_square_perimeter_equals_circle_area_l1031_103155

theorem circle_diameter_when_square_perimeter_equals_circle_area :
  ∀ (r : ℝ), r > 0 →
  let s := 2 * r
  let square_perimeter := 4 * s
  let circle_area := π * r^2
  square_perimeter = circle_area →
  2 * r = 16 / π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_when_square_perimeter_equals_circle_area_l1031_103155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_501_sequence_n_251_has_501_sequence_smallest_n_is_251_l1031_103168

theorem smallest_n_with_501_sequence (m n : ℕ) : 
  (∃ m : ℕ, (m : ℚ) / (n : ℚ) = 0.501992) → n ≥ 251 := by
  sorry

theorem n_251_has_501_sequence : 
  ∃ m : ℕ, ((m : ℚ) / 251 : ℚ) = 0.501992 := by
  sorry

theorem smallest_n_is_251 : 
  ∀ n < 251, ¬∃ m : ℕ, (∃ k : ℕ, (m : ℚ) / (n : ℚ) = k + 0.501) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_501_sequence_n_251_has_501_sequence_smallest_n_is_251_l1031_103168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubs_cardinals_home_run_difference_l1031_103165

/-- Represents a baseball team --/
inductive Team
  | Cubs
  | Cardinals

/-- The number of home runs scored by a team in a specific inning --/
def home_runs_in_inning (team : Team) (inning : Nat) : Nat :=
  match team, inning with
  | Team.Cubs, 3 => 2
  | Team.Cubs, 5 => 1
  | Team.Cubs, 8 => 2
  | Team.Cardinals, 2 => 1
  | Team.Cardinals, 5 => 1
  | _, _ => 0

/-- The total number of home runs scored by a team --/
def total_home_runs (team : Team) : Nat :=
  (List.range 9).foldl (fun acc inning => acc + home_runs_in_inning team inning) 0

/-- The statement to be proved --/
theorem cubs_cardinals_home_run_difference :
  total_home_runs Team.Cubs - total_home_runs Team.Cardinals = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubs_cardinals_home_run_difference_l1031_103165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_function_fixed_point_l1031_103140

/-- Definition of a T-function -/
def is_T_function (f : ℝ → ℝ) : Prop :=
  ∀ s t : ℝ, 0 ≤ s → 0 ≤ t → 0 ≤ f s ∧ 0 ≤ f t ∧ f s + f t ≤ f (s + t)

/-- Main theorem -/
theorem t_function_fixed_point
  (f : ℝ → ℝ)
  (h_T : is_T_function f)
  (x₀ : ℝ)
  (h_x₀_nonneg : 0 ≤ x₀)
  (h_x₀ : f (f x₀) = x₀) :
  f x₀ = x₀ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_function_fixed_point_l1031_103140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r3_bounds_l1031_103160

-- Define a regular pentagon
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  side_length : ∀ i : Fin 5, dist (vertices i) (vertices ((i + 1) % 5)) = 1
  symmetry : ∀ i j : Fin 5, dist (vertices i) (vertices j) = dist (vertices 0) (vertices (j - i))

-- Define a point in or on the pentagon
def PointInPentagon (p : RegularPentagon) : Type :=
  {m : ℝ × ℝ // 
    ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c ≤ 1 ∧
    m = a • p.vertices 0 + b • p.vertices 1 + c • p.vertices 2 + 
        (1 - a - b - c) • p.vertices 3}

-- Define the distances from a point to vertices
noncomputable def DistancesToVertices (p : RegularPentagon) (m : PointInPentagon p) : 
  {r : Fin 5 → ℝ // ∀ i : Fin 5, r i = dist m.val (p.vertices i) ∧ 
                   ∀ i j : Fin 5, i ≤ j → r i ≤ r j} := 
  sorry

-- The main theorem
theorem r3_bounds (p : RegularPentagon) (m : PointInPentagon p) :
  let r := (DistancesToVertices p m).val
  0.8090 ≤ r 2 ∧ r 2 ≤ 1.5590 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r3_bounds_l1031_103160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_max_triangle_area_l1031_103156

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  eq : ℝ → ℝ → Prop
  h_eq : ∀ x y, eq x y ↔ y^2 = 4*a*x

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.eq x y

/-- Triangle formed by three points -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

theorem parabola_max_triangle_area 
  (p : Parabola)
  (F : ℝ × ℝ)
  (A B : PointOnParabola p)
  (h_focus : F = (1, 0))
  (h_A : A.x = 1 ∧ A.y = 2)
  (h_B : B.x = 4 ∧ B.y = -4)
  (h_FA : Real.sqrt ((A.x - F.1)^2 + (A.y - F.2)^2) = 2)
  (h_FB : Real.sqrt ((B.x - F.1)^2 + (B.y - F.2)^2) = 5)
  : ∃ (P : PointOnParabola p), 
    P.x = 1/4 ∧ P.y = -1 ∧
    (∀ (Q : PointOnParabola p), 
      0 ≤ Q.x ∧ Q.x ≤ 4 → 
      triangleArea ⟨(A.x, A.y), (B.x, B.y), (Q.x, Q.y)⟩ ≤ 
      triangleArea ⟨(A.x, A.y), (B.x, B.y), (P.x, P.y)⟩) ∧
    triangleArea ⟨(A.x, A.y), (B.x, B.y), (P.x, P.y)⟩ = 27/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_max_triangle_area_l1031_103156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l1031_103147

-- Define a polynomial with integer coefficients
def IntPolynomial := Polynomial ℤ

-- Theorem statement
theorem divisibility_property (P : IntPolynomial) 
  (h1 : 2 ∣ P.eval 5)
  (h2 : 5 ∣ P.eval 2) :
  10 ∣ P.eval 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l1031_103147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_through_center_l1031_103198

-- Define the line in polar coordinates
def line (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 1

-- Define the circle in polar coordinates
def circle_eq (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 0)

-- Theorem stating that the line intersects the circle and passes through its center
theorem line_intersects_circle_through_center :
  ∃ (ρ θ : ℝ), line ρ θ ∧ circle_eq ρ θ ∧
  (ρ * Real.cos θ, ρ * Real.sin θ) = circle_center :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_through_center_l1031_103198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_range_l1031_103182

/-- The parabola defined by x^2 = 8y -/
def Parabola (x y : ℝ) : Prop := x^2 = 8*y

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (0, 2)

/-- The directrix of the parabola -/
def Directrix (y : ℝ) : Prop := y = -2

/-- Distance between two points -/
noncomputable def Distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Circle centered at (a, b) with radius r -/
def Circle (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

theorem parabola_point_range (x₀ y₀ : ℝ) :
  Parabola x₀ y₀ →
  (∃ (x y : ℝ), Directrix y ∧ Circle x y Focus.1 Focus.2 (Distance x₀ y₀ Focus.1 Focus.2)) →
  y₀ > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_range_l1031_103182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_intersection_points_l1031_103169

-- Define the curves
def curve1 (x y : ℝ) : Prop := x = y^2
def curve2 (x y : ℝ) : Prop := y = x^2

-- Define an intersection point
def is_intersection_point (x y : ℝ) : Prop :=
  curve1 x y ∧ curve2 x y

-- Define the set of all intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | is_intersection_point p.1 p.2}

-- Theorem statement
theorem num_intersection_points :
  ∃ (S : Finset (ℝ × ℝ)), S.toSet = intersection_points ∧ Finset.card S = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_intersection_points_l1031_103169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l1031_103167

/-- The time it takes for two trains to cross each other -/
noncomputable def train_crossing_time (train_speed : ℝ) (train_length : ℝ) : ℝ :=
  let relative_speed := 2 * train_speed * (5 / 18)  -- Convert km/hr to m/s
  let total_distance := 2 * train_length
  total_distance / relative_speed

/-- Theorem: Two trains with speed 36 km/hr and length 120 meters cross in 12 seconds -/
theorem train_crossing_theorem :
  train_crossing_time 36 120 = 12 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_crossing_time 36 120

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l1031_103167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_neg_abs_l1031_103148

noncomputable section

-- Define the interval (-∞, 0]
def interval_neg_inf_to_zero : Set ℝ := Set.Iic 0

-- Define the functions
def f1 (x : ℝ) : ℝ := -1 / x
def f2 (x : ℝ) : ℝ := -(x - 1)
def f3 (x : ℝ) : ℝ := x^2 - 2
def f4 (x : ℝ) : ℝ := -abs x

-- Define monotonic increase on an interval
def monotonic_increase_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

theorem monotonic_increase_neg_abs : 
  monotonic_increase_on f4 interval_neg_inf_to_zero ∧ 
  ¬monotonic_increase_on f1 interval_neg_inf_to_zero ∧
  ¬monotonic_increase_on f2 interval_neg_inf_to_zero ∧
  ¬monotonic_increase_on f3 interval_neg_inf_to_zero :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_neg_abs_l1031_103148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marked_cells_for_impossibility_l1031_103172

/-- Represents an L-shaped triomino on a grid --/
structure LTriomino where
  cells : Finset (Nat × Nat)
  size_eq_3 : cells.card = 3
  l_shape : ∃ (x y : Nat), 
    cells = {(x, y), (x + 1, y), (x, y + 1)} ∨
    cells = {(x, y), (x + 1, y), (x + 1, y + 1)} ∨
    cells = {(x, y), (x, y + 1), (x + 1, y + 1)} ∨
    cells = {(x, y), (x - 1, y), (x, y + 1)}

/-- Represents a 5x5 grid --/
def Grid5x5 : Finset (Nat × Nat) :=
  Finset.filter (fun (x, y) => x < 5 ∧ y < 5) (Finset.product (Finset.range 5) (Finset.range 5))

/-- A set of L-shaped triominoes is valid if they are non-overlapping and within the 5x5 grid --/
def ValidTriominoSet (triominos : Finset LTriomino) : Prop :=
  (triominos.biUnion (fun t => t.cells) ⊆ Grid5x5) ∧
  ∀ t1 t2, t1 ∈ triominos → t2 ∈ triominos → t1 ≠ t2 → t1.cells ∩ t2.cells = ∅

/-- The main theorem --/
theorem min_marked_cells_for_impossibility : 
  ∀ (marked : Finset (Nat × Nat)),
    (marked ⊆ Grid5x5) →
    (marked.card < 9 → 
      ∃ (triominos : Finset LTriomino), ValidTriominoSet triominos ∧ marked ⊆ triominos.biUnion (fun t => t.cells)) ∧
    (marked.card = 9 → 
      ∃ (arrangement : Finset (Nat × Nat)), 
        arrangement ⊆ Grid5x5 ∧ 
        arrangement.card = 9 ∧
        (∀ (triominos : Finset LTriomino), ValidTriominoSet triominos → 
          ¬(arrangement ⊆ triominos.biUnion (fun t => t.cells)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marked_cells_for_impossibility_l1031_103172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_l1031_103146

/-- Given a rectangle and a triangle, if the ratio of their areas is 2:5,
    the rectangle's width is 4 cm, and the triangle's area is 60 cm²,
    then the rectangle's length is 6 cm. -/
theorem rectangle_length (rectangle_width rectangle_length triangle_area : ℝ) :
  rectangle_width = 4 →
  triangle_area = 60 →
  (rectangle_width * rectangle_length) / triangle_area = 2 / 5 →
  rectangle_length = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_l1031_103146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_implies_a_range_l1031_103161

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + log x

-- State the theorem
theorem f_monotone_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → 
    f a x₁ + 2 * x₁ < f a x₂ + 2 * x₂) → 
  0 ≤ a ∧ a ≤ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_implies_a_range_l1031_103161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clyde_children_average_score_l1031_103110

/-- The average of four numbers is the sum of the numbers divided by 4. -/
noncomputable def average (a b c d : ℝ) : ℝ := (a + b + c + d) / 4

/-- Clyde's children's math test scores -/
def june_score : ℚ := 97
def patty_score : ℚ := 85
def josh_score : ℚ := 100
def henry_score : ℚ := 94

/-- Theorem: The average of Clyde's children's math test scores is 94 -/
theorem clyde_children_average_score :
  (average (june_score : ℝ) (patty_score : ℝ) (josh_score : ℝ) (henry_score : ℝ)) = 94 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clyde_children_average_score_l1031_103110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_in_interval_l1031_103179

theorem trig_inequality_in_interval (θ : ℝ) (h : π < θ ∧ θ < (5 * π) / 4) :
  Real.cos θ < Real.sin θ ∧ Real.sin θ < Real.tan θ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_in_interval_l1031_103179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_after_change_l1031_103114

/-- Given a rectangle with original length 2x cm and breadth x cm, 
    if the length is decreased by 5 cm and the breadth is increased by 4 cm, 
    resulting in an area increase of 75 sq. cm, 
    then the original length of the rectangle is 190/3 cm. -/
theorem rectangle_length_after_change (x : ℝ) : 
  (2*x - 5) * (x + 4) = 2*x^2 + 75 → 2*x = 190/3 := by
  sorry

/-- The diagonal of the rectangle can be expressed in terms of x using the Pythagorean theorem. -/
noncomputable def diagonal (x : ℝ) : ℝ := x * Real.sqrt 5

#check rectangle_length_after_change
#check diagonal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_after_change_l1031_103114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_cube_in_binomial_expansion_l1031_103144

/-- The coefficient of x^3 in the expansion of ((x-1/x^2)^6) -/
def coeff_x_cube : ℤ := -6

/-- The binomial expression ((x-1/x^2)^6) -/
noncomputable def binomial_expr (x : ℝ) : ℝ := (x - 1 / x^2)^6

theorem coeff_x_cube_in_binomial_expansion :
  coeff_x_cube = (Finset.range 7).sum (λ k ↦ 
    ((-1)^k : ℤ) * (Nat.choose 6 k) * 
    (if 6 - 3*k = 3 then 1 else 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_cube_in_binomial_expansion_l1031_103144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1031_103154

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 1 → b = 2 → C = (2 * Real.pi) / 3 →
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →
  c = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1031_103154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_m_value_l1031_103193

noncomputable def a : ℝ × ℝ := (3, -4)
noncomputable def b : ℝ × ℝ := (1, 2)

noncomputable def θ : ℝ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem sin_theta_value : Real.sin θ = (2 * Real.sqrt 5) / 5 := by sorry

theorem m_value (m : ℝ) (h : (m * a.1 - b.1) * (a.1 + b.1) + (m * a.2 - b.2) * (a.2 + b.2) = 0) : m = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_m_value_l1031_103193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l1031_103123

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define point P on the parabola
noncomputable def P : ℝ × ℝ := (4, Real.sqrt (8*4))

-- Theorem statement
theorem distance_to_focus :
  parabola P.1 P.2 →
  Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l1031_103123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_real_roots_l1031_103133

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * abs (Real.log x)

noncomputable def g (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then 0
  else if x > 1 then abs (x^2 - 4) - 2
  else 0  -- This case is not specified in the original problem, but needed for completeness

-- Define the equation
def equation (x : ℝ) : Prop :=
  abs (g x - f x) = 1

-- Theorem statement
theorem four_real_roots :
  ∃ (a b c d : ℝ), a < b ∧ b < c ∧ c < d ∧
    equation a ∧ equation b ∧ equation c ∧ equation d ∧
    ∀ (x : ℝ), equation x → (x = a ∨ x = b ∨ x = c ∨ x = d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_real_roots_l1031_103133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_minimum_length_l1031_103187

-- Define auxiliary functions
def get_foci (ellipse : Set (ℝ × ℝ)) : (ℝ × ℝ) × (ℝ × ℝ) := sorry
def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem ellipse_major_axis_minimum_length 
  (ellipse : Set (ℝ × ℝ)) 
  (max_triangle_area : ℝ) 
  (h1 : max_triangle_area = 2) :
  ∃ (major_axis_length : ℝ), 
    (∀ (point : ℝ × ℝ), point ∈ ellipse → 
      let foci := get_foci ellipse
      let triangle_area := area_triangle point foci.1 foci.2
      triangle_area ≤ max_triangle_area) ∧
    major_axis_length ≥ 4 ∧
    ∀ (other_length : ℝ), 
      (∀ (point : ℝ × ℝ), point ∈ ellipse → 
        let foci := get_foci ellipse
        let triangle_area := area_triangle point foci.1 foci.2
        triangle_area ≤ max_triangle_area) →
      other_length ≥ major_axis_length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_minimum_length_l1031_103187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_point_of_f_l1031_103124

-- Define the function f(x) = x - 2ln(x)
noncomputable def f (x : ℝ) : ℝ := x - 2 * Real.log x

-- State the theorem
theorem extreme_value_point_of_f :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≤ f x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_point_of_f_l1031_103124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_run_time_to_friends_house_l1031_103199

-- Define the constant pace
noncomputable def constant_pace (time : ℝ) (distance : ℝ) : ℝ := time / distance

-- Theorem statement
theorem run_time_to_friends_house 
  (store_time : ℝ) 
  (store_distance : ℝ) 
  (friend_distance : ℝ) 
  (h1 : store_time = 18) 
  (h2 : store_distance = 2) 
  (h3 : friend_distance = 1) :
  constant_pace store_time store_distance * friend_distance = 9 :=
by
  -- Unfold the definition of constant_pace
  unfold constant_pace
  -- Substitute known values
  rw [h1, h2, h3]
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_run_time_to_friends_house_l1031_103199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1031_103107

noncomputable def f (x : ℝ) := Real.sqrt x / Real.log (2 - x)

theorem domain_of_f :
  {x : ℝ | x ∈ Set.Icc 0 1 ∪ Set.Ioo 1 2} =
  {x : ℝ | x ≥ 0 ∧ 2 - x > 0 ∧ Real.log (2 - x) ≠ 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1031_103107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_roots_polynomial_l1031_103166

theorem shifted_roots_polynomial (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - 5*x^2 + 7 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℝ, x^3 - 14*x^2 + 57*x - 65 = 0 ↔ x = a + 3 ∨ x = b + 3 ∨ x = c + 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_roots_polynomial_l1031_103166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_coordinates_l1031_103115

-- Define the points
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (-3, 2)
def C : ℝ × ℝ := (3, 2)
def D : ℝ × ℝ := (4, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem point_P_coordinates :
  ∃ P : ℝ × ℝ,
    distance P A + distance P D = 10 ∧
    distance P B + distance P C = 10 ∧
    P.2 = (18 + 6 * Real.sqrt 2) / 7 ∧
    18 + 6 + 2 + 7 = 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_coordinates_l1031_103115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valuable_files_calculation_l1031_103117

/-- Calculates the number of valuable files after two rounds of downloads and deletions. -/
theorem valuable_files_calculation (initial_download : ℕ) (first_deletion_rate : ℚ)
  (second_download : ℕ) (second_deletion_rate : ℚ)
  (h1 : initial_download = 1200)
  (h2 : first_deletion_rate = 4/5)
  (h3 : second_download = 600)
  (h4 : second_deletion_rate = 4/5) :
  (initial_download - Int.floor (initial_download * first_deletion_rate : ℚ)) +
  (second_download - Int.floor (second_download * second_deletion_rate : ℚ)) = 360 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valuable_files_calculation_l1031_103117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l1031_103191

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (m a₁ a₂ : ℝ) : ℝ :=
  |a₁ - a₂| / Real.sqrt (1 + m^2)

/-- Theorem: The distance between y = -3x + 5 and y = -3x - 4 is 9√10/10 -/
theorem distance_specific_parallel_lines :
  distance_between_parallel_lines (-3) 5 (-4) = 9 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l1031_103191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1031_103164

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  2 * t.a * t.c * Real.sin t.A + t.a^2 + t.c^2 - t.b^2 = 0 ∧
  t.A = Real.pi / 6 ∧
  t.a = 2 ∧
  t.A + t.B + t.C = Real.pi

-- Define the theorem
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  (1/2 * t.a * t.c * Real.sin t.B = Real.sqrt 3) ∧
  (∀ x : Triangle, triangle_conditions x → 
    (4 * (Real.sin x.C)^2 + 3 * (Real.sin x.A)^2 + 2) / (Real.sin x.B)^2 ≥ 5) ∧
  ((4 * (Real.sin t.C)^2 + 3 * (Real.sin t.A)^2 + 2) / (Real.sin t.B)^2 = 5 ↔ t.B = 2 * Real.pi / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1031_103164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_primes_with_integer_root_sum_l1031_103108

theorem infinitely_many_primes_with_integer_root_sum :
  ∃ (f : ℕ → ℕ), Monotone f ∧
  (∀ k, Nat.Prime (f k)) ∧
  (∀ k, ∃ n : ℕ, ∃ m : ℕ, (Real.sqrt (f k + n) + Real.sqrt n : ℝ) = m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_primes_with_integer_root_sum_l1031_103108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1031_103121

-- Define the triangle and its properties
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the vectors
def m (t : Triangle) : ℝ × ℝ := ((t.b + t.c)^2, -1)
def n (t : Triangle) : ℝ × ℝ := (1, t.a^2 + t.b * t.c)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0)  -- Positive side lengths
  (h2 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0)  -- Positive angles
  (h3 : t.A + t.B + t.C = π)  -- Angle sum property
  (h4 : dot_product (m t) (n t) = 0)  -- Given condition
  : 
  (Real.cos t.A = -1/2) ∧  -- Part 1
  (∀ (t' : Triangle), t'.a = 3 → 
    6 < t'.a + t'.b + t'.c ∧ t'.a + t'.b + t'.c ≤ 3 + 2 * Real.sqrt 3) -- Part 2
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1031_103121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_advantageous_discount_l1031_103158

def effective_discount_1 : ℝ := 1 - (1 - 0.20) * (1 - 0.20)
def effective_discount_2 : ℝ := 1 - (1 - 0.15) * (1 - 0.15) * (1 - 0.15)
def effective_discount_3 : ℝ := 1 - (1 - 0.30) * (1 - 0.10)

def is_more_advantageous (n : ℕ) : Prop :=
  (n : ℝ) / 100 > effective_discount_1 ∧
  (n : ℝ) / 100 > effective_discount_2 ∧
  (n : ℝ) / 100 > effective_discount_3

theorem smallest_advantageous_discount :
  ∃ (n : ℕ), n = 39 ∧ is_more_advantageous n ∧ ∀ m, m < n → ¬is_more_advantageous m := by
  sorry

#check smallest_advantageous_discount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_advantageous_discount_l1031_103158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_count_l1031_103150

def points_on_sides : Nat × Nat × Nat := (10, 11, 12)

def total_points (p : Nat × Nat × Nat) : Nat := p.1 + p.2.1 + p.2.2

theorem triangle_count (p : Nat × Nat × Nat) (h : p = points_on_sides) : 
  (Nat.choose (total_points p) 3) - 
  (Nat.choose p.1 3 + Nat.choose p.2.1 3 + Nat.choose p.2.2 3) = 4951 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_count_l1031_103150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_union_complement_A_B_l1031_103194

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 2*x - 8 ≤ 0}
noncomputable def B : Set ℝ := {x | Real.rpow 3 x ≥ 1/3}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = Set.Icc (-1) 2 := by sorry

-- Theorem for the union of complement of A and B
theorem union_complement_A_B : (Set.univ \ A) ∪ B = Set.Ioi (-4) ∪ Set.Ici (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_union_complement_A_B_l1031_103194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_walk_distance_is_sqrt_13_l1031_103192

/-- The distance from the starting point after walking 5 km along the perimeter of a regular hexagon with side length 2 km -/
noncomputable def hexagon_walk_distance : ℝ :=
  let x : ℝ := 7/2  -- Final x-coordinate
  let y : ℝ := Real.sqrt 3 / 2  -- Final y-coordinate
  Real.sqrt (x^2 + y^2)

/-- Theorem stating that the distance is √13 km -/
theorem hexagon_walk_distance_is_sqrt_13 :
  hexagon_walk_distance = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_walk_distance_is_sqrt_13_l1031_103192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l1031_103118

-- Define the line
def line (x y : ℝ) : Prop := 3 * x + 4 * y + 13 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

-- Define the distance function between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem min_distance_line_circle :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line x₁ y₁ ∧ circle_eq x₂ y₂ ∧
    ∀ (x₃ y₃ x₄ y₄ : ℝ),
      line x₃ y₃ → circle_eq x₄ y₄ →
      distance x₁ y₁ x₂ y₂ ≤ distance x₃ y₃ x₄ y₄ ∧
      distance x₁ y₁ x₂ y₂ = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l1031_103118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assistant_impact_l1031_103101

/-- Represents a worker's productivity metrics -/
structure WorkerMetrics where
  bears_per_week : ℝ
  hours_per_week : ℝ

/-- Calculates the percent change between two values -/
noncomputable def percent_change (old_value new_value : ℝ) : ℝ :=
  (new_value - old_value) / old_value * 100

theorem assistant_impact (w : WorkerMetrics) :
  let with_assistant : WorkerMetrics :=
    { bears_per_week := w.bears_per_week * 1.8,
      hours_per_week := w.hours_per_week * (1 - 0.1) }
  percent_change w.bears_per_week with_assistant.bears_per_week = 80 ∧
  percent_change (w.bears_per_week / w.hours_per_week) (with_assistant.bears_per_week / with_assistant.hours_per_week) = 100 →
  percent_change w.hours_per_week with_assistant.hours_per_week = -10 := by
  sorry

#check assistant_impact

end NUMINAMATH_CALUDE_ERRORFEEDBACK_assistant_impact_l1031_103101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_running_time_l1031_103189

/-- The number of laps Michael runs -/
def num_laps : ℕ := 8

/-- The length of each lap in meters -/
noncomputable def lap_length : ℝ := 400

/-- The speed of the first half of each lap in meters per second -/
noncomputable def speed_first_half : ℝ := 6

/-- The speed of the second half of each lap in meters per second -/
noncomputable def speed_second_half : ℝ := 3

/-- The length of the first half of each lap in meters -/
noncomputable def first_half_length : ℝ := lap_length / 2

/-- The length of the second half of each lap in meters -/
noncomputable def second_half_length : ℝ := lap_length / 2

/-- Theorem: The total time Michael takes to complete all laps is 800 seconds -/
theorem michael_running_time : 
  (num_laps : ℝ) * (first_half_length / speed_first_half + second_half_length / speed_second_half) = 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_running_time_l1031_103189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l1031_103113

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_analysis 
  (A ω φ : ℝ) 
  (h1 : A > 0) 
  (h2 : ω > 0) 
  (h3 : |φ| < π) 
  (h4 : f A ω φ (π/12) = 3) 
  (h5 : f A ω φ (7*π/12) = -3) :
  (∃ k : ℤ, ∀ x : ℝ, 
    f A ω φ x = 3 * Real.sin (2*x + π/3) ∧
    (k*π + π/12 ≤ x ∧ x ≤ k*π + 7*π/12 → 
      ∀ y : ℝ, x < y → f A ω φ x > f A ω φ y)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l1031_103113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_five_digit_palindromes_l1031_103120

/-- A function that checks if a five-digit number is palindromic -/
def isPalindromic (n : ℕ) : Bool :=
  n ≥ 10000 ∧ n ≤ 99999 ∧ 
  (n / 10000 = n % 10) ∧ 
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The set of all five-digit palindromic numbers -/
def fiveDigitPalindromes : Set ℕ :=
  {n : ℕ | isPalindromic n = true}

theorem count_five_digit_palindromes : 
  Finset.card (Finset.filter (fun n => isPalindromic n) (Finset.range 100000)) = 900 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_five_digit_palindromes_l1031_103120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_implies_m_eq_6_l1031_103190

noncomputable def line1 (m : ℝ) (x : ℝ) : ℝ := (m - 2) / m * x + 2
noncomputable def line2 (m : ℝ) (x : ℝ) : ℝ := -x + 2 * m

noncomputable def triangleArea (m : ℝ) : ℝ := (1 / 2) * m * (2 * m - 2)

theorem triangle_area_implies_m_eq_6 (m : ℝ) (h1 : m > 2) 
  (h2 : triangleArea m = 30) : m = 6 := by
  sorry

#check triangle_area_implies_m_eq_6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_implies_m_eq_6_l1031_103190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_D_perimeter_proof_l1031_103109

-- Define square C
noncomputable def square_C_perimeter : ℝ := 40

-- Define the relationship between squares C and D
noncomputable def square_D_area_ratio : ℝ := 1/3

-- Define the perimeter of square D
noncomputable def square_D_perimeter : ℝ := (40 * Real.sqrt 3) / 3

-- Theorem statement
theorem square_D_perimeter_proof :
  let side_C := square_C_perimeter / 4
  let area_C := side_C ^ 2
  let area_D := area_C * square_D_area_ratio
  let side_D := Real.sqrt area_D
  4 * side_D = square_D_perimeter := by
  sorry

#check square_D_perimeter_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_D_perimeter_proof_l1031_103109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_time_calculation_l1031_103174

/-- Calculates the time required to earn the same interest at a different rate -/
noncomputable def calculate_equivalent_time (initial_rate : ℝ) (initial_time : ℝ) (final_rate : ℝ) : ℝ :=
  (initial_rate * initial_time) / final_rate

theorem equivalent_time_calculation :
  let initial_rate : ℝ := 0.04
  let initial_time : ℝ := 29 / 12  -- 2 years and 5 months in years
  let final_rate : ℝ := 0.0375
  let result := calculate_equivalent_time initial_rate initial_time final_rate
  abs (result - 2.5778) < 0.0001 := by
  -- Proof goes here
  sorry

#eval (0.04 * (29 / 12)) / 0.0375

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_time_calculation_l1031_103174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_legal_triangulation_odd_triangles_l1031_103139

-- Define a regular pentagon
structure RegularPentagon where
  vertices : Finset (Fin 5)

-- Define legal points
inductive LegalPoint (p : RegularPentagon)
  | vertex : Fin 5 → LegalPoint p
  | intersection : LegalPoint p → LegalPoint p → LegalPoint p

-- Define legal segments
def LegalSegment (p : RegularPentagon) := 
  {seg : LegalPoint p × LegalPoint p // seg.1 ≠ seg.2}

-- Define a legal triangulation
structure LegalTriangulation (p : RegularPentagon) where
  triangles : Finset (LegalPoint p × LegalPoint p × LegalPoint p)
  is_valid : ∀ t ∈ triangles, 
    (t.1 ≠ t.2.1) ∧ (t.2.1 ≠ t.2.2) ∧ (t.2.2 ≠ t.1)

-- Theorem: The number of triangles in any legal triangulation is odd
theorem legal_triangulation_odd_triangles (p : RegularPentagon) 
  (t : LegalTriangulation p) : Odd t.triangles.card := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_legal_triangulation_odd_triangles_l1031_103139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_implies_a_bound_l1031_103171

/-- The function f(x) defined on the positive real numbers. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - (1/2) * x^2 + 6*x

/-- The derivative of f(x) with respect to x. -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a / x - x + 6

/-- Theorem stating that if f(x) is monotonically decreasing on its domain (0, +∞),
    then a must be less than or equal to -9. -/
theorem monotone_decreasing_implies_a_bound :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → f_derivative a x ≤ 0) → a ≤ -9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_implies_a_bound_l1031_103171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_volume_formula_l1031_103152

/-- The volume of a right truncated quadrangular pyramid -/
noncomputable def truncated_pyramid_volume (a b : ℝ) : ℝ :=
  (a^3 - b^3) * Real.sqrt 2 / 6

/-- Theorem: The volume of a right truncated quadrangular pyramid with larger base side a,
    smaller base side b, and lateral face acute angle 60° is equal to (a³ - b³)√2 / 6 -/
theorem truncated_pyramid_volume_formula (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b < a) :
  truncated_pyramid_volume a b = (a^3 - b^3) * Real.sqrt 2 / 6 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_volume_formula_l1031_103152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1031_103176

theorem trigonometric_identities 
  (α θ : Real) 
  (h1 : Real.cos α = -4/5) 
  (h2 : α ∈ Set.Icc Real.pi (3*Real.pi/2)) 
  (h3 : Real.tan θ = 3) : 
  Real.sin α = -3/5 ∧ (Real.sin θ + Real.cos θ) / (2 * Real.sin θ + Real.cos θ) = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1031_103176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_pyramid_properties_l1031_103145

/-- A regular triangular frustum pyramid with an inscribed sphere -/
structure FrustumPyramid where
  k : ℝ  -- ratio of pyramid volume to sphere volume
  h : ℝ  -- height of the pyramid
  a : ℝ  -- side length of the lower base
  b : ℝ  -- side length of the upper base
  R : ℝ  -- radius of the inscribed sphere

/-- The theorem about the angle and permissible k values for a FrustumPyramid -/
theorem frustum_pyramid_properties (P : FrustumPyramid) :
  (P.k > (9 * Real.sqrt 3) / (2 * Real.pi)) ∧
  (Real.arctan (6 / Real.sqrt (2 * Real.sqrt 3 * Real.pi * P.k - 27)) =
   Real.arctan ((P.b - P.a) / (4 * Real.sqrt 3 * P.R))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_pyramid_properties_l1031_103145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_S_l1031_103132

-- Define the complex number j
noncomputable def j : ℂ := Complex.I * Real.sqrt 2

-- Define the function S
noncomputable def S (n : ℤ) : ℂ := j^n + j^(-n)

-- Theorem statement
theorem distinct_values_of_S :
  ∃ (A : Finset ℂ), (∀ n : ℤ, S n ∈ A) ∧ Finset.card A = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_S_l1031_103132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1031_103126

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if tan C = 3√7, CA • CB = 5/2, and a + b = 9,
    then the area of triangle ABC is 15√7/4 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  Real.tan C = 3 * Real.sqrt 7 →
  a * b * Real.cos C = 5/2 →
  a + b = 9 →
  (1/2) * a * b * Real.sin C = 15 * Real.sqrt 7 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1031_103126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jennifer_apples_l1031_103186

/-- Given that Jennifer starts with 7 apples and finds 74 more, 
    prove that she ends up with 81 apples in total. -/
theorem jennifer_apples : 7 + 74 = 81 := by
  -- Define the initial number of apples
  let initial_apples : ℕ := 7
  -- Define the number of apples found
  let found_apples : ℕ := 74
  -- Calculate the total number of apples
  let total_apples : ℕ := initial_apples + found_apples
  -- Prove that the total is equal to 81
  calc
    total_apples = initial_apples + found_apples := rfl
    _ = 7 + 74 := rfl
    _ = 81 := rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jennifer_apples_l1031_103186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l1031_103183

-- Define the function f(x)
noncomputable def f (x : ℝ) := Real.log (3*x/2) - 2/x

-- State the theorem
theorem zero_point_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0 :=
by
  -- Assume the following properties of f
  have h1 : Continuous f := sorry
  have h2 : StrictMono f := sorry
  have h3 : f 1 < 0 := sorry
  have h4 : f 2 > 0 := sorry
  
  sorry -- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l1031_103183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1031_103125

def mySequence (n : ℕ) : ℚ :=
  if n % 2 = 0 then 7 else -4

theorem sequence_formula (n : ℕ) : mySequence n = 3/2 + (-1)^n * 11/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1031_103125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mix_solutions_concentration_l1031_103100

/-- Represents a solution with water and alcohol components -/
structure Solution where
  water : ℚ
  alcohol : ℚ

/-- Calculates the concentration of alcohol in a solution -/
def alcoholConcentration (s : Solution) : ℚ :=
  s.alcohol / (s.water + s.alcohol)

/-- Given two solutions with specific water to alcohol ratios, 
    proves that mixing equal amounts results in 40% alcohol concentration -/
theorem mix_solutions_concentration 
  (solutionA solutionB mixedSolution : Solution)
  (hA : solutionA.water / solutionA.alcohol = 4 / 1)
  (hB : solutionB.water / solutionA.alcohol = 2 / 3)
  (hMix : mixedSolution.water = solutionA.water + solutionB.water ∧ 
          mixedSolution.alcohol = solutionA.alcohol + solutionB.alcohol)
  (hEqual : solutionA.water + solutionA.alcohol = solutionB.water + solutionB.alcohol) :
  alcoholConcentration mixedSolution = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mix_solutions_concentration_l1031_103100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l1031_103111

def A (m : ℝ) : Set ℝ := {x | x^2 - m*x + m^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {2, -4}

theorem find_m : ∃ m : ℝ, (A m ∩ B).Nonempty ∧ (A m ∩ C = ∅) ∧ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l1031_103111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_increasing_even_decreasing_l1031_103153

noncomputable def sequence_x (a : ℝ) : ℕ → ℝ
  | 0 => a  -- Add this case to handle Nat.zero
  | 1 => a
  | n + 1 => a^(sequence_x a n)

theorem odd_increasing_even_decreasing (a : ℝ) (ha : 0 < a ∧ a < 1) :
  (∀ k : ℕ, k > 0 → sequence_x a (2 * k - 1) < sequence_x a (2 * k + 1)) ∧
  (∀ k : ℕ, k > 0 → sequence_x a (2 * k) > sequence_x a (2 * k + 2)) :=
by
  sorry

#check odd_increasing_even_decreasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_increasing_even_decreasing_l1031_103153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sunny_days_probability_l1031_103159

theorem two_sunny_days_probability :
  let n : ℕ := 5  -- total number of days
  let k : ℕ := 2  -- number of sunny days we want
  let p : ℚ := 1/4  -- probability of a sunny day
  Finset.sum (Finset.range (n+1)) (λ i ↦ if i = k then (n.choose i) * p^i * (1-p)^(n-i) else 0) = 135/512 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sunny_days_probability_l1031_103159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1031_103103

theorem problem_solution (α β : ℝ) 
  (h1 : 0 < α ∧ α < Real.pi/2 ∧ Real.pi/2 < β ∧ β < Real.pi)
  (h2 : Real.tan (α/2) = 1/3)
  (h3 : Real.cos (β - α) = -Real.sqrt 2 / 10) :
  Real.sin α = 3/5 ∧ β = 3*Real.pi/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1031_103103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_y_approx_five_l1031_103143

/-- Definition of the function G --/
noncomputable def G (a b c d : ℝ) : ℝ := a^b + c * d

/-- Theorem stating the existence of y and its approximate value --/
theorem exists_y_approx_five : 
  ∃ y : ℝ, G 3 y 6 8 = 300 ∧ |y - 5| < 0.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_y_approx_five_l1031_103143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_buying_equation_l1031_103112

/-- Represents the number of people contributing to buy chickens. -/
def x : ℕ := sorry

/-- The amount of money needed to buy the chickens. -/
def total_cost : ℕ := sorry

/-- The equation representing the chicken-buying scenario is correct. -/
theorem chicken_buying_equation :
  (9 * x - 11 = total_cost) ∧ (6 * x + 16 = total_cost) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_buying_equation_l1031_103112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l1031_103177

def b : ℕ → ℚ
  | 0 => 2  -- Adding a case for 0 to cover all natural numbers
  | 1 => 2
  | 2 => 5/11
  | (n+3) => (b (n+1) * b (n+2)) / (3 * b (n+1) - b (n+2))

theorem b_formula (n : ℕ) : n ≥ 1 → b n = 5 / (5 * n + 1) := by sorry

#eval b 2023  -- This line is optional, for testing purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l1031_103177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sine_condition_l1031_103157

-- Define the function
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (1/2 * ω * x)

-- State the theorem
theorem increasing_sine_condition (ω : ℝ) :
  (ω > 0) →
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < π → f ω x₁ < f ω x₂) ↔
  (0 < ω ∧ ω ≤ 1) := by
  sorry

#check increasing_sine_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sine_condition_l1031_103157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l1031_103119

/-- An ellipse with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_e_eq : e = Real.sqrt (1 - b^2 / a^2)

/-- A line with slope k and y-intercept c -/
structure Line where
  k : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def dot_product (p q : Point) : ℝ := p.x * q.x + p.y * q.y

/-- Function to get y-coordinate of a point on the line -/
def Line.y (l : Line) (x : ℝ) : ℝ := l.k * x + l.c

theorem ellipse_problem (E : Ellipse) (l : Line) (A B P : Point) :
  E.e = Real.sqrt 2 / 2 →
  E.b = 1 →
  A.x = -E.a ∧ A.y = 0 →
  B.x^2 / E.a^2 + B.y^2 / E.b^2 = 1 →
  P.x = E.a →
  l.y P.x = l.k * (P.x + E.a) →
  (∃ (M : Point), M.x = 0 ∧ M.y = 1) →
  (dot_product B P = 2) ∧
  (Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 4/3 →
    l.k = 1 ∨ l.k = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l1031_103119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_difference_l1031_103137

theorem cube_root_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  ((5 - x₁^2 / 4)^(1/3 : ℝ) = -3) ∧
  ((5 - x₂^2 / 4)^(1/3 : ℝ) = -3) ∧
  (x₁ ≠ x₂) ∧
  (|x₁ - x₂| = 16 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_difference_l1031_103137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_formula_l1031_103127

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

def S (a : ℕ → ℚ) (n : ℕ) : ℚ := sum_of_arithmetic_sequence a n

theorem arithmetic_sequence_sum_formula 
  (a : ℕ → ℚ) (S7 S15 : ℚ) (h_arith : arithmetic_sequence a) 
  (h_S7 : S a 7 = 7) (h_S15 : S a 15 = 75) :
  ∃ Tn : ℕ → ℚ, ∀ n : ℕ, 
    Tn n = (1/4 : ℚ) * n^2 - (9/4 : ℚ) * n ∧
    Tn n = sum_of_arithmetic_sequence (λ k ↦ S a k / k) n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_formula_l1031_103127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_odd_fixed_points_exists_even_function_odd_fixed_points_l1031_103142

-- Define a real-valued function
noncomputable def f : ℝ → ℝ := sorry

-- Define a fixed point of f
def is_fixed_point (c : ℝ) : Prop := f c = c

-- Define an odd function
def is_odd_function : Prop := ∀ x, f (-x) = -f x

-- Define an even function
def is_even_function : Prop := ∀ x, f (-x) = f x

-- Assume f has a finite number of fixed points
axiom finite_fixed_points : ∃ (n : ℕ), ∃ (S : Finset ℝ), (∀ c, is_fixed_point c ↔ c ∈ S) ∧ S.card = n

-- Statement 1: If f is odd, the number of its fixed points is odd
theorem odd_function_odd_fixed_points :
  is_odd_function → ∃ (k : ℕ), (∃ (S : Finset ℝ), (∀ c, is_fixed_point c ↔ c ∈ S) ∧ S.card = 2*k + 1) :=
by sorry

-- Statement 2: There exists an even function with an odd number of fixed points
theorem exists_even_function_odd_fixed_points :
  ∃ (f : ℝ → ℝ), is_even_function ∧ ∃ (k : ℕ), (∃ (S : Finset ℝ), (∀ c, is_fixed_point c ↔ c ∈ S) ∧ S.card = 2*k + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_odd_fixed_points_exists_even_function_odd_fixed_points_l1031_103142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_1_3_6_9_l1031_103134

noncomputable def harmonic_mean (xs : List ℝ) : ℝ :=
  xs.length / (xs.map (λ x => 1 / x)).sum

theorem harmonic_mean_1_3_6_9 :
  harmonic_mean [1, 3, 6, 9] = 72 / 29 := by
  -- Expand the definition of harmonic_mean
  unfold harmonic_mean
  -- Simplify the expression
  simp [List.map, List.sum]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_1_3_6_9_l1031_103134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_value_l1031_103162

noncomputable section

def h : ℝ → ℝ := sorry
def k : ℝ → ℝ := sorry

axiom h_range : ∀ x, -3 ≤ h x ∧ h x ≤ 5
axiom k_range : ∀ x, -1 ≤ k x ∧ k x ≤ 4

def product_range : Set ℝ := { y | ∃ x, y = h x * k x }

theorem max_product_value :
  ∃ d, d ∈ product_range ∧ ∀ y ∈ product_range, y ≤ d ∧ d = 20 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_value_l1031_103162
