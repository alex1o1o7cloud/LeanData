import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l637_63741

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to A
  b : ℝ  -- Side opposite to B
  c : ℝ  -- Side opposite to C

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  -- Sum of angles is π
  t.A + t.B + t.C = Real.pi ∧
  -- Cosine of sum of A and B
  2 * Real.cos (t.A + t.B) = 1 ∧
  -- a and b are roots of the equation x^2 - 2√3x + 2 = 0
  t.a^2 - 2 * Real.sqrt 3 * t.a + 2 = 0 ∧
  t.b^2 - 2 * Real.sqrt 3 * t.b + 2 = 0

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : TriangleProperties t) :
  t.C = 2 * Real.pi / 3 ∧ t.c = Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l637_63741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_l637_63738

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x

-- State the theorem
theorem derivative_f : 
  deriv f = λ x => 2 * x * Real.log x + x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_l637_63738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_pi_4_plus_2cos_pi_4_cos_sq_alpha_eq_zero_l637_63774

theorem sin_2alpha_plus_pi_4_plus_2cos_pi_4_cos_sq_alpha_eq_zero 
  (α : ℝ) 
  (h1 : Real.sin (2 * α) = 3 / 5) 
  (h2 : π / 4 < α ∧ α < π / 2) : 
  Real.sin (2 * α + π / 4) + 2 * Real.cos (π / 4) * (Real.cos α) ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_pi_4_plus_2cos_pi_4_cos_sq_alpha_eq_zero_l637_63774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_san_antonio_bound_bus_passes_four_austin_bound_buses_l637_63777

/-- Represents the time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Represents a bus schedule -/
structure BusSchedule where
  departureTime : Time
  arrivalTime : Time
  direction : Bool  -- true for Austin to San Antonio, false for San Antonio to Austin

/-- Calculates the time difference between two Time values in hours -/
noncomputable def timeDifference (t1 t2 : Time) : ℚ := sorry

/-- Checks if two buses pass each other on the highway -/
def busesPass (b1 b2 : BusSchedule) : Bool := sorry

/-- The main theorem to be proved -/
theorem san_antonio_bound_bus_passes_four_austin_bound_buses 
  (austin_to_san_antonio_schedule : List BusSchedule)
  (san_antonio_to_austin_schedule : List BusSchedule)
  (h1 : ∀ b, b ∈ austin_to_san_antonio_schedule → b.direction = true)
  (h2 : ∀ b, b ∈ san_antonio_to_austin_schedule → b.direction = false)
  (h3 : ∀ b, b ∈ austin_to_san_antonio_schedule → b.departureTime.minutes = 0)
  (h4 : ∀ b, b ∈ san_antonio_to_austin_schedule → b.departureTime.minutes = 15)
  (h5 : ∀ b, b ∈ austin_to_san_antonio_schedule ++ san_antonio_to_austin_schedule → 
        timeDifference b.departureTime b.arrivalTime = 4)
  (h6 : ∀ b1 b2, b1 ∈ austin_to_san_antonio_schedule → b2 ∈ austin_to_san_antonio_schedule → b1 ≠ b2 → 
        timeDifference b1.departureTime b2.departureTime ≥ 1)
  (h7 : ∀ b1 b2, b1 ∈ san_antonio_to_austin_schedule → b2 ∈ san_antonio_to_austin_schedule → b1 ≠ b2 → 
        timeDifference b1.departureTime b2.departureTime ≥ 1)
  : ∃ (san_antonio_bus : BusSchedule),
    san_antonio_bus ∈ san_antonio_to_austin_schedule ∧
    (austin_to_san_antonio_schedule.filter (busesPass san_antonio_bus)).length = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_san_antonio_bound_bus_passes_four_austin_bound_buses_l637_63777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_video_votes_theorem_l637_63787

theorem video_votes_theorem (up_votes down_votes : ℕ) :
  up_votes = 3690 →
  (up_votes : ℚ) / (down_votes : ℚ) = 45 / 17 →
  down_votes = 1394 := by
  sorry

#check video_votes_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_video_votes_theorem_l637_63787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l637_63723

theorem max_value_expression (a b c d : ℝ) 
  (ha : a ∈ Set.Icc (-5.5) 5.5)
  (hb : b ∈ Set.Icc (-5.5) 5.5)
  (hc : c ∈ Set.Icc (-5.5) 5.5)
  (hd : d ∈ Set.Icc (-5.5) 5.5) :
  (∃ x y z w : ℝ, x ∈ Set.Icc (-5.5) 5.5 ∧ 
              y ∈ Set.Icc (-5.5) 5.5 ∧ 
              z ∈ Set.Icc (-5.5) 5.5 ∧ 
              w ∈ Set.Icc (-5.5) 5.5 ∧
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x = 132) ∧
  (∀ a b c d : ℝ, a ∈ Set.Icc (-5.5) 5.5 → 
              b ∈ Set.Icc (-5.5) 5.5 → 
              c ∈ Set.Icc (-5.5) 5.5 → 
              d ∈ Set.Icc (-5.5) 5.5 → 
              a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 132) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l637_63723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_and_g_continuity_l637_63702

noncomputable def f (x : ℝ) : ℝ := 
  if -1 ≤ x ∧ x ≤ 1 then 4*x*(1 - abs x) else 0

noncomputable def g (x : ℝ) : ℝ :=
  if x ≠ 0 then f x / x else 4

theorem f_derivative_and_g_continuity :
  (∃ (d : ℝ), HasDerivAt f d 0 ∧ d = 4) ∧
  ContinuousAt g 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_and_g_continuity_l637_63702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_price_is_eight_l637_63713

/-- The highest possible price per book Alice can afford given the conditions -/
def highest_price_per_book : ℕ :=
  let total_budget : ℕ := 180
  let entrance_fee : ℕ := 3
  let num_books : ℕ := 20
  let tax_rate : ℚ := 7 / 100
  let budget_after_fee : ℕ := total_budget - entrance_fee
  let max_total_cost : ℚ := (budget_after_fee : ℚ) / (1 + tax_rate)
  (max_total_cost / num_books).floor.toNat

/-- Theorem stating that the highest possible price per book Alice can afford is $8 -/
theorem highest_price_is_eight : highest_price_per_book = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_price_is_eight_l637_63713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_existence_l637_63798

noncomputable def f : ℝ → ℝ → ℕ := sorry

theorem function_existence :
  ∃ (f : ℝ → ℝ → ℕ), ∀ (x y z : ℝ), f x y = f y z → x = y ∧ y = z :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_existence_l637_63798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_l637_63799

/-- Given a sum of money that becomes 7/6 of itself in 5 years under simple interest, 
    the annual interest rate is 10/3%. -/
theorem simple_interest_rate (principal : ℝ) (h : principal > 0) : 
  (((7 / 6 : ℝ) * principal - principal) * 100) / (principal * 5) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_l637_63799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_guaranteed_trap_l637_63773

/-- Represents the state of the grasshopper game -/
structure GameState where
  n : ℕ  -- Number of cells between traps
  position : ℤ  -- Current position of the grasshopper

/-- Checks if the grasshopper is trapped -/
def is_trapped (state : GameState) : Prop :=
  state.position = 0 ∨ state.position = state.n + 1

/-- Represents a strategy for naming jump distances -/
def Strategy := GameState → ℕ

/-- Checks if a strategy guarantees trapping the grasshopper -/
def is_winning_strategy (s : Strategy) (n : ℕ) : Prop :=
  ∀ initial_pos : ℤ, 0 < initial_pos ∧ initial_pos < n + 1 →
    ∃ (moves : ℕ → Bool), 
      let final_state := (List.foldl (λ state b ↦ 
        let jump := s state
        let new_pos := if b then state.position + jump else state.position - jump
        { n := state.n, position := new_pos }
      ) (GameState.mk n initial_pos) (List.map moves (List.range n)))
      is_trapped final_state

/-- The main theorem stating the condition for guaranteed trapping -/
theorem guaranteed_trap (n : ℕ) : 
  (∃ (s : Strategy), is_winning_strategy s n) ↔ (∃ k : ℕ, n = 2^k - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_guaranteed_trap_l637_63773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l637_63752

noncomputable def f (a ω b x : ℝ) : ℝ := a * Real.sin (2 * ω * x + Real.pi / 6) + a / 2 + b

theorem function_properties :
  ∀ (a ω b : ℝ), a > 0 → ω > 0 →
  (∀ x : ℝ, f a ω b (x + Real.pi) = f a ω b x) →
  (∀ x : ℝ, f a ω b x ≤ 7/4) →
  (∀ x : ℝ, f a ω b x ≥ 3/4) →
  (∃ x : ℝ, f a ω b x = 7/4) →
  (∃ x : ℝ, f a ω b x = 3/4) →
  (ω = 1 ∧ a = 1/2 ∧ b = 1) ∧
  (∀ k : ℤ, ∀ x : ℝ, k * Real.pi - Real.pi/3 ≤ x ∧ x ≤ k * Real.pi + Real.pi/6 →
    ∀ y : ℝ, k * Real.pi - Real.pi/3 ≤ y ∧ y ≤ x → f (1/2) 1 1 y ≤ f (1/2) 1 1 x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l637_63752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_cubic_equation_l637_63720

theorem determinant_cubic_equation (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ p q : ℝ, (fun x : ℝ ↦ 
    Matrix.det (![![x, d, -b],
                 ![-d, x, c],
                 ![b, -c, x]] : Matrix (Fin 3) (Fin 3) ℝ)) = (fun x : ℝ ↦ x^3 + p*x + q) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_cubic_equation_l637_63720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_three_in_sixteen_factorial_l637_63737

theorem greatest_power_of_three_in_sixteen_factorial :
  (∃ n : ℕ, 3^n ∣ Nat.factorial 16 ∧ ∀ k > n, ¬(3^k ∣ Nat.factorial 16)) → 
  (∃ n : ℕ, 3^n ∣ Nat.factorial 16 ∧ ∀ k > n, ¬(3^k ∣ Nat.factorial 16) ∧ n = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_three_in_sixteen_factorial_l637_63737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_coefficient_sum_squares_l637_63786

/-- A line segment in 3D space connecting two points -/
structure LineSegment3D where
  start : Fin 3 → ℝ
  endpoint : Fin 3 → ℝ

/-- Coefficients for the parametric equation of a line segment -/
structure ParametricCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- The sum of squares of the parametric coefficients -/
def sumOfSquares (coeff : ParametricCoefficients) : ℝ :=
  coeff.a^2 + coeff.b^2 + coeff.c^2 + coeff.d^2 + coeff.e^2 + coeff.f^2

/-- Theorem: The sum of squares of parametric coefficients for the given line segment is 132 -/
theorem line_segment_coefficient_sum_squares 
  (seg : LineSegment3D)
  (coeff : ParametricCoefficients)
  (h1 : seg.start 0 = 1 ∧ seg.start 1 = 3 ∧ seg.start 2 = 6)
  (h2 : seg.endpoint 0 = 6 ∧ seg.endpoint 1 = -2 ∧ seg.endpoint 2 = 12)
  (h3 : coeff.b = seg.start 0 ∧ coeff.d = seg.start 1 ∧ coeff.f = seg.start 2)
  (h4 : coeff.a + coeff.b = seg.endpoint 0 ∧ 
        coeff.c + coeff.d = seg.endpoint 1 ∧ 
        coeff.e + coeff.f = seg.endpoint 2) :
  sumOfSquares coeff = 132 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_coefficient_sum_squares_l637_63786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sqrt_two_others_rational_l637_63719

-- Define the numbers given in the problem
noncomputable def a : ℝ := (8 : ℝ) ^ (1/3)
noncomputable def b : ℝ := Real.sqrt 2
def c : ℚ := 2/7
noncomputable def d : ℝ := 3.14

-- Theorem statement
theorem irrational_sqrt_two_others_rational :
  (∃ (q : ℚ), (q : ℝ) = a) ∧
  (∃ (q : ℚ), (q : ℝ) = d) ∧
  ¬(∃ (q : ℚ), (q : ℝ) = b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sqrt_two_others_rational_l637_63719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_c_coordinates_l637_63749

/-- Given points A and B in a plane, and conditions on BC, prove the possible coordinates of C -/
theorem point_c_coordinates (A B C : ℝ × ℝ) : 
  A = (-1, 0) → 
  B = (-3, -4) → 
  (C.2 = B.2) →  -- BC is parallel to OA (horizontal)
  |C.1 - B.1| = 4 * |A.1 - 0| →  -- BC = 4AO
  (C = (-7, -4) ∨ C = (1, -4)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_c_coordinates_l637_63749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_prime_factors_247520_l637_63739

theorem sum_of_distinct_prime_factors_247520 : 
  Finset.sum (Finset.filter Nat.Prime (Finset.range (247520 + 1))) id = 113 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_prime_factors_247520_l637_63739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_multiplication_l637_63754

theorem matrix_multiplication (M : Matrix (Fin 2) (Fin 2) ℚ) 
  (h1 : M.mulVec (![1, 2] : Fin 2 → ℚ) = ![2, 5])
  (h2 : M.mulVec (![3, -1] : Fin 2 → ℚ) = ![7, 0]) :
  M.mulVec (![2, 3] : Fin 2 → ℚ) = ![29/7, 55/7] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_multiplication_l637_63754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_equality_l637_63780

theorem cosine_value_equality (α : ℝ) :
  Real.sin (π / 6 - α) = 1 / 3 →
  2 * (Real.cos (π / 6 + α / 2))^2 - 1 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_equality_l637_63780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rem_four_sevenths_three_fifths_l637_63722

/-- Definition of remainder for real numbers -/
noncomputable def rem (x y : ℝ) : ℝ := x - y * ⌊x / y⌋

/-- Theorem stating that rem(4/7, 3/5) = 4/7 -/
theorem rem_four_sevenths_three_fifths : rem (4/7) (3/5) = 4/7 := by
  -- Unfold the definition of rem
  unfold rem
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rem_four_sevenths_three_fifths_l637_63722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_circle_intersection_l637_63740

/-- RightTriangle represents a right triangle with vertices P, Q, R, where Q is the right angle -/
structure RightTriangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  is_right_angle : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0

/-- Circle represents a circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- Intersection point S of circle and triangle side PR -/
noncomputable def S (t : RightTriangle) (c : Circle) : ℝ × ℝ := sorry

/-- Length of a line segment between two points -/
noncomputable def length (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Area of a triangle given its vertices -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

theorem right_triangle_circle_intersection 
  (t : RightTriangle) (c : Circle) :
  (c.O = t.Q ∧ c.r = length t.Q t.R / 2) →
  triangleArea t.P t.Q t.R = 98 →
  length t.P t.R = 14 →
  length t.Q (S t c) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_circle_intersection_l637_63740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_x_axis_l637_63769

/-- A circle with center at (-5, 4) that is tangent to the x-axis has the equation (x + 5)² + (y - 4)² = 16 -/
theorem circle_tangent_to_x_axis (x y : ℝ) :
  let center : ℝ × ℝ := (-5, 4)
  let is_tangent_to_x_axis := center.2 = abs center.2  -- The y-coordinate equals the radius
  is_tangent_to_x_axis →
  ((x + 5)^2 + (y - 4)^2 = 16) ↔ 
  ((x - center.1)^2 + (y - center.2)^2 = center.2^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_x_axis_l637_63769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_analysis_l637_63793

-- Define the structure of a tournament
structure Tournament where
  scores : List Nat
  win_points : Nat
  draw_points : Nat

-- Define the properties of our specific tournament
def our_tournament : Tournament :=
  { scores := [16, 14, 10, 10, 8, 6, 5, 3]
  , win_points := 2
  , draw_points := 1 }

-- Helper functions (definitions only, implementation with sorry)
def number_of_teams (t : Tournament) : Nat :=
  sorry

def points_lost_by_top_4 (t : Tournament) : Nat :=
  sorry

-- Theorem statement
theorem tournament_analysis (t : Tournament) 
  (h1 : t.scores = our_tournament.scores)
  (h2 : t.win_points = our_tournament.win_points)
  (h3 : t.draw_points = our_tournament.draw_points) :
  (number_of_teams t = 9) ∧ (points_lost_by_top_4 t = 14) := by
  sorry

#check tournament_analysis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_analysis_l637_63793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_universal_polynomial_with_n_integral_roots_l637_63714

theorem no_universal_polynomial_with_n_integral_roots :
  ¬ ∃ (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0),
    ∀ (n : ℕ) (hn : n > 3),
      ∃ (P : Polynomial ℝ) (roots : Finset ℤ),
        P.degree = n ∧
        P.coeff 2 = a ∧
        P.coeff 1 = b ∧
        P.coeff 0 = c ∧
        roots.card = n ∧
        ∀ (x : ℤ), x ∈ roots ↔ P.eval (x : ℝ) = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_universal_polynomial_with_n_integral_roots_l637_63714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_cafe_l637_63789

/-- Represents the time to run from the house to the store in minutes -/
noncomputable def store_time : ℝ := 24

/-- Represents the distance from the house to the store in miles -/
noncomputable def store_distance : ℝ := 3

/-- Represents the constant speed of the runner in miles per minute -/
noncomputable def runner_speed : ℝ := store_distance / store_time

/-- Represents the distance from the house to the café in miles -/
noncomputable def cafe_distance : ℝ := store_distance / 2

/-- Theorem stating that the time to run from the house to the café is 12 minutes -/
theorem time_to_cafe : cafe_distance / runner_speed = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_cafe_l637_63789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_314_decimal_l637_63715

theorem smallest_n_for_314_decimal (m n : ℕ) : 
  m > 0 → n > 0 → m < n → Nat.Coprime m n → 
  (∃ k : ℕ, (1000 * (10^k * m % n)) / n = 314) → 
  n ≥ 315 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_314_decimal_l637_63715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_five_l637_63717

theorem sum_of_solutions_is_five :
  ∃ (x₁ x₂ : ℝ),
    (2 : ℝ) ^ (x₁^2 - 2*x₁ - 8) = (8 : ℝ) ^ (x₁ - 2) ∧
    (2 : ℝ) ^ (x₂^2 - 2*x₂ - 8) = (8 : ℝ) ^ (x₂ - 2) ∧
    (∀ x : ℝ, (2 : ℝ) ^ (x^2 - 2*x - 8) = (8 : ℝ) ^ (x - 2) → x = x₁ ∨ x = x₂) ∧
    x₁ + x₂ = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_five_l637_63717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l637_63795

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def infinite_geometric_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: For an infinite geometric series with common ratio -1/3 and sum 9, the first term is 12 -/
theorem first_term_of_geometric_series :
  ∃ (a : ℝ), infinite_geometric_sum a (-1/3) = 9 ∧ a = 12 := by
  use 12
  constructor
  · -- Prove that infinite_geometric_sum 12 (-1/3) = 9
    simp [infinite_geometric_sum]
    norm_num
  · -- Prove that a = 12
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l637_63795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maddies_mom_coffee_cups_l637_63711

/-- Calculates the number of cups of coffee Maddie's mom makes per day -/
noncomputable def coffee_cups_per_day (
  ounces_per_cup : ℝ)
  (bag_cost : ℝ)
  (ounces_per_bag : ℝ)
  (milk_gallons_per_week : ℝ)
  (milk_cost_per_gallon : ℝ)
  (weekly_coffee_expense : ℝ) : ℝ :=
  let weekly_milk_cost := milk_gallons_per_week * milk_cost_per_gallon
  let weekly_beans_expense := weekly_coffee_expense - weekly_milk_cost
  let bags_per_week := weekly_beans_expense / bag_cost
  let ounces_per_week := bags_per_week * ounces_per_bag
  let cups_per_week := ounces_per_week / ounces_per_cup
  cups_per_week / 7

theorem maddies_mom_coffee_cups (
  ounces_per_cup : ℝ)
  (bag_cost : ℝ)
  (ounces_per_bag : ℝ)
  (milk_gallons_per_week : ℝ)
  (milk_cost_per_gallon : ℝ)
  (weekly_coffee_expense : ℝ)
  (h1 : ounces_per_cup = 1.5)
  (h2 : bag_cost = 8)
  (h3 : ounces_per_bag = 10.5)
  (h4 : milk_gallons_per_week = 0.5)
  (h5 : milk_cost_per_gallon = 4)
  (h6 : weekly_coffee_expense = 18) :
  coffee_cups_per_day ounces_per_cup bag_cost ounces_per_bag milk_gallons_per_week milk_cost_per_gallon weekly_coffee_expense = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maddies_mom_coffee_cups_l637_63711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_six_l637_63770

-- Define the circle
def Circle (r : ℝ) := { x : ℝ × ℝ | (x.1 - 0)^2 + (x.2 - 0)^2 = r^2 }

-- Define the circumference of a circle
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

-- Define the area of a circle
noncomputable def area (r : ℝ) : ℝ := Real.pi * r^2

-- Theorem statement
theorem circle_radius_is_six :
  ∃ (r : ℝ), r > 0 ∧ 3 * circumference r = area r ∧ r = 6 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_six_l637_63770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_enclosing_triangle_l637_63763

-- Define a convex polygon
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  is_convex : Prop
  is_polygon : Prop

-- Define a parallelogram
structure Parallelogram extends ConvexPolygon where
  is_parallelogram : Prop

-- Define a triangle
structure Triangle extends ConvexPolygon where
  is_triangle : Prop

-- Define the concept of extending a side
def extend_side (p : ConvexPolygon) (side : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

-- Define the concept of enclosing
def encloses (t : Triangle) (p : ConvexPolygon) : Prop := sorry

-- Theorem statement
theorem convex_polygon_enclosing_triangle 
  (M : ConvexPolygon) 
  (h : ¬ (∃ P : Parallelogram, P.vertices = M.vertices)) : 
  ∃ (T : Triangle), 
    ∃ (s1 s2 s3 : Set (ℝ × ℝ)), 
      s1 ⊆ M.vertices ∧ s2 ⊆ M.vertices ∧ s3 ⊆ M.vertices ∧
      T.vertices = extend_side M s1 ∪ extend_side M s2 ∪ extend_side M s3 ∧
      encloses T M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_enclosing_triangle_l637_63763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l637_63766

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : Real.cos t.B * Real.cos t.C - Real.sin t.B * Real.sin t.C = - (1/2 : ℝ))
  (h2 : t.a = 2)
  (h3 : (1/2 : ℝ) * t.b * t.c * Real.sin t.A = Real.sqrt 3) :
  t.A = π/3 ∧ t.b = 2 ∧ t.c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l637_63766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_neg_one_l637_63760

noncomputable section

variable (f : ℝ → ℝ)

axiom f_def : ∀ x, f x = (1/3) * x^3 - (deriv f (-1)) * x^2 + x + 5

theorem f_derivative_at_neg_one :
  deriv f (-1) = -2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_neg_one_l637_63760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squared_distances_l637_63747

theorem max_sum_squared_distances (a b c d : ℝ × ℝ × ℝ) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) : 
  ‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2 ≤ 16 ∧
  ∃ (a' b' c' d' : ℝ × ℝ × ℝ), ‖a'‖ = 1 ∧ ‖b'‖ = 1 ∧ ‖c'‖ = 1 ∧ ‖d'‖ = 1 ∧
    ‖a' - b'‖^2 + ‖a' - c'‖^2 + ‖a' - d'‖^2 + ‖b' - c'‖^2 + ‖b' - d'‖^2 + ‖c' - d'‖^2 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squared_distances_l637_63747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_yellow_marbles_l637_63781

/-- Represents the total number of marbles Marcy has -/
def total_marbles : ℕ := 20

/-- The number of blue marbles Marcy has -/
def blue_marbles : ℕ := total_marbles / 4

/-- The number of red marbles Marcy has -/
def red_marbles : ℕ := total_marbles / 5

/-- The number of green marbles Marcy has -/
def green_marbles : ℕ := 10

/-- The number of yellow marbles Marcy has -/
def yellow_marbles : ℕ := total_marbles - (blue_marbles + red_marbles + green_marbles)

/-- Theorem stating that the smallest possible number of yellow marbles is 1 -/
theorem smallest_yellow_marbles :
  ∃ (n : ℕ), n > 0 ∧ 
  total_marbles = n ∧
  blue_marbles = n / 4 ∧
  red_marbles = n / 5 ∧
  green_marbles = 10 ∧
  yellow_marbles = 1 ∧
  ∀ (m : ℕ), m > 0 → 
    (m / 4 + m / 5 + 10 < m → m - (m / 4 + m / 5 + 10) ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_yellow_marbles_l637_63781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l637_63765

theorem circular_table_seating (n : ℕ) (k : ℕ) (h : n = 8 ∧ k = 7) :
  (n.choose k) * Nat.factorial (k - 1) = 5760 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l637_63765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_relation_l637_63732

theorem triangle_area_relation (a b c : ℝ) (α : ℝ) (S : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 0 < α ∧ α < π) 
  (h5 : S = (1/2) * b * c * Real.sin α) 
  (h6 : a^2 = b^2 + c^2 - 2*b*c*(Real.cos α)) : 
  4 * S = (a^2 - (b-c)^2) * Real.tan (α/2)⁻¹ := by
  sorry

#check triangle_area_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_relation_l637_63732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_cut_perimeter_l637_63750

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle (α : Type*) where
  A : α
  B : α
  C : α

/-- Represents a quadrilateral with vertices A, C, E, and D -/
structure Quadrilateral (α : Type*) where
  A : α
  C : α
  E : α
  D : α

/-- A triangle is equilateral if all its sides are equal -/
def is_equilateral {α : Type*} [MetricSpace α] (t : Triangle α) : Prop :=
  dist t.A t.B = dist t.B t.C ∧ dist t.B t.C = dist t.C t.A

/-- The side length of a triangle -/
def side_length {α : Type*} [MetricSpace α] (t : Triangle α) : ℝ :=
  dist t.A t.B

/-- The perimeter of a quadrilateral -/
def perimeter {α : Type*} [MetricSpace α] (q : Quadrilateral α) : ℝ :=
  dist q.A q.C + dist q.C q.E + dist q.E q.D + dist q.D q.A

/-- Main theorem -/
theorem corner_cut_perimeter 
  {α : Type*} [MetricSpace α] 
  (ABC : Triangle α) 
  (DBE : Triangle α) 
  (ACED : Quadrilateral α) :
  is_equilateral ABC →
  is_equilateral DBE →
  side_length ABC = 3 →
  dist DBE.B DBE.C = 1 →
  dist DBE.C DBE.A = 1 →
  ABC.A = ACED.A →
  ABC.B = DBE.A →
  ABC.C = ACED.C →
  DBE.B = ACED.D →
  DBE.C = ACED.E →
  perimeter ACED = 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_cut_perimeter_l637_63750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_at_specific_angle_midpoint_trajectory_equation_l637_63729

noncomputable section

-- Define the polar coordinate system
def polar_to_rect (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define curves C1 and C2
def C1 (θ : ℝ) : ℝ := 4 * Real.cos θ
def C2 (θ : ℝ) : ℝ := Real.sqrt 3 * Real.sin θ

-- Define line l
def l (θ₀ : ℝ) (ρ : ℝ) : ℝ := θ₀

-- Define the distance between two points in polar coordinates
def polar_distance (ρ₁ θ₁ ρ₂ θ₂ : ℝ) : ℝ :=
  Real.sqrt ((ρ₁ * Real.cos θ₁ - ρ₂ * Real.cos θ₂)^2 + (ρ₁ * Real.sin θ₁ - ρ₂ * Real.sin θ₂)^2)

-- Theorem for part (I)
theorem distance_AB_at_specific_angle :
  let θ₀ : ℝ := 3 * Real.pi / 4
  let ρ_A : ℝ := C1 θ₀
  let ρ_B : ℝ := C2 θ₀
  polar_distance ρ_A θ₀ ρ_B θ₀ = 2 * Real.sqrt 2 + Real.sqrt 6 / 2 := by sorry

-- Theorem for part (II)
theorem midpoint_trajectory_equation (x y : ℝ) :
  (∃ θ : ℝ, 
    x = (C1 θ + C2 θ) / 2 * Real.cos θ ∧ 
    y = (C1 θ + C2 θ) / 2 * Real.sin θ) ↔ 
  x^2 + y^2 - 2*x - Real.sqrt 3 / 2 * y = 0 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_at_specific_angle_midpoint_trajectory_equation_l637_63729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_of_f_l637_63791

-- Define the function g
def g : ℝ → ℝ := sorry

-- Define the function f in terms of g
def f (x : ℝ) : ℝ := g (2 * x - 1) + x^2

-- State the theorem
theorem tangent_line_of_f (h : HasDerivAt g 2 1) :
  ∃ (a b c : ℝ), a = 6 ∧ b = -1 ∧ c = -2 ∧
  HasDerivAt f 6 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_of_f_l637_63791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l637_63710

def sequenceProperty (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, x (n + 1) = (-x n + Real.sqrt (3 - 3 * (x n)^2)) / 2

theorem sequence_properties (x : ℕ → ℝ) (h : sequenceProperty x) (h1 : |x 1| < 1) :
  (∀ n : ℕ, x (n + 2) = x n) ∧
  (∀ n : ℕ, x n > 0 ↔ 0 < x 1 ∧ x 1 < Real.sqrt 3 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l637_63710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_relation_quadrilaterals_l637_63779

/-- Represents a quadrilateral in 2D space -/
def Quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

/-- Checks if a point is inside a quadrilateral -/
def PointInsideQuadrilateral (P A B C D : ℝ × ℝ) : Prop := sorry

/-- Calculates the area of a quadrilateral -/
noncomputable def area (A B C D : ℝ × ℝ) : ℝ := sorry

/-- Given two quadrilaterals ABCD and A'B'C'D', if there exists a point O inside ABCD
    such that OA = A'B', OB = B'C', OC = C'D', and OD = D'A', then the area of ABCD
    is twice the area of A'B'C'D'. -/
theorem area_relation_quadrilaterals 
  (A B C D A' B' C' D' O : ℝ × ℝ) 
  (h_quad : Quadrilateral A B C D) 
  (h_quad' : Quadrilateral A' B' C' D') 
  (h_inside : PointInsideQuadrilateral O A B C D) 
  (h_OA : O.fst - A.fst = A'.fst - B'.fst ∧ O.snd - A.snd = A'.snd - B'.snd) 
  (h_OB : O.fst - B.fst = B'.fst - C'.fst ∧ O.snd - B.snd = B'.snd - C'.snd) 
  (h_OC : O.fst - C.fst = C'.fst - D'.fst ∧ O.snd - C.snd = C'.snd - D'.snd) 
  (h_OD : O.fst - D.fst = D'.fst - A'.fst ∧ O.snd - D.snd = D'.snd - A'.snd) : 
  area A B C D = 2 * area A' B' C' D' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_relation_quadrilaterals_l637_63779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_pair_exists_l637_63708

/-- Represents a chess player in the tournament -/
structure Player where
  id : Nat

/-- Represents the number of wins for a player with each color -/
structure Wins where
  white : Nat
  black : Nat

/-- Defines the notion of one player not being weaker than another -/
def not_weaker (wins : Player → Wins) (a b : Player) : Prop :=
  (wins a).white ≥ (wins b).white ∧ (wins a).black ≥ (wins b).black

theorem chess_tournament_pair_exists :
  ∀ (n : Nat) (wins : Player → Wins),
    n = 20 →
    (∀ p : Player, p.id < n) →
    (∀ p : Player, (wins p).white < n ∧ (wins p).black < n) →
    ∃ a b : Player, a ≠ b ∧ not_weaker wins a b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_pair_exists_l637_63708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_tan_relation_log_relation_l637_63772

-- Problem 1
theorem sin_cos_tan_relation (α : ℝ) (h : Real.tan α = 2) :
  Real.sin α * (Real.sin α + Real.cos α) = 6/5 := by sorry

-- Problem 2
theorem log_relation (a b : ℝ) (h1 : (5 : ℝ)^a = 10) (h2 : (4 : ℝ)^b = 10) :
  2/a + 1/b = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_tan_relation_log_relation_l637_63772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_theorem_l637_63794

theorem sum_of_cubes_theorem (a b : ℚ) : 
  a * b = 19.999999999999996 → a^3 + b^3 = 8001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_theorem_l637_63794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_of_cos_roots_l637_63753

theorem sin_product_of_cos_roots (α β : Real) : 
  α ∈ Set.Ioo 0 π → β ∈ Set.Ioo 0 π → 
  (∃ x y : Real, x = Real.cos α ∧ y = Real.cos β ∧ 5 * x^2 - 3 * x - 1 = 0 ∧ 5 * y^2 - 3 * y - 1 = 0) →
  Real.sin α * Real.sin β = Real.sqrt 7 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_of_cos_roots_l637_63753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_greater_than_octagon_l637_63718

/-- The area between circumscribed and inscribed circles of a regular polygon -/
noncomputable def area_between_circles (n : ℕ) (s : ℝ) : ℝ :=
  let θ := 2 * Real.pi / n
  let R := s / (2 * Real.sin (θ / 2))
  let A := s / (2 * Real.tan (θ / 2))
  Real.pi * (R^2 - A^2)

/-- Theorem stating that the area between circles for a pentagon with side length 3 
    is greater than that for an octagon with side length 2 -/
theorem pentagon_area_greater_than_octagon :
  area_between_circles 5 3 > area_between_circles 8 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_greater_than_octagon_l637_63718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_calculation_l637_63726

theorem original_price_calculation (discounted_price : ℝ) (discount_percentage : ℝ) 
  (h1 : discounted_price = 7500)
  (h2 : discount_percentage = 25) : 
  discounted_price / (1 - discount_percentage / 100) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_calculation_l637_63726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencils_left_formula_l637_63784

/-- The number of children -/
def n : ℕ := 15

/-- The number of pencils each child starts with -/
def initial_pencils : ℕ := 2

/-- The first term of the arithmetic progression (number of pencils given away by the first child) -/
def a : ℕ → ℕ := fun _ => 0  -- Placeholder function

/-- The common difference of the arithmetic progression -/
def d : ℕ → ℕ := fun _ => 0  -- Placeholder function

/-- The total number of pencils left among all children -/
def pencils_left (a d : ℕ → ℕ) : ℤ :=
  n * initial_pencils - (n * a 0 + (n * (n - 1) * d 0) / 2)

theorem pencils_left_formula (a d : ℕ → ℕ) :
  pencils_left a d = 30 - (15 * a 0 + 105 * d 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencils_left_formula_l637_63784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l637_63701

theorem range_of_a : 
  ∃ a : ℝ, 0 ≤ a ∧ a ≤ 1/2 ∧
  ¬(∀ x : ℝ, (¬(2 * x^2 - 3 * x + 1 ≤ 0)) ↔ 
              (¬(x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l637_63701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutated_frogs_percentage_l637_63716

theorem mutated_frogs_percentage (extra_legs frogs_2_heads bright_red normal : ℕ) :
  extra_legs = 5 →
  frogs_2_heads = 2 →
  bright_red = 2 →
  normal = 18 →
  let total_mutated := extra_legs + frogs_2_heads + bright_red
  let total_frogs := total_mutated + normal
  let percentage := (total_mutated : ℚ) / (total_frogs : ℚ) * 100
  ⌊percentage⌋ = 33 ∧ percentage < 34 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutated_frogs_percentage_l637_63716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l637_63725

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

noncomputable def Ellipse.foci (e : Ellipse) : (Point × Point) :=
  let c := Real.sqrt (e.a^2 - e.b^2)
  ({ x := -c, y := 0 }, { x := c, y := 0 })

def Ellipse.center : Point :=
  { x := 0, y := 0 }

def dot_product (p q : Point) : ℝ :=
  p.x * q.x + p.y * q.y

noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem stating that the eccentricity of the ellipse with given conditions is √10/4 -/
theorem ellipse_eccentricity (e : Ellipse) (A M : Point) :
  let (F1, F2) := e.foci
  let O := Ellipse.center
  A.x^2 / e.a^2 + A.y^2 / e.b^2 = 1 →
  dot_product (Point.mk (A.x - F1.x) (A.y - F1.y)) (Point.mk (A.x - F2.x) (A.y - F2.y)) = 0 →
  M.x = 0 →
  distance F1 F2 = 6 * distance O M →
  Real.sqrt (e.a^2 - e.b^2) / e.a = Real.sqrt 10 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l637_63725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_placement_l637_63785

def number_of_ways_to_place (n m : ℕ) : ℕ := m^n

theorem letter_placement (n m : ℕ) (hn : n = 4) (hm : m = 3) : 
  (number_of_ways_to_place n m) = m^n := by
  rfl

#check letter_placement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_placement_l637_63785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_mean_variance_update_l637_63745

/-- Given a sample of size 8 with mean 5 and variance 2, adding a new data point 5
    results in a new sample of size 9 with mean 5 and variance less than 2. -/
theorem sample_mean_variance_update (initial_size : ℕ) (initial_mean initial_variance : ℝ) 
    (new_point : ℝ) (new_size : ℕ) :
  initial_size = 8 →
  initial_mean = 5 →
  initial_variance = 2 →
  new_point = 5 →
  new_size = initial_size + 1 →
  (let new_mean := (initial_size * initial_mean + new_point) / new_size
   let new_variance := (initial_size * initial_variance + (new_point - new_mean)^2) / new_size
   new_mean = 5 ∧ new_variance < 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_mean_variance_update_l637_63745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_has_max_and_min_l637_63709

noncomputable def a (n : ℕ) : ℝ := (4/9)^(n-1) - (2/3)^(n-1)

theorem sequence_has_max_and_min :
  (∃ m : ℕ, ∀ n : ℕ, a n ≤ a m) ∧
  (∃ m : ℕ, ∀ n : ℕ, a m ≤ a n) := by
  sorry

#check sequence_has_max_and_min

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_has_max_and_min_l637_63709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_from_angle_ratio_l637_63771

/-- A triangle with angles in the ratio 1:2:1 is an isosceles right triangle -/
theorem isosceles_right_triangle_from_angle_ratio :
  ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  (a : ℝ) / (a + b + c) = 1 / 4 →
  (b : ℝ) / (a + b + c) = 1 / 2 →
  (c : ℝ) / (a + b + c) = 1 / 4 →
  (a = 45 ∧ b = 90 ∧ c = 45) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_from_angle_ratio_l637_63771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_from_box_properties_l637_63706

/-- The radius of a sphere containing an inscribed rectangular box -/
noncomputable def sphere_radius (surface_area edge_sum : ℝ) : ℝ :=
  3 * Real.sqrt 33

/-- Theorem stating the radius of the sphere given the properties of the inscribed box -/
theorem sphere_radius_from_box_properties :
  ∀ (surface_area edge_sum : ℝ),
    surface_area = 576 →
    edge_sum = 168 →
    sphere_radius surface_area edge_sum = 3 * Real.sqrt 33 := by
  intros surface_area edge_sum h_surface h_edge
  unfold sphere_radius
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_from_box_properties_l637_63706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_x_coordinate_l637_63733

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.exp (-x)

theorem tangent_point_x_coordinate 
  (a : ℝ) 
  (h1 : ∀ x, f_derivative a x = -f_derivative a (-x)) 
  (h2 : ∃ x, f_derivative a x = 3/2) :
  ∃ x, f_derivative a x = 3/2 ∧ x = Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_x_coordinate_l637_63733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_eq_zero_l637_63731

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x : ℤ | x^2 ≤ 0}

theorem M_intersect_N_eq_zero : M ∩ N = {0} := by
  -- The proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_eq_zero_l637_63731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_range_l637_63705

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x + a

theorem f_minimum_range (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ f a 0) → 0 ≤ a ∧ a ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_range_l637_63705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pave_ways_10_l637_63761

/-- The number of ways to pave a 1 × n block with tiles of sizes 1 × 1, 1 × 2, and 1 × 4 -/
def pave_ways : ℕ → ℕ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 6
  | n + 5 => pave_ways (n + 4) + pave_ways (n + 3) + pave_ways (n + 1)

/-- Theorem: The number of ways to pave a 1 × 10 block is 169 -/
theorem pave_ways_10 : pave_ways 10 = 169 := by
  -- Compute the result
  rfl

-- Additional lemmas to help with the proof
lemma pave_ways_5 : pave_ways 5 = 10 := by rfl
lemma pave_ways_6 : pave_ways 6 = 18 := by rfl
lemma pave_ways_7 : pave_ways 7 = 31 := by rfl
lemma pave_ways_8 : pave_ways 8 = 55 := by rfl
lemma pave_ways_9 : pave_ways 9 = 96 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pave_ways_10_l637_63761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_and_max_function_l637_63783

/-- Given real numbers a, b, c, and d forming a geometric sequence,
    and the function y = ln(x+2) - x reaching its maximum value c when x = b,
    prove that ad = -1 -/
theorem geometric_sequence_and_max_function (a b c d : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric sequence condition
  (∀ x : ℝ, Real.log (x + 2) - x ≤ Real.log (b + 2) - b) →   -- maximum value condition
  (Real.log (b + 2) - b = c) →                         -- maximum value is c
  a * d = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_and_max_function_l637_63783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_roots_of_three_quadratics_l637_63778

/-- Helper function to count the number of real roots of a quadratic polynomial -/
noncomputable def number_of_real_roots (f : ℝ → ℝ) : ℕ := sorry

/-- Given three positive real numbers, the maximum total number of real roots
    among three specific quadratic polynomials is 4. -/
theorem max_roots_of_three_quadratics (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (n : ℕ),
    n ≤ 4 ∧
    (∀ (m : ℕ),
      (∃ (x y z : ℕ),
        x = number_of_real_roots (λ t ↦ a * t^2 + b * t + c) ∧
        y = number_of_real_roots (λ t ↦ b * t^2 + c * t + a) ∧
        z = number_of_real_roots (λ t ↦ c * t^2 + a * t + b) ∧
        m = x + y + z) →
      m ≤ n) ∧
    (∃ (x y z : ℕ),
      x = number_of_real_roots (λ t ↦ a * t^2 + b * t + c) ∧
      y = number_of_real_roots (λ t ↦ b * t^2 + c * t + a) ∧
      z = number_of_real_roots (λ t ↦ c * t^2 + a * t + b) ∧
      n = x + y + z) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_roots_of_three_quadratics_l637_63778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_l637_63756

/-- Represents a triangle -/
structure Triangle where
  base_length : ℝ

/-- Represents a line segment -/
structure Segment where
  length : ℝ

/-- Checks if a segment is parallel to the base of a triangle -/
def Segment.parallel_to (s : Segment) (base : ℝ) : Prop :=
  sorry

/-- Calculates the ratio of the area below the crease to the total area of the triangle -/
def area_ratio_below_base (t : Triangle) (s : Segment) : ℝ :=
  sorry

/-- Given a triangle ABC with base 15 cm and a crease DE parallel to the base,
    where the area of the triangle below the base is 25% of the area of ABC,
    prove that the length of DE is 7.5 cm. -/
theorem crease_length (ABC : Triangle) (DE : Segment) :
  ABC.base_length = 15 →
  DE.parallel_to ABC.base_length →
  area_ratio_below_base ABC DE = 0.25 →
  DE.length = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_l637_63756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_in_range_l637_63746

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2)*x + 2

theorem monotonic_f_implies_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  4 ≤ a ∧ a < 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_in_range_l637_63746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_theorem_l637_63768

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem t_range_theorem (f : ℝ → ℝ) (t : ℝ) :
  (is_odd_function f) →
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f x ∈ Set.Icc (-1 : ℝ) 1) →
  (f 1 = 1) →
  (∀ m n, m ∈ Set.Icc (-1 : ℝ) 1 → n ∈ Set.Icc (-1 : ℝ) 1 → m + n ≠ 0 → (f m + f n) / (m + n) > 0) →
  (∀ x a, x ∈ Set.Icc (-1 : ℝ) 1 → a ∈ Set.Icc (-1 : ℝ) 1 → f x ≤ t^2 - 2*a*t + 1) →
  (t ≤ -2 ∨ t = 0 ∨ t ≥ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_theorem_l637_63768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_inequality_solution_set_l637_63724

theorem fraction_inequality_solution_set (x : ℝ) : 
  (x + 1) / (x - 4) ≥ 3 ↔ x ∈ Set.Iic (13/2) \ {4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_inequality_solution_set_l637_63724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_umbrella_numbers_count_l637_63796

/-- A function that returns the number of permutations of n items taken k at a time -/
def permutations (n : ℕ) (k : ℕ) : ℕ :=
  if k ≤ n then
    List.range n |>.take k |>.foldl (· * ·) 1
  else
    0

/-- The set of digits that can be used to form the numbers -/
def digit_set : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- A function that checks if a three-digit number is an "umbrella number" -/
def is_umbrella_number (hundreds tens ones : ℕ) : Bool :=
  hundreds ∈ digit_set ∧ tens ∈ digit_set ∧ ones ∈ digit_set ∧
  hundreds ≠ tens ∧ tens ≠ ones ∧ hundreds ≠ ones ∧
  tens > hundreds ∧ tens > ones

/-- The theorem stating that the number of "umbrella numbers" is 40 -/
theorem umbrella_numbers_count :
  (Finset.filter (fun (n : ℕ × ℕ × ℕ) => is_umbrella_number n.1 n.2.1 n.2.2)
    (Finset.product digit_set (Finset.product digit_set digit_set))).card = 40 := by
  sorry

#check umbrella_numbers_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_umbrella_numbers_count_l637_63796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l637_63759

noncomputable section

/-- Definition of the ellipse C -/
def ellipse (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

/-- Definition of eccentricity -/
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

/-- Definition of the area of triangle A1F1B -/
def triangle_area (a b c : ℝ) : ℝ := (1/2) * abs (a*b + b*c)

/-- Theorem stating the properties of the ellipse and the ratio |OM|/|ON| -/
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity a b = 2/3) 
  (h4 : triangle_area a b (a * eccentricity a b) = Real.sqrt 5 / 2) :
  (∃ (x y : ℝ), ellipse x y 3 (Real.sqrt 5)) ∧ 
  (∃ (l : ℝ → ℝ), ∃ (E F M N : ℝ × ℝ),
    l 0 = 1 ∧ 
    ellipse (l (E.2)) (E.2) 3 (Real.sqrt 5) ∧
    ellipse (l (F.2)) (F.2) 3 (Real.sqrt 5) ∧
    E.2 > 0 ∧ F.2 < 0 ∧
    M.1 = 0 ∧ N.1 = 0 ∧
    M.2 / N.2 = -1/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l637_63759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_increase_l637_63730

/-- Represents the distance traveled by a car with increasing speed over time -/
noncomputable def distance_traveled (initial_speed : ℝ) (speed_increase : ℝ) (hours : ℕ) : ℝ :=
  (hours : ℝ) * initial_speed + speed_increase * ((hours : ℝ) * ((hours : ℝ) - 1) / 2)

/-- Theorem stating the hourly speed increase of a car given initial and total distance -/
theorem car_speed_increase
  (initial_distance : ℝ)
  (total_distance : ℝ)
  (total_hours : ℕ)
  (h1 : initial_distance = 30)
  (h2 : total_distance = 492)
  (h3 : total_hours = 12)
  (h4 : distance_traveled initial_distance speed_increase total_hours = total_distance) :
  speed_increase = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_increase_l637_63730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_α_value_tan_β_value_angle_sum_l637_63751

-- Define the coordinate system and angles
variable (α β : Real)

-- Define point A on the unit circle
variable (A : Real × Real)

-- Conditions
axiom A_on_unit_circle : A.1^2 + A.2^2 = 1
axiom A_ordinate : A.2 = Real.sqrt 10 / 10
axiom β_terminal_side : ∀ (x y : Real), x ≥ 0 → (x - 7*y = 0 ↔ y = (Real.tan β) * x)

-- Theorems to prove
theorem tan_α_value : Real.tan α = 1/3 := by sorry

theorem tan_β_value : Real.tan β = 1/7 := by sorry

theorem angle_sum : 2*α + β = π/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_α_value_tan_β_value_angle_sum_l637_63751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l637_63776

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) + Real.sin (ω * x) ^ 2 - 1/2

theorem f_properties :
  ∃ (ω : ℝ), 
    (∀ (x : ℝ), f ω (x + π / ω) = f ω x) ∧  -- period is π
    (∀ (x : ℝ), f ω x = Real.sin (2 * x - π / 6)) ∧  -- simplified expression
    (∀ (x : ℝ), x ∈ Set.Icc 0 (π / 2) → f ω x ≤ 1) ∧  -- maximum value
    (∃ (x : ℝ), x ∈ Set.Icc 0 (π / 2) ∧ f ω x = 1) ∧  -- maximum is attained
    (∀ (x : ℝ), x ∈ Set.Icc 0 (π / 2) → f ω x ≥ -1/2) ∧  -- minimum value
    (∃ (x : ℝ), x ∈ Set.Icc 0 (π / 2) ∧ f ω x = -1/2)  -- minimum is attained
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l637_63776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_two_digit_factors_eq_8_l637_63742

/-- The number of positive two-digit integers that are factors of 3^20 - 1 -/
def count_two_digit_factors : ℕ :=
  Finset.filter (λ n ↦ 10 ≤ n ∧ n < 100 ∧ (3^20 - 1) % n = 0) (Finset.range 100) |>.card

theorem count_two_digit_factors_eq_8 : count_two_digit_factors = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_two_digit_factors_eq_8_l637_63742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l637_63700

/-- Given that the terminal side of angle α passes through point P(-5,12), prove that cos α = -5/13 -/
theorem cos_alpha_value (α : ℝ) (h : ∃ (t : ℝ), t > 0 ∧ t * Real.cos α = -5 ∧ t * Real.sin α = 12) :
  Real.cos α = -5/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l637_63700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_nine_l637_63767

/-- Regular triangular pyramid with base edge length 6 and side edge length √15 -/
structure RegularTriangularPyramid where
  base_edge : ℝ
  side_edge : ℝ
  is_regular : base_edge = 6 ∧ side_edge = Real.sqrt 15

/-- Volume of a regular triangular pyramid -/
noncomputable def volume (p : RegularTriangularPyramid) : ℝ :=
  (1 / 3) * ((Real.sqrt 3) / 4 * p.base_edge ^ 2) * Real.sqrt 3

/-- The volume of the specific regular triangular pyramid is 9 -/
theorem volume_is_nine (p : RegularTriangularPyramid) : volume p = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_nine_l637_63767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_sum_diff_l637_63728

theorem lcm_gcd_sum_diff (a b : ℕ) : 
  Nat.lcm a b = 2010 → Nat.gcd a b = 2 → 
  (∃ (x y : ℕ), Nat.lcm x y = 2010 ∧ Nat.gcd x y = 2 ∧ x + y = 2012) ∧
  (∃ (u v : ℕ), Nat.lcm u v = 2010 ∧ Nat.gcd u v = 2 ∧ Int.natAbs (u - v) = 104) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_sum_diff_l637_63728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_rhombus_area_l637_63703

/-- A rhombus with given diagonal lengths and side length -/
structure Rhombus where
  d1 : ℝ  -- Length of the first diagonal
  d2 : ℝ  -- Length of the second diagonal
  s : ℝ   -- Length of each side
  h_positive : d1 > 0 ∧ d2 > 0 ∧ s > 0  -- All lengths are positive
  h_pythagoras : (d1 / 2) ^ 2 + (d2 / 2) ^ 2 = s ^ 2  -- Pythagoras theorem holds

/-- The area of a rhombus -/
noncomputable def rhombusArea (r : Rhombus) : ℝ := r.d1 * r.d2 / 2

/-- Theorem: The area of the specific rhombus is 360 square units -/
theorem specific_rhombus_area :
  let r : Rhombus := {
    d1 := 40,
    d2 := 18,
    s := 25,
    h_positive := by sorry
    h_pythagoras := by sorry
  }
  rhombusArea r = 360 := by
  unfold rhombusArea
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_rhombus_area_l637_63703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l637_63704

/-- Definition of the ellipse E -/
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- Definition of the unit circle -/
def unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- Definition of the line l -/
def line (k m x y : ℝ) : Prop :=
  y = k * x + m

/-- Theorem statement -/
theorem ellipse_and_line_properties :
  ∃ (k m : ℝ),
    /- The ellipse passes through (1, √3/2) -/
    ellipse 1 (Real.sqrt 3 / 2) ∧
    /- The eccentricity of the ellipse is √3/2 -/
    Real.sqrt (1 - (1/4)) = Real.sqrt 3 / 2 ∧
    /- There exists a line intersecting the ellipse at two points -/
    ∃ (x₁ y₁ x₂ y₂ : ℝ),
      ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
      line k m x₁ y₁ ∧ line k m x₂ y₂ ∧
      /- The sum of slopes of OP and OQ is 2 -/
      y₁ / x₁ + y₂ / x₂ = 2 ∧
      /- The line is tangent to the unit circle -/
      ∃ (xt yt : ℝ), unit_circle xt yt ∧ line k m xt yt ∧
        ∀ (x y : ℝ), unit_circle x y → (y - yt)^2 ≤ k^2 * (x - xt)^2 ∧
    /- The line equation is y = -x ± √2 -/
    ((k = -1 ∧ m = Real.sqrt 2) ∨ (k = -1 ∧ m = -Real.sqrt 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l637_63704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_square_inequality_l637_63762

open Real Set

theorem function_inequality_implies_square_inequality 
  (f : ℝ → ℝ) 
  (x₁ x₂ : ℝ) 
  (h_def : ∀ x, f x = exp (1 + sin x) + exp (1 - sin x))
  (h_range : x₁ ∈ Icc (-π/2) (π/2) ∧ x₂ ∈ Icc (-π/2) (π/2))
  (h_ineq : f x₁ > f x₂) : 
  x₁^2 > x₂^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_square_inequality_l637_63762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_for_trig_equation_l637_63790

open Real

theorem unique_n_for_trig_equation :
  ∀ n : ℕ, n > 0 → (∀ θ : ℝ, θ % (π / 2) ≠ 0 →
    (sin (n * θ) / sin θ) - (cos (n * θ) / cos θ) = ↑n - 1) ↔ n = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_for_trig_equation_l637_63790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mapping_validity_l637_63734

-- Define the sets and functions for each option
def A_optionA : Set ℝ := {x | x ≥ 0}
def B_optionA : Set ℝ := {y | y ≥ 0}
noncomputable def f_optionA : ℝ → ℝ := λ x ↦ x^2

def A_optionB : Set ℝ := {x | x > 0 ∨ x < 0}
def B_optionB : Set ℝ := {1}
def f_optionB : ℝ → ℝ := λ x ↦ x

def A_optionC : Set ℝ := Set.univ
def B_optionC : Set ℝ := Set.univ
noncomputable def f_optionC : ℝ → ℝ := λ x ↦ 2^x

def A_optionD : Finset ℕ := {2, 3}
def B_optionD : Finset ℕ := {4, 9}
def f_optionD : ℕ → ℕ := λ x ↦ x

-- Theorem statement
theorem mapping_validity :
  (∀ x ∈ A_optionA, f_optionA x ∈ B_optionA) ∧
  (∀ x ∈ A_optionB, f_optionB x ∈ B_optionB) ∧
  (∀ x ∈ A_optionC, f_optionC x ∈ B_optionC) ∧
  ¬(∀ x ∈ A_optionD, f_optionD x ∈ B_optionD) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mapping_validity_l637_63734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l637_63727

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * (x - 4)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≠ -27} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l637_63727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l637_63712

-- Define the line and parabola equations
noncomputable def line (x : ℝ) : ℝ := x - 2
noncomputable def parabola (x : ℝ) : ℝ := Real.sqrt (8 * x)

-- Define the intersection points
noncomputable def point_A : ℝ × ℝ := (4 - 4 * Real.sqrt 2, 2 - 4 * Real.sqrt 2)
noncomputable def point_B : ℝ × ℝ := (4 + 4 * Real.sqrt 2, 2 + 4 * Real.sqrt 2)

-- Theorem statement
theorem intersection_distance :
  let d := Real.sqrt ((point_B.1 - point_A.1)^2 + (point_B.2 - point_A.2)^2)
  d = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l637_63712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_revolution_volume_l637_63735

-- Define the triangle vertices
def vertex1 : ℝ × ℝ := (1003, 0)
def vertex2 : ℝ × ℝ := (1004, 3)
def vertex3 : ℝ × ℝ := (1005, 1)

-- Define the triangle
def triangle : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
  p = (a * vertex1.1 + b * vertex2.1 + c * vertex3.1, a * vertex1.2 + b * vertex2.2 + c * vertex3.2)}

-- Define the volume of the solid obtained by revolving the triangle around the y-axis
noncomputable def volume : ℝ := 2 * Real.pi * ∫ (p : ℝ × ℝ) in triangle, p.1

-- Theorem statement
theorem triangle_revolution_volume : volume = 5020 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_revolution_volume_l637_63735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_tripping_percentage_l637_63758

theorem jake_tripping_percentage :
  let p_trip_drop : ℝ := 0.25  -- Probability of dropping coffee when tripping
  let p_not_drop : ℝ := 0.9    -- Probability of not dropping coffee on any morning
  ∀ p_trip : ℝ,                -- Probability of tripping (to be proven)
  p_trip_drop * p_trip = 1 - p_not_drop → p_trip = 0.4
  := by
    intro p_trip
    intro h
    -- The proof steps would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_tripping_percentage_l637_63758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_A_inter_complement_B_l637_63797

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Theorem for the complement of A in ℝ
theorem complement_of_A : Set.compl A = {x : ℝ | x ≤ 1 ∨ x ≥ 4} := by sorry

-- Theorem for A ∩ (complement of B in ℝ)
theorem A_inter_complement_B : A ∩ Set.compl B = Set.Ioo 3 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_A_inter_complement_B_l637_63797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_m_mod_5_l637_63744

theorem remainder_m_mod_5 (m n : ℕ) (h : m = 15 * n - 1) : m % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_m_mod_5_l637_63744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_functions_l637_63721

noncomputable def f1 (x : ℝ) : ℝ := Real.cos (abs (2 * x))
noncomputable def f2 (x : ℝ) : ℝ := abs (Real.cos x)
noncomputable def f3 (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)
noncomputable def f4 (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 4)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  has_period f p ∧ p > 0 ∧ ∀ q, 0 < q ∧ q < p → ¬(has_period f q)

theorem period_of_functions :
  (smallest_positive_period f1 Real.pi) ∧
  (smallest_positive_period f2 Real.pi) ∧
  (smallest_positive_period f3 Real.pi) ∧
  (smallest_positive_period f4 (Real.pi / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_functions_l637_63721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l637_63782

theorem triangle_problem (A B C : Real) (AM : Real) :
  Real.sin A = Real.sin B ∧ 
  Real.sin A = -Real.cos C ∧ 
  A + B + C = Real.pi ∧
  AM = Real.sqrt 7 →
  A = Real.pi / 6 ∧ 
  B = Real.pi / 6 ∧ 
  C = 2 * Real.pi / 3 ∧
  (let S := 1/2 * 2 * 2 * Real.sin C; S = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l637_63782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l637_63757

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | x^2 - 3*x ≤ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l637_63757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_david_pushups_difference_l637_63764

theorem david_pushups_difference (zachary_pushups zachary_crunches zachary_total david_crunches_difference : ℕ) 
  (h1 : zachary_pushups = 53)
  (h2 : zachary_crunches = 14)
  (h3 : zachary_total = 67)
  (h4 : david_crunches_difference = 10)
  (h5 : zachary_total = zachary_pushups + zachary_crunches) :
  ∃ (david_pushups : ℕ), david_pushups = zachary_pushups + 10 := by
  -- Proof steps would go here
  sorry

#check david_pushups_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_david_pushups_difference_l637_63764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l637_63743

-- Define the original expression
noncomputable def original_expression (a : ℝ) : ℝ := 1 - (a - 1) / (a + 2) / ((a^2 - 1) / (a^2 + 2*a))

-- Define the simplified expression
noncomputable def simplified_expression (a : ℝ) : ℝ := 1 / (a + 1)

-- Theorem stating that the original expression equals the simplified expression
theorem expression_simplification (a : ℝ) 
  (h1 : a ≠ -2) 
  (h2 : a ≠ -1) 
  (h3 : a ≠ 1) 
  (h4 : a ≠ 0) : 
  original_expression a = simplified_expression a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l637_63743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_in_interval_l637_63736

noncomputable def f (x φ : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x - φ) - Real.cos (2 * x - φ)

theorem f_range_in_interval (φ : ℝ) (h_φ : |φ| < π / 2) :
  ∃ (a b : ℝ), a = -2 ∧ b = 1 ∧
  (∀ x ∈ Set.Icc (-π / 6) (π / 3), f x φ ∈ Set.Icc a b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc (-π / 6) (π / 3), f x φ = y) ∧
  (∀ x : ℝ, f x φ = f (-x) φ) := by
  sorry

#check f_range_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_in_interval_l637_63736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l637_63775

noncomputable def data : List ℝ := [3, 6, 9, 8, 4]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - mean xs) ^ 2)).sum / xs.length

theorem variance_of_data : variance data = 5.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l637_63775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_78_degrees_l637_63707

noncomputable def clock_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  let hour_angle := (hour % 12 + minute / 60 : ℝ) * 30
  let minute_angle := minute * 6
  min (abs (hour_angle - minute_angle)) (360 - abs (hour_angle - minute_angle))

def is_valid_time (hour : ℕ) (minute : ℕ) : Prop :=
  hour ≥ 7 ∧ hour < 8 ∧ minute < 60

theorem clock_hands_78_degrees :
  ∃ (h₁ m₁ h₂ m₂ : ℕ),
    is_valid_time h₁ m₁ ∧
    is_valid_time h₂ m₂ ∧
    clock_angle h₁ m₁ = 78 ∧
    clock_angle h₂ m₂ = 78 ∧
    ((h₁ = 7 ∧ m₁ = 24) ∨ (h₁ = 7 ∧ m₁ = 52)) ∧
    ((h₂ = 7 ∧ m₂ = 24) ∨ (h₂ = 7 ∧ m₂ = 52)) ∧
    h₁ ≠ h₂ ∨ m₁ ≠ m₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_78_degrees_l637_63707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sqrt_function_l637_63755

-- Define the function f(x) = √x
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- Theorem statement
theorem range_of_sqrt_function :
  Set.range f = Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sqrt_function_l637_63755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_oscillation_correct_l637_63788

/-- Represents a water pogo as a rectangular parallelepiped -/
structure WaterPogo where
  mass : ℝ
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the properties of water -/
structure Water where
  density : ℝ

/-- Calculates the period of small oscillations for a water pogo -/
noncomputable def periodOfOscillation (pogo : WaterPogo) (water : Water) (g : ℝ) : ℝ :=
  2 * Real.pi * Real.sqrt (pogo.mass / (water.density * pogo.width * pogo.length * g))

/-- Theorem stating that the calculated period is correct for small oscillations -/
theorem period_of_oscillation_correct (pogo : WaterPogo) (water : Water) (g : ℝ) 
    (h_mass_pos : pogo.mass > 0)
    (h_length_pos : pogo.length > 0)
    (h_width_pos : pogo.width > 0)
    (h_height_pos : pogo.height > 0)
    (h_density_pos : water.density > 0)
    (h_g_pos : g > 0)
    (h_floating : pogo.mass = water.density * pogo.length * pogo.width * pogo.height)
    (h_small_oscillations : True) -- Assumption for small oscillations
    : periodOfOscillation pogo water g = 2 * Real.pi * Real.sqrt (pogo.mass / (water.density * pogo.width * pogo.length * g)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_oscillation_correct_l637_63788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_a_equals_4_l637_63748

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -4 * x else x^2

-- State the theorem
theorem f_a_equals_4 (a : ℝ) (h : f a = 4) : a = -1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_a_equals_4_l637_63748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l637_63792

theorem function_inequality (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 3 ∧ 1 ≤ x₂ ∧ x₂ ≤ 3 → 
    (x₁^2 - x₁ - 1) / (x₁ + 1) ≥ -Real.exp (x₂ - 1) - Real.log x₂ + a) →
  a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l637_63792
