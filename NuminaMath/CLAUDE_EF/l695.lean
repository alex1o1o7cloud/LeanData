import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_eq1_solutions_eq2_solutions_eq3_solutions_eq4_solutions_eq5_solution_eq6_l695_69569

-- 1
theorem no_solutions_eq1 : ∀ x : ℝ, x^2 - 3*x ≠ 0 ∨ x - 3 < 0 := by sorry

-- 2
theorem solutions_eq2 : {x : ℝ | x^2 - 6*|x| + 9 = 0} = {3, -3} := by sorry

-- 3
theorem solutions_eq3 : {x : ℝ | (x - 1)^4 = 4} = {1 + Real.sqrt 2, 1 - Real.sqrt 2} := by sorry

-- 4
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem solutions_eq4 : {x : ℝ | floor (x^2) + x = 6} = {-3, 2} := by sorry

-- 5
theorem solutions_eq5 : {x : ℝ | x^2 + x = 0 ∨ x^2 - 1 = 0} = {0, -1, 1} := by sorry

-- 6
theorem solution_eq6 : {x : ℝ | x^2 - 2*x - 3 = 0 ∧ |x| < 2} = {-1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_eq1_solutions_eq2_solutions_eq3_solutions_eq4_solutions_eq5_solution_eq6_l695_69569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zara_july_earnings_l695_69545

/-- Represents Zara's work and earnings for two weeks in July -/
structure ZaraWork where
  hours_week1 : ℕ
  hours_week2 : ℕ
  extra_earnings : ℚ
  hourly_wage : ℚ

/-- Calculates the total earnings for two weeks given Zara's work structure -/
def total_earnings (z : ZaraWork) : ℚ :=
  z.hourly_wage * (z.hours_week1 + z.hours_week2)

/-- Theorem stating Zara's total earnings for two weeks in July -/
theorem zara_july_earnings :
  ∀ (z : ZaraWork),
  z.hours_week1 = 18 →
  z.hours_week2 = 24 →
  z.extra_earnings = 53.20 →
  z.hourly_wage * (z.hours_week2 - z.hours_week1) = z.extra_earnings →
  ∃ (earnings : ℚ),
  total_earnings z = earnings ∧
  (⌊earnings * 100⌋ : ℚ) / 100 = 371.60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zara_july_earnings_l695_69545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_property_l695_69560

-- Define the curves C1 and C2
noncomputable def C1 (a : ℝ) (t : ℝ) : ℝ × ℝ := (a + Real.sqrt 2 * t, 1 + Real.sqrt 2 * t)

def C2 (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (a, 1)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_points_property (a : ℝ) :
  (∃ t1 t2 : ℝ, t1 ≠ t2 ∧
    C2 (C1 a t1).1 (C1 a t1).2 ∧
    C2 (C1 a t2).1 (C1 a t2).2 ∧
    distance (P a) (C1 a t1) = 2 * distance (P a) (C1 a t2)) ↔
  (a = 1/36 ∨ a = 9/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_property_l695_69560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l695_69580

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x^4 + k*x^2 + 1) / (x^4 + x^2 + 1)

-- State the theorem
theorem f_properties (k : ℝ) :
  (∀ x : ℝ, f k x ≤ max 1 ((k + 2) / 3)) ∧
  (∀ x : ℝ, f k x ≥ min 1 ((k + 2) / 3)) ∧
  (∀ a b c : ℝ, ∃ (triangle : ℝ → ℝ → ℝ → Prop),
    triangle (f k a) (f k b) (f k c) ↔ -1/2 < k ∧ k < 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l695_69580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_problem_l695_69524

theorem teacher_age_problem (num_students : ℕ) (student_avg_age : ℝ) (total_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 30 →
  student_avg_age = 15 →
  total_avg_age = 16 →
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = total_avg_age →
  teacher_age = 46 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_problem_l695_69524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_value_l695_69564

theorem max_cos_value (a b : ℝ) (h : Real.cos (a - b) = Real.cos a - Real.cos b) :
  ∃ (max_cos : ℝ), max_cos = Real.sqrt ((3 + Real.sqrt 5) / 2) ∧
    ∀ x, Real.cos x ≤ max_cos := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_value_l695_69564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sum_equals_twelve_plus_one_eighteenth_l695_69553

noncomputable def a : ℕ → ℝ
  | n => if n ≤ 2 then 2^(n+1) else (1/3)^n

noncomputable def S : ℕ → ℝ
  | n => (Finset.range n).sum (λ i => a (i+1))

theorem limit_of_sum_equals_twelve_plus_one_eighteenth :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |S n - (12 + 1/18)| < ε := by
  sorry

#check limit_of_sum_equals_twelve_plus_one_eighteenth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sum_equals_twelve_plus_one_eighteenth_l695_69553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_ratio_l695_69572

/-- Parabola defined by x^2 = 4y -/
def Parabola := {p : ℝ × ℝ | p.1^2 = 4 * p.2}

/-- Line with slope √3 passing through (0, 1) -/
noncomputable def Line := {p : ℝ × ℝ | p.1 = Real.sqrt 3 * (p.2 - 1)}

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (0, 1)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_line_intersection_ratio :
  ∃ (A B : ℝ × ℝ),
    A ∈ Parabola ∧ B ∈ Parabola ∧
    A ∈ Line ∧ B ∈ Line ∧
    A.1 > 0 ∧ A.2 > 0 ∧
    distance A Focus / distance B Focus = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_ratio_l695_69572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_comparison_l695_69554

noncomputable def pyramid_volume (lateral_edge : ℝ) : ℝ :=
  let base_half := lateral_edge / 2
  let height := Real.sqrt (1 - base_half^2)
  (1/3) * lateral_edge^2 * height

theorem pyramid_volume_comparison :
  pyramid_volume 1.33 > pyramid_volume 1.25 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_comparison_l695_69554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_queries_for_card_determination_l695_69516

/-- Represents a query that reveals the set of numbers on three cards. -/
structure Query where
  cards : Finset Nat

/-- The type of a function that determines the number on each card. -/
def DeterminationFunction := Nat → Nat

/-- Theorem stating the minimum number of queries needed to determine the numbers on 2005 cards. -/
theorem minimum_queries_for_card_determination :
  ∀ (n : Nat) (f : DeterminationFunction),
  n = 2005 →
  (∀ i j, i < n → j < n → i ≠ j → f i ≠ f j) →
  ∃ (queries : Finset Query),
  (∀ q ∈ queries, q.cards.card = 3) ∧
  (∀ i < n, ∃ q ∈ queries, i ∈ q.cards) ∧
  (∀ g : DeterminationFunction,
    (∀ q ∈ queries, {f i | i ∈ q.cards} = {g i | i ∈ q.cards}) →
    ∀ i < n, f i = g i) →
  queries.card ≥ 1003 := by
  sorry

#check minimum_queries_for_card_determination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_queries_for_card_determination_l695_69516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_17_2024_l695_69598

def units_digit (n : ℕ) : ℕ := n % 10

def power_cycle (base : ℕ) (cycle : List ℕ) : ℕ → ℕ
  | n => cycle[n % cycle.length]'sorry

theorem units_digit_17_2024 
  (h1 : ∀ n : ℕ, units_digit (17^n) = units_digit (7^n))
  (h2 : power_cycle 7 [7, 9, 3, 1] = units_digit ∘ (λ n => 7^n))
  (h3 : 2024 % 4 = 0) :
  units_digit (17^2024) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_17_2024_l695_69598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l695_69505

/-- Sequence a_n with given properties -/
def a : ℕ → ℝ := sorry

/-- Sum of first n terms of sequence a_n -/
def S : ℕ → ℝ := sorry

/-- Sequence b_n defined in terms of a_n -/
def b : ℕ → ℝ := sorry

/-- Sum of first n terms of sequence b_n -/
def T : ℕ → ℝ := sorry

/-- Main theorem stating the properties of sequences a_n and b_n -/
theorem sequence_properties :
  (∀ n, 4 * S n = (2 * n - 1) * a (n + 1) + 1) →
  a 1 = 1 →
  (∀ n, a n = 2 * n - 1) ∧
  (∀ n, b n / a n = (Real.sqrt 2) ^ (1 + a n)) →
  (∀ n, T n = (2 * n - 3) * 2^(n + 1) + 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l695_69505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_fourth_l695_69537

theorem cos_alpha_plus_pi_fourth (α : ℝ) 
  (h1 : Real.sin (π - α) = 4/5)
  (h2 : 0 < α ∧ α < π/2) : 
  Real.cos (α + π/4) = -Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_fourth_l695_69537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_theta_is_plane_l695_69530

-- Define spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the equation θ = c
def constantThetaEquation (c : ℝ) (point : SphericalCoord) : Prop :=
  point.θ = c

-- Define a plane in 3D space
noncomputable def isPlane (S : Set SphericalCoord) : Prop :=
  ∃ (a b d : ℝ), ∀ (p : SphericalCoord), p ∈ S ↔ 
    a * (p.ρ * Real.sin p.φ * Real.cos p.θ) + 
    b * (p.ρ * Real.sin p.φ * Real.sin p.θ) + 
    d * (p.ρ * Real.cos p.φ) = 0

-- Theorem statement
theorem constant_theta_is_plane (c : ℝ) :
  isPlane {p : SphericalCoord | constantThetaEquation c p} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_theta_is_plane_l695_69530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_tracks_sides_l695_69523

-- Define the structure for a square track
structure SquareTrack where
  side : ℚ
  deriving Repr

-- Define the structure for an ant
structure Ant where
  track : SquareTrack
  direction : Bool  -- true for counterclockwise, false for clockwise
  deriving Repr

-- Define the problem setup
def problemSetup (a b c : ℚ) : Prop :=
  let smallTrack := SquareTrack.mk a
  let mediumTrack := SquareTrack.mk b
  let largeTrack := SquareTrack.mk c
  let mu := Ant.mk largeTrack true
  let ra := Ant.mk mediumTrack true
  let vey := Ant.mk smallTrack false
  
  -- Conditions
  (b - a = 2) ∧ 
  (c - b = 2) ∧
  (mu.track.side = 8) ∧
  (ra.track.side = 6) ∧
  (vey.track.side = 4) ∧
  -- When Mu reaches the lower right corner
  (mu.track.side = 8) →
  -- Ra and Vey are on the right sides of their tracks
  (∃ (ra_pos vey_pos : ℚ),
    0 < ra_pos ∧ ra_pos < ra.track.side ∧
    0 < vey_pos ∧ vey_pos < vey.track.side ∧
    -- All three ants are on the same straight line
    ra_pos + 1 = vey_pos + 2)

-- Theorem statement
theorem square_tracks_sides :
  ∀ a b c : ℚ, problemSetup a b c → (a = 4 ∧ b = 6 ∧ c = 8) := by
  sorry

#check square_tracks_sides

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_tracks_sides_l695_69523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l695_69533

open Real

/-- The probability that two randomly chosen integers are relatively prime -/
noncomputable def prob_two_coprime : ℝ := 6 / Real.pi^2

/-- The probability that four randomly chosen integers have a common factor -/
noncomputable def prob_four_common_factor : ℝ := 1 - 90 / Real.pi^4

/-- Theorem stating the probabilities for coprime pairs and common factors -/
theorem probability_theorem :
  (prob_two_coprime = 6 / Real.pi^2) ∧
  (prob_four_common_factor = 1 - 90 / Real.pi^4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l695_69533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_average_speed_l695_69519

/-- Calculate average speed given total distance and total time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem max_average_speed :
  let total_distance : ℝ := 56
  let total_time : ℝ := 7
  average_speed total_distance total_time = 8 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_average_speed_l695_69519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l695_69590

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^3 else -x^3

-- Theorem statement
theorem range_of_a (a : ℝ) :
  f (3*a - 1) - 8 * f a ≥ 0 → a ≤ 1/5 ∨ a ≥ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l695_69590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l695_69558

-- Define the parametric equations for curve M
noncomputable def curve_M (t : ℝ) : ℝ × ℝ :=
  (2 * Real.sqrt 3 / (Real.sqrt 3 - t), 2 * Real.sqrt 3 * t / (Real.sqrt 3 - t))

-- Define the polar equation for curve C
noncomputable def curve_C (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Theorem statement
theorem intersection_point :
  ∃ (t θ : ℝ), t > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  let (x, y) := curve_M t
  let ρ := Real.sqrt (x^2 + y^2)
  ρ = curve_C θ ∧
  ρ = 2 * Real.sqrt 3 ∧
  θ = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l695_69558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_kth_powers_l695_69509

-- Define the arithmetic progression
def arithmeticProgression (a d : ℕ) : ℕ → ℕ
  | 0 => a
  | n + 1 => arithmeticProgression a d n + d

theorem arithmetic_progression_kth_powers (a d k : ℕ) (ha : a > 0) (hd : d > 0) (hk : k > 0) :
  (∃ n, ∃ m, arithmeticProgression a d n = m^k) →
  ∀ N, ∃ n > N, ∃ m, arithmeticProgression a d n = m^k :=
by
  sorry

#check arithmetic_progression_kth_powers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_kth_powers_l695_69509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l695_69514

-- Define the constants a and b
def a : ℝ := 20.3
def b : ℝ := 0.32

-- Define the function c(x)
noncomputable def c (x : ℝ) : ℝ := Real.log (x^2 + 0.3) / Real.log x

-- State the theorem
theorem relationship_abc :
  ∀ x > 1, b < a ∧ a < c x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l695_69514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_equivalence_eval_at_zero_eval_at_three_l695_69557

-- Define the original expression
noncomputable def original_expr (x : ℝ) : ℝ := (1 / (x + 3) - 1) / ((x^2 - 4) / (x^2 + 6*x + 9))

-- Define the simplified expression
noncomputable def simplified_expr (x : ℝ) : ℝ := (x + 3) / (2 - x)

-- Theorem stating the equivalence of the expressions
theorem expr_equivalence (x : ℝ) (h1 : x ≠ -3) (h2 : x ≠ -2) (h3 : x ≠ 2) : 
  original_expr x = simplified_expr x := by sorry

-- Theorem for the evaluation when x = 0
theorem eval_at_zero : simplified_expr 0 = 3/2 := by sorry

-- Theorem for the evaluation when x = 3
theorem eval_at_three : simplified_expr 3 = -6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_equivalence_eval_at_zero_eval_at_three_l695_69557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_after_removing_top_scores_l695_69589

theorem new_average_after_removing_top_scores (n : ℕ) (orig_avg top1 top2 : ℚ) :
  n = 60 →
  orig_avg = 72 →
  top1 = 85 →
  top2 = 90 →
  let orig_sum := n * orig_avg
  let new_sum := orig_sum - (top1 + top2)
  let new_avg := new_sum / (n - 2)
  abs (new_avg - 71.47) < 0.01 :=
by
  intros hn horig htop1 htop2
  -- The proof steps would go here
  sorry

#eval abs (((60 : ℚ) * 72 - (85 + 90)) / 58 - 71.47) < 0.01

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_after_removing_top_scores_l695_69589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l695_69541

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x + 1)

def point_of_tangency : ℝ × ℝ := (0, 1)

theorem tangent_line_equation :
  let x₀ := point_of_tangency.1
  let y₀ := point_of_tangency.2
  let f' := λ x => (1 / 2) * (2 / Real.sqrt (2 * x + 1))
  let slope := f' x₀
  let tangent_line := λ x y => x - y + 1
  tangent_line x₀ y₀ = 0 ∧ ∀ x, tangent_line x (y₀ + slope * (x - x₀)) = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l695_69541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_extreme_points_l695_69503

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x - 3 * a * abs (2 * Real.log x - x^2 + 1)

def has_two_extreme_points (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
  (∀ x ∈ Set.Ioo 0 x₁, HasDerivAt f (deriv f x) x) ∧
  (∀ x ∈ Set.Ioo x₁ x₂, HasDerivAt f (deriv f x) x) ∧
  (∀ x ∈ Set.Ioi x₂, HasDerivAt f (deriv f x) x) ∧
  deriv f x₁ = 0 ∧ deriv f x₂ = 0 ∧
  (∀ x ∈ Set.Ioo 0 x₁, deriv f x ≠ 0) ∧
  (∀ x ∈ Set.Ioo x₁ x₂, deriv f x ≠ 0) ∧
  (∀ x ∈ Set.Ioi x₂, deriv f x ≠ 0)

theorem f_monotonicity_and_extreme_points :
  (∀ x ∈ Set.Ioo 0 1, HasDerivAt (f 0) (deriv (f 0) x) x ∧ deriv (f 0) x < 0) ∧
  (∀ x ∈ Set.Ioi 1, HasDerivAt (f 0) (deriv (f 0) x) x ∧ deriv (f 0) x > 0) ∧
  (∀ a : ℝ, has_two_extreme_points (f a) ↔ a ∈ Set.Ioo 0 1 ∪ Set.Ioi 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_extreme_points_l695_69503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_area_l695_69587

/-- Represents a triangle with given perimeter and inradius -/
structure Triangle where
  perimeter : ℝ
  inradius : ℝ

/-- The area of a triangle given its perimeter and inradius -/
noncomputable def triangle_area (t : Triangle) : ℝ :=
  t.inradius * (t.perimeter / 2)

/-- Theorem stating that a triangle with perimeter 39 and inradius 1.5 has an area of 29.25 -/
theorem specific_triangle_area :
  let t : Triangle := { perimeter := 39, inradius := 1.5 }
  triangle_area t = 29.25 := by
  -- Expand the definition of triangle_area
  unfold triangle_area
  -- Perform the calculation
  simp [Triangle.perimeter, Triangle.inradius]
  -- The proof is complete
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_area_l695_69587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_decreasing_l695_69593

theorem sequence_eventually_decreasing :
  ∃ N : ℕ, ∀ n : ℕ, n > N → (100^(n+1) : ℝ) / (Nat.factorial (n+1) : ℝ) < (100^n : ℝ) / (Nat.factorial n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_decreasing_l695_69593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_northeast_southwest_angle_120_l695_69562

/-- Represents a circular arrangement of equally spaced radial lines -/
structure CircularArrangement where
  num_lines : ℕ
  first_line_north : Bool

/-- Calculates the angle between two lines in the circular arrangement -/
noncomputable def angle_between_lines (ca : CircularArrangement) (line1 : ℕ) (line2 : ℕ) : ℝ :=
  ((line2 - line1) % ca.num_lines : ℝ) * (360 / ca.num_lines)

/-- Represents cardinal directions as numbers -/
inductive Direction
  | North : Direction
  | Northeast : Direction
  | Southwest : Direction

/-- Converts a direction to its corresponding line number -/
def direction_to_line (d : Direction) : ℕ :=
  match d with
  | Direction.North => 0
  | Direction.Northeast => 2
  | Direction.Southwest => 6

/-- The main theorem to be proved -/
theorem northeast_southwest_angle_120 (ca : CircularArrangement) 
  (h1 : ca.num_lines = 12) 
  (h2 : ca.first_line_north = true) : 
  angle_between_lines ca 
    (direction_to_line Direction.Northeast) 
    (direction_to_line Direction.Southwest) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_northeast_southwest_angle_120_l695_69562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_video_recorder_price_l695_69546

/-- Calculate the final price of a video recorder for an employee, including sales tax -/
def video_recorder_price (wholesale_cost : ℝ) (markup_rate : ℝ) (employee_discount : ℝ) 
  (weekend_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  let retail_price := wholesale_cost * (1 + markup_rate)
  let price_after_employee_discount := retail_price * (1 - employee_discount)
  let price_after_weekend_discount := price_after_employee_discount * (1 - weekend_discount)
  let final_price := price_after_weekend_discount * (1 + sales_tax)
  final_price

/-- The final price an employee pays for the video recorder is $221.62 (rounded to nearest cent) -/
theorem employee_video_recorder_price : 
  (⌊video_recorder_price 200 0.20 0.05 0.10 0.08 * 100 + 0.5⌋ : ℝ) / 100 = 221.62 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_video_recorder_price_l695_69546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_speed_difference_percentage_l695_69510

/-- Represents the water flow speed of the river in km/h -/
noncomputable def river_speed : ℝ := 5

/-- Represents the distance between points A and B in km -/
noncomputable def distance : ℝ := 60

/-- Represents the still water speed of the boat in km/h -/
noncomputable def boat_speed : ℝ := 25

/-- Calculates the downstream speed of the boat -/
noncomputable def downstream_speed : ℝ := boat_speed + river_speed

/-- Calculates the upstream speed of the boat -/
noncomputable def upstream_speed : ℝ := boat_speed - river_speed

/-- Calculates the time difference between downstream and upstream travel -/
noncomputable def time_difference : ℝ := distance / upstream_speed - distance / downstream_speed

/-- Calculates the percentage by which downstream speed exceeds upstream speed -/
noncomputable def speed_difference_percentage : ℝ := (downstream_speed - upstream_speed) / upstream_speed * 100

theorem max_speed_difference_percentage :
  time_difference ≥ 1 → speed_difference_percentage ≤ 50 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_speed_difference_percentage_l695_69510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_to_focus_l695_69534

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

/-- The focus of the parabola y^2 = 4x -/
def parabola_focus : ℝ × ℝ := (1, 0)

/-- The distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_point_distance_to_focus 
  (P : ParabolaPoint) 
  (h_distance_to_y_axis : P.x = 2) :
  distance (P.x, P.y) parabola_focus = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_to_focus_l695_69534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_geometric_series_convergence_and_bound_l695_69542

noncomputable def geometric_series (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

noncomputable def alternating_geometric_series (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := 
  geometric_series a (-r) n

theorem alternating_geometric_series_convergence_and_bound 
  (a : ℝ) (r : ℝ) (h_a : a = 3) (h_r : r = 1/3) :
  (∀ n : ℕ, alternating_geometric_series a r n > 1) ∧
  (∃ L : ℝ, L = 9/4 ∧ ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |alternating_geometric_series a r n - L| < ε) := by
  sorry

#check alternating_geometric_series_convergence_and_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_geometric_series_convergence_and_bound_l695_69542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_cows_l695_69550

/-- The number of horses on the farm -/
def x : ℕ := sorry

/-- The number of ducks on the farm -/
def y : ℕ := sorry

/-- The number of cows on the farm -/
def z : ℕ := sorry

/-- There are more horses than ducks -/
axiom more_horses : x > y

/-- The number of cows is one-third of the total number of horses and ducks -/
axiom cows_ratio : z = (x + y) / 3

/-- The sum of the number of heads and legs of the ducks and horses is 100 -/
axiom heads_and_legs : 5 * x + 3 * y = 100

/-- The number of cows on the farm is 8 -/
theorem farm_cows : z = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_cows_l695_69550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l695_69571

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 4*x

/-- The focus of the parabola -/
def F : ℝ × ℝ := (1, 0)

/-- Point B -/
def B : ℝ × ℝ := (3, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_distance_theorem (A : Parabola) :
  distance (A.x, A.y) F = distance B F →
  distance (A.x, A.y) B = 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l695_69571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_from_circle_tangents_l695_69582

/-- A line is tangent to a circle at a given point -/
def is_tangent_line_to_circle (r : ℝ) (P : ℝ × ℝ) : Prop :=
  sorry

/-- A triangle is equilateral -/
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  sorry

/-- Calculate the area of a triangle given its vertices -/
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  sorry

/-- Given a circle with radius 10 cm and an equilateral triangle formed by tangent lines to the circle,
    prove that the area of the triangle is 75√3 cm². -/
theorem equilateral_triangle_area_from_circle_tangents (r : ℝ) (A B C : ℝ × ℝ) :
  r = 10 →
  is_tangent_line_to_circle r B →
  is_tangent_line_to_circle r C →
  is_equilateral_triangle A B C →
  area_triangle A B C = 75 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_from_circle_tangents_l695_69582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_g_is_transformation_l695_69506

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if -5 ≤ x ∧ x < -2 then 0  -- Linear part (placeholder)
  else if -2 ≤ x ∧ x ≤ 2 then 0  -- Upper half of circle (placeholder)
  else if 2 < x ∧ x ≤ 5 then 0  -- Linear part (placeholder)
  else 0  -- Outside the defined regions

-- Define the transformed function
noncomputable def transformed_g (x : ℝ) : ℝ :=
  g ((x + 3) / (-2)) + 1

-- Theorem statement
theorem transformed_g_is_transformation :
  ∀ x : ℝ, transformed_g x = g ((x + 3) / (-2)) + 1 :=
by
  intro x
  rfl  -- reflexivity proves the equality


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_g_is_transformation_l695_69506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_m_l695_69567

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (x^m) + 2 * Real.exp (x - 1) - 2 * x + m

-- Part 1: Tangent line equation when m = 2
theorem tangent_line_at_one :
  let m := 2
  let f' := deriv (f m)
  ∀ x, (f' 1 * (x - 1) + f m 1) = 2 * x :=
sorry

-- Part 2: Range of m for which f(x) ≥ mx holds
theorem range_of_m :
  {m : ℝ | ∀ x ≥ 1, f m x ≥ m * x} = Set.Iic 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_m_l695_69567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_and_h_lower_bound_l695_69543

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x

noncomputable def g (x : ℝ) : ℝ := Real.log x

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := f a x - g x

theorem parallel_tangents_and_h_lower_bound :
  ∃ a : ℝ, (deriv (f a) 0 = deriv g 1) ∧
  (∀ x : ℝ, x > 0 → h a x > 2) := by
  sorry

#check parallel_tangents_and_h_lower_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_and_h_lower_bound_l695_69543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shingle_area_calculation_l695_69501

/-- Calculates the total area of shingles needed for a house and porch -/
theorem shingle_area_calculation (house_length house_width porch_length porch_width : ℝ) 
  (h1 : house_length = 20.5)
  (h2 : house_width = 10)
  (h3 : porch_length = 6)
  (h4 : porch_width = 4.5) :
  house_length * house_width + porch_length * porch_width = 232 := by
  -- Substitute the given values
  rw [h1, h2, h3, h4]
  -- Simplify the expression
  norm_num
  -- The proof is complete
  done

#check shingle_area_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shingle_area_calculation_l695_69501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daniels_marbles_l695_69518

theorem daniels_marbles (x : ℚ) (hx : x > 0) : 
  let lost := (1 / 3 : ℚ) * x
  let found := (3 / 4 : ℚ) * lost
  let still_missing := x - ((x - lost) + found)
  still_missing / x = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daniels_marbles_l695_69518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_distances_l695_69532

-- Define the circles C₁ and C₂
noncomputable def C₁ (φ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos φ, 2 * Real.sin φ)
noncomputable def C₂ (φ : ℝ) : ℝ × ℝ := (Real.cos φ, 1 + Real.sin φ)

-- Define the ray OM
noncomputable def rayOM (a : ℝ) : ℝ → ℝ × ℝ := λ r => (r * Real.cos a, r * Real.sin a)

-- Define the intersection points P and Q
noncomputable def P (a : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos a, 2 * Real.sin a)
noncomputable def Q (a : ℝ) : ℝ × ℝ := (Real.cos a, 1 + Real.sin a)

-- Define the distances |OP| and |OQ|
noncomputable def distOP (a : ℝ) : ℝ := Real.sqrt (8 + 8 * Real.cos a)
noncomputable def distOQ (a : ℝ) : ℝ := Real.sqrt (2 + 2 * Real.sin a)

-- State the theorem
theorem max_product_of_distances :
  ∃ (M : ℝ), M = 4 + 2 * Real.sqrt 2 ∧
  ∀ (a : ℝ), distOP a * distOQ a ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_distances_l695_69532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distinct_values_l695_69538

theorem sum_distinct_values (x : Fin 2004 → ℝ) 
  (h : ∀ i, x i = Real.sqrt 2 - 1 ∨ x i = Real.sqrt 2 + 1) :
  (Finset.range 1002).sum (λ k => x (2*k) * x (2*k+1)) ∈ 
    Finset.image (λ n : ℕ => (1002 : ℝ) + 4*n) (Finset.range 502) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distinct_values_l695_69538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_walk_prob_l695_69507

structure Octahedron :=
  (vertices : Finset ℕ)
  (bottom : ℕ)
  (top : ℕ)
  (middle : Finset ℕ)
  (adjacent : ℕ → Finset ℕ)
  (h_vertices : vertices.card = 6)
  (h_middle : middle.card = 4)
  (h_bottom_adj : adjacent bottom = middle)
  (h_top_adj : adjacent top ⊆ middle)
  (h_middle_adj : ∀ v ∈ middle, adjacent v = insert bottom (insert top (middle.erase v)))

noncomputable def random_walk (O : Octahedron) : ℝ :=
  (1 : ℝ) / 4

theorem random_walk_prob (O : Octahedron) :
  random_walk O = (1 : ℝ) / 4 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_walk_prob_l695_69507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_of_factorial_divided_by_factorial_l695_69539

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 5)) / Nat.factorial 5 = Nat.factorial 119 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_of_factorial_divided_by_factorial_l695_69539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_20_10_l695_69561

theorem binomial_coefficient_20_10 (h1 : Nat.choose 18 8 = 31824)
                                   (h2 : Nat.choose 18 9 = 48620)
                                   (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 172822 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_20_10_l695_69561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_bars_count_l695_69575

/-- Calculates the number of ice cream bars ordered given the total number of sundaes,
    total price, price per ice cream bar, and price per sundae. -/
def ice_cream_bars_ordered (total_sundaes : ℕ) (total_price : ℚ) 
  (ice_cream_bar_price : ℚ) (sundae_price : ℚ) : ℕ :=
  let ice_cream_bars := 
    (total_price - (sundae_price * total_sundaes : ℚ)) / ice_cream_bar_price
  (ice_cream_bars.num / ice_cream_bars.den).natAbs

/-- Proves that the number of ice cream bars ordered is 125 given the problem conditions. -/
theorem ice_cream_bars_count : 
  ice_cream_bars_ordered 125 250 (60/100) (140/100) = 125 := by
  rfl

#eval ice_cream_bars_ordered 125 250 (60/100) (140/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_bars_count_l695_69575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_arithmetic_sequence_sines_l695_69556

theorem no_arithmetic_sequence_sines :
  ¬ ∃ b : ℝ, 0 < b ∧ b < 2 * Real.pi ∧
  ∃ d : ℝ, (Real.sin b - Real.sin (2*b) = d) ∧
           (Real.sin (2*b) - Real.sin (3*b) = d) ∧
           (Real.sin (3*b) - Real.sin (4*b) = d) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_arithmetic_sequence_sines_l695_69556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_amount_example_l695_69591

/-- Calculates the profit amount for a product given its selling price and profit percentage. -/
noncomputable def profit_amount (selling_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  let cost_price := selling_price / (1 + profit_percentage / 100)
  selling_price - cost_price

/-- Theorem: The profit amount for a product sold at $900 with a 20% profit is $150. -/
theorem profit_amount_example : profit_amount 900 20 = 150 := by
  -- Unfold the definition of profit_amount
  unfold profit_amount
  -- Simplify the expression
  simp
  -- Check that the result is equal to 150
  norm_num

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check profit_amount 900 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_amount_example_l695_69591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_experimental_combinations_l695_69552

/-- The number of ways to select an even number of elements from a set of 6 elements -/
def evenSelections : ℕ := 
  Finset.sum (Finset.range 4) (fun k => Nat.choose 6 (2 * k))

/-- The number of ways to select at least 2 elements from a set of 4 elements -/
def atLeastTwoSelections : ℕ := 
  Finset.sum (Finset.range 3) (fun k => Nat.choose 4 (k + 2))

/-- The total number of experimental combinations -/
def totalCombinations : ℕ := evenSelections * atLeastTwoSelections

theorem experimental_combinations : totalCombinations = 353 := by
  sorry

#eval totalCombinations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_experimental_combinations_l695_69552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_shape_graphs_l695_69573

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := 2 / x
noncomputable def g (x : ℝ) : ℝ := (x + 1) / (x - 1)

-- Define a translation function
noncomputable def translate (f : ℝ → ℝ) (h : ℝ) (k : ℝ) : ℝ → ℝ :=
  λ x ↦ f (x - h) + k

-- Theorem statement
theorem same_shape_graphs :
  ∃ (h k : ℝ), ∀ (x : ℝ), x ≠ 1 → g x = translate f h k x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_shape_graphs_l695_69573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l695_69574

/-- The circle with center (2, 3) and radius 1 -/
def myCircle (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

/-- Point A -/
def point_A : ℝ × ℝ := (-1, 4)

/-- A line passing through point A -/
def line_through_A (k : ℝ) (x y : ℝ) : Prop := y - 4 = k * (x + 1)

/-- Distance between a point and a line -/
noncomputable def distance_point_line (x₀ y₀ k m : ℝ) : ℝ :=
  |k * x₀ - y₀ + m| / Real.sqrt (k^2 + 1)

theorem tangent_line_equation :
  ∃ (k : ℝ), (∀ x y, line_through_A k x y ↔ (y = 4 ∨ 15 * x + 8 * y = 53)) ∧
             (∀ x y, line_through_A k x y → myCircle x y →
                     distance_point_line 2 3 k (k + 4) = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l695_69574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_one_branch_hyperbola_l695_69520

-- Define the points F₁ and F₂
def F₁ : ℝ × ℝ := (0, -1)
def F₂ : ℝ × ℝ := (0, 1)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the set of points P satisfying the condition
def trajectory : Set (ℝ × ℝ) :=
  {P | distance P F₁ - distance P F₂ = 1}

-- Theorem statement
theorem trajectory_is_one_branch_hyperbola :
  trajectory = {P | ∃ (x y : ℝ), P = (x, y) ∧ (x^2 / 3^2) - (y^2 / 2^2) = 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_one_branch_hyperbola_l695_69520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pendulum_constants_and_variables_l695_69529

/-- The period of small oscillations of a simple pendulum -/
noncomputable def period (l g : ℝ) : ℝ := 2 * Real.pi * Real.sqrt (l / g)

theorem pendulum_constants_and_variables :
  ∀ (l₁ l₂ g : ℝ) (h : g > 0),
  (∃ (T₁ T₂ : ℝ), T₁ ≠ T₂ ∧ T₁ = period l₁ g ∧ T₂ = period l₂ g) ∧
  (∀ (l : ℝ), period l g = 2 * Real.pi * Real.sqrt (l / g)) := by
  sorry

#check pendulum_constants_and_variables

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pendulum_constants_and_variables_l695_69529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_9_l695_69526

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem max_sum_at_9 (a₁ d : ℝ) :
  S a₁ d 17 > 0 → S a₁ d 18 < 0 →
  ∀ n : ℕ, S a₁ d n ≤ S a₁ d 9 :=
by
  sorry

#check max_sum_at_9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_9_l695_69526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_arithmetic_sequence_l695_69528

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def isArithmeticSequence (s : List ℕ) : Prop :=
  s.length > 1 ∧ ∃ d : ℕ, ∀ i : ℕ, i + 1 < s.length → s[i + 1]! - s[i]! = d

theorem max_prime_arithmetic_sequence :
  ∀ s : List ℕ,
    (∀ n ∈ s, isPrime n ∧ n < 150) →
    isArithmeticSequence s →
    s.length ≤ 5 :=
by sorry

#check max_prime_arithmetic_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_arithmetic_sequence_l695_69528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_implication_l695_69595

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (x + a) / (x^2 + b*x + 1)

theorem odd_function_and_inequality_implication 
  (a b : ℝ) 
  (h1 : ∀ x, f a b x = -f a b (-x)) 
  (h2 : ∀ k, k < 0 → ∀ t, f a b (t^2 - 2*t + 3) + f a b (k - 1) < 0) :
  (a = 0 ∧ b = 0) ∧ 
  (∀ x > 1, ∀ y > x, f 0 0 y < f 0 0 x) ∧
  (∃ k, -1 < k ∧ k < 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_implication_l695_69595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_growth_equation_l695_69517

/-- Represents the average annual growth rate of sales volume -/
noncomputable def average_growth_rate (initial_volume : ℕ) (final_volume : ℕ) (years : ℕ) : ℝ :=
  ((final_volume : ℝ) / initial_volume) ^ (1 / years : ℝ) - 1

/-- The equation representing the average annual growth rate for the given problem -/
theorem sales_growth_equation (x : ℝ) (initial_volume final_volume years : ℕ) :
  initial_volume = 150000 →
  final_volume = 216000 →
  years = 2 →
  x = average_growth_rate initial_volume final_volume years →
  15 * (1 + x)^2 = 21.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_growth_equation_l695_69517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_with_pi_3_asymptote_l695_69570

/-- A hyperbola with center at the origin and axes of symmetry along the coordinate axes -/
structure CenteredHyperbola where
  a : ℝ
  b : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : CenteredHyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a) ^ 2)

/-- The inclination angle of an asymptote of a hyperbola -/
noncomputable def asymptoteAngle (h : CenteredHyperbola) : ℝ :=
  Real.arctan (h.b / h.a)

theorem hyperbola_eccentricity_with_pi_3_asymptote :
  ∀ h : CenteredHyperbola,
    asymptoteAngle h = π / 3 →
    eccentricity h = 2 ∨ eccentricity h = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_with_pi_3_asymptote_l695_69570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_in_second_quadrant_l695_69536

theorem cosine_value_in_second_quadrant (α : Real) 
  (h1 : Real.sin α = 3/5) 
  (h2 : π/2 < α ∧ α < π) : 
  Real.cos α = -4/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_in_second_quadrant_l695_69536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_inequality_l695_69525

theorem cosine_sum_inequality (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ π/2) 
  (hy : 0 ≤ y ∧ y ≤ π/2) 
  (hz : 0 ≤ z ∧ z ≤ π/2) : 
  Real.cos x + Real.cos y + Real.cos z + Real.cos (x + y + z) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_inequality_l695_69525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l695_69502

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1/2)^x - m

-- State the theorem
theorem m_range (m : ℝ) :
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∃ x₂ ∈ Set.Icc (-1 : ℝ) 1, f x₁ = g m x₂) →
  m ∈ Set.Icc (1/2 : ℝ) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l695_69502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_mixture_percentages_l695_69577

-- Define the initial solution parameters
def initial_volume : ℝ := 60
def initial_alcohol_percentage : ℝ := 0.05
def initial_methanol_percentage : ℝ := 0.10
def initial_water_percentage : ℝ := 1 - initial_alcohol_percentage - initial_methanol_percentage

-- Define the added amounts
def added_alcohol : ℝ := 4.5
def added_methanol : ℝ := 6.5
def added_water : ℝ := 3

-- Calculate the initial amounts
def initial_alcohol : ℝ := initial_volume * initial_alcohol_percentage
def initial_methanol : ℝ := initial_volume * initial_methanol_percentage
def initial_water : ℝ := initial_volume * initial_water_percentage

-- Calculate the final amounts
def final_alcohol : ℝ := initial_alcohol + added_alcohol
def final_methanol : ℝ := initial_methanol + added_methanol
def final_water : ℝ := initial_water + added_water

-- Calculate the total final volume
def final_volume : ℝ := final_alcohol + final_methanol + final_water

-- Define a helper function for approximate equality
def approx_equal (x y : ℝ) (ε : ℝ) : Prop := abs (x - y) < ε

-- Define the theorem to prove
theorem final_mixture_percentages :
  approx_equal (final_alcohol / final_volume) 0.1014 0.0001 ∧
  approx_equal (final_methanol / final_volume) 0.1689 0.0001 ∧
  approx_equal (final_water / final_volume) 0.7297 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_mixture_percentages_l695_69577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninetieth_element_is_twenty_l695_69540

/-- Calculates the number of elements in a row given its index -/
def elementsInRow (n : ℕ) : ℕ :=
  if n % 2 = 0 then n * 2 else n + 3

/-- Calculates the value of elements in a row given its index -/
def valueInRow (n : ℕ) : ℕ := 2 * n

/-- Finds the row index containing the kth element -/
def findRow (k : ℕ) : ℕ :=
  let rec aux (acc : ℕ) (row : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then row - 1
    else if acc ≥ k then row - 1
    else aux (acc + elementsInRow row) (row + 1) (fuel - 1)
  aux 0 1 1000  -- Assuming 1000 is a sufficient upper bound for the number of rows

theorem ninetieth_element_is_twenty :
  valueInRow (findRow 90) = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninetieth_element_is_twenty_l695_69540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l695_69549

-- Define the function f(x) = ln x
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Define the point of tangency
def x₀ : ℝ := 2

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), ∀ x y : ℝ,
    (y - f x₀ = m * (x - x₀)) ↔ (x - 2*y + 2*(Real.log 2) - 2 = 0) := by
  -- Introduce m and b
  let m := 1 / 2
  let b := f x₀ - m * x₀
  
  -- Prove existence
  use m, b
  
  -- Prove equivalence
  intro x y
  
  sorry -- Placeholder for the full proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l695_69549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l695_69584

-- Define the function g
noncomputable def g (A B C : ℤ) (x : ℝ) : ℝ := x^2 / (A * x^2 + B * x + C)

-- State the theorem
theorem sum_of_coefficients 
  (A B C : ℤ) 
  (h1 : ∀ x : ℝ, x ≠ -3 ∧ x ≠ 2 → g A B C x ≠ 0) 
  (h2 : ∀ x : ℝ, x ≠ -3 ∧ x ≠ 2 → |g A B C x| < |1 / (A : ℝ)|) 
  (h3 : (1 : ℝ) / A < 1) 
  (h4 : ∀ x : ℝ, x > 3 → g A B C x > 0.5) :
  A + B + C = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l695_69584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_given_positive_test_l695_69563

/-- The probability of having the disease in the population -/
noncomputable def disease_prob : ℝ := 1 / 400

/-- The probability of not having the disease in the population -/
noncomputable def no_disease_prob : ℝ := 1 - disease_prob

/-- The probability of testing positive given that one has the disease -/
noncomputable def positive_given_disease : ℝ := 1

/-- The probability of testing positive given that one does not have the disease (false positive rate) -/
noncomputable def false_positive_rate : ℝ := 0.05

/-- The probability of testing positive -/
noncomputable def positive_test_prob : ℝ := 
  disease_prob * positive_given_disease + no_disease_prob * false_positive_rate

/-- The theorem stating that the probability of having the disease given a positive test is 20/419 -/
theorem disease_given_positive_test : 
  (disease_prob * positive_given_disease) / positive_test_prob = 20 / 419 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_given_positive_test_l695_69563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_polyominoes_on_grid_l695_69565

/-- Represents a "b"-shaped polyomino -/
structure BPolyomino :=
  (position : Nat × Nat)
  (rotation : Nat)

/-- Represents the 8x8 grid -/
def Grid : Type := Fin 8 → Fin 8 → Bool

/-- Checks if a polyomino placement is valid on the grid -/
def isValidPlacement (grid : Grid) (poly : BPolyomino) : Prop :=
  sorry

/-- Checks if polyominoes are equally distributed across rows and columns -/
def isEquallyDistributed (grid : Grid) (polys : List BPolyomino) : Prop :=
  sorry

/-- The maximum number of "b"-shaped polyominoes that can be placed on the grid -/
def maxPolyominoes : Nat := 7

theorem max_polyominoes_on_grid :
  ∀ (grid : Grid) (polys : List BPolyomino),
    (∀ p ∈ polys, isValidPlacement grid p) →
    isEquallyDistributed grid polys →
    polys.length ≤ maxPolyominoes := by
  sorry

#check max_polyominoes_on_grid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_polyominoes_on_grid_l695_69565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_base_l695_69504

/-- Triangle ABC with a point P inside it -/
structure TriangleWithPoint where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ

/-- Properties of the triangle and point P -/
structure TriangleProperties (t : TriangleWithPoint) where
  altitude_to_AB : ℝ
  line_through_P_parallel_to_AB : ℝ
  area_ratio : ℝ

/-- The theorem to be proved -/
theorem distance_to_base
  (t : TriangleWithPoint)
  (props : TriangleProperties t)
  (h1 : props.altitude_to_AB = 6)
  (h2 : props.area_ratio = 1/3) :
  ∃ (d : ℝ), d = 2 ∧ d = props.altitude_to_AB - props.line_through_P_parallel_to_AB :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_base_l695_69504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_general_term_l695_69597

/-- An arithmetic sequence with first term a and common difference b -/
def arithmetic_seq (a b : ℕ) : ℕ → ℕ := λ n ↦ a + b * (n - 1)

/-- A geometric sequence with first term b and common ratio a -/
def geometric_seq (a b : ℕ) : ℕ → ℕ := λ n ↦ b * a^(n - 1)

theorem arithmetic_seq_general_term (a b : ℕ) :
  (a > 1) →
  (b > 1) →
  (arithmetic_seq a b 1 < geometric_seq a b 1) →
  (geometric_seq a b 2 < arithmetic_seq a b 3) →
  (∀ n : ℕ, n > 0 → ∃ m : ℕ, m > 0 ∧ arithmetic_seq a b m + 3 = geometric_seq a b n) →
  (∀ n : ℕ, arithmetic_seq a b n = 5 * n - 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_general_term_l695_69597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l695_69500

theorem negation_of_proposition (x : ℝ) :
  (¬(x^2 - 1 = 0 → x = -1 ∨ x = 1)) ↔ (x^2 - 1 = 0 → x ≠ -1 ∧ x ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l695_69500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_theta_l695_69512

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7*θ) = -953/1024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_theta_l695_69512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_at_pi_over_two_l695_69535

noncomputable def f (x : ℝ) : ℝ := (2 - Real.cos x) / Real.sin x

theorem tangent_perpendicular_at_pi_over_two (a : ℝ) :
  (∃ (m : ℝ), deriv f (π / 2) = m ∧ m * (-1 / a) = -1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_at_pi_over_two_l695_69535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_count_range_l695_69568

/-- Represents a permutation of 10 volumes --/
def Arrangement := Fin 10 → Fin 10

/-- Counts the number of inversions in an arrangement --/
def inversionCount (arr : Arrangement) : ℕ :=
  Finset.sum (Finset.range 10) (λ i =>
    Finset.sum (Finset.range 10) (λ j =>
      if i < j ∧ arr i > arr j then 1 else 0))

/-- Theorem: The number of inversions in any arrangement of 10 volumes
    can be any integer from 0 to 45, inclusive --/
theorem inversion_count_range :
  ∀ n : ℕ, n ≤ 45 → ∃ arr : Arrangement, inversionCount arr = n := by
  sorry

#eval (Finset.range 10).sum (λ _ => (Finset.range 10).sum (λ _ => 1))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_count_range_l695_69568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_a_part_b_l695_69522

open BigOperators

/-- The transformed sequence -/
def transformed_sequence (x : Fin n → ℝ) : Fin n → ℝ :=
  λ i => sorry

/-- Part (a) of the theorem -/
theorem part_a (x : Fin n → ℝ) (t : ℝ) (ht : t > 0) :
    (Finset.filter (λ i => transformed_sequence x i > t) Finset.univ).card ≤
    (2 / t) * ∑ i, x i := by
  sorry

/-- Part (b) of the theorem -/
theorem part_b (x : Fin n → ℝ) :
    ∑ i, transformed_sequence x i ≤ 32 * n * Real.sqrt ((∑ i, x i ^ 2) / (32 * n)) := by
  sorry

#check part_a
#check part_b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_a_part_b_l695_69522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_a_11_equals_6_l695_69521

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x * (1 - x) else x * (1 + x)

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1/2
  | n + 1 => 1 / (1 - sequence_a n)

theorem f_a_11_equals_6 (h_odd : ∀ x, f (-x) = -f x) : f (sequence_a 11) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_a_11_equals_6_l695_69521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_intercept_triangle_l695_69586

-- Define the curve function
noncomputable def f (x : ℝ) : ℝ := (x - 2)^2 * (x + 3)

-- Define the x-intercepts
noncomputable def x_intercepts : Set ℝ := {x : ℝ | f x = 0}

-- Define the y-intercept
noncomputable def y_intercept : ℝ := f 0

-- Define the triangle area function
noncomputable def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

-- Theorem statement
theorem area_of_intercept_triangle : 
  ∃ (x₁ x₂ : ℝ), x₁ ∈ x_intercepts ∧ x₂ ∈ x_intercepts ∧ x₁ ≠ x₂ ∧
  triangle_area (x₂ - x₁) y_intercept = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_intercept_triangle_l695_69586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_huabeisai_l695_69588

def complement_digit (d : ℕ) : ℕ := 9 - d

def transform_number (n : ℕ) : ℕ :=
  let digits := n.digits 10
  let transformed_digits := List.zipWith 
    (λ (i : ℕ) (d : ℕ) => if i % 2 = 0 then d else complement_digit d) 
    (List.range digits.length) 
    digits
  transformed_digits.foldl (λ acc d => acc * 10 + d) 0

theorem transform_huabeisai : transform_number 244041993088 = 254948903981 := by
  sorry

#eval transform_number 244041993088

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_huabeisai_l695_69588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_7pi_over_6_l695_69511

theorem sin_alpha_plus_7pi_over_6 (α : ℝ) 
  (h : Real.cos (α - π/6) + Real.sin α = (4/5) * Real.sqrt 3) : 
  Real.sin (α + 7*π/6) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_7pi_over_6_l695_69511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_ratio_l695_69508

theorem orange_ratio (total : ℕ) (ripe : ℕ) (unripe : ℕ) (eaten_ripe : ℚ) (eaten_unripe : ℚ) (uneaten : ℕ) :
  total = 96 →
  ripe + unripe = total →
  eaten_ripe = 1/4 →
  eaten_unripe = 1/8 →
  uneaten = 78 →
  (1 - eaten_ripe) * (ripe : ℚ) + (1 - eaten_unripe) * (unripe : ℚ) = uneaten →
  (ripe : ℚ) / total = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_ratio_l695_69508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_selection_l695_69544

def systematic_sample (population_size : ℕ) (sample_size : ℕ) (first_sample : ℕ) : Prop :=
  ∃ (interval : ℕ), 
    interval = population_size / sample_size ∧
    ∀ (n : ℕ), n ∈ Set.range (λ i ↦ first_sample + i * interval) → n ≤ population_size

theorem sample_selection (population_size : ℕ) (sample_size : ℕ) (first_sample : ℕ) 
    (h1 : population_size = 1000) 
    (h2 : sample_size = 20) 
    (h3 : first_sample = 15) :
  systematic_sample population_size sample_size first_sample →
  65 ∈ Set.range (λ i ↦ first_sample + i * (population_size / sample_size)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_selection_l695_69544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_quadrilateral_pyramid_l695_69581

/-- The radius of a sphere inscribed around a regular quadrilateral pyramid -/
theorem inscribed_sphere_radius_quadrilateral_pyramid 
  (b : ℝ) (α : ℝ) (h_b_pos : 0 < b) (h_α_acute : 0 < α ∧ α < π / 2) :
  ∃ R : ℝ, R = b / (2 * Real.sqrt (Real.cos α)) ∧ 
  R > 0 ∧ 
  R = (b / Real.sqrt (2 * Real.cos α)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_quadrilateral_pyramid_l695_69581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_root_product_l695_69566

def g (x : ℝ) : ℝ := x^4 + 20*x^3 + 98*x^2 + 100*x + 25

def roots (g : ℝ → ℝ) : Set ℝ := {x | g x = 0}

theorem min_root_product (w₁ w₂ w₃ w₄ : ℝ) 
  (h_roots : roots g = {w₁, w₂, w₃, w₄}) : 
  (∀ (σ : Equiv.Perm (Fin 4)), 
    |w₁ * w₂ + w₃ * w₄| ≤ |w₁ * w₂ + w₃ * w₄|) ∧ 
  (∃ (σ : Equiv.Perm (Fin 4)), |w₁ * w₂ + w₃ * w₄| = 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_root_product_l695_69566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_l695_69548

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x - Real.sqrt x - 2

-- State the theorem
theorem f_has_one_zero : ∃! x : ℝ, f x = 0 ∧ x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_l695_69548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_objects_dont_meet_l695_69555

structure Object where
  x : ℤ
  y : ℤ

def moves_right_or_up (a : Object) : Prop :=
  a.x ≥ 0 ∧ a.y ≥ 0

def moves_left_or_down (b : Object) : Prop :=
  b.x ≤ 0 ∧ b.y ≤ 0

def four_steps (initial final : Object) : Prop :=
  (final.x - initial.x).natAbs + (final.y - initial.y).natAbs = 4

def meet (a b : Object) : Prop :=
  a.x = b.x ∧ a.y = b.y

def object_diff (a b : Object) : Object :=
  ⟨a.x - b.x, a.y - b.y⟩

theorem objects_dont_meet :
  ∀ (a_final b_final : Object),
    let a_initial : Object := ⟨0, 0⟩
    let b_initial : Object := ⟨6, 8⟩
    moves_right_or_up (object_diff a_final a_initial) →
    moves_left_or_down (object_diff b_final b_initial) →
    four_steps a_initial a_final →
    four_steps b_initial b_final →
    ¬(meet a_final b_final) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_objects_dont_meet_l695_69555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l695_69551

/-- The line l is defined by the equation kx - y - 2 - k = 0, where k is a real number -/
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y - 2 - k = 0

/-- The fixed point that the line passes through -/
noncomputable def fixed_point : ℝ × ℝ := (1, -2)

/-- The area of triangle AOB formed by the line's intersections with the axes -/
noncomputable def triangle_area (k : ℝ) : ℝ := (2 + k)^2 / (2 * k)

theorem line_l_properties :
  ∀ k : ℝ,
  (∀ x y : ℝ, line_l k x y → (x, y) = fixed_point) ∧
  (k ≥ 0 ↔ ¬ ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ line_l k x y) ∧
  (∃ k_min : ℝ, k_min > 0 ∧
    (∀ k : ℝ, k > 0 → triangle_area k ≥ triangle_area k_min) ∧
    triangle_area k_min = 4 ∧
    (∀ x y : ℝ, line_l k_min x y ↔ 2 * x - y - 4 = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l695_69551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2alpha_value_l695_69576

theorem tan_2alpha_value (α : ℝ) 
  (h1 : Real.sin α - Real.cos α = 4/3)
  (h2 : α ∈ Set.Icc (π/2) (3*π/4)) :
  Real.tan (2*α) = (7*Real.sqrt 2)/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2alpha_value_l695_69576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_is_perfect_square_l695_69515

/-- The polynomial p(x) = x^4 + x^3 + 2x^2 + (7/8)x + 49/64 is a perfect square. -/
theorem polynomial_is_perfect_square :
  ∃ (q : ℝ → ℝ), (λ x : ℝ ↦ x^4 + x^3 + 2*x^2 + (7/8)*x + 49/64) = (λ x ↦ q x * q x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_is_perfect_square_l695_69515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l695_69594

theorem complex_equation_solution (z : ℂ) (h : z + Complex.I - 3 = 3 - Complex.I) : z = 6 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l695_69594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_l695_69547

def s : ℕ → ℚ
  | 0 => 2  -- Add this case to handle Nat.zero
  | 1 => 2
  | n + 1 =>
    if n % 4 = 0 then 2 + s (n / 4)
    else if n % 2 = 1 then 1 / s n
    else s n + 1

theorem sequence_value (n : ℕ) : s n = 5 / 36 → n = 129 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_l695_69547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_ranking_sequences_l695_69592

/-- Represents a team in the tournament -/
inductive Team
  | A | B | C | D | E | F

/-- Represents a match between two teams -/
structure Match where
  team1 : Team
  team2 : Team

/-- Represents the tournament structure -/
structure Tournament where
  saturday_matches : List Match
  no_ties : Bool

/-- Represents a ranking of teams -/
def Ranking := List Team

/-- The number of teams in the tournament -/
def num_teams : Nat := 6

/-- The number of Saturday matches -/
def num_saturday_matches : Nat := 3

/-- Calculate the number of possible ranking sequences -/
def count_ranking_sequences (t : Tournament) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem tournament_ranking_sequences (t : Tournament) 
  (h1 : t.saturday_matches.length = num_saturday_matches)
  (h2 : t.no_ties = true) :
  count_ranking_sequences t = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_ranking_sequences_l695_69592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_organic_vegetables_profit_l695_69578

/-- Represents the sales data for organic vegetables -/
structure SalesData where
  cost_price : ℕ
  full_price : ℕ
  discount_price : ℕ
  total_days : ℕ
  x : ℕ
  y : ℕ

/-- Calculates the expected profit for a given number of portions -/
def expected_profit (data : SalesData) (portions : ℕ) : ℚ :=
  sorry

/-- The main theorem stating the conditions for x -/
theorem organic_vegetables_profit (data : SalesData) :
  data.cost_price = 10 ∧
  data.full_price = 15 ∧
  data.discount_price = 5 ∧
  data.total_days = 100 ∧
  data.x + data.y = 30 ∧
  data.x > 0 ∧
  data.y > 0 →
  (∀ x : ℕ, x ∈ ({26, 27, 28, 29} : Set ℕ) ↔
    (expected_profit data 17 > expected_profit data 18 ∧
     data.x = x)) :=
by
  sorry

#check organic_vegetables_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_organic_vegetables_profit_l695_69578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l695_69596

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2*x)

-- State the theorem
theorem f_monotone_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ < f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l695_69596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_x_cubed_l695_69531

theorem binomial_coefficient_x_cubed (a : ℝ) : 
  (Finset.range 6).sum (λ k ↦ (Nat.choose 5 k) * a^k * (1 : ℝ)^(5-k)) = 1 + 5*a + 10*a^2 + 80*a^3 + 5*a^4 + a^5 → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_x_cubed_l695_69531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_unfair_coin_probability_l695_69599

/-- Probability of heads for the unfair coin -/
noncomputable def p_heads : ℝ := 3/4

/-- Number of coin tosses -/
def n_tosses : ℕ := 40

/-- 
Probability that the total number of heads plus the number of tails divisible by 3 is even 
after n tosses of an unfair coin with probability p of heads
-/
noncomputable def P (n : ℕ) (p : ℝ) : ℝ :=
  1/2 * (1 + (1/2)^n)

/-- 
Theorem stating that the probability of the total number of heads plus 
the number of tails divisible by 3 being even after 40 tosses of the unfair coin 
is equal to 1/2 * (1 + 1/2^40)
-/
theorem unfair_coin_probability : 
  P n_tosses p_heads = 1/2 * (1 + 1/2^40) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_unfair_coin_probability_l695_69599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_in_direction_of_b_l695_69513

noncomputable def a : ℝ × ℝ := (-7 * Real.sqrt 2 / 2, Real.sqrt 2 / 2)
noncomputable def b : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection (v w : ℝ × ℝ) : ℝ := dot_product v w / magnitude w

theorem projection_of_a_in_direction_of_b :
  projection a b = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_in_direction_of_b_l695_69513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_symmetry_l695_69583

theorem function_value_symmetry (a b : ℝ) :
  let f (x : ℝ) := a * x^5 + b / x^3 + 2
  f 2023 = 16 → f (-2023) = -12 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_symmetry_l695_69583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_profit_calculation_l695_69559

/-- ≈ denotes approximate equality -/
def approx_equal (a b : ℝ) : Prop := abs (a - b) < 1e-6

notation a " ≈ " b => approx_equal a b

theorem retailer_profit_calculation (cost : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) (actual_profit_percentage : ℝ) :
  markup_percentage = 40 →
  discount_percentage = 25 →
  approx_equal actual_profit_percentage 5 →
  let marked_price := cost * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  let actual_profit := selling_price - cost
  approx_equal (actual_profit / cost * 100) actual_profit_percentage →
  (marked_price - cost) / cost * 100 = 40 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_profit_calculation_l695_69559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_of_specific_quadrilateral_l695_69527

/-- Represents a quadrilateral with one diagonal and its offsets -/
structure Quadrilateral where
  diagonal : ℝ
  offset1 : ℝ
  offset2 : ℝ

/-- Calculates the area of a quadrilateral given its diagonal and offsets -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  (1/2) * q.diagonal * q.offset1 + (1/2) * q.diagonal * q.offset2

/-- Theorem stating that a quadrilateral with offsets 9 and 6, and area 150, has a diagonal of length 20 -/
theorem diagonal_length_of_specific_quadrilateral :
  ∃ (q : Quadrilateral), q.offset1 = 9 ∧ q.offset2 = 6 ∧ area q = 150 ∧ q.diagonal = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_of_specific_quadrilateral_l695_69527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_interval_of_f_l695_69579

open Real

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * sin (ω * x + π / 4)

noncomputable def g (x : ℝ) : ℝ := 2 * cos (2 * x - π / 4)

theorem increase_interval_of_f (ω : ℝ) (h_ω : ω > 0) :
  (∃ k : ℤ, ∀ x : ℝ, f ω x = f ω (k * π + π / 4 - x)) →
  (∃ k : ℤ, ∀ x : ℝ, g x = g (k * π + π / 4 - x)) →
  ∃ a b : ℝ, a = 0 ∧ b = π / 8 ∧
    (∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f ω x < f ω y) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 →
      (x < a → ∃ y, a < y ∧ y ≤ π / 2 ∧ f ω x > f ω y) ∧
      (b < x → ∃ y, 0 ≤ y ∧ y < x ∧ f ω x < f ω y)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_interval_of_f_l695_69579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_l695_69585

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (2 * a + b / c) = a * Real.sqrt (b / c)) ↔ (c = b * (a^2 - 1) / (2 * a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_l695_69585
