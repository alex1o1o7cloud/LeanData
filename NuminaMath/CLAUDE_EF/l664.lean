import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_e_pow_k_eq_four_l664_66440

/-- The number of 1s in the base 2 representation of n -/
def f (n : ℕ+) : ℕ := sorry

/-- The sum of f(n) / (n + n^2) over all positive integers -/
noncomputable def k : ℝ := ∑' n : ℕ+, (f n : ℝ) / (n + n^2)

/-- e^k is rational and equal to 4 -/
theorem e_pow_k_eq_four : Real.exp k = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_e_pow_k_eq_four_l664_66440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_microphotonics_percentage_l664_66449

noncomputable def budget_circle : ℝ := 360

noncomputable def home_electronics : ℝ := 24
noncomputable def food_additives : ℝ := 15
noncomputable def genetically_modified_microorganisms : ℝ := 29
noncomputable def industrial_lubricants : ℝ := 8
noncomputable def basic_astrophysics_degrees : ℝ := 39.6

noncomputable def basic_astrophysics_percentage : ℝ := (basic_astrophysics_degrees / budget_circle) * 100

noncomputable def total_known_percentage : ℝ := 
  home_electronics + food_additives + genetically_modified_microorganisms + 
  industrial_lubricants + basic_astrophysics_percentage

theorem microphotonics_percentage : 
  100 - total_known_percentage = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_microphotonics_percentage_l664_66449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_B_more_profitable_l664_66466

-- Define the probability of a machine breaking down in workshop A
def prob_A : ℚ := 2/5

-- Define the probabilities of machines breaking down in workshop B
def prob_B1 : ℚ := 1/5
def prob_B2 : ℚ := 1/5
def prob_B3 : ℚ := 3/5

-- Define profit values based on number of machines breaking down
def profit_0 : ℚ := 20000
def profit_1 : ℚ := 10000
def profit_2 : ℚ := 0
def profit_3 : ℚ := -30000

-- Function to calculate binomial probability
def binomial_prob (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

-- Expected profit for workshop A
noncomputable def expected_profit_A : ℚ :=
  profit_0 * binomial_prob 3 0 prob_A +
  profit_1 * binomial_prob 3 1 prob_A +
  profit_2 * binomial_prob 3 2 prob_A +
  profit_3 * binomial_prob 3 3 prob_A

-- Expected profit for workshop B
def expected_profit_B : ℚ :=
  profit_0 * ((1 - prob_B1) * (1 - prob_B2) * (1 - prob_B3)) +
  profit_1 * (prob_B1 * (1 - prob_B2) * (1 - prob_B3) +
              (1 - prob_B1) * prob_B2 * (1 - prob_B3) +
              (1 - prob_B1) * (1 - prob_B2) * prob_B3) +
  profit_2 * (prob_B1 * prob_B2 * (1 - prob_B3) +
              prob_B1 * (1 - prob_B2) * prob_B3 +
              (1 - prob_B1) * prob_B2 * prob_B3) +
  profit_3 * (prob_B1 * prob_B2 * prob_B3)

-- Theorem stating that expected profit of workshop B is greater than workshop A
theorem workshop_B_more_profitable : expected_profit_B > expected_profit_A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_B_more_profitable_l664_66466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_usual_time_l664_66488

/-- Represents the usual time (in minutes) taken by a man to cover a certain distance -/
def usual_time : ℝ → ℝ := fun t ↦ t

/-- Given that a man takes 24 minutes more when walking at 70% of his usual speed,
    prove that his usual time to cover the distance is 56 minutes -/
theorem mans_usual_time (t : ℝ) :
  (usual_time t = t) →
  (0.7 * t = t + 24) →
  t = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_usual_time_l664_66488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l664_66432

open Real

-- Part 1
theorem part_one (α : ℝ) (h : Real.tan α = 1/3) :
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 2/3 := by sorry

-- Part 2
theorem part_two (α : ℝ) :
  (Real.tan (π - α) * Real.cos (2*π - α) * Real.sin (-α + 3*π/2)) / 
  (Real.cos (-α - π) * Real.sin (-π - α)) = Real.cos α := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l664_66432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goat_accessible_area_l664_66428

open Real

/-- The area accessible to a point tied to a corner of a rectangle --/
noncomputable def accessible_area (barn_length barn_width leash_length : ℝ) : ℝ :=
  let main_arc_area := (3/4) * π * leash_length^2
  let extra_sector_radius := leash_length - barn_width
  let extra_sector_area := 2 * ((1/4) * π * extra_sector_radius^2)
  main_arc_area + extra_sector_area

/-- Theorem stating the accessible area for given dimensions --/
theorem goat_accessible_area :
  accessible_area 5 4 6 = 27.5 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goat_accessible_area_l664_66428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_some_number_value_l664_66425

/-- Given that (some_number * 12) / 100 = 0.038903999999999994, 
    prove that some_number ≈ 0.3242 -/
theorem some_number_value (some_number : ℝ) 
  (h : (some_number * 12) / 100 = 0.038903999999999994) : 
  ‖some_number - 0.3242‖ < 1e-4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_some_number_value_l664_66425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_difference_magnitude_l664_66463

theorem complex_difference_magnitude (z₁ z₂ : ℂ) :
  Complex.abs z₁ = 2 →
  Complex.abs z₂ = 2 →
  z₁ + z₂ = Complex.ofReal (Real.sqrt 3) + Complex.I →
  Complex.abs (z₁ - z₂) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_difference_magnitude_l664_66463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_decomposition_theorem_l664_66414

structure RectangularParallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : a > 0
  positive_b : b > 0
  positive_c : c > 0

noncomputable def circumsphereVolume (p : RectangularParallelepiped) : ℝ :=
  (Real.pi / 6) * (p.a^2 + p.b^2 + p.c^2)^(3/2)

def isCube (p : RectangularParallelepiped) : Prop :=
  p.a = p.b ∧ p.b = p.c

theorem cube_decomposition_theorem (s : ℝ) (parallelepipeds : List RectangularParallelepiped)
    (h_positive_s : s > 0)
    (h_decomposition : s^3 = (parallelepipeds.map (fun p => p.a * p.b * p.c)).sum)
    (h_circumsphere_volume : 
      (Real.pi * Real.sqrt 3 * s^3) / 2 = (parallelepipeds.map circumsphereVolume).sum) :
    ∀ p ∈ parallelepipeds, isCube p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_decomposition_theorem_l664_66414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_nonnegative_implies_a_range_l664_66450

-- Define the constant e
noncomputable def e : ℝ := Real.exp 1

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (Real.exp x) + x^2 - x

-- Define the function g
noncomputable def g (a x : ℝ) : ℝ := -a * Real.log (f x - x^2 + x) - 1/x - Real.log x - a + 1

-- Theorem statement
theorem g_nonnegative_implies_a_range (a : ℝ) :
  (∀ x ≥ 1, g a x ≥ 0) → a ∈ Set.Ici 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_nonnegative_implies_a_range_l664_66450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compute_α_l664_66439

variable (α γ : ℂ)

axiom α_plus_γ_positive : (α + γ).re > 0
axiom i_α_minus_3γ_positive : (Complex.I * (α - 3 * γ)).re > 0
axiom γ_value : γ = 4 + 3 * Complex.I

theorem compute_α : α = 12 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compute_α_l664_66439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ac_length_l664_66451

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define a function to check if two lines are perpendicular
def perpendicular (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  (p2.1 - p1.1) * (p4.1 - p3.1) + (p2.2 - p1.2) * (p4.2 - p3.2) = 0

-- Define the theorem
theorem triangle_ac_length (ABC : Triangle) 
  (D F : ℝ × ℝ) 
  (h1 : D.1 = ABC.A.1 ∧ D.2 ≥ ABC.A.2 ∧ D.2 ≤ ABC.C.2) -- D is on AC
  (h2 : F.1 = ABC.B.1 ∧ F.2 ≥ ABC.B.2 ∧ F.2 ≤ ABC.C.2) -- F is on BC
  (h3 : perpendicular ABC.A ABC.B ABC.A ABC.C) -- AB ⟂ AC
  (h4 : perpendicular ABC.A F ABC.B ABC.C) -- AF ⟂ BC
  (h5 : distance ABC.B D = distance D ABC.C) -- BD = DC
  (h6 : distance F ABC.C = 2 * distance ABC.B D) -- FC is twice BD
  (h7 : distance F ABC.C = Real.sqrt 3) -- length of FC is √3
  : distance ABC.A ABC.C = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ac_length_l664_66451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_construction_possible_l664_66480

/-- A bounded region in a plane -/
structure BoundedRegion where
  -- Placeholder for region definition
  dummy : Unit

/-- A line in a plane -/
structure Line where
  -- Placeholder for line definition
  dummy : Unit

/-- An angle formed by two lines -/
structure Angle where
  line1 : Line
  line2 : Line

/-- A point in a plane -/
structure Point where
  -- Placeholder for point definition
  dummy : Unit

/-- Represents a compass and straightedge construction -/
inductive CompassStraightedgeConstruction where
  | dummy : CompassStraightedgeConstruction

/-- Predicate to check if a point is inside a bounded region -/
def isInside (p : Point) (r : BoundedRegion) : Prop :=
  sorry

/-- Predicate to check if a line segment is an angle bisector -/
def isAngleBisector (seg : Line) (ang : Angle) : Prop :=
  sorry

/-- Predicate to check if a point is on a line -/
def isOnLine (p : Point) (l : Line) : Prop :=
  sorry

/-- Theorem stating that the angle bisector segment can be constructed -/
theorem angle_bisector_construction_possible 
  (r : BoundedRegion) (ang : Angle) 
  (h1 : ∃ p : Point, isOnLine p ang.line1 ∧ isOnLine p ang.line2 ∧ ¬isInside p r) : 
  ∃ (construction : CompassStraightedgeConstruction) (bisector : Line),
    (∀ p : Point, isInside p r → isOnLine p bisector) ∧ 
    isAngleBisector bisector ang :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_construction_possible_l664_66480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_plus_x_l664_66452

theorem cos_pi_half_plus_x (x : ℝ) 
  (h1 : Real.sin (x + π/6) = 3/5) 
  (h2 : π/3 < x) 
  (h3 : x < 5*π/6) : 
  Real.cos (π/2 + x) = -(4 + 3 * Real.sqrt 3) / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_plus_x_l664_66452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_congruence_l664_66476

theorem smallest_n_congruence (n : ℕ) : 
  (∀ k : ℕ, 0 < k ∧ k < n → (5 : ℤ) ^ k % 7 ≠ k ^ 5 % 7) ∧ 
  (5 : ℤ) ^ n % 7 = n ^ 5 % 7 → 
  n = 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_congruence_l664_66476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_l664_66492

-- Define the arithmetic sequence
noncomputable def arithmetic_seq (a d : ℝ) : ℕ → ℝ := λ n => a + (n - 1) * d

-- Define the geometric sequence b_n
noncomputable def b (n : ℕ) : ℝ := (5/4) * 2^(n-1)

-- Define the sum of the first n terms of b_n
noncomputable def S (n : ℕ) : ℝ := (5/4) * (2^n - 1)

-- Theorem statement
theorem arithmetic_geometric_sequence :
  ∃ (a d : ℝ),
    (∀ i : ℕ, i ∈ ({1, 2, 3} : Set ℕ) → arithmetic_seq a d i > 0) ∧
    (arithmetic_seq a d 1 + arithmetic_seq a d 2 + arithmetic_seq a d 3 = 15) ∧
    (arithmetic_seq a d 1 + 2 = b 3) ∧
    (arithmetic_seq a d 2 + 5 = b 4) ∧
    (arithmetic_seq a d 3 + 13 = b 5) ∧
    (∀ n : ℕ, n ≥ 1 → b n = (5/4) * 2^(n-1)) ∧
    (∀ n : ℕ, n ≥ 1 → S n + 5/4 = (5/2) * 2^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_l664_66492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l664_66464

open Real

theorem min_value_trig_expression (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (sin x + 1 / sin x)^2 + (cos x + 1 / cos x)^2 ≥ 9 ∧
  ((sin x + 1 / sin x)^2 + (cos x + 1 / cos x)^2 = 9 ↔ x = π / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l664_66464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_C_to_C_l664_66499

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a 2x2 matrix -/
structure Matrix2x2 where
  a11 : ℝ
  a12 : ℝ
  a21 : ℝ
  a22 : ℝ

/-- The original curve C -/
def C (p : Point) : Prop :=
  p.y^2 = 2 * p.x

/-- The reflection matrix about x-axis -/
def A : Matrix2x2 :=
  { a11 := 1, a12 := 0, a21 := 0, a22 := -1 }

/-- The 90° counterclockwise rotation matrix -/
def B : Matrix2x2 :=
  { a11 := 0, a12 := 1, a21 := -1, a22 := 0 }

/-- The combined transformation matrix -/
def M : Matrix2x2 :=
  { a11 := B.a11 * A.a11 + B.a12 * A.a21,
    a12 := B.a11 * A.a12 + B.a12 * A.a22,
    a21 := B.a21 * A.a11 + B.a22 * A.a21,
    a22 := B.a21 * A.a12 + B.a22 * A.a22 }

/-- The transformation function -/
def transform (p : Point) : Point :=
  { x := M.a11 * p.x + M.a12 * p.y,
    y := M.a21 * p.x + M.a22 * p.y }

/-- The transformed curve C' -/
def C' (p : Point) : Prop :=
  p.y = -1/2 * p.x^2

/-- The main theorem to prove -/
theorem transform_C_to_C' :
  ∀ p : Point, C p → C' (transform p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_C_to_C_l664_66499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_duration_proof_flight_hours_l664_66471

/-- The length of Emily's flight from New York to Hawaii -/
def flight_length (tv_episodes : ℕ) (tv_episode_length : ℕ) (sleep_time : ℕ) 
  (movie_count : ℕ) (movie_length : ℕ) : ℕ :=
  tv_episodes * tv_episode_length + sleep_time + movie_count * movie_length

theorem flight_duration_proof :
  flight_length 3 25 270 2 105 + 45 = 600 :=
by
  -- Unfold the definition of flight_length
  unfold flight_length
  -- Perform the arithmetic
  norm_num

theorem flight_hours :
  600 / 60 = 10 :=
by
  norm_num

#eval flight_length 3 25 270 2 105 + 45
#eval 600 / 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_duration_proof_flight_hours_l664_66471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_real_l664_66437

theorem complex_fraction_real (a : ℝ) : 
  ((-a + Complex.I) / (1 - Complex.I)).re = ((-a + Complex.I) / (1 - Complex.I)).im → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_real_l664_66437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_15_terms_l664_66486

def sequence_term (n : ℕ) : ℤ := 2 * n - 7

def sum_of_absolute_values (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => Int.natAbs (sequence_term (i + 1)))

theorem sum_of_first_15_terms :
  sum_of_absolute_values 15 = 153 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_15_terms_l664_66486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_max_omega_for_increasing_f_l664_66478

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  4 * Real.cos (ω * x - Real.pi / 6) * Real.sin (ω * x) - Real.cos (2 * ω * x + Real.pi)

-- Theorem for the range of f
theorem range_of_f (ω : ℝ) (h : ω > 0) :
  Set.range (f ω) = Set.Icc (1 - Real.sqrt 3) (1 + Real.sqrt 3) := by sorry

-- Theorem for the maximum value of ω
theorem max_omega_for_increasing_f :
  (∃ (ω : ℝ), ω > 0 ∧ 
   (∀ x y : ℝ, -3*Real.pi/2 ≤ x ∧ x < y ∧ y ≤ Real.pi/2 → f ω x < f ω y) ∧
   (∀ ω' : ℝ, ω' > ω → 
     ∃ x y : ℝ, -3*Real.pi/2 ≤ x ∧ x < y ∧ y ≤ Real.pi/2 ∧ f ω' x ≥ f ω' y)) ∧
  (∀ ω : ℝ, (∀ x y : ℝ, -3*Real.pi/2 ≤ x ∧ x < y ∧ y ≤ Real.pi/2 → f ω x < f ω y) → ω ≤ 1/6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_max_omega_for_increasing_f_l664_66478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_f_differentiable_f_condition_inequality_theorem_l664_66473

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the properties of f
theorem f_nonnegative : ∀ x > 0, f x ≥ 0 := sorry
theorem f_differentiable : Differentiable ℝ f := sorry
theorem f_condition : ∀ x > 0, x * f x + f x ≤ 0 := sorry

-- State the theorem
theorem inequality_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a < b) :
  b * f a ≤ a * f b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_f_differentiable_f_condition_inequality_theorem_l664_66473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_of_octagon_is_45_l664_66405

/-- The degree of each exterior angle of a regular octagon -/
noncomputable def exterior_angle_of_octagon : ℚ :=
  360 / 8

/-- Theorem: The degree of each exterior angle of a regular octagon is 45° -/
theorem exterior_angle_of_octagon_is_45 :
  exterior_angle_of_octagon = 45 := by
  -- Unfold the definition of exterior_angle_of_octagon
  unfold exterior_angle_of_octagon
  -- Simplify the fraction
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_of_octagon_is_45_l664_66405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_time_problem_l664_66497

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Converts time to total minutes -/
noncomputable def timeToMinutes (t : Time) : ℝ :=
  t.hours * 60 + t.minutes + t.seconds / 60

/-- The rate at which the watch loses time -/
noncomputable def watchRate : ℝ := 58.2 / 60

/-- The problem statement -/
theorem watch_time_problem (actualTime : Time) :
  -- The watch is set correctly at noon
  let startTime : Time := ⟨12, 0, 0⟩
  -- At 1:00 PM actual time, the watch reads 12:58:12
  let oneHourLater : Time := ⟨13, 0, 0⟩
  let watchTimeOneHourLater : Time := ⟨12, 58, 12⟩
  -- The watch loses time at a constant rate
  watchRate = (timeToMinutes watchTimeOneHourLater - timeToMinutes startTime) / 
              (timeToMinutes oneHourLater - timeToMinutes startTime) →
  -- When the watch reads 8:00 PM
  let watchTime : Time := ⟨20, 0, 0⟩
  -- The actual time is 8:14:51 PM
  actualTime = ⟨20, 14, 51⟩ ∧ 
  timeToMinutes actualTime = (timeToMinutes watchTime - timeToMinutes startTime) / watchRate + 
                             timeToMinutes startTime := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_time_problem_l664_66497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_of_parabola_l664_66444

/-- A parabola defined by y = -x^2 + cx + d, where c and d are real numbers. -/
structure Parabola where
  c : ℝ
  d : ℝ

/-- The parabola satisfies y ≤ 0 for x in [-5, 2] -/
def satisfies_inequality (p : Parabola) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc (-5) 2 → -x^2 + p.c * x + p.d ≤ 0

/-- The vertex of a parabola -/
noncomputable def vertex (p : Parabola) : ℝ × ℝ :=
  (p.c / 2, -((p.c / 2)^2) + p.c * (p.c / 2) + p.d)

/-- Theorem: The vertex of the parabola satisfying the given condition is (3/2, 13/4) -/
theorem vertex_of_parabola (p : Parabola) (h : satisfies_inequality p) : 
  vertex p = (3/2, 13/4) := by
  sorry

#check vertex_of_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_of_parabola_l664_66444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_has_property_P_infinitely_many_composite_with_property_P_l664_66446

/-- Property P for a positive integer n -/
def has_property_P (n : ℕ) : Prop :=
  ∀ a : ℕ, a > 0 → n ∣ a^n - 1 → n^2 ∣ a^n - 1

theorem prime_has_property_P (p : ℕ) (hp : Nat.Prime p) :
  has_property_P p :=
sorry

theorem infinitely_many_composite_with_property_P :
  ∃ f : ℕ → ℕ, 
    (∀ n, ¬Nat.Prime (f n)) ∧ 
    (∀ n, has_property_P (f n)) ∧
    (∀ m n, m ≠ n → f m ≠ f n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_has_property_P_infinitely_many_composite_with_property_P_l664_66446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_and_b_l664_66498

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + (a^2 - 1) * x + b

theorem tangent_line_implies_a_and_b :
  ∀ a b : ℝ,
  (∀ x : ℝ, (deriv (f a b)) 1 = -1) →
  (f a b 1 = 2) →
  (a = 1 ∧ b = 8/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_and_b_l664_66498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_and_b_formulas_smallest_m_l664_66457

def a : ℕ → ℚ
| 0 => 1
| n + 1 => 1 - 1 / (4 * a n)

def b (n : ℕ) : ℚ := 2 / (2 * a n - 1)

def c (n : ℕ) : ℚ := 4 * a n / (n + 1)

def T (n : ℕ) : ℚ := (Finset.range n).sum (λ i => c i * c (i + 2))

theorem a_and_b_formulas :
  (∀ n : ℕ, n > 0 → b n = 2 * n) ∧
  (∀ n : ℕ, n > 0 → a n = (n + 1) / (2 * n)) :=
sorry

theorem smallest_m :
  ∃ m : ℕ, m > 0 ∧
    (∀ n : ℕ, n > 0 → T n = 1 / (c m * c (m + 1))) ∧
    (∀ k : ℕ, 0 < k ∧ k < m → ∃ n : ℕ, n > 0 ∧ T n ≠ 1 / (c k * c (k + 1))) ∧
    m = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_and_b_formulas_smallest_m_l664_66457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_yummy_integer_l664_66454

-- Define what it means for an integer to be yummy
def is_yummy (a : ℤ) : Prop :=
  ∃ (n : ℕ) (start : ℤ), 
    start ≤ a ∧ a ≤ start + n ∧ 
    (Finset.range (n + 1)).sum (λ i ↦ start + i) = 2014

-- State the theorem
theorem smallest_yummy_integer : 
  is_yummy (-2013) ∧ ∀ a : ℤ, is_yummy a → -2013 ≤ a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_yummy_integer_l664_66454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daisies_percentage_l664_66421

/-- Represents the total number of flowers in the garden -/
def total : ℕ → ℕ := id

/-- Represents the number of yellow flowers -/
def yellow (t : ℕ) : ℕ := (9 * t) / 10

/-- Represents the number of blue flowers -/
def blue (t : ℕ) : ℕ := t - yellow t

/-- Represents the number of yellow tulips -/
def yellow_tulips (t : ℕ) : ℕ := yellow t / 2

/-- Represents the number of yellow daisies -/
def yellow_daisies (t : ℕ) : ℕ := yellow t - yellow_tulips t

/-- Represents the number of blue daisies -/
def blue_daisies (t : ℕ) : ℕ := (4 * blue t) / 5

/-- Represents the total number of daisies -/
def total_daisies (t : ℕ) : ℕ := yellow_daisies t + blue_daisies t

/-- Theorem stating that 53% of the flowers are daisies -/
theorem daisies_percentage (t : ℕ) (h : t > 0) : 
  (total_daisies t : ℚ) / t = 53 / 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daisies_percentage_l664_66421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_cup_height_l664_66427

/-- The volume of a cone given its radius and height -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The height of a cone given its volume and radius -/
noncomputable def cone_height (v r : ℝ) : ℝ := (3 * v) / (Real.pi * r^2)

theorem conical_cup_height :
  let r : ℝ := 4
  let v : ℝ := 150
  let h : ℝ := cone_height v r
  ⌊h⌋₊ = 9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_cup_height_l664_66427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saturday_earnings_l664_66481

/-- Represents the earnings from a baseball game -/
structure GameEarnings where
  total : ℝ
  ticketPercentage : ℝ
  merchandisePercentage : ℝ
  foodPercentage : ℝ

/-- The total earnings from both games -/
def totalEarnings : ℝ := 7250

/-- Wednesday's game earnings -/
def wednesdayGame : GameEarnings := {
  total := 0,  -- to be calculated
  ticketPercentage := 0.60,
  merchandisePercentage := 0.30,
  foodPercentage := 0.10
}

/-- Saturday's game earnings -/
def saturdayGame : GameEarnings := {
  total := 0,  -- to be calculated
  ticketPercentage := 0.50,
  merchandisePercentage := 0.35,
  foodPercentage := 0.15
}

/-- The difference in ticket sales between Saturday and Wednesday -/
def ticketSalesDifference : ℝ := 142.50

theorem saturday_earnings : 
  ∃ (wTotal sTotal : ℝ),
    wTotal + sTotal = totalEarnings ∧
    wednesdayGame.ticketPercentage * wTotal = saturdayGame.ticketPercentage * sTotal - ticketSalesDifference ∧
    abs (sTotal - 4084.09) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saturday_earnings_l664_66481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_l664_66495

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

theorem sum_and_round :
  round_to_hundredth (75.2591 + 34.3214) = 109.58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_l664_66495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l664_66484

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Point type representing a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Main theorem: Minimum value of |MP| + |MF| is 4 -/
theorem min_distance_sum (para : Parabola) (P : Point) :
  para.equation = (fun x y => y^2 = 4*x) →
  para.focus = (1, 0) →
  P = ⟨3, 1⟩ →
  (∃ (c : ℝ), ∀ (M : Point), para.equation M.x M.y →
    distance M P + distance M ⟨para.focus.1, para.focus.2⟩ ≥ c) ∧
  (∃ (M : Point), para.equation M.x M.y ∧
    distance M P + distance M ⟨para.focus.1, para.focus.2⟩ = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l664_66484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_correct_propositions_l664_66448

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation
variable (perp : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_line : Line → Line → Prop)

-- Define the "not in" relation
variable (not_in : Line → Plane → Prop)

-- Define the planes and lines
variable (α β : Plane)
variable (m n : Line)

-- Define the conditions
variable (diff_planes : α ≠ β)
variable (diff_lines : m ≠ n)
variable (m_not_in_α : not_in m α)
variable (m_not_in_β : not_in m β)
variable (n_not_in_α : not_in n α)
variable (n_not_in_β : not_in n β)

-- Define the four statements
def statement1 := perp_line m n
def statement2 := perp α β
def statement3 := perp_line_plane n β
def statement4 := perp_line_plane m α

-- Define a function to check if a proposition is correct
def is_correct_proposition (conditions : List Prop) (conclusion : Prop) : Prop :=
  (∀ c ∈ conditions, c) → conclusion

-- Define the theorem
theorem exactly_two_correct_propositions :
  (∃ (prop1 prop2 : List Prop × Prop),
    prop1 ≠ prop2 ∧
    is_correct_proposition prop1.1 prop1.2 ∧
    is_correct_proposition prop2.1 prop2.2 ∧
    ∀ (prop3 : List Prop × Prop),
      prop3 ≠ prop1 → prop3 ≠ prop2 →
      ¬(is_correct_proposition prop3.1 prop3.2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_correct_propositions_l664_66448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_natural_number_l664_66472

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | 1 => 1
  | (n + 2) => sequence_a (n + 1) / 2 + 1 / (4 * sequence_a (n + 1))

theorem sqrt_natural_number (n : ℕ) (h : n > 1) : 
  ∃ k : ℕ, k^2 = (2 : ℝ) / (2 * (sequence_a n)^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_natural_number_l664_66472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_with_eraser_cost_l664_66436

/-- The cost of a pencil with an eraser -/
def cost_pencil_with_eraser : ℝ := 0.8

/-- The number of pencils with eraser sold -/
def num_pencils_with_eraser : ℕ := 200

/-- The number of regular pencils sold -/
def num_regular_pencils : ℕ := 40

/-- The number of short pencils sold -/
def num_short_pencils : ℕ := 35

/-- The cost of a regular pencil -/
def cost_regular_pencil : ℝ := 0.5

/-- The cost of a short pencil -/
def cost_short_pencil : ℝ := 0.4

/-- The total revenue from all sales -/
def total_revenue : ℝ := 194

theorem pencil_with_eraser_cost :
  cost_pencil_with_eraser * num_pencils_with_eraser +
  cost_regular_pencil * num_regular_pencils +
  cost_short_pencil * num_short_pencils = total_revenue := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_with_eraser_cost_l664_66436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survival_expectation_l664_66434

/-- The probability of survival for each month -/
noncomputable def survivalProbability : ℝ := 9/10

/-- The number of months considered -/
def numberOfMonths : ℕ := 3

/-- The initial population size -/
def initialPopulation : ℕ := 300

/-- The expected number of survivors after three months -/
noncomputable def expectedSurvivors : ℝ := initialPopulation * survivalProbability ^ numberOfMonths

theorem survival_expectation :
  ⌊expectedSurvivors⌋ = 219 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_survival_expectation_l664_66434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_statements_true_l664_66402

noncomputable section

open Real

def f₁ (x : ℝ) := sin (2 * x - π / 4)
def f₂ (x : ℝ) := sin x + sqrt 3 * cos x
def f₃ (x : ℝ) := sin (cos x) - 1
def f₄ (x : ℝ) := sin (x + π / 4)

def statement₁ : Prop := ∀ k : ℤ, ∃ x : ℝ, x = k * π / 2 + 3 * π / 8 ∧ ∀ y : ℝ, f₁ (2 * x - y) = f₁ y
def statement₂ : Prop := ∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, f₂ x ≤ M
def statement₃ : Prop := ∀ x : ℝ, f₃ (x + 2 * π) = f₃ x
def statement₄ : Prop := ∀ x y : ℝ, -π/2 ≤ x ∧ x < y ∧ y ≤ π/2 → f₄ x < f₄ y

theorem exactly_two_statements_true : 
  (statement₁ ∧ statement₂ ∧ ¬statement₃ ∧ ¬statement₄) ∨
  (statement₁ ∧ ¬statement₂ ∧ statement₃ ∧ ¬statement₄) ∨
  (statement₁ ∧ ¬statement₂ ∧ ¬statement₃ ∧ statement₄) ∨
  (¬statement₁ ∧ statement₂ ∧ statement₃ ∧ ¬statement₄) ∨
  (¬statement₁ ∧ statement₂ ∧ ¬statement₃ ∧ statement₄) ∨
  (¬statement₁ ∧ ¬statement₂ ∧ statement₃ ∧ statement₄) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_statements_true_l664_66402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l664_66482

theorem log_problem (x : ℝ) (h1 : x < 1) (h2 : (Real.log x)^3 - 2 * Real.log (x^3) = 150 * Real.log 10) :
  (Real.log x)^4 - Real.log (x^4) = 645 * Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l664_66482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l664_66423

/-- Parabola structure -/
structure Parabola where
  focus : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.equation point.1 point.2

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_theorem (p : Parabola) 
  (P Q : PointOnParabola p) :
  p.equation = fun x y => y^2 = 4*x →
  distance P.point p.focus = 2 →
  distance Q.point p.focus = 5 →
  distance P.point Q.point = 3 * Real.sqrt 5 ∨ 
  distance P.point Q.point = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l664_66423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l664_66443

def S (n : ℕ) : ℚ := (n + 1) / n

def a : ℕ → ℚ
  | 0 => 0  -- Adding a case for 0 to handle all natural numbers
  | 1 => 2
  | (n+2) => -1 / ((n+1) * (n+2))

theorem sequence_general_term : ∀ n : ℕ, n ≥ 1 → 
  (a n = if n = 1 then S 1 else S n - S (n-1)) := by
  sorry

#check sequence_general_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l664_66443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_dot_product_minimum_l664_66490

/-- Parabola type representing y = x^2 --/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola --/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  hy : y = x^2

/-- Vector between two points --/
def vector (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

/-- Dot product of two vectors --/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Theorem statement --/
theorem parabola_dot_product_minimum (C : Parabola) (l : ℝ) (hl : l ≥ 0)
  (E : ℝ × ℝ) (hE : E = (-l, 0))
  (A B : PointOnParabola C)
  (h_min : ∀ (A' B' : PointOnParabola C),
    dot_product (vector E (A'.x, A'.y)) (vector E (B'.x, B'.y)) ≥
    dot_product (vector E (A.x, A.y)) (vector E (B.x, B.y)))
  (h_zero : dot_product (vector E (A.x, A.y)) (vector E (B.x, B.y)) = 0) :
  l = C.p / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_dot_product_minimum_l664_66490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l664_66403

theorem coefficient_x_squared_in_expansion : ∃ c : ℤ,
  (c = -26 ∧
   ∀ n : ℕ, n ≠ 2 → 
   (Polynomial.coeff ((1 - 2 * Polynomial.X) ^ 5 * 
    (1 + 3 * Polynomial.X) ^ 4 : Polynomial ℤ) n ≠ c)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l664_66403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_coordinate_range_l664_66431

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the foci
noncomputable def F1 : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def F2 : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the condition for an obtuse angle
def is_obtuse_angle (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x + Real.sqrt 3) * (x - Real.sqrt 3) + y^2 < 0

-- Theorem statement
theorem ellipse_x_coordinate_range (x y : ℝ) :
  is_on_ellipse x y → is_obtuse_angle (x, y) →
  -2 * Real.sqrt 6 / 3 < x ∧ x < 2 * Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_coordinate_range_l664_66431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_theorem_l664_66496

/-- Given conditions on a complex number z -/
def ComplexConditions (z : ℂ) : Prop :=
  (z + 2*Complex.I).im = 0 ∧ (z / (2 - Complex.I)).im = 0

/-- Statement of the theorem -/
theorem complex_number_theorem (z : ℂ) (h : ComplexConditions z) :
  z = 4 - 2*Complex.I ∧
  ∀ a : ℝ, (z + a*Complex.I)^2 ∈ {w : ℂ | w.re > 0 ∧ w.im < 0} ↔ -2 < a ∧ a < 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_theorem_l664_66496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_given_cos_l664_66453

theorem sin_cos_sum_given_cos (α : ℝ) :
  Real.cos (75 * π / 180 + α) = 1/3 →
  Real.sin (α - 15 * π / 180) + Real.cos (105 * π / 180 - α) = -2/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_given_cos_l664_66453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_existence_l664_66474

/-- Given distinct positive real numbers a and b, and a function f,
    there exists a unique positive real α such that f(α) equals the s-th root of (a^s + b^s)/2
    for all s in (0,1). -/
theorem unique_root_existence (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ∀ s : ℝ, s ∈ Set.Ioo 0 1 →
  ∃! α : ℝ, α > 0 ∧ -α + Real.sqrt ((α + a) * (α + b)) = ((a^s + b^s) / 2) ^ (1/s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_existence_l664_66474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_ten_thousandth_l664_66467

/-- Rounds a real number to the nearest ten-thousandth -/
noncomputable def round_to_ten_thousandth (x : ℝ) : ℝ :=
  (⌊x * 10000 + 0.5⌋) / 10000

/-- The given number to be rounded -/
def given_number : ℝ := 0.00356

/-- Theorem stating that rounding the given number to the nearest ten-thousandth results in 0.0036 -/
theorem round_to_nearest_ten_thousandth :
  round_to_ten_thousandth given_number = 0.0036 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_ten_thousandth_l664_66467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_expression_l664_66465

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).foldl (λ sum i => sum + a (i + 1)) 0

theorem sequence_expression (a : ℕ → ℕ) :
  (∀ n : ℕ, sequence_sum a n = 2 * a n - n) →
  (∀ n : ℕ, a n = 2^n - 1) :=
by
  sorry

#check sequence_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_expression_l664_66465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inside_c_outside_a_b_l664_66416

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are tangent to each other -/
def areTangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Calculates the area of a circle -/
noncomputable def circleArea (c : Circle) : ℝ :=
  Real.pi * c.radius^2

/-- Theorem: Area inside circle C but outside circles A and B -/
theorem area_inside_c_outside_a_b (a b c : Circle) :
  a.radius = 1 →
  b.radius = 1 →
  c.radius = 2 →
  areTangent a b →
  areTangent a c →
  areTangent b c →
  ∃ (area : ℝ), area = 4 * Real.pi - (2 * Real.pi / 3) + Real.sqrt 3 ∧
                area = circleArea c - 2 * (Real.pi / 3 - Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inside_c_outside_a_b_l664_66416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_with_tan_l664_66458

theorem sin_double_angle_with_tan (α : ℝ) (h : Real.tan α = -3/5) : 
  Real.sin (2 * α) = -15/17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_with_tan_l664_66458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_qr_length_l664_66479

/-- Define the equilateral triangle DEF -/
def Triangle (D E F : ℝ × ℝ) : Prop :=
  ‖E - D‖ = 9 ∧ ‖F - E‖ = 9 ∧ ‖D - F‖ = 9

/-- Define the circle centered at Q -/
def CircleQ (Q : ℝ × ℝ) (D E F : ℝ × ℝ) : Prop :=
  ‖Q - D‖ = ‖Q - F‖ ∧ (Q - D) • (E - D) = 0

/-- Define the circle centered at R -/
def CircleR (R : ℝ × ℝ) (D E F : ℝ × ℝ) : Prop :=
  ‖R - F‖ = ‖R - E‖ ∧ (R - F) • (D - F) = 0

/-- Theorem statement -/
theorem qr_length 
  (D E F Q R : ℝ × ℝ) 
  (h_triangle : Triangle D E F)
  (h_circleQ : CircleQ Q D E F)
  (h_circleR : CircleR R D E F) :
  ‖Q - R‖ = 9 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_qr_length_l664_66479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_power_product_equals_one_l664_66475

theorem fraction_power_product_equals_one :
  (5 / 6 : ℚ) ^ 4 * (5 / 6 : ℚ) ^ ((-4) : ℤ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_power_product_equals_one_l664_66475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l664_66462

noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 2) + 1

theorem g_neither_even_nor_odd :
  ¬(∀ x, g x = g (-x)) ∧ ¬(∀ x, g x = -g (-x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l664_66462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_integer_solutions_l664_66470

theorem sum_of_integer_solutions : ∃ (S : Finset ℤ), 
  (∀ x ∈ S, (5 * x + 2 > 3 * (x - 1)) ∧ ((1 / 2 : ℚ) * x - 1 ≤ 7 - (3 / 2 : ℚ) * x)) ∧
  (∀ x : ℤ, (5 * x + 2 > 3 * (x - 1)) ∧ ((1 / 2 : ℚ) * x - 1 ≤ 7 - (3 / 2 : ℚ) * x) → x ∈ S) ∧
  (S.sum id = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_integer_solutions_l664_66470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_customers_is_80_l664_66487

/-- Represents a table with a number of women and men -/
structure Table where
  women : Nat
  men : Nat

/-- Calculates the total number of customers at a table -/
def Table.totalCustomers (t : Table) : Nat := t.women + t.men

/-- The list of tables with their respective number of women and men -/
def tables : List Table := [
  ⟨3, 5⟩, ⟨4, 4⟩, ⟨6, 2⟩, ⟨5, 3⟩, ⟨7, 1⟩,
  ⟨8, 0⟩, ⟨2, 6⟩, ⟨4, 4⟩, ⟨3, 5⟩, ⟨5, 3⟩
]

/-- Theorem stating that the total number of customers across all tables is 80 -/
theorem total_customers_is_80 : 
  (tables.map Table.totalCustomers).sum = 80 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_customers_is_80_l664_66487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_not_both_ends_eq_nine_tenths_l664_66406

/-- The number of people in the arrangement -/
def n : ℕ := 5

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := Nat.factorial n

/-- The number of arrangements where A and B are at both ends -/
def both_ends_arrangements : ℕ := 2 * Nat.factorial (n - 2)

/-- The probability that A and B do not stand at both ends simultaneously -/
def prob_not_both_ends : ℚ :=
  (total_arrangements - both_ends_arrangements : ℚ) / total_arrangements

theorem prob_not_both_ends_eq_nine_tenths :
  prob_not_both_ends = 9/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_not_both_ends_eq_nine_tenths_l664_66406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coins_needed_l664_66407

-- Define the coin denominations
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def half_dollar : ℕ := 50

-- Function to check if a combination of coins can make all amounts from 1 to 99 cents
def can_make_all_amounts (coins : List ℕ) : Prop :=
  ∀ (amount : ℕ), 1 ≤ amount ∧ amount ≤ 99 → 
    ∃ (combination : List ℕ), combination.sum = amount ∧ combination.toFinset ⊆ coins.toFinset

-- Function to check if a list of coins contains at least one half-dollar
def contains_half_dollar (coins : List ℕ) : Prop :=
  half_dollar ∈ coins

-- Theorem stating that 14 is the smallest number of coins needed
theorem min_coins_needed : 
  ∃ (coins : List ℕ), 
    coins.length = 14 ∧ 
    can_make_all_amounts coins ∧ 
    contains_half_dollar coins ∧
    ∀ (other_coins : List ℕ), 
      can_make_all_amounts other_coins → 
      contains_half_dollar other_coins → 
      other_coins.length ≥ 14 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coins_needed_l664_66407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_operations_equality_l664_66485

/-- Define the circle operation -/
def circle_op (α β : ℝ) (a b : ℝ) : ℝ := α * a + β * b

/-- Define the star operation -/
def star_op (γ δ : ℝ) (a b : ℝ) : ℝ := γ * a + δ * b

/-- The main theorem stating the conditions for the equalities to hold -/
theorem operations_equality (α β γ δ : ℝ) :
  (∀ a b c : ℝ, star_op γ δ (circle_op α β a b) c = circle_op α β (star_op γ δ a c) (star_op γ δ b c)) ∧
  (∀ a b c : ℝ, circle_op α β (star_op γ δ a b) c = star_op γ δ (circle_op α β a c) (circle_op α β b c)) ↔
  ((β = 0 ∧ δ = 0) ∨
   (β = 0 ∧ α = 1) ∨
   (δ = 0 ∧ γ = 1) ∨
   (α + β = 1 ∧ γ + δ = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_operations_equality_l664_66485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l664_66441

noncomputable def a (n : ℝ) : Fin 3 → ℝ := ![1, n, 1/2]
def b : Fin 3 → ℝ := ![-2, 1, -1]

theorem vector_magnitude (n : ℝ) :
  (∃ k : ℝ, (2 • (a n) - b) = k • b) →
  Real.sqrt ((a n 0)^2 + (a n 1)^2 + (a n 2)^2) = Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l664_66441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_s_squared_l664_66469

/-- A hyperbola centered at the origin passing through specific points -/
structure Hyperbola where
  set : Set (ℝ × ℝ)
  -- The hyperbola passes through (5, 3)
  point1 : (5, 3) ∈ set
  -- The hyperbola passes through (0, -3)
  point2 : (0, -3) ∈ set
  -- The hyperbola passes through (-4, s) for some real s
  point3 : ∃ s : ℝ, (-4, s) ∈ set
  -- The hyperbola is centered at the origin
  center : (0, 0) ∈ set

/-- The theorem stating that s^2 = 369/25 for the given hyperbola -/
theorem hyperbola_s_squared (h : Hyperbola) : 
  ∃ s : ℝ, (-4, s) ∈ h.set ∧ s^2 = 369/25 := by
  sorry

#check hyperbola_s_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_s_squared_l664_66469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_exists_l664_66460

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point inside a circle
def PointInsideCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

-- Define a rectangle
structure Rectangle where
  center : ℝ × ℝ
  width : ℝ
  height : ℝ

-- Define when a rectangle is inscribed in a circle
def RectangleInscribed (r : Rectangle) (c : Circle) : Prop :=
  ∀ corner : ℝ × ℝ,
    (corner.1 = r.center.1 + r.width / 2 ∨ corner.1 = r.center.1 - r.width / 2) ∧
    (corner.2 = r.center.2 + r.height / 2 ∨ corner.2 = r.center.2 - r.height / 2) →
    (corner.1 - c.center.1)^2 + (corner.2 - c.center.2)^2 = c.radius^2

-- Define when a point lies on the side of a rectangle
def PointOnRectangleSide (p : ℝ × ℝ) (r : Rectangle) : Prop :=
  (p.1 = r.center.1 - r.width / 2 ∨ p.1 = r.center.1 + r.width / 2) ∧ 
  (r.center.2 - r.height / 2 ≤ p.2 ∧ p.2 ≤ r.center.2 + r.height / 2) ∨
  (p.2 = r.center.2 - r.height / 2 ∨ p.2 = r.center.2 + r.height / 2) ∧ 
  (r.center.1 - r.width / 2 ≤ p.1 ∧ p.1 ≤ r.center.1 + r.width / 2)

-- Theorem statement
theorem inscribed_rectangle_exists (c : Circle) (p1 p2 : ℝ × ℝ) 
  (h1 : PointInsideCircle c p1) (h2 : PointInsideCircle c p2) (h3 : p1 ≠ p2) : 
  ∃ r : Rectangle, RectangleInscribed r c ∧ 
                   PointOnRectangleSide p1 r ∧ 
                   PointOnRectangleSide p2 r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_exists_l664_66460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_three_six_equals_eleven_thirtieths_l664_66455

/-- Represents a repeating decimal with a single repeating digit after the decimal point. -/
def repeating_decimal (whole : ℕ) (decimal : ℕ) (repeating : ℕ) : ℚ :=
  whole + (decimal + repeating / 9) / 10

/-- Proves that 0.3666... (repeating 6) is equal to 11/30 -/
theorem repeating_decimal_three_six_equals_eleven_thirtieths :
  repeating_decimal 0 3 6 = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_three_six_equals_eleven_thirtieths_l664_66455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_l664_66491

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_e :
  let df := λ x => Real.log x + 1
  let slope := df (Real.exp 1)
  let point := (Real.exp 1, f (Real.exp 1))
  let tangent_line := λ x => slope * (x - point.1) + point.2
  tangent_line = λ x => 2 * x - Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_l664_66491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_is_121_l664_66429

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := 1 + (n - 1) * 2

-- Define the sum S_n
noncomputable def S (n : ℕ) : ℝ := n * (2 + (n - 1) * 2) / 2

-- State the theorem
theorem max_ratio_is_121 :
  (∀ n : ℕ, n > 0 → a n > 0) →
  (∀ n : ℕ, n > 0 → ∃ k : ℝ, ∀ m : ℕ, m > 0 → Real.sqrt (S (m + 1)) - Real.sqrt (S m) = k) →
  (∀ n : ℕ, n > 0 → S (n + 10) / (a n)^2 ≤ 121) ∧
  (∃ n : ℕ, n > 0 ∧ S (n + 10) / (a n)^2 = 121) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_is_121_l664_66429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_is_correct_l664_66418

noncomputable def f (x : ℝ) : ℝ := 4 * x - 5

noncomputable def g (x : ℝ) : ℝ := 3 * x + 7

noncomputable def h (x : ℝ) : ℝ := f (g x)

noncomputable def h_inverse (x : ℝ) : ℝ := (x - 23) / 12

theorem h_inverse_is_correct : Function.LeftInverse h_inverse h ∧ Function.RightInverse h_inverse h := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_is_correct_l664_66418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milkshake_cost_is_five_l664_66415

/-- The cost of items and money brought by Jim and his cousin -/
structure RestaurantScenario where
  cheeseburger_cost : ℚ
  cheese_fries_cost : ℚ
  jim_money : ℚ
  cousin_money : ℚ
  total_spent_percentage : ℚ

/-- The solution to the restaurant problem -/
def milkshake_cost (scenario : RestaurantScenario) : ℚ :=
  let total_money := scenario.jim_money + scenario.cousin_money
  let total_spent := total_money * scenario.total_spent_percentage
  let known_costs := 2 * scenario.cheeseburger_cost + scenario.cheese_fries_cost
  (total_spent - known_costs) / 2

/-- Theorem stating that the milkshake cost is $5 given the specific scenario -/
theorem milkshake_cost_is_five :
  let scenario : RestaurantScenario := {
    cheeseburger_cost := 3,
    cheese_fries_cost := 8,
    jim_money := 20,
    cousin_money := 10,
    total_spent_percentage := 4/5
  }
  milkshake_cost scenario = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milkshake_cost_is_five_l664_66415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_when_a_is_2_A_B_intersection_empty_iff_l664_66413

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0}
def B : Set ℝ := {x | -2 < x ∧ x < 2}

-- Part 1: Prove that when a = 2, A ∪ B = (-2, 3]
theorem union_A_B_when_a_is_2 : 
  A 2 ∪ B = Set.Ioc (-2 : ℝ) 3 := by sorry

-- Part 2: Prove the condition for A ∩ B = ∅
theorem A_B_intersection_empty_iff (a : ℝ) : 
  A a ∩ B = ∅ ↔ a ≤ -3 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_when_a_is_2_A_B_intersection_empty_iff_l664_66413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_hobbits_l664_66417

/-- Represents the types of participants at the festival -/
inductive Participant where
  | Human
  | Elf
  | Hobbit
deriving Repr, DecidableEq

/-- The festival has more than 20 participants -/
def festival_size : ℕ := 21

/-- Any subset of 15 participants contains at least 4 humans and 5 elves -/
axiom subset_composition (s : Finset Participant) :
  s.card = 15 → (s.filter (· = Participant.Human)).card ≥ 4 ∧
              (s.filter (· = Participant.Elf)).card ≥ 5

/-- The number of hobbits at the festival is zero -/
theorem no_hobbits (participants : Finset Participant)
  (h_size : participants.card = festival_size) :
  (participants.filter (· = Participant.Hobbit)).card = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_hobbits_l664_66417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_progression_theorem_l664_66401

noncomputable section

open Real

theorem angle_progression_theorem (α β γ : ℝ) : 
  0 < α ∧ α < π/2 ∧
  0 < β ∧ β < π/2 ∧
  0 < γ ∧ γ < π/2 ∧
  β - α = π/12 ∧
  γ - β = π/12 ∧
  ∃ q : ℝ, tan β = q * tan α ∧ tan γ = q^2 * tan α
  →
  α = π/6 ∧ β = π/4 ∧ γ = π/3 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_progression_theorem_l664_66401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_30_and_pythagorean_l664_66493

-- Define 30 degrees in radians
noncomputable def angle_30 : ℝ := Real.pi / 6

-- Theorem statement
theorem sin_cos_30_and_pythagorean :
  (Real.sin angle_30 = 1 / 2) ∧ 
  (Real.cos angle_30 = Real.sqrt 3 / 2) ∧ 
  (Real.sin angle_30)^2 + (Real.cos angle_30)^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_30_and_pythagorean_l664_66493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_properties_l664_66400

noncomputable def f (x : ℝ) := Real.cos (2 * x + Real.pi / 4)

theorem cosine_properties (k : ℤ) :
  let center := (Real.pi / 8 + k * Real.pi / 2, 0)
  let axis := -Real.pi / 8 + k * Real.pi / 2
  let decreasing_interval := Set.Icc (-Real.pi / 8 + k * Real.pi) (3 * Real.pi / 8 + k * Real.pi)
  let period := Real.pi
  -- Center of symmetry
  (∀ x, f (center.1 + x) = f (center.1 - x)) ∧
  -- Axis of symmetry
  (∀ x, f (axis + x) = f (axis - x)) ∧
  -- Decreasing interval
  (∀ x ∈ decreasing_interval, ∀ y ∈ decreasing_interval, x < y → f x > f y) ∧
  -- Smallest positive period
  (∀ x, f (x + period) = f x) ∧
  (∀ t, 0 < t ∧ t < period → ∃ x, f (x + t) ≠ f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_properties_l664_66400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_on_x_axis_l664_66433

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 1/4

noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem tangent_on_x_axis (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ = 0 ∧ f_deriv a x₀ = 0) → a = -3/4 := by
  sorry

#check tangent_on_x_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_on_x_axis_l664_66433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_distribution_arrangements_l664_66445

theorem mask_distribution_arrangements : ∃ (total : ℕ), total = 8400 := by
  -- Define the number of masks and people
  let num_masks : ℕ := 7
  let num_people : ℕ := 4

  -- Define the number of ways to distribute masks
  let distribute_masks : ℕ := 350

  -- Calculate the number of ways to arrange people
  let arrange_people : ℕ := Nat.factorial num_people

  -- Calculate the total number of arrangements
  let total_arrangements : ℕ := distribute_masks * arrange_people

  -- Prove that there exists a total number of arrangements equal to 8400
  use total_arrangements
  
  -- The actual proof would go here
  sorry

#eval 350 * Nat.factorial 4 -- This will evaluate to 8400

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_distribution_arrangements_l664_66445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peach_tree_max_profit_l664_66494

/-- Profit function for a peach tree -/
noncomputable def L (x : ℝ) : ℝ := 64 - 48 / (x + 1) - 3 * x

/-- The maximum profit occurs at x = 3 and is equal to 4300/100 -/
theorem peach_tree_max_profit :
  ∃ (x : ℝ), x ∈ Set.Icc 0 5 ∧
  (∀ y ∈ Set.Icc 0 5, L y ≤ L x) ∧
  x = 3 ∧ L x = 43 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_peach_tree_max_profit_l664_66494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_point_l664_66408

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 6*y + 3

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 3)

-- Define the point we're measuring distance to
def point : ℝ × ℝ := (10, 3)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem distance_to_point : distance circle_center point = 8 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_point_l664_66408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corn_amount_proof_l664_66477

/-- The price of corn per pound in cents -/
def corn_price : ℕ := 110

/-- The price of beans per pound in cents -/
def bean_price : ℕ := 40

/-- The total number of pounds of corn and beans bought -/
def total_pounds : ℕ := 24

/-- The total cost in cents -/
def total_cost : ℕ := 1920

/-- The amount of corn bought in pounds -/
def corn_pounds : ℚ := 960 / 70

theorem corn_amount_proof :
  ∃ (bean_pounds : ℚ),
    bean_pounds + corn_pounds = total_pounds ∧
    bean_price * bean_pounds + corn_price * corn_pounds = total_cost ∧
    (⌊corn_pounds * 10⌋ : ℚ) / 10 = 137 / 10 := by
  sorry

#eval (⌊corn_pounds * 10⌋ : ℚ) / 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corn_amount_proof_l664_66477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_empty_l664_66404

/-- Set M of integer points on the unit circle -/
def M : Set (ℤ × ℤ) :=
  {p | p.1^2 + p.2^2 = 1}

/-- Set N of integer points on the circle with center (1,0) and radius 1 -/
def N : Set (ℤ × ℤ) :=
  {p | (p.1 - 1)^2 + p.2^2 = 1}

/-- The intersection of M and N is empty -/
theorem M_intersect_N_empty : M ∩ N = ∅ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_empty_l664_66404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_upper_bound_for_phi_d_ratio_l664_66424

/-- Number of positive divisors of n -/
def d (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def φ (n : ℕ) : ℕ := sorry

/-- The main theorem: For any constant C > 0, there exists a positive integer n 
    such that φ(d(n)) / d(φ(n)) > C -/
theorem no_upper_bound_for_phi_d_ratio : 
  ∀ C : ℚ, C > 0 → ∃ n : ℕ, n > 0 ∧ (φ (d n) : ℚ) / (d (φ n)) > C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_upper_bound_for_phi_d_ratio_l664_66424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clark_family_park_visit_cost_l664_66420

/-- Represents the cost structure and family composition for the Clark family's amusement park visit -/
structure ClarkFamilyParkVisit where
  regular_ticket_price : ℚ
  youngest_gen_count : ℕ
  second_youngest_gen_count : ℕ
  second_oldest_gen_count : ℕ
  oldest_gen_count : ℕ
  youngest_discount : ℚ
  oldest_discount : ℚ
  senior_ticket_price : ℚ

/-- Calculates the total cost for the Clark family's amusement park visit -/
def total_cost (visit : ClarkFamilyParkVisit) : ℚ :=
  let youngest_price := visit.regular_ticket_price * (1 - visit.youngest_discount)
  let oldest_price := visit.regular_ticket_price * (1 - visit.oldest_discount)
  youngest_price * visit.youngest_gen_count +
  visit.regular_ticket_price * visit.second_youngest_gen_count +
  visit.senior_ticket_price * visit.second_oldest_gen_count +
  oldest_price * visit.oldest_gen_count

/-- Theorem stating that the total cost for the Clark family's amusement park visit is $52 -/
theorem clark_family_park_visit_cost :
  ∃ (visit : ClarkFamilyParkVisit),
    visit.regular_ticket_price = 10 ∧
    visit.youngest_gen_count = 3 ∧
    visit.second_youngest_gen_count = 1 ∧
    visit.second_oldest_gen_count = 2 ∧
    visit.oldest_gen_count = 1 ∧
    visit.youngest_discount = 2/5 ∧
    visit.oldest_discount = 3/10 ∧
    visit.senior_ticket_price = 7 ∧
    total_cost visit = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clark_family_park_visit_cost_l664_66420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_in_ellipse_l664_66459

/-- The radius of two externally tangent circles that are internally tangent to a specific ellipse -/
theorem circle_radius_in_ellipse : ∃ (r : ℝ), 
  -- Two circles with radius r
  -- Centers of circles at (±r, 0)
  let circle1 := λ (x y : ℝ) => (x - r)^2 + y^2 = r^2
  let circle2 := λ (x y : ℝ) => (x + r)^2 + y^2 = r^2
  -- Ellipse equation
  let ellipse := λ (x y : ℝ) => 4*x^2 + 9*y^2 = 18
  -- Circles are externally tangent to each other
  (circle1 r 0 ∧ circle2 (-r) 0)
  -- Circles are internally tangent to the ellipse
  ∧ (∃ (x y : ℝ), circle1 x y ∧ ellipse x y)
  ∧ (∃ (x y : ℝ), circle2 x y ∧ ellipse x y)
  -- The radius r is equal to 3√5/5
  ∧ r = 3 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_in_ellipse_l664_66459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l664_66412

open Real

noncomputable def f (x : ℝ) : ℝ := arctan x + arctan ((1 + x) / (1 - x))

theorem f_range (x : ℝ) (h : x ≠ 1) : f x = -Real.pi/4 ∨ f x = -3*Real.pi/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l664_66412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_15_value_l664_66422

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add this case for 0
  | 1 => 2
  | n + 2 => 1 - 2 / (sequence_a (n + 1) + 1)

theorem a_15_value : sequence_a 15 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_15_value_l664_66422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l664_66438

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (2 * ω * x + Real.pi / 6)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω x - Real.sqrt 3

def is_monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def has_n_zeros (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : Prop :=
  ∃ (zeros : Finset ℝ), zeros.card = n ∧ (∀ x ∈ zeros, a ≤ x ∧ x ≤ b ∧ f x = 0)

theorem f_properties (ω : ℝ) (h_ω_pos : ω > 0) :
  (∀ x : ℝ, f ω (x + Real.pi / (2 * ω)) = f ω x → ω = 1) ∧
  (∀ k : ℤ, is_monotone_decreasing (f 1) (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3)) ∧
  (∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ Real.pi / 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ Real.pi / 2 ∧
    f 1 x₁ = 2 ∧ f 1 x₂ = -1 ∧
    ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f 1 x ≤ 2 ∧ f 1 x ≥ -1) ∧
  (has_n_zeros (g ω) 0 (3 * Real.pi) 5 → 25 / 36 ≤ ω ∧ ω < 3 / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l664_66438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l664_66410

-- Define the triangle ABC
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- State the given condition
def Condition (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * (Real.sin A + Real.sin B) + b * Real.sin B = c * Real.sin (A + B)

theorem triangle_theorem (A B C : ℝ) (a b c : ℝ) 
  (h1 : Triangle A B C a b c) (h2 : Condition A B C a b c) :
  C = 2 * Real.pi / 3 ∧ 
  (a = 2 ∧ c = 3 * Real.sqrt 3 → 
    (1/2) * a * (2 * Real.sqrt 6 - 1) * Real.sin C = (6 * Real.sqrt 2 - Real.sqrt 3) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l664_66410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_min_negative_l664_66435

-- Define the real functions f and g
variable (f g : ℝ → ℝ)

-- Define the function F
def F (a b : ℝ) : ℝ → ℝ := λ x => a * f x + b * g x + 2

-- Axioms for odd functions
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_odd : ∀ x, g (-x) = -g x

-- Axiom for the maximum value of F on (0, +∞)
axiom F_max_positive : ∃ a b : ℝ, ∀ x > 0, F f g a b x ≤ 8

-- Theorem to prove
theorem F_min_negative :
  ∃ a b : ℝ, ∀ x < 0, F f g a b x ≥ -4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_min_negative_l664_66435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_set_properties_l664_66409

theorem empty_set_properties : ∃! s : Set Nat, 
  (∀ (A : Set Nat), s ⊆ A) ∧ 
  (s ≠ ({0} : Set Nat)) ∧ 
  (∃ (B : Set (Set Nat)), B = {s}) ∧ 
  (∀ (C : Set Nat), ∃ (D E : Set Nat), D ⊆ C ∧ E ⊆ C ∧ D ≠ E) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_set_properties_l664_66409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weather_probability_l664_66426

/-- The probability of rain on any given day -/
noncomputable def P_rain : ℝ := 1/4

/-- The probability of rain on a day given that it rained the previous day -/
noncomputable def P_rain_given_rain : ℝ := 2/3

/-- The probability of no rain on a day given that it did not rain the previous day -/
noncomputable def P_no_rain_given_no_rain : ℝ := 8/9

/-- Theorem stating that the probability of no rain given no rain the previous day is 8/9 -/
theorem weather_probability :
  P_no_rain_given_no_rain = 8/9 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weather_probability_l664_66426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relation_between_x_and_y_l664_66447

-- Define the variables and conditions
variable (t : ℝ)
variable (h1 : t > 0)
variable (h2 : t ≠ 1)

-- Define x and y as functions of t
noncomputable def x (t : ℝ) : ℝ := t^(2/(t-1))
noncomputable def y (t : ℝ) : ℝ := t^((t+1)/(t-1))

-- State the theorem
theorem relation_between_x_and_y (t : ℝ) (h1 : t > 0) (h2 : t ≠ 1) : 
  (y t)^(1/(x t)) = (x t)^(y t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_relation_between_x_and_y_l664_66447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_l664_66419

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ

/-- Calculates the sum of squared distances from a point to the vertices of an equilateral triangle -/
noncomputable def sumSquaredDistances (p : Point) (t : EquilateralTriangle) : ℝ :=
  (p.x^2 + p.y^2) + 
  ((p.x - t.sideLength)^2 + p.y^2) + 
  ((p.x - t.sideLength/2)^2 + (p.y - (Real.sqrt 3 * t.sideLength)/2)^2)

/-- Theorem: The locus of points with constant sum of squared distances to vertices of an equilateral triangle is a circle -/
theorem locus_is_circle (t : EquilateralTriangle) (a : ℝ) (h : a > t.sideLength^2) :
  ∃ (center : Point) (radius : ℝ), 
    ∀ (p : Point), sumSquaredDistances p t = a ↔ 
      (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_l664_66419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l664_66483

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and focal distance c,
    if the triangle formed by the center, focus, and a point where a perpendicular
    from the focus intersects the ellipse is acute-angled, then the eccentricity e
    is in the range (0, (√5 - 1)/2) -/
theorem ellipse_eccentricity_range (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : c^2 = a^2 - b^2) (h4 : b^2 < a * c) : 
  let e := c / a
  0 < e ∧ e < (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l664_66483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_factors_630_l664_66411

def is_even (n : ℕ) : Bool := n % 2 = 0

def is_divisible_by (n m : ℕ) : Bool := m % n = 0

def is_factor_of (n m : ℕ) : Bool := m % n = 0

def sum_of_special_factors (n : ℕ) : ℕ :=
  (Finset.filter (λ x => is_even x ∧ is_divisible_by 15 x ∧ is_factor_of x n) (Finset.range (n + 1))).sum id

theorem sum_of_special_factors_630 :
  sum_of_special_factors 630 = 960 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_factors_630_l664_66411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_to_stream_and_cabin_l664_66468

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents the cowboy's journey -/
noncomputable def cowboy_journey (initial_position stream_point cabin : Point) : ℝ :=
  2 * distance initial_position stream_point + distance stream_point cabin

/-- The theorem statement -/
theorem shortest_path_to_stream_and_cabin : 
  let initial_position : Point := ⟨-2, -6⟩
  let cabin : Point := ⟨10, -15⟩
  ∃ (stream_point : Point), stream_point.y = stream_point.x ∧ 
    cowboy_journey initial_position stream_point cabin = 8 + Real.sqrt 545 := by
  sorry

#check shortest_path_to_stream_and_cabin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_to_stream_and_cabin_l664_66468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l664_66489

theorem hyperbola_equation (a b k : ℝ) (ha : a > 0) (hb : b > 0) (hk : k > 0) :
  b / a = k →
  Real.sqrt 5 * k = (Real.sqrt (a^2 + b^2)) / a →
  4^2 / a^2 - 1^2 / b^2 = 1 →
  a^2 = 12 ∧ b^2 = 3 :=
by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l664_66489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_billing_theorem_l664_66442

/-- Represents the tariff rates for electricity consumption -/
structure TariffRates where
  peak : ℝ
  semiPeak : ℝ
  night : ℝ

/-- Represents the meter readings for a given month -/
structure MeterReadings where
  current : List ℝ
  previous : List ℝ

/-- Calculates the maximum additional payment based on meter readings and tariff rates -/
noncomputable def maxAdditionalPayment (readings : MeterReadings) (rates : TariffRates) : ℝ :=
  sorry

/-- Calculates the expected difference between company's calculation and client's payment -/
noncomputable def expectedDifference (readings : MeterReadings) (rates : TariffRates) (clientPayment : ℝ) : ℝ :=
  sorry

/-- Main theorem stating the results of the calculations -/
theorem electricity_billing_theorem (readings : MeterReadings) (rates : TariffRates) (clientPayment : ℝ) :
  readings.current.length = 6 ∧ readings.previous.length = 6 ∧
  (∀ i j, i < j → i < readings.current.length → j < readings.current.length → readings.current[i]! < readings.current[j]!) ∧
  (∀ i j, i < j → i < readings.previous.length → j < readings.previous.length → readings.previous[i]! < readings.previous[j]!) ∧
  rates.peak = 4.03 ∧ rates.semiPeak = 3.39 ∧ rates.night = 1.01 ∧
  clientPayment = 660.72 →
  maxAdditionalPayment readings rates = 397.34 ∧
  expectedDifference readings rates clientPayment = 19.30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_billing_theorem_l664_66442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l664_66456

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_calculation (principal interest time : ℝ) 
  (h_principal : principal = 2000)
  (h_interest : interest = 500)
  (h_time : time = 2) :
  ∃ rate : ℝ, simple_interest principal rate time = interest ∧ rate = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l664_66456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_equals_sqrt_l664_66461

theorem sin_cos_sum_equals_sqrt (k : ℕ) (hk : k > 0) : 
  (Real.sin (π / (3 * k : ℝ)) + Real.cos (π / (3 * k : ℝ)) = 2 * Real.sqrt k / 3) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_equals_sqrt_l664_66461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_reciprocals_l664_66430

theorem min_value_sum_reciprocals (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  let a : ℝ × ℝ := (x - 1, y)
  let b : ℝ × ℝ := (1, 2)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  (1 / x + 1 / y ≥ 3 + 2 * Real.sqrt 2) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    let a₀ : ℝ × ℝ := (x₀ - 1, y₀)
    (a₀.1 * b.1 + a₀.2 * b.2 = 0) ∧
    (1 / x₀ + 1 / y₀ = 3 + 2 * Real.sqrt 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_reciprocals_l664_66430
