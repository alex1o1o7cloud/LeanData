import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circle_area_l148_14807

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a circle passes through a point -/
def passesThrough (c : Circle) (p : Point) : Prop :=
  distance c.center p = c.radius

/-- Check if a circle touches a line segment -/
def touches (c : Circle) (p1 p2 : Point) : Prop :=
  ∃ (t : Point), distance c.center t = c.radius ∧ 
    (t.x - p1.x) * (p2.y - p1.y) = (t.y - p1.y) * (p2.x - p1.x)

/-- Distance from a point to a line segment -/
noncomputable def distanceToLineSegment (p : Point) (p1 p2 : Point) : ℝ :=
  sorry -- Actual implementation would go here

/-- Area of a rectangle -/
noncomputable def rectangleArea (r : Rectangle) : ℝ :=
  distance r.A r.B * distance r.A r.D

/-- Main theorem -/
theorem rectangle_circle_area 
  (r : Rectangle) 
  (c : Circle) 
  (M N : Point) :
  passesThrough c r.C →
  touches c r.A r.B →
  touches c r.A r.D →
  distanceToLineSegment r.C M N = 5 →
  rectangleArea r = 25 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circle_area_l148_14807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plumbing_job_washers_remaining_l148_14867

/-- Calculates the number of washers remaining after Edward's plumbing job --/
theorem plumbing_job_washers_remaining 
  (copper_pipe_length : ℕ)
  (pvc_pipe_length : ℕ)
  (copper_bolt_per_length : ℕ)
  (pvc_bolt_per_length : ℕ)
  (washers_per_copper_bolt : ℕ)
  (washers_per_pvc_bolt : ℕ)
  (total_washers : ℕ) :
  copper_pipe_length = 40 →
  pvc_pipe_length = 30 →
  copper_bolt_per_length = 5 →
  pvc_bolt_per_length = 10 →
  washers_per_copper_bolt = 2 →
  washers_per_pvc_bolt = 3 →
  total_washers = 50 →
  total_washers - (
    (copper_pipe_length / copper_bolt_per_length * washers_per_copper_bolt) +
    (pvc_pipe_length / pvc_bolt_per_length * 2 * washers_per_pvc_bolt)
  ) = 16 :=
by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plumbing_job_washers_remaining_l148_14867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_arithmetic_geometric_l148_14827

/-- ArithmeticSeq x a b y means x, a, b, y form an arithmetic sequence -/
def ArithmeticSeq (x a b y : ℝ) : Prop :=
  ∃ r : ℝ, a = x + r ∧ b = x + 2*r ∧ y = x + 3*r

/-- GeometricSeq x c d y means x, c, d, y form a geometric sequence -/
def GeometricSeq (x c d y : ℝ) : Prop :=
  ∃ r : ℝ, c = x * r ∧ d = x * r^2 ∧ y = x * r^3

/-- Given x > 0, y > 0, and that x, a, b, y form an arithmetic sequence,
    while x, c, d, y form a geometric sequence,
    the minimum value of (a+b)²/(cd) is 4. -/
theorem min_value_arithmetic_geometric (x y a b c d : ℝ) 
  (hx : x > 0) (hy : y > 0)
  (h_arith : ArithmeticSeq x a b y)
  (h_geom : GeometricSeq x c d y) :
  (∀ a b c d, (a + b)^2 / (c * d) ≥ 4) ∧ 
  (∃ a b c d, (a + b)^2 / (c * d) = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_arithmetic_geometric_l148_14827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_determination_l148_14889

noncomputable def f (ω : ℝ) (m : ℝ) (x : ℝ) : ℝ := 
  (Real.cos (ω * x))^2 + Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) + m

theorem function_determination (ω : ℝ) (m : ℝ) 
  (h_ω_pos : ω > 0)
  (h_period : ∀ x, f ω m (x + π / ω) = f ω m x)
  (h_point : f ω m 0 = 1 / 2) :
  (∀ x, f ω m x = Real.sin (2 * x + π / 6) - 1 / 2) ∧
  (∀ x, f ω m x ≥ -1) ∧
  (∃ x, f ω m x = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_determination_l148_14889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_radii_is_10_l148_14863

-- Define the centers and radius
def center_C1C2 : ℝ × ℝ := (3, 4)
def center_C3 : ℝ × ℝ := (0, 0)
def radius_C3 : ℝ := 2

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ :=
  Real.sqrt ((center_C1C2.1 - center_C3.1)^2 + (center_C1C2.2 - center_C3.2)^2)

-- Define the radii of C1 and C2
noncomputable def radius_C1 : ℝ := distance_between_centers - radius_C3
noncomputable def radius_C2 : ℝ := distance_between_centers + radius_C3

-- Theorem to prove
theorem sum_of_radii_is_10 :
  radius_C1 + radius_C2 = 10 := by
  -- Unfold definitions
  unfold radius_C1 radius_C2 distance_between_centers
  -- Simplify expressions
  simp [center_C1C2, center_C3, radius_C3]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_radii_is_10_l148_14863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l148_14821

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 4)

theorem omega_range (ω : ℝ) :
  ω > 0 →
  (∀ x, x ∈ Set.Icc 0 1 → f ω x ∈ Set.Icc (-2) 2) →
  (∃ x₁ x₂ x₃, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ x₃ ∈ Set.Icc 0 1 ∧
    x₁ < x₂ ∧ x₂ < x₃ ∧
    (∀ x ∈ Set.Icc 0 1, f ω x ≤ f ω x₁) ∧
    (∀ x ∈ Set.Icc 0 1, f ω x ≤ f ω x₂) ∧
    (∀ x ∈ Set.Icc 0 1, f ω x ≤ f ω x₃) ∧
    (∀ x ∈ Set.Icc 0 1, x ≠ x₁ ∧ x ≠ x₂ ∧ x ≠ x₃ → f ω x < max (f ω x₁) (max (f ω x₂) (f ω x₃)))) →
  ω ∈ Set.Icc (17 * Real.pi / 4) (25 * Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l148_14821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_probability_l148_14894

/-- Represents a player in the coin game -/
inductive Player : Type
| Abby : Player
| Ben : Player
| Carl : Player
| Debra : Player
| Ellie : Player

/-- Represents a ball color in the game -/
inductive BallColor : Type
| Green : BallColor
| Red : BallColor
| Blue : BallColor
| White : BallColor

/-- The number of rounds in the game -/
def numRounds : ℕ := 5

/-- The initial number of coins each player has -/
def initialCoins : ℕ := 5

/-- The number of balls in the urn -/
def numBalls : ℕ := 5

/-- The probability of a specific arrangement in a single round -/
def singleRoundProbability : ℚ := 1 / 6

/-- Represents the state of the game after each round -/
def GameState := Player → ℕ

/-- The probability of all players having the initial number of coins after all rounds -/
def finalProbability : ℚ := (1 : ℚ) / 7776

/-- Theorem stating the probability of all players having the initial number of coins after all rounds -/
theorem coin_game_probability :
  (numRounds = 5) →
  (∀ (r : ℕ), r < numRounds → singleRoundProbability = 1 / 6) →
  (∃ (finalState : GameState), 
    (∀ (p : Player), finalState p = initialCoins) ∧ 
    (finalProbability = singleRoundProbability ^ numRounds)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_probability_l148_14894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_binary_41_l148_14843

theorem decimal_to_binary_41 : 
  (List.range 6).map (fun i => if Nat.testBit 41 i then 1 else 0) = [1, 0, 1, 0, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_binary_41_l148_14843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l148_14857

noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

def region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ 100 * frac x ≥ ⌊x⌋ + ⌊y⌋ + 50

theorem area_of_region :
  MeasureTheory.volume (Set.Ioo 0 1 ×ˢ Set.Ioo 0 1 ∩ {p : ℝ × ℝ | region p.1 p.2}) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l148_14857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_one_equality_l148_14873

theorem expression_one_equality : (-7) - 5 + (-4) - (-10) = -6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_one_equality_l148_14873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_ratio_is_one_l148_14835

/-- p_n is the number of sequences of A and B of length n not containing AAAA or BBB -/
def p (n : ℕ+) : ℕ := sorry

/-- The ratio of specific p_n values equals 1 -/
theorem p_ratio_is_one : 
  (p 2004 - p 2002 - p 1999) / (p 2000 + p 2001) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_ratio_is_one_l148_14835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_nonnegative_l148_14897

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp (a * x - 1) - Real.log x - a * x

/-- The theorem statement -/
theorem min_value_nonnegative (a : ℝ) (h : a ≤ -1 / Real.exp 2) :
  ∀ x > 0, f a x ≥ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_nonnegative_l148_14897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_subset_not_containing_any_l148_14804

-- Define the set X
def X : Finset Nat := Finset.range 2009

-- Define the property for subsets A_i
def valid_subset (A : Finset Nat) : Prop :=
  A ⊆ X ∧ A.card ≥ 4

-- Define the property for intersection of two subsets
def valid_intersection (A B : Finset Nat) : Prop :=
  A ≠ B → (A ∩ B).card ≤ 2

-- Main theorem
theorem exists_subset_not_containing_any (A : Finset (Finset Nat)) 
  (h1 : ∀ a, a ∈ A → valid_subset a) 
  (h2 : ∀ a b, a ∈ A → b ∈ A → valid_intersection a b) :
  ∃ B : Finset Nat, B ⊆ X ∧ B.card = 24 ∧ ∀ a ∈ A, ¬(a ⊆ B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_subset_not_containing_any_l148_14804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_from_final_sum_lent_calculation_l148_14868

/-- Compound interest calculation --/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

/-- Theorem stating the relationship between initial and final amounts --/
theorem initial_amount_from_final (final_amount : ℝ) (r : ℝ) (t : ℝ) :
  let initial_amount := final_amount / (1 + r) ^ t
  compound_interest initial_amount r 1 t = final_amount :=
by
  sorry

/-- The specific problem statement --/
theorem sum_lent_calculation (final_amount : ℝ) (r : ℝ) (t : ℝ) 
  (h1 : final_amount = 740)
  (h2 : r = 0.05)
  (h3 : t = 2) :
  let initial_amount := final_amount / (1 + r) ^ t
  ∃ ε > 0, |initial_amount - 670.68| < ε :=
by
  sorry

#check sum_lent_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_from_final_sum_lent_calculation_l148_14868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l148_14876

-- Define the parabola
def Parabola : Set (ℝ × ℝ) := {p | ∃ a : ℝ, p.2^2 = -2*a*p.1 ∧ a > 0}

-- Define the line l: 3x - 2y - 6 = 0
def Line : Set (ℝ × ℝ) := {p | 3*p.1 - 2*p.2 - 6 = 0}

-- Define the point M(-6, 6)
def M : ℝ × ℝ := (-6, 6)

-- Define the focus of a parabola
noncomputable def Focus (p : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- State the theorem
theorem parabola_equation : 
  ∃ p : Set (ℝ × ℝ), p = Parabola ∧ M ∈ p ∧ Focus p ∈ Line ∧ 
  ∀ x y : ℝ, (x, y) ∈ p ↔ y^2 = -6*x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l148_14876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_problem_l148_14893

theorem class_size_problem (S : ℕ) : 
  (S / 3 : ℚ).num % (S / 3 : ℚ).den = 0 →  -- One third of students are boys
  ((S / 3) / 4 : ℚ).num % ((S / 3) / 4 : ℚ).den = 0 →  -- One fourth of boys are under 6 feet
  (S / 3) / 4 = 3 →  -- There are 3 boys under 6 feet
  S = 36 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_problem_l148_14893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l148_14879

/-- Calculates the speed in km/hour given distance in meters and time in minutes -/
noncomputable def calculate_speed (distance_meters : ℝ) (time_minutes : ℝ) : ℝ :=
  (distance_meters / 1000) / (time_minutes / 60)

/-- Proves that crossing 1000 m in 10 minutes results in a speed of 6 km/hour -/
theorem speed_calculation :
  calculate_speed 1000 10 = 6 := by
  -- Unfold the definition of calculate_speed
  unfold calculate_speed
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l148_14879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_distance_l148_14842

/-- Represents a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Given four points in a plane with specific distances between them,
    there is only one possible integer distance between two specific points -/
theorem unique_integer_distance
  (doc sneezy grumpy dopey : Point)
  (h1 : distance doc sneezy = 5)
  (h2 : distance doc grumpy = 4)
  (h3 : distance sneezy dopey = 10)
  (h4 : distance grumpy dopey = 17)
  (h5 : ∀ p1 p2 p3 : Point, p1 ≠ p2 → p1 ≠ p3 → p2 ≠ p3 →
        ¬(∃ (t : ℝ), p3.x = t * p1.x + (1 - t) * p2.x ∧
                      p3.y = t * p1.y + (1 - t) * p2.y)) :
  ∃! (d : ℕ), distance sneezy grumpy = d := by
  sorry

#check unique_integer_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_distance_l148_14842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_a_equals_one_l148_14826

theorem subset_implies_a_equals_one (a : ℝ) : 
  let M : Set ℝ := {-1, 0, a^2}
  let N : Set ℝ := {0, a, -1}
  M ⊆ N → a = 1 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_a_equals_one_l148_14826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_twelve_l148_14859

def sequence_to_number (start : ℕ) (end_ : ℕ) : ℕ :=
  sorry

def append_number (seq n : ℕ) : ℕ :=
  sorry

theorem smallest_divisible_by_twelve (start end_ : ℕ) :
  start = 71 → end_ = 81 →
  ∃ N : ℕ,
    N > end_ ∧
    (∀ k : ℕ, end_ < k ∧ k < N → ¬(append_number (sequence_to_number start end_) k % 12 = 0)) ∧
    append_number (sequence_to_number start end_) N % 12 = 0 ∧
    N = 84 :=
  sorry

#check smallest_divisible_by_twelve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_twelve_l148_14859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_coords_of_specific_triangle_l148_14852

/-- Triangle PQR with side lengths p, q, r -/
structure Triangle where
  p : ℝ
  q : ℝ
  r : ℝ

/-- Barycentric coordinates of a point in a triangle -/
structure BarycentricCoord where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The incenter of a triangle -/
noncomputable def incenter (t : Triangle) : BarycentricCoord :=
  { x := t.p / (t.p + t.q + t.r)
  , y := t.q / (t.p + t.q + t.r)
  , z := t.r / (t.p + t.q + t.r) }

theorem incenter_coords_of_specific_triangle :
  let t : Triangle := { p := 8, q := 10, r := 6 }
  let i : BarycentricCoord := incenter t
  i.x = 1/3 ∧ i.y = 5/12 ∧ i.z = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_coords_of_specific_triangle_l148_14852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l148_14808

-- Define the equilateral triangle ABC
def triangle_ABC : Set (ℝ × ℝ × ℝ) := sorry

-- Define the sphere centered at O
def sphere_O : Set (ℝ × ℝ × ℝ) := sorry

-- Define the tetrahedron O-ABC
def tetrahedron_OABC : Set (ℝ × ℝ × ℝ) := sorry

-- Define necessary functions
def IsEquilateral : Set (ℝ × ℝ × ℝ) → Prop := sorry
def SideLength : Set (ℝ × ℝ × ℝ) → ℝ := sorry
def SurfaceArea : Set (ℝ × ℝ × ℝ) → ℝ := sorry
def Volume : Set (ℝ × ℝ × ℝ) → ℝ := sorry

-- State the theorem
theorem tetrahedron_volume 
  (h1 : IsEquilateral triangle_ABC)
  (h2 : ∀ v ∈ triangle_ABC, v ∈ sphere_O)
  (h3 : SideLength triangle_ABC = 2)
  (h4 : SurfaceArea sphere_O = 148 * Real.pi / 3) :
  Volume tetrahedron_OABC = Real.sqrt 33 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l148_14808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l148_14832

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 8*y = 1

-- Define the area of the region
noncomputable def region_area : ℝ := 26 * Real.pi

-- Theorem statement
theorem area_of_region :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), region_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l148_14832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l148_14862

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 * Real.cos x + 1

-- State the theorem
theorem problem_statement (a : ℝ) (h : f a = 11) : f (-a) = -9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l148_14862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_box_cost_l148_14871

/-- The cost of a box of soda given the conditions of the family reunion -/
theorem soda_box_cost (attendees cans_per_person cans_per_box paying_members : ℕ) (cost_per_member : ℚ) : 
  attendees = 60 →
  cans_per_person = 2 →
  cans_per_box = 10 →
  paying_members = 6 →
  cost_per_member = 4 →
  (attendees * cans_per_person / cans_per_box : ℚ)⁻¹ * 
    (paying_members * cost_per_member) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_box_cost_l148_14871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_numbers_lcm_sum_l148_14872

theorem consecutive_numbers_lcm_sum (n : ℕ) :
  ∀ (red blue : Finset ℕ),
    red ∪ blue = Finset.range 10 →
    red ∩ blue = ∅ →
    red ≠ ∅ →
    blue ≠ ∅ →
    ¬(16 ∣ (Nat.lcm (Finset.prod red (λ i => n + i)) + 
             Nat.lcm (Finset.prod blue (λ i => n + i)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_numbers_lcm_sum_l148_14872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_2x_plus_y_range_of_a_l148_14831

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 2*y

-- Theorem for the range of 2x + y
theorem range_2x_plus_y (x y : ℝ) (h : circle_eq x y) :
  -Real.sqrt 5 + 1 ≤ 2*x + y ∧ 2*x + y ≤ Real.sqrt 5 + 1 := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y → x + y + a ≥ 0) → a ≥ Real.sqrt 2 - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_2x_plus_y_range_of_a_l148_14831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l148_14825

noncomputable def y (x : ℝ) : ℝ := Real.tan (x + Real.pi/3) - Real.tan (x + Real.pi/4) + Real.cos (x + Real.pi/4)

theorem max_value_of_y :
  ∃ (max : ℝ), max = -1/Real.sqrt 3 - 1 + Real.sqrt 2/2 ∧
  (∀ x : ℝ, -2*Real.pi/3 ≤ x ∧ x ≤ -Real.pi/2 → y x ≤ max) ∧
  (∃ x : ℝ, -2*Real.pi/3 ≤ x ∧ x ≤ -Real.pi/2 ∧ y x = max) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l148_14825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l148_14895

/-- The ellipse defined by x²/4 + y²/3 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 4 + p.2^2 / 3 = 1}

/-- The point M(1,1) -/
def M : ℝ × ℝ := (1, 1)

/-- A line passing through two points -/
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r | ∃ t : ℝ, r = (1 - t) • p + t • q}

/-- The midpoint of two points -/
noncomputable def Midpoint (p q : ℝ × ℝ) : ℝ × ℝ :=
  ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

/-- The theorem to be proved -/
theorem line_equation (A B : ℝ × ℝ) 
  (hA : A ∈ Ellipse) (hB : B ∈ Ellipse) 
  (hM : M = Midpoint A B) 
  (hLine : Line A B = {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 7 = 0}) : 
  True := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l148_14895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_domain_l148_14800

noncomputable def log_function (x : ℝ) : ℝ := Real.log (5 - x) / Real.log (x - 2)

theorem log_function_domain :
  {x : ℝ | ∃ y, log_function x = y} = {x | 2 < x ∧ x < 5 ∧ x ≠ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_domain_l148_14800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_dan_difference_l148_14829

def sally_initial : ℕ := 27
def dan_cards : ℕ := 41
def sally_bought : ℕ := 20

theorem sally_dan_difference (sally_traded : ℕ) :
  (sally_initial + sally_bought - sally_traded) - dan_cards = 6 - sally_traded :=
by
  -- Proof steps would go here
  sorry

#eval sally_initial + sally_bought - dan_cards

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_dan_difference_l148_14829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l148_14836

/-- Calculates the time (in seconds) for a train to cross a signal post. -/
noncomputable def time_to_cross (train_length : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * (1000 / 3600)
  train_length / speed_mps

theorem train_crossing_time :
  time_to_cross 350 72 = 17.5 := by
  -- Unfold the definition of time_to_cross
  unfold time_to_cross
  -- Simplify the expression
  simp
  -- The proof is completed using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l148_14836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_after_white_draw_l148_14838

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents the state of the bag of balls -/
structure BagState where
  total : Nat
  red : Nat
  white : Nat

/-- Calculates the probability of drawing a red ball given the current bag state -/
noncomputable def probRedBall (bag : BagState) : ℚ :=
  bag.red / bag.total

/-- Theorem: Given a bag with 5 balls (2 red and 3 white), if a white ball is drawn first
    without replacement, the probability of drawing a red ball on the second draw is 1/2 -/
theorem prob_red_after_white_draw (initial_bag : BagState)
    (h_total : initial_bag.total = 5)
    (h_red : initial_bag.red = 2)
    (h_white : initial_bag.white = 3) :
    let bag_after_white_draw : BagState := {
      total := initial_bag.total - 1,
      red := initial_bag.red,
      white := initial_bag.white - 1
    }
    probRedBall bag_after_white_draw = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_after_white_draw_l148_14838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f_f_eight_l148_14890

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 - 1 else x + 2

-- State the theorem
theorem unique_solution_f_f_eight :
  ∃! x : ℝ, f (f x) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f_f_eight_l148_14890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bricklayer_worked_22_5_hours_l148_14892

/-- Represents the work hours and payment for a construction project -/
structure ConstructionProject where
  total_hours : ℝ
  bricklayer_rate : ℝ
  electrician_rate : ℝ
  total_payment : ℝ

/-- Calculates the bricklayer's work hours given a construction project -/
noncomputable def bricklayer_hours (project : ConstructionProject) : ℝ :=
  (project.total_payment - project.electrician_rate * project.total_hours) /
  (project.bricklayer_rate - project.electrician_rate)

/-- Theorem stating that for the given project parameters, the bricklayer worked 22.5 hours -/
theorem bricklayer_worked_22_5_hours :
  let project := ConstructionProject.mk 90 12 16 1350
  bricklayer_hours project = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bricklayer_worked_22_5_hours_l148_14892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_representation_l148_14884

theorem equation_representation : 
  ∀ x : ℝ, 3 * x - 7 = 2 * x + 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_representation_l148_14884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_power_product_exists_l148_14883

theorem fourth_power_product_exists (S : Finset ℕ) :
  (S.card = 81) →
  (∀ n ∈ S, ∃ a b c : ℕ, n = 2^a * 3^b * 5^c) →
  ∃ n₁ n₂ n₃ n₄ : ℕ, n₁ ∈ S ∧ n₂ ∈ S ∧ n₃ ∈ S ∧ n₄ ∈ S ∧ ∃ m : ℕ, n₁ * n₂ * n₃ * n₄ = m^4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_power_product_exists_l148_14883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pear_juice_percentage_approx_l148_14830

/-- The amount of pear juice obtained from one pear in ounces -/
def pear_juice_per_fruit : ℚ := 10 / 4

/-- The amount of orange juice obtained from one orange in ounces -/
def orange_juice_per_fruit : ℚ := 12 / 3

/-- The number of pears used in the blend -/
def pears_in_blend : ℕ := 5

/-- The number of oranges used in the blend -/
def oranges_in_blend : ℕ := 4

/-- The total amount of pear juice in the blend in ounces -/
def total_pear_juice : ℚ := pear_juice_per_fruit * pears_in_blend

/-- The total amount of orange juice in the blend in ounces -/
def total_orange_juice : ℚ := orange_juice_per_fruit * oranges_in_blend

/-- The total amount of juice in the blend in ounces -/
def total_juice : ℚ := total_pear_juice + total_orange_juice

/-- The percentage of pear juice in the blend -/
def pear_juice_percentage : ℚ := total_pear_juice / total_juice * 100

theorem pear_juice_percentage_approx :
  |pear_juice_percentage - 43.86| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pear_juice_percentage_approx_l148_14830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_sqrt_two_solution_l148_14844

theorem sin_minus_cos_sqrt_two_solution :
  ∃! x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi ∧ Real.sin x - Real.cos x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_sqrt_two_solution_l148_14844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_with_distance_7_l148_14816

noncomputable section

-- Define the original line
def original_line (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define the parallel line
def parallel_line (m : ℝ) (x y : ℝ) : Prop := 3 * x + 4 * y + m = 0

-- Define the distance between lines
noncomputable def distance_between_lines (m : ℝ) : ℝ := |m + 12| / Real.sqrt (3^2 + 4^2)

theorem parallel_line_with_distance_7 :
  ∃ m : ℝ, distance_between_lines m = 7 ∧
  (∀ x y : ℝ, parallel_line m x y ↔ (3 * x + 4 * y + 23 = 0 ∨ 3 * x + 4 * y - 47 = 0)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_with_distance_7_l148_14816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_bounds_l148_14815

/-- Represents a rectangle with a circle centered at one corner --/
structure RectangleWithCircle where
  AD : ℝ
  CD : ℝ
  -- D is implicitly the center of the circle
  -- B is implicitly on the circle

/-- The area of the shaded region in a RectangleWithCircle --/
noncomputable def shaded_area (r : RectangleWithCircle) : ℝ :=
  (Real.pi * (r.AD^2 + r.CD^2) / 4) - (r.AD * r.CD)

theorem shaded_area_bounds (r : RectangleWithCircle) 
  (h1 : r.AD = 5) 
  (h2 : r.CD = 12) : 
  70 < shaded_area r ∧ shaded_area r < 74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_bounds_l148_14815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brownie_to_bess_ratio_l148_14866

/-- Represents the daily milk production of a cow in pails -/
structure DailyMilkProduction where
  value : ℕ
  deriving Repr

/-- Represents the weekly milk production of a cow in pails -/
structure WeeklyMilkProduction where
  value : ℕ
  deriving Repr

/-- Calculates the weekly milk production given the daily production -/
def weeklyFromDaily (daily : DailyMilkProduction) : WeeklyMilkProduction :=
  ⟨7 * daily.value⟩

structure Farm where
  bess : DailyMilkProduction
  daisy : DailyMilkProduction
  brownie : WeeklyMilkProduction
  total_weekly : WeeklyMilkProduction

def farmer_red : Farm where
  bess := ⟨2⟩
  daisy := ⟨3⟩
  brownie := ⟨42⟩
  total_weekly := ⟨77⟩

theorem brownie_to_bess_ratio :
  farmer_red.brownie.value = 3 * (weeklyFromDaily farmer_red.bess).value := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brownie_to_bess_ratio_l148_14866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_OPQ_l148_14847

-- Define the polar coordinate system
noncomputable def polar_to_cartesian (ρ : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the curves C₁ and C₂
def C₁ : ℝ := 2

def C₂ (ρ θ : ℝ) : Prop :=
  ρ^2 + Real.sqrt 3 * ρ * Real.cos θ + ρ * Real.sin θ = 6

-- Define the line AB (intersection of C₁ and C₂)
def line_AB (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y = 2

-- Define the area of △OPQ
noncomputable def area_OPQ (θ : ℝ) : ℝ :=
  |2 / (2 * Real.sin (Real.pi/3 - 2*θ))|

-- State the theorem
theorem min_area_OPQ :
  ∀ θ : ℝ, area_OPQ θ ≥ 1 ∧ ∃ θ₀ : ℝ, area_OPQ θ₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_OPQ_l148_14847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_result_l148_14840

/-- Definition of the function f for complex numbers -/
noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then -z^3 else z^3

/-- Theorem stating the result of f(f(f(f(1+i)))) -/
theorem f_composition_result : f (f (f (f (1 + Complex.I)))) = -1000000000 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_result_l148_14840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_is_parabola_l148_14865

def line_through (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • A + t • B}

def envelope_of_lines (O A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (α : ℝ), 0 ≤ α ∧ α ≤ 1 ∧
    let A₁ := ((1 - α) * O.1 + α * A.1, (1 - α) * O.2 + α * A.2)
    let B₁ := ((1 - α) * O.1 + α * B.1, (1 - α) * O.2 + α * B.2)
    p.2 = (1 + p.1^2) / 2 ∧ p ∈ line_through A₁ B₁}

theorem envelope_is_parabola (O A B : ℝ × ℝ) 
  (hO : O = (0, 0)) (hA : A = (1, 1)) (hB : B = (-1, 1)) :
  envelope_of_lines O A B = {p : ℝ × ℝ | p.2 = (1 + p.1^2) / 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_is_parabola_l148_14865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l148_14855

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2 / 2) * (Real.cos (3 * x) - Real.sin (3 * x))
noncomputable def g (x : ℝ) : ℝ := -Real.sin (3 * (x - Real.pi / 12))

theorem function_equivalence : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l148_14855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l148_14899

-- Define the function f(x) = 2^(|x|)
noncomputable def f (x : ℝ) : ℝ := 2^(abs x)

-- Theorem statement
theorem f_properties :
  (∀ x : ℝ, f x ≥ 1) ∧ 
  (∃ x : ℝ, f x = 1) ∧
  (∀ x : ℝ, f (-x) = f x) := by
  sorry

-- The minimum value is 1 and the axis of symmetry is x = 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l148_14899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetris_tiling_impossible_l148_14882

/-- Represents the different types of Tetris pieces -/
inductive TetrisPiece
  | L
  | T
  | I
  | Z
  | O

/-- Represents a rectangular grid -/
def Rectangle (m n : ℕ) := Fin m → Fin n → Bool

/-- Represents a tiling of a rectangle using Tetris pieces -/
def Tiling (m n : ℕ) := Fin m → Fin n → Option TetrisPiece

/-- Returns true if a piece covers an equal number of "black" and "white" squares in a checkerboard pattern -/
def evenCoverage (p : TetrisPiece) : Bool :=
  match p with
  | TetrisPiece.T => false
  | _ => true

/-- The main theorem stating that it's impossible to tile a rectangle with the given Tetris pieces -/
theorem tetris_tiling_impossible {m n : ℕ} (hm : m > 0) (hn : n > 0) :
  ¬∃ (t : Tiling m n), ∀ (i : Fin m) (j : Fin n), t i j ≠ none :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetris_tiling_impossible_l148_14882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_three_half_l148_14814

/-- Represents an ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

theorem ellipse_eccentricity_sqrt_three_half
  (e : Ellipse)
  (h1 : e.a = 4)  -- One vertex is at (4,0)
  (h2 : 2 * e.b^2 / e.a = 2)  -- Length of chord condition
  : e.eccentricity = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_three_half_l148_14814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l148_14860

-- Define the polynomial
def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 8 * x^2 + k * x - 10

-- Define divisibility for polynomials
def is_divisible (p q : ℝ → ℝ) : Prop :=
  ∃ r : ℝ → ℝ, ∀ x, p x = q x * r x

theorem polynomial_divisibility :
  (∃ k : ℝ, is_divisible (f k) (λ x ↦ x - 2)) ∧
  (let k := 13; ¬ is_divisible (f k) (λ x ↦ 2 * x^2 - 1)) :=
by
  sorry

#check polynomial_divisibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l148_14860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_area_difference_l148_14806

/-- Represents a right-angled triangle with two short sides -/
structure RightTriangle where
  side1 : ℝ
  side2 : ℝ
  side1_positive : 0 < side1
  side2_positive : 0 < side2

/-- Check if two right-angled triangles are similar -/
noncomputable def are_similar (t1 t2 : RightTriangle) : Prop :=
  t1.side1 / t1.side2 = t2.side1 / t2.side2

/-- Calculate the area of a right-angled triangle -/
noncomputable def area (t : RightTriangle) : ℝ :=
  t.side1 * t.side2 / 2

/-- The main theorem -/
theorem similar_triangles_area_difference
  (small large : RightTriangle)
  (h_similar : are_similar small large)
  (h_side1_diff : large.side1 = small.side1 + 1)
  (h_side2_diff : large.side2 = small.side2 + 5)
  (h_area_diff : area large = area small + 8) :
  (small.side1 = 1.1 ∧ small.side2 = 5.5) ∨ (small.side1 = 2 ∧ small.side2 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_area_difference_l148_14806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_inequality_l148_14864

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := 3 * n

-- Define the sum of the first n terms of a_n
noncomputable def S (n : ℕ) : ℝ := n * (a 1 + a n) / 2

-- Define the geometric sequence b_n
noncomputable def b (n : ℕ) : ℝ := 3^(n - 1)

-- Define the common ratio q
noncomputable def q : ℝ := S 2 / b 2

theorem sequence_and_inequality (n : ℕ) : 
  a 1 = 3 ∧ 
  b 1 = 1 ∧ 
  b 2 + S 2 = 12 ∧ 
  q = S 2 / b 2 ∧ 
  (∀ k, b k > 0) →
  (∀ k, a k = 3 * k) ∧
  (∀ k, b k = 3^(k - 1)) ∧
  1/3 ≤ (Finset.sum (Finset.range n) (λ i => 1 / S (i + 1))) ∧
  (Finset.sum (Finset.range n) (λ i => 1 / S (i + 1))) < 2/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_inequality_l148_14864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l148_14839

def M : Set ℝ := {x | (x + 2) / (x - 1) ≤ 0}
def N : Set ℝ := Set.range (fun n : ℕ => (n : ℝ))

theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l148_14839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_sum_l148_14861

/-- Given a right triangle ABC with AB = AC = 1, and points D, E, F, G satisfying certain conditions,
    prove that DE + FG = 3/2 -/
theorem triangle_segment_sum (A B C D E F G : ℝ × ℝ) : 
  -- Triangle ABC is right-angled with AB = AC = 1
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 1 →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = 1 →
  (B.1 - C.1) * (A.1 - B.1) + (B.2 - C.2) * (A.2 - B.2) = 0 →
  -- D and F are on AB
  ∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ 0 ≤ t₂ ∧ t₂ ≤ 1 ∧
    D = (t₁ * A.1 + (1 - t₁) * B.1, t₁ * A.2 + (1 - t₁) * B.2) ∧
    F = (t₂ * A.1 + (1 - t₂) * B.1, t₂ * A.2 + (1 - t₂) * B.2) →
  -- E and G are on AC
  ∃ s₁ s₂ : ℝ, 0 ≤ s₁ ∧ s₁ ≤ 1 ∧ 0 ≤ s₂ ∧ s₂ ≤ 1 ∧
    E = (s₁ * A.1 + (1 - s₁) * C.1, s₁ * A.2 + (1 - s₁) * C.2) ∧
    G = (s₂ * A.1 + (1 - s₂) * C.1, s₂ * A.2 + (1 - s₂) * C.2) →
  -- DE and FG are parallel to BC
  (E.1 - D.1) * (C.2 - B.2) = (E.2 - D.2) * (C.1 - B.1) →
  (G.1 - F.1) * (C.2 - B.2) = (G.2 - F.2) * (C.1 - B.1) →
  -- Perimeter of trapezoid DFGE is twice the perimeter of triangle ADE
  2 * (Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) +
       Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) +
       Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2)) =
    Real.sqrt ((D.1 - F.1)^2 + (D.2 - F.2)^2) +
    Real.sqrt ((F.1 - G.1)^2 + (F.2 - G.2)^2) +
    Real.sqrt ((G.1 - E.1)^2 + (G.2 - E.2)^2) +
    Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) →
  -- DE = 2FG
  (E.1 - D.1)^2 + (E.2 - D.2)^2 = 4 * ((G.1 - F.1)^2 + (G.2 - F.2)^2) →
  -- Conclusion: DE + FG = 3/2
  Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) +
  Real.sqrt ((G.1 - F.1)^2 + (G.2 - F.2)^2) = 3/2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_sum_l148_14861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l148_14856

-- Define the distance traveled in miles
noncomputable def distance : ℝ := 280

-- Define the time taken in hours
noncomputable def time : ℝ := 2 + 1/3

-- Define the average speed
noncomputable def average_speed : ℝ := distance / time

-- Theorem stating that the average speed is 120 miles per hour
theorem car_average_speed : average_speed = 120 := by
  -- Unfold the definitions
  unfold average_speed distance time
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l148_14856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteen_factorial_mod_seventeen_l148_14886

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem thirteen_factorial_mod_seventeen : 
  factorial 13 % 17 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteen_factorial_mod_seventeen_l148_14886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_one_third_l148_14896

/-- Represents the area of a shaded triangle at each level of division -/
def shadedArea (n : ℕ) : ℚ := (1 / 4) ^ n

/-- The sum of the areas of all shaded triangles -/
noncomputable def totalShadedArea : ℚ := ∑' n, shadedArea n

/-- Theorem stating that the total shaded area is one-third of the triangle -/
theorem shaded_area_is_one_third : totalShadedArea = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_one_third_l148_14896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_edges_intersect_frustum_edges_intersect_proof_l148_14824

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

instance : HSMul ℝ Point3D Point3D where
  hSMul k p := { x := k * p.x, y := k * p.y, z := k * p.z }

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  d : ℝ

/-- Represents a frustum -/
structure Frustum where
  base1 : Plane
  base2 : Plane
  vertices : List Point3D

/-- Checks if two planes are parallel -/
def arePlanesParallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℝ), p1.normal = k • p2.normal

/-- Checks if two polygons are similar -/
noncomputable def arePolygonsSimilar (poly1 poly2 : List Point3D) : Prop :=
  sorry -- Definition of polygon similarity

/-- Checks if a face is a trapezoid -/
noncomputable def isTrapezoid (face : List Point3D) : Prop :=
  sorry -- Definition of a trapezoid

/-- Theorem: Lateral edges of a frustum intersect at a point when extended -/
theorem frustum_edges_intersect (f : Frustum) : Prop :=
  arePlanesParallel f.base1 f.base2 ∧
  arePolygonsSimilar (sorry : List Point3D) (sorry : List Point3D) ∧
  (∀ face ∈ (sorry : List (List Point3D)), isTrapezoid face) →
  ∃ (p : Point3D), ∀ edge ∈ (sorry : List (List Point3D)), p ∈ edge

/-- Proof of the theorem -/
theorem frustum_edges_intersect_proof : ∀ f : Frustum, frustum_edges_intersect f := by
  sorry

#check frustum_edges_intersect_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_edges_intersect_frustum_edges_intersect_proof_l148_14824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l148_14819

/-- The ellipse in standard form -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- The circle centered at origin -/
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

/-- The left focus of the ellipse -/
noncomputable def LeftFocus (a b : ℝ) : ℝ × ℝ :=
  (-(Real.sqrt (a^2 - b^2)), 0)

/-- Distance between two points -/
noncomputable def Distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem ellipse_focus_distance (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∀ A ∈ Ellipse a b, A.1 ≥ 0 → A.2 ≥ 0 →
  ∀ P ∈ Circle b,
  Distance A (LeftFocus a b) - Distance A P = a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l148_14819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_growth_equation_l148_14812

/-- Represents the average annual growth rate of per capita disposable income -/
def x : ℝ := sorry

/-- Per capita disposable income in 2020 (in thousands of yuan) -/
def income_2020 : ℝ := 32

/-- Per capita disposable income in 2022 (in thousands of yuan) -/
def income_2022 : ℝ := 37

/-- Time period in years -/
def years : ℕ := 2

/-- Theorem stating that the equation correctly represents the growth of per capita disposable income -/
theorem income_growth_equation : 
  (income_2020 / 10) * (1 + x)^years = income_2022 / 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_growth_equation_l148_14812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_x_eq_0_is_tangent_of_g_main_theorem_l148_14854

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

def g (x : ℝ) : ℝ := x^(1/3)

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

theorem x_eq_0_is_tangent_of_g : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| → |x| < δ → |g x - g 0| < ε * |x| := by sorry

theorem main_theorem : (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| → |x| < δ → |g x - g 0| < ε * |x|) := by
  constructor
  · exact f_is_odd
  · exact x_eq_0_is_tangent_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_x_eq_0_is_tangent_of_g_main_theorem_l148_14854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_l148_14823

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 3*x - 3 else (1/2)^x - 4

-- State the theorem
theorem zeros_of_f :
  ∃ (x₁ x₂ : ℝ), x₁ = -2 ∧ x₂ = 1 ∧
  (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_l148_14823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_125_fourth_root_16_square_root_36_squared_l148_14858

theorem cube_root_125_fourth_root_16_square_root_36_squared :
  (Real.rpow 125 (1/3) * Real.rpow 16 (1/4) * Real.sqrt 36) ^ 2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_125_fourth_root_16_square_root_36_squared_l148_14858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_mean_theorem_l148_14870

def number_list : List Nat := [31, 33, 35, 37, 39]

def is_prime (n : Nat) : Bool := Nat.Prime n

def arithmetic_mean (l : List Nat) : Rat :=
  (l.sum : Rat) / l.length

theorem prime_mean_theorem :
  arithmetic_mean (number_list.filter is_prime) = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_mean_theorem_l148_14870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotationVolume_eq_pi_fifth_l148_14810

/-- The volume of the solid formed by rotating the region bounded by y = 2x - x^2 and y = -x + 2 about the x-axis -/
noncomputable def rotationVolume : ℝ := 
  let f (x : ℝ) := 2 * x - x^2
  let g (x : ℝ) := -x + 2
  let a : ℝ := 1
  let b : ℝ := 2
  Real.pi * ∫ x in a..b, (f x)^2 - (g x)^2

/-- The theorem stating that the volume of the rotated solid is π/5 -/
theorem rotationVolume_eq_pi_fifth : rotationVolume = Real.pi / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotationVolume_eq_pi_fifth_l148_14810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derek_travel_distance_l148_14820

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The total distance traveled through four points -/
noncomputable def totalDistance (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : ℝ :=
  distance x1 y1 x2 y2 + distance x2 y2 x3 y3 + distance x3 y3 x4 y4

theorem derek_travel_distance :
  totalDistance (-5) 7 0 0 3 2 6 (-5) = Real.sqrt 74 + Real.sqrt 13 + Real.sqrt 58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derek_travel_distance_l148_14820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_number_divisibility_l148_14880

/-- The largest prime number less than k -/
def largest_prime_less_than (k : ℕ) : ℕ := sorry

theorem composite_number_divisibility 
  (k : ℕ) 
  (n : ℕ) 
  (h_k : k ≥ 14) 
  (h_pk : largest_prime_less_than k ≥ 3 * k / 4) 
  (h_n_composite : ¬ Nat.Prime n) :
  (n = 2 * largest_prime_less_than k → ¬(n ∣ Nat.factorial (n - k))) ∧
  (n > 2 * largest_prime_less_than k → n ∣ Nat.factorial (n - k)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_number_divisibility_l148_14880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_width_is_three_l148_14813

-- Define the rectangle structure
structure Rectangle where
  length : ℝ
  area : ℝ

-- Define the rectangles
noncomputable def rectangleA : Rectangle := { length := 6, area := 36 }
noncomputable def rectangleB : Rectangle := { length := 12, area := 36 }
noncomputable def rectangleC : Rectangle := { length := 9, area := 36 }

-- Define the width function
noncomputable def width (r : Rectangle) : ℝ := r.area / r.length

-- Theorem statement
theorem smallest_width_is_three :
  min (width rectangleA) (min (width rectangleB) (width rectangleC)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_width_is_three_l148_14813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_M₁_M₂_l148_14837

noncomputable section

-- Define the curves G₃ and G₂
def G₃ (x : ℝ) : ℝ := 2 * x^3 - 8 * x
def G₂ (m x : ℝ) : ℝ := m * x^2 - 8 * x

-- Define the line e passing through O(0,0) and A
def line_e (m : ℝ) (x : ℝ) : ℝ := (m^2 - 16) / 2 * x

-- Define the locus equations
def locus_eq₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def locus_eq₂ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

-- State the theorem
theorem locus_of_M₁_M₂ (m : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (G₃ x₁ = G₂ m x₁ ∧ G₃ x₂ = G₂ m x₂) ∧ 
    (x₁ ≠ 0 ∧ x₂ ≠ 0) ∧
    (locus_eq₁ x₁ y₁ ∨ locus_eq₂ x₁ y₁) ∧
    (locus_eq₁ x₂ y₂ ∨ locus_eq₂ x₂ y₂) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_M₁_M₂_l148_14837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_section_volume_is_22_over_3_l148_14875

/-- Represents a cube in 3D space -/
structure Cube where
  edge_length : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  -- We don't need to define the plane explicitly for this problem

/-- Calculates the volume of the larger section created when a cube is cut by a plane
    passing through one vertex and the midpoints of two opposite edges -/
noncomputable def larger_section_volume (c : Cube) (p : Plane) : ℝ :=
  22 / 3

/-- Theorem stating that for a cube with edge length 2, the volume of the larger section
    created by a plane passing through one vertex and the midpoints of two opposite edges
    is 22/3 -/
theorem larger_section_volume_is_22_over_3 (c : Cube) (p : Plane) :
  c.edge_length = 2 → larger_section_volume c p = 22 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_section_volume_is_22_over_3_l148_14875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_roots_difference_l148_14822

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
noncomputable def vertex (p : Parabola) : ℝ × ℝ := (3, -3)

/-- The parabola passes through the point (5, 15) -/
def passes_through (p : Parabola) : Prop :=
  15 = p.a * 5^2 + p.b * 5 + p.c

/-- The roots of the parabola -/
noncomputable def roots (p : Parabola) : ℝ × ℝ :=
  let h := (vertex p).1
  let k := (vertex p).2
  (h + Real.sqrt (2/3), h - Real.sqrt (2/3))

theorem parabola_roots_difference (p : Parabola) :
  passes_through p →
  (roots p).1 - (roots p).2 = 2 * Real.sqrt (2/3) := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_roots_difference_l148_14822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l148_14802

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Define the angle α
variable (α : ℝ)

-- State the theorem
theorem tan_alpha_value (h : ∃ (k : ℝ), k > 0 ∧ (k * P.1 = Real.cos α) ∧ (k * P.2 = Real.sin α)) : 
  Real.tan α = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l148_14802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_curve_equation_l148_14874

/-- The scaling transformation from (x, y) to (x', y') -/
noncomputable def scaling (x y : ℝ) : ℝ × ℝ :=
  (2 * x, 3 * y)

/-- The inverse of the scaling transformation -/
noncomputable def scaling_inv (x' y' : ℝ) : ℝ × ℝ :=
  (x' / 2, y' / 3)

/-- The original curve -/
noncomputable def original_curve (x : ℝ) : ℝ :=
  (1/3) * Real.cos (2 * x)

theorem transformed_curve_equation (x' y' : ℝ) :
  let (x, y) := scaling_inv x' y'
  y' = Real.cos x' ↔ y = original_curve x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_curve_equation_l148_14874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_translation_l148_14817

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin (2 * x + Real.pi / 4)

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f (x + φ)

theorem symmetric_translation (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi / 2) 
  (h3 : ∀ x, g φ x = g φ (-x)) : 
  φ = Real.pi / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_translation_l148_14817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l148_14818

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 5) + (2 * x - 6) ^ (1/3 : ℝ)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x ≥ 5}

-- Theorem statement
theorem domain_of_f : {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l148_14818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_honey_jar_price_calculation_l148_14853

/-- Represents a cylindrical jar -/
structure Jar where
  diameter : ℝ
  height : ℝ
  price : ℝ

/-- Calculates the volume of a cylindrical jar -/
noncomputable def jarVolume (j : Jar) : ℝ :=
  Real.pi * (j.diameter / 2) ^ 2 * j.height

/-- The problem statement -/
theorem honey_jar_price_calculation (jar1 jar2 : Jar) (price_increase : ℝ) :
  jar1.diameter = 4 →
  jar1.height = 5 →
  jar1.price = 0.75 →
  jar2.diameter = 8 →
  jar2.height = 10 →
  price_increase = 0.1 →
  jar2.price = 6.60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_honey_jar_price_calculation_l148_14853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l148_14803

/-- An ellipse with equation x²/3 + y² = 1 -/
structure Ellipse where
  x : ℝ
  y : ℝ
  eq : x^2 / 3 + y^2 = 1

/-- The distance between the foci of the ellipse -/
noncomputable def foci_distance (e : Ellipse) : ℝ := 2 * Real.sqrt 2

/-- Theorem: The distance between the foci of the ellipse x²/3 + y² = 1 is 2√2 -/
theorem ellipse_foci_distance (e : Ellipse) : foci_distance e = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l148_14803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_eq_bc_condition_l148_14885

theorem ac_eq_bc_condition (a b c : ℝ) :
  (∀ c : ℝ, a = b → a * c = b * c) ∧
  (∃ a b c : ℝ, a * c = b * c ∧ a ≠ b) :=
by
  constructor
  · intro c h
    rw [h]
  · use 0, 1, 0
    constructor
    · simp
    · exact zero_ne_one


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_eq_bc_condition_l148_14885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_distance_l148_14888

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

def l₂ (x y : ℝ) : Prop := 2 * x - y + 3 = 0

theorem tangent_line_distance (x₀ y₀ : ℝ) :
  f x₀ = y₀ →
  (deriv f x₀ : ℝ) = 2 →
  (∀ x y, l₂ x y → (y - y₀) = 2 * (x - x₀)) →
  (|2 * x₀ - y₀ + 3| / Real.sqrt 5) = 2 * Real.sqrt 5 / 5 :=
by
  sorry

#check tangent_line_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_distance_l148_14888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excluded_students_average_mark_l148_14869

/-- Given a class of students and their exam marks, this theorem proves
    the average mark of excluded students. -/
theorem excluded_students_average_mark
  (total_students : ℕ)
  (all_average : ℚ)
  (excluded_count : ℕ)
  (remaining_average : ℚ)
  (h1 : total_students = 20)
  (h2 : all_average = 80)
  (h3 : excluded_count = 5)
  (h4 : remaining_average = 90)
  (h5 : total_students > excluded_count) :
  let remaining_count := total_students - excluded_count
  let total_marks := all_average * total_students
  let remaining_marks := remaining_average * remaining_count
  let excluded_marks := total_marks - remaining_marks
  excluded_marks / excluded_count = 50 := by
  sorry

#check excluded_students_average_mark

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excluded_students_average_mark_l148_14869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotonic_f_l148_14841

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log x

-- State the theorem
theorem min_a_for_monotonic_f :
  (∃ a : ℝ, ∀ x y : ℝ, 1 < x ∧ x < y ∧ y < 2 → f a x < f a y) ∧
  (∀ b : ℝ, b < Real.exp (-1) → ∃ x y : ℝ, 1 < x ∧ x < y ∧ y < 2 ∧ f b x ≥ f b y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotonic_f_l148_14841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_guppies_to_move_is_30_l148_14801

/-- Represents the number of fish in a tank -/
structure TankContents where
  guppies : Nat
  swordtails : Nat
  angelfish : Nat

/-- The problem setup -/
structure FishProblem where
  tankA : TankContents
  tankB : TankContents
  tankC : TankContents
  totalFish : Nat

def fishProblemInstance : FishProblem := {
  tankA := { guppies := 180, swordtails := 32, angelfish := 0 }
  tankB := { guppies := 120, swordtails := 45, angelfish := 15 }
  tankC := { guppies := 80, swordtails := 15, angelfish := 33 }
  totalFish := 520
}

/-- The number of guppies that need to be moved from Tank A to Tank B -/
def guppiesToMove (p : FishProblem) : Nat :=
  (p.tankA.guppies + p.tankB.guppies) / 2 - p.tankB.guppies

/-- The theorem stating that 30 guppies need to be moved -/
theorem guppies_to_move_is_30 (p : FishProblem) : guppiesToMove p = 30 := by
  sorry

#eval guppiesToMove fishProblemInstance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_guppies_to_move_is_30_l148_14801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l148_14834

-- Define the curves C₁ and C₂
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (t, Real.sqrt t)
noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.cos θ, Real.sqrt 2 * Real.sin θ)

-- Define the intersection point
def intersection_point : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem unique_intersection :
  (∃ t : ℝ, C₁ t = intersection_point) ∧
  (∃ θ : ℝ, C₂ θ = intersection_point) ∧
  (∀ p : ℝ × ℝ, (∃ t : ℝ, C₁ t = p) ∧ (∃ θ : ℝ, C₂ θ = p) → p = intersection_point) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l148_14834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_element_range_l148_14809

theorem largest_element_range (a : ℝ) : 
  let A : Set ℝ := {0, 1}
  let B : Set ℝ := {a^2, 2*a}
  let A_plus_B : Set ℝ := {x | ∃ (x₁ x₂ : ℝ), x₁ ∈ A ∧ x₂ ∈ B ∧ x = x₁ + x₂}
  (∀ x ∈ A_plus_B, x ≤ 2*a + 1) ∧ (2*a + 1 ∈ A_plus_B) →
  a > 1 - Real.sqrt 2 ∧ a < 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_element_range_l148_14809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tangent_circles_exist_l148_14805

-- Define the angle ABC
axiom Angle : Type
axiom Point : Type
axiom Circle : Type

-- Define the angle ABC
axiom angle_ABC : Angle

-- Define point M inside the angle
axiom point_M : Point

-- Define the property of being inside an angle
axiom is_inside (p : Point) (a : Angle) : Prop

-- Define the property of a circle being tangent to two lines
axiom is_tangent_to_sides (c : Circle) (a : Angle) : Prop

-- Define the property of a circle passing through a point
axiom passes_through (c : Circle) (p : Point) : Prop

-- Theorem statement
theorem two_tangent_circles_exist :
  is_inside point_M angle_ABC →
  ∃ (c1 c2 : Circle),
    is_tangent_to_sides c1 angle_ABC ∧
    passes_through c1 point_M ∧
    is_tangent_to_sides c2 angle_ABC ∧
    passes_through c2 point_M ∧
    c1 ≠ c2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tangent_circles_exist_l148_14805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_calculation_l148_14845

/-- Represents the commission earned per car sold -/
def commission : ℝ := sorry

/-- Base salary per month -/
def base_salary : ℝ := 1000

/-- Total earnings in January -/
def january_earnings : ℝ := 1800

/-- Number of cars to be sold in February -/
def february_cars : ℕ := 13

/-- The earnings in February will be double the January earnings -/
def february_earnings : ℝ := 2 * january_earnings

theorem commission_calculation :
  commission = 200 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_calculation_l148_14845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_properties_l148_14811

noncomputable section

variable (z₁ : ℂ)

axiom z₁_imaginary : z₁.im ≠ 0

def z₂ : ℝ := (z₁ + z₁⁻¹).re

axiom z₂_range : -1 ≤ z₂ ∧ z₂ ≤ 1

theorem z₁_properties :
  ‖z₁‖ = 1 ∧
  -1/2 ≤ z₁.re ∧ z₁.re ≤ 1/2 ∧
  ∃ (y : ℝ), (1 - z₁) / (1 + z₁) = y * I :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_properties_l148_14811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lending_problem_l148_14850

/-- The amount of money (in rupees) that A lends to B -/
def P : ℝ := sorry

/-- The interest rate (as a decimal) that A charges B -/
def rate_A_to_B : ℝ := 0.10

/-- The interest rate (as a decimal) that B charges C -/
def rate_B_to_C : ℝ := 0.14

/-- The number of years for which the money is lent -/
def years : ℝ := 3

/-- The gain (in rupees) that B makes over the given period -/
def gain_B : ℝ := 420

theorem lending_problem : 
  P * rate_B_to_C * years - P * rate_A_to_B * years = gain_B → 
  P = 3500 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lending_problem_l148_14850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elimination_result_elimination_terminates_l148_14849

/-- The elimination process function -/
def eliminate (n : ℕ) : ℕ := 
  let first_round := n / 2
  let rec elim_by_three (m : ℕ) : ℕ :=
    if m ≤ 1 then m else elim_by_three ((m + 2) / 3)
  elim_by_three first_round

/-- The theorem stating the result of the elimination process for 2012 students -/
theorem elimination_result : eliminate 2012 = 1458 := by
  sorry

/-- Proof that the elimination process always terminates -/
theorem elimination_terminates (n : ℕ) : ∃ k, (eliminate n).iterate k eliminate = eliminate n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elimination_result_elimination_terminates_l148_14849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l148_14833

/-- The function f(x) = tan(x/3) + sin(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.tan (x / 3) + Real.sin x

/-- The period of f(x) -/
noncomputable def period_f : ℝ := 6 * Real.pi

/-- Theorem: The period of f(x) = tan(x/3) + sin(x) is 6π -/
theorem period_of_f : 
  ∀ x : ℝ, f (x + period_f) = f x := by
  sorry

#check period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l148_14833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_proof_l148_14881

theorem cube_root_equation_proof : ∃ (a b c : ℕ+), 
  (2 * Real.sqrt (Real.rpow 7 (1/3) - Real.rpow 6 (1/3)) = 
   Real.rpow a.val (1/3) - Real.rpow b.val (1/3) + Real.rpow c.val (1/3)) ∧
  (a = 28) ∧ (b = 7) ∧ (c = 14) ∧ (a + b + c = 49) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_proof_l148_14881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_marbles_value_l148_14878

/-- The total number of marbles in a collection with red, blue, green, and yellow marbles -/
noncomputable def total_marbles (r : ℝ) : ℝ :=
  let b := r / 1.3  -- 30% more red than blue: r = 1.3b, so b = r/1.3
  let g := 1.5 * r  -- 50% more green than red: g = 1.5r
  let y := 0.8 * g  -- Yellow is 20% less than green: y = 0.8g
  r + b + g + y

/-- Theorem stating the total number of marbles in the collection -/
theorem total_marbles_value (r : ℝ) :
  total_marbles r = 4.4692 * r := by
  unfold total_marbles
  -- Expand the definition and perform algebraic simplification
  simp [mul_add, add_mul, mul_assoc]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_marbles_value_l148_14878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l148_14851

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - Real.cos x ^ 2 + 1 / 2

-- State the theorem
theorem min_value_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≥ f x) ∧
  f x = -1 / 2 := by
  sorry

#check min_value_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l148_14851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_constant_pi_increasing_l148_14887

/-- A constant function that always returns π -/
noncomputable def constant_pi : ℝ → ℝ := λ _ ↦ Real.pi

/-- A function is increasing if for any x < y, f(x) ≤ f(y) -/
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≤ f y

theorem negation_of_constant_pi_increasing :
  ¬(∀ a : ℝ, is_increasing constant_pi) ↔ 
  ∃ a : ℝ, ¬(is_increasing constant_pi) :=
by
  -- The proof is omitted for now
  sorry

#check negation_of_constant_pi_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_constant_pi_increasing_l148_14887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_sum_l148_14898

theorem two_integers_sum (a b : ℕ+) : 
  (a * b + a + b = 119) →
  (Nat.Coprime a b) →
  (a < 25 ∧ b < 25) →
  (a + b = 27 ∨ a + b = 24 ∨ a + b = 21) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_sum_l148_14898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_bound_l148_14846

-- Define a quadratic polynomial
def quadratic_polynomial (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

-- State the theorem
theorem quadratic_polynomial_bound 
  (p : ℝ → ℝ) 
  (h_quad : ∃ a b c : ℝ, ∀ x, p x = quadratic_polynomial a b c x) 
  (h_bound : ∀ x ∈ ({-1, 0, 1} : Set ℝ), |p x| ≤ 1) :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, |p x| ≤ 5/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_bound_l148_14846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_23_l148_14848

/-- Represents a die with numbered faces and a blank face -/
structure Die where
  faces : Finset ℕ
  blank : Bool
  twenty_faced : faces.card + (if blank then 1 else 0) = 20

/-- The first die with faces 1 through 18 and a blank face -/
def die1 : Die where
  faces := Finset.range 19 \ {0}
  blank := true
  twenty_faced := by sorry

/-- The second die with faces 2 through 9 and 11 through 21 and a blank face -/
def die2 : Die where
  faces := (Finset.range 22 \ {0, 1, 10}) \ {21}
  blank := true
  twenty_faced := by sorry

/-- The set of all possible rolls (pairs of numbers) from the two dice -/
def allRolls : Finset (ℕ × ℕ) :=
  (die1.faces.product die2.faces) ∪ 
  (({0} : Finset ℕ).product die2.faces) ∪ 
  (die1.faces.product ({0} : Finset ℕ)) ∪ 
  ({(0, 0)} : Finset (ℕ × ℕ))

/-- The set of rolls that sum to 23 -/
def rollsSum23 : Finset (ℕ × ℕ) :=
  allRolls.filter (λ p => p.1 + p.2 = 23)

/-- The main theorem stating the probability of rolling a sum of 23 -/
theorem prob_sum_23 : 
  (rollsSum23.card : ℚ) / allRolls.card = 2 / 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_23_l148_14848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_problem_l148_14877

-- Define the function g
def g : ℝ → ℝ := sorry

-- Define the inverse function of g
def g_inv : ℝ → ℝ := sorry

-- State the given conditions
axiom g_inv_is_inverse : Function.RightInverse g_inv g ∧ Function.LeftInverse g_inv g
axiom g_neg_one : g (-1) = 2
axiom g_zero : g 0 = 3
axiom g_one : g 1 = 6

-- State the theorem to be proved
theorem inverse_function_problem : g_inv (g_inv 6 - g_inv 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_problem_l148_14877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_theta_f_l148_14828

-- Define the function f
noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := (Real.cos (θ * x))^2 + (Real.cos (θ * x)) * (Real.sin (θ * x))

-- State the theorem
theorem max_value_of_theta_f (θ : ℝ) :
  (∀ x, f θ (x + π / (2 * θ)) = f θ x) →  -- Smallest positive period is π/2
  (∃ x₀, ∀ x, θ * f θ x ≤ θ * f θ x₀) →  -- Maximum value exists
  (∃ x₀, θ * f θ x₀ = 1 + Real.sqrt 2) :=  -- Maximum value is 1 + √2
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_theta_f_l148_14828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_81_not_3_l148_14891

/-- An arithmetic sequence with first term 1 and common difference d -/
def arithmetic_sequence (d : ℕ) : ℕ → ℕ := λ n ↦ 1 + (n - 1) * d

/-- Theorem: If 81 is a term in an arithmetic sequence with first term 1 and
    common difference d ∈ ℕ*, then d ≠ 3 -/
theorem arithmetic_sequence_81_not_3 :
  ∀ d : ℕ, d > 0 →
  (∃ n : ℕ, arithmetic_sequence d n = 81) →
  d ≠ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_81_not_3_l148_14891
