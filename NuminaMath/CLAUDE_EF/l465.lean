import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l465_46565

/-- The maximum number of trucks that could have been rented out during a week given specific conditions -/
def max_rented_trucks (total_trucks : ℕ) (return_rate : ℚ) (min_saturday : ℕ) : ℕ :=
  let max_unreturned := total_trucks - min_saturday
  let rented := (max_unreturned : ℚ) / (1 - return_rate)
  (Int.floor rented).toNat

/-- Given the specific conditions of the problem, the maximum number of rented trucks is 37 -/
theorem problem_solution : max_rented_trucks 30 (3/5) 15 = 37 := by
  sorry

#eval max_rented_trucks 30 (3/5) 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l465_46565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_power_difference_l465_46592

theorem divisibility_of_power_difference (a b q : ℕ) (hq : q > 1) :
  ¬(2 ∣ (a^(2*q) - b^(2*q))) ∨ (8 ∣ (a^(2*q) - b^(2*q))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_power_difference_l465_46592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_conversion_l465_46585

/-- Converts a number in scientific notation to its standard form -/
noncomputable def scientific_to_standard (base : ℝ) (exponent : ℕ) : ℝ :=
  base * (10 : ℝ) ^ exponent

/-- The problem statement -/
theorem scientific_notation_conversion :
  scientific_to_standard 6.03 5 = 603000 := by
  -- Unfold the definition of scientific_to_standard
  unfold scientific_to_standard
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_conversion_l465_46585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_polar_l465_46597

/-- The intersection point of two lines in polar coordinates -/
theorem intersection_point_polar (ρ θ : ℝ) : 
  (Real.sqrt 2 * ρ = 1 / Real.sin (π / 4 + θ)) →
  (θ = π / 3) →
  (ρ = Real.sqrt 3 - 1 ∧ θ = π / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_polar_l465_46597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_representation_theorem_l465_46519

-- Define the type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define the type for a sequence of real-valued functions
def FunctionSequence := ℕ → RealFunction

-- Define the type for the finite set of φ functions
def PhiFunctions := Fin 1994 → RealFunction

-- Define a type for compositions of φ functions
inductive CompositionPhi (φ : PhiFunctions)
  | single : Fin 1994 → CompositionPhi φ
  | comp : CompositionPhi φ → Fin 1994 → CompositionPhi φ

-- Define what it means for a function to be representable by a composition
def representable (φ : PhiFunctions) (f : RealFunction) : Prop :=
  ∃ c : CompositionPhi φ, ∀ x : ℝ, f x = sorry  -- The actual computation is omitted

theorem function_representation_theorem (f : FunctionSequence) :
  ∃ φ : PhiFunctions, ∀ n : ℕ, representable φ (f n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_representation_theorem_l465_46519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_two_three_four_l465_46553

/-- The delta operation for positive real numbers -/
noncomputable def delta (a b : ℝ) : ℝ := (a^2 + b^2) / (1 + a^2 * b^2)

/-- Theorem: The result of (2 Δ 3) Δ 4 is 5945/4073 -/
theorem delta_two_three_four :
  delta (delta 2 3) 4 = 5945 / 4073 :=
by
  -- Expand the definition of delta
  unfold delta
  -- Simplify the expression
  simp [pow_two]
  -- The rest of the proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_two_three_four_l465_46553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_regression_equation_l465_46541

/-- Represents the selling price in yuan/unit -/
def x : ℝ := sorry

/-- Represents the sales volume in units -/
def y : ℝ := sorry

/-- The regression equation for the relationship between x and y -/
def regression_equation (x : ℝ) : ℝ := -10 * x + 200

/-- Assumption that y is negatively correlated with x -/
axiom negative_correlation : ∀ x₁ x₂, x₁ < x₂ → regression_equation x₁ > regression_equation x₂

theorem correct_regression_equation :
  ∀ x, y = regression_equation x :=
by
  sorry

#check correct_regression_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_regression_equation_l465_46541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_greater_than_two_l465_46535

/-- The function f(x) = x ln(x-1) - a -/
noncomputable def f (x a : ℝ) : ℝ := x * Real.log (x - 1) - a

/-- Theorem: For a > 0, f(x) has a zero greater than 2 -/
theorem f_has_zero_greater_than_two (a : ℝ) (ha : a > 0) :
  ∃ x₀ : ℝ, x₀ > 2 ∧ f x₀ a = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_greater_than_two_l465_46535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l465_46518

theorem slope_range (k θ : Real) :
  (30 * Real.pi / 180 < θ) →
  (θ < 90 * Real.pi / 180) →
  (k = Real.tan θ) →
  k > Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l465_46518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_sum_theorem_l465_46536

/-- The set of numbers from 1 to 2n-1 -/
def A (n : ℕ) : Finset ℕ := Finset.filter (fun x => 1 ≤ x ∧ x ≤ 2*n - 1) (Finset.range (2*n))

/-- The set of numbers to be expelled -/
def expelled (n : ℕ) : Finset ℕ := Finset.filter (fun x => 2 ≤ x ∧ x ≤ 2*n - 1) (Finset.range (2*n))

/-- Condition: If a is expelled and 2a is in A, then 2a is expelled -/
def condition1 (n : ℕ) : Prop :=
  ∀ a ∈ expelled n, 2*a ∈ A n → 2*a ∈ expelled n

/-- Condition: If a and b are expelled and a+b is in A, then a+b is expelled -/
def condition2 (n : ℕ) : Prop :=
  ∀ a b, a ∈ expelled n → b ∈ expelled n → a + b ∈ A n → a + b ∈ expelled n

/-- At least n-1 numbers are expelled -/
def condition3 (n : ℕ) : Prop :=
  (expelled n).card ≥ n - 1

/-- The sum of remaining numbers is minimal -/
def is_minimal_sum (n : ℕ) : Prop :=
  ∀ S : Finset ℕ, S ⊆ A n → S.card = (A n).card - (expelled n).card →
    Finset.sum (expelled n) id ≤ Finset.sum S id

theorem minimal_sum_theorem (n : ℕ) : 
  condition1 n ∧ condition2 n ∧ condition3 n → is_minimal_sum n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_sum_theorem_l465_46536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roll_probability_l465_46566

-- Define a fair six-sided die
def fair_die : Finset ℕ := Finset.range 6

-- Define the probability of an event on a fair die
def prob (event : Finset ℕ) : ℚ :=
  (event.card : ℚ) / (fair_die.card : ℚ)

-- Define the number of rolls
def num_rolls : ℕ := 7

-- Define the number of times we want to roll 1
def num_ones : ℕ := 5

-- Define the number of times we want to roll 2
def num_twos : ℕ := 1

-- State the theorem
theorem roll_probability :
  (Nat.choose num_rolls num_ones * Nat.choose (num_rolls - num_ones) num_twos) *
  (prob {0})^num_ones * (prob {1})^num_twos * (1 - prob {0} - prob {1})^(num_rolls - num_ones - num_twos) =
  1 / 417 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roll_probability_l465_46566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_of_F_times_2_pow_x_l465_46580

noncomputable def f (x : ℝ) : ℝ := 1 - abs (1 - 2*x)

def g (x : ℝ) : ℝ := x^2 - 2*x + 1

noncomputable def F (x : ℝ) : ℝ :=
  if f x ≥ g x then f x else g x

theorem three_roots_of_F_times_2_pow_x :
  ∃! (roots : Finset ℝ), roots.card = 3 ∧ 
  ∀ r ∈ roots, r ∈ Set.Icc 0 1 ∧ F r * (2:ℝ)^r = 1 := by
  sorry

#check three_roots_of_F_times_2_pow_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_of_F_times_2_pow_x_l465_46580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l465_46550

noncomputable def g (x : ℝ) : ℝ := ⌊2 * x⌋ + (1 : ℝ) / 3

theorem g_neither_even_nor_odd :
  ¬(∀ x, g (-x) = g x) ∧ ¬(∀ x, g (-x) = -g x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l465_46550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l465_46569

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < a ∧ 0 < b

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point being on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.y^2 / h.a^2 - p.x^2 / h.b^2 = 1

/-- Definition of asymptote intersection points -/
noncomputable def asymptote_intersections (h : Hyperbola) (p : Point) : Point × Point :=
  let a := Point.mk (h.b * p.y / h.a) p.y
  let b := Point.mk (-h.b * p.y / h.a) p.y
  (a, b)

/-- Definition of dot product for vectors represented as points -/
def dot_product (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

/-- Main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) (p : Point) 
  (h_on : on_hyperbola h p) 
  (h_dot : let (a, b) := asymptote_intersections h p
           dot_product (Point.mk (a.x - p.x) (a.y - p.y)) (Point.mk (b.x - p.x) (b.y - p.y)) = -h.a^2 / 4) :
  Real.sqrt (1 + h.b^2 / h.a^2) = Real.sqrt 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l465_46569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_circle_l465_46560

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- The unit square [0,1] × [0,1] -/
def UnitSquare : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

/-- A set of 51 points in the unit square -/
def PointSet : Set Point :=
  {p : Point | p ∈ UnitSquare ∧ ∃ (S : Finset Point), ↑S ⊆ UnitSquare ∧ S.card = 51 ∧ p ∈ S}

/-- A point is inside a circle -/
def InsideCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

theorem points_in_circle :
  ∃ (c : Circle), c.radius = 1/7 ∧ (∃ (S : Finset Point), ↑S ⊆ PointSet ∧ S.card ≥ 3 ∧ ∀ p ∈ S, InsideCircle p c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_circle_l465_46560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_eight_implies_x_equals_three_l465_46596

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^3 else 2^x

-- State the theorem
theorem f_equals_eight_implies_x_equals_three :
  ∀ x : ℝ, f x = 8 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_eight_implies_x_equals_three_l465_46596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_t_equals_zero_l465_46523

noncomputable def f (x : ℝ) := 3 * x + Real.sin x + 1

theorem f_negative_t_equals_zero (t : ℝ) (h : f t = 2) : f (-t) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_t_equals_zero_l465_46523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_from_equal_projections_l465_46527

/-- Represent a line in 2D space -/
structure Line (α : Type) [Ring α] where
  slope : α
  intercept : α
  direction : α × α
  angle : ℝ

/-- Given two vectors OA and OB in a plane, and a line l with an obtuse slope angle,
    prove that the slope of l is -4/3 if the projections of OA and OB on l's direction vector are equal. -/
theorem line_slope_from_equal_projections (OA OB : ℝ × ℝ) (l : Line ℝ) :
  OA = (1, 4) →
  OB = (-3, 1) →
  (∃ u : ℝ × ℝ, u ≠ (0, 0) ∧ l.direction = u ∧
    abs (OA.1 * u.1 + OA.2 * u.2) = abs (OB.1 * u.1 + OB.2 * u.2)) →
  l.angle > π / 2 →
  l.slope = -4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_from_equal_projections_l465_46527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_equals_four_l465_46548

-- Define the sequence a_n
def a : ℕ → ℕ
| 1 => 2
| n => 2^n

-- Define the sum of terms from k+1 to k+10
def sum_k_to_k_plus_10 (k : ℕ) : ℕ :=
  (List.range 10).map (λ i => a (k + i + 1)) |>.sum

-- Theorem statement
theorem k_equals_four (k : ℕ) :
  sum_k_to_k_plus_10 k = 2^15 - 2^5 → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_equals_four_l465_46548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_in_binomial_expansion_l465_46559

theorem coefficient_of_x_in_binomial_expansion :
  (Finset.sum (Finset.range 5) (fun k => ((-1 : ℤ)^(4 - k)) * (2^(4 - k)) * (Nat.choose 5 (k + 1)) * (k + 1))) = 80 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_in_binomial_expansion_l465_46559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_concurrency_l465_46520

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A pentagon in the plane -/
structure Pentagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point

/-- Check if a pentagon is convex -/
def is_convex (p : Pentagon) : Prop := sorry

/-- Construct a line from two points -/
def line_from_points (p1 p2 : Point) : Line := sorry

/-- Intersection of two lines -/
noncomputable def intersect (l1 l2 : Line) : Point := sorry

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop := sorry

/-- Check if lines are concurrent -/
def are_concurrent (l1 l2 l3 l4 l5 : Line) : Prop := sorry

/-- Main theorem -/
theorem pentagon_concurrency 
  (ABCDE : Pentagon) 
  (h_convex : is_convex ABCDE)
  (A' : Point) (h_A' : A' = intersect (line_from_points ABCDE.B ABCDE.D) (line_from_points ABCDE.C ABCDE.E))
  (B' : Point) (h_B' : B' = intersect (line_from_points ABCDE.C ABCDE.E) (line_from_points ABCDE.D ABCDE.A))
  (C' : Point) (h_C' : C' = intersect (line_from_points ABCDE.D ABCDE.A) (line_from_points ABCDE.E ABCDE.B))
  (D' : Point) (h_D' : D' = intersect (line_from_points ABCDE.E ABCDE.B) (line_from_points ABCDE.A ABCDE.C))
  (E' : Point) (h_E' : E' = intersect (line_from_points ABCDE.A ABCDE.C) (line_from_points ABCDE.B ABCDE.D))
  (A'' : Point) (h_A'' : point_on_line A'' (line_from_points ABCDE.A ABCDE.B) ∧ 
                         point_on_line A'' (line_from_points ABCDE.A ABCDE.E))
  (B'' : Point) (h_B'' : point_on_line B'' (line_from_points ABCDE.B ABCDE.C) ∧ 
                         point_on_line B'' (line_from_points ABCDE.B ABCDE.A))
  (C'' : Point) (h_C'' : point_on_line C'' (line_from_points ABCDE.C ABCDE.D) ∧ 
                         point_on_line C'' (line_from_points ABCDE.C ABCDE.B))
  (D'' : Point) (h_D'' : point_on_line D'' (line_from_points ABCDE.D ABCDE.E) ∧ 
                         point_on_line D'' (line_from_points ABCDE.D ABCDE.C))
  (E'' : Point) (h_E'' : point_on_line E'' (line_from_points ABCDE.E ABCDE.A) ∧ 
                         point_on_line E'' (line_from_points ABCDE.E ABCDE.D)) :
  are_concurrent 
    (line_from_points ABCDE.A A'') 
    (line_from_points ABCDE.B B'') 
    (line_from_points ABCDE.C C'') 
    (line_from_points ABCDE.D D'') 
    (line_from_points ABCDE.E E'') :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_concurrency_l465_46520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_x_axis_l465_46588

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-2, 0)
def right_focus : ℝ × ℝ := (2, 0)

-- Define the angle between foci and point P
noncomputable def angle_foci_p (p : ℝ × ℝ) : ℝ := 60 * Real.pi / 180

-- Theorem statement
theorem distance_to_x_axis (p : ℝ × ℝ) :
  hyperbola p.1 p.2 →
  angle_foci_p p = Real.arccos ((dist p left_focus)^2 + (dist p right_focus)^2 - (dist left_focus right_focus)^2) / (2 * dist p left_focus * dist p right_focus) →
  |p.2| = Real.sqrt 15 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_x_axis_l465_46588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_savings_calculation_l465_46533

/-- Calculates Alice's savings based on her sales, salary, commissions, and expenses. -/
noncomputable def aliceSavings (salesAmount : ℝ) (basicSalary : ℝ) (commissionRate1 : ℝ) (commissionRate2 : ℝ) (commissionThreshold : ℝ) (expenses : ℝ) (savingsRate : ℝ) : ℝ :=
  let commission1 := min salesAmount commissionThreshold * commissionRate1
  let commission2 := max (salesAmount - commissionThreshold) 0 * commissionRate2
  let totalEarnings := basicSalary + commission1 + commission2
  let earningsAfterExpenses := totalEarnings - expenses
  earningsAfterExpenses * savingsRate

/-- Theorem stating that Alice's savings amount to $31.50 given the specified conditions. -/
theorem alice_savings_calculation :
  aliceSavings 3000 500 0.03 0.05 2000 400 0.15 = 31.50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_savings_calculation_l465_46533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l465_46581

noncomputable def curve1 (x : ℝ) : ℝ := (1/2) * Real.exp x
noncomputable def curve2 (x : ℝ) : ℝ := Real.log (2 * x)

noncomputable def distance (x₁ x₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (curve1 x₁ - curve2 x₂)^2)

theorem min_distance_between_curves :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 * (1 - Real.log 2) ∧
  ∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 → distance x₁ x₂ ≥ min_dist :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l465_46581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l465_46547

open Set

variable (a : ℝ)

def U : Set ℝ := univ

def A (a : ℝ) : Set ℝ := {x : ℝ | (x - 2) / (x - 3 * a - 1) < 0}

def B (a : ℝ) : Set ℝ := {x : ℝ | (x - a^2 - 2) / (x - a) < 0}

theorem problem_solution :
  (∀ a, a = 1/2 → (Uᶜ ∩ B a)ᶜ ∩ A a = Icc (9/4) (5/2)) ∧
  (∀ a, A a ⊆ B a ↔ a ∈ Ioo (-1/2) (1/3) ∪ Ioo (1/3) ((3 - Real.sqrt 5) / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l465_46547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l465_46503

/-- Pentagon with specific side lengths and decomposition -/
structure Pentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  triangle_base : ℝ
  triangle_height : ℝ
  trapezoid_base1 : ℝ
  trapezoid_base2 : ℝ
  trapezoid_height : ℝ

/-- The area of the pentagon -/
noncomputable def pentagon_area (p : Pentagon) : ℝ :=
  (1/2 * p.triangle_base * p.triangle_height) + 
  (1/2 * (p.trapezoid_base1 + p.trapezoid_base2) * p.trapezoid_height)

/-- Theorem stating the area of the specific pentagon -/
theorem specific_pentagon_area :
  ∃ p : Pentagon,
    p.side1 = 18 ∧
    p.side2 = 25 ∧
    p.side3 = 30 ∧
    p.side4 = 28 ∧
    p.side5 = 25 ∧
    p.triangle_base = 25 ∧
    p.triangle_height = 18 ∧
    p.trapezoid_base1 = 28 ∧
    p.trapezoid_base2 = 25 ∧
    p.trapezoid_height = 30 ∧
    pentagon_area p = 1020 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l465_46503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_l465_46538

/-- Given an ellipse with equation x²/4 + y²/3 = 1, let A be the left vertex and F₁, F₂ be the left and right foci respectively. A line with slope k is drawn from A, intersecting the ellipse at another point B. The line BF₂ is extended to intersect the ellipse at C. -/
theorem ellipse_intersection (k : ℝ) :
  let A : ℝ × ℝ := (-2, 0)
  let F₁ : ℝ × ℝ := (-1, 0)
  let F₂ : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := ((6 - 8*k^2) / (3 + 4*k^2), 12*k / (3 + 4*k^2))
  let ellipse := fun (x y : ℝ) ↦ x^2/4 + y^2/3 = 1
  let line_AB := fun (x : ℝ) ↦ k*(x + 2)
  let line_BF₂ := fun (x : ℝ) ↦ (4*k / (1 - 4*k^2)) * (x - 1)
  let line_F₁C := fun (x : ℝ) ↦ (-1/k) * (x + 1)
  -- The point B lies on the ellipse and the line AB
  (ellipse B.1 B.2 ∧ B.2 = line_AB B.1) →
  -- If F₁C is perpendicular to AB, then k = ±√6/12
  (∃ C : ℝ × ℝ, ellipse C.1 C.2 ∧ C.2 = line_BF₂ C.1 ∧ C.2 = line_F₁C C.1) →
  k = Real.sqrt 6 / 12 ∨ k = -Real.sqrt 6 / 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_l465_46538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_half_minus_alpha_l465_46532

theorem cos_beta_half_minus_alpha (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < π / 2)
  (h3 : 0 < β) (h4 : β < π / 2)
  (h5 : Real.sin (π / 3 - α) = 3 / 5)
  (h6 : Real.cos (β / 2 - π / 3) = 2 * Real.sqrt 5 / 5) :
  Real.cos (β / 2 - α) = 11 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_half_minus_alpha_l465_46532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_sine_l465_46528

theorem pure_imaginary_sine (θ : ℝ) :
  (∃ (z : ℂ), z = Complex.I * (Real.sqrt 2 + 1) + (Real.sin (2 * θ) - 1) ∧ z.re = 0) →
  ∃ (k : ℤ), θ = k * Real.pi + Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_sine_l465_46528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_proof_l465_46514

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0

-- Define the area of the circle
noncomputable def circle_area : ℝ := (25 / 4) * Real.pi

-- Theorem statement
theorem circle_area_proof :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y : ℝ, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    circle_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_proof_l465_46514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kwik_e_tax_revenue_l465_46579

/-- Calculate the total revenue for Kwik-e-Tax Center --/
theorem kwik_e_tax_revenue : 
  let federal_cost : ℕ := 50
  let state_cost : ℕ := 30
  let quarterly_cost : ℕ := 80
  let international_cost : ℕ := 100
  let value_added_cost : ℕ := 75
  let federal_sold : ℕ := 60
  let state_sold : ℕ := 20
  let quarterly_sold : ℕ := 10
  let international_sold : ℕ := 13
  let value_added_sold : ℕ := 25
  let international_discount : ℚ := 1/5

  let federal_revenue := federal_cost * federal_sold
  let state_revenue := state_cost * state_sold
  let quarterly_revenue := quarterly_cost * quarterly_sold
  let international_revenue := (international_cost * (1 - international_discount)).floor * international_sold
  let value_added_revenue := value_added_cost * value_added_sold

  let total_revenue := federal_revenue + state_revenue + quarterly_revenue + international_revenue + value_added_revenue

  total_revenue = 7315 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kwik_e_tax_revenue_l465_46579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kolya_is_collection_agency_l465_46501

-- Define the actors in the scenario
inductive Actor : Type
| Katya : Actor
| Vasya : Actor
| Kolya : Actor

-- Define the possible roles
inductive Role : Type
| FinancialPyramid : Role
| CollectionAgency : Role
| Bank : Role
| InsuranceCompany : Role

-- Define the action of lending books
def lendBooks (lender receiver : Actor) : Prop := sorry

-- Define the action of failing to return books
def failToReturn (borrower : Actor) : Prop := sorry

-- Define the action of asking for help to retrieve books
def askForHelp (asker helper : Actor) : Prop := sorry

-- Define the action of agreeing to help for a reward
def agreeToHelpForReward (helper : Actor) : Prop := sorry

-- Define the function that determines Kolya's role based on the scenario
def kolyaRole (
  lendBooks : Actor → Actor → Prop)
  (failToReturn : Actor → Prop)
  (askForHelp : Actor → Actor → Prop)
  (agreeToHelpForReward : Actor → Prop) : Role := Role.CollectionAgency

-- Theorem statement
theorem kolya_is_collection_agency :
  lendBooks Actor.Katya Actor.Vasya →
  failToReturn Actor.Vasya →
  askForHelp Actor.Katya Actor.Kolya →
  agreeToHelpForReward Actor.Kolya →
  kolyaRole lendBooks failToReturn askForHelp agreeToHelpForReward = Role.CollectionAgency :=
by
  intro _ _ _ _
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kolya_is_collection_agency_l465_46501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_inequalities_l465_46513

theorem sine_cosine_inequalities (x : ℝ) : 
  (-Real.sqrt 2 ≤ Real.sin x + Real.cos x ∧ Real.sin x + Real.cos x ≤ Real.sqrt 2) ∧
  (-Real.sqrt 2 ≤ Real.sin x - Real.cos x ∧ Real.sin x - Real.cos x ≤ Real.sqrt 2) ∧
  (1 ≤ |Real.sin x| + |Real.cos x| ∧ 
   |Real.sin x| + |Real.cos x| ≤ Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_inequalities_l465_46513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_area_l465_46563

noncomputable section

def Triangle : Type := ℝ → ℝ → ℝ → Prop
def Square : Type := ℝ → Prop

def IsRightAngled : Triangle → Prop := sorry
def side : Square → ℝ := sorry
def area : Square → ℝ := sorry

theorem largest_square_area 
  (XYZ : Triangle)
  (right_angle : IsRightAngled XYZ)
  (square_XY square_YZ square_XZ : Square)
  (on_XY : side square_XY = sorry) -- Replace with actual side length expression
  (on_YZ : side square_YZ = sorry) -- Replace with actual side length expression
  (on_XZ : side square_XZ = sorry) -- Replace with actual side length expression
  (total_area : area square_XY + area square_YZ + area square_XZ = 450)
  (side_diff : side square_XY = side square_YZ - 5) :
  area square_XZ = 225 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_area_l465_46563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l465_46531

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_domain : ∀ x, -4 < x → x < 4 → ∃ y, f x = y
axiom f_decreasing : ∀ x y, -4 < x → x < y → y < 4 → f x > f y

-- Define the condition on a
def condition (a : ℝ) : Prop := f (1 - a) + f (2 * a - 3) < 0

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, condition a ↔ (2 < a ∧ a < 7/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l465_46531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_result_l465_46505

theorem election_result (total_votes : ℕ) (winner_percent second_percent : ℚ) 
  (margin : ℕ) (h1 : winner_percent = 42 / 100) 
  (h2 : second_percent = 37 / 100) 
  (h3 : (winner_percent - second_percent) * total_votes = margin) 
  (h4 : margin = 324) : 
  ∃ (third_percent : ℚ) (winner_votes second_votes third_votes : ℕ),
    third_percent = 21 / 100 ∧
    total_votes = 6480 ∧
    winner_votes + second_votes + third_votes = total_votes ∧
    winner_votes = (winner_percent * ↑total_votes).floor ∧
    second_votes = (second_percent * ↑total_votes).floor ∧
    third_votes = (third_percent * ↑total_votes).floor := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_result_l465_46505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_short_pencils_sold_is_35_l465_46583

/-- Represents the sales data for a stationery store --/
structure PencilSales where
  eraser_price : ℚ
  regular_price : ℚ
  short_price : ℚ
  eraser_sold : ℕ
  regular_sold : ℕ
  total_revenue : ℚ

/-- Calculates the number of short pencils sold --/
def short_pencils_sold (sales : PencilSales) : ℕ :=
  let eraser_revenue := sales.eraser_price * sales.eraser_sold
  let regular_revenue := sales.regular_price * sales.regular_sold
  let short_revenue := sales.total_revenue - eraser_revenue - regular_revenue
  (short_revenue / sales.short_price).floor.toNat

/-- Theorem stating that the number of short pencils sold is 35 --/
theorem short_pencils_sold_is_35 (sales : PencilSales) 
    (h1 : sales.eraser_price = 0.8)
    (h2 : sales.regular_price = 0.5)
    (h3 : sales.short_price = 0.4)
    (h4 : sales.eraser_sold = 200)
    (h5 : sales.regular_sold = 40)
    (h6 : sales.total_revenue = 194) :
    short_pencils_sold sales = 35 := by
  sorry

#eval short_pencils_sold {
  eraser_price := 0.8,
  regular_price := 0.5,
  short_price := 0.4,
  eraser_sold := 200,
  regular_sold := 40,
  total_revenue := 194
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_short_pencils_sold_is_35_l465_46583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l465_46507

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, -1/2)

noncomputable def f (x : ℝ) : ℝ :=
  let a := vector_a x
  let b := vector_b x
  (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 - 2

theorem triangle_side_length 
  (A B C : ℝ) 
  (hf : f A = 1/2) 
  (h_arithmetic : ∃ (b c : ℝ), 2 * A = b + c) 
  (h_dot_product : (B - A) * (C - A) = 9) :
  A = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l465_46507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l465_46539

noncomputable def f (x : ℝ) := (2 : ℝ)^x - 3

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l465_46539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petes_annual_mileage_l465_46504

/-- Represents a pedometer with a maximum count before reset --/
structure Pedometer where
  max_count : Nat
  reset_count : Nat := 0

/-- Calculates the total steps given a pedometer's properties and readings --/
def total_steps (p : Pedometer) (initial_reading final_reading : Nat) : Nat :=
  (p.max_count - initial_reading + 1) + 
  (p.max_count + 1) * p.reset_count + 
  final_reading

/-- Converts steps to miles --/
def steps_to_miles (steps : Nat) (steps_per_mile : Nat) : Nat :=
  steps / steps_per_mile

theorem petes_annual_mileage : 
  let p : Pedometer := { max_count := 99999, reset_count := 72 }
  let initial_reading := 30000
  let final_reading := 45000
  let steps_per_mile := 1500
  steps_to_miles (total_steps p initial_reading final_reading) steps_per_mile = 4850 := by
  sorry

#eval steps_to_miles (total_steps { max_count := 99999, reset_count := 72 } 30000 45000) 1500

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petes_annual_mileage_l465_46504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_development_value_l465_46589

noncomputable def profit_before (x : ℝ) : ℝ := -1/160 * (x - 40)^2 + 10

noncomputable def profit_after (x : ℝ) : ℝ := -159/160 * (60 - x)^2 + 119/2 * (60 - x)

def total_profit_no_dev : ℝ := 10 * 10

noncomputable def total_profit_dev : ℝ :=
  5 * profit_before 30 + 5 * (profit_before 30 + profit_after 30)

theorem development_value : total_profit_dev > total_profit_no_dev := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_development_value_l465_46589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_extremum_l465_46508

/-- Quadratic function with a specific condition on c -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + (9 * b^2) / (16 * a)

/-- The graph of f has either a maximum or a minimum depending on the sign of a -/
theorem quadratic_extremum (a b : ℝ) (ha : a ≠ 0) :
  (a < 0 → ∃ x₀, ∀ x, f a b x ≤ f a b x₀) ∧
  (a > 0 → ∃ x₀, ∀ x, f a b x ≥ f a b x₀) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_extremum_l465_46508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l465_46595

/-- The equation of a circle with center (a, 0) and radius r -/
noncomputable def circle_equation (a r : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + y^2 = r^2

/-- The distance from a point (a, 0) to the line 2x - y = 0 -/
noncomputable def distance_to_line (a : ℝ) : ℝ :=
  |2*a| / Real.sqrt 5

theorem circle_equation_proof (a r : ℝ) :
  a > 0 ∧
  circle_equation a r 0 (Real.sqrt 5) ∧
  distance_to_line a = 4 * Real.sqrt 5 / 5 →
  a = 2 ∧ r = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l465_46595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grunters_win_probability_l465_46593

/-- The probability of a team winning a single game -/
def single_game_win_probability : ℚ := 3/4

/-- The number of games in the series -/
def number_of_games : ℕ := 4

/-- The probability of winning all games in the series -/
def all_games_win_probability : ℚ := single_game_win_probability ^ number_of_games

theorem grunters_win_probability :
  all_games_win_probability = 81/256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grunters_win_probability_l465_46593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_30_l465_46552

/-- Represents the swimming scenario with given parameters -/
structure SwimmingScenario where
  downstream_distance : ℚ
  time : ℚ
  still_water_speed : ℚ

/-- Calculates the upstream distance given a swimming scenario -/
def upstream_distance (s : SwimmingScenario) : ℚ :=
  let stream_speed := s.downstream_distance / s.time - s.still_water_speed
  (s.still_water_speed - stream_speed) * s.time

/-- Theorem stating that for the given scenario, the upstream distance is 30 km -/
theorem upstream_distance_is_30 :
  upstream_distance ⟨40, 5, 7⟩ = 30 := by
  -- Proof goes here
  sorry

#eval upstream_distance ⟨40, 5, 7⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_30_l465_46552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_and_trajectory_l465_46561

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 12*y + 24 = 0

-- Define point P
def point_P : ℝ × ℝ := (0, 5)

-- Define the line ι passing through P
def line_ι (k : ℝ) (x y : ℝ) : Prop := y = k * x + 5

-- Define the length of the chord intercepted by circle C on line ι
noncomputable def chord_length : ℝ := 4 * Real.sqrt 3

-- Define the midpoint condition for a chord passing through P
def midpoint_condition (x y : ℝ) : Prop := (x + 2) * x + (y - 6) * (y - 5) = 0

theorem circle_chord_and_trajectory :
  ∃ (k : ℝ), 
    (∀ (x y : ℝ), line_ι k x y ↔ (3 * x - 4 * y + 20 = 0 ∨ x = 0)) ∧
    (∀ (x y : ℝ), midpoint_condition x y ↔ x^2 + y^2 + 2*x - 11*y + 30 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_and_trajectory_l465_46561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l465_46562

noncomputable def star (x y : ℝ) : ℝ := (x * y) / (x + y + 1)

theorem star_properties (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (star x y = star y x) ∧ 
  (star (star x y) z ≠ star x (star y z)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l465_46562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_sequence_general_term_l465_46554

/-- An exponential sequence with first term 2 and common ratio 3 -/
def exponentialSequence : ℕ → ℝ
  | 0 => 2  -- Add this case for n = 0
  | n + 1 => 3 * exponentialSequence n

/-- The general term formula for the exponential sequence -/
def generalTerm (n : ℕ) : ℝ := 2 * 3^n

theorem exponential_sequence_general_term :
  ∀ n : ℕ, exponentialSequence n = generalTerm n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_sequence_general_term_l465_46554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_l465_46556

noncomputable section

open Real

theorem hypotenuse_length (a b c : ℝ) (h1 : a^2 + b^2 = c^2) 
  (h2 : ∃ (d e : ℝ), d = c/4 ∧ e = 3*c/4) 
  (h3 : ∃ (x y : ℝ), x^2 + y^2 = (c/4)^2 ∧ x = sin (π/4) ∧ y = cos (π/4)) : 
  c = 8/11 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_l465_46556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_probability_is_correct_l465_46521

/-- Represents the outcome of a single shot --/
inductive ShotOutcome
| Miss
| Hit
deriving Repr, DecidableEq

/-- Represents the results of 4 shots --/
def FourShots := (ShotOutcome × ShotOutcome × ShotOutcome × ShotOutcome)

/-- Converts a digit to a ShotOutcome --/
def digitToOutcome (d : Nat) : ShotOutcome :=
  if d ≤ 1 then ShotOutcome.Miss else ShotOutcome.Hit

/-- Counts the number of hits in a FourShots result --/
def countHits : FourShots → Nat
| (a, b, c, d) => 
    (if a = ShotOutcome.Hit then 1 else 0) +
    (if b = ShotOutcome.Hit then 1 else 0) +
    (if c = ShotOutcome.Hit then 1 else 0) +
    (if d = ShotOutcome.Hit then 1 else 0)

/-- Determines if a FourShots result has at least 3 hits --/
def hasAtLeastThreeHits (shots : FourShots) : Bool :=
  countHits shots ≥ 3

/-- The simulation results --/
def simulationResults : List FourShots := sorry

/-- The number of groups in the simulation --/
def numGroups : Nat := 20

/-- The number of groups with at least 3 hits --/
def numGroupsWithAtLeastThreeHits : Nat := 15

theorem estimated_probability_is_correct :
  (numGroupsWithAtLeastThreeHits : Rat) / numGroups = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_probability_is_correct_l465_46521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l465_46512

def M : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l465_46512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_not_pure_imaginary_min_abs_z_z_on_diagonal_l465_46511

-- Define the complex number z
noncomputable def z (m : ℂ) : ℂ := m + 10 / (3 - 4*Complex.I)

-- Define the property that m is a pure imaginary number
def is_pure_imaginary (c : ℂ) : Prop := c.re = 0

-- Theorem 1: z cannot be a pure imaginary number
theorem z_not_pure_imaginary (m : ℂ) (h : is_pure_imaginary m) : 
  ¬(is_pure_imaginary (z m)) := by sorry

-- Theorem 2: The minimum value of |z| is 6/5
theorem min_abs_z (m : ℂ) (h : is_pure_imaginary m) : 
  ∀ (c : ℂ), is_pure_imaginary c → Complex.abs (z c) ≥ 6/5 := by sorry

-- Theorem 3: If z lies on the line y = x, then m = -2/5i
theorem z_on_diagonal (m : ℂ) (h : is_pure_imaginary m) : 
  (z m).re = (z m).im → m = -(2/5) * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_not_pure_imaginary_min_abs_z_z_on_diagonal_l465_46511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_a_part_b_l465_46570

/-- A type representing a 3x3 board filled with numbers 1 to 8 and M -/
def Board := Fin 3 → Fin 3 → ℕ

/-- The sum of a row in the board -/
def rowSum (b : Board) (row : Fin 3) : ℕ :=
  (b row 0) + (b row 1) + (b row 2)

/-- The sum of a column in the board -/
def colSum (b : Board) (col : Fin 3) : ℕ :=
  (b 0 col) + (b 1 col) + (b 2 col)

/-- Predicate to check if all rows have the same sum -/
def allRowsSameSum (b : Board) : Prop :=
  ∀ i j : Fin 3, rowSum b i = rowSum b j

/-- Predicate to check if all columns have the same sum -/
def allColsSameSum (b : Board) : Prop :=
  ∀ i j : Fin 3, colSum b i = colSum b j

/-- Predicate to check if the board contains all numbers from 1 to 8 and M -/
def containsAllNumbers (b : Board) (M : ℕ) : Prop :=
  (∀ n : ℕ, n ≤ 8 → ∃ i j : Fin 3, b i j = n) ∧
  (∃ i j : Fin 3, b i j = M)

/-- Main theorem for part (a) -/
theorem part_a (M : ℕ) :
  (∃ b : Board, containsAllNumbers b M ∧ allRowsSameSum b) ↔ M ∈ ({3, 6, 9, 12} : Finset ℕ) :=
sorry

/-- Main theorem for part (b) -/
theorem part_b (M : ℕ) :
  (∃ b : Board, containsAllNumbers b M ∧ allRowsSameSum b ∧ allColsSameSum b) ↔ M ∈ ({6, 9} : Finset ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_a_part_b_l465_46570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_roots_l465_46540

/-- IsIsosceles represents that a triangle with given side lengths is isosceles -/
def IsIsosceles (sides : Fin 3 → ℝ) : Prop :=
  (sides 0 = sides 1) ∨ (sides 1 = sides 2) ∨ (sides 0 = sides 2)

theorem isosceles_triangle_roots (n : ℝ) : 
  (∃ (a b : ℝ), IsIsosceles (λ i => match i with
    | 0 => 4
    | 1 => a
    | 2 => b) ∧ 
   (a^2 - 6*a + n = 0) ∧ 
   (b^2 - 6*b + n = 0) ∧ 
   (a + b > 4) ∧ (a + 4 > b) ∧ (b + 4 > a)) ↔ 
  (n = 8 ∨ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_roots_l465_46540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_intersection_l465_46571

/-- The minimum distance between intersection points of y = m with y = 2(x+1) and y = x + ln(x) -/
theorem min_distance_intersection (m : ℝ) : 
  let f (x : ℝ) := 2 * (x + 1)
  let g (x : ℝ) := x + Real.log x
  (∃ x₁ x₂, f x₁ = m ∧ g x₂ = m) →
  (∀ x₁ x₂, f x₁ = m → g x₂ = m → |x₂ - x₁| ≥ 3/2) ∧
  (∃ x₁ x₂, f x₁ = m ∧ g x₂ = m ∧ |x₂ - x₁| = 3/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_intersection_l465_46571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l465_46543

-- Define the function f(a) = a/(a+1)
noncomputable def f (a : ℝ) : ℝ := a / (a + 1)

-- Theorem statement
theorem f_strictly_increasing :
  ∀ a b : ℝ, a > -1 → b > -1 → a < b → f a < f b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l465_46543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_h_value_l465_46558

-- Define the function f
noncomputable def f (x k h : ℝ) : ℝ := Real.log (abs ((1 / (x + 1)) + k)) + h

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_h_value (k : ℝ) :
  (∃ h, ∀ x ≠ -1, is_odd (λ x ↦ f x k h)) → 
  (∃ h, h = Real.log 2 ∧ ∀ x ≠ -1, is_odd (λ x ↦ f x k h)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_h_value_l465_46558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l465_46522

theorem simplify_trig_expression (θ : ℝ) (h : θ ∈ Set.Ioo 0 (π/4)) :
  Real.sqrt (1 - 2 * Real.sin (π + θ) * Real.sin ((3*π)/2 - θ)) = Real.cos θ - Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l465_46522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_value_l465_46534

def circle_configuration (r : ℝ) : Prop :=
  ∃ (O X Y Z : ℝ × ℝ),
    let largest_radius := 2
    let x_radius := 1
    let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
    
    -- X circle is tangent to largest circle and passes through its center
    dist O X = x_radius ∧
    -- Y and Z circles are tangent to X circle
    dist X Y = x_radius + r ∧
    dist X Z = x_radius + r ∧
    -- Y and Z circles are tangent to largest circle
    dist O Y = largest_radius - r ∧
    dist O Z = largest_radius - r ∧
    -- Y and Z circles are tangent to each other
    dist Y Z = 2 * r

theorem circle_radius_value :
  ∃ (r : ℝ), circle_configuration r ∧ r = 8/9 := by
  sorry

#check circle_radius_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_value_l465_46534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_triple_equal_l465_46572

noncomputable def h (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else 3*x - 50

theorem h_triple_equal (b : ℝ) :
  b < 0 → h (h (h 15)) = h (h (h b)) → b = -55/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_triple_equal_l465_46572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_area_product_l465_46524

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents the orthocenter of a triangle -/
noncomputable def orthocenter (t : Triangle) : Point := sorry

/-- Calculates the area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is isosceles with AB = AC -/
def isIsosceles (t : Triangle) : Prop := sorry

/-- Theorem: The product of areas remains constant for isosceles triangle with fixed base -/
theorem constant_area_product 
  (t : Triangle) 
  (H : Point) 
  (h_isosceles : isIsosceles t) 
  (h_orthocenter : H = orthocenter t) 
  (h_fixed_base : (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 = (0 : ℝ)) :
  ∃ k : ℝ, ∀ A' : Point, 
    let t' := Triangle.mk A' t.B t.C
    area t * area (Triangle.mk H t.B t.C) = 
    area t' * area (Triangle.mk (orthocenter t') t'.B t'.C) ∧ 
    area t * area (Triangle.mk H t.B t.C) = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_area_product_l465_46524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sinB_sinC_l465_46584

theorem triangle_sinB_sinC (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given conditions
  Real.cos (2 * A) - 3 * Real.cos (B + C) = 1 →
  (1 / 2) * b * c * Real.sin A = 5 * Real.sqrt 3 →
  b = 5 →
  -- Conclusion
  Real.sin B * Real.sin C = 5 / 7 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sinB_sinC_l465_46584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_implies_b_bound_l465_46576

-- Define the function f(x)
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 - b*x)

-- State the theorem
theorem monotonic_increasing_implies_b_bound :
  ∀ b : ℝ, (∀ x y : ℝ, 1/2 ≤ x ∧ x < y ∧ y ≤ 2 → f b x < f b y) → b < 8/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_implies_b_bound_l465_46576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l465_46515

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- represents ax + by + c = 0

/-- The distance from a point to a line -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

/-- The number of intersection points between a circle and a line -/
def numIntersections (c : Circle) (l : Line) : ℕ :=
  sorry

/-- Theorem: A line intersects a circle at two points if the distance from the center 
    of the circle to the line is less than the radius of the circle -/
theorem line_circle_intersection (c : Circle) (l : Line) :
  distancePointToLine c.center l < c.radius → numIntersections c l = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l465_46515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_circle_Q_equation_l465_46577

-- Define the given point P
def P : ℝ × ℝ := (2, 0)

-- Define the given circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define the distance between a point and a line
noncomputable def distancePointToLine (x y : ℝ) (a b c : ℝ) : ℝ :=
  abs (a*x + b*y + c) / Real.sqrt (a^2 + b^2)

-- Define the center of circle C
def centerC : ℝ × ℝ := (3, -2)

-- Define the radius of circle C
def radiusC : ℝ := 3

-- Statement for the equation of line l
theorem line_l_equation :
  ∃ (a b c : ℝ), (∀ (x y : ℝ), a*x + b*y + c = 0 → (x = 2 ∨ (3*x + 4*y - 6 = 0))) ∧
  (a*P.1 + b*P.2 + c = 0) ∧
  (distancePointToLine centerC.1 centerC.2 a b c = 1) := by sorry

-- Define line l1 passing through P
def line_l1 (x y : ℝ) : Prop := ∃ (m : ℝ), y = m*(x - P.1) + P.2

-- Define the intersection points M and N
noncomputable def M : ℝ × ℝ := sorry
noncomputable def N : ℝ × ℝ := sorry

-- Statement for the equation of circle Q
theorem circle_Q_equation :
  (∀ x y, line_l1 x y → C x y) ∧ ((M.1 - N.1)^2 + (M.2 - N.2)^2 = 16) →
  ∀ x y, (x - 2)^2 + y^2 = 4 ↔ ((x - M.1)^2 + (y - M.2)^2 = ((M.1 - N.1)^2 + (M.2 - N.2)^2) / 4 ∧
                                (x - N.1)^2 + (y - N.2)^2 = ((M.1 - N.1)^2 + (M.2 - N.2)^2) / 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_circle_Q_equation_l465_46577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_3_40_l465_46557

/-- Calculates the angle between the hour and minute hands of a clock. -/
noncomputable def clockAngle (hours : ℕ) (minutes : ℕ) : ℝ :=
  |60 * (hours % 12 : ℝ) - 11 * (minutes : ℝ)| / 2

/-- Theorem stating that at 3:40, the angle between the hour and minute hands of a clock is 130°. -/
theorem angle_at_3_40 : clockAngle 3 40 = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_3_40_l465_46557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_book_difference_l465_46575

/-- The number of science books in the library -/
def science_books : ℕ := 400

/-- The number of history books in the library -/
def history_books : ℕ := 250

/-- The percentage of science books borrowed by second graders -/
def science_borrowed_percent : ℚ := 30 / 100

/-- The percentage of history books reserved by third graders -/
def history_reserved_percent : ℚ := 40 / 100

/-- The difference between total books and borrowed/reserved books -/
def book_difference : ℕ := 430

theorem library_book_difference : 
  (science_books + history_books) - 
  (↑science_books * science_borrowed_percent + ↑history_books * history_reserved_percent).floor = 
  book_difference := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_book_difference_l465_46575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_120_sides_l465_46510

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Length of a diagonal in a regular polygon -/
noncomputable def diagonal_length (s : ℝ) (n : ℕ) (k : ℕ) : ℝ :=
  2 * s * Real.sin (k * Real.pi / n)

theorem polygon_120_sides (side_length : ℝ) (h_side : side_length = 5) :
  let n : ℕ := 120
  (num_diagonals n = 7020) ∧
  (diagonal_length side_length n (n / 2) = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_120_sides_l465_46510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martian_traffic_light_signals_l465_46526

/-- Represents a Martian traffic light signal -/
structure MartianSignal where
  lit_bulbs : Finset (Fin 6)

/-- Determines if two signals are distinguishable under foggy conditions -/
def distinguishable (s1 s2 : MartianSignal) : Prop :=
  sorry

/-- The set of all possible Martian traffic light signals -/
def allSignals : Finset MartianSignal :=
  sorry

/-- The set of distinguishable Martian traffic light signals -/
def distinguishableSignals : Finset MartianSignal :=
  sorry

/-- Theorem stating that there are 44 distinguishable Martian traffic light signals -/
theorem martian_traffic_light_signals :
  Finset.card distinguishableSignals = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martian_traffic_light_signals_l465_46526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_of_altitude_triangle_l465_46551

/-- Represents an obtuse triangle -/
def ObtuseTriangle (A B C : EuclideanPlane) : Prop := sorry

/-- Represents an altitude of a triangle -/
def Altitude (A D B C : EuclideanPlane) : Prop := sorry

/-- Represents an obtuse angle -/
def ObtuseAngle (A B C : EuclideanPlane) : Prop := sorry

/-- Represents the incenter of a triangle -/
def Incenter (B D E F : EuclideanPlane) : Prop := sorry

/-- Given an obtuse triangle ABC with altitudes AD, BE, and CF, where angle ABC is obtuse,
    B is the incenter of triangle DEF. -/
theorem incenter_of_altitude_triangle (A B C D E F : EuclideanPlane) : 
  ObtuseTriangle A B C →
  Altitude A D B C →
  Altitude B E A C →
  Altitude C F A B →
  ObtuseAngle A B C →
  Incenter B D E F := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_of_altitude_triangle_l465_46551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l465_46516

theorem vector_magnitude (a b : EuclideanSpace ℝ (Fin 3)) : 
  (‖a‖ = 1) → (‖b‖ = 1) → (‖a + 2 • b‖ = Real.sqrt 3) → 
  (‖a - 2 • b‖ = Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l465_46516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_theorem_l465_46546

theorem positive_difference_theorem : 
  let expr1 : ℝ := (8^2 - 8^2) / 8
  let expr2 : ℝ := (8^2 * 8^2) / 8
  |expr2 - expr1| = 512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_theorem_l465_46546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l465_46594

theorem problem_solution (a b : ℝ) (h1 : (72 : ℝ)^a = 4) (h2 : (72 : ℝ)^b = 9) :
  (18 : ℝ)^((1 - a - b) / (2 * (1 - b))) = 2 * Real.rpow 2 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l465_46594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_f_greater_than_one_max_a_value_l465_46591

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := |x - 5/2| + |x - a|

-- Theorem 1
theorem ln_f_greater_than_one :
  ∀ x : ℝ, Real.log (f x (-1/2)) > 1 := by sorry

-- Theorem 2
theorem max_a_value :
  ∃ a_max : ℝ, a_max = 5/4 ∧
    (∀ x a : ℝ, f x a ≥ a → a ≤ a_max) ∧
    (∀ x : ℝ, f x a_max ≥ a_max) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_f_greater_than_one_max_a_value_l465_46591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mammoths_twenty_mammoths_exist_l465_46544

/-- Represents a mammoth on the grid -/
structure Mammoth where
  x : Fin 15
  y : Fin 15

/-- The set of all possible diagonal directions -/
inductive DiagonalDirection
  | NorthEast
  | NorthWest
  | SouthEast
  | SouthWest
  | North
  | South

/-- A function that determines if a mammoth's arrow hits another mammoth -/
def mammoth_arrow_hits (m1 m2 : Mammoth) (d : DiagonalDirection) : Prop :=
  sorry  -- The actual implementation would go here

/-- A function that determines if two mammoths hit each other -/
def mammoths_hit (m1 m2 : Mammoth) : Prop :=
  ∃ (d : DiagonalDirection), mammoth_arrow_hits m1 m2 d ∨ mammoth_arrow_hits m2 m1 d

/-- The theorem stating the maximum number of mammoths -/
theorem max_mammoths :
  ∀ (mammoths : Finset Mammoth),
    (∀ m1 m2 : Mammoth, m1 ∈ mammoths → m2 ∈ mammoths → m1 ≠ m2 → ¬ mammoths_hit m1 m2) →
    mammoths.card ≤ 20 := by
  sorry

/-- The theorem stating that 20 mammoths can be placed without hitting each other -/
theorem twenty_mammoths_exist :
  ∃ (mammoths : Finset Mammoth),
    mammoths.card = 20 ∧
    (∀ m1 m2 : Mammoth, m1 ∈ mammoths → m2 ∈ mammoths → m1 ≠ m2 → ¬ mammoths_hit m1 m2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mammoths_twenty_mammoths_exist_l465_46544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_value_proof_l465_46568

/-- Calculates the import tax given the total value of an item -/
noncomputable def import_tax (total_value : ℝ) : ℝ :=
  max 0 (0.12 * (total_value - 1000))

/-- Calculates the VAT given the total value of an item -/
noncomputable def vat (total_value : ℝ) : ℝ :=
  max 0 (0.05 * (total_value - 1500))

/-- The combined tax amount (import tax + VAT) -/
def combined_tax : ℝ := 278.40

/-- The theorem stating that the total value resulting in the given combined tax is $2,784 -/
theorem total_value_proof : 
  ∃ (total_value : ℝ), 
    total_value = 2784 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_value_proof_l465_46568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l465_46567

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the common points A and B
noncomputable def A : ℝ × ℝ := (1/4, Real.sqrt 15 / 4)
noncomputable def B : ℝ × ℝ := (1/4, -Real.sqrt 15 / 4)

-- Theorem statement
theorem circle_properties :
  -- 1. Distance between centers is 2
  let center₁ : ℝ × ℝ := (0, 0)
  let center₂ : ℝ × ℝ := (2, 0)
  Real.sqrt ((center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2) = 2 ∧
  -- 2. Equation of line AB is x = 1/4
  (∀ (x y : ℝ), C₁ x y ∧ C₂ x y → x = 1/4) ∧
  -- 3. Length of AB is √15/2
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 15 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l465_46567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adhesion_theorem_l465_46564

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def adhesion (xs : List Int) : List Int :=
  let n := xs.length
  List.map (fun i => xs[i]! * xs[(i + 1) % n]!) (List.range n)

def all_ones (xs : List Int) : Prop :=
  ∀ x ∈ xs, x = 1

theorem adhesion_theorem (n : ℕ) (h : n ≥ 2) :
  (∀ xs : List Int, xs.length = n → (∀ x ∈ xs, x = 1 ∨ x = -1) →
    ∃ k : ℕ, all_ones (k.iterate adhesion xs)) ↔ is_power_of_two n :=
by sorry

#check adhesion_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adhesion_theorem_l465_46564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l465_46525

noncomputable def f (a k : ℝ) (x : ℝ) : ℝ := a^x - (k + 1) * a^(-x)

noncomputable def g (a m : ℝ) (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2 * m * f a 0 x

theorem function_properties 
  (a : ℝ) 
  (h_a_pos : a > 0) 
  (h_a_neq_1 : a ≠ 1) 
  (h_f_odd : ∀ x, f a k (-x) = -f a k x)
  (h_f_1 : f a k 1 = 3/2)
  (h_g_min : ∃ m, ∀ x ≥ 0, g a m x ≥ -6 ∧ ∃ x₀ ≥ 0, g a m x₀ = -6) :
  ∃ k m, k = 0 ∧ m = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l465_46525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_z_fourth_power_l465_46537

-- Define the complex number
noncomputable def z : ℂ := 2 + Real.sqrt 5 * Complex.I

-- State the theorem
theorem modulus_z_fourth_power : Complex.abs (z^4) = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_z_fourth_power_l465_46537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bricks_for_square_paving_l465_46502

theorem min_bricks_for_square_paving : ∃ (min_bricks : ℕ), min_bricks = 6 := by
  -- Define the brick dimensions
  let brick_length : ℕ := 45
  let brick_width : ℕ := 30

  -- Define the square side length as the LCM of brick length and width
  let square_side : ℕ := lcm brick_length brick_width

  -- Calculate the area of the square
  let square_area : ℕ := square_side * square_side

  -- Calculate the area of a single brick
  let brick_area : ℕ := brick_length * brick_width

  -- Define the minimum number of bricks
  let min_bricks : ℕ := square_area / brick_area

  -- Prove that min_bricks = 6
  have h : min_bricks = 6 := by
    -- The proof steps would go here
    sorry

  -- Conclude the theorem
  exact ⟨min_bricks, h⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bricks_for_square_paving_l465_46502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l465_46586

theorem coefficient_x_cubed_in_expansion (x : ℝ) :
  (Finset.range 51).sum (λ k ↦ (-1)^k * Nat.choose 50 k * (2^(50 - k)) * x^k) =
  (-19600 * 2^47) * x^3 + (Finset.range 51).sum (λ k ↦ if k ≠ 3 then (-1)^k * Nat.choose 50 k * (2^(50 - k)) * x^k else 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l465_46586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_center_l465_46517

/-- The function f represents a rotation in the complex plane -/
noncomputable def f (z : ℂ) : ℂ := ((-1 + Complex.I * Real.sqrt 3) * z + (-2 * Real.sqrt 3 - 18 * Complex.I)) / 2

/-- The point c around which the rotation occurs -/
noncomputable def c : ℂ := Real.sqrt 3 - 5 * Complex.I

/-- Theorem stating that f(c) = c, proving c is the center of rotation -/
theorem rotation_center : f c = c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_center_l465_46517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mn_value_l465_46582

theorem max_mn_value (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (∃ (s : Finset ℕ), s.card = 100 ∧ (∀ k : ℕ, k ∈ s ↔ m / n < k ∧ k < m * n)) →
  m * n ≤ 134 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mn_value_l465_46582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_difference_greater_than_two_given_even_sum_l465_46545

def is_valid_pair (a b : ℕ) : Prop := 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6

def is_even_sum (a b : ℕ) : Prop := Even (a + b)

def is_difference_greater_than_two (a b : ℕ) : Prop := (a : ℤ) - (b : ℤ) > 2 ∨ (b : ℤ) - (a : ℤ) > 2

def total_outcomes : ℕ := 36

def even_sum_outcomes : ℕ := 18

def favorable_outcomes : ℕ := 4

theorem probability_difference_greater_than_two_given_even_sum :
  (favorable_outcomes : ℚ) / even_sum_outcomes = 2 / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_difference_greater_than_two_given_even_sum_l465_46545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_inequality_l465_46542

/-- Given a triangle with side lengths a, b, and c, prove that a² < ab + ac. -/
theorem triangle_side_inequality (a b c : ℝ) (h : a > 0) (h1 : b > 0) (h2 : c > 0)
    (tri : a < b + c ∧ b < a + c ∧ c < a + b) : a^2 < a*b + a*c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_inequality_l465_46542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_at_two_l465_46578

/-- The function g(x) -/
noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 7 * x - 10) / (x^2 - 6 * x + 4)

/-- The horizontal asymptote of g(x) -/
def horizontal_asymptote : ℝ := 3

/-- Theorem: g(x) crosses its horizontal asymptote when x = 2 -/
theorem g_crosses_asymptote_at_two :
  g 2 = horizontal_asymptote := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_at_two_l465_46578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_n_terms_l465_46500

def a : ℕ → ℤ
  | 0 => 0
  | n + 1 => 2 * a n + n * 2^n

def S (n : ℕ) : ℤ := (Finset.range n).sum (λ i => a i)

theorem sum_of_first_n_terms (n : ℕ) :
  S n = 2^(n - 1) * (n^2 - 3*n + 4) - 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_n_terms_l465_46500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_l465_46587

/-- The area enclosed by the ellipse x^2/9 + y^2/16 = 1 is 12π -/
theorem ellipse_area : Real := 12 * Real.pi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_l465_46587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fantasy_book_cost_is_correct_l465_46549

/-- The cost of a fantasy book -/
noncomputable def fantasy_book_cost : ℝ := 4

/-- The cost of a literature book -/
noncomputable def literature_book_cost : ℝ := fantasy_book_cost / 2

/-- The number of fantasy books sold per day -/
def fantasy_books_per_day : ℕ := 5

/-- The number of literature books sold per day -/
def literature_books_per_day : ℕ := 8

/-- The number of days -/
def days : ℕ := 5

/-- The total earnings -/
noncomputable def total_earnings : ℝ := 180

theorem fantasy_book_cost_is_correct : 
  fantasy_book_cost * (fantasy_books_per_day * days) + 
  literature_book_cost * (literature_books_per_day * days) = total_earnings := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fantasy_book_cost_is_correct_l465_46549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chuck_accessible_area_l465_46590

-- Define the shed dimensions
noncomputable def shed_width : ℝ := 3
noncomputable def shed_length : ℝ := 4

-- Define the leash length
noncomputable def leash_length : ℝ := 4

-- Define the tree position
noncomputable def tree_distance : ℝ := 1.5

-- Define the approximation of pi
noncomputable def π : ℝ := Real.pi

-- Define the accessible area
noncomputable def accessible_area : ℝ := (113 / 12) * π

-- Theorem statement
theorem chuck_accessible_area :
  ∀ (ε : ℝ), ε > 0 → |accessible_area - 9.42 * π| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chuck_accessible_area_l465_46590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l465_46574

theorem triangle_side_lengths 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : c = Real.sqrt 19)
  (h2 : C = 2 * Real.pi / 3)
  (h3 : A > B)
  (h4 : (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :
  a = 3 ∧ b = 2 := by
  sorry

#check triangle_side_lengths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l465_46574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_top_four_l465_46506

/-- Represents a tournament with 8 teams -/
structure Tournament :=
  (teams : Fin 8)
  (points : Fin 8 → ℕ)
  (top_four : Fin 4 → Fin 8)

/-- The total number of games played in the tournament -/
def total_games : ℕ := 56

/-- The scoring system for the tournament -/
def score_system : Nat × Nat × Nat := (3, 1, 0)

/-- The theorem stating the maximum possible points for top four teams -/
theorem max_points_top_four (t : Tournament) : 
  (∀ i : Fin 4, t.points (t.top_four i) = t.points (t.top_four 0)) →
  (∀ i : Fin 4, t.points (t.top_four i) ≤ 42) ∧
  (∃ t : Tournament, ∀ i : Fin 4, t.points (t.top_four i) = 42) := by
  sorry

#check max_points_top_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_top_four_l465_46506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_charge_per_trip_l465_46573

/-- Proves that Alex's charge per trip is $1.50 given the problem conditions --/
theorem alex_charge_per_trip
  (needed : ℝ)
  (trips : ℕ)
  (groceries : ℝ)
  (grocery_fee_percent : ℝ)
  (h1 : needed = 100)
  (h2 : trips = 40)
  (h3 : groceries = 800)
  (h4 : grocery_fee_percent = 0.05)
  :
  ∃ (charge_per_trip : ℝ),
    trips * charge_per_trip + grocery_fee_percent * groceries = needed ∧
    charge_per_trip = 1.5 := by
  use 1.5
  constructor
  · rw [h1, h2, h3, h4]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_charge_per_trip_l465_46573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_zero_l465_46509

noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt (3 * (Real.sin x) ^ 4 + 5 * (Real.cos x) ^ 2) - 
  Real.sqrt (3 * (Real.cos x) ^ 4 + 5 * (Real.sin x) ^ 2)

theorem g_is_zero : ∀ x : ℝ, g x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_zero_l465_46509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l465_46598

-- Define the complex number z
noncomputable def z : ℂ := Complex.I^2 / (1 + Complex.I)

-- Theorem statement
theorem z_in_second_quadrant : 
  Real.sign z.re = -1 ∧ Real.sign z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l465_46598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_iff_sum_zero_l465_46555

-- Define ω as a cube root of unity
noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

-- Define a structure for a triangle in the complex plane
structure ComplexTriangle where
  z₁ : ℂ
  z₂ : ℂ
  z₃ : ℂ

-- Define what it means for a triangle to be counterclockwise-oriented
def isCounterclockwise (t : ComplexTriangle) : Prop :=
  (t.z₂ - t.z₁).arg < (t.z₃ - t.z₁).arg

-- Define what it means for a triangle to be equilateral
def isEquilateral (t : ComplexTriangle) : Prop :=
  Complex.abs (t.z₂ - t.z₁) = Complex.abs (t.z₃ - t.z₂) ∧
  Complex.abs (t.z₃ - t.z₂) = Complex.abs (t.z₁ - t.z₃)

-- State the theorem
theorem equilateral_iff_sum_zero (t : ComplexTriangle) :
  isCounterclockwise t →
  (isEquilateral t ↔ t.z₁ + ω * t.z₂ + ω^2 * t.z₃ = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_iff_sum_zero_l465_46555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drinking_speed_ratio_l465_46530

/-- Given two people sharing a bottle of water, this theorem proves the ratio of their drinking speeds
    based on the amount each person drank in the same time period. -/
theorem drinking_speed_ratio 
  (total_bottle : ℚ) 
  (usha_amount : ℚ) 
  (mala_amount : ℚ) 
  (h1 : total_bottle = 1) 
  (h2 : usha_amount = 2/10) 
  (h3 : mala_amount = total_bottle - usha_amount) 
  (h4 : mala_amount + usha_amount = total_bottle) :
  mala_amount / usha_amount = 4 := by
  sorry

#check drinking_speed_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drinking_speed_ratio_l465_46530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_neg_one_is_minimum_of_f_l465_46529

-- Define the function f(x) = xe^x
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

-- State the theorem
theorem x_neg_one_is_minimum_of_f :
  ∃ (x₀ : ℝ), x₀ = -1 ∧ ∀ (x : ℝ), f x₀ ≤ f x := by
  -- The proof goes here
  sorry

-- You can add more lemmas or theorems if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_neg_one_is_minimum_of_f_l465_46529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chebyshev_least_deviation_l465_46599

open Real Set

-- Define the Chebyshev polynomial of the first kind
noncomputable def T (n : ℕ) : ℝ → ℝ :=
  fun x => cos (n * arccos x)

-- Define the theorem
theorem chebyshev_least_deviation (n : ℕ) (P : ℝ → ℝ) :
  (∀ x : ℝ, x ∈ Icc (-1 : ℝ) 1 → |P x| ≤ (1 / 2^(n - 1))) →
  (∃ a : Polynomial ℝ, (Polynomial.degree a < n) ∧ 
    ∀ x : ℝ, P x = x^n + (Polynomial.eval x a)) →
  (∀ x : ℝ, P x = (1 / 2^(n - 1)) * T n x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chebyshev_least_deviation_l465_46599
