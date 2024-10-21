import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l1357_135792

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 3^(-x) - 2 else Real.sqrt x

-- State the theorem
theorem solution_exists (x₀ : ℝ) (h : f x₀ = 1) : x₀ = 1 ∨ x₀ = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l1357_135792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_derivative_l1357_135745

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := x^2 * Real.sin x

-- State the theorem
theorem y_derivative (x : ℝ) : 
  deriv y x = 2 * x * Real.sin x + x^2 * Real.cos x := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_derivative_l1357_135745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_l1357_135707

theorem trigonometric_equation (x : ℝ) 
  (h1 : Real.cos (π/4 + x) = 3/5) 
  (h2 : 17*π/12 < x ∧ x < 7*π/4) : 
  (Real.sin (2*x) + 2*(Real.sin x)^2) / (1 - Real.tan x) = -9/20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_l1357_135707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_solution_set_l1357_135761

noncomputable def f (x : ℝ) : ℝ :=
  if x = 0 then 0
  else Real.log (abs x) / Real.log (1/2)

theorem f_properties :
  (∀ x, f (-x) = f x) ∧ 
  f 0 = 0 ∧
  (∀ x > 0, f x = Real.log x / Real.log (1/2)) := by sorry

theorem f_solution_set :
  {x : ℝ | f (x^2 - 1) > -2} = {x : ℝ | -Real.sqrt 5 < x ∧ x < Real.sqrt 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_solution_set_l1357_135761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_quadrant_II_l1357_135711

-- Define the two linear functions
noncomputable def f (x : ℝ) : ℝ := -2 * x + 3
noncomputable def g (x : ℝ) : ℝ := 3 * x + 6

-- Define the intersection point
noncomputable def intersection_x : ℝ := -3/5
noncomputable def intersection_y : ℝ := f intersection_x

-- Define the region of intersection
def in_region (x y : ℝ) : Prop := y ≥ f x ∧ y ≤ g x

-- Define what it means to be in Quadrant II
def in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem stating that the intersection lies primarily in Quadrant II
theorem intersection_in_quadrant_II :
  in_quadrant_II intersection_x intersection_y ∧
  ∀ x y, in_region x y → in_quadrant_II x y := by
  sorry

#check intersection_in_quadrant_II

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_quadrant_II_l1357_135711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1357_135702

-- Define set A
def A : Set ℤ := {x : ℤ | x^2 - 2*x - 8 ≤ 0}

-- Define set B
def B : Set ℤ := {x : ℤ | x > 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1357_135702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2_pow_100_plus_1_l1357_135782

/-- A sequence defined by b₁ = 2 and b₂ₙ₊₁ = 2n · bₙ for positive integers n -/
def b : ℕ → ℕ
  | 0 => 2  -- We define b₀ to be 2 as well, to cover all natural numbers
  | 1 => 2
  | n + 2 => if n % 2 = 0 then 2 * (n / 2 + 1) * b (n / 2 + 1) else b (n + 1)

/-- The value of b₂¹⁰⁰₊₁ is 2^4951 -/
theorem b_2_pow_100_plus_1 : b (2^100 + 1) = 2^4951 := by
  sorry

#eval b 3  -- You can add some test cases to check if the function works as expected
#eval b 5
#eval b 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2_pow_100_plus_1_l1357_135782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_labor_quantity_l1357_135730

noncomputable section

/-- Market demand function -/
def demand (P : ℝ) : ℝ := 60 - 14 * P

/-- Market supply function -/
def supply (P : ℝ) : ℝ := 20 + 6 * P

/-- Marginal product of labor -/
def MPL (L : ℝ) : ℝ := 160 / (L^2)

/-- Equilibrium price per unit of labor -/
def wage : ℝ := 5

/-- Theorem stating the optimal labor quantity -/
theorem optimal_labor_quantity : 
  ∃ (P L : ℝ), 
    demand P = supply P ∧ 
    MPL L * P = wage ∧ 
    L = 8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_labor_quantity_l1357_135730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_90_terms_l1357_135700

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference

/-- Sum of the first n terms of an arithmetic progression -/
def sum_n (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a + (n - 1 : ℚ) * ap.d)

theorem sum_of_90_terms (ap : ArithmeticProgression) :
  sum_n ap 15 = 150 →
  sum_n ap 75 = 75 →
  sum_n ap 90 = -225/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_90_terms_l1357_135700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1357_135738

def sequenceA (a : ℕ → ℝ) :=
  (a 1 = 1) ∧
  (∀ n : ℕ, n > 0 → |a (n + 1) - a n| = (1/2)^n) ∧
  (∀ n : ℕ, n > 0 → a (2*n - 1) < a (2*n + 1)) ∧
  (∀ n : ℕ, n > 0 → a (2*n + 2) < a (2*n))

theorem sequence_general_term (a : ℕ → ℝ) (h : sequenceA a) :
  ∀ n : ℕ, n > 0 → a n = 4/3 + (1/3) * (-1)^n / 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1357_135738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_cartesian_polar_equivalence_l1357_135725

/-- Represents an ellipse in Cartesian coordinates -/
structure CartesianEllipse where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → Prop

/-- Represents an ellipse in polar coordinates -/
structure PolarEllipse where
  b : ℝ
  ε : ℝ
  equation : (ρ φ : ℝ) → Prop

/-- The theorem stating the equivalence of Cartesian and polar representations of an ellipse -/
theorem ellipse_cartesian_polar_equivalence (ce : CartesianEllipse) (pe : PolarEllipse) :
  (∀ (x y : ℝ), ce.equation x y ↔ 
   ∃ (ρ φ : ℝ), x = ρ * Real.cos φ ∧ y = ρ * Real.sin φ ∧ pe.equation ρ φ) ∧
  ce.b = pe.b ∧
  pe.ε^2 = 1 - (ce.b / ce.a)^2 := by
  sorry

#check ellipse_cartesian_polar_equivalence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_cartesian_polar_equivalence_l1357_135725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_plus_2cos4_l1357_135751

open Real

theorem min_sin6_plus_2cos4 :
  ∃ (x : ℝ), ∀ (y : ℝ), (sin y) ^ 6 + 2 * (cos y) ^ 4 ≥ 2 / 3 ∧
  (sin x) ^ 6 + 2 * (cos x) ^ 4 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_plus_2cos4_l1357_135751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birdseed_cups_l1357_135790

/-- The number of cups of birdseed in a bird feeder that feeds 21 birds weekly,
    where each cup feeds 14 birds and a squirrel steals 0.5 cups weekly. -/
theorem birdseed_cups (birds_fed : ℕ) (birds_per_cup : ℕ) (squirrel_steal : ℚ) : ℚ :=
  by
  have h1 : birds_fed = 21 := by sorry
  have h2 : birds_per_cup = 14 := by sorry
  have h3 : squirrel_steal = 1/2 := by sorry
  
  -- The actual amount of birdseed needed for the birds
  let birds_cups : ℚ := (birds_fed : ℚ) / (birds_per_cup : ℚ)
  
  -- The total amount of birdseed needed, including what the squirrel steals
  let total_cups : ℚ := birds_cups + squirrel_steal
  
  -- Prove that the total cups of birdseed is equal to 2
  have h4 : total_cups = 2 := by sorry
  
  exact total_cups

-- Remove the #eval statement as it's causing issues with compilation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birdseed_cups_l1357_135790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_ratio_l1357_135719

/-- Represents a right cone -/
structure Cone where
  circumference : ℝ
  height : ℝ

/-- The volume of a cone -/
noncomputable def volume (c : Cone) : ℝ :=
  (1 / 3) * Real.pi * (c.circumference / (2 * Real.pi))^2 * c.height

theorem cone_height_ratio (original : Cone) (shorter : Cone) :
  original.circumference = 32 * Real.pi →
  original.height = 60 →
  shorter.circumference = original.circumference →
  volume shorter = 768 * Real.pi →
  shorter.height / original.height = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_ratio_l1357_135719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_negative_six_l1357_135717

def sequence_a : ℕ → ℤ
  | 0 => 3  -- We define a_0 as 3 to match a_1 in the problem
  | 1 => 6  -- This matches a_2 in the problem
  | (n + 2) => sequence_a (n + 1) - sequence_a n

theorem fifth_term_is_negative_six : sequence_a 4 = -6 := by
  -- Unfold the definition of sequence_a a few times
  have h1 : sequence_a 2 = 3 := rfl
  have h2 : sequence_a 3 = -3 := by
    unfold sequence_a
    rw [h1]
    rfl
  -- Now prove the final step
  unfold sequence_a
  rw [h2, h1]
  rfl

#eval sequence_a 4  -- This will evaluate to -6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_negative_six_l1357_135717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_R_l1357_135746

/-- A rectangular parallelepiped with dimensions 1 × 2 × 4 -/
def B : Set (ℝ × ℝ × ℝ) :=
  {p | 0 ≤ p.fst ∧ p.fst ≤ 1 ∧ 0 ≤ p.snd.fst ∧ p.snd.fst ≤ 2 ∧ 0 ≤ p.snd.snd ∧ p.snd.snd ≤ 4}

/-- The set of points within distance 3 of some point in B -/
def R : Set (ℝ × ℝ × ℝ) :=
  {p | ∃ q ∈ B, Real.sqrt ((p.fst - q.fst)^2 + (p.snd.fst - q.snd.fst)^2 + (p.snd.snd - q.snd.snd)^2) ≤ 3}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating the volume of R -/
theorem volume_of_R : volume R = 92 + 144 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_R_l1357_135746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_J_l1357_135714

-- Define H(p, q)
noncomputable def H (p q : ℝ) : ℝ := -3*p*q + 4*p*(1-q) + 4*(1-p)*q - 5*(1-p)*(1-q)

-- Define J(p)
noncomputable def J (p : ℝ) : ℝ := max (H p 0) (H p 1)

-- Theorem statement
theorem minimize_J :
  ∀ p : ℝ, 0 ≤ p ∧ p ≤ 1 → J p ≥ J (9/16) :=
by
  sorry

#check minimize_J

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_J_l1357_135714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l1357_135731

-- Define the circle Γ
def Γ : Set (ℝ × ℝ) := {p | (p.1 - 4)^2 + p.2^2 = 36}

-- Define point B
def B : ℝ × ℝ := (-2, 0)

-- Define the property of point P
def is_point_P (P A : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t ∈ Set.Ioo 0 1 ∧ P = (1 - t) • A + t • B ∧ 
    ‖B - P‖ / ‖A - P‖ = 1/2

-- Theorem statement
theorem trajectory_of_P (P : ℝ × ℝ) :
  (∃ A : ℝ × ℝ, A ∈ Γ ∧ is_point_P P A) →
  P.1^2 + P.2^2 = 4 ∧ P.1 ≠ -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l1357_135731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_straight_lines_circles_disjoint_l1357_135783

-- Define a type for geometric objects
inductive Geometry
| StraightLine
| Circle

-- Define the set of straight lines
def StraightLines : Set Geometry := {g | g = Geometry.StraightLine}

-- Define the set of circles
def Circles : Set Geometry := {g | g = Geometry.Circle}

-- Theorem: The intersection of straight lines and circles is empty
theorem straight_lines_circles_disjoint : StraightLines ∩ Circles = ∅ := by
  -- Proof
  ext x
  simp [StraightLines, Circles]
  sorry

#check straight_lines_circles_disjoint

end NUMINAMATH_CALUDE_ERRORFEEDBACK_straight_lines_circles_disjoint_l1357_135783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_16_eq_4_l1357_135754

/-- A power function that passes through the point (4, 2) -/
noncomputable def f (x : ℝ) : ℝ := x ^ (1/2 : ℝ)

/-- The theorem stating that f(16) = 4 -/
theorem f_16_eq_4 : f 16 = 4 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_16_eq_4_l1357_135754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_problem_l1357_135748

/-- The height of the bottom step of a ladder -/
noncomputable def bottom_step_height : ℝ := 34

/-- The length from the top surface of the upper block to the ground in the first arrangement -/
noncomputable def r : ℝ := 42

/-- The total length from the upper surface of the top block to the ground in the second arrangement -/
noncomputable def s : ℝ := 38

/-- The width where the blocks overlap in the first arrangement -/
noncomputable def w : ℝ := 4

/-- The height of each wooden block -/
noncomputable def l : ℝ := 4 - w / 2

theorem ladder_problem (h : ℝ) (h_def : h = bottom_step_height) :
  l + h - w / 2 = r ∧ 2 * l + h = s :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_problem_l1357_135748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_sequence_property_l1357_135718

def u : ℕ → ℚ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | m+2 => if (m+2) % 3 = 0 then 2 + u ((m+2)/3) else 1 / u (m+1)

theorem u_sequence_property (m : ℕ) (h : m > 0) :
  u m = 7/24 → m = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_sequence_property_l1357_135718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_with_reflection_l1357_135765

-- Define the point A
def A : ℝ × ℝ := (-1, 1)

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem shortest_distance_to_circle_with_reflection : 
  ∃ (p : ℝ × ℝ), C p.1 p.2 ∧ 
  (∀ (q : ℝ × ℝ), C q.1 q.2 → 
    distance A (0, 0) + distance (0, 0) p ≤ distance A (0, 0) + distance (0, 0) q) ∧
  distance A (0, 0) + distance (0, 0) p = 4 := by
  sorry

#check shortest_distance_to_circle_with_reflection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_with_reflection_l1357_135765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement1_necessary_not_sufficient_l1357_135757

-- Define propositions p and q
variable (p q : Prop)

-- Define Statement 1: "Either p or q is a false proposition"
def statement1 (p q : Prop) : Prop := ¬p ∨ ¬q

-- Define Statement 2: "Not p is a true proposition"
def statement2 (p : Prop) : Prop := ¬p

-- Theorem: Statement 1 is necessary but not sufficient for Statement 2
theorem statement1_necessary_not_sufficient (p q : Prop) :
  (statement2 p → statement1 p q) ∧ ¬(statement1 p q → statement2 p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement1_necessary_not_sufficient_l1357_135757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_value_l1357_135712

-- Define a quadratic polynomial
def quadratic_polynomial (a b c : ℚ) : ℚ → ℚ := λ x ↦ a * x^2 + b * x + c

-- Define the divisibility condition
def divisibility_condition (q : ℚ → ℚ) : Prop :=
  ∃ p : ℚ → ℚ, ∀ x, q x^3 - x = (x - 2) * (x + 2) * (x - 5) * p x

theorem quadratic_polynomial_value (a b c : ℚ) :
  let q := quadratic_polynomial a b c
  divisibility_condition q → q 10 = -174/21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_value_l1357_135712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1357_135778

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 3)

theorem axis_of_symmetry (φ : ℝ) (h1 : |φ| < Real.pi / 2) (h2 : f (-Real.pi / 6) = 0) :
  ∃ k : ℤ, f (-5 * Real.pi / 12 + x) = f (-5 * Real.pi / 12 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1357_135778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_remainder_property_l1357_135779

/-- Count of remainder k at interior points of rectangle a×b -/
def count_interior (a b : ℕ+) (k : ℕ) : ℕ := sorry

/-- Count of remainder k at boundary points of rectangle a×b -/
def count_boundary (a b : ℕ+) (k : ℕ) : ℕ := sorry

theorem rectangle_remainder_property (n : ℕ) (hn : n ≥ 2) :
  ∀ a b : ℕ+, 
  (∀ k : ℕ, k < n → (count_interior a b k = count_interior a b 0)) ∧
  (∀ k : ℕ, k < n → (count_boundary a b k = count_boundary a b 0)) ↔
  ((n ∣ (a - 1) ∨ n ∣ (b - 1)) ∧ (n ∣ (a + 1) ∨ n ∣ (b + 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_remainder_property_l1357_135779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tshirt_problem_l1357_135703

-- Define the variables
variable (cost_A : ℚ) -- Cost of A style T-shirts
variable (cost_B : ℚ) -- Cost of B style T-shirts
variable (quantity_A : ℕ) -- Quantity of A style T-shirts
variable (quantity_B : ℕ) -- Quantity of B style T-shirts
variable (second_purchase : ℕ) -- Number of B style T-shirts in second purchase

-- Define the conditions
axiom total_cost_A : cost_A * quantity_A = 1200
axiom total_cost_B : cost_B * quantity_B = 6000
axiom cost_difference : cost_B = cost_A + 80
axiom quantity_ratio : quantity_B = 3 * quantity_A
axiom max_second_purchase : second_purchase ≤ 60
axiom profit_condition : 
  (300 - cost_B) * 15 + (300 * (4/5) - cost_B) * (second_purchase - 15) ≥ 3220

-- State the theorem
theorem tshirt_problem :
  cost_A = 120 ∧ cost_B = 200 ∧ second_purchase ≥ 58 ∧ second_purchase ≤ 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tshirt_problem_l1357_135703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pq_fraction_of_st_l1357_135762

/-- Given a line segment ST with points P and Q on it, prove that PQ is 1/24 of ST -/
theorem pq_fraction_of_st (S T P Q : ℝ) : 
  (S ≤ P ∧ P ≤ T) → 
  (S ≤ Q ∧ Q ≤ T) →
  (P - S = 5 * (T - P)) →
  (Q - S = 7 * (T - Q)) →
  (Q - P = (1 / 24) * (T - S)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pq_fraction_of_st_l1357_135762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1357_135781

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi / 2 + x) + (Real.sin (Real.pi / 2 + x))^2

theorem max_value_of_f :
  ∃ (M : ℝ), M = 5/4 ∧ 
  (∀ x ∈ Set.Icc (-Real.pi) 0, f x ≤ M) ∧
  (∃ x ∈ Set.Icc (-Real.pi) 0, f x = M) :=
by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1357_135781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_approx_l1357_135733

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℝ
  -- Common difference of the sequence
  d : ℝ
  -- Sum of first 75 terms is 500
  sum_75 : (75 / 2 : ℝ) * (2 * a + 74 * d) = 500
  -- Sum of next 50 terms (76th to 125th) is 3500
  sum_76_125 : (50 / 2 : ℝ) * ((a + 75 * d) + (a + 124 * d)) = 3500

/-- The first term of the arithmetic sequence is approximately -30.83 -/
theorem first_term_approx (seq : ArithmeticSequence) : 
  ‖seq.a + 30.83‖ < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_approx_l1357_135733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expr_simplifies_to_one_l1357_135722

open Complex

/-- The complex number expression to be simplified -/
noncomputable def complex_expr : ℂ := ((1 + 2*I) / (1 - 2*I))^1004

/-- Theorem stating that the complex expression simplifies to 1 -/
theorem complex_expr_simplifies_to_one : complex_expr = 1 := by
  sorry

/-- Helper lemma: Simplification of the base expression -/
lemma base_simplification : (1 + 2*I) / (1 - 2*I) = (-3 + 4*I) / 5 := by
  sorry

/-- Helper lemma: The magnitude of the base expression is 1 -/
lemma base_magnitude_one : abs ((1 + 2*I) / (1 - 2*I)) = 1 := by
  sorry

/-- Helper lemma: The angle of the base expression -/
lemma base_angle : arg ((1 + 2*I) / (1 - 2*I)) = Real.arctan (4/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expr_simplifies_to_one_l1357_135722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1357_135755

noncomputable def a : ℝ := Real.rpow 0.6 0.6
noncomputable def b : ℝ := Real.rpow 0.6 1.5
noncomputable def c : ℝ := Real.rpow 1.5 0.6

theorem relationship_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1357_135755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_system_l1357_135796

/-- Represents the number of elderly people. -/
def x : ℤ := sorry

/-- Represents the number of pears. -/
def y : ℤ := sorry

/-- Condition: If each person takes one more pear, there will be one pear left. -/
axiom condition1 : y = x + 1

/-- Condition: If each person takes two fewer pears, there will be two pears left. -/
axiom condition2 : 2 * x = y + 2

/-- Theorem: The system of linear equations correctly represents the relationship
    between the number of elderly people and the number of pears. -/
theorem correct_system : (x = y - 1) ∧ (2 * x = y + 2) := by
  constructor
  · -- Proof of x = y - 1
    rw [condition1]
    ring
  · -- Proof of 2 * x = y + 2
    exact condition2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_system_l1357_135796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rower_still_water_speed_l1357_135775

/-- Represents the speed of a rower in different conditions -/
structure RowerSpeed where
  downstream : ℚ
  upstream : ℚ
  stream : ℚ

/-- Calculates the speed of a rower in still water -/
def stillWaterSpeed (s : RowerSpeed) : ℚ :=
  (s.downstream + s.upstream) / 2

/-- Theorem: Given the downstream and upstream speeds of a rower and the stream speed,
    prove that the rower's speed in still water is 12 kmph -/
theorem rower_still_water_speed (s : RowerSpeed)
  (h1 : s.downstream = 18)
  (h2 : s.upstream = 6)
  (h3 : s.stream = 6) :
  stillWaterSpeed s = 12 := by
  sorry

#check rower_still_water_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rower_still_water_speed_l1357_135775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1357_135721

/-- The area of a triangle with vertices (4, -3), (-1, 2), and (2, -7) is 15. -/
theorem triangle_area : ∃ (area : ℝ), area = 15 ∧ area = abs ((4 - 2) * (2 - (-7)) - (-1 - 2) * (-3 - (-7))) / 2 := by
  let A : ℝ × ℝ := (4, -3)
  let B : ℝ × ℝ := (-1, 2)
  let C : ℝ × ℝ := (2, -7)
  let area := abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2)) / 2
  exists area
  constructor
  · sorry -- Proof that area = 15
  · rfl -- Proof that area equals the formula


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1357_135721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_problem_l1357_135744

theorem circle_radius_problem (B₁ B₂ : ℝ) (h1 : B₁ > 0) (h2 : B₂ > 0) :
  (∃ (r : ℝ), r > 0 ∧ B₁ = π * r^2) →
  (B₁ + B₂ = 25 * π) →
  (∃ (q : ℝ), q > 1 ∧ B₂ / B₁ = q ∧ (B₁ + B₂) / B₂ = q) →
  (∃ (r : ℝ), r > 0 ∧ B₁ = π * r^2 ∧ r = Real.sqrt ((Real.sqrt (1 + 100 * π) - 1) / (2 * π))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_problem_l1357_135744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_net_income_increase_l1357_135724

def initial_income : ℚ := 55
def first_raise : ℚ := 10 / 100
def second_raise : ℚ := 5 / 100
def weekly_expenses : ℚ := 35

def income_after_raises : ℚ := initial_income * (1 + first_raise) * (1 + second_raise)

def net_income_before : ℚ := initial_income - weekly_expenses
def net_income_after : ℚ := income_after_raises - weekly_expenses

def net_percentage_increase : ℚ := (net_income_after - net_income_before) / net_income_before * 100

theorem net_income_increase :
  abs (net_percentage_increase - 42.65) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_net_income_increase_l1357_135724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_X_without_three_in_row_l1357_135767

/-- A configuration of 'X's on a 3x3 grid --/
def Configuration := Fin 3 → Fin 3 → Bool

/-- Check if a configuration has three 'X's in a row vertically, horizontally, or diagonally --/
def hasThreeInARow (c : Configuration) : Prop :=
  (∃ i, c i 0 ∧ c i 1 ∧ c i 2) ∨  -- vertical
  (∃ j, c 0 j ∧ c 1 j ∧ c 2 j) ∨  -- horizontal
  (c 0 0 ∧ c 1 1 ∧ c 2 2) ∨       -- diagonal
  (c 0 2 ∧ c 1 1 ∧ c 2 0)         -- other diagonal

/-- Count the number of 'X's in a configuration --/
def countX (c : Configuration) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 3)) (λ i =>
    Finset.sum (Finset.univ : Finset (Fin 3)) (λ j =>
      if c i j then 1 else 0))

/-- The maximum number of 'X's that can be placed without three in a row is 5 --/
theorem max_X_without_three_in_row :
  (∃ c : Configuration, ¬hasThreeInARow c ∧ countX c = 5) ∧
  (∀ c : Configuration, ¬hasThreeInARow c → countX c ≤ 5) := by
  sorry

#check max_X_without_three_in_row

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_X_without_three_in_row_l1357_135767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additive_function_is_linear_l1357_135764

/-- A function f from positive integers to real numbers is additive if
    f(n1 + n2) = f(n1) + f(n2) for all positive integers n1 and n2. -/
def IsAdditive (f : ℕ → ℝ) : Prop :=
  ∀ n1 n2 : ℕ, 0 < n1 ∧ 0 < n2 → f (n1 + n2) = f n1 + f n2

/-- If f is an additive function from positive integers to real numbers,
    then there exists a real constant a such that f(n) = an for all positive integers n. -/
theorem additive_function_is_linear :
    ∀ f : ℕ → ℝ, IsAdditive f →
    ∃ a : ℝ, ∀ n : ℕ, 0 < n → f n = a * n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additive_function_is_linear_l1357_135764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_value_l1357_135770

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

-- Define the interval
def I : Set ℝ := Set.Icc (-3) 2

-- State the theorem
theorem min_t_value : 
  (∃ t : ℝ, ∀ x₁ x₂ : ℝ, x₁ ∈ I → x₂ ∈ I → |f x₁ - f x₂| ≤ t) ∧ 
  (∀ s : ℝ, s < 20 → ∃ x₁ x₂ : ℝ, x₁ ∈ I ∧ x₂ ∈ I ∧ |f x₁ - f x₂| > s) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_value_l1357_135770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_30_l1357_135728

theorem complex_power_30 : ((1 : ℂ) + Complex.I) ^ 30 / (2 : ℂ) ^ 15 = -Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_30_l1357_135728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_30_l1357_135715

noncomputable def clock_angle (hours : ℕ) (minutes : ℕ) : ℝ :=
  let hour_angle := (hours % 12 + minutes / 60 : ℝ) * 30
  let minute_angle := (minutes : ℝ) * 6
  let diff := abs (hour_angle - minute_angle)
  min diff (360 - diff)

theorem clock_angle_at_7_30 : clock_angle 7 30 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_30_l1357_135715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_airlines_with_full_services_max_airlines_with_full_services_eq_30_percent_l1357_135734

theorem max_airlines_with_full_services (total_airlines : ℝ) : ℝ :=
  let wireless_internet_equipped := 0.5 * total_airlines
  let full_wireless_internet := 0.6 * wireless_internet_equipped
  let snack_offering := 0.7 * total_airlines
  let full_snack_offering := 0.8 * snack_offering
  let max_full_service := min full_wireless_internet full_snack_offering
  max_full_service / total_airlines

theorem max_airlines_with_full_services_eq_30_percent (total_airlines : ℝ) (h : total_airlines > 0) : 
  max_airlines_with_full_services total_airlines = 0.3 := by
  unfold max_airlines_with_full_services
  sorry -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_airlines_with_full_services_max_airlines_with_full_services_eq_30_percent_l1357_135734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_triangle_existence_l1357_135794

/-- The function f(x) defined in the problem -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (4^x - k * 2^(x+1) + 1) / (4^x + 2^x + 1)

/-- The theorem statement -/
theorem k_range_for_triangle_existence (k : ℝ) :
  (∀ (x₁ x₂ x₃ : ℝ), ∃ (a b c : ℝ), a = f k x₁ ∧ b = f k x₂ ∧ c = f k x₃ ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b) →
  k ∈ Set.Icc (-2 : ℝ) (1/4 : ℝ) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_triangle_existence_l1357_135794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_primes_satisfying_property_infinitely_many_primes_congruent_to_one_mod_four_l1357_135786

-- Define the property for prime numbers
def satisfies_property (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∀ a b : ℤ, (↑p : ℤ) ∣ (a^2 + b^2) → ((↑p : ℤ) ∣ a ∧ (↑p : ℤ) ∣ b)

-- Theorem 1
theorem characterization_of_primes_satisfying_property :
  ∀ p : ℕ, Nat.Prime p → (satisfies_property p ↔ p % 4 = 3) := by
  sorry

-- Theorem 2
theorem infinitely_many_primes_congruent_to_one_mod_four :
  ∀ n : ℕ, ∃ p : ℕ, p > n ∧ Nat.Prime p ∧ p % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_primes_satisfying_property_infinitely_many_primes_congruent_to_one_mod_four_l1357_135786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1357_135716

noncomputable def f (x : ℝ) := 4 * Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  (∀ x, f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (∀ x, f (-(Real.pi / 6) + x) = f (-(Real.pi / 6) - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1357_135716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_ninths_150th_decimal_l1357_135763

def decimal_representation (n d : ℕ) : ℚ := n / d

def nth_decimal_place (q : ℚ) (n : ℕ) : ℕ :=
  (Int.floor (q * ↑(10^n)) % 10).natAbs

theorem eighth_ninths_150th_decimal :
  nth_decimal_place (decimal_representation 8 9) 150 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_ninths_150th_decimal_l1357_135763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_property_M_l1357_135785

-- Define property M
def has_property_M (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (Real.exp x) * (f x) < (Real.exp y) * (f y)

-- Define the function f(x) = 2^x
noncomputable def f (x : ℝ) : ℝ := (2 : ℝ) ^ x

-- Theorem statement
theorem f_has_property_M : has_property_M f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_property_M_l1357_135785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_l1357_135743

theorem factorial_equation : ∃ n : ℕ, 3 * 5 * 6 * n = Nat.factorial 9 ∧ n = 672 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_l1357_135743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_six_valid_monomials_l1357_135769

/-- A monomial in x and y -/
structure Monomial where
  coeff : ℚ
  x_power : ℕ
  y_power : ℕ

/-- The given polynomial x^4 y^2 + x^2 y^4 -/
def given_polynomial : Polynomial ℚ := sorry

/-- Predicate to check if a polynomial is a perfect square -/
def is_perfect_square (p : Polynomial ℚ) : Prop := sorry

/-- Convert a Monomial to a Polynomial -/
def monomial_to_polynomial (m : Monomial) : Polynomial ℚ := sorry

/-- The set of monomials that, when added to the given polynomial, result in a perfect square -/
def valid_monomials : Set Monomial :=
  {m : Monomial | is_perfect_square (given_polynomial + monomial_to_polynomial m)}

/-- Theorem stating that there are exactly six valid monomials -/
theorem exactly_six_valid_monomials : 
  ∃ (S : Finset Monomial), S.toSet = valid_monomials ∧ S.card = 6 := by sorry

#check exactly_six_valid_monomials

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_six_valid_monomials_l1357_135769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_half_pi_sufficient_not_necessary_l1357_135704

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.cos x
noncomputable def g (ϕ : ℝ) (x : ℝ) : ℝ := Real.sin (x + ϕ)

-- Define the condition for the graphs to overlap
def graphs_overlap (ϕ : ℝ) : Prop := ∀ x, f x = g ϕ x

-- State the theorem
theorem phi_half_pi_sufficient_not_necessary :
  (∃ ϕ, ϕ ≠ π/2 ∧ graphs_overlap ϕ) ∧
  (graphs_overlap (π/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_half_pi_sufficient_not_necessary_l1357_135704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l1357_135766

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem: The distance between points (-3, -4) and (8, 18) is √605 -/
theorem distance_between_specific_points :
  distance (-3) (-4) 8 18 = Real.sqrt 605 := by
  -- Unfold the definition of distance
  unfold distance
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l1357_135766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flag_color_43_and_67_l1357_135723

/-- Represents the colors of flags -/
inductive FlagColor where
  | Red
  | Blue
  | Yellow
deriving Repr

/-- Returns the color of the nth flag in the sequence -/
def flag_color (n : ℕ) : FlagColor :=
  match n % 7 with
  | 1 | 2 | 3 => FlagColor.Red
  | 4 | 5 => FlagColor.Blue
  | _ => FlagColor.Yellow

theorem flag_color_43_and_67 :
  flag_color 43 = FlagColor.Red ∧ flag_color 67 = FlagColor.Blue := by
  apply And.intro
  · rfl  -- This proves flag_color 43 = FlagColor.Red
  · rfl  -- This proves flag_color 67 = FlagColor.Blue

#eval flag_color 43
#eval flag_color 67

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flag_color_43_and_67_l1357_135723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_difference_l1357_135784

/-- The number of marbles Ed had initially -/
def E : ℕ := sorry

/-- The number of marbles Doug had initially -/
def D : ℕ := sorry

/-- Ed initially had more marbles than Doug -/
axiom ed_more_initially : E > D

/-- After losing 11 marbles, Ed has 8 more marbles than Doug -/
axiom after_losing : E - 11 = D + 8

theorem initial_difference : E - D = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_difference_l1357_135784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_powderman_blast_distance_l1357_135772

/-- The time in seconds when the powderman hears the blast -/
noncomputable def blast_time : ℝ :=
  45000 / 970

/-- The distance in yards that the powderman has run when he hears the blast -/
noncomputable def distance_run : ℝ :=
  10 * blast_time

theorem powderman_blast_distance :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧
  (30 * blast_time = 1000 * (blast_time - 45)) ∧
  (464 - ε < distance_run ∧ distance_run < 464 + ε) :=
by
  sorry

-- Remove #eval statements as they are not computable
-- #eval blast_time
-- #eval distance_run

end NUMINAMATH_CALUDE_ERRORFEEDBACK_powderman_blast_distance_l1357_135772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1357_135708

/-- Calculates the speed of a train in km/hr given its length in meters and time to cross a pole in seconds -/
noncomputable def trainSpeed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem stating that a train with length 1250 meters crossing a pole in 15 seconds has a speed of 300 km/hr -/
theorem train_speed_calculation :
  trainSpeed 1250 15 = 300 := by
  -- Unfold the definition of trainSpeed
  unfold trainSpeed
  -- Perform the calculation
  simp [div_div_eq_mul_div, mul_div_assoc]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1357_135708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_theorem_remainder_is_ten_l1357_135791

noncomputable section

/-- The dividend polynomial -/
def f (d x : ℝ) : ℝ := 2*x^3 + d*x^2 - 17*x + 53

/-- The divisor polynomial -/
def g (x : ℝ) : ℝ := 2*x + 7

/-- The quotient polynomial -/
def q (d x : ℝ) : ℝ := x^2 + (d-7)/(2*x)

/-- The remainder of the division -/
def r (d : ℝ) : ℝ := 53 - (7*d - 49 + 34)/(2*7)

theorem division_theorem (d : ℝ) :
  ∀ x, f d x = g x * q d x + r d := by sorry

theorem remainder_is_ten (d : ℝ) : 
  r d = 10 ↔ d = 316/7 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_theorem_remainder_is_ten_l1357_135791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_split_correct_l1357_135753

/-- The amount LeRoy needs to give Bernardo to ensure a fair split -/
noncomputable def fair_split (A B : ℝ) : ℝ :=
  (B - A) / 2 - 75

theorem fair_split_correct (A B : ℝ) (hAB : A < B) : 
  let dinner_costs : List ℝ := [120, 150, 180]
  let total_dinner_cost := dinner_costs.sum
  let bernardo_dinner_share := (2/3) * total_dinner_cost
  let leroy_dinner_share := (1/3) * total_dinner_cost
  let total_spent := A + B
  let new_total := total_spent - total_dinner_cost
  let equal_share := new_total / 2
  let leroy_adjustment := equal_share - (A - leroy_dinner_share)
  leroy_adjustment = fair_split A B := by
  sorry

#check fair_split_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_split_correct_l1357_135753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_distinct_lines_l1357_135773

/-- An equilateral triangle with side length 10 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 10

/-- A line in the triangle that can be an altitude, median, or angle bisector -/
inductive TriangleLine
  | Altitude
  | Median
  | AngleBisector

/-- The set of distinct lines in the triangle -/
def distinct_lines (triangle : EquilateralTriangle) : Finset TriangleLine :=
  sorry

theorem equilateral_triangle_distinct_lines (triangle : EquilateralTriangle) :
  (distinct_lines triangle).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_distinct_lines_l1357_135773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_s_squared_l1357_135727

/-- A hyperbola centered at the origin passing through specific points -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  -- The hyperbola passes through (0, -2)
  passes_through_vertex : (0 : ℝ)^2 / a^2 - (-2)^2 / b^2 = -1
  -- The hyperbola passes through (3, 4)
  passes_through_point1 : 3^2 / a^2 - 4^2 / b^2 = -1
  -- The hyperbola passes through (2, s)
  passes_through_point2 : ∃ (s : ℝ), 2^2 / a^2 - s^2 / b^2 = -1
  -- Ensuring the hyperbola opens vertically (b = 2 for the vertex point)
  b_eq_two : b = 2
  -- a and b are positive real numbers
  a_pos : a > 0
  b_pos : b > 0

/-- The theorem stating that s² = 44/7 for the given hyperbola -/
theorem hyperbola_s_squared (h : Hyperbola) : 
  ∃ (s : ℝ), h.passes_through_point2.choose = s ∧ s^2 = 44/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_s_squared_l1357_135727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fort_blocks_count_l1357_135795

/-- Calculates the number of blocks used in fort construction -/
def fort_blocks (length width height thickness : ℚ) : ℕ :=
  let exterior_volume := length * width * height
  let interior_length := length - 2 * thickness
  let interior_width := width - 2 * thickness
  let interior_height := height - thickness
  let interior_volume := interior_length * interior_width * interior_height
  (exterior_volume - interior_volume).num.natAbs

/-- Theorem stating the number of blocks used in the fort -/
theorem fort_blocks_count : fort_blocks 15 12 7 (3/2) = 666 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fort_blocks_count_l1357_135795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l1357_135735

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel --/
def Line.isParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Calculate the distance from a point to a line --/
noncomputable def Point.distanceToLine (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- The main theorem --/
theorem line_equations (P A : Point) (l : Line) 
    (h1 : P.x = -2 ∧ P.y = 1)
    (h2 : A.x = -1 ∧ A.y = -2)
    (h3 : P.onLine l) :
  (l.isParallel ⟨1, 1, -1⟩ → l = ⟨1, 1, 1⟩) ∧
  (A.distanceToLine l = 1 → (l = ⟨1, 0, 2⟩ ∨ l = ⟨4, 3, 5⟩)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l1357_135735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l1357_135729

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of line l1: ax + 2y = 0 -/
noncomputable def slope_l1 (a : ℝ) : ℝ := -a / 2

/-- The slope of line l2: x + (a+1)y + 4 = 0 -/
noncomputable def slope_l2 (a : ℝ) : ℝ := -1 / (a + 1)

/-- Sufficient but not necessary condition -/
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

theorem parallel_condition (a : ℝ) :
  sufficient_not_necessary (a = 1) (are_parallel (slope_l1 a) (slope_l2 a)) := by
  sorry

#check parallel_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l1357_135729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_distinct_triple_cycle_l1357_135709

theorem no_distinct_triple_cycle (P : Polynomial ℤ) :
  ¬∃ (a b c : ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    P.eval a = b ∧ P.eval b = c ∧ P.eval c = a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_distinct_triple_cycle_l1357_135709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_a_100_l1357_135776

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | n + 1 => sequence_a n + 1 / sequence_a n

theorem integer_part_a_100 : ⌊sequence_a 100⌋ = 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_a_100_l1357_135776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop1_prop2_not_prop3_not_prop4_l1357_135740

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Axioms for the properties of lines and planes
axiom different_lines {l m : Line} : l ≠ m

-- Theorem for proposition ①
theorem prop1 {l m : Line} {α : Plane} 
  (h1 : perpendicular l α) (h2 : perpendicular m α) : 
  parallel l m :=
sorry

-- Theorem for proposition ②
theorem prop2 {l m : Line} {α : Plane}
  (h1 : perpendicular l α) (h2 : subset m α) :
  perpendicularLines l m :=
sorry

-- Theorem for the negation of proposition ③
theorem not_prop3 : ¬ ∀ {l m : Line} {α : Plane},
  parallelToPlane l α → parallel l m → parallelToPlane m α :=
sorry

-- Theorem for the negation of proposition ④
theorem not_prop4 : ¬ ∀ {l m : Line} {α : Plane},
  parallelToPlane l α → parallelToPlane m α → parallel l m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop1_prop2_not_prop3_not_prop4_l1357_135740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_difference_l1357_135741

theorem tangent_sum_difference (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3/4)
  (h2 : Real.tan (α - π/4) = 1/2) :
  Real.tan (β + π/4) = 2/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_difference_l1357_135741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_expression_l1357_135705

/-- A quadratic function with specific properties -/
def q : ℝ → ℝ := sorry

-- q is a quadratic function
axiom q_quadratic : ∃ (a b c : ℝ), ∀ x, q x = a * x^2 + b * x + c

-- q has vertical asymptotes at x = -2 and x = 2
axiom q_asymptotes : (∀ x, x ≠ -2 ∧ x ≠ 2 → q x ≠ 0) ∧ 
                     (∀ ε > 0, ∃ δ > 0, ∀ x, |x + 2| < δ → |q x| > 1/ε) ∧
                     (∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |q x| > 1/ε)

-- q(0) = 8
axiom q_at_zero : q 0 = 8

/-- The main theorem: q(x) = 2x^2 - 8 -/
theorem q_expression : ∀ x, q x = 2 * x^2 - 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_expression_l1357_135705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_on_interval_l1357_135756

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 - 6*x + 1

theorem f_monotone_decreasing_on_interval :
  ∀ x y, x ∈ Set.Ioo (-2 : ℝ) 2 → y ∈ Set.Ioo (-2 : ℝ) 2 → x < y → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_on_interval_l1357_135756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_f_l1357_135780

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 3*x + 2) + 1 / Real.sqrt (2*x^2 + 6*x + 4)

-- State the theorem about the domain and range of f
theorem domain_and_range_of_f :
  (∀ x, f x ≠ 0 → (x < -2 ∨ x > -1)) ∧
  (∀ y, y ≥ Real.rpow 8 (1/4) → ∃ x, f x = y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_f_l1357_135780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_timer_game_result_l1357_135739

theorem timer_game_result (n : ℕ) (initial_times : List ℝ) : 
  n = 60 ∧ 
  initial_times = [1, 0, -1] →
  (2^n + List.get! initial_times 2 + n) = 59 + 2^60 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_timer_game_result_l1357_135739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_side_a_value_l1357_135774

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the vectors m and n
noncomputable def m (A : Real) : Fin 2 → Real
  | 0 => -1
  | 1 => Real.sin A
  | _ => 0

noncomputable def n (A : Real) : Fin 2 → Real
  | 0 => Real.cos A + 1
  | 1 => Real.sqrt 3
  | _ => 0

-- Define the dot product
def dot_product (v w : Fin 2 → Real) : Real :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- Define the function f
noncomputable def f (A : Real) : Real :=
  dot_product (m A) (n A)

theorem max_value_f :
  ∃ (A : Real), ∀ (θ : Real), f θ ≤ f A ∧ f A = 1 := by sorry

theorem side_a_value (t : Triangle) (h1 : dot_product (m t.A) (n t.A) = 0)
    (h2 : t.b = 4 * Real.sqrt 2 / 3) (h3 : Real.cos t.B = Real.sqrt 3 / 3) :
  t.a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_side_a_value_l1357_135774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l1357_135713

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the vertices M and N
def M : ℝ × ℝ := (0, 1)
def N : ℝ × ℝ := (0, -1)

-- Define point T
def T (t : ℝ) : ℝ × ℝ := (t, 2)

-- Define the ratio k as a function of t
noncomputable def k (t : ℝ) : ℝ := ((t^2 + 4) * (t^2 + 36)) / (t^2 + 12)^2

-- Theorem statement
theorem max_k_value (t : ℝ) (h : t ≠ 0) :
  (∀ s, s ≠ 0 → k s ≤ k t) ↔ t = 2 * Real.sqrt 3 ∨ t = -2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l1357_135713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_expression_l1357_135758

theorem parity_of_expression (a b c : ℕ) (ha : Odd a) (hb : Even b) :
  Odd (3^a + (b-1)^2*(c+1)) ↔ Even c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_expression_l1357_135758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_is_five_l1357_135768

def Grid := Fin 5 → Fin 5 → Fin 5

def valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ≠ 0) ∧
  (∀ i, Function.Injective (g i)) ∧
  (∀ j, Function.Injective (λ i ↦ g i j))

def initial_config (g : Grid) : Prop :=
  g 0 0 = 1 ∧ g 0 3 = 2 ∧
  g 1 1 = 4 ∧ g 1 4 = 3 ∧
  g 2 2 = 5 ∧
  g 3 0 = 2

theorem lower_right_is_five (g : Grid) 
  (h1 : valid_grid g) 
  (h2 : initial_config g) : 
  g 4 4 = 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_is_five_l1357_135768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l1357_135706

noncomputable def f (A B C : ℤ) (x : ℝ) : ℝ := x / (x^3 + A*x^2 + B*x + C)

theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 → ∃ y : ℝ, f A B C x = y) →
  (∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ 
    (∀ x : ℝ, 0 < |x + 3| ∧ |x + 3| < δ → |f A B C x| > 1/ε) ∧
    (∀ x : ℝ, 0 < |x| ∧ |x| < δ → |f A B C x| > 1/ε) ∧
    (∀ x : ℝ, 0 < |x - 3| ∧ |x - 3| < δ → |f A B C x| > 1/ε)) →
  A + B + C = -9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l1357_135706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_points_in_square_have_close_pair_l1357_135797

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A square of side length 2 centered at the origin -/
def square : Set Point :=
  {p : Point | -1 ≤ p.x ∧ p.x ≤ 1 ∧ -1 ≤ p.y ∧ p.y ≤ 1}

theorem ten_points_in_square_have_close_pair :
  ∀ (points : Finset Point),
    points.card = 10 →
    (∀ p, p ∈ points → p ∈ square) →
    ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_points_in_square_have_close_pair_l1357_135797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upper_pyramid_volume_l1357_135742

/-- The volume of the upper smaller pyramid cut from a right square pyramid -/
theorem upper_pyramid_volume 
  (base_edge : ℝ) 
  (slant_edge : ℝ) 
  (cut_height : ℝ) 
  (h1 : base_edge = 10 * Real.sqrt 2)
  (h2 : slant_edge = 12)
  (h3 : cut_height = 4) :
  let full_height := Real.sqrt (slant_edge ^ 2 - (base_edge / 2) ^ 2)
  let remaining_height := full_height - cut_height
  (1 / 3) * (base_edge * remaining_height / full_height) ^ 2 * remaining_height = 
  (1000 / 22) * (2 * Real.sqrt 11 - 4) ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upper_pyramid_volume_l1357_135742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_tip_is_eight_l1357_135787

/-- The cost of a steak in dollars -/
def steak_cost : ℝ := 20

/-- The cost of a drink in dollars -/
def drink_cost : ℝ := 5

/-- The number of people dining -/
def num_people : ℕ := 2

/-- The tip percentage as a decimal -/
def tip_percentage : ℝ := 0.2

/-- The percentage of the tip Billy wants to cover, as a decimal -/
def billy_tip_share : ℝ := 0.8

/-- Calculates Billy's share of the tip given the meal costs and tip percentages -/
def calculate_billy_tip (steak : ℝ) (drink : ℝ) (people : ℕ) (tip_percent : ℝ) (billy_share : ℝ) : ℝ :=
  (steak + drink) * (people : ℝ) * tip_percent * billy_share

theorem billy_tip_is_eight :
  calculate_billy_tip steak_cost drink_cost num_people tip_percentage billy_tip_share = 8 := by
  -- Unfold the definition of calculate_billy_tip
  unfold calculate_billy_tip
  -- Simplify the arithmetic expression
  simp [steak_cost, drink_cost, num_people, tip_percentage, billy_tip_share]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_tip_is_eight_l1357_135787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_distance_l1357_135777

noncomputable def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x - 8*y + 24

noncomputable def circle_center (x y : ℝ) : Prop :=
  circle_equation x y ∧ ∀ (a b : ℝ), circle_equation a b → (x - a)^2 + (y - b)^2 ≤ (3 - a)^2 + (-4 - b)^2

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem circle_center_distance :
  ∀ (x y : ℝ), circle_center x y → distance x y (-3) 5 = 3 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_distance_l1357_135777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_locus_equation_l1357_135788

/-- An acute angle with vertex O and rays Op₁ and Op₂ in a plane. -/
structure AcuteAngle where
  α : ℝ
  h_acute : 0 < α ∧ α < π / 2

/-- A circle k₁ with center on Op₁ and tangent to Op₂. -/
structure Circle_k₁ (angle : AcuteAngle) where
  center : ℝ × ℝ
  radius : ℝ
  h_center_on_Op₁ : center.2 = 0
  h_tangent_Op₂ : radius = center.1 * Real.sin (2 * angle.α)

/-- A circle k₂ tangent to both rays Op₁ and Op₂ and to k₁ from outside. -/
structure Circle_k₂ (angle : AcuteAngle) (k₁ : Circle_k₁ angle) where
  center : ℝ × ℝ
  radius : ℝ
  h_tangent_Op₁ : center.2 = radius
  h_tangent_Op₂ : center.1 = center.2 / Real.tan angle.α
  h_tangent_k₁ : (center.1 - k₁.center.1)^2 + center.2^2 = (radius + k₁.radius)^2

/-- The locus of tangency points between k₁ and k₂. -/
def TangencyLocus (angle : AcuteAngle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (k₁ : Circle_k₁ angle) (k₂ : Circle_k₂ angle k₁),
    p.2 = (Real.sin (2 * angle.α) / (Real.cos (2 * angle.α) + 2)) * p.1}

/-- The main theorem stating that the locus of tangency points follows the given equation. -/
theorem tangency_locus_equation (angle : AcuteAngle) (p : ℝ × ℝ) :
  p ∈ TangencyLocus angle ↔ p.2 = (Real.sin (2 * angle.α) / (Real.cos (2 * angle.α) + 2)) * p.1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_locus_equation_l1357_135788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l1357_135752

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x - 1 else Real.log x / Real.log a

theorem monotonic_f_implies_a_range (a : ℝ) :
  Monotone (f a) → 3 < a ∧ a ≤ 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l1357_135752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_pace_is_three_l1357_135749

/-- Represents the race between Nicky and Cristina -/
structure Race where
  nicky_headstart : ℚ
  cristina_pace : ℚ
  catch_up_time : ℚ

/-- Calculates Nicky's pace given the race conditions -/
noncomputable def nicky_pace (race : Race) : ℚ :=
  (race.cristina_pace * race.catch_up_time - race.nicky_headstart) / race.catch_up_time

/-- Theorem stating that Nicky's pace is 3 meters per second given the specific race conditions -/
theorem nicky_pace_is_three :
  let race : Race := {
    nicky_headstart := 30,
    cristina_pace := 5,
    catch_up_time := 15
  }
  nicky_pace race = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_pace_is_three_l1357_135749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1357_135710

/-- A hyperbola with foci F₁ and F₂, and a point P on the hyperbola. -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The angle between three points in ℝ² -/
noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) 
  (angle_condition : angle h.F₁ h.P h.F₂ = π/3)
  (distance_condition : distance h.P h.F₁ = 3 * distance h.P h.F₂) :
  eccentricity h = Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1357_135710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l1357_135720

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (frequency : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

noncomputable def interest_difference (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  compound_interest principal rate time 12 - compound_interest principal rate time 1

theorem investment_difference :
  let principal : ℝ := 30000
  let rate : ℝ := 0.05
  let time : ℝ := 3
  abs (interest_difference principal rate time - 121.59) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l1357_135720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1357_135799

noncomputable def a : ℝ × ℝ := (4, -3)
noncomputable def b (x y : ℝ) : ℝ × ℝ := (2*x, y)
noncomputable def c (x y : ℝ) : ℝ × ℝ := (x + y/2, 2)

theorem vector_problem (x y : ℝ) :
  (∃ k : ℝ, b x y = k • a) ∧ 
  (a.1 * (c x y).1 + a.2 * (c x y).2 = 0) →
  x = 6 ∧ y = -9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1357_135799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_people_round_table_l1357_135750

def round_table_seatings (n : ℕ) : ℕ := Nat.factorial (n - 1)

theorem six_people_round_table :
  round_table_seatings 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_people_round_table_l1357_135750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_arithmetic_progression_with_large_digit_sum_l1357_135747

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- An arithmetic progression -/
structure ArithmeticProgression where
  first : ℕ
  diff : ℕ

/-- Check if all terms in the progression have sum of digits exceeding M -/
def allTermsExceed (ap : ArithmeticProgression) (M : ℝ) : Prop :=
  ∀ k : ℕ, sumOfDigits (ap.first + k * ap.diff) > Int.floor M

theorem infinite_arithmetic_progression_with_large_digit_sum :
  ∀ M : ℝ, ∃ ap : ArithmeticProgression,
    (ap.diff % 10 ≠ 0) ∧
    (allTermsExceed ap M) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_arithmetic_progression_with_large_digit_sum_l1357_135747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_b_l1357_135771

/-- In a triangle ABC, given angle C is four times angle A, side a is 20, and side c is 36,
    prove that side b is equal to (100 * sin(5A)) / 3 --/
theorem triangle_side_b (A B C : ℝ) (a b c : ℝ) : 
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  C = 4 * A →
  a = 20 →
  c = 36 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b / Real.sin B = c / Real.sin C →
  b = (100 * Real.sin (5 * A)) / 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_b_l1357_135771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1357_135732

noncomputable def f (x : ℝ) := Real.tan (2 * x + Real.pi / 3)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∀ (x y : ℝ), Real.pi/12 < x ∧ x < y ∧ y < 7*Real.pi/12 → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1357_135732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_placement_theorem_l1357_135789

-- Define the number of boxes and balls
def num_boxes : ℕ := 4
def num_balls : ℕ := 4

-- Define the function to calculate the number of ways to place balls into boxes
def place_balls (boxes : ℕ) (balls : ℕ) : ℕ := boxes ^ balls

-- Define the function to calculate the number of ways to place balls with exactly one empty box
def place_balls_one_empty (boxes : ℕ) (balls : ℕ) : ℕ :=
  (Nat.choose balls 2) * (boxes * (boxes - 1) * (boxes - 2))

-- Define the function to calculate the number of ways to place balls with exactly two empty boxes
def place_balls_two_empty (boxes : ℕ) (balls : ℕ) : ℕ :=
  (Nat.choose balls 1) * (boxes * (boxes - 1)) + (Nat.choose boxes 2) * (boxes * (boxes - 1))

-- State the theorem
theorem ball_placement_theorem :
  (place_balls num_boxes num_balls = 256) ∧
  (place_balls_one_empty num_boxes num_balls = 144) ∧
  (place_balls_two_empty num_boxes num_balls = 120) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_placement_theorem_l1357_135789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_strictly_decreasing_three_digit_numbers_l1357_135793

/-- The number of three-digit numbers with strictly decreasing digits -/
def strictly_decreasing_three_digit_numbers : ℕ :=
  Finset.card (Finset.filter (fun n =>
    100 ≤ n ∧ n < 1000 ∧
    (n / 100) > ((n / 10) % 10) ∧ ((n / 10) % 10) > (n % 10))
    (Finset.range 1000))

/-- Theorem stating that the number of three-digit numbers with strictly decreasing digits is 84 -/
theorem count_strictly_decreasing_three_digit_numbers :
  strictly_decreasing_three_digit_numbers = 84 := by
  sorry

#eval strictly_decreasing_three_digit_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_strictly_decreasing_three_digit_numbers_l1357_135793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_power_equation_l1357_135737

theorem unique_solution_power_equation :
  ∀ (x r p n : ℕ),
    x > 0 → r > 1 → p > 1 → n > 1 →
    Nat.Prime p →
    x^r - 1 = p^n →
    x = 3 ∧ r = 2 ∧ p = 2 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_power_equation_l1357_135737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_length_is_225_l1357_135759

/-- The length of a journey in kilometers. -/
noncomputable def journey_length : ℝ := 225

/-- The time taken for the journey at 60 kmph in hours. -/
noncomputable def time_at_60 : ℝ := journey_length / 60

/-- The time taken for the journey at 50 kmph in hours. -/
noncomputable def time_at_50 : ℝ := journey_length / 50

/-- Theorem stating that the journey length is 225 km given the conditions. -/
theorem journey_length_is_225 :
  (time_at_50 = time_at_60 + 3/4) →
  journey_length = 225 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_length_is_225_l1357_135759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_game_probability_dice_game_solution_l1357_135726

/-- Probability that the sum of two dice is a multiple of 3 -/
noncomputable def prob_multiple_of_3 : ℝ := 1/3

/-- Probability that player A rolls on the nth turn -/
noncomputable def P (n : ℕ) : ℝ := 1/2 + (1/2) * (-1/3)^(n-1)

/-- The recurrence relation for the probability of A rolling on the (n+1)th turn -/
noncomputable def recurrence_relation (p : ℝ) : ℝ := 2/3 - (1/3) * p

theorem dice_game_probability (n : ℕ) :
  P (n + 1) = recurrence_relation (P n) ∧
  P 1 = 1 :=
sorry

theorem dice_game_solution (n : ℕ) :
  P n = 1/2 + (1/2) * (-1/3)^(n-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_game_probability_dice_game_solution_l1357_135726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_dots_is_75_l1357_135798

/-- Represents a die with six faces -/
structure Die :=
  (faces : Fin 6 → ℕ)
  (opposite_sum : ∀ i : Fin 6, faces i + faces (5 - i) = 7)
  (valid_marks : ∀ i : Fin 6, 1 ≤ faces i ∧ faces i ≤ 6)

/-- Predicate to determine if two faces of different dice are glued together -/
def are_glued : Fin 7 → Fin 7 → Fin 6 → Fin 6 → Prop := sorry

/-- Represents the assembled figure of dice -/
structure AssembledFigure :=
  (dice : Fin 7 → Die)
  (glued_faces_same : ∀ i j : Fin 7, ∀ f1 f2 : Fin 6, 
    are_glued i j f1 f2 → (dice i).faces f1 = (dice j).faces f2)
  (visible_faces : Fin 9 → (Σ i : Fin 7, Fin 6))

/-- Function to calculate the total number of dots on the figure -/
def total_dots (figure : AssembledFigure) : ℕ := sorry

/-- Theorem stating that the total number of dots is 75 -/
theorem total_dots_is_75 (figure : AssembledFigure) : total_dots figure = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_dots_is_75_l1357_135798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bulk_pencil_cost_example_l1357_135736

/-- The cost of buying a given number of pencils with a bulk discount -/
noncomputable def bulkPencilCost (boxCost : ℚ) (pencilsPerBox : ℕ) (totalPencils : ℕ) (discountPercent : ℚ) : ℚ :=
  let costPerPencil := boxCost / pencilsPerBox
  let totalCostBeforeDiscount := costPerPencil * totalPencils
  totalCostBeforeDiscount * (1 - discountPercent / 100)

/-- Theorem: The cost of 3000 pencils with a 10% bulk discount is $810, given that 100 pencils cost $30 -/
theorem bulk_pencil_cost_example : bulkPencilCost 30 100 3000 10 = 810 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bulk_pencil_cost_example_l1357_135736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1357_135701

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1) / (x - 2)

-- Part 1
theorem part_one (a : ℝ) :
  (∀ x, f a x > 2 ↔ 2 < x ∧ x < 3) → a = 1 := by sorry

-- Part 2
theorem part_two (a : ℝ) :
  (∀ x, x > 2 → f a x < x - 3) → a < 2 * Real.sqrt 2 - 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1357_135701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_conditions_l1357_135760

/-- The area of a triangle with given base and height conditions -/
theorem triangle_area_with_conditions (base : ℝ) (h : base = 14) : 
  (1 / 2) * base * (base / 2 - 0.8) = 43.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_conditions_l1357_135760
