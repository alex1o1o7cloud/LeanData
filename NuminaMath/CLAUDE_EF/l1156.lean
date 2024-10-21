import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1156_115661

-- Problem 1
theorem problem_1 (α : ℝ) : 
  Real.cos (α + π/6) - Real.sin α = 3*Real.sqrt 3/5 → Real.sin (α + 5*π/6) = 3/5 := by sorry

-- Problem 2
theorem problem_2 (α β : ℝ) : 
  Real.sin α + Real.sin β = 1/2 ∧ Real.cos α + Real.cos β = Real.sqrt 2/2 → 
  Real.cos (α - β) = -5/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1156_115661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l1156_115658

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (x + 1) + x + 1
  else Real.exp (-x + 1) - x + 1

-- State the theorem
theorem tangent_line_at_one (h : ∀ x, f x = f (-x)) :
  ∃ (m b : ℝ), m = -2 ∧ b = 3 ∧
  ∀ x y, y = f x → (x = 1 ∧ y = 1) → y = m * x + b := by
  sorry

-- Additional lemma to show f(1) = 1
lemma f_one : f 1 = 1 := by
  sorry

-- Lemma for the derivative of f at x = 1
lemma f_deriv_one : deriv f 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l1156_115658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_by_multiple_l1156_115614

theorem smallest_number_divisible_by_multiple : ∃! n : ℕ, 
  (∀ d : ℕ, d ∈ ({12, 24, 36, 48, 56} : Set ℕ) → (n - 12) % d = 0) ∧
  (∀ m : ℕ, m < n → ∃ d : ℕ, d ∈ ({12, 24, 36, 48, 56} : Set ℕ) ∧ (m - 12) % d ≠ 0) ∧
  n = 1020 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_by_multiple_l1156_115614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_b_2010_l1156_115624

def M (x : ℕ) : ℕ := x % 10

def sequence_b : ℕ → ℚ
  | 0 => 1
  | n + 1 => 2 * sequence_b n

theorem units_digit_b_2010 : M (Int.toNat (Int.floor (sequence_b 2009))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_b_2010_l1156_115624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l1156_115619

/-- The distance between two parallel planes -/
noncomputable def distance_between_planes (a b c d₁ d₂ : ℝ) : ℝ :=
  |d₁ - d₂| / Real.sqrt (a^2 + b^2 + c^2)

/-- Theorem: The distance between the planes 2x - 4y + 4z = 10 and x - 2y + 2z = 2 is 1 -/
theorem distance_between_specific_planes :
  distance_between_planes 2 (-4) 4 10 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l1156_115619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_B_l1156_115699

/-- The time it takes for A to complete the work alone -/
noncomputable def time_A : ℝ := 12

/-- The time it takes for A and B to complete the work together -/
noncomputable def time_AB : ℝ := 6.461538461538462

/-- The time it takes for B to complete the work alone -/
noncomputable def time_B : ℝ := (time_A * time_AB) / (time_A - time_AB)

theorem work_time_B : 
  ‖time_B - 14‖ < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_B_l1156_115699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_area_in_square_l1156_115631

theorem remaining_area_in_square (square_side triangle_side circle_diameter : Real) :
  square_side = 5 →
  triangle_side = 2 →
  circle_diameter = 1 →
  ∃ (remaining_area : Real),
    remaining_area = square_side^2 - (Real.sqrt 3 / 4 * triangle_side^2 + Real.pi / 4 * (circle_diameter / 2)^2) ∧
    remaining_area = 25 - (Real.sqrt 3 + Real.pi / 4) := by
  sorry

#check remaining_area_in_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_area_in_square_l1156_115631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_ratio_implies_k_value_l1156_115687

-- Define the line l: y = kx
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x

-- Define circle C1: (x-1)^2 + y^2 = 1
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define circle C2: (x-3)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1

-- Function to calculate the length of a chord intercepted by the line on a circle
noncomputable def chordLength (k : ℝ) (centerX : ℝ) : ℝ := 
  2 * Real.sqrt (1 - (centerX * k)^2 / (k^2 + 1))

-- Theorem statement
theorem chord_ratio_implies_k_value (k : ℝ) (hk : k > 0) :
  (chordLength k 1) = 3 * (chordLength k 3) → k = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_ratio_implies_k_value_l1156_115687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l1156_115649

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2^(x+1)) / (2^x - 2)

-- Define the theorem
theorem intersection_dot_product 
  (O P A B : ℝ × ℝ) 
  (hO : O = (0, 0)) 
  (hP : P = (1, 1)) 
  (hl : ∃ (t : ℝ), A = t • P ∧ B = t • P) 
  (hf : ∃ (x y : ℝ), f x = y ∧ (A.1 = x ∧ A.2 = y ∨ B.1 = x ∧ B.2 = y)) :
  (A + B) • P = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l1156_115649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_return_probability_theorem_l1156_115605

/-- Probability of returning to the starting point after 2k steps in a 3D random walk -/
noncomputable def return_probability (k : ℕ) : ℝ :=
  1/4 + 3/4 * (1/9)^k

/-- The actual probability of returning after 2k steps in the 3D random walk -/
noncomputable def probability_of_return_after_2k_steps (k : ℕ) : ℝ :=
  sorry  -- This would be defined based on the problem's conditions

/-- Theorem stating the probability of returning to the starting point after 2k steps -/
theorem return_probability_theorem (k : ℕ) (h : k > 0) :
  return_probability k = probability_of_return_after_2k_steps k :=
by sorry

#check return_probability_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_return_probability_theorem_l1156_115605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_finite_sequences_l1156_115642

/-- The function f(x) = 5x - x^2 -/
def f (x : ℝ) : ℝ := 5 * x - x^2

/-- The sequence x_n defined by x_n = f(x_{n-1}) for all n ≥ 1 -/
def x_seq (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => f (x_seq x₀ n)

/-- The set of values in the sequence starting from x₀ -/
def seq_values (x₀ : ℝ) : Set ℝ :=
  {x | ∃ n : ℕ, x_seq x₀ n = x}

/-- The theorem stating that there are infinitely many x₀ such that seq_values x₀ is finite -/
theorem infinitely_many_finite_sequences :
  ∃ S : Set ℝ, (Set.Infinite S) ∧ (∀ x₀ ∈ S, Set.Finite (seq_values x₀)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_finite_sequences_l1156_115642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_roots_l1156_115617

theorem positive_integer_roots (a b : ℕ) : 
  (∃ k l m n : ℕ, 
    k > 0 ∧ l > 0 ∧ m > 0 ∧ n > 0 ∧
    (k * l = a + b - 3) ∧ 
    (m * n = a + b - 3) ∧ 
    (k + l = a) ∧ 
    (m + n = b) ∧ 
    (k^2 - a*k + a + b - 3 = 0) ∧ 
    (l^2 - a*l + a + b - 3 = 0) ∧
    (m^2 - b*m + a + b - 3 = 0) ∧ 
    (n^2 - b*n + a + b - 3 = 0)) ↔ 
  ((a = 2 ∧ b = 2) ∨ (a = 8 ∧ b = 7) ∨ (a = 7 ∧ b = 8) ∨ (a = 6 ∧ b = 6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_roots_l1156_115617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_when_m_zero_two_zeros_iff_m_in_range_l1156_115629

-- Define the piecewise function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x ≤ m then -x^2 - 2*x else x - 4

-- Theorem 1: When m = 0, f has exactly 3 zeros
theorem zeros_when_m_zero :
  ∃! (s : Finset ℝ), (∀ x ∈ s, f 0 x = 0) ∧ s.card = 3 := by
  sorry

-- Theorem 2: f has exactly two zeros iff m ∈ [-2, 0) ∪ [4, +∞)
theorem two_zeros_iff_m_in_range :
  ∀ m : ℝ, (∃! (s : Finset ℝ), (∀ x ∈ s, f m x = 0) ∧ s.card = 2) ↔
    (m ∈ Set.Icc (-2) 0 ∪ Set.Ici 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_when_m_zero_two_zeros_iff_m_in_range_l1156_115629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l1156_115638

/-- Calculates the speed of a train in km/h given its length, platform length, and time to cross -/
noncomputable def trainSpeed (trainLength platformLength : ℝ) (timeToCross : ℝ) : ℝ :=
  let totalDistance := trainLength + platformLength
  let speedMS := totalDistance / timeToCross
  speedMS * 3.6

/-- The speed of the goods train is approximately 72.006 km/h -/
theorem goods_train_speed :
  let trainLength : ℝ := 350.048
  let platformLength : ℝ := 250
  let timeToCross : ℝ := 30
  abs (trainSpeed trainLength platformLength timeToCross - 72.006) < 0.001 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l1156_115638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_surface_area_8cm_l1156_115613

/-- The total surface area of a hemisphere with radius r -/
noncomputable def hemisphere_surface_area (r : ℝ) : ℝ :=
  2 * Real.pi * r^2 + Real.pi * r^2

/-- Theorem: The total surface area of a hemisphere with radius 8 cm is 192π square cm -/
theorem hemisphere_surface_area_8cm :
  hemisphere_surface_area 8 = 192 * Real.pi := by
  -- Unfold the definition of hemisphere_surface_area
  unfold hemisphere_surface_area
  -- Simplify the expression
  simp [Real.pi]
  -- Perform algebraic manipulations
  ring

#check hemisphere_surface_area_8cm

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_surface_area_8cm_l1156_115613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_susie_investment_interest_l1156_115662

/-- Compound interest calculation --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounds_per_year : ℕ) (years : ℕ) : ℝ :=
  principal * (1 + rate / (compounds_per_year : ℝ)) ^ ((compounds_per_year : ℝ) * (years : ℝ))

/-- Total interest earned calculation --/
noncomputable def total_interest (principal : ℝ) (rate : ℝ) (compounds_per_year : ℕ) (years : ℕ) : ℝ :=
  compound_interest principal rate compounds_per_year years - principal

theorem susie_investment_interest :
  let principal : ℝ := 1500
  let rate : ℝ := 0.12
  let compounds_per_year : ℕ := 4
  let years : ℕ := 4
  abs (total_interest principal rate compounds_per_year years - 901.55) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_susie_investment_interest_l1156_115662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_count_l1156_115677

theorem rectangle_count : ∃! n : ℕ, n = (Finset.filter (λ p : ℕ × ℕ => p.1 + p.2 = 40 ∧ p.1 > p.2 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.range 40 ×ˢ Finset.range 40)).card := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_count_l1156_115677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_m_value_l1156_115634

/-- Three points are collinear if they lie on the same straight line. -/
def are_collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, B - A = t • (C - A)

theorem collinear_points_m_value (m : ℝ) :
  let A : ℝ × ℝ := (0, 1)
  let B : ℝ × ℝ := (m, 3)
  let C : ℝ × ℝ := (4, 7)
  are_collinear A B C → m = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_m_value_l1156_115634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_number_tenth_row_l1156_115693

/-- Represents a lattice with a given number of rows and columns. -/
structure MyLattice where
  rows : ℕ
  columns : ℕ

/-- Returns the last number in a given row of the lattice. -/
def lastNumberInRow (l : MyLattice) (row : ℕ) : ℕ :=
  l.columns * row

/-- Returns the nth number from the end in a given row of the lattice. -/
def nthNumberFromEnd (l : MyLattice) (row : ℕ) (n : ℕ) : ℕ :=
  lastNumberInRow l row - (n - 1)

/-- Theorem stating that the fourth number in the 10th row of a specific lattice is 58. -/
theorem fourth_number_tenth_row :
  let l : MyLattice := ⟨10, 6⟩
  nthNumberFromEnd l 10 4 = 58 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_number_tenth_row_l1156_115693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1156_115643

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x + x^2 else 2 * x - x^2

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ = 2 * a^2 + a ∧ f x₂ = 2 * a^2 + a ∧ f x₃ = 2 * a^2 + a) →
  -1 < a ∧ a < 1/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1156_115643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_construction_l1156_115692

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  A : Point
  B : Point
  C : Point

/-- Checks if three points form an equilateral triangle -/
def isEquilateral (t : EquilateralTriangle) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Theorem stating the number of possible equilateral triangles -/
theorem equilateral_triangle_construction
  (P : Point) (rA rB rC : ℝ) 
  (h1 : rA ≤ rB) (h2 : rB ≤ rC) :
  let solutions := { t : EquilateralTriangle | 
    isEquilateral t ∧ 
    distance P t.A = rA ∧
    distance P t.B = rB ∧
    distance P t.C = rC }
  (∃ n : ℕ, n ∈ ({0, 1, 2} : Set ℕ) ∧ ∃ (l : List EquilateralTriangle), l.length = n ∧ ∀ t ∈ l, t ∈ solutions) := by
  sorry

#check equilateral_triangle_construction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_construction_l1156_115692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_difference_is_two_l1156_115660

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + (n - 1) * d

theorem common_difference_is_two 
  (a₁ : ℝ) (d : ℝ) 
  (h1 : arithmetic_sequence a₁ d 4 + arithmetic_sequence a₁ d 6 = 10)
  (h2 : (Finset.range 5).sum (arithmetic_sequence a₁ d) = 5) :
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_difference_is_two_l1156_115660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1156_115664

/-- Calculates the number of days two workers worked together given their individual work rates and the remaining work fraction -/
noncomputable def days_worked_together (a_days : ℝ) (b_days : ℝ) (remaining_fraction : ℝ) : ℝ :=
  (1 - remaining_fraction) / ((1 / a_days) + (1 / b_days))

theorem work_completion_time (a_days b_days : ℝ) (h1 : a_days = 15) (h2 : b_days = 20) :
  days_worked_together a_days b_days 0.65 = 3 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1156_115664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_of_f_l1156_115683

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

-- State the theorem
theorem sum_of_zeros_of_f : 
  ∃ (x₁ x₂ : ℝ), 
    0 ≤ x₁ ∧ x₁ ≤ π ∧
    0 ≤ x₂ ∧ x₂ ≤ π ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧
    x₁ + x₂ = 7 * π / 6 := by
  -- The proof is omitted and replaced with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_of_f_l1156_115683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_pq_length_l1156_115641

/-- A triangle with special median properties -/
structure SpecialTriangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  median_P_perpendicular_median_Q : Bool
  QR_length : ℝ
  PR_length : ℝ
  PR_greater_PQ : Bool

/-- The length of PQ in a SpecialTriangle -/
noncomputable def pq_length (t : SpecialTriangle) : ℝ :=
  let (px, py) := t.P
  let (qx, qy) := t.Q
  Real.sqrt ((qx - px)^2 + (qy - py)^2)

/-- Theorem: In a SpecialTriangle, if QR = 8 and PR = 5, then PQ = √29 -/
theorem special_triangle_pq_length 
  (t : SpecialTriangle) 
  (h1 : t.median_P_perpendicular_median_Q = true)
  (h2 : t.QR_length = 8)
  (h3 : t.PR_length = 5)
  (h4 : t.PR_greater_PQ = true) :
  pq_length t = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_pq_length_l1156_115641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_1_to_36_base7_l1156_115654

/-- Represents a number in base 7 --/
structure Base7 where
  value : Nat

/-- Converts a natural number to base 7 --/
def toBase7 (n : Nat) : Base7 :=
  ⟨n⟩  -- We're not actually converting here, just wrapping the number

/-- Converts a base 7 number to a natural number --/
def fromBase7 (b : Base7) : Nat :=
  b.value  -- We're not actually converting here, just unwrapping the number

/-- Computes the sum of an arithmetic sequence --/
def arithmeticSum (first last n : Nat) : Nat :=
  n * (first + last) / 2

/-- Provide an instance of OfNat for Base7 --/
instance : OfNat Base7 n where
  ofNat := ⟨n⟩

/-- The main theorem --/
theorem sum_1_to_36_base7 :
  toBase7 (arithmeticSum 1 (fromBase7 36) 36) = toBase7 666 := by
  sorry

#eval arithmeticSum 1 36 36  -- This should evaluate to 666

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_1_to_36_base7_l1156_115654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_equal_lateral_area_implies_60_degree_angle_l1156_115671

/-- Given a cone and a cylinder with the same height and volume, 
    if their lateral surface areas are equal, 
    then the cone's opening angle is 60°. -/
theorem cone_cylinder_equal_lateral_area_implies_60_degree_angle 
  (r₁ r₂ m a : ℝ) (h_positive : r₁ > 0 ∧ r₂ > 0 ∧ m > 0 ∧ a > 0) :
  (1/3 * Real.pi * r₁^2 * m = Real.pi * r₂^2 * m) →  -- Same volume
  (Real.pi * r₁ * a = 2 * Real.pi * r₂ * m) →        -- Same lateral surface area
  Real.arccos (m / a) * (180 / Real.pi) = 60 :=  -- Opening angle in degrees
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_equal_lateral_area_implies_60_degree_angle_l1156_115671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proof_original_shepherds_count_l1156_115688

/-- The number of legs a sheep has -/
def sheep_legs : ℕ := 4

/-- The number of legs a shepherd has -/
def shepherd_legs : ℕ := 2

/-- The original number of sheep -/
def original_sheep : ℕ := 45

/-- The total number of legs of remaining shepherds and sheep -/
def total_remaining_legs : ℕ := 126

/-- Calculate the original number of shepherds -/
def original_shepherds_count : ℕ :=
  let remaining_sheep := (2 * original_sheep) / 3
  let remaining_sheep_legs := remaining_sheep * sheep_legs
  let remaining_shepherd_legs := total_remaining_legs - remaining_sheep_legs
  let remaining_shepherds := remaining_shepherd_legs / shepherd_legs
  2 * remaining_shepherds

/-- Proof that the original number of shepherds is 6 -/
theorem proof_original_shepherds_count : original_shepherds_count = 6 := by
  -- Unfold the definition of original_shepherds_count
  unfold original_shepherds_count
  -- Perform the calculation
  norm_num
  -- QED
  rfl

#eval original_shepherds_count -- This should output 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proof_original_shepherds_count_l1156_115688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1156_115616

-- Define the ellipse C
noncomputable def ellipse_C (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

-- Theorem statement
theorem ellipse_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (F₁ F₂ P : ℝ × ℝ) :
  ellipse_C a b P.1 P.2 →
  (P.1 - F₂.1) * (F₂.1 - F₁.1) + (P.2 - F₂.2) * (F₂.2 - F₁.2) = 0 →
  Real.cos (Real.pi / 6) * ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 
    (F₂.1 - F₁.1) * (P.1 - F₁.1) + (F₂.2 - F₁.2) * (P.2 - F₁.2) →
  eccentricity a ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2).sqrt / 2 = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1156_115616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_θ_tan_θ_eq_sqrt_3_l1156_115690

/-- The function f that reaches its maximum at θ -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x + Real.pi / 6)

/-- θ is the point where f reaches its maximum -/
noncomputable def θ : ℝ := Real.pi / 3

/-- Theorem stating that f reaches its maximum at θ -/
theorem f_max_at_θ (x : ℝ) : f x ≤ f θ := by sorry

/-- Theorem stating that tan(θ) equals √3 -/
theorem tan_θ_eq_sqrt_3 : Real.tan θ = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_θ_tan_θ_eq_sqrt_3_l1156_115690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_when_max_area_l1156_115622

-- Define the circle equation
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 + k*x + 2*y + k^2 = 0

-- Define the line equation
def line_equation (x y k : ℝ) : Prop :=
  y = (k - 1) * x + 2

-- Define the condition for maximum area
def max_area (k : ℝ) : Prop :=
  ∀ k' : ℝ, (1 - 3 * k^2 / 4) ≥ (1 - 3 * k'^2 / 4)

-- Define the slope angle
noncomputable def slope_angle (k : ℝ) : ℝ :=
  Real.arctan (k - 1)

-- Theorem statement
theorem slope_angle_when_max_area :
  ∀ k : ℝ, max_area k → slope_angle k = 3 * Real.pi / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_when_max_area_l1156_115622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_and_inverse_proof_l1156_115666

def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 3; 2, 4]

def α₁ : Fin 2 → ℝ := ![1, 1]
def α₂ : Fin 2 → ℝ := ![3, -2]

theorem matrix_and_inverse_proof :
  (A.mulVec α₁ = (6 : ℝ) • α₁) ∧
  (A.mulVec α₂ = (1 : ℝ) • α₂) →
  (A = !![3, 3; 2, 4]) ∧
  (A⁻¹ = !![2/3, -1/2; -1/3, 1/2]) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_and_inverse_proof_l1156_115666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l1156_115659

/-- Given a car's speeds in two consecutive hours, calculate its average speed. -/
noncomputable def average_speed (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  (speed1 + speed2) / 2

/-- Theorem: The average speed of a car traveling 90 km/h for the first hour
    and 50 km/h for the second hour is 70 km/h. -/
theorem car_average_speed :
  average_speed 90 50 = 70 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l1156_115659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quadrilateral_area_max_area_achieved_l1156_115678

/-- Curve C₁ defined by parametric equations -/
def C₁ (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ θ : ℝ, p.1 = a * Real.cos θ ∧ p.2 = b * Real.sin θ}

/-- Curve C₂ defined by its Cartesian equation -/
def C₂ (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

/-- The area of the quadrilateral formed by the intersection points of C₁ and C₂ -/
noncomputable def quadrilateralArea (a b r : ℝ) : ℝ :=
  4 * a * b * Real.sin (2 * Real.arccos (r / a))

theorem max_quadrilateral_area (a b r : ℝ) 
  (ha : a > 0) (hb : b > 0) (hr : r > 0) (hab : a > b) (hbr : b < r) (hra : r < a) :
  ∀ θ : ℝ, quadrilateralArea a b r ≤ 2 * a * b := by
  sorry

theorem max_area_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  ∃ r : ℝ, b < r ∧ r < a ∧ quadrilateralArea a b r = 2 * a * b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quadrilateral_area_max_area_achieved_l1156_115678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1156_115689

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, (2 : ℝ)^(x^2 - 2) > 1)) ↔ (∃ x₀ : ℝ, (2 : ℝ)^(x₀^2 - 2) ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1156_115689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l1156_115681

-- Problem 1
theorem problem_1 : (9 : ℤ) - (-3) + 1 = 13 := by sorry

-- Problem 2
theorem problem_2 : (-3 : ℚ) * (5/6) + (-10/3) = -35/6 := by sorry

-- Problem 3
theorem problem_3 : (36 : ℤ) * ((1/2 : ℚ) - 5/9) - |(-1)| = -3 := by sorry

-- Problem 4
theorem problem_4 (h : abs (Real.sqrt 3 - 1.732) < 0.01) : 
  abs (Real.sqrt ((-3)^2) + 2 * (Real.sqrt 3 + 4) - 14.46) < 0.01 := by sorry

-- Definition of approximation
def approx (x y : ℝ) : Prop := abs (x - y) < 0.01

notation:50 x " ≈ " y => approx x y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l1156_115681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_draining_time_l1156_115640

/-- Represents the efficiency of a pump -/
structure PumpEfficiency where
  value : ℝ
  positive : value > 0

/-- Represents the time taken to complete a task -/
def Time := ℝ

/-- Represents the amount of work done -/
def Work := ℝ

theorem water_tank_draining_time 
  (efficiency_ratio : ℝ) 
  (total_time : ℝ) 
  (pump_b_time : ℝ) 
  (h_efficiency_ratio : efficiency_ratio = 3 / 4)
  (h_total_time : total_time = 15)
  (h_pump_b_time : pump_b_time = 9)
  : ∃ (pump_a_efficiency pump_b_efficiency : PumpEfficiency),
    pump_a_efficiency.value / pump_b_efficiency.value = efficiency_ratio ∧
    (pump_a_efficiency.value + pump_b_efficiency.value) * total_time = 
      pump_b_efficiency.value * pump_b_time + 
      pump_a_efficiency.value * 23 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_draining_time_l1156_115640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_cosine_inequality_triangle_sine_cosine_equality_l1156_115675

/-- For a non-obtuse triangle ABC, the sum of squares of sines of its angles
    is greater than or equal to the square of the sum of cosines of its angles -/
theorem triangle_sine_cosine_inequality (A B C : ℝ) 
  (h_non_obtuse : A + B + C = Real.pi ∧ 0 ≤ A ∧ A ≤ Real.pi/2 ∧ 0 ≤ B ∧ B ≤ Real.pi/2 ∧ 0 ≤ C ∧ C ≤ Real.pi/2) :
  Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 ≥ (Real.cos A + Real.cos B + Real.cos C) ^ 2 := by
  sorry

/-- The equality in the above inequality holds if and only if 
    the triangle is isosceles right-angled or equilateral -/
theorem triangle_sine_cosine_equality (A B C : ℝ) 
  (h_non_obtuse : A + B + C = Real.pi ∧ 0 ≤ A ∧ A ≤ Real.pi/2 ∧ 0 ≤ B ∧ B ≤ Real.pi/2 ∧ 0 ≤ C ∧ C ≤ Real.pi/2) :
  Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = (Real.cos A + Real.cos B + Real.cos C) ^ 2 ↔ 
  ((A = Real.pi/4 ∧ B = Real.pi/4 ∧ C = Real.pi/2) ∨ (A = Real.pi/3 ∧ B = Real.pi/3 ∧ C = Real.pi/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_cosine_inequality_triangle_sine_cosine_equality_l1156_115675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l1156_115611

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi / 4 - 2 * x)

theorem f_strictly_increasing (k : ℤ) :
  StrictMonoOn f (Set.Ioo (-3 * Real.pi / 8 + k * Real.pi) (Real.pi / 8 + k * Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l1156_115611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_with_specific_remainder_l1156_115648

theorem multiple_with_specific_remainder (y : ℕ) (h1 : y % 9 = 5) :
  ∃ k : ℕ, k > 0 ∧ (k * y) % 9 = 6 ∧ ∀ m : ℕ, m > 0 → (m * y) % 9 = 6 → k ≤ m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_with_specific_remainder_l1156_115648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_towel_savings_l1156_115669

theorem paper_towel_savings (package_cost : ℚ) (individual_cost : ℚ) (rolls : ℕ) :
  package_cost = 9 →
  individual_cost = 1 →
  rolls = 12 →
  (1 - package_cost / (individual_cost * rolls)) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_towel_savings_l1156_115669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_divides_rectangle_equally_l1156_115653

/-- A rectangle in the coordinate plane -/
structure Rectangle where
  width : ℝ
  height : ℝ
  lower_left : ℝ × ℝ

/-- A line in the coordinate plane defined by two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

/-- Theorem: A line divides a rectangle into two equal areas iff d = 1 -/
theorem line_divides_rectangle_equally (r : Rectangle) (l : Line) (d : ℝ) :
  r.width = 3 ∧ r.height = 2 ∧ r.lower_left = (0, 0) ∧
  l.point1 = (d, 0) ∧ l.point2 = (4, 2) →
  (triangleArea (4 - d) 2 = (r.width * r.height) / 2) ↔ d = 1 := by
  sorry

#check line_divides_rectangle_equally

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_divides_rectangle_equally_l1156_115653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_three_digit_integers_l1156_115646

/-- A function that returns the number of valid pairs (a, b) for a given ones digit c -/
def validPairs (c : Nat) : Nat :=
  if c = 1 then 28
  else if c = 3 then 15
  else if c = 5 then 6
  else if c = 7 then 1
  else 0

/-- The set of valid ones digits for odd three-digit integers -/
def validOnesDigits : Finset Nat := {1, 3, 5, 7, 9}

theorem odd_decreasing_three_digit_integers :
  (Finset.sum validOnesDigits validPairs) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_three_digit_integers_l1156_115646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_l1156_115620

theorem sin_squared_sum (α : ℝ) : 
  (Real.sin (α - 60 * π / 180))^2 + (Real.sin α)^2 + (Real.sin (α + 60 * π / 180))^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_l1156_115620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1156_115607

def set_A : Set ℝ := {x | 2 - x < 1}
def set_B : Set ℝ := {x | x^2 + 2*x - 15 < 0}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x : ℝ | 1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1156_115607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1156_115601

theorem inequality_proof (a b : ℝ) (h1 : (2 : ℝ)^a > (2 : ℝ)^b) (h2 : (2 : ℝ)^b > 1) :
  ((1/2) : ℝ)^a < ((1/2) : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1156_115601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_tileable_rectangle_dimensions_l1156_115695

/-- Represents a rectangle that can be tiled with (a, b)-tiles. -/
structure TileableRectangle (a b : ℝ) where
  width : ℝ
  height : ℝ
  is_tileable : width ≥ 0 ∧ height ≥ 0

/-- The smallest area of a tileable rectangle for given tile dimensions. -/
noncomputable def smallest_tileable_area (a b : ℝ) : ℝ :=
  (max 1 (2 * a)) * (2 * b)

/-- Theorem stating that the smallest tileable rectangle has dimensions max{1, 2a} × 2b. -/
theorem smallest_tileable_rectangle_dimensions (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) :
  ∃ (r : TileableRectangle a b), 
    r.width = max 1 (2 * a) ∧ 
    r.height = 2 * b ∧
    ∀ (s : TileableRectangle a b), r.width * r.height ≤ s.width * s.height :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_tileable_rectangle_dimensions_l1156_115695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_l1156_115639

theorem rectangle_breadth (square_area : ℝ) (rectangle_area : ℝ) : 
  square_area = 3025 →
  rectangle_area = 220 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 5) * circle_radius
  let rectangle_breadth := rectangle_area / rectangle_length
  rectangle_breadth = 10 := by
    intro h1 h2
    -- The proof steps would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_l1156_115639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l1156_115682

/-- The line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := 3 * x - y - 6 = 0

/-- The point P where line l intersects the x-axis -/
def point_P : ℝ × ℝ := (2, 0)

/-- The angle of rotation in radians -/
noncomputable def rotation_angle : ℝ := Real.pi / 4

/-- The line l₁ obtained after rotating line l -/
def line_l1 (x y : ℝ) : Prop := 2 * x + y - 4 = 0

/-- The other given line -/
def other_line (x y : ℝ) : Prop := 4 * x + 2 * y + 1 = 0

/-- The theorem to be proved -/
theorem distance_between_lines :
  let d := (9 * Real.sqrt 5) / 10
  ∀ x y : ℝ, line_l1 x y → 
    (∃ x' y', other_line x' y' ∧ 
      ((x - x')^2 + (y - y')^2)^(1/2 : ℝ) = d) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l1156_115682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_ratio_l1156_115651

noncomputable def f₁ (a x : ℝ) : ℝ := x^2 - 2*a*x + 3
noncomputable def f₂ (b x : ℝ) : ℝ := x^2 + x + b
noncomputable def f₃ (a b x : ℝ) : ℝ := 3*x^2 + (1-4*a)*x + 6 + b
noncomputable def f₄ (a b x : ℝ) : ℝ := 3*x^2 + (2-2*a)*x + 3 + 2*b

noncomputable def A (a : ℝ) : ℝ := Real.sqrt (4*a^2 - 12)
noncomputable def B (b : ℝ) : ℝ := Real.sqrt (1 - 4*b)
noncomputable def C (a b : ℝ) : ℝ := (1/3) * Real.sqrt ((1 - 4*a)^2 - 12*(6 + b))
noncomputable def D (a b : ℝ) : ℝ := (1/3) * Real.sqrt ((2 - 2*a)^2 - 12*(3 + 2*b))

theorem root_difference_ratio (a b : ℝ) (h : A a ≠ B b) :
  (C a b)^2 - (D a b)^2 = (1/3) * ((A a)^2 - (B b)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_ratio_l1156_115651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_three_l1156_115667

/-- The profit function for a spherical bottle -/
noncomputable def profit_function (r : ℝ) : ℝ := 0.1 * Real.pi * (4 * r^3 - r^4)

/-- The theorem stating that the profit is maximized when r = 3 -/
theorem profit_maximized_at_three :
  ∃ (r : ℝ), 0 < r ∧ r ≤ 8 ∧
  ∀ (s : ℝ), 0 < s ∧ s ≤ 8 → profit_function r ≥ profit_function s ∧ r = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_three_l1156_115667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drainage_theorem_l1156_115603

/-- Represents the drainage system of a pool -/
structure DrainageSystem where
  x : ℝ  -- rate of pipes A and B in gallons per day
  y : ℝ  -- rate of pipes C and D in gallons per day
  z : ℝ  -- rate of pipe E in gallons per day
  pool_volume : ℝ  -- total volume of the pool in gallons

/-- Calculates the number of additional pipes needed to drain the pool in 4 days -/
noncomputable def additional_pipes_needed (s : DrainageSystem) : ℝ :=
  2 + 5 * s.y / s.x + s.z / s.x

theorem drainage_theorem (s : DrainageSystem) 
  (h1 : s.pool_volume = 12 * (2 * s.x + 2 * s.y + s.z))  -- all pipes drain pool in 12 days
  (h2 : s.pool_volume = 12 * s.x + 28 * s.y + 8 * s.z)   -- actual drainage considering start times
  (h3 : s.x > 0) : -- ensure x is not zero for division
  4 * ((1 + additional_pipes_needed s) * s.x + 2 * s.y + s.z) = s.pool_volume :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drainage_theorem_l1156_115603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_is_4_sqrt_2_l1156_115604

/-- The distance between the foci of a hyperbola defined by xy = 4 with foci along its diagonals -/
noncomputable def hyperbola_foci_distance : ℝ := 4 * Real.sqrt 2

/-- Theorem: The distance between the foci of a hyperbola defined by xy = 4 with foci along its diagonals is 4√2 -/
theorem hyperbola_foci_distance_is_4_sqrt_2 :
  let h := {p : ℝ × ℝ | p.1 * p.2 = 4}
  let foci := {p : ℝ × ℝ | ∃ c : ℝ, p = (c, c) ∨ p = (-c, -c)}
  ∃ f₁ f₂ : ℝ × ℝ, f₁ ∈ foci ∧ f₂ ∈ foci ∧ f₁ ≠ f₂ ∧
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = hyperbola_foci_distance :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_is_4_sqrt_2_l1156_115604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l1156_115650

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x ^ (1/3)) * Real.sin x

-- State the theorem
theorem derivative_f_at_one :
  deriv f 1 = (1/3) * Real.sin 1 + Real.cos 1 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l1156_115650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_average_speed_approx_l1156_115680

-- Define the segments of the truck's journey
noncomputable def segment1_speed : ℝ := 30
noncomputable def segment1_distance : ℝ := 15
noncomputable def segment2_speed : ℝ := 55
noncomputable def segment2_distance : ℝ := 35
noncomputable def segment3_speed : ℝ := 45
noncomputable def segment3_time : ℝ := 0.5  -- 30 minutes in hours
noncomputable def segment4_speed : ℝ := 50
noncomputable def segment4_time : ℝ := 1/3  -- 20 minutes in hours

-- Calculate total distance and total time
noncomputable def total_distance : ℝ :=
  segment1_distance +
  segment2_distance +
  segment3_speed * segment3_time +
  segment4_speed * segment4_time

noncomputable def total_time : ℝ :=
  segment1_distance / segment1_speed +
  segment2_distance / segment2_speed +
  segment3_time +
  segment4_time

-- Define the average speed
noncomputable def average_speed : ℝ := total_distance / total_time

-- Theorem to prove
theorem truck_average_speed_approx :
  |average_speed - 45.24| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_average_speed_approx_l1156_115680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jills_plot_width_l1156_115668

/-- Calculates the width of a plot given the total sod area needed and the dimensions of non-sodded areas -/
noncomputable def calculate_plot_width (total_sod_area : ℝ) (plot_length : ℝ) (sidewalk_width : ℝ) (sidewalk_length : ℝ)
  (flowerbed1_depth : ℝ) (flowerbed1_length : ℝ) (flowerbed1_count : ℕ)
  (flowerbed2_width : ℝ) (flowerbed2_length : ℝ)
  (flowerbed3_width : ℝ) (flowerbed3_length : ℝ) : ℝ :=
  let sidewalk_area := sidewalk_width * sidewalk_length
  let flowerbed1_area := flowerbed1_depth * flowerbed1_length * (flowerbed1_count : ℝ)
  let flowerbed2_area := flowerbed2_width * flowerbed2_length
  let flowerbed3_area := flowerbed3_width * flowerbed3_length
  let total_non_sodded_area := sidewalk_area + flowerbed1_area + flowerbed2_area + flowerbed3_area
  let sodded_area := total_sod_area - total_non_sodded_area
  sodded_area / plot_length

/-- The width of Jill's plot is approximately 178.96 feet -/
theorem jills_plot_width :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (calculate_plot_width 9474 50 3 50 4 25 2 10 12 7 8 - 178.96) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jills_plot_width_l1156_115668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_number_31st_row_l1156_115652

/-- Pascal's triangle function -/
def pascal (n k : ℕ) : ℕ :=
  match n, k with
  | _, 0 => 1
  | 0, _ => 0
  | n+1, k+1 => pascal n k + pascal n (k+1)

/-- The second number in the nth row of Pascal's triangle -/
def secondNumber (n : ℕ) : ℕ := pascal n 1

theorem second_number_31st_row : secondNumber 30 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_number_31st_row_l1156_115652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangles_from_square_l1156_115618

/-- The side length of the square paper in centimeters -/
def square_side : ℚ := 10

/-- The width of the right triangle in centimeters -/
def triangle_width : ℚ := 1

/-- The height of the right triangle in centimeters -/
def triangle_height : ℚ := 3

/-- The area of the square paper in square centimeters -/
def square_area : ℚ := square_side * square_side

/-- The area of one right triangle in square centimeters -/
def triangle_area : ℚ := (triangle_width * triangle_height) / 2

/-- The maximum number of whole right triangles that can be cut from the square -/
def max_triangles : ℕ := (square_area / triangle_area).floor.toNat

theorem max_triangles_from_square :
  max_triangles = 66 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangles_from_square_l1156_115618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l1156_115645

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being invertible
def IsInvertible (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Theorem statement
theorem intersection_points (h : IsInvertible f) :
  ∃! s : Finset ℝ, (∀ x ∈ s, f (x^3) = f (x^6)) ∧ s.card = 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l1156_115645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_of_article_l1156_115625

/-- The cost price of an article, given that the profit when selling for 56 
    is equal to the loss when selling for 42. -/
theorem cost_price_of_article : ℝ := by
  -- Let C be the cost price of the article
  let C : ℝ := 49

  -- Define profit and loss
  let profit : ℝ := 56 - C
  let loss : ℝ := C - 42

  -- The profit is equal to the loss
  have h : profit = loss := by
    -- Proof skipped
    sorry

  -- Prove that C = 49
  have : C = 49 := by
    -- Proof skipped
    sorry

  -- Return the cost price
  exact C


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_of_article_l1156_115625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1156_115600

-- Define the point M
def M (m : ℝ) : ℝ × ℝ := (m + 3, 2 * m - 1)

-- Define the point N (M moved up by 4 units)
def N (m : ℝ) : ℝ × ℝ := ((M m).1, (M m).2 + 4)

-- Part 1
theorem part_one :
  ∀ m : ℝ, (N m).2 = (N m).1 + 3 → M m = (6, 5) := by sorry

-- Part 2
theorem part_two :
  ∀ m : ℝ, abs ((M m).2) = 2 ∧ (M m).1 > 0 ∧ (M m).2 < 0 → N m = (5/2, 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1156_115600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jana_walking_distance_l1156_115697

/-- Jana's walking rate in miles per minute -/
def jana_rate : ℚ := 1 / 18

/-- The time Jana walks in minutes -/
def walking_time : ℚ := 45

/-- The distance Jana walks in the given time -/
def distance : ℚ := jana_rate * walking_time

/-- Rounds a rational number to the nearest hundredth -/
def round_to_hundredth (q : ℚ) : ℚ := 
  (q * 100).floor / 100

theorem jana_walking_distance : round_to_hundredth distance = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jana_walking_distance_l1156_115697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_on_unit_interval_f_upper_bound_on_interval_l1156_115632

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x - a * Real.log (x + 1)

-- Part 1
theorem f_nonnegative_on_unit_interval :
  ∀ x : ℝ, x ∈ Set.Icc 0 1 → f 1 x ≥ 0 := by sorry

-- Part 2
theorem f_upper_bound_on_interval (a : ℝ) :
  a ≥ -1 →
  ∀ x : ℝ, x ∈ Set.Icc 0 Real.pi →
  f a x ≤ 2 * Real.exp x - 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_on_unit_interval_f_upper_bound_on_interval_l1156_115632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_sqrt_greater_than_ln_exists_x0_for_M_l1156_115685

noncomputable def f (x : ℝ) := Real.sqrt x + (-1) * (Real.log x + 1) + 1

theorem tangent_line_at_one :
  ∃ (a b : ℝ), f 1 = 1 ∧ deriv f 1 = -1/2 ∧ 
  (fun x ↦ (deriv f 1) * (x - 1) + f 1) = (fun x ↦ (-1/2) * x + 3/2) := by sorry

theorem sqrt_greater_than_ln (x : ℝ) (h : x > 0) : Real.sqrt x > Real.log x := by sorry

theorem exists_x0_for_M (M : ℝ) (h : M > 0) :
  ∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > x₀, Real.sqrt x > M * Real.log x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_sqrt_greater_than_ln_exists_x0_for_M_l1156_115685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hare_race_result_l1156_115612

/-- Represents the different types of jumps the hare can make -/
inductive JumpType
  | Left
  | Right
  | Both

/-- Represents the hare's race -/
structure HareRace where
  race_length : ℕ
  left_jump : ℕ
  right_jump : ℕ
  both_jump : ℕ

/-- Returns the length of a jump based on the jump type -/
def jump_length (race : HareRace) (jump : JumpType) : ℕ :=
  match jump with
  | JumpType.Left => race.left_jump
  | JumpType.Right => race.right_jump
  | JumpType.Both => race.both_jump

/-- Calculates the total distance covered by a sequence of jumps -/
def total_distance (race : HareRace) (jumps : List JumpType) : ℕ :=
  jumps.foldl (fun acc jump => acc + jump_length race jump) 0

/-- Theorem: The hare will make 548 jumps and the final jump will be with the right leg -/
theorem hare_race_result (race : HareRace) 
  (h1 : race.race_length = 2024 * 10)  -- Convert meters to decimeters
  (h2 : race.left_jump = 35)
  (h3 : race.right_jump = 15)
  (h4 : race.both_jump = 61) :
  ∃ (jumps : List JumpType), 
    total_distance race jumps ≥ race.race_length ∧ 
    jumps.length = 548 ∧ 
    jumps.getLast? = some JumpType.Right := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hare_race_result_l1156_115612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_solutions_l1156_115630

/-- The system of equations -/
def system (x y z w : ℝ) : Prop :=
  x = z + w + x*z ∧
  y = w + x + y*w ∧
  z = x + y + z*x ∧
  w = y + z + w*z

/-- The theorem stating that there are exactly 5 solutions to the system -/
theorem five_solutions :
  ∃! (s : Set (ℝ × ℝ × ℝ × ℝ)), 
    (∀ (x y z w : ℝ), (x, y, z, w) ∈ s ↔ system x y z w) ∧ 
    Finite s ∧ 
    Nat.card s = 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_solutions_l1156_115630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l1156_115623

noncomputable def A : ℝ × ℝ := (-2, 1)
noncomputable def B : ℝ × ℝ := (4, 9)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def is_on_segment (C A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

theorem point_C_coordinates :
  ∃ C : ℝ × ℝ, is_on_segment C A B ∧ 
              distance A C = 2 * distance C B ∧ 
              C = (2, 19/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l1156_115623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_and_minimum_value_l1156_115636

-- Define the vectors
noncomputable def a (x : ℝ) : Fin 2 → ℝ
  | 0 => Real.sin x
  | 1 => Real.cos x

noncomputable def b (x k : ℝ) : Fin 2 → ℝ
  | 0 => Real.sin x
  | 1 => k

noncomputable def c (x k : ℝ) : Fin 2 → ℝ
  | 0 => -2 * Real.cos x
  | 1 => Real.sin x - k

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- Define the function g
noncomputable def g (x k : ℝ) : ℝ :=
  dot_product (fun i => a x i + b x k i) (c x k)

-- State the theorem
theorem vector_magnitude_and_minimum_value :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 4 →
    1 ≤ Real.sqrt ((b x 0 0 + c x 0 0)^2 + (b x 0 1 + c x 0 1)^2) ∧
    Real.sqrt ((b x 0 0 + c x 0 0)^2 + (b x 0 1 + c x 0 1)^2) ≤ 2) ∧
  (∃ k : ℝ, ∀ x : ℝ, g x k ≥ -3/2 ∧ ∃ x₀ : ℝ, g x₀ k = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_and_minimum_value_l1156_115636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_of_specific_cone_l1156_115670

/-- The central angle of a sector obtained by unfolding the lateral surface of a cone -/
noncomputable def central_angle_of_unfolded_cone (base_radius : ℝ) (slant_height : ℝ) : ℝ :=
  (2 * Real.pi * base_radius) / slant_height

/-- Theorem: The central angle of the sector obtained by unfolding the lateral surface
    of a cone with base radius 1 and slant height 2 is π radians -/
theorem central_angle_of_specific_cone :
  central_angle_of_unfolded_cone 1 2 = Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_of_specific_cone_l1156_115670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_difference_magnitude_l1156_115656

theorem complex_difference_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2)
  (h2 : Complex.abs z₂ = 2)
  (h3 : z₁ + z₂ = Complex.ofReal (Real.sqrt 3) + Complex.I) :
  Complex.abs (z₁ - z₂) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_difference_magnitude_l1156_115656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_deg_to_rad_l1156_115672

-- Define the conversion factor from degrees to radians
noncomputable def deg_to_rad (deg : ℝ) : ℝ := (deg * Real.pi) / 180

-- Theorem statement
theorem sixty_deg_to_rad : deg_to_rad 60 = Real.pi / 3 := by
  -- Unfold the definition of deg_to_rad
  unfold deg_to_rad
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_deg_to_rad_l1156_115672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_16_hours_l1156_115657

/-- Represents the journey details -/
structure Journey where
  totalDistance : ℝ
  carSpeed : ℝ
  walkSpeed : ℝ
  switchPoint : ℝ

/-- Calculates the total time for the journey -/
noncomputable def journeyTime (j : Journey) : ℝ :=
  let carTime := j.switchPoint / j.carSpeed
  let walkTime := (j.totalDistance - j.switchPoint) / j.walkSpeed
  carTime + walkTime

/-- Theorem stating that the journey time is 16 hours -/
theorem journey_time_is_16_hours (j : Journey) 
  (h1 : j.totalDistance = 120)
  (h2 : j.carSpeed = 30)
  (h3 : j.walkSpeed = 3)
  (h4 : j.switchPoint = 80)
  (h5 : j.switchPoint / j.carSpeed + (j.totalDistance - j.switchPoint) / j.walkSpeed = 
        2 * j.switchPoint / j.carSpeed + (j.totalDistance - j.switchPoint) / j.carSpeed) :
  journeyTime j = 16 := by
  sorry

#check journey_time_is_16_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_16_hours_l1156_115657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probability_l1156_115694

/-- The number of possible outcomes when rolling six six-sided dice -/
def total_outcomes : ℕ := 6^6

/-- The number of ways to roll one pair -/
def one_pair : ℕ := 6 * (Nat.choose 6 2) * (5 * 4 * 3 * 2)

/-- The number of ways to roll two pairs -/
def two_pairs : ℕ := (Nat.choose 6 2) * ((Nat.choose 6 2) * (Nat.choose 4 2)) * 4

/-- The number of ways to roll exactly three of a kind -/
def three_of_a_kind : ℕ := 6 * (Nat.choose 6 3) * (5 * 4 * 3)

/-- The total number of favorable outcomes -/
def favorable_outcomes : ℕ := one_pair + two_pairs + three_of_a_kind

theorem dice_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 195 / 389 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probability_l1156_115694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1156_115679

noncomputable def f (x : ℝ) : ℝ := x + 4 / x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 2^x + a

theorem problem_statement :
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 → f x₁ > f x₂) ∧
  (∀ a, a ≤ 1 → ∀ x₁, 1/2 ≤ x₁ ∧ x₁ ≤ 1 → ∃ x₂, 2 ≤ x₂ ∧ x₂ ≤ 3 ∧ f x₁ ≥ g a x₂) ∧
  (∀ a, a ≥ 0 → ∃ x, 0 < x ∧ x ≤ 2 ∧ f x ≤ g a x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1156_115679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_equals_two_l1156_115635

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 * x + Real.sqrt (4 * x^2 + 1)) + a

-- State the theorem
theorem function_sum_equals_two (a : ℝ) (h : f a 0 = 1) : 
  f a (Real.log 2) + f a (Real.log (1/2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_equals_two_l1156_115635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_set_characterization_l1156_115673

/-- The arithmetic mean of two real numbers -/
noncomputable def S (x y : ℝ) : ℝ := (x + y) / 2

/-- A set X satisfies the given condition if for any a and b in X, 
    the solution to S(a, x) = b is also in X -/
def SatisfiesCondition (X : Set ℝ) : Prop :=
  ∀ a b, a ∈ X → b ∈ X → ∃ x, x ∈ X ∧ S a x = b

theorem arithmetic_mean_set_characterization (X : Set ℝ) :
  (Finite X ∧ SatisfiesCondition X) ↔ ∃ a : ℝ, X = {a} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_set_characterization_l1156_115673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_distribution_maximizes_triangles_optimal_distribution_is_valid_optimal_distribution_is_optimal_l1156_115627

/-- Represents a distribution of points into groups -/
def Distribution := List Nat

/-- The total number of points in the space -/
def totalPoints : Nat := 1989

/-- The number of groups -/
def numGroups : Nat := 30

/-- Condition: No three points are collinear -/
axiom no_collinear_points : ∀ (p1 p2 p3 : Nat), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬(p1 = p2 ∧ p2 = p3)

/-- Function to calculate the number of triangles formed by a distribution -/
def triangleCount (d : Distribution) : Nat :=
  sorry

/-- Function to check if a distribution is valid -/
def isValidDistribution (d : Distribution) : Prop :=
  d.length = numGroups ∧
  d.sum = totalPoints ∧
  d.Nodup ∧
  List.Sorted (· < ·) d

/-- Theorem: The optimal distribution maximizes the triangle count -/
theorem optimal_distribution_maximizes_triangles (d : Distribution) :
  isValidDistribution d →
  ∀ (d' : Distribution), isValidDistribution d' →
  triangleCount d ≥ triangleCount d' :=
by
  sorry

/-- The optimal distribution -/
def optimalDistribution : Distribution :=
  [51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]

/-- Theorem: The optimal distribution is valid -/
theorem optimal_distribution_is_valid :
  isValidDistribution optimalDistribution :=
by
  sorry

/-- Theorem: The optimal distribution maximizes the triangle count -/
theorem optimal_distribution_is_optimal :
  ∀ (d : Distribution), isValidDistribution d →
  triangleCount optimalDistribution ≥ triangleCount d :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_distribution_maximizes_triangles_optimal_distribution_is_valid_optimal_distribution_is_optimal_l1156_115627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_at_origin_line_equation_at_0_1_l1156_115628

-- Define the two given lines
def l₁ (x y : ℝ) : Prop := 4 * x + y + 6 = 0
def l₂ (x y : ℝ) : Prop := 3 * x - 5 * y - 6 = 0

-- Define a line segment intercepted by two lines
def intercepted_segment (l₁ l₂ : ℝ → ℝ → Prop) (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c d : ℝ), l₁ a b ∧ l₂ c d ∧ s = {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (t * a + (1 - t) * c, t * b + (1 - t) * d)}

-- Define the midpoint of a line segment
def segment_midpoint (s : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ × ℝ), a ∈ s ∧ b ∈ s ∧ p = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

-- Theorem for the case when P is at (0, 0)
theorem line_equation_at_origin (s : Set (ℝ × ℝ)) :
  intercepted_segment l₁ l₂ s → segment_midpoint s (0, 0) →
  ∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ s → x + 6 * y = 0 :=
sorry

-- Theorem for the case when P is at (0, 1)
theorem line_equation_at_0_1 (s : Set (ℝ × ℝ)) :
  intercepted_segment l₁ l₂ s → segment_midpoint s (0, 1) →
  ∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ s → x + 2 * y - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_at_origin_line_equation_at_0_1_l1156_115628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_function_inequality_l1156_115606

-- Define an odd function that is monotonically decreasing on [-1,0]
def odd_decreasing_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f x > f y)

-- Define acute angles of a triangle
def acute_angles_of_triangle (α β : ℝ) : Prop :=
  0 < α ∧ α < Real.pi/2 ∧ 0 < β ∧ β < Real.pi/2 ∧ α + β < Real.pi

-- Theorem statement
theorem odd_decreasing_function_inequality 
  (f : ℝ → ℝ) (α β : ℝ) 
  (h_f : odd_decreasing_function f) 
  (h_angles : acute_angles_of_triangle α β) : 
  f (Real.sin α) < f (Real.cos β) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_function_inequality_l1156_115606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_comparison_l1156_115691

noncomputable def parabola1 (x : ℝ) : ℝ := x^2 - x + 3
noncomputable def parabola2 (x : ℝ) : ℝ := x^2 + x + 1

noncomputable def vertex (a b c : ℝ) : ℝ × ℝ :=
  let h := -b / (2 * a)
  let k := c - b^2 / (4 * a)
  (h, k)

theorem parabola_comparison :
  let v1 := vertex 1 (-1) 3
  let v2 := vertex 1 1 1
  v1.1 > v2.1 ∧ v1.2 > v2.2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_comparison_l1156_115691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_single_elimination_tournament_games_l1156_115665

/-- The number of games needed in a single-elimination tournament with n players -/
def number_of_games (n : ℕ) : ℕ := n - 1

theorem single_elimination_tournament_games (n : ℕ) (h : n > 0) : 
  number_of_games n = n - 1 :=
by
  -- Unfold the definition of number_of_games
  unfold number_of_games
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_single_elimination_tournament_games_l1156_115665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l1156_115626

theorem sin_double_angle_special_case (α : ℝ) 
  (h1 : Real.sin (π - α) = 2/3) 
  (h2 : π/2 < α ∧ α < π) : 
  Real.sin (2*α) = -(4*Real.sqrt 5)/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l1156_115626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l1156_115621

theorem largest_power_of_18_dividing_30_factorial :
  ∃ n : ℕ, n = 7 ∧ 
   (∀ m : ℕ, 18^m ∣ Nat.factorial 30 → m ≤ n) ∧
   18^n ∣ Nat.factorial 30 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l1156_115621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_arch_water_width_l1156_115686

/-- Represents a parabolic arch bridge --/
structure ParabolicArch where
  a : ℝ
  init_width : ℝ
  init_height : ℝ

/-- Calculates the width of the water for a given height --/
noncomputable def water_width (arch : ParabolicArch) (height : ℝ) : ℝ :=
  2 * Real.sqrt (-arch.a * height)

theorem parabolic_arch_water_width 
  (arch : ParabolicArch)
  (h_a : arch.a = -8)
  (h_init_width : arch.init_width = 8)
  (h_init_height : arch.init_height = -2)
  (h_new_height : arch.init_height + (1/2) = -3/2) :
  water_width arch (arch.init_height + (1/2)) = 4 * Real.sqrt 3 := by
  sorry

#check parabolic_arch_water_width

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_arch_water_width_l1156_115686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1156_115608

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / (1 - 5*x)^2

-- State the theorem about the range of f
theorem range_of_f : 
  Set.range f = {y : ℝ | y > 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1156_115608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birdseed_mix_proportion_l1156_115655

/-- Given two varieties of birdseed with different millet percentages,
    this theorem proves the proportion of the first variety needed to create
    a mix with a specific millet percentage. -/
theorem birdseed_mix_proportion
  (millet_a millet_b target_millet : ℝ)
  (ha : 0 ≤ millet_a ∧ millet_a ≤ 1)
  (hb : 0 ≤ millet_b ∧ millet_b ≤ 1)
  (ht : 0 ≤ target_millet ∧ target_millet ≤ 1)
  (h_distinct : millet_a ≠ millet_b)
  (h_millet_a : millet_a = 0.4)
  (h_millet_b : millet_b = 0.65)
  (h_target : target_millet = 0.5) :
  (millet_b - target_millet) / (millet_b - millet_a) = 0.6 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_birdseed_mix_proportion_l1156_115655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_neg_one_l1156_115647

noncomputable section

/-- A function f is odd if f(-x) = -f(x) for all x in its domain --/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = ((x+1)(x+a))/x --/
def f (a : ℝ) (x : ℝ) : ℝ := ((x + 1) * (x + a)) / x

theorem odd_function_implies_a_eq_neg_one :
  ∀ a : ℝ, IsOdd (f a) → a = -1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_neg_one_l1156_115647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_range_l1156_115696

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 - 2*a)^x

theorem decreasing_exponential_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) →
  0 < a ∧ a < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_range_l1156_115696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_columns_count_l1156_115637

/-- The length of the line surrounding the monument in centimeters -/
noncomputable def L : ℝ := sorry

/-- The number of columns -/
def N : ℕ := sorry

/-- The first condition: if columns are placed 10 cm apart, 150 columns are not enough -/
axiom condition1 : N > L / 10 ∧ L / 10 > 150

/-- The second condition: if columns are placed 30 cm apart, 70 columns remain -/
axiom condition2 : N = Int.floor (L / 30) + 70

/-- The theorem stating that N equals 180 -/
theorem columns_count : N = 180 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_columns_count_l1156_115637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l1156_115684

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + 5/3

theorem cubic_function_properties (a b : ℝ) :
  (∀ x, (deriv (deriv (f a b))) x = 0 → x = 1) →
  f a b 1 = 1 →
  (a = 1/3 ∧ b = -1) ∧
  (∃ x₁ x₂, x₁ < x₂ ∧ 
    (∀ x, x < x₁ → deriv (f a b) x > 0) ∧
    (∀ x, x₁ < x ∧ x < x₂ → deriv (f a b) x < 0) ∧
    (∀ x, x > x₂ → deriv (f a b) x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l1156_115684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weakly_increasing_h_iff_conditions_l1156_115663

noncomputable section

/-- A function is weakly increasing on an interval if it is increasing and its ratio to x is decreasing on that interval -/
def WeaklyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a < x ∧ x < y ∧ y < b → f x ≤ f y) ∧
  (∀ x y, a < x ∧ x < y ∧ y < b → f x / x ≥ f y / y)

/-- The function h(x) = x^2 + (sin θ - 1/2)x + b -/
def h (θ : ℝ) (b : ℝ) (x : ℝ) : ℝ :=
  x^2 + (Real.sin θ - 1/2) * x + b

theorem weakly_increasing_h_iff_conditions (θ : ℝ) (b : ℝ) :
  WeaklyIncreasing (h θ b) 0 1 ↔
    b ≥ 1 ∧ ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 6 ≤ θ ∧ θ ≤ 2 * k * Real.pi + 5 * Real.pi / 6 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weakly_increasing_h_iff_conditions_l1156_115663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_is_correct_l1156_115698

open Real

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (-x^5 + 25*x^3 + 1) / (x^2 + 5*x)

-- Define the antiderivative
noncomputable def F (x : ℝ) : ℝ := -x^4/4 + 5*x^3/3 + (1/5) * log (abs x) - (1/5) * log (abs (x+5))

-- State the theorem
theorem integral_is_correct (x : ℝ) (h : x ≠ 0 ∧ x ≠ -5) : 
  deriv F x = f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_is_correct_l1156_115698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faulty_odometer_003006_l1156_115615

/-- Represents a faulty odometer that skips digits 4 and 7 --/
structure FaultyOdometer where
  reading : Nat

/-- Converts a faulty odometer reading to actual miles --/
def actualMiles (o : FaultyOdometer) : Nat :=
  sorry

/-- The theorem to be proved --/
theorem faulty_odometer_003006 :
  actualMiles ⟨3006⟩ = 1541 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faulty_odometer_003006_l1156_115615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_half_l1156_115676

noncomputable def f (x a : ℝ) : ℝ := 1 / (2^x - 1) + a

theorem odd_function_implies_a_half :
  (∀ x, f x a = -f (-x) a) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_half_l1156_115676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_with_odd_rightmost_nonzero_digit_l1156_115633

def b (n : ℕ) : ℕ := (n + 6).factorial / (n - 1).factorial

def rightmostNonzeroDigitIsOdd (n : ℕ) : Prop :=
  ∃ (m : ℕ), n % (10 ^ (m + 1)) ≠ 0 ∧ (n % (10 ^ (m + 1))) % 2 = 1

theorem smallest_k_with_odd_rightmost_nonzero_digit :
  (∀ k < 25, ¬rightmostNonzeroDigitIsOdd (b k)) ∧
  rightmostNonzeroDigitIsOdd (b 25) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_with_odd_rightmost_nonzero_digit_l1156_115633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_line_l1156_115602

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the line C₂
def C₂ (x y : ℝ) : Prop := x + y - 4 = 0

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Statement of the theorem
theorem min_distance_ellipse_line :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧
    (∀ (x₃ y₃ x₄ y₄ : ℝ), C₁ x₃ y₃ → C₂ x₄ y₄ →
      distance x₁ y₁ x₂ y₂ ≤ distance x₃ y₃ x₄ y₄) ∧
    distance x₁ y₁ x₂ y₂ = Real.sqrt 2 ∧
    x₁ = 3/2 ∧ y₁ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_line_l1156_115602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_perimeter_product_l1156_115609

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the area of a square given its side length -/
def squareArea (sideLength : ℝ) : ℝ :=
  sideLength^2

/-- Calculate the perimeter of a square given its side length -/
def squarePerimeter (sideLength : ℝ) : ℝ :=
  4 * sideLength

theorem square_area_perimeter_product :
  let e : Point := ⟨5, 5⟩
  let h : Point := ⟨0, 4⟩
  let sideLength := distance e h
  let area := squareArea sideLength
  let perimeter := squarePerimeter sideLength
  area * perimeter = 104 * Real.sqrt 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_perimeter_product_l1156_115609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_beta_l1156_115674

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : β ∈ Set.Ioo 0 (π/2)) 
  (h3 : Real.cos α = 1/3) 
  (h4 : Real.cos (α + β) = -1/3) : 
  Real.cos (α - β) = 23/27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_beta_l1156_115674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undamaged_boats_count_l1156_115644

/-- Calculates the number of undamaged boats after a series of events --/
def undamagedBoats (initialBoats : ℕ) (fishPercent arrowPercent windPercent sinkPercent : ℚ) : ℕ :=
  let remainingAfterFish := initialBoats - Int.floor (initialBoats * fishPercent)
  let remainingAfterArrows := remainingAfterFish - Int.floor (remainingAfterFish * arrowPercent)
  let remainingAfterWind := remainingAfterArrows - Int.floor (remainingAfterArrows * windPercent)
  (remainingAfterWind - Int.floor (remainingAfterWind * sinkPercent)).toNat

/-- Theorem stating that the number of undamaged boats is 15 --/
theorem undamaged_boats_count :
  undamagedBoats 30 (20/100) (10/100) (25/100) (15/100) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_undamaged_boats_count_l1156_115644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_travel_time_l1156_115610

/-- Represents the speed of an object in the river -/
structure RiverSpeed where
  boat : ℝ  -- Speed of the boat in still water
  current : ℝ  -- Speed of the river current

/-- Represents a dock on the river -/
inductive Dock
  | Branch1 : Dock  -- 1 km from confluence on branch 1
  | Branch2 : Dock  -- 1 km from confluence on branch 2
  | Downstream : Dock  -- 2 km downstream from confluence

/-- Calculates the travel time between two docks -/
noncomputable def travelTime (speed : RiverSpeed) (from_dock to_dock : Dock) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem river_travel_time 
  (speed : RiverSpeed)
  (dock1 dock2 dock3 : Dock)
  (h1 : dock1 ≠ dock2 ∧ dock2 ≠ dock3 ∧ dock1 ≠ dock3)
  (h2 : travelTime speed dock1 dock2 = 30)
  (h3 : travelTime speed dock2 dock3 = 18) :
  travelTime speed dock3 dock1 = 24 ∨ travelTime speed dock3 dock1 = 72 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_travel_time_l1156_115610
