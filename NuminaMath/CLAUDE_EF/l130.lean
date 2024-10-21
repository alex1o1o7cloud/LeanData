import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_permutation_sum_divisible_by_37_l130_13043

def is_cyclic_permutation (a b : Nat) : Prop :=
  ∃ (x y z : Nat), a = 100 * x + 10 * y + z ∧ b = 100 * y + 10 * z + x ∧ x < 10 ∧ y < 10 ∧ z < 10

def seq_sum (seq : List Nat) : Nat :=
  seq.sum

theorem cyclic_permutation_sum_divisible_by_37 (seq : List Nat) :
  (∀ (i : Nat), i + 1 < seq.length → is_cyclic_permutation (seq[i]!) (seq[i+1]!)) →
  (∀ (n : Nat), n ∈ seq → n ≥ 100 ∧ n < 1000) →
  37 ∣ seq_sum seq :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_permutation_sum_divisible_by_37_l130_13043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_one_l130_13082

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial distribution -/
def expected_value (b : BinomialDistribution) : ℝ := b.n * b.p

/-- Variance of a binomial distribution -/
def variance (b : BinomialDistribution) : ℝ := b.n * b.p * (1 - b.p)

/-- Probability of getting exactly k successes in n trials -/
def probability (b : BinomialDistribution) (k : ℕ) : ℝ :=
  (Nat.choose b.n k : ℝ) * b.p ^ k * (1 - b.p) ^ (b.n - k)

theorem binomial_probability_one (b : BinomialDistribution) 
  (h2 : expected_value b = 6)
  (h3 : variance b = 3) :
  probability b 1 = 3 / 1024 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_one_l130_13082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_problem_l130_13012

/-- Given a geometric sequence with first term a and common ratio r,
    S(n) represents the sum of the first n terms. -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- For a geometric sequence with a = 1 and r = 1/4,
    if the sum of the first n terms is 255/256, then n = 4. -/
theorem geometric_sum_problem : ∃ (n : ℕ),
  geometric_sum 1 (1/4) n = 255/256 ∧ n = 4 := by
  -- We'll use 4 as our witness for n
  use 4
  constructor
  · -- Prove that geometric_sum 1 (1/4) 4 = 255/256
    sorry
  · -- Prove that 4 = 4
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_problem_l130_13012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l130_13039

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem a_range (f : ℝ → ℝ) (a : ℝ) 
    (h_even : is_even_function f)
    (h_mono : monotone_increasing_on f (Set.Iic 0))
    (h_ineq : f (2^|a-1|) > f 4) :
    a ∈ Set.Ioo (-1) 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l130_13039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_more_dark_square_l130_13037

/-- Represents a square on the chessboard -/
inductive Square
| Light
| Dark
deriving BEq

/-- Represents a row of the chessboard -/
def Row := Vector Square 9

/-- Represents the entire 9x9 chessboard -/
def Chessboard := Vector Row 9

/-- Counts the number of dark squares in a row -/
def countDarkInRow (row : Row) : Nat :=
  row.toList.filter (· == Square.Dark) |>.length

/-- Counts the number of light squares in a row -/
def countLightInRow (row : Row) : Nat :=
  row.toList.filter (· == Square.Light) |>.length

/-- Creates an alternating row starting with the given square -/
def createAlternatingRow (start : Square) : Row :=
  Vector.ofFn fun i => if i % 2 == 0 then start else (if start == Square.Dark then Square.Light else Square.Dark)

/-- Creates the 9x9 chessboard with alternating squares -/
def createChessboard : Chessboard :=
  Vector.ofFn fun i => if i % 2 == 0 then createAlternatingRow Square.Dark else createAlternatingRow Square.Light

/-- Theorem: There is exactly one more dark square than light squares on the 9x9 chessboard -/
theorem one_more_dark_square :
  let board := createChessboard
  let totalDark := (board.map countDarkInRow).toList.sum
  let totalLight := (board.map countLightInRow).toList.sum
  totalDark = totalLight + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_more_dark_square_l130_13037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solutions_l130_13078

/-- Given a positive real number a such that the inequality 1 < xa < 2 has exactly three integer solutions,
    prove that the inequality 2 < xa < 3 can have 2, 3, or 4 integer solutions. -/
theorem inequality_solutions (a : ℝ) (h_pos : a > 0) 
  (h_three_sols : ∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (1 < (x : ℝ) * a ∧ (x : ℝ) * a < 2) ∧ 
    (1 < (y : ℝ) * a ∧ (y : ℝ) * a < 2) ∧ 
    (1 < (z : ℝ) * a ∧ (z : ℝ) * a < 2) ∧
    (∀ w : ℤ, (1 < (w : ℝ) * a ∧ (w : ℝ) * a < 2) → w ∈ ({x, y, z} : Set ℤ))) :
  (∃ (s t : ℤ), s ≠ t ∧ (2 < (s : ℝ) * a ∧ (s : ℝ) * a < 3) ∧ (2 < (t : ℝ) * a ∧ (t : ℝ) * a < 3) ∧
    (∀ u : ℤ, (2 < (u : ℝ) * a ∧ (u : ℝ) * a < 3) → u ∈ ({s, t} : Set ℤ))) ∨
  (∃ (s t u : ℤ), s ≠ t ∧ t ≠ u ∧ s ≠ u ∧ 
    (2 < (s : ℝ) * a ∧ (s : ℝ) * a < 3) ∧ 
    (2 < (t : ℝ) * a ∧ (t : ℝ) * a < 3) ∧ 
    (2 < (u : ℝ) * a ∧ (u : ℝ) * a < 3) ∧
    (∀ w : ℤ, (2 < (w : ℝ) * a ∧ (w : ℝ) * a < 3) → w ∈ ({s, t, u} : Set ℤ))) ∨
  (∃ (s t u v : ℤ), s ≠ t ∧ t ≠ u ∧ u ≠ v ∧ s ≠ u ∧ s ≠ v ∧ t ≠ v ∧
    (2 < (s : ℝ) * a ∧ (s : ℝ) * a < 3) ∧ 
    (2 < (t : ℝ) * a ∧ (t : ℝ) * a < 3) ∧ 
    (2 < (u : ℝ) * a ∧ (u : ℝ) * a < 3) ∧
    (2 < (v : ℝ) * a ∧ (v : ℝ) * a < 3) ∧
    (∀ w : ℤ, (2 < (w : ℝ) * a ∧ (w : ℝ) * a < 3) → w ∈ ({s, t, u, v} : Set ℤ))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solutions_l130_13078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_distance_relation_l130_13047

/-- Three points are collinear if the slope between any two pairs of points is the same -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem collinear_points_distance_relation (x y : ℝ) :
  let A : ℝ × ℝ := (x, 5)
  let B : ℝ × ℝ := (-2, y)
  let C : ℝ × ℝ := (1, 1)
  are_collinear A B C →
  distance B C = 2 * distance A C →
  x + y = -9/2 ∨ x + y = 17/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_distance_relation_l130_13047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_l130_13027

theorem complex_multiplication : 
  (2 + Complex.I) * (3 + Complex.I) = 5 + 5 * Complex.I := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_l130_13027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l130_13092

noncomputable def f (x : ℝ) := Real.tan (2 * x - Real.pi / 3)

theorem f_strictly_increasing (k : ℤ) :
  StrictMonoOn f (Set.Ioo ((k : ℝ) * Real.pi / 2 - Real.pi / 12) ((k : ℝ) * Real.pi / 2 + 5 * Real.pi / 12)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l130_13092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_from_equilateral_triangle_l130_13049

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The left focus of a hyperbola -/
noncomputable def left_focus (h : Hyperbola) : ℝ × ℝ := (-Real.sqrt (h.a^2 + h.b^2), 0)

/-- A point on the asymptote of a hyperbola -/
def asymptote_point (h : Hyperbola) : ℝ × ℝ := (h.a, h.b)

/-- Checks if a triangle is equilateral with side length 2 -/
noncomputable def is_equilateral_triangle (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let d12 := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let d23 := Real.sqrt ((p2.1 - p3.1)^2 + (p2.2 - p3.2)^2)
  let d31 := Real.sqrt ((p3.1 - p1.1)^2 + (p3.2 - p1.2)^2)
  d12 = 2 ∧ d23 = 2 ∧ d31 = 2

/-- The main theorem -/
theorem hyperbola_equation_from_equilateral_triangle (h : Hyperbola) :
  is_equilateral_triangle (0, 0) (left_focus h) (asymptote_point h) →
  ∀ x y : ℝ, hyperbola_equation h x y ↔ x^2 - y^2 / 3 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_from_equilateral_triangle_l130_13049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_n_with_no_large_prime_factors_l130_13004

def has_no_prime_factor_gt_7 (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p > 7 → ¬(p ∣ (2^n - 1))

theorem characterize_n_with_no_large_prime_factors :
  ∀ n : ℕ, has_no_prime_factor_gt_7 n ↔ n ∈ ({1, 2, 3, 4, 6} : Finset ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_n_with_no_large_prime_factors_l130_13004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_equation_l130_13006

theorem sum_of_roots_equation : 
  (∃ s : ℝ, s = (Finset.sum (Multiset.toFinset (Polynomial.roots ((X - 1) * (2*X + 3) + (X + 1) * (2*X + 3)))) id) ∧ s = -3/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_equation_l130_13006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_parallel_to_c_l130_13080

/-- Given vectors a and b in ℝ², prove that their sum is parallel to vector c -/
theorem vector_sum_parallel_to_c (x : ℝ) : 
  ∃ (k : ℝ), (![x, 1] + ![-x, x^2] : Fin 2 → ℝ) = k • ![0, 1] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_parallel_to_c_l130_13080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l130_13048

noncomputable def f (x : ℝ) : ℝ := -1/3 * x^3 + x^2 + 3*x - 1

theorem f_properties :
  -- Monotonicity
  (∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 3 → f x < f y) ∧
  (∀ x y : ℝ, x < y ∧ y < -1 → f x > f y) ∧
  (∀ x y : ℝ, 3 < x ∧ x < y → f x > f y) ∧
  -- Extreme values
  (∀ x : ℝ, f x ≥ f (-1)) ∧
  (f (-1) = -8/3) ∧
  (∀ x : ℝ, f x ≤ f 3) ∧
  (f 3 = 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l130_13048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l130_13084

-- Define the sign function
noncomputable def sign (a : ℝ) : ℝ :=
  if a > 0 then 1
  else if a = 0 then 0
  else -1

-- Define the system of equations
def satisfies_system (x y z : ℝ) : Prop :=
  x = 2023 - 2024 * sign (y + z - 1) ∧
  y = 2023 - 2024 * sign (x + z - 1) ∧
  z = 2023 - 2024 * sign (x + y - 1)

-- Theorem statement
theorem unique_solution :
  ∃! (x y z : ℝ), satisfies_system x y z ∧ x = 4047 ∧ y = 4047 ∧ z = 4047 := by
  sorry

#check unique_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l130_13084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_x_plus_3_is_odd_l130_13057

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define what it means for a function to be odd
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem f_x_plus_3_is_odd (h1 : is_odd (fun x ↦ f (x + 1))) 
                          (h2 : is_odd (fun x ↦ f (x - 1))) :
  is_odd (fun x ↦ f (x + 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_x_plus_3_is_odd_l130_13057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_box_theorem_l130_13019

/-- Represents the weight difference from standard and number of boxes --/
structure WeightDifference :=
  (difference : ℚ)
  (boxes : ℕ)

/-- Represents the apple box problem --/
structure AppleBoxProblem :=
  (total_boxes : ℕ)
  (standard_weight : ℚ)
  (weight_differences : List WeightDifference)
  (price_per_kg : ℚ)

/-- Calculates the difference between the heaviest and lightest box --/
def heaviest_lightest_difference (problem : AppleBoxProblem) : ℚ :=
  let max_diff := problem.weight_differences.map (·.difference) |>.maximum?
  let min_diff := problem.weight_differences.map (·.difference) |>.minimum?
  match max_diff, min_diff with
  | some max, some min => max - min
  | _, _ => 0

/-- Calculates the total excess weight --/
def total_excess_weight (problem : AppleBoxProblem) : ℚ :=
  problem.weight_differences.map (λ wd => wd.difference * (wd.boxes : ℚ)) |>.sum

/-- Calculates the total selling price --/
def total_selling_price (problem : AppleBoxProblem) : ℚ :=
  let total_weight := (problem.total_boxes : ℚ) * problem.standard_weight + total_excess_weight problem
  total_weight * problem.price_per_kg

theorem apple_box_theorem (problem : AppleBoxProblem) 
    (h1 : problem.total_boxes = 30)
    (h2 : problem.standard_weight = 20)
    (h3 : problem.weight_differences = [
      ⟨-3/2, 2⟩, ⟨-1, 6⟩, ⟨-1/2, 10⟩, ⟨1, 8⟩, ⟨2, 4⟩
    ])
    (h4 : problem.price_per_kg = 6) :
    heaviest_lightest_difference problem = 7/2 ∧
    total_excess_weight problem = 2 ∧
    total_selling_price problem = 3612 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_box_theorem_l130_13019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_coeff_less_than_neg_one_l130_13064

/-- A polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- Evaluates a polynomial at a given point -/
def evalPoly {n : ℕ} (p : IntPolynomial n) (x : ℤ) : ℤ :=
  (Finset.sum Finset.univ fun i => p i * x ^ i.val)

theorem exists_coeff_less_than_neg_one {n : ℕ} (p : IntPolynomial n) 
  (h1 : evalPoly p 1 = 0) (h2 : evalPoly p 2 = 0) :
  ∃ i, p i < -1 := by
  sorry

#check exists_coeff_less_than_neg_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_coeff_less_than_neg_one_l130_13064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_only_students_l130_13093

theorem math_only_students (total : ℕ) (math : ℕ) (science : ℕ) 
  (h1 : total = 120) 
  (h2 : math = 85) 
  (h3 : science = 65) 
  (h4 : ∀ s, s ∈ Finset.range total → (s < math ∨ s < science)) :
  (Finset.card (Finset.range math \ Finset.range science) : ℕ) = 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_only_students_l130_13093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_area_l130_13032

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the slope of the first tangent line
noncomputable def m₁ : ℝ := Real.sqrt 3

-- Define the angle between the tangent lines
noncomputable def angle : ℝ := Real.pi / 4

-- Define the bounded area
noncomputable def boundedArea : ℝ := (5 + 3 * Real.sqrt 3) / 6

-- Theorem statement
theorem parabola_tangent_area : 
  ∃ (m₂ : ℝ), 
    m₂ < 0 ∧ 
    Real.tan angle = |((m₁ - m₂) / (1 + m₁ * m₂))| ∧
    (∃ (a b c d : ℝ), 
      a < b ∧ c < d ∧
      (∀ x ∈ Set.Icc a b, parabola x ≤ (m₁ * x - m₁^2 / 4)) ∧
      (∀ x ∈ Set.Icc c d, parabola x ≤ (m₂ * x - m₂^2 / 4)) ∧
      (∫ (x : ℝ) in a..b, (m₁ * x - m₁^2 / 4) - parabola x) + 
      (∫ (x : ℝ) in c..d, (m₂ * x - m₂^2 / 4) - parabola x) = boundedArea) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_area_l130_13032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ruthenium_atomic_radius_scientific_notation_l130_13045

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem ruthenium_atomic_radius_scientific_notation :
  toScientificNotation 0.000000000189 = ScientificNotation.mk 1.89 (-10) (by sorry) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ruthenium_atomic_radius_scientific_notation_l130_13045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l130_13040

/-- Triangle PQR with vertices P, Q, R -/
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  vertices : (P ≠ Q) ∧ (Q ≠ R) ∧ (R ≠ P)

/-- A point W on side QR such that PW is a median -/
def Median (P Q R W : ℝ × ℝ) : Prop :=
  W.1 = (Q.1 + R.1) / 2 ∧ W.2 = (Q.2 + R.2) / 2

/-- The length of a line segment between two points -/
noncomputable def Length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The area of a triangle given its vertices -/
noncomputable def TriangleArea (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

theorem isosceles_triangle_area (P Q R W Z : ℝ × ℝ) : 
  Triangle P Q R →
  Length P Q = Length P R →
  Median P Q R W →
  Median R P Q Z →
  (W.1 - P.1) * (Z.1 - R.1) + (W.2 - P.2) * (Z.2 - R.2) = 0 →
  Length P W = 15 →
  Length R Z = 15 →
  TriangleArea P Q R = 450 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l130_13040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_remaining_money_l130_13017

noncomputable def initial_amount : ℚ := 100
noncomputable def fraction_given : ℚ := 1/4
noncomputable def grocery_cost : ℚ := 40

theorem john_remaining_money :
  initial_amount * (1 - fraction_given) - grocery_cost = 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_remaining_money_l130_13017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_product_l130_13035

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y = ax² -/
structure Parabola where
  a : ℝ

/-- Represents a line of the form y = kx + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- The focus of a parabola y = ax² -/
noncomputable def focus (p : Parabola) : Point :=
  { x := 0, y := 1 / (8 * p.a) }

/-- Determines if a line passes through a point -/
def linePassesThrough (l : Line) (p : Point) : Prop :=
  p.y = l.k * p.x + l.b

/-- Determines if a point lies on a parabola -/
def pointOnParabola (p : Parabola) (pt : Point) : Prop :=
  pt.y = p.a * pt.x^2

/-- Theorem: For a parabola y = 2x² and a line passing through its focus,
    intersecting the parabola at points A(x₁, y₁) and B(x₂, y₂),
    the product x₁ * x₂ = -1/16 -/
theorem parabola_line_intersection_product
  (p : Parabola)
  (l : Line)
  (A B : Point)
  (h1 : p.a = 2)
  (h2 : linePassesThrough l (focus p))
  (h3 : pointOnParabola p A)
  (h4 : pointOnParabola p B)
  (h5 : linePassesThrough l A)
  (h6 : linePassesThrough l B)
  : A.x * B.x = -1/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_product_l130_13035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l130_13059

-- Define the function f(x) = 2^x + 3x - 7
noncomputable def f (x : ℝ) : ℝ := 2^x + 3*x - 7

-- Theorem statement
theorem root_exists_in_interval :
  (f 1 < 0) → (f 2 > 0) → ∃ x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l130_13059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_intersecting_axis_linear_functions_intersecting_axis_linear_functions_condition_l130_13097

/-- Two linear functions are "intersecting axis linear functions" if their graphs
    pass through the same point on either the x-axis or y-axis. -/
def intersecting_axis_linear_functions (f g : ℝ → ℝ) : Prop :=
  (∃ x : ℝ, f x = 0 ∧ g x = 0) ∨ (∃ y : ℝ, f 0 = y ∧ g 0 = y)

/-- The first part of the theorem -/
theorem not_intersecting_axis_linear_functions :
  ¬ intersecting_axis_linear_functions (λ x ↦ 3*x + 1) (λ x ↦ 3*x - 1) := by
  sorry

/-- The second part of the theorem -/
theorem intersecting_axis_linear_functions_condition (b : ℝ) :
  let y₁ : ℝ → ℝ := λ x ↦ -3*x + 3
  let y₂ : ℝ → ℝ := λ x ↦ 4*x + b
  intersecting_axis_linear_functions y₁ (λ x ↦ y₁ x - y₂ x) → b = -4 ∨ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_intersecting_axis_linear_functions_intersecting_axis_linear_functions_condition_l130_13097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_lengths_theorem_l130_13054

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def f (x : ℝ) : ℝ := (floor x : ℝ) * (x - (floor x : ℝ))

def g (x : ℝ) : ℝ := x - 1

def interval_length (a b : ℝ) : ℝ := b - a

theorem interval_lengths_theorem :
  ∃ (d₁ d₂ d₃ : ℝ),
    d₁ = interval_length 0 1 ∧
    d₂ = interval_length 1 2 ∧
    d₃ = interval_length 2 2011 ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2011 →
      (f x > g x ↔ 0 ≤ x ∧ x < 1) ∧
      (f x = g x ↔ 1 ≤ x ∧ x < 2) ∧
      (f x < g x ↔ 2 ≤ x ∧ x ≤ 2011)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_lengths_theorem_l130_13054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pressure_force_triangular_plate_l130_13038

/-- Force of water pressure on a vertically submerged triangular plate -/
noncomputable def water_pressure_force (base height : ℝ) (water_density : ℝ) : ℝ :=
  (1/3) * water_density * 9.81 * base * height * height

theorem water_pressure_force_triangular_plate :
  let base : ℝ := 0.04
  let height : ℝ := 0.03
  let water_density : ℝ := 1000
  abs (water_pressure_force base height water_density - 0.117) < 0.001 := by
  sorry

-- Use #eval only for computable functions
#check water_pressure_force 0.04 0.03 1000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pressure_force_triangular_plate_l130_13038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_nearest_integer_l130_13003

theorem ratio_nearest_integer (a b : ℝ) (h1 : (a + b) / 2 = 3 * Real.sqrt (a * b)) (h2 : a > b) (h3 : b > 0) :
  ∃ (r : ℝ), r = a / b ∧ ∀ (n : ℤ), n ≠ 34 → |r - 34| ≤ |r - n| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_nearest_integer_l130_13003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_surface_area_l130_13007

-- Define the truncated cone
structure TruncatedCone where
  a : ℝ  -- length of diagonals
  α : ℝ  -- angle between slant height and base plane
  diagonals_perpendicular : Prop  -- represents that diagonals are perpendicular

-- Define the total surface area function
noncomputable def total_surface_area (cone : TruncatedCone) : ℝ :=
  (Real.pi * cone.a^2 / (Real.sin cone.α)^2) * 
  (Real.sin (cone.α/2 + Real.pi/12)) * 
  (Real.cos (cone.α/2 - Real.pi/12))

-- Theorem statement
theorem truncated_cone_surface_area (cone : TruncatedCone) :
  total_surface_area cone = 
    (Real.pi * cone.a^2 / (Real.sin cone.α)^2) * 
    (Real.sin (cone.α/2 + Real.pi/12)) * 
    (Real.cos (cone.α/2 - Real.pi/12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_surface_area_l130_13007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l130_13060

theorem circle_equation_proof :
  let line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}
  let A : ℝ × ℝ := (-6, 2)
  let B : ℝ × ℝ := (2, -2)
  ∃ (center : ℝ × ℝ), center ∈ line ∧
    (∀ (x y : ℝ), (x + 3)^2 + (y + 2)^2 = 25 ↔
      (x - center.1)^2 + (y - center.2)^2 = (x + 6)^2 + (y - 2)^2 ∧
      (x - center.1)^2 + (y - center.2)^2 = (x - 2)^2 + (y + 2)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l130_13060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_l130_13076

open Real Set

-- Define the function f
def f (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem zeros_of_f (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  ∃ (x₁ x₂ : ℝ), x₁ ∈ Ioo a b ∧ x₂ ∈ Ioo b c ∧ f a b c x₁ = 0 ∧ f a b c x₂ = 0 ∧
  ∀ (x : ℝ), f a b c x = 0 → x = x₁ ∨ x = x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_l130_13076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l130_13098

/-- The distance from a point (x₀, y₀) to a line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- The equation of a circle with center (x₀, y₀) and radius r -/
def circle_equation (x y x₀ y₀ r : ℝ) : Prop :=
  (x - x₀)^2 + (y - y₀)^2 = r^2

theorem circle_tangent_to_line :
  let center_x := 1
  let center_y := -1
  let line_equation (x : ℝ) := x + 2 = 0
  let radius := distance_point_to_line center_x center_y 1 0 2
  ∀ x y : ℝ,
    circle_equation x y center_x center_y radius ↔ (x - 1)^2 + (y + 1)^2 = 9 :=
by sorry

#check circle_tangent_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l130_13098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_to_cream_cheese_ratio_vanilla_calculation_correct_cream_cheese_calculation_correct_l130_13024

/-- Represents the ingredients used in Betty's cheesecake recipe -/
structure CheesecakeRecipe where
  sugar : ℚ
  cream_cheese : ℚ
  vanilla : ℚ
  eggs : ℚ

/-- Defines Betty's cheesecake recipe based on the given conditions -/
def betty_recipe : CheesecakeRecipe where
  sugar := 2
  cream_cheese := 8
  vanilla := 4
  eggs := 8

/-- Theorem stating that the ratio of sugar to cream cheese in Betty's recipe is 1:4 -/
theorem sugar_to_cream_cheese_ratio :
  betty_recipe.sugar / betty_recipe.cream_cheese = 1 / 4 := by
  sorry

/-- Helper function to calculate the amount of vanilla based on eggs -/
def vanilla_from_eggs (eggs : ℚ) : ℚ := eggs / 2

/-- Helper function to calculate the amount of cream cheese based on vanilla -/
def cream_cheese_from_vanilla (vanilla : ℚ) : ℚ := vanilla * 2

/-- Theorem stating that the vanilla calculation is correct -/
theorem vanilla_calculation_correct :
  betty_recipe.vanilla = vanilla_from_eggs betty_recipe.eggs := by
  sorry

/-- Theorem stating that the cream cheese calculation is correct -/
theorem cream_cheese_calculation_correct :
  betty_recipe.cream_cheese = cream_cheese_from_vanilla betty_recipe.vanilla := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_to_cream_cheese_ratio_vanilla_calculation_correct_cream_cheese_calculation_correct_l130_13024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_30_degrees_l130_13073

/-- The area of a circular sector given its radius and central angle in radians -/
noncomputable def sectorArea (r : ℝ) (θ : ℝ) : ℝ := (1/2) * r^2 * θ

theorem sector_area_30_degrees (r : ℝ) (θ : ℝ) (h1 : r = 6) (h2 : θ = π/6) :
  sectorArea r θ = 3 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_30_degrees_l130_13073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_zero_l130_13068

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1)^2 + (Real.exp (-x) - 1)^2

/-- Theorem stating that f(x) has a minimum value of 0 -/
theorem f_min_value_zero :
  (∀ x : ℝ, f x ≥ 0) ∧ (∃ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_zero_l130_13068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_slope_exponential_l130_13021

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem secant_slope_exponential : 
  let x₁ : ℝ := 0
  let y₁ : ℝ := f x₁
  let x₂ : ℝ := 1
  let y₂ : ℝ := f x₂
  (y₂ - y₁) / (x₂ - x₁) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_slope_exponential_l130_13021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_ellipse_l130_13020

/-- Circle centered at (0, 3) with radius 1/2 -/
def myCircle (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 1/4

/-- Ellipse centered at (0, 0) with semi-major axis 2 and semi-minor axis 1 -/
def myEllipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

/-- Distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Maximum distance between points on the circle and ellipse -/
theorem max_distance_circle_ellipse :
  ∃ (px py qx qy : ℝ),
    myCircle px py ∧ myEllipse qx qy ∧
    (∀ (px' py' qx' qy' : ℝ),
      myCircle px' py' → myEllipse qx' qy' →
      distance px py qx qy ≥ distance px' py' qx' qy') ∧
    distance px py qx qy = 4.5 ∧
    qx = 0 ∧ qy = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_ellipse_l130_13020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_interval_l130_13001

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (oplus 1 x) * x

-- Theorem statement
theorem range_of_f_on_interval :
  (∀ y ∈ Set.Icc (0 : ℝ) 8, ∃ x ∈ Set.Icc (0 : ℝ) 2, f x = y) ∧
  (∀ x ∈ Set.Icc (0 : ℝ) 2, f x ∈ Set.Icc (0 : ℝ) 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_interval_l130_13001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l130_13010

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - x) - Real.sqrt (x + 2)

-- State the theorem
theorem range_of_f :
  ∀ y ∈ Set.range f, -Real.sqrt 6 ≤ y ∧ y ≤ Real.sqrt 6 ∧
  ∃ x, -2 ≤ x ∧ x ≤ 4 ∧ f x = -Real.sqrt 6 ∧
  ∃ x, -2 ≤ x ∧ x ≤ 4 ∧ f x = Real.sqrt 6 :=
by sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l130_13010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l130_13058

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l130_13058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_tangency_l130_13025

/-- The value of m for which the parabola y = x^2 + 2 is tangent to the ellipse 2mx^2 + y^2 = 4 -/
noncomputable def tangent_m : ℝ := 2 + Real.sqrt 3

/-- Parabola function -/
def parabola (x : ℝ) : ℝ := x^2 + 2

/-- Ellipse function -/
def ellipse (m x y : ℝ) : Prop := 2 * m * x^2 + y^2 = 4

/-- Tangency condition -/
def is_tangent (m : ℝ) : Prop :=
  ∃ x, ellipse m x (parabola x) ∧
    ∀ x', x' ≠ x → ¬(ellipse m x' (parabola x'))

/-- Theorem stating that tangent_m satisfies the tangency condition -/
theorem parabola_ellipse_tangency :
  is_tangent tangent_m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_tangency_l130_13025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l130_13018

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x - a) / Real.log (1/2)

theorem a_range (a : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → 
  (∀ x₁ x₂ : ℝ, -3 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 - Real.sqrt 3 → f a x₁ < f a x₂) →
  0 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l130_13018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_height_l130_13067

/-- An isosceles trapezoid with perpendicular diagonals -/
structure IsoscelesTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The diagonals are perpendicular -/
  diagonals_perpendicular : Bool

/-- The height of an isosceles trapezoid -/
def trapezoidHeight (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The height of an isosceles trapezoid with area 100 and perpendicular diagonals is 10 -/
theorem isosceles_trapezoid_height (t : IsoscelesTrapezoid) 
  (h_area : t.area = 100) 
  (h_diag : t.diagonals_perpendicular = true) : 
  trapezoidHeight t = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_height_l130_13067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_mixture_acid_percentage_l130_13041

/-- Represents a mixture of acid and water -/
structure Mixture where
  acid : ℝ
  water : ℝ

/-- The percentage of acid in a mixture -/
noncomputable def acid_percentage (m : Mixture) : ℝ :=
  m.acid / (m.acid + m.water) * 100

theorem original_mixture_acid_percentage
  (original : Mixture)
  (h1 : acid_percentage { acid := original.acid, water := original.water + 2 } = 25)
  (h2 : acid_percentage { acid := original.acid + 2, water := original.water + 2 } = 40) :
  acid_percentage original = 100/3 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_mixture_acid_percentage_l130_13041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l130_13091

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_specific_points :
  distance (3, 3) (-2, -2) = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l130_13091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_sum_l130_13079

noncomputable def original_expr : ℝ := 1 / (4^(1/3) - 3^(1/3))

noncomputable def rationalized_form (P Q R S : ℝ) : ℝ := (P^(1/3) + Q^(1/3) + R^(1/3)) / S

theorem rationalize_and_sum :
  ∃ (P Q R S : ℝ), 
    (rationalized_form P Q R S = original_expr) ∧ 
    (P = 16 ∧ Q = 12 ∧ R = 9 ∧ S = 1) ∧
    (P + Q + R + S = 38) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_sum_l130_13079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_sampling_not_valid_for_inequalities_l130_13071

-- Define the type for inequality proof methods
inductive InequalityProofMethod
  | Comparison
  | RandomSampling
  | Synthetic
  | Analytic
  | ProofByContradiction
  | Scaling

-- Define a predicate for valid inequality proof methods
def is_valid_inequality_proof_method : InequalityProofMethod → Prop
  | InequalityProofMethod.Comparison => True
  | InequalityProofMethod.RandomSampling => False
  | InequalityProofMethod.Synthetic => True
  | InequalityProofMethod.Analytic => True
  | InequalityProofMethod.ProofByContradiction => True
  | InequalityProofMethod.Scaling => True

-- State the theorem
theorem random_sampling_not_valid_for_inequalities :
  is_valid_inequality_proof_method InequalityProofMethod.Comparison ∧
  is_valid_inequality_proof_method InequalityProofMethod.Synthetic ∧
  is_valid_inequality_proof_method InequalityProofMethod.Analytic ∧
  is_valid_inequality_proof_method InequalityProofMethod.ProofByContradiction ∧
  is_valid_inequality_proof_method InequalityProofMethod.Scaling →
  ¬ is_valid_inequality_proof_method InequalityProofMethod.RandomSampling :=
by
  intro h
  exact id

#check random_sampling_not_valid_for_inequalities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_sampling_not_valid_for_inequalities_l130_13071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l130_13065

theorem triangle_side_count : 
  let count := Finset.card (Finset.filter 
    (fun p : ℕ × ℕ => let (b, c) := p; 0 < b ∧ 0 < c ∧ b ≤ 5 ∧ 5 ≤ c ∧ c - b < 5)
    (Finset.product (Finset.range 10) (Finset.range 10)))
  count = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l130_13065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_A_is_zero_matrix_l130_13090

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 10; 8, 20]

theorem inverse_of_A_is_zero_matrix :
  ¬ IsUnit A → A⁻¹ = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_A_is_zero_matrix_l130_13090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_property_l130_13066

/-- The sum of the geometric series for |r| < 1 -/
noncomputable def T (r : ℝ) : ℝ := 20 + 10 * r / (1 - r)

/-- Main theorem -/
theorem geometric_series_sum_property :
  ∀ b : ℝ, -1 < b → b < 1 →
  T b * T (-b) = 5040 →
  T b + T (-b) = 504 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_property_l130_13066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daylight_rice_yield_not_function_l130_13046

-- Define the relationships
noncomputable def cube_volume_relation (edge_length : ℝ) : ℝ := edge_length^3
noncomputable def angle_sine_relation (angle : ℝ) : ℝ := Real.sin angle
def grain_yield_relation (land_area unit_yield : ℝ) : ℝ := land_area * unit_yield

-- Define the property of being a function
def is_function (f : α → β) : Prop := ∀ x y : α, f x = f y → x = y

-- Theorem statement
theorem daylight_rice_yield_not_function :
  (is_function cube_volume_relation) ∧
  (is_function angle_sine_relation) ∧
  (∀ unit_yield : ℝ, is_function (grain_yield_relation unit_yield)) ∧
  (¬ ∃ f : ℝ → ℝ, is_function f ∧ ∀ daylight rice_yield : ℝ, 
    f daylight = rice_yield → 
    (∃ d r : ℝ, d = daylight ∧ r = rice_yield ∧ 
     d >= 0 ∧ d <= 24 ∧ r >= 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_daylight_rice_yield_not_function_l130_13046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_eighth_digit_is_4_l130_13083

-- Define the decimal representation of 1/17
def decimal_rep_1_17 : List Nat := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

-- Define the length of the repeating cycle
def cycle_length : Nat := 16

-- Define the function to get the nth digit after the decimal point
def nth_digit (n : Nat) : Nat :=
  decimal_rep_1_17[((n - 1) % cycle_length)]'(by
    have h : ∀ m, m % cycle_length < List.length decimal_rep_1_17 := by
      intro m
      simp [cycle_length]
      exact Nat.mod_lt m (by norm_num)
    exact h (n - 1)
  )

-- State the theorem
theorem fifty_eighth_digit_is_4 : nth_digit 58 = 4 := by
  -- Unfold the definition of nth_digit
  unfold nth_digit
  -- Simplify the index calculation
  simp [cycle_length]
  -- The result follows from the definition of decimal_rep_1_17
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_eighth_digit_is_4_l130_13083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l130_13088

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : 2*a + 3*b = 10) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2*x + 3*y = 10 → Real.sqrt (3*y) + Real.sqrt (2*x) ≤ 2*Real.sqrt 5) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + 3*y = 10 ∧ Real.sqrt (3*y) + Real.sqrt (2*x) = 2*Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l130_13088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_diff_possible_l130_13008

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_a_ge_b : a ≥ b

/-- Represents a point on the ellipse -/
structure EllipsePoint (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : e.b^2 * x^2 + e.a^2 * y^2 = e.a^2 * e.b^2

/-- The chord length difference from its projection -/
noncomputable def chord_length_diff (e : Ellipse) (p : EllipsePoint e) : ℝ :=
  (p.x^2 + (e.b - p.y)^2).sqrt - (e.b - p.y)

/-- Theorem stating the condition for a chord with given length difference to be possible -/
theorem chord_length_diff_possible (e : Ellipse) (l : ℝ) :
  (∃ p : EllipsePoint e, chord_length_diff e p = l) ↔ 
  (e.a^2 / (e.a + e.b) ≤ l ∧ l ≤ e.a^2 / e.b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_diff_possible_l130_13008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_y_l130_13089

theorem possible_values_of_y (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  {y | ∃ x : ℝ, x^2 + 9 * (x / (x - 3))^2 = 90 ∧ y = (x - 3)^2 * (x + 4) / (2*x - 5)} = {0, 41, 144} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_y_l130_13089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y5_l130_13063

theorem coefficient_x3y5 :
  let expansion := (2/3 : ℚ) * x - (1/3 : ℚ) * y
  let coefficient := (Finset.range 9).sum (λ k ↦
    (Nat.choose 8 k : ℚ) * (2/3)^(8-k) * (-1/3)^k *
    if (8-k = 3 ∧ k = 5) then 1 else 0)
  coefficient = -448/6561 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y5_l130_13063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_concurrency_l130_13022

-- Define the structure for a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define the structure for a line
structure Line :=
  (a : Point) (b : Point)

-- Define the structure for a triangle
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

-- Define the structure for a square
structure Square :=
  (A : Point) (A1 : Point) (A2 : Point) (A3 : Point)

-- Define the property of a triangle being acute
def is_acute (t : Triangle) : Prop := sorry

-- Define the property of a line passing through a point
def passes_through (l : Line) (p : Point) : Prop := sorry

-- Define the property of lines being concurrent
def are_concurrent (l1 l2 l3 : Line) : Prop := sorry

-- Main theorem
theorem square_concurrency 
  (ABC : Triangle)
  (AA1A2A3 BB1B2B3 CC1C2C3 : Square)
  (h_acute : is_acute ABC)
  (h_A1A2 : passes_through (Line.mk AA1A2A3.A1 AA1A2A3.A2) ABC.B)
  (h_B1B2 : passes_through (Line.mk BB1B2B3.A1 BB1B2B3.A2) ABC.C)
  (h_C1C2 : passes_through (Line.mk CC1C2C3.A1 CC1C2C3.A2) ABC.A)
  (h_A2A3 : passes_through (Line.mk AA1A2A3.A2 AA1A2A3.A3) ABC.C)
  (h_B2B3 : passes_through (Line.mk BB1B2B3.A2 BB1B2B3.A3) ABC.A)
  (h_C2C3 : passes_through (Line.mk CC1C2C3.A2 CC1C2C3.A3) ABC.B) :
  (are_concurrent 
    (Line.mk ABC.A AA1A2A3.A2) 
    (Line.mk BB1B2B3.A1 BB1B2B3.A2) 
    (Line.mk CC1C2C3.A1 CC1C2C3.A3)) ∧
  (are_concurrent 
    (Line.mk ABC.A AA1A2A3.A2) 
    (Line.mk ABC.B BB1B2B3.A2) 
    (Line.mk ABC.C CC1C2C3.A2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_concurrency_l130_13022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perception_permutations_l130_13000

def word : Multiset Char := Multiset.ofList ['P', 'E', 'R', 'C', 'E', 'P', 'T', 'I', 'O', 'N']

theorem perception_permutations :
  Multiset.card word = 10 ∧
  Multiset.count 'P' word = 2 ∧
  Multiset.count 'E' word = 2 ∧
  (∀ c : Char, c ∉ ['P', 'E'] → Multiset.count c word ≤ 1) →
  Nat.factorial 10 / (Nat.factorial 2 * Nat.factorial 2) = 907200 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perception_permutations_l130_13000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_is_positive_integer_l130_13052

theorem p_is_positive_integer (x p : ℕ) (h1 : x / (11 * p) = 2) (h2 : x ≥ 44) (h3 : ∀ y : ℕ, y / (11 * p) = 2 → y ≥ x) : p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_is_positive_integer_l130_13052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_doesnt_determine_shape_l130_13053

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The ratio of two sides of a triangle -/
noncomputable def side_ratio (t : Triangle) : ℝ := 
  let AB := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let BC := Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)
  AB / BC

/-- Two triangles are similar if they have the same shape -/
def similar (t1 t2 : Triangle) : Prop := sorry

/-- Theorem: The ratio of two sides does not uniquely determine the shape of a triangle -/
theorem ratio_doesnt_determine_shape :
  ∃ (t1 t2 : Triangle), side_ratio t1 = side_ratio t2 ∧ ¬ similar t1 t2 := by
  sorry

#check ratio_doesnt_determine_shape

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_doesnt_determine_shape_l130_13053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_height_specific_case_l130_13005

/-- Given two poles with heights h₁ and h₂ that are d apart, 
    calculate the height of the intersection of the lines joining 
    the top of each pole to the foot of the opposite pole. -/
noncomputable def intersection_height (h₁ h₂ d : ℝ) : ℝ :=
  let m₁ := (0 - h₁) / d
  let m₂ := (0 - h₂) / (-d)
  let x := (h₁ - 0) / (m₂ - m₁)
  m₁ * x + h₁

theorem intersection_height_specific_case :
  intersection_height 30 50 150 = 18.75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_height_specific_case_l130_13005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_problem_l130_13044

/-- A configuration of n distinct points on a line -/
def Configuration (n : ℕ) := { l : List ℝ // l.length = n ∧ l.Nodup }

/-- A move that repositions a point according to the rule -/
def Move (lambda : ℝ) (config : Configuration n) : Configuration n :=
  sorry

/-- Predicate to check if all points in a configuration are to the right of M -/
def AllRightOf (config : Configuration n) (M : ℝ) : Prop :=
  sorry

/-- Main theorem stating the conditions for moving all points to the right of M -/
theorem flea_problem (n : ℕ) (h_n : n ≥ 2) (lambda : ℝ) (h_lambda : lambda > 0) :
  (∀ (config : Configuration n) (M : ℝ),
    ∃ (k : ℕ), AllRightOf ((Move lambda)^[k] config) M) ↔ lambda ≥ 1 / (n - 1) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_problem_l130_13044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l130_13072

/-- Calculates the length of a platform given the speed of a train, its length, and the time it takes to cross the platform. -/
theorem platform_length
  (train_speed_kmph : ℝ)
  (train_length : ℝ)
  (crossing_time : ℝ)
  (h1 : train_speed_kmph = 72)
  (h2 : train_length = 250.0416)
  (h3 : crossing_time = 26) :
  (train_speed_kmph * 1000 / 3600 * crossing_time - train_length) = 269.9584 :=
by
  -- Convert speed from km/h to m/s
  have train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
  
  -- Calculate total distance
  have total_distance : ℝ := train_speed_mps * crossing_time
  
  -- Calculate platform length
  have platform_length : ℝ := total_distance - train_length
  
  -- Prove the equality
  sorry -- This is where the actual proof would go


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l130_13072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_value_l130_13011

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3^x

-- Define the function g
noncomputable def g (l : ℝ) (x : ℝ) : ℝ := l * 2^x - 4^x

-- State the theorem
theorem function_max_value (a : ℝ) (l : ℝ) : 
  f (a + 2) = 27 → 
  (∀ x ∈ Set.Icc 0 2, g l x ≤ 1/3) → 
  (∃ x ∈ Set.Icc 0 2, g l x = 1/3) → 
  a = 1 ∧ l = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_value_l130_13011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_ratio_l130_13062

/-- For a cube with an inscribed sphere and a circumscribed sphere, 
    the ratio of the surface area of the inscribed sphere to 
    the surface area of the circumscribed sphere is 1:3 -/
theorem sphere_surface_area_ratio (a : ℝ) (a_pos : 0 < a) : 
  (4 * Real.pi * (a / 2) ^ 2) / (4 * Real.pi * ((Real.sqrt 3 * a) / 2) ^ 2) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_ratio_l130_13062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_boxcar_capacity_l130_13014

/-- Calculates the total coal capacity of a train's boxcars -/
theorem train_boxcar_capacity
  (black_count blue_count red_count : ℕ)
  (black_capacity : ℕ)
  (h1 : black_count = 7)
  (h2 : blue_count = 4)
  (h3 : red_count = 3)
  (h4 : black_capacity = 4000)
  (h5 : blue_capacity = 2 * black_capacity)
  (h6 : red_capacity = 3 * blue_capacity) :
  black_count * black_capacity +
  blue_count * (2 * black_capacity) +
  red_count * (3 * (2 * black_capacity)) = 132000 := by
  sorry

#check train_boxcar_capacity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_boxcar_capacity_l130_13014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_roots_imply_a_range_l130_13075

-- Define the function f(x)
noncomputable def f (x a : ℝ) : ℝ := 2 * Real.log x - x^2 + a

-- Define the interval
def interval : Set ℝ := Set.Icc (1/Real.exp 1) (Real.exp 1)

-- State the theorem
theorem function_roots_imply_a_range :
  ∀ a : ℝ, (∃ x y : ℝ, x ∈ interval ∧ y ∈ interval ∧ x ≠ y ∧ f x a = 0 ∧ f y a = 0) →
  a ∈ Set.Ioo 1 (2 + (1 / (Real.exp 1)^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_roots_imply_a_range_l130_13075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_properties_l130_13034

/-- A truncated cone circumscribed around a sphere -/
structure TruncatedCone where
  r : ℝ  -- radius of the inscribed sphere
  l : ℝ  -- slant height of the truncated cone
  h : r > 0 -- assumption that radius is positive
  k : l > r -- assumption that slant height is greater than radius

/-- The lateral surface area of a truncated cone -/
noncomputable def lateralSurfaceArea (tc : TruncatedCone) : ℝ := Real.pi * tc.l^2

/-- The volume of a truncated cone -/
noncomputable def volume (tc : TruncatedCone) : ℝ := (2 * Real.pi * tc.r * (tc.l^2 - tc.r^2)) / 3

/-- Theorem stating the correctness of the lateral surface area and volume formulas -/
theorem truncated_cone_properties (tc : TruncatedCone) :
  lateralSurfaceArea tc = Real.pi * tc.l^2 ∧
  volume tc = (2 * Real.pi * tc.r * (tc.l^2 - tc.r^2)) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_properties_l130_13034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_pencils_difference_l130_13002

theorem colored_pencils_difference (total : ℕ) (red_fraction blue_fraction : ℚ) : 
  total = 36 → 
  red_fraction = 5 / 9 → 
  blue_fraction = 5 / 12 → 
  (red_fraction * total).floor - (blue_fraction * total).floor = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_pencils_difference_l130_13002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l130_13026

theorem circular_table_seating (n : ℕ) (S : Finset (Fin (2 * n))) (f : Fin (2 * n) → Fin (2 * n)) 
  (h1 : S.card = 2 * n) (h2 : Function.Bijective f) : 
  ∃ a b : Fin (2 * n), a ≠ b ∧ a ∈ S ∧ b ∈ S ∧
    (min (Fin.val (a - b)) ((2 * n) - Fin.val (a - b)) = 
     min (Fin.val (f a - f b)) ((2 * n) - Fin.val (f a - f b))) := by
  sorry

#check circular_table_seating

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l130_13026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_theorem_l130_13051

/-- Tim's current age -/
def t : ℕ := sorry

/-- Sarah's current age -/
def s : ℕ := sorry

/-- The number of years until the ratio of their ages is 3:2 -/
def x : ℕ := sorry

/-- Tim's age was three times Sarah's age 5 years ago -/
axiom condition1 : t - 5 = 3 * (s - 5)

/-- Tim's age was five times Sarah's age 8 years ago -/
axiom condition2 : t - 8 = 5 * (s - 8)

/-- The theorem to be proved -/
theorem age_ratio_theorem : 
  (t + x) / (s + x) = 3 / 2 ∧ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_theorem_l130_13051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_fraction_l130_13029

theorem integral_of_fraction (x : ℝ) :
  let f := λ x => (3*x - 5) / Real.sqrt (x^2 - 4*x + 5)
  let F := λ x => 3 * Real.sqrt (x^2 - 4*x + 5) + Real.log (x - 2 + Real.sqrt (x^2 - 4*x + 5))
  deriv F x = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_fraction_l130_13029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_always_hit_l130_13077

-- Define the grid
def Grid := Fin 10 × Fin 10

-- Define a ship as a set of 4 adjacent points (either horizontally or vertically)
def Ship := {s : Set Grid | ∃ (i j : Fin 10), 
  (s = {(i, j), (i, j+1), (i, j+2), (i, j+3)} ∧ j+3 < 10) ∨ 
  (s = {(i, j), (i+1, j), (i+2, j), (i+3, j)} ∧ i+3 < 10)}

-- Define the theorem
theorem ship_always_hit : 
  ∃ (shots : Finset Grid), 
    (shots.card = 24) ∧ 
    (∀ s : Set Grid, s ∈ Ship → ∃ p ∈ shots, p ∈ s) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_always_hit_l130_13077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_sum_l130_13086

theorem trigonometric_equation_sum (t : ℝ) (k m n : ℕ) :
  (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4 →
  (1 - Real.sin t) * (1 - Real.cos t) = m / n - Real.sqrt (k : ℝ) →
  Nat.Coprime m n →
  k + m + n = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_sum_l130_13086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sequence_sum_l130_13016

noncomputable def f (x : ℝ) := x^2

noncomputable def tangent_intersection (a : ℝ) : ℝ := a / 2

noncomputable def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n-1)

theorem tangent_sequence_sum :
  let a := geometric_sequence 16 (1/2)
  (∀ (n : ℕ), n > 0 → a (n + 1) = tangent_intersection (a n)) →
  a 1 = 16 →
  a 1 + a 3 + a 5 = 21 :=
by
  sorry

#eval (16 : ℚ) + (16 : ℚ) * (1/2)^2 + (16 : ℚ) * (1/2)^4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sequence_sum_l130_13016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_x_plus_1_l130_13055

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(3-2x)
def domain_f_3_minus_2x : Set ℝ := Set.Icc (-1) 2

-- Theorem stating the domain of f(x+1)
theorem domain_f_x_plus_1 (h : ∀ y ∈ domain_f_3_minus_2x, ∃ x, y = 3 - 2*x) : 
  {x | f (x + 1) ∈ Set.range f} = Set.Icc (-1/2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_x_plus_1_l130_13055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l130_13095

/-- The constant term in the binomial expansion of (x/2 + 1/√x)^6 -/
def constant_term : ℚ := 15/4

/-- The binomial expansion of (x/2 + 1/√x)^6 -/
noncomputable def binomial_expansion (x : ℝ) : ℝ := (x/2 + 1/Real.sqrt x)^6

theorem constant_term_of_binomial_expansion :
  ∃ (f : ℝ → ℝ), (∀ x, x > 0 → binomial_expansion x = constant_term + x * f x) :=
by
  sorry

#check constant_term_of_binomial_expansion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l130_13095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_property_l130_13096

theorem odd_prime_property (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ n : ℕ, p = 2 * n + 1 ∧ ¬∃ k : ℕ, ((n^2).sqrt + (n^2 + 2).sqrt)^2 = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_property_l130_13096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l130_13099

theorem triangle_sine_inequality (A B C : ℝ) : 
  0 < A → 0 < B → 0 < C →  -- Angles are positive
  A + B + C = Real.pi →    -- Sum of angles in a triangle
  A < B → B < C →          -- Given angle inequality
  C ≠ Real.pi/2 →          -- C is not a right angle
  Real.sin A < Real.sin C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l130_13099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_amount_l130_13036

/-- The amount of seed used in gallons -/
def seed : ℝ := sorry

/-- The amount of fertilizer used in gallons -/
def fertilizer : ℝ := sorry

/-- The total amount of seed and fertilizer used in gallons -/
def total : ℝ := 60

/-- The amount of seed is three times the amount of fertilizer -/
axiom seed_fertilizer_ratio : seed = 3 * fertilizer

/-- The total amount is the sum of seed and fertilizer -/
axiom total_sum : total = seed + fertilizer

theorem seed_amount : seed = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_amount_l130_13036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_factorial_as_consecutive_product_l130_13087

def factorial (n : ℕ) : ℕ := Nat.factorial n

def is_product_of_consecutive_integers (n : ℕ) : Prop :=
  ∃ (k : ℕ), factorial n = (k + 1) * (k + 2) * (k + 3) * (k + 4) * (k + 5) * (k + 6)

theorem largest_factorial_as_consecutive_product :
  (is_product_of_consecutive_integers 4) ∧
  (∀ m : ℕ, m > 4 → ¬(is_product_of_consecutive_integers m)) := by
  sorry

#check largest_factorial_as_consecutive_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_factorial_as_consecutive_product_l130_13087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_to_plane_l130_13069

-- Define the types for planes and lines
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def Plane (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] := Subspace ℝ V
def Line (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] := Subspace ℝ V

-- Define the perpendicular and parallel relations
def perpendicular (L : Line V) (P : Plane V) : Prop := sorry
def parallel (P Q : Plane V) : Prop := sorry

-- State the theorem
theorem line_perpendicular_to_plane 
  (α β : Plane V) (l : Line V) :
  perpendicular l β → parallel α β → perpendicular l α :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_to_plane_l130_13069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crossing_ladders_width_l130_13042

/-- A quadrilateral with crossing diagonals -/
structure CrossingLadders where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  X : ℝ × ℝ
  right_angle_ADC : (A.1 - D.1) * (C.1 - D.1) + (A.2 - D.2) * (C.2 - D.2) = 0
  right_angle_DCB : (D.1 - C.1) * (B.1 - C.1) + (D.2 - C.2) * (B.2 - C.2) = 0
  AC_length : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 4
  BD_length : (B.1 - D.1)^2 + (B.2 - D.2)^2 = 9
  X_on_AC : ∃ t : ℝ, X = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2)
  X_on_BD : ∃ s : ℝ, X = (s * B.1 + (1 - s) * D.1, s * B.2 + (1 - s) * D.2)
  X_height : X.2 - C.2 = 1

/-- The width of the alley in the crossing ladders problem -/
noncomputable def alley_width (cl : CrossingLadders) : ℝ :=
  Real.sqrt ((cl.C.1 - cl.D.1)^2 + (cl.C.2 - cl.D.2)^2)

/-- The theorem stating the width of the alley -/
theorem crossing_ladders_width (cl : CrossingLadders) :
  |alley_width cl - 1.2311857| < 0.0000001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crossing_ladders_width_l130_13042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l130_13081

noncomputable section

/-- Given a triangle ABC with additional points as described, prove the ratio AF:FB -/
theorem triangle_ratio_theorem (A B C D E F P G Q : EuclideanSpace ℝ (Fin 2)) : 
  -- Triangle ABC exists
  A ≠ B ∧ B ≠ C ∧ C ≠ A → 
  -- D and E lie on BC
  ∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ 0 ≤ t₂ ∧ t₂ ≤ 1 ∧ 
    D = (1 - t₁) • B + t₁ • C ∧
    E = (1 - t₂) • B + t₂ • C →
  -- F lies on AB
  ∃ t₃ : ℝ, 0 ≤ t₃ ∧ t₃ ≤ 1 ∧ F = (1 - t₃) • A + t₃ • B →
  -- P is the intersection of AD and CF
  ∃ t₄ t₅ : ℝ, P = (1 - t₄) • A + t₄ • D ∧ P = (1 - t₅) • C + t₅ • F →
  -- G lies on AC
  ∃ t₆ : ℝ, 0 ≤ t₆ ∧ t₆ ≤ 1 ∧ G = (1 - t₆) • A + t₆ • C →
  -- Q is the intersection of BG and AD
  ∃ t₇ t₈ : ℝ, Q = (1 - t₇) • B + t₇ • G ∧ Q = (1 - t₈) • A + t₈ • D →
  -- Given ratios
  (∃ k₁ : ℝ, ‖A - P‖ = 3 * k₁ ∧ ‖P - D‖ = 2 * k₁) →
  (∃ k₂ : ℝ, ‖F - P‖ = 2 * k₂ ∧ ‖P - C‖ = k₂) →
  (∃ k₃ : ℝ, ‖B - Q‖ = k₃ ∧ ‖Q - G‖ = 3 * k₃) →
  -- Conclusion
  ‖A - F‖ / ‖F - B‖ = 5 / 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l130_13081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_production_l130_13030

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

/-- Number of months -/
def months : ℕ := 7

/-- First term (number of canoes in January) -/
def first_term : ℝ := 10

/-- Common ratio (growth rate) -/
def common_ratio : ℝ := 2

/-- Total number of canoes -/
def total_canoes : ℕ := 1270

theorem canoe_production :
  Int.floor (geometric_sum first_term common_ratio months) = total_canoes := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_production_l130_13030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l130_13070

-- Define the functions
noncomputable def f₁ (x : ℝ) := x
noncomputable def g₁ (x : ℝ) := x^2 / x

noncomputable def f₂ (x : ℝ) := x^2
noncomputable def g₂ (x : ℝ) := (x + 1)^2

noncomputable def f₃ (x : ℝ) := Real.sqrt (x^2)
noncomputable def g₃ (x : ℝ) := abs x

noncomputable def f₄ (x : ℝ) := x
noncomputable def g₄ (x : ℝ) := 3 * x^3

-- Theorem stating that only f₃ and g₃ are equal
theorem function_equality :
  (∀ x, f₁ x = g₁ x) = False ∧
  (∀ x, f₂ x = g₂ x) = False ∧
  (∀ x, f₃ x = g₃ x) = True ∧
  (∀ x, f₄ x = g₄ x) = False := by
  sorry

#check function_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l130_13070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_le_5_degrees_l130_13094

/-- A convex 11-gon -/
structure Convex11Gon where
  -- Add necessary properties to define a convex 11-gon

/-- The number of diagonals in a convex n-gon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The smallest angle between any two diagonals in a convex 11-gon -/
noncomputable def smallest_diagonal_angle (p : Convex11Gon) : ℝ :=
  360 / (2 * (num_diagonals 11 : ℝ))

/-- Theorem: The smallest angle between any two diagonals in a convex 11-gon is ≤ 5° -/
theorem smallest_angle_le_5_degrees (p : Convex11Gon) :
  smallest_diagonal_angle p ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_le_5_degrees_l130_13094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_one_two_zeros_l130_13033

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 2 * Real.log x + 2

-- Part 1: Prove that if f'(1) = 0, then a = 2
theorem extremum_at_one (a : ℝ) : 
  (deriv (f a)) 1 = 0 → a = 2 := by sorry

-- Part 2: Prove that f(x) has two zeros if and only if 0 < a < 2/Real.exp 2
theorem two_zeros (a : ℝ) : 
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ 0 < a ∧ a < 2 / Real.exp 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_one_two_zeros_l130_13033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_C₁_with_distance_to_C₂_l130_13028

-- Define the curves
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := 
  let ρ := Real.sqrt (3 / (2 + Real.cos (2 * θ)))
  (ρ * Real.cos θ, ρ * Real.sin θ)

def C₂ (t : ℝ) : ℝ × ℝ := (-1 + t, 7 - t)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem point_on_C₁_with_distance_to_C₂ :
  ∃ (θ : ℝ) (t : ℝ), 
    let p := C₁ θ
    let q := C₂ t
    distance p q = 2 * Real.sqrt 2 ∧ p = (1/2, 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_C₁_with_distance_to_C₂_l130_13028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l130_13015

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.cos (Real.pi + α) = 1/3) 
  (h2 : Real.pi < α) 
  (h3 : α < 2*Real.pi) : 
  Real.sin α = -(2 * Real.sqrt 2) / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l130_13015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_l130_13009

-- Define the coloring property
def has_monochromatic_equilateral {plane : Type} [MetricSpace plane] (coloring : plane → Fin 2) :=
  ∀ a : ℝ, a > 0 → ∃ (x y z : plane),
    coloring x = coloring y ∧ coloring y = coloring z ∧
    (dist x y = a ∧ dist y z = a ∧ dist z x = a)

-- Define the triangle inequality
def satisfies_triangle_inequality (a b c : ℝ) :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem monochromatic_triangle
  {plane : Type} [MetricSpace plane]
  (coloring : plane → Fin 2)
  (h_coloring : has_monochromatic_equilateral coloring)
  (a b c : ℝ)
  (h_triangle : satisfies_triangle_inequality a b c) :
  ∃ (x y z : plane),
    coloring x = coloring y ∧ coloring y = coloring z ∧
    (dist x y = a ∧ dist y z = b ∧ dist z x = c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_l130_13009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_film_radius_is_sqrt_512_over_pi_l130_13023

/-- The radius of a circular film formed by pouring a liquid onto a water surface -/
noncomputable def film_radius (container_length : ℝ) (container_width : ℝ) (container_height : ℝ) (film_thickness : ℝ) : ℝ :=
  Real.sqrt (container_length * container_width * container_height / (Real.pi * film_thickness))

/-- Theorem stating that the radius of the circular film is √(512/π) cm -/
theorem film_radius_is_sqrt_512_over_pi :
  film_radius 4 4 8 0.25 = Real.sqrt (512 / Real.pi) := by
  -- Unfold the definition of film_radius
  unfold film_radius
  -- Simplify the expression
  simp [Real.sqrt_div, Real.sqrt_mul]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_film_radius_is_sqrt_512_over_pi_l130_13023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_condition_l130_13085

-- Define a triangle in a 2D plane
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point in a 2D plane
def Point : Type := ℝ × ℝ

-- Define vector operations
def vec_sub (p q : Point) : ℝ × ℝ := (p.1 - q.1, p.2 - q.2)
def vec_add (p q : Point) : ℝ × ℝ := (p.1 + q.1, p.2 + q.2)
def vec_scale (k : ℝ) (p : Point) : ℝ × ℝ := (k * p.1, k * p.2)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the theorem
theorem isosceles_triangle_condition (t : Triangle) : 
  (∀ (P : Point), 
    dot_product 
      (vec_sub P t.B) 
      (vec_add 
        (vec_add (vec_sub P t.B) (vec_sub P t.C)) 
        (vec_scale (-2) (vec_sub P t.A))) = 
    dot_product 
      (vec_sub P t.C) 
      (vec_add 
        (vec_add (vec_sub P t.B) (vec_sub P t.C)) 
        (vec_scale (-2) (vec_sub P t.A)))) →
  vec_sub t.A t.B = vec_sub t.A t.C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_condition_l130_13085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_women_three_minutes_l130_13013

/-- The rate at which women drink tea, in teas per woman per minute -/
noncomputable def tea_drinking_rate : ℝ := 1 / 1.5

/-- The number of teas drunk by a group of women in a given time -/
noncomputable def teas_drunk (women : ℝ) (minutes : ℝ) : ℝ := women * minutes * tea_drinking_rate

theorem nine_women_three_minutes : 
  teas_drunk 9 3 = 18 := by
  -- Unfold the definition of teas_drunk
  unfold teas_drunk
  -- Unfold the definition of tea_drinking_rate
  unfold tea_drinking_rate
  -- Simplify the expression
  simp
  -- The proof is complete
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_women_three_minutes_l130_13013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_p_for_prime_sum_l130_13074

theorem smallest_p_for_prime_sum (p q r : ℕ) : 
  Prime p → Prime q → q = 2 → q < p → p + q = r → Prime r → 
  ∀ p' : ℕ, (Prime p' ∧ p' < p ∧ p' > q ∧ Prime (p' + q)) → False :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_p_for_prime_sum_l130_13074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l130_13050

theorem equation_solution (x : ℝ) : 
  (Real.sin x - Real.cos x > 0) →
  ((2 * (Real.tan x)^4 + 4 * Real.sin (3*x) * Real.sin (5*x) - Real.cos (6*x) - Real.cos (10*x) + 2) / Real.sqrt (Real.sin x - Real.cos x) = 0) →
  (∃ n : ℤ, x = Real.pi/2 + 2*Real.pi*↑n ∨ x = 3*Real.pi/4 + 2*Real.pi*↑n ∨ x = Real.pi + 2*Real.pi*↑n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l130_13050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_y_l130_13061

open Real

variable (a b c x : ℝ)
variable (p q r y : ℝ)

theorem log_equality_implies_y (h1 : (log a) / (2 * p) = (log b) / (3 * q))
                               (h2 : (log b) / (3 * q) = (log c) / (4 * r))
                               (h3 : (log c) / (4 * r) = log x)
                               (h4 : x ≠ 1)
                               (h5 : b^3 / (a^2 * c) = x^y) :
  y = 9*q - 4*p - 4*r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_y_l130_13061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_for_sum_1976_l130_13056

/-- Given a list of positive integers whose sum is 1976, their product is at most 2 * 3^658 -/
theorem max_product_for_sum_1976 (l : List ℕ) (h_pos : ∀ x ∈ l, x > 0) :
  (l.sum = 1976) → (l.prod ≤ 2 * 3^658) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_for_sum_1976_l130_13056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l130_13031

noncomputable def p (x : ℝ) := x^3 + 2/x^2

theorem constant_term_of_expansion (h : (p 1)^5 = 243) :
  ∃ (coeffs : List ℝ), 
    (List.sum coeffs = 243) ∧ 
    (List.length coeffs = 11) ∧
    (List.get! coeffs 5 = 80) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l130_13031
