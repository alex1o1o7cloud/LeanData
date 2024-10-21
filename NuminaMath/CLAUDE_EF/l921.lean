import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l921_92191

theorem inequality_solution_set (x : ℝ) :
  x ∈ {x : ℝ | |3 - x| + |x - 7| ≤ 8} ↔ x ∈ Set.Icc 1 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l921_92191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_theorem_l921_92147

-- Define a right triangle ABC
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  right_angle : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0  -- Dot product of AC and BC is 0

-- Define the lengths of the sides
noncomputable def side_length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the Pythagorean theorem
theorem pythagorean_theorem (triangle : RightTriangle) :
  (side_length triangle.A triangle.C)^2 + (side_length triangle.B triangle.C)^2 =
  (side_length triangle.A triangle.B)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_theorem_l921_92147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_logarithms_and_power_l921_92137

theorem ordering_of_logarithms_and_power (a b c : ℝ) : 
  a = Real.log 0.3 → b = (Real.log 0.5) / (Real.log 0.3) → c = 5^(0.3 : ℝ) → a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_logarithms_and_power_l921_92137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l921_92177

-- Define the Gauss function (floor function)
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^(x+1) / (1 + 2^(2*x))

-- Define the composite function g
noncomputable def g (x : ℝ) : ℤ := floor (f x)

-- Theorem statement
theorem range_of_g : Set.range g = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l921_92177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_height_l921_92103

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  leg : ℝ

/-- The height of an isosceles trapezoid -/
noncomputable def trapezoidHeight (t : IsoscelesTrapezoid) : ℝ :=
  Real.sqrt (t.leg ^ 2 - ((t.base2 - t.base1) / 2) ^ 2)

/-- Theorem: The height of the specific isosceles trapezoid is 24 -/
theorem specific_trapezoid_height :
  let t : IsoscelesTrapezoid := { base1 := 10, base2 := 24, leg := 25 }
  trapezoidHeight t = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_height_l921_92103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_to_line_l921_92114

/-- The distance from a point (x, y) to the line ax + by + c = 0 is |ax + by + c| / √(a² + b²) -/
noncomputable def distancePointToLine (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- Given point A(m, 2) is at a distance of √2 from the line x - y + 3 = 0, then m equals 1 or -3 -/
theorem point_distance_to_line (m : ℝ) : 
  distancePointToLine m 2 1 (-1) 3 = Real.sqrt 2 → m = 1 ∨ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_to_line_l921_92114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_one_max_min_l921_92172

/-- The function f(x) with parameter ω -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) * (Real.sin (ω * x) + Real.cos (ω * x)) - 1/2

/-- Theorem stating the range of ω for which f(x) has exactly one maximum and one minimum on (0, π) -/
theorem omega_range_for_one_max_min (ω : ℝ) :
  ω > 0 →
  (∃! xmax : ℝ, xmax ∈ Set.Ioo 0 π ∧ ∀ x ∈ Set.Ioo 0 π, f ω x ≤ f ω xmax) →
  (∃! xmin : ℝ, xmin ∈ Set.Ioo 0 π ∧ ∀ x ∈ Set.Ioo 0 π, f ω x ≥ f ω xmin) →
  7/8 < ω ∧ ω ≤ 11/8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_one_max_min_l921_92172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_island_challenge_probability_l921_92146

/-- The number of contestants in the game show -/
def total_contestants : ℕ := 20

/-- The number of tribes -/
def num_tribes : ℕ := 2

/-- The number of contestants who quit the show -/
def num_quitters : ℕ := 3

/-- The number of contestants in each tribe -/
def contestants_per_tribe : ℕ := total_contestants / num_tribes

/-- The probability that all quitters come from the same tribe -/
def prob_same_tribe : ℚ := 20 / 95

theorem island_challenge_probability :
  (Nat.choose total_contestants num_quitters) ≠ 0 →
  (2 * (Nat.choose contestants_per_tribe num_quitters) : ℚ) / (Nat.choose total_contestants num_quitters) = prob_same_tribe :=
by
  sorry

#eval total_contestants
#eval num_tribes
#eval num_quitters
#eval contestants_per_tribe
#eval prob_same_tribe

end NUMINAMATH_CALUDE_ERRORFEEDBACK_island_challenge_probability_l921_92146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l921_92129

-- Define the hyperbola
noncomputable def hyperbola (x y b : ℝ) : Prop := x^2 / 4 - y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (b : ℝ) : ℝ := (Real.sqrt 3 / 3) * b

-- Define the focal length
def focal_length (c : ℝ) : ℝ := 2 * c

-- Theorem statement
theorem hyperbola_focal_length (b : ℝ) (h1 : b > 0) :
  ∃ (x y : ℝ), hyperbola x y b ∧ eccentricity b > 0 → focal_length 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l921_92129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l921_92160

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3) + 1

theorem f_properties :
  (∀ x : ℝ, f (2 * Real.pi / 3 + x) = f (2 * Real.pi / 3 - x)) ∧
  (∀ ε > 0, ∃ T : ℝ, T > 0 ∧ T ≤ Real.pi ∧ ∀ x : ℝ, f (x + T) = f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l921_92160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vitali_covering_theorem_l921_92163

-- Define a disk in the plane
structure Disk :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the scaled disk (3D)
def scaledDisk (d : Disk) : Disk :=
  { center := d.center, radius := 3 * d.radius }

-- Define the theorem
theorem vitali_covering_theorem 
  (X : Set (ℝ × ℝ)) 
  (diskCollection : Finset Disk) 
  (covers : X ⊆ ⋃ d ∈ diskCollection, {p | dist p d.center ≤ d.radius}) :
  ∃ (S : Finset Disk), 
    S ⊆ diskCollection ∧ 
    (∀ d₁ d₂, d₁ ∈ S → d₂ ∈ S → d₁ ≠ d₂ → dist d₁.center d₂.center > d₁.radius + d₂.radius) ∧
    X ⊆ ⋃ d ∈ S, {p | dist p d.center ≤ (scaledDisk d).radius} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vitali_covering_theorem_l921_92163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l921_92175

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 7*x^2 - 4*x + 2) / (x^3 - 4*x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | x < -2 ∨ (-2 < x ∧ x < 0) ∨ (0 < x ∧ x < 2) ∨ 2 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l921_92175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_mult_expression_l921_92195

/-- Custom multiplication operation for nonzero real numbers -/
noncomputable def custom_mult (a b : ℝ) : ℝ := b^2 / a

notation a " ⊗ " b => custom_mult a b

/-- Theorem stating the result of the given expression -/
theorem custom_mult_expression :
  ∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 →
  ((custom_mult (custom_mult a b) c) - (custom_mult (custom_mult a c) b)) = 1295/12 :=
by
  intros a b c ha hb hc
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check custom_mult_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_mult_expression_l921_92195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_x_l921_92108

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.sin (x / 7)

theorem smallest_max_x : 
  ∃ (x : ℝ), x > 0 ∧ x = 10080 ∧ 
  (∀ (y : ℝ), y > 0 → f y ≤ f x) ∧
  (∀ (z : ℝ), 0 < z ∧ z < x → f z < f x) := by
  sorry

#check smallest_max_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_x_l921_92108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l921_92111

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Evaluate a polynomial at a given point -/
def eval (p : IntPolynomial) (x : ℕ) : ℤ :=
  p x

/-- Check if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop :=
  ∃ m : ℤ, n = m * m

/-- The property that p(a) + p(b) is a perfect square when a + b is a perfect square -/
def has_square_sum_property (p : IntPolynomial) : Prop :=
  ∀ a b : ℕ, is_perfect_square (↑(a + b)) → is_perfect_square (eval p a + eval p b)

/-- The form of the polynomial: k^2 * x or 2 * u^2 -/
def is_valid_form (p : IntPolynomial) : Prop :=
  (∃ k : ℤ, ∀ x : ℕ, eval p x = k * k * ↑x) ∨
  (∃ u : ℤ, ∀ x : ℕ, eval p x = 2 * u * u)

theorem polynomial_characterization (p : IntPolynomial) :
  has_square_sum_property p → is_valid_form p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l921_92111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_problem_l921_92151

theorem factorial_problem (m n : ℕ) (h : Nat.factorial (m + n) / Nat.factorial n = 5040) : 
  Nat.factorial m * n = 144 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_problem_l921_92151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_special_case_l921_92102

theorem right_triangle_special_case (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (right_triangle : a^2 + b^2 = c^2)
  (hypotenuse_condition : c^2 = 2*a*b) :
  ∃ θ : ℝ, θ = 45 * π / 180 ∧ (Real.sin θ = a / c ∨ Real.sin θ = b / c) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_special_case_l921_92102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubs_world_series_probability_l921_92166

noncomputable def probability_cubs_win_game : ℝ := 4/7

noncomputable def probability_red_sox_win_game : ℝ := 1 - probability_cubs_win_game

noncomputable def probability_cubs_win_series : ℝ :=
  (Nat.choose 3 0) * (probability_cubs_win_game ^ 4) * (probability_red_sox_win_game ^ 0) +
  (Nat.choose 4 1) * (probability_cubs_win_game ^ 4) * (probability_red_sox_win_game ^ 1) +
  (Nat.choose 5 2) * (probability_cubs_win_game ^ 4) * (probability_red_sox_win_game ^ 2) +
  (Nat.choose 6 3) * (probability_cubs_win_game ^ 4) * (probability_red_sox_win_game ^ 3)

theorem cubs_world_series_probability :
  ∃ ε > 0, |probability_cubs_win_series - 0.73617| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubs_world_series_probability_l921_92166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_bounds_l921_92107

noncomputable def mySequence (n : ℕ) : ℝ :=
  match n with
  | 0 => 1
  | n + 1 => mySequence n + (mySequence n) ^ (1 / (n + 1 : ℝ))

theorem mySequence_bounds (n : ℕ) (hn : n > 0) : 
  (n : ℝ) ≤ mySequence n ∧ mySequence n < 2 * n ∧ mySequence n ≤ n + 4 * Real.sqrt (n - 1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_bounds_l921_92107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_solutions_l921_92190

theorem sin_2x_solutions :
  let S := {x : ℝ | (∃ k : ℤ, x = π * k / 2 ∧ (k ≤ 1 ∨ k ≥ 5)) ∨ x = 7 * π / 4}
  let T := {x : ℝ | (x ≤ π / 2 ∨ x ≥ 2 * π) ∧ Real.sin (2 * x) = 0} ∪
           {x : ℝ | π < x ∧ x ≤ 2 * π ∧ Real.sin (2 * x) = -1}
  S = T :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_solutions_l921_92190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_14gon_inequality_l921_92122

theorem regular_14gon_inequality (a : ℝ) : 
  a > 0 → -- side length is positive
  2 * Real.sin (π / 14) = a → -- relation between side length and central angle
  (2 - a) / (2 * a) > Real.sqrt (3 * Real.cos (π / 7)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_14gon_inequality_l921_92122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_1_condition_2_condition_3_school_distance_l921_92135

/-- The distance between the student's house and school -/
noncomputable def distance : ℚ := 24

/-- The time it takes for the student to reach school on time -/
noncomputable def on_time : ℚ := 7/3

/-- Condition 1: At 9 kmph, the student reaches school 20 minutes late -/
theorem condition_1 : 9 * (on_time + 1/3) = distance := by sorry

/-- Condition 2: At 12 kmph, the student reaches school 20 minutes early -/
theorem condition_2 : 12 * (on_time - 1/3) = distance := by sorry

/-- Condition 3: At 15 kmph, the student reaches school 40 minutes early -/
theorem condition_3 : 15 * (on_time - 2/3) = distance := by sorry

/-- Theorem: Given the three conditions, the distance between the student's house and school is 24 kilometers -/
theorem school_distance : distance = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_1_condition_2_condition_3_school_distance_l921_92135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_at_distance_one_l921_92100

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line
def line_equation (x y : ℝ) (c : ℝ) : Prop := 12*x - 5*y + c = 0

-- Define the distance from a point to the line
noncomputable def distance_to_line (x y c : ℝ) : ℝ := 
  |12*x - 5*y + c| / Real.sqrt (12^2 + (-5)^2)

-- Define the condition for a point to be at distance 1 from the line
def at_distance_one (x y c : ℝ) : Prop := 
  distance_to_line x y c = 1

-- Theorem statement
theorem four_points_at_distance_one (c : ℝ) : 
  (∃! (p1 p2 p3 p4 : ℝ × ℝ), 
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    (∀ (x y : ℝ), circle_equation x y ∧ at_distance_one x y c ↔ 
      (x, y) = p1 ∨ (x, y) = p2 ∨ (x, y) = p3 ∨ (x, y) = p4))
  ↔ 
  c > -13 ∧ c < 13 := 
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_at_distance_one_l921_92100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_difference_is_zero_l921_92143

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the function f_n
noncomputable def f (n : ℕ) : ℝ :=
  (1 + 1 / (n : ℝ)) ^ (n : ℝ) * ((Nat.factorial (2 * n - 1) * fib n : ℝ)) ^ (1 / (n : ℝ))

-- State the theorem
theorem limit_f_difference_is_zero :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |f (n + 1) - f n| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_difference_is_zero_l921_92143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l921_92171

def A : Set Int := {-1, 0, 1}
def B : Set Int := {x : Int | x ∈ Set.range (Nat.cast : ℕ → ℤ) ∧ x < 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l921_92171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_number_l921_92154

theorem smallest_divisible_number (n m : ℕ) : 
  n ≥ 40 → n % 8 = 0 → n % m = 0 → m ∉ ({1, 2, 4, 8} : Set ℕ) → (∀ k, k < n → (k % 8 = 0 → k % m = 0 → False)) →
  n = 40 ∧ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_number_l921_92154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phirme_sequence_representation_l921_92196

/-- Fibonacci sequence -/
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Phirme sequence -/
def is_phirme (a : ℕ → ℤ) : Prop :=
  ∃ k : ℕ, ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = fib (n + k)

theorem phirme_sequence_representation (a : ℕ → ℤ) (h : is_phirme a) :
  ∃ k c : ℤ, ∀ n : ℕ, n ≥ 1 → a n = fib (Int.toNat (n + k - 2)) + (-1 : ℤ)^n * c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phirme_sequence_representation_l921_92196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arctan_equality_l921_92120

theorem sin_arctan_equality (x : ℝ) (h1 : x > 0) (h2 : Real.sin (Real.arctan x) = 1 / x) : 
  x^2 = (-1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arctan_equality_l921_92120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_propositions_l921_92184

theorem quadratic_equation_propositions :
  let a := 1
  let b := 3
  let c := -1
  let p := ∃ x y, x^2 + b*x + c = 0 ∧ y^2 + b*y + c = 0 ∧ x ≠ y ∧ x * y < 0
  let q := ∃ x y, x^2 + b*x + c = 0 ∧ y^2 + b*y + c = 0 ∧ x ≠ y ∧ x + y = -b
  2 = ([@Bool.false, @Bool.true, @Bool.false, @Bool.true].filter id).length :=
by
  -- The proof goes here
  sorry

#eval ([@Bool.false, @Bool.true, @Bool.false, @Bool.true].filter id).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_propositions_l921_92184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_l921_92152

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the line l in polar form
def line_l (ρ θ : ℝ) : Prop := 2 * ρ * Real.sin (θ + Real.pi/6) = 5 * Real.sqrt 3

-- Define the ray OM
def ray_OM (θ : ℝ) : Prop := θ = Real.pi/6

-- Define point P
def point_P (ρ θ : ℝ) : Prop := circle_C (ρ * Real.cos θ) (ρ * Real.sin θ) ∧ ray_OM θ

-- Define point Q
def point_Q (ρ θ : ℝ) : Prop := line_l ρ θ ∧ ray_OM θ

theorem length_PQ : ∃ (ρ₁ ρ₂ θ : ℝ), point_P ρ₁ θ ∧ point_Q ρ₂ θ ∧ ρ₂ - ρ₁ = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_l921_92152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_vertical_line_l921_92173

-- Define IsAngleOfSlope as it's not a built-in concept
def IsAngleOfSlope (α : ℝ) (m : ℝ) : Prop :=
  m = Real.tan α

theorem slope_angle_vertical_line : 
  ∀ (x : ℝ), (∀ y : ℝ, x = 1) → ∃ α : ℝ, α = π / 2 ∧ IsAngleOfSlope α x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_vertical_line_l921_92173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_solution_l921_92157

open Real Set

theorem sin_equation_solution (f : ℝ → ℝ) (θ : ℝ) : 
  (f = sin) →
  (θ = 5 * π / 13) ↔ 
  (∀ x₁ ∈ Icc 0 (π/2), ∃ x₂ ∈ Icc 0 (π/2), f x₁ - 2 * f (x₂ + θ) = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_solution_l921_92157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_revenue_solutions_l921_92185

/-- The number of solutions to the store revenue equation -/
def revenue_solutions : ℕ := 121

/-- A solution to the store revenue equation is a triple of positive integers (x, y, z) -/
def is_revenue_solution (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ 10 * x + 5 * y + z = 120

/-- The set of all solutions to the store revenue equation -/
def revenue_solution_set : Set (ℕ × ℕ × ℕ) :=
  {s | is_revenue_solution s.1 s.2.1 s.2.2}

/-- The set of all solutions is finite -/
instance : Fintype revenue_solution_set := by
  sorry

theorem count_revenue_solutions :
  Fintype.card revenue_solution_set = revenue_solutions := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_revenue_solutions_l921_92185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frank_game_points_l921_92136

theorem frank_game_points 
  (enemies_defeated : ℕ) 
  (points_per_enemy : ℕ) 
  (level_completion_points : ℕ) 
  (special_challenges : ℕ) 
  (points_per_challenge : ℕ) 
  (wrong_moves : ℕ) 
  (points_lost_per_wrong_move : ℕ)
  (time_bonus : ℕ) :
  enemies_defeated = 18 →
  points_per_enemy = 15 →
  level_completion_points = 25 →
  special_challenges = 7 →
  points_per_challenge = 12 →
  wrong_moves = 3 →
  points_lost_per_wrong_move = 10 →
  time_bonus = 50 →
  enemies_defeated * points_per_enemy + 
  level_completion_points + 
  special_challenges * points_per_challenge - 
  wrong_moves * points_lost_per_wrong_move + 
  time_bonus = 399 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frank_game_points_l921_92136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_condition_inequality_proof_l921_92133

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.log x - a)

-- Theorem for the monotonically increasing condition
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 4, Monotone (f a)) ↔ a ≤ 1 :=
by sorry

-- Theorem for the inequality
theorem inequality_proof (a : ℝ) (h : a > 0) :
  ∀ x > 0, f a x ≤ x * (x - 2 - Real.log a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_condition_inequality_proof_l921_92133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_minor_axis_angle_is_45_degrees_l921_92119

/-- An ellipse with semi-major axis a and focal distance c -/
structure Ellipse where
  a : ℝ
  c : ℝ
  h_pos_a : 0 < a
  h_pos_c : 0 < c
  h_c_lt_a : c < a

/-- The angle between the line connecting a focus to the endpoint of the minor axis and the major axis -/
noncomputable def focus_minor_axis_angle (e : Ellipse) : ℝ :=
  Real.arccos (e.c / e.a)

theorem focus_minor_axis_angle_is_45_degrees (e : Ellipse) :
  focus_minor_axis_angle e = π / 4 := by
  sorry

#check focus_minor_axis_angle_is_45_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_minor_axis_angle_is_45_degrees_l921_92119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hamburger_bun_probability_l921_92176

theorem hamburger_bun_probability : 
  let n : ℕ := 5  -- number of people
  let p : ℚ := (Finset.range n).prod (fun i => (n - i) / (2 * n - 2 * i - 1)) -- probability calculation
  p = 8 / 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hamburger_bun_probability_l921_92176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_ratio_l921_92132

theorem sphere_radius_ratio (V_large V_small : ℝ) (h1 : V_large = 216 * Real.pi) 
  (h2 : V_small = 0.2 * V_large) : 
  (V_small * 3 / (4 * Real.pi)) ^ (1/3) / (V_large * 3 / (4 * Real.pi)) ^ (1/3) = 1 / 5 ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_ratio_l921_92132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_sum_l921_92142

theorem polynomial_roots_sum (a b c d : ℝ) (m n : ℕ) : 
  (∃ (z : ℂ), z^3 + a*z + b = 0 ∧ z^3 + c*z^2 + d = 0) →
  (-20)^3 + a*(-20) + b = 0 →
  (-21)^3 + c*(-21)^2 + d = 0 →
  (m + Complex.I * Real.sqrt (n : ℝ) : ℂ)^3 + a*(m + Complex.I * Real.sqrt (n : ℝ) : ℂ) + b = 0 →
  (m + Complex.I * Real.sqrt (n : ℝ) : ℂ)^3 + c*(m + Complex.I * Real.sqrt (n : ℝ) : ℂ)^2 + d = 0 →
  m + n = 330 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_sum_l921_92142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l921_92149

-- Define the slope of line l
variable (k : ℝ)

-- Define the slope angle of line l
noncomputable def θ : ℝ := Real.arctan k

-- Theorem statement
theorem slope_angle_range :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = k * x - Real.sqrt 3 ∧ x + y = 3) →
  π / 6 < θ k ∧ θ k < π / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l921_92149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_is_id_l921_92198

/-- The function f(x) = (1-x)/(1+x) -/
noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

/-- The sequence of functions f_k -/
noncomputable def f_k : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ (f_k n)

/-- Theorem: f_{2016}(x) = x -/
theorem f_2016_is_id : f_k 2016 = id := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_is_id_l921_92198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_placement_count_l921_92113

/-- Represents a 2 × 100 board -/
def Board := Fin 2 → Fin 100 → Bool

/-- Checks if two cells are adjacent -/
def adjacent (c1 c2 : Fin 2 × Fin 100) : Prop :=
  (c1.1 = c2.1 ∧ (c1.2 = c2.2 + 1 ∨ c2.2 = c1.2 + 1)) ∨
  (c1.2 = c2.2 ∧ (c1.1 = c2.1 + 1 ∨ c2.1 = c1.1 + 1))

/-- Checks if a board configuration is valid -/
def valid_board (b : Board) : Prop :=
  (∀ i j, b i j → ∀ i' j', b i' j' → ¬(adjacent (i, j) (i', j'))) ∧
  (∀ i j, b i j → ∀ i' j', b i' j' → (i, j) ≠ (i', j'))

/-- Counts the number of coins on the board -/
def coin_count (b : Board) : Nat :=
  (Finset.sum Finset.univ fun i => (Finset.sum Finset.univ fun j => if b i j then 1 else 0))

/-- The main theorem stating the number of valid board configurations -/
theorem coin_placement_count :
  ∃ s : Finset Board, (∀ b ∈ s, valid_board b ∧ coin_count b = 99) ∧ s.card = 396 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_placement_count_l921_92113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clear_time_is_20_seconds_l921_92117

/-- The time for two trains to clear each other -/
noncomputable def train_clear_time (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / ((speed1 + speed2) * (1000 / 3600))

/-- Theorem: The time for two trains to clear each other is 20 seconds -/
theorem train_clear_time_is_20_seconds :
  train_clear_time 120 280 42 30 = 20 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clear_time_is_20_seconds_l921_92117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_plus_alpha_l921_92156

theorem cos_pi_third_plus_alpha (α : ℝ) (h1 : Real.sin α = 3/5) (h2 : 0 < α ∧ α < π/2) :
  Real.cos (π/3 + α) = (4 - 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_plus_alpha_l921_92156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_union_equality_l921_92118

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x > 0}
def N : Set ℝ := {x : ℝ | x^2 - 4 ≥ 0}

-- Define the union of intervals
def union_intervals : Set ℝ := Set.Ici (-2) ∪ Set.Ioi 0

-- State the theorem
theorem set_union_equality : M ∪ N = union_intervals := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_union_equality_l921_92118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_point_with_angle_l921_92174

/-- A circle centered at (-2, 0) with radius 1 -/
def my_circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

/-- The locus of points P(x, y) satisfying the tangent-to-distance ratio condition -/
def curve_C (ι : ℝ) (x y : ℝ) : Prop :=
  (1 - ι^2) * x^2 + y^2 + 4*x + 3 = 0

/-- The angle between lines from a point to the foci of the curve -/
noncomputable def focal_angle (F₁ F₂ Q : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the condition for no point Q on curve C to have a specific focal angle -/
theorem no_point_with_angle (θ : ℝ) (hθ : 0 < θ ∧ θ < π) :
  (∀ ι > 0, ι ≠ 1 →
    (∀ x y : ℝ, curve_C ι x y →
      ∀ F₁ F₂ : ℝ × ℝ, focal_angle F₁ F₂ (x, y) ≠ θ)) ↔
  (∀ ι > 0, ι < Real.sin (θ / 2)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_point_with_angle_l921_92174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_extreme_value_point_l921_92141

/-- A function f(x) with parameter a -/
noncomputable def f (a : ℕ) (x : ℝ) : ℝ := Real.log x + a / (x + 1)

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℕ) (x : ℝ) : ℝ := 1 / x - a / ((x + 1) ^ 2)

/-- Theorem stating that there is only one natural number a for which f(x) has exactly one extreme value in (1, 3) -/
theorem unique_extreme_value_point :
  ∃! a : ℕ, ∃! x : ℝ, x ∈ Set.Ioo 1 3 ∧ f_derivative a x = 0 := by
  sorry

#check unique_extreme_value_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_extreme_value_point_l921_92141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_intersection_points_iff_l921_92125

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ -2 ∨ x ≥ 1 then 0 else -x^2 - x + 2

def intersects_at_three_points (a b : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    a * x₁ + b = g x₁ ∧ a * x₂ + b = g x₂ ∧ a * x₃ + b = g x₃

theorem three_intersection_points_iff (a b : ℝ) :
  a > 0 → (intersects_at_three_points a b ↔ 
    0 < a ∧ a < 3 ∧ 2 * a < b ∧ b < (1/4) * (a + 1)^2 + 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_intersection_points_iff_l921_92125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_of_angle_between_tangents_l921_92140

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the intersection point
def intersection_point : ℝ × ℝ := (1, 3)

-- Define a tangent line to the circle
def is_tangent_line (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), my_circle x y ∧ (∀ (x' y' : ℝ), l x' y' → ((x' - x)^2 + (y' - y)^2 ≥ 1))

-- Theorem statement
theorem tangent_of_angle_between_tangents :
  ∀ (l₁ l₂ : ℝ → ℝ → Prop),
    is_tangent_line l₁ →
    is_tangent_line l₂ →
    l₁ (intersection_point.1) (intersection_point.2) →
    l₂ (intersection_point.1) (intersection_point.2) →
    ∃ (θ : ℝ), Real.tan θ = 4/3 ∧ 
      (∀ (x y : ℝ), (l₁ x y ∧ l₂ x y) → 
        ((y - intersection_point.2) * (x - intersection_point.1))^2 = 
        Real.tan θ * (x - intersection_point.1)^2 * (y - intersection_point.2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_of_angle_between_tangents_l921_92140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_range_l921_92130

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x / Real.sqrt (1 - x)

-- State the theorem
theorem meaningful_range (x : ℝ) : 
  (∃ y : ℝ, f x = y) ↔ x < 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_range_l921_92130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_common_tangents_l921_92104

open Real

/-- Two curves have exactly two common tangent lines if and only if the parameter a is less than 2ln(2)-3 -/
theorem two_common_tangents (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (∀ (x : ℝ), (∃ (y : ℝ), y = exp (x + a) ∧ y = (x - 1)^2) → 
      (x = x₁ ∨ x = x₂)) ∧
    (∀ (x : ℝ), HasDerivAt (exp) (exp (x + a)) (x + a) ∧ 
      HasDerivAt (fun x => (x - 1)^2) (2 * (x - 1)) x ∧
      exp (x + a) = 2 * (x - 1))) ↔ 
  a < 2 * log 2 - 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_common_tangents_l921_92104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l921_92112

-- Define the power function as noncomputable
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_through_point (α : ℝ) :
  power_function α 3 = Real.sqrt 3 → α = 1/2 := by
  intro h
  -- The proof steps would go here
  sorry

#check power_function_through_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l921_92112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_fiftieth_l921_92168

def y : ℕ → ℝ
  | 0 => 50  -- Add this case to cover Nat.zero
  | 1 => 50
  | k + 2 => y (k + 1) ^ 2 - 2 * y (k + 1)

noncomputable def series_sum : ℝ := ∑' n, 1 / (y n + 2)

theorem series_sum_equals_one_fiftieth : series_sum = 1 / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_fiftieth_l921_92168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_B_tin_copper_ratio_l921_92183

/-- Represents the composition of an alloy -/
structure Alloy where
  lead : ℚ
  tin : ℚ
  copper : ℚ

/-- The ratio of two components in an alloy -/
def ratio (a b : ℚ) : ℚ := a / b

theorem alloy_B_tin_copper_ratio 
  (alloy_A alloy_B : Alloy)
  (total_tin : ℚ)
  (h1 : alloy_A.lead + alloy_A.tin = 90)
  (h2 : ratio alloy_A.lead alloy_A.tin = 3/4)
  (h3 : alloy_B.tin + alloy_B.copper = 140)
  (h4 : alloy_A.tin + alloy_B.tin = total_tin)
  (h5 : total_tin = 91.42857142857143) : 
  ratio alloy_B.tin alloy_B.copper = 2/5 := by
  sorry

#eval (2 : ℚ) / 5  -- Expected output: 0.4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_B_tin_copper_ratio_l921_92183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l921_92101

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.sin x + 1) / Real.exp Real.pi - a / Real.exp x

-- State the theorem
theorem f_properties :
  (∀ x ∈ Set.Icc (-Real.pi) 0, f (-1) x > 1) ∧
  (∃! (z₁ z₂ : ℝ), z₁ ∈ Set.Icc Real.pi (2 * Real.pi) ∧
                   z₂ ∈ Set.Icc Real.pi (2 * Real.pi) ∧
                   z₁ ≠ z₂ ∧
                   f 1 z₁ = 0 ∧ f 1 z₂ = 0 ∧
                   ∀ z ∈ Set.Icc Real.pi (2 * Real.pi), f 1 z = 0 → z = z₁ ∨ z = z₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l921_92101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_tangent_lengths_l921_92144

/-- Two circles in a plane -/
structure TwoCircles where
  ω₁ : Set (ℝ × ℝ)
  ω₂ : Set (ℝ × ℝ)

/-- Intersection points of two circles -/
def intersectionPoints (c : TwoCircles) : Set (ℝ × ℝ) :=
  c.ω₁ ∩ c.ω₂

/-- A point on the line through intersection points but not between them -/
structure PointOnLine (c : TwoCircles) where
  X : ℝ × ℝ
  on_line : ∃ (t : ℝ) (A B : ℝ × ℝ), A ∈ intersectionPoints c ∧ B ∈ intersectionPoints c ∧ 
    X = ((1 - t) • A + t • B)
  not_between : ∀ (A B : ℝ × ℝ), A ∈ intersectionPoints c → B ∈ intersectionPoints c → 
    (X.1 < A.1 ∧ X.1 < B.1) ∨ (X.1 > A.1 ∧ X.1 > B.1)

/-- Length of a tangent from a point to a circle -/
noncomputable def tangentLength (X : ℝ × ℝ) (ω : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem equal_tangent_lengths (c : TwoCircles) (p : PointOnLine c) :
  tangentLength p.X c.ω₁ = tangentLength p.X c.ω₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_tangent_lengths_l921_92144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raisin_weight_approximation_l921_92193

/-- The proportion of water in grapes -/
def grape_water_proportion : ℚ := 9/10

/-- The proportion of water in raisins -/
def raisin_water_proportion : ℚ := 17/100

/-- The initial weight of grapes in kilograms -/
def initial_grape_weight : ℚ := 1162/10

/-- Calculate the final weight of raisins given the initial weight of grapes,
    the water proportion in grapes, and the water proportion in raisins -/
noncomputable def final_raisin_weight (initial_weight : ℚ) (grape_water : ℚ) (raisin_water : ℚ) : ℚ :=
  initial_weight * (1 - grape_water) / (1 - raisin_water)

/-- Theorem stating that the final weight of raisins is approximately 14 kilograms -/
theorem raisin_weight_approximation :
  ∃ ε > 0, |final_raisin_weight initial_grape_weight grape_water_proportion raisin_water_proportion - 14| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raisin_weight_approximation_l921_92193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_proof_l921_92106

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if a point lies on a plane -/
def pointOnPlane (pt : Point3D) (pl : Plane) : Prop :=
  pl.a * pt.x + pl.b * pt.y + pl.c * pt.z + pl.d = 0

/-- Check if two planes are perpendicular -/
def planesArePerpendicular (pl1 : Plane) (pl2 : Plane) : Prop :=
  pl1.a * pl2.a + pl1.b * pl2.b + pl1.c * pl2.c = 0

/-- The greatest common divisor of four integers -/
def gcd4 (a b c d : ℤ) : ℕ :=
  Nat.gcd (Nat.gcd (Nat.gcd (a.natAbs) (b.natAbs)) (c.natAbs)) (d.natAbs)

theorem plane_equation_proof (pl : Plane) : 
  pl.a = 6 ∧ pl.b = 6 ∧ pl.c = 1 ∧ pl.d = -13 →
  pointOnPlane ⟨2, 0, 1⟩ pl ∧
  pointOnPlane ⟨0, 2, 1⟩ pl ∧
  planesArePerpendicular pl ⟨2, -3, 6, -4⟩ ∧
  pl.a > 0 ∧
  gcd4 (Int.floor pl.a) (Int.floor pl.b) (Int.floor pl.c) (Int.floor pl.d) = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_proof_l921_92106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_on_zero_one_l921_92110

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

-- State the theorem
theorem f_monotone_decreasing_on_zero_one :
  StrictMonoOn f (Set.Ioo 0 1) := by
  sorry

-- Note: Set.Ioo represents an open interval (a, b)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_on_zero_one_l921_92110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l921_92155

/-- A positive integer has exactly 8 positive divisors -/
def has_eight_divisors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 8

/-- The equation r + p^4 = q^4 holds -/
def equation_holds (p q r : ℕ) : Prop :=
  r + p^4 = q^4

theorem unique_solution :
  ∀ p q r : ℕ,
    Nat.Prime p →
    Nat.Prime q →
    r > 0 →
    has_eight_divisors r →
    equation_holds p q r →
    p = 2 ∧ q = 5 ∧ r = 609 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l921_92155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_range_l921_92134

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 1 ∧
  (1 - t.b) * (Real.sin t.A + Real.sin t.B) = (t.c - t.b) * Real.sin t.C

-- Define the perimeter
def perimeter (t : Triangle) : Real := t.a + t.b + t.c

-- Theorem statement
theorem triangle_perimeter_range (t : Triangle) 
  (h : triangle_conditions t) : 2 < perimeter t ∧ perimeter t ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_range_l921_92134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_inequality_l921_92158

theorem vector_inequality (a b c d : ℝ × ℝ) (h : a + b + c + d = (0, 0)) :
  Real.sqrt ((a.1)^2 + (a.2)^2) + Real.sqrt ((b.1)^2 + (b.2)^2) + 
  Real.sqrt ((c.1)^2 + (c.2)^2) + Real.sqrt ((d.1)^2 + (d.2)^2) ≥ 
  Real.sqrt (((a.1 + d.1)^2 + (a.2 + d.2)^2)) + 
  Real.sqrt (((b.1 + d.1)^2 + (b.2 + d.2)^2)) + 
  Real.sqrt (((c.1 + d.1)^2 + (c.2 + d.2)^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_inequality_l921_92158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_value_factor_is_three_l921_92179

/-- The factor by which a card's value increases, given its original price and profit. -/
noncomputable def card_value_increase_factor (original_price profit : ℝ) : ℝ :=
  (original_price + profit) / original_price

/-- Theorem: The factor by which the card's value increased is 3. -/
theorem card_value_factor_is_three :
  card_value_increase_factor 100 200 = 3 := by
  -- Unfold the definition of card_value_increase_factor
  unfold card_value_increase_factor
  -- Simplify the arithmetic
  simp [add_div]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_value_factor_is_three_l921_92179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l921_92170

open Polynomial

theorem polynomial_division_theorem :
  ∃ (q r : Polynomial ℚ),
    X^5 - 22*X^3 + 12*X^2 - 16*X + 8 = (X - 3) * q + r ∧
    q = X^4 + 3*X^3 - 13*X^2 - 27*X - 97 ∧
    r = -211 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l921_92170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_properties_l921_92159

def M : Matrix (Fin 4) (Fin 4) ℕ := ![
  ![0, 7, 9, 14],
  ![11, 12, 2, 5],
  ![6, 1, 15, 8],
  ![13, 10, 4, 3]
]

def a (i j : Fin 4) : ℕ := M i j

def P (i : Fin 4) : ℕ := a 0 i + a 1 ((i + 1) % 4) + a 2 ((i + 2) % 4) + a 3 ((i + 3) % 4)

def S (i : Fin 4) : ℕ := a 3 i + a 2 ((i + 1) % 4) + a 1 ((i + 2) % 4) + a 0 ((i + 3) % 4)

def L (i : Fin 4) : ℕ := a i 0 + a i 1 + a i 2 + a i 3

def C (i : Fin 4) : ℕ := a 0 i + a 1 i + a 2 i + a 3 i

theorem matrix_properties :
  (∀ i j : Fin 4, P i = P j) ∧
  (∀ i j : Fin 4, S i = S j) ∧
  (∀ i j : Fin 4, L i = L j) ∧
  (∀ i j : Fin 4, C i = C j) ∧
  (a 0 0 = 0) ∧ (a 0 1 = 7) ∧ (a 1 0 = 11) ∧ (a 1 2 = 2) ∧ (a 2 2 = 15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_properties_l921_92159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_one_half_l921_92124

-- Define the function f
noncomputable def f (y : ℝ) : ℝ := 
  let x := (1 - y) / 2
  (1 - x^2) / x^2

-- Theorem statement
theorem f_at_one_half : f (1/2) = 15 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the let expression
  simp
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_one_half_l921_92124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_second_quadrant_l921_92178

noncomputable def complex_number : ℂ := (2 * Complex.I) / (1 - Complex.I)

theorem complex_number_in_second_quadrant :
  Real.sign (complex_number.re) = -1 ∧ Real.sign (complex_number.im) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_second_quadrant_l921_92178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l921_92194

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxesEllipse where
  center : ℝ × ℝ
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- The distance between the foci of an ellipse -/
noncomputable def focal_distance (e : ParallelAxesEllipse) : ℝ :=
  2 * Real.sqrt (e.semi_major_axis^2 - e.semi_minor_axis^2)

theorem ellipse_focal_distance :
  ∃ (e : ParallelAxesEllipse),
    e.center = (5, 2) ∧
    e.semi_major_axis = 5 ∧
    e.semi_minor_axis = 2 ∧
    focal_distance e = 2 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l921_92194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l921_92145

open Real

/-- An angle is in the second quadrant if it's between π/2 and π (modulo 2π) -/
def is_second_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi

/-- An angle is in the first or third quadrant if it's between 0 and π/2 or between π and 3π/2 (modulo 2π) -/
def is_first_or_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, (0 < α - 2 * k * Real.pi ∧ α - 2 * k * Real.pi < Real.pi / 2) ∨
           (Real.pi < α - 2 * k * Real.pi ∧ α - 2 * k * Real.pi < 3 * Real.pi / 2)

/-- If α is in the second quadrant, then α/2 is in the first or third quadrant -/
theorem half_angle_quadrant (α : ℝ) :
  is_second_quadrant α → is_first_or_third_quadrant (α / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l921_92145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_sum_l921_92180

-- Define the complex numbers
variable (z : ℂ)

-- State the theorem
theorem complex_equation_sum (a b : ℝ) :
  (a + 2 * Complex.I) * Complex.I = b + Complex.I → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_sum_l921_92180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_symmetry_l921_92148

open Real

/-- A function with the given properties -/
noncomputable def f (x : ℝ) : ℝ := sin (2 * x - π / 6)

/-- The minimum positive period of f is π -/
theorem f_period : ∃ (p : ℝ), p > 0 ∧ p = π ∧ ∀ (x : ℝ), f (x + p) = f x := by
  sorry

/-- The graph of f is symmetrical about the line x = π/3 -/
theorem f_symmetry : ∀ (x : ℝ), f (π/3 + x) = f (π/3 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_symmetry_l921_92148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_probability_l921_92199

-- Define the number of students
def num_students : ℕ := 7

-- Define the seating arrangement
def seating_arrangement : List (List ℕ) := [[2, 2], [2, 2], [3]]

-- Define the number of adjacent pairs in the same row
def same_row_pairs : ℕ := 7

-- Define the number of adjacent pairs in adjacent rows
def adjacent_row_pairs : ℕ := 4

-- Define the probability of Abby and Bridget being adjacent
def probability_adjacent : ℚ := 11 / 21

-- Theorem statement
theorem adjacent_probability :
  (((same_row_pairs + adjacent_row_pairs) * 2 * (num_students - 2).factorial : ℚ) / num_students.factorial) = probability_adjacent := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_probability_l921_92199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_to_cube_volume_ratio_l921_92169

noncomputable section

variable (s : ℝ)

def cube_side_length : ℝ := s

def tetrahedron_volume : ℝ := (s^3 * Real.sqrt 6) / 32

def cube_volume : ℝ := s^3

theorem tetrahedron_to_cube_volume_ratio :
  tetrahedron_volume s / cube_volume s = Real.sqrt 6 / 32 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_to_cube_volume_ratio_l921_92169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_value_l921_92123

def f (x : ℤ) : ℤ := |8 * x^2 - 50 * x + 21|

theorem smallest_prime_value (x : ℤ) :
  f x = |(4 * x - 21) * (2 * x - 1)| →
  (∀ y : ℤ, y < 1 → ¬(Nat.Prime (Int.natAbs (f y)))) ∧ Nat.Prime (Int.natAbs (f 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_value_l921_92123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_tan_inequality_l921_92105

theorem min_m_for_tan_inequality : 
  (∃ m : ℝ, ∀ x : ℝ, x ∈ Set.Icc 0 (π/4) → Real.tan x ≤ m) ∧ 
  (∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 (π/4) → Real.tan x ≤ m) → m ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_tan_inequality_l921_92105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_k_equals_one_l921_92138

/-- The curve function -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x + (1/k) * Real.log x

/-- The derivative of the curve function -/
noncomputable def f_derivative (k : ℝ) (x : ℝ) : ℝ := 1 + 1/(k*x)

theorem tangent_perpendicular_implies_k_equals_one (k : ℝ) :
  k > 0 →  -- Ensure k is positive for log to be defined
  f k 1 = 1 →  -- The curve passes through (1,1)
  (f_derivative k 1) * (-1/2) = -1 →  -- Tangent is perpendicular to x + 2y = 0
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_k_equals_one_l921_92138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candidate_vote_percentage_l921_92167

def total_votes : ℕ := 6450
def loss_margin : ℕ := 2451

theorem candidate_vote_percentage :
  let candidate_votes := (total_votes - loss_margin) / 2
  let percentage := (candidate_votes : ℚ) / total_votes * 100
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |percentage - 30.99| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candidate_vote_percentage_l921_92167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l921_92126

theorem trig_problem (α β : ℝ) 
  (h1 : 0 < α ∧ α < Real.pi/2) 
  (h2 : 0 < β ∧ β < Real.pi/2) 
  (h3 : Real.sin β / Real.sin α = Real.cos (α + β)) : 
  (Real.tan β = (Real.sin α * Real.cos α) / (1 + Real.sin α ^ 2)) ∧ 
  (∀ γ, (0 < γ ∧ γ < Real.pi/2) → Real.tan γ ≤ Real.sqrt 2 / 4) ∧ 
  (∃ δ, (0 < δ ∧ δ < Real.pi/2) ∧ Real.tan δ = Real.sqrt 2 / 4) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l921_92126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_solutions_l921_92188

noncomputable section

open Real

theorem cubic_root_equation_solutions :
  {x : ℝ | (18 * x - 2) ^ (1/3) + (16 * x + 2) ^ (1/3) = 4 * (2 * x) ^ (1/3)} =
  {0, -1/10, 8/9} := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_solutions_l921_92188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_distance_after_training_l921_92182

/-- Represents John's ultramarathon training progress -/
structure UltramarathonTraining where
  initial_time : ℝ
  time_increase_percentage : ℝ
  initial_speed : ℝ
  speed_increase : ℝ

/-- Calculates the distance John can run after training -/
noncomputable def distance_after_training (t : UltramarathonTraining) : ℝ :=
  (t.initial_time * (1 + t.time_increase_percentage / 100)) * (t.initial_speed + t.speed_increase)

/-- Theorem stating that John can run 168 miles after training -/
theorem john_distance_after_training :
  let john_training := UltramarathonTraining.mk 8 75 8 4
  distance_after_training john_training = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_distance_after_training_l921_92182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_not_contained_l921_92165

/-- 
Given a set of integers from 1 to n (where n ≥ 4), this theorem states that
the maximum number of subsets that can be formed, where:
1. The i-th subset has i elements
2. No subset is contained within another subset
is equal to n - 2.
-/
theorem max_subsets_not_contained (n : ℕ) (h : n ≥ 4) :
  (∃ (m : ℕ) (A : Fin m → Finset (Fin n)),
    (∀ i : Fin m, (A i).card = i.val + 1) ∧
    (∀ i j : Fin m, i ≠ j → ¬(A i ⊆ A j)) ∧
    m = n - 2) ∧
  ¬(∃ (m : ℕ) (A : Fin m → Finset (Fin n)),
    (∀ i : Fin m, (A i).card = i.val + 1) ∧
    (∀ i j : Fin m, i ≠ j → ¬(A i ⊆ A j)) ∧
    m > n - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_not_contained_l921_92165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_with_distance_3_l921_92121

/-- Line passing through a point -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
noncomputable def distancePointToLine (x y : ℝ) (l : Line) : ℝ :=
  (abs (l.a * x + l.b * y + l.c)) / (Real.sqrt (l.a^2 + l.b^2))

/-- Intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : (ℝ × ℝ) :=
  let x := (l1.b * l2.c - l2.b * l1.c) / (l1.a * l2.b - l2.a * l1.b)
  let y := (l2.a * l1.c - l1.a * l2.c) / (l1.a * l2.b - l2.a * l1.b)
  (x, y)

theorem line_through_intersection_with_distance_3 (l1 l2 : Line) :
  l1.a = -1/3 ∧ l1.b = 1 ∧ l1.c = -10 ∧
  l2.a = 3 ∧ l2.b = -1 ∧ l2.c = 0 →
  ∃ l : Line,
    (l.a = 1 ∧ l.b = 0 ∧ l.c = -3) ∨
    (l.a = 4 ∧ l.b = -3 ∧ l.c = 15) ∧
    let (x, y) := intersectionPoint l1 l2
    (l.a * x + l.b * y + l.c = 0) ∧
    distancePointToLine 0 0 l = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_with_distance_3_l921_92121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_square_area_ratio_l921_92162

theorem rectangle_square_area_ratio : 
  ∀ (s : ℝ), s > 0 →
  (let square_side := s
   let rectangle_long_side := 1.15 * s
   let rectangle_short_side := 0.95 * s
   let square_area := s ^ 2
   let rectangle_area := rectangle_long_side * rectangle_short_side
   rectangle_area / square_area = 109.25 / 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_square_area_ratio_l921_92162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_five_factorial_l921_92161

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem probability_factor_five_factorial (n : ℕ) (h : n = 36) :
  (Finset.filter (λ x => factorial 5 % x = 0) (Finset.range n)).card / n = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_five_factorial_l921_92161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_is_random_l921_92153

-- Define the events
def event_A : Prop := ∀ p : Real, p > 0 → p < 1  -- Simplified placeholder

def event_B (a : ℝ) : Prop := a ≠ 0  -- Simplified condition for quadratic equation

def event_C : Prop := ∀ x y z : ℝ, x + y + z = 180  -- Simplified angle sum condition

def event_D : Prop := ∀ x y : ℝ, x < 0 → y < 0 → x * y > 0

-- Define what it means for an event to be random
def is_random_event (P : Prop) : Prop := 
  ∃ (Ω : Type) (p : Ω → Prop), (∃ ω₁ ω₂ : Ω, p ω₁ ∧ ¬p ω₂) ∧ (P ↔ ∃ ω : Ω, p ω)

-- Theorem statement
theorem only_B_is_random : 
  ¬is_random_event event_A ∧
  is_random_event (∃ a : ℝ, event_B a) ∧
  ¬is_random_event event_C ∧
  ¬is_random_event event_D :=
by
  sorry  -- Proof is omitted

#check only_B_is_random

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_is_random_l921_92153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solutions_l921_92139

theorem trigonometric_equation_solutions (x : ℝ) : 
  (Real.sin (2 * x))^4 + (Real.sin (2 * x))^3 * (Real.cos (2 * x)) - 
  8 * (Real.sin (2 * x)) * (Real.cos (2 * x))^3 - 8 * (Real.cos (2 * x))^4 = 0 ↔ 
  (∃ k : ℤ, x = -π/8 + π*k/2) ∨ (∃ n : ℤ, x = Real.arctan 2/2 + π*n/2) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solutions_l921_92139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l921_92109

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2017 * x + Real.pi / 6) + Real.cos (2017 * x + Real.pi / 3)

/-- The theorem statement -/
theorem min_value_theorem (A : ℝ) (x₁ x₂ : ℝ) 
  (h_max : ∀ x, f x ≤ A)
  (h_bounds : ∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂) :
  2 * Real.pi / 2017 ≤ A * |x₁ - x₂| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l921_92109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_transformation_l921_92192

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x^2)
def domain_f_x_squared : Set ℝ := Set.Ioc (-3) 1

-- Define the domain of f(x-1)
def domain_f_x_minus_one : Set ℝ := Set.Icc 1 10

-- Theorem statement
theorem domain_transformation (h : ∀ x, f (x^2) ∈ domain_f_x_squared ↔ x ∈ domain_f_x_squared) :
  ∀ x, f (x - 1) ∈ domain_f_x_minus_one ↔ x ∈ domain_f_x_minus_one :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_transformation_l921_92192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l921_92116

/-- An ellipse passing through two given points has the specified equation, vertices, and eccentricity -/
theorem ellipse_properties (x y : ℝ → ℝ) :
  (∀ t, x t^2 / 9 + y t^2 / 16 = 1) →  -- Standard equation
  x 0 = 2 ∧ y 0 = -4 * Real.sqrt 5 / 3 →  -- Point A
  x 1 = -1 ∧ y 1 = 8 * Real.sqrt 2 / 3 →  -- Point B
  (∃ t₁ t₂ t₃ t₄, x t₁ = 3 ∧ y t₁ = 0 ∧
                  x t₂ = -3 ∧ y t₂ = 0 ∧
                  x t₃ = 0 ∧ y t₃ = 4 ∧
                  x t₄ = 0 ∧ y t₄ = -4) ∧  -- Vertices
  (Real.sqrt 7 / 4)^2 = (3^2 - 4^2) / 3^2 :=  -- Eccentricity
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l921_92116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circus_ticket_sales_l921_92127

theorem circus_ticket_sales (lower_price upper_price total_tickets total_revenue : ℕ) 
  (h1 : lower_price = 30)
  (h2 : upper_price = 20)
  (h3 : total_tickets = 80)
  (h4 : total_revenue = 2100)
  : ∃ (lower_seats : ℕ), 
    lower_seats * lower_price + (total_tickets - lower_seats) * upper_price = total_revenue ∧
    lower_seats = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circus_ticket_sales_l921_92127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_plus_pi_fourth_l921_92181

theorem cos_theta_plus_pi_fourth (θ : ℝ) 
  (h1 : Real.cos θ = -12/13) 
  (h2 : θ ∈ Set.Ioo π (3*π/2)) : 
  Real.cos (θ + π/4) = -7*Real.sqrt 2/26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_plus_pi_fourth_l921_92181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_third_side_product_l921_92128

noncomputable section

open Real Set

/-- Represents a triangle as a set of three points in ℝ² -/
def Triangle := Set (ℝ × ℝ)

/-- Checks if a triangle is a right triangle -/
def IsRightTriangle (T : Triangle) : Prop := sorry

/-- Calculates the area of a triangle -/
def area (T : Triangle) : ℝ := sorry

/-- Returns the set of side lengths of a triangle -/
def sides (T : Triangle) : Set ℝ := sorry

theorem right_triangles_third_side_product (T₁ T₂ : Triangle) 
  (h_right₁ : IsRightTriangle T₁) (h_right₂ : IsRightTriangle T₂)
  (h_area₁ : area T₁ = 2) (h_area₂ : area T₂ = 3)
  (h_side₁ : ∃ (s₁ s₂ : ℝ), s₁ ∈ sides T₁ ∧ s₂ ∈ sides T₂ ∧ s₁ = 2 * s₂)
  (h_side₂ : ∃ (s₃ s₄ : ℝ), s₃ ∈ sides T₁ ∧ s₄ ∈ sides T₂ ∧ s₃ = 2 * s₄ ∧ 
    ∀ (s₁ s₂ : ℝ), (s₁ ∈ sides T₁ ∧ s₂ ∈ sides T₂ ∧ s₁ = 2 * s₂) → (s₃ ≠ s₁ ∧ s₄ ≠ s₂)) :
  ∃ (z₁ z₂ : ℝ), z₁ ∈ sides T₁ ∧ z₂ ∈ sides T₂ ∧ (z₁ * z₂)^2 = 676 / 9 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_third_side_product_l921_92128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_theorem_l921_92197

/-- A cubic function with parameter a and constant c -/
def f (a c x : ℝ) : ℝ := x^3 + a*x^2 + (c - a)

/-- The set of a values for which f has three distinct zeros -/
def three_zeros_range : Set ℝ :=
  Set.Ioi (-3) ∪ Set.Ioo 1 (3/2) ∪ Set.Ioi (3/2)

/-- Theorem stating that if f has three distinct zeros exactly when a is in the specified range, then c must equal 1 -/
theorem cubic_function_theorem (c : ℝ) :
  (∀ a : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a c x = 0 ∧ f a c y = 0 ∧ f a c z = 0) ↔ a ∈ three_zeros_range) →
  c = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_theorem_l921_92197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_true_discount_is_60_l921_92189

/-- Banker's gain on a bill -/
noncomputable def bankers_gain : ℝ := 7.2

/-- Annual interest rate as a decimal -/
noncomputable def interest_rate : ℝ := 0.12

/-- Time period in years -/
noncomputable def time : ℝ := 1

/-- True discount on the bill -/
noncomputable def true_discount : ℝ := bankers_gain * 100 / (interest_rate * 100)

/-- Theorem stating that the true discount is 60 given the conditions -/
theorem true_discount_is_60 : true_discount = 60 := by
  -- Unfold the definition of true_discount
  unfold true_discount
  -- Simplify the expression
  simp [bankers_gain, interest_rate]
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_true_discount_is_60_l921_92189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_society_theorem_l921_92186

/-- Represents the relationship between two knights -/
inductive Relationship
| Friend
| Enemy
deriving DecidableEq

/-- Represents a knight -/
structure Knight where
  id : Nat
deriving DecidableEq

/-- The social structure of the knights -/
structure KnightSociety where
  n : Nat
  knights : Finset Knight
  relationship : Knight → Knight → Relationship

/-- Checks if the given KnightSociety satisfies all conditions -/
def isValidSociety (society : KnightSociety) : Prop :=
  -- Every pair of knights is either friends or enemies
  ∀ k1 k2 : Knight, k1 ∈ society.knights → k2 ∈ society.knights → k1 ≠ k2 →
    (society.relationship k1 k2 = Relationship.Friend ∨ society.relationship k1 k2 = Relationship.Enemy) ∧
  -- Each knight has exactly three enemies
  (∀ k : Knight, k ∈ society.knights →
    (society.knights.filter (fun k' => society.relationship k k' = Relationship.Enemy)).card = 3) ∧
  -- The enemies of a knight's friends are also that knight's enemies
  (∀ k1 k2 k3 : Knight, k1 ∈ society.knights → k2 ∈ society.knights → k3 ∈ society.knights →
    society.relationship k1 k2 = Relationship.Friend →
    society.relationship k2 k3 = Relationship.Enemy →
    society.relationship k1 k3 = Relationship.Enemy)

theorem knight_society_theorem :
  ∀ n : Nat, (∃ society : KnightSociety, society.n = n ∧ isValidSociety society) ↔ (n = 4 ∨ n = 6) := by
  sorry

#check knight_society_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_society_theorem_l921_92186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_correct_l921_92164

/-- The symmetric point of (2, 3, 4) with respect to the xOz plane -/
def symmetric_point : ℝ × ℝ × ℝ := (2, -3, 4)

/-- The original point -/
def original_point : ℝ × ℝ × ℝ := (2, 3, 4)

/-- The xOz plane is defined by y = 0 -/
def xOz_plane (p : ℝ × ℝ × ℝ) : Prop := p.2.1 = 0

theorem symmetric_point_correct :
  (symmetric_point.1 = original_point.1) ∧
  (symmetric_point.2.1 = -original_point.2.1) ∧
  (symmetric_point.2.2 = original_point.2.2) ∧
  (xOz_plane (0, symmetric_point.2.1 + original_point.2.1, 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_correct_l921_92164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_value_l921_92150

noncomputable def floor_d (d : ℝ) : ℤ := Int.floor d

noncomputable def frac_d (d : ℝ) : ℝ := d - floor_d d

theorem d_value :
  ∀ d : ℝ,
  (∃ x : ℤ, x = floor_d d ∧ 3 * (x : ℝ)^2 + 19 * (x : ℝ) - 63 = 0) →
  (4 * (frac_d d)^2 - 21 * frac_d d + 8 = 0) →
  (0 ≤ frac_d d ∧ frac_d d < 1) →
  d = -35/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_value_l921_92150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_placements_count_l921_92115

/-- Represents a 5x5 chess board -/
def ChessBoard := Fin 5 → Fin 5

/-- Represents a knight's position on the board -/
structure Knight where
  row : Fin 5
  col : Fin 5

/-- Checks if two knights threaten each other -/
def threatens (k1 k2 : Knight) : Prop :=
  (Int.natAbs (k1.row - k2.row) = 2 ∧ Int.natAbs (k1.col - k2.col) = 1) ∨
  (Int.natAbs (k1.row - k2.row) = 1 ∧ Int.natAbs (k1.col - k2.col) = 2)

/-- A valid placement of 5 knights on the board -/
structure KnightPlacement where
  knights : Fin 5 → Knight
  distinct : ∀ i j, i ≠ j → knights i ≠ knights j
  non_threatening : ∀ i j, i ≠ j → ¬threatens (knights i) (knights j)

/-- Represents symmetries of the board (rotations and reflections) -/
inductive BoardSymmetry
| rotate90 | rotate180 | rotate270
| reflectHorizontal | reflectVertical
| reflectDiagonal1 | reflectDiagonal2

/-- Applies a symmetry to a knight placement -/
noncomputable def applySymmetry (sym : BoardSymmetry) (placement : KnightPlacement) : KnightPlacement :=
  sorry

/-- Two placements are equivalent if one can be obtained from the other by applying a symmetry -/
def equivalent (p1 p2 : KnightPlacement) : Prop :=
  ∃ sym, applySymmetry sym p1 = p2

theorem knight_placements_count :
  ∃! (placements : Finset KnightPlacement),
    Finset.card placements = 8 ∧
    (∀ p : KnightPlacement, p ∈ placements) ∧
    (∀ p1 p2 : KnightPlacement, p1 ∈ placements → p2 ∈ placements → equivalent p1 p2 → p1 = p2) :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_placements_count_l921_92115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_value_l921_92187

open Real

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := sin x + cos x

-- Define the translated function F'
noncomputable def F' (X : ℝ) (m : ℝ) : ℝ := f (X + m)

-- Theorem statement
theorem graph_translation_value (m : ℝ) 
  (h1 : 0 < m) (h2 : m < π) : 
  (∀ X, F' X m = f X) → m = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_value_l921_92187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_system_l921_92131

theorem unique_solution_for_system : 
  ∃! p : ℝ × ℝ, 
    let (x, y) := p
    y = (x + 1)^4 ∧ 
    x*y + y = Real.cos (Real.pi * x) ∧ 
    x = 0 ∧ 
    y = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_system_l921_92131
