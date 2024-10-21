import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_and_minimum_k_l332_33223

def S (n : ℕ) (a : ℕ → ℝ) : ℝ := 2 * (a n - (2 : ℝ)^n + 1)

theorem sequence_property_and_minimum_k 
  (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, S n a = 2 * (a n - (2 : ℝ)^n + 1)) :
  (∀ n : ℕ, a n / (2 : ℝ)^n = n) ∧ 
  (∀ k : ℝ, (∀ n : ℕ, k > (S n a - 2) / a n) ↔ k > 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_and_minimum_k_l332_33223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_distance_is_1_plus_sqrt_11_l332_33262

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- Define the starting point P
noncomputable def P : ℝ := 
  let x₁ := 0  -- x-coordinate of P where f(x) = -3
  let x₂ := -2 -- other possible x-coordinate
  if f x₁ = -3 then x₁ else x₂

-- Define the ending point Q
noncomputable def Q : ℝ := 
  let x₁ := -1 + Real.sqrt 11  -- x-coordinate of Q where f(x) = 7
  let x₂ := -1 - Real.sqrt 11  -- other possible x-coordinate
  if |P - x₁| < |P - x₂| then x₁ else x₂

theorem horizontal_distance_is_1_plus_sqrt_11 :
  f P = -3 ∧ f Q = 7 ∧ |Q - P| = 1 + Real.sqrt 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_distance_is_1_plus_sqrt_11_l332_33262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l332_33257

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi/3 + x) * Real.cos (Real.pi/3 - x) - Real.sin x * Real.cos x + 1/4

theorem f_properties :
  -- Smallest positive period is π
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ T, T > 0 → (∀ x, f (x + T) = f x) → T ≥ Real.pi) ∧
  -- Maximum value is √2/2
  (∀ x, f x ≤ Real.sqrt 2 / 2) ∧
  (∃ x, f x = Real.sqrt 2 / 2) ∧
  -- Maximum value obtained when x = kπ - π/8, k ∈ ℤ
  (∀ x, f x = Real.sqrt 2 / 2 ↔ ∃ k : ℤ, x = k * Real.pi - Real.pi/8) ∧
  -- Monotonically decreasing when kπ - π/8 ≤ x ≤ kπ + 3π/8, k ∈ ℤ
  (∀ k : ℤ, ∀ x y, k * Real.pi - Real.pi/8 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + 3 * Real.pi/8 → f y < f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l332_33257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_plus_b_l332_33238

theorem min_value_a_plus_b (a b : ℝ) 
  (h1 : Real.log (3 * a + 4 * b) / Real.log 4 = Real.log (Real.sqrt (a * b)) / Real.log 2)
  (h2 : 3 * a + 4 * b > 0)
  (h3 : a * b > 0) :
  a + b ≥ 7 + 4 * Real.sqrt 3 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ + b₀ = 7 + 4 * Real.sqrt 3 ∧
    Real.log (3 * a₀ + 4 * b₀) / Real.log 4 = Real.log (Real.sqrt (a₀ * b₀)) / Real.log 2 ∧
    3 * a₀ + 4 * b₀ > 0 ∧ a₀ * b₀ > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_plus_b_l332_33238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABC_is_6_times_ORD_l332_33290

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define points M, N, O, D, and R
variable (M N O D R : EuclideanSpace ℝ (Fin 2))

-- AM and CN are medians
def is_median_AM (A B C M : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def is_median_CN (A B C N : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- O is the intersection of AM and CN
def O_is_intersection (A B C M N O : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- D is the midpoint of BC
def D_is_midpoint_BC (B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- R is the intersection of AD and CN
def R_is_intersection (A C D N R : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Area of triangle ORD
noncomputable def area_ORD (O R D : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Area of triangle ABC
noncomputable def area_ABC (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Main theorem
theorem area_ABC_is_6_times_ORD 
  (h1 : is_median_AM A B C M)
  (h2 : is_median_CN A B C N)
  (h3 : O_is_intersection A B C M N O)
  (h4 : D_is_midpoint_BC B C D)
  (h5 : R_is_intersection A C D N R)
  : area_ABC A B C = 6 * area_ORD O R D := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABC_is_6_times_ORD_l332_33290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_challenge_sequence_properties_l332_33231

def challenge_sequence : ℕ → ℚ
  | 0 => 2/3  -- Define for 0 to cover all cases
  | 1 => 2/3
  | n+2 => challenge_sequence (n+1) * (1/3) + (1 - challenge_sequence (n+1)) * (2/3)

theorem challenge_sequence_properties :
  (challenge_sequence 2 = 4/9) ∧
  (∀ n : ℕ, n ≥ 1 → challenge_sequence n = 1/2 + 1/6 * (-1/3)^(n-1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_challenge_sequence_properties_l332_33231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_l332_33205

/-- Given a square with perimeter 800 cm and a rectangle with width 64 cm, 
    if the area of the square is five times the area of the rectangle, 
    then the length of the rectangle is 125 cm. -/
theorem rectangle_length (square_perimeter : ℝ) (rectangle_width : ℝ) 
  (h1 : square_perimeter = 800) 
  (h2 : rectangle_width = 64) : ℝ := by
  let square_side := square_perimeter / 4
  let square_area := square_side ^ 2
  let rectangle_length := square_area / (5 * rectangle_width)
  have : rectangle_length = 125 := by
    -- Proof goes here
    sorry
  exact rectangle_length


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_l332_33205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l332_33298

/-- A sequence of positive real numbers -/
def a : ℕ+ → ℝ := sorry

/-- Sum of the first n terms of sequence a -/
def S : ℕ+ → ℝ := sorry

/-- The sequence c_n defined as 1 / (a_n * a_{n+1}) -/
noncomputable def c (n : ℕ+) : ℝ := 1 / (a n * a (n + 1))

/-- Sum of the first n terms of sequence c -/
def T : ℕ+ → ℝ := sorry

axiom a_positive (n : ℕ+) : 0 < a n

axiom point_on_curve (n : ℕ+) : S n = (1/8) * (a n)^2 + (1/2) * (a n) + (1/2)

theorem sequence_properties :
  (∀ n : ℕ+, a n = 4 * n.val - 2) ∧
  (∀ n : ℕ+, T n = n.val / (4 * (2 * n.val + 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l332_33298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_operations_impossible_transformation_l332_33258

theorem calculator_operations (n : ℕ) :
  (n % 5 = 4 → n^2 % 5 = 1) ∧
  ((n % 5 = 4 ∨ n % 5 = 1) → ((n^2 - 5) % 5 = 4 ∨ (n^2 - 5) % 5 = 1)) :=
by sorry

def square (n : ℕ) : ℕ := n^2

def subtract_five (n : ℕ) : ℕ := n - 5

theorem impossible_transformation :
  ¬ ∃ (seq : List (ℕ → ℕ)), (∀ f ∈ seq, f = square ∨ f = subtract_five) ∧
    (seq.foldl (λ acc f => f acc) 9 = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_operations_impossible_transformation_l332_33258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rock_pile_problem_l332_33253

/-- Represents a pile of rocks -/
structure RockPile where
  totalWeight : ℝ
  count : ℕ+

/-- Calculates the mean weight of a pile of rocks -/
noncomputable def meanWeight (pile : RockPile) : ℝ :=
  pile.totalWeight / pile.count

/-- Combines two piles of rocks -/
def combinePiles (pile1 pile2 : RockPile) : RockPile where
  totalWeight := pile1.totalWeight + pile2.totalWeight
  count := pile1.count + pile2.count

/-- Theorem statement for the rock pile problem -/
theorem rock_pile_problem (A B C : RockPile)
    (hA : meanWeight A = 50)
    (hB : meanWeight B = 60)
    (hAB : meanWeight (combinePiles A B) = 53)
    (hAC : meanWeight (combinePiles A C) = 54) :
    ∃ (n : ℕ), n = 63 ∧ 
    ∀ (m : ℕ), (m : ℝ) ≤ meanWeight (combinePiles B C) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rock_pile_problem_l332_33253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_of_regular_quadrilateral_pyramid_l332_33247

/-- The lateral surface area of a regular quadrilateral pyramid -/
noncomputable def lateral_surface_area (a : ℝ) (α : ℝ) : ℝ :=
  a^2 / Real.sqrt (-Real.cos α)

/-- Theorem: The lateral surface area of a regular quadrilateral pyramid
    with base side length a and angle α between adjacent lateral faces
    is equal to a²/√(-cos α) -/
theorem lateral_surface_area_of_regular_quadrilateral_pyramid
  (a : ℝ) (α : ℝ) (h1 : a > 0) (h2 : 0 < α ∧ α < Real.pi) :
  lateral_surface_area a α = a^2 / Real.sqrt (-Real.cos α) := by
  sorry

#check lateral_surface_area_of_regular_quadrilateral_pyramid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_of_regular_quadrilateral_pyramid_l332_33247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_roots_l332_33203

theorem sum_of_squares_of_roots (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let r₁ := -b / (2 * a) + Real.sqrt ((b^2 - 4*a*c) / (4 * a^2))
  let r₂ := -b / (2 * a) - Real.sqrt ((b^2 - 4*a*c) / (4 * a^2))
  (a = 5) → (b = 15) → (c = -25) →
  r₁^2 + r₂^2 = 19 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_roots_l332_33203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_factorial_base16_zeros_l332_33220

/-- The number of trailing zeros in n! when expressed in base b -/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- 15 factorial -/
def factorial15 : ℕ := Nat.factorial 15

theorem fifteen_factorial_base16_zeros :
  trailingZeros factorial15 16 = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_factorial_base16_zeros_l332_33220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_l332_33296

theorem sin_half_angle (α : Real) (h1 : 0 < α ∧ α < Real.pi / 2) 
  (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_l332_33296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_approximation_q_l332_33273

theorem closest_approximation_q : 
  let q : ℚ := (6928/100 * 4/1000) / (3/100)
  round (q * 100) / 100 = 924/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_approximation_q_l332_33273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l332_33251

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_sum_ratio 
  (seq : ArithmeticSequence) 
  (h_nonzero : ∀ n, seq.a n ≠ 0) : 
  sum_n seq 5 / seq.a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l332_33251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_implies_a_bound_l332_33228

/-- Given a ≥ 0 and f(x) = (x^2 - 2ax)e^x is monotonically decreasing on [-1, 1], prove that a ≥ 3/4 -/
theorem monotonic_decreasing_implies_a_bound (a : ℝ) (h_a : a ≥ 0) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, ∀ y ∈ Set.Icc (-1 : ℝ) 1, x ≤ y →
    (x^2 - 2*a*x) * Real.exp x ≥ (y^2 - 2*a*y) * Real.exp y) →
  a ≥ 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_implies_a_bound_l332_33228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_juice_percentage_approx_43_l332_33216

/-- Represents the juice extraction rate for a fruit -/
structure JuiceRate where
  fruit : String
  quantity : ℕ
  juice : ℚ

/-- Represents a juice blend -/
structure JuiceBlend where
  apple_count : ℕ
  plum_count : ℕ

def apple_juice_rate : JuiceRate := ⟨"apple", 2, 9⟩
def plum_juice_rate : JuiceRate := ⟨"plum", 3, 12⟩
def blend : JuiceBlend := ⟨4, 6⟩

/-- Calculates the amount of juice from a given number of fruits -/
def juice_amount (rate : JuiceRate) (count : ℕ) : ℚ :=
  (rate.juice / rate.quantity) * count

/-- Calculates the percentage of apple juice in the blend -/
def apple_juice_percentage (blend : JuiceBlend) : ℚ :=
  let apple_juice := juice_amount apple_juice_rate blend.apple_count
  let plum_juice := juice_amount plum_juice_rate blend.plum_count
  let total_juice := apple_juice + plum_juice
  (apple_juice / total_juice) * 100

theorem apple_juice_percentage_approx_43 :
  abs (apple_juice_percentage blend - 43) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_juice_percentage_approx_43_l332_33216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_odd_nor_even_l332_33229

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.exp (Real.log 3 * x)

theorem f_neither_odd_nor_even :
  (∀ x, f (-x) = -f x) = false ∧ (∀ x, f (-x) = f x) = false :=
by
  apply And.intro
  · show (∀ x, f (-x) = -f x) = false
    sorry
  · show (∀ x, f (-x) = f x) = false
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_odd_nor_even_l332_33229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l332_33280

/-- Given a binomial (x^2 - 1/x)^n, if its expansion has 6 terms,
    then the coefficient of x^4 in the expansion is 10 -/
theorem binomial_expansion_coefficient (x : ℝ) (n : ℕ) : 
  (∃ (a b c d e f : ℝ), (x^2 - 1/x)^n = a*x^(2*n) + b*x^(2*n-3) + c*x^(2*n-6) + d*x^(2*n-9) + e*x^(2*n-12) + f*x^(2*n-15)) →
  (∃ (k : ℝ), ∃ (rest : ℝ → ℝ), (x^2 - 1/x)^n = k*x^4 + rest x) →
  k = 10 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l332_33280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_largest_l332_33282

-- Define the constants
noncomputable def a : ℝ := Real.log 2022 / Real.log 2021
noncomputable def b : ℝ := Real.log 2023 / Real.log 2022
noncomputable def c : ℝ := 2022 / 2021
noncomputable def d : ℝ := 2023 / 2022

-- Theorem statement
theorem c_is_largest : c > a ∧ c > b ∧ c > d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_largest_l332_33282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_OH_ON_ratio_MH_intersects_C_only_at_H_l332_33292

-- Define the basic parameters and points
variable (t p : ℝ)
variable (M P N H : ℝ × ℝ)

-- Define the conditions
axiom t_nonzero : t ≠ 0
axiom p_positive : p > 0
axiom line_l : (∀ x : ℝ, (x, t) ∈ {z : ℝ × ℝ | z.2 = t})
axiom parabola_C : (∀ x y : ℝ, (x, y) ∈ {z : ℝ × ℝ | z.2^2 = 2*p*z.1})
axiom M_on_yaxis : M.1 = 0 ∧ M.2 = t
axiom P_on_l_and_C : P.2 = t ∧ P.1 = t^2 / (2*p)
axiom N_symmetric_to_M : N.1 + M.1 = 2 * P.1 ∧ N.2 + M.2 = 2 * P.2
axiom H_on_ON_and_C : ∃ k : ℝ, H = (k * N.1, k * N.2) ∧ H.2^2 = 2*p*H.1

-- State the theorems to be proved
theorem OH_ON_ratio : |H.2| / |N.2| = 2 := by
  sorry

theorem MH_intersects_C_only_at_H :
  ∀ Q : ℝ × ℝ, Q ∈ {z : ℝ × ℝ | z.2^2 = 2*p*z.1} ∧ 
  Q ∈ {z : ℝ × ℝ | (z.2 - M.2) = ((H.2 - M.2) / (H.1 - M.1)) * (z.1 - M.1)} →
  Q = H := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_OH_ON_ratio_MH_intersects_C_only_at_H_l332_33292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_prime_pairs_l332_33232

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define the condition function
def condition (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ is_prime (p^q - p*q)

-- State the theorem
theorem exactly_two_prime_pairs :
  ∃! (pairs : Finset (ℕ × ℕ)), (∀ (p q : ℕ), (p, q) ∈ pairs ↔ condition p q) ∧ pairs.card = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_prime_pairs_l332_33232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_plus_pi_sixth_l332_33268

theorem sin_double_plus_pi_sixth (θ : ℝ) (h : Real.sin (θ - π/6) = 1/3) :
  Real.sin (2*θ + π/6) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_plus_pi_sixth_l332_33268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l332_33274

open Real

-- Define the triangle ABC
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Conditions for a valid triangle
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  -- Angles and sides are related correctly
  a * sin B = b * sin A ∧
  b * sin C = c * sin B ∧
  c * sin A = a * sin C

-- Main theorem
theorem triangle_properties 
  (A B C a b c : ℝ)
  (triangle : Triangle A B C a b c)
  (h1 : a * (cos (C/2))^2 + c * (cos (A/2))^2 = 3*b/2)
  (h2 : b = 2)
  (h3 : b * c * cos A = 3) :
  (sin A + sin C = 2 * sin B) ∧
  (1/2 * b * c * sin A = 3*sqrt 5/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l332_33274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l332_33202

-- Define the function f(x) = 1 + log₃(x)
noncomputable def f (x : ℝ) : ℝ := 1 + (Real.log x) / (Real.log 3)

-- State the theorem
theorem range_of_f :
  {y | ∃ x > 9, f x = y} = Set.Ioi 3 := by
  sorry

-- Define the set of y values
def range_set : Set ℝ := {y | ∃ x > 9, f x = y}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l332_33202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l332_33200

-- Define the hyperbola C
def hyperbola (x y b : ℝ) : Prop := x^2 - y^2/b^2 = 1

-- Define the circle P
def circle_P (x y : ℝ) : Prop := (x-2)^2 + (y+4)^2 = 1

-- Define the center of circle P
def circle_center : ℝ × ℝ := (2, -4)

-- Define the asymptote of hyperbola C
def asymptote (x y b : ℝ) : Prop := y = b * x

-- Define the eccentricity of hyperbola C
noncomputable def eccentricity (b : ℝ) : ℝ := Real.sqrt (1 + b^2)

-- Theorem statement
theorem hyperbola_eccentricity :
  ∃ b : ℝ, 
    (asymptote (circle_center.1) (circle_center.2) b) ∧ 
    (eccentricity b = Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l332_33200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_ratio_l332_33260

/-- Represents a right circular cone --/
structure RightCircularCone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a frustum of a right circular cone --/
structure ConeFrustum where
  lowerRadius : ℝ
  upperRadius : ℝ
  height : ℝ

noncomputable def surfaceArea (cone : RightCircularCone) : ℝ :=
  Real.pi * cone.baseRadius * Real.sqrt (cone.baseRadius^2 + cone.height^2)

noncomputable def frustumSurfaceArea (frustum : ConeFrustum) : ℝ :=
  Real.pi * (frustum.lowerRadius + frustum.upperRadius) * 
    Real.sqrt ((frustum.lowerRadius - frustum.upperRadius)^2 + frustum.height^2)

theorem cone_surface_area_ratio : 
  let fullCone : RightCircularCone := ⟨8, 5⟩
  let smallerCone : RightCircularCone := ⟨3, 3 * 5 / 8⟩
  let frustum : ConeFrustum := ⟨5, 3 * 5 / 8, 5⟩
  (surfaceArea smallerCone) / (frustumSurfaceArea frustum) = 2 / 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_ratio_l332_33260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_sum_ratio_exists_l332_33241

def is_valid_sequence (y : List ℤ) : Prop :=
  ∀ i ∈ y, 0 ≤ i ∧ i ≤ 3

theorem cubic_sum_ratio_exists :
  ∃ r : ℚ, ∀ y : List ℤ,
    is_valid_sequence y →
    y.sum = 27 →
    (y.map (λ x => x^2)).sum = 135 →
    ∃ M m : ℤ,
      (∀ z : List ℤ,
        is_valid_sequence z →
        z.sum = 27 →
        (z.map (λ x => x^2)).sum = 135 →
        m ≤ (z.map (λ x => x^3)).sum ∧ (z.map (λ x => x^3)).sum ≤ M) ∧
      r = (M : ℚ) / m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_sum_ratio_exists_l332_33241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_correct_probability_at_least_one_correct_probability_exactly_three_correct_probability_l332_33212

noncomputable def prob_A : ℝ := 4/5
noncomputable def prob_B : ℝ := 3/4

-- Theorem for part (I)
theorem both_correct_probability :
  prob_A * prob_B = 3/5 := by sorry

-- Theorem for part (II)
theorem at_least_one_correct_probability :
  1 - (1 - prob_A) * (1 - prob_B) = 19/20 := by sorry

-- Theorem for part (III)
theorem exactly_three_correct_probability :
  (4 : ℝ) * (prob_A ^ 3) * (1 - prob_A) = 256/625 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_correct_probability_at_least_one_correct_probability_exactly_three_correct_probability_l332_33212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_equilateral_triangle_l332_33233

-- Define Triangle type and isEquilateral property
structure Triangle where
  -- You might want to add some fields here, e.g., vertices : Fin 3 → ℝ × ℝ

def Triangle.isEquilateral (t : Triangle) : Prop :=
  sorry -- Define what it means for a triangle to be equilateral

theorem negation_of_existence_equilateral_triangle :
  (¬∃ (t : Triangle), t.isEquilateral) ↔ (∀ (t : Triangle), ¬t.isEquilateral) :=
by
  apply Iff.intro
  · intro h t heq
    exact h ⟨t, heq⟩
  · intro h ⟨t, heq⟩
    exact h t heq

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_equilateral_triangle_l332_33233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gvatemala_encryptions_l332_33299

/-- Represents a letter in the encrypted number -/
inductive Letter
| G | V | A | T | E | M | L

/-- Represents the encrypted number as a list of letters -/
def EncryptedNumber := List Letter

/-- The encrypted number "GVATEMALA" -/
def gvatemala : EncryptedNumber :=
  [Letter.G, Letter.V, Letter.A, Letter.T, Letter.E, Letter.M, Letter.A, Letter.L, Letter.A]

/-- A function that maps letters to digits -/
def Encryption := Letter → Fin 10

/-- Checks if an encryption is valid (different letters map to different digits) -/
def isValidEncryption (e : Encryption) : Prop :=
  ∀ l1 l2 : Letter, l1 ≠ l2 → e l1 ≠ e l2

/-- Checks if a number is divisible by 8 -/
def isDivisibleBy8 (n : ℕ) : Prop := n % 8 = 0

/-- Converts an encrypted number to a natural number using the given encryption -/
def toNatural (en : EncryptedNumber) (e : Encryption) : ℕ :=
  en.foldl (fun acc l => acc * 10 + (e l).val) 0

/-- The main theorem: there are 100800 valid encryptions that result in a number divisible by 8 -/
theorem gvatemala_encryptions :
  ∃ s : Finset Encryption, 
    (∀ e ∈ s, isValidEncryption e ∧ isDivisibleBy8 (toNatural gvatemala e)) ∧
    s.card = 100800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gvatemala_encryptions_l332_33299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l332_33259

noncomputable def average_speed (distance1 : ℝ) (distance2 : ℝ) (time1 : ℝ) (time2 : ℝ) : ℝ :=
  (distance1 + distance2) / (time1 + time2)

theorem car_average_speed :
  let distance1 : ℝ := 100
  let distance2 : ℝ := 60
  let time1 : ℝ := 1
  let time2 : ℝ := 1
  average_speed distance1 distance2 time1 time2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l332_33259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_l332_33267

theorem undefined_values (a : ℝ) : 
  (a^2 + 2*a + 1)/(a^2 - 9) = 0/0 ↔ a = -3 ∨ a = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_l332_33267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_polynomial_roots_l332_33266

-- Define a polynomial with non-negative coefficients
def NonNegativePolynomial (n : ℕ) := { p : Polynomial ℝ // p.degree = n ∧ ∀ i, p.coeff i ≥ 0 }

-- Define an acute triangle
def IsAcuteTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a < b + c ∧ b < a + c ∧ c < a + b ∧
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- The main theorem
theorem acute_triangle_polynomial_roots
  (n : ℕ) (hn : n ≥ 2)
  (P : NonNegativePolynomial n)
  (a b c : ℝ) (habc : IsAcuteTriangle a b c) :
  IsAcuteTriangle ((P.val.eval a)^(1/n : ℝ)) ((P.val.eval b)^(1/n : ℝ)) ((P.val.eval c)^(1/n : ℝ)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_polynomial_roots_l332_33266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pentagons_for_2011gon_l332_33236

theorem min_pentagons_for_2011gon : ℕ := by
  -- Define the number of sides in the target polygon
  let n : ℕ := 2011

  -- Define the number of sides in a pentagon
  let pentagon_sides : ℕ := 5

  -- Function to calculate the sum of interior angles of a polygon
  let sum_interior_angles (sides : ℕ) : ℕ := (sides - 2) * 180

  -- Calculate the number of pentagons needed
  let pentagons_needed : ℕ := 
    (sum_interior_angles n + sum_interior_angles pentagon_sides - 1) / sum_interior_angles pentagon_sides

  -- The theorem statement
  have : pentagons_needed = 670 := by sorry

  exact pentagons_needed


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pentagons_for_2011gon_l332_33236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_value_l332_33235

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1
def g (x : ℝ) : ℝ := -x^2 + 4*x - 3

-- State the theorem
theorem max_b_value (a b : ℝ) (ha : 0 ≤ a) (h : f a = g b) :
  b ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_value_l332_33235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_y_axis_l332_33222

/-- A line passing through two points (2, 8) and (4, 12) intersects the y-axis at (0, 4) -/
theorem line_intersection_y_axis :
  let p1 : ℝ × ℝ := (2, 8)
  let p2 : ℝ × ℝ := (4, 12)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let line : ℝ → ℝ := λ x ↦ m * x + b
  (0, line 0) = (0, 4) := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_y_axis_l332_33222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_one_minus_sin_squared_l332_33208

theorem sqrt_one_minus_sin_squared (α : ℝ) : 
  Real.sqrt (1 - Real.sin α ^ 2) = abs (Real.cos α) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_one_minus_sin_squared_l332_33208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rain_probability_theorem_l332_33285

/-- Represents a day's weather condition -/
inductive Weather
  | Rain
  | NoRain
deriving Repr, DecidableEq

/-- Converts a random number to a weather condition -/
def numberToWeather (n : Nat) : Weather :=
  if n ≤ 4 then Weather.Rain else Weather.NoRain

/-- Represents three days of weather -/
structure ThreeDayWeather where
  day1 : Weather
  day2 : Weather
  day3 : Weather
deriving Repr

/-- Counts the number of rainy days in a three-day period -/
def countRainyDays (tdw : ThreeDayWeather) : Nat :=
  (if tdw.day1 = Weather.Rain then 1 else 0) +
  (if tdw.day2 = Weather.Rain then 1 else 0) +
  (if tdw.day3 = Weather.Rain then 1 else 0)

/-- Generates a ThreeDayWeather from three random numbers -/
def generateThreeDayWeather (n1 n2 n3 : Nat) : ThreeDayWeather where
  day1 := numberToWeather n1
  day2 := numberToWeather n2
  day3 := numberToWeather n3

/-- The probability of rain on any given day -/
def probRainOneDay : Real := 0.4

/-- The probability of having exactly two days of rain in three days -/
def probTwoRainyDays : Real := 0.25

theorem rain_probability_theorem :
  (∀ n : Nat, n < 10 → (numberToWeather n = Weather.Rain) = (n ≤ 4)) →
  probRainOneDay = 0.4 ∧
  probTwoRainyDays = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rain_probability_theorem_l332_33285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_in_metal_bar_l332_33204

/-- Represents the properties of a metal bar composed of tin and silver -/
structure MetalBar where
  total_weight : ℝ
  water_loss : ℝ
  tin_loss_rate : ℝ
  silver_loss_rate : ℝ
  tin_silver_ratio : ℝ

/-- Calculates the amount of tin in the metal bar -/
noncomputable def tin_amount (bar : MetalBar) : ℝ :=
  let silver_amount := bar.total_weight / (1 + bar.tin_silver_ratio)
  bar.tin_silver_ratio * silver_amount

/-- Theorem stating the amount of tin in the specific metal bar -/
theorem tin_in_metal_bar :
  let bar : MetalBar := {
    total_weight := 50,
    water_loss := 5,
    tin_loss_rate := 1.375,
    silver_loss_rate := 0.375 / 5,
    tin_silver_ratio := 2 / 3
  }
  ∃ ε > 0, |tin_amount bar - 3.361| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_in_metal_bar_l332_33204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base9_addition_l332_33289

/-- Represents a number in base 9 --/
structure Base9 where
  value : Nat
  isValid : value < 9^64 := by sorry

/-- Converts a base 9 number to its decimal (base 10) representation --/
def to_decimal (n : Base9) : Nat :=
  n.value

/-- Converts a decimal (base 10) number to its base 9 representation --/
def to_base9 (n : Nat) : Base9 :=
  ⟨n % 9^64, by sorry⟩

/-- Addition operation for base 9 numbers --/
def add_base9 (a b : Base9) : Base9 :=
  to_base9 (to_decimal a + to_decimal b)

/-- Helper function to create a Base9 number from a Nat --/
def mk_base9 (n : Nat) : Base9 :=
  to_base9 n

theorem base9_addition :
  add_base9 (add_base9 (mk_base9 263) (mk_base9 452)) (mk_base9 247) = mk_base9 1073 :=
by sorry

#eval to_decimal (add_base9 (add_base9 (mk_base9 263) (mk_base9 452)) (mk_base9 247))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base9_addition_l332_33289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_value_c_value_l332_33283

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
noncomputable def triangle_ABC : Triangle := {
  a := 6
  b := 0  -- b is not given, so we set it to 0 for now
  c := 0  -- c will be defined differently in each theorem
  A := 0  -- A is not given, so we set it to 0 for now
  B := 0  -- B is not given, so we set it to 0 for now
  C := 2 * Real.pi / 3
}

-- Theorem for part (I)
theorem sin_A_value (t : Triangle) (h : t = { triangle_ABC with c := 14 }) :
  Real.sin t.A = (3 * Real.sqrt 3) / 14 := by
  sorry

-- Theorem for part (II)
theorem c_value (t : Triangle) (h : t.a * t.b * Real.sin t.C / 2 = 3 * Real.sqrt 3) :
  t.c = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_value_c_value_l332_33283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_correct_l332_33286

/-- Represents a person in the company -/
structure Person where
  id : Nat
deriving Repr, DecidableEq

/-- Represents the company -/
structure Company where
  people : Finset Person
  n : Nat
  h_n : people.card = n
  z : Person
  h_z : z ∈ people
  h_z_knows_all : ∀ p ∈ people, p ≠ z → z.id ≠ p.id
  h_nobody_knows_z : ∀ p ∈ people, p ≠ z → p.id ≠ z.id

/-- Represents a question asked by the journalist -/
inductive Question where
  | ask : Person → Person → Question

/-- The result of asking a question -/
def ask_question (c : Company) (q : Question) : Bool :=
  match q with
  | Question.ask a b => 
    if a = c.z then true
    else if b = c.z then false
    else a.id = b.id

/-- The minimum number of questions required to identify Z -/
def min_questions (c : Company) : Nat := c.n - 1

/-- Theorem: The minimum number of questions required to identify Z is n-1 -/
theorem min_questions_correct (c : Company) : 
  ∀ k : Nat, k < min_questions c → 
  ∃ c1 c2 : Company, c1.n = c.n ∧ c2.n = c.n ∧ c1.z ≠ c2.z ∧
  ∀ qs : Finset Question, qs.card = k → 
  (∀ q ∈ qs, ask_question c1 q = ask_question c2 q) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_correct_l332_33286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_rate_10_to_40_l332_33275

/-- Represents a class interval with its frequency -/
structure ClassInterval where
  lower : ℚ
  upper : ℚ
  frequency : ℕ

/-- The sample data -/
def sampleData : List ClassInterval := [
  ⟨10, 20, 2⟩,
  ⟨20, 30, 4⟩,
  ⟨30, 40, 3⟩,
  ⟨40, 50, 5⟩,
  ⟨50, 60, 4⟩,
  ⟨60, 70, 2⟩
]

/-- The total sample size -/
def sampleSize : ℕ := 20

/-- Calculates the frequency within a given range -/
def frequencyInRange (data : List ClassInterval) (lower upper : ℚ) : ℕ :=
  data.filter (fun ci => ci.lower > lower ∧ ci.upper ≤ upper)
    |>.map (fun ci => ci.frequency)
    |>.sum

/-- Theorem: The frequency rate in the interval (10, 40] is 9/20 -/
theorem frequency_rate_10_to_40 :
  (frequencyInRange sampleData 10 40 : ℚ) / sampleSize = 9 / 20 := by
  sorry

#eval frequencyInRange sampleData 10 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_rate_10_to_40_l332_33275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_radius_sum_l332_33221

/-- The equation of circle C is x^2 + 4y - 16 = -y^2 + 8x + 16 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 4*y - 16 = -y^2 + 8*x + 16

/-- The center of the circle is (a, b) -/
def is_center (a b r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

/-- The radius of the circle is r -/
def is_radius (r : ℝ) : Prop :=
  ∃ a b : ℝ, is_center a b r ∧ r > 0

theorem circle_center_radius_sum :
  ∃ a b r : ℝ, is_center a b r ∧ is_radius r ∧ a + b + r = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_radius_sum_l332_33221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_exceeding_10_pow_100_l332_33295

def c : ℕ → ℕ
  | 0 => 3  -- Add this case to handle Nat.zero
  | 1 => 3
  | 2 => 6
  | (n + 3) => c (n + 2) * c (n + 1)

theorem smallest_n_exceeding_10_pow_100 :
  ∀ n : ℕ, n < 45 → c n ≤ 10^100 ∧ c 45 > 10^100 := by
  sorry

#eval c 45  -- Optional: to check the value of c 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_exceeding_10_pow_100_l332_33295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_P_tangent_through_P_l332_33252

/-- The given curve function -/
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + 4/3

/-- Tangent line equation at a point (a, f a) on the curve -/
def tangent_line_at (a : ℝ) (x y : ℝ) : Prop :=
  y - f a = (deriv f a) * (x - a)

/-- Point P lies on the curve -/
axiom point_on_curve : f 2 = 4

/-- Theorem for the tangent line equation at P(2,4) -/
theorem tangent_at_P : 
  ∀ x y : ℝ, tangent_line_at 2 x y ↔ 4*x - y - 4 = 0 := by
  sorry

/-- Theorem for the tangent lines passing through P(2,4) -/
theorem tangent_through_P :
  ∀ x y : ℝ, (∃ a : ℝ, tangent_line_at a x y ∧ y = f a) ∧ x = 2 ∧ y = 4 ↔ 
  (4*x - y - 4 = 0) ∨ (x - y + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_P_tangent_through_P_l332_33252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_cubic_function_coefficients_l332_33254

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_cubic_function_coefficients
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h1 : f = λ x ↦ a * x^3 + b * x^2 + c)
  (h2 : is_odd_function f) :
  b = 0 ∧ c = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_cubic_function_coefficients_l332_33254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_nine_l332_33269

-- Define the line function
def line (x : ℝ) : ℝ := 2 * x

-- Define the curve function
def curve (x : ℝ) : ℝ := 4 - 2 * x^2

-- Define the area function
noncomputable def area : ℝ := ∫ x in Set.Icc (-2) 1, curve x - line x

-- Theorem statement
theorem enclosed_area_is_nine : area = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_nine_l332_33269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_subset_size_for_divisible_chain_l332_33213

/-- Given a set M of consecutive positive integers from 1 to 2^m * n,
    this theorem states the minimum size of a subset that guarantees
    the existence of m+1 numbers forming a divisible chain. -/
theorem min_subset_size_for_divisible_chain (m n : ℕ+) :
  let M := Finset.range (2^m.val * n.val + 1)
  ∃ (k : ℕ), k = (2^m.val - 1) * n.val + 1 ∧
    ∀ (S : Finset ℕ), S ⊆ M → S.card = k →
      ∃ (chain : List ℕ), chain.toFinset ⊆ S ∧ chain.length = m.val + 1 ∧
        ∀ (i : Fin m.val), chain[i.val]?.isSome ∧ chain[i.val + 1]?.isSome →
          (chain[i.val + 1]?.get! % chain[i.val]?.get! = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_subset_size_for_divisible_chain_l332_33213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slowly_increasing_interval_l332_33209

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - x + 3/2

-- Define what it means for a function to be increasing on an interval
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

-- Define what it means for a function to be decreasing on an interval
def IsDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x

-- Define what it means for a function to be slowly increasing on an interval
def IsSlowlyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  IsIncreasing f a b ∧ IsDecreasing (fun x => f x / x) a b

-- State the theorem
theorem slowly_increasing_interval :
  IsSlowlyIncreasing f 1 (Real.sqrt 3) := by
  sorry

#check slowly_increasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slowly_increasing_interval_l332_33209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_sum_approx_l332_33218

-- Define constants
noncomputable def meters_to_feet : ℝ := 3.28084
noncomputable def pyramid_height_meters : ℝ := 170
noncomputable def slope_tangent : ℝ := 4/3

-- Define the sum of height and base length in feet
noncomputable def pyramid_sum : ℝ :=
  let height_feet := pyramid_height_meters * meters_to_feet
  let base_length_feet := 2 * height_feet / slope_tangent
  height_feet + base_length_feet

-- Theorem statement
theorem pyramid_sum_approx :
  |pyramid_sum - 1394.357| < 0.001 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_sum_approx_l332_33218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_geometric_subseq_l332_33217

/-- An arithmetic sequence contains an infinite geometric subsequence iff a/b is rational -/
theorem arithmetic_seq_geometric_subseq (a b : ℝ) (hb : b ≠ 0) :
  (∃ (f : ℕ → ℕ) (k : ℝ), StrictMono f ∧ k ≠ 1 ∧
    ∀ n, a + (f n) * b = (a + (f 0) * b) * k^n) ↔
  ∃ (p q : ℤ), q ≠ 0 ∧ a / b = p / q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_geometric_subseq_l332_33217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_ratio_of_equal_volume_cylinders_l332_33272

/-- Represents a right circular cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The volume of a cylinder --/
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

theorem height_ratio_of_equal_volume_cylinders (c1 c2 : Cylinder) 
  (h_vol : volume c1 = volume c2) 
  (h_radius : c2.radius = 1.2 * c1.radius) : 
  c1.height = 1.44 * c2.height := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_ratio_of_equal_volume_cylinders_l332_33272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_function_l332_33215

theorem max_value_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, ∀ y ∈ Set.Icc (-1 : ℝ) 2, 2 * a^x - 5 ≥ 2 * a^y - 5) ∧ 
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, 2 * a^x - 5 = 10) →
  a = 2/15 ∨ a = Real.sqrt 30 / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_function_l332_33215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_x_y_l332_33284

theorem existence_of_x_y (k : ℕ) : ∃ x y : ℤ, 
  (x^2 + 2*y^2 = 3^k) ∧ 
  (x % 3 ≠ 0) ∧ 
  (y % 3 ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_x_y_l332_33284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_naturals_l332_33210

theorem partition_naturals (c : ℚ) (hc : c > 0) (hc1 : c ≠ 1) :
  ∃ (A B : Set ℕ), 
    (A ∪ B = Set.univ) ∧ 
    (A ∩ B = ∅) ∧
    (∀ a b, a ∈ A → b ∈ A → (a : ℚ) / b ≠ c) ∧
    (∀ a b, a ∈ B → b ∈ B → (a : ℚ) / b ≠ c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_naturals_l332_33210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_theorem_l332_33230

-- Define the rectangle ABCD
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the properties of the rectangle
def is_valid_rectangle (rect : Rectangle) : Prop :=
  ∃ (m n : ℝ), 
    rect.A.1 - rect.B.1 = m ∧ rect.A.2 - rect.B.2 = n ∧
    rect.C.1 - rect.D.1 = m ∧ rect.C.2 - rect.D.2 = n ∧
    rect.A.1 - rect.D.1 = n ∧ rect.A.2 - rect.D.2 = -m ∧
    rect.B.1 - rect.C.1 = n ∧ rect.B.2 - rect.C.2 = -m

-- Define the diagonal property
def diagonal_relative_to_point (rect : Rectangle) (p : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), rect.C.1 - rect.A.1 = k * (rect.A.1 - p.1) ∧
              rect.C.2 - rect.A.2 = k * (rect.A.2 - p.2)

-- Define the equation of line AB
def line_AB_equation (rect : Rectangle) : Prop :=
  ∀ (x y : ℝ), (x = rect.A.1 ∧ y = rect.A.2) ∨ (x = rect.B.1 ∧ y = rect.B.2) →
    x - 2*y - 2 = 0

-- Define the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 9

-- Define the theorem
theorem chord_length_theorem (rect : Rectangle) :
  is_valid_rectangle rect →
  diagonal_relative_to_point rect (0, 1) →
  line_AB_equation rect →
  ∃ (x1 y1 x2 y2 : ℝ),
    circle_equation x1 y1 ∧ circle_equation x2 y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 16 ∧
    (∃ (m : ℝ), x1 - 2*y1 + m = 0 ∧ x2 - 2*y2 + m = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_theorem_l332_33230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_selling_price_l332_33288

/-- The selling price that results in a 15% profit -/
noncomputable def profit_price : ℝ := 15.33

/-- The percentage of cost price at which the mangoes are currently sold (90% due to 10% loss) -/
noncomputable def current_sale_percentage : ℝ := 0.90

/-- The percentage of cost price that would result in a 15% profit -/
noncomputable def profit_percentage : ℝ := 1.15

/-- The actual selling price of mangoes per kg -/
noncomputable def selling_price : ℝ := current_sale_percentage * (profit_price / profit_percentage)

/-- Theorem stating that the selling price is approximately 12 -/
theorem mango_selling_price : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |selling_price - 12| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_selling_price_l332_33288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l332_33234

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := ⌊x⌋ - x + 1

-- State the theorem about the range of g
theorem range_of_g :
  ∀ y : ℝ, (∃ x : ℝ, g x = y) ↔ (0 ≤ y ∧ y < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l332_33234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_squared_for_given_cones_l332_33237

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones -/
structure IntersectingCones where
  cone : Cone
  intersectionDistance : ℝ

/-- Calculates the maximum possible squared radius of a sphere fitting within the intersecting cones -/
noncomputable def maxSphereRadiusSquared (ic : IntersectingCones) : ℝ :=
  (ic.cone.baseRadius * ic.intersectionDistance / 
   Real.sqrt (ic.cone.height^2 + ic.cone.baseRadius^2))^2

theorem max_sphere_radius_squared_for_given_cones :
  let ic : IntersectingCones := {
    cone := { baseRadius := 5, height := 12 },
    intersectionDistance := 4
  }
  maxSphereRadiusSquared ic = 400 / 169 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and might cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_squared_for_given_cones_l332_33237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_equation_of_C_chord_length_l332_33240

-- Define the polar equation of curve C
noncomputable def polar_equation (ρ θ : ℝ) : Prop := ρ^2 * Real.cos (2 * θ) = 1

-- Define the parametric equations of line l
noncomputable def line_equation (t : ℝ) : ℝ × ℝ := (2 + (1/2) * t, (Real.sqrt 3 / 2) * t)

-- Standard equation of curve C
def standard_equation (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Theorem 1: The standard equation of curve C is x² - y² = 1
theorem standard_equation_of_C :
  (∀ ρ θ : ℝ, polar_equation ρ θ ↔ standard_equation (ρ * Real.cos θ) (ρ * Real.sin θ)) := by
  sorry

-- Theorem 2: The length of the chord cut from curve C by line l is 2√10
theorem chord_length :
  ∃ t₁ t₂ : ℝ,
    let (x₁, y₁) := line_equation t₁
    let (x₂, y₂) := line_equation t₂
    standard_equation x₁ y₁ ∧ standard_equation x₂ y₂ ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_equation_of_C_chord_length_l332_33240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_alpha_value_l332_33291

noncomputable def f (x : ℝ) (α : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + Real.sin x
  else -x^2 + Real.cos (x + α)

theorem odd_function_alpha_value (α : ℝ) 
  (h1 : α ≥ 0) (h2 : α < 2 * Real.pi) 
  (h3 : ∀ x, f x α = -f (-x) α) : α = 3 * Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_alpha_value_l332_33291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_ABC_is_19_l332_33255

-- Define the points in the plane
variable (A B C Q : ℝ × ℝ)

-- Define the distances
def QA : ℝ := 3
def QB : ℝ := 4
def QC : ℝ := 5
def BC : ℝ := 6

-- Define the function to calculate the area of a triangle
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define the function to calculate the maximum area of triangle ABC
noncomputable def max_area_ABC (A B C Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_area_ABC_is_19 (A B C Q : ℝ × ℝ) :
  max_area_ABC A B C Q = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_ABC_is_19_l332_33255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_36_l332_33248

theorem probability_factor_of_36 : 
  let n : ℕ := 36
  let factors := Finset.filter (fun k => n % k = 0) (Finset.range (n + 1))
  (factors.card : ℚ) / n = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_36_l332_33248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_squared_is_identity_l332_33243

/-- A reflection matrix over a non-zero vector -/
def reflection_matrix (v : Fin 2 → ℝ) : Matrix (Fin 2) (Fin 2) ℝ := sorry

/-- The identity matrix -/
def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0],
    ![0, 1]]

/-- Theorem: The square of a reflection matrix is the identity matrix -/
theorem reflection_squared_is_identity (v : Fin 2 → ℝ) (h : v ≠ 0) :
  (reflection_matrix v) ^ 2 = identity_matrix :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_squared_is_identity_l332_33243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_problem_l332_33276

/-- The function we're optimizing -/
def f (x y : ℝ) : ℝ := |x^2 - 2*x*y|

/-- The set of x values we're considering -/
def X : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

/-- The maximum value of f(x,y) for a fixed y and x in X -/
noncomputable def g (y : ℝ) : ℝ := ⨆ (x : X), f x.1 y

/-- The statement of the optimization problem -/
theorem min_max_problem : ∃ (y : ℝ), g y = 0 ∧ ∀ (y' : ℝ), g y' ≥ 0 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_problem_l332_33276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l332_33256

/-- The eccentricity of a hyperbola with equation x²/a² - y² = 1 (a > 0),
    given that one of its asymptotes has an angle of inclination of 30°. -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 = 1) →
  (∃ (m : ℝ), m = Real.tan (30 * π / 180) ∧ m = 1 / a) →
  Real.sqrt (a^2 + 1) / a = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l332_33256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_is_two_l332_33219

/-- A hyperbola with given parameters and properties -/
structure Hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) where
  /-- The focus of the hyperbola -/
  F : ℝ × ℝ
  /-- The endpoint of the imaginary axis -/
  B : ℝ × ℝ
  /-- The point where BF intersects the asymptote -/
  A : ℝ × ℝ
  /-- The equation of the hyperbola -/
  eq : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1
  /-- B is on the imaginary axis -/
  hB : B.1 = 0 ∧ B.2 = b
  /-- A is on the asymptote -/
  hA : A.2 = (b / a) * A.1
  /-- Vector FA is twice vector AB -/
  hFA_AB : (F.1 - A.1, F.2 - A.2) = (2 * (A.1 - B.1), 2 * (A.2 - B.2))

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : Hyperbola a b ha hb) : ℝ :=
  Real.sqrt ((h.F.1)^2 + (h.F.2)^2) / a

/-- The theorem stating that the eccentricity is 2 under given conditions -/
theorem eccentricity_is_two (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : Hyperbola a b ha hb) : eccentricity a b ha hb h = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_is_two_l332_33219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_graph_has_triangle_l332_33225

/-- A graph with 21 vertices and 102 edges, where the edges form a cycle of odd length. -/
structure CycleGraph where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  vertex_count : vertices.card = 21
  edge_count : edges.card = 102
  is_cycle : ∃ m : ℕ, Odd m ∧ 
    (∀ i ∈ vertices, (i, (i + 1) % 21) ∈ edges) ∧
    (m - 1, 0) ∈ edges

/-- A triangle in a graph is a set of three vertices, each connected to the other two. -/
def HasTriangle (g : CycleGraph) : Prop :=
  ∃ a b c, a ∈ g.vertices ∧ b ∈ g.vertices ∧ c ∈ g.vertices ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a, b) ∈ g.edges ∧ (b, c) ∈ g.edges ∧ (c, a) ∈ g.edges

/-- Theorem: Every CycleGraph contains a triangle. -/
theorem cycle_graph_has_triangle (g : CycleGraph) : HasTriangle g := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_graph_has_triangle_l332_33225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_roots_of_quartic_equations_l332_33278

/-- The set of equations x^4 + a_3x^3 + a_2x^2 + a_1x + a_0 = 0 -/
def QuarticEquation (a₃ a₂ a₁ a₀ : ℝ) (x : ℝ) : Prop :=
  x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀ = 0

/-- The condition that |a_i| < 3 for i = 0, 1, 2, 3 -/
def CoefficientBound (a₃ a₂ a₁ a₀ : ℝ) : Prop :=
  (|a₃| < 3) ∧ (|a₂| < 3) ∧ (|a₁| < 3) ∧ (|a₀| < 3)

/-- The largest positive and negative roots of the quartic equations -/
theorem largest_roots_of_quartic_equations :
  ∃ (r s : ℝ),
    (r > 0 ∧ s < 0) ∧
    (∀ (a₃ a₂ a₁ a₀ x : ℝ),
      CoefficientBound a₃ a₂ a₁ a₀ →
      QuarticEquation a₃ a₂ a₁ a₀ x →
      (x > 0 → x ≤ r) ∧ (x < 0 → x ≥ s)) ∧
    (abs (r - 4) < 0.1 ∧ abs (s + 2) < 0.1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_roots_of_quartic_equations_l332_33278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_comparison_l332_33250

-- Define the ionic product as a function of temperature
def Kw (t : ℝ) : ℝ := sorry

-- Define the reference temperature (25°C)
def t_ref : ℝ := 25

-- State the properties of Kw
axiom Kw_increasing : ∀ t₁ t₂ : ℝ, t₁ < t₂ → Kw t₁ < Kw t₂
axiom Kw_at_25 : Kw t_ref = 10^(-14 : ℤ)

-- State the theorem
theorem temperature_comparison (t : ℝ) (h : Kw t = 10^(-12 : ℤ)) : t > t_ref := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_comparison_l332_33250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_integral_l332_33271

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the integrand
def integrand (x y : ℝ) : ℝ × ℝ := (x^2 - 2*x*y, y^2 - 2*x*y)

-- State the theorem
theorem parabola_line_integral :
  let a := -1
  let b := 1
  ∫ x in a..b, (integrand x (parabola x)).1 + (integrand x (parabola x)).2 * (deriv parabola x) = -14/15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_integral_l332_33271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3825_deg_equals_one_l332_33264

-- Define the degree-to-radian conversion factor
noncomputable def deg_to_rad : ℝ := Real.pi / 180

-- Define the tangent function for degrees
noncomputable def tan_deg (x : ℝ) : ℝ := Real.tan (x * deg_to_rad)

-- State the theorem
theorem tan_3825_deg_equals_one :
  tan_deg 3825 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3825_deg_equals_one_l332_33264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_length_triangle_area_l332_33206

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  BC : Real
  AC : Real

-- Define the given triangle
noncomputable def triangle_ABC : Triangle where
  A := Real.pi - (Real.pi / 4 + Real.arccos (3/5))
  B := Real.pi / 4
  C := Real.arccos (3/5)
  AB := 8  -- We now know this value
  BC := 0  -- Not given and not needed for the proof
  AC := 5 * Real.sqrt 2

-- Theorem for the length of AB
theorem AB_length (t : Triangle) (h1 : t = triangle_ABC) : t.AB = 8 := by
  sorry

-- Theorem for the area of triangle ABC
theorem triangle_area (t : Triangle) (h1 : t = triangle_ABC) : 
  (1/2) * t.AB * t.AC * Real.sin t.A = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_length_triangle_area_l332_33206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l332_33287

-- Define the ellipse parameters
variable (a b : ℝ)

-- Define the conditions
def ellipse_equation (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def major_axis_condition : Prop := a = 2 * b
def positive_parameters : Prop := a > 0 ∧ b > 0 ∧ a > b

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define the line passing through (0,2)
def line_equation (k x : ℝ) : ℝ := k * x + 2

-- Define the area of triangle OMN
noncomputable def triangle_area (k : ℝ) : ℝ := 8 * abs k / (1 + 4 * k^2)

-- State the theorem
theorem ellipse_properties :
  positive_parameters a b →
  major_axis_condition a b →
  (eccentricity a b = Real.sqrt 3 / 2) ∧
  (∃ k : ℝ, k ≠ 0 ∧
    (∀ x y : ℝ, ellipse_equation a b x y ∧ y = line_equation k x →
      x^2 / 8 + y^2 / 2 = 1) ∧
    (∀ k' : ℝ, triangle_area k' ≤ triangle_area k)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l332_33287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_angle_is_pi_over_three_l332_33207

noncomputable section

-- Define the points
def A : ℝ × ℝ := (-2, Real.sqrt 3)
def C : ℝ × ℝ := (1, 2 * Real.sqrt 3)

-- Define that B is on the x-axis
def B_on_x_axis (B : ℝ × ℝ) : Prop := B.2 = 0

-- Define the reflection property
def reflects_off (A B C : ℝ × ℝ) : Prop :=
  ∃ A' : ℝ × ℝ, A'.1 = A.1 ∧ A'.2 = -A.2 ∧ (B.2 - A'.2) / (B.1 - A'.1) = (C.2 - B.2) / (C.1 - B.1)

-- Define the angle of inclination
def angle_of_inclination (B C : ℝ × ℝ) : ℝ := Real.arctan ((C.2 - B.2) / (C.1 - B.1))

-- The theorem to prove
theorem reflection_angle_is_pi_over_three (B : ℝ × ℝ) 
  (h1 : B_on_x_axis B) (h2 : reflects_off A B C) : 
  angle_of_inclination B C = π / 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_angle_is_pi_over_three_l332_33207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_divisibility_theorem_l332_33211

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def divisible_by_3 (n : ℕ) : Bool := n % 3 = 0
def divisible_by_5 (n : ℕ) : Bool := n % 5 = 0
def divisible_by_9 (n : ℕ) : Bool := n % 9 = 0
def divisible_by_15 (n : ℕ) : Bool := n % 15 = 0
def divisible_by_25 (n : ℕ) : Bool := n % 25 = 0
def divisible_by_45 (n : ℕ) : Bool := n % 45 = 0

def exactly_three_true (n : ℕ) : Prop :=
  let conditions := [
    divisible_by_3 n,
    divisible_by_5 n,
    divisible_by_9 n,
    divisible_by_15 n,
    divisible_by_25 n,
    divisible_by_45 n
  ]
  (conditions.filter id).length = 3

theorem two_digit_divisibility_theorem :
  ∀ x : ℕ, is_two_digit x → (exactly_three_true x ↔ (x = 15 ∨ x = 30 ∨ x = 60)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_divisibility_theorem_l332_33211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_30_parts_x_values_l332_33261

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the integer part function
noncomputable def intPart (x : ℝ) : ℤ := Int.floor x

-- Define the decimal part function
noncomputable def decPart (x : ℝ) : ℝ := x - Int.floor x

-- Theorem for part (1)
theorem cube_root_30_parts :
  (intPart (cubeRoot 30) = 3) ∧ (decPart (cubeRoot 30) = cubeRoot 30 - 3) := by
  sorry

-- Theorem for part (2)
theorem x_values (m : ℤ) (h : m = intPart (7 - cubeRoot 20)) :
  ∃ x : ℝ, (x + 1)^2 = m ∧ (x = 1 ∨ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_30_parts_x_values_l332_33261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survival_curve_represents_death_process_not_all_populations_have_all_age_groups_l332_33242

/-- Represents a population of organisms -/
structure Population where
  individuals : Set Organism

/-- Represents a community of populations -/
structure Community where
  populations : Set Population

/-- Represents the survival curve of a population -/
def survival_curve (p : Population) : ℝ → ℝ := sorry

/-- Represents the death process of a population -/
def death_process (p : Population) : ℝ → ℝ := sorry

/-- Theorem: The survival curve of a population represents the death process and death situation of all individuals in the population -/
theorem survival_curve_represents_death_process (p : Population) :
  survival_curve p = death_process p := by sorry

/-- Populations can have different spatial distributions -/
inductive SpatialDistribution
  | Uniform
  | Random
  | Clumped

/-- A population has a spatial distribution -/
def population_distribution (p : Population) : SpatialDistribution := sorry

/-- Communities have a spatial configuration pattern -/
def community_spatial_pattern (c : Community) : Set (ℝ × ℝ) := sorry

/-- Represents the age structure of a population -/
inductive AgeGroup
  | PreReproductive
  | Reproductive
  | PostReproductive

/-- A population may have different age groups -/
def population_age_groups (p : Population) : Set AgeGroup := sorry

/-- Not all populations necessarily have all three age groups -/
theorem not_all_populations_have_all_age_groups :
  ∃ p : Population, population_age_groups p ≠ {AgeGroup.PreReproductive, AgeGroup.Reproductive, AgeGroup.PostReproductive} := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_survival_curve_represents_death_process_not_all_populations_have_all_age_groups_l332_33242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_must_be_80_l332_33224

def scores : List ℤ := [71, 76, 80, 82, 91]

def is_integer_average (partial_scores : List ℤ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ (partial_scores.sum : ℚ) / n = ↑(partial_scores.sum / n)

def all_partial_averages_integer (scores : List ℤ) : Prop :=
  ∀ k : ℕ, k > 0 → k ≤ scores.length → is_integer_average (scores.take k)

theorem last_score_must_be_80 (h : all_partial_averages_integer scores) :
  scores.reverse.head? = some 80 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_must_be_80_l332_33224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l332_33293

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0
def circle_O₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define the centers of the circles
def center_O₁ : ℝ × ℝ := (1, 0)
def center_O₂ : ℝ × ℝ := (0, 2)

-- Define the radii of the circles
def radius_O₁ : ℝ := 1
def radius_O₂ : ℝ := 2

-- Define the distance between the centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 5

-- Theorem stating that the circles intersect
theorem circles_intersect :
  distance_between_centers > |radius_O₁ - radius_O₂| ∧
  distance_between_centers < radius_O₁ + radius_O₂ := by
  apply And.intro
  · -- Prove distance_between_centers > |radius_O₁ - radius_O₂|
    sorry
  · -- Prove distance_between_centers < radius_O₁ + radius_O₂
    sorry

#check circles_intersect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l332_33293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l332_33265

/-- The distance between a point (x₀, y₀) and a line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  (|A * x₀ + B * y₀ + C|) / Real.sqrt (A^2 + B^2)

/-- The equation of a circle with center (h, k) and radius r -/
def circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_tangent_to_line : 
  let h := 1
  let k := 2
  let A := 2
  let B := 1
  let C := 1
  let r := distance_point_to_line h k A B C
  ∀ x y : ℝ, circle_equation x y h k r ↔ (x - h)^2 + (y - k)^2 = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l332_33265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l332_33226

-- Define the propositions p, q, and r
def p (x y : ℝ) : Prop := (x - 2) * (y - 5) ≠ 0
def q (x y : ℝ) : Prop := x ≠ 2 ∨ y ≠ 5
def r (x y : ℝ) : Prop := x + y ≠ 7

-- Define what it means for one proposition to be a sufficient condition for another
def is_sufficient (P Q : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, P x y → Q x y

-- Define what it means for one proposition to be a necessary condition for another
def is_necessary (P Q : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, Q x y → P x y

-- State the theorem
theorem problem_statement :
  (¬ is_sufficient p r ∧ ¬ is_necessary p r) ∧
  (is_sufficient p q ∧ ¬ is_necessary p q) ∧
  (is_necessary q r ∧ ¬ is_sufficient q r) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l332_33226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_one_two_l332_33239

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem statement
theorem tangent_at_one_two :
  ∃ (m : ℝ), 
    (f 1 = 2) ∧ 
    (∀ (h : ℝ), h ≠ 0 → (((f (1 + h) - f 1) / h) = m)) ∧
    (∀ (x y : ℝ), y = m * (x - 1) + 2 ↔ tangent_line x y) := by
  sorry

#check tangent_at_one_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_one_two_l332_33239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characteristic_not_determine_degree_l332_33279

/-- A characteristic of a polynomial -/
def A_P (P : Polynomial ℝ) : Set ℝ := sorry

/-- Theorem stating that the characteristic A_P does not uniquely determine the degree of a polynomial -/
theorem characteristic_not_determine_degree :
  ∃ (P1 P2 : Polynomial ℝ), Polynomial.degree P1 ≠ Polynomial.degree P2 ∧ A_P P1 = A_P P2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characteristic_not_determine_degree_l332_33279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_attendees_l332_33245

theorem conference_attendees (k : ℕ+) 
  (total_attendees : ℕ)
  (known_per_person : ℕ)
  (h_total : total_attendees = 12 * k)
  (h_known : known_per_person = 3 * k + 6)
  (h : ∀ (i j : ℕ), i ≠ j → i ≤ total_attendees → j ≤ total_attendees → 
    ∃ (m : ℕ), (known_per_person.choose 2) * total_attendees = m * (total_attendees.choose 2)) :
  k = 3 ∧ total_attendees = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_attendees_l332_33245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l332_33244

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where c = 2 and sin²A + sin²B = sin A sin B + sin²C,
    the maximum area of the triangle is √3. -/
theorem max_triangle_area (a b c A B C : ℝ) : 
  c = 2 →
  Real.sin A ^ 2 + Real.sin B ^ 2 = Real.sin A * Real.sin B + Real.sin C ^ 2 →
  (∃ (S : ℝ), S = (1/2) * a * b * Real.sin C ∧ 
    ∀ (S' : ℝ), S' = (1/2) * a * b * Real.sin C → S' ≤ S) →
  (1/2) * a * b * Real.sin C ≤ Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l332_33244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_l332_33227

def number_list : List Nat := [24, 25, 27, 29, 31]

def is_prime (n : Nat) : Bool := Nat.Prime n

def arithmetic_mean (list : List Nat) : Rat :=
  (list.sum : Rat) / list.length

theorem arithmetic_mean_of_primes :
  arithmetic_mean (number_list.filter is_prime) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_l332_33227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_sets_l332_33263

/-- Given a quadratic inequality ax² + bx + c < 0 with solution set {x | x < -2 or x > -1/2},
    prove that the solution set of ax² - bx + c > 0 is {x | 1/2 < x < 2} -/
theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : {x : ℝ | x < -2 ∨ x > -1/2} = {x : ℝ | a*x^2 + b*x + c < 0}) :
  {x : ℝ | 1/2 < x ∧ x < 2} = {x : ℝ | a*x^2 - b*x + c > 0} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_sets_l332_33263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_conjugates_not_both_on_extension_l332_33270

-- Define the points in the Cartesian plane
variable (A₁ A₂ A₃ A₄ : ℝ × ℝ)

-- Define the vectors
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

-- Define the scalar multiplication of a vector
def scale (s : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (s * v.1, s * v.2)

-- Define the harmonic conjugate condition
def is_harmonic_conjugate (A₁ A₂ A₃ A₄ : ℝ × ℝ) : Prop :=
  ∃ (l m : ℝ), 
    vec A₁ A₃ = scale l (vec A₁ A₂) ∧
    vec A₁ A₄ = scale m (vec A₁ A₂) ∧
    1 / l + 1 / m = 2

-- Define what it means for a point to be on the extension of a segment
def on_extension (P Q R : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t > 1 ∧ vec P R = scale t (vec P Q)

-- The theorem to prove
theorem harmonic_conjugates_not_both_on_extension
  (h : is_harmonic_conjugate A₁ A₂ A₃ A₄) :
  ¬(on_extension A₁ A₂ A₃ ∧ on_extension A₁ A₂ A₄) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_conjugates_not_both_on_extension_l332_33270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l332_33246

-- Define the function f(x) = ln((1-x)/(1+x))
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x, x ∈ Set.Ioo (-1) 1 → f (-x) = -f x) ∧ 
  (∀ x y, x ∈ Set.Ioo (-1) 1 → y ∈ Set.Ioo (-1) 1 → x < y → f x > f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l332_33246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_royal_children_count_l332_33281

/-- Represents the number of years that have passed -/
def n : ℕ → Prop := sorry

/-- Represents the number of daughters -/
def d : ℕ → Prop := sorry

/-- The total number of children -/
def total_children (d : ℕ) : ℕ := d + 3

/-- The initial age of the king and queen -/
def initial_royal_age : ℕ := 35

/-- The initial total age of the children -/
def initial_children_age : ℕ := 35

/-- The condition that after n years, the ages of parents and children are equal -/
def age_equality (n d : ℕ) : Prop :=
  2 * (initial_royal_age + n) = initial_children_age + n * (total_children d)

/-- The main theorem stating the possible number of children -/
theorem royal_children_count :
  ∃ (n d : ℕ), 
    age_equality n d ∧ 
    total_children d ≤ 20 ∧ 
    (total_children d = 7 ∨ total_children d = 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_royal_children_count_l332_33281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_volume_is_six_l332_33214

/-- Represents a 3D configuration of unit cubes -/
def Configuration := List (ℕ × ℕ × ℕ)

/-- Checks if a configuration matches the given front view -/
def matchesFrontView (config : Configuration) : Prop :=
  ∃ col, (config.filter (fun (x, _, _) => x = col)).length = 4

/-- Checks if a configuration matches the given left view -/
def matchesLeftView (config : Configuration) : Prop :=
  ∃ row, (config.filter (fun (_, y, _) => y = row)).length ≥ 2

/-- The volume of a configuration is the number of cubes in it -/
def volume (config : Configuration) : ℕ := config.length

/-- A configuration is valid if it matches both views -/
def isValidConfiguration (config : Configuration) : Prop :=
  matchesFrontView config ∧ matchesLeftView config

theorem min_volume_is_six :
  ∀ config : Configuration, 
    isValidConfiguration config → volume config ≥ 6 :=
by
  intro config h
  sorry -- Proof is omitted for now

#check min_volume_is_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_volume_is_six_l332_33214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_standing_time_l332_33249

/-- Represents the time it takes Clea to ride down the escalator in different scenarios -/
structure EscalatorTime where
  /-- Time to walk down non-operating escalator -/
  walkNonOperating : ℝ
  /-- Time to walk down operating escalator -/
  walkOperating : ℝ

/-- Calculates the time it takes to ride down the operating escalator while standing still -/
noncomputable def timeStandingStill (et : EscalatorTime) : ℝ :=
  (et.walkNonOperating * et.walkOperating) / (et.walkNonOperating - et.walkOperating)

/-- Theorem stating that given the specific times, the time to ride down while standing still is 40 seconds -/
theorem escalator_standing_time (et : EscalatorTime) 
    (h1 : et.walkNonOperating = 60) 
    (h2 : et.walkOperating = 24) : 
    timeStandingStill et = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_standing_time_l332_33249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_pie_cost_l332_33277

/-- The cost to create one pie in a bakery --/
theorem bakery_pie_cost (price_per_piece : ℚ) (pieces_per_pie : ℕ) 
  (pies_per_hour : ℕ) (total_profit : ℚ) 
  (h1 : price_per_piece = 4)
  (h2 : pieces_per_pie = 3)
  (h3 : pies_per_hour = 12)
  (h4 : total_profit = 138)
  : ∃ (cost_per_pie : ℚ), cost_per_pie = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_pie_cost_l332_33277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_l332_33297

/-- Given that α is inversely proportional to β, proves that α = 200 when β = 0.5,
    given that α = 5 when β = 20. -/
theorem inverse_proportion (k : ℝ) (h : ∀ x y, x * y = k ↔ y = k / x)
    (h1 : 5 * 20 = k) : k / 0.5 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_l332_33297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_circles_l332_33201

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the given conditions
axiom C₁ : Circle
axiom C₂ : Circle
axiom P : ℝ × ℝ

-- Define the tangency of C₁ and C₂
axiom C₁_C₂_tangent : ∃ (t : ℝ × ℝ), 
  dist t C₁.center = C₁.radius ∧ 
  dist t C₂.center = C₂.radius

-- Define that P is on the radical axis
axiom P_on_radical_axis : 
  dist P C₁.center^2 - C₁.radius^2 = dist P C₂.center^2 - C₂.radius^2

-- Define that the radical axis is perpendicular to the line joining centers
axiom radical_axis_perpendicular : 
  let v := (C₂.center.1 - C₁.center.1, C₂.center.2 - C₁.center.2)
  let w := (P.1 - C₁.center.1, P.2 - C₁.center.2)
  v.1 * w.1 + v.2 * w.2 = 0

-- Define a function that checks if a circle is tangent to both C₁ and C₂ and passes through P
def is_valid_circle (C : Circle) : Prop :=
  (dist C.center C₁.center = C.radius + C₁.radius ∨ dist C.center C₁.center = abs (C.radius - C₁.radius)) ∧
  (dist C.center C₂.center = C.radius + C₂.radius ∨ dist C.center C₂.center = abs (C.radius - C₂.radius)) ∧
  dist C.center P = C.radius

-- The theorem to be proved
theorem exactly_two_circles : 
  ∃! (B₁ B₂ : Circle), is_valid_circle B₁ ∧ is_valid_circle B₂ ∧ B₁ ≠ B₂ ∧
  ∀ (C : Circle), is_valid_circle C → (C = B₁ ∨ C = B₂) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_circles_l332_33201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l332_33294

theorem sin_alpha_value (α β : ℝ) : 
  α ∈ Set.Ioo (0 : ℝ) (Real.pi / 2) →
  β ∈ Set.Ioo (Real.pi / 2) Real.pi →
  Real.sin (α + β) = 3 / 5 →
  Real.cos β = -5 / 13 →
  Real.sin α = 33 / 65 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l332_33294
