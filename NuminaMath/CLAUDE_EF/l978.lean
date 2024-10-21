import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_l978_97862

/-- The function f(x) = x^2 + log(x + sqrt(x^2 + 1)) -/
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log (x + Real.sqrt (x^2 + 1))

/-- Theorem: If f(a) = M, then f(-a) = 2a^2 - M -/
theorem f_negative (a M : ℝ) (h : f a = M) : f (-a) = 2 * a^2 - M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_l978_97862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l978_97872

-- Define the line equation
def line (a x y : ℝ) : Prop := a * x + y - a = 0

-- Define the circle equation
def circle' (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the discriminant function
def discriminant (a : ℝ) : ℝ := 12 * a^2 - 16

-- Theorem statement
theorem line_intersects_circle (a : ℝ) : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    line a x₁ y₁ ∧ circle' x₁ y₁ ∧
    line a x₂ y₂ ∧ circle' x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l978_97872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_length_l978_97831

/-- An isosceles triangle with equal sides of length b, where t is the length of the segment
    between the intersections of angle bisectors from base angles with the equal sides. -/
structure IsoscelesTriangle where
  b : ℝ
  t : ℝ
  b_pos : b > 0
  t_pos : t > 0
  t_lt_b : t < b

/-- The base of an isosceles triangle given its parameters. -/
noncomputable def baseLength (triangle : IsoscelesTriangle) : ℝ :=
  (triangle.b * triangle.t) / (triangle.b - triangle.t)

/-- Theorem stating that the base of the isosceles triangle is equal to bt/(b-t). -/
theorem isosceles_triangle_base_length (triangle : IsoscelesTriangle) :
  baseLength triangle = (triangle.b * triangle.t) / (triangle.b - triangle.t) := by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_length_l978_97831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l978_97841

noncomputable def a : ℝ := Real.sin (14 * Real.pi / 180) + Real.cos (14 * Real.pi / 180)
noncomputable def b : ℝ := 2 * Real.sqrt 2 * Real.sin (30.5 * Real.pi / 180) * Real.cos (30.5 * Real.pi / 180)
noncomputable def c : ℝ := Real.sqrt 6 / 2

theorem trigonometric_inequality : a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l978_97841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_exist_l978_97858

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -2 then x^2 - 3 else x + 4

-- State the theorem
theorem four_solutions_exist :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x ∈ s, f (f x) = 7 ∧
    ∀ y ∉ s, f (f y) ≠ 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_exist_l978_97858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_trip_theorem_l978_97813

/-- Represents the number of buses needed when using 45-seat buses -/
def x : ℕ := sorry

/-- The number of empty seats in the last bus when using 60-seat buses -/
def empty_seats : ℕ := 15 * x - 60

/-- The total number of people on the trip -/
def total_people : ℕ := 60 * x - 10

/-- Theorem stating the relationship between the number of buses, empty seats, and total people -/
theorem bus_trip_theorem : 
  (45 * x = total_people) ∧ 
  (60 * (x - 1) + (60 - empty_seats) = total_people) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_trip_theorem_l978_97813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_equation_implies_r_value_l978_97856

/-- Given vectors in ℝ³ -/
def a : Fin 3 → ℝ := ![2, 3, -1]
def b : Fin 3 → ℝ := ![1, 1, 0]

/-- The result vector -/
def result : Fin 3 → ℝ := ![3, 4, -1]

/-- Cross product of two vectors in ℝ³ -/
def cross (u v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![u 1 * v 2 - u 2 * v 1, u 2 * v 0 - u 0 * v 2, u 0 * v 1 - u 1 * v 0]

theorem cross_product_equation_implies_r_value :
  ∃ (p q r : ℝ), (∀ i, result i = p * a i + q * b i + r * cross a b i) → r = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_equation_implies_r_value_l978_97856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chords_theorem_l978_97875

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a chord passing through the focus
def chord_through_focus (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, A = focus + t • (B - focus) ∧ parabola A.1 A.2 ∧ parabola B.1 B.2

-- Define perpendicular chords
def perpendicular_chords (A B D E : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (E.2 - D.2) + (B.1 - A.1) * (E.1 - D.1) = 0

-- Define the length of a chord
noncomputable def chord_length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem parabola_chords_theorem (A B D E : ℝ × ℝ) :
  chord_through_focus A B →
  chord_through_focus D E →
  perpendicular_chords A B D E →
  1 / chord_length A B + 1 / chord_length D E = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chords_theorem_l978_97875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_volume_after_two_hours_l978_97811

/-- Calculates the volume of ice after two hours of melting -/
theorem ice_volume_after_two_hours (initial_volume : ℝ) : 
  initial_volume = 6.4 → 
  (initial_volume * (1/4) * (1/4)) = 0.4 := by
  intro h
  rw [h]
  norm_num

#check ice_volume_after_two_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_volume_after_two_hours_l978_97811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_median_on_hypotenuse_l978_97828

theorem right_triangle_median_on_hypotenuse 
  (a b : ℝ) 
  (h1 : Real.sqrt (a - 5) + abs (b - 12) = 0) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  (Real.sqrt (a^2 + b^2)) / 2 = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_median_on_hypotenuse_l978_97828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_sum_equals_143_l978_97820

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.zipWith (λ i bit => if bit then 2^i else 0) (List.range b.length) b).sum

def binary1 : List Bool := [true, false, true, false, true, false, true]
def binary2 : List Bool := [false, true, true, false, true, true]

theorem binary_sum_equals_143 : 
  binary_to_decimal binary1 + binary_to_decimal binary2 = 143 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_sum_equals_143_l978_97820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l978_97854

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define vectors
def m (t : Triangle) : ℝ × ℝ := (t.a + t.c, t.b - t.a)
def n (t : Triangle) : ℝ × ℝ := (t.a - t.c, t.b)
def s : ℝ × ℝ := (0, -1)
noncomputable def t (tr : Triangle) : ℝ × ℝ := (Real.cos tr.A, 2 * (Real.cos (tr.B / 2))^2)

-- Define perpendicularity
def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

-- State the theorem
theorem triangle_properties (tr : Triangle) 
  (h_perp : perpendicular (m tr) (n tr)) : 
  tr.C = π/3 ∧ 
  ∀ x, (Real.sqrt 2/2 ≤ x ∧ x < Real.sqrt 5/2) ↔ ∃ y, y = ‖s + t tr‖ ∧ y = x :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l978_97854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_equivalence_l978_97830

theorem equation_equivalence (x : ℝ) (h : x > 0) :
  (2 * Real.sqrt x + 2 * x^(-(1/2 : ℝ)) = 5) ↔ (4 * x^2 - 17 * x + 4 = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_equivalence_l978_97830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l978_97840

-- Define the circle
def circleC (x y : ℝ) : Prop := x^2 - 4*x + y^2 + 1 = 0

-- Define the line
def lineL (k x y : ℝ) : Prop := y = k*(x-1) + 1

-- Define the chord length
noncomputable def chord_length (k : ℝ) : ℝ := 2 * Real.sqrt (2 - 2*k / (k^2 + 1))

theorem min_chord_length :
  ∀ k : ℝ, ∃ A B : ℝ × ℝ,
    circleC A.1 A.2 ∧ circleC B.1 B.2 ∧
    lineL k A.1 A.2 ∧ lineL k B.1 B.2 ∧
    chord_length k ≥ 2 ∧
    (∀ k' : ℝ, chord_length k' ≥ chord_length k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l978_97840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l978_97832

-- Define the parameter a
variable (a : ℝ)

-- Define set A
def A (a : ℝ) : Set ℝ := {x | -2 ≤ x ∧ x ≤ a}

-- Define set B
def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 2*x + 3}

-- Define set C
def C (a : ℝ) : Set ℝ := {z | ∃ x ∈ A a, z = x^2}

-- State the theorem
theorem range_of_a (h : C a ⊆ B a) : 1/2 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l978_97832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_proof_l978_97809

/-- The smallest positive integer a ≥ 8 such that x^4 + a^2 is not prime for any integer x -/
def smallest_a : ℕ := 12

/-- Predicate to check if a number is composite (not prime) -/
def is_composite (n : ℕ) : Prop := ∃ (m : ℕ), 1 < m ∧ m < n ∧ n % m = 0

theorem smallest_a_proof :
  (∀ (x : ℤ), is_composite (Int.natAbs (x^4) + smallest_a^2)) ∧
  (∀ (a : ℕ), 8 ≤ a → a < smallest_a → ∃ (x : ℤ), ¬is_composite (Int.natAbs (x^4) + a^2)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_proof_l978_97809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_parameters_l978_97816

noncomputable def f (a b c d x : ℝ) : ℝ := a * Real.sin (b * x + c) + d

theorem sine_function_parameters :
  ∀ a b c d : ℝ,
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f a b c d x = f a b c d (x + 2 * Real.pi / 5)) →
  (∃ x, f a b c d x = 5) →
  (∃ x, f a b c d x = -3) →
  a = 4 ∧ b = 5 ∧ d = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_parameters_l978_97816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_palm_oil_in_cheese_l978_97883

/-- Represents the percentage of palm oil in cheese -/
noncomputable def palm_oil_percentage (palm_oil_price_increase : ℝ) (cheese_price_increase : ℝ) : ℝ :=
  (cheese_price_increase / palm_oil_price_increase) * 100

/-- 
Theorem: If a 10% increase in palm oil price results in a 3% increase in cheese price,
then the percentage of palm oil in the cheese is 30%.
-/
theorem palm_oil_in_cheese 
  (h1 : palm_oil_price_increase = 10)
  (h2 : cheese_price_increase = 3)
  : palm_oil_percentage palm_oil_price_increase cheese_price_increase = 30 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_palm_oil_in_cheese_l978_97883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_game_probability_l978_97884

/-- Represents the probability that player A rolls on the nth turn in a dice game -/
noncomputable def p (n : ℕ) : ℝ :=
  1/2 + 1/2 * (-2/3)^(n-1)

/-- The probability of rolling a specific number on a fair six-sided die -/
noncomputable def roll_prob : ℝ := 1/6

/-- Theorem stating the probability formula for the dice game -/
theorem dice_game_probability (n : ℕ) :
  let p_next (k : ℕ) := roll_prob * p k + (1 - roll_prob) * (1 - p k)
  (∀ k, k ≥ 1 → p (k + 1) = p_next k) →
  p 1 = 1 →
  p n = 1/2 + 1/2 * (-2/3)^(n-1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_game_probability_l978_97884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_socks_approximation_l978_97808

/-- Represents the number of sock pairs -/
def n : ℕ := 1

/-- Represents the random variable for the number of socks picked until a complete pair is found -/
noncomputable def ξ : ℕ → ℝ
| 0 => 0
| n + 1 => sorry  -- Placeholder for the actual definition

/-- The expected value of ξ for n sock pairs -/
noncomputable def E (n : ℕ) : ℝ := ξ n

/-- Approximation of the expected value for large n -/
noncomputable def approximation (n : ℕ) : ℝ := Real.sqrt (Real.pi * n)

/-- Theorem stating that for large n, E(n) is approximately √(πn) -/
theorem expected_socks_approximation :
  ∃ (C : ℝ), C > 0 ∧ ∀ (ε : ℝ), ε > 0 → ∃ (N : ℕ), ∀ (m : ℕ), m ≥ N →
  |E m - approximation m| ≤ C * ε * approximation m := by
  sorry  -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_socks_approximation_l978_97808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l978_97887

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := -Real.log x

-- Define the second derivative of g
noncomputable def g'' (x : ℝ) : ℝ := 1 / x^2

-- Theorem statement
theorem tangent_line_slope :
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧
  (∀ x : ℝ, f x₁ + (x - x₁) * ((deriv f) x₁) = g'' x₂ + (x - x₂) * ((deriv g'') x₂)) →
  (deriv f) x₁ = 4 := by
  sorry

#check tangent_line_slope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l978_97887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_θ_values_l978_97886

-- Define the angle θ
noncomputable def θ : ℝ := sorry

-- Define the conditions
axiom θ_range : 0 < θ ∧ θ < 360
axiom θ_equation : (2022 : ℝ)^(2 * Real.sin θ^2 - 3 * Real.sin θ + 1) = 1

-- Define the theorem
theorem sum_of_possible_θ_values : 
  ∃ (θ₁ θ₂ θ₃ : ℝ), 
    (0 < θ₁ ∧ θ₁ < 360) ∧
    (0 < θ₂ ∧ θ₂ < 360) ∧
    (0 < θ₃ ∧ θ₃ < 360) ∧
    (2022 : ℝ)^(2 * Real.sin θ₁^2 - 3 * Real.sin θ₁ + 1) = 1 ∧
    (2022 : ℝ)^(2 * Real.sin θ₂^2 - 3 * Real.sin θ₂ + 1) = 1 ∧
    (2022 : ℝ)^(2 * Real.sin θ₃^2 - 3 * Real.sin θ₃ + 1) = 1 ∧
    θ₁ + θ₂ + θ₃ = 270 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_θ_values_l978_97886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l978_97805

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.b * t.c * Real.sin t.A

/-- Theorem: In a triangle ABC, if b = 3, c = 1, and area = √2, 
    then cos A = ± 1/3 and a = 2√2 or 2√3 -/
theorem triangle_theorem (t : Triangle) 
  (hb : t.b = 3) 
  (hc : t.c = 1) 
  (harea : area t = Real.sqrt 2) :
  (Real.cos t.A = 1/3 ∨ Real.cos t.A = -1/3) ∧ 
  (t.a = 2 * Real.sqrt 2 ∨ t.a = 2 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l978_97805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_and_fifth_friend_payment_l978_97880

/-- The total cost of the camping tent -/
noncomputable def total_cost : ℝ := 120

/-- The payment of the first friend as a fraction of the sum of others' payments -/
noncomputable def first_friend_ratio : ℝ := 1/3

/-- The payment of the second friend as a fraction of the sum of others' payments -/
noncomputable def second_friend_ratio : ℝ := 1/4

/-- The payment of the third friend as a fraction of the sum of others' payments -/
noncomputable def third_friend_ratio : ℝ := 1/5

/-- The theorem stating that the sum of the fourth and fifth friends' payments is $26 -/
theorem fourth_and_fifth_friend_payment :
  ∃ (a b c d e : ℝ),
    a = first_friend_ratio * (b + c + d + e) ∧
    b = second_friend_ratio * (a + c + d + e) ∧
    c = third_friend_ratio * (a + b + d + e) ∧
    a + b + c + d + e = total_cost ∧
    d + e = 26 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_and_fifth_friend_payment_l978_97880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_nested_calculation_l978_97848

-- Define the ♠ operation
noncomputable def spade (a b : ℝ) : ℝ := a - 1 / b

-- Theorem statement
theorem spade_nested_calculation :
  ∀ (a b : ℝ), a > 0 → b > 0 → spade 3 (spade 5 3) = 39 / 14 :=
by
  intros a b ha hb
  -- Unfold the definition of spade
  unfold spade
  -- Simplify the expression
  simp
  -- The actual proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_nested_calculation_l978_97848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_difference_inequality_l978_97835

open Set
open Function

-- Define the interval [a, b]
variable {a b : ℝ}
variable (hab : a ≤ b)

-- Define functions f and g
variable {f g : ℝ → ℝ}

-- Define the derivatives of f and g
variable {f' g' : ℝ → ℝ}

-- Define the condition f'(x) < g'(x) in [a, b]
variable (h_deriv : ∀ x ∈ Icc a b, HasDerivAt f (f' x) x ∧ HasDerivAt g (g' x) x ∧ f' x < g' x)

-- State the theorem
theorem function_difference_inequality (x : ℝ) (hx : x ∈ Icc a b) :
  f x - f b ≥ g x - g b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_difference_inequality_l978_97835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_divisibility_l978_97871

theorem remainder_divisibility (n : ℕ) (h : n % 18 = 11) : n % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_divisibility_l978_97871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_plus_n_equals_n_power_k_l978_97826

theorem factorial_plus_n_equals_n_power_k (n k : ℕ) : 
  n > 0 → k > 0 → (Nat.factorial n + n = n ^ k) ↔ ((n = 2 ∧ k = 2) ∨ (n = 3 ∧ k = 2) ∨ (n = 5 ∧ k = 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_plus_n_equals_n_power_k_l978_97826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_roots_l978_97879

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - x + a

-- State the theorem
theorem f_composition_roots (a : ℝ) :
  (∃ x : ℝ, f a x = 0) →
  (∃ n : Nat, n = 2 ∨ n = 4) ∧ 
  (∃ roots : Finset ℝ, roots.card = n ∧ ∀ x : ℝ, f a (f a x) = 0 ↔ x ∈ roots) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_roots_l978_97879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_e_negative_pi_over_six_l978_97801

theorem imaginary_part_of_e_negative_pi_over_six :
  (Complex.exp (-π/6 * Complex.I)).im = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_e_negative_pi_over_six_l978_97801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l978_97888

-- Define the line equation
def line_equation (x y θ : ℝ) : Prop := x * Real.cos θ + Real.sqrt 3 * y - 2 = 0

-- Define the inclination angle of the line
noncomputable def inclination_angle (θ : ℝ) : ℝ := 
  Real.arctan (-Real.sqrt 3 / 3 * Real.cos θ)

-- State the theorem
theorem inclination_angle_range :
  ∀ θ : ℝ, ∃ α : ℝ, 
    line_equation (1 : ℝ) (1 : ℝ) θ →
    (0 ≤ α ∧ α ≤ Real.pi / 6) ∨ (5 * Real.pi / 6 ≤ α ∧ α < Real.pi) ∧
    α = inclination_angle θ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l978_97888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_nonintersecting_subset_pairs_l978_97804

theorem distinct_nonintersecting_subset_pairs (n : ℕ) : 
  (3^n + 1) / 2 = (Finset.powerset (Finset.range n)).card.choose 2 - 
    (Finset.filter (λ (pair : Finset ℕ × Finset ℕ) => 
      pair.1 ∩ pair.2 = ∅) ((Finset.powerset (Finset.range n)).product 
        (Finset.powerset (Finset.range n)))).card / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_nonintersecting_subset_pairs_l978_97804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_eq_neg_3_l978_97864

/-- Sequence a_n defined by a recurrence relation -/
def a : ℕ → ℤ
  | 0 => 3  -- We define a(0) as 3 to match a(1) in the original problem
  | 1 => 7  -- This matches a(2) in the original problem
  | (n + 2) => a (n + 1) - a n

/-- The 100th term of the sequence a_n is -3 -/
theorem a_100_eq_neg_3 : a 99 = -3 := by  -- We use 99 to match the 100th term in the original problem
  sorry

#eval a 99  -- This will evaluate a_100 (0-indexed) for us

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_eq_neg_3_l978_97864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_circles_cover_two_square_l978_97859

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- The minimum number of circles needed to cover a square -/
def minCirclesCoveringSquare (c : Circle) (s : Square) : ℕ := sorry

/-- A circle with radius 1 -/
def unitCircle : Circle := ⟨1⟩

/-- A square with side length 2 -/
def twoSquare : Square := ⟨2⟩

/-- Theorem stating that 4 unit circles are needed to cover a 2x2 square -/
theorem four_circles_cover_two_square : 
  minCirclesCoveringSquare unitCircle twoSquare = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_circles_cover_two_square_l978_97859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_plus_one_is_even_l978_97806

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (2 * x + φ)

theorem f_plus_one_is_even (φ : ℝ) 
  (h : ∀ x, f x φ ≤ f 1 φ) : 
  ∀ x, f (x + 1) φ = f (-x + 1) φ := by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_plus_one_is_even_l978_97806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_equation_l978_97800

-- Define the circle equations
def circle1 (r : ℝ) (ρ : ℝ) : Prop := ρ = r
def circle2 (r : ℝ) (ρ θ : ℝ) : Prop := ρ = -2 * r * Real.sin (θ + Real.pi/4)

-- Define the common chord equation
def commonChordEq (r : ℝ) (ρ θ : ℝ) : Prop := 
  Real.sqrt 2 * ρ * (Real.sin θ + Real.cos θ) = -r

theorem common_chord_equation (r : ℝ) (h : r > 0) :
  ∀ ρ θ : ℝ, (circle1 r ρ ∧ circle2 r ρ θ) → commonChordEq r ρ θ :=
by
  sorry

#check common_chord_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_equation_l978_97800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_6_addition_correct_l978_97803

/-- Converts a base-6 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 6 + d) 0

/-- Theorem stating that the base-6 addition is correct -/
theorem base_6_addition_correct :
  toDecimal [4, 3, 2, 3] + toDecimal [3, 5, 0] + toDecimal [3, 3] = toDecimal [5, 3, 3, 4] := by
  sorry

#eval toDecimal [4, 3, 2, 3] + toDecimal [3, 5, 0] + toDecimal [3, 3] = toDecimal [5, 3, 3, 4]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_6_addition_correct_l978_97803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_approx_18_l978_97838

noncomputable section

def point := ℝ × ℝ

def A : point := (15, 0)
def B : point := (0, 5)
def D : point := (6, 8)

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem sum_of_distances_approx_18 :
  ∃ ε > 0, |distance A D + distance B D - 18| < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_approx_18_l978_97838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_correct_conclusions_l978_97818

theorem two_correct_conclusions (a b c : ℝ) 
  (h1 : a + b = 0) 
  (h2 : b - c > c - a) 
  (h3 : c - a > 0) : 
  (if |a| > |b| then 1 else 0) + 
  (if a > 0 then 1 else 0) + 
  (if b < 0 then 1 else 0) + 
  (if c < 0 then 1 else 0) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_correct_conclusions_l978_97818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l978_97882

theorem line_parameterization (v d : ℝ × ℝ) : 
  (∀ (t : ℝ), let (x, y) := v + t • d; y = (5 * x - 7) / 6) →
  (∀ (t : ℝ), let (x, y) := v + t • d; x ≥ 4 → 
    ‖(x - 4, y - 2)‖ = t) →
  d = (6 / Real.sqrt 61, 5 / Real.sqrt 61) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l978_97882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_f_greater_than_t_plus_ln_l978_97842

open Real

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Theorem 1: Range of k
theorem range_of_k (k : ℝ) :
  (∀ x, f x ≥ k * x) ↔ (0 ≤ k ∧ k ≤ Real.exp 1) :=
sorry

-- Theorem 2: f(x) > t + ln(x)
theorem f_greater_than_t_plus_ln (t : ℝ) (h : t ≤ 2) :
  ∀ x > 0, f x > t + log x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_f_greater_than_t_plus_ln_l978_97842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_on_circle_l978_97866

/-- The range of m for which the roots of two quadratic equations lie on the same circle -/
theorem roots_on_circle (m : ℝ) : 
  (∀ x : ℂ, x^2 - 2*x + 2 = 0 ∨ x^2 + 2*m*x + 1 = 0 → 
    ∃ (center : ℂ) (radius : ℝ), Complex.abs (x - center) = radius) ↔ 
  (-1 < m ∧ m < 1) ∨ m = -3/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_on_circle_l978_97866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_sum_l978_97877

theorem min_value_exponential_sum (a b : ℝ) (h : a + 3 * b - 2 = 0) :
  ∀ x y : ℝ, x + 3 * y - 2 = 0 → (2 : ℝ)^a + (8 : ℝ)^b ≤ (2 : ℝ)^x + (8 : ℝ)^y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_sum_l978_97877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l978_97869

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 2*x - 3) / Real.log 0.5

theorem f_monotone_increasing :
  ∃ (a : ℝ), a = -3 ∧ StrictMonoOn f (Set.Iio a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l978_97869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_numbers_l978_97836

def digits : Finset ℕ := {0, 1, 2, 3, 4}

def valid_first_digit (d : ℕ) : Prop := d ∈ digits ∧ d ≠ 0

def three_digit_numbers : Finset ℕ := 
  Finset.filter (λ n => n ≥ 100 ∧ n < 1000 ∧ (n / 100) ∈ digits ∧ ((n / 10) % 10) ∈ digits ∧ (n % 10) ∈ digits)
    (Finset.range 1000)

theorem count_three_digit_numbers : Finset.card three_digit_numbers = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_numbers_l978_97836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l978_97891

noncomputable section

/-- Definition of the ellipse (C) -/
def ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of eccentricity -/
def eccentricity (a c : ℝ) : ℝ := c / a

/-- Definition of focus -/
def focus (x y : ℝ) : Prop := x = 1 ∧ y = 0

/-- Definition of perpendicular bisector -/
def perp_bisector (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (x - (x₁ + x₂) / 2) * (x₂ - x₁) + (y - (y₁ + y₂) / 2) * (y₂ - y₁) = 0

theorem ellipse_properties :
  ∀ (a b c : ℝ),
  (∀ x y, ellipse x y a b) →
  (∃ x y, focus x y) →
  eccentricity a c = 1/2 →
  (∀ x y, ellipse x y a b ↔ x^2/4 + y^2/3 = 1) ∧
  (∀ x₁ y₁ x₂ y₂ y₀,
    ellipse x₁ y₁ a b →
    ellipse x₂ y₂ a b →
    (y₁ - 0) / (x₁ - 1) = (y₂ - 0) / (x₂ - 1) →
    perp_bisector x₁ y₁ x₂ y₂ 0 y₀ →
    -Real.sqrt 3 / 12 ≤ y₀ ∧ y₀ ≤ Real.sqrt 3 / 12) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l978_97891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_sum_count_l978_97847

theorem two_digit_sum_count : 
  (Finset.filter (fun p => 10 ≤ p.1 ∧ p.1 ≤ 99 ∧ 10 ≤ p.2 ∧ p.2 ≤ 99 ∧ 10 ≤ p.1 + p.2 ∧ p.1 + p.2 ≤ 99)
    (Finset.product (Finset.range 90) (Finset.range 90))).card = 3240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_sum_count_l978_97847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l978_97833

theorem solve_exponential_equation (x : ℝ) : 3 * (2 : ℝ)^x + 5 * (2 : ℝ)^x = 2048 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l978_97833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_in_engineering_department_l978_97878

theorem women_in_engineering_department 
  (total_percentage : ℝ) 
  (men_percentage : ℝ) 
  (num_men : ℕ) : ℕ := by
  have h1 : total_percentage = 100 := by sorry
  have h2 : men_percentage = 70 := by sorry
  have h3 : num_men = 420 := by sorry
  
  let women_percentage : ℝ := total_percentage - men_percentage
  let total_students : ℝ := (num_men : ℝ) / (men_percentage / 100)
  let num_women : ℝ := (women_percentage / 100) * total_students
  
  have h4 : num_women = 180 := by sorry
  
  exact 180

#check women_in_engineering_department

end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_in_engineering_department_l978_97878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sine_power_l978_97868

theorem integer_sine_power (a b n : ℕ) (h1 : a > b) (h2 : b > 0) : 
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ 
  Real.sin θ = (2 * a * b : ℝ) / ((a^2 + b^2) : ℝ) → 
  ∃ k : ℤ, ((a^2 + b^2 : ℕ)^n : ℝ) * Real.sin θ = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sine_power_l978_97868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_greater_than_five_l978_97815

/-- A fair 10-sided die -/
structure Die where
  faces : Finset Nat
  fair : faces = Finset.range 10

/-- The event of a die showing a number greater than 5 -/
def greaterThanFive (d : Die) : Finset Nat :=
  d.faces.filter (λ n => n > 5)

/-- The probability of the event for a single die -/
def probGreaterThanFive (d : Die) : ℚ :=
  (greaterThanFive d).card / d.faces.card

/-- Rolling 6 dice -/
def rollSixDice : Finset Die :=
  sorry

theorem probability_three_greater_than_five :
  let allRolls := Finset.powerset rollSixDice
  let favorableRolls := allRolls.filter (λ roll => 
    (roll.biUnion greaterThanFive).card = 3)
  (favorableRolls.card : ℚ) / allRolls.card = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_greater_than_five_l978_97815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shaded_triangles_l978_97845

/-- Represents an equilateral triangle --/
structure EquilateralTriangle :=
  (sideLength : ℕ)

/-- Represents a division of a larger triangle into smaller triangles --/
structure TriangleDivision :=
  (largeTriangle : EquilateralTriangle)
  (smallTriangleSideLength : ℕ)

/-- Represents a point in the triangle --/
structure Point :=
  (x : ℕ) (y : ℕ)

/-- Represents a small triangle in the division --/
structure SmallTriangle :=
  (vertices : List Point)

/-- Calculates the number of intersection points in a triangle division --/
def intersectionPoints (td : TriangleDivision) : ℕ :=
  let n := td.largeTriangle.sideLength / td.smallTriangleSideLength
  n * (n + 1) / 2

/-- Theorem stating the minimum number of shaded triangles required --/
theorem min_shaded_triangles (td : TriangleDivision) 
  (h1 : td.largeTriangle.sideLength = 8)
  (h2 : td.smallTriangleSideLength = 1) :
  ∃ (n : ℕ) (shaded : List SmallTriangle), 
    n = 15 ∧ 
    (∀ m : ℕ, m < n → 
      ¬(∀ p : Point, ∃ t ∈ (List.take m shaded), p ∈ t.vertices)) ∧
    (∀ p : Point, ∃ t ∈ shaded, p ∈ t.vertices) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shaded_triangles_l978_97845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l978_97821

/-- Given two curves C₁ and C₂, prove that compressing C₁'s x-coordinates by half
and shifting left by π/12 results in C₂ -/
theorem curve_transformation (x : ℝ) :
  let C₁ := λ x => Real.cos x
  let C₂ := λ x => Real.sin (2 * x + 2 * Real.pi / 3)
  let compressed := λ x => C₁ (2 * x)
  let shifted := λ x => compressed (x + Real.pi / 12)
  shifted x = C₂ x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l978_97821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_rope_cut_l978_97823

/-- The length of the rope in meters -/
noncomputable def rope_length : ℝ := 5

/-- The minimum length of each piece in meters -/
noncomputable def min_piece_length : ℝ := 1

/-- The probability of cutting the rope into two pieces, both not shorter than the minimum length -/
noncomputable def probability_both_pieces_long : ℝ :=
  (rope_length - 2 * min_piece_length) / rope_length

theorem probability_rope_cut :
  probability_both_pieces_long = 3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_rope_cut_l978_97823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l978_97844

/-- Given a natural number n, proves that if the sum of coefficients in the expansion
    of (x^3 + 2/x^2)^n is 243, then the constant term in the expansion is 80. -/
theorem constant_term_of_expansion (n : ℕ) :
  (((1 : ℝ) + 2)^n = 243) →
  (∃ k : ℕ → ℝ, (∀ i, k i = 0 ∨ ∃ j, k i = (n.choose j) * 2^j) ∧
               (∑' i, k i = 243) ∧
               (k 3 = 80)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l978_97844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_bisector_equidistant_l978_97892

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define an angle
structure Angle where
  vertex : Point
  side1 : Point
  side2 : Point

-- Define distance between a point and a line
noncomputable def distanceToLine (p : Point) (l : Line) : ℝ := sorry

-- Define angle bisector
noncomputable def angleBisector (a : Angle) : Line := sorry

-- Define a function to create a line through two points
noncomputable def lineThrough (p1 p2 : Point) : Line := sorry

-- Define set membership for a point on a line
def onLine (p : Point) (l : Line) : Prop := sorry

-- Theorem statement
theorem point_on_bisector_equidistant (a : Angle) (p : Point) :
  onLine p (angleBisector a) →
  distanceToLine p (lineThrough a.vertex a.side1) =
  distanceToLine p (lineThrough a.vertex a.side2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_bisector_equidistant_l978_97892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_shifted_l978_97853

/-- Given a function f(x) = x^2 + 4x + 5 - c with minimum value 2,
    prove that the function f(x-2011) also has a minimum value of 2. -/
theorem min_value_shifted (c : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), x^2 + 4*x + 5 - c ≥ m) ∧ 
  (∀ (ε : ℝ), ε > 0 → ∃ (x : ℝ), x^2 + 4*x + 5 - c < 2 + ε) →
  (∃ (m : ℝ), ∀ (x : ℝ), (x - 2011)^2 + 4*(x - 2011) + 5 - c ≥ m) ∧
  (∀ (ε : ℝ), ε > 0 → ∃ (x : ℝ), (x - 2011)^2 + 4*(x - 2011) + 5 - c < 2 + ε) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_shifted_l978_97853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_on_parabola_tangent_to_directrix_and_x_axis_l978_97861

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -1/2

-- Define the x-axis
def x_axis (y : ℝ) : Prop := y = 0

-- Define the circle
def circle_equation (x y h k r : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem circle_on_parabola_tangent_to_directrix_and_x_axis :
  ∃ (h k r : ℝ),
    (parabola h k) ∧
    (∃ (x y : ℝ), directrix x ∧ circle_equation x y h k r) ∧
    (∃ (x y : ℝ), x_axis y ∧ circle_equation x y h k r) ∧
    (h = 1/2 ∧ (k = 1 ∨ k = -1) ∧ r = 1) :=
by
  -- Proof goes here
  sorry

#check circle_on_parabola_tangent_to_directrix_and_x_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_on_parabola_tangent_to_directrix_and_x_axis_l978_97861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_proof_l978_97889

theorem complex_magnitude_proof (z : ℂ) :
  (∀ (θ : ℝ) (n : ℕ), (Complex.exp (θ * Complex.I)) ^ n = Complex.exp (n * θ * Complex.I)) →
  z * (Complex.exp ((π / 9) * Complex.I)) ^ 6 = 2 →
  Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_proof_l978_97889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_perimeter_consecutive_even_integers_l978_97885

theorem smallest_triangle_perimeter_consecutive_even_integers :
  ∃ (a b c : ℕ),
    (Even a ∧ Even b ∧ Even c) ∧
    (b = a + 2) ∧ (c = b + 2) ∧
    (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧
    (∀ (x y z : ℕ),
      (Even x ∧ Even y ∧ Even z) →
      (y = x + 2) → (z = y + 2) →
      (x + y > z) → (x + z > y) → (y + z > x) →
      (a + b + c ≤ x + y + z)) ∧
    (a + b + c = 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_perimeter_consecutive_even_integers_l978_97885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_cone_volume_is_108π_l978_97876

/-- The volume of a cone with diameter 12 cm and height 9 cm is 108π cubic centimeters. -/
theorem cone_volume (π : ℝ) : ℝ :=
  let diameter : ℝ := 12
  let height : ℝ := 9
  let radius : ℝ := diameter / 2
  (1/3) * π * radius^2 * height

/-- Proof that the volume of the cone is equal to 108π cubic centimeters. -/
theorem cone_volume_is_108π (π : ℝ) : cone_volume π = 108 * π := by
  unfold cone_volume
  simp
  ring

#check cone_volume
#check cone_volume_is_108π

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_cone_volume_is_108π_l978_97876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l978_97895

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.sin (2 * α) = -24/25) 
  (h2 : α > 3*Real.pi/4 ∧ α < Real.pi) : 
  Real.sin α = 3/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l978_97895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_shade_is_500_1001_l978_97857

/-- Represents a rectangle in the 2 by 1001 grid -/
structure Rectangle where
  left : ℕ
  right : ℕ
  top : ℕ
  bottom : ℕ

/-- The total number of rectangles in the grid -/
def total_rectangles : ℕ := 3 * Nat.choose 1002 2

/-- The number of rectangles that contain a shaded square -/
def shaded_rectangles : ℕ := 3 * 501 * 501

/-- Probability of choosing a rectangle that doesn't include a shaded square -/
def probability_no_shade : ℚ :=
  1 - (shaded_rectangles : ℚ) / (total_rectangles : ℚ)

theorem probability_no_shade_is_500_1001 :
  probability_no_shade = 500 / 1001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_shade_is_500_1001_l978_97857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_externally_tangent_sphere_radius_l978_97894

/-- A sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Three spheres on a horizontal plane that are externally tangent to each other -/
structure ThreeTangentSpheres where
  s1 : Sphere
  s2 : Sphere
  s3 : Sphere
  radius_eq : s1.radius = s2.radius ∧ s2.radius = s3.radius
  radius_val : s1.radius = 3
  on_plane : (s1.center).2.2 = (s2.center).2.2 ∧ (s2.center).2.2 = (s3.center).2.2
  externally_tangent : 
    ((s1.center).1 - (s2.center).1)^2 + ((s1.center).2.1 - (s2.center).2.1)^2 + ((s1.center).2.2 - (s2.center).2.2)^2 = (s1.radius + s2.radius)^2 ∧
    ((s2.center).1 - (s3.center).1)^2 + ((s2.center).2.1 - (s3.center).2.1)^2 + ((s2.center).2.2 - (s3.center).2.2)^2 = (s2.radius + s3.radius)^2 ∧
    ((s3.center).1 - (s1.center).1)^2 + ((s3.center).2.1 - (s1.center).2.1)^2 + ((s3.center).2.2 - (s1.center).2.2)^2 = (s3.radius + s1.radius)^2

/-- A sphere externally tangent to three given spheres and the plane -/
def ExternallyTangentSphere (ts : ThreeTangentSpheres) (s : Sphere) : Prop :=
  (s.center).2.2 = (ts.s1.center).2.2 ∧  -- On the same plane
  ((s.center).1 - (ts.s1.center).1)^2 + ((s.center).2.1 - (ts.s1.center).2.1)^2 + ((s.center).2.2 - (ts.s1.center).2.2)^2 = (s.radius + ts.s1.radius)^2 ∧
  ((s.center).1 - (ts.s2.center).1)^2 + ((s.center).2.1 - (ts.s2.center).2.1)^2 + ((s.center).2.2 - (ts.s2.center).2.2)^2 = (s.radius + ts.s2.radius)^2 ∧
  ((s.center).1 - (ts.s3.center).1)^2 + ((s.center).2.1 - (ts.s3.center).2.1)^2 + ((s.center).2.2 - (ts.s3.center).2.2)^2 = (s.radius + ts.s3.radius)^2

theorem externally_tangent_sphere_radius (ts : ThreeTangentSpheres) (s : Sphere) 
  (h : ExternallyTangentSphere ts s) : s.radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_externally_tangent_sphere_radius_l978_97894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_increase_l978_97867

/-- Arithmetic progression sum -/
noncomputable def arithmetic_sum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := n / 2 * (2 * a₁ + (n - 1) * d)

/-- Theorem: If tripling the common difference doubles the sum, then quadrupling it increases the sum by 5/2 -/
theorem arithmetic_progression_sum_increase
  (a₁ : ℝ) (d : ℝ) (n : ℕ) 
  (h₁ : a₁ > 0)
  (h₂ : d > 0)
  (h₃ : arithmetic_sum a₁ (3 * d) n = 2 * arithmetic_sum a₁ d n) :
  arithmetic_sum a₁ (4 * d) n = (5/2) * arithmetic_sum a₁ d n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_increase_l978_97867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeff_works_82_hours_l978_97824

/-- Represents Jeff's weekly schedule --/
structure JeffSchedule where
  facebook_hours : ℚ
  weekend_work_ratio : ℚ
  weekend_twitter : ℚ
  weekend_instagram : ℚ
  weekend_exercise : ℚ
  weekend_errands : ℚ
  weekday_work_ratio : ℚ
  weekday_exercise : ℚ

/-- Calculates the total work hours in a week based on Jeff's schedule --/
def total_work_hours (schedule : JeffSchedule) : ℚ :=
  let weekend_work_hours := schedule.facebook_hours / schedule.weekend_work_ratio
  let weekday_work_hours := schedule.weekday_work_ratio * (schedule.facebook_hours + schedule.weekend_instagram)
  2 * weekend_work_hours + 5 * weekday_work_hours

/-- Theorem stating that Jeff works 82 hours in a week --/
theorem jeff_works_82_hours (schedule : JeffSchedule)
  (h1 : schedule.facebook_hours = 3)
  (h2 : schedule.weekend_work_ratio = 3)
  (h3 : schedule.weekend_twitter = 2)
  (h4 : schedule.weekend_instagram = 1)
  (h5 : schedule.weekend_exercise = 3/2)
  (h6 : schedule.weekend_errands = 3/2)
  (h7 : schedule.weekday_work_ratio = 4)
  (h8 : schedule.weekday_exercise = 1/2) :
  total_work_hours schedule = 82 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeff_works_82_hours_l978_97824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a2_plus_b2_l978_97898

noncomputable def f (a b u : ℝ) : ℝ := u^2 + a*u + (b-2)

noncomputable def u_constraint (x : ℝ) : ℝ := x + 1/x

theorem min_value_of_a2_plus_b2 (a b : ℝ) :
  (∃ u : ℝ, f a b u = 0 ∧ ∃ x : ℝ, x ≠ 0 ∧ u = u_constraint x) →
  a^2 + b^2 ≥ 4/5 := by
  sorry

#check min_value_of_a2_plus_b2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a2_plus_b2_l978_97898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_property_l978_97863

-- Define the function f as noncomputable
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

-- State the theorem
theorem sine_function_property (ω : ℝ) (a : ℝ) (h1 : ω > 0) :
  (∀ x : ℝ, f ω (x - 1/2) = f ω (x + 1/2)) →
  f ω (-1/4) = a →
  f ω (9/4) = -a := by
  intro h2 h3
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_property_l978_97863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_330_degrees_l978_97839

theorem cos_330_degrees : 
  Real.cos (330 * (Real.pi / 180)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_330_degrees_l978_97839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_distance_l978_97814

/-- Calculates the total distance traveled by a man rowing in a river -/
noncomputable def total_distance_traveled (man_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) : ℝ :=
  let upstream_speed := man_speed - river_speed
  let downstream_speed := man_speed + river_speed
  let one_way_distance := (upstream_speed * downstream_speed * total_time) / (2 * (upstream_speed + downstream_speed))
  2 * one_way_distance

/-- Theorem stating the total distance traveled by the man -/
theorem man_rowing_distance :
  let man_speed : ℝ := 6
  let river_speed : ℝ := 3
  let total_time : ℝ := 1
  total_distance_traveled man_speed river_speed total_time = 4.5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_distance_l978_97814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_stacking_problem_l978_97881

/-- The total number of steel rods --/
def total_rods : ℕ := 2009

/-- The diameter of each rod in centimeters --/
def rod_diameter : ℕ := 10

/-- The maximum allowed height of the stack in centimeters --/
def max_height : ℕ := 400

/-- Calculates the sum of the first n natural numbers --/
def triangle_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents an arrangement of rods in an isosceles trapezoid formation --/
structure TrapezoidArrangement where
  layers : ℕ
  top_layer : ℕ

/-- Checks if a trapezoid arrangement is valid --/
def is_valid_arrangement (arr : TrapezoidArrangement) : Prop :=
  arr.layers ≥ 7 ∧ 
  arr.layers * (2 * arr.top_layer + arr.layers - 1) = 2 * total_rods

/-- Calculates the height of a trapezoid arrangement in centimeters --/
noncomputable def arrangement_height (arr : TrapezoidArrangement) : ℝ :=
  (arr.layers - 1 : ℝ) * (rod_diameter : ℝ) * Real.sqrt 3

theorem rod_stacking_problem :
  (∃ n : ℕ, triangle_sum n ≤ total_rods ∧ total_rods - triangle_sum n = 56) ∧
  (∃ arrangements : List TrapezoidArrangement, 
    arrangements.length = 4 ∧ 
    ∀ arr ∈ arrangements, is_valid_arrangement arr) ∧
  (∃ best_arr : TrapezoidArrangement, 
    is_valid_arrangement best_arr ∧
    best_arr.layers = 41 ∧
    arrangement_height best_arr < (max_height : ℝ) ∧
    ∀ arr : TrapezoidArrangement, 
      is_valid_arrangement arr ∧ arrangement_height arr < (max_height : ℝ) →
      arr.layers ≤ best_arr.layers) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_stacking_problem_l978_97881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_archit_win_probability_m_plus_n_equals_seven_l978_97825

-- Define the set of points
def ValidPoint (x y : ℤ) : Prop := -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1

-- Define the distance between two points
def Distance (x1 y1 x2 y2 : ℤ) : ℤ :=
  Int.natAbs (x1 - x2) + Int.natAbs (y1 - y2)

-- Define the possible moves
def ValidMove (x1 y1 x2 y2 : ℤ) : Prop :=
  ValidPoint x1 y1 ∧ ValidPoint x2 y2 ∧ Distance x1 y1 x2 y2 = 1

-- Define the starting positions
def ArchitStart : Prop := ValidPoint 1 1
def AyushStart : Prop := ValidPoint 1 0

-- Define the probability of Archit reaching (0,0) before Ayush
def ProbArchitWins : ℚ := 1/2

-- State the theorem
theorem archit_win_probability :
  ArchitStart ∧ AyushStart →
  (∀ x1 y1 x2 y2, ValidMove x1 y1 x2 y2 → ∃ p : ℚ, p > 0 ∧ p < 1) →
  ProbArchitWins = 1/2 := by
  sorry

-- Prove that m + n = 7
theorem m_plus_n_equals_seven :
  let m : ℕ := 1
  let n : ℕ := 2
  (m : ℚ) / n = ProbArchitWins →
  m + n = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_archit_win_probability_m_plus_n_equals_seven_l978_97825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_harvest_sacks_total_sacks_match_l978_97843

/-- The number of sacks of rice in the first harvest. -/
def first_harvest : ℝ := 20

/-- The yield increase rate after each harvest. -/
def yield_increase_rate : ℝ := 0.2

/-- The total number of sacks after the first and second harvest. -/
def total_sacks : ℝ := 44

/-- Theorem stating that the number of sacks in the first harvest is 20. -/
theorem first_harvest_sacks : first_harvest = 20 :=
  by
    -- Unfold the definition of first_harvest
    unfold first_harvest
    -- Reflexivity (x = x is true)
    rfl

/-- Theorem verifying that the calculated total matches the given total. -/
theorem total_sacks_match :
    first_harvest + first_harvest * (1 + yield_increase_rate) = total_sacks :=
  by
    -- Unfold definitions
    unfold first_harvest yield_increase_rate total_sacks
    -- Simplify the left side of the equation
    simp
    -- Check equality
    norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_harvest_sacks_total_sacks_match_l978_97843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l978_97890

-- Define the train length in meters
noncomputable def train_length : ℝ := 110

-- Define the train speed in km/hr
noncomputable def train_speed_kmh : ℝ := 45

-- Define the bridge length in meters
noncomputable def bridge_length : ℝ := 265

-- Convert km/hr to m/s
noncomputable def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)

-- Calculate the total distance the train needs to travel
noncomputable def total_distance : ℝ := train_length + bridge_length

-- Theorem: The time taken for the train to cross the bridge is 30 seconds
theorem train_crossing_time :
  total_distance / train_speed_ms = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l978_97890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_successful_formula_expected_successful_pairs_gt_half_l978_97850

/-- The number of pairs of socks -/
def n : ℕ := sorry

/-- The probability that all resulting pairs are successful -/
def prob_all_successful (n : ℕ) : ℚ :=
  (2^n * n.factorial) / (2*n).factorial

/-- The expected number of successful pairs -/
def expected_successful_pairs (n : ℕ) : ℚ :=
  n / (2*n - 1)

/-- Theorem stating the probability of all pairs being successful -/
theorem prob_all_successful_formula (n : ℕ) :
  prob_all_successful n = (2^n * n.factorial) / (2*n).factorial := by
  sorry

/-- Theorem stating that the expected number of successful pairs is greater than 0.5 -/
theorem expected_successful_pairs_gt_half (n : ℕ) (h : n > 0) :
  expected_successful_pairs n > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_successful_formula_expected_successful_pairs_gt_half_l978_97850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l978_97893

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- Calculates the distance between two points -/
noncomputable def distance (a b : Point) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- Theorem: For a parabola y^2 = 4x, if a line passing through its focus 
    intersects the parabola at points A and B, and the sum of their 
    x-coordinates is 6, then the length of chord AB is 8 -/
theorem parabola_chord_length 
  (a b : Point) 
  (ha : a ∈ Parabola) 
  (hb : b ∈ Parabola) 
  (h_sum : a.x + b.x = 6) 
  (h_line : ∃ (t : ℝ), a.x = t * (1 - t) ∧ b.x = t * (1 + t)) :
  distance a b = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l978_97893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l978_97899

-- Define the constants as noncomputable
noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.exp (1 / Real.exp 1)
noncomputable def c : ℝ := Real.rpow 3 (1/3)

-- State the theorem
theorem ordering_abc : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l978_97899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_purchase_l978_97802

/-- Calculates the total number of items (pies and coffee) that can be purchased
    given a total budget, pie cost, coffee cost, service fee, and the strategy
    of buying as many pies as possible before buying coffee. -/
def total_items (total_budget : ℚ) (pie_cost : ℚ) (coffee_cost : ℚ) (service_fee : ℚ) : ℕ :=
  let remaining_budget := total_budget - service_fee
  let num_pies := (remaining_budget / pie_cost).floor.toNat
  let money_for_coffee := remaining_budget - (num_pies : ℚ) * pie_cost
  let num_coffee := (money_for_coffee / coffee_cost).floor.toNat
  num_pies + num_coffee

/-- Theorem stating that given the specific conditions of the problem,
    the total number of items purchased is 6. -/
theorem bakery_purchase :
  total_items 50 7 (5/2) 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_purchase_l978_97802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_problem_l978_97807

theorem divisibility_problem (a : ℤ) 
  (h1 : 0 ≤ a) (h2 : a ≤ 13) 
  (h3 : (51^2023 + a) % 13 = 0) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_problem_l978_97807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_relation_l978_97874

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line passing through A and B
def line (x y : ℝ) : Prop := y = (4/5) * (x - 1)

-- Define the circle
def my_circle (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem parabola_circle_relation (a b r : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line x₁ y₁ ∧ line x₂ y₂ ∧
    my_circle x₁ y₁ a b r ∧ my_circle x₂ y₂ a b r ∧
    my_circle focus.1 focus.2 a b r →
    a^2 = r^2 + 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_relation_l978_97874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_l978_97851

-- Define the complex number z
noncomputable def z (r : ℝ) (θ : ℝ) : ℂ := r * (Complex.cos θ + Complex.I * Complex.sin θ)

-- Define omega
noncomputable def ω (r : ℝ) (z : ℂ) : ℂ := z + 1 / z

-- State the theorem
theorem trajectory_is_ellipse (r : ℝ) (h : r > 1) :
  ∃ (a b : ℝ), (∀ θ : ℝ, 
    let x := (ω r (z r θ)).re
    let y := (ω r (z r θ)).im
    x^2 / a^2 + y^2 / b^2 = 1) ∧
  (a^2 - b^2 = 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_l978_97851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boiling_point_l978_97817

/-- Converts Fahrenheit to Celsius -/
noncomputable def f_to_c (f : ℝ) : ℝ := (f - 32) * (5/9)

/-- Converts Celsius to Fahrenheit -/
noncomputable def c_to_f (c : ℝ) : ℝ := c * (9/5) + 32

theorem water_boiling_point : 
  ∀ (boiling_point_f boiling_point_c : ℝ),
  boiling_point_f = 212 → -- Water boils at 212 °F
  f_to_c 32 = 0 → -- Ice melts at 32 °F and 0 °C
  c_to_f 50 = 122 → -- 50 °C corresponds to 122 °F
  boiling_point_c = f_to_c boiling_point_f →
  boiling_point_c = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boiling_point_l978_97817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l978_97849

theorem inequality_proof (t : ℝ) (n : ℕ) (h : t ≥ 1/2) :
  t^(2*n) ≥ (t-1)^(2*n) + (2*t-1)^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l978_97849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_asymptote_distance_l978_97855

-- Define the hyperbola
noncomputable def hyperbola (x y b : ℝ) : Prop := y^2 / 9 - x^2 / b^2 = 1

-- Define eccentricity
noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

-- Define the distance from focus to asymptote
noncomputable def distance_focus_to_asymptote (b : ℝ) : ℝ := 3 * Real.sqrt 3

-- Theorem statement
theorem focus_to_asymptote_distance 
  (b : ℝ) 
  (h1 : ∃ x y, hyperbola x y b) 
  (h2 : eccentricity (Real.sqrt (9 + b^2)) 3 = 2) :
  distance_focus_to_asymptote b = 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_asymptote_distance_l978_97855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_completion_time_B_completes_in_5_point_6_days_l978_97822

/-- Given two workers A and B who can complete a task together in 4 days,
    and A alone can complete it in 14 days, prove that B alone can complete
    the task in 5.6 days. -/
theorem worker_completion_time (A B : ℝ) : 
  (A + B = 1 / 4) → (A = 1 / 14) → B = 5 / 28 := by
  sorry

/-- The time it takes B to complete the task alone. -/
noncomputable def B_completion_time (B : ℝ) : ℝ := 1 / B

/-- Theorem stating that B completes the task in 5.6 days. -/
theorem B_completes_in_5_point_6_days (A B : ℝ) 
  (h1 : A + B = 1 / 4) (h2 : A = 1 / 14) : 
  B_completion_time B = 5.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_completion_time_B_completes_in_5_point_6_days_l978_97822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_giraffe_jade_amount_l978_97873

/-- The amount of jade (in grams) needed for a giraffe statue -/
def G : ℝ := sorry

/-- The price of a giraffe statue in dollars -/
def giraffe_price : ℝ := 150

/-- The price of an elephant statue in dollars -/
def elephant_price : ℝ := 350

/-- The total amount of jade Nancy has in grams -/
def total_jade : ℝ := 1920

/-- The additional revenue from making all elephants instead of giraffes -/
def additional_revenue : ℝ := 400

theorem giraffe_jade_amount :
  (elephant_price * (total_jade / (2 * G)) = giraffe_price * (total_jade / G) + additional_revenue) →
  G = 120 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_giraffe_jade_amount_l978_97873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_A_neither_necessary_nor_sufficient_for_not_B_l978_97810

structure BallSet where
  red : ℕ
  yellow : ℕ
  white : ℕ

def total_balls (s : BallSet) : ℕ := s.red + s.yellow + s.white

def event_A (s : BallSet) : Prop := ∃ (r : Fin s.red) (y : Fin s.yellow), True

def event_B (s : BallSet) : Prop := ∃ (a b : Fin (total_balls s)), a ≠ b

theorem not_A_neither_necessary_nor_sufficient_for_not_B (s : BallSet) 
  (h_red : s.red = 5) (h_yellow : s.yellow = 3) (h_white : s.white = 2) :
  ¬(∀ (x : BallSet), ¬(event_A x) → ¬(event_B x)) ∧ 
  ¬(∀ (x : BallSet), ¬(event_B x) → ¬(event_A x)) := by
  sorry

#check not_A_neither_necessary_nor_sufficient_for_not_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_A_neither_necessary_nor_sufficient_for_not_B_l978_97810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_average_playing_time_l978_97897

/-- Calculates the average playing time given the hours played on each day -/
noncomputable def average_playing_time (wed_hours : ℝ) (thu_hours : ℝ) (fri_hours : ℝ) : ℝ :=
  (wed_hours + thu_hours + fri_hours) / 3

/-- Theorem stating that Max's average playing time over three days is approximately 4.83 hours -/
theorem max_average_playing_time :
  let wed_hours := (2 : ℝ)
  let thu_hours := (2 : ℝ)
  let fri_hours := (10.5 : ℝ)
  abs (average_playing_time wed_hours thu_hours fri_hours - 4.83) < 0.01 := by
  sorry

-- This will not evaluate due to the noncomputable nature of the function
-- #eval average_playing_time 2 2 10.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_average_playing_time_l978_97897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_value_at_pi_over_12_l978_97896

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := 6 * Real.sin (2 * x - Real.pi / 3)

/-- The translated function g(x) -/
noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 12)

/-- Theorem stating that g(π/12) equals -3√3 -/
theorem g_value_at_pi_over_12 : g (Real.pi / 12) = -3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_value_at_pi_over_12_l978_97896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_sum_l978_97834

theorem complex_fraction_sum (a b : ℝ) : 
  (Complex.I - 2) / (1 + Complex.I) = Complex.ofReal a + Complex.I * Complex.ofReal b → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_sum_l978_97834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l978_97812

theorem polynomial_divisibility :
  let k : ℝ := 6
  let p : ℝ → ℝ := λ x => 4*x^3 - 16*x^2 + k*x - 24
  let d1 : ℝ → ℝ := λ x => x - 4
  let d2 : ℝ → ℝ := λ x => 4*x^2 - 6
  ∃ q1 q2 : ℝ → ℝ, (∀ x, p x = d1 x * q1 x) ∧ (∀ x, p x = d2 x * q2 x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l978_97812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l978_97865

theorem greatest_integer_fraction : 
  ⌊(5^100 + 3^100 : ℝ) / (5^96 + 3^96 : ℝ)⌋ = 624 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l978_97865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_2_f_2_eq_1_f_derivative_2_eq_2_l978_97819

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the given condition
axiom f_condition : ∀ x : ℝ, 2 * f (4 - x) = f x + x^2 - 10*x + 17

-- Define the tangent line equation
def tangent_line (x : ℝ) : ℝ := 2*x - 3

-- State the theorem
theorem tangent_at_2 : 
  (∀ x : ℝ, (∃ ε > 0, ∀ h : ℝ, |h| < ε → 
    |f (2 + h) - f 2 - 2*h| ≤ |h| * |h|)) ∧
  f 2 = tangent_line 2 :=
sorry

-- Prove that f(2) = 1
theorem f_2_eq_1 : f 2 = 1 := sorry

-- Prove that f'(2) = 2
theorem f_derivative_2_eq_2 : 
  ∃ ε > 0, ∀ h : ℝ, |h| < ε → 
    |f (2 + h) - f 2 - 2*h| ≤ |h| * |h| := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_2_f_2_eq_1_f_derivative_2_eq_2_l978_97819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circumcircle_theorem_l978_97860

noncomputable section

-- Define the ellipse E
def ellipse_E (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

-- Define the foci of the ellipse
def focus1 : ℝ × ℝ := (-1, 0)
def focus2 : ℝ × ℝ := (1, 0)

-- Define the point that the ellipse passes through
def point_on_ellipse : ℝ × ℝ := (1, Real.sqrt 2 / 2)

-- Define point P
def P : ℝ × ℝ := (-2, 0)

-- Define the relationship between PA and PB
def PB_eq_3PA (A B : ℝ × ℝ) : Prop :=
  B.1 - P.1 = 3 * (A.1 - P.1) ∧ B.2 - P.2 = 3 * (A.2 - P.2)

-- Define the circumcircle
def circumcircle (x y : ℝ) : Prop :=
  (x + 1/3)^2 + y^2 = 10/9

theorem ellipse_circumcircle_theorem :
  ∀ A B : ℝ × ℝ,
  ellipse_E A.1 A.2 → 
  ellipse_E B.1 B.2 → 
  PB_eq_3PA A B →
  ∃ C D : ℝ × ℝ,
  C.1 = A.1 ∧ C.2 = -A.2 ∧
  D.1 = B.1 ∧ D.2 = -B.2 →
  (∀ x y : ℝ, circumcircle x y ↔ 
    (x - A.1)^2 + (y - A.2)^2 = (x - C.1)^2 + (y - C.2)^2 ∧
    (x - B.1)^2 + (y - B.2)^2 = (x - D.1)^2 + (y - D.2)^2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circumcircle_theorem_l978_97860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2phi_value_l978_97870

-- Define the given equation
def given_equation (φ : ℝ) : Prop :=
  (3 : ℝ)^(-4/3 + 3 * Real.sin φ) + 1 = (3 : ℝ)^(1/3 + Real.sin φ)

-- Theorem statement
theorem sin_2phi_value (φ : ℝ) :
  given_equation φ → Real.sin (2 * φ) = 4 * Real.sqrt 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2phi_value_l978_97870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_pens_problem_l978_97827

/-- The number of pens bought by the retailer -/
def num_pens : ℕ := 80

/-- The market price of one pen -/
noncomputable def P : ℝ := sorry

/-- The total cost for the retailer -/
noncomputable def total_cost : ℝ := 36 * P

/-- The selling price per pen -/
noncomputable def selling_price_per_pen : ℝ := 0.99 * P

/-- The total selling price -/
noncomputable def total_selling_price : ℝ := num_pens * selling_price_per_pen

theorem retailer_pens_problem :
  total_selling_price = 2.2 * total_cost := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_pens_problem_l978_97827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irreducible_odd_degree_divides_sum_of_squares_l978_97846

/-- An irreducible polynomial of odd degree over ℚ that divides a sum of squares divides it with multiplicity at least 2 -/
theorem irreducible_odd_degree_divides_sum_of_squares
  {K : Type*} [Field K] [CharZero K]
  (p q r : Polynomial K)
  (h_irred : Irreducible p)
  (h_odd : Odd p.degree)
  (h_div : p ∣ q^2 + q*r + r^2) :
  p^2 ∣ q^2 + q*r + r^2 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irreducible_odd_degree_divides_sum_of_squares_l978_97846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l978_97837

/-- A right triangle PQR in the xy-plane -/
structure RightTriangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  is_right_angle_at_R : Prop
  hypotenuse_length : ℝ

/-- The equation of a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Median of a triangle -/
def Median (T : RightTriangle) (vertex : ℝ × ℝ) (l : Line) : Prop :=
  sorry

/-- The area of a right triangle -/
def TriangleArea (T : RightTriangle) : ℝ :=
  sorry

theorem right_triangle_area (T : RightTriangle) 
  (hyp_length : T.hypotenuse_length = 50)
  (median_P : Median T T.P (Line.mk 1 5))
  (median_Q : Median T T.Q (Line.mk 3 6)) : 
  TriangleArea T = 11250/31 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l978_97837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_players_lost_to_ai_l978_97829

theorem chess_players_lost_to_ai (total_players : ℕ) (never_lost_fraction : ℚ) 
  (h1 : total_players = 40)
  (h2 : never_lost_fraction = 1 / 4) : 
  total_players - (never_lost_fraction * total_players).floor = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_players_lost_to_ai_l978_97829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relation_l978_97852

/-- 
In a triangle with sides a, b, and c, if the angle opposite side c 
is twice the angle opposite side b, then c = √(b(a+b)).
-/
theorem triangle_side_relation (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∃ α : ℝ, 0 < α ∧ α < Real.pi / 2 ∧ 
    Real.sin (2 * α) / c = Real.sin α / b ∧
    Real.sin (2 * α) / a = Real.sin α / b) →
  c = Real.sqrt (b * (a + b)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relation_l978_97852
