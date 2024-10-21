import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_in_NH3_approx_l1204_120436

/-- The atomic mass of hydrogen in g/mol -/
noncomputable def atomic_mass_H : ℝ := 1.008

/-- The atomic mass of nitrogen in g/mol -/
noncomputable def atomic_mass_N : ℝ := 14.007

/-- The number of hydrogen atoms in ammonia -/
def num_H_in_NH3 : ℕ := 3

/-- The number of nitrogen atoms in ammonia -/
def num_N_in_NH3 : ℕ := 1

/-- The mass of hydrogen in one molecule of NH3 in g/mol -/
noncomputable def mass_H_in_NH3 : ℝ := num_H_in_NH3 * atomic_mass_H

/-- The total mass of one molecule of NH3 in g/mol -/
noncomputable def total_mass_NH3 : ℝ := num_N_in_NH3 * atomic_mass_N + num_H_in_NH3 * atomic_mass_H

/-- The mass percentage of hydrogen in NH3 -/
noncomputable def mass_percentage_H_in_NH3 : ℝ := (mass_H_in_NH3 / total_mass_NH3) * 100

theorem mass_percentage_H_in_NH3_approx :
  |mass_percentage_H_in_NH3 - 17.75| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_in_NH3_approx_l1204_120436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1204_120449

-- Define the custom operation ⊕
noncomputable def oplus (a b : ℝ) : ℝ := if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (oplus 1 x) * x - 2 * (oplus 2 x)

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m ∈ Set.Icc 0 1) ↔ 
  (m - 2 ∈ Set.Icc (-2) 2 ∧ 
   2 * m ∈ Set.Icc (-2) 2 ∧ 
   f (m - 2) ≤ f (2 * m)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1204_120449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1204_120434

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The area of a triangle -/
def area (t : Triangle) : Real := sorry

/-- The dot product of two vectors -/
def dot_product (v w : Real × Real) : Real := sorry

/-- Vector AB in triangle ABC -/
def vec_AB (t : Triangle) : Real × Real := sorry

/-- Vector AC in triangle ABC -/
def vec_AC (t : Triangle) : Real × Real := sorry

theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.sqrt 3 * area t - dot_product (vec_AB t) (vec_AC t) = 0)
  (h2 : t.c = 2)
  (h3 : t.a^2 + t.b^2 - t.c^2 = 6/5 * t.a * t.b) :
  t.A = π/6 ∧ t.b = (3 + 4 * Real.sqrt 3) / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1204_120434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_hyperbola_l1204_120489

-- Define the curve in polar coordinates
noncomputable def polar_curve (θ : ℝ) : ℝ := 1 / (Real.sin θ + Real.cos θ)

-- Define what it means for a curve to be a hyperbola
def is_hyperbola (f : ℝ → ℝ) : Prop := 
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 
  ∀ (x y : ℝ), f x = y → (x^2 / a^2) - (y^2 / b^2) = 1

-- State the theorem
theorem polar_curve_is_hyperbola : is_hyperbola polar_curve := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_hyperbola_l1204_120489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inv_f_seven_equals_two_l1204_120409

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the inverse of g
noncomputable def g_inv : ℝ → ℝ := sorry

-- Define the given condition
axiom inverse_composition : ∀ x : ℝ, (f.invFun (g x) = x^3 - 1)

-- State the theorem to be proved
theorem g_inv_f_seven_equals_two : g_inv (f 7) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inv_f_seven_equals_two_l1204_120409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liz_spent_amount_l1204_120435

noncomputable def recipe_book_cost : ℝ := 6
noncomputable def baking_dish_cost : ℝ := 2 * recipe_book_cost
noncomputable def ingredients_cost : ℝ := 5 * 3
noncomputable def apron_cost : ℝ := recipe_book_cost + 1
noncomputable def mixer_cost : ℝ := 3 * baking_dish_cost
noncomputable def measuring_cups_cost : ℝ := apron_cost / 2
noncomputable def spices_cost : ℝ := 4 * 2
noncomputable def cooking_utensils_cost : ℝ := 3 * 4
noncomputable def baking_cups_cost : ℝ := 6 * 0.5

noncomputable def total_cost : ℝ := recipe_book_cost + baking_dish_cost + ingredients_cost + apron_cost + 
                      mixer_cost + measuring_cups_cost + spices_cost + cooking_utensils_cost + 
                      baking_cups_cost

noncomputable def discount_rate : ℝ := 0.1
noncomputable def discount_amount : ℝ := discount_rate * total_cost
noncomputable def final_cost : ℝ := total_cost - discount_amount

theorem liz_spent_amount : final_cost = 92.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liz_spent_amount_l1204_120435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_odd_divisor_25_factorial_l1204_120424

-- Define 25!
def factorial_25 : ℕ := Nat.factorial 25

-- Define the set of positive integer divisors of 25!
def divisors_of_factorial_25 : Set ℕ := {d : ℕ | d > 0 ∧ factorial_25 % d = 0}

-- Define the set of odd divisors of 25!
def odd_divisors_of_factorial_25 : Set ℕ := {d ∈ divisors_of_factorial_25 | d % 2 = 1}

-- Theorem statement
theorem probability_of_odd_divisor_25_factorial :
  ∃ (n m : ℕ), n ≠ 0 ∧ m ≠ 0 ∧
  (n : ℚ) / m = 1 / 23 ∧
  n = Finset.card (Finset.filter (λ x => x % 2 = 1) (Finset.filter (λ d => factorial_25 % d = 0) (Finset.range (factorial_25 + 1)))) ∧
  m = Finset.card (Finset.filter (λ d => factorial_25 % d = 0) (Finset.range (factorial_25 + 1))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_odd_divisor_25_factorial_l1204_120424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3147_to_hundredth_l1204_120441

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem round_3147_to_hundredth :
  round_to_hundredth 3.147 = 3.15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3147_to_hundredth_l1204_120441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_time_proof_l1204_120440

/-- Calculates the time taken given distance and speed -/
noncomputable def time_taken (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Proves that bicycling 40 kilometers at 8 kilometers per hour takes 5 hours -/
theorem bicycle_time_proof :
  let distance : ℝ := 40
  let speed : ℝ := 8
  time_taken distance speed = 5 := by
  -- Unfold the definition of time_taken
  unfold time_taken
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_time_proof_l1204_120440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_approximation_l1204_120403

/-- The markup percentage applied to the cost price -/
noncomputable def markup : ℚ := 15 / 100

/-- The selling price of the computer table in rupees -/
def selling_price : ℚ := 6400

/-- The cost price of the computer table in rupees -/
noncomputable def cost_price : ℚ := selling_price / (1 + markup)

/-- Theorem stating that the cost price is approximately 5565.22 rupees -/
theorem cost_price_approximation : 
  ∃ ε > 0, |cost_price - 5565.22| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_approximation_l1204_120403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_years_l1204_120469

-- Define the parameters
def principal : ℝ := 7500
def rate : ℝ := 0.04
def interest : ℝ := 612

-- Define the compound interest formula
noncomputable def compound_interest (P r t : ℝ) : ℝ := P * ((1 + r) ^ t - 1)

-- Define the function to calculate the number of years
noncomputable def calculate_years (P r I : ℝ) : ℝ := Real.log (1 + I / P) / Real.log (1 + r)

-- Theorem statement
theorem compound_interest_years :
  Int.floor (calculate_years principal rate interest) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_years_l1204_120469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1204_120413

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x : ℝ, y = 2^x / (2^x + 1)}
def B (m : ℝ) : Set ℝ := {y | ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ y = (1/3) * x + m}

-- Define the proposition p and q
def p (x : ℝ) : Prop := x ∈ A
def q (m : ℝ) (x : ℝ) : Prop := x ∈ B m

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, q m x → p x) ∧ (∃ x : ℝ, p x ∧ ¬q m x) ↔ m ∈ Set.Ioo (1/3) (2/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1204_120413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_z_111_l1204_120472

noncomputable def z : ℕ → ℂ
  | 0 => 0
  | n + 1 => z n ^ 2 + (1 + Complex.I)

theorem distance_to_z_111 :
  let z_111 := z 111
  Complex.abs z_111 = Real.sqrt (z_111.re ^ 2 + z_111.im ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_z_111_l1204_120472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_of_hyperbola_with_eccentricity_two_l1204_120431

/-- Represents a hyperbola with equation x²/a² - y² = 1 -/
structure Hyperbola where
  a : ℝ
  h_pos : a > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (h.a^2 + 1) / h.a

/-- The equation of the asymptotes of a hyperbola -/
def asymptotes_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => y = h.a⁻¹ * x ∨ y = -h.a⁻¹ * x

/-- Theorem: For a hyperbola with eccentricity 2, its asymptotes are y = ±√3x -/
theorem asymptotes_of_hyperbola_with_eccentricity_two (h : Hyperbola) 
  (h_ecc : eccentricity h = 2) :
  asymptotes_equation h = fun x y => y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_of_hyperbola_with_eccentricity_two_l1204_120431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1204_120400

noncomputable def f (x : ℝ) := Real.log (x^2 + 2*x - 3) / Real.log 2

def IsValidArg (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ y, f x = y

theorem domain_of_f :
  {x : ℝ | IsValidArg f x} = {x : ℝ | x < -3 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1204_120400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_closest_to_longest_side_is_half_l1204_120419

/-- Represents a trapezoid field with specific dimensions -/
structure TrapezoidField where
  short_base : ℝ
  long_base : ℝ
  side_length : ℝ
  angle : ℝ
  short_base_eq : short_base = 80
  long_base_eq : long_base = 160
  side_length_eq : side_length = 120
  angle_eq : angle = π/4

/-- Calculates the fraction of the trapezoid's area closer to the longest side -/
noncomputable def fraction_closest_to_longest_side (field : TrapezoidField) : ℝ :=
  1/2

/-- Theorem stating that the fraction of the area closer to the longest side is 1/2 -/
theorem fraction_closest_to_longest_side_is_half (field : TrapezoidField) :
  fraction_closest_to_longest_side field = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_closest_to_longest_side_is_half_l1204_120419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l1204_120454

theorem ascending_order (a b c : ℝ) : 
  a = (0.8 : ℝ)^(0.7 : ℝ) → b = (0.8 : ℝ)^(0.9 : ℝ) → c = (1.2 : ℝ)^(0.8 : ℝ) → b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l1204_120454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1204_120421

/-- The circle with equation x^2 + (y-3)^2 = 4 -/
def my_circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - 3)^2 = 4}

/-- The line with equation x + y + 1 = 0 -/
def perpendicular_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 + 1 = 0}

/-- The line we're trying to prove the equation of -/
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 3 = 0}

theorem line_equation_proof :
  ∀ l : Set (ℝ × ℝ),
  (∃ c ∈ my_circle, c ∈ l) →  -- l passes through the center of the circle
  (∀ p q : ℝ × ℝ, p ∈ l → q ∈ l → p ≠ q →
    (p.1 - q.1) * (p.1 - q.1 + p.2 - q.2) = 0) →  -- l is perpendicular to perpendicular_line
  l = line_l :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1204_120421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisor_l1204_120497

theorem smallest_divisor (x n : ℕ) (hn : n > 0) : 
  x % n = 5 ∧ (4 * x) % n = 2 → n ≥ 9 ∧ ∃ m : ℕ, m ≥ 9 ∧ (∀ k : ℕ, k ≥ 9 → m ≤ k) ∧ m = n := by
  sorry

#check smallest_divisor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisor_l1204_120497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iaras_purchases_l1204_120496

/-- Represents the possible combinations of plates and cups Iara can buy --/
structure Purchase where
  plates : ℕ
  cups : ℕ

/-- Checks if a purchase satisfies Iara's constraints --/
def is_valid_purchase (p : Purchase) : Prop :=
  p.plates ≥ 4 ∧ 
  p.cups ≥ 6 ∧ 
  2.5 * (p.cups : ℝ) + 7 * (p.plates : ℝ) ≤ 50

/-- The theorem stating the only valid purchases for Iara --/
theorem iaras_purchases : 
  ∀ p : Purchase, is_valid_purchase p ↔ (p = ⟨4, 8⟩ ∨ p = ⟨5, 6⟩) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_iaras_purchases_l1204_120496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_angle_ratio_l1204_120423

/-- A parallelogram with specific side and diagonal ratios has a 5:1 angle ratio -/
theorem parallelogram_angle_ratio :
  ∀ (a b d₁ d₂ θ₁ θ₂ : ℝ),
  (b = Real.sqrt 3 * a) →                -- One side is √3 times longer than the other
  (d₂ = Real.sqrt 7 * d₁) →              -- One diagonal is √7 times longer than the other
  (d₁^2 + d₂^2 = 2 * (a^2 + b^2)) →  -- Parallelogram diagonal formula
  (θ₁ + θ₂ = Real.pi) →               -- Sum of angles in a parallelogram is π (180°)
  (θ₁ < θ₂) →                   -- θ₁ is the smaller angle
  (θ₂ = 5 * θ₁) :=              -- The larger angle is 5 times the smaller angle
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_angle_ratio_l1204_120423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_l1204_120492

-- Define a type for planes
structure Plane where

-- Define a parallel relation between planes
def parallel (p q : Plane) : Prop := sorry

-- State the theorem
theorem parallel_transitivity (α β γ : Plane) 
  (h1 : β ≠ α) (h2 : γ ≠ α) (h3 : γ ≠ β)
  (h4 : parallel β α) (h5 : parallel γ α) : 
  parallel β γ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_l1204_120492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_specific_circle_l1204_120473

/-- The equation of a circle in the form x^2 + y^2 + Dx + Ey + F = 0 --/
structure CircleEquation where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Calculate the radius of a circle given its equation --/
noncomputable def radiusOfCircle (eq : CircleEquation) : ℝ :=
  (1/2) * Real.sqrt (eq.D^2 + eq.E^2 - 4*eq.F)

theorem radius_of_specific_circle :
  let eq : CircleEquation := ⟨2, -2, -7⟩
  radiusOfCircle eq = 3 := by
  sorry

#eval "Theorem statement type-checks correctly."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_specific_circle_l1204_120473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l1204_120474

theorem arithmetic_sequence_solution (x : ℝ) : 
  x > 0 → 
  (2^2 : ℝ) + 5^2 = 2 * x^2 → 
  x = Real.sqrt 14.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l1204_120474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_a_b_is_half_l1204_120401

noncomputable def a : ℝ × ℝ := (Real.cos (25 * Real.pi / 180), Real.sin (25 * Real.pi / 180))
noncomputable def b : ℝ × ℝ := (Real.cos (85 * Real.pi / 180), Real.cos (5 * Real.pi / 180))

theorem dot_product_a_b_is_half :
  a.1 * b.1 + a.2 * b.2 = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_a_b_is_half_l1204_120401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_sum_first_20_a_l1204_120463

def a : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | n + 2 => if n % 2 = 0 then a (n + 1) + 2 else a (n + 1) + 1

def b (n : ℕ) : ℕ := a (2 * n)

theorem b_formula (n : ℕ) : b n = 3 * n - 1 := by
  sorry

theorem sum_first_20_a : (Finset.range 20).sum a = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_sum_first_20_a_l1204_120463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1204_120443

noncomputable def f (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

theorem min_value_of_f :
  ∀ x y : ℝ, 0.4 ≤ x → x ≤ 0.6 → 0.3 ≤ y → y ≤ 0.4 →
  f x y ≥ 2/5 ∧ ∃ x₀ y₀ : ℝ, 0.4 ≤ x₀ ∧ x₀ ≤ 0.6 ∧ 0.3 ≤ y₀ ∧ y₀ ≤ 0.4 ∧ f x₀ y₀ = 2/5 :=
by sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1204_120443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1204_120460

/-- Time taken for a train to pass a man moving in the same direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 350 →
  train_speed = 68 * (1000 / 3600) →
  man_speed = 8 * (1000 / 3600) →
  let relative_speed := train_speed - man_speed
  ⌊train_length / relative_speed⌋ = 21 := by
  sorry

-- Remove the #eval line as it's causing issues with universe levels
-- #eval train_passing_time 350 (68 * (1000 / 3600)) (8 * (1000 / 3600))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1204_120460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_product_lower_bound_l1204_120439

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse or hyperbola -/
structure Conic where
  center : Point
  foci : Point × Point
  eccentricity : ℝ

/-- Theorem: Product of eccentricities is greater than 1/3 -/
theorem eccentricity_product_lower_bound
  (ellipse hyperbola : Conic)
  (P : Point)
  (h_common_focus : ellipse.center = hyperbola.center ∧ ellipse.center = ⟨0, 0⟩)
  (h_shared_foci : ellipse.foci.1 = hyperbola.foci.1 ∧ ellipse.foci.2 = hyperbola.foci.2)
  (h_first_quadrant : P.x > 0 ∧ P.y > 0)
  (h_isosceles : (P.x - ellipse.foci.1.x)^2 + (P.y - ellipse.foci.1.y)^2 =
                 (P.x - ellipse.foci.2.x)^2 + (P.y - ellipse.foci.2.y)^2)
  (h_base_length : (P.x - ellipse.foci.1.x)^2 + (P.y - ellipse.foci.1.y)^2 = 100) :
  ellipse.eccentricity * hyperbola.eccentricity > 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_product_lower_bound_l1204_120439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_problem_l1204_120499

/-- Given a function h and its inverse f⁻¹, prove that 7c + 7d = 2 -/
theorem inverse_function_problem (c d : ℝ) :
  let h : ℝ → ℝ := λ x => 7 * x - 6
  let f : ℝ → ℝ := λ x => c * x + d
  let f_inv : ℝ → ℝ := Function.invFun f
  (∀ x, h x = f_inv x - 5) →
  Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f →
  7 * c + 7 * d = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_problem_l1204_120499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_reduction_is_five_percent_l1204_120415

/-- Calculates the percentage of the first price reduction given the original price,
    the total reduction, and the percentage of the second reduction. -/
noncomputable def first_reduction_percentage (original_price total_reduction second_reduction_percent : ℝ) : ℝ :=
  let remaining_after_second := 1 - second_reduction_percent / 100
  100 * (1 - (1 - total_reduction / original_price) / remaining_after_second)

/-- Theorem stating that given the specific conditions of the problem,
    the first reduction percentage is 5%. -/
theorem first_reduction_is_five_percent :
  first_reduction_percentage 500 44 4 = 5 := by
  sorry

-- Remove the #eval statement as it's not necessary for building
-- and might cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_reduction_is_five_percent_l1204_120415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_calculation_l1204_120444

/-- Given a 12% service fee, a $5 tip, and a total spend of $61, prove that the cost of the food is $50. -/
theorem restaurant_bill_calculation (service_fee_rate : ℝ) (tip : ℝ) (total_spent : ℝ) (food_cost : ℝ) : 
  service_fee_rate = 0.12 →
  tip = 5 →
  total_spent = 61 →
  food_cost + service_fee_rate * food_cost + tip = total_spent →
  food_cost = 50 := by
  intro h1 h2 h3 h4
  sorry

#check restaurant_bill_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_calculation_l1204_120444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_to_white_ratio_l1204_120402

/-- Represents the area of a circle with given radius -/
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

/-- Represents the area of a ring between two radii -/
noncomputable def ring_area (r1 r2 : ℝ) : ℝ := circle_area r2 - circle_area r1

theorem black_to_white_ratio :
  let r1 : ℝ := 2
  let r2 : ℝ := 4
  let r3 : ℝ := 6
  let r4 : ℝ := 8
  let white_area : ℝ := circle_area r1 + ring_area r2 r3
  let black_area : ℝ := ring_area r1 r2 + ring_area r3 r4
  black_area / white_area = 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_to_white_ratio_l1204_120402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_tan_function_l1204_120455

noncomputable def f (x : ℝ) : ℝ := 2 * Real.tan (3 * x - Real.pi / 4)

def is_symmetry_center (c : ℝ × ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (c.1 + x) = -f (c.1 - x)

theorem symmetry_center_of_tan_function :
  is_symmetry_center (-Real.pi/4, 0) f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_tan_function_l1204_120455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_decrease_l1204_120487

theorem stock_price_decrease (initial_price : ℝ) (h : initial_price > 0) :
  let increased_price := 1.15 * initial_price
  let decrease_percentage := (1 - initial_price / increased_price) * 100
  ∃ ε > 0, |decrease_percentage - 13.04| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_decrease_l1204_120487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_condition_l1204_120479

-- Define the function f(x) = a ln x - (1 - 1/x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - (1 - 1/x)

-- State the theorem
theorem inequality_condition (a : ℝ) : 
  (∀ x > 1, f a x > 0) ↔ a ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_condition_l1204_120479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_32_div_3_l1204_120405

-- Define the parabola and line equations
def parabola (y : ℝ) : ℝ := y^2
def line (y : ℝ) : ℝ := 2*y + 3

-- Define the lower and upper bounds of integration
def lower_bound : ℝ := -1
def upper_bound : ℝ := 3

-- Define the area of the enclosed shape
noncomputable def enclosed_area : ℝ := ∫ y in lower_bound..upper_bound, line y - parabola y

-- Theorem statement
theorem enclosed_area_is_32_div_3 : enclosed_area = 32/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_32_div_3_l1204_120405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_intersection_A_B_union_complement_A_B_l1204_120446

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 - x) + Real.log (x + 2) / Real.log 4

-- Define the domain set A
def A : Set ℝ := {x | -2 < x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x | (x - 2) * (x + 3) > 0}

-- Theorem for the domain of f
theorem domain_of_f : {x : ℝ | ∃ y, f x = y} = A := by sorry

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | 2 < x ∧ x ≤ 3} := by sorry

-- Theorem for (ℝ\A) ∪ B
theorem union_complement_A_B : (Set.univ \ A) ∪ B = {x | x ≤ -2 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_intersection_A_B_union_complement_A_B_l1204_120446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l1204_120408

-- Define the operation ※
noncomputable def star (a b : ℝ) : ℝ := a * b - a + (1/2) * b

-- Theorem statements
theorem part1 : star 5 (-4) = -27 := by sorry

theorem part2 : star (-2) (2/3) = 1 := by sorry

theorem part3 (a b : ℝ) (h : a - b = 2022) : star a b - star b a = -3033 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l1204_120408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l1204_120464

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of a sequence -/
noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ) (h_arith : arithmetic_sequence a)
  (h_ineq : -a 2015 < a 1 ∧ a 1 < -a 2016) :
  S a 2015 > 0 ∧ S a 2016 < 0 := by
  sorry

#check arithmetic_sequence_sum_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l1204_120464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l1204_120432

-- Define the function f(x) = 2ln(x) + 1/x
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 1 / x

-- State the theorem
theorem f_monotone_decreasing :
  ∀ x y : ℝ, 0 < x → x < y → y ≤ 1/2 → f y ≤ f x :=
by
  -- The proof is omitted and replaced with 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l1204_120432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1204_120476

/-- The parabola y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

/-- The focus of a parabola y = x^2 -/
noncomputable def focus : ℝ × ℝ := (0, 1/4)

/-- Theorem: The focus of the parabola y = x^2 has coordinates (0, 1/4) -/
theorem parabola_focus :
  focus = (0, 1/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1204_120476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumference_area_growth_l1204_120416

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ
  radius_pos : radius > 0

/-- Calculates the circumference of a circle -/
noncomputable def circumference (c : Circle) : ℝ := 2 * Real.pi * c.radius

/-- Calculates the area of a circle -/
noncomputable def area (c : Circle) : ℝ := Real.pi * c.radius ^ 2

/-- Theorem: For circles with increasing radii, the relationship between
    circumference and area shows linear growth in circumference and
    quadratic growth in area -/
theorem circumference_area_growth
  (c₁ c₂ : Circle)
  (h : c₁.radius < c₂.radius) :
  circumference c₁ < circumference c₂ ∧
  (circumference c₂ - circumference c₁) / (c₂.radius - c₁.radius) = 2 * Real.pi ∧
  area c₁ < area c₂ ∧
  (area c₂ - area c₁) / (c₂.radius ^ 2 - c₁.radius ^ 2) = Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumference_area_growth_l1204_120416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l1204_120417

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence. -/
noncomputable def CommonDifference (a : ℕ → ℝ) (h : ArithmeticSequence a) : ℝ :=
  Classical.choose h

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h : ArithmeticSequence a)
  (h1 : a 1 + a 5 = 10)
  (h2 : a 4 = 7) :
  CommonDifference a h = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l1204_120417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_marble_product_l1204_120495

/-- The number of marbles in the bag -/
def n : ℕ := 7

/-- The set of marble numbers -/
def marbles : Finset ℕ := Finset.range n

/-- The set of all pairs of different marbles -/
def marblePairs : Finset (ℕ × ℕ) :=
  (marbles.product marbles).filter (fun p => p.1 < p.2)

/-- The product of a pair of marbles -/
def pairProduct (p : ℕ × ℕ) : ℕ := p.1 * p.2

/-- The sum of products of all pairs -/
def productSum : ℕ := marblePairs.sum (fun p => pairProduct p)

/-- The number of different pairs -/
def pairCount : ℕ := marblePairs.card

/-- The expected value of the product of two randomly drawn marbles -/
noncomputable def expectedValue : ℚ := (productSum : ℚ) / pairCount

theorem expected_value_of_marble_product :
  expectedValue = 295 / 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_marble_product_l1204_120495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cereal_eating_time_l1204_120447

/-- The time it takes for three people to eat a certain amount of cereal together -/
def eating_time (quick_rate fat_rate thin_rate total_cereal : ℚ) : ℚ :=
  total_cereal / (quick_rate + fat_rate + thin_rate)

/-- Theorem: Given the eating rates of Miss Quick, Mr. Fat, and Mr. Thin,
    it takes 80/3 minutes for them to eat 4 pounds of cereal together -/
theorem cereal_eating_time :
  let quick_rate : ℚ := 1 / 15
  let fat_rate : ℚ := 1 / 20
  let thin_rate : ℚ := 1 / 30
  let total_cereal : ℚ := 4
  eating_time quick_rate fat_rate thin_rate total_cereal = 80 / 3 := by
  sorry

#eval eating_time (1/15 : ℚ) (1/20 : ℚ) (1/30 : ℚ) 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cereal_eating_time_l1204_120447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l1204_120461

def A : Set ℤ := {x | 1 < x ∧ x < 7}
def B : Set ℝ := {x | x ≥ 10 ∨ x ≤ 2}

theorem intersection_complement_equality :
  (A.image (coe : ℤ → ℝ)) ∩ (Set.univ \ B) = {3, 4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l1204_120461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scrap_cookie_radius_equals_small_cookie_radius_l1204_120466

/-- The radius of a small cookie -/
def small_cookie_radius : ℝ := 1.5

/-- The number of small cookies cut from the large dough -/
def num_small_cookies : ℕ := 8

/-- The radius of the large cookie dough -/
def large_dough_radius : ℝ := small_cookie_radius + 2 * small_cookie_radius

/-- The area of the large cookie dough -/
noncomputable def large_dough_area : ℝ := Real.pi * large_dough_radius ^ 2

/-- The area of a single small cookie -/
noncomputable def small_cookie_area : ℝ := Real.pi * small_cookie_radius ^ 2

/-- The total area of all small cookies -/
noncomputable def total_small_cookies_area : ℝ := (num_small_cookies : ℝ) * small_cookie_area

/-- The area of the leftover scrap -/
noncomputable def scrap_area : ℝ := large_dough_area - total_small_cookies_area

/-- The theorem stating that the radius of the scrap cookie is equal to the radius of a small cookie -/
theorem scrap_cookie_radius_equals_small_cookie_radius :
  Real.sqrt (scrap_area / Real.pi) = small_cookie_radius := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scrap_cookie_radius_equals_small_cookie_radius_l1204_120466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1204_120478

/-- A quadratic function f(x) = ax^2 + bx + c with vertex (3, 5), 
    vertical axis of symmetry, and passing through (1, 2) -/
def quadratic_function (a b c : ℚ) : ℚ → ℚ := λ x ↦ a * x^2 + b * x + c

theorem sum_of_coefficients (a b c : ℚ) :
  (∀ x, quadratic_function a b c x = a * x^2 + b * x + c) →
  (∀ x, quadratic_function a b c x = a * (x - 3)^2 + 5) →
  quadratic_function a b c 1 = 2 →
  a + b + c = 35 / 4 := by
  sorry

#check sum_of_coefficients

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1204_120478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_property_l1204_120428

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_decreasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f y < f x

noncomputable def power_function (m : ℝ) : ℝ → ℝ := fun x ↦ x^m

theorem power_function_property (m : ℝ) :
  is_even_function (power_function m) ∧
  is_decreasing_on (power_function m) (Set.Ioo 0 Real.pi) →
  m = -2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_property_l1204_120428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l1204_120465

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 3 then x^2 else 2^x

-- Theorem statement
theorem f_composition_value : f (f 2) = 16 := by
  -- Evaluate f(2)
  have h1 : f 2 = 4 := by
    simp [f]
    norm_num
  
  -- Evaluate f(4)
  have h2 : f 4 = 16 := by
    simp [f]
    norm_num
  
  -- Combine the results
  calc
    f (f 2) = f 4 := by rw [h1]
    _       = 16  := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l1204_120465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_for_equal_representation_l1204_120445

/-- Represents a number in a given base -/
def representationInBase (n : ℕ) (base : ℕ) : ℕ := 
  (n / base) * base + (n % base)

/-- Checks if all digits are valid in the given base -/
def validDigitsInBase (n : ℕ) (base : ℕ) : Prop :=
  ∀ d, d ∈ Nat.digits base n → d < base

theorem smallest_sum_for_equal_representation :
  ∃ (a b : ℕ), 
    a > 0 ∧ b > 0 ∧
    representationInBase 56 a = representationInBase 65 b ∧
    validDigitsInBase 56 a ∧ validDigitsInBase 65 b ∧
    (∀ (a' b' : ℕ), a' > 0 → b' > 0 → 
      representationInBase 56 a' = representationInBase 65 b' →
      validDigitsInBase 56 a' → validDigitsInBase 65 b' →
      a + b ≤ a' + b') ∧
    a + b = 13 :=
by
  -- The proof goes here
  sorry

#check smallest_sum_for_equal_representation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_for_equal_representation_l1204_120445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barbara_scores_theorem_l1204_120450

def scores : List ℕ := [80, 82, 85, 86, 90, 92, 95]

def alan_score_count : ℕ := 5
def barbara_score_count : ℕ := 2
def alan_mean : ℕ := 87

def barbara_scores : List ℕ := [92, 83]

theorem barbara_scores_theorem :
  (barbara_scores.sum.toFloat / barbara_score_count.toFloat = 87.5) ∧
  (barbara_scores.filter (λ x => x > 90)).length = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barbara_scores_theorem_l1204_120450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quadrilateral_area_l1204_120477

def side_lengths : List ℝ := [1, 4, 7, 8]

/-- Predicate to check if a real number is a valid area of a quadrilateral with given side lengths -/
def is_area_of_quadrilateral (A : ℝ) (s : List ℝ) : Prop :=
  ∃ (a b c d : ℝ), s = [a, b, c, d] ∧
  ∃ (x y z w : ℝ), x + y + z + w = 2 * Real.pi ∧
  A = (1/4) * Real.sqrt ((a + b + c + d) * (-a + b + c + d) * (a - b + c + d) * (a + b - c + d))

theorem largest_quadrilateral_area (s : List ℝ) (h : s = side_lengths) :
  ∃ (A : ℝ), A = 18 ∧ ∀ (B : ℝ), is_area_of_quadrilateral B s → B ≤ A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quadrilateral_area_l1204_120477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_proof_l1204_120451

/-- The curve on which point P lies -/
def curve (x y : ℝ) : Prop := x^2 - y - 2 * Real.log (Real.sqrt x) = 0

/-- The line to which we're calculating the distance -/
def line (x y : ℝ) : Prop := 4*x + 4*y + 1 = 0

/-- The shortest distance from a point on the curve to the line -/
noncomputable def shortest_distance : ℝ := (Real.sqrt 2 / 2) * (1 + Real.log 2)

/-- Theorem stating that the shortest_distance is indeed the shortest distance
    from any point on the curve to the line -/
theorem shortest_distance_proof :
  ∃ (x y : ℝ), curve x y ∧
    (∀ (x' y' : ℝ), line x' y' →
      shortest_distance ≤ Real.sqrt ((x - x')^2 + (y - y')^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_proof_l1204_120451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_exists_l1204_120426

theorem unique_divisor_exists : ∃! D : ℕ, 
  D > 0 ∧ 
  242 % D = 15 ∧ 
  698 % D = 27 ∧ 
  940 % D = 5 ∧
  D = 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_exists_l1204_120426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfying_set_eq_not_power_of_a_l1204_120437

/-- Sum of digits in base a -/
noncomputable def S_a (a : ℕ) (x : ℕ) : ℕ := sorry

/-- Number of digits in base a -/
noncomputable def F_a (a : ℕ) (x : ℕ) : ℕ := sorry

/-- Position of first non-zero digit from right in base a -/
noncomputable def f_a (a : ℕ) (x : ℕ) : ℕ := sorry

/-- Set of positive integers satisfying the given conditions -/
def satisfying_set (a : ℕ) (M : ℕ) : Set ℕ :=
  {n : ℕ | n > 0 ∧ ∃ f : ℕ → ℕ, Filter.Tendsto f Filter.atTop Filter.atTop ∧
    ∀ k, S_a a (f k * n) = S_a a n ∧ F_a a (f k * n) - f_a a (f k * n) > M}

/-- Set of positive integers that are not powers of a -/
def not_power_of_a (a : ℕ) : Set ℕ :=
  {n : ℕ | n > 0 ∧ ¬∃ α : ℕ, n = a ^ α}

theorem satisfying_set_eq_not_power_of_a (a M : ℕ) (ha : a > 1) (hM : M ≥ 2020) :
    satisfying_set a M = not_power_of_a a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfying_set_eq_not_power_of_a_l1204_120437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_profit_percentage_l1204_120411

/-- Calculate the profit percentage given cost price, selling price, and discount rate -/
noncomputable def profit_percentage (cost_price selling_price : ℝ) (discount_rate : ℝ) : ℝ :=
  let marked_price := selling_price / (1 - discount_rate)
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating the profit percentage for the given problem -/
theorem article_profit_percentage :
  let cost_price : ℝ := 47.50
  let selling_price : ℝ := 63.16
  let discount_rate : ℝ := 0.06
  abs (profit_percentage cost_price selling_price discount_rate - 32.97) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_profit_percentage_l1204_120411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_min_positive_sum_l1204_120404

theorem arithmetic_sequence_min_positive_sum (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d)  -- arithmetic sequence
  → (a 9 / a 8 < -1)  -- condition on a_9 and a_8
  → (∃ k : ℕ, ∀ n : ℕ, (n : ℝ) * (a 1 + a n) / 2 ≥ (k : ℝ) * (a 1 + a k) / 2)  -- S_n has a minimum value
  → (∀ n : ℕ, n < 16 → (n : ℝ) * (a 1 + a n) / 2 ≤ 0)  -- S_n ≤ 0 for n < 16
  → (16 : ℝ) * (a 1 + a 16) / 2 > 0  -- S_16 > 0
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_min_positive_sum_l1204_120404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_increase_proof_l1204_120481

/-- Represents the percentage increase in income for the second year -/
def income_increase_percentage : ℝ := 60

/-- Represents the savings rate in the first year -/
def savings_rate : ℝ := 20

theorem income_increase_proof (income₁ : ℝ) (income₂ : ℝ) (savings₁ : ℝ) (savings₂ : ℝ) 
    (expenditure₁ : ℝ) (expenditure₂ : ℝ) :
  savings₁ = (savings_rate / 100) * income₁ →
  income₂ = income₁ * (1 + income_increase_percentage / 100) →
  savings₂ = 2 * savings₁ →
  expenditure₁ = income₁ - savings₁ →
  expenditure₂ = income₂ - savings₂ →
  expenditure₁ + expenditure₂ = 2 * expenditure₁ →
  income_increase_percentage = 60 := by
  sorry

#check income_increase_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_increase_proof_l1204_120481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l1204_120453

/-- Represents a triangle with sides a, b, c and angles A, B, C. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle. -/
noncomputable def Triangle.area (t : Triangle) : ℝ := (1/2) * t.b * t.c * Real.sin t.A

/-- A triangle is acute if all its angles are less than π/2. -/
def Triangle.isAcute (t : Triangle) : Prop := t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

theorem triangle_area_range (t : Triangle) 
  (h_acute : t.isAcute)
  (h_angle_A : t.A = Real.pi/3)
  (h_side_a : t.a = 2 * Real.sqrt 3) :
  2 * Real.sqrt 3 < t.area ∧ t.area ≤ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l1204_120453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_transformation_l1204_120457

/-- Represents a transformation of a sinusoidal function -/
noncomputable def transform_sin (f : ℝ → ℝ) (h : ℝ) (k : ℝ) : ℝ → ℝ := 
  fun x => f ((x / k) + h)

/-- The original function y = sin(2x) -/
noncomputable def original_function : ℝ → ℝ := fun x => Real.sin (2 * x)

/-- The transformed function -/
noncomputable def transformed_function : ℝ → ℝ := 
  transform_sin original_function (Real.pi/3) 2

theorem sin_transformation :
  ∀ x : ℝ, transformed_function x = Real.sin (x + Real.pi/3) := by
  sorry

#check sin_transformation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_transformation_l1204_120457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_product_theorem_l1204_120442

theorem complex_sum_product_theorem (x y z : ℂ) 
  (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) :
  (Complex.abs x = Complex.abs y) → 
  (Complex.abs y = Complex.abs z) →
  (x + y + z = -Complex.I * Real.sqrt (3 : ℝ) / 2 - Complex.I * Real.sqrt (5 : ℝ)) →
  (x * y * z = Real.sqrt (3 : ℝ) + Complex.I * Real.sqrt (5 : ℝ)) →
  (x = Complex.mk x₁ x₂) →
  (y = Complex.mk y₁ y₂) →
  (z = Complex.mk z₁ z₂) →
  (x₁ * x₂ + y₁ * y₂ + z₁ * z₂)^2 = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_product_theorem_l1204_120442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_system_l1204_120456

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def system (A : ℝ) (x : ℝ) : Prop :=
  A * x^2 - 4 = 0 ∧ 3 + 2 * (x + ↑(floor x)) = 0

theorem solve_system : 
  ∃ b : ℝ, system 16 b ∧ b = -1/2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_system_l1204_120456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_condition_implies_k_values_l1204_120475

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x ≤ 2 then 2*x + 1
  else if 2 < x ∧ x ≤ 4 then 1 + x^2
  else 0  -- We define f as 0 outside the given intervals for completeness

-- State the theorem
theorem integral_condition_implies_k_values (k : ℝ) :
  k < 2 →
  (∫ (x : ℝ) in k..3, f x) = 40/3 →
  k = 0 ∨ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_condition_implies_k_values_l1204_120475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_f_domain_correct_l1204_120429

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 2*x + 3)

-- State the theorem
theorem f_monotone_decreasing :
  ∀ x y : ℝ, x ∈ Set.Icc 1 3 → y ∈ Set.Icc 1 3 → x ≤ y → f y ≤ f x := by
  sorry

-- Define the domain of f
def f_domain : Set ℝ := Set.Icc (-1) 3

-- State that the domain of f is [-1, 3]
theorem f_domain_correct : 
  ∀ x : ℝ, x ∈ f_domain ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_f_domain_correct_l1204_120429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_lightning_l1204_120470

/-- The speed of sound in feet per second -/
def speed_of_sound : ℚ := 1120

/-- The time delay between seeing lightning and hearing thunder in seconds -/
def time_delay : ℚ := 12

/-- The number of feet in a mile -/
def feet_per_mile : ℚ := 5280

/-- Rounds a rational number to the nearest quarter -/
def round_to_nearest_quarter (x : ℚ) : ℚ :=
  (⌊x * 4 + 1/2⌋ : ℚ) / 4

/-- The theorem stating that the distance to the lightning, rounded to the nearest quarter-mile, is 2.5 miles -/
theorem distance_to_lightning : 
  round_to_nearest_quarter ((speed_of_sound * time_delay) / feet_per_mile) = 5/2 := by
  sorry

#eval round_to_nearest_quarter ((speed_of_sound * time_delay) / feet_per_mile)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_lightning_l1204_120470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1204_120482

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x + x^2

-- State the theorem
theorem inequality_solution_set (x : ℝ) :
  (f (Real.log x) + f (Real.log (1/x)) < 2 * f 1) ↔ (1/Real.exp 1 < x ∧ x < Real.exp 1) :=
by
  sorry

#check inequality_solution_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1204_120482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l1204_120459

-- Define the line equation
def line_equation (x y α : ℝ) : Prop :=
  x * Real.cos α + Real.sqrt 3 * y + 2 = 0

-- Define the range of inclination angles
def inclination_range (θ : ℝ) : Prop :=
  (0 ≤ θ ∧ θ ≤ Real.pi / 6) ∨ (5 * Real.pi / 6 ≤ θ ∧ θ < Real.pi)

-- Theorem statement
theorem inclination_angle_range :
  ∀ θ : ℝ, 0 ≤ θ ∧ θ < Real.pi →
  (∃ x y α : ℝ, line_equation x y α ∧ θ = Real.arctan (-Real.cos α / Real.sqrt 3)) →
  inclination_range θ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l1204_120459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_angle_m_values_l1204_120458

theorem second_quadrant_angle_m_values (α : ℝ) (m : ℝ) :
  (π / 2 < α) ∧ (α < π) →  -- α is in the second quadrant
  (Real.sin α = (3 * m - 2) / (m + 3)) →
  (Real.cos α = (m - 5) / (m + 3)) →
  (m = 10 / 9 ∨ m = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_angle_m_values_l1204_120458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1204_120480

noncomputable def f (x : ℝ) : ℝ := Real.sin x * (Real.cos x - Real.sqrt 3 * Real.sin x)

def interval : Set ℝ := { x | -Real.pi/3 ≤ x ∧ x ≤ 5*Real.pi/12 }

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ f x) ∧
  (∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x) ∧
  (∃ (x : ℝ), x ∈ interval ∧ f x = 1 - Real.sqrt 3 / 2) ∧
  (∃ (x : ℝ), x ∈ interval ∧ f x = -Real.sqrt 3) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1204_120480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_intersection_point_l1204_120486

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The curve xy = 2 -/
def onCurve (p : Point) : Prop := p.x * p.y = 2

/-- A circle in the 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The four intersection points of the curve and the circle -/
def intersectionPoints : Finset Point := sorry

/-- The given three intersection points -/
noncomputable def p1 : Point := ⟨4, 1/2⟩
noncomputable def p2 : Point := ⟨-6, -1/3⟩
noncomputable def p3 : Point := ⟨1/4, 8⟩

/-- The fourth intersection point to be proved -/
noncomputable def p4 : Point := ⟨-2/3, -3⟩

theorem fourth_intersection_point (c : Circle) :
  (∀ p ∈ intersectionPoints, onCurve p ∧ onCircle p c) ∧
  p1 ∈ intersectionPoints ∧
  p2 ∈ intersectionPoints ∧
  p3 ∈ intersectionPoints ∧
  Finset.card intersectionPoints = 4 →
  p4 ∈ intersectionPoints ∧ onCurve p4 ∧ onCircle p4 c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_intersection_point_l1204_120486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_g_l1204_120491

-- Define the piecewise function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then 4 * x + 8 else 3 * x - 15

-- State the theorem
theorem solutions_of_g (x : ℝ) : g x = 5 ↔ x = -3/4 ∨ x = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_g_l1204_120491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_equation_l1204_120422

/-- Represents the average monthly growth rate of a commodity price -/
def x : Real := Real.mk 0  -- We define x as a real number with an arbitrary value

/-- The initial price of the commodity -/
def initial_price : ℝ := 7.5

/-- The final price of the commodity after 2 months -/
def final_price : ℝ := 8.4

/-- The number of months over which the price change occurs -/
def months : ℕ := 2

/-- Theorem stating that the equation correctly represents the price change -/
theorem price_change_equation : initial_price * (1 + x)^months = final_price := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_equation_l1204_120422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_prism_volume_l1204_120433

/-- 
Given a right prism with an isosceles triangle base, where:
- The base of the triangle has length a
- The angle at the base of the triangle is 45°
- The lateral surface area of the prism is equal to the sum of the areas of the bases

Prove that the volume of the prism is (a^3 * (√2 - 1)) / 8
-/
theorem right_prism_volume (a : ℝ) (h : a > 0) : 
  ∃ (volume : ℝ), 
    let base_angle : ℝ := π / 4
    let base_area : ℝ := a^2 / 2
    let perimeter : ℝ := a * (2 + Real.sqrt 2)
    let lateral_area : ℝ := 2 * base_area
    let height : ℝ := lateral_area / perimeter
    volume = (a^3 * (Real.sqrt 2 - 1)) / 8 ∧
    volume = base_area * height :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_prism_volume_l1204_120433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_last_equal_probability_l1204_120414

/-- Represents a lottery with a given number of students and tickets -/
structure Lottery where
  num_students : ℕ
  num_tickets : ℕ
  num_prize_tickets : ℕ

/-- The probability of a student winning the prize in a lottery -/
noncomputable def win_probability (l : Lottery) (student : ℕ) : ℝ :=
  (l.num_prize_tickets : ℝ) / (l.num_tickets : ℝ)

/-- Theorem: In a lottery where the number of students equals the number of tickets,
    and there is exactly one prize ticket, the probability of the first student
    winning is equal to the probability of the last student winning -/
theorem first_last_equal_probability (l : Lottery) 
    (h1 : l.num_students = l.num_tickets)
    (h2 : l.num_prize_tickets = 1) :
    win_probability l 1 = win_probability l l.num_students := by
  sorry

#check first_last_equal_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_last_equal_probability_l1204_120414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_babblese_word_count_l1204_120494

/-- The number of letters in the Babblese alphabet -/
def alphabet_size : ℕ := 6

/-- The maximum word length in the Babblese language -/
def max_word_length : ℕ := 4

/-- The number of possible words of length n in the Babblese language -/
def words_of_length (n : ℕ) : ℕ := alphabet_size ^ n

/-- The total number of possible words in the Babblese language -/
def total_words : ℕ := (Finset.range max_word_length).sum (λ n => words_of_length (n + 1))

theorem babblese_word_count :
  total_words = 1554 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_babblese_word_count_l1204_120494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1204_120483

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  9 * x^2 + 36 * x + 4 * y^2 - 8 * y + 1 = 0

/-- The distance between the foci of the ellipse -/
noncomputable def foci_distance : ℝ := 2 * Real.sqrt 195 / 3

/-- Theorem stating that the distance between the foci of the given ellipse is 2√195/3 -/
theorem ellipse_foci_distance :
  ∃ (x₀ y₀ x₁ y₁ a b : ℝ), 
    (∀ x y, ellipse_equation x y ↔ ((x - x₀)^2 / a^2 + (y - y₀)^2 / b^2 = 1)) ∧
    ((x₁ - x₀)^2 + (y₁ - y₀)^2 = foci_distance^2) ∧
    (∀ x y, ellipse_equation x y → (x - x₀)^2 / a^2 + (y - y₀)^2 / b^2 ≤ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1204_120483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l1204_120468

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x + 3)^2 + (y - 1)^2 = 2

-- Define points O and Q
def O : ℝ × ℝ := (0, 0)
def Q : ℝ × ℝ := (2, 2)

-- Define the area of triangle OPQ
noncomputable def triangle_area (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  abs (x * 2 - y * 2) / 2

-- Theorem statement
theorem min_triangle_area :
  ∃ (min_area : ℝ), min_area = 2 ∧
  ∀ (P : ℝ × ℝ), circle_eq P.1 P.2 → triangle_area P ≥ min_area :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l1204_120468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_organization_growth_after_four_years_l1204_120420

def organization_growth (initial_population : ℕ) (years : ℕ) : ℕ :=
  let growth_function := λ n ↦ 4 * n - 9
  Nat.iterate growth_function years initial_population

theorem organization_growth_after_four_years :
  organization_growth 21 4 = 4611 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_organization_growth_after_four_years_l1204_120420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l1204_120484

-- Define the set of given expressions
noncomputable def given_expressions : List ℝ := [Real.sqrt 8, Real.sqrt 10, Real.sqrt 0.5, 1 / Real.sqrt 3]

-- Define what it means to be a simplest quadratic radical
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (n : ℕ), x = Real.sqrt (n : ℝ) ∧ ∀ (m : ℕ), m ≠ 1 → ¬(n % (m * m) = 0)

-- State the theorem
theorem simplest_quadratic_radical :
  ∃ (x : ℝ), x ∈ given_expressions ∧ is_simplest_quadratic_radical x ∧
  ∀ (y : ℝ), y ∈ given_expressions → y ≠ x → ¬(is_simplest_quadratic_radical y) := by
  sorry

#check simplest_quadratic_radical

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l1204_120484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_positive_roots_polynomial2020_exists_polynomial_with_10_roots_l1204_120452

/-- A polynomial of degree 2020 with coefficients in {-1, 0, 1} -/
def Polynomial2020 := {f : Polynomial ℤ | ∀ i, i > 2020 → f.coeff i = 0} ∩ 
                      {f | ∀ i, f.coeff i ∈ ({-1, 0, 1} : Set ℤ)}

/-- The number of positive integer roots (including multiplicities) of a polynomial -/
noncomputable def num_positive_integer_roots (f : Polynomial ℤ) : ℕ := sorry

/-- Theorem: The maximum number of positive integer roots for polynomials in Polynomial2020 
    with no negative integer roots is 10 -/
theorem max_positive_roots_polynomial2020 (f : Polynomial ℤ) 
  (hf : f ∈ Polynomial2020)
  (h : ∀ x : ℤ, x < 0 → f.eval x ≠ 0) : 
  num_positive_integer_roots f ≤ 10 :=
sorry

/-- There exists a polynomial in Polynomial2020 with exactly 10 positive integer roots -/
theorem exists_polynomial_with_10_roots : 
  ∃ f : Polynomial ℤ, f ∈ Polynomial2020 ∧ num_positive_integer_roots f = 10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_positive_roots_polynomial2020_exists_polynomial_with_10_roots_l1204_120452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_calculation_l1204_120425

/-- Calculates the annual interest for a loan with quarterly compounding -/
noncomputable def annual_interest (principal : ℝ) (quarterly_rate : ℝ) : ℝ :=
  4 * (principal * (quarterly_rate / 4))

/-- Theorem: The annual interest on a $10,000 loan with 5% quarterly compound rate is $500 -/
theorem loan_interest_calculation :
  let principal : ℝ := 10000
  let quarterly_rate : ℝ := 0.05
  annual_interest principal quarterly_rate = 500 := by
  -- Unfold the definition of annual_interest
  unfold annual_interest
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_div_cancel]
  -- Check that 10000 * 0.05 = 500
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_calculation_l1204_120425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_18000_l1204_120448

/-- Represents the profit distribution for a business partnership -/
structure ProfitDistribution where
  partners : Nat
  ratios : List Nat
  totalProfit : Nat

/-- Calculates the maximum amount received by any partner -/
def maxPartnerProfit (pd : ProfitDistribution) : Nat :=
  let totalShares := pd.ratios.sum
  let valuePerShare := pd.totalProfit / totalShares
  match pd.ratios.maximum? with
  | some max => max * valuePerShare
  | none => 0

/-- Theorem: The maximum profit for the given scenario is $18,000 -/
theorem max_profit_is_18000 :
  let pd : ProfitDistribution := {
    partners := 5,
    ratios := [2, 4, 3, 5, 6],
    totalProfit := 60000
  }
  maxPartnerProfit pd = 18000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_18000_l1204_120448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1204_120493

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc 1 4

-- Define the function g(x) = f(x+2)
def g (x : ℝ) : ℝ := f (x + 2)

-- Theorem statement
theorem domain_of_g :
  {x | g x ∈ Set.range f} = Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1204_120493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grandmother_age_l1204_120498

theorem grandmother_age (n : ℕ) (x y : ℝ) :
  n > 0 →  -- number of grandmothers is positive
  y = x + 5 →  -- average age of grandmothers is 5 years more than grandfathers
  77 < (y + 2 * x) / 3 →  -- average age of all retirees is greater than 77
  (y + 2 * x) / 3 < 78 →  -- average age of all retirees is less than 78
  ∃ (k : ℤ), y = k →  -- average age of grandmothers is an integer
  y = 81 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grandmother_age_l1204_120498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_correct_l1204_120427

def solution : String := "equal"

theorem solution_is_correct : solution = "equal" := by
  rfl

#check solution_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_correct_l1204_120427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_neg_two_one_zero_point_condition_l1204_120406

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - (2*a + 1) * x

-- Theorem for part (I)
theorem max_value_when_a_neg_two :
  ∃ (x : ℝ), x > 0 ∧ f (-2) x = 1 ∧ ∀ (y : ℝ), y > 0 → f (-2) y ≤ f (-2) x :=
by sorry

-- Theorem for part (II)
theorem one_zero_point_condition (a : ℝ) :
  (∃! (x : ℝ), 0 < x ∧ x < Real.exp 1 ∧ f a x = 0) ↔ a < (1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_neg_two_one_zero_point_condition_l1204_120406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gamma_success_ratio_l1204_120488

/-- Represents a participant's performance in a two-day work efficiency study -/
structure Participant where
  day1_score : ℕ
  day1_attempted : ℕ
  day2_score : ℕ
  day2_attempted : ℕ

/-- The work efficiency study setup -/
structure WorkEfficiencyStudy where
  alpha : Participant
  gamma : Participant

/-- Conditions of the study -/
def valid_study (study : WorkEfficiencyStudy) : Prop :=
  -- Total points for each participant
  study.alpha.day1_score + study.alpha.day2_score = 600 ∧
  study.gamma.day1_score + study.gamma.day2_score = 600 ∧
  -- Alpha's scores
  study.alpha.day1_score = 180 ∧
  study.alpha.day1_attempted = 360 ∧
  study.alpha.day2_score = 150 ∧
  study.alpha.day2_attempted = 240 ∧
  -- Gamma's scores are positive integers
  study.gamma.day1_score > 0 ∧
  study.gamma.day2_score > 0 ∧
  -- Gamma's daily success ratios are less than Alpha's
  (study.gamma.day1_score : ℚ) * study.alpha.day1_attempted < (study.alpha.day1_score : ℚ) * study.gamma.day1_attempted ∧
  (study.gamma.day2_score : ℚ) * study.alpha.day2_attempted < (study.alpha.day2_score : ℚ) * study.gamma.day2_attempted ∧
  -- Gamma attempted fewer points than Alpha on day 1
  study.gamma.day1_attempted < study.alpha.day1_attempted ∧
  -- Total attempted points for Gamma
  study.gamma.day1_attempted + study.gamma.day2_attempted = 600

/-- Gamma's two-day success ratio -/
def gamma_success_ratio (study : WorkEfficiencyStudy) : ℚ :=
  (study.gamma.day1_score + study.gamma.day2_score : ℚ) / 600

/-- Theorem: The maximum two-day success ratio for Gamma is 67/120 -/
theorem max_gamma_success_ratio (study : WorkEfficiencyStudy) 
  (h : valid_study study) : 
  gamma_success_ratio study ≤ 67/120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gamma_success_ratio_l1204_120488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1204_120418

def A : Set ℝ := {x | (x + 1) * (x - 2) ≤ 0}
def B : Set ℝ := {x | ∃ n : ℤ, x = n}

theorem intersection_of_A_and_B :
  A ∩ B = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1204_120418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_pair_probability_l1204_120467

def roll_dice (n : ℕ) : ℕ := 6^n

def choose (n k : ℕ) : ℕ := Nat.choose n k

def factorial (n : ℕ) : ℕ := Nat.factorial n

def favorable_outcomes : ℕ :=
  -- One pair case
  6 * choose 4 2 * 5 * 4 +
  -- Two pairs case
  choose 6 2 * (factorial 4 / (factorial 2 * factorial 2))

def total_outcomes : ℕ := roll_dice 4

theorem dice_pair_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_pair_probability_l1204_120467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l1204_120412

/-- A parabola is defined by its quadratic equation coefficients -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola -/
noncomputable def focus (p : Parabola) : ℝ × ℝ :=
  let h := -p.b / (2 * p.a)
  let k := p.c - p.b^2 / (4 * p.a)
  (h, k + 1 / (4 * p.a))

/-- Theorem: The focus of the parabola y = 4x^2 - 8x + 5 is (1, 17/16) -/
theorem focus_of_specific_parabola :
  let p : Parabola := { a := 4, b := -8, c := 5 }
  focus p = (1, 17/16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l1204_120412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_principle_birth_months_l1204_120407

theorem pigeonhole_principle_birth_months :
  ∀ (students : Finset Nat) (birth_month : Nat → Nat),
    Finset.card students = 13 →
    (∀ s ∈ students, birth_month s ∈ Finset.range 12) →
    ∃ (s1 s2 : Nat), s1 ∈ students ∧ s2 ∈ students ∧ s1 ≠ s2 ∧ birth_month s1 = birth_month s2 :=
by
  sorry

#check pigeonhole_principle_birth_months

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_principle_birth_months_l1204_120407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_theorem_l1204_120490

/-- A parabola is defined by its equation in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola is a point (h, k + 1/(4a)) where (h, k) is the vertex -/
noncomputable def focus (p : Parabola) : ℝ × ℝ :=
  let h := -p.b / (2 * p.a)
  let k := p.c - p.b^2 / (4 * p.a)
  (h, k + 1 / (4 * p.a))

theorem parabola_focus_theorem (p : Parabola) 
    (h : p = Parabola.mk (-5) 10 (-2)) : 
    focus p = (1, 59/20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_theorem_l1204_120490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_tenth_of_number_l1204_120438

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The specific number we're rounding -/
def number : ℝ := 3967149.8587234

theorem round_to_nearest_tenth_of_number :
  roundToNearestTenth number = 3967149.9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_tenth_of_number_l1204_120438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_length_l1204_120430

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

theorem trapezium_other_side_length (a h area : ℝ) (ha : a > 0) (hh : h > 0) (harea : area > 0) :
  a = 26 ∧ h = 15 ∧ area = 330 →
  ∃ b : ℝ, b > 0 ∧ trapeziumArea a b h = area ∧ b = 18 := by
  intro h_given
  use 18
  constructor
  · exact Real.zero_lt_one.trans (by norm_num : (1 : ℝ) < 18)
  constructor
  · rw [trapeziumArea, h_given.1, h_given.2.1]
    field_simp
    norm_num
    rw [h_given.2.2]
    ring
  · rfl

#check trapezium_other_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_length_l1204_120430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_l1204_120471

theorem sum_difference 
  (a₁ d₁ n₁ a₂ d₂ n₂ : ℕ) 
  (h1 : a₁ = 2001)
  (h2 : d₁ = 1)
  (h3 : n₁ = 100)
  (h4 : a₂ = 51)
  (h5 : d₂ = 2)
  (h6 : n₂ = 50) :
  (n₁ * (2 * a₁ + (n₁ - 1) * d₁)) / 2 - (n₂ * (2 * a₂ + (n₂ - 1) * d₂)) / 2 = 200050 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_l1204_120471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_two_solutions_l1204_120462

/-- Given a triangle ABC with side lengths a = 18, b = 24, and angle A = 30°,
    there are exactly 2 possible triangles satisfying these conditions. -/
theorem triangle_two_solutions :
  ∃ (c₁ c₂ : ℝ), c₁ ≠ c₂ ∧
  (∀ c : ℝ, (c^2 = 18^2 + 24^2 - 2*18*24*(Real.cos (30 * π / 180))) ↔ (c = c₁ ∨ c = c₂)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_two_solutions_l1204_120462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1204_120485

-- Define the triangle
def triangle_sides : Fin 3 → ℕ := ![7, 24, 25]

-- Define what it means for a triangle to be right-angled
def is_right_triangle (sides : Fin 3 → ℕ) : Prop :=
  sides 0^2 + sides 1^2 = sides 2^2 ∨
  sides 0^2 + sides 2^2 = sides 1^2 ∨
  sides 1^2 + sides 2^2 = sides 0^2

-- Define the area calculation for a right triangle
noncomputable def right_triangle_area (sides : Fin 3 → ℕ) : ℚ :=
  (1/2) * sides 0 * sides 1

-- Theorem statement
theorem triangle_properties :
  is_right_triangle triangle_sides ∧
  right_triangle_area triangle_sides = 84 := by
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1204_120485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_partition_exists_l1204_120410

-- Define the set of natural numbers greater than one
def NatGreaterThanOne : Set ℕ := {n : ℕ | n > 1}

-- Define the property that a set is closed under the operation ab - 1
def ClosedUnderOperation (S : Set ℕ) : Prop :=
  ∀ a b, a ∈ S → b ∈ S → (a * b - 1) ∈ S

-- State the theorem
theorem no_partition_exists : ¬ ∃ (A B : Set ℕ),
  (A ⊆ NatGreaterThanOne) ∧
  (B ⊆ NatGreaterThanOne) ∧
  (A ∪ B = NatGreaterThanOne) ∧
  (A ∩ B = ∅) ∧
  (A ≠ ∅) ∧
  (B ≠ ∅) ∧
  ClosedUnderOperation A ∧
  ClosedUnderOperation B :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_partition_exists_l1204_120410
