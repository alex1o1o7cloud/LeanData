import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_2013_in_403rd_bracket_l249_24980

def odd_sequence (n : ℕ) : ℕ := 2 * n + 1

def bracket_size (k : ℕ) : ℕ :=
  match k % 4 with
  | 0 => 4
  | r => r

def bracket_start (k : ℕ) : ℕ :=
  if k = 0 then 0 else
  (bracket_start (k - 1) + bracket_size (k - 1))

theorem number_2013_in_403rd_bracket :
  ∃ i : ℕ, bracket_start 403 < i ∧ i ≤ bracket_start 403 + bracket_size 403 ∧ odd_sequence i = 2013 := by
  sorry

#eval odd_sequence 1006  -- Should output 2013
#eval bracket_size 403   -- Should output 3
#eval bracket_start 403  -- Should output the start of the 403rd bracket

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_2013_in_403rd_bracket_l249_24980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_union_equals_16_l249_24908

-- Define the triangle vertices
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (4, -2)
def C : ℝ × ℝ := (5, 3)

-- Define the reflection function about y = 0
def reflect (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Calculate the area of a triangle given its vertices
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

-- Theorem statement
theorem area_of_union_equals_16 :
  triangleArea A B C + triangleArea (reflect A) (reflect B) (reflect C) = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_union_equals_16_l249_24908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_test_point_0618_method_l249_24933

/-- The 0.618 method for selecting test points -/
theorem second_test_point_0618_method (a b : ℝ) (h : a = 2 ∧ b = 4) :
  let x₁ := a + ((Real.sqrt 5 - 1) / 2) * (b - a)
  let x₂ := a + (b - x₁)
  x₂ = 2.764 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_test_point_0618_method_l249_24933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_minimum_value_l249_24954

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Theorem for the solution set of the inequality
theorem inequality_solution_set :
  {x : ℝ | f (2 * x) ≤ f (x + 1)} = Set.Icc 0 1 := by sorry

-- Theorem for the minimum value
theorem minimum_value (a b : ℝ) (h : a + b = 2) :
  2 ≤ f (a^2) + f (b^2) ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ + b₀ = 2 ∧ f (a₀^2) + f (b₀^2) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_minimum_value_l249_24954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_second_diagonal_length_l249_24920

/-- Represents a rhombus with given area and one diagonal length -/
structure Rhombus where
  area : ℝ
  diagonal1 : ℝ

/-- Calculates the length of the second diagonal of a rhombus -/
noncomputable def secondDiagonalLength (r : Rhombus) : ℝ :=
  (2 * r.area) / r.diagonal1

/-- Theorem: The second diagonal of a rhombus with area 140 cm² and one diagonal 20 cm is 14 cm -/
theorem rhombus_second_diagonal_length :
  let r : Rhombus := { area := 140, diagonal1 := 20 }
  secondDiagonalLength r = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_second_diagonal_length_l249_24920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_inequality_l249_24967

def sequence_a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => if n % 2 = 0 then sequence_a (n + 1)
             else sequence_a ((n + 1) / 2) + sequence_a (n + 1)

theorem sequence_a_inequality (n : ℕ) (h : n > 0) : 
  sequence_a (2^n) > 2^(n^2 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_inequality_l249_24967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yunas_initial_money_l249_24940

theorem yunas_initial_money (initial_money : ℕ) : 
  (initial_money / 2) / 2 + 800 = initial_money / 4 + 800 ∧ 
  initial_money / 4 + 800 = initial_money / 2 → 
  initial_money = 3200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yunas_initial_money_l249_24940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probabilities_l249_24971

/-- Definition of the sample space for rolling two dice -/
def sampleSpace : Finset (Fin 6 × Fin 6) :=
  Finset.product (Finset.univ : Finset (Fin 6)) (Finset.univ : Finset (Fin 6))

/-- Event A: rolling two consecutive numbers -/
def eventA : Finset (Fin 6 × Fin 6) :=
  sampleSpace.filter (fun (x, y) => x.val + 1 = y.val ∨ y.val + 1 = x.val)

/-- Event B: rolling two of the same number -/
def eventB : Finset (Fin 6 × Fin 6) :=
  sampleSpace.filter (fun (x, y) => x = y)

/-- Event C: rolling two numbers with the same parity but different values -/
def eventC : Finset (Fin 6 × Fin 6) :=
  sampleSpace.filter (fun (x, y) => x ≠ y ∧ x.val % 2 = y.val % 2)

/-- Appreciation prize: remaining outcomes -/
def eventAppreciation : Finset (Fin 6 × Fin 6) :=
  sampleSpace \ (eventA ∪ eventB ∪ eventC)

theorem dice_probabilities :
  (eventA.card : Rat) / sampleSpace.card = 5 / 18 ∧
  (eventB.card : Rat) / sampleSpace.card = 1 / 6 ∧
  (eventC.card : Rat) / sampleSpace.card = 1 / 3 ∧
  (eventAppreciation.card : Rat) / sampleSpace.card = 2 / 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probabilities_l249_24971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l249_24936

open Real

/-- A monotonic function f on (0, +∞) satisfying the given conditions -/
noncomputable def f : ℝ → ℝ := sorry

/-- The property that f is monotonic on (0, +∞) -/
axiom f_monotonic : Monotone f

/-- The functional equation satisfied by f -/
axiom f_equation (x : ℝ) (h : x > 0) : f (f x - exp x + x) = exp 1

/-- The inequality satisfied by f and its derivative -/
axiom f_inequality (x : ℝ) (h : x > 0) (a : ℝ) : f x + deriv f x ≥ a * x

/-- The theorem stating the maximum value of a -/
theorem max_a_value : ∀ a : ℝ, (∀ x : ℝ, x > 0 → f x + deriv f x ≥ a * x) → a ≤ 2 * exp 1 - 1 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l249_24936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_18_l249_24937

/-- Represents a trapezoid with given dimensions -/
structure Trapezoid where
  upper_base : ℝ
  lower_base : ℝ
  leg1 : ℝ
  leg2 : ℝ

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ :=
  ((t.upper_base + t.lower_base) / 2) * 
  Real.sqrt (((t.leg1 * t.leg2) / (t.lower_base - t.upper_base))^2 + (t.lower_base - t.upper_base)^2)

/-- Theorem: The area of a trapezoid with upper base 5, lower base 10, and legs 3 and 4 is 18 -/
theorem trapezoid_area_is_18 : 
  trapezoid_area { upper_base := 5, lower_base := 10, leg1 := 3, leg2 := 4 } = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_18_l249_24937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_and_decreasing_l249_24900

noncomputable def a : ℕ → ℝ
  | 0 => 5  -- Adding the case for 0 to avoid the missing case error
  | 1 => 5
  | n + 2 => ((a (n + 1)) ^ (n + 1) + 2 ^ (n + 1) + 2 * 3 ^ (n + 1)) ^ (1 / (n + 2 : ℝ))

theorem a_formula_and_decreasing :
  (∀ n : ℕ, n ≥ 1 → a n = (2 ^ n + 3 ^ n) ^ (1 / n : ℝ)) ∧
  (∀ n : ℕ, n ≥ 1 → a n > a (n + 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_and_decreasing_l249_24900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_sqrt_2_l249_24997

/-- The function f(x) = x^2 - ln(x) -/
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

/-- The line equation x - y - 2 = 0 -/
def line (x y : ℝ) : Prop := x - y - 2 = 0

/-- The minimum distance between a point on f(x) and a point on the line -/
noncomputable def min_distance : ℝ := Real.sqrt 2

/-- Theorem stating that the minimum distance is √2 -/
theorem min_distance_is_sqrt_2 :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    y₁ = f x₁ ∧
    line x₂ y₂ ∧
    ∀ (x₃ y₃ x₄ y₄ : ℝ),
      y₃ = f x₃ → line x₄ y₄ →
      Real.sqrt ((x₃ - x₄)^2 + (y₃ - y₄)^2) ≥ min_distance :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_sqrt_2_l249_24997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_A_theorem_l249_24939

/-- Represents the ratio of liquids A, B, and C in a mixture --/
structure Ratio where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the initial amount of liquid A given the initial ratio, 
    final ratio, and amount of mixture drawn off and replaced --/
noncomputable def initialAmountA (initialRatio : Ratio) (finalRatio : Ratio) 
                   (drawnOff : ℝ) (replaced : ℝ) : ℝ :=
  sorry

theorem initial_amount_A_theorem 
  (initialRatio : Ratio)
  (finalRatio : Ratio)
  (drawnOff : ℝ)
  (replaced : ℝ)
  (h1 : initialRatio = Ratio.mk 7 5 3)
  (h2 : finalRatio = Ratio.mk 7 9 3)
  (h3 : drawnOff = 18)
  (h4 : replaced = 18)
  : ∃ (ε : ℝ), ε > 0 ∧ |initialAmountA initialRatio finalRatio drawnOff replaced - 54| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_A_theorem_l249_24939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_fixed_points_l249_24904

/-- A line that always passes through a fixed point regardless of the value of m -/
structure FixedPointLine where
  a : ℚ
  b : ℚ
  c : ℚ
  h : ∀ m : ℚ, ∃ x y : ℚ, a * m * x + b * y + c * m + (1 - a - c) = 0

/-- The fixed point that a FixedPointLine always passes through -/
def fixedPoint (l : FixedPointLine) : ℚ × ℚ :=
  (-(1 - l.a - l.c) / l.a, -(1 - l.a - l.c) / l.b)

theorem distance_between_fixed_points :
  let l₁ : FixedPointLine := ⟨1, 1, 2, sorry⟩
  let l₂ : FixedPointLine := ⟨1, 1, -1, sorry⟩
  let A := fixedPoint l₁
  let B := fixedPoint l₂
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_fixed_points_l249_24904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cost_at_twenty_l249_24991

/-- Represents the number of people from unit B -/
def x : ℕ → ℕ := id

/-- Cost function for travel agency A -/
def costA (x : ℕ) : ℕ := 150 * x + 3300

/-- Cost function for travel agency B -/
def costB (x : ℕ) : ℕ := 210 * x + 2100

/-- The original price of each ticket -/
def originalPrice : ℕ := 300

/-- The number of people from unit A -/
def peopleFromA : ℕ := 10

/-- The number of full-price tickets for agency A's discount -/
def fullPriceTickets : ℕ := 12

/-- The discount rate for agency B -/
def discountRateB : ℚ := 7/10

theorem equal_cost_at_twenty :
  costA 20 = costB 20 := by
  rfl

#eval costA 20
#eval costB 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cost_at_twenty_l249_24991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2a_value_l249_24928

noncomputable def f (A : ℝ) (x : ℝ) : ℝ := A * Real.sin (Real.pi * x / 4 + Real.pi / 4)

theorem cos_2a_value (A : ℝ) (a : ℝ) (h1 : f A (-2015) = 3) 
  (h2 : a ∈ Set.Icc 0 Real.pi) 
  (h3 : f A (4 * a / Real.pi - 1) + f A (4 * a / Real.pi + 1) = 3 / 5) : 
  Real.cos (2 * a) = -7 / 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2a_value_l249_24928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_radius_sum_l249_24905

/-- A hexagon inscribed in a circle with alternating side lengths -/
structure InscribedHexagon where
  radius : ℝ
  side1 : ℝ
  side2 : ℝ
  h1 : side1 > 0
  h2 : side2 > 0

/-- The specific hexagon from the problem -/
noncomputable def specialHexagon : InscribedHexagon where
  radius := 5 + Real.sqrt 267
  side1 := 22
  side2 := 20
  h1 := by norm_num
  h2 := by norm_num

/-- The theorem to be proved -/
theorem hexagon_radius_sum (h : InscribedHexagon) 
  (h_side1 : h.side1 = 22) 
  (h_side2 : h.side2 = 20) 
  (h_radius : ∃ (p q : ℕ), h.radius = p + Real.sqrt (q : ℝ)) : 
  ∃ (p q : ℕ), h.radius = p + Real.sqrt (q : ℝ) ∧ p + q = 272 := by
  sorry

#check hexagon_radius_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_radius_sum_l249_24905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_tangent_relation_l249_24924

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the altitudes and their intersections with the circumcircle
structure Altitudes (t : Triangle) where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  A' : ℝ × ℝ
  B' : ℝ × ℝ
  C' : ℝ × ℝ

-- Define the side lengths
noncomputable def side_lengths (t : Triangle) : ℝ × ℝ × ℝ := by
  sorry

-- Define the angles
noncomputable def angles (t : Triangle) : ℝ × ℝ × ℝ := by
  sorry

-- Define the distances from each vertex to the intersection of its altitude with the circumcircle
noncomputable def altitude_lengths (t : Triangle) (alt : Altitudes t) : ℝ × ℝ × ℝ := by
  sorry

-- State the theorem
theorem altitude_tangent_relation (t : Triangle) (alt : Altitudes t) :
  let (a, b, c) := side_lengths t
  let (A, B, C) := angles t
  let (DA', EB', FC') := altitude_lengths t alt
  (a / DA' + b / EB' + c / FC') = 2 * Real.tan A * Real.tan B * Real.tan C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_tangent_relation_l249_24924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petrol_percentage_is_30_percent_l249_24944

/-- Calculates the percentage of income spent on petrol given petrol and house rent expenses -/
noncomputable def petrol_percentage (petrol_expense : ℚ) (house_rent_expense : ℚ) : ℚ :=
  let total_income := (house_rent_expense * 5) + petrol_expense
  (petrol_expense / total_income) * 100

/-- Theorem stating that the percentage of income spent on petrol is 30% 
    given specific petrol and house rent expenses -/
theorem petrol_percentage_is_30_percent :
  petrol_percentage 300 140 = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petrol_percentage_is_30_percent_l249_24944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_price_theorem_l249_24992

/-- Represents the financial details of a product sale --/
structure ProductSale where
  sellingPrice : ℝ
  profitRate : ℝ
  discountRate : ℝ
  lossRate : ℝ
  salesTaxRate : ℝ
  transactionCost : ℝ

/-- Calculates the cost price of a product given its sale details --/
noncomputable def calculateCostPrice (sale : ProductSale) : ℝ :=
  if sale.lossRate > 0 then
    sale.sellingPrice / (1 - sale.lossRate)
  else if sale.profitRate > 0 then
    (sale.sellingPrice / (1 + sale.profitRate)) / (1 - sale.discountRate) - sale.transactionCost
  else
    sale.sellingPrice

/-- The main theorem stating the total cost price of the three products --/
theorem total_cost_price_theorem (ε : ℝ) (h_ε : ε > 0) :
  let product1 := ProductSale.mk 600 0.25 0.05 0 0 0
  let product2 := ProductSale.mk 800 0 0 0.20 0.10 0
  let product3 := ProductSale.mk 1000 0.30 0 0 0 50
  let totalCostPrice := calculateCostPrice product1 + calculateCostPrice product2 + calculateCostPrice product3
  abs (totalCostPrice - 2224.49) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_price_theorem_l249_24992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l249_24923

-- Define the circle and its properties
def circle_radius : ℝ := 6

-- Define the shaded area function
noncomputable def shaded_area (r : ℝ) : ℝ := 2 * (r^2) + 2 * (Real.pi/4 * r^2)

-- Theorem statement
theorem shaded_area_calculation :
  shaded_area circle_radius = 36 + 18 * Real.pi := by
  -- Unfold the definition of shaded_area
  unfold shaded_area
  -- Simplify the expression
  simp [circle_radius]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l249_24923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_value_l249_24951

noncomputable def f (ω θ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + θ - Real.pi / 6)

theorem theta_value (ω θ : ℝ) :
  (∃ T > 0, ∀ x, f ω θ (x + T) = f ω θ x ∧ ∀ S, 0 < S → S < T → ∃ y, f ω θ (y + S) ≠ f ω θ y) →
  (∀ x, f ω θ (x + Real.pi / 6) = -f ω θ (-x + Real.pi / 6)) →
  θ = -Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_value_l249_24951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_around_point_l249_24998

theorem angle_around_point (y z : ℝ) : 
  y + z + 175 = 360 → 
  z = y + 10 → 
  ∃ (n : ℕ), n ≥ 88 ∧ n ≤ 89 ∧ (n : ℝ) - y < 1 ∧ y - (n : ℝ) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_around_point_l249_24998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_theorem_l249_24909

/-- An angle in the plane with vertex at the origin. -/
structure AngleAtOrigin where
  /-- The angle measure in radians. -/
  α : Real
  /-- The initial side of the angle coincides with the positive x-axis. -/
  initial_side_on_x_axis : True
  /-- The terminal side of the angle lies on the line y = 2x. -/
  terminal_side_on_line : Real.tan α = 2

/-- 
If an angle α is formed such that its vertex is at the origin, 
its initial side is on the positive x-axis, and its terminal side 
lies on the line y = 2x, then (sin α + cos α) / (sin α - cos α) = 3.
-/
theorem angle_ratio_theorem (θ : AngleAtOrigin) : 
  (Real.sin θ.α + Real.cos θ.α) / (Real.sin θ.α - Real.cos θ.α) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_theorem_l249_24909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l249_24922

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (3 * x + Real.pi / 3)

theorem min_shift_for_symmetry :
  ∀ θ : ℝ, θ > 0 →
  (∀ x : ℝ, f (x - θ) = f (-x - θ)) →
  ∀ φ : ℝ, φ > 0 → (∀ x : ℝ, f (x - φ) = f (-x - φ)) →
  θ ≤ φ →
  θ = 5 * Real.pi / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l249_24922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l249_24950

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- Condition: f(p) and f(q) are relatively prime for any two distinct primes p and q -/
def RelativelyPrimeOnPrimes (f : IntPolynomial) : Prop :=
  ∀ p q : ℕ, Nat.Prime p → Nat.Prime q → p ≠ q → (Int.gcd (f.eval p) (f.eval q) = 1)

/-- The theorem to be proved -/
theorem polynomial_characterization (f : IntPolynomial) 
  (h : RelativelyPrimeOnPrimes f) : 
  ∃ m : ℕ, m > 0 ∧ (f = Polynomial.monomial m 1 ∨ f = Polynomial.monomial m (-1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l249_24950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_common_divisors_150_30_l249_24913

def divisors (n : ℤ) : Set ℤ :=
  {d : ℤ | d ∣ n}

def common_divisors (a b : ℤ) : Set ℤ :=
  (divisors a) ∩ (divisors b)

theorem product_of_common_divisors_150_30 :
  (Finset.prod (Finset.filter (λ d => d ∣ 150 ∧ d ∣ 30) (Finset.range 31)) id) ^ 2 = 16443022500 := by
  sorry

#eval (Finset.prod (Finset.filter (λ d => d ∣ 150 ∧ d ∣ 30) (Finset.range 31)) id) ^ 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_common_divisors_150_30_l249_24913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_on_interval_f_monotone_increasing_condition_l249_24930

/-- Function f(x) with parameter m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - m * Real.log x

/-- Theorem for the minimum and maximum values of f(x) when m = 2 on [1, e] -/
theorem f_min_max_on_interval :
  let f₂ := f 2
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc 1 (Real.exp 1), f₂ x ≥ min ∧ f₂ x ≤ max) ∧
    min = 1 - Real.log 2 ∧
    max = ((Real.exp 1)^2 - 4) / 2 := by sorry

/-- Theorem for the range of m that makes f(x) monotonically increasing on (1/2, +∞) -/
theorem f_monotone_increasing_condition :
  ∀ m : ℝ, (∀ x > (1/2 : ℝ), Monotone (f m)) ↔ m ≤ (1/4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_on_interval_f_monotone_increasing_condition_l249_24930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_in_triangle_l249_24958

/-- Given a triangle ABC, where AD is the median to side BC and E is the midpoint of AD,
    prove that the vector EB is equal to 3/4 AB - 1/4 AC. -/
theorem vector_in_triangle (A B C D E : EuclideanSpace ℝ (Fin 3)) :
  (D - B) = (1 / 2 : ℝ) • (C - B) →  -- AD is the median to side BC
  E = (1 / 2 : ℝ) • (A + D) →        -- E is the midpoint of AD
  (E - B) = (3 / 4 : ℝ) • (A - B) - (1 / 4 : ℝ) • (A - C) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_in_triangle_l249_24958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_fraction_sum_2007_l249_24956

theorem unit_fraction_sum_2007 : 
  let count := Finset.filter (fun (p : ℕ × ℕ) => 
    let (m, n) := p
    m ≠ n ∧ 
    m > 0 ∧ 
    n > 0 ∧ 
    (1 : ℚ) / 2007 = 1 / m + 1 / n) (Finset.range 4014 ×ˢ Finset.range 4014)
  count.card / 2 = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_fraction_sum_2007_l249_24956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l249_24906

open Set Real

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * Real.sin x - 1)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | ∃ k : ℤ, π/6 + 2*k*π < x ∧ x < 5*π/6 + 2*k*π} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l249_24906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boiling_point_celsius_l249_24995

/-- Conversion function from Celsius to Fahrenheit -/
noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ := c * (9/5) + 32

/-- Conversion function from Fahrenheit to Celsius -/
noncomputable def fahrenheit_to_celsius (f : ℝ) : ℝ := (f - 32) * (5/9)

/-- The boiling point of water in Fahrenheit -/
def boiling_point_f : ℝ := 212

/-- The melting point of ice in Fahrenheit -/
def melting_point_f : ℝ := 32

/-- The melting point of ice in Celsius -/
def melting_point_c : ℝ := 0

theorem boiling_point_celsius : 
  fahrenheit_to_celsius boiling_point_f = 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boiling_point_celsius_l249_24995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oldest_child_age_l249_24925

/-- Proves that the oldest child is 19 years old given the conditions of the problem -/
theorem oldest_child_age (n : ℕ) (avg : ℝ) (diff : ℕ) : 
  n = 7 → 
  avg = 10 → 
  diff = 3 → 
  let ages := List.range n |>.map (λ i => avg + (i - (n - 1) / 2 : ℝ) * diff)
  ages.sum / n = avg → 
  List.Pairwise (· ≠ ·) ages → 
  ages.maximum? = some 19 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oldest_child_age_l249_24925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l249_24965

/-- Represents a hyperbola with parameters a, b, and c -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_c_eq : c^2 = a^2 + b^2

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

/-- The inradius of a triangle formed by the origin, left vertex, and the intersection point -/
noncomputable def inradius (h : Hyperbola) : ℝ := (h.a * h.b) / (3 * h.c)

/-- The main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_inradius : inradius h = (h.a * h.b) / (3 * h.c)) :
  eccentricity h = (2 + Real.sqrt 10) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l249_24965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l249_24953

theorem problem_solution (a b c : ℤ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_eq1 : a^2 - b^2 - c^2 + a*b = 2100)
  (h_eq2 : a^2 + 2*b^2 + 4*c^2 - 3*a*b - 2*a*c - b*c = -2000) :
  a = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l249_24953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conjunctivitis_infection_rate_l249_24917

/-- Represents the average number of people infected by one person in each round -/
def average_infections : ℕ → Prop := sorry

/-- The total number of infected people after two rounds of infection -/
def total_infections (avg : ℕ) : ℕ := 1 + avg + avg * avg

theorem conjunctivitis_infection_rate :
  ∃ (avg : ℕ), average_infections avg ∧ total_infections avg = 144 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conjunctivitis_infection_rate_l249_24917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_cut_probability_is_three_fourths_l249_24972

/-- The probability that both resulting pieces will have a length greater than 1/8 meter 
    when a 1-meter rope is randomly cut from the middle. -/
noncomputable def rope_cut_probability : ℝ := 3/4

/-- Theorem stating that the probability of both resulting pieces having a length 
    greater than 1/8 meter when a 1-meter rope is randomly cut from the middle is 3/4. -/
theorem rope_cut_probability_is_three_fourths : 
  rope_cut_probability = 3/4 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_cut_probability_is_three_fourths_l249_24972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_trees_correct_l249_24952

/-- Given a yard with trees planted at equal distances, calculates the distance between consecutive trees. -/
noncomputable def distanceBetweenTrees (yardLength : ℝ) (numTrees : ℕ) : ℝ :=
  yardLength / (numTrees - 1 : ℝ)

/-- Theorem stating that the distance between consecutive trees is correct given the problem conditions. -/
theorem distance_between_trees_correct (yardLength : ℝ) (numTrees : ℕ) 
  (h1 : yardLength = 1500)
  (h2 : numTrees = 52)
  (h3 : numTrees > 1) :
  ∃ ε > 0, |distanceBetweenTrees yardLength numTrees - 29.41| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_trees_correct_l249_24952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_angle_bisector_segment_length_l249_24975

/-- Represents a square with vertices A, B, C, D in 2D space. -/
structure Square (A B C D : ℝ × ℝ) : Prop where
  side_length : ∀ (X Y : ℝ × ℝ), (X, Y) ∈ [(A, B), (B, C), (C, D), (D, A)] → dist X Y = 1

/-- Defines the angle bisector of ∠APB. -/
def angle_bisector (A P B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {Q | ∃ (t : ℝ), 0 < t ∧ Q = (1 - t) • A + t • P ∧ dist A Q / dist B Q = dist A P / dist B P}

/-- Defines a line segment between two points. -/
def line_segment (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B}

/-- Main theorem about the length of the segment formed by angle bisector intersections. -/
theorem square_angle_bisector_segment_length 
  (A B C D : ℝ × ℝ) 
  (h_square : Square A B C D) :
  ∃ (E F : ℝ × ℝ), 
    (∀ (P : ℝ × ℝ), P ∈ line_segment C D → 
      ∃ (Q : ℝ × ℝ), Q ∈ angle_bisector A P B ∧ Q ∈ line_segment A B) ∧
    (∀ (Q : ℝ × ℝ), (∃ (P : ℝ × ℝ), P ∈ line_segment C D ∧ Q ∈ angle_bisector A P B) → 
      Q ∈ line_segment E F) ∧
    dist E F = 3 - 2 * Real.sqrt 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_angle_bisector_segment_length_l249_24975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_f_minus_x_is_closed_interval_l249_24907

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then -x
  else x - 1

-- Define the range of f(x) - x
def range_f_minus_x : Set ℝ := { y | ∃ x ∈ Set.Icc (-4) 4, f x - x = y }

-- Theorem statement
theorem range_f_minus_x_is_closed_interval :
  range_f_minus_x = Set.Icc (-4) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_f_minus_x_is_closed_interval_l249_24907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_nine_l249_24903

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1/3)^x else Real.log x / Real.log 3

-- State the theorem
theorem f_composition_equals_nine : f (f (1/9)) = 9 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_nine_l249_24903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l249_24926

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x^3 / (x^3 + 2*y^2*(Real.sqrt (z*x)))) + 
  (y^3 / (y^3 + 2*z^2*(Real.sqrt (x*y)))) + 
  (z^3 / (z^3 + 2*x^2*(Real.sqrt (y*z)))) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l249_24926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_to_eighth_power_l249_24964

noncomputable def z : ℂ := (2/3 : ℂ) + (5/6 : ℂ) * Complex.I

theorem magnitude_of_z_to_eighth_power :
  Complex.abs (z^8) = 2825761/1679616 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_to_eighth_power_l249_24964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stereographic_projection_is_inversion_circle_through_B_maps_to_line_circle_not_through_B_maps_to_circle_angle_preservation_l249_24916

-- Define the sphere and plane
def Sphere : Type := ℝ × ℝ × ℝ
def Plane : Type := ℝ × ℝ

-- Define points on the sphere and plane
variable (S : Type) [MetricSpace S]
variable (P : Type) [MetricSpace P]
variable (A B : S)
variable (X : S)
variable (Y : P)

-- Define stereographic projection
noncomputable def stereographic_projection (S P : Type) [MetricSpace S] [MetricSpace P] (B : S) (X : S) : P :=
  sorry

-- Define inversion
noncomputable def inversion (S P : Type) [MetricSpace S] [MetricSpace P] (B : S) (X : S) : P :=
  sorry

-- Define circle on sphere
def circle_on_sphere (S : Type) [MetricSpace S] : Set S :=
  sorry

-- Define line on plane
def line_on_plane (P : Type) [MetricSpace P] : Set P :=
  sorry

-- Define angle between circles
noncomputable def angle_between_circles (S : Type) [MetricSpace S] (c1 c2 : Set S) : ℝ :=
  sorry

-- Theorem statements
theorem stereographic_projection_is_inversion
  (S P : Type) [MetricSpace S] [MetricSpace P] (B : S) :
  ∀ (X : S), stereographic_projection S P B X = inversion S P B X :=
by sorry

theorem circle_through_B_maps_to_line
  (S P : Type) [MetricSpace S] [MetricSpace P] (B : S) :
  ∀ (c : Set S), B ∈ c →
    ∃ (l : Set P), ∀ (X : S), X ∈ c →
      stereographic_projection S P B X ∈ l :=
by sorry

theorem circle_not_through_B_maps_to_circle
  (S P : Type) [MetricSpace S] [MetricSpace P] (B : S) :
  ∀ (c : Set S), B ∉ c →
    ∃ (c' : Set P), ∀ (X : S), X ∈ c →
      stereographic_projection S P B X ∈ c' :=
by sorry

theorem angle_preservation
  (S P : Type) [MetricSpace S] [MetricSpace P] (B : S) :
  ∀ (c1 c2 : Set S),
    angle_between_circles S c1 c2 =
    angle_between_circles P (Set.image (stereographic_projection S P B) c1) (Set.image (stereographic_projection S P B) c2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stereographic_projection_is_inversion_circle_through_B_maps_to_line_circle_not_through_B_maps_to_circle_angle_preservation_l249_24916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_2s_l249_24990

/-- Represents the time it takes for a train to cross a pole -/
noncomputable def train_crossing_time (speed_km_hr : ℝ) (length_m : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * 1000 / 3600
  length_m / speed_m_s

/-- Theorem stating that a train with given speed and length takes approximately 2 seconds to cross a pole -/
theorem train_crossing_time_approx_2s :
  ∃ ε > 0, |train_crossing_time 6 3.3333333333333335 - 2| < ε :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_2s_l249_24990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_representation_unique_factorial_representation_exists_unique_l249_24973

/-- Factorial representation of a natural number -/
def FactorialRep (n : ℕ) : (ℕ → ℕ) → Prop :=
  λ a => (n = ∑' k, a k * Nat.factorial k) ∧ 
         (∀ k, a k ≤ k)

/-- Uniqueness of factorial representation -/
theorem factorial_representation_unique (n : ℕ) : 
  ∃! a, FactorialRep n a :=
sorry

/-- Existence and uniqueness of factorial representation for all natural numbers -/
theorem factorial_representation_exists_unique : 
  ∀ n : ℕ, ∃! a, FactorialRep n a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_representation_unique_factorial_representation_exists_unique_l249_24973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_odd_nor_even_l249_24949

noncomputable def f (x : ℝ) : ℝ := 1 / (3^(x-1)) - 3

theorem f_neither_odd_nor_even :
  (∃ x : ℝ, f (-x) ≠ f x) ∧ (∃ x : ℝ, f (-x) ≠ -f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_odd_nor_even_l249_24949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_y_coordinate_l249_24942

/-- A point on the parabola y = x² -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  eq : y = x^2

/-- The line passing through a parabola point with slope 2x -/
def lineThrough (p : ParabolaPoint) : Set (ℝ × ℝ) :=
  {(x, y) | y - p.y = 2 * p.x * (x - p.x)}

/-- The intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Set (ℝ × ℝ)) : ℝ × ℝ :=
  sorry  -- Definition of intersection point

/-- An equilateral triangle formed by the intersection of three lines -/
structure EquilateralTriangle where
  p1 : ParabolaPoint
  p2 : ParabolaPoint
  p3 : ParabolaPoint
  isEquilateral : Prop

/-- The center of an equilateral triangle -/
noncomputable def center (t : EquilateralTriangle) : ℝ × ℝ :=
  sorry  -- Definition of triangle center

/-- The theorem stating that the y-coordinate of the center is always -1/4 -/
theorem center_y_coordinate (t : EquilateralTriangle) :
    (center t).2 = -1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_y_coordinate_l249_24942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_upper_bound_fractional_part_l249_24989

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The smallest upper bound for the fractional part problem -/
theorem smallest_upper_bound_fractional_part (C : ℝ) :
  (C = φ - 1) ↔
  (∀ (x y : ℕ+), x ≠ y →
    min (frac (Real.sqrt (x.val^2 + 2*y.val))) (frac (Real.sqrt (y.val^2 + 2*x.val))) < C) ∧
  (∀ ε > 0, ∃ (x y : ℕ+), x ≠ y ∧
    min (frac (Real.sqrt (x.val^2 + 2*y.val))) (frac (Real.sqrt (y.val^2 + 2*x.val))) > C - ε) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_upper_bound_fractional_part_l249_24989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l249_24996

-- Define the domain for both functions
def Domain : Set ℝ := {x : ℝ | x ≠ -2 ∧ x^2 ≥ 1 ∧ x > -2}

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((x^2 - 1) / (x + 2))
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x^2 - 1) / Real.sqrt (x + 2)

-- Theorem statement
theorem f_equals_g : ∀ x ∈ Domain, f x = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l249_24996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_triangle_area_l249_24955

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to A
  b : ℝ  -- Side opposite to B
  c : ℝ  -- Side opposite to C

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi ∧
  (Real.cos t.B * Real.cos t.C - Real.sin t.B * Real.sin t.C) = 1/2 ∧
  t.a = 2 * Real.sqrt 3 ∧
  t.b + t.c = 4

-- Theorem for part I
theorem angle_A_value (t : Triangle) (h : triangle_conditions t) : t.A = 2*Real.pi/3 := by
  sorry

-- Theorem for part II
theorem triangle_area (t : Triangle) (h : triangle_conditions t) : 
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_triangle_area_l249_24955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_triple_4444_power_l249_24968

/-- Function that maps a positive integer to the sum of its digits in base 10 -/
def f (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that f(f(f(4444^4444))) = 7 -/
theorem f_triple_4444_power : f (f (f (4444^4444))) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_triple_4444_power_l249_24968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_exists_l249_24902

theorem divisibility_property_exists (n : ℕ) (h : n ≥ 2) :
  ∃ S : Finset ℤ, Finset.card S = n ∧
    ∀ a b : ℤ, a ∈ S → b ∈ S → a ≠ b → (a - b)^2 ∣ (a * b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_exists_l249_24902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_circle_l249_24974

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the vertices and foci
def A₁ : ℝ × ℝ := (-2, 0)
def A₂ : ℝ × ℝ := (2, 0)
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the line l₁
def l₁ (x : ℝ) : Prop := x = -2

-- Define the circle (renamed to avoid conflict)
def target_circle (x y : ℝ) : Prop := x^2 + y^2 + x - 2 = 0

-- Theorem statement
theorem ellipse_intersection_circle :
  ∀ (P : ℝ × ℝ) (M N : ℝ × ℝ),
    ellipse P.1 P.2 →
    l₁ M.1 →
    (∃ t : ℝ, M = (1 - t) • A₂ + t • P) →
    (∃ s : ℝ, N = (1 - s) • A₁ + s • P) →
    (∃ u : ℝ, N = (1 - u) • M + u • F₂) →
    target_circle N.1 N.2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_circle_l249_24974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_distance_sum_bound_l249_24982

/-- An acute angled triangle -/
structure AcuteTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_acute : Prop

/-- The orthocenter of a triangle -/
noncomputable def orthocenter (t : AcuteTriangle) : ℝ × ℝ := sorry

/-- The largest altitude of a triangle -/
noncomputable def largest_altitude (t : AcuteTriangle) : ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem orthocenter_distance_sum_bound (t : AcuteTriangle) :
  let H := orthocenter t
  let h_max := largest_altitude t
  distance t.A H + distance t.B H + distance t.C H ≤ 2 * h_max := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_distance_sum_bound_l249_24982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l249_24994

noncomputable def C (x : ℝ) : ℝ := (Real.log x) / x

theorem tangent_line_at_one : 
  let P : ℝ × ℝ := (1, 0)
  let f (x : ℝ) := x - 1
  ∀ x, HasDerivAt C (f x - f P.1) P.1 → 
    (∀ t, t ≠ P.1 → (C t - C P.1) / (t - P.1) = (f t - f P.1) / (t - P.1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l249_24994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_on_hyperbola_l249_24988

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

-- Define a point on the hyperbola
def point_on_hyperbola (p : ℝ × ℝ) : Prop :=
  hyperbola p.1 p.2

-- Define the midpoint of two points
def midpoint_of (p1 p2 m : ℝ × ℝ) : Prop :=
  m.1 = (p1.1 + p2.1) / 2 ∧ m.2 = (p1.2 + p2.2) / 2

-- Theorem statement
theorem midpoint_on_hyperbola :
  ∃ (A B : ℝ × ℝ), point_on_hyperbola A ∧ point_on_hyperbola B ∧
  midpoint_of A B (-1, -4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_on_hyperbola_l249_24988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l249_24931

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of the eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Definition of the line l -/
def line_l (x y : ℝ) (m : ℝ) : Prop :=
  x - m * y - 1 = 0

/-- Definition of the right focus F -/
noncomputable def right_focus (a b : ℝ) : ℝ × ℝ :=
  (Real.sqrt (a^2 - b^2), 0)

/-- Definition of the fixed line l₂ -/
def line_l2 (x : ℝ) : Prop :=
  x = 4

/-- Main theorem -/
theorem ellipse_and_line_properties
  (a b : ℝ)
  (h_ellipse : ∀ x y, ellipse_C x y a b → x^2 / 4 + y^2 / 3 = 1)
  (h_eccentricity : eccentricity a b = 1/2)
  (h_line_passes_focus : ∃ m, line_l ((right_focus a b).1) ((right_focus a b).2) m)
  (h_line_intersects : ∃ (A B : ℝ × ℝ) (m : ℝ), ellipse_C A.1 A.2 a b ∧ ellipse_C B.1 B.2 a b ∧
    line_l A.1 A.2 m ∧ line_l B.1 B.2 m)
  : ∃ D : ℝ × ℝ, D = (5/2, 0) ∧
    ∀ (A B : ℝ × ℝ) (m : ℝ),
      ellipse_C A.1 A.2 a b → ellipse_C B.1 B.2 a b →
      line_l A.1 A.2 m → line_l B.1 B.2 m →
      ∃ P : ℝ × ℝ, line_l2 P.1 ∧ P.2 = A.2 ∧
        (B.2 - D.2) / (B.1 - D.1) = (P.2 - D.2) / (P.1 - D.1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l249_24931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_odd_f_not_odd_l249_24983

-- Define a real-valued function
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + 1)

-- State that the sine function is odd
theorem sine_odd (x : ℝ) : Real.sin (-x) = -Real.sin x := by sorry

-- Theorem: f is not an odd function
theorem f_not_odd : ¬(∀ x : ℝ, f (-x) = -f x) := by
  intro h
  have : f (-0) ≠ -f 0 := by
    calc
      f (-0) = Real.sin (2 * (-0) + 1) := rfl
      _ = Real.sin 1 := by simp
      _ ≠ -Real.sin 1 := by sorry
      _ = -(Real.sin (2 * 0 + 1)) := by simp
      _ = -f 0 := rfl
  exact this (h 0)


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_odd_f_not_odd_l249_24983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_implies_m_range_l249_24962

/-- The function f(x) defined as the square root of a quadratic expression in x -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (m * x^2 + 4 * m * x + m + 3)

/-- Theorem stating that for f to have a domain of all real numbers, m must be in [0, 1] -/
theorem f_domain_implies_m_range (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f m x = y) ↔ (0 ≤ m ∧ m ≤ 1) := by
  sorry

#check f_domain_implies_m_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_implies_m_range_l249_24962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_characterization_l249_24919

/-- The number of intersection points between a line and an ellipse -/
inductive IntersectionCount
  | Two
  | One
  | Zero

/-- Given a line y = 2x + m and an ellipse x²/4 + y²/2 = 1,
    determine the number of intersection points based on the value of m -/
noncomputable def intersectionCount (m : ℝ) : IntersectionCount :=
  if -3 * Real.sqrt 2 < m ∧ m < 3 * Real.sqrt 2 then IntersectionCount.Two
  else if m = 3 * Real.sqrt 2 ∨ m = -3 * Real.sqrt 2 then IntersectionCount.One
  else IntersectionCount.Zero

theorem intersection_characterization (m : ℝ) :
  (intersectionCount m = IntersectionCount.Two ↔ -3 * Real.sqrt 2 < m ∧ m < 3 * Real.sqrt 2) ∧
  (intersectionCount m = IntersectionCount.One ↔ m = 3 * Real.sqrt 2 ∨ m = -3 * Real.sqrt 2) ∧
  (intersectionCount m = IntersectionCount.Zero ↔ m < -3 * Real.sqrt 2 ∨ m > 3 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_characterization_l249_24919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_inverse_squares_l249_24927

/-- A plane intersecting the coordinate axes -/
structure IntersectingPlane where
  α : ℝ
  β : ℝ
  γ : ℝ
  α_pos : α > 0
  β_pos : β > 0
  γ_pos : γ > 0
  sum_intercepts : α + β + γ = 10
  origin_distance : 1 / α^2 + 1 / β^2 + 1 / γ^2 = 1/4

/-- The centroid of the triangle formed by the intersections -/
noncomputable def centroid (p : IntersectingPlane) : ℝ × ℝ × ℝ :=
  (p.α / 3, p.β / 3, p.γ / 3)

/-- The theorem to be proved -/
theorem centroid_sum_inverse_squares (p : IntersectingPlane) :
  let (x, y, z) := centroid p
  1 / x^2 + 1 / y^2 + 1 / z^2 = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_inverse_squares_l249_24927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_circles_theorem_l249_24912

/-- Given two points A and B in a 2D Euclidean space, a radius r, and a tangent length l,
    this function determines the number of circles that satisfy the given conditions. -/
noncomputable def number_of_circles (A B : EuclideanSpace ℝ (Fin 2)) (r l : ℝ) : ℕ :=
  let AB := dist A B
  let condition := r + Real.sqrt (l^2 + r^2)
  if AB < condition then 2
  else if AB = condition then 1
  else 0

/-- Theorem stating that the number of circles satisfying the given conditions
    is determined by the relationship between AB and r + √(l^2 + r^2). -/
theorem number_of_circles_theorem (A B : EuclideanSpace ℝ (Fin 2)) (r l : ℝ) :
  (number_of_circles A B r l = 2 ∨ number_of_circles A B r l = 1 ∨ number_of_circles A B r l = 0) ∧
  (number_of_circles A B r l = 2 ↔ dist A B < r + Real.sqrt (l^2 + r^2)) ∧
  (number_of_circles A B r l = 1 ↔ dist A B = r + Real.sqrt (l^2 + r^2)) ∧
  (number_of_circles A B r l = 0 ↔ dist A B > r + Real.sqrt (l^2 + r^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_circles_theorem_l249_24912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_area_l249_24976

noncomputable section

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Theorem: The area of the kite with vertices (0,6), (4,8), (8,6), and (4,1) is 28 square inches -/
theorem kite_area : 
  let p1 : Point := ⟨0, 6⟩
  let p2 : Point := ⟨4, 8⟩
  let p3 : Point := ⟨8, 6⟩
  let p4 : Point := ⟨4, 1⟩
  triangleArea p1 p2 p3 + triangleArea p1 p3 p4 = 28 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_area_l249_24976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_satisfies_conditions_l249_24969

noncomputable def A : ℝ × ℝ := (0, 9)
noncomputable def B : ℝ × ℝ := (6, 3)
noncomputable def C : ℝ × ℝ := (-1/3, -1)
noncomputable def D : ℝ × ℝ := (3, -7/2)

noncomputable def P₁ : ℝ × ℝ := (1, 4)
noncomputable def P₂ : ℝ × ℝ := (-6, -3)

noncomputable def distance (p q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem point_P_satisfies_conditions : 
  (distance P₁ A = distance P₁ B ∧ 
   distance P₁ C / distance P₁ D = 2 / 3) ∧
  (distance P₂ A = distance P₂ B ∧ 
   distance P₂ C / distance P₂ D = 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_satisfies_conditions_l249_24969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_linear_functions_l249_24947

theorem intersection_point_of_linear_functions (a b : ℝ) (h : b > a) :
  ∃ x y : ℝ, b * x + a = a * x + b ∧ x = 1 ∧ y = a + b :=
by
  -- We'll use x = 1 and y = a + b
  use 1, a + b
  constructor
  · -- Prove that the functions intersect at x = 1
    simp
    ring
  constructor
  · -- Prove that x = 1
    rfl
  · -- Prove that y = a + b
    rfl

#check intersection_point_of_linear_functions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_linear_functions_l249_24947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unwatched_planet_l249_24915

/-- Represents a planet in the system -/
structure Planet where
  id : Nat

/-- The set of all planets in the system -/
def P : Finset Planet := sorry

/-- The function representing which planet each astronomer watches -/
def f : Planet → Planet := sorry

/-- The number of planets is odd -/
axiom h_odd : Odd P.card

/-- Each astronomer watches a different planet (injective) -/
axiom h_injective : Function.Injective f

/-- No planet can watch itself -/
axiom h_no_fixed_point : ∀ p : Planet, f p ≠ p

/-- Theorem: There exists a planet that is not being watched -/
theorem exists_unwatched_planet : ¬Function.Surjective f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unwatched_planet_l249_24915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_means_properties_l249_24938

/-- Arithmetic mean of two real numbers -/
noncomputable def arithmetic_mean (a b : ℝ) : ℝ := (a + b) / 2

/-- Geometric mean of two non-negative real numbers -/
noncomputable def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)

/-- Harmonic mean of two positive real numbers -/
noncomputable def harmonic_mean (a b : ℝ) : ℝ := 2 * a * b / (a + b)

theorem means_properties (a b : ℝ) (h_ab : a ≤ b) (h_pos : 0 < a) :
  let m := arithmetic_mean a b
  let g := geometric_mean a b
  let h := harmonic_mean a b
  (a ≤ m ∧ m ≤ b) ∧ (a ≤ g ∧ g ≤ b) ∧ (a ≤ h ∧ h ≤ b) ∧
  h ≤ g ∧ g ≤ m ∧
  (m = g ∧ g = h ↔ a = b) ∧
  (m = a ∨ m = b ∨ g = a ∨ g = b ∨ h = a ∨ h = b ↔ a = b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_means_properties_l249_24938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_islander_counts_l249_24910

/-- Represents the type of islanders: knights always tell the truth, liars always lie. -/
inductive IslanderType
| Knight
| Liar

/-- Represents an islander with their claim about the percentage of knights. -/
structure Islander where
  type : IslanderType
  claim : Nat

/-- Checks if the given number of knights matches the claim of the islander. -/
def claimIsTrue (i : Islander) (totalCount : Nat) (knightCount : Nat) : Prop :=
  knightCount = i.claim * totalCount / 100

/-- Checks if the islander's claim is consistent with their type. -/
def isConsistent (i : Islander) (totalCount : Nat) (knightCount : Nat) : Prop :=
  match i.type with
  | IslanderType.Knight => claimIsTrue i totalCount knightCount
  | IslanderType.Liar => ¬(claimIsTrue i totalCount knightCount)

/-- The main theorem stating the possible numbers of islanders in the room. -/
theorem possible_islander_counts :
  ∀ (n : Nat),
    (∃ (islanders : List Islander),
      islanders.length = n ∧
      (∀ i : Nat, i < n → ∃ (islander : Islander), islander ∈ islanders ∧ islander.claim = i + 1) ∧
      (∃ k : Nat, k > 0 ∧ k ≤ n ∧ 
        (∀ i : Islander, i ∈ islanders → isConsistent i n k))) →
    n = 10 ∨ n = 20 ∨ n = 25 ∨ n = 50 ∨ n = 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_islander_counts_l249_24910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_pi_l249_24911

noncomputable def f (x : ℝ) : ℝ := Real.tan x + (1 / Real.tan x)

theorem f_period_pi : ∀ x : ℝ, f (x + π) = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_pi_l249_24911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l249_24970

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sin x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∀ k : ℤ, ∃ x : ℝ, x = k * Real.pi / 2 + 3 * Real.pi / 8 ∧ f x = f (-x)) ∧
  (∀ k : ℤ, ∀ x y : ℝ, k * Real.pi + 3 * Real.pi / 8 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + 7 * Real.pi / 8 → f x < f y) ∧
  (∀ k : ℤ, ∀ x y : ℝ, k * Real.pi - Real.pi / 8 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + 3 * Real.pi / 8 → f x > f y) ∧
  (∀ m : ℝ, (∀ x : ℝ, Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → f x - m < 2) ↔ m > (Real.sqrt 3 - 5) / 4) :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l249_24970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overtake_relation_l249_24977

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℚ
  deriving Repr

/-- Represents a circular track -/
structure Track where
  length : ℚ
  deriving Repr

/-- Time taken for one runner to overtake another -/
noncomputable def overtakeTime (r1 r2 : Runner) (t : Track) : ℚ :=
  t.length / (r1.speed - r2.speed)

/-- Theorem stating the relationship between overtaking times -/
theorem overtake_relation (r1 r2 r3 : Runner) (t : Track) 
  (h1 : overtakeTime r1 r2 t = 6)
  (h2 : overtakeTime r1 r3 t = 10) :
  overtakeTime r3 r2 t = 15 := by
  sorry

#check overtake_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overtake_relation_l249_24977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l249_24979

/-- A triangle with no equal sides, two known medians, and a known area -/
structure SpecialTriangle where
  /-- The triangle has no equal sides -/
  no_equal_sides : True
  /-- Length of the first median -/
  median1 : ℝ
  /-- Length of the second median -/
  median2 : ℝ
  /-- Area of the triangle -/
  area : ℝ
  /-- The first median is 4 inches -/
  h_median1 : median1 = 4
  /-- The second median is 8 inches -/
  h_median2 : median2 = 8
  /-- The area of the triangle is 4√30 square inches -/
  h_area : area = 4 * Real.sqrt 30

/-- The theorem stating the length of the third median -/
theorem third_median_length (t : SpecialTriangle) : 
  ∃ (x : ℝ), x = (3 * Real.sqrt 60) / 8 ∧ 
  x = t.median1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l249_24979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_has_no_diagonals_l249_24935

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  vertices : Finset (Fin 4)
  edges : Finset (Fin 4 × Fin 4)
  vertex_count : vertices.card = 4
  edge_count : edges.card = 6
  edge_connects_vertices : ∀ e ∈ edges, e.fst ∈ vertices ∧ e.snd ∈ vertices
  all_vertices_connected : ∀ v w, v ∈ vertices → w ∈ vertices → v ≠ w → (v, w) ∈ edges ∨ (w, v) ∈ edges

/-- The number of diagonals in a regular tetrahedron -/
def diagonal_count (t : RegularTetrahedron) : ℕ :=
  (t.vertices.card * (t.vertices.card - 1) / 2) - t.edges.card

/-- Theorem: The number of diagonals in a regular tetrahedron is 0 -/
theorem regular_tetrahedron_has_no_diagonals (t : RegularTetrahedron) :
  diagonal_count t = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_has_no_diagonals_l249_24935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l249_24987

def sequence_a : ℕ → ℚ
  | 0 => 5
  | (n + 1) => 1 + 1 / sequence_a n

theorem a_5_value : sequence_a 4 = 28 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l249_24987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_triangle_theorem_l249_24957

/-- Triangle on a circle with specific properties -/
structure ClockTriangle where
  -- Points on the circle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Additional points
  D : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  -- Conditions
  angle_A : A = (Real.cos (30 * π / 180), Real.sin (30 * π / 180))
  angle_B : B = (Real.cos (150 * π / 180), Real.sin (150 * π / 180))
  angle_C : C = (Real.cos (240 * π / 180), Real.sin (240 * π / 180))
  D_on_BC : ∃ t : ℝ, D = (1 - t) • B + t • C
  F_on_AB : ∃ s : ℝ, F = (1 - s) • A + s • B
  G_on_AC : ∃ r : ℝ, G = (1 - r) • A + r • C
  AD_perp_BC : (A.1 - D.1) * (B.2 - C.2) + (A.2 - D.2) * (C.1 - B.1) = 0
  CF_perp_AB : (C.1 - F.1) * (A.2 - B.2) + (C.2 - F.2) * (B.1 - A.1) = 0
  FG_perp_AC : (F.1 - G.1) * (A.2 - C.2) + (F.2 - G.2) * (C.1 - A.1) = 0

/-- The main theorem to prove -/
theorem clock_triangle_theorem (t : ClockTriangle) : 
  (t.C.1 - t.D.1)^2 + (t.C.2 - t.D.2)^2 = (t.A.1 - t.G.1)^2 + (t.A.2 - t.G.2)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_triangle_theorem_l249_24957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_darnel_jogging_distance_l249_24921

theorem darnel_jogging_distance : ∃ (jog_distance : ℝ),
  let sprint_distance : ℝ := 0.88
  let sprint_jog_difference : ℝ := 0.13
  jog_distance = sprint_distance - sprint_jog_difference :=
by
  -- Define the jog distance
  let jog_distance : ℝ := 0.75
  -- Prove that this jog distance satisfies the equation
  have h : jog_distance = 0.88 - 0.13 := by norm_num
  -- Show that jog_distance exists and satisfies the equation
  exact ⟨jog_distance, h⟩
  
#check darnel_jogging_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_darnel_jogging_distance_l249_24921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_percentage_approx_l249_24941

noncomputable def jacket_price : ℝ := 100
noncomputable def shirt_price : ℝ := 50
noncomputable def hat_price : ℝ := 20

noncomputable def jacket_discount : ℝ := 0.30
noncomputable def shirt_discount : ℝ := 0.50
noncomputable def hat_discount : ℝ := 0.25

noncomputable def total_original_cost : ℝ := jacket_price + shirt_price + hat_price

noncomputable def total_savings : ℝ := 
  jacket_price * jacket_discount + 
  shirt_price * shirt_discount + 
  hat_price * hat_discount

noncomputable def savings_percentage : ℝ := (total_savings / total_original_cost) * 100

theorem savings_percentage_approx :
  ∃ ε > 0, |savings_percentage - 35.29| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_percentage_approx_l249_24941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_strictly_increasing_l249_24914

noncomputable def a (n : ℕ) : ℝ := ((n + 1)^n * n^(2-n : ℤ)) / (7*n^2 + 1)

theorem a_strictly_increasing : ∀ n : ℕ, a (n + 1) > a n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_strictly_increasing_l249_24914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l249_24929

/-- The eccentricity of a hyperbola with equation x²/(m²-4) + y²/m² = 1, where m is an integer -/
theorem hyperbola_eccentricity (m : ℤ) : 
  let a : ℝ := |m|
  let b : ℝ := Real.sqrt (4 - m^2)
  let c : ℝ := Real.sqrt (a^2 + b^2)
  c / a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l249_24929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l249_24986

-- Define the function f(x) = x^(1/3)
noncomputable def f (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Theorem stating that f is both odd and increasing
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l249_24986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wifes_speed_calculation_l249_24993

/-- The speed of the wife when she meets her husband -/
noncomputable def wifes_speed (mans_speed : ℝ) (mans_travel_time : ℝ) (wifes_delay : ℝ) : ℝ :=
  (mans_speed * mans_travel_time) / (mans_travel_time - wifes_delay)

/-- Theorem stating the wife's speed given the conditions -/
theorem wifes_speed_calculation :
  let mans_speed : ℝ := 40
  let mans_travel_time : ℝ := 2
  let wifes_delay : ℝ := 0.5
  wifes_speed mans_speed mans_travel_time wifes_delay = 160 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wifes_speed_calculation_l249_24993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_tax_percentage_is_3_672_l249_24943

noncomputable def calculate_total_tax_percentage (total_amount : ℝ) 
  (clothing_percent : ℝ) (food_percent : ℝ) (other_percent : ℝ)
  (clothing_discount : ℝ) (food_discount : ℝ) (other_discount : ℝ)
  (clothing_tax : ℝ) (food_tax : ℝ) (other_tax : ℝ) : ℝ :=
  let clothing_amount := clothing_percent * total_amount
  let food_amount := food_percent * total_amount
  let other_amount := other_percent * total_amount
  
  let clothing_after_discount := clothing_amount * (1 - clothing_discount)
  let food_after_discount := food_amount * (1 - food_discount)
  let other_after_discount := other_amount * (1 - other_discount)
  
  let clothing_tax_amount := clothing_after_discount * clothing_tax
  let food_tax_amount := food_after_discount * food_tax
  let other_tax_amount := other_after_discount * other_tax
  
  let total_tax := clothing_tax_amount + food_tax_amount + other_tax_amount
  
  (total_tax / total_amount) * 100

theorem total_tax_percentage_is_3_672 :
  calculate_total_tax_percentage 100 0.4 0.3 0.3 0.1 0.05 0.07 0.04 0 0.08 = 3.672 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_tax_percentage_is_3_672_l249_24943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l249_24981

/-- The function we're analyzing -/
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (2 * x) + Real.cos x ^ 2

/-- Definition of a periodic function -/
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

/-- The smallest positive period of f is π -/
theorem smallest_positive_period_of_f :
  ∃ T : ℝ, T > 0 ∧ is_periodic f T ∧ ∀ T' : ℝ, T' > 0 → is_periodic f T' → T ≤ T' := by
  sorry

#check smallest_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l249_24981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_x_minus_y_equals_four_l249_24960

noncomputable def data_set (x y : ℝ) : Finset ℝ := {x, y, 10, 11, 9}

noncomputable def average (s : Finset ℝ) : ℝ := (s.sum id) / s.card

noncomputable def variance (s : Finset ℝ) (μ : ℝ) : ℝ :=
  (s.sum (fun x => (x - μ)^2)) / s.card

theorem abs_x_minus_y_equals_four (x y : ℝ) 
  (h_avg : average (data_set x y) = 10)
  (h_var : variance (data_set x y) 10 = 2) : 
  |x - y| = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_x_minus_y_equals_four_l249_24960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_l249_24901

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) 
  (h0 : quadratic_function a b c 0 = -1)
  (h1 : quadratic_function a b c 1 = 2) 
  (h2 : quadratic_function a b c 2 = 3)
  (h3 : quadratic_function a b c 3 = 2) :
  ∀ x₁ x₂ : ℝ, -1 < x₁ → x₁ < 0 → 3 < x₂ → x₂ < 4 →
  quadratic_function a b c x₁ < quadratic_function a b c x₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_l249_24901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_30_l249_24984

/-- Represents a special 20-faced die -/
structure SpecialDie :=
  (faces : Finset ℕ)
  (blank : Bool)
  (fair : Bool)

/-- Die 1: numbers 2 to 20 and a blank face -/
def die1 : SpecialDie :=
  { faces := Finset.image (fun x => x + 2) (Finset.range 19),
    blank := true,
    fair := true }

/-- Die 2: numbers 1 to 19 and a blank face -/
def die2 : SpecialDie :=
  { faces := Finset.image (fun x => x + 1) (Finset.range 19),
    blank := true,
    fair := true }

/-- The probability of rolling a sum of 30 with die1 and die2 -/
theorem probability_sum_30 :
  (let totalOutcomes := 20 * 20
   let favorableOutcomes := 9
   favorableOutcomes / totalOutcomes : ℚ) = 9 / 400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_30_l249_24984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_proof_l249_24918

noncomputable def initialHeight : ℝ := 120
noncomputable def reboundFactor : ℝ := 3/4
def bounceCount : ℕ := 5

noncomputable def bounceSequence (n : ℕ) : ℝ :=
  initialHeight * reboundFactor^n

noncomputable def totalDistance : ℝ :=
  2 * (initialHeight + (Finset.range (bounceCount - 1)).sum bounceSequence) - bounceSequence (bounceCount - 1)

theorem total_distance_proof :
  totalDistance = 612.1875 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_proof_l249_24918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_in_triangle_l249_24959

theorem shortest_side_in_triangle (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_inequality : a^2 + b^2 > 5*c^2) : c ≤ a ∧ c ≤ b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_in_triangle_l249_24959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_implies_a_zero_l249_24945

/-- A function f with a unique zero -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (abs x) + 2 * a - 1

/-- Theorem stating that if f has a unique zero, then a must be 0 -/
theorem unique_zero_implies_a_zero (a : ℝ) :
  (∃! x, f a x = 0) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_implies_a_zero_l249_24945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runway_trip_time_is_two_minutes_l249_24978

/-- Represents the fashion show setup -/
structure FashionShow where
  num_models : ℕ
  outfits_per_model : ℕ
  total_show_time : ℕ

/-- Calculates the time for one runway trip -/
def runway_trip_time (s : FashionShow) : ℚ :=
  s.total_show_time / (s.num_models * s.outfits_per_model)

/-- Theorem stating that the runway trip time is 2 minutes for the given conditions -/
theorem runway_trip_time_is_two_minutes (s : FashionShow) 
    (h1 : s.num_models = 6)
    (h2 : s.outfits_per_model = 5)
    (h3 : s.total_show_time = 60) : 
  runway_trip_time s = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runway_trip_time_is_two_minutes_l249_24978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_tetrahedron_volume_ratio_l249_24946

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  -- We don't need to specify the exact coordinates, just that it exists
  exists_tetrahedron : Unit

/-- An octahedron formed by the midpoints of a tetrahedron's edges -/
def midpoint_octahedron (t : RegularTetrahedron) : Type :=
  Unit

/-- The volume of a geometric shape -/
noncomputable def volume (α : Type) (shape : α) : ℝ :=
  sorry

/-- The theorem stating the volume ratio of the octahedron to the tetrahedron -/
theorem octahedron_tetrahedron_volume_ratio 
  (t : RegularTetrahedron) (o : midpoint_octahedron t) : 
  volume (midpoint_octahedron t) o / volume RegularTetrahedron t = 1 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_tetrahedron_volume_ratio_l249_24946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sum_example_l249_24963

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  ⌊x⌋

-- State the theorem
theorem floor_sum_example : floor (-3.73) + floor 1.4 = -3 := by
  -- Convert the real numbers to rationals for exact computation
  have h1 : floor (-3.73) = floor (-373/100) := by sorry
  have h2 : floor 1.4 = floor (14/10) := by sorry
  
  -- Compute the floor values
  have f1 : floor (-373/100) = -4 := by sorry
  have f2 : floor (14/10) = 1 := by sorry
  
  -- Rewrite using the computed values
  rw [h1, h2, f1, f2]
  
  -- Perform the addition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sum_example_l249_24963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nineteen_in_binary_l249_24948

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- The decimal number we want to convert -/
def decimal_number : ℕ := 19

/-- The expected binary representation -/
def expected_binary : List Bool := [true, true, false, false, true]

/-- Theorem stating that the binary representation of 19 is correct -/
theorem nineteen_in_binary :
  to_binary decimal_number = expected_binary := by
  -- The proof is omitted for now
  sorry

#eval to_binary decimal_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nineteen_in_binary_l249_24948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l249_24932

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 - x) + Real.log ((2 * x - 1) / (3 - x))

-- State the theorem
theorem domain_of_f : 
  {x : ℝ | f x ∈ Set.univ} = {x : ℝ | 1/2 < x ∧ x ≤ 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l249_24932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_properties_l249_24966

/-- A point is a vertex of a cube with edge length a -/
def is_vertex_of_cube (a : ℝ) (v : Fin 3 → ℝ) : Prop :=
  ∀ i, v i = 0 ∨ v i = a

/-- A set of points forms a regular tetrahedron -/
def is_regular_tetrahedron (s : Set (Fin 3 → ℝ)) : Prop :=
  ∃ (edge : ℝ), ∀ v w, v ∈ s → w ∈ s → v ≠ w → dist v w = edge

/-- The volume of a set of points in 3D space -/
noncomputable def volume (s : Set (Fin 3 → ℝ)) : ℝ := sorry

/-- A cube with edge length a -/
def cube (a : ℝ) : Set (Fin 3 → ℝ) :=
  {v | is_vertex_of_cube a v}

/-- Given a cube with edge length a, we can prove properties about tetrahedrons formed by its vertices -/
theorem cube_tetrahedron_properties (a : ℝ) (h : a > 0) :
  ∃ (tetra1 tetra2 : Set (Fin 3 → ℝ)),
    -- 1. There exist two distinct sets of four vertices forming regular tetrahedrons
    (∀ v, v ∈ tetra1 → is_vertex_of_cube a v) ∧
    (∀ v, v ∈ tetra2 → is_vertex_of_cube a v) ∧
    is_regular_tetrahedron tetra1 ∧
    is_regular_tetrahedron tetra2 ∧
    tetra1 ≠ tetra2 ∧
    -- 2. The volume ratio of tetrahedron to cube is 1/3
    (volume tetra1 / volume (cube a) = 1/3) ∧
    -- 3. The volume of intersection is (1/6)a³
    (volume (tetra1 ∩ tetra2) = 1/6 * a^3) ∧
    -- 4. The volume of union is (1/2)a³
    (volume (tetra1 ∪ tetra2) = 1/2 * a^3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_properties_l249_24966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_wins_l249_24961

/-- Represents a pile of stones -/
structure Pile where
  stones : Nat

/-- Represents the game state -/
structure GameState where
  piles : List Pile

/-- Represents a player -/
inductive Player where
  | Petya
  | Vasya

/-- Represents a move in the game -/
structure Move where
  player : Player
  from_piles : List Nat
  stones_taken : Nat

/-- Defines the initial game state -/
def initial_state : GameState :=
  { piles := List.replicate 11 { stones := 10 } }

/-- Checks if a move is valid for a given player and game state -/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  match move.player with
  | Player.Petya => 
    move.from_piles.length = 1 ∧ 
    move.stones_taken ∈ [1, 2, 3] ∧
    (∃ pile : Pile, pile ∈ state.piles ∧ pile.stones = move.stones_taken)
  | Player.Vasya => 
    move.from_piles.length = move.stones_taken ∧
    move.stones_taken ∈ [1, 2, 3] ∧
    (∀ i ∈ move.from_piles, ∃ pile : Pile, pile ∈ state.piles ∧ pile.stones ≥ 1)

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if a player has a winning strategy -/
def has_winning_strategy (player : Player) : Prop :=
  sorry

/-- The main theorem stating that Vasya has a winning strategy -/
theorem vasya_wins : has_winning_strategy Player.Vasya := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_wins_l249_24961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_relationship_l249_24934

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The volume of a cylinder -/
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- Approximate equality for real numbers -/
def approx_eq (x y : ℝ) (ε : ℝ := 0.0001) : Prop := abs (x - y) < ε

notation:50 a " ≈ " b => approx_eq a b

theorem cylinder_height_relationship (c1 c2 : Cylinder) 
  (h_vol : volume c1 = volume c2)
  (h_radius : c1.radius = 1.2 * c2.radius) :
  c1.height ≈ 0.6944 * c2.height :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_relationship_l249_24934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baguette_cost_is_1_50_l249_24999

/-- Represents the cost structure and purchases at the bakery --/
structure BakeryPurchase where
  white_bread_price : ℚ
  sourdough_bread_price : ℚ
  croissant_price : ℚ
  white_bread_quantity : ℕ
  sourdough_bread_quantity : ℕ
  croissant_quantity : ℕ
  total_weeks : ℕ
  total_spent : ℚ

/-- Calculates the cost of a baguette given the bakery purchase information --/
noncomputable def baguette_cost (purchase : BakeryPurchase) : ℚ :=
  let weekly_known_cost := purchase.white_bread_price * purchase.white_bread_quantity +
                           purchase.sourdough_bread_price * purchase.sourdough_bread_quantity +
                           purchase.croissant_price * purchase.croissant_quantity
  let total_known_cost := weekly_known_cost * purchase.total_weeks
  let baguette_total_cost := purchase.total_spent - total_known_cost
  baguette_total_cost / purchase.total_weeks

/-- Theorem stating that the baguette cost is $1.50 given the specified purchase information --/
theorem baguette_cost_is_1_50 (purchase : BakeryPurchase) 
    (h1 : purchase.white_bread_price = 7/2)
    (h2 : purchase.sourdough_bread_price = 9/2)
    (h3 : purchase.croissant_price = 2)
    (h4 : purchase.white_bread_quantity = 2)
    (h5 : purchase.sourdough_bread_quantity = 2)
    (h6 : purchase.croissant_quantity = 1)
    (h7 : purchase.total_weeks = 4)
    (h8 : purchase.total_spent = 78) :
  baguette_cost purchase = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baguette_cost_is_1_50_l249_24999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_l249_24985

theorem isosceles_triangle (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0)
  (h4 : ∀ n : ℕ, ∃ x y z : ℝ, 
    x = a^n ∧ y = b^n ∧ z = c^n ∧
    x + y > z ∧ y + z > x ∧ z + x > y) :
  a = b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_l249_24985
