import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l3767_376701

theorem inequality_proof (x y z : ℝ) 
  (non_neg_x : 0 ≤ x) (non_neg_y : 0 ≤ y) (non_neg_z : 0 ≤ z)
  (sum_one : x + y + z = 1) : 
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ 
  y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3767_376701


namespace NUMINAMATH_CALUDE_exact_fourth_power_implies_zero_coefficients_exact_square_implies_perfect_square_l3767_376778

-- Part (a)
theorem exact_fourth_power_implies_zero_coefficients 
  (a b c : ℤ) 
  (h : ∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^4) :
  a = 0 ∧ b = 0 := by sorry

-- Part (b)
theorem exact_square_implies_perfect_square 
  (a b c : ℤ) 
  (h : ∀ x : ℤ, ∃ z : ℤ, a * x^2 + b * x + c = z^2) :
  ∃ d e : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (d * x + e)^2 := by sorry

end NUMINAMATH_CALUDE_exact_fourth_power_implies_zero_coefficients_exact_square_implies_perfect_square_l3767_376778


namespace NUMINAMATH_CALUDE_intersection_implies_equality_l3767_376789

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b
def g (c d : ℝ) (x : ℝ) : ℝ := x^2 + c*x + d

-- State the theorem
theorem intersection_implies_equality (a b c d : ℝ) 
  (h1 : f a b 1 = 1) 
  (h2 : g c d 1 = 1) : 
  a^5 + d^6 = c^6 - b^5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_equality_l3767_376789


namespace NUMINAMATH_CALUDE_triangle_third_side_l3767_376750

theorem triangle_third_side (a b c : ℝ) (θ : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : θ = Real.pi / 3) :
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos θ → c = Real.sqrt 57 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_l3767_376750


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l3767_376795

theorem absolute_value_simplification : |-4^2 + 7| = 9 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l3767_376795


namespace NUMINAMATH_CALUDE_circle_area_through_isosceles_triangle_vertices_l3767_376753

/-- The area of a circle passing through the vertices of an isosceles triangle -/
theorem circle_area_through_isosceles_triangle_vertices (a b c : ℝ) :
  a = 4 →  -- Two sides of the triangle are 4 units long
  b = 4 →  -- Two sides of the triangle are 4 units long
  c = 3 →  -- The base of the triangle is 3 units long
  a = b →  -- The triangle is isosceles
  ∃ (r : ℝ), r > 0 ∧ π * r^2 = (256/55) * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_through_isosceles_triangle_vertices_l3767_376753


namespace NUMINAMATH_CALUDE_sampling_interval_is_nine_l3767_376769

/-- The sampling interval for systematic sampling in a book binding factory. -/
def sampling_interval (total_books : ℕ) (sample_size : ℕ) : ℕ :=
  (total_books - 2) / sample_size

/-- Theorem stating that the sampling interval is 9 under given conditions. -/
theorem sampling_interval_is_nine :
  let total_books := 362
  let sample_size := 40
  sampling_interval total_books sample_size = 9 := by
sorry

#eval sampling_interval 362 40

end NUMINAMATH_CALUDE_sampling_interval_is_nine_l3767_376769


namespace NUMINAMATH_CALUDE_counterexamples_count_l3767_376764

def sumOfDigits (n : ℕ) : ℕ := sorry

def hasNoZeroDigit (n : ℕ) : Prop := sorry

def isPrime (n : ℕ) : Prop := sorry

theorem counterexamples_count :
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, sumOfDigits n = 5 ∧ hasNoZeroDigit n ∧ ¬isPrime n) ∧
    (∀ n ∉ S, ¬(sumOfDigits n = 5 ∧ hasNoZeroDigit n ∧ ¬isPrime n)) ∧
    Finset.card S = 6 := by sorry

end NUMINAMATH_CALUDE_counterexamples_count_l3767_376764


namespace NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l3767_376715

/-- The area of a circle with diameter endpoints C(-2,3) and D(6,9) is 25π square units. -/
theorem circle_area_from_diameter_endpoints :
  let c : ℝ × ℝ := (-2, 3)
  let d : ℝ × ℝ := (6, 9)
  let diameter_squared := (d.1 - c.1)^2 + (d.2 - c.2)^2
  let radius := Real.sqrt diameter_squared / 2
  let area := π * radius^2
  area = 25 * π :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l3767_376715


namespace NUMINAMATH_CALUDE_dartboard_angles_l3767_376792

theorem dartboard_angles (p₁ p₂ : ℝ) (θ₁ θ₂ : ℝ) :
  p₁ = 1/8 →
  p₂ = 2 * p₁ →
  p₁ = θ₁ / 360 →
  p₂ = θ₂ / 360 →
  θ₁ = 45 ∧ θ₂ = 90 :=
by sorry

end NUMINAMATH_CALUDE_dartboard_angles_l3767_376792


namespace NUMINAMATH_CALUDE_factorization_left_to_right_l3767_376726

theorem factorization_left_to_right (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_left_to_right_l3767_376726


namespace NUMINAMATH_CALUDE_sqrt_ln_relation_l3767_376783

theorem sqrt_ln_relation (a b : ℝ) :
  (∀ a b, (Real.log a > Real.log b) → (Real.sqrt a > Real.sqrt b)) ∧
  (∃ a b, (Real.sqrt a > Real.sqrt b) ∧ ¬(Real.log a > Real.log b)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ln_relation_l3767_376783


namespace NUMINAMATH_CALUDE_maximal_arithmetic_progression_1996_maximal_arithmetic_progression_1997_l3767_376770

/-- The set of reciprocals of natural numbers -/
def S : Set ℚ := {q : ℚ | ∃ n : ℕ, q = 1 / n}

/-- An arithmetic progression in S -/
def is_arithmetic_progression (a : ℕ → ℚ) (n : ℕ) : Prop :=
  ∃ (first d : ℚ), ∀ i < n, a i = first + i • d ∧ a i ∈ S

/-- A maximal arithmetic progression in S -/
def is_maximal_arithmetic_progression (a : ℕ → ℚ) (n : ℕ) : Prop :=
  is_arithmetic_progression a n ∧
  ¬∃ (b : ℕ → ℚ) (m : ℕ), m > n ∧ is_arithmetic_progression b m ∧
    (∀ i < n, a i = b i)

theorem maximal_arithmetic_progression_1996 :
  ∃ (a : ℕ → ℚ), is_maximal_arithmetic_progression a 1996 :=
sorry

theorem maximal_arithmetic_progression_1997 :
  ∃ (a : ℕ → ℚ), is_maximal_arithmetic_progression a 1997 :=
sorry

end NUMINAMATH_CALUDE_maximal_arithmetic_progression_1996_maximal_arithmetic_progression_1997_l3767_376770


namespace NUMINAMATH_CALUDE_binomial_prob_X_eq_one_l3767_376773

/-- A random variable X following a binomial distribution B(n, p) with given expectation and variance -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  mean_eq : n * p = 5 / 2
  var_eq : n * p * (1 - p) = 5 / 4

/-- The probability of X = 1 for the given binomial random variable -/
def prob_X_eq_one (X : BinomialRV) : ℝ :=
  X.n.choose 1 * X.p^1 * (1 - X.p)^(X.n - 1)

/-- Theorem stating that P(X=1) = 5/32 for the given binomial random variable -/
theorem binomial_prob_X_eq_one (X : BinomialRV) : prob_X_eq_one X = 5 / 32 := by
  sorry

end NUMINAMATH_CALUDE_binomial_prob_X_eq_one_l3767_376773


namespace NUMINAMATH_CALUDE_triangle_inequalities_l3767_376784

/-- Triangle properties and inequalities -/
theorem triangle_inequalities (a b c S h_a h_b h_c r_a r_b r_c r : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ S > 0 ∧ h_a > 0 ∧ h_b > 0 ∧ h_c > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ r > 0)
  (h_area : S = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))))
  (h_altitude : h_a = 2 * S / a ∧ h_b = 2 * S / b ∧ h_c = 2 * S / c)
  (h_excircle : (r_a * r_b * r_c)^2 = S^4 / r^2) :
  (S^3 ≤ (Real.sqrt 3 / 4)^3 * (a * b * c)^2) ∧
  ((h_a * h_b * h_c)^(1/3) ≤ 3^(1/4) * Real.sqrt S) ∧
  (3^(1/4) * Real.sqrt S ≤ (r_a * r_b * r_c)^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l3767_376784


namespace NUMINAMATH_CALUDE_probability_red_ball_experiment_l3767_376787

/-- The probability of picking a red ball in an experiment -/
def probability_red_ball (total_experiments : ℕ) (red_picks : ℕ) : ℚ :=
  red_picks / total_experiments

/-- Theorem: Given 10 experiments where red balls were picked 4 times, 
    the probability of picking a red ball is 0.4 -/
theorem probability_red_ball_experiment : 
  probability_red_ball 10 4 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_ball_experiment_l3767_376787


namespace NUMINAMATH_CALUDE_coin_packing_theorem_l3767_376711

/-- A coin is represented by its center and radius -/
structure Coin where
  center : ℝ × ℝ
  radius : ℝ

/-- The configuration of 12 coins forming a regular 12-gon -/
def outer_ring : List Coin := sorry

/-- The configuration of 7 coins inside the outer ring -/
def inner_coins : List Coin := sorry

/-- Two coins are tangent if the distance between their centers equals the sum of their radii -/
def are_tangent (c1 c2 : Coin) : Prop := sorry

/-- All coins in a list are mutually tangent -/
def all_tangent (coins : List Coin) : Prop := sorry

/-- The centers of the outer coins form a regular 12-gon -/
def is_regular_12gon (coins : List Coin) : Prop := sorry

theorem coin_packing_theorem :
  is_regular_12gon outer_ring ∧
  all_tangent outer_ring ∧
  (∀ c ∈ inner_coins, ∀ o ∈ outer_ring, are_tangent c o ∨ c = o) ∧
  all_tangent inner_coins ∧
  (List.length outer_ring = 12) ∧
  (List.length inner_coins = 7) := by
  sorry

end NUMINAMATH_CALUDE_coin_packing_theorem_l3767_376711


namespace NUMINAMATH_CALUDE_central_position_theorem_l3767_376756

/-- Represents a row of stones -/
def StoneRow := List Bool

/-- An action changes the color of neighboring stones of a black stone -/
def action (row : StoneRow) (pos : Nat) : StoneRow :=
  sorry

/-- Checks if all stones in the row are black -/
def allBlack (row : StoneRow) : Prop :=
  sorry

/-- Checks if a given initial position can lead to all black stones -/
def canMakeAllBlack (initialPos : Nat) (totalStones : Nat) : Prop :=
  sorry

theorem central_position_theorem :
  ∀ initialPos : Nat,
    initialPos ≤ 2009 →
    canMakeAllBlack initialPos 2009 ↔ initialPos = 1005 :=
  sorry

end NUMINAMATH_CALUDE_central_position_theorem_l3767_376756


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l3767_376749

theorem quadratic_root_theorem (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - (a+b+c)*x + (ab+bc+ca)
  (f 2 = 0) → (∃ x, f x = 0 ∧ x ≠ 2) → (∃ x, f x = 0 ∧ x = a+b+c-2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l3767_376749


namespace NUMINAMATH_CALUDE_solution_implies_a_zero_l3767_376714

/-- Given a system of linear equations and an additional equation with parameter a,
    prove that a must be zero if the solution of the system satisfies the additional equation. -/
theorem solution_implies_a_zero (x y a : ℝ) : 
  2 * x + 7 * y = 11 →
  5 * x - 4 * y = 6 →
  3 * x - 6 * y + 2 * a = 0 →
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_zero_l3767_376714


namespace NUMINAMATH_CALUDE_double_earnings_days_theorem_l3767_376786

/-- Calculate the number of additional days needed to double earnings -/
def daysToDoubleEarnings (daysSoFar : ℕ) (earningsSoFar : ℚ) : ℕ :=
  daysSoFar

/-- Theorem: The number of additional days needed to double earnings
    is equal to the number of days already worked -/
theorem double_earnings_days_theorem (daysSoFar : ℕ) (earningsSoFar : ℚ) 
    (hDays : daysSoFar > 0) (hEarnings : earningsSoFar > 0) :
  daysToDoubleEarnings daysSoFar earningsSoFar = daysSoFar := by
  sorry

#eval daysToDoubleEarnings 10 250  -- Should output 10

end NUMINAMATH_CALUDE_double_earnings_days_theorem_l3767_376786


namespace NUMINAMATH_CALUDE_complex_number_problem_l3767_376713

variable (z : ℂ)

theorem complex_number_problem (h1 : ∃ (r : ℝ), z + 2*I = r) 
  (h2 : ∃ (s : ℝ), z / (2 - I) = s) : 
  z = 4 - 2*I ∧ Complex.abs (z / (1 + I)) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3767_376713


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3767_376728

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b) / (a * b + a^5 + b^5) + (b * c) / (b * c + b^5 + c^5) + (c * a) / (c * a + c^5 + a^5) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3767_376728


namespace NUMINAMATH_CALUDE_x_squared_y_squared_value_l3767_376736

theorem x_squared_y_squared_value (x y : ℝ) 
  (h1 : x + y = 25)
  (h2 : x^2 + y^2 = 169)
  (h3 : x^3*y^3 + y^3*x^3 = 243) :
  x^2 * y^2 = 51984 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_y_squared_value_l3767_376736


namespace NUMINAMATH_CALUDE_correct_selling_prices_l3767_376776

-- Define the types of items
inductive Item
| Pencil
| Eraser
| Sharpener

-- Define the cost price function in A-coins
def costPriceA (item : Item) : ℝ :=
  match item with
  | Item.Pencil => 15
  | Item.Eraser => 25
  | Item.Sharpener => 35

-- Define the exchange rate
def exchangeRate : ℝ := 2

-- Define the profit percentage function
def profitPercentage (item : Item) : ℝ :=
  match item with
  | Item.Pencil => 0.20
  | Item.Eraser => 0.25
  | Item.Sharpener => 0.30

-- Define the selling price function in B-coins
def sellingPriceB (item : Item) : ℝ :=
  let costB := costPriceA item * exchangeRate
  costB + (costB * profitPercentage item)

-- Theorem to prove the selling prices are correct
theorem correct_selling_prices :
  sellingPriceB Item.Pencil = 36 ∧
  sellingPriceB Item.Eraser = 62.5 ∧
  sellingPriceB Item.Sharpener = 91 := by
  sorry

end NUMINAMATH_CALUDE_correct_selling_prices_l3767_376776


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3767_376742

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x + 5 > 3 ∧ x > a) ↔ x > -2) → a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3767_376742


namespace NUMINAMATH_CALUDE_intersection_area_bound_l3767_376774

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the area of a triangle
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

-- Define a function to reflect a triangle about a point
def reflectTriangle (t : Triangle) (p : ℝ × ℝ) : Triangle := sorry

-- Define a function to calculate the area of the intersection polygon
noncomputable def intersectionArea (t1 t2 : Triangle) : ℝ := sorry

-- Theorem statement
theorem intersection_area_bound (ABC : Triangle) (P : ℝ × ℝ) :
  intersectionArea ABC (reflectTriangle ABC P) ≤ (2/3) * triangleArea ABC := by
  sorry

end NUMINAMATH_CALUDE_intersection_area_bound_l3767_376774


namespace NUMINAMATH_CALUDE_quadruple_primes_l3767_376716

theorem quadruple_primes (p q r : ℕ) (n : ℕ+) : 
  (Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p^2 = q^2 + r^(n : ℕ)) ↔ 
  ((p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨ (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4)) :=
sorry

end NUMINAMATH_CALUDE_quadruple_primes_l3767_376716


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3767_376732

/-- 
For a quadratic equation x^2 + kx + 1 = 0 to have two equal real roots,
k must equal ±2.
-/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + 1 = 0 ∧ 
   ∀ y : ℝ, y^2 + k*y + 1 = 0 → y = x) ↔ 
  k = 2 ∨ k = -2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3767_376732


namespace NUMINAMATH_CALUDE_ellipse_axis_lengths_l3767_376724

/-- Given an ellipse with equation x²/16 + y²/25 = 1, prove that its major axis length is 10 and its minor axis length is 8 -/
theorem ellipse_axis_lengths :
  let ellipse := {(x, y) : ℝ × ℝ | x^2/16 + y^2/25 = 1}
  ∃ (major_axis minor_axis : ℝ),
    major_axis = 10 ∧
    minor_axis = 8 ∧
    (∀ (p : ℝ × ℝ), p ∈ ellipse →
      (p.1^2 + p.2^2 ≤ (major_axis/2)^2 ∧
       p.1^2 + p.2^2 ≥ (minor_axis/2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_lengths_l3767_376724


namespace NUMINAMATH_CALUDE_coefficient_of_x4_l3767_376779

theorem coefficient_of_x4 (x : ℝ) : 
  let expression := 2*(x^2 - x^4 + 2*x^3) + 4*(x^4 - x^3 + x^2 + 2*x^5 - x^6) + 3*(2*x^3 + x^4 - 4*x^2)
  ∃ (a b c d e f : ℝ), expression = a*x^6 + b*x^5 + 5*x^4 + c*x^3 + d*x^2 + e*x + f :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x4_l3767_376779


namespace NUMINAMATH_CALUDE_inequality_preservation_l3767_376704

theorem inequality_preservation (a b c : ℝ) (h : a > b) : a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3767_376704


namespace NUMINAMATH_CALUDE_sum_of_primes_divisible_by_12_l3767_376785

theorem sum_of_primes_divisible_by_12 (p q : ℕ) : 
  Prime p → Prime q → p - q = 2 → q > 3 → ∃ k : ℕ, p + q = 12 * k := by
sorry

end NUMINAMATH_CALUDE_sum_of_primes_divisible_by_12_l3767_376785


namespace NUMINAMATH_CALUDE_even_function_four_zeroes_range_l3767_376780

/-- An even function is a function that is symmetric about the y-axis -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- A function has four distinct zeroes if there exist four different real numbers that make the function equal to zero -/
def HasFourDistinctZeroes (f : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0

theorem even_function_four_zeroes_range (f : ℝ → ℝ) (h_even : EvenFunction f) :
  (∃ m : ℝ, HasFourDistinctZeroes (fun x => f x - m)) →
  (∀ m : ℝ, m ≠ 0 → ∃ x : ℝ, f x = m) ∧ (¬∃ x : ℝ, f x = 0) :=
sorry

end NUMINAMATH_CALUDE_even_function_four_zeroes_range_l3767_376780


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l3767_376719

theorem absolute_value_simplification : |(-4^2 + 6)| = 10 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l3767_376719


namespace NUMINAMATH_CALUDE_sixteen_tourists_remain_l3767_376797

/-- Calculates the number of tourists remaining after a dangerous rainforest tour --/
def tourists_remaining (initial : ℕ) : ℕ :=
  let after_anaconda := initial - 3
  let poisoned := (2 * after_anaconda) / 3
  let recovered := (2 * poisoned) / 9
  let after_poison := after_anaconda - poisoned + recovered
  let snake_bitten := after_poison / 4
  let saved_from_snakes := (3 * snake_bitten) / 5
  after_poison - snake_bitten + saved_from_snakes

/-- Theorem stating that 16 tourists remain at the end of the tour --/
theorem sixteen_tourists_remain : tourists_remaining 42 = 16 := by
  sorry


end NUMINAMATH_CALUDE_sixteen_tourists_remain_l3767_376797


namespace NUMINAMATH_CALUDE_hotel_arrangement_count_l3767_376754

/-- Represents the number of ways to arrange people in rooms -/
def arrangement_count (n : ℕ) (r : ℕ) (m : ℕ) : ℕ := sorry

/-- The number of people -/
def total_people : ℕ := 5

/-- The number of rooms -/
def total_rooms : ℕ := 3

/-- The number of people who cannot be in the same room -/
def restricted_people : ℕ := 2

/-- Theorem stating the number of possible arrangements -/
theorem hotel_arrangement_count :
  arrangement_count total_people total_rooms restricted_people = 114 := by
  sorry

end NUMINAMATH_CALUDE_hotel_arrangement_count_l3767_376754


namespace NUMINAMATH_CALUDE_f_compose_three_equals_43_l3767_376717

-- Define the function f
def f (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2*n + 1

-- Theorem statement
theorem f_compose_three_equals_43 : f (f (f 3)) = 43 := by
  sorry

end NUMINAMATH_CALUDE_f_compose_three_equals_43_l3767_376717


namespace NUMINAMATH_CALUDE_inequality_solution_l3767_376772

def solution_set (m : ℝ) : Set ℝ :=
  if m = 0 ∨ m = -12 then
    {x | x ≠ m / 6}
  else if m < -12 ∨ m > 0 then
    {x | x < (m - Real.sqrt (m^2 + 12*m)) / 6 ∨ x > (m + Real.sqrt (m^2 + 12*m)) / 6}
  else
    Set.univ

theorem inequality_solution (m : ℝ) :
  {x : ℝ | 3 * x^2 - m * x - m > 0} = solution_set m :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3767_376772


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l3767_376746

theorem triangle_angle_inequality (A B C : Real) 
  (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) 
  (h4 : A + B + C = π) : A * Real.cos B + Real.sin A * Real.sin C > 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l3767_376746


namespace NUMINAMATH_CALUDE_min_sum_given_log_condition_l3767_376705

theorem min_sum_given_log_condition (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  Real.log m / Real.log 3 + Real.log n / Real.log 3 ≥ 4 → m + n ≥ 18 := by
  sorry


end NUMINAMATH_CALUDE_min_sum_given_log_condition_l3767_376705


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3767_376707

theorem system_of_equations_solution
  (a b c x y z : ℝ)
  (h1 : x - a * y + a^2 * z = a^3)
  (h2 : x - b * y + b^2 * z = b^3)
  (h3 : x - c * y + c^2 * z = c^3)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hca : c ≠ a) :
  x = a * b * c ∧ y = a * b + b * c + c * a ∧ z = a + b + c :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3767_376707


namespace NUMINAMATH_CALUDE_johns_distance_conversion_l3767_376712

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 8^3 + d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

/-- John's weekly hiking distance in base 8 is 3762 --/
def johns_distance_base8 : ℕ × ℕ × ℕ × ℕ := (3, 7, 6, 2)

theorem johns_distance_conversion :
  let (d₃, d₂, d₁, d₀) := johns_distance_base8
  base8_to_base10 d₃ d₂ d₁ d₀ = 2034 :=
by sorry

end NUMINAMATH_CALUDE_johns_distance_conversion_l3767_376712


namespace NUMINAMATH_CALUDE_age_difference_l3767_376755

theorem age_difference (albert_age mary_age betty_age : ℕ) : 
  albert_age = 2 * mary_age →
  albert_age = 4 * betty_age →
  betty_age = 4 →
  albert_age - mary_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3767_376755


namespace NUMINAMATH_CALUDE_running_time_ratio_l3767_376782

theorem running_time_ratio :
  ∀ (danny_time steve_time : ℝ),
    danny_time = 27 →
    steve_time / 2 = danny_time / 2 + 13.5 →
    danny_time / steve_time = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_running_time_ratio_l3767_376782


namespace NUMINAMATH_CALUDE_parabola_reflection_l3767_376735

/-- Reflects a point (x, y) over the point (1, 1) -/
def reflect (x y : ℝ) : ℝ × ℝ := (2 - x, 2 - y)

/-- The original parabola y = x^2 -/
def original_parabola (x y : ℝ) : Prop := y = x^2

/-- The reflected parabola y = -x^2 + 4x - 2 -/
def reflected_parabola (x y : ℝ) : Prop := y = -x^2 + 4*x - 2

theorem parabola_reflection :
  ∀ x y : ℝ, original_parabola x y ↔ reflected_parabola (reflect x y).1 (reflect x y).2 :=
sorry

end NUMINAMATH_CALUDE_parabola_reflection_l3767_376735


namespace NUMINAMATH_CALUDE_shaded_area_is_2100_l3767_376700

/-- The area of a square with side length 50, minus two right triangles with sides of length 20, is 2100 -/
theorem shaded_area_is_2100 : 
  let square_side : ℝ := 50
  let triangle_side : ℝ := 20
  let square_area := square_side * square_side
  let triangle_area := (1 / 2) * triangle_side * triangle_side
  square_area - 2 * triangle_area = 2100 := by sorry

end NUMINAMATH_CALUDE_shaded_area_is_2100_l3767_376700


namespace NUMINAMATH_CALUDE_triangle_area_is_3_2_l3767_376745

/-- The area of the triangle bounded by the y-axis and two lines -/
def triangle_area : ℝ :=
  let line1 : ℝ → ℝ → Prop := fun x y ↦ y - 2*x = 1
  let line2 : ℝ → ℝ → Prop := fun x y ↦ 2*y + x = 10
  let y_axis : ℝ → ℝ → Prop := fun x _ ↦ x = 0
  3.2

/-- The area of the triangle is 3.2 -/
theorem triangle_area_is_3_2 : triangle_area = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_3_2_l3767_376745


namespace NUMINAMATH_CALUDE_word_to_number_correct_l3767_376734

def word_to_number (s : String) : ℝ :=
  match s with
  | "fifty point zero zero one" => 50.001
  | "seventy-five point zero six" => 75.06
  | _ => 0  -- Default case for other inputs

theorem word_to_number_correct :
  (word_to_number "fifty point zero zero one" = 50.001) ∧
  (word_to_number "seventy-five point zero six" = 75.06) := by
  sorry

end NUMINAMATH_CALUDE_word_to_number_correct_l3767_376734


namespace NUMINAMATH_CALUDE_correct_propositions_l3767_376738

theorem correct_propositions (a b : ℝ) : 
  ((a > |b| → a^2 > b^2) ∧ (a > b → a^3 > b^3)) := by
  sorry

end NUMINAMATH_CALUDE_correct_propositions_l3767_376738


namespace NUMINAMATH_CALUDE_jack_morning_emails_indeterminate_l3767_376762

/-- Represents the number of emails received at different times of the day -/
structure EmailCount where
  morning : ℕ
  afternoon : ℕ
  evening : ℕ

/-- Defines the properties of Jack's email counts -/
def jack_email_properties (e : EmailCount) : Prop :=
  e.afternoon = 5 ∧ 
  e.evening = 8 ∧ 
  e.afternoon + e.evening = 13

/-- Theorem stating that Jack's morning email count cannot be uniquely determined -/
theorem jack_morning_emails_indeterminate :
  ∃ e1 e2 : EmailCount, 
    jack_email_properties e1 ∧ 
    jack_email_properties e2 ∧ 
    e1.morning ≠ e2.morning :=
sorry

end NUMINAMATH_CALUDE_jack_morning_emails_indeterminate_l3767_376762


namespace NUMINAMATH_CALUDE_smallest_sum_l3767_376727

theorem smallest_sum (A B C D : ℕ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (∃ d : ℤ, C - B = B - A ∧ B - A = d) →  -- A, B, C form an arithmetic sequence
  (∃ r : ℚ, C = B * r ∧ D = C * r) →  -- B, C, D form a geometric sequence
  C = (4 * B) / 3 →  -- C/B = 4/3
  (∀ A' B' C' D' : ℕ, 
    A' > 0 → B' > 0 → C' > 0 →
    (∃ d : ℤ, C' - B' = B' - A' ∧ B' - A' = d) →
    (∃ r : ℚ, C' = B' * r ∧ D' = C' * r) →
    C' = (4 * B') / 3 →
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 43 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_l3767_376727


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3767_376744

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (5 - I) / (1 - I)
  (z.im : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3767_376744


namespace NUMINAMATH_CALUDE_part_one_part_two_l3767_376740

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part 1
theorem part_one (m : ℝ) :
  (∀ x : ℝ, f m x < 0) → -4 < m ∧ m ≤ 0 := by sorry

-- Part 2
theorem part_two (m : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 3 → f m x > -m + x - 1) → m > 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3767_376740


namespace NUMINAMATH_CALUDE_second_rectangle_weight_l3767_376708

-- Define the properties of the rectangles
def length1 : ℝ := 4
def width1 : ℝ := 3
def weight1 : ℝ := 18
def length2 : ℝ := 6
def width2 : ℝ := 4

-- Theorem to prove
theorem second_rectangle_weight :
  ∀ (density : ℝ),
  density > 0 →
  let area1 := length1 * width1
  let area2 := length2 * width2
  let weight2 := (area2 / area1) * weight1
  weight2 = 36 := by
sorry

end NUMINAMATH_CALUDE_second_rectangle_weight_l3767_376708


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3767_376798

theorem square_area_from_diagonal (a b : ℝ) :
  let diagonal := Real.sqrt (a^2 + 4 * b^2)
  (diagonal^2 / 2) = (a^2 + 4 * b^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3767_376798


namespace NUMINAMATH_CALUDE_b_profit_l3767_376751

-- Define the basic variables
def total_profit : ℕ := 21000

-- Define the investment ratio
def investment_ratio : ℕ := 3

-- Define the time ratio
def time_ratio : ℕ := 2

-- Define the profit sharing ratio
def profit_sharing_ratio : ℕ := investment_ratio * time_ratio

-- Theorem to prove
theorem b_profit (a_investment b_investment : ℕ) (a_time b_time : ℕ) :
  a_investment = investment_ratio * b_investment →
  a_time = time_ratio * b_time →
  (profit_sharing_ratio * b_investment * b_time + b_investment * b_time) * 3000 = total_profit * b_investment * b_time :=
by sorry


end NUMINAMATH_CALUDE_b_profit_l3767_376751


namespace NUMINAMATH_CALUDE_lcm_48_90_l3767_376766

theorem lcm_48_90 : Nat.lcm 48 90 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_90_l3767_376766


namespace NUMINAMATH_CALUDE_x_cubed_coefficient_l3767_376725

/-- The coefficient of x³ in the expansion of (3x³ + 2x² + 4x + 5)(4x³ + 3x² + 5x + 6) is 32 -/
theorem x_cubed_coefficient (x : ℝ) : 
  (3*x^3 + 2*x^2 + 4*x + 5) * (4*x^3 + 3*x^2 + 5*x + 6) = 
  32*x^3 + (12*x^5 + 15*x^4 + 23*x^2 + 34*x + 30) := by
sorry

end NUMINAMATH_CALUDE_x_cubed_coefficient_l3767_376725


namespace NUMINAMATH_CALUDE_age_difference_l3767_376763

theorem age_difference (frank_age john_age : ℕ) : 
  (frank_age + 4 = 16) → 
  (john_age + 3 = 2 * (frank_age + 3)) → 
  (john_age - frank_age = 15) :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l3767_376763


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l3767_376709

theorem no_positive_integer_solutions :
  ¬ ∃ (x : ℕ), 15 < 3 - 2 * (x : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l3767_376709


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3767_376702

-- Define the eccentricities
variable (e₁ e₂ : ℝ)

-- Define the parameters of the hyperbola
variable (a b : ℝ)

-- Define the coordinates of the intersection point M
variable (x y : ℝ)

-- Define the coordinates of the foci
variable (c : ℝ)

-- Theorem statement
theorem hyperbola_eccentricity_range 
  (h1 : a > b) 
  (h2 : b > 0)
  (h3 : x^2 / a^2 - y^2 / b^2 = 1)  -- Hyperbola equation
  (h4 : x > 0 ∧ y > 0)  -- M is in the first quadrant
  (h5 : (x + c) * (x - c) + y^2 = 0)  -- F₁M · F₂M = 0
  (h6 : 3/4 ≤ e₁ ∧ e₁ ≤ 3*Real.sqrt 10/10)  -- Range of e₁
  (h7 : 1/e₁^2 + 1/e₂^2 = 1)  -- Relationship between e₁ and e₂
  : 3*Real.sqrt 2/4 ≤ e₂ ∧ e₂ < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3767_376702


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l3767_376720

theorem fahrenheit_to_celsius (F C : ℝ) : 
  F = 95 → F = (9/5) * C + 32 → C = 35 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l3767_376720


namespace NUMINAMATH_CALUDE_sin_two_x_value_l3767_376761

theorem sin_two_x_value (x : ℝ) (h : Real.sin (π / 4 - x) = 1 / 6) : 
  Real.sin (2 * x) = 17 / 18 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_x_value_l3767_376761


namespace NUMINAMATH_CALUDE_game_wheel_probability_l3767_376752

theorem game_wheel_probability (pX pY pZ pW : ℚ) : 
  pX = 1/4 → pY = 1/3 → pW = 1/6 → pX + pY + pZ + pW = 1 → pZ = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_game_wheel_probability_l3767_376752


namespace NUMINAMATH_CALUDE_sum_of_numeric_values_l3767_376777

/-- The numeric value assigned to a letter based on its position in the alphabet. -/
def letterValue (n : ℕ) : ℤ :=
  match n % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 1
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -1
  | 0 => 0
  | _ => 0  -- This case should never occur, but Lean requires it for completeness

/-- The positions of the letters in "numeric" in the alphabet. -/
def numericPositions : List ℕ := [14, 21, 13, 5, 18, 9, 3]

/-- The theorem stating that the sum of the numeric values of the letters in "numeric" is -1. -/
theorem sum_of_numeric_values :
  (numericPositions.map letterValue).sum = -1 := by
  sorry

#eval (numericPositions.map letterValue).sum

end NUMINAMATH_CALUDE_sum_of_numeric_values_l3767_376777


namespace NUMINAMATH_CALUDE_range_of_b_l3767_376722

theorem range_of_b (a : ℝ) (h1 : 0 < a) (h2 : a ≤ 5/4) :
  (∃ (b : ℝ), b > 0 ∧ 
    (∀ (x : ℝ), |x - a| < b → |x - a^2| < 1/2) ∧
    (∀ (c : ℝ), c > b → ∃ (y : ℝ), |y - a| < c ∧ |y - a^2| ≥ 1/2)) ∧
  (∀ (b : ℝ), (∀ (x : ℝ), |x - a| < b → |x - a^2| < 1/2) → b ≤ 3/16) :=
sorry

end NUMINAMATH_CALUDE_range_of_b_l3767_376722


namespace NUMINAMATH_CALUDE_minimal_distance_point_l3767_376799

/-- Given points A, B, and C in the xy-plane, prove that the value of m that minimizes 
    the sum of distances AC + CB is -7/5 when C is constrained to the y-axis. -/
theorem minimal_distance_point (A B C : ℝ × ℝ) : 
  A = (-2, -3) → 
  B = (3, 1) → 
  C.1 = 0 →
  (∀ m' : ℝ, dist A C + dist C B ≤ dist A (0, m') + dist (0, m') B) →
  C.2 = -7/5 := by
sorry

end NUMINAMATH_CALUDE_minimal_distance_point_l3767_376799


namespace NUMINAMATH_CALUDE_v_2008_equals_3703_l3767_376788

/-- Defines the sequence v_n as described in the problem -/
def v : ℕ → ℕ := sorry

/-- The 2008th term of the sequence v_n is 3703 -/
theorem v_2008_equals_3703 : v 2008 = 3703 := by sorry

end NUMINAMATH_CALUDE_v_2008_equals_3703_l3767_376788


namespace NUMINAMATH_CALUDE_inequality_proof_l3767_376794

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + 3*b^3) / (5*a + b) + (b^3 + 3*c^3) / (5*b + c) + (c^3 + 3*a^3) / (5*c + a) ≥ 2/3 * (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3767_376794


namespace NUMINAMATH_CALUDE_pirate_treasure_division_l3767_376775

theorem pirate_treasure_division (S a b c d e : ℚ) : 
  a = (S - a) / 2 →
  b = (S - b) / 3 →
  c = (S - c) / 4 →
  d = (S - d) / 5 →
  e = 90 →
  S = a + b + c + d + e →
  S = 1800 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_division_l3767_376775


namespace NUMINAMATH_CALUDE_fresh_fruit_water_percentage_l3767_376757

theorem fresh_fruit_water_percentage
  (dried_water_percentage : ℝ)
  (dried_weight : ℝ)
  (fresh_weight : ℝ)
  (h1 : dried_water_percentage = 0.15)
  (h2 : dried_weight = 12)
  (h3 : fresh_weight = 101.99999999999999) :
  (fresh_weight - dried_weight * (1 - dried_water_percentage)) / fresh_weight = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_fresh_fruit_water_percentage_l3767_376757


namespace NUMINAMATH_CALUDE_sine_cosine_transform_l3767_376737

theorem sine_cosine_transform (x : ℝ) : 
  Real.sqrt 3 * Real.sin (3 * x) + Real.cos (3 * x) = 2 * Real.sin (3 * x + π / 6) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_transform_l3767_376737


namespace NUMINAMATH_CALUDE_union_M_complement_N_l3767_376781

universe u

def U : Finset ℕ := {0, 1, 2, 3, 4, 5}
def M : Finset ℕ := {0, 3, 5}
def N : Finset ℕ := {1, 4, 5}

theorem union_M_complement_N : M ∪ (U \ N) = {0, 2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_union_M_complement_N_l3767_376781


namespace NUMINAMATH_CALUDE_kaleb_toys_l3767_376706

def number_of_toys (initial_savings allowance toy_cost : ℕ) : ℕ :=
  (initial_savings + allowance) / toy_cost

theorem kaleb_toys : number_of_toys 21 15 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_toys_l3767_376706


namespace NUMINAMATH_CALUDE_sum_of_variables_l3767_376771

/-- Given a system of equations, prove that 2x + 2y + 2z = 8 -/
theorem sum_of_variables (x y z : ℝ) 
  (eq1 : y + z = 20 - 4*x)
  (eq2 : x + z = -10 - 4*y)
  (eq3 : x + y = 14 - 4*z) :
  2*x + 2*y + 2*z = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_variables_l3767_376771


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l3767_376731

theorem cubic_equation_solutions :
  let f : ℂ → ℂ := λ x => x^3 + 4*x^2*Real.sqrt 3 + 12*x + 8*Real.sqrt 3 + x + Real.sqrt 3
  ∃ (z₁ z₂ z₃ : ℂ), 
    z₁ = -Real.sqrt 3 ∧ 
    z₂ = -Real.sqrt 3 + Complex.I ∧ 
    z₃ = -Real.sqrt 3 - Complex.I ∧
    (∀ z : ℂ, f z = 0 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l3767_376731


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3767_376765

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3}

-- Define set A
def A : Set Nat := {0, 1}

-- Define set B
def B : Set Nat := {1, 2, 3}

-- Theorem statement
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {2, 3} := by
  sorry

-- Note: Aᶜ represents the complement of A in the universal set U

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3767_376765


namespace NUMINAMATH_CALUDE_billion_product_without_zeros_l3767_376793

theorem billion_product_without_zeros :
  ∃ (a b : ℕ), 
    a * b = 1000000000 ∧ 
    (∀ d : ℕ, d > 0 → d ≤ 9 → (a / 10^d) % 10 ≠ 0) ∧
    (∀ d : ℕ, d > 0 → d ≤ 9 → (b / 10^d) % 10 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_billion_product_without_zeros_l3767_376793


namespace NUMINAMATH_CALUDE_outfit_combinations_l3767_376703

theorem outfit_combinations : 
  let blue_shirts : ℕ := 6
  let green_shirts : ℕ := 4
  let pants : ℕ := 7
  let blue_hats : ℕ := 9
  let green_hats : ℕ := 7
  let blue_shirt_green_hat := blue_shirts * pants * green_hats
  let green_shirt_blue_hat := green_shirts * pants * blue_hats
  blue_shirt_green_hat + green_shirt_blue_hat = 546 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l3767_376703


namespace NUMINAMATH_CALUDE_parallel_vectors_l3767_376796

def a : ℝ × ℝ := (-1, 1)
def b (m : ℝ) : ℝ × ℝ := (3, m)

theorem parallel_vectors (m : ℝ) : 
  (∃ (k : ℝ), a = k • (a + b m)) → m = -7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l3767_376796


namespace NUMINAMATH_CALUDE_fourth_root_sum_of_fourth_powers_l3767_376733

/-- Given segments a and b, there exists a segment x such that x^4 = a^4 + b^4 -/
theorem fourth_root_sum_of_fourth_powers (a b : ℝ) : ∃ x : ℝ, x^4 = a^4 + b^4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_sum_of_fourth_powers_l3767_376733


namespace NUMINAMATH_CALUDE_imo_2007_problem_5_l3767_376729

theorem imo_2007_problem_5 (k : ℕ+) :
  (∃ (n : ℕ+), (8 * k * n - 1) ∣ (4 * k^2 - 1)^2) ↔ Even k := by
  sorry

end NUMINAMATH_CALUDE_imo_2007_problem_5_l3767_376729


namespace NUMINAMATH_CALUDE_green_bows_count_l3767_376758

theorem green_bows_count (total : ℕ) (white : ℕ) : 
  (3 : ℚ) / 20 + 3 / 10 + 1 / 5 + 1 / 20 + (white : ℚ) / total = 1 →
  white = 24 →
  (1 : ℚ) / 5 * total = 16 :=
by sorry

end NUMINAMATH_CALUDE_green_bows_count_l3767_376758


namespace NUMINAMATH_CALUDE_problem_statement_l3767_376710

theorem problem_statement (m : ℤ) : 
  2^2000 - 3 * 2^1998 + 5 * 2^1996 - 2^1995 = m * 2^1995 → m = 17 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3767_376710


namespace NUMINAMATH_CALUDE_hiker_distance_theorem_l3767_376790

/-- Calculates the total distance walked by a hiker over three days given specific conditions -/
def total_distance_walked (day1_distance : ℕ) (day1_speed : ℕ) (day2_speed_increase : ℕ) (day3_speed : ℕ) (day3_hours : ℕ) : ℕ :=
  let day1_hours : ℕ := day1_distance / day1_speed
  let day2_hours : ℕ := day1_hours - 1
  let day2_speed : ℕ := day1_speed + day2_speed_increase
  let day2_distance : ℕ := day2_speed * day2_hours
  let day3_distance : ℕ := day3_speed * day3_hours
  day1_distance + day2_distance + day3_distance

/-- Theorem stating that the total distance walked is 53 miles given the specific conditions -/
theorem hiker_distance_theorem :
  total_distance_walked 18 3 1 5 3 = 53 := by
  sorry

end NUMINAMATH_CALUDE_hiker_distance_theorem_l3767_376790


namespace NUMINAMATH_CALUDE_range_of_x_l3767_376767

def p (x : ℝ) := 1 / (x - 2) < 0
def q (x : ℝ) := x^2 - 4*x - 5 < 0

theorem range_of_x (x : ℝ) :
  (p x ∨ q x) ∧ ¬(p x ∧ q x) →
  x ∈ Set.Iic (-1) ∪ Set.Ico 3 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l3767_376767


namespace NUMINAMATH_CALUDE_vincent_sticker_packs_l3767_376791

/-- The number of packs Vincent bought yesterday -/
def yesterday_packs : ℕ := sorry

/-- The number of packs Vincent bought today -/
def today_packs : ℕ := yesterday_packs + 10

/-- The total number of packs Vincent has -/
def total_packs : ℕ := 40

theorem vincent_sticker_packs : yesterday_packs = 15 := by
  sorry

end NUMINAMATH_CALUDE_vincent_sticker_packs_l3767_376791


namespace NUMINAMATH_CALUDE_common_number_in_list_l3767_376743

theorem common_number_in_list (list : List ℝ) : 
  list.length = 7 →
  (list.take 4).sum / 4 = 7 →
  (list.drop 3).sum / 4 = 9 →
  list.sum / 7 = 8 →
  ∃ x ∈ list.take 4 ∩ list.drop 3, x = 8 := by
sorry

end NUMINAMATH_CALUDE_common_number_in_list_l3767_376743


namespace NUMINAMATH_CALUDE_angle_D_measure_l3767_376723

-- Define the geometric figure
def geometric_figure (B C D E F : Real) : Prop :=
  -- Angle B measures 120°
  B = 120 ∧
  -- Angle B and C form a linear pair
  B + C = 180 ∧
  -- In triangle DEF, angle E = 45°
  E = 45 ∧
  -- Angle F is vertically opposite to angle C
  F = C ∧
  -- Triangle DEF sum of angles
  D + E + F = 180

-- Theorem statement
theorem angle_D_measure (B C D E F : Real) :
  geometric_figure B C D E F → D = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l3767_376723


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3767_376759

theorem necessary_but_not_sufficient (a b : ℝ) :
  (((a > 1) ∧ (b > 1)) → (a + b > 2)) ∧
  (∃ a b : ℝ, (a + b > 2) ∧ ¬((a > 1) ∧ (b > 1))) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3767_376759


namespace NUMINAMATH_CALUDE_john_rachel_toy_difference_l3767_376768

theorem john_rachel_toy_difference (jason_toys : ℕ) (rachel_toys : ℕ) :
  jason_toys = 21 →
  rachel_toys = 1 →
  ∃ (john_toys : ℕ),
    jason_toys = 3 * john_toys ∧
    john_toys > rachel_toys ∧
    john_toys - rachel_toys = 6 :=
by sorry

end NUMINAMATH_CALUDE_john_rachel_toy_difference_l3767_376768


namespace NUMINAMATH_CALUDE_fun_run_participation_l3767_376741

/-- Fun Run Participation Theorem -/
theorem fun_run_participation (signed_up_last_year : ℕ) (no_show_last_year : ℕ) : 
  signed_up_last_year = 200 →
  no_show_last_year = 40 →
  (signed_up_last_year - no_show_last_year) * 2 = 320 := by
  sorry

#check fun_run_participation

end NUMINAMATH_CALUDE_fun_run_participation_l3767_376741


namespace NUMINAMATH_CALUDE_range_of_x_l3767_376748

theorem range_of_x (x : ℝ) : (16 - x^2 ≥ 0) ↔ (-4 ≤ x ∧ x ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l3767_376748


namespace NUMINAMATH_CALUDE_soap_brands_survey_l3767_376747

theorem soap_brands_survey (total : ℕ) (neither : ℕ) (only_a : ℕ) (both : ℕ) : 
  total = 160 →
  neither = 80 →
  only_a = 60 →
  (3 * both) = (total - neither - only_a - both) →
  both = 5 := by
sorry

end NUMINAMATH_CALUDE_soap_brands_survey_l3767_376747


namespace NUMINAMATH_CALUDE_divisibility_criterion_l3767_376721

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem divisibility_criterion (n : ℕ) (h : n > 1) :
  (Nat.factorial (n - 1)) % n = 0 ↔ is_composite n ∧ n ≠ 4 :=
sorry

end NUMINAMATH_CALUDE_divisibility_criterion_l3767_376721


namespace NUMINAMATH_CALUDE_building_height_l3767_376760

theorem building_height :
  let standard_floor_height : ℝ := 3
  let taller_floor_height : ℝ := 3.5
  let num_standard_floors : ℕ := 18
  let num_taller_floors : ℕ := 2
  let total_floors : ℕ := num_standard_floors + num_taller_floors
  total_floors = 20 →
  (num_standard_floors : ℝ) * standard_floor_height + (num_taller_floors : ℝ) * taller_floor_height = 61 := by
  sorry

end NUMINAMATH_CALUDE_building_height_l3767_376760


namespace NUMINAMATH_CALUDE_smallest_next_divisor_l3767_376730

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_divisor (d m : ℕ) : Prop := ∃ k, m = d * k

theorem smallest_next_divisor 
  (m : ℕ) 
  (h1 : is_even m) 
  (h2 : is_four_digit m) 
  (h3 : is_divisor 437 m) :
  ∃ d : ℕ, 
    is_divisor d m ∧ 
    d > 437 ∧ 
    (∀ d' : ℕ, is_divisor d' m → d' > 437 → d ≤ d') ∧ 
    d = 475 :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_l3767_376730


namespace NUMINAMATH_CALUDE_book_cost_price_l3767_376718

theorem book_cost_price (selling_price : ℝ) (profit_percentage : ℝ) (cost_price : ℝ) : 
  selling_price = 260 ∧ profit_percentage = 20 → 
  selling_price = cost_price * (1 + profit_percentage / 100) →
  cost_price = 216.67 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l3767_376718


namespace NUMINAMATH_CALUDE_convex_polygon_partition_l3767_376739

/-- A convex polygon represented by its side lengths -/
structure ConvexPolygon where
  sides : List ℝ
  sides_positive : ∀ s ∈ sides, s > 0
  convexity : ∀ s ∈ sides, s ≤ (sides.sum / 2)

/-- The perimeter of a convex polygon -/
def perimeter (p : ConvexPolygon) : ℝ := p.sides.sum

/-- A partition of the sides of a polygon into two sets -/
structure Partition (p : ConvexPolygon) where
  set1 : List ℝ
  set2 : List ℝ
  partition_complete : set1 ∪ set2 = p.sides
  partition_disjoint : set1 ∩ set2 = ∅

theorem convex_polygon_partition (p : ConvexPolygon) :
  ∃ (part : Partition p), |part.set1.sum - part.set2.sum| ≤ (perimeter p) / 3 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_partition_l3767_376739
