import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_alphas_l556_55672

noncomputable def Q (x : ℂ) : ℂ := ((x^20 - 1) / (x - 1))^2 - x^20

def is_zero_of_Q (z : ℂ) : Prop := Q z = 0

def complex_form (z : ℂ) (r α : ℝ) : Prop :=
  z = r * (Complex.cos (2 * Real.pi * α) + Complex.I * Complex.sin (2 * Real.pi * α))

def valid_alpha (α : ℝ) : Prop := 0 < α ∧ α < 1

def valid_r (r : ℝ) : Prop := r > 0

theorem sum_of_first_five_alphas :
  ∃ (α₁ α₂ α₃ α₄ α₅ : ℝ) (r₁ r₂ r₃ r₄ r₅ : ℝ) (z₁ z₂ z₃ z₄ z₅ : ℂ),
    (∀ k, k ∈ [1, 2, 3, 4, 5] → valid_alpha (α₁ + (k - 1 : ℝ))) ∧
    (∀ k, k ∈ [1, 2, 3, 4, 5] → valid_r (r₁ + (k - 1 : ℝ))) ∧
    (∀ k, k ∈ [1, 2, 3, 4, 5] → is_zero_of_Q (z₁ + (k - 1 : ℝ))) ∧
    (∀ k, k ∈ [1, 2, 3, 4, 5] → complex_form (z₁ + (k - 1 : ℝ)) (r₁ + (k - 1 : ℝ)) (α₁ + (k - 1 : ℝ))) ∧
    α₁ < α₂ ∧ α₂ < α₃ ∧ α₃ < α₄ ∧ α₄ < α₅ ∧
    α₁ + α₂ + α₃ + α₄ + α₅ = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_alphas_l556_55672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l556_55669

theorem equation_solution : ∃! x : ℝ, (10 : ℝ)^x * (100 : ℝ)^(2*x) = (1000 : ℝ)^5 :=
  by
    use 3
    constructor
    · -- Prove that the equation holds for x = 3
      simp [Real.rpow_mul, Real.rpow_add]
      norm_num
    · -- Prove uniqueness
      intro y hy
      -- Proof of uniqueness goes here
      sorry
    done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l556_55669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_equals_30_l556_55614

/-- Given a natural number n and a real number a, 
    let S(n, a) be the sum (1+a)+(1+a)^2+(1+a)^3+...+(1+a)^n -/
def S (n : ℕ) (a : ℝ) : ℝ :=
  Finset.sum (Finset.range n) (fun i => (1 + a) ^ (i + 1))

/-- Given a natural number n and real numbers b₀, b₁, ..., bₙ,
    let P(n, a) be the polynomial b₀+b₁a+b₂a²+...+bₙaⁿ -/
def P (n : ℕ) (b : ℕ → ℝ) (a : ℝ) : ℝ :=
  Finset.sum (Finset.range (n + 1)) (fun i => b i * a ^ i)

theorem sum_of_coefficients_equals_30 (n : ℕ) :
  (∀ a : ℝ, S n a = P n (fun i => (Nat.choose (n + 1) (i + 1) : ℝ)) a) →
  (Finset.sum (Finset.range (n + 1)) (fun i => Nat.choose (n + 1) (i + 1)) = 30) →
  n = 4 := by
  sorry

#check sum_of_coefficients_equals_30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_equals_30_l556_55614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_star_result_l556_55673

noncomputable def star (a b : ℝ) : ℝ := (a - b) / (1 - a * b)

noncomputable def nested_star : ℕ → ℝ
  | 0 => 1001
  | n + 1 => star (n + 1) (nested_star n)

noncomputable def x : ℝ := nested_star 1000

theorem nested_star_result : star 0 x = -x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_star_result_l556_55673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_l556_55621

-- Define the function f
def f (a : ℝ) : ℝ → ℝ := λ x ↦ (2 * x + a) ^ 2

-- State the theorem
theorem a_value (a : ℝ) : deriv (f a) 2 = 20 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_l556_55621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_from_equilateral_triangle_l556_55633

/-- The area of a hexagon formed by constructing equilateral triangles on the sides of an equilateral triangle -/
theorem hexagon_area_from_equilateral_triangle (side_length : ℝ) (h : side_length = 2) : 
  let base_triangle_area := (Real.sqrt 3 / 4) * side_length ^ 2
  let outer_triangle_area := 3 * base_triangle_area
  let big_triangle_side := side_length + 2 * (Real.sqrt 3 / 2 * side_length)
  let big_triangle_area := (Real.sqrt 3 / 4) * big_triangle_side ^ 2
  big_triangle_area - outer_triangle_area = 4 * Real.sqrt 3 - 3 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_from_equilateral_triangle_l556_55633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_expr_l556_55678

-- Define the imaginary unit j
noncomputable def j : ℂ := Complex.I

-- Define the property of j
axiom j_squared : j^2 = -1

-- Define the expression
noncomputable def expr : ℂ := 3 * j - (1/3) * j⁻¹

-- State the theorem
theorem inverse_of_expr : expr⁻¹ = -(3/10) * j := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_expr_l556_55678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l556_55619

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def A : ℝ × ℝ := (-3, 0)

def B : ℝ × ℝ := (2, 5)

def P : ℝ × ℝ := (0, 2)

theorem equidistant_point : 
  distance A.1 A.2 P.1 P.2 = distance B.1 B.2 P.1 P.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l556_55619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_condition_l556_55626

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = 2^x + a * 2^(-x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (2 : ℝ)^x + a * (2 : ℝ)^(-x)

theorem even_function_condition (a : ℝ) :
  IsEven (f a) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_condition_l556_55626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l556_55662

/-- A circle C passes through points A(5,1) and B(1,3), and its center is on the x-axis. -/
theorem circle_equation (C : Set (ℝ × ℝ)) (A B : ℝ × ℝ) (h1 : A ∈ C) (h2 : B ∈ C)
    (h3 : A = (5, 1)) (h4 : B = (1, 3)) (h5 : ∃ x, (x, 0) ∈ C) :
  C = {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 10} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l556_55662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l556_55666

-- Define the equation
noncomputable def equation (a : ℝ) (x : ℝ) : Prop := x^2 + a*x + 2 = 0

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, ¬(equation a x)

-- Define proposition q
def q (a : ℝ) : Prop := StrictMono (f a)

-- Define the range of a
def range_a : Set ℝ := Set.Ioc (-Real.sqrt 2) 1 ∪ Set.Ici (Real.sqrt 2)

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) → a ∈ range_a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l556_55666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_same_flips_l556_55613

-- Define a fair coin flip
noncomputable def fairCoinFlip : ℝ := 1 / 2

-- Define the probability of getting the first head on the n-th flip
noncomputable def probFirstHeadOnNthFlip (n : ℕ) : ℝ := fairCoinFlip ^ n

-- Define the probability of all four players getting their first head on the n-th flip
noncomputable def probAllFirstHeadOnNthFlip (n : ℕ) : ℝ := (probFirstHeadOnNthFlip n) ^ 4

-- Define the sum of probabilities for all possible n
noncomputable def sumProbAllFirstHead : ℝ := ∑' n, probAllFirstHeadOnNthFlip n

-- Theorem: The probability of all four players flipping their coins the same number of times is 1/15
theorem prob_all_same_flips : sumProbAllFirstHead = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_same_flips_l556_55613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_minus_gamma_bounds_l556_55634

noncomputable def cos_beta_minus_gamma (k : ℝ) (β γ : ℝ) : ℝ :=
  1 + 3 / (2 * ((k - 1)^2 - 1))

theorem cos_beta_minus_gamma_bounds (k : ℝ) (α β γ : ℝ) 
  (h1 : 0 < k) (h2 : k < 2)
  (eq1 : Real.cos α + k * Real.cos β + (2 - k) * Real.cos γ = 0)
  (eq2 : Real.sin α + k * Real.sin β + (2 - k) * Real.sin γ = 0) :
  (∀ θ, cos_beta_minus_gamma k β γ ≤ θ → θ ≤ -1/2) ∧
  (cos_beta_minus_gamma k β γ ≥ -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_minus_gamma_bounds_l556_55634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_negative_27_l556_55684

-- Define the cube root function
def cubeRoot (x : ℝ) : ℝ := x^(1/3)

-- Theorem statement
theorem cube_root_of_negative_27 : cubeRoot (-27) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_negative_27_l556_55684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l556_55620

/-- For positive real numbers a and b, the maximum value of 2(a - x)(x^3 + √(x^6 + b^6)) is a^2 + b^6 -/
theorem max_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (M : ℝ), M = a^2 + b^6 ∧ ∀ (x : ℝ), 2*(a - x)*(x^3 + Real.sqrt (x^6 + b^6)) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l556_55620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_and_equations_imply_difference_l556_55652

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem product_and_equations_imply_difference (a b c d : ℕ+) : 
  (a * b * c * d = factorial 9) →
  (a * b - a - b = 194) →
  (b * c + b + c = 230) →
  (c * d - c - d = 272) →
  (a : ℤ) - d = -11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_and_equations_imply_difference_l556_55652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_formula_l556_55643

def a : ℕ → ℝ
  | 0 => 2^(5/2)  -- Added case for 0
  | 1 => 2^(5/2)
  | n + 2 => 4 * (4 * a (n + 1))^(1/4)

theorem general_term_formula (n : ℕ) :
  a n = 2^((10/3) * (1 - 1/4^n)) :=
by
  sorry

#eval a 1  -- This line is optional, for testing purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_formula_l556_55643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_rates_l556_55624

noncomputable def biking_rate : ℕ := sorry
noncomputable def jogging_rate : ℕ := sorry
noncomputable def swimming_rate : ℕ := sorry

axiom ed_distance : biking_rate * 1 + jogging_rate * 5 + swimming_rate * 4 = 90
axiom sue_distance : biking_rate * 4 + jogging_rate * 1 + swimming_rate * 5 = 115

theorem sum_of_squares_rates : 
  biking_rate^2 + jogging_rate^2 + swimming_rate^2 = 525 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_rates_l556_55624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_existence_l556_55671

/-- Given the medians to the legs and the angle at the vertex, 
    prove the existence of an isosceles triangle -/
theorem isosceles_triangle_existence 
  (m : ℝ) -- median to the leg
  (θ : ℝ) -- angle at the vertex
  (h_m : m > 0) -- median is positive
  (h_θ : 0 < θ ∧ θ < π) -- angle is between 0 and π
  : ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ -- sides are positive
    a = b ∧ -- isosceles condition
    c = 2 * m * Real.sin (θ / 2) ∧ -- relation between base, median, and vertex angle
    a^2 = m^2 + (c/2)^2 -- Apollonius theorem for isosceles triangle
  := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_existence_l556_55671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_l556_55632

/-- Number of ways to distribute k red and k white balls into n boxes under the given constraints -/
def number_of_distributions (n k : ℕ) : ℕ := 
  Nat.choose n k * Nat.choose (n - 1) k

/-- Theorem stating that the number of distributions is correct -/
theorem ball_distribution (n k : ℕ) (h : 1 ≤ k ∧ k ≤ n - 1) :
  number_of_distributions n k = Nat.choose n k * Nat.choose (n - 1) k := by
  -- Unfold the definition of number_of_distributions
  unfold number_of_distributions
  -- The equality holds by definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_l556_55632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_neg_one_sufficient_not_necessary_l556_55686

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of a line ax + y = b is -a -/
def line_slope (a : ℝ) : ℝ := -a

/-- The condition for l₁ ∥ l₂ -/
def parallel_condition (a : ℝ) : Prop := are_parallel (line_slope a) (1 / (line_slope a))

theorem a_equals_neg_one_sufficient_not_necessary (a : ℝ) :
  (a = -1 → parallel_condition a) ∧ ¬(parallel_condition a → a = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_neg_one_sufficient_not_necessary_l556_55686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trisectable_angles_l556_55664

-- Define a type for constructible angles
structure ConstructibleAngle where
  angle : ℝ

-- Define a predicate for trisectable angles
def IsTrisectable (angle : ConstructibleAngle) : Prop :=
  ∃ (smaller_angle : ConstructibleAngle), 3 * smaller_angle.angle = angle.angle

-- Define the property that if α is constructible, then 3α is constructible
noncomputable def triple_constructible (α : ConstructibleAngle) : ConstructibleAngle :=
  { angle := 3 * α.angle }

-- Define the given constructible angles
def angle_180 : ConstructibleAngle := { angle := 180 }
def angle_135 : ConstructibleAngle := { angle := 135 }
def angle_90 : ConstructibleAngle := { angle := 90 }
def angle_67_5 : ConstructibleAngle := { angle := 67.5 }
def angle_45 : ConstructibleAngle := { angle := 45 }
def angle_22_5 : ConstructibleAngle := { angle := 22.5 }

-- Theorem stating that the given angles are trisectable
theorem trisectable_angles :
  IsTrisectable angle_180 ∧
  IsTrisectable angle_135 ∧
  IsTrisectable angle_90 ∧
  IsTrisectable angle_67_5 ∧
  IsTrisectable angle_45 ∧
  IsTrisectable angle_22_5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trisectable_angles_l556_55664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_estimate_l556_55646

noncomputable def score (A E : ℝ) : ℝ := 2 / (1 + 0.05 * |A - E|)

theorem optimal_estimate :
  ∀ E : ℝ, 0 ≤ E ∧ E ≤ 1000 →
    ∃ A : ℝ, 0 ≤ A ∧ A ≤ 1000 ∧
      score A E ≤ score A 500 :=
by
  sorry

#check optimal_estimate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_estimate_l556_55646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_division_theorem_l556_55608

/-- Given a triangle ABC and a point M inside it, with AM, BM, and CM intersecting
    the opposite sides at D, E, and F respectively, there exist two segments among
    AD, BE, and CF such that M divides one in a ratio ≥ 2 and the other in a ratio ≤ 2. -/
theorem segment_division_theorem (A B C M D E F : EuclideanSpace ℝ (Fin 2)) :
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ M = t • A + (1 - t) • B + (1 - t) • C) →
  (∃ s : ℝ, D = s • A + (1 - s) • B) →
  (∃ u : ℝ, E = u • B + (1 - u) • C) →
  (∃ v : ℝ, F = v • C + (1 - v) • A) →
  (∃ (i j : Fin 3) (r₁ r₂ : ℝ), i ≠ j ∧
    ((i = 0 → r₁ = ‖A - M‖ / ‖M - D‖) ∧
     (i = 1 → r₁ = ‖B - M‖ / ‖M - E‖) ∧
     (i = 2 → r₁ = ‖C - M‖ / ‖M - F‖)) ∧
    ((j = 0 → r₂ = ‖A - M‖ / ‖M - D‖) ∧
     (j = 1 → r₂ = ‖B - M‖ / ‖M - E‖) ∧
     (j = 2 → r₂ = ‖C - M‖ / ‖M - F‖)) ∧
    r₁ ≥ (2 : ℝ) ∧ r₂ ≤ (2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_division_theorem_l556_55608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_triangle_numbers_are_pythagorean_l556_55674

/-- A Pythagorean triple is a tuple of three positive integers (a, b, c) that satisfy a² + b² = c² -/
def isPythagoreanTriple (a b c : ℕ+) : Prop :=
  a^2 + b^2 = c^2

/-- Pythagorean numbers are positive integers that form a Pythagorean triple -/
def isPythagoreanNumber (n : ℕ+) : Prop :=
  ∃ (a b : ℕ+), isPythagoreanTriple a b n ∨ isPythagoreanTriple a n b ∨ isPythagoreanTriple n a b

/-- Numbers that can form a right-angled triangle are Pythagorean numbers -/
theorem right_angle_triangle_numbers_are_pythagorean :
  ∀ (a b c : ℕ+), isPythagoreanTriple a b c → 
    isPythagoreanNumber a ∧ isPythagoreanNumber b ∧ isPythagoreanNumber c :=
by
  sorry

#check right_angle_triangle_numbers_are_pythagorean

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_triangle_numbers_are_pythagorean_l556_55674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_five_l556_55681

noncomputable def g (x : ℝ) : ℝ := 1 / (x + 1)

theorem g_composition_five : g (g (g (g (g (g 5))))) = 33 / 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_five_l556_55681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_plot_area_l556_55641

/-- Proves that a square plot with a perimeter that costs 4080 units to fence at 60 units per foot has an area of 289 square units. -/
theorem square_plot_area (cost_per_foot : ℝ) (total_cost : ℝ) 
  (h1 : cost_per_foot = 60) (h2 : total_cost = 4080) : 
  (total_cost / cost_per_foot / 4) ^ 2 = 289 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_plot_area_l556_55641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_m_for_integer_sum_l556_55600

/-- The product of the digits of a positive integer n -/
def f (n : ℕ+) : ℕ := sorry

/-- The sum of f(n) / m^(⌊log₁₀n⌋) from n = 1 to ∞ -/
noncomputable def S (m : ℕ+) : ℝ := sorry

/-- 2070 is the largest positive integer m such that S(m) is an integer -/
theorem largest_m_for_integer_sum : 
  (∀ k : ℕ+, k > 2070 → ¬(Int.floor (S k) = (S k))) ∧ 
  (Int.floor (S 2070) = (S 2070)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_m_for_integer_sum_l556_55600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l556_55690

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  f : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c
  max_value : ∃ x, f x = 12.5 ∧ ∀ y, f y ≤ 12.5
  solution_set : ∀ x, f x > 0 ↔ -2 < x ∧ x < 3

theorem quadratic_function_properties (qf : QuadraticFunction) :
  qf.a = -2 ∧ qf.b = 2 ∧ qf.c = 12 ∧
  (∃ x, qf.f x = 25/2 ∧ ∀ y, qf.f y ≤ qf.f x) ∧
  qf.f (1/2) = 25/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l556_55690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_partition_to_sparse_sets_l556_55603

def Is_n (n : ℕ) : Set ℕ := {i | 1 ≤ i ∧ i ≤ n}

def P_n (n : ℕ) : Set ℚ :=
  {q | ∃ (m k : ℕ), m ∈ Is_n n ∧ k ∈ Is_n n ∧ q = m / Real.sqrt (k : ℝ)}

def is_sparse_set (A : Set ℚ) : Prop :=
  ∀ x y, x ∈ A → y ∈ A → ¬∃ (z : ℕ), (x + y = (z : ℚ)^2)

def can_partition_to_sparse_sets (n : ℕ) : Prop :=
  ∃ A B : Set ℚ, A ∪ B = P_n n ∧ A ∩ B = ∅ ∧ is_sparse_set A ∧ is_sparse_set B

theorem max_partition_to_sparse_sets :
  (∀ n > 14, ¬can_partition_to_sparse_sets n) ∧
  can_partition_to_sparse_sets 14 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_partition_to_sparse_sets_l556_55603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_a_simplify_expression_b_l556_55649

theorem simplify_expression_a (a b x y : ℝ) (h : a * b * (x^2 - y^2) + x * y * (a^2 - b^2) ≠ 0) :
  (a * b * (x^2 + y^2) + x * y * (a^2 + b^2)) / (a * b * (x^2 - y^2) + x * y * (a^2 - b^2)) = (a * x + b * y) / (a * x - b * y) := by
  sorry

theorem simplify_expression_b (a : ℝ) (n x : ℤ) (h : a ≠ 0) :
  ((a^(n+x) - a^n) * (a^n - a^(n-x))) / ((a^(n+x) - a^n) - (a^n - a^(n-x))) = a^n := by
  sorry

#check simplify_expression_a
#check simplify_expression_b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_a_simplify_expression_b_l556_55649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_inspectors_B_is_12_l556_55665

/-- Represents the number of workshops -/
def num_workshops : ℕ := 9

/-- Represents the number of inspectors in group A -/
def num_inspectors_A : ℕ := 8

/-- Represents the number of workshops inspected by group A in the first two days -/
def workshops_A_first : ℕ := 2

/-- Represents the number of workshops inspected by group A in the next three days -/
def workshops_A_second : ℕ := 2

/-- Represents the number of workshops inspected by group B -/
def workshops_B : ℕ := 5

/-- Represents the number of days taken by group A for the first inspection -/
def days_A_first : ℕ := 2

/-- Represents the number of days taken by group A for the second inspection -/
def days_A_second : ℕ := 3

/-- Represents the number of days taken by group B for their inspection -/
def days_B : ℕ := 5

/-- Theorem stating that the number of inspectors in group B is 12 -/
theorem num_inspectors_B_is_12 (a b : ℕ) : ∃ num_inspectors_B : ℕ, num_inspectors_B = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_inspectors_B_is_12_l556_55665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_divides_2_pow_n_minus_n_l556_55631

theorem infinitely_many_n_divides_2_pow_n_minus_n (p : ℕ) (hp : Prime p) :
  ∃ (f : ℕ → ℕ), StrictMono f ∧ ∀ (k : ℕ), p ∣ (2^(f k) - f k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_divides_2_pow_n_minus_n_l556_55631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l556_55653

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

theorem symmetry_axis_of_f :
  ∀ (x : ℝ), f (Real.pi / 8 + x) = f (Real.pi / 8 - x) := by
  intro x
  unfold f
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l556_55653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_l556_55637

-- Define the square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define the equal segments
noncomputable def EqualSegment (s : Square) : ℝ := s.side / 4

-- Define the angles
noncomputable def AngleAdjacentSide (s : Square) : ℝ := sorry
noncomputable def AngleAtCorner (s : Square) : ℝ := sorry

-- State the theorem
theorem equal_angles (s : Square) : 
  AngleAdjacentSide s = AngleAtCorner s := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_l556_55637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base5_product_theorem_l556_55661

/-- Represents a number in base 5 --/
structure Base5 where
  value : Nat
  is_valid : value < 5^64 := by sorry

/-- Converts a base 5 number to its decimal representation --/
def to_decimal (n : Base5) : Nat := sorry

/-- Multiplies two base 5 numbers --/
def base5_mul (a b : Base5) : Base5 := sorry

/-- Converts a natural number to Base5 --/
def nat_to_base5 (n : Nat) : Base5 := ⟨n, sorry⟩

instance : Coe Nat Base5 where
  coe := nat_to_base5

/-- The main theorem stating that 132₅ * 12₅ = 2114₅ in base 5 --/
theorem base5_product_theorem :
  base5_mul (nat_to_base5 132) (nat_to_base5 12) = nat_to_base5 2114 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base5_product_theorem_l556_55661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_eval_l556_55679

theorem trig_expression_eval : 
  (Real.sin (35 * π / 180) - Real.sin (25 * π / 180)) / (Real.cos (35 * π / 180) - Real.cos (25 * π / 180)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_eval_l556_55679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_arrangements_l556_55602

/-- The number of distinct arrangements of n items with one fixed at the top -/
def arrangementsWithTopFixed (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The problem statement -/
theorem ice_cream_arrangements :
  arrangementsWithTopFixed 5 = 24 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_arrangements_l556_55602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_area_bounds_l556_55618

/-- The total length of fencing available -/
noncomputable def total_fencing : ℝ := 60

/-- The area of a rectangular pen given one side length -/
noncomputable def pen_area (x : ℝ) : ℝ := x * (total_fencing / 2 - x)

/-- The maximum possible area of the rectangular pen -/
noncomputable def max_area : ℝ := 225

/-- The minimum possible area of the rectangular pen -/
noncomputable def min_area : ℝ := 0

/-- Theorem stating the maximum and minimum areas of the rectangular pen -/
theorem pen_area_bounds :
  (∀ x, 0 < x → x < total_fencing / 2 → pen_area x ≤ max_area) ∧
  (∀ ε > 0, ∃ x > 0, x < total_fencing / 2 ∧ pen_area x < min_area + ε) := by
  sorry

#check pen_area_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_area_bounds_l556_55618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l556_55606

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, -2)

theorem angle_between_vectors :
  Real.arccos (-(Real.sqrt 10) / 10) = Real.pi - Real.arccos ((Real.sqrt 10) / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l556_55606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_mixed_fractions_l556_55660

theorem calculate_mixed_fractions : 
  (53 * ((3 + 1 / 4) - (3 + 3 / 4))) / ((1 + 2 / 3) + (2 + 2 / 5)) = -(6 + 57 / 122) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_mixed_fractions_l556_55660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_circle_center_l556_55605

noncomputable def polar_to_cartesian (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

def circle_equation (ρ θ : ℝ) : Prop :=
  ρ = -2 * Real.cos θ

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem distance_to_circle_center : 
  let P := polar_to_cartesian 2 (-π/3)
  let circle := circle_equation
  let center := (-1, 0)
  distance P.1 P.2 center.1 center.2 = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_circle_center_l556_55605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solution_l556_55604

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℤ := ⌊x⌋ + ⌊2*x⌋ + ⌊4*x⌋ + ⌊8*x⌋ + ⌊16*x⌋ + ⌊32*x⌋

-- Theorem statement
theorem no_real_solution : ∀ x : ℝ, f x ≠ 12345 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solution_l556_55604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pulley_centers_distance_l556_55650

/-- The distance between the centers of two pulleys -/
noncomputable def distance_between_centers (r₁ r₂ contact_distance : ℝ) : ℝ :=
  Real.sqrt ((r₁ - r₂)^2 + contact_distance^2)

/-- Theorem: Distance between centers of pulleys with given radii and contact distance -/
theorem pulley_centers_distance :
  let r₁ : ℝ := 10  -- radius of larger pulley
  let r₂ : ℝ := 6   -- radius of smaller pulley
  let d : ℝ := 30   -- distance between contact points
  distance_between_centers r₁ r₂ d = 2 * Real.sqrt 229 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pulley_centers_distance_l556_55650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_max_value_l556_55640

/-- Given a cubic polynomial with three real roots satisfying certain conditions,
    the maximum value of a specific expression is 3√3 / 2. -/
theorem cubic_polynomial_max_value 
  (a b c : ℝ) (lambda : ℝ) (x₁ x₂ x₃ : ℝ)
  (h_pos : lambda > 0)
  (h_roots : ∀ x, x^3 + a*x^2 + b*x + c = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)
  (h_diff : x₂ - x₁ = lambda)
  (h_order : x₃ > (x₁ + x₂) / 2) :
  ∃ (x : ℝ), ∀ (y : ℝ), (2*a^3 + 27*c - 9*a*b) / lambda^3 ≤ y ∧ 
  (∃ (a' b' c' : ℝ) (lambda' : ℝ) (x₁' x₂' x₃' : ℝ),
    lambda' > 0 ∧
    (∀ x, x^3 + a'*x^2 + b'*x + c' = 0 ↔ x = x₁' ∨ x = x₂' ∨ x = x₃') ∧
    x₂' - x₁' = lambda' ∧
    x₃' > (x₁' + x₂') / 2 ∧
    (2*a'^3 + 27*c' - 9*a'*b') / lambda'^3 = y) →
  y = 3 * Real.sqrt 3 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_max_value_l556_55640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_vertical_shift_l556_55616

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the shifted function g
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- Theorem statement
theorem graph_vertical_shift (f : ℝ → ℝ) (x y : ℝ) :
  y = g f x ↔ y - 2 = f x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_vertical_shift_l556_55616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_has_17_pennies_l556_55609

/-- The number of pennies Alex currently has -/
def a : ℚ := sorry

/-- The number of pennies Bob currently has -/
def b : ℚ := sorry

/-- Condition 1: If Alex gives Bob a penny, Bob will have four times as many pennies as Alex has -/
axiom condition1 : b + 1 = 4 * (a - 1)

/-- Condition 2: If Bob gives Alex two pennies, Bob will have twice as many pennies as Alex has -/
axiom condition2 : b - 2 = 2 * (a + 2)

/-- Theorem: Given the conditions, Bob currently has 17 pennies -/
theorem bob_has_17_pennies : b = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_has_17_pennies_l556_55609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_increasing_l556_55622

noncomputable def f (x : ℝ) : ℝ := 2^(-|x|)

theorem f_even_and_increasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 0 → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_increasing_l556_55622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l556_55680

def set_A : Set ℤ := {x | x^2 - 4*x ≤ 0}
def set_B : Set ℤ := {x : ℤ | -1 ≤ x ∧ x < 4}

theorem intersection_of_A_and_B :
  (set_A ∩ set_B) = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l556_55680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_20_l556_55691

-- Define the arithmetic sequence
noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

-- Define the sum of the first n terms
noncomputable def S (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- State the theorem
theorem max_sum_at_20 (a₁ : ℝ) (d : ℝ) (h₁ : a₁ > 0) 
  (h₂ : 3 * (arithmetic_sequence a₁ d 8) = 5 * (arithmetic_sequence a₁ d 13)) :
  ∀ n : ℕ, S a₁ d 20 ≥ S a₁ d n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_20_l556_55691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l556_55656

noncomputable def f (a b c : ℤ) (x : ℝ) : ℝ := (a * x^2 + 1) / (b * x + c)

theorem function_properties (a b c : ℤ) :
  (∀ x, f a b c (-x) = -f a b c x) →  -- f is an odd function
  f a b c 1 = 2 →                    -- f(1) = 2
  f a b c 2 < 3 →                    -- f(2) < 3
  (a = 1 ∧ b = 1 ∧ c = 0) ∧          -- Part 1: a = 1, b = 1, c = 0
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 →     -- Part 2: f is strictly decreasing in (0, 1)
    f a b c x > f a b c y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l556_55656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_proof_l556_55685

-- Define the function f
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- Define the theorem
theorem function_and_range_proof 
  (A ω φ : ℝ) 
  (h_A : A > 0) 
  (h_ω : ω > 0) 
  (h_φ : 0 < φ ∧ φ < Real.pi / 2) 
  (h_intersect : ∃ x, f A ω φ x = 0) 
  (h_distance : ∀ x y, f A ω φ x = 0 → f A ω φ y = 0 → x ≠ y → |x - y| = Real.pi / 2) 
  (h_lowest : f A ω φ (2 * Real.pi / 3) = -2) :
  (f A ω φ = λ x ↦ 2 * Real.sin (2 * x + Real.pi / 6)) ∧ 
  (∀ x, x ∈ Set.Icc (Real.pi / 12) (Real.pi / 2) → 
    f A ω φ x ∈ Set.Icc (-1) 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_proof_l556_55685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_sum_l556_55642

theorem complex_fraction_sum (a b : ℝ) : 
  (Complex.I + 1) / (1 - Complex.I) = Complex.mk a b → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_sum_l556_55642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_to_origin_l556_55607

noncomputable def f (x : ℝ) := x + 1/x

theorem closest_point_to_origin (x : ℝ) (hx : x > 0) :
  let y := f x
  let distance_squared := x^2 + y^2
  distance_squared ≥ 2 + 2*Real.sqrt 2 ∧
  distance_squared = 2 + 2*Real.sqrt 2 ↔ x = 1 / Real.sqrt (Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_to_origin_l556_55607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l556_55636

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

noncomputable def sum_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 2017 = 1) :
  sum_arithmetic_sequence a 2019 = 2019 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l556_55636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_area_l556_55628

/-- The area of the largest circle formed from a string that fits exactly around a rectangle with area 180 and length-to-width ratio of 3:2 -/
theorem largest_circle_area (rectangle_area : ℝ) (length_width_ratio : ℝ) (circle_area : ℕ) : 
  rectangle_area = 180 →
  length_width_ratio = 3/2 →
  circle_area = Int.floor (750 / Real.pi + 0.5) →
  circle_area = 239 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_area_l556_55628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_term_coefficient_l556_55612

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expansion function
noncomputable def expansion_coefficient (x : ℝ) : ℝ := (x - 1) * (1/x + x)^6

-- Theorem statement
theorem linear_term_coefficient :
  (deriv expansion_coefficient) 0 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_term_coefficient_l556_55612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_husband_catches_up_in_15_minutes_l556_55696

/-- The time it takes for the husband to catch up to Yolanda -/
noncomputable def catchUpTime (yolandaSpeed : ℝ) (husbandSpeed : ℝ) (headStart : ℝ) : ℝ :=
  (yolandaSpeed * headStart) / (husbandSpeed - yolandaSpeed)

/-- Theorem stating that the husband catches up to Yolanda in 15 minutes -/
theorem husband_catches_up_in_15_minutes :
  let yolandaSpeed : ℝ := 20  -- miles per hour
  let husbandSpeed : ℝ := 40  -- miles per hour
  let headStart : ℝ := 15 / 60  -- 15 minutes converted to hours
  catchUpTime yolandaSpeed husbandSpeed headStart * 60 = 15 := by
  sorry

#check husband_catches_up_in_15_minutes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_husband_catches_up_in_15_minutes_l556_55696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_government_purchase_amount_l556_55683

/-- Represents the market for a good with linear demand and supply functions -/
structure Market where
  initial_price : ℚ
  initial_quantity : ℚ
  gov_purchase_price : ℚ
  market_quantity_increase : ℚ
  max_price : ℚ

/-- Calculates the quantity of goods purchased by the government -/
noncomputable def government_purchase (m : Market) : ℚ :=
  let new_market_quantity := m.initial_quantity + m.market_quantity_increase
  let demand_slope := (m.max_price - m.gov_purchase_price) / new_market_quantity
  let demand_intercept := m.max_price
  let equilibrium_quantity := (demand_intercept - m.initial_price) / demand_slope
  equilibrium_quantity - m.initial_quantity

/-- Theorem stating that given the market conditions, the government purchased 48 units -/
theorem government_purchase_amount (m : Market) 
    (h1 : m.initial_price = 14)
    (h2 : m.initial_quantity = 42)
    (h3 : m.gov_purchase_price = 20)
    (h4 : m.market_quantity_increase = 12)
    (h5 : m.max_price = 29) :
  government_purchase m = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_government_purchase_amount_l556_55683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrasta_um_solvable_l556_55676

/-- Represents a move in the Arrasta Um game -/
inductive Move
| Up    : Move
| Down  : Move
| Left  : Move
| Right : Move

/-- Represents a position on the board -/
structure Position :=
  (row : Nat) (col : Nat)

/-- Represents the Arrasta Um game board -/
structure Board :=
  (size : Nat)
  (blackPiece : Position)

/-- Checks if a position is valid on the board -/
def Board.isValidPosition (b : Board) (p : Position) : Prop :=
  p.row < b.size ∧ p.col < b.size

/-- Checks if a move is valid from a given position -/
def Board.isValidMove (b : Board) (p : Position) (m : Move) : Prop :=
  match m with
  | Move.Up    => p.row < b.size - 1
  | Move.Down  => p.row > 0
  | Move.Left  => p.col > 0
  | Move.Right => p.col < b.size - 1

/-- Applies a move to a position -/
def applyMove (p : Position) (m : Move) : Position :=
  match m with
  | Move.Up    => ⟨p.row + 1, p.col⟩
  | Move.Down  => ⟨p.row - 1, p.col⟩
  | Move.Left  => ⟨p.row, p.col - 1⟩
  | Move.Right => ⟨p.row, p.col + 1⟩

/-- Represents a sequence of moves -/
def MoveSequence := List Move

/-- Checks if a sequence of moves is valid and solves the game -/
def isSolvingSequence (b : Board) (seq : MoveSequence) : Prop :=
  seq.length = 6 * b.size - 8 ∧
  (seq.foldl (λ pos move => applyMove pos move) b.blackPiece) = ⟨b.size - 1, b.size - 1⟩ ∧
  ∀ (i : Fin seq.length),
    let pos := (seq.take i.val).foldl (λ pos move => applyMove pos move) b.blackPiece
    b.isValidPosition pos ∧ b.isValidMove pos (seq.get i)

/-- The main theorem stating that Arrasta Um can be solved in 6n - 8 moves -/
theorem arrasta_um_solvable (n : Nat) (h : n ≥ 2) :
  ∃ (seq : MoveSequence), isSolvingSequence ⟨n, ⟨0, 0⟩⟩ seq := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrasta_um_solvable_l556_55676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l556_55627

/-- Given a triangle ABC with angles A, B, C and sides a, b, c opposite to these angles respectively. -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem triangle_properties (t : Triangle) 
  (hm : (Real.sin t.A, Real.sin t.B) = (Real.sin t.A, Real.sin t.B))
  (hn : (Real.cos t.B, Real.cos t.A) = (Real.cos t.B, Real.cos t.A))
  (h_dot : dot_product (Real.sin t.A, Real.sin t.B) (Real.cos t.B, Real.cos t.A) = Real.sin (2 * t.C))
  (h_arithmetic : ∃ k, Real.sin t.A + Real.sin t.B = 2 * Real.sin t.C + 2 * k)
  (h_dot_sides : t.a * t.b * Real.cos t.C = 18) :
  t.C = π / 3 ∧ t.c = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l556_55627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_probability_seven_out_of_twelve_l556_55693

theorem odd_sum_probability_seven_out_of_twelve : 
  (Nat.choose 6 5 * Nat.choose 6 2 + 
   Nat.choose 6 3 * Nat.choose 6 4 + 
   Nat.choose 6 1 * Nat.choose 6 6 : ℚ) / Nat.choose 12 7 = 1 / 2 := by
  sorry

#eval (Nat.choose 6 5 * Nat.choose 6 2 + 
       Nat.choose 6 3 * Nat.choose 6 4 + 
       Nat.choose 6 1 * Nat.choose 6 6 : ℚ) / Nat.choose 12 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_probability_seven_out_of_twelve_l556_55693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_value_from_limit_l556_55651

open Real

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define x₀
variable (x₀ : ℝ)

-- Define the limit condition
variable (h : Tendsto (λ Δx => (f x₀ - f (x₀ + 2*Δx)) / Δx) (𝓝 0) (𝓝 2))

-- State the theorem
theorem derivative_value_from_limit :
  deriv f x₀ = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_value_from_limit_l556_55651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_part1_binomial_coefficient_part2_l556_55675

-- Part 1
theorem binomial_coefficient_part1 (a : ℝ) (h_a : a > 0) :
  (Finset.range 7).sum (λ i ↦ (Nat.choose 6 i) * (2^i) * (a^(6-i))) = 3^10 →
  (Nat.choose 6 2) * (2^2) * (a^4) = 960 →
  a = 2 := by sorry

-- Part 2
theorem binomial_coefficient_part2 (a : ℝ) (n : ℕ) (h_a : a > 0) (h_n : n > 1) :
  (a + 2)^n = 3^10 →
  n + a = 12 →
  (Finset.range (n+1)).sum (λ i ↦ Nat.choose n i) = 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_part1_binomial_coefficient_part2_l556_55675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_construction_l556_55648

/-- Represents a plane with points and lines -/
structure Plane :=
  (point : Type)
  (line : Type)
  (on_line : point → line → Prop)
  (perpendicular : line → line → Prop)

/-- Represents an ellipse on a plane -/
structure Ellipse (π : Plane) :=
  (center : π.point)
  (major_axis : π.line)
  (minor_axis : π.line)

/-- Defines when an ellipse is tangent to a line at a point -/
def tangent_at {π : Plane} (e : Ellipse π) (l : π.line) (p : π.point) : Prop := sorry

/-- Defines when an ellipse passes through a point -/
def passes_through {π : Plane} (e : Ellipse π) (p : π.point) : Prop := sorry

/-- Defines when two points are in the same quadrant formed by two lines -/
def same_quadrant {π : Plane} (p1 p2 : π.point) (l1 l2 : π.line) : Prop := sorry

/-- Theorem stating the existence of an ellipse satisfying the given conditions -/
theorem ellipse_construction 
  (π : Plane) 
  (e f : π.line) 
  (G H : π.point) 
  (h_perp : π.perpendicular e f)
  (h_not_on_e : ¬π.on_line G e ∧ ¬π.on_line H e)
  (h_not_on_f : ¬π.on_line G f ∧ ¬π.on_line H f)
  (h_same_quadrant : same_quadrant G H e f) :
  ∃ (E : Ellipse π), 
    passes_through E G ∧ 
    passes_through E H ∧ 
    (∃ (B : π.point), π.on_line B e ∧ tangent_at E e B) ∧
    (∃ (C : π.point), π.on_line C f ∧ tangent_at E f C) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_construction_l556_55648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocals_negative_two_and_negative_half_l556_55697

theorem reciprocals_negative_two_and_negative_half :
  let a : ℚ := -2
  let b : ℚ := -1/2
  (a * b = 1) → (a = b⁻¹) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocals_negative_two_and_negative_half_l556_55697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_canonical_form_l556_55625

/-- The line of intersection of two planes in 3D space -/
def IntersectionLine (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) : Set (ℝ × ℝ × ℝ) :=
  {p | a₁ * p.1 + b₁ * p.2.1 + c₁ * p.2.2 + d₁ = 0 ∧
       a₂ * p.1 + b₂ * p.2.1 + c₂ * p.2.2 + d₂ = 0}

/-- The canonical form of a line in 3D space -/
def CanonicalLine (x₀ y₀ z₀ m n p : ℝ) : Set (ℝ × ℝ × ℝ) :=
  {q | ∃ (t : ℝ), q = (x₀ + m * t, y₀ + n * t, z₀ + p * t)}

theorem intersection_line_canonical_form :
  IntersectionLine 1 (-2) 1 (-4) 2 2 (-1) (-8) =
  CanonicalLine 4 0 0 0 1 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_canonical_form_l556_55625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_six_l556_55638

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x) * Real.sin (x - 2) + x + 1

-- Define the interval
def interval : Set ℝ := Set.Icc (-1) 5

-- State the theorem
theorem max_min_sum_equals_six :
  ∃ (M m : ℝ), (∀ x ∈ interval, f x ≤ M) ∧
                (∀ x ∈ interval, m ≤ f x) ∧
                (∃ x₁ ∈ interval, f x₁ = M) ∧
                (∃ x₂ ∈ interval, f x₂ = m) ∧
                (M + m = 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_six_l556_55638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_occurrence_l556_55670

/-- A polynomial with natural number coefficients -/
def MyPolynomial := ℕ → ℕ

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Sequence of sums of digits of P(n) -/
def digit_sum_sequence (P : MyPolynomial) (n : ℕ) : ℕ :=
  sum_of_digits (P n)

/-- There exists a number that appears infinitely many times in the sequence -/
theorem infinite_occurrence (P : MyPolynomial) :
  ∃ k : ℕ, ∀ N : ℕ, ∃ n ≥ N, digit_sum_sequence P n = k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_occurrence_l556_55670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_units_l556_55689

-- Define the custom units
inductive CustomUnit
| Centimeter
| Decimeter
| Minute
| Kilometer

-- Define a measurement with a value and a unit
structure Measurement where
  value : ℕ
  unit : CustomUnit

-- Define the function to determine the appropriate unit
def appropriate_unit (m : Measurement) : CustomUnit :=
  match m with
  | ⟨150, _⟩ => CustomUnit.Centimeter
  | ⟨7, _⟩ => CustomUnit.Decimeter
  | ⟨40, _⟩ => CustomUnit.Minute
  | ⟨1000, _⟩ => CustomUnit.Kilometer
  | _ => CustomUnit.Centimeter  -- Default case

-- Theorem statement
theorem correct_units 
  (child : Measurement)
  (table : Measurement)
  (class_period : Measurement)
  (railway : Measurement)
  (h1 : child.value = 150)
  (h2 : table.value = 7)
  (h3 : class_period.value = 40)
  (h4 : railway.value = 1000) :
  appropriate_unit child = CustomUnit.Centimeter ∧
  appropriate_unit table = CustomUnit.Decimeter ∧
  appropriate_unit class_period = CustomUnit.Minute ∧
  appropriate_unit railway = CustomUnit.Kilometer :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_units_l556_55689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_is_focus_l556_55658

/-- The parabola defined by y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The fixed line l defined by x = -1 -/
def line_l (x : ℝ) : Prop := x = -1

/-- The distance from a point (x,y) to the line x = -1 -/
def dist_to_line (x y : ℝ) : ℝ := |x + 1|

/-- The distance from a point (x,y) to the point (1,0) -/
noncomputable def dist_to_point (x y : ℝ) : ℝ := Real.sqrt ((x - 1)^2 + y^2)

/-- The theorem stating that (1,0) is the fixed point F -/
theorem fixed_point_is_focus :
  ∀ x y : ℝ, parabola x y →
  (∀ p q : ℝ, parabola p q → dist_to_line p q = dist_to_point p q) →
  (1, 0) = (1, 0) := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_is_focus_l556_55658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_proof_l556_55647

/-- Calculates the final amount for a compound interest investment -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / periods) ^ (periods * time)

theorem investment_difference_proof :
  let principal := 60000
  let rate := 0.05
  let time := 3
  let john_periods := 1
  let emma_periods := 2
  let john_final := compound_interest principal rate john_periods time
  let emma_final := compound_interest principal rate emma_periods time
  ⌊emma_final - john_final⌋ = 99 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_proof_l556_55647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_eq_cos_x_div_2_solutions_l556_55630

theorem tan_2x_eq_cos_x_div_2_solutions :
  ∃ (S : Finset ℝ), S.card = 5 ∧
  (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi) ∧
  (∀ x ∈ S, Real.tan (2 * x) = Real.cos (x / 2)) ∧
  (∀ y, 0 ≤ y ∧ y ≤ 2 * Real.pi → Real.tan (2 * y) = Real.cos (y / 2) → y ∈ S) :=
by sorry

#check tan_2x_eq_cos_x_div_2_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_eq_cos_x_div_2_solutions_l556_55630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cowboy_shortest_path_l556_55639

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The shortest path problem for the cowboy -/
theorem cowboy_shortest_path (stream_y cabin_x cabin_y cowboy_x cowboy_y : ℝ)
  (h1 : stream_y = 0)
  (h2 : cabin_x = cowboy_x + 10)
  (h3 : cabin_y = cowboy_y - 9)
  (h4 : cowboy_y = -3) :
  let cowboy := Point.mk cowboy_x cowboy_y
  let cabin := Point.mk cabin_x cabin_y
  let stream_point := Point.mk cowboy_x stream_y
  (distance cowboy stream_point) + (distance stream_point cabin) = 3 + Real.sqrt 325 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cowboy_shortest_path_l556_55639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_periodic_decreasing_function_inequality_l556_55644

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def periodic_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 - x) = f x

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def acute_angles_of_obtuse_triangle (α β : ℝ) : Prop :=
  0 < α ∧ α < Real.pi/2 ∧ 0 < β ∧ β < Real.pi/2 ∧ Real.pi/2 < α + β ∧ α + β < Real.pi

theorem even_periodic_decreasing_function_inequality
  (f : ℝ → ℝ) (α β : ℝ)
  (h1 : is_even_function f)
  (h2 : periodic_two f)
  (h3 : decreasing_on f (-3) (-2))
  (h4 : acute_angles_of_obtuse_triangle α β) :
  f (Real.sin α) < f (Real.cos β) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_periodic_decreasing_function_inequality_l556_55644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_b_value_l556_55615

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ : ℝ} : 
  (∃ (b₁ b₂ : ℝ), ∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The first line equation: 3y - 3b = 9x -/
def line1 (b : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ 3 * y - 3 * b = 9 * x

/-- The second line equation: y + 2 = (b + 9)x -/
def line2 (b : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ y + 2 = (b + 9) * x

/-- Two lines are parallel -/
def Parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ m₁ m₂ b₁ b₂, (∀ x y, f x y ↔ y = m₁ * x + b₁) ∧
                 (∀ x y, g x y ↔ y = m₂ * x + b₂) ∧
                 m₁ = m₂

theorem parallel_lines_b_value :
  (∃ b, ∀ x y, (line1 b x y ∧ line2 b x y) → Parallel (line1 b) (line2 b)) →
  ∃ b, b = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_b_value_l556_55615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l556_55682

-- Define the train's length in meters
noncomputable def train_length : ℝ := 80

-- Define the train's speed in kilometers per hour
noncomputable def train_speed_kmph : ℝ := 36

-- Define the conversion factor from km/h to m/s
noncomputable def kmph_to_ms : ℝ := 1000 / 3600

-- Calculate the train's speed in meters per second
noncomputable def train_speed_ms : ℝ := train_speed_kmph * kmph_to_ms

-- Theorem: The time for the train to pass a telegraph post is 8 seconds
theorem train_passing_time :
  train_length / train_speed_ms = 8 := by
  -- Expand the definitions
  unfold train_length train_speed_ms train_speed_kmph kmph_to_ms
  -- Perform the calculation
  norm_num
  -- The proof is completed automatically
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l556_55682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l556_55657

/-- Regular triangular pyramid with base side length a and lateral edge length b -/
structure RegularTriangularPyramid where
  a : ℝ  -- base side length
  b : ℝ  -- lateral edge length
  a_pos : 0 < a
  b_pos : 0 < b

/-- Plane passing through midpoints of two base edges and one lateral edge -/
structure MidpointPlane (p : RegularTriangularPyramid)

/-- Area of the cross-section formed by the midpoint plane -/
noncomputable def crossSectionArea (p : RegularTriangularPyramid) (plane : MidpointPlane p) : ℝ :=
  1/4 * p.a * p.b

theorem cross_section_area_theorem (p : RegularTriangularPyramid) (plane : MidpointPlane p) :
  crossSectionArea p plane = 1/4 * p.a * p.b := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l556_55657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l556_55663

/-- Converts polar coordinates to rectangular coordinates -/
noncomputable def polar_to_rectangular (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

/-- The point in polar coordinates -/
noncomputable def polar_point : ℝ × ℝ := (4, Real.pi / 3)

/-- The expected point in rectangular coordinates -/
noncomputable def rectangular_point : ℝ × ℝ := (2, 2 * Real.sqrt 3)

theorem polar_to_rectangular_conversion :
  polar_to_rectangular polar_point.1 polar_point.2 = rectangular_point := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l556_55663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_form_sum_l556_55677

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let (xE, yE) := q.E
  let (xF, yF) := q.F
  let (xG, yG) := q.G
  let (xH, yH) := q.H
  -- EF = 9
  (xE - xF)^2 + (yE - yF)^2 = 81 ∧
  -- FG = 5
  (xF - xG)^2 + (yF - yG)^2 = 25 ∧
  -- GH = 12
  (xG - xH)^2 + (yG - yH)^2 = 144 ∧
  -- EH = 12
  (xE - xH)^2 + (yE - yH)^2 = 144 ∧
  -- ∠EHG = 75°
  let vEH := (xE - xH, yE - yH)
  let vGH := (xG - xH, yG - yH)
  Real.arccos ((vEH.1 * vGH.1 + vEH.2 * vGH.2) / (12 * 12)) = 75 * Real.pi / 180

-- Define the area function
noncomputable def area (q : Quadrilateral) : ℝ := sorry

-- Define squarefree property
def is_squarefree (n : ℕ) : Prop :=
  ∀ d : ℕ, d > 1 → d * d ∣ n → d = n

-- Theorem statement
theorem area_form_sum (q : Quadrilateral) 
  (h : is_valid_quadrilateral q) : 
  ∃ (a b c : ℕ), (is_squarefree a ∧ is_squarefree c) ∧ 
  area q = Real.sqrt (a : ℝ) + b * Real.sqrt (c : ℝ) ∧
  a + b + c = 50 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_form_sum_l556_55677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_survival_probability_l556_55694

/-- Represents the data for the largest sample size in the tree transplantation experiment -/
structure TreeData where
  transplanted : ℕ
  survived : ℕ

/-- Calculates the survival rate given the tree data -/
def survivalRate (data : TreeData) : ℚ :=
  data.survived / data.transplanted

/-- Rounds a rational number to the nearest tenth -/
def roundToNearestTenth (x : ℚ) : ℚ :=
  ⌊(x * 10 + 1/2)⌋ / 10

/-- The estimated probability of survival for young trees -/
theorem estimated_survival_probability (data : TreeData)
  (h1 : data.transplanted = 20000)
  (h2 : data.survived = 18044) :
  roundToNearestTenth (survivalRate data) = 9/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_survival_probability_l556_55694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l556_55692

theorem inequality_holds_iff_a_in_range (a : ℝ) : 
  (∀ θ : ℝ, θ ∈ Set.Icc 0 (π / 2) → 
    ∀ x : ℝ, (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (7/2 ≤ a ∧ a ≤ Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l556_55692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_four_digit_special_number_l556_55659

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def all_digits_different (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i j, 0 ≤ i ∧ i < j ∧ j < digits.length → digits[i]! ≠ digits[j]!

def divisible_by_all_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ≠ 0 → n % d = 0

def no_sequential_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i, 0 ≤ i ∧ i < digits.length - 1 → digits[i]! + 1 ≠ digits[i+1]!

theorem least_four_digit_special_number :
  ∃ n : ℕ, 
    is_four_digit n ∧
    all_digits_different n ∧
    divisible_by_all_digits n ∧
    no_sequential_digits n ∧
    (∀ m : ℕ, m < n →
      ¬(is_four_digit m ∧
        all_digits_different m ∧
        divisible_by_all_digits m ∧
        no_sequential_digits m)) ∧
    n = 1328 :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_four_digit_special_number_l556_55659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_subject_A_prob_one_subject_B_prefer_university_A_l556_55687

-- Define the probabilities for University A
noncomputable def prob_A : ℝ := 1/2

-- Define the probabilities for University B
noncomputable def prob_B1 : ℝ := 1/6
noncomputable def prob_B2 : ℝ := 3/5
noncomputable def prob_B3 (m : ℝ) : ℝ := m

-- Define the number of subjects
def num_subjects : ℕ := 3

-- Theorem for the probability of passing exactly one subject in University A
theorem prob_one_subject_A :
  (3 : ℝ) * prob_A * (1 - prob_A)^2 = 3/8 := by sorry

-- Theorem for the probability of passing exactly one subject in University B when m = 3/5
theorem prob_one_subject_B :
  let m : ℝ := 3/5
  (prob_B1 * (1 - prob_B2) * (1 - m) +
   (1 - prob_B1) * prob_B2 * (1 - m) +
   (1 - prob_B1) * (1 - prob_B2) * m) = 32/75 := by sorry

-- Theorem for the range of m where University A is preferred
theorem prefer_university_A (m : ℝ) :
  0 < m ∧ m < 11/15 ↔
  (3 : ℝ) * prob_A < prob_B1 + prob_B2 + m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_subject_A_prob_one_subject_B_prefer_university_A_l556_55687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_circle_area_l556_55688

theorem quarter_circle_area : 
  (∫ x in (Set.Icc 0 1), Real.sqrt (1 - x^2)) = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_circle_area_l556_55688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_theorem_l556_55611

def vector_problem (a b : ℝ × ℝ) : Prop :=
  let angle := Real.pi / 3
  let norm_b := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  a = (2, 0) ∧ 
  norm_b = 1 ∧
  (a.1 * b.1 + a.2 * b.2) = (Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))) * norm_b * Real.cos angle →
  Real.sqrt (((a.1 - 2 * b.1) ^ 2) + ((a.2 - 2 * b.2) ^ 2)) = 2

theorem vector_problem_theorem : 
  ∃ (a b : ℝ × ℝ), vector_problem a b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_theorem_l556_55611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_plastering_cost_l556_55629

/-- The cost per square meter for plastering a tank -/
noncomputable def cost_per_square_meter (length width depth : ℝ) (total_cost : ℝ) : ℝ :=
  let wall_area := 2 * (length * depth + width * depth)
  let bottom_area := length * width
  let total_area := wall_area + bottom_area
  total_cost / total_area

/-- Theorem: The cost per square meter for plastering the given tank is 0.75 -/
theorem tank_plastering_cost :
  cost_per_square_meter 25 12 6 558 = 0.75 := by
  -- Unfold the definition of cost_per_square_meter
  unfold cost_per_square_meter
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_plastering_cost_l556_55629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l556_55610

/-- The equation of the region -/
def region_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y = -9

/-- The area of the region -/
noncomputable def region_area : ℝ := 4 * Real.pi

theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y : ℝ, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l556_55610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_denver_to_la_distance_l556_55623

/-- The distance between two points on a complex plane -/
noncomputable def distance (z₁ z₂ : ℂ) : ℝ :=
  Real.sqrt ((z₁.re - z₂.re)^2 + (z₁.im - z₂.im)^2)

theorem denver_to_la_distance :
  let los_angeles : ℂ := 0
  let boston : ℂ := 0 + 3200 * I
  let denver : ℂ := 1200 + 1600 * I
  distance denver los_angeles = 3200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_denver_to_la_distance_l556_55623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l556_55655

/-- The cost price of a piece of clothing given its marked price, discount, and profit margin. -/
theorem cost_price_calculation (marked_price discount profit : ℝ) 
  (h1 : marked_price = 132)
  (h2 : discount = 0.1)
  (h3 : profit = 0.1) :
  (marked_price * (1 - discount)) / (1 + profit) = 108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l556_55655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_william_land_percentage_l556_55617

/-- Given the total tax collected and an individual's tax payment, 
    calculate the percentage of land owned by the individual. -/
noncomputable def land_percentage (total_tax : ℝ) (individual_tax : ℝ) : ℝ :=
  (individual_tax / total_tax) * 100

/-- Theorem stating that Mr. William's land percentage is 9.6% -/
theorem william_land_percentage :
  let total_tax := (5000 : ℝ)
  let william_tax := (480 : ℝ)
  land_percentage total_tax william_tax = 9.6 := by
  -- Unfold the definition of land_percentage
  unfold land_percentage
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_william_land_percentage_l556_55617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l556_55601

open Real

def solution_set (x : ℝ) : Prop :=
  (∃ k : ℤ, x = k * Real.pi / 4 ∧ ∀ m : ℤ, k ≠ m / 3 + 2 * m) ∨
  (∃ n : ℤ, x = Real.pi / 18 + n * Real.pi / 3 ∨ x = -Real.pi / 18 + n * Real.pi / 3)

theorem trigonometric_equation_solution :
  ∀ x : ℝ, cos (6 * x) ≠ 0 →
  (tan (6 * x) * cos (2 * x) - sin (2 * x) - 2 * sin (4 * x) = 0 ↔ solution_set x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l556_55601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_count_l556_55635

theorem worker_count (total extra_total extra_per_worker : ℕ) 
  (h1 : total = 300000)
  (h2 : extra_total = 320000)
  (h3 : extra_per_worker = 50) :
  ∃ (w : ℕ), w * (extra_total / w - total / w) = extra_total - total ∧ w = 400 := by
  
  -- Proof goes here
  sorry

#check worker_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_count_l556_55635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_complex_numbers_with_equilateral_property_l556_55645

/-- A complex number z satisfies the equilateral triangle property if 0, z, and z^4 form the vertices of an equilateral triangle in the complex plane. -/
def has_equilateral_property (z : ℂ) : Prop :=
  z ≠ 0 ∧ 
  (Complex.abs z = Complex.abs (z^4 - z)) ∧ 
  (Complex.abs z = Complex.abs z^4) ∧
  (z ≠ z^4) ∧ (z^4 ≠ 0)

/-- There are exactly two nonzero complex numbers that satisfy the equilateral triangle property. -/
theorem two_complex_numbers_with_equilateral_property : 
  ∃! (s : Finset ℂ), (∀ z ∈ s, has_equilateral_property z) ∧ Finset.card s = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_complex_numbers_with_equilateral_property_l556_55645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_inner_square_area_8cm_l556_55699

/-- The area of the square formed by connecting alternate vertices of a regular octagon -/
noncomputable def octagon_inner_square_area (side_length : ℝ) : ℝ :=
  let diagonal := side_length * (1 + Real.sqrt 2)
  diagonal ^ 2

/-- Theorem: The area of the square formed by connecting alternate vertices 
    of a regular octagon with side length 8 cm is equal to 192 + 128√2 square cm -/
theorem octagon_inner_square_area_8cm :
  octagon_inner_square_area 8 = 192 + 128 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_inner_square_area_8cm_l556_55699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_circle_C_intersection_point_P_l556_55668

-- Define the circle C
noncomputable def circle_C (φ : ℝ) : ℝ × ℝ := (1 + Real.cos φ, Real.sin φ)

-- Define the ray OM
noncomputable def ray_OM : ℝ := Real.pi / 4

-- Theorem for the polar equation of circle C
theorem polar_equation_circle_C :
  ∀ θ : ℝ, (2 * Real.cos θ = Real.sqrt ((circle_C θ).1^2 + (circle_C θ).2^2)) := by sorry

-- Theorem for the intersection point P
theorem intersection_point_P :
  ∃ P : ℝ × ℝ, 
    P.1 = Real.sqrt 2 ∧ 
    P.2 = Real.pi / 4 ∧ 
    P ∈ {p | ∃ φ : ℝ, p = circle_C φ} ∧
    P.2 = ray_OM := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_circle_C_intersection_point_P_l556_55668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l556_55654

/-- Given vectors a, b, and c in ℝ², if λa + b is collinear with c, then λ = -1 -/
theorem collinear_vectors (a b c : ℝ × ℝ) (lambda : ℝ) 
  (ha : a = (1, 2)) 
  (hb : b = (2, 0)) 
  (hc : c = (1, -2)) 
  (h_collinear : ∃ (k : ℝ), k ≠ 0 ∧ lambda • a + b = k • c) : 
  lambda = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l556_55654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_l556_55667

-- Define the function
noncomputable def f (A : ℝ) (x : ℝ) : ℝ := Real.cos (2 * x + A)

-- State the theorem
theorem angle_value (A : ℝ) : 
  (0 < A ∧ A < Real.pi) →  -- A is an internal angle
  (∀ x, f A (Real.pi/3 + x) = f A (Real.pi/3 - x)) →  -- (π/3, 0) is a symmetric center
  A = 5 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_l556_55667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_value_l556_55698

theorem unique_x_value : ∃! x : ℝ, (2 ∈ ({x + 4, x^2 + x} : Set ℝ)) ∧ (x + 4 ≠ x^2 + x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_value_l556_55698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_five_correct_l556_55695

/-- The probability of the digit 5 appearing among the first n digits
    in the decimal representation of a randomly selected number from [0,1] -/
noncomputable def probability_of_five (n : ℕ) : ℝ :=
  1 - (9/10)^n

theorem probability_of_five_correct (n : ℕ) :
  probability_of_five n =
  Finset.sum (Finset.range n) (fun m => (9:ℝ)^(m:ℕ) * (1/10)^(m+1:ℕ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_five_correct_l556_55695
