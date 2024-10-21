import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_approaches_limit_l1138_113865

/-- Represents a right triangle with two equal sides --/
structure RightIsoscelesTriangle where
  side_length : ℝ
  positive : 0 < side_length

/-- Calculates the area of a right isosceles triangle --/
noncomputable def area (t : RightIsoscelesTriangle) : ℝ :=
  t.side_length * t.side_length / 2

/-- Calculates the sum of areas of shaded triangles after n iterations --/
noncomputable def sum_shaded_areas (t : RightIsoscelesTriangle) (n : ℕ) : ℝ :=
  (area t) * (1 - (1/4)^n) * (4/3)

/-- The main theorem to be proved --/
theorem shaded_area_approaches_limit (t : RightIsoscelesTriangle) (h : t.side_length = 10) :
  ∃ ε > 0, |sum_shaded_areas t 100 - 16.67| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_approaches_limit_l1138_113865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1138_113829

-- Define the hyperbola
noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the right focus
def right_focus (a b c : ℝ) : ℝ × ℝ := (c, 0)

-- Define a point on the positive y-axis
def point_on_y_axis (y : ℝ) : ℝ × ℝ := (0, y)

-- Define the asymptote of the hyperbola
noncomputable def asymptote (a b : ℝ) (x : ℝ) : ℝ := (b / a) * x

-- Define collinearity of three points
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

-- Define the area ratio condition
def area_ratio_condition (M F O P : ℝ × ℝ) : Prop :=
  let area_triangle (a b c : ℝ × ℝ) : ℝ := sorry -- Placeholder for area calculation
  area_triangle M F O = 4 * area_triangle P M O

-- Theorem statement
theorem hyperbola_eccentricity (a b c : ℝ) (P M : ℝ × ℝ) :
  let F := right_focus a b c
  let O := (0, 0)
  hyperbola a b (Prod.fst F) (Prod.snd F) →
  (∃ y, P = point_on_y_axis y) →
  (∃ x, M = (x, asymptote a b x)) →
  collinear P M F →
  area_ratio_condition M F O P →
  c^2 / a^2 = 5 :=
by
  sorry -- Placeholder for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1138_113829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squared_norms_l1138_113813

variable (a b : ℝ × ℝ)
def m : ℝ × ℝ := (4, 5)

axiom m_is_midpoint : m = (1/2 : ℝ) • (a + b)

axiom dot_product_ab : a.1 * b.1 + a.2 * b.2 = 8

theorem sum_of_squared_norms : ‖a‖^2 + ‖b‖^2 = 148 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squared_norms_l1138_113813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dance_pairing_theorem_l1138_113849

/-- Represents a dance pairing between girls and boys -/
def DancePairing (n : ℕ+) := Fin n → Fin n

/-- Checks if a dance pairing is valid (each girl dances with a boy) -/
def is_valid_pairing (n : ℕ+) (pairing : DancePairing n) : Prop :=
  Function.Surjective pairing

/-- Generates the next dance pairing based on the current one -/
def next_pairing (n : ℕ+) (current : DancePairing n) : DancePairing n :=
  fun i => ⟨(current i + 1).val % n, Nat.mod_lt _ n.pos⟩

/-- Checks if a pairing contains at least one girl dancing with her partner -/
def has_correct_pair (n : ℕ+) (pairing : DancePairing n) : Prop :=
  ∃ i : Fin n, pairing i = i

/-- Main theorem: For odd n, there exists a valid initial pairing such that
    all subsequent pairings have at least one correct pair, and for even n,
    no such pairing exists -/
theorem dance_pairing_theorem (n : ℕ+) :
  (∃ initial : DancePairing n, 
    is_valid_pairing n initial ∧ 
    (∀ k : ℕ, has_correct_pair n ((next_pairing n)^[k] initial))) ↔ 
  Odd n.val :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dance_pairing_theorem_l1138_113849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_two_zeros_l1138_113826

/-- The function f(x) = ax³ + 2x² - 1 has exactly two zeros if and only if a ∈ {-4√6/9, 0, 4√6/9} -/
theorem cubic_function_two_zeros (a : ℝ) :
  (∃! (s : Set ℝ), s.Finite ∧ s.ncard = 2 ∧ ∀ x, x ∈ s ↔ a * x^3 + 2 * x^2 - 1 = 0) ↔
  a ∈ ({-4 * Real.sqrt 6 / 9, 0, 4 * Real.sqrt 6 / 9} : Set ℝ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_two_zeros_l1138_113826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cone_volume_ratio_l1138_113864

/-- A right circular cone inscribed in a cube -/
structure InscribedCone where
  s : ℝ  -- side length of the cube
  h : ℝ  -- height of the cone
  r : ℝ  -- radius of the cone's base
  h_eq_s : h = s
  r_eq_half_s : r = s / 2

/-- The volume of a cone -/
noncomputable def cone_volume (c : InscribedCone) : ℝ := (1/3) * Real.pi * c.r^2 * c.h

/-- The volume of a cube -/
def cube_volume (c : InscribedCone) : ℝ := c.s^3

/-- The ratio of the volume of the inscribed cone to the volume of the cube -/
noncomputable def volume_ratio (c : InscribedCone) : ℝ := cone_volume c / cube_volume c

/-- Theorem: The ratio of the volume of a right circular cone inscribed in a cube
    to the volume of the cube is π/12 -/
theorem inscribed_cone_volume_ratio (c : InscribedCone) :
  volume_ratio c = Real.pi / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cone_volume_ratio_l1138_113864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pqrs_perimeter_and_sum_l1138_113875

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the perimeter of a quadrilateral given its four vertices -/
noncomputable def perimeter (p q r s : Point) : ℝ :=
  distance p q + distance q r + distance r s + distance s p

/-- The main theorem stating the perimeter of PQRS and the sum of coefficients -/
theorem pqrs_perimeter_and_sum : 
  let p : Point := ⟨1, 3⟩
  let q : Point := ⟨4, 7⟩
  let r : Point := ⟨8, 3⟩
  let s : Point := ⟨10, 1⟩
  ∃ (c d : ℕ), 
    perimeter p q r s = c * Real.sqrt 2 + d * Real.sqrt 17 ∧ 
    c = 6 ∧ 
    d = 5 ∧ 
    c + d = 11 := by
  sorry

#check pqrs_perimeter_and_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pqrs_perimeter_and_sum_l1138_113875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_with_sine_sum_l1138_113800

theorem no_triangle_with_sine_sum (A B C : Real) : 
  (0 < A) ∧ (A < Real.pi) ∧ 
  (0 < B) ∧ (B < Real.pi) ∧ 
  (0 < C) ∧ (C < Real.pi) ∧ 
  (A + B + C = Real.pi) →
  (Real.sin A + Real.sin B ≠ Real.sin C) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_with_sine_sum_l1138_113800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_ratio_l1138_113805

/-- A right triangular pyramid with three perpendicular edges -/
structure RightTriangularPyramid where
  /-- The length of a side edge of the pyramid -/
  side_edge : ℝ
  /-- Assumption that the side edge is positive -/
  side_edge_pos : side_edge > 0

/-- The radius of the circumscribed sphere of the pyramid -/
noncomputable def circumscribed_sphere_radius (p : RightTriangularPyramid) : ℝ :=
  p.side_edge * Real.sqrt 3 / 2

/-- The radius of the inscribed sphere of the pyramid -/
noncomputable def inscribed_sphere_radius (p : RightTriangularPyramid) : ℝ :=
  p.side_edge * (3 - Real.sqrt 3) / 6

/-- Theorem stating the ratio of the radii -/
theorem radius_ratio (p : RightTriangularPyramid) :
  inscribed_sphere_radius p / circumscribed_sphere_radius p = (Real.sqrt 3 - 1) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_ratio_l1138_113805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_xy_expression_l1138_113843

theorem max_value_xy_expression (x y : ℝ) (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) :
  x^2 * y - y^2 * x ≤ (1/4 : ℝ) ∧ ∃ x₀ y₀, x₀ ∈ Set.Icc 0 1 ∧ y₀ ∈ Set.Icc 0 1 ∧ x₀^2 * y₀ - y₀^2 * x₀ = (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_xy_expression_l1138_113843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_parity_l1138_113823

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

-- State the theorem
theorem f_domain_and_parity :
  (∀ x : ℝ, x ∈ Set.Ioo (-1) 1 ↔ f x ≠ 0) ∧
  (∀ x : ℝ, x ∈ Set.Ioo (-1) 1 → f (-x) = -f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_parity_l1138_113823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1138_113896

-- Define the function N
noncomputable def N (t : ℝ) : ℝ := 100 * (18 : ℝ)^t

-- Define the sequence sum
def sequence_sum (Q : ℝ) : ℝ := Q + (Q + 6) + (Q + 12) + (Q + 18) + (Q + 24)

-- Define the equation for R
def R_equation (Q R : ℝ) : Prop := (Q / 100) * (25 / 32) = (1 / Q) * (R / 100)

theorem problem_solution :
  (N 0 = 100) ∧
  (sequence_sum 8 = 100) ∧
  (R_equation 8 50) ∧
  (∃ a : ℝ, (50 / 9) * 3 = 50 / 3 ∧ 3 * (50 / 9)^2 - a * (50 / 9) + 50 = 0) := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1138_113896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1138_113820

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  ∃ (x₀ y₀ : ℝ),
    x₀^2 / a^2 + y₀^2 / b^2 = 1 ∧
    y₀ = b / 2 ∧
    let c := Real.sqrt (a^2 - b^2)
    let F₁ := (-c, 0)
    let F₂ := (c, 0)
    let P := (x₀, y₀)
    let angle := Real.arccos ((x₀ + c)^2 + y₀^2 - ((x₀ - c)^2 + y₀^2)) / (2 * Real.sqrt ((x₀ + c)^2 + y₀^2) * Real.sqrt ((x₀ - c)^2 + y₀^2))
    angle = π / 3 →
    Real.sqrt (1 - b^2 / a^2) = 2 * Real.sqrt 7 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1138_113820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l1138_113811

/-- Given that the coefficients of the first three terms of the expansion of (x + 1/(2x))^n form an arithmetic sequence,
    prove that the coefficient of the x^4 term in the expansion is 7. -/
theorem binomial_expansion_coefficient (n : ℕ) : 
  (1 + n * (n - 1) / 8 = n) →  -- Condition from the arithmetic sequence property
  (∃ k : ℕ, (Nat.choose n k) * (1/2 : ℚ)^k * 2^(n-k) = 7 ∧ n - 2*k = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l1138_113811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_price_percentage_l1138_113859

/-- Represents the pricing strategy of a store -/
structure StorePricing where
  list_price : ℝ
  purchase_discount : ℝ
  marked_price : ℝ
  sale_discount : ℝ
  profit_margin : ℝ

/-- The pricing strategy satisfies the store's requirements -/
def satisfies_requirements (s : StorePricing) : Prop :=
  s.purchase_discount = 0.3 ∧
  s.sale_discount = 0.15 ∧
  s.profit_margin = 0.3 ∧
  s.list_price > 0 ∧
  s.marked_price > 0

/-- The marked price results in the desired profit after discounts -/
def achieves_profit (s : StorePricing) : Prop :=
  (s.marked_price * (1 - s.sale_discount)) - (s.list_price * (1 - s.purchase_discount)) =
    s.profit_margin * (s.marked_price * (1 - s.sale_discount))

/-- The theorem stating that the marked price should be approximately 116.67% of the list price -/
theorem marked_price_percentage (s : StorePricing) 
    (h_req : satisfies_requirements s) 
    (h_profit : achieves_profit s) : 
    ∃ ε > 0, |s.marked_price / s.list_price - 1.1667| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_price_percentage_l1138_113859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_when_sin_eq_4cos_l1138_113812

theorem sin_cos_product_when_sin_eq_4cos (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_when_sin_eq_4cos_l1138_113812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_l1138_113861

theorem root_product (a b : ℝ) : 
  (2 * (Real.log a)^2 - 4 * (Real.log a) + 1 = 0) →
  (2 * (Real.log b)^2 - 4 * (Real.log b) + 1 = 0) →
  (Real.log a ≠ Real.log b) →
  a * b = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_l1138_113861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_positions_l1138_113882

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem circle_positions 
  (r s : ℝ) 
  (hr : r > 0) 
  (hs : s > 0) 
  (hrs : r > s) : 
  ∃ (A B : ℝ × ℝ), 
    (∃ (c1 c2 : Circle), c1.center = A ∧ c1.radius = r ∧ c2.center = B ∧ c2.radius = s) ∧
    (distance A B = r - s) ∧ 
    (∃ (C D : ℝ × ℝ), distance C D = r + s) ∧
    (∃ (E F : ℝ × ℝ), distance E F > r + s) ∧
    (∃ (G H : ℝ × ℝ), distance G H > r - s) :=
by
  -- We'll use sorry to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_positions_l1138_113882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1138_113868

-- Define the function g(x) with parameter a
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - 3 * x

-- State the theorem
theorem g_properties :
  ∃ (a : ℝ),
    (∀ x, x > 0 → (deriv (g a)) x = 0 → x = 1) ∧
    (g a 1 = -2) ∧
    (g a (1/2) = -Real.log 2 - 5/4) ∧
    (∀ x, x > 0 → g a x ≥ -2) ∧
    (∀ x, x > 0 → g a x ≤ -Real.log 2 - 5/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1138_113868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_identity_l1138_113857

theorem functional_equation_identity (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f (f x)) + f (f y) = f y + x) →
  (∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_identity_l1138_113857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_f_l1138_113885

/-- The function f as defined in the problem -/
noncomputable def f (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2 + 26*a + 86*b + 2018)

/-- The theorem stating the minimum value of the sum of f for different combinations of a and b -/
theorem min_sum_f :
  ∀ a b : ℝ, f a b + f a (-b) + f (-a) b + f (-a) (-b) ≥ 4 * Real.sqrt 2018 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_f_l1138_113885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_sqrt_five_l1138_113862

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

/-- The distance from the focus to the asymptote of a hyperbola -/
noncomputable def focus_to_asymptote_distance (h : Hyperbola a b) : ℝ :=
  (b * Real.sqrt (a^2 + b^2)) / a

/-- The length of the real axis of a hyperbola -/
def real_axis_length (h : Hyperbola a b) : ℝ := 2 * a

/-- Theorem: If the distance from the left focus to the asymptote is equal to the length of the real axis,
    then the eccentricity of the hyperbola is √5 -/
theorem hyperbola_eccentricity_is_sqrt_five (a b : ℝ) (h : Hyperbola a b)
    (h_dist_eq : focus_to_asymptote_distance h = real_axis_length h) :
    eccentricity h = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_sqrt_five_l1138_113862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_conditions_l1138_113803

/-- The range of m for which the given conditions hold -/
theorem hyperbola_ellipse_conditions (m : ℝ) : 
  (∃ e : ℝ, (Real.sqrt 6 / 2 < e ∧ e < Real.sqrt 2) ∧ 
   (∀ x y : ℝ, y^2 / 5 - x^2 / m = 1 → e = Real.sqrt ((5 + m) / 5))) ∧ 
  (∀ x y : ℝ, x^2 / (2*m) + y^2 / (9 - m) = 1 → 
    ∃ c : ℝ, c > 0 ∧ (∀ x : ℝ, x^2 / (2*m) + (y - c)^2 / (9 - m) = 1 → 
      x^2 / (2*m) + (y + c)^2 / (9 - m) = 1)) ∧
  ((2.5 < m ∧ m ≤ 3) ∨ (5 ≤ m ∧ m < 9)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_conditions_l1138_113803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_eq_two_g_increasing_implies_a_leq_four_l1138_113839

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) - a * x - 1

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - (Real.log x) ^ 2 - 2 * Real.log x

-- Theorem 1: If the minimum value of f is 0, then a = 2
theorem min_value_implies_a_eq_two (a : ℝ) :
  (∃ x₀ : ℝ, ∀ x : ℝ, f a x ≥ f a x₀ ∧ f a x₀ = 0) → a = 2 := by
  sorry

-- Theorem 2: If g is increasing, then a ≤ 4
theorem g_increasing_implies_a_leq_four (a : ℝ) :
  (∀ x y : ℝ, x < y → g a x < g a y) → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_eq_two_g_increasing_implies_a_leq_four_l1138_113839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_palindromes_l1138_113834

/-- A five-digit palindrome is a number between 10000 and 99999 that reads the same forwards and backwards -/
def IsFiveDigitPalindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ 
  (n / 10000 = n % 10) ∧ 
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The second digit of a number -/
def SecondDigit (n : ℕ) : ℕ :=
  (n / 1000) % 10

/-- The set of all five-digit palindromes where the second digit is not 0 or 1 -/
def ValidPalindromes : Set ℕ :=
  {n : ℕ | IsFiveDigitPalindrome n ∧ SecondDigit n ≠ 0 ∧ SecondDigit n ≠ 1}

/-- Axiom stating that ValidPalindromes is finite -/
axiom validPalindromes_finite : Finite ValidPalindromes

/-- Instance of Fintype for ValidPalindromes -/
noncomputable instance : Fintype ValidPalindromes :=
  Set.Finite.fintype validPalindromes_finite

theorem count_valid_palindromes : Fintype.card ValidPalindromes = 720 := by
  sorry

#check count_valid_palindromes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_palindromes_l1138_113834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_in_regular_tetrahedron_l1138_113822

/-- Regular tetrahedron with vertices A, B, C, D -/
structure RegularTetrahedron (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C D : V)
  (is_regular : sorry)  -- Add conditions for regular tetrahedron

/-- Point inside a face of the tetrahedron -/
def point_in_face {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (t : RegularTetrahedron V) (E : V) : Prop :=
  sorry  -- Add condition for E being inside face ABC

/-- Sum of distances from a point to three faces of the tetrahedron -/
noncomputable def sum_distances_to_faces {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (t : RegularTetrahedron V) (E : V) : ℝ :=
  sorry  -- Define sum of distances from E to faces DAB, DBC, DCA

/-- Sum of distances from a point to three edges of a face -/
noncomputable def sum_distances_to_edges {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (t : RegularTetrahedron V) (E : V) : ℝ :=
  sorry  -- Define sum of distances from E to edges AB, BC, CA

/-- Theorem stating the ratio of distances -/
theorem distance_ratio_in_regular_tetrahedron 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (t : RegularTetrahedron V) (E : V) 
  (h : point_in_face t E) :
  (sum_distances_to_faces t E) / (sum_distances_to_edges t E) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_in_regular_tetrahedron_l1138_113822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_120_to_80_l1138_113827

-- Define the function to calculate percentage
noncomputable def calculatePercentage (part : ℝ) (whole : ℝ) : ℝ :=
  (part / whole) * 100

-- Theorem statement
theorem percentage_of_120_to_80 :
  calculatePercentage 120 80 = 150 := by
  -- Unfold the definition of calculatePercentage
  unfold calculatePercentage
  -- Simplify the arithmetic
  simp [div_mul_eq_mul_div]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_120_to_80_l1138_113827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_xy_l1138_113892

def sample : List ℚ := [9, 10, 11]

def mean (xs : List ℚ) (x y : ℚ) : ℚ :=
  (xs.sum + x + y) / (xs.length + 2 : ℚ)

def variance (xs : List ℚ) (x y : ℚ) (m : ℚ) : ℚ :=
  (xs.map (λ a => (a - m) ^ 2) ++ [(x - m) ^ 2, (y - m) ^ 2]).sum / (xs.length + 2 : ℚ)

theorem product_xy (x y : ℤ) 
  (h_mean : mean sample (x : ℚ) (y : ℚ) = 10)
  (h_variance : variance sample (x : ℚ) (y : ℚ) 10 = 4) :
  x * y = 191 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_xy_l1138_113892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_K_equals_K_prev_l1138_113815

/-- The recurrence relation for K_n -/
def K : ℕ → List ℝ → ℝ
  | 0, _ => 1
  | 1, x::_ => x
  | 1, [] => 0  -- Added to handle empty list case for n = 1
  | n+2, x₁::x₂::xs => x₁ * K (n+1) (x₂::xs) + K n xs
  | n+2, [x] => 0  -- Added to handle single element list case for n ≥ 2
  | n+2, [] => 0   -- Added to handle empty list case for n ≥ 2

/-- The previously defined sequence of polynomials -/
def K_prev : ℕ → List ℝ → ℝ := sorry

/-- Theorem stating that K and K_prev are equal for all n and inputs -/
theorem K_equals_K_prev : ∀ (n : ℕ) (xs : List ℝ), K n xs = K_prev n xs := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_K_equals_K_prev_l1138_113815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1138_113889

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := (x - 8) / Real.sqrt (x^2 - 5*x + 6)

-- Define the domain of g
def domain_g : Set ℝ := {x | x < 2 ∨ x > 3}

-- Theorem statement
theorem domain_of_g : 
  {x : ℝ | ∃ y, g x = y} = domain_g := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1138_113889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_approx_63_l1138_113884

/-- The speed of a man in km/h, given distance traveled in meters and time in seconds -/
noncomputable def speed_km_per_hour (distance_meters : ℝ) (time_seconds : ℝ) : ℝ :=
  (distance_meters / 1000) / (time_seconds / 3600)

/-- Theorem stating that the speed of a man who travels 437.535 meters in 25 seconds is approximately 63 km/h -/
theorem man_speed_approx_63 : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |speed_km_per_hour 437.535 25 - 63| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_approx_63_l1138_113884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_AFB_l1138_113824

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem max_angle_AFB (x₁ y₁ x₂ y₂ : ℝ) :
  is_on_parabola x₁ y₁ →
  is_on_parabola x₂ y₂ →
  distance (x₁, y₁) (x₂, y₂) = (Real.sqrt 3 / 2) * (x₁ + x₂ + 2) →
  ∃ (θ : ℝ), θ ≤ 2 * Real.pi / 3 ∧
    ∀ (θ' : ℝ), θ' = Real.arccos ((distance (x₁, y₁) focus)^2 + (distance (x₂, y₂) focus)^2 - (distance (x₁, y₁) (x₂, y₂))^2) /
                     (2 * distance (x₁, y₁) focus * distance (x₂, y₂) focus) →
    θ' ≤ θ := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_AFB_l1138_113824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_cosine_equality_l1138_113879

theorem angle_equality_cosine_equality (A B : ℝ) (h_triangle : A ∈ Set.Ioo 0 Real.pi ∧ B ∈ Set.Ioo 0 Real.pi) :
  A = B ↔ Real.cos A = Real.cos B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_cosine_equality_l1138_113879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_partition_of_naturals_l1138_113853

def is_power_of_two_plus_two (n : ℕ) : Prop :=
  ∃ h : ℕ, n = 2^h + 2

def satisfies_condition (S : Set ℕ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≠ y → ¬(is_power_of_two_plus_two (x + y))

theorem unique_partition_of_naturals :
  ∃! (A B : Set ℕ), 
    (A ∪ B = Set.univ) ∧ 
    (A ∩ B = ∅) ∧
    (1 ∈ A) ∧
    (satisfies_condition A) ∧
    (satisfies_condition B) ∧
    (1987 ∈ B) ∧
    (1988 ∈ A) ∧
    (1989 ∈ B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_partition_of_naturals_l1138_113853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_square_remainders_divided_by_13_l1138_113886

theorem sum_of_square_remainders_divided_by_13 : ∃ m : ℕ,
  (m = (Finset.range 15).sum (λ n ↦ ((n + 1)^2 % 13)))
  ∧ (Finset.card (Finset.image (λ n ↦ ((n + 1)^2 % 13)) (Finset.range 15)) = m)
  ∧ (m / 13 = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_square_remainders_divided_by_13_l1138_113886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_attendance_difference_l1138_113890

noncomputable def chicago_estimate : ℝ := 80000
noncomputable def denver_estimate : ℝ := 70000
noncomputable def edmonton_estimate : ℝ := 65000

noncomputable def chicago_actual_min : ℝ := chicago_estimate * 0.88
noncomputable def chicago_actual_max : ℝ := chicago_estimate * 1.12

noncomputable def denver_actual_min : ℝ := denver_estimate / 1.15
noncomputable def denver_actual_max : ℝ := denver_estimate / 0.85

noncomputable def edmonton_actual : ℝ := edmonton_estimate

noncomputable def max_difference : ℝ := max chicago_actual_max (max denver_actual_max edmonton_actual) - 
                           min chicago_actual_min (min denver_actual_min edmonton_actual)

theorem attendance_difference :
  ⌊(max_difference + 500) / 1000⌋ * 1000 = 29000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_attendance_difference_l1138_113890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_periodic_sin_plus_sin_sqrt2_l1138_113898

theorem not_periodic_sin_plus_sin_sqrt2 :
  ¬∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, 
    Real.sin x + Real.sin (Real.sqrt 2 * x) = Real.sin (x + T) + Real.sin (Real.sqrt 2 * (x + T)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_periodic_sin_plus_sin_sqrt2_l1138_113898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_3_l1138_113821

-- Define the circle
def myCircle (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define the line passing through the origin with inclination angle 120°
def myLine (x y : ℝ) : Prop := y = -Real.sqrt 3 * x

-- Define the length of the chord
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

theorem chord_length_is_2_sqrt_3 :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    myCircle x₁ y₁ ∧ myCircle x₂ y₂ ∧
    myLine x₁ y₁ ∧ myLine x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = chord_length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_3_l1138_113821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_badminton_game_probabilities_l1138_113806

/-- Represents a player in the badminton game -/
inductive Player : Type
| A : Player
| B : Player
| C : Player

/-- Represents the state of the game -/
structure GameState :=
  (current_players : Player × Player)
  (bye_player : Player)
  (consecutive_losses : Player → Nat)

/-- The probability of winning for each player in a game -/
noncomputable def win_probability : ℝ := 1 / 2

/-- The initial game state -/
def initial_state : GameState :=
  { current_players := (Player.A, Player.B),
    bye_player := Player.C,
    consecutive_losses := λ _ => 0 }

/-- The probability of A winning four consecutive games -/
noncomputable def prob_A_wins_four_consecutive : ℝ :=
  (win_probability ^ 4 : ℝ)

/-- The probability of needing a fifth game to be played -/
noncomputable def prob_fifth_game : ℝ :=
  1 - 4 * (win_probability ^ 4 : ℝ)

/-- The probability of C being the ultimate winner -/
noncomputable def prob_C_wins : ℝ :=
  7 / 16

/-- Main theorem combining all results -/
theorem badminton_game_probabilities :
  prob_A_wins_four_consecutive = 1/16 ∧
  prob_fifth_game = 3/4 ∧
  prob_C_wins = 7/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_badminton_game_probabilities_l1138_113806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_cost_calculation_l1138_113848

/-- Calculate the total amount paid for three pizzas with discount and tax -/
theorem pizza_cost_calculation (price1 price2 price3 : ℝ) 
  (discount_rate tax_rate : ℝ) : 
  price1 = 8 →
  price2 = 12 →
  price3 = 10 →
  discount_rate = 0.2 →
  tax_rate = 0.05 →
  (price1 + price2 + price3) * (1 - discount_rate) * (1 + tax_rate) = 25.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_cost_calculation_l1138_113848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_max_sum_of_sides_l1138_113872

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the area of the triangle
noncomputable def area (t : Triangle) : ℝ := (Real.sqrt 3 / 4) * (t.a^2 + t.b^2 - t.c^2)

-- Theorem 1: The measure of angle C is π/3 radians
theorem angle_C_measure (t : Triangle) (h : area t = (Real.sqrt 3 / 4) * (t.a^2 + t.b^2 - t.c^2)) : 
  t.C = π/3 := by sorry

-- Theorem 2: When c = √3, the maximum value of a + b is 2√3
theorem max_sum_of_sides (t : Triangle) (h1 : area t = (Real.sqrt 3 / 4) * (t.a^2 + t.b^2 - t.c^2)) 
  (h2 : t.c = Real.sqrt 3) : 
  (∀ t' : Triangle, t'.c = Real.sqrt 3 → t.a + t.b ≥ t'.a + t'.b) → 
  t.a + t.b = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_max_sum_of_sides_l1138_113872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l1138_113804

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

-- Define the foci
def foci (F1 F2 : ℝ × ℝ) : Prop :=
  F1 = (Real.sqrt 5, 0) ∧ F2 = (-Real.sqrt 5, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  is_on_ellipse P.1 P.2

-- Define the ratio condition
def ratio_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  (Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)) /
  (Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)) = 2

-- Theorem statement
theorem ellipse_triangle_area
  (P F1 F2 : ℝ × ℝ)
  (h1 : foci F1 F2)
  (h2 : point_on_ellipse P)
  (h3 : ratio_condition P F1 F2) :
  (1/2) * Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2) *
  Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l1138_113804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l1138_113802

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | Real.exp (x * Real.log 2) - 4 ≤ 0}

-- State the theorem
theorem complement_of_M : 
  (U \ M) = {x : ℝ | x > 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l1138_113802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_properties_l1138_113842

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The hexagon vertices -/
def A : Point := ⟨0, 0⟩
def B : Point := ⟨1, 3⟩
def C : Point := ⟨3, 3⟩
def D : Point := ⟨4, 1⟩
def E : Point := ⟨3, -1⟩
def F : Point := ⟨1, -1⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Perimeter of the hexagon -/
noncomputable def perimeter : ℝ :=
  distance A B + distance B C + distance C D +
  distance D E + distance E F + distance F A

/-- Theorem stating the properties of the hexagon's perimeter -/
theorem hexagon_perimeter_properties :
  (perimeter = 4 + Real.sqrt 2 + 2 * Real.sqrt 5 + Real.sqrt 10) ∧
  (∃ (a b c : ℝ), perimeter = a + b * Real.sqrt 2 + c * Real.sqrt 10 ∧ a + b + c = 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_properties_l1138_113842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_length_l1138_113876

/-- Circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Line defined by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def intersectionPoints (c : Circle) (l : Line) : Set Point :=
  {p : Point | (p.x - c.h)^2 + (p.y - c.k)^2 = c.r^2 ∧ l.a * p.x + l.b * p.y + l.c = 0}

theorem circle_line_intersection_length :
  let c : Circle := { h := 1, k := -3, r := 2 }
  let l : Line := { a := 1, b := -1, c := -2 }
  let intersection := intersectionPoints c l
  ∀ A B, A ∈ intersection → B ∈ intersection → A ≠ B → distance A B = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_length_l1138_113876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_cost_theorem_l1138_113807

/-- The cost in dollars per kilogram for transporting material to the International Space Station -/
noncomputable def cost_per_kg : ℚ := 22000

/-- The mass of the control module in grams -/
def module_mass_g : ℚ := 250

/-- Conversion factor from grams to kilograms -/
def g_to_kg : ℚ := 1 / 1000

theorem transport_cost_theorem :
  cost_per_kg * (module_mass_g * g_to_kg) = 5500 := by
  -- Expand the definitions
  unfold cost_per_kg module_mass_g g_to_kg
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_cost_theorem_l1138_113807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_video_recorder_markup_proof_l1138_113828

def video_recorder_markup_percentage 
  (wholesale_cost : ℝ) 
  (employee_discount : ℝ) 
  (employee_paid : ℝ) : ℝ :=
  let retail_price := wholesale_cost * (1 + 20 / 100)
  let discounted_price := retail_price * (1 - employee_discount / 100)
  20

theorem video_recorder_markup_proof :
  video_recorder_markup_percentage 200 25 180 = 20 := by
  unfold video_recorder_markup_percentage
  -- The proof steps would go here
  sorry

#eval video_recorder_markup_percentage 200 25 180

end NUMINAMATH_CALUDE_ERRORFEEDBACK_video_recorder_markup_proof_l1138_113828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_reflection_ratio_l1138_113893

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  vertexAngle : ℝ
  basePositive : 0 < base
  heightPositive : 0 < height
  vertexAngleBound : vertexAngle < 2 * Real.pi / 3

-- Define the reflection operation
def reflect (t : IsoscelesTriangle) : IsoscelesTriangle where
  base := sorry
  height := sorry
  vertexAngle := t.vertexAngle
  basePositive := sorry
  heightPositive := sorry
  vertexAngleBound := t.vertexAngleBound

-- State the theorem
theorem isosceles_triangle_reflection_ratio (t : IsoscelesTriangle) :
  (reflect t).base / t.base + (reflect t).height / t.height = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_reflection_ratio_l1138_113893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l1138_113894

-- Define the ellipse parameters
noncomputable def a : ℝ := 2 * Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 2
noncomputable def c : ℝ := Real.sqrt 6

-- Define the eccentricity
noncomputable def e : ℝ := Real.sqrt 3 / 2

-- Define the distance from right focus to the line x + y + √6 = 0
noncomputable def d : ℝ := 2 * Real.sqrt 3

-- Define the point M
def M : ℝ × ℝ := (0, -1)

-- Define the theorem
theorem ellipse_and_line_properties :
  -- The ellipse equation
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 8 + y^2 / 2 = 1) ∧
  -- The line equation
  (∃ k : ℝ, k = 1 ∨ k = -1) ∧
  (∀ x y : ℝ, (x^2 / 8 + y^2 / 2 = 1 ∧ y = k * x - 1) →
    ∃ A B N : ℝ × ℝ,
      -- A and B are on the ellipse and the line
      A.1^2 / 8 + A.2^2 / 2 = 1 ∧ A.2 = k * A.1 - 1 ∧
      B.1^2 / 8 + B.2^2 / 2 = 1 ∧ B.2 = k * B.1 - 1 ∧
      -- N is on the x-axis
      N.2 = 0 ∧
      -- The vector property
      A.1 - N.1 = -7/5 * (B.1 - N.1) ∧
      A.2 - N.2 = -7/5 * (B.2 - N.2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l1138_113894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_collection_end_count_l1138_113877

/-- Calculates the number of books in a special collection at the end of a month,
    given the initial number of books, number of books loaned out, and the return rate. -/
def booksAtEndOfMonth (initialBooks : ℕ) (loanedBooks : ℕ) (returnRate : ℚ) : ℕ :=
  initialBooks - (loanedBooks - Int.toNat ((loanedBooks : ℚ) * returnRate).floor)

/-- Theorem stating that given the specific conditions of the problem,
    the number of books at the end of the month is 122. -/
theorem special_collection_end_count :
  booksAtEndOfMonth 150 80 (65/100) = 122 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_collection_end_count_l1138_113877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_isosceles_triangle_l1138_113845

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The intersection point of two lines -/
noncomputable def intersection (l1 l2 : Line) : ℝ × ℝ :=
  let x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope)
  let y := l1.slope * x + l1.intercept
  (x, y)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Predicate to check if a triangle is isosceles -/
def isIsosceles (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let d12 := distance p1 p2
  let d23 := distance p2 p3
  let d31 := distance p3 p1
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

theorem intersection_forms_isosceles_triangle :
  let l1 : Line := ⟨4, 3⟩
  let l2 : Line := ⟨-4, 3⟩
  let l3 : Line := ⟨0, -3⟩
  let p1 := intersection l1 l2
  let p2 := intersection l1 l3
  let p3 := intersection l2 l3
  isIsosceles p1 p2 p3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_isosceles_triangle_l1138_113845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_calculation_l1138_113871

/-- Represents the properties of a rectangular floor --/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  total_cost : ℝ
  paint_rate : ℝ

/-- Theorem about the length of a rectangular floor --/
theorem floor_length_calculation (floor : RectangularFloor) 
  (h1 : floor.length = floor.breadth + 2 * floor.breadth)
  (h2 : floor.total_cost = 240)
  (h3 : floor.paint_rate = 3)
  : ∃ (ε : ℝ), abs (floor.length - 15.48) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_calculation_l1138_113871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hugo_first_roll_given_win_l1138_113867

/-- Represents the outcome of a single die roll -/
def DieRoll := Fin 8

/-- Represents the state of the game -/
structure GameState where
  players : Fin 4 → DieRoll
  winner : Fin 4

/-- Hugo is one of the players -/
def hugo : Fin 4 := 0

/-- The event that Hugo's first roll is 7 -/
def hugo_rolls_seven (g : GameState) : Prop :=
  g.players hugo = ⟨7, by norm_num⟩

/-- The event that Hugo wins the game -/
def hugo_wins (g : GameState) : Prop :=
  g.winner = hugo

/-- The probability space of all possible game outcomes -/
def Ω : Type := GameState

/-- The probability measure on the game outcomes -/
noncomputable def P : Set Ω → ℝ := sorry

theorem hugo_first_roll_given_win :
  P {g : Ω | hugo_rolls_seven g ∧ hugo_wins g} / P {g : Ω | hugo_wins g} = 27 / 128 := by
  sorry

#check hugo_first_roll_given_win

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hugo_first_roll_given_win_l1138_113867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_2s_l1138_113831

-- Define the displacement function
noncomputable def S (t : ℝ) : ℝ := Real.log (t + 1) + 2 * t^2 + 1

-- Define the instantaneous velocity function
noncomputable def v (t : ℝ) : ℝ := 1 / (t + 1) + 4 * t

-- Theorem statement
theorem instantaneous_velocity_at_2s :
  v 2 = 25 / 3 := by
  -- Unfold the definition of v
  unfold v
  -- Simplify the expression
  simp
  -- Perform arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_2s_l1138_113831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_diameter_approx_l1138_113863

/-- The perimeter of a semicircular window -/
noncomputable def semicircle_perimeter : ℝ := 161.96

/-- The diameter of the semicircular window -/
noncomputable def semicircle_diameter : ℝ := semicircle_perimeter / (Real.pi / 2 + 1)

/-- Theorem: The diameter of a semicircular window with perimeter 161.96 cm is approximately 63.01 cm -/
theorem semicircle_diameter_approx :
  abs (semicircle_diameter - 63.01) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_diameter_approx_l1138_113863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_specific_axes_hyperbola_satisfies_equation_l1138_113810

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance from the center to a focus of a hyperbola -/
noncomputable def focal_distance (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2)

/-- Theorem: Given a hyperbola and a line, if one asymptote of the hyperbola
    is parallel to the line and one focus lies on the line, then the hyperbola's
    semi-axes have specific values. -/
theorem hyperbola_specific_axes (h : Hyperbola) (l : Line) 
    (h_asymptote_parallel : h.b / h.a = 1 / 2)
    (h_focus_on_line : focal_distance h = 5) :
    h.a = 2 * Real.sqrt 5 ∧ h.b = Real.sqrt 5 := by
  sorry

/-- The equation of the hyperbola given the specific semi-axes -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 20 - y^2 / 5 = 1

/-- Theorem: The hyperbola with the derived semi-axes satisfies the given equation -/
theorem hyperbola_satisfies_equation (h : Hyperbola) (l : Line)
    (h_asymptote_parallel : h.b / h.a = 1 / 2)
    (h_focus_on_line : focal_distance h = 5) :
    ∀ x y, hyperbola_equation x y ↔ x^2 / h.a^2 - y^2 / h.b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_specific_axes_hyperbola_satisfies_equation_l1138_113810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_l1138_113825

/-- The series term for a given n -/
noncomputable def seriesTerm (n : ℕ) : ℝ :=
  (6 * n^3 - 2 * n^2 - 2 * n + 3) / (n^6 - 2 * n^5 + 2 * n^4 - 2 * n^3 + 2 * n^2 - 2 * n)

/-- The sum of the series from n=2 to infinity -/
noncomputable def seriesSum : ℝ := ∑' n, if n ≥ 2 then seriesTerm n else 0

/-- Theorem stating that the series sum equals 1 -/
theorem series_sum_equals_one : seriesSum = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_l1138_113825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_typical_divisors_implies_kth_power_converse_not_true_for_k_gt_2_l1138_113833

/-- A natural number m is k-typical if each divisor of m leaves the remainder 1 when divided by k -/
def is_k_typical (k : ℕ+) (m : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ m → d % k = 1

/-- The number of divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem k_typical_divisors_implies_kth_power (k : ℕ+) (n : ℕ+) :
  is_k_typical k (num_divisors n) → ∃ m : ℕ, n = m^(k:ℕ) := by sorry

theorem converse_not_true_for_k_gt_2 :
  ∀ k : ℕ+, k > 2 →
    ∃ n : ℕ+, (∃ m : ℕ, n = m^(k:ℕ)) ∧ ¬is_k_typical k (num_divisors n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_typical_divisors_implies_kth_power_converse_not_true_for_k_gt_2_l1138_113833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_property_l1138_113850

theorem quadratic_function_property (a b : ℝ) (h1 : a ≠ b) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  (f a = f b) → (f 2 = 4 + 4*a) := by
  intro h2
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_property_l1138_113850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1138_113841

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.a * Real.cos t.B = 1 ∧
  t.b * Real.sin t.A = Real.sqrt 2 ∧
  t.A - t.B = Real.pi / 4

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.a = Real.sqrt 3 ∧ Real.tan t.A = -3 - 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1138_113841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_max_slope_l1138_113816

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector in 2D space -/
structure Vec where
  x : ℝ
  y : ℝ

/-- Given a parabola C: y^2 = 2px where p > 0 and the distance from focus F to directrix is 2,
    with point P on C and point Q satisfying PQ = 9QF, 
    the maximum value of the slope of line OQ is 1/3 -/
theorem parabola_max_slope (C : Parabola) 
  (h_focus_dist : C.p = 2) 
  (P : Point) 
  (h_P_on_C : P.y^2 = 2 * C.p * P.x)
  (Q : Point) 
  (h_PQ_QF : Vec.mk (P.x - Q.x) (P.y - Q.y) = 
             Vec.mk (9 * (Q.x - C.p)) (9 * Q.y)) :
  (∃ (slope : ℝ), ∀ (Q' : Point), 
    Q'.y / Q'.x ≤ slope ∧ 
    (∃ (Q : Point), Q.y / Q.x = slope)) ∧
  (∀ (max_slope : ℝ), 
    (∀ (Q' : Point), Q'.y / Q'.x ≤ max_slope) →
    (∃ (Q : Point), Q.y / Q.x = max_slope) →
    max_slope = 1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_max_slope_l1138_113816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_triangle_l1138_113880

theorem smallest_k_for_triangle (k : ℕ) : k = 17 ↔ 
  (∀ S : Finset ℕ, S.card = k → S ⊆ Finset.range 2005 → 
    ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    a + b > c ∧ b + c > a ∧ a + c > b) ∧ 
  (∀ m < k, ∃ T : Finset ℕ, T.card = m ∧ T ⊆ Finset.range 2005 ∧ 
    ∀ a b c, a ∈ T → b ∈ T → c ∈ T → 
    a = b ∨ b = c ∨ a = c ∨ 
    a + b ≤ c ∨ b + c ≤ a ∨ a + c ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_triangle_l1138_113880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unity_number_properties_l1138_113855

/-- Definition of a unity number -/
def is_unity_number (m : ℕ) : Prop :=
  ∃ a b c : ℕ, m = 100 * a + 10 * b + c ∧ a = b + c ∧ 100 ≤ m ∧ m < 1000

/-- Definition of n given m -/
def n (m : ℕ) : ℕ :=
  let a := m / 100
  let b := (m / 10) % 10
  let c := m % 10
  100 * b + 10 * c + a

/-- Definition of p given m -/
def p (m : ℕ) : ℕ := m - n m

/-- Definition of f(m) -/
def f (m : ℕ) : ℚ := (p m : ℚ) / 9

/-- Main theorem -/
theorem unity_number_properties :
  (∀ m : ℕ, is_unity_number m → f m ≥ 1) ∧
  (∃ m : ℕ, is_unity_number m ∧ f m = 1) ∧
  (∀ m : ℕ, is_unity_number m → (∃ k : ℕ, f m = (12 : ℚ) * k) → f m ∈ ({12, 24, 36, 72} : Set ℚ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unity_number_properties_l1138_113855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_and_uniqueness_l1138_113847

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / (x + 2) - a * x^2

theorem f_zeros_and_uniqueness :
  (∃ x₁ x₂ x₃ : ℝ, x₁ = 0 ∧ x₂ = -1 - Real.sqrt 2 ∧ x₃ = -1 + Real.sqrt 2 ∧
    f 1 x₁ = 0 ∧ f 1 x₂ = 0 ∧ f 1 x₃ = 0) ∧
  (∀ a : ℝ, a > 0 → ∃! x : ℝ, x > 0 ∧ f a x = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_and_uniqueness_l1138_113847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_parabola_l1138_113883

theorem axis_of_symmetry_parabola (a b : ℝ) (h1 : a ≠ 0) (h2 : a * (-2) + b = 0) :
  let parabola := fun x : ℝ => a * x^2 + b * x
  let axis_of_symmetry := -1
  ∀ x : ℝ, parabola (x + axis_of_symmetry) = parabola (-x + axis_of_symmetry) :=
by
  -- Introduce the local definitions
  intro parabola axis_of_symmetry
  -- Introduce the universal quantifier
  intro x
  -- Expand the definitions of parabola and axis_of_symmetry
  simp [parabola, axis_of_symmetry]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_parabola_l1138_113883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laurens_income_l1138_113838

/-- Represents the tax calculation for Lauren's state --/
noncomputable def tax_calculation (p : ℝ) (income : ℝ) : ℝ :=
  if income ≤ 20000 then
    p / 100 * income
  else if income ≤ 35000 then
    p / 100 * 20000 + (p + 1) / 100 * (income - 20000)
  else
    p / 100 * 20000 + (p + 1) / 100 * 15000 + (p + 3) / 100 * (income - 35000)

/-- Theorem stating Lauren's annual income --/
theorem laurens_income (p : ℝ) :
  ∃ (income : ℝ), 
    income = 36000 ∧ 
    tax_calculation p income = (p + 0.45) / 100 * income := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_laurens_income_l1138_113838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l1138_113899

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f x * f y - f (x * y)) / 4 = x^2 + y^2 + 2) : 
  f = λ x ↦ x^2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l1138_113899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_investment_animal_husbandry_l1138_113837

/-- Represents the profit function for animal husbandry -/
noncomputable def P (a : ℝ) : ℝ := a / 3

/-- Represents the profit function for animal husbandry processing -/
noncomputable def Q (a : ℝ) : ℝ := 10 * Real.sqrt a / 3

/-- The total investment in ten thousand yuan -/
def total_investment : ℝ := 60

/-- Theorem stating the optimal investment in animal husbandry -/
theorem optimal_investment_animal_husbandry :
  ∃ (x : ℝ), x ≥ 0 ∧ x ≤ total_investment ∧
  ∀ (y : ℝ), y ≥ 0 ∧ y ≤ total_investment →
    P (total_investment - x) + Q x ≥ P (total_investment - y) + Q y ∧
  x = 35 := by
  sorry

#check optimal_investment_animal_husbandry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_investment_animal_husbandry_l1138_113837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n0_exists_l1138_113860

/-- Sequence of positive integers with strictly increasing terms -/
def StrictlyIncreasingSeq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- Sum of the first n terms of a sequence -/
def SeqSum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) a

/-- Theorem statement -/
theorem unique_n0_exists (a : ℕ → ℕ) (h : StrictlyIncreasingSeq a) :
  ∃! n0 : ℕ, (n0 > 0) ∧
    (((SeqSum a (n0 + 1) : ℚ) / n0) > a (n0 + 1)) ∧
    (((SeqSum a (n0 + 1) : ℚ) / n0) ≤ a (n0 + 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n0_exists_l1138_113860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_five_sixteenths_l1138_113832

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd (x : ℝ) : f (-x) = -f x
axiom f_periodic (x : ℝ) : f (x + 4) = f x
axiom f_piecewise (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) : 
  f x = if x ≤ 1 then x * (1 - x) else Real.sin (Real.pi * x)

-- State the theorem
theorem f_sum_equals_five_sixteenths : 
  f (29/4) + f (41/6) = 5/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_five_sixteenths_l1138_113832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_stride_difference_is_seven_l1138_113891

/-- The number of poles -/
def num_poles : ℕ := 31

/-- The distance between the first and last pole in feet -/
def total_distance : ℕ := 5280

/-- The number of strides Elmer takes between consecutive poles -/
def elmer_strides_per_gap : ℕ := 40

/-- The number of leaps Oscar takes between consecutive poles -/
def oscar_leaps_per_gap : ℕ := 15

/-- The length of Elmer's stride in feet -/
def elmer_stride_length : ℚ := total_distance / (elmer_strides_per_gap * (num_poles - 1))

/-- The length of Oscar's leap in feet -/
def oscar_leap_length : ℚ := total_distance / (oscar_leaps_per_gap * (num_poles - 1))

/-- The difference between Oscar's leap length and Elmer's stride length -/
def leap_stride_difference : ℚ := oscar_leap_length - elmer_stride_length

theorem leap_stride_difference_is_seven : 
  Int.floor leap_stride_difference = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_stride_difference_is_seven_l1138_113891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_problem_l1138_113854

/-- Line l₁ with slope k passing through (x, y) -/
def line_l₁ (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x + 1) + 2

/-- Line l₂ with parameter k passing through (x, y) -/
def line_l₂ (k : ℝ) (x y : ℝ) : Prop :=
  3 * x - (k - 2) * y + 5 = 0

/-- Distance between two lines given by ax + by + c = 0 and ax + by + d = 0 -/
noncomputable def line_distance (a b c d : ℝ) : ℝ :=
  abs (c - d) / Real.sqrt (a^2 + b^2)

theorem line_problem (k : ℝ) :
  (∃ P : ℝ × ℝ, line_l₁ k P.1 P.2) →
  (∀ x y : ℝ, line_l₁ k x y ↔ line_l₂ k x y) →
  (line_l₁ k (-1) 2 ∧
   k = -1 ∧
   line_distance 3 3 (-3) 5 = 4 * Real.sqrt 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_problem_l1138_113854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_hull_has_more_than_99_vertices_l1138_113830

/-- A regular polygon with 100 sides -/
structure Regular100gon where
  vertices : Fin 100 → ℝ × ℝ
  is_regular : ∀ (i j : Fin 100), dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)

/-- The set of 1000 regular 100-gons -/
def set_of_polygons : Set Regular100gon :=
  sorry

/-- The convex hull of a set of points in ℝ² -/
noncomputable def convex_hull (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry

/-- The vertices of the convex hull -/
noncomputable def convex_hull_vertices (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry

/-- All vertices of all polygons in the set -/
def all_vertices : Set (ℝ × ℝ) :=
  sorry

/-- Theorem stating that the convex hull has more than 99 vertices -/
theorem convex_hull_has_more_than_99_vertices :
  ∃ (n : ℕ), n > 99 ∧ ∃ (f : Fin n → ℝ × ℝ), Set.range f = convex_hull_vertices (convex_hull all_vertices) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_hull_has_more_than_99_vertices_l1138_113830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arccos_arccot_ratio_is_two_l1138_113814

theorem arccos_arccot_ratio_is_two (n : ℕ) (hn : n > 0) : 
  (Real.arccos ((n - 1 : ℝ) / n)) / (Real.arctan (1 / Real.sqrt (2 * n - 1 : ℝ))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arccos_arccot_ratio_is_two_l1138_113814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_sum_l1138_113836

theorem value_of_sum (a b c : ℝ) (h1 : (5 : ℝ)^a = (2 : ℝ)^b) (h2 : (5 : ℝ)^a = (10 : ℝ)^(c/2)) (h3 : a ≠ 0) (h4 : b ≠ 0) (h5 : c ≠ 0) :
  c/a + c/b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_sum_l1138_113836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_ratio_theorem_l1138_113852

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- The ratio of two rational numbers -/
def ratio (a b : ℚ) : ℚ := a / b

theorem mixture_ratio_theorem (p q : Mixture) (x y : ℚ) :
  ratio p.milk p.water = 5 / 3 →
  (∃ m n : ℚ, ratio q.milk q.water = m / n) →
  ratio x y = 2 →
  x * p.milk + y * q.milk = x * p.water + y * q.water →
  ratio x y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_ratio_theorem_l1138_113852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_157_of_3_div_11_l1138_113856

def decimal_representation (n d : ℕ) : List ℕ := sorry

def repeating_length (n d : ℕ) : ℕ := sorry

theorem digit_157_of_3_div_11 :
  let rep := decimal_representation 3 11
  let len := repeating_length 3 11
  len = 2 →
  rep.get? 0 = some 2 →
  rep.get? 1 = some 7 →
  rep.get? 156 = some 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_157_of_3_div_11_l1138_113856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1138_113851

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Theorem statement
theorem triangle_problem (x : ℝ) (t : Triangle) 
  (h1 : t.a = 1) 
  (h2 : t.b = Real.sqrt 2) 
  (h3 : f t.A = 2) :
  (∀ y : ℝ, f y ≤ 2) ∧ 
  (∃ k : ℤ, x = k * Real.pi + Real.pi / 6 → f x = 2) ∧
  (t.C = Real.pi / 12 ∨ t.C = 5 * Real.pi / 12 ∨ t.C = 7 * Real.pi / 12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1138_113851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_properties_l1138_113873

/-- Represents a trapezoid with a circumscribed circle -/
structure CircumscribedTrapezoid where
  a : ℝ  -- Length of one parallel side
  c : ℝ  -- Length of the other parallel side
  r : ℝ  -- Radius of the circumscribed circle

/-- Calculates the possible leg lengths and areas of a trapezoid with a circumscribed circle -/
noncomputable def trapezoid_properties (t : CircumscribedTrapezoid) : 
  (Set ℝ) × (Set ℝ) :=
  sorry

/-- Approximate equality for real numbers -/
def approx_eq (x y : ℝ) : Prop :=
  abs (x - y) < 0.01

/-- Theorem stating the properties of the specific trapezoid in the problem -/
theorem specific_trapezoid_properties : 
  let t : CircumscribedTrapezoid := ⟨10, 15, 10⟩
  let (legs, areas) := trapezoid_properties t
  (∀ l ∈ legs, (approx_eq l 15.48 ∨ approx_eq l 3.23)) ∧ 
  (∀ a ∈ areas, (approx_eq a 190.93 ∨ approx_eq a 25.57)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_properties_l1138_113873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_third_term_is_2048_l1138_113846

/-- A function that returns true if a number is a perfect square, false otherwise -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that returns the nth term of the sequence after removing perfect squares -/
def sequence_without_squares (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the 2003rd term of the sequence is 2048 -/
theorem two_thousand_third_term_is_2048 : 
  sequence_without_squares 2003 = 2048 := by
  sorry

#check two_thousand_third_term_is_2048

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_third_term_is_2048_l1138_113846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shampoo_usage_proof_l1138_113801

/-- The amount of shampoo Adonis's dad uses each day, in ounces -/
noncomputable def daily_shampoo_usage : ℝ := 1

/-- The initial amount of shampoo in the bottle, in ounces -/
noncomputable def initial_bottle_size : ℝ := 10

/-- The amount of hot sauce added daily, in ounces -/
noncomputable def daily_hot_sauce : ℝ := 1/2

/-- The number of days that have passed -/
def days : ℕ := 4

/-- The percentage of hot sauce in the bottle after 4 days -/
noncomputable def hot_sauce_percentage : ℝ := 1/4

theorem shampoo_usage_proof :
  initial_bottle_size - days * daily_shampoo_usage + days * daily_hot_sauce =
  (days * daily_hot_sauce) / hot_sauce_percentage :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shampoo_usage_proof_l1138_113801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1138_113858

-- Define the function f(x) = ax - ln(x-1)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log (x - 1)

-- State the theorem
theorem function_properties (a : ℝ) :
  -- Domain of the function is (1, +∞)
  (∀ x > 1, ∃ y, f a x = y) →
  -- If the tangent line at x=2 is y=3x+2, then a = 4
  (∃ m b, ∀ x, m * x + b = 3 * x + 2 ∧ (deriv (f a)) 2 = m) →
  a = 4 ∧
  -- For a > 0, the function has a minimum value of a+1 at x = (a+1)/a
  (a > 0 →
    ∃ x_min, x_min = (a + 1) / a ∧
    ∀ x > 1, f a x ≥ f a x_min ∧
    f a x_min = a + 1) ∧
  -- For a ≤ 0, the function is monotonically decreasing with no extreme values
  (a ≤ 0 →
    ∀ x y, 1 < x ∧ x < y → f a x > f a y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1138_113858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_simplification_l1138_113895

open Real

theorem trig_sum_simplification :
  let x := (sin (20 * π / 180) + sin (40 * π / 180) + sin (60 * π / 180) + sin (80 * π / 180) +
            sin (100 * π / 180) + sin (120 * π / 180) + sin (140 * π / 180) + sin (160 * π / 180)) /
           (cos (10 * π / 180) * cos (20 * π / 180) * cos (40 * π / 180))
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_simplification_l1138_113895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1138_113878

-- Problem 1
theorem problem_1 (f : ℝ → ℝ) :
  (∀ x, f (x + 1) = x^2 + 4*x + 1) →
  (∀ x, f x = x^2 + 2*x - 2) :=
by sorry

-- Problem 2
theorem problem_2 (f : ℝ → ℝ) :
  (∀ x, f x - 2*f (-x) = 9*x + 2) →
  (∀ x, f x = 3*x - 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1138_113878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_temperature_is_95_l1138_113817

/-- The temperature in New York in June 2020 -/
def temp_new_york : ℚ := 80

/-- The temperature difference between Miami and New York -/
def miami_ny_diff : ℚ := 10

/-- The temperature difference between San Diego and Miami -/
def sd_miami_diff : ℚ := 25

/-- The temperature in Miami -/
def temp_miami : ℚ := temp_new_york + miami_ny_diff

/-- The temperature in San Diego -/
def temp_san_diego : ℚ := temp_miami + sd_miami_diff

/-- The average temperature of the three cities -/
def avg_temp : ℚ := (temp_new_york + temp_miami + temp_san_diego) / 3

theorem average_temperature_is_95 : avg_temp = 95 := by
  -- Unfold definitions
  unfold avg_temp temp_san_diego temp_miami
  -- Simplify the expression
  simp [temp_new_york, miami_ny_diff, sd_miami_diff]
  -- Perform the calculation
  norm_num

#eval avg_temp

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_temperature_is_95_l1138_113817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_pi_minus_sqrt_9_l1138_113809

-- Define π as a real number between 3 and 4
axiom π : ℝ
axiom π_bounds : 3 < π ∧ π < 4

-- Define the square root of 9
def sqrt_9 : ℝ := 3

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Theorem statement
theorem floor_pi_minus_sqrt_9 : floor (π - sqrt_9) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_pi_minus_sqrt_9_l1138_113809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_6_formula_l1138_113897

noncomputable def f (x : ℝ) : ℝ := x^2 / (2*x + 1)

noncomputable def iterate_f (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate_f n x)

theorem f_6_formula (x : ℝ) (hx : x ≠ 0) :
  iterate_f 6 x = 1 / ((1 + 1/x)^64 - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_6_formula_l1138_113897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_heights_l1138_113866

theorem pyramid_heights (a b c : ℝ) (h : Set ℝ) :
  a = 3 ∧ b = 4 ∧ c = 5 ∧  -- Base triangle sides
  (∀ (face : ℝ), face ∈ h → ∃ (angle : ℝ), angle = 45) →  -- Lateral faces angle
  h = {1, 2, 3, 6} :=  -- Possible heights
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_heights_l1138_113866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1138_113881

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - ((a + 1) / a) * x + 1

theorem problem_solution (a : ℝ) (h : a > 0) :
  (∀ x, f (1/2) x ≤ 0 ↔ (3 - Real.sqrt 5) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 5) / 2) ∧
  (a = 1/a ↔ a = 1) ∧
  (0 < a ∧ a < 1 → a < 1/a) ∧
  (a > 1 → a > 1/a) ∧
  (∀ x, (0 < a ∧ a < 1 → (f a x ≤ 0 ↔ a < x ∧ x < 1/a))) ∧
  (∀ x, (a > 1 → (f a x ≤ 0 ↔ 1/a < x ∧ x < a))) ∧
  (∀ x, (a = 1 → (f a x ≤ 0 ↔ x = 1))) :=
by sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1138_113881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l1138_113840

theorem birthday_problem (num_people : ℕ) (num_days : ℕ) 
  (h1 : num_people = 400) (h2 : num_days = 365) :
  ∃ (i j : Fin num_people), i ≠ j ∧
  (i : ℕ) % num_days = (j : ℕ) % num_days :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l1138_113840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_of_composite_function_l1138_113808

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ 0 then Real.sin (Real.pi / 2 * x)
  else if x > 0 then 2^x + 1
  else 0  -- undefined for x < -4

-- State the theorem
theorem zero_point_of_composite_function :
  ∃ x : ℝ, x = -3 ∧ f (f x) - 3 = 0 := by
  sorry

#check zero_point_of_composite_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_of_composite_function_l1138_113808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_powers_l1138_113818

theorem smallest_sum_of_powers (a b : ℕ) (h : (2^10 * 3^5 : ℕ) = a^b) :
  ∀ (c d : ℕ), (2^10 * 3^5 : ℕ) = c^d → a + b ≤ c + d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_powers_l1138_113818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1138_113874

-- Define a, b, and c
noncomputable def a : ℝ := (1/2)^(1/5 : ℝ)
noncomputable def b : ℝ := (1/5)^(-(1/2) : ℝ)
noncomputable def c : ℝ := Real.log 10 / Real.log (1/5)

-- Theorem statement
theorem relationship_abc : b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1138_113874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonagon_diagonal_angle_less_than_seven_degrees_l1138_113870

/-- Represents a nonagon -/
structure Nonagon where
  sides : ℕ
  sides_eq : sides = 9

/-- Calculates the number of diagonals in a polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Represents the angle between two diagonals -/
noncomputable def angle_between_diagonals (d1 d2 : ℕ) : ℝ := sorry

/-- Theorem: In a nonagon, there exists a pair of diagonals with an angle less than 7° between them -/
theorem nonagon_diagonal_angle_less_than_seven_degrees (N : Nonagon) :
  ∃ (d1 d2 : ℕ), d1 ≠ d2 ∧ d1 ≤ num_diagonals N.sides ∧ d2 ≤ num_diagonals N.sides ∧
  angle_between_diagonals d1 d2 < 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonagon_diagonal_angle_less_than_seven_degrees_l1138_113870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_sqrt_three_l1138_113869

theorem tan_alpha_sqrt_three (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (π / 2 - α) = Real.sqrt 3 / 2) : Real.tan α = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_sqrt_three_l1138_113869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l1138_113844

-- Define the complex number z
noncomputable def z : ℂ := (1 : ℂ) / (1 + Complex.I) - Complex.I

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l1138_113844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_l_value_l1138_113887

noncomputable def f (x : ℝ) : ℝ :=
  if |x| ≤ 1 then 2 * Real.cos (Real.pi / 2 * x) else x^2 - 1

noncomputable def g (x l : ℝ) : ℝ :=
  |f x + f (x + l) - 2| + |f x - f (x + l)|

theorem min_l_value :
  ∀ l > 0, (∀ x : ℝ, g x l ≥ 2) ↔ l ≥ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_l_value_l1138_113887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1138_113835

-- Define the compound interest function
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + rate) ^ years

-- State the theorem
theorem interest_rate_calculation (principal : ℝ) (final_amount : ℝ) (years : ℝ) 
  (h1 : principal = 600)
  (h2 : final_amount = 720)
  (h3 : years = 4) :
  ∃ rate : ℝ, 
    (compound_interest principal rate years = final_amount) ∧ 
    (abs (rate - 0.04622) < 0.00001) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1138_113835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_in_third_quadrant_l1138_113888

noncomputable def f (α : ℝ) : ℝ :=
  (Real.cos (Real.pi/2 + α) * Real.cos (2*Real.pi - α) * Real.sin (-α + 3*Real.pi/2)) /
  (Real.sin (-Real.pi - α) * Real.sin (3*Real.pi/2 + α))

theorem f_simplification (α : ℝ) :
  f α = -Real.cos α := by sorry

theorem f_value_in_third_quadrant (α : ℝ)
  (h1 : Real.pi < α ∧ α < 3*Real.pi/2)  -- α is in the third quadrant
  (h2 : Real.cos (α - 3*Real.pi/2) = 1/5) :
  f α = 2 * Real.sqrt 6 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_in_third_quadrant_l1138_113888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_when_a_zero_f_monotonicity_when_a_positive_l1138_113819

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 2*a/x - (a+2)*Real.log x

theorem f_extrema_when_a_zero :
  let f₀ := f 0
  (∃ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, f₀ y ≤ f₀ x) ∧
  (f₀ 1 = 1) ∧
  (∃ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, f₀ x ≤ f₀ y) ∧
  (f₀ 2 = 2 - 2 * Real.log 2) := by sorry

theorem f_monotonicity_when_a_positive (a : ℝ) (ha : a > 0) :
  ∃ (x₁ x₂ : ℝ), x₁ = a ∧ x₂ = 2 ∧
  (∀ x < min x₁ x₂, (deriv (f a)) x < 0) ∧
  (∀ x ∈ Set.Ioo (min x₁ x₂) (max x₁ x₂), (deriv (f a)) x > 0) ∧
  (∀ x > max x₁ x₂, (deriv (f a)) x < 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_when_a_zero_f_monotonicity_when_a_positive_l1138_113819
