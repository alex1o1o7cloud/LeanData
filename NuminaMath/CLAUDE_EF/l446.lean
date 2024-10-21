import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_person_with_many_acquaintances_l446_44684

-- Define the total population
def total_population : ℕ := 1000000

-- Define the proportion of believers
def believer_proportion : ℚ := 9/10

-- Define the proportion of believing acquaintances
def believing_acquaintance_proportion : ℚ := 1/10

-- Define the minimum number of acquaintances to prove
def min_acquaintances : ℕ := 810

-- Theorem statement
theorem exists_person_with_many_acquaintances :
  ∃ (person : Fin total_population),
    ∃ (acquaintances : Finset (Fin total_population)),
      acquaintances.card ≥ min_acquaintances ∧
      ∀ a, a ∈ acquaintances → a ≠ person :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_person_with_many_acquaintances_l446_44684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l446_44677

theorem abc_inequality (n : ℕ) :
  (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 →
    a * b * c * (a ^ n + b ^ n + c ^ n) ≤ 1 / 3 ^ (n + 2)) ↔
  n = 1 ∨ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l446_44677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_sum_of_a_and_b_l446_44628

-- Define the complex number z
noncomputable def z : ℂ := ((1 + Complex.I)^2 + 3*(1 - Complex.I)) / (2 + Complex.I)

-- Theorem for the modulus of z
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by sorry

-- Theorem for the sum of a and b
theorem sum_of_a_and_b (a b : ℝ) (h : z^2 + a*z + b = 1 + Complex.I) : a + b = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_sum_of_a_and_b_l446_44628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_on_line_and_curves_l446_44601

/-- The distance between two points on a line in polar coordinates, where one point is on the curve ρ = 4sinθ and the other is on the curve ρ = 2sinθ -/
theorem distance_between_points_on_line_and_curves :
  let line := λ t : Real => (-t, Real.sqrt 3 * t)
  let curve1 := λ θ : Real => 4 * Real.sin θ
  let curve2 := λ θ : Real => 2 * Real.sin θ
  ∀ θ : Real,
  let ρ1 := curve1 θ
  let ρ2 := curve2 θ
  (∃ t : Real, line t = (ρ1 * Real.cos θ, ρ1 * Real.sin θ)) →
  (∃ t : Real, line t = (ρ2 * Real.cos θ, ρ2 * Real.sin θ)) →
  ρ1 ≠ 0 → ρ2 ≠ 0 →
  |ρ1 - ρ2| = Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_on_line_and_curves_l446_44601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_bijection_exists_l446_44600

/-- For any set of n distinct positive integers (n ≥ 3), there exists a bijection
    from {1,2,...,n} to the set such that for any three elements in increasing order,
    the square of the middle element is not equal to the product of the other two. -/
theorem special_bijection_exists (n : ℕ) (hn : n ≥ 3) (S : Finset ℕ)
  (hS : S.card = n) (hSpos : ∀ x ∈ S, x > 0) :
  ∃ f : Fin n → ℕ, Function.Bijective f ∧ (∀ x, f x ∈ S) ∧
    ∀ i j k : Fin n, i < j → j < k → (f j) ^ 2 ≠ (f i) * (f k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_bijection_exists_l446_44600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_equals_one_l446_44671

def p (n : ℕ) (x : ℕ) : ℕ :=
  (Finset.range n).sum (λ k =>
    x^k * (Finset.range (n - k)).prod (λ i => 1 - x^(k + i + 1)))

theorem p_equals_one (n : ℕ) (x : ℕ) : p n x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_equals_one_l446_44671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poor_man_makes_fortune_l446_44666

/-- Represents the financial status of a person -/
inductive FinancialStatus
  | Poor
  | Wealthy

/-- Represents the outcome of doing business -/
inductive BusinessOutcome
  | MakeFortune
  | Other

/-- A function that determines the outcome of doing business for a given number of years -/
def businessResult (initialStatus : FinancialStatus) (years : ℕ) : BusinessOutcome :=
  sorry -- Implementation details omitted for simplicity

/-- Theorem stating that a poor man doing business for three years results in making a fortune -/
theorem poor_man_makes_fortune :
  ∀ (man : FinancialStatus),
    man = FinancialStatus.Poor →
    businessResult man 3 = BusinessOutcome.MakeFortune :=
by
  sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_poor_man_makes_fortune_l446_44666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_cone_l446_44624

/-- The maximum volume of a cone with slant height a -/
noncomputable def max_cone_volume (a : ℝ) : ℝ := (2 * Real.pi * a^3) / (9 * Real.sqrt 3)

/-- Theorem: The maximum volume of a cone with slant height a is (2 * π * a^3) / (9 * √3) -/
theorem max_volume_cone (a : ℝ) (h : a > 0) :
  ∀ (r h : ℝ), r > 0 → h > 0 → r^2 + h^2 = a^2 →
  (1/3 * Real.pi * r^2 * h) ≤ max_cone_volume a :=
by
  sorry

#check max_volume_cone

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_cone_l446_44624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_l446_44695

def digits : Finset Nat := {1, 3, 5, 7, 8}

def is_valid_subtraction (a b : Nat) : Prop :=
  a ≥ 100 ∧ a < 1000 ∧ b ≥ 10 ∧ b < 100 ∧
  (∀ d, d ∈ digits → (d ∈ (Nat.digits 10 a).toFinset ∨ d ∈ (Nat.digits 10 b).toFinset)) ∧
  (∀ d, d ∈ (Nat.digits 10 a).toFinset ∨ d ∈ (Nat.digits 10 b).toFinset → d ∈ digits) ∧
  (Nat.digits 10 a).length + (Nat.digits 10 b).length = 5

theorem smallest_difference : 
  ∀ a b, is_valid_subtraction a b → a - b ≥ 48 := by
  sorry

#check smallest_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_l446_44695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_sets_relation_l446_44642

/-- Given a quadratic function f(x) = x^2 + bx + c, if the set A = {x | f(x) = x} = {2},
    then there exist values for b and c such that the set B = {x | f(x-1) = x+1} is non-empty. -/
theorem quadratic_sets_relation : 
  ∃ b c : ℝ, 
    let f := λ x : ℝ => x^2 + b*x + c
    let A := {x : ℝ | f x = x}
    let B := {x : ℝ | f (x-1) = x+1}
    A = {2} ∧ ∃ x, x ∈ B :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_sets_relation_l446_44642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_with_min_period_pi_over_2_l446_44650

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def min_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  has_period f p ∧ ∀ q, 0 < q → q < p → ¬ has_period f q

theorem odd_function_with_min_period_pi_over_2 (f : ℝ → ℝ) :
  is_odd f → min_positive_period f (Real.pi / 2) → f = λ x ↦ Real.sin (4 * x) :=
by
  sorry

#check odd_function_with_min_period_pi_over_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_with_min_period_pi_over_2_l446_44650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l446_44697

theorem trig_problem (α β : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2))
  (h2 : β ∈ Set.Ioo 0 (π/2))
  (h3 : Real.sin (α + 2*β) = 1/3) :
  (α + β = 2*π/3 → Real.sin β = (2*Real.sqrt 6 - 1) / 6) ∧
  (Real.sin β = 4/5 → Real.cos α = (24 + 14*Real.sqrt 2) / 75) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l446_44697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_hyperbola_with_perpendicular_asymptotes_l446_44689

/-- A hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2) / h.a

/-- Condition for perpendicular asymptotes -/
def has_perpendicular_asymptotes (h : Hyperbola) : Prop :=
  h.a = h.b

/-- Theorem: The eccentricity of a hyperbola with perpendicular asymptotes is √2 -/
theorem eccentricity_of_hyperbola_with_perpendicular_asymptotes (h : Hyperbola) 
  (perp : has_perpendicular_asymptotes h) : eccentricity h = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_hyperbola_with_perpendicular_asymptotes_l446_44689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_composed_function_l446_44662

-- Define the Gauss function
noncomputable def gaussFunction (x : ℝ) : ℤ := ⌊x⌋

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3^x) / (1 + 3^x) - 1/3

-- Define the composition of Gauss function and f
noncomputable def composedFunction (x : ℝ) : ℤ := gaussFunction (f x)

-- Theorem statement
theorem range_of_composed_function :
  Set.range composedFunction = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_composed_function_l446_44662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_l446_44675

def sequenceA (y : ℕ) : ℕ → ℤ
  | 0 => 2000
  | 1 => y
  | (n + 2) => sequenceA y n - sequenceA y (n + 1)

def sequence_length (y : ℕ) : ℕ :=
  (Finset.filter (λ n => sequenceA y n > 0) (Finset.range 1000)).card

theorem max_sequence_length :
  ∀ y : ℕ, y > 0 → sequence_length y ≤ sequence_length 1333 := by
  sorry

#eval sequence_length 1333

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_l446_44675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_caloric_deficit_l446_44641

/-- Jonathan's daily calorie intake except for Saturday -/
def regular_intake : ℤ := 2500

/-- Extra calories Jonathan consumes on Saturday -/
def saturday_extra : ℤ := 1000

/-- Jonathan's daily calorie burn -/
def daily_burn : ℤ := 3000

/-- Number of days in a week -/
def days_in_week : ℤ := 7

/-- Calculate Jonathan's weekly caloric deficit -/
theorem weekly_caloric_deficit :
  (days_in_week - 1) * regular_intake + (regular_intake + saturday_extra) - days_in_week * daily_burn = -2500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_caloric_deficit_l446_44641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equality_l446_44690

/-- For any real number x > 1, the sum of the infinite series
    Σ(1 / (x^(3^n) - x^(-3^n))) from n = 1 to ∞ is equal to 1 / (x^3 - 1). -/
theorem infinite_sum_equality (x : ℝ) (hx : x > 1) :
  ∑' n : ℕ, 1 / (x^(3^n) - (1/x)^(3^n)) = 1 / (x^3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equality_l446_44690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l446_44685

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 3 * x^4 + 4 / x^3

-- Theorem statement
theorem function_properties :
  (∃ (m : ℝ), m = 7 ∧ ∀ x, x > 0 → f x ≥ m) ∧
  (∀ x₁ x₂, x₁ > 0 → x₂ > 0 → f x₁ = f x₂ → x₁ < x₂ →
    (x₁^3 + (2-x₁)^3 < x₁^4 + (2-x₁)^4) ∧
    (x₁ + x₂ > 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l446_44685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_diagonal_l446_44611

/-- Given a rectangular prism with volume v and a triangle with perimeter k and area t,
    where the sides of the triangle are equal to the lengths of the three mutually
    perpendicular edges of the rectangular prism, this theorem proves that the distance d
    between two opposite vertices of the rectangular prism is
    (1/(2k)) * sqrt(2(k^4 - 16t^2 - 8vk)). -/
theorem rectangular_prism_diagonal (v k t : ℝ) (hv : v > 0) (hk : k > 0) (ht : t > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a * b * c = v ∧
    a + b + c = k ∧
    16 * t^2 = k * (k - 2*a) * (k - 2*b) * (k - 2*c) ∧
    (Real.sqrt (a^2 + b^2 + c^2) : ℝ) = (1/(2*k)) * Real.sqrt (2*(k^4 - 16*t^2 - 8*v*k)) := by
  sorry

#check rectangular_prism_diagonal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_diagonal_l446_44611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_tangents_l446_44694

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the point P
def P : ℝ × ℝ := (3, 2)

-- Define the angle between tangents
noncomputable def angle_between_tangents : ℝ := Real.arccos (3/5)

-- Theorem statement
theorem cosine_of_angle_between_tangents :
  Real.cos angle_between_tangents = 3/5 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_tangents_l446_44694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_multiplication_l446_44604

theorem repeating_decimal_multiplication (x y : ℕ) :
  x < 10 ∧ y < 10 →
  (45 : ℚ) * (2 + (10 * x + y : ℚ) / 99) - 45 * (2 + (10 * x + y : ℚ) / 100) = 135/100 →
  10 * x + y = 30 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_multiplication_l446_44604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l446_44621

theorem tan_alpha_value (α : ℝ) 
  (h1 : Real.cos α = -4/5) 
  (h2 : π/2 < α) 
  (h3 : α < π) : 
  Real.tan α = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l446_44621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l446_44602

noncomputable def solution_count (C : ℝ) : ℕ :=
  if C < 0 ∨ C = 0 ∨ C > 1 ∨ (0 < C ∧ C < 1 / Real.sqrt 2) then 0
  else if C = 1 ∨ C = 1 / Real.sqrt 2 then 4
  else if 1 / Real.sqrt 2 < C ∧ C < 1 then 8
  else 0

theorem system_solutions (C : ℝ) :
  (∃ x y : ℝ, |x| + |y| = 1 ∧ x^2 + y^2 = C) →
  solution_count C = (
    if C < 0 ∨ C = 0 ∨ C > 1 ∨ (0 < C ∧ C < 1 / Real.sqrt 2) then 0
    else if C = 1 ∨ C = 1 / Real.sqrt 2 then 4
    else if 1 / Real.sqrt 2 < C ∧ C < 1 then 8
    else 0
  ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l446_44602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_form_parallelogram_with_area_l446_44606

noncomputable def polynomial (x : ℂ) : ℂ := x^4 - 4*Complex.I*x^3 + 3*x^2 - 14*Complex.I*x - 44

def roots (p : ℂ → ℂ) : Set ℂ := {x : ℂ | p x = 0}

def is_parallelogram (s : Set ℂ) : Prop :=
  ∃ (a b c d : ℂ), s = {a, b, c, d} ∧ (b - a = d - c) ∧ (c - a = d - b)

noncomputable def area_of_parallelogram (s : Set ℂ) : ℝ :=
  sorry  -- Definition of area calculation

theorem polynomial_roots_form_parallelogram_with_area :
  is_parallelogram (roots polynomial) ∧
  area_of_parallelogram (roots polynomial) = 8 * Real.sqrt 3 := by
  sorry

#check polynomial_roots_form_parallelogram_with_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_form_parallelogram_with_area_l446_44606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_is_two_l446_44646

-- Define the sequence (marked as noncomputable due to dependence on reals)
noncomputable def f (n : ℝ) : ℝ := (2*n - 5) / (n + 1)

-- State the theorem
theorem limit_of_f_is_two :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |f n - 2| < ε :=
by
  sorry -- Proof is omitted for now

-- Optional: You can add a specific example or test case if needed
example : ∀ ε > 0, ∃ N, ∀ n ≥ N, |f n - 2| < ε :=
limit_of_f_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_is_two_l446_44646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_parabola_l446_44692

/-- The parabola is defined by the equation x = y^2/4 -/
def parabola (x y : ℝ) : Prop := x = y^2 / 4

/-- The point on the plane -/
def point : ℝ × ℝ := (8, 8)

/-- The shortest distance between a point and the parabola -/
noncomputable def shortest_distance : ℝ := 4 * Real.sqrt 2

/-- Theorem stating that the shortest distance is indeed the minimum distance -/
theorem shortest_distance_to_parabola :
  ∀ (x y : ℝ), parabola x y →
  Real.sqrt ((x - point.1)^2 + (y - point.2)^2) ≥ shortest_distance :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_parabola_l446_44692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tshirt_max_discount_l446_44699

/-- The maximum discount percentage that can be offered while maintaining
    a minimum profit percentage on a product. -/
noncomputable def max_discount (cost_price selling_price min_profit_percent : ℝ) : ℝ :=
  (1 - (cost_price * (1 + min_profit_percent / 100)) / selling_price) * 100

/-- Theorem stating that the maximum discount for the given scenario is 90% -/
theorem tshirt_max_discount :
  max_discount 400 600 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tshirt_max_discount_l446_44699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_eccentricity_l446_44687

/-- Given a hyperbola x^2 - y^2 = 1 with foci F₁ and F₂, and a point P on the hyperbola
    such that PF₁ ⊥ PF₂, the eccentricity of the ellipse with foci F₁ and F₂ passing
    through P is √6/3 -/
theorem hyperbola_ellipse_eccentricity 
  (F₁ F₂ P : ℝ × ℝ) -- F₁, F₂, and P are points in ℝ²
  (h_hyperbola : ∀ (x y : ℝ), (x, y) ∈ {(x, y) | x^2 - y^2 = 1} → 
    ∃ (t : ℝ), (x, y) = (Real.cosh t, Real.sinh t)) -- Definition of hyperbola
  (h_foci : F₁ ∈ {(x, y) | x^2 - y^2 = 1} ∧ F₂ ∈ {(x, y) | x^2 - y^2 = 1}) -- F₁ and F₂ are foci
  (h_P_on_hyperbola : P ∈ {(x, y) | x^2 - y^2 = 1}) -- P is on the hyperbola
  (h_perpendicular : (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0) -- PF₁ ⊥ PF₂
  : ∃ (a c : ℝ), a > c ∧ c > 0 ∧ 
    (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 + (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = 4 * a^2 ∧
    (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 4 * c^2 ∧
    c / a = Real.sqrt 6 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_eccentricity_l446_44687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_53_factorial_mod_59_l446_44636

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem remainder_53_factorial_mod_59 : ∃ k : ℕ, factorial 53 = 59 * k + 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_53_factorial_mod_59_l446_44636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l446_44634

theorem exponential_inequality (x : ℝ) : (2 : ℝ)^(x - 2) < 1 ↔ x < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l446_44634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l446_44679

-- Define the constants
noncomputable def a : ℝ := Real.sqrt 0.5
noncomputable def b : ℝ := (0.9 : ℝ) ^ (-1/4 : ℝ)
noncomputable def c : ℝ := Real.log 0.3 / Real.log 5

-- State the theorem
theorem ascending_order : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l446_44679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chip_ratio_l446_44678

theorem chip_ratio (total_chips : ℕ) (lyle_percentage : ℚ) :
  total_chips = 100 →
  lyle_percentage = 60 / 100 →
  (total_chips * (1 - lyle_percentage) : ℚ) / (total_chips * lyle_percentage) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chip_ratio_l446_44678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_and_value_l446_44618

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2 * x * Real.exp (-x) else -2 * (2 - x) * Real.exp (-(2 - x))

-- State the theorem
theorem f_symmetry_and_value : 
  (∀ x, f (1 + x) = -f (1 - x)) →  -- Symmetry about (1,0)
  f (2 + 3 * Real.log 2) = 48 * Real.log 2 := by
  sorry

-- Additional lemma to show the symmetry property
lemma f_symmetry : ∀ x, f (1 + x) = -f (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_and_value_l446_44618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_equivalence_l446_44649

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x

noncomputable def g (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * sin (2 * x) + sin x ^ 2 - 1 / 2

-- State the theorem
theorem graph_shift_equivalence :
  ∀ x : ℝ, f x = g (x + π / 12) :=
by
  intro x
  -- The proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_equivalence_l446_44649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_bijection_l446_44667

def isDivisor (d n : ℕ) : Prop := n % d = 0

def setA (n : ℕ) : Set ℕ := {d | isDivisor d n ∧ d < Nat.sqrt n}
def setB (n : ℕ) : Set ℕ := {d | isDivisor d n ∧ d > Nat.sqrt n}

theorem divisor_bijection (n : ℕ) (hn : n > 1) :
  ∃ f : setA n → setB n,
    Function.Bijective f ∧ ∀ a : setA n, isDivisor a (f a) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_bijection_l446_44667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_measure_of_angle_A_area_of_triangle_ABC_l446_44644

noncomputable section

open Real

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Define the triangle ABC
def triangle_ABC (a b c A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

-- Define the condition b^2 + c^2 = a^2 + bc
def condition (a b c : ℝ) : Prop :=
  b^2 + c^2 = a^2 + b*c

-- Theorem 1: Measure of angle A
theorem measure_of_angle_A (a b c A B C : ℝ) 
  (h1 : triangle_ABC a b c A B C) 
  (h2 : condition a b c) : 
  A = Real.pi/3 := by sorry

-- Theorem 2: Area of triangle ABC
theorem area_of_triangle_ABC (a b c A B C : ℝ) 
  (h1 : triangle_ABC a b c A B C) 
  (h2 : condition a b c)
  (h3 : a = sqrt 7)
  (h4 : b = 2) : 
  (1/2) * b * c * sin A = (3 * sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_measure_of_angle_A_area_of_triangle_ABC_l446_44644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_lower_bound_when_a_is_one_l446_44696

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x * Real.log x

-- Part 1: Monotonically increasing condition
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x > 1, Monotone (fun x => f a x)) ↔ a ≥ -2 :=
sorry

-- Part 2: Lower bound when a = 1
theorem lower_bound_when_a_is_one :
  ∀ x > 1, f 1 x ≥ x - Real.exp (-x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_lower_bound_when_a_is_one_l446_44696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowball_melting_halved_velocity_l446_44645

/-- The specific heat of fusion of snow in J/kg -/
noncomputable def lambda : ℝ := 330000

/-- The initial percentage of snowball that melts -/
noncomputable def k : ℝ := 0.0002

/-- The relation between initial velocity and melting percentage -/
def velocity_melting_relation (v : ℝ) : Prop :=
  (1/2) * v^2 = lambda * (k / 100)

/-- The theorem to prove -/
theorem snowball_melting_halved_velocity (v : ℝ) :
  velocity_melting_relation v →
  (1/2) * (v/2)^2 = lambda * (0.0001 / 100) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowball_melting_halved_velocity_l446_44645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_fraction_is_one_fourth_l446_44665

-- Define the range of factors
def factor_range : Set ℕ := {n | n ≤ 15}

-- Define a function to check if a number is odd
def is_odd (n : ℕ) : Bool := n % 2 = 1

-- Define the multiplication table
def mult_table : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 16) (Finset.range 16)

-- Theorem: The fraction of odd numbers in the multiplication table is 1/4
theorem odd_fraction_is_one_fourth :
  (Finset.filter (fun pair => is_odd (pair.1 * pair.2)) mult_table).card /
  mult_table.card = 1 / 4 := by
  sorry

#eval (Finset.filter (fun pair => is_odd (pair.1 * pair.2)) mult_table).card
#eval mult_table.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_fraction_is_one_fourth_l446_44665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_in_fourth_quadrant_l446_44603

/-- Given 0 < m < 1, the complex number z = (m+1) + (m-1)i is located in Quadrant IV -/
theorem complex_in_fourth_quadrant (m : ℝ) (h : 0 < m ∧ m < 1) :
  let z : ℂ := Complex.mk (m + 1) (m - 1)
  z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_in_fourth_quadrant_l446_44603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l446_44691

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- The common ratio
  h_geom : ∀ n, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sum_n (seq : GeometricSequence) (n : ℕ) : ℝ :=
  if seq.q = 1 then n * seq.a 1
  else seq.a 1 * (1 - seq.q^n) / (1 - seq.q)

/-- Predicate to check if a list of three real numbers forms an arithmetic sequence -/
def IsArithmetic (l : List ℝ) : Prop :=
  match l with
  | [a, b, c] => b - a = c - b
  | _ => False

theorem geometric_sequence_property 
  (seq : GeometricSequence) 
  (h_arith : IsArithmetic [sum_n seq 3, sum_n seq 9, sum_n seq 6])
  (h_a8 : seq.a 8 = 3) :
  seq.a 5 = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l446_44691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l446_44630

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - Real.pi/2) * Real.cos (3*Real.pi/2 + α) * Real.tan (Real.pi - α)) / 
  (Real.tan (-α - Real.pi) * Real.sin (-α - Real.pi))

theorem f_value_in_third_quadrant (α : Real) 
  (h1 : Real.pi < α ∧ α < 3*Real.pi/2)  -- α is in the third quadrant
  (h2 : Real.cos (α - 3*Real.pi/2) = 1/5) : 
  f α = 2 * Real.sqrt 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l446_44630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_and_rectangle_l446_44653

/-- Right triangle ABC with legs a and b -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- The side length of the largest square with vertex C within the triangle -/
noncomputable def largest_square_side (t : RightTriangle) : ℝ := (t.a * t.b) / (t.a + t.b)

/-- The dimensions of the largest rectangle with vertex C within the triangle -/
noncomputable def largest_rectangle_dims (t : RightTriangle) : ℝ × ℝ := (t.a / 2, t.b / 2)

/-- Main theorem stating the properties of the largest square and rectangle -/
theorem largest_square_and_rectangle (t : RightTriangle) :
  (largest_square_side t = (t.a * t.b) / (t.a + t.b)) ∧
  (largest_rectangle_dims t = (t.a / 2, t.b / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_and_rectangle_l446_44653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_roots_l446_44615

noncomputable def geometric_sum (x : ℂ) (n : ℕ) : ℂ := (1 - x^(n+1)) / (1 - x)

noncomputable def Q (x : ℂ) : ℂ := (geometric_sum x 20)^2 - x^20

def is_root (x : ℂ) : Prop := Q x = 0

noncomputable def arg (z : ℂ) : ℝ := Real.arctan (z.im / z.re)

theorem sum_of_specific_roots :
  ∃ (β₁ β₂ β₆ : ℂ),
    is_root β₁ ∧ is_root β₂ ∧ is_root β₆ ∧
    0 < arg β₁ ∧ arg β₁ ≤ arg β₂ ∧ arg β₂ ≤ arg β₆ ∧
    (∀ β : ℂ, is_root β → 0 < arg β → arg β < arg β₆ → (β = β₁ ∨ β = β₂)) ∧
    β₁ + β₂ + β₆ = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_roots_l446_44615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_bases_for_45_and_54_l446_44605

/-- Converts a number from base b representation to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The problem statement --/
theorem smallest_bases_for_45_and_54 :
  ∃ m n, m = 11 ∧ n = 9 ∧
  to_decimal [5, 4] m = to_decimal [4, 5] n ∧
  (∀ m' n', m' < m ∨ n' < n → to_decimal [5, 4] m' ≠ to_decimal [4, 5] n') := by
  sorry

#eval to_decimal [5, 4] 11
#eval to_decimal [4, 5] 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_bases_for_45_and_54_l446_44605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l446_44680

theorem sin_cos_product (θ : Real) : 
  (π / 2 < θ ∧ θ < π) →  -- θ is in the second quadrant
  (Real.tan (θ + π / 4) = 1 / 2) → 
  Real.sin θ * Real.cos θ = -3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l446_44680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l446_44614

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the vectors m and n
noncomputable def m (t : Triangle) : Fin 2 → Real
  | 0 => Real.cos (3 * t.A / 2)
  | 1 => Real.sin (3 * t.A / 2)
  | _ => 0

noncomputable def n (t : Triangle) : Fin 2 → Real
  | 0 => Real.cos (t.A / 2)
  | 1 => Real.sin (t.A / 2)
  | _ => 0

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sqrt ((m t 0 + n t 0)^2 + (m t 1 + n t 1)^2) = Real.sqrt 3)
  (h2 : t.b + t.c = Real.sqrt 3 * t.a) :
  t.A = π / 3 ∧ (t.B = π / 6 ∧ t.C = π / 2 ∨ t.B = π / 2 ∧ t.C = π / 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l446_44614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nate_current_age_l446_44640

-- Define Ember's and Nate's ages as natural numbers
def ember_age : ℕ := sorry
def nate_age : ℕ := sorry

-- Condition 1: Ember is half as old as Nate
axiom ember_half_nate : ember_age = nate_age / 2

-- Condition 2: When Ember is 14, Nate will be 21
axiom future_ages : nate_age + (14 - ember_age) = 21

-- Theorem to prove
theorem nate_current_age : nate_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nate_current_age_l446_44640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hotdog_eating_competition_l446_44683

/-- The number of hotdogs the first competitor can eat per minute -/
def first_competitor_rate : ℝ := 10

/-- The number of hotdogs the second competitor can eat per minute -/
def second_competitor_rate : ℝ := 3 * first_competitor_rate

/-- The number of hotdogs the third competitor can eat per minute -/
def third_competitor_rate : ℝ := 2 * second_competitor_rate

theorem hotdog_eating_competition :
  third_competitor_rate * 5 = 300 ∧
  first_competitor_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hotdog_eating_competition_l446_44683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_implies_omega_l446_44638

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (ω * x + Real.pi / 3)

theorem period_implies_omega (ω : ℝ) (h : ∃ (T : ℝ), T > 0 ∧ T = Real.pi / 2 ∧ ∀ (x : ℝ), f ω x = f ω (x + T)) :
  ω = 4 ∨ ω = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_implies_omega_l446_44638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_odd_sums_l446_44670

theorem max_odd_sums (X Y Z W : ℕ) : 
  (Finset.filter (λ sum => sum % 2 = 1) 
    {X + Y, X + Z, X + W, Y + Z, Y + W, Z + W}).card ≤ 4 ∧ 
  ∃ (A B C D : ℕ), (Finset.filter (λ sum => sum % 2 = 1) 
    {A + B, A + C, A + D, B + C, B + D, C + D}).card = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_odd_sums_l446_44670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_chord_length_l446_44643

/-- Two concentric circles with radii 4 and 6 -/
structure ConcentricCircles where
  center : EuclideanSpace ℝ (Fin 2)
  radius₁ : ℝ
  radius₂ : ℝ
  radius₁_eq : radius₁ = 4
  radius₂_eq : radius₂ = 6

/-- A chord in the smaller circle -/
structure Chord (cc : ConcentricCircles) where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  in_O₁ : ‖A - cc.center‖ = cc.radius₁ ∧ ‖B - cc.center‖ = cc.radius₁
  length : ‖A - B‖ = 2

/-- The extended chord intersecting the larger circle -/
structure ExtendedChord (cc : ConcentricCircles) (ch : Chord cc) where
  C : EuclideanSpace ℝ (Fin 2)
  D : EuclideanSpace ℝ (Fin 2)
  on_line : ∃ t : ℝ, C = ch.A + t • (ch.B - ch.A) ∧
                     D = ch.A + (1 - t) • (ch.B - ch.A)
  in_O₂ : ‖C - cc.center‖ = cc.radius₂ ∧ ‖D - cc.center‖ = cc.radius₂

/-- The theorem statement -/
theorem extended_chord_length 
  (cc : ConcentricCircles) 
  (ch : Chord cc) 
  (ech : ExtendedChord cc ch) : 
  ‖ech.C - ech.D‖ = 2 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_chord_length_l446_44643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_existence_l446_44626

noncomputable def f (x : ℝ) := Real.log x - 2 / x

theorem zero_point_existence (h1 : Continuous f) 
  (h2 : StrictMono f) 
  (h3 : f 2 < 0) 
  (h4 : f (Real.exp 1) > 0) : 
  ∃! x : ℝ, 2 < x ∧ x < Real.exp 1 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_existence_l446_44626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_value_l446_44658

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.sin (x / 11)

theorem smallest_max_value (x : ℝ) :
  (∀ y, 0 < y → y < x → f y < f x) ∧
  (∀ z, f z ≤ f x) →
  x = 8910 * Real.pi / 180 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_value_l446_44658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l446_44616

noncomputable def f (x : ℝ) := Real.log (abs (Real.sin x))

theorem f_properties :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, f (x + π) = f x) ∧ 
  (∀ x y, 0 < x ∧ x < y ∧ y < π / 2 → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l446_44616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_180_kmh_5_sec_l446_44660

/-- Calculates the length of a train given its speed in km/h and the time it takes to cross a pole. -/
noncomputable def trainLength (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time_s

/-- Proves that a train traveling at 180 km/h and crossing a pole in 5 seconds has a length of 250 meters. -/
theorem train_length_180_kmh_5_sec : trainLength 180 5 = 250 := by
  -- Unfold the definition of trainLength
  unfold trainLength
  -- Simplify the expression
  simp
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_180_kmh_5_sec_l446_44660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vessel_base_length_for_given_cube_l446_44688

/-- The length of the base of a rectangular vessel containing a fully immersed cube -/
noncomputable def vessel_base_length (cube_edge : ℝ) (vessel_width : ℝ) (water_rise : ℝ) : ℝ :=
  (cube_edge ^ 3) / (vessel_width * water_rise)

/-- Theorem stating that for a cube with edge 15 cm immersed in a vessel with width 15 cm 
    and water rise of 11.25 cm, the vessel base length is 20 cm -/
theorem vessel_base_length_for_given_cube : 
  vessel_base_length 15 15 11.25 = 20 := by
  -- Unfold the definition of vessel_base_length
  unfold vessel_base_length
  -- Simplify the expression
  simp
  -- The proof is completed using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vessel_base_length_for_given_cube_l446_44688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_bill_calculation_l446_44610

/-- Calculates the final bill for ice cream sundaes and milkshakes --/
theorem ice_cream_bill_calculation 
  (alicia_sundae alicia_shake : ℚ)
  (brant_sundae brant_shake : ℚ)
  (josh_sundae josh_shake : ℚ)
  (yvette_sundae yvette_shake : ℚ)
  (tax_rate tip_rate : ℚ)
  (h1 : alicia_sundae = 7.5)
  (h2 : alicia_shake = 4)
  (h3 : brant_sundae = 10)
  (h4 : brant_shake = 4.5)
  (h5 : josh_sundae = 8.5)
  (h6 : josh_shake = 4)
  (h7 : yvette_sundae = 9)
  (h8 : yvette_shake = 4.5)
  (h9 : tax_rate = 0.08)
  (h10 : tip_rate = 0.2) :
  (let pre_tax_total := alicia_sundae + alicia_shake + brant_sundae + brant_shake + 
                       josh_sundae + josh_shake + yvette_sundae + yvette_shake
   let tax := pre_tax_total * tax_rate
   let total_with_tax := pre_tax_total + tax
   let tip := pre_tax_total * tip_rate
   let final_bill := total_with_tax + tip
   final_bill) = 66.56 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_bill_calculation_l446_44610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_eccentricity_l446_44674

/-- An ellipse with semi-major axis a and semi-minor axis b. -/
structure Ellipse (a b : ℝ) where
  h1 : a > 0
  h2 : b > 0
  h3 : a > b

/-- A line with slope m. -/
structure Line (m : ℝ) where

/-- The eccentricity of an ellipse. -/
noncomputable def eccentricity (e : Ellipse a b) : ℝ := Real.sqrt (1 - (b / a)^2)

/-- The x-coordinate of a focus of an ellipse. -/
noncomputable def focus_x (e : Ellipse a b) : ℝ := a * Real.sqrt (1 - (b / a)^2)

theorem ellipse_line_intersection_eccentricity 
  (a b : ℝ) (e : Ellipse a b) (l : Line 2) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / a^2 + y₁^2 / b^2 = 1 ∧
    x₂^2 / a^2 + y₂^2 / b^2 = 1 ∧
    y₁ = 2*x₁ ∧ y₂ = 2*x₂ ∧
    x₁ ≠ x₂ ∧
    x₁ = -focus_x e ∧ x₂ = focus_x e) →
  eccentricity e = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_eccentricity_l446_44674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_no_adjacent_birch_l446_44655

/-- The number of maple trees -/
def num_maple : ℕ := 3

/-- The number of oak trees -/
def num_oak : ℕ := 4

/-- The number of birch trees -/
def num_birch : ℕ := 5

/-- The total number of trees -/
def total_trees : ℕ := num_maple + num_oak + num_birch

/-- The probability of arranging the trees such that no two birch trees are adjacent -/
theorem prob_no_adjacent_birch : 
  (Nat.factorial (total_trees - num_birch) * Nat.choose (total_trees - num_birch + 1) num_birch * Nat.factorial num_birch) / 
  Nat.factorial total_trees = 7 / 99 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_no_adjacent_birch_l446_44655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l446_44633

/-- The sum of the infinite series 1/(4^1) + 2/(4^2) + 3/(4^3) + ... + k/(4^k) + ... -/
noncomputable def infiniteSeries : ℝ := ∑' k, (k : ℝ) / (4 ^ k)

/-- Theorem: The sum of the infinite series is equal to 4/9 -/
theorem infiniteSeriesSum : infiniteSeries = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l446_44633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paperclip_ratio_l446_44639

def initial_yun : ℕ := 20
def lost_yun : ℕ := 12
def marion_count : ℕ := 9

def current_yun : ℕ := initial_yun - lost_yun

theorem paperclip_ratio :
  ∃ (f : ℚ), 
    (f * (current_yun : ℚ) + 7 = marion_count) ∧
    ((marion_count - current_yun) : ℚ) / current_yun = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paperclip_ratio_l446_44639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_l446_44669

-- Define the volume of the cone
noncomputable def cone_volume : ℝ := 16384 * Real.pi

-- Define the vertex angle of the vertical cross section
def vertex_angle : ℝ := 90

-- Theorem statement
theorem cone_height (h : ℝ) (h_positive : h > 0) : 
  (cone_volume = (1/3) * Real.pi * h^3) → 
  (vertex_angle = 90) → 
  h = (49152 : ℝ)^(1/3) :=
by
  intro volume_eq angle_eq
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check cone_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_l446_44669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l446_44623

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - x + 3) / Real.log a

theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 2 4, StrictMono (f a)) →
    a ∈ Set.Ioo (1/16) (1/8) ∪ Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l446_44623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_placement_possible_l446_44672

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a unit cube in 3D space -/
structure UnitCube where
  center : Point3D

/-- The main theorem stating that a sphere can always be placed in the prism without overlapping cubes -/
theorem sphere_placement_possible (cubes : Finset UnitCube) 
    (h_count : cubes.card = 9) 
    (h_non_overlapping : ∀ c1 c2, c1 ∈ cubes → c2 ∈ cubes → c1 ≠ c2 → 
      (c1.center.x - c2.center.x)^2 + (c1.center.y - c2.center.y)^2 + (c1.center.z - c2.center.z)^2 ≥ 1) :
  ∃ p : Point3D, 
    (1 ≤ p.x ∧ p.x ≤ 9) ∧ 
    (1 ≤ p.y ∧ p.y ≤ 7) ∧ 
    (1 ≤ p.z ∧ p.z ≤ 5) ∧
    ∀ c, c ∈ cubes → (p.x - c.center.x)^2 + (p.y - c.center.y)^2 + (p.z - c.center.z)^2 > 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_placement_possible_l446_44672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_problem_l446_44607

/-- Represents the number of students in a class -/
structure ClassSize where
  total : Nat
  boys : Nat
  girls : Nat
  total_eq : total = boys + girls

/-- The data for the school classes -/
structure SchoolData where
  classes : Finset ClassSize
  total_students : Nat
  responses : Finset Nat

/-- The conditions of the problem -/
def ValidSchoolData (data : SchoolData) : Prop :=
  data.classes.card = 4 ∧
  data.total_students > 70 ∧
  data.total_students = data.classes.sum (λ c => c.total) ∧
  data.responses = data.classes.image (λ c => c.total) ∪ data.classes.image (λ c => c.boys) ∧
  data.responses = {7, 9, 10, 12, 15, 16, 19, 21}

/-- The theorem to be proved -/
theorem school_problem (data : SchoolData) (h : ValidSchoolData data) :
  (∃ c ∈ data.classes, c.total = 21) ∧
  (data.classes.sum (λ c => c.girls) = 33) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_problem_l446_44607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounded_to_331_accuracy_l446_44612

/-- A number rounded to 3.31 is accurate to two decimal places. -/
theorem rounded_to_331_accuracy : 
  ∀ (x : ℝ), (round x = (3.31 : ℝ)) → (∃ (y : ℝ), abs (x - y) < 0.005 ∧ round y = (3.31 : ℝ)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounded_to_331_accuracy_l446_44612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_F_l446_44648

variable (a b m : ℝ)
variable (f : ℝ → ℝ)

def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 3

theorem sum_of_max_min_F (h1 : ∀ x ∈ Set.Icc a b, f (-x) = -f x)
                         (h2 : ∀ x ∈ Set.Icc a b, f x ≤ m)
                         (h3 : ∃ x ∈ Set.Icc a b, f x = m) :
  (⨆ x ∈ Set.Icc a b, F f x) + (⨅ x ∈ Set.Icc a b, F f x) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_F_l446_44648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_after_transfer_l446_44608

-- Define the classes
def class_one : Type := Unit
def class_two : Type := Unit

-- Define the initial states
variable (n₁ n₂ : ℕ) -- Number of students in each class
variable (H₁ H₂ : ℚ) -- Initial average heights

-- Define the heights of the transferred students
def h_lei_lei : ℕ := 158
def h_rong_rong : ℕ := 140

-- Define the changes in average heights
def change_class_one : ℚ := 2
def change_class_two : ℚ := -3

-- State the theorem
theorem total_students_after_transfer :
  (n₁ * (H₁ + change_class_one) = n₁ * H₁ - h_rong_rong + h_lei_lei) →
  (n₂ * (H₂ + change_class_two) = n₂ * H₂ - h_lei_lei + h_rong_rong) →
  n₁ + n₂ = 15 := by
  sorry

#check total_students_after_transfer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_after_transfer_l446_44608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_1296_l446_44629

theorem largest_prime_factor_of_1296 :
  (Nat.factors 1296).maximum? = some 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_1296_l446_44629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_floor_divisibility_l446_44668

theorem binomial_floor_divisibility (n : ℕ) (h : n > 7) :
  (Nat.choose n 7 - Int.floor (n / 7 : ℚ)) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_floor_divisibility_l446_44668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l446_44631

theorem triangle_properties (A B C : Real) (a b c : Real) :
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  b > 0 →
  c > 0 →
  Real.cos A = 1/2 →
  b = 2 →
  c = 3 →
  A = Real.pi/3 ∧ a = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l446_44631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cars_ac_no_stripes_l446_44622

/-- Given a group of cars, proves that the maximum number of cars with air conditioning but without racing stripes is 0 under specific conditions. -/
theorem max_cars_ac_no_stripes (total : ℕ) (no_ac : ℕ) (min_stripes : ℕ) 
  (h_total : total = 100)
  (h_no_ac : no_ac = 49)
  (h_min_stripes : min_stripes = 51)
  (h_stripes_ge : min_stripes ≤ total - no_ac) : 
  0 = max (total - no_ac - min_stripes) 0 := by
  sorry

#check max_cars_ac_no_stripes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cars_ac_no_stripes_l446_44622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_tub_time_l446_44617

/-- The time in seconds for eight faucets to fill a 50-gallon tub, given that four faucets fill a 150-gallon tub in 8 minutes and all faucets dispense water at the same rate. -/
noncomputable def time_to_fill_tub (tub_size_four_faucets : ℝ) (time_four_faucets : ℝ) (tub_size_eight_faucets : ℝ) : ℝ :=
  let rate_per_faucet := tub_size_four_faucets / (4 * time_four_faucets)
  let total_rate_eight_faucets := 8 * rate_per_faucet
  (tub_size_eight_faucets / total_rate_eight_faucets) * 60

/-- Theorem stating that eight faucets will fill a 50-gallon tub in 80 seconds, given the conditions. -/
theorem fill_tub_time :
  time_to_fill_tub 150 8 50 = 80 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_to_fill_tub 150 8 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_tub_time_l446_44617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lena_contributes_nothing_l446_44661

/-- The amount Lena needs to contribute for the coffee and pastry set. -/
noncomputable def lenas_contribution (set_cost : ℝ) (masons_money : ℝ) (exchange_rate : ℝ) : ℝ :=
  max 0 (set_cost - masons_money / exchange_rate)

/-- Theorem stating that Lena's contribution is 0 euros under the given conditions. -/
theorem lena_contributes_nothing :
  let set_cost : ℝ := 8
  let masons_money : ℝ := 10
  let exchange_rate : ℝ := 1.1
  lenas_contribution set_cost masons_money exchange_rate = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lena_contributes_nothing_l446_44661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l446_44620

-- Define the set M
def M : Set ℝ := {x | x > 3 ∨ x < 1}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2:ℝ)^x + 2 - 3 * (4:ℝ)^x

-- Theorem statement
theorem max_value_of_f :
  ∃ (max : ℝ), max = 25/12 ∧ ∀ (x : ℝ), x ∈ M → f x ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l446_44620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_distribution_l446_44681

theorem book_distribution (n : ℕ) (k : ℕ) : n = 5 ∧ k = 3 →
  (Nat.choose n 3 * Nat.factorial 3) + (Nat.choose n 2 * Nat.choose 3 2 * Nat.factorial 3 / Nat.factorial 2) = 150 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_distribution_l446_44681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crossSectionArea_formula_l446_44651

/-- A regular triangular prism inscribed in a sphere -/
structure InscribedPrism where
  R : ℝ  -- radius of the sphere
  a : ℝ  -- side length of the prism's base
  h : 0 < R ∧ 0 < a ∧ a < 2*R  -- conditions for valid inscribed prism

/-- The area of the cross-section of the prism -/
noncomputable def crossSectionArea (p : InscribedPrism) : ℝ :=
  (2 * p.a / 3) * Real.sqrt (4 * p.R^2 - p.a^2)

/-- Theorem: The area of the cross-section is (2a/3) * sqrt(4R^2 - a^2) -/
theorem crossSectionArea_formula (p : InscribedPrism) :
  crossSectionArea p = (2 * p.a / 3) * Real.sqrt (4 * p.R^2 - p.a^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crossSectionArea_formula_l446_44651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_coloring_exists_l446_44647

/-- A coloring function for the integer lattice points -/
def ColoringFunction := ℤ → ℤ → Fin 3

/-- Predicate to check if a coloring function satisfies the infinite occurrence condition -/
def InfiniteOccurrence (f : ColoringFunction) : Prop :=
  ∀ c : Fin 3, ∀ y : ℤ, ∀ n : ℕ, ∃ x : ℤ, x > n ∧ f x y = c

/-- Predicate to check if three points are collinear -/
def AreCollinear (p₁ p₂ p₃ : ℤ × ℤ) : Prop :=
  let (x₁, y₁) := p₁
  let (x₂, y₂) := p₂
  let (x₃, y₃) := p₃
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Predicate to check if a coloring function satisfies the non-collinearity condition -/
def NonCollinear (f : ColoringFunction) : Prop :=
  ∀ p₁ p₂ p₃ : ℤ × ℤ,
    f p₁.1 p₁.2 ≠ f p₂.1 p₂.2 →
    f p₂.1 p₂.2 ≠ f p₃.1 p₃.2 →
    f p₃.1 p₃.2 ≠ f p₁.1 p₁.2 →
    ¬AreCollinear p₁ p₂ p₃

/-- The main theorem stating that a valid coloring function exists -/
theorem valid_coloring_exists : ∃ f : ColoringFunction, InfiniteOccurrence f ∧ NonCollinear f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_coloring_exists_l446_44647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_compounding_difference_l446_44664

/-- Calculates the future value of an investment with compound interest -/
noncomputable def futureValue (principal : ℝ) (rate : ℝ) (time : ℝ) (compoundingsPerYear : ℝ) : ℝ :=
  principal * (1 + rate / compoundingsPerYear) ^ (time * compoundingsPerYear)

/-- The difference in earnings between monthly and yearly compounding -/
noncomputable def earningsDifference : ℝ :=
  let principal := 70000
  let rate := 0.05
  let time := 3
  futureValue principal rate time 12 - futureValue principal rate time 1

theorem investment_compounding_difference :
  abs (earningsDifference - 263.71) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_compounding_difference_l446_44664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l446_44652

def y_A (x : ℝ) : ℝ := 2 * x
def y_B (x : ℝ) : ℝ := -x^2 + 10 * x

theorem investment_problem :
  (y_A 10 = 20) ∧
  (∃ m : ℝ, m > 0 ∧ y_A m = y_B m ∧ m = 8) ∧
  (let f : ℝ → ℝ := λ n ↦ y_A (32 - n) + y_B n;
   ∃ max_profit : ℝ,
     (f 4 = max_profit) ∧
     (∀ n : ℝ, 0 ≤ n ∧ n ≤ 32 → f n ≤ max_profit) ∧
     max_profit = 80) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l446_44652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_Z_l446_44698

-- Define set A
def A : Set ℝ := {x | x^2 < 3*x + 4}

-- Define the theorem
theorem A_intersect_Z : A ∩ (Set.range (Int.cast : ℤ → ℝ)) = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_Z_l446_44698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_3_plus_2sqrt2_to_4th_l446_44686

theorem nearest_integer_to_3_plus_2sqrt2_to_4th :
  ∃ n : ℤ, n = 1090 ∧ |(3 + 2 * Real.sqrt 2)^4 - ↑n| < 1/2 :=
by
  -- We claim that n = 1090 is the nearest integer
  use 1090
  constructor
  -- First part: n = 1090
  · rfl
  -- Second part: |(3 + 2√2)^4 - 1090| < 1/2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_3_plus_2sqrt2_to_4th_l446_44686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_property_l446_44637

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- Define an arithmetic sequence for three terms
def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem geometric_sequence_ratio_property
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : is_geometric_sequence a)
  (h_arith : is_arithmetic_sequence (a 1) ((1/2) * a 3) (2 * a 2)) :
  a 10 / a 8 = 3 + 2 * Real.sqrt 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_property_l446_44637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_grid_exists_l446_44659

/-- Represents the possible contents of a cell in the grid -/
inductive Cell
| P
| E
| N1
| N2
| Y
| Blank

/-- Represents a 5x5 grid -/
def Grid := Fin 5 → Fin 5 → Cell

/-- Checks if a row or column contains exactly one of each letter and one blank -/
def isValidLine (line : Fin 5 → Cell) : Prop :=
  (∃! i, line i = Cell.P) ∧
  (∃! i, line i = Cell.E) ∧
  (∃! i, line i = Cell.N1) ∧
  (∃! i, line i = Cell.N2) ∧
  (∃! i, line i = Cell.Y) ∧
  (∃! i, line i = Cell.Blank)

/-- Checks if the grid satisfies the exterior letter constraints -/
def satisfiesExteriorConstraints (grid : Grid) : Prop :=
  (grid 0 0 = Cell.E) ∧
  (grid 0 2 = Cell.P) ∧
  (grid 0 4 = Cell.Y) ∧
  (grid 1 0 = Cell.N1) ∧
  (grid 2 2 = Cell.P) ∧
  (grid 3 2 = Cell.E) ∧
  (grid 4 2 = Cell.Y) ∧
  (grid 4 0 = Cell.Y) ∧
  (grid 4 1 = Cell.N2) ∧
  (grid 4 2 = Cell.E) ∧
  (grid 4 3 = Cell.N1) ∧
  (grid 4 4 = Cell.P)

/-- The main theorem stating that there exists a unique valid grid -/
theorem unique_valid_grid_exists :
  ∃! grid : Grid,
    (∀ i : Fin 5, isValidLine (λ j ↦ grid i j)) ∧
    (∀ j : Fin 5, isValidLine (λ i ↦ grid i j)) ∧
    satisfiesExteriorConstraints grid := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_grid_exists_l446_44659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_third_side_l446_44609

theorem right_triangle_third_side 
  (a b : ℤ) 
  (c : ℝ) 
  (ha : a = Int.floor (Real.sqrt 6))
  (hbc : 2 + Real.sqrt 6 = b + c)
  (hb : b ∈ Set.range Int.cast)
  (hc : 0 < c ∧ c < 1) :
  ∃ (x : ℝ), (x = 2 * Real.sqrt 5 ∨ x = 2 * Real.sqrt 3) ∧
    (x^2 = (a : ℝ)^2 + (b : ℝ)^2 ∨ x^2 = (b : ℝ)^2 - (a : ℝ)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_third_side_l446_44609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_v_shaped_square_v_shaped_log_square_plus_two_v_shaped_log_exp_plus_a_l446_44693

-- Define V-shaped function
def isVShaped (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f (x₁ + x₂) ≤ f x₁ + f x₂

-- Define the three functions
def f₁ (x : ℝ) : ℝ := x^2

noncomputable def f₂ (x : ℝ) : ℝ := Real.log (x^2 + 2)

noncomputable def f₃ (a : ℝ) (x : ℝ) : ℝ := Real.log (2^x + a)

-- Theorem statements
theorem not_v_shaped_square : ¬(isVShaped f₁) := by sorry

theorem v_shaped_log_square_plus_two : isVShaped f₂ := by sorry

theorem v_shaped_log_exp_plus_a (a : ℝ) : 
  isVShaped (f₃ a) ↔ (a ≥ 1 ∨ a = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_v_shaped_square_v_shaped_log_square_plus_two_v_shaped_log_exp_plus_a_l446_44693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_midpoint_l446_44663

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- The focus of a parabola -/
noncomputable def focus (c : Parabola) : Point :=
  { x := c.p / 2, y := 0 }

/-- A point on the parabola -/
noncomputable def pointOnParabola (c : Parabola) (y : ℝ) : Point :=
  { x := y^2 / (2 * c.p), y := y }

/-- The midpoint of two points -/
noncomputable def midpointOfPoints (a b : Point) : Point :=
  { x := (a.x + b.x) / 2, y := (a.y + b.y) / 2 }

theorem parabola_focus_midpoint (c : Parabola) (m : Point) 
  (h1 : m.y^2 = 2 * c.p * m.x)  -- m is on the parabola
  (h2 : midpointOfPoints m (focus c) = Point.mk 2 2)  -- midpoint of MF is (2, 2)
  : c.p = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_midpoint_l446_44663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_max_value_on_interval_inequality_range_l446_44613

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x^2

-- Theorem 1
theorem tangent_line_values (a b : ℝ) :
  (∀ x, f a b x ≥ -1/2) ∧ (f a b 1 = -1/2) ∧ (∀ x, x ≠ 1 → f a b x > -1/2) →
  a = 1 ∧ b = 1/2 := by sorry

-- Theorem 2
theorem max_value_on_interval :
  let f := f 1 (1/2)
  ∀ x ∈ Set.Icc (Real.exp (-1)) (Real.exp 1), f x ≤ -1/2 := by sorry

-- Theorem 3
theorem inequality_range (m : ℝ) :
  (∀ a ∈ Set.Icc 0 (3/2), ∀ x ∈ Set.Ioo 1 (Real.exp 2), f a 0 x ≥ m + x) →
  m ≤ -(Real.exp 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_max_value_on_interval_inequality_range_l446_44613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_in_triangle_arrangement_l446_44635

-- Define a triangle as a structure with three angles
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180

-- Define the arrangement of three triangles
def TriangleArrangement (t1 t2 t3 : Triangle) : Prop :=
  -- Each triangle is distinct
  t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧
  -- Each triangle shares at least one angle with another
  (t1.angle1 = t2.angle1 ∨ t1.angle1 = t2.angle2 ∨ t1.angle1 = t2.angle3 ∨
   t1.angle2 = t2.angle1 ∨ t1.angle2 = t2.angle2 ∨ t1.angle2 = t2.angle3 ∨
   t1.angle3 = t2.angle1 ∨ t1.angle3 = t2.angle2 ∨ t1.angle3 = t2.angle3) ∧
  (t1.angle1 = t3.angle1 ∨ t1.angle1 = t3.angle2 ∨ t1.angle1 = t3.angle3 ∨
   t1.angle2 = t3.angle1 ∨ t1.angle2 = t3.angle2 ∨ t1.angle2 = t3.angle3 ∨
   t1.angle3 = t3.angle1 ∨ t1.angle3 = t3.angle2 ∨ t1.angle3 = t3.angle3) ∧
  (t2.angle1 = t3.angle1 ∨ t2.angle1 = t3.angle2 ∨ t2.angle1 = t3.angle3 ∨
   t2.angle2 = t3.angle1 ∨ t2.angle2 = t3.angle2 ∨ t2.angle2 = t3.angle3 ∨
   t2.angle3 = t3.angle1 ∨ t2.angle3 = t3.angle2 ∨ t2.angle3 = t3.angle3)

-- Define the function to count distinct angles
noncomputable def countDistinctAngles (t1 t2 t3 : Triangle) : ℕ :=
  let allAngles := [t1.angle1, t1.angle2, t1.angle3, t2.angle1, t2.angle2, t2.angle3, t3.angle1, t3.angle2, t3.angle3]
  (allAngles.toFinset).card

-- Theorem statement
theorem sum_of_angles_in_triangle_arrangement (t1 t2 t3 : Triangle) 
  (h : TriangleArrangement t1 t2 t3) 
  (h_distinct : countDistinctAngles t1 t2 t3 = 9) : 
  t1.angle1 + t1.angle2 + t1.angle3 + 
  t2.angle1 + t2.angle2 + t2.angle3 + 
  t3.angle1 + t3.angle2 + t3.angle3 = 360 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_in_triangle_arrangement_l446_44635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_to_breadth_ratio_is_three_to_one_l446_44656

/-- Represents a rectangular plot -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area : ℝ

/-- The ratio of length to breadth for a rectangular plot -/
noncomputable def lengthToBreadthRatio (plot : RectangularPlot) : ℝ :=
  plot.length / plot.breadth

/-- Theorem: For a rectangular plot with area 2700 sq m and breadth 30 m, 
    the ratio of length to breadth is 3:1 -/
theorem length_to_breadth_ratio_is_three_to_one 
  (plot : RectangularPlot) 
  (h_area : plot.area = 2700)
  (h_breadth : plot.breadth = 30)
  (h_length_multiple : ∃ k : ℝ, plot.length = k * plot.breadth)
  (h_area_formula : plot.area = plot.length * plot.breadth) : 
  lengthToBreadthRatio plot = 3 := by
  sorry

#check length_to_breadth_ratio_is_three_to_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_to_breadth_ratio_is_three_to_one_l446_44656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_unit_vectors_in_halfplane_l446_44627

/-- A vector in ℝ² -/
def Vector2D := ℝ × ℝ

/-- The magnitude (length) of a vector -/
noncomputable def magnitude (v : Vector2D) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

/-- Check if a vector is in the upper half-plane (y ≥ 0) -/
def inUpperHalfPlane (v : Vector2D) : Prop :=
  v.2 ≥ 0

/-- Sum of a list of vectors -/
def vectorSum (vs : List Vector2D) : Vector2D :=
  vs.foldl (fun acc v => (acc.1 + v.1, acc.2 + v.2)) (0, 0)

/-- Main theorem -/
theorem sum_of_odd_unit_vectors_in_halfplane (n : ℕ) (vs : List Vector2D) 
  (h_odd : vs.length = 2 * n + 1)
  (h_unit : ∀ v ∈ vs, magnitude v = 1)
  (h_halfplane : ∀ v ∈ vs, inUpperHalfPlane v) :
  magnitude (vectorSum vs) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_unit_vectors_in_halfplane_l446_44627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_hypotenuse_l446_44619

/-- Predicate indicating that a set of points forms a right triangle -/
def RightTriangle (triangle : Set ℝ) : Prop := sorry

/-- Predicate indicating that two triangles are similar -/
def Similar (triangle1 triangle2 : Set ℝ) : Prop := sorry

/-- Predicate indicating that a given length is the length of a leg in a triangle -/
def LegLength (triangle : Set ℝ) (length : ℝ) : Prop := sorry

/-- Predicate indicating that a given length is the length of the hypotenuse in a triangle -/
def HypotenuseLength (triangle : Set ℝ) (length : ℝ) : Prop := sorry

/-- Given two similar right triangles, where the first triangle has a leg of 15 inches
    and a hypotenuse of 39 inches, and the second triangle has a corresponding leg of 45 inches,
    the hypotenuse of the second triangle is 48.75 inches. -/
theorem similar_triangles_hypotenuse (triangle1 triangle2 : Set ℝ) :
  RightTriangle triangle1 →
  RightTriangle triangle2 →
  Similar triangle1 triangle2 →
  (∃ (leg1 hyp1 : ℝ), LegLength triangle1 leg1 ∧ HypotenuseLength triangle1 hyp1 ∧ leg1 = 15 ∧ hyp1 = 39) →
  (∃ (leg2 : ℝ), LegLength triangle2 leg2 ∧ leg2 = 45) →
  (∃ (hyp2 : ℝ), HypotenuseLength triangle2 hyp2 ∧ hyp2 = 48.75) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_hypotenuse_l446_44619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_is_981_l446_44673

/-- Represents the sequence of positive integers formed by powers of 3 or sums of different powers of 3 -/
def powerOf3Sequence : ℕ → ℕ := sorry

/-- The 100th term of the sequence -/
def hundredthTerm : ℕ := powerOf3Sequence 100

/-- Theorem stating that the 100th term of the sequence is 981 -/
theorem hundredth_term_is_981 : hundredthTerm = 981 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_is_981_l446_44673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_triangle_side_length_l446_44676

noncomputable section

open Real

def acute_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2

def vector_m (A : ℝ) : ℝ × ℝ :=
  (cos (A + Real.pi/3), sin (A + Real.pi/3))

def vector_n (B : ℝ) : ℝ × ℝ :=
  (cos B, sin B)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem triangle_property (A B C : ℝ) :
  acute_triangle A B C →
  perpendicular (vector_m A) (vector_n B) →
  A - B = Real.pi/6 :=
by sorry

theorem triangle_side_length (A B C : ℝ) (AC : ℝ) :
  acute_triangle A B C →
  perpendicular (vector_m A) (vector_n B) →
  cos B = 3/5 →
  AC = 8 →
  ∃ (BC : ℝ), BC = 4 * sqrt 3 + 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_triangle_side_length_l446_44676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_fiftieth_term_is_2280_l446_44654

-- Define the sequence of powers of 3 and sums of distinct powers of 3
def sequencePowersOfThree : ℕ → ℕ
| n => sorry

-- Define the function to convert a natural number to its binary representation
def toBinary : ℕ → List Bool
| n => sorry

-- Define the function to calculate the sum of powers of 3 based on binary representation
def sumPowersOfThree : List Bool → ℕ
| bits => sorry

-- Theorem statement
theorem hundred_fiftieth_term_is_2280 : sequencePowersOfThree 150 = 2280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_fiftieth_term_is_2280_l446_44654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_apex_angle_l446_44682

theorem isosceles_triangle_apex_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) :
  Real.sin α = Real.sqrt 5 / 5 →
  Real.cos (π - 2 * α) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_apex_angle_l446_44682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_hemisphere_cube_sphere_tetrahedron_l446_44657

theorem volume_ratio_hemisphere_cube_sphere_tetrahedron (r : ℝ) (hr : r > 0) :
  ∃ (k : ℝ), k > 0 ∧
    (2 / 3) * Real.pi * r^3 = k * (27 * Real.pi * Real.sqrt 2) ∧
    ((2 * r) / Real.sqrt 3)^3 = k * (18 * Real.sqrt 3) ∧
    (4 / 3) * Real.pi * (r / Real.sqrt 3)^3 = k * (3 * Real.pi * Real.sqrt 3) ∧
    (((4 * r) / (3 * Real.sqrt 2))^3 * Real.sqrt 2) / 12 = k * 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_hemisphere_cube_sphere_tetrahedron_l446_44657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_isosceles_triangle_l446_44632

/-- The radius of the inscribed circle in an isosceles triangle -/
theorem inscribed_circle_radius_isosceles_triangle (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  (let s := (2 * a + b) / 2
   let area := Real.sqrt (s * (s - a) * (s - a) * (s - b))
   area / s = (5 * Real.sqrt 39) / 13) → a = 8 ∧ b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_isosceles_triangle_l446_44632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_arc_length_l446_44625

/-- The length of an arc in a circular sector --/
noncomputable def arcLength (centralAngle : ℝ) (radius : ℝ) : ℝ := 
  (centralAngle * Real.pi * radius) / 180

/-- Theorem: The arc length of a sector with a 60° central angle and radius 9 is 3π --/
theorem sector_arc_length : arcLength 60 9 = 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_arc_length_l446_44625
