import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_and_solution_set_l904_90409

noncomputable def f (x : ℝ) := 9 / (Real.sin x)^2 + 4 / (Real.cos x)^2

theorem max_t_and_solution_set :
  (∃ (t : ℝ), ∀ (x : ℝ), x ∈ Set.Ioo 0 (Real.pi / 2) → f x ≥ t ∧
    (∀ (t' : ℝ), (∀ (x : ℝ), x ∈ Set.Ioo 0 (Real.pi / 2) → f x ≥ t') → t' ≤ t) ∧
    t = 25) ∧
  {x : ℝ | |x + 5| + |2*x - 1| ≤ 6} = Set.Icc 0 (2/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_and_solution_set_l904_90409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_loss_ratio_l904_90421

/-- Proves that the ratio of Jerome's weight loss to Seth's weight loss is 3:1 given the conditions of the problem -/
theorem weight_loss_ratio : 
  ∃ x : ℝ, 
    let seth_loss : ℝ := 17.5
    let jerome_loss : ℝ := seth_loss * x
    let veronica_loss : ℝ := seth_loss + 1.5
    let total_loss : ℝ := 89
    seth_loss + jerome_loss + veronica_loss = total_loss ∧ 
    jerome_loss / seth_loss = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_loss_ratio_l904_90421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l904_90406

/-- Predicate to check if an equation is quadratic in x -/
noncomputable def IsQuadraticInX (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 2 -/
noncomputable def eqD (x : ℝ) : ℝ := x^2 - 2

/-- The equation 3x^2 = 1 - 1/(3x) -/
noncomputable def eqA (x : ℝ) : ℝ := 3 * x^2 - 1 + 1 / (3 * x)

/-- The equation (m-2)x^2 - mx + 3 = 0, parameterized by m -/
noncomputable def eqB (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 - m * x + 3

/-- The equation (x^2 - 3)(x - 1) = 0 -/
noncomputable def eqC (x : ℝ) : ℝ := (x^2 - 3) * (x - 1)

theorem quadratic_equation_identification :
  IsQuadraticInX eqD ∧
  ¬IsQuadraticInX eqA ∧
  (∃ m : ℝ, ¬IsQuadraticInX (eqB m)) ∧
  ¬IsQuadraticInX eqC :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l904_90406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_four_theta_l904_90401

theorem cos_four_theta (θ : ℝ) (h : Real.cos θ = 1/2) : Real.cos (4 * θ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_four_theta_l904_90401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rain_probability_theorem_l904_90441

/-- The probability of rain on a single day in July -/
noncomputable def p : ℝ := 1 / 5

/-- The number of days in July -/
def n : ℕ := 31

/-- The probability of rain on at most 3 days in July -/
noncomputable def prob_at_most_3 : ℝ :=
  (1 - p)^n +
  n * p * (1 - p)^(n-1) +
  (n * (n-1) / 2) * p^2 * (1 - p)^(n-2) +
  (n * (n-1) * (n-2) / 6) * p^3 * (1 - p)^(n-3)

/-- The approximate probability of rain on at most 3 days in July -/
def approx_prob : ℝ := 0.338

theorem rain_probability_theorem : 
  |prob_at_most_3 - approx_prob| < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rain_probability_theorem_l904_90441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_expression_l904_90494

/-- The golden ratio -/
noncomputable def t : ℝ := (Real.sqrt 5 - 1) / 2

/-- The expression to be evaluated -/
noncomputable def f (t : ℝ) : ℝ := (1 - 2 * Real.sin (27 * Real.pi / 180) ^ 2) / (2 * t * Real.sqrt (4 - t^2))

/-- Theorem stating that the expression evaluates to 1/4 when t is the golden ratio -/
theorem golden_ratio_expression : f t = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_expression_l904_90494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jiho_initial_money_l904_90438

/-- The initial amount of money Jiho had -/
def M : ℕ := 30000

/-- The amount spent at the fish shop -/
def fish_shop_expense : ℕ := M / 2

/-- The amount remaining after the fish shop -/
def remaining_after_fish : ℕ := M - fish_shop_expense

/-- The amount spent at the fruit shop -/
def fruit_shop_expense : ℕ := remaining_after_fish / 3 + 5000

/-- The final amount remaining -/
def final_remaining : ℕ := remaining_after_fish - fruit_shop_expense

theorem jiho_initial_money :
  final_remaining = 5000 → M = 30000 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jiho_initial_money_l904_90438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_roots_and_vector_angle_l904_90488

/-- Given that z₁ = 1 + √2i is a root of x² + mx + n = 0, prove m = -2 and n = 3,
    and find the range of t for which the angle between t*a + b and a + t*b is acute,
    where a and b are vectors corresponding to z₁ and z₂ respectively. -/
theorem complex_roots_and_vector_angle (m n t : ℝ) : 
  let z₁ : ℂ := 1 + Complex.I * Real.sqrt 2
  let z₂ : ℂ := 1 - Complex.I * Real.sqrt 2
  let a : ℝ × ℝ := (1, Real.sqrt 2)
  let b : ℝ × ℝ := (1, -Real.sqrt 2)
  (z₁^2 + m*z₁ + n = 0) →
  (m = -2 ∧ n = 3) ∧
  (((t * a.1 + b.1) * (a.1 + t * b.1) + (t * a.2 + b.2) * (a.2 + t * b.2) > 0) ↔ 
    (3 - 2 * Real.sqrt 2 < t ∧ t < 1) ∨ (1 < t ∧ t < 3 + 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_roots_and_vector_angle_l904_90488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_T_l904_90420

-- Define the solid T
def T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p;
                    (|x| + |y| ≤ 2) ∧ (|x| + |z| ≤ 2) ∧ (|y| + |z| ≤ 2)}

-- State the theorem about the volume of T
theorem volume_of_T : MeasureTheory.volume T = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_T_l904_90420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_from_triangles_l904_90463

/-- Represents an isosceles right-angled triangle -/
structure IsoscelesRightTriangle where
  side : ℝ
  hypotenuse : ℝ
  hypotenuse_eq : hypotenuse = side * Real.sqrt 2

/-- Represents a square formed by isosceles right-angled triangles -/
structure SquareFromTriangles where
  side_length : ℝ
  num_triangles : ℕ

/-- The maximum number of different-sized squares that can be formed -/
def max_different_squares : ℕ := 8

/-- The total number of available triangles -/
def total_triangles : ℕ := 52

/-- Theorem stating that the maximum number of different-sized squares is 8 -/
theorem max_squares_from_triangles (squares : List SquareFromTriangles) :
  (∀ s, s ∈ squares → s.num_triangles ≤ total_triangles) →
  (∀ s1 s2, s1 ∈ squares → s2 ∈ squares → s1 ≠ s2 → s1.side_length ≠ s2.side_length) →
  squares.length ≤ max_different_squares :=
by
  sorry

#check max_squares_from_triangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_from_triangles_l904_90463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_theorem_l904_90435

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a 4-digit number -/
def FourDigitNum := Nat

/-- Represents a 3-digit number -/
def ThreeDigitNum := Nat

theorem digit_sum_theorem 
  (A B C D E F G : Digit)
  (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧
        B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧
        C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧
        D ≠ E ∧ D ≠ F ∧ D ≠ G ∧
        E ≠ F ∧ E ≠ G ∧
        F ≠ G)
  (h2 : (A.val * 1000 + B.val * 100 + C.val * 10 + D.val : FourDigitNum) +
        (E.val * 100 + F.val * 10 + G.val : ThreeDigitNum) = 2020) :
  A.val + B.val + C.val + D.val + E.val + F.val + G.val = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_theorem_l904_90435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duck_family_snails_theorem_l904_90425

def duck_family_snails (total_ducklings first_group_size second_group_size first_group_snails second_group_snails : ℕ) : ℕ :=
  let remaining_ducklings := total_ducklings - first_group_size - second_group_size
  let first_group_total := first_group_size * first_group_snails
  let second_group_total := second_group_size * second_group_snails
  let first_two_groups_total := first_group_total + second_group_total
  let mother_duck_snails := 3 * first_two_groups_total
  let remaining_ducklings_snails := remaining_ducklings * (mother_duck_snails / 2)
  first_two_groups_total + mother_duck_snails + remaining_ducklings_snails

theorem duck_family_snails_theorem : 
  duck_family_snails 8 3 3 5 9 = 294 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_duck_family_snails_theorem_l904_90425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_105_106_l904_90443

def f (x : Int) : Int := x^2 - x + 2023

theorem gcd_f_105_106 : Int.gcd (f 105) (f 106) = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_105_106_l904_90443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l904_90432

/-- The area of a parallelogram with vertices at (0, 0), (4, 0), (5, 10), and (1, 10) is 40 square units. -/
theorem parallelogram_area : ∃ (area : ℝ), area = 40 := by
  -- Define the vertices of the parallelogram
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (4, 0)
  let v3 : ℝ × ℝ := (5, 10)
  let v4 : ℝ × ℝ := (1, 10)

  -- Define the base and height
  let base : ℝ := v2.1 - v1.1
  let height : ℝ := v4.2 - v1.2

  -- Calculate the area
  let area : ℝ := base * height

  -- Prove that the area is equal to 40
  have h : area = 40 := by
    -- Computation steps
    sorry

  -- Conclude the proof
  exact ⟨area, h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l904_90432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l904_90497

noncomputable def f (x : ℝ) := Real.sin x * Real.cos x - Real.cos x ^ 2

theorem f_properties :
  (∀ x ∈ Set.Ioo 0 (π/8), ∀ y ∈ Set.Ioo 0 (π/8), x < y → f x < f y) ∧
  (∀ x : ℝ, f (3*π/8 + x) = f (3*π/8 - x)) ∧
  (∀ x : ℝ, f (π/4 + x) + f (-x) = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l904_90497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_cost_is_54_cents_l904_90452

-- Define the cost of a pen and a pencil as rational numbers (ℚ)
variable (pen_cost pencil_cost : ℚ)

-- Define the conditions as propositions
def condition1 (pen_cost pencil_cost : ℚ) : Prop := 3 * pen_cost + 4 * pencil_cost = 316
def condition2 (pen_cost pencil_cost : ℚ) : Prop := 5 * pen_cost + 2 * pencil_cost = 348

-- Theorem statement
theorem pen_cost_is_54_cents 
  (h1 : condition1 pen_cost pencil_cost) (h2 : condition2 pen_cost pencil_cost) : 
  ⌊pen_cost⌋ = 54 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_cost_is_54_cents_l904_90452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l904_90474

noncomputable def f (x : ℝ) := Real.exp x + x^2 - 4

theorem root_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l904_90474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_visitors_average_l904_90456

/-- Calculates the average number of visitors per day in a 30-day month starting on Sunday -/
theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (h1 : sunday_visitors = 600) (h2 : other_day_visitors = 240) : 
  (4 * sunday_visitors + 26 * other_day_visitors) / 30 = 288 := by
  have num_sundays : ℕ := 4
  have num_other_days : ℕ := 26
  have total_visitors : ℕ := num_sundays * sunday_visitors + num_other_days * other_day_visitors
  have average_visitors : ℕ := total_visitors / 30
  sorry -- placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_visitors_average_l904_90456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_latin_squares_l904_90445

/-- A Latin square is a square matrix where each row and column contains 
    each number exactly once. -/
def is_latin_square (M : Matrix (Fin 7) (Fin 7) ℕ) : Prop :=
  ∀ i j : Fin 7, 
    (∀ k : Fin 7, ∃! x : Fin 7, M i x = k + 1) ∧ 
    (∀ k : Fin 7, ∃! y : Fin 7, M y j = k + 1)

/-- The number of distinct Latin squares obtainable by permuting rows and columns -/
def num_distinct_squares (M : Matrix (Fin 7) (Fin 7) ℕ) : ℕ :=
  (Nat.factorial 7) * (Nat.factorial 7)

/-- Theorem: The number of distinct 7x7 Latin squares obtainable by row and column 
    permutations from a given 7x7 Latin square is equal to (7!)^2 -/
theorem distinct_latin_squares 
  (M : Matrix (Fin 7) (Fin 7) ℕ) 
  (h : is_latin_square M) : 
  num_distinct_squares M = (Nat.factorial 7) * (Nat.factorial 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_latin_squares_l904_90445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_l904_90444

theorem no_real_solutions :
  ∀ x : ℝ, x^2 + 4*x + 4*x*Real.sqrt (x + 3) ≠ 17 := by
  intro x
  -- The proof will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_l904_90444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_angle_l904_90419

/-- Represents a parametric line. -/
structure ParametricLine where
  slopeAngle : ℝ

/-- Represents a parametric circle. -/
structure ParametricCircle where
  center : ℝ × ℝ
  radius : ℝ

/-- Defines the tangency relationship between a line and a circle. -/
def IsTangent (line : ParametricLine) (circle : ParametricCircle) : Prop :=
  sorry

/-- If a line is tangent to a circle, then its slope angle is either π/2 or π. -/
theorem tangent_line_slope_angle (line : ParametricLine) (circle : ParametricCircle) :
  IsTangent line circle → (line.slopeAngle = π / 2 ∨ line.slopeAngle = π) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_angle_l904_90419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_winning_figure_l904_90427

/-- A figure on graph paper represented as a set of cells --/
def Figure := Finset (ℕ × ℕ)

/-- A function that checks if a figure allows the starting player to always win --/
def always_win (f : Figure) : Prop := sorry

/-- The theorem stating that the smallest winning figure has 7 cells --/
theorem smallest_winning_figure :
  ∃ (f : Figure), always_win f ∧ f.card = 7 ∧
  ∀ (g : Figure), always_win g → g.card ≥ 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_winning_figure_l904_90427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrange_2025_l904_90464

def digits : List Nat := [2, 0, 2, 5]

def is_valid_permutation (perm : List Nat) : Bool :=
  perm.length = 4 && perm.headD 0 ≠ 0 && perm.toFinset = digits.toFinset

def count_valid_permutations : Nat :=
  (digits.permutations.filter is_valid_permutation).length

theorem rearrange_2025 : count_valid_permutations = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrange_2025_l904_90464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_subinterval_l904_90431

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the interval
noncomputable def a : ℝ := 2
noncomputable def b : ℝ := 4

-- Define the midpoint
noncomputable def x₁ : ℝ := (a + b) / 2

-- State the theorem
theorem root_in_subinterval 
  (hf : Continuous f)
  (h1 : f a * f b < 0)
  (h2 : f a * f x₁ < 0) :
  ∃ x₀ ∈ Set.Ioo a x₁, f x₀ = 0 :=
by
  sorry

#check root_in_subinterval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_subinterval_l904_90431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_chalk_pieces_l904_90487

theorem original_chalk_pieces (siblings friends lost_pieces added_pieces per_person : ℕ) :
  siblings + friends > 0 →
  ∃ (result : ℕ), (result - lost_pieces + added_pieces) = (siblings + friends) * per_person ∧ result = 11 :=
by
  intro h_positive
  use 11
  constructor
  · -- Proof of the equation
    sorry
  · -- Proof that result = 11
    rfl

#check original_chalk_pieces

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_chalk_pieces_l904_90487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_positive_period_of_f_l904_90454

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 * Real.sin x + Real.cos x) * (Real.sqrt 3 * Real.cos x - Real.sin x)

theorem minimum_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = π := by
  sorry

#check minimum_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_positive_period_of_f_l904_90454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l904_90482

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x^3

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1 - 3 * x^2

-- Theorem statement
theorem tangent_line_at_one :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, y = f' 1 * (x - 1) + f 1 ↔ a * x + b * y + c = 0) ∧
    a = 2 ∧ b = 1 ∧ c = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l904_90482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distinct_roots_of_polynomial_equation_l904_90475

/-- A polynomial with real coefficients -/
structure RealPolynomial where
  coeffs : List ℝ
  nonzero_last : coeffs ≠ []

/-- The degree of a polynomial -/
def degree (p : RealPolynomial) : ℕ := p.coeffs.length - 1

/-- The constant term of a polynomial -/
def constantTerm (p : RealPolynomial) : ℝ := p.coeffs.head!

/-- The number of distinct complex roots of a polynomial -/
noncomputable def distinctRoots (p : RealPolynomial) : ℕ := sorry

/-- Multiplication of RealPolynomials -/
noncomputable instance : HMul RealPolynomial RealPolynomial RealPolynomial where
  hMul := λ a b => ⟨sorry, sorry⟩

/-- Subtraction of RealPolynomials -/
noncomputable instance : HSub RealPolynomial RealPolynomial RealPolynomial where
  hSub := λ a b => ⟨sorry, sorry⟩

/-- The theorem to be proved -/
theorem min_distinct_roots_of_polynomial_equation :
  ∀ (A B C : RealPolynomial),
    degree A = 3 →
    degree B = 4 →
    degree C = 8 →
    constantTerm A = 4 →
    constantTerm B = 5 →
    constantTerm C = 9 →
    (∃ (M : ℕ), M = distinctRoots (C - A * B) ∧ ∀ (N : ℕ), N = distinctRoots (C - A * B) → M ≤ N) →
    1 ≤ distinctRoots (C - A * B) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distinct_roots_of_polynomial_equation_l904_90475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_film_length_approx_l904_90453

/-- The total length of a film wound around a reel -/
noncomputable def film_length (core_diameter : ℝ) (film_thickness : ℝ) (turns : ℕ) : ℝ :=
  (core_diameter * (turns : ℝ) + film_thickness * ((turns : ℝ) * ((turns : ℝ) - 1) / 2)) * Real.pi

theorem film_length_approx :
  let core_diameter : ℝ := 60
  let film_thickness : ℝ := 0.15
  let turns : ℕ := 600
  let pi_approx : ℝ := 3.14
  abs (film_length core_diameter film_thickness turns / 1000 - 282.3) < 0.1 :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_film_length_approx_l904_90453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_universal_number_min_digits_l904_90496

/-- A number is universal if any number composed of distinct digits can be obtained from it by deleting certain digits. -/
def IsUniversal (n : ℕ) : Prop :=
  ∀ m : ℕ, (∀ d₁ d₂ : ℕ, d₁ < 10 ∧ d₂ < 10 ∧ d₁ ∈ Nat.digits 10 m ∧ d₂ ∈ Nat.digits 10 m → d₁ ≠ d₂) →
    ∃ s : List ℕ, s.Sublist (Nat.digits 10 n) ∧ Nat.digits 10 m = s

/-- The minimum number of digits required for a universal number. -/
def MinUniversalDigits : ℕ := 55

/-- Theorem stating that any universal number must have at least 55 digits. -/
theorem universal_number_min_digits (n : ℕ) (h : IsUniversal n) : 
  (Nat.digits 10 n).length ≥ MinUniversalDigits := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_universal_number_min_digits_l904_90496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_maximizes_profit_l904_90403

/-- Represents the daily profit function for a product -/
noncomputable def daily_profit (cost base_price base_sales price_increase_effect price_decrease_effect : ℝ) 
  (selling_price : ℝ) : ℝ :=
  if selling_price ≥ base_price then
    (base_sales - price_increase_effect * (selling_price - base_price)) * (selling_price - cost)
  else
    (base_sales + price_decrease_effect * (base_price - selling_price)) * (selling_price - cost)

/-- Theorem stating that 20 yuan maximizes daily profit under given conditions -/
theorem optimal_price_maximizes_profit :
  let cost : ℝ := 10
  let base_price : ℝ := 18
  let base_sales : ℝ := 60
  let price_increase_effect : ℝ := 5
  let price_decrease_effect : ℝ := 10
  ∀ p : ℝ, daily_profit cost base_price base_sales price_increase_effect price_decrease_effect 20 
         ≥ daily_profit cost base_price base_sales price_increase_effect price_decrease_effect p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_maximizes_profit_l904_90403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_curve_properties_l904_90417

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-1 - 3/5 * t, 2 + 4/5 * t)

noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.cos (θ - Real.pi/4)

theorem line_and_curve_properties :
  -- General equation of line l
  ∀ x y : ℝ, (∃ t : ℝ, line_l t = (x, y)) ↔ 4*x + 3*y - 2 = 0 ∧
  -- Cartesian equation of curve C
  ∀ x y : ℝ, (∃ θ : ℝ, x = curve_C θ * Real.cos θ ∧ y = curve_C θ * Real.sin θ) ↔ x^2 + y^2 - 2*x - 2*y = 0 ∧
  -- Length of AB
  ∃ A B : ℝ × ℝ, 
    (∃ t : ℝ, line_l t = A) ∧ 
    (∃ θ : ℝ, A.1 = curve_C θ * Real.cos θ ∧ A.2 = curve_C θ * Real.sin θ) ∧
    (∃ t : ℝ, line_l t = B) ∧ 
    (∃ θ : ℝ, B.1 = curve_C θ * Real.cos θ ∧ B.2 = curve_C θ * Real.sin θ) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_curve_properties_l904_90417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statementMakerIsTrulalya_l904_90491

-- Define the brothers
inductive Brother
| Tralalya
| Trulalya

-- Define the card colors
inductive CardColor
| Black
| Red

-- Define a function to represent whether a brother has a black card
def hasBlackCard (b : Brother) : Prop := ∃ (c : CardColor), c = CardColor.Black

-- Define the statement made
def statement : Prop := hasBlackCard Brother.Tralalya

-- Define the truth-telling behavior of each brother
def tellsTruth (b : Brother) (c : CardColor) : Prop :=
  match b with
  | Brother.Tralalya => c = CardColor.Red
  | Brother.Trulalya => True

-- Define that each brother has a card
axiom hasCard : Brother → CardColor

-- Define that one brother made the statement
axiom statementMaker : Brother

-- Define that the statement maker tells the truth based on their card
axiom statementMakerTellsTruth : tellsTruth statementMaker (hasCard statementMaker)

-- Theorem to prove
theorem statementMakerIsTrulalya : statementMaker = Brother.Trulalya := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statementMakerIsTrulalya_l904_90491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_problem_l904_90490

theorem car_speed_problem (s1 s2 s : ℝ) (v1 : ℝ) :
  s1 = 1600 →
  s2 = 800 →
  s = 200 →
  v1 = 72 * 1000 / 3600 →
  let t1 := s1 / v1
  let l1 := s2 - s
  let l2 := s2 + s
  let v2_1 := l1 / t1
  let v2_2 := l2 / t1
  v2_1 = 7.5 ∧ v2_2 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_problem_l904_90490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_increasing_l904_90424

-- Define the property of being an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define the property of being increasing on (0, +∞)
def is_increasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y

-- Define the function we're proving about
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

-- State the theorem
theorem f_is_even_and_increasing : is_even f ∧ is_increasing_on_positive f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_increasing_l904_90424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mame_on_top_probability_l904_90476

/-- Represents a piece of paper that can be folded -/
structure Paper where
  segments : Nat

/-- Represents the state of the folded paper -/
structure FoldedPaper where
  paper : Paper
  foldedSegments : Nat

/-- Folds a paper into a given number of segments -/
def foldPaper (p : Paper) (folds : Nat) : FoldedPaper :=
  { paper := p, foldedSegments := p.segments * folds }

/-- Calculates the probability of a specific segment being on top -/
noncomputable def probabilityOfSegmentOnTop (fp : FoldedPaper) : ℚ :=
  1 / fp.foldedSegments

/-- The main theorem stating the probability of "MAME" being on top -/
theorem mame_on_top_probability (p : Paper) :
  p.segments = 4 →
  let fp := foldPaper p 2
  probabilityOfSegmentOnTop fp = 1 / 8 := by
  sorry

#check mame_on_top_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mame_on_top_probability_l904_90476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_equals_one_l904_90472

theorem sum_of_reciprocals_equals_one (a b : ℝ) (h1 : (2 : ℝ)^a = 6) (h2 : (3 : ℝ)^b = 6) : 
  1/a + 1/b = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_equals_one_l904_90472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l904_90423

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  area : Real

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  Real.cos t.A + Real.cos (t.A / 2) = 0 ∧
  t.b + t.c = 10 ∧
  t.area = 2 * Real.sqrt 3

-- Define the angle bisector
noncomputable def angle_bisector (t : Triangle) : Real := 
  sorry

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : satisfies_conditions t) : 
  t.A = 2 * Real.pi / 3 ∧ angle_bisector t = 4 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l904_90423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_g_condition_l904_90446

-- Define the functions f and g
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log (x - 1) - k * (x - 1) + 1

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - Real.log (x + 1) + f k (x + 2)

-- State the theorem
theorem k_range_for_g_condition (k : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → |g k x₁ - g k x₂| ≤ 1) ↔
  k ∈ Set.Icc (-2/3) (4/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_g_condition_l904_90446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_8_c_terms_l904_90415

-- Define the sequences and their properties
def a : ℕ → ℝ := sorry
def S : ℕ → ℝ := sorry
def b : ℕ → ℝ := sorry
def T : ℕ → ℝ := sorry
def c : ℕ → ℝ := sorry

-- Define the properties of the sequences
axiom S_def : ∀ n, S n = 2 * (a n) - 2
axiom b_7 : b 7 = 3
axiom T_5 : T 5 = -25
axiom c_def : ∀ n, c n = if n % 2 = 1 then a n else b n

-- State the theorem
theorem sum_of_first_8_c_terms :
  (c 1) + (c 2) + (c 3) + (c 4) + (c 5) + (c 6) + (c 7) + (c 8) = 166 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_8_c_terms_l904_90415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_exists_l904_90402

noncomputable def f (x : ℝ) : ℝ := Real.pi * x + Real.log x / Real.log 2

theorem zero_point_exists : ∃ x : ℝ, x ∈ Set.Icc (1/4 : ℝ) (1/2 : ℝ) ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_exists_l904_90402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_two_diamonds_or_aces_l904_90412

/-- Probability of drawing at least two cards that are either diamonds or aces in three draws with replacement from a standard 52-card deck -/
theorem probability_at_least_two_diamonds_or_aces :
  let total_cards : ℕ := 52
  let target_cards : ℕ := 16
  let num_draws : ℕ := 3
  let prob_target : ℚ := target_cards / total_cards
  let prob_non_target : ℚ := 1 - prob_target
  let prob_at_least_two : ℚ := 1 - (prob_non_target ^ num_draws + num_draws * prob_target * prob_non_target ^ (num_draws - 1))
  prob_at_least_two = 580 / 2197 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_two_diamonds_or_aces_l904_90412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l904_90473

open Real

-- Define the function f(x) = x cos x - sin x
noncomputable def f (x : ℝ) : ℝ := x * cos x - sin x

-- State the theorem about the derivative of f
theorem derivative_of_f :
  deriv f = fun x => -x * sin x := by
  -- The proof is omitted for now
  sorry

#check derivative_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l904_90473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_average_score_l904_90462

/-- Represents a high school with its student population and test scores. -/
structure School where
  boys : ℕ
  girls : ℕ
  boys_score : ℚ
  girls_score : ℚ

/-- Calculates the weighted average score for a given category across two schools. -/
def weighted_average (s1 s2 : School) (score1 score2 : ℚ) : ℚ :=
  (s1.boys * score1 + s2.boys * score2) / (s1.boys + s2.boys : ℚ)

/-- The main theorem stating the average score for girls across both schools. -/
theorem girls_average_score (lincoln monroe : School) : 
  lincoln.boys * 4 = lincoln.girls * 3 →
  monroe.boys * 2 = monroe.girls * 5 →
  weighted_average lincoln monroe lincoln.boys_score monroe.boys_score = 78 →
  lincoln.boys_score = 68 →
  monroe.boys_score = 85 →
  lincoln.girls_score = 80 →
  monroe.girls_score = 95 →
  weighted_average lincoln monroe lincoln.girls_score monroe.girls_score = 83 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_average_score_l904_90462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_plus_two_l904_90422

noncomputable def g (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x ≤ 1 then -x
  else if 1 < x ∧ x ≤ 4 then (x - 3)^2 + 1
  else if 4 < x ∧ x ≤ 5 then x - 2
  else 0  -- Default value for x outside the specified ranges

theorem g_plus_two (x : ℝ) :
  (-2 ≤ x ∧ x ≤ 5) →
  g x + 2 = if -2 ≤ x ∧ x ≤ 1 then -x + 2
            else if 1 < x ∧ x ≤ 4 then (x - 3)^2 + 3
            else if 4 < x ∧ x ≤ 5 then x
            else 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_plus_two_l904_90422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_is_half_l904_90413

/-- Proves that tan θ = 1/2 given specific matrix conditions -/
theorem tan_theta_is_half (k t : ℝ) (θ : ℝ) (hk : k > 0) (ht : t > 0) : 
  let D : Matrix (Fin 2) (Fin 2) ℝ := !![k, 0; 0, k]
  let R : Matrix (Fin 2) (Fin 2) ℝ := !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ]
  let T : Matrix (Fin 2) (Fin 2) ℝ := !![t, 0; 0, t]
  R * (D + T) = !![10, -5; 5, 10] → Real.tan θ = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_is_half_l904_90413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_max_major_axis_theorem_l904_90405

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def line (x y : ℝ) : Prop := y = -x + 1

noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

def focalLength (c : ℝ) : ℝ := 2 * c

def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem ellipse_intersection_theorem
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (e : ℝ) (he : e = Real.sqrt 2 / 2)
  (c : ℝ) (hc : focalLength c = 2)
  (A B : ℝ × ℝ) (hA : ellipse a b A.1 A.2 ∧ line A.1 A.2)
  (hB : ellipse a b B.1 B.2 ∧ line B.1 B.2) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 2 / 3 :=
by sorry

theorem max_major_axis_theorem
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (e : ℝ) (he : 1/2 ≤ e ∧ e ≤ Real.sqrt 2 / 2)
  (A B : ℝ × ℝ) (hA : ellipse a b A.1 A.2 ∧ line A.1 A.2)
  (hB : ellipse a b B.1 B.2 ∧ line B.1 B.2)
  (hAB : perpendicular A.1 A.2 B.1 B.2) :
  2 * a ≤ Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_max_major_axis_theorem_l904_90405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagonal_pyramid_angles_l904_90408

/-- Regular hexagonal pyramid with dihedral angles and edge angle -/
structure RegularHexagonalPyramid where
  α : Real  -- dihedral angle between opposite faces
  β : Real  -- dihedral angle between adjacent faces
  γ : Real  -- angle between lateral edge and base edge

/-- Properties of a regular hexagonal pyramid -/
def is_valid_regular_hexagonal_pyramid (p : RegularHexagonalPyramid) : Prop :=
  0 < p.α ∧ p.α < Real.pi ∧
  0 < p.β ∧ p.β < Real.pi ∧
  0 < p.γ ∧ p.γ < Real.pi/2 ∧
  Real.cos (p.α/2) = 2 * Real.cos (p.β/2) ∧
  Real.sin p.γ = Real.sqrt 3 / (2 * Real.sin (p.β/2)) ∧
  Real.tan p.γ = Real.sqrt 3 / Real.sin (p.α/2)

theorem regular_hexagonal_pyramid_angles 
  (p : RegularHexagonalPyramid) 
  (h : is_valid_regular_hexagonal_pyramid p) : 
  Real.cos (p.α/2) = 2 * Real.cos (p.β/2) ∧
  Real.sin p.γ = Real.sqrt 3 / (2 * Real.sin (p.β/2)) ∧
  Real.tan p.γ = Real.sqrt 3 / Real.sin (p.α/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagonal_pyramid_angles_l904_90408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_fraction_l904_90430

-- Define what constitutes a fraction in this context
noncomputable def is_fraction (e : ℝ → ℝ) : Prop :=
  ∃ (n d : ℝ → ℝ), ∀ x, e x = (n x) / (d x) ∧ d x ≠ 0 ∧ (∃ y, d y ≠ d 0)

-- Define the given expressions
noncomputable def expr_A (x : ℝ) : ℝ := 3 * x + 1/2
noncomputable def expr_B (m n : ℝ) : ℝ := -(m + n) / 3
noncomputable def expr_C (x : ℝ) : ℝ := 3 / (x + 3)
noncomputable def expr_D (x : ℝ) : ℝ := x - 1

-- Theorem stating that only expr_C is a fraction
theorem only_C_is_fraction :
  is_fraction expr_C ∧
  ¬is_fraction expr_A ∧
  ¬is_fraction (λ x ↦ expr_B x x) ∧
  ¬is_fraction expr_D :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_fraction_l904_90430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_KLM_ABC_l904_90447

noncomputable section

open Real

variable (A B C X Y Z K L M : ℝ × ℝ)

def on_side (P Q R : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • Q + t • R

def is_midpoint (P Q R : ℝ × ℝ) : Prop :=
  P = (Q + R) / 2

def area (P Q R : ℝ × ℝ) : ℝ :=
  abs ((P.1 - R.1) * (Q.2 - R.2) - (Q.1 - R.1) * (P.2 - R.2)) / 2

theorem area_ratio_KLM_ABC 
  (hX : on_side X B C)
  (hY : on_side Y C A)
  (hZ : on_side Z A B)
  (hK : is_midpoint K B Y)
  (hL : is_midpoint L C Z)
  (hM : is_midpoint M A X)
  (hAXK : is_midpoint K A X)
  (hBYL : is_midpoint L B Y)
  (hCZM : is_midpoint M C Z) :
  area K L M / area A B C = (7 - 3 * sqrt 5) / 4 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_KLM_ABC_l904_90447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l904_90489

-- Define the curve C in polar coordinates
noncomputable def C (ρ θ : ℝ) : Prop := ρ * (Real.sin θ)^2 = 8 * Real.cos θ

-- Define the line l in parametric form
noncomputable def l (t : ℝ) : ℝ × ℝ := (2 + t/2, Real.sqrt 3 * t/2)

-- Define the Cartesian form of curve C
def C_cartesian (x y : ℝ) : Prop := y^2 = 8*x

-- State the theorem
theorem intersection_chord_length :
  ∃ (t₁ t₂ : ℝ),
    let (x₁, y₁) := l t₁
    let (x₂, y₂) := l t₂
    C_cartesian x₁ y₁ ∧ C_cartesian x₂ y₂ ∧
    (t₁ - t₂)^2 = (32/3)^2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l904_90489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l904_90468

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and eccentricity √3,
    prove that its asymptotes have the equation y = ±√2x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.sqrt (a^2 + b^2) / a = Real.sqrt 3) →
  ∃ k : ℝ, k = Real.sqrt 2 ∧ (∀ x y : ℝ, y = k * x ∨ y = -k * x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l904_90468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l904_90418

theorem triangle_angle_theorem (A B C : ℝ) (h1 : A + B + C = π) 
  (h2 : Real.sin C = 2 * Real.sin (A - B)) (h3 : B = π / 6) :
  A = π / 3 ∧ 
  ∀ x, Real.cos (A + x) = -1/3 → Real.cos (A - 2*x) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l904_90418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_distance_l904_90461

/-- A hyperbola with equation x^2/2 - y^2/4 = 1 -/
structure Hyperbola where
  C : (ℝ × ℝ) → Prop
  h : C (x, y) ↔ x^2/2 - y^2/4 = 1

/-- The foci of the hyperbola -/
noncomputable def foci (h : Hyperbola) : (ℝ × ℝ) × (ℝ × ℝ) := ((-Real.sqrt 6, 0), (Real.sqrt 6, 0))

/-- An asymptote of the hyperbola -/
noncomputable def asymptote (h : Hyperbola) : (ℝ × ℝ) → Prop :=
  fun (x, y) ↦ y = Real.sqrt 2 * x

/-- The dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

theorem hyperbola_point_distance (h : Hyperbola) (P : ℝ × ℝ) :
  asymptote h P →
  let (F1, F2) := foci h
  dot_product (P.1 - F1.1, P.2 - F1.2) (P.1 - F2.1, P.2 - F2.2) = 0 →
  distance_to_x_axis P = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_distance_l904_90461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_B_card_is_three_l904_90483

-- Define the set of possible card numbers
def CardNumbers : Set Nat := {1, 2, 3, 4, 5}

-- Define the players
inductive Player
  | A
  | B

-- Define a function to represent a player's card
variable (card : Player → Nat)

-- Define predicates for the players' statements
def A_statement (a : Nat) : Prop :=
  ∀ b ∈ CardNumbers, a ≠ 1 ∧ a ≠ 5

def B_statement (b : Nat) : Prop :=
  ∀ a ∈ CardNumbers, A_statement a → b ≠ 1 ∧ b ≠ 2 ∧ b ≠ 4 ∧ b ≠ 5

-- Theorem statement
theorem player_B_card_is_three :
  card Player.A ∈ CardNumbers →
  card Player.B ∈ CardNumbers →
  A_statement (card Player.A) →
  B_statement (card Player.B) →
  card Player.B = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_B_card_is_three_l904_90483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_of_33kg_apples_l904_90484

/-- The price of apples per kilogram for the first 30 kgs -/
noncomputable def l : ℝ := 200 / 10

/-- The price of apples per kilogram for each additional kilogram beyond 30 kgs -/
noncomputable def q : ℝ := (726 - 30 * l) / 6

/-- The price of n kilograms of apples -/
noncomputable def price (n : ℝ) : ℝ :=
  if n ≤ 30 then n * l
  else 30 * l + (n - 30) * q

theorem price_of_33kg_apples : price 33 = 663 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_of_33kg_apples_l904_90484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_is_10_5_l904_90433

-- Define the vertices of the polygon
noncomputable def vertices : List (ℝ × ℝ) := [(0, 1), (3, 4), (7, 1), (3, 7)]

-- Define the shoelace formula for calculating the area of a polygon
noncomputable def shoelaceArea (vs : List (ℝ × ℝ)) : ℝ :=
  (1/2) * (vs.zip (vs.rotateRight 1)).foldl
    (fun acc (p1, p2) => acc + p1.1 * p2.2 - p1.2 * p2.1)
    0

-- Theorem statement
theorem polygon_area_is_10_5 :
  shoelaceArea vertices = 10.5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_is_10_5_l904_90433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_qr_equals_three_l904_90465

/-- Triangle PQR with specific angle and side length properties -/
structure SpecialTriangle where
  P : Real
  Q : Real
  R : Real
  PQ : Real
  angle_sum : P + Q + R = Real.pi
  cosine_sine_sum : Real.cos (Q - 2*P) + Real.sin (P + Q) = 2
  side_length : PQ = 6

/-- The length of side QR in the special triangle -/
noncomputable def qr_length (t : SpecialTriangle) : Real :=
  3

/-- Theorem: In the special triangle, QR = 3 -/
theorem qr_equals_three (t : SpecialTriangle) : qr_length t = 3 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_qr_equals_three_l904_90465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l904_90436

theorem sin_minus_cos_value (α : Real) 
  (h1 : Real.sin α * Real.cos α = 1/4)
  (h2 : 0 < α ∧ α < Real.pi/4) :
  Real.sin α - Real.cos α = -Real.sqrt 2/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l904_90436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_competition_probabilities_l904_90448

/-- The probability of student A making a shot -/
noncomputable def prob_A : ℝ := 1/3

/-- The probability of student B making a shot -/
noncomputable def prob_B : ℝ := 1/2

/-- The number of shots each student attempts -/
def num_shots : ℕ := 3

/-- The probability that B makes at most 2 shots -/
noncomputable def prob_B_at_most_2 : ℝ := 7/8

/-- The probability that B makes exactly 2 more shots than A -/
noncomputable def prob_B_2_more_than_A : ℝ := 1/6

theorem basketball_competition_probabilities :
  (1 - (prob_B ^ num_shots) = prob_B_at_most_2) ∧
  (Finset.sum (Finset.range 2) (λ k ↦ (Nat.choose num_shots k * prob_A^k * (1-prob_A)^(num_shots-k)) *
    (Nat.choose num_shots (k+2) * prob_B^(k+2) * (1-prob_B)^(num_shots-(k+2)))) = prob_B_2_more_than_A) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_competition_probabilities_l904_90448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_size_lower_bound_l904_90439

theorem set_size_lower_bound (F : Finset ℤ) (n : ℕ) 
  (h_nonempty : F.Nonempty)
  (h_sum : ∀ x ∈ F, ∃ y z, y ∈ F ∧ z ∈ F ∧ x = y + z)
  (h_nonzero : ∀ (k : ℕ) (hk1 : 1 ≤ k) (hk2 : k ≤ n) (xs : Finset ℤ), 
    xs ⊆ F → xs.card = k → (xs.sum id) ≠ 0) :
  F.card ≥ 2 * n + 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_size_lower_bound_l904_90439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_parallel_rotating_lines_l904_90486

/-- Two points in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane -/
structure Line where
  point : Point
  direction : ℝ × ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Two parallel lines that can rotate around fixed points -/
structure ParallelRotatingLines where
  l₁ : Line
  l₂ : Line
  isParallel : l₁.direction = l₂.direction

theorem max_distance_parallel_rotating_lines :
  let P : Point := ⟨-1, 2⟩
  let Q : Point := ⟨2, -3⟩
  ∀ l : ParallelRotatingLines,
    l.l₁.point = P →
    l.l₂.point = Q →
    (∀ d, d = distance l.l₁.point l.l₂.point → d ≤ Real.sqrt 34) ∧
    (∃ d, d = distance l.l₁.point l.l₂.point ∧ d = Real.sqrt 34) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_parallel_rotating_lines_l904_90486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_savings_percentage_l904_90493

noncomputable def monthly_salary : ℚ := 3300
noncomputable def discretionary_income : ℚ := monthly_salary / 5
noncomputable def vacation_fund_percentage : ℚ := 30
noncomputable def eating_out_percentage : ℚ := 35
noncomputable def gifts_charity_amount : ℚ := 99

theorem jill_savings_percentage :
  let vacation_fund := (vacation_fund_percentage / 100) * discretionary_income
  let eating_out := (eating_out_percentage / 100) * discretionary_income
  let total_spent := vacation_fund + eating_out + gifts_charity_amount
  let savings := discretionary_income - total_spent
  (savings / discretionary_income) * 100 = 20 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_savings_percentage_l904_90493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_false_proposition_l904_90470

theorem range_of_a_for_false_proposition :
  {a : ℝ | ∃ x : ℝ, x^2 - 2*a*x + 3 < 0} = {a : ℝ | a > Real.sqrt 3 ∨ a < -Real.sqrt 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_false_proposition_l904_90470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mass_range_l904_90467

/-- The maximum mass calculation function -/
noncomputable def max_mass (radius : ℝ) : ℝ := 2000 / (10 * radius)

/-- The nominal radius of the winch drum -/
def nominal_radius : ℝ := 0.25

/-- The tolerance of the radius -/
def radius_tolerance : ℝ := 0.05

/-- The nominal maximum mass -/
def nominal_max_mass : ℝ := 800

/-- The maximum rounding error -/
def max_rounding_error : ℝ := 160

/-- Theorem stating the range of the maximum mass -/
theorem max_mass_range :
  ∃ (lower upper : ℝ),
    lower = nominal_max_mass - max_rounding_error ∧
    upper = nominal_max_mass + max_rounding_error ∧
    (∀ r : ℝ, 
      nominal_radius - radius_tolerance ≤ r ∧ 
      r ≤ nominal_radius + radius_tolerance →
      lower ≤ max_mass r ∧ max_mass r ≤ upper) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mass_range_l904_90467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_is_70_l904_90498

/-- Represents the scores of students in a physics class -/
def scores : List Nat := [62, 65, 70, 73, 85, 90]

/-- The number of students in the class -/
def num_students : Nat := 6

/-- Function to check if a number is divisible by another number -/
def is_divisible (a b : Nat) : Prop := a % b = 0

/-- Function to calculate the sum of a list of numbers -/
def list_sum (l : List Nat) : Nat := l.foldl (·+·) 0

/-- Proposition: The last score entered was 70 -/
theorem last_score_is_70 : 
  ∃ (permutation : List Nat), 
    permutation.length = num_students ∧ 
    (∀ x, x ∈ permutation ↔ x ∈ scores) ∧
    permutation.getLast? = some 70 ∧
    (∀ k, k ∈ Finset.range num_students → 
      is_divisible (list_sum (permutation.take (k+1))) (k+1)) := by
  sorry

#check last_score_is_70

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_is_70_l904_90498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l904_90407

theorem range_of_m (x m : ℕ) (h1 : x ≤ m) 
  (h2 : x = 1 ∨ x = 2) : 2 ≤ m ∧ m < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l904_90407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_payment_difference_l904_90434

theorem pizza_payment_difference :
  let total_slices : ℕ := 12
  let plain_cost : ℚ := 12
  let cheese_cost : ℚ := 3
  let mushroom_cost : ℚ := 4
  let dan_cheese_slices : ℕ := 6
  let dan_plain_slices : ℕ := 2
  let carl_mushroom_slices : ℕ := 6
  let carl_plain_slices : ℕ := 1

  let total_cost : ℚ := plain_cost + cheese_cost + mushroom_cost
  let cost_per_slice : ℚ := total_cost / total_slices
  let cheese_per_slice : ℚ := cheese_cost / dan_cheese_slices
  let mushroom_per_slice : ℚ := mushroom_cost / carl_mushroom_slices

  let dan_cost : ℚ := dan_cheese_slices * (cost_per_slice + cheese_per_slice) + 
                      dan_plain_slices * cost_per_slice
  let carl_cost : ℚ := carl_mushroom_slices * (cost_per_slice + mushroom_per_slice) + 
                       carl_plain_slices * cost_per_slice

  ∃ (n : ℤ), (dan_cost - carl_cost) * 100 = n ∧ n = 58 :=
by
  sorry

#eval (12 + 3 + 4 : ℚ) / 12 -- cost per slice
#eval ((6 * ((19 : ℚ) / 12 + 3 / 6)) + (2 * (19 : ℚ) / 12)) - ((6 * ((19 : ℚ) / 12 + 4 / 6)) + ((19 : ℚ) / 12))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_payment_difference_l904_90434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maintaining_value_interval_2x_squared_maintaining_value_interval_mx_squared_minus_3x_plus_4_l904_90400

-- Define the concept of a "maintaining value" interval
def is_maintaining_value_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ 
  (∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x ≤ y → f x ≤ f y ∨ f x ≥ f y) ∧ 
  (∀ y, y ∈ Set.Icc a b → ∃ x, x ∈ Set.Icc a b ∧ f x = y)

-- Theorem 1: [0, 1/2] is a "maintaining value" interval for f(x) = 2x^2
theorem maintaining_value_interval_2x_squared :
  is_maintaining_value_interval (fun x => 2 * x^2) 0 (1/2) := by sorry

-- Theorem 2: Characterization of m for which g(x) = mx^2 - 3x + 4 has a "maintaining value" interval
theorem maintaining_value_interval_mx_squared_minus_3x_plus_4 (m : ℝ) :
  (m > 0) →
  (∃ a b, is_maintaining_value_interval (fun x => m * x^2 - 3 * x + 4) a b) ↔ 
  (m ∈ Set.Icc (11/16) (3/4) ∪ Set.Ico (15/16) 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maintaining_value_interval_2x_squared_maintaining_value_interval_mx_squared_minus_3x_plus_4_l904_90400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l904_90485

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1/2)^x

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 8 ∧ f x = y) ↔ (y > 0 ∧ y ≤ 1/256) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l904_90485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_cutting_schemes_l904_90451

theorem wire_cutting_schemes :
  let total_length : ℕ := 150
  let short_length : ℕ := 10
  let long_length : ℕ := 20
  let valid_scheme (x y : ℕ) : Prop :=
    x * short_length + y * long_length = total_length ∧ x > 0 ∧ y > 0
  (∃! n : ℕ, n = (Finset.filter (λ p : ℕ × ℕ => valid_scheme p.1 p.2)
    (Finset.product (Finset.range (total_length / short_length + 1))
                    (Finset.range (total_length / long_length + 1)))).card ∧ n = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_cutting_schemes_l904_90451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_author_productivity_l904_90477

/-- Calculates the average words per hour for an author writing a book -/
theorem author_productivity (total_words : ℕ) (total_hours : ℕ) 
  (increased_productivity_ratio : ℚ) (increased_time_ratio : ℚ) : 
  total_words = 60000 → 
  total_hours = 100 → 
  increased_productivity_ratio = 3/2 → 
  increased_time_ratio = 1/5 → 
  (total_words : ℚ) / total_hours = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_author_productivity_l904_90477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l904_90460

-- Define the function f(x) = 3 - 2sin(x)
noncomputable def f (x : ℝ) : ℝ := 3 - 2 * Real.sin x

-- Define the interval [π/2 + 2kπ, 3π/2 + 2kπ] for any integer k
def monotonic_interval (k : ℤ) : Set ℝ := Set.Icc (Real.pi / 2 + 2 * ↑k * Real.pi) (3 * Real.pi / 2 + 2 * ↑k * Real.pi)

-- Theorem stating that f is monotonically increasing on the given intervals
theorem f_monotone_increasing :
  ∀ k : ℤ, MonotoneOn f (monotonic_interval k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l904_90460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_minus_2alpha_l904_90416

theorem sin_pi_minus_2alpha (α : ℝ) 
  (h1 : Real.sin (π / 2 + α) = -3 / 5) 
  (h2 : π / 2 < α) 
  (h3 : α < π) : 
  Real.sin (π - 2 * α) = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_minus_2alpha_l904_90416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_matrices_sum_l904_90410

-- Define the matrices A and B
def A (a b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![
  1, a, 2;
  b, 3, c;
  4, d, 5
]

def B (e f g h : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![
  -7, e,  f;
  -6, -15, g;
   3,  h,  6
]

-- Theorem statement
theorem inverse_matrices_sum (a b c d e f g h : ℝ) :
  (A a b c d) * (B e f g h) = 1 →
  a + b + c + d + e + f + g + h = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_matrices_sum_l904_90410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l904_90469

open Real

/-- The function f(x) defined in the problem -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := exp x * (log x + 1/2 * x^2 - m * x)

/-- The derivative of f(x) -/
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := 
  exp x * (log x + 1/2 * x^2 - m * x) + exp x * (1/x + x - m)

theorem problem_statement (m : ℝ) : 
  (∀ x > 0, f_deriv m x - f m x > 0) → m < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l904_90469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_distance_l904_90414

/-- Proves that a car with given fuel efficiency and gas cost can travel a certain distance with a fixed amount of money --/
theorem car_travel_distance (fuel_efficiency gas_cost money : ℚ) : 
  fuel_efficiency = 40 → gas_cost = 5 → money = 25 → 
  (money / gas_cost) * fuel_efficiency = 200 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  
#check car_travel_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_distance_l904_90414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l904_90426

/-- The speed of a train in km/h, given its length in meters and time to cross an electric pole in seconds. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem stating that a train 1000 m long crossing an electric pole in 200 sec has a speed of 18 km/h. -/
theorem train_speed_calculation :
  train_speed 1000 200 = 18 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  simp [div_mul_eq_mul_div]
  -- The rest of the proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l904_90426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_extrema_l904_90458

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := (1/3) * x^3 - (a/2) * x^2 + b * x + c

-- Theorem statement
theorem tangent_and_extrema :
  ∀ (a b c : ℝ),
  (∀ x, (deriv (f a b c)) x = x^2 - a*x + b) →
  (f a b c 0 = 2 ∧ deriv (f a b c) 0 = 0) →
  (b = 0 ∧ c = 2) ∧
  (a = 2 → 
    (∀ x, f 2 0 2 x ≤ 2) ∧
    (∀ x, f 2 0 2 x ≥ 2/3) ∧
    (∃ x, f 2 0 2 x = 2) ∧
    (∃ x, f 2 0 2 x = 2/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_extrema_l904_90458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_perimeter_value_l904_90478

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sine law relation for the triangle -/
def sineLaw (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a * Real.sin t.C = t.c + t.c * Real.cos t.A

/-- The area of the triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  (1 / 2) * t.b * t.c * Real.sin t.A

theorem angle_A_value (t : Triangle) (h : sineLaw t) : t.A = π / 3 := by
  sorry

theorem perimeter_value (t : Triangle) (h1 : t.a = 2 * Real.sqrt 3) 
  (h2 : area t = Real.sqrt 3) (h3 : t.A = π / 3) : 
  t.a + t.b + t.c = 2 * Real.sqrt 3 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_perimeter_value_l904_90478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_quadratic_and_linear_l904_90480

theorem gcd_of_quadratic_and_linear (b : ℤ) (h : ∃ k : ℤ, b = 2160 * k) :
  Int.gcd (b^2 + 9*b + 30) (b + 6) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_quadratic_and_linear_l904_90480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l904_90429

theorem rectangle_area_increase (L W : ℝ) (h : L > 0 ∧ W > 0) : 
  (((1.25 * L) * (1.25 * W) - L * W) / (L * W)) * 100 = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l904_90429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_regular_hexagon_l904_90481

/-- A regular hexagon with side length 4 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 4)

/-- The area of a triangle formed by connecting every other vertex in a regular hexagon -/
noncomputable def triangle_area (h : RegularHexagon) : ℝ := 8 * Real.sqrt 3

/-- Theorem: The area of triangle LNP in a regular hexagon LMNOPQ with side length 4 is 8√3 -/
theorem triangle_area_in_regular_hexagon (h : RegularHexagon) : 
  triangle_area h = 8 * Real.sqrt 3 := by
  -- Proof steps would go here
  sorry

#check triangle_area_in_regular_hexagon

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_regular_hexagon_l904_90481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_and_reflect_sine_l904_90495

open Real

noncomputable def f : ℝ → ℝ := fun x ↦ sin (x - π/3)
noncomputable def g : ℝ → ℝ := fun x ↦ sin (2*x)

theorem transform_and_reflect_sine (f g : ℝ → ℝ) :
  (∀ x, g (2*x) = f x) →
  (∀ x, g (x + π/6) = sin (2*x)) →
  ∀ x, f x = sin (x - π/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_and_reflect_sine_l904_90495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_six_l904_90449

def is_valid_number (n : ℕ) : Prop :=
  (100000 ≤ n ∧ n ≤ 999999) ∧
  (∃ (a b c d e f : ℕ), 
    n = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ∧
    Finset.toSet {a, b, c, d, e, f} = Finset.toSet {1, 2, 3, 4, 5, 6})

theorem smallest_divisible_by_six :
  ∀ n : ℕ, is_valid_number n → n % 6 = 0 → 123456 ≤ n :=
by
  sorry

#check smallest_divisible_by_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_six_l904_90449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_number_theorem_l904_90459

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def sum_of_proper_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun d ↦ d ∣ n ∧ d ≠ n) (Finset.range n)).sum id

theorem perfect_number_theorem (p : ℕ) (n : ℕ) (h1 : n = 2^(p-1) * (2^p - 1)) 
  (h2 : is_prime (2^p - 1)) : sum_of_proper_divisors n = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_number_theorem_l904_90459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_and_parallel_line_l904_90411

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ a * x + 2 * y + 6 = 0
def l₂ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ x + (a - 1) * y + a^2 - 1 = 0

-- Define perpendicularity of two lines
def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ a b c d : ℝ, f a b → f c d → g a b → g c d →
    (a - c) * (a - c) + (b - d) * (b - d) ≠ 0 →
    (a - c) * (a - c) = (b - d) * (b - d)

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, f x y ↔ g (x + k) y

theorem perpendicular_lines_and_parallel_line :
  ∃ a : ℝ,
    (perpendicular (l₁ a) (l₂ a)) ∧
    (a = 2/3) ∧
    (parallel (l₂ a) (λ x y ↦ x - (1/3) * y - 2 = 0)) ∧
    (1 - (1/3) * (-3) - 2 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_and_parallel_line_l904_90411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_and_point_l904_90442

-- Define the curves C₁ and C₂
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (2 * (Real.cos α) ^ 2, Real.sin (2 * α))

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 1 / (Real.sin θ - Real.cos θ)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the distance function between a point and a curve
noncomputable def distance_to_curve (p : ℝ × ℝ) (C : ℝ → ℝ × ℝ) (t : ℝ) : ℝ :=
  Real.sqrt ((p.1 - (C t).1)^2 + (p.2 - (C t).2)^2)

-- State the theorem
theorem min_distance_and_point :
  ∃ (p : ℝ × ℝ), p ∈ Set.range C₁ ∧
    (∀ q ∈ Set.range C₁, ∀ θ : ℝ,
      distance_to_curve p C₂ θ ≤ distance_to_curve q C₂ θ) ∧
    p = ((2 - Real.sqrt 2) / 2, Real.sqrt 2 / 2) ∧
    (∃ θ : ℝ, distance_to_curve p C₂ θ = Real.sqrt 2 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_and_point_l904_90442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_count_theorem_l904_90437

/-- The number of movies Timothy and Theresa went to in 2009 and 2010 -/
def total_movies (timothy_2009 timothy_2010 theresa_2010 : ℕ) : ℕ :=
  timothy_2009 + timothy_2010 + (timothy_2009 / 2) + theresa_2010

theorem movie_count_theorem (timothy_2009 timothy_2010 theresa_2010 : ℕ)
  (h1 : timothy_2010 = timothy_2009 + 7)
  (h2 : timothy_2009 = 24)
  (h3 : theresa_2010 = 2 * timothy_2010) :
  total_movies timothy_2009 timothy_2010 theresa_2010 = 129 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_count_theorem_l904_90437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_one_implies_b_zero_l904_90450

/-- The function f(x) with parameter b -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * (b-1) * x^2 + b^2 * x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (b : ℝ) (x : ℝ) : ℝ := x^2 + (b-1) * x + b^2

/-- Theorem stating that if f(x) has an extremum at x = 1, then b = 0 -/
theorem extremum_at_one_implies_b_zero :
  ∀ b : ℝ, (∃ (ε : ℝ), ε > 0 ∧ 
    (∀ x : ℝ, x ∈ Set.Ioo (1 - ε) (1 + ε) → f b x ≤ f b 1 ∨ f b x ≥ f b 1)) →
  f_deriv b 1 = 0 →
  b = 0 := by
  sorry

#check extremum_at_one_implies_b_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_one_implies_b_zero_l904_90450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_example_l904_90428

/-- Calculates the compound interest given principal, rate, time, and compounding frequency -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (frequency : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time) - principal

/-- The compound interest on $500 invested for 5 years at 5% per annum, compounded yearly, is approximately $138.14 -/
theorem compound_interest_example : 
  ∃ ε > 0, |compound_interest 500 0.05 5 1 - 138.14| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_example_l904_90428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_equation_l904_90455

theorem power_of_three_equation (x : ℝ) : (3 : ℝ)^4 * (3 : ℝ)^x = 81 ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_equation_l904_90455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fertilization_changes_rates_l904_90440

/-- Represents a cell in an organism -/
structure Cell where
  chromosomes : ℕ
  respirationRate : ℝ
  synthesisRate : ℝ

/-- Represents the process of fertilization -/
def fertilization (sperm : Cell) (egg : Cell) : Cell :=
  sorry

/-- Represents the process of meiosis -/
def meiosis (parent : Cell) : Cell :=
  sorry

/-- Represents the state of a cell before and after fertilization -/
structure FertilizationState where
  before : Cell
  after : Cell

/-- The main theorem stating that cellular respiration and substance synthesis rates change after fertilization -/
theorem fertilization_changes_rates (egg : Cell) (sperm : Cell) : 
  let state := FertilizationState.mk egg (fertilization sperm egg)
  state.before.respirationRate ≠ state.after.respirationRate ∧
  state.before.synthesisRate ≠ state.after.synthesisRate := by
  sorry

/-- The number of chromosomes in gametes is half that of somatic cells -/
axiom gamete_chromosome_count (parent : Cell) : 
  (meiosis parent).chromosomes = parent.chromosomes / 2

/-- Fertilization restores the full chromosome count -/
axiom fertilization_restores_chromosomes (sperm : Cell) (egg : Cell) :
  (fertilization sperm egg).chromosomes = sperm.chromosomes + egg.chromosomes

/-- Free combination occurs during meiosis -/
axiom meiosis_free_combination (parent1 : Cell) (parent2 : Cell) :
  meiosis parent1 ≠ meiosis parent2

/-- The cytoplasm of the fertilized egg comes mostly from the egg cell -/
axiom egg_cytoplasm_dominance (sperm : Cell) (egg : Cell) :
  ∃ (k : ℝ), k > 0.9 ∧ k < 1 ∧ 
  (fertilization sperm egg).respirationRate = k * egg.respirationRate +
  (1 - k) * sperm.respirationRate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fertilization_changes_rates_l904_90440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cut_cube_surface_area_l904_90479

/-- Represents the geometry of a cut cube --/
structure CutCube where
  side_length : ℝ
  cut_distance_from_edge : ℝ
  cut_distance_from_opposite_edge : ℝ

/-- Calculates the total surface area of the two parts resulting from cutting a cube --/
noncomputable def total_surface_area (cube : CutCube) : ℝ :=
  let original_surface_area := 6 * cube.side_length ^ 2
  let cut_diagonal := (cube.side_length ^ 2 + (cube.side_length - cube.cut_distance_from_edge - cube.cut_distance_from_opposite_edge) ^ 2).sqrt
  let new_surface_area := 2 * cube.side_length * cut_diagonal
  original_surface_area + new_surface_area

/-- Theorem stating that the total surface area of the cut cube is 1176 square centimeters --/
theorem cut_cube_surface_area :
  let cube : CutCube := ⟨12, 4, 3⟩
  total_surface_area cube = 1176 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cut_cube_surface_area_l904_90479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l904_90499

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a/x

-- State the theorem
theorem a_range (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x < f a y) → 
  -3 ≤ a ∧ a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l904_90499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_2_min_value_a_less_than_1_min_value_is_a_l904_90404

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / (x + 1)

-- Theorem for the case a = 2
theorem min_value_a_2 :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 2 - 1 ∧
  ∀ x : ℝ, x ≥ 0 → f 2 x ≥ min_val :=
sorry

-- Theorem for the case 0 < a < 1
theorem min_value_a_less_than_1 (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∀ x : ℝ, x ≥ 0 → f a x ≥ f a 0 :=
sorry

-- Corollary: The minimum value when 0 < a < 1 is a
theorem min_value_is_a (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∃ (min_val : ℝ), min_val = a ∧
  ∀ x : ℝ, x ≥ 0 → f a x ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_2_min_value_a_less_than_1_min_value_is_a_l904_90404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l904_90471

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line_l (y : ℝ) : Prop := y = 2 * Real.sqrt 3 / 3

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := ellipse_C x y

-- Define a point on the line
def point_on_line (y : ℝ) : Prop := line_l y

-- Define the right angle condition
def right_angle (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

-- Define the distance from origin to a line
noncomputable def distance_to_line (x1 y1 x2 y2 : ℝ) : ℝ :=
  abs (x1 * y2 - x2 * y1) / Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- State the theorem
theorem ellipse_and_line_properties :
  -- The ellipse passes through (1, √3/2)
  point_on_ellipse 1 (Real.sqrt 3 / 2) ∧
  -- For any points A on C and B on l forming a right angle with O
  ∀ (x1 y1 y2 : ℝ),
    point_on_ellipse x1 y1 → point_on_line y2 → right_angle x1 y1 0 y2 →
    -- The distance from O to AB is constant
    ∃ (d : ℝ), distance_to_line x1 y1 0 y2 = d ∧
    -- The minimum value of |AB| is 2
    ∃ (min_dist : ℝ), min_dist = 2 ∧
      ∀ (x y : ℝ), point_on_ellipse x y →
        Real.sqrt (x^2 + (y - y2)^2) ≥ min_dist :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l904_90471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_normal_line_equation_l904_90492

-- Define the curve
noncomputable def curve (a : ℝ) (t : ℝ) : ℝ × ℝ :=
  (a * Real.sin t ^ 3, a * Real.cos t ^ 3)

-- Define the parameter value
noncomputable def t₀ : ℝ := Real.pi / 6

-- Theorem for the tangent line equation
theorem tangent_line_equation (a : ℝ) :
  let (x₀, y₀) := curve a t₀
  let m := -Real.sqrt 3
  (λ (x y : ℝ) => y - y₀ = m * (x - x₀)) =
  (λ (x y : ℝ) => y = -Real.sqrt 3 * x + (Real.sqrt 3 * a) / 2) :=
by sorry

-- Theorem for the normal line equation
theorem normal_line_equation (a : ℝ) :
  let (x₀, y₀) := curve a t₀
  let m := 1 / Real.sqrt 3
  (λ (x y : ℝ) => y - y₀ = m * (x - x₀)) =
  (λ (x y : ℝ) => y = x / Real.sqrt 3 + a / Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_normal_line_equation_l904_90492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_value_l904_90466

/-- Represents an ellipse with equation kx^2 + 5y^2 = 5 and one focus at (2, 0) -/
structure Ellipse where
  k : ℝ
  eq : ∀ (x y : ℝ), k * x^2 + 5 * y^2 = 5
  focus : ℝ × ℝ
  focus_correct : focus = (2, 0)

/-- Theorem stating that for an ellipse with the given properties, k = 1 -/
theorem ellipse_k_value (e : Ellipse) : e.k = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_value_l904_90466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_perpendicular_projection_l904_90457

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a 3D line
  mk :: -- Constructor

/-- A line in a plane -/
structure Line2D where
  -- Define properties of a 2D line
  mk :: -- Constructor

/-- Represents a plane in 3D space -/
structure Plane where
  -- Define properties of a plane
  mk :: -- Constructor

/-- Projects a 3D line onto a plane -/
def project (l : Line3D) (p : Plane) : Line2D :=
  sorry

/-- Checks if two 3D lines are perpendicular -/
def perpendicular3D (l1 l2 : Line3D) : Prop :=
  sorry

/-- Checks if two 2D lines are perpendicular -/
def perpendicular2D (l1 l2 : Line2D) : Prop :=
  sorry

/-- Converts a 2D line to a 3D line -/
def to3D (l : Line2D) : Line3D :=
  sorry

/-- The main theorem to prove -/
theorem inverse_perpendicular_projection 
  (l : Line2D) (s : Line3D) (p : Plane) : 
  ¬(perpendicular2D l (project s p)) → ¬(perpendicular3D (to3D l) s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_perpendicular_projection_l904_90457
