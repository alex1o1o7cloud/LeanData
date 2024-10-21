import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_decreases_f_passes_through_one_four_f_decreases_when_x_positive_l1131_113144

-- Define the inverse proportion function
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

-- Define the theorem
theorem inverse_proportion_decreases (k : ℝ) :
  k > 0 →
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ →
  inverse_proportion k x₂ < inverse_proportion k x₁ := by
  sorry

-- Define the specific function passing through (1, 4)
noncomputable def f (x : ℝ) : ℝ := inverse_proportion 4 x

-- Prove that f passes through (1, 4)
theorem f_passes_through_one_four :
  f 1 = 4 := by
  sorry

-- Prove that f decreases when x > 0
theorem f_decreases_when_x_positive :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ →
  f x₂ < f x₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_decreases_f_passes_through_one_four_f_decreases_when_x_positive_l1131_113144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_axes_l1131_113168

theorem sine_symmetry_axes (ω φ : ℝ) (h1 : ω > 0) (h2 : 0 < φ) (h3 : φ < π) 
  (h4 : ∀ x : ℝ, Real.sin (ω * (π/4 + x) + φ) = Real.sin (ω * (π/4 - x) + φ))
  (h5 : ∀ x : ℝ, Real.sin (ω * (5*π/4 + x) + φ) = Real.sin (ω * (5*π/4 - x) + φ)) :
  φ = π/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_axes_l1131_113168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1131_113162

open Real

theorem inequality_solution_set (x : ℝ) :
  (abs (log x) > abs (log (4 * x)) ∧ abs (log (4 * x)) > abs (log (2 * x))) ↔
  (Real.sqrt 2 / 4 < x ∧ x < 1 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1131_113162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_l1131_113149

-- Define the function f
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^α

-- Define the function g
noncomputable def g (α : ℝ) (x : ℝ) : ℝ := (2*x - 1) * f α x

-- State the theorem
theorem min_value_g (α : ℝ) :
  (f α 3 = 1/3) →
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, g α x ≥ 0) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, g α x = 0) :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_l1131_113149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_win_probability_l1131_113152

/-- Represents the number of voters supporting each candidate -/
def n : ℕ := sorry

/-- Represents the total number of voters -/
def total_voters : ℕ := 2 * n

/-- Represents the number of voters in the smaller district -/
def m : ℕ := sorry

/-- The probability of Miraflores winning in a given district configuration -/
def win_probability (m : ℕ) : ℚ :=
  if m = 1 then (n - 1) / (total_voters - 1) else 1 / 4

/-- Theorem stating that the winning probability is maximized when m = 1 -/
theorem max_win_probability :
  ∀ m : ℕ, m ≤ total_voters → win_probability 1 ≥ win_probability m := by
  sorry

#check max_win_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_win_probability_l1131_113152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_fourth_vertex_l1131_113181

/-- Predicate to check if four complex numbers form a square -/
def is_square (vertices : List ℂ) : Prop :=
  vertices.length = 4 ∧ 
  ∃ (a b c d : ℂ), 
    vertices = [a, b, c, d] ∧ 
    (b - a) = Complex.I * (c - b) ∧
    (c - b) = Complex.I * (d - c) ∧
    (d - c) = Complex.I * (a - d)

/-- Given three vertices of a square in the complex plane, find the fourth vertex -/
theorem square_fourth_vertex 
  (z₁ z₂ z₃ z₄ : ℂ) 
  (h₁ : z₁ = (3 + Complex.I) / (1 - Complex.I))
  (h₂ : z₂ = -2 + Complex.I)
  (h₃ : z₃ = 0)
  (h_square : is_square [z₁, z₂, z₃, z₄]) : 
  z₄ = -1 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_fourth_vertex_l1131_113181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_superfruit_cocktail_cost_l1131_113156

/-- The cost per litre of the superfruit juice cocktail -/
noncomputable def cocktail_cost_per_litre (mixed_fruit_cost : ℝ) (acai_cost : ℝ) 
  (mixed_fruit_volume : ℝ) (acai_volume : ℝ) : ℝ :=
  ((mixed_fruit_cost * mixed_fruit_volume) + (acai_cost * acai_volume)) / 
  (mixed_fruit_volume + acai_volume)

/-- Theorem stating the cost per litre of the superfruit juice cocktail -/
theorem superfruit_cocktail_cost :
  cocktail_cost_per_litre 262.85 3104.35 33 22 = 1399.45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_superfruit_cocktail_cost_l1131_113156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_fibonacci_factorial_last_two_digits_sum_l1131_113116

/-- The last two digits of a number -/
def lastTwoDigits (n : ℕ) : ℕ := n % 100

/-- The sum of the last two digits of a list of numbers -/
def sumLastTwoDigits (list : List ℕ) : ℕ :=
  (list.map lastTwoDigits).sum % 100

/-- The modified Fibonacci Factorial series -/
def modifiedFibonacciFactorial : List ℕ := [1, 1, 2, 6, 120, 40320, 6227020800, 51090942171709440000]

theorem modified_fibonacci_factorial_last_two_digits_sum :
  sumLastTwoDigits modifiedFibonacciFactorial = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_fibonacci_factorial_last_two_digits_sum_l1131_113116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_power_expression_value_l1131_113145

-- Problem 1
theorem power_equation_solution (x : ℝ) :
  9 * (27 : ℝ)^x = 3^17 → x = 5 := by sorry

-- Problem 2
theorem power_expression_value (a x y : ℝ) :
  a^x = -2 → a^y = 3 → a^(3*x - 2*y) = -8/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_power_expression_value_l1131_113145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_second_quadrant_implies_k_range_l1131_113178

-- Define the lines
def line1 (k x y : ℝ) : Prop := k * x - y + 1 = 0
def line2 (k x y : ℝ) : Prop := x - k * y = 0

-- Define the intersection point
noncomputable def intersection (k : ℝ) : ℝ × ℝ := (1 / (k^2 - 1), k / (k^2 - 1))

-- Define the second quadrant
def second_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0

-- Theorem statement
theorem intersection_in_second_quadrant_implies_k_range :
  ∀ k : ℝ, k ≠ 1 → k ≠ -1 →
  (∃ x y : ℝ, line1 k x y ∧ line2 k x y) →
  second_quadrant (intersection k) →
  -1 < k ∧ k < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_second_quadrant_implies_k_range_l1131_113178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l1131_113143

/-- Represents a triangle with side lengths and a median --/
structure Triangle where
  xy : ℝ
  xz : ℝ
  xm : ℝ

/-- Calculates the area of a triangle given its side lengths and median --/
noncomputable def triangle_area (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating that for a triangle with given measurements, the area is approximately 185.38 --/
theorem triangle_area_approx (t : Triangle) 
  (h1 : t.xy = 9) 
  (h2 : t.xz = 17) 
  (h3 : t.xm = 13) : 
  |triangle_area t - 185.38| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l1131_113143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_license_plates_l1131_113135

/-- Represents a license plate with n digits -/
def LicensePlate (n : ℕ) := Fin n → Fin 10

/-- Two license plates differ in at least two places -/
def DifferInTwoPlaces {n : ℕ} (p1 p2 : LicensePlate n) : Prop :=
  ∃ i j : Fin n, i ≠ j ∧ p1 i ≠ p2 i ∧ p1 j ≠ p2 j

/-- A set of license plates where any two plates differ in at least two places -/
def ValidPlateSet {n : ℕ} (s : Set (LicensePlate n)) : Prop :=
  ∀ p1 p2, p1 ∈ s → p2 ∈ s → p1 ≠ p2 → DifferInTwoPlaces p1 p2

theorem max_license_plates (n : ℕ) (h : n ≥ 2) :
  ∃ (s : Set (LicensePlate n)), ValidPlateSet s ∧ 
  (∃ (f : s → Fin (10^(n-1))), Function.Bijective f) ∧
  ∀ (t : Set (LicensePlate n)), ValidPlateSet t → 
  (∃ (g : t → Fin (10^(n-1))), Function.Injective g) := by
  sorry

#check max_license_plates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_license_plates_l1131_113135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_full_path_l1131_113192

/-- Represents a friendship graph with 6 nodes --/
def FriendshipGraph := Fin 6 → Fin 6 → Prop

/-- A valid friendship graph satisfies the given conditions --/
def valid_friendship_graph (G : FriendshipGraph) : Prop :=
  ∀ i : Fin 6, (∃ j k l : Fin 6, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    G i j ∧ G i k ∧ G i l) ∧
  ∀ i j : Fin 6, G i j → G j i

/-- A path in the graph --/
inductive GraphPath (G : FriendshipGraph) : List (Fin 6) → Prop
  | nil : GraphPath G []
  | single : (i : Fin 6) → GraphPath G [i]
  | cons : (i j : Fin 6) → (rest : List (Fin 6)) → G i j → GraphPath G (j :: rest) → GraphPath G (i :: j :: rest)

/-- The main theorem: there exists a path that connects all 6 people --/
theorem exists_full_path (G : FriendshipGraph) (h : valid_friendship_graph G) :
  ∃ p : List (Fin 6), GraphPath G p ∧ p.length = 6 ∧ p.toFinset = Finset.univ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_full_path_l1131_113192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_cosine_graph_l1131_113182

open Real

noncomputable def f (x : ℝ) : ℝ := cos (2 * x + π / 4)

noncomputable def g (x : ℝ) : ℝ := f (x - π / 8)

-- Theorem statement
theorem shift_cosine_graph :
  ∀ x : ℝ, g x = cos (2 * x) := by
  intro x
  unfold g f
  simp [cos_add]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_cosine_graph_l1131_113182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_sum_l1131_113176

-- Define the square root of 3
noncomputable def sqrt3 : ℝ := Real.sqrt 3

-- Define the number we're working with
noncomputable def x : ℝ := 7 - 3 * sqrt3

-- Define a as the integer part of x
noncomputable def a : ℤ := ⌊x⌋

-- Define b as the fractional part of x
noncomputable def b : ℝ := x - (a : ℝ)

-- State the theorem
theorem integer_part_of_sum : ⌊(a : ℝ) + 9 / b⌋ = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_sum_l1131_113176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_arrangements_l1131_113129

def fibonacci : List ℕ := [1, 1, 2, 3, 5]

def is_valid_arrangement (arr : List ℕ) : Bool :=
  arr.length = 5 ∧ 
  arr.toFinset = fibonacci.toFinset ∧
  ¬(List.range 4).any (fun i => arr[i]? = some 1 ∧ arr[i+1]? = some 1)

theorem fibonacci_arrangements :
  (List.filter is_valid_arrangement (List.permutations fibonacci)).length = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_arrangements_l1131_113129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1131_113107

theorem tan_alpha_value (α : Real) 
  (h1 : Real.cos α = -4/5) 
  (h2 : π / 2 < α ∧ α < π) : 
  Real.tan α = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1131_113107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_in_interval_l1131_113115

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * (sin x)^2 + 2 * sqrt 3 * sin x * sin (x + π/2)

-- Theorem for the smallest positive period
theorem smallest_positive_period : 
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = π :=
sorry

-- Theorem for the range of f(x) in the interval [0, 2π/3]
theorem range_in_interval :
  ∀ (y : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 0 (2*π/3) ∧ f x = y) ↔ y ∈ Set.Icc 0 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_in_interval_l1131_113115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jing_not_implies_north_l1131_113142

-- Define the universe of candidates
variable (U : Type)

-- Define the sets of candidates for each alliance
variable (North Hua Excellence Jing : Set U)

-- State the conditions
variable (h1 : North ∩ Hua = ∅)
variable (h2 : Hua ⊆ Jing)
variable (h3 : Excellence ∩ Jing = ∅)
variable (h4 : (Set.univ \ Excellence) ⊆ Hua)

-- Theorem to prove
theorem jing_not_implies_north : ¬ (Jing ⊆ North) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jing_not_implies_north_l1131_113142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_fraction_l1131_113111

/-- The area of a right triangle with vertices at (3,3), (4,4), and (3,5) on a 6 by 6 grid
    is 1/36 of the total grid area. -/
theorem triangle_area_fraction : 
  let triangle_area := (1 : ℝ) / 2 * ((4 - 3) * (5 - 3) : ℝ)
  let total_area := (6 : ℝ) ^ 2
  triangle_area / total_area = 1 / 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_fraction_l1131_113111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_without_calls_eq_224_l1131_113175

/-- Represents the number of days in a year -/
def days_in_year : ℕ := 365

/-- Represents the calling frequency of the first child -/
def freq1 : ℕ := 4

/-- Represents the calling frequency of the second child -/
def freq2 : ℕ := 6

/-- Represents the calling frequency of the third child -/
def freq3 : ℕ := 9

/-- Calculates the number of days without calls in a year -/
def days_without_calls : ℕ :=
  days_in_year - (
    (days_in_year / freq1 + days_in_year / freq2 + days_in_year / freq3) -
    (days_in_year / Nat.lcm freq1 freq2 + days_in_year / Nat.lcm freq1 freq3 + days_in_year / Nat.lcm freq2 freq3) +
    days_in_year / Nat.lcm (Nat.lcm freq1 freq2) freq3
  )

theorem days_without_calls_eq_224 : days_without_calls = 224 := by
  -- The proof goes here
  sorry

#eval days_without_calls

end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_without_calls_eq_224_l1131_113175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_factors_of_450_l1131_113161

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- We need to make this function noncomputable due to the use of Finset.filter
noncomputable def count_perfect_square_factors (n : ℕ) : ℕ :=
  (Finset.filter (fun x => Nat.sqrt x * Nat.sqrt x = x) (Finset.range (n + 1))).card

theorem perfect_square_factors_of_450 :
  count_perfect_square_factors 450 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_factors_of_450_l1131_113161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_implies_lambda_l1131_113193

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is monotonic if it's either increasing or decreasing -/
def IsMonotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

/-- A function has only one zero point if there exists exactly one x such that f(x) = 0 -/
def HasUniqueZero (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

/-- Main theorem -/
theorem unique_zero_implies_lambda (f : ℝ → ℝ) (lambda : ℝ) 
    (h1 : IsOdd f)
    (h2 : IsMonotonic f)
    (h3 : HasUniqueZero (fun x => f (2 * x^2 + 1) + f (lambda - x))) :
    lambda = -7/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_implies_lambda_l1131_113193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1131_113189

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 2 3 → x * y ≤ a * x^2 + 2 * y^2) → 
  a ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1131_113189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_in_tray_is_30_l1131_113159

/-- The number of eggs in a tray, given the price per egg, tray price, and savings --/
def number_of_eggs_in_tray (price_per_egg : ℚ) (tray_price : ℚ) (savings_per_egg : ℚ) : ℕ :=
  Int.natAbs ((tray_price / (price_per_egg - savings_per_egg)).num)

/-- Theorem: The number of eggs in the tray is 30 --/
theorem eggs_in_tray_is_30 :
  number_of_eggs_in_tray (50 / 100) 12 (10 / 100) = 30 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_in_tray_is_30_l1131_113159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_of_30_l1131_113138

theorem sum_of_divisors_of_30 : 
  (Finset.filter (λ x ↦ 30 % x = 0) (Finset.range 31)).sum id = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_of_30_l1131_113138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_y_intercept_positive_l1131_113164

/-- Given a quadratic function f(x) = x^2 + 3x + m - 2, 
    if it intersects the y-axis in the positive half-axis, 
    then m > 2 -/
theorem quadratic_y_intercept_positive (m : ℝ) : 
  (λ x : ℝ ↦ x^2 + 3*x + m - 2) 0 > 0 → m > 2 := by
  intro h
  -- The y-intercept is f(0) = m - 2
  have y_intercept : (λ x : ℝ ↦ x^2 + 3*x + m - 2) 0 = m - 2 := by
    simp
  -- Substitute this into the hypothesis
  rw [y_intercept] at h
  -- Solve the inequality
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_y_intercept_positive_l1131_113164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_OMN_l1131_113163

/-- Given a point M(√2, 1) and a circle O with equation x^2 + y^2 = 1,
    the maximum value of angle OMN, where N is any point on circle O, is π/6. -/
theorem max_angle_OMN :
  let M : ℝ × ℝ := (Real.sqrt 2, 1)
  let O : ℝ × ℝ := (0, 0)
  let circle := {N : ℝ × ℝ | N.1^2 + N.2^2 = 1}
  ∃ (max_angle : ℝ), max_angle = π / 6 ∧
    ∀ N ∈ circle, Real.arctan ((N.2 - M.2) / (N.1 - M.1)) - Real.arctan (M.2 / M.1) ≤ max_angle :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_OMN_l1131_113163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_143_l1131_113191

theorem sum_of_divisors_143 : 
  (Finset.sum (Finset.filter (λ x : ℕ => 143 % x = 0) (Finset.range (143 + 1))) id) = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_143_l1131_113191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_iter_closed_form_l1131_113104

/-- Definition of the function f --/
noncomputable def f (x : ℝ) : ℝ := (1010 * x + 1009) / (1009 * x + 1010)

/-- Definition of the iterated function f^(n) --/
noncomputable def f_iter : ℕ → ℝ → ℝ
  | 0 => id
  | n + 1 => f ∘ f_iter n

/-- The main theorem stating the closed form of f^(n)(x) --/
theorem f_iter_closed_form (n : ℕ) (x : ℝ) :
  f_iter n x = ((2019^n + 1) * x + 2019^n - 1) / ((2019^n - 1) * x + 2019^n + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_iter_closed_form_l1131_113104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_equation_solution_l1131_113198

-- Define the expression
noncomputable def f (x : ℝ) : ℝ := (4 / (x^2 - 4)) - (1 / (x - 2))

-- Theorem 1: Simplification
theorem simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  f x = -1 / (x + 2) := by sorry

-- Theorem 2: Solution to the equation
theorem equation_solution :
  ∃ x : ℝ, x ≠ 2 ∧ x ≠ -2 ∧ f x = 1/2 ∧ x = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_equation_solution_l1131_113198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l1131_113125

-- Define the circle on which P moves
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- Define the coordinates of point Q
def Q : ℝ × ℝ := (4, 3)

-- Define the midpoint M of PQ
def is_midpoint (M P : ℝ × ℝ) : Prop :=
  M.1 = (P.1 + Q.1) / 2 ∧ M.2 = (P.2 + Q.2) / 2

-- Theorem statement
theorem midpoint_trajectory (x y : ℝ) :
  (∃ P : ℝ × ℝ, circle_eq P.1 P.2 ∧ is_midpoint (x, y) P) ↔
  (x - 3/2)^2 + (y - 3/2)^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l1131_113125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_nonzero_l1131_113167

def u : Fin 4 → ℝ := ![10, 5, 3, 2]
def v : Fin 4 → ℝ := ![7, 3, 1, 1]
def w : Fin 4 → ℝ := ![25, 15, 9, 6]
def x : Fin 4 → ℝ := ![4, 7, 5, 3]

def quadrilateral_vertices : Fin 4 → (Fin 4 → ℝ) := ![u, v, w, x]

-- Define a hypothetical function for quadrilateral area
noncomputable def quadrilateral_area (vertices : Fin 4 → (Fin 4 → ℝ)) : ℝ := 
  sorry -- Placeholder for the actual area calculation

theorem quadrilateral_area_nonzero :
  ∃ (area : ℝ), area > 0 ∧ area = quadrilateral_area quadrilateral_vertices :=
by
  sorry -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_nonzero_l1131_113167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_between_curves_l1131_113123

/-- The curve C₁ defined as y = x³ - x -/
def C₁ (x : ℝ) : ℝ := x^3 - x

/-- The curve C₂ defined as y = (x - a)³ - (x - a) -/
def C₂ (a x : ℝ) : ℝ := (x - a)^3 - (x - a)

/-- The area between C₁ and C₂ as a function of the shift parameter m -/
noncomputable def area (m : ℝ) : ℝ := (4 * m * (1 - m^2)^(3/2)) / Real.sqrt 3

/-- The theorem stating that the maximum area between C₁ and C₂ is 3/4 -/
theorem max_area_between_curves :
  ∃ (a : ℝ), (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ C₁ x₁ = C₂ a x₁ ∧ C₁ x₂ = C₂ a x₂) →
  (∀ m : ℝ, 0 < m → m < 1 → area m ≤ 3/4) ∧
  (∃ m : ℝ, 0 < m ∧ m < 1 ∧ area m = 3/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_between_curves_l1131_113123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_equals_51_l1131_113140

/-- The sum of the alternating series from 2 to 100 -/
def alternating_sum : ℕ → ℤ
| 0 => 0
| n + 1 => alternating_sum n + if n % 2 == 0 then (n + 2 : ℤ) else -(n + 2 : ℤ)

/-- The number of terms in the series -/
def num_terms : ℕ := 99

theorem alternating_sum_equals_51 : alternating_sum num_terms = 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_equals_51_l1131_113140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_evaluate_l1131_113188

theorem simplify_and_evaluate : 
  ∀ x : ℝ, x = 2 → 
  (1 / (x^2 - 1)) / (x / (x^2 - 2*x + 1)) - 2 / (x + 1) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_evaluate_l1131_113188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_average_decrease_main_bowling_theorem_l1131_113196

/-- Represents a bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  initialWickets : ℝ
  lastMatchWickets : ℝ
  lastMatchRuns : ℝ

/-- Calculates the new bowling average after the last match -/
noncomputable def newAverage (stats : BowlerStats) : ℝ :=
  (stats.initialAverage * stats.initialWickets + stats.lastMatchRuns) /
  (stats.initialWickets + stats.lastMatchWickets)

/-- Theorem stating the decrease in bowling average -/
theorem bowling_average_decrease (stats : BowlerStats)
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.initialWickets = 25)
  (h3 : stats.lastMatchWickets = 3)
  (h4 : stats.lastMatchRuns = 26) :
  stats.initialAverage - newAverage stats = 0.4 := by
  sorry

/-- Main theorem to be proved -/
theorem main_bowling_theorem : ∃ (stats : BowlerStats),
  stats.initialAverage = 12.4 ∧
  stats.initialWickets = 25 ∧
  stats.lastMatchWickets = 3 ∧
  stats.lastMatchRuns = 26 ∧
  stats.initialAverage - newAverage stats = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_average_decrease_main_bowling_theorem_l1131_113196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l1131_113194

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : Real.sin α = 5/13) : 
  Real.tan (α + π/4) = 7/17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l1131_113194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_midpoint_l1131_113187

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop :=
  x - y + 2 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse A.fst A.snd ∧ ellipse B.fst B.snd ∧
  line A.fst A.snd ∧ line B.fst B.snd ∧
  A ≠ B

-- Theorem statement
theorem ellipse_line_intersection_midpoint :
  ∀ A B : ℝ × ℝ,
  intersection_points A B →
  let midpoint := ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)
  midpoint = (-9/5, 1/5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_midpoint_l1131_113187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1131_113100

noncomputable def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a else a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (a : ℝ) (r : ℝ) :
  (geometric_sum a r 2 = 7) →
  (geometric_sum a r 6 = 91) →
  (geometric_sum a r 4 = 28) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1131_113100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_d_value_l1131_113148

theorem smallest_d_value : ∃ d : ℝ, d > 0 ∧ 
  (∀ d' : ℝ, d' > 0 → (4 * Real.sqrt 5)^2 + (2 * d' - 4)^2 = (4 * d')^2 → d ≤ d') ∧ 
  (4 * Real.sqrt 5)^2 + (2 * d - 4)^2 = (4 * d)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_d_value_l1131_113148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_properties_l1131_113127

/-- A function satisfying the given properties -/
structure SpecialFunction where
  f : ℝ → ℝ
  add : ∀ x y, f (x + y) = f x + f y
  one_third : f (1/3) = 1
  pos : ∀ x, x > 0 → f x > 0

theorem special_function_properties (F : SpecialFunction) :
  (F.f 0 = 0) ∧ 
  (∀ x, F.f (-x) = -F.f x) ∧
  ({x : ℝ | F.f x + F.f (2 + x) < 2} = Set.Ioi (-2/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_properties_l1131_113127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_element_group_l1131_113126

/-- Represents the groups in the periodic table -/
inductive PeriodicGroup
| IA
| IIA
| IIIA
| IIIB
| IVA

/-- Represents an element in the periodic table -/
structure Element where
  atomicNumber : ℕ
  group : PeriodicGroup

/-- Defines the relationship between adjacent elements in the periodic table -/
def adjacentElementRelation (e1 e2 : Element) : Prop :=
  e2.atomicNumber = e1.atomicNumber + 1

/-- Theorem: Given an element in Group IIA, the next element is either in Group IIIA or Group IIIB -/
theorem next_element_group (e1 e2 : Element) :
  e1.group = PeriodicGroup.IIA →
  adjacentElementRelation e1 e2 →
  e2.group = PeriodicGroup.IIIA ∨ e2.group = PeriodicGroup.IIIB := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_element_group_l1131_113126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l1131_113136

-- Helper definitions
def is_arithmetic_seq (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_seq (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem arithmetic_geometric_sequence_ratio : ∀ (a₁ a₂ b₁ b₂ b₃ : ℝ),
  (is_arithmetic_seq 1 a₁ a₂ 4) →
  (is_geometric_seq 1 b₁ b₂ b₃ 4) →
  (a₁ + a₂) / b₂ = 5/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l1131_113136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_arrangements_l1131_113137

def digits : List Nat := [2, 0, 2, 1]

def is_valid_arrangement (arrangement : List Nat) : Bool :=
  arrangement.length = 4 ∧ 
  arrangement.head? ≠ some 0 ∧ 
  arrangement.getLast? ≠ some 0 ∧
  (arrangement.toFinset = digits.toFinset)

def valid_arrangements : Finset (List Nat) :=
  Finset.filter (fun arr => is_valid_arrangement arr) (List.permutations digits).toFinset

theorem count_valid_arrangements : valid_arrangements.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_arrangements_l1131_113137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simon_lego_count_l1131_113134

theorem simon_lego_count (kent_legos : ℕ) (bruce_extra_legos : ℕ) (simon_legos : ℕ) :
  kent_legos = 40 →
  simon_legos = 72 →
  simon_legos = (kent_legos + bruce_extra_legos) * (6/5 : ℚ) →
  simon_legos = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simon_lego_count_l1131_113134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l1131_113103

theorem expression_equality : (1/2)⁻¹ - 27⁻¹/3 - Real.log 4 / Real.log 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l1131_113103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_alpha_l1131_113158

theorem parallel_vectors_tan_alpha (α : ℝ) :
  (3 : ℝ) * Real.cos α = 4 * Real.sin α →
  Real.tan α = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_alpha_l1131_113158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annealing_temperature_optimization_l1131_113119

/-- The 0.618 method constant -/
noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

/-- Calculate the first test point using the 0.618 method -/
noncomputable def first_test_point (lower upper : ℝ) : ℝ :=
  lower + (1 - 1 / golden_ratio) * (upper - lower)

/-- Calculate the second test point using the 0.618 method -/
noncomputable def second_test_point (lower upper : ℝ) : ℝ :=
  upper - (first_test_point lower upper - lower)

/-- The annealing temperature optimization problem -/
theorem annealing_temperature_optimization
  (lower upper : ℝ)
  (h_lower : lower = 1400)
  (h_upper : upper = 1600) :
  (second_test_point lower upper = 1523.6 ∨ second_test_point lower upper = 1476.4) ∧
  (first_test_point lower upper = 1523.6 ∨ first_test_point lower upper = 1476.4) :=
by sorry

#eval println! "Annealing temperature optimization theorem defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annealing_temperature_optimization_l1131_113119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l1131_113113

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  4 * x^2 - 24 * x - y^2 - 6 * y + 34 = 0

-- Define the distance between vertices
noncomputable def vertex_distance (eq : (ℝ → ℝ → Prop)) : ℝ :=
  Real.sqrt 7

-- Theorem statement
theorem hyperbola_vertex_distance :
  vertex_distance hyperbola_eq = Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l1131_113113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_n_equals_two_l1131_113124

theorem power_equality_implies_n_equals_two (n : ℝ) : (9 : ℝ)^3 = (27 : ℝ)^n → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_n_equals_two_l1131_113124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_75_squared_tan_sum_product_l1131_113199

-- Define the constant π as noncomputable
noncomputable def π : ℝ := Real.pi

-- Define degree to radian conversion as noncomputable
noncomputable def deg_to_rad (x : ℝ) : ℝ := x * π / 180

-- Statement for the first identity
theorem cos_75_squared : 
  Real.cos (deg_to_rad 75) ^ 2 = (2 - Real.sqrt 3) / 4 := by sorry

-- Statement for the second identity
theorem tan_sum_product : 
  Real.tan (deg_to_rad 1) + Real.tan (deg_to_rad 44) + 
  Real.tan (deg_to_rad 1) * Real.tan (deg_to_rad 44) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_75_squared_tan_sum_product_l1131_113199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1131_113133

noncomputable def train_speed_with_stops (speed_without_stops : ℝ) (stop_time : ℝ) : ℝ :=
  speed_without_stops * (60 - stop_time) / 60

theorem train_speed_theorem (speed_without_stops : ℝ) (stop_time : ℝ) 
  (h1 : speed_without_stops = 48)
  (h2 : stop_time = 10) :
  train_speed_with_stops speed_without_stops stop_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1131_113133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1131_113155

open Real

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := cos (2*x - 4*π/3) + 2*(cos x)^2

-- Define the theorem
theorem f_properties :
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (y : ℝ), f y = M) ∧
  (∀ (A B C : ℝ), 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → 
    f (B + C) = 3/2 → A = π/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1131_113155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hueys_hip_pizza_theorem_l1131_113109

/-- The side length of the large pizza at Huey's Hip Pizza --/
noncomputable def large_pizza_side : ℝ := Real.sqrt 78

/-- The side length of the small pizza at Huey's Hip Pizza --/
def small_pizza_side : ℝ := 6

/-- The cost of the small pizza at Huey's Hip Pizza --/
def small_pizza_cost : ℝ := 10

/-- The cost of the large pizza at Huey's Hip Pizza --/
def large_pizza_cost : ℝ := 20

/-- The amount of money each friend has --/
def friend_money : ℝ := 30

/-- The additional pizza area gained by pooling money --/
def additional_area : ℝ := 9

theorem hueys_hip_pizza_theorem :
  (3 * large_pizza_side ^ 2) = 
  ((friend_money / small_pizza_cost) * small_pizza_side ^ 2 +
   (friend_money / large_pizza_cost) * large_pizza_side ^ 2 +
   additional_area) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hueys_hip_pizza_theorem_l1131_113109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeannie_hike_time_l1131_113122

/-- Calculates the total time for Jeannie's hike given the distances and paces --/
theorem jeannie_hike_time 
  (distance_to_overlook : ℝ) 
  (pace_to_overlook : ℝ) 
  (distance_overlook_to_lake : ℝ) 
  (pace_overlook_to_lake : ℝ) 
  (return_pace : ℝ) 
  (h1 : distance_to_overlook = 12)
  (h2 : pace_to_overlook = 4)
  (h3 : distance_overlook_to_lake = 9)
  (h4 : pace_overlook_to_lake = 3)
  (h5 : return_pace = 6) :
  (distance_to_overlook / pace_to_overlook) +
  (distance_overlook_to_lake / pace_overlook_to_lake) +
  ((distance_to_overlook + distance_overlook_to_lake) / return_pace) = 9.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeannie_hike_time_l1131_113122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_cosine_graph_l1131_113131

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos ((1/2) * x + Real.pi/3)

theorem symmetry_center_of_cosine_graph :
  ∃ (c : ℝ × ℝ), c.1 = Real.pi/3 ∧ c.2 = 0 ∧
  ∀ (x : ℝ), f (2 * c.1 - x) = f x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_cosine_graph_l1131_113131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_difference_l1131_113101

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi/3) * Real.cos x

theorem triangle_cosine_difference (A B : ℝ) (a b c : ℝ) : 
  0 < A → A < Real.pi/2 →  -- A is acute
  f A = Real.sqrt 3/2 →
  b = 2 →
  c = 3 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →  -- Cosine rule
  Real.sin A / a = Real.sin B / b →  -- Sine rule
  Real.cos (A - B) = 5 * Real.sqrt 7 / 14 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_difference_l1131_113101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_waste_calculation_l1131_113114

/-- The radius of the original circular piece of metal -/
noncomputable def R : ℝ := 10

/-- The side length of the largest square that can be cut from the circle -/
noncomputable def a : ℝ := R * Real.sqrt 2

/-- The radius of the largest circle that can be cut from the square -/
noncomputable def r : ℝ := a / 2

/-- The total area of metal wasted -/
noncomputable def wastedArea : ℝ := Real.pi * R^2 - a^2 + a^2 - Real.pi * r^2

theorem metal_waste_calculation :
  wastedArea = 50 * Real.pi - 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_waste_calculation_l1131_113114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l1131_113165

/-- The rational function under consideration -/
noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x^3 - 3*x^2 - 4*x)

/-- The number of holes in the graph of f -/
def p : ℕ := 2

/-- The number of vertical asymptotes of f -/
def q : ℕ := 1

/-- The number of horizontal asymptotes of f -/
def r : ℕ := 1

/-- The number of oblique asymptotes of f -/
def s : ℕ := 0

/-- Theorem stating that p + 2q + 3r + 4s = 7 for the function f -/
theorem asymptote_sum : p + 2*q + 3*r + 4*s = 7 := by
  -- Unfold the definitions
  unfold p q r s
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l1131_113165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_popcorn_fraction_l1131_113185

/-- Proves that the fraction of popcorn eaten by the squirrel is 1/4 -/
theorem squirrel_popcorn_fraction 
  (distance_to_school : ℕ) 
  (drop_interval : ℕ) 
  (remaining_kernels : ℕ) 
  (h1 : distance_to_school = 5000)
  (h2 : drop_interval = 25)
  (h3 : remaining_kernels = 150)
  : (distance_to_school / drop_interval - remaining_kernels : ℚ) / (distance_to_school / drop_interval) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_popcorn_fraction_l1131_113185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_nested_calculation_l1131_113153

noncomputable def p (x y : ℝ) : ℝ :=
  if x ≥ 0 ∧ y ≥ 0 then x * y
  else if x < 0 ∧ y < 0 then x - 2 * y
  else if x ≥ 0 ∧ y < 0 then 2 * x + 3 * y
  else if x < 0 ∧ y ≥ 0 then x + 3 * y
  else 3 * x + y

theorem p_nested_calculation : p (p 2 (-3)) (p (-1) 4) = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_nested_calculation_l1131_113153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_tournament_draws_l1131_113177

theorem football_tournament_draws (n : ℕ) (total_points : ℕ) : 
  n = 20 →
  total_points = 500 →
  let total_matches := n * (n - 1) / 2;
  let max_points := 3 * total_matches;
  (max_points - total_points : ℕ) = 70 := by
    intro h_n h_total_points
    sorry

#check football_tournament_draws

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_tournament_draws_l1131_113177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_solutions_l1131_113108

-- Define the interval
def interval : Set ℝ := Set.Ioo 0 (150 * Real.pi)

-- Define the equation
def equation (x : ℝ) : Prop := Real.sin x = (1/3) ^ x

-- Define the solution set
def solution_set : Set ℝ := {x ∈ interval | equation x}

-- Theorem statement
theorem number_of_solutions : ∃ (s : Finset ℝ), s.card = 75 ∧ ∀ x ∈ s, x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_solutions_l1131_113108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retreat_handshakes_l1131_113183

/-- Represents a group of people at a corporate retreat -/
structure RetreatGroup where
  total : ℕ
  employees : ℕ
  interns : ℕ
  knowledgeable_intern : ℕ
  employee_connections : ℕ

/-- Calculates the number of handshakes in the group -/
def handshakes (g : RetreatGroup) : ℕ :=
  let unknown_interns := g.interns - 1
  let unknown_intern_handshakes := unknown_interns * (g.total - 1)
  let knowledgeable_intern_handshakes := g.total - g.employee_connections - 1
  unknown_intern_handshakes + knowledgeable_intern_handshakes

/-- The corporate retreat group -/
def retreat_group : RetreatGroup where
  total := 40
  employees := 25
  interns := 15
  knowledgeable_intern := 1
  employee_connections := 5

theorem retreat_handshakes :
  handshakes retreat_group = 580 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_retreat_handshakes_l1131_113183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1131_113166

/-- Given two vectors AB and a in ℝ², prove that if they are parallel and
    AB = (3,1) and a = (2,lambda), then lambda = 2/3 -/
theorem parallel_vectors_lambda (AB a : ℝ × ℝ) (lambda : ℝ) :
  AB = (3, 1) →
  a = (2, lambda) →
  (∃ (k : ℝ), a = k • AB) →
  lambda = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1131_113166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_winning_vertex_l1131_113173

-- Define the vertex type
inductive Vertex
  | v0 | v1 | v2 | v3 | v4
deriving Repr, DecidableEq

-- Define the game state
structure GameState where
  values : Vertex → Int
  sum_constraint : values Vertex.v0 + values Vertex.v1 + values Vertex.v2 + values Vertex.v3 + values Vertex.v4 = 2011

-- Define the invariant quantity S
def S (state : GameState) : Int :=
  (state.values Vertex.v1 + 2 * state.values Vertex.v2 + 
   3 * state.values Vertex.v3 + 4 * state.values Vertex.v4) % 5

-- Define a game move
def move (state : GameState) (v1 v2 v3 : Vertex) (m : Int) : GameState :=
  { values := fun v =>
      if v = v1 ∨ v = v2 then state.values v - m
      else if v = v3 then state.values v + 2*m
      else state.values v,
    sum_constraint := by sorry }

-- Define the winning condition
def is_winning (state : GameState) (v : Vertex) : Prop :=
  state.values v = 2011 ∧ ∀ u, u ≠ v → state.values u = 0

-- The main theorem
theorem unique_winning_vertex (initial : GameState) :
  ∃! v, ∃ final : GameState, is_winning final v ∧ 
  ∃ moves : List (Vertex × Vertex × Vertex × Int), 
    final = moves.foldl (fun s (v1, v2, v3, m) => move s v1 v2 v3 m) initial :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_winning_vertex_l1131_113173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1131_113141

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 2 4, x^2 - 2*x - 2*a ≤ 0

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

def q (a : ℝ) : Prop := MonotoneOn (f a) (Set.Ici (1/2))

-- Define the range of a
def range_a : Set ℝ := Set.Iic 1 ∪ Set.Ici 4

-- State the theorem
theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ range_a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1131_113141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_three_digit_square_with_digit_product_l1131_113139

-- Helper function to get digits of a number
def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

-- Helper function to calculate the product of a list of natural numbers
def prod_list (l : List ℕ) : ℕ :=
  l.foldl (·*·) 1

theorem unique_three_digit_square_with_digit_product (n : ℕ) : 
  (100 ≤ n ∧ n ≤ 999) ∧  -- n is a three-digit number
  (∃ H : ℕ, n = H^2) ∧   -- n is a perfect square
  (prod_list (digits n) = Nat.sqrt n - 1) -- product of digits equals H-1
  ↔ 
  n = 324 := by
sorry

#eval digits 324
#eval prod_list (digits 324)
#eval Nat.sqrt 324 - 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_three_digit_square_with_digit_product_l1131_113139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_AB_l1131_113171

-- Define the rectangles and points
structure SmallRectangle where
  perimeter : ℝ
  h : perimeter = 10

structure LargeRectangle where
  area : ℝ
  h : area = 24

def PointA := ℝ × ℝ
def PointB := ℝ × ℝ

-- Define the theorem
theorem shortest_distance_AB (small : SmallRectangle) (large : LargeRectangle) 
  (A : PointA) (B : PointB) :
  -- Conditions
  (∃ w h, w * h = large.area ∧ A = (w, h)) →
  (∃ w h, 2 * (w + h) = small.perimeter ∧ B = (0, 0)) →
  -- Conclusion
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 52 := by
  sorry

#check shortest_distance_AB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_AB_l1131_113171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_slope_angle_and_intercept_l1131_113118

/-- Given a line with a slope angle of 60° and an x-axis intercept of √3, 
    its equation is √3x - y - 3 = 0 --/
theorem line_equation_from_slope_angle_and_intercept :
  ∀ (m b : ℝ),
    m = Real.sqrt 3 →
    b = -3 →
    ∀ (x y : ℝ),
      y = m * x + b ↔ Real.sqrt 3 * x - y - 3 = 0 :=
by
  intros m b hm hb x y
  rw [hm, hb]
  constructor
  · intro h
    rw [h]
    ring
  · intro h
    linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_slope_angle_and_intercept_l1131_113118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_f_range_of_b_when_a_is_zero_h_has_fixed_point_l1131_113184

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log x + 1

def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 3

def h (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- Theorem statements
theorem fixed_point_f : ∃ x : ℝ, f x = x ∧ x = 1 := by
  sorry

theorem range_of_b_when_a_is_zero (a b c : ℝ) :
  a = 0 →
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (1/2) 2 ∧ g a b c x₀ = x₀ ∧ h a b c x₀ = x₀) →
  b ∈ Set.Icc (5/4) 11 := by
  sorry

theorem h_has_fixed_point (a b c : ℝ) (m : ℝ) :
  a ≠ 0 →
  (∃ q : ℝ, q > 0 ∧ h a b c m = q * m ∧ h a b c (h a b c m) = q * (h a b c m) ∧
    h a b c (h a b c (h a b c m)) = q * (h a b c (h a b c m))) →
  ∃ x : ℝ, h a b c x = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_f_range_of_b_when_a_is_zero_h_has_fixed_point_l1131_113184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_satisfies_conditions_l1131_113195

noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

noncomputable def intersection_point (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ × ℝ :=
  let x := (b₁ * c₂ - b₂ * c₁) / (a₁ * b₂ - a₂ * b₁)
  let y := (a₂ * c₁ - a₁ * c₂) / (a₁ * b₂ - a₂ * b₁)
  (x, y)

theorem line_satisfies_conditions :
  let line1 : ℝ → ℝ → ℝ := fun x y ↦ x + 2 * y - 3
  let line2 : ℝ → ℝ → ℝ := fun x y ↦ 2 * x - y - 1
  let target_line : ℝ → ℝ → ℝ := fun x _ ↦ x - 1
  let intersection := intersection_point 1 2 (-3) 2 (-1) (-1)
  (target_line intersection.1 intersection.2 = 0) ∧
  (distance_point_to_line 0 1 1 0 (-1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_satisfies_conditions_l1131_113195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anthony_free_throw_improvement_l1131_113154

/-- Calculates the increase in success rate percentage for free throw attempts -/
def free_throw_success_rate_increase 
  (initial_attempts : ℕ) 
  (initial_success : ℕ) 
  (next_attempts : ℕ) 
  (next_success_rate : ℚ) : ℕ :=
  let initial_rate := (initial_success : ℚ) / initial_attempts
  let next_success := (next_success_rate * next_attempts).floor
  let total_attempts := initial_attempts + next_attempts
  let total_success := initial_success + next_success
  let new_rate := (total_success : ℚ) / total_attempts
  let increase := (new_rate - initial_rate) * 100
  (increase + 0.5).floor.toNat

/-- Main theorem applying the calculation to the given problem -/
theorem anthony_free_throw_improvement : 
  free_throw_success_rate_increase 10 4 28 (3/4) = 26 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anthony_free_throw_improvement_l1131_113154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_congruence_l1131_113102

/-- Sum of digits in base q -/
def S_q (q : ℕ) (x : ℕ) : ℕ := sorry

theorem digit_sum_congruence 
  (a b b' c m q : ℕ) 
  (hm : m > 1)
  (hq : q > 1)
  (hbb' : |Int.ofNat b - Int.ofNat b'| ≥ a)
  (hM : ∃ M : ℕ, ∀ n ≥ M, S_q q (a * n + b) ≡ S_q q (a * n + b') + c [MOD m]) :
  ∀ n : ℕ, S_q q (a * n + b) ≡ S_q q (a * n + b') + c [MOD m] := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_congruence_l1131_113102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_egg_weight_l1131_113172

theorem chocolate_egg_weight 
  (total_eggs : ℕ) 
  (total_boxes : ℕ) 
  (discarded_boxes : ℕ) 
  (remaining_weight : ℝ) 
  (h1 : total_eggs = 12)
  (h2 : total_boxes = 4)
  (h3 : discarded_boxes = 1)
  (h4 : remaining_weight = 90) :
  (remaining_weight / ((total_eggs / total_boxes) * (total_boxes - discarded_boxes))) = 10 := by
  have remaining_boxes : ℕ := total_boxes - discarded_boxes
  have eggs_per_box : ℕ := total_eggs / total_boxes
  have weight_per_egg : ℝ := remaining_weight / (eggs_per_box * remaining_boxes)
  
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_egg_weight_l1131_113172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1131_113120

noncomputable def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = 6 * Real.pi) ∧
  (∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M) ∧ M = Real.sqrt 2) :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1131_113120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_and_quotient_determine_divisor_l1131_113130

theorem remainder_and_quotient_determine_divisor
  (x y : ℕ) -- x and y are natural numbers
  (hx : x > 0) -- x is positive
  (hy : y > 0) -- y is positive
  (h_remainder : x % y = 9) -- remainder when x is divided by y is 9
  (h_quotient : (x : ℚ) / (y : ℚ) = 96.45) -- x / y as a rational number equals 96.45
  : y = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_and_quotient_determine_divisor_l1131_113130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_value_at_point_M_independent_of_x_l1131_113160

-- Define the polynomial M
def M (x y : ℝ) : ℝ := (2*x^2 + 3*x*y + 2*y) - 2*(x^2 + x + y*x + 1)

-- Theorem 1: M = 2 when x = 1 and y = 2
theorem M_value_at_point : M 1 2 = 2 := by sorry

-- Theorem 2: M is independent of x if and only if y = 2
theorem M_independent_of_x (y : ℝ) : (∀ x, deriv (fun x => M x y) x = 0) ↔ y = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_value_at_point_M_independent_of_x_l1131_113160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_equation_l1131_113150

theorem smallest_positive_solution_tan_equation :
  let f : ℝ → ℝ := λ x => Real.tan (4 * x) + Real.tan (6 * x) - (1 / Real.cos (6 * x)) - 1
  ∃ x : ℝ, x > 0 ∧ f x = 0 ∧ ∀ y : ℝ, y > 0 → f y = 0 → x ≤ y ∧ x = π / 28 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_equation_l1131_113150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_232_point_5_l1131_113112

/-- Represents the number of minutes it takes to travel one mile on the first day -/
def initial_minutes_per_mile : ℕ := 1

/-- Calculates the distance traveled on a given day -/
def distance_on_day (day : ℕ) : ℚ :=
  120 / (initial_minutes_per_mile * 2^(day - 1))

/-- Conditions of the problem -/
axiom initial_minutes_divides_120 : 120 % initial_minutes_per_mile = 0
axiom initial_minutes_less_than_120 : initial_minutes_per_mile ≤ 120
axiom distance_is_integer (day : ℕ) : day ≥ 1 → day ≤ 5 → ∃ (n : ℕ), distance_on_day day = n

/-- The main theorem to prove -/
theorem total_distance_is_232_point_5 : 
  (distance_on_day 1) + (distance_on_day 2) + (distance_on_day 3) + 
  (distance_on_day 4) + (distance_on_day 5) = 232.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_232_point_5_l1131_113112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1131_113180

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := x / (2 * x - 1)

-- Define the point of tangency
def point : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ x + y - 2 = 0) ∧
    (m = -(1 / (2 * point.1 - 1)^2)) ∧
    (point.2 = m * point.1 + b) ∧
    (HasDerivAt f m point.1) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1131_113180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_correlation_theorem_l1131_113106

noncomputable section

-- Define a random function X(t)
variable (X : ℝ → ℝ → ℝ)

-- Define the correlation function K_x
def K_x (X : ℝ → ℝ → ℝ) (t₁ t₂ : ℝ) : ℝ := sorry

-- Define the cross-correlation function R_{x dot_x}
def R_x_dotx (X : ℝ → ℝ → ℝ) (t₁ t₂ : ℝ) : ℝ := sorry

-- Define the cross-correlation function R_{dot_x x}
def R_dotx_x (X : ℝ → ℝ → ℝ) (t₁ t₂ : ℝ) : ℝ := sorry

-- State the theorem
theorem cross_correlation_theorem (X : ℝ → ℝ → ℝ) :
  (∀ t₁ t₂ : ℝ, R_x_dotx X t₁ t₂ = deriv (fun t₂ => K_x X t₁ t₂) t₂) ∧
  (∀ t₁ t₂ : ℝ, R_dotx_x X t₁ t₂ = deriv (fun t₁ => K_x X t₁ t₂) t₁) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_correlation_theorem_l1131_113106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_19_l1131_113190

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- Main theorem -/
theorem arithmetic_sequence_sum_19 (seq : ArithmeticSequence) 
  (h1 : seq.a 6 + seq.a 10 - seq.a 12 = 8)
  (h2 : seq.a 14 - seq.a 8 = 4) :
  S seq 19 = 228 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_19_l1131_113190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_five_consecutive_integers_l1131_113157

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, 
  ∃ m : ℕ, 
    m = 60 ∧ 
    (∀ k : ℤ, m ∣ (((n + k) * (n + k + 1) * (n + k + 2) * (n + k + 3) * (n + k + 4)).natAbs)) ∧
    (∀ l : ℕ, l > m → ∃ j : ℤ, ¬(l ∣ ((j + 0) * (j + 1) * (j + 2) * (j + 3) * (j + 4)).natAbs)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_five_consecutive_integers_l1131_113157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_roots_problem_l1131_113105

theorem square_roots_problem (a m : ℝ) (h1 : a > 0) (h2 : (3 * m - 1)^2 = a) (h3 : (7 - 5 * m)^2 = a) :
  m = 1 ∧ a = 64 ∧ -(Real.rpow (-a) (1/3)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_roots_problem_l1131_113105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_equation_solutions_prime_equation_with_sum_solutions_l1131_113132

theorem prime_equation_solutions (p : ℕ) (hp : Prime p) :
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ p * (x - 2) = x * (y - 1)) ↔
  ((2, 1) ∈ {(x, y) : ℕ × ℕ | x > 0 ∧ y > 0 ∧ p * (x - 2) = x * (y - 1)} ∧
   (p, p - 1) ∈ {(x, y) : ℕ × ℕ | x > 0 ∧ y > 0 ∧ p * (x - 2) = x * (y - 1)} ∧
   (2 * p, p) ∈ {(x, y) : ℕ × ℕ | x > 0 ∧ y > 0 ∧ p * (x - 2) = x * (y - 1)}) :=
by sorry

theorem prime_equation_with_sum_solutions (p : ℕ) (hp : Prime p) :
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ p * (x - 2) = x * (y - 1) ∧ x + y = 21) ↔
  ((11, 10, 11) ∈ {(x, y, p) : ℕ × ℕ × ℕ | x > 0 ∧ y > 0 ∧ p * (x - 2) = x * (y - 1) ∧ x + y = 21} ∧
   (14, 7, 7) ∈ {(x, y, p) : ℕ × ℕ × ℕ | x > 0 ∧ y > 0 ∧ p * (x - 2) = x * (y - 1) ∧ x + y = 21}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_equation_solutions_prime_equation_with_sum_solutions_l1131_113132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_divisors_l1131_113146

theorem consecutive_divisors (N : ℤ) : 
  (∀ a : ℕ, (∀ i : ℕ, i ∈ Finset.range 10 → (a + i) ∣ N.natAbs)) ∧
  (¬∃ a : ℕ, (∀ i : ℕ, i ∈ Finset.range 11 → (a + i) ∣ N.natAbs)) ↔
  ∃ k : ℕ, k > 0 ∧ ¬(11 ∣ k) ∧ N = k * (Nat.factorial 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_divisors_l1131_113146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_ratio_proof_l1131_113186

/-- The ratio of map distance to actual distance -/
noncomputable def map_to_actual_ratio (actual_distance : ℝ) (map_distance : ℝ) : ℝ :=
  map_distance / (actual_distance * 100)

/-- Theorem stating that the ratio of map distance to actual distance is 1/5000 -/
theorem map_ratio_proof (actual_distance map_distance : ℝ) 
  (h1 : actual_distance = 250) 
  (h2 : map_distance = 5) : 
  map_to_actual_ratio actual_distance map_distance = 1 / 5000 := by
  sorry

#check map_ratio_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_ratio_proof_l1131_113186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_with_symmetric_roots_l1131_113110

/-- A polynomial of degree 6 with real coefficients and constant term 2024 -/
def MyPolynomial (a b c d e : ℝ) : ℂ → ℂ :=
  fun x => x^6 + a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + 2024

/-- Property that if r is a root, then i·r is also a root -/
def HasSymmetricRoots (P : ℂ → ℂ) : Prop :=
  ∀ r : ℂ, P r = 0 → P (Complex.I * r) = 0

/-- There exists exactly one polynomial with the required property -/
theorem unique_polynomial_with_symmetric_roots :
  ∃! (a b c d e : ℝ), HasSymmetricRoots (MyPolynomial a b c d e) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_with_symmetric_roots_l1131_113110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_value_l1131_113147

/-- The width of the grid -/
noncomputable def grid_width : ℝ := 15

/-- The height of the grid -/
noncomputable def grid_height : ℝ := 5

/-- The base of the unshaded triangle -/
noncomputable def triangle_base : ℝ := 15

/-- The height of the unshaded triangle -/
noncomputable def triangle_height : ℝ := 5

/-- The area of the shaded region in the grid -/
noncomputable def shaded_area : ℝ := grid_width * grid_height - (triangle_base * triangle_height / 2)

theorem shaded_area_value : shaded_area = 37.5 := by
  -- Unfold the definitions
  unfold shaded_area grid_width grid_height triangle_base triangle_height
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_value_l1131_113147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1131_113174

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := -1/3 * x^3 + x^2

-- Define the interval [0, 4]
def interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ 4 }

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ interval ∧ f x = 4/3 ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ f x :=
by
  -- The proof goes here
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1131_113174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_remaining_numbers_l1131_113121

theorem mean_of_remaining_numbers (numbers : Finset ℕ) : 
  numbers = {2347, 2573, 2689, 2725, 2839, 2841} →
  ∃ (subset : Finset ℕ), subset ⊆ numbers ∧ subset.card = 4 ∧ 
    (↑(subset.sum id) / 4 : ℚ) = 2666 →
  ∃ (remaining : Finset ℕ), remaining = numbers \ subset ∧
    (↑(remaining.sum id) / 2 : ℚ) = 2675 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_remaining_numbers_l1131_113121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_conditions_l1131_113117

variable (m : ℝ)
def z (m : ℝ) : ℂ := m + 1 + (m - 1) * Complex.I

theorem complex_number_conditions (m : ℝ) :
  ((z m).im = 0 ↔ m = 1) ∧
  ((z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = -1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_conditions_l1131_113117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_relation_l1131_113179

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := Real.sqrt ((a^2 - b^2) / a^2)

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt ((a^2 + b^2) / a^2)

theorem eccentricity_relation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ellipse_eccentricity a b = Real.sqrt 2 / 2) : 
  hyperbola_eccentricity a b = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_relation_l1131_113179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_graph_l1131_113151

/-- The area enclosed by the graph of |x| + |3y| = 12 is 96 square units. -/
theorem area_enclosed_by_graph : 
  let graph := {(x, y) : ℝ × ℝ | |x| + |3 * y| = 12}
  ∃ A : Set (ℝ × ℝ), A = graph ∧ MeasureTheory.volume A = 96 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_graph_l1131_113151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_real_l1131_113170

/-- The common ratio of a geometric sequence -/
def common_ratio : Type := ℝ

/-- The common ratio is non-zero -/
axiom common_ratio_nonzero (q : common_ratio) : q ≠ (0 : ℝ)

/-- The common ratio can be any real number except 0 -/
theorem common_ratio_real (q : ℝ) (h : q ≠ 0) : ∃ (r : common_ratio), q = r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_real_l1131_113170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_income_tax_calculation_l1131_113169

noncomputable def base_tax_rate (q : ℝ) : ℝ := q / 100

noncomputable def higher_tax_rate (q : ℝ) : ℝ := (q + 1.5) / 100

noncomputable def johns_income : ℝ := 34615

noncomputable def total_tax_rate (q : ℝ) : ℝ := (q + 0.20) / 100

theorem johns_income_tax_calculation (q : ℝ) :
  let base_tax := base_tax_rate q * 30000
  let higher_tax := higher_tax_rate q * (johns_income - 30000)
  base_tax + higher_tax = total_tax_rate q * johns_income := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_income_tax_calculation_l1131_113169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1131_113128

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_properties :
  -- The function f is defined for x > 0
  ∀ x > 0,
  -- 1. The maximum value of f(x) is 1/e
  (∃ x_max > 0, f x_max = (1 : ℝ) / Real.exp 1 ∧ ∀ y > 0, f y ≤ f x_max) ∧
  -- 2. For 0 < x < e, f(e+x) > f(e-x)
  (∀ x, 0 < x → x < Real.exp 1 → f (Real.exp 1 + x) > f (Real.exp 1 - x)) ∧
  -- 3. For any real m, if x₁ and x₂ are the x-coordinates of the intersection points of f(x) and y = m,
  --    and x₀ = (x₁ + x₂) / 2, then f'(x₀) < 0
  (∀ m : ℝ, ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
    f x₁ = m → f x₂ = m → 
    (deriv f) ((x₁ + x₂) / 2) < 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1131_113128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_ratio_theorem_l1131_113197

noncomputable section

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

-- Define the sum of the first n terms of a geometric sequence
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁
  else a₁ * (1 - q^n) / (1 - q)

-- Theorem statement
theorem geometric_ratio_theorem (a₁ : ℝ) (q : ℝ) :
  (geometric_sum a₁ q 3) / (geometric_sum a₁ q 2) = 3 / 2 →
  q = 1 ∨ q = -1/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_ratio_theorem_l1131_113197
