import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_and_no_parallel_line_l574_57458

-- Define the circle C
def circle_C (a r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = r^2}

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}

-- Define the y-axis
def y_axis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0}

theorem circle_equation_and_no_parallel_line
  (h1 : ∃ a r : ℝ, circle_C a r ∩ tangent_line ≠ ∅)
  (h2 : ∃ y1 y2 : ℝ, (0, y1) ∈ circle_C a r ∧ (0, y2) ∈ circle_C a r ∧ y2 - y1 = 2) :
  (circle_C 1 (Real.sqrt 2) = {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 2}) ∧
  (¬ ∃ k : ℝ, 
    let l := {p : ℝ × ℝ | p.2 = k * p.1 + 3}
    ∃ A B : ℝ × ℝ, A ∈ l ∩ circle_C 1 (Real.sqrt 2) ∧ B ∈ l ∩ circle_C 1 (Real.sqrt 2) ∧ A ≠ B ∧
    ∃ D : ℝ × ℝ, (D.1, D.2) = (A.1 + B.1, A.2 + B.2) ∧ D.2 / D.1 = -3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_and_no_parallel_line_l574_57458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_arguments_l574_57467

noncomputable def target_complex : ℂ := -1 / Real.sqrt 2 + Complex.I / Real.sqrt 2

def equation (z : ℂ) : Prop := z^5 = target_complex

def solution_set : Set ℂ := {z : ℂ | equation z ∧ 0 ≤ Complex.arg z ∧ Complex.arg z < 2 * Real.pi}

theorem sum_of_arguments : 
  ∃ (z₁ z₂ z₃ z₄ z₅ : ℂ), 
    z₁ ∈ solution_set ∧ 
    z₂ ∈ solution_set ∧ 
    z₃ ∈ solution_set ∧ 
    z₄ ∈ solution_set ∧ 
    z₅ ∈ solution_set ∧ 
    z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₁ ≠ z₅ ∧ 
    z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₂ ≠ z₅ ∧ 
    z₃ ≠ z₄ ∧ z₃ ≠ z₅ ∧ 
    z₄ ≠ z₅ ∧
    Complex.arg z₁ + Complex.arg z₂ + Complex.arg z₃ + Complex.arg z₄ + Complex.arg z₅ = 
      1575 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_arguments_l574_57467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l574_57413

def U : Set Nat := {1,2,3,4,5}
def M : Set Nat := {1,2,3}
def N : Set Nat := {2,5}

theorem intersection_complement_equality : M ∩ (U \ N) = {1,3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l574_57413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l574_57441

-- Define the line equation
def line_eq (x y b : ℝ) : Prop := 3 * x + 4 * y = b

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Theorem statement
theorem line_intersects_circle (b : ℝ) :
  (∃ x y : ℝ, line_eq x y b ∧ circle_eq x y) ↔ 2 < b ∧ b < 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l574_57441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_108_in_terms_of_a_b_l574_57415

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_108_in_terms_of_a_b (a b : ℝ) 
  (h1 : log10 2 = a) 
  (h2 : (10 : ℝ)^b = 3) : 
  log10 108 = 2*a + 3*b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_108_in_terms_of_a_b_l574_57415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l574_57489

noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => q * geometric_sequence a₁ q n

noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_properties :
  -- Part 1
  ∀ a₁ : ℝ, geometric_sum a₁ 2 4 = 1 → geometric_sum a₁ 2 8 = 17 ∧
  -- Part 2
  ∀ a₁ q : ℝ,
    geometric_sequence a₁ q 1 + geometric_sequence a₁ q 3 = 10 ∧
    geometric_sequence a₁ q 4 + geometric_sequence a₁ q 6 = 5/4 →
    geometric_sequence a₁ q 4 = 1 ∧ geometric_sum a₁ q 5 = 31/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l574_57489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_never_zero_l574_57495

/-- The complex number i -/
def i : ℂ := Complex.I

/-- The function P(x) -/
noncomputable def P (x : ℝ) : ℂ :=
  1 + Complex.exp (Complex.I * x) - Complex.exp (Complex.I * 2 * x) + Complex.exp (Complex.I * 3 * x)

/-- Theorem stating that P(x) is never zero for x in [0, 2π) -/
theorem P_never_zero : ∀ x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi → P x ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_never_zero_l574_57495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_l574_57448

theorem marble_probability : 
  let total_marbles : ℕ := 10
  let green_marbles : ℕ := 6
  let purple_marbles : ℕ := 4
  let num_picks : ℕ := 5
  let num_green_picks : ℕ := 2

  (Nat.choose num_picks num_green_picks : ℚ) * 
  ((green_marbles : ℚ) / (total_marbles : ℚ)) ^ num_green_picks * 
  ((purple_marbles : ℚ) / (total_marbles : ℚ)) ^ (num_picks - num_green_picks) = 
  144 / 625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_l574_57448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_cubic_solution_l574_57443

theorem complex_cubic_solution (a b c : ℂ) (h_real : a.re = a)
  (h_sum : a + b + c = 4)
  (h_prod_sum : a * b + b * c + c * a = 4)
  (h_prod : a * b * c = 4) :
  a = 2 + Complex.I * Real.sqrt 2 ∨ a = 2 - Complex.I * Real.sqrt 2 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_cubic_solution_l574_57443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_property_l574_57436

-- Define the recursive function g
noncomputable def g : ℕ → (ℝ → ℝ)
| 0 => λ x ↦ 0  -- Add a base case for 0
| 1 => λ x ↦ Real.sqrt (2 - x)
| (n + 1) => λ x ↦ g n (Real.sqrt ((n + 1)^2 + 2 - x))

-- Define the domain of g for each n
def domain (n : ℕ) : Set ℝ :=
  {x | ∃ y, g n x = y}

-- State the theorem
theorem g_domain_property :
  (∃ M : ℕ, (∀ n > M, domain n = ∅) ∧
            (domain M = {34}) ∧
            (∀ n ≤ M, domain n ≠ ∅)) ∧
  (∀ M : ℕ, (∀ n > M, domain n = ∅) ∧
            (domain M = {34}) ∧
            (∀ n ≤ M, domain n ≠ ∅) →
            M = 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_property_l574_57436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_representable_l574_57411

/-- Represents the coin denominations in Limonia -/
def coin_denominations (n : ℕ) : List ℕ :=
  (List.range (n + 1)).reverse.map (fun i => 2^i * 3^(n - i))

/-- Checks if a number is representable using the given coin denominations -/
def is_representable (s : ℕ) (n : ℕ) : Prop :=
  ∃ (coeffs : List ℕ), coeffs.length = n + 1 ∧ 
    s = List.sum (List.zipWith (·*·) coeffs (coin_denominations n))

/-- The main theorem stating the largest non-representable amount -/
theorem largest_non_representable (n : ℕ) :
  ¬ is_representable (3^(n+1) - 2^(n+2)) n ∧
  ∀ s > 3^(n+1) - 2^(n+2), is_representable s n := by
  sorry

#check largest_non_representable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_representable_l574_57411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carrots_thursday_l574_57403

def carrots_tuesday : ℕ := 4
def carrots_wednesday : ℕ := 6
def total_carrots : ℕ := 15

theorem carrots_thursday : 
  total_carrots - (carrots_tuesday + carrots_wednesday) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carrots_thursday_l574_57403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_equality_not_always_true_l574_57490

theorem triangle_sine_equality_not_always_true : 
  ¬ ∀ (a b c A B C : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) → 
    (A > 0 ∧ B > 0 ∧ C > 0) → 
    (A + B + C = π) → 
    (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) → 
    (b^2 = a^2 + c^2 - 2*a*c*Real.cos B) → 
    (c^2 = a^2 + b^2 - 2*a*b*Real.cos C) → 
    (a * Real.sin A = b * Real.sin B) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_equality_not_always_true_l574_57490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equipment_value_theorem_l574_57491

/-- The value of equipment after n years of depreciation -/
noncomputable def equipment_value (a : ℝ) (b : ℝ) (n : ℕ) : ℝ :=
  a * (1 - b / 100) ^ n

/-- 
Theorem: The value of equipment worth 'a' ten thousand yuan, 
depreciating at a rate of 'b%' annually, after 'n' years 
is equal to a(1-b%)^n.
-/
theorem equipment_value_theorem (a : ℝ) (b : ℝ) (n : ℕ) 
  (ha : a > 0) (hb : 0 < b ∧ b < 100) :
  equipment_value a b n = a * (1 - b / 100) ^ n :=
by
  -- The proof is omitted for now
  sorry

#check equipment_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equipment_value_theorem_l574_57491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mail_cars_solution_l574_57455

/-- Represents a train with mail cars -/
structure Train :=
  (total_cars : ℕ)
  (mail_cars : List ℕ)
  (h_total : total_cars = 20)
  (h_range : ∀ n ∈ mail_cars, 1 ≤ n ∧ n ≤ total_cars)
  (h_even : Even mail_cars.length)
  (h_first : mail_cars.head? = some mail_cars.length)
  (h_last : mail_cars.getLast? = some (4 * mail_cars.length))
  (h_connected : ∀ n ∈ mail_cars, ∃ m ∈ mail_cars, m ≠ n ∧ (m = n + 1 ∨ m = n - 1))

/-- The main theorem stating the properties of the mail cars -/
theorem mail_cars_solution (t : Train) : t.mail_cars = [4, 5, 15, 16] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mail_cars_solution_l574_57455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_values_l574_57488

noncomputable section

-- Define f as a function from non-negative reals to reals
variable (f : {x : ℝ // x ≥ 0} → ℝ)

-- State the functional equation
axiom func_eq : ∀ (x y : {z : ℝ // z ≥ 0}), f (⟨x.val + y.val, by sorry⟩) = f x * f y

-- State the additional condition
axiom additional_cond : f ⟨3, by norm_num⟩ = (f ⟨1, by norm_num⟩) ^ 3

-- Theorem statement
theorem f_one_values :
  ∀ (c : ℝ), c ≥ 0 → ∃ (f : {x : ℝ // x ≥ 0} → ℝ), 
    (∀ (x y : {z : ℝ // z ≥ 0}), f (⟨x.val + y.val, by sorry⟩) = f x * f y) ∧
    f ⟨3, by norm_num⟩ = (f ⟨1, by norm_num⟩) ^ 3 ∧
    f ⟨1, by norm_num⟩ = c :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_values_l574_57488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapped_area_after_folding_correct_answer_is_d_l574_57426

/-- The area of the overlapped region after folding a rectangle with specific dimensions. -/
theorem overlapped_area_after_folding (w : Real) (l : Real) (A : Real) :
  w > 0 → l = 3 * w → A = w * l → A = 27 →
  let overlap_area := (3 * Real.sqrt 11.25) / 2
  overlap_area = 5.031 := by
  sorry

/-- The main theorem proving that the correct answer is 5.031 square meters. -/
theorem correct_answer_is_d :
  let options := ["3", "4.5", "5", "5.031", "6"]
  options[3] = "5.031" ∧
  ∃ w l A, w > 0 ∧ l = 3 * w ∧ A = w * l ∧ A = 27 ∧
    (3 * Real.sqrt 11.25) / 2 = 5.031 := by
  sorry

#check overlapped_area_after_folding
#check correct_answer_is_d

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapped_area_after_folding_correct_answer_is_d_l574_57426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_munificence_monic_cubic_l574_57475

noncomputable def munificence (p : ℝ → ℝ) : ℝ :=
  ⨆ (x : ℝ) (h : x ∈ Set.Icc (-2) 2), |p x|

def is_monic_cubic (p : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, p x = x^3 + a*x^2 + b*x + c

theorem smallest_munificence_monic_cubic :
  (∀ p : ℝ → ℝ, is_monic_cubic p → munificence p ≥ 2) ∧
  (∃ p : ℝ → ℝ, is_monic_cubic p ∧ munificence p = 2) := by
  sorry

#check smallest_munificence_monic_cubic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_munificence_monic_cubic_l574_57475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_odd_l574_57445

theorem not_all_odd (a₁ a₂ a₃ a₄ a₅ b : ℤ) 
  (h : a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 = b^2) : 
  ¬(Odd a₁ ∧ Odd a₂ ∧ Odd a₃ ∧ Odd a₄ ∧ Odd a₅ ∧ Odd b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_odd_l574_57445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_above_g_min_a_for_common_tangent_l574_57473

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

noncomputable def g (x : ℝ) : ℝ := Real.log x

-- Part I
theorem f_above_g : ∀ x > 0, f 1 1 x > g x := by sorry

-- Part II
theorem min_a_for_common_tangent : 
  ∃ a₀ : ℝ, a₀ = -1 / (2 * Real.exp 3) ∧
  (∀ a ≥ a₀, ∃ x > 0, f a ((1 / x) - 2 * a * x) x = g x ∧
                       (2 * a * x + ((1 / x) - 2 * a * x) = 1 / x)) ∧
  (∀ a < a₀, ¬∃ x > 0, f a ((1 / x) - 2 * a * x) x = g x ∧
                        (2 * a * x + ((1 / x) - 2 * a * x) = 1 / x)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_above_g_min_a_for_common_tangent_l574_57473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l574_57492

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (3 - x^2) * Real.exp (-x)

-- State the theorem
theorem f_increasing :
  ∀ x : ℝ, (x < -1 ∨ x > 3) → (deriv f) x > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l574_57492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_balls_count_l574_57421

/-- Represents a box containing balls -/
structure Box where
  white : Nat
  red : Nat
  sum_is_three : white + red = 3

/-- The problem setup -/
structure BallProblem where
  total_balls : Nat
  total_boxes : Nat
  boxes : Finset Box
  total_balls_is_300 : total_balls = 300
  total_boxes_is_100 : total_boxes = 100
  boxes_size_is_100 : boxes.card = 100
  boxes_with_one_white : (boxes.filter (fun b => b.white = 1)).card = 27
  boxes_with_two_or_three_red : (boxes.filter (fun b => b.red ≥ 2)).card = 42
  boxes_with_three_white_equals_three_red : 
    (boxes.filter (fun b => b.white = 3)).card = (boxes.filter (fun b => b.red = 3)).card

theorem white_balls_count (bp : BallProblem) : 
  (bp.boxes.sum fun b => b.white) = 158 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_balls_count_l574_57421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_vertical_shift_l574_57438

/-- Given a sinusoidal function y = a * sin(b * x + c) + d where a, b, c, and d are positive constants,
    if the function oscillates between 5 and -3, then d = 1. -/
theorem sinusoidal_vertical_shift 
  (a b c d : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_pos_c : c > 0) 
  (h_pos_d : d > 0) 
  (h_max : ∀ x, a * Real.sin (b * x + c) + d ≤ 5) 
  (h_min : ∀ x, a * Real.sin (b * x + c) + d ≥ -3) 
  (h_reaches_max : ∃ x, a * Real.sin (b * x + c) + d = 5) 
  (h_reaches_min : ∃ x, a * Real.sin (b * x + c) + d = -3) : 
  d = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_vertical_shift_l574_57438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l574_57430

theorem evaluate_expression : (125 : ℝ)^(1/3 : ℝ) * (64 : ℝ)^(-(1/2) : ℝ) * (81 : ℝ)^(1/4 : ℝ) = 15/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l574_57430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_properties_l574_57410

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (3 * x + 4) / (x + 3)

-- Define the domain
def domain : Set ℝ := {x | x ≥ 1}

-- Define the range
def range : Set ℝ := {y | ∃ x ∈ domain, g x = y}

-- State the theorem
theorem g_range_properties :
  ∃ (m M : ℝ),
    (∀ x ∈ domain, g x ≥ m) ∧
    (∀ x ∈ domain, g x < M) ∧
    (m ∈ range) ∧
    (M ∉ range) ∧
    (m = 7/4) ∧
    (M = 3) := by
  -- Proof goes here
  sorry

#check g_range_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_properties_l574_57410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l574_57499

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_neg : b < 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt ((h.a^2 - h.b^2) / h.a^2)

/-- The x-coordinate of the right focus -/
noncomputable def focus_x (h : Hyperbola) : ℝ := 
  Real.sqrt (h.a^2 + h.b^2)

/-- The x-coordinate of the left directrix -/
noncomputable def directrix_x (h : Hyperbola) : ℝ := 
  -(h.a^2 / focus_x h)

theorem hyperbola_eccentricity (h : Hyperbola) :
  (h.a = (focus_x h + directrix_x h) / 2) →
  eccentricity h = Real.sqrt 2 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l574_57499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_distribution_parameters_l574_57433

variable (n : ℕ)
variable (p : ℝ)

-- ξ follows a binomial distribution B(n, p)
def is_binomial (ξ : ℕ → ℝ) (n : ℕ) (p : ℝ) : Prop :=
  ∀ k : ℕ, k ≤ n → ξ k = (n.choose k) * p^k * (1-p)^(n-k)

-- Expected value of ξ
noncomputable def expected_value (n : ℕ) (p : ℝ) : ℝ := n * p

-- Variance of ξ
noncomputable def variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_distribution_parameters (ξ : ℕ → ℝ) :
  is_binomial ξ n p →
  expected_value n p = 3 →
  variance n p = 9/4 →
  n = 12 ∧ p = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_distribution_parameters_l574_57433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_rerolls_two_probability_l574_57453

/-- Represents the outcome of rolling a six-sided die -/
inductive DieOutcome
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the initial roll of three dice -/
structure InitialRoll :=
  (first second third : DieOutcome)

/-- Represents Jason's decision to reroll -/
structure RerollDecision :=
  (rerollFirst rerollSecond rerollThird : Bool)

/-- Calculates the sum of the dice outcomes after potential rerolls -/
def finalSum (initial : InitialRoll) (decision : RerollDecision) : ℕ :=
  sorry

/-- Determines if Jason wins given the initial roll and reroll decision -/
def isWinning (initial : InitialRoll) (decision : RerollDecision) : Bool :=
  finalSum initial decision = 9

/-- Represents Jason's optimal strategy -/
def optimalStrategy (initial : InitialRoll) : RerollDecision :=
  sorry

/-- Counts the number of dice Jason decides to reroll -/
def rerollCount (decision : RerollDecision) : ℕ :=
  sorry

/-- The probability space of all possible initial rolls -/
def Ω : Type := InitialRoll

/-- The probability measure on the sample space -/
noncomputable def P : Set Ω → ℝ :=
  sorry

theorem jason_rerolls_two_probability :
  P {ω : Ω | rerollCount (optimalStrategy ω) = 2} = 1 / 72 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_rerolls_two_probability_l574_57453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_conversion_l574_57409

-- Define the conversion factor
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * (Real.pi / 180)
noncomputable def rad_to_deg (rad : ℝ) : ℝ := rad * (180 / Real.pi)

-- State the theorem
theorem angle_conversion :
  (deg_to_rad 210 = 7 * Real.pi / 6) ∧
  (rad_to_deg (-5 * Real.pi / 2) = -450) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_conversion_l574_57409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_permutations_with_one_repeat_l574_57478

theorem four_digit_permutations_with_one_repeat : ∃ n : ℕ, n = 12 := by
  let total_digits : ℕ := 4  -- total number of digits
  let repeated_digits : ℕ := 2  -- number of repetitions of one digit
  let result := (Nat.factorial total_digits) / (Nat.factorial repeated_digits)
  use result
  sorry

#eval (Nat.factorial 4) / (Nat.factorial 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_permutations_with_one_repeat_l574_57478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l574_57400

theorem problem_solution :
  ∀ a b c : ℤ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  a ≥ b ∧ b ≥ c →
  a^2 - b^2 - c^2 + a*b = 2011 →
  a^2 + 3*b^2 + 3*c^2 - 3*a*b - 2*a*c - 2*b*c = -1997 →
  a = 253 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l574_57400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mahdi_cycles_sunday_l574_57414

-- Define the days of the week
inductive Day : Type
  | monday | tuesday | wednesday | thursday | friday | saturday | sunday
  deriving Repr, DecidableEq

-- Define the sports
inductive Sport : Type
  | running | basketball | golf | swimming | tennis | cycling
  deriving Repr, DecidableEq

-- Define Mahdi's schedule
def schedule : Day → Sport := sorry

-- Define successor function for Day
def Day.succ : Day → Day
  | .monday => .tuesday
  | .tuesday => .wednesday
  | .wednesday => .thursday
  | .thursday => .friday
  | .friday => .saturday
  | .saturday => .sunday
  | .sunday => .monday

-- Conditions
axiom basketball_tuesday : schedule Day.tuesday = Sport.basketball
axiom golf_friday : schedule Day.friday = Sport.golf

axiom three_running_days : ∃ (d1 d2 d3 : Day), 
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧
  schedule d1 = Sport.running ∧ 
  schedule d2 = Sport.running ∧ 
  schedule d3 = Sport.running

axiom consecutive_running_once : ∃! (d : Day), 
  schedule d = Sport.running ∧ 
  schedule (Day.succ d) = Sport.running

axiom no_cycle_after_tennis : ∀ (d : Day), 
  schedule d = Sport.tennis → schedule (Day.succ d) ≠ Sport.cycling

axiom no_cycle_before_swimming : ∀ (d : Day), 
  schedule (Day.succ d) = Sport.swimming → schedule d ≠ Sport.cycling

-- Theorem to prove
theorem mahdi_cycles_sunday : schedule Day.sunday = Sport.cycling := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mahdi_cycles_sunday_l574_57414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_is_one_l574_57447

/-- The slope of the line passing through the left focus of the ellipse -/
theorem slope_is_one (k : ℝ) 
  (hk : k ≠ 0)
  (x₁ y₁ x₂ y₂ : ℝ)
  (hA : x₁^2 / 2 + y₁^2 = 1)
  (hB : x₂^2 / 2 + y₂^2 = 1)
  (hLineA : y₁ = k * (x₁ + 1))
  (hLineB : y₂ = k * (x₂ + 1))
  (hMidpoint : (x₁ + x₂) / 2 + 2 * (y₁ + y₂) / 2 = 0)
  : k = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_is_one_l574_57447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_horizontal_asymptote_l574_57498

/-- The function for which we want to find the horizontal asymptote -/
noncomputable def f (x : ℝ) : ℝ := (8 * x^3 + 3 * x^2 + 6 * x + 4) / (2 * x^3 + x^2 + 5 * x + 2)

/-- The value of the horizontal asymptote -/
def horizontal_asymptote : ℝ := 4

/-- Theorem stating that the horizontal asymptote of f(x) is 4 -/
theorem f_has_horizontal_asymptote :
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - horizontal_asymptote| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_horizontal_asymptote_l574_57498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_operations_exist_l574_57434

def digits : List Int := [1, 2, 3, 4, 5]

inductive Operation
| Add
| Subtract

def applyOperation (op : Operation) (a b : Int) : Int :=
  match op with
  | Operation.Add => a + b
  | Operation.Subtract => a - b

def evaluateExpression (ops : List Operation) : Int :=
  let pairs := List.zip digits.tail (List.zip ops digits.tail.tail)
  pairs.foldl
    (fun acc (d, (op, next)) => applyOperation op acc next)
    (digits.head!)

theorem arithmetic_operations_exist :
  ∃ (ops : List Operation), evaluateExpression ops = 1 := by
  -- Proof goes here
  sorry

#eval evaluateExpression [Operation.Add, Operation.Subtract, Operation.Subtract, Operation.Add]
#eval evaluateExpression [Operation.Subtract, Operation.Add, Operation.Add, Operation.Subtract]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_operations_exist_l574_57434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l574_57454

-- Define the type for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the type for a line in 2D space (ax + by + c = 0)
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

def perpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = -l1.b * l2.a

def passes_through (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

noncomputable def distance_from_origin (l : Line2D) : ℝ :=
  abs l.c / Real.sqrt (l.a^2 + l.b^2)

theorem line_equation_theorem (A B : Point2D) (given_line : Line2D) :
  A.x = 3 ∧ A.y = 0 ∧
  B.x = 5 ∧ B.y = 10 ∧
  given_line.a = 2 ∧ given_line.b = 1 ∧ given_line.c = -5 →
  ∃ (l : Line2D),
    (passes_through l A ∧ perpendicular l given_line) ∨
    (passes_through l B ∧ distance_from_origin l = 5) →
    (l.a = 1 ∧ l.b = 0 ∧ l.c = -5) ∨ (l.a = 3 ∧ l.b = -4 ∧ l.c = 25) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l574_57454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_problem_l574_57470

theorem lcm_problem (a b : ℚ) (ha : a > 0) (hb : b > 0) :
  a + b = 55 →
  Int.gcd (Int.floor a) (Int.floor b) = 5 →
  1 / a + 1 / b = 0.09166666666666666 →
  Int.lcm (Int.floor a) (Int.floor b) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_problem_l574_57470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_segment_arrangement_l574_57405

noncomputable section

def IsLineSegment (s : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ × ℝ, a ≠ b ∧ s = {x | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = (1 - t) • a + t • b}

def StrictlyInside (s1 s2 : Set (ℝ × ℝ)) : Prop :=
  ∀ x, x ∈ s1 → ∃ y z, y ∈ s2 ∧ z ∈ s2 ∧ y ≠ z ∧ ∃ t : ℝ, 0 < t ∧ t < 1 ∧ x = (1 - t) • y + t • z

theorem impossibility_of_segment_arrangement (n : ℕ) (h : n = 1000) :
  ¬ ∃ (segments : Fin n → Set (ℝ × ℝ)),
    (∀ i : Fin n, IsLineSegment (segments i)) ∧
    (∀ i : Fin n, ∃ j : Fin n, i ≠ j ∧ StrictlyInside (segments i) (segments j)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_segment_arrangement_l574_57405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l574_57477

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-1 + (Real.sqrt 3 / 2) * t, (1 / 2) * t)

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_l t ∧ circle_C p.1 p.2}

-- Theorem statement
theorem intersection_distance :
  ∃ p q : ℝ × ℝ, p ∈ intersection_points ∧ q ∈ intersection_points ∧ p ≠ q ∧
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = Real.sqrt 7 := by
  sorry

#check intersection_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l574_57477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_bounds_l574_57450

-- Define the curve C'
def C' (x y : ℝ) : Prop := x^2 + y^2/4 = 4

-- Define the expression we're interested in
noncomputable def expr (x y : ℝ) : ℝ := 2 * Real.sqrt 3 * x + y

-- Theorem statement
theorem expr_bounds :
  ∀ x₀ y₀ : ℝ, C' x₀ y₀ → -8 ≤ expr x₀ y₀ ∧ expr x₀ y₀ ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_bounds_l574_57450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_5x_over_sin_x_l574_57442

theorem sin_5x_over_sin_x (x : ℝ) (h : Real.sin (3 * x) / Real.sin x = 6 / 5) :
  Real.sin (5 * x) / Real.sin x = -0.76 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_5x_over_sin_x_l574_57442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_shared_l574_57401

/-- Represents the share of person A in rupees -/
noncomputable def share_A : ℚ := 35

/-- Represents the conversion rate from rupees to paisa -/
def rupee_to_paisa : ℚ := 100

/-- Represents B's share relative to A's share -/
noncomputable def relative_share_B : ℚ := 125 / rupee_to_paisa

/-- Represents C's share relative to A's share -/
noncomputable def relative_share_C : ℚ := 200 / rupee_to_paisa

/-- Represents D's share relative to A's share -/
noncomputable def relative_share_D : ℚ := 70 / rupee_to_paisa

/-- Represents E's share relative to A's share -/
noncomputable def relative_share_E : ℚ := 50 / rupee_to_paisa

/-- Represents C's actual share in rupees -/
def share_C : ℚ := 70

/-- Theorem stating that the total amount shared is 190.75 rupees -/
theorem total_amount_shared : 
  share_A + (relative_share_B * share_A) + (relative_share_C * share_A) + 
  (relative_share_D * share_A) + (relative_share_E * share_A) = 190.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_shared_l574_57401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_fourth_plus_alpha_equals_sin_pi_fourth_minus_alpha_l574_57417

theorem cos_pi_fourth_plus_alpha_equals_sin_pi_fourth_minus_alpha (α m : ℝ) : 
  Real.sin (π/4 - α) = m → Real.cos (π/4 + α) = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_fourth_plus_alpha_equals_sin_pi_fourth_minus_alpha_l574_57417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_functions_l574_57437

-- Define a type for graphs
inductive Graph : Type
  | F | G | H | I | J

-- Define a predicate for the Horizontal Line Test
def passes_horizontal_line_test : Graph → Prop
  | Graph.F => False
  | Graph.G => True
  | Graph.H => True
  | Graph.I => False
  | Graph.J => True

-- Define a predicate for having an inverse
def has_inverse (g : Graph) : Prop := passes_horizontal_line_test g

-- State the theorem
theorem inverse_functions :
  {g : Graph | has_inverse g} = {Graph.G, Graph.H, Graph.J} := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_functions_l574_57437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l574_57472

theorem remainder_theorem (N : ℕ) (h : N % 60 = 49) : N % 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l574_57472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_range_of_a_l574_57463

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1) - x

-- Define the domain of x
def domain (x : ℝ) : Prop := x ≥ 2

-- Theorem 1: f is decreasing on the domain
theorem f_decreasing :
  ∀ x₁ x₂ : ℝ, domain x₁ → domain x₂ → x₁ > x₂ → f x₁ < f x₂ :=
by
  sorry

-- Theorem 2: Range of a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, domain x → (a + x) * (x - 1) > 2) → a > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_range_of_a_l574_57463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neighboring_cells_difference_l574_57485

/-- A cell in the n × n board -/
structure Cell where
  row : Nat
  col : Nat

/-- The n × n board with numbers from 1 to n^2 -/
def Board (n : Nat) := Cell → Fin (n^2)

/-- Two cells are neighbors if they share a common vertex or side -/
def isNeighbor (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ Int.natAbs (c1.col - c2.col) = 1) ∨
  (c1.col = c2.col ∧ Int.natAbs (c1.row - c2.row) = 1) ∨
  (Int.natAbs (c1.row - c2.row) = 1 ∧ Int.natAbs (c1.col - c2.col) = 1)

/-- Main theorem: There exist two neighboring cells with a difference of at least n+1 -/
theorem neighboring_cells_difference (n : Nat) (board : Board n) :
  ∃ (c1 c2 : Cell), isNeighbor c1 c2 ∧ 
    Int.natAbs ((board c1).val - (board c2).val) ≥ n + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_neighboring_cells_difference_l574_57485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l574_57483

theorem empty_subset_singleton_zero : ∅ ⊆ ({0} : Set ℕ) := by
  apply Set.empty_subset


end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l574_57483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_domain_all_reals_l574_57446

/-- A function b(x) parameterized by k -/
noncomputable def b (k : ℝ) (x : ℝ) : ℝ := (k * x^2 + 3 * x - 4) / (3 * x^2 - 4 * x + k)

/-- The domain of b(x) is all real numbers iff k > 4/3 -/
theorem b_domain_all_reals (k : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, b k x = y) ↔ k > 4/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_domain_all_reals_l574_57446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_ratio_l574_57471

theorem square_perimeter_ratio (s : ℝ) (h : s > 0) : 
  (4 * (3 * s)) / (4 * s) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_ratio_l574_57471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_a_l574_57466

def vector_a : ℝ × ℝ := (2, 1)

theorem unit_vector_a : 
  (vector_a.1 / Real.sqrt (vector_a.1^2 + vector_a.2^2), 
   vector_a.2 / Real.sqrt (vector_a.1^2 + vector_a.2^2)) = 
  (2 * Real.sqrt 5 / 5, Real.sqrt 5 / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_a_l574_57466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_max_sum_distances_l574_57423

-- Define the curves
noncomputable def C₁ (t α : ℝ) : ℝ × ℝ := (t * Real.cos α, t * Real.sin α)
noncomputable def C₂ (β : ℝ) : ℝ × ℝ := (Real.cos β, 1 + Real.sin β)
noncomputable def C₃ (θ : ℝ) : ℝ := 1 + Real.cos θ
def C₄ (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 1

-- Define the conditions
def valid_t : ℝ → Prop := λ t ↦ t > 0
def valid_α : ℝ → Prop := λ α ↦ 0 < α ∧ α < Real.pi / 2
def valid_β : ℝ → Prop := λ β ↦ -Real.pi / 2 < β ∧ β < Real.pi / 2
def valid_θ : ℝ → Prop := λ θ ↦ 0 < θ ∧ θ < Real.pi / 2

-- Theorem statements
theorem intersection_distance :
  ∃ ρ θ, valid_θ θ ∧ C₄ ρ θ ∧ ρ = C₃ θ ∧ ρ = (1 + Real.sqrt 5) / 2 := by sorry

theorem max_sum_distances :
  ∃ max_sum : ℝ, 
    (∀ α, valid_α α → 
      ∃ t β θ, valid_t t ∧ valid_β β ∧ valid_θ θ ∧
      C₁ t α = C₂ β ∧ 
      (C₁ t α).1^2 + (C₁ t α).2^2 + 
      ((C₁ t α).1 - (C₁ t θ).1)^2 + ((C₁ t α).2 - (C₁ t θ).2)^2 ≤ max_sum^2) ∧
    max_sum = 1 + Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_max_sum_distances_l574_57423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_coins_correct_l574_57432

/-- Proves that the total number of coins is 324 given the conditions of the problem -/
def total_coins_count : ℕ :=
  let total_sum : ℕ := 7000  -- Total sum in paise
  let coin_20_count : ℕ := 220  -- Number of 20 paise coins
  let coin_20_value : ℕ := 20  -- Value of 20 paise coin
  let coin_25_value : ℕ := 25  -- Value of 25 paise coin
  let coin_20_sum : ℕ := coin_20_count * coin_20_value
  let coin_25_sum : ℕ := total_sum - coin_20_sum
  let coin_25_count : ℕ := coin_25_sum / coin_25_value
  coin_20_count + coin_25_count

#eval total_coins_count  -- This will evaluate to 324

/-- Proves that the total sum of all coins is 7000 paise (Rs. 70) -/
def total_sum_correct : ℕ :=
  let total_sum : ℕ := 7000
  let coin_20_count : ℕ := 220
  let coin_20_value : ℕ := 20
  let coin_25_value : ℕ := 25
  let coin_25_count : ℕ := total_coins_count - coin_20_count
  coin_20_count * coin_20_value + coin_25_count * coin_25_value

#eval total_sum_correct  -- This will evaluate to 7000

/-- Proves that the calculated total number of coins matches the given conditions -/
theorem total_coins_correct : total_coins_count = 324 ∧ total_sum_correct = 7000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_coins_correct_l574_57432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l574_57465

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 5*Real.sqrt 2 - 2 = 0

-- State the theorem
theorem min_distance_circle_line :
  ∃ (d : ℝ), d = 3 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ →
    line_l x₂ y₂ →
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≥ d :=
by
  -- Proof goes here
  sorry

-- Helper lemma for the distance between a point and a line
lemma distance_point_line (x y : ℝ) : 
  let d := |2 - 5*Real.sqrt 2 - 2| / Real.sqrt 2
  d = 5 :=
by
  -- Proof goes here
  sorry

-- Helper lemma for the radius of the circle
lemma circle_radius : 
  ∀ (x y : ℝ), circle_C x y → Real.sqrt ((x - 2)^2 + y^2) = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l574_57465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_role_assignment_count_l574_57479

def num_male_roles : ℕ := 3
def num_female_roles : ℕ := 2
def num_either_gender_roles : ℕ := 1
def num_men : ℕ := 6
def num_women : ℕ := 5

def assign_roles : ℕ :=
  (Nat.descFactorial num_men num_male_roles) *
  (Nat.descFactorial num_women num_female_roles) *
  (num_men + num_women - num_male_roles - num_female_roles)

theorem role_assignment_count : assign_roles = 14400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_role_assignment_count_l574_57479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_exp_inequality_l574_57439

theorem log_exp_inequality (a : ℝ) (h : a > 1) :
  Real.log a / Real.log 0.2 < (0.2 : ℝ)^a ∧ (0.2 : ℝ)^a < a^(0.2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_exp_inequality_l574_57439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l574_57476

-- Define the circle
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 3*y + a^2 + a = 0

-- Define the point A
def point_A (a : ℝ) : ℝ × ℝ := (a, 3)

-- Define the condition that A is outside the circle
def A_outside_circle (a : ℝ) : Prop :=
  let (x, y) := point_A a
  x^2 + y^2 - 2*a*x - 3*y + a^2 + a > 0

-- State the theorem
theorem a_range : 
  ∀ a : ℝ, A_outside_circle a → 0 < a ∧ a < 9/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l574_57476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l574_57496

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, 2 * x ^ 2 + 1 > 0)) ↔ (∃ x₀ : ℝ, 2 * x₀ ^ 2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l574_57496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l574_57474

/-- An inverse proportion function -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := (2 - k) / x

/-- Condition for the function to be in the second and fourth quadrants -/
def in_second_and_fourth_quadrants (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → (x > 0 → f x < 0) ∧ (x < 0 → f x > 0)

/-- Theorem stating the condition on k for the inverse proportion function 
    to be in the second and fourth quadrants -/
theorem inverse_proportion_quadrants (k : ℝ) : 
  in_second_and_fourth_quadrants (inverse_proportion k) ↔ k > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l574_57474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_1998_no_solution_for_2001_l574_57451

/-- Represents the sequence where each term is either 1 or 2 -/
def sequenceterm : ℕ → ℕ
  | 0 => 1  -- First term is 1
  | n + 1 => sorry  -- Definition based on the problem description

/-- The sum of the first n terms of the sequence -/
def sequence_sum (n : ℕ) : ℕ :=
  (List.range n).map sequenceterm |>.sum

theorem sequence_sum_1998 : sequence_sum 1998 = 3985 := by
  sorry

theorem no_solution_for_2001 : ¬ ∃ (n : ℕ), sequence_sum n = 2001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_1998_no_solution_for_2001_l574_57451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_tangent_circle_l574_57429

-- Define the circle Ω
variable (Ω : Set (ℝ × ℝ))

-- Define points B and C on Ω
variable (B C : ℝ × ℝ)
variable (hB : B ∈ Ω)
variable (hC : C ∈ Ω)

-- Define a variable point A on Ω
variable (A : ℝ × ℝ)
variable (hA : A ∈ Ω)

-- Define X as the foot of the altitude from B in triangle ABC
noncomputable def X (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define Y as the foot of the altitude from C in triangle ABC
noncomputable def Y (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define a line segment
def LineSegment (P Q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define tangency
def TangentTo (s : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) (P : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem fixed_tangent_circle (Ω : Set (ℝ × ℝ)) (B C : ℝ × ℝ) (hB : B ∈ Ω) (hC : C ∈ Ω) :
  ∃ (Γ : Set (ℝ × ℝ)), ∀ (A : ℝ × ℝ), A ∈ Ω →
    ∃ (P : ℝ × ℝ), P ∈ Γ ∧ TangentTo (LineSegment (X A B C) (Y A B C)) Γ P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_tangent_circle_l574_57429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_point_distances_l574_57422

/-- Given a rectangle ABCD and a point P, prove the lengths of AP, DP, PQ, and CQ -/
theorem rectangle_point_distances (A B C D P Q : ℝ × ℝ) 
  (AB : ℝ) (BC : ℝ) (PC : ℝ)
  (h_AB : AB = 1200)
  (h_BC : BC = 150)
  (h_PC : PC = 350)
  (h_rect : (B.1 - A.1 = AB ∧ B.2 - A.2 = 0) ∧ 
            (C.1 - B.1 = 0 ∧ C.2 - B.2 = BC))
  (h_P : P.1 - C.1 = 0 ∧ P.2 - C.2 = PC)
  (h_Q : Q.2 = A.2) :
  (Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) = 1300) ∧
  (Real.sqrt ((D.1 - P.1)^2 + (D.2 - P.2)^2) = 1250) ∧
  (Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 910) ∧
  (Real.sqrt ((C.1 - Q.1)^2 + (C.2 - Q.2)^2) = 840) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_point_distances_l574_57422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_f_eight_l574_57461

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the condition that g has an inverse
axiom g_has_inverse : Function.Bijective g

-- Define the given condition
axiom inverse_condition : ∀ x, f⁻¹ (g x) = x^2 + 2*x - 3

-- State the theorem
theorem inverse_g_f_eight :
  g⁻¹ (f 8) = -1 + 2 * Real.sqrt 3 ∨ g⁻¹ (f 8) = -1 - 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_f_eight_l574_57461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_ten_l574_57431

/-- The equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def is_circle (h k r : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The given equation -/
def equation (k : ℝ) (x y : ℝ) : ℝ :=
  x^2 + 14*x + y^2 + 8*y - k

theorem circle_radius_ten (k : ℝ) :
  (∃ h l, is_circle h l 10 (equation k)) ↔ k = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_ten_l574_57431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_largest_shaded_area_l574_57412

-- Define the shaded areas for each figure
noncomputable def shaded_area_X : ℝ := 8 - Real.pi
noncomputable def shaded_area_Y : ℝ := 8 - Real.pi / 2
def shaded_area_Z : ℝ := 12

-- Theorem stating that Z has the largest shaded area
theorem z_largest_shaded_area :
  shaded_area_Z > shaded_area_X ∧ shaded_area_Z > shaded_area_Y := by
  apply And.intro
  · -- Prove shaded_area_Z > shaded_area_X
    sorry
  · -- Prove shaded_area_Z > shaded_area_Y
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_largest_shaded_area_l574_57412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_data_l574_57406

noncomputable def data : List ℝ := [10, 6, 8, 5, 6]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (λ x => (x - mean xs) ^ 2)).sum / xs.length

noncomputable def standardDeviation (xs : List ℝ) : ℝ :=
  Real.sqrt (variance xs)

theorem standard_deviation_of_data :
  standardDeviation data = (4 * Real.sqrt 5) / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_data_l574_57406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonagon_area_approx_l574_57486

/-- The area of a regular nonagon inscribed in a circle with radius r -/
noncomputable def nonagon_area (r : ℝ) : ℝ := (9 / 2) * r^2 * Real.sin (40 * Real.pi / 180)

/-- Theorem stating that the area of a regular nonagon inscribed in a circle 
    with radius r is approximately equal to 2.891 * r^2 -/
theorem nonagon_area_approx (r : ℝ) (h : r > 0) : 
  ∃ ε > 0, |nonagon_area r - 2.891 * r^2| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonagon_area_approx_l574_57486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_specific_coin_l574_57456

/-- A type representing a coin with a unique weight -/
structure Coin where
  weight : ℝ
  unique : Prop

/-- A function representing a weighing on a balance scale -/
noncomputable def compare (a b : Coin) : Bool :=
  if a.weight > b.weight then true else false

/-- The theorem stating that we can find the 101st heaviest coin in at most 8 weighings -/
theorem find_specific_coin (coins : Finset Coin) (h1 : coins.card = 201) 
  (h2 : ∀ (c1 c2 : Coin), c1 ∈ coins → c2 ∈ coins → c1 ≠ c2 → c1.weight ≠ c2.weight) :
  ∃ (n : ℕ) (f : Coin → Coin → Bool), 
    n ≤ 8 ∧ 
    ∃ (c : Coin), c ∈ coins ∧ 
      (coins.filter (λ x => x.weight > c.weight)).card = 100 ∧
      (∀ (x y : Coin), x ∈ coins → y ∈ coins → f x y = compare x y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_specific_coin_l574_57456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_boundary_correct_l574_57459

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The sphere with given properties -/
noncomputable def sphere : Sphere := {
  center := { x := 1, y := 0, z := 1 },
  radius := 1
}

/-- The light source position -/
noncomputable def lightSource : Point3D := { x := 1, y := -1, z := 2 }

/-- The function describing the shadow boundary -/
noncomputable def shadowBoundary (x : ℝ) : ℝ := (x - 1)^2 / 4 - 1

/-- Theorem stating that the given function describes the shadow boundary -/
theorem shadow_boundary_correct :
  ∀ x y : ℝ,
  y = shadowBoundary x ↔
  (∃ (t : Point3D),
    (t.x - sphere.center.x)^2 + (t.y - sphere.center.y)^2 + (t.z - sphere.center.z)^2 = sphere.radius^2 ∧
    (x - lightSource.x) / (t.x - lightSource.x) = (y - lightSource.y) / (t.y - lightSource.y) ∧
    (x - lightSource.x) / (t.x - lightSource.x) = (-lightSource.z) / (t.z - lightSource.z) ∧
    t.z > 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_boundary_correct_l574_57459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_range_of_a_l574_57494

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * cos x * sin (x + π/6) - 1

-- Theorem for the interval of monotonic increase
theorem monotonic_increase_interval (k : ℤ) :
  StrictMonoOn f (Set.Icc (-π/3 + k * π) (π/6 + k * π)) :=
sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ioo (-π/4) (π/4), sin x ^ 2 + a * f (x + π/6) + 1 > 6 * cos x ^ 4) →
  a > 5/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_range_of_a_l574_57494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l574_57469

open Real

theorem problem_statement :
  (¬ ∃ x : ℝ, (2 : ℝ)^x + (2 : ℝ)^(-x) = 1) ∧
  (∀ x : ℝ, log (x^2 + 2*x + 3) > 0) ∧
  (¬ ∃ x : ℝ, (2 : ℝ)^x + (2 : ℝ)^(-x) = 1) ∧ (∀ x : ℝ, log (x^2 + 2*x + 3) > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l574_57469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l574_57449

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 + 3 * Real.log x

def g (b : ℝ) (x : ℝ) : ℝ := -b * x

noncomputable def h (a b : ℝ) (x : ℝ) : ℝ := f a x - g b x

theorem problem_solution (a b : ℝ) :
  (deriv (f a)) (Real.sqrt 2 / 2) = 0 ∧
  (deriv (f a)) 1 = g b (-1) - 2 →
  a = -6 ∧ b = -1 ∧
  (deriv (h a b)) 1 = -4 ∧
  h a b 1 = -4 := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l574_57449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l574_57487

noncomputable section

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = a^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = b^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = c^2

-- Define the angle between two vectors
noncomputable def Angle (v1 v2 : ℝ × ℝ) : ℝ :=
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2)))

-- Convert degrees to radians
noncomputable def degToRad (deg : ℝ) : ℝ := deg * Real.pi / 180

-- Define the perimeter of a triangle
noncomputable def Perimeter (A B C : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) +
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) +
  Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)

theorem triangle_perimeter (A B C : ℝ × ℝ) :
  Triangle A B C →
  Angle (B.1 - A.1, B.2 - A.2) (C.1 - A.1, C.2 - A.2) = degToRad 75 →
  Angle (A.1 - B.1, A.2 - B.2) (C.1 - B.1, C.2 - B.2) = degToRad 45 →
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = Real.sqrt 3 + 1 →
  Perimeter A B C = (3 * Real.sqrt 2 + 5 * Real.sqrt 3 - Real.sqrt 6 + 2) / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l574_57487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_traveled_l574_57440

-- Define the velocity function
def v (t : ℝ) : ℝ := 2 * t - 3

-- Define the distance function as the integral of velocity
noncomputable def distance (a b : ℝ) : ℝ := -(∫ (x : ℝ) in a..b, v x)

-- Theorem statement
theorem distance_traveled : distance 0 (3/2) = 9/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_traveled_l574_57440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l574_57427

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (Real.log x / Real.log 2 - 1)}
def N : Set ℝ := {x | |x - 1| ≤ 2}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Icc 2 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l574_57427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_l574_57435

theorem journey_distance (initial_speed reduced_speed total_time x : ℝ) 
  (h1 : initial_speed = 4)
  (h2 : reduced_speed = 2)
  (h3 : total_time = 3.5) : ℝ :=
by
  -- Define the total distance as 4x
  let total_distance := 4 * x

  -- Define the distance before detour
  let distance_before_detour := 3 * x

  -- Define the distance after detour
  let distance_after_detour := x

  -- Time taken for the first part of the journey
  let time_before_detour := distance_before_detour / initial_speed

  -- Time taken for the detour part of the journey
  let time_after_detour := distance_after_detour / reduced_speed

  -- Total time equation
  have time_eq : total_time = time_before_detour + time_after_detour := by sorry

  -- The total distance is 11.2 km
  have distance_eq : total_distance = 11.2 := by sorry

  -- Return the total distance
  exact total_distance


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_l574_57435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_property_l574_57480

/-- Given a function f(x) = m*sin(πx+α) + n*cos(πx+β) + 8,
    where m, n, α, β are real numbers,
    if f(2000) = -2000, then f(2015) = 2016 -/
theorem periodic_function_property (m n α β : ℝ) :
  let f : ℝ → ℝ := λ x ↦ m * Real.sin (π * x + α) + n * Real.cos (π * x + β) + 8
  f 2000 = -2000 → f 2015 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_property_l574_57480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l574_57462

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi / 3 - x / 2)

def is_period (T : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def smallest_positive_period (T : ℝ) (f : ℝ → ℝ) : Prop :=
  is_period T f ∧ T > 0 ∧ ∀ T' > 0, is_period T' f → T ≤ T'

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem f_properties :
  (smallest_positive_period (4 * Real.pi) f) ∧
  (∀ k : ℤ, monotone_increasing_on f (Set.Icc (5 * Real.pi / 3 + 4 * k * Real.pi) (11 * Real.pi / 3 + 4 * k * Real.pi))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l574_57462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_l574_57428

/-- A cube with sum of all edge lengths equal to 96 cm has edge length 8 cm -/
theorem cube_edge_length (cube : Set (Fin 3 → ℝ)) (sum_of_edges : ℝ) : 
  (∀ p, p ∈ cube ↔ (∀ i, 0 ≤ p i ∧ p i ≤ 1)) →
  sum_of_edges = 96 →
  ∃ edge_length : ℝ, edge_length * 12 = sum_of_edges ∧ edge_length = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_l574_57428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_different_digits_l574_57497

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 3 ∧ digits.toFinset.card = 3

-- Add this instance to make has_different_digits decidable
instance (n : ℕ) : Decidable (has_different_digits n) :=
  show Decidable ((n.digits 10).length = 3 ∧ (n.digits 10).toFinset.card = 3) from inferInstance

theorem probability_different_digits :
  (Finset.filter has_different_digits (Finset.range 900)).card / 900 = 99 / 100 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_different_digits_l574_57497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_green_mandms_l574_57468

/-- Represents the number of M&Ms of each color in the jar -/
structure MandMs where
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- The initial state of the jar -/
def initial_jar (G : ℕ) : MandMs :=
  { green := G,  -- G is the unknown initial number of green M&Ms
    red := 20,
    yellow := 0 }

/-- The state of the jar after Carter and his sister's actions -/
def final_jar (G : ℕ) : MandMs :=
  { green := G - 12,
    red := 10,
    yellow := 14 }

/-- The total number of M&Ms in the final jar -/
def total_mandms (G : ℕ) : ℕ :=
  (final_jar G).green + (final_jar G).red + (final_jar G).yellow

/-- The theorem stating the initial number of green M&Ms -/
theorem initial_green_mandms :
  ∃ G : ℕ, G ≥ 12 ∧ (final_jar G).green = (25 : ℚ) / 100 * (total_mandms G : ℚ) ∧ G = 20 := by
  sorry

#eval total_mandms 20  -- This will evaluate the total M&Ms for G = 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_green_mandms_l574_57468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_meeting_problem_l574_57482

def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p^2 ∣ n) → p = 1

def arrival_time := Set.Icc (0 : ℝ) 60

theorem library_meeting_problem (d e f : ℕ) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (hsf : is_square_free f) :
  let n : ℝ := d - e * Real.sqrt (f : ℝ)
  let prob_meet := 1 - 2 * (60 - n)^2 / 3600
  prob_meet = 0.3 → d + e + f = 112 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_meeting_problem_l574_57482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l574_57493

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a = (B.sin / A.sin) * c ∧
  b = (C.sin / A.sin) * c ∧
  A = π / 3 ∧
  b = 4 ∧
  (1/2) * b * c * A.sin = 2 * Real.sqrt 3 →
  a = 2 * Real.sqrt 3 := by
  sorry

#check triangle_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l574_57493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l574_57408

/-- The parabola y² = -8x -/
def parabola (x y : ℝ) : Prop := y^2 = -8*x

/-- The line x = 2 -/
def line1 (x : ℝ) : Prop := x = 2

/-- The line 3x + 5y - 30 = 0 -/
def line2 (x y : ℝ) : Prop := 3*x + 5*y - 30 = 0

/-- Distance from a point (x, y) to the line x = 2 -/
def dist_to_line1 (x y : ℝ) : ℝ := |x - 2|

/-- Distance from a point (x, y) to the line 3x + 5y - 30 = 0 -/
noncomputable def dist_to_line2 (x y : ℝ) : ℝ := |3*x + 5*y - 30| / Real.sqrt 34

/-- The theorem stating the minimum value of the sum of distances -/
theorem min_sum_distances : 
  ∃ (min : ℝ), min = 18/17 * Real.sqrt 34 ∧ 
  ∀ (x y : ℝ), parabola x y → 
  dist_to_line1 x y + dist_to_line2 x y ≥ min :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l574_57408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l574_57481

noncomputable def binomial_expansion (x : ℝ) : ℝ := (x - 2 / Real.sqrt x) ^ 6

theorem constant_term_of_binomial_expansion : 
  ∃ (f : ℝ → ℝ), (∀ x, x ≠ 0 → f x = binomial_expansion x) ∧ 
  (∃ c, c = 240 ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x - c| < ε)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l574_57481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_five_minus_g_eight_l574_57419

/-- A linear function g satisfying g(x+1) - g(x) = 4 for all real numbers x -/
noncomputable def g : ℝ → ℝ := sorry

/-- g is a linear function -/
axiom g_linear : IsLinearMap ℝ g

/-- g satisfies g(x+1) - g(x) = 4 for all real numbers x -/
axiom g_property (x : ℝ) : g (x + 1) - g x = 4

/-- Theorem: g(5) - g(8) = -12 -/
theorem g_five_minus_g_eight : g 5 - g 8 = -12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_five_minus_g_eight_l574_57419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_victor_needs_three_more_candies_l574_57484

-- Define the number of friends and initial candy count
def num_friends : ℚ := 7.5
def initial_candy : ℕ := 4692

-- Define the function to calculate the additional candy needed
def additional_candy_needed (friends : ℚ) (candy : ℕ) : ℕ :=
  let candies_per_friend := (candy : ℚ) / friends
  let rounded_candies_per_friend := ⌈candies_per_friend⌉
  let total_needed := (rounded_candies_per_friend * friends).ceil
  (total_needed - candy).toNat

-- Theorem statement
theorem victor_needs_three_more_candies :
  additional_candy_needed num_friends initial_candy = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_victor_needs_three_more_candies_l574_57484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_clever_integers_divisible_by_18_l574_57452

def is_clever_integer (n : ℕ) : Prop :=
  n % 2 = 0 ∧ n > 10 ∧ n < 130 ∧ (n.repr.toList.map (λ c => c.toString.toNat!)).sum = 12

theorem all_clever_integers_divisible_by_18 :
  ∀ n : ℕ, is_clever_integer n → n % 18 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_clever_integers_divisible_by_18_l574_57452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_decomposition_product_l574_57418

noncomputable def f (x : ℝ) : ℝ := (x^2 - 16) / (x^3 - 3*x^2 - 4*x + 12)

noncomputable def g (x A B C : ℝ) : ℝ := A / (x - 2) + B / (x + 2) + C / (x - 3)

theorem fractional_decomposition_product (A B C : ℝ) :
  (∀ x, x ≠ 2 ∧ x ≠ -2 ∧ x ≠ 3 → f x = g x A B C) →
  A * B * C = -63/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_decomposition_product_l574_57418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_catch_percentage_increase_l574_57460

/-- Represents the number of fish caught in each round and the total caught -/
structure FishCatch where
  first_round : ℕ
  second_round : ℕ
  last_round : ℕ
  total : ℕ

/-- Calculates the percentage increase between two numbers -/
def percentageIncrease (original : ℕ) (new : ℕ) : ℚ :=
  (new - original : ℚ) / original * 100

/-- Theorem stating the percentage increase in fish caught in the last round compared to the second round -/
theorem fish_catch_percentage_increase 
  (c : FishCatch) 
  (h1 : c.first_round = 8)
  (h2 : c.second_round = c.first_round + 12)
  (h3 : c.total = 60)
  (h4 : c.total = c.first_round + c.second_round + c.last_round) :
  percentageIncrease c.second_round c.last_round = 60 := by
  sorry

#eval percentageIncrease 20 32

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_catch_percentage_increase_l574_57460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_painters_time_l574_57457

/-- The number of work-days required for a given number of painters to complete a job -/
noncomputable def work_days (painters : ℕ) (job_size : ℝ) (rate : ℝ) : ℝ :=
  job_size / (painters * rate)

theorem four_painters_time (job_size : ℝ) (rate : ℝ) :
  work_days 5 job_size rate = 2 →
  work_days 4 job_size rate = 2.5 := by
  sorry

#check four_painters_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_painters_time_l574_57457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_proportion_problem_l574_57407

theorem ratio_proportion_problem (x : ℝ) : 
  (215 : ℝ) / x = 537 / 26 → Int.floor x = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_proportion_problem_l574_57407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_password_possibilities_l574_57464

-- Define the type for a password (a list of 5 digits)
def Password := List Nat

-- Define the guesses
def guess1 : Password := [5, 1, 9, 3, 2]
def guess2 : Password := [8, 5, 4, 7, 8]
def guess3 : Password := [3, 4, 9, 0, 6]

-- Define a function to check if a password is valid
def is_valid_password (p : Password) : Prop :=
  p.length = 5 ∧ 
  p.toFinset.card = 5 ∧
  p.all (· < 10)

-- Define a function to count correct guesses
def correct_guesses (guess actual : Password) : Nat :=
  (List.zip guess actual).filter (λ (g, a) => g = a) |>.length

-- Define a function to check if correct guesses are non-adjacent
def non_adjacent_correct (guess actual : Password) : Prop :=
  ∀ i j, i < j → 
    guess.get? i = actual.get? i → 
    guess.get? j = actual.get? j → 
    j - i > 1

-- Main theorem
theorem password_possibilities : 
  ∃ p : Password, 
    is_valid_password p ∧
    correct_guesses guess1 p = 2 ∧
    correct_guesses guess2 p = 2 ∧
    correct_guesses guess3 p = 2 ∧
    non_adjacent_correct guess1 p ∧
    non_adjacent_correct guess2 p ∧
    non_adjacent_correct guess3 p ∧
    (p = [5, 5, 9, 7, 6] ∨ p = [7, 5, 9, 7, 2]) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_password_possibilities_l574_57464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_and_parallel_lines_l574_57416

-- Define the circles and points
def circle_M (r : ℝ) : ℝ → ℝ → Prop := λ x y ↦ (x + 2)^2 + (y + 2)^2 = r^2
def circle_C : ℝ → ℝ → Prop := λ x y ↦ x^2 + y^2 = 2
def point_P : ℝ × ℝ := (1, 1)
def point_O : ℝ × ℝ := (0, 0)

-- Define the symmetry line
def symmetry_line : ℝ → ℝ → Prop := λ x y ↦ x + y + 2 = 0

-- Define the theorem
theorem circle_C_and_parallel_lines 
  (r : ℝ) 
  (h_r_pos : r > 0)
  (h_C_through_P : circle_C point_P.1 point_P.2)
  (h_symmetry : ∀ x y, circle_C x y ↔ ∃ x' y', circle_M r x' y' ∧ symmetry_line ((x + x') / 2) ((y + y') / 2))
  (A B : ℝ × ℝ)
  (h_A_on_C : circle_C A.1 A.2)
  (h_B_on_C : circle_C B.1 B.2)
  (h_A_neq_B : A ≠ B)
  (h_PA_PB_complementary : ∃ k : ℝ, k ≠ 0 ∧ 
    (A.2 - point_P.2) / (A.1 - point_P.1) = k ∧
    (B.2 - point_P.2) / (B.1 - point_P.1) = -1/k) :
  (∀ x y, circle_C x y ↔ x^2 + y^2 = 2) ∧
  (A.2 - B.2) / (A.1 - B.1) = (point_P.2 - point_O.2) / (point_P.1 - point_O.1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_and_parallel_lines_l574_57416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_k_with_n_prime_divisors_l574_57425

/-- Definition of a_k -/
def a (k : ℕ) : ℕ := 2024 * 10^(k + 1) + 1

/-- Theorem statement -/
theorem exists_k_with_n_prime_divisors (N : ℕ) :
  ∃ k : ℕ, ∃ S : Finset ℕ,
    (∀ p ∈ S, Nat.Prime p) ∧
    (S.card ≥ N) ∧
    (∀ p ∈ S, p ∣ a k) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_k_with_n_prime_divisors_l574_57425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_log_functions_increasing_log_half_decreasing_l574_57402

noncomputable def logarithmic_function (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem not_all_log_functions_increasing :
  ¬ (∀ (a : ℝ), a > 0 ∧ a ≠ 1 → 
    ∀ (x₁ x₂ : ℝ), x₁ < x₂ → logarithmic_function a x₁ < logarithmic_function a x₂) :=
by
  sorry

theorem log_half_decreasing :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → logarithmic_function (1/2) x₁ > logarithmic_function (1/2) x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_log_functions_increasing_log_half_decreasing_l574_57402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l574_57420

theorem sin_beta_value (α β : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.pi / 2 < β) (h4 : β < Real.pi)
  (h5 : Real.sin α = 3 / 5) (h6 : Real.cos (α + β) = - 4 / 5) : 
  Real.sin β = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l574_57420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_cup_flip_l574_57444

def flip_cups (m n : ℕ) (k : ℕ) : Fin m → Int :=
  sorry

theorem tea_cup_flip (m n : ℕ) (h_m_odd : Odd m) (h_m_ge_3 : m ≥ 3)
  (h_n_even : Even n) (h_n_ge_2 : n ≥ 2) (h_n_lt_m : n < m) :
  ∀ k : ℕ, ∃ i : Fin m, (flip_cups m n k) i = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_cup_flip_l574_57444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_times_54_equals_70000_l574_57424

theorem number_times_54_equals_70000 : ∃ x : ℚ, x * 54 = 70000 ∧ Int.floor x = 1296 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_times_54_equals_70000_l574_57424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l574_57404

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x + 5 else 2 * a / x

theorem f_decreasing_implies_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) → 0 < a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l574_57404
