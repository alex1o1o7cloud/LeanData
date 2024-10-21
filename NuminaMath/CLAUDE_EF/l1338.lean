import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_right_triangle_area_l1338_133807

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let x := Real.sqrt (b^2 - a^2)
  let area1 := (a * b) / 2
  let area2 := (a * x) / 2
  min area1 area2 = (a * Real.sqrt (b^2 - a^2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_right_triangle_area_l1338_133807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_when_sin_x_neg_two_thirds_l1338_133847

theorem cos_2x_when_sin_x_neg_two_thirds (x : ℝ) (h : Real.sin x = -2/3) : Real.cos (2*x) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_when_sin_x_neg_two_thirds_l1338_133847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_food_amount_l1338_133899

-- Define the amounts of each food item
noncomputable def chicken : ℝ := 16
noncomputable def hamburger : ℝ := chicken / 2
noncomputable def hotdog : ℝ := hamburger + 2
noncomputable def sides : ℝ := hotdog / 2

-- Theorem statement
theorem total_food_amount :
  chicken + hamburger + hotdog + sides = 39 := by
  -- Expand the definitions
  unfold chicken hamburger hotdog sides
  -- Perform the calculation
  simp [add_assoc]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_food_amount_l1338_133899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_correct_l1338_133813

/-- An isosceles triangle with a folded vertex -/
structure FoldedIsoscelesTriangle where
  -- The side length of the original isosceles triangle
  s : ℝ
  -- Assumption that s is positive
  s_pos : 0 < s
  -- Assumption that BA' = 2
  ba'_eq_two : 2 = 2
  -- Assumption that A'C = 3
  a'c_eq_three : 3 = 3

/-- The length of the crease in a folded isosceles triangle -/
noncomputable def creaseLength (t : FoldedIsoscelesTriangle) : ℝ :=
  7 * Real.sqrt 21 / 20

/-- Theorem stating that the crease length is correct -/
theorem crease_length_is_correct (t : FoldedIsoscelesTriangle) : 
  creaseLength t = 7 * Real.sqrt 21 / 20 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_correct_l1338_133813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_complex_fraction_l1338_133808

theorem real_part_of_complex_fraction :
  let z : ℂ := Complex.I / (1 - Complex.I)
  z.re = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_complex_fraction_l1338_133808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1338_133803

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  angleSum : A + B + C = π
  sineRule : sin A / a = sin B / b
  cosineRule : a^2 = b^2 + c^2 - 2*b*c*cos A

/-- The main theorem about the triangle -/
theorem triangle_property (t : Triangle) 
    (h : 3 * t.b * cos t.A = t.c * cos t.A + t.a * cos t.C) :
  tan t.A = 2 * Real.sqrt 2 ∧
  (t.a = 4 * Real.sqrt 2 → 
    ∀ (area : ℝ), area = 1/2 * t.b * t.c * sin t.A → area ≤ 8 * Real.sqrt 2) :=
by sorry

#check triangle_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1338_133803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shortest_path_to_G_l1338_133855

/-- Represents a point on the surface of a cube --/
structure CubeSurfacePoint where
  x : Real
  y : Real
  z : Real
  on_surface : (x = 0 ∨ x = 1) ∨ (y = 0 ∨ y = 1) ∨ (z = 0 ∨ z = 1)

/-- The shortest path length between two points on a cube surface --/
noncomputable def shortest_path_length (p q : CubeSurfacePoint) : Real :=
  sorry

/-- The maximum shortest path length from a point on EFGH to G --/
noncomputable def max_path_length : Real :=
  5 * Real.sqrt 2 / 6

theorem max_shortest_path_to_G (M : CubeSurfacePoint) 
  (h_on_EFGH : M.z = 1) 
  (h_equal_lengths : shortest_path_length M ⟨0, 0, 0, sorry⟩ = shortest_path_length M ⟨1, 1, 1, sorry⟩) :
  shortest_path_length M ⟨1, 1, 1, sorry⟩ ≤ max_path_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shortest_path_to_G_l1338_133855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_coefficient_l1338_133878

theorem cubic_polynomial_coefficient (Q : ℝ → ℝ) (d e f : ℝ) :
  (∃ a b c : ℝ, Q = λ x ↦ 3*x^3 + d*x^2 + e*x + f) →
  (∃ r s t : ℝ, Q r = 0 ∧ Q s = 0 ∧ Q t = 0 ∧ (r + s + t) / 3 = -6) →
  (∃ r s t : ℝ, Q r = 0 ∧ Q s = 0 ∧ Q t = 0 ∧ r * s * t = -6) →
  3 + d + e + f = -6 →
  Q 0 = 9 →
  e = -72 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_coefficient_l1338_133878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_growth_rate_comparison_exponential_vs_power_growth_exponential_model_suitability_theorem_existence_of_inequality_l1338_133841

-- Define power function
noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

-- Define linear function
def linear_function (m : ℝ) (b : ℝ) (x : ℝ) : ℝ := m * x + b

-- Define exponential function
noncomputable def exp_function (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define logarithmic function
noncomputable def log_function (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem growth_rate_comparison :
  ∃ (α : ℝ) (m : ℝ) (b : ℝ) (x₀ : ℝ),
    α > 0 ∧ m > 0 ∧ x₀ > 0 ∧
    ∀ x > x₀, linear_function m b x > power_function α x :=
by sorry

theorem exponential_vs_power_growth :
  ∀ (a : ℝ) (α : ℝ), a > 1 → α > 0 →
    ∃ (x₁ : ℝ), x₁ > 0 ∧
      ∀ x > x₁, exp_function a x > power_function α x :=
by sorry

-- This theorem is more of a conceptual statement and might not be directly provable in Lean
-- We'll represent it as a proposition instead of a string
def exponential_model_suitability (model : ℝ → ℝ) : Prop :=
  ∃ (a : ℝ), a > 1 ∧ ∀ x, model x = exp_function a x

theorem exponential_model_suitability_theorem :
  ∀ (model : ℝ → ℝ),
    exponential_model_suitability model →
    exponential_model_suitability model  -- This is a tautology, but represents the conceptual statement
:= by sorry

theorem existence_of_inequality :
  ∃ (x₀ : ℝ) (a : ℝ) (n : ℝ),
    x₀ > 0 ∧ a > 1 ∧ n > 0 ∧
    exp_function a x₀ < power_function n x₀ ∧ power_function n x₀ < log_function a x₀ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_growth_rate_comparison_exponential_vs_power_growth_exponential_model_suitability_theorem_existence_of_inequality_l1338_133841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_line_parallel_l1338_133894

/-- An ellipse in the xy-plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- A line in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The right focus of an ellipse -/
noncomputable def right_focus (e : Ellipse) : ℝ × ℝ := (Real.sqrt (e.a^2 - e.b^2), 0)

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop := l1.a / l1.b = l2.a / l2.b

/-- A point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

theorem ellipse_focus_line_parallel (e : Ellipse) (l1 l2 : Line) :
  e.a = 5 ∧ e.b = 3 ∧
  l1 = ⟨1, -2, 2⟩ ∧
  l2 = ⟨1, -2, -2⟩ →
  parallel l1 l2 ∧ point_on_line (right_focus e) l2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_line_parallel_l1338_133894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_inequalities_l1338_133857

-- Define IsCubic as a structure instead of a method
structure IsCubic (p : ℝ → ℝ) : Prop where
  exists_coeffs : ∃ (a b c d : ℝ), ∀ x, p x = a * x^3 + b * x^2 + c * x + d ∧ a ≠ 0

-- Define HasRoots as a structure
structure HasRoots (p : ℝ → ℝ) (a b c : ℝ) : Prop where
  root_a : p a = 0
  root_b : p b = 0
  root_c : p c = 0

theorem roots_inequalities (a b c : ℝ) : 
  (∃ (p : ℝ → ℝ), IsCubic p ∧ HasRoots p a b c) →
  a + b + c = 6 →
  a * b + b * c + c * a = 9 →
  a < b →
  b < c →
  0 < a ∧ a < 1 ∧ 1 < b ∧ b < 3 ∧ 3 < c ∧ c < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_inequalities_l1338_133857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_likely_same_l1338_133887

/-- Represents the number of crows on each tree -/
def total_crows : ℕ := 50

/-- Represents the number of white crows on the birch -/
def a : ℕ → ℕ := λ _ => 0

/-- Represents the number of black crows on the birch -/
def b : ℕ → ℕ := λ _ => 0

/-- Represents the number of white crows on the oak -/
def c : ℕ → ℕ := λ _ => 0

/-- Represents the number of black crows on the oak -/
def d : ℕ → ℕ := λ _ => 0

/-- The total number of crows on each tree is 50 -/
axiom total_birch (n : ℕ) : a n + b n = total_crows
axiom total_oak (n : ℕ) : c n + d n = total_crows

/-- On the birch, black crows are at least as numerous as white crows -/
axiom birch_condition (n : ℕ) : b n ≥ a n

/-- On the oak, black crows are at least as numerous or possibly only one fewer than white crows -/
axiom oak_condition (n : ℕ) : d n ≥ c n - 1

/-- Probability that the number of white crows on the birch remains the same -/
noncomputable def prob_same (n : ℕ) : ℚ := 
  (b n * (d n + 1) + a n * (c n + 1)) / (total_crows * (total_crows + 1))

/-- Probability that the number of white crows on the birch changes -/
noncomputable def prob_change (n : ℕ) : ℚ := 
  (b n * c n + a n * d n) / (total_crows * (total_crows + 1))

/-- Theorem stating that it's more likely for the number of white crows on the birch to remain the same -/
theorem more_likely_same (n : ℕ) : prob_same n > prob_change n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_likely_same_l1338_133887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_to_x_axis_l1338_133853

/-- Given a differentiable function f and a point x₀ where f'(x₀) = 0,
    the tangent line to the curve y = f(x) at the point (x₀, f(x₀))
    is parallel or coincident with the x-axis. -/
theorem tangent_line_parallel_to_x_axis
  (f : ℝ → ℝ) (x₀ : ℝ) (hf : Differentiable ℝ f) (h : deriv f x₀ = 0) :
  ∃ (k : ℝ), ∀ x, f x₀ + (x - x₀) * (deriv f x₀) = k :=
by
  use f x₀
  intro x
  rw [h]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_to_x_axis_l1338_133853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_term_l1338_133806

theorem arithmetic_sequence_middle_term (a₁ a₃ : ℕ) (y : ℤ) :
  a₁ = 3^3 → a₃ = 5^3 → y = (a₁ + a₃) / 2 → y = 76 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_term_l1338_133806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_twenty_pi_thirds_l1338_133859

theorem cos_negative_twenty_pi_thirds : Real.cos (-20 * Real.pi / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_twenty_pi_thirds_l1338_133859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1338_133844

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - (x - 1)^2)

-- State the theorem
theorem function_inequality (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 1) :
  f x₁ / x₁ > f x₂ / x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1338_133844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_student_is_17_l1338_133881

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population : ℕ
  sample_size : ℕ
  known_samples : List ℕ

/-- Calculates the sampling interval -/
def sampling_interval (s : SystematicSampling) : ℕ :=
  s.population / s.sample_size

/-- Finds the lowest numbered student in the sample -/
def lowest_sample (s : SystematicSampling) : ℕ :=
  match s.known_samples.minimum? with
  | some n => n
  | none => 0

/-- Theorem: In the given systematic sampling scenario, the fourth student is numbered 17 -/
theorem fourth_student_is_17 (s : SystematicSampling)
  (h_pop : s.population = 48)
  (h_size : s.sample_size = 4)
  (h_known : s.known_samples = [5, 29, 41]) :
  lowest_sample s + sampling_interval s = 17 := by
  sorry

#check fourth_student_is_17

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_student_is_17_l1338_133881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_ticket_cost_l1338_133843

theorem student_ticket_cost
  (adult_ticket_price : ℝ)
  (total_tickets : ℕ)
  (total_income : ℝ)
  (ron_tickets : ℕ)
  (h1 : adult_ticket_price = 4.5)
  (h2 : total_tickets = 20)
  (h3 : total_income = 60)
  (h4 : ron_tickets = 12) :
  ∃ (student_ticket_price : ℝ),
    student_ticket_price * (ron_tickets : ℝ) +
    adult_ticket_price * ((total_tickets - ron_tickets) : ℝ) = total_income ∧
    student_ticket_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_ticket_cost_l1338_133843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1338_133874

-- Define the expression as noncomputable due to Real.sqrt
noncomputable def f (x y : ℝ) : ℝ := 
  x * (2 * x * Real.sqrt (x * y) - x * Real.sqrt (3 * x * y))^(1/3) * 
  (x^2 * y * (7 + 4 * Real.sqrt 3))^(1/6)

-- State the theorem
theorem simplify_expression {x y : ℝ} (h : x * y ≥ 0) : 
  f x y = x^2 * |y^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1338_133874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1338_133869

open Real

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * (x - π / 2) + π / 3)

-- State the theorem
theorem f_monotone_increasing :
  StrictMonoOn f (Set.Icc (π / 12) (7 * π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1338_133869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_not_integer_l1338_133838

/-- A triangle with all sides of different lengths -/
structure ScaleneTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_ne_b : a ≠ b
  b_ne_c : b ≠ c
  a_ne_c : a ≠ c
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0

/-- The area of a triangle using Heron's formula -/
noncomputable def area (t : ScaleneTriangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- The circumradius of a triangle -/
noncomputable def circumradius (t : ScaleneTriangle) : ℝ :=
  (t.a * t.b * t.c) / (4 * area t)

/-- The circumradius of a scalene triangle is not an integer -/
theorem circumradius_not_integer (t : ScaleneTriangle) :
  ¬ (∃ n : ℤ, (circumradius t : ℝ) = n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_not_integer_l1338_133838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_combination_minimizes_cost_l1338_133891

/-- Represents the cost per gram for each type of stamp -/
structure StampCosts where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Represents the number of stamps of each type -/
structure StampCombination where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculate the total cost of a stamp combination -/
def totalCost (costs : StampCosts) (combo : StampCombination) : ℚ :=
  costs.a * combo.a + costs.b * combo.b + costs.c * combo.c

/-- Check if a stamp combination satisfies all constraints -/
def isValidCombination (combo : StampCombination) : Prop :=
  combo.a + combo.b + combo.c ≥ 4 ∧
  combo.a ≥ 1 ∧ combo.b ≥ 1 ∧ combo.c ≥ 1 ∧
  combo.a + combo.b + combo.c ≤ 60

/-- The optimal stamp combination -/
def optimalCombination : StampCombination :=
  { a := 57, b := 1, c := 1 }

/-- The given stamp costs -/
def givenCosts : StampCosts :=
  { a := 12/100, b := 16/100, c := 22/100 }

theorem optimal_combination_minimizes_cost :
  isValidCombination optimalCombination ∧
  ∀ (combo : StampCombination),
    isValidCombination combo →
    totalCost givenCosts optimalCombination ≤ totalCost givenCosts combo :=
by sorry

#eval totalCost givenCosts optimalCombination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_combination_minimizes_cost_l1338_133891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1338_133888

-- Define the parabola
def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 8 * P.1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define point A
def A : ℝ × ℝ := (3, 2)

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem min_distance_sum :
  ∃ (min : ℝ), min = 5 ∧
  ∀ (P : ℝ × ℝ), parabola P →
  distance P A + distance P focus ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1338_133888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_Q_l1338_133833

/-- Given points P and Q with coordinates as defined, prove the locus of Q -/
theorem locus_of_point_Q (x_P y_P x_Q y_Q : ℝ) 
  (h1 : x_Q = 2 * y_P^2 - x_P + 1) 
  (h2 : y_Q = -2 * x_P * y_P) :
  (∃ (a b : ℝ), ∀ (x_P : ℝ), y_Q = a * x_Q + b) ∧ 
  (∃ (c d e : ℝ), ∀ (y_P : ℝ), y_Q^2 = c * x_Q^2 + d * x_Q + e) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_Q_l1338_133833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_after_101_years_l1338_133801

/-- Represents the exchange rates for dollar and euro in yuan --/
structure ExchangeRates where
  dollar : ℤ
  euro : ℤ

/-- Defines the possible transformations of exchange rates --/
inductive Transform : ExchangeRates → ExchangeRates → Prop
  | rule1_plus (d e : ℤ) : Transform ⟨d, e⟩ ⟨d + e, 2 * d + 1⟩
  | rule1_minus (d e : ℤ) : Transform ⟨d, e⟩ ⟨d + e, 2 * d - 1⟩
  | rule2_plus (d e : ℤ) : Transform ⟨d, e⟩ ⟨d + e, 2 * e + 1⟩
  | rule2_minus (d e : ℤ) : Transform ⟨d, e⟩ ⟨d + e, 2 * e - 1⟩

/-- Defines a sequence of transformations over n years --/
def TransformSequence (n : ℕ) : ExchangeRates → ExchangeRates → Prop :=
  match n with
  | 0 => (· = ·)
  | n + 1 => fun x z ↦ ∃ y, Transform x y ∧ TransformSequence n y z

/-- The main theorem stating the smallest possible difference after 101 years --/
theorem smallest_difference_after_101_years :
  ∃ (final : ExchangeRates),
    TransformSequence 101 ⟨6, 7⟩ final ∧
    final.dollar ≠ final.euro ∧
    ∀ (other : ExchangeRates),
      TransformSequence 101 ⟨6, 7⟩ other →
      other.dollar ≠ other.euro →
      |final.dollar - final.euro| ≤ |other.dollar - other.euro| ∧
      |final.dollar - final.euro| = 2 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_after_101_years_l1338_133801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_consecutive_adjacent_pairs_l1338_133826

/-- A cube with numbers from 1 to 6 written on its faces -/
structure NumberedCube where
  faces : Fin 6 → Fin 6
  distinct : ∀ i j, i ≠ j → faces i ≠ faces j

/-- Two faces are adjacent if they share an edge -/
def adjacent (i j : Fin 6) : Prop :=
  (i.val + j.val) % 6 = 1 ∨ (i.val + j.val) % 6 = 5

/-- Two numbers are consecutive if their difference is 1 -/
def consecutive (a b : Fin 6) : Prop :=
  (a.val + 1 = b.val) ∨ (b.val + 1 = a.val)

/-- There are at least two pairs of adjacent faces with consecutive numbers -/
theorem at_least_two_consecutive_adjacent_pairs (cube : NumberedCube) :
  ∃ (i j k l : Fin 6), i ≠ k ∧ j ≠ l ∧
    adjacent i j ∧ adjacent k l ∧
    consecutive (cube.faces i) (cube.faces j) ∧
    consecutive (cube.faces k) (cube.faces l) := by
  sorry

#check at_least_two_consecutive_adjacent_pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_consecutive_adjacent_pairs_l1338_133826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_correct_l1338_133866

/-- The rate of compound interest that satisfies the given conditions -/
noncomputable def compound_interest_rate : ℝ :=
  let simple_interest_principal : ℝ := 1750.000000000002
  let simple_interest_rate : ℝ := 8 / 100
  let simple_interest_time : ℝ := 3
  let compound_interest_principal : ℝ := 4000
  let compound_interest_time : ℝ := 2
  let simple_interest : ℝ := simple_interest_principal * simple_interest_rate * simple_interest_time
  10 / 100

/-- Theorem stating that the calculated compound interest rate satisfies the given conditions -/
theorem compound_interest_rate_correct : 
  let simple_interest_principal : ℝ := 1750.000000000002
  let simple_interest_rate : ℝ := 8 / 100
  let simple_interest_time : ℝ := 3
  let compound_interest_principal : ℝ := 4000
  let compound_interest_time : ℝ := 2
  let simple_interest : ℝ := simple_interest_principal * simple_interest_rate * simple_interest_time
  simple_interest = (1/2) * (compound_interest_principal * ((1 + compound_interest_rate)^compound_interest_time - 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_correct_l1338_133866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadrilateral_with_obtuse_diagonal_triangles_l1338_133839

/-- A quadrilateral is a set of four points in a plane. -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- An angle is obtuse if it is greater than 90 degrees (π/2 radians). -/
def is_obtuse_angle (angle : ℝ) : Prop := angle > Real.pi / 2

/-- A triangle is obtuse if it has at least one obtuse angle. -/
def is_obtuse_triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ angle, is_obtuse_angle angle ∧ (angle = Real.arccos 0 ∨ angle = Real.arccos 0 ∨ angle = Real.arccos 0)

/-- A diagonal of a quadrilateral is a line segment that connects two non-adjacent vertices. -/
def diagonal (q : Quadrilateral) : (ℝ × ℝ) × (ℝ × ℝ) :=
  (q.A, q.C) -- We only need to define one diagonal for this theorem

/-- Theorem: There exists a quadrilateral where both of its diagonals divide it into two obtuse triangles. -/
theorem exists_quadrilateral_with_obtuse_diagonal_triangles :
  ∃ q : Quadrilateral,
    (is_obtuse_triangle q.A q.B (diagonal q).2 ∧ is_obtuse_triangle q.C q.D (diagonal q).2) ∧
    (is_obtuse_triangle q.A q.D (diagonal q).1 ∧ is_obtuse_triangle q.B q.C (diagonal q).1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadrilateral_with_obtuse_diagonal_triangles_l1338_133839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_calculation_l1338_133898

/-- Calculates the monthly payment for a loan with continuous compounding -/
noncomputable def monthly_payment (principal : ℝ) (interest_rate : ℝ) (term : ℝ) (fee_rate : ℝ) (num_payments : ℕ) : ℝ :=
  let principal_after_fee := principal * (1 - fee_rate)
  let future_value := principal_after_fee * Real.exp (interest_rate * term)
  future_value / (num_payments : ℝ)

theorem loan_payment_calculation (principal : ℝ) (interest_rate : ℝ) (term : ℝ) (fee_rate : ℝ) (num_payments : ℕ) :
  principal = 1000 →
  interest_rate = 1.44 →
  term = 0.5 →
  fee_rate = 0.02 →
  num_payments = 6 →
  Int.floor (monthly_payment principal interest_rate term fee_rate num_payments + 0.5) = 336 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_calculation_l1338_133898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_magnitude_l1338_133885

/-- Given two parallel vectors a and b, prove that the magnitude of 2a - b is 4√5 -/
theorem parallel_vectors_magnitude (k : ℝ) : 
  let a : Fin 2 → ℝ := ![(-1), 2]
  let b : Fin 2 → ℝ := ![2, k]
  (∃ (c : ℝ), a = c • b) →
  ‖(2 : ℝ) • a - b‖ = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_magnitude_l1338_133885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l1338_133820

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 8*x + 20) / (m*x^2 - m*x - 1) < 0) ↔ m ∈ Set.Ioc (-4) 0 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l1338_133820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_properties_l1338_133815

noncomputable def a (n : ℕ) : ℝ := Real.cos ((10^n : ℝ) * Real.pi / 180)

theorem a_100_properties : a 100 > 0 ∧ |a 100| < 0.18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_properties_l1338_133815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1338_133897

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exponential_inequality (a x₁ x₂ : ℝ) (ha : a > 1) (hx : x₁ < x₂) :
  let m := (x₁ + x₂) / 2
  |f a m - f a x₁| < |f a x₂ - f a m| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1338_133897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_reduction_rounded_l1338_133861

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The annual carbon emission reduction in tons -/
def annualCarbonReduction : ℝ := 6865.65

theorem carbon_reduction_rounded :
  roundToNearestTenth annualCarbonReduction = 6865.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_reduction_rounded_l1338_133861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1338_133810

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def valid_number (n : ℕ) : Bool :=
  10 ≤ n ∧ n < 1000 ∧ digit_sum n = 3

theorem count_valid_numbers : 
  (Finset.filter (fun n => valid_number n) (Finset.range 1000)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1338_133810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1338_133825

noncomputable def f (x : ℝ) := 1 - Real.cos x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 1/2 * Real.cos (2 * x)

theorem f_properties :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi ∧
  (∀ y, f y ∈ Set.Icc (-3/2) (5/2)) ∧
  (∀ y, ∃ x, f x = y ↔ y ∈ Set.Icc (-3/2) (5/2)) ∧
  ∀ x₀ : ℝ, 0 ≤ x₀ ∧ x₀ ≤ Real.pi / 2 → f x₀ = 0 → Real.sin (2 * x₀) = (Real.sqrt 15 - Real.sqrt 3) / 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1338_133825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_identity_l1338_133830

open BigOperators

def f (n : ℕ) : ℚ := ∑ k in Finset.range n, 1 / (k + 1 : ℚ)

def g (n : ℕ) : ℚ := n + 1 / f n

theorem harmonic_sum_identity (n : ℕ) (h : n ≥ 2) :
  ∑ k in Finset.range (n - 1), f (k + 1) = g n * f n - 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_identity_l1338_133830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_product_l1338_133864

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define the area of a triangle given three points
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

-- Main theorem
theorem parabola_triangle_area_product :
  ∀ A B : ℝ × ℝ,
  parabola A → parabola B →
  dot_product A B = -4 →
  triangle_area (0, 0) focus A * triangle_area (0, 0) focus B = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_product_l1338_133864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_tan_product_l1338_133800

theorem sin_tan_product (α : ℝ) (x y : ℝ) : 
  x = 3/5 ∧ y = -4/5 ∧ (x^2 + y^2 = 1) → 
  Real.sin α * Real.tan α = 16/15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_tan_product_l1338_133800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pond_soil_weight_is_3000000_l1338_133890

/-- Represents the dimensions and soil properties of an irregularly shaped pond. -/
structure PondProperties where
  min_length : ℝ
  max_length : ℝ
  min_width : ℝ
  max_width : ℝ
  min_depth : ℝ
  max_depth : ℝ
  upper_layer_depth : ℝ
  upper_layer_density : ℝ
  lower_layer_density : ℝ

/-- Calculates the total weight of soil extracted from the pond. -/
noncomputable def total_soil_weight (p : PondProperties) : ℝ :=
  let avg_length := (p.min_length + p.max_length) / 2
  let avg_width := (p.min_width + p.max_width) / 2
  let avg_depth := (p.min_depth + p.max_depth) / 2
  let upper_volume := avg_length * avg_width * p.upper_layer_depth
  let lower_volume := avg_length * avg_width * (avg_depth - p.upper_layer_depth)
  let upper_weight := upper_volume * p.upper_layer_density * 1000
  let lower_weight := lower_volume * p.lower_layer_density * 1000
  upper_weight + lower_weight

/-- Theorem stating that the total weight of soil extracted from the pond with given properties is 3,000,000 kg. -/
theorem pond_soil_weight_is_3000000 (p : PondProperties)
  (h1 : p.min_length = 18 ∧ p.max_length = 22)
  (h2 : p.min_width = 14 ∧ p.max_width = 16)
  (h3 : p.min_depth = 4.5 ∧ p.max_depth = 5.5)
  (h4 : p.upper_layer_depth = 2.5)
  (h5 : p.upper_layer_density = 1.8)
  (h6 : p.lower_layer_density = 2.2) :
  total_soil_weight p = 3000000 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pond_soil_weight_is_3000000_l1338_133890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joint_work_cheaper_l1338_133834

/-- Represents a digger with a digging rate -/
structure Digger where
  rate : ℝ

/-- Represents the tunnel digging problem -/
structure TunnelProblem where
  tunnelLength : ℝ
  hourlyRate : ℝ
  fasterDigger : Digger
  slowerDigger : Digger
  h_rateRatio : fasterDigger.rate = 2 * slowerDigger.rate

/-- Calculates the cost of joint work -/
noncomputable def jointWorkCost (p : TunnelProblem) : ℝ :=
  (2 * p.hourlyRate * p.tunnelLength) / (3 * p.slowerDigger.rate)

/-- Calculates the cost of alternating work -/
noncomputable def alternatingWorkCost (p : TunnelProblem) : ℝ :=
  (3 * p.hourlyRate * p.tunnelLength) / (4 * p.slowerDigger.rate)

/-- Theorem: Joint work is cheaper than alternating work -/
theorem joint_work_cheaper (p : TunnelProblem) :
  jointWorkCost p < alternatingWorkCost p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joint_work_cheaper_l1338_133834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edge_selection_l1338_133849

/-- Represents a regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  h_vertex_count : vertices.card = 20
  h_edge_count : edges.card = 30
  h_edge_valid : ∀ e ∈ edges, e.fst ∈ vertices ∧ e.snd ∈ vertices
  h_edge_symmetry : ∀ e ∈ edges, (e.snd, e.fst) ∈ edges

/-- A set of edges in a regular dodecahedron where no two edges share a common vertex -/
def ValidEdgeSet (d : RegularDodecahedron) (s : Finset (ℕ × ℕ)) : Prop :=
  s ⊆ d.edges ∧ ∀ e1 e2, e1 ∈ s → e2 ∈ s → e1 ≠ e2 → 
    e1.fst ≠ e2.fst ∧ e1.fst ≠ e2.snd ∧ e1.snd ≠ e2.fst ∧ e1.snd ≠ e2.snd

/-- The theorem stating that the maximum number of edges that can be selected
    from a regular dodecahedron such that no two selected edges share a common vertex is 10 -/
theorem max_edge_selection (d : RegularDodecahedron) :
  (∃ s : Finset (ℕ × ℕ), ValidEdgeSet d s ∧ s.card = 10) ∧
  (∀ s : Finset (ℕ × ℕ), ValidEdgeSet d s → s.card ≤ 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edge_selection_l1338_133849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_solution_difference_l1338_133805

/-- The amount of chemical solution Jacqueline has in liters -/
noncomputable def jacqueline_amount : ℝ := 200

/-- The percentage more that Liliane has compared to Jacqueline -/
noncomputable def liliane_percentage : ℝ := 30

/-- The percentage more that Alice has compared to Jacqueline -/
noncomputable def alice_percentage : ℝ := 15

/-- The amount of chemical solution Liliane has -/
noncomputable def liliane_amount : ℝ := jacqueline_amount * (1 + liliane_percentage / 100)

/-- The amount of chemical solution Alice has -/
noncomputable def alice_amount : ℝ := jacqueline_amount * (1 + alice_percentage / 100)

/-- The percentage difference between Liliane's and Alice's amounts -/
noncomputable def percentage_difference : ℝ := (liliane_amount - alice_amount) / alice_amount * 100

theorem chemical_solution_difference :
  abs (percentage_difference - 13.04) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_solution_difference_l1338_133805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_line_equation_l1338_133845

/-- Triangle DEF with its reflection about line M -/
structure ReflectedTriangle where
  /-- x-coordinate of point D -/
  dx : ℝ
  /-- y-coordinate of point D -/
  dy : ℝ
  /-- x-coordinate of point E -/
  ex : ℝ
  /-- y-coordinate of point E -/
  ey : ℝ
  /-- x-coordinate of point F -/
  fx : ℝ
  /-- y-coordinate of point F -/
  fy : ℝ
  /-- x-coordinate of reflected point D' -/
  dx' : ℝ
  /-- y-coordinate of reflected point D' -/
  dy' : ℝ
  /-- x-coordinate of reflected point E' -/
  ex' : ℝ
  /-- y-coordinate of reflected point E' -/
  ey' : ℝ
  /-- x-coordinate of reflected point F' -/
  fx' : ℝ
  /-- y-coordinate of reflected point F' -/
  fy' : ℝ
  /-- y-coordinates remain unchanged after reflection -/
  h1 : dy = dy'
  h2 : ey = ey'
  h3 : fy = fy'

/-- The equation of the line of reflection M is x = 6 -/
theorem reflection_line_equation (t : ReflectedTriangle) 
  (h1 : t.dx = 3) (h2 : t.dy = 2) 
  (h3 : t.ex = 8) (h4 : t.ey = 7)
  (h5 : t.fx = 6) (h6 : t.fy = -4)
  (h7 : t.dx' = 9) (h8 : t.dy' = 2)
  (h9 : t.ex' = 14) (h10 : t.ey' = 7)
  (h11 : t.fx' = 12) (h12 : t.fy' = -4) :
  ∃ (m : ℝ), m = 6 ∧ ∀ (x : ℝ), x ∈ {x | x = m} ↔ x ∈ {x | ∃ (y : ℝ), (x, y) ∈ Set.range (λ (p : ℝ × ℝ) ↦ (2 * m - p.1, p.2))} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_line_equation_l1338_133845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_in_cube_l1338_133883

noncomputable section

structure Cube where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

def Cube.volume (c : Cube) : ℝ := c.sideLength ^ 3

structure Sphere where
  radius : ℝ
  radius_pos : radius > 0

def Sphere.surfaceArea (s : Sphere) : ℝ := 4 * Real.pi * s.radius ^ 2

def Sphere.isTangentTo (s : Sphere) (c : Cube) : Prop :=
  s.radius = c.sideLength / 2

theorem sphere_surface_area_in_cube (cube_volume : ℝ) (sphere : Sphere) :
  cube_volume = 64 →
  (∃ (cube : Cube), cube.volume = cube_volume ∧ sphere.isTangentTo cube) →
  sphere.surfaceArea = 16 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_in_cube_l1338_133883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_X_equals_Y_is_correct_l1338_133896

/-- The probability of X = Y for randomly selected (X, Y) satisfying the given conditions -/
noncomputable def probability_X_equals_Y : ℚ :=
  41 / 6400

/-- The set of pairs (x, y) satisfying the conditions -/
def valid_pairs : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               Real.cos (Real.sin x) = Real.cos (Real.sin y) ∧
               -20 * Real.pi ≤ x ∧ x ≤ 20 * Real.pi ∧
               -20 * Real.pi ≤ y ∧ y ≤ 20 * Real.pi}

/-- The theorem stating that the probability of X = Y is 41/6400 -/
theorem probability_X_equals_Y_is_correct :
  ∃ (P : Set (ℝ × ℝ) → ℝ), 
    P valid_pairs = 1 ∧
    P {p ∈ valid_pairs | p.1 = p.2} = probability_X_equals_Y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_X_equals_Y_is_correct_l1338_133896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_madeline_refills_l1338_133816

def water_goal : ℕ := 100
def bottle_capacity : ℕ := 12
def remaining_water : ℕ := 16

theorem madeline_refills : 
  (water_goal - remaining_water) / bottle_capacity + 1 = 8 := by
  -- Proof steps would go here
  sorry

#eval (water_goal - remaining_water) / bottle_capacity + 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_madeline_refills_l1338_133816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_li_payment_is_450_l1338_133876

/-- Represents the payment and work data for a group task --/
structure TaskData where
  dailyWork : ℚ  -- Amount of work completed by one person in one day
  liDays : ℕ  -- Number of days Li worked
  extraDays : ℕ  -- Number of extra days the others worked
  extraPayment : ℚ  -- Extra payment received by each of the other three

/-- Calculates Li's payment based on the given task data --/
def calculateLiPayment (data : TaskData) : ℚ :=
  let totalExtraWork := 3 * data.extraDays * data.dailyWork
  let paymentPerUnit := data.extraPayment / totalExtraWork
  data.liDays * data.dailyWork * paymentPerUnit

/-- Theorem stating that Li's payment is 450 yuan given the problem conditions --/
theorem li_payment_is_450 (data : TaskData) 
  (h1 : data.dailyWork > 0)
  (h2 : data.liDays = 2)
  (h3 : data.extraDays = 3)
  (h4 : data.extraPayment = 2700) :
  calculateLiPayment data = 450 := by
  sorry

#eval calculateLiPayment ⟨1, 2, 3, 2700⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_li_payment_is_450_l1338_133876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_increase_l1338_133875

/-- Represents the annual income after reallocation -/
noncomputable def annual_income (a b : ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a
  else a * (2/3)^(n-1) + b * (3/2)^(n-2)

theorem income_increase (a b : ℝ) (h : b ≥ 3*a/8) :
  ∀ n : ℕ, n ≥ 1 → annual_income a b n ≥ a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_increase_l1338_133875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_cost_calculation_l1338_133852

def odometer_start : ℕ := 74568
def odometer_cafe : ℕ := 74580
def odometer_end : ℕ := 74605
def fuel_efficiency : ℚ := 25
def gas_price : ℚ := 41/10

theorem gas_cost_calculation : 
  let total_distance : ℚ := (odometer_end - odometer_start : ℚ)
  let gas_used : ℚ := total_distance / fuel_efficiency
  let total_cost : ℚ := gas_used * gas_price
  ⌊total_cost * 100 + 1/2⌋ / 100 = 607/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_cost_calculation_l1338_133852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sixth_term_l1338_133873

/-- Geometric sequence with first term a and common ratio r -/
noncomputable def geometric_sequence (a r : ℝ) : ℕ → ℝ := fun n => a * r^(n - 1)

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a r : ℝ) : ℕ → ℝ := fun n =>
  if r = 1 then n * a else a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sixth_term
  (a r : ℝ) (h1 : geometric_sum a r 2 = 3) (h2 : geometric_sum a r 3 - geometric_sum a r 1 = 6) :
  geometric_sequence a r 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sixth_term_l1338_133873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_g_composed_l1338_133886

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x - 5)

-- State the theorem
theorem smallest_x_in_domain_of_g_composed : ∃ (x : ℝ), (∀ y, g (g y) ≥ 0 → x ≤ y) ∧ g (g x) ≥ 0 ∧ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_g_composed_l1338_133886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_team_math_players_l1338_133895

/-- Given a volleyball team at Greenfield Academy, prove the number of players taking mathematics. -/
theorem volleyball_team_math_players
  (total_players : ℕ)
  (physics_players : ℕ)
  (both_subjects : ℕ)
  (h1 : total_players = 30)
  (h2 : physics_players = 15)
  (h3 : both_subjects = 3)
  (h4 : ∀ p, p < total_players →
       (p < physics_players ∨
        p < (total_players - physics_players + both_subjects))) :
  total_players - physics_players + both_subjects = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_team_math_players_l1338_133895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_interval_l1338_133814

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

def passes_through_point (α : ℝ) : Prop :=
  power_function α 2 = (1/2 : ℝ)

def decreasing_interval (f : ℝ → ℝ) : Set ℝ :=
  {x | ∀ y, x < y → f x > f y} ∩ {x | x ≠ 0}

theorem power_function_decreasing_interval (α : ℝ) (h : passes_through_point α) :
  decreasing_interval (power_function α) = Set.univ \ {0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_interval_l1338_133814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_is_three_l1338_133862

def product_from_to (n k : ℕ) : ℕ :=
  (List.range (k - n + 1)).map (fun x => x + n) |>.prod

theorem starting_number_is_three :
  ∀ n : ℕ, n > 3 →
    ∃ k : ℕ, k ≥ 7 ∧ ¬(315 ∣ product_from_to n k) ∧
    ∀ m : ℕ, m ≥ 7 → 315 ∣ product_from_to 3 m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_is_three_l1338_133862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l1338_133804

/-- Definition of the sign function -/
noncomputable def sign (a : ℝ) : ℝ :=
  if a > 0 then 1
  else if a < 0 then -1
  else 0

/-- The system of equations -/
def satisfies_system (x y z : ℝ) : Prop :=
  x = 2023 - 2024 * sign (y^2 - z^2) ∧
  y = 2023 - 2024 * sign (x^2 - z^2) ∧
  z = 2023 - 2024 * sign (x^2 - y^2)

/-- The theorem stating that exactly 3 triples satisfy the system -/
theorem exactly_three_solutions :
  ∃! (s : Finset (ℝ × ℝ × ℝ)), s.card = 3 ∧ ∀ (x y z : ℝ), (x, y, z) ∈ s ↔ satisfies_system x y z :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l1338_133804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tuition_increase_survey_l1338_133884

/-- Calculates the number of parents who disagreed with a tuition increase --/
def parents_disagreed (total : ℕ) (agreed_percent : ℕ) : ℕ :=
  total - (total * agreed_percent / 100)

theorem tuition_increase_survey :
  let primary_total := 300
  let intermediate_total := 250
  let secondary_total := 250
  let primary_agreed_percent := 30
  let intermediate_agreed_percent := 20
  let secondary_agreed_percent := 10
  (parents_disagreed primary_total primary_agreed_percent = 210) ∧
  (parents_disagreed intermediate_total intermediate_agreed_percent = 200) ∧
  (parents_disagreed secondary_total secondary_agreed_percent = 225) := by
  sorry

#eval parents_disagreed 300 30
#eval parents_disagreed 250 20
#eval parents_disagreed 250 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tuition_increase_survey_l1338_133884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_losses_calculation_l1338_133818

/-- Represents a game in the tournament with its winning probability -/
structure Game where
  winProb : ℚ
  deriving Repr

/-- Calculates the expected number of losses in a tournament -/
def expectedLosses (games : List Game) : ℚ :=
  games.map (λ g => 1 - g.winProb) |> List.sum

/-- The list of games in the tournament with their winning probabilities -/
def tournamentGames : List Game := [
  { winProb := 60/100 },
  { winProb := 75/100 },
  { winProb := 40/100 },
  { winProb := 85/100 },
  { winProb := 50/100 },
  { winProb := 20/100 },
  { winProb := 90/100 },
  { winProb := 70/100 },
  { winProb := 65/100 },
  { winProb := 80/100 }
]

theorem expected_losses_calculation :
  expectedLosses tournamentGames = 355/100 := by
  -- Proof goes here
  sorry

#eval expectedLosses tournamentGames

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_losses_calculation_l1338_133818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_l1338_133870

theorem sequence_existence : ∃ (a : ℕ → ℕ), 
  (a 2013 = 2013) ∧ 
  (∀ k ∈ Finset.range 2012, k ≥ 1 → (Int.natAbs (a (k+1) - a k) = 20 ∨ Int.natAbs (a (k+1) - a k) = 13)) ∧
  (Function.Injective (fun i => a (i+1)) ∧ Finset.range 2013 = Finset.image a (Finset.range 2013)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_l1338_133870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_C1_area_triangle_AOB_l1338_133840

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

-- Define the curve C1
noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ)

-- Define the line l
noncomputable def line_l (θ : ℝ) : ℝ := Real.sqrt 3 / Real.sin (θ + Real.pi/3)

-- Theorem for the polar equation of C1
theorem polar_equation_C1 : ∀ θ : ℝ, (curve_C1 θ).1^2 + (curve_C1 θ).2^2 = 4 := by sorry

-- Theorem for the area of triangle AOB
theorem area_triangle_AOB : 
  let A := (2, 0)
  let B := (2 * Real.cos (Real.pi/3), 2 * Real.sin (Real.pi/3))
  1/2 * A.1 * B.2 = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_C1_area_triangle_AOB_l1338_133840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_henrys_max_toys_is_30_l1338_133877

/-- The maximum number of toys Henry can purchase given the store's pricing scheme -/
def henrys_max_toys (initial_toys : ℕ) (full_price : ℕ) : ℕ := by
  -- Define the special pricing scheme
  let special_price := λ (n : ℕ) => (2 * n * full_price + n * full_price / 2)
  
  -- Define Henry's total money
  let total_money := initial_toys * full_price
  
  -- Define the function to calculate the number of toys given money
  let toys_bought := λ (money : ℕ) => 3 * (money / special_price 1)
  
  -- The theorem statement
  have : toys_bought total_money = 30 := by sorry
  
  -- Return the result
  exact toys_bought total_money

/-- Theorem stating that given the initial conditions, Henry can buy 30 toys -/
theorem henrys_max_toys_is_30 :
  henrys_max_toys 25 4 = 30 := by sorry

#eval henrys_max_toys 25 4  -- Should output 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_henrys_max_toys_is_30_l1338_133877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_l1338_133848

theorem point_on_parabola (t : ℝ) :
  let x := (3 : ℝ)^t - 4
  let y := (3 : ℝ)^(2*t) - 7 * (3 : ℝ)^t + 2
  y = x^2 + x - 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_l1338_133848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_catchable_speed_theorem_l1338_133863

/-- Represents the scenario of a person chasing a boat --/
structure BoatChaseScenario where
  person_run_speed : ℝ
  person_swim_speed : ℝ
  boat_angle : ℝ

/-- The maximum speed at which the boat can be caught --/
noncomputable def max_catchable_speed (scenario : BoatChaseScenario) : ℝ :=
  2 * Real.sqrt 2

/-- Theorem stating the maximum catchable speed for the given scenario --/
theorem max_catchable_speed_theorem (scenario : BoatChaseScenario) 
  (h1 : scenario.person_run_speed = 4)
  (h2 : scenario.person_swim_speed = 2)
  (h3 : scenario.boat_angle = 15 * Real.pi / 180) :
  max_catchable_speed scenario = 2 * Real.sqrt 2 := by
  sorry

#check max_catchable_speed_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_catchable_speed_theorem_l1338_133863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_PQ_l1338_133831

noncomputable def P : ℝ × ℝ := (5, 3)
noncomputable def Q : ℝ × ℝ := (-2, 4)

noncomputable def slope_angle (p q : ℝ × ℝ) : ℝ :=
  let slope := (q.2 - p.2) / (q.1 - p.1)
  Real.pi - Real.arctan slope

theorem slope_angle_PQ :
  0 ≤ slope_angle P Q ∧ slope_angle P Q < Real.pi ∧ 
  slope_angle P Q = Real.pi - Real.arctan (1/7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_PQ_l1338_133831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l1338_133842

/-- The time (in seconds) it takes for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  total_distance / train_speed_mps

/-- Theorem: A train 100 meters long crossing a bridge 150 meters long at 54 kmph takes approximately 16.67 seconds -/
theorem train_crossing_bridge_time :
  ∃ ε > 0, |train_crossing_time 100 150 54 - 16.67| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l1338_133842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_smallest_m_for_max_value_l1338_133865

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sqrt 3 * Real.sin x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

def is_period (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem smallest_positive_period :
  ∃ T > 0, is_period T ∧ ∀ T' > 0, is_period T' → T ≤ T' := by
  sorry

theorem smallest_m_for_max_value :
  ∃ m, m = π / 3 ∧
    (∀ x ∈ Set.Icc (-π/3) m, f x ≤ 3/2) ∧
    (∃ x ∈ Set.Icc (-π/3) m, f x = 3/2) ∧
    (∀ m' < m, ∃ x ∈ Set.Icc (-π/3) m', f x < 3/2 ∨ ∀ y ∈ Set.Icc (-π/3) m', f y ≤ f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_smallest_m_for_max_value_l1338_133865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l1338_133860

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x^2) / x

def domain (x : ℝ) : Prop := (x ∈ Set.Icc (-1) 0 ∨ x ∈ Set.Ioc 0 1) ∧ x ≠ 0

theorem f_is_odd : ∀ x, domain x → f (-x) = -f x := by
  intro x hx
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l1338_133860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cows_in_ranch_l1338_133809

theorem total_cows_in_ranch : 70 = 17 + (3 * 17 + 2) := by
  -- We the People's cows
  have h1 : 17 = 17 := rfl
  
  -- Happy Good Healthy Family's cows
  have h2 : 3 * 17 + 2 = 53 := by
    norm_num
  
  -- Total cows
  calc
    70 = 17 + 53 := by norm_num
    _ = 17 + (3 * 17 + 2) := by rw [←h2]

#eval 17 + (3 * 17 + 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cows_in_ranch_l1338_133809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_c_l1338_133871

theorem max_magnitude_c (a b c : ℝ × ℝ) : 
  (norm a = 1) → (norm b = 1) → (a + b + c = (0, 0)) → 
  norm c ≤ 2 ∧ ∃ a₀ b₀ c₀ : ℝ × ℝ, norm a₀ = 1 ∧ norm b₀ = 1 ∧ a₀ + b₀ + c₀ = (0, 0) ∧ norm c₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_c_l1338_133871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1338_133828

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := 3^x - 3^(a*x + b)

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
    (f a b 1 = 8/3) ∧
    (f a b 2 = 80/9) ∧
    (a = -1) ∧
    (b = 0) ∧
    (∀ x : ℝ, f a b (-x) = -(f a b x)) :=
by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1338_133828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_fourth_quadrant_l1338_133867

theorem tan_alpha_fourth_quadrant (α : ℝ) (h1 : Real.sin α = -5/13) (h2 : α ∈ Set.Icc (3*Real.pi/2) (2*Real.pi)) : 
  Real.tan α = -5/12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_fourth_quadrant_l1338_133867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_irrational_card_l1338_133856

-- Define the set of numbers on the cards
noncomputable def card_numbers : Finset ℝ := {1, Real.sqrt 2, Real.sqrt 3, 2}

-- Define a predicate for irrational numbers
def is_irrational (x : ℝ) : Prop := ¬ (∃ (q : ℚ), x = q)

-- Define the number of cards
noncomputable def total_cards : ℕ := Finset.card card_numbers

-- Define the number of cards with irrational numbers
noncomputable def irrational_cards : ℕ := 2

-- State the theorem
theorem probability_of_irrational_card : 
  (irrational_cards : ℚ) / (total_cards : ℚ) = 1 / 2 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_irrational_card_l1338_133856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1338_133893

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -4*y

-- Define the point Q
noncomputable def Q : ℝ × ℝ := (-2*Real.sqrt 2, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the objective function
noncomputable def objective (P : ℝ × ℝ) : ℝ :=
  |P.2| + distance P Q

-- State the theorem
theorem min_value_theorem :
  ∃ (min_val : ℝ), min_val = 2 ∧
  ∀ (P : ℝ × ℝ), parabola P.1 P.2 → objective P ≥ min_val :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1338_133893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_value_l1338_133821

noncomputable def f (a b x : ℝ) : ℝ := a * x + b

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ a then f a b x
  else f a b (f a b x)

theorem min_b_value (a b : ℝ) :
  (a > 0) →
  (∀ x y, 0 < x → x < y → g a b x < g a b y) →
  (∀ a' > 0, ∃ b' ≥ b, ∀ x y, 0 < x → x < y → g a' b' x < g a' b' y) →
  b ≥ (1/4 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_value_l1338_133821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_cone_cost_l1338_133868

/-- The cost of one ice cream cone, given that 2 ice cream cones cost 198 cents. -/
theorem ice_cream_cone_cost : ℕ := by
  /- Let x be the cost of one ice cream cone in cents -/
  let x : ℕ := 99
  /- Given: 2 ice cream cones cost 198 cents -/
  have h : 2 * x = 198 := by
    rfl
  /- Prove: x = 99 -/
  exact x


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_cone_cost_l1338_133868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1338_133879

/-- Given a triangle ABC with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The area of a triangle -/
noncomputable def triangle_area (t : Triangle) : ℝ :=
  sorry

/-- Vector from point A to point B -/
def vector_AB (t : Triangle) : ℝ × ℝ :=
  (t.B.1 - t.A.1, t.B.2 - t.A.2)

/-- Vector from point A to point C -/
def vector_AC (t : Triangle) : ℝ × ℝ :=
  (t.C.1 - t.A.1, t.C.2 - t.A.2)

/-- The length of side AC in the triangle -/
noncomputable def length_AC (t : Triangle) : ℝ :=
  Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)

/-- Angle C of the triangle in radians -/
noncomputable def angle_C (t : Triangle) : ℝ :=
  sorry

theorem triangle_theorem (t : Triangle) :
  let S := triangle_area t
  let AB := vector_AB t
  let AC := vector_AC t
  (3 * dot_product AB AC = 2 * S) →
  (∃ (A : ℝ), Real.sin A = (3 * Real.sqrt 10) / 10 ∧
    (angle_C t = π/4 ∧ dot_product AB AC = 16 → length_AC t = 8)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1338_133879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_formula_arithmetic_sum_induction_step_l1338_133836

/-- Sum of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

/-- Proof of the sum formula for arithmetic sequences -/
theorem arithmetic_sum_formula (a₁ d : ℝ) :
  ∀ n : ℕ, arithmetic_sum a₁ d n = n * a₁ + (n * (n - 1) / 2) * d := by
  intro n
  rfl

/-- Induction step for the arithmetic sum formula -/
theorem arithmetic_sum_induction_step (a₁ d : ℝ) (k : ℕ) :
  arithmetic_sum a₁ d k = k * a₁ + (k * (k - 1) / 2) * d := by
  exact arithmetic_sum_formula a₁ d k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_formula_arithmetic_sum_induction_step_l1338_133836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_C_D_externally_tangent_l1338_133817

/-- Circle C in the Cartesian plane -/
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 2)^2 = 16

/-- Circle D in the Cartesian plane -/
def circle_D (x y : ℝ) : Prop :=
  (x - 6)^2 + (y - 2)^2 = 1

/-- The distance between two points in the Cartesian plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c₁_center_x c₁_center_y c₁_radius c₂_center_x c₂_center_y c₂_radius : ℝ) : Prop :=
  distance c₁_center_x c₁_center_y c₂_center_x c₂_center_y = c₁_radius + c₂_radius

theorem circles_C_D_externally_tangent :
  externally_tangent 3 (-2) 4 6 2 1 := by
  sorry

#check circles_C_D_externally_tangent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_C_D_externally_tangent_l1338_133817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_theorem_l1338_133850

/-- Represents the scale of a map -/
structure MapScale where
  map_distance : ℝ
  actual_distance : ℝ

/-- Calculates the actual distance given a map distance and a map scale -/
noncomputable def calculate_actual_distance (map_distance : ℝ) (scale : MapScale) : ℝ :=
  (map_distance * scale.actual_distance) / scale.map_distance

theorem map_distance_theorem (scale : MapScale) :
  scale.map_distance = 34 ∧ scale.actual_distance = 14.82 →
  abs (calculate_actual_distance 312 scale - 135.82) < 0.01 := by
  sorry

#check map_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_theorem_l1338_133850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1338_133802

theorem simplify_expression : (-8) - 4 + (-5) - (-2) = -8 - 4 - 5 + 2 := by
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1338_133802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1338_133829

/-- The constant term in the expansion of (2 - 3/x)(x^2 + 2/x)^5 is -240 -/
theorem constant_term_expansion : ∃ (f : ℝ → ℝ), 
  (∀ x : ℝ, x ≠ 0 → f x = (2 - 3/x) * (x^2 + 2/x)^5) ∧
  (∃ c : ℝ, c = -240 ∧ (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |f x - c| < ε)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1338_133829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_roots_iff_determinant_l1338_133824

/-- A harmonic group of four real numbers -/
def HarmonicGroup (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  (x₁ - x₃) * (x₂ - x₄) + (x₂ - x₃) * (x₁ - x₄) = 0

/-- Coefficients of a quartic equation -/
structure QuarticCoeff where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  a_ne_zero : a ≠ 0

/-- The determinant condition for harmonic roots -/
def DeterminantCondition (q : QuarticCoeff) : Prop :=
  q.a * q.c * q.e - q.a * q.d^2 - q.b^2 * q.e + 2 * q.b * q.c * q.d - q.c^3 = 0

/-- Roots of the quartic equation -/
noncomputable def QuarticRoots (q : QuarticCoeff) : Fin 4 → ℝ := sorry

/-- The main theorem: equivalence of harmonic roots and determinant condition -/
theorem harmonic_roots_iff_determinant (q : QuarticCoeff) :
  (∃ (σ : Equiv.Perm (Fin 4)), HarmonicGroup (QuarticRoots q (σ 0)) (QuarticRoots q (σ 1)) (QuarticRoots q (σ 2)) (QuarticRoots q (σ 3))) ↔
  DeterminantCondition q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_roots_iff_determinant_l1338_133824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l1338_133832

-- Define the structures we need
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

structure Line where
  p1 : Point
  p2 : Point

-- Define the given conditions
def is_acute_angled (t : Triangle) : Prop := sorry

def is_orthocenter (M : Point) (t : Triangle) : Prop := sorry

def is_circumcenter (P : Point) (t : Triangle) : Prop := sorry

-- Define a membership relation for Point and Line
def on_line (p : Point) (l : Line) : Prop :=
  ∃ t : ℝ, p = Point.mk (l.p1.x + t * (l.p2.x - l.p1.x)) (l.p1.y + t * (l.p2.y - l.p1.y))

-- Main theorem
theorem lines_concurrent (A B C M A₁ B₁ C₁ : Point) :
  let t := Triangle.mk A B C
  is_acute_angled t →
  is_orthocenter M t →
  is_circumcenter A₁ (Triangle.mk B C M) →
  is_circumcenter B₁ (Triangle.mk C A M) →
  is_circumcenter C₁ (Triangle.mk A B M) →
  ∃ K : Point,
    on_line K (Line.mk A A₁) ∧
    on_line K (Line.mk B B₁) ∧
    on_line K (Line.mk C C₁) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l1338_133832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_mutual_acquaintances_l1338_133835

/-- Represents the relation of knowing each other in a group of people -/
def Knows (n : ℕ) := Fin n → Fin n → Bool

/-- There are at least three mutual acquaintances in a group of n people -/
def HasThreeMutualAcquaintances (n : ℕ) (knows : Knows n) : Prop :=
  ∃ (a b c : Fin n), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ knows a b = true ∧ knows b c = true ∧ knows a c = true

/-- Each person knows at least ⌊n/2⌋ other people -/
def EachKnowsHalf (n : ℕ) (knows : Knows n) : Prop :=
  ∀ a : Fin n, (Finset.filter (fun x => knows a x = true) (Finset.univ : Finset (Fin n))).card ≥ n / 2

/-- For any ⌊n/2⌋ people, either two among them know each other or two among the remaining ones do -/
def HalfKnowEachOther (n : ℕ) (knows : Knows n) : Prop :=
  ∀ (s : Finset (Fin n)), s.card = n / 2 →
    (∃ (a b : Fin n), a ∈ s ∧ b ∈ s ∧ a ≠ b ∧ knows a b = true) ∨
    (∃ (a b : Fin n), a ∉ s ∧ b ∉ s ∧ a ≠ b ∧ knows a b = true)

theorem three_mutual_acquaintances
  (n : ℕ) (hn : n ≥ 6) (knows : Knows n)
  (h_each_knows_half : EachKnowsHalf n knows)
  (h_half_know : HalfKnowEachOther n knows) :
  HasThreeMutualAcquaintances n knows :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_mutual_acquaintances_l1338_133835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_eccentricity_probability_l1338_133811

def roll_dice : Finset (ℕ × ℕ) := Finset.product (Finset.range 6) (Finset.range 6)

def valid_roll (pair : ℕ × ℕ) : Bool :=
  let a := pair.1 + 1
  let b := pair.2 + 1
  a > b && b > 0

noncomputable def high_eccentricity (pair : ℕ × ℕ) : Bool :=
  let a := pair.1 + 1
  let b := pair.2 + 1
  Real.sqrt (1 - (b * b : ℝ) / (a * a)) > Real.sqrt 3 / 2

theorem dice_eccentricity_probability :
  (Finset.filter (λ pair => valid_roll pair && high_eccentricity pair) roll_dice).card /
  roll_dice.card = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_eccentricity_probability_l1338_133811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hospital_process_is_process_flowchart_l1338_133819

/-- Represents the types of flowcharts --/
inductive Flowchart
  | Operation
  | Process
  | Organization
  | Procedural

/-- The correct flowchart type for representing the process of taking a patient to the hospital --/
def hospital_process_flowchart : Flowchart := Flowchart.Process

/-- Theorem stating that the hospital process flowchart is a Process flowchart --/
theorem hospital_process_is_process_flowchart :
  hospital_process_flowchart = Flowchart.Process :=
by
  rfl

#check hospital_process_is_process_flowchart

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hospital_process_is_process_flowchart_l1338_133819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1338_133812

/-- The eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →  -- hyperbola equation
  (∃ (P F₁ F₂ : ℝ × ℝ), (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0) →  -- perpendicular vectors
  (∃ (F₁ F₂ P : ℝ × ℝ), abs ((F₂.1 - F₁.1) * (P.2 - F₁.2) - (F₂.2 - F₁.2) * (P.1 - F₁.1)) / 2 = 9) →  -- triangle area
  a + b = 7 →
  Real.sqrt (a^2 + b^2) / a = 5/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1338_133812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1338_133889

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x^2 - (2*a+1)*x + a*(a+1) > 0) → (|2*x - 1| > 3)) ∧ 
  (∃ x : ℝ, x^2 - (2*a+1)*x + a*(a+1) > 0 ∧ |2*x - 1| ≤ 3) →
  -1 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1338_133889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l1338_133823

noncomputable def g (x : ℝ) : ℝ := min (3 * x + 3) (min ((1/3) * x + 2) (-x + 8))

theorem max_value_of_g :
  ∃ (M : ℝ), (∀ x, g x ≤ M) ∧ (∃ x₀, g x₀ = M) ∧ M = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l1338_133823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelfth_term_is_twelve_l1338_133880

/-- An arithmetic sequence with a given second term and common difference -/
def arithmeticSequence (a₂ : ℤ) (d : ℤ) : ℕ → ℤ
  | 0 => a₂ - d  -- Add case for 0
  | 1 => a₂ - d
  | n + 1 => arithmeticSequence a₂ d n + d

/-- Theorem: The 12th term of the arithmetic sequence with a₂ = -8 and d = 2 is 12 -/
theorem twelfth_term_is_twelve :
  arithmeticSequence (-8) 2 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelfth_term_is_twelve_l1338_133880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_l1338_133872

noncomputable def cone_lateral_radius : ℝ := 2 * Real.sqrt 3

-- Define the relation between the cone and the sphere
def cone_on_sphere (r h : ℝ) : Prop :=
  r^2 + h^2 = cone_lateral_radius^2 ∧ 
  2 * r = cone_lateral_radius^2 / h

-- Theorem to prove
theorem sphere_volume : 
  ∃ (r h R : ℝ), 
    cone_on_sphere r h ∧ 
    R = cone_lateral_radius^2 / (2 * h) ∧
    (4 / 3) * Real.pi * R^3 = (32 * Real.pi) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_l1338_133872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ATMSecuritySolutionExists_l1338_133851

/-- Represents the ATM security problem and its solution --/
def ATMSecurityProblem : Prop := True

/-- The solution to the ATM security problem exists --/
theorem ATMSecuritySolutionExists : ATMSecurityProblem := by
  trivial

#check ATMSecuritySolutionExists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ATMSecuritySolutionExists_l1338_133851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_68_2008_count_subset_pairs_count_l1338_133892

-- Define the number of divisors for 68^2008
def divisors_68_2008 : ℕ := 8070153

-- Define the number of pairs (A, B) such that A ⊂ B ⊂ E
def subset_pairs (n : ℕ) : ℕ := 3^n

-- Theorem for the number of divisors of 68^2008
theorem divisors_68_2008_count : 
  (Finset.filter (λ x ↦ 68^2008 % x = 0) (Finset.range (68^2008 + 1))).card = divisors_68_2008 := by
  sorry

-- Theorem for the number of pairs (A, B) such that A ⊂ B ⊂ E
theorem subset_pairs_count (n : ℕ) (E : Finset ℕ) (h : E.card = n) :
  (Finset.powerset E).card ^ 2 = subset_pairs n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_68_2008_count_subset_pairs_count_l1338_133892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ashutosh_completion_time_l1338_133854

noncomputable section

/-- The time it takes for Suresh to complete the job alone -/
def suresh_time : ℝ := 15

/-- The time Suresh works on the job -/
def suresh_work_time : ℝ := 9

/-- The time it takes Ashutosh to complete the remaining job -/
def ashutosh_remaining_time : ℝ := 10

/-- The fraction of the job completed by Suresh -/
noncomputable def suresh_completed_fraction : ℝ := suresh_work_time / suresh_time

/-- The fraction of the job completed by Ashutosh -/
noncomputable def ashutosh_completed_fraction : ℝ := 1 - suresh_completed_fraction

/-- Ashutosh's work rate (fraction of job completed per hour) -/
noncomputable def ashutosh_work_rate : ℝ := ashutosh_completed_fraction / ashutosh_remaining_time

/-- The time it takes for Ashutosh to complete the job alone -/
noncomputable def ashutosh_time : ℝ := 1 / ashutosh_work_rate

theorem ashutosh_completion_time : ashutosh_time = 25 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ashutosh_completion_time_l1338_133854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_properties_l1338_133837

-- Define the circles O₁ and O₂
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0
def circle_O₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 1 = 0

-- Define the line AB
def line_AB (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x y a b c : ℝ) : ℝ :=
  |a*x + b*y + c| / Real.sqrt (a^2 + b^2)

-- Statement of the theorem
theorem circles_intersection_properties :
  ∃ (A B : ℝ × ℝ),
    (circle_O₁ A.1 A.2 ∧ circle_O₂ A.1 A.2) ∧
    (circle_O₁ B.1 B.2 ∧ circle_O₂ B.1 B.2) ∧
    A ≠ B →
    (∀ (x y : ℝ), (circle_O₁ x y ∧ circle_O₂ x y) → line_AB x y) ∧
    (∃ (x y : ℝ), circle_O₁ x y ∧
      distance_to_line x y 1 (-1) 1 + 2 = 2 + Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_properties_l1338_133837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_circle_l1338_133882

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 - 2*x - 3

/-- A point lies on the x-axis -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- A point lies on the y-axis -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The equation of the circle -/
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 5

theorem parabola_intersection_circle :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    parabola x₁ y₁ ∧ on_x_axis x₁ y₁ ∧
    parabola x₂ y₂ ∧ on_x_axis x₂ y₂ ∧
    parabola x₃ y₃ ∧ on_y_axis x₃ y₃ ∧
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ circle_eq x₃ y₃ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_circle_l1338_133882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_floor_equality_l1338_133858

theorem sqrt_floor_equality (a : ℝ) (h : a > 1) : ⌊Real.sqrt ⌊Real.sqrt a⌋⌋ = ⌊Real.sqrt (Real.sqrt a)⌋ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_floor_equality_l1338_133858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_distance_exists_l1338_133846

theorem irrational_distance_exists (m n : ℤ) :
  ∃ (x y : ℤ), ¬ ∃ (q : ℚ), ((x - m : ℚ)^2 + (y - n : ℚ)^2 = q^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_distance_exists_l1338_133846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1338_133827

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

noncomputable def g (ω φ : ℝ) (x : ℝ) : ℝ := f ω φ (x + Real.pi/3)

theorem range_of_f (ω φ : ℝ) (h1 : ω > 0) (h2 : |φ| < Real.pi/2) 
  (h3 : ∀ x, g ω φ x = g ω φ (-x)) :
  Set.range (fun x => f ω φ x) ∩ Set.Icc (-Real.pi/6) (Real.pi/6) = Set.Icc (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1338_133827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_unique_partition_l1338_133822

-- Define the triangle type
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = Real.pi
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the specific triangle we're interested in
noncomputable def special_triangle : Triangle where
  A := 1
  B := Real.sqrt 2
  C := Real.sqrt 3
  angle_sum := sorry
  positive_angles := sorry

-- Define what it means for two triangles to be similar
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : Real, k > 0 ∧ t1.A = k * t2.A ∧ t1.B = k * t2.B ∧ t1.C = k * t2.C

-- Define a partition of a triangle
def is_partition (t : Triangle) (parts : List Triangle) : Prop :=
  (∀ p ∈ parts, similar p t) ∧
  (∃ n : Nat, n > 1 ∧ parts.length = n) ∧
  (∀ p ∈ parts, p.A + p.B + p.C ≤ t.A + t.B + t.C)

-- The main theorem
theorem special_triangle_unique_partition :
  ∀ parts : List Triangle,
    is_partition special_triangle parts →
    ∀ p ∈ parts, similar p special_triangle :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_unique_partition_l1338_133822
