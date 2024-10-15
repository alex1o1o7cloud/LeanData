import Mathlib

namespace NUMINAMATH_CALUDE_tangent_y_intercept_l599_59961

/-- The curve function f(x) = x^2 + 11 -/
def f (x : ℝ) : ℝ := x^2 + 11

/-- The point P on the curve -/
def P : ℝ × ℝ := (1, 12)

/-- The slope of the tangent line at P -/
def m : ℝ := 2 * P.1

/-- The y-intercept of the tangent line -/
def b : ℝ := P.2 - m * P.1

theorem tangent_y_intercept :
  b = 10 := by sorry

end NUMINAMATH_CALUDE_tangent_y_intercept_l599_59961


namespace NUMINAMATH_CALUDE_expected_socks_theorem_l599_59982

/-- The expected number of socks taken until a pair is found -/
def expected_socks (n : ℕ) : ℝ := 2 * n

/-- Theorem: The expected number of socks taken until a pair is found is 2n -/
theorem expected_socks_theorem (n : ℕ) : expected_socks n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_expected_socks_theorem_l599_59982


namespace NUMINAMATH_CALUDE_probability_prime_or_square_l599_59962

/-- A function that returns true if a number is prime --/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns true if a number is a perfect square --/
def isPerfectSquare (n : ℕ) : Prop := sorry

/-- The number of sides on each die --/
def numSides : ℕ := 8

/-- The set of possible outcomes when rolling two dice --/
def outcomes : Finset (ℕ × ℕ) := sorry

/-- The set of favorable outcomes (sum is prime or perfect square) --/
def favorableOutcomes : Finset (ℕ × ℕ) := sorry

/-- Theorem stating the probability of getting a sum that is either prime or a perfect square --/
theorem probability_prime_or_square :
  (Finset.card favorableOutcomes : ℚ) / (Finset.card outcomes : ℚ) = 35 / 64 := by sorry

end NUMINAMATH_CALUDE_probability_prime_or_square_l599_59962


namespace NUMINAMATH_CALUDE_subset_condition_intersection_empty_condition_l599_59974

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | (x + 2*m) * (x - m + 4) < 0}
def B : Set ℝ := {x | (1 - x) / (x + 2) > 0}

-- Statement for the first part of the problem
theorem subset_condition (m : ℝ) : 
  B ⊆ A m ↔ m ≥ 5 ∨ m ≤ -1/2 := by sorry

-- Statement for the second part of the problem
theorem intersection_empty_condition (m : ℝ) :
  A m ∩ B = ∅ ↔ 1 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_subset_condition_intersection_empty_condition_l599_59974


namespace NUMINAMATH_CALUDE_ab_minus_one_lt_a_minus_b_l599_59934

theorem ab_minus_one_lt_a_minus_b (a b : ℝ) (ha : a > 0) (hb : b < 1) :
  a * b - 1 < a - b := by
  sorry

end NUMINAMATH_CALUDE_ab_minus_one_lt_a_minus_b_l599_59934


namespace NUMINAMATH_CALUDE_second_quadrant_condition_l599_59937

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m - 2) (m + 1)

-- Define what it means for a complex number to be in the second quadrant
def in_second_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im > 0

-- State the theorem
theorem second_quadrant_condition (m : ℝ) : 
  in_second_quadrant (z m) ↔ -1 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_second_quadrant_condition_l599_59937


namespace NUMINAMATH_CALUDE_constant_term_expansion_l599_59935

theorem constant_term_expansion (b : ℝ) (h : b = -1/2) :
  let c := 6 * b^2
  c = 3/2 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l599_59935


namespace NUMINAMATH_CALUDE_either_shooter_hits_probability_l599_59969

-- Define the probabilities for shooters A and B
def prob_A_hits : ℝ := 0.9
def prob_B_hits : ℝ := 0.8

-- Define the probability that either A or B hits the target
def prob_either_hits : ℝ := 1 - (1 - prob_A_hits) * (1 - prob_B_hits)

-- Theorem statement
theorem either_shooter_hits_probability :
  prob_either_hits = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_either_shooter_hits_probability_l599_59969


namespace NUMINAMATH_CALUDE_bacon_vs_mashed_potatoes_l599_59919

theorem bacon_vs_mashed_potatoes (mashed_potatoes bacon : ℕ) 
  (h1 : mashed_potatoes = 479) 
  (h2 : bacon = 489) : 
  bacon - mashed_potatoes = 10 := by
  sorry

end NUMINAMATH_CALUDE_bacon_vs_mashed_potatoes_l599_59919


namespace NUMINAMATH_CALUDE_ratio_change_after_subtraction_l599_59966

theorem ratio_change_after_subtraction (a b : ℕ) (h1 : a * 5 = b * 6) (h2 : a > 5 ∧ b > 5) 
  (h3 : (a - 5) - (b - 5) = 5) : (a - 5) * 4 = (b - 5) * 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_change_after_subtraction_l599_59966


namespace NUMINAMATH_CALUDE_manuscript_productivity_l599_59979

/-- Given a manuscript with 60,000 words, written over 120 hours including 20 hours of breaks,
    the average productivity during actual writing time is 600 words per hour. -/
theorem manuscript_productivity (total_words : ℕ) (total_hours : ℕ) (break_hours : ℕ)
    (h1 : total_words = 60000)
    (h2 : total_hours = 120)
    (h3 : break_hours = 20) :
  (total_words : ℝ) / (total_hours - break_hours : ℝ) = 600 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_productivity_l599_59979


namespace NUMINAMATH_CALUDE_solve_matrix_inverse_l599_59946

def matrix_inverse_problem (c d x y : ℚ) : Prop :=
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![4, c; x, 13]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![13, y; 3, d]
  (A * B = 1) → (x = -3 ∧ y = 17/4 ∧ c + d = -16)

theorem solve_matrix_inverse :
  ∃ c d x y : ℚ, matrix_inverse_problem c d x y :=
sorry

end NUMINAMATH_CALUDE_solve_matrix_inverse_l599_59946


namespace NUMINAMATH_CALUDE_correct_subtraction_l599_59967

/-- Given a two-digit number XY and another number Z, prove that the correct subtraction result is 49 -/
theorem correct_subtraction (X Y Z : ℕ) : 
  X = 2 → 
  Y = 4 → 
  Z - 59 = 14 → 
  Z - (10 * X + Y) = 49 := by
sorry

end NUMINAMATH_CALUDE_correct_subtraction_l599_59967


namespace NUMINAMATH_CALUDE_print_height_preservation_l599_59952

/-- Given a painting and its print with preserved aspect ratio, calculate the height of the print -/
theorem print_height_preservation (original_width original_height print_width : ℝ) 
  (hw : original_width = 15) 
  (hh : original_height = 10) 
  (pw : print_width = 37.5) :
  let print_height := (print_width * original_height) / original_width
  print_height = 25 := by
  sorry

end NUMINAMATH_CALUDE_print_height_preservation_l599_59952


namespace NUMINAMATH_CALUDE_grocery_store_soda_l599_59902

theorem grocery_store_soda (diet_soda apples : ℕ) 
  (h1 : diet_soda = 32)
  (h2 : apples = 78)
  (h3 : ∃ regular_soda : ℕ, regular_soda + diet_soda = apples + 26) :
  ∃ regular_soda : ℕ, regular_soda = 72 := by
sorry

end NUMINAMATH_CALUDE_grocery_store_soda_l599_59902


namespace NUMINAMATH_CALUDE_systematic_sampling_l599_59933

/-- Systematic sampling for a given population and sample size -/
theorem systematic_sampling
  (population : ℕ)
  (sample_size : ℕ)
  (h_pop : population = 1650)
  (h_sample : sample_size = 35)
  : ∃ (removed : ℕ) (segments : ℕ),
    removed = 5 ∧
    segments = 35 ∧
    (population - removed) % segments = 0 ∧
    (population - removed) / segments = sample_size :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l599_59933


namespace NUMINAMATH_CALUDE_power_mod_five_l599_59909

theorem power_mod_five : 2^345 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_five_l599_59909


namespace NUMINAMATH_CALUDE_candy_distribution_l599_59996

theorem candy_distribution (n : Nat) (f : Nat) (h1 : n = 30) (h2 : f = 4) :
  (∃ x : Nat, (n - x) % f = 0 ∧ ∀ y : Nat, y < x → (n - y) % f ≠ 0) →
  (∃ x : Nat, (n - x) % f = 0 ∧ ∀ y : Nat, y < x → (n - y) % f ≠ 0 ∧ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l599_59996


namespace NUMINAMATH_CALUDE_negation_equivalence_l599_59918

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  all_positive : ∀ i, angles i > 0

-- Define what it means for an angle to be obtuse
def is_obtuse (angle : ℝ) : Prop := angle > 90

-- Define the proposition "A triangle has at most one obtuse angle"
def at_most_one_obtuse (t : Triangle) : Prop :=
  (is_obtuse (t.angles 0) → ¬is_obtuse (t.angles 1) ∧ ¬is_obtuse (t.angles 2)) ∧
  (is_obtuse (t.angles 1) → ¬is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 2)) ∧
  (is_obtuse (t.angles 2) → ¬is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 1))

-- Define the negation of the proposition
def negation_at_most_one_obtuse (t : Triangle) : Prop :=
  ¬(at_most_one_obtuse t)

-- Define the condition "There are at least two obtuse angles in the triangle"
def at_least_two_obtuse (t : Triangle) : Prop :=
  (is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 1)) ∨
  (is_obtuse (t.angles 1) ∧ is_obtuse (t.angles 2)) ∨
  (is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 2))

-- Theorem: The negation of "at most one obtuse angle" is equivalent to "at least two obtuse angles"
theorem negation_equivalence (t : Triangle) :
  negation_at_most_one_obtuse t ↔ at_least_two_obtuse t :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l599_59918


namespace NUMINAMATH_CALUDE_hyperbola_y_axis_condition_l599_59923

/-- Represents a conic section of the form mx^2 + ny^2 = 1 -/
structure Conic (m n : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1

/-- Predicate to check if a conic is a hyperbola with foci on the y-axis -/
def is_hyperbola_y_axis (m n : ℝ) : Prop :=
  m < 0 ∧ n > 0

theorem hyperbola_y_axis_condition (m n : ℝ) :
  (∃ (c : Conic m n), is_hyperbola_y_axis m n) → m * n < 0 ∧
  ∃ (m' n' : ℝ), m' * n' < 0 ∧ ¬∃ (c : Conic m' n'), is_hyperbola_y_axis m' n' :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_y_axis_condition_l599_59923


namespace NUMINAMATH_CALUDE_equilateral_triangle_symmetry_l599_59980

-- Define the shape types
inductive Shape
  | Rectangle
  | Rhombus
  | EquilateralTriangle
  | Circle

-- Define symmetry properties
def hasAxisSymmetry (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle          => true
  | Shape.Rhombus            => true
  | Shape.EquilateralTriangle => true
  | Shape.Circle             => true

def hasCenterSymmetry (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle          => true
  | Shape.Rhombus            => true
  | Shape.EquilateralTriangle => false
  | Shape.Circle             => true

-- Theorem statement
theorem equilateral_triangle_symmetry :
  ∃ (s : Shape), hasAxisSymmetry s ∧ ¬hasCenterSymmetry s ∧
  (∀ (t : Shape), t ≠ s → (hasAxisSymmetry t → hasCenterSymmetry t)) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_symmetry_l599_59980


namespace NUMINAMATH_CALUDE_relationship_increases_with_ratio_difference_l599_59948

-- Define the structure for a 2x2 contingency table
structure ContingencyTable :=
  (a b c d : ℕ)

-- Define the ratios
def ratio1 (t : ContingencyTable) : ℚ := t.a / (t.a + t.b)
def ratio2 (t : ContingencyTable) : ℚ := t.c / (t.c + t.d)

-- Define the difference between ratios
def ratioDifference (t : ContingencyTable) : ℚ := |ratio1 t - ratio2 t|

-- Define a measure of relationship possibility (e.g., chi-square value)
noncomputable def relationshipPossibility (t : ContingencyTable) : ℝ := sorry

-- State the theorem
theorem relationship_increases_with_ratio_difference (t : ContingencyTable) :
  ∀ (t1 t2 : ContingencyTable),
    ratioDifference t1 < ratioDifference t2 →
    relationshipPossibility t1 < relationshipPossibility t2 :=
sorry

end NUMINAMATH_CALUDE_relationship_increases_with_ratio_difference_l599_59948


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_specific_l599_59983

def arithmetic_series_sum (a₁ aₙ d : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_series_sum_specific :
  arithmetic_series_sum 12 50 (1/10) = 11811 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_specific_l599_59983


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_three_l599_59981

def is_divisible_by_three (n : ℕ) : Prop :=
  ∃ k : ℤ, 9 * (n - 1)^3 - 3 * n^3 + 19 * n + 27 = 3 * k

theorem largest_n_divisible_by_three :
  (∀ m : ℕ, m < 50000 → is_divisible_by_three m → m ≤ 49998) ∧
  (49998 < 50000) ∧
  is_divisible_by_three 49998 :=
sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_three_l599_59981


namespace NUMINAMATH_CALUDE_f_increasing_implies_a_range_l599_59938

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (|x - a|)

theorem f_increasing_implies_a_range (a : ℝ) :
  (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x ≤ y → f a x ≤ f a y) →
  a ∈ Set.Iic 1 :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_implies_a_range_l599_59938


namespace NUMINAMATH_CALUDE_jade_savings_l599_59908

/-- Calculates Jade's monthly savings based on her income and expenses --/
def calculate_savings (
  monthly_income : ℝ)
  (contribution_401k_rate : ℝ)
  (tax_deduction_rate : ℝ)
  (living_expenses_rate : ℝ)
  (insurance_rate : ℝ)
  (transportation_rate : ℝ)
  (utilities_rate : ℝ) : ℝ :=
  let contribution_401k := monthly_income * contribution_401k_rate
  let tax_deduction := monthly_income * tax_deduction_rate
  let post_deduction_income := monthly_income - contribution_401k - tax_deduction
  let total_expenses := post_deduction_income * (living_expenses_rate + insurance_rate + transportation_rate + utilities_rate)
  post_deduction_income - total_expenses

/-- Theorem stating Jade's monthly savings --/
theorem jade_savings :
  calculate_savings 2800 0.08 0.10 0.55 0.20 0.12 0.08 = 114.80 := by
  sorry


end NUMINAMATH_CALUDE_jade_savings_l599_59908


namespace NUMINAMATH_CALUDE_line_point_a_value_l599_59916

theorem line_point_a_value (k : ℝ) (a : ℝ) :
  k = 0.75 →
  5 = k * a + 1 →
  a = 16/3 := by sorry

end NUMINAMATH_CALUDE_line_point_a_value_l599_59916


namespace NUMINAMATH_CALUDE_sum_g_equals_negative_one_l599_59964

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the conditions
axiom functional_equation : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y
axiom f_equality : f (-2) = f 1
axiom f_nonzero : f 1 ≠ 0

-- State the theorem to be proved
theorem sum_g_equals_negative_one : g 1 + g (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_sum_g_equals_negative_one_l599_59964


namespace NUMINAMATH_CALUDE_range_of_a_l599_59928

-- Define the propositions p and q
def p (x : ℝ) : Prop := (2*x - 1) / (x - 1) ≤ 0

def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) < 0

-- Define the set of x that satisfy p
def P : Set ℝ := {x | p x}

-- Define the set of x that satisfy q
def Q (a : ℝ) : Set ℝ := {x | q x a}

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, (P ⊆ Q a ∧ ¬(Q a ⊆ P))) → 
  {a : ℝ | 0 ≤ a ∧ a < 1/2} = {a : ℝ | ∃ x, q x a} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l599_59928


namespace NUMINAMATH_CALUDE_equivalent_statements_l599_59926

theorem equivalent_statements :
  (∀ x : ℝ, x ≥ 0 → x^2 ≤ 0) ↔ (∀ x : ℝ, x^2 ≤ 0 → x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_statements_l599_59926


namespace NUMINAMATH_CALUDE_crayons_per_child_l599_59929

theorem crayons_per_child (total_children : ℕ) (total_crayons : ℕ) 
  (h1 : total_children = 10) 
  (h2 : total_crayons = 50) : 
  total_crayons / total_children = 5 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_child_l599_59929


namespace NUMINAMATH_CALUDE_sphere_tangency_configurations_l599_59985

-- Define the types for our geometric objects
def Plane : Type := ℝ → ℝ → ℝ → Prop
def Sphere : Type := ℝ × ℝ × ℝ × ℝ  -- (center_x, center_y, center_z, radius)

-- Define the concept of tangency between a sphere and a plane
def spherePlaneTangent (s : Sphere) (p : Plane) : Prop := sorry

-- Define the concept of tangency between two spheres
def sphereSphereTangent (s1 s2 : Sphere) : Prop := sorry

-- Main theorem
theorem sphere_tangency_configurations 
  (p1 p2 p3 : Plane) (s : Sphere) : 
  ∃ (n : ℕ), n ≤ 16 ∧ 
  (∃ (configurations : Finset Sphere), 
    (∀ s' ∈ configurations, 
      spherePlaneTangent s' p1 ∧ 
      spherePlaneTangent s' p2 ∧ 
      spherePlaneTangent s' p3 ∧ 
      sphereSphereTangent s' s) ∧
    configurations.card = n) := by sorry

end NUMINAMATH_CALUDE_sphere_tangency_configurations_l599_59985


namespace NUMINAMATH_CALUDE_average_first_five_primes_gt_50_l599_59906

def first_five_primes_gt_50 : List Nat := [53, 59, 61, 67, 71]

def average (lst : List Nat) : ℚ :=
  (lst.sum : ℚ) / lst.length

theorem average_first_five_primes_gt_50 :
  average first_five_primes_gt_50 = 62.2 := by
  sorry

end NUMINAMATH_CALUDE_average_first_five_primes_gt_50_l599_59906


namespace NUMINAMATH_CALUDE_evaluate_expression_l599_59953

theorem evaluate_expression : 11 + Real.sqrt (-4 + 6 * 4 / 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l599_59953


namespace NUMINAMATH_CALUDE_xavier_yvonne_not_zelda_probability_l599_59925

/-- The probability that Xavier and Yvonne solve a problem but Zelda does not,
    given their individual success probabilities -/
theorem xavier_yvonne_not_zelda_probability 
  (p_xavier : ℚ) (p_yvonne : ℚ) (p_zelda : ℚ)
  (h_xavier : p_xavier = 1/5)
  (h_yvonne : p_yvonne = 1/2)
  (h_zelda : p_zelda = 5/8)
  (h_independent : True) -- Assumption of independence
  : p_xavier * p_yvonne * (1 - p_zelda) = 3/80 := by
  sorry

end NUMINAMATH_CALUDE_xavier_yvonne_not_zelda_probability_l599_59925


namespace NUMINAMATH_CALUDE_decimal_equivalent_of_one_fourth_squared_l599_59987

theorem decimal_equivalent_of_one_fourth_squared :
  (1 / 4 : ℚ) ^ 2 = (0.0625 : ℚ) := by sorry

end NUMINAMATH_CALUDE_decimal_equivalent_of_one_fourth_squared_l599_59987


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l599_59960

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n, prove the common difference is 2. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)  -- Definition of S_n for arithmetic sequence
  (h2 : S 4 = 3 * S 2)  -- Given condition
  (h3 : a 7 = 15)  -- Given condition
  : ∃ d : ℝ, ∀ n, a (n + 1) - a n = d ∧ d = 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l599_59960


namespace NUMINAMATH_CALUDE_max_sphere_in_intersecting_cones_l599_59994

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones -/
structure IntersectingCones where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- The maximum squared radius of a sphere that can fit inside two intersecting cones -/
def maxSphereRadiusSquared (ic : IntersectingCones) : ℝ := sorry

/-- The specific configuration of cones described in the problem -/
def problemCones : IntersectingCones :=
  { cone1 := { baseRadius := 3, height := 8 },
    cone2 := { baseRadius := 3, height := 8 },
    intersectionDistance := 3 }

theorem max_sphere_in_intersecting_cones :
  maxSphereRadiusSquared problemCones = 225 / 73 := by sorry

end NUMINAMATH_CALUDE_max_sphere_in_intersecting_cones_l599_59994


namespace NUMINAMATH_CALUDE_shanghai_score_is_75_l599_59970

/-- The score of the Shanghai team in the basketball game -/
def shanghai_score : ℕ := 75

/-- The score of the Beijing team in the basketball game -/
def beijing_score : ℕ := shanghai_score - 10

/-- Yao Ming's score in the basketball game -/
def yao_ming_score : ℕ := 30

theorem shanghai_score_is_75 :
  (shanghai_score - beijing_score = 10) ∧
  (shanghai_score + beijing_score = 5 * yao_ming_score - 10) →
  shanghai_score = 75 := by
sorry

end NUMINAMATH_CALUDE_shanghai_score_is_75_l599_59970


namespace NUMINAMATH_CALUDE_line_symmetry_l599_59941

/-- Given two lines l₁ and l₂ in the xy-plane, prove that if the angle bisector between them
    is y = x, and l₁ has the equation x + 2y + 3 = 0, then l₂ has the equation 2x + y + 3 = 0. -/
theorem line_symmetry (l₁ l₂ : Set (ℝ × ℝ)) : 
  (∀ p : ℝ × ℝ, p ∈ l₁ ↔ p.1 + 2 * p.2 + 3 = 0) →
  (∀ p : ℝ × ℝ, p ∈ l₂ ↔ 2 * p.1 + p.2 + 3 = 0) →
  (∀ p : ℝ × ℝ, p ∈ l₁ ∨ p ∈ l₂ → p.1 = p.2 → 
    ∃ q : ℝ × ℝ, (q ∈ l₁ ∧ q.1 + q.2 = p.1 + p.2) ∨ (q ∈ l₂ ∧ q.1 + q.2 = p.1 + p.2)) :=
by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_l599_59941


namespace NUMINAMATH_CALUDE_power_relation_l599_59991

theorem power_relation (a m n : ℝ) (h1 : a^m = 8) (h2 : a^n = 4) : a^(m-2*n) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l599_59991


namespace NUMINAMATH_CALUDE_soccer_lineup_count_l599_59945

theorem soccer_lineup_count (n : ℕ) (h : n = 18) : 
  n * (n - 1) * (Nat.choose (n - 2) 9) = 3501120 :=
by sorry

end NUMINAMATH_CALUDE_soccer_lineup_count_l599_59945


namespace NUMINAMATH_CALUDE_ab_neg_necessary_not_sufficient_for_hyperbola_l599_59975

-- Define the condition for a hyperbola
def is_hyperbola (a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), a * x^2 + b * y^2 = c ∧ c ≠ 0 ∧ a * b < 0

-- State the theorem
theorem ab_neg_necessary_not_sufficient_for_hyperbola :
  (∀ a b c : ℝ, is_hyperbola a b c → a * b < 0) ∧
  (∃ a b c : ℝ, a * b < 0 ∧ ¬(is_hyperbola a b c)) :=
sorry

end NUMINAMATH_CALUDE_ab_neg_necessary_not_sufficient_for_hyperbola_l599_59975


namespace NUMINAMATH_CALUDE_bodyguard_hourly_rate_l599_59903

/-- Proves that the hourly rate for each bodyguard is $20 -/
theorem bodyguard_hourly_rate :
  let num_bodyguards : ℕ := 2
  let hours_per_day : ℕ := 8
  let days_per_week : ℕ := 7
  let total_weekly_payment : ℕ := 2240
  (num_bodyguards * hours_per_day * days_per_week * hourly_rate = total_weekly_payment) →
  hourly_rate = 20 := by
  sorry

end NUMINAMATH_CALUDE_bodyguard_hourly_rate_l599_59903


namespace NUMINAMATH_CALUDE_intersection_equality_subset_relation_l599_59958

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x | -1-2*a ≤ x ∧ x ≤ a-2}

-- Theorem for part (1)
theorem intersection_equality (a : ℝ) : A ∩ B a = A ↔ a ≥ 7 := by sorry

-- Theorem for part (2)
theorem subset_relation (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) ↔ a < 1/3 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_subset_relation_l599_59958


namespace NUMINAMATH_CALUDE_local_extremum_and_inequality_l599_59977

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

-- State the theorem
theorem local_extremum_and_inequality (a b : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (-1 - δ) (-1 + δ), f a b x ≥ f a b (-1)) ∧
  (f a b (-1) = 0) ∧
  (∀ x ∈ Set.Icc (-2) 1, f a b x ≤ 20) →
  a = 2 ∧ b = 9 ∧ (∀ m : ℝ, (∀ x ∈ Set.Icc (-2) 1, f a b x ≤ m) ↔ m ≥ 20) :=
by sorry

end NUMINAMATH_CALUDE_local_extremum_and_inequality_l599_59977


namespace NUMINAMATH_CALUDE_segment_ratios_l599_59950

/-- Given four points P, Q, R, S on a line in that order, with given distances between them,
    prove the ratios of certain segments. -/
theorem segment_ratios (P Q R S : ℝ) (h_order : P < Q ∧ Q < R ∧ R < S)
    (h_PQ : Q - P = 3) (h_QR : R - Q = 7) (h_PS : S - P = 20) :
    (R - P) / (S - Q) = 10 / 17 ∧ (S - P) / (Q - P) = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_segment_ratios_l599_59950


namespace NUMINAMATH_CALUDE_clothing_profit_l599_59988

theorem clothing_profit (price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) : 
  price = 180 ∧ 
  profit_percent = 20 ∧ 
  loss_percent = 10 → 
  (2 * price) - (price / (1 + profit_percent / 100) + price / (1 - loss_percent / 100)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_clothing_profit_l599_59988


namespace NUMINAMATH_CALUDE_area_of_right_triangle_l599_59940

-- Define the right triangle ABC
structure RightTriangle where
  AB : ℝ
  BD : ℝ
  isRightTriangle : AB > 0 ∧ BD > 0

-- Define the theorem
theorem area_of_right_triangle (t : RightTriangle) 
  (h1 : t.AB = 13) 
  (h2 : t.BD = 12) : 
  (1 / 2 : ℝ) * t.AB * t.BD = 202.8 := by
  sorry


end NUMINAMATH_CALUDE_area_of_right_triangle_l599_59940


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_squared_l599_59911

theorem x_squared_plus_reciprocal_squared (x : ℝ) (h : x + 1/x = 8) : x^2 + 1/x^2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_squared_l599_59911


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l599_59993

theorem partial_fraction_decomposition :
  ∃ (A B C : ℝ),
    (∀ x : ℝ, x ≠ 2 ∧ x ≠ 4 →
      (5 * x + 2) / ((x - 2) * (x - 4)^2) =
      A / (x - 2) + B / (x - 4) + C / (x - 4)^2) ∧
    A = 3 ∧ B = -3 ∧ C = 11 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l599_59993


namespace NUMINAMATH_CALUDE_f_sum_inequality_l599_59998

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else 2^x

theorem f_sum_inequality (x : ℝ) :
  (f x + f (x - 1/2) > 1) ↔ x > -1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_inequality_l599_59998


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l599_59943

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles A, B, and C in degrees

-- Define the condition for option D
def angle_ratio (t : Triangle) : Prop :=
  ∃ (k : Real), t.A = 3 * k ∧ t.B = 4 * k ∧ t.C = 5 * k

-- Theorem: If the angles of a triangle satisfy the given ratio, it's not necessarily a right triangle
theorem not_necessarily_right_triangle (t : Triangle) : 
  angle_ratio t → ¬ (t.A = 90 ∨ t.B = 90 ∨ t.C = 90) :=
by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l599_59943


namespace NUMINAMATH_CALUDE_profit_share_difference_l599_59990

/-- Given investments and B's profit share, calculate the difference between A's and C's profit shares -/
theorem profit_share_difference (a b c b_profit : ℕ) 
  (h1 : a = 8000) 
  (h2 : b = 10000) 
  (h3 : c = 12000) 
  (h4 : b_profit = 1700) : 
  (c * b_profit / b) - (a * b_profit / b) = 680 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_difference_l599_59990


namespace NUMINAMATH_CALUDE_average_headcount_proof_l599_59914

def fall_headcount_03_04 : ℕ := 11500
def fall_headcount_04_05 : ℕ := 11600
def fall_headcount_05_06 : ℕ := 11300

def average_headcount : ℕ := 
  (fall_headcount_03_04 + fall_headcount_04_05 + fall_headcount_05_06 + 1) / 3

theorem average_headcount_proof :
  average_headcount = 11467 := by sorry

end NUMINAMATH_CALUDE_average_headcount_proof_l599_59914


namespace NUMINAMATH_CALUDE_problem_solution_l599_59924

theorem problem_solution (x y : ℝ) 
  (h1 : x = 51) 
  (h2 : x^3*y - 2*x^2*y + x*y = 127500) : 
  y = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l599_59924


namespace NUMINAMATH_CALUDE_range_of_f_l599_59913

def f (x : ℤ) : ℤ := (x - 1)^2 - 1

def domain : Set ℤ := {-1, 0, 1, 2, 3}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l599_59913


namespace NUMINAMATH_CALUDE_ellipse_equation_from_parameters_l599_59927

/-- Represents an ellipse with axes aligned to the coordinate system -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  e : ℝ  -- Eccentricity
  h : 0 < a ∧ 0 < b ∧ 0 ≤ e ∧ e < 1  -- Constraints on a, b, and e

/-- The equation of an ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

theorem ellipse_equation_from_parameters :
  ∀ E : Ellipse,
    E.e = 2/3 →
    E.b = 4 * Real.sqrt 5 →
    (∀ x y : ℝ, ellipse_equation E x y ↔ x^2/144 + y^2/80 = 1) ∨
    (∀ x y : ℝ, ellipse_equation E x y ↔ y^2/144 + x^2/80 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_parameters_l599_59927


namespace NUMINAMATH_CALUDE_special_polyhedron_property_l599_59999

-- Define the structure of our polyhedron
structure Polyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  T : ℕ  -- number of triangular faces
  P : ℕ  -- number of square faces

-- Define the properties of our specific polyhedron
def SpecialPolyhedron (poly : Polyhedron) : Prop :=
  poly.V - poly.E + poly.F = 2 ∧  -- Euler's formula
  poly.F = 40 ∧                   -- Total number of faces
  poly.T + poly.P = poly.F ∧      -- Faces are either triangles or squares
  poly.T = 1 ∧                    -- Number of triangular faces at a vertex
  poly.P = 3 ∧                    -- Number of square faces at a vertex
  poly.E = (3 * poly.T + 4 * poly.P) / 2  -- Edge calculation

-- Theorem statement
theorem special_polyhedron_property (poly : Polyhedron) 
  (h : SpecialPolyhedron poly) : 
  100 * poly.P + 10 * poly.T + poly.V = 351 := by
  sorry

end NUMINAMATH_CALUDE_special_polyhedron_property_l599_59999


namespace NUMINAMATH_CALUDE_sam_read_100_pages_l599_59917

def minimum_assigned : ℕ := 25

def harrison_pages (minimum : ℕ) : ℕ := minimum + 10

def pam_pages (harrison : ℕ) : ℕ := harrison + 15

def sam_pages (pam : ℕ) : ℕ := 2 * pam

theorem sam_read_100_pages :
  sam_pages (pam_pages (harrison_pages minimum_assigned)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_sam_read_100_pages_l599_59917


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_6_to_1993_l599_59949

theorem rightmost_three_digits_of_6_to_1993 :
  6^1993 ≡ 296 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_6_to_1993_l599_59949


namespace NUMINAMATH_CALUDE_ratio_tr_ur_l599_59921

-- Define the square PQRS
def Square (P Q R S : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  let (sx, sy) := S
  (qx - px)^2 + (qy - py)^2 = 4 ∧
  (rx - qx)^2 + (ry - qy)^2 = 4 ∧
  (sx - rx)^2 + (sy - ry)^2 = 4 ∧
  (px - sx)^2 + (py - sy)^2 = 4

-- Define the quarter circle QS
def QuarterCircle (Q S : ℝ × ℝ) : Prop :=
  let (qx, qy) := Q
  let (sx, sy) := S
  (sx - qx)^2 + (sy - qy)^2 = 4

-- Define U as the midpoint of QR
def Midpoint (U Q R : ℝ × ℝ) : Prop :=
  let (ux, uy) := U
  let (qx, qy) := Q
  let (rx, ry) := R
  ux = (qx + rx) / 2 ∧ uy = (qy + ry) / 2

-- Define T lying on SR
def PointOnLine (T S R : ℝ × ℝ) : Prop :=
  let (tx, ty) := T
  let (sx, sy) := S
  let (rx, ry) := R
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ tx = sx + t * (rx - sx) ∧ ty = sy + t * (ry - sy)

-- Define TU as tangent to the arc QS
def Tangent (T U Q S : ℝ × ℝ) : Prop :=
  let (tx, ty) := T
  let (ux, uy) := U
  let (qx, qy) := Q
  let (sx, sy) := S
  (tx - ux) * (qy - sy) = (ty - uy) * (qx - sx)

-- Theorem statement
theorem ratio_tr_ur (P Q R S T U : ℝ × ℝ) 
  (h1 : Square P Q R S)
  (h2 : QuarterCircle Q S)
  (h3 : Midpoint U Q R)
  (h4 : PointOnLine T S R)
  (h5 : Tangent T U Q S) :
  let (tx, ty) := T
  let (rx, ry) := R
  let (ux, uy) := U
  (tx - rx)^2 + (ty - ry)^2 = 16/9 * ((ux - rx)^2 + (uy - ry)^2) := by sorry

end NUMINAMATH_CALUDE_ratio_tr_ur_l599_59921


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l599_59910

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  h_p_pos : p > 0

/-- Point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

/-- Circle structure -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem: For a parabola y^2 = 2px (p > 0) with focus F, and a point M(x_0, 2√2) on the parabola,
    if a circle with center M is tangent to the y-axis and intersects MF at A such that |MA| / |AF| = 2,
    then p = 2 -/
theorem parabola_circle_intersection (C : Parabola) (M : PointOnParabola C)
  (circ : Circle) (A : ℝ × ℝ) :
  M.y = 2 * Real.sqrt 2 →
  circ.center = (M.x, M.y) →
  circ.radius = M.x →
  A.1 = M.x - C.p →
  (M.x - A.1) / A.1 = 2 →
  C.p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l599_59910


namespace NUMINAMATH_CALUDE_smallest_b_value_l599_59904

theorem smallest_b_value (a b : ℕ+) 
  (h1 : a.val - b.val = 8)
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 16) :
  ∀ x : ℕ+, x.val < b.val → 
    (∃ y : ℕ+, y.val - x.val = 8 ∧ 
      Nat.gcd ((y.val^3 + x.val^3) / (y.val + x.val)) (y.val * x.val) ≠ 16) ∧
    b.val = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l599_59904


namespace NUMINAMATH_CALUDE_water_poured_out_l599_59956

-- Define the initial and final amounts of water
def initial_amount : ℝ := 0.8
def final_amount : ℝ := 0.6

-- Define the amount of water poured out
def poured_out : ℝ := initial_amount - final_amount

-- Theorem to prove
theorem water_poured_out : poured_out = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_water_poured_out_l599_59956


namespace NUMINAMATH_CALUDE_trapezoid_cd_length_l599_59915

structure Trapezoid (A B C D : ℝ × ℝ) :=
  (parallel : (A.2 - D.2) / (A.1 - D.1) = (B.2 - C.2) / (B.1 - C.1))
  (bd_length : Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 2)
  (angle_dbc : Real.arccos ((B.1 - D.1) * (C.1 - B.1) + (B.2 - D.2) * (C.2 - B.2)) / 
    (Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) * Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)) = 36 * π / 180)
  (angle_bda : Real.arccos ((B.1 - D.1) * (A.1 - D.1) + (B.2 - D.2) * (A.2 - D.2)) / 
    (Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) * Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)) = 72 * π / 180)
  (ratio : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) / Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 5/3)

theorem trapezoid_cd_length (A B C D : ℝ × ℝ) (t : Trapezoid A B C D) :
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_cd_length_l599_59915


namespace NUMINAMATH_CALUDE_fair_attendance_l599_59995

/-- Given the number of people attending a fair over three years, prove the values of x, y, and z. -/
theorem fair_attendance (x y z : ℕ) 
  (h1 : z = 2 * y)
  (h2 : x = z - 200)
  (h3 : y = 600) :
  x = 1000 ∧ y = 600 ∧ z = 1200 := by
  sorry

end NUMINAMATH_CALUDE_fair_attendance_l599_59995


namespace NUMINAMATH_CALUDE_ellipse_sum_theorem_l599_59972

/-- Represents an ellipse with center (h, k) and semi-axes a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The sum of h, k, a, and b for a specific ellipse -/
def ellipse_sum (e : Ellipse) : ℝ :=
  e.h + e.k + e.a + e.b

/-- Theorem: For an ellipse with center (-3, 5), semi-major axis 7, and semi-minor axis 4,
    the sum h + k + a + b equals 13 -/
theorem ellipse_sum_theorem (e : Ellipse)
    (center_h : e.h = -3)
    (center_k : e.k = 5)
    (semi_major : e.a = 7)
    (semi_minor : e.b = 4) :
    ellipse_sum e = 13 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_theorem_l599_59972


namespace NUMINAMATH_CALUDE_bhaskar_tour_days_l599_59932

def total_budget : ℕ := 360
def extension_days : ℕ := 4
def expense_reduction : ℕ := 3

theorem bhaskar_tour_days :
  ∃ (x : ℕ), x > 0 ∧
  (total_budget / x : ℚ) - expense_reduction = (total_budget / (x + extension_days) : ℚ) ∧
  x = 20 := by
  sorry

end NUMINAMATH_CALUDE_bhaskar_tour_days_l599_59932


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l599_59942

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let set_size := 2 * n
  let special_num := 1 + 1 / n
  let regular_num := 1
  let sum := (set_size - 1) * regular_num + special_num
  sum / set_size = 1 + 1 / (2 * n^2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l599_59942


namespace NUMINAMATH_CALUDE_max_intersection_points_l599_59922

theorem max_intersection_points (x_points y_points : ℕ) : x_points = 15 → y_points = 6 → 
  (x_points.choose 2) * (y_points.choose 2) = 1575 := by sorry

end NUMINAMATH_CALUDE_max_intersection_points_l599_59922


namespace NUMINAMATH_CALUDE_inverse_100_mod_101_l599_59978

theorem inverse_100_mod_101 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 100 ∧ (100 * x) % 101 = 1 :=
by
  use 100
  sorry

end NUMINAMATH_CALUDE_inverse_100_mod_101_l599_59978


namespace NUMINAMATH_CALUDE_percentage_calculation_l599_59976

theorem percentage_calculation (N : ℚ) (h : (1/2) * N = 16) : (3/4) * N = 24 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l599_59976


namespace NUMINAMATH_CALUDE_size_relationship_l599_59965

theorem size_relationship (a b c : ℝ) 
  (ha : a = (0.2 : ℝ) ^ (1.5 : ℝ))
  (hb : b = (2 : ℝ) ^ (0.1 : ℝ))
  (hc : c = (0.2 : ℝ) ^ (1.3 : ℝ)) :
  a < c ∧ c < b :=
by sorry

end NUMINAMATH_CALUDE_size_relationship_l599_59965


namespace NUMINAMATH_CALUDE_hamlet_47_impossible_hamlet_41_possible_hamlet_59_possible_hamlet_61_possible_hamlet_66_possible_l599_59912

/-- Represents the total number of animals and people in Hamlet -/
def hamlet_total (h c : ℕ) : ℕ := 13 * h + 5 * c

/-- Theorem stating that 47 cannot be expressed as a hamlet total -/
theorem hamlet_47_impossible : ¬ ∃ (h c : ℕ), hamlet_total h c = 47 := by sorry

/-- Theorem stating that 41 can be expressed as a hamlet total -/
theorem hamlet_41_possible : ∃ (h c : ℕ), hamlet_total h c = 41 := by sorry

/-- Theorem stating that 59 can be expressed as a hamlet total -/
theorem hamlet_59_possible : ∃ (h c : ℕ), hamlet_total h c = 59 := by sorry

/-- Theorem stating that 61 can be expressed as a hamlet total -/
theorem hamlet_61_possible : ∃ (h c : ℕ), hamlet_total h c = 61 := by sorry

/-- Theorem stating that 66 can be expressed as a hamlet total -/
theorem hamlet_66_possible : ∃ (h c : ℕ), hamlet_total h c = 66 := by sorry

end NUMINAMATH_CALUDE_hamlet_47_impossible_hamlet_41_possible_hamlet_59_possible_hamlet_61_possible_hamlet_66_possible_l599_59912


namespace NUMINAMATH_CALUDE_min_a_value_for_quasi_periodic_function_l599_59951

-- Define the a-level quasi-periodic function
def is_a_level_quasi_periodic (f : ℝ → ℝ) (a : ℝ) (D : Set ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x ∈ D, a * f x = f (x + T)

-- Define the function f on [1, 2)
def f_on_initial_interval (x : ℝ) : ℝ := 2 * x + 1

-- Main theorem
theorem min_a_value_for_quasi_periodic_function :
  ∀ f : ℝ → ℝ,
  (∀ a : ℝ, is_a_level_quasi_periodic f a (Set.Ici 1)) →
  (∀ x ∈ Set.Icc 1 2, f x = f_on_initial_interval x) →
  (∀ x y, x < y → x ≥ 1 → f x < f y) →
  (∃ a : ℝ, ∀ b : ℝ, is_a_level_quasi_periodic f b (Set.Ici 1) → a ≤ b) →
  (∀ a : ℝ, is_a_level_quasi_periodic f a (Set.Ici 1) → a ≥ 5/3) :=
by sorry

end NUMINAMATH_CALUDE_min_a_value_for_quasi_periodic_function_l599_59951


namespace NUMINAMATH_CALUDE_total_suitcase_weight_is_434_l599_59971

/-- The total weight of all suitcases for a family vacation --/
def total_suitcase_weight : ℕ :=
  let siblings_suitcases := List.range 6 |>.sum
  let siblings_weight := siblings_suitcases * 10
  let parents_suitcases := 2 * 3
  let parents_weight := parents_suitcases * 12
  let grandparents_suitcases := 2 * 2
  let grandparents_weight := grandparents_suitcases * 8
  let relatives_suitcases := 8
  let relatives_weight := relatives_suitcases * 15
  siblings_weight + parents_weight + grandparents_weight + relatives_weight

theorem total_suitcase_weight_is_434 : total_suitcase_weight = 434 := by
  sorry

end NUMINAMATH_CALUDE_total_suitcase_weight_is_434_l599_59971


namespace NUMINAMATH_CALUDE_c_investment_is_1200_l599_59931

/-- Represents the investment and profit distribution of a business partnership --/
structure BusinessPartnership where
  investmentA : ℕ
  investmentB : ℕ
  investmentC : ℕ
  totalProfit : ℕ
  profitShareC : ℕ

/-- Calculates C's investment amount based on the given conditions --/
def calculateInvestmentC (bp : BusinessPartnership) : ℕ :=
  sorry

/-- Theorem stating that C's investment is 1200 given the specified conditions --/
theorem c_investment_is_1200 : 
  ∀ (bp : BusinessPartnership), 
  bp.investmentA = 800 ∧ 
  bp.investmentB = 1000 ∧ 
  bp.totalProfit = 1000 ∧ 
  bp.profitShareC = 400 →
  calculateInvestmentC bp = 1200 :=
sorry

end NUMINAMATH_CALUDE_c_investment_is_1200_l599_59931


namespace NUMINAMATH_CALUDE_min_value_product_l599_59968

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 3) ≥ 48 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l599_59968


namespace NUMINAMATH_CALUDE_system_solution_l599_59963

theorem system_solution (x y : ℝ) : 
  (x = 1 ∧ y = 4) → 
  (Real.sqrt (y / x) - 2 * Real.sqrt (x / y) = 1 ∧ 
   Real.sqrt (5 * x + y) + Real.sqrt (5 * x - y) = 4) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l599_59963


namespace NUMINAMATH_CALUDE_restaurant_bill_tax_calculation_l599_59900

/-- Calculates the tax amount for a restaurant bill given specific conditions. -/
theorem restaurant_bill_tax_calculation
  (cheeseburger_price : ℚ)
  (milkshake_price : ℚ)
  (coke_price : ℚ)
  (fries_price : ℚ)
  (cookie_price : ℚ)
  (toby_initial_amount : ℚ)
  (toby_change : ℚ)
  (h1 : cheeseburger_price = 365/100)
  (h2 : milkshake_price = 2)
  (h3 : coke_price = 1)
  (h4 : fries_price = 4)
  (h5 : cookie_price = 1/2)
  (h6 : toby_initial_amount = 15)
  (h7 : toby_change = 7) :
  let subtotal := 2 * cheeseburger_price + milkshake_price + coke_price + fries_price + 3 * cookie_price
  let toby_spent := toby_initial_amount - toby_change
  let total_paid := 2 * toby_spent
  let tax := total_paid - subtotal
  tax = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_tax_calculation_l599_59900


namespace NUMINAMATH_CALUDE_freds_allowance_l599_59997

theorem freds_allowance (allowance : ℝ) : 
  allowance / 2 + 6 = 14 → allowance = 16 := by sorry

end NUMINAMATH_CALUDE_freds_allowance_l599_59997


namespace NUMINAMATH_CALUDE_clock_sale_correct_l599_59936

/-- Represents the clock selling scenario --/
structure ClockSale where
  originalCost : ℝ
  collectorPrice : ℝ
  buybackPrice : ℝ
  finalPrice : ℝ

/-- The clock sale scenario satisfying all given conditions --/
def clockScenario : ClockSale :=
  { originalCost := 250,
    collectorPrice := 300,
    buybackPrice := 150,
    finalPrice := 270 }

/-- Theorem stating that the given scenario satisfies all conditions and results in the correct final price --/
theorem clock_sale_correct (c : ClockSale) (h : c = clockScenario) : 
  c.collectorPrice = c.originalCost * 1.2 ∧ 
  c.buybackPrice = c.collectorPrice * 0.5 ∧
  c.originalCost - c.buybackPrice = 100 ∧
  c.finalPrice = c.buybackPrice * 1.8 := by
  sorry

#check clock_sale_correct

end NUMINAMATH_CALUDE_clock_sale_correct_l599_59936


namespace NUMINAMATH_CALUDE_max_sum_digits_24hour_clock_l599_59930

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Fin 24
  minutes : Fin 60

/-- Calculates the sum of digits in a natural number -/
def sumDigits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is even -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Calculates the total sum of digits in a Time24 -/
def totalSumDigits (t : Time24) : ℕ :=
  sumDigits t.hours.val + sumDigits t.minutes.val

/-- The theorem to be proved -/
theorem max_sum_digits_24hour_clock :
  ∃ (t : Time24), 
    (isEven (sumDigits t.hours.val)) ∧ 
    (∀ (t' : Time24), isEven (sumDigits t'.hours.val) → totalSumDigits t' ≤ totalSumDigits t) ∧
    totalSumDigits t = 22 := by sorry

end NUMINAMATH_CALUDE_max_sum_digits_24hour_clock_l599_59930


namespace NUMINAMATH_CALUDE_real_roots_of_x_squared_minus_four_l599_59959

theorem real_roots_of_x_squared_minus_four (x : ℝ) : x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_of_x_squared_minus_four_l599_59959


namespace NUMINAMATH_CALUDE_distribute_10_8_l599_59957

/-- The number of ways to distribute n distinct objects into k distinct groups,
    with each group containing at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Stirling number of the second kind: the number of ways to partition
    a set of n elements into k non-empty subsets -/
def stirling2 (n k : ℕ) : ℕ := sorry

theorem distribute_10_8 :
  distribute 10 8 = 30240000 := by sorry

end NUMINAMATH_CALUDE_distribute_10_8_l599_59957


namespace NUMINAMATH_CALUDE_probability_of_winning_more_than_4000_l599_59984

/-- Represents the number of boxes and keys -/
def num_boxes : ℕ := 3

/-- Represents the total number of ways to assign keys to boxes -/
def total_assignments : ℕ := Nat.factorial num_boxes

/-- Represents the number of ways to correctly assign keys to both the second and third boxes -/
def correct_assignments : ℕ := 1

/-- Theorem stating the probability of correctly assigning keys to both the second and third boxes -/
theorem probability_of_winning_more_than_4000 :
  (correct_assignments : ℚ) / total_assignments = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_winning_more_than_4000_l599_59984


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l599_59920

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = |x - 5| :=
by
  -- The unique solution is x = 4
  use 4
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l599_59920


namespace NUMINAMATH_CALUDE_sum_m_n_equals_51_l599_59947

/-- A function that returns the number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The smallest positive integer with only two positive divisors -/
def m : ℕ := sorry

/-- The largest integer less than 50 with exactly three positive divisors -/
def n : ℕ := sorry

theorem sum_m_n_equals_51 : m + n = 51 := by
  sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_51_l599_59947


namespace NUMINAMATH_CALUDE_function_max_min_condition_l599_59954

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a + 2)*x + 1

-- State the theorem
theorem function_max_min_condition (a : ℝ) : 
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) ↔ (a > 2 ∨ a < -1) := by
  sorry

end NUMINAMATH_CALUDE_function_max_min_condition_l599_59954


namespace NUMINAMATH_CALUDE_class_size_calculation_l599_59973

/-- The number of students supposed to be in Miss Smith's second period English class -/
def total_students : ℕ :=
  let tables := 6
  let students_per_table := 3
  let present_students := tables * students_per_table
  let bathroom_students := 3
  let canteen_students := 3 * bathroom_students
  let new_group_size := 4
  let new_groups := 2
  let new_students := new_groups * new_group_size
  let foreign_students := 3 + 3 + 3  -- Germany, France, Norway

  present_students + bathroom_students + canteen_students + new_students + foreign_students

theorem class_size_calculation :
  total_students = 47 := by
  sorry

end NUMINAMATH_CALUDE_class_size_calculation_l599_59973


namespace NUMINAMATH_CALUDE_parabola_vertex_l599_59955

/-- A parabola with vertex (h, k) has the general form y = (x - h)² + k -/
def is_parabola_with_vertex (f : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∀ x, f x = (x - h)^2 + k

/-- The specific parabola we're considering -/
def f (x : ℝ) : ℝ := (x - 4)^2 - 3

/-- Theorem stating that f is a parabola with vertex (4, -3) -/
theorem parabola_vertex : is_parabola_with_vertex f 4 (-3) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l599_59955


namespace NUMINAMATH_CALUDE_meeting_distance_l599_59992

/-- Represents the distance walked by a person -/
def distance_walked (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Two people walking towards each other from 35 miles apart, 
    one at 2 mph and the other at 5 mph, will meet when the faster one has walked 25 miles -/
theorem meeting_distance (initial_distance : ℝ) (speed_fred : ℝ) (speed_sam : ℝ) :
  initial_distance = 35 →
  speed_fred = 2 →
  speed_sam = 5 →
  ∃ (time : ℝ), 
    distance_walked speed_fred time + distance_walked speed_sam time = initial_distance ∧
    distance_walked speed_sam time = 25 := by
  sorry

end NUMINAMATH_CALUDE_meeting_distance_l599_59992


namespace NUMINAMATH_CALUDE_linear_equation_solution_l599_59989

theorem linear_equation_solution (x y m : ℝ) : 
  x = 2 ∧ y = -3 ∧ 5 * x + m * y + 2 = 0 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l599_59989


namespace NUMINAMATH_CALUDE_negative_sqrt_six_squared_equals_six_l599_59907

theorem negative_sqrt_six_squared_equals_six : (-Real.sqrt 6)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_six_squared_equals_six_l599_59907


namespace NUMINAMATH_CALUDE_max_walk_distance_l599_59905

/-- Represents the board system with a person walking on it. -/
structure BoardSystem where
  l : ℝ  -- Length of the board
  m : ℝ  -- Mass of the board
  x : ℝ  -- Distance the person walks from the stone

/-- The conditions for the board system to be in equilibrium. -/
def is_equilibrium (bs : BoardSystem) : Prop :=
  bs.l = 20 ∧  -- Board length is 20 meters
  bs.x ≤ bs.l ∧  -- Person cannot walk beyond the board length
  2 * bs.m * (bs.l / 4) = bs.m * (3 * bs.l / 8) + (bs.m / 2) * (bs.x - bs.l / 4)

/-- The theorem stating the maximum distance a person can walk. -/
theorem max_walk_distance (bs : BoardSystem) :
  is_equilibrium bs → bs.x = bs.l / 2 := by
  sorry

#check max_walk_distance

end NUMINAMATH_CALUDE_max_walk_distance_l599_59905


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l599_59901

theorem degree_to_radian_conversion (π : Real) (h : π * 1 = 180) : 
  60 * (π / 180) = π / 3 := by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l599_59901


namespace NUMINAMATH_CALUDE_inequality_holds_iff_theta_in_range_l599_59939

theorem inequality_holds_iff_theta_in_range :
  ∀ k : ℤ, ∀ θ : ℝ,
    (2 * k * Real.pi + Real.pi / 12 < θ ∧ θ < 2 * k * Real.pi + 5 * Real.pi / 12) ↔
    (∀ x : ℝ, x ∈ Set.Icc 0 1 →
      x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_theta_in_range_l599_59939


namespace NUMINAMATH_CALUDE_min_value_sin_product_l599_59986

theorem min_value_sin_product (x₁ x₂ x₃ x₄ : ℝ) 
  (h_positive : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ = Real.pi) : 
  (2 * Real.sin x₁ ^ 2 + 1 / Real.sin x₁ ^ 2) *
  (2 * Real.sin x₂ ^ 2 + 1 / Real.sin x₂ ^ 2) *
  (2 * Real.sin x₃ ^ 2 + 1 / Real.sin x₃ ^ 2) *
  (2 * Real.sin x₄ ^ 2 + 1 / Real.sin x₄ ^ 2) ≥ 81 :=
by sorry

#check min_value_sin_product

end NUMINAMATH_CALUDE_min_value_sin_product_l599_59986


namespace NUMINAMATH_CALUDE_line_equation_with_triangle_area_l599_59944

/-- The equation of a line passing through two points and forming a triangle -/
theorem line_equation_with_triangle_area 
  (b S : ℝ) (hb : b ≠ 0) (hS : S > 0) :
  let k := 2 * S / b
  let line_eq := fun (x y : ℝ) ↦ 2 * S * x - b^2 * y + 2 * b * S
  (∀ y, line_eq (-b) y = 0) ∧ 
  (∀ x, line_eq x k = 0) ∧
  (∃ x y, x < 0 ∧ y > 0 ∧ line_eq x y = 0) ∧
  (S = (1/2) * b * k) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_with_triangle_area_l599_59944
