import Mathlib

namespace NUMINAMATH_CALUDE_eulers_formula_l1381_138126

/-- A convex polyhedron with vertices, edges, and faces. -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's formula for convex polyhedra. -/
theorem eulers_formula (p : ConvexPolyhedron) : p.vertices - p.edges + p.faces = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l1381_138126


namespace NUMINAMATH_CALUDE_division_problem_l1381_138151

theorem division_problem (dividend : ℕ) (quotient : ℕ) (divisor : ℕ) : 
  dividend = 62976 → quotient = 123 → divisor = 512 → 
  dividend = divisor * quotient ∧ dividend = 62976 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1381_138151


namespace NUMINAMATH_CALUDE_vector_equality_properties_l1381_138173

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def same_direction (a b : E) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ a = k • b

theorem vector_equality_properties (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) (heq : a = b) :
  same_direction a b ∧ ‖a‖ = ‖b‖ := by sorry

end NUMINAMATH_CALUDE_vector_equality_properties_l1381_138173


namespace NUMINAMATH_CALUDE_lighthouse_distance_l1381_138194

theorem lighthouse_distance (a : ℝ) (h : a > 0) :
  let A : ℝ × ℝ := (a * Real.cos (20 * π / 180), a * Real.sin (20 * π / 180))
  let B : ℝ × ℝ := (a * Real.cos (220 * π / 180), a * Real.sin (220 * π / 180))
  let C : ℝ × ℝ := (0, 0)
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 3 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_lighthouse_distance_l1381_138194


namespace NUMINAMATH_CALUDE_not_perfect_square_l1381_138183

theorem not_perfect_square (a : ℤ) : a ≠ 0 → ¬∃ x : ℤ, a^2 + 4 = x^2 := by sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1381_138183


namespace NUMINAMATH_CALUDE_inverse_proportion_ratios_l1381_138148

/-- Given that c is inversely proportional to d, prove the ratios of their values -/
theorem inverse_proportion_ratios 
  (k : ℝ) 
  (c d : ℝ → ℝ) 
  (h1 : ∀ x, c x * d x = k) 
  (c1 c2 d1 d2 c3 d3 : ℝ) 
  (h2 : c1 / c2 = 4 / 5) 
  (h3 : c3 = 2 * c1) :
  d1 / d2 = 5 / 4 ∧ d3 = d1 / 2 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratios_l1381_138148


namespace NUMINAMATH_CALUDE_difference_of_squares_l1381_138107

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1381_138107


namespace NUMINAMATH_CALUDE_angle_COD_measure_l1381_138147

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric relations
variable (belongs_to : Point → Circle → Prop)
variable (intersect_at : Circle → Circle → Point → Point → Prop)
variable (tangent_to : Point → Circle → Prop)
variable (lies_on_ray : Point → Point → Point → Prop)
variable (is_midpoint : Point → Point → Point → Prop)
variable (is_circumcenter : Point → Point → Point → Point → Prop)
variable (angle_measure : Point → Point → Point → ℝ)

-- Define the given points and circles
variable (ω₁ ω₂ : Circle)
variable (A B P Q R S O C D : Point)

-- State the theorem
theorem angle_COD_measure :
  intersect_at ω₁ ω₂ A B →
  tangent_to P ω₁ →
  tangent_to Q ω₂ →
  lies_on_ray R P A →
  lies_on_ray S Q A →
  angle_measure A P Q = 45 →
  angle_measure A Q P = 30 →
  is_circumcenter O A S R →
  is_midpoint C A P →
  is_midpoint D A Q →
  angle_measure C O D = 142.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_COD_measure_l1381_138147


namespace NUMINAMATH_CALUDE_quadratic_function_existence_l1381_138152

theorem quadratic_function_existence : ∃ (a b c : ℝ), 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a * x^2 + b * x + c| ≤ 1) ∧
  |a * 2^2 + b * 2 + c| ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_existence_l1381_138152


namespace NUMINAMATH_CALUDE_indeterminate_existence_l1381_138155

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (Q : U → Prop)  -- Q(x) means x is a quadrilateral
variable (A : U → Prop)  -- A(x) means x has property A

-- State the theorem
theorem indeterminate_existence (h : ¬(∀ x, Q x → A x)) :
  ¬(∀ p q : Prop, p = (∃ x, Q x ∧ A x) → (q = True ∨ q = False)) :=
sorry

end NUMINAMATH_CALUDE_indeterminate_existence_l1381_138155


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l1381_138153

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2*a - 1)*x + 4*a else Real.log x / Real.log a

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/6) (1/2) ∧ a ≠ 1/2 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l1381_138153


namespace NUMINAMATH_CALUDE_snyder_income_proof_l1381_138177

/-- Mrs. Snyder's previous monthly income -/
def previous_income : ℝ := 1700

/-- Mrs. Snyder's salary increase -/
def salary_increase : ℝ := 850

/-- Percentage of income spent on rent and utilities before salary increase -/
def previous_percentage : ℝ := 0.45

/-- Percentage of income spent on rent and utilities after salary increase -/
def new_percentage : ℝ := 0.30

theorem snyder_income_proof :
  (previous_percentage * previous_income = new_percentage * (previous_income + salary_increase)) ∧
  previous_income = 1700 := by
  sorry

end NUMINAMATH_CALUDE_snyder_income_proof_l1381_138177


namespace NUMINAMATH_CALUDE_squared_difference_product_l1381_138172

theorem squared_difference_product (a b : ℝ) : 
  a = 4 + 2 * Real.sqrt 5 → 
  b = 4 - 2 * Real.sqrt 5 → 
  a^2 * b - a * b^2 = -16 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_squared_difference_product_l1381_138172


namespace NUMINAMATH_CALUDE_large_monkey_cost_is_correct_l1381_138197

/-- The cost of a large monkey doll -/
def large_monkey_cost : ℝ := 6

/-- The total amount spent on dolls -/
def total_spent : ℝ := 300

/-- The cost difference between large and small monkey dolls -/
def small_large_diff : ℝ := 2

/-- The cost difference between elephant and large monkey dolls -/
def elephant_large_diff : ℝ := 1

/-- The number of additional dolls if buying only small monkeys -/
def small_monkey_diff : ℕ := 25

/-- The number of fewer dolls if buying only elephants -/
def elephant_diff : ℕ := 15

theorem large_monkey_cost_is_correct : 
  (total_spent / (large_monkey_cost - small_large_diff) = 
   total_spent / large_monkey_cost + small_monkey_diff) ∧
  (total_spent / (large_monkey_cost + elephant_large_diff) = 
   total_spent / large_monkey_cost - elephant_diff) := by
  sorry

end NUMINAMATH_CALUDE_large_monkey_cost_is_correct_l1381_138197


namespace NUMINAMATH_CALUDE_max_value_of_reciprocal_sum_l1381_138122

theorem max_value_of_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1) 
  (hax : a^x = 3) (hby : b^y = 3) 
  (hab : a + b = 2 * Real.sqrt 3) : 
  (∀ x' y' a' b' : ℝ, a' > 1 → b' > 1 → a'^x' = 3 → b'^y' = 3 → a' + b' = 2 * Real.sqrt 3 → 
    1/x' + 1/y' ≤ 1/x + 1/y) ∧ 1/x + 1/y = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_reciprocal_sum_l1381_138122


namespace NUMINAMATH_CALUDE_total_pastries_is_97_l1381_138139

/-- Given the number of pastries for Grace, calculate the total number of pastries for Grace, Calvin, Phoebe, and Frank. -/
def totalPastries (grace : ℕ) : ℕ :=
  let calvin := grace - 5
  let phoebe := grace - 5
  let frank := calvin - 8
  grace + calvin + phoebe + frank

/-- Theorem stating that given Grace has 30 pastries, the total number of pastries for all four is 97. -/
theorem total_pastries_is_97 : totalPastries 30 = 97 := by
  sorry

#eval totalPastries 30

end NUMINAMATH_CALUDE_total_pastries_is_97_l1381_138139


namespace NUMINAMATH_CALUDE_betty_boxes_l1381_138193

def total_oranges : ℕ := 24
def oranges_per_box : ℕ := 8

theorem betty_boxes : 
  total_oranges / oranges_per_box = 3 := by sorry

end NUMINAMATH_CALUDE_betty_boxes_l1381_138193


namespace NUMINAMATH_CALUDE_chocolate_distribution_l1381_138104

/-- The number of students -/
def num_students : ℕ := 211

/-- The number of possible combinations of chocolate choices -/
def num_combinations : ℕ := 35

/-- The minimum number of students in the largest group -/
def min_largest_group : ℕ := 7

theorem chocolate_distribution :
  ∃ (group : Finset (Fin num_students)),
    group.card ≥ min_largest_group ∧
    ∀ (s₁ s₂ : Fin num_students),
      s₁ ∈ group → s₂ ∈ group →
      ∃ (c : Fin num_combinations), true :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l1381_138104


namespace NUMINAMATH_CALUDE_min_t_value_l1381_138181

theorem min_t_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a + b = 1) :
  (∀ t : ℝ, 2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ t - 1/2) →
  t ≥ Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_t_value_l1381_138181


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l1381_138196

theorem probability_of_white_ball (p_red p_black p_white : ℝ) : 
  p_red = 0.3 → p_black = 0.5 → p_red + p_black + p_white = 1 → p_white = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l1381_138196


namespace NUMINAMATH_CALUDE_non_prime_sequence_300th_term_l1381_138115

/-- A sequence of positive integers with primes omitted -/
def non_prime_sequence : ℕ → ℕ := sorry

/-- The 300th term of the non-prime sequence -/
def term_300 : ℕ := 609

theorem non_prime_sequence_300th_term :
  non_prime_sequence 300 = term_300 := by sorry

end NUMINAMATH_CALUDE_non_prime_sequence_300th_term_l1381_138115


namespace NUMINAMATH_CALUDE_V₃_at_one_horner_equiv_f_l1381_138167

-- Define the polynomial f(x) = 3x^5 + 2x^3 - 8x + 5
def f (x : ℝ) : ℝ := 3 * x^5 + 2 * x^3 - 8 * x + 5

-- Define Horner's method for this polynomial
def horner (x : ℝ) : ℝ := (((((3 * x + 0) * x + 2) * x + 0) * x - 8) * x + 5)

-- Define V₃ in Horner's method
def V₃ (x : ℝ) : ℝ := ((3 * x + 0) * x + 2) * x + 0

-- Theorem: V₃(1) = 2
theorem V₃_at_one : V₃ 1 = 2 := by
  sorry

-- Prove that Horner's method is equivalent to the original polynomial
theorem horner_equiv_f : ∀ x, horner x = f x := by
  sorry

end NUMINAMATH_CALUDE_V₃_at_one_horner_equiv_f_l1381_138167


namespace NUMINAMATH_CALUDE_consecutive_digits_difference_l1381_138138

theorem consecutive_digits_difference (a : ℕ) (h : 1 ≤ a ∧ a ≤ 8) : 
  (100 * (a + 1) + 10 * a + (a - 1)) - (100 * (a - 1) + 10 * a + (a + 1)) = 198 := by
sorry

end NUMINAMATH_CALUDE_consecutive_digits_difference_l1381_138138


namespace NUMINAMATH_CALUDE_tan_123_negative_l1381_138133

theorem tan_123_negative (a : ℝ) (h : Real.sin (123 * π / 180) = a) :
  Real.tan (123 * π / 180) < 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_123_negative_l1381_138133


namespace NUMINAMATH_CALUDE_ones_digit_of_6_pow_45_l1381_138103

theorem ones_digit_of_6_pow_45 : ∃ n : ℕ, 6^45 ≡ 6 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_ones_digit_of_6_pow_45_l1381_138103


namespace NUMINAMATH_CALUDE_pizza_problem_l1381_138145

/-- The sum of a geometric series with first term a, common ratio r, and n terms -/
def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The fraction of pizza eaten after n trips to the refrigerator -/
def pizza_eaten (n : ℕ) : ℚ :=
  geometric_sum (1/3) (1/3) n

theorem pizza_problem : pizza_eaten 6 = 364/729 := by
  sorry

end NUMINAMATH_CALUDE_pizza_problem_l1381_138145


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l1381_138106

theorem subtraction_of_fractions : (5 : ℚ) / 9 - (1 : ℚ) / 6 = (7 : ℚ) / 18 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l1381_138106


namespace NUMINAMATH_CALUDE_businessmen_drinks_l1381_138116

theorem businessmen_drinks (total : ℕ) (coffee : ℕ) (tea : ℕ) (both : ℕ) :
  total = 30 →
  coffee = 15 →
  tea = 14 →
  both = 7 →
  total - (coffee + tea - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_drinks_l1381_138116


namespace NUMINAMATH_CALUDE_live_flowers_l1381_138127

theorem live_flowers (total : ℕ) (withered : ℕ) (h1 : total = 13) (h2 : withered = 7) :
  total - withered = 6 := by
  sorry

end NUMINAMATH_CALUDE_live_flowers_l1381_138127


namespace NUMINAMATH_CALUDE_min_remote_uses_l1381_138113

/-- Represents the state of lamps --/
def LampState := Fin 169 → Bool

/-- The remote control operation --/
def remote_control (s : LampState) (switches : Finset (Fin 169)) : LampState :=
  λ i => if i ∈ switches then !s i else s i

/-- All lamps are initially on --/
def initial_state : LampState := λ _ => true

/-- All lamps are off --/
def all_off (s : LampState) : Prop := ∀ i, s i = false

/-- The remote control changes exactly 19 switches --/
def valid_remote_use (switches : Finset (Fin 169)) : Prop :=
  switches.card = 19

theorem min_remote_uses :
  ∃ (sequence : List (Finset (Fin 169))),
    sequence.length = 9 ∧
    (∀ switches ∈ sequence, valid_remote_use switches) ∧
    all_off (sequence.foldl remote_control initial_state) ∧
    (∀ (shorter_sequence : List (Finset (Fin 169))),
      shorter_sequence.length < 9 →
      (∀ switches ∈ shorter_sequence, valid_remote_use switches) →
      ¬ all_off (shorter_sequence.foldl remote_control initial_state)) :=
sorry

end NUMINAMATH_CALUDE_min_remote_uses_l1381_138113


namespace NUMINAMATH_CALUDE_complex_on_y_axis_l1381_138171

theorem complex_on_y_axis (a : ℝ) : 
  let z : ℂ := (a - 3 * Complex.I) / (1 - Complex.I)
  (Complex.re z = 0) → a = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_on_y_axis_l1381_138171


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l1381_138195

theorem least_number_for_divisibility : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬((5918273 + y) % (41 * 71 * 139) = 0)) ∧ 
  ((5918273 + x) % (41 * 71 * 139) = 0) := by
  sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l1381_138195


namespace NUMINAMATH_CALUDE_max_area_triangle_l1381_138163

/-- Given points A, B, C, and P in a plane with specific distances, 
    prove that the maximum possible area of triangle ABC is 18.5 -/
theorem max_area_triangle (A B C P : ℝ × ℝ) : 
  let PA := Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)
  let PB := Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2)
  let PC := Real.sqrt ((C.1 - P.1)^2 + (C.2 - P.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  PA = 5 ∧ PB = 4 ∧ PC = 3 ∧ BC = 5 →
  (∀ A' : ℝ × ℝ, 
    let PA' := Real.sqrt ((A'.1 - P.1)^2 + (A'.2 - P.2)^2)
    PA' = 5 →
    let area := abs ((A'.1 - B.1) * (C.2 - B.2) - (A'.2 - B.2) * (C.1 - B.1)) / 2
    area ≤ 18.5) :=
by sorry


end NUMINAMATH_CALUDE_max_area_triangle_l1381_138163


namespace NUMINAMATH_CALUDE_birds_in_tree_l1381_138187

theorem birds_in_tree (initial_birds final_birds : ℕ) (h1 : initial_birds = 179) (h2 : final_birds = 217) :
  final_birds - initial_birds = 38 := by
sorry

end NUMINAMATH_CALUDE_birds_in_tree_l1381_138187


namespace NUMINAMATH_CALUDE_alcohol_amount_l1381_138135

/-- Represents the amount of alcohol in liters -/
def alcohol : ℝ := 14

/-- Represents the amount of water in liters -/
def water : ℝ := 10.5

/-- The amount of water added to the mixture in liters -/
def water_added : ℝ := 7

/-- The initial ratio of alcohol to water -/
def initial_ratio : ℚ := 4/3

/-- The final ratio of alcohol to water after adding more water -/
def final_ratio : ℚ := 4/5

theorem alcohol_amount :
  (alcohol / water = initial_ratio) ∧
  (alcohol / (water + water_added) = final_ratio) →
  alcohol = 14 := by
sorry

end NUMINAMATH_CALUDE_alcohol_amount_l1381_138135


namespace NUMINAMATH_CALUDE_total_apples_l1381_138159

/-- Represents the number of apples Tessa has -/
def tessas_apples : ℕ := 4

/-- Represents the number of apples Anita gave to Tessa -/
def anitas_gift : ℕ := 5

/-- Theorem stating that Tessa's total apples is the sum of her initial apples and Anita's gift -/
theorem total_apples : tessas_apples + anitas_gift = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_l1381_138159


namespace NUMINAMATH_CALUDE_line_equation_proof_l1381_138192

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallelLines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_equation_proof (given_line : Line) (p : Point) :
  given_line.a = 1 →
  given_line.b = -2 →
  given_line.c = 3 →
  p.x = -1 →
  p.y = 3 →
  ∃ (result_line : Line),
    result_line.a = 1 ∧
    result_line.b = -2 ∧
    result_line.c = 7 ∧
    pointOnLine p result_line ∧
    parallelLines given_line result_line :=
by sorry


end NUMINAMATH_CALUDE_line_equation_proof_l1381_138192


namespace NUMINAMATH_CALUDE_complex_cube_equality_l1381_138108

theorem complex_cube_equality (a b c : ℝ) : 
  ((2 * a - b - c : ℂ) + (b - c) * Complex.I * Real.sqrt 3) ^ 3 = 
  ((2 * b - c - a : ℂ) + (c - a) * Complex.I * Real.sqrt 3) ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_equality_l1381_138108


namespace NUMINAMATH_CALUDE_sin_B_value_max_perimeter_l1381_138198

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition (2a-c)cosB = bcosC -/
def triangle_condition (t : Triangle) : Prop :=
  (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C

/-- Theorem 1: If (2a-c)cosB = bcosC, then sinB = √3/2 -/
theorem sin_B_value (t : Triangle) (h : triangle_condition t) : 
  Real.sin t.B = Real.sqrt 3 / 2 := by sorry

/-- Theorem 2: If b = √7, then the maximum perimeter is 3√7 -/
theorem max_perimeter (t : Triangle) (h : t.b = Real.sqrt 7) :
  t.a + t.b + t.c ≤ 3 * Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_sin_B_value_max_perimeter_l1381_138198


namespace NUMINAMATH_CALUDE_min_buses_required_l1381_138160

theorem min_buses_required (total_students : ℕ) (bus_capacity : ℕ) (h1 : total_students = 325) (h2 : bus_capacity = 45) :
  ∃ (n : ℕ), n * bus_capacity ≥ total_students ∧ ∀ m : ℕ, m * bus_capacity ≥ total_students → m ≥ n ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_buses_required_l1381_138160


namespace NUMINAMATH_CALUDE_analysis_seeks_sufficient_condition_l1381_138120

/-- Represents a mathematical method for proving inequalities -/
inductive ProofMethod
| Analysis
| Synthesis

/-- Represents types of conditions in mathematical proofs -/
inductive ConditionType
| Sufficient
| Necessary
| NecessaryAndSufficient
| Neither

/-- Represents an inequality to be proved -/
structure Inequality where
  -- We don't need to specify the actual inequality, just that it exists
  dummy : Unit

/-- Function that represents the process of seeking a condition in the analysis method -/
def seekCondition (m : ProofMethod) (i : Inequality) : ConditionType :=
  match m with
  | ProofMethod.Analysis => ConditionType.Sufficient
  | ProofMethod.Synthesis => ConditionType.Neither -- This is arbitrary for non-Analysis methods

/-- Theorem stating that the analysis method seeks a sufficient condition -/
theorem analysis_seeks_sufficient_condition (i : Inequality) :
  seekCondition ProofMethod.Analysis i = ConditionType.Sufficient := by
  sorry

#check analysis_seeks_sufficient_condition

end NUMINAMATH_CALUDE_analysis_seeks_sufficient_condition_l1381_138120


namespace NUMINAMATH_CALUDE_initial_overs_played_l1381_138166

/-- Proves that the number of overs played initially is 15, given the target score,
    initial run rate, required run rate for remaining overs, and the number of remaining overs. -/
theorem initial_overs_played (target_score : ℝ) (initial_run_rate : ℝ) (required_run_rate : ℝ) (remaining_overs : ℝ) :
  target_score = 275 →
  initial_run_rate = 3.2 →
  required_run_rate = 6.485714285714286 →
  remaining_overs = 35 →
  ∃ (initial_overs : ℝ), initial_overs = 15 ∧
    target_score = initial_run_rate * initial_overs + required_run_rate * remaining_overs :=
by
  sorry


end NUMINAMATH_CALUDE_initial_overs_played_l1381_138166


namespace NUMINAMATH_CALUDE_polygon_sides_l1381_138168

theorem polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → 
  (exterior_angle = 30) → 
  (n * exterior_angle = 360) → 
  n = 12 := by sorry

end NUMINAMATH_CALUDE_polygon_sides_l1381_138168


namespace NUMINAMATH_CALUDE_problem_solution_l1381_138162

theorem problem_solution (m a b c d : ℚ) 
  (h1 : |m + 1| = 4)
  (h2 : a + b = 0)
  (h3 : c * d = 1) :
  a + b + 3 * c * d - m = 0 ∨ a + b + 3 * c * d - m = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1381_138162


namespace NUMINAMATH_CALUDE_blood_cell_count_l1381_138188

/-- Given two blood samples with a total of 7341 blood cells, where the first sample
    contains 4221 blood cells, prove that the second sample contains 3120 blood cells. -/
theorem blood_cell_count (total : ℕ) (first_sample : ℕ) (second_sample : ℕ) 
    (h1 : total = 7341)
    (h2 : first_sample = 4221)
    (h3 : total = first_sample + second_sample) : 
  second_sample = 3120 := by
  sorry

end NUMINAMATH_CALUDE_blood_cell_count_l1381_138188


namespace NUMINAMATH_CALUDE_identical_solutions_iff_k_neg_one_l1381_138158

/-- 
Proves that the equations y = x^2 and y = 2x + k have two identical solutions 
if and only if k = -1.
-/
theorem identical_solutions_iff_k_neg_one (k : ℝ) : 
  (∃ x y : ℝ, y = x^2 ∧ y = 2*x + k ∧ 
   (∀ x' y' : ℝ, y' = x'^2 ∧ y' = 2*x' + k → x' = x ∧ y' = y)) ↔ 
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_identical_solutions_iff_k_neg_one_l1381_138158


namespace NUMINAMATH_CALUDE_initial_customers_l1381_138142

theorem initial_customers (initial leaving new final : ℕ) : 
  leaving = 8 → new = 4 → final = 9 → 
  initial - leaving + new = final → 
  initial = 13 := by sorry

end NUMINAMATH_CALUDE_initial_customers_l1381_138142


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1381_138129

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  a*x^2 + b*x + c = 0 ∧ a = 1 ∧ b = -5 ∧ c = 6 →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + b*x₁ + c = 0 ∧ a*x₂^2 + b*x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1381_138129


namespace NUMINAMATH_CALUDE_sqrt_divided_by_two_is_ten_l1381_138137

theorem sqrt_divided_by_two_is_ten (x : ℝ) : (Real.sqrt x) / 2 = 10 → x = 400 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_divided_by_two_is_ten_l1381_138137


namespace NUMINAMATH_CALUDE_sequence_strictly_increasing_l1381_138165

theorem sequence_strictly_increasing (n : ℕ) (h : n ≥ 14) : 
  let a : ℕ → ℤ := λ k => k^4 - 20*k^2 - 10*k + 1
  a n > a (n-1) := by sorry

end NUMINAMATH_CALUDE_sequence_strictly_increasing_l1381_138165


namespace NUMINAMATH_CALUDE_base_conversion_property_l1381_138178

def convert_base (n : ℕ) (from_base to_base : ℕ) : ℕ :=
  sorry

def digits_to_nat (digits : List ℕ) (base : ℕ) : ℕ :=
  sorry

def nat_to_digits (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

theorem base_conversion_property :
  ∀ b : ℕ, b ∈ [13, 12, 11] →
    let n := digits_to_nat [1, 2, 2, 1] b
    nat_to_digits (convert_base n b (b - 1)) (b - 1) = [1, 2, 2, 1] ∧
  let n₁₀ := digits_to_nat [1, 2, 2, 1] 10
  nat_to_digits (convert_base n₁₀ 10 9) 9 ≠ [1, 2, 2, 1] :=
by
  sorry

end NUMINAMATH_CALUDE_base_conversion_property_l1381_138178


namespace NUMINAMATH_CALUDE_infinitely_many_consecutive_almost_squares_l1381_138182

/-- A natural number is almost a square if it can be represented as a product of two numbers
    that differ by no more than one percent of the larger of them. -/
def AlmostSquare (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a * b ∧ (a : ℝ) ≥ (b : ℝ) ∧ (a : ℝ) ≤ (b : ℝ) * 1.01

/-- There exist infinitely many natural numbers m such that 4m^4 - 1, 4m^4, 4m^4 + 1, and 4m^4 + 2
    are all almost squares. -/
theorem infinitely_many_consecutive_almost_squares :
  ∀ N : ℕ, ∃ m : ℕ, m > N ∧
    AlmostSquare (4 * m^4 - 1) ∧
    AlmostSquare (4 * m^4) ∧
    AlmostSquare (4 * m^4 + 1) ∧
    AlmostSquare (4 * m^4 + 2) := by
  sorry


end NUMINAMATH_CALUDE_infinitely_many_consecutive_almost_squares_l1381_138182


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1381_138128

theorem inverse_variation_problem (a b : ℝ) (k : ℝ) (h1 : a * b^3 = k) (h2 : 8 * 1^3 = k) :
  a * 4^3 = k → a = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1381_138128


namespace NUMINAMATH_CALUDE_library_books_fraction_l1381_138124

theorem library_books_fraction (total : ℕ) (sold : ℕ) (h1 : total = 9900) (h2 : sold = 3300) :
  (total - sold : ℚ) / total = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_library_books_fraction_l1381_138124


namespace NUMINAMATH_CALUDE_coronavirus_case_increase_l1381_138169

theorem coronavirus_case_increase (initial_cases : ℕ) 
  (second_day_recoveries : ℕ) (third_day_new_cases : ℕ) 
  (third_day_recoveries : ℕ) (final_total_cases : ℕ) :
  initial_cases = 2000 →
  second_day_recoveries = 50 →
  third_day_new_cases = 1500 →
  third_day_recoveries = 200 →
  final_total_cases = 3750 →
  ∃ (second_day_increase : ℕ),
    final_total_cases = initial_cases + second_day_increase - second_day_recoveries + 
      third_day_new_cases - third_day_recoveries ∧
    second_day_increase = 750 :=
by sorry

end NUMINAMATH_CALUDE_coronavirus_case_increase_l1381_138169


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l1381_138190

/-- Given two digits A and B in base d > 7, if AB + BA = 202 in base d, then A - B = 2 in base d -/
theorem digit_difference_in_base_d (d : ℕ) (A B : ℕ) : 
  d > 7 →
  A < d →
  B < d →
  (A * d + B) + (B * d + A) = 2 * d^2 + 2 →
  A - B = 2 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l1381_138190


namespace NUMINAMATH_CALUDE_triangle_perpendicular_segment_length_l1381_138114

-- Define the triangle XYZ
structure Triangle (X Y Z : ℝ × ℝ) : Prop where
  right_angle : (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0
  xy_length : Real.sqrt ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) = 5
  xz_length : Real.sqrt ((Z.1 - X.1)^2 + (Z.2 - X.2)^2) = 12

-- Define the perpendicular segment LM
def perpendicular_segment (X Y Z M : ℝ × ℝ) : Prop :=
  (M.1 - X.1) * (Y.1 - X.1) + (M.2 - X.2) * (Y.2 - X.2) = 0

-- Theorem statement
theorem triangle_perpendicular_segment_length 
  (X Y Z M : ℝ × ℝ) (h : Triangle X Y Z) (h_perp : perpendicular_segment X Y Z M) :
  Real.sqrt ((M.1 - Y.1)^2 + (M.2 - Y.2)^2) = (5 * Real.sqrt 119) / 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perpendicular_segment_length_l1381_138114


namespace NUMINAMATH_CALUDE_cosine_sum_120_l1381_138140

theorem cosine_sum_120 (α : ℝ) : 
  Real.cos (α - 120 * π / 180) + Real.cos α + Real.cos (α + 120 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_120_l1381_138140


namespace NUMINAMATH_CALUDE_expression_simplification_l1381_138191

theorem expression_simplification (x : ℝ) : 3*x + 4*x^2 + 2 - (9 - 3*x - 4*x^2) + Real.sin x = 8*x^2 + 6*x - 7 + Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1381_138191


namespace NUMINAMATH_CALUDE_circle_center_l1381_138130

/-- The center of a circle given by the equation 4x^2 + 8x + 4y^2 - 12y + 20 = 0 is (-1, 3/2) -/
theorem circle_center (x y : ℝ) : 
  (4 * x^2 + 8 * x + 4 * y^2 - 12 * y + 20 = 0) → 
  (∃ r : ℝ, (x + 1)^2 + (y - 3/2)^2 = r^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l1381_138130


namespace NUMINAMATH_CALUDE_complex_number_problem_l1381_138149

theorem complex_number_problem (z : ℂ) (hz : z ≠ 0) :
  Complex.abs (z + 2) = 2 ∧ (z + 4 / z).im = 0 →
  z = -1 + Complex.I * Real.sqrt 3 ∨ z = -1 - Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1381_138149


namespace NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l1381_138110

theorem unique_solution_logarithmic_equation (a b x : ℝ) :
  a > 0 ∧ b > 0 ∧ x > 1 ∧
  9 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = 17 ∧
  (Real.log b / Real.log a) * (Real.log a / Real.log b) = 2 →
  a = Real.exp (Real.sqrt 2 * Real.log 10) ∧ b = 10 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l1381_138110


namespace NUMINAMATH_CALUDE_prob_six_queen_is_4_663_l1381_138146

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of sixes in a standard deck -/
def NumSixes : ℕ := 4

/-- Number of queens in a standard deck -/
def NumQueens : ℕ := 4

/-- Probability of drawing a 6 as the first card and a Queen as the second card -/
def ProbSixQueen : ℚ := (NumSixes : ℚ) / StandardDeck * NumQueens / (StandardDeck - 1)

theorem prob_six_queen_is_4_663 : ProbSixQueen = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_six_queen_is_4_663_l1381_138146


namespace NUMINAMATH_CALUDE_problem_solution_l1381_138175

theorem problem_solution (a b : ℕ+) (h : (a.val^3 - a.val^2 + 1) * (b.val^3 - b.val^2 + 2) = 2020) :
  10 * a.val + b.val = 53 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1381_138175


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1381_138179

theorem diophantine_equation_solution :
  ∀ (a b c : ℤ), 5 * a^2 + 9 * b^2 = 13 * c^2 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1381_138179


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1381_138141

/-- Theorem: Area of a rectangular field -/
theorem rectangular_field_area (width : ℝ) : 
  (width ≥ 0) →
  (16 * width + 54 = 22 * width) → 
  (16 * width = 144) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1381_138141


namespace NUMINAMATH_CALUDE_intersection_complement_equal_l1381_138144

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 3, 5, 6}
def B : Set ℕ := {x ∈ U | x^2 - 5*x ≥ 0}

theorem intersection_complement_equal : A ∩ (U \ B) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_l1381_138144


namespace NUMINAMATH_CALUDE_eighteenth_replacement_november_l1381_138136

/-- Represents months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Converts a number of months to a Month -/
def monthsToMonth (n : ℕ) : Month :=
  match n % 12 with
  | 0 => Month.December
  | 1 => Month.January
  | 2 => Month.February
  | 3 => Month.March
  | 4 => Month.April
  | 5 => Month.May
  | 6 => Month.June
  | 7 => Month.July
  | 8 => Month.August
  | 9 => Month.September
  | 10 => Month.October
  | _ => Month.November

/-- The month of the nth wheel replacement, given a 7-month cycle starting in January -/
def wheelReplacementMonth (n : ℕ) : Month :=
  monthsToMonth ((n - 1) * 7 + 1)

theorem eighteenth_replacement_november :
  wheelReplacementMonth 18 = Month.November := by
  sorry

end NUMINAMATH_CALUDE_eighteenth_replacement_november_l1381_138136


namespace NUMINAMATH_CALUDE_exactly_three_sets_l1381_138123

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)
  (length : ℕ)
  (length_ge_two : length ≥ 2)

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- Predicate for a valid set of consecutive integers summing to 150 -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  sum_consecutive s = 150

theorem exactly_three_sets : 
  ∃! (sets : Finset ConsecutiveSet), 
    (∀ s ∈ sets, is_valid_set s) ∧ 
    sets.card = 3 := by sorry

end NUMINAMATH_CALUDE_exactly_three_sets_l1381_138123


namespace NUMINAMATH_CALUDE_amoeba_population_after_10_days_l1381_138118

/-- The number of amoebas after n days, given an initial population of 2 -/
def amoeba_population (n : ℕ) : ℕ := 2 * 3^n

/-- Theorem stating that the amoeba population after 10 days is 118098 -/
theorem amoeba_population_after_10_days : amoeba_population 10 = 118098 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_population_after_10_days_l1381_138118


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1381_138125

def M (a : ℕ) : Set ℕ := {3, 2^a}
def N (a b : ℕ) : Set ℕ := {a, b}

theorem union_of_M_and_N (a b : ℕ) :
  M a ∩ N a b = {2} →
  M a ∪ N a b = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1381_138125


namespace NUMINAMATH_CALUDE_negative_power_division_l1381_138176

theorem negative_power_division : -3^7 / 3^2 = -3^5 := by sorry

end NUMINAMATH_CALUDE_negative_power_division_l1381_138176


namespace NUMINAMATH_CALUDE_sum_of_integers_l1381_138185

theorem sum_of_integers (a b c d : ℤ) 
  (eq1 : a - b + 2*c = 7)
  (eq2 : b - c + d = 8)
  (eq3 : c - d + a = 5)
  (eq4 : d - a + b = 4) : 
  a + b + c + d = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1381_138185


namespace NUMINAMATH_CALUDE_cos_330_deg_l1381_138174

/-- Cosine of 330 degrees is equal to √3/2 -/
theorem cos_330_deg : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_deg_l1381_138174


namespace NUMINAMATH_CALUDE_probability_y_l1381_138134

theorem probability_y (x y : Set Ω) (z : Set Ω → ℝ) 
  (hx : z x = 0.02)
  (hxy : z (x ∩ y) = 0.10)
  (hcond : z x / z y = 0.2) :
  z y = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_probability_y_l1381_138134


namespace NUMINAMATH_CALUDE_category_A_sample_size_l1381_138121

/-- Represents the number of students in each school category -/
structure SchoolCategories where
  categoryA : ℕ
  categoryB : ℕ
  categoryC : ℕ

/-- Calculates the number of students selected from a category in stratified sampling -/
def stratifiedSample (categories : SchoolCategories) (totalSample : ℕ) (categorySize : ℕ) : ℕ :=
  (categorySize * totalSample) / (categories.categoryA + categories.categoryB + categories.categoryC)

/-- Theorem: The number of students selected from Category A in the given scenario is 200 -/
theorem category_A_sample_size :
  let categories := SchoolCategories.mk 2000 3000 4000
  let totalSample := 900
  stratifiedSample categories totalSample categories.categoryA = 200 := by
  sorry

end NUMINAMATH_CALUDE_category_A_sample_size_l1381_138121


namespace NUMINAMATH_CALUDE_average_side_lengths_of_squares_l1381_138102

theorem average_side_lengths_of_squares (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 36) (h₂ : a₂ = 64) (h₃ : a₃ = 144) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 26 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_side_lengths_of_squares_l1381_138102


namespace NUMINAMATH_CALUDE_one_thirds_in_eleven_fifths_l1381_138170

theorem one_thirds_in_eleven_fifths : (11 / 5 : ℚ) / (1 / 3 : ℚ) = 33 / 5 := by sorry

end NUMINAMATH_CALUDE_one_thirds_in_eleven_fifths_l1381_138170


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1381_138112

theorem complex_equation_solution :
  ∀ (z : ℂ), z = Complex.I * (2 - z) → z = 1 + Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1381_138112


namespace NUMINAMATH_CALUDE_sochi_puzzle_solution_l1381_138184

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Digit
  hundreds : Digit
  tens : Digit
  ones : Digit

/-- Convert a FourDigitNumber to a natural number -/
def FourDigitNumber.toNat (n : FourDigitNumber) : Nat :=
  1000 * n.thousands.val + 100 * n.hundreds.val + 10 * n.tens.val + n.ones.val

/-- Check if all digits in a FourDigitNumber are unique -/
def FourDigitNumber.uniqueDigits (n : FourDigitNumber) : Prop :=
  n.thousands ≠ n.hundreds ∧ n.thousands ≠ n.tens ∧ n.thousands ≠ n.ones ∧
  n.hundreds ≠ n.tens ∧ n.hundreds ≠ n.ones ∧
  n.tens ≠ n.ones

theorem sochi_puzzle_solution :
  ∃ (year sochi : FourDigitNumber),
    year.uniqueDigits ∧
    sochi.uniqueDigits ∧
    2014 + year.toNat = sochi.toNat :=
  sorry

end NUMINAMATH_CALUDE_sochi_puzzle_solution_l1381_138184


namespace NUMINAMATH_CALUDE_simplify_and_factorize_l1381_138164

theorem simplify_and_factorize (x : ℝ) : 
  3 * x^2 + 4 * x + 5 - (7 - 3 * x^2 - 5 * x) = (x + 2) * (6 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_factorize_l1381_138164


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1381_138154

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 1) / (3 * x^3))^2) = x^3 / 3 + 1 / (3 * x^3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1381_138154


namespace NUMINAMATH_CALUDE_power_simplification_l1381_138101

theorem power_simplification : 16^6 * 4^6 * 16^10 * 4^10 = 64^16 := by
  sorry

end NUMINAMATH_CALUDE_power_simplification_l1381_138101


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1381_138156

/-- Proves that a parallelogram with area 44 cm² and height 11 cm has a base of 4 cm -/
theorem parallelogram_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (is_parallelogram : Bool) 
  (h1 : is_parallelogram = true)
  (h2 : area = 44)
  (h3 : height = 11) :
  area / height = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1381_138156


namespace NUMINAMATH_CALUDE_committee_problem_l1381_138199

/-- The number of ways to form a committee with the given constraints -/
def committee_formations (n m k r : ℕ) : ℕ :=
  Nat.choose n k - Nat.choose (n - m) k

theorem committee_problem :
  let total_members : ℕ := 30
  let founding_members : ℕ := 10
  let committee_size : ℕ := 5
  committee_formations total_members founding_members committee_size = 126992 := by
  sorry

end NUMINAMATH_CALUDE_committee_problem_l1381_138199


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1381_138186

/-- The complex number (1-i)·i corresponds to a point in the fourth quadrant of the complex plane. -/
theorem point_in_fourth_quadrant : ∃ (z : ℂ), z = (1 - Complex.I) * Complex.I ∧ z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1381_138186


namespace NUMINAMATH_CALUDE_ap_sum_terms_l1381_138161

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a₁ : ℤ     -- First term
  d : ℤ      -- Common difference

/-- Calculates the sum of the first n terms of an arithmetic progression -/
def sum_of_terms (ap : ArithmeticProgression) (n : ℕ) : ℤ :=
  n * (2 * ap.a₁ + (n - 1) * ap.d) / 2

/-- Theorem: The number of terms needed for the sum to equal 3069 in the given arithmetic progression is either 9 or 31 -/
theorem ap_sum_terms (ap : ArithmeticProgression) 
  (h1 : ap.a₁ = 429) 
  (h2 : ap.d = -22) : 
  (∃ n : ℕ, sum_of_terms ap n = 3069) → (n = 9 ∨ n = 31) :=
sorry

end NUMINAMATH_CALUDE_ap_sum_terms_l1381_138161


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l1381_138150

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {3, 4, 5}

-- Define set N
def N : Set Nat := {2, 3}

-- Theorem statement
theorem complement_intersection_equals_set :
  (U \ N) ∩ M = {4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l1381_138150


namespace NUMINAMATH_CALUDE_white_washing_cost_is_4530_l1381_138111

/-- Calculates the cost of white washing a room with given dimensions and openings -/
def white_washing_cost (room_length room_width room_height : ℝ)
                       (door_length door_width : ℝ)
                       (window_length window_width : ℝ)
                       (num_windows : ℕ)
                       (cost_per_sqft : ℝ) : ℝ :=
  let wall_area := 2 * (room_length + room_width) * room_height
  let door_area := door_length * door_width
  let window_area := window_length * window_width * num_windows
  let paintable_area := wall_area - door_area - window_area
  paintable_area * cost_per_sqft

/-- The cost of white washing the room is 4530 rupees -/
theorem white_washing_cost_is_4530 :
  white_washing_cost 25 15 12 6 3 4 3 3 5 = 4530 := by
  sorry

end NUMINAMATH_CALUDE_white_washing_cost_is_4530_l1381_138111


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_ending_4_l1381_138180

theorem greatest_three_digit_multiple_of_17_ending_4 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ n % 10 = 4 → n ≤ 204 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_ending_4_l1381_138180


namespace NUMINAMATH_CALUDE_unique_solution_iff_m_not_neg_two_and_not_zero_l1381_138109

/-- Given an equation (m^2 + 2m + 3)x = 3(x + 2) + m - 4, it has a unique solution
    with respect to x if and only if m ≠ -2 and m ≠ 0 -/
theorem unique_solution_iff_m_not_neg_two_and_not_zero (m : ℝ) :
  (∃! x : ℝ, (m^2 + 2*m + 3)*x = 3*(x + 2) + m - 4) ↔ (m ≠ -2 ∧ m ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_iff_m_not_neg_two_and_not_zero_l1381_138109


namespace NUMINAMATH_CALUDE_household_coffee_expense_l1381_138100

def weekly_coffee_expense (person_a_cups : ℕ) (person_a_ounces : ℝ)
                          (person_b_cups : ℕ) (person_b_ounces : ℝ)
                          (person_c_cups : ℕ) (person_c_ounces : ℝ)
                          (person_c_days : ℕ) (coffee_cost : ℝ) : ℝ :=
  let person_a_weekly := person_a_cups * person_a_ounces * 7
  let person_b_weekly := person_b_cups * person_b_ounces * 7
  let person_c_weekly := person_c_cups * person_c_ounces * person_c_days
  let total_weekly_ounces := person_a_weekly + person_b_weekly + person_c_weekly
  total_weekly_ounces * coffee_cost

theorem household_coffee_expense :
  weekly_coffee_expense 3 0.4 1 0.6 2 0.5 5 1.25 = 22 := by
  sorry

end NUMINAMATH_CALUDE_household_coffee_expense_l1381_138100


namespace NUMINAMATH_CALUDE_jeremy_watermelons_l1381_138132

/-- The number of watermelons Jeremy eats per week -/
def jeremy_eats_per_week : ℕ := 3

/-- The number of watermelons Jeremy gives to his dad per week -/
def jeremy_gives_dad_per_week : ℕ := 2

/-- The number of weeks the watermelons will last -/
def weeks_watermelons_last : ℕ := 6

/-- The total number of watermelons Jeremy bought -/
def total_watermelons : ℕ := 30

theorem jeremy_watermelons :
  total_watermelons = (jeremy_eats_per_week + jeremy_gives_dad_per_week) * weeks_watermelons_last :=
by sorry

end NUMINAMATH_CALUDE_jeremy_watermelons_l1381_138132


namespace NUMINAMATH_CALUDE_zacks_marbles_l1381_138119

theorem zacks_marbles (friends : ℕ) (ratio : List ℕ) (leftover : ℕ) (initial : ℕ) :
  friends = 9 →
  ratio = [5, 6, 7, 8, 9, 10, 11, 12, 13] →
  leftover = 27 →
  initial = (ratio.sum * 3) + leftover →
  initial = 270 :=
by sorry

end NUMINAMATH_CALUDE_zacks_marbles_l1381_138119


namespace NUMINAMATH_CALUDE_kylie_picked_220_apples_l1381_138189

/-- The number of apples Kylie picked in the first hour -/
def first_hour_apples : ℕ := 66

/-- The number of apples Kylie picked in the second hour -/
def second_hour_apples : ℕ := 2 * first_hour_apples

/-- The number of apples Kylie picked in the third hour -/
def third_hour_apples : ℕ := first_hour_apples / 3

/-- The total number of apples Kylie picked -/
def total_apples : ℕ := first_hour_apples + second_hour_apples + third_hour_apples

/-- Theorem stating that the total number of apples Kylie picked is 220 -/
theorem kylie_picked_220_apples : total_apples = 220 := by
  sorry

end NUMINAMATH_CALUDE_kylie_picked_220_apples_l1381_138189


namespace NUMINAMATH_CALUDE_custom_baseball_caps_l1381_138131

theorem custom_baseball_caps (jack_circumference bill_circumference : ℝ)
  (h1 : jack_circumference = 12)
  (h2 : bill_circumference = 10)
  (h3 : ∃ f : ℝ, charlie_circumference = f * jack_circumference + 9)
  (h4 : bill_circumference = (2/3) * charlie_circumference) :
  ∃ f : ℝ, charlie_circumference = f * jack_circumference + 9 ∧ f = (1/2) :=
by
  sorry
where
  charlie_circumference : ℝ := bill_circumference / (2/3)

end NUMINAMATH_CALUDE_custom_baseball_caps_l1381_138131


namespace NUMINAMATH_CALUDE_food_bank_donation_ratio_l1381_138157

theorem food_bank_donation_ratio :
  let foster_chickens : ℕ := 45
  let american_water := 2 * foster_chickens
  let hormel_chickens := 3 * foster_chickens
  let del_monte_water := american_water - 30
  let total_items : ℕ := 375
  let boudin_chickens := total_items - (foster_chickens + american_water + hormel_chickens + del_monte_water)
  (boudin_chickens : ℚ) / hormel_chickens = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_food_bank_donation_ratio_l1381_138157


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1381_138143

theorem fraction_multiplication : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1381_138143


namespace NUMINAMATH_CALUDE_inequality_solution_l1381_138105

theorem inequality_solution (x : ℝ) : 
  (x ∈ Set.Iio (-2) ∪ Set.Ioo (-1) 1 ∪ Set.Ioo 2 3 ∪ Set.Ioo 4 6 ∪ Set.Ioi 7) ↔ 
  (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ 
   (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 24)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1381_138105


namespace NUMINAMATH_CALUDE_benjamin_weekly_miles_l1381_138117

/-- Calculates the total miles Benjamin walks in a week --/
def total_miles_walked : ℕ :=
  let work_distance := 6
  let dog_walk_distance := 2
  let friend_house_distance := 1
  let store_distance := 3
  let work_days := 5
  let dog_walks_per_day := 2
  let days_in_week := 7
  let store_visits := 2
  let friend_visits := 1

  let work_miles := work_distance * 2 * work_days
  let dog_walk_miles := dog_walk_distance * dog_walks_per_day * days_in_week
  let store_miles := store_distance * 2 * store_visits
  let friend_miles := friend_house_distance * 2 * friend_visits

  work_miles + dog_walk_miles + store_miles + friend_miles

theorem benjamin_weekly_miles :
  total_miles_walked = 95 := by
  sorry

end NUMINAMATH_CALUDE_benjamin_weekly_miles_l1381_138117
