import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l1160_116077

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + d * (n - 1)

noncomputable def arithmetic_sum (a d : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_common_difference
  (a d : ℝ) :
  arithmetic_sequence a d 2 + arithmetic_sequence a d 3 = 8 ∧
  arithmetic_sum a d 5 = 25 →
  d = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l1160_116077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vaccine_test_theorem_l1160_116000

/-- Represents the effectiveness of a vaccine sample -/
inductive Effectiveness
| Effective
| Ineffective

/-- Represents a group of vaccine samples -/
structure SampleGroup where
  effective : ℕ
  ineffective : ℕ

/-- Represents the vaccine test scenario -/
structure VaccineTest where
  totalSamples : ℕ
  groupA : SampleGroup
  groupB : SampleGroup
  groupC : SampleGroup
  probBEffective : ℚ
  stratifiedSampleSize : ℕ

def VaccineTest.isValid (test : VaccineTest) : Prop :=
  test.totalSamples = 2000 ∧
  test.groupA = ⟨673, 77⟩ ∧
  test.groupB.ineffective = 90 ∧
  test.probBEffective = 33/100 ∧
  test.stratifiedSampleSize = 360 ∧
  test.groupB.effective + test.groupB.ineffective + test.groupC.effective + test.groupC.ineffective = 
    test.totalSamples - (test.groupA.effective + test.groupA.ineffective) ∧
  test.groupC.effective ≥ 465 ∧
  test.groupC.ineffective ≥ 30

def VaccineTest.stratifiedSamplesFromC (test : VaccineTest) : ℕ :=
  (test.stratifiedSampleSize * (test.groupC.effective + test.groupC.ineffective)) / test.totalSamples

def VaccineTest.probabilityOfPassing (test : VaccineTest) : ℚ :=
  let totalEffective := test.groupA.effective + test.groupB.effective + test.groupC.effective
  if totalEffective ≥ (9 * test.totalSamples / 10) then 1 else 0

theorem vaccine_test_theorem (test : VaccineTest) (h : test.isValid) :
  test.stratifiedSamplesFromC = 90 ∧ test.probabilityOfPassing = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vaccine_test_theorem_l1160_116000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_function_intersection_g_is_y_function_of_f_l1160_116086

/-- Definition of Y function -/
def is_y_function (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

/-- The given quadratic function -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  (k/4) * x^2 + (k-1) * x + k - 3

/-- The Y function of f -/
noncomputable def g (k : ℝ) (x : ℝ) : ℝ :=
  (k/4) * x^2 - (k-1) * x + k - 3

/-- Theorem statement -/
theorem y_function_intersection (k : ℝ) :
  (∃! x, f k x = 0) →
  (∃ x, g k x = 0 ∧ (x = 3 ∨ x = 4)) := by
  sorry

/-- Proof that g is indeed the Y function of f -/
theorem g_is_y_function_of_f :
  ∀ k, is_y_function (f k) (g k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_function_intersection_g_is_y_function_of_f_l1160_116086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_club_overlap_l1160_116049

theorem sports_club_overlap (total : ℕ) (badminton tennis neither : ℕ) 
  (h1 : total = 100)
  (h2 : badminton = 60)
  (h3 : tennis = 70)
  (h4 : neither = 10)
  (h5 : total = badminton + tennis - (Finset.card (Finset.range badminton ∩ Finset.range tennis)) + neither) :
  Finset.card (Finset.range badminton ∩ Finset.range tennis) = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_club_overlap_l1160_116049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_E_l1160_116042

-- Define the expression
def E (a b c d : ℝ) : ℝ := a + 2*b + c + 2*d - a*b - b*c - c*d - d*a

-- Define the interval
def I : Set ℝ := Set.Icc (-4.5) 4.5

-- State the theorem
theorem max_value_E :
  ∃ (a b c d : ℝ), a ∈ I ∧ b ∈ I ∧ c ∈ I ∧ d ∈ I ∧ E a b c d = 90 ∧
  ∀ (x y z w : ℝ), x ∈ I → y ∈ I → z ∈ I → w ∈ I → E x y z w ≤ 90 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_E_l1160_116042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l1160_116058

-- Define the constants as noncomputable
noncomputable def a : ℝ := (1/2)^(1/2)
noncomputable def b : ℝ := (1/2)^(1/3)
noncomputable def c : ℝ := Real.log 2 / Real.log (1/2)

-- State the theorem
theorem order_of_abc : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l1160_116058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_trisects_median_implies_ratio_l1160_116039

/-- Represents a triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the median AM of a triangle -/
noncomputable def median (t : Triangle) : ℝ := sorry

/-- Predicate that checks if the incircle trisects the median -/
def incircle_trisects_median (t : Triangle) : Prop := sorry

/-- Theorem: If the incircle of a triangle trisects the median, then the ratio of its sides is 5:10:13 -/
theorem incircle_trisects_median_implies_ratio (t : Triangle) :
  incircle_trisects_median t → (t.a : ℝ) / 10 = (t.b : ℝ) / 13 ∧ (t.b : ℝ) / 13 = (t.c : ℝ) / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_trisects_median_implies_ratio_l1160_116039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_function_equality_l1160_116036

-- Define the functions h and k
noncomputable def h (x : ℝ) : ℝ := x - 3
noncomputable def k (x : ℝ) : ℝ := 2 * x

-- Define the inverse functions
noncomputable def h_inv (x : ℝ) : ℝ := x + 3
noncomputable def k_inv (x : ℝ) : ℝ := x / 2

-- State the theorem
theorem composite_function_equality :
  h (k_inv (h_inv (h_inv (k (h 28))))) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_function_equality_l1160_116036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_quadrants_l1160_116066

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a^x + b

-- State the theorem
theorem graph_quadrants (a b : ℝ) (ha : a > 1) (hb : b < -1) :
  ∃ (x₁ x₂ x₃ : ℝ),
    (f a b x₁ > 0 ∧ x₁ > 0) ∧  -- Quadrant I
    (f a b x₂ < 0 ∧ x₂ < 0) ∧  -- Quadrant III
    (f a b x₃ > 0 ∧ x₃ < 0) ∧  -- Quadrant IV
    (∀ x : ℝ, x > 0 → f a b x > 0)  -- Not in Quadrant II
    := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_quadrants_l1160_116066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_c_drain_rate_l1160_116069

/- Define the parameters of the problem -/
noncomputable def tank_capacity : ℝ := 1000
noncomputable def pipe_a_rate : ℝ := 200
noncomputable def pipe_a_time : ℝ := 1
noncomputable def pipe_b_rate : ℝ := 50
noncomputable def pipe_b_time : ℝ := 2
noncomputable def pipe_c_time : ℝ := 2
noncomputable def total_time : ℝ := 20

/- Define the cycle time -/
noncomputable def cycle_time : ℝ := pipe_a_time + pipe_b_time + pipe_c_time

/- Define the amount filled by pipes A and B in one cycle -/
noncomputable def filled_per_cycle : ℝ := pipe_a_rate * pipe_a_time + pipe_b_rate * pipe_b_time

/- Define the number of cycles -/
noncomputable def num_cycles : ℝ := total_time / cycle_time

/- Define the total amount filled by pipes A and B -/
noncomputable def total_filled : ℝ := filled_per_cycle * num_cycles

/- Define the amount drained by pipe C -/
noncomputable def amount_drained : ℝ := total_filled - tank_capacity

/- Define the total time pipe C is open -/
noncomputable def total_drain_time : ℝ := pipe_c_time * num_cycles

/- Theorem: The rate at which Pipe C drains the tank is 25 L/min -/
theorem pipe_c_drain_rate : 
  amount_drained / total_drain_time = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_c_drain_rate_l1160_116069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_principle_birthday_l1160_116068

def birthday (i : Fin 368) : Fin 365 := sorry

theorem pigeonhole_principle_birthday :
  ∃ (i j : Fin 368), i ≠ j ∧ (birthday i = birthday j) :=
by
  sorry

#check pigeonhole_principle_birthday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_principle_birthday_l1160_116068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_S_l1160_116019

-- Define the solid S
def S : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p;
                    (abs x + abs y ≤ 2) ∧ 
                    (abs x + abs z ≤ 2) ∧ 
                    (abs y + abs z ≤ 2)}

-- Define the volume of a set in ℝ³
noncomputable def volume (A : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem volume_of_S : volume S = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_S_l1160_116019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_at_seven_l1160_116046

/-- P is a polynomial of degree 8 with the given form and roots --/
def P (a b c d e f : ℝ) (x : ℂ) : ℂ :=
  (2 * x^4 - 26 * x^3 + a * x^2 + b * x + c) *
  (5 * x^4 - 80 * x^3 + d * x^2 + e * x + f)

/-- The set of roots of P includes 1, 2, 3, 4, 1/2, and 3/2 --/
axiom has_roots (a b c d e f : ℝ) :
  ∀ r : ℂ, r ∈ ({1, 2, 3, 4, (1:ℂ)/2, (3:ℂ)/2} : Set ℂ) → P a b c d e f r = 0

/-- P(7) = 0 --/
theorem P_at_seven (a b c d e f : ℝ) : P a b c d e f 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_at_seven_l1160_116046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ken_dawn_distance_l1160_116056

/-- The distance between Ken's house and Dawn's house -/
def distance_ken_dawn : ℝ := sorry

/-- The distance between Dawn's house and Mary's house -/
def distance_dawn_mary : ℝ := sorry

/-- Ken's house is twice as far from Dawn's house as Mary's house -/
axiom twice_distance : distance_ken_dawn = 2 * distance_dawn_mary

/-- The total distance of Ken's round trip is 12 miles -/
axiom total_distance : distance_ken_dawn + distance_dawn_mary + distance_dawn_mary + distance_ken_dawn = 12

/-- Theorem: The distance between Ken's house and Dawn's house is 4 miles -/
theorem ken_dawn_distance : distance_ken_dawn = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ken_dawn_distance_l1160_116056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_range_l1160_116084

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x > 0, Monotone (λ x => Real.log x / Real.log a)

def q (a : ℝ) : Prop := ∀ x, x^2 + a*x + 1 > 0

-- Define the range of a
def range_of_a : Set ℝ := Set.Ioc (-2) 1 ∪ Set.Ici 2

-- State the theorem
theorem proposition_range : 
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) ↔ range_of_a = Set.Ioc (-2) 1 ∪ Set.Ici 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_range_l1160_116084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1160_116029

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a.1 * k = b.1 ∧ a.2 * k = b.2

theorem parallel_vectors_lambda (l : ℝ) :
  are_parallel (2, 5) (l, 4) → l = 8/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1160_116029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_set_existence_l1160_116089

/-- A set of positive integers is "good" if for each element, its 2015th power
    is divisible by the product of all other elements in the set. -/
def is_good_set (A : Finset ℕ) : Prop :=
  ∀ a ∈ A, (a ^ 2015) % ((A.prod id) / a) = 0

/-- The main theorem stating the existence condition for "good" sets. -/
theorem good_set_existence (n : ℕ) :
  (∃ A : Finset ℕ, A.card = n ∧ A.Nonempty ∧ is_good_set A) ↔ 3 ≤ n ∧ n ≤ 2015 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_set_existence_l1160_116089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_ten_element_set_l1160_116096

theorem subsets_of_ten_element_set {α : Type*} (S : Finset α) (h : S.card = 10) : 
  (Finset.powerset S).card = 2^10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_ten_element_set_l1160_116096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l1160_116003

noncomputable def g (n : ℕ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((3 + Real.sqrt 3) / 6) ^ n +
  (3 - 2 * Real.sqrt 3) / 6 * ((3 - Real.sqrt 3) / 6) ^ n

theorem g_difference (n : ℕ) : g (n + 2) - g n = (1 / 4) * g n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l1160_116003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_17_integer_part_l1160_116001

-- Define the integer part function as noncomputable
noncomputable def integerPart (x : ℝ) : ℤ :=
  ⌊x⌋

-- State the theorem
theorem sqrt_17_integer_part :
  integerPart (-Real.sqrt 17 + 1) = -4 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_17_integer_part_l1160_116001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l1160_116074

noncomputable section

variable (a b x y z t : ℝ)

noncomputable def f (a b x y z t : ℝ) : ℝ := 
  (a * x^2 + b * y^2) / (a * x + b * y) + (a * z^2 + b * t^2) / (a * z + b * t)

theorem f_bounds (ha : a > 0) (hb : b > 0) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (ht : t ≥ 0)
  (hxz : x + z = 1) (hyt : y + t = 1) :
  1 ≤ f a b x y z t ∧ f a b x y z t ≤ 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l1160_116074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_perimeter_ratio_not_unique_l1160_116095

/-- Represents a triangle -/
structure Triangle where
  a : ℝ  -- side length a
  b : ℝ  -- side length b
  c : ℝ  -- side length c
  ha : a > 0
  hb : b > 0
  hc : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Calculate the area of a triangle using Heron's formula -/
noncomputable def Triangle.area (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- Calculate the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- The theorem stating that the area to perimeter ratio does not uniquely determine a triangle's shape -/
theorem area_perimeter_ratio_not_unique :
  ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧ 
    t1.area / t1.perimeter = t2.area / t2.perimeter := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_perimeter_ratio_not_unique_l1160_116095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_m_range_l1160_116057

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 - 2*x - 1

-- State the theorem
theorem tangent_line_and_m_range (a : ℝ) :
  (∃ (m : ℝ), (∀ x, f a x = m → (∃! (y z : ℝ), y ≠ x ∧ z ≠ x ∧ y ≠ z ∧ f a y = m ∧ f a z = m))) →
  (deriv (f a) 1 = 0) →
  (∃ (y : ℝ → ℝ), (∀ x, y x = -2*x - 1) ∧ (∀ x, (deriv (f a) 0) * x + f a 0 = y x)) ∧
  (∀ m : ℝ, -13/6 < m ∧ m < 7/3 ↔ (∃! (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = m ∧ f a y = m ∧ f a z = m)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_m_range_l1160_116057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l1160_116088

-- Define the side lengths of the isosceles triangle
variable (a b : ℝ)

-- Define the condition given in the problem
axiom side_lengths_condition : (a - 3)^2 + |b - 4| = 0

-- Define the isosceles property
axiom is_isosceles : (a = a ∧ b = a) ∨ (a = b ∧ b = b)

-- Define the triangle inequality
axiom triangle_inequality : a + b > a ∧ a + a > b ∧ b + b > a

-- Theorem to prove
theorem isosceles_triangle_perimeter :
  ∃ (p : ℝ), (p = 10 ∨ p = 11) ∧ p = a + b + min a b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l1160_116088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1160_116071

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 / (x + 3)

-- Theorem statement
theorem f_properties :
  -- Part 1: f is decreasing on (-3, +∞)
  (∀ x y : ℝ, -3 < x → x < y → f x > f y) ∧
  -- Part 2: Maximum value of f on [-1, 2] is 1
  (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → f x ≤ 1) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ f x = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1160_116071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_AMFD_is_49_4_l1160_116064

/-- Trapezoid ABCD with the given properties -/
structure Trapezoid where
  height : ℝ
  base_BC : ℝ
  base_AD : ℝ
  BE : ℝ
  midpoint_F : ℝ × ℝ
  intersection_M : ℝ × ℝ

/-- The specific trapezoid from the problem -/
def problem_trapezoid : Trapezoid where
  height := 5
  base_BC := 3
  base_AD := 5
  BE := 2
  midpoint_F := (0, 0)  -- Placeholder coordinates
  intersection_M := (0, 0)  -- Placeholder coordinates

/-- The area of quadrilateral AMFD in the given trapezoid -/
noncomputable def area_AMFD (t : Trapezoid) : ℝ := 49/4

/-- Theorem stating that the area of AMFD in the problem_trapezoid is 49/4 -/
theorem area_AMFD_is_49_4 : area_AMFD problem_trapezoid = 49/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_AMFD_is_49_4_l1160_116064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_intercept_l1160_116070

noncomputable def x : List ℝ := [1, 2, 3, 4, 5]
noncomputable def y : List ℝ := [0.5, 0.8, 1.0, 1.2, 1.5]

noncomputable def mean (list : List ℝ) : ℝ :=
  (list.sum) / (list.length : ℝ)

def linear_regression_slope : ℝ := 0.24

theorem linear_regression_intercept :
  let x_mean := mean x
  let y_mean := mean y
  y_mean = linear_regression_slope * x_mean + 0.28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_intercept_l1160_116070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_circle_radius_l1160_116017

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is tangent to two other circles and their common external tangent -/
def is_tangent_to_circles_and_tangent (c1 c2 c3 : Circle) : Prop :=
  ∃ (t : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ)),
    (∀ p q : ℝ × ℝ, t p q = t q p) ∧
    (t c1.center c2.center).1^2 + (t c1.center c2.center).2^2 = (c1.radius - c2.radius)^2 ∧
    (t c1.center c3.center).1^2 + (t c1.center c3.center).2^2 = (c1.radius + c3.radius)^2 ∧
    (t c2.center c3.center).1^2 + (t c2.center c3.center).2^2 = (c2.radius + c3.radius)^2

theorem third_circle_radius 
  (c1 c2 c3 : Circle)
  (h1 : c1.radius = 2)
  (h2 : c2.radius = 5)
  (h3 : are_externally_tangent c1 c2)
  (h4 : is_tangent_to_circles_and_tangent c1 c2 c3) :
  c3.radius = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_circle_radius_l1160_116017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_is_71_l1160_116002

/-- The sum of the first n natural numbers -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The list described in the problem -/
def special_list : List ℕ := sorry

/-- The total number of elements in the list -/
def total_elements : ℕ := sum_to_n 100

/-- The position of the median in the list -/
def median_position : ℕ := total_elements / 2 + 1

theorem median_is_71 : 
  ∃ (l : List ℕ), 
    l = special_list ∧ 
    l.length = total_elements ∧ 
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → l.count n = n) ∧
    l.get? (median_position - 1) = some 71 := by
  sorry

#eval total_elements
#eval median_position

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_is_71_l1160_116002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l1160_116030

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0
  | (n + 1) => sequence_a n + 4 * Real.sqrt (sequence_a n + 1) + 4

theorem sequence_a_closed_form (n : ℕ) (hn : n ≥ 1) : 
  sequence_a n = 4 * n^2 - 4 * n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l1160_116030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1160_116081

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  (∀ x, f (x + Real.pi) = f x) ∧ 
  (∀ x, f (2 * Real.pi / 3 - x) = f (2 * Real.pi / 3 + x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1160_116081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_plants_count_l1160_116025

theorem unique_plants_count (X Y Z : Finset ℕ) 
  (hX : X.card = 600)
  (hY : Y.card = 500)
  (hZ : Z.card = 400)
  (hXY : (X ∩ Y).card = 80)
  (hXZ : (X ∩ Z).card = 120)
  (hYZ : (Y ∩ Z).card = 70)
  (hXYZ : (X ∩ Y ∩ Z).card = 0) :
  (X ∪ Y ∪ Z).card = 1230 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_plants_count_l1160_116025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_m_value_l1160_116043

noncomputable section

-- Define the curve function
def f (x : ℝ) : ℝ := x^2 - 3 * Real.log x

-- Define the tangent line function
def g (m : ℝ) (x : ℝ) : ℝ := -x + m

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 2 * x - 3 / x

-- Theorem statement
theorem tangent_line_m_value :
  ∀ x₀ : ℝ, x₀ > 0 →
  (∃ m : ℝ, g m x₀ = f x₀ ∧ (-1 : ℝ) = f' x₀) →
  (∃ m : ℝ, g m x₀ = f x₀ ∧ (-1 : ℝ) = f' x₀ ∧ m = 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_m_value_l1160_116043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_l1160_116037

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5/8) (h2 : Real.sin (x - y) = 1/4) :
  Real.tan x / Real.tan y = 7/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_l1160_116037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_known_cards_l1160_116055

/-- Represents a card with a unique number -/
structure Card where
  number : ℕ
  unique : ℕ

/-- Represents the game setup -/
structure GameSetup where
  cards : Finset Card
  card_count : cards.card = 2013
  unique_numbers : ∀ c₁ c₂, c₁ ∈ cards → c₂ ∈ cards → c₁.number = c₂.number → c₁ = c₂

/-- Represents a move in the game -/
structure Move where
  selected : Finset Card
  selected_count : selected.card = 10
  revealed : Card

/-- Function that determines if a card's number is known -/
def is_known (c : Card) (moves : List Move) : Prop :=
  ∃ m, m ∈ moves ∧ m.revealed = c

/-- The main theorem to be proved -/
theorem max_known_cards (setup : GameSetup) :
  (∃ strategy : List Move, 
    (∀ c, c ∈ setup.cards → is_known c strategy → strategy.length ≤ 2013 - 27) ∧
    (∃ known : Finset Card, known.card = 1986 ∧ ∀ c, c ∈ known → is_known c strategy)) ∧
  (∀ strategy : List Move, 
    ∃ unknown : Finset Card, unknown.card = 27 ∧ ∀ c, c ∈ unknown → ¬is_known c strategy) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_known_cards_l1160_116055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_a_l1160_116061

-- Define a type for our sequence of positive integers
def PositiveIntegerSequence := ℕ → ℕ+

-- Function to calculate the sum of digits in base 4038
def sumOfDigitsBase4038 (n : ℕ+) : ℕ :=
  sorry

-- Predicate to check if a sequence satisfies the given condition
def satisfiesCondition (a : ℝ) (seq : PositiveIntegerSequence) : Prop :=
  (∀ n : ℕ, (seq n : ℝ) ≤ a * n) ∧
  (∀ m : ℕ, ∃ n > m, ¬ 2019 ∣ sumOfDigitsBase4038 (seq n))

-- Main theorem
theorem characterization_of_a : ∀ a : ℝ, 
  (∀ seq : PositiveIntegerSequence, Function.Injective seq → satisfiesCondition a seq) ↔ 
  (1 ≤ a ∧ a < 2019) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_a_l1160_116061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l1160_116041

/-- The circle with center (1, 0) and radius √5 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 5}

/-- The line x + 2y - 6 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + 2*p.2 - 6 = 0}

/-- The point (2, 2) -/
def P : ℝ × ℝ := (2, 2)

theorem tangent_line_to_circle :
  P ∈ Circle ∧ P ∈ Line ∧
  ∀ Q : ℝ × ℝ, Q ∈ Circle ∩ Line → Q = P :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l1160_116041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l1160_116012

/-- A function f: ℝ → ℝ defined as f(x) = 8^x -/
noncomputable def f (x : ℝ) : ℝ := 8^x

/-- Theorem stating that f(x+1) - f(x) = 7*f(x) for all real x -/
theorem f_difference (x : ℝ) : f (x + 1) - f x = 7 * f x := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the exponential expression
  simp [Real.rpow_add]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l1160_116012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_condition_l1160_116013

-- Define the function f(x) = (x+3)^2 / x
noncomputable def f (x : ℝ) : ℝ := (x + 3)^2 / x

-- Define the property of having a unique root
def has_unique_root (a : ℝ) : Prop :=
  ∃! x : ℝ, x > 0 ∧ (x + 3)^2 = a * x

-- State the theorem
theorem unique_root_condition (a : ℝ) :
  has_unique_root a ↔ a = 12 ∨ a < 0 := by
  sorry

#check unique_root_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_condition_l1160_116013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l1160_116006

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - b)

theorem b_range (h1 : ∀ x, f (-x) = -f x)
                (h2 : ∀ x₁ x₂, f x₁ ≤ g b x₂) :
  b ∈ Set.Iic (-Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l1160_116006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1160_116027

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the vectors m and n
noncomputable def m (t : Triangle) : Fin 2 → ℝ
  | 0 => Real.sin t.B
  | 1 => -2 * Real.sin t.A
  | _ => 0

noncomputable def n (t : Triangle) : Fin 2 → ℝ
  | 0 => Real.sin t.B
  | 1 => Real.sin t.C
  | _ => 0

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- Define the theorem
theorem triangle_properties (t : Triangle) :
  (t.a = t.b ∧ dot_product (m t) (n t) = 0 → Real.cos t.B = 1/4) ∧
  (t.B = Real.pi/2 ∧ t.a = Real.sqrt 2 ∧ t.b^2 = 2*t.a*t.c → 
   (1/2) * t.b * t.c = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1160_116027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_with_cap_l1160_116073

-- Define the possible colors for shorts and jerseys
inductive ShortsColor
| Black
| Gold
| Red
deriving Fintype, Repr

inductive JerseyColor
| Black
| White
| Gold
| Blue
deriving Fintype, Repr

-- Define the cap choice
inductive CapChoice
| Yes
| No
deriving Fintype, Repr

-- Define the uniform configuration
structure UniformConfig where
  shorts : ShortsColor
  jersey : JerseyColor
  cap : CapChoice

def is_different_color (config : UniformConfig) : Prop :=
  match config.shorts, config.jersey with
  | ShortsColor.Black, JerseyColor.Black => False
  | ShortsColor.Gold, JerseyColor.Gold => False
  | _, _ => True

def cap_included (config : UniformConfig) : Prop :=
  config.cap = CapChoice.Yes

-- The main theorem
theorem different_color_probability_with_cap :
  let total_configs := (Fintype.card ShortsColor) * (Fintype.card JerseyColor) * (Fintype.card CapChoice)
  let different_color_configs := 10 * (Fintype.card CapChoice)
  let cap_included_configs := (Fintype.card ShortsColor) * (Fintype.card JerseyColor)
  (different_color_configs / cap_included_configs : ℚ) = 5 / 6 :=
by
  -- Proof goes here
  sorry

#eval Fintype.card ShortsColor
#eval Fintype.card JerseyColor
#eval Fintype.card CapChoice

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_with_cap_l1160_116073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_length_l1160_116032

-- Define the parabola C
def C (x y : ℝ) : Prop := y^2 = 6*x

-- Define the focus of the parabola
noncomputable def focus : ℝ × ℝ := (3/2, 0)

-- Define the line l
def l (x y : ℝ) : Prop := y = x - 3/2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  C A.1 A.2 ∧ C B.1 B.2 ∧ l A.1 A.2 ∧ l B.1 B.2

-- Theorem statement
theorem parabola_line_intersection_length :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_length_l1160_116032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1160_116028

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + Real.sqrt (5 - x)

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 5, f x ≤ |m - 2|) →
  m ∈ Set.Iic (-3) ∪ Set.Ici 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1160_116028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_germination_mode_l1160_116054

/-- The binomial probability mass function -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The mode of a binomial distribution -/
def binomial_mode (n : ℕ) (p : ℝ) : Set ℕ :=
  {k : ℕ | ∀ j : ℕ, binomial_pmf n p k ≥ binomial_pmf n p j}

theorem germination_mode :
  let n : ℕ := 9
  let p : ℝ := 0.8
  binomial_mode n p = {7, 8} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_germination_mode_l1160_116054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1160_116065

noncomputable def a (x : ℝ) : ℝ × ℝ := (-Real.sin x, 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (1, Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

noncomputable def g (x : ℝ) : ℝ := 
  (Real.sin (Real.pi + x) + 4 * Real.cos (2*Real.pi - x)) / 
  (Real.sin (Real.pi/2 - x) - 4 * Real.sin (-x))

theorem problem_solution :
  (f (Real.pi/6) = Real.sqrt 3 - 1/2) ∧
  (∀ x, (a x).1 * (b x).1 + (a x).2 * (b x).2 = 0 → g x = 2/9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1160_116065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1160_116009

/-- Sum of first n terms of a geometric sequence with first term a and common ratio q -/
noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_ratio (a : ℝ) (q : ℝ) :
  (2 * geometric_sum a q 4 = geometric_sum a q 5 + geometric_sum a q 6) → q = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1160_116009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_sharing_proof_l1160_116059

noncomputable def total_cost : ℝ := 80 + 150 + 120 + 200
noncomputable def individual_share : ℝ := total_cost / 4
def jack_payment : ℝ := 80
def emma_payment : ℝ := 150
def noah_payment : ℝ := 120
def liam_payment : ℝ := 200

noncomputable def jack_owes : ℝ := individual_share - jack_payment
noncomputable def noah_owes : ℝ := individual_share - noah_payment

theorem cost_sharing_proof :
  jack_owes - noah_owes = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_sharing_proof_l1160_116059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_approx_l1160_116034

/-- The area of a circular sector with given radius and central angle -/
noncomputable def sectorArea (radius : ℝ) (angle : ℝ) : ℝ :=
  (angle / 360) * Real.pi * radius^2

/-- Theorem: The area of a circular sector with radius 12 meters and central angle 39° 
    is approximately 48.943 square meters -/
theorem sector_area_approx :
  let r : ℝ := 12
  let θ : ℝ := 39
  abs (sectorArea r θ - 48.943) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_approx_l1160_116034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_between_sqrt8_and_sqrt80_l1160_116024

theorem whole_numbers_between_sqrt8_and_sqrt80 : 
  (Finset.range (Int.toNat (⌊Real.sqrt 80⌋ - ⌈Real.sqrt 8⌉ + 1))).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_between_sqrt8_and_sqrt80_l1160_116024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_in_cone_l1160_116092

/-- Represents a right circular cone with a 90-degree vertex angle -/
structure RightCircularCone :=
  (baseDiameter : ℝ)

/-- Represents a sphere inside the cone -/
structure SphereInCone :=
  (radius : ℝ)

/-- The volume of a sphere given its radius -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

theorem sphere_volume_in_cone (cone : RightCircularCone) (sphere : SphereInCone) :
  cone.baseDiameter = 24 →
  sphere.radius = cone.baseDiameter / 4 →
  sphereVolume sphere.radius = 288 * Real.pi :=
by
  intros h1 h2
  simp [sphereVolume, h1, h2]
  sorry  -- We skip the actual proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_in_cone_l1160_116092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtyfourth_term_is_one_hundredth_l1160_116093

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define the base case for n = 0 (which corresponds to a₁)
  | (n + 1) => sequence_a n / (3 * sequence_a n + 1)

theorem thirtyfourth_term_is_one_hundredth :
  sequence_a 33 = 1 / 100 := by
  sorry

-- Optional: Helper lemma to show the reciprocal relation
lemma reciprocal_relation (n : ℕ) :
  1 / sequence_a (n + 1) = 3 + 1 / sequence_a n := by
  sorry

-- Optional: Helper lemma to show the arithmetic sequence property
lemma arithmetic_sequence (n : ℕ) :
  1 / sequence_a n = 3 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtyfourth_term_is_one_hundredth_l1160_116093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existential_tangent_proposition_sine_bounded_sine_supplementary_angles_sine_15_identity_l1160_116091

open Real

theorem existential_tangent_proposition :
  ∃ α : ℝ, ∃ y : ℝ, Real.tan (π/2 - α) = y :=
by
  -- Proof goes here
  sorry

-- Additional definitions to represent the given conditions
def tan_45_defined : ∃ y : ℝ, Real.tan (π/4) = y := by
  sorry

theorem sine_bounded (x : ℝ) : Real.sin x ≤ 1 := by
  sorry

theorem sine_supplementary_angles (α : ℝ) : Real.sin (π - α) = Real.sin α := by
  sorry

theorem sine_15_identity : 
  Real.sin (π/12) = Real.sin (π/3) * Real.cos (π/4) - Real.cos (π/3) * Real.sin (π/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existential_tangent_proposition_sine_bounded_sine_supplementary_angles_sine_15_identity_l1160_116091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1160_116007

-- Define the function as noncomputable due to Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (12 + x - x^2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -3 ≤ x ∧ x ≤ 4} :=
by
  -- The proof is omitted and replaced with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1160_116007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_t_range_l1160_116082

/-- Given a function f(x) = 2^(x^2 - 2x), if its maximum value in the interval [-1,t] is 8,
    then t is in the range (-1,3]. -/
theorem max_value_implies_t_range (t : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) t, (2 : ℝ)^(x^2 - 2*x) ≤ 8) ∧ 
  (∃ x ∈ Set.Icc (-1 : ℝ) t, (2 : ℝ)^(x^2 - 2*x) = 8) →
  t ∈ Set.Ioo (-1 : ℝ) 3 ∪ {3} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_t_range_l1160_116082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l1160_116097

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in the compound -/
def carbon_atoms : ℕ := 7

/-- The number of Hydrogen atoms in the compound -/
def hydrogen_atoms : ℕ := 6

/-- The number of Oxygen atoms in the compound -/
def oxygen_atoms : ℕ := 2

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 
  (carbon_atoms : ℝ) * carbon_weight + 
  (hydrogen_atoms : ℝ) * hydrogen_atoms + 
  (oxygen_atoms : ℝ) * oxygen_weight

theorem compound_molecular_weight : 
  |molecular_weight - 122.118| < 0.001 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l1160_116097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_hat_colors_l1160_116016

-- Define the colors
inductive Color
| Pink
| Purple
| Turquoise
deriving Repr, BEq, Inhabited

-- Define a structure for a person's outfit
structure Outfit :=
  (dress : Color)
  (hat : Color)
deriving Repr, BEq, Inhabited

-- Define the problem
theorem dress_hat_colors :
  ∃! (vera nadia lyuba : Outfit),
    -- Only Vera's dress and hat are the same color
    vera.dress = vera.hat ∧
    -- Nadia's dress and hat are not pink
    nadia.dress ≠ Color.Pink ∧ nadia.hat ≠ Color.Pink ∧
    -- Lyuba's hat is purple
    lyuba.hat = Color.Purple ∧
    -- All dresses and hats use different colors
    ({vera.dress, nadia.dress, lyuba.dress} : Set Color) = {Color.Pink, Color.Purple, Color.Turquoise} ∧
    ({vera.hat, nadia.hat, lyuba.hat} : Set Color) = {Color.Pink, Color.Purple, Color.Turquoise} ∧
    -- The solution
    vera = Outfit.mk Color.Pink Color.Pink ∧
    nadia = Outfit.mk Color.Purple Color.Turquoise ∧
    lyuba = Outfit.mk Color.Turquoise Color.Purple :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_hat_colors_l1160_116016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_reach_target_expected_y_at_target_l1160_116085

/-- A random walk on an integer grid where each step is either right or up with equal probability -/
def RandomWalk := ℕ → ℕ × ℕ

/-- The probability of moving right at each step -/
noncomputable def p_right : ℝ := 1/2

/-- The probability of moving up at each step -/
noncomputable def p_up : ℝ := 1/2

/-- The x-coordinate to be reached -/
def target_x : ℕ := 2011

/-- Theorem stating that the random walk will eventually reach the target x-coordinate -/
theorem eventually_reach_target (walk : RandomWalk) :
  ∃ n : ℕ, (walk n).1 = target_x := by sorry

/-- Theorem stating the expected y-coordinate when reaching the target x-coordinate -/
theorem expected_y_at_target (walk : RandomWalk) :
  Finset.sum (Finset.range (target_x + 1)) (fun n => n * (Nat.choose target_x n * (p_right ^ (target_x - n) * p_up ^ n))) = target_x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_reach_target_expected_y_at_target_l1160_116085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_theorem_l1160_116031

theorem sum_remainder_theorem (a b c d e : ℕ) : 
  a % 13 = 3 → b % 13 = 5 → c % 13 = 7 → d % 13 = 9 → e % 13 = 11 → 
  (a + b + c + d + e) % 13 = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_theorem_l1160_116031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_calculate_expression_l1160_116026

open Real

-- Define the second quadrant
def SecondQuadrant (α : ℝ) : Prop := 0 < α ∧ α < π ∧ cos α < 0 ∧ sin α > 0

-- Statement 1
theorem simplify_expression (α : ℝ) (h : SecondQuadrant α) :
  cos α * sqrt ((1 - sin α) / (1 + sin α)) + sin α * sqrt ((1 - cos α) / (1 + cos α)) = sin α - cos α :=
by sorry

-- Statement 2
theorem calculate_expression :
  cos (25 * π / 6) + cos (25 * π / 3) + tan (-(25 * π / 4)) + sin (5 * π / 6) = sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_calculate_expression_l1160_116026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_at_2001_l1160_116004

noncomputable def f (n : ℕ) : ℝ := (n^2 : ℝ) / 1.001^n

theorem max_value_at_2001 :
  ∀ k : ℕ, f 2001 ≥ f k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_at_2001_l1160_116004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sum_of_roots_l1160_116040

theorem irrational_sum_of_roots (α : ℝ) (n : ℕ+) 
  (h_irr : Irrational α) :
  Irrational ((α + Real.sqrt (α^2 - 1))^(1/(n : ℝ)) + 
              (α - Real.sqrt (α^2 - 1))^(1/(n : ℝ))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sum_of_roots_l1160_116040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_parabola_l1160_116035

/-- The value of c for which the line 2x - y + c = 0 is tangent to the parabola x^2 = 4y -/
theorem tangent_line_to_parabola (c : ℝ) :
  (∀ x y : ℝ, x^2 = 4*y ∧ 2*x - y + c = 0 → 
    ∃! p : ℝ × ℝ, p.1^2 = 4*p.2 ∧ 2*p.1 - p.2 + c = 0) →
  c = -4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_parabola_l1160_116035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_line_ab_range_bisecting_line_ab_lower_bound_l1160_116063

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the line
def my_line (a b x y : ℝ) : Prop := a*x - b*y + 1 = 0

-- Define the property of bisecting the circumference
def bisects_circle (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), my_circle x y ∧ my_line a b x y

-- Theorem statement
theorem bisecting_line_ab_range :
  ∀ (a b : ℝ), bisects_circle a b → (∀ t : ℝ, t > 1/8 → a*b < t) := by
  sorry

-- Additional theorem for the lower bound
theorem bisecting_line_ab_lower_bound :
  ¬∃ (M : ℝ), ∀ (a b : ℝ), bisects_circle a b → a*b ≥ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_line_ab_range_bisecting_line_ab_lower_bound_l1160_116063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_problem_solve_inequality_system_l1160_116083

-- Part 1: Factorization
theorem factorization_problem (a x y : ℝ) : 
  -8*a*x^2 + 16*a*x*y - 8*a*y^2 = -8*a*(x-y)^2 := by sorry

-- Part 2: Inequality System
def inequality_system (x : ℝ) : Prop :=
  (2*x - 7 < 3*(x-1)) ∧ (4/3*x + 3 ≥ 1 - 2/3*x)

theorem solve_inequality_system :
  {x : ℝ | inequality_system x} = {x : ℝ | x ≥ -1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_problem_solve_inequality_system_l1160_116083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_k_range_l1160_116067

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Triangle inequality
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  -- Angles are positive and sum to π
  angle_sum : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi
  -- Law of sines
  law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- Define the property that angles form an arithmetic sequence
def arithmetic_sequence (t : Triangle) : Prop :=
  ∃ d : ℝ, t.B - t.A = d ∧ t.C - t.B = d

-- Define the property a^2 + c^2 = kb^2
def side_relation (t : Triangle) (k : ℝ) : Prop :=
  t.a^2 + t.c^2 = k * t.b^2

-- Main theorem
theorem triangle_k_range (t : Triangle) (k : ℝ) 
  (h1 : arithmetic_sequence t) 
  (h2 : side_relation t k) : 
  1 < k ∧ k ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_k_range_l1160_116067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l1160_116044

-- Define the ellipse type
structure Ellipse where
  a : ℝ
  b : ℝ
  center : ℝ × ℝ

-- Define the set of points on the ellipse
def ellipse_points (e : Ellipse) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - e.center.1)^2 / e.a^2 + (p.2 - e.center.2)^2 / e.b^2 = 1}

-- Define the conditions
def ellipse_conditions (e : Ellipse) : Prop :=
  e.center = (0, 0) ∧ e.a = 3 * e.b ∧ (3, 0) ∈ ellipse_points e

-- Define the standard equations
def standard_equation_x_axis (e : Ellipse) : Prop :=
  ∀ x y, (x, y) ∈ ellipse_points e ↔ x^2 / 9 + y^2 = 1

def standard_equation_y_axis (e : Ellipse) : Prop :=
  ∀ x y, (x, y) ∈ ellipse_points e ↔ x^2 / 9 + y^2 / 81 = 1

-- Theorem statement
theorem ellipse_standard_equation (e : Ellipse) :
  ellipse_conditions e →
  (standard_equation_x_axis e ∨ standard_equation_y_axis e) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l1160_116044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operation_result_vector_linear_combination_unique_scalars_l1160_116021

/-- Given vectors in R^2 -/
noncomputable def a : Fin 2 → ℝ := ![3, 2]
noncomputable def b : Fin 2 → ℝ := ![-1, 2]
noncomputable def c : Fin 2 → ℝ := ![4, 1]

/-- The result of vector operation 3a + b - 2c -/
noncomputable def result : Fin 2 → ℝ := ![0, 6]

/-- The scalar values m and n -/
noncomputable def m : ℝ := 5/9
noncomputable def n : ℝ := 8/9

/-- Theorem stating that 3a + b - 2c equals the result -/
theorem vector_operation_result : 
  (3 • a) + b - (2 • c) = result := by sorry

/-- Theorem stating that a = mb + nc -/
theorem vector_linear_combination : 
  a = m • b + n • c := by sorry

/-- Theorem stating the uniqueness of m and n -/
theorem unique_scalars (m' n' : ℝ) : 
  a = m' • b + n' • c → m' = m ∧ n' = n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operation_result_vector_linear_combination_unique_scalars_l1160_116021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_theorem_initial_machines_count_l1160_116053

/-- Represents the production rate of a single machine in units per day -/
noncomputable def machine_rate (x : ℝ) : ℝ := x / 120

/-- Represents the number of machines in the initial scenario -/
def initial_machines (x : ℝ) : ℕ := 5

/-- The production theorem stating the relationships between machines, time, and output -/
theorem production_theorem (x : ℝ) (h_pos : x > 0) :
  (initial_machines x : ℝ) * machine_rate x * 8 = x ∧
  30 * machine_rate x * 4 = 3 * x :=
by sorry

/-- The main theorem proving that the number of initial machines is 5 -/
theorem initial_machines_count (x : ℝ) (h_pos : x > 0) :
  initial_machines x = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_theorem_initial_machines_count_l1160_116053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_traveled_velocity_change_magnitude_l1160_116023

-- Define the robot's motion
def robot_x (t : ℝ) : ℝ := (t - 6)^2
noncomputable def robot_y (t : ℝ) : ℝ := if t ≥ 7 then (t - 7)^2 else 0

-- Define the velocity components
def velocity_x (t : ℝ) : ℝ := 2 * (t - 6)
noncomputable def velocity_y (t : ℝ) : ℝ := if t ≥ 7 then 2 * (t - 7) else 0

-- Theorem for the distance traveled in the first 7 minutes
theorem distance_traveled : abs (robot_x 7 - robot_x 0) = 35 := by sorry

-- Theorem for the magnitude of velocity change in the 8th minute
theorem velocity_change_magnitude : 
  Real.sqrt ((velocity_x 8 - velocity_x 7)^2 + (velocity_y 8 - velocity_y 7)^2) = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_traveled_velocity_change_magnitude_l1160_116023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1160_116098

noncomputable def f (x : ℝ) : ℝ := Real.cos x + abs (Real.sin x)

theorem f_properties :
  (∀ x, f x = f (-x)) ∧
  (∀ x y, π / 4 < x ∧ x < y ∧ y < π → f y ≤ f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1160_116098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_equality_sine_equivalence_l1160_116010

noncomputable section

open Real

theorem triangle_side_equality_sine_equivalence (A B C : ℝ) (a b c : ℝ) :
  let triangle_condition := 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π
  let side_angle_relation := a / sin A = b / sin B ∧ b / sin B = c / sin C
  triangle_condition ∧ side_angle_relation →
  (sin A * sin (A/2 + B) = sin B * sin (B/2 + A)) ↔ (a = b) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_equality_sine_equivalence_l1160_116010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_and_angle_l1160_116045

theorem orthogonal_vectors_and_angle (θ φ : Real) : 
  θ ∈ Set.Ioo 0 (π/2) →
  φ ∈ Set.Ioo 0 (π/2) →
  (Real.sin θ) * 1 + (-2) * (Real.cos θ) = 0 →
  5 * Real.cos (θ - φ) = 3 * Real.sqrt 5 * Real.cos φ →
  Real.sin θ = 2 * Real.sqrt 5 / 5 ∧ 
  Real.cos θ = Real.sqrt 5 / 5 ∧ 
  φ = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_and_angle_l1160_116045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_theorem_l1160_116005

noncomputable def f (x : ℝ) : ℝ := x^2 + 4/x^2 - 3

def g (k x : ℝ) : ℝ := k*x + 2

theorem k_range_theorem (k : ℝ) :
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₂ ∈ Set.Icc 1 (Real.sqrt 3), g k x₁ > f x₂) →
  k ∈ Set.Ioo (-1/2 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_theorem_l1160_116005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l1160_116075

theorem triangle_area_range (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ A < Real.pi →
  B > 0 ∧ B < Real.pi →
  C > 0 ∧ C < Real.pi →
  A + B + C = Real.pi →
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  a = 3 →
  let S := (1/2) * b * c * Real.sin A
  0 < S ∧ S ≤ (9 * Real.sqrt 3) / 4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l1160_116075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_2_l1160_116099

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^2017 + a*x^3 - b/x - 8

theorem f_value_at_2 (a b : ℝ) :
  f a b (-2) = 10 → f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_2_l1160_116099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l1160_116047

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 3*a) / Real.log (1/2)

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, 2 ≤ x ∧ x < y → f a y < f a x) ↔ -4 < a ∧ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l1160_116047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1160_116072

open Function Real

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_properties (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h1 : ∀ x, f (4 + x) - f (-x) = 0)
  (h2 : is_odd (fun x ↦ f (x + 1)))
  (h3 : f' 1 = -1)
  (h4 : ∀ x, (deriv f) x = f' x) :
  f 1 = 0 ∧ (∀ x, f (x + 4) = f x) ∧ f' 2023 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1160_116072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_with_diagonal_l1160_116051

/-- Definition: A point is on a line defined by two points -/
def OnLine (P A B : Point) : Prop := sorry

/-- Definition: A quadrilateral is a rectangle -/
def Rectangle (P Q R S : Point) : Prop := sorry

/-- Definition: The length of a line segment between two points -/
noncomputable def SegmentLength (P Q : Point) : ℝ := sorry

/-- Given a triangle and a length, there exists an inscribed rectangle with a diagonal of that length -/
theorem inscribed_rectangle_with_diagonal (A B C : Point) (d : ℝ) : 
  ∃ (P Q R S : Point), 
    Rectangle P Q R S ∧ 
    OnLine R A B ∧ 
    OnLine Q B C ∧ 
    OnLine P A C ∧ 
    OnLine S A C ∧ 
    SegmentLength P Q = d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_with_diagonal_l1160_116051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_is_80_minutes_l1160_116090

/-- Represents the travel scenario with given conditions -/
structure TravelScenario where
  freeway_distance : ℚ
  mountain_distance : ℚ
  speed_ratio : ℚ
  mountain_time : ℚ

/-- Calculates the total travel time given a TravelScenario -/
def total_travel_time (scenario : TravelScenario) : ℚ :=
  scenario.mountain_time + (scenario.freeway_distance / (scenario.speed_ratio * scenario.mountain_distance / scenario.mountain_time))

/-- Theorem stating that under the given conditions, the total travel time is 80 minutes -/
theorem travel_time_is_80_minutes (scenario : TravelScenario) 
    (h1 : scenario.freeway_distance = 80)
    (h2 : scenario.mountain_distance = 20)
    (h3 : scenario.speed_ratio = 4)
    (h4 : scenario.mountain_time = 40) : 
  total_travel_time scenario = 80 := by
  sorry

def main : IO Unit := do
  let scenario : TravelScenario := {
    freeway_distance := 80,
    mountain_distance := 20,
    speed_ratio := 4,
    mountain_time := 40
  }
  IO.println s!"Total travel time: {total_travel_time scenario}"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_is_80_minutes_l1160_116090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bases_solutions_l1160_116038

theorem trapezoid_bases_solutions :
  ∃ (S : Finset (ℤ × ℤ)), 
    (∀ (b₁ b₂ : ℤ), (b₁, b₂) ∈ S ↔ 
      (10 ∣ b₁) ∧ 
      (10 ∣ b₂) ∧ 
      (b₁ + b₂ = 60) ∧ 
      ((b₁ * 60 + b₂ * 60) / 2 = 1800)) ∧
    (Finset.card S > 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bases_solutions_l1160_116038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l1160_116015

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (3 + Complex.I) - m * (2 + Complex.I)

-- Theorem statement
theorem point_in_fourth_quadrant (m : ℝ) (h : 1 < m ∧ m < 3/2) : 
  (z m).re > 0 ∧ (z m).im < 0 :=
by
  sorry

-- Note: (z m).re > 0 ∧ (z m).im < 0 represents a point in the fourth quadrant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l1160_116015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_price_per_jin_l1160_116094

/-- Represents a price in yuan and jiao -/
structure Price where
  yuan : ℕ
  jiao : ℕ
  jiao_valid : jiao < 10

/-- Converts a Price to a real number -/
noncomputable def Price.toReal (p : Price) : ℝ :=
  p.yuan + p.jiao / 10

/-- The price of 5 jin of cucumbers -/
def cucumber_price : Price :=
  { yuan := 11, jiao := 8, jiao_valid := by norm_num }

/-- The additional cost of 4 jin of tomatoes compared to 5 jin of cucumbers -/
def additional_cost : Price :=
  { yuan := 1, jiao := 4, jiao_valid := by norm_num }

/-- Theorem stating the price per jin of tomatoes -/
theorem tomato_price_per_jin :
  ∃ (p : Price), p.toReal = (cucumber_price.toReal + additional_cost.toReal) / 4 ∧
  p.yuan = 3 ∧ p.jiao = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_price_per_jin_l1160_116094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_in_interval_l1160_116022

-- Define the function f(x) = e^x - x
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

-- State the theorem
theorem max_value_of_f_in_interval :
  ∃ (max : ℝ), max = Real.exp 1 - 1 ∧
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f x ≤ max :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_in_interval_l1160_116022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1160_116079

theorem min_value_theorem (a b : ℝ) (h1 : a > 2*b) 
  (h2 : Set.range (fun x => a*x^2 + x + 2*b) = Set.Ici 0) :
  ∃ (m : ℝ), m = Real.sqrt 2 ∧ ∀ x, x ≥ m ↔ ∃ (a b : ℝ), 
    a > 2*b ∧ 
    Set.range (fun x => a*x^2 + x + 2*b) = Set.Ici 0 ∧
    x = (a^2 + 4*b^2) / (a - 2*b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1160_116079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_implies_specific_coordinates_l1160_116078

/-- Given points A, B, and C in a 2D Cartesian coordinate system, 
    prove that if OBC is similar to ABO, then C has specific coordinates. -/
theorem triangle_similarity_implies_specific_coordinates :
  ∀ (x : ℝ),
  x > 0 →
  let A : ℝ × ℝ := (-4, -2)
  let B : ℝ × ℝ := (0, -2)
  let C : ℝ × ℝ := (x, 0)
  let O : ℝ × ℝ := (0, 0)
  (∃ (k : ℝ), k > 0 ∧ 
    (C.1 - O.1) = k * (B.1 - O.1) ∧
    (C.2 - O.2) = k * (B.2 - O.2) ∧
    (B.1 - O.1) = k * (A.1 - O.1) ∧
    (B.2 - O.2) = k * (A.2 - O.2)) →
  x = 1 ∨ x = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_implies_specific_coordinates_l1160_116078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_worth_calculation_l1160_116033

/-- Represents the total worth of the stock in Rupees -/
def total_stock : ℝ := sorry

/-- The proportion of stock sold at a profit -/
def profit_portion : ℝ := 0.2

/-- The profit percentage on the portion sold at profit -/
def profit_percentage : ℝ := 0.1

/-- The proportion of stock sold at a loss -/
def loss_portion : ℝ := 0.8

/-- The loss percentage on the portion sold at loss -/
def loss_percentage : ℝ := 0.05

/-- The overall loss in Rupees -/
def overall_loss : ℝ := 400

theorem stock_worth_calculation :
  profit_portion * profit_percentage * total_stock - 
  loss_portion * loss_percentage * total_stock = overall_loss →
  total_stock = 20000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_worth_calculation_l1160_116033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_equation_IV_has_nontrivial_solution_l1160_116080

/-- Represents a complex number -/
def a : ℂ := sorry
def b : ℂ := sorry

/-- Equation I: a² + b² = 0 -/
def equation_I (a b : ℂ) : Prop := a^2 + b^2 = 0

/-- Equation II: e^(a+b) = ab -/
def equation_II (a b : ℂ) : Prop := Complex.exp (a + b) = a * b

/-- Equation III: sin(a² + b²) = a + b -/
def equation_III (a b : ℂ) : Prop := Complex.sin (a^2 + b^2) = a + b

/-- Equation IV: ab² = ba² -/
def equation_IV (a b : ℂ) : Prop := a * b^2 = b * a^2

/-- Theorem: Only equation IV has non-trivial complex solutions -/
theorem only_equation_IV_has_nontrivial_solution :
  (∃ a b : ℂ, a ≠ 0 ∨ b ≠ 0) ∧ equation_IV a b ∧
  (∀ a b : ℂ, a ≠ 0 ∨ b ≠ 0 → ¬equation_I a b) ∧
  (∀ a b : ℂ, a ≠ 0 ∨ b ≠ 0 → ¬equation_II a b) ∧
  (∀ a b : ℂ, a ≠ 0 ∨ b ≠ 0 → ¬equation_III a b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_equation_IV_has_nontrivial_solution_l1160_116080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_tax_is_simplified_income_tax_example_minimum_tax_l1160_116011

/-- Represents the financial data of a business -/
structure BusinessData where
  annual_income : ℚ
  annual_expenses : ℚ
  expense_payment_ratio : ℚ
  main_tax_rate : ℚ
  simplified_income_tax_rate : ℚ
  simplified_profit_tax_rate : ℚ
  minimum_tax_rate : ℚ

/-- Calculates the tax under the main taxation system -/
def main_system_tax (data : BusinessData) : ℚ :=
  (data.annual_income - data.annual_expenses) * data.main_tax_rate

/-- Calculates the tax under the simplified taxation system (income) -/
def simplified_income_tax (data : BusinessData) : ℚ :=
  data.annual_income * data.simplified_income_tax_rate

/-- Calculates the tax under the simplified taxation system (income minus expenses) -/
def simplified_profit_tax (data : BusinessData) : ℚ :=
  max ((data.annual_income - data.annual_expenses * data.expense_payment_ratio) * data.simplified_profit_tax_rate)
      (data.annual_income * data.minimum_tax_rate)

/-- Theorem stating that the minimum tax is the simplified income tax -/
theorem minimum_tax_is_simplified_income_tax (data : BusinessData) :
  let main_tax := main_system_tax data
  let income_tax := simplified_income_tax data
  let profit_tax := simplified_profit_tax data
  income_tax ≤ main_tax ∧ income_tax ≤ profit_tax := by
  sorry

/-- Example data based on the problem -/
def example_data : BusinessData := {
  annual_income := 4500000
  annual_expenses := 3636000
  expense_payment_ratio := 45/100
  main_tax_rate := 20/100
  simplified_income_tax_rate := 6/100
  simplified_profit_tax_rate := 15/100
  minimum_tax_rate := 1/100
}

/-- Theorem proving the minimum tax for the example data -/
theorem example_minimum_tax :
  simplified_income_tax example_data = 135000 ∧
  simplified_income_tax example_data ≤ main_system_tax example_data ∧
  simplified_income_tax example_data ≤ simplified_profit_tax example_data := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_tax_is_simplified_income_tax_example_minimum_tax_l1160_116011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_student_correct_l1160_116052

/-- The circle game function that determines the number of the last remaining student. -/
def last_student (n : ℕ) : ℤ :=
  let binary_digits := Nat.digits 2 n
  let k := binary_digits.length - 1
  (List.range (k + 1)).foldl (λ acc i =>
    acc + 2^i * (if binary_digits.get? i == some 1 then -1 else 1)) 0

/-- Theorem stating that the last_student function correctly determines
    the number of the last remaining student in the circle game. -/
theorem last_student_correct (n : ℕ) :
  last_student n = (List.range (Nat.digits 2 n).length).foldl
    (λ acc i => acc + 2^i * (if (Nat.digits 2 n).get? i == some 1 then -1 else 1)) 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_student_correct_l1160_116052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_three_thirty_l1160_116062

/-- Calculates the angle between hour and minute hands of a clock --/
noncomputable def clockAngle (hour : ℝ) (minute : ℝ) : ℝ :=
  |60 * hour - 11 * minute| / 2

theorem angle_at_three_thirty : 
  clockAngle 3 30 = 75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_three_thirty_l1160_116062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l1160_116048

/-- Two lines are parallel if their slopes are equal -/
def parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  (a₁ / b₁ = -a₂ / b₂) ∧ (b₁ ≠ 0) ∧ (b₂ ≠ 0)

/-- Definition of line l₁ -/
def l₁ (m x y : ℝ) : Prop :=
  (2*m + 1)*x - 4*y + 3*m = 0

/-- Definition of line l₂ -/
def l₂ (m x y : ℝ) : Prop :=
  x + (m + 5)*y - 3*m = 0

theorem parallel_lines_m_value :
  ∀ m : ℝ, parallel (2*m + 1) (-4) 1 (m + 5) → m = -9/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l1160_116048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1160_116087

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 7784) : 
  Int.gcd (5 * b^2 + 68 * b + 143) (3 * b + 14) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1160_116087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l1160_116014

/-- Given an ellipse with focal length 2√2 and equation x²/a + y² = 1 (a > 1),
    prove that for a point P on the ellipse with |PF₁| = 2, |PF₂| = 2√3 - 2 -/
theorem ellipse_focal_distance (a : ℝ) (P F₁ F₂ : ℝ × ℝ) (ellipse : Set (ℝ × ℝ)) :
  a > 1 →
  (∀ x y, x^2 / a + y^2 = 1 ↔ (x, y) ∈ ellipse) →
  P ∈ ellipse →
  ‖F₁ - F₂‖ = 2 * Real.sqrt 2 →
  ‖P - F₁‖ = 2 →
  ‖P - F₂‖ = 2 * Real.sqrt 3 - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l1160_116014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_matrix_max_value_l1160_116050

open Real Matrix

noncomputable def det_matrix (θ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 1 + Real.sin θ, 1 + Real.cos θ],
    ![1 + Real.cos θ, 1, 1 + Real.sin θ],
    ![1 + Real.sin θ, 1 + Real.cos θ, 1]]

theorem det_matrix_max_value :
  ∀ θ : ℝ, det (det_matrix θ) ≤ 2 ∧ ∃ θ₀ : ℝ, det (det_matrix θ₀) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_matrix_max_value_l1160_116050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l1160_116076

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-1 + Real.sqrt 3 / 2 * t, 1 / 2 * t)

-- Define the circle C
def circle_C (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 4*p.1

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_l t ∧ circle_C p}

-- State the theorem
theorem intersection_segment_length :
  ∃ p q : ℝ × ℝ, p ∈ intersection_points ∧ q ∈ intersection_points ∧ p ≠ q ∧
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l1160_116076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_axes_intersection_circle_l1160_116020

/-- The parabola defined by y = x^2 - 2x - 3 -/
def parabola (x y : ℝ) : Prop := y = x^2 - 2*x - 3

/-- A point (x, y) lies on the coordinate axes -/
def on_axes (x y : ℝ) : Prop := x = 0 ∨ y = 0

/-- The circle equation (x-1)^2 + (y+1)^2 = 5 -/
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 5

/-- Theorem: The circle passing through the intersections of the parabola
    y = x^2 - 2x - 3 with the coordinate axes has the equation (x-1)^2 + (y+1)^2 = 5 -/
theorem parabola_axes_intersection_circle : 
  ∀ x y : ℝ, (parabola x y ∧ on_axes x y) → circle_eq x y := by
  sorry

/-- The set of points satisfying the circle equation -/
def circle_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | circle_eq p.1 p.2}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_axes_intersection_circle_l1160_116020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_P_l1160_116008

/-- Definition of a hyperbola -/
def is_hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Definition of the right focal point of a hyperbola -/
noncomputable def right_focal_point (a b : ℝ) : ℝ × ℝ :=
  (Real.sqrt (a^2 + b^2), 0)

/-- Definition of a point being outside the hyperbola -/
def is_outside_hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 > 1

/-- Definition of perpendicularity -/
def perpendicular (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : Prop :=
  (x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3) = 0

/-- Theorem about the locus of point P -/
theorem locus_of_point_P (a b : ℝ) (x y : ℝ) :
  ∀ F A B : ℝ × ℝ,
  a > 0 → b > 0 →
  is_hyperbola a b x y →
  F = right_focal_point a b →
  is_outside_hyperbola a b x y →
  perpendicular A.1 A.2 B.1 B.2 x y F.1 F.2 →
  (y = 0 ∧ -a < x ∧ x < a) ∨ x = a^2 / Real.sqrt (a^2 + b^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_P_l1160_116008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_sum_l1160_116060

/-- A sine function with specific properties -/
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

/-- Theorem stating the sum of specific function values -/
theorem sine_function_sum 
  (A ω φ : ℝ) 
  (h1 : A > 0) 
  (h2 : ω > 0) 
  (h3 : |φ| ≤ π/2) 
  (h4 : ∀ x, f A ω φ (-x) = -(f A ω φ x)) -- odd function
  (h5 : f A ω φ 2 = 2) -- maximum value at x = 2
  (h6 : ∀ x, f A ω φ x ≤ 2) -- 2 is the maximum value
  : f A ω φ 1 + f A ω φ 2 + f A ω φ 3 + 1/4 + f A ω φ 100 = 2 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_sum_l1160_116060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_build_time_l1160_116018

theorem tom_build_time : ∃ (tom_time : ℝ), tom_time = 5 := by
  let avery_time : ℝ := 3
  let tom_completion_time : ℝ := 139.99999999999997 / 60

  -- Define Tom's build time
  let tom_time : ℝ := 5

  -- Work done by Avery and Tom together in the first hour
  let work_together : ℝ := 1 / avery_time + 1 / tom_time

  -- Work done by Tom alone after Avery leaves
  let work_tom_alone : ℝ := (1 / tom_time) * tom_completion_time

  -- Total work equals one complete wall
  have total_work : work_together + work_tom_alone = 1 := by sorry

  -- Prove that Tom's build time is 5 hours
  existsi tom_time
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_build_time_l1160_116018
