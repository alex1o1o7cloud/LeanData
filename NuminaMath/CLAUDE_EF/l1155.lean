import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_probability_l1155_115519

/-- The length of the line segment MN -/
noncomputable def segment_length : ℝ := 16

/-- The minimum area of the rectangle -/
noncomputable def min_area : ℝ := 60

/-- The probability that the area of the rectangle is greater than the minimum area -/
noncomputable def probability : ℝ := 1/4

/-- Theorem stating the probability of the rectangle's area being greater than the minimum area -/
theorem rectangle_area_probability : 
  ∃ (x : ℝ), 0 < x ∧ x < segment_length ∧
  (probability = (10 - 6) / segment_length ∧
   ∀ (y : ℝ), 0 < y → y < segment_length → 
   (y * (segment_length - y) > min_area ↔ 6 < y ∧ y < 10)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_probability_l1155_115519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_l1155_115515

theorem sequence_existence (n : ℕ) : 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
  ∃ (x : ℕ → ℕ), (∀ i j : ℕ, i < j ∧ j ≤ n → x i < x j) ∧ 
  (∀ i : ℕ, i ≤ n → x i > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_l1155_115515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_f_passes_through_point_l1155_115588

/-- A power function that passes through the point (3, √3/3) -/
noncomputable def f (x : ℝ) : ℝ := x^(-(1/2 : ℝ))

/-- The function g(x) = √x + f(x) -/
noncomputable def g (x : ℝ) : ℝ := Real.sqrt x + f x

/-- The theorem stating the range of g(x) on the interval [1/2, 3] -/
theorem g_range :
  ∀ x ∈ Set.Icc (1/2 : ℝ) 3,
    2 ≤ g x ∧ g x ≤ 4 * Real.sqrt 3 / 3 ∧
    ∃ y ∈ Set.Icc (1/2 : ℝ) 3, g y = 2 ∧
    ∃ z ∈ Set.Icc (1/2 : ℝ) 3, g z = 4 * Real.sqrt 3 / 3 :=
by sorry

/-- Verification that f(3) = √3/3 -/
theorem f_passes_through_point : f 3 = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_f_passes_through_point_l1155_115588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_properties_l1155_115586

/-- Represents a trapezoid ABCD with bases AD and BC -/
structure Trapezoid where
  a : ℝ  -- Length of base AD
  b : ℝ  -- Length of base BC
  p : ℝ  -- Ratio part for AM and DN
  q : ℝ  -- Ratio part for MB and NC
  h_positive : 0 < a ∧ 0 < b
  h_a_gt_b : a > b
  h_positive_ratio : 0 < p ∧ 0 < q

/-- The length of the segment cut by the diagonals on the midline of the trapezoid -/
noncomputable def diagonalMidlineSegment (t : Trapezoid) : ℝ := (t.a - t.b) / 2

/-- The length of the segment MN, where M and N divide the sides AB and CD in the ratio p:q -/
noncomputable def segmentMN (t : Trapezoid) : ℝ := (t.q * t.a + t.p * t.b) / (t.p + t.q)

/-- Theorem stating the properties of the trapezoid -/
theorem trapezoid_properties (t : Trapezoid) :
  diagonalMidlineSegment t = (t.a - t.b) / 2 ∧
  segmentMN t = (t.q * t.a + t.p * t.b) / (t.p + t.q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_properties_l1155_115586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_divisor_problem_l1155_115532

theorem third_divisor_problem (smallest_number : ℕ) 
  (h1 : smallest_number = 1012)
  (h2 : ∀ (x : ℕ), x ∈ ({12, 16, 21, 28} : Set ℕ) → (smallest_number - 4) % x = 0)
  (h3 : ∃ (third_divisor : ℕ), (smallest_number - 4) % third_divisor = 0) :
  ∃ (third_divisor : ℕ), third_divisor = 3 ∧ (smallest_number - 4) % third_divisor = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_divisor_problem_l1155_115532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_equality_l1155_115501

variable (a₁ d : ℝ)  -- first term and common difference of the arithmetic progression
variable (p n k : ℕ+)  -- positive natural numbers

-- Definition of the sum of an arithmetic progression
noncomputable def S (m : ℕ+) : ℝ := (m : ℝ) * (2 * a₁ + ((m : ℝ) - 1) * d) / 2

-- The theorem to be proved
theorem arithmetic_progression_sum_equality (a₁ d : ℝ) (p n k : ℕ+) :
  (S a₁ d p / (p : ℝ)) * ((n : ℝ) - (k : ℝ)) + 
  (S a₁ d n / (n : ℝ)) * ((k : ℝ) - (p : ℝ)) + 
  (S a₁ d k / (k : ℝ)) * ((p : ℝ) - (n : ℝ)) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_equality_l1155_115501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_1000_stars_l1155_115526

/-- Definition of a regular n-pointed quadratic star -/
def RegularQuadraticStar (n : ℕ) := {S : Set (ℝ × ℝ) // true}

/-- The number of non-similar regular n-pointed quadratic stars -/
def CountNonSimilarStars (n : ℕ) : ℕ := 0

/-- No regular 3-pointed quadratic stars exist -/
axiom no_3_star : ¬ ∃ (S : RegularQuadraticStar 3), true

/-- No regular 4-pointed quadratic stars exist -/
axiom no_4_star : ¬ ∃ (S : RegularQuadraticStar 4), true

/-- No regular 6-pointed quadratic stars exist -/
axiom no_6_star : ¬ ∃ (S : RegularQuadraticStar 6), true

/-- Similarity relation for RegularQuadraticStar -/
def Similar (n : ℕ) : RegularQuadraticStar n → RegularQuadraticStar n → Prop := fun _ _ => true

/-- All regular 5-pointed quadratic stars are similar -/
axiom all_5_star_similar : ∀ (S T : RegularQuadraticStar 5), Similar 5 S T

/-- The main theorem: There are 200 non-similar regular 1000-pointed quadratic stars -/
theorem count_1000_stars : CountNonSimilarStars 1000 = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_1000_stars_l1155_115526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_KLMN_is_cyclic_l1155_115565

-- Define the basic geometric objects
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

structure Chord where
  start : Point
  finish : Point

-- Define the setup
structure GeometricSetup where
  circle1 : Circle
  circle2 : Circle
  commonChord : Chord
  P : Point
  chordKM : Chord
  chordLN : Chord
  intersecting : circle1 ≠ circle2
  P_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = Point.mk 
    ((1 - t) * commonChord.start.x + t * commonChord.finish.x)
    ((1 - t) * commonChord.start.y + t * commonChord.finish.y)
  KM_in_circle1 : (chordKM.start.x - circle1.center.x)^2 + (chordKM.start.y - circle1.center.y)^2 = circle1.radius^2 ∧
                  (chordKM.finish.x - circle1.center.x)^2 + (chordKM.finish.y - circle1.center.y)^2 = circle1.radius^2
  LN_in_circle2 : (chordLN.start.x - circle2.center.x)^2 + (chordLN.start.y - circle2.center.y)^2 = circle2.radius^2 ∧
                  (chordLN.finish.x - circle2.center.x)^2 + (chordLN.finish.y - circle2.center.y)^2 = circle2.radius^2
  P_on_KM : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = Point.mk 
    ((1 - t) * chordKM.start.x + t * chordKM.finish.x)
    ((1 - t) * chordKM.start.y + t * chordKM.finish.y)
  P_on_LN : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = Point.mk 
    ((1 - t) * chordLN.start.x + t * chordLN.finish.x)
    ((1 - t) * chordLN.start.y + t * chordLN.finish.y)

-- Define what it means for a quadrilateral to be cyclic
def is_cyclic (A B C D : Point) : Prop :=
  ∃ (circle : Circle), 
    (A.x - circle.center.x)^2 + (A.y - circle.center.y)^2 = circle.radius^2 ∧
    (B.x - circle.center.x)^2 + (B.y - circle.center.y)^2 = circle.radius^2 ∧
    (C.x - circle.center.x)^2 + (C.y - circle.center.y)^2 = circle.radius^2 ∧
    (D.x - circle.center.x)^2 + (D.y - circle.center.y)^2 = circle.radius^2

-- State the theorem
theorem quadrilateral_KLMN_is_cyclic (setup : GeometricSetup) :
  is_cyclic setup.chordKM.start setup.chordLN.start setup.chordKM.finish setup.chordLN.finish :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_KLMN_is_cyclic_l1155_115565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_of_parabola_l1155_115548

/-- Given a parabola and a chord, prove the equation of the line containing the chord -/
theorem chord_equation_of_parabola (x y : ℝ → ℝ) (A B : ℝ × ℝ) :
  (∀ t : ℝ, (x t)^2 = -2 * y t) →  -- Parabola equation
  (A.1 + B.1) / 2 = -1 →           -- x-coordinate of midpoint
  (A.2 + B.2) / 2 = -5 →           -- y-coordinate of midpoint
  ∃ (m b : ℝ), ∀ t : ℝ, y t = m * x t + b ∧ m = 1 ∧ b = -4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_of_parabola_l1155_115548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kendras_learned_words_l1155_115567

/-- Given Kendra's goal of learning new words and the number of words she still needs to learn,
    calculate the number of words she has already learned. -/
theorem kendras_learned_words : 60 - 24 = 36 := by
  -- The proof goes here
  sorry

#check kendras_learned_words

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kendras_learned_words_l1155_115567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_size_l1155_115534

/-- Represents the number of employees in each age group -/
structure EmployeeCount where
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Represents the sample size for each age group -/
structure SampleSize where
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Calculates the total sample size given the employee count and young sample size -/
def calculateTotalSampleSize (employees : EmployeeCount) (youngSampleSize : ℕ) : ℕ :=
  let ratio : ℚ := youngSampleSize / employees.young
  let middleAgedSample := Int.floor (ratio * employees.middleAged)
  let elderlySample := Int.floor (ratio * employees.elderly)
  youngSampleSize + middleAgedSample.toNat + elderlySample.toNat

/-- Theorem stating that for the given employee counts and young sample size, 
    the total sample size is 15 -/
theorem stratified_sample_size 
  (employees : EmployeeCount) 
  (youngSampleSize : ℕ) :
  employees.young = 350 → 
  employees.middleAged = 250 → 
  employees.elderly = 150 → 
  youngSampleSize = 7 → 
  calculateTotalSampleSize employees youngSampleSize = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_size_l1155_115534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_sum_equals_four_l1155_115550

theorem ratio_sum_equals_four (x y : ℝ) (θ : ℝ) (hx : x > 0) (hy : y > 0)
  (hθ : ∀ n : ℤ, θ ≠ n * π / 2)
  (h1 : Real.sin θ / x = Real.cos θ / y)
  (h2 : Real.cos θ ^ 4 / x ^ 4 + Real.sin θ ^ 4 / y ^ 4 = 97 * Real.sin (2 * θ) / (x ^ 3 * y + y ^ 3 * x)) :
  y / x + x / y = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_sum_equals_four_l1155_115550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_area_angle_l1155_115590

/-- Given a triangle ABC with sides a, b, and c, if its area S is (a^2 + b^2 - c^2) / 4,
    then the measure of angle C is 45°. -/
theorem triangle_special_area_angle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (a^2 + b^2 - c^2) / 4
  S = (a * b * Real.sin (Real.pi / 4)) / 2 → Real.cos (Real.pi / 4) = Real.sin (Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_area_angle_l1155_115590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_implies_a_range_l1155_115529

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 1)

-- Define the property of f having a minimum value
def has_minimum (f : ℝ → ℝ) : Prop := ∃ (x₀ : ℝ), ∀ (x : ℝ), f x₀ ≤ f x

-- Theorem statement
theorem f_min_implies_a_range (a : ℝ) : 
  has_minimum (f a) → -2 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_implies_a_range_l1155_115529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_symmetric_axis_l1155_115587

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((x + 1) / (x - 1)) + Real.log (x - 1) + Real.log (a - x)

/-- The theorem stating that no symmetric axis exists for any a > 1 -/
theorem no_symmetric_axis (a : ℝ) (h : a > 1) :
  ¬ ∃ k : ℝ, ∀ x : ℝ, 1 < x ∧ x < a → f a x = f a (2 * k - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_symmetric_axis_l1155_115587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_three_l1155_115568

theorem opposite_of_three : 
  (∀ x : ℤ, x + (-x) = 0) → -3 = (λ x ↦ -x) 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_three_l1155_115568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_product_at_vertices_l1155_115505

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the foci of the ellipse
def leftFocus : ℝ × ℝ := (-4, 0)
def rightFocus : ℝ × ℝ := (4, 0)

-- Define the product of distances from a point to the foci
noncomputable def distanceProduct (x y : ℝ) : ℝ :=
  distance x y leftFocus.1 leftFocus.2 * distance x y rightFocus.1 rightFocus.2

-- Theorem statement
theorem max_distance_product_at_vertices :
  ∀ x y : ℝ, ellipse x y →
  distanceProduct x y ≤ distanceProduct 0 3 ∧
  (distanceProduct x y = distanceProduct 0 3 ↔ (x = 0 ∧ (y = 3 ∨ y = -3))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_product_at_vertices_l1155_115505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l1155_115574

/-- The sum of the infinite series Σ(n=1 to ∞) (2n-1)(1/1000)^(n-1) -/
noncomputable def series_sum : ℝ := ∑' n, (2 * n - 1) * (1 / 1000) ^ (n - 1)

/-- The theorem stating that the sum of the infinite series is equal to 1001999/998001 -/
theorem series_sum_value : series_sum = 1001999 / 998001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l1155_115574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1155_115584

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of lengths 20 cm and 18 cm, 
    and a distance of 12 cm between them, is equal to 228 square centimeters. -/
theorem trapezium_area_example : trapezium_area 20 18 12 = 228 := by
  -- Unfold the definition of trapezium_area
  unfold trapezium_area
  -- Simplify the arithmetic
  simp [mul_add, mul_div_right_comm]
  -- Check that the result is equal to 228
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1155_115584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l1155_115595

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
structure ArithmeticSequence (α : Type*) [Add α] where
  a : ℕ → α
  d : α
  h : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence. -/
noncomputable def sum_n (seq : ArithmeticSequence ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence ℝ)
  (h1 : seq.a 4 + seq.a 6 = 10)
  (h2 : sum_n seq 5 = 5) :
  seq.d = 2 := by
  sorry

#check arithmetic_sequence_common_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l1155_115595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_sum_constant_l1155_115585

-- Define the curve E
def E (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the point F
def F : ℝ × ℝ := (1, 0)

-- Define the line l passing through F
def l (m : ℝ) (x y : ℝ) : Prop := y = m * (x - 1)

-- Define the intersection of l with the y-axis
def R (m : ℝ) : ℝ × ℝ := (0, m)

-- Define the conditions for points P and Q
def P_condition (m lambda1 : ℝ) (P : ℝ × ℝ) : Prop :=
  E P.1 P.2 ∧ l m P.1 P.2 ∧ R m - P = lambda1 • (P - F)

def Q_condition (m lambda2 : ℝ) (Q : ℝ × ℝ) : Prop :=
  E Q.1 Q.2 ∧ l m Q.1 Q.2 ∧ R m - Q = lambda2 • (Q - F)

-- Theorem statement
theorem lambda_sum_constant (m : ℝ) (P Q : ℝ × ℝ) (lambda1 lambda2 : ℝ) :
  P_condition m lambda1 P → Q_condition m lambda2 Q → ∃ (c : ℝ), lambda1 + lambda2 = c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_sum_constant_l1155_115585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l1155_115518

/-- Revenue function --/
noncomputable def revenue (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then 400 * x - (1/2) * x^2 else 80000

/-- Cost function --/
noncomputable def cost (x : ℝ) : ℝ := 20000 + 100 * x

/-- Profit function --/
noncomputable def profit (x : ℝ) : ℝ := revenue x - cost x

/-- Theorem stating the maximum profit and the corresponding production volume --/
theorem max_profit :
  ∃ (x : ℝ), x = 300 ∧ profit x = 25000 ∧ ∀ y, profit y ≤ profit x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l1155_115518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_divisor_of_1008_l1155_115566

theorem second_divisor_of_1008 :
  let n := 1008
  let divisors := [12, 18, 21, 28]
  (∀ d ∈ divisors, n % d = 0) →
  (∃ m : ℕ, m > 12 ∧ m < 14 ∧ n % m = 0) →
  14 = (Finset.filter (λ x => x > 12 ∧ n % x = 0) (Finset.range (n + 1))).min' sorry :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_divisor_of_1008_l1155_115566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dorothy_museum_trip_l1155_115540

/-- Calculates the remaining money for Dorothy after a museum trip with her family -/
def dorothy_remaining_money (dorothy_age : ℕ) (family_size : ℕ) (regular_ticket_cost : ℚ) 
  (discount_percentage : ℚ) (initial_money : ℚ) : ℚ :=
  let discounted_members := 2  -- Dorothy and her younger brother
  let full_price_members := family_size - discounted_members
  let discount_amount := regular_ticket_cost * discount_percentage
  let discounted_ticket_cost := regular_ticket_cost - discount_amount
  let total_cost := (discounted_members * discounted_ticket_cost) + (full_price_members * regular_ticket_cost)
  initial_money - total_cost

/-- Main theorem proving Dorothy's remaining money after the museum trip -/
theorem dorothy_museum_trip : ∃ (remaining_money : ℚ), 
  dorothy_remaining_money 15 5 10 (30/100) 70 = remaining_money ∧ remaining_money = 26 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dorothy_museum_trip_l1155_115540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_f_l1155_115504

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 - 2*x - x^2)

theorem domain_and_range_of_f :
  (∀ x, f x ∈ Set.Icc 0 2 ↔ x ∈ Set.Icc (-3) 1) ∧
  Set.range f = Set.Icc 0 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_f_l1155_115504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tier_level_is_10000_l1155_115535

/-- Represents the two-tiered tax system for imported cars -/
structure TaxSystem where
  firstTierRate : ℝ
  secondTierRate : ℝ
  firstTierLevel : ℝ

/-- Calculates the total tax for a given car price and tax system -/
noncomputable def calculateTax (price : ℝ) (system : TaxSystem) : ℝ :=
  if price ≤ system.firstTierLevel then
    system.firstTierRate * price
  else
    system.firstTierRate * system.firstTierLevel +
    system.secondTierRate * (price - system.firstTierLevel)

/-- Theorem stating that for the given conditions, the first tier's price level is $10,000 -/
theorem first_tier_level_is_10000 :
  ∃ (system : TaxSystem),
    system.firstTierRate = 0.25 ∧
    system.secondTierRate = 0.15 ∧
    calculateTax 30000 system = 5500 ∧
    system.firstTierLevel = 10000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tier_level_is_10000_l1155_115535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_discount_proof_l1155_115555

theorem store_discount_proof (original_price : ℝ) (original_price_positive : 0 < original_price) : 
  (1 - 0.25) * ((1 / 3) * original_price) = 0.25 * original_price := by
  sorry

#check store_discount_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_discount_proof_l1155_115555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_abs_plus_power_linear_function_through_points_l1155_115521

-- Problem 1
theorem cube_root_plus_abs_plus_power : (8 : ℝ) ^ (1/3) + |(-5)| + (-1)^2023 = 6 := by sorry

-- Problem 2
theorem linear_function_through_points :
  ∀ k b : ℝ,
  (k * 0 + b = 1) →
  (k * 2 + b = 5) →
  ∀ x : ℝ, k * x + b = 2 * x + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_abs_plus_power_linear_function_through_points_l1155_115521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3theta_and_sin_3theta_l1155_115507

theorem tan_3theta_and_sin_3theta (θ : ℝ) (h : Real.tan θ = 2) :
  Real.tan (3 * θ) = 2 / 11 ∧ Real.sin (3 * θ) = 22 / 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3theta_and_sin_3theta_l1155_115507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_calculation_l1155_115559

/-- Given a voltage V and impedance Z, calculate the current I --/
noncomputable def calculate_current (V Z : ℂ) : ℂ := V / Z

/-- Theorem stating that for the given voltage and impedance, the calculated current is (1/2)i --/
theorem current_calculation :
  let V : ℂ := 2 + Complex.I
  let Z : ℂ := 2 - 4 * Complex.I
  calculate_current V Z = (1/2) * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_calculation_l1155_115559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_and_lines_l1155_115528

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation
variable (perp : Plane → Plane → Prop)
variable (perpL : Line → Line → Prop)
variable (perpPL : Plane → Line → Prop)

-- Define the intersection operation
variable (intersect : Plane → Plane → Line)

-- Given planes and lines
variable (α β γ : Plane)
variable (l m : Line)

-- Conditions
variable (h1 : perp α γ)
variable (h2 : intersect γ α = m)
variable (h3 : intersect γ β = l)
variable (h4 : perpL l m)

-- Theorem to prove
theorem perpendicular_planes_and_lines :
  perpPL α l ∧ perp α β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_and_lines_l1155_115528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1155_115542

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (Real.pi - ω * x) * Real.cos (ω * x) + Real.cos (ω * x) ^ 2

theorem problem_solution (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_period : ∀ (x : ℝ), f ω (x + Real.pi) = f ω x) 
  (h_smallest_period : ∀ (T : ℝ), T > 0 → (∀ (x : ℝ), f ω (x + T) = f ω x) → T ≥ Real.pi) : 
  ω = 1 ∧ 
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 16) → f ω (2 * x) ≥ 1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 16) ∧ f ω (2 * x) = 1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1155_115542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_l1155_115551

theorem x_value (x y : ℝ) (h1 : (7 : ℝ)^(x - y) = 343) (h2 : (7 : ℝ)^(x + y) = 16807) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_l1155_115551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_product_l1155_115599

def is_not_divisible_by_2_or_5 (n : ℕ) : Bool :=
  n % 2 ≠ 0 && n % 5 ≠ 0

def product_of_numbers (n : ℕ) : ℕ :=
  (List.range n).filter is_not_divisible_by_2_or_5
    |>.foldl (·*·) 1

theorem last_digit_of_product :
  (product_of_numbers 101) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_product_l1155_115599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l1155_115543

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (3*x - 2) / Real.log x + 2

-- Theorem statement
theorem fixed_point_of_f : 
  ∀ x : ℝ, x > 0 → x ≠ 1 → f 1 = 2 := by
  intro x h1 h2
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l1155_115543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l1155_115572

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

/-- The foci of an ellipse -/
noncomputable def Ellipse.foci (e : Ellipse a b) : ℝ × ℝ := sorry

/-- A point on an ellipse -/
def Ellipse.point (e : Ellipse a b) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse a b) : ℝ := sorry

/-- A line perpendicular to the x-axis passing through a point -/
def perpendicular_line (x : ℝ) := {p : ℝ × ℝ | p.1 = x}

theorem ellipse_eccentricity_theorem (a b : ℝ) (e : Ellipse a b) 
  (hpos : 0 < a ∧ 0 < b ∧ b < a) :
  let (f₁, f₂) := e.foci
  let l := perpendicular_line f₁
  ∃ A B : ℝ × ℝ, 
    (A ∈ l ∧ e.point A.1 A.2) ∧ 
    (B ∈ l ∧ e.point B.1 B.2) ∧ 
    ((A.1 - f₂)^2 + (A.2 - 0)^2 + (B.1 - f₂)^2 + (B.2 - 0)^2 = 
     (A.1 - B.1)^2 + (A.2 - B.2)^2) →
  e.eccentricity = Real.sqrt 2 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l1155_115572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2015_l1155_115573

def sequence_a : ℕ → ℚ
  | 0 => 3/5  -- Adding a case for 0 to cover all natural numbers
  | 1 => 3/5
  | n + 1 => 
    let a_n := sequence_a n
    if 0 ≤ a_n ∧ a_n < 1/2 then 2 * a_n
    else if 1/2 ≤ a_n ∧ a_n < 1 then 2 * a_n - 1
    else a_n  -- This case should never occur given the constraints

theorem sequence_a_2015 : sequence_a 2015 = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2015_l1155_115573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_average_l1155_115547

theorem consecutive_integers_average (c : ℤ) :
  let d := (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6
  (d + 1 + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 6 = c + 6 := by
  sorry

#check consecutive_integers_average

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_average_l1155_115547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l1155_115557

-- Define curve C₁
def C₁ (t : ℝ) : ℝ × ℝ := (2 * t - 1, -4 * t - 2)

-- Define curve C₂ in polar form
noncomputable def C₂_polar (θ : ℝ) : ℝ := 2 / (1 - Real.cos θ)

-- Convert C₂ to Cartesian form
def C₂ (x y : ℝ) : Prop := y^2 = 4 * (x - 1)

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Theorem statement
theorem min_distance_C₁_C₂ :
  ∃ (d : ℝ), d = 3 * Real.sqrt 5 / 10 ∧
  ∀ (t θ : ℝ), 
    let p₁ := C₁ t
    let x := C₂_polar θ * Real.cos θ
    let y := C₂_polar θ * Real.sin θ
    let p₂ := (x, y)
    C₂ x y → distance p₁ p₂ ≥ d :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l1155_115557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_intersection_l1155_115502

noncomputable def f (x : ℝ) : ℝ := (x^2 - 6*x + 8) / (x^2 - 6*x + 9)

theorem asymptote_intersection :
  let vertical_asymptote : ℝ := 3
  let horizontal_asymptote : ℝ := 1
  (vertical_asymptote, horizontal_asymptote) = (3, 1) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - vertical_asymptote| < δ ∧ x ≠ vertical_asymptote →
    |f x| > (1/ε)) ∧
  (∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - horizontal_asymptote| < ε) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_intersection_l1155_115502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_line_movement_l1155_115506

theorem number_line_movement
  (start left_move right_move : Int) :
  start = 0 → left_move = 6 → right_move = 3 →
  start - left_move + right_move = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_line_movement_l1155_115506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l1155_115522

-- Define a triangle with two perpendicular medians
structure TriangleWithPerpendicularMedians where
  median1 : ℝ
  median2 : ℝ
  perpendicular : median1 ≠ 0 ∧ median2 ≠ 0

-- Define the theorem
theorem third_median_length 
  (t : TriangleWithPerpendicularMedians) 
  (h1 : t.median1 = 18) 
  (h2 : t.median2 = 24) : 
  ∃ (third_median : ℝ), third_median = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l1155_115522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_formula_l1155_115553

/-- The area of a regular octagon inscribed in a circle with radius s -/
noncomputable def octagon_area (s : ℝ) : ℝ :=
  2 * Real.sqrt 2 * s^2

/-- Theorem: The area of a regular octagon inscribed in a circle with radius s
    is equal to 2√2 * s^2 -/
theorem octagon_area_formula (s : ℝ) (h : s > 0) :
  octagon_area s = 2 * Real.sqrt 2 * s^2 := by
  -- Unfold the definition of octagon_area
  unfold octagon_area
  -- The equality holds by definition
  rfl

#check octagon_area_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_formula_l1155_115553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadrilateral_ratio_l1155_115544

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  -- Points of the quadrilateral
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  -- Right angles at B and C
  right_angle_B : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0
  right_angle_C : (B.1 - C.1) * (D.1 - C.1) + (B.2 - C.2) * (D.2 - C.2) = 0
  -- Triangle ABC similar to triangle BCD
  similar_ABC_BCD : ∃ k : ℝ, k > 0 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = k * ((B.1 - C.1)^2 + (B.2 - C.2)^2) ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = k * ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  -- AB > BC
  AB_greater_BC : (A.1 - B.1)^2 + (A.2 - B.2)^2 > (B.1 - C.1)^2 + (B.2 - C.2)^2
  -- E is in the interior of ABCD (simplified condition)
  E_interior : E.1 > min A.1 (min B.1 (min C.1 D.1)) ∧ E.1 < max A.1 (max B.1 (max C.1 D.1)) ∧
               E.2 > min A.2 (min B.2 (min C.2 D.2)) ∧ E.2 < max A.2 (max B.2 (max C.2 D.2))
  -- Triangle ABC similar to triangle CEB
  similar_ABC_CEB : ∃ m : ℝ, m > 0 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = m * ((C.1 - E.1)^2 + (C.2 - E.2)^2) ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = m * ((E.1 - B.1)^2 + (E.2 - B.2)^2)
  -- Area of triangle AED is 25 times the area of triangle CEB
  area_ratio : abs ((A.1 - E.1) * (D.2 - E.2) - (A.2 - E.2) * (D.1 - E.1)) =
               25 * abs ((C.1 - E.1) * (B.2 - E.2) - (C.2 - E.2) * (B.1 - E.1))

/-- The ratio AB/BC in a SpecialQuadrilateral is 5 + 2√5 -/
theorem special_quadrilateral_ratio (q : SpecialQuadrilateral) :
  Real.sqrt ((q.A.1 - q.B.1)^2 + (q.A.2 - q.B.2)^2) /
  Real.sqrt ((q.B.1 - q.C.1)^2 + (q.B.2 - q.C.2)^2) = 5 + 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadrilateral_ratio_l1155_115544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_height_at_16_minutes_l1155_115558

/-- Represents a Ferris wheel with constant rotation speed -/
structure FerrisWheel where
  revolution_time : ℝ  -- Time for one complete revolution in minutes
  lowest_point : ℝ     -- Height of the lowest point in meters
  highest_point : ℝ    -- Height of the highest point in meters

/-- Calculates the height of a point on the Ferris wheel at a given time -/
noncomputable def height_at_time (wheel : FerrisWheel) (time : ℝ) : ℝ :=
  let radius := (wheel.highest_point - wheel.lowest_point) / 2
  let center_height := wheel.lowest_point + radius
  let angle := 2 * Real.pi * (time / wheel.revolution_time)
  center_height + radius * Real.cos angle

/-- The main theorem to be proved -/
theorem ferris_wheel_height_at_16_minutes 
  (wheel : FerrisWheel) 
  (h1 : wheel.revolution_time = 12)
  (h2 : wheel.lowest_point = 2)
  (h3 : wheel.highest_point = 18) : 
  height_at_time wheel 16 = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_height_at_16_minutes_l1155_115558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l1155_115560

noncomputable def f (x : ℝ) (φ : ℝ) := 2 * Real.sin (2 * x + φ)

theorem f_range_theorem (φ : ℝ) (h1 : -Real.pi < φ) (h2 : φ < 0) 
  (h3 : ∀ x, f x φ = f (-x) φ) :
  let range := { y | ∃ x ∈ Set.Icc (Real.pi / 6) ((2 * Real.pi) / 3), f x φ = y }
  range = Set.Icc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l1155_115560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_section_area_l1155_115591

/-- Represents a cylinder with given radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents a planar section of a cylinder -/
noncomputable def CylinderSection (c : Cylinder) (arcAngle : ℝ) : ℝ :=
  (c.radius^2 * arcAngle / 4) * (2 * c.height / c.radius)

/-- Theorem stating the area of the planar section for a specific cylinder and arc angle -/
theorem cylinder_section_area (c : Cylinder) (h1 : c.radius = 5) (h2 : c.height = 10) :
  CylinderSection c (Real.pi/2) = 25/2 * Real.pi := by
  sorry

#check cylinder_section_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_section_area_l1155_115591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1155_115533

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (4 * a^(2/3 : ℝ) * b^(-(1/3) : ℝ)) / (-2/3 * a^(-(1/3) : ℝ) * b^(2/3 : ℝ)) = -6 * a / b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1155_115533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_participants_l1155_115564

theorem chess_tournament_participants (n : ℕ) 
  (h1 : n > 0)
  (h2 : (n * 4) / 2 = 12) :
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_participants_l1155_115564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1155_115561

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptotes
noncomputable def asymptote1 (x y : ℝ) : ℝ := x - 2*y
noncomputable def asymptote2 (x y : ℝ) : ℝ := x + 2*y

-- Define the distance from a point to a line ax + by + c = 0
noncomputable def distanceToLine (x y a b c : ℝ) : ℝ :=
  (|a*x + b*y + c|) / Real.sqrt (a^2 + b^2)

-- Define the distance between two points
noncomputable def distanceBetweenPoints (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem hyperbola_properties :
  ∀ x y : ℝ, hyperbola x y →
    (∃ d : ℝ, d = distanceToLine x y 1 (-2) 0 * distanceToLine x y 1 2 0 ∧ d = 4/5) ∧
    (∃ minDist : ℝ, minDist = 2 ∧ ∀ x' y' : ℝ, hyperbola x' y' →
      distanceBetweenPoints x' y' 5 0 ≥ minDist) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1155_115561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_formula_l1155_115596

/-- The area of a parallelogram with slant heights a and b, and an angle θ between them -/
noncomputable def parallelogramArea (a b θ : ℝ) : ℝ := a * b * Real.sin θ

/-- Theorem: The area of a parallelogram with slant heights a and b, and an angle θ between them,
    is equal to a * b * sin(θ) -/
theorem parallelogram_area_formula (a b θ : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_angle : 0 < θ ∧ θ < Real.pi) :
  parallelogramArea a b θ = a * b * Real.sin θ := by
  sorry

#check parallelogram_area_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_formula_l1155_115596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression1_value_expression2_value_l1155_115552

open Real

-- Define the first expression
noncomputable def expression1 : ℝ := sin (20160 * π / 180) * sin (-20180 * π / 180) + 
                                     cos (20190 * π / 180) * sin (-840 * π / 180)

-- Define the second expression
noncomputable def expression2 : ℝ := sin (4 * π / 3) * cos (19 * π / 6) * tan (21 * π / 4)

-- Theorem for the first expression
theorem expression1_value : expression1 = -3/4 := by sorry

-- Theorem for the second expression
theorem expression2_value : expression2 = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression1_value_expression2_value_l1155_115552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_for_three_roots_l1155_115520

-- Define the piecewise function f
noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
  if x < t then -6 + Real.exp (x - 1) else x^2 - 4*x

-- Define the property of having exactly three distinct roots
def has_three_distinct_roots (t : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f t x₁ = x₁ - 6 ∧ f t x₂ = x₂ - 6 ∧ f t x₃ = x₃ - 6 ∧
    ∀ x, f t x = x - 6 → x = x₁ ∨ x = x₂ ∨ x = x₃

-- Theorem statement
theorem t_range_for_three_roots :
  ∀ t : ℝ, has_three_distinct_roots t ↔ 1 < t ∧ t ≤ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_for_three_roots_l1155_115520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_x4_term_implies_a_zero_l1155_115581

theorem no_x4_term_implies_a_zero (a : ℝ) : 
  (∀ x : ℝ, -5 * x^5 - 5*a*x^4 - 25*x^3 = -5 * x^3 * (x^2 + a*x + 5)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_x4_term_implies_a_zero_l1155_115581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_intersection_points_l1155_115593

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) : ℝ := x^2 - 4*x + 4

-- Define the set of intersection points
def intersection_points : Set ℝ := {x | f x = g x ∧ x > 0}

-- State the theorem
theorem two_intersection_points : ∃ (S : Finset ℝ), S.card = 2 ∧ ∀ x ∈ S, x ∈ intersection_points := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_intersection_points_l1155_115593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_derivative_positive_l1155_115531

-- Define the function f(x) = e^x / x
noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

-- Theorem stating that f is monotonically increasing on (1, +∞)
theorem f_monotone_increasing :
  ∀ x y : ℝ, 1 < x → x < y → f x < f y :=
by
  -- The proof is omitted for now
  sorry

-- Additional lemma to show that f'(x) > 0 for x > 1
theorem f_derivative_positive :
  ∀ x : ℝ, x > 1 → (deriv f) x > 0 :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_derivative_positive_l1155_115531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1155_115517

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x - a * Real.cos x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (2 * x + Real.pi / 3)

theorem function_properties (a : ℝ) :
  (∀ x, f a x + f a ((2 * Real.pi) / 3 - x) = 0) →
  (a = Real.sqrt 3) ∧
  (¬ ∀ x, g a (-x) = -g a x) ∧
  (¬ ∀ x ∈ Set.Ioo (Real.pi / 6) (Real.pi / 2), 
    ∀ y ∈ Set.Ioo (Real.pi / 6) (Real.pi / 2), x < y → g a x < g a y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1155_115517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_three_two_l1155_115570

-- Define the custom operation @
noncomputable def custom_op (a b : ℝ) : ℝ := (a ^ b) / 2

-- Theorem statement
theorem custom_op_three_two : custom_op 3 2 = 4.5 := by
  -- Unfold the definition of custom_op
  unfold custom_op
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_three_two_l1155_115570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_8th_graders_l1155_115563

/-- The number of 8th grade students in a chess tournament -/
def num_8th_grade_students : ℕ := sorry

/-- The number of points scored by each 8th grade student -/
def points_per_8th_grader : ℚ := sorry

/-- The total number of participants in the tournament -/
def total_participants : ℕ := num_8th_grade_students + 2

/-- The total number of games played in the tournament -/
def total_games : ℕ := (total_participants * (total_participants - 1)) / 2

/-- The total number of points distributed in the tournament -/
def total_points : ℚ := total_games

/-- The statement to be proven -/
theorem chess_tournament_8th_graders :
  (8 : ℚ) + num_8th_grade_students * points_per_8th_grader = total_points ∧
  points_per_8th_grader = (num_8th_grade_students + 3) / 2 - 7 / num_8th_grade_students ∧
  (num_8th_grade_students = 7 ∨ num_8th_grade_students = 14) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_8th_graders_l1155_115563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_exponent_calculation_l1155_115503

theorem fraction_exponent_calculation :
  (1 / 3 : ℚ)^9 * (2 / 5 : ℚ)^(-4 : ℤ) = 625 / 314928 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_exponent_calculation_l1155_115503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1155_115578

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def N : Set ℝ := {x | Real.log x / Real.log 2 > 1}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1155_115578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_eq_sqrt_three_l1155_115589

theorem tan_alpha_eq_sqrt_three (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.sin α ^ 2 + Real.cos (2 * α) = 1 / 4) : 
  Real.tan α = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_eq_sqrt_three_l1155_115589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_negative_l1155_115576

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := x * Real.log x - a * (x - 1)^2 - x + 1

-- Theorem for part 1
theorem f_monotonicity (x : ℝ) :
  (∀ y ∈ Set.Ioo 0 1, f 0 x < f 0 y → x < y) ∧
  (∀ y ∈ Set.Ioi 1, f 0 x < f 0 y → x < y) :=
sorry

-- Theorem for part 2
theorem f_negative (x a : ℝ) (hx : x > 1) (ha : a ≥ 1/2) :
  f a x < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_negative_l1155_115576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perspective_drawing_area_l1155_115527

/-- Represents a triangle -/
structure Triangle where
  -- Triangle definition (omitted for brevity)
  mk :: -- Add this line to create a default constructor

/-- Checks if a triangle is equilateral -/
def Triangle.IsEquilateral (t : Triangle) : Prop :=
  sorry -- Definition of equilateral triangle (omitted for brevity)

/-- Returns the side length of a triangle -/
def Triangle.SideLength (t : Triangle) : ℝ :=
  sorry -- Definition of side length (omitted for brevity)

/-- Calculates the area of the perspective drawing of a triangle using oblique projection -/
def AreaOfPerspectiveDrawing (t : Triangle) : ℝ :=
  sorry -- Definition of area calculation for perspective drawing (omitted for brevity)

/-- Given an equilateral triangle with side length 4, 
    the area of its perspective drawing using oblique projection is √6 -/
theorem perspective_drawing_area (ABC : Triangle) (h1 : ABC.IsEquilateral) 
  (h2 : ABC.SideLength = 4) : 
  AreaOfPerspectiveDrawing ABC = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perspective_drawing_area_l1155_115527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cog_production_average_l1155_115597

/-- Represents the production of cogs on an assembly line -/
structure CogProduction where
  initial_rate : ℚ
  initial_order : ℚ
  reduced_rate : ℚ
  second_batch : ℚ

/-- Calculates the overall average output of cog production -/
def overallAverageOutput (prod : CogProduction) : ℚ :=
  let initial_time := prod.initial_order / prod.initial_rate
  let reduced_time := prod.second_batch / prod.reduced_rate
  let total_time := initial_time + reduced_time
  let total_cogs := prod.initial_order + prod.second_batch
  total_cogs / total_time

/-- Theorem stating that the overall average output is 72 cogs per hour -/
theorem cog_production_average (prod : CogProduction)
  (h1 : prod.initial_rate = 90)
  (h2 : prod.initial_order = 60)
  (h3 : prod.reduced_rate = 60)
  (h4 : prod.second_batch = 60) :
  overallAverageOutput prod = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cog_production_average_l1155_115597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_problem_l1155_115508

/-- Represents the capacity of the pool -/
noncomputable def poolCapacity : ℝ := 1

/-- Time to fill empty pool with inlet pipe alone (in hours) -/
noncomputable def inletFillTime : ℝ := 2 + 1/3

/-- Time to drain dirty water with outlet pipe (in hours) -/
noncomputable def outletDrainTime : ℝ := 1 + 1/3

/-- Initial fraction of pool filled with dirty water -/
noncomputable def initialDirtyFraction : ℝ := 1/3

/-- Final fraction of pool filled with clean water -/
noncomputable def finalCleanFraction : ℝ := 1/2

/-- Inlet pipe flow rate (pool capacity per hour) -/
noncomputable def inletRate : ℝ := poolCapacity / inletFillTime

/-- Outlet pipe flow rate (pool capacity per hour) -/
noncomputable def outletRate : ℝ := initialDirtyFraction * poolCapacity / outletDrainTime

/-- Time from opening outlet pipe to closing it (in hours) -/
noncomputable def totalTime : ℝ := outletDrainTime + (finalCleanFraction * poolCapacity) / (inletRate - outletRate)

theorem pool_problem :
  totalTime = 4 + 2/15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_problem_l1155_115508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_odd_function_l1155_115569

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin x - Real.cos x

noncomputable def g (φ : ℝ) (x : ℝ) := f (x - φ)

theorem min_phi_for_odd_function :
  ∀ φ : ℝ, φ > 0 →
  (∀ x : ℝ, g φ (-x) = -(g φ x)) →
  φ ≥ 5 * Real.pi / 6 :=
by
  sorry

#check min_phi_for_odd_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_odd_function_l1155_115569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1155_115583

/-- Given a geometric sequence, calculate the sum of its first n terms -/
noncomputable def geometricSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (a r : ℝ) :
  geometricSum a r 1000 = 500 →
  geometricSum a r 2000 = 950 →
  geometricSum a r 3000 = 1355 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1155_115583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_video_game_purchase_total_l1155_115580

theorem video_game_purchase_total (basketball_price racing_price : ℚ) 
  (tax_rate : ℚ) (h1 : basketball_price = 5.20) (h2 : racing_price = 4.23) 
  (h3 : tax_rate = 0.065) : 
  (basketball_price * (1 + tax_rate)).ceil.toNat / 100 + 
  (racing_price * (1 + tax_rate)).ceil.toNat / 100 = 1004/100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_video_game_purchase_total_l1155_115580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_fifth_term_is_2050_l1155_115530

/-- Function that returns true if a number is a perfect square, false otherwise -/
def isPerfectSquare (n : ℕ) : Prop := 
  ∃ m : ℕ, m * m = n

/-- Function that returns the nth term of the sequence after removing perfect squares -/
def sequenceWithoutSquares (n : ℕ) : ℕ := 
  sorry

theorem two_thousand_fifth_term_is_2050 : 
  sequenceWithoutSquares 2005 = 2050 := by
  sorry

#check two_thousand_fifth_term_is_2050

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_fifth_term_is_2050_l1155_115530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_has_6_in_tens_place_l1155_115524

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧
  (n % 3 = 0) ∧ (n % 5 = 0) ∧
  (∃ a b c d e : ℕ, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    ({a, b, c, d, e} : Finset ℕ) = {1, 2, 3, 5, 6} ∧
    n = 10000 * a + 1000 * b + 100 * c + 10 * d + e)

theorem smallest_valid_number_has_6_in_tens_place :
  ∃ n : ℕ, is_valid_number n ∧
    (∀ m : ℕ, is_valid_number m → n ≤ m) ∧
    (n / 10) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_has_6_in_tens_place_l1155_115524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_equality_l1155_115556

theorem sin_sum_equality : 
  Real.sin (45 * π / 180) * Real.sin (105 * π / 180) + 
  Real.sin (45 * π / 180) * Real.sin (15 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_equality_l1155_115556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1155_115511

-- Define the function f(x) = e^x - x - 2
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

-- Theorem statement
theorem root_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1155_115511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_polygons_perimeter_ratio_l1155_115514

-- Define the necessary concepts
def Similar (P Q : Set (ℝ × ℝ)) : Prop := sorry
def Area (P : Set (ℝ × ℝ)) : ℝ := sorry
def Perimeter (P : Set (ℝ × ℝ)) : ℝ := sorry

theorem similar_polygons_perimeter_ratio 
  (P Q : Set (ℝ × ℝ)) -- P and Q are two polygons in the real plane
  (h_similar : Similar P Q) -- P and Q are similar
  (h_area_ratio : Area P / Area Q = 1 / 2) -- The ratio of their areas is 1:2
  : Perimeter P / Perimeter Q = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_polygons_perimeter_ratio_l1155_115514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_relationship_l1155_115538

-- Define the given values
noncomputable def y₁ : ℝ := (4 : ℝ)^(0.2 : ℝ)
noncomputable def y₂ : ℝ := ((1/2) : ℝ)^(-(0.3 : ℝ))
noncomputable def y₃ : ℝ := Real.log 8 / Real.log (1/2)

-- State the theorem
theorem y_relationship : y₁ > y₂ ∧ y₂ > y₃ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_relationship_l1155_115538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_age_is_41_l1155_115582

-- Define variables for daughter's and mother's ages 5 years ago
def daughter_age_5_years_ago : ℕ := sorry
def mother_age_5_years_ago : ℕ := sorry

-- Define the relationship between mother's and daughter's ages 5 years ago
axiom mother_four_times_daughter : mother_age_5_years_ago = 4 * daughter_age_5_years_ago

-- Define the sum of their ages 8 years from now
axiom sum_of_ages_in_8_years : (daughter_age_5_years_ago + 13) + (mother_age_5_years_ago + 13) = 71

-- Define the mother's current age
def mother_current_age : ℕ := mother_age_5_years_ago + 5

-- Theorem to prove
theorem mother_age_is_41 : mother_current_age = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_age_is_41_l1155_115582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_last_four_digits_l1155_115562

/-- Given a positive integer N where both N and N^2 end in the same sequence 
    of four non-zero digits in base 10, the first three digits of this sequence are 937. -/
theorem same_last_four_digits (N : ℕ+) : 
  (∃ (d : ℕ), d < 10 ∧ 
    (N : ℕ) % 10000 = (N : ℕ)^2 % 10000 ∧ 
    (N : ℕ) % 10000 ≥ 1000) → 
  (N : ℕ) % 10000 ≥ 9370 ∧ (N : ℕ) % 10000 < 9380 := by
  intro h
  -- The proof goes here
  sorry

#check same_last_four_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_last_four_digits_l1155_115562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_rearranged_square_l1155_115516

/-- The perimeter of a figure formed by cutting a square with side length 100 into two equal rectangles
    and placing them next to each other. -/
theorem perimeter_of_rearranged_square : ℕ := by
  let square_side : ℕ := 100
  let rectangle_width : ℕ := square_side
  let rectangle_height : ℕ := square_side / 2
  let new_figure_length : ℕ := square_side
  let new_figure_width : ℕ := rectangle_height
  let perimeter : ℕ := 3 * new_figure_length + 4 * new_figure_width
  have h : perimeter = 500 := by sorry
  exact 500

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_rearranged_square_l1155_115516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1155_115577

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.cos x - 3/4

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 1 ∧ 
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ M) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = M) := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1155_115577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_health_risk_factors_probability_l1155_115512

theorem health_risk_factors_probability (X Y Z : Type) 
  (p_one : ℝ) (p_two : ℝ) (p_three_given_xy : ℝ) 
  (p q : ℕ) : 
  (p_one = 0.08) →
  (p_two = 0.12) →
  (p_three_given_xy = 1/4) →
  (Nat.Coprime p q) →
  (p_one * 3 + p_two * 3 + p_three_given_xy * p_two = 0.64) →
  ((1 - (p_one + 2 * p_two + p_three_given_xy * p_two)) / 
   (1 - (p_one + 2 * p_two + p_three_given_xy * p_two)) = p / q) →
  p + q = 25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_health_risk_factors_probability_l1155_115512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_x_equals_one_l1155_115549

noncomputable def f (x : ℝ) : ℝ := |⌊x + 1⌋| - |⌊2 - x⌋|

theorem symmetry_about_x_equals_one :
  ∀ x : ℝ, f x = f (2 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_x_equals_one_l1155_115549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_theorem_l1155_115537

variable (K : Type*) [Field K]
variable (P : K → K)
variable (a b : K)

theorem polynomial_remainder_theorem (ha : a ≠ b) 
  (h1 : ∃ Q1 : K → K, ∀ x, P x = (x - a) * (Q1 x) + 1)
  (h2 : ∃ Q2 : K → K, ∀ x, P x = (x - b) * (Q2 x) - 1) :
  ∃ Q3 : K → K, ∃ c d : K, 
    (∀ x, P x = (x - a) * (x - b) * (Q3 x) + c * x + d) ∧ 
    c = 2 / (a - b) ∧ 
    d = (b + a) / (b - a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_theorem_l1155_115537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_a_range_l1155_115513

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 3/a| + |x - 2*a|

/-- Part 1: For all real x and non-zero real a, f(x) ≥ 2√6 -/
theorem f_lower_bound (a : ℝ) (ha : a ≠ 0) : ∀ x : ℝ, f a x ≥ 2 * Real.sqrt 6 := by
  sorry

/-- Part 2: For a > 0 and f(2) < 5, 1 < a < 1.5 -/
theorem a_range (a : ℝ) (ha : a > 0) (hf : f a 2 < 5) : 1 < a ∧ a < 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_a_range_l1155_115513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_terms_natural_l1155_115554

/-- Represents our special sequence -/
noncomputable def SpecialSequence (a : ℕ) (q : ℚ) : ℕ → ℚ
  | 0 => a
  | 1 => a * q
  | 2 => a * q^2
  | n + 3 => if n % 2 = 0 
    then (SpecialSequence a q (n + 2) + SpecialSequence a q n) / 2  -- arithmetic mean
    else (SpecialSequence a q (n + 2) * SpecialSequence a q n).sqrt  -- geometric mean

/-- The main theorem stating that all terms in the sequence are natural numbers -/
theorem all_terms_natural (a : ℕ) (q : ℚ) (h : q > 1) :
  ∀ n : ℕ, ∃ k : ℕ, SpecialSequence a q n = k :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_terms_natural_l1155_115554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_travel_time_l1155_115571

/-- K's speed in miles per hour -/
noncomputable def x : ℝ := sorry

/-- M's speed in miles per hour -/
noncomputable def y : ℝ := x - 1

/-- The distance traveled in miles -/
noncomputable def distance : ℝ := 40

/-- The time difference between M and K in hours -/
noncomputable def time_diff : ℝ := 2/3

theorem k_travel_time :
  (distance / y - distance / x = time_diff) →
  (distance / x = 10) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_travel_time_l1155_115571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_relation_l1155_115579

/-- A structure representing a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The volume of a right circular cylinder -/
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- Theorem stating the relationship between the heights of two cylinders with equal volumes -/
theorem cylinder_height_relation (c1 c2 : Cylinder) :
  volume c1 = volume c2 →
  c2.radius = 1.2 * c1.radius →
  c1.height = 1.44 * c2.height := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_relation_l1155_115579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_per_row_l1155_115509

-- Define the area of the room in square feet
def room_area : ℝ := 256

-- Define the width of a tile in inches
def tile_width : ℝ := 8

-- Define the conversion factor from feet to inches
def feet_to_inches : ℝ := 12

-- Theorem statement
theorem tiles_per_row :
  ⌊(Real.sqrt room_area * feet_to_inches) / tile_width⌋ = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_per_row_l1155_115509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_upper_base_length_l1155_115525

/-- Represents the properties of a trapezoid -/
structure Trapezoid where
  area : ℝ
  lower_base : ℝ
  height : ℝ
  upper_base : ℝ

/-- The formula for the area of a trapezoid -/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ :=
  (1/2) * (t.lower_base + t.upper_base) * t.height

/-- Theorem stating that for a trapezoid with given properties, the upper base is 14 -/
theorem trapezoid_upper_base_length
  (t : Trapezoid)
  (h_area : t.area = 222)
  (h_lower_base : t.lower_base = 23)
  (h_height : t.height = 12)
  (h_formula : t.area = trapezoid_area t) :
  t.upper_base = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_upper_base_length_l1155_115525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1155_115500

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (a^2 / ((b+c)/2 + Real.sqrt (b*c))) + 
  (b^2 / ((c+a)/2 + Real.sqrt (c*a))) + 
  (c^2 / ((a+b)/2 + Real.sqrt (a*b))) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1155_115500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_or_two_l1155_115575

/-- Pascal's Triangle row -/
def pascal_row (n : ℕ) : List ℕ :=
  match n with
  | 0 => [1]
  | n + 1 => List.zipWith (· + ·) (0 :: pascal_row n) (pascal_row n ++ [0])

/-- First 20 rows of Pascal's Triangle -/
def pascal_20 : List (List ℕ) :=
  List.map pascal_row (List.range 20)

/-- Total number of elements in the first 20 rows -/
def total_elements : ℕ :=
  (pascal_20.map List.length).sum

/-- Number of 1s and 2s in the first 20 rows -/
def count_ones_and_twos : ℕ :=
  (pascal_20.map (λ row => (row.filter (λ x => x = 1 ∨ x = 2)).length)).sum

/-- Probability of selecting 1 or 2 from the first 20 rows of Pascal's Triangle -/
theorem probability_one_or_two :
  (count_ones_and_twos : ℚ) / total_elements = 5 / 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_or_two_l1155_115575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_lucky_multiple_of_8_l1155_115539

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_lucky (n : ℕ) : Prop :=
  n % (sum_of_digits n) = 0

theorem smallest_non_lucky_multiple_of_8 :
  (∀ k : ℕ, k > 0 ∧ k < 16 ∧ k % 8 = 0 → is_lucky k) ∧
  ¬ is_lucky 16 ∧
  16 % 8 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_lucky_multiple_of_8_l1155_115539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_semicircles_area_l1155_115523

/-- The area of the shaded region in a regular octagon with side length 3 and eight inscribed semicircles -/
theorem octagon_semicircles_area : ∃ (A : ℝ), 
  let s : ℝ := 3
  let octagon_area : ℝ := 2 * (1 + Real.sqrt 2) * s^2
  let semicircle_area : ℝ := 8 * (π * (s/2)^2 / 2)
  A = octagon_area - semicircle_area ∧ 
  A = 54 + 24 * Real.sqrt 2 - 9 * π :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_semicircles_area_l1155_115523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_distance_theorem_l1155_115510

/-- Calculates the distance traveled downstream by a boat given its speed in still water and the time taken for downstream and upstream journeys. -/
noncomputable def distance_downstream (boat_speed : ℝ) (time_downstream time_upstream : ℝ) : ℝ :=
  let current_speed := (boat_speed * (time_upstream - time_downstream)) / (time_upstream + time_downstream)
  (boat_speed + current_speed) * time_downstream

/-- Theorem stating that a boat with a speed of 12 kmph in still water, traveling a certain distance downstream in 3 hours and the same distance upstream in 4.2 hours, travels 42 km downstream. -/
theorem boat_distance_theorem :
  distance_downstream 12 3 4.2 = 42 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_distance_theorem_l1155_115510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bucket_weight_problem_bucket_weight_solution_l1155_115536

theorem bucket_weight_problem (a b : ℚ) :
  let x := (a/5 + 4*b/5 : ℚ)  -- weight of empty bucket
  let y := (12*(b - a)/5 : ℚ) -- weight of water when full
  x + y = 16*b/5 - 11*a/5 := by
  ring

#check bucket_weight_problem

theorem bucket_weight_solution (a b : ℚ) :
  (16*b/5 - 11*a/5 : ℚ) = 
  (a/5 + 4*b/5) + (12*(b - a)/5) := by
  ring

#check bucket_weight_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bucket_weight_problem_bucket_weight_solution_l1155_115536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1155_115594

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  R : ℝ -- radius of circumscribed circle

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 21 ∧
  t.c - t.b = 1 ∧
  t.R = Real.sqrt 7

-- Define the area of the triangle
noncomputable def Triangle.area (t : Triangle) : ℝ :=
  1 / 2 * t.b * t.c * Real.sin t.A

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  Real.sin t.A = Real.sqrt 3 / 2 ∧
  (t.area = 5 * Real.sqrt 3 ∨ t.area = 5 * Real.sqrt 3 / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1155_115594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_of_f_l1155_115546

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 + Real.sqrt (1 - x^2)

-- Define the domain of f
def domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

-- Theorem statement
theorem enclosed_area_of_f (f : ℝ → ℝ) (h : ∀ x ∈ domain, f x = 1 + Real.sqrt (1 - x^2)) :
  (∫ x in Set.Icc 0 1, (f x - x)) * 2 = Real.pi / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_of_f_l1155_115546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coordinates_equals_4_5_l1155_115545

-- Define the piecewise function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -2 then 3*x + 6
  else if -2 < x ∧ x ≤ 0 then -x + 2
  else if 0 < x ∧ x ≤ 3 then 2*x - 2
  else if 3 < x ∧ x ≤ 5 then -2*x + 8
  else 0  -- Default value for x outside the defined ranges

-- Theorem statement
theorem sum_of_x_coordinates_equals_4_5 :
  ∃ (x₁ x₂ x₃ : ℝ),
    g x₁ = 2.5 ∧ g x₂ = 2.5 ∧ g x₃ = 2.5 ∧
    (∀ x, g x = 2.5 → x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    x₁ + x₂ + x₃ = 4.5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coordinates_equals_4_5_l1155_115545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_range_l1155_115598

/-- An ellipse with equation x²/3 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 3 + p.2^2 = 1}

/-- The vertex A of the ellipse -/
def A : ℝ × ℝ := (0, -1)

/-- A line y = kx + m -/
def Line (k m : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = k * p.1 + m}

/-- Theorem: If a line y = kx + m (k ≠ 0) intersects the ellipse at two distinct
    points M and N such that |AM| = |AN|, then 1/2 < m < 2 -/
theorem ellipse_intersection_range (k m : ℝ) (hk : k ≠ 0)
    (M N : ℝ × ℝ)
    (hM : M ∈ Ellipse ∩ Line k m) (hN : N ∈ Ellipse ∩ Line k m) (hMN : M ≠ N)
    (hAM_eq_AN : ‖A - M‖ = ‖A - N‖) :
    1/2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_range_l1155_115598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_subset_perfect_square_l1155_115592

theorem product_subset_perfect_square 
  (n : ℕ) 
  (h_n : n = 1986) 
  (a : Fin n → ℕ) 
  (h_prime_divisors : (Finset.univ.prod (λ i => a i)).factors.card = 1985) :
  ∃ (s : Finset (Fin n)), s.Nonempty ∧ ∃ (m : ℕ), s.prod a = m^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_subset_perfect_square_l1155_115592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pentagon_theorem_l1155_115541

/-- A pentagon with two right angles and three known angles -/
structure SpecialPentagon where
  /-- The measure of angle P in degrees -/
  angle_P : ℝ
  /-- The measure of angle Q in degrees -/
  angle_Q : ℝ
  /-- The measure of angle R in degrees -/
  angle_R : ℝ
  /-- The measure of angle a in degrees -/
  a : ℝ
  /-- The measure of angle b in degrees -/
  b : ℝ
  /-- The sum of all angles in the pentagon is 540° -/
  angle_sum : angle_P + angle_Q + angle_R + a + b = 540
  /-- The pentagon has two right angles -/
  right_angles : a + b + 180 = 360
  /-- Angle P measures 34° -/
  P_measure : angle_P = 34
  /-- Angle Q measures 80° -/
  Q_measure : angle_Q = 80
  /-- Angle R measures 30° -/
  R_measure : angle_R = 30

/-- The theorem stating that the sum of the two unknown angles is 144° -/
theorem special_pentagon_theorem (p : SpecialPentagon) : p.a + p.b = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pentagon_theorem_l1155_115541
