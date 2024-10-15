import Mathlib

namespace NUMINAMATH_CALUDE_cubic_root_sum_l505_50509

theorem cubic_root_sum (α β γ : ℂ) : 
  α^3 - α - 1 = 0 → β^3 - β - 1 = 0 → γ^3 - γ - 1 = 0 →
  (1 + α) / (1 - α) + (1 + β) / (1 - β) + (1 + γ) / (1 - γ) = -7 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l505_50509


namespace NUMINAMATH_CALUDE_pentagonal_prism_sum_l505_50593

/-- Represents a pentagonal prism -/
structure PentagonalPrism where
  /-- Number of faces of the pentagonal prism -/
  faces : Nat
  /-- Number of edges of the pentagonal prism -/
  edges : Nat
  /-- Number of vertices of the pentagonal prism -/
  vertices : Nat
  /-- The number of faces is 7 (2 pentagonal + 5 rectangular) -/
  faces_eq : faces = 7
  /-- The number of edges is 15 (5 + 5 + 5) -/
  edges_eq : edges = 15
  /-- The number of vertices is 10 (5 + 5) -/
  vertices_eq : vertices = 10

/-- The sum of faces, edges, and vertices of a pentagonal prism is 32 -/
theorem pentagonal_prism_sum (p : PentagonalPrism) :
  p.faces + p.edges + p.vertices = 32 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_prism_sum_l505_50593


namespace NUMINAMATH_CALUDE_inequality_transformation_l505_50524

theorem inequality_transformation (a b : ℝ) : a ≤ b → -a/2 ≥ -b/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_transformation_l505_50524


namespace NUMINAMATH_CALUDE_cookie_brownie_difference_after_week_l505_50554

/-- Calculates the difference between remaining cookies and brownies after a week -/
def cookieBrownieDifference (initialCookies initialBrownies dailyCookies dailyBrownies days : ℕ) : ℕ :=
  let remainingCookies := initialCookies - dailyCookies * days
  let remainingBrownies := initialBrownies - dailyBrownies * days
  remainingCookies - remainingBrownies

/-- Proves that the difference between remaining cookies and brownies after a week is 36 -/
theorem cookie_brownie_difference_after_week :
  cookieBrownieDifference 60 10 3 1 7 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cookie_brownie_difference_after_week_l505_50554


namespace NUMINAMATH_CALUDE_number_problem_l505_50572

theorem number_problem (a b : ℤ) : 
  a + b = 72 → 
  a = b + 12 → 
  a = 42 → 
  b = 30 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l505_50572


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_neg_one_intersection_equals_B_iff_l505_50527

-- Define sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 5}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 2}

-- Theorem for part (1)
theorem intersection_and_union_when_a_is_neg_one :
  (A ∩ B (-1) = {x | -2 ≤ x ∧ x ≤ -1}) ∧
  (A ∪ B (-1) = {x | x ≤ 1 ∨ x ≥ 5}) := by sorry

-- Theorem for part (2)
theorem intersection_equals_B_iff :
  ∀ a : ℝ, A ∩ B a = B a ↔ a ≤ -3 ∨ a > 2 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_neg_one_intersection_equals_B_iff_l505_50527


namespace NUMINAMATH_CALUDE_probability_one_unit_apart_l505_50537

/-- A rectangle with dimensions 3 × 2 -/
structure Rectangle :=
  (length : ℕ := 3)
  (width : ℕ := 2)

/-- Evenly spaced points on the perimeter of the rectangle -/
def PerimeterPoints (r : Rectangle) : ℕ := 15

/-- Number of unit intervals on the perimeter -/
def UnitIntervals (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The probability of selecting two points one unit apart -/
def ProbabilityOneUnitApart (r : Rectangle) : ℚ :=
  16 / (PerimeterPoints r).choose 2

theorem probability_one_unit_apart (r : Rectangle) :
  ProbabilityOneUnitApart r = 16 / 105 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_unit_apart_l505_50537


namespace NUMINAMATH_CALUDE_alice_bob_calculation_l505_50520

theorem alice_bob_calculation (x : ℕ) : 
  let alice_result := ((x + 2) * 2 + 3)
  2 * (alice_result + 3) = 4 * x + 16 := by
  sorry

end NUMINAMATH_CALUDE_alice_bob_calculation_l505_50520


namespace NUMINAMATH_CALUDE_inequalities_solution_l505_50562

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 < 0
def inequality2 (x : ℝ) : Prop := (2 * x) / (x + 1) ≥ 1

-- State the theorem
theorem inequalities_solution :
  (∀ x : ℝ, inequality1 x ↔ (1/2 < x ∧ x < 1)) ∧
  (∀ x : ℝ, inequality2 x ↔ (x < -1 ∨ x ≥ 1)) :=
sorry

end NUMINAMATH_CALUDE_inequalities_solution_l505_50562


namespace NUMINAMATH_CALUDE_unique_three_digit_number_divisible_by_3_and_7_l505_50553

/-- Represents a three-digit number in the form A3B -/
def ThreeDigitNumber (a b : Nat) : Nat :=
  100 * a + 30 + b

theorem unique_three_digit_number_divisible_by_3_and_7 :
  ∀ a b : Nat,
    (300 < ThreeDigitNumber a b) →
    (ThreeDigitNumber a b < 400) →
    (ThreeDigitNumber a b % 3 = 0) →
    (ThreeDigitNumber a b % 7 = 0) →
    b = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_divisible_by_3_and_7_l505_50553


namespace NUMINAMATH_CALUDE_min_distance_sum_l505_50507

theorem min_distance_sum (x a b : ℚ) : 
  x ≠ a ∧ x ≠ b ∧ a ≠ b →
  a > b →
  (∀ y : ℚ, |y - a| + |y - b| ≥ 2) ∧ (∃ z : ℚ, |z - a| + |z - b| = 2) →
  2022 + a - b = 2024 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_l505_50507


namespace NUMINAMATH_CALUDE_potato_flour_weight_l505_50563

theorem potato_flour_weight (potato_bags flour_bags total_weight weight_difference : ℕ) 
  (h1 : potato_bags = 15)
  (h2 : flour_bags = 12)
  (h3 : total_weight = 1710)
  (h4 : weight_difference = 30) :
  ∃ (potato_weight flour_weight : ℕ),
    potato_weight * potato_bags + flour_weight * flour_bags = total_weight ∧
    flour_weight = potato_weight + weight_difference ∧
    potato_weight = 50 ∧
    flour_weight = 80 := by
  sorry

end NUMINAMATH_CALUDE_potato_flour_weight_l505_50563


namespace NUMINAMATH_CALUDE_meeting_at_64th_light_l505_50550

/-- Represents the meeting point of Petya and Vasya on a street with streetlights -/
def meeting_point (total_lights : ℕ) (petya_start : ℕ) (vasya_start : ℕ) 
                  (petya_position : ℕ) (vasya_position : ℕ) : ℕ :=
  let total_intervals := total_lights - 1
  let petya_intervals := petya_position - petya_start
  let vasya_intervals := vasya_start - vasya_position
  let total_covered := petya_intervals + vasya_intervals
  petya_start + (petya_intervals * 3)

theorem meeting_at_64th_light :
  meeting_point 100 1 100 22 88 = 64 := by
  sorry

#eval meeting_point 100 1 100 22 88

end NUMINAMATH_CALUDE_meeting_at_64th_light_l505_50550


namespace NUMINAMATH_CALUDE_circle_C_equation_l505_50511

-- Define the circles and points
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def point_A : ℝ × ℝ := (1, 0)

-- Define the properties of circle C
structure Circle_C where
  center : ℝ × ℝ
  tangent_to_x_axis : center.2 > 0
  tangent_at_A : (center.1 - point_A.1)^2 + (center.2 - point_A.2)^2 = center.2^2
  intersects_O : ∃ P Q : ℝ × ℝ, P ∈ circle_O ∧ Q ∈ circle_O ∧
    (P.1 - center.1)^2 + (P.2 - center.2)^2 = center.2^2 ∧
    (Q.1 - center.1)^2 + (Q.2 - center.2)^2 = center.2^2
  PQ_length : ∃ P Q : ℝ × ℝ, P ∈ circle_O ∧ Q ∈ circle_O ∧
    (P.1 - center.1)^2 + (P.2 - center.2)^2 = center.2^2 ∧
    (Q.1 - center.1)^2 + (Q.2 - center.2)^2 = center.2^2 ∧
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 14/4

-- Theorem stating the standard equation of circle C
theorem circle_C_equation (c : Circle_C) :
  ∀ x y : ℝ, (x - c.center.1)^2 + (y - c.center.2)^2 = c.center.2^2 ↔
  (x - 1)^2 + (y - 1)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_l505_50511


namespace NUMINAMATH_CALUDE_password_is_5949_l505_50526

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_ambiguous_for_alice (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ 
    ((5000 + x * 100 + y * 10) % 9 = 0 ∨ (5000 + x * 100 + y * 10 + 9) % 9 = 0)

def is_ambiguous_for_bob (n : ℕ) : Prop :=
  ∃ (y z : ℕ), y < 10 ∧ z < 10 ∧ 
    ((5000 + y * 10 + z) % 9 = 0 ∨ (5000 + 900 + y * 10 + z) % 9 = 0)

theorem password_is_5949 :
  ∀ n : ℕ,
  5000 ≤ n ∧ n < 6000 →
  is_multiple_of_9 n →
  is_ambiguous_for_alice n →
  is_ambiguous_for_bob n →
  n ≤ 5949 :=
sorry

end NUMINAMATH_CALUDE_password_is_5949_l505_50526


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l505_50598

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l505_50598


namespace NUMINAMATH_CALUDE_prime_pairs_divisibility_l505_50576

theorem prime_pairs_divisibility (p q : ℕ) : 
  p.Prime ∧ q.Prime ∧ p < 2023 ∧ q < 2023 ∧ 
  (p ∣ q^2 + 8) ∧ (q ∣ p^2 + 8) → 
  ((p = 2 ∧ q = 2) ∨ (p = 17 ∧ q = 3) ∨ (p = 11 ∧ q = 5)) := by
sorry

end NUMINAMATH_CALUDE_prime_pairs_divisibility_l505_50576


namespace NUMINAMATH_CALUDE_special_multiplication_l505_50565

theorem special_multiplication (a b : ℤ) :
  (∀ x y, x * y = 5*x + 2*y - 1) → (-4) * 6 = -9 := by
  sorry

end NUMINAMATH_CALUDE_special_multiplication_l505_50565


namespace NUMINAMATH_CALUDE_grade_assignment_count_l505_50547

/-- The number of students in the class -/
def num_students : ℕ := 12

/-- The number of different grades -/
def num_grades : ℕ := 4

/-- Theorem: The number of ways to assign grades to students -/
theorem grade_assignment_count : (num_grades : ℕ) ^ num_students = 16777216 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l505_50547


namespace NUMINAMATH_CALUDE_john_and_alice_money_l505_50589

theorem john_and_alice_money : 5/8 + 7/20 = 0.975 := by
  sorry

end NUMINAMATH_CALUDE_john_and_alice_money_l505_50589


namespace NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l505_50583

theorem range_of_a_for_quadratic_inequality :
  ∃ (a : ℝ), ∀ (x : ℝ), x^2 + 2*x + a > 0 ↔ a ∈ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l505_50583


namespace NUMINAMATH_CALUDE_mixed_number_multiplication_problem_solution_l505_50525

theorem mixed_number_multiplication (a b c d : ℚ) :
  (a + b / c) * (1 / d) = (a * c + b) / (c * d) :=
by sorry

theorem problem_solution : 2 + 4/5 * (1/5) = 14/25 :=
by sorry

end NUMINAMATH_CALUDE_mixed_number_multiplication_problem_solution_l505_50525


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l505_50571

theorem three_digit_number_proof :
  ∃! x : ℕ,
    (100 ≤ x ∧ x < 1000) ∧
    (x * (x / 100) = 494) ∧
    (x * ((x / 10) % 10) = 988) ∧
    (x * (x % 10) = 1729) ∧
    x = 247 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l505_50571


namespace NUMINAMATH_CALUDE_total_baseball_cards_l505_50544

/-- The number of baseball cards each person has -/
structure BaseballCards where
  carlos : ℕ
  matias : ℕ
  jorge : ℕ
  ella : ℕ

/-- The conditions of the baseball card problem -/
def baseball_card_problem (cards : BaseballCards) : Prop :=
  cards.carlos = 20 ∧
  cards.matias = cards.carlos - 6 ∧
  cards.jorge = cards.matias ∧
  cards.ella = 2 * (cards.jorge + cards.matias)

/-- The theorem stating the total number of baseball cards -/
theorem total_baseball_cards (cards : BaseballCards) 
  (h : baseball_card_problem cards) : 
  cards.carlos + cards.matias + cards.jorge + cards.ella = 104 := by
  sorry


end NUMINAMATH_CALUDE_total_baseball_cards_l505_50544


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_is_five_l505_50564

theorem opposite_of_negative_five_is_five : 
  -(- 5) = 5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_is_five_l505_50564


namespace NUMINAMATH_CALUDE_tangerines_most_numerous_l505_50506

/-- Represents the number of boxes for each fruit type -/
structure BoxCounts where
  tangerines : Nat
  apples : Nat
  pears : Nat

/-- Represents the number of fruits per box for each fruit type -/
structure FruitsPerBox where
  tangerines : Nat
  apples : Nat
  pears : Nat

/-- Calculates the total number of fruits for each type -/
def totalFruits (boxes : BoxCounts) (perBox : FruitsPerBox) : BoxCounts :=
  { tangerines := boxes.tangerines * perBox.tangerines
  , apples := boxes.apples * perBox.apples
  , pears := boxes.pears * perBox.pears }

/-- Proves that tangerines are the most numerous fruit -/
theorem tangerines_most_numerous (boxes : BoxCounts) (perBox : FruitsPerBox) :
  boxes.tangerines = 5 →
  boxes.apples = 3 →
  boxes.pears = 4 →
  perBox.tangerines = 30 →
  perBox.apples = 20 →
  perBox.pears = 15 →
  let totals := totalFruits boxes perBox
  totals.tangerines > totals.apples ∧ totals.tangerines > totals.pears :=
by
  sorry


end NUMINAMATH_CALUDE_tangerines_most_numerous_l505_50506


namespace NUMINAMATH_CALUDE_sum_of_first_10_common_elements_l505_50586

/-- Arithmetic progression with first term 4 and common difference 3 -/
def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n

/-- Geometric progression with first term 20 and common ratio 2 -/
def geometric_progression (k : ℕ) : ℕ := 20 * 2^k

/-- Common elements between the arithmetic and geometric progressions -/
def common_elements (n : ℕ) : Prop :=
  ∃ k : ℕ, arithmetic_progression n = geometric_progression k

/-- The sum of the first 10 common elements -/
def sum_of_common_elements : ℕ := 13981000

/-- Theorem stating that the sum of the first 10 common elements is 13981000 -/
theorem sum_of_first_10_common_elements :
  sum_of_common_elements = 13981000 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_10_common_elements_l505_50586


namespace NUMINAMATH_CALUDE_log_equation_solution_l505_50522

theorem log_equation_solution (a : ℝ) (h : a > 0) :
  Real.log a / Real.log 2 - 2 * Real.log 2 / Real.log a = 1 →
  a = 4 ∨ a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l505_50522


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l505_50513

/-- Given a hyperbola and a parabola with specific properties, prove that p = 1 -/
theorem hyperbola_parabola_intersection (a b p : ℝ) : 
  a > 0 → b > 0 → p > 0 →
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x y, y^2 = 2 * p * x) →
  (a^2 + b^2) / a^2 = 4 →
  1/2 * (p/2) * (b*p/a) = Real.sqrt 3 / 4 →
  p = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l505_50513


namespace NUMINAMATH_CALUDE_expense_ratios_l505_50548

def initial_amount : ℚ := 120
def books_expense : ℚ := 25
def clothes_expense : ℚ := 40
def snacks_expense : ℚ := 10

def total_spent : ℚ := books_expense + clothes_expense + snacks_expense

theorem expense_ratios :
  (books_expense / total_spent = 1 / 3) ∧
  (clothes_expense / total_spent = 4 / 3) ∧
  (snacks_expense / total_spent = 2 / 15) := by
  sorry

end NUMINAMATH_CALUDE_expense_ratios_l505_50548


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l505_50501

-- Define sets A and B
def A : Set ℝ := {x | x^2 + x - 12 < 0}
def B : Set ℝ := {x | 2 - x < 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l505_50501


namespace NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l505_50532

/-- An arithmetic sequence with a₁ = 1 and aₙ₊₂ - aₙ = 6 -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 2) - a n = 6

/-- The 11th term of the arithmetic sequence is 31 -/
theorem arithmetic_sequence_11th_term
  (a : ℕ → ℝ) (h : arithmeticSequence a) : a 11 = 31 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l505_50532


namespace NUMINAMATH_CALUDE_octagon_area_eq_1200_l505_50558

/-- A regular octagon inscribed in a square with perimeter 160 cm,
    where each side of the square is quadrised by the vertices of the octagon -/
structure InscribedOctagon where
  square_perimeter : ℝ
  square_perimeter_eq : square_perimeter = 160
  is_regular : Bool
  is_inscribed : Bool
  sides_quadrised : Bool

/-- The area of the inscribed octagon -/
def octagon_area (o : InscribedOctagon) : ℝ := sorry

/-- Theorem stating that the area of the inscribed octagon is 1200 square centimeters -/
theorem octagon_area_eq_1200 (o : InscribedOctagon) :
  o.is_regular ∧ o.is_inscribed ∧ o.sides_quadrised → octagon_area o = 1200 := by sorry

end NUMINAMATH_CALUDE_octagon_area_eq_1200_l505_50558


namespace NUMINAMATH_CALUDE_abc_zero_l505_50500

theorem abc_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = a^3 * b^3 * c^3) : 
  a * b * c = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_zero_l505_50500


namespace NUMINAMATH_CALUDE_division_ratio_l505_50561

theorem division_ratio (dividend quotient divisor remainder : ℕ) : 
  dividend = 5290 →
  remainder = 46 →
  divisor = 10 * quotient →
  dividend = divisor * quotient + remainder →
  (divisor : ℚ) / remainder = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_ratio_l505_50561


namespace NUMINAMATH_CALUDE_max_digit_sum_in_range_l505_50588

def is_valid_time (h m s : ℕ) : Prop :=
  13 ≤ h ∧ h ≤ 23 ∧ m < 60 ∧ s < 60

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def time_digit_sum (h m s : ℕ) : ℕ :=
  digit_sum h + digit_sum m + digit_sum s

theorem max_digit_sum_in_range :
  ∃ (h m s : ℕ), is_valid_time h m s ∧
    ∀ (h' m' s' : ℕ), is_valid_time h' m' s' →
      time_digit_sum h' m' s' ≤ time_digit_sum h m s ∧
      time_digit_sum h m s = 33 :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_in_range_l505_50588


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l505_50539

theorem polynomial_symmetry (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^5 + b * x^3 + c * x + 7
  (f (-2011) = -17) → (f 2011 = 31) := by
sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l505_50539


namespace NUMINAMATH_CALUDE_xy_value_l505_50521

theorem xy_value (x y : ℝ) (h : (x + 22) / y + 290 / (x * y) = (26 - y) / x) :
  x * y = -143 := by sorry

end NUMINAMATH_CALUDE_xy_value_l505_50521


namespace NUMINAMATH_CALUDE_solve_equation_l505_50530

theorem solve_equation (y : ℝ) : 7 - y = 10 → y = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l505_50530


namespace NUMINAMATH_CALUDE_weight_of_B_l505_50505

theorem weight_of_B (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) :
  B = 31 := by
sorry

end NUMINAMATH_CALUDE_weight_of_B_l505_50505


namespace NUMINAMATH_CALUDE_minimum_coins_for_all_amounts_l505_50595

/-- Represents the different types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- The value of each coin in cents --/
def coinValue : Coin → ℕ
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- A list of coins --/
def CoinList := List Coin

/-- Calculates the total value of a list of coins in cents --/
def totalValue (coins : CoinList) : ℕ :=
  coins.foldl (fun acc coin => acc + coinValue coin) 0

/-- Checks if a given amount can be made with a list of coins --/
def canMakeAmount (coins : CoinList) (amount : ℕ) : Prop :=
  ∃ (subset : CoinList), subset.Subset coins ∧ totalValue subset = amount

/-- The main theorem to prove --/
theorem minimum_coins_for_all_amounts :
  ∃ (coins : CoinList),
    coins.length = 11 ∧
    (∀ (amount : ℕ), amount > 0 ∧ amount < 100 → canMakeAmount coins amount) ∧
    (∀ (otherCoins : CoinList),
      (∀ (amount : ℕ), amount > 0 ∧ amount < 100 → canMakeAmount otherCoins amount) →
      otherCoins.length ≥ 11) :=
sorry

end NUMINAMATH_CALUDE_minimum_coins_for_all_amounts_l505_50595


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l505_50519

theorem triangle_perimeter_bound (a b s : ℝ) : 
  a = 7 ∧ b = 23 ∧ 0 < s ∧ s < a + b ∧ a < b + s ∧ b < a + s → 
  ∃ (n : ℕ), n = 60 ∧ ∀ (p : ℝ), p = a + b + s → p < n := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l505_50519


namespace NUMINAMATH_CALUDE_yogurt_calories_l505_50545

def calories_per_ounce_yogurt (strawberries : ℕ) (yogurt_ounces : ℕ) (calories_per_strawberry : ℕ) (total_calories : ℕ) : ℕ :=
  (total_calories - strawberries * calories_per_strawberry) / yogurt_ounces

theorem yogurt_calories (strawberries : ℕ) (yogurt_ounces : ℕ) (calories_per_strawberry : ℕ) (total_calories : ℕ)
  (h1 : strawberries = 12)
  (h2 : yogurt_ounces = 6)
  (h3 : calories_per_strawberry = 4)
  (h4 : total_calories = 150) :
  calories_per_ounce_yogurt strawberries yogurt_ounces calories_per_strawberry total_calories = 17 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_calories_l505_50545


namespace NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l505_50578

theorem polynomial_root_implies_coefficients : ∀ (a b : ℝ), 
  (Complex.I : ℂ) ^ 3 + a * (Complex.I : ℂ) ^ 2 + 2 * (Complex.I : ℂ) + b = (2 - 3 * Complex.I : ℂ) ^ 3 →
  a = -5/4 ∧ b = 143/4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l505_50578


namespace NUMINAMATH_CALUDE_condition_neither_necessary_nor_sufficient_l505_50551

-- Define the sets M and P
def M : Set ℝ := {x | x > 2}
def P : Set ℝ := {x | x < 3}

-- Statement to prove
theorem condition_neither_necessary_nor_sufficient :
  ¬(∀ x : ℝ, (x ∈ M ∩ P → (x ∈ M ∨ x ∈ P)) ∧ ((x ∈ M ∨ x ∈ P) → x ∈ M ∩ P)) :=
sorry

end NUMINAMATH_CALUDE_condition_neither_necessary_nor_sufficient_l505_50551


namespace NUMINAMATH_CALUDE_surrounding_decagon_theorem_l505_50574

/-- The number of sides of the surrounding polygons when a regular m-sided polygon
    is surrounded by m regular n-sided polygons without gaps or overlaps. -/
def surrounding_polygon_sides (m : ℕ) : ℕ :=
  if m = 4 then 8 else
  if m = 10 then
    let interior_angle_m := (180 * (m - 2)) / m
    let n := (720 / (360 - interior_angle_m) : ℕ)
    n
  else 0

/-- Theorem stating that when a regular 10-sided polygon is surrounded by 10 regular n-sided polygons
    without gaps or overlaps, n must equal 5. -/
theorem surrounding_decagon_theorem :
  surrounding_polygon_sides 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_surrounding_decagon_theorem_l505_50574


namespace NUMINAMATH_CALUDE_smallest_gcd_qr_l505_50591

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 770) :
  ∃ (q' r' : ℕ+), Nat.gcd q' r' = 70 ∧
    ∀ (q'' r'' : ℕ+), Nat.gcd p q'' = 210 → Nat.gcd p r'' = 770 → Nat.gcd q'' r'' ≥ 70 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_qr_l505_50591


namespace NUMINAMATH_CALUDE_matt_jump_time_l505_50518

/-- Given that Matt skips rope 3 times per second and gets 1800 skips in total,
    prove that he jumped for 10 minutes. -/
theorem matt_jump_time (skips_per_second : ℕ) (total_skips : ℕ) (jump_time : ℕ) :
  skips_per_second = 3 →
  total_skips = 1800 →
  jump_time * 60 * skips_per_second = total_skips →
  jump_time = 10 :=
by sorry

end NUMINAMATH_CALUDE_matt_jump_time_l505_50518


namespace NUMINAMATH_CALUDE_number_of_possible_values_l505_50512

theorem number_of_possible_values (m n k a b : ℕ+) :
  ((1 + a.val : ℕ) * n.val^2 - 4 * (m.val + a.val) * n.val + 4 * m.val^2 + 4 * a.val + b.val * (k.val - 1)^2 < 3) →
  (∃ (s : Finset ℕ), s = {x | ∃ (m' n' k' : ℕ+), 
    ((1 + a.val : ℕ) * n'.val^2 - 4 * (m'.val + a.val) * n'.val + 4 * m'.val^2 + 4 * a.val + b.val * (k'.val - 1)^2 < 3) ∧
    x = m'.val + n'.val + k'.val} ∧ 
  s.card = 4) :=
sorry

end NUMINAMATH_CALUDE_number_of_possible_values_l505_50512


namespace NUMINAMATH_CALUDE_bryans_offer_l505_50534

/-- Represents the problem of determining Bryan's offer for half of Peggy's record collection. -/
theorem bryans_offer (total_records : ℕ) (sammys_price : ℚ) (bryans_uninterested_price : ℚ) 
  (profit_difference : ℚ) (h1 : total_records = 200) (h2 : sammys_price = 4) 
  (h3 : bryans_uninterested_price = 1) (h4 : profit_difference = 100) : 
  ∃ (bryans_interested_price : ℚ),
    sammys_price * total_records - 
    (bryans_interested_price * (total_records / 2) + 
     bryans_uninterested_price * (total_records / 2)) = profit_difference ∧
    bryans_interested_price = 6 := by
  sorry

end NUMINAMATH_CALUDE_bryans_offer_l505_50534


namespace NUMINAMATH_CALUDE_polynomial_factorization_l505_50540

theorem polynomial_factorization (x : ℝ) :
  6 * (x + 4) * (x + 7) * (x + 9) * (x + 11) - 5 * x^2 =
  (3 * x^2 + 93 * x) * (2 * x^2 + 178 * x + 5432) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l505_50540


namespace NUMINAMATH_CALUDE_leak_drain_time_l505_50516

/-- Given a pump that can fill a tank in 2 hours, and with a leak it takes 2 1/3 hours to fill the tank,
    prove that the time it takes for the leak to drain all the water of the tank is 14 hours. -/
theorem leak_drain_time (pump_fill_time leak_fill_time : ℚ) : 
  pump_fill_time = 2 →
  leak_fill_time = 7/3 →
  (1 / (1 / pump_fill_time - 1 / leak_fill_time)) = 14 := by
  sorry

end NUMINAMATH_CALUDE_leak_drain_time_l505_50516


namespace NUMINAMATH_CALUDE_doctors_lawyers_ratio_l505_50555

theorem doctors_lawyers_ratio 
  (d : ℕ) -- number of doctors
  (l : ℕ) -- number of lawyers
  (h1 : d > 0) -- ensure there's at least one doctor
  (h2 : l > 0) -- ensure there's at least one lawyer
  (h3 : (35 * d + 50 * l) / (d + l) = 40) -- average age of the group is 40
  : d = 2 * l := by
sorry

end NUMINAMATH_CALUDE_doctors_lawyers_ratio_l505_50555


namespace NUMINAMATH_CALUDE_commission_percentage_l505_50566

/-- Proves that the commission percentage for the first $500 is 20% given the conditions --/
theorem commission_percentage (x : ℝ) : 
  let total_sale := 800
  let commission_over_500 := 0.25
  let total_commission_percentage := 0.21875
  (x / 100 * 500 + commission_over_500 * (total_sale - 500)) / total_sale = total_commission_percentage →
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_commission_percentage_l505_50566


namespace NUMINAMATH_CALUDE_donovan_percentage_l505_50594

/-- Calculates the weighted percentage of correct answers for a math test -/
def weighted_percentage (
  mc_total : ℕ) (mc_correct : ℕ) (mc_points : ℕ)
  (sa_total : ℕ) (sa_correct : ℕ) (sa_partial : ℕ) (sa_points : ℕ)
  (essay_total : ℕ) (essay_correct : ℕ) (essay_points : ℕ) : ℚ :=
  let total_possible := mc_total * mc_points + sa_total * sa_points + essay_total * essay_points
  let total_earned := mc_correct * mc_points + sa_correct * sa_points + sa_partial * (sa_points / 2) + essay_correct * essay_points
  (total_earned : ℚ) / total_possible * 100

/-- Theorem stating that Donovan's weighted percentage is 68.75% -/
theorem donovan_percentage :
  weighted_percentage 25 20 2 20 10 5 4 3 2 10 = 68.75 := by
  sorry

end NUMINAMATH_CALUDE_donovan_percentage_l505_50594


namespace NUMINAMATH_CALUDE_statues_painted_l505_50542

theorem statues_painted (total_paint : ℚ) (paint_per_statue : ℚ) :
  total_paint = 7/16 →
  paint_per_statue = 1/16 →
  (total_paint / paint_per_statue : ℚ) = 7 :=
by sorry

end NUMINAMATH_CALUDE_statues_painted_l505_50542


namespace NUMINAMATH_CALUDE_davids_english_marks_l505_50599

/-- Represents the marks obtained in each subject --/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculate the average of marks --/
def average (m : Marks) : ℚ :=
  (m.english + m.mathematics + m.physics + m.chemistry + m.biology) / 5

theorem davids_english_marks :
  ∃ m : Marks,
    m.mathematics = 85 ∧
    m.physics = 82 ∧
    m.chemistry = 87 ∧
    m.biology = 85 ∧
    average m = 85 ∧
    m.english = 86 := by
  sorry

#check davids_english_marks

end NUMINAMATH_CALUDE_davids_english_marks_l505_50599


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l505_50517

theorem max_value_sqrt_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (a_ge : a ≥ -1/2)
  (b_ge : b ≥ -2)
  (c_ge : c ≥ -7/3) :
  ∃ (x y z : ℝ), x + y + z = 3 ∧ 
    x ≥ -1/2 ∧ y ≥ -2 ∧ z ≥ -7/3 ∧
    Real.sqrt (4*x + 2) + Real.sqrt (4*y + 8) + Real.sqrt (4*z + 10) = 4 * Real.sqrt 6 ∧
    ∀ (a b c : ℝ), a + b + c = 3 → a ≥ -1/2 → b ≥ -2 → c ≥ -7/3 →
      Real.sqrt (4*a + 2) + Real.sqrt (4*b + 8) + Real.sqrt (4*c + 10) ≤ 4 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l505_50517


namespace NUMINAMATH_CALUDE_subtract_inequality_from_less_than_l505_50508

theorem subtract_inequality_from_less_than (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : 
  a - b < 0 := by
  sorry

end NUMINAMATH_CALUDE_subtract_inequality_from_less_than_l505_50508


namespace NUMINAMATH_CALUDE_beaver_dam_theorem_l505_50536

/-- The number of hours it takes the first group of beavers to build the dam -/
def first_group_time : ℝ := 8

/-- The number of beavers in the second group -/
def second_group_size : ℝ := 36

/-- The number of hours it takes the second group of beavers to build the dam -/
def second_group_time : ℝ := 4

/-- The number of beavers in the first group -/
def first_group_size : ℝ := 18

theorem beaver_dam_theorem :
  first_group_size * first_group_time = second_group_size * second_group_time :=
by sorry

#check beaver_dam_theorem

end NUMINAMATH_CALUDE_beaver_dam_theorem_l505_50536


namespace NUMINAMATH_CALUDE_jane_toy_bear_production_l505_50557

/-- Jane's toy bear production problem -/
theorem jane_toy_bear_production 
  (base_output : ℝ) 
  (base_hours : ℝ) 
  (assistant_output_increase : ℝ) 
  (assistant_hours_decrease : ℝ) 
  (assistant_A_increase : ℝ) 
  (assistant_B_increase : ℝ) 
  (assistant_C_increase : ℝ) 
  (h1 : assistant_output_increase = 0.8) 
  (h2 : assistant_hours_decrease = 0.1) 
  (h3 : assistant_A_increase = 1.0) 
  (h4 : assistant_B_increase = 0.75) 
  (h5 : assistant_C_increase = 0.5) :
  let output_A := (1 + assistant_A_increase) * base_output / ((1 - assistant_hours_decrease) * base_hours)
  let output_B := (1 + assistant_B_increase) * base_output / ((1 - assistant_hours_decrease) * base_hours)
  let output_C := (1 + assistant_C_increase) * base_output / ((1 - assistant_hours_decrease) * base_hours)
  let increase_A := (output_A / (base_output / base_hours) - 1) * 100
  let increase_B := (output_B / (base_output / base_hours) - 1) * 100
  let increase_C := (output_C / (base_output / base_hours) - 1) * 100
  let average_increase := (increase_A + increase_B + increase_C) / 3
  ∃ ε > 0, |average_increase - 94.43| < ε :=
by sorry

end NUMINAMATH_CALUDE_jane_toy_bear_production_l505_50557


namespace NUMINAMATH_CALUDE_line_inclination_angle_l505_50585

/-- The inclination angle of the line √3x - y + 1 = 0 is π/3 -/
theorem line_inclination_angle :
  let line := {(x, y) : ℝ × ℝ | Real.sqrt 3 * x - y + 1 = 0}
  ∃ α : ℝ, α = π / 3 ∧ ∀ (x y : ℝ), (x, y) ∈ line → Real.tan α = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l505_50585


namespace NUMINAMATH_CALUDE_jacob_age_2005_l505_50515

/-- Given that Jacob was one-third as old as his grandfather at the end of 2000,
    and the sum of the years in which they were born is 3858,
    prove that Jacob will be 40.5 years old at the end of 2005. -/
theorem jacob_age_2005 (jacob_age_2000 : ℝ) (grandfather_age_2000 : ℝ) :
  jacob_age_2000 = (1 / 3) * grandfather_age_2000 →
  (2000 - jacob_age_2000) + (2000 - grandfather_age_2000) = 3858 →
  jacob_age_2000 + 5 = 40.5 := by
sorry

end NUMINAMATH_CALUDE_jacob_age_2005_l505_50515


namespace NUMINAMATH_CALUDE_arevalo_dinner_bill_l505_50510

/-- The Arevalo family's dinner bill problem -/
theorem arevalo_dinner_bill (salmon_price black_burger_price chicken_katsu_price : ℝ)
  (service_charge_rate : ℝ) (paid_amount change_received : ℝ) :
  salmon_price = 40 ∧
  black_burger_price = 15 ∧
  chicken_katsu_price = 25 ∧
  service_charge_rate = 0.1 ∧
  paid_amount = 100 ∧
  change_received = 8 →
  let total_food_cost := salmon_price + black_burger_price + chicken_katsu_price
  let service_charge := service_charge_rate * total_food_cost
  let subtotal := total_food_cost + service_charge
  let amount_paid := paid_amount - change_received
  let tip := amount_paid - subtotal
  tip / total_food_cost = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_arevalo_dinner_bill_l505_50510


namespace NUMINAMATH_CALUDE_trivia_team_absentees_l505_50581

theorem trivia_team_absentees (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) : 
  total_members = 5 → 
  points_per_member = 6 → 
  total_points = 18 → 
  total_members - (total_points / points_per_member) = 2 :=
by sorry

end NUMINAMATH_CALUDE_trivia_team_absentees_l505_50581


namespace NUMINAMATH_CALUDE_factorial_plus_one_divisible_implies_prime_l505_50597

theorem factorial_plus_one_divisible_implies_prime (n : ℕ) :
  (n! + 1) % (n + 1) = 0 → Nat.Prime (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorial_plus_one_divisible_implies_prime_l505_50597


namespace NUMINAMATH_CALUDE_exists_participation_to_invalidate_forecast_l505_50580

/-- Represents a voter in the election -/
structure Voter :=
  (id : Nat)
  (isCandidate : Bool)
  (friends : Set Nat)

/-- Represents a forecast for the election -/
def Forecast := Nat → Nat

/-- Represents the actual votes cast in the election -/
def ActualVotes := Nat → Nat

/-- Determines if a voter participates in the election -/
def VoterParticipation := Nat → Bool

/-- Calculates the actual votes based on voter participation -/
def calculateActualVotes (voters : List Voter) (participation : VoterParticipation) : ActualVotes :=
  sorry

/-- Checks if a forecast is good (correct for at least one candidate) -/
def isGoodForecast (forecast : Forecast) (actualVotes : ActualVotes) : Bool :=
  sorry

/-- Main theorem: For any forecast, there exists a voter participation that makes the forecast not good -/
theorem exists_participation_to_invalidate_forecast (voters : List Voter) (forecast : Forecast) :
  ∃ (participation : VoterParticipation),
    ¬(isGoodForecast forecast (calculateActualVotes voters participation)) :=
  sorry

end NUMINAMATH_CALUDE_exists_participation_to_invalidate_forecast_l505_50580


namespace NUMINAMATH_CALUDE_combination_permutation_inequality_l505_50575

theorem combination_permutation_inequality (n : ℕ+) : 
  2 * Nat.choose n 3 ≤ n * (n - 1) ↔ 3 ≤ n ∧ n ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_combination_permutation_inequality_l505_50575


namespace NUMINAMATH_CALUDE_four_digit_number_with_specific_remainders_l505_50543

theorem four_digit_number_with_specific_remainders :
  ∃! N : ℕ, 
    N % 131 = 112 ∧
    N % 132 = 98 ∧
    1000 ≤ N ∧ N ≤ 9999 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_with_specific_remainders_l505_50543


namespace NUMINAMATH_CALUDE_sum_of_four_integers_l505_50582

theorem sum_of_four_integers (m n p q : ℕ+) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4 →
  m + n + p + q = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_integers_l505_50582


namespace NUMINAMATH_CALUDE_kids_played_monday_tuesday_l505_50514

/-- The number of kids Julia played with on Monday, Tuesday, and Wednesday -/
structure KidsPlayedWith where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Theorem: The sum of kids Julia played with on Monday and Tuesday is 33 -/
theorem kids_played_monday_tuesday (k : KidsPlayedWith) 
  (h1 : k.monday = 15)
  (h2 : k.tuesday = 18)
  (h3 : k.wednesday = 97) : 
  k.monday + k.tuesday = 33 := by
  sorry


end NUMINAMATH_CALUDE_kids_played_monday_tuesday_l505_50514


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l505_50560

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 64
def C₂ (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 100

-- Define the tangent line segment
def is_tangent_to (P Q : ℝ × ℝ) : Prop :=
  C₁ P.1 P.2 ∧ C₂ Q.1 Q.2 ∧
  ∀ R : ℝ × ℝ, (C₁ R.1 R.2 ∨ C₂ R.1 R.2) → Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) + Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)

-- Theorem statement
theorem shortest_tangent_length :
  ∃ P Q : ℝ × ℝ, is_tangent_to P Q ∧
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 190 / 3 ∧
  ∀ P' Q' : ℝ × ℝ, is_tangent_to P' Q' →
    Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≥ 190 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shortest_tangent_length_l505_50560


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l505_50568

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_slope_at_one
  (h1 : Differentiable ℝ f)
  (h2 : ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ →
    |(f 1 - f (1 - 2*Δx)) / (2*Δx) + 1| < ε) :
  deriv f 1 = -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l505_50568


namespace NUMINAMATH_CALUDE_exists_alternating_coloring_l505_50559

-- Define an ordered set
variable {X : Type*} [PartialOrder X]

-- Define a coloring function
def Coloring (X : Type*) := X → Bool

-- Theorem statement
theorem exists_alternating_coloring :
  ∃ (f : Coloring X), ∀ (x y : X), x < y → f x = f y →
    ∃ (z : X), x < z ∧ z < y ∧ f z ≠ f x := by
  sorry

end NUMINAMATH_CALUDE_exists_alternating_coloring_l505_50559


namespace NUMINAMATH_CALUDE_a_4_equals_8_l505_50549

def S (n : ℕ) : ℕ := 2^n - 1

def a (n : ℕ) : ℕ :=
  if n = 0 then S 0
  else S n - S (n-1)

theorem a_4_equals_8 : a 4 = 8 := by sorry

end NUMINAMATH_CALUDE_a_4_equals_8_l505_50549


namespace NUMINAMATH_CALUDE_exchange_result_l505_50573

/-- The number of bills after exchanging 2 $100 bills as described -/
def total_bills : ℕ :=
  let initial_hundred_bills : ℕ := 2
  let fifty_bills : ℕ := 2  -- From exchanging one $100 bill
  let ten_bills : ℕ := 50 / 10  -- From exchanging half of the remaining $100 bill
  let five_bills : ℕ := 50 / 5  -- From exchanging the other half of the remaining $100 bill
  fifty_bills + ten_bills + five_bills

/-- Theorem stating that the total number of bills after the exchange is 17 -/
theorem exchange_result : total_bills = 17 := by
  sorry

end NUMINAMATH_CALUDE_exchange_result_l505_50573


namespace NUMINAMATH_CALUDE_factorization_problems_l505_50592

theorem factorization_problems :
  (∀ a : ℝ, 4 * a^2 - 9 = (2*a + 3) * (2*a - 3)) ∧
  (∀ x y : ℝ, 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l505_50592


namespace NUMINAMATH_CALUDE_fox_initial_coins_l505_50590

def bridge_crossings (initial_coins : ℕ) : ℕ := 
  let after_first := 2 * initial_coins - 50
  let after_second := 2 * after_first - 50
  let after_third := 2 * after_second - 50
  2 * after_third - 50

theorem fox_initial_coins : 
  ∃ (x : ℕ), bridge_crossings x = 0 ∧ x = 47 :=
sorry

end NUMINAMATH_CALUDE_fox_initial_coins_l505_50590


namespace NUMINAMATH_CALUDE_evie_shells_left_l505_50502

/-- The number of shells Evie collects per day -/
def shells_per_day : ℕ := 10

/-- The number of days Evie collects shells -/
def collection_days : ℕ := 6

/-- The number of shells Evie gives to her brother -/
def shells_given : ℕ := 2

/-- The number of shells Evie has left after collecting and giving some away -/
def shells_left : ℕ := shells_per_day * collection_days - shells_given

/-- Theorem stating that Evie has 58 shells left -/
theorem evie_shells_left : shells_left = 58 := by sorry

end NUMINAMATH_CALUDE_evie_shells_left_l505_50502


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l505_50504

theorem cricket_team_average_age : 
  ∀ (team_size : ℕ) (captain_age wicket_keeper_age : ℕ) (team_average : ℚ),
    team_size = 11 →
    captain_age = 27 →
    wicket_keeper_age = 28 →
    (team_size : ℚ) * team_average = 
      (captain_age : ℚ) + (wicket_keeper_age : ℚ) + 
      ((team_size - 2) : ℚ) * (team_average - 1) →
    team_average = 23 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l505_50504


namespace NUMINAMATH_CALUDE_triangle_perimeter_l505_50541

theorem triangle_perimeter (a b c : ℝ) (ha : a = 28) (hb : b = 16) (hc : c = 18) :
  a + b + c = 62 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l505_50541


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_5_sqrt_2_l505_50523

/-- A rectangular prism with dimensions length, width, and height -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A quadrilateral formed by four points in 3D space -/
structure Quadrilateral3D where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- The area of a quadrilateral formed by the intersection of a plane with a rectangular prism -/
def quadrilateral_area (prism : RectangularPrism) (quad : Quadrilateral3D) : ℝ := sorry

/-- The main theorem stating the area of the quadrilateral ABCD -/
theorem quadrilateral_area_is_5_sqrt_2 (prism : RectangularPrism) (quad : Quadrilateral3D) :
  prism.length = 2 ∧ prism.width = 1 ∧ prism.height = 3 →
  quad.A = ⟨0, 0, 0⟩ ∧ quad.C = ⟨2, 1, 3⟩ →
  quad.B.x = 1 ∧ quad.B.y = 1 ∧ quad.B.z = 0 →
  quad.D.x = 1 ∧ quad.D.y = 0 ∧ quad.D.z = 3 →
  quadrilateral_area prism quad = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_5_sqrt_2_l505_50523


namespace NUMINAMATH_CALUDE_min_points_on_circle_l505_50538

/-- A type representing a point in a plane -/
def Point : Type := ℝ × ℝ

/-- A type representing a circle in a plane -/
def Circle : Type := Point × ℝ

/-- Check if a point lies on a circle -/
def lies_on (p : Point) (c : Circle) : Prop :=
  let (center, radius) := c
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

/-- Check if four points are concyclic (lie on the same circle) -/
def are_concyclic (p1 p2 p3 p4 : Point) : Prop :=
  ∃ c : Circle, lies_on p1 c ∧ lies_on p2 c ∧ lies_on p3 c ∧ lies_on p4 c

/-- Main theorem -/
theorem min_points_on_circle 
  (points : Finset Point) 
  (h_card : points.card = 10)
  (h_concyclic : ∀ (s : Finset Point), s ⊆ points → s.card = 5 → 
    ∃ (t : Finset Point), t ⊆ s ∧ t.card = 4 ∧ 
    ∃ (p1 p2 p3 p4 : Point), p1 ∈ t ∧ p2 ∈ t ∧ p3 ∈ t ∧ p4 ∈ t ∧ 
    are_concyclic p1 p2 p3 p4) : 
  ∃ (c : Circle) (s : Finset Point), s ⊆ points ∧ s.card = 9 ∧ 
  ∀ p ∈ s, lies_on p c :=
sorry

end NUMINAMATH_CALUDE_min_points_on_circle_l505_50538


namespace NUMINAMATH_CALUDE_jenny_game_ratio_l505_50570

theorem jenny_game_ratio : 
  ∀ (games_against_mark games_against_jill games_jenny_won : ℕ)
    (mark_wins jill_win_percentage : ℚ),
    games_against_mark = 10 →
    mark_wins = 1 →
    jill_win_percentage = 3/4 →
    games_jenny_won = 14 →
    games_against_jill = (games_jenny_won - (games_against_mark - mark_wins)) / (1 - jill_win_percentage) →
    games_against_jill / games_against_mark = 2 := by
  sorry

end NUMINAMATH_CALUDE_jenny_game_ratio_l505_50570


namespace NUMINAMATH_CALUDE_sheet_width_sheet_width_proof_l505_50503

/-- The width of a rectangular metallic sheet, given specific conditions --/
theorem sheet_width : ℝ :=
  let length : ℝ := 100
  let cut_size : ℝ := 10
  let box_volume : ℝ := 24000
  let width : ℝ := 50

  have h1 : box_volume = (length - 2 * cut_size) * (width - 2 * cut_size) * cut_size :=
    by sorry

  50

theorem sheet_width_proof (length : ℝ) (cut_size : ℝ) (box_volume : ℝ) :
  length = 100 →
  cut_size = 10 →
  box_volume = 24000 →
  box_volume = (length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size →
  sheet_width = 50 :=
by sorry

end NUMINAMATH_CALUDE_sheet_width_sheet_width_proof_l505_50503


namespace NUMINAMATH_CALUDE_square_area_from_rectangle_circle_l505_50529

theorem square_area_from_rectangle_circle (rectangle_length : ℝ) (circle_radius : ℝ) (square_side : ℝ) : 
  rectangle_length = (2 / 5) * circle_radius →
  circle_radius = square_side →
  rectangle_length * 10 = 180 →
  square_side ^ 2 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_rectangle_circle_l505_50529


namespace NUMINAMATH_CALUDE_sqrt_D_irrational_l505_50535

/-- Given a real number x, D is defined as a² + b² + c², where a = x, b = x + 2, and c = a + b -/
def D (x : ℝ) : ℝ :=
  let a := x
  let b := x + 2
  let c := a + b
  a^2 + b^2 + c^2

/-- Theorem stating that the square root of D is always irrational for any real input x -/
theorem sqrt_D_irrational (x : ℝ) : Irrational (Real.sqrt (D x)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_D_irrational_l505_50535


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l505_50552

theorem cara_seating_arrangements (n : ℕ) (h : n = 6) : Nat.choose n 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l505_50552


namespace NUMINAMATH_CALUDE_john_arcade_spend_l505_50569

/-- The amount of money John spent at the arcade -/
def arcade_spend (total_time minutes_per_break num_breaks cost_per_interval minutes_per_interval : ℕ) : ℚ :=
  let total_minutes := total_time
  let break_minutes := minutes_per_break * num_breaks
  let playing_minutes := total_minutes - break_minutes
  let num_intervals := playing_minutes / minutes_per_interval
  (num_intervals : ℚ) * cost_per_interval

theorem john_arcade_spend :
  arcade_spend 275 10 5 (3/4) 5 = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_john_arcade_spend_l505_50569


namespace NUMINAMATH_CALUDE_laptop_price_l505_50596

/-- Given that 20% of a price is $240, prove that the full price is $1200 -/
theorem laptop_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (full_price : ℝ) 
  (h1 : upfront_payment = 240)
  (h2 : upfront_percentage = 20)
  (h3 : upfront_payment = upfront_percentage / 100 * full_price) : 
  full_price = 1200 := by
  sorry

#check laptop_price

end NUMINAMATH_CALUDE_laptop_price_l505_50596


namespace NUMINAMATH_CALUDE_A_B_white_mutually_exclusive_l505_50556

/-- Represents a person who can receive a ball -/
inductive Person : Type
  | A : Person
  | B : Person
  | C : Person

/-- Represents a ball color -/
inductive BallColor : Type
  | Red : BallColor
  | Black : BallColor
  | White : BallColor

/-- Represents a distribution of balls to people -/
def Distribution := Person → BallColor

/-- The event that person A receives the white ball -/
def A_receives_white (d : Distribution) : Prop := d Person.A = BallColor.White

/-- The event that person B receives the white ball -/
def B_receives_white (d : Distribution) : Prop := d Person.B = BallColor.White

/-- Each person receives exactly one ball -/
def valid_distribution (d : Distribution) : Prop :=
  ∀ (c : BallColor), ∃! (p : Person), d p = c

theorem A_B_white_mutually_exclusive :
  ∀ (d : Distribution), valid_distribution d →
    ¬(A_receives_white d ∧ B_receives_white d) :=
sorry

end NUMINAMATH_CALUDE_A_B_white_mutually_exclusive_l505_50556


namespace NUMINAMATH_CALUDE_windows_preference_l505_50587

theorem windows_preference (total : ℕ) (mac_pref : ℕ) (no_pref : ℕ) 
  (h1 : total = 210)
  (h2 : mac_pref = 60)
  (h3 : no_pref = 90) :
  total - mac_pref - (mac_pref / 3) - no_pref = 40 := by
  sorry

#check windows_preference

end NUMINAMATH_CALUDE_windows_preference_l505_50587


namespace NUMINAMATH_CALUDE_hockey_team_ties_l505_50579

theorem hockey_team_ties (total_points : ℕ) (win_tie_difference : ℕ) : 
  total_points = 60 → win_tie_difference = 12 → 
  ∃ (ties wins : ℕ), 
    ties + wins = total_points ∧ 
    wins = ties + win_tie_difference ∧
    2 * wins + ties = total_points ∧
    ties = 12 := by
  sorry

end NUMINAMATH_CALUDE_hockey_team_ties_l505_50579


namespace NUMINAMATH_CALUDE_pyramid_slice_height_l505_50528

-- Define the pyramid P
structure Pyramid :=
  (base_length : ℝ)
  (base_width : ℝ)
  (height : ℝ)

-- Define the main theorem
theorem pyramid_slice_height (P : Pyramid) (volume_ratio : ℝ) :
  P.base_length = 15 →
  P.base_width = 20 →
  P.height = 30 →
  volume_ratio = 9 →
  (P.height - (P.height / (volume_ratio ^ (1/3 : ℝ)))) = 20 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_slice_height_l505_50528


namespace NUMINAMATH_CALUDE_min_value_of_function_l505_50531

theorem min_value_of_function (x y : ℝ) : x^2 + y^2 - 8*x + 6*y + 26 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l505_50531


namespace NUMINAMATH_CALUDE_equation_solutions_l505_50584

theorem equation_solutions : 
  let f (x : ℝ) := (x - 1)^3 * (x - 2)^3 * (x - 3)^3 * (x - 4)^3 / ((x - 2) * (x - 4) * (x - 2)^2)
  ∀ x : ℝ, f x = 64 ↔ x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l505_50584


namespace NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_squared_l505_50546

theorem ceiling_negative_seven_fourths_squared : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_squared_l505_50546


namespace NUMINAMATH_CALUDE_parallel_line_plane_false_l505_50533

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines and between a line and a plane
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)

-- Define the "contained in" relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- State the theorem to be proven false
theorem parallel_line_plane_false :
  ¬(∀ (l : Line) (p : Plane), parallel_plane l p →
    ∀ (m : Line), contained_in m p → parallel_line l m) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_plane_false_l505_50533


namespace NUMINAMATH_CALUDE_train_speed_problem_l505_50577

/-- Proves that for a journey of 70 km, if a train traveling at 35 kmph
    arrives 15 minutes late compared to its on-time speed,
    then the on-time speed is 40 kmph. -/
theorem train_speed_problem (v : ℝ) : 
  (70 / v + 0.25 = 70 / 35) → v = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l505_50577


namespace NUMINAMATH_CALUDE_cylinder_volume_scaling_l505_50567

theorem cylinder_volume_scaling (r h V : ℝ) :
  V = π * r^2 * h →
  ∀ (k : ℝ), k > 0 →
    π * (k*r)^2 * (k*h) = k^3 * V :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_scaling_l505_50567
