import Mathlib

namespace NUMINAMATH_CALUDE_triangle_perimeter_increase_l1075_107546

/-- Given three equilateral triangles where each subsequent triangle has sides 200% of the previous,
    prove that the percent increase in perimeter from the first to the third triangle is 300%. -/
theorem triangle_perimeter_increase (side_length : ℝ) (side_length_positive : side_length > 0) :
  let first_perimeter := 3 * side_length
  let third_perimeter := 3 * (4 * side_length)
  let percent_increase := (third_perimeter - first_perimeter) / first_perimeter * 100
  percent_increase = 300 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_increase_l1075_107546


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l1075_107548

/-- Scientific notation representation of 185000 -/
def scientific_notation : ℝ := 1.85 * (10 : ℝ) ^ 5

/-- The original number -/
def original_number : ℕ := 185000

theorem scientific_notation_proof : 
  (original_number : ℝ) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l1075_107548


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l1075_107564

/-- Given that -9, a, -1 form an arithmetic sequence and 
    -9, m, b, n, -1 form a geometric sequence, prove that ab = 15 -/
theorem arithmetic_geometric_sequence_product (a m b n : ℝ) : 
  ((-9 + (-1)) / 2 = a) →  -- arithmetic sequence condition
  ((-1 / -9) ^ (1/4) = (-1 / -9) ^ (1/4)) →  -- geometric sequence condition
  a * b = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l1075_107564


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1075_107566

theorem negation_of_proposition (a : ℝ) (h : 0 < a ∧ a < 1) :
  (¬ (∀ x : ℝ, x < 0 → a^x > 1)) ↔ (∃ x₀ : ℝ, x₀ < 0 ∧ a^x₀ ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1075_107566


namespace NUMINAMATH_CALUDE_alpha_range_l1075_107511

theorem alpha_range (α : Real) :
  (Complex.exp (Complex.I * α) + 2 * Complex.I * Complex.cos α = 2 * Complex.I) ↔
  ∃ k : ℤ, α = 2 * k * Real.pi := by
sorry

end NUMINAMATH_CALUDE_alpha_range_l1075_107511


namespace NUMINAMATH_CALUDE_base_equation_solution_l1075_107522

/-- Represents a digit in base b --/
def Digit (b : ℕ) := Fin b

/-- Converts a natural number to its representation in base b --/
def toBase (n : ℕ) (b : ℕ) : List (Digit b) :=
  sorry

/-- Adds two numbers in base b --/
def addBase (x y : List (Digit b)) : List (Digit b) :=
  sorry

/-- Checks if a list of digits is equal to another list of digits --/
def digitListEq (x y : List (Digit b)) : Prop :=
  sorry

theorem base_equation_solution :
  ∀ b : ℕ, b > 1 →
    (digitListEq (addBase (toBase 295 b) (toBase 467 b)) (toBase 762 b)) ↔ b = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_equation_solution_l1075_107522


namespace NUMINAMATH_CALUDE_no_three_integer_solutions_l1075_107584

theorem no_three_integer_solutions (b : ℤ) : 
  ¬(∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (x^2 + b*x + 5 ≤ 0) ∧ (y^2 + b*y + 5 ≤ 0) ∧ (z^2 + b*z + 5 ≤ 0) ∧
    (∀ (w : ℤ), w ≠ x ∧ w ≠ y ∧ w ≠ z → w^2 + b*w + 5 > 0)) :=
by sorry

#check no_three_integer_solutions

end NUMINAMATH_CALUDE_no_three_integer_solutions_l1075_107584


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1075_107565

-- Define the universal set U
def U : Set ℝ := {x : ℝ | -Real.sqrt 3 < x}

-- Define set A
def A : Set ℝ := {x : ℝ | 1 < 4 - x^2 ∧ 4 - x^2 ≤ 2}

-- State the theorem
theorem complement_of_A_in_U :
  (U \ A) = {x : ℝ | (-Real.sqrt 3 < x ∧ x < -Real.sqrt 2) ∨ (Real.sqrt 2 < x)} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1075_107565


namespace NUMINAMATH_CALUDE_number_ordering_l1075_107592

def A : ℕ := (Nat.factorial 8) ^ (Nat.factorial 8)
def B : ℕ := 8 ^ (8 ^ 8)
def C : ℕ := 8 ^ 88
def D : ℕ := (8 ^ 8) ^ 8

theorem number_ordering : D < C ∧ C < B ∧ B < A := by sorry

end NUMINAMATH_CALUDE_number_ordering_l1075_107592


namespace NUMINAMATH_CALUDE_x_squared_eq_one_necessary_not_sufficient_l1075_107504

theorem x_squared_eq_one_necessary_not_sufficient (x : ℝ) :
  (x = 1 → x^2 = 1) ∧ ¬(x^2 = 1 → x = 1) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_eq_one_necessary_not_sufficient_l1075_107504


namespace NUMINAMATH_CALUDE_horner_method_v3_l1075_107593

def horner_polynomial (x : ℝ) : ℝ := 2*x^6 + 5*x^4 + x^3 + 7*x^2 + 3*x + 1

def horner_v3 (x : ℝ) : ℝ :=
  let v0 := 2
  let v1 := v0 * x + 0
  let v2 := v1 * x + 5
  v2 * x + 1

theorem horner_method_v3 :
  horner_v3 3 = 70 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l1075_107593


namespace NUMINAMATH_CALUDE_constant_proof_l1075_107554

theorem constant_proof (n : ℤ) (c : ℝ) : 
  (∀ k : ℤ, c * k^2 ≤ 12100 → k ≤ 10) →
  (c * 10^2 ≤ 12100) →
  c = 121 := by
  sorry

end NUMINAMATH_CALUDE_constant_proof_l1075_107554


namespace NUMINAMATH_CALUDE_complex_product_positive_implies_zero_l1075_107507

theorem complex_product_positive_implies_zero (a : ℝ) :
  (Complex.I * (a - Complex.I)).re > 0 → a = 0 := by sorry

end NUMINAMATH_CALUDE_complex_product_positive_implies_zero_l1075_107507


namespace NUMINAMATH_CALUDE_sports_club_overlap_l1075_107506

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h_total : total = 80)
  (h_badminton : badminton = 48)
  (h_tennis : tennis = 46)
  (h_neither : neither = 7)
  : badminton + tennis - (total - neither) = 21 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l1075_107506


namespace NUMINAMATH_CALUDE_corrected_mean_calculation_l1075_107559

/-- Given a set of observations with incorrect recordings, calculate the corrected mean. -/
theorem corrected_mean_calculation 
  (n : ℕ) 
  (original_mean : ℚ) 
  (incorrect_value1 incorrect_value2 correct_value1 correct_value2 : ℚ) :
  n = 50 →
  original_mean = 36 →
  incorrect_value1 = 23 →
  incorrect_value2 = 55 →
  correct_value1 = 34 →
  correct_value2 = 45 →
  let original_sum := n * original_mean
  let adjusted_sum := original_sum - incorrect_value1 - incorrect_value2 + correct_value1 + correct_value2
  let new_mean := adjusted_sum / n
  new_mean = 36.02 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_calculation_l1075_107559


namespace NUMINAMATH_CALUDE_square_equality_necessary_not_sufficient_l1075_107514

theorem square_equality_necessary_not_sufficient :
  (∀ x y : ℝ, x = y → x^2 = y^2) ∧
  (∃ x y : ℝ, x^2 = y^2 ∧ x ≠ y) := by
  sorry

end NUMINAMATH_CALUDE_square_equality_necessary_not_sufficient_l1075_107514


namespace NUMINAMATH_CALUDE_binary_1001101_equals_octal_115_l1075_107512

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enumFrom 0 b).foldl (λ acc (i, x) => acc + if x then 2^i else 0) 0

def octal_to_decimal (o : List ℕ) : ℕ :=
  (List.enumFrom 0 o).foldl (λ acc (i, x) => acc + x * 8^i) 0

theorem binary_1001101_equals_octal_115 :
  binary_to_decimal [true, false, true, true, false, false, true] =
  octal_to_decimal [5, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_1001101_equals_octal_115_l1075_107512


namespace NUMINAMATH_CALUDE_not_perfect_square_l1075_107582

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℤ, (3 : ℤ)^n + 2 * (17 : ℤ)^n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1075_107582


namespace NUMINAMATH_CALUDE_divides_trans_divides_mul_l1075_107553

/-- Divisibility relation for positive integers -/
def divides (a b : ℕ+) : Prop := ∃ k : ℕ+, b = a * k

/-- Transitivity of divisibility -/
theorem divides_trans {a b c : ℕ+} (h1 : divides a b) (h2 : divides b c) : 
  divides a c := by sorry

/-- Product of divisibilities -/
theorem divides_mul {a b c d : ℕ+} (h1 : divides a b) (h2 : divides c d) :
  divides (a * c) (b * d) := by sorry

end NUMINAMATH_CALUDE_divides_trans_divides_mul_l1075_107553


namespace NUMINAMATH_CALUDE_inequality_proof_l1075_107510

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1075_107510


namespace NUMINAMATH_CALUDE_partition_contains_perfect_square_sum_l1075_107545

/-- A partition of a set of natural numbers -/
def Partition (n : ℕ) := Fin n → Bool

/-- Checks if a pair of numbers sum to a perfect square -/
def IsPerfectSquareSum (a b : ℕ) : Prop :=
  ∃ k : ℕ, a + b = k * k

/-- The main theorem statement -/
theorem partition_contains_perfect_square_sum (n : ℕ) (h : n ≥ 15) :
  ∀ (p : Partition n), ∃ (i j : Fin n), i ≠ j ∧ p i = p j ∧ IsPerfectSquareSum (i.val + 1) (j.val + 1) :=
sorry

end NUMINAMATH_CALUDE_partition_contains_perfect_square_sum_l1075_107545


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_is_four_min_value_achievable_l1075_107500

theorem min_value_theorem (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 2 → 1 / m + 2 / n ≤ 1 / x + 2 / y :=
by sorry

theorem min_value_is_four (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) :
  1 / m + 2 / n ≥ 4 :=
by sorry

theorem min_value_achievable (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 2 ∧ 1 / x + 2 / y = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_is_four_min_value_achievable_l1075_107500


namespace NUMINAMATH_CALUDE_towel_price_problem_l1075_107555

theorem towel_price_problem (x : ℚ) : 
  (3 * 100 + 5 * 150 + 2 * x) / 10 = 150 → x = 225 := by
  sorry

end NUMINAMATH_CALUDE_towel_price_problem_l1075_107555


namespace NUMINAMATH_CALUDE_points_four_units_from_negative_two_l1075_107577

theorem points_four_units_from_negative_two (x : ℝ) : 
  (|x - (-2)| = 4) ↔ (x = 2 ∨ x = -6) := by sorry

end NUMINAMATH_CALUDE_points_four_units_from_negative_two_l1075_107577


namespace NUMINAMATH_CALUDE_factorization_2a_squared_minus_2a_l1075_107501

theorem factorization_2a_squared_minus_2a (a : ℝ) : 2*a^2 - 2*a = 2*a*(a-1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2a_squared_minus_2a_l1075_107501


namespace NUMINAMATH_CALUDE_both_make_basket_l1075_107543

-- Define the probabilities
def prob_A : ℚ := 2/5
def prob_B : ℚ := 1/2

-- Define the theorem
theorem both_make_basket : 
  prob_A * prob_B = 1/5 := by sorry

end NUMINAMATH_CALUDE_both_make_basket_l1075_107543


namespace NUMINAMATH_CALUDE_inequality_solution_l1075_107598

theorem inequality_solution (x : ℝ) :
  (2 ≤ |3*x - 6| ∧ |3*x - 6| ≤ 12) ↔ (x ∈ Set.Icc (-2) (4/3) ∪ Set.Icc (8/3) 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1075_107598


namespace NUMINAMATH_CALUDE_binary_linear_equation_l1075_107578

theorem binary_linear_equation (x y : ℝ) : x + y = 5 → x = 3 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_linear_equation_l1075_107578


namespace NUMINAMATH_CALUDE_quadratic_factor_sum_l1075_107526

theorem quadratic_factor_sum (a w c d : ℤ) : 
  (∀ x : ℚ, 6 * x^2 + x - 12 = (a * x + w) * (c * x + d)) →
  |a| + |w| + |c| + |d| = 22 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factor_sum_l1075_107526


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_divisible_by_eight_l1075_107580

theorem consecutive_even_numbers_divisible_by_eight (n : ℤ) : 
  ∃ k : ℤ, 4 * n * (n + 1) = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_divisible_by_eight_l1075_107580


namespace NUMINAMATH_CALUDE_correct_matching_probability_l1075_107557

-- Define the number of celebrities and child photos
def num_celebrities : ℕ := 4

-- Define the function to calculate the number of possible arrangements
def num_arrangements (n : ℕ) : ℕ := Nat.factorial n

-- Define the probability of correct matching
def probability_correct_matching : ℚ := 1 / num_arrangements num_celebrities

-- Theorem statement
theorem correct_matching_probability :
  probability_correct_matching = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_correct_matching_probability_l1075_107557


namespace NUMINAMATH_CALUDE_num_sandwiches_al_can_order_l1075_107505

/-- Represents the number of different types of bread offered at the deli. -/
def num_breads : Nat := 5

/-- Represents the number of different types of meat offered at the deli. -/
def num_meats : Nat := 7

/-- Represents the number of different types of cheese offered at the deli. -/
def num_cheeses : Nat := 6

/-- Represents the number of restricted sandwich combinations. -/
def num_restricted : Nat := 16

/-- Theorem stating the number of different sandwiches Al could order. -/
theorem num_sandwiches_al_can_order :
  (num_breads * num_meats * num_cheeses) - num_restricted = 194 := by
  sorry

end NUMINAMATH_CALUDE_num_sandwiches_al_can_order_l1075_107505


namespace NUMINAMATH_CALUDE_cubic_root_l1075_107594

/-- Given a cubic expression ax³ - 2x + c, prove that if it equals -5 when x = 1
and equals 52 when x = 4, then it equals 0 when x = 2. -/
theorem cubic_root (a c : ℝ) 
  (h1 : a * 1^3 - 2 * 1 + c = -5)
  (h2 : a * 4^3 - 2 * 4 + c = 52) :
  a * 2^3 - 2 * 2 + c = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_l1075_107594


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l1075_107530

theorem prime_sum_theorem (x y : ℕ) : 
  Nat.Prime x → Nat.Prime y → x < y → x + y = 36 → 4 * x + y = 51 := by sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l1075_107530


namespace NUMINAMATH_CALUDE_bell_rings_count_l1075_107532

/-- Represents a school event that causes the bell to ring at its start and end -/
structure SchoolEvent where
  name : String

/-- Represents the school schedule for a day -/
structure SchoolSchedule where
  events : List SchoolEvent

/-- Counts the number of bell rings for a given schedule up to and including a specific event -/
def countBellRings (schedule : SchoolSchedule) (currentEvent : SchoolEvent) : Nat :=
  sorry

/-- Monday's altered schedule -/
def mondaySchedule : SchoolSchedule :=
  { events := [
    { name := "Assembly" },
    { name := "Maths" },
    { name := "History" },
    { name := "Surprise Quiz" },
    { name := "Geography" },
    { name := "Science" },
    { name := "Music" }
  ] }

/-- The current event (Geography class) -/
def currentEvent : SchoolEvent :=
  { name := "Geography" }

theorem bell_rings_count :
  countBellRings mondaySchedule currentEvent = 9 := by
  sorry

end NUMINAMATH_CALUDE_bell_rings_count_l1075_107532


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1075_107591

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  a + b = -14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1075_107591


namespace NUMINAMATH_CALUDE_simeon_water_consumption_l1075_107563

/-- Simeon's daily water consumption in fluid ounces -/
def daily_water : ℕ := 64

/-- Size of old serving in fluid ounces -/
def old_serving : ℕ := 8

/-- Size of new serving in fluid ounces -/
def new_serving : ℕ := 16

/-- Difference in number of servings -/
def serving_difference : ℕ := 4

theorem simeon_water_consumption :
  ∃ (old_servings new_servings : ℕ),
    old_servings * old_serving = daily_water ∧
    new_servings * new_serving = daily_water ∧
    old_servings = new_servings + serving_difference :=
by sorry

end NUMINAMATH_CALUDE_simeon_water_consumption_l1075_107563


namespace NUMINAMATH_CALUDE_three_isosceles_triangles_l1075_107547

-- Define a point in 2D space
structure Point :=
  (x : Int) (y : Int)

-- Define a triangle by its three vertices
structure Triangle :=
  (v1 : Point) (v2 : Point) (v3 : Point)

-- Function to calculate the squared distance between two points
def squaredDistance (p1 p2 : Point) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d1 := squaredDistance t.v1 t.v2
  let d2 := squaredDistance t.v2 t.v3
  let d3 := squaredDistance t.v3 t.v1
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

-- Define the five triangles
def triangle1 : Triangle := ⟨⟨1, 5⟩, ⟨3, 5⟩, ⟨2, 2⟩⟩
def triangle2 : Triangle := ⟨⟨4, 3⟩, ⟨4, 6⟩, ⟨7, 3⟩⟩
def triangle3 : Triangle := ⟨⟨1, 1⟩, ⟨5, 2⟩, ⟨9, 1⟩⟩
def triangle4 : Triangle := ⟨⟨7, 5⟩, ⟨6, 6⟩, ⟨9, 3⟩⟩
def triangle5 : Triangle := ⟨⟨8, 2⟩, ⟨10, 5⟩, ⟨10, 0⟩⟩

-- Theorem: Exactly 3 out of the 5 given triangles are isosceles
theorem three_isosceles_triangles :
  (isIsosceles triangle1 ∧ isIsosceles triangle2 ∧ isIsosceles triangle3 ∧
   ¬isIsosceles triangle4 ∧ ¬isIsosceles triangle5) :=
by sorry

end NUMINAMATH_CALUDE_three_isosceles_triangles_l1075_107547


namespace NUMINAMATH_CALUDE_standard_deviation_shift_l1075_107515

-- Define the standard deviation function for a list of real numbers
def standardDeviation (xs : List ℝ) : ℝ := sorry

-- Define a function to add a constant to each element of a list
def addConstant (xs : List ℝ) (c : ℝ) : List ℝ := sorry

-- Theorem statement
theorem standard_deviation_shift (a b c : ℝ) :
  standardDeviation [a + 2, b + 2, c + 2] = 2 →
  standardDeviation [a, b, c] = 2 := by sorry

end NUMINAMATH_CALUDE_standard_deviation_shift_l1075_107515


namespace NUMINAMATH_CALUDE_enrollment_theorem_l1075_107519

-- Define the schools and their enrollments
def schools : Fin 4 → ℕ
| 0 => 1300  -- Varsity
| 1 => 1500  -- Northwest
| 2 => 1800  -- Central
| 3 => 1600  -- Greenbriar
| _ => 0     -- This case should never occur

-- Calculate the average enrollment
def average_enrollment : ℚ := (schools 0 + schools 1 + schools 2 + schools 3) / 4

-- Calculate the positive difference between a school's enrollment and the average
def positive_difference (i : Fin 4) : ℚ := |schools i - average_enrollment|

-- Theorem stating the average enrollment and positive differences
theorem enrollment_theorem :
  average_enrollment = 1550 ∧
  positive_difference 0 = 250 ∧
  positive_difference 1 = 50 ∧
  positive_difference 2 = 250 ∧
  positive_difference 3 = 50 :=
by sorry

end NUMINAMATH_CALUDE_enrollment_theorem_l1075_107519


namespace NUMINAMATH_CALUDE_socks_bought_l1075_107531

/-- Given John's sock inventory changes, prove the number of new socks bought. -/
theorem socks_bought (initial : ℕ) (thrown_away : ℕ) (final : ℕ) 
  (h1 : initial = 33)
  (h2 : thrown_away = 19)
  (h3 : final = 27) :
  final - (initial - thrown_away) = 13 := by
  sorry

end NUMINAMATH_CALUDE_socks_bought_l1075_107531


namespace NUMINAMATH_CALUDE_min_value_sqrt_expression_l1075_107597

theorem min_value_sqrt_expression (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 16) :
  Real.sqrt (x + 36) + Real.sqrt (16 - x) + 2 * Real.sqrt x ≥ 13.46 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_expression_l1075_107597


namespace NUMINAMATH_CALUDE_find_a_l1075_107509

theorem find_a : ∃ (a : ℝ), (∀ (x : ℝ), (2 * x - a ≤ -1) ↔ (x ≤ 1)) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_find_a_l1075_107509


namespace NUMINAMATH_CALUDE_well_climbing_l1075_107552

/-- Proves that a man climbing out of a well slips down 3 meters each day -/
theorem well_climbing (well_depth : ℝ) (days : ℕ) (climb_up : ℝ) (slip_down : ℝ) 
  (h1 : well_depth = 30)
  (h2 : days = 27)
  (h3 : climb_up = 4)
  (h4 : (days - 1) * (climb_up - slip_down) + climb_up = well_depth) :
  slip_down = 3 := by
  sorry


end NUMINAMATH_CALUDE_well_climbing_l1075_107552


namespace NUMINAMATH_CALUDE_probability_sum_le_four_l1075_107585

-- Define the type for a die
def Die := Fin 6

-- Define the sum of two dice
def diceSum (d1 d2 : Die) : Nat := d1.val + d2.val + 2

-- Define the condition for the sum being less than or equal to 4
def sumLEFour (d1 d2 : Die) : Prop := diceSum d1 d2 ≤ 4

-- Define the probability space
def totalOutcomes : Nat := 36

-- Define the number of favorable outcomes
def favorableOutcomes : Nat := 6

-- Theorem statement
theorem probability_sum_le_four :
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_sum_le_four_l1075_107585


namespace NUMINAMATH_CALUDE_mrs_hilt_reading_rate_l1075_107535

/-- Mrs. Hilt's daily reading rate -/
def daily_reading_rate (total_books : ℕ) (total_days : ℕ) : ℚ :=
  total_books / total_days

/-- Theorem: Mrs. Hilt reads 5 books per day -/
theorem mrs_hilt_reading_rate :
  daily_reading_rate 15 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_reading_rate_l1075_107535


namespace NUMINAMATH_CALUDE_triathlon_speed_l1075_107534

/-- Triathlon completion problem -/
theorem triathlon_speed (swim_distance : Real) (bike_distance : Real) (run_distance : Real)
  (total_time : Real) (swim_speed : Real) (run_speed : Real) :
  swim_distance = 0.5 ∧ 
  bike_distance = 20 ∧ 
  run_distance = 4 ∧ 
  total_time = 1.75 ∧ 
  swim_speed = 1 ∧ 
  run_speed = 4 →
  (bike_distance / (total_time - (swim_distance / swim_speed) - (run_distance / run_speed))) = 80 := by
  sorry

#check triathlon_speed

end NUMINAMATH_CALUDE_triathlon_speed_l1075_107534


namespace NUMINAMATH_CALUDE_words_with_consonants_l1075_107590

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 6

/-- The number of vowels in the alphabet --/
def vowel_count : ℕ := 2

/-- The length of words we're considering --/
def word_length : ℕ := 5

/-- The total number of possible words --/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of words containing only vowels --/
def vowel_only_words : ℕ := vowel_count ^ word_length

theorem words_with_consonants :
  total_words - vowel_only_words = 7744 := by sorry

end NUMINAMATH_CALUDE_words_with_consonants_l1075_107590


namespace NUMINAMATH_CALUDE_calculation_proof_equation_no_solution_l1075_107539

-- Part 1
theorem calculation_proof : (Real.sqrt 12 - 3 * Real.sqrt (1/3)) / Real.sqrt 3 = 1 := by
  sorry

-- Part 2
theorem equation_no_solution :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → (x - 1) / (x + 1) + 4 / (x^2 - 1) ≠ (x + 1) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_equation_no_solution_l1075_107539


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1075_107540

theorem system_of_equations_solution (x y m : ℝ) : 
  2 * x + y = 4 → 
  x + 2 * y = m → 
  x + y = 1 → 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1075_107540


namespace NUMINAMATH_CALUDE_tommy_nickels_count_l1075_107567

/-- Proves that Tommy has 100 nickels given the relationships between his coins -/
theorem tommy_nickels_count : 
  ∀ (quarters pennies dimes nickels : ℕ),
    quarters = 4 →
    pennies = 10 * quarters →
    dimes = pennies + 10 →
    nickels = 2 * dimes →
    nickels = 100 := by sorry

end NUMINAMATH_CALUDE_tommy_nickels_count_l1075_107567


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l1075_107596

theorem factor_difference_of_squares (x : ℝ) : x^2 - 81 = (x - 9) * (x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l1075_107596


namespace NUMINAMATH_CALUDE_gcd_seven_eight_factorial_l1075_107544

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem gcd_seven_eight_factorial :
  Nat.gcd (factorial 7) (factorial 8) = factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_seven_eight_factorial_l1075_107544


namespace NUMINAMATH_CALUDE_min_length_shared_side_l1075_107569

/-- Given two triangles PQR and SQR that share side QR, with PQ = 7, PR = 15, SR = 10, and QS = 25,
    prove that the length of QR is at least 15. -/
theorem min_length_shared_side (PQ PR SR QS : ℝ) (hPQ : PQ = 7) (hPR : PR = 15) (hSR : SR = 10) (hQS : QS = 25) :
  ∃ (QR : ℝ), QR ≥ 15 ∧ QR > PR - PQ ∧ QR > QS - SR :=
by sorry

end NUMINAMATH_CALUDE_min_length_shared_side_l1075_107569


namespace NUMINAMATH_CALUDE_equation_solution_l1075_107589

theorem equation_solution : ∀ x : ℝ, (3 / (x - 3) = 3 / (x^2 - 9)) ↔ x = -2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1075_107589


namespace NUMINAMATH_CALUDE_work_completion_time_l1075_107556

/-- Given workers A, B, and C who can complete a work individually in 4, 8, and 8 days respectively,
    prove that they can complete the work together in 2 days. -/
theorem work_completion_time (work : ℝ) (days_A days_B days_C : ℝ) 
    (h_work : work > 0)
    (h_A : days_A = 4)
    (h_B : days_B = 8)
    (h_C : days_C = 8) :
    work / (work / days_A + work / days_B + work / days_C) = 2 := by
  sorry


end NUMINAMATH_CALUDE_work_completion_time_l1075_107556


namespace NUMINAMATH_CALUDE_square_plus_one_positive_l1075_107583

theorem square_plus_one_positive (a : ℝ) : a^2 + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_positive_l1075_107583


namespace NUMINAMATH_CALUDE_max_product_l1075_107576

def digits : List Nat := [3, 5, 7, 8, 9]

def is_valid_pair (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def three_digit (a b c : Nat) : Nat := 100 * a + 10 * b + c

def two_digit (d e : Nat) : Nat := 10 * d + e

def product (a b c d e : Nat) : Nat := (three_digit a b c) * (two_digit d e)

theorem max_product :
  ∀ a b c d e,
    is_valid_pair a b c d e →
    product a b c d e ≤ product 9 7 5 8 3 :=
by sorry

end NUMINAMATH_CALUDE_max_product_l1075_107576


namespace NUMINAMATH_CALUDE_females_with_advanced_degrees_l1075_107550

theorem females_with_advanced_degrees 
  (total_employees : ℕ)
  (total_females : ℕ)
  (employees_with_advanced_degrees : ℕ)
  (males_with_college_only : ℕ)
  (h1 : total_employees = 148)
  (h2 : total_females = 92)
  (h3 : employees_with_advanced_degrees = 78)
  (h4 : males_with_college_only = 31)
  : total_females - (total_employees - employees_with_advanced_degrees - males_with_college_only) = 53 := by
  sorry

end NUMINAMATH_CALUDE_females_with_advanced_degrees_l1075_107550


namespace NUMINAMATH_CALUDE_solution_set_f_leq_6_range_a_for_nonempty_solution_l1075_107513

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

-- Theorem for part I
theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
by sorry

-- Theorem for part II
theorem range_a_for_nonempty_solution :
  {a : ℝ | ∃ x, f x < |a - 1|} = {a : ℝ | a < -3 ∨ a > 5} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_6_range_a_for_nonempty_solution_l1075_107513


namespace NUMINAMATH_CALUDE_trajectory_and_max_distance_l1075_107533

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the line l: x = 2
def l (x : ℝ) : Prop := x = 2

-- Define the distance ratio condition
def distance_ratio (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  (((x - F.1)^2 + (y - F.2)^2).sqrt / |2 - x|) = Real.sqrt 2 / 2

-- Define the ellipse equation
def ellipse (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  x^2 / 2 + y^2 = 1

-- Define the line for maximum distance calculation
def max_distance_line (x y : ℝ) : Prop :=
  x / Real.sqrt 2 + y = 1

-- Theorem statement
theorem trajectory_and_max_distance :
  -- Part 1: Trajectory is an ellipse
  (∀ M : ℝ × ℝ, distance_ratio M ↔ ellipse M) ∧
  -- Part 2: Maximum distance exists
  (∃ d : ℝ, ∀ M : ℝ × ℝ, ellipse M →
    let (x, y) := M
    abs (x / Real.sqrt 2 + y - 1) / Real.sqrt ((1 / 2) + 1) ≤ d) ∧
  -- Part 3: Maximum distance value
  (let max_d := (2 * Real.sqrt 3 + Real.sqrt 6) / 3
   ∃ M : ℝ × ℝ, ellipse M ∧
     let (x, y) := M
     abs (x / Real.sqrt 2 + y - 1) / Real.sqrt ((1 / 2) + 1) = max_d) :=
by sorry


end NUMINAMATH_CALUDE_trajectory_and_max_distance_l1075_107533


namespace NUMINAMATH_CALUDE_vectors_parallel_when_m_neg_one_l1075_107521

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Vector a parameterized by m -/
def a (m : ℝ) : ℝ × ℝ := (2*m - 1, m)

/-- Vector b -/
def b : ℝ × ℝ := (3, 1)

/-- Theorem stating that vectors a and b are parallel when m = -1 -/
theorem vectors_parallel_when_m_neg_one :
  are_parallel (a (-1)) b := by sorry

end NUMINAMATH_CALUDE_vectors_parallel_when_m_neg_one_l1075_107521


namespace NUMINAMATH_CALUDE_randy_tower_blocks_l1075_107575

/-- Given information about Randy's blocks and constructions -/
structure RandysBlocks where
  total : ℕ
  house : ℕ
  tower_and_house : ℕ

/-- The number of blocks Randy used for the tower -/
def blocks_for_tower (r : RandysBlocks) : ℕ :=
  r.tower_and_house - r.house

/-- Theorem stating that Randy used 27 blocks for the tower -/
theorem randy_tower_blocks (r : RandysBlocks)
  (h1 : r.total = 58)
  (h2 : r.house = 53)
  (h3 : r.tower_and_house = 80) :
  blocks_for_tower r = 27 := by
  sorry

end NUMINAMATH_CALUDE_randy_tower_blocks_l1075_107575


namespace NUMINAMATH_CALUDE_ellipse_circle_tangent_l1075_107523

/-- Given an ellipse M and a circle N with specific properties, prove their equations and the equation of their common tangent line. -/
theorem ellipse_circle_tangent (a b c : ℝ) (k m : ℝ) :
  a > 0 ∧ b > 0 ∧ a > b ∧  -- conditions on a, b
  c / a = 1 / 2 ∧  -- eccentricity
  a^2 / c - c = 3 ∧  -- distance condition
  c > 0 →  -- c is positive (implied by being a distance)
  -- Prove:
  ((∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 / 3 = 1) ∧  -- equation of M
   (∀ x y : ℝ, (x - c)^2 + y^2 = a^2 + c^2 ↔ (x - 1)^2 + y^2 = 5) ∧  -- equation of N
   ((k = 1/2 ∧ m = 2) ∨ (k = -1/2 ∧ m = -2)) ∧  -- equation of tangent line l
   (∀ x : ℝ, (x^2 / 4 + (k * x + m)^2 / 3 = 1 →  -- l is tangent to M
              ∃! y : ℝ, y = k * x + m ∧ x^2 / 4 + y^2 / 3 = 1) ∧
    ((k * 1 + m)^2 + 1^2 = 5)))  -- l is tangent to N
  := by sorry

end NUMINAMATH_CALUDE_ellipse_circle_tangent_l1075_107523


namespace NUMINAMATH_CALUDE_feb_first_is_monday_l1075_107538

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a date in February of a leap year -/
structure FebruaryDate :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- Given that February 29th is a Monday in a leap year, 
    prove that February 1st is also a Monday -/
theorem feb_first_is_monday 
  (feb29 : FebruaryDate)
  (h1 : feb29.day = 29)
  (h2 : feb29.dayOfWeek = DayOfWeek.Monday) :
  ∃ (feb1 : FebruaryDate), 
    feb1.day = 1 ∧ feb1.dayOfWeek = DayOfWeek.Monday :=
by sorry

end NUMINAMATH_CALUDE_feb_first_is_monday_l1075_107538


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1075_107561

theorem arithmetic_computation : 5 + 4 * (4 - 9)^2 = 105 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1075_107561


namespace NUMINAMATH_CALUDE_sum_of_tenth_set_l1075_107581

/-- Calculates the sum of the first n triangular numbers -/
def sumOfTriangularNumbers (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- Calculates the first element of the nth set -/
def firstElementOfSet (n : ℕ) : ℕ := sumOfTriangularNumbers (n - 1) + 1

/-- Calculates the number of elements in the nth set -/
def numberOfElementsInSet (n : ℕ) : ℕ := n + 2 * (n - 1)

/-- Calculates the last element of the nth set -/
def lastElementOfSet (n : ℕ) : ℕ := firstElementOfSet n + numberOfElementsInSet n - 1

/-- Calculates the sum of elements in the nth set -/
def sumOfSet (n : ℕ) : ℕ := 
  (numberOfElementsInSet n * (firstElementOfSet n + lastElementOfSet n)) / 2

theorem sum_of_tenth_set : sumOfSet 10 = 5026 := by sorry

end NUMINAMATH_CALUDE_sum_of_tenth_set_l1075_107581


namespace NUMINAMATH_CALUDE_min_value_a_k_l1075_107599

/-- A positive arithmetic sequence satisfying the given condition -/
def PositiveArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧
  (∀ k : ℕ, k ≥ 2 → 1 / a 1 + 4 / a (2 * k - 1) ≤ 1)

/-- The theorem stating the minimum value of a_k -/
theorem min_value_a_k (a : ℕ → ℝ) (h : PositiveArithmeticSequence a) :
    ∀ k : ℕ, k ≥ 2 → a k ≥ 9/2 :=
  sorry

end NUMINAMATH_CALUDE_min_value_a_k_l1075_107599


namespace NUMINAMATH_CALUDE_intersection_M_N_l1075_107542

-- Define set M
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define set N
def N : Set ℝ := {x | 2*x - x^2 ≥ 0}

-- Statement to prove
theorem intersection_M_N : M ∩ N = Set.Ioo 1 2 ∪ {2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1075_107542


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l1075_107595

/-- A parabola with equation y^2 = 6x -/
structure Parabola where
  equation : ∀ x y, y^2 = 6*x

/-- A point on the parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 6*x

/-- Two lines intersecting the parabola -/
structure IntersectingLines (C : Parabola) (P : PointOnParabola C) where
  A : PointOnParabola C
  B : PointOnParabola C
  slope_AB : (B.y - A.y) / (B.x - A.x) = 2
  sum_reciprocal_slopes : 
    ((P.y - A.y) / (P.x - A.x))⁻¹ + ((P.y - B.y) / (P.x - B.x))⁻¹ = 3

/-- The theorem to be proved -/
theorem parabola_intersection_theorem 
  (C : Parabola) 
  (P : PointOnParabola C) 
  (L : IntersectingLines C P) : 
  P.y = 15/2 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l1075_107595


namespace NUMINAMATH_CALUDE_sum_of_three_circles_l1075_107572

theorem sum_of_three_circles (square circle : ℝ) 
  (eq1 : 3 * square + 2 * circle = 27)
  (eq2 : 2 * square + 3 * circle = 25) : 
  3 * circle = 12.6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_circles_l1075_107572


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l1075_107587

/-- Given a line symmetric to 4x - 3y + 5 = 0 with respect to the y-axis, prove its equation is 4x + 3y - 5 = 0 -/
theorem symmetric_line_equation : 
  ∃ (l : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ (-x, y) ∈ {(x, y) | 4*x - 3*y + 5 = 0}) → 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ 4*x + 3*y - 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l1075_107587


namespace NUMINAMATH_CALUDE_mixed_groups_count_l1075_107502

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ)
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  (∃ (mixed_groups : ℕ),
    mixed_groups = 72 ∧
    mixed_groups * 2 = total_groups * group_size - boy_boy_photos - girl_girl_photos) :=
by sorry

end NUMINAMATH_CALUDE_mixed_groups_count_l1075_107502


namespace NUMINAMATH_CALUDE_renovation_project_material_l1075_107568

theorem renovation_project_material (sand dirt cement : ℝ) 
  (h_sand : sand = 0.17)
  (h_dirt : dirt = 0.33)
  (h_cement : cement = 0.17) :
  sand + dirt + cement = 0.67 := by
  sorry

end NUMINAMATH_CALUDE_renovation_project_material_l1075_107568


namespace NUMINAMATH_CALUDE_teresa_jogging_distance_l1075_107527

def speed : ℝ := 5
def time : ℝ := 5
def distance : ℝ := speed * time

theorem teresa_jogging_distance : distance = 25 := by
  sorry

end NUMINAMATH_CALUDE_teresa_jogging_distance_l1075_107527


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1075_107570

theorem inequality_system_solution (x : ℝ) :
  (2 * (x - 1) ≤ x + 1 ∧ (x + 2) / 2 ≥ (x + 3) / 3) ↔ (0 ≤ x ∧ x ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1075_107570


namespace NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l1075_107562

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

/-- The fixed point K -/
def K : ℝ × ℝ := (2, 0)

/-- Dot product of two 2D vectors -/
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  (v₁.1 * v₂.1) + (v₁.2 * v₂.2)

/-- Vector from K to a point -/
def vector_from_K (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - K.1, p.2 - K.2)

theorem min_dot_product_on_ellipse :
  ∀ (M N : ℝ × ℝ),
  is_on_ellipse M.1 M.2 →
  is_on_ellipse N.1 N.2 →
  dot_product (vector_from_K M) (vector_from_K N) = 0 →
  ∃ (min_value : ℝ),
    min_value = 23/3 ∧
    ∀ (P Q : ℝ × ℝ),
    is_on_ellipse P.1 P.2 →
    is_on_ellipse Q.1 Q.2 →
    dot_product (vector_from_K P) (vector_from_K Q) = 0 →
    dot_product (vector_from_K P) (vector_from_K Q - vector_from_K P) ≥ min_value :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l1075_107562


namespace NUMINAMATH_CALUDE_average_monthly_production_theorem_l1075_107537

def initial_production : ℝ := 1000

def monthly_increases : List ℝ := [0.05, 0.07, 0.10, 0.04, 0.08, 0.05, 0.07, 0.06, 0.12, 0.10, 0.08]

def calculate_monthly_production (prev : ℝ) (increase : ℝ) : ℝ :=
  prev * (1 + increase)

def calculate_yearly_production (initial : ℝ) (increases : List ℝ) : ℝ :=
  initial + (increases.scanl calculate_monthly_production initial).sum

theorem average_monthly_production_theorem :
  let yearly_production := calculate_yearly_production initial_production monthly_increases
  let average_production := yearly_production / 12
  ∃ ε > 0, |average_production - 1445.084204| < ε :=
sorry

end NUMINAMATH_CALUDE_average_monthly_production_theorem_l1075_107537


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1075_107551

theorem solution_set_of_inequality (x : ℝ) :
  (1/2 - x) * (x - 1/3) > 0 ↔ 1/3 < x ∧ x < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1075_107551


namespace NUMINAMATH_CALUDE_binomial_floor_divisibility_l1075_107571

theorem binomial_floor_divisibility (p n : ℕ) (hp : Prime p) (hn : n ≥ p) :
  p ∣ (Nat.choose n p - n / p) := by
  sorry

end NUMINAMATH_CALUDE_binomial_floor_divisibility_l1075_107571


namespace NUMINAMATH_CALUDE_fifty_eight_impossible_l1075_107528

/-- Represents the population of Rivertown -/
structure RivertownPopulation where
  people : ℕ
  dogs : ℕ
  cats : ℕ
  rabbits : ℕ
  chickens : ℕ
  people_dog_ratio : people = 5 * dogs
  cat_rabbit_ratio : cats = 2 * rabbits
  chicken_people_ratio : chickens = 4 * people

/-- The total population of Rivertown -/
def totalPopulation (pop : RivertownPopulation) : ℕ :=
  pop.people + pop.dogs + pop.cats + pop.rabbits + pop.chickens

/-- Theorem stating that 58 cannot be the total population of Rivertown -/
theorem fifty_eight_impossible (pop : RivertownPopulation) : totalPopulation pop ≠ 58 := by
  sorry

end NUMINAMATH_CALUDE_fifty_eight_impossible_l1075_107528


namespace NUMINAMATH_CALUDE_cube_base_diagonal_l1075_107574

/-- Given a cube with space diagonal length of 5 units, 
    the diagonal of its base has length 5 * sqrt(2/3) units. -/
theorem cube_base_diagonal (c : Real) (h : c > 0) 
  (space_diagonal : c * Real.sqrt 3 = 5) : 
  c * Real.sqrt 2 = 5 * Real.sqrt (2/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_base_diagonal_l1075_107574


namespace NUMINAMATH_CALUDE_dalmatian_spots_l1075_107560

theorem dalmatian_spots (bill_spots phil_spots : ℕ) : 
  bill_spots = 39 → 
  bill_spots = 2 * phil_spots - 1 → 
  bill_spots + phil_spots = 59 := by
sorry

end NUMINAMATH_CALUDE_dalmatian_spots_l1075_107560


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_75_plus_6_l1075_107558

theorem units_digit_of_7_power_75_plus_6 : 
  (7^75 + 6) % 10 = 9 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_75_plus_6_l1075_107558


namespace NUMINAMATH_CALUDE_triangle_segment_sum_squares_l1075_107586

-- Define the triangle ABC and points D and E
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = a^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = b^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = c^2

def RightTriangle (A B C : ℝ × ℝ) : Prop :=
  Triangle A B C ∧ 
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

def DivideHypotenuse (A B C D E : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t > 0 ∧
  D = ((2*B.1 + C.1)/3, (2*B.2 + C.2)/3) ∧
  E = ((B.1 + 2*C.1)/3, (B.2 + 2*C.2)/3)

-- State the theorem
theorem triangle_segment_sum_squares 
  (A B C D E : ℝ × ℝ) 
  (h1 : RightTriangle A B C) 
  (h2 : DivideHypotenuse A B C D E) : 
  ((D.1 - A.1)^2 + (D.2 - A.2)^2) + 
  ((E.1 - D.1)^2 + (E.2 - D.2)^2) + 
  ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 
  2/3 * ((C.1 - B.1)^2 + (C.2 - B.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_segment_sum_squares_l1075_107586


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l1075_107573

def quadratic_equation (m n x : ℝ) : ℝ := 9 * x^2 - 2 * m * x + n

def has_two_real_roots (m n : ℤ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation m n x = 0 ∧ quadratic_equation m n y = 0

def roots_in_interval (m n : ℤ) : Prop :=
  ∀ x : ℝ, quadratic_equation m n x = 0 → 0 < x ∧ x < 1

theorem quadratic_roots_theorem :
  ∀ m n : ℤ, has_two_real_roots m n ∧ roots_in_interval m n ↔ (m = 4 ∧ n = 1) ∨ (m = 5 ∧ n = 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l1075_107573


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1075_107517

theorem sin_cos_identity (x : ℝ) (h : Real.sin (x + π / 3) = 1 / 3) :
  Real.sin (5 * π / 3 - x) - Real.cos (2 * x - π / 3) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1075_107517


namespace NUMINAMATH_CALUDE_toms_apple_purchase_l1075_107536

/-- The problem of determining how many kg of apples Tom purchased -/
theorem toms_apple_purchase (apple_price mango_price total_paid : ℕ) 
  (mango_quantity : ℕ) (h1 : apple_price = 70) (h2 : mango_price = 75) 
  (h3 : mango_quantity = 9) (h4 : total_paid = 1235) :
  ∃ (apple_quantity : ℕ), 
    apple_quantity * apple_price + mango_quantity * mango_price = total_paid ∧ 
    apple_quantity = 8 := by
  sorry

end NUMINAMATH_CALUDE_toms_apple_purchase_l1075_107536


namespace NUMINAMATH_CALUDE_alice_sales_above_quota_l1075_107508

def alice_quota : ℕ := 2000

def shoe_prices : List (String × ℕ) := [
  ("Adidas", 45),
  ("Nike", 60),
  ("Reeboks", 35),
  ("Puma", 50),
  ("Converse", 40)
]

def sales : List (String × ℕ) := [
  ("Nike", 12),
  ("Adidas", 10),
  ("Reeboks", 15),
  ("Puma", 8),
  ("Converse", 14)
]

def total_sales : ℕ := (sales.map (fun (s : String × ℕ) =>
  match shoe_prices.find? (fun (p : String × ℕ) => p.1 = s.1) with
  | some price => s.2 * price.2
  | none => 0
)).sum

theorem alice_sales_above_quota :
  total_sales - alice_quota = 655 := by sorry

end NUMINAMATH_CALUDE_alice_sales_above_quota_l1075_107508


namespace NUMINAMATH_CALUDE_population_decrease_l1075_107529

theorem population_decrease (k : ℝ) (P₀ : ℝ) (n : ℕ) 
  (h1 : -1 < k) (h2 : k < 0) (h3 : P₀ > 0) : 
  P₀ * (1 + k)^(n + 1) < P₀ * (1 + k)^n := by
  sorry

#check population_decrease

end NUMINAMATH_CALUDE_population_decrease_l1075_107529


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l1075_107518

theorem angle_sum_theorem (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (eq1 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (eq2 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l1075_107518


namespace NUMINAMATH_CALUDE_dealer_purchase_fraction_l1075_107549

/-- Represents the pricing details of an article sold by a dealer -/
structure ArticlePricing where
  listPrice : ℝ
  sellingPrice : ℝ
  purchasePrice : ℝ

/-- Conditions for the dealer's pricing strategy -/
def validPricing (a : ArticlePricing) : Prop :=
  a.sellingPrice = 1.5 * a.listPrice ∧ 
  a.sellingPrice = 2 * a.purchasePrice ∧
  a.listPrice > 0

/-- The theorem to be proved -/
theorem dealer_purchase_fraction (a : ArticlePricing) 
  (h : validPricing a) : 
  a.purchasePrice = (3/8 : ℝ) * a.listPrice :=
sorry

end NUMINAMATH_CALUDE_dealer_purchase_fraction_l1075_107549


namespace NUMINAMATH_CALUDE_proposition_truth_l1075_107541

theorem proposition_truth : ∃ (a b : ℝ), (a * b = 0 ∧ a ≠ 0) ∧ (3 ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l1075_107541


namespace NUMINAMATH_CALUDE_unique_pair_cube_prime_l1075_107516

theorem unique_pair_cube_prime : 
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ 
  ∃ (p : ℕ), Prime p ∧ (x * y^3) / (x + y) = p^3 ∧ 
  x = 2 ∧ y = 14 := by
sorry

end NUMINAMATH_CALUDE_unique_pair_cube_prime_l1075_107516


namespace NUMINAMATH_CALUDE_susans_gift_is_eight_l1075_107524

/-- The number of apples Sean had initially -/
def initial_apples : ℕ := 9

/-- The total number of apples Sean had after Susan's gift -/
def total_apples : ℕ := 17

/-- The number of apples Susan gave to Sean -/
def susans_gift : ℕ := total_apples - initial_apples

theorem susans_gift_is_eight : susans_gift = 8 := by
  sorry

end NUMINAMATH_CALUDE_susans_gift_is_eight_l1075_107524


namespace NUMINAMATH_CALUDE_divisibility_of_2_power_n_minus_1_l1075_107525

theorem divisibility_of_2_power_n_minus_1 :
  ∃ (n : ℕ+), ∃ (k : ℕ), 2^n.val - 1 = 17 * k ∧
  ∀ (m : ℕ), 10 ≤ m → m ≤ 20 → m ≠ 17 → ¬∃ (l : ℕ+), ∃ (j : ℕ), 2^l.val - 1 = m * j :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_2_power_n_minus_1_l1075_107525


namespace NUMINAMATH_CALUDE_bowling_team_average_weight_l1075_107503

theorem bowling_team_average_weight 
  (original_players : ℕ) 
  (original_average : ℝ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) :
  original_players = 7 →
  original_average = 103 →
  new_player1_weight = 110 →
  new_player2_weight = 60 →
  let total_weight := original_players * original_average + new_player1_weight + new_player2_weight
  let new_total_players := original_players + 2
  let new_average := total_weight / new_total_players
  new_average = 99 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_average_weight_l1075_107503


namespace NUMINAMATH_CALUDE_cherry_pie_pitting_time_l1075_107520

/-- Represents the time needed to pit cherries for each pound -/
structure PittingTime where
  first : ℕ  -- Time in minutes for the first pound
  second : ℕ -- Time in minutes for the second pound
  third : ℕ  -- Time in minutes for the third pound

/-- Calculates the total time in hours to pit cherries for a cherry pie -/
def total_pitting_time (pt : PittingTime) : ℚ :=
  (pt.first + pt.second + pt.third) / 60

/-- Theorem: Given the conditions, it takes 2 hours to pit all cherries for the pie -/
theorem cherry_pie_pitting_time :
  ∀ (pt : PittingTime),
    (∃ (n : ℕ), pt.first = 10 * (80 / 20) ∧
                pt.second = 8 * (80 / 20) ∧
                pt.third = 12 * (80 / 20) ∧
                n = 3) →
    total_pitting_time pt = 2 := by
  sorry


end NUMINAMATH_CALUDE_cherry_pie_pitting_time_l1075_107520


namespace NUMINAMATH_CALUDE_group_size_is_ten_l1075_107588

/-- The number of people in a group that can hold a certain number of boxes. -/
def group_size (total_boxes : ℕ) (boxes_per_person : ℕ) : ℕ :=
  total_boxes / boxes_per_person

/-- Theorem: The group size is 10 when the total boxes is 20 and each person can hold 2 boxes. -/
theorem group_size_is_ten : group_size 20 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_group_size_is_ten_l1075_107588


namespace NUMINAMATH_CALUDE_painting_cost_after_modification_l1075_107579

/-- Represents the dimensions of a rectangular surface -/
structure Dimensions where
  length : Float
  width : Float

/-- Calculates the area of a rectangular surface -/
def area (d : Dimensions) : Float :=
  d.length * d.width

/-- Represents a room with walls, windows, and doors -/
structure Room where
  walls : List Dimensions
  windows : List Dimensions
  doors : List Dimensions

/-- Calculates the total wall area of a room -/
def totalWallArea (r : Room) : Float :=
  r.walls.map area |> List.sum

/-- Calculates the total area of openings (windows and doors) in a room -/
def totalOpeningArea (r : Room) : Float :=
  (r.windows.map area |> List.sum) + (r.doors.map area |> List.sum)

/-- Calculates the net paintable area of a room -/
def netPaintableArea (r : Room) : Float :=
  totalWallArea r - totalOpeningArea r

/-- Calculates the cost to paint a room given the cost per square foot -/
def paintingCost (r : Room) (costPerSqFt : Float) : Float :=
  netPaintableArea r * costPerSqFt

/-- Increases the dimensions of a room by a given factor -/
def increaseRoomSize (r : Room) (factor : Float) : Room :=
  { walls := r.walls.map fun d => { length := d.length * factor, width := d.width * factor },
    windows := r.windows,
    doors := r.doors }

/-- Adds additional windows and doors to a room -/
def addOpenings (r : Room) (additionalWindows : List Dimensions) (additionalDoors : List Dimensions) : Room :=
  { walls := r.walls,
    windows := r.windows ++ additionalWindows,
    doors := r.doors ++ additionalDoors }

theorem painting_cost_after_modification (originalRoom : Room) (costPerSqFt : Float) : 
  let modifiedRoom := addOpenings (increaseRoomSize originalRoom 1.5) 
                        [⟨3, 4⟩, ⟨3, 4⟩] [⟨3, 7⟩]
  paintingCost modifiedRoom costPerSqFt = 1732.50 :=
by
  sorry

#check painting_cost_after_modification

end NUMINAMATH_CALUDE_painting_cost_after_modification_l1075_107579
