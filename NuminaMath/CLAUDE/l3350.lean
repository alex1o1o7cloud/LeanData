import Mathlib

namespace NUMINAMATH_CALUDE_fedora_cleaning_time_l3350_335021

/-- Represents the cleaning problem of Fedora Egorovna's stove wall. -/
def CleaningProblem (total_sections : ℕ) (cleaned_sections : ℕ) (time_spent : ℕ) : Prop :=
  let cleaning_rate := time_spent / cleaned_sections
  let total_time := total_sections * cleaning_rate
  let additional_time := total_time - time_spent
  additional_time = 192

/-- Theorem stating that given the conditions of Fedora's cleaning,
    the additional time required is 192 minutes. -/
theorem fedora_cleaning_time :
  CleaningProblem 27 3 24 :=
by
  sorry

#check fedora_cleaning_time

end NUMINAMATH_CALUDE_fedora_cleaning_time_l3350_335021


namespace NUMINAMATH_CALUDE_product_quotient_positive_l3350_335074

theorem product_quotient_positive 
  (a b c d : ℝ) 
  (x₁ x₂ x₃ x₄ : ℝ) 
  (ha : a < 0) 
  (hb : b < 0) 
  (hc : c < 0) 
  (hd : d < 0) 
  (h : |x₁ - a| + |x₂ + b| + |x₃ - c| + |x₄ + d| = 0) : 
  (x₁ * x₂) / (x₃ * x₄) > 0 :=
by sorry

end NUMINAMATH_CALUDE_product_quotient_positive_l3350_335074


namespace NUMINAMATH_CALUDE_sum_of_roots_l3350_335064

theorem sum_of_roots (a b : ℝ) : 
  a * (a - 4) = 12 → b * (b - 4) = 12 → a ≠ b → a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3350_335064


namespace NUMINAMATH_CALUDE_total_baseball_cards_l3350_335014

theorem total_baseball_cards : 
  let number_of_people : ℕ := 6
  let cards_per_person : ℕ := 8
  number_of_people * cards_per_person = 48 :=
by sorry

end NUMINAMATH_CALUDE_total_baseball_cards_l3350_335014


namespace NUMINAMATH_CALUDE_increasing_sequence_range_l3350_335094

def a_n (a : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 7 then (4 - a) * n - 10 else a^(n - 6)

theorem increasing_sequence_range (a : ℝ) :
  (∀ n m : ℕ, n < m → a_n a n < a_n a m) →
  2 < a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_increasing_sequence_range_l3350_335094


namespace NUMINAMATH_CALUDE_variance_of_surviving_trees_l3350_335026

/-- The number of osmanthus trees transplanted -/
def n : ℕ := 4

/-- The probability of survival for each tree -/
def p : ℚ := 4/5

/-- The random variable representing the number of surviving trees -/
def X : ℕ → ℚ := sorry

/-- The expected value of X -/
def E_X : ℚ := n * p

/-- The variance of X -/
def Var_X : ℚ := n * p * (1 - p)

theorem variance_of_surviving_trees :
  Var_X = 16/25 := by sorry

end NUMINAMATH_CALUDE_variance_of_surviving_trees_l3350_335026


namespace NUMINAMATH_CALUDE_successive_discounts_equivalence_l3350_335078

theorem successive_discounts_equivalence :
  let original_price : ℝ := 50
  let first_discount : ℝ := 0.30
  let second_discount : ℝ := 0.15
  let equivalent_discount : ℝ := 0.405

  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)

  final_price = original_price * (1 - equivalent_discount) :=
by sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalence_l3350_335078


namespace NUMINAMATH_CALUDE_prime_cube_plus_seven_composite_l3350_335009

theorem prime_cube_plus_seven_composite (P : ℕ) (h1 : Nat.Prime P) (h2 : Nat.Prime (P^3 + 5)) :
  ¬Nat.Prime (P^3 + 7) ∧ (P^3 + 7) > 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_cube_plus_seven_composite_l3350_335009


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3350_335059

theorem least_subtraction_for_divisibility : 
  ∃! x : ℕ, x ≤ 14 ∧ (42398 - x) % 15 = 0 ∧ ∀ y : ℕ, y < x → (42398 - y) % 15 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3350_335059


namespace NUMINAMATH_CALUDE_no_natural_numbers_satisfying_condition_l3350_335035

theorem no_natural_numbers_satisfying_condition :
  ∀ (x y : ℕ), x + y - 2021 ≥ Nat.gcd x y + Nat.lcm x y :=
by sorry

end NUMINAMATH_CALUDE_no_natural_numbers_satisfying_condition_l3350_335035


namespace NUMINAMATH_CALUDE_quadratic_prime_roots_l3350_335054

theorem quadratic_prime_roots (k : ℕ) : 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ 
   p + q = 99 ∧ p * q = k ∧
   ∀ x : ℝ, x^2 - 99*x + k = 0 ↔ (x = p ∨ x = q)) →
  k = 194 :=
sorry

end NUMINAMATH_CALUDE_quadratic_prime_roots_l3350_335054


namespace NUMINAMATH_CALUDE_both_are_dwarves_l3350_335088

-- Define the types of inhabitants
inductive Inhabitant : Type
| Elf : Inhabitant
| Dwarf : Inhabitant

-- Define the types of statements
inductive Statement : Type
| GoldStatement : Statement
| AboutDwarf : Statement
| Other : Statement

-- Define a function to determine if a statement is true based on the speaker and the type of statement
def isTruthful (speaker : Inhabitant) (stmnt : Statement) : Prop :=
  match speaker, stmnt with
  | Inhabitant.Dwarf, Statement.GoldStatement => False
  | Inhabitant.Elf, Statement.AboutDwarf => False
  | _, _ => True

-- A's statement
def a_statement : Statement := Statement.GoldStatement

-- B's statement about A
def b_statement (a_type : Inhabitant) : Statement :=
  match a_type with
  | Inhabitant.Dwarf => Statement.Other
  | Inhabitant.Elf => Statement.AboutDwarf

-- Theorem to prove
theorem both_are_dwarves :
  ∃ (a_type b_type : Inhabitant),
    a_type = Inhabitant.Dwarf ∧
    b_type = Inhabitant.Dwarf ∧
    isTruthful a_type a_statement = False ∧
    isTruthful b_type (b_statement a_type) = True :=
sorry

end NUMINAMATH_CALUDE_both_are_dwarves_l3350_335088


namespace NUMINAMATH_CALUDE_a_gt_1_sufficient_not_necessary_for_a_sq_gt_a_l3350_335089

theorem a_gt_1_sufficient_not_necessary_for_a_sq_gt_a :
  (∀ a : ℝ, a > 1 → a^2 > a) ∧
  (∃ a : ℝ, a^2 > a ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_1_sufficient_not_necessary_for_a_sq_gt_a_l3350_335089


namespace NUMINAMATH_CALUDE_christophers_gabrielas_age_ratio_l3350_335046

/-- Proves that given Christopher is 2 times as old as Gabriela and Christopher is 24 years old, 
    the ratio of Christopher's age to Gabriela's age nine years ago is 5:1. -/
theorem christophers_gabrielas_age_ratio : 
  ∀ (christopher_age gabriela_age : ℕ),
    christopher_age = 2 * gabriela_age →
    christopher_age = 24 →
    (christopher_age - 9) / (gabriela_age - 9) = 5 := by
  sorry

end NUMINAMATH_CALUDE_christophers_gabrielas_age_ratio_l3350_335046


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l3350_335047

theorem min_value_of_sum_of_squares (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 → (a + 2)^2 + (b + 2)^2 ≤ (x + 2)^2 + (y + 2)^2) ∧
  (a + 2)^2 + (b + 2)^2 = 25/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l3350_335047


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3350_335057

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 4| + 3 * y = 11 :=
by
  -- The unique solution is y = 3.5
  use 3.5
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3350_335057


namespace NUMINAMATH_CALUDE_sports_enthusiasts_difference_l3350_335001

theorem sports_enthusiasts_difference (total : ℕ) (basketball : ℕ) (football : ℕ)
  (h_total : total = 46)
  (h_basketball : basketball = 23)
  (h_football : football = 29) :
  basketball - (basketball + football - total) = 17 :=
by sorry

end NUMINAMATH_CALUDE_sports_enthusiasts_difference_l3350_335001


namespace NUMINAMATH_CALUDE_boat_upstream_downstream_distance_l3350_335017

/-- Proves that a boat with a given speed in still water, traveling a certain distance upstream in one hour, will travel a specific distance downstream in one hour. -/
theorem boat_upstream_downstream_distance 
  (v : ℝ) -- Speed of the boat in still water (km/h)
  (d_upstream : ℝ) -- Distance traveled upstream in one hour (km)
  (h1 : v = 8) -- The boat's speed in still water is 8 km/h
  (h2 : d_upstream = 5) -- The boat travels 5 km upstream in one hour
  : ∃ d_downstream : ℝ, d_downstream = 11 ∧ d_downstream = v + (v - d_upstream) := by
  sorry

end NUMINAMATH_CALUDE_boat_upstream_downstream_distance_l3350_335017


namespace NUMINAMATH_CALUDE_encoded_CDE_value_l3350_335013

/-- Represents the digits in the base 7 encoding system -/
inductive Digit
  | A | B | C | D | E | F | G

/-- Represents a number in the base 7 encoding system -/
def EncodedNumber := List Digit

/-- Converts an EncodedNumber to its base 10 representation -/
def to_base_10 : EncodedNumber → ℕ := sorry

/-- Checks if two EncodedNumbers are consecutive -/
def are_consecutive (a b : EncodedNumber) : Prop := sorry

/-- The main theorem -/
theorem encoded_CDE_value :
  ∃ (bcg bcf bad : EncodedNumber),
    (are_consecutive bcg bcf) ∧
    (are_consecutive bcf bad) ∧
    bcg = [Digit.B, Digit.C, Digit.G] ∧
    bcf = [Digit.B, Digit.C, Digit.F] ∧
    bad = [Digit.B, Digit.A, Digit.D] →
    to_base_10 [Digit.C, Digit.D, Digit.E] = 329 := by
  sorry

end NUMINAMATH_CALUDE_encoded_CDE_value_l3350_335013


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3350_335093

theorem inequality_and_equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b + b / c + c / a + 1 ≥ (2 * Real.sqrt 2 / 3) * (Real.sqrt ((a + b) / c) + Real.sqrt ((b + c) / a) + Real.sqrt ((c + a) / b))) ∧
  (a / b + b / c + c / a + 1 = (2 * Real.sqrt 2 / 3) * (Real.sqrt ((a + b) / c) + Real.sqrt ((b + c) / a) + Real.sqrt ((c + a) / b)) ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3350_335093


namespace NUMINAMATH_CALUDE_other_intersection_point_l3350_335012

/-- Two circles with centers on a line intersecting at two points -/
structure TwoCirclesIntersection where
  -- The line equation: x - y + 1 = 0
  line : ℝ → ℝ → Prop
  line_eq : ∀ x y, line x y ↔ x - y + 1 = 0
  
  -- The circles intersect at two different points
  intersect_points : Fin 2 → ℝ × ℝ
  different_points : intersect_points 0 ≠ intersect_points 1
  
  -- One intersection point is (-2, 2)
  known_point : intersect_points 0 = (-2, 2)

/-- The other intersection point has coordinates (1, -1) -/
theorem other_intersection_point (c : TwoCirclesIntersection) : 
  c.intersect_points 1 = (1, -1) := by
  sorry

end NUMINAMATH_CALUDE_other_intersection_point_l3350_335012


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l3350_335045

theorem arccos_one_over_sqrt_two (π : Real) :
  Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l3350_335045


namespace NUMINAMATH_CALUDE_perpendicular_equal_diagonals_not_sufficient_for_square_l3350_335022

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of quadrilaterals
def has_perpendicular_diagonals (q : Quadrilateral) : Prop := sorry
def has_equal_diagonals (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem perpendicular_equal_diagonals_not_sufficient_for_square :
  ∃ (q : Quadrilateral), has_perpendicular_diagonals q ∧ has_equal_diagonals q ∧ ¬is_square q :=
sorry

end NUMINAMATH_CALUDE_perpendicular_equal_diagonals_not_sufficient_for_square_l3350_335022


namespace NUMINAMATH_CALUDE_f_range_of_a_l3350_335090

/-- The function f(x) defined as |x-1| + |x-a| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

/-- The theorem stating that if f(x) ≥ 2 for all real x, then a is in (-∞, -1] ∪ [3, +∞) -/
theorem f_range_of_a (a : ℝ) : (∀ x : ℝ, f a x ≥ 2) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_f_range_of_a_l3350_335090


namespace NUMINAMATH_CALUDE_steve_commute_speed_l3350_335067

theorem steve_commute_speed (distance : ℝ) (total_time : ℝ) : 
  distance > 0 → 
  total_time > 0 → 
  ∃ (outbound_speed : ℝ), 
    outbound_speed > 0 ∧ 
    (distance / outbound_speed + distance / (2 * outbound_speed) = total_time) → 
    2 * outbound_speed = 14 := by
  sorry

end NUMINAMATH_CALUDE_steve_commute_speed_l3350_335067


namespace NUMINAMATH_CALUDE_only_C_in_position_I_l3350_335077

-- Define a structure for a rectangle with labeled sides
structure LabeledRectangle where
  top : ℕ
  bottom : ℕ
  left : ℕ
  right : ℕ

-- Define the five rectangles
def rectangle_A : LabeledRectangle := ⟨1, 9, 4, 6⟩
def rectangle_B : LabeledRectangle := ⟨0, 6, 1, 3⟩
def rectangle_C : LabeledRectangle := ⟨8, 2, 3, 5⟩
def rectangle_D : LabeledRectangle := ⟨5, 8, 7, 4⟩
def rectangle_E : LabeledRectangle := ⟨2, 0, 9, 7⟩

-- Define a function to check if a rectangle can be placed in position I
def can_be_placed_in_position_I (r : LabeledRectangle) : Prop :=
  ∃ (r2 r4 : LabeledRectangle), 
    r.right = r2.left ∧ r.bottom = r4.top

-- Theorem stating that only rectangle C can be placed in position I
theorem only_C_in_position_I : 
  can_be_placed_in_position_I rectangle_C ∧
  ¬can_be_placed_in_position_I rectangle_A ∧
  ¬can_be_placed_in_position_I rectangle_B ∧
  ¬can_be_placed_in_position_I rectangle_D ∧
  ¬can_be_placed_in_position_I rectangle_E :=
sorry

end NUMINAMATH_CALUDE_only_C_in_position_I_l3350_335077


namespace NUMINAMATH_CALUDE_ratio_problem_l3350_335068

/-- Custom operation @ for positive integers -/
def custom_op (k j : ℕ+) : ℕ+ :=
  sorry

theorem ratio_problem (a b : ℕ+) (t : ℚ) : 
  a = 2020 → t = (a : ℚ) / (b : ℚ) → t = 1/2 → b = 4040 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3350_335068


namespace NUMINAMATH_CALUDE_parallelogram_vector_operations_l3350_335099

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (A B C D : V)

-- Define the parallelogram condition
def is_parallelogram (A B C D : V) : Prop :=
  B - A = C - D ∧ D - A = C - B

-- Theorem statement
theorem parallelogram_vector_operations 
  (h : is_parallelogram A B C D) : 
  (B - A) + (D - A) = C - A ∧ (D - A) - (B - A) = D - B := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vector_operations_l3350_335099


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l3350_335080

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 1)^2
def parabola2 (x y : ℝ) : Prop := x - 2 = (y + 1)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

theorem intersection_sum_zero :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧
    (x₃, y₃) ≠ (x₄, y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_zero_l3350_335080


namespace NUMINAMATH_CALUDE_construct_triangle_from_equilateral_vertices_l3350_335048

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is acute-angled -/
def isAcute (t : Triangle) : Prop :=
  sorry

/-- Checks if three points form an equilateral triangle -/
def isEquilateral (A B C : Point) : Prop :=
  sorry

/-- Main theorem: Given an acute-angled triangle A₁B₁C₁, there exists a unique triangle ABC
    such that A₁, B₁, and C₁ are the vertices of equilateral triangles drawn outward
    on the sides BC, CA, and AB respectively -/
theorem construct_triangle_from_equilateral_vertices
  (A₁ B₁ C₁ : Point) (h : isAcute (Triangle.mk A₁ B₁ C₁)) :
  ∃! (ABC : Triangle),
    isEquilateral ABC.B ABC.C A₁ ∧
    isEquilateral ABC.C ABC.A B₁ ∧
    isEquilateral ABC.A ABC.B C₁ :=
  sorry

end NUMINAMATH_CALUDE_construct_triangle_from_equilateral_vertices_l3350_335048


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_10_l3350_335055

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  (∀ n, a (n + 1) = a n + d) ∧ (d > 0) ∧ (a 1 + a 5 = 4) ∧ (a 2 * a 4 = -5)

/-- The sum of the first n terms of an arithmetic sequence -/
def ArithmeticSum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

/-- Theorem: Sum of first 10 terms of the arithmetic sequence is 95 -/
theorem arithmetic_sequence_sum_10 (a : ℕ → ℚ) (d : ℚ) :
  ArithmeticSequence a d → ArithmeticSum a 10 = 95 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_10_l3350_335055


namespace NUMINAMATH_CALUDE_binary_1101_equals_13_l3350_335052

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1101₂ -/
def binary_1101 : List Bool := [true, false, true, true]

theorem binary_1101_equals_13 : binary_to_decimal binary_1101 = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_equals_13_l3350_335052


namespace NUMINAMATH_CALUDE_susie_piggy_bank_l3350_335095

theorem susie_piggy_bank (X : ℝ) : X + 0.2 * X = 240 → X = 200 := by
  sorry

end NUMINAMATH_CALUDE_susie_piggy_bank_l3350_335095


namespace NUMINAMATH_CALUDE_max_non_overlapping_ge_min_covering_l3350_335024

/-- A polygon in a 2D plane -/
structure Polygon where
  -- Add necessary fields for a polygon

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a circle's center is inside a polygon -/
def Circle.centerInside (c : Circle) (p : Polygon) : Prop :=
  sorry

/-- Checks if two circles are non-overlapping -/
def Circle.nonOverlapping (c1 c2 : Circle) : Prop :=
  sorry

/-- Checks if a set of circles covers a polygon -/
def covers (circles : Set Circle) (p : Polygon) : Prop :=
  sorry

/-- The maximum number of non-overlapping circles of diameter 1 with centers inside the polygon -/
def maxNonOverlappingCircles (p : Polygon) : ℕ :=
  sorry

/-- The minimum number of circles of radius 1 that can cover the polygon -/
def minCoveringCircles (p : Polygon) : ℕ :=
  sorry

/-- Theorem: The maximum number of non-overlapping circles of diameter 1 with centers inside a polygon
    is greater than or equal to the minimum number of circles of radius 1 needed to cover the polygon -/
theorem max_non_overlapping_ge_min_covering (p : Polygon) :
  maxNonOverlappingCircles p ≥ minCoveringCircles p :=
sorry

end NUMINAMATH_CALUDE_max_non_overlapping_ge_min_covering_l3350_335024


namespace NUMINAMATH_CALUDE_both_not_land_l3350_335003

-- Define the propositions
variable (p q : Prop)

-- p represents "A lands within the designated area"
-- q represents "B lands within the designated area"

-- Theorem: "Both trainees did not land within the designated area" 
-- is equivalent to (¬p) ∧ (¬q)
theorem both_not_land (p q : Prop) : 
  (¬p ∧ ¬q) ↔ ¬(p ∨ q) :=
sorry

end NUMINAMATH_CALUDE_both_not_land_l3350_335003


namespace NUMINAMATH_CALUDE_prime_condition_l3350_335050

theorem prime_condition (p : ℕ) : 
  Nat.Prime p ∧ Nat.Prime (p^4 - 3*p^2 + 9) → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_condition_l3350_335050


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3350_335085

theorem inequality_solution_range (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 2 * x + 6 * k < 0) ↔ k < Real.sqrt 6 / 6 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3350_335085


namespace NUMINAMATH_CALUDE_range_of_m_for_real_roots_l3350_335070

theorem range_of_m_for_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - x + m = 0 ∨ (m-1)*x^2 + 2*x + 1 = 0 ∨ (m-2)*x^2 + 2*x - 1 = 0) →
  (∃ y : ℝ, y^2 - y + m = 0 ∨ (m-1)*y^2 + 2*y + 1 = 0 ∨ (m-2)*y^2 + 2*y - 1 = 0) →
  (x ≠ y) →
  (m ≤ 1/4 ∨ (1 ≤ m ∧ m ≤ 2)) := by
sorry

end NUMINAMATH_CALUDE_range_of_m_for_real_roots_l3350_335070


namespace NUMINAMATH_CALUDE_committee_selection_l3350_335039

theorem committee_selection (n : ℕ) (h : Nat.choose n 2 = 15) : Nat.choose n 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l3350_335039


namespace NUMINAMATH_CALUDE_arithmetic_sequence_60th_term_l3350_335025

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_60th_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic a)
  (h_first : a 1 = 3)
  (h_fifteenth : a 15 = 31) :
  a 60 = 121 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_60th_term_l3350_335025


namespace NUMINAMATH_CALUDE_complement_of_α_l3350_335058

-- Define a custom type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the given angle α
def α : Angle := ⟨25, 39⟩

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  let total_minutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  ⟨total_minutes / 60, total_minutes % 60⟩

-- Theorem statement
theorem complement_of_α :
  complement α = ⟨64, 21⟩ := by
  sorry

end NUMINAMATH_CALUDE_complement_of_α_l3350_335058


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3350_335073

theorem arithmetic_mean_problem (x : ℝ) : 
  ((x + 10) + 20 + 3*x + 16 + (3*x + 6)) / 5 = 30 → x = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3350_335073


namespace NUMINAMATH_CALUDE_polynomial_real_root_l3350_335019

theorem polynomial_real_root (b : ℝ) :
  (∃ x : ℝ, x^3 + b*x^2 - x + b = 0) ↔ b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l3350_335019


namespace NUMINAMATH_CALUDE_divisible_by_seven_l3350_335097

def number (x : ℕ) : ℕ := 
  666666666666666666666666666666666666666666666666666 * 10^51 + 
  x * 10^50 + 
  555555555555555555555555555555555555555555555555555

theorem divisible_by_seven (x : ℕ) : 
  x < 10 → (number x % 7 = 0 ↔ x = 2 ∨ x = 9) := by sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l3350_335097


namespace NUMINAMATH_CALUDE_ranch_problem_l3350_335098

theorem ranch_problem (ponies horses : ℕ) (horseshoe_fraction : ℚ) :
  ponies + horses = 163 →
  horses = ponies + 3 →
  ∃ (iceland_ponies : ℕ), iceland_ponies = (5 : ℚ) / 8 * horseshoe_fraction * ponies →
  horseshoe_fraction = 1 / 10 :=
by sorry

end NUMINAMATH_CALUDE_ranch_problem_l3350_335098


namespace NUMINAMATH_CALUDE_five_distinct_naturals_product_1000_l3350_335056

theorem five_distinct_naturals_product_1000 :
  ∃ (a b c d e : ℕ), a * b * c * d * e = 1000 ∧
                     a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
                     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
                     c ≠ d ∧ c ≠ e ∧
                     d ≠ e :=
by
  use 1, 2, 4, 5, 25
  sorry

end NUMINAMATH_CALUDE_five_distinct_naturals_product_1000_l3350_335056


namespace NUMINAMATH_CALUDE_S_is_open_line_segment_l3350_335066

-- Define the set of points satisfying the conditions
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = 1 ∧ p.1^2 + p.2^2 < 25}

-- Theorem statement
theorem S_is_open_line_segment :
  ∃ (a b : ℝ × ℝ), a ≠ b ∧
    S = {p : ℝ × ℝ | ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ p = (1 - t) • a + t • b} :=
sorry

end NUMINAMATH_CALUDE_S_is_open_line_segment_l3350_335066


namespace NUMINAMATH_CALUDE_octagon_area_l3350_335015

/-- The area of an octagon inscribed in a rectangle --/
theorem octagon_area (rectangle_width rectangle_height triangle_base triangle_height : ℝ) 
  (hw : rectangle_width = 5)
  (hh : rectangle_height = 8)
  (htb : triangle_base = 1)
  (hth : triangle_height = 4) :
  rectangle_width * rectangle_height - 4 * (1/2 * triangle_base * triangle_height) = 32 :=
by sorry

end NUMINAMATH_CALUDE_octagon_area_l3350_335015


namespace NUMINAMATH_CALUDE_bear_shelves_l3350_335016

def bear_problem (initial_stock : ℕ) (new_shipment : ℕ) (bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

theorem bear_shelves :
  bear_problem 17 10 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bear_shelves_l3350_335016


namespace NUMINAMATH_CALUDE_martha_blocks_l3350_335023

/-- The number of blocks Martha ends with is equal to her initial blocks plus the blocks she finds -/
theorem martha_blocks (initial_blocks found_blocks : ℕ) :
  initial_blocks + found_blocks = initial_blocks + found_blocks :=
by sorry

#check martha_blocks 4 80

end NUMINAMATH_CALUDE_martha_blocks_l3350_335023


namespace NUMINAMATH_CALUDE_road_repaving_l3350_335051

theorem road_repaving (total_repaved : ℕ) (repaved_today : ℕ) 
  (h1 : total_repaved = 4938)
  (h2 : repaved_today = 805) :
  total_repaved - repaved_today = 4133 := by
  sorry

end NUMINAMATH_CALUDE_road_repaving_l3350_335051


namespace NUMINAMATH_CALUDE_point_on_bisector_implies_a_eq_neg_five_l3350_335030

/-- A point P with coordinates (x, y) is on the bisector of the second and fourth quadrants if x + y = 0 -/
def on_bisector (x y : ℝ) : Prop := x + y = 0

/-- Given that point P (a+3, 7+a) is on the bisector of the second and fourth quadrants, prove that a = -5 -/
theorem point_on_bisector_implies_a_eq_neg_five (a : ℝ) :
  on_bisector (a + 3) (7 + a) → a = -5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_bisector_implies_a_eq_neg_five_l3350_335030


namespace NUMINAMATH_CALUDE_problem_solution_l3350_335086

theorem problem_solution : 3^(0^(2^3)) + ((3^1)^0)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3350_335086


namespace NUMINAMATH_CALUDE_not_right_triangle_not_triangle_l3350_335004

theorem not_right_triangle (a b c : ℝ) (ha : a = 1) (hb : b = 1) (hc : c = 2) :
  ¬(a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) :=
by sorry

theorem not_triangle (a b c : ℝ) (ha : a = 1) (hb : b = 1) (hc : c = 2) :
  ¬(a + b > c ∧ b + c > a ∧ c + a > b) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_not_triangle_l3350_335004


namespace NUMINAMATH_CALUDE_markup_markdown_equivalence_l3350_335028

theorem markup_markdown_equivalence (original_price : ℝ) (markup_percentage : ℝ) (markdown_percentage : ℝ)
  (h1 : markup_percentage = 25)
  (h2 : original_price * (1 + markup_percentage / 100) * (1 - markdown_percentage / 100) = original_price) :
  markdown_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_markup_markdown_equivalence_l3350_335028


namespace NUMINAMATH_CALUDE_incorrect_transformation_l3350_335002

theorem incorrect_transformation (a b c : ℝ) : 
  (a = b) → ¬(∀ c, a / c = b / c) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_transformation_l3350_335002


namespace NUMINAMATH_CALUDE_sequence_general_term_l3350_335005

theorem sequence_general_term (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) :
  a 1 = 1 ∧
  (∀ n : ℕ+, S n = (n + 2 : ℚ) / 3 * a n) →
  ∀ n : ℕ+, a n = (n * (n + 1) : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3350_335005


namespace NUMINAMATH_CALUDE_derivative_at_one_l3350_335043

/-- Given a function f: ℝ → ℝ satisfying f(x) = 2x * f'(1) + 1/x for all x ≠ 0,
    prove that f'(1) = 1 -/
theorem derivative_at_one (f : ℝ → ℝ) (hf : ∀ x ≠ 0, f x = 2 * x * (deriv f 1) + 1 / x) :
  deriv f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l3350_335043


namespace NUMINAMATH_CALUDE_bus_problem_solution_l3350_335044

def bus_problem (initial : ℕ) 
  (stop1_off stop1_on : ℕ) 
  (stop2_off stop2_on : ℕ) 
  (stop3_off stop3_on : ℕ) : ℕ :=
  initial - stop1_off + stop1_on - stop2_off + stop2_on - stop3_off + stop3_on

theorem bus_problem_solution : 
  bus_problem 10 3 2 1 4 2 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_solution_l3350_335044


namespace NUMINAMATH_CALUDE_arithmetic_sequence_reaches_negative_27_l3350_335006

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_reaches_negative_27 :
  ∃ n : ℕ, arithmetic_sequence 1 (-2) n = -27 ∧ n = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_reaches_negative_27_l3350_335006


namespace NUMINAMATH_CALUDE_proposition_equivalence_l3350_335049

-- Define the concept of angles
def Angle : Type := ℝ

-- Define what it means for two angles to be equal
def equal_angles (a b : Angle) : Prop := a = b

-- Define what it means for two angles to be vertical angles
def vertical_angles (a b : Angle) : Prop := sorry

-- State the original proposition
def original_proposition : Prop :=
  ∀ a b : Angle, equal_angles a b → vertical_angles a b

-- State the conditional form
def conditional_form : Prop :=
  ∀ a b : Angle, vertical_angles a b → equal_angles a b

-- Theorem stating the equivalence of the two forms
theorem proposition_equivalence : original_proposition ↔ conditional_form :=
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l3350_335049


namespace NUMINAMATH_CALUDE_wheel_turns_time_l3350_335020

theorem wheel_turns_time (turns_per_two_hours : ℕ) (h : turns_per_two_hours = 1440) :
  (6 : ℝ) * (3600 : ℝ) / (turns_per_two_hours : ℝ) * 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_wheel_turns_time_l3350_335020


namespace NUMINAMATH_CALUDE_cube_expansion_2013_l3350_335092

theorem cube_expansion_2013 : ∃! n : ℕ, 
  n > 0 ∧ 
  (n - 1)^2 + (n - 1) ≤ 2013 ∧ 
  2013 < n^2 + n ∧
  n = 45 := by sorry

end NUMINAMATH_CALUDE_cube_expansion_2013_l3350_335092


namespace NUMINAMATH_CALUDE_tiles_needed_for_room_l3350_335065

/-- Proves that the number of 3-inch by 5-inch tiles needed to cover a 10-foot by 15-foot room is 1440 -/
theorem tiles_needed_for_room : 
  let room_length : ℚ := 10
  let room_width : ℚ := 15
  let tile_length : ℚ := 3 / 12  -- 3 inches in feet
  let tile_width : ℚ := 5 / 12   -- 5 inches in feet
  let room_area := room_length * room_width
  let tile_area := tile_length * tile_width
  let tiles_needed := room_area / tile_area
  ⌈tiles_needed⌉ = 1440 := by sorry

end NUMINAMATH_CALUDE_tiles_needed_for_room_l3350_335065


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l3350_335031

theorem intersection_of_three_lines (k : ℝ) : 
  (∃ x y : ℝ, y = 7 * x - 2 ∧ y = -3 * x + 14 ∧ y = 4 * x + k) → k = 2.8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l3350_335031


namespace NUMINAMATH_CALUDE_derivative_f_at_specific_point_l3350_335010

-- Define the function f
def f (x : ℝ) : ℝ := x^2008

-- State the theorem
theorem derivative_f_at_specific_point :
  deriv f ((1 / 2008 : ℝ)^(1 / 2007)) = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_specific_point_l3350_335010


namespace NUMINAMATH_CALUDE_range_of_PQ_length_l3350_335062

/-- Circle C in the Cartesian coordinate system -/
def CircleC (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 2

/-- Point A is on the x-axis -/
def PointA (x : ℝ) : Prop := true

/-- AP is tangent to circle C at point P -/
def TangentAP (A P : ℝ × ℝ) : Prop := sorry

/-- AQ is tangent to circle C at point Q -/
def TangentAQ (A Q : ℝ × ℝ) : Prop := sorry

/-- The length of segment PQ -/
def LengthPQ (P Q : ℝ × ℝ) : ℝ := sorry

theorem range_of_PQ_length :
  ∀ A P Q : ℝ × ℝ,
    PointA A.1 →
    CircleC P.1 P.2 →
    CircleC Q.1 Q.2 →
    TangentAP A P →
    TangentAQ A Q →
    (2 * Real.sqrt 14 / 3 ≤ LengthPQ P Q) ∧ (LengthPQ P Q < 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_PQ_length_l3350_335062


namespace NUMINAMATH_CALUDE_square_reassembly_l3350_335007

/-- Given two squares with side lengths a and b (where a > b), 
    they can be cut and reassembled into a single square with side length √(a² + b²) -/
theorem square_reassembly (a b : ℝ) (h : a > b) (h' : a > 0) (h'' : b > 0) :
  ∃ (new_side : ℝ), 
    new_side = Real.sqrt (a^2 + b^2) ∧ 
    new_side^2 = a^2 + b^2 :=
by sorry

end NUMINAMATH_CALUDE_square_reassembly_l3350_335007


namespace NUMINAMATH_CALUDE_calculation_proof_l3350_335008

theorem calculation_proof : (1/2)⁻¹ + (Real.sqrt 2)^2 - 4 * |(-(1/2))| = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3350_335008


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3350_335071

-- Define the function f
def f (x : ℝ) : ℝ := x^2 * (x - 2) + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x * (x - 2) + x^2

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  (∀ x y : ℝ, y - y₀ = m * (x - x₀)) ↔ (∀ x y : ℝ, x + y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3350_335071


namespace NUMINAMATH_CALUDE_ten_row_triangle_pieces_l3350_335096

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Calculates the nth triangular number -/
def triangularNumber (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Represents the structure of a triangle made of rods and connectors -/
structure RodTriangle where
  rows : ℕ
  firstRowRods : ℕ
  rodIncrement : ℕ

/-- Calculates the total number of rods in a RodTriangle -/
def totalRods (t : RodTriangle) : ℕ :=
  arithmeticSum t.firstRowRods t.rodIncrement t.rows

/-- Calculates the total number of connectors in a RodTriangle -/
def totalConnectors (t : RodTriangle) : ℕ :=
  triangularNumber (t.rows + 1)

/-- Calculates the total number of pieces (rods and connectors) in a RodTriangle -/
def totalPieces (t : RodTriangle) : ℕ :=
  totalRods t + totalConnectors t

/-- Theorem: The total number of pieces in a ten-row triangle is 231 -/
theorem ten_row_triangle_pieces :
  totalPieces { rows := 10, firstRowRods := 3, rodIncrement := 3 } = 231 := by
  sorry

end NUMINAMATH_CALUDE_ten_row_triangle_pieces_l3350_335096


namespace NUMINAMATH_CALUDE_mayoral_election_votes_l3350_335042

theorem mayoral_election_votes (z : ℕ) (hz : z = 25000) : ∃ x y : ℕ,
  y = z - (2 * z / 5) ∧
  x = y + (y / 2) ∧
  x = 22500 :=
by
  sorry

end NUMINAMATH_CALUDE_mayoral_election_votes_l3350_335042


namespace NUMINAMATH_CALUDE_biased_coin_heads_probability_l3350_335083

/-- The probability of getting heads on a single flip of a biased coin -/
theorem biased_coin_heads_probability (p : ℚ) (h : p = 3/4) : 1 - p = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_heads_probability_l3350_335083


namespace NUMINAMATH_CALUDE_restaurant_meals_count_l3350_335091

theorem restaurant_meals_count (kids_meals : ℕ) (adult_meals : ℕ) : 
  kids_meals = 8 → 
  2 * adult_meals = kids_meals → 
  kids_meals + adult_meals = 12 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_meals_count_l3350_335091


namespace NUMINAMATH_CALUDE_min_value_and_inequality_solution_l3350_335041

theorem min_value_and_inequality_solution :
  ∃ m : ℝ,
    (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 →
      (1 / a^3 + 1 / b^3 + 1 / c^3 + 27 * a * b * c) ≥ m) ∧
    (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
      (1 / a^3 + 1 / b^3 + 1 / c^3 + 27 * a * b * c) = m) ∧
    m = 18 ∧
    (∀ x : ℝ, |x + 1| - 2 * x < m ↔ x > -19/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_solution_l3350_335041


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l3350_335081

theorem opposite_of_negative_2023 : -((-2023 : ℚ)) = (2023 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l3350_335081


namespace NUMINAMATH_CALUDE_modern_literature_marks_l3350_335018

theorem modern_literature_marks
  (geography : ℕ) (history_gov : ℕ) (art : ℕ) (comp_sci : ℕ) (avg : ℚ) :
  geography = 56 →
  history_gov = 60 →
  art = 72 →
  comp_sci = 85 →
  avg = 70.6 →
  ∃ (modern_lit : ℕ),
    (geography + history_gov + art + comp_sci + modern_lit : ℚ) / 5 = avg ∧
    modern_lit = 80 := by
  sorry

end NUMINAMATH_CALUDE_modern_literature_marks_l3350_335018


namespace NUMINAMATH_CALUDE_basic_astrophysics_degrees_l3350_335038

/-- Represents the budget allocation for Megatech Corporation's research and development --/
structure BudgetAllocation where
  microphotonics : Float
  home_electronics : Float
  food_additives : Float
  genetically_modified_microorganisms : Float
  industrial_lubricants : Float

/-- Calculates the degrees in a circle for a given percentage --/
def percentageToDegrees (percentage : Float) : Float :=
  percentage * 360 / 100

/-- Theorem stating that the degrees for basic astrophysics research is 43.2 --/
theorem basic_astrophysics_degrees 
  (budget : BudgetAllocation)
  (h1 : budget.microphotonics = 12)
  (h2 : budget.home_electronics = 24)
  (h3 : budget.food_additives = 15)
  (h4 : budget.genetically_modified_microorganisms = 29)
  (h5 : budget.industrial_lubricants = 8)
  : percentageToDegrees (100 - (budget.microphotonics + budget.home_electronics + 
    budget.food_additives + budget.genetically_modified_microorganisms + 
    budget.industrial_lubricants)) = 43.2 := by
  sorry

end NUMINAMATH_CALUDE_basic_astrophysics_degrees_l3350_335038


namespace NUMINAMATH_CALUDE_seokjin_paper_count_l3350_335029

theorem seokjin_paper_count (jimin_count : ℕ) (difference : ℕ) 
  (h1 : jimin_count = 41)
  (h2 : difference = 1)
  (h3 : jimin_count = seokjin_count + difference) :
  seokjin_count = 40 := by
  sorry

end NUMINAMATH_CALUDE_seokjin_paper_count_l3350_335029


namespace NUMINAMATH_CALUDE_B_inverse_proof_l3350_335000

variable (A B : Matrix (Fin 2) (Fin 2) ℚ)

def A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![1, 2; 3, 4]

theorem B_inverse_proof :
  A⁻¹ = A_inv →
  B * A = 1 →
  B⁻¹ = !![(-2), 1; (3/2), (-1/2)] := by sorry

end NUMINAMATH_CALUDE_B_inverse_proof_l3350_335000


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l3350_335084

theorem x_range_for_inequality (m : ℝ) (hm : m ∈ Set.Icc 0 1) :
  {x : ℝ | m * x^2 - 2 * x - m ≥ 2} ⊆ Set.Iic (-1) := by
sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l3350_335084


namespace NUMINAMATH_CALUDE_square_difference_equals_product_l3350_335034

theorem square_difference_equals_product : (51 + 15)^2 - (51^2 + 15^2) = 1530 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_product_l3350_335034


namespace NUMINAMATH_CALUDE_function_properties_l3350_335069

open Real

theorem function_properties (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Set.Icc a b))
    (h_diff : DifferentiableOn ℝ f (Set.Icc a b)) (h_a_lt_b : a < b)
    (h_f'_a : deriv f a > 0) (h_f'_b : deriv f b < 0) :
  (∃ x₀ ∈ Set.Icc a b, f x₀ > f b) ∧
  (∃ x₀ ∈ Set.Icc a b, f a - f b = (deriv (deriv f)) x₀ * (a - b)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3350_335069


namespace NUMINAMATH_CALUDE_range_of_product_l3350_335061

theorem range_of_product (x y z w : ℝ) 
  (sum_zero : x + y + z + w = 0)
  (sum_seventh_power_zero : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_product_l3350_335061


namespace NUMINAMATH_CALUDE_units_digit_7_pow_1995_l3350_335075

/-- The units digit of 7^n -/
def units_digit_7_pow (n : ℕ) : ℕ := 7^n % 10

/-- The sequence of units digits of powers of 7 -/
def a : ℕ → ℕ := units_digit_7_pow

theorem units_digit_7_pow_1995 : a 1995 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_7_pow_1995_l3350_335075


namespace NUMINAMATH_CALUDE_unique_solution_l3350_335079

/-- Represents a 3-digit number with distinct digits -/
structure ThreeDigitNumber where
  f : Nat
  o : Nat
  g : Nat
  h_distinct : f ≠ o ∧ f ≠ g ∧ o ≠ g
  h_valid : f ≠ 0 ∧ f < 10 ∧ o < 10 ∧ g < 10

def value (n : ThreeDigitNumber) : Nat :=
  100 * n.f + 10 * n.o + n.g

theorem unique_solution (n : ThreeDigitNumber) :
  value n * (n.f + n.o + n.g) = value n →
  n.f = 1 ∧ n.o = 0 ∧ n.g = 0 ∧ n.f + n.o + n.g = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3350_335079


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l3350_335033

theorem smaller_number_in_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a / b = 3 / 8 → (a - 24) / (b - 24) = 4 / 9 → a = 72 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l3350_335033


namespace NUMINAMATH_CALUDE_complex_square_simplification_l3350_335032

theorem complex_square_simplification :
  (5 - 3 * Real.sqrt 2 * Complex.I) ^ 2 = 43 - 30 * Real.sqrt 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l3350_335032


namespace NUMINAMATH_CALUDE_sphere_radius_in_cone_l3350_335076

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Represents the configuration of spheres in the cone -/
structure SphereConfiguration where
  cone : Cone
  spheres : Fin 4 → Sphere
  bottomThreeTangent : Bool
  bottomThreeTouchBase : Bool
  bottomThreeTouchSide : Bool
  topTouchesOthers : Bool
  topTouchesSide : Bool
  topNotTouchBase : Bool

/-- The main theorem statement -/
theorem sphere_radius_in_cone (config : SphereConfiguration)
  (h_cone : config.cone = Cone.mk 7 15)
  (h_spheres_congruent : ∀ i j, (config.spheres i).radius = (config.spheres j).radius)
  (h_bottom_tangent : config.bottomThreeTangent = true)
  (h_bottom_base : config.bottomThreeTouchBase = true)
  (h_bottom_side : config.bottomThreeTouchSide = true)
  (h_top_others : config.topTouchesOthers = true)
  (h_top_side : config.topTouchesSide = true)
  (h_top_not_base : config.topNotTouchBase = true) :
  (config.spheres 0).radius = (162 - 108 * Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_in_cone_l3350_335076


namespace NUMINAMATH_CALUDE_division_of_decimals_l3350_335082

theorem division_of_decimals : (0.045 : ℝ) / 0.0075 = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_of_decimals_l3350_335082


namespace NUMINAMATH_CALUDE_l_plate_four_equal_parts_l3350_335053

/-- Represents an L-shaped plate -/
structure LPlate where
  width : ℝ
  height : ℝ
  isRightAngled : Bool

/-- Represents a cut on the L-shaped plate -/
inductive Cut
  | Vertical : ℝ → Cut  -- x-coordinate of the vertical cut
  | Horizontal : ℝ → Cut  -- y-coordinate of the horizontal cut

/-- Checks if a set of cuts divides an L-shaped plate into four equal parts -/
def dividesIntoFourEqualParts (plate : LPlate) (cuts : List Cut) : Prop :=
  sorry

/-- Theorem stating that an L-shaped plate can be divided into four equal L-shaped pieces -/
theorem l_plate_four_equal_parts (plate : LPlate) :
  ∃ (cuts : List Cut), dividesIntoFourEqualParts plate cuts :=
sorry

end NUMINAMATH_CALUDE_l_plate_four_equal_parts_l3350_335053


namespace NUMINAMATH_CALUDE_circle_tangency_l3350_335063

structure Circle where
  center : Point
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

def touches_internally (c1 c2 : Circle) (p : Point) : Prop := sorry

def center_on_circle (c1 c2 : Circle) : Prop := sorry

def common_chord_intersects (c1 c2 c3 : Circle) (a b : Point) : Prop := sorry

def line_intersects_circle (p1 p2 : Point) (c : Circle) (q : Point) : Prop := sorry

def is_tangent (c : Circle) (p1 p2 : Point) : Prop := sorry

theorem circle_tangency 
  (Ω Ω₁ Ω₂ : Circle) 
  (M N A B C D : Point) :
  touches_internally Ω₁ Ω M →
  touches_internally Ω₂ Ω N →
  center_on_circle Ω₂ Ω₁ →
  common_chord_intersects Ω₁ Ω₂ Ω A B →
  line_intersects_circle M A Ω₁ C →
  line_intersects_circle M B Ω₁ D →
  is_tangent Ω₂ C D := by
    sorry

end NUMINAMATH_CALUDE_circle_tangency_l3350_335063


namespace NUMINAMATH_CALUDE_prob_sum_nine_l3350_335037

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The sum we're looking for -/
def targetSum : ℕ := 9

/-- The set of all possible outcomes when throwing two dice -/
def allOutcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range numSides) (Finset.range numSides)

/-- The set of favorable outcomes (pairs that sum to targetSum) -/
def favorableOutcomes : Finset (ℕ × ℕ) :=
  allOutcomes.filter (fun p => p.1 + p.2 = targetSum)

/-- The probability of obtaining the target sum -/
def probability : ℚ :=
  (favorableOutcomes.card : ℚ) / (allOutcomes.card : ℚ)

theorem prob_sum_nine : probability = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_nine_l3350_335037


namespace NUMINAMATH_CALUDE_min_disks_needed_l3350_335040

def total_files : ℕ := 45
def disk_capacity : ℚ := 1.44

def file_size_1 : ℚ := 0.9
def file_count_1 : ℕ := 5

def file_size_2 : ℚ := 0.6
def file_count_2 : ℕ := 15

def file_size_3 : ℚ := 0.5
def file_count_3 : ℕ := total_files - file_count_1 - file_count_2

theorem min_disks_needed : 
  ∃ (n : ℕ), n = 20 ∧ 
  (∀ m : ℕ, m < n → 
    m * disk_capacity < 
      file_count_1 * file_size_1 + 
      file_count_2 * file_size_2 + 
      file_count_3 * file_size_3) ∧
  n * disk_capacity ≥ 
    file_count_1 * file_size_1 + 
    file_count_2 * file_size_2 + 
    file_count_3 * file_size_3 :=
by sorry

end NUMINAMATH_CALUDE_min_disks_needed_l3350_335040


namespace NUMINAMATH_CALUDE_root_product_value_l3350_335027

theorem root_product_value (m n : ℝ) : 
  m^2 - 2019*m - 1 = 0 → 
  n^2 - 2019*n - 1 = 0 → 
  (m^2 - 2019*m + 3) * (n^2 - 2019*n + 4) = 20 := by
sorry

end NUMINAMATH_CALUDE_root_product_value_l3350_335027


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_equality_condition_l3350_335036

theorem sum_of_fractions_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c :=
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_equality_condition_l3350_335036


namespace NUMINAMATH_CALUDE_chastity_money_left_l3350_335060

/-- The amount of money Chastity was left with after buying lollipops and gummies -/
def money_left (initial_amount : ℝ) (lollipop_price : ℝ) (lollipop_count : ℕ) 
                (gummy_price : ℝ) (gummy_count : ℕ) : ℝ :=
  initial_amount - (lollipop_price * lollipop_count + gummy_price * gummy_count)

/-- Theorem stating that Chastity was left with $5 after her candy purchase -/
theorem chastity_money_left : 
  money_left 15 1.5 4 2 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_chastity_money_left_l3350_335060


namespace NUMINAMATH_CALUDE_vector_operations_l3350_335011

/-- Given vectors in ℝ², prove that they are not collinear, find the cosine of the angle between them, and calculate the projection of one vector onto another. -/
theorem vector_operations (a b c : ℝ × ℝ) (h1 : a = (-1, 1)) (h2 : b = (4, 3)) (h3 : c = (5, -2)) :
  ¬ (∃ k : ℝ, a = k • b) ∧
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -Real.sqrt 2 / 10 ∧
  ((a.1 * c.1 + a.2 * c.2) / (a.1^2 + a.2^2)) • a = (7/2 * Real.sqrt 2) • (-1, 1) :=
by sorry

end NUMINAMATH_CALUDE_vector_operations_l3350_335011


namespace NUMINAMATH_CALUDE_square_value_proof_l3350_335072

theorem square_value_proof : ∃ (square : ℚ), 
  (13.5 / (11 + (2.25 / (1 - square))) - 1 / 7) * (7/6) = 1 ∧ square = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_square_value_proof_l3350_335072


namespace NUMINAMATH_CALUDE_word_problems_count_l3350_335087

theorem word_problems_count (total_questions : ℕ) (addition_subtraction_problems : ℕ) 
  (h1 : total_questions = 45)
  (h2 : addition_subtraction_problems = 28) :
  total_questions - addition_subtraction_problems = 17 := by
  sorry

end NUMINAMATH_CALUDE_word_problems_count_l3350_335087
