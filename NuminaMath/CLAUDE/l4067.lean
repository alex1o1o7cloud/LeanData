import Mathlib

namespace NUMINAMATH_CALUDE_add_fractions_with_same_denominator_l4067_406712

theorem add_fractions_with_same_denominator (a : ℝ) (h : a ≠ 0) :
  3 / a + 2 / a = 5 / a := by sorry

end NUMINAMATH_CALUDE_add_fractions_with_same_denominator_l4067_406712


namespace NUMINAMATH_CALUDE_quadratic_equation_one_solution_l4067_406707

theorem quadratic_equation_one_solution (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + x + 1 = 0) ↔ a = 0 ∨ a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_one_solution_l4067_406707


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l4067_406754

theorem quadratic_roots_relation (m n : ℝ) (r₁ r₂ : ℝ) (p q : ℝ) : 
  r₁^2 - 2*m*r₁ + n = 0 →
  r₂^2 - 2*m*r₂ + n = 0 →
  r₁^4 + p*r₁^4 + q = 0 →
  r₂^4 + p*r₂^4 + q = 0 →
  r₁ + r₂ = 2*m - 3 →
  p = -(2*m - 3)^4 + 4*n*(2*m - 3)^2 - 2*n^2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l4067_406754


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4067_406722

theorem quadratic_inequality_solution (b c : ℝ) : 
  (∀ x, x^2 + b*x + c > 0 ↔ -1 < x ∧ x < 2) → c + b = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4067_406722


namespace NUMINAMATH_CALUDE_henry_lawn_mowing_earnings_l4067_406799

/-- Henry's lawn mowing earnings problem -/
theorem henry_lawn_mowing_earnings 
  (earnings_per_lawn : ℕ) 
  (total_lawns : ℕ) 
  (forgotten_lawns : ℕ) 
  (h1 : earnings_per_lawn = 5)
  (h2 : total_lawns = 12)
  (h3 : forgotten_lawns = 7) :
  (total_lawns - forgotten_lawns) * earnings_per_lawn = 25 := by
  sorry

end NUMINAMATH_CALUDE_henry_lawn_mowing_earnings_l4067_406799


namespace NUMINAMATH_CALUDE_fixed_points_existence_l4067_406732

-- Define the fixed point F and line l
def F : ℝ × ℝ := (1, 0)
def l : ℝ → Prop := λ x => x = 4

-- Define the trajectory E
def E : ℝ × ℝ → Prop := λ p => (p.1^2 / 4) + (p.2^2 / 3) = 1

-- Define the distance ratio condition
def distance_ratio (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) / |P.1 - 4| = 1/2

-- Define point A
def A : ℝ × ℝ := (-2, 0)

-- Define the theorem
theorem fixed_points_existence :
  ∃ Q₁ Q₂ : ℝ × ℝ,
    Q₁.2 = 0 ∧ Q₂.2 = 0 ∧
    Q₁ ≠ Q₂ ∧
    (∀ B C M N : ℝ × ℝ,
      E B ∧ E C ∧
      (∃ m : ℝ, B.1 = m * B.2 + 1 ∧ C.1 = m * C.2 + 1) ∧
      (M.1 = 4 ∧ N.1 = 4) ∧
      (∃ t : ℝ, M.2 = t * (B.1 + 2) ∧ N.2 = t * (C.1 + 2)) →
      ((Q₁.1 - M.1) * (Q₁.1 - N.1) + (Q₁.2 - M.2) * (Q₁.2 - N.2) = 0 ∧
       (Q₂.1 - M.1) * (Q₂.1 - N.1) + (Q₂.2 - M.2) * (Q₂.2 - N.2) = 0)) ∧
    Q₁ = (1, 0) ∧ Q₂ = (7, 0) :=
by sorry

end NUMINAMATH_CALUDE_fixed_points_existence_l4067_406732


namespace NUMINAMATH_CALUDE_probability_multiple_four_l4067_406795

-- Define the types for the dice
def DodecahedralDie := Fin 12
def SixSidedDie := Fin 6

-- Define the probability space
def Ω := DodecahedralDie × SixSidedDie

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define the event that the product is a multiple of 4
def MultipleFour : Set Ω := {ω | 4 ∣ (ω.1.val + 1) * (ω.2.val + 1)}

-- Theorem statement
theorem probability_multiple_four : P MultipleFour = 3/8 := by sorry

end NUMINAMATH_CALUDE_probability_multiple_four_l4067_406795


namespace NUMINAMATH_CALUDE_factorization_equality_l4067_406705

theorem factorization_equality (x y : ℝ) : (x + y)^2 - 14*(x + y) + 49 = (x + y - 7)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l4067_406705


namespace NUMINAMATH_CALUDE_number_of_fractions_l4067_406704

/-- A function that determines if an expression is a fraction in the form a/b -/
def isFraction (expr : String) : Bool :=
  match expr with
  | "5/(a-x)" => true
  | "(m+n)/(mn)" => true
  | "5x^2/x" => true
  | _ => false

/-- The list of expressions given in the problem -/
def expressions : List String :=
  ["1/5(1-x)", "5/(a-x)", "4x/(π-3)", "(m+n)/(mn)", "(x^2-y^2)/2", "5x^2/x"]

/-- Theorem stating that the number of fractions in the given list is 3 -/
theorem number_of_fractions : 
  (expressions.filter isFraction).length = 3 := by sorry

end NUMINAMATH_CALUDE_number_of_fractions_l4067_406704


namespace NUMINAMATH_CALUDE_circle_and_triangle_properties_l4067_406736

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  (k - 1) * x - 2 * y + 5 - 3 * k = 0

-- Define the fixed point P
def point_P : ℝ × ℝ := (3, 1)

-- Define point A
def point_A : ℝ × ℝ := (4, 0)

-- Define the line on which the center of circle C lies
def center_line (x y : ℝ) : Prop :=
  x - 2 * y + 1 = 0

-- Define circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 14*x - 8*y + 40 = 0

-- Define point Q
def point_Q : ℝ × ℝ := (11, 7)

theorem circle_and_triangle_properties :
  ∀ k : ℝ,
  (∀ x y : ℝ, line_l k x y → (x = point_P.1 ∧ y = point_P.2)) →
  circle_C point_A.1 point_A.2 →
  circle_C point_P.1 point_P.2 →
  (∃ x y : ℝ, center_line x y ∧ (x - point_P.1)^2 + (y - point_P.2)^2 = (x - point_A.1)^2 + (y - point_A.2)^2) →
  (point_Q.1 - point_P.1)^2 + (point_Q.2 - point_P.2)^2 = 4 * ((point_P.1 - 7)^2 + (point_P.2 - 4)^2) →
  ∃ m : ℝ, (m = 5 ∨ m = 65/3) ∧
    ((point_P.1 - 0)^2 + (point_P.2 - m)^2 + (point_Q.1 - 0)^2 + (point_Q.2 - m)^2 =
     (point_Q.1 - point_P.1)^2 + (point_Q.2 - point_P.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_triangle_properties_l4067_406736


namespace NUMINAMATH_CALUDE_total_rounded_to_nearest_dollar_l4067_406720

def purchase1 : ℚ := 245/100
def purchase2 : ℚ := 358/100
def purchase3 : ℚ := 796/100

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  if x - x.floor < 1/2 then x.floor else x.ceil

theorem total_rounded_to_nearest_dollar :
  round_to_nearest_dollar (purchase1 + purchase2 + purchase3) = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_rounded_to_nearest_dollar_l4067_406720


namespace NUMINAMATH_CALUDE_shape_partition_count_l4067_406748

/-- Represents a cell in the shape -/
structure Cell :=
  (x : ℕ) (y : ℕ)

/-- Represents a rectangle in the partition -/
inductive Rectangle
  | small : Cell → Rectangle  -- 1×1 square
  | large : Cell → Cell → Rectangle  -- 1×2 rectangle

/-- A partition of the shape -/
def Partition := List Rectangle

/-- The shape with 17 cells -/
def shape : List Cell := sorry

/-- Check if a partition is valid for the given shape -/
def is_valid_partition (p : Partition) (s : List Cell) : Prop := sorry

/-- Count the number of distinct valid partitions -/
def count_valid_partitions (s : List Cell) : ℕ := sorry

/-- The main theorem -/
theorem shape_partition_count :
  count_valid_partitions shape = 10 := by sorry

end NUMINAMATH_CALUDE_shape_partition_count_l4067_406748


namespace NUMINAMATH_CALUDE_inequality_proof_l4067_406785

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  a^2 + b^2 + c^2 + 3 ≥ 1/a + 1/b + 1/c + a + b + c :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4067_406785


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4067_406749

theorem complex_equation_solution (a b : ℝ) :
  (Complex.mk a b) * (Complex.mk 2 (-1)) = Complex.I →
  a + b = 1/5 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4067_406749


namespace NUMINAMATH_CALUDE_expression_value_l4067_406711

theorem expression_value : 
  (10^2005 + 10^2007) / (10^2006 + 10^2006) = 101 / 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4067_406711


namespace NUMINAMATH_CALUDE_sum_of_base3_digits_345_l4067_406784

/-- Converts a natural number to its base-3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 3) :: aux (m / 3)
  aux n

/-- Sums the digits in a list of natural numbers -/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.foldl (·+·) 0

theorem sum_of_base3_digits_345 :
  sumDigits (toBase3 345) = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_base3_digits_345_l4067_406784


namespace NUMINAMATH_CALUDE_tan_beta_value_l4067_406798

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 2) 
  (h2 : Real.tan (α + β) = -1) : 
  Real.tan β = 3 := by sorry

end NUMINAMATH_CALUDE_tan_beta_value_l4067_406798


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_theorem_l4067_406714

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 8

-- Define the line
def Line (k m x y : ℝ) : Prop := y = k*x + m

-- Define the intersection points
def Intersects (k m : ℝ) (P Q : ℝ × ℝ) : Prop :=
  Line k m P.1 P.2 ∧ Line k m Q.1 Q.2 ∧
  Ellipse P.1 P.2 ∧ Ellipse Q.1 Q.2

-- Define the x-axis and y-axis intersection points
def AxisIntersections (k m : ℝ) (C D : ℝ × ℝ) : Prop :=
  C = (-m/k, 0) ∧ D = (0, m)

-- Define the trisection condition
def Trisection (O P Q C D : ℝ × ℝ) : Prop :=
  (D.1 - O.1, D.2 - O.2) = (1/3 * (P.1 - O.1), 1/3 * (P.2 - O.2)) + (2/3 * (Q.1 - O.1), 2/3 * (Q.2 - O.2)) ∧
  (C.1 - O.1, C.2 - O.2) = (1/3 * (Q.1 - O.1), 1/3 * (Q.2 - O.2)) + (2/3 * (P.1 - O.1), 2/3 * (P.2 - O.2))

theorem ellipse_line_intersection_theorem :
  ∃ (k m : ℝ) (P Q C D : ℝ × ℝ),
    Intersects k m P Q ∧
    AxisIntersections k m C D ∧
    Trisection (0, 0) P Q C D :=
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_theorem_l4067_406714


namespace NUMINAMATH_CALUDE_grass_seed_bags_for_park_lot_l4067_406760

/-- Calculates the number of grass seed bags needed for a rectangular lot with a concrete section --/
def grassSeedBags (lotLength lotWidth concreteSize seedCoverage : ℕ) : ℕ :=
  let totalArea := lotLength * lotWidth
  let concreteArea := concreteSize * concreteSize
  let grassyArea := totalArea - concreteArea
  (grassyArea + seedCoverage - 1) / seedCoverage

/-- Theorem stating that 100 bags of grass seeds are needed for the given lot specifications --/
theorem grass_seed_bags_for_park_lot : 
  grassSeedBags 120 60 40 56 = 100 := by
  sorry

#eval grassSeedBags 120 60 40 56

end NUMINAMATH_CALUDE_grass_seed_bags_for_park_lot_l4067_406760


namespace NUMINAMATH_CALUDE_circle_condition_l4067_406702

/-- The equation of a potential circle -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y + a = 0

/-- Definition of a circle in 2D space -/
def is_circle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

theorem circle_condition (a : ℝ) :
  (∀ x y, ∃ center radius, circle_equation x y a → is_circle center radius x y) ↔ a < 2 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l4067_406702


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4067_406770

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  ∀ a b : ℝ, a > 0 → b > 0 → a * b = 4 → (1 / x + 1 / y) ≤ (1 / a + 1 / b) ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = 4 ∧ 1 / x + 1 / y = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4067_406770


namespace NUMINAMATH_CALUDE_inequality_proof_l4067_406756

theorem inequality_proof (a : ℝ) (h : a > 1) : (1/2 : ℝ) + (1 / Real.log a) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4067_406756


namespace NUMINAMATH_CALUDE_greatest_3digit_base7_divisible_by_7_l4067_406747

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (a b c : Nat) : Nat :=
  a * 7^2 + b * 7 + c

/-- Checks if a number is a valid 3-digit base 7 number --/
def isValidBase7 (a b c : Nat) : Prop :=
  a > 0 ∧ a < 7 ∧ b < 7 ∧ c < 7

/-- The proposed solution in base 7 --/
def solution : (Nat × Nat × Nat) := (6, 6, 0)

theorem greatest_3digit_base7_divisible_by_7 :
  let (a, b, c) := solution
  isValidBase7 a b c ∧
  base7ToBase10 a b c % 7 = 0 ∧
  ∀ x y z, isValidBase7 x y z → 
    base7ToBase10 x y z % 7 = 0 → 
    base7ToBase10 x y z ≤ base7ToBase10 a b c :=
by sorry

end NUMINAMATH_CALUDE_greatest_3digit_base7_divisible_by_7_l4067_406747


namespace NUMINAMATH_CALUDE_smallest_with_20_divisors_l4067_406773

/-- The number of positive divisors of a positive integer n -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- n has exactly 20 positive divisors -/
def has_20_divisors (n : ℕ+) : Prop := num_divisors n = 20

theorem smallest_with_20_divisors : 
  has_20_divisors 432 ∧ ∀ m : ℕ+, m < 432 → ¬(has_20_divisors m) := by sorry

end NUMINAMATH_CALUDE_smallest_with_20_divisors_l4067_406773


namespace NUMINAMATH_CALUDE_four_correct_propositions_l4067_406753

theorem four_correct_propositions :
  (∀ a b : ℝ, |a + b| - 2 * |a| ≤ |a - b|) ∧
  (∀ a b : ℝ, |a - b| < 1 → |a| < |b| + 1) ∧
  (∀ x y : ℝ, |x| < 2 ∧ |y| > 3 → |x / y| < 2 / 3) ∧
  (∀ A B : ℝ, A ≠ 0 ∧ B ≠ 0 → Real.log ((|A| + |B|) / 2) ≥ (1 / 2) * (Real.log |A| + Real.log |B|)) :=
sorry

end NUMINAMATH_CALUDE_four_correct_propositions_l4067_406753


namespace NUMINAMATH_CALUDE_population_reaches_capacity_in_90_years_l4067_406718

def usable_land : ℕ := 32500
def acres_per_person : ℕ := 2
def initial_population : ℕ := 500
def growth_factor : ℕ := 4
def growth_period : ℕ := 30

def max_capacity : ℕ := usable_land / acres_per_person

def population_after_years (years : ℕ) : ℕ :=
  initial_population * (growth_factor ^ (years / growth_period))

theorem population_reaches_capacity_in_90_years :
  population_after_years 90 ≥ max_capacity ∧
  population_after_years 60 < max_capacity :=
sorry

end NUMINAMATH_CALUDE_population_reaches_capacity_in_90_years_l4067_406718


namespace NUMINAMATH_CALUDE_functional_equation_solution_l4067_406741

theorem functional_equation_solution (f : ℚ → ℚ) :
  (∀ x y : ℚ, f x + f y = 2 * f ((x + y) / 2)) →
  ∃ c : ℚ, ∀ x : ℚ, f x = c * x :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l4067_406741


namespace NUMINAMATH_CALUDE_g_2010_equals_342_l4067_406757

/-- The function g satisfies the given property for positive integers -/
def g_property (g : ℕ+ → ℕ) : Prop :=
  ∀ (x y m : ℕ+), x + y = 2^(m : ℕ) → g x + g y = 3 * m^2

/-- The main theorem stating that g(2010) = 342 -/
theorem g_2010_equals_342 (g : ℕ+ → ℕ) (h : g_property g) : g 2010 = 342 := by
  sorry

end NUMINAMATH_CALUDE_g_2010_equals_342_l4067_406757


namespace NUMINAMATH_CALUDE_quadratic_root_l4067_406738

theorem quadratic_root (p q r : ℝ) (h : p ≠ 0 ∧ q ≠ r) :
  let f : ℝ → ℝ := λ x ↦ p * (q - r) * x^2 + q * (r - p) * x + r * (p - q)
  (f 1 = 0) →
  ∃ x : ℝ, x ≠ 1 ∧ f x = 0 ∧ x = r * (p - q) / (p * (q - r)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_l4067_406738


namespace NUMINAMATH_CALUDE_election_outcomes_count_l4067_406735

def boys : ℕ := 28
def girls : ℕ := 22
def total_students : ℕ := boys + girls
def committee_size : ℕ := 5

theorem election_outcomes_count :
  (Nat.descFactorial total_students committee_size) -
  (Nat.descFactorial boys committee_size) -
  (Nat.descFactorial girls committee_size) = 239297520 :=
by sorry

end NUMINAMATH_CALUDE_election_outcomes_count_l4067_406735


namespace NUMINAMATH_CALUDE_maurice_earnings_l4067_406796

/-- Calculates the total earnings for a given number of tasks -/
def totalEarnings (tasksCompleted : ℕ) (earnPerTask : ℕ) (bonusPerTenTasks : ℕ) : ℕ :=
  let regularEarnings := tasksCompleted * earnPerTask
  let bonusEarnings := (tasksCompleted / 10) * bonusPerTenTasks
  regularEarnings + bonusEarnings

/-- Proves that Maurice's earnings for 30 tasks is $78 -/
theorem maurice_earnings : totalEarnings 30 2 6 = 78 := by
  sorry

end NUMINAMATH_CALUDE_maurice_earnings_l4067_406796


namespace NUMINAMATH_CALUDE_stratified_sampling_total_l4067_406771

/-- Calculates the total number of students sampled using stratified sampling -/
def totalSampleSize (firstGradeTotal : ℕ) (secondGradeTotal : ℕ) (thirdGradeTotal : ℕ) (firstGradeSample : ℕ) : ℕ :=
  let totalStudents := firstGradeTotal + secondGradeTotal + thirdGradeTotal
  (firstGradeSample * totalStudents) / firstGradeTotal

theorem stratified_sampling_total (firstGradeTotal : ℕ) (secondGradeTotal : ℕ) (thirdGradeTotal : ℕ) (firstGradeSample : ℕ)
    (h1 : firstGradeTotal = 600)
    (h2 : secondGradeTotal = 500)
    (h3 : thirdGradeTotal = 400)
    (h4 : firstGradeSample = 30) :
    totalSampleSize firstGradeTotal secondGradeTotal thirdGradeTotal firstGradeSample = 75 := by
  sorry

#eval totalSampleSize 600 500 400 30

end NUMINAMATH_CALUDE_stratified_sampling_total_l4067_406771


namespace NUMINAMATH_CALUDE_point_on_y_axis_l4067_406710

/-- A point on the y-axis has an x-coordinate of 0 -/
axiom y_axis_x_zero (x y : ℝ) : (x, y) ∈ Set.range (λ t : ℝ => (0, t)) ↔ x = 0

/-- The point A with coordinates (2-a, -3a+1) lies on the y-axis -/
def A_on_y_axis (a : ℝ) : Prop := (2 - a, -3 * a + 1) ∈ Set.range (λ t : ℝ => (0, t))

theorem point_on_y_axis (a : ℝ) (h : A_on_y_axis a) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l4067_406710


namespace NUMINAMATH_CALUDE_whiteboard_washing_time_l4067_406779

/-- If four kids can wash three whiteboards in 20 minutes, then one kid can wash six whiteboards in 160 minutes. -/
theorem whiteboard_washing_time 
  (time_four_kids : ℕ) 
  (num_whiteboards_four_kids : ℕ) 
  (num_kids : ℕ) 
  (num_whiteboards_one_kid : ℕ) :
  time_four_kids = 20 ∧ 
  num_whiteboards_four_kids = 3 ∧ 
  num_kids = 4 ∧ 
  num_whiteboards_one_kid = 6 →
  (time_four_kids * num_kids * num_whiteboards_one_kid) / num_whiteboards_four_kids = 160 := by
  sorry

#check whiteboard_washing_time

end NUMINAMATH_CALUDE_whiteboard_washing_time_l4067_406779


namespace NUMINAMATH_CALUDE_max_consecutive_integers_l4067_406758

def consecutive_sum (start : ℕ) (n : ℕ) : ℕ :=
  n * (2 * start + n - 1) / 2

def is_valid_sequence (start : ℕ) (n : ℕ) : Prop :=
  consecutive_sum start n = 2014 ∧ start > 0

theorem max_consecutive_integers :
  (∃ (start : ℕ), is_valid_sequence start 53) ∧
  (∀ (m : ℕ) (start : ℕ), m > 53 → ¬ is_valid_sequence start m) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_l4067_406758


namespace NUMINAMATH_CALUDE_equation_solution_l4067_406713

theorem equation_solution : ∃ x : ℝ, (24 - 4 = 3 + x) ∧ (x = 17) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4067_406713


namespace NUMINAMATH_CALUDE_sum_of_digits_congruence_l4067_406729

def S (n : ℕ+) : ℕ := sorry

theorem sum_of_digits_congruence (n : ℕ+) (h : S n = 29) : 
  S (n + 1) % 9 = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_congruence_l4067_406729


namespace NUMINAMATH_CALUDE_max_seated_people_is_14_l4067_406726

/-- Represents the seating arrangement in the break room --/
structure BreakRoom where
  totalTables : Nat
  maxSeatsPerTable : Nat
  maxSeatsWithDistancing : Nat
  occupiedTables : Nat
  currentlySeated : Nat
  availableChairs : Nat

/-- The maximum number of people that can be seated in the break room --/
def maxSeatedPeople (room : BreakRoom) : Nat :=
  min (room.currentlySeated + (room.totalTables - room.occupiedTables) * room.maxSeatsWithDistancing)
      room.availableChairs

/-- Theorem stating the maximum number of people that can be seated --/
theorem max_seated_people_is_14 (room : BreakRoom) 
    (h1 : room.totalTables = 7)
    (h2 : room.maxSeatsPerTable = 6)
    (h3 : room.maxSeatsWithDistancing = 3)
    (h4 : room.occupiedTables = 4)
    (h5 : room.currentlySeated = 7)
    (h6 : room.availableChairs = 14) :
    maxSeatedPeople room = 14 := by
  sorry

#eval maxSeatedPeople { totalTables := 7, maxSeatsPerTable := 6, maxSeatsWithDistancing := 3, 
                        occupiedTables := 4, currentlySeated := 7, availableChairs := 14 }

end NUMINAMATH_CALUDE_max_seated_people_is_14_l4067_406726


namespace NUMINAMATH_CALUDE_floor_2a_eq_floor_a_plus_floor_a_half_l4067_406723

theorem floor_2a_eq_floor_a_plus_floor_a_half (a : ℝ) (h : a > 0) :
  ⌊2 * a⌋ = ⌊a⌋ + ⌊a + 1/2⌋ := by sorry

end NUMINAMATH_CALUDE_floor_2a_eq_floor_a_plus_floor_a_half_l4067_406723


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_8_l4067_406700

/-- A geometric sequence with its sum of terms -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_geometric : ∀ n, a (n + 1) = a n * (a 1)⁻¹ * a 2
  sum_formula : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - a 2 / a 1)

/-- The main theorem -/
theorem geometric_sequence_sum_8 (seq : GeometricSequence) 
    (h2 : seq.S 2 = 3)
    (h4 : seq.S 4 = 15) :
  seq.S 8 = 255 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_8_l4067_406700


namespace NUMINAMATH_CALUDE_percentage_increase_lines_l4067_406737

theorem percentage_increase_lines (initial : ℕ) (final : ℕ) (increase_percent : ℚ) : 
  initial = 500 →
  final = 800 →
  increase_percent = 60 →
  (final - initial : ℚ) / initial * 100 = increase_percent :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_lines_l4067_406737


namespace NUMINAMATH_CALUDE_jordans_income_l4067_406724

/-- Represents the state income tax calculation and Jordan's specific case -/
theorem jordans_income (q : ℝ) : 
  ∃ (I : ℝ),
    I > 35000 ∧
    0.01 * q * 35000 + 0.01 * (q + 3) * (I - 35000) = (0.01 * q + 0.004) * I ∧
    I = 40000 := by
  sorry

end NUMINAMATH_CALUDE_jordans_income_l4067_406724


namespace NUMINAMATH_CALUDE_delta_max_ratio_l4067_406774

def charlie_day1_score : ℕ := 200
def charlie_day1_attempted : ℕ := 400
def charlie_day2_score : ℕ := 160
def charlie_day2_attempted : ℕ := 200
def total_points_attempted : ℕ := 600

def charlie_day1_ratio : ℚ := charlie_day1_score / charlie_day1_attempted
def charlie_day2_ratio : ℚ := charlie_day2_score / charlie_day2_attempted
def charlie_total_ratio : ℚ := (charlie_day1_score + charlie_day2_score) / total_points_attempted

theorem delta_max_ratio (delta_day1_score delta_day1_attempted delta_day2_score delta_day2_attempted : ℕ) :
  delta_day1_attempted + delta_day2_attempted = total_points_attempted →
  delta_day1_attempted ≠ charlie_day1_attempted →
  delta_day1_score > 0 →
  delta_day2_score > 0 →
  (delta_day1_score : ℚ) / delta_day1_attempted < charlie_day1_ratio →
  (delta_day2_score : ℚ) / delta_day2_attempted < charlie_day2_ratio →
  (delta_day1_score + delta_day2_score : ℚ) / total_points_attempted ≤ 479 / 600 :=
by sorry

end NUMINAMATH_CALUDE_delta_max_ratio_l4067_406774


namespace NUMINAMATH_CALUDE_sum_of_coefficients_eq_value_at_one_l4067_406727

/-- The polynomial for which we want to calculate the sum of coefficients -/
def p (x : ℝ) : ℝ := 3 * (x^8 - 2*x^5 + x^3 - 7) - 5 * (x^6 + 3*x^2 - 6) + 2 * (x^4 - 5)

/-- The sum of coefficients of a polynomial is equal to the value of the polynomial at x = 1 -/
theorem sum_of_coefficients_eq_value_at_one :
  p 1 = -19 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_eq_value_at_one_l4067_406727


namespace NUMINAMATH_CALUDE_y₁_y₂_friendly_l4067_406715

/-- Two functions are friendly if their difference is between -1 and 1 for all x in (0,1) -/
def friendly (f g : ℝ → ℝ) : Prop :=
  ∀ x, 0 < x → x < 1 → -1 < f x - g x ∧ f x - g x < 1

/-- The function y₁(x) = x² - 1 -/
def y₁ (x : ℝ) : ℝ := x^2 - 1

/-- The function y₂(x) = 2x - 1 -/
def y₂ (x : ℝ) : ℝ := 2*x - 1

/-- Theorem: y₁ and y₂ are friendly functions -/
theorem y₁_y₂_friendly : friendly y₁ y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_y₂_friendly_l4067_406715


namespace NUMINAMATH_CALUDE_three_speakers_from_different_companies_l4067_406745

/-- The number of companies -/
def num_companies : ℕ := 5

/-- The number of representatives for Company A -/
def company_a_reps : ℕ := 2

/-- The number of representatives for each of the other companies -/
def other_company_reps : ℕ := 1

/-- The number of speakers at the meeting -/
def num_speakers : ℕ := 3

/-- The number of ways to select 3 speakers from 3 different companies -/
def num_ways : ℕ := 16

theorem three_speakers_from_different_companies :
  let total_reps := company_a_reps + (num_companies - 1) * other_company_reps
  (Nat.choose total_reps num_speakers) = num_ways := by sorry

end NUMINAMATH_CALUDE_three_speakers_from_different_companies_l4067_406745


namespace NUMINAMATH_CALUDE_distance_between_vertices_l4067_406701

-- Define the equation
def equation (x y : ℝ) : Prop := Real.sqrt (x^2 + y^2) + |y - 2| = 4

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = 3 - (1/12) * x^2
def parabola2 (x y : ℝ) : Prop := y = (1/4) * x^2 - 1

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem distance_between_vertices : 
  ∃ (x1 y1 x2 y2 : ℝ), 
    parabola1 x1 y1 ∧ 
    parabola2 x2 y2 ∧ 
    (x1, y1) = vertex1 ∧ 
    (x2, y2) = vertex2 ∧ 
    Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l4067_406701


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l4067_406708

/-- Given two points A and B as the endpoints of a circle's diameter, 
    prove that the equation of the circle is (x - 1)² + (y - 2)² = 25 -/
theorem circle_equation_from_diameter (A B : ℝ × ℝ) : 
  A = (-3, -1) → B = (5, 5) → 
  ∀ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 25 ↔ 
    ((x - (-3))^2 + (y - (-1))^2 = ((5 - (-3))^2 + (5 - (-1))^2) / 4 ∧ 
     (x - 5)^2 + (y - 5)^2 = ((5 - (-3))^2 + (5 - (-1))^2) / 4) :=
by sorry


end NUMINAMATH_CALUDE_circle_equation_from_diameter_l4067_406708


namespace NUMINAMATH_CALUDE_johns_remaining_money_l4067_406768

/-- Calculates the remaining money after John's purchases -/
def remaining_money (initial_amount : ℚ) : ℚ :=
  let after_snacks := initial_amount * (1 - 1/5)
  let after_necessities := after_snacks * (1 - 3/4)
  after_necessities * (1 - 1/4)

/-- Theorem stating that John's remaining money is $3 -/
theorem johns_remaining_money :
  remaining_money 20 = 3 := by sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l4067_406768


namespace NUMINAMATH_CALUDE_f_minimum_value_l4067_406791

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x^2 + 1/(x^2 + 1/x^2)

theorem f_minimum_value (x : ℝ) (h : x > 0) : 
  f x ≥ 2.5 ∧ f 1 = 2.5 := by sorry

end NUMINAMATH_CALUDE_f_minimum_value_l4067_406791


namespace NUMINAMATH_CALUDE_women_picnic_attendance_l4067_406766

/-- Represents the percentage of employees in a company -/
structure CompanyPercentage where
  total : Real
  men : Real
  women : Real
  menAttended : Real
  womenAttended : Real
  totalAttended : Real

/-- Conditions for the company picnic attendance problem -/
def picnicConditions (c : CompanyPercentage) : Prop :=
  c.total = 100 ∧
  c.men = 50 ∧
  c.women = 50 ∧
  c.menAttended = 20 * c.men / 100 ∧
  c.totalAttended = 30.000000000000004 ∧
  c.womenAttended = c.totalAttended - c.menAttended

/-- Theorem stating that 40% of women attended the picnic -/
theorem women_picnic_attendance (c : CompanyPercentage) 
  (h : picnicConditions c) : c.womenAttended / c.women * 100 = 40 := by
  sorry


end NUMINAMATH_CALUDE_women_picnic_attendance_l4067_406766


namespace NUMINAMATH_CALUDE_unique_solution_l4067_406762

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x^2*y + x*y^2 - 2*x - 2*y + 10 = 0
def equation2 (x y : ℝ) : Prop := x^3*y - x*y^3 - 2*x^2 + 2*y^2 - 30 = 0

-- State the theorem
theorem unique_solution :
  ∃! p : ℝ × ℝ, equation1 p.1 p.2 ∧ equation2 p.1 p.2 ∧ p = (-4, -1) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l4067_406762


namespace NUMINAMATH_CALUDE_common_difference_from_terms_l4067_406742

/-- An arithmetic sequence with given terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The common difference of an arithmetic sequence -/
def commonDifference (seq : ArithmeticSequence) : ℝ :=
  seq.a 1 - seq.a 0

theorem common_difference_from_terms
  (seq : ArithmeticSequence)
  (h5 : seq.a 5 = 10)
  (h12 : seq.a 12 = 31) :
  commonDifference seq = 3 := by
  sorry


end NUMINAMATH_CALUDE_common_difference_from_terms_l4067_406742


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l4067_406751

theorem cubic_equation_solutions :
  let f : ℝ → ℝ := λ x => x^3 - 13*x - 12
  (f 4 = 0) ∧ (f (-1) = 0) ∧ (f (-3) = 0) ∧
  (∀ x : ℝ, f x = 0 → (x = 4 ∨ x = -1 ∨ x = -3)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l4067_406751


namespace NUMINAMATH_CALUDE_factorial_square_root_simplification_l4067_406772

theorem factorial_square_root_simplification :
  Real.sqrt ((4 * 3 * 2 * 1) * (4 * 3 * 2 * 1) + 4) = 2 * Real.sqrt 145 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_simplification_l4067_406772


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l4067_406775

theorem fraction_equation_solution : 
  ∃ x : ℝ, (1 / (x - 1) = 2 / (1 - x) + 1) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l4067_406775


namespace NUMINAMATH_CALUDE_inequality_proof_l4067_406728

theorem inequality_proof (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 0) : c * a < c * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4067_406728


namespace NUMINAMATH_CALUDE_x_coordinate_C_l4067_406725

/-- Parabola function -/
def parabola (x : ℝ) : ℝ := x^2

/-- Triangle ABC with vertices on parabola y = x^2 -/
structure TriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_A : A.2 = parabola A.1
  h_B : B.2 = parabola B.1
  h_C : C.2 = parabola C.1
  h_A_origin : A = (0, 0)
  h_B_coords : B = (-3, 9)
  h_C_positive : C.1 > 0
  h_BC_parallel : B.2 = C.2
  h_area : (1/2) * |C.1 + 3| * C.2 = 45

/-- The x-coordinate of vertex C is 7 -/
theorem x_coordinate_C (t : TriangleABC) : t.C.1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_coordinate_C_l4067_406725


namespace NUMINAMATH_CALUDE_rebus_solution_exists_l4067_406782

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def rebus_equation (h1 h2 h3 ch1 ch2 : ℕ) : Prop :=
  10 * h1 + h2 + 4000 + 400 * h3 + ch1 * 10 + ch2 = 4000 + 100 * h1 + 10 * h2 + h3

theorem rebus_solution_exists :
  ∃ (h1 h2 h3 ch1 ch2 : ℕ),
    is_odd h1 ∧ is_odd h2 ∧ is_odd h3 ∧
    is_even ch1 ∧ is_even ch2 ∧
    rebus_equation h1 h2 h3 ch1 ch2 ∧
    h1 = 5 ∧ h2 = 5 ∧ h3 = 5 :=
by sorry

end NUMINAMATH_CALUDE_rebus_solution_exists_l4067_406782


namespace NUMINAMATH_CALUDE_swim_club_members_l4067_406763

theorem swim_club_members :
  ∀ (total_members : ℕ) 
    (passed_test : ℕ) 
    (not_passed_with_course : ℕ) 
    (not_passed_without_course : ℕ),
  passed_test = (30 * total_members) / 100 →
  not_passed_with_course = 5 →
  not_passed_without_course = 30 →
  total_members = passed_test + not_passed_with_course + not_passed_without_course →
  total_members = 50 := by
sorry

end NUMINAMATH_CALUDE_swim_club_members_l4067_406763


namespace NUMINAMATH_CALUDE_work_completion_time_l4067_406780

theorem work_completion_time (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  (1 / x = 1 / 15) →
  (1 / x + 1 / y = 1 / 10) →
  y = 30 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l4067_406780


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l4067_406764

theorem quadratic_expression_value (x : ℝ) (h : x^2 + 3*x - 5 = 0) : 2*x^2 + 6*x - 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l4067_406764


namespace NUMINAMATH_CALUDE_triangle_problem_l4067_406719

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : Real.sqrt 3 * t.b * Real.sin t.A = t.a * Real.cos t.B)
  (h2 : t.b = 3)
  (h3 : Real.sin t.C = Real.sqrt 3 * Real.sin t.A) :
  t.B = π / 6 ∧ t.a = 3 ∧ t.c = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l4067_406719


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4067_406721

theorem inequality_solution_set (x : ℝ) : x^2 + 3 < 4*x ↔ 1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4067_406721


namespace NUMINAMATH_CALUDE_no_cube_root_exists_l4067_406744

theorem no_cube_root_exists (n : ℤ) : ¬ ∃ k : ℤ, k^3 = 3*n^2 + 3*n + 7 := by
  sorry

end NUMINAMATH_CALUDE_no_cube_root_exists_l4067_406744


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l4067_406793

theorem quadratic_form_minimum (x y : ℝ) : 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10 ≥ 14/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l4067_406793


namespace NUMINAMATH_CALUDE_austin_bicycle_weeks_l4067_406703

/-- The number of weeks Austin needs to work to buy a bicycle -/
def weeks_to_buy_bicycle (hourly_rate : ℚ) (monday_hours : ℚ) (wednesday_hours : ℚ) (friday_hours : ℚ) (bicycle_cost : ℚ) : ℚ :=
  bicycle_cost / (hourly_rate * (monday_hours + wednesday_hours + friday_hours))

/-- Proof that Austin needs 6 weeks to buy the bicycle -/
theorem austin_bicycle_weeks : 
  weeks_to_buy_bicycle 5 2 1 3 180 = 6 := by
  sorry

end NUMINAMATH_CALUDE_austin_bicycle_weeks_l4067_406703


namespace NUMINAMATH_CALUDE_solve_system_l4067_406740

theorem solve_system (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 2 / x) :
  y = (5 + Real.sqrt 41) / 4 ∨ y = (5 - Real.sqrt 41) / 4 :=
by sorry

end NUMINAMATH_CALUDE_solve_system_l4067_406740


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l4067_406769

open Set

-- Define the universal set U as the real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem intersection_complement_theorem :
  M ∩ (U \ N) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l4067_406769


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l4067_406765

/-- Proves that a boat traveling 12 km downstream in 2 hours and 12 km upstream in 3 hours has a speed of 5 km/h in still water. -/
theorem boat_speed_in_still_water (downstream_distance : ℝ) (upstream_distance : ℝ)
  (downstream_time : ℝ) (upstream_time : ℝ) (h1 : downstream_distance = 12)
  (h2 : upstream_distance = 12) (h3 : downstream_time = 2) (h4 : upstream_time = 3) :
  ∃ (boat_speed : ℝ) (stream_speed : ℝ),
    boat_speed = 5 ∧
    downstream_distance / downstream_time = boat_speed + stream_speed ∧
    upstream_distance / upstream_time = boat_speed - stream_speed := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l4067_406765


namespace NUMINAMATH_CALUDE_cyclic_inequality_l4067_406755

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y * z) + y / (z * x) + z / (x * y) ≥ 1 / x + 1 / y + 1 / z := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l4067_406755


namespace NUMINAMATH_CALUDE_sequence_formula_l4067_406717

def sequence_a (n : ℕ) : ℝ := 2 * n - 1

theorem sequence_formula :
  (sequence_a 1 = 1) ∧
  (∀ n : ℕ, sequence_a n - sequence_a (n + 1) + 2 = 0) →
  ∀ n : ℕ, sequence_a n = 2 * n - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l4067_406717


namespace NUMINAMATH_CALUDE_first_discount_percentage_l4067_406730

theorem first_discount_percentage
  (original_price : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ)
  (h1 : original_price = 149.99999999999997)
  (h2 : final_price = 108)
  (h3 : second_discount = 0.2)
  : (original_price - (final_price / (1 - second_discount))) / original_price = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l4067_406730


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_12321_l4067_406739

theorem largest_prime_factor_of_12321 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 12321 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 12321 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_12321_l4067_406739


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l4067_406778

theorem polynomial_identity_sum_of_squares : 
  ∀ (p q r s t u : ℤ), 
  (∀ x : ℝ, 1728 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 23456 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l4067_406778


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l4067_406734

theorem inscribed_circle_radius : ∃ (r : ℝ), 
  (1 / r = 1 / 6 + 1 / 10 + 1 / 15 + 3 * Real.sqrt (1 / (6 * 10) + 1 / (6 * 15) + 1 / (10 * 15))) ∧
  r = 30 / (10 * Real.sqrt 26 + 3) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l4067_406734


namespace NUMINAMATH_CALUDE_max_production_theorem_l4067_406716

/-- Represents a clothing factory -/
structure Factory where
  production_per_month : ℕ
  top_time_ratio : ℕ
  pant_time_ratio : ℕ

/-- Calculates the maximum number of sets two factories can produce in a month -/
def max_production (factory_a factory_b : Factory) : ℕ :=
  sorry

/-- Theorem stating the maximum production of two specific factories -/
theorem max_production_theorem :
  let factory_a : Factory := ⟨2700, 2, 1⟩
  let factory_b : Factory := ⟨3600, 3, 2⟩
  max_production factory_a factory_b = 6700 := by
  sorry

end NUMINAMATH_CALUDE_max_production_theorem_l4067_406716


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l4067_406787

theorem rectangle_area_perimeter_relation (x : ℝ) : 
  let length : ℝ := 4 * x
  let width : ℝ := x + 8
  let area : ℝ := length * width
  let perimeter : ℝ := 2 * (length + width)
  (area = 2 * perimeter) → (x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l4067_406787


namespace NUMINAMATH_CALUDE_green_marble_probability_l4067_406761

theorem green_marble_probability (total : ℕ) (p_white p_red_or_blue : ℚ) : 
  total = 84 →
  p_white = 1/4 →
  p_red_or_blue = 463/1000 →
  (total : ℚ) * (1 - p_white - p_red_or_blue) / total = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_green_marble_probability_l4067_406761


namespace NUMINAMATH_CALUDE_total_choices_is_81_l4067_406731

/-- The number of bases available for students to choose from. -/
def num_bases : ℕ := 3

/-- The number of students choosing bases. -/
def num_students : ℕ := 4

/-- The total number of ways students can choose bases. -/
def total_choices : ℕ := num_bases ^ num_students

/-- Theorem stating that the total number of choices is 81. -/
theorem total_choices_is_81 : total_choices = 81 := by
  sorry

end NUMINAMATH_CALUDE_total_choices_is_81_l4067_406731


namespace NUMINAMATH_CALUDE_gcd_1785_840_l4067_406781

theorem gcd_1785_840 : Nat.gcd 1785 840 = 105 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1785_840_l4067_406781


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l4067_406790

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + 2*x - 8 ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l4067_406790


namespace NUMINAMATH_CALUDE_abs_eq_sum_l4067_406777

theorem abs_eq_sum (x : ℝ) : (|x - 5| = 23) → (∃ y : ℝ, |y - 5| = 23 ∧ x + y = 10) := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_sum_l4067_406777


namespace NUMINAMATH_CALUDE_arrangement_theorem_l4067_406788

/-- The number of ways to arrange n boys and m girls in a row with girls standing together -/
def arrange_girls_together (n m : ℕ) : ℕ := sorry

/-- The number of ways to arrange n boys and m girls in a row with no two boys next to each other -/
def arrange_boys_apart (n m : ℕ) : ℕ := sorry

theorem arrangement_theorem :
  (arrange_girls_together 3 4 = 576) ∧ 
  (arrange_boys_apart 3 4 = 1440) := by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l4067_406788


namespace NUMINAMATH_CALUDE_adjacent_sums_odd_in_circular_arrangement_l4067_406794

/-- A circular arrangement of 2020 natural numbers -/
def CircularArrangement := Fin 2020 → ℕ

/-- The property that the sum of any two adjacent numbers in the arrangement is odd -/
def AdjacentSumsOdd (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 2020, Odd ((arr i) + (arr (i + 1)))

/-- Theorem stating that for any circular arrangement of 2020 natural numbers,
    the sum of any two adjacent numbers is odd -/
theorem adjacent_sums_odd_in_circular_arrangement :
  ∀ arr : CircularArrangement, AdjacentSumsOdd arr :=
sorry

end NUMINAMATH_CALUDE_adjacent_sums_odd_in_circular_arrangement_l4067_406794


namespace NUMINAMATH_CALUDE_girls_percentage_is_60_percent_l4067_406759

/-- Represents the number of students in the school -/
def total_students : ℕ := 150

/-- Represents the number of boys who did not join varsity clubs -/
def boys_not_in_varsity : ℕ := 40

/-- Represents the fraction of boys who joined varsity clubs -/
def boys_varsity_fraction : ℚ := 1/3

/-- Calculates the percentage of girls in the school -/
def girls_percentage : ℚ :=
  let total_boys : ℕ := boys_not_in_varsity * 3 / 2
  let total_girls : ℕ := total_students - total_boys
  (total_girls : ℚ) / total_students * 100

/-- Theorem stating that the percentage of girls in the school is 60% -/
theorem girls_percentage_is_60_percent : girls_percentage = 60 := by
  sorry

end NUMINAMATH_CALUDE_girls_percentage_is_60_percent_l4067_406759


namespace NUMINAMATH_CALUDE_impossibility_of_2005_vectors_l4067_406743

/-- A type representing a vector in a plane -/
def PlaneVector : Type := ℝ × ℝ

/-- A function to check if a vector is non-zero -/
def is_nonzero (v : PlaneVector) : Prop := v ≠ (0, 0)

/-- A function to calculate the sum of three vectors -/
def sum_three (v1 v2 v3 : PlaneVector) : PlaneVector :=
  (v1.1 + v2.1 + v3.1, v1.2 + v2.2 + v3.2)

/-- The main theorem -/
theorem impossibility_of_2005_vectors :
  ¬ ∃ (vectors : Fin 2005 → PlaneVector),
    (∀ i, is_nonzero (vectors i)) ∧
    (∀ (subset : Fin 10 → Fin 2005),
      ∃ (i j k : Fin 10), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
        sum_three (vectors (subset i)) (vectors (subset j)) (vectors (subset k)) = (0, 0)) :=
by sorry

end NUMINAMATH_CALUDE_impossibility_of_2005_vectors_l4067_406743


namespace NUMINAMATH_CALUDE_det_scaled_columns_l4067_406797

variable {α : Type*} [LinearOrderedField α]

noncomputable def det (a b c : α × α × α) : α := sorry

theorem det_scaled_columns (a b c : α × α × α) :
  let D := det a b c
  det (3 • a) (2 • b) c = 6 * D :=
sorry

end NUMINAMATH_CALUDE_det_scaled_columns_l4067_406797


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l4067_406789

/-- The cost of fruits at Patty's Produce Palace -/
structure FruitCost where
  banana : ℕ → ℚ
  apple : ℕ → ℚ
  orange : ℕ → ℚ

/-- The cost relationships between fruits -/
def cost_relation (c : FruitCost) : Prop :=
  c.banana 4 = c.apple 3 ∧ c.apple 5 = c.orange 2

/-- Theorem: 20 bananas cost the same as 6 oranges -/
theorem banana_orange_equivalence (c : FruitCost) (h : cost_relation c) : 
  c.banana 20 = c.orange 6 := by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l4067_406789


namespace NUMINAMATH_CALUDE_fox_can_eat_80_fox_cannot_eat_65_l4067_406746

/-- Represents a distribution of candies into three piles -/
structure CandyDistribution :=
  (pile1 pile2 pile3 : ℕ)
  (sum_eq_100 : pile1 + pile2 + pile3 = 100)

/-- Calculates the number of candies the fox eats given a distribution -/
def fox_candies (d : CandyDistribution) : ℕ :=
  if d.pile1 = d.pile2 ∨ d.pile1 = d.pile3 ∨ d.pile2 = d.pile3
  then max d.pile1 (max d.pile2 d.pile3)
  else d.pile1 + d.pile2 + d.pile3 - 2 * min d.pile1 (min d.pile2 d.pile3)

theorem fox_can_eat_80 : ∃ d : CandyDistribution, fox_candies d = 80 := by
  sorry

theorem fox_cannot_eat_65 : ¬ ∃ d : CandyDistribution, fox_candies d = 65 := by
  sorry

end NUMINAMATH_CALUDE_fox_can_eat_80_fox_cannot_eat_65_l4067_406746


namespace NUMINAMATH_CALUDE_gumdrops_problem_l4067_406733

/-- The maximum number of gumdrops that can be bought with a given amount of money and cost per gumdrop. -/
def max_gumdrops (total_money : ℕ) (cost_per_gumdrop : ℕ) : ℕ :=
  total_money / cost_per_gumdrop

/-- Theorem stating that with 80 cents and gumdrops costing 4 cents each, the maximum number of gumdrops that can be bought is 20. -/
theorem gumdrops_problem :
  max_gumdrops 80 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_gumdrops_problem_l4067_406733


namespace NUMINAMATH_CALUDE_digit_product_is_30_l4067_406706

/-- Represents a 3x3 grid of digits -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Check if all digits from 1 to 9 are used exactly once in the grid -/
def allDigitsUsedOnce (g : Grid) : Prop := ∀ d : Fin 9, ∃! (i j : Fin 3), g i j = d

/-- Product of digits in a row -/
def rowProduct (g : Grid) (row : Fin 3) : ℕ := (g row 0).val.succ * (g row 1).val.succ * (g row 2).val.succ

/-- Product of digits in a column -/
def colProduct (g : Grid) (col : Fin 3) : ℕ := (g 0 col).val.succ * (g 1 col).val.succ * (g 2 col).val.succ

/-- Product of digits in the shaded cells (top-left, center, bottom-right) -/
def shadedProduct (g : Grid) : ℕ := (g 0 0).val.succ * (g 1 1).val.succ * (g 2 2).val.succ

theorem digit_product_is_30 (g : Grid) 
  (h1 : allDigitsUsedOnce g)
  (h2 : rowProduct g 0 = 12)
  (h3 : rowProduct g 1 = 112)
  (h4 : colProduct g 0 = 216)
  (h5 : colProduct g 1 = 12) :
  shadedProduct g = 30 := by
  sorry

end NUMINAMATH_CALUDE_digit_product_is_30_l4067_406706


namespace NUMINAMATH_CALUDE_rectangle_longer_side_length_l4067_406776

/-- Given a circle of radius 6 cm tangent to three sides of a rectangle,
    and the area of the rectangle is three times the area of the circle,
    prove that the length of the longer side of the rectangle is 4.5π cm. -/
theorem rectangle_longer_side_length (r : ℝ) (circle_area rectangle_area : ℝ) 
  (h1 : r = 6)
  (h2 : circle_area = Real.pi * r^2)
  (h3 : rectangle_area = 3 * circle_area)
  (h4 : ∃ (shorter_side longer_side : ℝ), 
        shorter_side = 2 * (2 * r) ∧ 
        rectangle_area = shorter_side * longer_side) :
  ∃ (longer_side : ℝ), longer_side = 4.5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_length_l4067_406776


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4067_406786

/-- A hyperbola with center at the origin, axes of symmetry being coordinate axes,
    one focus coinciding with the focus of y^2 = 8x, and one asymptote being x + y = 0 -/
structure Hyperbola where
  /-- The focus of the parabola y^2 = 8x is (2, 0) -/
  focus : ℝ × ℝ
  /-- One asymptote of the hyperbola is x + y = 0 -/
  asymptote : ℝ → ℝ
  /-- The hyperbola's equation is in the form (x^2 / a^2) - (y^2 / b^2) = 1 -/
  a : ℝ
  b : ℝ
  focus_eq : focus = (2, 0)
  asymptote_eq : asymptote = fun x => -x
  ab_relation : b / a = 1

/-- The equation of the hyperbola is x^2/2 - y^2/2 = 1 -/
theorem hyperbola_equation (C : Hyperbola) : 
  ∀ x y : ℝ, (x^2 / 2) - (y^2 / 2) = 1 ↔ 
    (x^2 / C.a^2) - (y^2 / C.b^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4067_406786


namespace NUMINAMATH_CALUDE_branch_A_more_profitable_l4067_406767

/-- Represents a branch of the factory -/
inductive Branch
| A
| B

/-- Represents a grade of the product -/
inductive Grade
| A
| B
| C
| D

/-- Returns the processing fee for a given grade -/
def processingFee (g : Grade) : Int :=
  match g with
  | Grade.A => 90
  | Grade.B => 50
  | Grade.C => 20
  | Grade.D => -50

/-- Returns the processing cost for a given branch -/
def processingCost (b : Branch) : Int :=
  match b with
  | Branch.A => 25
  | Branch.B => 20

/-- Returns the frequency of a grade for a given branch -/
def frequency (b : Branch) (g : Grade) : Rat :=
  match b, g with
  | Branch.A, Grade.A => 40 / 100
  | Branch.A, Grade.B => 20 / 100
  | Branch.A, Grade.C => 20 / 100
  | Branch.A, Grade.D => 20 / 100
  | Branch.B, Grade.A => 28 / 100
  | Branch.B, Grade.B => 17 / 100
  | Branch.B, Grade.C => 34 / 100
  | Branch.B, Grade.D => 21 / 100

/-- Calculates the average profit for a given branch -/
def averageProfit (b : Branch) : Rat :=
  (processingFee Grade.A - processingCost b) * frequency b Grade.A +
  (processingFee Grade.B - processingCost b) * frequency b Grade.B +
  (processingFee Grade.C - processingCost b) * frequency b Grade.C +
  (processingFee Grade.D - processingCost b) * frequency b Grade.D

/-- Theorem stating that Branch A has higher average profit than Branch B -/
theorem branch_A_more_profitable : averageProfit Branch.A > averageProfit Branch.B := by
  sorry


end NUMINAMATH_CALUDE_branch_A_more_profitable_l4067_406767


namespace NUMINAMATH_CALUDE_star_calculation_l4067_406792

def star (x y : ℝ) : ℝ := x^2 + y^2

theorem star_calculation : (star (star 3 5) 4) = 1172 := by sorry

end NUMINAMATH_CALUDE_star_calculation_l4067_406792


namespace NUMINAMATH_CALUDE_pushup_progression_l4067_406752

/-- 
Given a person who does push-ups 3 times a week, increasing by 5 each time,
prove that if the total for the week is 45, then the number of push-ups on the first day is 10.
-/
theorem pushup_progression (first_day : ℕ) : 
  first_day + (first_day + 5) + (first_day + 10) = 45 → first_day = 10 := by
  sorry

end NUMINAMATH_CALUDE_pushup_progression_l4067_406752


namespace NUMINAMATH_CALUDE_james_money_l4067_406783

/-- Calculates the total money James has after finding additional bills -/
def total_money (bills_found : ℕ) (bill_value : ℕ) (existing_money : ℕ) : ℕ :=
  bills_found * bill_value + existing_money

/-- Proves that James has $135 after finding 3 $20 bills when he already had $75 -/
theorem james_money :
  total_money 3 20 75 = 135 := by
  sorry

end NUMINAMATH_CALUDE_james_money_l4067_406783


namespace NUMINAMATH_CALUDE_no_prime_solution_l4067_406709

/-- Represents a number in base p notation -/
def BaseP (coeffs : List Nat) (p : Nat) : Nat :=
  coeffs.enum.foldl (fun acc (i, a) => acc + a * p^i) 0

theorem no_prime_solution :
  ¬∃ (p : Nat), 
    Nat.Prime p ∧ 
    (BaseP [7, 1, 0, 2] p + BaseP [2, 0, 4] p + BaseP [4, 1, 1] p + 
     BaseP [0, 3, 2] p + BaseP [7] p = 
     BaseP [1, 0, 3] p + BaseP [2, 7, 4] p + BaseP [3, 1, 5] p) :=
by sorry

#eval BaseP [7, 1, 0, 2] 10  -- Should output 2017
#eval BaseP [2, 0, 4] 10     -- Should output 402
#eval BaseP [4, 1, 1] 10     -- Should output 114
#eval BaseP [0, 3, 2] 10     -- Should output 230
#eval BaseP [7] 10           -- Should output 7
#eval BaseP [1, 0, 3] 10     -- Should output 301
#eval BaseP [2, 7, 4] 10     -- Should output 472
#eval BaseP [3, 1, 5] 10     -- Should output 503

end NUMINAMATH_CALUDE_no_prime_solution_l4067_406709


namespace NUMINAMATH_CALUDE_company_max_people_l4067_406750

/-- Represents a company with three clubs and their membership information -/
structure Company where
  M : ℕ  -- Number of people in club M
  S : ℕ  -- Number of people in club S
  Z : ℕ  -- Number of people in club Z
  none : ℕ  -- Maximum number of people not in any club

/-- The maximum number of people in the company -/
def Company.maxPeople (c : Company) : ℕ := c.M + c.S + c.Z + c.none

/-- Theorem stating the maximum number of people in the company under given conditions -/
theorem company_max_people :
  ∀ (c : Company),
  c.M = 16 →
  c.S = 18 →
  c.Z = 11 →
  c.none ≤ 26 →
  c.maxPeople ≤ 71 := by
  sorry

#check company_max_people

end NUMINAMATH_CALUDE_company_max_people_l4067_406750
