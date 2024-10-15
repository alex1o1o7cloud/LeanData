import Mathlib

namespace NUMINAMATH_CALUDE_trapezoid_division_l1079_107985

/-- Represents a trapezoid with the given side lengths -/
structure Trapezoid where
  short_base : ℝ
  long_base : ℝ
  side1 : ℝ
  side2 : ℝ

/-- Represents a point that divides a line segment -/
structure DivisionPoint where
  ratio : ℝ

/-- 
Given a trapezoid with parallel sides of length 3 and 9, and non-parallel sides of length 4 and 6,
if a line parallel to the bases divides the trapezoid into two trapezoids of equal perimeters,
then this line divides each of the non-parallel sides in the ratio 3:2.
-/
theorem trapezoid_division (t : Trapezoid) (d : DivisionPoint) : 
  t.short_base = 3 ∧ t.long_base = 9 ∧ t.side1 = 4 ∧ t.side2 = 6 →
  (t.long_base - t.short_base) * d.ratio + t.short_base = 
    (t.side1 * d.ratio + t.side2 * d.ratio) / 2 →
  d.ratio = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_division_l1079_107985


namespace NUMINAMATH_CALUDE_upper_limit_of_set_D_l1079_107934

def is_prime (n : ℕ) : Prop := sorry

def set_D (upper_bound : ℕ) : Set ℕ :=
  {n : ℕ | 10 < n ∧ n ≤ upper_bound ∧ is_prime n}

theorem upper_limit_of_set_D (upper_bound : ℕ) :
  (∃ (a b : ℕ), a ∈ set_D upper_bound ∧ b ∈ set_D upper_bound ∧ b - a = 12) →
  (∃ (max : ℕ), max ∈ set_D upper_bound ∧ ∀ (x : ℕ), x ∈ set_D upper_bound → x ≤ max) →
  (∃ (max : ℕ), max ∈ set_D upper_bound ∧ ∀ (x : ℕ), x ∈ set_D upper_bound → x ≤ max ∧ max = 23) :=
by sorry

end NUMINAMATH_CALUDE_upper_limit_of_set_D_l1079_107934


namespace NUMINAMATH_CALUDE_certain_number_value_l1079_107911

theorem certain_number_value : ∃ x : ℝ, 0.65 * 40 = (4/5) * x + 6 ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l1079_107911


namespace NUMINAMATH_CALUDE_expired_milk_probability_l1079_107968

theorem expired_milk_probability (total_bags : ℕ) (expired_bags : ℕ) 
  (h1 : total_bags = 25) (h2 : expired_bags = 4) :
  (expired_bags : ℚ) / total_bags = 4 / 25 :=
by sorry

end NUMINAMATH_CALUDE_expired_milk_probability_l1079_107968


namespace NUMINAMATH_CALUDE_dealer_gain_percent_l1079_107949

theorem dealer_gain_percent (list_price : ℝ) (list_price_pos : list_price > 0) :
  let purchase_price := (3/4) * list_price
  let selling_price := (3/2) * list_price
  let gain := selling_price - purchase_price
  let gain_percent := (gain / purchase_price) * 100
  gain_percent = 100 := by sorry

end NUMINAMATH_CALUDE_dealer_gain_percent_l1079_107949


namespace NUMINAMATH_CALUDE_initial_songs_count_l1079_107909

/-- 
Given an album where:
- Each song is 3 minutes long
- Adding 10 more songs will make the total listening time 105 minutes
Prove that the initial number of songs in the album is 25.
-/
theorem initial_songs_count (song_duration : ℕ) (additional_songs : ℕ) (total_duration : ℕ) :
  song_duration = 3 →
  additional_songs = 10 →
  total_duration = 105 →
  ∃ (initial_songs : ℕ), song_duration * (initial_songs + additional_songs) = total_duration ∧ initial_songs = 25 :=
by sorry

end NUMINAMATH_CALUDE_initial_songs_count_l1079_107909


namespace NUMINAMATH_CALUDE_max_area_of_similar_triangle_l1079_107932

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a triangle in the grid -/
structure GridTriangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint

/-- The side length of the square grid -/
def gridSize : ℕ := 5

/-- Function to check if a point is within the grid -/
def isInGrid (p : GridPoint) : Prop :=
  0 ≤ p.x ∧ p.x < gridSize ∧ 0 ≤ p.y ∧ p.y < gridSize

/-- Function to check if a triangle is within the grid -/
def isTriangleInGrid (t : GridTriangle) : Prop :=
  isInGrid t.A ∧ isInGrid t.B ∧ isInGrid t.C

/-- Function to calculate the area of a triangle given its vertices -/
noncomputable def triangleArea (t : GridTriangle) : ℝ :=
  sorry -- Actual calculation would go here

/-- Function to check if two triangles are similar -/
def areSimilarTriangles (t1 t2 : GridTriangle) : Prop :=
  sorry -- Actual similarity check would go here

theorem max_area_of_similar_triangle :
  ∀ (ABC : GridTriangle),
    isTriangleInGrid ABC →
    ∃ (DEF : GridTriangle),
      isTriangleInGrid DEF ∧
      areSimilarTriangles ABC DEF ∧
      triangleArea DEF ≤ 2.5 ∧
      ∀ (XYZ : GridTriangle),
        isTriangleInGrid XYZ →
        areSimilarTriangles ABC XYZ →
        triangleArea XYZ ≤ triangleArea DEF :=
by sorry


end NUMINAMATH_CALUDE_max_area_of_similar_triangle_l1079_107932


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l1079_107979

theorem unique_number_with_three_prime_divisors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧
    (∀ d : ℕ, d ∣ x → d = 1 ∨ d = p ∨ d = q ∨ d = 11 ∨ d = p*q ∨ d = p*11 ∨ d = q*11 ∨ d = p*q*11)) →
  11 ∣ x →
  x = 59048 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l1079_107979


namespace NUMINAMATH_CALUDE_sin_15_sin_75_equals_half_l1079_107973

theorem sin_15_sin_75_equals_half : 2 * Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_sin_75_equals_half_l1079_107973


namespace NUMINAMATH_CALUDE_diane_stamp_arrangements_l1079_107984

/-- Represents a collection of stamps with their quantities -/
def StampCollection := List (Nat × Nat)

/-- Represents an arrangement of stamps -/
def StampArrangement := List Nat

/-- Returns true if the arrangement sums to the target value -/
def isValidArrangement (arrangement : StampArrangement) (target : Nat) : Bool :=
  arrangement.sum = target

/-- Returns true if the arrangement is possible given the stamp collection -/
def isPossibleArrangement (arrangement : StampArrangement) (collection : StampCollection) : Bool :=
  sorry

/-- Counts the number of unique arrangements given a stamp collection and target sum -/
def countUniqueArrangements (collection : StampCollection) (target : Nat) : Nat :=
  sorry

/-- Diane's stamp collection -/
def dianeCollection : StampCollection :=
  [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]

theorem diane_stamp_arrangements :
  countUniqueArrangements dianeCollection 12 = 30 := by sorry

end NUMINAMATH_CALUDE_diane_stamp_arrangements_l1079_107984


namespace NUMINAMATH_CALUDE_intersecting_line_passes_through_fixed_point_l1079_107948

/-- An ellipse with eccentricity 1/2 passing through (1, 3/2) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 1/4
  h_point : 1^2 / a^2 + (3/2)^2 / b^2 = 1

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ
  h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≠ x₂ ∧
    y₁ = k * x₁ + m ∧
    y₂ = k * x₂ + m ∧
    x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1 ∧
    x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1

/-- The theorem stating that the line passes through a fixed point -/
theorem intersecting_line_passes_through_fixed_point (E : Ellipse) (l : IntersectingLine E) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≠ x₂ ∧
    y₁ = l.k * x₁ + l.m ∧
    y₂ = l.k * x₂ + l.m ∧
    x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1 ∧
    x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1 ∧
    (x₁ - E.a) * (x₂ - E.a) + y₁ * y₂ = 0 →
    l.k * (2/7) + l.m = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_line_passes_through_fixed_point_l1079_107948


namespace NUMINAMATH_CALUDE_alphabet_letter_count_l1079_107958

theorem alphabet_letter_count (total : ℕ) (both : ℕ) (line_only : ℕ) 
  (h1 : total = 40)
  (h2 : both = 11)
  (h3 : line_only = 24)
  (h4 : total = both + line_only + (total - (both + line_only))) :
  total - (both + line_only) = 5 := by
  sorry

end NUMINAMATH_CALUDE_alphabet_letter_count_l1079_107958


namespace NUMINAMATH_CALUDE_sum_of_digits_product_72_sevens_72_fives_l1079_107975

/-- Represents a number consisting of n repetitions of a single digit --/
def repeatedDigit (d : Nat) (n : Nat) : Nat :=
  d * (10^n - 1) / 9

/-- Calculates the sum of digits in a natural number --/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem to be proved --/
theorem sum_of_digits_product_72_sevens_72_fives :
  sumOfDigits (repeatedDigit 7 72 * repeatedDigit 5 72) = 576 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_product_72_sevens_72_fives_l1079_107975


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1079_107962

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The theorem states that for an arithmetic sequence satisfying
    the given conditions, the general term is 2n - 3. -/
theorem arithmetic_sequence_formula (a : ℕ → ℝ)
    (h_arith : is_arithmetic_sequence a)
    (h_mean1 : (a 2 + a 6) / 2 = 5)
    (h_mean2 : (a 3 + a 7) / 2 = 7) :
    ∀ n : ℕ, a n = 2 * n - 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1079_107962


namespace NUMINAMATH_CALUDE_largest_sum_is_994_l1079_107954

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The sum of the given configuration -/
def sum (x y : Digit) : ℕ := 113 * x.val + 10 * y.val

/-- The largest possible 3-digit sum for the given configuration -/
def largest_sum : ℕ := 994

theorem largest_sum_is_994 (x y z : Digit) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) :
  sum x y ≤ largest_sum ∧
  ∃ (a b : Digit), sum a b = largest_sum ∧ a ≠ b :=
sorry

end NUMINAMATH_CALUDE_largest_sum_is_994_l1079_107954


namespace NUMINAMATH_CALUDE_binary_polynomial_form_l1079_107991

/-- A binary homogeneous polynomial of degree n -/
def BinaryHomogeneousPolynomial (n : ℕ) := ℝ → ℝ → ℝ

/-- The polynomial condition for all real numbers a, b, c -/
def SatisfiesCondition (P : BinaryHomogeneousPolynomial n) : Prop :=
  ∀ a b c : ℝ, P (a + b) c + P (b + c) a + P (c + a) b = 0

/-- The theorem stating the form of the polynomial P -/
theorem binary_polynomial_form (n : ℕ) (P : BinaryHomogeneousPolynomial n)
  (h1 : SatisfiesCondition P) (h2 : P 1 0 = 1) :
  ∃ f : ℝ → ℝ → ℝ, (∀ x y : ℝ, P x y = f x y * (x - 2*y)) ∧
                    (∀ x y : ℝ, f x y = (x + y)^(n-1)) :=
sorry

end NUMINAMATH_CALUDE_binary_polynomial_form_l1079_107991


namespace NUMINAMATH_CALUDE_susan_correct_percentage_l1079_107923

theorem susan_correct_percentage (y : ℝ) (h : y ≠ 0) :
  let total_questions : ℝ := 8 * y
  let unattempted_questions : ℝ := 2 * y + 3
  let correct_questions : ℝ := total_questions - unattempted_questions
  let percentage_correct : ℝ := (correct_questions / total_questions) * 100
  percentage_correct = 75 * (2 * y - 1) / y :=
by sorry

end NUMINAMATH_CALUDE_susan_correct_percentage_l1079_107923


namespace NUMINAMATH_CALUDE_a_in_range_and_negative_one_in_A_l1079_107924

def A : Set ℝ := {x | x^2 - 2 < 0}

theorem a_in_range_and_negative_one_in_A (a : ℝ) (h : a ∈ A) :
  -Real.sqrt 2 < a ∧ a < Real.sqrt 2 ∧ -1 ∈ A := by
  sorry

end NUMINAMATH_CALUDE_a_in_range_and_negative_one_in_A_l1079_107924


namespace NUMINAMATH_CALUDE_f_local_min_at_neg_one_f_two_extrema_iff_l1079_107904

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x - a * Real.exp x)

-- Theorem 1: When a = 0, f has a local minimum at x = -1
theorem f_local_min_at_neg_one :
  ∃ δ > 0, ∀ x, |x - (-1)| < δ ∧ x ≠ -1 → f 0 x > f 0 (-1) :=
sorry

-- Theorem 2: f has two different extremum points iff 0 < a < 1/2
theorem f_two_extrema_iff (a : ℝ) :
  (∃ x₁ x₂, x₁ < x₂ ∧ 
    (∀ h, 0 < h → f a (x₁ - h) > f a x₁ ∧ f a (x₁ + h) > f a x₁) ∧
    (∀ h, 0 < h → f a (x₂ - h) < f a x₂ ∧ f a (x₂ + h) < f a x₂))
  ↔ 0 < a ∧ a < 1/2 :=
sorry

end

end NUMINAMATH_CALUDE_f_local_min_at_neg_one_f_two_extrema_iff_l1079_107904


namespace NUMINAMATH_CALUDE_area_of_special_parallelogram_l1079_107995

/-- Represents a parallelogram with base and altitude. -/
structure Parallelogram where
  base : ℝ
  altitude : ℝ

/-- The area of a parallelogram. -/
def area (p : Parallelogram) : ℝ := p.base * p.altitude

/-- A parallelogram with altitude twice the base and base length 12. -/
def special_parallelogram : Parallelogram where
  base := 12
  altitude := 2 * 12

theorem area_of_special_parallelogram :
  area special_parallelogram = 288 := by
  sorry

end NUMINAMATH_CALUDE_area_of_special_parallelogram_l1079_107995


namespace NUMINAMATH_CALUDE_postman_speed_calculation_postman_speed_is_30_l1079_107937

/-- Calculates the downhill average speed of a postman's round trip, given the following conditions:
  * The route length is 5 miles each way
  * The uphill delivery takes 2 hours
  * The uphill average speed is 4 miles per hour
  * The overall average speed for the round trip is 6 miles per hour
  * There's an extra 15 minutes (0.25 hours) delay on the return trip due to rain
-/
theorem postman_speed_calculation (route_length : ℝ) (uphill_time : ℝ) (uphill_speed : ℝ) 
  (overall_speed : ℝ) (rain_delay : ℝ) : ℝ :=
  let downhill_speed := 
    route_length / (((2 * route_length) / overall_speed) - uphill_time - rain_delay)
  30

/-- The main theorem that proves the downhill speed is 30 mph given the specific conditions -/
theorem postman_speed_is_30 : 
  postman_speed_calculation 5 2 4 6 0.25 = 30 := by
  sorry

end NUMINAMATH_CALUDE_postman_speed_calculation_postman_speed_is_30_l1079_107937


namespace NUMINAMATH_CALUDE_gambler_final_amount_l1079_107933

def gamble (initial : ℚ) (rounds : ℕ) (wins : ℕ) (losses : ℕ) : ℚ :=
  let bet_fraction : ℚ := 1/3
  let win_multiplier : ℚ := 2
  let loss_multiplier : ℚ := 1
  sorry

theorem gambler_final_amount :
  let initial_amount : ℚ := 100
  let total_rounds : ℕ := 4
  let wins : ℕ := 2
  let losses : ℕ := 2
  gamble initial_amount total_rounds wins losses = 8000/81 := by sorry

end NUMINAMATH_CALUDE_gambler_final_amount_l1079_107933


namespace NUMINAMATH_CALUDE_fraction_equality_l1079_107938

theorem fraction_equality (a b : ℝ) (h : a / b = 2 / 5) : a / (a + b) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1079_107938


namespace NUMINAMATH_CALUDE_algae_coverage_on_day_17_algae_doubles_daily_full_coverage_on_day_20_l1079_107925

/-- Represents the coverage of algae on the lake on a given day -/
def algae_coverage (day : ℕ) : ℝ :=
  2^(day - 17)

/-- The day when the lake is completely covered with algae -/
def full_coverage_day : ℕ := 20

theorem algae_coverage_on_day_17 :
  algae_coverage 17 = 0.125 ∧ 1 - algae_coverage 17 = 0.875 := by sorry

theorem algae_doubles_daily (d : ℕ) (h : d < full_coverage_day) :
  algae_coverage (d + 1) = 2 * algae_coverage d := by sorry

theorem full_coverage_on_day_20 :
  algae_coverage full_coverage_day = 1 := by sorry

end NUMINAMATH_CALUDE_algae_coverage_on_day_17_algae_doubles_daily_full_coverage_on_day_20_l1079_107925


namespace NUMINAMATH_CALUDE_equal_digit_probability_l1079_107912

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The number of one-digit outcomes on a die -/
def one_digit_outcomes : ℕ := 9

/-- The number of two-digit outcomes on a die -/
def two_digit_outcomes : ℕ := 11

/-- The probability of rolling an equal number of one-digit and two-digit numbers with 5 20-sided dice -/
theorem equal_digit_probability : 
  (Nat.choose num_dice (num_dice / 2) *
   (one_digit_outcomes ^ (num_dice / 2) * two_digit_outcomes ^ (num_dice - num_dice / 2))) /
  (num_sides ^ num_dice) = 539055 / 1600000 := by
sorry

end NUMINAMATH_CALUDE_equal_digit_probability_l1079_107912


namespace NUMINAMATH_CALUDE_book_purchase_equation_l1079_107993

/-- Represents a book purchase scenario with two purchases -/
structure BookPurchase where
  first_cost : ℝ
  second_cost : ℝ
  quantity_difference : ℕ
  first_quantity : ℝ

/-- The equation correctly represents the book purchase scenario -/
def correct_equation (bp : BookPurchase) : Prop :=
  bp.first_cost / bp.first_quantity = bp.second_cost / (bp.first_quantity + bp.quantity_difference)

/-- Theorem stating that the given equation correctly represents the book purchase scenario -/
theorem book_purchase_equation (bp : BookPurchase) 
  (h1 : bp.first_cost = 7000)
  (h2 : bp.second_cost = 9000)
  (h3 : bp.quantity_difference = 60)
  (h4 : bp.first_quantity > 0) :
  correct_equation bp := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_equation_l1079_107993


namespace NUMINAMATH_CALUDE_trig_identity_l1079_107977

open Real

theorem trig_identity (α : ℝ) : 
  (1 / sin (-α) - sin (π + α)) / (1 / cos (3*π - α) + cos (2*π - α)) = 1 / tan α^3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1079_107977


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1079_107982

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 2 ↔ 3 * x + 4 > 5 * x - 1 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1079_107982


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1079_107992

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_sum : a 1 + 2 * a 2 = 3)
  (h_prod : a 3 ^ 2 = 4 * a 2 * a 6)
  (h_geo : GeometricSequence a) :
  a 4 = 3 / 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1079_107992


namespace NUMINAMATH_CALUDE_simplify_expression_l1079_107926

theorem simplify_expression : 
  (3 * Real.sqrt 12 - 2 * Real.sqrt (1/3) + Real.sqrt 48) / (2 * Real.sqrt 3) = 14/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1079_107926


namespace NUMINAMATH_CALUDE_friend_of_gcd_l1079_107963

/-- Two integers are friends if their product is a perfect square -/
def are_friends (a b : ℤ) : Prop := ∃ k : ℤ, a * b = k * k

/-- Main theorem: If a is a friend of b, then a is a friend of gcd(a, b) -/
theorem friend_of_gcd {a b : ℤ} (h : are_friends a b) : are_friends a (Int.gcd a b) := by
  sorry

end NUMINAMATH_CALUDE_friend_of_gcd_l1079_107963


namespace NUMINAMATH_CALUDE_max_radius_of_circle_l1079_107988

-- Define a circle in 2D space
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the theorem
theorem max_radius_of_circle (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ)
  (h1 : C = Circle center radius)
  (h2 : (4, 0) ∈ C)
  (h3 : (-4, 0) ∈ C)
  (h4 : ∃ (x y : ℝ), (x, y) ∈ C) :
  radius ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_max_radius_of_circle_l1079_107988


namespace NUMINAMATH_CALUDE_billys_restaurant_bill_l1079_107944

/-- Calculates the total bill for a group at Billy's Restaurant -/
def calculate_bill (num_adults : ℕ) (num_children : ℕ) (cost_per_meal : ℕ) : ℕ :=
  (num_adults + num_children) * cost_per_meal

/-- Proves that the bill for 2 adults and 5 children, with meals costing $3 each, is $21 -/
theorem billys_restaurant_bill :
  calculate_bill 2 5 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_billys_restaurant_bill_l1079_107944


namespace NUMINAMATH_CALUDE_snooker_ticket_difference_l1079_107953

theorem snooker_ticket_difference (vip_price gen_price : ℚ) 
  (total_tickets total_revenue : ℚ) (min_vip min_gen : ℕ) :
  vip_price = 40 →
  gen_price = 15 →
  total_tickets = 320 →
  total_revenue = 7500 →
  min_vip = 80 →
  min_gen = 100 →
  ∃ (vip_sold gen_sold : ℕ),
    vip_sold + gen_sold = total_tickets ∧
    vip_price * vip_sold + gen_price * gen_sold = total_revenue ∧
    vip_sold ≥ min_vip ∧
    gen_sold ≥ min_gen ∧
    gen_sold - vip_sold = 104 :=
by sorry

end NUMINAMATH_CALUDE_snooker_ticket_difference_l1079_107953


namespace NUMINAMATH_CALUDE_watermelon_pricing_l1079_107901

/-- Represents the number of watermelons each brother brought --/
structure Watermelons :=
  (elder : ℕ)
  (second : ℕ)
  (youngest : ℕ)

/-- Represents the number of watermelons sold in the morning --/
structure MorningSales :=
  (elder : ℕ)
  (second : ℕ)
  (youngest : ℕ)

/-- Theorem: Given the conditions, prove that the morning price was 3.75 yuan and the afternoon price was 1.25 yuan --/
theorem watermelon_pricing
  (w : Watermelons)
  (m : MorningSales)
  (h1 : w.elder = 10)
  (h2 : w.second = 16)
  (h3 : w.youngest = 26)
  (h4 : m.elder ≤ w.elder)
  (h5 : m.second ≤ w.second)
  (h6 : m.youngest ≤ w.youngest)
  (h7 : ∃ (morning_price afternoon_price : ℚ),
    morning_price > afternoon_price ∧
    afternoon_price > 0 ∧
    morning_price * m.elder + afternoon_price * (w.elder - m.elder) = 35 ∧
    morning_price * m.second + afternoon_price * (w.second - m.second) = 35 ∧
    morning_price * m.youngest + afternoon_price * (w.youngest - m.youngest) = 35) :
  ∃ (morning_price afternoon_price : ℚ),
    morning_price = 3.75 ∧ afternoon_price = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_pricing_l1079_107901


namespace NUMINAMATH_CALUDE_binomial_coefficient_15_l1079_107906

theorem binomial_coefficient_15 (n : ℕ) (h1 : n > 0) 
  (h2 : Nat.choose n 2 = 15) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_15_l1079_107906


namespace NUMINAMATH_CALUDE_lemonade_recipe_l1079_107935

/-- Lemonade recipe problem -/
theorem lemonade_recipe (lemon_juice sugar water : ℚ) : 
  water = 3 * sugar →  -- Water is 3 times sugar
  sugar = 3 * lemon_juice →  -- Sugar is 3 times lemon juice
  lemon_juice = 4 →  -- Luka uses 4 cups of lemon juice
  water = 36 := by  -- The amount of water needed is 36 cups
sorry


end NUMINAMATH_CALUDE_lemonade_recipe_l1079_107935


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l1079_107919

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Define triangle ABC
def triangle_ABC (a b c A B C : ℝ) : Prop :=
  -- Sides a, b, c are opposite to angles A, B, C respectively
  true

-- Define the given equation
def given_equation (a b c A C : ℝ) : Prop :=
  (2 * b - c) * Real.cos A = a * Real.cos C

-- Define the given conditions
def given_conditions (a b c : ℝ) : Prop :=
  a = 2 ∧ b + c = 4

-- Theorem statement
theorem triangle_ABC_properties 
  (h_triangle : triangle_ABC a b c A B C)
  (h_equation : given_equation a b c A C)
  (h_conditions : given_conditions a b c) :
  A = Real.pi / 3 ∧ 
  (1/2) * b * c * Real.sin A = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l1079_107919


namespace NUMINAMATH_CALUDE_factor_of_100140001_l1079_107928

theorem factor_of_100140001 : ∃ (n : ℕ), 
  8000 < n ∧ 
  n < 9000 ∧ 
  100140001 % n = 0 :=
by
  use 8221
  sorry

end NUMINAMATH_CALUDE_factor_of_100140001_l1079_107928


namespace NUMINAMATH_CALUDE_odd_function_fourth_composition_even_l1079_107980

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem statement
theorem odd_function_fourth_composition_even (f : ℝ → ℝ) (h : OddFunction f) : 
  EvenFunction (fun x ↦ f (f (f (f x)))) :=
sorry

end NUMINAMATH_CALUDE_odd_function_fourth_composition_even_l1079_107980


namespace NUMINAMATH_CALUDE_kendra_shirts_l1079_107931

/-- Represents the number of shirts Kendra needs for a two-week period --/
def shirts_needed : ℕ :=
  let weekday_shirts := 5
  let club_shirts := 3
  let saturday_shirt := 1
  let sunday_shirts := 2
  let weekly_shirts := weekday_shirts + club_shirts + saturday_shirt + sunday_shirts
  2 * weekly_shirts

/-- Theorem stating that Kendra needs 22 shirts for a two-week period --/
theorem kendra_shirts : shirts_needed = 22 := by
  sorry

end NUMINAMATH_CALUDE_kendra_shirts_l1079_107931


namespace NUMINAMATH_CALUDE_inequality_proof_l1079_107986

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1079_107986


namespace NUMINAMATH_CALUDE_wheels_in_garage_l1079_107902

theorem wheels_in_garage : 
  let bicycles : ℕ := 9
  let cars : ℕ := 16
  let wheels_per_bicycle : ℕ := 2
  let wheels_per_car : ℕ := 4
  bicycles * wheels_per_bicycle + cars * wheels_per_car = 82 :=
by sorry

end NUMINAMATH_CALUDE_wheels_in_garage_l1079_107902


namespace NUMINAMATH_CALUDE_expression_simplification_l1079_107964

theorem expression_simplification (x : ℝ) : 
  3 * x + 4 * (2 - x) - 2 * (3 - 2 * x) + 5 * (2 + 3 * x) = 18 * x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1079_107964


namespace NUMINAMATH_CALUDE_james_annual_training_hours_l1079_107905

/-- Represents James' training schedule and calculates his total training hours in a year --/
def jamesTrainingHours : ℕ :=
  let weeklyHours : ℕ := 3 * 2 * 4 + 2 * (3 + 5)  -- Weekly training hours
  let totalWeeks : ℕ := 52  -- Weeks in a year
  let holidayWeeks : ℕ := 1  -- Week off for holidays
  let missedDays : ℕ := 10  -- Additional missed days
  let trainingDaysPerWeek : ℕ := 5  -- Number of training days per week
  let effectiveTrainingWeeks : ℕ := totalWeeks - holidayWeeks - (missedDays / trainingDaysPerWeek)
  weeklyHours * effectiveTrainingWeeks

/-- Theorem stating that James trains for 1960 hours in a year --/
theorem james_annual_training_hours :
  jamesTrainingHours = 1960 := by
  sorry

end NUMINAMATH_CALUDE_james_annual_training_hours_l1079_107905


namespace NUMINAMATH_CALUDE_cafeteria_pie_problem_l1079_107927

/-- Given a cafeteria with initial apples, apples handed out, and number of pies made,
    calculate the number of apples used for each pie. -/
def apples_per_pie (initial_apples : ℕ) (apples_handed_out : ℕ) (num_pies : ℕ) : ℕ :=
  (initial_apples - apples_handed_out) / num_pies

/-- Theorem stating that given 47 initial apples, 27 apples handed out, and 5 pies made,
    the number of apples used for each pie is 4. -/
theorem cafeteria_pie_problem :
  apples_per_pie 47 27 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pie_problem_l1079_107927


namespace NUMINAMATH_CALUDE_hexagon_side_length_l1079_107978

/-- A regular hexagon with a point inside it -/
structure RegularHexagonWithPoint where
  /-- Side length of the hexagon -/
  side_length : ℝ
  /-- The point inside the hexagon -/
  point : ℝ × ℝ
  /-- First vertex of the hexagon -/
  vertex1 : ℝ × ℝ
  /-- Second vertex of the hexagon -/
  vertex2 : ℝ × ℝ
  /-- Third vertex of the hexagon -/
  vertex3 : ℝ × ℝ
  /-- The hexagon is regular -/
  regular : side_length > 0
  /-- The distance between the point and the first vertex is 1 -/
  dist1 : Real.sqrt ((point.1 - vertex1.1)^2 + (point.2 - vertex1.2)^2) = 1
  /-- The distance between the point and the second vertex is 1 -/
  dist2 : Real.sqrt ((point.1 - vertex2.1)^2 + (point.2 - vertex2.2)^2) = 1
  /-- The distance between the point and the third vertex is 2 -/
  dist3 : Real.sqrt ((point.1 - vertex3.1)^2 + (point.2 - vertex3.2)^2) = 2
  /-- The vertices are consecutive -/
  consecutive : Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = side_length ∧
                Real.sqrt ((vertex2.1 - vertex3.1)^2 + (vertex2.2 - vertex3.2)^2) = side_length

/-- The theorem stating that the side length of the hexagon is √3 -/
theorem hexagon_side_length (h : RegularHexagonWithPoint) : h.side_length = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l1079_107978


namespace NUMINAMATH_CALUDE_james_total_earnings_l1079_107950

def january_earnings : ℝ := 4000

def february_earnings (jan : ℝ) : ℝ := jan * 1.5

def march_earnings (feb : ℝ) : ℝ := feb * 0.8

def total_earnings (jan feb mar : ℝ) : ℝ := jan + feb + mar

theorem james_total_earnings :
  let feb := february_earnings january_earnings
  let mar := march_earnings feb
  total_earnings january_earnings feb mar = 14800 := by sorry

end NUMINAMATH_CALUDE_james_total_earnings_l1079_107950


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l1079_107969

/-- Linear function f(x) = 8x - 1 -/
def f (x : ℝ) : ℝ := 8 * x - 1

/-- Point P₁ lies on the graph of f -/
def P₁_on_f (y₁ : ℝ) : Prop := f 3 = y₁

/-- Point P₂ lies on the graph of f -/
def P₂_on_f (y₂ : ℝ) : Prop := f 4 = y₂

theorem y₁_less_than_y₂ (y₁ y₂ : ℝ) (h₁ : P₁_on_f y₁) (h₂ : P₂_on_f y₂) : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l1079_107969


namespace NUMINAMATH_CALUDE_max_candy_count_l1079_107920

/-- The number of candy pieces Frankie got -/
def frankies_candy : ℕ := 74

/-- The additional candy pieces Max got compared to Frankie -/
def extra_candy : ℕ := 18

/-- The number of candy pieces Max got -/
def maxs_candy : ℕ := frankies_candy + extra_candy

theorem max_candy_count : maxs_candy = 92 := by
  sorry

end NUMINAMATH_CALUDE_max_candy_count_l1079_107920


namespace NUMINAMATH_CALUDE_welders_left_correct_l1079_107908

/-- The number of welders who started working on another project --/
def welders_left : ℕ := 12

/-- The initial number of welders --/
def initial_welders : ℕ := 36

/-- The number of days to complete the order with all welders --/
def initial_days : ℕ := 5

/-- The number of additional days needed after some welders left --/
def additional_days : ℕ := 6

/-- The rate at which each welder works --/
def welder_rate : ℝ := 1

/-- The total work to be done --/
def total_work : ℝ := initial_welders * initial_days * welder_rate

theorem welders_left_correct :
  (initial_welders - welders_left) * (additional_days * welder_rate) =
  total_work - (initial_welders * welder_rate) := by sorry

end NUMINAMATH_CALUDE_welders_left_correct_l1079_107908


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1079_107915

def f (x : ℝ) := -2 * x^2 + 12 * x - 10

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1079_107915


namespace NUMINAMATH_CALUDE_thirtythree_by_thirtythree_black_count_l1079_107960

/-- Represents a checkerboard with alternating colors -/
structure Checkerboard where
  size : ℕ
  blackInCorners : Bool

/-- Counts the number of black squares on a checkerboard -/
def countBlackSquares (board : Checkerboard) : ℕ :=
  sorry

/-- Theorem: A 33x33 checkerboard with black corners has 545 black squares -/
theorem thirtythree_by_thirtythree_black_count :
  ∀ (board : Checkerboard),
    board.size = 33 ∧ board.blackInCorners = true →
    countBlackSquares board = 545 :=
by sorry

end NUMINAMATH_CALUDE_thirtythree_by_thirtythree_black_count_l1079_107960


namespace NUMINAMATH_CALUDE_arithmetic_geometric_comparison_l1079_107951

/-- An arithmetic sequence with positive terms and positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = a n + d

/-- A geometric sequence with positive terms -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n, b n > 0 ∧ b (n + 1) = b n * q

theorem arithmetic_geometric_comparison
  (a b : ℕ → ℝ) (d : ℝ)
  (h_arith : ArithmeticSequence a d)
  (h_geom : GeometricSequence b)
  (h_equal_1 : a 1 = b 1)
  (h_equal_2 : a 2 = b 2) :
  ∀ n ≥ 3, a n < b n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_comparison_l1079_107951


namespace NUMINAMATH_CALUDE_chocolate_bar_count_l1079_107918

/-- Represents the number of chocolate bars in a crate -/
def chocolate_bars_in_crate (large_boxes : ℕ) (small_boxes_per_large : ℕ) (bars_per_small : ℕ) : ℕ :=
  large_boxes * small_boxes_per_large * bars_per_small

/-- Proves that the total number of chocolate bars in the crate is 116,640 -/
theorem chocolate_bar_count :
  chocolate_bars_in_crate 45 36 72 = 116640 := by
  sorry

#eval chocolate_bars_in_crate 45 36 72

end NUMINAMATH_CALUDE_chocolate_bar_count_l1079_107918


namespace NUMINAMATH_CALUDE_simplify_expression_find_k_l1079_107987

-- Problem 1
theorem simplify_expression (x : ℝ) :
  (2*x + 1)^2 - (2*x + 1)*(2*x - 1) + (x + 1)*(x - 3) = x^2 + 2*x - 1 := by
  sorry

-- Problem 2
theorem find_k (x y k : ℝ) 
  (eq1 : x + y = 1)
  (eq2 : k*x + (k - 1)*y = 7)
  (eq3 : 3*x - 2*y = 5) :
  k = 33/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_find_k_l1079_107987


namespace NUMINAMATH_CALUDE_probability_at_least_one_head_l1079_107952

theorem probability_at_least_one_head (p : ℝ) (h1 : p = 1 / 2) :
  1 - (1 - p)^4 = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_head_l1079_107952


namespace NUMINAMATH_CALUDE_sum_equals_seventeen_l1079_107945

theorem sum_equals_seventeen 
  (a b c d : ℝ) 
  (h1 : a * (c + d) + b * (c + d) = 42) 
  (h2 : c + d = 3) : 
  a + b + c + d = 17 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_seventeen_l1079_107945


namespace NUMINAMATH_CALUDE_dividend_calculation_l1079_107970

theorem dividend_calculation (remainder quotient divisor dividend : ℕ) : 
  remainder = 8 →
  divisor = 3 * quotient →
  divisor = 3 * remainder + 3 →
  dividend = divisor * quotient + remainder →
  dividend = 251 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1079_107970


namespace NUMINAMATH_CALUDE_intersection_M_N_l1079_107913

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {-3, -2, -1, 0, 1}

theorem intersection_M_N :
  M ∩ N = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1079_107913


namespace NUMINAMATH_CALUDE_aluminum_carbonate_weight_l1079_107955

/-- The atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- The atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The molecular formula of aluminum carbonate -/
structure AluminumCarbonate where
  Al : Fin 2
  CO3 : Fin 3

/-- Calculate the molecular weight of aluminum carbonate -/
def molecular_weight (ac : AluminumCarbonate) : ℝ :=
  2 * Al_weight + 3 * C_weight + 9 * O_weight

/-- Theorem: The molecular weight of aluminum carbonate is 233.99 g/mol -/
theorem aluminum_carbonate_weight :
  ∀ ac : AluminumCarbonate, molecular_weight ac = 233.99 := by
  sorry

end NUMINAMATH_CALUDE_aluminum_carbonate_weight_l1079_107955


namespace NUMINAMATH_CALUDE_sunlovers_always_happy_l1079_107900

theorem sunlovers_always_happy (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sunlovers_always_happy_l1079_107900


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_p_iff_p_minus_one_l1079_107940

theorem smallest_n_divisible_by_p_iff_p_minus_one : ∃ (n : ℕ), n = 1806 ∧
  (∀ (p : ℕ), Nat.Prime p → (p ∣ n ↔ (p - 1) ∣ n)) ∧
  (∀ (m : ℕ), m < n → ∃ (q : ℕ), Nat.Prime q ∧ ((q ∣ m ↔ (q - 1) ∣ m) → False)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_p_iff_p_minus_one_l1079_107940


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1079_107967

theorem sphere_surface_area (r : ℝ) (R : ℝ) :
  r > 0 → R > 0 →
  r^2 + 1^2 = R^2 →
  π * r^2 = π →
  4 * π * R^2 = 8 * π := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1079_107967


namespace NUMINAMATH_CALUDE_combine_like_terms_l1079_107999

theorem combine_like_terms (a b : ℝ) : 
  2 * a^3 * b - (1/2) * a^3 * b - a^2 * b + (1/2) * a^2 * b - a * b^2 = 
  (3/2) * a^3 * b - (1/2) * a^2 * b - a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_l1079_107999


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1079_107929

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality (a b c : ℝ) (h1 : a > 0) 
  (h2 : ∀ x : ℝ, f a b c (2 + x) = f a b c (2 - x)) :
  f a b c 2 < f a b c 1 ∧ f a b c 1 < f a b c 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1079_107929


namespace NUMINAMATH_CALUDE_soccer_substitutions_remainder_l1079_107946

/-- Represents the number of ways to make substitutions in a soccer game -/
def substitutions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => 11 * (12 - m) * substitutions m

/-- The total number of ways to make 0 to 3 substitutions -/
def total_substitutions : ℕ :=
  substitutions 0 + substitutions 1 + substitutions 2 + substitutions 3

theorem soccer_substitutions_remainder :
  total_substitutions ≡ 122 [MOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_soccer_substitutions_remainder_l1079_107946


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l1079_107910

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- The sum of interior angles of a polygon with n sides -/
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: Each interior angle of a regular octagon measures 135 degrees -/
theorem regular_octagon_interior_angle :
  (interior_angle_sum octagon_sides) / octagon_sides = 135 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l1079_107910


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l1079_107942

theorem quadratic_perfect_square (x : ℝ) : ∃ (a : ℝ), x^2 - 20*x + 100 = (x + a)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l1079_107942


namespace NUMINAMATH_CALUDE_problem_solution_l1079_107996

theorem problem_solution (x : ℝ) : (20 / 100 * 30 = 25 / 100 * x + 2) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1079_107996


namespace NUMINAMATH_CALUDE_smallest_n_with_constant_term_l1079_107997

theorem smallest_n_with_constant_term : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → k < n → 
    ¬ ∃ (r : ℕ), r ≤ k ∧ 3 * k = (7 * r) / 2) ∧
  (∃ (r : ℕ), r ≤ n ∧ 3 * n = (7 * r) / 2) ∧
  n = 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_constant_term_l1079_107997


namespace NUMINAMATH_CALUDE_count_valid_selections_32_card_deck_l1079_107941

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- Calculates the number of ways to choose 6 cards from a deck
    such that all suits are represented -/
def count_valid_selections (d : Deck) : Nat :=
  let s1 := Nat.choose 4 2 * (Nat.choose 8 2)^2 * 8^2
  let s2 := Nat.choose 4 1 * Nat.choose 8 3 * 8^3
  s1 + s2

/-- The main theorem to be proved -/
theorem count_valid_selections_32_card_deck :
  ∃ (d : Deck), d.total_cards = 32 ∧ d.num_suits = 4 ∧ d.cards_per_suit = 8 ∧
  count_valid_selections d = 415744 :=
by
  sorry

end NUMINAMATH_CALUDE_count_valid_selections_32_card_deck_l1079_107941


namespace NUMINAMATH_CALUDE_smallest_c_inequality_l1079_107972

theorem smallest_c_inequality (c : ℝ) : 
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + c * |x^2 - y^2| ≥ (x + y) / 2) ↔ c ≥ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_inequality_l1079_107972


namespace NUMINAMATH_CALUDE_y_equals_zero_l1079_107939

theorem y_equals_zero (x y : ℝ) : (x + y)^5 - x^5 + y = 0 → y = 0 := by
  sorry

end NUMINAMATH_CALUDE_y_equals_zero_l1079_107939


namespace NUMINAMATH_CALUDE_flour_for_cake_l1079_107956

theorem flour_for_cake (total_flour : ℚ) (scoop_size : ℚ) (num_scoops : ℕ) : 
  total_flour = 8 →
  scoop_size = 1/4 →
  num_scoops = 8 →
  total_flour - (↑num_scoops * scoop_size) = 6 :=
by sorry

end NUMINAMATH_CALUDE_flour_for_cake_l1079_107956


namespace NUMINAMATH_CALUDE_farmer_duck_sales_l1079_107965

/-- A farmer sells ducks and chickens, buys a wheelbarrow, and resells it. -/
theorem farmer_duck_sales
  (duck_price : ℕ)
  (chicken_price : ℕ)
  (chicken_count : ℕ)
  (duck_count : ℕ)
  (wheelbarrow_profit : ℕ)
  (h1 : duck_price = 10)
  (h2 : chicken_price = 8)
  (h3 : chicken_count = 5)
  (h4 : wheelbarrow_profit = 60)
  (h5 : (duck_price * duck_count + chicken_price * chicken_count) / 2 = wheelbarrow_profit / 2) :
  duck_count = 2 := by
sorry


end NUMINAMATH_CALUDE_farmer_duck_sales_l1079_107965


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l1079_107974

theorem geometric_progression_ratio (x y z r : ℝ) 
  (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h2 : x * (2 * y - z) ≠ y * (2 * z - x))
  (h3 : y * (2 * z - x) ≠ z * (2 * x - y))
  (h4 : x * (2 * y - z) ≠ z * (2 * x - y))
  (h5 : ∃ (a : ℝ), a ≠ 0 ∧ 
    x * (2 * y - z) = a ∧ 
    y * (2 * z - x) = a * r ∧ 
    z * (2 * x - y) = a * r^2) :
  r^2 + r + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l1079_107974


namespace NUMINAMATH_CALUDE_residue_of_12_pow_2040_mod_19_l1079_107943

theorem residue_of_12_pow_2040_mod_19 :
  (12 : ℤ) ^ 2040 ≡ 7 [ZMOD 19] := by
  sorry

end NUMINAMATH_CALUDE_residue_of_12_pow_2040_mod_19_l1079_107943


namespace NUMINAMATH_CALUDE_sum_of_roots_squared_diff_eq_sum_of_roots_eq_fourteen_l1079_107947

theorem sum_of_roots_squared_diff_eq (a c : ℝ) : 
  (∀ x : ℝ, (x - a)^2 = c) → (∃ x₁ x₂ : ℝ, (x₁ - a)^2 = c ∧ (x₂ - a)^2 = c ∧ x₁ + x₂ = 2 * a) :=
by sorry

theorem sum_of_roots_eq_fourteen : 
  (∃ x₁ x₂ : ℝ, (x₁ - 7)^2 = 16 ∧ (x₂ - 7)^2 = 16 ∧ x₁ + x₂ = 14) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_squared_diff_eq_sum_of_roots_eq_fourteen_l1079_107947


namespace NUMINAMATH_CALUDE_equivalent_expression_proof_l1079_107903

theorem equivalent_expression_proof (n : ℕ) (hn : n > 1) :
  ∃ (p q : ℕ → ℕ),
    (∀ m : ℕ, m > 1 → 16^m + 4^m + 1 = (2^(p m) - 1) / (2^(q m) - 1)) ∧
    (∃ k : ℚ, ∀ m : ℕ, m > 1 → p m / q m = k) ∧
    p 2006 - q 2006 = 8024 :=
by
  sorry

end NUMINAMATH_CALUDE_equivalent_expression_proof_l1079_107903


namespace NUMINAMATH_CALUDE_euler_line_concurrency_l1079_107957

/-- A point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The Euler line of a triangle -/
def EulerLine (A B C : Point) : Set Point := sorry

/-- The point of concurrency of three lines -/
def Concurrent (l1 l2 l3 : Set Point) : Point := sorry

/-- Predicate to check if a triangle is not obtuse -/
def NotObtuse (A B C : Point) : Prop := sorry

theorem euler_line_concurrency 
  (A B C D : Point) 
  (h1 : NotObtuse A B C) 
  (h2 : NotObtuse B C D) 
  (h3 : NotObtuse C A D) 
  (h4 : NotObtuse D A B) 
  (P : Point) 
  (hP : P = Concurrent (EulerLine A B C) (EulerLine B C D) (EulerLine C A D)) :
  P ∈ EulerLine D A B := by
  sorry

end NUMINAMATH_CALUDE_euler_line_concurrency_l1079_107957


namespace NUMINAMATH_CALUDE_probability_no_red_square_l1079_107921

/-- Represents a coloring of a 4-by-4 grid -/
def Coloring := Fin 4 → Fin 4 → Bool

/-- Returns true if the coloring contains a 2-by-2 square of red squares -/
def has_red_square (c : Coloring) : Bool :=
  ∃ i j, i < 3 ∧ j < 3 ∧ 
    c i j ∧ c i (j+1) ∧ c (i+1) j ∧ c (i+1) (j+1)

/-- The probability of a square being red -/
def p_red : ℚ := 1/2

/-- The total number of possible colorings -/
def total_colorings : ℕ := 2^16

/-- The number of colorings without a 2-by-2 red square -/
def valid_colorings : ℕ := 40512

theorem probability_no_red_square :
  (valid_colorings : ℚ) / total_colorings = 315 / 512 :=
sorry

end NUMINAMATH_CALUDE_probability_no_red_square_l1079_107921


namespace NUMINAMATH_CALUDE_min_value_of_b_is_negative_two_l1079_107983

/-- The function that represents b in terms of a, where y = 2x + b is a tangent line to y = a ln x --/
noncomputable def b (a : ℝ) : ℝ := a * Real.log (a / 2) - a

/-- The theorem stating that the minimum value of b is -2 when a > 0 --/
theorem min_value_of_b_is_negative_two :
  ∀ a : ℝ, a > 0 → (∀ x : ℝ, x > 0 → b x ≥ b 2) ∧ b 2 = -2 := by sorry

end NUMINAMATH_CALUDE_min_value_of_b_is_negative_two_l1079_107983


namespace NUMINAMATH_CALUDE_probability_sum_nine_l1079_107994

/-- The number of sides on a standard die -/
def sides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The target sum we're looking for -/
def targetSum : ℕ := 9

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := sides ^ numDice

/-- The number of favorable outcomes (ways to get a sum of 9) -/
def favorableOutcomes : ℕ := 19

/-- The probability of rolling a sum of 9 with three fair, standard six-sided dice -/
theorem probability_sum_nine :
  (favorableOutcomes : ℚ) / totalOutcomes = 19 / 216 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_nine_l1079_107994


namespace NUMINAMATH_CALUDE_complex_product_real_l1079_107914

theorem complex_product_real (m : ℝ) :
  (Complex.I + 1) * (Complex.I * m + 1) ∈ Set.range Complex.ofReal → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_l1079_107914


namespace NUMINAMATH_CALUDE_boxes_with_neither_l1079_107936

/-- Given a set of boxes with markers and stickers, calculate the number of boxes
    containing neither markers nor stickers. -/
theorem boxes_with_neither (total : ℕ) (markers : ℕ) (stickers : ℕ) (both : ℕ)
    (h_total : total = 15)
    (h_markers : markers = 9)
    (h_stickers : stickers = 5)
    (h_both : both = 4) :
    total - (markers + stickers - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l1079_107936


namespace NUMINAMATH_CALUDE_sqrt_sum_product_equals_twenty_l1079_107959

theorem sqrt_sum_product_equals_twenty : (Real.sqrt 8 + Real.sqrt (1/2)) * Real.sqrt 32 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_product_equals_twenty_l1079_107959


namespace NUMINAMATH_CALUDE_interesting_trapezoid_area_interesting_trapezoid_area_range_l1079_107961

/-- An interesting isosceles trapezoid inscribed in a unit square. -/
structure InterestingTrapezoid where
  /-- Parameter determining the position of the trapezoid's vertices. -/
  a : ℝ
  /-- The parameter a is between 0 and 1/2 inclusive. -/
  h_a_range : 0 ≤ a ∧ a ≤ 1/2

/-- The vertices of the trapezoid. -/
def vertices (t : InterestingTrapezoid) : Fin 4 → ℝ × ℝ
  | 0 => (t.a, 0)
  | 1 => (1, t.a)
  | 2 => (1 - t.a, 1)
  | 3 => (0, 1 - t.a)

/-- The area of an interesting isosceles trapezoid. -/
def area (t : InterestingTrapezoid) : ℝ := 1 - 2 * t.a

/-- Theorem: The area of an interesting isosceles trapezoid is 1 - 2a. -/
theorem interesting_trapezoid_area (t : InterestingTrapezoid) :
  area t = 1 - 2 * t.a :=
by sorry

/-- Theorem: The area of an interesting isosceles trapezoid is between 0 and 1 inclusive. -/
theorem interesting_trapezoid_area_range (t : InterestingTrapezoid) :
  0 ≤ area t ∧ area t ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_interesting_trapezoid_area_interesting_trapezoid_area_range_l1079_107961


namespace NUMINAMATH_CALUDE_trig_expression_equals_seven_l1079_107981

theorem trig_expression_equals_seven :
  2 * Real.sin (390 * π / 180) - Real.tan (-45 * π / 180) + 5 * Real.cos (360 * π / 180) = 7 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_seven_l1079_107981


namespace NUMINAMATH_CALUDE_properties_dependency_l1079_107907

-- Define a type for geometric figures
inductive GeometricFigure
| Square
| Rectangle

-- Define properties for geometric figures
def hasEqualSides (f : GeometricFigure) : Prop :=
  match f with
  | GeometricFigure.Square => true
  | GeometricFigure.Rectangle => true

def hasRightAngles (f : GeometricFigure) : Prop :=
  match f with
  | GeometricFigure.Square => true
  | GeometricFigure.Rectangle => true

-- Define dependency of properties
def arePropertiesDependent (f : GeometricFigure) : Prop :=
  match f with
  | GeometricFigure.Square => hasEqualSides f ↔ hasRightAngles f
  | GeometricFigure.Rectangle => ¬(hasEqualSides f ↔ hasRightAngles f)

-- Theorem statement
theorem properties_dependency :
  arePropertiesDependent GeometricFigure.Square ∧
  ¬(arePropertiesDependent GeometricFigure.Rectangle) :=
sorry

end NUMINAMATH_CALUDE_properties_dependency_l1079_107907


namespace NUMINAMATH_CALUDE_project_duration_l1079_107976

theorem project_duration (x : ℝ) : 
  (1 / (x - 6) = 1.4 * (1 / x)) → x = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_project_duration_l1079_107976


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1079_107971

/-- Given a circle described by the equation x^2 + y^2 - 8 = 2x - 4y,
    prove that its center is at (1, -2) and its radius is √13. -/
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -2) ∧
    radius = Real.sqrt 13 ∧
    ∀ (x y : ℝ), x^2 + y^2 - 8 = 2*x - 4*y ↔
      (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1079_107971


namespace NUMINAMATH_CALUDE_new_person_weight_l1079_107966

/-- 
Given a group of 8 people where one person weighing 65 kg is replaced by a new person,
and the average weight of the group increases by 2.5 kg, prove that the weight of the new person is 85 kg.
-/
theorem new_person_weight (initial_count : ℕ) (leaving_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  leaving_weight = 65 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + leaving_weight = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1079_107966


namespace NUMINAMATH_CALUDE_power_two_greater_than_sum_of_powers_l1079_107930

theorem power_two_greater_than_sum_of_powers (n : ℕ) (x : ℝ) 
  (h1 : n ≥ 2) (h2 : |x| < 1) : 
  (2 : ℝ) ^ n > (1 - x) ^ n + (1 + x) ^ n := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_sum_of_powers_l1079_107930


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l1079_107922

theorem three_digit_number_problem :
  ∃ (A : ℕ),
    (A ≥ 100 ∧ A < 1000) ∧  -- A is a three-digit number
    (A / 100 ≠ 0 ∧ (A / 10) % 10 ≠ 0 ∧ A % 10 ≠ 0) ∧  -- A does not contain zeroes
    (∃ (B : ℕ),
      (B ≥ 10 ∧ B < 100) ∧  -- B is a two-digit number
      (B = (A / 100 + (A / 10) % 10) * 10 + A % 10) ∧  -- B is formed by summing first two digits of A
      (A = 3 * B)) ∧  -- A = 3B
    A = 135  -- The specific value of A
  := by sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l1079_107922


namespace NUMINAMATH_CALUDE_dangerous_animals_count_l1079_107916

/-- The number of crocodiles pointed out by the teacher -/
def num_crocodiles : ℕ := 22

/-- The number of alligators pointed out by the teacher -/
def num_alligators : ℕ := 23

/-- The number of vipers pointed out by the teacher -/
def num_vipers : ℕ := 5

/-- The total number of dangerous animals pointed out by the teacher -/
def total_dangerous_animals : ℕ := num_crocodiles + num_alligators + num_vipers

theorem dangerous_animals_count : total_dangerous_animals = 50 := by
  sorry

end NUMINAMATH_CALUDE_dangerous_animals_count_l1079_107916


namespace NUMINAMATH_CALUDE_catherine_bottle_caps_l1079_107989

def number_of_friends : ℕ := 6
def bottle_caps_per_friend : ℕ := 3

theorem catherine_bottle_caps : 
  number_of_friends * bottle_caps_per_friend = 18 := by
  sorry

end NUMINAMATH_CALUDE_catherine_bottle_caps_l1079_107989


namespace NUMINAMATH_CALUDE_max_value_3a_plus_b_l1079_107917

theorem max_value_3a_plus_b (a b : ℝ) (h : 9 * a^2 + b^2 - 6 * a - 2 * b = 0) :
  ∀ x y : ℝ, 9 * x^2 + y^2 - 6 * x - 2 * y = 0 → 3 * x + y ≤ 3 * a + b → 3 * a + b ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_3a_plus_b_l1079_107917


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1079_107998

/-- The quadratic equation x^2 - 2x - 6 = 0 has two distinct real roots -/
theorem quadratic_two_distinct_roots : ∃ (x₁ x₂ : ℝ), 
  x₁ ≠ x₂ ∧ 
  x₁^2 - 2*x₁ - 6 = 0 ∧ 
  x₂^2 - 2*x₂ - 6 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1079_107998


namespace NUMINAMATH_CALUDE_semicircle_pattern_area_l1079_107990

/-- The area of shaded region formed by semicircles in a pattern --/
theorem semicircle_pattern_area (d : ℝ) (l : ℝ) (h1 : d = 4) (h2 : l = 24) : 
  (l / d) / 2 * (π * (d / 2)^2) = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_semicircle_pattern_area_l1079_107990
