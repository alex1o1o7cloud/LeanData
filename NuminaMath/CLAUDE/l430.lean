import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l430_43025

theorem problem_solution (m n : ℝ) (h : 2 * m - n = 3) :
  (∀ x : ℝ, |x| + |n + 3| ≥ 9 → x ≤ -3 ∨ x ≥ 3) ∧
  (∃ min : ℝ, min = 3 ∧ ∀ x y : ℝ, 2 * x - y = 3 →
    |5/3 * x - 1/3 * y| + |1/3 * x - 2/3 * y| ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l430_43025


namespace NUMINAMATH_CALUDE_sum_of_squares_l430_43079

/-- Given a system of equations, prove that x² + y² + z² = 29 -/
theorem sum_of_squares (x y z : ℝ) 
  (eq1 : 2*x + y + 4*x*y + 6*x*z = -6)
  (eq2 : y + 2*z + 2*x*y + 6*y*z = 4)
  (eq3 : x - z + 2*x*z - 4*y*z = -3) :
  x^2 + y^2 + z^2 = 29 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l430_43079


namespace NUMINAMATH_CALUDE_problem_solution_l430_43005

theorem problem_solution :
  (∀ x : ℝ, |x + 2| + |6 - x| ≥ 8) ∧
  (∃ x : ℝ, |x + 2| + |6 - x| = 8) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 8 / (5 * a + b) + 2 / (2 * a + 3 * b) = 8 → 7 * a + 4 * b ≥ 9 / 4) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 8 / (5 * a + b) + 2 / (2 * a + 3 * b) = 8 ∧ 7 * a + 4 * b = 9 / 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l430_43005


namespace NUMINAMATH_CALUDE_total_clothing_is_934_l430_43065

/-- The number of shirts Mr. Anderson gave out -/
def shirts : ℕ := 589

/-- The number of trousers Mr. Anderson gave out -/
def trousers : ℕ := 345

/-- The total number of clothing pieces Mr. Anderson gave out -/
def total_clothing : ℕ := shirts + trousers

/-- Theorem stating that the total number of clothing pieces is 934 -/
theorem total_clothing_is_934 : total_clothing = 934 := by
  sorry

end NUMINAMATH_CALUDE_total_clothing_is_934_l430_43065


namespace NUMINAMATH_CALUDE_divisor_problem_l430_43003

theorem divisor_problem (number : ℕ) (divisor : ℕ) : 
  number = 36 →
  ((number + 10) * 2 / divisor) - 2 = 88 / 2 →
  divisor = 2 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l430_43003


namespace NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l430_43073

-- Define the equations
def equation1 (x : ℝ) : Prop := (2*x + 9) / (3 - x) = (4*x - 7) / (x - 3)
def equation2 (x : ℝ) : Prop := (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1

-- Theorem for equation1
theorem equation1_solution :
  ∃! x : ℝ, equation1 x ∧ x ≠ 3 := by sorry

-- Theorem for equation2
theorem equation2_no_solution :
  ¬∃ x : ℝ, equation2 x ∧ x ≠ 1 ∧ x ≠ -1 := by sorry

end NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l430_43073


namespace NUMINAMATH_CALUDE_fries_popcorn_ratio_is_two_to_one_l430_43092

/-- Represents the movie night scenario with Joseph and his friends -/
structure MovieNight where
  first_movie_length : ℕ
  second_movie_length : ℕ
  popcorn_time : ℕ
  total_time : ℕ

/-- Calculates the ratio of fries-making time to popcorn-making time -/
def fries_to_popcorn_ratio (mn : MovieNight) : ℚ :=
  let total_movie_time := mn.first_movie_length + mn.second_movie_length
  let fries_time := mn.total_time - total_movie_time - mn.popcorn_time
  fries_time / mn.popcorn_time

/-- Theorem stating the ratio of fries-making time to popcorn-making time is 2:1 -/
theorem fries_popcorn_ratio_is_two_to_one (mn : MovieNight)
    (h1 : mn.first_movie_length = 90)
    (h2 : mn.second_movie_length = mn.first_movie_length + 30)
    (h3 : mn.popcorn_time = 10)
    (h4 : mn.total_time = 240) :
    fries_to_popcorn_ratio mn = 2 := by
  sorry

end NUMINAMATH_CALUDE_fries_popcorn_ratio_is_two_to_one_l430_43092


namespace NUMINAMATH_CALUDE_four_char_word_count_l430_43038

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels (excluding 'Y') -/
def vowel_count : ℕ := 5

/-- The number of consonants -/
def consonant_count : ℕ := alphabet_size - vowel_count

/-- The number of four-character words formed by arranging two consonants and two vowels
    in the order consonant-vowel-vowel-consonant -/
def word_count : ℕ := consonant_count * vowel_count * vowel_count * consonant_count

theorem four_char_word_count : word_count = 11025 := by
  sorry

end NUMINAMATH_CALUDE_four_char_word_count_l430_43038


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l430_43093

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l430_43093


namespace NUMINAMATH_CALUDE_charlie_horns_l430_43033

/-- Represents the number of musical instruments owned by a person -/
structure Instruments where
  flutes : ℕ
  horns : ℕ
  harps : ℕ

/-- The problem statement -/
theorem charlie_horns (charlie carli : Instruments) : 
  charlie.flutes = 1 →
  charlie.harps = 1 →
  carli.flutes = 2 * charlie.flutes →
  carli.horns = charlie.horns / 2 →
  carli.harps = 0 →
  charlie.flutes + charlie.horns + charlie.harps + 
    carli.flutes + carli.horns + carli.harps = 7 →
  charlie.horns = 2 := by
  sorry

#check charlie_horns

end NUMINAMATH_CALUDE_charlie_horns_l430_43033


namespace NUMINAMATH_CALUDE_outfits_count_l430_43077

/-- The number of possible outfits given a set of clothing items -/
def number_of_outfits (shirts : ℕ) (ties : ℕ) (pants : ℕ) : ℕ :=
  shirts * pants * (ties + 1)

/-- Theorem stating that with 6 shirts, 4 ties, and 3 pairs of pants,
    the number of possible outfits is 90 -/
theorem outfits_count : number_of_outfits 6 4 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l430_43077


namespace NUMINAMATH_CALUDE_min_total_cost_l430_43019

/-- Represents the number of rooms of each type -/
structure RoomAllocation where
  triple : ℕ
  double : ℕ
  single : ℕ

/-- Calculates the total cost for a given room allocation -/
def totalCost (a : RoomAllocation) : ℕ :=
  300 * a.triple + 300 * a.double + 200 * a.single

/-- Checks if a room allocation is valid for the given constraints -/
def isValidAllocation (a : RoomAllocation) : Prop :=
  a.triple + a.double + a.single = 20 ∧
  3 * a.triple + 2 * a.double + a.single = 50

/-- Theorem: The minimum total cost for the given constraints is 5500 yuan -/
theorem min_total_cost :
  ∃ (a : RoomAllocation), isValidAllocation a ∧
    totalCost a = 5500 ∧
    ∀ (b : RoomAllocation), isValidAllocation b → totalCost a ≤ totalCost b :=
  sorry

end NUMINAMATH_CALUDE_min_total_cost_l430_43019


namespace NUMINAMATH_CALUDE_extreme_values_and_range_l430_43007

/-- The function f(x) with parameters a, b, and c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_and_range (a b c : ℝ) :
  (f' a b 1 = 0 ∧ f' a b 2 = 0) →
  (∀ x ∈ Set.Icc 0 3, f a b c x < c^2) →
  (a = -3 ∧ b = 4 ∧ c < -1 ∨ c > 9) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_and_range_l430_43007


namespace NUMINAMATH_CALUDE_ski_down_time_l430_43006

-- Define the lift ride time
def lift_time : ℕ := 15

-- Define the number of round trips in 2 hours
def round_trips : ℕ := 6

-- Define the total time for 6 round trips in minutes
def total_time : ℕ := 2 * 60

-- Theorem: The time to ski down the mountain is 20 minutes
theorem ski_down_time : 
  (total_time / round_trips) - lift_time = 20 :=
sorry

end NUMINAMATH_CALUDE_ski_down_time_l430_43006


namespace NUMINAMATH_CALUDE_production_days_calculation_l430_43013

theorem production_days_calculation (n : ℕ) 
  (h1 : (n * 50 : ℝ) / n = 50)  -- Average of past n days
  (h2 : ((n * 50 + 95 : ℝ) / (n + 1) = 55)) : -- New average including today
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_production_days_calculation_l430_43013


namespace NUMINAMATH_CALUDE_circle_equation_alternatives_l430_43074

/-- A circle with center on the y-axis, radius 5, passing through (3, -4) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_y_axis : center.1 = 0
  radius_is_5 : radius = 5
  passes_through_point : (center.1 - 3)^2 + (center.2 - (-4))^2 = radius^2

/-- The equation of the circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_equation_alternatives (c : Circle) :
  (∀ x y, circle_equation c x y ↔ x^2 + y^2 = 25) ∨
  (∀ x y, circle_equation c x y ↔ x^2 + (y + 8)^2 = 25) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_alternatives_l430_43074


namespace NUMINAMATH_CALUDE_line_passes_through_point_l430_43061

theorem line_passes_through_point :
  ∀ (k : ℝ), (1 + 4 * k^2) * 2 - (2 - 3 * k) * 2 + (2 - 14 * k) = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l430_43061


namespace NUMINAMATH_CALUDE_solid_is_cone_l430_43097

/-- Represents a three-dimensional solid -/
structure Solid where
  -- Add necessary fields

/-- Represents a view of a solid -/
inductive View
  | Front
  | Side
  | Top

/-- Represents a shape -/
inductive Shape
  | Cone
  | Pyramid
  | Prism
  | Cylinder

/-- Returns true if the given view of the solid is an equilateral triangle -/
def isEquilateralTriangle (s : Solid) (v : View) : Prop :=
  sorry

/-- Returns true if the given view of the solid is a circle with its center -/
def isCircleWithCenter (s : Solid) (v : View) : Prop :=
  sorry

/-- Returns true if the front and side view triangles have equal sides -/
def hasFrontSideEqualSides (s : Solid) : Prop :=
  sorry

/-- Determines the shape of the solid based on its properties -/
def determineShape (s : Solid) : Shape :=
  sorry

theorem solid_is_cone (s : Solid) 
  (h1 : isEquilateralTriangle s View.Front)
  (h2 : isEquilateralTriangle s View.Side)
  (h3 : hasFrontSideEqualSides s)
  (h4 : isCircleWithCenter s View.Top) :
  determineShape s = Shape.Cone :=
sorry

end NUMINAMATH_CALUDE_solid_is_cone_l430_43097


namespace NUMINAMATH_CALUDE_parallel_lines_l430_43089

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁ ∧ m₁ * c₂ ≠ n₁ * c₂

theorem parallel_lines (a : ℝ) :
  parallel 1 (2*a) (-1) (a-1) (-a) 1 ↔ a = 1/2 :=
sorry

#check parallel_lines

end NUMINAMATH_CALUDE_parallel_lines_l430_43089


namespace NUMINAMATH_CALUDE_certain_value_proof_l430_43004

theorem certain_value_proof (x w : ℝ) (h1 : 13 = x / (1 - w)) (h2 : w^2 = 1) : x = 26 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_proof_l430_43004


namespace NUMINAMATH_CALUDE_hyperbola_real_semiaxis_range_l430_43041

/-- The range of values for the length of the real semi-axis of a hyperbola -/
theorem hyperbola_real_semiaxis_range (a b : ℝ) (c : ℝ) :
  a > 0 →
  b > 0 →
  c = 4 →
  c^2 = a^2 + b^2 →
  b / a < Real.sqrt 3 →
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ y / x = Real.tan (60 * π / 180)) →
  2 < a ∧ a < 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_real_semiaxis_range_l430_43041


namespace NUMINAMATH_CALUDE_rectangle_area_l430_43071

theorem rectangle_area (x : ℝ) (h : x > 0) : 
  ∃ (w l : ℝ), w > 0 ∧ l > 0 ∧ l = 3 * w ∧ x^2 = w^2 + l^2 ∧ w * l = (3 * x^2) / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l430_43071


namespace NUMINAMATH_CALUDE_prime_square_sum_equation_l430_43068

theorem prime_square_sum_equation (p q : ℕ) : 
  (Prime p ∧ Prime q) → 
  (∃ (x y z : ℕ), p^(2*x) + q^(2*y) = z^2) ↔ 
  ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) := by
  sorry

#check prime_square_sum_equation

end NUMINAMATH_CALUDE_prime_square_sum_equation_l430_43068


namespace NUMINAMATH_CALUDE_prob_at_least_one_2_l430_43078

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The probability of at least one die showing 2 when two fair 8-sided dice are rolled -/
def probAtLeastOne2 : ℚ := 15 / 64

/-- Theorem stating that the probability of at least one die showing 2 
    when two fair 8-sided dice are rolled is 15/64 -/
theorem prob_at_least_one_2 : 
  probAtLeastOne2 = (numSides^2 - (numSides - 1)^2) / numSides^2 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_2_l430_43078


namespace NUMINAMATH_CALUDE_function_inequality_l430_43091

-- Define the function f and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State the theorem
theorem function_inequality (hf : ∀ x > 0, x * f' x + x^2 < f x) :
  (2 * f 1 > f 2 + 2) ∧ (3 * f 1 > f 3 + 3) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l430_43091


namespace NUMINAMATH_CALUDE_probability_woman_lawyer_l430_43039

theorem probability_woman_lawyer (total_members : ℕ) 
  (women_percentage : ℝ) (young_lawyer_percentage : ℝ) (old_lawyer_percentage : ℝ) 
  (h1 : women_percentage = 0.4)
  (h2 : young_lawyer_percentage = 0.3)
  (h3 : old_lawyer_percentage = 0.1)
  (h4 : young_lawyer_percentage + old_lawyer_percentage + 0.6 = 1) :
  (women_percentage * (young_lawyer_percentage + old_lawyer_percentage)) = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_probability_woman_lawyer_l430_43039


namespace NUMINAMATH_CALUDE_eight_digit_numbers_with_consecutive_digits_l430_43001

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Total number of 8-digit numbers with digits 1 or 2 -/
def total_numbers : ℕ := 2^8

/-- Number of 8-digit numbers with no consecutive same digits -/
def numbers_without_consecutive : ℕ := 2 * fib 7

theorem eight_digit_numbers_with_consecutive_digits : 
  total_numbers - numbers_without_consecutive = 230 := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_numbers_with_consecutive_digits_l430_43001


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l430_43036

/-- Represents a number as a sequence of digits in base 10 -/
def DigitSequence (d : Nat) (n : Nat) : Nat :=
  (10^n - 1) / 9 * d

/-- Calculates the sum of digits of a number in base 10 -/
def sumOfDigits (n : Nat) : Nat :=
  sorry

theorem sum_of_digits_9ab :
  let a := DigitSequence 9 2023
  let b := DigitSequence 6 2023
  sumOfDigits (9 * a * b) = 28314 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l430_43036


namespace NUMINAMATH_CALUDE_min_teachers_for_our_school_l430_43054

/-- Represents a school with math, physics, and chemistry teachers -/
structure School where
  mathTeachers : ℕ
  physicsTeachers : ℕ
  chemistryTeachers : ℕ
  maxSubjectsPerTeacher : ℕ

/-- The minimum number of teachers required for a given school -/
def minTeachersRequired (s : School) : ℕ := sorry

/-- Our specific school configuration -/
def ourSchool : School :=
  { mathTeachers := 4
    physicsTeachers := 3
    chemistryTeachers := 3
    maxSubjectsPerTeacher := 2 }

/-- Theorem stating that the minimum number of teachers required for our school is 6 -/
theorem min_teachers_for_our_school :
  minTeachersRequired ourSchool = 6 := by sorry

end NUMINAMATH_CALUDE_min_teachers_for_our_school_l430_43054


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l430_43047

/-- The roots of the quadratic equation x^2 - 7x + 10 = 0 -/
def roots : Set ℝ := {x : ℝ | x^2 - 7*x + 10 = 0}

/-- An isosceles triangle with two sides equal to the roots of x^2 - 7x + 10 = 0 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : a = b ∨ b = c ∨ a = c
  rootSides : a ∈ roots ∧ b ∈ roots

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

/-- Theorem: The perimeter of the isosceles triangle is 12 -/
theorem isosceles_triangle_perimeter : 
  ∀ t : IsoscelesTriangle, perimeter t = 12 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l430_43047


namespace NUMINAMATH_CALUDE_inequality_proof_l430_43072

theorem inequality_proof (a b c d : ℝ) (n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hn : n ≥ 9) :
  a^n + b^n + c^n + d^n ≥ 
  a^(n-9) * b^4 * c^3 * d^2 + 
  b^(n-9) * c^4 * d^3 * a^2 + 
  c^(n-9) * d^4 * a^3 * b^2 + 
  d^(n-9) * a^4 * b^3 * c^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l430_43072


namespace NUMINAMATH_CALUDE_total_cost_is_180_l430_43066

/-- The cost to fill all planter pots at the corners of a rectangle-shaped pool -/
def total_cost : ℝ :=
  let corners_of_rectangle := 4
  let palm_fern_cost := 15.00
  let creeping_jenny_cost := 4.00
  let geranium_cost := 3.50
  let palm_ferns_per_pot := 1
  let creeping_jennies_per_pot := 4
  let geraniums_per_pot := 4
  let cost_per_pot := palm_fern_cost * palm_ferns_per_pot + 
                      creeping_jenny_cost * creeping_jennies_per_pot + 
                      geranium_cost * geraniums_per_pot
  corners_of_rectangle * cost_per_pot

/-- Theorem stating that the total cost to fill all planter pots is $180.00 -/
theorem total_cost_is_180 : total_cost = 180.00 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_180_l430_43066


namespace NUMINAMATH_CALUDE_quadratic_root_value_l430_43042

theorem quadratic_root_value (k : ℝ) : 
  (∀ x : ℂ, 5 * x^2 + 7 * x + k = 0 ↔ x = Complex.mk (-7/10) ((Real.sqrt 399)/10) ∨ x = Complex.mk (-7/10) (-(Real.sqrt 399)/10)) →
  k = 22.4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l430_43042


namespace NUMINAMATH_CALUDE_pears_juice_calculation_l430_43064

/-- The amount of pears processed into juice given a total harvest and export percentage -/
def pears_processed_into_juice (total_harvest : ℝ) (export_percentage : ℝ) (juice_percentage : ℝ) : ℝ :=
  total_harvest * (1 - export_percentage) * juice_percentage

theorem pears_juice_calculation (total_harvest : ℝ) (export_percentage : ℝ) (juice_percentage : ℝ) 
  (h1 : total_harvest = 8.5)
  (h2 : export_percentage = 0.3)
  (h3 : juice_percentage = 0.6) :
  pears_processed_into_juice total_harvest export_percentage juice_percentage = 3.57 := by
  sorry

#eval pears_processed_into_juice 8.5 0.3 0.6

end NUMINAMATH_CALUDE_pears_juice_calculation_l430_43064


namespace NUMINAMATH_CALUDE_olivia_supermarket_spending_l430_43024

theorem olivia_supermarket_spending (initial_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 54)
  (h2 : remaining_amount = 29) :
  initial_amount - remaining_amount = 25 := by
sorry

end NUMINAMATH_CALUDE_olivia_supermarket_spending_l430_43024


namespace NUMINAMATH_CALUDE_valid_word_count_l430_43080

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'I'}
def vowels : Finset Char := {'A', 'E', 'I'}
def consonants : Finset Char := {'B', 'C', 'D'}

def is_valid_word (word : List Char) : Prop :=
  word.length = 5 ∧ word.toFinset ⊆ letters ∧ (∃ c ∈ word, c ∈ consonants)

def count_valid_words : ℕ := (letters.powerset.filter (λ s => s.card = 5)).card

theorem valid_word_count : count_valid_words = 7533 := by
  sorry

end NUMINAMATH_CALUDE_valid_word_count_l430_43080


namespace NUMINAMATH_CALUDE_x_equals_plus_minus_fifteen_l430_43043

theorem x_equals_plus_minus_fifteen (x : ℝ) : 
  (x / 5) / 3 = 5 / (x / 3) → x = 15 ∨ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_plus_minus_fifteen_l430_43043


namespace NUMINAMATH_CALUDE_billy_ticket_usage_l430_43062

/-- The number of times Billy rode the ferris wheel -/
def ferris_rides : ℕ := 7

/-- The number of times Billy rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost in tickets for each ride -/
def tickets_per_ride : ℕ := 5

/-- The total number of tickets Billy used -/
def total_tickets : ℕ := (ferris_rides + bumper_rides) * tickets_per_ride

theorem billy_ticket_usage : total_tickets = 50 := by
  sorry

end NUMINAMATH_CALUDE_billy_ticket_usage_l430_43062


namespace NUMINAMATH_CALUDE_sum_of_squares_power_l430_43021

theorem sum_of_squares_power (a b n : ℕ+) : ∃ x y : ℤ, (a.val ^ 2 + b.val ^ 2) ^ n.val = x ^ 2 + y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_power_l430_43021


namespace NUMINAMATH_CALUDE_relay_race_probability_l430_43087

/-- The number of short-distance runners --/
def total_runners : ℕ := 6

/-- The number of runners needed for the relay race --/
def team_size : ℕ := 4

/-- The probability that athlete A is not running the first leg
    and athlete B is not running the last leg in a 4x100 meter relay race --/
theorem relay_race_probability : 
  (total_runners.factorial / (total_runners - team_size).factorial - 
   (total_runners - 1).factorial / (total_runners - team_size).factorial - 
   (total_runners - 1).factorial / (total_runners - team_size).factorial + 
   (total_runners - 2).factorial / (total_runners - team_size + 1).factorial) /
  (total_runners.factorial / (total_runners - team_size).factorial) = 7 / 10 := by
sorry

end NUMINAMATH_CALUDE_relay_race_probability_l430_43087


namespace NUMINAMATH_CALUDE_eccentricity_difference_l430_43012

/-- Given an ellipse and a hyperbola sharing the same foci, prove that
    the difference of their eccentricities is √2 under certain conditions. -/
theorem eccentricity_difference (a b m n : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) (h4 : n > 0) :
  let C₁ := {(x, y) : ℝ × ℝ | x^2/a^2 + y^2/b^2 = 1}
  let C₂ := {(x, y) : ℝ × ℝ | x^2/m^2 - y^2/n^2 = 1}
  let c := Real.sqrt (a^2 - b^2)
  let f := Real.sqrt (m^2 + n^2)
  let e₁ := c / a
  let e₂ := f / m
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  ∃ P ∈ C₁ ∩ C₂, P.1 > 0 ∧ P.2 > 0 ∧ 
  c = f ∧ 
  dist P F₁ = dist P F₂ ∧
  dist P F₁ = dist F₁ F₂ →
  e₂ - e₁ = Real.sqrt 2 :=
sorry


end NUMINAMATH_CALUDE_eccentricity_difference_l430_43012


namespace NUMINAMATH_CALUDE_line_reflection_x_axis_l430_43055

/-- Given a line with equation x - y + 1 = 0, its reflection with respect to the x-axis has the equation x + y + 1 = 0 -/
theorem line_reflection_x_axis :
  let original_line := {(x, y) : ℝ × ℝ | x - y + 1 = 0}
  let reflected_line := {(x, y) : ℝ × ℝ | x + y + 1 = 0}
  (∀ (x y : ℝ), (x, y) ∈ original_line ↔ (x, -y) ∈ reflected_line) :=
by sorry

end NUMINAMATH_CALUDE_line_reflection_x_axis_l430_43055


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l430_43020

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- Theorem statement
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (g ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l430_43020


namespace NUMINAMATH_CALUDE_boys_without_calculators_l430_43046

theorem boys_without_calculators (total_boys : ℕ) (students_with_calc : ℕ) (girls_with_calc : ℕ) (forgot_calc : ℕ) :
  total_boys = 20 →
  students_with_calc = 26 →
  girls_with_calc = 15 →
  forgot_calc = 3 →
  total_boys - (students_with_calc - girls_with_calc + (forgot_calc * total_boys) / (students_with_calc + forgot_calc)) = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_boys_without_calculators_l430_43046


namespace NUMINAMATH_CALUDE_triangle_sides_proportion_l430_43086

/-- Represents a triangle with sides a, b, c and incircle diameter 2r --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ

/-- 
  Theorem: If the lengths of the sides of a triangle and the diameter of its incircle 
  form four consecutive terms of an arithmetic progression, then the sides of the 
  triangle are proportional to 3, 4, and 5.
--/
theorem triangle_sides_proportion (t : Triangle) : 
  (∃ (d : ℝ), d > 0 ∧ t.a = t.r * 2 + d ∧ t.b = t.r * 2 + 2 * d ∧ t.c = t.r * 2 + 3 * d) →
  ∃ (k : ℝ), k > 0 ∧ t.a = 3 * k ∧ t.b = 4 * k ∧ t.c = 5 * k :=
by sorry

end NUMINAMATH_CALUDE_triangle_sides_proportion_l430_43086


namespace NUMINAMATH_CALUDE_square_difference_to_fourth_power_l430_43049

theorem square_difference_to_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_to_fourth_power_l430_43049


namespace NUMINAMATH_CALUDE_polynomial_value_at_n_plus_one_l430_43044

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℚ :=
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the polynomial P
noncomputable def P (n : ℕ) (x : ℚ) : ℚ :=
  sorry  -- The actual definition is not provided in the problem statement

-- State the theorem
theorem polynomial_value_at_n_plus_one (n : ℕ) :
  (∀ k : ℕ, k ≤ n → P n k = 1 / binomial n k) →
  P n (n + 1) = if n % 2 = 0 then 1 else 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_n_plus_one_l430_43044


namespace NUMINAMATH_CALUDE_summer_lecture_team_selection_probability_l430_43027

/-- Represents the probability of a teacher being selected for the summer lecture team -/
def selection_probability (total : ℕ) (eliminated : ℕ) (team_size : ℕ) : ℚ :=
  team_size / (total - eliminated)

theorem summer_lecture_team_selection_probability :
  selection_probability 118 6 16 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_summer_lecture_team_selection_probability_l430_43027


namespace NUMINAMATH_CALUDE_minimum_value_expression_minimum_value_achievable_l430_43016

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (5 * c) / (a + b) + (5 * a) / (b + c) + (3 * b) / (a + c) + 1 ≥ 7.25 :=
by sorry

theorem minimum_value_achievable :
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧
    (5 * c) / (a + b) + (5 * a) / (b + c) + (3 * b) / (a + c) + 1 = 7.25 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_expression_minimum_value_achievable_l430_43016


namespace NUMINAMATH_CALUDE_tennis_players_count_l430_43094

theorem tennis_players_count (total_members : ℕ) (badminton_players : ℕ) (both_players : ℕ) (neither_players : ℕ) :
  total_members = 30 →
  badminton_players = 16 →
  both_players = 7 →
  neither_players = 2 →
  ∃ (tennis_players : ℕ), tennis_players = 19 ∧
    tennis_players = total_members - neither_players - (badminton_players - both_players) := by
  sorry


end NUMINAMATH_CALUDE_tennis_players_count_l430_43094


namespace NUMINAMATH_CALUDE_total_camp_attendance_l430_43095

/-- The number of kids from Lawrence county who went to camp -/
def lawrence_camp : ℕ := 34044

/-- The number of kids from outside the county who attended the camp -/
def outside_camp : ℕ := 424944

/-- The total number of kids who attended the camp -/
def total_camp : ℕ := lawrence_camp + outside_camp

/-- Theorem stating that the total number of kids who attended the camp is 458988 -/
theorem total_camp_attendance : total_camp = 458988 := by
  sorry

end NUMINAMATH_CALUDE_total_camp_attendance_l430_43095


namespace NUMINAMATH_CALUDE_equation_solution_l430_43063

theorem equation_solution : ∃! x : ℚ, (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) ∧ x = -19/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l430_43063


namespace NUMINAMATH_CALUDE_triangle_equal_sides_l430_43053

/-- 
Given a triangle with one side of length 6 cm and two equal sides,
where the sum of all sides is 20 cm, prove that each of the equal sides is 7 cm long.
-/
theorem triangle_equal_sides (a b c : ℝ) : 
  a = 6 → -- One side is 6 cm
  b = c → -- Two sides are equal
  a + b + c = 20 → -- Sum of all sides is 20 cm
  b = 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_equal_sides_l430_43053


namespace NUMINAMATH_CALUDE_fraction_equality_l430_43084

theorem fraction_equality (a b c d : ℝ) 
  (h1 : a / b = 4)
  (h2 : b / c = 1 / 3)
  (h3 : c / d = 6) :
  d / a = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l430_43084


namespace NUMINAMATH_CALUDE_milk_cost_is_1_15_l430_43023

/-- The cost of Anna's breakfast items and lunch sandwich, and the difference between lunch and breakfast costs -/
structure AnnasMeals where
  bagel_cost : ℚ
  juice_cost : ℚ
  sandwich_cost : ℚ
  lunch_breakfast_diff : ℚ

/-- Calculate the cost of the milk carton based on Anna's meal expenses -/
def milk_cost (meals : AnnasMeals) : ℚ :=
  let breakfast_cost := meals.bagel_cost + meals.juice_cost
  let lunch_cost := breakfast_cost + meals.lunch_breakfast_diff
  lunch_cost - meals.sandwich_cost

/-- Theorem stating that the cost of the milk carton is $1.15 -/
theorem milk_cost_is_1_15 (meals : AnnasMeals) 
  (h1 : meals.bagel_cost = 95/100)
  (h2 : meals.juice_cost = 85/100)
  (h3 : meals.sandwich_cost = 465/100)
  (h4 : meals.lunch_breakfast_diff = 4) :
  milk_cost meals = 115/100 := by
  sorry

end NUMINAMATH_CALUDE_milk_cost_is_1_15_l430_43023


namespace NUMINAMATH_CALUDE_range_of_m_solution_set_l430_43099

-- Define the function f
def f (x : ℝ) : ℝ := |x - 5| - |x - 2|

-- Theorem for the range of m
theorem range_of_m : 
  {m : ℝ | ∃ x, f x ≤ m} = {m : ℝ | m ≥ -3} := by sorry

-- Theorem for the solution set of the inequality
theorem solution_set : 
  {x : ℝ | x^2 - 8*x + 15 + f x ≤ 0} = {x : ℝ | 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6} := by sorry

end NUMINAMATH_CALUDE_range_of_m_solution_set_l430_43099


namespace NUMINAMATH_CALUDE_f_properties_l430_43000

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - a)^2 * (x + b) * Real.exp x

-- Define the first derivative of f
def f' (a b x : ℝ) : ℝ := Real.exp x * (x - a) * (x^2 + (3 - a + b) * x + 2 * b - a * b - a)

-- State the theorem
theorem f_properties (a b : ℝ) :
  (∀ x, f' a b x ≤ f' a b a) →
  ((a = 0 → b < 0) ∧
   (∃ x₄, (x₄ = a + 2 * Real.sqrt 6 ∨ x₄ = a - 2 * Real.sqrt 6) ∧ b = -a - 3) ∨
   (∃ x₄, x₄ = a + (1 + Real.sqrt 13) / 2 ∧ b = -a - (7 + Real.sqrt 13) / 2) ∨
   (∃ x₄, x₄ = a + (1 - Real.sqrt 13) / 2 ∧ b = -a - (7 - Real.sqrt 13) / 2)) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l430_43000


namespace NUMINAMATH_CALUDE_product_equals_quadratic_l430_43083

theorem product_equals_quadratic : ∃ m : ℤ, 72516 * 9999 = m^2 - 5*m + 7 ∧ m = 26926 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_quadratic_l430_43083


namespace NUMINAMATH_CALUDE_driving_distance_differences_l430_43058

/-- Represents the driving scenario with Ian, Han, and Jan -/
structure DrivingScenario where
  ian_time : ℝ
  ian_speed : ℝ
  han_extra_time : ℝ := 2
  han_extra_speed : ℝ := 10
  jan_extra_time : ℝ := 3
  jan_extra_speed : ℝ := 15
  han_extra_distance : ℝ := 100

/-- Calculate the distance driven by Ian -/
def ian_distance (scenario : DrivingScenario) : ℝ :=
  scenario.ian_time * scenario.ian_speed

/-- Calculate the distance driven by Han -/
def han_distance (scenario : DrivingScenario) : ℝ :=
  (scenario.ian_time + scenario.han_extra_time) * (scenario.ian_speed + scenario.han_extra_speed)

/-- Calculate the distance driven by Jan -/
def jan_distance (scenario : DrivingScenario) : ℝ :=
  (scenario.ian_time + scenario.jan_extra_time) * (scenario.ian_speed + scenario.jan_extra_speed)

/-- The main theorem stating the differences in distances driven -/
theorem driving_distance_differences (scenario : DrivingScenario) :
  jan_distance scenario - ian_distance scenario = 150 ∧
  jan_distance scenario - han_distance scenario = 150 := by
  sorry


end NUMINAMATH_CALUDE_driving_distance_differences_l430_43058


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l430_43010

/-- The set of possible slopes for a line with y-intercept (0, -3) that intersects the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m^2 ≥ 1/20}

/-- Theorem stating the condition for the line to intersect the ellipse -/
theorem line_intersects_ellipse (m : ℝ) :
  (∃ x y : ℝ, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ m ∈ possible_slopes := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l430_43010


namespace NUMINAMATH_CALUDE_icosahedron_edge_probability_l430_43031

/-- A regular icosahedron -/
structure Icosahedron :=
  (vertices : Nat)
  (edges : Nat)
  (is_regular : vertices = 12 ∧ edges = 30)

/-- The probability of selecting two vertices that form an edge in a regular icosahedron -/
def edge_probability (i : Icosahedron) : ℚ :=
  i.edges / (i.vertices.choose 2)

/-- Theorem stating that the probability of randomly selecting two vertices 
    of a regular icosahedron that form an edge is 5/11 -/
theorem icosahedron_edge_probability :
  ∀ i : Icosahedron, edge_probability i = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_icosahedron_edge_probability_l430_43031


namespace NUMINAMATH_CALUDE_lucy_balance_l430_43026

/-- Calculates the final balance after a deposit and withdrawal --/
def final_balance (initial : ℕ) (deposit : ℕ) (withdrawal : ℕ) : ℕ :=
  initial + deposit - withdrawal

/-- Proves that Lucy's final balance is $76 --/
theorem lucy_balance : final_balance 65 15 4 = 76 := by
  sorry

end NUMINAMATH_CALUDE_lucy_balance_l430_43026


namespace NUMINAMATH_CALUDE_min_value_expression_l430_43075

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (ha' : a < 2) (hb' : b < 2) (hc' : c < 2) :
  (1 / ((2 - a) * (2 - b) * (2 - c))) + (1 / ((2 + a) * (2 + b) * (2 + c))) ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l430_43075


namespace NUMINAMATH_CALUDE_ellipse_condition_l430_43011

def is_ellipse (k : ℝ) : Prop :=
  1 < k ∧ k < 5 ∧ k ≠ 3

theorem ellipse_condition (k : ℝ) :
  (∀ x y : ℝ, x^2 / (k - 1) + y^2 / (5 - k) = 1 → is_ellipse k) ∧
  ¬(∀ k : ℝ, 1 < k ∧ k < 5 → is_ellipse k) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l430_43011


namespace NUMINAMATH_CALUDE_tangent_triangle_angles_correct_l430_43014

structure Triangle where
  α : Real
  β : Real
  γ : Real
  sum_angles : α + β + γ = Real.pi
  not_right : α ≠ Real.pi/2 ∧ β ≠ Real.pi/2 ∧ γ ≠ Real.pi/2

def tangent_triangle_angles (t : Triangle) : Set Real :=
  if t.α < Real.pi/2 ∧ t.β < Real.pi/2 ∧ t.γ < Real.pi/2 then
    {Real.pi - 2*t.α, Real.pi - 2*t.β, Real.pi - 2*t.γ}
  else
    {2*t.α - Real.pi, 2*t.γ, 2*t.β}

theorem tangent_triangle_angles_correct (t : Triangle) :
  ∃ (a b c : Real), tangent_triangle_angles t = {a, b, c} ∧ a + b + c = Real.pi :=
sorry

end NUMINAMATH_CALUDE_tangent_triangle_angles_correct_l430_43014


namespace NUMINAMATH_CALUDE_triangle_angle_at_least_60_degrees_l430_43002

theorem triangle_angle_at_least_60_degrees (A B C : ℝ) :
  A + B + C = 180 → A > 0 → B > 0 → C > 0 → (A ≥ 60 ∨ B ≥ 60 ∨ C ≥ 60) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_at_least_60_degrees_l430_43002


namespace NUMINAMATH_CALUDE_santiago_roses_count_l430_43081

/-- The number of red roses Mrs. Garrett has -/
def garrett_roses : ℕ := 24

/-- The number of additional roses Mrs. Santiago has compared to Mrs. Garrett -/
def additional_roses : ℕ := 34

/-- The number of red roses Mrs. Santiago has -/
def santiago_roses : ℕ := garrett_roses + additional_roses

theorem santiago_roses_count : santiago_roses = 58 := by
  sorry

end NUMINAMATH_CALUDE_santiago_roses_count_l430_43081


namespace NUMINAMATH_CALUDE_point_P_coordinates_l430_43059

def P (m : ℝ) : ℝ × ℝ := (m + 3, 2*m - 1)

theorem point_P_coordinates :
  (∀ m : ℝ, P m = (0, -7) ↔ (P m).1 = 0) ∧
  (∀ m : ℝ, P m = (10, 13) ↔ (P m).2 = (P m).1 + 3) ∧
  (∀ m : ℝ, P m = (5/2, -2) ↔ |(P m).2| = 2 ∧ (P m).1 > 0 ∧ (P m).2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l430_43059


namespace NUMINAMATH_CALUDE_additional_blue_tickets_for_bible_l430_43057

/-- Represents the number of tickets Tom has of each color -/
structure TicketCounts where
  yellow : ℕ
  red : ℕ
  green : ℕ
  blue : ℕ

/-- Represents the conversion rates between ticket colors -/
structure TicketRates where
  yellow_to_red : ℕ
  red_to_green : ℕ
  green_to_blue : ℕ

def calculate_additional_blue_tickets (
  bible_yellow_requirement : ℕ
  ) (rates : TicketRates) (current_tickets : TicketCounts) : ℕ :=
  sorry

theorem additional_blue_tickets_for_bible (
  bible_yellow_requirement : ℕ
  ) (rates : TicketRates) (current_tickets : TicketCounts) :
  bible_yellow_requirement = 20 →
  rates.yellow_to_red = 15 →
  rates.red_to_green = 12 →
  rates.green_to_blue = 10 →
  current_tickets.yellow = 12 →
  current_tickets.red = 8 →
  current_tickets.green = 14 →
  current_tickets.blue = 27 →
  calculate_additional_blue_tickets bible_yellow_requirement rates current_tickets = 13273 :=
by sorry

end NUMINAMATH_CALUDE_additional_blue_tickets_for_bible_l430_43057


namespace NUMINAMATH_CALUDE_diamond_ruby_difference_l430_43035

theorem diamond_ruby_difference (d r : ℕ) (h1 : d = 3 * r) : d - r = 2 * r := by
  sorry

end NUMINAMATH_CALUDE_diamond_ruby_difference_l430_43035


namespace NUMINAMATH_CALUDE_bruce_payment_l430_43096

/-- The amount Bruce paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Bruce paid 1125 to the shopkeeper -/
theorem bruce_payment :
  total_amount 9 70 9 55 = 1125 := by
  sorry

end NUMINAMATH_CALUDE_bruce_payment_l430_43096


namespace NUMINAMATH_CALUDE_quadratic_function_comparison_l430_43048

theorem quadratic_function_comparison : ∀ (y₁ y₂ : ℝ),
  y₁ = -(1:ℝ)^2 + 2 →
  y₂ = -(3:ℝ)^2 + 2 →
  y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_comparison_l430_43048


namespace NUMINAMATH_CALUDE_rate_squares_sum_l430_43082

theorem rate_squares_sum : ∀ (b j s : ℕ),
  (3 * b + 4 * j + 2 * s = 86) →
  (5 * b + 2 * j + 4 * s = 110) →
  (b * b + j * j + s * s = 3349) :=
by sorry

end NUMINAMATH_CALUDE_rate_squares_sum_l430_43082


namespace NUMINAMATH_CALUDE_a_2_times_a_3_l430_43009

def a : ℕ → ℤ
  | n => if n % 2 = 1 then 3 * n + 1 else 2 * n - 2

theorem a_2_times_a_3 : a 2 * a 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_a_2_times_a_3_l430_43009


namespace NUMINAMATH_CALUDE_stone_137_is_5_l430_43017

/-- Represents the number of stones in the sequence -/
def num_stones : ℕ := 11

/-- Represents the length of a full counting cycle -/
def cycle_length : ℕ := 20

/-- Represents the target count number -/
def target_count : ℕ := 137

/-- Represents the original stone number we want to prove -/
def original_stone : ℕ := 5

/-- Function to determine the stone number given a count in the sequence -/
def stone_at_count (count : ℕ) : ℕ :=
  sorry

theorem stone_137_is_5 : stone_at_count target_count = original_stone := by
  sorry

end NUMINAMATH_CALUDE_stone_137_is_5_l430_43017


namespace NUMINAMATH_CALUDE_sugar_sacks_weight_difference_l430_43008

theorem sugar_sacks_weight_difference (x y : ℝ) : 
  x + y = 40 →
  x - 1 = 0.6 * (y + 1) →
  |x - y| = 8 := by
sorry

end NUMINAMATH_CALUDE_sugar_sacks_weight_difference_l430_43008


namespace NUMINAMATH_CALUDE_max_value_of_expression_l430_43070

def numbers : Finset ℕ := {12, 14, 16, 18}

def expression (A B C D : ℕ) : ℕ := A * B + B * C + B * D + C * D

theorem max_value_of_expression :
  ∃ (A B C D : ℕ), A ∈ numbers ∧ B ∈ numbers ∧ C ∈ numbers ∧ D ∈ numbers ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  expression A B C D = 1116 ∧
  ∀ (A' B' C' D' : ℕ), A' ∈ numbers → B' ∈ numbers → C' ∈ numbers → D' ∈ numbers →
  A' ≠ B' → A' ≠ C' → A' ≠ D' → B' ≠ C' → B' ≠ D' → C' ≠ D' →
  expression A' B' C' D' ≤ 1116 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l430_43070


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l430_43018

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_a2 : a 2 = 9)
  (h_a4 : a 4 = 4) :
  ∀ n : ℕ, a n = 9 * (2/3)^(n - 2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l430_43018


namespace NUMINAMATH_CALUDE_find_b_plus_c_l430_43088

theorem find_b_plus_c (a b c d : ℚ) 
  (h1 : a * b + a * c + b * d + c * d = 40) 
  (h2 : a + d = 6) : 
  b + c = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_find_b_plus_c_l430_43088


namespace NUMINAMATH_CALUDE_problem_statement_l430_43028

theorem problem_statement (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : a * b^2 = c/a - b) : 
  ((a^2 * b^2 / c^2) - (2/c) + (1/(a^2 * b^2)) + (2*a*b / c^2) - (2/(a*b*c))) / 
  ((2/(a*b)) - (2*a*b/c)) / (101/c) = -1/202 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l430_43028


namespace NUMINAMATH_CALUDE_parallel_condition_l430_43051

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m₁ n₁ : ℝ) (m₂ n₂ : ℝ) : Prop := m₁ * n₂ = m₂ * n₁

/-- Definition of line l₁ -/
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - 1 = 0

/-- Definition of line l₂ -/
def l₂ (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

theorem parallel_condition (a : ℝ) :
  (a = 1 → are_parallel a 2 1 (a + 1)) ∧
  (∃ b : ℝ, b ≠ 1 ∧ are_parallel b 2 1 (b + 1)) :=
by sorry

end NUMINAMATH_CALUDE_parallel_condition_l430_43051


namespace NUMINAMATH_CALUDE_skirt_cost_l430_43034

/-- Calculates the cost of each skirt in Marcia's wardrobe purchase --/
theorem skirt_cost (num_skirts num_pants num_blouses : ℕ)
  (blouse_price pant_price total_spend : ℚ) :
  num_skirts = 3 →
  num_pants = 2 →
  num_blouses = 5 →
  blouse_price = 15 →
  pant_price = 30 →
  total_spend = 180 →
  (total_spend - (num_blouses * blouse_price + pant_price * 1.5)) / num_skirts = 20 := by
  sorry

end NUMINAMATH_CALUDE_skirt_cost_l430_43034


namespace NUMINAMATH_CALUDE_square_intersection_perimeter_l430_43015

/-- Given a square with side length 2a and a line y = -x/3 intersecting it,
    the perimeter of one resulting quadrilateral divided by a is (8 + 2√10) / 3 -/
theorem square_intersection_perimeter (a : ℝ) (a_pos : a > 0) :
  let square_vertices := [(-a, -a), (a, -a), (-a, a), (a, a)]
  let intersecting_line (x : ℝ) := -x/3
  let intersection_points := [(-a, a/3), (a, -a/3)]
  let quadrilateral_vertices := [(-a, a), (-a, a/3), (a, -a/3), (a, -a)]
  let perimeter := 
    2 * (a - a/3) +  -- vertical sides
    2 * a +          -- horizontal side
    Real.sqrt ((2*a)^2 + (2*a/3)^2)  -- diagonal
  perimeter / a = (8 + 2 * Real.sqrt 10) / 3 := by
  sorry


end NUMINAMATH_CALUDE_square_intersection_perimeter_l430_43015


namespace NUMINAMATH_CALUDE_complex_parts_of_1_plus_sqrt3_i_l430_43032

theorem complex_parts_of_1_plus_sqrt3_i : 
  let z : ℂ := Complex.I * (1 + Real.sqrt 3)
  (z.re = 0) ∧ (z.im = 1 + Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_complex_parts_of_1_plus_sqrt3_i_l430_43032


namespace NUMINAMATH_CALUDE_phi_equality_iff_in_solution_set_l430_43022

/-- Euler's totient function -/
def phi (n : ℕ+) : ℕ := sorry

/-- The set of solutions to the equation φ(2019n) = φ(n²) -/
def solution_set : Set ℕ+ := {1346, 2016, 2019}

/-- Theorem stating that n satisfies φ(2019n) = φ(n²) if and only if n is in the solution set -/
theorem phi_equality_iff_in_solution_set (n : ℕ+) : 
  phi (2019 * n) = phi (n * n) ↔ n ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_phi_equality_iff_in_solution_set_l430_43022


namespace NUMINAMATH_CALUDE_billy_watched_79_videos_l430_43090

/-- The number of videos Billy watches before finding one he likes -/
def total_videos_watched (suggestions_per_attempt : ℕ) (unsuccessful_attempts : ℕ) (position_of_liked_video : ℕ) : ℕ :=
  suggestions_per_attempt * unsuccessful_attempts + (position_of_liked_video - 1)

/-- Theorem stating that Billy watches 79 videos before finding one he likes -/
theorem billy_watched_79_videos :
  total_videos_watched 15 5 5 = 79 := by
sorry

end NUMINAMATH_CALUDE_billy_watched_79_videos_l430_43090


namespace NUMINAMATH_CALUDE_final_inventory_calculation_l430_43067

def initial_inventory : ℕ := 4500
def monday_sales : ℕ := 2445
def tuesday_sales : ℕ := 900
def daily_sales_wed_to_sun : ℕ := 50
def days_wed_to_sun : ℕ := 5
def saturday_delivery : ℕ := 650

theorem final_inventory_calculation :
  initial_inventory - 
  (monday_sales + tuesday_sales + daily_sales_wed_to_sun * days_wed_to_sun) + 
  saturday_delivery = 1555 := by
  sorry

end NUMINAMATH_CALUDE_final_inventory_calculation_l430_43067


namespace NUMINAMATH_CALUDE_sphere_volume_surface_area_relation_l430_43045

theorem sphere_volume_surface_area_relation (r : ℝ) : 
  (4 / 3 * Real.pi * r^3) = 2 * (4 * Real.pi * r^2) → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_surface_area_relation_l430_43045


namespace NUMINAMATH_CALUDE_equal_cubic_expressions_l430_43069

theorem equal_cubic_expressions (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 3)
  (sum_squares_eq : a^2 + b^2 + c^2 + d^2 = 3)
  (sum_triple_products_eq : a*b*c + b*c*d + c*d*a + d*a*b = 1) :
  a*(1-a)^3 = b*(1-b)^3 ∧ b*(1-b)^3 = c*(1-c)^3 ∧ c*(1-c)^3 = d*(1-d)^3 := by
  sorry

end NUMINAMATH_CALUDE_equal_cubic_expressions_l430_43069


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l430_43060

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ y : ℝ, (3 - 9 * I) * (a + b * I) = y * I) : a / b = -3 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l430_43060


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l430_43076

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l430_43076


namespace NUMINAMATH_CALUDE_impossible_coloring_exists_l430_43052

-- Define the grid
def Grid := ℤ × ℤ

-- Define a chessboard polygon
def ChessboardPolygon := Set Grid

-- Define a coloring of the grid
def Coloring := Grid → Bool

-- Define congruence for chessboard polygons
def Congruent (F G : ChessboardPolygon) : Prop := sorry

-- Define the number of green cells in a polygon given a coloring
def GreenCells (F : ChessboardPolygon) (c : Coloring) : ℕ := sorry

-- The main theorem
theorem impossible_coloring_exists :
  ∃ F : ChessboardPolygon,
    ∀ c : Coloring,
      (∃ G : ChessboardPolygon, Congruent F G ∧ GreenCells G c = 0) ∨
      (∃ G : ChessboardPolygon, Congruent F G ∧ GreenCells G c > 2020) := by
  sorry

end NUMINAMATH_CALUDE_impossible_coloring_exists_l430_43052


namespace NUMINAMATH_CALUDE_value_of_expression_l430_43030

theorem value_of_expression : 70 * Real.sqrt ((8^10 + 4^10) / (8^4 + 4^11)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l430_43030


namespace NUMINAMATH_CALUDE_family_game_score_l430_43040

theorem family_game_score : 
  let dad_score : ℕ := 7
  let olaf_score : ℕ := 3 * dad_score
  let sister_score : ℕ := dad_score + 4
  let mom_score : ℕ := 2 * sister_score
  dad_score + olaf_score + sister_score + mom_score = 61 :=
by sorry

end NUMINAMATH_CALUDE_family_game_score_l430_43040


namespace NUMINAMATH_CALUDE_simple_interest_months_l430_43050

/-- Simple interest calculation -/
theorem simple_interest_months (P R SI : ℚ) (h1 : P = 10000) (h2 : R = 4/100) (h3 : SI = 400) :
  SI = P * R * (12/12) :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_months_l430_43050


namespace NUMINAMATH_CALUDE_document_typing_time_l430_43037

theorem document_typing_time 
  (total_time : ℝ) 
  (susan_time : ℝ) 
  (jack_time : ℝ) 
  (h1 : total_time = 10) 
  (h2 : susan_time = 30) 
  (h3 : jack_time = 24) : 
  ∃ jonathan_time : ℝ, 
    jonathan_time = 40 ∧ 
    1 / total_time = 1 / jonathan_time + 1 / susan_time + 1 / jack_time :=
by
  sorry

#check document_typing_time

end NUMINAMATH_CALUDE_document_typing_time_l430_43037


namespace NUMINAMATH_CALUDE_general_drinking_horse_shortest_distance_l430_43056

/-- The shortest distance for the "General Drinking Horse" problem -/
theorem general_drinking_horse_shortest_distance :
  let camp := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 1}
  let A : ℝ × ℝ := (2, 0)
  let riverbank := {p : ℝ × ℝ | p.1 + p.2 = 3}
  ∃ (B : ℝ × ℝ) (C : ℝ × ℝ),
    B ∈ riverbank ∧ C ∈ camp ∧
    ∀ (B' : ℝ × ℝ) (C' : ℝ × ℝ),
      B' ∈ riverbank → C' ∈ camp →
      Real.sqrt 10 - 1 ≤ dist A B' + dist B' C' :=
sorry

end NUMINAMATH_CALUDE_general_drinking_horse_shortest_distance_l430_43056


namespace NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l430_43098

theorem isosceles_triangle_largest_angle (α β γ : ℝ) : 
  -- The triangle is isosceles with two 60° angles
  α = 60 ∧ β = 60 ∧ 
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- The largest angle is 60°
  max α (max β γ) = 60 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l430_43098


namespace NUMINAMATH_CALUDE_intersection_value_l430_43029

theorem intersection_value (A B : Set ℝ) (m : ℝ) : 
  A = {-1, 1, 3} → 
  B = {1, m} → 
  A ∩ B = {1, 3} → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_value_l430_43029


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l430_43085

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation 6x^2 - 14x + 10 has discriminant -44 -/
theorem quadratic_discriminant :
  discriminant 6 (-14) 10 = -44 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l430_43085
