import Mathlib

namespace NUMINAMATH_CALUDE_cd_length_in_isosceles_triangles_l1336_133687

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.leg + t.base

theorem cd_length_in_isosceles_triangles 
  (abc : IsoscelesTriangle) 
  (cbd : IsoscelesTriangle) 
  (h1 : perimeter cbd = 25)
  (h2 : perimeter abc = 20)
  (h3 : cbd.base = 9) :
  cbd.leg = 8 := by
  sorry

end NUMINAMATH_CALUDE_cd_length_in_isosceles_triangles_l1336_133687


namespace NUMINAMATH_CALUDE_yellow_balls_count_l1336_133634

/-- The number of yellow balls in a bag, given the total number of balls,
    the number of balls of each color (except yellow), and the probability
    of choosing a ball that is neither red nor purple. -/
def yellow_balls (total : ℕ) (white green red purple : ℕ) (prob_not_red_purple : ℚ) : ℕ :=
  total - white - green - red - purple

/-- Theorem stating that the number of yellow balls is 5 under the given conditions. -/
theorem yellow_balls_count :
  yellow_balls 60 22 18 6 9 (3/4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l1336_133634


namespace NUMINAMATH_CALUDE_fifteenth_term_of_specific_sequence_l1336_133648

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 15th term of the specific arithmetic sequence -/
theorem fifteenth_term_of_specific_sequence (a : ℕ → ℝ) 
    (h_arithmetic : ArithmeticSequence a)
    (h_first : a 1 = 3)
    (h_second : a 2 = 15)
    (h_third : a 3 = 27) :
  a 15 = 171 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_specific_sequence_l1336_133648


namespace NUMINAMATH_CALUDE_x_plus_inv_x_power_n_is_integer_l1336_133681

theorem x_plus_inv_x_power_n_is_integer
  (x : ℝ) (h : ∃ (k : ℤ), x + 1 / x = k) :
  ∀ n : ℕ, ∃ (m : ℤ), x^n + 1 / x^n = m :=
by sorry

end NUMINAMATH_CALUDE_x_plus_inv_x_power_n_is_integer_l1336_133681


namespace NUMINAMATH_CALUDE_max_knights_between_knights_is_32_l1336_133624

/-- Represents a seating arrangement of knights and samurais around a round table. -/
structure SeatingArrangement where
  total_knights : ℕ
  total_samurais : ℕ
  knights_with_samurai_right : ℕ

/-- The maximum number of knights that could be seated next to two other knights. -/
def max_knights_between_knights (arrangement : SeatingArrangement) : ℕ :=
  arrangement.total_knights - (arrangement.knights_with_samurai_right + 1)

/-- Theorem stating the maximum number of knights that could be seated next to two other knights
    in the given arrangement. -/
theorem max_knights_between_knights_is_32 (arrangement : SeatingArrangement) 
  (h1 : arrangement.total_knights = 40)
  (h2 : arrangement.total_samurais = 10)
  (h3 : arrangement.knights_with_samurai_right = 7) :
  max_knights_between_knights arrangement = 32 := by
  sorry

#check max_knights_between_knights_is_32

end NUMINAMATH_CALUDE_max_knights_between_knights_is_32_l1336_133624


namespace NUMINAMATH_CALUDE_min_value_theorem_l1336_133611

/-- Two circles C₁ and C₂ with given equations -/
def C₁ (x y a : ℝ) : Prop := x^2 + y^2 + 2*a*x + a^2 - 9 = 0
def C₂ (x y b : ℝ) : Prop := x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0

/-- The condition that the circles have only one common tangent line -/
def one_common_tangent (a b : ℝ) : Prop := sorry

theorem min_value_theorem (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : one_common_tangent a b) :
  (∀ x y, C₁ x y a → C₂ x y b → 4/a^2 + 1/b^2 ≥ 4) ∧ 
  (∃ x y, C₁ x y a ∧ C₂ x y b ∧ 4/a^2 + 1/b^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1336_133611


namespace NUMINAMATH_CALUDE_bug_crawl_distance_l1336_133609

theorem bug_crawl_distance (r : ℝ) (second_leg : ℝ) (h1 : r = 75) (h2 : second_leg = 100) :
  let diameter := 2 * r
  let third_leg := Real.sqrt (diameter^2 - second_leg^2)
  diameter + second_leg + third_leg = 150 + 100 + 50 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_bug_crawl_distance_l1336_133609


namespace NUMINAMATH_CALUDE_equation_solution_l1336_133663

theorem equation_solution (k : ℤ) : 
  (∃ x : ℤ, x > 0 ∧ 9 * x - 3 = k * x + 14) ↔ (k = 8 ∨ k = -8) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1336_133663


namespace NUMINAMATH_CALUDE_annulus_area_single_element_l1336_133661

/-- The area of an annulus can be expressed using only one linear element -/
theorem annulus_area_single_element (R r : ℝ) (h : R > r) :
  ∃ (d : ℝ), (d = R - r ∨ d = R + r) ∧
  (π * (R^2 - r^2) = π * d * (2*R - d) ∨ π * (R^2 - r^2) = π * d * (2*r + d)) := by
  sorry

end NUMINAMATH_CALUDE_annulus_area_single_element_l1336_133661


namespace NUMINAMATH_CALUDE_smaller_rectangle_dimensions_l1336_133692

theorem smaller_rectangle_dimensions 
  (square_side : ℝ) 
  (h_square_side : square_side = 10) 
  (small_length small_width : ℝ) 
  (h_rectangles : small_length + 2 * small_length = square_side) 
  (h_square : small_width = small_length) : 
  small_length = 10 / 3 ∧ small_width = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_smaller_rectangle_dimensions_l1336_133692


namespace NUMINAMATH_CALUDE_no_nine_diagonals_intersection_l1336_133668

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry

/-- A diagonal of a polygon -/
def Diagonal (n : ℕ) (p : RegularPolygon n) (i j : Fin n) : Set (ℝ × ℝ) :=
  sorry

/-- The set of all diagonals in a polygon -/
def AllDiagonals (n : ℕ) (p : RegularPolygon n) : Set (Set (ℝ × ℝ)) :=
  sorry

/-- A point is internal to a polygon if it's inside the polygon -/
def IsInternal (n : ℕ) (p : RegularPolygon n) (point : ℝ × ℝ) : Prop :=
  sorry

/-- The number of diagonals passing through a point -/
def DiagonalsThroughPoint (n : ℕ) (p : RegularPolygon n) (point : ℝ × ℝ) : ℕ :=
  sorry

theorem no_nine_diagonals_intersection (p : RegularPolygon 25) 
  (diags : AllDiagonals 25 p) :
  ¬ ∃ (point : ℝ × ℝ), IsInternal 25 p point ∧ DiagonalsThroughPoint 25 p point = 9 :=
sorry

end NUMINAMATH_CALUDE_no_nine_diagonals_intersection_l1336_133668


namespace NUMINAMATH_CALUDE_wheat_mixture_profit_percentage_l1336_133640

/-- Calculates the profit percentage for a wheat mixture sale --/
theorem wheat_mixture_profit_percentage
  (wheat1_weight : ℝ)
  (wheat1_price : ℝ)
  (wheat2_weight : ℝ)
  (wheat2_price : ℝ)
  (selling_price : ℝ)
  (h1 : wheat1_weight = 30)
  (h2 : wheat1_price = 11.5)
  (h3 : wheat2_weight = 20)
  (h4 : wheat2_price = 14.25)
  (h5 : selling_price = 15.75) :
  let total_cost := wheat1_weight * wheat1_price + wheat2_weight * wheat2_price
  let total_weight := wheat1_weight + wheat2_weight
  let total_selling_price := total_weight * selling_price
  let profit := total_selling_price - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage = 25 := by
    sorry

end NUMINAMATH_CALUDE_wheat_mixture_profit_percentage_l1336_133640


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1336_133688

/-- The area of a square inscribed in an isosceles right triangle -/
theorem inscribed_square_area (leg_length : ℝ) (h : leg_length = 28 * Real.sqrt 2) :
  let diagonal := leg_length
  let side := diagonal / Real.sqrt 2
  side ^ 2 = 784 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1336_133688


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l1336_133633

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The repeating decimal 8.137137137... -/
def x : RepeatingDecimal :=
  { integerPart := 8, repeatingPart := 137 }

theorem repeating_decimal_as_fraction :
  toRational x = 2709 / 333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l1336_133633


namespace NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l1336_133620

/-- A complex number is purely imaginary if its real part is zero. -/
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

/-- For a, b ∈ ℝ, "a = 0" is a necessary but not sufficient condition 
    for the complex number a + bi to be purely imaginary. -/
theorem a_zero_necessary_not_sufficient (a b : ℝ) :
  (is_purely_imaginary (Complex.mk a b) → a = 0) ∧
  ¬(a = 0 → is_purely_imaginary (Complex.mk a b)) :=
sorry

end NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l1336_133620


namespace NUMINAMATH_CALUDE_square_difference_pattern_l1336_133686

theorem square_difference_pattern (n : ℕ) : (2*n + 1)^2 - (2*n - 1)^2 = 8*n := by
  sorry

end NUMINAMATH_CALUDE_square_difference_pattern_l1336_133686


namespace NUMINAMATH_CALUDE_problem_solution_l1336_133691

def set_product (A B : Set ℝ) : Set ℝ :=
  {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}

def A : Set ℝ := {0, 2}
def B : Set ℝ := {1, 3}
def C : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem problem_solution :
  (set_product A B) ∩ (set_product B C) = {2, 6} := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1336_133691


namespace NUMINAMATH_CALUDE_prime_representation_l1336_133651

theorem prime_representation (N : ℕ) (hN : Nat.Prime N) :
  ∃ (n : ℤ) (p : ℕ), Nat.Prime p ∧ p < 30 ∧ N = 30 * n.natAbs + p :=
sorry

end NUMINAMATH_CALUDE_prime_representation_l1336_133651


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l1336_133682

theorem absolute_value_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 12*a*b) :
  |((a + b) / (a - b))| = Real.sqrt (7/5) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l1336_133682


namespace NUMINAMATH_CALUDE_amy_balloons_l1336_133677

theorem amy_balloons (red green blue : ℕ) (h1 : red = 29) (h2 : green = 17) (h3 : blue = 21) :
  red + green + blue = 67 := by
  sorry

end NUMINAMATH_CALUDE_amy_balloons_l1336_133677


namespace NUMINAMATH_CALUDE_negation_square_positive_l1336_133674

theorem negation_square_positive :
  (¬ ∀ n : ℕ, n^2 > 0) ↔ (∃ n : ℕ, n^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_square_positive_l1336_133674


namespace NUMINAMATH_CALUDE_art_club_enrollment_l1336_133656

theorem art_club_enrollment (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ)
  (h1 : total = 80)
  (h2 : biology = 50)
  (h3 : chemistry = 40)
  (h4 : both = 30) :
  total - (biology + chemistry - both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_art_club_enrollment_l1336_133656


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocal_squares_l1336_133696

theorem min_sum_of_reciprocal_squares (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = 1) : 
  27 ≤ (1/a^2 + 1/b^2 + 1/c^2) := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocal_squares_l1336_133696


namespace NUMINAMATH_CALUDE_class_height_most_suitable_for_census_l1336_133654

/-- Represents a scenario that could be investigated --/
inductive Scenario
| WaterQuality
| StudentMentalHealth
| ClassHeight
| TVRatings

/-- Characteristics of a scenario --/
structure ScenarioCharacteristics where
  population_size : ℕ
  accessibility : Bool
  feasibility : Bool

/-- Defines what makes a scenario suitable for a census --/
def suitable_for_census (c : ScenarioCharacteristics) : Prop :=
  c.population_size ≤ 100 ∧ c.accessibility ∧ c.feasibility

/-- Assigns characteristics to each scenario --/
def scenario_characteristics : Scenario → ScenarioCharacteristics
| Scenario.WaterQuality => ⟨1000, false, false⟩
| Scenario.StudentMentalHealth => ⟨1000000, false, false⟩
| Scenario.ClassHeight => ⟨30, true, true⟩
| Scenario.TVRatings => ⟨10000000, false, false⟩

theorem class_height_most_suitable_for_census :
  ∀ s : Scenario, s ≠ Scenario.ClassHeight →
    ¬(suitable_for_census (scenario_characteristics s)) ∧
    suitable_for_census (scenario_characteristics Scenario.ClassHeight) :=
by sorry

end NUMINAMATH_CALUDE_class_height_most_suitable_for_census_l1336_133654


namespace NUMINAMATH_CALUDE_part1_part2_l1336_133616

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 2*|x + 1|

-- Part 1
theorem part1 : 
  {x : ℝ | f 2 x > 4} = {x : ℝ | x < -4/3 ∨ x > 0} := by sorry

-- Part 2
theorem part2 : 
  ({x : ℝ | f a x < 3*x + 4} = {x : ℝ | x > 2}) → a = 6 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1336_133616


namespace NUMINAMATH_CALUDE_plane_P_satisfies_conditions_l1336_133601

def plane1 (x y z : ℝ) : ℝ := 2*x - y + 2*z - 4
def plane2 (x y z : ℝ) : ℝ := 3*x + 4*y - z - 6
def planeP (x y z : ℝ) : ℝ := x + 63*y - 35*z - 34

def point : ℝ × ℝ × ℝ := (4, -2, 2)

theorem plane_P_satisfies_conditions :
  (∀ x y z, plane1 x y z = 0 ∧ plane2 x y z = 0 → planeP x y z = 0) ∧
  (planeP ≠ plane1 ∧ planeP ≠ plane2) ∧
  (abs (planeP point.1 point.2.1 point.2.2) / 
   Real.sqrt ((1:ℝ)^2 + 63^2 + (-35)^2) = 3 / Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_plane_P_satisfies_conditions_l1336_133601


namespace NUMINAMATH_CALUDE_smallest_number_l1336_133683

theorem smallest_number (a b c d : Int) (h1 : a = 2023) (h2 : b = 2022) (h3 : c = -2023) (h4 : d = -2022) :
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1336_133683


namespace NUMINAMATH_CALUDE_solve_equation_l1336_133627

theorem solve_equation (y : ℝ) : 7 - y = 10 → y = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1336_133627


namespace NUMINAMATH_CALUDE_fraction_sum_l1336_133671

theorem fraction_sum (x y : ℚ) (h : x / y = 5 / 2) : (x + y) / y = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1336_133671


namespace NUMINAMATH_CALUDE_vector_at_negative_two_l1336_133632

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  /-- The vector on the line at parameter t -/
  vector_at : ℝ → ℝ × ℝ

/-- Theorem: Given a parameterized line with specific points, the vector at t = -2 can be determined -/
theorem vector_at_negative_two
  (line : ParameterizedLine)
  (h1 : line.vector_at 1 = (2, 5))
  (h4 : line.vector_at 4 = (8, -7)) :
  line.vector_at (-2) = (-4, 17) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_negative_two_l1336_133632


namespace NUMINAMATH_CALUDE_problem_solution_l1336_133664

theorem problem_solution (x : ℝ) (h : x = 13 / Real.sqrt (19 + 8 * Real.sqrt 3)) :
  (x^4 - 6*x^3 - 2*x^2 + 18*x + 23) / (x^2 - 8*x + 15) = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1336_133664


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_d_value_l1336_133658

theorem quadratic_roots_imply_d_value (d : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 9 * x + d = 0 ↔ x = (-9 + Real.sqrt 17) / 4 ∨ x = (-9 - Real.sqrt 17) / 4) →
  d = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_d_value_l1336_133658


namespace NUMINAMATH_CALUDE_two_players_goals_l1336_133625

theorem two_players_goals (total_goals : ℕ) (players : ℕ) (percentage : ℚ) 
  (h1 : total_goals = 300)
  (h2 : players = 2)
  (h3 : percentage = 1/5) : 
  (↑total_goals * percentage) / players = 30 := by
  sorry

end NUMINAMATH_CALUDE_two_players_goals_l1336_133625


namespace NUMINAMATH_CALUDE_simplify_fraction_l1336_133672

theorem simplify_fraction : (5 : ℚ) * (13 / 3) * (21 / -65) = -7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1336_133672


namespace NUMINAMATH_CALUDE_sum_of_digits_difference_l1336_133619

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits for all numbers in a list -/
def sumOfDigitsForList (list : List ℕ) : ℕ := sorry

/-- List of odd numbers from 1 to 99 -/
def oddNumbers : List ℕ := sorry

/-- List of even numbers from 2 to 100 -/
def evenNumbers : List ℕ := sorry

theorem sum_of_digits_difference : 
  sumOfDigitsForList oddNumbers - sumOfDigitsForList evenNumbers = 49 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_difference_l1336_133619


namespace NUMINAMATH_CALUDE_sum_x_y_equals_negative_three_l1336_133610

theorem sum_x_y_equals_negative_three (x y : ℝ) :
  (3 * x - y + 5)^2 + |2 * x - y + 3| = 0 → x + y = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_negative_three_l1336_133610


namespace NUMINAMATH_CALUDE_k_range_given_a_n_geq_a_3_l1336_133699

/-- A sequence where a_n = n^2 - k*n -/
def a (n : ℕ+) (k : ℝ) : ℝ := n.val^2 - k * n.val

/-- The theorem stating that if a_n ≥ a_3 for all positive integers n, then k is in [5, 7] -/
theorem k_range_given_a_n_geq_a_3 :
  (∀ n : ℕ+, a n k ≥ a 3 k) → k ∈ Set.Icc (5 : ℝ) 7 := by
  sorry

end NUMINAMATH_CALUDE_k_range_given_a_n_geq_a_3_l1336_133699


namespace NUMINAMATH_CALUDE_g_of_3_l1336_133639

def g (x : ℝ) : ℝ := 5 * x^3 + 7 * x^2 - 3 * x - 6

theorem g_of_3 : g 3 = 183 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_l1336_133639


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1336_133669

theorem fraction_to_decimal : (59 : ℚ) / (2^2 * 5^7) = (1888 : ℚ) / 10^7 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1336_133669


namespace NUMINAMATH_CALUDE_gear_speed_ratio_l1336_133662

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Proves that for three interconnected gears, if the product of teeth and speed is equal,
    then the ratio of their speeds is proportional to the product of the other two gears' teeth -/
theorem gear_speed_ratio
  (G H I : Gear)
  (h : G.teeth * G.speed = H.teeth * H.speed ∧ H.teeth * H.speed = I.teeth * I.speed) :
  ∃ (k : ℝ), G.speed = k * (H.teeth * I.teeth) ∧
             H.speed = k * (G.teeth * I.teeth) ∧
             I.speed = k * (G.teeth * H.teeth) := by
  sorry

end NUMINAMATH_CALUDE_gear_speed_ratio_l1336_133662


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l1336_133690

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

end NUMINAMATH_CALUDE_cricket_team_average_age_l1336_133690


namespace NUMINAMATH_CALUDE_smallest_norm_v_l1336_133653

open Real
open Vector

/-- Given a vector v such that ||v + (-2, 4)|| = 10, the smallest possible value of ||v|| is 10 - 2√5 -/
theorem smallest_norm_v (v : ℝ × ℝ) (h : ‖v + (-2, 4)‖ = 10) :
  ∀ w : ℝ × ℝ, ‖w + (-2, 4)‖ = 10 → ‖v‖ ≤ ‖w‖ ∧ ‖v‖ = 10 - 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_norm_v_l1336_133653


namespace NUMINAMATH_CALUDE_max_value_fg_unique_root_condition_inequality_condition_l1336_133647

noncomputable section

-- Define the functions f and g
def f (a x : ℝ) : ℝ := x^2 + a*x + 1
def g (x : ℝ) : ℝ := Real.exp x

-- Part 1
theorem max_value_fg (x : ℝ) (hx : x ∈ Set.Icc (-2) 0) :
  (f 1 x) * (g x) ≤ 1 :=
sorry

-- Part 2
theorem unique_root_condition (k : ℝ) :
  (∃! x, f (-1) x = k * g x) ↔ (k > 3 / Real.exp 2 ∨ 0 < k ∧ k < 1 / Real.exp 1) :=
sorry

-- Part 3
theorem inequality_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 → x₁ ≠ x₂ →
    |f a x₁ - f a x₂| < |g x₁ - g x₂|) ↔
  (-1 ≤ a ∧ a ≤ 2 - 2 * Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_max_value_fg_unique_root_condition_inequality_condition_l1336_133647


namespace NUMINAMATH_CALUDE_sophomore_selection_l1336_133649

/-- Represents the number of students selected from a grade -/
structure GradeSelection where
  total : Nat
  selected : Nat

/-- Represents the stratified sampling of students across grades -/
structure StratifiedSampling where
  freshmen : GradeSelection
  sophomores : GradeSelection
  seniors : GradeSelection

/-- 
Given a stratified sampling where:
- There are 210 freshmen, 270 sophomores, and 300 seniors
- 7 freshmen were selected
- The same selection rate is applied across all grades

Prove that 9 sophomores were selected
-/
theorem sophomore_selection (s : StratifiedSampling) 
  (h1 : s.freshmen.total = 210)
  (h2 : s.sophomores.total = 270)
  (h3 : s.seniors.total = 300)
  (h4 : s.freshmen.selected = 7)
  (h5 : s.freshmen.selected * s.sophomores.total = s.sophomores.selected * s.freshmen.total) :
  s.sophomores.selected = 9 := by
  sorry


end NUMINAMATH_CALUDE_sophomore_selection_l1336_133649


namespace NUMINAMATH_CALUDE_total_balls_l1336_133608

/-- Given 2 boxes, each containing 3 balls, the total number of balls is 6. -/
theorem total_balls (num_boxes : ℕ) (balls_per_box : ℕ) (h1 : num_boxes = 2) (h2 : balls_per_box = 3) :
  num_boxes * balls_per_box = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_l1336_133608


namespace NUMINAMATH_CALUDE_common_solution_z_values_l1336_133605

theorem common_solution_z_values : 
  ∃ (z₁ z₂ : ℝ), 
    (∀ x : ℝ, x^2 + z₁^2 - 9 = 0 ∧ x^2 - 4*z₁ + 5 = 0) ∧
    (∀ x : ℝ, x^2 + z₂^2 - 9 = 0 ∧ x^2 - 4*z₂ + 5 = 0) ∧
    z₁ = -2 + 3 * Real.sqrt 2 ∧
    z₂ = -2 - 3 * Real.sqrt 2 ∧
    (∀ z : ℝ, (∃ x : ℝ, x^2 + z^2 - 9 = 0 ∧ x^2 - 4*z + 5 = 0) → (z = z₁ ∨ z = z₂)) :=
by sorry

end NUMINAMATH_CALUDE_common_solution_z_values_l1336_133605


namespace NUMINAMATH_CALUDE_petes_flag_problem_l1336_133698

/-- The number of stars on the US flag -/
def us_stars : ℕ := 50

/-- The number of stripes on the US flag -/
def us_stripes : ℕ := 13

/-- The number of circles on Pete's flag -/
def petes_circles : ℕ := us_stars / 2 - 3

/-- The number of squares on Pete's flag -/
def petes_squares (x : ℕ) : ℕ := 2 * us_stripes + x

/-- The total number of shapes on Pete's flag -/
def total_shapes : ℕ := 54

theorem petes_flag_problem :
  ∃ x : ℕ, petes_circles + petes_squares x = total_shapes ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_petes_flag_problem_l1336_133698


namespace NUMINAMATH_CALUDE_warehouse_worker_wage_l1336_133670

/-- Represents the problem of calculating warehouse workers' hourly wage --/
theorem warehouse_worker_wage :
  let num_warehouse_workers : ℕ := 4
  let num_managers : ℕ := 2
  let manager_hourly_wage : ℚ := 20
  let fica_tax_rate : ℚ := 1/10
  let days_per_month : ℕ := 25
  let hours_per_day : ℕ := 8
  let total_monthly_cost : ℚ := 22000

  let total_hours : ℕ := days_per_month * hours_per_day
  let manager_monthly_wage : ℚ := num_managers * manager_hourly_wage * total_hours
  
  ∃ (warehouse_hourly_wage : ℚ),
    warehouse_hourly_wage = 15 ∧
    total_monthly_cost = (1 + fica_tax_rate) * (num_warehouse_workers * warehouse_hourly_wage * total_hours + manager_monthly_wage) :=
by sorry

end NUMINAMATH_CALUDE_warehouse_worker_wage_l1336_133670


namespace NUMINAMATH_CALUDE_pizza_division_l1336_133676

theorem pizza_division (total_pizza : ℚ) (num_friends : ℕ) : 
  total_pizza = 5/6 ∧ num_friends = 4 → 
  total_pizza / num_friends = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_pizza_division_l1336_133676


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l1336_133673

theorem infinite_solutions_condition (b : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l1336_133673


namespace NUMINAMATH_CALUDE_parabola_equation_l1336_133655

/-- A parabola with vertex at the origin and focus at (0, 3) has the equation x^2 = 12y -/
theorem parabola_equation (p : ℝ × ℝ → Prop) :
  (∀ x y, p (x, y) ↔ x^2 = 12*y) →
  (p (0, 0)) →  -- vertex at origin
  (∀ x y, x^2 + (y - 3)^2 = 4 → p (0, y)) →  -- focus at center of circle
  ∀ x y, p (x, y) ↔ x^2 = 12*y :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1336_133655


namespace NUMINAMATH_CALUDE_sum_of_integers_l1336_133613

theorem sum_of_integers (a b : ℕ+) : a^2 - b^2 = 52 → a * b = 168 → a + b = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1336_133613


namespace NUMINAMATH_CALUDE_pie_division_l1336_133623

theorem pie_division (total_pie : ℚ) (people : ℕ) : 
  total_pie = 8 / 9 → people = 3 → total_pie / people = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_pie_division_l1336_133623


namespace NUMINAMATH_CALUDE_f_negative_a_l1336_133675

/-- Given a function f(x) = x³cos(x) + 1 and a real number a such that f(a) = 11,
    prove that f(-a) = -9 -/
theorem f_negative_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^3 * Real.cos x + 1) 
    (ha : f a = 11) : f (-a) = -9 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_l1336_133675


namespace NUMINAMATH_CALUDE_mrs_hilt_hotdog_cost_l1336_133629

/-- The total cost in cents for a given number of hot dogs at a given price per hot dog -/
def total_cost (num_hotdogs : ℕ) (price_per_hotdog : ℕ) : ℕ :=
  num_hotdogs * price_per_hotdog

/-- Proof that Mrs. Hilt paid 300 cents for 6 hot dogs at 50 cents each -/
theorem mrs_hilt_hotdog_cost : total_cost 6 50 = 300 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_hotdog_cost_l1336_133629


namespace NUMINAMATH_CALUDE_amount_after_two_years_l1336_133618

/-- The amount after n years given an initial amount and yearly increase rate -/
def amount_after_years (initial_amount : ℝ) (increase_rate : ℝ) (years : ℕ) : ℝ :=
  initial_amount * (1 + increase_rate) ^ years

/-- Theorem stating the amount after two years -/
theorem amount_after_two_years :
  let initial_amount : ℝ := 62000
  let increase_rate : ℝ := 1/8
  let years : ℕ := 2
  amount_after_years initial_amount increase_rate years = 78468.75 := by
sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l1336_133618


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1336_133680

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  (a 3)^2 - 3 * (a 3) - 5 = 0 →
  (a 11)^2 - 3 * (a 11) - 5 = 0 →
  a 5 + a 6 + a 10 = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1336_133680


namespace NUMINAMATH_CALUDE_max_cabbages_is_256_l1336_133612

structure Region where
  area : ℕ
  sunlight : ℕ
  water : ℕ

def is_suitable (r : Region) : Bool :=
  r.sunlight ≥ 4 ∧ r.water ≤ 16

def count_cabbages (regions : List Region) : ℕ :=
  (regions.filter is_suitable).foldl (fun acc r => acc + r.area) 0

def garden : List Region :=
  [
    ⟨30, 5, 15⟩,
    ⟨25, 6, 12⟩,
    ⟨35, 8, 18⟩,
    ⟨40, 4, 10⟩,
    ⟨20, 7, 14⟩
  ]

theorem max_cabbages_is_256 :
  count_cabbages garden + 181 = 256 :=
by sorry

end NUMINAMATH_CALUDE_max_cabbages_is_256_l1336_133612


namespace NUMINAMATH_CALUDE_rectangle_bisector_slope_l1336_133602

/-- The slope of a line passing through the origin and the center of a rectangle
    with vertices (1, 0), (5, 0), (1, 2), and (5, 2) is 1/3. -/
theorem rectangle_bisector_slope :
  let vertices : List (ℝ × ℝ) := [(1, 0), (5, 0), (1, 2), (5, 2)]
  let center : ℝ × ℝ := (
    (vertices[0].1 + vertices[3].1) / 2,
    (vertices[0].2 + vertices[3].2) / 2
  )
  let slope : ℝ := center.2 / center.1
  slope = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_bisector_slope_l1336_133602


namespace NUMINAMATH_CALUDE_external_polygon_sides_l1336_133646

/-- Represents a regular polygon with a given number of sides -/
structure RegularPolygon :=
  (sides : ℕ)

/-- Represents the arrangement of polygons as described in the problem -/
structure PolygonArrangement :=
  (hexagon : RegularPolygon)
  (triangle : RegularPolygon)
  (square : RegularPolygon)
  (pentagon : RegularPolygon)
  (heptagon : RegularPolygon)
  (nonagon : RegularPolygon)

/-- Calculates the number of exposed sides in the resulting external polygon -/
def exposedSides (arrangement : PolygonArrangement) : ℕ :=
  arrangement.hexagon.sides +
  arrangement.triangle.sides +
  arrangement.square.sides +
  arrangement.pentagon.sides +
  arrangement.heptagon.sides +
  arrangement.nonagon.sides -
  10 -- Subtracting the sides that are shared between polygons

/-- The main theorem stating that the resulting external polygon has 20 sides -/
theorem external_polygon_sides (arrangement : PolygonArrangement)
  (h1 : arrangement.hexagon.sides = 6)
  (h2 : arrangement.triangle.sides = 3)
  (h3 : arrangement.square.sides = 4)
  (h4 : arrangement.pentagon.sides = 5)
  (h5 : arrangement.heptagon.sides = 7)
  (h6 : arrangement.nonagon.sides = 9) :
  exposedSides arrangement = 20 := by
  sorry

end NUMINAMATH_CALUDE_external_polygon_sides_l1336_133646


namespace NUMINAMATH_CALUDE_nested_expression_evaluation_l1336_133614

theorem nested_expression_evaluation : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_evaluation_l1336_133614


namespace NUMINAMATH_CALUDE_downstream_distance_l1336_133685

-- Define the given conditions
def boat_speed : ℝ := 16
def stream_speed : ℝ := 5
def time_downstream : ℝ := 7

-- Define the theorem
theorem downstream_distance :
  let effective_speed := boat_speed + stream_speed
  effective_speed * time_downstream = 147 :=
by sorry

end NUMINAMATH_CALUDE_downstream_distance_l1336_133685


namespace NUMINAMATH_CALUDE_distance_traveled_l1336_133600

/-- Given a person traveling at 65 km/hr for 3 hours, prove that the distance traveled is 195 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 65)
  (h2 : time = 3)
  (h3 : distance = speed * time) :
  distance = 195 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l1336_133600


namespace NUMINAMATH_CALUDE_min_value_of_function_l1336_133628

theorem min_value_of_function (x y : ℝ) : x^2 + y^2 - 8*x + 6*y + 26 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1336_133628


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l1336_133645

theorem arithmetic_to_geometric_sequence (a d : ℝ) : 
  (2 * (a - d)) * ((a + d) + 7) = a^2 ∧ 
  (a - d) * a * (a + d) = 1000 →
  d = 8 ∨ d = -15 := by sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l1336_133645


namespace NUMINAMATH_CALUDE_function_identification_l1336_133657

-- Define the exponential function
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

-- Define the property of being symmetric about the y-axis
def symmetric_about_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

-- Define the property of being a translation to the right by 1 unit
def translated_right_by_one (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (x - 1)

-- State the theorem
theorem function_identification
  (f g : ℝ → ℝ)
  (h1 : translated_right_by_one f g)
  (h2 : symmetric_about_y_axis g exp) :
  ∀ x, f x = exp (-x - 1) :=
by sorry

end NUMINAMATH_CALUDE_function_identification_l1336_133657


namespace NUMINAMATH_CALUDE_henry_earnings_l1336_133666

/-- Henry's lawn mowing earnings problem -/
theorem henry_earnings (dollars_per_lawn : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ) : 
  dollars_per_lawn = 5 → 
  total_lawns = 12 → 
  forgotten_lawns = 7 → 
  dollars_per_lawn * (total_lawns - forgotten_lawns) = 25 := by
  sorry


end NUMINAMATH_CALUDE_henry_earnings_l1336_133666


namespace NUMINAMATH_CALUDE_absolute_fraction_inequality_l1336_133643

theorem absolute_fraction_inequality (x : ℝ) : 
  |((3 * x - 2) / (x + 1))| > 3 ↔ x < -1 ∨ (-1 < x ∧ x < 1/6) :=
by sorry

end NUMINAMATH_CALUDE_absolute_fraction_inequality_l1336_133643


namespace NUMINAMATH_CALUDE_problem_statement_l1336_133615

-- Define the base conversion function
def baseToDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

-- Define the problem statement
theorem problem_statement (A B : Nat) (h1 : A = 2 * B) 
  (h2 : baseToDecimal [2, 2, 4] B + baseToDecimal [5, 5] A = baseToDecimal [1, 3, 4] (A + B)) :
  A + B = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1336_133615


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1336_133638

theorem binomial_coefficient_equality (x : ℕ) : 
  Nat.choose 28 x = Nat.choose 28 (2 * x - 1) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1336_133638


namespace NUMINAMATH_CALUDE_max_b_theorem_l1336_133622

def is_lattice_point (x y : ℤ) : Prop := True

def line_equation (m : ℚ) (x : ℚ) : ℚ := m * x + 3

def no_lattice_points (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x ∧ x ≤ 200 → ¬(is_lattice_point x y ∧ line_equation m x = y)

def max_b : ℚ := 68 / 203

theorem max_b_theorem :
  (∀ m : ℚ, 1/3 < m → m < max_b → no_lattice_points m) ∧
  (∀ b : ℚ, b > max_b → ∃ m : ℚ, 1/3 < m ∧ m < b ∧ ¬(no_lattice_points m)) :=
sorry

end NUMINAMATH_CALUDE_max_b_theorem_l1336_133622


namespace NUMINAMATH_CALUDE_ellipse_slope_product_ellipse_fixed_point_l1336_133617

/-- Represents an ellipse with eccentricity √6/3 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_ecc : (a^2 - b^2) / a^2 = 2/3

/-- A point on the ellipse -/
def PointOnEllipse (e : Ellipse) := {p : ℝ × ℝ // p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1}

theorem ellipse_slope_product (e : Ellipse) (M A B : PointOnEllipse e) 
  (h_sym : A.val = (-B.val.1, -B.val.2)) :
  let k₁ := (A.val.2 - M.val.2) / (A.val.1 - M.val.1)
  let k₂ := (B.val.2 - M.val.2) / (B.val.1 - M.val.1)
  k₁ * k₂ = -1/3 := by sorry

theorem ellipse_fixed_point (e : Ellipse) (M A B : PointOnEllipse e)
  (h_M : M.val = (0, 1))
  (h_slopes : let k₁ := (A.val.2 - M.val.2) / (A.val.1 - M.val.1)
              let k₂ := (B.val.2 - M.val.2) / (B.val.1 - M.val.1)
              k₁ + k₂ = 3) :
  ∃ (k m : ℝ), A.val.2 = k * A.val.1 + m ∧ 
                B.val.2 = k * B.val.1 + m ∧ 
                -2/3 = k * (-2/3) + m ∧ 
                -1 = k * (-2/3) + m := by sorry

end NUMINAMATH_CALUDE_ellipse_slope_product_ellipse_fixed_point_l1336_133617


namespace NUMINAMATH_CALUDE_exchange_impossibility_l1336_133678

theorem exchange_impossibility : ¬ ∃ (N : ℤ), 5 * N = 2001 := by sorry

end NUMINAMATH_CALUDE_exchange_impossibility_l1336_133678


namespace NUMINAMATH_CALUDE_least_possible_difference_l1336_133635

theorem least_possible_difference (x y z : ℤ) 
  (h_x_even : Even x)
  (h_y_odd : Odd y)
  (h_z_odd : Odd z)
  (h_order : x < y ∧ y < z)
  (h_diff : y - x > 5) :
  ∀ w, w = z - x → w ≥ 9 ∧ ∃ (a b c : ℤ), a - b = 9 ∧ Even b ∧ Odd a ∧ Odd c ∧ b < c ∧ c < a ∧ c - b > 5 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_difference_l1336_133635


namespace NUMINAMATH_CALUDE_equation_solution_l1336_133684

theorem equation_solution : 
  ∀ x : ℝ, x^4 + (3 - x)^4 = 82 ↔ x = 2.5 ∨ x = 0.5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1336_133684


namespace NUMINAMATH_CALUDE_larger_number_proof_l1336_133641

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1335)
  (h2 : L = 6 * S + 15) :
  L = 1599 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1336_133641


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1336_133630

theorem sin_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.tan (α + π/4) = 3 * Real.cos (2*α)) : 
  Real.sin (2*α) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1336_133630


namespace NUMINAMATH_CALUDE_sixth_quiz_score_l1336_133607

def john_scores : List ℕ := [85, 88, 90, 92, 83]
def target_mean : ℕ := 90
def num_quizzes : ℕ := 6

theorem sixth_quiz_score (score : ℕ) : 
  (john_scores.sum + score) / num_quizzes = target_mean ↔ score = 102 := by
  sorry

end NUMINAMATH_CALUDE_sixth_quiz_score_l1336_133607


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l1336_133697

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem subset_implies_a_values :
  ∀ a : ℝ, (B a ⊆ A) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5) := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l1336_133697


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l1336_133642

/-- Given two 2D vectors a and b, if a + b is parallel to 2a - b, then the x-coordinate of b is -4. -/
theorem parallel_vectors_imply_x_value (a b : ℝ × ℝ) (h : a = (2, 1)) (h' : b.2 = -2) :
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (2 • a - b)) → b.1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l1336_133642


namespace NUMINAMATH_CALUDE_school_boys_count_l1336_133695

/-- Given a school with more girls than boys, calculate the number of boys. -/
theorem school_boys_count (girls : ℕ) (difference : ℕ) (h1 : girls = 739) (h2 : difference = 402) :
  girls - difference = 337 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l1336_133695


namespace NUMINAMATH_CALUDE_cone_base_radius_l1336_133606

/-- A cone formed by a semicircle with radius 2 cm has a base circle with radius 1 cm -/
theorem cone_base_radius (r : ℝ) (h : r = 2) : 
  (2 * Real.pi * r / 2) / (2 * Real.pi) = 1 := by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l1336_133606


namespace NUMINAMATH_CALUDE_sheet_width_sheet_width_proof_l1336_133689

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

end NUMINAMATH_CALUDE_sheet_width_sheet_width_proof_l1336_133689


namespace NUMINAMATH_CALUDE_marble_selection_ways_l1336_133660

def total_marbles : ℕ := 9
def marbles_to_choose : ℕ := 4
def remaining_marbles : ℕ := total_marbles - 1
def remaining_to_choose : ℕ := marbles_to_choose - 1

theorem marble_selection_ways :
  (remaining_marbles.choose remaining_to_choose) = 56 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l1336_133660


namespace NUMINAMATH_CALUDE_correct_pairings_l1336_133604

/-- The number of possible pairings for the first round of a tennis tournament with 2n players -/
def numPairings (n : ℕ) : ℚ :=
  (Nat.factorial (2 * n)) / ((2 ^ n) * Nat.factorial n)

/-- Theorem stating that numPairings gives the correct number of possible pairings -/
theorem correct_pairings (n : ℕ) :
  numPairings n = (Nat.factorial (2 * n)) / ((2 ^ n) * Nat.factorial n) := by
  sorry

end NUMINAMATH_CALUDE_correct_pairings_l1336_133604


namespace NUMINAMATH_CALUDE_triangle_inequality_l1336_133659

/-- Given that a, b, and c are the side lengths of a triangle, 
    prove that a^2(b+c-a) + b^2(c+a-b) + c^2(a+b-c) ≤ 3abc -/
theorem triangle_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1336_133659


namespace NUMINAMATH_CALUDE_scout_cookies_unpacked_l1336_133621

/-- The number of boxes that cannot be fully packed into cases -/
def unpacked_boxes (total_boxes : ℕ) (boxes_per_case : ℕ) : ℕ :=
  total_boxes % boxes_per_case

/-- Proof that 7 boxes cannot be fully packed into cases -/
theorem scout_cookies_unpacked :
  unpacked_boxes 31 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_scout_cookies_unpacked_l1336_133621


namespace NUMINAMATH_CALUDE_max_imaginary_part_at_84_degrees_l1336_133679

/-- The polynomial whose roots we're investigating -/
def f (z : ℂ) : ℂ := z^12 - z^9 + z^6 - z^3 + 1

/-- The set of roots of the polynomial -/
def roots : Set ℂ := {z : ℂ | f z = 0}

/-- The set of angles corresponding to the roots -/
def root_angles : Set Real := {θ : Real | ∃ z ∈ roots, z = Complex.exp (θ * Complex.I)}

/-- The theorem stating the maximum imaginary part occurs at 84 degrees -/
theorem max_imaginary_part_at_84_degrees :
  ∃ θ ∈ root_angles,
    θ * Real.pi / 180 = 84 * Real.pi / 180 ∧
    ∀ φ ∈ root_angles, -Real.pi/2 ≤ φ ∧ φ ≤ Real.pi/2 →
      Complex.abs (Complex.sin (Complex.ofReal φ)) ≤ 
      Complex.abs (Complex.sin (Complex.ofReal (θ * Real.pi / 180))) :=
sorry

end NUMINAMATH_CALUDE_max_imaginary_part_at_84_degrees_l1336_133679


namespace NUMINAMATH_CALUDE_pet_store_kittens_l1336_133693

theorem pet_store_kittens (initial : ℕ) : initial + 3 = 9 → initial = 6 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_kittens_l1336_133693


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1336_133667

theorem trigonometric_simplification :
  (Real.tan (12 * π / 180) - Real.sqrt 3) / (Real.sin (12 * π / 180) * Real.cos (24 * π / 180)) = -8 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1336_133667


namespace NUMINAMATH_CALUDE_problem_solution_l1336_133652

noncomputable def f (a k x : ℝ) : ℝ := a^x - (k-1) * a^(-x)

theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- Part 1: k = 2
  (∃ k : ℝ, ∀ x : ℝ, f a k x = -f a k (-x)) ∧
  -- Part 2: f is monotonically decreasing
  (f a 2 1 < 0 → ∀ x y : ℝ, x < y → f a 2 x > f a 2 y) ∧
  -- Part 3: range of t
  (∃ t1 t2 : ℝ, t1 = -3 ∧ t2 = 5 ∧
    ∀ t : ℝ, (∀ x : ℝ, f a 2 (x^2 + t*x) + f a 2 (4-x) < 0) ↔ t1 < t ∧ t < t2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1336_133652


namespace NUMINAMATH_CALUDE_data_transmission_time_l1336_133603

theorem data_transmission_time (blocks : Nat) (chunks_per_block : Nat) (transmission_rate : Nat) :
  blocks = 50 →
  chunks_per_block = 1024 →
  transmission_rate = 100 →
  (blocks * chunks_per_block : ℚ) / transmission_rate / 60 = 9 := by
  sorry

end NUMINAMATH_CALUDE_data_transmission_time_l1336_133603


namespace NUMINAMATH_CALUDE_word_probabilities_l1336_133631

def word : String := "дифференцициал"

def is_vowel (c : Char) : Bool :=
  c ∈ ['а', 'е', 'и']

def is_consonant (c : Char) : Bool :=
  c ∈ ['д', 'ф', 'р', 'н', 'ц', 'л']

theorem word_probabilities :
  let total_letters := word.length
  let vowels := (word.toList.filter is_vowel).length
  let consonants := (word.toList.filter is_consonant).length
  (vowels : ℚ) / total_letters = 5 / 12 ∧
  (consonants : ℚ) / total_letters = 7 / 12 ∧
  ((word.toList.filter (· = 'ч')).length : ℚ) / total_letters = 0 := by
  sorry


end NUMINAMATH_CALUDE_word_probabilities_l1336_133631


namespace NUMINAMATH_CALUDE_gcd_180_126_l1336_133694

theorem gcd_180_126 : Nat.gcd 180 126 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_126_l1336_133694


namespace NUMINAMATH_CALUDE_max_product_sum_300_l1336_133665

theorem max_product_sum_300 (x y : ℤ) (h : x + y = 300) :
  x * y ≤ 22500 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l1336_133665


namespace NUMINAMATH_CALUDE_python_eating_theorem_l1336_133650

/-- Represents the eating rate of a python in terms of days per alligator --/
structure PythonEatingRate where
  days_per_alligator : ℕ

/-- Calculates the number of alligators a python can eat in a given number of days --/
def alligators_eaten (rate : PythonEatingRate) (days : ℕ) : ℕ :=
  days / rate.days_per_alligator

/-- The total number of alligators eaten by all pythons --/
def total_alligators_eaten (p1 p2 p3 : PythonEatingRate) (days : ℕ) : ℕ :=
  alligators_eaten p1 days + alligators_eaten p2 days + alligators_eaten p3 days

theorem python_eating_theorem (p1 p2 p3 : PythonEatingRate) 
  (h1 : p1.days_per_alligator = 7)  -- P1 eats one alligator per week
  (h2 : p2.days_per_alligator = 5)  -- P2 eats one alligator every 5 days
  (h3 : p3.days_per_alligator = 10) -- P3 eats one alligator every 10 days
  : total_alligators_eaten p1 p2 p3 21 = 9 := by
  sorry

#check python_eating_theorem

end NUMINAMATH_CALUDE_python_eating_theorem_l1336_133650


namespace NUMINAMATH_CALUDE_andrews_yearly_donation_l1336_133636

/-- Calculates the yearly donation amount given the starting age, current age, and total donation --/
def yearly_donation (start_age : ℕ) (current_age : ℕ) (total_donation : ℕ) : ℚ :=
  (total_donation : ℚ) / ((current_age - start_age) : ℚ)

/-- Theorem stating that Andrew's yearly donation is approximately 7388.89 --/
theorem andrews_yearly_donation :
  let start_age : ℕ := 11
  let current_age : ℕ := 29
  let total_donation : ℕ := 133000
  abs (yearly_donation start_age current_age total_donation - 7388.89) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_andrews_yearly_donation_l1336_133636


namespace NUMINAMATH_CALUDE_checkout_lane_shoppers_l1336_133637

theorem checkout_lane_shoppers (total_shoppers : ℕ) (avoid_fraction : ℚ) : 
  total_shoppers = 480 →
  avoid_fraction = 5/8 →
  total_shoppers - (total_shoppers * avoid_fraction).floor = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_checkout_lane_shoppers_l1336_133637


namespace NUMINAMATH_CALUDE_common_chord_length_l1336_133644

/-- Given two circles with radius 12 and centers 16 units apart,
    the length of their common chord is 8√5. -/
theorem common_chord_length (r : ℝ) (d : ℝ) (h1 : r = 12) (h2 : d = 16) :
  let chord_length := 2 * Real.sqrt (r^2 - (d/2)^2)
  chord_length = 8 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_common_chord_length_l1336_133644


namespace NUMINAMATH_CALUDE_even_increasing_ordering_l1336_133626

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x < f y

theorem even_increasing_ordering (f : ℝ → ℝ) 
  (h_even : is_even f) (h_incr : increasing_on_pos f) : 
  f 3 < f (-Real.pi) ∧ f (-Real.pi) < f (-4) := by sorry

end NUMINAMATH_CALUDE_even_increasing_ordering_l1336_133626
