import Mathlib

namespace NUMINAMATH_CALUDE_double_age_in_two_years_l601_60101

/-- The number of years it takes for a man's age to be twice his son's age -/
def years_until_double_age (son_age : ℕ) (age_difference : ℕ) : ℕ :=
  let man_age := son_age + age_difference
  (man_age - 2 * son_age) / (2 - 1)

theorem double_age_in_two_years (son_age : ℕ) (age_difference : ℕ) 
  (h1 : son_age = 18) (h2 : age_difference = 20) : 
  years_until_double_age son_age age_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_double_age_in_two_years_l601_60101


namespace NUMINAMATH_CALUDE_quadratic_function_range_l601_60152

theorem quadratic_function_range (m : ℝ) : 
  (∀ x : ℝ, -1 < x → x < 0 → x^2 - 4*m*x + 3 > 1) → m > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l601_60152


namespace NUMINAMATH_CALUDE_triangle_properties_l601_60191

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle given by the dot product formula -/
def area_dot_product (t : Triangle) : ℝ := (Real.sqrt 3 / 2) * (t.b * t.c * Real.cos t.A)

/-- The area of the triangle given by the sine formula -/
def area_sine (t : Triangle) : ℝ := (1 / 2) * t.b * t.c * Real.sin t.A

theorem triangle_properties (t : Triangle) 
  (h1 : area_dot_product t = area_sine t) 
  (h2 : t.b + t.c = 5) 
  (h3 : t.a = Real.sqrt 7) : 
  t.A = π / 3 ∧ area_sine t = (3 * Real.sqrt 3) / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l601_60191


namespace NUMINAMATH_CALUDE_seokjin_paper_count_prove_seokjin_paper_count_l601_60168

theorem seokjin_paper_count : ℕ → ℕ → ℕ → Prop :=
  fun jimin_count seokjin_count difference =>
    (jimin_count = 41) →
    (seokjin_count = jimin_count - difference) →
    (difference = 1) →
    (seokjin_count = 40)

#check seokjin_paper_count

theorem prove_seokjin_paper_count :
  seokjin_paper_count 41 40 1 := by sorry

end NUMINAMATH_CALUDE_seokjin_paper_count_prove_seokjin_paper_count_l601_60168


namespace NUMINAMATH_CALUDE_trihedral_angle_range_a_trihedral_angle_range_b_l601_60185

-- Define a trihedral angle
structure TrihedralAngle where
  α : Real
  β : Real
  γ : Real
  sum_less_than_360 : α + β + γ < 360
  each_less_than_sum_of_others : α < β + γ ∧ β < α + γ ∧ γ < α + β

-- Theorem for part (a)
theorem trihedral_angle_range_a (t : TrihedralAngle) (h1 : t.β = 70) (h2 : t.γ = 100) :
  30 < t.α ∧ t.α < 170 := by sorry

-- Theorem for part (b)
theorem trihedral_angle_range_b (t : TrihedralAngle) (h1 : t.β = 130) (h2 : t.γ = 150) :
  20 < t.α ∧ t.α < 80 := by sorry

end NUMINAMATH_CALUDE_trihedral_angle_range_a_trihedral_angle_range_b_l601_60185


namespace NUMINAMATH_CALUDE_paper_cut_end_time_l601_60144

def minutes_per_cut : ℕ := 3
def rest_minutes : ℕ := 1
def start_time : ℕ := 9 * 60 + 40  -- 9:40 in minutes since midnight
def num_cuts : ℕ := 10

def total_time : ℕ := (num_cuts - 1) * (minutes_per_cut + rest_minutes) + minutes_per_cut

def end_time : ℕ := start_time + total_time

theorem paper_cut_end_time :
  (end_time / 60, end_time % 60) = (10, 19) := by
  sorry

end NUMINAMATH_CALUDE_paper_cut_end_time_l601_60144


namespace NUMINAMATH_CALUDE_cone_lateral_area_l601_60127

/-- Given a cone with base circumference 4π and slant height 3, its lateral area is 6π. -/
theorem cone_lateral_area (c : ℝ) (l : ℝ) (h1 : c = 4 * Real.pi) (h2 : l = 3) :
  let r := c / (2 * Real.pi)
  π * r * l = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l601_60127


namespace NUMINAMATH_CALUDE_union_determines_m_l601_60103

theorem union_determines_m (A B : Set ℕ) (m : ℕ) : 
  A = {1, 3, m} → 
  B = {3, 4} → 
  A ∪ B = {1, 2, 3, 4} → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_union_determines_m_l601_60103


namespace NUMINAMATH_CALUDE_roots_product_minus_three_l601_60122

theorem roots_product_minus_three (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 7 * x₁ - 6 = 0) → 
  (3 * x₂^2 - 7 * x₂ - 6 = 0) → 
  (x₁ - 3) * (x₂ - 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_product_minus_three_l601_60122


namespace NUMINAMATH_CALUDE_negative_square_power_two_l601_60112

theorem negative_square_power_two (a b : ℝ) : (-a^2 * b)^2 = a^4 * b^2 := by sorry

end NUMINAMATH_CALUDE_negative_square_power_two_l601_60112


namespace NUMINAMATH_CALUDE_consecutive_four_plus_one_is_square_l601_60170

theorem consecutive_four_plus_one_is_square (n : ℕ) (h : n ≥ 1) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_four_plus_one_is_square_l601_60170


namespace NUMINAMATH_CALUDE_original_recipe_pasta_amount_l601_60132

theorem original_recipe_pasta_amount
  (original_servings : ℕ)
  (scaled_servings : ℕ)
  (scaled_pasta : ℝ)
  (h1 : original_servings = 7)
  (h2 : scaled_servings = 35)
  (h3 : scaled_pasta = 10) :
  let pasta_per_person : ℝ := scaled_pasta / scaled_servings
  let original_pasta : ℝ := pasta_per_person * original_servings
  original_pasta = 2 := by sorry

end NUMINAMATH_CALUDE_original_recipe_pasta_amount_l601_60132


namespace NUMINAMATH_CALUDE_boys_without_calculators_l601_60188

/-- Given a class with boys and girls, and information about calculator possession,
    prove the number of boys without calculators. -/
theorem boys_without_calculators
  (total_boys : ℕ)
  (total_with_calculators : ℕ)
  (girls_with_calculators : ℕ)
  (h1 : total_boys = 20)
  (h2 : total_with_calculators = 26)
  (h3 : girls_with_calculators = 13) :
  total_boys - (total_with_calculators - girls_with_calculators) = 7 :=
by
  sorry

#check boys_without_calculators

end NUMINAMATH_CALUDE_boys_without_calculators_l601_60188


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l601_60162

/-- The sum of an arithmetic sequence with first term a₁ = k^2 - k + 1 and
    common difference d = 1, for the first 2k terms. -/
theorem arithmetic_sequence_sum (k : ℕ) : 
  let a₁ := k^2 - k + 1
  let d := 1
  let n := 2 * k
  let S := n * (2 * a₁ + (n - 1) * d) / 2
  S = 2 * k^3 + k := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l601_60162


namespace NUMINAMATH_CALUDE_quadratic_root_equivalence_l601_60133

theorem quadratic_root_equivalence (a b c : ℝ) (h : a ≠ 0) :
  (1 = 1 ∧ a * 1^2 + b * 1 + c = 0) ↔ (a + b + c = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_equivalence_l601_60133


namespace NUMINAMATH_CALUDE_no_coin_solution_l601_60154

theorem no_coin_solution : ¬∃ (x y z : ℕ), 
  x + y + z = 50 ∧ 10 * x + 34 * y + 62 * z = 910 := by
  sorry

end NUMINAMATH_CALUDE_no_coin_solution_l601_60154


namespace NUMINAMATH_CALUDE_projectile_meeting_time_l601_60159

theorem projectile_meeting_time : 
  let initial_distance : ℝ := 2520
  let speed1 : ℝ := 432
  let speed2 : ℝ := 576
  let combined_speed : ℝ := speed1 + speed2
  let time_hours : ℝ := initial_distance / combined_speed
  let time_minutes : ℝ := time_hours * 60
  time_minutes = 150 := by sorry

end NUMINAMATH_CALUDE_projectile_meeting_time_l601_60159


namespace NUMINAMATH_CALUDE_remove_horizontal_eliminates_triangles_fifteen_is_minimum_removal_l601_60146

/-- Represents a triangular figure constructed with toothpicks -/
structure TriangularFigure where
  total_toothpicks : ℕ
  horizontal_toothpicks : ℕ
  has_upward_triangles : Bool
  has_downward_triangles : Bool

/-- Represents the number of toothpicks that need to be removed -/
def toothpicks_to_remove (figure : TriangularFigure) : ℕ := figure.horizontal_toothpicks

/-- Theorem stating that removing horizontal toothpicks eliminates all triangles -/
theorem remove_horizontal_eliminates_triangles (figure : TriangularFigure) 
  (h1 : figure.total_toothpicks = 45)
  (h2 : figure.horizontal_toothpicks = 15)
  (h3 : figure.has_upward_triangles = true)
  (h4 : figure.has_downward_triangles = true) :
  toothpicks_to_remove figure = 15 ∧ 
  (∀ n : ℕ, n < 15 → ∃ triangle_remains : Bool, triangle_remains = true) := by
  sorry

/-- Theorem stating that 15 is the minimum number of toothpicks to remove -/
theorem fifteen_is_minimum_removal (figure : TriangularFigure)
  (h1 : figure.total_toothpicks = 45)
  (h2 : figure.horizontal_toothpicks = 15)
  (h3 : figure.has_upward_triangles = true)
  (h4 : figure.has_downward_triangles = true) :
  ∀ n : ℕ, n < 15 → ∃ triangle_remains : Bool, triangle_remains = true := by
  sorry

end NUMINAMATH_CALUDE_remove_horizontal_eliminates_triangles_fifteen_is_minimum_removal_l601_60146


namespace NUMINAMATH_CALUDE_min_distance_to_line_l601_60194

theorem min_distance_to_line (x y : ℝ) (h : 6 * x + 8 * y - 1 = 0) :
  ∃ (min_val : ℝ), min_val = 7 / 10 ∧
  ∀ (x' y' : ℝ), 6 * x' + 8 * y' - 1 = 0 →
    Real.sqrt (x'^2 + y'^2 - 2*y' + 1) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l601_60194


namespace NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l601_60180

/-- Given a sphere and a right circular cylinder with equal surface areas,
    where the cylinder has height and diameter both equal to 14 cm,
    prove that the radius of the sphere is 7 cm. -/
theorem sphere_cylinder_equal_area (r : ℝ) : 
  r > 0 → -- Ensure the radius is positive
  (4 * Real.pi * r^2 = 2 * Real.pi * 7 * 14) → -- Surface areas are equal
  r = 7 := by
  sorry

#check sphere_cylinder_equal_area

end NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l601_60180


namespace NUMINAMATH_CALUDE_amoeba_fill_time_l601_60166

def amoeba_population (initial : ℕ) (time : ℕ) : ℕ :=
  initial * 2^time

theorem amoeba_fill_time :
  ∀ (tube_capacity : ℕ),
  tube_capacity > 0 →
  (∃ (t : ℕ), amoeba_population 1 t = tube_capacity) →
  (∃ (s : ℕ), amoeba_population 2 s = tube_capacity ∧ s + 1 = t) :=
by sorry

end NUMINAMATH_CALUDE_amoeba_fill_time_l601_60166


namespace NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l601_60139

/-- The slope of the tangent line to y = x³ at x = 1 -/
def tangent_slope : ℝ := 3

/-- The line equation ax - by - 2 = 0 -/
def line_equation (a b : ℝ) (x y : ℝ) : Prop :=
  a * x - b * y - 2 = 0

/-- The curve equation y = x³ -/
def curve_equation (x y : ℝ) : Prop :=
  y = x^3

theorem perpendicular_tangents_ratio (a b : ℝ) :
  line_equation a b 1 1 ∧ 
  curve_equation 1 1 ∧
  (a / b) * tangent_slope = -1 →
  a / b = -1/3 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l601_60139


namespace NUMINAMATH_CALUDE_intersection_implies_m_range_l601_60174

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | m/2 ≤ (p.1 - 2)^2 + p.2^2 ∧ (p.1 - 2)^2 + p.2^2 ≤ m^2}

def B (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2*m ≤ p.1 + p.2 ∧ p.1 + p.2 ≤ 2*m + 1}

-- State the theorem
theorem intersection_implies_m_range (m : ℝ) :
  (A m ∩ B m).Nonempty → 1/2 ≤ m ∧ m ≤ 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_m_range_l601_60174


namespace NUMINAMATH_CALUDE_seventh_equation_properties_l601_60192

/-- Defines the last number on the left side of the nth equation -/
def last_left (n : ℕ) : ℕ := 3 * n - 2

/-- Defines the result on the right side of the nth equation -/
def right_result (n : ℕ) : ℕ := (2 * n - 1) ^ 2

/-- Defines the sum of consecutive integers from n to (3n-2) -/
def left_sum (n : ℕ) : ℕ := 
  (n + last_left n) * (last_left n - n + 1) / 2

theorem seventh_equation_properties :
  last_left 7 = 19 ∧ right_result 7 = 169 ∧ left_sum 7 = right_result 7 := by
  sorry

end NUMINAMATH_CALUDE_seventh_equation_properties_l601_60192


namespace NUMINAMATH_CALUDE_least_integer_with_two_prime_factors_l601_60126

/-- A function that returns true if a number has exactly two prime factors -/
def has_two_prime_factors (n : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p * q

/-- The main theorem stating that 33 is the least positive integer satisfying the condition -/
theorem least_integer_with_two_prime_factors :
  (∀ m : ℕ, m > 0 ∧ m < 33 → ¬(has_two_prime_factors m ∧ has_two_prime_factors (m + 1) ∧ has_two_prime_factors (m + 2))) ∧
  (has_two_prime_factors 33 ∧ has_two_prime_factors 34 ∧ has_two_prime_factors 35) :=
by sorry

#check least_integer_with_two_prime_factors

end NUMINAMATH_CALUDE_least_integer_with_two_prime_factors_l601_60126


namespace NUMINAMATH_CALUDE_sum_first_eight_primes_ending_in_3_l601_60119

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def hasUnitsDigitOf3 (n : ℕ) : Prop := n % 10 = 3

def firstEightPrimesEndingIn3 : List ℕ := 
  [3, 13, 23, 43, 53, 73, 83, 103]

theorem sum_first_eight_primes_ending_in_3 :
  (∀ n ∈ firstEightPrimesEndingIn3, isPrime n ∧ hasUnitsDigitOf3 n) →
  (∀ p : ℕ, isPrime p → hasUnitsDigitOf3 p → 
    p ∉ firstEightPrimesEndingIn3 → 
    p > (List.maximum firstEightPrimesEndingIn3).getD 0) →
  List.sum firstEightPrimesEndingIn3 = 394 := by
sorry

end NUMINAMATH_CALUDE_sum_first_eight_primes_ending_in_3_l601_60119


namespace NUMINAMATH_CALUDE_quadratic_coefficient_values_l601_60169

/-- A quadratic function f(x) = ax^2 + 2ax + 1 with a maximum value of 5 on the interval [-2, 3] -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

/-- The maximum value of the quadratic function on the given interval -/
def max_value : ℝ := 5

/-- The lower bound of the interval -/
def lower_bound : ℝ := -2

/-- The upper bound of the interval -/
def upper_bound : ℝ := 3

/-- Theorem stating that the value of 'a' in the quadratic function
    with the given properties is either 4/15 or -4 -/
theorem quadratic_coefficient_values :
  ∃ (a : ℝ), (∀ x ∈ Set.Icc lower_bound upper_bound,
    quadratic_function a x ≤ max_value) ∧
  (∃ x ∈ Set.Icc lower_bound upper_bound,
    quadratic_function a x = max_value) ∧
  (a = 4/15 ∨ a = -4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_values_l601_60169


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l601_60156

theorem triangle_angle_sum (A B C : ℝ) : 
  (0 < A) → (0 < B) → (0 < C) →
  (A + B = 90) → (A + B + C = 180) →
  C = 90 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l601_60156


namespace NUMINAMATH_CALUDE_allison_not_lowest_prob_l601_60104

/-- Represents a 6-sided cube with specific face values -/
structure Cube where
  faces : Fin 6 → ℕ

/-- Allison's cube with all faces showing 3 -/
def allison_cube : Cube :=
  ⟨λ _ => 3⟩

/-- Brian's cube with faces numbered 1 to 6 -/
def brian_cube : Cube :=
  ⟨λ i => i.val + 1⟩

/-- Noah's cube with three faces showing 1 and three faces showing 4 -/
def noah_cube : Cube :=
  ⟨λ i => if i.val < 3 then 1 else 4⟩

/-- The probability of rolling a value less than or equal to 3 on Brian's cube -/
def brian_prob_le_3 : ℚ :=
  1/2

/-- The probability of rolling a 4 on Noah's cube -/
def noah_prob_4 : ℚ :=
  1/2

/-- The probability of both Brian and Noah rolling lower than Allison -/
def prob_both_lower : ℚ :=
  1/6

theorem allison_not_lowest_prob :
  1 - prob_both_lower = 5/6 :=
sorry

end NUMINAMATH_CALUDE_allison_not_lowest_prob_l601_60104


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l601_60120

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 3 + a 5 + a 11 + a 13 = 80 →
  a 8 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l601_60120


namespace NUMINAMATH_CALUDE_hall_length_l601_60108

/-- Proves that given the conditions of the hall and verandah, the length of the hall is 20 meters -/
theorem hall_length (hall_breadth : ℝ) (verandah_width : ℝ) (flooring_rate : ℝ) (total_cost : ℝ) :
  hall_breadth = 15 →
  verandah_width = 2.5 →
  flooring_rate = 3.5 →
  total_cost = 700 →
  ∃ (hall_length : ℝ),
    hall_length = 20 ∧
    (hall_length + 2 * verandah_width) * (hall_breadth + 2 * verandah_width) -
    hall_length * hall_breadth = total_cost / flooring_rate :=
by sorry

end NUMINAMATH_CALUDE_hall_length_l601_60108


namespace NUMINAMATH_CALUDE_both_players_is_zero_l601_60109

/-- The number of people who play both kabadi and kho kho -/
def both_players : ℕ := sorry

/-- The number of people who play kabadi (including those who play both) -/
def kabadi_players : ℕ := 10

/-- The number of people who play kho kho only -/
def kho_kho_only_players : ℕ := 15

/-- The total number of players -/
def total_players : ℕ := 25

/-- Theorem stating that the number of people playing both games is 0 -/
theorem both_players_is_zero : both_players = 0 := by
  sorry

#check both_players_is_zero

end NUMINAMATH_CALUDE_both_players_is_zero_l601_60109


namespace NUMINAMATH_CALUDE_average_price_approx_1_645_l601_60123

/-- Calculate the average price per bottle given the number and prices of large and small bottles, and a discount rate for large bottles. -/
def averagePricePerBottle (largeBottles smallBottles : ℕ) (largePricePerBottle smallPricePerBottle : ℚ) (discountRate : ℚ) : ℚ :=
  let largeCost := largeBottles * largePricePerBottle
  let largeDiscount := largeCost * discountRate
  let discountedLargeCost := largeCost - largeDiscount
  let smallCost := smallBottles * smallPricePerBottle
  let totalCost := discountedLargeCost + smallCost
  let totalBottles := largeBottles + smallBottles
  totalCost / totalBottles

/-- The average price per bottle is approximately $1.645 given the specific conditions. -/
theorem average_price_approx_1_645 :
  let largeBattles := 1325
  let smallBottles := 750
  let largePricePerBottle := 189/100  -- $1.89
  let smallPricePerBottle := 138/100  -- $1.38
  let discountRate := 5/100  -- 5%
  abs (averagePricePerBottle largeBattles smallBottles largePricePerBottle smallPricePerBottle discountRate - 1645/1000) < 1/1000 := by
  sorry


end NUMINAMATH_CALUDE_average_price_approx_1_645_l601_60123


namespace NUMINAMATH_CALUDE_negation_of_existence_equiv_forall_l601_60167

theorem negation_of_existence_equiv_forall :
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_equiv_forall_l601_60167


namespace NUMINAMATH_CALUDE_angle_between_AO₂_and_CO₁_is_45_degrees_l601_60113

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the orthocenter H
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the incenter of a triangle
def incenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the angle between two lines given by two points each
def angle_between_lines (P₁ Q₁ P₂ Q₂ : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem angle_between_AO₂_and_CO₁_is_45_degrees 
  (ABC : Triangle) 
  (acute_angled : sorry) -- Condition: ABC is acute-angled
  (angle_B_30 : sorry) -- Condition: ∠B = 30°
  : 
  let H := orthocenter ABC
  let O₁ := incenter ABC.A ABC.B H
  let O₂ := incenter ABC.C ABC.B H
  angle_between_lines ABC.A O₂ ABC.C O₁ = 45 := by sorry

end NUMINAMATH_CALUDE_angle_between_AO₂_and_CO₁_is_45_degrees_l601_60113


namespace NUMINAMATH_CALUDE_tank_fill_time_is_30_l601_60158

/-- Represents the time it takes to fill a tank given two pipes with different fill/empty rates and a specific operating scenario. -/
def tank_fill_time (fill_rate_A : ℚ) (empty_rate_B : ℚ) (both_open_time : ℚ) : ℚ :=
  let net_fill_rate := fill_rate_A - empty_rate_B
  let filled_portion := net_fill_rate * both_open_time
  let remaining_portion := 1 - filled_portion
  both_open_time + remaining_portion / fill_rate_A

/-- Theorem stating that under the given conditions, the tank will be filled in 30 minutes. -/
theorem tank_fill_time_is_30 :
  tank_fill_time (1/16) (1/24) 21 = 30 :=
by sorry

end NUMINAMATH_CALUDE_tank_fill_time_is_30_l601_60158


namespace NUMINAMATH_CALUDE_flour_to_add_l601_60114

/-- The total number of cups of flour required by the recipe. -/
def total_flour : ℕ := 7

/-- The number of cups of flour Mary has already added. -/
def flour_added : ℕ := 2

/-- The number of cups of flour Mary still needs to add. -/
def flour_needed : ℕ := total_flour - flour_added

theorem flour_to_add : flour_needed = 5 := by
  sorry

end NUMINAMATH_CALUDE_flour_to_add_l601_60114


namespace NUMINAMATH_CALUDE_earth_land_area_scientific_notation_l601_60178

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a given number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

theorem earth_land_area_scientific_notation :
  let earthLandArea : ℝ := 149000000
  let scientificNotation := toScientificNotation earthLandArea 3
  scientificNotation.coefficient = 1.49 ∧ scientificNotation.exponent = 8 := by
  sorry

end NUMINAMATH_CALUDE_earth_land_area_scientific_notation_l601_60178


namespace NUMINAMATH_CALUDE_min_value_w_z_cubes_l601_60147

/-- Given complex numbers w and z satisfying |w + z| = 1 and |w² + z²| = 14,
    the smallest possible value of |w³ + z³| is 41/2. -/
theorem min_value_w_z_cubes (w z : ℂ) 
    (h1 : Complex.abs (w + z) = 1)
    (h2 : Complex.abs (w^2 + z^2) = 14) :
    ∃ (m : ℝ), m = 41/2 ∧ ∀ (x : ℝ), x ≥ m → Complex.abs (w^3 + z^3) ≤ x :=
by sorry

end NUMINAMATH_CALUDE_min_value_w_z_cubes_l601_60147


namespace NUMINAMATH_CALUDE_goldfinch_percentage_is_30_percent_l601_60111

def number_of_goldfinches : ℕ := 6
def number_of_sparrows : ℕ := 9
def number_of_grackles : ℕ := 5

def total_birds : ℕ := number_of_goldfinches + number_of_sparrows + number_of_grackles

def goldfinch_percentage : ℚ := (number_of_goldfinches : ℚ) / (total_birds : ℚ) * 100

theorem goldfinch_percentage_is_30_percent : goldfinch_percentage = 30 := by
  sorry

end NUMINAMATH_CALUDE_goldfinch_percentage_is_30_percent_l601_60111


namespace NUMINAMATH_CALUDE_triangle_theorem_l601_60177

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.sqrt 3 * t.b * Real.sin (t.B + t.C) + t.a * Real.cos t.B = t.c) 
  (h2 : t.a = 6)
  (h3 : t.b + t.c = 6 + 6 * Real.sqrt 3) : 
  t.A = π / 6 ∧ 
  (1 / 2 : ℝ) * t.b * t.c * Real.sin t.A = 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l601_60177


namespace NUMINAMATH_CALUDE_defective_product_selection_l601_60151

def total_products : ℕ := 10
def qualified_products : ℕ := 8
def defective_products : ℕ := 2
def products_to_select : ℕ := 3

theorem defective_product_selection :
  (Nat.choose total_products products_to_select - 
   Nat.choose qualified_products products_to_select) = 64 := by
  sorry

end NUMINAMATH_CALUDE_defective_product_selection_l601_60151


namespace NUMINAMATH_CALUDE_present_age_of_A_l601_60138

/-- Given the ages of three people A, B, and C, prove that A's present age is 11 years. -/
theorem present_age_of_A (A B C : ℕ) : 
  A + B + C = 57 → 
  ∃ (x : ℕ), A - 3 = x ∧ B - 3 = 2 * x ∧ C - 3 = 3 * x → 
  A = 11 := by
sorry

end NUMINAMATH_CALUDE_present_age_of_A_l601_60138


namespace NUMINAMATH_CALUDE_modulus_of_z_l601_60183

def i : ℂ := Complex.I

theorem modulus_of_z (z : ℂ) (h : z * (1 + i) = 2 - i) : Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l601_60183


namespace NUMINAMATH_CALUDE_product_expansion_sum_l601_60130

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x : ℝ, (4 * x^2 - 3 * x + 2) * (5 - x) = a * x^3 + b * x^2 + c * x + d) →
  9 * a + 3 * b + c + d = 26 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l601_60130


namespace NUMINAMATH_CALUDE_rose_difference_is_34_l601_60189

/-- The number of red roses Mrs. Santiago has -/
def santiago_roses : ℕ := 58

/-- The number of red roses Mrs. Garrett has -/
def garrett_roses : ℕ := 24

/-- The difference in the number of red roses between Mrs. Santiago and Mrs. Garrett -/
def rose_difference : ℕ := santiago_roses - garrett_roses

theorem rose_difference_is_34 : rose_difference = 34 := by
  sorry

end NUMINAMATH_CALUDE_rose_difference_is_34_l601_60189


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l601_60182

theorem quadratic_equation_condition (a : ℝ) : 
  (∀ x, a * x^2 = (x + 1) * (x - 1)) → a ≠ 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l601_60182


namespace NUMINAMATH_CALUDE_equation_solution_l601_60136

theorem equation_solution (a : ℝ) : 
  ((4 - 2) / 2 + a = 4) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l601_60136


namespace NUMINAMATH_CALUDE_square_garden_area_l601_60100

theorem square_garden_area (s : ℝ) (h1 : s > 0) : 
  (4 * s = 40) → (s^2 = 2 * (4 * s) + 20) → s^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_area_l601_60100


namespace NUMINAMATH_CALUDE_find_integers_with_sum_and_lcm_l601_60102

theorem find_integers_with_sum_and_lcm : ∃ (a b : ℕ+), 
  (a + b : ℕ) = 3972 ∧ 
  Nat.lcm a b = 985928 ∧ 
  a = 1964 ∧ 
  b = 2008 := by
sorry

end NUMINAMATH_CALUDE_find_integers_with_sum_and_lcm_l601_60102


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l601_60134

theorem simplify_and_rationalize :
  (Real.sqrt 5 / Real.sqrt 2) * (Real.sqrt 9 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 14) =
  3 * Real.sqrt 420 / 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l601_60134


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l601_60141

/-- The line equation passes through a fixed point for all real k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l601_60141


namespace NUMINAMATH_CALUDE_ratio_equality_l601_60125

theorem ratio_equality (x y : ℚ) (h : x / (2 * y) = 27) :
  (7 * x + 6 * y) / (x - 2 * y) = 96 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l601_60125


namespace NUMINAMATH_CALUDE_stream_speed_l601_60172

theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) 
  (h1 : downstream_distance = 120)
  (h2 : upstream_distance = 60)
  (h3 : time = 2) :
  ∃ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = downstream_distance / time ∧
    boat_speed - stream_speed = upstream_distance / time ∧
    stream_speed = 15 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l601_60172


namespace NUMINAMATH_CALUDE_virginia_adrienne_difference_l601_60115

/-- Represents the teaching years of Virginia, Adrienne, and Dennis -/
structure TeachingYears where
  virginia : ℕ
  adrienne : ℕ
  dennis : ℕ

/-- The conditions of the teaching years problem -/
def TeachingProblem (t : TeachingYears) : Prop :=
  t.virginia + t.adrienne + t.dennis = 75 ∧
  t.dennis = 34 ∧
  ∃ (x : ℕ), t.virginia = t.adrienne + x ∧ t.virginia = t.dennis - x

/-- The theorem stating that Virginia has taught 9 more years than Adrienne -/
theorem virginia_adrienne_difference (t : TeachingYears) 
  (h : TeachingProblem t) : t.virginia - t.adrienne = 9 := by
  sorry

end NUMINAMATH_CALUDE_virginia_adrienne_difference_l601_60115


namespace NUMINAMATH_CALUDE_apple_lovers_joined_correct_number_joined_l601_60143

theorem apple_lovers_joined (total_apples : ℕ) (initial_per_person : ℕ) (decrease : ℕ) : ℕ :=
  let initial_group_size := total_apples / initial_per_person
  let final_per_person := initial_per_person - decrease
  let final_group_size := total_apples / final_per_person
  final_group_size - initial_group_size

theorem correct_number_joined :
  apple_lovers_joined 1430 22 9 = 45 :=
by sorry

end NUMINAMATH_CALUDE_apple_lovers_joined_correct_number_joined_l601_60143


namespace NUMINAMATH_CALUDE_max_apple_recipients_l601_60150

theorem max_apple_recipients : ∃ n : ℕ, n = 13 ∧ 
  (∀ k : ℕ, k > n → k * (k + 1) > 200) ∧
  (n * (n + 1) ≤ 200) := by
  sorry

end NUMINAMATH_CALUDE_max_apple_recipients_l601_60150


namespace NUMINAMATH_CALUDE_jury_duty_ratio_l601_60181

theorem jury_duty_ratio (jury_selection : ℕ) (trial_duration : ℕ) (jury_deliberation : ℕ) (total_days : ℕ) :
  jury_selection = 2 →
  jury_deliberation = 6 →
  total_days = 19 →
  total_days = jury_selection + trial_duration + jury_deliberation →
  (trial_duration : ℚ) / (jury_selection : ℚ) = 11 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_jury_duty_ratio_l601_60181


namespace NUMINAMATH_CALUDE_inequality_equivalence_l601_60190

theorem inequality_equivalence (x : ℝ) :
  (1 / x > -4 ∧ 1 / x < 3) ↔ (x > 1 / 3 ∨ x < -1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l601_60190


namespace NUMINAMATH_CALUDE_equation_solution_l601_60153

theorem equation_solution :
  ∃ x : ℝ, (5 * x - 3 * x = 360 + 6 * (x + 4)) ∧ (x = -96) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l601_60153


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l601_60193

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 18/49
  let a₃ : ℚ := 162/343
  let r : ℚ := a₂ / a₁
  r = 63/98 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l601_60193


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l601_60163

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²) / a -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (a^2 + b^2) / a
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → e = Real.sqrt 13 / 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l601_60163


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l601_60137

-- Define a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l601_60137


namespace NUMINAMATH_CALUDE_horner_first_coefficient_l601_60118

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

-- Define Horner's method for a 5th degree polynomial
def horner_method (a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((((a₅ * x + a₄) * x + a₃) * x + a₂) * x + a₁) * x + a₀

-- Theorem statement
theorem horner_first_coefficient (x : ℝ) :
  ∃ (a₁ : ℝ), horner_method 0.5 4 0 (-3) a₁ (-1) x = f x ∧ a₁ = 1 :=
sorry

end NUMINAMATH_CALUDE_horner_first_coefficient_l601_60118


namespace NUMINAMATH_CALUDE_division_problem_l601_60175

theorem division_problem (x : ℝ) : 25.25 / x = 0.012625 → x = 2000 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l601_60175


namespace NUMINAMATH_CALUDE_factorization_equality_l601_60157

theorem factorization_equality (a b : ℝ) : (a - b)^2 + 6*(b - a) + 9 = (a - b - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l601_60157


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l601_60176

/-- The number of letters in the word BANANA -/
def word_length : ℕ := 6

/-- The number of occurrences of the letter B in BANANA -/
def b_count : ℕ := 1

/-- The number of occurrences of the letter N in BANANA -/
def n_count : ℕ := 2

/-- The number of occurrences of the letter A in BANANA -/
def a_count : ℕ := 3

/-- The number of unique arrangements of the letters in BANANA -/
def banana_arrangements : ℕ := word_length.factorial / (b_count.factorial * n_count.factorial * a_count.factorial)

theorem banana_arrangement_count : banana_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l601_60176


namespace NUMINAMATH_CALUDE_tomato_egg_soup_min_time_l601_60105

/-- Represents a cooking step with its duration -/
structure CookingStep where
  name : String
  duration : ℕ

/-- The set of cooking steps for Tomato Egg Soup -/
def tomatoEggSoupSteps : List CookingStep := [
  ⟨"A", 1⟩,
  ⟨"B", 2⟩,
  ⟨"C", 3⟩,
  ⟨"D", 1⟩,
  ⟨"E", 1⟩
]

/-- Calculates the minimum time required to complete all cooking steps -/
def minCookingTime (steps : List CookingStep) : ℕ := sorry

/-- Theorem: The minimum time to make Tomato Egg Soup is 6 minutes -/
theorem tomato_egg_soup_min_time :
  minCookingTime tomatoEggSoupSteps = 6 := by sorry

end NUMINAMATH_CALUDE_tomato_egg_soup_min_time_l601_60105


namespace NUMINAMATH_CALUDE_power_mod_eleven_l601_60198

theorem power_mod_eleven : 3^225 ≡ 1 [MOD 11] := by sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l601_60198


namespace NUMINAMATH_CALUDE_impossibility_of_three_similar_parts_l601_60107

theorem impossibility_of_three_similar_parts :
  ∀ (x : ℝ), x > 0 → ¬∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = x ∧
    (a ≤ Real.sqrt 2 * b ∧ b ≤ Real.sqrt 2 * a) ∧
    (b ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * b) ∧
    (a ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * a) :=
by
  sorry


end NUMINAMATH_CALUDE_impossibility_of_three_similar_parts_l601_60107


namespace NUMINAMATH_CALUDE_glued_cubes_surface_area_l601_60149

/-- Represents a 3D shape formed by two glued cubes -/
structure GluedCubes where
  large_cube_side : ℝ
  small_cube_side : ℝ
  glued : Bool

/-- Calculate the surface area of the GluedCubes shape -/
def surface_area (shape : GluedCubes) : ℝ :=
  let large_cube_area := 6 * shape.large_cube_side ^ 2
  let small_cube_area := 5 * shape.small_cube_side ^ 2
  large_cube_area + small_cube_area

/-- The theorem stating that the surface area of the specific GluedCubes shape is 74 -/
theorem glued_cubes_surface_area :
  let shape := GluedCubes.mk 3 1 true
  surface_area shape = 74 := by
  sorry

end NUMINAMATH_CALUDE_glued_cubes_surface_area_l601_60149


namespace NUMINAMATH_CALUDE_polynomial_remainder_l601_60196

theorem polynomial_remainder (q : ℝ → ℝ) :
  (∃ r : ℝ → ℝ, ∀ x, q x = (x - 2) * r x + 3) →
  (∃ s : ℝ → ℝ, ∀ x, q x = (x + 3) * s x - 9) →
  ∃ t : ℝ → ℝ, ∀ x, q x = (x - 2) * (x + 3) * t x + (12/5 * x - 9/5) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l601_60196


namespace NUMINAMATH_CALUDE_yogurt_combinations_l601_60128

theorem yogurt_combinations (num_flavors : Nat) (num_toppings : Nat) : 
  num_flavors = 6 → num_toppings = 8 → num_flavors * (num_toppings.choose 3) = 336 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l601_60128


namespace NUMINAMATH_CALUDE_prob_king_ace_value_l601_60187

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the event of drawing a King as the first card -/
def first_card_king (d : Deck) : Finset (Fin 52) :=
  d.cards.filter (λ c => c ≤ 4)

/-- Represents the event of drawing an Ace as the second card -/
def second_card_ace (d : Deck) : Finset (Fin 52) :=
  d.cards.filter (λ c => c ≤ 4 ∧ c ≠ c)

/-- The probability of drawing a King first and an Ace second -/
def prob_king_ace (d : Deck) : ℚ :=
  (first_card_king d).card * (second_card_ace d).card / (d.cards.card * (d.cards.card - 1))

theorem prob_king_ace_value (d : Deck) : prob_king_ace d = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_ace_value_l601_60187


namespace NUMINAMATH_CALUDE_percentage_not_covering_politics_l601_60173

/-- Represents the percentage of reporters covering local politics in country X -/
def local_politics_coverage : ℝ := 28

/-- Represents the percentage of political reporters not covering local politics in country X -/
def non_local_politics_coverage : ℝ := 30

/-- Theorem stating that 60% of reporters do not cover politics given the conditions -/
theorem percentage_not_covering_politics :
  let total_political_coverage := local_politics_coverage / (1 - non_local_politics_coverage / 100)
  100 - total_political_coverage = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_covering_politics_l601_60173


namespace NUMINAMATH_CALUDE_solution_count_equals_r_l601_60110

def r (n : ℕ) : ℚ := (1/2 : ℚ) * (n + 1 : ℚ) + (1/4 : ℚ) * (1 + (-1)^n : ℚ)

def count_solutions (n : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => p.1 + 2 * p.2 = n) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card

theorem solution_count_equals_r (n : ℕ) : 
  (count_solutions n : ℚ) = r n :=
sorry

end NUMINAMATH_CALUDE_solution_count_equals_r_l601_60110


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l601_60161

theorem square_of_binomial_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 164*x + c = (x + a)^2) → c = 6724 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l601_60161


namespace NUMINAMATH_CALUDE_divisibility_property_l601_60160

theorem divisibility_property (n : ℕ) (a b : ℤ) :
  (a ≠ b) →
  (∀ m : ℕ, (n^m : ℤ) ∣ (a^m - b^m)) →
  (n : ℤ) ∣ a ∧ (n : ℤ) ∣ b :=
by sorry

end NUMINAMATH_CALUDE_divisibility_property_l601_60160


namespace NUMINAMATH_CALUDE_sqrt_four_squared_l601_60197

theorem sqrt_four_squared : (Real.sqrt 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_squared_l601_60197


namespace NUMINAMATH_CALUDE_restaurant_combinations_l601_60179

theorem restaurant_combinations (menu_items : ℕ) (special_dish : ℕ) : menu_items = 12 ∧ special_dish = 1 →
  (menu_items - special_dish) * (menu_items - special_dish) + 
  2 * special_dish * (menu_items - special_dish) = 143 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_combinations_l601_60179


namespace NUMINAMATH_CALUDE_share_of_y_is_36_l601_60124

/-- The share of y in rupees when a sum is divided among x, y, and z -/
def share_of_y (total : ℚ) (x_share : ℚ) (y_share : ℚ) (z_share : ℚ) : ℚ :=
  (y_share / x_share) * (total / (1 + y_share / x_share + z_share / x_share))

/-- Theorem: The share of y is 36 rupees given the problem conditions -/
theorem share_of_y_is_36 :
  share_of_y 156 1 (45/100) (1/2) = 36 := by
  sorry

#eval share_of_y 156 1 (45/100) (1/2)

end NUMINAMATH_CALUDE_share_of_y_is_36_l601_60124


namespace NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_l601_60186

theorem a_gt_one_sufficient_not_necessary (a : ℝ) (h : a ≠ 0) :
  (∀ a, a > 1 → a > 1/a) ∧
  (∃ a, a > 1/a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_l601_60186


namespace NUMINAMATH_CALUDE_complex_magnitude_l601_60142

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 3 - 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l601_60142


namespace NUMINAMATH_CALUDE_simplify_fraction_l601_60131

theorem simplify_fraction (x y : ℚ) (hx : x = 2) (hy : y = 3) :
  (18 * x^3 * y^2) / (9 * x^2 * y^4) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l601_60131


namespace NUMINAMATH_CALUDE_expression_evaluation_l601_60129

theorem expression_evaluation :
  let a : ℝ := 2 + Real.sqrt 3
  (a - 1 - (2 * a - 1) / (a + 1)) / ((a^2 - 4 * a + 4) / (a + 1)) = (2 * Real.sqrt 3 + 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l601_60129


namespace NUMINAMATH_CALUDE_partner_b_investment_l601_60165

/-- Represents the investment and profit share of a partner in a partnership. -/
structure Partner where
  investment : ℝ
  profitShare : ℝ

/-- Represents a partnership with three partners. -/
structure Partnership where
  a : Partner
  b : Partner
  c : Partner

/-- Theorem stating that given the conditions of the problem, partner b's investment is $21000. -/
theorem partner_b_investment (p : Partnership)
  (h1 : p.a.investment = 15000)
  (h2 : p.c.investment = 27000)
  (h3 : p.b.profitShare = 1540)
  (h4 : p.a.profitShare = 1100)
  : p.b.investment = 21000 := by
  sorry

#check partner_b_investment

end NUMINAMATH_CALUDE_partner_b_investment_l601_60165


namespace NUMINAMATH_CALUDE_multiply_special_polynomials_l601_60184

theorem multiply_special_polynomials (y : ℝ) :
  (y^4 + 30*y^2 + 900) * (y^2 - 30) = y^6 - 27000 := by
  sorry

end NUMINAMATH_CALUDE_multiply_special_polynomials_l601_60184


namespace NUMINAMATH_CALUDE_dinner_time_calculation_l601_60195

/-- Calculates the time spent eating dinner during a train ride given the total duration and time spent on other activities. -/
theorem dinner_time_calculation (total_duration reading_time movie_time nap_time : ℕ) 
  (h1 : total_duration = 9)
  (h2 : reading_time = 2)
  (h3 : movie_time = 3)
  (h4 : nap_time = 3) :
  total_duration - (reading_time + movie_time + nap_time) = 1 := by
  sorry

#check dinner_time_calculation

end NUMINAMATH_CALUDE_dinner_time_calculation_l601_60195


namespace NUMINAMATH_CALUDE_geometric_series_problem_l601_60164

theorem geometric_series_problem (x y : ℝ) (h : y ≠ 1.375) :
  (∑' n, x / y^n) = 10 →
  (∑' n, x / (x - 2*y)^n) = 5/4 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l601_60164


namespace NUMINAMATH_CALUDE_meeting_point_one_third_distance_l601_60117

/-- Given two points in a 2D plane, this function calculates a point that is a fraction of the distance from the first point to the second point. -/
def intermediatePoint (x1 y1 x2 y2 t : ℝ) : ℝ × ℝ :=
  (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

/-- Theorem stating that the point (10, 5) is one-third of the way from (8, 3) to (14, 9). -/
theorem meeting_point_one_third_distance :
  intermediatePoint 8 3 14 9 (1/3) = (10, 5) := by
sorry

end NUMINAMATH_CALUDE_meeting_point_one_third_distance_l601_60117


namespace NUMINAMATH_CALUDE_intersection_implies_a_values_l601_60121

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + (a^2-5) = 0}

theorem intersection_implies_a_values (a : ℝ) : A ∩ B a = {2} → a = -1 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_values_l601_60121


namespace NUMINAMATH_CALUDE_flower_bed_circumference_l601_60145

/-- Given a square garden with a circular flower bed, prove the circumference of the flower bed -/
theorem flower_bed_circumference 
  (a p t : ℝ) 
  (h1 : a > 0) 
  (h2 : p > 0) 
  (h3 : t > 0) 
  (h4 : a = 2 * p + 14.25) 
  (h5 : ∃ s : ℝ, s > 0 ∧ a = s^2 ∧ p = 4 * s) 
  (h6 : ∃ r : ℝ, r > 0 ∧ r = s / 4 ∧ t = a + π * r^2) : 
  ∃ C : ℝ, C = 4.75 * π := by sorry

end NUMINAMATH_CALUDE_flower_bed_circumference_l601_60145


namespace NUMINAMATH_CALUDE_initial_men_is_four_l601_60140

/-- The number of men initially checking exam papers -/
def initial_men : ℕ := 4

/-- The number of days for the initial group to check papers -/
def initial_days : ℕ := 8

/-- The number of hours per day for the initial group -/
def initial_hours_per_day : ℕ := 5

/-- The number of men in the second group -/
def second_men : ℕ := 2

/-- The number of days for the second group to check papers -/
def second_days : ℕ := 20

/-- The number of hours per day for the second group -/
def second_hours_per_day : ℕ := 8

/-- Theorem stating that the initial number of men is 4 -/
theorem initial_men_is_four :
  initial_men * initial_days * initial_hours_per_day = 
  (second_men * second_days * second_hours_per_day) / 2 :=
by sorry

end NUMINAMATH_CALUDE_initial_men_is_four_l601_60140


namespace NUMINAMATH_CALUDE_james_car_sale_l601_60148

/-- The percentage at which James sold his car -/
def sell_percentage : ℝ → Prop := λ P =>
  let old_car_value : ℝ := 20000
  let new_car_sticker : ℝ := 30000
  let new_car_discount : ℝ := 0.9
  let out_of_pocket : ℝ := 11000
  new_car_sticker * new_car_discount - old_car_value * (P / 100) = out_of_pocket

theorem james_car_sale : 
  sell_percentage 80 := by sorry

end NUMINAMATH_CALUDE_james_car_sale_l601_60148


namespace NUMINAMATH_CALUDE_gcd_lcm_product_28_45_l601_60199

theorem gcd_lcm_product_28_45 : Nat.gcd 28 45 * Nat.lcm 28 45 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_28_45_l601_60199


namespace NUMINAMATH_CALUDE_curve_equation_l601_60116

/-- Given a curve parameterized by (x,y) = (3t + 5, 6t - 8) where t is a real number,
    prove that the equation of the line is y = 2x - 18 -/
theorem curve_equation (t : ℝ) (x y : ℝ) 
    (h1 : x = 3 * t + 5) 
    (h2 : y = 6 * t - 8) : 
  y = 2 * x - 18 := by
  sorry

end NUMINAMATH_CALUDE_curve_equation_l601_60116


namespace NUMINAMATH_CALUDE_orchid_bushes_after_planting_l601_60135

/-- The number of orchid bushes in a park after planting new ones -/
def total_bushes (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem: The total number of orchid bushes after planting is the sum of initial and planted bushes -/
theorem orchid_bushes_after_planting (initial : ℕ) (planted : ℕ) :
  total_bushes initial planted = initial + planted := by
  sorry

/-- Example with given values -/
example : total_bushes 2 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_orchid_bushes_after_planting_l601_60135


namespace NUMINAMATH_CALUDE_all_expressions_distinct_exactly_five_distinct_expressions_l601_60171

/-- Represents the different ways to parenthesize 3^(3^(3^3)) -/
inductive ExpressionType
  | Type1  -- 3^(3^(3^3))
  | Type2  -- 3^((3^3)^3)
  | Type3  -- ((3^3)^3)^3
  | Type4  -- (3^(3^3))^3
  | Type5  -- (3^3)^(3^3)

/-- Evaluates the expression based on its type -/
noncomputable def evaluate (e : ExpressionType) : ℕ :=
  match e with
  | ExpressionType.Type1 => 3^(3^(3^3))
  | ExpressionType.Type2 => 3^((3^3)^3)
  | ExpressionType.Type3 => ((3^3)^3)^3
  | ExpressionType.Type4 => (3^(3^3))^3
  | ExpressionType.Type5 => (3^3)^(3^3)

/-- Theorem stating that all expression types result in distinct values -/
theorem all_expressions_distinct :
  ∀ (e1 e2 : ExpressionType), e1 ≠ e2 → evaluate e1 ≠ evaluate e2 := by
  sorry

/-- Theorem stating that there are exactly 5 distinct ways to parenthesize the expression -/
theorem exactly_five_distinct_expressions :
  ∃! (s : Finset ExpressionType), (∀ e, e ∈ s) ∧ s.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_all_expressions_distinct_exactly_five_distinct_expressions_l601_60171


namespace NUMINAMATH_CALUDE_sixth_term_is_27_eighth_term_is_46_l601_60106

-- First sequence
def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem sixth_term_is_27 : arithmetic_sequence 2 5 6 = 27 := by sorry

-- Second sequence
def even_indexed_term (n : ℕ) : ℕ → ℕ
  | 0 => 4
  | m + 1 => 2 * even_indexed_term n m + 1

def odd_indexed_term (n : ℕ) : ℕ → ℕ
  | 0 => 2
  | m + 1 => 2 * odd_indexed_term n m + 2

def combined_sequence (n : ℕ) : ℕ :=
  if n % 2 = 0 then even_indexed_term (n / 2) (n / 2 - 1)
  else odd_indexed_term ((n + 1) / 2) ((n - 1) / 2)

theorem eighth_term_is_46 : combined_sequence 8 = 46 := by sorry

end NUMINAMATH_CALUDE_sixth_term_is_27_eighth_term_is_46_l601_60106


namespace NUMINAMATH_CALUDE_bathtub_volume_l601_60155

/-- Represents the problem of calculating the volume of a bathtub filled with jello --/
def BathtubProblem (jello_per_pound : ℚ) (gallons_per_cubic_foot : ℚ) (pounds_per_gallon : ℚ) 
                   (cost_per_tablespoon : ℚ) (total_spent : ℚ) : Prop :=
  let tablespoons := total_spent / cost_per_tablespoon
  let pounds_of_water := tablespoons / jello_per_pound
  let gallons_of_water := pounds_of_water / pounds_per_gallon
  let cubic_feet := gallons_of_water / gallons_per_cubic_foot
  cubic_feet = 6

/-- The main theorem stating that given the problem conditions, the bathtub holds 6 cubic feet of water --/
theorem bathtub_volume : 
  BathtubProblem (3/2) (15/2) 8 (1/2) 270 := by
  sorry

#check bathtub_volume

end NUMINAMATH_CALUDE_bathtub_volume_l601_60155
