import Mathlib

namespace NUMINAMATH_CALUDE_parsley_sprigs_left_l801_80120

/-- Calculates the number of parsley sprigs left after decorating plates --/
theorem parsley_sprigs_left 
  (initial_sprigs : ℕ) 
  (whole_sprig_plates : ℕ) 
  (half_sprig_plates : ℕ) : 
  initial_sprigs = 25 → 
  whole_sprig_plates = 8 → 
  half_sprig_plates = 12 → 
  initial_sprigs - (whole_sprig_plates + half_sprig_plates / 2) = 11 :=
by sorry

end NUMINAMATH_CALUDE_parsley_sprigs_left_l801_80120


namespace NUMINAMATH_CALUDE_distance_city_A_to_B_l801_80147

/-- The distance between city A and city B given the travel times and speeds of Eddy and Freddy -/
theorem distance_city_A_to_B 
  (time_eddy : ℝ) 
  (time_freddy : ℝ) 
  (distance_AC : ℝ) 
  (speed_ratio : ℝ) : 
  time_eddy = 3 → 
  time_freddy = 4 → 
  distance_AC = 300 → 
  speed_ratio = 2.1333333333333333 → 
  time_eddy * (speed_ratio * (distance_AC / time_freddy)) = 480 :=
by sorry

end NUMINAMATH_CALUDE_distance_city_A_to_B_l801_80147


namespace NUMINAMATH_CALUDE_intersection_y_coordinate_l801_80166

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : y = parabola x

-- Define the slope of the tangent at a point
def tangent_slope (p : PointOnParabola) : ℝ := 4 * p.x

-- Define perpendicular tangents
def perpendicular_tangents (p1 p2 : PointOnParabola) : Prop :=
  tangent_slope p1 * tangent_slope p2 = -1

-- Theorem statement
theorem intersection_y_coordinate (A B : PointOnParabola) 
  (h : perpendicular_tangents A B) : 
  ∃ P : ℝ × ℝ, P.2 = -1/8 ∧ 
    (P.2 - A.y = tangent_slope A * (P.1 - A.x)) ∧
    (P.2 - B.y = tangent_slope B * (P.1 - B.x)) :=
sorry

end NUMINAMATH_CALUDE_intersection_y_coordinate_l801_80166


namespace NUMINAMATH_CALUDE_line_quadrants_l801_80137

theorem line_quadrants (a b c : ℝ) (ha : a > 0) (hb : b < 0) (hc : c > 0) :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (a * x₁ + b * y₁ - c = 0 ∧ x₁ > 0 ∧ y₁ > 0) ∧  -- First quadrant
    (a * x₂ + b * y₂ - c = 0 ∧ x₂ < 0 ∧ y₂ < 0) ∧  -- Third quadrant
    (a * x₃ + b * y₃ - c = 0 ∧ x₃ > 0 ∧ y₃ < 0) :=  -- Fourth quadrant
by
  sorry

end NUMINAMATH_CALUDE_line_quadrants_l801_80137


namespace NUMINAMATH_CALUDE_expression_evaluation_l801_80143

theorem expression_evaluation : 6^2 + 4*5 - 2^3 + 4^2/2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l801_80143


namespace NUMINAMATH_CALUDE_fish_and_shrimp_prices_l801_80112

/-- The regular price of fish per pound -/
def regular_fish_price : ℝ := 10

/-- The discounted price of fish per quarter-pound package -/
def discounted_fish_price : ℝ := 1.5

/-- The price of shrimp per half-pound -/
def shrimp_price : ℝ := 5

/-- The discount rate on fish -/
def discount_rate : ℝ := 0.6

theorem fish_and_shrimp_prices :
  (regular_fish_price * (1 - discount_rate) / 4 = discounted_fish_price) ∧
  (regular_fish_price = 2 * shrimp_price) :=
sorry

end NUMINAMATH_CALUDE_fish_and_shrimp_prices_l801_80112


namespace NUMINAMATH_CALUDE_system_solutions_l801_80159

/-- The system of equations has exactly eight solutions -/
theorem system_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ)),
    solutions.card = 8 ∧
    (∀ (x y z : ℝ), (x, y, z) ∈ solutions ↔
      ((x - 2)^2 + (y + 1)^2 = 5 ∧
       (x - 2)^2 + (z - 3)^2 = 13 ∧
       (y + 1)^2 + (z - 3)^2 = 10)) ∧
    solutions = {(0, 0, 0), (0, -2, 0), (0, 0, 6), (0, -2, 6),
                 (4, 0, 0), (4, -2, 0), (4, 0, 6), (4, -2, 6)} := by
  sorry


end NUMINAMATH_CALUDE_system_solutions_l801_80159


namespace NUMINAMATH_CALUDE_book_cost_calculation_l801_80188

theorem book_cost_calculation (initial_amount : ℕ) (books_bought : ℕ) (remaining_amount : ℕ) :
  initial_amount = 79 →
  books_bought = 9 →
  remaining_amount = 16 →
  (initial_amount - remaining_amount) / books_bought = 7 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_calculation_l801_80188


namespace NUMINAMATH_CALUDE_monthly_savings_prediction_l801_80108

/-- Linear regression equation for monthly savings prediction -/
def linear_regression (x : ℝ) (b_hat : ℝ) (a_hat : ℝ) : ℝ :=
  b_hat * x + a_hat

theorem monthly_savings_prediction 
  (n : ℕ) (x_bar : ℝ) (b_hat : ℝ) (a_hat : ℝ) :
  n = 10 →
  x_bar = 8 →
  b_hat = 0.3 →
  a_hat = -0.4 →
  linear_regression 7 b_hat a_hat = 1.7 :=
by sorry

end NUMINAMATH_CALUDE_monthly_savings_prediction_l801_80108


namespace NUMINAMATH_CALUDE_evaluate_expression_l801_80149

theorem evaluate_expression : 3^2 * 4 * 6^3 * Nat.factorial 7 = 39191040 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l801_80149


namespace NUMINAMATH_CALUDE_value_of_expression_l801_80124

theorem value_of_expression (a b : ℝ) (h : a - b = 1) : 3*a - 3*b - 4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l801_80124


namespace NUMINAMATH_CALUDE_solve_linear_equation_l801_80174

theorem solve_linear_equation (x y : ℝ) :
  2 * x - 3 * y = 4 → y = (2 * x - 4) / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l801_80174


namespace NUMINAMATH_CALUDE_area_of_triple_square_l801_80126

/-- Given a square (square I) with diagonal length a + b√2, 
    prove that the area of a square (square II) that is three times 
    the area of square I is 3a^2 + 6ab√2 + 6b^2 -/
theorem area_of_triple_square (a b : ℝ) : 
  let diagonal_I := a + b * Real.sqrt 2
  let area_II := 3 * (diagonal_I^2 / 2)
  area_II = 3 * a^2 + 6 * a * b * Real.sqrt 2 + 6 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triple_square_l801_80126


namespace NUMINAMATH_CALUDE_like_terms_imply_m_minus_n_eq_two_l801_80179

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def are_like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (x y : ℕ), term1 x y ≠ 0 ∧ term2 x y ≠ 0 → 
    ∃ (c1 c2 : ℚ), term1 x y = c1 * x^x * y^y ∧ term2 x y = c2 * x^x * y^y

/-- The first monomial 3x^m*y -/
def term1 (m : ℕ) (x y : ℕ) : ℚ := 3 * x^m * y

/-- The second monomial -x^3*y^n -/
def term2 (n : ℕ) (x y : ℕ) : ℚ := -1 * x^3 * y^n

theorem like_terms_imply_m_minus_n_eq_two (m n : ℕ) :
  are_like_terms (term1 m) (term2 n) → m - n = 2 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_m_minus_n_eq_two_l801_80179


namespace NUMINAMATH_CALUDE_negative_of_negative_equals_absolute_value_l801_80135

theorem negative_of_negative_equals_absolute_value : -(-5) = |(-5)| := by
  sorry

end NUMINAMATH_CALUDE_negative_of_negative_equals_absolute_value_l801_80135


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l801_80165

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 4 ^ 2 + 3 * a 4 + 1 = 0) →
  (a 12 ^ 2 + 3 * a 12 + 1 = 0) →
  a 8 = -1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l801_80165


namespace NUMINAMATH_CALUDE_extended_tile_pattern_ratio_l801_80116

theorem extended_tile_pattern_ratio :
  let initial_black : ℕ := 10
  let initial_white : ℕ := 15
  let initial_side_length : ℕ := (initial_black + initial_white).sqrt
  let extended_side_length : ℕ := initial_side_length + 2
  let border_black : ℕ := 4 * (extended_side_length - 1)
  let border_white : ℕ := 4 * (extended_side_length - 2)
  let total_black : ℕ := initial_black + border_black
  let total_white : ℕ := initial_white + border_white
  (total_black : ℚ) / (total_white : ℚ) = 26 / 23 := by
sorry

end NUMINAMATH_CALUDE_extended_tile_pattern_ratio_l801_80116


namespace NUMINAMATH_CALUDE_average_of_xyz_l801_80145

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 20) :
  (x + y + z) / 3 = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_xyz_l801_80145


namespace NUMINAMATH_CALUDE_moon_radius_scientific_notation_l801_80185

/-- The radius of the moon in meters -/
def moon_radius : ℝ := 1738000

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Theorem stating that the moon's radius is equal to its scientific notation representation -/
theorem moon_radius_scientific_notation :
  ∃ (sn : ScientificNotation), moon_radius = sn.coefficient * (10 : ℝ) ^ sn.exponent :=
sorry

end NUMINAMATH_CALUDE_moon_radius_scientific_notation_l801_80185


namespace NUMINAMATH_CALUDE_baseball_player_at_bats_against_left_handers_l801_80167

theorem baseball_player_at_bats_against_left_handers 
  (total_at_bats : ℕ) 
  (total_hits : ℕ) 
  (avg_left : ℚ) 
  (avg_right : ℚ) 
  (h : total_at_bats = 600) 
  (i : total_hits = 192) 
  (j : avg_left = 1/4) 
  (k : avg_right = 7/20) : 
  ∃ (left_at_bats right_at_bats : ℕ), 
    left_at_bats + right_at_bats = total_at_bats ∧ 
    left_at_bats * avg_left + right_at_bats * avg_right = total_hits ∧ 
    left_at_bats = 180 := by
  sorry

end NUMINAMATH_CALUDE_baseball_player_at_bats_against_left_handers_l801_80167


namespace NUMINAMATH_CALUDE_two_digit_divisible_number_exists_l801_80122

theorem two_digit_divisible_number_exists : ∃ n : ℕ, 
  10 ≤ n ∧ n ≤ 99 ∧ 
  n % 8 = 0 ∧ n % 12 = 0 ∧ n % 18 = 0 ∧
  60 ≤ n ∧ n ≤ 79 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_divisible_number_exists_l801_80122


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l801_80132

theorem solve_exponential_equation : ∃ x : ℝ, (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x = (256 : ℝ)^4 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l801_80132


namespace NUMINAMATH_CALUDE_smallest_pascal_family_pascal_family_with_five_children_l801_80194

/-- Represents a family with children -/
structure Family :=
  (boys : ℕ)
  (girls : ℕ)

/-- Defines the conditions for the Pascal family -/
def isPascalFamily (f : Family) : Prop :=
  f.boys ≥ 3 ∧ f.girls ≥ 2

/-- The total number of children in a family -/
def totalChildren (f : Family) : ℕ := f.boys + f.girls

/-- Theorem: The smallest possible number of children in a Pascal family is 5 -/
theorem smallest_pascal_family :
  ∀ f : Family, isPascalFamily f → totalChildren f ≥ 5 :=
by
  sorry

/-- Theorem: There exists a Pascal family with exactly 5 children -/
theorem pascal_family_with_five_children :
  ∃ f : Family, isPascalFamily f ∧ totalChildren f = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_pascal_family_pascal_family_with_five_children_l801_80194


namespace NUMINAMATH_CALUDE_tax_calculation_l801_80146

/-- Calculate the tax amount given gross pay and net pay -/
def calculate_tax (gross_pay : ℝ) (net_pay : ℝ) : ℝ :=
  gross_pay - net_pay

theorem tax_calculation :
  let gross_pay : ℝ := 450
  let net_pay : ℝ := 315
  calculate_tax gross_pay net_pay = 135 := by
sorry

end NUMINAMATH_CALUDE_tax_calculation_l801_80146


namespace NUMINAMATH_CALUDE_additional_fabric_needed_l801_80180

def yards_to_feet (yards : ℝ) : ℝ := yards * 3

def fabric_needed_for_dresses : ℝ :=
  2 * (yards_to_feet 5.5) +
  2 * (yards_to_feet 6) +
  2 * (yards_to_feet 6.5)

def current_fabric : ℝ := 10

theorem additional_fabric_needed :
  fabric_needed_for_dresses - current_fabric = 98 :=
by sorry

end NUMINAMATH_CALUDE_additional_fabric_needed_l801_80180


namespace NUMINAMATH_CALUDE_total_marbles_l801_80123

/-- The total number of marbles given the conditions of red, blue, and green marbles -/
theorem total_marbles (r : ℝ) (b : ℝ) (g : ℝ) : 
  r > 0 → 
  r = 1.5 * b → 
  g = 1.8 * r → 
  r + b + g = 3.467 * r := by
sorry


end NUMINAMATH_CALUDE_total_marbles_l801_80123


namespace NUMINAMATH_CALUDE_stones_for_hall_l801_80178

/-- Calculates the number of stones required to pave a hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℕ :=
  let hall_area := hall_length * hall_width * 100
  let stone_area := stone_length * stone_width
  (hall_area / stone_area).num.natAbs

/-- Theorem stating that 9000 stones are required to pave the given hall -/
theorem stones_for_hall : stones_required 72 30 4 6 = 9000 := by
  sorry

end NUMINAMATH_CALUDE_stones_for_hall_l801_80178


namespace NUMINAMATH_CALUDE_unique_self_opposite_l801_80151

theorem unique_self_opposite : ∃! x : ℝ, x = -x := by sorry

end NUMINAMATH_CALUDE_unique_self_opposite_l801_80151


namespace NUMINAMATH_CALUDE_age_difference_l801_80155

/-- Given three people A, B, and C, with B being 8 years old, B twice as old as C,
    and the total of their ages being 22, prove that A is 2 years older than B. -/
theorem age_difference (B C : ℕ) (A : ℕ) : 
  B = 8 → B = 2 * C → A + B + C = 22 → A = B + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_l801_80155


namespace NUMINAMATH_CALUDE_k_value_is_four_thirds_l801_80163

/-- The function f(x) = x + 1 -/
def f (x : ℝ) : ℝ := x + 1

/-- The function g(x) = kx^2 - x - (k+1) -/
def g (k : ℝ) (x : ℝ) : ℝ := k * x^2 - x - (k + 1)

/-- The theorem stating that k = 4/3 given the conditions -/
theorem k_value_is_four_thirds (k : ℝ) (h1 : k > 1) :
  (∀ x₁ ∈ Set.Icc 2 4, ∃ x₂ ∈ Set.Icc 2 4, f x₁ / g k x₁ = g k x₂ / f x₂) →
  k = 4/3 := by sorry

end NUMINAMATH_CALUDE_k_value_is_four_thirds_l801_80163


namespace NUMINAMATH_CALUDE_sum_of_last_three_digits_of_fibonacci_factorial_series_l801_80144

def fibonacci_factorial_series : List Nat := [1, 2, 3, 5, 8, 13, 21]

def last_three_digits (n : Nat) : Nat :=
  n % 1000

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem sum_of_last_three_digits_of_fibonacci_factorial_series :
  (fibonacci_factorial_series.map (λ n => last_three_digits (factorial n))).sum % 1000 = 249 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_three_digits_of_fibonacci_factorial_series_l801_80144


namespace NUMINAMATH_CALUDE_integral_sqrt_minus_x_equals_pi_half_l801_80115

theorem integral_sqrt_minus_x_equals_pi_half :
  ∫ x in (-1)..(1), (Real.sqrt (1 - x^2) - x) = π / 2 := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_minus_x_equals_pi_half_l801_80115


namespace NUMINAMATH_CALUDE_tory_cookie_sales_l801_80169

/-- Proves that Tory sold 7 packs of cookies to his uncle given the problem conditions -/
theorem tory_cookie_sales : 
  ∀ (total_goal : ℕ) (sold_to_grandmother : ℕ) (sold_to_neighbor : ℕ) (remaining_to_sell : ℕ),
    total_goal = 50 →
    sold_to_grandmother = 12 →
    sold_to_neighbor = 5 →
    remaining_to_sell = 26 →
    ∃ (sold_to_uncle : ℕ),
      sold_to_uncle = total_goal - remaining_to_sell - sold_to_grandmother - sold_to_neighbor ∧
      sold_to_uncle = 7 :=
by sorry

end NUMINAMATH_CALUDE_tory_cookie_sales_l801_80169


namespace NUMINAMATH_CALUDE_freshman_psychology_liberal_arts_percentage_l801_80150

/-- Represents the student categories -/
inductive StudentCategory
| Freshman
| Sophomore
| Junior
| Senior

/-- Represents the schools -/
inductive School
| LiberalArts
| Science
| Business

/-- Represents the distribution of students across categories and schools -/
structure StudentDistribution where
  totalStudents : ℕ
  categoryPercentage : StudentCategory → ℚ
  schoolPercentage : StudentCategory → School → ℚ
  psychologyMajorPercentage : ℚ

/-- The given student distribution -/
def givenDistribution : StudentDistribution :=
  { totalStudents := 1000,  -- Arbitrary total, doesn't affect the percentage
    categoryPercentage := fun c => match c with
      | StudentCategory.Freshman => 2/5
      | StudentCategory.Sophomore => 3/10
      | StudentCategory.Junior => 1/5
      | StudentCategory.Senior => 1/10,
    schoolPercentage := fun c s => match c, s with
      | StudentCategory.Freshman, School.LiberalArts => 3/5
      | StudentCategory.Freshman, School.Science => 3/10
      | StudentCategory.Freshman, School.Business => 1/10
      | _, _ => 0,  -- Other percentages are not needed for this problem
    psychologyMajorPercentage := 1/2 }

theorem freshman_psychology_liberal_arts_percentage 
  (d : StudentDistribution) 
  (h1 : d.categoryPercentage StudentCategory.Freshman = 2/5)
  (h2 : d.schoolPercentage StudentCategory.Freshman School.LiberalArts = 3/5)
  (h3 : d.psychologyMajorPercentage = 1/2) :
  d.categoryPercentage StudentCategory.Freshman * 
  d.schoolPercentage StudentCategory.Freshman School.LiberalArts * 
  d.psychologyMajorPercentage = 12/100 := by
  sorry

end NUMINAMATH_CALUDE_freshman_psychology_liberal_arts_percentage_l801_80150


namespace NUMINAMATH_CALUDE_quadratic_prime_roots_l801_80193

theorem quadratic_prime_roots (k : ℕ) : 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p + q = 63 ∧ p * q = k ∧ ∀ x : ℝ, x^2 - 63*x + k = 0 ↔ (x = p ∨ x = q)) → 
  k = 122 :=
sorry

end NUMINAMATH_CALUDE_quadratic_prime_roots_l801_80193


namespace NUMINAMATH_CALUDE_crosswalk_distance_l801_80102

/-- Given a parallelogram with the following properties:
  * One side has length 22 feet
  * An adjacent side has length 65 feet
  * The altitude perpendicular to the 22-foot side is 60 feet
  Then the altitude perpendicular to the 65-foot side is 264/13 feet. -/
theorem crosswalk_distance (a b h₁ h₂ : ℝ) 
  (ha : a = 22) 
  (hb : b = 65) 
  (hh₁ : h₁ = 60) : 
  a * h₁ = b * h₂ → h₂ = 264 / 13 := by
  sorry

#check crosswalk_distance

end NUMINAMATH_CALUDE_crosswalk_distance_l801_80102


namespace NUMINAMATH_CALUDE_largest_seven_digit_divisible_by_337_l801_80173

theorem largest_seven_digit_divisible_by_337 :
  ∀ n : ℕ, n ≤ 9999999 → n % 337 = 0 → n ≤ 9999829 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_seven_digit_divisible_by_337_l801_80173


namespace NUMINAMATH_CALUDE_bridge_length_is_4km_l801_80109

/-- The length of a bridge crossed by a man -/
def bridge_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: The length of a bridge is 4 km when crossed by a man walking at 10 km/hr in 24 minutes -/
theorem bridge_length_is_4km (speed : ℝ) (time : ℝ) 
    (h1 : speed = 10) 
    (h2 : time = 24 / 60) : 
  bridge_length speed time = 4 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_is_4km_l801_80109


namespace NUMINAMATH_CALUDE_min_stamps_for_60_cents_l801_80121

/-- Represents the number of ways to make a certain amount using given denominations -/
def numWays (amount : ℕ) (denominations : List ℕ) : ℕ :=
  sorry

/-- Represents the minimum number of coins needed to make a certain amount using given denominations -/
def minCoins (amount : ℕ) (denominations : List ℕ) : ℕ :=
  sorry

theorem min_stamps_for_60_cents :
  minCoins 60 [5, 6] = 10 :=
sorry

end NUMINAMATH_CALUDE_min_stamps_for_60_cents_l801_80121


namespace NUMINAMATH_CALUDE_starburst_candies_l801_80142

theorem starburst_candies (mm_ratio : ℕ) (starburst_ratio : ℕ) (total_mm : ℕ) : ℕ :=
  let starburst_count := (starburst_ratio * total_mm) / mm_ratio
  by
    sorry

#check starburst_candies 13 8 143 = 88

end NUMINAMATH_CALUDE_starburst_candies_l801_80142


namespace NUMINAMATH_CALUDE_hundreds_digit_of_factorial_difference_l801_80196

theorem hundreds_digit_of_factorial_difference : (25 - 20).factorial ≡ 0 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_hundreds_digit_of_factorial_difference_l801_80196


namespace NUMINAMATH_CALUDE_final_women_count_room_population_problem_l801_80127

/-- Represents the number of people in a room -/
structure RoomPopulation where
  men : ℕ
  women : ℕ

/-- Represents the changes in population -/
structure PopulationChange where
  menEntered : ℕ
  womenLeft : ℕ
  womenMultiplier : ℕ

/-- The theorem to prove -/
theorem final_women_count 
  (initialRatio : Rat) 
  (changes : PopulationChange) 
  (finalMenCount : ℕ) : ℕ :=
  by
    sorry

/-- The main theorem that encapsulates the problem -/
theorem room_population_problem : 
  final_women_count (7/8) ⟨4, 5, 3⟩ 16 = 27 :=
  by
    sorry

end NUMINAMATH_CALUDE_final_women_count_room_population_problem_l801_80127


namespace NUMINAMATH_CALUDE_ordered_pair_solution_l801_80181

theorem ordered_pair_solution (a b : ℤ) :
  Real.sqrt (9 - 8 * Real.sin (50 * π / 180)) = a + b * (1 / Real.sin (50 * π / 180)) →
  a = 3 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_ordered_pair_solution_l801_80181


namespace NUMINAMATH_CALUDE_polynomial_remainder_l801_80176

theorem polynomial_remainder (x : ℝ) : 
  let p := fun x => 5*x^4 - 9*x^3 + 3*x^2 - 7*x - 30
  let d := fun x => 3*x - 9
  p 3 = 138 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l801_80176


namespace NUMINAMATH_CALUDE_blue_marbles_count_l801_80186

/-- Given a bag of marbles with a 3:5 ratio of red to blue marbles and 18 red marbles,
    prove that there are 30 blue marbles. -/
theorem blue_marbles_count (red_count : ℕ) (ratio_red : ℕ) (ratio_blue : ℕ)
    (h_red_count : red_count = 18)
    (h_ratio : ratio_red = 3 ∧ ratio_blue = 5) :
    red_count * ratio_blue / ratio_red = 30 := by
  sorry

end NUMINAMATH_CALUDE_blue_marbles_count_l801_80186


namespace NUMINAMATH_CALUDE_second_person_share_correct_l801_80110

/-- Represents the rent sharing scenario -/
structure RentSharing where
  total_rent : ℕ
  base_share : ℕ
  first_multiplier : ℕ
  second_multiplier : ℕ
  third_multiplier : ℕ

/-- Calculates the share of the second person -/
def second_person_share (rs : RentSharing) : ℕ :=
  rs.base_share * rs.second_multiplier

/-- Theorem stating the correct share for the second person -/
theorem second_person_share_correct (rs : RentSharing) 
  (h1 : rs.total_rent = 5400)
  (h2 : rs.first_multiplier = 5)
  (h3 : rs.second_multiplier = 3)
  (h4 : rs.third_multiplier = 1)
  (h5 : rs.total_rent = rs.base_share * (rs.first_multiplier + rs.second_multiplier + rs.third_multiplier)) :
  second_person_share rs = 1800 := by
  sorry

#eval second_person_share { total_rent := 5400, base_share := 600, first_multiplier := 5, second_multiplier := 3, third_multiplier := 1 }

end NUMINAMATH_CALUDE_second_person_share_correct_l801_80110


namespace NUMINAMATH_CALUDE_power_function_property_l801_80154

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 4 / f 2 = 3) : 
  f (1/2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_power_function_property_l801_80154


namespace NUMINAMATH_CALUDE_xy_sum_inequality_l801_80140

theorem xy_sum_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + x*y = 3) :
  (x + y ≥ 2) ∧ (x + y = 2 ↔ x = 1 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_inequality_l801_80140


namespace NUMINAMATH_CALUDE_car_travel_distance_l801_80134

theorem car_travel_distance (speed : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) (distance : ℝ) : 
  speed = 80 →
  speed_increase = 40 →
  time_decrease = 0.5 →
  distance / speed - distance / (speed + speed_increase) = time_decrease →
  distance = 120 := by
  sorry

#check car_travel_distance

end NUMINAMATH_CALUDE_car_travel_distance_l801_80134


namespace NUMINAMATH_CALUDE_a_not_zero_l801_80190

theorem a_not_zero 
  (a b c d : ℝ) 
  (h1 : a / b < -3 * c / d) 
  (h2 : b * d ≠ 0) 
  (h3 : c = 2 * a) : 
  a ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_a_not_zero_l801_80190


namespace NUMINAMATH_CALUDE_modulus_of_z_l801_80164

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l801_80164


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_9_l801_80128

theorem ceiling_neg_sqrt_64_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_9_l801_80128


namespace NUMINAMATH_CALUDE_complement_of_union_equals_specific_set_l801_80113

-- Define the universal set U
def U : Set Int := {x | -3 < x ∧ x ≤ 4}

-- Define sets A and B
def A : Set Int := {-2, -1, 3}
def B : Set Int := {1, 2, 3}

-- State the theorem
theorem complement_of_union_equals_specific_set :
  (U \ (A ∪ B)) = {0, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_specific_set_l801_80113


namespace NUMINAMATH_CALUDE_angle_abc_measure_l801_80114

/-- A configuration of a regular hexagon with an inscribed square sharing a side. -/
structure HexagonSquareConfig where
  /-- The measure of an interior angle of the regular hexagon in degrees. -/
  hexagon_interior_angle : ℝ
  /-- The measure of an interior angle of the square in degrees. -/
  square_interior_angle : ℝ
  /-- Assumption that the hexagon is regular. -/
  hexagon_regular : hexagon_interior_angle = 120
  /-- Assumption that the inscribed shape is a square. -/
  square_regular : square_interior_angle = 90

/-- The theorem stating that the angle ABC in the given configuration is 45°. -/
theorem angle_abc_measure (config : HexagonSquareConfig) :
  let angle_bdc : ℝ := config.hexagon_interior_angle - config.square_interior_angle
  let angle_cbd : ℝ := (180 - angle_bdc) / 2
  let angle_abc : ℝ := config.hexagon_interior_angle - angle_cbd
  angle_abc = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_abc_measure_l801_80114


namespace NUMINAMATH_CALUDE_system_solutions_l801_80117

theorem system_solutions (x y z : ℝ) : 
  (x * (3 * y^2 + 1) = y * (y^2 + 3) ∧
   y * (3 * z^2 + 1) = z * (z^2 + 3) ∧
   z * (3 * x^2 + 1) = x * (x^2 + 3)) ↔ 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ 
   (x = -1 ∧ y = -1 ∧ z = -1) ∨ 
   (x = 0 ∧ y = 0 ∧ z = 0)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l801_80117


namespace NUMINAMATH_CALUDE_domain_transformation_l801_80199

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x)
def domain_f : Set ℝ := Set.Ioo (1/3) 1

-- Define the domain of f(3^x)
def domain_f_exp : Set ℝ := Set.Ico (-1) 0

-- Theorem statement
theorem domain_transformation (h : ∀ x ∈ domain_f, f x ≠ 0) :
  ∀ x, f (3^x) ≠ 0 ↔ x ∈ domain_f_exp :=
sorry

end NUMINAMATH_CALUDE_domain_transformation_l801_80199


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l801_80184

theorem partial_fraction_decomposition_product (A B C : ℚ) :
  (∀ x : ℚ, x ≠ 2 ∧ x ≠ -2 ∧ x ≠ 3 →
    (x^2 - 13) / ((x - 2) * (x + 2) * (x - 3)) =
    A / (x - 2) + B / (x + 2) + C / (x - 3)) →
  A * B * C = 81 / 100 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l801_80184


namespace NUMINAMATH_CALUDE_james_bag_weight_l801_80119

/-- The weight of James's bag given Oliver's bags' weights -/
theorem james_bag_weight (oliver_bag1 oliver_bag2 james_bag : ℝ) : 
  oliver_bag1 = (1 / 6) * james_bag →
  oliver_bag2 = (1 / 6) * james_bag →
  oliver_bag1 + oliver_bag2 = 6 →
  james_bag = 18 := by
  sorry

end NUMINAMATH_CALUDE_james_bag_weight_l801_80119


namespace NUMINAMATH_CALUDE_probability_non_yellow_jelly_bean_l801_80198

/-- The probability of selecting a non-yellow jelly bean from a bag -/
theorem probability_non_yellow_jelly_bean 
  (red : ℕ) (green : ℕ) (yellow : ℕ) (blue : ℕ)
  (h_red : red = 4)
  (h_green : green = 5)
  (h_yellow : yellow = 9)
  (h_blue : blue = 10) :
  (red + green + blue : ℚ) / (red + green + yellow + blue) = 19 / 28 := by
sorry

end NUMINAMATH_CALUDE_probability_non_yellow_jelly_bean_l801_80198


namespace NUMINAMATH_CALUDE_team_points_distribution_l801_80148

theorem team_points_distribution (x : ℝ) (y : ℕ) : 
  (1/3 : ℝ) * x + (3/8 : ℝ) * x + 18 + y = x ∧ 
  y ≤ 24 ∧ 
  ∀ (z : ℕ), z ≤ 8 → (y : ℝ) / 8 ≤ 3 →
  y = 17 :=
sorry

end NUMINAMATH_CALUDE_team_points_distribution_l801_80148


namespace NUMINAMATH_CALUDE_point_B_in_first_quadrant_l801_80197

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Determines if a point is in the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

theorem point_B_in_first_quadrant 
  (a b : ℝ) 
  (h : isInFirstQuadrant ⟨a, -b⟩) : 
  isInFirstQuadrant ⟨a, b⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_B_in_first_quadrant_l801_80197


namespace NUMINAMATH_CALUDE_classroom_desks_l801_80136

theorem classroom_desks :
  ∀ N y : ℕ,
  (3 * N = 4 * y) →  -- After 1/4 of students leave, 3/4N = 4/7y simplifies to 3N = 4y
  y ≤ 30 →
  y = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_classroom_desks_l801_80136


namespace NUMINAMATH_CALUDE_total_addresses_is_40_l801_80107

/-- The number of commencement addresses given by Governor Sandoval -/
def sandoval_addresses : ℕ := 12

/-- The number of commencement addresses given by Governor Hawkins -/
def hawkins_addresses : ℕ := sandoval_addresses / 2

/-- The number of commencement addresses given by Governor Sloan -/
def sloan_addresses : ℕ := sandoval_addresses + 10

/-- The total number of commencement addresses given by all three governors -/
def total_addresses : ℕ := sandoval_addresses + hawkins_addresses + sloan_addresses

theorem total_addresses_is_40 : total_addresses = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_addresses_is_40_l801_80107


namespace NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l801_80168

def a : ℝ × ℝ := (1, 2)
def b (y : ℝ) : ℝ × ℝ := (2, y)

theorem perpendicular_vectors_magnitude (y : ℝ) 
  (h : a.1 * (b y).1 + a.2 * (b y).2 = 0) : 
  ‖(2 : ℝ) • a + b y‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l801_80168


namespace NUMINAMATH_CALUDE_power_division_rule_l801_80191

theorem power_division_rule (a : ℝ) : a^4 / a^2 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l801_80191


namespace NUMINAMATH_CALUDE_alyssa_picked_32_limes_l801_80129

/-- The number of limes Alyssa picked -/
def alyssas_limes (total_limes fred_limes nancy_limes : ℕ) : ℕ :=
  total_limes - (fred_limes + nancy_limes)

/-- Proof that Alyssa picked 32 limes -/
theorem alyssa_picked_32_limes :
  alyssas_limes 103 36 35 = 32 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_picked_32_limes_l801_80129


namespace NUMINAMATH_CALUDE_plum_picking_l801_80157

/-- The number of plums picked by Melanie -/
def melanie_plums : ℕ := 4

/-- The number of plums picked by Dan -/
def dan_plums : ℕ := 9

/-- The number of plums picked by Sally -/
def sally_plums : ℕ := 3

/-- The total number of plums picked -/
def total_plums : ℕ := melanie_plums + dan_plums + sally_plums

theorem plum_picking :
  total_plums = 16 := by sorry

end NUMINAMATH_CALUDE_plum_picking_l801_80157


namespace NUMINAMATH_CALUDE_product_in_S_and_counterexample_l801_80160

def S : Set ℤ := {x | ∃ n : ℤ, x = n^2 + n + 1}

theorem product_in_S_and_counterexample :
  (∀ n : ℤ, (n^2 + n + 1) * ((n+1)^2 + (n+1) + 1) ∈ S) ∧
  (∃ a b : ℤ, a ∈ S ∧ b ∈ S ∧ a * b ∉ S) := by
  sorry

end NUMINAMATH_CALUDE_product_in_S_and_counterexample_l801_80160


namespace NUMINAMATH_CALUDE_parallel_postulate_l801_80177

-- Define a structure for points and lines in a 2D Euclidean plane
structure EuclideanPlane where
  Point : Type
  Line : Type
  on_line : Point → Line → Prop
  parallel : Line → Line → Prop

-- State the theorem
theorem parallel_postulate (plane : EuclideanPlane) 
  (l : plane.Line) (p : plane.Point) (h : ¬ plane.on_line p l) :
  ∃! m : plane.Line, plane.on_line p m ∧ plane.parallel m l :=
sorry

end NUMINAMATH_CALUDE_parallel_postulate_l801_80177


namespace NUMINAMATH_CALUDE_smallest_m_for_exact_tax_l801_80100

theorem smallest_m_for_exact_tax : ∃ (x : ℕ+), 
  (106 * x : ℕ) % 100 = 0 ∧ 
  (106 * x : ℕ) / 100 = 53 ∧ 
  ∀ (m : ℕ+), m < 53 → ¬∃ (y : ℕ+), (106 * y : ℕ) % 100 = 0 ∧ (106 * y : ℕ) / 100 = m := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_for_exact_tax_l801_80100


namespace NUMINAMATH_CALUDE_lake_crossing_wait_time_l801_80104

theorem lake_crossing_wait_time 
  (lake_width : ℝ) 
  (janet_initial_speed : ℝ) 
  (janet_speed_decrease : ℝ) 
  (sister_initial_speed : ℝ) 
  (sister_speed_increase : ℝ) 
  (h1 : lake_width = 60) 
  (h2 : janet_initial_speed = 30) 
  (h3 : janet_speed_decrease = 0.15) 
  (h4 : sister_initial_speed = 12) 
  (h5 : sister_speed_increase = 0.20) :
  ∃ (wait_time : ℝ), 
    abs (wait_time - 2.156862745) < 0.000001 ∧ 
    wait_time = 
      ((lake_width / sister_initial_speed) + 
       ((lake_width - sister_initial_speed) / (sister_initial_speed * (1 + sister_speed_increase)))) - 
      ((lake_width / (2 * janet_initial_speed)) + 
       (lake_width / (2 * janet_initial_speed * (1 - janet_speed_decrease)))) := by
  sorry

end NUMINAMATH_CALUDE_lake_crossing_wait_time_l801_80104


namespace NUMINAMATH_CALUDE_vanessa_points_l801_80175

/-- Calculates the points scored by a player given the total team points,
    number of other players, and average points of other players. -/
def player_points (total_points : ℕ) (other_players : ℕ) (avg_other_points : ℚ) : ℚ :=
  total_points - other_players * avg_other_points

/-- Proves that Vanessa scored 27 points given the problem conditions. -/
theorem vanessa_points :
  let total_points : ℕ := 48
  let other_players : ℕ := 6
  let avg_other_points : ℚ := 7/2
  player_points total_points other_players avg_other_points = 27 := by
sorry

#eval player_points 48 6 (7/2)

end NUMINAMATH_CALUDE_vanessa_points_l801_80175


namespace NUMINAMATH_CALUDE_complex_equation_problem_l801_80118

theorem complex_equation_problem (a b : ℝ) : 
  Complex.mk 1 (-2) = Complex.mk a b → a - b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_problem_l801_80118


namespace NUMINAMATH_CALUDE_unique_triple_sum_l801_80111

theorem unique_triple_sum (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y : ℚ) / z + (y * z : ℚ) / x + (z * x : ℚ) / y = 3 → x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_sum_l801_80111


namespace NUMINAMATH_CALUDE_new_average_production_l801_80138

theorem new_average_production (n : ℕ) (past_avg : ℝ) (today_prod : ℝ) :
  n = 9 →
  past_avg = 50 →
  today_prod = 100 →
  (n * past_avg + today_prod) / (n + 1) = 55 :=
by sorry

end NUMINAMATH_CALUDE_new_average_production_l801_80138


namespace NUMINAMATH_CALUDE_zero_to_positive_power_l801_80156

theorem zero_to_positive_power (n : ℕ+) : 0 ^ (n : ℕ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_to_positive_power_l801_80156


namespace NUMINAMATH_CALUDE_max_profit_at_16_l801_80187

/-- Represents the annual profit function for a factory -/
def annual_profit (x : ℕ+) : ℚ :=
  if x ≤ 20 then -x^2 + 32*x - 100 else 160 - x

/-- Theorem stating that the maximum annual profit occurs at 16 units -/
theorem max_profit_at_16 :
  ∀ x : ℕ+, annual_profit 16 ≥ annual_profit x :=
by sorry

end NUMINAMATH_CALUDE_max_profit_at_16_l801_80187


namespace NUMINAMATH_CALUDE_solve_sandwich_problem_l801_80141

def sandwich_problem (sandwich_cost : ℕ) (paid_amount : ℕ) (change_received : ℕ) : Prop :=
  let spent_amount := paid_amount - change_received
  let num_sandwiches := spent_amount / sandwich_cost
  num_sandwiches = 3

theorem solve_sandwich_problem :
  sandwich_problem 5 20 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_sandwich_problem_l801_80141


namespace NUMINAMATH_CALUDE_rectangles_must_be_squares_l801_80170

theorem rectangles_must_be_squares (n : ℕ) (is_prime : ℕ → Prop) 
  (total_squares : ℕ) (h_prime : is_prime total_squares) : 
  ∀ (a b : ℕ) (h_rect : ∀ i : Fin n, ∃ (k : ℕ), a * b = (total_squares / n) * k^2), a = b :=
by
  sorry

end NUMINAMATH_CALUDE_rectangles_must_be_squares_l801_80170


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l801_80171

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, (0 < x ∧ x < 7) → (|x - 2| < 5)) ∧
  (∃ x : ℝ, |x - 2| < 5 ∧ ¬(0 < x ∧ x < 7)) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l801_80171


namespace NUMINAMATH_CALUDE_monotonic_cubic_function_a_range_l801_80158

/-- The function f(x) = -x^3 + ax^2 - x - 1 is monotonic on ℝ if and only if a ∈ [-√3, √3] -/
theorem monotonic_cubic_function_a_range (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => -x^3 + a*x^2 - x - 1)) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_function_a_range_l801_80158


namespace NUMINAMATH_CALUDE_probability_sum_ten_l801_80130

/-- Represents an octahedral die with 8 faces -/
def OctahedralDie := Fin 8

/-- The set of possible outcomes when rolling two octahedral dice -/
def DiceOutcomes := OctahedralDie × OctahedralDie

/-- The total number of possible outcomes when rolling two octahedral dice -/
def totalOutcomes : ℕ := 64

/-- Predicate to check if a pair of dice rolls sums to 10 -/
def sumsToTen (roll : DiceOutcomes) : Prop :=
  (roll.1.val + 1) + (roll.2.val + 1) = 10

/-- The number of favorable outcomes (sum of 10) -/
def favorableOutcomes : ℕ := 5

/-- Theorem stating the probability of rolling a sum of 10 -/
theorem probability_sum_ten :
  (favorableOutcomes : ℚ) / totalOutcomes = 5 / 64 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_ten_l801_80130


namespace NUMINAMATH_CALUDE_flow_across_cut_equals_flow_from_single_vertex_l801_80125

-- Define a network
variable (N : Type*) [Fintype N]

-- Define the flow function
variable (f : Set N → Set N → ℝ)

-- Define the set of all vertices
variable (V : Set N)

-- Theorem statement
theorem flow_across_cut_equals_flow_from_single_vertex
  (S : Set N) (s : N) (h_s_in_S : s ∈ S) (h_S_subset_V : S ⊆ V) :
  f S (V \ S) = f {s} V :=
sorry

end NUMINAMATH_CALUDE_flow_across_cut_equals_flow_from_single_vertex_l801_80125


namespace NUMINAMATH_CALUDE_board_zero_condition_l801_80139

/-- Represents a board with positive integers -/
def Board (m n : ℕ) := Fin m → Fin n → ℕ+

/-- Checks if two positions are adjacent on the board -/
def adjacent (m n : ℕ) (x1 y1 x2 y2 : ℕ) : Prop :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y2 = y1 + 1)) ∨
  (y1 = y2 ∧ (x1 = x2 + 1 ∨ x2 = x1 + 1))

/-- Represents a move on the board -/
structure Move (m n : ℕ) where
  x1 : Fin m
  y1 : Fin n
  x2 : Fin m
  y2 : Fin n
  k : ℤ
  adj : adjacent m n x1.val y1.val x2.val y2.val

/-- Applies a move to the board -/
def applyMove (b : Board m n) (move : Move m n) : Board m n :=
  sorry

/-- Checks if a position is on a black square in chessboard coloring -/
def isBlack (x y : ℕ) : Bool :=
  (x + y) % 2 = 0

/-- Calculates the sum of numbers on black squares -/
def sumBlack (b : Board m n) : ℕ :=
  sorry

/-- Calculates the sum of numbers on white squares -/
def sumWhite (b : Board m n) : ℕ :=
  sorry

/-- Represents a sequence of moves -/
def MoveSequence (m n : ℕ) := List (Move m n)

/-- Applies a sequence of moves to the board -/
def applyMoveSequence (b : Board m n) (moves : MoveSequence m n) : Board m n :=
  sorry

/-- Checks if all numbers on the board are zero -/
def allZero (b : Board m n) : Prop :=
  ∀ x y, (b x y : ℕ) = 0

theorem board_zero_condition (m n : ℕ) :
  ∀ (b : Board m n),
    (∃ (moves : MoveSequence m n), allZero (applyMoveSequence b moves)) ↔
    (sumBlack b = sumWhite b) :=
  sorry

end NUMINAMATH_CALUDE_board_zero_condition_l801_80139


namespace NUMINAMATH_CALUDE_yellow_square_ratio_l801_80182

/-- Represents a square banner with a symmetric cross -/
structure Banner where
  side : ℝ
  cross_area_ratio : ℝ
  yellow_area_ratio : ℝ

/-- The banner satisfies the problem conditions -/
def valid_banner (b : Banner) : Prop :=
  b.side > 0 ∧
  b.cross_area_ratio = 0.25 ∧
  b.yellow_area_ratio > 0 ∧
  b.yellow_area_ratio < b.cross_area_ratio

theorem yellow_square_ratio (b : Banner) (h : valid_banner b) :
  b.yellow_area_ratio = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_yellow_square_ratio_l801_80182


namespace NUMINAMATH_CALUDE_derivative_even_implies_a_zero_l801_80131

/-- Given a real number a and a function f(x) = x^3 + ax^2 + (a-2)x,
    if f'(x) is an even function, then a = 0 -/
theorem derivative_even_implies_a_zero (a : ℝ) :
  let f := fun x : ℝ => x^3 + a*x^2 + (a-2)*x
  let f' := fun x : ℝ => 3*x^2 + 2*a*x + (a-2)
  (∀ x : ℝ, f' x = f' (-x)) →
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_even_implies_a_zero_l801_80131


namespace NUMINAMATH_CALUDE_boys_girls_percentage_difference_l801_80161

theorem boys_girls_percentage_difference : ¬ (∀ (girls boys : ℝ), 
  boys = girls * (1 + 0.25) → girls = boys * (1 - 0.25)) := by
  sorry

end NUMINAMATH_CALUDE_boys_girls_percentage_difference_l801_80161


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l801_80189

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x - 2 * x + 15 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y - 2 * y + 15 = 0 → y = x) ↔ 
  (m = 6 * Real.sqrt 5 - 2 ∨ m = -6 * Real.sqrt 5 - 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l801_80189


namespace NUMINAMATH_CALUDE_inequality_solution_set_l801_80152

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - x - 6) / (x - 4) ≥ 3 ↔ x ∈ Set.Iio 4 ∪ Set.Ioi 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l801_80152


namespace NUMINAMATH_CALUDE_kendy_transfer_proof_l801_80172

-- Define the initial balance
def initial_balance : ℚ := 190

-- Define the remaining balance
def remaining_balance : ℚ := 100

-- Define the amount transferred to mom
def amount_to_mom : ℚ := 60

-- Define the amount transferred to sister
def amount_to_sister : ℚ := amount_to_mom / 2

-- Theorem statement
theorem kendy_transfer_proof :
  initial_balance - (amount_to_mom + amount_to_sister) = remaining_balance :=
by sorry

end NUMINAMATH_CALUDE_kendy_transfer_proof_l801_80172


namespace NUMINAMATH_CALUDE_max_sum_of_products_l801_80183

/-- Represents the assignment of numbers to cube faces -/
def CubeAssignment := Fin 6 → Fin 6

/-- Computes the sum of products at cube vertices given a face assignment -/
def sumOfProducts (assignment : CubeAssignment) : ℕ :=
  sorry

/-- The set of all possible cube assignments -/
def allAssignments : Set CubeAssignment :=
  sorry

theorem max_sum_of_products :
  ∃ (assignment : CubeAssignment),
    assignment ∈ allAssignments ∧
    sumOfProducts assignment = 343 ∧
    ∀ (other : CubeAssignment),
      other ∈ allAssignments →
      sumOfProducts other ≤ 343 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_products_l801_80183


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l801_80162

/-- Given a circle C with equation x^2 + y^2 + y = 0, its center is (0, -1/2) and its radius is 1/2 -/
theorem circle_center_and_radius (x y : ℝ) :
  x^2 + y^2 + y = 0 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (0, -1/2) ∧
    radius = 1/2 ∧
    ∀ (point : ℝ × ℝ), point.1^2 + point.2^2 + point.2 = 0 ↔
      (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l801_80162


namespace NUMINAMATH_CALUDE_plates_arrangement_theorem_l801_80101

-- Define the number of plates of each color
def yellow_plates : ℕ := 4
def blue_plates : ℕ := 3
def red_plates : ℕ := 2
def purple_plates : ℕ := 1

-- Define the total number of plates
def total_plates : ℕ := yellow_plates + blue_plates + red_plates + purple_plates

-- Function to calculate circular arrangements
def circular_arrangements (n : ℕ) : ℕ :=
  (Nat.factorial n) / n

-- Function to calculate arrangements with restrictions
def arrangements_with_restrictions (total : ℕ) (y : ℕ) (b : ℕ) (r : ℕ) (p : ℕ) : ℕ :=
  circular_arrangements total - circular_arrangements (total - 1)

-- Theorem statement
theorem plates_arrangement_theorem :
  arrangements_with_restrictions total_plates yellow_plates blue_plates red_plates purple_plates = 980 := by
  sorry

end NUMINAMATH_CALUDE_plates_arrangement_theorem_l801_80101


namespace NUMINAMATH_CALUDE_alcohol_mixture_ratio_l801_80153

/-- Proves that mixing equal volumes of two alcohol solutions results in a specific alcohol-to-water ratio -/
theorem alcohol_mixture_ratio (volume : ℝ) (p_concentration q_concentration : ℝ)
  (h_volume_pos : volume > 0)
  (h_p_conc : p_concentration = 0.625)
  (h_q_conc : q_concentration = 0.875) :
  let total_volume := 2 * volume
  let total_alcohol := volume * (p_concentration + q_concentration)
  let total_water := total_volume - total_alcohol
  (total_alcohol / total_water) = 3 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_ratio_l801_80153


namespace NUMINAMATH_CALUDE_area_covered_by_specific_strips_l801_80133

/-- Calculates the area covered by four rectangular strips on a table. -/
def areaCoveredByStrips (lengths : List Nat) (width : Nat) (overlaps : Nat) : Nat :=
  let totalArea := (lengths.sum * width)
  let overlapArea := overlaps * width
  totalArea - overlapArea

/-- Theorem: The area covered by four specific strips is 33. -/
theorem area_covered_by_specific_strips :
  areaCoveredByStrips [12, 10, 8, 6] 1 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_area_covered_by_specific_strips_l801_80133


namespace NUMINAMATH_CALUDE_outfits_count_l801_80106

/-- The number of possible outfits given a set of clothing items -/
def number_of_outfits (shirts : ℕ) (ties : ℕ) (pants : ℕ) : ℕ :=
  shirts * pants * (ties + 1)

/-- Theorem: Given 5 shirts, 3 ties, and 4 pairs of pants, the total number of outfits is 80 -/
theorem outfits_count :
  number_of_outfits 5 3 4 = 80 := by
  sorry

#eval number_of_outfits 5 3 4

end NUMINAMATH_CALUDE_outfits_count_l801_80106


namespace NUMINAMATH_CALUDE_particle_speed_l801_80192

/-- Given a particle with position (3t + 4, 5t - 9) at time t, 
    its speed after a time interval of 2 units is √136. -/
theorem particle_speed (t : ℝ) : 
  let pos (t : ℝ) := (3 * t + 4, 5 * t - 9)
  let Δt := 2
  let Δx := (pos (t + Δt)).1 - (pos t).1
  let Δy := (pos (t + Δt)).2 - (pos t).2
  Real.sqrt (Δx ^ 2 + Δy ^ 2) = Real.sqrt 136 := by
  sorry

end NUMINAMATH_CALUDE_particle_speed_l801_80192


namespace NUMINAMATH_CALUDE_negation_of_all_children_good_l801_80105

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Child : U → Prop)
variable (GoodAtMusic : U → Prop)

-- Define the original statement and its negation
def AllChildrenGood : Prop := ∀ x, Child x → GoodAtMusic x
def AllChildrenPoor : Prop := ∀ x, Child x → ¬GoodAtMusic x

-- Theorem statement
theorem negation_of_all_children_good :
  AllChildrenPoor U Child GoodAtMusic ↔ ¬AllChildrenGood U Child GoodAtMusic :=
sorry

end NUMINAMATH_CALUDE_negation_of_all_children_good_l801_80105


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_integers_l801_80103

theorem largest_divisor_of_consecutive_odd_integers (n : ℕ) :
  ∃ (Q : ℕ), Q = (2*n - 3) * (2*n - 1) * (2*n + 1) * (2*n + 3) ∧
  15 ∣ Q ∧
  ∀ (k : ℕ), k > 15 → ¬(∀ (m : ℕ), k ∣ ((2*m - 3) * (2*m - 1) * (2*m + 1) * (2*m + 3))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_integers_l801_80103


namespace NUMINAMATH_CALUDE_inequality_proof_l801_80195

theorem inequality_proof (a b c d : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (h_sum : a*b + b*c + c*d + d*a = 1) : 
  a^3 / (b+c+d) + b^3 / (c+d+a) + c^3 / (d+a+b) + d^3 / (a+b+c) ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l801_80195
