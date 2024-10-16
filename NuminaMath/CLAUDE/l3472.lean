import Mathlib

namespace NUMINAMATH_CALUDE_zero_at_one_zero_at_five_value_at_three_l3472_347296

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

/-- The quadratic function has a zero at x = 1 -/
theorem zero_at_one : f 1 = 0 := by sorry

/-- The quadratic function has a zero at x = 5 -/
theorem zero_at_five : f 5 = 0 := by sorry

/-- The quadratic function takes the value 10 when x = 3 -/
theorem value_at_three : f 3 = 10 := by sorry

end NUMINAMATH_CALUDE_zero_at_one_zero_at_five_value_at_three_l3472_347296


namespace NUMINAMATH_CALUDE_landscape_length_l3472_347213

/-- A rectangular landscape with a playground -/
structure Landscape where
  breadth : ℝ
  length : ℝ
  playground_area : ℝ
  length_is_four_times_breadth : length = 4 * breadth
  playground_area_is_1200 : playground_area = 1200
  playground_is_one_third : playground_area = (1/3) * (length * breadth)

/-- The length of the landscape is 120 meters -/
theorem landscape_length (L : Landscape) : L.length = 120 := by
  sorry

end NUMINAMATH_CALUDE_landscape_length_l3472_347213


namespace NUMINAMATH_CALUDE_range_of_a_l3472_347223

-- Define propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

-- Define the theorem
theorem range_of_a :
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∀ a : ℝ, (a < 0 ∨ (1/4 < a ∧ a < 4)) ↔ (p a ∨ q a) ∧ ¬(p a ∧ q a)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3472_347223


namespace NUMINAMATH_CALUDE_exists_real_sqrt_x_minus_one_l3472_347219

theorem exists_real_sqrt_x_minus_one : ∃ x : ℝ, ∃ y : ℝ, y ^ 2 = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_real_sqrt_x_minus_one_l3472_347219


namespace NUMINAMATH_CALUDE_cost_per_meal_is_8_l3472_347283

-- Define the number of adults
def num_adults : ℕ := 2

-- Define the number of children
def num_children : ℕ := 5

-- Define the total bill amount
def total_bill : ℚ := 56

-- Define the total number of people
def total_people : ℕ := num_adults + num_children

-- Theorem to prove
theorem cost_per_meal_is_8 : 
  total_bill / total_people = 8 := by sorry

end NUMINAMATH_CALUDE_cost_per_meal_is_8_l3472_347283


namespace NUMINAMATH_CALUDE_Z_in_third_quadrant_l3472_347234

-- Define the complex number Z
def Z : ℂ := -1 + (1 - Complex.I)^2

-- Theorem statement
theorem Z_in_third_quadrant : 
  Z.re < 0 ∧ Z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_Z_in_third_quadrant_l3472_347234


namespace NUMINAMATH_CALUDE_six_digit_permutations_count_l3472_347232

/-- The number of different positive, six-digit integers that can be formed using the digits 1, 2, 2, 5, 9, and 9 -/
def six_digit_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of such integers is 180 -/
theorem six_digit_permutations_count : six_digit_permutations = 180 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_permutations_count_l3472_347232


namespace NUMINAMATH_CALUDE_suzanna_textbook_pages_l3472_347277

/-- Calculate the total number of pages in Suzanna's textbooks --/
theorem suzanna_textbook_pages : 
  let history_pages : ℕ := 160
  let geography_pages : ℕ := history_pages + 70
  let math_pages : ℕ := (history_pages + geography_pages) / 2
  let science_pages : ℕ := 2 * history_pages
  history_pages + geography_pages + math_pages + science_pages = 905 :=
by sorry

end NUMINAMATH_CALUDE_suzanna_textbook_pages_l3472_347277


namespace NUMINAMATH_CALUDE_quadrilateral_sides_diagonals_inequality_l3472_347228

/-- Theorem: For any quadrilateral, the sum of the squares of its sides is not less than
    the sum of the squares of its diagonals. -/
theorem quadrilateral_sides_diagonals_inequality 
  (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) : 
  (x₂ - x₁)^2 + (y₂ - y₁)^2 + (x₃ - x₂)^2 + (y₃ - y₂)^2 + 
  (x₄ - x₃)^2 + (y₄ - y₃)^2 + (x₄ - x₁)^2 + (y₄ - y₁)^2 ≥ 
  (x₃ - x₁)^2 + (y₃ - y₁)^2 + (x₄ - x₂)^2 + (y₄ - y₂)^2 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_sides_diagonals_inequality_l3472_347228


namespace NUMINAMATH_CALUDE_minimum_m_value_l3472_347241

theorem minimum_m_value (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : ∀ a b c, a > b ∧ b > c → (1 / (a - b) + m / (b - c) ≥ 9 / (a - c))) : 
  m ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_minimum_m_value_l3472_347241


namespace NUMINAMATH_CALUDE_sliding_ladder_inequality_l3472_347244

/-- Represents a sliding ladder against a wall -/
structure SlidingLadder where
  length : ℝ
  topSlideDistance : ℝ
  bottomSlipDistance : ℝ

/-- The bottom slip distance is always greater than the top slide distance for a sliding ladder -/
theorem sliding_ladder_inequality (ladder : SlidingLadder) :
  ladder.bottomSlipDistance > ladder.topSlideDistance :=
sorry

end NUMINAMATH_CALUDE_sliding_ladder_inequality_l3472_347244


namespace NUMINAMATH_CALUDE_two_team_property_min_teams_three_team_property_min_teams_l3472_347271

/-- A tournament is a relation between teams representing victories -/
def Tournament (α : Type*) := α → α → Prop

/-- In a tournament, team a has defeated team b -/
def Defeated {α : Type*} (t : Tournament α) (a b : α) : Prop := t a b

/-- A tournament satisfies the two-team property if for any two teams,
    there exists a third team that has defeated both -/
def TwoTeamProperty {α : Type*} (t : Tournament α) : Prop :=
  ∀ a b : α, ∃ c : α, Defeated t c a ∧ Defeated t c b

/-- A tournament satisfies the three-team property if for any three teams,
    there exists a fourth team that has defeated all three -/
def ThreeTeamProperty {α : Type*} (t : Tournament α) : Prop :=
  ∀ a b c : α, ∃ d : α, Defeated t d a ∧ Defeated t d b ∧ Defeated t d c

theorem two_team_property_min_teams
  {α : Type*} [Fintype α] (t : Tournament α) (h : TwoTeamProperty t) :
  Fintype.card α ≥ 7 := by
  sorry

theorem three_team_property_min_teams
  {α : Type*} [Fintype α] (t : Tournament α) (h : ThreeTeamProperty t) :
  Fintype.card α ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_two_team_property_min_teams_three_team_property_min_teams_l3472_347271


namespace NUMINAMATH_CALUDE_root_in_interval_l3472_347208

def f (x : ℝ) := x^3 - 2*x - 1

theorem root_in_interval :
  f 1 < 0 →
  f 2 > 0 →
  f (3/2) < 0 →
  ∃ x : ℝ, 3/2 < x ∧ x < 2 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l3472_347208


namespace NUMINAMATH_CALUDE_polynomial_no_x_term_l3472_347270

theorem polynomial_no_x_term (n : ℚ) : 
  (∀ x, (x + n) * (3 * x - 1) = 3 * x^2 - n) → n = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_no_x_term_l3472_347270


namespace NUMINAMATH_CALUDE_paving_stone_size_l3472_347260

theorem paving_stone_size (length width : ℝ) (num_stones : ℕ) (stone_side : ℝ) : 
  length = 30 → 
  width = 18 → 
  num_stones = 135 → 
  (length * width) = (num_stones : ℝ) * stone_side^2 → 
  stone_side = 2 := by
  sorry

end NUMINAMATH_CALUDE_paving_stone_size_l3472_347260


namespace NUMINAMATH_CALUDE_gift_amount_proof_l3472_347247

/-- The amount of money Josie received as a gift -/
def gift_amount : ℕ := 50

/-- The cost of one cassette tape -/
def cassette_cost : ℕ := 9

/-- The number of cassette tapes Josie plans to buy -/
def num_cassettes : ℕ := 2

/-- The cost of the headphone set -/
def headphone_cost : ℕ := 25

/-- The amount of money Josie will have left after her purchases -/
def money_left : ℕ := 7

/-- Theorem stating that the gift amount is equal to the sum of the purchases and remaining money -/
theorem gift_amount_proof : 
  gift_amount = num_cassettes * cassette_cost + headphone_cost + money_left :=
by sorry

end NUMINAMATH_CALUDE_gift_amount_proof_l3472_347247


namespace NUMINAMATH_CALUDE_polynomial_absolute_value_l3472_347200

/-- A second-degree polynomial with real coefficients -/
def SecondDegreePolynomial (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

/-- The absolute value of f at 1, 2, and 3 is equal to 9 -/
def AbsValueCondition (f : ℝ → ℝ) : Prop :=
  |f 1| = 9 ∧ |f 2| = 9 ∧ |f 3| = 9

theorem polynomial_absolute_value (f : ℝ → ℝ) 
  (h1 : SecondDegreePolynomial f) 
  (h2 : AbsValueCondition f) : 
  |f 0| = 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_absolute_value_l3472_347200


namespace NUMINAMATH_CALUDE_rachel_total_problems_l3472_347284

/-- The number of math problems Rachel solved in total -/
def total_problems (problems_per_minute : ℕ) (minutes : ℕ) (problems_next_day : ℕ) : ℕ :=
  problems_per_minute * minutes + problems_next_day

/-- Theorem stating that Rachel solved 151 math problems in total -/
theorem rachel_total_problems :
  total_problems 7 18 25 = 151 := by
  sorry

end NUMINAMATH_CALUDE_rachel_total_problems_l3472_347284


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_sum_l3472_347299

theorem rectangular_prism_diagonals_sum 
  (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 50) 
  (h2 : x*y + y*z + z*x = 47) : 
  4 * Real.sqrt (x^2 + y^2 + z^2) = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_sum_l3472_347299


namespace NUMINAMATH_CALUDE_pqr_sum_bounds_l3472_347243

theorem pqr_sum_bounds (p q r : ℝ) (h : 5 * (p + q + r) = p^2 + q^2 + r^2) :
  let R := p*q + p*r + q*r
  ∃ (N n : ℝ),
    (∀ x y z : ℝ, 5 * (x + y + z) = x^2 + y^2 + z^2 → x*y + x*z + y*z ≤ N) ∧
    (∀ x y z : ℝ, 5 * (x + y + z) = x^2 + y^2 + z^2 → n ≤ x*y + x*z + y*z) ∧
    N = 150 ∧
    n = -12.5 ∧
    N + 15*n = -37.5 :=
by sorry

end NUMINAMATH_CALUDE_pqr_sum_bounds_l3472_347243


namespace NUMINAMATH_CALUDE_journey_time_ratio_l3472_347275

/-- Represents a two-part journey with given speeds and times -/
structure Journey where
  v : ℝ  -- Initial speed
  t : ℝ  -- Initial time
  total_distance : ℝ  -- Total distance traveled

/-- The theorem statement -/
theorem journey_time_ratio 
  (j : Journey) 
  (h1 : j.v = 30)  -- Initial speed is 30 mph
  (h2 : j.v * j.t + (2 * j.v) * (2 * j.t) = j.total_distance)  -- Total distance equation
  (h3 : j.total_distance = 75)  -- Total distance is 75 miles
  : j.t / (2 * j.t) = 1 / 2 := by
  sorry

#check journey_time_ratio

end NUMINAMATH_CALUDE_journey_time_ratio_l3472_347275


namespace NUMINAMATH_CALUDE_ian_says_smallest_unclaimed_number_l3472_347203

/-- Represents a student in the counting game -/
structure Student where
  name : String
  index : Nat

/-- The list of students in alphabetical order -/
def students : List Student := [
  ⟨"Alice", 0⟩, ⟨"Barbara", 1⟩, ⟨"Candice", 2⟩, ⟨"Debbie", 3⟩,
  ⟨"Eliza", 4⟩, ⟨"Fatima", 5⟩, ⟨"Greg", 6⟩, ⟨"Helen", 7⟩
]

/-- The maximum number in the counting sequence -/
def maxNumber : Nat := 1200

/-- Determines if a student says a given number -/
def saysNumber (s : Student) (n : Nat) : Prop :=
  n ≤ maxNumber ∧ n % (4 * 4^s.index) ≠ 0

/-- The number that Ian says -/
def iansNumber : Nat := 1021

/-- Theorem stating that Ian's number is the smallest not said by any other student -/
theorem ian_says_smallest_unclaimed_number :
  (∀ n < iansNumber, ∃ s ∈ students, saysNumber s n) ∧
  (∀ s ∈ students, ¬saysNumber s iansNumber) :=
sorry

end NUMINAMATH_CALUDE_ian_says_smallest_unclaimed_number_l3472_347203


namespace NUMINAMATH_CALUDE_arcsin_arccos_bound_l3472_347211

theorem arcsin_arccos_bound (x y : ℝ) (h : x^2 + y^2 = 1) :
  -5*π/2 ≤ 3 * Real.arcsin x - 2 * Real.arccos y ∧
  3 * Real.arcsin x - 2 * Real.arccos y ≤ π/2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_arccos_bound_l3472_347211


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_tangent_circle_l3472_347265

theorem hyperbola_asymptote_tangent_circle (m : ℝ) : 
  m > 0 →
  (∀ x y : ℝ, y^2 - x^2/m^2 = 1 → x^2 + y^2 - 4*y + 3 = 0) →
  m = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_tangent_circle_l3472_347265


namespace NUMINAMATH_CALUDE_cistern_length_l3472_347276

/-- Given a cistern with specified dimensions, calculate its length -/
theorem cistern_length (width : Real) (water_depth : Real) (wet_area : Real)
  (h1 : width = 4)
  (h2 : water_depth = 1.25)
  (h3 : wet_area = 49)
  : ∃ (length : Real), length = wet_area / (width + 2 * water_depth) :=
by
  sorry

#check cistern_length

end NUMINAMATH_CALUDE_cistern_length_l3472_347276


namespace NUMINAMATH_CALUDE_factorization_sum_l3472_347252

theorem factorization_sum (a b : ℤ) :
  (∀ x : ℚ, 25 * x^2 - 85 * x - 150 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = -24 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l3472_347252


namespace NUMINAMATH_CALUDE_total_cost_of_flowers_l3472_347239

/-- The cost of a single flower in dollars -/
def flower_cost : ℕ := 3

/-- The number of roses bought -/
def roses_bought : ℕ := 2

/-- The number of daisies bought -/
def daisies_bought : ℕ := 2

/-- The total number of flowers bought -/
def total_flowers : ℕ := roses_bought + daisies_bought

/-- The theorem stating the total cost of the flowers -/
theorem total_cost_of_flowers : total_flowers * flower_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_flowers_l3472_347239


namespace NUMINAMATH_CALUDE_soup_offer_ratio_l3472_347226

/-- Represents the soup can offer --/
structure SoupOffer where
  total_cans : ℕ
  normal_price : ℚ
  total_paid : ℚ

/-- Calculates the buy to get ratio for a soup offer --/
def buyToGetRatio (offer : SoupOffer) : ℚ × ℚ :=
  let paid_cans := offer.total_paid / offer.normal_price
  let free_cans := offer.total_cans - paid_cans
  (paid_cans, free_cans)

/-- Theorem stating that the given offer results in a 1:1 ratio --/
theorem soup_offer_ratio (offer : SoupOffer) 
  (h1 : offer.total_cans = 30)
  (h2 : offer.normal_price = 0.6)
  (h3 : offer.total_paid = 9) :
  buyToGetRatio offer = (15, 15) := by
  sorry

#eval buyToGetRatio ⟨30, 0.6, 9⟩

end NUMINAMATH_CALUDE_soup_offer_ratio_l3472_347226


namespace NUMINAMATH_CALUDE_isosceles_triangle_parallel_cut_l3472_347285

/-- An isosceles triangle with given area and altitude --/
structure IsoscelesTriangle :=
  (area : ℝ)
  (altitude : ℝ)

/-- A line segment parallel to the base of the triangle --/
structure ParallelLine :=
  (length : ℝ)
  (trapezoidArea : ℝ)

/-- The theorem statement --/
theorem isosceles_triangle_parallel_cut (t : IsoscelesTriangle) (l : ParallelLine) :
  t.area = 150 ∧ t.altitude = 30 ∧ l.trapezoidArea = 100 →
  l.length = 10 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_parallel_cut_l3472_347285


namespace NUMINAMATH_CALUDE_bag_selling_price_l3472_347291

/-- The selling price of a discounted item -/
def selling_price (marked_price : ℝ) (discount_rate : ℝ) : ℝ :=
  marked_price * (1 - discount_rate)

/-- Theorem: The selling price of a bag marked at $80 with a 15% discount is $68 -/
theorem bag_selling_price :
  selling_price 80 0.15 = 68 := by
  sorry

end NUMINAMATH_CALUDE_bag_selling_price_l3472_347291


namespace NUMINAMATH_CALUDE_gcd_3270_594_l3472_347255

theorem gcd_3270_594 : Nat.gcd 3270 594 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_3270_594_l3472_347255


namespace NUMINAMATH_CALUDE_tetrahedron_volume_from_triangle_l3472_347267

/-- Given a triangle ABC with sides of length 11, 20, and 21 units,
    the volume of the tetrahedron formed by folding the triangle along
    the lines connecting the midpoints of its sides is 45 cubic units. -/
theorem tetrahedron_volume_from_triangle (a b c : ℝ) (h1 : a = 11) (h2 : b = 20) (h3 : c = 21) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let p := b / 2
  let q := c / 2
  let r := a / 2
  let s_mid := (p + q + r) / 2
  let area_mid := Real.sqrt (s_mid * (s_mid - p) * (s_mid - q) * (s_mid - r))
  let height := Real.sqrt ((q^2) - (area_mid^2 / area^2) * (a^2 / 4))
  (1/3) * area_mid * height = 45 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_from_triangle_l3472_347267


namespace NUMINAMATH_CALUDE_marcy_spears_count_l3472_347264

/-- The number of spears that can be made from one sapling -/
def spears_per_sapling : ℕ := 3

/-- The number of spears that can be made from one log -/
def spears_per_log : ℕ := 9

/-- The number of saplings Marcy has -/
def num_saplings : ℕ := 6

/-- The number of logs Marcy has -/
def num_logs : ℕ := 1

/-- Theorem: Marcy can make 27 spears from 6 saplings and 1 log -/
theorem marcy_spears_count :
  spears_per_sapling * num_saplings + spears_per_log * num_logs = 27 := by
  sorry

end NUMINAMATH_CALUDE_marcy_spears_count_l3472_347264


namespace NUMINAMATH_CALUDE_water_jars_count_l3472_347238

/-- Proves that 7 gallons of water stored in equal numbers of quart, half-gallon, and one-gallon jars results in 12 water-filled jars -/
theorem water_jars_count (total_water : ℚ) (jar_sizes : Fin 3 → ℚ) :
  total_water = 7 →
  jar_sizes 0 = 1/4 →
  jar_sizes 1 = 1/2 →
  jar_sizes 2 = 1 →
  ∃ (x : ℚ), x > 0 ∧ x * (jar_sizes 0 + jar_sizes 1 + jar_sizes 2) = total_water ∧
  (3 * x : ℚ) = 12 :=
by sorry

end NUMINAMATH_CALUDE_water_jars_count_l3472_347238


namespace NUMINAMATH_CALUDE_root_equation_a_value_l3472_347212

theorem root_equation_a_value (a b : ℚ) : 
  ((-2 : ℝ) - 5 * Real.sqrt 3)^3 + a * ((-2 : ℝ) - 5 * Real.sqrt 3)^2 + 
  b * ((-2 : ℝ) - 5 * Real.sqrt 3) - 48 = 0 → a = 4 := by
sorry

end NUMINAMATH_CALUDE_root_equation_a_value_l3472_347212


namespace NUMINAMATH_CALUDE_xyz_remainder_mod_9_l3472_347216

theorem xyz_remainder_mod_9 
  (x y z : ℕ) 
  (h_x : x < 9) (h_y : y < 9) (h_z : z < 9)
  (h1 : x + 3*y + 2*z ≡ 0 [ZMOD 9])
  (h2 : 3*x + 2*y + z ≡ 5 [ZMOD 9])
  (h3 : 2*x + y + 3*z ≡ 5 [ZMOD 9]) :
  x*y*z ≡ 0 [ZMOD 9] := by
sorry

end NUMINAMATH_CALUDE_xyz_remainder_mod_9_l3472_347216


namespace NUMINAMATH_CALUDE_cubic_function_symmetry_l3472_347245

/-- Given a cubic function f(x) = ax³ + bx + 5 where f(-9) = -7, prove that f(9) = 17 -/
theorem cubic_function_symmetry (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * x + 5)
  (h2 : f (-9) = -7) : 
  f 9 = 17 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_symmetry_l3472_347245


namespace NUMINAMATH_CALUDE_fraction_equality_l3472_347263

theorem fraction_equality (a b : ℝ) (h : a/b = 2) : a/(a-b) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3472_347263


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3472_347230

/-- Proves that 37/80 is equal to 0.4625 -/
theorem fraction_to_decimal : (37 : ℚ) / 80 = 0.4625 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3472_347230


namespace NUMINAMATH_CALUDE_policeman_speed_l3472_347229

/-- Given a chase scenario between a policeman and a thief, this theorem proves
    the speed of the policeman required to catch the thief. -/
theorem policeman_speed (initial_distance : ℝ) (thief_speed : ℝ) (thief_distance : ℝ) :
  initial_distance = 0.15 →
  thief_speed = 8 →
  thief_distance = 0.6 →
  ∃ (policeman_speed : ℝ),
    policeman_speed * (thief_distance / thief_speed) = initial_distance + thief_distance ∧
    policeman_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_policeman_speed_l3472_347229


namespace NUMINAMATH_CALUDE_square_side_length_l3472_347289

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2 * Real.sqrt 2) :
  ∃ (side : ℝ), side * Real.sqrt 2 = diagonal ∧ side = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3472_347289


namespace NUMINAMATH_CALUDE_power_of_three_and_seven_hundreds_digit_l3472_347288

theorem power_of_three_and_seven_hundreds_digit : 
  ∃ (a b : ℕ), 
    100 ≤ 3^a ∧ 3^a < 1000 ∧
    100 ≤ 7^b ∧ 7^b < 1000 ∧
    (3^a / 100 % 10 = 7) ∧ (7^b / 100 % 10 = 7) := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_and_seven_hundreds_digit_l3472_347288


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l3472_347222

theorem quadratic_solution_difference : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 5*x₁ + 15 = x₁ + 55) ∧
  (x₂^2 - 5*x₂ + 15 = x₂ + 55) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l3472_347222


namespace NUMINAMATH_CALUDE_total_laundry_loads_l3472_347274

/-- The number of families sharing the vacation rental -/
def num_families : ℕ := 7

/-- The number of days of the vacation -/
def num_days : ℕ := 12

/-- The number of adults in each family -/
def adults_per_family : ℕ := 2

/-- The number of children in each family -/
def children_per_family : ℕ := 4

/-- The number of towels used by each adult per day -/
def towels_per_adult : ℕ := 2

/-- The number of towels used by each child per day -/
def towels_per_child : ℕ := 1

/-- The washing machine capacity for the first half of the vacation -/
def machine_capacity_first_half : ℕ := 8

/-- The washing machine capacity for the second half of the vacation -/
def machine_capacity_second_half : ℕ := 6

/-- The number of days in each half of the vacation -/
def days_per_half : ℕ := 6

/-- Theorem stating that the total number of loads of laundry is 98 -/
theorem total_laundry_loads : 
  let towels_per_family := adults_per_family * towels_per_adult + children_per_family * towels_per_child
  let total_towels_per_day := num_families * towels_per_family
  let total_towels := total_towels_per_day * num_days
  let loads_first_half := (total_towels_per_day * days_per_half) / machine_capacity_first_half
  let loads_second_half := (total_towels_per_day * days_per_half) / machine_capacity_second_half
  loads_first_half + loads_second_half = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_laundry_loads_l3472_347274


namespace NUMINAMATH_CALUDE_base4_multiplication_l3472_347236

/-- Converts a base 4 number represented as a list of digits to its decimal equivalent -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a decimal number to its base 4 representation -/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base4_multiplication (a b : List Nat) :
  decimalToBase4 (base4ToDecimal a * base4ToDecimal b) = [3, 2, 1, 3, 3] ↔
  a = [3, 1, 2, 1] ∧ b = [1, 2] :=
sorry

end NUMINAMATH_CALUDE_base4_multiplication_l3472_347236


namespace NUMINAMATH_CALUDE_regression_line_equation_l3472_347286

theorem regression_line_equation (slope : ℝ) (center_x center_y : ℝ) :
  slope = 2.03 →
  center_x = 5 →
  center_y = 11 →
  ∀ x y : ℝ, y = slope * x + (center_y - slope * center_x) ↔ y = 2.03 * x + 0.85 :=
by sorry

end NUMINAMATH_CALUDE_regression_line_equation_l3472_347286


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l3472_347215

def f (x : ℝ) := 3 * x^3 - 9 * x + 5

theorem f_monotonicity_and_extrema :
  (∀ x y, x < y ∧ y < -1 → f x < f y) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧
  (∀ x, f x ≤ f (-1)) ∧
  (∀ x, f 1 ≤ f x) ∧
  (f (-1) = 11) ∧
  (f 1 = -1) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l3472_347215


namespace NUMINAMATH_CALUDE_fifty_roses_cost_l3472_347251

/-- The cost of a bouquet of roses, given the number of roses -/
def bouquetCost (roses : ℕ) : ℚ :=
  let baseCost := 24 * roses / 12
  if roses ≥ 45 then baseCost * (1 - 1/10) else baseCost

theorem fifty_roses_cost :
  bouquetCost 50 = 90 := by sorry

end NUMINAMATH_CALUDE_fifty_roses_cost_l3472_347251


namespace NUMINAMATH_CALUDE_lottery_distribution_l3472_347202

/-- The total amount received by 100 students, each getting one-thousandth of $155250 -/
def total_amount (lottery_win : ℚ) (num_students : ℕ) : ℚ :=
  (lottery_win / 1000) * num_students

theorem lottery_distribution :
  total_amount 155250 100 = 15525 := by
  sorry

end NUMINAMATH_CALUDE_lottery_distribution_l3472_347202


namespace NUMINAMATH_CALUDE_range_of_a_l3472_347272

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3472_347272


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l3472_347292

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a := by sorry

theorem sum_of_roots_specific_equation :
  let r₁ := (6 + Real.sqrt (36 - 32)) / 2
  let r₂ := (6 - Real.sqrt (36 - 32)) / 2
  r₁ + r₂ = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l3472_347292


namespace NUMINAMATH_CALUDE_superinvariant_characterization_l3472_347242

/-- A set S is superinvariant if for any stretching A of S, there exists a translation B
    such that the images of S under A and B agree. -/
def Superinvariant (S : Set ℝ) : Prop :=
  ∀ (x₀ a : ℝ) (h : a > 0),
    ∃ b : ℝ,
      (∀ x ∈ S, ∃ y ∈ S, x₀ + a * (x - x₀) = y + b) ∧
      (∀ t ∈ S, ∃ u ∈ S, t + b = x₀ + a * (u - x₀))

/-- The set of all possible superinvariant sets for a given Γ. -/
def SuperinvariantSets (Γ : ℝ) : Set (Set ℝ) :=
  {∅, {Γ}, Set.Iio Γ, Set.Iic Γ, Set.Ioi Γ, Set.Ici Γ, (Set.Iio Γ) ∪ (Set.Ioi Γ), Set.univ}

/-- Theorem stating that a set is superinvariant if and only if it belongs to
    SuperinvariantSets for some Γ. -/
theorem superinvariant_characterization (S : Set ℝ) :
  Superinvariant S ↔ ∃ Γ : ℝ, S ∈ SuperinvariantSets Γ := by
  sorry

end NUMINAMATH_CALUDE_superinvariant_characterization_l3472_347242


namespace NUMINAMATH_CALUDE_factorial_program_components_l3472_347240

/-- A structure representing a simple programming language --/
structure SimpleProgram where
  input : String
  loop_start : String
  loop_end : String

/-- Definition of a program that calculates factorial --/
def factorial_program (p : SimpleProgram) : Prop :=
  p.input = "INPUT" ∧ 
  p.loop_start = "WHILE" ∧ 
  p.loop_end = "WEND"

/-- Theorem stating that a program calculating factorial requires specific components --/
theorem factorial_program_components :
  ∃ (p : SimpleProgram), factorial_program p :=
sorry

end NUMINAMATH_CALUDE_factorial_program_components_l3472_347240


namespace NUMINAMATH_CALUDE_alice_rearrangement_time_l3472_347233

/-- The time in hours required to write all rearrangements of a name -/
def time_to_write_rearrangements (name_length : ℕ) (rearrangements_per_minute : ℕ) : ℚ :=
  (Nat.factorial name_length : ℚ) / (rearrangements_per_minute : ℚ) / 60

/-- Theorem: Given a name with 5 unique letters and the ability to write 12 rearrangements per minute,
    it takes 1/6 hours to write all possible rearrangements -/
theorem alice_rearrangement_time :
  time_to_write_rearrangements 5 12 = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_alice_rearrangement_time_l3472_347233


namespace NUMINAMATH_CALUDE_point_between_parallel_lines_l3472_347237

-- Define the two line equations
def line1 (x y : ℝ) : Prop := 6 * x - 8 * y + 1 = 0
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

-- Define what it means for a point to be between two lines
def between_lines (x y : ℝ) : Prop :=
  (line1 x y ∧ ¬line2 x y) ∨ (¬line1 x y ∧ line2 x y) ∨ (¬line1 x y ∧ ¬line2 x y)

-- Theorem statement
theorem point_between_parallel_lines :
  between_lines 5 b → b = 4 := by sorry

end NUMINAMATH_CALUDE_point_between_parallel_lines_l3472_347237


namespace NUMINAMATH_CALUDE_category_A_sample_size_l3472_347235

/-- Represents the number of students in each school category -/
structure SchoolCategories where
  categoryA : ℕ
  categoryB : ℕ
  categoryC : ℕ

/-- Calculates the number of students selected from a category in stratified sampling -/
def stratifiedSample (categories : SchoolCategories) (totalSample : ℕ) (categorySize : ℕ) : ℕ :=
  (categorySize * totalSample) / (categories.categoryA + categories.categoryB + categories.categoryC)

/-- Theorem: The number of students selected from Category A in the given scenario is 200 -/
theorem category_A_sample_size :
  let categories := SchoolCategories.mk 2000 3000 4000
  let totalSample := 900
  stratifiedSample categories totalSample categories.categoryA = 200 := by
  sorry

end NUMINAMATH_CALUDE_category_A_sample_size_l3472_347235


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3472_347248

theorem trigonometric_identity (α β γ : Real) 
  (h : Real.sin α + Real.sin γ = 2 * Real.sin β) : 
  Real.tan ((α + β) / 2) + Real.tan ((β + γ) / 2) = 2 * Real.tan ((γ + α) / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3472_347248


namespace NUMINAMATH_CALUDE_tyrone_gives_seven_point_five_verify_final_ratio_l3472_347220

/-- The number of marbles Tyrone gives to Eric to end up with three times as many marbles as Eric, given their initial marble counts. -/
def marbles_given (tyrone_initial : ℚ) (eric_initial : ℚ) : ℚ :=
  let x : ℚ := (tyrone_initial + eric_initial) / 4 - eric_initial
  x

/-- Theorem stating that given the initial conditions, Tyrone gives 7.5 marbles to Eric. -/
theorem tyrone_gives_seven_point_five :
  marbles_given 120 30 = 7.5 := by
  sorry

/-- Verification that after giving marbles, Tyrone has three times as many as Eric. -/
theorem verify_final_ratio 
  (tyrone_initial eric_initial : ℚ) 
  (h : tyrone_initial = 120 ∧ eric_initial = 30) :
  let x := marbles_given tyrone_initial eric_initial
  (tyrone_initial - x) = 3 * (eric_initial + x) := by
  sorry

end NUMINAMATH_CALUDE_tyrone_gives_seven_point_five_verify_final_ratio_l3472_347220


namespace NUMINAMATH_CALUDE_fifth_root_of_1024_l3472_347269

theorem fifth_root_of_1024 : (1024 : ℝ) ^ (1/5) = 4 := by sorry

end NUMINAMATH_CALUDE_fifth_root_of_1024_l3472_347269


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3472_347278

theorem sum_of_x_and_y (x y : ℝ) (hx : x + 2 = 10) (hy : y - 1 = 6) : x + y = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3472_347278


namespace NUMINAMATH_CALUDE_choir_arrangement_l3472_347273

theorem choir_arrangement (n : ℕ) : 
  (∃ k : ℕ, n = k^2 + 11) ∧ 
  (∃ c : ℕ, n = c * (c + 5)) →
  n ≤ 300 :=
by sorry

end NUMINAMATH_CALUDE_choir_arrangement_l3472_347273


namespace NUMINAMATH_CALUDE_f_positive_iff_m_range_f_root_in_zero_one_iff_m_range_l3472_347224

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (m-1)*x + 2*m

theorem f_positive_iff_m_range (m : ℝ) :
  (∀ x > 0, f m x > 0) ↔ -2*Real.sqrt 6 + 5 ≤ m ∧ m ≤ 2*Real.sqrt 6 + 5 := by
  sorry

theorem f_root_in_zero_one_iff_m_range (m : ℝ) :
  (∃ x ∈ Set.Ioo 0 1, f m x = 0) ↔ m ∈ Set.Ioo (-2) 0 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_iff_m_range_f_root_in_zero_one_iff_m_range_l3472_347224


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3472_347266

theorem coefficient_x_squared_in_expansion : 
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  (∀ x : ℝ, (x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) → 
  a₂ = 6 := by
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3472_347266


namespace NUMINAMATH_CALUDE_winnie_balloon_distribution_l3472_347201

/-- Calculates the number of balloons left after equal distribution --/
def balloons_left (red blue green purple friends : ℕ) : ℕ :=
  (red + blue + green + purple) % friends

/-- Proves that Winnie has 0 balloons left after distribution --/
theorem winnie_balloon_distribution :
  balloons_left 22 44 66 88 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_winnie_balloon_distribution_l3472_347201


namespace NUMINAMATH_CALUDE_same_remainder_problem_l3472_347258

theorem same_remainder_problem : ∃ (N : ℕ), N > 1 ∧
  N = 23 ∧
  ∀ (M : ℕ), M > N → ¬(1743 % M = 2019 % M ∧ 2019 % M = 3008 % M) :=
by sorry

end NUMINAMATH_CALUDE_same_remainder_problem_l3472_347258


namespace NUMINAMATH_CALUDE_road_repaving_today_distance_l3472_347221

/-- Represents the repaving progress of a road construction project -/
structure RoadRepaving where
  totalRepaved : ℕ
  repavedBefore : ℕ

/-- Calculates the distance repaved today given the total repaved and repaved before -/
def distanceRepavedToday (r : RoadRepaving) : ℕ :=
  r.totalRepaved - r.repavedBefore

/-- Theorem: For the given road repaving project, the distance repaved today is 805 inches -/
theorem road_repaving_today_distance 
  (r : RoadRepaving) 
  (h1 : r.totalRepaved = 4938) 
  (h2 : r.repavedBefore = 4133) : 
  distanceRepavedToday r = 805 := by
  sorry

#eval distanceRepavedToday { totalRepaved := 4938, repavedBefore := 4133 }

end NUMINAMATH_CALUDE_road_repaving_today_distance_l3472_347221


namespace NUMINAMATH_CALUDE_total_production_proof_l3472_347217

def day_shift_production : ℕ := 4400
def day_shift_multiplier : ℕ := 4

theorem total_production_proof :
  let second_shift_production := day_shift_production / day_shift_multiplier
  day_shift_production + second_shift_production = 5500 := by
  sorry

end NUMINAMATH_CALUDE_total_production_proof_l3472_347217


namespace NUMINAMATH_CALUDE_water_bottles_divisible_by_kits_l3472_347261

/-- Represents the number of emergency-preparedness kits Veronica can make. -/
def num_kits : ℕ := 4

/-- Represents the total number of cans of food Veronica has. -/
def total_cans : ℕ := 12

/-- Represents the number of bottles of water Veronica has. -/
def water_bottles : ℕ := sorry

/-- Theorem stating that the number of water bottles is divisible by the number of kits. -/
theorem water_bottles_divisible_by_kits : 
  water_bottles % num_kits = 0 ∧ 
  total_cans % num_kits = 0 :=
sorry

end NUMINAMATH_CALUDE_water_bottles_divisible_by_kits_l3472_347261


namespace NUMINAMATH_CALUDE_product_xyz_l3472_347206

theorem product_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + 1/y = 5)
  (eq2 : y + 1/z = 2)
  (eq3 : z + 1/x = 8/3) :
  x * y * z = (11 + Real.sqrt 117) / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_l3472_347206


namespace NUMINAMATH_CALUDE_seventh_term_is_25_over_3_l3472_347249

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term
  a : ℚ
  -- Common difference
  d : ℚ
  -- Sum of first five terms is 15
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 15
  -- Sixth term is 7
  sixth_term : a + 5*d = 7

/-- The seventh term of the arithmetic sequence is 25/3 -/
theorem seventh_term_is_25_over_3 (seq : ArithmeticSequence) :
  seq.a + 6*seq.d = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_25_over_3_l3472_347249


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3472_347205

theorem geometric_sequence_ratio (a₁ a₂ a₃ : ℝ) (h1 : a₁ = 9) (h2 : a₂ = -18) (h3 : a₃ = 36) :
  ∃ r : ℝ, r = a₂ / a₁ ∧ r = a₃ / a₂ ∧ r = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3472_347205


namespace NUMINAMATH_CALUDE_sum_positive_implies_at_least_one_positive_l3472_347209

theorem sum_positive_implies_at_least_one_positive (a b : ℝ) :
  a + b > 0 → a > 0 ∨ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_implies_at_least_one_positive_l3472_347209


namespace NUMINAMATH_CALUDE_calculate_swimming_speed_triathlete_swimming_speed_l3472_347250

/-- Calculates the swimming speed given the total distance, running speed, and average speed -/
theorem calculate_swimming_speed 
  (total_distance : ℝ) 
  (running_distance : ℝ) 
  (running_speed : ℝ) 
  (average_speed : ℝ) : ℝ :=
  let swimming_distance := total_distance - running_distance
  let total_time := total_distance / average_speed
  let running_time := running_distance / running_speed
  let swimming_time := total_time - running_time
  let swimming_speed := swimming_distance / swimming_time
  swimming_speed

/-- Proves that the swimming speed is 6 miles per hour given the problem conditions -/
theorem triathlete_swimming_speed :
  calculate_swimming_speed 8 4 10 7.5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_calculate_swimming_speed_triathlete_swimming_speed_l3472_347250


namespace NUMINAMATH_CALUDE_transport_cost_effectiveness_l3472_347253

/-- Represents the transportation cost functions and conditions for fruit shipping --/
structure FruitTransport where
  x : ℝ  -- distance in kilometers
  truck_cost : ℝ → ℝ  -- trucking company cost function
  train_cost : ℝ → ℝ  -- train freight station cost function

/-- Theorem stating the cost-effectiveness of different transportation methods --/
theorem transport_cost_effectiveness (ft : FruitTransport) 
  (h_truck : ft.truck_cost = λ x => 94 * x + 4000)
  (h_train : ft.train_cost = λ x => 81 * x + 6600) :
  (∀ x, x > 0 ∧ x < 200 → ft.truck_cost x < ft.train_cost x) ∧
  (∀ x, x > 200 → ft.train_cost x < ft.truck_cost x) := by
  sorry

#check transport_cost_effectiveness

end NUMINAMATH_CALUDE_transport_cost_effectiveness_l3472_347253


namespace NUMINAMATH_CALUDE_largest_divided_by_smallest_l3472_347207

def numbers : List ℝ := [10, 11, 12]

theorem largest_divided_by_smallest : 
  (List.maximum numbers).get! / (List.minimum numbers).get! = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_largest_divided_by_smallest_l3472_347207


namespace NUMINAMATH_CALUDE_dime_position_l3472_347254

/-- Represents the two possible coin values -/
inductive CoinValue : Type
  | nickel : CoinValue
  | dime : CoinValue

/-- Represents the two possible pocket locations -/
inductive Pocket : Type
  | left : Pocket
  | right : Pocket

/-- Returns the value of a coin in cents -/
def coinValue (c : CoinValue) : Nat :=
  match c with
  | CoinValue.nickel => 5
  | CoinValue.dime => 10

/-- Represents the arrangement of coins in pockets -/
structure CoinArrangement :=
  (leftCoin : CoinValue)
  (rightCoin : CoinValue)

/-- Calculates the sum based on the given formula -/
def calculateSum (arr : CoinArrangement) : Nat :=
  3 * (coinValue arr.rightCoin) + 2 * (coinValue arr.leftCoin)

/-- The main theorem to prove -/
theorem dime_position (arr : CoinArrangement) :
  Even (calculateSum arr) ↔ arr.rightCoin = CoinValue.dime :=
sorry

end NUMINAMATH_CALUDE_dime_position_l3472_347254


namespace NUMINAMATH_CALUDE_box_tape_theorem_l3472_347279

theorem box_tape_theorem (L S : ℝ) (h1 : L > 0) (h2 : S > 0) :
  5 * (L + 2 * S) + 240 = 540 → S = (60 - L) / 2 := by
  sorry

end NUMINAMATH_CALUDE_box_tape_theorem_l3472_347279


namespace NUMINAMATH_CALUDE_matrix_determinant_l3472_347259

theorem matrix_determinant : 
  let A : Matrix (Fin 3) (Fin 3) ℤ := !![1, -3, 3; 0, 5, -1; 4, -2, 2]
  Matrix.det A = -40 := by
sorry

end NUMINAMATH_CALUDE_matrix_determinant_l3472_347259


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3472_347294

theorem right_triangle_hypotenuse (longer_leg shorter_leg hypotenuse : ℝ) : 
  shorter_leg = longer_leg - 3 →
  (1 / 2) * longer_leg * shorter_leg = 120 →
  longer_leg > 0 →
  shorter_leg > 0 →
  hypotenuse^2 = longer_leg^2 + shorter_leg^2 →
  hypotenuse = Real.sqrt 425 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3472_347294


namespace NUMINAMATH_CALUDE_intersection_y_intercept_sum_l3472_347210

/-- Given two lines that intersect at (3,6), prove that the sum of their y-intercepts is 6 -/
theorem intersection_y_intercept_sum (a b : ℝ) : 
  (∀ x y : ℝ, x = (1/3)*y + a ∧ y = (1/3)*x + b → (x = 3 ∧ y = 6)) → 
  a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_y_intercept_sum_l3472_347210


namespace NUMINAMATH_CALUDE_savings_after_expense_increase_l3472_347225

def monthly_savings (salary : ℝ) (initial_savings_rate : ℝ) (expense_increase_rate : ℝ) : ℝ :=
  let initial_savings := salary * initial_savings_rate
  let initial_expenses := salary - initial_savings
  let new_expenses := initial_expenses * (1 + expense_increase_rate)
  salary - new_expenses

theorem savings_after_expense_increase :
  monthly_savings 1000 0.25 0.1 = 175 := by sorry

end NUMINAMATH_CALUDE_savings_after_expense_increase_l3472_347225


namespace NUMINAMATH_CALUDE_vector_sum_scalar_multiple_l3472_347297

/-- Given two planar vectors a and b, prove that 3a + b equals the expected result. -/
theorem vector_sum_scalar_multiple (a b : ℝ × ℝ) : 
  a = (-1, 2) → b = (1, 0) → (3 • a) + b = (-2, 6) := by sorry

end NUMINAMATH_CALUDE_vector_sum_scalar_multiple_l3472_347297


namespace NUMINAMATH_CALUDE_power_of_power_l3472_347293

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3472_347293


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_three_l3472_347290

theorem fraction_zero_implies_x_negative_three (x : ℝ) :
  (x^2 - 9) / (x - 3) = 0 ∧ x ≠ 3 → x = -3 := by sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_three_l3472_347290


namespace NUMINAMATH_CALUDE_toy_store_inventory_l3472_347204

structure Toy where
  name : String
  week1_sales : ℕ
  week2_sales : ℕ
  remaining : ℕ

def initial_stock (t : Toy) : ℕ :=
  t.remaining + t.week1_sales + t.week2_sales

theorem toy_store_inventory (action_figures board_games puzzles stuffed_animals : Toy) 
  (h1 : action_figures.name = "Action Figures" ∧ action_figures.week1_sales = 38 ∧ action_figures.week2_sales = 26 ∧ action_figures.remaining = 19)
  (h2 : board_games.name = "Board Games" ∧ board_games.week1_sales = 27 ∧ board_games.week2_sales = 15 ∧ board_games.remaining = 8)
  (h3 : puzzles.name = "Puzzles" ∧ puzzles.week1_sales = 43 ∧ puzzles.week2_sales = 39 ∧ puzzles.remaining = 12)
  (h4 : stuffed_animals.name = "Stuffed Animals" ∧ stuffed_animals.week1_sales = 20 ∧ stuffed_animals.week2_sales = 18 ∧ stuffed_animals.remaining = 30) :
  initial_stock action_figures = 83 ∧ 
  initial_stock board_games = 50 ∧ 
  initial_stock puzzles = 94 ∧ 
  initial_stock stuffed_animals = 68 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_inventory_l3472_347204


namespace NUMINAMATH_CALUDE_soap_usage_ratio_l3472_347268

/-- Represents the survey results of household soap usage --/
structure SoapSurvey where
  total : ℕ
  neither : ℕ
  onlyA : ℕ
  both : ℕ
  neitherLtTotal : neither < total
  onlyALtTotal : onlyA < total
  bothLtTotal : both < total

/-- Calculates the number of households using only brand B soap --/
def onlyB (s : SoapSurvey) : ℕ :=
  s.total - s.neither - s.onlyA - s.both

/-- Theorem stating the ratio of households using only brand B to those using both brands --/
theorem soap_usage_ratio (s : SoapSurvey)
  (h1 : s.total = 260)
  (h2 : s.neither = 80)
  (h3 : s.onlyA = 60)
  (h4 : s.both = 30) :
  (onlyB s) / s.both = 3 := by
  sorry

end NUMINAMATH_CALUDE_soap_usage_ratio_l3472_347268


namespace NUMINAMATH_CALUDE_sum_not_divisible_by_ten_l3472_347246

theorem sum_not_divisible_by_ten (n : ℕ) :
  ¬(10 ∣ (1981^n + 1982^n + 1983^n + 1984^n)) ↔ 4 ∣ n :=
sorry

end NUMINAMATH_CALUDE_sum_not_divisible_by_ten_l3472_347246


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3472_347214

theorem arithmetic_expression_equality : 50 + 5 * 12 / (180 / 3) = 51 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3472_347214


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3472_347262

theorem partial_fraction_decomposition :
  ∀ (x : ℝ), x ≠ 4 → x ≠ 5 →
  (7 * x - 4) / (x^2 - 9*x + 20) = (-20) / (x - 4) + 31 / (x - 5) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3472_347262


namespace NUMINAMATH_CALUDE_empty_subset_of_A_l3472_347287

def A : Set ℤ := {x | 0 < x ∧ x < 3}

theorem empty_subset_of_A : ∅ ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_empty_subset_of_A_l3472_347287


namespace NUMINAMATH_CALUDE_positive_sum_l3472_347256

theorem positive_sum (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  0 < b + c := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_l3472_347256


namespace NUMINAMATH_CALUDE_smallest_value_of_reciprocal_sum_l3472_347295

theorem smallest_value_of_reciprocal_sum (u q a₁ a₂ : ℝ) : 
  (a₁^2 - u*a₁ + q = 0) →
  (a₂^2 - u*a₂ + q = 0) →
  (a₁ + a₂ = a₁^2 + a₂^2) →
  (a₁ + a₂ = a₁^3 + a₂^3) →
  (a₁ + a₂ = a₁^4 + a₂^4) →
  (a₁ ≠ 0 ∧ a₂ ≠ 0) →
  2 ≤ (1 / a₁^10 + 1 / a₂^10) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_reciprocal_sum_l3472_347295


namespace NUMINAMATH_CALUDE_students_playing_neither_sport_l3472_347231

theorem students_playing_neither_sport (total : ℕ) (hockey : ℕ) (basketball : ℕ) (both : ℕ) 
  (h_total : total = 25)
  (h_hockey : hockey = 15)
  (h_basketball : basketball = 16)
  (h_both : both = 10) :
  total - (hockey + basketball - both) = 4 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_sport_l3472_347231


namespace NUMINAMATH_CALUDE_max_distinct_pairs_l3472_347281

theorem max_distinct_pairs (k : ℕ) 
  (a b : Fin k → ℕ)
  (h_range : ∀ i : Fin k, 1 ≤ a i ∧ a i < b i ∧ b i ≤ 150)
  (h_distinct : ∀ i j : Fin k, i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j)
  (h_sum_distinct : ∀ i j : Fin k, i ≠ j → a i + b i ≠ a j + b j)
  (h_sum_bound : ∀ i : Fin k, a i + b i ≤ 150) :
  k ≤ 59 :=
sorry

end NUMINAMATH_CALUDE_max_distinct_pairs_l3472_347281


namespace NUMINAMATH_CALUDE_train_overtake_l3472_347280

/-- Proves that Train B overtakes Train A at 285 miles from the station -/
theorem train_overtake (speed_A speed_B : ℝ) (time_diff : ℝ) : 
  speed_A = 30 →
  speed_B = 38 →
  time_diff = 2 →
  speed_B > speed_A →
  (speed_A * time_diff + speed_A * ((speed_B * time_diff) / (speed_B - speed_A))) = 285 := by
  sorry

#check train_overtake

end NUMINAMATH_CALUDE_train_overtake_l3472_347280


namespace NUMINAMATH_CALUDE_three_fifths_of_five_times_nine_l3472_347298

theorem three_fifths_of_five_times_nine : (3 : ℚ) / 5 * (5 * 9) = 27 := by sorry

end NUMINAMATH_CALUDE_three_fifths_of_five_times_nine_l3472_347298


namespace NUMINAMATH_CALUDE_square_key_presses_l3472_347257

theorem square_key_presses (start : ℝ) (target : ℝ) : ∃ (n : ℕ), n = 4 ∧ start^(2^n) > target ∧ ∀ m < n, start^(2^m) ≤ target :=
by
  -- We define the starting number and the target
  let start := 1.5
  let target := 300
  -- The proof goes here
  sorry

#check square_key_presses

end NUMINAMATH_CALUDE_square_key_presses_l3472_347257


namespace NUMINAMATH_CALUDE_shop_ratio_l3472_347227

/-- Given a shop with pencils, pens, and exercise books in the ratio 14 : 4 : 3,
    and 140 pencils, the ratio of exercise books to pens is 3 : 4. -/
theorem shop_ratio (pencils pens books : ℕ) : 
  pencils = 140 →
  pencils / 14 = pens / 4 →
  pencils / 14 = books / 3 →
  books / pens = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_shop_ratio_l3472_347227


namespace NUMINAMATH_CALUDE_hannah_lost_eight_pieces_l3472_347282

-- Define the initial state of the chess game
def initial_pieces : ℕ := 32
def initial_pieces_per_player : ℕ := 16

-- Define the given conditions
def scarlett_lost : ℕ := 6
def total_pieces_left : ℕ := 18

-- Define Hannah's lost pieces
def hannah_lost : ℕ := initial_pieces_per_player - (total_pieces_left - (initial_pieces_per_player - scarlett_lost))

-- Theorem to prove
theorem hannah_lost_eight_pieces : hannah_lost = 8 := by
  sorry

end NUMINAMATH_CALUDE_hannah_lost_eight_pieces_l3472_347282


namespace NUMINAMATH_CALUDE_marias_stationery_cost_l3472_347218

/-- The total cost of Maria's stationery purchase after applying a coupon and including sales tax. -/
theorem marias_stationery_cost :
  let notebook_a_count : ℕ := 4
  let notebook_b_count : ℕ := 3
  let notebook_c_count : ℕ := 3
  let pen_count : ℕ := 5
  let highlighter_pack_count : ℕ := 1
  let notebook_a_price : ℚ := 3.5
  let notebook_b_price : ℚ := 2.25
  let notebook_c_price : ℚ := 1.75
  let pen_price : ℚ := 2
  let highlighter_pack_price : ℚ := 4.5
  let coupon_discount : ℚ := 0.1
  let sales_tax_rate : ℚ := 0.05

  let total_before_discount : ℚ := 
    notebook_a_count * notebook_a_price +
    notebook_b_count * notebook_b_price +
    notebook_c_count * notebook_c_price +
    pen_count * pen_price +
    highlighter_pack_count * highlighter_pack_price

  let discount_amount : ℚ := total_before_discount * coupon_discount
  let total_after_discount : ℚ := total_before_discount - discount_amount
  let sales_tax : ℚ := total_after_discount * sales_tax_rate
  let final_cost : ℚ := total_after_discount + sales_tax

  final_cost = 38.27 := by sorry

end NUMINAMATH_CALUDE_marias_stationery_cost_l3472_347218
