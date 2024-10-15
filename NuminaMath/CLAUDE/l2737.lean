import Mathlib

namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2737_273747

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | x^2 - x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2737_273747


namespace NUMINAMATH_CALUDE_smallest_block_with_231_hidden_cubes_l2737_273763

/-- Represents the dimensions of a rectangular block. -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in a block given its dimensions. -/
def totalCubes (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of hidden cubes in a block given its dimensions. -/
def hiddenCubes (d : BlockDimensions) : ℕ :=
  (d.length - 1) * (d.width - 1) * (d.height - 1)

/-- Theorem stating that the smallest possible number of cubes in a block
    with 231 hidden cubes is 384. -/
theorem smallest_block_with_231_hidden_cubes :
  ∃ (d : BlockDimensions),
    hiddenCubes d = 231 ∧
    totalCubes d = 384 ∧
    ∀ (d' : BlockDimensions),
      hiddenCubes d' = 231 → totalCubes d' ≥ 384 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_block_with_231_hidden_cubes_l2737_273763


namespace NUMINAMATH_CALUDE_count_distinct_sums_l2737_273716

def S : Finset ℕ := {2, 5, 8, 11, 14, 17, 20, 23}

def sumOfFourDistinct (s : Finset ℕ) : Finset ℕ :=
  (s.powerset.filter (fun t => t.card = 4)).image (fun t => t.sum id)

theorem count_distinct_sums : (sumOfFourDistinct S).card = 49 := by
  sorry

end NUMINAMATH_CALUDE_count_distinct_sums_l2737_273716


namespace NUMINAMATH_CALUDE_oil_depth_in_cylindrical_tank_l2737_273773

/-- Represents a horizontal cylindrical tank --/
structure HorizontalCylindricalTank where
  length : Real
  diameter : Real

/-- Represents the oil in the tank --/
structure Oil where
  depth : Real
  surface_area : Real

theorem oil_depth_in_cylindrical_tank
  (tank : HorizontalCylindricalTank)
  (oil : Oil)
  (h_length : tank.length = 12)
  (h_diameter : tank.diameter = 4)
  (h_surface_area : oil.surface_area = 24) :
  oil.depth = 2 - Real.sqrt 3 ∨ oil.depth = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_oil_depth_in_cylindrical_tank_l2737_273773


namespace NUMINAMATH_CALUDE_work_distance_calculation_l2737_273795

/-- The distance to Tim's work in miles -/
def work_distance : ℝ := 20

/-- The number of workdays Tim rides his bike -/
def workdays : ℕ := 5

/-- The distance of Tim's weekend bike ride in miles -/
def weekend_ride : ℝ := 200

/-- Tim's biking speed in miles per hour -/
def biking_speed : ℝ := 25

/-- The total time Tim spends biking in a week in hours -/
def total_biking_time : ℝ := 16

theorem work_distance_calculation : 
  2 * workdays * work_distance + weekend_ride = biking_speed * total_biking_time := by
  sorry

end NUMINAMATH_CALUDE_work_distance_calculation_l2737_273795


namespace NUMINAMATH_CALUDE_extra_birds_calculation_l2737_273700

structure BirdPopulation where
  totalBirds : Nat
  sparrows : Nat
  robins : Nat
  bluebirds : Nat
  totalNests : Nat
  sparrowNests : Nat
  robinNests : Nat
  bluebirdNests : Nat

def extraBirds (bp : BirdPopulation) : Nat :=
  (bp.sparrows - bp.sparrowNests) + (bp.robins - bp.robinNests) + (bp.bluebirds - bp.bluebirdNests)

theorem extra_birds_calculation (bp : BirdPopulation) 
  (h1 : bp.totalBirds = bp.sparrows + bp.robins + bp.bluebirds)
  (h2 : bp.totalNests = bp.sparrowNests + bp.robinNests + bp.bluebirdNests)
  (h3 : bp.totalBirds = 18) (h4 : bp.sparrows = 10) (h5 : bp.robins = 5) (h6 : bp.bluebirds = 3)
  (h7 : bp.totalNests = 8) (h8 : bp.sparrowNests = 4) (h9 : bp.robinNests = 2) (h10 : bp.bluebirdNests = 2) :
  extraBirds bp = 10 := by
  sorry

end NUMINAMATH_CALUDE_extra_birds_calculation_l2737_273700


namespace NUMINAMATH_CALUDE_sakshi_work_time_l2737_273779

/-- Proves that given Tanya is 25% more efficient than Sakshi and takes 8 days to complete a piece of work, Sakshi will take 10 days to complete the same work. -/
theorem sakshi_work_time (sakshi_time tanya_time : ℝ) 
  (h1 : tanya_time = 8)
  (h2 : sakshi_time * 1 = tanya_time * 1.25) : 
  sakshi_time = 10 := by
sorry

end NUMINAMATH_CALUDE_sakshi_work_time_l2737_273779


namespace NUMINAMATH_CALUDE_fourDigitNumbersTheorem_l2737_273758

/-- Represents the multiset of numbers on the cards -/
def cardNumbers : Multiset ℕ := {1, 1, 1, 2, 2, 3, 4}

/-- Number of cards drawn -/
def cardsDrawn : ℕ := 4

/-- Function to calculate the number of different four-digit numbers -/
def fourDigitNumbersCount (cards : Multiset ℕ) (drawn : ℕ) : ℕ := sorry

/-- Theorem stating that the number of different four-digit numbers is 114 -/
theorem fourDigitNumbersTheorem : fourDigitNumbersCount cardNumbers cardsDrawn = 114 := by
  sorry

end NUMINAMATH_CALUDE_fourDigitNumbersTheorem_l2737_273758


namespace NUMINAMATH_CALUDE_ice_cream_cost_l2737_273792

/-- The cost of ice cream problem -/
theorem ice_cream_cost (ice_cream_quantity : ℕ) (yogurt_quantity : ℕ) (yogurt_cost : ℕ) (price_difference : ℕ) :
  ice_cream_quantity = 20 →
  yogurt_quantity = 2 →
  yogurt_cost = 1 →
  price_difference = 118 →
  ∃ (ice_cream_cost : ℕ), 
    ice_cream_cost * ice_cream_quantity = yogurt_cost * yogurt_quantity + price_difference ∧
    ice_cream_cost = 6 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l2737_273792


namespace NUMINAMATH_CALUDE_zoo_ticket_price_l2737_273702

theorem zoo_ticket_price (total_people : ℕ) (num_children : ℕ) (child_price : ℕ) (total_bill : ℕ) :
  total_people = 201 →
  num_children = 161 →
  child_price = 4 →
  total_bill = 964 →
  (total_people - num_children) * 8 + num_children * child_price = total_bill :=
by sorry

end NUMINAMATH_CALUDE_zoo_ticket_price_l2737_273702


namespace NUMINAMATH_CALUDE_gcd_special_powers_l2737_273778

theorem gcd_special_powers : Nat.gcd (2^1001 - 1) (2^1012 - 1) = 2^11 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_special_powers_l2737_273778


namespace NUMINAMATH_CALUDE_simplify_roots_l2737_273712

theorem simplify_roots : (256 : ℝ)^(1/4) * (625 : ℝ)^(1/2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_simplify_roots_l2737_273712


namespace NUMINAMATH_CALUDE_smallest_n_value_l2737_273704

theorem smallest_n_value (N : ℕ) (h1 : N > 70) (h2 : (21 * N) % 70 = 0) : 
  (∀ m : ℕ, m > 70 ∧ (21 * m) % 70 = 0 → m ≥ N) → N = 80 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_value_l2737_273704


namespace NUMINAMATH_CALUDE_range_of_m_l2737_273726

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 3 - |x| ≥ 0}

-- Define the set C parameterized by m
def C (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - 2*m - 1) < 0}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (C m ⊆ B) ↔ m ∈ Set.Icc (-2) 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2737_273726


namespace NUMINAMATH_CALUDE_barry_dime_value_l2737_273742

def dime_value : ℕ := 10

theorem barry_dime_value (dan_dimes : ℕ) (barry_dimes : ℕ) : 
  dan_dimes = 52 ∧ 
  dan_dimes = barry_dimes / 2 + 2 →
  barry_dimes * dime_value = 1000 := by
sorry

end NUMINAMATH_CALUDE_barry_dime_value_l2737_273742


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2737_273711

theorem polynomial_expansion (t : ℝ) :
  (3 * t^3 + 2 * t^2 - 4 * t + 1) * (-2 * t^2 + 3 * t - 5) =
  -6 * t^5 + 5 * t^4 - t^3 - 24 * t^2 + 23 * t - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2737_273711


namespace NUMINAMATH_CALUDE_print_shop_price_differences_l2737_273761

/-- Represents a print shop with its pricing structure -/
structure PrintShop where
  base_price : ℝ
  discount_threshold : ℕ
  discount_rate : ℝ
  flat_discount : ℝ

/-- Calculates the price for a given number of copies at a print shop -/
def calculate_price (shop : PrintShop) (copies : ℕ) : ℝ :=
  let base_total := shop.base_price * copies
  if copies ≥ shop.discount_threshold then
    base_total * (1 - shop.discount_rate) - shop.flat_discount
  else
    base_total

/-- Theorem stating the price differences between print shops for 60 copies -/
theorem print_shop_price_differences
  (shop_x shop_y shop_z shop_w : PrintShop)
  (hx : shop_x = { base_price := 1.25, discount_threshold := 0, discount_rate := 0, flat_discount := 0 })
  (hy : shop_y = { base_price := 2.75, discount_threshold := 0, discount_rate := 0, flat_discount := 0 })
  (hz : shop_z = { base_price := 3.00, discount_threshold := 50, discount_rate := 0.1, flat_discount := 0 })
  (hw : shop_w = { base_price := 2.00, discount_threshold := 60, discount_rate := 0, flat_discount := 5 }) :
  let copies := 60
  let min_price := min (min (min (calculate_price shop_x copies) (calculate_price shop_y copies))
                            (calculate_price shop_z copies))
                       (calculate_price shop_w copies)
  (calculate_price shop_y copies - min_price = 90) ∧
  (calculate_price shop_z copies - min_price = 87) ∧
  (calculate_price shop_w copies - min_price = 40) := by
  sorry

end NUMINAMATH_CALUDE_print_shop_price_differences_l2737_273761


namespace NUMINAMATH_CALUDE_circle_area_ratio_when_diameter_tripled_l2737_273783

theorem circle_area_ratio_when_diameter_tripled :
  ∀ (r : ℝ), r > 0 →
  (π * r^2) / (π * (3*r)^2) = 1/9 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_when_diameter_tripled_l2737_273783


namespace NUMINAMATH_CALUDE_expressway_lengths_l2737_273740

theorem expressway_lengths (total : ℕ) (difference : ℕ) 
  (h1 : total = 519)
  (h2 : difference = 45) : 
  ∃ (new expanded : ℕ), 
    new + expanded = total ∧ 
    new = 2 * expanded - difference ∧
    new = 331 ∧ 
    expanded = 188 := by
  sorry

end NUMINAMATH_CALUDE_expressway_lengths_l2737_273740


namespace NUMINAMATH_CALUDE_katie_math_problems_l2737_273728

/-- Given that Katie had 9 math problems for homework and 4 problems left to do after the bus ride,
    prove that she finished 5 problems on the bus ride home. -/
theorem katie_math_problems (total : ℕ) (remaining : ℕ) (h1 : total = 9) (h2 : remaining = 4) :
  total - remaining = 5 := by
  sorry

end NUMINAMATH_CALUDE_katie_math_problems_l2737_273728


namespace NUMINAMATH_CALUDE_chip_drawing_probability_l2737_273789

/-- The number of tan chips in the bag -/
def tan_chips : ℕ := 4

/-- The number of pink chips in the bag -/
def pink_chips : ℕ := 3

/-- The number of violet chips in the bag -/
def violet_chips : ℕ := 5

/-- The number of green chips in the bag -/
def green_chips : ℕ := 2

/-- The total number of chips in the bag -/
def total_chips : ℕ := tan_chips + pink_chips + violet_chips + green_chips

/-- The probability of drawing the chips as specified -/
def probability : ℚ := 1 / 42000

theorem chip_drawing_probability :
  (tan_chips.factorial * pink_chips.factorial * violet_chips.factorial * (3 + green_chips).factorial) / total_chips.factorial = probability := by
  sorry

end NUMINAMATH_CALUDE_chip_drawing_probability_l2737_273789


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_product_l2737_273762

-- Define the two polynomials
def p (x : ℝ) : ℝ := 5*x^3 - 3*x^2 + 9*x - 2
def q (x : ℝ) : ℝ := 3*x^2 - 4*x + 2

-- Define the product of the polynomials
def product (x : ℝ) : ℝ := p x * q x

-- Theorem: The coefficient of x^2 in the product is -48
theorem coefficient_x_squared_in_product : 
  ∃ (a b c d : ℝ), product = fun x ↦ a*x^3 + (-48)*x^2 + b*x + c + d*x^4 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_product_l2737_273762


namespace NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l2737_273770

/-- An arithmetic sequence with specified terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  a2_eq_3 : a 2 = 3
  a4_eq_7 : a 4 = 7

/-- The theorem stating that k = 8 for the given conditions -/
theorem arithmetic_sequence_k_value (seq : ArithmeticSequence) :
  ∃ k : ℕ, seq.a k = 15 ∧ k = 8 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l2737_273770


namespace NUMINAMATH_CALUDE_k_lower_bound_l2737_273706

/-- Piecewise function f(x) -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else k * x

/-- Theorem stating the lower bound of k -/
theorem k_lower_bound (k : ℝ) :
  (∃ x₀ : ℝ, f k (-x₀) = f k x₀) → k ≥ -Real.exp (-1) :=
by sorry

end NUMINAMATH_CALUDE_k_lower_bound_l2737_273706


namespace NUMINAMATH_CALUDE_trigonometric_expression_simplification_l2737_273780

open Real

theorem trigonometric_expression_simplification (α : ℝ) :
  (sin (2 * π - α) * cos (π + α) * cos (π / 2 + α) * cos (11 * π / 2 - α)) /
  (cos (π - α) * sin (3 * π - α) * sin (-π - α) * sin (9 * π / 2 + α)) = -tan α :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_simplification_l2737_273780


namespace NUMINAMATH_CALUDE_least_b_value_l2737_273784

theorem least_b_value (a b : ℕ+) : 
  (∃ p : ℕ+, p.val.Prime ∧ p > 2 ∧ a = p^2) → -- a is the square of the next smallest prime after 2
  (Finset.card (Nat.divisors a) = 3) →        -- a has 3 factors
  (Finset.card (Nat.divisors b) = a) →        -- b has a factors
  (a ∣ b) →                                   -- b is divisible by a
  b ≥ 36 :=                                   -- the least possible value of b is 36
by sorry

end NUMINAMATH_CALUDE_least_b_value_l2737_273784


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2737_273703

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k > 60 ∧ ¬(k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ m : ℤ, m ≤ 60 → (m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2737_273703


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2737_273785

theorem arithmetic_sequence_problem (n : ℕ) (sum : ℝ) (min_term max_term : ℝ) :
  n = 300 ∧ 
  sum = 22500 ∧ 
  min_term = 5 ∧ 
  max_term = 150 →
  let avg : ℝ := sum / n
  let d : ℝ := min ((avg - min_term) / (n - 1)) ((max_term - avg) / (n - 1))
  let L : ℝ := avg - (75 - 1) * d
  let G : ℝ := avg + (75 - 1) * d
  G - L = 31500 / 299 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2737_273785


namespace NUMINAMATH_CALUDE_circle_area_increase_l2737_273769

theorem circle_area_increase (c : ℝ) (r_increase : ℝ) (h1 : c = 16 * Real.pi) (h2 : r_increase = 2) :
  let r := c / (2 * Real.pi)
  let new_r := r + r_increase
  let area_increase := Real.pi * new_r^2 - Real.pi * r^2
  area_increase = 36 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_area_increase_l2737_273769


namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l2737_273732

theorem rectangle_side_ratio (a b : ℝ) (h : b = 2 * a) : (b / a) ^ 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_ratio_l2737_273732


namespace NUMINAMATH_CALUDE_equation_negative_roots_a_range_l2737_273765

theorem equation_negative_roots_a_range :
  ∀ a : ℝ,
  (∀ x : ℝ, x < 0 → 4^x - 2^(x-1) + a = 0) →
  (-1/2 < a ∧ a ≤ 1/16) :=
by sorry

end NUMINAMATH_CALUDE_equation_negative_roots_a_range_l2737_273765


namespace NUMINAMATH_CALUDE_product_inequality_l2737_273724

theorem product_inequality (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  (x^2 - 2*x + 2) * (y^2 - 2*y + 2) * (z^2 - 2*z + 2) ≤ (x*y*z)^2 - 2*(x*y*z) + 2 :=
by sorry

end NUMINAMATH_CALUDE_product_inequality_l2737_273724


namespace NUMINAMATH_CALUDE_initial_girls_count_l2737_273744

theorem initial_girls_count (p : ℕ) : 
  (60 : ℚ) / 100 * p = 18 ∧ 
  ((60 : ℚ) / 100 * p - 3) / p = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_girls_count_l2737_273744


namespace NUMINAMATH_CALUDE_line_equation_through_points_l2737_273705

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℚ
  y₁ : ℚ
  x₂ : ℚ
  y₂ : ℚ

/-- The slope of a line -/
def Line.slope (l : Line) : ℚ := (l.y₂ - l.y₁) / (l.x₂ - l.x₁)

/-- The y-intercept of a line -/
def Line.yIntercept (l : Line) : ℚ := l.y₁ - l.slope * l.x₁

/-- The equation of a line in the form y = mx + b -/
def Line.equation (l : Line) (x : ℚ) : ℚ := l.slope * x + l.yIntercept

theorem line_equation_through_points :
  let l : Line := { x₁ := 2, y₁ := 3, x₂ := -1, y₂ := -1 }
  ∀ x, l.equation x = (4/3) * x + (1/3) := by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l2737_273705


namespace NUMINAMATH_CALUDE_new_water_height_after_cube_submersion_l2737_273771

/-- Calculates the new water height in a fish tank after submerging a cube -/
theorem new_water_height_after_cube_submersion
  (tank_width : ℝ)
  (tank_length : ℝ)
  (initial_height : ℝ)
  (cube_edge : ℝ)
  (h_width : tank_width = 50)
  (h_length : tank_length = 16)
  (h_initial_height : initial_height = 15)
  (h_cube_edge : cube_edge = 10) :
  let tank_area := tank_width * tank_length
  let cube_volume := cube_edge ^ 3
  let height_increase := cube_volume / tank_area
  let new_height := initial_height + height_increase
  new_height = 16.25 := by sorry

end NUMINAMATH_CALUDE_new_water_height_after_cube_submersion_l2737_273771


namespace NUMINAMATH_CALUDE_parking_methods_count_l2737_273782

/-- Represents the number of parking spaces -/
def n : ℕ := 6

/-- Represents the number of cars to be parked -/
def k : ℕ := 3

/-- Calculates the number of ways to park cars when they are not adjacent -/
def non_adjacent_ways : ℕ := (n - k + 1).choose k * 2^k

/-- Calculates the number of ways to park cars when two are adjacent -/
def two_adjacent_ways : ℕ := 2 * k.choose 2 * (n - k).choose 1 * 2^2

/-- Calculates the number of ways to park cars when all are adjacent -/
def all_adjacent_ways : ℕ := (n - k + 1) * 2

/-- The total number of parking methods -/
def total_parking_methods : ℕ := non_adjacent_ways + two_adjacent_ways + all_adjacent_ways

theorem parking_methods_count : total_parking_methods = 528 := by sorry

end NUMINAMATH_CALUDE_parking_methods_count_l2737_273782


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_31_l2737_273786

theorem smallest_n_divisible_by_31 :
  ∃ (n : ℕ), n > 0 ∧ (31 ∣ (5^n + n)) ∧ ∀ (m : ℕ), m > 0 ∧ (31 ∣ (5^m + m)) → n ≤ m :=
by
  use 30
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_31_l2737_273786


namespace NUMINAMATH_CALUDE_not_all_x_heartsuit_zero_eq_x_l2737_273707

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := |x - y|

-- Theorem stating that "x ♡ 0 = x for all x" is false
theorem not_all_x_heartsuit_zero_eq_x : ¬ ∀ x : ℝ, heartsuit x 0 = x := by
  sorry

end NUMINAMATH_CALUDE_not_all_x_heartsuit_zero_eq_x_l2737_273707


namespace NUMINAMATH_CALUDE_part_one_part_two_l2737_273798

/-- Set A defined as {x | -2 ≤ x ≤ 2} -/
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

/-- Set B defined as {x | 1-m ≤ x ≤ 2m-2} where m is a real number -/
def B (m : ℝ) : Set ℝ := {x | 1-m ≤ x ∧ x ≤ 2*m-2}

/-- Theorem for part (1) -/
theorem part_one (m : ℝ) : A ⊆ B m ∧ A ≠ B m → m ∈ Set.Ici 3 := by sorry

/-- Theorem for part (2) -/
theorem part_two (m : ℝ) : A ∩ B m = B m → m ∈ Set.Iic 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2737_273798


namespace NUMINAMATH_CALUDE_correct_age_order_l2737_273790

-- Define the set of friends
inductive Friend : Type
| David : Friend
| Emma : Friend
| Fiona : Friend
| George : Friend

-- Define a type for age comparisons
def AgeOrder := Friend → Friend → Prop

-- Define the problem conditions
def ProblemConditions (order : AgeOrder) : Prop :=
  -- All friends have different ages
  (∀ x y : Friend, x ≠ y → (order x y ∨ order y x)) ∧
  (∀ x y : Friend, order x y → ¬order y x) ∧
  -- Exactly one of the following statements is true
  (((order Friend.Emma Friend.David) ∧ 
    ¬(¬(order Friend.Fiona Friend.Emma ∧ order Friend.Fiona Friend.David ∧ order Friend.Fiona Friend.George)) ∧
    ¬(∀ x : Friend, order Friend.George x) ∧
    ¬(∃ x : Friend, order x Friend.David)) ∨
   (¬(order Friend.Emma Friend.David) ∧ 
    (¬(order Friend.Fiona Friend.Emma ∧ order Friend.Fiona Friend.David ∧ order Friend.Fiona Friend.George)) ∧
    ¬(∀ x : Friend, order Friend.George x) ∧
    ¬(∃ x : Friend, order x Friend.David)) ∨
   (¬(order Friend.Emma Friend.David) ∧ 
    ¬(¬(order Friend.Fiona Friend.Emma ∧ order Friend.Fiona Friend.David ∧ order Friend.Fiona Friend.George)) ∧
    (∀ x : Friend, order Friend.George x) ∧
    ¬(∃ x : Friend, order x Friend.David)) ∨
   (¬(order Friend.Emma Friend.David) ∧ 
    ¬(¬(order Friend.Fiona Friend.Emma ∧ order Friend.Fiona Friend.David ∧ order Friend.Fiona Friend.George)) ∧
    ¬(∀ x : Friend, order Friend.George x) ∧
    (∃ x : Friend, order x Friend.David)))

-- State the theorem
theorem correct_age_order (order : AgeOrder) :
  ProblemConditions order →
  (order Friend.David Friend.Emma ∧
   order Friend.Emma Friend.George ∧
   order Friend.George Friend.Fiona) :=
by sorry

end NUMINAMATH_CALUDE_correct_age_order_l2737_273790


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2737_273750

theorem quadratic_equation_roots (k : ℕ) 
  (distinct_roots : ∃ x y : ℕ+, x ≠ y ∧ 
    (k^2 - 1) * x^2 - 6 * (3*k - 1) * x + 72 = 0 ∧
    (k^2 - 1) * y^2 - 6 * (3*k - 1) * y + 72 = 0) :
  k = 2 ∧ ∃ x y : ℕ+, x = 6 ∧ y = 4 ∧
    (k^2 - 1) * x^2 - 6 * (3*k - 1) * x + 72 = 0 ∧
    (k^2 - 1) * y^2 - 6 * (3*k - 1) * y + 72 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2737_273750


namespace NUMINAMATH_CALUDE_a_range_l2737_273715

theorem a_range (a : ℝ) : 
  (∀ x : ℝ, |x - a| - |x| < 2 - a^2) → 
  a > -1 ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_a_range_l2737_273715


namespace NUMINAMATH_CALUDE_solve_lawyer_problem_l2737_273723

def lawyer_problem (upfront_fee : ℝ) (hourly_rate : ℝ) (court_hours : ℝ) (total_payment : ℝ) : Prop :=
  let court_cost := hourly_rate * court_hours
  let total_cost := upfront_fee + court_cost
  let prep_cost := total_payment - total_cost
  let prep_hours := prep_cost / hourly_rate
  let johns_payment := total_payment / 2
  johns_payment = 4000 ∧ prep_hours / court_hours = 2 / 5

theorem solve_lawyer_problem : 
  lawyer_problem 1000 100 50 8000 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_lawyer_problem_l2737_273723


namespace NUMINAMATH_CALUDE_tv_price_change_l2737_273727

theorem tv_price_change (P : ℝ) : P > 0 →
  let price_after_decrease := P * (1 - 0.20)
  let price_after_increase := price_after_decrease * (1 + 0.30)
  price_after_increase = P * 1.04 :=
by
  sorry

end NUMINAMATH_CALUDE_tv_price_change_l2737_273727


namespace NUMINAMATH_CALUDE_wang_speed_inequality_l2737_273714

theorem wang_speed_inequality (a b v : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a < b) 
  (hv : v = 2 * a * b / (a + b)) : a < v ∧ v < Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_wang_speed_inequality_l2737_273714


namespace NUMINAMATH_CALUDE_playground_area_l2737_273775

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  breadth : ℝ
  length : ℝ
  playgroundArea : ℝ

/-- Theorem: The area of the playground in a rectangular landscape -/
theorem playground_area (l : Landscape) : 
  l.length = 4 * l.breadth → 
  l.length = 120 → 
  l.playgroundArea = (1/3) * (l.length * l.breadth) → 
  l.playgroundArea = 1200 := by
  sorry

/-- The main result -/
def main_result : ℝ := 1200

#check playground_area
#check main_result

end NUMINAMATH_CALUDE_playground_area_l2737_273775


namespace NUMINAMATH_CALUDE_xyz_divisible_by_55_l2737_273794

theorem xyz_divisible_by_55 (x y z a b c : ℤ) 
  (h1 : x^2 + y^2 = a^2) 
  (h2 : y^2 + z^2 = b^2) 
  (h3 : z^2 + x^2 = c^2) : 
  55 ∣ (x * y * z) := by
  sorry

end NUMINAMATH_CALUDE_xyz_divisible_by_55_l2737_273794


namespace NUMINAMATH_CALUDE_f_max_at_neg_two_l2737_273753

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := -2 * x^2 - 8 * x + 18

/-- The statement that f attains its maximum at x = -2 with a value of 26 -/
theorem f_max_at_neg_two :
  (∃ (a : ℝ), ∀ (x : ℝ), f x ≤ f a) ∧
  (∀ (x : ℝ), f x ≤ 26) ∧
  (f (-2) = 26) :=
sorry

end NUMINAMATH_CALUDE_f_max_at_neg_two_l2737_273753


namespace NUMINAMATH_CALUDE_stating_line_triangle_intersection_count_l2737_273733

/-- Represents the number of intersection points between a line and a triangle's boundary. -/
inductive IntersectionCount
  | Zero
  | One
  | Two
  | Infinite

/-- A triangle in a 2D plane. -/
structure Triangle where
  -- Add necessary fields (e.g., vertices) here

/-- A line in a 2D plane. -/
structure Line where
  -- Add necessary fields (e.g., points or coefficients) here

/-- 
  Theorem stating that the number of intersection points between a line and 
  a triangle's boundary is either 0, 1, 2, or infinitely many.
-/
theorem line_triangle_intersection_count 
  (t : Triangle) (l : Line) : 
  ∃ (count : IntersectionCount), 
    (count = IntersectionCount.Zero) ∨ 
    (count = IntersectionCount.One) ∨ 
    (count = IntersectionCount.Two) ∨ 
    (count = IntersectionCount.Infinite) :=
by
  sorry


end NUMINAMATH_CALUDE_stating_line_triangle_intersection_count_l2737_273733


namespace NUMINAMATH_CALUDE_mia_money_l2737_273767

/-- Given that Darwin has $45 and Mia has $20 more than twice as much money as Darwin,
    prove that Mia has $110. -/
theorem mia_money (darwin_money : ℕ) (mia_money : ℕ) : 
  darwin_money = 45 → 
  mia_money = 2 * darwin_money + 20 → 
  mia_money = 110 := by
sorry

end NUMINAMATH_CALUDE_mia_money_l2737_273767


namespace NUMINAMATH_CALUDE_simplify_expression_l2737_273718

theorem simplify_expression (b : ℝ) : (1 : ℝ) * (2 * b) * (3 * b^2) * (4 * b^3) * (6 * b^5) = 144 * b^11 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2737_273718


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2737_273799

/-- The repeating decimal 0.565656... -/
def repeating_decimal : ℚ :=
  0 + (56 / 100) * (1 / (1 - 1/100))

/-- The target fraction 56/99 -/
def target_fraction : ℚ := 56 / 99

/-- Theorem stating that the repeating decimal 0.565656... is equal to 56/99 -/
theorem repeating_decimal_equals_fraction :
  repeating_decimal = target_fraction := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2737_273799


namespace NUMINAMATH_CALUDE_greatest_c_for_no_minus_seven_l2737_273751

theorem greatest_c_for_no_minus_seven : ∃ c : ℤ, 
  (∀ x : ℝ, x^2 + c*x + 20 ≠ -7) ∧
  (∀ d : ℤ, d > c → ∃ x : ℝ, x^2 + d*x + 20 = -7) ∧
  c = 10 := by
sorry

end NUMINAMATH_CALUDE_greatest_c_for_no_minus_seven_l2737_273751


namespace NUMINAMATH_CALUDE_smoking_health_negative_correlation_l2737_273797

-- Define the type for relationships
inductive Relationship
| ParentChildHeight
| SmokingHealth
| CropYieldFertilization
| MathPhysicsGrades

-- Define a function to determine if a relationship is negatively correlated
def is_negatively_correlated (r : Relationship) : Prop :=
  match r with
  | Relationship.SmokingHealth => True
  | _ => False

-- Theorem statement
theorem smoking_health_negative_correlation :
  ∀ r : Relationship, is_negatively_correlated r ↔ r = Relationship.SmokingHealth :=
by sorry

end NUMINAMATH_CALUDE_smoking_health_negative_correlation_l2737_273797


namespace NUMINAMATH_CALUDE_chord_intersection_sum_of_squares_l2737_273787

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit
def O : Point := Unit.unit -- Center of the circle
def A : Point := Unit.unit
def B : Point := Unit.unit
def C : Point := Unit.unit
def D : Point := Unit.unit
def E : Point := Unit.unit

-- Define the radius of the circle
def radius : ℝ := 10

-- Define the necessary functions and properties
def isOnCircle (p : Point) : Prop := sorry
def isChord (p q : Point) : Prop := sorry
def isDiameter (p q : Point) : Prop := sorry
def intersectsAt (l1 l2 : Point × Point) (p : Point) : Prop := sorry
def distance (p q : Point) : ℝ := sorry
def angle (p q r : Point) : ℝ := sorry

-- State the theorem
theorem chord_intersection_sum_of_squares 
  (h1 : isOnCircle A ∧ isOnCircle B ∧ isOnCircle C ∧ isOnCircle D)
  (h2 : isDiameter A B)
  (h3 : isChord C D)
  (h4 : intersectsAt (A, B) (C, D) E)
  (h5 : distance B E = 6)
  (h6 : angle A E C = 60) :
  (distance C E)^2 + (distance D E)^2 = 300 := by sorry

end NUMINAMATH_CALUDE_chord_intersection_sum_of_squares_l2737_273787


namespace NUMINAMATH_CALUDE_range_of_a_range_of_x_l2737_273755

-- Define the conditions p and q
def p (x : ℝ) : Prop := Real.sqrt (x - 1) ≤ 1
def q (x a : ℝ) : Prop := -1 ≤ x ∧ x ≤ a

-- Define the set A based on condition p
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

-- Define the set B based on condition q
def B (a : ℝ) : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}

-- Theorem 1: Range of a when q is necessary but not sufficient for p
theorem range_of_a : 
  ∀ a : ℝ, (∀ x : ℝ, p x → q x a) ∧ (∃ x : ℝ, q x a ∧ ¬p x) ↔ 2 ≤ a :=
sorry

-- Theorem 2: Range of x when a = 1 and at least one of p or q holds true
theorem range_of_x : 
  ∀ x : ℝ, (p x ∨ q x 1) ↔ -1 ≤ x ∧ x ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_x_l2737_273755


namespace NUMINAMATH_CALUDE_binomial_variance_l2737_273749

/-- A random variable following a binomial distribution with two outcomes -/
structure BinomialRV where
  p : ℝ  -- Probability of success (X = 1)
  q : ℝ  -- Probability of failure (X = 0)
  sum_one : p + q = 1  -- Sum of probabilities is 1

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.p * X.q

/-- Theorem: The variance of a binomial random variable X is equal to pq -/
theorem binomial_variance (X : BinomialRV) : variance X = X.p * X.q := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_l2737_273749


namespace NUMINAMATH_CALUDE_binomial_coefficient_equation_l2737_273739

theorem binomial_coefficient_equation : 
  ∀ n : ℤ, (Nat.choose 25 n.toNat + Nat.choose 25 12 = Nat.choose 26 13) ↔ (n = 11 ∨ n = 13) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equation_l2737_273739


namespace NUMINAMATH_CALUDE_problem_solution_g_minimum_l2737_273737

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x - a| + a

-- State the theorem
theorem problem_solution :
  (∀ x, f 1 x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧
  (∀ m : ℝ, (∃ t : ℝ, f 1 (t/2) ≤ m - f 1 (-t)) ↔ 3.5 ≤ m) := by
  sorry

-- Define the function for the second part
def g (t : ℝ) : ℝ := |t - 1| + |2 * t + 1| + 2

-- State the minimum value theorem
theorem g_minimum : ∀ t : ℝ, g t ≥ 3.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_g_minimum_l2737_273737


namespace NUMINAMATH_CALUDE_largest_angle_in_345_ratio_triangle_l2737_273754

theorem largest_angle_in_345_ratio_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 →
    b = (4/3) * a →
    c = (5/3) * a →
    a + b + c = 180 →
    c = 75 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_345_ratio_triangle_l2737_273754


namespace NUMINAMATH_CALUDE_prob_equals_two_thirteenths_l2737_273730

-- Define the deck
def total_cards : ℕ := 52
def num_queens : ℕ := 4
def num_jacks : ℕ := 4

-- Define the event
def prob_two_jacks_or_at_least_one_queen : ℚ :=
  (num_jacks * (num_jacks - 1)) / (total_cards * (total_cards - 1)) +
  (num_queens * (total_cards - num_queens)) / (total_cards * (total_cards - 1)) +
  (num_queens * (num_queens - 1)) / (total_cards * (total_cards - 1))

-- State the theorem
theorem prob_equals_two_thirteenths :
  prob_two_jacks_or_at_least_one_queen = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_equals_two_thirteenths_l2737_273730


namespace NUMINAMATH_CALUDE_largest_quantity_l2737_273719

theorem largest_quantity : 
  let A := (3010 : ℚ) / 3009 + 3010 / 3011
  let B := (3010 : ℚ) / 3011 + 3012 / 3011
  let C := (3011 : ℚ) / 3010 + 3011 / 3012
  A > B ∧ A > C := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l2737_273719


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2737_273757

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y, x > 0 ∧ y > 0 → x * y > 0) ∧
  (∃ x y, x * y > 0 ∧ ¬(x > 0 ∧ y > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2737_273757


namespace NUMINAMATH_CALUDE_total_red_cards_l2737_273720

/-- The number of decks of playing cards --/
def num_decks : ℕ := 8

/-- The number of red cards in one standard deck --/
def red_cards_per_deck : ℕ := 26

/-- Theorem: The total number of red cards in 8 decks is 208 --/
theorem total_red_cards : num_decks * red_cards_per_deck = 208 := by
  sorry

end NUMINAMATH_CALUDE_total_red_cards_l2737_273720


namespace NUMINAMATH_CALUDE_inequality_comparison_l2737_273736

theorem inequality_comparison (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < b + c) (hbc : b < c + a) : 
  let K := a^4 + b^4 + c^4 - 2*(a^2*b^2 + b^2*c^2 + c^2*a^2)
  (K < 0 ↔ c < a + b) ∧ 
  (K = 0 ↔ c = a + b) ∧ 
  (K > 0 ↔ c > a + b) := by
sorry

end NUMINAMATH_CALUDE_inequality_comparison_l2737_273736


namespace NUMINAMATH_CALUDE_cyclists_meet_time_l2737_273735

/-- Two cyclists on a circular track meet at the starting point -/
theorem cyclists_meet_time (v1 v2 circumference : ℝ) (h1 : v1 = 7) (h2 : v2 = 8) (h3 : circumference = 300) : 
  (circumference / (v1 + v2) = 20) := by
  sorry

end NUMINAMATH_CALUDE_cyclists_meet_time_l2737_273735


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l2737_273777

theorem prime_sum_theorem (a b : ℕ) : 
  Prime a → Prime b → a^2 + b = 2003 → a + b = 2001 := by sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l2737_273777


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l2737_273774

theorem smallest_integer_satisfying_conditions : ∃ (N x y : ℕ), 
  N > 0 ∧ 
  (N : ℚ) = 1.2 * x ∧ 
  (N : ℚ) = 0.81 * y ∧ 
  (∀ (M z w : ℕ), M > 0 → (M : ℚ) = 1.2 * z → (M : ℚ) = 0.81 * w → M ≥ N) ∧
  N = 162 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l2737_273774


namespace NUMINAMATH_CALUDE_complex_multiplication_l2737_273748

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 - i) = 1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2737_273748


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l2737_273722

-- Define a function to represent the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Define a predicate to check if a number only uses allowed digits
def uses_allowed_digits (n : ℕ) : Prop := sorry

theorem sum_of_digits_M (M : ℕ) 
  (h_even : Even M)
  (h_digits : uses_allowed_digits M)
  (h_double : sum_of_digits (2 * M) = 31)
  (h_half : sum_of_digits (M / 2) = 28) :
  sum_of_digits M = 29 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l2737_273722


namespace NUMINAMATH_CALUDE_german_team_goals_l2737_273756

def journalist1 (x : ℕ) : Prop := 10 < x ∧ x < 17

def journalist2 (x : ℕ) : Prop := 11 < x ∧ x < 18

def journalist3 (x : ℕ) : Prop := x % 2 = 1

def exactlyTwoCorrect (x : ℕ) : Prop :=
  (journalist1 x ∧ journalist2 x ∧ ¬journalist3 x) ∨
  (journalist1 x ∧ ¬journalist2 x ∧ journalist3 x) ∨
  (¬journalist1 x ∧ journalist2 x ∧ journalist3 x)

theorem german_team_goals :
  {x : ℕ | exactlyTwoCorrect x} = {11, 12, 14, 16, 17} := by sorry

end NUMINAMATH_CALUDE_german_team_goals_l2737_273756


namespace NUMINAMATH_CALUDE_largest_smallest_three_digit_div_six_with_seven_l2737_273793

/-- A function that checks if a number contains the digit 7 --/
def contains_seven (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ (a = 7 ∨ b = 7 ∨ c = 7)

/-- A function that checks if a number is a three-digit number --/
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

/-- The main theorem --/
theorem largest_smallest_three_digit_div_six_with_seven :
  (∀ n : ℕ, is_three_digit n → n % 6 = 0 → contains_seven n → n ≤ 978) ∧
  (∀ n : ℕ, is_three_digit n → n % 6 = 0 → contains_seven n → 174 ≤ n) ∧
  is_three_digit 978 ∧ 978 % 6 = 0 ∧ contains_seven 978 ∧
  is_three_digit 174 ∧ 174 % 6 = 0 ∧ contains_seven 174 :=
by sorry

end NUMINAMATH_CALUDE_largest_smallest_three_digit_div_six_with_seven_l2737_273793


namespace NUMINAMATH_CALUDE_circles_intersect_l2737_273788

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

-- Define the distance between centers
def distance_between_centers : ℝ := 2

-- Define the radii of the circles
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  distance_between_centers > abs (radius1 - radius2) ∧
  distance_between_centers < radius1 + radius2 :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l2737_273788


namespace NUMINAMATH_CALUDE_complex_sum_simplification_l2737_273772

theorem complex_sum_simplification :
  let z₁ : ℂ := (-1 + Complex.I * Real.sqrt 7) / 2
  let z₂ : ℂ := (-1 - Complex.I * Real.sqrt 7) / 2
  z₁^8 + z₂^8 = -7.375 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_simplification_l2737_273772


namespace NUMINAMATH_CALUDE_base_for_125_with_4_digits_l2737_273796

theorem base_for_125_with_4_digits : ∃! b : ℕ, b > 1 ∧ b^3 ≤ 125 ∧ 125 < b^4 := by
  sorry

end NUMINAMATH_CALUDE_base_for_125_with_4_digits_l2737_273796


namespace NUMINAMATH_CALUDE_axe_sharpening_cost_l2737_273759

theorem axe_sharpening_cost
  (trees_per_sharpening : ℕ)
  (total_sharpening_cost : ℚ)
  (min_trees_chopped : ℕ)
  (h1 : trees_per_sharpening = 13)
  (h2 : total_sharpening_cost = 35)
  (h3 : min_trees_chopped ≥ 91) :
  let sharpenings := min_trees_chopped / trees_per_sharpening
  total_sharpening_cost / sharpenings = 5 := by
sorry

end NUMINAMATH_CALUDE_axe_sharpening_cost_l2737_273759


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2737_273764

/-- Atomic weight of Nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- Atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.01

/-- Atomic weight of Bromine in g/mol -/
def bromine_weight : ℝ := 79.90

/-- Number of Nitrogen atoms in the compound -/
def nitrogen_count : ℕ := 1

/-- Number of Hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 4

/-- Number of Bromine atoms in the compound -/
def bromine_count : ℕ := 1

/-- Molecular weight of the compound -/
def molecular_weight : ℝ := 
  nitrogen_count * nitrogen_weight + 
  hydrogen_count * hydrogen_weight + 
  bromine_count * bromine_weight

theorem compound_molecular_weight : 
  molecular_weight = 97.95 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2737_273764


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2737_273791

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) < 0
def solution_set_f_neg (x : ℝ) : Prop := x < -1 ∨ x > 1/3

-- Define the solution set of f(e^x) > 0
def solution_set_f_exp_pos (x : ℝ) : Prop := x < -Real.log 3

-- Theorem statement
theorem solution_set_equivalence :
  (∀ x, f x < 0 ↔ solution_set_f_neg x) →
  (∀ x, f (Real.exp x) > 0 ↔ solution_set_f_exp_pos x) :=
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2737_273791


namespace NUMINAMATH_CALUDE_inequality_solution_l2737_273721

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 4) ↔ 
  (x < -2 ∨ (-1 < x ∧ x < 0) ∨ 1 < x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2737_273721


namespace NUMINAMATH_CALUDE_total_students_l2737_273781

/-- The number of students taking history -/
def H : ℕ := 36

/-- The number of students taking statistics -/
def S : ℕ := 32

/-- The number of students taking history or statistics or both -/
def H_or_S : ℕ := 57

/-- The number of students taking history but not statistics -/
def H_not_S : ℕ := 25

/-- The theorem stating that the total number of students in the group is 57 -/
theorem total_students : H_or_S = 57 := by sorry

end NUMINAMATH_CALUDE_total_students_l2737_273781


namespace NUMINAMATH_CALUDE_scale_division_l2737_273731

/-- Represents the length of a scale in inches -/
def scale_length : ℕ := 125

/-- Represents the length of each part in inches -/
def part_length : ℕ := 25

/-- Theorem stating that the scale is divided into 5 equal parts -/
theorem scale_division :
  scale_length / part_length = 5 := by sorry

end NUMINAMATH_CALUDE_scale_division_l2737_273731


namespace NUMINAMATH_CALUDE_max_pencils_theorem_l2737_273741

/-- Represents the discount rules for pencil purchases -/
structure DiscountRules where
  large_set : Nat
  large_discount : Rat
  small_set : Nat
  small_discount : Rat

/-- Calculates the maximum number of pencils that can be purchased given initial funds and discount rules -/
def max_pencils (initial_funds : Nat) (rules : DiscountRules) : Nat :=
  sorry

/-- The theorem stating that given the specific initial funds and discount rules, the maximum number of pencils that can be purchased is 36 -/
theorem max_pencils_theorem (initial_funds : Nat) (rules : DiscountRules) :
  initial_funds = 30 ∧
  rules.large_set = 20 ∧
  rules.large_discount = 1/4 ∧
  rules.small_set = 5 ∧
  rules.small_discount = 1/10
  → max_pencils initial_funds rules = 36 := by
  sorry

end NUMINAMATH_CALUDE_max_pencils_theorem_l2737_273741


namespace NUMINAMATH_CALUDE_simplify_expression_l2737_273710

theorem simplify_expression (y : ℝ) : 4 * y^3 + 8 * y + 6 - (3 - 4 * y^3 - 8 * y) = 8 * y^3 + 16 * y + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2737_273710


namespace NUMINAMATH_CALUDE_nested_sqrt_power_l2737_273738

theorem nested_sqrt_power (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (15/16) := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_power_l2737_273738


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l2737_273709

theorem logarithm_sum_simplification :
  let a := (1 / (Real.log 3 / Real.log 21 + 1))
  let b := (1 / (Real.log 4 / Real.log 14 + 1))
  let c := (1 / (Real.log 7 / Real.log 9 + 1))
  let d := (1 / (Real.log 11 / Real.log 8 + 1))
  a + b + c + d = 1 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l2737_273709


namespace NUMINAMATH_CALUDE_emma_widget_production_difference_l2737_273743

/-- 
Given Emma's widget production rates and working hours on Monday and Tuesday, 
prove the difference in total widgets produced.
-/
theorem emma_widget_production_difference 
  (w t : ℕ) 
  (h1 : w = 3 * t) : 
  w * t - (w + 5) * (t - 3) = 4 * t + 15 := by
  sorry

end NUMINAMATH_CALUDE_emma_widget_production_difference_l2737_273743


namespace NUMINAMATH_CALUDE_contrapositive_false_proposition_l2737_273708

theorem contrapositive_false_proposition : 
  ¬(∀ x : ℝ, x ≠ 1 → x^2 ≠ 1) := by sorry

end NUMINAMATH_CALUDE_contrapositive_false_proposition_l2737_273708


namespace NUMINAMATH_CALUDE_common_root_and_other_roots_l2737_273734

def f (x : ℝ) : ℝ := x^4 - x^3 - 22*x^2 + 16*x + 96
def g (x : ℝ) : ℝ := x^3 - 2*x^2 - 3*x + 10

theorem common_root_and_other_roots :
  (f (-2) = 0 ∧ g (-2) = 0) ∧
  (f 3 = 0 ∧ f (-4) = 0 ∧ f 4 = 0) :=
sorry

end NUMINAMATH_CALUDE_common_root_and_other_roots_l2737_273734


namespace NUMINAMATH_CALUDE_class_composition_l2737_273717

/-- Represents the percentage of men in a college class -/
def percentage_men : ℝ := 40

/-- Represents the percentage of women in a college class -/
def percentage_women : ℝ := 100 - percentage_men

/-- Represents the percentage of women who are science majors -/
def women_science_percentage : ℝ := 20

/-- Represents the percentage of non-science majors in the class -/
def non_science_percentage : ℝ := 60

/-- Represents the percentage of men who are science majors -/
def men_science_percentage : ℝ := 70

theorem class_composition :
  percentage_men + percentage_women = 100 ∧
  women_science_percentage / 100 * percentage_women +
    men_science_percentage / 100 * percentage_men = 100 - non_science_percentage :=
by sorry

end NUMINAMATH_CALUDE_class_composition_l2737_273717


namespace NUMINAMATH_CALUDE_sum_of_digits_of_expression_l2737_273768

-- Define the expression
def expression : ℕ := (2 + 4)^15

-- Function to get the tens digit of a number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Function to get the ones digit of a number
def ones_digit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem sum_of_digits_of_expression :
  tens_digit expression + ones_digit expression = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_expression_l2737_273768


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l2737_273745

def n : Nat := 483045

theorem largest_prime_factors_difference (p q : Nat) :
  Nat.Prime p ∧ Nat.Prime q ∧
  p ∣ n ∧ q ∣ n ∧
  (∀ r, Nat.Prime r → r ∣ n → r ≤ p ∧ r ≤ q) →
  p ≠ q →
  (max p q) - (min p q) = 8 := by
sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l2737_273745


namespace NUMINAMATH_CALUDE_product_evaluation_l2737_273701

theorem product_evaluation (a : ℤ) (h : a = 3) :
  (a - 12) * (a - 11) * (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a * 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2737_273701


namespace NUMINAMATH_CALUDE_initial_lambs_correct_l2737_273713

/-- The number of lambs Mary initially had -/
def initial_lambs : ℕ := 6

/-- The number of lambs that had babies -/
def lambs_with_babies : ℕ := 2

/-- The number of babies each lamb had -/
def babies_per_lamb : ℕ := 2

/-- The number of lambs Mary traded -/
def traded_lambs : ℕ := 3

/-- The number of extra lambs Mary found -/
def found_lambs : ℕ := 7

/-- The total number of lambs Mary has now -/
def total_lambs : ℕ := 14

/-- Theorem stating that the initial number of lambs is correct given the conditions -/
theorem initial_lambs_correct : 
  initial_lambs + (lambs_with_babies * babies_per_lamb) - traded_lambs + found_lambs = total_lambs :=
by sorry

end NUMINAMATH_CALUDE_initial_lambs_correct_l2737_273713


namespace NUMINAMATH_CALUDE_two_zeros_iff_a_positive_l2737_273776

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x + a * (x - 1)^2

theorem two_zeros_iff_a_positive (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ 
   ∀ x₃ : ℝ, f a x₃ = 0 → x₃ = x₁ ∨ x₃ = x₂) ↔ 
  a > 0 :=
sorry

end NUMINAMATH_CALUDE_two_zeros_iff_a_positive_l2737_273776


namespace NUMINAMATH_CALUDE_inequality_proof_l2737_273766

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h : (a + b + c + d) * (1/a + 1/b + 1/c + 1/d) = 20) :
  (a^2 + b^2 + c^2 + d^2) * (1/a^2 + 1/b^2 + 1/c^2 + 1/d^2) ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2737_273766


namespace NUMINAMATH_CALUDE_tangent_segment_length_l2737_273725

/-- Given three circles where two touch externally and a common tangent, 
    calculate the length of the tangent segment within the third circle. -/
theorem tangent_segment_length 
  (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ = 3) 
  (h₂ : r₂ = 4) 
  (h₃ : r₃ = 5) 
  (h_touch : r₁ + r₂ = 7) : 
  ∃ (y : ℝ), y = (40 * Real.sqrt 3) / 7 ∧ 
  y = 2 * Real.sqrt (r₃^2 - ((r₂ - r₁)^2 / (4 * (r₁ + r₂)^2)) * r₃^2) := by
  sorry


end NUMINAMATH_CALUDE_tangent_segment_length_l2737_273725


namespace NUMINAMATH_CALUDE_state_return_cost_l2737_273760

/-- The cost of a federal tax return -/
def federal_cost : ℕ := 50

/-- The cost of quarterly business taxes -/
def quarterly_cost : ℕ := 80

/-- The number of federal returns sold -/
def federal_sold : ℕ := 60

/-- The number of state returns sold -/
def state_sold : ℕ := 20

/-- The number of quarterly returns sold -/
def quarterly_sold : ℕ := 10

/-- The total revenue -/
def total_revenue : ℕ := 4400

/-- The cost of a state return -/
def state_cost : ℕ := 30

theorem state_return_cost :
  federal_cost * federal_sold + state_cost * state_sold + quarterly_cost * quarterly_sold = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_state_return_cost_l2737_273760


namespace NUMINAMATH_CALUDE_complex_number_problem_l2737_273752

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I →
  z₂.im = 2 →
  ∃ (r : ℝ), z₁ * z₂ = r →
  z₂ = 4 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2737_273752


namespace NUMINAMATH_CALUDE_roots_of_polynomials_product_DE_l2737_273729

theorem roots_of_polynomials (r : ℝ) : 
  r^2 = r + 1 → r^6 = 8*r + 5 := by sorry

theorem product_DE : ∃ (D E : ℤ), 
  (∀ (r : ℝ), r^2 = r + 1 → r^6 = D*r + E) ∧ D*E = 40 := by sorry

end NUMINAMATH_CALUDE_roots_of_polynomials_product_DE_l2737_273729


namespace NUMINAMATH_CALUDE_exists_tetrahedron_all_obtuse_dihedral_angles_l2737_273746

/-- A tetrahedron is represented by its four vertices in 3D space -/
def Tetrahedron := Fin 4 → ℝ × ℝ × ℝ

/-- The dihedral angle between two faces of a tetrahedron -/
def dihedralAngle (t : Tetrahedron) (i j : Fin 4) : ℝ :=
  sorry  -- Definition of dihedral angle calculation

/-- A dihedral angle is obtuse if it's greater than π/2 -/
def isObtuse (angle : ℝ) : Prop := angle > Real.pi / 2

/-- Theorem: There exists a tetrahedron where all dihedral angles are obtuse -/
theorem exists_tetrahedron_all_obtuse_dihedral_angles :
  ∃ t : Tetrahedron, ∀ i j : Fin 4, i ≠ j → isObtuse (dihedralAngle t i j) :=
sorry

end NUMINAMATH_CALUDE_exists_tetrahedron_all_obtuse_dihedral_angles_l2737_273746
