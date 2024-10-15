import Mathlib

namespace NUMINAMATH_CALUDE_cube_space_diagonal_length_l2593_259390

/-- The length of a space diagonal in a cube with side length 15 -/
theorem cube_space_diagonal_length :
  ∀ (s : ℝ), s = 15 →
  ∃ (d : ℝ), d = s * Real.sqrt 3 ∧ d^2 = 3 * s^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_space_diagonal_length_l2593_259390


namespace NUMINAMATH_CALUDE_min_apples_in_basket_sixty_two_satisfies_conditions_min_apples_is_sixty_two_l2593_259376

theorem min_apples_in_basket (N : ℕ) : 
  (N % 3 = 2) ∧ (N % 4 = 2) ∧ (N % 5 = 2) → N ≥ 62 :=
by sorry

theorem sixty_two_satisfies_conditions : 
  (62 % 3 = 2) ∧ (62 % 4 = 2) ∧ (62 % 5 = 2) :=
by sorry

theorem min_apples_is_sixty_two : 
  ∃ (N : ℕ), (N % 3 = 2) ∧ (N % 4 = 2) ∧ (N % 5 = 2) ∧ N = 62 :=
by sorry

end NUMINAMATH_CALUDE_min_apples_in_basket_sixty_two_satisfies_conditions_min_apples_is_sixty_two_l2593_259376


namespace NUMINAMATH_CALUDE_inequality_holds_l2593_259301

theorem inequality_holds (x y : ℝ) (h : 2 * y + 5 * x = 10) : 3 * x * y - x^2 - y^2 < 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l2593_259301


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l2593_259313

theorem profit_percentage_calculation 
  (tv_cost dvd_cost selling_price : ℕ) : 
  tv_cost = 16000 → 
  dvd_cost = 6250 → 
  selling_price = 35600 → 
  (selling_price - (tv_cost + dvd_cost)) * 100 / (tv_cost + dvd_cost) = 60 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l2593_259313


namespace NUMINAMATH_CALUDE_line_segments_in_proportion_l2593_259392

theorem line_segments_in_proportion (a b c d : ℝ) : 
  a = 5 ∧ b = 15 ∧ c = 3 ∧ d = 9 → a * d = b * c := by
  sorry

end NUMINAMATH_CALUDE_line_segments_in_proportion_l2593_259392


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2593_259318

theorem negative_fraction_comparison : -4/5 < -2/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2593_259318


namespace NUMINAMATH_CALUDE_factorization_proof_l2593_259336

theorem factorization_proof (x y : ℝ) : x * y^2 - 6 * x * y + 9 * x = x * (y - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2593_259336


namespace NUMINAMATH_CALUDE_range_of_fraction_l2593_259379

theorem range_of_fraction (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 4) (hy : 3 ≤ y ∧ y ≤ 6) :
  (∃ (x₁ y₁ : ℝ), 1 ≤ x₁ ∧ x₁ ≤ 4 ∧ 3 ≤ y₁ ∧ y₁ ≤ 6 ∧ x₁ / y₁ = 1/6) ∧
  (∃ (x₂ y₂ : ℝ), 1 ≤ x₂ ∧ x₂ ≤ 4 ∧ 3 ≤ y₂ ∧ y₂ ≤ 6 ∧ x₂ / y₂ = 4/3) ∧
  (∀ (x' y' : ℝ), 1 ≤ x' ∧ x' ≤ 4 → 3 ≤ y' ∧ y' ≤ 6 → 1/6 ≤ x' / y' ∧ x' / y' ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_fraction_l2593_259379


namespace NUMINAMATH_CALUDE_eleven_million_scientific_notation_l2593_259380

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem eleven_million_scientific_notation :
  toScientificNotation 11000000 = ScientificNotation.mk 1.1 7 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_eleven_million_scientific_notation_l2593_259380


namespace NUMINAMATH_CALUDE_new_person_weight_l2593_259320

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (replaced_weight : ℝ) (average_increase : ℝ) : ℝ :=
  initial_count * average_increase + replaced_weight

/-- Theorem stating that the weight of the new person is 93 kg -/
theorem new_person_weight :
  weight_of_new_person 8 65 3.5 = 93 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2593_259320


namespace NUMINAMATH_CALUDE_negative_two_exponent_sum_l2593_259304

theorem negative_two_exponent_sum : (-2)^2023 + (-2)^2024 = 2^2023 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_exponent_sum_l2593_259304


namespace NUMINAMATH_CALUDE_non_union_women_percent_is_75_percent_l2593_259328

/-- Represents the composition of employees in a company -/
structure CompanyComposition where
  total : ℕ
  men_percent : ℚ
  union_percent : ℚ
  union_men_percent : ℚ

/-- Calculates the percentage of women among non-union employees -/
def non_union_women_percent (c : CompanyComposition) : ℚ :=
  let total_men := c.men_percent * c.total
  let total_women := c.total - total_men
  let union_employees := c.union_percent * c.total
  let union_men := c.union_men_percent * union_employees
  let non_union_men := total_men - union_men
  let non_union_total := c.total - union_employees
  let non_union_women := non_union_total - non_union_men
  non_union_women / non_union_total

/-- Theorem stating that given the company composition, 
    the percentage of women among non-union employees is 75% -/
theorem non_union_women_percent_is_75_percent 
  (c : CompanyComposition) 
  (h1 : c.men_percent = 52/100)
  (h2 : c.union_percent = 60/100)
  (h3 : c.union_men_percent = 70/100) :
  non_union_women_percent c = 75/100 := by
  sorry

end NUMINAMATH_CALUDE_non_union_women_percent_is_75_percent_l2593_259328


namespace NUMINAMATH_CALUDE_smallest_sum_20_consecutive_twice_square_l2593_259356

/-- The sum of 20 consecutive integers starting from n -/
def sum_20_consecutive (n : ℕ) : ℕ := 10 * (2 * n + 19)

/-- Predicate to check if a number is twice a perfect square -/
def is_twice_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = 2 * k^2

/-- The smallest sum of 20 consecutive positive integers that is twice a perfect square -/
theorem smallest_sum_20_consecutive_twice_square : 
  (∃ n : ℕ, sum_20_consecutive n = 450 ∧ 
    is_twice_perfect_square (sum_20_consecutive n) ∧
    ∀ m : ℕ, m < n → ¬(is_twice_perfect_square (sum_20_consecutive m))) :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_20_consecutive_twice_square_l2593_259356


namespace NUMINAMATH_CALUDE_chocolate_bunny_value_is_100_l2593_259367

/-- The number of points needed to win the Nintendo Switch -/
def total_points_needed : ℕ := 2000

/-- The number of chocolate bunnies already sold -/
def chocolate_bunnies_sold : ℕ := 8

/-- The number of points earned per Snickers bar -/
def points_per_snickers : ℕ := 25

/-- The number of Snickers bars needed to win the Nintendo Switch -/
def snickers_bars_needed : ℕ := 48

/-- The value of each chocolate bunny in points -/
def chocolate_bunny_value : ℕ := (total_points_needed - (points_per_snickers * snickers_bars_needed)) / chocolate_bunnies_sold

theorem chocolate_bunny_value_is_100 : chocolate_bunny_value = 100 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bunny_value_is_100_l2593_259367


namespace NUMINAMATH_CALUDE_dice_probability_l2593_259349

/-- The number of dice -/
def num_dice : ℕ := 4

/-- The number of sides on each die -/
def sides_per_die : ℕ := 8

/-- The probability of all dice showing the same number -/
def prob_all_same : ℚ := 1 / (sides_per_die ^ (num_dice - 1))

theorem dice_probability :
  prob_all_same = 1 / 512 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l2593_259349


namespace NUMINAMATH_CALUDE_cube_space_division_l2593_259309

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- A plane is a flat, two-dimensional surface -/
structure Plane where
  -- We don't need to define the specifics of a plane for this problem

/-- The number of parts that a cube and its face planes divide space into -/
def space_division (c : Cube) : Nat :=
  sorry

/-- Theorem stating that a cube and its face planes divide space into 27 parts -/
theorem cube_space_division (c : Cube) : space_division c = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_space_division_l2593_259309


namespace NUMINAMATH_CALUDE_square_sum_of_difference_and_product_l2593_259374

theorem square_sum_of_difference_and_product (a b : ℝ) 
  (h1 : a - b = 4) 
  (h2 : a * b = 1) : 
  a^2 + b^2 = 18 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_difference_and_product_l2593_259374


namespace NUMINAMATH_CALUDE_terminal_zeros_fifty_times_three_sixty_l2593_259325

-- Define the prime factorizations of 50 and 360
def fifty : ℕ := 2 * 5^2
def three_sixty : ℕ := 2^3 * 3^2 * 5

-- Function to count terminal zeros
def count_terminal_zeros (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem terminal_zeros_fifty_times_three_sixty : 
  count_terminal_zeros (fifty * three_sixty) = 3 := by sorry

end NUMINAMATH_CALUDE_terminal_zeros_fifty_times_three_sixty_l2593_259325


namespace NUMINAMATH_CALUDE_phi_value_l2593_259395

/-- Given a function f(x) = 2sin(ωx + φ) with the following properties:
    - ω > 0
    - |φ| < π/2
    - x = 5π/8 is an axis of symmetry for y = f(x)
    - x = 11π/8 is a zero of f(x)
    - The smallest positive period of f(x) is greater than 2π
    Prove that φ = π/12 -/
theorem phi_value (ω φ : Real) (h1 : ω > 0) (h2 : |φ| < π/2)
  (h3 : ∀ x, 2 * Real.sin (ω * (5*π/4 - (x - 5*π/8)) + φ) = 2 * Real.sin (ω * x + φ))
  (h4 : 2 * Real.sin (ω * 11*π/8 + φ) = 0)
  (h5 : 2*π / ω > 2*π) : φ = π/12 := by
  sorry

end NUMINAMATH_CALUDE_phi_value_l2593_259395


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2593_259368

theorem unique_three_digit_number : ∃! abc : ℕ,
  (abc ≥ 100 ∧ abc < 1000) ∧
  (abc % 100 = (abc / 100) ^ 2) ∧
  (abc % 9 = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2593_259368


namespace NUMINAMATH_CALUDE_circles_tangent_radius_l2593_259338

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 5 = 0
def circle_O2 (x y r : ℝ) : Prop := (x+2)^2 + y^2 = r^2

-- Theorem statement
theorem circles_tangent_radius (r : ℝ) :
  (r > 0) →
  (∃! p : ℝ × ℝ, circle_O1 p.1 p.2 ∧ circle_O2 p.1 p.2 r) →
  r = 1 ∨ r = 9 := by
  sorry

end NUMINAMATH_CALUDE_circles_tangent_radius_l2593_259338


namespace NUMINAMATH_CALUDE_two_dress_combinations_l2593_259321

def num_colors : Nat := 4
def num_patterns : Nat := 5

theorem two_dress_combinations : 
  (num_colors * num_patterns) * ((num_colors - 1) * (num_patterns - 1)) = 240 := by
  sorry

end NUMINAMATH_CALUDE_two_dress_combinations_l2593_259321


namespace NUMINAMATH_CALUDE_half_dollar_percentage_l2593_259360

def nickel_value : ℚ := 5
def half_dollar_value : ℚ := 50
def num_nickels : ℕ := 75
def num_half_dollars : ℕ := 30

theorem half_dollar_percentage :
  (num_half_dollars * half_dollar_value) / 
  (num_nickels * nickel_value + num_half_dollars * half_dollar_value) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_half_dollar_percentage_l2593_259360


namespace NUMINAMATH_CALUDE_complement_union_A_B_l2593_259333

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | (x - 2) * (x + 1) ≤ 0}

def B : Set ℝ := {x | 0 ≤ x ∧ x < 3}

theorem complement_union_A_B : 
  (Uᶜ ∩ (A ∪ B)ᶜ) = {x | x < -1 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l2593_259333


namespace NUMINAMATH_CALUDE_election_percentage_l2593_259382

theorem election_percentage (total_votes : ℕ) (second_candidate_votes : ℕ) :
  total_votes = 1200 →
  second_candidate_votes = 240 →
  (total_votes - second_candidate_votes : ℝ) / total_votes * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_election_percentage_l2593_259382


namespace NUMINAMATH_CALUDE_collins_savings_l2593_259344

-- Define the constants
def cans_at_home : ℕ := 12
def cans_from_neighbor : ℕ := 46
def cans_from_office : ℕ := 250
def price_per_can : ℚ := 25 / 100

-- Define the functions
def cans_at_grandparents : ℕ := 3 * cans_at_home

def total_cans : ℕ := cans_at_home + cans_at_grandparents + cans_from_neighbor + cans_from_office

def total_money : ℚ := (total_cans : ℚ) * price_per_can

def savings_amount : ℚ := total_money / 2

-- Theorem statement
theorem collins_savings : savings_amount = 43 := by
  sorry

end NUMINAMATH_CALUDE_collins_savings_l2593_259344


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2593_259387

/-- The complex number z = i(3+i) corresponds to a point in the second quadrant of the complex plane. -/
theorem complex_number_in_second_quadrant : ∃ (x y : ℝ), Complex.I * (3 + Complex.I) = Complex.mk x y ∧ x < 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2593_259387


namespace NUMINAMATH_CALUDE_infinitely_many_n_divisible_by_sqrt3_d_l2593_259381

def d (n : ℕ+) : ℕ := (Nat.divisors n.val).card

theorem infinitely_many_n_divisible_by_sqrt3_d :
  Set.Infinite {n : ℕ+ | ∃ k : ℕ+, n = k * ⌊Real.sqrt 3 * d n⌋} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_n_divisible_by_sqrt3_d_l2593_259381


namespace NUMINAMATH_CALUDE_pants_price_is_54_l2593_259346

/-- The price of a pair of pants Laura bought -/
def price_of_pants : ℕ := sorry

/-- The number of pairs of pants Laura bought -/
def num_pants : ℕ := 2

/-- The number of shirts Laura bought -/
def num_shirts : ℕ := 4

/-- The price of each shirt -/
def price_of_shirt : ℕ := 33

/-- The amount Laura gave to the cashier -/
def amount_given : ℕ := 250

/-- The change Laura received -/
def change_received : ℕ := 10

theorem pants_price_is_54 : price_of_pants = 54 := by
  sorry

end NUMINAMATH_CALUDE_pants_price_is_54_l2593_259346


namespace NUMINAMATH_CALUDE_sin_double_angle_l2593_259335

theorem sin_double_angle (x : Real) (h : Real.sin (x + π/4) = 4/5) : 
  Real.sin (2*x) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_l2593_259335


namespace NUMINAMATH_CALUDE_mrs_blue_garden_yield_l2593_259396

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected tomato yield from a garden -/
def expectedTomatoYield (garden : GardenDimensions) (stepLength : ℚ) (yieldPerSqFt : ℚ) : ℚ :=
  (garden.length : ℚ) * stepLength * (garden.width : ℚ) * stepLength * yieldPerSqFt

/-- Theorem stating the expected tomato yield for Mrs. Blue's garden -/
theorem mrs_blue_garden_yield :
  let garden : GardenDimensions := { length := 18, width := 24 }
  let stepLength : ℚ := 3/2
  let yieldPerSqFt : ℚ := 2/3
  expectedTomatoYield garden stepLength yieldPerSqFt = 648 := by
  sorry

end NUMINAMATH_CALUDE_mrs_blue_garden_yield_l2593_259396


namespace NUMINAMATH_CALUDE_range_of_a_l2593_259345

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x) 
  (h2 : ∃ x : ℝ, x^2 + 4*x + a = 0) : 
  a ∈ Set.Icc (Real.exp 1) 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2593_259345


namespace NUMINAMATH_CALUDE_diamond_three_four_l2593_259350

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem diamond_three_four : diamond 3 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_four_l2593_259350


namespace NUMINAMATH_CALUDE_switch_pairs_bound_l2593_259359

/-- Represents a row in Pascal's Triangle --/
def PascalRow (n : ℕ) := List ℕ

/-- Counts the number of odd entries in a Pascal's Triangle row --/
def countOddEntries (row : PascalRow n) : ℕ := sorry

/-- Counts the number of switch pairs in a Pascal's Triangle row --/
def countSwitchPairs (row : PascalRow n) : ℕ := sorry

/-- Theorem: The number of switch pairs in a Pascal's Triangle row is at most twice the number of odd entries --/
theorem switch_pairs_bound (n : ℕ) (row : PascalRow n) :
  countSwitchPairs row ≤ 2 * countOddEntries row := by sorry

end NUMINAMATH_CALUDE_switch_pairs_bound_l2593_259359


namespace NUMINAMATH_CALUDE_alice_age_l2593_259384

theorem alice_age (alice_age mother_age : ℕ) 
  (h1 : alice_age = mother_age - 18)
  (h2 : alice_age + mother_age = 50) : 
  alice_age = 16 := by sorry

end NUMINAMATH_CALUDE_alice_age_l2593_259384


namespace NUMINAMATH_CALUDE_quaternary_1320_to_binary_l2593_259315

/-- Converts a quaternary (base 4) number to decimal (base 10) --/
def quaternary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldl (λ sum (i, digit) => sum + digit * (4 ^ i)) 0

/-- Converts a decimal (base 10) number to binary (base 2) --/
def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec to_binary_aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else to_binary_aux (m / 2) ((m % 2) :: acc)
  to_binary_aux n []

/-- The main theorem stating that 1320₄ in binary is 1111000₂ --/
theorem quaternary_1320_to_binary :
  decimal_to_binary (quaternary_to_decimal [0, 2, 3, 1]) = [1, 1, 1, 1, 0, 0, 0] := by
  sorry


end NUMINAMATH_CALUDE_quaternary_1320_to_binary_l2593_259315


namespace NUMINAMATH_CALUDE_chef_leftover_potatoes_l2593_259362

theorem chef_leftover_potatoes 
  (fries_per_potato : ℕ) 
  (total_potatoes : ℕ) 
  (required_fries : ℕ) 
  (h1 : fries_per_potato = 25)
  (h2 : total_potatoes = 15)
  (h3 : required_fries = 200) :
  total_potatoes - (required_fries / fries_per_potato) = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_chef_leftover_potatoes_l2593_259362


namespace NUMINAMATH_CALUDE_diesel_consumption_calculation_l2593_259302

/-- Calculates the diesel consumption of a car given its fuel efficiency, travel time, and speed. -/
theorem diesel_consumption_calculation
  (fuel_efficiency : ℝ)  -- Diesel consumption in liters per kilometer
  (travel_time : ℝ)      -- Travel time in hours
  (speed : ℝ)            -- Speed in kilometers per hour
  (h1 : fuel_efficiency = 0.14)
  (h2 : travel_time = 2.5)
  (h3 : speed = 93.6) :
  fuel_efficiency * travel_time * speed = 32.76 := by
    sorry

#check diesel_consumption_calculation

end NUMINAMATH_CALUDE_diesel_consumption_calculation_l2593_259302


namespace NUMINAMATH_CALUDE_min_value_of_z_l2593_259388

theorem min_value_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) : 
  ∀ z : ℝ, z = x^2 + 4*y^2 → z ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_z_l2593_259388


namespace NUMINAMATH_CALUDE_odd_function_sum_l2593_259340

def f (x : ℝ) (b : ℝ) : ℝ := 2016 * x^3 - 5 * x + b + 2

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (a b : ℝ) :
  (∃ f : ℝ → ℝ, is_odd f ∧ (∀ x, f x = 2016 * x^3 - 5 * x + b + 2) ∧
   (∃ c d : ℝ, c = a - 4 ∧ d = 2 * a - 2 ∧ Set.Icc c d = Set.range f)) →
  f a + f b = 0 :=
sorry

end NUMINAMATH_CALUDE_odd_function_sum_l2593_259340


namespace NUMINAMATH_CALUDE_bag_pieces_problem_l2593_259342

theorem bag_pieces_problem (w b n : ℕ) : 
  b = 2 * w →                 -- The number of black pieces is twice the number of white pieces
  w - 2 * n = 1 →             -- After n rounds, 1 white piece is left
  b - 3 * n = 31 →            -- After n rounds, 31 black pieces are left
  b = 118 :=                  -- The initial number of black pieces was 118
by sorry

end NUMINAMATH_CALUDE_bag_pieces_problem_l2593_259342


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l2593_259398

/-- Represents the dimensions of a framed painting -/
structure FramedPainting where
  painting_width : ℝ
  painting_height : ℝ
  side_frame_width : ℝ

/-- Calculates the dimensions of the framed painting -/
def framedDimensions (fp : FramedPainting) : ℝ × ℝ :=
  (fp.painting_width + 2 * fp.side_frame_width, fp.painting_height + 4 * fp.side_frame_width)

/-- Calculates the area of the frame -/
def frameArea (fp : FramedPainting) : ℝ :=
  let (w, h) := framedDimensions fp
  w * h - fp.painting_width * fp.painting_height

/-- Theorem stating the ratio of dimensions for a specific framed painting -/
theorem framed_painting_ratio :
  ∃ (fp : FramedPainting),
    fp.painting_width = 15 ∧
    fp.painting_height = 30 ∧
    frameArea fp = fp.painting_width * fp.painting_height ∧
    let (w, h) := framedDimensions fp
    w / h = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l2593_259398


namespace NUMINAMATH_CALUDE_distribute_6_3_l2593_259366

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_6_3 : distribute 6 3 = 729 := by sorry

end NUMINAMATH_CALUDE_distribute_6_3_l2593_259366


namespace NUMINAMATH_CALUDE_fixed_points_equality_l2593_259326

def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def FixedPoints (f : ℝ → ℝ) : Set ℝ :=
  {x | f x = x}

theorem fixed_points_equality
  (f : ℝ → ℝ)
  (h_inj : Function.Injective f)
  (h_incr : StrictlyIncreasing f) :
  FixedPoints f = FixedPoints (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_fixed_points_equality_l2593_259326


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l2593_259365

theorem fractional_equation_solution_range (x m : ℝ) :
  (3 * x) / (x - 1) = m / (x - 1) + 2 →
  x ≥ 0 →
  x ≠ 1 →
  m ≥ 2 ∧ m ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l2593_259365


namespace NUMINAMATH_CALUDE_sacks_filled_twice_l2593_259354

/-- Represents the number of times sacks can be filled with wood --/
def times_sacks_filled (father_capacity : ℕ) (ranger_capacity : ℕ) (volunteer_capacity : ℕ) (num_volunteers : ℕ) (total_wood : ℕ) : ℕ :=
  total_wood / (father_capacity + ranger_capacity + num_volunteers * volunteer_capacity)

/-- Theorem stating that under the given conditions, the sacks can be filled 2 times --/
theorem sacks_filled_twice :
  times_sacks_filled 20 30 25 2 200 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sacks_filled_twice_l2593_259354


namespace NUMINAMATH_CALUDE_oil_price_reduction_l2593_259334

/-- Proves that the percentage reduction in oil price is 40% given the problem conditions -/
theorem oil_price_reduction (original_price reduced_price : ℝ) : 
  reduced_price = 120 →
  2400 / reduced_price - 2400 / original_price = 8 →
  (original_price - reduced_price) / original_price * 100 = 40 := by
  sorry

#check oil_price_reduction

end NUMINAMATH_CALUDE_oil_price_reduction_l2593_259334


namespace NUMINAMATH_CALUDE_wolf_and_nobel_count_l2593_259307

/-- Represents the number of scientists in various categories at a workshop --/
structure WorkshopAttendees where
  total : ℕ
  wolf : ℕ
  nobel : ℕ
  wolf_and_nobel : ℕ

/-- The conditions of the workshop --/
def workshop : WorkshopAttendees where
  total := 50
  wolf := 31
  nobel := 25
  wolf_and_nobel := 0  -- This is what we need to prove

/-- Theorem stating the number of scientists who were both Wolf and Nobel laureates --/
theorem wolf_and_nobel_count (w : WorkshopAttendees) 
  (h1 : w.total = 50)
  (h2 : w.wolf = 31)
  (h3 : w.nobel = 25)
  (h4 : w.nobel - w.wolf = 3 + (w.total - w.nobel - (w.wolf - w.wolf_and_nobel))) :
  w.wolf_and_nobel = 3 := by
  sorry

end NUMINAMATH_CALUDE_wolf_and_nobel_count_l2593_259307


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2593_259357

universe u

def U : Set ℕ := {2, 4, 6, 8, 9}
def A : Set ℕ := {2, 4, 9}

theorem complement_of_A_in_U :
  (U \ A) = {6, 8} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2593_259357


namespace NUMINAMATH_CALUDE_count_valid_concatenations_eq_825957_l2593_259399

def is_valid_integer (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 99

def concatenate (a b c : ℕ) : ℕ := sorry

def count_valid_concatenations : ℕ := sorry

theorem count_valid_concatenations_eq_825957 :
  count_valid_concatenations = 825957 := by sorry

end NUMINAMATH_CALUDE_count_valid_concatenations_eq_825957_l2593_259399


namespace NUMINAMATH_CALUDE_absolute_value_calculation_system_of_inequalities_l2593_259361

-- Part 1
theorem absolute_value_calculation : |(-2 : ℝ)| + Real.sqrt 4 - 2^(0 : ℕ) = 3 := by sorry

-- Part 2
theorem system_of_inequalities (x : ℝ) : 
  (2 * x < 6 ∧ 3 * x > -2 * x + 5) ↔ (1 < x ∧ x < 3) := by sorry

end NUMINAMATH_CALUDE_absolute_value_calculation_system_of_inequalities_l2593_259361


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2593_259389

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 → 
  (∃ m : ℕ, m > 12 ∧ ∀ k : ℕ, k > 0 → m ∣ (k * (k + 1) * (k + 2) * (k + 3))) → False :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2593_259389


namespace NUMINAMATH_CALUDE_charts_brought_is_eleven_l2593_259300

/-- The number of charts brought to a committee meeting --/
def charts_brought (associate_profs assistant_profs : ℕ) : ℕ :=
  associate_profs + 2 * assistant_profs

/-- Proof that 11 charts were brought to the meeting --/
theorem charts_brought_is_eleven :
  ∃ (associate_profs assistant_profs : ℕ),
    associate_profs + assistant_profs = 7 ∧
    2 * associate_profs + assistant_profs = 10 ∧
    charts_brought associate_profs assistant_profs = 11 :=
by
  sorry

#check charts_brought_is_eleven

end NUMINAMATH_CALUDE_charts_brought_is_eleven_l2593_259300


namespace NUMINAMATH_CALUDE_quarters_remaining_l2593_259377

-- Define the initial number of quarters
def initial_quarters : ℕ := 375

-- Define the cost of the dress in cents
def dress_cost_cents : ℕ := 4263

-- Define the value of a quarter in cents
def quarter_value_cents : ℕ := 25

-- Theorem to prove
theorem quarters_remaining :
  initial_quarters - (dress_cost_cents / quarter_value_cents) = 205 := by
  sorry

end NUMINAMATH_CALUDE_quarters_remaining_l2593_259377


namespace NUMINAMATH_CALUDE_parabola_focus_and_tangent_point_l2593_259329

noncomputable section

/-- Parabola C with parameter p -/
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y

/-- Line passing through the focus -/
def focus_line (x y : ℝ) : Prop := x + 2*y - 2 = 0

/-- Directrix of the parabola -/
def directrix (p : ℝ) (y : ℝ) : Prop := y = -p/2

/-- Tangent line to the parabola from point (m, -p/2) -/
def tangent_line (p m k x y : ℝ) : Prop := y = -p/2 + k*(x - m)

/-- Area of triangle AMN -/
def triangle_area (m : ℝ) : ℝ := (1/2) * Real.sqrt (m^2 + 4)

theorem parabola_focus_and_tangent_point (p : ℝ) :
  p > 0 →
  (∃ x y : ℝ, parabola p x y ∧ focus_line x y) →
  (∃ m : ℝ, directrix p (-p/2) ∧ triangle_area m = Real.sqrt 5 / 2) →
  (∃ m : ℝ, m = 1 ∨ m = -1) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_and_tangent_point_l2593_259329


namespace NUMINAMATH_CALUDE_lg_sum_equals_three_l2593_259341

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_three : lg 8 + 3 * lg 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lg_sum_equals_three_l2593_259341


namespace NUMINAMATH_CALUDE_midpoint_implies_equation_ratio_implies_equation_l2593_259347

/-- A line passing through point M(-2,1) and intersecting x and y axes at A and B respectively -/
structure Line :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (h_A : A.2 = 0)
  (h_B : B.1 = 0)

/-- The point M(-2,1) -/
def M : ℝ × ℝ := (-2, 1)

/-- M is the midpoint of AB -/
def is_midpoint (l : Line) : Prop :=
  M = ((l.A.1 + l.B.1) / 2, (l.A.2 + l.B.2) / 2)

/-- M divides AB in the ratio of 2:1 or 1:2 -/
def divides_in_ratio (l : Line) : Prop :=
  (M.1 - l.A.1, M.2 - l.A.2) = (2 * (l.B.1 - M.1), 2 * (l.B.2 - M.2)) ∨
  (M.1 - l.A.1, M.2 - l.A.2) = (-2 * (l.B.1 - M.1), -2 * (l.B.2 - M.2))

/-- The equation of the line in the form ax + by + c = 0 -/
structure LineEquation :=
  (a b c : ℝ)

theorem midpoint_implies_equation (l : Line) (h : is_midpoint l) :
  ∃ (eq : LineEquation), eq.a * l.A.1 + eq.b * l.A.2 + eq.c = 0 ∧
                         eq.a * l.B.1 + eq.b * l.B.2 + eq.c = 0 ∧
                         eq.a * M.1 + eq.b * M.2 + eq.c = 0 ∧
                         eq.a = 1 ∧ eq.b = -2 ∧ eq.c = 4 := by sorry

theorem ratio_implies_equation (l : Line) (h : divides_in_ratio l) :
  ∃ (eq1 eq2 : LineEquation),
    (eq1.a * l.A.1 + eq1.b * l.A.2 + eq1.c = 0 ∧
     eq1.a * l.B.1 + eq1.b * l.B.2 + eq1.c = 0 ∧
     eq1.a * M.1 + eq1.b * M.2 + eq1.c = 0 ∧
     eq1.a = 1 ∧ eq1.b = -4 ∧ eq1.c = 6) ∨
    (eq2.a * l.A.1 + eq2.b * l.A.2 + eq2.c = 0 ∧
     eq2.a * l.B.1 + eq2.b * l.B.2 + eq2.c = 0 ∧
     eq2.a * M.1 + eq2.b * M.2 + eq2.c = 0 ∧
     eq2.a = 1 ∧ eq2.b = 4 ∧ eq2.c = -2) := by sorry

end NUMINAMATH_CALUDE_midpoint_implies_equation_ratio_implies_equation_l2593_259347


namespace NUMINAMATH_CALUDE_second_smallest_perimeter_l2593_259303

/-- A triangle with consecutive integer side lengths -/
structure ConsecutiveIntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  consecutive : b = a + 1 ∧ c = b + 1
  is_triangle : a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of a triangle -/
def perimeter (t : ConsecutiveIntegerTriangle) : ℕ :=
  t.a + t.b + t.c

/-- The second smallest perimeter of a triangle with consecutive integer side lengths is 12 -/
theorem second_smallest_perimeter :
  ∃ (t : ConsecutiveIntegerTriangle), perimeter t = 12 ∧
  ∀ (s : ConsecutiveIntegerTriangle), perimeter s ≠ 9 → perimeter s ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_second_smallest_perimeter_l2593_259303


namespace NUMINAMATH_CALUDE_positive_real_inequalities_l2593_259378

theorem positive_real_inequalities (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a * b + a + b + 1) * (a * b + a * c + b * c + c^2) ≥ 16 * a * b * c) ∧
  ((b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequalities_l2593_259378


namespace NUMINAMATH_CALUDE_distance_incenter_circumcenter_squared_l2593_259372

-- Define a 30-60-90 right triangle with hypotenuse 2
structure Triangle30_60_90 where
  hypotenuse : ℝ
  is_30_60_90 : hypotenuse = 2

-- Define the distance between incenter and circumcenter
def distance_incenter_circumcenter (t : Triangle30_60_90) : ℝ := sorry

theorem distance_incenter_circumcenter_squared (t : Triangle30_60_90) :
  (distance_incenter_circumcenter t)^2 = 2 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_distance_incenter_circumcenter_squared_l2593_259372


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l2593_259386

/-- Given a circle C with equation x^2 - 8y - 5 = -y^2 - 6x, 
    prove that a + b + r = 1 + √30, 
    where (a,b) is the center of C and r is its radius. -/
theorem circle_center_radius_sum (x y : ℝ) :
  let C : Set (ℝ × ℝ) := {(x, y) | x^2 - 8*y - 5 = -y^2 - 6*x}
  ∃ (a b r : ℝ), (∀ (x y : ℝ), (x, y) ∈ C ↔ (x - a)^2 + (y - b)^2 = r^2) ∧
                 (a + b + r = 1 + Real.sqrt 30) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l2593_259386


namespace NUMINAMATH_CALUDE_greatest_consecutive_mixed_number_l2593_259310

/-- 
Given 6 consecutive mixed numbers with a sum of 75.5, 
prove that the greatest number is 15 1/12.
-/
theorem greatest_consecutive_mixed_number :
  ∀ (a b c d e f : ℚ),
    a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f →  -- consecutive
    b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ f = a + 5 →  -- mixed numbers
    a + b + c + d + e + f = 75.5 →  -- sum condition
    f = 15 + 1/12 :=  -- greatest number
by sorry

end NUMINAMATH_CALUDE_greatest_consecutive_mixed_number_l2593_259310


namespace NUMINAMATH_CALUDE_prob_at_least_two_tails_is_half_l2593_259364

/-- The probability of getting at least 2 tails when tossing 3 fair coins -/
def prob_at_least_two_tails : ℚ := 1/2

/-- The number of possible outcomes when tossing 3 coins -/
def total_outcomes : ℕ := 2^3

/-- The number of favorable outcomes (at least 2 tails) -/
def favorable_outcomes : ℕ := 4

theorem prob_at_least_two_tails_is_half :
  prob_at_least_two_tails = (favorable_outcomes : ℚ) / total_outcomes :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_tails_is_half_l2593_259364


namespace NUMINAMATH_CALUDE_fathers_age_l2593_259306

theorem fathers_age (man_age father_age : ℚ) : 
  man_age = (2 / 5) * father_age →
  man_age + 10 = (1 / 2) * (father_age + 10) →
  father_age = 50 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l2593_259306


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2593_259317

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 - 7 * p - 6 = 0) → 
  (3 * q^2 - 7 * q - 6 = 0) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2593_259317


namespace NUMINAMATH_CALUDE_worker_weekly_pay_l2593_259393

/-- Worker's weekly pay calculation --/
theorem worker_weekly_pay (regular_rate : ℝ) (total_surveys : ℕ) (cellphone_rate_increase : ℝ) (cellphone_surveys : ℕ) :
  regular_rate = 10 →
  total_surveys = 100 →
  cellphone_rate_increase = 0.3 →
  cellphone_surveys = 60 →
  let non_cellphone_surveys := total_surveys - cellphone_surveys
  let cellphone_rate := regular_rate * (1 + cellphone_rate_increase)
  let non_cellphone_pay := non_cellphone_surveys * regular_rate
  let cellphone_pay := cellphone_surveys * cellphone_rate
  let total_pay := non_cellphone_pay + cellphone_pay
  total_pay = 1180 := by
sorry

end NUMINAMATH_CALUDE_worker_weekly_pay_l2593_259393


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_complementary_l2593_259385

-- Define the set of cards
inductive Card : Type
  | Hearts | Spades | Diamonds | Clubs

-- Define the set of people
inductive Person : Type
  | A | B | C | D

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "Person A gets a club"
def event_A_club (d : Distribution) : Prop := d Person.A = Card.Clubs

-- Define the event "Person B gets a club"
def event_B_club (d : Distribution) : Prop := d Person.B = Card.Clubs

-- Theorem statement
theorem mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(event_A_club d ∧ event_B_club d)) ∧
  (∃ d : Distribution, ¬event_A_club d ∧ ¬event_B_club d) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_complementary_l2593_259385


namespace NUMINAMATH_CALUDE_binomial_2023_2_l2593_259324

theorem binomial_2023_2 : Nat.choose 2023 2 = 2045323 := by
  sorry

end NUMINAMATH_CALUDE_binomial_2023_2_l2593_259324


namespace NUMINAMATH_CALUDE_father_age_and_pen_cost_l2593_259391

/-- Xiao Ming's age -/
def xiao_ming_age : ℕ := 9

/-- The factor by which Xiao Ming's father's age is greater than Xiao Ming's -/
def father_age_factor : ℕ := 5

/-- The cost of one pen in yuan -/
def pen_cost : ℕ := 2

/-- The number of pens to be purchased -/
def pen_quantity : ℕ := 60

theorem father_age_and_pen_cost :
  (xiao_ming_age * father_age_factor = 45) ∧
  (pen_cost * pen_quantity = 120) := by
  sorry


end NUMINAMATH_CALUDE_father_age_and_pen_cost_l2593_259391


namespace NUMINAMATH_CALUDE_event_X_6_equivalent_to_draw_6_and_two_others_l2593_259337

/-- Represents a ball with a number -/
structure Ball :=
  (number : Nat)

/-- The set of all balls in the bag -/
def bag : Finset Ball := sorry

/-- The number of balls to be drawn -/
def numDrawn : Nat := 3

/-- X represents the highest number on the drawn balls -/
def X (drawn : Finset Ball) : Nat := sorry

/-- The event where X equals 6 -/
def event_X_equals_6 (drawn : Finset Ball) : Prop :=
  X drawn = 6

/-- The event of drawing 3 balls with one numbered 6 and two others from 1 to 5 -/
def event_draw_6_and_two_others (drawn : Finset Ball) : Prop := sorry

theorem event_X_6_equivalent_to_draw_6_and_two_others :
  ∀ drawn : Finset Ball,
  drawn.card = numDrawn →
  (event_X_equals_6 drawn ↔ event_draw_6_and_two_others drawn) :=
sorry

end NUMINAMATH_CALUDE_event_X_6_equivalent_to_draw_6_and_two_others_l2593_259337


namespace NUMINAMATH_CALUDE_congruence_solution_l2593_259351

theorem congruence_solution (n : Int) : n ≡ 26 [ZMOD 47] ↔ 13 * n ≡ 9 [ZMOD 47] ∧ 0 ≤ n ∧ n < 47 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2593_259351


namespace NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l2593_259305

theorem power_of_eight_sum_equals_power_of_two : 8^18 + 8^18 + 8^18 = 2^56 := by
  sorry

end NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l2593_259305


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2593_259308

theorem arithmetic_expression_equality : 5 + 2 * (8 - 3) = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2593_259308


namespace NUMINAMATH_CALUDE_strawberry_problem_l2593_259348

theorem strawberry_problem (initial : Float) (eaten : Float) (remaining : Float) : 
  initial = 78.0 → eaten = 42.0 → remaining = initial - eaten → remaining = 36.0 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_problem_l2593_259348


namespace NUMINAMATH_CALUDE_problem_solution_l2593_259397

theorem problem_solution (x y : ℝ) (h : x^2 + y^2 = 12*x - 4*y - 40) :
  x * Real.cos (-23/3 * Real.pi) + y * Real.tan (-15/4 * Real.pi) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2593_259397


namespace NUMINAMATH_CALUDE_min_value_zero_l2593_259314

/-- The quadratic form representing the expression -/
def Q (x y : ℝ) : ℝ := 5 * x^2 - 8 * x * y + 7 * y^2 - 6 * x - 6 * y + 9

/-- The theorem stating that the minimum value of Q is 0 -/
theorem min_value_zero : 
  ∀ x y : ℝ, Q x y ≥ 0 ∧ ∃ x₀ y₀ : ℝ, Q x₀ y₀ = 0 := by sorry

end NUMINAMATH_CALUDE_min_value_zero_l2593_259314


namespace NUMINAMATH_CALUDE_max_value_x_minus_2y_l2593_259355

theorem max_value_x_minus_2y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + 2*y) :
  ∃ (max : ℝ), max = 2/3 ∧ x - 2*y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_x_minus_2y_l2593_259355


namespace NUMINAMATH_CALUDE_julia_dimes_count_l2593_259311

theorem julia_dimes_count : ∃ d : ℕ, 
  20 < d ∧ d < 200 ∧ 
  d % 6 = 1 ∧ 
  d % 7 = 1 ∧ 
  d % 8 = 1 ∧ 
  d = 169 := by sorry

end NUMINAMATH_CALUDE_julia_dimes_count_l2593_259311


namespace NUMINAMATH_CALUDE_emelya_balls_count_l2593_259363

def total_balls : ℕ := 10
def broken_balls : ℕ := 3
def lost_balls : ℕ := 3

theorem emelya_balls_count :
  ∀ (M : ℝ),
  M > 0 →
  (broken_balls : ℝ) * M * (35/100) = (7/20) * M →
  ∃ (remaining_balls : ℕ),
  remaining_balls > 0 ∧
  (remaining_balls : ℝ) * M * (8/13) = (2/5) * M ∧
  total_balls = remaining_balls + broken_balls + lost_balls :=
by sorry

end NUMINAMATH_CALUDE_emelya_balls_count_l2593_259363


namespace NUMINAMATH_CALUDE_johnson_family_seating_l2593_259371

def num_sons : ℕ := 5
def num_daughters : ℕ := 4
def total_children : ℕ := num_sons + num_daughters

def total_arrangements : ℕ := Nat.factorial total_children

def arrangements_without_bbg : ℕ := Nat.factorial 7 * 4

theorem johnson_family_seating :
  total_arrangements - arrangements_without_bbg = 342720 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l2593_259371


namespace NUMINAMATH_CALUDE_substitution_ways_mod_1000_l2593_259369

/-- Represents the number of players in a soccer team --/
def total_players : ℕ := 22

/-- Represents the number of starting players --/
def starting_players : ℕ := 11

/-- Represents the maximum number of substitutions allowed --/
def max_substitutions : ℕ := 4

/-- Calculates the number of ways to make substitutions in a soccer game --/
def substitution_ways : ℕ := 
  1 + 
  (starting_players * starting_players) + 
  (starting_players^3 * (starting_players - 1)) + 
  (starting_players^5 * (starting_players - 1) * (starting_players - 2)) + 
  (starting_players^7 * (starting_players - 1) * (starting_players - 2) * (starting_players - 3))

/-- Theorem stating that the number of substitution ways modulo 1000 is 712 --/
theorem substitution_ways_mod_1000 : 
  substitution_ways % 1000 = 712 := by sorry

end NUMINAMATH_CALUDE_substitution_ways_mod_1000_l2593_259369


namespace NUMINAMATH_CALUDE_bertha_family_childless_l2593_259343

/-- Represents the family structure of Bertha and her descendants -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  daughters_with_children : ℕ

/-- The properties of Bertha's family -/
def bertha_family_properties (f : BerthaFamily) : Prop :=
  f.daughters = 6 ∧
  f.granddaughters = 6 * f.daughters_with_children ∧
  f.daughters + f.granddaughters = 30

/-- The theorem stating the number of Bertha's daughters and granddaughters without children -/
theorem bertha_family_childless (f : BerthaFamily) 
  (h : bertha_family_properties f) : 
  f.daughters + f.granddaughters - f.daughters_with_children = 26 := by
  sorry


end NUMINAMATH_CALUDE_bertha_family_childless_l2593_259343


namespace NUMINAMATH_CALUDE_power_three_nineteen_mod_ten_l2593_259373

theorem power_three_nineteen_mod_ten : 3^19 % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_power_three_nineteen_mod_ten_l2593_259373


namespace NUMINAMATH_CALUDE_jeans_to_shirt_cost_ratio_l2593_259332

/-- The ratio of the cost of a pair of jeans to the cost of a shirt is 2:1 -/
theorem jeans_to_shirt_cost_ratio :
  ∀ (jeans_cost : ℚ),
  20 * 10 + 10 * jeans_cost = 400 →
  jeans_cost / 10 = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_jeans_to_shirt_cost_ratio_l2593_259332


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_in_special_pentagon_l2593_259323

/-- A pentagon with two interior lines -/
structure PentagonWithInteriorLines where
  -- Exterior angles
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ
  -- Interior angles formed by the lines
  angle_x : ℝ
  angle_y : ℝ

/-- Theorem: Sum of interior angles in special pentagon configuration -/
theorem sum_of_interior_angles_in_special_pentagon
  (p : PentagonWithInteriorLines)
  (h_A : p.angle_A = 35)
  (h_B : p.angle_B = 65)
  (h_C : p.angle_C = 40) :
  p.angle_x + p.angle_y = 140 := by
  sorry

#check sum_of_interior_angles_in_special_pentagon

end NUMINAMATH_CALUDE_sum_of_interior_angles_in_special_pentagon_l2593_259323


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_9_l2593_259330

def is_divisible_by_9 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 9 * k

def digit_sum (a : ℕ) : ℕ :=
  3 + a + a + 1

theorem four_digit_divisible_by_9 :
  ∃ A : ℕ, A < 10 ∧ is_divisible_by_9 (3000 + 100 * A + 10 * A + 1) ∧ A = 7 :=
sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_9_l2593_259330


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l2593_259331

/-- The minimum distance between curves C₁ and C₂ is 0 -/
theorem min_distance_between_curves (x y : ℝ) : 
  let C₁ := {(x, y) | x^2/8 + y^2/4 = 1}
  let C₂ := {(x, y) | x - Real.sqrt 2 * y - 4 = 0}
  ∃ (p q : ℝ × ℝ), p ∈ C₁ ∧ q ∈ C₂ ∧ 
    ∀ (p' q' : ℝ × ℝ), p' ∈ C₁ → q' ∈ C₂ → 
      Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) ≥ 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ∧
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l2593_259331


namespace NUMINAMATH_CALUDE_sqrt_equation_equivalence_l2593_259375

theorem sqrt_equation_equivalence (x : ℝ) (h : x > 9) :
  (Real.sqrt (x - 9 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 9 * Real.sqrt (x - 9)) - 3) ↔ 
  x ≥ 40.5 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_equivalence_l2593_259375


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l2593_259353

theorem triangle_side_calculation (a b c : ℝ) (A B C : ℝ) :
  2 * Real.sin (2 * B + π / 6) = 2 →
  a * c = 3 * Real.sqrt 3 →
  a + c = 4 →
  b ^ 2 = 16 - 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l2593_259353


namespace NUMINAMATH_CALUDE_range_of_M_l2593_259339

theorem range_of_M (x y : ℝ) (h : x^2 + x*y + y^2 = 2) : 
  let M := x^2 - x*y + y^2
  2/3 ≤ M ∧ M ≤ 6 := by
sorry

end NUMINAMATH_CALUDE_range_of_M_l2593_259339


namespace NUMINAMATH_CALUDE_inequality_multiplication_l2593_259370

theorem inequality_multiplication (x y : ℝ) : y > x → 2 * y > 2 * x := by
  sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l2593_259370


namespace NUMINAMATH_CALUDE_share_ratio_l2593_259352

/-- Represents the shares of money for three individuals -/
structure Shares where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The problem setup -/
def problem_setup (s : Shares) : Prop :=
  s.a = 80 ∧                          -- a's share is $80
  s.a + s.b + s.c = 200 ∧             -- Total amount is $200
  s.a = (2/3) * (s.b + s.c) ∧         -- a gets 2/3 as much as b and c together
  ∃ x, s.b = x * (s.a + s.c)          -- b gets some fraction of a and c together

/-- The theorem to be proved -/
theorem share_ratio (s : Shares) (h : problem_setup s) : 
  s.b / (s.a + s.c) = 2/3 := by sorry

end NUMINAMATH_CALUDE_share_ratio_l2593_259352


namespace NUMINAMATH_CALUDE_proposition_and_converse_l2593_259319

theorem proposition_and_converse (a b : ℝ) : 
  (((a + b ≥ 2) → (a ≥ 1 ∨ b ≥ 1)) ∧ 
  (∃ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) ∧ ¬(a + b ≥ 2))) :=
by sorry

end NUMINAMATH_CALUDE_proposition_and_converse_l2593_259319


namespace NUMINAMATH_CALUDE_fraction_sum_difference_l2593_259327

theorem fraction_sum_difference (a b c d e f : ℤ) :
  (a : ℚ) / b + (c : ℚ) / d - (e : ℚ) / f = (53 : ℚ) / 72 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_fraction_sum_difference_l2593_259327


namespace NUMINAMATH_CALUDE_parallelogram_area_triangle_area_l2593_259358

-- Define the parallelogram
def parallelogram_base : ℝ := 16
def parallelogram_height : ℝ := 25

-- Define the right-angled triangle
def triangle_side1 : ℝ := 3
def triangle_side2 : ℝ := 4

-- Theorem for parallelogram area
theorem parallelogram_area : 
  parallelogram_base * parallelogram_height = 400 := by sorry

-- Theorem for right-angled triangle area
theorem triangle_area : 
  (triangle_side1 * triangle_side2) / 2 = 6 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_triangle_area_l2593_259358


namespace NUMINAMATH_CALUDE_equal_group_formations_20_people_l2593_259312

/-- The number of ways to form a group with an equal number of boys and girls -/
def equalGroupFormations (totalPeople boys girls : ℕ) : ℕ :=
  Nat.choose totalPeople boys

/-- Theorem stating that the number of ways to form a group with an equal number
    of boys and girls from 20 people (10 boys and 10 girls) is equal to C(20,10) -/
theorem equal_group_formations_20_people :
  equalGroupFormations 20 10 10 = Nat.choose 20 10 := by
  sorry

#eval equalGroupFormations 20 10 10

end NUMINAMATH_CALUDE_equal_group_formations_20_people_l2593_259312


namespace NUMINAMATH_CALUDE_expression_value_l2593_259394

theorem expression_value (x : ℝ) (h : 2 * x^2 - x - 1 = 5) : 6 * x^2 - 3 * x - 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2593_259394


namespace NUMINAMATH_CALUDE_function_satisfying_property_is_square_l2593_259322

open Real

-- Define the property for the function
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = ⨆ y : ℝ, (2 * x * y - f y)

-- Theorem statement
theorem function_satisfying_property_is_square (f : ℝ → ℝ) :
  SatisfiesProperty f → ∀ x : ℝ, f x = x^2 := by
  sorry


end NUMINAMATH_CALUDE_function_satisfying_property_is_square_l2593_259322


namespace NUMINAMATH_CALUDE_largest_non_prime_sequence_l2593_259316

theorem largest_non_prime_sequence : ∃ (a : ℕ), 
  (a ≥ 10 ∧ a + 6 ≤ 50) ∧ 
  (∀ i ∈ (Finset.range 7), ¬ Nat.Prime (a + i)) ∧
  (∀ b : ℕ, b > a + 6 → 
    ¬(b ≥ 10 ∧ b + 6 ≤ 50 ∧ 
      (∀ i ∈ (Finset.range 7), ¬ Nat.Prime (b + i)))) :=
by sorry

end NUMINAMATH_CALUDE_largest_non_prime_sequence_l2593_259316


namespace NUMINAMATH_CALUDE_b_range_for_inequality_l2593_259383

/-- Given an inequality ax + b > 2(x + 1) with solution set {x | x < 1}, 
    prove that the range of values for b is (4, +∞) -/
theorem b_range_for_inequality (a b : ℝ) : 
  (∀ x, ax + b > 2*(x + 1) ↔ x < 1) → 
  ∃ y, y > 4 ∧ b > y :=
sorry

end NUMINAMATH_CALUDE_b_range_for_inequality_l2593_259383
