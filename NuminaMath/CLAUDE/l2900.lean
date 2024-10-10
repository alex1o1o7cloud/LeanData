import Mathlib

namespace pirate_coin_distribution_l2900_290007

/-- The number of rounds in the coin distribution process -/
def y : ℕ := sorry

/-- The total number of coins Pete has after distribution -/
def peteCoins : ℕ := y * (y + 1) / 2

/-- The total number of coins Paul has after distribution -/
def paulCoins : ℕ := y

/-- The ratio of Pete's coins to Paul's coins -/
def coinRatio : ℕ := 5

theorem pirate_coin_distribution :
  peteCoins = coinRatio * paulCoins ∧ peteCoins + paulCoins = 54 := by
  sorry

end pirate_coin_distribution_l2900_290007


namespace bella_galya_distance_l2900_290013

/-- The distance between two houses -/
def distance (house1 house2 : ℕ) : ℕ := sorry

/-- The order of houses along the road -/
def house_order : List String := ["Alya", "Bella", "Valya", "Galya", "Dilya"]

/-- The total distance from a house to all other houses -/
def total_distance (house : String) : ℕ := sorry

theorem bella_galya_distance :
  distance 1 3 = 150 ∧
  house_order = ["Alya", "Bella", "Valya", "Galya", "Dilya"] ∧
  total_distance "Bella" = 700 ∧
  total_distance "Valya" = 600 ∧
  total_distance "Galya" = 650 :=
by sorry

end bella_galya_distance_l2900_290013


namespace blue_tetrahedron_volume_l2900_290096

/-- The volume of the tetrahedron formed by alternating vertices of a cube -/
theorem blue_tetrahedron_volume (cube_side_length : ℝ) (h : cube_side_length = 8) :
  let cube_volume := cube_side_length ^ 3
  let blue_tetrahedron_volume := cube_volume / 3
  blue_tetrahedron_volume = 512 / 3 := by
  sorry

end blue_tetrahedron_volume_l2900_290096


namespace maria_candy_eaten_l2900_290043

/-- Given that Maria initially had 67 pieces of candy and now has 3 pieces,
    prove that she ate 64 pieces of candy. -/
theorem maria_candy_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 67 → remaining = 3 → eaten = initial - remaining → eaten = 64 := by
  sorry

end maria_candy_eaten_l2900_290043


namespace candidate_vote_difference_l2900_290073

theorem candidate_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 8200 →
  candidate_percentage = 35/100 →
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 2460 := by
  sorry

end candidate_vote_difference_l2900_290073


namespace second_term_value_l2900_290058

theorem second_term_value (A B : ℝ) (h1 : A > 0) (h2 : B > 0) 
  (h3 : A / B = 3 / 4) (h4 : (A + 10) / (B + 10) = 4 / 5) : B = 40 := by
  sorry

end second_term_value_l2900_290058


namespace kittens_per_female_cat_l2900_290068

theorem kittens_per_female_cat 
  (total_adult_cats : ℕ)
  (female_ratio : ℚ)
  (sold_kittens : ℕ)
  (kitten_ratio_after_sale : ℚ)
  (h1 : total_adult_cats = 6)
  (h2 : female_ratio = 1/2)
  (h3 : sold_kittens = 9)
  (h4 : kitten_ratio_after_sale = 67/100) :
  ∃ (kittens_per_female : ℕ),
    kittens_per_female = 7 ∧
    (female_ratio * total_adult_cats : ℚ) * kittens_per_female = 
      (1 - kitten_ratio_after_sale) * 
        ((total_adult_cats : ℚ) / (1 - kitten_ratio_after_sale) - total_adult_cats) +
      sold_kittens :=
by
  sorry

end kittens_per_female_cat_l2900_290068


namespace estimate_at_25_l2900_290005

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the estimated y value for a given x on a regression line -/
def estimate_y (line : RegressionLine) (x : ℝ) : ℝ :=
  line.slope * x + line.intercept

/-- The specific regression line y = 0.5x - 0.81 -/
def specific_line : RegressionLine :=
  { slope := 0.5, intercept := -0.81 }

/-- Theorem: The estimated y value when x = 25 on the specific regression line is 11.69 -/
theorem estimate_at_25 :
  estimate_y specific_line 25 = 11.69 := by sorry

end estimate_at_25_l2900_290005


namespace book_has_120_pages_l2900_290067

/-- Represents a book reading plan. -/
structure ReadingPlan where
  pagesPerNight : ℕ
  totalDays : ℕ

/-- Calculates the total number of pages in a book given a reading plan. -/
def totalPages (plan : ReadingPlan) : ℕ :=
  plan.pagesPerNight * plan.totalDays

/-- Theorem stating that the book has 120 pages given the specified reading plan. -/
theorem book_has_120_pages :
  ∃ (plan : ReadingPlan),
    plan.pagesPerNight = 12 ∧
    plan.totalDays = 10 ∧
    totalPages plan = 120 := by
  sorry


end book_has_120_pages_l2900_290067


namespace least_positive_integer_modulo_solution_satisfies_congruence_twelve_is_least_positive_solution_l2900_290017

theorem least_positive_integer_modulo (x : ℕ) : x + 3001 ≡ 1723 [ZMOD 15] → x ≥ 12 := by
  sorry

theorem solution_satisfies_congruence : 12 + 3001 ≡ 1723 [ZMOD 15] := by
  sorry

theorem twelve_is_least_positive_solution : ∃! x : ℕ, x + 3001 ≡ 1723 [ZMOD 15] ∧ ∀ y : ℕ, y + 3001 ≡ 1723 [ZMOD 15] → x ≤ y := by
  sorry

end least_positive_integer_modulo_solution_satisfies_congruence_twelve_is_least_positive_solution_l2900_290017


namespace polyhedron_volume_l2900_290028

/-- The volume of a polyhedron composed of a regular quadrilateral prism and two regular quadrilateral pyramids -/
theorem polyhedron_volume (prism_volume pyramid_volume : ℝ) 
  (h_prism : prism_volume = Real.sqrt 2 - 1)
  (h_pyramid : pyramid_volume = 1 / 6) :
  prism_volume + 2 * pyramid_volume = Real.sqrt 2 - 2 / 3 := by
  sorry

end polyhedron_volume_l2900_290028


namespace solution_set_not_three_elements_l2900_290052

noncomputable section

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a^(|x - b|)

-- Theorem statement
theorem solution_set_not_three_elements
  (a b m n p : ℝ)
  (ha : a > 0)
  (ha_neq : a ≠ 1)
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (hp : p ≠ 0) :
  ¬ ∃ (x y z : ℝ),
    (x ≠ y ∧ x ≠ z ∧ y ≠ z) ∧
    (∀ w, m * (f a b w)^2 + n * (f a b w) + p = 0 ↔ w = x ∨ w = y ∨ w = z) :=
sorry

end

end solution_set_not_three_elements_l2900_290052


namespace total_swordfish_catch_l2900_290035

/-- The number of times Shelly and Sam go fishing -/
def fishing_trips : ℕ := 5

/-- The number of swordfish Shelly catches each time -/
def shelly_catch : ℕ := 5 - 2

/-- The number of swordfish Sam catches each time -/
def sam_catch : ℕ := shelly_catch - 1

/-- The total number of swordfish caught by Shelly and Sam after their fishing trips -/
def total_catch : ℕ := fishing_trips * (shelly_catch + sam_catch)

theorem total_swordfish_catch : total_catch = 25 := by
  sorry

end total_swordfish_catch_l2900_290035


namespace hyperbola_asymptote_slope_l2900_290086

/-- The value of m for a hyperbola with given equation and asymptote form -/
theorem hyperbola_asymptote_slope (x y : ℝ) :
  (y^2 / 16 - x^2 / 9 = 1) →
  (∃ (m : ℝ), ∀ (x y : ℝ), y = m * x ∨ y = -m * x) →
  (∃ (m : ℝ), m = 4/3 ∧ (∀ (x y : ℝ), y = m * x ∨ y = -m * x)) :=
by sorry

end hyperbola_asymptote_slope_l2900_290086


namespace inequality_equivalence_l2900_290027

theorem inequality_equivalence (y : ℝ) : 
  (7 / 30 + |y - 19 / 60| < 17 / 30) ↔ (-1 / 60 < y ∧ y < 13 / 20) := by
  sorry

end inequality_equivalence_l2900_290027


namespace cloth_trimming_l2900_290087

theorem cloth_trimming (x : ℝ) :
  x > 0 →
  (x - 6) * (x - 5) = 120 →
  x = 15 :=
by sorry

end cloth_trimming_l2900_290087


namespace pin_purchase_cost_l2900_290054

/-- The total cost of pins with a discount -/
def total_cost (num_pins : ℕ) (regular_price : ℚ) (discount_percent : ℚ) : ℚ :=
  num_pins * (regular_price * (1 - discount_percent / 100))

/-- Theorem stating the total cost of 10 pins with a 15% discount -/
theorem pin_purchase_cost :
  total_cost 10 20 15 = 170 := by
  sorry

end pin_purchase_cost_l2900_290054


namespace hibiscus_flower_ratio_l2900_290094

/-- Given Mario's hibiscus plants, prove the ratio of flowers on the third to second plant -/
theorem hibiscus_flower_ratio :
  let first_plant_flowers : ℕ := 2
  let second_plant_flowers : ℕ := 2 * first_plant_flowers
  let total_flowers : ℕ := 22
  let third_plant_flowers : ℕ := total_flowers - first_plant_flowers - second_plant_flowers
  third_plant_flowers / second_plant_flowers = 4 := by
sorry

end hibiscus_flower_ratio_l2900_290094


namespace quadratic_roots_relation_l2900_290076

-- Define the coefficients of the original quadratic equation
def a : ℝ := 3
def b : ℝ := 4
def c : ℝ := 2

-- Define the roots of the original quadratic equation
def α : ℝ := sorry
def β : ℝ := sorry

-- Define the coefficients of the new quadratic equation
def a' : ℝ := 4
def p : ℝ := sorry
def q : ℝ := sorry

-- State the theorem
theorem quadratic_roots_relation :
  (3 * α^2 + 4 * α + 2 = 0) ∧
  (3 * β^2 + 4 * β + 2 = 0) ∧
  (4 * (2*α + 1)^2 + p * (2*α + 1) + q = 0) ∧
  (4 * (2*β + 1)^2 + p * (2*β + 1) + q = 0) →
  p = 8/3 := by sorry

end quadratic_roots_relation_l2900_290076


namespace water_evaporation_per_day_l2900_290045

/-- Given a glass of water with initial amount, evaporation period, and total evaporation percentage,
    calculate the amount of water evaporated per day. -/
theorem water_evaporation_per_day 
  (initial_amount : ℝ) 
  (evaporation_period : ℕ) 
  (evaporation_percentage : ℝ) 
  (h1 : initial_amount = 10)
  (h2 : evaporation_period = 20)
  (h3 : evaporation_percentage = 4) : 
  (initial_amount * evaporation_percentage / 100) / evaporation_period = 0.02 := by
  sorry

#check water_evaporation_per_day

end water_evaporation_per_day_l2900_290045


namespace line_angle_theorem_l2900_290089

/-- Given a line with equation (√6 sin θ)x + √3y - 2 = 0 and oblique angle θ ≠ 0, prove θ = 3π/4 -/
theorem line_angle_theorem (θ : Real) (h1 : θ ≠ 0) :
  (∃ (x y : Real), (Real.sqrt 6 * Real.sin θ) * x + Real.sqrt 3 * y - 2 = 0) →
  (∀ (x y : Real), (Real.sqrt 6 * Real.sin θ) * x + Real.sqrt 3 * y - 2 = 0 →
    Real.tan θ = -(Real.sqrt 6 / Real.sqrt 3) * Real.sin θ) →
  θ = 3 * Real.pi / 4 := by
  sorry

end line_angle_theorem_l2900_290089


namespace negative_greater_than_reciprocal_is_proper_fraction_l2900_290099

theorem negative_greater_than_reciprocal_is_proper_fraction (a : ℝ) :
  a < 0 ∧ a > 1 / a → -1 < a ∧ a < 0 :=
sorry

end negative_greater_than_reciprocal_is_proper_fraction_l2900_290099


namespace composite_and_prime_divisors_l2900_290060

/-- Given two distinct positive integers a and b where a, b > 1, and s_n = a^n + b^(n+1) -/
theorem composite_and_prime_divisors (a b : ℕ) (ha : a > 1) (hb : b > 1) (hab : a ≠ b) :
  let s : ℕ → ℕ := fun n => a^n + b^(n+1)
  (∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, ¬ Nat.Prime (s n)) ∧
  (∃ (P : Set ℕ), Set.Infinite P ∧ ∀ p ∈ P, Nat.Prime p ∧ ∃ n, p ∣ s n) := by
  sorry

end composite_and_prime_divisors_l2900_290060


namespace product_zero_l2900_290000

theorem product_zero (a b c : ℝ) : 
  (a^2 + b^2 = 1 ∧ a + b = 1 → a * b = 0) ∧
  (a^3 + b^3 + c^3 = 1 ∧ a^2 + b^2 + c^2 = 1 ∧ a + b + c = 1 → a * b * c = 0) := by
  sorry

end product_zero_l2900_290000


namespace square_sum_zero_implies_both_zero_l2900_290044

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end square_sum_zero_implies_both_zero_l2900_290044


namespace ferris_wheel_capacity_l2900_290091

/-- Calculates the total number of people who can ride a Ferris wheel -/
theorem ferris_wheel_capacity 
  (capacity : ℕ)           -- Number of people per ride
  (ride_duration : ℕ)      -- Duration of one ride in minutes
  (operation_time : ℕ) :   -- Total operation time in hours
  capacity * (60 / ride_duration) * operation_time = 1260 :=
by
  sorry

#check ferris_wheel_capacity 70 20 6

end ferris_wheel_capacity_l2900_290091


namespace z_in_second_quadrant_l2900_290080

/-- A complex number is in the second quadrant if its real part is negative and its imaginary part is positive -/
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

/-- If a complex number z satisfies (1-i)z = 2i, then z is in the second quadrant -/
theorem z_in_second_quadrant (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) :
  in_second_quadrant z := by
  sorry


end z_in_second_quadrant_l2900_290080


namespace D_144_l2900_290053

/-- D(n) represents the number of ways to write a positive integer n as a product of 
    integers strictly greater than 1, where the order of factors matters. -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem stating that D(144) = 45 -/
theorem D_144 : D 144 = 45 := by sorry

end D_144_l2900_290053


namespace greatest_x_value_l2900_290034

theorem greatest_x_value (x : ℤ) : 
  (2.134 * (10 : ℝ) ^ (x : ℝ) < 240000) ↔ x ≤ 5 :=
sorry

end greatest_x_value_l2900_290034


namespace base5_divisible_by_13_l2900_290019

/-- Converts a base 5 number to decimal --/
def base5ToDecimal (a b c d : ℕ) : ℕ :=
  a * 5^3 + b * 5^2 + c * 5 + d

/-- Checks if a number is divisible by 13 --/
def isDivisibleBy13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

theorem base5_divisible_by_13 :
  let y := 2
  let base5Num := base5ToDecimal 2 3 y 2
  isDivisibleBy13 base5Num :=
by sorry

end base5_divisible_by_13_l2900_290019


namespace largest_inscribed_square_area_l2900_290002

/-- The side length of the equilateral triangle -/
def triangle_side : ℝ := 10

/-- The rhombus formed by two identical equilateral triangles -/
structure Rhombus where
  side : ℝ
  is_formed_by_triangles : side = triangle_side

/-- The largest square inscribed in the rhombus -/
def largest_inscribed_square (r : Rhombus) : ℝ := sorry

/-- Theorem stating that the area of the largest inscribed square is 50 -/
theorem largest_inscribed_square_area (r : Rhombus) :
  (largest_inscribed_square r) ^ 2 = 50 := by sorry

end largest_inscribed_square_area_l2900_290002


namespace intersection_sum_l2900_290015

theorem intersection_sum : ∃ (x₁ x₂ : ℝ),
  (x₁^2 = 2*x₁ + 3) ∧
  (x₂^2 = 2*x₂ + 3) ∧
  (x₁ ≠ x₂) ∧
  (x₁ + x₂ = 2) :=
by sorry

end intersection_sum_l2900_290015


namespace equation_solution_l2900_290062

theorem equation_solution :
  ∃ y : ℝ, y ≠ 2 ∧ (7 * y / (y - 2) - 5 / (y - 2) = 2 / (y - 2)) ∧ y = 1 :=
by sorry

end equation_solution_l2900_290062


namespace inequality_solution_set_l2900_290021

-- Define the functions DE, BC, and DB
def DE (x : ℝ) : ℝ := sorry
def BC (x : ℝ) : ℝ := sorry
def DB (x : ℝ) : ℝ := sorry

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | DE x * BC x = DE x * (2 * DB x) ∧ DE x * BC x = 2 * (DE x)^2} = 
  {x : ℝ | 9/4 < x ∧ x < 19/4} :=
by sorry

end inequality_solution_set_l2900_290021


namespace sunset_delay_theorem_l2900_290025

/-- Calculates the minutes until sunset given the initial sunset time,
    daily sunset delay, days passed, and current time. -/
def minutesUntilSunset (initialSunsetMinutes : ℕ) (dailyDelayMinutes : ℚ)
                       (daysPassed : ℕ) (currentTimeMinutes : ℕ) : ℚ :=
  let newSunsetMinutes : ℚ := initialSunsetMinutes + daysPassed * dailyDelayMinutes
  newSunsetMinutes - currentTimeMinutes

/-- Proves that 40 days after March 1st, at 6:10 PM, 
    there are 38 minutes until sunset. -/
theorem sunset_delay_theorem :
  minutesUntilSunset 1080 1.2 40 1090 = 38 := by
  sorry

#eval minutesUntilSunset 1080 1.2 40 1090

end sunset_delay_theorem_l2900_290025


namespace absolute_value_equation_l2900_290072

theorem absolute_value_equation (x : ℚ) :
  |6 + x| = |6| + |x| ↔ x ≥ 0 := by sorry

end absolute_value_equation_l2900_290072


namespace sphere_and_cube_l2900_290037

/-- Given a sphere with surface area 256π cm² circumscribed around a cube,
    prove its volume and the cube's side length. -/
theorem sphere_and_cube (S : Real) (r : Real) (V : Real) (s : Real) : 
  S = 256 * Real.pi → -- Surface area of the sphere
  S = 4 * Real.pi * r^2 → -- Surface area formula
  V = (4/3) * Real.pi * r^3 → -- Volume formula for sphere
  2 * r = s * Real.sqrt 3 → -- Relation between sphere diameter and cube diagonal
  V = (2048/3) * Real.pi ∧ s = (16 * Real.sqrt 3) / 3 := by
  sorry

end sphere_and_cube_l2900_290037


namespace sum_of_xy_is_one_l2900_290029

theorem sum_of_xy_is_one (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^5 + 5*x^3*y + 5*x^2*y^2 + 5*x*y^3 + y^5 = 1) : 
  x + y = 1 := by sorry

end sum_of_xy_is_one_l2900_290029


namespace f_is_quadratic_l2900_290008

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem stating that f is a quadratic equation in one variable -/
theorem f_is_quadratic : is_quadratic_one_var f := by sorry

end f_is_quadratic_l2900_290008


namespace min_value_trig_expression_l2900_290093

theorem min_value_trig_expression (α : ℝ) : 
  9 / (Real.sin α)^2 + 1 / (Real.cos α)^2 ≥ 16 := by
  sorry

end min_value_trig_expression_l2900_290093


namespace binary_to_base4_conversion_l2900_290023

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc
    else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem binary_to_base4_conversion :
  decimal_to_base4 (binary_to_decimal [false, true, false, false, true, true, false, true, true, false, true]) = [2, 3, 1, 2, 2] := by
  sorry

end binary_to_base4_conversion_l2900_290023


namespace adoption_time_proof_l2900_290024

/-- The number of days required to adopt all puppies -/
def adoptionDays (initialPuppies : ℕ) (additionalPuppies : ℕ) (adoptionRate : ℕ) : ℕ :=
  (initialPuppies + additionalPuppies) / adoptionRate

/-- Theorem stating that it takes 11 days to adopt all puppies under given conditions -/
theorem adoption_time_proof :
  adoptionDays 15 62 7 = 11 := by
  sorry

end adoption_time_proof_l2900_290024


namespace square_area_15m_l2900_290009

theorem square_area_15m (side_length : ℝ) (h : side_length = 15) : 
  side_length * side_length = 225 := by
sorry

end square_area_15m_l2900_290009


namespace min_value_theorem_l2900_290078

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 1/x' + 1/y' = 1 → 
    1/(x' - 1) + 4/(y' - 1) ≥ min := by
  sorry

end min_value_theorem_l2900_290078


namespace cars_in_driveway_three_cars_in_driveway_l2900_290026

/-- Calculates the number of cars in the driveway given the total number of wheels and the number of wheels for each item. -/
theorem cars_in_driveway (total_wheels : ℕ) (car_wheels bike_wheels trash_can_wheels tricycle_wheels roller_skate_wheels : ℕ)
  (num_bikes num_trash_cans num_tricycles num_roller_skate_pairs : ℕ) : ℕ :=
  let other_wheels := num_bikes * bike_wheels + num_trash_cans * trash_can_wheels +
                      num_tricycles * tricycle_wheels + num_roller_skate_pairs * roller_skate_wheels
  let remaining_wheels := total_wheels - other_wheels
  remaining_wheels / car_wheels

/-- Proves that there are 3 cars in the driveway given the specific conditions. -/
theorem three_cars_in_driveway :
  cars_in_driveway 25 4 2 2 3 4 2 1 1 1 = 3 := by
  sorry

end cars_in_driveway_three_cars_in_driveway_l2900_290026


namespace quadratic_inequality_condition_l2900_290075

theorem quadratic_inequality_condition (m : ℝ) :
  (∀ x : ℝ, x > 1 ∧ x < 2 → x^2 + m*x + 4 < 0) ↔ m ≤ -5 :=
by sorry

end quadratic_inequality_condition_l2900_290075


namespace circumcenter_rational_coords_l2900_290001

/-- Given a triangle with rational coordinates, its circumcenter has rational coordinates. -/
theorem circumcenter_rational_coords 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℚ) :
  ∃ (x y : ℚ), 
    (x - a₁)^2 + (y - b₁)^2 = (x - a₂)^2 + (y - b₂)^2 ∧
    (x - a₁)^2 + (y - b₁)^2 = (x - a₃)^2 + (y - b₃)^2 :=
by sorry

end circumcenter_rational_coords_l2900_290001


namespace parking_lot_cars_parking_lot_problem_l2900_290046

theorem parking_lot_cars (total_wheels : ℕ) (num_bikes : ℕ) : ℕ :=
  let car_wheels := 4
  let bike_wheels := 2
  let num_cars := (total_wheels - num_bikes * bike_wheels) / car_wheels
  num_cars

theorem parking_lot_problem :
  parking_lot_cars 44 2 = 10 := by sorry

end parking_lot_cars_parking_lot_problem_l2900_290046


namespace arithmetic_progression_x_value_l2900_290088

/-- An arithmetic progression with first three terms x - 1, x + 1, and 2x + 3 -/
def arithmetic_progression (x : ℝ) : ℕ → ℝ
| 0 => x - 1
| 1 => x + 1
| 2 => 2*x + 3
| _ => 0  -- We only care about the first three terms

/-- The common difference of the arithmetic progression -/
def common_difference (x : ℝ) : ℝ := arithmetic_progression x 1 - arithmetic_progression x 0

theorem arithmetic_progression_x_value :
  ∀ x : ℝ, 
  (arithmetic_progression x 1 - arithmetic_progression x 0 = common_difference x) ∧
  (arithmetic_progression x 2 - arithmetic_progression x 1 = common_difference x) →
  x = 0 :=
by sorry

end arithmetic_progression_x_value_l2900_290088


namespace value_of_expression_l2900_290016

theorem value_of_expression (a b : ℤ) (ha : a = -3) (hb : b = 2) : a * (b - 3) = 3 := by
  sorry

end value_of_expression_l2900_290016


namespace square_property_of_natural_numbers_l2900_290098

theorem square_property_of_natural_numbers (a b : ℕ) 
  (h1 : ∃ k : ℕ, a * b = k^2)
  (h2 : ∃ m : ℕ, (2 * a + 1) * (2 * b + 1) = m^2) :
  ∃ n : ℕ, 
    2 < n ∧ 
    Even n ∧ 
    ∃ p : ℕ, (a + n) * (b + n) = p^2 := by
  sorry

end square_property_of_natural_numbers_l2900_290098


namespace uniqueRootIff_l2900_290033

/-- A function that represents the quadratic equation ax^2 + (a-3)x + 1 --/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 3) * x + 1

/-- Predicate that determines if the graph of f(a, x) intersects the x-axis at only one point --/
def hasUniqueRoot (a : ℝ) : Prop :=
  ∃! x, f a x = 0

/-- Theorem stating that f(a, x) has a unique root if and only if a is 0, 1, or 9 --/
theorem uniqueRootIff (a : ℝ) : hasUniqueRoot a ↔ a = 0 ∨ a = 1 ∨ a = 9 := by
  sorry

end uniqueRootIff_l2900_290033


namespace garrett_granola_bars_l2900_290059

/-- The number of oatmeal raisin granola bars Garrett bought -/
def oatmeal_raisin_bars : ℕ := 6

/-- The number of peanut granola bars Garrett bought -/
def peanut_bars : ℕ := 8

/-- The total number of granola bars Garrett bought -/
def total_bars : ℕ := oatmeal_raisin_bars + peanut_bars

theorem garrett_granola_bars : total_bars = 14 := by sorry

end garrett_granola_bars_l2900_290059


namespace length_MN_is_eleven_thirds_l2900_290051

/-- Triangle ABC with given side lengths and points M and N -/
structure TriangleABC where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Point M on AB such that CM is the angle bisector of ∠ACB
  M : ℝ
  -- Point N on AB such that CN is the altitude to AB
  N : ℝ
  -- Conditions
  h_AB : AB = 50
  h_BC : BC = 20
  h_AC : AC = 40
  h_M_angle_bisector : M = AB / 3
  h_N_altitude : N = BC * (AB^2 + BC^2 - AC^2) / (2 * AB * BC)

/-- The length of MN in the given triangle -/
def length_MN (t : TriangleABC) : ℝ := t.M - t.N

/-- Theorem stating that the length of MN is 11/3 -/
theorem length_MN_is_eleven_thirds (t : TriangleABC) :
  length_MN t = 11 / 3 := by
  sorry

end length_MN_is_eleven_thirds_l2900_290051


namespace intersection_A_B_l2900_290048

-- Define set A
def A : Set ℝ := {x | 0 < x ∧ x < 5}

-- Define set B
def B : Set ℝ := {x | (x + 1) / (x - 4) ≤ 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 4} := by
  sorry

end intersection_A_B_l2900_290048


namespace garden_area_theorem_l2900_290010

/-- The area of a rectangle with a square cut out from each of two different corners -/
def garden_area (length width cut1_side cut2_side : ℝ) : ℝ :=
  length * width - cut1_side^2 - cut2_side^2

/-- Theorem: The area of a 20x18 rectangle with 4x4 and 2x2 squares cut out is 340 sq ft -/
theorem garden_area_theorem :
  garden_area 20 18 4 2 = 340 := by
  sorry

end garden_area_theorem_l2900_290010


namespace intersection_equals_T_l2900_290079

-- Define set S
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}

-- Define set T
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Theorem statement
theorem intersection_equals_T : S ∩ T = T := by
  sorry

end intersection_equals_T_l2900_290079


namespace monic_quartic_polynomial_value_l2900_290022

theorem monic_quartic_polynomial_value (q : ℝ → ℝ) : 
  (∃ a b c : ℝ, ∀ x, q x = x^4 + a*x^3 + b*x^2 + c*x + 3) →
  q 0 = 3 →
  q 1 = 4 →
  q 2 = 7 →
  q 3 = 12 →
  q 4 = 43 := by
sorry

end monic_quartic_polynomial_value_l2900_290022


namespace water_percentage_in_fresh_grapes_l2900_290011

/-- The percentage of water in fresh grapes by weight -/
def water_percentage_fresh : ℝ := 90

/-- The percentage of water in dried grapes by weight -/
def water_percentage_dried : ℝ := 20

/-- The weight of fresh grapes in kg -/
def fresh_weight : ℝ := 20

/-- The weight of dried grapes in kg -/
def dried_weight : ℝ := 2.5

theorem water_percentage_in_fresh_grapes :
  water_percentage_fresh = 90 ∧
  (100 - water_percentage_fresh) / 100 * fresh_weight = 
  (100 - water_percentage_dried) / 100 * dried_weight :=
by sorry

end water_percentage_in_fresh_grapes_l2900_290011


namespace newspaper_spend_l2900_290055

/-- The cost of a weekday newspaper edition -/
def weekday_cost : ℚ := 0.50

/-- The cost of a Sunday newspaper edition -/
def sunday_cost : ℚ := 2.00

/-- The number of weekday editions Hillary buys per week -/
def weekday_editions : ℕ := 3

/-- The number of weeks -/
def weeks : ℕ := 8

/-- Hillary's total newspaper spend over 8 weeks -/
def total_spend : ℚ := weeks * (weekday_editions * weekday_cost + sunday_cost)

theorem newspaper_spend : total_spend = 28 := by
  sorry

end newspaper_spend_l2900_290055


namespace magic_square_sum_l2900_290004

/-- Represents a 3x3 magic square with some known values and variables -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  sum : ℕ
  row1_sum : a + 27 + b = sum
  row2_sum : 15 + c + d = sum
  row3_sum : 30 + e + 18 = sum
  col1_sum : 30 + 15 + a = sum
  col2_sum : e + c + 27 = sum
  col3_sum : 18 + d + b = sum
  diag1_sum : 30 + c + b = sum
  diag2_sum : 18 + c + a = sum

/-- Theorem: In a 3x3 magic square with the given known numbers, 
    if the sums of all rows, columns, and diagonals are equal, then d + e = 108 -/
theorem magic_square_sum (ms : MagicSquare) : ms.d + ms.e = 108 := by
  sorry

end magic_square_sum_l2900_290004


namespace female_officers_count_l2900_290082

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_ratio : ℚ) :
  total_on_duty = 170 →
  female_on_duty_ratio = 17 / 100 →
  female_ratio = 1 / 2 →
  ∃ (total_female : ℕ), 
    (female_on_duty_ratio * total_female = female_ratio * total_on_duty) ∧
    total_female = 500 :=
by sorry

end female_officers_count_l2900_290082


namespace sunflower_seed_contest_l2900_290038

theorem sunflower_seed_contest (player1 player2 player3 total : ℕ) : 
  player1 = 78 → 
  player2 = 53 → 
  player3 = player2 + 30 → 
  total = player1 + player2 + player3 → 
  total = 214 := by
sorry

end sunflower_seed_contest_l2900_290038


namespace least_distinct_values_l2900_290070

/-- Given a list of 2023 positive integers with a unique mode occurring 15 times,
    the least number of distinct values is 145. -/
theorem least_distinct_values (l : List ℕ+) (h1 : l.length = 2023) 
    (h2 : ∃! m, m ∈ l ∧ l.count m = 15) : 
    (∃ (s : Finset ℕ+), s.card = 145 ∧ ∀ x ∈ l, x ∈ s) ∧ 
    (∀ (s : Finset ℕ+), (∀ x ∈ l, x ∈ s) → s.card ≥ 145) :=
sorry

end least_distinct_values_l2900_290070


namespace f_range_is_real_l2900_290071

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 4 - Real.cos x * Real.sin x + Real.sin x ^ 4 + Real.tan x

theorem f_range_is_real : Set.range f = Set.univ :=
sorry

end f_range_is_real_l2900_290071


namespace largest_class_proof_l2900_290018

/-- The number of students in the largest class of a school with the following properties:
  - There are 5 classes
  - Each class has 2 students less than the previous class
  - The total number of students is 95
-/
def largest_class : ℕ := 23

theorem largest_class_proof :
  let classes := 5
  let student_difference := 2
  let total_students := 95
  let class_sizes := List.range classes |>.map (λ i => largest_class - i * student_difference)
  classes = 5 ∧
  student_difference = 2 ∧
  total_students = 95 ∧
  class_sizes.sum = total_students ∧
  largest_class ≥ 0 ∧
  (∀ i ∈ class_sizes, i ≥ 0) →
  largest_class = 23 :=
by sorry

end largest_class_proof_l2900_290018


namespace solution_implies_m_value_l2900_290050

theorem solution_implies_m_value (m : ℝ) : 
  (2 * 2 + m - 1 = 0) → m = -3 := by
  sorry

end solution_implies_m_value_l2900_290050


namespace missing_donuts_percentage_l2900_290047

def initial_donuts : ℕ := 30
def remaining_donuts : ℕ := 9

theorem missing_donuts_percentage :
  (initial_donuts - remaining_donuts : ℚ) / initial_donuts * 100 = 70 := by
  sorry

end missing_donuts_percentage_l2900_290047


namespace faye_earnings_l2900_290040

/-- Calculates the earnings from selling necklaces at a garage sale -/
def necklace_earnings (bead_necklaces gem_stone_necklaces price_per_necklace : ℕ) : ℕ :=
  (bead_necklaces + gem_stone_necklaces) * price_per_necklace

/-- Proves that Faye's earnings from selling necklaces are 70 dollars -/
theorem faye_earnings : necklace_earnings 3 7 7 = 70 := by
  sorry

end faye_earnings_l2900_290040


namespace geometric_sequence_303rd_term_l2900_290036

/-- Represents a geometric sequence -/
def GeometricSequence (a₁ : ℝ) (r : ℝ) := fun (n : ℕ) => a₁ * r ^ (n - 1)

theorem geometric_sequence_303rd_term :
  let seq := GeometricSequence 5 (-2)
  seq 303 = 5 * 2^302 := by
  sorry

end geometric_sequence_303rd_term_l2900_290036


namespace power_two_gt_sum_powers_l2900_290092

theorem power_two_gt_sum_powers (n : ℕ) (x : ℝ) (h1 : n ≥ 2) (h2 : |x| < 1) :
  2^n > (1 - x)^n + (1 + x)^n := by
  sorry

end power_two_gt_sum_powers_l2900_290092


namespace minimum_raft_capacity_l2900_290090

/-- Represents an animal with its weight -/
structure Animal where
  weight : ℕ

/-- Represents the raft with its capacity -/
structure Raft where
  capacity : ℕ

/-- Checks if the raft can carry at least two of the lightest animals -/
def canCarryTwoLightest (r : Raft) (animals : List Animal) : Prop :=
  r.capacity ≥ 2 * (animals.map Animal.weight).minimum

/-- Checks if all animals can be transported using the given raft -/
def canTransportAll (r : Raft) (animals : List Animal) : Prop :=
  canCarryTwoLightest r animals

/-- The main theorem stating the minimum raft capacity -/
theorem minimum_raft_capacity 
  (mice : List Animal)
  (moles : List Animal)
  (hamsters : List Animal)
  (h_mice : mice.length = 5 ∧ ∀ m ∈ mice, m.weight = 70)
  (h_moles : moles.length = 3 ∧ ∀ m ∈ moles, m.weight = 90)
  (h_hamsters : hamsters.length = 4 ∧ ∀ h ∈ hamsters, h.weight = 120)
  : ∃ (r : Raft), r.capacity = 140 ∧ 
    canTransportAll r (mice ++ moles ++ hamsters) ∧
    ∀ (r' : Raft), r'.capacity < 140 → ¬canTransportAll r' (mice ++ moles ++ hamsters) :=
sorry

end minimum_raft_capacity_l2900_290090


namespace square_sum_given_conditions_l2900_290061

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : x^2 + y^2 = 20) 
  (h2 : x * y = 6) : 
  (x + y)^2 = 32 := by
sorry

end square_sum_given_conditions_l2900_290061


namespace cantor_set_cardinality_cantor_set_operations_l2900_290066

-- Define the Cantor set
def CantorSet : Set ℝ := sorry

-- Theorem for part (a)
theorem cantor_set_cardinality : Cardinal.mk CantorSet = Cardinal.mk (Set.Icc 0 1) := by sorry

-- Define the sum and difference operations on sets
def setSum (A B : Set ℝ) : Set ℝ := {x | ∃ a ∈ A, ∃ b ∈ B, x = a + b}
def setDiff (A B : Set ℝ) : Set ℝ := {x | ∃ a ∈ A, ∃ b ∈ B, x = a - b}

-- Theorem for part (b)
theorem cantor_set_operations :
  (setSum CantorSet CantorSet = Set.Icc 0 2) ∧
  (setDiff CantorSet CantorSet = Set.Icc (-1) 1) := by sorry

end cantor_set_cardinality_cantor_set_operations_l2900_290066


namespace right_triangle_acute_angles_l2900_290085

theorem right_triangle_acute_angles (θ₁ θ₂ : ℝ) : 
  θ₁ = 25 → θ₁ + θ₂ = 90 → θ₂ = 65 := by
  sorry

end right_triangle_acute_angles_l2900_290085


namespace max_value_of_sum_and_reciprocal_l2900_290084

theorem max_value_of_sum_and_reciprocal (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 13 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 13 :=
by sorry

end max_value_of_sum_and_reciprocal_l2900_290084


namespace sequence_perfect_squares_l2900_290020

theorem sequence_perfect_squares (n : ℕ) (h : n > 0) :
  ∃ k : ℕ, (3 * ((10^n - 1) / 9) + 4) = k^2 := by
  sorry

end sequence_perfect_squares_l2900_290020


namespace inequality_proof_l2900_290065

theorem inequality_proof (p q : ℝ) (m n : ℕ+) 
  (h1 : p ≥ 0) (h2 : q ≥ 0) (h3 : p + q = 1) : 
  (1 - p ^ (m : ℝ)) ^ (n : ℝ) + (1 - q ^ (n : ℝ)) ^ (m : ℝ) ≥ 1 := by
  sorry

end inequality_proof_l2900_290065


namespace abs_sum_minimum_l2900_290097

theorem abs_sum_minimum (x : ℝ) : 
  |x + 1| + |x + 3| + |x + 6| ≥ 7 ∧ ∃ y : ℝ, |y + 1| + |y + 3| + |y + 6| = 7 := by
  sorry

end abs_sum_minimum_l2900_290097


namespace prob_both_divisible_by_four_is_one_thirty_sixth_l2900_290063

/-- The probability of rolling a specific number on a fair 6-sided die -/
def prob_single : ℚ := 1 / 6

/-- The set of numbers on a 6-sided die -/
def die_numbers : Set ℕ := {1, 2, 3, 4, 5, 6}

/-- The set of numbers on a 6-sided die that are divisible by 4 -/
def divisible_by_four : Set ℕ := {n ∈ die_numbers | n % 4 = 0}

/-- The probability that both dice show numbers divisible by 4 -/
def prob_both_divisible_by_four : ℚ := prob_single * prob_single

theorem prob_both_divisible_by_four_is_one_thirty_sixth :
  prob_both_divisible_by_four = 1 / 36 := by
  sorry

end prob_both_divisible_by_four_is_one_thirty_sixth_l2900_290063


namespace minimum_fare_increase_l2900_290012

/-- Represents the fare structure for a taxi service -/
structure FareStructure where
  n : ℝ  -- Total number of passengers
  t : ℝ  -- Base fare
  X : ℝ  -- Fare increase for businessmen

/-- Calculates the total revenue under the given fare structure -/
def totalRevenue (f : FareStructure) : ℝ :=
  0.75 * f.n * f.t + 0.2 * f.n * (f.t + f.X)

/-- Theorem stating the minimum fare increase that doesn't decrease total revenue -/
theorem minimum_fare_increase (f : FareStructure) :
  (∀ X : ℝ, totalRevenue { n := f.n, t := f.t, X := X } ≥ f.n * f.t → X ≥ f.t / 4) ∧
  totalRevenue { n := f.n, t := f.t, X := f.t / 4 } ≥ f.n * f.t :=
by sorry

end minimum_fare_increase_l2900_290012


namespace smallest_sum_consecutive_primes_div_by_5_l2900_290042

/-- Three consecutive primes with sum divisible by 5 -/
def ConsecutivePrimesWithSumDivBy5 (p q r : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  q = Nat.succ p ∧ r = Nat.succ q ∧
  (p + q + r) % 5 = 0

/-- The smallest sum of three consecutive primes divisible by 5 -/
theorem smallest_sum_consecutive_primes_div_by_5 :
  ∃ (p q r : ℕ), ConsecutivePrimesWithSumDivBy5 p q r ∧
    ∀ (a b c : ℕ), ConsecutivePrimesWithSumDivBy5 a b c → p + q + r ≤ a + b + c ∧
    p + q + r = 10 :=
sorry

end smallest_sum_consecutive_primes_div_by_5_l2900_290042


namespace rectangle_perimeter_l2900_290069

/-- Given a rectangle with area 800 cm² and length twice its width, prove its perimeter is 120 cm. -/
theorem rectangle_perimeter (width : ℝ) (length : ℝ) :
  width > 0 →
  length > 0 →
  length = 2 * width →
  width * length = 800 →
  2 * (width + length) = 120 := by
sorry

end rectangle_perimeter_l2900_290069


namespace lily_cost_is_four_l2900_290064

/-- Represents the cost structure for wedding reception decorations --/
structure WeddingDecoration where
  numTables : Nat
  tableclothCost : Nat
  placeSettingCost : Nat
  placeSettingsPerTable : Nat
  rosesPerCenterpiece : Nat
  roseCost : Nat
  liliesPerCenterpiece : Nat
  totalCost : Nat

/-- Calculates the cost of each lily given the wedding decoration details --/
def lilyCost (d : WeddingDecoration) : Rat :=
  let tableCostWithoutLilies := d.tableclothCost + 
                                d.placeSettingCost * d.placeSettingsPerTable + 
                                d.rosesPerCenterpiece * d.roseCost
  let totalCostWithoutLilies := d.numTables * tableCostWithoutLilies
  let totalLilyCost := d.totalCost - totalCostWithoutLilies
  let totalLilies := d.numTables * d.liliesPerCenterpiece
  totalLilyCost / totalLilies

/-- Theorem stating that the lily cost for the given wedding decoration is $4 --/
theorem lily_cost_is_four (d : WeddingDecoration) 
  (h1 : d.numTables = 20)
  (h2 : d.tableclothCost = 25)
  (h3 : d.placeSettingCost = 10)
  (h4 : d.placeSettingsPerTable = 4)
  (h5 : d.rosesPerCenterpiece = 10)
  (h6 : d.roseCost = 5)
  (h7 : d.liliesPerCenterpiece = 15)
  (h8 : d.totalCost = 3500) : 
  lilyCost d = 4 := by
  sorry

#eval lilyCost {
  numTables := 20,
  tableclothCost := 25,
  placeSettingCost := 10,
  placeSettingsPerTable := 4,
  rosesPerCenterpiece := 10,
  roseCost := 5,
  liliesPerCenterpiece := 15,
  totalCost := 3500
}

end lily_cost_is_four_l2900_290064


namespace dogwood_trees_theorem_l2900_290049

def dogwood_trees_problem (initial_trees : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) : ℕ :=
  initial_trees + planted_today + planted_tomorrow

theorem dogwood_trees_theorem (initial_trees planted_today planted_tomorrow : ℕ) :
  dogwood_trees_problem initial_trees planted_today planted_tomorrow =
  initial_trees + planted_today + planted_tomorrow :=
by
  sorry

#eval dogwood_trees_problem 7 5 4

end dogwood_trees_theorem_l2900_290049


namespace sum_of_roots_equal_three_l2900_290014

-- Define the polynomial
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 - 48 * x - 12

-- State the theorem
theorem sum_of_roots_equal_three :
  ∃ (r p q : ℝ), f r = 0 ∧ f p = 0 ∧ f q = 0 ∧ r + p + q = 3 := by
  sorry

end sum_of_roots_equal_three_l2900_290014


namespace centipede_sock_shoe_permutations_l2900_290003

/-- Represents the number of legs a centipede has -/
def num_legs : ℕ := 10

/-- Represents the total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- Represents the constraint that for each leg, the sock must be put on before the shoe -/
def sock_before_shoe_constraint (leg : ℕ) : Prop :=
  leg ≤ num_legs ∧ ∃ (sock_pos shoe_pos : ℕ), sock_pos < shoe_pos

/-- The main theorem stating the number of valid permutations -/
theorem centipede_sock_shoe_permutations :
  (Nat.factorial total_items) / (2^num_legs) =
  (Nat.factorial 20) / (2^10) :=
sorry

end centipede_sock_shoe_permutations_l2900_290003


namespace intersection_implies_a_value_l2900_290032

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a + 1, a^2 + 4}
  A ∩ B = {3} → a = 2 := by
sorry

end intersection_implies_a_value_l2900_290032


namespace expression_equals_one_l2900_290039

theorem expression_equals_one : (2 * 6) / (12 * 14) * (3 * 12 * 14) / (2 * 6 * 3) = 1 := by
  sorry

end expression_equals_one_l2900_290039


namespace least_positive_integer_with_given_remainders_l2900_290041

theorem least_positive_integer_with_given_remainders : ∃ N : ℕ+,
  (N : ℤ) ≡ 3 [ZMOD 4] ∧
  (N : ℤ) ≡ 4 [ZMOD 5] ∧
  (N : ℤ) ≡ 5 [ZMOD 6] ∧
  (N : ℤ) ≡ 6 [ZMOD 7] ∧
  (N : ℤ) ≡ 10 [ZMOD 11] ∧
  (∀ m : ℕ+, m < N →
    ¬((m : ℤ) ≡ 3 [ZMOD 4] ∧
      (m : ℤ) ≡ 4 [ZMOD 5] ∧
      (m : ℤ) ≡ 5 [ZMOD 6] ∧
      (m : ℤ) ≡ 6 [ZMOD 7] ∧
      (m : ℤ) ≡ 10 [ZMOD 11])) ∧
  N = 4619 :=
sorry

end least_positive_integer_with_given_remainders_l2900_290041


namespace egg_count_l2900_290095

theorem egg_count (initial_eggs : ℕ) (used_eggs : ℕ) (num_chickens : ℕ) (eggs_per_chicken : ℕ) : 
  initial_eggs = 10 → 
  used_eggs = 5 → 
  num_chickens = 2 → 
  eggs_per_chicken = 3 → 
  initial_eggs - used_eggs + num_chickens * eggs_per_chicken = 11 := by
  sorry

#check egg_count

end egg_count_l2900_290095


namespace quadratic_roots_condition_l2900_290030

theorem quadratic_roots_condition (a : ℝ) : 
  (-1 < a ∧ a < 1) → 
  (∃ x₁ x₂ : ℝ, x₁ * x₂ = a - 2 ∧ x₁ + x₂ = -(a + 1) ∧ x₁ > 0 ∧ x₂ < 0) ∧
  ¬(∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ * x₂ = a - 2 ∧ x₁ + x₂ = -(a + 1) ∧ x₁ > 0 ∧ x₂ < 0) → (-1 < a ∧ a < 1)) :=
by sorry

end quadratic_roots_condition_l2900_290030


namespace flatbread_diameters_exist_l2900_290006

/-- The diameter of the skillet -/
def skillet_diameter : ℕ := 26

/-- Predicate to check if three positive integers satisfy the required conditions -/
def valid_diameters (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  x + y + z = skillet_diameter ∧
  x^2 + y^2 + z^2 = 338 ∧
  (x^2 + y^2 + z^2 : ℚ) / 4 = (skillet_diameter^2 : ℚ) / 8

/-- Theorem stating the existence of three positive integers satisfying the conditions -/
theorem flatbread_diameters_exist : ∃ x y z : ℕ, valid_diameters x y z := by
  sorry

end flatbread_diameters_exist_l2900_290006


namespace max_value_theorem_l2900_290057

theorem max_value_theorem (x y z : ℝ) (h : 3 * x + 4 * y + 2 * z = 12) :
  ∃ (max : ℝ), max = 3 ∧ ∀ (a b c : ℝ), 3 * a + 4 * b + 2 * c = 12 →
    a^2 * b + a^2 * c + b * c^2 ≤ max := by
  sorry

end max_value_theorem_l2900_290057


namespace three_distinct_zeros_range_l2900_290074

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

-- State the theorem
theorem three_distinct_zeros_range (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) →
  -2 < a ∧ a < 2 :=
by sorry

end three_distinct_zeros_range_l2900_290074


namespace faucet_leak_proof_l2900_290077

/-- Represents a linear function y = kt + b -/
structure LinearFunction where
  k : ℝ
  b : ℝ

/-- The linear function passes through the points (1, 7) and (2, 12) -/
def passesThrough (f : LinearFunction) : Prop :=
  f.k * 1 + f.b = 7 ∧ f.k * 2 + f.b = 12

/-- The value of the function at t = 20 -/
def valueAt20 (f : LinearFunction) : ℝ :=
  f.k * 20 + f.b

/-- The total water leaked in 30 days in milliliters -/
def totalLeaked (f : LinearFunction) : ℝ :=
  f.k * 60 * 24 * 30

theorem faucet_leak_proof (f : LinearFunction) 
  (h : passesThrough f) : 
  f.k = 5 ∧ f.b = 2 ∧ 
  valueAt20 f = 102 ∧ 
  totalLeaked f = 216000 := by
  sorry

#check faucet_leak_proof

end faucet_leak_proof_l2900_290077


namespace existence_of_constant_l2900_290056

theorem existence_of_constant : ∃ c : ℝ, c > 0 ∧
  ∀ a b n : ℕ, a > 0 → b > 0 → n > 0 →
  (∀ i j : ℕ, i ≤ n → j ≤ n → Nat.gcd (a + i) (b + j) > 1) →
  min a b > (c * n : ℝ) ^ (n / 2 : ℝ) := by
  sorry

end existence_of_constant_l2900_290056


namespace oxford_high_school_population_l2900_290031

/-- The total number of people in Oxford High School -/
def total_people (teachers : ℕ) (principal : ℕ) (classes : ℕ) (students_per_class : ℕ) : ℕ :=
  teachers + principal + (classes * students_per_class)

/-- Theorem: The total number of people in Oxford High School is 349 -/
theorem oxford_high_school_population :
  total_people 48 1 15 20 = 349 := by
  sorry

end oxford_high_school_population_l2900_290031


namespace factorization_equality_l2900_290081

theorem factorization_equality (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) := by
  sorry

end factorization_equality_l2900_290081


namespace smallest_max_sum_l2900_290083

theorem smallest_max_sum (p q r s t : ℕ+) (h : p + q + r + s + t = 2022) :
  let N := max (p + q) (max (q + r) (max (r + s) (s + t)))
  506 ≤ N ∧ ∃ (p' q' r' s' t' : ℕ+), p' + q' + r' + s' + t' = 2022 ∧ 
    max (p' + q') (max (q' + r') (max (r' + s') (s' + t'))) = 506 :=
by sorry

end smallest_max_sum_l2900_290083
