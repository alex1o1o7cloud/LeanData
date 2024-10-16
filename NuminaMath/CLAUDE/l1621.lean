import Mathlib

namespace NUMINAMATH_CALUDE_thompson_children_probability_l1621_162123

theorem thompson_children_probability :
  let n : ℕ := 8  -- number of children
  let p : ℚ := 1/2  -- probability of each child being male (or female)
  let total_outcomes : ℕ := 2^n  -- total number of possible gender combinations
  let equal_outcomes : ℕ := n.choose (n/2)  -- number of combinations with equal sons and daughters
  
  (total_outcomes - equal_outcomes : ℚ) / total_outcomes = 93/128 :=
by sorry

end NUMINAMATH_CALUDE_thompson_children_probability_l1621_162123


namespace NUMINAMATH_CALUDE_bake_sale_cookie_price_jack_cookie_price_l1621_162116

/-- Calculates the required price per cookie for Jack's bake sale -/
theorem bake_sale_cookie_price (brownie_price : ℝ) (brownie_count : ℕ) 
  (lemon_square_price : ℝ) (lemon_square_count : ℕ) 
  (total_goal : ℝ) (cookie_count : ℕ) : ℝ :=
  let current_sales := brownie_price * brownie_count + lemon_square_price * lemon_square_count
  let remaining_goal := total_goal - current_sales
  remaining_goal / cookie_count

/-- Proves that the cookie price is $4 given the specific conditions of Jack's bake sale -/
theorem jack_cookie_price : 
  bake_sale_cookie_price 3 4 2 5 50 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_cookie_price_jack_cookie_price_l1621_162116


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1621_162144

/-- A quadratic equation x^2 + 5x + k = 0 has distinct real roots if and only if k < 25/4 -/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 5*x + k = 0 ∧ y^2 + 5*y + k = 0) ↔ k < 25/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1621_162144


namespace NUMINAMATH_CALUDE_min_time_for_all_flashes_l1621_162101

/-- The number of colored lights -/
def num_lights : ℕ := 5

/-- The number of colors available -/
def num_colors : ℕ := 5

/-- The time for one light to shine in seconds -/
def shine_time : ℕ := 1

/-- The interval between two consecutive flashes in seconds -/
def interval_time : ℕ := 5

/-- The number of different possible flashes -/
def num_flashes : ℕ := Nat.factorial num_lights

/-- The minimum time required to achieve all different flashes in seconds -/
def min_time_required : ℕ := 
  (num_flashes * num_lights * shine_time) + ((num_flashes - 1) * interval_time)

theorem min_time_for_all_flashes : min_time_required = 1195 := by
  sorry

end NUMINAMATH_CALUDE_min_time_for_all_flashes_l1621_162101


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1621_162129

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3*x - 15 → x ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1621_162129


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1621_162160

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 3
  let θ : ℝ := 3 * π / 2
  let φ : ℝ := π / 3
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1621_162160


namespace NUMINAMATH_CALUDE_larger_integer_problem_l1621_162175

theorem larger_integer_problem (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  ((a > b ∧ a - b = 8) ∨ (b > a ∧ b - a = 8)) ∧ 
  a * b = 120 →
  max a b = 20 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l1621_162175


namespace NUMINAMATH_CALUDE_white_balls_count_l1621_162170

theorem white_balls_count (black_balls : ℕ) (prob_white : ℚ) (white_balls : ℕ) : 
  black_balls = 6 →
  prob_white = 45454545454545453 / 100000000000000000 →
  (white_balls : ℚ) / ((black_balls : ℚ) + (white_balls : ℚ)) = prob_white →
  white_balls = 5 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l1621_162170


namespace NUMINAMATH_CALUDE_difference_of_squares_l1621_162122

theorem difference_of_squares (x y : ℕ+) 
  (sum_eq : x + y = 18)
  (product_eq : x * y = 80) :
  x^2 - y^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1621_162122


namespace NUMINAMATH_CALUDE_f_properties_l1621_162192

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_properties :
  let e := Real.exp 1
  (∀ x ∈ Set.Ioo 0 e, ∀ y ∈ Set.Ioo 0 e, x < y → f x < f y) ∧
  (∀ x ∈ Set.Ioi e, ∀ y ∈ Set.Ioi e, x < y → f x > f y) ∧
  (∀ x ∈ Set.Ioo 0 (Real.exp 1), f x ≤ f e) ∧
  (∀ x ∈ Set.Ioi (Real.exp 1), f x < f e) ∧
  (∀ a : ℝ, (∀ x ≥ 1, f x ≤ a * (1 - 1 / x^2)) ↔ a ≥ 1/2) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l1621_162192


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l1621_162142

theorem absolute_value_simplification : |(-4^2 + 6)| = 10 := by sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l1621_162142


namespace NUMINAMATH_CALUDE_largest_digit_sum_l1621_162104

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem largest_digit_sum (a b c z : ℕ) : 
  is_digit a → is_digit b → is_digit c →
  (100 * a + 10 * b + c : ℚ) / 1000 = 1 / z →
  0 < z → z ≤ 12 →
  a + b + c ≤ 8 ∧ ∃ a' b' c' z', 
    is_digit a' ∧ is_digit b' ∧ is_digit c' ∧
    (100 * a' + 10 * b' + c' : ℚ) / 1000 = 1 / z' ∧
    0 < z' ∧ z' ≤ 12 ∧
    a' + b' + c' = 8 :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l1621_162104


namespace NUMINAMATH_CALUDE_lucille_paint_cans_l1621_162176

/-- Represents the dimensions of a wall -/
structure Wall where
  width : ℝ
  height : ℝ

/-- Calculates the area of a wall -/
def wallArea (w : Wall) : ℝ := w.width * w.height

/-- Represents the room to be painted -/
structure Room where
  wall1 : Wall
  wall2 : Wall
  wall3 : Wall
  wall4 : Wall

/-- Calculates the total area of all walls in the room -/
def totalArea (r : Room) : ℝ :=
  wallArea r.wall1 + wallArea r.wall2 + wallArea r.wall3 + wallArea r.wall4

/-- The coverage area of one can of paint -/
def paintCoverage : ℝ := 2

/-- Lucille's room configuration -/
def lucilleRoom : Room :=
  { wall1 := { width := 3, height := 2 }
  , wall2 := { width := 3, height := 2 }
  , wall3 := { width := 5, height := 2 }
  , wall4 := { width := 4, height := 2 } }

/-- Theorem: Lucille needs 15 cans of paint -/
theorem lucille_paint_cans : 
  ⌈(totalArea lucilleRoom) / paintCoverage⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_lucille_paint_cans_l1621_162176


namespace NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l1621_162124

/-- Given a circle with area M and circumference N, if M/N = 15, then the radius is 30 -/
theorem circle_radius_from_area_circumference_ratio (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 15) :
  ∃ r : ℝ, r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l1621_162124


namespace NUMINAMATH_CALUDE_ferris_wheel_small_seats_l1621_162172

/-- Represents a Ferris wheel with small and large seats -/
structure FerrisWheel where
  small_seats : ℕ
  large_seats : ℕ
  small_seat_capacity : ℕ
  people_on_small_seats : ℕ

/-- The number of small seats on the Ferris wheel is 2 -/
theorem ferris_wheel_small_seats (fw : FerrisWheel) 
  (h1 : fw.large_seats = 23)
  (h2 : fw.small_seat_capacity = 14)
  (h3 : fw.people_on_small_seats = 28) :
  fw.small_seats = 2 := by
  sorry

#check ferris_wheel_small_seats

end NUMINAMATH_CALUDE_ferris_wheel_small_seats_l1621_162172


namespace NUMINAMATH_CALUDE_ski_and_snowboard_intersection_l1621_162141

theorem ski_and_snowboard_intersection (total : ℕ) (ski : ℕ) (snowboard : ℕ) (neither : ℕ)
  (h_total : total = 20)
  (h_ski : ski = 11)
  (h_snowboard : snowboard = 13)
  (h_neither : neither = 3) :
  ski + snowboard - (total - neither) = 7 :=
by sorry

end NUMINAMATH_CALUDE_ski_and_snowboard_intersection_l1621_162141


namespace NUMINAMATH_CALUDE_cookies_eaten_difference_l1621_162112

theorem cookies_eaten_difference (initial_sweet initial_salty eaten_sweet eaten_salty : ℕ) 
  (h1 : initial_sweet = 37)
  (h2 : initial_salty = 11)
  (h3 : eaten_sweet = 5)
  (h4 : eaten_salty = 2) :
  eaten_sweet - eaten_salty = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_difference_l1621_162112


namespace NUMINAMATH_CALUDE_expected_ones_three_dice_l1621_162154

/-- A standard die with 6 sides -/
def StandardDie : Type := Fin 6

/-- The probability of rolling a 1 on a standard die -/
def probOne : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a standard die -/
def probNotOne : ℚ := 5 / 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The expected number of 1's when rolling three standard dice -/
def expectedOnes : ℚ := 1 / 2

/-- Theorem stating that the expected number of 1's when rolling three standard dice is 1/2 -/
theorem expected_ones_three_dice :
  (numDice : ℚ) * probOne = expectedOnes :=
sorry

end NUMINAMATH_CALUDE_expected_ones_three_dice_l1621_162154


namespace NUMINAMATH_CALUDE_picnic_age_problem_l1621_162108

theorem picnic_age_problem (initial_count : ℕ) (new_count : ℕ) (new_avg_age : ℝ) (final_avg_age : ℝ) :
  initial_count = 15 →
  new_count = 15 →
  new_avg_age = 15 →
  final_avg_age = 15.5 →
  ∃ (initial_avg_age : ℝ),
    initial_avg_age * initial_count + new_avg_age * new_count = 
    final_avg_age * (initial_count + new_count) ∧
    initial_avg_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_picnic_age_problem_l1621_162108


namespace NUMINAMATH_CALUDE_volume_of_specific_box_l1621_162162

/-- The volume of a rectangular box -/
def box_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a box with dimensions 20 cm, 15 cm, and 10 cm is 3000 cm³ -/
theorem volume_of_specific_box : box_volume 20 15 10 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_box_l1621_162162


namespace NUMINAMATH_CALUDE_find_unknown_number_l1621_162157

theorem find_unknown_number : ∃ x : ℝ, (20 + 40 + 60) / 3 = ((10 + 70 + x) / 3) + 4 := by
  sorry

end NUMINAMATH_CALUDE_find_unknown_number_l1621_162157


namespace NUMINAMATH_CALUDE_floor_sqrt_sum_equality_and_counterexample_l1621_162128

theorem floor_sqrt_sum_equality_and_counterexample :
  (∀ n : ℕ, ⌊Real.sqrt n + Real.sqrt (n + 2)⌋ = ⌊Real.sqrt (4 * n + 1)⌋) ∧
  (∃ x : ℝ, ⌊Real.sqrt x + Real.sqrt (x + 2)⌋ ≠ ⌊Real.sqrt (4 * x + 1)⌋) :=
by sorry

end NUMINAMATH_CALUDE_floor_sqrt_sum_equality_and_counterexample_l1621_162128


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l1621_162177

theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y : ℝ) :
  S.card = 50 →
  x ∈ S →
  y ∈ S →
  x = 45 →
  y = 55 →
  (S.sum id) / S.card = 38 →
  ((S.sum id - (x + y)) / (S.card - 2) : ℝ) = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l1621_162177


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1621_162171

/-- If x^2 + 6x + k^2 is exactly the square of a polynomial, then k = ±3 -/
theorem perfect_square_condition (k : ℝ) : 
  (∃ (p : ℝ → ℝ), ∀ x, x^2 + 6*x + k^2 = (p x)^2) → k = 3 ∨ k = -3 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1621_162171


namespace NUMINAMATH_CALUDE_EPC42_probability_l1621_162119

/-- The set of vowels used in Logicville license plates -/
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'Y'}

/-- The set of consonants used in Logicville license plates -/
def consonants : Finset Char := {'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Z'}

/-- The set of two-digit numbers used in Logicville license plates -/
def twoDigitNumbers : Finset Nat := Finset.range 100

/-- A Logicville license plate -/
structure LicensePlate where
  first : Char
  second : Char
  third : Char
  fourth : Nat
  first_in_vowels : first ∈ vowels
  second_in_consonants : second ∈ consonants
  third_in_consonants : third ∈ consonants
  second_neq_third : second ≠ third
  fourth_in_range : fourth ∈ twoDigitNumbers

/-- The probability of randomly selecting a specific license plate in Logicville -/
def licensePlateProbability (plate : LicensePlate) : ℚ :=
  1 / (vowels.card * consonants.card * (consonants.card - 1) * twoDigitNumbers.card)

/-- The specific license plate "EPC42" -/
def EPC42 : LicensePlate := {
  first := 'E',
  second := 'P',
  third := 'C',
  fourth := 42,
  first_in_vowels := by simp [vowels],
  second_in_consonants := by simp [consonants],
  third_in_consonants := by simp [consonants],
  second_neq_third := by decide,
  fourth_in_range := by simp [twoDigitNumbers]
}

/-- Theorem: The probability of randomly selecting "EPC42" in Logicville is 1/252,000 -/
theorem EPC42_probability :
  licensePlateProbability EPC42 = 1 / 252000 := by
  sorry

end NUMINAMATH_CALUDE_EPC42_probability_l1621_162119


namespace NUMINAMATH_CALUDE_shenny_vacation_shirts_l1621_162166

/-- The number of shirts Shenny needs to pack for her vacation -/
def shirts_to_pack (vacation_days : ℕ) (same_shirt_days : ℕ) (different_shirts_per_day : ℕ) : ℕ :=
  (vacation_days - same_shirt_days) * different_shirts_per_day + 1

/-- Proof that Shenny needs to pack 11 shirts for her vacation -/
theorem shenny_vacation_shirts :
  shirts_to_pack 7 2 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_shenny_vacation_shirts_l1621_162166


namespace NUMINAMATH_CALUDE_speed_conversion_l1621_162187

-- Define the conversion factor
def meters_per_second_to_kmph : ℝ := 3.6

-- Define the given speed in meters per second
def speed_in_mps : ℝ := 16.668

-- State the theorem
theorem speed_conversion :
  speed_in_mps * meters_per_second_to_kmph = 60.0048 := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l1621_162187


namespace NUMINAMATH_CALUDE_even_sum_probability_l1621_162126

theorem even_sum_probability (wheel1_even_prob wheel2_even_prob : ℚ) 
  (h1 : wheel1_even_prob = 3 / 5)
  (h2 : wheel2_even_prob = 1 / 2) : 
  wheel1_even_prob * wheel2_even_prob + (1 - wheel1_even_prob) * (1 - wheel2_even_prob) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_probability_l1621_162126


namespace NUMINAMATH_CALUDE_perfect_square_m_l1621_162147

theorem perfect_square_m (k m n : ℕ) (h1 : k > 0) (h2 : m > 0) (h3 : n > 0) 
  (h4 : Odd k) (h5 : (2 + Real.sqrt 3)^k = 1 + m + n * Real.sqrt 3) : 
  ∃ (q : ℕ), m = q^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_m_l1621_162147


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l1621_162168

def f (x : ℝ) := x^3 + 3*x - 1

theorem sum_of_a_and_b (a b : ℝ) (h1 : f (a - 3) = -3) (h2 : f (b - 3) = 1) : a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l1621_162168


namespace NUMINAMATH_CALUDE_golden_ratio_product_ab_pq_minus_n_l1621_162150

/-- The golden ratio is the positive root of x^2 + x - 1 = 0 -/
theorem golden_ratio : ∃ x : ℝ, x > 0 ∧ x^2 + x - 1 = 0 ∧ x = (-1 + Real.sqrt 5) / 2 := by sorry

/-- Given a^2 + ma = 1 and b^2 - 2mb = 4, ab = 2 -/
theorem product_ab (m a b : ℝ) (h1 : a^2 + m*a = 1) (h2 : b^2 - 2*m*b = 4) (h3 : b ≠ -2*a) : a * b = 2 := by sorry

/-- Given p^2 + np - 1 = q and q^2 + nq - 1 = p, pq - n = 0 -/
theorem pq_minus_n (n p q : ℝ) (h1 : p^2 + n*p - 1 = q) (h2 : q^2 + n*q - 1 = p) (h3 : p ≠ q) : p * q - n = 0 := by sorry

end NUMINAMATH_CALUDE_golden_ratio_product_ab_pq_minus_n_l1621_162150


namespace NUMINAMATH_CALUDE_favorite_books_probability_l1621_162199

variable (n : ℕ) (k : ℕ)

def P (n k : ℕ) : ℚ := (k.factorial * (n - k + 1).factorial) / n.factorial

theorem favorite_books_probability (h : k ≤ n) :
  (∀ m, m ≤ n → P n k ≥ P n m) ↔ (k = 1 ∨ k = n) ∧
  (n % 2 = 0 → P n k ≤ P n (n / 2)) ∧
  (n % 2 ≠ 0 → P n k ≤ P n ((n + 1) / 2)) :=
sorry

end NUMINAMATH_CALUDE_favorite_books_probability_l1621_162199


namespace NUMINAMATH_CALUDE_sticker_count_l1621_162106

/-- Given the ratio of stickers and Kate's sticker count, prove the combined count of Jenna's and Ava's stickers -/
theorem sticker_count (kate_ratio jenna_ratio ava_ratio : ℕ) 
  (kate_stickers : ℕ) (h_ratio : kate_ratio = 7 ∧ jenna_ratio = 4 ∧ ava_ratio = 5) 
  (h_kate : kate_stickers = 42) : 
  (jenna_ratio + ava_ratio) * (kate_stickers / kate_ratio) = 54 := by
  sorry

#check sticker_count

end NUMINAMATH_CALUDE_sticker_count_l1621_162106


namespace NUMINAMATH_CALUDE_triangle_area_is_two_l1621_162178

/-- The area of the triangle formed by the line x + y - 2 = 0 and the coordinate axes -/
def triangle_area : ℝ := 2

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x + y - 2 = 0

theorem triangle_area_is_two :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_equation x₁ y₁ ∧
    line_equation x₂ y₂ ∧
    x₁ = 0 ∧ y₂ = 0 ∧
    (1/2 : ℝ) * x₂ * y₁ = triangle_area :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_is_two_l1621_162178


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1621_162111

theorem tangent_line_to_circle (a : ℝ) : 
  (∃ (x y : ℝ), (x - a)^2 + (y - 3)^2 = 5 ∧ y = 2*x) →
  (a = -1 ∨ a = 4) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1621_162111


namespace NUMINAMATH_CALUDE_gabby_fruit_ratio_l1621_162109

/-- Represents the number of fruits Gabby harvested -/
structure FruitHarvest where
  watermelons : ℕ
  peaches : ℕ
  plums : ℕ

/-- Conditions of Gabby's fruit harvest -/
def gabbyHarvest : FruitHarvest where
  watermelons := 1
  peaches := 13
  plums := 39

theorem gabby_fruit_ratio :
  let h := gabbyHarvest
  h.watermelons = 1 ∧
  h.peaches = h.watermelons + 12 ∧
  h.watermelons + h.peaches + h.plums = 53 →
  h.plums / h.peaches = 3 := by
  sorry

end NUMINAMATH_CALUDE_gabby_fruit_ratio_l1621_162109


namespace NUMINAMATH_CALUDE_angle_D_value_l1621_162134

-- Define the angles as real numbers
variable (A B C D F : ℝ)

-- State the theorem
theorem angle_D_value (h1 : A + B = 180)
                      (h2 : C = D)
                      (h3 : B = 90)
                      (h4 : F = 50)
                      (h5 : A + C + F = 180) : D = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_value_l1621_162134


namespace NUMINAMATH_CALUDE_g_composition_points_sum_l1621_162169

/-- Given a function g with specific values, prove the existence of points on g(g(x)) with a certain sum property -/
theorem g_composition_points_sum (g : ℝ → ℝ) 
  (h1 : g 2 = 4) (h2 : g 3 = 2) (h3 : g 4 = 6) :
  ∃ (p q r s : ℝ), g (g p) = q ∧ g (g r) = s ∧ p * q + r * s = 24 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_points_sum_l1621_162169


namespace NUMINAMATH_CALUDE_balance_point_specific_rod_l1621_162153

/-- Represents the rod with attached weights -/
structure WeightedRod where
  length : Real
  weights : List (Real × Real)  -- List of (position, weight) pairs

/-- Calculates the balance point of a weighted rod -/
def balancePoint (rod : WeightedRod) : Real :=
  sorry

/-- Theorem stating the balance point for the specific rod configuration -/
theorem balance_point_specific_rod :
  let rod : WeightedRod := {
    length := 4,
    weights := [(0, 20), (1, 30), (2, 40), (3, 50), (4, 60)]
  }
  balancePoint rod = 2.5 := by sorry

end NUMINAMATH_CALUDE_balance_point_specific_rod_l1621_162153


namespace NUMINAMATH_CALUDE_nickel_difference_l1621_162198

/-- The number of cents in a nickel -/
def cents_per_nickel : ℕ := 5

/-- The total number of cents Ray has initially -/
def ray_initial_cents : ℕ := 175

/-- The number of cents Ray gives to Peter -/
def cents_to_peter : ℕ := 30

/-- Calculates the number of nickels given a number of cents -/
def cents_to_nickels (cents : ℕ) : ℕ := cents / cents_per_nickel

/-- Theorem stating the difference in nickels between Randi and Peter -/
theorem nickel_difference : 
  cents_to_nickels (2 * cents_to_peter) - cents_to_nickels cents_to_peter = 6 := by
  sorry

end NUMINAMATH_CALUDE_nickel_difference_l1621_162198


namespace NUMINAMATH_CALUDE_remainder_of_double_division_l1621_162156

theorem remainder_of_double_division (x : ℝ) : 
  let q₃ := (x^10 - 1) / (x - 1)
  let r₃ := x^10 - (x - 1) * q₃
  let q₄ := (q₃ - r₃) / (x - 1)
  let r₄ := q₃ - (x - 1) * q₄
  r₄ = 10 := by sorry

end NUMINAMATH_CALUDE_remainder_of_double_division_l1621_162156


namespace NUMINAMATH_CALUDE_scientific_notation_of_240000_l1621_162136

theorem scientific_notation_of_240000 : 
  240000 = 2.4 * (10 ^ 5) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_240000_l1621_162136


namespace NUMINAMATH_CALUDE_coin_and_die_probability_l1621_162196

/-- Probability of getting heads on a biased coin -/
def prob_heads : ℚ := 2/3

/-- Probability of getting an even number on a regular six-sided die -/
def prob_even_die : ℚ := 1/2

/-- Theorem: The probability of getting heads on a biased coin with 2/3 probability for heads
    and an even number on a regular six-sided die is 1/3 -/
theorem coin_and_die_probability :
  prob_heads * prob_even_die = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_coin_and_die_probability_l1621_162196


namespace NUMINAMATH_CALUDE_sin_780_degrees_l1621_162148

theorem sin_780_degrees : Real.sin (780 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_780_degrees_l1621_162148


namespace NUMINAMATH_CALUDE_equation_solution_l1621_162100

theorem equation_solution : ∃! x : ℚ, (9 - x)^2 = (x + 1/2)^2 ∧ x = 323/76 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1621_162100


namespace NUMINAMATH_CALUDE_janice_earnings_l1621_162183

/-- Represents Janice's work schedule and earnings --/
structure WorkSchedule where
  regularDays : ℕ
  regularPayPerDay : ℕ
  overtimeShifts : ℕ
  overtimePay : ℕ

/-- Calculates the total earnings for the week --/
def totalEarnings (schedule : WorkSchedule) : ℕ :=
  schedule.regularDays * schedule.regularPayPerDay + schedule.overtimeShifts * schedule.overtimePay

/-- Janice's work schedule for the week --/
def janiceSchedule : WorkSchedule :=
  { regularDays := 5
  , regularPayPerDay := 30
  , overtimeShifts := 3
  , overtimePay := 15 }

/-- Theorem stating that Janice's total earnings for the week equal $195 --/
theorem janice_earnings : totalEarnings janiceSchedule = 195 := by
  sorry

end NUMINAMATH_CALUDE_janice_earnings_l1621_162183


namespace NUMINAMATH_CALUDE_inequality_proof_l1621_162191

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_product : a * b * c = 1) : 
  a^2 + b^2 + c^2 + 3 ≥ 2 * (1/a + 1/b + 1/c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1621_162191


namespace NUMINAMATH_CALUDE_no_country_with_100_roads_and_3_per_city_l1621_162164

theorem no_country_with_100_roads_and_3_per_city :
  ¬ ∃ (n : ℕ), 3 * n = 200 :=
by sorry

end NUMINAMATH_CALUDE_no_country_with_100_roads_and_3_per_city_l1621_162164


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l1621_162179

theorem smallest_number_of_eggs (n : ℕ) (c : ℕ) : 
  n > 150 →
  n = 15 * c - 6 →
  c ≥ 11 →
  (∀ m : ℕ, m > 150 ∧ (∃ k : ℕ, m = 15 * k - 6) → m ≥ n) →
  n = 159 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l1621_162179


namespace NUMINAMATH_CALUDE_mixture_problem_l1621_162158

/-- A mixture problem involving milk and water ratios -/
theorem mixture_problem (x : ℝ) (h1 : x > 0) : 
  (4 * x) / x = 4 →                  -- Initial ratio of milk to water is 4:1
  (4 * x) / (x + 9) = 2 →            -- Final ratio after adding 9 litres of water is 2:1
  5 * x = 45 :=                      -- Initial volume of the mixture is 45 litres
by
  sorry

end NUMINAMATH_CALUDE_mixture_problem_l1621_162158


namespace NUMINAMATH_CALUDE_product_25_sum_0_l1621_162195

theorem product_25_sum_0 (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
  a * b * c * d = 25 → 
  a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_product_25_sum_0_l1621_162195


namespace NUMINAMATH_CALUDE_min_value_of_f_l1621_162135

/-- The quadratic function f(x) = 3(x+2)^2 - 5 -/
def f (x : ℝ) : ℝ := 3 * (x + 2)^2 - 5

/-- The minimum value of f(x) is -5 -/
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ -5 ∧ ∃ x₀ : ℝ, f x₀ = -5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1621_162135


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1621_162180

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x, x^2 + a*x - 3 ≤ 0 ↔ -1 ≤ x ∧ x ≤ 3) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1621_162180


namespace NUMINAMATH_CALUDE_clock_chime_theorem_l1621_162143

/-- Represents the number of chimes at a given time -/
def num_chimes (hour : ℕ) (minute : ℕ) : ℕ :=
  if minute = 0 then hour % 12
  else if minute = 30 then 1
  else 0

/-- Represents a sequence of four consecutive chimes -/
def chime_sequence (start_hour : ℕ) (start_minute : ℕ) : Prop :=
  num_chimes start_hour start_minute = 1 ∧
  num_chimes ((start_hour + (start_minute + 30) / 60) % 24) ((start_minute + 30) % 60) = 1 ∧
  num_chimes ((start_hour + (start_minute + 60) / 60) % 24) ((start_minute + 60) % 60) = 1 ∧
  num_chimes ((start_hour + (start_minute + 90) / 60) % 24) ((start_minute + 90) % 60) = 1

theorem clock_chime_theorem :
  ∀ (start_hour : ℕ) (start_minute : ℕ),
    chime_sequence start_hour start_minute →
    start_hour = 12 ∧ start_minute = 0 :=
by sorry

end NUMINAMATH_CALUDE_clock_chime_theorem_l1621_162143


namespace NUMINAMATH_CALUDE_power_of_product_cube_l1621_162163

theorem power_of_product_cube (m : ℝ) : (2 * m^2)^3 = 8 * m^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_cube_l1621_162163


namespace NUMINAMATH_CALUDE_square_sum_of_xy_l1621_162115

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 119)
  (h2 : x^2 * y + x * y^2 = 1680) : 
  x^2 + y^2 = 1057 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_xy_l1621_162115


namespace NUMINAMATH_CALUDE_right_triangle_probability_l1621_162155

/-- A 3x3 grid with 16 vertices -/
structure Grid :=
  (vertices : Finset (ℕ × ℕ))
  (is_3x3 : vertices.card = 16)

/-- Three vertices from the grid -/
structure TripleOfVertices (g : Grid) :=
  (v₁ v₂ v₃ : ℕ × ℕ)
  (v₁_in : v₁ ∈ g.vertices)
  (v₂_in : v₂ ∈ g.vertices)
  (v₃_in : v₃ ∈ g.vertices)
  (distinct : v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₂ ≠ v₃)

/-- Predicate to check if three vertices form a right triangle -/
def is_right_triangle (t : TripleOfVertices g) : Prop :=
  sorry

/-- The probability of forming a right triangle -/
def probability_right_triangle (g : Grid) : ℚ :=
  sorry

/-- The main theorem -/
theorem right_triangle_probability (g : Grid) :
  probability_right_triangle g = 9 / 35 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_probability_l1621_162155


namespace NUMINAMATH_CALUDE_square_perimeter_increase_l1621_162151

theorem square_perimeter_increase (s : ℝ) : 
  (s + 2) * 4 - s * 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_increase_l1621_162151


namespace NUMINAMATH_CALUDE_volunteer_selection_l1621_162182

theorem volunteer_selection (n : ℕ) (h : n = 5) : 
  (n.choose 1) * ((n - 1).choose 1 * (n - 2).choose 1) = 60 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_selection_l1621_162182


namespace NUMINAMATH_CALUDE_binomial_fraction_is_integer_l1621_162105

theorem binomial_fraction_is_integer (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) : 
  ∃ m : ℤ, (n - 2*k - 1 : ℚ) / (k + 1 : ℚ) * (n.choose k) = m := by
  sorry

end NUMINAMATH_CALUDE_binomial_fraction_is_integer_l1621_162105


namespace NUMINAMATH_CALUDE_cubic_inequality_l1621_162118

theorem cubic_inequality (a b c : ℝ) (h : ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  ∀ t : ℝ, t^3 + a*t^2 + b*t + c = 0 ↔ t = x ∨ t = y ∨ t = z) : 
  2*a^3 + 9*c ≤ 7*a*b ∧ 
  (2*a^3 + 9*c = 7*a*b ↔ ∃ r : ℝ, r > 0 ∧ ∀ t : ℝ, t^3 + a*t^2 + b*t + c = 0 ↔ t = r) :=
sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1621_162118


namespace NUMINAMATH_CALUDE_dot_product_example_l1621_162186

theorem dot_product_example : 
  let v1 : Fin 2 → ℝ := ![3, -2]
  let v2 : Fin 2 → ℝ := ![-5, 7]
  Finset.sum (Finset.range 2) (λ i => v1 i * v2 i) = -29 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_example_l1621_162186


namespace NUMINAMATH_CALUDE_boys_in_class_l1621_162113

theorem boys_in_class (total : ℕ) (girls_fraction : ℚ) (boys : ℕ) : 
  total = 160 → girls_fraction = 1 / 4 → boys = 120 → 
  boys = total * (1 - girls_fraction) := by
sorry

end NUMINAMATH_CALUDE_boys_in_class_l1621_162113


namespace NUMINAMATH_CALUDE_petya_more_likely_to_win_petya_wins_in_game_l1621_162133

/-- Represents a game between Petya and Vasya with two boxes of candies. -/
structure CandyGame where
  total_candies : ℕ
  prob_two_caramels : ℝ

/-- Defines the game setup with the given conditions. -/
def game : CandyGame :=
  { total_candies := 25,
    prob_two_caramels := 0.54 }

/-- Calculates the probability of Vasya winning (getting two chocolate candies). -/
def prob_vasya_wins (g : CandyGame) : ℝ :=
  1 - g.prob_two_caramels

/-- Theorem stating that Petya has a higher chance of winning than Vasya. -/
theorem petya_more_likely_to_win (g : CandyGame) :
  prob_vasya_wins g < 1 - prob_vasya_wins g :=
by sorry

/-- Corollary proving that Petya has a higher chance of winning in the specific game setup. -/
theorem petya_wins_in_game : prob_vasya_wins game < 1 - prob_vasya_wins game :=
by sorry

end NUMINAMATH_CALUDE_petya_more_likely_to_win_petya_wins_in_game_l1621_162133


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1621_162194

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1621_162194


namespace NUMINAMATH_CALUDE_tims_change_l1621_162190

def initial_amount : ℚ := 1.50
def candy_cost : ℚ := 0.45
def chips_cost : ℚ := 0.65
def toy_cost : ℚ := 0.40
def discount_rate : ℚ := 0.10

def total_snacks_cost : ℚ := candy_cost + chips_cost
def discounted_snacks_cost : ℚ := total_snacks_cost * (1 - discount_rate)
def total_cost : ℚ := discounted_snacks_cost + toy_cost
def change : ℚ := initial_amount - total_cost

theorem tims_change : change = 0.11 := by sorry

end NUMINAMATH_CALUDE_tims_change_l1621_162190


namespace NUMINAMATH_CALUDE_sum_smallest_largest_primes_1_to_50_l1621_162125

theorem sum_smallest_largest_primes_1_to_50 :
  (∃ p q : ℕ, 
    Prime p ∧ Prime q ∧
    1 < p ∧ p ≤ 50 ∧
    1 < q ∧ q ≤ 50 ∧
    (∀ r : ℕ, Prime r ∧ 1 < r ∧ r ≤ 50 → p ≤ r ∧ r ≤ q) ∧
    p + q = 49) :=
by sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_primes_1_to_50_l1621_162125


namespace NUMINAMATH_CALUDE_randy_store_trips_l1621_162107

/-- The number of trips Randy makes to the store each month -/
def trips_per_month (initial_amount : ℕ) (amount_per_trip : ℕ) (remaining_amount : ℕ) (months_per_year : ℕ) : ℕ :=
  ((initial_amount - remaining_amount) / amount_per_trip) / months_per_year

/-- Proof that Randy makes 4 trips to the store each month -/
theorem randy_store_trips :
  trips_per_month 200 2 104 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_randy_store_trips_l1621_162107


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1621_162103

/-- The number of games in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1)

/-- The number of players in the tournament -/
def num_players : ℕ := 20

/-- Each game is played twice -/
def games_per_pair : ℕ := 2

theorem chess_tournament_games :
  num_games num_players * games_per_pair = 760 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1621_162103


namespace NUMINAMATH_CALUDE_amy_bought_seven_pencils_l1621_162189

/-- Represents the number of pencils Amy had initially -/
def initial_pencils : ℕ := 3

/-- Represents the total number of pencils Amy has now -/
def total_pencils : ℕ := 10

/-- Represents the number of pencils Amy bought at the school store -/
def bought_pencils : ℕ := total_pencils - initial_pencils

theorem amy_bought_seven_pencils : bought_pencils = 7 := by
  sorry

end NUMINAMATH_CALUDE_amy_bought_seven_pencils_l1621_162189


namespace NUMINAMATH_CALUDE_complex_combination_equality_l1621_162165

/-- Given complex numbers A, M, S, P, and Q, prove that their combination equals 6 - 5i -/
theorem complex_combination_equality (A M S P Q : ℂ) : 
  A = 5 - 4*I ∧ 
  M = -5 + 2*I ∧ 
  S = 2*I ∧ 
  P = 3 ∧ 
  Q = 1 + I → 
  A - M + S - P - Q = 6 - 5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_combination_equality_l1621_162165


namespace NUMINAMATH_CALUDE_hidden_faces_sum_l1621_162139

def standard_die_sum : ℕ := 21

def visible_faces : List ℕ := [1, 1, 2, 2, 3, 3, 4, 5, 6, 6]

def total_faces : ℕ := 24

theorem hidden_faces_sum (num_dice : ℕ) (h1 : num_dice = 4) :
  num_dice * standard_die_sum - visible_faces.sum = 51 := by
  sorry

end NUMINAMATH_CALUDE_hidden_faces_sum_l1621_162139


namespace NUMINAMATH_CALUDE_triangle_count_2008_l1621_162130

/-- Given a set of points in a plane, where three of the points form a triangle
    and the rest are inside this triangle, this function calculates the number
    of non-overlapping small triangles that can be formed. -/
def count_small_triangles (n : ℕ) : ℕ :=
  1 + 2 * (n - 3)

/-- Theorem stating that for 2008 non-collinear points, where 3 form a triangle
    and the rest are inside, the number of non-overlapping small triangles is 4011. -/
theorem triangle_count_2008 :
  count_small_triangles 2008 = 4011 := by
  sorry

#eval count_small_triangles 2008  -- Should output 4011

end NUMINAMATH_CALUDE_triangle_count_2008_l1621_162130


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1621_162188

theorem sqrt_product_simplification (x : ℝ) (h : x > 0) :
  Real.sqrt (100 * x) * Real.sqrt (3 * x) * Real.sqrt (18 * x) = 30 * x * Real.sqrt (6 * x) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1621_162188


namespace NUMINAMATH_CALUDE_solve_for_e_l1621_162132

theorem solve_for_e (x e : ℝ) (h1 : (10 * x + 2) / 4 - (3 * x - e) / 18 = (2 * x + 4) / 3)
                     (h2 : x = 0.3) : e = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_e_l1621_162132


namespace NUMINAMATH_CALUDE_cos_negative_nineteen_pi_sixths_l1621_162184

theorem cos_negative_nineteen_pi_sixths :
  Real.cos (-19 * π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_nineteen_pi_sixths_l1621_162184


namespace NUMINAMATH_CALUDE_distribute_planets_l1621_162181

/-- The number of ways to distribute units among distinct objects --/
def distribute_units (total_units : ℕ) (earth_like : ℕ) (mars_like : ℕ) (earth_units : ℕ) (mars_units : ℕ) : ℕ :=
  sorry

theorem distribute_planets :
  distribute_units 15 7 8 3 1 = 2961 :=
sorry

end NUMINAMATH_CALUDE_distribute_planets_l1621_162181


namespace NUMINAMATH_CALUDE_sunglasses_cap_probability_l1621_162137

theorem sunglasses_cap_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (total_hats : ℕ) 
  (prob_cap_and_sunglasses : ℚ) 
  (h1 : total_sunglasses = 120) 
  (h2 : total_caps = 84) 
  (h3 : total_hats = 60) 
  (h4 : prob_cap_and_sunglasses = 3 / 7) : 
  (prob_cap_and_sunglasses * total_caps) / total_sunglasses = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_sunglasses_cap_probability_l1621_162137


namespace NUMINAMATH_CALUDE_quadratic_increases_iff_l1621_162146

/-- The quadratic function y = 2x^2 - 4x - 1 increases for x > a iff a ≥ 1 -/
theorem quadratic_increases_iff (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ > a ∧ x₂ > x₁ → (2*x₂^2 - 4*x₂ - 1) > (2*x₁^2 - 4*x₁ - 1)) ↔ 
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_increases_iff_l1621_162146


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1621_162197

theorem other_root_of_quadratic (b : ℝ) : 
  (1 : ℝ)^2 + b*(1 : ℝ) - 2 = 0 → 
  ∃ (x : ℝ), x ≠ 1 ∧ x^2 + b*x - 2 = 0 ∧ x = -2 := by
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1621_162197


namespace NUMINAMATH_CALUDE_max_time_at_8_l1621_162138

noncomputable def y (t : ℝ) : ℝ := -1/8 * t^3 - 3/4 * t^2 + 36*t - 629/4

theorem max_time_at_8 :
  ∃ (t_max : ℝ), t_max = 8 ∧
  ∀ (t : ℝ), 6 ≤ t ∧ t ≤ 9 → y t ≤ y t_max :=
by sorry

end NUMINAMATH_CALUDE_max_time_at_8_l1621_162138


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1621_162149

theorem complex_number_in_third_quadrant :
  let z : ℂ := (1 - Complex.I)^2 / (1 + Complex.I)
  ∃ (a b : ℝ), z = Complex.mk a b ∧ a < 0 ∧ b < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1621_162149


namespace NUMINAMATH_CALUDE_min_comparisons_correct_l1621_162174

/-- Represents a comparison between two numbers -/
structure Comparison where
  a : ℕ
  b : ℕ

/-- The minimum number of comparisons needed to find both max and min in a sequence -/
def min_comparisons (n : ℕ) : ℕ := 3 * n - 2

/-- A sequence of 2n numbers -/
def Sequence (n : ℕ) : Type := Fin (2 * n) → ℕ

/-- A function that finds both max and min in a sequence -/
def find_max_min (n : ℕ) (seq : Sequence n) : ℕ × ℕ := sorry

/-- The actual number of comparisons made by find_max_min -/
def actual_comparisons (n : ℕ) (seq : Sequence n) : ℕ := sorry

theorem min_comparisons_correct (n : ℕ) :
  ∀ (seq : Sequence n),
    actual_comparisons n seq ≥ min_comparisons n ∧
    ∃ (alg : Sequence n → ℕ × ℕ),
      (∀ (seq : Sequence n), alg seq = find_max_min n seq) ∧
      (∀ (seq : Sequence n), actual_comparisons n seq ≤ min_comparisons n) :=
sorry

end NUMINAMATH_CALUDE_min_comparisons_correct_l1621_162174


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_quadratic_l1621_162173

theorem negation_of_universal_positive_quadratic :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_quadratic_l1621_162173


namespace NUMINAMATH_CALUDE_expression_evaluation_l1621_162145

theorem expression_evaluation (a b c : ℚ) 
  (h1 : c = b - 10)
  (h2 : b = a + 2)
  (h3 : a = 4)
  (h4 : a + 1 ≠ 0)
  (h5 : b - 2 ≠ 0)
  (h6 : c + 6 ≠ 0) :
  (a + 2) / (a + 1) * (b - 1) / (b - 2) * (c + 8) / (c + 6) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1621_162145


namespace NUMINAMATH_CALUDE_age_sum_problem_l1621_162140

theorem age_sum_problem (leonard_age nina_age jerome_age : ℕ) : 
  leonard_age = 6 →
  nina_age = leonard_age + 4 →
  jerome_age = 2 * nina_age →
  leonard_age + nina_age + jerome_age = 36 := by
sorry

end NUMINAMATH_CALUDE_age_sum_problem_l1621_162140


namespace NUMINAMATH_CALUDE_shaded_area_sum_l1621_162121

def circle_setup (r₁ r₂ r₃ : ℝ) : Prop :=
  r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
  r₁ * r₁ = 100 ∧
  r₂ = r₁ / 2 ∧
  r₃ = r₂ / 2

theorem shaded_area_sum (r₁ r₂ r₃ : ℝ) 
  (h : circle_setup r₁ r₂ r₃) : 
  (π * r₁ * r₁ / 2) + (π * r₂ * r₂ / 2) + (π * r₃ * r₃ / 2) = 65.625 * π :=
by
  sorry

#check shaded_area_sum

end NUMINAMATH_CALUDE_shaded_area_sum_l1621_162121


namespace NUMINAMATH_CALUDE_arrangement_count_l1621_162185

def number_of_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  Nat.choose n k * Nat.choose m k * Nat.factorial (n + m - 2 * k)

theorem arrangement_count :
  let total_people : ℕ := 6
  let people_per_row : ℕ := 3
  number_of_arrangements total_people people_per_row 2 = 216 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l1621_162185


namespace NUMINAMATH_CALUDE_combined_cube_volume_l1621_162161

theorem combined_cube_volume : 
  let lily_cubes := 4
  let lily_side_length := 3
  let mark_cubes := 3
  let mark_side_length := 4
  let zoe_cubes := 2
  let zoe_side_length := 5
  lily_cubes * lily_side_length^3 + 
  mark_cubes * mark_side_length^3 + 
  zoe_cubes * zoe_side_length^3 = 550 := by
sorry

end NUMINAMATH_CALUDE_combined_cube_volume_l1621_162161


namespace NUMINAMATH_CALUDE_unwashed_shirts_l1621_162127

theorem unwashed_shirts 
  (short_sleeve : ℕ) 
  (long_sleeve : ℕ) 
  (washed : ℕ) 
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 27)
  (h3 : washed = 20) : 
  short_sleeve + long_sleeve - washed = 16 := by
  sorry

end NUMINAMATH_CALUDE_unwashed_shirts_l1621_162127


namespace NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_l1621_162159

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the parabola -/
theorem parabola_equation_from_hyperbola (x y : ℝ) :
  (x^2 / 3 - y^2 = 1) →  -- Given hyperbola equation
  (∃ (p : ℝ), 
    (p > 0) ∧  -- p is positive for a right-opening parabola
    ((2 : ℝ) = p / 2) ∧  -- Focus of parabola is at (2, 0), which is (p/2, 0) in standard form
    (y^2 = 2 * p * x))  -- Standard form of parabola equation
  →
  y^2 = 8 * x  -- Conclusion: specific equation of the parabola
:= by sorry

end NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_l1621_162159


namespace NUMINAMATH_CALUDE_minimum_cost_theorem_l1621_162117

/-- The unit price of a volleyball in yuan -/
def volleyball_price : ℝ := 50

/-- The unit price of a soccer ball in yuan -/
def soccer_ball_price : ℝ := 80

/-- The total number of balls to be purchased -/
def total_balls : ℕ := 11

/-- The minimum number of soccer balls to be purchased -/
def min_soccer_balls : ℕ := 2

/-- The cost function for purchasing x volleyballs -/
def cost_function (x : ℝ) : ℝ := -30 * x + 880

/-- The theorem stating the minimum cost of purchasing the balls -/
theorem minimum_cost_theorem :
  ∃ (x : ℝ), 
    0 ≤ x ∧ 
    x ≤ total_balls - min_soccer_balls ∧
    ∀ (y : ℝ), 0 ≤ y ∧ y ≤ total_balls - min_soccer_balls → 
      cost_function x ≤ cost_function y ∧
      cost_function x = 610 :=
sorry

end NUMINAMATH_CALUDE_minimum_cost_theorem_l1621_162117


namespace NUMINAMATH_CALUDE_nickel_count_proof_l1621_162120

/-- Represents the number of nickels in a collection of coins -/
def number_of_nickels (total_value : ℚ) (total_coins : ℕ) : ℕ :=
  2

/-- The value of a dime in dollars -/
def dime_value : ℚ := 1/10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 1/20

theorem nickel_count_proof (total_value : ℚ) (total_coins : ℕ) 
  (h1 : total_value = 7/10) 
  (h2 : total_coins = 8) :
  number_of_nickels total_value total_coins = 2 ∧ 
  ∃ (d n : ℕ), d + n = total_coins ∧ 
               d * dime_value + n * nickel_value = total_value :=
by
  sorry

#check nickel_count_proof

end NUMINAMATH_CALUDE_nickel_count_proof_l1621_162120


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1621_162131

theorem quadratic_root_difference (x : ℝ) : 
  5 * x^2 - 9 * x - 22 = 0 →
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
    (5 * r₁^2 - 9 * r₁ - 22 = 0) ∧
    (5 * r₂^2 - 9 * r₂ - 22 = 0) ∧
    |r₁ - r₂| = Real.sqrt 521 / 5 ∧
    (∀ (p : ℕ), p > 1 → ¬(p^2 ∣ 521)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1621_162131


namespace NUMINAMATH_CALUDE_fourth_year_exam_count_l1621_162167

/-- Represents the number of exams taken in each year -/
structure ExamCount where
  year1 : ℕ
  year2 : ℕ
  year3 : ℕ
  year4 : ℕ
  year5 : ℕ

/-- Conditions for the exam count problem -/
def ValidExamCount (e : ExamCount) : Prop :=
  e.year1 + e.year2 + e.year3 + e.year4 + e.year5 = 31 ∧
  e.year1 < e.year2 ∧ e.year2 < e.year3 ∧ e.year3 < e.year4 ∧ e.year4 < e.year5 ∧
  e.year5 = 3 * e.year1

/-- The theorem stating that if the exam count is valid, the fourth year must have 8 exams -/
theorem fourth_year_exam_count (e : ExamCount) : ValidExamCount e → e.year4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_year_exam_count_l1621_162167


namespace NUMINAMATH_CALUDE_range_of_g_l1621_162152

noncomputable def g (x : ℝ) : ℝ := 1 / x^2 + 3

theorem range_of_g :
  Set.range g = Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l1621_162152


namespace NUMINAMATH_CALUDE_order_of_abc_l1621_162193

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem order_of_abc : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l1621_162193


namespace NUMINAMATH_CALUDE_total_profit_is_100_l1621_162110

/-- Calculates the total profit given investments and A's profit share -/
def calculate_total_profit (a_investment : ℕ) (a_months : ℕ) (b_investment : ℕ) (b_months : ℕ) (a_profit_share : ℕ) : ℕ :=
  let a_investment_share := a_investment * a_months
  let b_investment_share := b_investment * b_months
  let total_investment_share := a_investment_share + b_investment_share
  let total_profit := a_profit_share * total_investment_share / a_investment_share
  total_profit

/-- Theorem stating that given the specified investments and A's profit share, the total profit is 100 -/
theorem total_profit_is_100 :
  calculate_total_profit 300 12 200 6 75 = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_100_l1621_162110


namespace NUMINAMATH_CALUDE_temperature_difference_l1621_162102

theorem temperature_difference (highest lowest : Int) 
  (h1 : highest = 8) 
  (h2 : lowest = -2) : 
  highest - lowest = 10 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l1621_162102


namespace NUMINAMATH_CALUDE_cube_volume_proof_l1621_162114

theorem cube_volume_proof (n : ℕ) (m : ℕ) : 
  (n^3 = 98 + m^3) ∧ 
  (m ≠ 1) ∧ 
  (∃ (k : ℕ), n^3 = 99 * k) →
  n^3 = 125 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_proof_l1621_162114
