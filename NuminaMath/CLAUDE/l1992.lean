import Mathlib

namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l1992_199273

/-- Represents the number of varieties of wrapping paper -/
def wrapping_paper_varieties : ℕ := 10

/-- Represents the number of colors of ribbon -/
def ribbon_colors : ℕ := 3

/-- Represents the number of types of gift cards -/
def gift_card_types : ℕ := 4

/-- Represents the number of designs of gift tags -/
def gift_tag_designs : ℕ := 5

/-- Calculates the total number of gift wrapping combinations -/
def total_combinations : ℕ := wrapping_paper_varieties * ribbon_colors * gift_card_types * gift_tag_designs

/-- Theorem stating that the total number of gift wrapping combinations is 600 -/
theorem gift_wrapping_combinations : total_combinations = 600 := by sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l1992_199273


namespace NUMINAMATH_CALUDE_total_students_l1992_199247

/-- Given a student's position from right and left in a line, calculate the total number of students -/
theorem total_students (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 6)
  (h2 : rank_from_left = 5) :
  rank_from_right + rank_from_left - 1 = 10 := by
  sorry

#check total_students

end NUMINAMATH_CALUDE_total_students_l1992_199247


namespace NUMINAMATH_CALUDE_first_bus_students_l1992_199228

theorem first_bus_students (total_buses : ℕ) (initial_avg : ℕ) (remaining_avg : ℕ) : 
  total_buses = 6 → 
  initial_avg = 28 → 
  remaining_avg = 26 → 
  (total_buses * initial_avg - (total_buses - 1) * remaining_avg) = 38 := by
sorry

end NUMINAMATH_CALUDE_first_bus_students_l1992_199228


namespace NUMINAMATH_CALUDE_union_of_sets_l1992_199250

theorem union_of_sets :
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {2, 4, 6}
  A ∪ B = {1, 2, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1992_199250


namespace NUMINAMATH_CALUDE_power_of_three_mod_thousand_l1992_199231

theorem power_of_three_mod_thousand :
  ∃ n : ℕ, n < 1000 ∧ 3^5000 ≡ n [MOD 1000] :=
sorry

end NUMINAMATH_CALUDE_power_of_three_mod_thousand_l1992_199231


namespace NUMINAMATH_CALUDE_basketball_game_price_l1992_199251

/-- The cost of Joan's video game purchase -/
def total_cost : ℝ := 9.43

/-- The cost of the racing game -/
def racing_game_cost : ℝ := 4.23

/-- The cost of the basketball game -/
def basketball_game_cost : ℝ := total_cost - racing_game_cost

theorem basketball_game_price : basketball_game_cost = 5.20 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_price_l1992_199251


namespace NUMINAMATH_CALUDE_delay_calculation_cottage_to_station_delay_l1992_199256

theorem delay_calculation (usual_time : ℝ) (speed_increase : ℝ) (lateness : ℝ) : ℝ :=
  let normal_distance := usual_time
  let increased_speed_time := normal_distance / speed_increase
  let total_time := increased_speed_time - lateness
  usual_time - total_time

theorem cottage_to_station_delay : delay_calculation 18 1.2 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_delay_calculation_cottage_to_station_delay_l1992_199256


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1992_199270

-- Define the concept of opposite for integers
def opposite (n : ℤ) : ℤ := -n

-- Theorem statement
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1992_199270


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1992_199280

theorem polynomial_simplification (x : ℝ) : 
  (5 * x^10 + 8 * x^9 + 2 * x^8) + (3 * x^10 + x^9 + 4 * x^8 + 7 * x^4 + 6 * x + 9) = 
  8 * x^10 + 9 * x^9 + 6 * x^8 + 7 * x^4 + 6 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1992_199280


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l1992_199274

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : EvenFunction f) :
  EvenFunction (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l1992_199274


namespace NUMINAMATH_CALUDE_least_positive_linear_combination_l1992_199205

theorem least_positive_linear_combination (x y : ℤ) : 
  ∃ (a b : ℤ), 24 * a + 18 * b = 6 ∧ 
  ∀ (c d : ℤ), 24 * c + 18 * d > 0 → 24 * c + 18 * d ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_linear_combination_l1992_199205


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_range_l1992_199284

def integer_range : List Int := List.range 14 |>.map (fun i => i - 6)

theorem arithmetic_mean_of_range : 
  (integer_range.sum : ℚ) / integer_range.length = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_range_l1992_199284


namespace NUMINAMATH_CALUDE_temperature_data_inconsistency_l1992_199230

theorem temperature_data_inconsistency (x_bar m S_squared : ℝ) 
  (h1 : x_bar = 0)
  (h2 : m = 4)
  (h3 : S_squared = 15.917)
  : |x_bar - m| > Real.sqrt S_squared := by
  sorry

end NUMINAMATH_CALUDE_temperature_data_inconsistency_l1992_199230


namespace NUMINAMATH_CALUDE_retail_markup_percentage_l1992_199220

theorem retail_markup_percentage 
  (wholesale : ℝ) 
  (retail : ℝ) 
  (h1 : retail > 0) 
  (h2 : wholesale > 0) 
  (h3 : retail * 0.75 = wholesale * 1.3500000000000001) : 
  (retail / wholesale - 1) * 100 = 80.00000000000002 := by
sorry

end NUMINAMATH_CALUDE_retail_markup_percentage_l1992_199220


namespace NUMINAMATH_CALUDE_print_time_calculation_l1992_199246

/-- Represents a printer with warm-up time and printing speed -/
structure Printer where
  warmupTime : ℕ
  pagesPerMinute : ℕ

/-- Calculates the total time required to print a given number of pages -/
def totalPrintTime (printer : Printer) (pages : ℕ) : ℕ :=
  printer.warmupTime + (pages + printer.pagesPerMinute - 1) / printer.pagesPerMinute

theorem print_time_calculation (printer : Printer) (pages : ℕ) :
  printer.warmupTime = 2 →
  printer.pagesPerMinute = 15 →
  pages = 225 →
  totalPrintTime printer pages = 17 :=
by
  sorry

#eval totalPrintTime ⟨2, 15⟩ 225

end NUMINAMATH_CALUDE_print_time_calculation_l1992_199246


namespace NUMINAMATH_CALUDE_leaders_photo_theorem_l1992_199259

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := n.factorial

/-- The number of ways to choose k objects from n distinct objects and arrange them. -/
def arrangements (n k : ℕ) : ℕ := 
  if k ≤ n then (permutations n) / (permutations (n - k)) else 0

/-- The number of arrangements for the leaders' photo. -/
def leaders_photo_arrangements : ℕ := 
  (arrangements 2 1) * (arrangements 18 18)

theorem leaders_photo_theorem : 
  leaders_photo_arrangements = (arrangements 2 1) * (arrangements 18 18) := by
  sorry

end NUMINAMATH_CALUDE_leaders_photo_theorem_l1992_199259


namespace NUMINAMATH_CALUDE_expression_value_l1992_199269

theorem expression_value (x y : ℝ) (h : x^2 - 4*x - 1 = 0) :
  (2*x - 3)^2 - (x + y)*(x - y) - y^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1992_199269


namespace NUMINAMATH_CALUDE_pants_cost_l1992_199253

theorem pants_cost (total_cost : ℕ) (tshirt_cost : ℕ) (num_tshirts : ℕ) (num_pants : ℕ) :
  total_cost = 1500 →
  tshirt_cost = 100 →
  num_tshirts = 5 →
  num_pants = 4 →
  (total_cost - num_tshirts * tshirt_cost) / num_pants = 250 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_l1992_199253


namespace NUMINAMATH_CALUDE_max_xy_over_x2_plus_y2_l1992_199279

theorem max_xy_over_x2_plus_y2 (x y : ℝ) 
  (hx : 1/3 ≤ x ∧ x ≤ 3/5) (hy : 1/4 ≤ y ∧ y ≤ 1/2) :
  (x * y) / (x^2 + y^2) ≤ 6/13 :=
sorry

end NUMINAMATH_CALUDE_max_xy_over_x2_plus_y2_l1992_199279


namespace NUMINAMATH_CALUDE_neither_odd_nor_even_and_increasing_l1992_199262

-- Define the function f(x) = |x + 1|
def f (x : ℝ) : ℝ := |x + 1|

-- State the theorem
theorem neither_odd_nor_even_and_increasing :
  (¬ ∀ x, f (-x) = -f x) ∧  -- not odd
  (¬ ∀ x, f (-x) = f x) ∧  -- not even
  (∀ x y, 0 < x → x < y → f x < f y) -- monotonically increasing on (0, +∞)
  := by sorry

end NUMINAMATH_CALUDE_neither_odd_nor_even_and_increasing_l1992_199262


namespace NUMINAMATH_CALUDE_charlyn_visible_area_l1992_199210

/-- The length of one side of the square in kilometers -/
def square_side : ℝ := 5

/-- The visibility range in kilometers -/
def visibility_range : ℝ := 1

/-- The area of the region Charlyn can see during her walk -/
noncomputable def visible_area : ℝ :=
  (square_side + 2 * visibility_range) ^ 2 - (square_side - 2 * visibility_range) ^ 2 + Real.pi * visibility_range ^ 2

theorem charlyn_visible_area :
  ‖visible_area - 43.14‖ < 0.01 :=
sorry

end NUMINAMATH_CALUDE_charlyn_visible_area_l1992_199210


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1992_199278

/-- An arithmetic sequence with given terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 1 - seq.a 0

theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h2 : seq.a 2 = 12)
  (h6 : seq.a 6 = 4) :
  common_difference seq = -2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1992_199278


namespace NUMINAMATH_CALUDE_symmetric_line_theorem_l1992_199208

/-- Given a line with equation ax + by + c = 0, 
    returns the equation of the line symmetric to it with respect to the x-axis -/
def symmetricLineEquation (a b c : ℝ) : ℝ × ℝ × ℝ := (a, -b, c)

/-- Proves that the equation of the line symmetric to 2x-y+4=0 
    with respect to the x-axis is 2x+y+4=0 -/
theorem symmetric_line_theorem :
  let original := (2, -1, 4)
  let symmetric := symmetricLineEquation 2 (-1) 4
  symmetric = (2, 1, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_line_theorem_l1992_199208


namespace NUMINAMATH_CALUDE_sound_distance_at_18C_l1992_199267

/-- Represents the speed of sound in air as a function of temperature -/
def speed_of_sound (t : ℝ) : ℝ := 331 + 0.6 * t

/-- Calculates the distance traveled by sound given time and temperature -/
def distance_traveled (time : ℝ) (temp : ℝ) : ℝ :=
  (speed_of_sound temp) * time

/-- Theorem: The distance traveled by sound in 5 seconds at 18°C is approximately 1709 meters -/
theorem sound_distance_at_18C : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |distance_traveled 5 18 - 1709| < ε :=
sorry

end NUMINAMATH_CALUDE_sound_distance_at_18C_l1992_199267


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1992_199289

theorem smallest_positive_integer_with_remainders : 
  ∃ (x : ℕ), x > 0 ∧ 
  x % 5 = 4 ∧ 
  x % 7 = 6 ∧ 
  x % 8 = 7 ∧ 
  ∀ (y : ℕ), y > 0 → 
    y % 5 = 4 → 
    y % 7 = 6 → 
    y % 8 = 7 → 
    x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1992_199289


namespace NUMINAMATH_CALUDE_linear_function_intersection_l1992_199203

/-- A linear function y = kx + 2 intersects the x-axis at a point 2 units away from the origin if and only if k = ±1 -/
theorem linear_function_intersection (k : ℝ) : 
  (∃ x : ℝ, k * x + 2 = 0 ∧ |x| = 2) ↔ (k = 1 ∨ k = -1) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_intersection_l1992_199203


namespace NUMINAMATH_CALUDE_vitamin_pack_size_l1992_199236

theorem vitamin_pack_size (vitamin_a_pack_size : ℕ) 
  (vitamin_a_packs : ℕ) (vitamin_d_packs : ℕ) : 
  (vitamin_a_pack_size * vitamin_a_packs = 17 * vitamin_d_packs) →  -- Equal quantities condition
  (vitamin_a_pack_size * vitamin_a_packs = 119) →                   -- Smallest number condition
  (∀ x y : ℕ, x * y = 119 → x ≤ vitamin_a_pack_size ∨ y ≤ vitamin_a_packs) →  -- Smallest positive integer values
  vitamin_a_pack_size = 7 :=
by sorry

end NUMINAMATH_CALUDE_vitamin_pack_size_l1992_199236


namespace NUMINAMATH_CALUDE_last_two_digits_of_7_2017_l1992_199204

-- Define a function to get the last two digits of a number
def lastTwoDigits (n : ℕ) : ℕ := n % 100

-- Define the pattern for the last two digits of powers of 7
def powerOf7Pattern (k : ℕ) : ℕ :=
  match k % 4 with
  | 0 => 01
  | 1 => 07
  | 2 => 49
  | 3 => 43
  | _ => 0  -- This case should never occur

theorem last_two_digits_of_7_2017 :
  lastTwoDigits (7^2017) = 07 :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_of_7_2017_l1992_199204


namespace NUMINAMATH_CALUDE_product_mod_fifty_l1992_199201

theorem product_mod_fifty : ∃ m : ℕ, 0 ≤ m ∧ m < 50 ∧ (289 * 673) % 50 = m ∧ m = 47 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_fifty_l1992_199201


namespace NUMINAMATH_CALUDE_smaller_circle_area_l1992_199287

/-- Two externally tangent circles with common tangents -/
structure TangentCircles where
  r : ℝ  -- radius of smaller circle
  center_small : ℝ × ℝ  -- center of smaller circle
  center_large : ℝ × ℝ  -- center of larger circle
  P : ℝ × ℝ  -- point P
  A : ℝ × ℝ  -- point A on smaller circle
  B : ℝ × ℝ  -- point B on larger circle
  externally_tangent : (center_small.1 - center_large.1)^2 + (center_small.2 - center_large.2)^2 = (r + 3*r)^2
  on_smaller_circle : (A.1 - center_small.1)^2 + (A.2 - center_small.2)^2 = r^2
  on_larger_circle : (B.1 - center_large.1)^2 + (B.2 - center_large.2)^2 = (3*r)^2
  PA_tangent : ((P.1 - A.1)*(A.1 - center_small.1) + (P.2 - A.2)*(A.2 - center_small.2))^2 = 
               ((P.1 - A.1)^2 + (P.2 - A.2)^2)*r^2
  AB_tangent : ((A.1 - B.1)*(B.1 - center_large.1) + (A.2 - B.2)*(B.2 - center_large.2))^2 = 
               ((A.1 - B.1)^2 + (A.2 - B.2)^2)*(3*r)^2
  PA_length : (P.1 - A.1)^2 + (P.2 - A.2)^2 = 36
  AB_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36

theorem smaller_circle_area (tc : TangentCircles) : Real.pi * tc.r^2 = 36 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_smaller_circle_area_l1992_199287


namespace NUMINAMATH_CALUDE_max_value_2a_minus_b_l1992_199235

theorem max_value_2a_minus_b :
  ∃ (M : ℝ), M = 2 + Real.sqrt 5 ∧
  (∀ a b : ℝ, a^2 + b^2 - 2*a = 0 → 2*a - b ≤ M) ∧
  (∃ a b : ℝ, a^2 + b^2 - 2*a = 0 ∧ 2*a - b = M) :=
by sorry

end NUMINAMATH_CALUDE_max_value_2a_minus_b_l1992_199235


namespace NUMINAMATH_CALUDE_product_equals_half_l1992_199240

theorem product_equals_half : 8 * 0.25 * 2 * 0.125 = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_half_l1992_199240


namespace NUMINAMATH_CALUDE_complex_equation_solution_existence_l1992_199291

theorem complex_equation_solution_existence :
  ∃ (z : ℂ), z * (z + 2*I) * (z + 4*I) = 4012*I ∧
  ∃ (a b : ℝ), z = a + b*I :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_existence_l1992_199291


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l1992_199276

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ n = 2^p - 1 ∧ is_prime n

theorem largest_mersenne_prime_under_500 :
  (∀ m : ℕ, is_mersenne_prime m → m < 500 → m ≤ 127) ∧
  is_mersenne_prime 127 ∧
  127 < 500 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l1992_199276


namespace NUMINAMATH_CALUDE_largest_class_size_l1992_199207

theorem largest_class_size (total_students : ℕ) (num_classes : ℕ) (diff : ℕ) : 
  total_students = 100 → 
  num_classes = 5 → 
  diff = 2 → 
  (∃ x : ℕ, total_students = x + (x - diff) + (x - 2*diff) + (x - 3*diff) + (x - 4*diff)) → 
  ∃ x : ℕ, x = 24 ∧ total_students = x + (x - diff) + (x - 2*diff) + (x - 3*diff) + (x - 4*diff) :=
by sorry

end NUMINAMATH_CALUDE_largest_class_size_l1992_199207


namespace NUMINAMATH_CALUDE_inequality_proof_l1992_199223

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (h : a * b + b * c + c * d + d * a = 1) : 
  a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (d + a + b) + d^3 / (a + b + c) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1992_199223


namespace NUMINAMATH_CALUDE_gear_system_rotation_l1992_199290

/-- Represents a circular arrangement of gears -/
structure GearSystem where
  n : ℕ  -- number of gears
  circular : Bool  -- true if the arrangement is circular

/-- Defines when a gear system can rotate -/
def can_rotate (gs : GearSystem) : Prop :=
  gs.circular ∧ Even gs.n

/-- Theorem: A circular gear system can rotate if and only if the number of gears is even -/
theorem gear_system_rotation (gs : GearSystem) (h : gs.circular = true) : 
  can_rotate gs ↔ Even gs.n :=
sorry

end NUMINAMATH_CALUDE_gear_system_rotation_l1992_199290


namespace NUMINAMATH_CALUDE_total_lives_for_eight_friends_l1992_199227

/-- Calculates the total number of lives for a group of friends in a video game -/
def totalLives (numFriends : ℕ) (livesPerFriend : ℕ) : ℕ :=
  numFriends * livesPerFriend

/-- Proves that the total number of lives for 8 friends with 8 lives each is 64 -/
theorem total_lives_for_eight_friends : totalLives 8 8 = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_for_eight_friends_l1992_199227


namespace NUMINAMATH_CALUDE_sum_of_floors_even_l1992_199244

theorem sum_of_floors_even (a b c : ℕ+) (h : a^2 + b^2 + 1 = c^2) :
  Even (⌊(a : ℝ) / 2⌋ + ⌊(c : ℝ) / 2⌋) := by sorry

end NUMINAMATH_CALUDE_sum_of_floors_even_l1992_199244


namespace NUMINAMATH_CALUDE_money_problem_l1992_199216

theorem money_problem (p q r s t : ℚ) : 
  p = q + r + 35 →
  q = (2/5) * p →
  r = (1/7) * p →
  s = 2 * p →
  t = (1/2) * (q + r) →
  p + q + r + s + t = 291.03125 := by
sorry

end NUMINAMATH_CALUDE_money_problem_l1992_199216


namespace NUMINAMATH_CALUDE_smallest_multiple_l1992_199258

theorem smallest_multiple : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a % 5 = 0) ∧ 
  ((a + 1) % 7 = 0) ∧ 
  ((a + 2) % 9 = 0) ∧ 
  ((a + 3) % 11 = 0) ∧ 
  (∀ (b : ℕ), b > 0 ∧ 
    (b % 5 = 0) ∧ 
    ((b + 1) % 7 = 0) ∧ 
    ((b + 2) % 9 = 0) ∧ 
    ((b + 3) % 11 = 0) → 
    a ≤ b) ∧
  a = 720 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1992_199258


namespace NUMINAMATH_CALUDE_min_value_theorem_l1992_199218

theorem min_value_theorem (a b : ℝ) (ha : a > 0) 
  (h : ∀ x > 0, (a * x - 1) * (x^2 + b * x - 4) ≥ 0) : 
  (∀ c, b + 2 / a ≥ c) → c = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1992_199218


namespace NUMINAMATH_CALUDE_nelly_outbid_joe_l1992_199296

def joes_bid : ℕ := 160000
def nellys_bid : ℕ := 482000

theorem nelly_outbid_joe : nellys_bid - 3 * joes_bid = 2000 := by
  sorry

end NUMINAMATH_CALUDE_nelly_outbid_joe_l1992_199296


namespace NUMINAMATH_CALUDE_car_speed_comparison_l1992_199206

theorem car_speed_comparison (u v : ℝ) (hu : u > 0) (hv : v > 0) :
  let x := 3 / (1 / u + 2 / v)
  let y := (u + 2 * v) / 3
  x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l1992_199206


namespace NUMINAMATH_CALUDE_sin_600_degrees_l1992_199294

theorem sin_600_degrees : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_degrees_l1992_199294


namespace NUMINAMATH_CALUDE_race_time_difference_l1992_199233

/-- Represents the race scenario with Malcolm and Joshua -/
structure RaceScenario where
  malcolm_speed : ℝ  -- Malcolm's speed in minutes per mile
  joshua_speed : ℝ   -- Joshua's speed in minutes per mile
  race_distance : ℝ  -- Race distance in miles

/-- Calculates the time difference between Malcolm and Joshua finishing the race -/
def time_difference (scenario : RaceScenario) : ℝ :=
  scenario.joshua_speed * scenario.race_distance - scenario.malcolm_speed * scenario.race_distance

/-- Theorem stating the time difference for the given race scenario -/
theorem race_time_difference (scenario : RaceScenario) 
  (h1 : scenario.malcolm_speed = 5)
  (h2 : scenario.joshua_speed = 7)
  (h3 : scenario.race_distance = 12) :
  time_difference scenario = 24 := by
  sorry

#eval time_difference { malcolm_speed := 5, joshua_speed := 7, race_distance := 12 }

end NUMINAMATH_CALUDE_race_time_difference_l1992_199233


namespace NUMINAMATH_CALUDE_remainder_equality_l1992_199283

theorem remainder_equality (a b : ℕ+) :
  (∀ p : ℕ, Nat.Prime p → 
    (a : ℕ) % p ≤ (b : ℕ) % p) →
  a = b := by
sorry

end NUMINAMATH_CALUDE_remainder_equality_l1992_199283


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_N_l1992_199263

def M : Set ℝ := {x | |x| ≥ 3}

def N : Set ℝ := {y | ∃ x ∈ M, y = x^2}

theorem M_intersect_N_eq_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_N_l1992_199263


namespace NUMINAMATH_CALUDE_two_rooks_placement_count_l1992_199226

/-- The size of a standard chessboard -/
def boardSize : Nat := 8

/-- The total number of squares on a chessboard -/
def totalSquares : Nat := boardSize * boardSize

/-- The number of squares attacked by a rook (excluding its own square) -/
def attackedSquares : Nat := 2 * boardSize - 1

/-- The number of ways to place two rooks of different colors on a chessboard
    such that they do not attack each other -/
def twoRooksPlacement : Nat := totalSquares * (totalSquares - attackedSquares)

theorem two_rooks_placement_count :
  twoRooksPlacement = 3136 := by sorry

end NUMINAMATH_CALUDE_two_rooks_placement_count_l1992_199226


namespace NUMINAMATH_CALUDE_promotion_equivalence_bottles_in_box_l1992_199221

/-- The cost of a box of beverage in yuan -/
def box_cost : ℝ := 26

/-- The discount per bottle in yuan due to the promotion -/
def discount_per_bottle : ℝ := 0.6

/-- The number of free bottles given in the promotion -/
def free_bottles : ℕ := 3

/-- The number of bottles in a box -/
def bottles_per_box : ℕ := 10

theorem promotion_equivalence : 
  (box_cost / bottles_per_box) - (box_cost / (bottles_per_box + free_bottles)) = discount_per_bottle :=
sorry

theorem bottles_in_box : 
  bottles_per_box = 10 :=
sorry

end NUMINAMATH_CALUDE_promotion_equivalence_bottles_in_box_l1992_199221


namespace NUMINAMATH_CALUDE_trajectory_equation_l1992_199217

/-- The trajectory of point M(x, y) with distance ratio 2 from F(4,0) and line x = 3 -/
theorem trajectory_equation (x y : ℝ) : 
  (((x - 4)^2 + y^2) / ((x - 3)^2)) = 4 → 
  3 * x^2 - y^2 - 16 * x + 20 = 0 := by
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1992_199217


namespace NUMINAMATH_CALUDE_xu_shou_achievements_l1992_199243

/-- Represents a historical figure in Chinese science and technology -/
structure HistoricalFigure where
  name : String

/-- Represents a scientific achievement -/
inductive Achievement
  | SteamEngine
  | RiverSteamer
  | ChemicalTranslationPrinciples
  | ElementTranslations

/-- Predicate to check if a historical figure accomplished a given achievement in a specific year -/
def accomplished (person : HistoricalFigure) (achievement : Achievement) (year : ℕ) : Prop :=
  match achievement with
  | Achievement.SteamEngine => person.name = "Xu Shou" ∧ year = 1863
  | Achievement.RiverSteamer => person.name = "Xu Shou"
  | Achievement.ChemicalTranslationPrinciples => person.name = "Xu Shou"
  | Achievement.ElementTranslations => person.name = "Xu Shou" ∧ ∃ n : ℕ, n = 36

/-- Theorem stating that Xu Shou accomplished all the mentioned achievements -/
theorem xu_shou_achievements (xu_shou : HistoricalFigure) 
    (h_name : xu_shou.name = "Xu Shou") :
    accomplished xu_shou Achievement.SteamEngine 1863 ∧
    accomplished xu_shou Achievement.RiverSteamer 0 ∧
    accomplished xu_shou Achievement.ChemicalTranslationPrinciples 0 ∧
    accomplished xu_shou Achievement.ElementTranslations 0 :=
  sorry

end NUMINAMATH_CALUDE_xu_shou_achievements_l1992_199243


namespace NUMINAMATH_CALUDE_jake_viewing_time_l1992_199285

/-- Calculates the number of hours Jake watched on Friday given his viewing schedule for the week --/
theorem jake_viewing_time (hours_per_day : ℕ) (show_length : ℕ) : 
  hours_per_day = 24 →
  show_length = 52 →
  let monday := hours_per_day / 2
  let tuesday := 4
  let wednesday := hours_per_day / 4
  let mon_to_wed := monday + tuesday + wednesday
  let thursday := mon_to_wed / 2
  let mon_to_thu := mon_to_wed + thursday
  19 = show_length - mon_to_thu := by sorry


end NUMINAMATH_CALUDE_jake_viewing_time_l1992_199285


namespace NUMINAMATH_CALUDE_digit_equation_sum_l1992_199288

theorem digit_equation_sum : 
  ∃ (Y M E T : ℕ), 
    Y < 10 ∧ M < 10 ∧ E < 10 ∧ T < 10 ∧  -- digits are less than 10
    Y ≠ M ∧ Y ≠ E ∧ Y ≠ T ∧ M ≠ E ∧ M ≠ T ∧ E ≠ T ∧  -- digits are unique
    (10 * Y + E) * (10 * M + E) = T * T * T ∧  -- (YE) * (ME) = T * T * T
    T % 2 = 0 ∧  -- T is even
    E + M + T + Y = 10 :=  -- sum equals 10
by sorry

end NUMINAMATH_CALUDE_digit_equation_sum_l1992_199288


namespace NUMINAMATH_CALUDE_side_c_value_l1992_199202

/-- Given an acute triangle ABC with sides a = 4, b = 5, and area 5√3, 
    prove that the length of side c is √21 -/
theorem side_c_value (A B C : ℝ) (a b c : ℝ) (h_acute : A + B + C = π) 
  (h_a : a = 4) (h_b : b = 5) (h_area : (1/2) * a * b * Real.sin C = 5 * Real.sqrt 3) :
  c = Real.sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_side_c_value_l1992_199202


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1992_199245

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < B → B < π →
  (a - b + c) * (a + b + c) = 3 * a * c →
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  B = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1992_199245


namespace NUMINAMATH_CALUDE_min_value_of_fraction_sum_l1992_199232

theorem min_value_of_fraction_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : 2*m + n = 1) :
  (1/m + 2/n) ≥ 8 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2*m₀ + n₀ = 1 ∧ 1/m₀ + 2/n₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_sum_l1992_199232


namespace NUMINAMATH_CALUDE_janes_bowling_score_l1992_199222

def janes_score (x : ℝ) := x
def toms_score (x : ℝ) := x - 50

theorem janes_bowling_score :
  ∀ x : ℝ,
  janes_score x = toms_score x + 50 →
  (janes_score x + toms_score x) / 2 = 90 →
  janes_score x = 115 :=
by
  sorry

end NUMINAMATH_CALUDE_janes_bowling_score_l1992_199222


namespace NUMINAMATH_CALUDE_algebra_test_female_students_l1992_199224

theorem algebra_test_female_students 
  (total_average : ℝ) 
  (male_count : ℕ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (h1 : total_average = 90) 
  (h2 : male_count = 8) 
  (h3 : male_average = 87) 
  (h4 : female_average = 92) : 
  ∃ (female_count : ℕ), 
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧ 
    female_count = 12 := by
  sorry

end NUMINAMATH_CALUDE_algebra_test_female_students_l1992_199224


namespace NUMINAMATH_CALUDE_triangle_side_range_l1992_199229

theorem triangle_side_range (a : ℝ) : 
  let AB := (5 : ℝ)
  let BC := 2 * a + 1
  let AC := (12 : ℝ)
  (AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB) → (3 < a ∧ a < 8) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l1992_199229


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1992_199265

/-- The set of points (x, y) satisfying y(x+1) = x^2 - 1 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 * (p.1 + 1) = p.1^2 - 1}

/-- The vertical line x = -1 -/
def L1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = -1}

/-- The line y = x - 1 -/
def L2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 1}

/-- Theorem stating that S is equivalent to the union of L1 and L2 -/
theorem solution_set_equivalence : S = L1 ∪ L2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1992_199265


namespace NUMINAMATH_CALUDE_noodles_given_correct_daniel_noodles_l1992_199293

/-- The number of noodles Daniel gave to William -/
def noodles_given (initial current : ℕ) : ℕ := initial - current

theorem noodles_given_correct (initial current : ℕ) (h : current ≤ initial) :
  noodles_given initial current = initial - current :=
by
  sorry

/-- The specific problem instance -/
theorem daniel_noodles :
  noodles_given 66 54 = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_noodles_given_correct_daniel_noodles_l1992_199293


namespace NUMINAMATH_CALUDE_max_popsicles_is_16_l1992_199275

/-- Represents the cost and quantity of a popsicle package -/
structure PopsiclePackage where
  cost : ℕ
  quantity : ℕ

/-- Calculates the maximum number of popsicles that can be bought with a given budget -/
def maxPopsicles (budget : ℕ) (packages : List PopsiclePackage) : ℕ := sorry

/-- The specific problem setup -/
def problemSetup : List PopsiclePackage := [
  ⟨1, 1⟩,  -- Single popsicle
  ⟨3, 3⟩,  -- 3-popsicle box
  ⟨4, 7⟩   -- 7-popsicle box
]

/-- Theorem stating that the maximum number of popsicles Pablo can buy is 16 -/
theorem max_popsicles_is_16 :
  maxPopsicles 10 problemSetup = 16 := by sorry

end NUMINAMATH_CALUDE_max_popsicles_is_16_l1992_199275


namespace NUMINAMATH_CALUDE_smallest_possible_a_l1992_199215

def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem smallest_possible_a :
  ∀ (a b c : ℝ),
  a > 0 →
  parabola a b c (-1/3) = -4/3 →
  (∃ n : ℤ, a + b + c = n) →
  (∀ a' : ℝ, a' > 0 ∧ 
    (∃ b' c' : ℝ, parabola a' b' c' (-1/3) = -4/3 ∧ 
    (∃ n : ℤ, a' + b' + c' = n)) → 
  a' ≥ 3/16) →
  a = 3/16 := by
sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l1992_199215


namespace NUMINAMATH_CALUDE_x_equals_six_l1992_199272

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem x_equals_six : ∃ x : ℕ, x * factorial x + 2 * factorial x = 40320 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_six_l1992_199272


namespace NUMINAMATH_CALUDE_function_zeros_l1992_199281

def has_at_least_n_zeros (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : Prop :=
  ∃ (S : Finset ℝ), S.card ≥ n ∧ (∀ x ∈ S, a < x ∧ x ≤ b ∧ f x = 0)

theorem function_zeros
  (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_symmetry : ∀ x, f (2 + x) = f (2 - x))
  (h_zero_in_interval : ∃ x, 0 < x ∧ x < 4 ∧ f x = 0)
  (h_zero_at_origin : f 0 = 0) :
  has_at_least_n_zeros f (-8) 10 9 :=
sorry

end NUMINAMATH_CALUDE_function_zeros_l1992_199281


namespace NUMINAMATH_CALUDE_min_value_quadratic_expression_l1992_199299

theorem min_value_quadratic_expression (a b c : ℝ) :
  a < b →
  (∀ x, a * x^2 + b * x + c ≥ 0) →
  (a + b + c) / (b - a) ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_expression_l1992_199299


namespace NUMINAMATH_CALUDE_intersection_condition_l1992_199292

/-- The set A parameterized by m -/
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1^2 + m * p.1 - p.2 + 2 = 0}

/-- The set B -/
def B : Set (ℝ × ℝ) :=
  {p | p.1 - p.2 + 1 = 0}

/-- Theorem stating the condition for non-empty intersection of A and B -/
theorem intersection_condition (m : ℝ) :
  (A m ∩ B).Nonempty ↔ m ≤ -1 ∨ m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l1992_199292


namespace NUMINAMATH_CALUDE_first_division_percentage_l1992_199239

theorem first_division_percentage (total_students : ℕ) 
  (second_division_percentage : ℚ) (just_passed : ℕ) :
  total_students = 300 →
  second_division_percentage = 54 / 100 →
  just_passed = 63 →
  (total_students : ℚ) * (25 / 100) = 
    total_students - (total_students : ℚ) * second_division_percentage - just_passed :=
by
  sorry

end NUMINAMATH_CALUDE_first_division_percentage_l1992_199239


namespace NUMINAMATH_CALUDE_range_of_m_l1992_199266

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 1/x + 4/y = 1) (h_sol : ∃ m : ℝ, x + y/4 < m^2 - 3*m) :
  ∀ m : ℝ, (x + y/4 < m^2 - 3*m) ↔ (m < -1 ∨ m > 4) := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1992_199266


namespace NUMINAMATH_CALUDE_sum_with_gap_l1992_199219

theorem sum_with_gap (x : ℝ) (h1 : |x - 5.46| = 3.97) (h2 : x < 5.46) : x + 5.46 = 6.95 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_gap_l1992_199219


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1992_199214

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
  a 1 = 3 →
  a 100 = 36 →
  a 3 + a 98 = 39 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1992_199214


namespace NUMINAMATH_CALUDE_students_with_both_pets_l1992_199261

theorem students_with_both_pets (total : ℕ) (dog_owners : ℕ) (cat_owners : ℕ) 
  (h1 : total = 50)
  (h2 : dog_owners = 35)
  (h3 : cat_owners = 40)
  (h4 : dog_owners + cat_owners - total ≤ dog_owners)
  (h5 : dog_owners + cat_owners - total ≤ cat_owners) :
  dog_owners + cat_owners - total = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_with_both_pets_l1992_199261


namespace NUMINAMATH_CALUDE_hall_area_l1992_199213

/-- Proves that the area of a rectangular hall is 500 square meters, given that its length is 25 meters and 5 meters more than its breadth. -/
theorem hall_area : 
  ∀ (length breadth : ℝ),
  length = 25 →
  length = breadth + 5 →
  length * breadth = 500 := by
sorry

end NUMINAMATH_CALUDE_hall_area_l1992_199213


namespace NUMINAMATH_CALUDE_floor_product_equality_l1992_199234

theorem floor_product_equality (x : ℝ) : ⌊x * ⌊x⌋⌋ = 49 ↔ 7 ≤ x ∧ x < 50 / 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_product_equality_l1992_199234


namespace NUMINAMATH_CALUDE_elevator_unreachable_l1992_199212

def is_valid_floor (n : ℤ) : Prop := 1 ≤ n ∧ n ≤ 15

def elevator_move (n : ℤ) : ℤ → ℤ
  | 0 => n  -- base case: no moves
  | 1 => n + 7  -- move up 7 floors
  | -1 => n - 9  -- move down 9 floors
  | _ => n  -- invalid move, stay on the same floor

def can_reach (start finish : ℤ) : Prop :=
  ∃ (moves : List ℤ), 
    (∀ m ∈ moves, m = 1 ∨ m = -1) ∧
    (List.foldl elevator_move start moves = finish) ∧
    (∀ i, is_valid_floor (List.foldl elevator_move start (moves.take i)))

theorem elevator_unreachable :
  ¬(can_reach 3 12) :=
sorry

end NUMINAMATH_CALUDE_elevator_unreachable_l1992_199212


namespace NUMINAMATH_CALUDE_bubble_bath_per_guest_l1992_199209

theorem bubble_bath_per_guest (couple_rooms : ℕ) (single_rooms : ℕ) (total_bubble_bath : ℕ) :
  couple_rooms = 13 →
  single_rooms = 14 →
  total_bubble_bath = 400 →
  (total_bubble_bath : ℚ) / (2 * couple_rooms + single_rooms) = 10 :=
by sorry

end NUMINAMATH_CALUDE_bubble_bath_per_guest_l1992_199209


namespace NUMINAMATH_CALUDE_range_of_a_for_subset_l1992_199295

-- Define the sets A and B
def A (a : ℝ) := { x : ℝ | |x - (a+1)^2/2| ≤ (a-1)^2/2 }
def B (a : ℝ) := { x : ℝ | x^2 - 3*(a+1)*x + 2*(3*a+1) ≤ 0 }

-- Define the subset relation
def is_subset (S T : Set ℝ) := ∀ x, x ∈ S → x ∈ T

-- State the theorem
theorem range_of_a_for_subset : 
  { a : ℝ | is_subset (A a) (B a) } = Set.union (Set.Icc 1 3) {-1} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_subset_l1992_199295


namespace NUMINAMATH_CALUDE_average_pushups_l1992_199277

theorem average_pushups (david zachary emily : ℕ) : 
  david = 510 ∧ 
  david = zachary + 210 ∧ 
  david = emily + 132 → 
  (david + zachary + emily) / 3 = 396 := by
    sorry

end NUMINAMATH_CALUDE_average_pushups_l1992_199277


namespace NUMINAMATH_CALUDE_probability_two_heads_in_four_tosses_l1992_199254

-- Define the number of coin tosses
def n : ℕ := 4

-- Define the number of heads we're looking for
def k : ℕ := 2

-- Define the probability of getting heads on a single toss
def p : ℚ := 1/2

-- Define the probability of getting tails on a single toss
def q : ℚ := 1/2

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability of getting exactly k heads in n tosses
def probability_k_heads (n k : ℕ) (p q : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * q^(n-k)

-- Theorem statement
theorem probability_two_heads_in_four_tosses :
  probability_k_heads n k p q = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_heads_in_four_tosses_l1992_199254


namespace NUMINAMATH_CALUDE_balloon_count_l1992_199286

theorem balloon_count (initial : Real) (given : Real) (total : Real) 
  (h1 : initial = 7.0)
  (h2 : given = 5.0)
  (h3 : total = initial + given) :
  total = 12.0 := by
sorry

end NUMINAMATH_CALUDE_balloon_count_l1992_199286


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1992_199211

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def M : Set Nat := {2, 3, 4, 5}
def N : Set Nat := {1, 4, 5, 7}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1992_199211


namespace NUMINAMATH_CALUDE_max_generatable_number_l1992_199237

def powers_of_three : List ℕ := [1, 3, 9, 27, 81, 243, 729]

def can_generate (n : ℤ) : Prop :=
  ∃ (coeffs : List ℤ), 
    coeffs.length = powers_of_three.length ∧ 
    (∀ c ∈ coeffs, c = 1 ∨ c = 0 ∨ c = -1) ∧
    n = List.sum (List.zipWith (· * ·) coeffs (powers_of_three.map Int.ofNat))

theorem max_generatable_number :
  (∀ n : ℕ, n ≤ 1093 → can_generate n) ∧
  ¬(can_generate 1094) :=
sorry

end NUMINAMATH_CALUDE_max_generatable_number_l1992_199237


namespace NUMINAMATH_CALUDE_rulers_in_drawer_l1992_199282

/-- The number of rulers originally in the drawer -/
def original_rulers : ℕ := 71 - 25

theorem rulers_in_drawer : original_rulers = 46 := by
  sorry

end NUMINAMATH_CALUDE_rulers_in_drawer_l1992_199282


namespace NUMINAMATH_CALUDE_m_plus_3_interpretation_l1992_199252

/-- Represents the possible interpretations of an assignment statement -/
inductive AssignmentInterpretation
  | AssignToSum
  | AddAndReassign
  | Equality
  | None

/-- Defines the meaning of an assignment statement -/
def assignmentMeaning (left : String) (right : String) : AssignmentInterpretation :=
  if left = right.take (right.length - 2) && right.takeRight 2 = "+3" then
    AssignmentInterpretation.AddAndReassign
  else
    AssignmentInterpretation.None

/-- Theorem stating the correct interpretation of M=M+3 -/
theorem m_plus_3_interpretation :
  assignmentMeaning "M" "M+3" = AssignmentInterpretation.AddAndReassign :=
by sorry

end NUMINAMATH_CALUDE_m_plus_3_interpretation_l1992_199252


namespace NUMINAMATH_CALUDE_equidistant_point_count_l1992_199298

/-- A quadrilateral is a polygon with four vertices. -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- A rectangle is a quadrilateral with four right angles. -/
def IsRectangle (q : Quadrilateral) : Prop := sorry

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides. -/
def IsTrapezoid (q : Quadrilateral) : Prop := sorry

/-- A trapezoid has congruent base angles if the angles adjacent to each parallel side are congruent. -/
def HasCongruentBaseAngles (q : Quadrilateral) : Prop := sorry

/-- A point is equidistant from all vertices of a quadrilateral if its distance to each vertex is the same. -/
def HasEquidistantPoint (q : Quadrilateral) : Prop := sorry

/-- The theorem states that among rectangles and trapezoids with congruent base angles, 
    exactly two types of quadrilaterals have a point equidistant from all four vertices. -/
theorem equidistant_point_count :
  ∃ (q1 q2 : Quadrilateral),
    (IsRectangle q1 ∨ (IsTrapezoid q1 ∧ HasCongruentBaseAngles q1)) ∧
    (IsRectangle q2 ∨ (IsTrapezoid q2 ∧ HasCongruentBaseAngles q2)) ∧
    q1 ≠ q2 ∧
    HasEquidistantPoint q1 ∧
    HasEquidistantPoint q2 ∧
    (∀ q : Quadrilateral,
      (IsRectangle q ∨ (IsTrapezoid q ∧ HasCongruentBaseAngles q)) →
      HasEquidistantPoint q →
      (q = q1 ∨ q = q2)) :=
by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_count_l1992_199298


namespace NUMINAMATH_CALUDE_distance_between_X_and_Y_l1992_199225

/-- The distance between X and Y in miles -/
def D : ℝ := 31

/-- Yolanda's walking rate in miles per hour -/
def yolanda_rate : ℝ := 1

/-- Bob's walking rate in miles per hour -/
def bob_rate : ℝ := 2

/-- The distance Bob walked before meeting Yolanda in miles -/
def bob_distance : ℝ := 20

/-- The time difference between Yolanda's and Bob's start times in hours -/
def time_difference : ℝ := 1

theorem distance_between_X_and_Y :
  D = bob_distance + yolanda_rate * (bob_distance / bob_rate + time_difference) :=
sorry

end NUMINAMATH_CALUDE_distance_between_X_and_Y_l1992_199225


namespace NUMINAMATH_CALUDE_percentage_only_cat_owners_l1992_199242

def total_students : ℕ := 500
def dog_owners : ℕ := 120
def cat_owners : ℕ := 80
def both_owners : ℕ := 40

def only_cat_owners : ℕ := cat_owners - both_owners

theorem percentage_only_cat_owners :
  (only_cat_owners : ℚ) / total_students * 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_only_cat_owners_l1992_199242


namespace NUMINAMATH_CALUDE_not_necessarily_square_lt_of_lt_l1992_199268

theorem not_necessarily_square_lt_of_lt {a b : ℝ} (h : a < b) : 
  ¬(∀ a b : ℝ, a < b → a^2 < b^2) :=
by sorry

end NUMINAMATH_CALUDE_not_necessarily_square_lt_of_lt_l1992_199268


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l1992_199255

theorem quadratic_equation_result (x : ℝ) : 
  7 * x^2 - 2 * x - 4 = 4 * x + 11 → (5 * x - 7)^2 = 570 / 49 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l1992_199255


namespace NUMINAMATH_CALUDE_line_point_k_value_l1992_199271

/-- A line contains the points (3,5), (1,k), and (7,9). Prove that k = 3. -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (m b : ℝ), m * 3 + b = 5 ∧ m * 1 + b = k ∧ m * 7 + b = 9) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_point_k_value_l1992_199271


namespace NUMINAMATH_CALUDE_carols_pool_water_carols_pool_water_proof_l1992_199238

/-- Calculates the amount of water left in Carol's pool after five hours of filling and a leak -/
theorem carols_pool_water (first_hour_rate : ℕ) (second_third_hour_rate : ℕ) (fourth_hour_rate : ℕ) (leak_amount : ℕ) : ℕ :=
  let total_added := first_hour_rate + 2 * second_third_hour_rate + fourth_hour_rate
  total_added - leak_amount

/-- Proves that the amount of water left in Carol's pool after five hours is 34 gallons -/
theorem carols_pool_water_proof :
  carols_pool_water 8 10 14 8 = 34 := by
  sorry

end NUMINAMATH_CALUDE_carols_pool_water_carols_pool_water_proof_l1992_199238


namespace NUMINAMATH_CALUDE_speed_in_still_water_l1992_199248

/-- Theorem: Given a man's upstream and downstream speeds, his speed in still water
    is the average of these two speeds. -/
theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) :
  upstream_speed = 55 →
  downstream_speed = 65 →
  (upstream_speed + downstream_speed) / 2 = 60 := by
sorry

end NUMINAMATH_CALUDE_speed_in_still_water_l1992_199248


namespace NUMINAMATH_CALUDE_primitive_pythagorean_triple_parity_l1992_199200

theorem primitive_pythagorean_triple_parity (a b c : ℕ+) 
  (h1 : a^2 + b^2 = c^2)
  (h2 : Nat.gcd a.val (Nat.gcd b.val c.val) = 1) :
  (Even a.val ∧ Odd b.val) ∨ (Odd a.val ∧ Even b.val) := by
sorry

end NUMINAMATH_CALUDE_primitive_pythagorean_triple_parity_l1992_199200


namespace NUMINAMATH_CALUDE_sqrt_difference_power_l1992_199257

theorem sqrt_difference_power (A B : ℤ) : 
  ∃ A B : ℤ, (Real.sqrt 1969 - Real.sqrt 1968) ^ 1969 = A * Real.sqrt 1969 - B * Real.sqrt 1968 ∧ 
  1969 * A^2 - 1968 * B^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_power_l1992_199257


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_three_pi_eighths_l1992_199249

theorem cos_squared_minus_sin_squared_three_pi_eighths :
  Real.cos (3 * Real.pi / 8) ^ 2 - Real.sin (3 * Real.pi / 8) ^ 2 = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_three_pi_eighths_l1992_199249


namespace NUMINAMATH_CALUDE_area_of_curve_l1992_199264

/-- The curve defined by x^2 + y^2 = |x| + 2|y| -/
def curve (x y : ℝ) : Prop := x^2 + y^2 = |x| + 2 * |y|

/-- The area enclosed by the curve -/
noncomputable def enclosed_area : ℝ := sorry

theorem area_of_curve : enclosed_area = (5 * π) / 4 := by sorry

end NUMINAMATH_CALUDE_area_of_curve_l1992_199264


namespace NUMINAMATH_CALUDE_profit_maximization_l1992_199260

/-- The profit function for a product with cost 20 yuan per kilogram -/
def profit_function (x : ℝ) : ℝ := (x - 20) * (-x + 150)

/-- The sales volume function -/
def sales_volume (x : ℝ) : ℝ := -x + 150

theorem profit_maximization :
  ∃ (max_price max_profit : ℝ),
    (∀ x : ℝ, 20 ≤ x ∧ x ≤ 90 → profit_function x ≤ max_profit) ∧
    max_price = 85 ∧
    max_profit = 4225 ∧
    profit_function max_price = max_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_maximization_l1992_199260


namespace NUMINAMATH_CALUDE_at_least_two_blue_bikes_count_l1992_199241

def yellow_bikes : ℕ := 6
def blue_bikes : ℕ := 4
def total_bikes : ℕ := yellow_bikes + blue_bikes
def selected_bikes : ℕ := 4

def ways_to_select_at_least_two_blue : ℕ :=
  Nat.choose blue_bikes 4 +
  Nat.choose blue_bikes 3 * Nat.choose yellow_bikes 1 +
  Nat.choose blue_bikes 2 * Nat.choose yellow_bikes 2

theorem at_least_two_blue_bikes_count :
  ways_to_select_at_least_two_blue = 115 := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_blue_bikes_count_l1992_199241


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l1992_199297

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for the range of a when B ∪ C = C
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l1992_199297
