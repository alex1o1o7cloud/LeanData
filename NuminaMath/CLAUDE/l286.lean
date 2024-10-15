import Mathlib

namespace NUMINAMATH_CALUDE_gcd_sum_and_sum_of_squares_minus_product_l286_28647

theorem gcd_sum_and_sum_of_squares_minus_product (a b : ℤ) : 
  Int.gcd a b = 1 → Int.gcd (a + b) (a^2 + b^2 - a*b) = 1 ∨ Int.gcd (a + b) (a^2 + b^2 - a*b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_and_sum_of_squares_minus_product_l286_28647


namespace NUMINAMATH_CALUDE_marilyn_bottle_caps_l286_28634

theorem marilyn_bottle_caps (initial_caps : ℝ) (received_caps : ℝ) : 
  initial_caps = 51.0 → received_caps = 36.0 → initial_caps + received_caps = 87.0 := by
  sorry

end NUMINAMATH_CALUDE_marilyn_bottle_caps_l286_28634


namespace NUMINAMATH_CALUDE_bouquet_calculation_l286_28668

theorem bouquet_calculation (total_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) : 
  total_flowers = 53 → 
  flowers_per_bouquet = 7 → 
  wilted_flowers = 18 → 
  (total_flowers - wilted_flowers) / flowers_per_bouquet = 5 := by
  sorry

end NUMINAMATH_CALUDE_bouquet_calculation_l286_28668


namespace NUMINAMATH_CALUDE_point_on_negative_x_axis_l286_28606

/-- Given a point A with coordinates (a+1, a^2-4) that lies on the negative half of the x-axis,
    prove that its coordinates are (-1, 0). -/
theorem point_on_negative_x_axis (a : ℝ) : 
  (a + 1 < 0) ∧ (a^2 - 4 = 0) → (a + 1 = -1 ∧ a^2 - 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_point_on_negative_x_axis_l286_28606


namespace NUMINAMATH_CALUDE_function_equality_l286_28653

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem function_equality (f : ℝ → ℝ) :
  (∀ x, f x^3 + f x ≤ x ∧ x ≤ f (x^3 + x)) →
  (∀ x, f x = Function.invFun g x) :=
by sorry

end NUMINAMATH_CALUDE_function_equality_l286_28653


namespace NUMINAMATH_CALUDE_adjacent_permutations_of_six_l286_28677

/-- The number of permutations of n elements where two specific elements are always adjacent -/
def adjacentPermutations (n : ℕ) : ℕ :=
  2 * Nat.factorial (n - 1)

/-- Given 6 people with two specific individuals, the number of permutations
    where these two individuals are always adjacent is 240 -/
theorem adjacent_permutations_of_six :
  adjacentPermutations 6 = 240 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_permutations_of_six_l286_28677


namespace NUMINAMATH_CALUDE_square_prism_sum_l286_28636

theorem square_prism_sum (a b c d e f : ℕ+) (h : a * b * e + a * b * f + a * c * e + a * c * f + 
                                               d * b * e + d * b * f + d * c * e + d * c * f = 1176) : 
  a + b + c + d + e + f = 33 := by
  sorry

end NUMINAMATH_CALUDE_square_prism_sum_l286_28636


namespace NUMINAMATH_CALUDE_back_seat_capacity_is_eleven_l286_28646

/-- Represents a bus with seats on left and right sides, and a back seat. -/
structure Bus where
  left_seats : Nat
  right_seats : Nat
  people_per_seat : Nat
  total_capacity : Nat

/-- Calculates the number of people that can sit at the back seat of the bus. -/
def back_seat_capacity (bus : Bus) : Nat :=
  bus.total_capacity - (bus.left_seats + bus.right_seats) * bus.people_per_seat

/-- Theorem stating the number of people that can sit at the back seat of the given bus. -/
theorem back_seat_capacity_is_eleven :
  let bus : Bus := {
    left_seats := 15,
    right_seats := 15 - 3,
    people_per_seat := 3,
    total_capacity := 92
  }
  back_seat_capacity bus = 11 := by
  sorry

#eval back_seat_capacity {
  left_seats := 15,
  right_seats := 15 - 3,
  people_per_seat := 3,
  total_capacity := 92
}

end NUMINAMATH_CALUDE_back_seat_capacity_is_eleven_l286_28646


namespace NUMINAMATH_CALUDE_simple_interest_problem_l286_28697

theorem simple_interest_problem (P R : ℝ) (h : P > 0) (h_rate : R > 0) :
  (P * (R + 5) * 9 / 100 = P * R * 9 / 100 + 1350) → P = 3000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l286_28697


namespace NUMINAMATH_CALUDE_set_mean_given_median_l286_28604

theorem set_mean_given_median (n : ℝ) :
  (Finset.range 5).card = 5 →
  n + 8 = 14 →
  let s := {n, n + 6, n + 8, n + 10, n + 18}
  (Finset.filter (λ x => x ≤ n + 8) s).card = 3 →
  (Finset.sum s id) / 5 = 14.4 := by
sorry

end NUMINAMATH_CALUDE_set_mean_given_median_l286_28604


namespace NUMINAMATH_CALUDE_round_table_knights_and_liars_l286_28662

theorem round_table_knights_and_liars (n : ℕ) (K : ℕ) : 
  n > 1000 →
  n = K + (n - K) →
  (∀ i : ℕ, i < n → (20 * K) % n = 0) →
  (∀ m : ℕ, m > 1000 → (20 * K) % m = 0 → m ≥ n) →
  n = 1020 :=
sorry

end NUMINAMATH_CALUDE_round_table_knights_and_liars_l286_28662


namespace NUMINAMATH_CALUDE_shipment_size_l286_28621

/-- The total number of novels in the shipment -/
def total_novels : ℕ := 300

/-- The fraction of novels displayed in the storefront -/
def display_fraction : ℚ := 30 / 100

/-- The number of novels in the storage room -/
def storage_novels : ℕ := 210

/-- Theorem stating that the total number of novels is 300 -/
theorem shipment_size :
  total_novels = 300 ∧
  display_fraction = 30 / 100 ∧
  storage_novels = 210 ∧
  (1 - display_fraction) * total_novels = storage_novels :=
by sorry

end NUMINAMATH_CALUDE_shipment_size_l286_28621


namespace NUMINAMATH_CALUDE_deck_card_count_l286_28632

theorem deck_card_count (r : ℕ) (b : ℕ) : 
  b = 2 * r →                 -- Initial condition: black cards are twice red cards
  (b + 4) = 3 * r →           -- After adding 4 black cards, they're three times red cards
  r + b = 12                  -- The initial total number of cards is 12
  := by sorry

end NUMINAMATH_CALUDE_deck_card_count_l286_28632


namespace NUMINAMATH_CALUDE_cats_left_after_sale_l286_28620

theorem cats_left_after_sale (siamese : ℕ) (house : ℕ) (sold : ℕ) : siamese = 38 → house = 25 → sold = 45 → siamese + house - sold = 18 := by
  sorry

end NUMINAMATH_CALUDE_cats_left_after_sale_l286_28620


namespace NUMINAMATH_CALUDE_correct_equation_l286_28645

theorem correct_equation : 500 - 9 * 7 = 437 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l286_28645


namespace NUMINAMATH_CALUDE_fourth_power_nested_sqrt_l286_28630

theorem fourth_power_nested_sqrt : 
  (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 2))))^4 = 
  2 + 2 * Real.sqrt (1 + Real.sqrt 2) + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_nested_sqrt_l286_28630


namespace NUMINAMATH_CALUDE_polynomial_satisfies_conditions_l286_28693

theorem polynomial_satisfies_conditions :
  ∃ (p : ℝ → ℝ), 
    (∀ x, p x = x^2 + 1) ∧ 
    (p 3 = 10) ∧ 
    (∀ x y, p x * p y = p x + p y + p (x * y) - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_satisfies_conditions_l286_28693


namespace NUMINAMATH_CALUDE_li_ming_weight_estimate_l286_28684

/-- Regression equation for weight based on height -/
def weight_estimate (height : ℝ) : ℝ := 0.7 * height - 52

/-- Li Ming's height in cm -/
def li_ming_height : ℝ := 180

/-- Theorem: Li Ming's estimated weight is 74 kg -/
theorem li_ming_weight_estimate :
  weight_estimate li_ming_height = 74 := by
  sorry

end NUMINAMATH_CALUDE_li_ming_weight_estimate_l286_28684


namespace NUMINAMATH_CALUDE_cliffs_rock_collection_l286_28695

theorem cliffs_rock_collection (igneous_rocks sedimentary_rocks : ℕ) : 
  igneous_rocks = sedimentary_rocks / 2 →
  igneous_rocks / 3 = 30 →
  igneous_rocks + sedimentary_rocks = 270 := by
  sorry

end NUMINAMATH_CALUDE_cliffs_rock_collection_l286_28695


namespace NUMINAMATH_CALUDE_tens_digit_of_2023_pow_2024_minus_2025_l286_28699

theorem tens_digit_of_2023_pow_2024_minus_2025 : ∃ n : ℕ, 2023^2024 - 2025 = 100*n + 4 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_2023_pow_2024_minus_2025_l286_28699


namespace NUMINAMATH_CALUDE_regular_tetrahedron_is_connected_l286_28641

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a regular tetrahedron
def RegularTetrahedron : Set Point3D := sorry

-- Define a line segment between two points
def LineSegment (p q : Point3D) : Set Point3D := sorry

-- Define the property of being a connected set
def IsConnectedSet (S : Set Point3D) : Prop :=
  ∀ p q : Point3D, p ∈ S → q ∈ S → LineSegment p q ⊆ S

-- Theorem statement
theorem regular_tetrahedron_is_connected : IsConnectedSet RegularTetrahedron := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_is_connected_l286_28641


namespace NUMINAMATH_CALUDE_smaller_cuboid_height_l286_28642

/-- Proves that the height of smaller cuboids is 2 meters when a large cuboid
    is divided into smaller ones with given dimensions. -/
theorem smaller_cuboid_height
  (large_length : ℝ) (large_width : ℝ) (large_height : ℝ)
  (small_length : ℝ) (small_width : ℝ)
  (num_small_cuboids : ℕ) :
  large_length = 12 →
  large_width = 14 →
  large_height = 10 →
  small_length = 5 →
  small_width = 3 →
  num_small_cuboids = 56 →
  ∃ (small_height : ℝ),
    large_length * large_width * large_height =
    ↑num_small_cuboids * small_length * small_width * small_height ∧
    small_height = 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_cuboid_height_l286_28642


namespace NUMINAMATH_CALUDE_square_difference_l286_28682

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l286_28682


namespace NUMINAMATH_CALUDE_workout_schedule_l286_28612

theorem workout_schedule (x : ℝ) 
  (h1 : x > 0)  -- Workout duration is positive
  (h2 : x + (x - 2) + 2*x + 2*(x - 2) = 18) :  -- Total workout time is 18 hours
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_workout_schedule_l286_28612


namespace NUMINAMATH_CALUDE_abs_4y_minus_6_not_positive_l286_28609

theorem abs_4y_minus_6_not_positive (y : ℚ) : ¬(|4*y - 6| > 0) ↔ y = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_4y_minus_6_not_positive_l286_28609


namespace NUMINAMATH_CALUDE_certain_number_problem_l286_28683

theorem certain_number_problem (x : ℝ) : 
  3 - (1/5) * 390 = 4 - (1/7) * x + 114 → x > 1351 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l286_28683


namespace NUMINAMATH_CALUDE_path_area_calculation_l286_28691

/-- Calculates the area of a path around a rectangular field -/
def pathArea (fieldLength fieldWidth pathWidth : ℝ) : ℝ :=
  (fieldLength + 2 * pathWidth) * (fieldWidth + 2 * pathWidth) - fieldLength * fieldWidth

/-- Theorem: The area of a 2.5m wide path around a 75m by 55m field is 675 sq m -/
theorem path_area_calculation :
  pathArea 75 55 2.5 = 675 := by sorry

end NUMINAMATH_CALUDE_path_area_calculation_l286_28691


namespace NUMINAMATH_CALUDE_selina_shirts_sold_l286_28678

/-- Calculates the number of shirts Selina sold given the conditions of the problem -/
def shirts_sold (pants_price shorts_price shirt_price : ℕ) 
  (pants_sold shorts_sold : ℕ) (bought_shirt_price : ℕ) 
  (bought_shirt_count : ℕ) (money_left : ℕ) : ℕ :=
  let total_before_buying := money_left + bought_shirt_price * bought_shirt_count
  let money_from_pants_shorts := pants_price * pants_sold + shorts_price * shorts_sold
  let money_from_shirts := total_before_buying - money_from_pants_shorts
  money_from_shirts / shirt_price

theorem selina_shirts_sold : 
  shirts_sold 5 3 4 3 5 10 2 30 = 5 := by
  sorry

end NUMINAMATH_CALUDE_selina_shirts_sold_l286_28678


namespace NUMINAMATH_CALUDE_girls_entered_l286_28692

theorem girls_entered (initial_children final_children boys_left : ℕ) 
  (h1 : initial_children = 85)
  (h2 : boys_left = 31)
  (h3 : final_children = 78) :
  final_children - (initial_children - boys_left) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_girls_entered_l286_28692


namespace NUMINAMATH_CALUDE_trapezium_height_l286_28654

theorem trapezium_height (a b area : ℝ) (ha : a = 30) (hb : b = 12) (harea : area = 336) :
  (area * 2) / (a + b) = 16 :=
by sorry

end NUMINAMATH_CALUDE_trapezium_height_l286_28654


namespace NUMINAMATH_CALUDE_customer_difference_l286_28639

theorem customer_difference (initial : ℕ) (remained : ℕ) : 
  initial = 11 → remained = 3 → (initial - remained) - remained = 5 := by
sorry

end NUMINAMATH_CALUDE_customer_difference_l286_28639


namespace NUMINAMATH_CALUDE_problem_solution_l286_28614

theorem problem_solution (m : ℤ) (a b c : ℝ) 
  (h1 : ∃! (x : ℤ), |2 * (x : ℝ) - m| ≤ 1 ∧ x = 2)
  (h2 : 4 * a^4 + 4 * b^4 + 4 * c^4 = m) :
  m = 4 ∧ (a^2 + b^2 + c^2 ≤ Real.sqrt 3 ∧ ∃ x y z, x^2 + y^2 + z^2 = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l286_28614


namespace NUMINAMATH_CALUDE_family_siblings_product_l286_28688

theorem family_siblings_product (total_sisters total_brothers : ℕ) 
  (h1 : total_sisters = 3) 
  (h2 : total_brothers = 5) : 
  ∃ (S B : ℕ), S * B = 10 ∧ S = total_sisters - 1 ∧ B = total_brothers :=
by sorry

end NUMINAMATH_CALUDE_family_siblings_product_l286_28688


namespace NUMINAMATH_CALUDE_target_perm_unreachable_cannot_reach_reverse_order_l286_28629

/-- Represents the three colors of balls -/
inductive Color
  | Red
  | Blue
  | White

/-- Represents a permutation of the three balls -/
def Permutation := (Color × Color × Color)

/-- The initial permutation of the balls -/
def initial_perm : Permutation := (Color.Red, Color.Blue, Color.White)

/-- Checks if a permutation is valid (no ball in its original position) -/
def is_valid_perm (p : Permutation) : Prop :=
  p.1 ≠ Color.Red ∧ p.2.1 ≠ Color.Blue ∧ p.2.2 ≠ Color.White

/-- The set of all valid permutations -/
def valid_perms : Set Permutation :=
  {p | is_valid_perm p}

/-- The target permutation (reverse of initial) -/
def target_perm : Permutation := (Color.White, Color.Blue, Color.Red)

/-- Theorem stating that the target permutation is unreachable -/
theorem target_perm_unreachable : target_perm ∉ valid_perms := by
  sorry

/-- Main theorem: It's impossible to reach the target permutation after any number of valid rearrangements -/
theorem cannot_reach_reverse_order :
  ∀ n : ℕ, ∀ f : ℕ → Permutation,
    (f 0 = initial_perm) →
    (∀ i, i < n → is_valid_perm (f (i + 1))) →
    (f n ≠ target_perm) := by
  sorry

end NUMINAMATH_CALUDE_target_perm_unreachable_cannot_reach_reverse_order_l286_28629


namespace NUMINAMATH_CALUDE_eliza_height_is_83_l286_28687

/-- The height of Eliza given the heights of her siblings -/
def elizaHeight (total_height : ℕ) (sibling1_height : ℕ) (sibling2_height : ℕ) 
  (sibling3_height : ℕ) (sibling4_height : ℕ) (sibling5_height : ℕ) : ℕ :=
  total_height - (sibling1_height + sibling2_height + sibling3_height + sibling4_height + sibling5_height)

theorem eliza_height_is_83 :
  let total_height := 435
  let sibling1_height := 66
  let sibling2_height := 66
  let sibling3_height := 60
  let sibling4_height := 75
  let sibling5_height := elizaHeight total_height sibling1_height sibling2_height sibling3_height sibling4_height 85 + 2
  elizaHeight total_height sibling1_height sibling2_height sibling3_height sibling4_height sibling5_height = 83 := by
  sorry

end NUMINAMATH_CALUDE_eliza_height_is_83_l286_28687


namespace NUMINAMATH_CALUDE_jills_nails_count_l286_28650

theorem jills_nails_count : ∃ N : ℕ,
  N > 0 ∧
  (8 : ℝ) / N * 100 - ((N : ℝ) - 14) / N * 100 = 10 ∧
  6 + 8 + (N - 14) = N :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_jills_nails_count_l286_28650


namespace NUMINAMATH_CALUDE_fewer_servings_l286_28665

def total_ounces : ℕ := 64
def old_serving_size : ℕ := 8
def new_serving_size : ℕ := 16

theorem fewer_servings :
  (total_ounces / old_serving_size) - (total_ounces / new_serving_size) = 4 :=
by sorry

end NUMINAMATH_CALUDE_fewer_servings_l286_28665


namespace NUMINAMATH_CALUDE_min_value_theorem_l286_28619

theorem min_value_theorem (a : ℝ) (h : a > 0) : 
  (∃ (x : ℝ), x = 3 / (2 * a) + 4 * a ∧ ∀ (y : ℝ), y = 3 / (2 * a) + 4 * a → x ≤ y) ∧ 
  (∃ (z : ℝ), z = 3 / (2 * a) + 4 * a ∧ z = 2 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l286_28619


namespace NUMINAMATH_CALUDE_equation_system_solution_l286_28658

theorem equation_system_solution :
  ∀ (x y a : ℝ),
  (2 * x + y = a) →
  (x + y = 3) →
  (x = 2) →
  (a = 5 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l286_28658


namespace NUMINAMATH_CALUDE_removed_triangles_area_l286_28633

theorem removed_triangles_area (original_side : ℝ) (h_original_side : original_side = 20) :
  let smaller_side : ℝ := original_side / 2
  let removed_triangle_leg : ℝ := (original_side - smaller_side) / Real.sqrt 2
  let single_triangle_area : ℝ := removed_triangle_leg ^ 2 / 2
  4 * single_triangle_area = 100 := by
  sorry

end NUMINAMATH_CALUDE_removed_triangles_area_l286_28633


namespace NUMINAMATH_CALUDE_skating_time_l286_28674

/-- Given a distance of 80 kilometers and a speed of 10 kilometers per hour,
    the time taken is 8 hours. -/
theorem skating_time (distance : ℝ) (speed : ℝ) (time : ℝ) 
    (h1 : distance = 80)
    (h2 : speed = 10)
    (h3 : time = distance / speed) : 
  time = 8 := by
sorry

end NUMINAMATH_CALUDE_skating_time_l286_28674


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l286_28651

/-- A two-digit number is a natural number between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The tens digit of a natural number. -/
def tensDigit (n : ℕ) : ℕ := n / 10

/-- The units digit of a natural number. -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The sum of digits of a two-digit number. -/
def sumOfDigits (n : ℕ) : ℕ := tensDigit n + unitsDigit n

/-- The main theorem stating that 24 is the unique two-digit number satisfying the given conditions. -/
theorem unique_two_digit_number : 
  ∃! n : ℕ, TwoDigitNumber n ∧ 
            tensDigit n = unitsDigit n / 2 ∧ 
            n - sumOfDigits n = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l286_28651


namespace NUMINAMATH_CALUDE_exists_nonzero_digits_multiple_of_power_of_two_l286_28670

/-- Returns true if all digits of n in decimal representation are non-zero -/
def allDigitsNonZero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0

/-- For every positive integer power of 2, there exists a multiple of it 
    such that all the digits (in decimal) are non-zero -/
theorem exists_nonzero_digits_multiple_of_power_of_two :
  ∀ k : ℕ+, ∃ n : ℕ, (2^k.val ∣ n) ∧ allDigitsNonZero n :=
sorry

end NUMINAMATH_CALUDE_exists_nonzero_digits_multiple_of_power_of_two_l286_28670


namespace NUMINAMATH_CALUDE_car_wash_earnings_l286_28698

theorem car_wash_earnings (total : ℝ) (lisa : ℝ) (tommy : ℝ) : 
  total = 60 →
  lisa = total / 2 →
  tommy = lisa / 2 →
  lisa - tommy = 15 := by
sorry

end NUMINAMATH_CALUDE_car_wash_earnings_l286_28698


namespace NUMINAMATH_CALUDE_favorite_numbers_exist_l286_28652

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem favorite_numbers_exist : ∃ (a b c : ℕ), 
  a * b * c = 71668 ∧ 
  a * sum_of_digits a = 10 * a ∧ 
  b * sum_of_digits b = 10 * b ∧ 
  c * sum_of_digits c = 10 * c ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end NUMINAMATH_CALUDE_favorite_numbers_exist_l286_28652


namespace NUMINAMATH_CALUDE_workers_completion_time_l286_28675

theorem workers_completion_time (A B : ℝ) : 
  (A > 0) →  -- A's completion time is positive
  (B > 0) →  -- B's completion time is positive
  ((2/3) * B + B * (1 - (2*B)/(3*A)) = A*B/(A+B) + 2) →  -- Total time equation
  ((A*B)/(A+B) * (1/A) = (1/2) * (1 - (2*B)/(3*A))) →  -- A's work proportion equation
  (A = 6 ∧ B = 3) := by
  sorry

end NUMINAMATH_CALUDE_workers_completion_time_l286_28675


namespace NUMINAMATH_CALUDE_division_problem_l286_28635

theorem division_problem :
  ∃ (quotient : ℕ),
    15968 = 179 * quotient + 37 ∧
    quotient = 89 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l286_28635


namespace NUMINAMATH_CALUDE_min_value_trig_function_min_value_attainable_l286_28696

theorem min_value_trig_function (θ : Real) (h : 1 - Real.cos θ ≠ 0) :
  (2 - Real.sin θ) / (1 - Real.cos θ) ≥ 3/4 :=
by sorry

theorem min_value_attainable :
  ∃ θ : Real, (1 - Real.cos θ ≠ 0) ∧ (2 - Real.sin θ) / (1 - Real.cos θ) = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_function_min_value_attainable_l286_28696


namespace NUMINAMATH_CALUDE_chicken_count_l286_28663

/-- The number of rabbits on the farm -/
def rabbits : ℕ := 49

/-- The number of frogs on the farm -/
def frogs : ℕ := 37

/-- The number of chickens on the farm -/
def chickens : ℕ := 21

/-- The total number of frogs and chickens is 9 more than the number of rabbits -/
axiom farm_equation : frogs + chickens = rabbits + 9

theorem chicken_count : chickens = 21 := by
  sorry

end NUMINAMATH_CALUDE_chicken_count_l286_28663


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l286_28694

/-- A circle passing through three points (0,0), (4,0), and (-1,1) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

/-- The three points that the circle passes through -/
def point1 : ℝ × ℝ := (0, 0)
def point2 : ℝ × ℝ := (4, 0)
def point3 : ℝ × ℝ := (-1, 1)

theorem circle_passes_through_points :
  circle_equation point1.1 point1.2 ∧
  circle_equation point2.1 point2.2 ∧
  circle_equation point3.1 point3.2 :=
sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l286_28694


namespace NUMINAMATH_CALUDE_like_terms_exponents_l286_28624

theorem like_terms_exponents (m n : ℤ) : 
  (∀ x y : ℝ, ∃ k : ℝ, -3 * x^(m-1) * y^3 = k * (5/2 * x^n * y^(m+n))) → 
  m = 2 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l286_28624


namespace NUMINAMATH_CALUDE_percentage_increase_l286_28676

theorem percentage_increase (initial : ℝ) (final : ℝ) : initial = 1200 → final = 1680 → (final - initial) / initial * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l286_28676


namespace NUMINAMATH_CALUDE_cube_sum_is_90_l286_28605

-- Define the type for the cube faces
def CubeFaces := Fin 6 → ℕ

-- Define the property of consecutive even numbers
def ConsecutiveEven (faces : CubeFaces) : Prop :=
  ∃ n : ℕ, ∀ i : Fin 6, faces i = 2 * (n + i.val)

-- Define the property of opposite face sums being equal
def OppositeFaceSumsEqual (faces : CubeFaces) : Prop :=
  ∃ s : ℕ, 
    faces 0 + faces 5 + 2 = s ∧
    faces 1 + faces 4 + 2 = s ∧
    faces 2 + faces 3 + 2 = s

-- Theorem statement
theorem cube_sum_is_90 (faces : CubeFaces) 
  (h1 : ConsecutiveEven faces) 
  (h2 : OppositeFaceSumsEqual faces) : 
  (faces 0 + faces 1 + faces 2 + faces 3 + faces 4 + faces 5 = 90) :=
sorry

end NUMINAMATH_CALUDE_cube_sum_is_90_l286_28605


namespace NUMINAMATH_CALUDE_complex_ratio_condition_l286_28600

theorem complex_ratio_condition (z : ℂ) :
  let x := z.re
  let y := z.im
  (((x + 5)^2 - y^2) / (2 * (x + 5) * y) = -3/4) ↔
  ((x + 2*y + 5) * (x - y/2 + 5) = 0 ∧ (x + 5) * y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_ratio_condition_l286_28600


namespace NUMINAMATH_CALUDE_correct_quotient_l286_28648

theorem correct_quotient (N : ℕ) : 
  (N / 7 = 12 ∧ N % 7 = 5) → N / 8 = 11 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_l286_28648


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l286_28679

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a / b = 5 / 4 →  -- ratio of measures is 5:4
  abs (a - b) = 10 :=  -- positive difference is 10°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l286_28679


namespace NUMINAMATH_CALUDE_ellipse_inscribed_parallelogram_slope_product_l286_28637

/-- Given an ellipse Γ: x²/3 + y²/2 = 1, with a parallelogram ABCD inscribed in it
    such that BD is a diagonal and B and D are symmetric about the origin,
    the product of the slopes of adjacent sides AB and BC is equal to -2/3. -/
theorem ellipse_inscribed_parallelogram_slope_product
  (Γ : Set (ℝ × ℝ))
  (h_ellipse : Γ = {(x, y) | x^2/3 + y^2/2 = 1})
  (A B C D : ℝ × ℝ)
  (h_inscribed : A ∈ Γ ∧ B ∈ Γ ∧ C ∈ Γ ∧ D ∈ Γ)
  (h_parallelogram : (A.1 + C.1 = B.1 + D.1) ∧ (A.2 + C.2 = B.2 + D.2))
  (h_diagonal : B.1 + D.1 = 0 ∧ B.2 + D.2 = 0)
  (k₁ k₂ : ℝ)
  (h_slope_AB : k₁ = (B.2 - A.2) / (B.1 - A.1))
  (h_slope_BC : k₂ = (C.2 - B.2) / (C.1 - B.1)) :
  k₁ * k₂ = -2/3 := by sorry

end NUMINAMATH_CALUDE_ellipse_inscribed_parallelogram_slope_product_l286_28637


namespace NUMINAMATH_CALUDE_shopkeeper_loss_percentage_l286_28686

theorem shopkeeper_loss_percentage
  (initial_value : ℝ)
  (profit_percentage : ℝ)
  (stolen_percentage : ℝ)
  (sales_tax_percentage : ℝ)
  (h_profit : profit_percentage = 20)
  (h_stolen : stolen_percentage = 85)
  (h_tax : sales_tax_percentage = 5)
  (h_positive : initial_value > 0) :
  let selling_price := initial_value * (1 + profit_percentage / 100)
  let remaining_value := initial_value * (1 - stolen_percentage / 100)
  let after_tax_value := remaining_value * (1 - sales_tax_percentage / 100)
  let loss := selling_price - after_tax_value
  loss / selling_price * 100 = 88.125 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_percentage_l286_28686


namespace NUMINAMATH_CALUDE_game_end_conditions_l286_28626

/-- Represents a game board of size n × n with k game pieces -/
structure GameBoard (n : ℕ) (k : ℕ) where
  size : n ≥ 2
  pieces : k ≥ 0

/-- Determines if the game never ends for any initial arrangement -/
def never_ends (n : ℕ) (k : ℕ) : Prop :=
  k > 3 * n^2 - 4 * n

/-- Determines if the game always ends for any initial arrangement -/
def always_ends (n : ℕ) (k : ℕ) : Prop :=
  k < 2 * n^2 - 2 * n

/-- Theorem stating the conditions for the game to never end or always end -/
theorem game_end_conditions (n : ℕ) (k : ℕ) (board : GameBoard n k) :
  (never_ends n k ↔ k > 3 * n^2 - 4 * n) ∧
  (always_ends n k ↔ k < 2 * n^2 - 2 * n) :=
sorry

end NUMINAMATH_CALUDE_game_end_conditions_l286_28626


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l286_28689

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
def midpoint_octagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The area of the midpoint octagon is 1/4 of the original octagon's area -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpoint_octagon o) = (1/4) * area o := by
  sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l286_28689


namespace NUMINAMATH_CALUDE_dhoni_leftover_percentage_l286_28656

/-- Represents Dhoni's spending and savings as percentages of his monthly earnings -/
structure DhoniFinances where
  rent_percent : ℝ
  dishwasher_percent : ℝ
  leftover_percent : ℝ

/-- Calculates Dhoni's finances based on given conditions -/
def calculate_finances (rent_percent : ℝ) : DhoniFinances :=
  let dishwasher_percent := rent_percent - (0.1 * rent_percent)
  let spent_percent := rent_percent + dishwasher_percent
  let leftover_percent := 100 - spent_percent
  { rent_percent := rent_percent,
    dishwasher_percent := dishwasher_percent,
    leftover_percent := leftover_percent }

/-- Theorem stating that Dhoni has 52.5% of his earnings left over -/
theorem dhoni_leftover_percentage :
  (calculate_finances 25).leftover_percent = 52.5 := by sorry

end NUMINAMATH_CALUDE_dhoni_leftover_percentage_l286_28656


namespace NUMINAMATH_CALUDE_premium_increases_after_accident_l286_28625

/-- Represents an insurance policy -/
structure InsurancePolicy where
  premium : ℝ
  hadAccident : Bool

/-- Represents an insurance company's policy for premium adjustment -/
class InsuranceCompany where
  adjustPremium : InsurancePolicy → ℝ

/-- Theorem: Insurance premium increases after an accident -/
theorem premium_increases_after_accident (company : InsuranceCompany) 
  (policy : InsurancePolicy) (h : policy.hadAccident = true) : 
  company.adjustPremium policy > policy.premium := by
  sorry

#check premium_increases_after_accident

end NUMINAMATH_CALUDE_premium_increases_after_accident_l286_28625


namespace NUMINAMATH_CALUDE_circle_configuration_l286_28602

-- Define the types of people
inductive PersonType
| Knight
| Liar
| Visitor

-- Define a person
structure Person where
  id : Fin 7
  type : PersonType

-- Define the circle of people
def Circle := Fin 7 → Person

-- Define a statement made by a pair of people
structure Statement where
  speaker1 : Fin 7
  speaker2 : Fin 7
  content : Nat
  category : PersonType

-- Define the function to check if a statement is true
def isStatementTrue (c : Circle) (s : Statement) : Prop :=
  (c s.speaker1).type = PersonType.Knight ∨
  (c s.speaker2).type = PersonType.Knight ∨
  ((c s.speaker1).type = PersonType.Visitor ∧ (c s.speaker2).type = PersonType.Visitor)

-- Define the list of statements
def statements : List Statement := [
  ⟨0, 1, 1, PersonType.Liar⟩,
  ⟨1, 2, 2, PersonType.Knight⟩,
  ⟨2, 3, 3, PersonType.Liar⟩,
  ⟨3, 4, 4, PersonType.Knight⟩,
  ⟨4, 5, 5, PersonType.Liar⟩,
  ⟨5, 6, 6, PersonType.Knight⟩,
  ⟨6, 0, 7, PersonType.Liar⟩
]

-- Define the theorem
theorem circle_configuration (c : Circle) :
  (∀ s ∈ statements, isStatementTrue c s ∨ ¬isStatementTrue c s) →
  (∃! (i j : Fin 7), i ≠ j ∧ 
    (c i).type = PersonType.Visitor ∧ 
    (c j).type = PersonType.Visitor ∧
    (∀ k : Fin 7, k ≠ i ∧ k ≠ j → (c k).type = PersonType.Liar)) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_configuration_l286_28602


namespace NUMINAMATH_CALUDE_floor_sum_abcd_l286_28666

theorem floor_sum_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^2 + b^2 = 1458) (h2 : c^2 + d^2 = 1458) (h3 : a * c = 1156) (h4 : b * d = 1156) :
  ⌊a + b + c + d⌋ = 77 := by sorry

end NUMINAMATH_CALUDE_floor_sum_abcd_l286_28666


namespace NUMINAMATH_CALUDE_circle_radius_zero_circle_equation_implies_zero_radius_l286_28615

theorem circle_radius_zero (x y : ℝ) :
  x^2 + 8*x + y^2 - 4*y + 20 = 0 → (x + 4)^2 + (y - 2)^2 = 0 := by
  sorry

theorem circle_equation_implies_zero_radius :
  ∃ (x y : ℝ), x^2 + 8*x + y^2 - 4*y + 20 = 0 → 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_zero_circle_equation_implies_zero_radius_l286_28615


namespace NUMINAMATH_CALUDE_probability_blue_given_glass_l286_28680

theorem probability_blue_given_glass (total_red : ℕ) (total_blue : ℕ)
  (red_glass : ℕ) (red_wooden : ℕ) (blue_glass : ℕ) (blue_wooden : ℕ)
  (h1 : total_red = red_glass + red_wooden)
  (h2 : total_blue = blue_glass + blue_wooden)
  (h3 : total_red = 5)
  (h4 : total_blue = 11)
  (h5 : red_glass = 2)
  (h6 : red_wooden = 3)
  (h7 : blue_glass = 4)
  (h8 : blue_wooden = 7) :
  (blue_glass : ℚ) / (red_glass + blue_glass) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_probability_blue_given_glass_l286_28680


namespace NUMINAMATH_CALUDE_floor_plus_x_equation_l286_28649

theorem floor_plus_x_equation (x : ℝ) : (⌊x⌋ : ℝ) + x = 20.5 ↔ x = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_x_equation_l286_28649


namespace NUMINAMATH_CALUDE_brandon_job_applications_l286_28617

theorem brandon_job_applications (total_businesses : ℕ) 
  (h1 : total_businesses = 72) 
  (fired : ℕ) (h2 : fired = total_businesses / 2)
  (quit : ℕ) (h3 : quit = total_businesses / 3) : 
  total_businesses - (fired + quit) = 12 :=
by sorry

end NUMINAMATH_CALUDE_brandon_job_applications_l286_28617


namespace NUMINAMATH_CALUDE_spheres_touching_triangle_and_other_spheres_l286_28638

/-- Given a scalene triangle ABC with sides a, b, c and circumradius R,
    prove the existence of two spheres with radii r and ρ (ρ > r) that touch
    the plane of the triangle and three other spheres (with radii r_A, r_B, r_C)
    that touch the triangle at its vertices, such that 1/r - 1/ρ = 2√3/R. -/
theorem spheres_touching_triangle_and_other_spheres
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hscalene : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (R : ℝ) (hR : R > 0)
  (r_A r_B r_C : ℝ)
  (hr_A : r_A = b * c / (2 * a))
  (hr_B : r_B = c * a / (2 * b))
  (hr_C : r_C = a * b / (2 * c)) :
  ∃ (r ρ : ℝ), r > 0 ∧ ρ > r ∧ 1/r - 1/ρ = 2 * Real.sqrt 3 / R :=
sorry

end NUMINAMATH_CALUDE_spheres_touching_triangle_and_other_spheres_l286_28638


namespace NUMINAMATH_CALUDE_f_properties_l286_28681

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x * Real.cos x - 5 * Real.sqrt 3 * (Real.cos x)^2 + 5/2 * Real.sqrt 3

theorem f_properties :
  let T := Real.pi
  ∀ (k : ℤ),
    (∀ (x : ℝ), f (x + T) = f x) ∧  -- f has period T
    (∀ (S : ℝ), S > 0 → (∀ (x : ℝ), f (x + S) = f x) → S ≥ T) ∧  -- T is the smallest positive period
    (∀ (x : ℝ), x ∈ Set.Icc (k * Real.pi - Real.pi/12) (k * Real.pi + 5 * Real.pi/12) → 
      ∀ (y : ℝ), y ∈ Set.Icc (k * Real.pi - Real.pi/12) (k * Real.pi + 5 * Real.pi/12) → 
        x ≤ y → f x ≤ f y) ∧  -- f is increasing on [kπ - π/12, kπ + 5π/12]
    (∀ (x : ℝ), x ∈ Set.Icc (k * Real.pi + 5 * Real.pi/12) (k * Real.pi + 11 * Real.pi/12) → 
      ∀ (y : ℝ), y ∈ Set.Icc (k * Real.pi + 5 * Real.pi/12) (k * Real.pi + 11 * Real.pi/12) → 
        x ≤ y → f x ≥ f y)  -- f is decreasing on [kπ + 5π/12, kπ + 11π/12]
  := by sorry

end NUMINAMATH_CALUDE_f_properties_l286_28681


namespace NUMINAMATH_CALUDE_quadratic_roots_characterization_l286_28601

/-- The quadratic equation a² - 18a + 72 = 0 has solutions a = 6 and a = 12 -/
def quad_eq (a : ℝ) : Prop := a^2 - 18*a + 72 = 0

/-- The general form of the roots -/
def root_form (a x : ℝ) : Prop := x = a + Real.sqrt (18*(a-4)) ∨ x = a - Real.sqrt (18*(a-4))

/-- Condition for distinct positive roots -/
def distinct_positive_roots (a : ℝ) : Prop :=
  (4 < a ∧ a < 6) ∨ a > 12

/-- Condition for equal roots -/
def equal_roots (a : ℝ) : Prop :=
  (6 ≤ a ∧ a ≤ 12) ∨ a = 22

theorem quadratic_roots_characterization :
  ∀ a : ℝ, quad_eq a →
    (∃ x y : ℝ, x ≠ y ∧ root_form a x ∧ root_form a y ∧ x > 0 ∧ y > 0 ↔ distinct_positive_roots a) ∧
    (∃ x : ℝ, root_form a x ∧ x > 0 ↔ equal_roots a) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_characterization_l286_28601


namespace NUMINAMATH_CALUDE_largest_s_value_l286_28622

/-- The interior angle of a regular n-gon -/
def interior_angle (n : ℕ) : ℚ :=
  (n - 2 : ℚ) * 180 / n

/-- The largest possible value of s for regular polygons Q_1 (r-gon) and Q_2 (s-gon) -/
theorem largest_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3) 
  (h_ratio : interior_angle r / interior_angle s = 39 / 38) : 
  s ≤ 76 ∧ ∃ (r' : ℕ), r' ≥ 76 ∧ interior_angle r' / interior_angle 76 = 39 / 38 :=
sorry

end NUMINAMATH_CALUDE_largest_s_value_l286_28622


namespace NUMINAMATH_CALUDE_sequence_sum_problem_l286_28672

theorem sequence_sum_problem (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 5)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 14)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 30)
  (eq4 : 16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 70) :
  25*x₁ + 36*x₂ + 49*x₃ + 64*x₄ + 81*x₅ + 100*x₆ + 121*x₇ = 130 := by
  sorry


end NUMINAMATH_CALUDE_sequence_sum_problem_l286_28672


namespace NUMINAMATH_CALUDE_cookie_radius_l286_28607

/-- The equation of the cookie boundary -/
def cookie_boundary (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 4

/-- The cookie is a circle -/
def is_circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∀ x y : ℝ, cookie_boundary x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2

theorem cookie_radius :
  ∃ center : ℝ × ℝ, is_circle center 3 :=
sorry

end NUMINAMATH_CALUDE_cookie_radius_l286_28607


namespace NUMINAMATH_CALUDE_pens_count_in_second_set_l286_28657

/-- The cost of a pencil in dollars -/
def pencil_cost : ℚ := 1/10

/-- The cost of 3 pencils and some pens in dollars -/
def first_set_cost : ℚ := 158/100

/-- The cost of 4 pencils and 5 pens in dollars -/
def second_set_cost : ℚ := 2

/-- The number of pens in the second set -/
def pens_in_second_set : ℕ := 5

/-- Theorem stating that given the conditions, the number of pens in the second set is 5 -/
theorem pens_count_in_second_set : 
  ∃ (pen_cost : ℚ) (pens_in_first_set : ℕ), 
    3 * pencil_cost + pens_in_first_set * pen_cost = first_set_cost ∧
    4 * pencil_cost + pens_in_second_set * pen_cost = second_set_cost :=
by
  sorry

#check pens_count_in_second_set

end NUMINAMATH_CALUDE_pens_count_in_second_set_l286_28657


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l286_28613

theorem fixed_point_of_exponential_function (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(4-x) + 3
  f 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l286_28613


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l286_28618

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 1

-- Theorem statement
theorem tangent_parallel_points :
  ∀ x y : ℝ, (y = f x) ∧ (f' x = 4) ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l286_28618


namespace NUMINAMATH_CALUDE_distance_between_points_with_given_distances_from_origin_l286_28655

def distance_between_points (a b : ℝ) : ℝ := |a - b|

theorem distance_between_points_with_given_distances_from_origin :
  ∀ (a b : ℝ),
  distance_between_points 0 a = 2 →
  distance_between_points 0 b = 7 →
  distance_between_points a b = 5 ∨ distance_between_points a b = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_with_given_distances_from_origin_l286_28655


namespace NUMINAMATH_CALUDE_room_population_lower_limit_l286_28623

theorem room_population_lower_limit :
  ∀ (P : ℕ),
  (P < 100) →
  ((3 : ℚ) / 8 * P = 36) →
  (∃ (n : ℕ), (5 : ℚ) / 12 * P = n) →
  P ≥ 96 :=
by
  sorry

end NUMINAMATH_CALUDE_room_population_lower_limit_l286_28623


namespace NUMINAMATH_CALUDE_quadratic_function_uniqueness_l286_28685

/-- A quadratic function f(x) = ax^2 + bx + c with roots at -2 and 4, and maximum value 9 -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    (∀ x, f x = a * x^2 + b * x + c) ∧
    f (-2) = 0 ∧
    f 4 = 0 ∧
    (∀ x, f x ≤ 9) ∧
    (∃ x₀, f x₀ = 9)

theorem quadratic_function_uniqueness (f : ℝ → ℝ) (hf : QuadraticFunction f) :
  ∀ x, f x = -x^2 + 2*x + 8 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_uniqueness_l286_28685


namespace NUMINAMATH_CALUDE_problem_solution_l286_28671

theorem problem_solution :
  -- Part 1(i)
  (∀ a b : ℝ, a + b = 13 ∧ a * b = 36 → (a - b)^2 = 25) ∧
  -- Part 1(ii)
  (∀ a b : ℝ, a^2 + a*b = 8 ∧ b^2 + a*b = 1 → 
    (a = 8/3 ∧ b = 1/3) ∨ (a = -8/3 ∧ b = -1/3)) ∧
  -- Part 2
  (∀ a b x y : ℝ, 
    a*x + b*y = 3 ∧ 
    a*x^2 + b*y^2 = 7 ∧ 
    a*x^3 + b*y^3 = 16 ∧ 
    a*x^4 + b*y^4 = 42 → 
    x + y = -14) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l286_28671


namespace NUMINAMATH_CALUDE_angle_measure_120_l286_28664

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (hsum : A + B + C = π)

-- State the theorem
theorem angle_measure_120 (t : Triangle) (h : t.a^2 = t.b^2 + t.b*t.c + t.c^2) :
  t.A = 2*π/3 := by sorry

end NUMINAMATH_CALUDE_angle_measure_120_l286_28664


namespace NUMINAMATH_CALUDE_apples_packed_in_two_weeks_l286_28660

/-- Calculates the total number of apples packed in two weeks under specific conditions -/
theorem apples_packed_in_two_weeks
  (apples_per_box : ℕ)
  (boxes_per_day : ℕ)
  (days_per_week : ℕ)
  (fewer_apples_second_week : ℕ)
  (h1 : apples_per_box = 40)
  (h2 : boxes_per_day = 50)
  (h3 : days_per_week = 7)
  (h4 : fewer_apples_second_week = 500) :
  apples_per_box * boxes_per_day * days_per_week +
  (apples_per_box * boxes_per_day - fewer_apples_second_week) * days_per_week = 24500 :=
by sorry

#check apples_packed_in_two_weeks

end NUMINAMATH_CALUDE_apples_packed_in_two_weeks_l286_28660


namespace NUMINAMATH_CALUDE_arithmetic_sequence_average_l286_28611

/-- 
Given an arithmetic sequence with:
- First term a₁ = 10
- Last term aₙ = 160
- Common difference d = 10

Prove that the average (arithmetic mean) of this sequence is 85.
-/
theorem arithmetic_sequence_average : 
  let a₁ : ℕ := 10
  let aₙ : ℕ := 160
  let d : ℕ := 10
  let n : ℕ := (aₙ - a₁) / d + 1
  (a₁ + aₙ) / 2 = 85 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_average_l286_28611


namespace NUMINAMATH_CALUDE_units_digit_of_2_pow_2023_l286_28643

theorem units_digit_of_2_pow_2023 : 2^2023 % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2_pow_2023_l286_28643


namespace NUMINAMATH_CALUDE_inequalities_theorem_l286_28673

theorem inequalities_theorem (a b c : ℝ) 
  (ha : a < 0) 
  (hab : a < b) 
  (hb : b ≤ 0) 
  (hbc : b < c) : 
  (a + b < b + c) ∧ (c / a < 1) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l286_28673


namespace NUMINAMATH_CALUDE_inequality_proof_l286_28659

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y)) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l286_28659


namespace NUMINAMATH_CALUDE_transform_equation_5x2_eq_6x_minus_8_l286_28616

/-- Represents a quadratic equation in general form ax² + bx + c = 0 --/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0

/-- Transforms an equation of the form px² = qx + r into general quadratic form --/
def transform_to_general_form (p q r : ℝ) (hp : p ≠ 0) : QuadraticEquation :=
  { a := p
  , b := -q
  , c := r
  , h := hp }

theorem transform_equation_5x2_eq_6x_minus_8 :
  let eq := transform_to_general_form 5 6 (-8) (by norm_num)
  eq.a = 5 ∧ eq.b = -6 ∧ eq.c = 8 := by sorry

end NUMINAMATH_CALUDE_transform_equation_5x2_eq_6x_minus_8_l286_28616


namespace NUMINAMATH_CALUDE_rectangle_segment_relation_l286_28627

-- Define the points and segments
variable (A B C D E F : EuclideanPlane) (BE CD AD AC BC : ℝ)

-- Define the conditions
variable (h1 : IsRectangle C D E F)
variable (h2 : A ∈ SegmentOpen E D)
variable (h3 : B ∈ Line E F)
variable (h4 : B ∈ PerpendicularLine A C C)

-- State the theorem
theorem rectangle_segment_relation :
  BE = CD + (AD / AC) * BC :=
sorry

end NUMINAMATH_CALUDE_rectangle_segment_relation_l286_28627


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l286_28667

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : ∀ x : ℝ, x^2 + c*x + d = 0 ↔ x = c ∨ x = d) : 
  c = 1 ∧ d = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l286_28667


namespace NUMINAMATH_CALUDE_prob_select_all_leaders_in_district_l286_28603

/-- Represents a math club with a given number of students and leaders -/
structure MathClub where
  students : Nat
  leaders : Nat

/-- Calculates the probability of selecting all leaders in a given club -/
def prob_select_all_leaders (club : MathClub) : Rat :=
  (club.students - club.leaders).choose 1 / club.students.choose 4

/-- The list of math clubs in the school district -/
def math_clubs : List MathClub := [
  ⟨6, 3⟩,
  ⟨8, 3⟩,
  ⟨9, 3⟩,
  ⟨10, 3⟩
]

/-- The main theorem stating the probability of selecting all leaders -/
theorem prob_select_all_leaders_in_district : 
  (1 / 4 : Rat) * (math_clubs.map prob_select_all_leaders).sum = 37 / 420 := by
  sorry

end NUMINAMATH_CALUDE_prob_select_all_leaders_in_district_l286_28603


namespace NUMINAMATH_CALUDE_min_value_expression_l286_28640

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 + y) / x + 1 / y ≥ 2 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l286_28640


namespace NUMINAMATH_CALUDE_beth_crayons_l286_28608

/-- The number of crayons Beth has altogether -/
def total_crayons (packs : ℕ) (crayons_per_pack : ℕ) (extra_crayons : ℕ) : ℕ :=
  packs * crayons_per_pack + extra_crayons

/-- Theorem stating that Beth has 175 crayons in total -/
theorem beth_crayons : total_crayons 8 20 15 = 175 := by
  sorry

end NUMINAMATH_CALUDE_beth_crayons_l286_28608


namespace NUMINAMATH_CALUDE_number_division_problem_l286_28661

theorem number_division_problem (x y : ℝ) : 
  (x - 5) / y = 7 → 
  (x - 14) / 10 = 4 → 
  y = 7 := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l286_28661


namespace NUMINAMATH_CALUDE_rectangle_ratio_l286_28690

/-- A configuration of squares and a rectangle forming a larger square -/
structure SquareConfiguration where
  /-- Side length of each small square -/
  s : ℝ
  /-- Side length of the large square -/
  bigSquareSide : ℝ
  /-- Length of the rectangle -/
  rectLength : ℝ
  /-- Width of the rectangle -/
  rectWidth : ℝ
  /-- The large square's side is 5 times the small square's side -/
  bigSquare_eq : bigSquareSide = 5 * s
  /-- The rectangle's length is equal to the large square's side -/
  rectLength_eq : rectLength = bigSquareSide
  /-- The rectangle's width is the large square's side minus 4 small square sides -/
  rectWidth_eq : rectWidth = bigSquareSide - 4 * s

/-- The ratio of the rectangle's length to its width is 5 -/
theorem rectangle_ratio (config : SquareConfiguration) :
    config.rectLength / config.rectWidth = 5 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_ratio_l286_28690


namespace NUMINAMATH_CALUDE_P_intersect_Q_eq_P_l286_28610

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {y | ∃ x, y = Real.cos x}

theorem P_intersect_Q_eq_P : P ∩ Q = P := by sorry

end NUMINAMATH_CALUDE_P_intersect_Q_eq_P_l286_28610


namespace NUMINAMATH_CALUDE_lines_cannot_form_triangle_l286_28631

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if three lines intersect at a single point -/
def intersect_at_point (l1 l2 l3 : Line) : Prop :=
  ∃ x y : ℝ, l1.a * x + l1.b * y + l1.c = 0 ∧
             l2.a * x + l2.b * y + l2.c = 0 ∧
             l3.a * x + l3.b * y + l3.c = 0

/-- The set of m values for which the three lines cannot form a triangle -/
def m_values : Set ℝ := {-3, 2, -1}

theorem lines_cannot_form_triangle (m : ℝ) :
  let l1 : Line := ⟨3, -1, 2⟩
  let l2 : Line := ⟨2, 1, 3⟩
  let l3 : Line := ⟨m, 1, 0⟩
  (parallel l1 l3 ∨ parallel l2 l3 ∨ intersect_at_point l1 l2 l3) ↔ m ∈ m_values := by
  sorry

end NUMINAMATH_CALUDE_lines_cannot_form_triangle_l286_28631


namespace NUMINAMATH_CALUDE_l₂_parallel_and_through_A_B_symmetric_to_A_l286_28628

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := 2 * x + 4 * y - 1 = 0

-- Define point A
def A : ℝ × ℝ := (3, 0)

-- Define the parallel line l₂ passing through A
def l₂ (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Define point B
def B : ℝ × ℝ := (2, -2)

-- Theorem 1: l₂ is parallel to l₁ and passes through A
theorem l₂_parallel_and_through_A :
  (∀ x y : ℝ, l₂ x y ↔ ∃ k : ℝ, k ≠ 0 ∧ 2 * x + 4 * y - 1 = k * (2 * A.1 + 4 * A.2 - 1)) ∧
  l₂ A.1 A.2 :=
sorry

-- Theorem 2: B is symmetric to A with respect to l₁
theorem B_symmetric_to_A :
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  l₁ midpoint.1 midpoint.2 ∧
  (B.2 - A.2) / (B.1 - A.1) = - (1 / (2 / 4)) :=
sorry

end NUMINAMATH_CALUDE_l₂_parallel_and_through_A_B_symmetric_to_A_l286_28628


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l286_28644

/-- Represents the number of students in each group -/
structure StudentCount where
  male : ℕ
  female : ℕ

/-- Represents the number of students to be sampled from each group -/
structure SampleCount where
  male : ℕ
  female : ℕ

/-- Calculates the correct stratified sample given the total student count and sample size -/
def stratifiedSample (students : StudentCount) (sampleSize : ℕ) : SampleCount :=
  { male := (students.male * sampleSize) / (students.male + students.female),
    female := (students.female * sampleSize) / (students.male + students.female) }

theorem correct_stratified_sample :
  let students : StudentCount := { male := 20, female := 30 }
  let sampleSize : ℕ := 10
  let sample := stratifiedSample students sampleSize
  sample.male = 4 ∧ sample.female = 6 := by sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l286_28644


namespace NUMINAMATH_CALUDE_special_hexagon_perimeter_l286_28669

/-- An equilateral hexagon with specific properties -/
structure SpecialHexagon where
  -- The side length of the hexagon
  side : ℝ
  -- Assertion that four nonadjacent interior angles are 45°
  has_four_45_angles : Bool
  -- The area of the hexagon
  area : ℝ
  -- The area is 12√2
  area_is_12_root_2 : area = 12 * Real.sqrt 2

/-- The perimeter of a hexagon is 6 times its side length -/
def perimeter (h : SpecialHexagon) : ℝ := 6 * h.side

/-- Theorem stating the perimeter of the special hexagon is 6√6 -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : 
  perimeter h = 6 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_special_hexagon_perimeter_l286_28669
