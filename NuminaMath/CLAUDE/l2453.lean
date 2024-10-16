import Mathlib

namespace NUMINAMATH_CALUDE_camping_site_problem_l2453_245380

theorem camping_site_problem (total : ℕ) (two_weeks_ago : ℕ) (difference : ℕ) :
  total = 150 →
  two_weeks_ago = 40 →
  difference = 10 →
  ∃ (three_weeks_ago last_week : ℕ),
    three_weeks_ago + two_weeks_ago + last_week = total ∧
    two_weeks_ago = three_weeks_ago + difference ∧
    last_week = 80 :=
by
  sorry

#check camping_site_problem

end NUMINAMATH_CALUDE_camping_site_problem_l2453_245380


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2453_245367

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^5 + 5 * X^4 - 13 * X^3 - 7 * X^2 + 52 * X - 34 = 
  (X^3 + 6 * X^2 + 5 * X - 7) * q + (50 * X^3 + 79 * X^2 - 39 * X - 34) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2453_245367


namespace NUMINAMATH_CALUDE_product_218_5_base9_l2453_245386

/-- Convert a base-9 number to base-10 --/
def base9ToBase10 (n : ℕ) : ℕ := sorry

/-- Convert a base-10 number to base-9 --/
def base10ToBase9 (n : ℕ) : ℕ := sorry

/-- Multiply two base-9 numbers and return the result in base-9 --/
def multiplyBase9 (a b : ℕ) : ℕ :=
  base10ToBase9 (base9ToBase10 a * base9ToBase10 b)

theorem product_218_5_base9 :
  multiplyBase9 218 5 = 1204 := by sorry

end NUMINAMATH_CALUDE_product_218_5_base9_l2453_245386


namespace NUMINAMATH_CALUDE_fraction_invariance_l2453_245300

theorem fraction_invariance (a b m n : ℚ) (h : b ≠ 0) (h' : b + n ≠ 0) :
  a / b = m / n → (a + m) / (b + n) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_invariance_l2453_245300


namespace NUMINAMATH_CALUDE_kids_difference_l2453_245379

/-- The number of kids Julia played with on Monday and Tuesday, and the difference between them. -/
def tag_game (monday tuesday : ℕ) : Prop :=
  monday = 16 ∧ tuesday = 4 ∧ monday - tuesday = 12

/-- Theorem stating the difference in the number of kids Julia played with. -/
theorem kids_difference : ∃ (monday tuesday : ℕ), tag_game monday tuesday :=
  sorry

end NUMINAMATH_CALUDE_kids_difference_l2453_245379


namespace NUMINAMATH_CALUDE_monotonic_function_value_l2453_245394

/-- A monotonically increasing function f: ℝ → ℝ satisfying f(f(x) - 2^x) = 3 for all x ∈ ℝ -/
def MonotonicFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (f x - 2^x) = 3)

/-- Theorem: For a monotonically increasing function f satisfying the given condition, f(3) = 9 -/
theorem monotonic_function_value (f : ℝ → ℝ) (h : MonotonicFunction f) : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_function_value_l2453_245394


namespace NUMINAMATH_CALUDE_symmetric_points_on_circle_l2453_245305

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 1 = 0

-- Define the line equation
def line_equation (x y : ℝ) (c : ℝ) : Prop :=
  2*x + y + c = 0

-- Theorem statement
theorem symmetric_points_on_circle (c : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_equation x₁ y₁ ∧
    circle_equation x₂ y₂ ∧
    (∃ (x_mid y_mid : ℝ),
      x_mid = (x₁ + x₂) / 2 ∧
      y_mid = (y₁ + y₂) / 2 ∧
      line_equation x_mid y_mid c)) →
  c = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_on_circle_l2453_245305


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2453_245301

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ (3^19 + 11^13) ∧ ∀ (q : ℕ), q.Prime → q ∣ (3^19 + 11^13) → p ≤ q :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2453_245301


namespace NUMINAMATH_CALUDE_euler_totient_even_bound_l2453_245326

theorem euler_totient_even_bound (n : ℕ) (h : Even n) (h_pos : n > 0) : 
  (Finset.filter (fun x => Nat.gcd n x = 1) (Finset.range n)).card ≤ n / 2 := by
  sorry

end NUMINAMATH_CALUDE_euler_totient_even_bound_l2453_245326


namespace NUMINAMATH_CALUDE_meal_combinations_count_l2453_245317

/-- Represents the number of main dishes available -/
def num_main_dishes : ℕ := 2

/-- Represents the number of stir-fry dishes available -/
def num_stir_fry_dishes : ℕ := 4

/-- Calculates the total number of meal combinations -/
def total_combinations : ℕ := num_main_dishes * num_stir_fry_dishes

/-- Theorem stating that the total number of meal combinations is 8 -/
theorem meal_combinations_count : total_combinations = 8 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_count_l2453_245317


namespace NUMINAMATH_CALUDE_horizontal_shift_right_l2453_245320

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define the horizontal shift
def horizontalShift (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  fun x ↦ f (x - a)

-- Theorem statement
theorem horizontal_shift_right (a : ℝ) :
  ∀ x : ℝ, (horizontalShift f a) x = f (x - a) :=
by
  sorry

-- Note: This theorem states that for all real x,
-- the horizontally shifted function is equal to f(x - a),
-- which is equivalent to shifting the graph of f(x) right by a units.

end NUMINAMATH_CALUDE_horizontal_shift_right_l2453_245320


namespace NUMINAMATH_CALUDE_king_probability_l2453_245371

/-- Custom deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (ranks : Nat)
  (one_card_per_rank_suit : cards = suits * ranks)

/-- Probability of drawing a specific rank -/
def prob_draw_rank (d : Deck) (rank_count : Nat) : ℚ :=
  rank_count / d.cards

theorem king_probability (d : Deck) (h1 : d.cards = 65) (h2 : d.suits = 5) (h3 : d.ranks = 13) :
  prob_draw_rank d d.suits = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_king_probability_l2453_245371


namespace NUMINAMATH_CALUDE_complex_sum_equals_eleven_l2453_245376

/-- Given complex numbers a and b, prove that a + 3b = 11 -/
theorem complex_sum_equals_eleven (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 2 + I) :
  a + 3*b = 11 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_eleven_l2453_245376


namespace NUMINAMATH_CALUDE_optimal_dimensions_maximize_volume_unique_maximum_volume_l2453_245311

/-- Represents the volume of a rectangular frame as a function of its width. -/
def volume (x : ℝ) : ℝ := 2 * x^2 * (4.5 - 3*x)

/-- The maximum volume of the rectangular frame. -/
def max_volume : ℝ := 3

/-- The width that maximizes the volume. -/
def optimal_width : ℝ := 1

/-- The length that maximizes the volume. -/
def optimal_length : ℝ := 2

/-- The height that maximizes the volume. -/
def optimal_height : ℝ := 1.5

/-- Theorem stating that the given dimensions maximize the volume of the rectangular frame. -/
theorem optimal_dimensions_maximize_volume :
  (∀ x, 0 < x → x < 3/2 → volume x ≤ max_volume) ∧
  volume optimal_width = max_volume ∧
  optimal_length = 2 * optimal_width ∧
  optimal_height = 4.5 - 3 * optimal_width :=
sorry

/-- Theorem stating that the maximum volume is unique. -/
theorem unique_maximum_volume :
  ∀ x, 0 < x → x < 3/2 → volume x = max_volume → x = optimal_width :=
sorry

end NUMINAMATH_CALUDE_optimal_dimensions_maximize_volume_unique_maximum_volume_l2453_245311


namespace NUMINAMATH_CALUDE_box_length_is_twelve_l2453_245360

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.height * d.width * d.length

/-- Theorem: If 40 building blocks with given dimensions can fit into a box with given height and width,
    then the length of the box is 12 inches -/
theorem box_length_is_twelve
  (box : Dimensions)
  (block : Dimensions)
  (h1 : box.height = 8)
  (h2 : box.width = 10)
  (h3 : block.height = 3)
  (h4 : block.width = 2)
  (h5 : block.length = 4)
  (h6 : volume box ≥ 40 * volume block) :
  box.length = 12 :=
sorry

end NUMINAMATH_CALUDE_box_length_is_twelve_l2453_245360


namespace NUMINAMATH_CALUDE_circle1_properties_circle2_and_circle3_properties_l2453_245327

-- Define the circle equations
def circle1 (x y : ℝ) : Prop := (x - 4)^2 + (y - 1)^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 13
def circle3 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 13

-- Define the line equations
def line1 (x y : ℝ) : Prop := x - 2*y - 2 = 0
def line2 (x y : ℝ) : Prop := 2*x + 3*y - 10 = 0

-- Theorem for the first circle
theorem circle1_properties :
  (∀ x y, circle1 x y → line1 x y) ∧
  circle1 0 4 ∧
  circle1 4 6 := by sorry

-- Theorem for the second and third circles
theorem circle2_and_circle3_properties :
  (∀ x y, (circle2 x y ∨ circle3 x y) → (x - 2)^2 + (y - 2)^2 = 13) ∧
  (∃ x y, (circle2 x y ∨ circle3 x y) ∧ line2 x y ∧ x = 2 ∧ y = 2) := by sorry

end NUMINAMATH_CALUDE_circle1_properties_circle2_and_circle3_properties_l2453_245327


namespace NUMINAMATH_CALUDE_gcd_36_54_l2453_245365

theorem gcd_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_36_54_l2453_245365


namespace NUMINAMATH_CALUDE_average_weight_solution_l2453_245344

def average_weight_problem (d e f : ℝ) : Prop :=
  (d + e + f) / 3 = 42 ∧
  (e + f) / 2 = 41 ∧
  e = 26 →
  (d + e) / 2 = 35

theorem average_weight_solution :
  ∀ d e f : ℝ, average_weight_problem d e f :=
by
  sorry

end NUMINAMATH_CALUDE_average_weight_solution_l2453_245344


namespace NUMINAMATH_CALUDE_doll_count_l2453_245335

/-- The number of dolls owned by the grandmother -/
def grandmother_dolls : ℕ := 50

/-- The number of dolls owned by the sister -/
def sister_dolls : ℕ := grandmother_dolls + 2

/-- The number of dolls owned by Rene -/
def rene_dolls : ℕ := 3 * sister_dolls

/-- The total number of dolls owned by all three people -/
def total_dolls : ℕ := grandmother_dolls + sister_dolls + rene_dolls

theorem doll_count : total_dolls = 258 := by
  sorry

end NUMINAMATH_CALUDE_doll_count_l2453_245335


namespace NUMINAMATH_CALUDE_complex_power_difference_l2453_245312

theorem complex_power_difference (x : ℂ) : 
  x - (1 / x) = 3 * Complex.I → x^3375 - (1 / x^3375) = -18 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l2453_245312


namespace NUMINAMATH_CALUDE_one_correct_statement_l2453_245336

theorem one_correct_statement : 
  (∃! n : ℕ, n = 1 ∧ 
    (∀ x : ℤ, x < 0 → x ≤ -1) ∧ 
    (∃ y : ℝ, -(y) ≤ 0) ∧
    (∃ z : ℚ, z = 0) ∧
    (∃ a : ℝ, -a > 0) ∧
    (∃ b₁ b₂ : ℚ, b₁ < 0 ∧ b₂ < 0 ∧ b₁ * b₂ > 0)) :=
by sorry

end NUMINAMATH_CALUDE_one_correct_statement_l2453_245336


namespace NUMINAMATH_CALUDE_conference_handshakes_l2453_245348

/-- The number of handshakes in a conference with n attendees -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a conference with 10 attendees, where each attendee shakes hands
    exactly once with every other attendee, the total number of handshakes is 45 -/
theorem conference_handshakes :
  handshakes 10 = 45 := by sorry

end NUMINAMATH_CALUDE_conference_handshakes_l2453_245348


namespace NUMINAMATH_CALUDE_bike_average_speed_l2453_245314

theorem bike_average_speed (initial_reading final_reading : ℕ) (total_time : ℝ) :
  initial_reading = 2332 →
  final_reading = 2552 →
  total_time = 9 →
  (final_reading - initial_reading : ℝ) / total_time = 220 / 9 := by
  sorry

end NUMINAMATH_CALUDE_bike_average_speed_l2453_245314


namespace NUMINAMATH_CALUDE_union_of_sets_l2453_245349

def A (a : ℝ) : Set ℝ := {-1, a}
def B (a b : ℝ) : Set ℝ := {2^a, b}

theorem union_of_sets (a b : ℝ) :
  (A a) ∩ (B a b) = {1} → (A a) ∪ (B a b) = {-1, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l2453_245349


namespace NUMINAMATH_CALUDE_brown_eyed_brunettes_l2453_245315

theorem brown_eyed_brunettes (total : ℕ) (blue_eyed_blondes : ℕ) (brunettes : ℕ) (brown_eyed : ℕ) :
  total = 60 →
  blue_eyed_blondes = 16 →
  brunettes = 36 →
  brown_eyed = 25 →
  (total - brunettes) - blue_eyed_blondes + brown_eyed = total →
  brown_eyed - ((total - brunettes) - blue_eyed_blondes) = 17 := by
  sorry

#check brown_eyed_brunettes

end NUMINAMATH_CALUDE_brown_eyed_brunettes_l2453_245315


namespace NUMINAMATH_CALUDE_range_of_a_range_of_x_min_value_ratio_l2453_245375

-- Define the quadratic function
def quadratic (a b x : ℝ) := a * x^2 + b * x + 2

-- Part 1
theorem range_of_a (a b : ℝ) :
  (∀ x, quadratic a b x > 0) →
  quadratic a b 1 = 1 →
  a ∈ Set.Ioo (3 - 2 * Real.sqrt 2) (3 + 2 * Real.sqrt 2) :=
sorry

-- Part 2
theorem range_of_x (a b : ℝ) :
  (∀ a ∈ Set.Icc (-2) (-1), ∀ x, quadratic a b x > 0) →
  quadratic a b 1 = 1 →
  ∃ x ∈ Set.Ioo ((1 - Real.sqrt 17) / 4) ((1 + Real.sqrt 17) / 4),
    quadratic a b x = 0 :=
sorry

-- Part 3
theorem min_value_ratio (a b : ℝ) :
  (∀ x, quadratic a b x ≥ 0) →
  b > 0 →
  (a + 2) / b ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_x_min_value_ratio_l2453_245375


namespace NUMINAMATH_CALUDE_simplified_fraction_l2453_245381

theorem simplified_fraction (a : ℤ) (ha : a > 0) :
  let expr := (a + 1) / a - a / (a + 1)
  let simplified := (2 * a + 1) / (a * (a + 1))
  expr = simplified ∧ (a = 2023 → 2 * a + 1 = 4047) := by
  sorry

#eval 2 * 2023 + 1

end NUMINAMATH_CALUDE_simplified_fraction_l2453_245381


namespace NUMINAMATH_CALUDE_sticker_pages_l2453_245382

theorem sticker_pages (stickers_per_page : ℕ) (remaining_stickers : ℕ) : 
  (stickers_per_page = 20 ∧ remaining_stickers = 220) → 
  ∃ (initial_pages : ℕ), 
    initial_pages * stickers_per_page - stickers_per_page = remaining_stickers ∧ 
    initial_pages = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sticker_pages_l2453_245382


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2453_245395

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n.gcd d = 1) ∧ 
  (n : ℚ) / (d : ℚ) = 3 + 17 / 99 ∧
  n + d = 413 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2453_245395


namespace NUMINAMATH_CALUDE_number_in_bases_is_61_l2453_245310

/-- Represents a number in different bases -/
def NumberInBases (n : ℕ) : Prop :=
  ∃ (a b : ℕ),
    (0 ≤ a ∧ a < 6) ∧
    (0 ≤ b ∧ b < 6) ∧
    n = 36 * a + 6 * b + a ∧
    n = 15 * b + a

theorem number_in_bases_is_61 :
  ∃ (n : ℕ), NumberInBases n ∧ n = 61 :=
sorry

end NUMINAMATH_CALUDE_number_in_bases_is_61_l2453_245310


namespace NUMINAMATH_CALUDE_five_variable_inequality_two_is_smallest_constant_l2453_245345

theorem five_variable_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) > 2 :=
by sorry

theorem two_is_smallest_constant :
  ∀ ε > 0, ∃ a b c d e : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) + 
    Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) + 
    Real.sqrt (e / (a + b + c + d)) < 2 + ε :=
by sorry

end NUMINAMATH_CALUDE_five_variable_inequality_two_is_smallest_constant_l2453_245345


namespace NUMINAMATH_CALUDE_shelf_capacity_l2453_245316

/-- The total capacity of jars on a shelf. -/
def total_capacity (total_jars small_jars : ℕ) (small_capacity large_capacity : ℕ) : ℕ :=
  small_jars * small_capacity + (total_jars - small_jars) * large_capacity

/-- Theorem stating the total capacity of jars on the shelf. -/
theorem shelf_capacity : total_capacity 100 62 3 5 = 376 := by
  sorry

end NUMINAMATH_CALUDE_shelf_capacity_l2453_245316


namespace NUMINAMATH_CALUDE_plates_needed_is_38_l2453_245321

/-- The number of plates needed for a week given the eating habits of Matt's family -/
def plates_needed : ℕ :=
  let days_with_son := 3
  let days_with_parents := 7 - days_with_son
  let plates_per_person_with_son := 1
  let plates_per_person_with_parents := 2
  let people_with_son := 2
  let people_with_parents := 4
  
  (days_with_son * people_with_son * plates_per_person_with_son) +
  (days_with_parents * people_with_parents * plates_per_person_with_parents)

theorem plates_needed_is_38 : plates_needed = 38 := by
  sorry

end NUMINAMATH_CALUDE_plates_needed_is_38_l2453_245321


namespace NUMINAMATH_CALUDE_set_intersection_range_l2453_245337

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}
def B : Set ℝ := {x | x^2 + 4*x = 0}

-- Define the theorem
theorem set_intersection_range (a : ℝ) :
  A a ∩ B = A a → (a ≤ -1 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_set_intersection_range_l2453_245337


namespace NUMINAMATH_CALUDE_average_price_rahim_l2453_245372

/-- Represents a book purchase from a shop -/
structure BookPurchase where
  quantity : ℕ
  totalPrice : ℕ

/-- Calculates the average price per book given a list of book purchases -/
def averagePrice (purchases : List BookPurchase) : ℚ :=
  let totalBooks := purchases.map (fun p => p.quantity) |>.sum
  let totalCost := purchases.map (fun p => p.totalPrice) |>.sum
  (totalCost : ℚ) / (totalBooks : ℚ)

theorem average_price_rahim (purchases : List BookPurchase) 
  (h1 : purchases = [
    ⟨40, 600⟩,  -- Shop A
    ⟨20, 240⟩,  -- Shop B
    ⟨15, 180⟩,  -- Shop C
    ⟨25, 325⟩   -- Shop D
  ]) : 
  averagePrice purchases = 1345 / 100 := by
  sorry

#eval (1345 : ℚ) / 100  -- To verify the result is indeed 13.45

end NUMINAMATH_CALUDE_average_price_rahim_l2453_245372


namespace NUMINAMATH_CALUDE_base8_subtraction_to_base4_l2453_245370

/-- Converts a number from base 8 to base 10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 --/
def base10ToBase4 (n : ℕ) : List ℕ := sorry

/-- Subtracts two numbers in base 8 --/
def subtractBase8 (a b : ℕ) : ℕ := sorry

theorem base8_subtraction_to_base4 :
  let a := 643
  let b := 257
  let result := subtractBase8 a b
  base10ToBase4 (base8ToBase10 result) = [3, 3, 1, 1, 0] := by sorry

end NUMINAMATH_CALUDE_base8_subtraction_to_base4_l2453_245370


namespace NUMINAMATH_CALUDE_square_area_ratio_l2453_245369

theorem square_area_ratio (y : ℝ) (hy : y > 0) : 
  (y^2) / ((3*y)^2) = 1/9 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2453_245369


namespace NUMINAMATH_CALUDE_exists_square_between_consecutive_prime_sums_l2453_245319

-- Define S_n as the sum of the first n prime numbers
def S (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem exists_square_between_consecutive_prime_sums : 
  ∃ k : ℕ, S 2023 < k^2 ∧ k^2 < S 2024 := by sorry

end NUMINAMATH_CALUDE_exists_square_between_consecutive_prime_sums_l2453_245319


namespace NUMINAMATH_CALUDE_rectangle_area_stage_7_l2453_245309

/-- The side length of each square in inches -/
def square_side : ℝ := 4

/-- The number of squares at Stage 7 -/
def num_squares : ℕ := 7

/-- The area of the rectangle at Stage 7 in square inches -/
def rectangle_area : ℝ := (square_side ^ 2) * num_squares

/-- Theorem: The area of the rectangle at Stage 7 is 112 square inches -/
theorem rectangle_area_stage_7 : rectangle_area = 112 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_stage_7_l2453_245309


namespace NUMINAMATH_CALUDE_unique_m_satisfying_lcm_conditions_l2453_245339

theorem unique_m_satisfying_lcm_conditions (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 60 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_satisfying_lcm_conditions_l2453_245339


namespace NUMINAMATH_CALUDE_combined_height_problem_l2453_245343

/-- Given that Mr. Martinez is two feet taller than Chiquita and Chiquita is 5 feet tall,
    prove that their combined height is 12 feet. -/
theorem combined_height_problem (chiquita_height : ℝ) (martinez_height : ℝ) :
  chiquita_height = 5 →
  martinez_height = chiquita_height + 2 →
  chiquita_height + martinez_height = 12 :=
by sorry

end NUMINAMATH_CALUDE_combined_height_problem_l2453_245343


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2453_245389

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, Real.exp x > Real.log x)) ↔ (∃ x₀ : ℝ, Real.exp x₀ ≤ Real.log x₀) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2453_245389


namespace NUMINAMATH_CALUDE_probability_of_colored_ball_l2453_245352

def urn_total : ℕ := 30
def red_balls : ℕ := 10
def blue_balls : ℕ := 5
def white_balls : ℕ := 15

theorem probability_of_colored_ball :
  (red_balls + blue_balls : ℚ) / urn_total = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_probability_of_colored_ball_l2453_245352


namespace NUMINAMATH_CALUDE_expression_evaluation_l2453_245347

theorem expression_evaluation :
  (1 / ((5^2)^4)) * 5^15 = 5^7 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2453_245347


namespace NUMINAMATH_CALUDE_complex_conversion_l2453_245351

theorem complex_conversion (z : ℂ) : z = Complex.exp (13 * Real.pi * Complex.I / 4) * (Real.sqrt 3) →
  z = Complex.mk (Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_conversion_l2453_245351


namespace NUMINAMATH_CALUDE_cafeteria_seats_available_l2453_245392

theorem cafeteria_seats_available 
  (total_tables : ℕ) 
  (seats_per_table : ℕ) 
  (people_dining : ℕ) : 
  total_tables = 40 → 
  seats_per_table = 12 → 
  people_dining = 325 → 
  total_tables * seats_per_table - people_dining = 155 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_seats_available_l2453_245392


namespace NUMINAMATH_CALUDE_min_team_size_proof_l2453_245303

def P₁ : ℝ := 0.3

def individual_prob : ℝ := 0.1

def P₂ (n : ℕ) : ℝ := 1 - (1 - individual_prob) ^ n

def min_team_size : ℕ := 4

theorem min_team_size_proof :
  ∀ n : ℕ, (P₂ n ≥ P₁) → n ≥ min_team_size :=
sorry

end NUMINAMATH_CALUDE_min_team_size_proof_l2453_245303


namespace NUMINAMATH_CALUDE_weighted_am_gm_inequality_l2453_245362

theorem weighted_am_gm_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  0.2 * a + 0.3 * b + 0.5 * c ≥ (a * b * c) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_weighted_am_gm_inequality_l2453_245362


namespace NUMINAMATH_CALUDE_certain_number_proof_l2453_245324

theorem certain_number_proof (n : ℝ) (h : 7125 / n = 5700) : n = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2453_245324


namespace NUMINAMATH_CALUDE_dot_product_range_l2453_245342

/-- The ellipse equation -/
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 16) + (P.2^2 / 15) = 1

/-- The circle equation -/
def is_on_circle (Q : ℝ × ℝ) : Prop :=
  (Q.1 - 1)^2 + Q.2^2 = 4

/-- Definition of a diameter of the circle -/
def is_diameter (E F : ℝ × ℝ) : Prop :=
  is_on_circle E ∧ is_on_circle F ∧ 
  (E.1 + F.1 = 2) ∧ (E.2 + F.2 = 0)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The main theorem -/
theorem dot_product_range (P E F : ℝ × ℝ) :
  is_on_ellipse P → is_diameter E F →
  5 ≤ dot_product (E.1 - P.1, E.2 - P.2) (F.1 - P.1, F.2 - P.2) ∧
  dot_product (E.1 - P.1, E.2 - P.2) (F.1 - P.1, F.2 - P.2) ≤ 21 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_range_l2453_245342


namespace NUMINAMATH_CALUDE_half_volume_convex_hull_cube_l2453_245306

theorem half_volume_convex_hull_cube : ∃ a : ℝ, 0 < a ∧ a < 1 ∧ 
  2 * (a^3 + (1-a)^3) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_half_volume_convex_hull_cube_l2453_245306


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_and_fraction_l2453_245330

/-- Represents a repeating decimal with a single digit repeating infinitely -/
def repeating_decimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + (repeating : ℚ) / 9

theorem sum_of_repeating_decimals_and_fraction :
  repeating_decimal 0 6 - repeating_decimal 0 2 + (1 : ℚ) / 4 = 25 / 36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_and_fraction_l2453_245330


namespace NUMINAMATH_CALUDE_function_composition_l2453_245359

theorem function_composition (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = x^2 + 2*x) →
  f (2*x + 1) = 4*x^2 + 8*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l2453_245359


namespace NUMINAMATH_CALUDE_class_size_problem_l2453_245333

theorem class_size_problem (total : ℕ) (sum_fraction : ℕ) 
  (h1 : total = 85) 
  (h2 : sum_fraction = 42) : ∃ (a b : ℕ), 
  a + b = total ∧ 
  (3 * a) / 8 + (3 * b) / 5 = sum_fraction ∧ 
  a = 40 ∧ 
  b = 45 := by
  sorry

end NUMINAMATH_CALUDE_class_size_problem_l2453_245333


namespace NUMINAMATH_CALUDE_range_of_a_l2453_245328

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0 → x^2 + 5*x + 4 ≤ 0) →
  a < 0 →
  -4/3 ≤ a ∧ a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2453_245328


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l2453_245340

theorem largest_constant_inequality (x y : ℝ) :
  (∃ (C : ℝ), ∀ (x y : ℝ), x^2 + y^2 + 1 ≥ C * (x + y)) ∧
  (∀ (D : ℝ), (∀ (x y : ℝ), x^2 + y^2 + 1 ≥ D * (x + y)) → D ≤ Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l2453_245340


namespace NUMINAMATH_CALUDE_zoo_zebra_count_l2453_245363

theorem zoo_zebra_count :
  ∀ (penguins zebras tigers zookeepers : ℕ),
    penguins = 30 →
    tigers = 8 →
    zookeepers = 12 →
    (penguins + zebras + tigers + zookeepers) = 
      (2 * penguins + 4 * zebras + 4 * tigers + 2 * zookeepers) - 132 →
    zebras = 22 := by
  sorry

end NUMINAMATH_CALUDE_zoo_zebra_count_l2453_245363


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l2453_245368

/-- A regular decagon is a 10-sided polygon with all sides and angles equal -/
structure RegularDecagon where
  sides : Nat
  sides_eq : sides = 10

/-- The probability that two randomly chosen diagonals of a regular decagon intersect inside the decagon -/
def diagonal_intersection_probability (d : RegularDecagon) : ℚ :=
  42 / 119

/-- Theorem stating that the probability of two randomly chosen diagonals 
    of a regular decagon intersecting inside the decagon is 42/119 -/
theorem decagon_diagonal_intersection_probability (d : RegularDecagon) :
  diagonal_intersection_probability d = 42 / 119 := by
  sorry

#check decagon_diagonal_intersection_probability

end NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l2453_245368


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2453_245323

-- Define sets M and N
def M : Set ℝ := {x | x < 5}
def N : Set ℝ := {x | x > 3}

-- Theorem statement
theorem condition_necessary_not_sufficient :
  (∀ x, x ∈ (M ∩ N) → x ∈ (M ∪ N)) ∧
  (∃ x, x ∈ (M ∪ N) ∧ x ∉ (M ∩ N)) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2453_245323


namespace NUMINAMATH_CALUDE_average_difference_with_data_error_l2453_245396

theorem average_difference_with_data_error (data : List ℝ) (wrong_value correct_value : ℝ) : 
  data.length = 30 →
  wrong_value = 15 →
  correct_value = 105 →
  wrong_value ∈ data →
  (data.sum / data.length) - ((data.sum - wrong_value + correct_value) / data.length) = -3 := by
sorry

end NUMINAMATH_CALUDE_average_difference_with_data_error_l2453_245396


namespace NUMINAMATH_CALUDE_platform_length_l2453_245373

/-- Calculates the length of a platform given train parameters --/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) : 
  train_length = 175 →
  train_speed_kmph = 36 →
  crossing_time = 40 →
  (train_speed_kmph * 1000 / 3600 * crossing_time) - train_length = 225 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l2453_245373


namespace NUMINAMATH_CALUDE_floor_plus_self_equal_five_l2453_245331

theorem floor_plus_self_equal_five (y : ℝ) : ⌊y⌋ + y = 5 → y = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_equal_five_l2453_245331


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_sufficient_but_not_necessary_l2453_245334

/-- Two lines are parallel if their slopes are equal -/
def parallel (m : ℝ) : Prop := 2 / m = (m - 1) / 1

/-- Sufficient condition: m = 2 implies the lines are parallel -/
theorem sufficient_condition : parallel 2 := by sorry

/-- Not necessary: there exists m ≠ 2 such that the lines are parallel -/
theorem not_necessary : ∃ m : ℝ, m ≠ 2 ∧ parallel m := by sorry

/-- m = 2 is a sufficient but not necessary condition for the lines to be parallel -/
theorem sufficient_but_not_necessary : 
  (parallel 2) ∧ (∃ m : ℝ, m ≠ 2 ∧ parallel m) := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_sufficient_but_not_necessary_l2453_245334


namespace NUMINAMATH_CALUDE_specific_trapezoid_dimensions_l2453_245378

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  -- The area of the trapezoid
  area : ℝ
  -- The height of the trapezoid
  height : ℝ
  -- The length of one parallel side (shorter)
  base_short : ℝ
  -- The length of the other parallel side (longer)
  base_long : ℝ
  -- The length of the non-parallel sides (legs)
  leg : ℝ
  -- The trapezoid is isosceles
  isosceles : True
  -- The lines containing the legs intersect at a right angle
  right_angle_intersection : True
  -- The area is calculated correctly
  area_eq : area = (base_short + base_long) * height / 2

/-- Theorem about a specific isosceles trapezoid -/
theorem specific_trapezoid_dimensions :
  ∃ t : IsoscelesTrapezoid,
    t.area = 12 ∧
    t.height = 2 ∧
    t.base_short = 4 ∧
    t.base_long = 8 ∧
    t.leg = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_dimensions_l2453_245378


namespace NUMINAMATH_CALUDE_delicious_delhi_bill_l2453_245341

/-- Calculates the total bill for a meal at Delicious Delhi restaurant --/
def calculate_bill (
  samosa_price : ℚ)
  (pakora_price : ℚ)
  (lassi_price : ℚ)
  (biryani_price : ℚ)
  (naan_price : ℚ)
  (samosa_quantity : ℕ)
  (pakora_quantity : ℕ)
  (lassi_quantity : ℕ)
  (biryani_quantity : ℕ)
  (naan_quantity : ℕ)
  (biryani_discount_rate : ℚ)
  (service_fee_rate : ℚ)
  (tip_rate : ℚ)
  (tax_rate : ℚ) : ℚ :=
  sorry

theorem delicious_delhi_bill :
  calculate_bill 2 3 2 (11/2) (3/2) 3 4 1 2 1 (1/10) (3/100) (1/5) (2/25) = 4125/100 :=
sorry

end NUMINAMATH_CALUDE_delicious_delhi_bill_l2453_245341


namespace NUMINAMATH_CALUDE_hot_dog_problem_l2453_245361

theorem hot_dog_problem (cost_per_hot_dog : ℕ) (total_paid : ℕ) (h1 : cost_per_hot_dog = 50) (h2 : total_paid = 300) :
  total_paid / cost_per_hot_dog = 6 := by
sorry

end NUMINAMATH_CALUDE_hot_dog_problem_l2453_245361


namespace NUMINAMATH_CALUDE_exists_valid_formula_l2453_245354

def uses_five_twos (formula : ℕ → ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    (a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 ∧ e = 2) ∧
    ∀ n, formula n = f a b c d e
  where f := λ a b c d e => sorry -- placeholder for the actual formula

def is_valid_formula (formula : ℕ → ℕ) : Prop :=
  uses_five_twos formula ∧
  (∀ n, n ∈ Finset.range 10 → formula (n + 11) = n + 11)

theorem exists_valid_formula : ∃ formula, is_valid_formula formula := by
  sorry

#check exists_valid_formula

end NUMINAMATH_CALUDE_exists_valid_formula_l2453_245354


namespace NUMINAMATH_CALUDE_largest_share_in_startup_l2453_245366

def profit_split (total_profit : ℚ) (ratios : List ℚ) : List ℚ :=
  let sum_ratios := ratios.sum
  ratios.map (λ r => (r / sum_ratios) * total_profit)

theorem largest_share_in_startup (total_profit : ℚ) :
  let ratios : List ℚ := [3, 4, 4, 6, 7]
  let shares := profit_split total_profit ratios
  total_profit = 48000 →
  shares.maximum = some 14000 := by
sorry

end NUMINAMATH_CALUDE_largest_share_in_startup_l2453_245366


namespace NUMINAMATH_CALUDE_yellow_or_blue_consecutive_rolls_l2453_245302

/-- A die with 12 sides and specific color distribution -/
structure Die :=
  (sides : Nat)
  (red : Nat)
  (yellow : Nat)
  (blue : Nat)
  (green : Nat)
  (total_eq : sides = red + yellow + blue + green)

/-- The probability of an event occurring -/
def probability (favorable : Nat) (total : Nat) : ℚ :=
  ↑favorable / ↑total

/-- The probability of two independent events both occurring -/
def probability_both (p1 : ℚ) (p2 : ℚ) : ℚ := p1 * p2

theorem yellow_or_blue_consecutive_rolls (d : Die) 
  (h : d.sides = 12 ∧ d.red = 5 ∧ d.yellow = 4 ∧ d.blue = 2 ∧ d.green = 1) : 
  probability_both 
    (probability (d.yellow + d.blue) d.sides) 
    (probability (d.yellow + d.blue) d.sides) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_yellow_or_blue_consecutive_rolls_l2453_245302


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_6_18_24_l2453_245399

def gcd3 (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

def lcm3 (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem gcd_lcm_sum_6_18_24 : 
  gcd3 6 18 24 + lcm3 6 18 24 = 78 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_6_18_24_l2453_245399


namespace NUMINAMATH_CALUDE_min_slope_and_sum_reciprocals_l2453_245308

noncomputable section

def f (x : ℝ) := x^3 - x^2 + (2 * Real.sqrt 2 - 3) * x + 3 - 2 * Real.sqrt 2

def f' (x : ℝ) := 3 * x^2 - 2 * x + 2 * Real.sqrt 2 - 3

theorem min_slope_and_sum_reciprocals :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f' x_min ≤ f' x ∧ f' x_min = 2 * Real.sqrt 2 - 10 / 3) ∧
  (∃ (x₁ x₂ x₃ : ℝ), f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ 
    1 / f' x₁ + 1 / f' x₂ + 1 / f' x₃ = 0) := by
  sorry

end

end NUMINAMATH_CALUDE_min_slope_and_sum_reciprocals_l2453_245308


namespace NUMINAMATH_CALUDE_triangle_angle_and_area_l2453_245358

/-- Given a triangle ABC with angle A and vectors m and n, prove the measure of A and the area of the triangle -/
theorem triangle_angle_and_area 
  (A B C : ℝ) 
  (m n : ℝ × ℝ) 
  (h1 : m = (Real.sin (A/2), Real.cos (A/2)))
  (h2 : n = (Real.cos (A/2), -Real.cos (A/2)))
  (h3 : 2 * (m.1 * n.1 + m.2 * n.2) + Real.sqrt (m.1^2 + m.2^2) = Real.sqrt 2 / 2)
  (h4 : Real.cos A = 1 / (Real.sin A)) :
  A = 5 * Real.pi / 12 ∧ 
  (Real.sin A) / 2 = (2 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_and_area_l2453_245358


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2453_245353

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2453_245353


namespace NUMINAMATH_CALUDE_consecutive_integers_median_l2453_245332

theorem consecutive_integers_median (n : ℕ) (sum : ℕ) (median : ℕ) : 
  n = 64 → sum = 4096 → sum = n * median → median = 64 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_median_l2453_245332


namespace NUMINAMATH_CALUDE_belle_treat_cost_l2453_245390

/-- The cost of feeding Belle treats for a week -/
def weekly_cost : ℚ := 21

/-- The number of dog biscuits Belle eats daily -/
def daily_biscuits : ℕ := 4

/-- The number of rawhide bones Belle eats daily -/
def daily_bones : ℕ := 2

/-- The cost of each rawhide bone in dollars -/
def bone_cost : ℚ := 1

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The cost of each dog biscuit in dollars -/
def biscuit_cost : ℚ := 1/4

theorem belle_treat_cost : 
  weekly_cost = days_in_week * (daily_biscuits * biscuit_cost + daily_bones * bone_cost) :=
by sorry

end NUMINAMATH_CALUDE_belle_treat_cost_l2453_245390


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2453_245357

theorem necessary_but_not_sufficient (a : ℝ) :
  (a^2 < 2*a → a < 2) ∧ ¬(∀ a, a < 2 → a^2 < 2*a) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2453_245357


namespace NUMINAMATH_CALUDE_student_average_problem_l2453_245322

theorem student_average_problem :
  let total_students : ℕ := 25
  let group_a_students : ℕ := 15
  let group_b_students : ℕ := 10
  let group_b_average : ℚ := 90
  let total_average : ℚ := 84
  let group_a_average : ℚ := (total_students * total_average - group_b_students * group_b_average) / group_a_students
  group_a_average = 80 := by sorry

end NUMINAMATH_CALUDE_student_average_problem_l2453_245322


namespace NUMINAMATH_CALUDE_polynomial_identity_solutions_l2453_245355

variable (x : ℝ)

noncomputable def p (x : ℝ) : ℝ := x^2 + x + 1

theorem polynomial_identity_solutions :
  ∃! (q₁ q₂ : ℝ → ℝ), 
    (∀ x, q₁ x = x^2 + 2*x) ∧ 
    (∀ x, q₂ x = x^2 - 1) ∧ 
    (∀ q : ℝ → ℝ, (∀ x, (p x)^2 - 2*(p x)*(q x) + (q x)^2 - 4*(p x) + 3*(q x) + 3 = 0) → 
      (q = q₁ ∨ q = q₂)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_identity_solutions_l2453_245355


namespace NUMINAMATH_CALUDE_parallel_tangents_theorem_l2453_245398

-- Define the curve C
def C (a b d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + d

-- Define the derivative of C
def C_derivative (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem parallel_tangents_theorem (a b d : ℝ) :
  C a b d 1 = 1 →  -- Point A(1,1) is on the curve
  C a b d (-1) = -3 →  -- Point B(-1,-3) is on the curve
  C_derivative a b 1 = C_derivative a b (-1) →  -- Tangents at A and B are parallel
  a^3 + b^2 + d = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangents_theorem_l2453_245398


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2453_245384

def A : Set ℤ := {x | x^2 ≤ 16}
def B : Set ℤ := {x | -1 ≤ x ∧ x < 4}

theorem complement_intersection_theorem : 
  (A \ (A ∩ B)) = {-4, -3, -2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2453_245384


namespace NUMINAMATH_CALUDE_winning_strategy_extends_l2453_245318

/-- Represents the winning player for a given game state -/
inductive Winner : Type
  | Player1 : Winner
  | Player2 : Winner

/-- Represents the game state -/
structure GameState :=
  (t : ℕ)  -- Current number on the blackboard
  (a : ℕ)  -- First subtraction option
  (b : ℕ)  -- Second subtraction option

/-- Determines the winner of the game given a game state -/
def winningPlayer (state : GameState) : Winner :=
  sorry

/-- Theorem stating that if Player 1 wins for x, they also win for x + 2005k -/
theorem winning_strategy_extends (x k a b : ℕ) :
  (1 ≤ x) →
  (x ≤ 2005) →
  (0 < a) →
  (0 < b) →
  (a + b = 2005) →
  (winningPlayer { t := x, a := a, b := b } = Winner.Player1) →
  (winningPlayer { t := x + 2005 * k, a := a, b := b } = Winner.Player1) :=
by
  sorry

end NUMINAMATH_CALUDE_winning_strategy_extends_l2453_245318


namespace NUMINAMATH_CALUDE_sum_85_to_100_l2453_245383

def sum_consecutive_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem sum_85_to_100 :
  sum_consecutive_integers 85 100 = 1480 :=
by sorry

end NUMINAMATH_CALUDE_sum_85_to_100_l2453_245383


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_empty_l2453_245385

def M : Set ℝ := {x | |x - 1| < 1}
def N : Set ℝ := {x | x^2 - 2*x < 3}

theorem intersection_M_complement_N_empty :
  M ∩ (Set.univ \ N) = ∅ := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_empty_l2453_245385


namespace NUMINAMATH_CALUDE_passengers_in_first_class_l2453_245307

theorem passengers_in_first_class (total_passengers : ℕ) 
  (women_percentage : ℚ) (men_percentage : ℚ)
  (women_first_class_percentage : ℚ) (men_first_class_percentage : ℚ)
  (h1 : total_passengers = 300)
  (h2 : women_percentage = 1/2)
  (h3 : men_percentage = 1/2)
  (h4 : women_first_class_percentage = 1/5)
  (h5 : men_first_class_percentage = 3/20) :
  ⌈(total_passengers : ℚ) * women_percentage * women_first_class_percentage + 
   (total_passengers : ℚ) * men_percentage * men_first_class_percentage⌉ = 53 :=
by sorry

end NUMINAMATH_CALUDE_passengers_in_first_class_l2453_245307


namespace NUMINAMATH_CALUDE_b_score_is_93_l2453_245304

/-- Represents the scores of five people in an exam -/
structure ExamScores where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ

/-- The average score of all five people is 90 -/
def average_all (scores : ExamScores) : Prop :=
  (scores.A + scores.B + scores.C + scores.D + scores.E) / 5 = 90

/-- The average score of A, B, and C is 86 -/
def average_ABC (scores : ExamScores) : Prop :=
  (scores.A + scores.B + scores.C) / 3 = 86

/-- The average score of B, D, and E is 95 -/
def average_BDE (scores : ExamScores) : Prop :=
  (scores.B + scores.D + scores.E) / 3 = 95

/-- Theorem: Given the conditions, B's score is 93 -/
theorem b_score_is_93 (scores : ExamScores) 
  (h1 : average_all scores) 
  (h2 : average_ABC scores) 
  (h3 : average_BDE scores) : 
  scores.B = 93 := by
  sorry

end NUMINAMATH_CALUDE_b_score_is_93_l2453_245304


namespace NUMINAMATH_CALUDE_paths_through_F_and_H_l2453_245391

/-- Represents a point on the grid -/
structure Point where
  x : Nat
  y : Nat

/-- Calculates the number of paths between two points on a grid -/
def numPaths (start finish : Point) : Nat :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The grid dimensions -/
def gridWidth : Nat := 7
def gridHeight : Nat := 6

/-- The points on the grid -/
def E : Point := ⟨0, 5⟩
def F : Point := ⟨4, 4⟩
def H : Point := ⟨5, 2⟩
def G : Point := ⟨6, 0⟩

/-- Theorem: The number of 12-step paths from E to G passing through F and then H is 135 -/
theorem paths_through_F_and_H : 
  numPaths E F * numPaths F H * numPaths H G = 135 := by
  sorry

end NUMINAMATH_CALUDE_paths_through_F_and_H_l2453_245391


namespace NUMINAMATH_CALUDE_order_of_ab_squared_a_ab_l2453_245338

theorem order_of_ab_squared_a_ab (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) : 
  a < a * b^2 ∧ a * b^2 < a * b := by sorry

end NUMINAMATH_CALUDE_order_of_ab_squared_a_ab_l2453_245338


namespace NUMINAMATH_CALUDE_magnitude_relationship_l2453_245313

theorem magnitude_relationship
  (a b c d : ℝ)
  (h_order : a > b ∧ b > c ∧ c > d)
  (h_positive : d > 0)
  (x : ℝ)
  (h_x : x = Real.sqrt (a * b) + Real.sqrt (c * d))
  (y : ℝ)
  (h_y : y = Real.sqrt (a * c) + Real.sqrt (b * d))
  (z : ℝ)
  (h_z : z = Real.sqrt (a * d) + Real.sqrt (b * c)) :
  x > y ∧ y > z :=
by sorry

end NUMINAMATH_CALUDE_magnitude_relationship_l2453_245313


namespace NUMINAMATH_CALUDE_fenced_field_area_l2453_245377

/-- A rectangular field with specific fencing requirements -/
structure FencedField where
  length : ℝ
  width : ℝ
  uncovered_side : ℝ
  fencing : ℝ
  uncovered_side_eq : uncovered_side = 20
  fencing_eq : uncovered_side + 2 * width = fencing
  fencing_length : fencing = 88

/-- The area of a rectangular field -/
def field_area (f : FencedField) : ℝ :=
  f.length * f.width

/-- Theorem stating that a field with the given specifications has an area of 680 square feet -/
theorem fenced_field_area (f : FencedField) : field_area f = 680 := by
  sorry

end NUMINAMATH_CALUDE_fenced_field_area_l2453_245377


namespace NUMINAMATH_CALUDE_ant_path_distance_l2453_245346

def ant_movements : List (ℝ × ℝ) := [
  (-7, 0),  -- 7 cm left
  (0, 5),   -- 5 cm up
  (3, 0),   -- 3 cm right
  (0, -2),  -- 2 cm down
  (9, 0),   -- 9 cm right
  (0, -2),  -- 2 cm down
  (-1, 0),  -- 1 cm left
  (0, -1)   -- 1 cm down
]

theorem ant_path_distance : 
  let total_movement := ant_movements.foldl (fun acc (x, y) => (acc.1 + x, acc.2 + y)) (0, 0)
  Real.sqrt (total_movement.1^2 + total_movement.2^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ant_path_distance_l2453_245346


namespace NUMINAMATH_CALUDE_translation_of_point_l2453_245364

/-- Given a point A with coordinates (-2, 3) in a Cartesian coordinate system,
    prove that translating it 3 units right and 5 units down results in
    point B with coordinates (1, -2). -/
theorem translation_of_point (A B : ℝ × ℝ) :
  A = (-2, 3) →
  B.1 = A.1 + 3 →
  B.2 = A.2 - 5 →
  B = (1, -2) := by
  sorry

end NUMINAMATH_CALUDE_translation_of_point_l2453_245364


namespace NUMINAMATH_CALUDE_angle_P_measure_l2453_245325

/-- A quadrilateral with specific angle relationships -/
structure SpecialQuadrilateral where
  P : ℝ  -- Angle P in degrees
  Q : ℝ  -- Angle Q in degrees
  R : ℝ  -- Angle R in degrees
  S : ℝ  -- Angle S in degrees
  angle_relation : P = 3*Q ∧ P = 4*R ∧ P = 6*S
  sum_360 : P + Q + R + S = 360

/-- The measure of angle P in a SpecialQuadrilateral is 206 degrees -/
theorem angle_P_measure (quad : SpecialQuadrilateral) : 
  ⌊quad.P⌋ = 206 := by sorry

end NUMINAMATH_CALUDE_angle_P_measure_l2453_245325


namespace NUMINAMATH_CALUDE_f_intersects_twice_l2453_245350

/-- An even function that is monotonically increasing for positive x and satisfies f(1) * f(2) < 0 -/
def f : ℝ → ℝ :=
  sorry

/-- f is an even function -/
axiom f_even : ∀ x, f (-x) = f x

/-- f is monotonically increasing for positive x -/
axiom f_increasing : ∀ x y, 0 < x → x < y → f x < f y

/-- f(1) * f(2) < 0 -/
axiom f_sign_change : f 1 * f 2 < 0

/-- The number of intersection points between f and the x-axis -/
def num_intersections : ℕ :=
  sorry

/-- Theorem: The number of intersection points between f and the x-axis is 2 -/
theorem f_intersects_twice : num_intersections = 2 :=
  sorry

end NUMINAMATH_CALUDE_f_intersects_twice_l2453_245350


namespace NUMINAMATH_CALUDE_difference_of_valid_pair_l2453_245329

def is_valid_pair (a b : ℕ) : Prop :=
  a + b = 21308 ∧ 
  a % 10 = 5 ∧
  b = (a / 10) * 10 + 50

theorem difference_of_valid_pair (a b : ℕ) (h : is_valid_pair a b) : 
  (max a b) - (min a b) = 17344 :=
sorry

end NUMINAMATH_CALUDE_difference_of_valid_pair_l2453_245329


namespace NUMINAMATH_CALUDE_exist_six_lines_equal_angles_l2453_245397

/-- A line in 3D space represented by a point and a direction vector -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Angle between two lines in 3D space -/
def angle (l1 l2 : Line3D) : ℝ := sorry

/-- A set of 6 lines in 3D space -/
def SixLines : Type := Fin 6 → Line3D

/-- Predicate to check if all pairs of lines are non-parallel -/
def all_non_parallel (lines : SixLines) : Prop :=
  ∀ i j, i ≠ j → lines i ≠ lines j

/-- Predicate to check if all pairwise angles are equal -/
def all_angles_equal (lines : SixLines) : Prop :=
  ∀ i j k l, i ≠ j → k ≠ l → angle (lines i) (lines j) = angle (lines k) (lines l)

/-- Theorem stating the existence of 6 lines satisfying the conditions -/
theorem exist_six_lines_equal_angles : 
  ∃ (lines : SixLines), all_non_parallel lines ∧ all_angles_equal lines :=
sorry

end NUMINAMATH_CALUDE_exist_six_lines_equal_angles_l2453_245397


namespace NUMINAMATH_CALUDE_sqrt_six_equals_r_squared_minus_five_over_two_l2453_245374

theorem sqrt_six_equals_r_squared_minus_five_over_two :
  Real.sqrt 6 = ((Real.sqrt 2 + Real.sqrt 3)^2 - 5) / 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_six_equals_r_squared_minus_five_over_two_l2453_245374


namespace NUMINAMATH_CALUDE_replaced_student_weight_l2453_245393

theorem replaced_student_weight 
  (n : ℕ) 
  (new_weight : ℝ) 
  (avg_decrease : ℝ) : 
  n = 8 → 
  new_weight = 46 → 
  avg_decrease = 5 → 
  ∃ (old_weight : ℝ), old_weight = n * avg_decrease + new_weight :=
by
  sorry

end NUMINAMATH_CALUDE_replaced_student_weight_l2453_245393


namespace NUMINAMATH_CALUDE_ellipse_equation_l2453_245387

/-- An ellipse with center at the origin, foci on the x-axis, and point P(2, √3) on the ellipse. 
    The distances |PF₁|, |F₁F₂|, and |PF₂| form an arithmetic sequence. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < b ∧ b < a
  h_foci : c^2 = a^2 - b^2
  h_point_on_ellipse : 4 / a^2 + 3 / b^2 = 1
  h_arithmetic_sequence : ∃ (d : ℝ), |2 - c| + d = 2*c ∧ 2*c + d = |2 + c|

/-- The equation of the ellipse is x²/8 + y²/6 = 1 -/
theorem ellipse_equation (e : Ellipse) : e.a^2 = 8 ∧ e.b^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2453_245387


namespace NUMINAMATH_CALUDE_angle_D_measure_l2453_245356

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  /-- Measure of angle E in degrees -/
  angle_E : ℝ
  /-- The triangle is isosceles with angle D congruent to angle F -/
  isosceles : True
  /-- The measure of angle F is three times the measure of angle E -/
  angle_F_eq_three_E : True

/-- Theorem: In the given isosceles triangle, the measure of angle D is 77 1/7 degrees -/
theorem angle_D_measure (t : IsoscelesTriangle) : 
  (3 * t.angle_E : ℝ) = 77 + 1/7 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l2453_245356


namespace NUMINAMATH_CALUDE_some_number_value_l2453_245388

theorem some_number_value (x : ℝ) :
  1 / 2 + ((2 / 3 * x) + 4) - 8 / 16 = 4.25 → x = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2453_245388
