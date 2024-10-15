import Mathlib

namespace NUMINAMATH_CALUDE_better_deal_gives_three_contacts_per_dollar_l2303_230311

/-- Represents a box of contacts with a given number of contacts and price --/
structure ContactBox where
  contacts : ℕ
  price : ℚ

/-- Calculates the number of contacts per dollar for a given box --/
def contactsPerDollar (box : ContactBox) : ℚ :=
  box.contacts / box.price

theorem better_deal_gives_three_contacts_per_dollar
  (box1 box2 : ContactBox)
  (h1 : box1 = ⟨50, 25⟩)
  (h2 : box2 = ⟨99, 33⟩)
  (h3 : contactsPerDollar box2 > contactsPerDollar box1) :
  contactsPerDollar box2 = 3 := by
  sorry

#check better_deal_gives_three_contacts_per_dollar

end NUMINAMATH_CALUDE_better_deal_gives_three_contacts_per_dollar_l2303_230311


namespace NUMINAMATH_CALUDE_second_car_speed_l2303_230342

/-- Two cars traveling on a road in the same direction -/
structure TwoCars where
  /-- Time of travel in seconds -/
  t : ℝ
  /-- Average speed of the first car in m/s -/
  v₁ : ℝ
  /-- Initial distance between cars in meters -/
  S₁ : ℝ
  /-- Final distance between cars in meters -/
  S₂ : ℝ

/-- Average speed of the second car -/
def averageSpeedSecondCar (cars : TwoCars) : Set ℝ :=
  let v_rel := (cars.S₁ - cars.S₂) / cars.t
  {cars.v₁ - v_rel, cars.v₁ + v_rel}

/-- Theorem stating the average speed of the second car -/
theorem second_car_speed (cars : TwoCars)
    (h_t : cars.t = 30)
    (h_v₁ : cars.v₁ = 30)
    (h_S₁ : cars.S₁ = 800)
    (h_S₂ : cars.S₂ = 200) :
    averageSpeedSecondCar cars = {10, 50} := by
  sorry

end NUMINAMATH_CALUDE_second_car_speed_l2303_230342


namespace NUMINAMATH_CALUDE_log_expression_equals_negative_one_l2303_230388

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_negative_one :
  log10 (5/2) + 2 * log10 2 - (1/2)⁻¹ = -1 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_negative_one_l2303_230388


namespace NUMINAMATH_CALUDE_circle_ratio_l2303_230320

theorem circle_ratio (α : Real) (r R x : Real) : 
  r > 0 → R > 0 → x > 0 → r < R →
  (R - r) = (R + r) * Real.sin α →
  x = (r * R) / ((Real.sqrt r + Real.sqrt R)^2) →
  (r / x) = 2 * (1 + Real.cos α) / (1 + Real.sin α) := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_l2303_230320


namespace NUMINAMATH_CALUDE_count_numbers_satisfying_condition_l2303_230353

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Define the property we're looking for
def satisfiesCondition (n : ℕ) : Prop :=
  n > 0 ∧ n < 1000 ∧ n = 7 * sumOfDigits n

-- State the theorem
theorem count_numbers_satisfying_condition :
  ∃ (S : Finset ℕ), S.card = 3 ∧ ∀ n, n ∈ S ↔ satisfiesCondition n :=
sorry

end NUMINAMATH_CALUDE_count_numbers_satisfying_condition_l2303_230353


namespace NUMINAMATH_CALUDE_same_solution_equations_l2303_230327

theorem same_solution_equations (x : ℝ) (d : ℝ) : 
  (3 * x + 8 = 5) ∧ (d * x - 15 = -7) → d = -8 := by
sorry

end NUMINAMATH_CALUDE_same_solution_equations_l2303_230327


namespace NUMINAMATH_CALUDE_don_rum_limit_l2303_230373

/-- The amount of rum Sally gave Don on his pancakes (in oz) -/
def sally_rum : ℝ := 10

/-- The multiplier for the maximum amount of rum Don can consume for a healthy diet -/
def max_multiplier : ℝ := 3

/-- The amount of rum Don had earlier that day (in oz) -/
def earlier_rum : ℝ := 12

/-- The amount of rum Don can have after eating all of the rum and pancakes (in oz) -/
def remaining_rum : ℝ := max_multiplier * sally_rum - earlier_rum

theorem don_rum_limit : remaining_rum = 18 := by sorry

end NUMINAMATH_CALUDE_don_rum_limit_l2303_230373


namespace NUMINAMATH_CALUDE_pages_per_side_l2303_230385

/-- Given the conditions of James' printing job, prove the number of pages per side. -/
theorem pages_per_side (num_books : ℕ) (pages_per_book : ℕ) (num_sheets : ℕ) : 
  num_books = 2 → 
  pages_per_book = 600 → 
  num_sheets = 150 → 
  (num_books * pages_per_book) / (num_sheets * 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_side_l2303_230385


namespace NUMINAMATH_CALUDE_latte_price_calculation_l2303_230332

-- Define the prices and quantities
def total_cost : ℚ := 25
def drip_coffee_price : ℚ := 2.25
def drip_coffee_quantity : ℕ := 2
def espresso_price : ℚ := 3.50
def espresso_quantity : ℕ := 1
def latte_quantity : ℕ := 2
def vanilla_syrup_price : ℚ := 0.50
def vanilla_syrup_quantity : ℕ := 1
def cold_brew_price : ℚ := 2.50
def cold_brew_quantity : ℕ := 2
def cappuccino_price : ℚ := 3.50
def cappuccino_quantity : ℕ := 1

-- Define the theorem
theorem latte_price_calculation :
  ∃ (latte_price : ℚ),
    latte_price * latte_quantity +
    drip_coffee_price * drip_coffee_quantity +
    espresso_price * espresso_quantity +
    vanilla_syrup_price * vanilla_syrup_quantity +
    cold_brew_price * cold_brew_quantity +
    cappuccino_price * cappuccino_quantity = total_cost ∧
    latte_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_latte_price_calculation_l2303_230332


namespace NUMINAMATH_CALUDE_ab_length_l2303_230316

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
def collinear (A B C D : ℝ × ℝ) : Prop := sorry
def distance (P Q : ℝ × ℝ) : ℝ := sorry
def perimeter (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem ab_length
  (h_collinear : collinear A B C D)
  (h_ab_cd : distance A B = distance C D)
  (h_bc : distance B C = 8)
  (h_be : distance B E = 12)
  (h_ce : distance C E = 12)
  (h_perimeter : perimeter A E D = 3 * perimeter B E C) :
  distance A B = 18 := by sorry

end NUMINAMATH_CALUDE_ab_length_l2303_230316


namespace NUMINAMATH_CALUDE_biker_bob_distance_l2303_230371

/-- The total distance covered by Biker Bob between town A and town B -/
def total_distance : ℝ := 155

/-- The distance of the first segment (west) -/
def distance_west : ℝ := 45

/-- The distance of the second segment (northwest) -/
def distance_northwest : ℝ := 25

/-- The distance of the third segment (south) -/
def distance_south : ℝ := 35

/-- The distance of the fourth segment (east) -/
def distance_east : ℝ := 50

/-- Theorem stating that the total distance is the sum of all segment distances -/
theorem biker_bob_distance : 
  total_distance = distance_west + distance_northwest + distance_south + distance_east := by
  sorry

#check biker_bob_distance

end NUMINAMATH_CALUDE_biker_bob_distance_l2303_230371


namespace NUMINAMATH_CALUDE_sumata_vacation_l2303_230394

/-- The Sumata family vacation problem -/
theorem sumata_vacation (miles_per_day : ℕ) (total_miles : ℕ) (h1 : miles_per_day = 250) (h2 : total_miles = 1250) :
  total_miles / miles_per_day = 5 := by
  sorry

end NUMINAMATH_CALUDE_sumata_vacation_l2303_230394


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2303_230378

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5/11) 
  (h2 : x - y = 1/55) : 
  x^2 - y^2 = 1/121 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2303_230378


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2303_230321

theorem simplify_and_evaluate (m : ℝ) (h : m = 2 - Real.sqrt 2) :
  (3 / (m + 1) + 1 - m) / ((m + 2) / (m + 1)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2303_230321


namespace NUMINAMATH_CALUDE_football_yards_gained_l2303_230347

/-- Represents the yards gained by a football team after an initial loss -/
def yards_gained (initial_loss : ℤ) (final_progress : ℤ) : ℤ :=
  final_progress - initial_loss

/-- Theorem: If a team loses 5 yards and ends with 6 yards of progress, they gained 11 yards -/
theorem football_yards_gained :
  yards_gained (-5) 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_football_yards_gained_l2303_230347


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2303_230346

theorem quadratic_roots_property (a b : ℝ) : 
  (3 * a^2 + 9 * a - 21 = 0) → 
  (3 * b^2 + 9 * b - 21 = 0) → 
  (3 * a - 4) * (2 * b - 2) = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2303_230346


namespace NUMINAMATH_CALUDE_chocolate_theorem_l2303_230386

/-- The difference between 75% of Robert's chocolates and the total number of chocolates Nickel and Penelope ate -/
def chocolate_difference (robert : ℝ) (nickel : ℝ) (penelope : ℝ) : ℝ :=
  0.75 * robert - (nickel + penelope)

/-- Theorem stating the difference in chocolates -/
theorem chocolate_theorem :
  chocolate_difference 13 4 7.5 = -1.75 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_theorem_l2303_230386


namespace NUMINAMATH_CALUDE_complement_union_eq_five_l2303_230374

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_eq_five : (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_union_eq_five_l2303_230374


namespace NUMINAMATH_CALUDE_picture_book_shelves_l2303_230383

theorem picture_book_shelves (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ)
  (h1 : books_per_shelf = 9)
  (h2 : mystery_shelves = 6)
  (h3 : total_books = 72) :
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 2 := by
  sorry

end NUMINAMATH_CALUDE_picture_book_shelves_l2303_230383


namespace NUMINAMATH_CALUDE_constant_expression_l2303_230331

-- Define the logarithm with base √2
noncomputable def log_sqrt2 (x : ℝ) : ℝ := Real.log x / Real.log (Real.sqrt 2)

-- State the theorem
theorem constant_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x^2 + y^2 = 18*x*y) :
  log_sqrt2 (x - y) - (log_sqrt2 x + log_sqrt2 y) / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_constant_expression_l2303_230331


namespace NUMINAMATH_CALUDE_f_odd_when_a_zero_f_increasing_iff_three_roots_iff_l2303_230372

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * abs (2 * a - x) + 2 * x

-- Statement 1: f is odd when a = 0
theorem f_odd_when_a_zero : 
  ∀ x : ℝ, f 0 (-x) = -(f 0 x) :=
sorry

-- Statement 2: f is increasing on ℝ iff -1 ≤ a ≤ 1
theorem f_increasing_iff :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ -1 ≤ a ∧ a ≤ 1 :=
sorry

-- Statement 3: f(x) - tf(2a) = 0 has three distinct roots iff 1 < t < 9/8
theorem three_roots_iff :
  ∀ a t : ℝ, a ∈ Set.Icc (-2) 2 →
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f a x - t * f a (2 * a) = 0 ∧
    f a y - t * f a (2 * a) = 0 ∧
    f a z - t * f a (2 * a) = 0) ↔
  1 < t ∧ t < 9/8 :=
sorry

end NUMINAMATH_CALUDE_f_odd_when_a_zero_f_increasing_iff_three_roots_iff_l2303_230372


namespace NUMINAMATH_CALUDE_specific_polyhedron_space_diagonals_l2303_230395

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ
  pentagon_faces : ℕ

/-- The number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  (Q.vertices.choose 2) - Q.edges - 
  (2 * Q.quadrilateral_faces + 5 * Q.pentagon_faces)

/-- Theorem: A specific convex polyhedron Q has 321 space diagonals -/
theorem specific_polyhedron_space_diagonals :
  ∃ Q : ConvexPolyhedron,
    Q.vertices = 30 ∧
    Q.edges = 70 ∧
    Q.faces = 42 ∧
    Q.triangular_faces = 26 ∧
    Q.quadrilateral_faces = 12 ∧
    Q.pentagon_faces = 4 ∧
    space_diagonals Q = 321 := by
  sorry

end NUMINAMATH_CALUDE_specific_polyhedron_space_diagonals_l2303_230395


namespace NUMINAMATH_CALUDE_odometer_skipping_four_l2303_230300

/-- Represents an odometer that skips the digit 4 -/
def SkippingOdometer : Type := ℕ

/-- Converts a regular number to its representation on the skipping odometer -/
def toSkippingOdometer (n : ℕ) : SkippingOdometer :=
  sorry

/-- Converts a skipping odometer reading back to the actual distance -/
def fromSkippingOdometer (s : SkippingOdometer) : ℕ :=
  sorry

/-- The theorem stating the relationship between the odometer reading and actual distance -/
theorem odometer_skipping_four (reading : SkippingOdometer) :
  reading = toSkippingOdometer 2005 →
  fromSkippingOdometer reading = 1462 :=
sorry

end NUMINAMATH_CALUDE_odometer_skipping_four_l2303_230300


namespace NUMINAMATH_CALUDE_total_writing_time_l2303_230335

theorem total_writing_time :
  let woody_time : ℝ := 18 -- Woody's writing time in months
  let ivanka_time : ℝ := woody_time + 3 -- Ivanka's writing time
  let alice_time : ℝ := woody_time / 2 -- Alice's writing time
  let tom_time : ℝ := alice_time * 2 -- Tom's writing time
  ivanka_time + woody_time + alice_time + tom_time = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_writing_time_l2303_230335


namespace NUMINAMATH_CALUDE_q_div_p_equals_48_l2303_230392

/-- The number of cards in the deck -/
def total_cards : ℕ := 52

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 13

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The number of cards for each number -/
def cards_per_number : ℕ := 4

/-- The probability of drawing all 5 cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The probability of drawing 4 cards of one number and 1 of another -/
def q : ℚ := (624 : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The theorem stating the ratio of q to p -/
theorem q_div_p_equals_48 : q / p = 48 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_48_l2303_230392


namespace NUMINAMATH_CALUDE_prob_at_least_one_heart_or_king_l2303_230396

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of cards that are either hearts or kings
def heart_or_king : ℕ := 16

-- Define the probability of not choosing a heart or king in one draw
def prob_not_heart_or_king : ℚ := (total_cards - heart_or_king) / total_cards

-- Theorem statement
theorem prob_at_least_one_heart_or_king :
  1 - prob_not_heart_or_king ^ 2 = 88 / 169 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_heart_or_king_l2303_230396


namespace NUMINAMATH_CALUDE_medicine_price_proof_l2303_230307

/-- Proves that the original price of a medicine is $150 given the specified conditions --/
theorem medicine_price_proof (cashback_rate : Real) (rebate : Real) (final_cost : Real) :
  cashback_rate = 0.1 →
  rebate = 25 →
  final_cost = 110 →
  ∃ (original_price : Real),
    original_price - (cashback_rate * original_price + rebate) = final_cost ∧
    original_price = 150 := by
  sorry

#check medicine_price_proof

end NUMINAMATH_CALUDE_medicine_price_proof_l2303_230307


namespace NUMINAMATH_CALUDE_calculation_proof_l2303_230365

theorem calculation_proof : 8 * 2.25 - 5 * 0.85 / 2.5 = 16.3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2303_230365


namespace NUMINAMATH_CALUDE_exists_number_divisible_by_digit_sum_l2303_230325

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if all digits of a number are non-zero -/
def all_digits_nonzero (n : ℕ) : Prop := sorry

/-- Number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For all positive integers n, there exists an n-digit number z
    such that none of its digits are 0 and z is divisible by the sum of its digits -/
theorem exists_number_divisible_by_digit_sum :
  ∀ n : ℕ, n > 0 → ∃ z : ℕ,
    num_digits z = n ∧
    all_digits_nonzero z ∧
    z % sum_of_digits z = 0 :=
by sorry

end NUMINAMATH_CALUDE_exists_number_divisible_by_digit_sum_l2303_230325


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2303_230309

theorem decimal_sum_to_fraction :
  (0.1 : ℚ) + 0.02 + 0.003 + 0.0004 + 0.00005 = 2469 / 20000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2303_230309


namespace NUMINAMATH_CALUDE_ab_minus_bc_plus_ac_equals_seven_l2303_230317

theorem ab_minus_bc_plus_ac_equals_seven 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 14) 
  (h2 : a = b + c) : 
  a*b - b*c + a*c = 7 := by
sorry

end NUMINAMATH_CALUDE_ab_minus_bc_plus_ac_equals_seven_l2303_230317


namespace NUMINAMATH_CALUDE_gain_percentage_calculation_l2303_230360

/-- Given a selling price and a gain, calculate the gain percentage. -/
theorem gain_percentage_calculation (selling_price gain : ℝ) 
  (h1 : selling_price = 195)
  (h2 : gain = 45) : 
  (gain / (selling_price - gain)) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gain_percentage_calculation_l2303_230360


namespace NUMINAMATH_CALUDE_optimal_shelf_arrangement_l2303_230389

def math_books : ℕ := 130
def portuguese_books : ℕ := 195

theorem optimal_shelf_arrangement :
  ∃ (n : ℕ), n > 0 ∧
  n ∣ math_books ∧
  n ∣ portuguese_books ∧
  (∀ m : ℕ, m > n → ¬(m ∣ math_books ∧ m ∣ portuguese_books)) ∧
  n = 65 := by
  sorry

end NUMINAMATH_CALUDE_optimal_shelf_arrangement_l2303_230389


namespace NUMINAMATH_CALUDE_intersects_x_axis_once_iff_a_eq_zero_or_one_or_nine_l2303_230391

/-- A function f(x) intersects the x-axis at only one point if and only if
    it has exactly one real root or it is a non-constant linear function. -/
def intersects_x_axis_once (f : ℝ → ℝ) : Prop :=
  (∃! x, f x = 0) ∨ (∃ m b, m ≠ 0 ∧ ∀ x, f x = m * x + b)

/-- The quadratic function f(x) = ax² + (a-3)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 3) * x + 1

theorem intersects_x_axis_once_iff_a_eq_zero_or_one_or_nine :
  ∀ a : ℝ, intersects_x_axis_once (f a) ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end NUMINAMATH_CALUDE_intersects_x_axis_once_iff_a_eq_zero_or_one_or_nine_l2303_230391


namespace NUMINAMATH_CALUDE_triangle_cannot_be_formed_l2303_230369

theorem triangle_cannot_be_formed (a b c : ℝ) (h1 : a = 8) (h2 : b = 6) (h3 : c = 9) : 
  ¬ (∃ (a' b' c' : ℝ), a' = a * 1.5 ∧ b' = b * (1 - 0.333) ∧ c' = c ∧ 
    a' + b' > c' ∧ a' + c' > b' ∧ b' + c' > a') :=
by sorry

end NUMINAMATH_CALUDE_triangle_cannot_be_formed_l2303_230369


namespace NUMINAMATH_CALUDE_malik_yards_per_game_l2303_230334

-- Define the number of games
def num_games : ℕ := 4

-- Define Josiah's yards per game
def josiah_yards_per_game : ℕ := 22

-- Define Darnell's average yards per game
def darnell_avg_yards : ℕ := 11

-- Define the total yards run by all three athletes
def total_yards : ℕ := 204

-- Theorem to prove
theorem malik_yards_per_game :
  ∃ (malik_yards : ℕ),
    malik_yards * num_games + 
    josiah_yards_per_game * num_games + 
    darnell_avg_yards * num_games = 
    total_yards ∧ 
    malik_yards = 18 := by
  sorry

end NUMINAMATH_CALUDE_malik_yards_per_game_l2303_230334


namespace NUMINAMATH_CALUDE_star_six_five_l2303_230345

-- Define the star operation
def star (a b : ℕ+) : ℚ :=
  (a.val * (2 * b.val)) / (a.val + 2 * b.val + 3)

-- Theorem statement
theorem star_six_five :
  star 6 5 = 60 / 19 := by
  sorry

end NUMINAMATH_CALUDE_star_six_five_l2303_230345


namespace NUMINAMATH_CALUDE_absolute_value_difference_l2303_230315

theorem absolute_value_difference (a b : ℝ) : 
  (|a| = 2) → (|b| = 5) → (a < b) → ((a - b = -3) ∨ (a - b = -7)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_difference_l2303_230315


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2303_230329

theorem solve_linear_equation (m : ℝ) (x : ℝ) : 
  (m * x + 1 = 2) → (x = -1) → (m = -1) := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2303_230329


namespace NUMINAMATH_CALUDE_willies_stickers_l2303_230308

/-- Willie's sticker problem -/
theorem willies_stickers (initial_stickers given_away : ℕ) 
  (h1 : initial_stickers = 36)
  (h2 : given_away = 7) :
  initial_stickers - given_away = 29 := by
  sorry

end NUMINAMATH_CALUDE_willies_stickers_l2303_230308


namespace NUMINAMATH_CALUDE_triangular_number_all_equal_digits_l2303_230367

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def all_digits_equal (num : ℕ) (digit : ℕ) : Prop :=
  ∀ d, d ∈ num.digits 10 → d = digit

theorem triangular_number_all_equal_digits :
  {a : ℕ | a < 10 ∧ ∃ n : ℕ, n ≥ 4 ∧ all_digits_equal (triangular_number n) a} = {5, 6} := by
  sorry

end NUMINAMATH_CALUDE_triangular_number_all_equal_digits_l2303_230367


namespace NUMINAMATH_CALUDE_max_table_coverage_max_table_side_optimal_l2303_230302

/-- The side length of each square tablecloth in centimeters -/
def tablecloth_side : ℝ := 144

/-- The number of tablecloths available -/
def num_tablecloths : ℕ := 3

/-- The maximum side length of the square table that can be covered -/
def max_table_side : ℝ := 183

/-- Theorem stating that the maximum side length of a square table that can be completely
    covered by three square tablecloths, each with a side length of 144 cm, is 183 cm -/
theorem max_table_coverage :
  ∀ (table_side : ℝ),
  table_side ≤ max_table_side →
  (table_side ^ 2 : ℝ) ≤ num_tablecloths * tablecloth_side ^ 2 :=
by sorry

/-- Theorem stating that 183 cm is the largest possible side length for the table -/
theorem max_table_side_optimal :
  ∀ (larger_side : ℝ),
  larger_side > max_table_side →
  (larger_side ^ 2 : ℝ) > num_tablecloths * tablecloth_side ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_table_coverage_max_table_side_optimal_l2303_230302


namespace NUMINAMATH_CALUDE_tonya_lemonade_revenue_l2303_230349

/-- Calculates the total revenue from Tonya's lemonade stand --/
def lemonade_revenue (small_price medium_price large_price : ℕ)
  (small_revenue medium_revenue : ℕ) (large_cups : ℕ) : ℕ :=
  small_revenue + medium_revenue + (large_cups * large_price)

theorem tonya_lemonade_revenue :
  lemonade_revenue 1 2 3 11 24 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_tonya_lemonade_revenue_l2303_230349


namespace NUMINAMATH_CALUDE_tangent_line_intercept_l2303_230362

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + a*x + 2

-- Define the derivative of f(x)
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 6*x + a

theorem tangent_line_intercept (a : ℝ) : 
  (f a 0 = 2) ∧ 
  (∃ m : ℝ, ∀ x : ℝ, m*x + 2 = f_prime a 0 * x + 2) ∧
  (∃ t : ℝ, t = -2 ∧ f_prime a 0 * t + 2 = 0) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_intercept_l2303_230362


namespace NUMINAMATH_CALUDE_intersection_when_m_is_5_subset_condition_l2303_230363

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 3 * m - 1}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 10}

-- Theorem for part 1
theorem intersection_when_m_is_5 :
  A 5 ∩ B = {x | 6 ≤ x ∧ x ≤ 10} := by sorry

-- Theorem for part 2
theorem subset_condition :
  ∀ m : ℝ, A m ⊆ B ↔ m ≤ 11/3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_5_subset_condition_l2303_230363


namespace NUMINAMATH_CALUDE_negation_equivalence_l2303_230398

theorem negation_equivalence (a b : ℝ) :
  ¬(((a - 2) * (b - 3) = 0) → (a = 2 ∨ b = 3)) ↔
  (((a - 2) * (b - 3) ≠ 0) → (a ≠ 2 ∧ b ≠ 3)) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2303_230398


namespace NUMINAMATH_CALUDE_fraction_equality_l2303_230356

theorem fraction_equality (a b : ℚ) (h : (a - 2*b) / b = 3/5) : a / b = 13/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2303_230356


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l2303_230330

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- First asymptote: y = 2x + 3 -/
  asymptote1 : ℝ → ℝ
  /-- Second asymptote: y = -2x - 1 -/
  asymptote2 : ℝ → ℝ
  /-- The hyperbola passes through this point -/
  point : ℝ × ℝ
  /-- The first asymptote has the form y = 2x + 3 -/
  h₁ : ∀ x, asymptote1 x = 2 * x + 3
  /-- The second asymptote has the form y = -2x - 1 -/
  h₂ : ∀ x, asymptote2 x = -2 * x - 1
  /-- The point (4, 5) lies on the hyperbola -/
  h₃ : point = (4, 5)

/-- The distance between the foci of the hyperbola is 6√2 -/
theorem hyperbola_foci_distance (h : Hyperbola) : 
  ∃ (f₁ f₂ : ℝ × ℝ), (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l2303_230330


namespace NUMINAMATH_CALUDE_jenny_reading_time_l2303_230370

/-- Calculates the average daily reading time including breaks -/
def averageDailyReadingTime (numBooks : ℕ) (totalDays : ℕ) (readingSpeed : ℕ) 
  (breakDuration : ℕ) (breakInterval : ℕ) (bookWords : List ℕ) : ℕ :=
  let totalWords := bookWords.sum
  let readingMinutes := totalWords / readingSpeed
  let readingHours := readingMinutes / 60
  let numBreaks := readingHours
  let breakMinutes := numBreaks * breakDuration
  let totalMinutes := readingMinutes + breakMinutes
  totalMinutes / totalDays

/-- Theorem: Jenny's average daily reading time is 124 minutes -/
theorem jenny_reading_time :
  let numBooks := 5
  let totalDays := 15
  let readingSpeed := 60  -- words per minute
  let breakDuration := 15  -- minutes
  let breakInterval := 60  -- minutes
  let bookWords := [12000, 18000, 24000, 15000, 21000]
  averageDailyReadingTime numBooks totalDays readingSpeed breakDuration breakInterval bookWords = 124 := by
  sorry

end NUMINAMATH_CALUDE_jenny_reading_time_l2303_230370


namespace NUMINAMATH_CALUDE_mean_chocolate_sales_l2303_230337

def week1_sales : ℕ := 75
def week2_sales : ℕ := 67
def week3_sales : ℕ := 75
def week4_sales : ℕ := 70
def week5_sales : ℕ := 68
def num_weeks : ℕ := 5

def total_sales : ℕ := week1_sales + week2_sales + week3_sales + week4_sales + week5_sales

theorem mean_chocolate_sales :
  (total_sales : ℚ) / num_weeks = 71 := by sorry

end NUMINAMATH_CALUDE_mean_chocolate_sales_l2303_230337


namespace NUMINAMATH_CALUDE_approximate_4_02_to_ten_thousandth_l2303_230366

/-- Represents a decimal number with a specific precision -/
structure DecimalNumber where
  value : ℚ
  precision : ℕ

/-- Represents the place value in a decimal number -/
inductive PlaceValue
  | Ones
  | Tenths
  | Hundredths
  | Thousandths
  | TenThousandths

/-- Determines the place value of the last non-zero digit in a decimal number -/
def lastNonZeroDigitPlace (n : DecimalNumber) : PlaceValue :=
  sorry

/-- Approximates a decimal number to a given place value -/
def approximateTo (n : DecimalNumber) (place : PlaceValue) : DecimalNumber :=
  sorry

/-- Theorem stating that approximating 4.02 to the ten thousandth place
    results in a number accurate to the hundredth place -/
theorem approximate_4_02_to_ten_thousandth :
  let original := DecimalNumber.mk (402 / 100) 2
  let approximated := approximateTo original PlaceValue.TenThousandths
  lastNonZeroDigitPlace approximated = PlaceValue.Hundredths :=
sorry

end NUMINAMATH_CALUDE_approximate_4_02_to_ten_thousandth_l2303_230366


namespace NUMINAMATH_CALUDE_multiple_properties_l2303_230379

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ 
  (∃ p : ℤ, a + b = 4 * p) ∧ 
  (∃ q : ℤ, a + b = 2 * q) := by
sorry

end NUMINAMATH_CALUDE_multiple_properties_l2303_230379


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l2303_230382

theorem max_value_sum_of_roots (a b c : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0) 
  (sum_constraint : a + b + c = 8) :
  (∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 8 ∧
    Real.sqrt (3 * x^2 + 1) + Real.sqrt (3 * y^2 + 1) + Real.sqrt (3 * z^2 + 1) > 
    Real.sqrt (3 * a^2 + 1) + Real.sqrt (3 * b^2 + 1) + Real.sqrt (3 * c^2 + 1)) ∨
  (Real.sqrt (3 * a^2 + 1) + Real.sqrt (3 * b^2 + 1) + Real.sqrt (3 * c^2 + 1) = Real.sqrt 201) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l2303_230382


namespace NUMINAMATH_CALUDE_frank_max_average_time_l2303_230310

/-- The maximum average time per maze Frank wants to maintain -/
def maxAverageTime (previousMazes : ℕ) (averagePreviousTime : ℕ) (currentTime : ℕ) (remainingTime : ℕ) : ℚ :=
  let totalPreviousTime := previousMazes * averagePreviousTime
  let totalCurrentTime := currentTime + remainingTime
  let totalTime := totalPreviousTime + totalCurrentTime
  let totalMazes := previousMazes + 1
  totalTime / totalMazes

/-- Theorem stating the maximum average time Frank wants to maintain -/
theorem frank_max_average_time :
  maxAverageTime 4 50 45 55 = 60 := by
  sorry

end NUMINAMATH_CALUDE_frank_max_average_time_l2303_230310


namespace NUMINAMATH_CALUDE_rational_numbers_composition_l2303_230351

-- Define the set of integers
def Integers : Set ℚ := {x : ℚ | ∃ n : ℤ, x = n}

-- Define the set of fractions
def Fractions : Set ℚ := {x : ℚ | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}

-- Theorem statement
theorem rational_numbers_composition :
  Set.univ = Integers ∪ Fractions :=
sorry

end NUMINAMATH_CALUDE_rational_numbers_composition_l2303_230351


namespace NUMINAMATH_CALUDE_average_of_25_results_l2303_230344

theorem average_of_25_results (results : List ℝ) 
  (h1 : results.length = 25)
  (h2 : (results.take 12).sum / 12 = 14)
  (h3 : (results.drop 13).sum / 12 = 17)
  (h4 : results[12] = 128) :
  results.sum / 25 = 20 := by
  sorry

end NUMINAMATH_CALUDE_average_of_25_results_l2303_230344


namespace NUMINAMATH_CALUDE_fair_number_exists_l2303_230358

/-- Represents a digit as a natural number between 0 and 9 -/
def Digit : Type := { n : ℕ // n < 10 }

/-- Represents a number as a list of digits -/
def Number := List Digit

/-- Checks if a digit is even -/
def isEven (d : Digit) : Bool :=
  d.val % 2 = 0

/-- Counts the number of even digits at odd positions and even positions -/
def countEvenDigits (n : Number) : ℕ × ℕ :=
  let rec count (digits : List Digit) (isOddPosition : Bool) (evenOdd evenEven : ℕ) : ℕ × ℕ :=
    match digits with
    | [] => (evenOdd, evenEven)
    | d :: ds =>
      if isEven d then
        if isOddPosition then
          count ds (not isOddPosition) (evenOdd + 1) evenEven
        else
          count ds (not isOddPosition) evenOdd (evenEven + 1)
      else
        count ds (not isOddPosition) evenOdd evenEven
  count n true 0 0

/-- Checks if a number is fair (equal number of even digits at odd and even positions) -/
def isFair (n : Number) : Bool :=
  let (evenOdd, evenEven) := countEvenDigits n
  evenOdd = evenEven

/-- Main theorem: For any number with an odd number of digits, 
    there exists a way to remove one digit to make it fair -/
theorem fair_number_exists (n : Number) (h : n.length % 2 = 1) :
  ∃ (i : Fin n.length), isFair (n.removeNth i) := by
  sorry

end NUMINAMATH_CALUDE_fair_number_exists_l2303_230358


namespace NUMINAMATH_CALUDE_franklin_valentines_l2303_230333

/-- The number of Valentines Mrs. Franklin gave away -/
def valentines_given : ℕ := 42

/-- The number of Valentines Mrs. Franklin has left -/
def valentines_left : ℕ := 16

/-- The initial number of Valentines Mrs. Franklin had -/
def initial_valentines : ℕ := valentines_given + valentines_left

theorem franklin_valentines : initial_valentines = 58 := by
  sorry

end NUMINAMATH_CALUDE_franklin_valentines_l2303_230333


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_less_than_four_l2303_230313

-- Define the system of inequalities
def has_solution (m : ℝ) : Prop :=
  ∃ x : ℝ, (2 * x - 6 + m < 0) ∧ (4 * x - m > 0)

-- State the theorem
theorem inequality_solution_implies_m_less_than_four :
  ∀ m : ℝ, has_solution m → m < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_less_than_four_l2303_230313


namespace NUMINAMATH_CALUDE_bridge_problem_l2303_230397

/-- A graph representing the bridge system. -/
structure BridgeGraph where
  /-- The set of nodes (islands) in the graph. -/
  nodes : Finset (Fin 4)
  /-- The set of edges (bridges) in the graph. -/
  edges : Finset (Fin 4 × Fin 4)
  /-- The degree of each node. -/
  degree : Fin 4 → Nat
  /-- Condition that node 0 (A) has degree 3. -/
  degree_A : degree 0 = 3
  /-- Condition that node 1 (B) has degree 5. -/
  degree_B : degree 1 = 5
  /-- Condition that node 2 (C) has degree 3. -/
  degree_C : degree 2 = 3
  /-- Condition that node 3 (D) has degree 3. -/
  degree_D : degree 3 = 3
  /-- The total number of edges is 9. -/
  edge_count : edges.card = 9

/-- The number of Eulerian paths in the bridge graph. -/
def countEulerianPaths (g : BridgeGraph) : Nat :=
  sorry

/-- Theorem stating that the number of Eulerian paths is 132. -/
theorem bridge_problem (g : BridgeGraph) : countEulerianPaths g = 132 :=
  sorry

end NUMINAMATH_CALUDE_bridge_problem_l2303_230397


namespace NUMINAMATH_CALUDE_root_relation_implies_a_value_l2303_230381

theorem root_relation_implies_a_value (m : ℝ) (h : m > 0) :
  ∃ (a : ℝ), ∀ (x : ℂ),
    (x^4 + 2*x^2 + 1) / (2*(x^3 + x)) = a →
    (∃ (y : ℂ), (x^4 + 2*x^2 + 1) / (2*(x^3 + x)) = a ∧ x = m * y) →
    a = (m + 1) / (2 * m) * Real.sqrt m :=
by sorry

end NUMINAMATH_CALUDE_root_relation_implies_a_value_l2303_230381


namespace NUMINAMATH_CALUDE_f_values_l2303_230340

noncomputable def f (x : ℝ) : ℝ :=
  if -1 < x ∧ x < 0 then Real.sin (Real.pi * x^2)
  else if x ≥ 0 then Real.exp (x - 1)
  else 0  -- undefined for x ≤ -1

theorem f_values (a : ℝ) : f 1 + f a = 2 → a = 1 ∨ a = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_values_l2303_230340


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2303_230312

/-- A quadratic function with positive leading coefficient and symmetry about x = 2 -/
def symmetric_quadratic (a b c : ℝ) : ℝ → ℝ :=
  fun x ↦ a * x^2 + b * x + c

theorem quadratic_inequality 
  (a b c : ℝ) 
  (ha : a > 0)
  (h_sym : ∀ x, symmetric_quadratic a b c (x + 2) = symmetric_quadratic a b c (2 - x)) :
  symmetric_quadratic a b c (Real.sqrt 2 / 2) > symmetric_quadratic a b c Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2303_230312


namespace NUMINAMATH_CALUDE_total_scheduling_arrangements_l2303_230339

/-- Represents the total number of periods in a day -/
def total_periods : ℕ := 6

/-- Represents the number of morning periods -/
def morning_periods : ℕ := 4

/-- Represents the number of afternoon periods -/
def afternoon_periods : ℕ := 2

/-- Represents the total number of subjects to be scheduled -/
def total_subjects : ℕ := 6

/-- Represents the number of ways to schedule Math in the morning -/
def math_morning_options : ℕ := 4

/-- Represents the number of ways to schedule Physical Education (excluding first morning period) -/
def pe_options : ℕ := 5

/-- Represents the number of ways to arrange the remaining subjects -/
def remaining_arrangements : ℕ := 24

/-- Theorem stating the total number of different scheduling arrangements -/
theorem total_scheduling_arrangements :
  math_morning_options * pe_options * remaining_arrangements = 480 := by
  sorry

end NUMINAMATH_CALUDE_total_scheduling_arrangements_l2303_230339


namespace NUMINAMATH_CALUDE_max_running_speed_l2303_230338

/-- The maximum speed at which a person can run to catch a train, given specific conditions -/
theorem max_running_speed (x : ℝ) (h : x > 0) : 
  let v := (30 : ℝ) / 3
  let train_speed := (30 : ℝ)
  let distance_fraction := (1 : ℝ) / 3
  (distance_fraction * x) / v = x / train_speed ∧ 
  ((1 - distance_fraction) * x) / v = (x + (distance_fraction * x)) / train_speed →
  v = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_running_speed_l2303_230338


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l2303_230393

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The main theorem stating that if A(a,-2) and B(4,b) are symmetric with respect to the origin, then a-b = -6 -/
theorem symmetric_points_difference (a b : ℝ) : 
  symmetric_wrt_origin a (-2) 4 b → a - b = -6 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l2303_230393


namespace NUMINAMATH_CALUDE_binomial_seven_four_l2303_230328

theorem binomial_seven_four : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_seven_four_l2303_230328


namespace NUMINAMATH_CALUDE_jennifer_grooming_time_l2303_230306

/-- Calculates the total grooming time in hours for a given number of dogs, 
    grooming time per dog, and number of days. -/
def totalGroomingTime (numDogs : ℕ) (groomTimePerDog : ℕ) (numDays : ℕ) : ℚ :=
  (numDogs * groomTimePerDog * numDays : ℚ) / 60

/-- Proves that Jennifer spends 20 hours grooming her dogs in 30 days. -/
theorem jennifer_grooming_time :
  totalGroomingTime 2 20 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_grooming_time_l2303_230306


namespace NUMINAMATH_CALUDE_christen_peeled_24_potatoes_l2303_230357

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  totalPotatoes : ℕ
  homerRate : ℕ
  christenRate : ℕ
  timeBeforeChristenJoins : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def christenPeeledPotatoes (scenario : PotatoPeeling) : ℕ :=
  let potatoesLeftAfterHomer := scenario.totalPotatoes - scenario.homerRate * scenario.timeBeforeChristenJoins
  let combinedRate := scenario.homerRate + scenario.christenRate
  let timeForRemaining := potatoesLeftAfterHomer / combinedRate
  scenario.christenRate * timeForRemaining

/-- Theorem stating that Christen peeled 24 potatoes -/
theorem christen_peeled_24_potatoes (scenario : PotatoPeeling) 
  (h1 : scenario.totalPotatoes = 60)
  (h2 : scenario.homerRate = 3)
  (h3 : scenario.christenRate = 4)
  (h4 : scenario.timeBeforeChristenJoins = 6) :
  christenPeeledPotatoes scenario = 24 := by
  sorry

end NUMINAMATH_CALUDE_christen_peeled_24_potatoes_l2303_230357


namespace NUMINAMATH_CALUDE_function_identity_proof_l2303_230336

theorem function_identity_proof (f : ℕ+ → ℕ+) 
  (h : ∀ (m n : ℕ+), (f m)^2 + f n ∣ (m^2 + n)^2) : 
  ∀ (n : ℕ+), f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_proof_l2303_230336


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2303_230305

theorem complex_magnitude_problem (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) :
  Complex.abs (z + Complex.I) = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2303_230305


namespace NUMINAMATH_CALUDE_runners_meeting_time_l2303_230384

def lap_time_bob : ℕ := 8
def lap_time_carol : ℕ := 9
def lap_time_ted : ℕ := 10

def meeting_time : ℕ := 360

theorem runners_meeting_time :
  Nat.lcm (Nat.lcm lap_time_bob lap_time_carol) lap_time_ted = meeting_time :=
by sorry

end NUMINAMATH_CALUDE_runners_meeting_time_l2303_230384


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_22_over_3_l2303_230303

theorem greatest_integer_less_than_negative_22_over_3 :
  ⌊-22 / 3⌋ = -8 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_22_over_3_l2303_230303


namespace NUMINAMATH_CALUDE_fifth_month_sale_l2303_230355

def average_sale : ℕ := 5600
def num_months : ℕ := 6
def sale_month1 : ℕ := 5400
def sale_month2 : ℕ := 9000
def sale_month3 : ℕ := 6300
def sale_month4 : ℕ := 7200
def sale_month6 : ℕ := 1200

theorem fifth_month_sale :
  ∃ (sale_month5 : ℕ),
    sale_month5 = average_sale * num_months - (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month6) ∧
    sale_month5 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l2303_230355


namespace NUMINAMATH_CALUDE_smallest_w_l2303_230323

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 ∧ 
  is_factor (2^5) (936 * w) ∧ 
  is_factor (3^3) (936 * w) ∧ 
  is_factor (10^2) (936 * w) →
  w ≥ 300 ∧ 
  ∃ (v : ℕ), v = 300 ∧ 
    v > 0 ∧ 
    is_factor (2^5) (936 * v) ∧ 
    is_factor (3^3) (936 * v) ∧ 
    is_factor (10^2) (936 * v) :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_l2303_230323


namespace NUMINAMATH_CALUDE_two_thousand_twelfth_digit_l2303_230364

def digit_sequence (n : ℕ) : ℕ :=
  sorry

theorem two_thousand_twelfth_digit :
  digit_sequence 2012 = 0 :=
sorry

end NUMINAMATH_CALUDE_two_thousand_twelfth_digit_l2303_230364


namespace NUMINAMATH_CALUDE_cucumbers_per_kind_paulines_garden_cucumbers_l2303_230376

/-- Calculates the number of cucumbers of each kind in Pauline's garden. -/
theorem cucumbers_per_kind (total_spaces : ℕ) (total_tomatoes : ℕ) (total_potatoes : ℕ) 
  (cucumber_kinds : ℕ) (empty_spaces : ℕ) : ℕ :=
  by
  have filled_spaces : ℕ := total_spaces - empty_spaces
  have non_cucumber_spaces : ℕ := total_tomatoes + total_potatoes
  have cucumber_spaces : ℕ := filled_spaces - non_cucumber_spaces
  exact cucumber_spaces / cucumber_kinds

/-- Proves that Pauline has planted 4 cucumbers of each kind in her garden. -/
theorem paulines_garden_cucumbers : 
  cucumbers_per_kind 150 15 30 5 85 = 4 :=
by sorry

end NUMINAMATH_CALUDE_cucumbers_per_kind_paulines_garden_cucumbers_l2303_230376


namespace NUMINAMATH_CALUDE_left_square_side_length_l2303_230324

/-- Given three squares with specific relationships between their side lengths,
    prove that the side length of the left square is 8 cm. -/
theorem left_square_side_length (x y z : ℝ) 
  (sum_condition : x + y + z = 52)
  (middle_square_condition : y = x + 17)
  (right_square_condition : z = y - 6) :
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_left_square_side_length_l2303_230324


namespace NUMINAMATH_CALUDE_garden_breadth_l2303_230304

/-- 
Given a rectangular garden with perimeter 800 meters and length 300 meters,
prove that its breadth is 100 meters.
-/
theorem garden_breadth (perimeter length breadth : ℝ) 
  (h1 : perimeter = 800)
  (h2 : length = 300)
  (h3 : perimeter = 2 * (length + breadth)) : 
  breadth = 100 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_l2303_230304


namespace NUMINAMATH_CALUDE_flu_free_inhabitants_l2303_230318

theorem flu_free_inhabitants (total_population : ℕ) (flu_percentage : ℚ) : 
  total_population = 14000000 →
  flu_percentage = 15 / 10000 →
  (total_population : ℚ) - (flu_percentage * total_population) = 13979000 := by
  sorry

end NUMINAMATH_CALUDE_flu_free_inhabitants_l2303_230318


namespace NUMINAMATH_CALUDE_salary_calculation_l2303_230380

theorem salary_calculation (S : ℝ) 
  (food_expense : S / 5 = S * (1 / 5))
  (rent_expense : S / 10 = S * (1 / 10))
  (clothes_expense : S * (3 / 5) = S * (3 / 5))
  (remaining : S - (S * (1 / 5) + S * (1 / 10) + S * (3 / 5)) = 16000) :
  S = 160000 := by
  sorry

end NUMINAMATH_CALUDE_salary_calculation_l2303_230380


namespace NUMINAMATH_CALUDE_inequality_proof_l2303_230354

/-- Proves that given a = 0.1e^0.1, b = 1/9, and c = -ln 0.9, the inequality c < a < b holds -/
theorem inequality_proof (a b c : ℝ) 
  (ha : a = 0.1 * Real.exp 0.1) 
  (hb : b = 1 / 9) 
  (hc : c = -Real.log 0.9) : 
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2303_230354


namespace NUMINAMATH_CALUDE_jennis_age_l2303_230343

theorem jennis_age (sum difference : ℕ) (h1 : sum = 70) (h2 : difference = 32) :
  ∃ (mrs_bai jenni : ℕ), mrs_bai + jenni = sum ∧ mrs_bai - jenni = difference ∧ jenni = 19 := by
  sorry

end NUMINAMATH_CALUDE_jennis_age_l2303_230343


namespace NUMINAMATH_CALUDE_sqrt_30_bounds_l2303_230326

theorem sqrt_30_bounds : 5 < Real.sqrt 30 ∧ Real.sqrt 30 < 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_30_bounds_l2303_230326


namespace NUMINAMATH_CALUDE_sin_less_than_x_in_interval_exp_x_plus_one_greater_than_neg_e_squared_l2303_230322

-- Option A
theorem sin_less_than_x_in_interval (x : ℝ) (h : x ∈ Set.Ioo 0 Real.pi) : x > Real.sin x := by
  sorry

-- Option C
theorem exp_x_plus_one_greater_than_neg_e_squared (x : ℝ) : (x + 1) * Real.exp x > -(1 / Real.exp 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_less_than_x_in_interval_exp_x_plus_one_greater_than_neg_e_squared_l2303_230322


namespace NUMINAMATH_CALUDE_diff_suit_prob_is_13_17_l2303_230375

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- The suits in a standard deck -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- A function that assigns a suit to each card in the deck -/
def card_suit : Fin 52 → Suit := sorry

/-- The probability of picking two cards of different suits -/
def diff_suit_prob (d : Deck) : ℚ :=
  (39 : ℚ) / 51

/-- Theorem stating that the probability of picking two cards of different suits is 13/17 -/
theorem diff_suit_prob_is_13_17 (d : Deck) :
  diff_suit_prob d = 13 / 17 := by
  sorry

end NUMINAMATH_CALUDE_diff_suit_prob_is_13_17_l2303_230375


namespace NUMINAMATH_CALUDE_unique_base_ten_l2303_230350

/-- Converts a list of digits in base b to its decimal representation -/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Checks if the equation is valid in base b -/
def isValidEquation (b : Nat) : Prop :=
  toDecimal [8, 7, 3, 6, 4] b + toDecimal [9, 2, 4, 1, 7] b = toDecimal [1, 8, 5, 8, 7, 1] b

theorem unique_base_ten :
  ∃! b, isValidEquation b ∧ b = 10 := by sorry

end NUMINAMATH_CALUDE_unique_base_ten_l2303_230350


namespace NUMINAMATH_CALUDE_book_price_is_480_l2303_230359

/-- The price of a book that Tara sells to reach her goal of buying a clarinet -/
def book_price : ℚ :=
  let clarinet_cost : ℚ := 90
  let initial_savings : ℚ := 10
  let needed_amount : ℚ := clarinet_cost - initial_savings
  let lost_savings : ℚ := needed_amount / 2
  let total_to_save : ℚ := needed_amount + lost_savings
  let num_books : ℚ := 25
  total_to_save / num_books

/-- Theorem stating that the book price is $4.80 -/
theorem book_price_is_480 : book_price = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_book_price_is_480_l2303_230359


namespace NUMINAMATH_CALUDE_stone_skipping_l2303_230301

/-- Represents the number of skips for each throw --/
structure Throws where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- Defines the conditions of the stone-skipping problem --/
def validThrows (t : Throws) : Prop :=
  t.second = t.first + 2 ∧
  t.third = 2 * t.second ∧
  t.fourth = t.third - 3 ∧
  t.fifth = 8 ∧
  t.first + t.second + t.third + t.fourth + t.fifth = 33

/-- The theorem to be proved --/
theorem stone_skipping (t : Throws) (h : validThrows t) : 
  t.fifth - t.fourth = 1 := by
  sorry

end NUMINAMATH_CALUDE_stone_skipping_l2303_230301


namespace NUMINAMATH_CALUDE_denise_crayon_sharing_l2303_230352

/-- The number of crayons Denise has -/
def total_crayons : ℕ := 210

/-- The number of crayons each friend gets -/
def crayons_per_friend : ℕ := 7

/-- The number of friends Denise shares crayons with -/
def number_of_friends : ℕ := total_crayons / crayons_per_friend

theorem denise_crayon_sharing :
  number_of_friends = 30 := by sorry

end NUMINAMATH_CALUDE_denise_crayon_sharing_l2303_230352


namespace NUMINAMATH_CALUDE_min_value_expression_l2303_230314

theorem min_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : a > 0) (h4 : b ≠ 0) :
  ((a + b)^3 + (b - c)^2 + (c - a)^3) / b^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2303_230314


namespace NUMINAMATH_CALUDE_range_of_a_l2303_230348

-- Define the conditions
def p (x : ℝ) : Prop := x^2 - 8*x - 33 > 0
def q (x a : ℝ) : Prop := |x - 1| > a

-- Define the theorem
theorem range_of_a (h : ∀ x a : ℝ, a > 0 → (p x → q x a) ∧ ¬(q x a → p x)) :
  ∃ a : ℝ, a > 0 ∧ a ≤ 4 ∧ ∀ b : ℝ, (b > 0 ∧ b ≤ 4 → ∃ x : ℝ, p x → q x b) ∧
    (b > 4 → ∃ x : ℝ, p x ∧ ¬(q x b)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2303_230348


namespace NUMINAMATH_CALUDE_segment_length_specific_case_l2303_230368

/-- A rectangle with an inscribed circle and a diagonal intersecting the circle -/
structure RectangleWithCircle where
  /-- Length of the shorter side of the rectangle -/
  short_side : ℝ
  /-- Length of the longer side of the rectangle -/
  long_side : ℝ
  /-- The circle is tangent to three sides of the rectangle -/
  circle_tangent : Bool
  /-- The diagonal intersects the circle at two points -/
  diagonal_intersects : Bool

/-- The length of the segment AB formed by the intersection of the diagonal with the circle -/
def segment_length (r : RectangleWithCircle) : ℝ :=
  sorry

/-- Theorem stating the length of AB in the specific case -/
theorem segment_length_specific_case :
  let r : RectangleWithCircle := {
    short_side := 2,
    long_side := 4,
    circle_tangent := true,
    diagonal_intersects := true
  }
  segment_length r = 4 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_segment_length_specific_case_l2303_230368


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2303_230377

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^4 + 2*x^2 + 2) % (x - 2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2303_230377


namespace NUMINAMATH_CALUDE_last_digit_base_9_of_221122211111_base_3_l2303_230361

def base_3_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

def last_digit_base_9 (n : Nat) : Nat :=
  n % 9

theorem last_digit_base_9_of_221122211111_base_3 :
  let y : Nat := base_3_to_10 [1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2]
  last_digit_base_9 y = 6 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_base_9_of_221122211111_base_3_l2303_230361


namespace NUMINAMATH_CALUDE_sphere_cylinder_volumes_l2303_230390

/-- Given a sphere with surface area 144π cm² that fits exactly inside a cylinder
    with height equal to the sphere's diameter, prove that the volume of the sphere
    is 288π cm³ and the volume of the cylinder is 432π cm³. -/
theorem sphere_cylinder_volumes (r : ℝ) (h : 4 * Real.pi * r^2 = 144 * Real.pi) :
  (4/3 : ℝ) * Real.pi * r^3 = 288 * Real.pi ∧
  Real.pi * r^2 * (2*r) = 432 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volumes_l2303_230390


namespace NUMINAMATH_CALUDE_opposite_face_is_D_l2303_230341

-- Define a cube net
structure CubeNet :=
  (faces : Finset Char)
  (is_valid : faces.card = 6)

-- Define a cube
structure Cube :=
  (faces : Finset Char)
  (is_valid : faces.card = 6)
  (opposite : Char → Char)
  (opposite_symm : ∀ x, opposite (opposite x) = x)

-- Define the folding operation
def fold (net : CubeNet) : Cube :=
  { faces := net.faces,
    is_valid := net.is_valid,
    opposite := sorry,
    opposite_symm := sorry }

-- Theorem statement
theorem opposite_face_is_D (net : CubeNet) 
  (h1 : net.faces = {'A', 'B', 'C', 'D', 'E', 'F'}) :
  (fold net).opposite 'A' = 'D' :=
sorry

end NUMINAMATH_CALUDE_opposite_face_is_D_l2303_230341


namespace NUMINAMATH_CALUDE_vip_price_is_60_l2303_230387

/-- Represents the ticket sales and pricing for a snooker tournament --/
structure SnookerTickets where
  totalTickets : ℕ
  totalRevenue : ℕ
  generalPrice : ℕ
  vipDifference : ℕ

/-- The specific ticket sales scenario for the tournament --/
def tournamentSales : SnookerTickets :=
  { totalTickets := 320
  , totalRevenue := 7500
  , generalPrice := 10
  , vipDifference := 148
  }

/-- Calculates the price of a VIP ticket --/
def vipPrice (s : SnookerTickets) : ℕ :=
  let generalTickets := (s.totalTickets + s.vipDifference) / 2
  let vipTickets := s.totalTickets - generalTickets
  (s.totalRevenue - s.generalPrice * generalTickets) / vipTickets

/-- Theorem stating that the VIP ticket price for the given scenario is $60 --/
theorem vip_price_is_60 : vipPrice tournamentSales = 60 := by
  sorry

end NUMINAMATH_CALUDE_vip_price_is_60_l2303_230387


namespace NUMINAMATH_CALUDE_chocolate_distribution_l2303_230399

/-- Calculates the number of chocolate squares each student receives when:
  * Gerald brings 7 chocolate bars
  * Each bar contains 8 squares
  * For every bar Gerald brings, the teacher brings 2 more identical ones
  * There are 24 students in class
-/
theorem chocolate_distribution (gerald_bars : Nat) (squares_per_bar : Nat) (teacher_ratio : Nat) (num_students : Nat)
    (h1 : gerald_bars = 7)
    (h2 : squares_per_bar = 8)
    (h3 : teacher_ratio = 2)
    (h4 : num_students = 24) :
    (gerald_bars + gerald_bars * teacher_ratio) * squares_per_bar / num_students = 7 := by
  sorry


end NUMINAMATH_CALUDE_chocolate_distribution_l2303_230399


namespace NUMINAMATH_CALUDE_increasing_linear_function_positive_slope_l2303_230319

/-- A linear function f(x) = mx + b -/
def LinearFunction (m b : ℝ) : ℝ → ℝ := fun x ↦ m * x + b

/-- A function is increasing if for any x₁ < x₂, f(x₁) < f(x₂) -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

theorem increasing_linear_function_positive_slope (m b : ℝ) :
  IsIncreasing (LinearFunction m b) → m > 0 := by
  sorry

end NUMINAMATH_CALUDE_increasing_linear_function_positive_slope_l2303_230319
