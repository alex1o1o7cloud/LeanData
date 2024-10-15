import Mathlib

namespace NUMINAMATH_CALUDE_no_winning_strategy_card_game_probability_l3490_349050

/-- Represents a deck of cards with red and black suits. -/
structure Deck :=
  (red : ℕ)
  (black : ℕ)

/-- Represents a strategy for playing the card game. -/
def Strategy := Deck → Bool

/-- The probability of winning given a deck and a strategy. -/
def winProbability (d : Deck) (s : Strategy) : ℚ :=
  d.red / (d.red + d.black)

/-- The theorem stating that no strategy can have a winning probability greater than 0.5. -/
theorem no_winning_strategy (d : Deck) (s : Strategy) :
  d.red = d.black → winProbability d s = 1/2 := by
  sorry

/-- The main theorem stating that for any strategy, 
    the probability of winning is always 0.5 for a standard deck. -/
theorem card_game_probability (s : Strategy) : 
  ∀ d : Deck, d.red = d.black → d.red + d.black = 52 → winProbability d s = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_no_winning_strategy_card_game_probability_l3490_349050


namespace NUMINAMATH_CALUDE_cartesian_to_polar_conversion_l3490_349026

theorem cartesian_to_polar_conversion (x y ρ θ : Real) :
  x = -1 ∧ y = Real.sqrt 3 →
  ρ = 2 ∧ θ = 2 * Real.pi / 3 →
  ρ * Real.cos θ = x ∧ ρ * Real.sin θ = y ∧ ρ^2 = x^2 + y^2 :=
by sorry

end NUMINAMATH_CALUDE_cartesian_to_polar_conversion_l3490_349026


namespace NUMINAMATH_CALUDE_dave_money_l3490_349025

theorem dave_money (dave_amount : ℝ) : 
  (2 / 3 * (3 * dave_amount - 12) = 84) → dave_amount = 46 := by
  sorry

end NUMINAMATH_CALUDE_dave_money_l3490_349025


namespace NUMINAMATH_CALUDE_max_ab_value_l3490_349081

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^a * 2^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt 2 = Real.sqrt (2^x * 2^y) → x * y ≤ a * b) → 
  a * b = 1/4 := by
sorry

end NUMINAMATH_CALUDE_max_ab_value_l3490_349081


namespace NUMINAMATH_CALUDE_max_volume_regular_pyramid_l3490_349095

/-- 
For a regular n-sided pyramid with surface area S, 
prove that the maximum volume V is given by the formula:
V = (√2 / 12) * (S^(3/2)) / √(n * tan(π/n))
-/
theorem max_volume_regular_pyramid (n : ℕ) (S : ℝ) (h₁ : n ≥ 3) (h₂ : S > 0) :
  ∃ V : ℝ, V = (Real.sqrt 2 / 12) * S^(3/2) / Real.sqrt (n * Real.tan (π / n)) ∧
    ∀ V' : ℝ, (∃ (Q h : ℝ), V' = (1/3) * Q * h ∧ 
      S = Q + n * Q / (2 * Real.cos (π / n))) → V' ≤ V := by
  sorry


end NUMINAMATH_CALUDE_max_volume_regular_pyramid_l3490_349095


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l3490_349014

theorem jason_pokemon_cards (initial_cards new_cards : ℕ) 
  (h1 : initial_cards = 676)
  (h2 : new_cards = 224) :
  initial_cards + new_cards = 900 := by
  sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l3490_349014


namespace NUMINAMATH_CALUDE_jed_cards_per_week_l3490_349047

/-- Represents the number of cards Jed has after a given number of weeks -/
def cards_after_weeks (initial_cards : ℕ) (cards_per_week : ℕ) (weeks : ℕ) : ℕ :=
  initial_cards + cards_per_week * weeks - 2 * (weeks / 2)

/-- Proves that Jed gets 6 cards per week given the conditions -/
theorem jed_cards_per_week :
  ∃ (cards_per_week : ℕ),
    cards_after_weeks 20 cards_per_week 4 = 40 ∧ cards_per_week = 6 := by
  sorry

#check jed_cards_per_week

end NUMINAMATH_CALUDE_jed_cards_per_week_l3490_349047


namespace NUMINAMATH_CALUDE_chord_distance_from_center_l3490_349096

/-- Given a circle where a chord intersects a diameter at an angle of 30° and
    divides it into segments of lengths a and b, the distance from the center
    of the circle to the chord is (1/4)|a - b|. -/
theorem chord_distance_from_center (a b : ℝ) :
  let chord_angle : ℝ := 30 * π / 180  -- 30° in radians
  let distance_to_chord : ℝ → ℝ → ℝ := λ x y => (1/4) * |x - y|
  chord_angle = 30 * π / 180 →
  distance_to_chord a b = (1/4) * |a - b| :=
by sorry

end NUMINAMATH_CALUDE_chord_distance_from_center_l3490_349096


namespace NUMINAMATH_CALUDE_star_1993_1935_l3490_349010

-- Define the operation *
def star (x y : ℤ) : ℤ := x - y

-- State the theorem
theorem star_1993_1935 : star 1993 1935 = 58 := by
  -- Assumptions
  have h1 : ∀ x : ℤ, star x x = 0 := by sorry
  have h2 : ∀ x y z : ℤ, star x (star y z) = star (star x y) z := by sorry
  
  -- Proof
  sorry

end NUMINAMATH_CALUDE_star_1993_1935_l3490_349010


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l3490_349094

theorem probability_at_least_one_woman (total : ℕ) (men women selected : ℕ) 
  (h_total : total = men + women)
  (h_men : men = 6)
  (h_women : women = 4)
  (h_selected : selected = 3) :
  1 - (Nat.choose men selected : ℚ) / (Nat.choose total selected) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l3490_349094


namespace NUMINAMATH_CALUDE_savings_calculation_l3490_349074

def num_machines : ℕ := 25
def bearings_per_machine : ℕ := 45
def regular_price : ℚ := 125/100
def sale_price : ℚ := 80/100
def discount_first_20 : ℚ := 25/100
def discount_remaining : ℚ := 35/100
def first_batch : ℕ := 20

def total_bearings : ℕ := num_machines * bearings_per_machine

def regular_total_cost : ℚ := (total_bearings : ℚ) * regular_price

def sale_cost_before_discount : ℚ := (total_bearings : ℚ) * sale_price

def first_batch_bearings : ℕ := first_batch * bearings_per_machine
def remaining_bearings : ℕ := total_bearings - first_batch_bearings

def first_batch_cost : ℚ := (first_batch_bearings : ℚ) * sale_price * (1 - discount_first_20)
def remaining_cost : ℚ := (remaining_bearings : ℚ) * sale_price * (1 - discount_remaining)

def total_discounted_cost : ℚ := first_batch_cost + remaining_cost

theorem savings_calculation : 
  regular_total_cost - total_discounted_cost = 74925/100 :=
by sorry

end NUMINAMATH_CALUDE_savings_calculation_l3490_349074


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l3490_349085

theorem opposite_of_negative_2023 : -((-2023 : ℚ)) = (2023 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l3490_349085


namespace NUMINAMATH_CALUDE_remainder_sum_mod_35_l3490_349036

theorem remainder_sum_mod_35 (f y z : ℤ) 
  (hf : f % 5 = 3) 
  (hy : y % 5 = 4) 
  (hz : z % 7 = 6) : 
  (f + y + z) % 35 = 13 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_35_l3490_349036


namespace NUMINAMATH_CALUDE_distribution_theorem_l3490_349056

-- Define the total number of employees
def total_employees : ℕ := 8

-- Define the number of departments
def num_departments : ℕ := 2

-- Define the number of English translators
def num_translators : ℕ := 2

-- Define the function to calculate the number of distribution schemes
def distribution_schemes (n : ℕ) (k : ℕ) (t : ℕ) : ℕ := 
  (Nat.choose (n - t) ((n - t) / 2)) * 2

-- Theorem statement
theorem distribution_theorem : 
  distribution_schemes total_employees num_departments num_translators = 40 := by
  sorry

end NUMINAMATH_CALUDE_distribution_theorem_l3490_349056


namespace NUMINAMATH_CALUDE_cookie_ratio_proof_l3490_349070

def cookie_problem (initial cookies_to_friend cookies_eaten cookies_left : ℕ) : Prop :=
  let cookies_after_friend := initial - cookies_to_friend
  let cookies_to_family := cookies_after_friend - cookies_eaten - cookies_left
  (2 * cookies_to_family = cookies_after_friend)

theorem cookie_ratio_proof :
  cookie_problem 19 5 2 5 := by
  sorry

end NUMINAMATH_CALUDE_cookie_ratio_proof_l3490_349070


namespace NUMINAMATH_CALUDE_shapes_can_form_both_rectangles_l3490_349002

/-- Represents a pentagon -/
structure Pentagon where
  area : ℝ

/-- Represents a triangle -/
structure Triangle where
  area : ℝ

/-- Represents a set of shapes consisting of two pentagons and a triangle -/
structure ShapeSet where
  pentagon1 : Pentagon
  pentagon2 : Pentagon
  triangle : Triangle

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Checks if a set of shapes can form a given rectangle -/
def can_form_rectangle (shapes : ShapeSet) (rect : Rectangle) : Prop :=
  shapes.pentagon1.area + shapes.pentagon2.area + shapes.triangle.area = rect.width * rect.height

/-- The main theorem stating that it's possible to have a set of shapes
    that can form both a 4x6 and a 3x8 rectangle -/
theorem shapes_can_form_both_rectangles :
  ∃ (shapes : ShapeSet),
    can_form_rectangle shapes (Rectangle.mk 4 6) ∧
    can_form_rectangle shapes (Rectangle.mk 3 8) := by
  sorry

end NUMINAMATH_CALUDE_shapes_can_form_both_rectangles_l3490_349002


namespace NUMINAMATH_CALUDE_ball_probability_l3490_349067

/-- The probability of choosing a ball that is neither red nor purple from a bag -/
theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 100)
  (h_white : white = 10)
  (h_green : green = 30)
  (h_yellow : yellow = 10)
  (h_red : red = 47)
  (h_purple : purple = 3)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ball_probability_l3490_349067


namespace NUMINAMATH_CALUDE_three_digit_sum_property_l3490_349013

theorem three_digit_sum_property : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧
  (∃ x y z : ℕ, 
    n = 100 * x + 10 * y + z ∧
    x < 10 ∧ y < 10 ∧ z < 10 ∧
    n = y + x^2 + z^3) ∧
  n = 357 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_property_l3490_349013


namespace NUMINAMATH_CALUDE_remainder_sum_powers_mod_five_l3490_349011

theorem remainder_sum_powers_mod_five :
  (9^7 + 8^8 + 7^9) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_powers_mod_five_l3490_349011


namespace NUMINAMATH_CALUDE_third_vertex_y_coordinate_l3490_349005

/-- An equilateral triangle with two vertices at (3, 4) and (13, 4), and the third vertex in the first quadrant -/
structure EquilateralTriangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  h1 : v1 = (3, 4)
  h2 : v2 = (13, 4)
  h3 : v3.1 > 0 ∧ v3.2 > 0  -- First quadrant condition
  h4 : (v1.1 - v2.1)^2 + (v1.2 - v2.2)^2 = (v1.1 - v3.1)^2 + (v1.2 - v3.2)^2
  h5 : (v1.1 - v2.1)^2 + (v1.2 - v2.2)^2 = (v2.1 - v3.1)^2 + (v2.2 - v3.2)^2

/-- The y-coordinate of the third vertex is 4 + 5√3 -/
theorem third_vertex_y_coordinate (t : EquilateralTriangle) : t.v3.2 = 4 + 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_third_vertex_y_coordinate_l3490_349005


namespace NUMINAMATH_CALUDE_markup_markdown_equivalence_l3490_349034

theorem markup_markdown_equivalence (original_price : ℝ) (markup_percentage : ℝ) (markdown_percentage : ℝ)
  (h1 : markup_percentage = 25)
  (h2 : original_price * (1 + markup_percentage / 100) * (1 - markdown_percentage / 100) = original_price) :
  markdown_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_markup_markdown_equivalence_l3490_349034


namespace NUMINAMATH_CALUDE_same_remainder_divisor_l3490_349077

theorem same_remainder_divisor : 
  ∃ (d : ℕ), d > 1 ∧ 
  (1059 % d = 1417 % d) ∧ 
  (1059 % d = 2312 % d) ∧ 
  (1417 % d = 2312 % d) ∧
  (∀ (k : ℕ), k > d → 
    (1059 % k ≠ 1417 % k) ∨ 
    (1059 % k ≠ 2312 % k) ∨ 
    (1417 % k ≠ 2312 % k)) →
  d = 179 := by
sorry

end NUMINAMATH_CALUDE_same_remainder_divisor_l3490_349077


namespace NUMINAMATH_CALUDE_cistern_filling_time_l3490_349035

theorem cistern_filling_time (empty_rate : ℝ) (combined_fill_time : ℝ) (fill_time : ℝ) : 
  empty_rate = 8 →
  combined_fill_time = 40 / 3 →
  1 / fill_time - 1 / empty_rate = 1 / combined_fill_time →
  fill_time = 5 := by
sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l3490_349035


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3490_349030

/-- The common ratio of the geometric sequence representing 0.72̄ -/
def q : ℚ := 1 / 100

/-- The first term of the geometric sequence representing 0.72̄ -/
def a₁ : ℚ := 72 / 100

/-- The sum of the infinite geometric series representing 0.72̄ -/
def S : ℚ := a₁ / (1 - q)

/-- The repeating decimal 0.72̄ as a rational number -/
def repeating_decimal : ℚ := 8 / 11

theorem repeating_decimal_equals_fraction : S = repeating_decimal := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3490_349030


namespace NUMINAMATH_CALUDE_l_plate_four_equal_parts_l3490_349071

/-- Represents an L-shaped plate -/
structure LPlate where
  width : ℝ
  height : ℝ
  isRightAngled : Bool

/-- Represents a cut on the L-shaped plate -/
inductive Cut
  | Vertical : ℝ → Cut  -- x-coordinate of the vertical cut
  | Horizontal : ℝ → Cut  -- y-coordinate of the horizontal cut

/-- Checks if a set of cuts divides an L-shaped plate into four equal parts -/
def dividesIntoFourEqualParts (plate : LPlate) (cuts : List Cut) : Prop :=
  sorry

/-- Theorem stating that an L-shaped plate can be divided into four equal L-shaped pieces -/
theorem l_plate_four_equal_parts (plate : LPlate) :
  ∃ (cuts : List Cut), dividesIntoFourEqualParts plate cuts :=
sorry

end NUMINAMATH_CALUDE_l_plate_four_equal_parts_l3490_349071


namespace NUMINAMATH_CALUDE_system_solution_unique_l3490_349092

theorem system_solution_unique :
  ∃! (x y : ℝ), x^2 + y * Real.sqrt (x * y) = 336 ∧ y^2 + x * Real.sqrt (x * y) = 112 ∧ x = 18 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3490_349092


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3490_349040

theorem diophantine_equation_solution (x y : ℤ) :
  7 * x - 3 * y = 2 ↔ ∃ k : ℤ, x = 3 * k + 2 ∧ y = 7 * k + 4 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3490_349040


namespace NUMINAMATH_CALUDE_bicycle_ride_time_l3490_349078

/-- Proves the total time Hyeonil rode the bicycle given the conditions -/
theorem bicycle_ride_time (speed : ℝ) (initial_time : ℝ) (additional_distance : ℝ)
  (h1 : speed = 4.25)
  (h2 : initial_time = 60)
  (h3 : additional_distance = 29.75) :
  initial_time + additional_distance / speed = 67 := by
  sorry

#check bicycle_ride_time

end NUMINAMATH_CALUDE_bicycle_ride_time_l3490_349078


namespace NUMINAMATH_CALUDE_haley_recycling_cans_l3490_349045

theorem haley_recycling_cans : ∃ (c : ℕ), c = 9 ∧ c - 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_haley_recycling_cans_l3490_349045


namespace NUMINAMATH_CALUDE_vector_decomposition_l3490_349057

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![8, 9, 4]
def p : Fin 3 → ℝ := ![1, 0, 1]
def q : Fin 3 → ℝ := ![0, -2, 1]
def r : Fin 3 → ℝ := ![1, 3, 0]

/-- Theorem: Vector x can be decomposed as 7p - 3q + r -/
theorem vector_decomposition :
  x = fun i => 7 * p i - 3 * q i + r i :=
by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l3490_349057


namespace NUMINAMATH_CALUDE_discount_calculation_l3490_349054

/-- Given the original cost of plants and the amount actually spent, prove that the discount received is $399.00 -/
theorem discount_calculation (original_cost spent_amount : ℚ) 
  (h1 : original_cost = 467) 
  (h2 : spent_amount = 68) : 
  original_cost - spent_amount = 399 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l3490_349054


namespace NUMINAMATH_CALUDE_table_count_l3490_349038

theorem table_count (num_books : ℕ) (h : num_books = 100000) :
  ∃ (num_tables : ℕ),
    (num_tables : ℚ) * (2 / 5 * num_tables) = num_books ∧
    num_tables = 500 := by
  sorry

end NUMINAMATH_CALUDE_table_count_l3490_349038


namespace NUMINAMATH_CALUDE_b1f_hex_to_dec_l3490_349001

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'B' => 11
  | '1' => 1
  | 'F' => 15
  | _ => 0  -- Default case, should not be reached for this problem

/-- Converts a hexadecimal number represented as a string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.foldl (fun acc d => 16 * acc + hex_to_dec d) 0

theorem b1f_hex_to_dec :
  hex_string_to_dec "B1F" = 2847 := by
  sorry


end NUMINAMATH_CALUDE_b1f_hex_to_dec_l3490_349001


namespace NUMINAMATH_CALUDE_sequence_terms_coprime_l3490_349021

def sequence_a : ℕ → ℕ
  | 0 => 2
  | n + 1 => (sequence_a n)^2 - sequence_a n + 1

theorem sequence_terms_coprime (m n : ℕ) (h : m ≠ n) : 
  Nat.gcd (sequence_a m) (sequence_a n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_terms_coprime_l3490_349021


namespace NUMINAMATH_CALUDE_salon_customers_l3490_349053

/-- Represents the daily operations of a hair salon -/
structure Salon where
  total_cans : ℕ
  extra_cans : ℕ
  cans_per_customer : ℕ

/-- Calculates the number of customers given a salon's daily operations -/
def customers (s : Salon) : ℕ :=
  (s.total_cans - s.extra_cans) / s.cans_per_customer

/-- Theorem stating that a salon with the given parameters has 14 customers per day -/
theorem salon_customers :
  let s : Salon := {
    total_cans := 33,
    extra_cans := 5,
    cans_per_customer := 2
  }
  customers s = 14 := by
  sorry

end NUMINAMATH_CALUDE_salon_customers_l3490_349053


namespace NUMINAMATH_CALUDE_lindsey_final_balance_l3490_349042

def september_savings : ℕ := 50
def october_savings : ℕ := 37
def november_savings : ℕ := 11
def mom_bonus_threshold : ℕ := 75
def mom_bonus : ℕ := 25
def video_game_cost : ℕ := 87

def total_savings : ℕ := september_savings + october_savings + november_savings

def final_balance : ℕ :=
  if total_savings > mom_bonus_threshold
  then total_savings + mom_bonus - video_game_cost
  else total_savings - video_game_cost

theorem lindsey_final_balance : final_balance = 36 := by
  sorry

end NUMINAMATH_CALUDE_lindsey_final_balance_l3490_349042


namespace NUMINAMATH_CALUDE_bee_travel_distance_l3490_349007

theorem bee_travel_distance (initial_distance : ℝ) (speed_A : ℝ) (speed_B : ℝ) (speed_bee : ℝ)
  (h1 : initial_distance = 120)
  (h2 : speed_A = 30)
  (h3 : speed_B = 10)
  (h4 : speed_bee = 60) :
  let relative_speed := speed_A + speed_B
  let meeting_time := initial_distance / relative_speed
  speed_bee * meeting_time = 180 := by
sorry

end NUMINAMATH_CALUDE_bee_travel_distance_l3490_349007


namespace NUMINAMATH_CALUDE_checkerboard_chips_l3490_349075

/-- The total number of chips on an n × n checkerboard where each square (i, j) has |i - j| chips -/
def total_chips (n : ℕ) : ℕ := n * (n + 1) * (n - 1) / 3

/-- Theorem stating that if the total number of chips is 2660, then n = 20 -/
theorem checkerboard_chips (n : ℕ) : total_chips n = 2660 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_chips_l3490_349075


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3490_349076

theorem completing_square_equivalence (x : ℝ) :
  x^2 - 4*x + 3 = 0 ↔ (x - 2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3490_349076


namespace NUMINAMATH_CALUDE_t_value_on_line_l3490_349064

/-- A straight line passing through points (1, 7), (3, 13), (5, 19), and (28, t) -/
def straightLine (t : ℝ) : Prop :=
  ∃ (m c : ℝ),
    (7 = m * 1 + c) ∧
    (13 = m * 3 + c) ∧
    (19 = m * 5 + c) ∧
    (t = m * 28 + c)

/-- Theorem stating that t = 88 for the given straight line -/
theorem t_value_on_line : straightLine 88 := by
  sorry

end NUMINAMATH_CALUDE_t_value_on_line_l3490_349064


namespace NUMINAMATH_CALUDE_ruth_math_class_hours_l3490_349041

/-- Calculates the number of hours spent in math class per week for a student with given school schedule and math class percentage. -/
def math_class_hours_per_week (hours_per_day : ℕ) (days_per_week : ℕ) (math_class_percentage : ℚ) : ℚ :=
  (hours_per_day * days_per_week : ℚ) * math_class_percentage

/-- Theorem stating that a student who attends school for 8 hours a day, 5 days a week, and spends 25% of their school time in math class, spends 10 hours per week in math class. -/
theorem ruth_math_class_hours :
  math_class_hours_per_week 8 5 (1/4) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ruth_math_class_hours_l3490_349041


namespace NUMINAMATH_CALUDE_child_height_at_last_visit_l3490_349052

/-- Given a child's current height and growth since last visit, 
    prove the height at the last visit. -/
theorem child_height_at_last_visit 
  (current_height : ℝ) 
  (growth_since_last_visit : ℝ) 
  (h1 : current_height = 41.5) 
  (h2 : growth_since_last_visit = 3.0) : 
  current_height - growth_since_last_visit = 38.5 := by
sorry

end NUMINAMATH_CALUDE_child_height_at_last_visit_l3490_349052


namespace NUMINAMATH_CALUDE_christmas_tree_lights_l3490_349006

theorem christmas_tree_lights (red : ℕ) (yellow : ℕ) (blue : ℕ)
  (h_red : red = 26)
  (h_yellow : yellow = 37)
  (h_blue : blue = 32) :
  red + yellow + blue = 95 := by
  sorry

end NUMINAMATH_CALUDE_christmas_tree_lights_l3490_349006


namespace NUMINAMATH_CALUDE_quadratic_root_is_one_l3490_349015

/-- 
Given a quadratic function f(x) = x^2 + ax + b, where:
- The graph of f intersects the y-axis at (0, b)
- The graph of f intersects the x-axis at (b, 0)
- b ≠ 0

Prove that the other root of f(x) = 0 is equal to 1.
-/
theorem quadratic_root_is_one (a b : ℝ) (hb : b ≠ 0) : 
  let f : ℝ → ℝ := fun x ↦ x^2 + a*x + b
  (f 0 = b) → (f b = 0) → ∃ c, c ≠ b ∧ f c = 0 ∧ c = 1 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_root_is_one_l3490_349015


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3490_349049

theorem min_value_expression (a c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
  a^2 + c^2 + 1/a^2 + c/a + 1/c^2 ≥ Real.sqrt 15 :=
sorry

theorem equality_condition (a c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
  ∃ a c, a^2 + c^2 + 1/a^2 + c/a + 1/c^2 = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3490_349049


namespace NUMINAMATH_CALUDE_percentage_increase_in_students_l3490_349022

theorem percentage_increase_in_students (students_this_year students_last_year : ℕ) 
  (h1 : students_this_year = 960)
  (h2 : students_last_year = 800) :
  (students_this_year - students_last_year) / students_last_year * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_in_students_l3490_349022


namespace NUMINAMATH_CALUDE_halloween_candy_eaten_l3490_349062

/-- The number of candy pieces eaten on Halloween night -/
def candy_eaten (debby_initial : ℕ) (sister_initial : ℕ) (remaining : ℕ) : ℕ :=
  debby_initial + sister_initial - remaining

/-- Theorem stating the number of candy pieces eaten on Halloween night -/
theorem halloween_candy_eaten :
  candy_eaten 32 42 39 = 35 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_eaten_l3490_349062


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3490_349090

theorem cube_root_equation_solution (x : ℝ) : 
  (15 * x + (15 * x + 8) ^ (1/3)) ^ (1/3) = 8 → x = 168/5 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3490_349090


namespace NUMINAMATH_CALUDE_first_triangular_covering_all_remainders_triangular_22_is_253_l3490_349088

/-- Triangular number function -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Function to check if a number covers all remainders modulo 10 -/
def covers_all_remainders (n : ℕ) : Prop :=
  ∀ r : Fin 10, ∃ k : ℕ, k ≤ n ∧ triangular_number k % 10 = r

/-- Main theorem: 22 is the smallest n for which triangular_number n covers all remainders modulo 10 -/
theorem first_triangular_covering_all_remainders :
  (covers_all_remainders 22 ∧ ∀ m < 22, ¬ covers_all_remainders m) :=
sorry

/-- Corollary: The 22nd triangular number is 253 -/
theorem triangular_22_is_253 : triangular_number 22 = 253 :=
sorry

end NUMINAMATH_CALUDE_first_triangular_covering_all_remainders_triangular_22_is_253_l3490_349088


namespace NUMINAMATH_CALUDE_equation_solution_l3490_349000

theorem equation_solution :
  ∀ x : ℚ, (1 / 4 : ℚ) + 7 / x = 13 / x + (1 / 9 : ℚ) → x = 216 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3490_349000


namespace NUMINAMATH_CALUDE_problem_solution_l3490_349046

theorem problem_solution : 3^(0^(2^3)) + ((3^1)^0)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3490_349046


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3490_349093

theorem possible_values_of_a (a b c : ℤ) :
  (∀ x, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) →
  (a = 3 ∨ a = 7) := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3490_349093


namespace NUMINAMATH_CALUDE_students_interested_in_both_l3490_349073

theorem students_interested_in_both (total : ℕ) (sports : ℕ) (entertainment : ℕ) (neither : ℕ) :
  total = 1400 →
  sports = 1250 →
  entertainment = 952 →
  neither = 60 →
  ∃ x : ℕ, x = 862 ∧
    total = neither + x + (sports - x) + (entertainment - x) :=
by sorry

end NUMINAMATH_CALUDE_students_interested_in_both_l3490_349073


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3490_349059

-- Define the function f
def f (x : ℝ) : ℝ := x * abs (x - 2)

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) : 
  f (2 - Real.log (x + 1)) > f 3 ↔ -1 < x ∧ x < Real.exp (-1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3490_349059


namespace NUMINAMATH_CALUDE_cylinder_surface_area_minimized_l3490_349051

/-- Theorem: For a cylinder with fixed volume, the surface area is minimized when the height is twice the radius. -/
theorem cylinder_surface_area_minimized (R H V : ℝ) (h_positive : R > 0 ∧ H > 0 ∧ V > 0) 
  (h_volume : π * R^2 * H = V / 2) :
  let A := 2 * π * R^2 + 2 * π * R * H
  ∀ R' H' : ℝ, R' > 0 → H' > 0 → π * R'^2 * H' = V / 2 → 
    2 * π * R'^2 + 2 * π * R' * H' ≥ A → H / R = 2 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_minimized_l3490_349051


namespace NUMINAMATH_CALUDE_collinear_points_d_values_l3490_349031

-- Define the points
def point_a (a : ℝ) : ℝ × ℝ × ℝ := (1, 0, a)
def point_b (b : ℝ) : ℝ × ℝ × ℝ := (b, 1, 0)
def point_c (c : ℝ) : ℝ × ℝ × ℝ := (0, c, 1)
def point_d (d : ℝ) : ℝ × ℝ × ℝ := (4*d, 4*d, -2*d)

-- Define collinearity
def collinear (p q r : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), q - p = t • (r - p)

theorem collinear_points_d_values (a b c d : ℝ) :
  collinear (point_a a) (point_b b) (point_c c) ∧
  collinear (point_a a) (point_b b) (point_d d) →
  d = 1 ∨ d = 1/4 :=
sorry

end NUMINAMATH_CALUDE_collinear_points_d_values_l3490_349031


namespace NUMINAMATH_CALUDE_walking_speeds_l3490_349012

/-- The speeds of two people walking on a highway -/
theorem walking_speeds (x y : ℝ) : 
  (30 * x - 30 * y = 300) →  -- If both walk eastward for 30 minutes, A catches up with B
  (2 * x + 2 * y = 300) →    -- If they walk towards each other, they meet after 2 minutes
  (x = 80 ∧ y = 70) :=        -- Then A's speed is 80 m/min and B's speed is 70 m/min
by
  sorry

end NUMINAMATH_CALUDE_walking_speeds_l3490_349012


namespace NUMINAMATH_CALUDE_hyperbola_satisfies_conditions_l3490_349068

/-- A hyperbola with the equation 4x² - 9y² = -32 -/
def hyperbola (x y : ℝ) : Prop := 4 * x^2 - 9 * y^2 = -32

/-- The asymptotes of the hyperbola -/
def asymptotes (x y : ℝ) : Prop := (2 * x + 3 * y = 0) ∨ (2 * x - 3 * y = 0)

theorem hyperbola_satisfies_conditions :
  (∀ x y : ℝ, asymptotes x y ↔ (4 * x^2 - 9 * y^2 = 0)) ∧
  hyperbola 1 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_satisfies_conditions_l3490_349068


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l3490_349039

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/2)^2 + (1/3)^2 = ((1/4)^2 + (1/5)^2) * (13*x)/(41*y)) : 
  Real.sqrt x / Real.sqrt y = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l3490_349039


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l3490_349023

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- The original number of sides of the polygon -/
def original_sides : ℕ := 7

/-- The number of sides after doubling -/
def doubled_sides : ℕ := 2 * original_sides

theorem polygon_interior_angles_sum : 
  sum_interior_angles doubled_sides = 2160 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l3490_349023


namespace NUMINAMATH_CALUDE_stevens_peaches_l3490_349020

-- Define the number of peaches Jake has
def jakes_peaches : ℕ := 7

-- Define the difference in peaches between Steven and Jake
def peach_difference : ℕ := 6

-- Theorem stating that Steven has 13 peaches
theorem stevens_peaches : 
  jakes_peaches + peach_difference = 13 := by sorry

end NUMINAMATH_CALUDE_stevens_peaches_l3490_349020


namespace NUMINAMATH_CALUDE_root_sum_product_l3490_349061

theorem root_sum_product (p q r : ℂ) : 
  (5 * p ^ 3 - 10 * p ^ 2 + 17 * p - 7 = 0) →
  (5 * q ^ 3 - 10 * q ^ 2 + 17 * q - 7 = 0) →
  (5 * r ^ 3 - 10 * r ^ 2 + 17 * r - 7 = 0) →
  p * q + p * r + q * r = 17 / 5 := by
sorry

end NUMINAMATH_CALUDE_root_sum_product_l3490_349061


namespace NUMINAMATH_CALUDE_interest_difference_theorem_l3490_349017

theorem interest_difference_theorem (P : ℝ) : 
  let r : ℝ := 5 / 100  -- 5% interest rate
  let t : ℝ := 2        -- 2 years
  let simple_interest := P * r * t
  let compound_interest := P * ((1 + r) ^ t - 1)
  compound_interest - simple_interest = 20 → P = 8000 := by
sorry

end NUMINAMATH_CALUDE_interest_difference_theorem_l3490_349017


namespace NUMINAMATH_CALUDE_average_rounds_is_four_l3490_349019

/-- Represents the distribution of golfers and rounds played --/
structure GolfData :=
  (rounds : Fin 6 → ℕ)
  (golfers : Fin 6 → ℕ)

/-- Calculates the average number of rounds played, rounded to the nearest whole number --/
def averageRoundsRounded (data : GolfData) : ℕ :=
  let totalRounds := (Finset.range 6).sum (λ i => (data.rounds i.succ) * (data.golfers i.succ))
  let totalGolfers := (Finset.range 6).sum (λ i => data.golfers i.succ)
  (totalRounds + totalGolfers / 2) / totalGolfers

/-- The given golf data --/
def givenData : GolfData :=
  { rounds := λ i => i,
    golfers := λ i => match i with
      | 1 => 6
      | 2 => 3
      | 3 => 2
      | 4 => 4
      | 5 => 6
      | 6 => 4 }

theorem average_rounds_is_four :
  averageRoundsRounded givenData = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_rounds_is_four_l3490_349019


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3490_349089

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and right focus F(c, 0),
    if point P on the hyperbola satisfies |FM| = 2|FP| where M is the intersection of the circle
    centered at F with radius 2c and the positive y-axis, then the eccentricity of the hyperbola
    is √3 + 1. -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  let F : ℝ × ℝ := (c, 0)
  let M : ℝ × ℝ := (0, Real.sqrt 3 * c)
  let P : ℝ × ℝ := (c / 2, Real.sqrt 3 / 2 * c)
  (P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1) →  -- P is on the hyperbola
  (Real.sqrt ((M.1 - F.1) ^ 2 + (M.2 - F.2) ^ 2) = 2 * Real.sqrt ((P.1 - F.1) ^ 2 + (P.2 - F.2) ^ 2)) →  -- |FM| = 2|FP|
  (c ^ 2 / a ^ 2 - b ^ 2 / a ^ 2 = 1) →  -- Relation between a, b, and c for a hyperbola
  Real.sqrt (c ^ 2 / a ^ 2) = Real.sqrt 3 + 1  -- Eccentricity is √3 + 1
:= by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3490_349089


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3490_349048

/-- An arithmetic sequence with a₃ = 10 and a₉ = 28 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a 3 = 10 ∧ a 9 = 28

/-- The 12th term of the arithmetic sequence is 37 -/
theorem arithmetic_sequence_12th_term (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 12 = 37 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3490_349048


namespace NUMINAMATH_CALUDE_complex_absolute_value_l3490_349008

open Complex

theorem complex_absolute_value : ∀ (i : ℂ), i * i = -1 → Complex.abs (2 * i * (1 - 2 * i)) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l3490_349008


namespace NUMINAMATH_CALUDE_second_derivative_implies_m_l3490_349086

/-- Given a function f(x) = 2/x, prove that if its second derivative at m is -1/2, then m = -2 -/
theorem second_derivative_implies_m (f : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = 2 / x) →
  (deriv^[2] f m = -1/2) →
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_second_derivative_implies_m_l3490_349086


namespace NUMINAMATH_CALUDE_student_count_l3490_349099

theorem student_count (avg_age : ℝ) (teacher_age : ℝ) (new_avg : ℝ) :
  avg_age = 20 →
  teacher_age = 40 →
  new_avg = avg_age + 1 →
  (∃ n : ℕ, n * avg_age + teacher_age = (n + 1) * new_avg ∧ n = 19) :=
by sorry

end NUMINAMATH_CALUDE_student_count_l3490_349099


namespace NUMINAMATH_CALUDE_tetrahedron_planes_intersection_l3490_349032

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  normal : Point3D
  point : Point3D

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- The circumcenter of a triangle -/
def circumcenter (a b c : Point3D) : Point3D := sorry

/-- The center of the circumsphere of a tetrahedron -/
def circumsphere_center (t : Tetrahedron) : Point3D := sorry

/-- A plane passing through a point and perpendicular to a line -/
def perpendicular_plane (point line_start line_end : Point3D) : Plane3D := sorry

/-- Check if a point lies on a plane -/
def point_on_plane (point : Point3D) (plane : Plane3D) : Prop := sorry

/-- Check if a tetrahedron is regular -/
def is_regular (t : Tetrahedron) : Prop := sorry

/-- The main theorem -/
theorem tetrahedron_planes_intersection
  (t : Tetrahedron)
  (A' : Point3D) (B' : Point3D) (C' : Point3D) (D' : Point3D)
  (h_A' : A' = circumcenter t.B t.C t.D)
  (h_B' : B' = circumcenter t.C t.D t.A)
  (h_C' : C' = circumcenter t.D t.A t.B)
  (h_D' : D' = circumcenter t.A t.B t.C)
  (P_A : Plane3D) (P_B : Plane3D) (P_C : Plane3D) (P_D : Plane3D)
  (h_P_A : P_A = perpendicular_plane t.A C' D')
  (h_P_B : P_B = perpendicular_plane t.B D' A')
  (h_P_C : P_C = perpendicular_plane t.C A' B')
  (h_P_D : P_D = perpendicular_plane t.D B' C')
  (P : Point3D)
  (h_P : P = circumsphere_center t) :
  ∃ (I : Point3D),
    point_on_plane I P_A ∧
    point_on_plane I P_B ∧
    point_on_plane I P_C ∧
    point_on_plane I P_D ∧
    (I = P ↔ is_regular t) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_planes_intersection_l3490_349032


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l3490_349066

def team_size : ℕ := 18
def lineup_size : ℕ := 8
def non_pg_players : ℕ := lineup_size - 1

theorem basketball_lineup_combinations :
  (team_size : ℕ) * (Nat.choose (team_size - 1) non_pg_players) = 349864 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l3490_349066


namespace NUMINAMATH_CALUDE_target_hit_probability_l3490_349003

theorem target_hit_probability (p_a p_b : ℝ) (h_a : p_a = 0.9) (h_b : p_b = 0.8) 
  (h_independent : True) : 1 - (1 - p_a) * (1 - p_b) = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l3490_349003


namespace NUMINAMATH_CALUDE_sum_of_squares_in_sequence_l3490_349037

/-- A sequence with the property that a_{2n-1} = a_{n-1}^2 + a_n^2 for all n -/
def phi_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (2*n - 1) = (a (n-1))^2 + (a n)^2

theorem sum_of_squares_in_sequence (a : ℕ → ℝ) (h : phi_sequence a) :
  ∀ n : ℕ, ∃ m : ℕ, a m = (a (n-1))^2 + (a n)^2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_squares_in_sequence_l3490_349037


namespace NUMINAMATH_CALUDE_min_value_sum_ratios_l3490_349018

theorem min_value_sum_ratios (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) / a + (a + b + c) / b + (a + b + c) / c ≥ 9 ∧
  ((a + b + c) / a + (a + b + c) / b + (a + b + c) / c = 9 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_ratios_l3490_349018


namespace NUMINAMATH_CALUDE_stair_climbing_time_l3490_349024

theorem stair_climbing_time (a : ℕ) (d : ℕ) (n : ℕ) :
  a = 15 → d = 10 → n = 4 →
  (n : ℝ) / 2 * (2 * a + (n - 1) * d) = 120 := by
  sorry

end NUMINAMATH_CALUDE_stair_climbing_time_l3490_349024


namespace NUMINAMATH_CALUDE_constant_term_proof_l3490_349043

theorem constant_term_proof (x y z : ℤ) (k : ℤ) : 
  x = 20 → 
  4 * x + y + z = k → 
  2 * x - y - z = 40 → 
  3 * x + y - z = 20 → 
  k = 80 := by
sorry

end NUMINAMATH_CALUDE_constant_term_proof_l3490_349043


namespace NUMINAMATH_CALUDE_valid_fractions_are_complete_l3490_349044

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_fraction_in_range (n d : ℕ) : Prop :=
  22 / 3 < n / d ∧ n / d < 15 / 2

def is_valid_fraction (n d : ℕ) : Prop :=
  is_two_digit n ∧ is_two_digit d ∧ is_fraction_in_range n d ∧ Nat.gcd n d = 1

def valid_fractions : Set (ℕ × ℕ) :=
  {(81, 11), (82, 11), (89, 12), (96, 13), (97, 13)}

theorem valid_fractions_are_complete :
  ∀ (n d : ℕ), is_valid_fraction n d ↔ (n, d) ∈ valid_fractions := by sorry

end NUMINAMATH_CALUDE_valid_fractions_are_complete_l3490_349044


namespace NUMINAMATH_CALUDE_f_zero_is_zero_l3490_349079

-- Define the function f
variable (f : ℝ → ℝ)

-- State the condition
axiom functional_equation : ∀ x y : ℝ, f (x + y) = f x + f y

-- Theorem to prove
theorem f_zero_is_zero : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_is_zero_l3490_349079


namespace NUMINAMATH_CALUDE_incorrect_spelling_probability_incorrect_spelling_probability_is_59_60_l3490_349069

/-- The probability of spelling "theer" incorrectly -/
theorem incorrect_spelling_probability : ℚ :=
  let total_letters : ℕ := 5
  let repeated_letter : ℕ := 2
  let distinct_letters : ℕ := 3
  let total_arrangements : ℕ := (Nat.choose total_letters repeated_letter) * (Nat.factorial distinct_letters)
  let correct_arrangements : ℕ := 1
  (total_arrangements - correct_arrangements : ℚ) / total_arrangements

/-- Proof that the probability of spelling "theer" incorrectly is 59/60 -/
theorem incorrect_spelling_probability_is_59_60 : 
  incorrect_spelling_probability = 59 / 60 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_spelling_probability_incorrect_spelling_probability_is_59_60_l3490_349069


namespace NUMINAMATH_CALUDE_region_upper_left_l3490_349087

def line (x y : ℝ) : ℝ := 3 * x - 2 * y - 6

theorem region_upper_left :
  ∀ (x y : ℝ), line x y < 0 →
  ∃ (x' y' : ℝ), x' > x ∧ y' < y ∧ line x' y' = 0 :=
by sorry

end NUMINAMATH_CALUDE_region_upper_left_l3490_349087


namespace NUMINAMATH_CALUDE_perpendicular_from_perpendicular_and_parallel_perpendicular_from_parallel_planes_l3490_349098

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem 1
theorem perpendicular_from_perpendicular_and_parallel
  (m n : Line) (α : Plane)
  (h1 : perpendicular_plane m α)
  (h2 : parallel_plane n α) :
  perpendicular m n :=
sorry

-- Theorem 2
theorem perpendicular_from_parallel_planes
  (m : Line) (α β γ : Plane)
  (h1 : parallel_planes α β)
  (h2 : parallel_planes β γ)
  (h3 : perpendicular_plane m α) :
  perpendicular_plane m γ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_from_perpendicular_and_parallel_perpendicular_from_parallel_planes_l3490_349098


namespace NUMINAMATH_CALUDE_exists_n_composite_power_of_two_plus_fifteen_l3490_349091

theorem exists_n_composite_power_of_two_plus_fifteen :
  ∃ n : ℕ, ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 2^n + 15 = a * b :=
by sorry

end NUMINAMATH_CALUDE_exists_n_composite_power_of_two_plus_fifteen_l3490_349091


namespace NUMINAMATH_CALUDE_percy_swimming_hours_l3490_349004

/-- Percy's daily swimming hours on weekdays -/
def weekday_hours : ℕ := 2

/-- Number of weekdays Percy swims per week -/
def weekdays_per_week : ℕ := 5

/-- Percy's weekend swimming hours -/
def weekend_hours : ℕ := 3

/-- Number of weeks -/
def num_weeks : ℕ := 4

/-- Total swimming hours over the given number of weeks -/
def total_swimming_hours : ℕ := 
  num_weeks * (weekday_hours * weekdays_per_week + weekend_hours)

theorem percy_swimming_hours : total_swimming_hours = 52 := by
  sorry

end NUMINAMATH_CALUDE_percy_swimming_hours_l3490_349004


namespace NUMINAMATH_CALUDE_cassidy_grounding_l3490_349082

/-- The number of days Cassidy is grounded for lying about her report card -/
def base_grounding : ℕ := 14

/-- The number of extra days Cassidy is grounded for each grade below a B -/
def extra_days_per_grade : ℕ := 3

/-- The number of grades Cassidy got below a B -/
def grades_below_b : ℕ := 4

/-- The total number of days Cassidy is grounded -/
def total_grounding : ℕ := base_grounding + extra_days_per_grade * grades_below_b

theorem cassidy_grounding :
  total_grounding = 26 := by
  sorry

end NUMINAMATH_CALUDE_cassidy_grounding_l3490_349082


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l3490_349097

theorem chess_tournament_participants (x : ℕ) : 
  (∃ y : ℕ, 2 * x * y + 16 = (x + 2) * (x + 1)) ↔ (x = 7 ∨ x = 14) :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l3490_349097


namespace NUMINAMATH_CALUDE_unique_solution_l3490_349083

/-- Represents a 3-digit number with distinct digits -/
structure ThreeDigitNumber where
  f : Nat
  o : Nat
  g : Nat
  h_distinct : f ≠ o ∧ f ≠ g ∧ o ≠ g
  h_valid : f ≠ 0 ∧ f < 10 ∧ o < 10 ∧ g < 10

def value (n : ThreeDigitNumber) : Nat :=
  100 * n.f + 10 * n.o + n.g

theorem unique_solution (n : ThreeDigitNumber) :
  value n * (n.f + n.o + n.g) = value n →
  n.f = 1 ∧ n.o = 0 ∧ n.g = 0 ∧ n.f + n.o + n.g = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3490_349083


namespace NUMINAMATH_CALUDE_kishore_savings_percentage_l3490_349065

def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 6100
def savings : ℕ := 2400

def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous
def total_salary : ℕ := total_expenses + savings

theorem kishore_savings_percentage :
  (savings : ℚ) / (total_salary : ℚ) = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_kishore_savings_percentage_l3490_349065


namespace NUMINAMATH_CALUDE_external_tangent_intercept_l3490_349058

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Calculates the common external tangent line with positive slope for two circles -/
def commonExternalTangent (c1 c2 : Circle) : Line :=
  sorry

theorem external_tangent_intercept :
  let c1 : Circle := { center := (2, 4), radius := 5 }
  let c2 : Circle := { center := (14, 9), radius := 10 }
  let tangent := commonExternalTangent c1 c2
  tangent.slope > 0 → tangent.intercept = 912 / 119 :=
sorry

end NUMINAMATH_CALUDE_external_tangent_intercept_l3490_349058


namespace NUMINAMATH_CALUDE_cone_prism_volume_ratio_l3490_349033

/-- The ratio of the volume of a right circular cone inscribed in a right rectangular prism -/
theorem cone_prism_volume_ratio (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  (1 / 3 * π * r^2 * h) / (6 * r^2 * h) = π / 18 := by
  sorry

#check cone_prism_volume_ratio

end NUMINAMATH_CALUDE_cone_prism_volume_ratio_l3490_349033


namespace NUMINAMATH_CALUDE_equation_solution_l3490_349060

theorem equation_solution : ∃ Z : ℤ, 80 - (5 - (Z + 2 * (7 - 8 - 5))) = 89 ∧ Z = 26 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3490_349060


namespace NUMINAMATH_CALUDE_pinedale_bus_distance_l3490_349029

theorem pinedale_bus_distance (average_speed : ℝ) (stop_interval : ℝ) (num_stops : ℕ) 
  (h1 : average_speed = 60) 
  (h2 : stop_interval = 5 / 60) 
  (h3 : num_stops = 8) : 
  average_speed * (stop_interval * num_stops) = 40 := by
  sorry

end NUMINAMATH_CALUDE_pinedale_bus_distance_l3490_349029


namespace NUMINAMATH_CALUDE_sin_90_degrees_l3490_349072

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l3490_349072


namespace NUMINAMATH_CALUDE_bakery_outdoor_tables_l3490_349027

/-- Given a bakery setup with indoor and outdoor tables, prove the number of outdoor tables. -/
theorem bakery_outdoor_tables
  (indoor_tables : ℕ)
  (indoor_chairs_per_table : ℕ)
  (outdoor_chairs_per_table : ℕ)
  (total_chairs : ℕ)
  (h1 : indoor_tables = 8)
  (h2 : indoor_chairs_per_table = 3)
  (h3 : outdoor_chairs_per_table = 3)
  (h4 : total_chairs = 60) :
  (total_chairs - indoor_tables * indoor_chairs_per_table) / outdoor_chairs_per_table = 12 := by
  sorry

end NUMINAMATH_CALUDE_bakery_outdoor_tables_l3490_349027


namespace NUMINAMATH_CALUDE_container_count_l3490_349016

theorem container_count (x y : ℕ) : 
  27 * x = 65 * y + 34 → 
  y ≤ 44 → 
  x + y = 66 :=
by sorry

end NUMINAMATH_CALUDE_container_count_l3490_349016


namespace NUMINAMATH_CALUDE_rectangle_area_l3490_349009

theorem rectangle_area (x y : ℕ) : 
  1 ≤ x ∧ x < 10 ∧ 1 ≤ y ∧ y < 10 →
  ∃ n : ℕ, (1100 * x + 11 * y) = n^2 →
  x * y = 28 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3490_349009


namespace NUMINAMATH_CALUDE_bookstore_shoe_store_sales_coincidence_l3490_349055

def is_multiple_of_5 (n : ℕ) : Prop := ∃ k, n = 5 * k

def shoe_store_sale_day (n : ℕ) : Prop := ∃ k, n = 3 + 6 * k

def july_day (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 31

theorem bookstore_shoe_store_sales_coincidence :
  (∃! d : ℕ, july_day d ∧ is_multiple_of_5 d ∧ shoe_store_sale_day d) := by
  sorry

end NUMINAMATH_CALUDE_bookstore_shoe_store_sales_coincidence_l3490_349055


namespace NUMINAMATH_CALUDE_progression_check_l3490_349028

theorem progression_check (a b c : ℝ) (ha : a = 2) (hb : b = Real.sqrt 6) (hc : c = 4.5) :
  (∃ (r m : ℝ), (b / a) ^ r = (c / a) ^ m) ∧
  ¬(∃ (d : ℝ), b - a = c - b) :=
by sorry

end NUMINAMATH_CALUDE_progression_check_l3490_349028


namespace NUMINAMATH_CALUDE_set_operations_and_complements_l3490_349063

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 2}

-- Define the theorem
theorem set_operations_and_complements :
  (A ∩ B = {x | -1 ≤ x ∧ x < 2}) ∧
  (A ∪ B = {x | -2 ≤ x ∧ x ≤ 3}) ∧
  ((Uᶜ ∪ (A ∩ B)) = {x | x < -1 ∨ 2 ≤ x}) ∧
  ((Uᶜ ∪ (A ∪ B)) = {x | x < -2 ∨ 3 < x}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_complements_l3490_349063


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l3490_349084

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 1)^2
def parabola2 (x y : ℝ) : Prop := x - 2 = (y + 1)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

theorem intersection_sum_zero :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧
    (x₃, y₃) ≠ (x₄, y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_zero_l3490_349084


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l3490_349080

theorem sin_sum_of_complex_exponentials (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) = (4/5 : ℂ) + (3/5 : ℂ) * Complex.I ∧
  Complex.exp (Complex.I * δ) = (-5/13 : ℂ) + (12/13 : ℂ) * Complex.I →
  Real.sin (γ + δ) = 21/65 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l3490_349080
