import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_cubes_equals_fourth_power_l1772_177296

theorem sum_of_cubes_equals_fourth_power : 5^3 + 5^3 + 5^3 + 5^3 = 5^4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equals_fourth_power_l1772_177296


namespace NUMINAMATH_CALUDE_seashell_solution_l1772_177214

/-- The number of seashells found by Mary, Jessica, and Kevin -/
def seashell_problem (mary_shells jessica_shells : ℕ) (kevin_multiplier : ℕ) : Prop :=
  let kevin_shells := kevin_multiplier * mary_shells
  mary_shells + jessica_shells + kevin_shells = 113

/-- Theorem stating the solution to the seashell problem -/
theorem seashell_solution : seashell_problem 18 41 3 := by
  sorry

end NUMINAMATH_CALUDE_seashell_solution_l1772_177214


namespace NUMINAMATH_CALUDE_proposition_two_l1772_177251

theorem proposition_two (a b c : ℝ) (h1 : c > 1) (h2 : 0 < b) (h3 : b < 2) :
  a^2 + a*b + c > 0 := by
  sorry

end NUMINAMATH_CALUDE_proposition_two_l1772_177251


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_sticker_albums_l1772_177268

theorem largest_common_divisor_of_sticker_albums : ∃ (n : ℕ), n > 0 ∧ 
  n ∣ 1050 ∧ n ∣ 1260 ∧ n ∣ 945 ∧ 
  ∀ (m : ℕ), m > 0 → m ∣ 1050 → m ∣ 1260 → m ∣ 945 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_sticker_albums_l1772_177268


namespace NUMINAMATH_CALUDE_sum_of_constants_l1772_177221

theorem sum_of_constants (a b : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = a + b / x^2) →
  (2 = a + b) →
  (6 = a + b / 9) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_constants_l1772_177221


namespace NUMINAMATH_CALUDE_jansi_shopping_ratio_l1772_177226

/-- Represents the shopping scenario of Jansi -/
structure ShoppingScenario where
  initial_rupees : ℕ
  initial_coins : ℕ
  spent : ℚ

/-- Conditions of Jansi's shopping trip -/
def jansi_shopping : ShoppingScenario :=
{ initial_rupees := 15,
  initial_coins := 15,
  spent := 9.6 }

/-- The ratio of the amount Jansi came back with to the amount she started out with -/
def shopping_ratio (s : ShoppingScenario) : ℚ × ℚ :=
  let initial_amount : ℚ := s.initial_rupees + 0.2 * s.initial_coins
  let final_amount : ℚ := initial_amount - s.spent
  (final_amount, initial_amount)

theorem jansi_shopping_ratio :
  shopping_ratio jansi_shopping = (9, 25) := by sorry

end NUMINAMATH_CALUDE_jansi_shopping_ratio_l1772_177226


namespace NUMINAMATH_CALUDE_infinite_pairs_divisibility_l1772_177244

theorem infinite_pairs_divisibility (m : ℕ) (h_m_even : Even m) (h_m_ge_2 : m ≥ 2) :
  ∃ n : ℕ, n = m + 1 ∧ 
    n ≥ 2 ∧ 
    (m^m - 1) % n = 0 ∧ 
    (n^n - 1) % m = 0 := by
  sorry

end NUMINAMATH_CALUDE_infinite_pairs_divisibility_l1772_177244


namespace NUMINAMATH_CALUDE_salary_increase_l1772_177277

/-- Given a salary increase of 100% resulting in a new salary of $80,
    prove that the original salary was $40. -/
theorem salary_increase (new_salary : ℝ) (increase_percentage : ℝ) : 
  new_salary = 80 ∧ increase_percentage = 100 → 
  new_salary / 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l1772_177277


namespace NUMINAMATH_CALUDE_remainder_4x_mod_7_l1772_177223

theorem remainder_4x_mod_7 (x : ℤ) (h : x % 7 = 5) : (4 * x) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4x_mod_7_l1772_177223


namespace NUMINAMATH_CALUDE_point_classification_l1772_177227

-- Define the region D
def D (x y : ℝ) : Prop := y < x ∧ x + y ≤ 1 ∧ y ≥ -3

-- Define points P and Q
def P : ℝ × ℝ := (0, -2)
def Q : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem point_classification :
  D P.1 P.2 ∧ ¬D Q.1 Q.2 := by sorry

end NUMINAMATH_CALUDE_point_classification_l1772_177227


namespace NUMINAMATH_CALUDE_vlad_height_l1772_177225

/-- Proves that Vlad is 3 inches taller than 6 feet given the conditions of the problem -/
theorem vlad_height (vlad_feet : ℕ) (vlad_inches : ℕ) (sister_feet : ℕ) (sister_inches : ℕ) 
  (height_difference : ℕ) :
  vlad_feet = 6 →
  sister_feet = 2 →
  sister_inches = 10 →
  height_difference = 41 →
  vlad_inches = 3 :=
by
  sorry

#check vlad_height

end NUMINAMATH_CALUDE_vlad_height_l1772_177225


namespace NUMINAMATH_CALUDE_dvd_fraction_proof_l1772_177280

def initial_amount : ℚ := 320
def book_fraction : ℚ := 1/4
def book_additional : ℚ := 10
def dvd_additional : ℚ := 8
def final_amount : ℚ := 130

theorem dvd_fraction_proof :
  ∃ f : ℚ, 
    initial_amount - (book_fraction * initial_amount + book_additional) - 
    (f * (initial_amount - (book_fraction * initial_amount + book_additional)) + dvd_additional) = 
    final_amount ∧ f = 46/115 := by
  sorry

end NUMINAMATH_CALUDE_dvd_fraction_proof_l1772_177280


namespace NUMINAMATH_CALUDE_central_circle_radius_l1772_177222

/-- The radius of a circle tangent to six semicircles evenly arranged inside a regular hexagon -/
theorem central_circle_radius (side_length : ℝ) (h : side_length = 3) :
  let apothem := side_length * (Real.sqrt 3 / 2)
  let semicircle_radius := side_length / 2
  let central_radius := apothem - semicircle_radius
  central_radius = (3 * (Real.sqrt 3 - 1)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_central_circle_radius_l1772_177222


namespace NUMINAMATH_CALUDE_lee_fruit_loading_l1772_177273

/-- Represents the capacity of a large truck in tons -/
def large_truck_capacity : ℕ := 15

/-- Represents the number of large trucks used -/
def num_large_trucks : ℕ := 8

/-- Represents the total amount of fruits to be loaded in tons -/
def total_fruits : ℕ := num_large_trucks * large_truck_capacity

theorem lee_fruit_loading :
  total_fruits = 120 :=
by sorry

end NUMINAMATH_CALUDE_lee_fruit_loading_l1772_177273


namespace NUMINAMATH_CALUDE_min_sum_with_log_condition_l1772_177294

theorem min_sum_with_log_condition (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_log : Real.log a + Real.log b = Real.log (a + b)) : 
  ∀ x y : ℝ, x > 0 → y > 0 → Real.log x + Real.log y = Real.log (x + y) → a + b ≤ x + y ∧ a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_log_condition_l1772_177294


namespace NUMINAMATH_CALUDE_ratio_p_to_q_l1772_177297

theorem ratio_p_to_q (p q : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) 
  (h3 : (p + q) / (p - q) = 4 / 3) : p / q = 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_p_to_q_l1772_177297


namespace NUMINAMATH_CALUDE_inheritance_calculation_l1772_177289

theorem inheritance_calculation (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 15000) → x = 41379.31 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l1772_177289


namespace NUMINAMATH_CALUDE_expression_value_l1772_177284

theorem expression_value (a b : ℝ) (ha : a = 0.137) (hb : b = 0.098) :
  ((a + b)^2 - (a - b)^2) / (a * b) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1772_177284


namespace NUMINAMATH_CALUDE_smallest_possible_d_l1772_177266

theorem smallest_possible_d : 
  let f : ℝ → ℝ := λ d => (5 * Real.sqrt 3)^2 + (2 * d + 6)^2 - (4 * d)^2
  ∃ d : ℝ, f d = 0 ∧ ∀ d' : ℝ, f d' = 0 → d ≤ d' ∧ d = 1 + Real.sqrt 41 / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_d_l1772_177266


namespace NUMINAMATH_CALUDE_new_person_weight_l1772_177288

/-- The weight of a new person who replaces one person in a group, given the change in average weight -/
def weight_of_new_person (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + n * avg_increase

/-- Theorem stating the weight of the new person in the given scenario -/
theorem new_person_weight :
  weight_of_new_person 10 6.3 65 = 128 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1772_177288


namespace NUMINAMATH_CALUDE_journey_average_speed_l1772_177298

/-- Prove that the average speed of a journey with four equal-length segments,
    traveled at speeds of 3, 2, 6, and 3 km/h respectively, is 3 km/h. -/
theorem journey_average_speed (x : ℝ) (hx : x > 0) : 
  let total_distance := 4 * x
  let total_time := x / 3 + x / 2 + x / 6 + x / 3
  total_distance / total_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_journey_average_speed_l1772_177298


namespace NUMINAMATH_CALUDE_special_sale_discount_l1772_177279

theorem special_sale_discount (list_price : ℝ) (regular_discount_min : ℝ) (regular_discount_max : ℝ) (lowest_sale_price_ratio : ℝ) :
  list_price = 80 →
  regular_discount_min = 0.3 →
  regular_discount_max = 0.5 →
  lowest_sale_price_ratio = 0.4 →
  ∃ (additional_discount : ℝ),
    additional_discount = 0.2 ∧
    list_price * (1 - regular_discount_max) * (1 - additional_discount) = list_price * lowest_sale_price_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_special_sale_discount_l1772_177279


namespace NUMINAMATH_CALUDE_simplified_expression_l1772_177253

theorem simplified_expression (x : ℝ) :
  Real.sqrt (4 * x^2 - 8 * x + 4) + Real.sqrt (4 * x^2 + 8 * x + 4) + 5 =
  2 * |x - 1| + 2 * |x + 1| + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_l1772_177253


namespace NUMINAMATH_CALUDE_sequence_existence_l1772_177224

theorem sequence_existence (n : ℕ) (hn : n ≥ 3) :
  (∃ (a : ℕ → ℝ), 
    (a 1 = a (n + 1)) ∧ 
    (a 2 = a (n + 2)) ∧ 
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i * a (i + 1) + 1 = a (i + 2)))
  ↔ 
  (∃ k : ℕ, n = 3 * k) :=
by sorry

end NUMINAMATH_CALUDE_sequence_existence_l1772_177224


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1772_177291

theorem closest_integer_to_cube_root (x : ℝ := (7^3 + 9^3) ^ (1/3)) : 
  ∃ (n : ℤ), ∀ (m : ℤ), |x - n| ≤ |x - m| ∧ n = 10 := by
sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1772_177291


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1772_177260

theorem sqrt_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt (a + 1/2) + Real.sqrt (b + 1/2) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1772_177260


namespace NUMINAMATH_CALUDE_min_value_expression_l1772_177234

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (9 * r) / (3 * p + 2 * q) + (9 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) ≥ 2 ∧
  ((9 * r) / (3 * p + 2 * q) + (9 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) = 2 ↔ 3 * p = 2 * q ∧ 2 * q = 3 * r) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1772_177234


namespace NUMINAMATH_CALUDE_negative_division_example_l1772_177272

theorem negative_division_example : (-150) / (-25) = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_division_example_l1772_177272


namespace NUMINAMATH_CALUDE_rectangle_width_l1772_177203

/-- Given a rectangle with length 3 inches and unknown width, and a square with width 5 inches,
    if the difference in area between the square and the rectangle is 7 square inches,
    then the width of the rectangle is 6 inches. -/
theorem rectangle_width (w : ℝ) : 
  (5 * 5 : ℝ) - (3 * w) = 7 → w = 6 := by sorry

end NUMINAMATH_CALUDE_rectangle_width_l1772_177203


namespace NUMINAMATH_CALUDE_book_selection_theorem_l1772_177283

/-- The number of ways to select books from odd and even positions -/
def select_books (total : Nat) : Nat :=
  (total / 2) * (total / 2)

/-- Theorem stating the total number of ways to select the books -/
theorem book_selection_theorem :
  let biology_books := 12
  let chemistry_books := 8
  (select_books biology_books) * (select_books chemistry_books) = 576 := by
  sorry

#eval select_books 12 * select_books 8

end NUMINAMATH_CALUDE_book_selection_theorem_l1772_177283


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1772_177216

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) * (x - 1) > 0 ↔ x < 1 ∨ x > 3 := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1772_177216


namespace NUMINAMATH_CALUDE_max_sum_xyz_l1772_177242

theorem max_sum_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) :
  x + y + z ≤ 4 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
  16 * x₀ * y₀ * z₀ = (x₀ + y₀)^2 * (x₀ + z₀)^2 ∧ x₀ + y₀ + z₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_xyz_l1772_177242


namespace NUMINAMATH_CALUDE_clothes_washer_discount_l1772_177265

theorem clothes_washer_discount (original_price : ℝ) 
  (discount1 discount2 discount3 : ℝ) : 
  original_price = 500 →
  discount1 = 0.1 →
  discount2 = 0.2 →
  discount3 = 0.05 →
  (original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)) / original_price = 0.684 := by
sorry

end NUMINAMATH_CALUDE_clothes_washer_discount_l1772_177265


namespace NUMINAMATH_CALUDE_coeff_comparison_l1772_177241

open Polynomial

/-- The coefficient of x^20 in (1 + x^2 - x^3)^1000 is greater than
    the coefficient of x^20 in (1 - x^2 + x^3)^1000 --/
theorem coeff_comparison (x : ℝ) : 
  (coeff ((1 + X^2 - X^3 : ℝ[X])^1000) 20) > 
  (coeff ((1 - X^2 + X^3 : ℝ[X])^1000) 20) := by
  sorry

end NUMINAMATH_CALUDE_coeff_comparison_l1772_177241


namespace NUMINAMATH_CALUDE_max_songs_in_three_hours_l1772_177208

/-- Represents the maximum number of songs that can be played in a given time -/
def max_songs_played (short_songs : ℕ) (long_songs : ℕ) (short_duration : ℕ) (long_duration : ℕ) (total_time : ℕ) : ℕ :=
  let short_used := min short_songs (total_time / short_duration)
  let remaining_time := total_time - short_used * short_duration
  let long_used := min long_songs (remaining_time / long_duration)
  short_used + long_used

/-- Theorem stating the maximum number of songs that can be played in 3 hours -/
theorem max_songs_in_three_hours :
  max_songs_played 50 50 3 5 180 = 56 := by
  sorry

end NUMINAMATH_CALUDE_max_songs_in_three_hours_l1772_177208


namespace NUMINAMATH_CALUDE_top_face_after_16_rounds_l1772_177256

/-- Represents the faces of a cube -/
inductive Face : Type
  | A | B | C | D | E | F

/-- Represents the state of the cube -/
structure CubeState :=
  (top : Face)
  (front : Face)
  (right : Face)
  (back : Face)
  (left : Face)
  (bottom : Face)

/-- Performs one round of operations on the cube -/
def perform_round (state : CubeState) : CubeState :=
  sorry

/-- Initial state of the cube -/
def initial_state : CubeState :=
  { top := Face.E,
    front := Face.A,
    right := Face.C,
    back := Face.B,
    left := Face.D,
    bottom := Face.F }

/-- Theorem stating that after 16 rounds, the top face will be E -/
theorem top_face_after_16_rounds (n : Nat) :
  (n = 16) → (perform_round^[n] initial_state).top = Face.E :=
sorry

end NUMINAMATH_CALUDE_top_face_after_16_rounds_l1772_177256


namespace NUMINAMATH_CALUDE_root_in_interval_iff_a_in_range_l1772_177218

/-- The function f(x) = x^2 - ax + 1 has a root in the interval (1/2, 3) if and only if a ∈ [2, 10/3) -/
theorem root_in_interval_iff_a_in_range (a : ℝ) : 
  (∃ x : ℝ, 1/2 < x ∧ x < 3 ∧ x^2 - a*x + 1 = 0) ↔ 2 ≤ a ∧ a < 10/3 := by
sorry

end NUMINAMATH_CALUDE_root_in_interval_iff_a_in_range_l1772_177218


namespace NUMINAMATH_CALUDE_parabola_and_triangle_area_l1772_177274

/-- Parabola C: y² = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on the parabola -/
structure PointOnParabola (c : Parabola) where
  x : ℝ
  y : ℝ
  hy : y^2 = 2 * c.p * x

/-- Circle E: (x-1)² + y² = 1 -/
def CircleE (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

/-- Theorem about the parabola equation and minimum area of triangle -/
theorem parabola_and_triangle_area (c : Parabola) (m : PointOnParabola c)
    (h_dist : (m.x - 2)^2 + m.y^2 = 3) (h_x : m.x > 2) :
    c.p = 1 ∧ ∃ (a b : ℝ), CircleE 0 a ∧ CircleE 0 b ∧
    (∀ a' b' : ℝ, CircleE 0 a' ∧ CircleE 0 b' →
      1/2 * |a - b| * m.x ≤ 1/2 * |a' - b'| * m.x) ∧
    1/2 * |a - b| * m.x = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_and_triangle_area_l1772_177274


namespace NUMINAMATH_CALUDE_jumping_contest_l1772_177261

theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) : 
  grasshopper_jump = 14 →
  frog_jump = grasshopper_jump + 37 →
  mouse_jump = frog_jump - 16 →
  mouse_jump - grasshopper_jump = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_jumping_contest_l1772_177261


namespace NUMINAMATH_CALUDE_card_draw_probability_l1772_177215

def standard_deck := 52
def face_cards := 12
def hearts := 13
def tens := 4

theorem card_draw_probability : 
  let p1 := face_cards / standard_deck
  let p2 := hearts / (standard_deck - 1)
  let p3 := tens / (standard_deck - 2)
  p1 * p2 * p3 = 1 / 217 := by
  sorry

end NUMINAMATH_CALUDE_card_draw_probability_l1772_177215


namespace NUMINAMATH_CALUDE_algebra_sum_is_5_l1772_177238

def letter_value (n : ℕ) : ℤ :=
  match n % 10 with
  | 1 => 2
  | 2 => 3
  | 3 => 2
  | 4 => 1
  | 5 => 0
  | 6 => -1
  | 7 => -2
  | 8 => -3
  | 9 => -2
  | 0 => -1
  | _ => 0  -- This case should never occur

def alphabet_position (c : Char) : ℕ :=
  match c with
  | 'a' => 1
  | 'l' => 12
  | 'g' => 7
  | 'e' => 5
  | 'b' => 2
  | 'r' => 18
  | _ => 0  -- This case should never occur for valid input

theorem algebra_sum_is_5 :
  (letter_value (alphabet_position 'a') +
   letter_value (alphabet_position 'l') +
   letter_value (alphabet_position 'g') +
   letter_value (alphabet_position 'e') +
   letter_value (alphabet_position 'b') +
   letter_value (alphabet_position 'r') +
   letter_value (alphabet_position 'a')) = 5 := by
  sorry

end NUMINAMATH_CALUDE_algebra_sum_is_5_l1772_177238


namespace NUMINAMATH_CALUDE_value_of_a_l1772_177293

theorem value_of_a : ∀ (a b c d : ℤ),
  a = b + 7 →
  b = c + 15 →
  c = d + 25 →
  d = 90 →
  a = 137 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l1772_177293


namespace NUMINAMATH_CALUDE_tom_payment_proof_l1772_177292

/-- Represents the purchase of a fruit with its quantity and price per kg -/
structure FruitPurchase where
  quantity : Float
  pricePerKg : Float

/-- Calculates the total cost of a fruit purchase -/
def calculateCost (purchase : FruitPurchase) : Float :=
  purchase.quantity * purchase.pricePerKg

/-- Represents Tom's fruit shopping trip -/
def tomShopping : List FruitPurchase := [
  { quantity := 15.3, pricePerKg := 1.85 },  -- apples
  { quantity := 12.7, pricePerKg := 2.45 },  -- mangoes
  { quantity := 10.5, pricePerKg := 3.20 },  -- grapes
  { quantity := 6.2,  pricePerKg := 4.50 }   -- strawberries
]

/-- The discount rate applied to the total bill -/
def discountRate : Float := 0.10

/-- The sales tax rate applied to the discounted amount -/
def taxRate : Float := 0.06

/-- Calculates the final amount Tom pays after discount and tax -/
def calculateFinalAmount (purchases : List FruitPurchase) (discount : Float) (tax : Float) : Float :=
  let totalCost := purchases.map calculateCost |>.sum
  let discountedCost := totalCost * (1 - discount)
  let finalCost := discountedCost * (1 + tax)
  (finalCost * 100).round / 100  -- Round to nearest cent

theorem tom_payment_proof :
  calculateFinalAmount tomShopping discountRate taxRate = 115.36 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_proof_l1772_177292


namespace NUMINAMATH_CALUDE_new_person_weight_l1772_177220

theorem new_person_weight (initial_total_weight : ℝ) : 
  let initial_avg := initial_total_weight / 10
  let new_avg := initial_avg + 5
  let new_total_weight := new_avg * 10
  new_total_weight - initial_total_weight + 60 = 110 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l1772_177220


namespace NUMINAMATH_CALUDE_f_is_even_l1772_177200

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even (g : ℝ → ℝ) (h : is_odd_function g) :
  is_even_function (fun x ↦ |g (x^5)|) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_l1772_177200


namespace NUMINAMATH_CALUDE_prime_sum_product_l1772_177202

theorem prime_sum_product (p q : ℕ) : 
  Prime p → Prime q → p + q = 91 → p * q = 178 := by sorry

end NUMINAMATH_CALUDE_prime_sum_product_l1772_177202


namespace NUMINAMATH_CALUDE_special_quadrilateral_angles_l1772_177237

/-- A quadrilateral with specific angle relationships and side equality -/
structure SpecialQuadrilateral where
  A : ℝ  -- Angle at vertex A
  B : ℝ  -- Angle at vertex B
  C : ℝ  -- Angle at vertex C
  D : ℝ  -- Angle at vertex D
  angle_B_triple_A : B = 3 * A
  angle_C_triple_B : C = 3 * B
  angle_D_triple_C : D = 3 * C
  sum_of_angles : A + B + C + D = 360
  sides_equal : True  -- Representing AD = BC (not used in angle calculations)

/-- The angles in the special quadrilateral are 9°, 27°, 81°, and 243° -/
theorem special_quadrilateral_angles (q : SpecialQuadrilateral) :
  q.A = 9 ∧ q.B = 27 ∧ q.C = 81 ∧ q.D = 243 := by
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_angles_l1772_177237


namespace NUMINAMATH_CALUDE_four_card_selection_theorem_l1772_177259

/-- Represents a standard deck of 52 cards -/
def StandardDeck : Type := Unit

/-- Represents the four suits in a deck of cards -/
inductive Suit
| hearts | diamonds | clubs | spades

/-- Represents the rank of a card -/
inductive Rank
| ace | two | three | four | five | six | seven | eight | nine | ten
| jack | queen | king

/-- Determines if a rank is royal (J, Q, K) -/
def isRoyal (r : Rank) : Bool :=
  match r with
  | Rank.jack | Rank.queen | Rank.king => true
  | _ => false

/-- Represents a card with a suit and rank -/
structure Card where
  suit : Suit
  rank : Rank

/-- The number of ways to choose 4 cards from two standard decks -/
def numWaysToChoose4Cards (deck1 deck2 : StandardDeck) : ℕ := sorry

theorem four_card_selection_theorem (deck1 deck2 : StandardDeck) :
  numWaysToChoose4Cards deck1 deck2 = 438400 := by sorry

end NUMINAMATH_CALUDE_four_card_selection_theorem_l1772_177259


namespace NUMINAMATH_CALUDE_sequence_equality_l1772_177299

def A : ℕ → ℚ
  | 0 => 1
  | n + 1 => (A n + 2) / (A n + 1)

def B : ℕ → ℚ
  | 0 => 1
  | n + 1 => (B n ^ 2 + 2) / (2 * B n)

theorem sequence_equality (n : ℕ) : B (n + 1) = A (2 ^ n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l1772_177299


namespace NUMINAMATH_CALUDE_pride_and_prejudice_watching_time_l1772_177290

/-- Calculates the total hours spent watching a TV series given the number of episodes and minutes per episode -/
def total_watching_hours (num_episodes : ℕ) (minutes_per_episode : ℕ) : ℚ :=
  (num_episodes * minutes_per_episode : ℚ) / 60

/-- Proves that watching 6 episodes of 50 minutes each takes 5 hours -/
theorem pride_and_prejudice_watching_time :
  total_watching_hours 6 50 = 5 := by
  sorry

#eval total_watching_hours 6 50

end NUMINAMATH_CALUDE_pride_and_prejudice_watching_time_l1772_177290


namespace NUMINAMATH_CALUDE_sin_2017pi_over_3_l1772_177295

theorem sin_2017pi_over_3 : Real.sin (2017 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2017pi_over_3_l1772_177295


namespace NUMINAMATH_CALUDE_parabola_point_distance_to_focus_l1772_177206

theorem parabola_point_distance_to_focus (x y : ℝ) : 
  y^2 = 4*x →  -- Point A(x, y) is on the parabola y^2 = 4x
  (x - 1)^2 + y^2 = 36 →  -- Distance from A to focus (1, 0) is 6
  x = 7 := by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_to_focus_l1772_177206


namespace NUMINAMATH_CALUDE_least_perimeter_triangle_l1772_177270

theorem least_perimeter_triangle (a b c : ℕ) : 
  a = 40 → b = 48 → c > 0 → a + b > c → a + c > b → b + c > a → 
  (∀ x : ℕ, x > 0 → a + b > x → a + x > b → b + x > a → a + b + x ≥ a + b + c) →
  a + b + c = 97 := by sorry

end NUMINAMATH_CALUDE_least_perimeter_triangle_l1772_177270


namespace NUMINAMATH_CALUDE_cistern_specific_area_l1772_177258

/-- Calculates the total wet surface area of a rectangular cistern -/
def cistern_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem stating that a cistern with given dimensions has a specific wet surface area -/
theorem cistern_specific_area :
  cistern_wet_surface_area 12 14 1.25 = 233 := by
  sorry

end NUMINAMATH_CALUDE_cistern_specific_area_l1772_177258


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_cubic_inequality_negation_l1772_177201

theorem negation_of_universal_proposition (p : ℝ → Prop) :
  (¬∀ x, p x) ↔ (∃ x, ¬p x) := by sorry

theorem cubic_inequality_negation :
  (¬∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_cubic_inequality_negation_l1772_177201


namespace NUMINAMATH_CALUDE_game_probability_l1772_177243

/-- The number of possible choices for each player -/
def num_choices : ℕ := 16

/-- The probability of not winning a prize in a single trial -/
def prob_not_winning : ℚ := 15 / 16

theorem game_probability :
  (1 : ℚ) - (num_choices : ℚ) / ((num_choices : ℚ) * (num_choices : ℚ)) = prob_not_winning :=
by sorry

end NUMINAMATH_CALUDE_game_probability_l1772_177243


namespace NUMINAMATH_CALUDE_identity_function_only_solution_l1772_177287

theorem identity_function_only_solution 
  (f : ℕ+ → ℕ+) 
  (h : ∀ a b : ℕ+, (a - f b) ∣ (a * f a - b * f b)) :
  ∀ x : ℕ+, f x = x :=
by sorry

end NUMINAMATH_CALUDE_identity_function_only_solution_l1772_177287


namespace NUMINAMATH_CALUDE_rectangle_side_length_l1772_177211

/-- 
Given a rectangular arrangement with all right angles, where the top length 
consists of segments 3 cm, 2 cm, Y cm, and 1 cm sequentially, and the total 
bottom length is 11 cm, prove that Y = 5 cm.
-/
theorem rectangle_side_length (Y : ℝ) : 
  (3 : ℝ) + 2 + Y + 1 = 11 → Y = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l1772_177211


namespace NUMINAMATH_CALUDE_noah_holidays_l1772_177210

/-- Calculates the number of holidays taken in a year given monthly holidays. -/
def holidays_per_year (monthly_holidays : ℕ) : ℕ :=
  monthly_holidays * 12

/-- Theorem: Given 3 holidays per month for a full year, the total holidays is 36. -/
theorem noah_holidays :
  holidays_per_year 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_noah_holidays_l1772_177210


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_3_l1772_177228

/-- Two lines are parallel if and only if their slopes are equal -/
def are_parallel (m1 n1 c1 m2 n2 c2 : ℝ) : Prop :=
  m1 * n2 = m2 * n1

/-- The first line: ax + 2y + 3a = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 3 * a = 0

/-- The second line: 3x + (a-1)y = a-7 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  3 * x + (a - 1) * y = a - 7

theorem parallel_iff_a_eq_3 :
  ∀ a : ℝ, are_parallel a 2 (3*a) 3 (a-1) (a-7) ↔ a = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_3_l1772_177228


namespace NUMINAMATH_CALUDE_min_sum_m_n_l1772_177264

theorem min_sum_m_n (m n : ℕ+) (h : 45 * m = n^3) : 
  (∀ (m' n' : ℕ+), 45 * m' = n'^3 → m' + n' ≥ m + n) → m + n = 90 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_m_n_l1772_177264


namespace NUMINAMATH_CALUDE_paperclips_exceed_300_l1772_177213

def paperclips (k : ℕ) : ℕ := 5 * 3^k

theorem paperclips_exceed_300 : 
  (∀ n < 4, paperclips n ≤ 300) ∧ paperclips 4 > 300 := by sorry

end NUMINAMATH_CALUDE_paperclips_exceed_300_l1772_177213


namespace NUMINAMATH_CALUDE_two_numbers_equal_sum_product_quotient_l1772_177255

theorem two_numbers_equal_sum_product_quotient :
  ∃! (x y : ℝ), x ≠ 0 ∧ x + y = x * y ∧ x * y = x / y ∧ x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_equal_sum_product_quotient_l1772_177255


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l1772_177269

/-- The number of diagonals in a convex n-gon --/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A heptagon is a polygon with 7 sides --/
def is_heptagon (n : ℕ) : Prop := n = 7

theorem heptagon_diagonals :
  ∀ n : ℕ, is_heptagon n → num_diagonals n = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l1772_177269


namespace NUMINAMATH_CALUDE_sin_negative_390_degrees_l1772_177263

theorem sin_negative_390_degrees : Real.sin (-(390 * π / 180)) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_390_degrees_l1772_177263


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l1772_177205

theorem two_digit_number_puzzle :
  ∃! n : ℕ, 
    10 ≤ n ∧ n < 100 ∧  -- n is a two-digit number
    (n / 10 = 2 * (n % 10)) ∧  -- tens digit is twice the units digit
    (n - ((n % 10) * 10 + (n / 10)) = 36)  -- swapping digits results in 36 less
  :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l1772_177205


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1772_177250

-- Define the polynomials
def p (x : ℝ) : ℝ := 3*x^2 - 4*x + 3
def q (x : ℝ) : ℝ := -2*x^2 + 3*x - 4

-- State the theorem
theorem polynomial_expansion :
  ∀ x : ℝ, p x * q x = -6*x^4 + 17*x^3 - 30*x^2 + 25*x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1772_177250


namespace NUMINAMATH_CALUDE_fractional_part_theorem_l1772_177248

theorem fractional_part_theorem (x : ℝ) (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ ∃ m : ℤ, |k * x - m| ≤ 1 / n := by sorry

end NUMINAMATH_CALUDE_fractional_part_theorem_l1772_177248


namespace NUMINAMATH_CALUDE_coin_jar_problem_l1772_177271

theorem coin_jar_problem (x : ℕ) : 
  (x : ℚ) * (1 + 5 + 10 + 25) / 100 = 20 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_coin_jar_problem_l1772_177271


namespace NUMINAMATH_CALUDE_team_formation_count_l1772_177231

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem team_formation_count : 
  let total_boys : ℕ := 4
  let total_girls : ℕ := 5
  let team_size : ℕ := 5
  let ways_3b2g : ℕ := choose total_boys 3 * choose total_girls 2
  let ways_4b1g : ℕ := choose total_boys 4 * choose total_girls 1
  ways_3b2g + ways_4b1g = 45 := by sorry

end NUMINAMATH_CALUDE_team_formation_count_l1772_177231


namespace NUMINAMATH_CALUDE_max_m_value_l1772_177207

/-- The maximum value of m for which f and g satisfy the given conditions -/
theorem max_m_value : ∀ (n : ℝ), 
  (∀ (m : ℝ), (∀ (t : ℝ), (t^2 + m*t + n^2 ≥ 0) ∨ (t^2 + (m+2)*t + n^2 + m + 1 ≥ 0)) → m ≤ 1) ∧
  (∃ (m : ℝ), m = 1 ∧ ∀ (t : ℝ), (t^2 + m*t + n^2 ≥ 0) ∨ (t^2 + (m+2)*t + n^2 + m + 1 ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l1772_177207


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l1772_177286

def num_math_books : ℕ := 4
def num_history_books : ℕ := 7

def ways_to_arrange_books : ℕ :=
  -- Ways to choose math books for the ends
  (num_math_books * (num_math_books - 1)) *
  -- Ways to choose and arrange 2 history books from 7
  (num_history_books * (num_history_books - 1)) *
  -- Ways to choose the third book (math or history)
  (num_math_books + num_history_books - 3) *
  -- Ways to permute the first three books
  6 *
  -- Ways to arrange the remaining 6 books
  (6 * 5 * 4 * 3 * 2 * 1)

theorem book_arrangement_theorem :
  ways_to_arrange_books = 19571200 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l1772_177286


namespace NUMINAMATH_CALUDE_arrange_five_balls_three_boxes_l1772_177245

/-- The number of ways to put n distinguishable objects into k distinguishable containers -/
def arrange_objects (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem arrange_five_balls_three_boxes : arrange_objects 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_arrange_five_balls_three_boxes_l1772_177245


namespace NUMINAMATH_CALUDE_solution_of_linear_equation_l1772_177219

theorem solution_of_linear_equation :
  let f : ℝ → ℝ := λ x => x + 2
  f (-2) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_of_linear_equation_l1772_177219


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l1772_177229

theorem hemisphere_surface_area (diameter : ℝ) (h : diameter = 12) :
  let radius := diameter / 2
  let curved_surface_area := 2 * π * radius^2
  let base_area := π * radius^2
  curved_surface_area + base_area = 108 * π :=
by sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l1772_177229


namespace NUMINAMATH_CALUDE_max_x_squared_y_l1772_177282

theorem max_x_squared_y (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + 3 * y = 9) :
  x^2 * y ≤ 36 :=
sorry

end NUMINAMATH_CALUDE_max_x_squared_y_l1772_177282


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1772_177247

def M : Set ℝ := {x | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x | x < -5 ∨ x > 5}

theorem union_of_M_and_N : M ∪ N = {x | x < -5 ∨ x > -3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1772_177247


namespace NUMINAMATH_CALUDE_train_carriages_count_l1772_177285

/-- Calculates the number of carriages in a train given specific conditions -/
theorem train_carriages_count (carriage_length engine_length : ℝ)
                               (train_speed : ℝ)
                               (bridge_crossing_time : ℝ)
                               (bridge_length : ℝ) :
  carriage_length = 60 →
  engine_length = 60 →
  train_speed = 60 * 1000 / 60 →
  bridge_crossing_time = 5 →
  bridge_length = 3.5 * 1000 →
  ∃ n : ℕ, n = 24 ∧ 
    n * carriage_length + engine_length = 
    train_speed * bridge_crossing_time - bridge_length :=
by
  sorry

end NUMINAMATH_CALUDE_train_carriages_count_l1772_177285


namespace NUMINAMATH_CALUDE_range_of_z_l1772_177267

theorem range_of_z (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (z_min z_max : ℝ), z_min = -4/3 ∧ z_max = 0 ∧
  ∀ z, z = (y - 1) / (x + 2) → z_min ≤ z ∧ z ≤ z_max :=
sorry

end NUMINAMATH_CALUDE_range_of_z_l1772_177267


namespace NUMINAMATH_CALUDE_inequality_reverse_l1772_177249

theorem inequality_reverse (a b : ℝ) (h : a > b) : -4 * a < -4 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_reverse_l1772_177249


namespace NUMINAMATH_CALUDE_solve_for_t_l1772_177239

theorem solve_for_t (s t : ℚ) (eq1 : 8 * s + 7 * t = 145) (eq2 : s = t + 3) : t = 121 / 15 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_t_l1772_177239


namespace NUMINAMATH_CALUDE_min_ratio_partition_l1772_177240

def S : Finset ℕ := Finset.range 10

theorem min_ratio_partition (p₁ p₂ : ℕ) 
  (h_partition : ∃ (A B : Finset ℕ), A ∪ B = S ∧ A ∩ B = ∅ ∧ 
    p₁ = A.prod id ∧ p₂ = B.prod id)
  (h_divisible : p₁ % p₂ = 0) :
  p₁ / p₂ ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_min_ratio_partition_l1772_177240


namespace NUMINAMATH_CALUDE_geometric_ratio_sum_condition_l1772_177276

theorem geometric_ratio_sum_condition (a b c d a' b' c' d' : ℝ) 
  (h1 : a / b = c / d) (h2 : a' / b' = c' / d') :
  (a + a') / (b + b') = (c + c') / (d + d') ↔ a / a' = b / b' ∧ b / b' = c / c' ∧ c / c' = d / d' :=
by sorry

end NUMINAMATH_CALUDE_geometric_ratio_sum_condition_l1772_177276


namespace NUMINAMATH_CALUDE_warehouse_total_boxes_l1772_177262

/-- Represents the number of boxes in each warehouse --/
structure Warehouses where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ
  E : ℕ

/-- Conditions for the warehouse problem --/
def warehouseConditions (w : Warehouses) : Prop :=
  ∃ x : ℕ,
    w.A = x ∧
    w.B = 3 * x ∧
    w.C = (3 * x) / 2 + 100 ∧
    w.D = 3 * x + 150 ∧
    w.E = 4 * x - 50 ∧
    w.B = w.E + 300

/-- The theorem to be proved --/
theorem warehouse_total_boxes (w : Warehouses) :
  warehouseConditions w → w.A + w.B + w.C + w.D + w.E = 4575 := by
  sorry


end NUMINAMATH_CALUDE_warehouse_total_boxes_l1772_177262


namespace NUMINAMATH_CALUDE_fraction_of_fraction_one_ninth_of_three_fourths_l1772_177217

theorem fraction_of_fraction (a b c d : ℚ) (h1 : b ≠ 0) (h2 : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem one_ninth_of_three_fourths :
  (1 / 9) / (3 / 4) = 4 / 27 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_one_ninth_of_three_fourths_l1772_177217


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l1772_177232

/-- The quadratic function f(x) = 2 - (x+1)^2 -/
def f (x : ℝ) : ℝ := 2 - (x + 1)^2

/-- The vertex of a quadratic function -/
structure Vertex where
  x : ℝ
  y : ℝ

/-- Theorem: The vertex of f(x) = 2 - (x+1)^2 is at (-1, 2) -/
theorem vertex_of_quadratic : 
  ∃ (v : Vertex), v.x = -1 ∧ v.y = 2 ∧ 
  ∀ (x : ℝ), f x ≤ f v.x := by
  sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l1772_177232


namespace NUMINAMATH_CALUDE_total_numbers_correction_l1772_177275

/-- Given an initial average of 15, where one number was misread as 26 instead of 36,
    and the correct average is 16, prove that the total number of numbers is 10. -/
theorem total_numbers_correction (initial_avg : ℚ) (misread : ℚ) (correct : ℚ) (correct_avg : ℚ)
  (h1 : initial_avg = 15)
  (h2 : misread = 26)
  (h3 : correct = 36)
  (h4 : correct_avg = 16) :
  ∃ (n : ℕ) (S : ℚ), n > 0 ∧ n = 10 ∧ 
    S / n + misread / n = initial_avg ∧
    S / n + correct / n = correct_avg :=
by sorry

end NUMINAMATH_CALUDE_total_numbers_correction_l1772_177275


namespace NUMINAMATH_CALUDE_no_common_real_root_l1772_177235

theorem no_common_real_root (a b : ℚ) : ¬∃ (r : ℝ), r^5 - r - 1 = 0 ∧ r^2 + a*r + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_real_root_l1772_177235


namespace NUMINAMATH_CALUDE_p_and_q_false_iff_a_range_l1772_177233

/-- The logarithm function with base 10 -/
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

/-- The function f(x) = lg(ax^2 - x + a/16) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := lg (a * x^2 - x + a/16)

/-- Proposition p: The range of f(x) is ℝ -/
def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, f a x = y

/-- Proposition q: 3^x - 9^x < a holds for all real numbers x -/
def q (a : ℝ) : Prop := ∀ x : ℝ, 3^x - 9^x < a

/-- Theorem: "p and q" is false iff a > 2 or a ≤ 1/4 -/
theorem p_and_q_false_iff_a_range (a : ℝ) : ¬(p a ∧ q a) ↔ a > 2 ∨ a ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_false_iff_a_range_l1772_177233


namespace NUMINAMATH_CALUDE_earth_surface_area_scientific_notation_l1772_177236

/-- The surface area of the Earth in square kilometers. -/
def earth_surface_area : ℝ := 510000000

/-- The scientific notation representation of the Earth's surface area. -/
def earth_surface_area_scientific : ℝ := 5.1 * (10 ^ 8)

/-- Theorem stating that the Earth's surface area is correctly represented in scientific notation. -/
theorem earth_surface_area_scientific_notation : 
  earth_surface_area = earth_surface_area_scientific := by sorry

end NUMINAMATH_CALUDE_earth_surface_area_scientific_notation_l1772_177236


namespace NUMINAMATH_CALUDE_sum_of_integers_l1772_177281

theorem sum_of_integers (p q r s : ℤ) 
  (eq1 : p - q + r = 7)
  (eq2 : q - r + s = 8)
  (eq3 : r - s + p = 4)
  (eq4 : s - p + q = 3) :
  p + q + r + s = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1772_177281


namespace NUMINAMATH_CALUDE_theatre_distance_is_340_l1772_177278

/-- Represents the problem of Julia's drive to the theatre. -/
structure JuliaDrive where
  initial_speed : ℝ
  speed_increase : ℝ
  initial_time : ℝ
  late_time : ℝ
  early_time : ℝ

/-- Calculates the total distance to the theatre based on the given conditions. -/
def calculate_distance (drive : JuliaDrive) : ℝ :=
  let total_time := drive.initial_time + (drive.late_time + drive.early_time)
  let remaining_time := total_time - drive.initial_time
  let remaining_distance := (drive.initial_speed + drive.speed_increase) * remaining_time
  drive.initial_speed * drive.initial_time + remaining_distance

/-- Theorem stating that the distance to the theatre is 340 miles. -/
theorem theatre_distance_is_340 (drive : JuliaDrive)
  (h1 : drive.initial_speed = 40)
  (h2 : drive.speed_increase = 20)
  (h3 : drive.initial_time = 1)
  (h4 : drive.late_time = 1.5)
  (h5 : drive.early_time = 1) :
  calculate_distance drive = 340 := by
  sorry

end NUMINAMATH_CALUDE_theatre_distance_is_340_l1772_177278


namespace NUMINAMATH_CALUDE_family_movie_night_l1772_177230

theorem family_movie_night (regular_price adult_price elderly_price : ℕ)
  (child_price_diff total_payment change num_adults num_elderly : ℕ) :
  regular_price = 15 →
  adult_price = 12 →
  elderly_price = 10 →
  child_price_diff = 5 →
  total_payment = 150 →
  change = 3 →
  num_adults = 4 →
  num_elderly = 2 →
  ∃ (num_children : ℕ),
    num_children = 11 ∧
    total_payment - change = 
      num_adults * adult_price + 
      num_elderly * elderly_price + 
      num_children * (adult_price - child_price_diff) :=
by sorry

end NUMINAMATH_CALUDE_family_movie_night_l1772_177230


namespace NUMINAMATH_CALUDE_total_revenue_equals_8189_35_l1772_177246

-- Define the types of ground beef
structure GroundBeef where
  regular : ℝ
  lean : ℝ
  extraLean : ℝ

-- Define the prices
def regularPrice : ℝ := 3.50
def leanPrice : ℝ := 4.25
def extraLeanPrice : ℝ := 5.00

-- Define the sales for each day
def mondaySales : GroundBeef := { regular := 198.5, lean := 276.2, extraLean := 150.7 }
def tuesdaySales : GroundBeef := { regular := 210, lean := 420, extraLean := 150 }
def wednesdaySales : GroundBeef := { regular := 230, lean := 324.6, extraLean := 120.4 }

-- Define the discount for Tuesday
def tuesdayDiscount : ℝ := 0.1

-- Define the sale price for lean ground beef on Wednesday
def wednesdayLeanSalePrice : ℝ := 3.75

-- Function to calculate revenue for a single day
def calculateDayRevenue (sales : GroundBeef) (regularPrice leanPrice extraLeanPrice : ℝ) : ℝ :=
  sales.regular * regularPrice + sales.lean * leanPrice + sales.extraLean * extraLeanPrice

-- Theorem statement
theorem total_revenue_equals_8189_35 :
  let mondayRevenue := calculateDayRevenue mondaySales regularPrice leanPrice extraLeanPrice
  let tuesdayRevenue := calculateDayRevenue tuesdaySales (regularPrice * (1 - tuesdayDiscount)) (leanPrice * (1 - tuesdayDiscount)) (extraLeanPrice * (1 - tuesdayDiscount))
  let wednesdayRevenue := calculateDayRevenue wednesdaySales regularPrice wednesdayLeanSalePrice extraLeanPrice
  mondayRevenue + tuesdayRevenue + wednesdayRevenue = 8189.35 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_equals_8189_35_l1772_177246


namespace NUMINAMATH_CALUDE_log_78903_between_consecutive_integers_l1772_177212

theorem log_78903_between_consecutive_integers :
  ∃ (c d : ℤ), c + 1 = d ∧ (c : ℝ) < Real.log 78903 / Real.log 10 ∧ Real.log 78903 / Real.log 10 < (d : ℝ) → c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_78903_between_consecutive_integers_l1772_177212


namespace NUMINAMATH_CALUDE_max_value_negative_x_min_value_greater_than_negative_one_l1772_177204

-- Problem 1
theorem max_value_negative_x (x : ℝ) (hx : x < 0) :
  (x^2 + x + 1) / x ≤ -1 :=
by sorry

-- Problem 2
theorem min_value_greater_than_negative_one (x : ℝ) (hx : x > -1) :
  ((x + 5) * (x + 2)) / (x + 1) ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_value_negative_x_min_value_greater_than_negative_one_l1772_177204


namespace NUMINAMATH_CALUDE_horizontal_chord_cubic_l1772_177257

/-- A cubic function f(x) = x^3 - x has a horizontal chord of length a 
    if and only if 0 < a ≤ 2 -/
theorem horizontal_chord_cubic (a : ℝ) :
  (∃ x : ℝ, (x + a)^3 - (x + a) = x^3 - x) ↔ (0 < a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_horizontal_chord_cubic_l1772_177257


namespace NUMINAMATH_CALUDE_quadrilateral_bf_length_l1772_177252

-- Define the points
variable (A B C D E F : ℝ × ℝ)

-- Define the conditions
variable (h1 : A.1 = 0 ∧ A.2 = 0)  -- A is at (0,0)
variable (h2 : C.1 = 10 ∧ C.2 = 0)  -- C is at (10,0)
variable (h3 : E.1 = 3 ∧ E.2 = 0)  -- E is at (3,0)
variable (h4 : F.1 = 7 ∧ F.2 = 0)  -- F is at (7,0)
variable (h5 : D.1 = 3 ∧ D.2 = -5)  -- D is at (3,-5)
variable (h6 : B.1 = 7 ∧ B.2 = 4.2)  -- B is at (7,4.2)

-- Define the geometric conditions
variable (h7 : (B.2 - A.2) * (D.1 - A.1) = (D.2 - A.2) * (B.1 - A.1))  -- ∠BAD is right
variable (h8 : (B.2 - C.2) * (D.1 - C.1) = (D.2 - C.2) * (B.1 - C.1))  -- ∠BCD is right
variable (h9 : (D.2 - E.2) * (C.1 - E.1) = (C.2 - E.2) * (D.1 - E.1))  -- DE ⊥ AC
variable (h10 : (B.2 - F.2) * (A.1 - F.1) = (A.2 - F.2) * (B.1 - F.1))  -- BF ⊥ AC

-- Define the length conditions
variable (h11 : Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 3)  -- AE = 3
variable (h12 : Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) = 5)  -- DE = 5
variable (h13 : Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2) = 7)  -- CE = 7

-- Theorem statement
theorem quadrilateral_bf_length : 
  Real.sqrt ((F.1 - B.1)^2 + (F.2 - B.2)^2) = 4.2 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_bf_length_l1772_177252


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1772_177209

/-- An isosceles triangle with two sides of length 9 and one side of length 2 has a perimeter of 20. -/
theorem isosceles_triangle_perimeter :
  ∀ (a b c : ℝ), 
    a = 9 → b = 9 → c = 2 →
    (a + b > c) ∧ (a + c > b) ∧ (b + c > a) →  -- Triangle inequality
    a = b →  -- Isosceles condition
    a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1772_177209


namespace NUMINAMATH_CALUDE_candidates_appeared_l1772_177254

theorem candidates_appeared (x : ℝ) 
  (h1 : 0.07 * x = 0.06 * x + 82) : x = 8200 := by
  sorry

end NUMINAMATH_CALUDE_candidates_appeared_l1772_177254
