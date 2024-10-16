import Mathlib

namespace NUMINAMATH_CALUDE_same_color_sock_pairs_l1280_128068

def total_socks : ℕ := 12
def red_socks : ℕ := 5
def green_socks : ℕ := 3
def blue_socks : ℕ := 4

theorem same_color_sock_pairs :
  (Nat.choose red_socks 2) + (Nat.choose green_socks 2) + (Nat.choose blue_socks 2) = 19 := by
  sorry

end NUMINAMATH_CALUDE_same_color_sock_pairs_l1280_128068


namespace NUMINAMATH_CALUDE_paul_sandwich_consumption_l1280_128011

/-- Calculates the number of sandwiches eaten in one 3-day cycle -/
def sandwiches_per_cycle (initial : ℕ) : ℕ :=
  initial + 2 * initial + 4 * initial

/-- Calculates the total number of sandwiches eaten in a given number of days -/
def total_sandwiches (days : ℕ) (initial : ℕ) : ℕ :=
  (days / 3) * sandwiches_per_cycle initial + 
  if days % 3 = 1 then initial
  else if days % 3 = 2 then initial + 2 * initial
  else 0

theorem paul_sandwich_consumption :
  total_sandwiches 6 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_paul_sandwich_consumption_l1280_128011


namespace NUMINAMATH_CALUDE_apple_banana_ratio_l1280_128055

/-- Proves that the ratio of apples to bananas is 2:1 given the total number of fruits,
    number of bananas, and number of oranges in a bowl of fruit. -/
theorem apple_banana_ratio (total : ℕ) (bananas : ℕ) (oranges : ℕ)
    (h_total : total = 12)
    (h_bananas : bananas = 2)
    (h_oranges : oranges = 6) :
    (total - bananas - oranges) / bananas = 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_banana_ratio_l1280_128055


namespace NUMINAMATH_CALUDE_impossibility_of_sequence_conditions_l1280_128016

def is_valid_sequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  (∀ n : ℕ, a (n + 3) = a n) ∧
  (∀ n : ℕ, a n * a (n + 3) - a (n + 1) * a (n + 2) = c)

theorem impossibility_of_sequence_conditions : 
  ¬∃ (a : ℕ → ℝ) (c : ℝ), is_valid_sequence a c ∧ a 1 = 2 ∧ c = 2 :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_sequence_conditions_l1280_128016


namespace NUMINAMATH_CALUDE_gcd_45_75_l1280_128010

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45_75_l1280_128010


namespace NUMINAMATH_CALUDE_onions_remaining_l1280_128034

theorem onions_remaining (initial : Nat) (sold : Nat) (h1 : initial = 98) (h2 : sold = 65) :
  initial - sold = 33 := by
  sorry

end NUMINAMATH_CALUDE_onions_remaining_l1280_128034


namespace NUMINAMATH_CALUDE_correct_num_dimes_l1280_128076

/-- The number of dimes used to make a purchase of $2 with 50 coins (dimes and nickels) -/
def num_dimes : ℕ := 10

theorem correct_num_dimes :
  ∀ (d n : ℕ),
  d + n = 50 →  -- Total number of coins
  10 * d + 5 * n = 200 →  -- Total value in cents
  d = num_dimes :=
by sorry

end NUMINAMATH_CALUDE_correct_num_dimes_l1280_128076


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1280_128074

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

-- Define the point P
def P : ℝ × ℝ := (2, 4)

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x + 2*y - 10 = 0

theorem tangent_line_to_circle :
  (∀ x y, C x y → ¬(tangent_line x y)) ∧
  tangent_line P.1 P.2 ∧
  ∃! p : ℝ × ℝ, C p.1 p.2 ∧ tangent_line p.1 p.2 ∧ p ≠ P :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1280_128074


namespace NUMINAMATH_CALUDE_bicycle_price_calculation_l1280_128049

theorem bicycle_price_calculation (initial_price : ℝ) : 
  let first_sale_price := initial_price * 1.20
  let final_price := first_sale_price * 1.25
  final_price = 225 → initial_price = 150 := by
sorry

end NUMINAMATH_CALUDE_bicycle_price_calculation_l1280_128049


namespace NUMINAMATH_CALUDE_total_amount_192_rupees_l1280_128019

/-- Represents the denominations of rupee notes -/
inductive Denomination
  | One
  | Five
  | Ten

/-- Calculates the value of a single note of a given denomination -/
def noteValue (d : Denomination) : Nat :=
  match d with
  | Denomination.One => 1
  | Denomination.Five => 5
  | Denomination.Ten => 10

/-- Represents the collection of notes -/
structure NoteCollection where
  totalNotes : Nat
  denominations : List Denomination
  equalDenominations : List.length denominations = 3
  equalDistribution : totalNotes % (List.length denominations) = 0

/-- Theorem stating that a collection of 36 notes equally distributed among
    one-rupee, five-rupee, and ten-rupee denominations totals 192 rupees -/
theorem total_amount_192_rupees (nc : NoteCollection)
    (h1 : nc.totalNotes = 36)
    (h2 : nc.denominations = [Denomination.One, Denomination.Five, Denomination.Ten]) :
    (nc.totalNotes / 3) * (noteValue Denomination.One +
                           noteValue Denomination.Five +
                           noteValue Denomination.Ten) = 192 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_192_rupees_l1280_128019


namespace NUMINAMATH_CALUDE_numeral_with_seven_difference_63_l1280_128000

/-- Represents a numeral with a specific digit. -/
structure Numeral where
  value : ℕ
  digit : ℕ
  digit_position : ℕ

/-- The face value of a digit is the digit itself. -/
def face_value (n : Numeral) : ℕ := n.digit

/-- The place value of a digit in a numeral. -/
def place_value (n : Numeral) : ℕ := n.digit * (10 ^ n.digit_position)

/-- Theorem: If the difference between the place value and face value of 7 in a numeral is 63,
    then the numeral is 70. -/
theorem numeral_with_seven_difference_63 (n : Numeral) 
    (h1 : n.digit = 7)
    (h2 : place_value n - face_value n = 63) : 
  n.value = 70 := by
  sorry

end NUMINAMATH_CALUDE_numeral_with_seven_difference_63_l1280_128000


namespace NUMINAMATH_CALUDE_three_good_sets_l1280_128089

-- Define the concept of a "good set"
def is_good_set (C : Set (ℝ × ℝ)) : Prop :=
  ∀ p₁ ∈ C, ∃ p₂ ∈ C, p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

-- Define the four sets
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}
def C₂ : Set (ℝ × ℝ) := {p | p.1^2 - p.2^2 = 9}
def C₃ : Set (ℝ × ℝ) := {p | 2 * p.1^2 + p.2^2 = 9}
def C₄ : Set (ℝ × ℝ) := {p | p.1^2 + p.2 = 9}

-- Theorem: Exactly three of the four sets are "good sets"
theorem three_good_sets : 
  (is_good_set C₁ ∧ is_good_set C₃ ∧ is_good_set C₄ ∧ ¬is_good_set C₂) := by
  sorry


end NUMINAMATH_CALUDE_three_good_sets_l1280_128089


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1280_128085

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1280_128085


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1280_128060

/-- Given a square with side length 40 cm that is cut into 5 identical rectangles,
    the length of the shorter side of each rectangle that maximizes its area is 8 cm. -/
theorem rectangle_max_area (square_side : ℝ) (num_rectangles : ℕ) 
  (h1 : square_side = 40)
  (h2 : num_rectangles = 5) :
  let rectangle_area := square_side^2 / num_rectangles
  let shorter_side := square_side / num_rectangles
  shorter_side = 8 ∧ 
  ∀ (w : ℝ), w > 0 → w * (square_side^2 / (num_rectangles * w)) ≤ rectangle_area :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1280_128060


namespace NUMINAMATH_CALUDE_subway_speed_increase_l1280_128040

/-- The speed equation for the subway train -/
def speed (s : ℝ) : ℝ := s^2 + 2*s

/-- The theorem stating the time at which the train is moving 55 km/h faster -/
theorem subway_speed_increase (s : ℝ) (h1 : 0 ≤ s) (h2 : s ≤ 7) :
  speed s = speed 2 + 55 ↔ s = 7 := by sorry

end NUMINAMATH_CALUDE_subway_speed_increase_l1280_128040


namespace NUMINAMATH_CALUDE_system_solution_l1280_128009

theorem system_solution : ∃ (x y : ℝ), 2 * x - y = 8 ∧ 3 * x + 2 * y = 5 := by
  use 3, -2
  sorry

end NUMINAMATH_CALUDE_system_solution_l1280_128009


namespace NUMINAMATH_CALUDE_order_of_powers_l1280_128042

theorem order_of_powers : 
  let a : ℕ := 2^55
  let b : ℕ := 3^44
  let c : ℕ := 5^33
  let d : ℕ := 6^22
  a < d ∧ d < b ∧ b < c :=
by sorry

end NUMINAMATH_CALUDE_order_of_powers_l1280_128042


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_conditions_l1280_128043

theorem greatest_integer_satisfying_conditions : 
  ∃ (n : ℕ), n < 150 ∧ 
  (∃ (k m : ℕ), n = 9 * k - 2 ∧ n = 11 * m - 4) ∧
  (∀ (n' : ℕ), n' < 150 → 
    (∃ (k' m' : ℕ), n' = 9 * k' - 2 ∧ n' = 11 * m' - 4) → 
    n' ≤ n) ∧
  n = 139 := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_conditions_l1280_128043


namespace NUMINAMATH_CALUDE_temporary_wall_area_l1280_128094

theorem temporary_wall_area : 
  let width : Real := 5.4
  let length : Real := 2.5
  width * length = 13.5 := by
sorry

end NUMINAMATH_CALUDE_temporary_wall_area_l1280_128094


namespace NUMINAMATH_CALUDE_not_R_intersection_A_B_l1280_128096

def set_A : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def set_B : Set ℝ := {x | x - 2 > 0}

theorem not_R_intersection_A_B :
  (set_A ∩ set_B)ᶜ = {x : ℝ | x ≤ 2 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_not_R_intersection_A_B_l1280_128096


namespace NUMINAMATH_CALUDE_y_variation_l1280_128035

/-- A function that varies directly as x and inversely as the square of z -/
def y (x z : ℝ) : ℝ := sorry

theorem y_variation (k : ℝ) :
  (∀ x z, y x z = k * x / (z^2)) →
  y 5 1 = 10 →
  y (-10) 2 = -5 := by sorry

end NUMINAMATH_CALUDE_y_variation_l1280_128035


namespace NUMINAMATH_CALUDE_refrigerator_installation_cost_l1280_128038

def refrigerator_problem (purchase_price : ℝ) (discount_rate : ℝ) 
  (transport_cost : ℝ) (selling_price : ℝ) : Prop :=
  let labelled_price := purchase_price / (1 - discount_rate)
  let profit_rate := 0.1
  let total_cost := labelled_price + transport_cost + 
    (selling_price - labelled_price * (1 + profit_rate))
  total_cost - purchase_price - transport_cost = 287.5

theorem refrigerator_installation_cost :
  refrigerator_problem 13500 0.2 125 18975 :=
sorry

end NUMINAMATH_CALUDE_refrigerator_installation_cost_l1280_128038


namespace NUMINAMATH_CALUDE_triangle_side_length_l1280_128062

/-- Given a triangle ABC with side lengths a, b, c and angle B, 
    prove that if b = √3, c = 3, and B = 30°, then a = 2√3 -/
theorem triangle_side_length (a b c : ℝ) (B : ℝ) :
  b = Real.sqrt 3 → c = 3 → B = π / 6 → a = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1280_128062


namespace NUMINAMATH_CALUDE_special_triangle_c_eq_9_l1280_128091

/-- A triangle with sides a, b, and c satisfying the given conditions -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_triangle : a + b > c ∧ b + c > a ∧ a + c > b
  a_eq_9 : a = 9
  b_eq_2 : b = 2
  c_is_odd : ∃ (k : ℤ), c = 2 * k + 1

/-- The value of c in the special triangle is 9 -/
theorem special_triangle_c_eq_9 (t : SpecialTriangle) : t.c = 9 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_c_eq_9_l1280_128091


namespace NUMINAMATH_CALUDE_exponent_of_five_in_30_factorial_l1280_128026

theorem exponent_of_five_in_30_factorial :
  ∃ k : ℕ, (30 : ℕ).factorial = 5^7 * k ∧ ¬(5 ∣ k) :=
by sorry

end NUMINAMATH_CALUDE_exponent_of_five_in_30_factorial_l1280_128026


namespace NUMINAMATH_CALUDE_two_digit_number_divisible_by_55_l1280_128003

theorem two_digit_number_divisible_by_55 (a b : ℕ) : 
  (a ≥ 1 ∧ a ≤ 9) →  -- 'a' is a single digit (tens place)
  (b ≥ 0 ∧ b ≤ 9) →  -- 'b' is a single digit (units place)
  (10 * a + b) % 55 = 0 →  -- number is divisible by 55
  (∀ (x y : ℕ), (x ≥ 1 ∧ x ≤ 9) → (y ≥ 0 ∧ y ≤ 9) → (10 * x + y) % 55 = 0 → x * y ≤ 15) →  -- greatest possible value of b × a is 15
  10 * a + b = 65 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_divisible_by_55_l1280_128003


namespace NUMINAMATH_CALUDE_coffee_shop_lattes_l1280_128069

/-- The number of teas sold -/
def T : ℕ := 6

/-- The number of lattes sold -/
def L : ℕ := 4 * T + 8

theorem coffee_shop_lattes : L = 32 := by
  sorry

end NUMINAMATH_CALUDE_coffee_shop_lattes_l1280_128069


namespace NUMINAMATH_CALUDE_equal_domain_interval_exponential_l1280_128037

/-- Definition of an "equal domain interval" for a function --/
def is_equal_domain_interval (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  m < n ∧
  (∀ x y, m ≤ x ∧ x < y ∧ y ≤ n → f x < f y) ∧
  (∀ y, m ≤ y ∧ y ≤ n → ∃ x, m ≤ x ∧ x ≤ n ∧ f x = y)

/-- Theorem: If [1,2] is an "equal domain interval" for f(x) = a⋅2^x + b (a > 0), then a = 1/2 and b = 0 --/
theorem equal_domain_interval_exponential (a b : ℝ) (h : a > 0) :
  is_equal_domain_interval (λ x => a * 2^x + b) 1 2 → a = 1/2 ∧ b = 0 := by
  sorry


end NUMINAMATH_CALUDE_equal_domain_interval_exponential_l1280_128037


namespace NUMINAMATH_CALUDE_fence_length_of_area_200_l1280_128057

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  short_side : ℝ
  area : ℝ
  area_eq : area = 2 * short_side * short_side

/-- The total fence length of the special rectangle -/
def fence_length (r : SpecialRectangle) : ℝ :=
  2 * r.short_side + 2 * r.short_side

/-- Theorem: The fence length of a special rectangle with area 200 is 40 -/
theorem fence_length_of_area_200 :
  ∃ r : SpecialRectangle, r.area = 200 ∧ fence_length r = 40 := by
  sorry


end NUMINAMATH_CALUDE_fence_length_of_area_200_l1280_128057


namespace NUMINAMATH_CALUDE_no_right_angle_in_sequence_l1280_128006

/-- Represents a triangle with three angles -/
structure Triangle where
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

/-- Generates the next triangle in the sequence -/
def nextTriangle (t : Triangle) : Triangle :=
  { angleA := t.angleA, angleB := t.angleB, angleC := t.angleC }

/-- The original triangle ABC -/
def originalTriangle : Triangle :=
  { angleA := 59, angleB := 61, angleC := 60 }

/-- Generates the nth triangle in the sequence -/
def nthTriangle (n : ℕ) : Triangle :=
  match n with
  | 0 => originalTriangle
  | n+1 => nextTriangle (nthTriangle n)

theorem no_right_angle_in_sequence :
  ∀ n : ℕ, (nthTriangle n).angleA ≠ 90 ∧ (nthTriangle n).angleB ≠ 90 ∧ (nthTriangle n).angleC ≠ 90 :=
sorry

end NUMINAMATH_CALUDE_no_right_angle_in_sequence_l1280_128006


namespace NUMINAMATH_CALUDE_chord_segment_lengths_l1280_128058

theorem chord_segment_lengths (r : ℝ) (chord_length : ℝ) :
  r = 7 ∧ chord_length = 12 →
  ∃ (ak kb : ℝ),
    ak = 7 - Real.sqrt 13 ∧
    kb = 7 + Real.sqrt 13 ∧
    ak + kb = 2 * r :=
by sorry

end NUMINAMATH_CALUDE_chord_segment_lengths_l1280_128058


namespace NUMINAMATH_CALUDE_parallel_vectors_tangent_l1280_128084

theorem parallel_vectors_tangent (θ : ℝ) (a b : ℝ × ℝ) : 
  a = (2, Real.sin θ) → 
  b = (1, Real.cos θ) → 
  (∃ (k : ℝ), a = k • b) → 
  Real.tan θ = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_tangent_l1280_128084


namespace NUMINAMATH_CALUDE_green_beads_count_l1280_128023

/-- The number of green beads in a jewelry pattern -/
def green_beads : ℕ := 3

/-- The number of purple beads in the pattern -/
def purple_beads : ℕ := 5

/-- The number of red beads in the pattern -/
def red_beads : ℕ := 2 * green_beads

/-- The number of times the pattern repeats in a bracelet -/
def bracelet_repeats : ℕ := 3

/-- The number of times the pattern repeats in a necklace -/
def necklace_repeats : ℕ := 5

/-- The total number of beads needed for 1 bracelet and 10 necklaces -/
def total_beads : ℕ := 742

/-- The number of bracelets to be made -/
def num_bracelets : ℕ := 1

/-- The number of necklaces to be made -/
def num_necklaces : ℕ := 10

theorem green_beads_count : 
  num_bracelets * bracelet_repeats * (green_beads + purple_beads + red_beads) + 
  num_necklaces * necklace_repeats * (green_beads + purple_beads + red_beads) = total_beads :=
by sorry

end NUMINAMATH_CALUDE_green_beads_count_l1280_128023


namespace NUMINAMATH_CALUDE_max_value_is_12_l1280_128078

/-- Represents an arithmetic expression using the given operations and numbers -/
inductive Expr
  | num : ℕ → Expr
  | add : Expr → Expr → Expr
  | div : Expr → Expr → Expr
  | mul : Expr → Expr → Expr

/-- Evaluates an arithmetic expression -/
def eval : Expr → ℚ
  | Expr.num n => n
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.div e1 e2 => eval e1 / eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2

/-- Checks if an expression uses the given numbers in order -/
def usesNumbers (e : Expr) (nums : List ℕ) : Prop := sorry

/-- Counts the number of times each operation is used in an expression -/
def countOps (e : Expr) : (ℕ × ℕ × ℕ) := sorry

/-- Checks if an expression uses at most one pair of parentheses -/
def atMostOneParenthesis (e : Expr) : Prop := sorry

/-- The main theorem statement -/
theorem max_value_is_12 :
  ∀ e : Expr,
    usesNumbers e [7, 2, 3, 4] →
    countOps e = (1, 1, 1) →
    atMostOneParenthesis e →
    eval e ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_value_is_12_l1280_128078


namespace NUMINAMATH_CALUDE_undefined_values_l1280_128050

theorem undefined_values (a : ℝ) : 
  (a + 2) / (a^2 - 9) = 0/0 ↔ a = -3 ∨ a = 3 :=
by sorry

end NUMINAMATH_CALUDE_undefined_values_l1280_128050


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l1280_128018

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.1 : ℝ)⌉ = 31 := by sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l1280_128018


namespace NUMINAMATH_CALUDE_land_tax_calculation_l1280_128004

/-- Calculates the land tax for a given plot --/
def calculate_land_tax (area : ℝ) (cadastral_value_per_acre : ℝ) (tax_rate : ℝ) : ℝ :=
  area * cadastral_value_per_acre * tax_rate

/-- Proves that the land tax for the given conditions is 4500 rubles --/
theorem land_tax_calculation :
  let area : ℝ := 15
  let cadastral_value_per_acre : ℝ := 100000
  let tax_rate : ℝ := 0.003
  calculate_land_tax area cadastral_value_per_acre tax_rate = 4500 := by
  sorry

#eval calculate_land_tax 15 100000 0.003

end NUMINAMATH_CALUDE_land_tax_calculation_l1280_128004


namespace NUMINAMATH_CALUDE_correct_sunset_time_l1280_128017

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds a duration to a time -/
def addDuration (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.minutes + d.minutes
  let additionalHours := totalMinutes / 60
  let newMinutes := totalMinutes % 60
  let newHours := (t.hours + d.hours + additionalHours) % 24
  { hours := newHours, minutes := newMinutes }

theorem correct_sunset_time :
  let sunrise : Time := { hours := 6, minutes := 32 }
  let daylight : Duration := { hours := 11, minutes := 35 }
  let sunset := addDuration sunrise daylight
  sunset = { hours := 18, minutes := 7 } :=
by sorry

end NUMINAMATH_CALUDE_correct_sunset_time_l1280_128017


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l1280_128087

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Statement of the problem -/
theorem imaginary_power_sum : i^23 + i^52 + i^103 = 1 - 2*i := by sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l1280_128087


namespace NUMINAMATH_CALUDE_min_four_dollar_frisbees_l1280_128071

/-- Given a total of 64 frisbees sold at either $3 or $4 each, with total receipts of $196,
    the minimum number of $4 frisbees sold is 4. -/
theorem min_four_dollar_frisbees :
  ∀ (x y : ℕ),
    x + y = 64 →
    3 * x + 4 * y = 196 →
    y ≥ 4 ∧ ∃ (z : ℕ), z + 4 = 64 ∧ 3 * z + 4 * 4 = 196 :=
by sorry

end NUMINAMATH_CALUDE_min_four_dollar_frisbees_l1280_128071


namespace NUMINAMATH_CALUDE_dice_probability_l1280_128001

/-- The number of sides on each die -/
def sides : ℕ := 15

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The threshold for "low" numbers -/
def low_threshold : ℕ := 10

/-- The number of low outcomes on a single die -/
def low_outcomes : ℕ := low_threshold - 1

/-- The number of high outcomes on a single die -/
def high_outcomes : ℕ := sides - low_outcomes

/-- The probability of rolling a low number on a single die -/
def prob_low : ℚ := low_outcomes / sides

/-- The probability of rolling a high number on a single die -/
def prob_high : ℚ := high_outcomes / sides

/-- The number of ways to choose 3 dice out of 5 -/
def ways_to_choose : ℕ := (num_dice.choose 3)

theorem dice_probability : 
  (ways_to_choose : ℚ) * prob_low^3 * prob_high^2 = 216/625 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l1280_128001


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1280_128079

theorem sqrt_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c ≥ d) (h4 : d > 0) : 
  Real.sqrt (a / d) > Real.sqrt (b / c) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1280_128079


namespace NUMINAMATH_CALUDE_square_diff_cubed_l1280_128047

theorem square_diff_cubed : (7^2 - 5^2)^3 = 13824 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_cubed_l1280_128047


namespace NUMINAMATH_CALUDE_beans_remaining_fraction_l1280_128022

/-- Given a jar and coffee beans, where:
  1. The weight of the jar is 10% of the total weight when filled with beans.
  2. After removing some beans, the weight of the jar and remaining beans is 60% of the original total weight.
  Prove that the fraction of beans remaining in the jar is 5/9. -/
theorem beans_remaining_fraction (jar_weight : ℝ) (full_beans_weight : ℝ) 
  (remaining_beans_weight : ℝ) : 
  (jar_weight = 0.1 * (jar_weight + full_beans_weight)) →
  (jar_weight + remaining_beans_weight = 0.6 * (jar_weight + full_beans_weight)) →
  (remaining_beans_weight / full_beans_weight = 5 / 9) :=
by sorry

end NUMINAMATH_CALUDE_beans_remaining_fraction_l1280_128022


namespace NUMINAMATH_CALUDE_product_ab_equals_negative_one_l1280_128020

theorem product_ab_equals_negative_one (a b : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → 0 ≤ x^4 - x^3 + a*x + b ∧ x^4 - x^3 + a*x + b ≤ (x^2 - 1)^2) → 
  a * b = -1 := by
sorry

end NUMINAMATH_CALUDE_product_ab_equals_negative_one_l1280_128020


namespace NUMINAMATH_CALUDE_sqrt_neg_five_squared_l1280_128052

theorem sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_five_squared_l1280_128052


namespace NUMINAMATH_CALUDE_minimum_distances_to_pond_l1280_128063

/-- Represents a point in 2D space -/
structure Point where
  x : Int
  y : Int

/-- Represents a walk in cardinal directions -/
inductive Walk
  | North : Nat → Walk
  | South : Nat → Walk
  | East : Nat → Walk
  | West : Nat → Walk

/-- Calculates the end point after a series of walks -/
def end_point (start : Point) (walks : List Walk) : Point :=
  walks.foldl
    (fun p w =>
      match w with
      | Walk.North n => { x := p.x, y := p.y + n }
      | Walk.South n => { x := p.x, y := p.y - n }
      | Walk.East n => { x := p.x + n, y := p.y }
      | Walk.West n => { x := p.x - n, y := p.y })
    start

/-- Calculates the Manhattan distance between two points -/
def manhattan_distance (p1 p2 : Point) : Nat :=
  (p1.x - p2.x).natAbs + (p1.y - p2.y).natAbs

/-- Anička's initial walk -/
def anicka_walk : List Walk :=
  [Walk.North 5, Walk.East 2, Walk.South 3, Walk.West 4]

/-- Vojta's initial walk -/
def vojta_walk : List Walk :=
  [Walk.South 3, Walk.West 4, Walk.North 1]

theorem minimum_distances_to_pond :
  let anicka_start : Point := { x := 0, y := 0 }
  let vojta_start : Point := { x := 0, y := 0 }
  let pond := end_point anicka_start anicka_walk
  let vojta_end := end_point vojta_start vojta_walk
  vojta_end.x + 5 = pond.x →
  manhattan_distance anicka_start pond = 4 ∧
  manhattan_distance vojta_start pond = 3 :=
by sorry


end NUMINAMATH_CALUDE_minimum_distances_to_pond_l1280_128063


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l1280_128081

def normal_distribution (μ σ : ℝ) (X : ℝ → ℝ) : Prop :=
  ∃ f : ℝ → ℝ, ∀ x, f x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

theorem normal_distribution_probability (X : ℝ → ℝ) (μ σ : ℝ) :
  normal_distribution μ σ X →
  (∫ x in Set.Ioo (μ - 2*σ) (μ + 2*σ), X x) = 0.9544 →
  (∫ x in Set.Ioo (μ - σ) (μ + σ), X x) = 0.6826 →
  μ = 4 →
  σ = 1 →
  (∫ x in Set.Ioo 5 6, X x) = 0.1359 :=
by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l1280_128081


namespace NUMINAMATH_CALUDE_max_colored_cells_4x3000_exists_optimal_board_l1280_128021

/-- A tetromino is a geometric shape composed of four square cells connected orthogonally. -/
def Tetromino : Type := Unit

/-- A board is represented as a 2D array of boolean values, where true represents a colored cell. -/
def Board : Type := Array (Array Bool)

/-- Check if a given board contains a tetromino. -/
def containsTetromino (board : Board) : Bool :=
  sorry

/-- Count the number of colored cells in a board. -/
def countColoredCells (board : Board) : Nat :=
  sorry

/-- Create a 4 × 3000 board. -/
def create4x3000Board : Board :=
  sorry

/-- The main theorem stating the maximum number of cells that can be colored. -/
theorem max_colored_cells_4x3000 :
  ∀ (board : Board),
    board = create4x3000Board →
    ¬containsTetromino board →
    countColoredCells board ≤ 7000 :=
  sorry

/-- The existence of a board with exactly 7000 colored cells and no tetromino. -/
theorem exists_optimal_board :
  ∃ (board : Board),
    board = create4x3000Board ∧
    ¬containsTetromino board ∧
    countColoredCells board = 7000 :=
  sorry

end NUMINAMATH_CALUDE_max_colored_cells_4x3000_exists_optimal_board_l1280_128021


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1280_128039

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (x^2 - x) / (x^2 + 2*x + 1) / ((x - 1) / (x + 1)) = x / (x + 1) ∧
  (3^2 - 3) / (3^2 + 2*3 + 1) / ((3 - 1) / (3 + 1)) = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1280_128039


namespace NUMINAMATH_CALUDE_product_of_numbers_l1280_128025

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 20) (h2 : (x - y)^2 = 16) : x * y = 96 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1280_128025


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1280_128015

/-- The sum of a geometric series with first term 3, common ratio -2, and last term 768 is 513. -/
theorem geometric_series_sum : 
  ∀ (n : ℕ) (a : ℝ) (r : ℝ) (S : ℝ),
  a = 3 →
  r = -2 →
  a * r^(n-1) = 768 →
  S = (a * (1 - r^n)) / (1 - r) →
  S = 513 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1280_128015


namespace NUMINAMATH_CALUDE_function_properties_l1280_128045

open Set

def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
  (h_even : IsEven f)
  (h_periodic : IsPeriodic f 2)
  (h_interval : ∀ x ∈ Icc 1 2, f x = x^2 + 2*x - 1) :
  ∀ x ∈ Icc 0 1, f x = x^2 - 6*x + 7 := by
sorry

end NUMINAMATH_CALUDE_function_properties_l1280_128045


namespace NUMINAMATH_CALUDE_log_product_equals_three_eighths_l1280_128082

theorem log_product_equals_three_eighths :
  (1/2) * (Real.log 3 / Real.log 2) * (1/2) * (Real.log 8 / Real.log 9) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_three_eighths_l1280_128082


namespace NUMINAMATH_CALUDE_tip_percentage_is_twenty_percent_l1280_128036

def lunch_cost : ℝ := 60.80
def total_spent : ℝ := 72.96

theorem tip_percentage_is_twenty_percent :
  (total_spent - lunch_cost) / lunch_cost * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_tip_percentage_is_twenty_percent_l1280_128036


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1280_128098

/-- Given an infinite geometric series with a specific pattern, prove the value of k that makes the series sum to 10 -/
theorem geometric_series_sum (k : ℝ) : 
  (∑' n : ℕ, (4 + n * k) / 5^n) = 10 → k = 19.2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1280_128098


namespace NUMINAMATH_CALUDE_bus_people_count_l1280_128030

/-- Represents the number of people who got off the bus -/
def people_off : ℕ := 47

/-- Represents the number of people remaining on the bus -/
def people_remaining : ℕ := 43

/-- Represents the total number of people on the bus before -/
def total_people : ℕ := people_off + people_remaining

theorem bus_people_count : total_people = 90 := by
  sorry

end NUMINAMATH_CALUDE_bus_people_count_l1280_128030


namespace NUMINAMATH_CALUDE_sin_translation_left_l1280_128028

/-- Translating the graph of y = sin(2x) to the left by π/3 units results in y = sin(2x + 2π/3) -/
theorem sin_translation_left (x : ℝ) : 
  let f (t : ℝ) := Real.sin (2 * t)
  let g (t : ℝ) := f (t + π/3)
  g x = Real.sin (2 * x + 2 * π/3) := by
  sorry

end NUMINAMATH_CALUDE_sin_translation_left_l1280_128028


namespace NUMINAMATH_CALUDE_negation_existence_l1280_128024

theorem negation_existence (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ ¬(∀ x : ℝ, x^2 - a*x + 1 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_existence_l1280_128024


namespace NUMINAMATH_CALUDE_guard_arrangement_exists_l1280_128093

/-- Represents a guard with a position and direction of sight -/
structure Guard where
  position : ℝ × ℝ
  direction : ℝ × ℝ

/-- Represents the arrangement of guards around a point object -/
structure GuardArrangement where
  guards : List Guard
  object : ℝ × ℝ
  visibility_range : ℝ

/-- Predicate to check if a point is inside or on the boundary of a convex hull -/
def is_inside_or_on_convex_hull (point : ℝ × ℝ) (hull : List (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if a list of points forms a convex hull -/
def is_convex_hull (points : List (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if it's impossible to approach any point unnoticed -/
def is_approach_impossible (arrangement : GuardArrangement) : Prop :=
  sorry

/-- Theorem stating that it's possible to arrange guards to prevent unnoticed approach -/
theorem guard_arrangement_exists : ∃ (arrangement : GuardArrangement),
  arrangement.visibility_range = 100 ∧
  arrangement.guards.length ≥ 6 ∧
  is_convex_hull (arrangement.guards.map Guard.position) ∧
  is_inside_or_on_convex_hull arrangement.object (arrangement.guards.map Guard.position) ∧
  is_approach_impossible arrangement :=
by
  sorry

end NUMINAMATH_CALUDE_guard_arrangement_exists_l1280_128093


namespace NUMINAMATH_CALUDE_high_school_sample_size_l1280_128088

/-- Prove that in a high school with the given student distribution and selection probability, the sample size is 200. -/
theorem high_school_sample_size
  (freshmen : ℕ) (sophomores : ℕ) (juniors : ℕ) (prob : ℝ)
  (h1 : freshmen = 400)
  (h2 : sophomores = 320)
  (h3 : juniors = 280)
  (h4 : prob = 0.2)
  : ℕ :=
by
  sorry

#check high_school_sample_size

end NUMINAMATH_CALUDE_high_school_sample_size_l1280_128088


namespace NUMINAMATH_CALUDE_inlet_fill_rate_l1280_128048

/-- The rate at which the inlet pipe fills the tank, given the tank's capacity,
    leak emptying time, and combined emptying time with inlet open. -/
theorem inlet_fill_rate (capacity : ℝ) (leak_empty_time : ℝ) (combined_empty_time : ℝ) :
  capacity = 5760 →
  leak_empty_time = 6 →
  combined_empty_time = 8 →
  (capacity / leak_empty_time) - (capacity / combined_empty_time) = 240 := by
  sorry

end NUMINAMATH_CALUDE_inlet_fill_rate_l1280_128048


namespace NUMINAMATH_CALUDE_solutions_of_quadratic_equation_l1280_128073

theorem solutions_of_quadratic_equation :
  ∀ x : ℝ, x * (2 * x + 1) = 0 ↔ x = 0 ∨ x = -1/2 := by sorry

end NUMINAMATH_CALUDE_solutions_of_quadratic_equation_l1280_128073


namespace NUMINAMATH_CALUDE_factorial_divisibility_l1280_128086

theorem factorial_divisibility (p : ℕ) (h : Prime p) :
  ∃ k : ℕ, (p^2).factorial = k * (p.factorial ^ (p + 1)) :=
sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l1280_128086


namespace NUMINAMATH_CALUDE_days_to_pay_cash_register_l1280_128032

/-- Represents the financial data for Marie's bakery --/
structure BakeryFinances where
  cash_register_cost : ℕ
  bread_price : ℕ
  bread_quantity : ℕ
  cake_price : ℕ
  cake_quantity : ℕ
  daily_rent : ℕ
  daily_electricity : ℕ

/-- Calculates the number of days required to pay for the cash register --/
def days_to_pay (finances : BakeryFinances) : ℕ :=
  let daily_income := finances.bread_price * finances.bread_quantity + finances.cake_price * finances.cake_quantity
  let daily_expenses := finances.daily_rent + finances.daily_electricity
  let daily_profit := daily_income - daily_expenses
  finances.cash_register_cost / daily_profit

/-- Theorem stating that it takes 8 days to pay for the cash register --/
theorem days_to_pay_cash_register :
  let maries_finances : BakeryFinances := {
    cash_register_cost := 1040,
    bread_price := 2,
    bread_quantity := 40,
    cake_price := 12,
    cake_quantity := 6,
    daily_rent := 20,
    daily_electricity := 2
  }
  days_to_pay maries_finances = 8 := by
  sorry


end NUMINAMATH_CALUDE_days_to_pay_cash_register_l1280_128032


namespace NUMINAMATH_CALUDE_min_value_a2_b2_l1280_128077

/-- A quadratic function f(x) = ax^2 + (2b+1)x - a - 2 -/
def f (a b x : ℝ) : ℝ := a * x^2 + (2*b + 1) * x - a - 2

/-- The theorem stating the minimum value of a^2 + b^2 -/
theorem min_value_a2_b2 (a b : ℝ) (ha : a ≠ 0) :
  (∃ x ∈ Set.Icc 3 4, f a b x = 0) →
  ∃ m : ℝ, m = 1/100 ∧ ∀ a' b' : ℝ, (∃ x ∈ Set.Icc 3 4, f a' b' x = 0) → a'^2 + b'^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_a2_b2_l1280_128077


namespace NUMINAMATH_CALUDE_exact_three_wins_probability_l1280_128070

/-- The probability of winning a prize in a single draw -/
def p : ℚ := 2/5

/-- The number of participants (trials) -/
def n : ℕ := 4

/-- The number of desired successes -/
def k : ℕ := 3

/-- The probability of exactly k successes in n independent trials 
    with probability p of success in each trial -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem exact_three_wins_probability :
  binomial_probability n k p = 96/625 := by
  sorry

end NUMINAMATH_CALUDE_exact_three_wins_probability_l1280_128070


namespace NUMINAMATH_CALUDE_modulus_of_z_l1280_128046

-- Define the complex number z
def z : ℂ := Complex.I * (1 - Complex.I)

-- Theorem stating that the modulus of z is √2
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1280_128046


namespace NUMINAMATH_CALUDE_probability_at_least_one_event_l1280_128066

theorem probability_at_least_one_event (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/2) (h2 : p2 = 1/3) (h3 : p3 = 1/4) : 
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_event_l1280_128066


namespace NUMINAMATH_CALUDE_jack_deer_hunting_l1280_128092

/-- The number of times Jack goes hunting per month -/
def hunts_per_month : ℕ := 6

/-- The duration of the hunting season in months -/
def hunting_season_months : ℕ := 3

/-- The number of deer Jack catches per hunting trip -/
def deer_per_hunt : ℕ := 2

/-- The weight of each deer in pounds -/
def deer_weight : ℕ := 600

/-- The fraction of deer weight Jack keeps -/
def kept_fraction : ℚ := 1 / 2

/-- The total amount of deer Jack keeps in pounds -/
def deer_kept : ℕ := 10800

theorem jack_deer_hunting :
  hunts_per_month * hunting_season_months * deer_per_hunt * deer_weight * kept_fraction = deer_kept := by
  sorry

end NUMINAMATH_CALUDE_jack_deer_hunting_l1280_128092


namespace NUMINAMATH_CALUDE_transformed_area_is_450_l1280_128008

-- Define the transformation matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; -8, 6]

-- Define the original area
def original_area : ℝ := 9

-- Theorem statement
theorem transformed_area_is_450 :
  let det := A.det
  let new_area := original_area * |det|
  new_area = 450 := by sorry

end NUMINAMATH_CALUDE_transformed_area_is_450_l1280_128008


namespace NUMINAMATH_CALUDE_jenny_ate_65_chocolates_l1280_128072

/-- The number of chocolate squares Mike ate -/
def mike_chocolates : ℕ := 20

/-- The number of chocolate squares Jenny ate -/
def jenny_chocolates : ℕ := 3 * mike_chocolates + 5

theorem jenny_ate_65_chocolates : jenny_chocolates = 65 := by
  sorry

end NUMINAMATH_CALUDE_jenny_ate_65_chocolates_l1280_128072


namespace NUMINAMATH_CALUDE_camp_children_count_l1280_128012

/-- The number of children currently in the camp -/
def current_children : ℕ := 25

/-- The percentage of boys currently in the camp -/
def boys_percentage : ℚ := 85/100

/-- The number of boys to be added -/
def boys_added : ℕ := 50

/-- The desired percentage of girls after adding boys -/
def desired_girls_percentage : ℚ := 5/100

theorem camp_children_count :
  (boys_percentage * current_children).num = 
    (desired_girls_percentage * (current_children + boys_added)).num * 
    ((1 - boys_percentage) * current_children).den := by sorry

end NUMINAMATH_CALUDE_camp_children_count_l1280_128012


namespace NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l1280_128044

/-- Given a point A with coordinates (3, y) and its reflection B over the x-axis,
    prove that the sum of all coordinates of A and B is 6. -/
theorem sum_of_coordinates_after_reflection (y : ℝ) : 
  let A : ℝ × ℝ := (3, y)
  let B : ℝ × ℝ := (3, -y)  -- reflection of A over x-axis
  (A.1 + A.2 + B.1 + B.2) = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l1280_128044


namespace NUMINAMATH_CALUDE_find_number_l1280_128064

theorem find_number : ∃ n : ℕ, n + 3427 = 13200 ∧ n = 9773 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1280_128064


namespace NUMINAMATH_CALUDE_sin_seven_pi_sixths_l1280_128083

theorem sin_seven_pi_sixths : Real.sin (7 * Real.pi / 6) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_seven_pi_sixths_l1280_128083


namespace NUMINAMATH_CALUDE_adjacent_i_probability_is_one_fifth_l1280_128041

/-- The probability of forming a 10-letter code with two adjacent i's -/
def adjacent_i_probability : ℚ :=
  let total_arrangements := Nat.factorial 10
  let favorable_arrangements := Nat.factorial 9 * Nat.factorial 2
  favorable_arrangements / total_arrangements

/-- Theorem stating that the probability of forming a 10-letter code
    with two adjacent i's is 1/5 -/
theorem adjacent_i_probability_is_one_fifth :
  adjacent_i_probability = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_i_probability_is_one_fifth_l1280_128041


namespace NUMINAMATH_CALUDE_certain_number_proof_l1280_128065

theorem certain_number_proof (x N : ℝ) 
  (h1 : 3 * x = (N - x) + 14) 
  (h2 : x = 10) : 
  N = 26 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1280_128065


namespace NUMINAMATH_CALUDE_tomato_multiple_l1280_128053

theorem tomato_multiple : 
  ∀ (before_vacation after_growth : ℕ),
    before_vacation = 36 →
    after_growth = 3564 →
    (before_vacation + after_growth) / before_vacation = 100 := by
  sorry

end NUMINAMATH_CALUDE_tomato_multiple_l1280_128053


namespace NUMINAMATH_CALUDE_tangent_line_at_P_tangent_line_not_at_P_l1280_128002

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Theorem for the first part
theorem tangent_line_at_P :
  ∃ (l : ℝ → ℝ), (l 1 = -2) ∧ 
  (∀ x : ℝ, l x = -2) ∧
  (∀ x : ℝ, x ≠ 1 → (l x - f x) / (x - 1) ≤ (l 1 - f 1) / (1 - 1)) :=
sorry

-- Theorem for the second part
theorem tangent_line_not_at_P :
  ∃ (l : ℝ → ℝ), (l 1 = -2) ∧ 
  (∀ x : ℝ, 9*x + 4*(l x) - 1 = 0) ∧
  (∃ x₀ : ℝ, x₀ ≠ 1 ∧ 
    (∀ x : ℝ, x ≠ x₀ → (l x - f x) / (x - x₀) ≤ (l x₀ - f x₀) / (x₀ - x₀))) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_P_tangent_line_not_at_P_l1280_128002


namespace NUMINAMATH_CALUDE_dodecahedron_triangles_l1280_128059

/-- Represents a dodecahedron -/
structure Dodecahedron where
  num_faces : ℕ
  faces_are_pentagonal : num_faces = 12
  vertices_per_face : ℕ
  vertices_shared_by_three_faces : vertices_per_face = 3

/-- Calculates the number of vertices in a dodecahedron -/
def num_vertices (d : Dodecahedron) : ℕ := 20

/-- Calculates the number of triangles that can be formed using the vertices of a dodecahedron -/
def num_triangles (d : Dodecahedron) : ℕ := (num_vertices d).choose 3

/-- Theorem: The number of triangles that can be formed using the vertices of a dodecahedron is 1140 -/
theorem dodecahedron_triangles (d : Dodecahedron) : num_triangles d = 1140 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_triangles_l1280_128059


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1280_128080

def A : Set ℝ := {x | x^2 - 16 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

theorem intersection_of_A_and_B : 
  A ∩ B = {x | -4 < x ∧ x < 1 ∨ 3 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1280_128080


namespace NUMINAMATH_CALUDE_andrew_appointments_l1280_128029

/-- Calculates the number of 3-hour appointments given total work hours, permits stamped per hour, and total permits stamped. -/
def appointments (total_hours : ℕ) (permits_per_hour : ℕ) (total_permits : ℕ) : ℕ :=
  (total_hours - (total_permits / permits_per_hour)) / 3

/-- Theorem stating that given the problem conditions, Andrew has 2 appointments. -/
theorem andrew_appointments : appointments 8 50 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_andrew_appointments_l1280_128029


namespace NUMINAMATH_CALUDE_max_grandchildren_problem_l1280_128027

/-- The number of children Max has -/
def max_children : ℕ := 8

/-- The number of Max's children who have the same number of children as Max -/
def children_with_same : ℕ := 6

/-- The total number of Max's grandchildren -/
def total_grandchildren : ℕ := 58

/-- The number of children each exception has -/
def exception_children : ℕ := 5

theorem max_grandchildren_problem :
  (children_with_same * max_children) + 
  (2 * exception_children) = total_grandchildren :=
by sorry

end NUMINAMATH_CALUDE_max_grandchildren_problem_l1280_128027


namespace NUMINAMATH_CALUDE_proportion_problem_l1280_128051

theorem proportion_problem (y : ℝ) : (0.75 : ℝ) / 2 = y / 8 → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l1280_128051


namespace NUMINAMATH_CALUDE_parallel_lines_slope_l1280_128054

/-- Theorem: For three parallel lines with y-intercepts 2, 3, and 4, 
    if the sum of their x-intercepts is 36, then their slope is -1/4. -/
theorem parallel_lines_slope (m : ℝ) 
  (h1 : m * (-2/m) + m * (-3/m) + m * (-4/m) = 36) : m = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_slope_l1280_128054


namespace NUMINAMATH_CALUDE_rectangle_length_equals_eight_l1280_128075

theorem rectangle_length_equals_eight
  (square_perimeter : ℝ)
  (rectangle_width : ℝ)
  (triangle_height : ℝ)
  (h1 : square_perimeter = 64)
  (h2 : rectangle_width = 8)
  (h3 : triangle_height = 64)
  (h4 : (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_height * rectangle_length) :
  rectangle_length = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_equals_eight_l1280_128075


namespace NUMINAMATH_CALUDE_inequality_implication_l1280_128061

theorem inequality_implication (a b c : ℝ) : a > b → a * c^2 ≥ b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1280_128061


namespace NUMINAMATH_CALUDE_pipe_cut_theorem_l1280_128056

/-- Given a pipe of length 68 feet cut into two pieces, where one piece is 12 feet shorter than the other, 
    the length of the shorter piece is 28 feet. -/
theorem pipe_cut_theorem : 
  ∀ (shorter_piece longer_piece : ℝ),
  shorter_piece + longer_piece = 68 →
  longer_piece = shorter_piece + 12 →
  shorter_piece = 28 := by
sorry

end NUMINAMATH_CALUDE_pipe_cut_theorem_l1280_128056


namespace NUMINAMATH_CALUDE_semicircle_rectangle_property_l1280_128095

-- Define the semicircle and its properties
structure Semicircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the rectangle
structure Rectangle where
  base : ℝ
  height : ℝ

-- Define a point on the semicircle
def PointOnSemicircle (s : Semicircle) := { p : ℝ × ℝ // (p.1 - s.center.1)^2 + (p.2 - s.center.2)^2 = s.radius^2 ∧ p.2 ≥ s.center.2 }

-- Theorem statement
theorem semicircle_rectangle_property
  (s : Semicircle)
  (r : Rectangle)
  (h_square : r.height = s.radius / Real.sqrt 2)  -- Height equals side of inscribed square
  (h_base : r.base = 2 * s.radius)  -- Base is diameter
  (M : PointOnSemicircle s)
  (E F : ℝ)  -- E and F are x-coordinates on the diameter
  (h_E : E ∈ Set.Icc s.center.1 (s.center.1 + s.radius))
  (h_F : F ∈ Set.Icc s.center.1 (s.center.1 + s.radius))
  : (F - s.center.1)^2 + (s.center.1 + 2*s.radius - E)^2 = (2*s.radius)^2 := by
  sorry


end NUMINAMATH_CALUDE_semicircle_rectangle_property_l1280_128095


namespace NUMINAMATH_CALUDE_circle_and_symmetry_l1280_128067

-- Define the circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define the chord line
def ChordLine (x : ℝ) := x + 1

-- Define the variable line
def VariableLine (k : ℝ) (x : ℝ) := k * (x - 1)

-- Define the fixed point N
def N : ℝ × ℝ := (4, 0)

-- Main theorem
theorem circle_and_symmetry :
  -- The chord intercepted by y = x + 1 has length √14
  (∃ (a b : ℝ), (a, ChordLine a) ∈ Circle ∧ (b, ChordLine b) ∈ Circle ∧ (b - a)^2 + (ChordLine b - ChordLine a)^2 = 14) →
  -- The equation of the circle is x² + y² = 4
  (∀ (x y : ℝ), (x, y) ∈ Circle ↔ x^2 + y^2 = 4) ∧
  -- N is the fixed point of symmetry
  (∀ (k : ℝ) (A B : ℝ × ℝ),
    k ≠ 0 →
    A ∈ Circle →
    B ∈ Circle →
    A.2 = VariableLine k A.1 →
    B.2 = VariableLine k B.1 →
    (A.2 / (A.1 - N.1) + B.2 / (B.1 - N.1) = 0)) :=
by sorry


end NUMINAMATH_CALUDE_circle_and_symmetry_l1280_128067


namespace NUMINAMATH_CALUDE_inner_cube_surface_area_l1280_128033

/-- Given a cube of volume 64 cubic meters with an inscribed sphere, which in turn has an inscribed cube, 
    the surface area of the inner cube is 32 square meters. -/
theorem inner_cube_surface_area (outer_cube : Real → Real → Real → Bool) 
  (outer_sphere : Real → Real → Real → Bool) (inner_cube : Real → Real → Real → Bool) :
  (∀ x y z, outer_cube x y z ↔ (0 ≤ x ∧ x ≤ 4) ∧ (0 ≤ y ∧ y ≤ 4) ∧ (0 ≤ z ∧ z ≤ 4)) →
  (∀ x y z, outer_sphere x y z ↔ (x - 2)^2 + (y - 2)^2 + (z - 2)^2 ≤ 4) →
  (∀ x y z, inner_cube x y z → outer_sphere x y z) →
  (∃! l : Real, ∀ x y z, inner_cube x y z ↔ 
    (0 ≤ x ∧ x ≤ l) ∧ (0 ≤ y ∧ y ≤ l) ∧ (0 ≤ z ∧ z ≤ l) ∧ l^2 + l^2 + l^2 = 16) →
  (∃ sa : Real, sa = 6 * (4 * Real.sqrt 3 / 3)^2 ∧ sa = 32) :=
by sorry

end NUMINAMATH_CALUDE_inner_cube_surface_area_l1280_128033


namespace NUMINAMATH_CALUDE_divisibility_implies_inequality_l1280_128007

theorem divisibility_implies_inequality (k m : ℕ+) (h1 : k > m) 
  (h2 : (k^3 - m^3) ∣ (k * m * (k^2 - m^2))) : 
  (k - m)^3 > 3 * k * m := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_inequality_l1280_128007


namespace NUMINAMATH_CALUDE_complementary_angles_l1280_128099

theorem complementary_angles (A B : ℝ) : 
  A + B = 90 →  -- angles are complementary
  A = 5 * B →   -- measure of A is 5 times B
  A = 75 :=     -- measure of A is 75 degrees
by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_l1280_128099


namespace NUMINAMATH_CALUDE_white_animals_count_l1280_128005

theorem white_animals_count (total : ℕ) (black : ℕ) (white : ℕ) : 
  total = 13 → black = 6 → white = total - black → white = 7 := by sorry

end NUMINAMATH_CALUDE_white_animals_count_l1280_128005


namespace NUMINAMATH_CALUDE_shorter_can_radius_l1280_128090

/-- Given two cylindrical cans with equal volume, where one can's height is twice 
    the other's and the taller can's radius is 10 units, the radius of the shorter 
    can is 10√2 units. -/
theorem shorter_can_radius (h : ℝ) (r : ℝ) : 
  h > 0 → -- height is positive
  π * (10^2) * (2*h) = π * r^2 * h → -- volumes are equal
  r = 10 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_shorter_can_radius_l1280_128090


namespace NUMINAMATH_CALUDE_cos_equality_exists_l1280_128031

theorem cos_equality_exists (n : ℤ) : ∃ n, 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (315 * π / 180) ∧ n = 45 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_exists_l1280_128031


namespace NUMINAMATH_CALUDE_triangle_height_l1280_128014

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) :
  area = 46 →
  base = 10 →
  area = (base * height) / 2 →
  height = 9.2 := by
sorry

end NUMINAMATH_CALUDE_triangle_height_l1280_128014


namespace NUMINAMATH_CALUDE_pascal_triangle_30th_row_28th_number_l1280_128097

/-- The number of elements in the row of Pascal's triangle we're considering -/
def row_length : ℕ := 30

/-- The position of the number we're looking for (1-indexed) -/
def target_position : ℕ := 28

/-- The value we're proving the target position contains -/
def target_value : ℕ := 406

theorem pascal_triangle_30th_row_28th_number :
  Nat.choose (row_length - 1) (target_position - 1) = target_value := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_30th_row_28th_number_l1280_128097


namespace NUMINAMATH_CALUDE_linear_function_properties_l1280_128013

/-- A linear function passing through points (3,5) and (-4,-9) -/
def f (x : ℝ) : ℝ := 2 * x - 1

theorem linear_function_properties :
  (∃ k b : ℝ, ∀ x, f x = k * x + b) ∧
  f 3 = 5 ∧
  f (-4) = -9 ∧
  f 0 = -1 ∧
  f (1/2) = 0 ∧
  (1/2 * |f 0|) / 2 = 1/4 ∧
  (∀ a : ℝ, f a = 2 → a = 3/2) :=
sorry

end NUMINAMATH_CALUDE_linear_function_properties_l1280_128013
