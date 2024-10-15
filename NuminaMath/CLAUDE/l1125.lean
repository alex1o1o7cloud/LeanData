import Mathlib

namespace NUMINAMATH_CALUDE_sandals_sold_l1125_112560

theorem sandals_sold (shoes : ℕ) (sandals : ℕ) : 
  (shoes : ℚ) / sandals = 15 / 8 → shoes = 135 → sandals = 72 := by
sorry

end NUMINAMATH_CALUDE_sandals_sold_l1125_112560


namespace NUMINAMATH_CALUDE_unique_four_digit_int_l1125_112594

/-- Represents a four-digit positive integer -/
structure FourDigitInt where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_pos : a > 0
  a_lt_10 : a < 10
  b_lt_10 : b < 10
  c_lt_10 : c < 10
  d_lt_10 : d < 10

def to_int (n : FourDigitInt) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

theorem unique_four_digit_int :
  ∃! (n : FourDigitInt),
    n.a + n.b + n.c + n.d = 16 ∧
    n.b + n.c = 10 ∧
    n.a - n.d = 2 ∧
    (to_int n) % 9 = 0 ∧
    to_int n = 4622 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_int_l1125_112594


namespace NUMINAMATH_CALUDE_simultaneous_inequalities_l1125_112522

theorem simultaneous_inequalities (a b : ℝ) :
  (a > b ∧ 1 / a > 1 / b) ↔ (a > 0 ∧ 0 > b) :=
sorry

end NUMINAMATH_CALUDE_simultaneous_inequalities_l1125_112522


namespace NUMINAMATH_CALUDE_complex_arithmetic_result_l1125_112502

theorem complex_arithmetic_result :
  let A : ℂ := 5 - 2*I
  let M : ℂ := -3 + 3*I
  let S : ℂ := 2*I
  let P : ℝ := (1/2 : ℝ)
  A - M + S - (P : ℂ) = 7.5 - 3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_result_l1125_112502


namespace NUMINAMATH_CALUDE_weights_division_impossibility_l1125_112547

theorem weights_division_impossibility : 
  let weights : List Nat := List.range 23
  let total_sum : Nat := (weights.sum + 23) - 21
  ¬ ∃ (half : Nat), 2 * half = total_sum
  := by sorry

end NUMINAMATH_CALUDE_weights_division_impossibility_l1125_112547


namespace NUMINAMATH_CALUDE_star_calculation_l1125_112568

-- Define the new operation
def star (m n : Int) : Int := m - n + 1

-- Theorem statement
theorem star_calculation : star (star 2 3) 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l1125_112568


namespace NUMINAMATH_CALUDE_right_triangle_area_l1125_112587

theorem right_triangle_area (a b : ℝ) (ha : a = 45) (hb : b = 48) :
  (1 / 2 : ℝ) * a * b = 1080 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1125_112587


namespace NUMINAMATH_CALUDE_find_a_value_l1125_112527

theorem find_a_value (a : ℝ) : 
  let A := {x : ℝ | x^2 - a*x + a^2 - 19 = 0}
  let B := {x : ℝ | x^2 - 5*x + 6 = 0}
  let C := {x : ℝ | x^2 + 2*x - 8 = 0}
  (∃ x, x ∈ A ∩ B) ∧ (A ∩ C = ∅) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l1125_112527


namespace NUMINAMATH_CALUDE_blue_red_ratio_13_l1125_112599

/-- Represents the ratio of blue to red face areas in a cube cutting problem -/
def blue_to_red_ratio (n : ℕ) : ℚ :=
  (6 * n^3 - 6 * n^2) / (6 * n^2)

/-- Theorem stating that for a cube of side length 13, the ratio of blue to red face areas is 12 -/
theorem blue_red_ratio_13 : blue_to_red_ratio 13 = 12 := by
  sorry

#eval blue_to_red_ratio 13

end NUMINAMATH_CALUDE_blue_red_ratio_13_l1125_112599


namespace NUMINAMATH_CALUDE_first_five_valid_numbers_l1125_112569

def random_table : List (List Nat) := [
  [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 06, 76],
  [63, 01, 63, 78, 59, 16, 95, 56, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79],
  [33, 21, 12, 34, 29, 78, 64, 56, 07, 82, 52, 42, 07, 44, 38, 15, 51, 00, 13, 42, 99, 66, 02, 79, 54]
]

def start_row : Nat := 8
def start_col : Nat := 7
def max_bag_number : Nat := 799

def is_valid_number (n : Nat) : Bool :=
  n <= max_bag_number

def find_valid_numbers (table : List (List Nat)) (row : Nat) (col : Nat) (count : Nat) : List Nat :=
  sorry

theorem first_five_valid_numbers :
  find_valid_numbers random_table start_row start_col 5 = [785, 667, 199, 507, 175] :=
sorry

end NUMINAMATH_CALUDE_first_five_valid_numbers_l1125_112569


namespace NUMINAMATH_CALUDE_proportion_problem_l1125_112530

/-- Given that a, d, b, c are in proportion, where a = 3 cm, b = 4 cm, and c = 6 cm, prove that d = 9/2 cm. -/
theorem proportion_problem (a d b c : ℚ) : 
  a = 3 → b = 4 → c = 6 → (a / d = b / c) → d = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l1125_112530


namespace NUMINAMATH_CALUDE_set_B_equality_l1125_112561

def A : Set ℤ := {-1, 0, 1}

def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem set_B_equality : B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_set_B_equality_l1125_112561


namespace NUMINAMATH_CALUDE_haley_extra_tickets_l1125_112526

/-- The number of extra concert tickets Haley bought -/
def extra_tickets (ticket_price : ℕ) (tickets_for_friends : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent / ticket_price) - tickets_for_friends

/-- Theorem: Haley bought 5 extra tickets -/
theorem haley_extra_tickets :
  extra_tickets 4 3 32 = 5 := by
  sorry

end NUMINAMATH_CALUDE_haley_extra_tickets_l1125_112526


namespace NUMINAMATH_CALUDE_min_sum_of_prime_factors_l1125_112506

theorem min_sum_of_prime_factors (x : ℕ) : 
  let sequence_sum := 25 * (x + 12)
  ∃ (p₁ p₂ p₃ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ sequence_sum = p₁ * p₂ * p₃ →
  ∀ (q₁ q₂ q₃ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ sequence_sum = q₁ * q₂ * q₃ →
  q₁ + q₂ + q₃ ≥ 23 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_prime_factors_l1125_112506


namespace NUMINAMATH_CALUDE_perimeter_after_increase_l1125_112592

/-- Represents a triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_pos : 0 < a ∧ 0 < b ∧ 0 < c
  h_ineq : a < b + c ∧ b < a + c ∧ c < a + b

/-- The perimeter of a triangle. -/
def Triangle.perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Given a triangle, returns a new triangle with two sides increased by 4 and one by 1. -/
def increaseSides (t : Triangle) : Triangle where
  a := t.a + 4
  b := t.b + 4
  c := t.c + 1
  h_pos := sorry
  h_ineq := sorry

theorem perimeter_after_increase (t : Triangle) 
    (h1 : t.a = 8)
    (h2 : t.b = 5)
    (h3 : t.c = 6) :
    (increaseSides t).perimeter = 28 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_after_increase_l1125_112592


namespace NUMINAMATH_CALUDE_faster_train_speed_l1125_112583

/-- The speed of the faster train given two trains moving in opposite directions --/
theorem faster_train_speed
  (slow_speed : ℝ)
  (length_slow : ℝ)
  (length_fast : ℝ)
  (crossing_time : ℝ)
  (h_slow_speed : slow_speed = 60)
  (h_length_slow : length_slow = 1.10)
  (h_length_fast : length_fast = 0.9)
  (h_crossing_time : crossing_time = 47.99999999999999 / 3600) :
  ∃ (fast_speed : ℝ), fast_speed = 90 ∧
    (fast_speed + slow_speed) * crossing_time = length_slow + length_fast :=
by sorry

end NUMINAMATH_CALUDE_faster_train_speed_l1125_112583


namespace NUMINAMATH_CALUDE_line_direction_vector_k_l1125_112537

/-- A line passing through two points with a specific direction vector form -/
def Line (p1 p2 : ℝ × ℝ) (k : ℝ) : Prop :=
  let dir := (p2.1 - p1.1, p2.2 - p1.2)
  ∃ (t : ℝ), dir = (3 * t, k * t)

/-- The main theorem stating that k = -3 for the given line -/
theorem line_direction_vector_k (k : ℝ) : 
  Line (2, -1) (-4, 5) k → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_k_l1125_112537


namespace NUMINAMATH_CALUDE_students_walking_home_l1125_112523

theorem students_walking_home (bus_fraction automobile_fraction bicycle_fraction skateboard_fraction : ℚ) :
  bus_fraction = 1/3 →
  automobile_fraction = 1/5 →
  bicycle_fraction = 1/10 →
  skateboard_fraction = 1/15 →
  1 - (bus_fraction + automobile_fraction + bicycle_fraction + skateboard_fraction) = 3/10 := by
sorry

end NUMINAMATH_CALUDE_students_walking_home_l1125_112523


namespace NUMINAMATH_CALUDE_triangle_sum_l1125_112512

theorem triangle_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + (1/3)*y^2 = 25)
  (eq2 : (1/3)*y^2 + z^2 = 9)
  (eq3 : z^2 + x*z + x^2 = 16) :
  x*y + 2*y*z + 3*z*x = 24*Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_sum_l1125_112512


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l1125_112564

/-- 
A quadratic expression x^2 - 20x + k is a perfect square binomial 
if and only if k = 100.
-/
theorem perfect_square_binomial (k : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 - 20*x + k = (x + b)^2) ↔ k = 100 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l1125_112564


namespace NUMINAMATH_CALUDE_flies_needed_for_week_l1125_112562

/-- The number of flies a frog eats per day -/
def flies_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of flies Betty caught in total -/
def flies_caught : ℕ := 11

/-- The number of flies that escaped -/
def flies_escaped : ℕ := 1

/-- Theorem stating how many more flies Betty needs for a week -/
theorem flies_needed_for_week : 
  flies_per_day * days_in_week - (flies_caught - flies_escaped) = 4 := by
  sorry

end NUMINAMATH_CALUDE_flies_needed_for_week_l1125_112562


namespace NUMINAMATH_CALUDE_function_expression_l1125_112541

theorem function_expression (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 1) :
  ∀ x, f x = x^2 - 2*x :=
by sorry

end NUMINAMATH_CALUDE_function_expression_l1125_112541


namespace NUMINAMATH_CALUDE_rectangle_dimension_increase_l1125_112598

/-- Given a rectangle with original length L and breadth B, prove that if the breadth is
    increased by 25% and the total area is increased by 37.5%, then the length must be
    increased by 10% -/
theorem rectangle_dimension_increase (L B : ℝ) (L_pos : L > 0) (B_pos : B > 0) :
  let new_B := 1.25 * B
  let new_area := 1.375 * (L * B)
  ∃ x : ℝ, x = 0.1 ∧ new_area = (L * (1 + x)) * new_B := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_increase_l1125_112598


namespace NUMINAMATH_CALUDE_square_screen_diagonal_l1125_112549

theorem square_screen_diagonal (d : ℝ) : 
  d > 0 → 
  (d / Real.sqrt 2) ^ 2 = 20 ^ 2 + 42 → 
  d = Real.sqrt 884 := by
  sorry

end NUMINAMATH_CALUDE_square_screen_diagonal_l1125_112549


namespace NUMINAMATH_CALUDE_hexagon_area_equal_perimeter_l1125_112531

theorem hexagon_area_equal_perimeter (s t : ℝ) : 
  s > 0 → 
  t > 0 → 
  3 * s = 6 * t → -- Equal perimeters condition
  s^2 * Real.sqrt 3 / 4 = 16 → -- Triangle area condition
  6 * (t^2 * Real.sqrt 3 / 4) = 24 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_equal_perimeter_l1125_112531


namespace NUMINAMATH_CALUDE_geometric_sequence_21st_term_l1125_112575

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_21st_term
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_first_term : a 1 = 3)
  (h_common_product : ∀ n : ℕ, a n * a (n + 1) = 15) :
  a 21 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_21st_term_l1125_112575


namespace NUMINAMATH_CALUDE_optimal_pole_is_twelve_l1125_112518

/-- Represents the number of intervals in the path -/
def intervals : ℕ := 28

/-- Represents Dodson's walking time for one interval (in minutes) -/
def dodson_walk_time : ℕ := 9

/-- Represents Williams' walking time for one interval (in minutes) -/
def williams_walk_time : ℕ := 11

/-- Represents the riding time on Bolivar for one interval (in minutes) -/
def bolivar_ride_time : ℕ := 3

/-- Calculates Dodson's total travel time given the pole number -/
def dodson_total_time (pole : ℕ) : ℚ :=
  (pole * bolivar_ride_time + (intervals - pole) * dodson_walk_time) / intervals

/-- Calculates Williams' total travel time given the pole number -/
def williams_total_time (pole : ℕ) : ℚ :=
  (pole * williams_walk_time + (intervals - pole) * bolivar_ride_time) / intervals

/-- Theorem stating that the 12th pole is the optimal point to tie Bolivar -/
theorem optimal_pole_is_twelve :
  ∃ (pole : ℕ), pole = 12 ∧
  ∀ (k : ℕ), 1 ≤ k ∧ k ≤ intervals →
    max (dodson_total_time pole) (williams_total_time pole) ≤
    max (dodson_total_time k) (williams_total_time k) :=
by sorry

end NUMINAMATH_CALUDE_optimal_pole_is_twelve_l1125_112518


namespace NUMINAMATH_CALUDE_complex_square_root_of_negative_four_l1125_112582

theorem complex_square_root_of_negative_four (z : ℂ) 
  (h1 : z^2 = -4)
  (h2 : z.im > 0) : 
  z = Complex.I * 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_of_negative_four_l1125_112582


namespace NUMINAMATH_CALUDE_six_eight_ten_pythagorean_triple_l1125_112559

/-- A Pythagorean triple is a set of three positive integers (a, b, c) that satisfies a² + b² = c² --/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- The set (6, 8, 10) is a Pythagorean triple --/
theorem six_eight_ten_pythagorean_triple : is_pythagorean_triple 6 8 10 := by
  sorry

end NUMINAMATH_CALUDE_six_eight_ten_pythagorean_triple_l1125_112559


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_b_l1125_112524

def b (n : ℕ) : ℕ := n.factorial + n^2

theorem max_gcd_consecutive_b : ∀ n : ℕ, Nat.gcd (b n) (b (n + 1)) ≤ 2 ∧ 
  ∃ m : ℕ, Nat.gcd (b m) (b (m + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_b_l1125_112524


namespace NUMINAMATH_CALUDE_least_years_to_double_l1125_112555

-- Define the interest rate
def interest_rate : ℝ := 0.5

-- Define the function for the amount after t years
def amount (t : ℕ) : ℝ := (1 + interest_rate) ^ t

-- Theorem statement
theorem least_years_to_double :
  ∀ t : ℕ, t < 2 → amount t ≤ 2 ∧ 2 < amount 2 :=
by sorry

end NUMINAMATH_CALUDE_least_years_to_double_l1125_112555


namespace NUMINAMATH_CALUDE_geometric_sequence_product_property_l1125_112528

/-- A sequence is geometric if there exists a non-zero common ratio between consecutive terms. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- The property that a_m * a_n = a_p * a_q for specific m, n, p, q. -/
def HasProductProperty (a : ℕ → ℝ) (m n p q : ℕ) : Prop :=
  a m * a n = a p * a q

theorem geometric_sequence_product_property 
  (a : ℕ → ℝ) (m n p q : ℕ) 
  (hm : m > 0) (hn : n > 0) (hp : p > 0) (hq : q > 0)
  (h_sum : m + n = p + q) :
  IsGeometricSequence a → HasProductProperty a m n p q :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_property_l1125_112528


namespace NUMINAMATH_CALUDE_nuts_in_third_box_l1125_112540

theorem nuts_in_third_box (A B C : ℝ) 
  (h1 : A = B + C - 6) 
  (h2 : B = A + C - 10) : C = 8 := by
  sorry

end NUMINAMATH_CALUDE_nuts_in_third_box_l1125_112540


namespace NUMINAMATH_CALUDE_max_b_value_l1125_112584

theorem max_b_value (a b c : ℕ) : 
  a * b * c = 240 →
  1 < c →
  c < b →
  b < a →
  b ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l1125_112584


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l1125_112525

def total_players : ℕ := 15
def preselected_players : ℕ := 3
def lineup_size : ℕ := 5

theorem starting_lineup_combinations :
  Nat.choose (total_players - preselected_players) (lineup_size - preselected_players) = 66 :=
by sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l1125_112525


namespace NUMINAMATH_CALUDE_sprint_team_total_distance_l1125_112591

theorem sprint_team_total_distance (team_size : ℕ) (distance_per_person : ℝ) :
  team_size = 250 →
  distance_per_person = 7.5 →
  team_size * distance_per_person = 1875 := by
sorry

end NUMINAMATH_CALUDE_sprint_team_total_distance_l1125_112591


namespace NUMINAMATH_CALUDE_sqrt_two_minus_x_real_range_l1125_112509

theorem sqrt_two_minus_x_real_range :
  {x : ℝ | ∃ y : ℝ, y ^ 2 = 2 - x} = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_x_real_range_l1125_112509


namespace NUMINAMATH_CALUDE_power_inequality_l1125_112589

theorem power_inequality (x y a b : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : a > b) (h4 : b > 1) :
  a^x > b^y := by sorry

end NUMINAMATH_CALUDE_power_inequality_l1125_112589


namespace NUMINAMATH_CALUDE_watermelon_problem_l1125_112521

theorem watermelon_problem (selling_price : ℕ) (total_profit : ℕ) (watermelons_left : ℕ) :
  selling_price = 3 →
  total_profit = 105 →
  watermelons_left = 18 →
  selling_price * ((total_profit / selling_price) + watermelons_left) = 53 * selling_price :=
by sorry

end NUMINAMATH_CALUDE_watermelon_problem_l1125_112521


namespace NUMINAMATH_CALUDE_probability_three_kings_or_ace_value_l1125_112529

/-- Represents a standard deck of cards --/
structure Deck :=
  (total_cards : ℕ)
  (queens : ℕ)
  (kings : ℕ)
  (aces : ℕ)

/-- The probability of drawing either three Kings or at least one Ace --/
def probability_three_kings_or_ace (d : Deck) : ℚ :=
  let p_three_kings := (d.kings : ℚ) / d.total_cards * (d.kings - 1) / (d.total_cards - 1) * (d.kings - 2) / (d.total_cards - 2)
  let p_no_aces := (d.total_cards - d.aces : ℚ) / d.total_cards * (d.total_cards - d.aces - 1) / (d.total_cards - 1) * (d.total_cards - d.aces - 2) / (d.total_cards - 2)
  p_three_kings + (1 - p_no_aces)

/-- The theorem to be proved --/
theorem probability_three_kings_or_ace_value :
  let d : Deck := ⟨52, 4, 4, 4⟩
  probability_three_kings_or_ace d = 961 / 4420 := by
  sorry


end NUMINAMATH_CALUDE_probability_three_kings_or_ace_value_l1125_112529


namespace NUMINAMATH_CALUDE_minutes_before_noon_l1125_112539

/-- 
Given that 20 minutes ago it was 3 times as many minutes after 9 am, 
and there are 180 minutes between 9 am and 12 noon, 
prove that it is 130 minutes before 12 noon.
-/
theorem minutes_before_noon : 
  ∀ x : ℕ, 
  (x + 20 = 3 * (180 - x)) → 
  x = 130 := by
sorry

end NUMINAMATH_CALUDE_minutes_before_noon_l1125_112539


namespace NUMINAMATH_CALUDE_P_in_third_quadrant_iff_m_less_than_two_l1125_112548

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the third quadrant -/
def is_in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The point P with coordinates (-1, -2+m) -/
def P (m : ℝ) : Point :=
  ⟨-1, -2+m⟩

/-- Theorem stating the condition for P to be in the third quadrant -/
theorem P_in_third_quadrant_iff_m_less_than_two (m : ℝ) :
  is_in_third_quadrant (P m) ↔ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_P_in_third_quadrant_iff_m_less_than_two_l1125_112548


namespace NUMINAMATH_CALUDE_intersection_point_equality_l1125_112593

/-- Given a system of linear equations and its solution, prove that the intersection
    point of two related lines is the same as the solution. -/
theorem intersection_point_equality (x y : ℝ) :
  x - y = -5 →
  x + 2*y = -2 →
  x = -4 →
  y = 1 →
  ∃! (x' y' : ℝ), y' = x' + 5 ∧ y' = -1/2 * x' - 1 ∧ x' = -4 ∧ y' = 1 :=
by sorry


end NUMINAMATH_CALUDE_intersection_point_equality_l1125_112593


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1125_112517

/-- Proves that the complex number z = (-8 - 7i)(-3i) is located in the second quadrant of the complex plane. -/
theorem complex_number_in_second_quadrant : 
  let z : ℂ := (-8 - 7*I) * (-3*I)
  (z.re < 0 ∧ z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1125_112517


namespace NUMINAMATH_CALUDE_fish_aquarium_problem_l1125_112538

theorem fish_aquarium_problem (x y : ℕ) :
  x + y = 100 ∧ x - 30 = y - 40 → x = 45 ∧ y = 55 := by
  sorry

end NUMINAMATH_CALUDE_fish_aquarium_problem_l1125_112538


namespace NUMINAMATH_CALUDE_bird_migration_difference_l1125_112516

/-- The number of bird families living near the mountain -/
def mountain_families : ℕ := 38

/-- The number of bird families that flew to Africa -/
def africa_families : ℕ := 47

/-- The number of bird families that flew to Asia -/
def asia_families : ℕ := 94

/-- Theorem: The difference between the number of bird families that flew to Asia
    and the number of bird families that flew to Africa is 47 -/
theorem bird_migration_difference :
  asia_families - africa_families = 47 := by
  sorry

end NUMINAMATH_CALUDE_bird_migration_difference_l1125_112516


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1125_112514

theorem geometric_sequence_property (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a (n + 1) = a n * r) →  -- Geometric sequence definition
  a 4 = 1.5 →                   -- 4th term is 1.5
  a 10 = 1.62 →                 -- 10th term is 1.62
  a 7 = Real.sqrt 2.43 :=        -- 7th term is √2.43
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l1125_112514


namespace NUMINAMATH_CALUDE_ellipse_circle_dot_product_range_l1125_112542

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 9) + (P.2^2 / 8) = 1

-- Define the circle
def is_on_circle (P : ℝ × ℝ) : Prop :=
  (P.1 - 1)^2 + P.2^2 = 1

-- Define a diameter of the circle
def is_diameter (A B : ℝ × ℝ) : Prop :=
  is_on_circle A ∧ is_on_circle B ∧ (A.1 + B.1 = 2) ∧ (A.2 + B.2 = 0)

-- Define the dot product
def dot_product (P A B : ℝ × ℝ) : ℝ :=
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2)

theorem ellipse_circle_dot_product_range :
  ∀ (P A B : ℝ × ℝ),
    is_on_ellipse P →
    is_diameter A B →
    3 ≤ dot_product P A B ∧ dot_product P A B ≤ 15 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_circle_dot_product_range_l1125_112542


namespace NUMINAMATH_CALUDE_greatest_value_of_fraction_l1125_112543

theorem greatest_value_of_fraction (y : ℝ) : 
  (∀ θ : ℝ, y ≥ 14 / (5 + 3 * Real.sin θ)) → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_greatest_value_of_fraction_l1125_112543


namespace NUMINAMATH_CALUDE_pension_fund_strategy_optimizes_portfolio_l1125_112577

/-- Represents different types of assets --/
inductive AssetType
  | DebtAsset
  | EquityAsset

/-- Represents an investment portfolio --/
structure Portfolio where
  debtAssets : ℝ
  equityAssets : ℝ

/-- Represents the investment strategy --/
structure InvestmentStrategy where
  portfolio : Portfolio
  maxEquityProportion : ℝ

/-- Defines the concept of a balanced portfolio --/
def isBalanced (s : InvestmentStrategy) : Prop :=
  s.portfolio.equityAssets / (s.portfolio.debtAssets + s.portfolio.equityAssets) ≤ s.maxEquityProportion

/-- Defines the concept of an optimized portfolio --/
def isOptimized (s : InvestmentStrategy) : Prop :=
  isBalanced s ∧ s.portfolio.equityAssets > 0 ∧ s.portfolio.debtAssets > 0

/-- Main theorem: The investment strategy optimizes the portfolio and balances returns and risks --/
theorem pension_fund_strategy_optimizes_portfolio (s : InvestmentStrategy) 
  (h1 : s.portfolio.debtAssets > 0)
  (h2 : s.portfolio.equityAssets > 0)
  (h3 : s.maxEquityProportion = 0.3)
  (h4 : isBalanced s) :
  isOptimized s :=
sorry


end NUMINAMATH_CALUDE_pension_fund_strategy_optimizes_portfolio_l1125_112577


namespace NUMINAMATH_CALUDE_fg_length_l1125_112570

/-- Represents a parallelogram ABCD and a right triangle DFG with specific properties -/
structure GeometricFigures where
  AB : ℝ
  AD : ℝ
  DG : ℝ
  area_equality : AB * AD = 1/2 * DG * AB

/-- The length of FG in the given geometric configuration is 8 -/
theorem fg_length (figures : GeometricFigures) 
  (h1 : figures.AB = 8)
  (h2 : figures.AD = 3)
  (h3 : figures.DG = 6) :
  figures.AB = 8 := by sorry

end NUMINAMATH_CALUDE_fg_length_l1125_112570


namespace NUMINAMATH_CALUDE_subset_intersection_union_equivalence_l1125_112520

theorem subset_intersection_union_equivalence (A B C : Set α) :
  (B ⊆ A ∧ C ⊆ A) ↔ ((A ∩ B) ∪ (A ∩ C) = B ∪ C) := by
  sorry

end NUMINAMATH_CALUDE_subset_intersection_union_equivalence_l1125_112520


namespace NUMINAMATH_CALUDE_polynomial_problem_l1125_112581

-- Define the polynomials
def B (x : ℝ) : ℝ := 4 * x^2 - 5 * x - 7

theorem polynomial_problem (A : ℝ → ℝ) 
  (h : ∀ x, A x - 2 * (B x) = -2 * x^2 + 10 * x + 14) :
  (∀ x, A x = 6 * x^2) ∧ 
  (∀ x, A x + 2 * (B x) = 14 * x^2 - 10 * x - 14) ∧
  (A (-1) + 2 * (B (-1)) = 10) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_problem_l1125_112581


namespace NUMINAMATH_CALUDE_min_value_theorem_l1125_112566

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y + 2 * x + 3 * y = 42) :
  x * y + 5 * x + 4 * y ≥ 55 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀ + 2 * x₀ + 3 * y₀ = 42 ∧ x₀ * y₀ + 5 * x₀ + 4 * y₀ = 55 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1125_112566


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1125_112507

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, 0 < x ∧ x < π/2 → x > Real.sin x) ↔
  ∃ x : ℝ, 0 < x ∧ x < π/2 ∧ x ≤ Real.sin x :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1125_112507


namespace NUMINAMATH_CALUDE_blue_balloon_count_l1125_112503

/-- The number of blue balloons owned by Joan, Melanie, Alex, and Gary, respectively --/
def joan_balloons : ℕ := 60
def melanie_balloons : ℕ := 85
def alex_balloons : ℕ := 37
def gary_balloons : ℕ := 48

/-- The total number of blue balloons --/
def total_blue_balloons : ℕ := joan_balloons + melanie_balloons + alex_balloons + gary_balloons

theorem blue_balloon_count : total_blue_balloons = 230 := by
  sorry

end NUMINAMATH_CALUDE_blue_balloon_count_l1125_112503


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l1125_112554

/-- Proves that the initial ratio of milk to water is 3:1 given the conditions -/
theorem initial_milk_water_ratio 
  (total_volume : ℝ) 
  (added_water : ℝ) 
  (final_ratio : ℝ) :
  total_volume = 50 →
  added_water = 100 →
  final_ratio = 1/3 →
  ∃ (initial_milk initial_water : ℝ),
    initial_milk + initial_water = total_volume ∧
    initial_milk / (initial_water + added_water) = final_ratio ∧
    initial_milk / initial_water = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l1125_112554


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1125_112565

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, x^2 + 4*x - 1 = 0 ↔ (x + 2)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1125_112565


namespace NUMINAMATH_CALUDE_f_value_at_2_l1125_112550

def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem f_value_at_2 (a b : ℝ) :
  f a b (-2) = 10 → f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l1125_112550


namespace NUMINAMATH_CALUDE_square_of_1035_l1125_112544

theorem square_of_1035 : (1035 : ℕ)^2 = 1071225 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1035_l1125_112544


namespace NUMINAMATH_CALUDE_foil_covered_prism_width_l1125_112513

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (d : PrismDimensions) : ℝ := d.length * d.width * d.height

/-- Represents the properties of the inner prism -/
structure InnerPrism where
  dimensions : PrismDimensions
  cubeCount : ℕ

/-- Represents the properties of the foil-covered prism -/
structure FoilCoveredPrism where
  innerPrism : InnerPrism
  foilThickness : ℝ

theorem foil_covered_prism_width
  (p : FoilCoveredPrism)
  (h1 : p.innerPrism.cubeCount = 128)
  (h2 : p.innerPrism.dimensions.width = 2 * p.innerPrism.dimensions.length)
  (h3 : p.innerPrism.dimensions.width = 2 * p.innerPrism.dimensions.height)
  (h4 : volume p.innerPrism.dimensions = p.innerPrism.cubeCount)
  (h5 : p.foilThickness = 1) :
  p.innerPrism.dimensions.width + 2 * p.foilThickness = 10 := by
  sorry


end NUMINAMATH_CALUDE_foil_covered_prism_width_l1125_112513


namespace NUMINAMATH_CALUDE_elizabeth_lost_bottles_l1125_112535

/-- The number of water bottles Elizabeth lost at school -/
def bottles_lost_at_school : ℕ := 2

theorem elizabeth_lost_bottles (initial_bottles : ℕ) (stolen_bottle : ℕ) (stickers_per_bottle : ℕ) (total_stickers : ℕ) :
  initial_bottles = 10 →
  stolen_bottle = 1 →
  stickers_per_bottle = 3 →
  total_stickers = 21 →
  stickers_per_bottle * (initial_bottles - bottles_lost_at_school - stolen_bottle) = total_stickers →
  bottles_lost_at_school = 2 := by
sorry

end NUMINAMATH_CALUDE_elizabeth_lost_bottles_l1125_112535


namespace NUMINAMATH_CALUDE_quadratic_function_through_point_l1125_112511

theorem quadratic_function_through_point (a b : ℝ) :
  (∀ t : ℝ, (t^2 + t + 1) * 1^2 - 2*(a+t)^2 * 1 + t^2 + 3*a*t + b = 0) →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_through_point_l1125_112511


namespace NUMINAMATH_CALUDE_no_snow_probability_l1125_112532

theorem no_snow_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 2/3)
  (h2 : p2 = 2/3)
  (h3 : p3 = 3/5)
  (h_independent : True)  -- Representing independence of events
  : (1 - p1) * (1 - p2) * (1 - p3) = 2/45 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l1125_112532


namespace NUMINAMATH_CALUDE_intersection_A_B_l1125_112515

-- Define set A
def A : Set ℝ := {x | x^2 - x - 6 > 0}

-- Define set B
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | -3 ≤ x ∧ x < -2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1125_112515


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l1125_112552

theorem smallest_undefined_inverse (a : ℕ) : a > 0 ∧ 
  (∀ b : ℕ, b < a → (Nat.gcd b 60 = 1 ∨ Nat.gcd b 75 = 1)) ∧
  Nat.gcd a 60 ≠ 1 ∧ Nat.gcd a 75 ≠ 1 → a = 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l1125_112552


namespace NUMINAMATH_CALUDE_hypergeom_problem_l1125_112578

/-- Hypergeometric distribution parameters -/
structure HyperGeomParams where
  N : ℕ  -- Population size
  M : ℕ  -- Number of successes in the population
  n : ℕ  -- Number of draws
  h1 : M ≤ N
  h2 : n ≤ N

/-- Probability of k successes in n draws -/
def prob_k_successes (p : HyperGeomParams) (k : ℕ) : ℚ :=
  (Nat.choose p.M k * Nat.choose (p.N - p.M) (p.n - k)) / Nat.choose p.N p.n

/-- Expected value of hypergeometric distribution -/
def expected_value (p : HyperGeomParams) : ℚ :=
  (p.n * p.M : ℚ) / p.N

/-- Theorem for the specific problem -/
theorem hypergeom_problem (p : HyperGeomParams) 
    (h3 : p.N = 10) (h4 : p.M = 5) (h5 : p.n = 4) : 
    prob_k_successes p 3 = 5 / 21 ∧ expected_value p = 2 := by
  sorry


end NUMINAMATH_CALUDE_hypergeom_problem_l1125_112578


namespace NUMINAMATH_CALUDE_log_inequality_l1125_112533

theorem log_inequality (a b c : ℝ) (ha : a = Real.log 6 / Real.log 4)
  (hb : b = Real.log 0.2 / Real.log 4) (hc : c = Real.log 3 / Real.log 2) :
  c > a ∧ a > b :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l1125_112533


namespace NUMINAMATH_CALUDE_sum_of_squares_l1125_112585

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c = 0) (h_power : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) :
  a^2 + b^2 + c^2 = 6/5 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1125_112585


namespace NUMINAMATH_CALUDE_running_distance_proof_l1125_112574

/-- Calculates the total distance run over a number of days, given a constant daily distance. -/
def totalDistance (dailyDistance : ℕ) (days : ℕ) : ℕ :=
  dailyDistance * days

/-- Proves that running 1700 meters for 6 consecutive days results in a total distance of 10200 meters. -/
theorem running_distance_proof :
  let dailyDistance : ℕ := 1700
  let days : ℕ := 6
  totalDistance dailyDistance days = 10200 := by
sorry

end NUMINAMATH_CALUDE_running_distance_proof_l1125_112574


namespace NUMINAMATH_CALUDE_network_connections_l1125_112576

/-- 
Given a network of switches where:
- There are 30 switches
- Each switch is directly connected to exactly 4 other switches
This theorem states that the total number of connections in the network is 60.
-/
theorem network_connections (n : ℕ) (c : ℕ) : 
  n = 30 → c = 4 → (n * c) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_network_connections_l1125_112576


namespace NUMINAMATH_CALUDE_megan_pop_albums_l1125_112590

def country_albums : ℕ := 2
def songs_per_album : ℕ := 7
def total_songs : ℕ := 70

def pop_albums : ℕ := (total_songs - country_albums * songs_per_album) / songs_per_album

theorem megan_pop_albums : pop_albums = 8 := by
  sorry

end NUMINAMATH_CALUDE_megan_pop_albums_l1125_112590


namespace NUMINAMATH_CALUDE_rod_length_proof_l1125_112579

/-- The length of a rod in meters, given the number of pieces and the length of each piece in centimeters. -/
def rod_length_meters (num_pieces : ℕ) (piece_length_cm : ℕ) : ℚ :=
  (num_pieces * piece_length_cm : ℚ) / 100

/-- Theorem stating that a rod from which 45 pieces of 85 cm can be cut is 38.25 meters long. -/
theorem rod_length_proof : rod_length_meters 45 85 = 38.25 := by
  sorry

end NUMINAMATH_CALUDE_rod_length_proof_l1125_112579


namespace NUMINAMATH_CALUDE_race_to_top_floor_l1125_112571

/-- Represents the time taken by a person to reach the top floor of a building -/
def TimeTaken (stories : ℕ) (timePerStory : ℕ) (stopTime : ℕ) (stopsPerStory : ℕ) : ℕ :=
  stories * timePerStory + (stories - 2) * stopTime * stopsPerStory

/-- The maximum time taken between two people to reach the top floor -/
def MaxTimeTaken (time1 : ℕ) (time2 : ℕ) : ℕ :=
  max time1 time2

theorem race_to_top_floor :
  let stories := 20
  let lolaTimePerStory := 10
  let elevatorTimePerStory := 8
  let elevatorStopTime := 3
  let elevatorStopsPerStory := 1
  let lolaTime := TimeTaken stories lolaTimePerStory 0 0
  let taraTime := TimeTaken stories elevatorTimePerStory elevatorStopTime elevatorStopsPerStory
  MaxTimeTaken lolaTime taraTime = 214 :=
by sorry


end NUMINAMATH_CALUDE_race_to_top_floor_l1125_112571


namespace NUMINAMATH_CALUDE_belle_biscuits_l1125_112545

/-- The number of dog biscuits Belle eats every evening -/
def num_biscuits : ℕ := 4

/-- The number of rawhide bones Belle eats every evening -/
def num_bones : ℕ := 2

/-- The cost of one rawhide bone in dollars -/
def cost_bone : ℚ := 1

/-- The cost of one dog biscuit in dollars -/
def cost_biscuit : ℚ := 1/4

/-- The total cost to feed Belle these treats for a week in dollars -/
def total_cost : ℚ := 21

theorem belle_biscuits :
  num_biscuits = 4 ∧
  (7 : ℚ) * (num_bones * cost_bone + num_biscuits * cost_biscuit) = total_cost :=
sorry

end NUMINAMATH_CALUDE_belle_biscuits_l1125_112545


namespace NUMINAMATH_CALUDE_min_value_theorem_l1125_112573

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  x + 3 * y ≥ 4 + 8 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧
    x₀ + 3 * y₀ = 4 + 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1125_112573


namespace NUMINAMATH_CALUDE_triangle_equilateral_l1125_112508

theorem triangle_equilateral (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (condition : a^2 + b^2 + 2*c^2 - 2*a*c - 2*b*c = 0) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l1125_112508


namespace NUMINAMATH_CALUDE_max_area_isosceles_triangle_l1125_112586

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The angle at vertex A of a triangle -/
def angle_at_A (t : Triangle) : ℝ := sorry

/-- The semiperimeter of a triangle -/
def semiperimeter (t : Triangle) : ℝ := sorry

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is isosceles with base BC -/
def is_isosceles_BC (t : Triangle) : Prop := sorry

/-- Theorem: Among all triangles with fixed angle α at A and fixed semiperimeter p,
    the isosceles triangle with base BC has the largest area -/
theorem max_area_isosceles_triangle (α p : ℝ) :
  ∀ t : Triangle,
    angle_at_A t = α →
    semiperimeter t = p →
    ∀ t' : Triangle,
      angle_at_A t' = α →
      semiperimeter t' = p →
      is_isosceles_BC t' →
      area t ≤ area t' :=
sorry

end NUMINAMATH_CALUDE_max_area_isosceles_triangle_l1125_112586


namespace NUMINAMATH_CALUDE_distance_to_origin_l1125_112558

theorem distance_to_origin : Real.sqrt (3^2 + (-2)^2) = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_distance_to_origin_l1125_112558


namespace NUMINAMATH_CALUDE_a_greater_than_b_l1125_112536

theorem a_greater_than_b (m : ℝ) (h : m > 1) : 
  (Real.sqrt m - Real.sqrt (m - 1)) > (Real.sqrt (m + 1) - Real.sqrt m) := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l1125_112536


namespace NUMINAMATH_CALUDE_remainder_of_sum_mod_11_l1125_112557

theorem remainder_of_sum_mod_11 : (100001 + 100002 + 100003 + 100004 + 100005 + 100006 + 100007) % 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_mod_11_l1125_112557


namespace NUMINAMATH_CALUDE_correct_first_grade_sample_size_l1125_112500

/-- Represents a school with stratified sampling -/
structure School where
  total_students : ℕ
  first_grade_students : ℕ
  sample_size : ℕ

/-- Calculates the number of first-grade students to be selected in a stratified sample -/
def stratified_sample_size (school : School) : ℕ :=
  (school.first_grade_students * school.sample_size) / school.total_students

/-- Theorem stating the correct number of first-grade students to be selected -/
theorem correct_first_grade_sample_size (school : School) 
  (h1 : school.total_students = 2000)
  (h2 : school.first_grade_students = 400)
  (h3 : school.sample_size = 200) :
  stratified_sample_size school = 40 := by
  sorry

#eval stratified_sample_size { total_students := 2000, first_grade_students := 400, sample_size := 200 }

end NUMINAMATH_CALUDE_correct_first_grade_sample_size_l1125_112500


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1125_112510

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a set of four points
def FourPoints := Fin 4 → Point3D

-- Define collinearity for three points
def collinear (p q r : Point3D) : Prop := sorry

-- Define coplanarity for four points
def coplanar (points : FourPoints) : Prop := sorry

-- No three points are collinear
def no_three_collinear (points : FourPoints) : Prop :=
  ∀ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬collinear (points i) (points j) (points k)

theorem sufficient_but_not_necessary :
  (∀ (points : FourPoints), no_three_collinear points → ¬coplanar points) ∧
  (∃ (points : FourPoints), ¬coplanar points ∧ ¬no_three_collinear points) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1125_112510


namespace NUMINAMATH_CALUDE_tower_construction_modulo_l1125_112551

/-- Represents the number of towers that can be built using cubes up to size n -/
def T : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| (n + 1) => if n ≥ 2 then 4 * T n else 3 * T n

/-- The problem statement -/
theorem tower_construction_modulo :
  T 10 % 1000 = 304 := by
  sorry

end NUMINAMATH_CALUDE_tower_construction_modulo_l1125_112551


namespace NUMINAMATH_CALUDE_problem_solution_l1125_112588

theorem problem_solution : 
  ∀ x y : ℤ, x > y ∧ y > 0 ∧ x + y + x * y = 152 → x = 16 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1125_112588


namespace NUMINAMATH_CALUDE_solve_lemonade_problem_l1125_112596

def lemonade_problem (price_per_cup : ℝ) (cups_sold : ℕ) (cost_lemons : ℝ) (cost_sugar : ℝ) (total_profit : ℝ) : Prop :=
  let total_revenue := price_per_cup * (cups_sold : ℝ)
  let known_expenses := cost_lemons + cost_sugar
  let cost_cups := total_revenue - known_expenses - total_profit
  cost_cups = 3

theorem solve_lemonade_problem :
  lemonade_problem 4 21 10 5 66 := by
  sorry

end NUMINAMATH_CALUDE_solve_lemonade_problem_l1125_112596


namespace NUMINAMATH_CALUDE_words_lost_proof_l1125_112567

/-- The number of letters in the language --/
def num_letters : ℕ := 69

/-- The index of the forbidden letter --/
def forbidden_letter_index : ℕ := 7

/-- The number of words lost due to prohibition --/
def words_lost : ℕ := 139

/-- Theorem stating the number of words lost due to prohibition --/
theorem words_lost_proof :
  (num_letters : ℕ) = 69 →
  (forbidden_letter_index : ℕ) = 7 →
  (words_lost : ℕ) = 139 :=
by
  sorry

#check words_lost_proof

end NUMINAMATH_CALUDE_words_lost_proof_l1125_112567


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1125_112595

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 35) (h3 : y = 3 * x) :
  y = -21 → x = -10.9375 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1125_112595


namespace NUMINAMATH_CALUDE_carpet_exchange_theorem_l1125_112504

theorem carpet_exchange_theorem (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  ∃ c : ℝ, c > 0 ∧ ((c > 1 ∧ a / c < 1) ∨ (c < 1 ∧ a / c > 1)) := by
  sorry

end NUMINAMATH_CALUDE_carpet_exchange_theorem_l1125_112504


namespace NUMINAMATH_CALUDE_cos_B_value_l1125_112563

-- Define the angle B
def B : ℝ := sorry

-- Define the conditions
def B_in_third_quadrant : 3 * π / 2 < B ∧ B < 2 * π := sorry
def sin_B : Real.sin B = -5/13 := sorry

-- Theorem to prove
theorem cos_B_value : Real.cos B = -12/13 := by sorry

end NUMINAMATH_CALUDE_cos_B_value_l1125_112563


namespace NUMINAMATH_CALUDE_mary_hospital_time_l1125_112556

/-- Given the conditions of Mary's ambulance ride and Don's drive to the hospital,
    prove that Mary reaches the hospital in 15 minutes. -/
theorem mary_hospital_time (ambulance_speed : ℝ) (don_speed : ℝ) (don_time : ℝ) :
  ambulance_speed = 60 →
  don_speed = 30 →
  don_time = 0.5 →
  (don_speed * don_time) / ambulance_speed = 0.25 := by
  sorry

#check mary_hospital_time

end NUMINAMATH_CALUDE_mary_hospital_time_l1125_112556


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l1125_112572

/-- Given two functions f and g that intersect at (2,5) and (8,3), prove that a + c = 10 -/
theorem intersection_implies_sum (a b c d : ℝ) : 
  (∀ x, -|x - a| + b = |x - c| + d → x = 2 ∨ x = 8) →
  -|2 - a| + b = 5 →
  -|8 - a| + b = 3 →
  |2 - c| + d = 5 →
  |8 - c| + d = 3 →
  a + c = 10 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l1125_112572


namespace NUMINAMATH_CALUDE_hotel_room_charge_comparison_l1125_112597

theorem hotel_room_charge_comparison (P R G : ℝ) 
  (h1 : P = R - 0.2 * R) 
  (h2 : P = G - 0.1 * G) : 
  R = G * (1 + 0.125) := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_charge_comparison_l1125_112597


namespace NUMINAMATH_CALUDE_selected_students_l1125_112534

-- Define the set of students
inductive Student : Type
| A | B | C | D | E

-- Define a type for the selection of students
def Selection := Student → Prop

-- Define the conditions
def valid_selection (s : Selection) : Prop :=
  -- 3 students are selected
  (∃ (x y z : Student), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ s x ∧ s y ∧ s z ∧
    ∀ (w : Student), s w → (w = x ∨ w = y ∨ w = z)) ∧
  -- If A is selected, then B is selected and E is not selected
  (s Student.A → s Student.B ∧ ¬s Student.E) ∧
  -- If B or E is selected, then D is not selected
  ((s Student.B ∨ s Student.E) → ¬s Student.D) ∧
  -- At least one of C or D must be selected
  (s Student.C ∨ s Student.D)

-- Theorem statement
theorem selected_students (s : Selection) :
  valid_selection s → s Student.A → s Student.B ∧ s Student.C :=
by sorry

end NUMINAMATH_CALUDE_selected_students_l1125_112534


namespace NUMINAMATH_CALUDE_jelly_bean_difference_l1125_112553

theorem jelly_bean_difference (total : ℕ) (vanilla : ℕ) (grape : ℕ) 
  (h1 : total = 770)
  (h2 : vanilla = 120)
  (h3 : total = grape + vanilla)
  (h4 : grape > 5 * vanilla) :
  grape - 5 * vanilla = 50 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_difference_l1125_112553


namespace NUMINAMATH_CALUDE_no_eight_roots_for_composite_quadratics_l1125_112501

/-- A quadratic trinomial is a polynomial of degree 2 -/
def QuadraticTrinomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem no_eight_roots_for_composite_quadratics :
  ¬ ∃ (f g h : ℝ → ℝ),
    QuadraticTrinomial f ∧ QuadraticTrinomial g ∧ QuadraticTrinomial h ∧
    (∀ x, f (g (h x)) = 0 ↔ x ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : Set ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_no_eight_roots_for_composite_quadratics_l1125_112501


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1125_112580

theorem complex_equation_sum (a b : ℝ) :
  (a - 2 * Complex.I) * Complex.I = b + Complex.I →
  a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1125_112580


namespace NUMINAMATH_CALUDE_gordons_second_restaurant_meals_l1125_112546

/-- Given Gordon's restaurants and their meal serving information, prove that the second restaurant serves 40 meals per day. -/
theorem gordons_second_restaurant_meals (total_weekly_meals : ℕ)
  (first_restaurant_daily_meals : ℕ) (third_restaurant_daily_meals : ℕ)
  (h1 : total_weekly_meals = 770)
  (h2 : first_restaurant_daily_meals = 20)
  (h3 : third_restaurant_daily_meals = 50) :
  ∃ (second_restaurant_daily_meals : ℕ),
    second_restaurant_daily_meals = 40 ∧
    total_weekly_meals = 7 * (first_restaurant_daily_meals + second_restaurant_daily_meals + third_restaurant_daily_meals) :=
by sorry

end NUMINAMATH_CALUDE_gordons_second_restaurant_meals_l1125_112546


namespace NUMINAMATH_CALUDE_inner_square_prob_10x10_l1125_112519

/-- Represents a square checkerboard -/
structure Checkerboard where
  size : ℕ
  total_squares : ℕ
  edge_squares : ℕ
  inner_squares : ℕ

/-- Calculates the probability of choosing an inner square -/
def inner_square_probability (board : Checkerboard) : ℚ :=
  board.inner_squares / board.total_squares

/-- Properties of a 10x10 checkerboard -/
def board_10x10 : Checkerboard :=
  { size := 10
  , total_squares := 100
  , edge_squares := 36
  , inner_squares := 64 }

/-- Theorem: The probability of choosing an inner square on a 10x10 board is 16/25 -/
theorem inner_square_prob_10x10 :
  inner_square_probability board_10x10 = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_inner_square_prob_10x10_l1125_112519


namespace NUMINAMATH_CALUDE_range_of_trig_function_l1125_112505

theorem range_of_trig_function :
  ∀ x : ℝ, (3 / 8 : ℝ) ≤ Real.sin x ^ 6 + Real.cos x ^ 4 ∧
            Real.sin x ^ 6 + Real.cos x ^ 4 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_trig_function_l1125_112505
