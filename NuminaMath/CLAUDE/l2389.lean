import Mathlib

namespace NUMINAMATH_CALUDE_lunks_needed_for_bananas_l2389_238973

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks_rate : ℚ := 4 / 7

/-- Exchange rate between kunks and bananas -/
def kunks_to_bananas_rate : ℚ := 5 / 3

/-- Number of bananas to purchase -/
def bananas_to_buy : ℕ := 20

/-- Theorem stating the number of lunks needed to buy the specified number of bananas -/
theorem lunks_needed_for_bananas : 
  ⌈(bananas_to_buy : ℚ) / (kunks_to_bananas_rate * lunks_to_kunks_rate)⌉ = 21 := by
  sorry

end NUMINAMATH_CALUDE_lunks_needed_for_bananas_l2389_238973


namespace NUMINAMATH_CALUDE_simplify_expression_l2389_238930

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  (20 * x^2) * (5 * x) * (1 / (2 * x)^2) * (2 * x)^2 = 100 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2389_238930


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2389_238938

theorem rectangle_perimeter (a b : ℝ) (h1 : a * b = 24) (h2 : a^2 + b^2 = 11^2) : 
  2 * (a + b) = 26 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2389_238938


namespace NUMINAMATH_CALUDE_quadratic_binomial_square_l2389_238977

theorem quadratic_binomial_square (a b : ℝ) : 
  (∃ c d : ℝ, ∀ x : ℝ, 6 * x^2 + 18 * x + a = (c * x + d)^2) ∧
  (∃ c d : ℝ, ∀ x : ℝ, 3 * x^2 + b * x + 4 = (c * x + d)^2) →
  a = 13.5 ∧ b = 18 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_binomial_square_l2389_238977


namespace NUMINAMATH_CALUDE_decagon_adjacent_vertices_probability_l2389_238953

theorem decagon_adjacent_vertices_probability :
  let n : ℕ := 10  -- number of vertices in a decagon
  let adjacent_pairs : ℕ := 2  -- number of adjacent vertices for any chosen vertex
  let total_choices : ℕ := n - 1  -- total number of choices for the second vertex
  (adjacent_pairs : ℚ) / total_choices = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_decagon_adjacent_vertices_probability_l2389_238953


namespace NUMINAMATH_CALUDE_enclosed_area_theorem_l2389_238911

/-- The area enclosed by a curve composed of 9 congruent circular arcs, each with length π/3,
    centered at the vertices of a regular hexagon with side length 3 -/
def enclosed_area (arc_length : Real) (num_arcs : Nat) (hexagon_side : Real) : Real :=
  sorry

/-- The theorem stating the enclosed area for the given conditions -/
theorem enclosed_area_theorem :
  enclosed_area (π/3) 9 3 = (27 * Real.sqrt 3) / 2 + (3 * π) / 8 := by
  sorry

end NUMINAMATH_CALUDE_enclosed_area_theorem_l2389_238911


namespace NUMINAMATH_CALUDE_michael_has_270_eggs_l2389_238997

/-- The number of eggs Michael has after buying and giving away crates -/
def michael_eggs (initial_crates : ℕ) (given_away : ℕ) (bought_later : ℕ) (eggs_per_crate : ℕ) : ℕ :=
  ((initial_crates - given_away) + bought_later) * eggs_per_crate

/-- Theorem stating that Michael has 270 eggs given the problem conditions -/
theorem michael_has_270_eggs :
  michael_eggs 6 2 5 30 = 270 := by
  sorry

end NUMINAMATH_CALUDE_michael_has_270_eggs_l2389_238997


namespace NUMINAMATH_CALUDE_projection_ratio_l2389_238937

def projection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  !![9/41, -20/41; -20/41, 16/41]

theorem projection_ratio :
  ∀ (a b : ℚ),
  (a ≠ 0) →
  (projection_matrix.vecMul ![a, b] = ![a, b]) →
  b / a = 8 / 5 := by
sorry

end NUMINAMATH_CALUDE_projection_ratio_l2389_238937


namespace NUMINAMATH_CALUDE_dave_final_tickets_l2389_238986

/-- Calculates the number of tickets Dave had left after a series of transactions at the arcade. -/
def dave_tickets_left (initial : ℕ) (won : ℕ) (spent1 : ℕ) (traded : ℕ) (spent2 : ℕ) : ℕ :=
  initial + won - spent1 + traded - spent2

/-- Proves that Dave had 57 tickets left at the end of his arcade visit. -/
theorem dave_final_tickets :
  dave_tickets_left 25 127 84 45 56 = 57 := by
  sorry

end NUMINAMATH_CALUDE_dave_final_tickets_l2389_238986


namespace NUMINAMATH_CALUDE_expand_product_l2389_238941

theorem expand_product (x : ℝ) : (3 * x + 4) * (x - 2) = 3 * x^2 - 2 * x - 8 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2389_238941


namespace NUMINAMATH_CALUDE_find_m_l2389_238918

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the theorem
theorem find_m (z m : ℂ) (h1 : det z i m i = 1 - 2*I) (h2 : z.re = 0) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l2389_238918


namespace NUMINAMATH_CALUDE_pigeonhole_principle_interns_l2389_238903

theorem pigeonhole_principle_interns (n : ℕ) (h : n > 0) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧
  (∃ (f : Fin n → ℕ), (∀ x, f x < n - 1) ∧ f i = k ∧ f j = k) :=
sorry

end NUMINAMATH_CALUDE_pigeonhole_principle_interns_l2389_238903


namespace NUMINAMATH_CALUDE_cat_dog_positions_l2389_238956

/-- Represents the number of positions for the cat -/
def cat_positions : Nat := 4

/-- Represents the number of positions for the dog -/
def dog_positions : Nat := 6

/-- Represents the total number of moves -/
def total_moves : Nat := 317

/-- Calculates the final position of an animal given its number of positions and total moves -/
def final_position (positions : Nat) (moves : Nat) : Nat :=
  moves % positions

theorem cat_dog_positions :
  (final_position cat_positions total_moves = 0) ∧
  (final_position dog_positions total_moves = 5) := by
  sorry

end NUMINAMATH_CALUDE_cat_dog_positions_l2389_238956


namespace NUMINAMATH_CALUDE_greatest_three_digit_base7_divisible_by_7_l2389_238940

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 3-digit base 7 number --/
def isThreeDigitBase7 (n : ℕ) : Prop := sorry

/-- The greatest 3-digit base 7 number --/
def greatestThreeDigitBase7 : ℕ := 666

theorem greatest_three_digit_base7_divisible_by_7 :
  isThreeDigitBase7 greatestThreeDigitBase7 ∧
  base7ToDecimal greatestThreeDigitBase7 % 7 = 0 ∧
  ∀ n : ℕ, isThreeDigitBase7 n ∧ base7ToDecimal n % 7 = 0 →
    base7ToDecimal n ≤ base7ToDecimal greatestThreeDigitBase7 :=
sorry

end NUMINAMATH_CALUDE_greatest_three_digit_base7_divisible_by_7_l2389_238940


namespace NUMINAMATH_CALUDE_root_values_l2389_238979

theorem root_values (a b c d k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a * k^3 + b * k^2 + c * k + d = 0)
  (h2 : b * k^3 + c * k^2 + d * k + a = 0) :
  k = 1 ∨ k = -1 ∨ k = I ∨ k = -I :=
sorry

end NUMINAMATH_CALUDE_root_values_l2389_238979


namespace NUMINAMATH_CALUDE_least_fourth_integer_l2389_238957

theorem least_fourth_integer (a b c d : ℕ+) : 
  (a + b + c + d : ℚ) / 4 = 18 →
  a = 3 * b →
  b = c - 2 →
  (c : ℚ) = 1.5 * d →
  d ≥ 10 ∧ ∀ x : ℕ+, x < 10 → 
    ¬∃ a' b' c' : ℕ+, (a' + b' + c' + x : ℚ) / 4 = 18 ∧
                      a' = 3 * b' ∧
                      b' = c' - 2 ∧
                      (c' : ℚ) = 1.5 * x := by
  sorry

#check least_fourth_integer

end NUMINAMATH_CALUDE_least_fourth_integer_l2389_238957


namespace NUMINAMATH_CALUDE_cube_preserves_order_l2389_238939

theorem cube_preserves_order (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l2389_238939


namespace NUMINAMATH_CALUDE_minimum_races_for_top_three_l2389_238927

/-- Represents a horse in the race -/
structure Horse : Type :=
  (id : Nat)

/-- Represents a race with at most 5 horses -/
structure Race : Type :=
  (participants : Finset Horse)
  (size_constraint : participants.card ≤ 5)

/-- The set of all horses -/
def all_horses : Finset Horse := sorry

/-- The proposition that a given number of races is sufficient to determine the top 3 fastest horses -/
def can_determine_top_three (n : Nat) : Prop := sorry

/-- The proposition that a given number of races is necessary to determine the top 3 fastest horses -/
def is_necessary (n : Nat) : Prop := sorry

theorem minimum_races_for_top_three :
  (all_horses.card = 25) →
  (can_determine_top_three 7) ∧
  (∀ m : Nat, m < 7 → ¬(can_determine_top_three m)) ∧
  (is_necessary 7) :=
sorry

end NUMINAMATH_CALUDE_minimum_races_for_top_three_l2389_238927


namespace NUMINAMATH_CALUDE_airport_exchange_rate_l2389_238922

theorem airport_exchange_rate (euros : ℝ) (official_rate : ℝ) (airport_rate_factor : ℝ) :
  euros = 70 →
  official_rate = 5 →
  airport_rate_factor = 5 / 7 →
  (euros / official_rate) * airport_rate_factor = 10 := by
  sorry

end NUMINAMATH_CALUDE_airport_exchange_rate_l2389_238922


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2389_238985

theorem arithmetic_geometric_ratio (a : ℕ → ℝ) (d : ℝ) : 
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  d ≠ 0 →  -- non-zero common difference
  ∃ r, r = (a 3) / (a 2) ∧ r = (a 6) / (a 3) →  -- geometric sequence condition
  (a 3) / (a 2) = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2389_238985


namespace NUMINAMATH_CALUDE_altitude_length_right_triangle_l2389_238984

/-- Given a right triangle where the angle bisector divides the hypotenuse into segments
    of lengths p and q, the length of the altitude to the hypotenuse (m) is:
    m = (pq(p+q)) / (p^2 + q^2) -/
theorem altitude_length_right_triangle (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let m := (p * q * (p + q)) / (p^2 + q^2)
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 = b^2 + c^2 ∧
    (b / p = c / q) ∧
    m = (b * c) / a :=
by
  sorry

end NUMINAMATH_CALUDE_altitude_length_right_triangle_l2389_238984


namespace NUMINAMATH_CALUDE_semicircles_to_circle_area_ratio_l2389_238952

theorem semicircles_to_circle_area_ratio : 
  let r₁ : ℝ := 10  -- radius of the larger circle
  let r₂ : ℝ := 8   -- radius of the first semicircle
  let r₃ : ℝ := 6   -- radius of the second semicircle
  let circle_area := π * r₁^2
  let semicircle_area_1 := (π * r₂^2) / 2
  let semicircle_area_2 := (π * r₃^2) / 2
  let combined_semicircle_area := semicircle_area_1 + semicircle_area_2
  (combined_semicircle_area / circle_area) = (1 / 2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_semicircles_to_circle_area_ratio_l2389_238952


namespace NUMINAMATH_CALUDE_fifteenth_term_ratio_l2389_238910

-- Define the sums of arithmetic sequences
def U (n : ℕ) (c f : ℚ) : ℚ := n * (2 * c + (n - 1) * f) / 2
def V (n : ℕ) (g h : ℚ) : ℚ := n * (2 * g + (n - 1) * h) / 2

-- Define the ratio condition
def ratio_condition (n : ℕ) (c f g h : ℚ) : Prop :=
  U n c f / V n g h = (5 * n^2 + 3 * n + 2) / (3 * n^2 + 2 * n + 30)

-- Define the 15th term of each sequence
def term_15 (c f : ℚ) : ℚ := c + 14 * f

-- Theorem statement
theorem fifteenth_term_ratio 
  (c f g h : ℚ) 
  (h1 : ∀ (n : ℕ), n > 0 → ratio_condition n c f g h) :
  term_15 c f / term_15 g h = 125 / 99 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_ratio_l2389_238910


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2389_238929

theorem two_numbers_difference (x y : ℕ) : 
  x ∈ Finset.range 38 ∧ 
  y ∈ Finset.range 38 ∧ 
  x < y ∧ 
  (Finset.sum (Finset.range 38) id) - x - y = x * y →
  y - x = 10 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2389_238929


namespace NUMINAMATH_CALUDE_division_makes_equation_true_l2389_238982

theorem division_makes_equation_true : (6 / 3) + 4 - (2 - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_makes_equation_true_l2389_238982


namespace NUMINAMATH_CALUDE_duty_arrangements_count_l2389_238959

def staff_count : ℕ := 7
def days_count : ℕ := 7
def restricted_days : ℕ := 2
def restricted_staff : ℕ := 2

theorem duty_arrangements_count : 
  (staff_count.factorial) / ((staff_count - days_count).factorial) *
  ((days_count - restricted_days).factorial) / 
  ((days_count - restricted_days - restricted_staff).factorial) = 2400 := by
  sorry

end NUMINAMATH_CALUDE_duty_arrangements_count_l2389_238959


namespace NUMINAMATH_CALUDE_car_distance_problem_l2389_238983

/-- Proves that Car X travels 245 miles from when Car Y starts until both stop -/
theorem car_distance_problem :
  let speed_x : ℝ := 35  -- speed of Car X in miles per hour
  let speed_y : ℝ := 41  -- speed of Car Y in miles per hour
  let head_start_time : ℝ := 72 / 60  -- head start time for Car X in hours
  let head_start_distance : ℝ := speed_x * head_start_time  -- distance Car X travels before Car Y starts
  let catch_up_time : ℝ := head_start_distance / (speed_y - speed_x)  -- time it takes for Car Y to catch up
  let distance_x : ℝ := speed_x * catch_up_time  -- distance Car X travels while Car Y is moving
  distance_x = 245 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_problem_l2389_238983


namespace NUMINAMATH_CALUDE_laptop_sticker_price_l2389_238917

theorem laptop_sticker_price :
  ∀ (sticker_price : ℝ),
    (0.8 * sticker_price - 120 = 0.7 * sticker_price - 18) →
    sticker_price = 1020 := by
  sorry

end NUMINAMATH_CALUDE_laptop_sticker_price_l2389_238917


namespace NUMINAMATH_CALUDE_complex_power_evaluation_l2389_238901

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_evaluation :
  3 * i ^ 44 - 2 * i ^ 333 = 3 - 2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_power_evaluation_l2389_238901


namespace NUMINAMATH_CALUDE_root_sum_fraction_l2389_238980

theorem root_sum_fraction (a b c : ℝ) : 
  a^3 - 15*a^2 + 25*a - 10 = 0 →
  b^3 - 15*b^2 + 25*b - 10 = 0 →
  c^3 - 15*c^2 + 25*c - 10 = 0 →
  (a / (1/a + b*c)) + (b / (1/b + c*a)) + (c / (1/c + a*b)) = 175/11 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l2389_238980


namespace NUMINAMATH_CALUDE_four_digit_perfect_cubes_divisible_by_16_l2389_238935

theorem four_digit_perfect_cubes_divisible_by_16 :
  (∃! (count : ℕ), ∃ (S : Finset ℕ),
    S.card = count ∧
    (∀ n ∈ S, 1000 ≤ n ∧ n ≤ 9999) ∧
    (∀ n ∈ S, ∃ m : ℕ, n = m^3) ∧
    (∀ n ∈ S, n % 16 = 0) ∧
    (∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ m : ℕ, n = m^3) ∧ n % 16 = 0 → n ∈ S)) ∧
  count = 3 :=
sorry

end NUMINAMATH_CALUDE_four_digit_perfect_cubes_divisible_by_16_l2389_238935


namespace NUMINAMATH_CALUDE_binomial_12_6_l2389_238981

theorem binomial_12_6 : Nat.choose 12 6 = 924 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_6_l2389_238981


namespace NUMINAMATH_CALUDE_cube_of_m_equals_64_l2389_238912

theorem cube_of_m_equals_64 (m : ℕ) (h : 3^m = 81) : m^3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_m_equals_64_l2389_238912


namespace NUMINAMATH_CALUDE_constant_intersection_point_range_l2389_238989

/-- Given that when m ∈ ℝ, the function f(x) = m(x^2 - 1) + x - a has a constant 
    intersection point with the x-axis, then a ∈ ℝ when m = 0 and a ∈ [-1, 1] when m ≠ 0 -/
theorem constant_intersection_point_range (m a : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, m * (x^2 - 1) + x - a = 0 → x = k) → 
  ((m = 0 → a ∈ Set.univ) ∧ (m ≠ 0 → a ∈ Set.Icc (-1) 1)) :=
sorry

end NUMINAMATH_CALUDE_constant_intersection_point_range_l2389_238989


namespace NUMINAMATH_CALUDE_orvin_balloon_purchase_l2389_238972

def regular_price : ℕ := 4
def initial_balloons : ℕ := 35
def discount_ratio : ℚ := 1/2

def max_balloons : ℕ := 42

theorem orvin_balloon_purchase :
  let total_money := initial_balloons * regular_price
  let discounted_set_cost := 2 * regular_price + discount_ratio * regular_price
  let num_sets := total_money / discounted_set_cost
  num_sets * 3 = max_balloons :=
by sorry

end NUMINAMATH_CALUDE_orvin_balloon_purchase_l2389_238972


namespace NUMINAMATH_CALUDE_three_fifths_equivalence_l2389_238958

/-- Proves the equivalence of various representations of 3/5 -/
theorem three_fifths_equivalence :
  (3 : ℚ) / 5 = 12 / 20 ∧
  (3 : ℚ) / 5 = (10 : ℚ) / (50 / 3) ∧
  (3 : ℚ) / 5 = 60 / 100 ∧
  (3 : ℚ) / 5 = 0.60 ∧
  (3 : ℚ) / 5 = 60 / 100 := by
  sorry

#check three_fifths_equivalence

end NUMINAMATH_CALUDE_three_fifths_equivalence_l2389_238958


namespace NUMINAMATH_CALUDE_min_value_fraction_min_value_is_four_min_value_achieved_min_value_fraction_is_four_l2389_238965

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 2) : 
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 2 → (x + y) / (x * y * z) ≤ (a + b) / (a * b * c) :=
by sorry

theorem min_value_is_four (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 2) : 
  (x + y) / (x * y * z) ≥ 4 :=
by sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 2) : 
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 2 ∧ (a + b) / (a * b * c) = 4 :=
by sorry

theorem min_value_fraction_is_four :
  ∃ m : ℝ, m = 4 ∧ 
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 2 → (x + y) / (x * y * z) ≥ m) ∧
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 2 ∧ (a + b) / (a * b * c) = m) :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_min_value_is_four_min_value_achieved_min_value_fraction_is_four_l2389_238965


namespace NUMINAMATH_CALUDE_distance_from_point_to_x_axis_l2389_238963

/-- The distance from a point to the x-axis in a Cartesian coordinate system -/
def distance_to_x_axis (x y : ℝ) : ℝ :=
  |y|

theorem distance_from_point_to_x_axis :
  distance_to_x_axis 3 (-4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_point_to_x_axis_l2389_238963


namespace NUMINAMATH_CALUDE_box_weight_sum_l2389_238993

theorem box_weight_sum (a b c d : ℝ) 
  (h1 : a + b + c = 135)
  (h2 : a + b + d = 139)
  (h3 : a + c + d = 142)
  (h4 : b + c + d = 145)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  a + b + c + d = 187 := by
  sorry

end NUMINAMATH_CALUDE_box_weight_sum_l2389_238993


namespace NUMINAMATH_CALUDE_parabola_symmetry_l2389_238949

def C₁ (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 1

def C₂ (x : ℝ) : ℝ := 2 * (x - 3)^2 - 4 * (x - 3) - 1

def is_symmetry_line (f g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a - x) = g (a + x)

theorem parabola_symmetry :
  is_symmetry_line C₁ C₂ (5/2) :=
sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l2389_238949


namespace NUMINAMATH_CALUDE_group_size_is_eight_l2389_238964

/-- The number of people in a group, given certain weight conditions -/
def number_of_people : ℕ :=
  let weight_increase_per_person : ℕ := 5
  let weight_difference : ℕ := 75 - 35
  weight_difference / weight_increase_per_person

theorem group_size_is_eight :
  number_of_people = 8 :=
by
  -- Proof goes here
  sorry

#eval number_of_people  -- Should output 8

end NUMINAMATH_CALUDE_group_size_is_eight_l2389_238964


namespace NUMINAMATH_CALUDE_vector_parallel_and_perpendicular_l2389_238944

def a (x : ℝ) : Fin 2 → ℝ := ![x, x + 2]
def b : Fin 2 → ℝ := ![1, 2]

theorem vector_parallel_and_perpendicular :
  (∃ (k : ℝ), a 2 = k • b) ∧
  (a (1/3) - b) • b = 0 := by sorry

end NUMINAMATH_CALUDE_vector_parallel_and_perpendicular_l2389_238944


namespace NUMINAMATH_CALUDE_abs_eq_solution_l2389_238971

theorem abs_eq_solution (x : ℝ) : |x + 1| = 2*x + 4 ↔ x = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_solution_l2389_238971


namespace NUMINAMATH_CALUDE_divides_two_pow_36_minus_1_l2389_238992

theorem divides_two_pow_36_minus_1 : 
  ∃! (n : ℕ), 40 ≤ n ∧ n ≤ 50 ∧ (2^36 - 1) % n = 0 ∧ n = 49 := by
  sorry

end NUMINAMATH_CALUDE_divides_two_pow_36_minus_1_l2389_238992


namespace NUMINAMATH_CALUDE_average_weight_is_15_l2389_238969

def regression_weight (age : ℕ) : ℝ := 2 * age + 7

def children_ages : List ℕ := [2, 3, 3, 5, 2, 6, 7, 3, 4, 5]

theorem average_weight_is_15 :
  (children_ages.map regression_weight).sum / children_ages.length = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_is_15_l2389_238969


namespace NUMINAMATH_CALUDE_opposite_expressions_l2389_238919

theorem opposite_expressions (x : ℚ) : x = -3/2 → (3 + x/3 = -(x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_expressions_l2389_238919


namespace NUMINAMATH_CALUDE_sphere_radii_difference_l2389_238960

theorem sphere_radii_difference (r₁ r₂ : ℝ) 
  (h_surface : 4 * π * (r₁^2 - r₂^2) = 48 * π) 
  (h_circumference : 2 * π * (r₁ + r₂) = 12 * π) : 
  |r₁ - r₂| = 2 := by
sorry

end NUMINAMATH_CALUDE_sphere_radii_difference_l2389_238960


namespace NUMINAMATH_CALUDE_yunas_grandfather_age_l2389_238942

/-- Proves the age of Yuna's grandfather given the ages and age differences of family members. -/
theorem yunas_grandfather_age 
  (yuna_age : ℕ) 
  (father_age_diff : ℕ) 
  (grandfather_age_diff : ℕ) 
  (h1 : yuna_age = 9)
  (h2 : father_age_diff = 27)
  (h3 : grandfather_age_diff = 23) : 
  yuna_age + father_age_diff + grandfather_age_diff = 59 :=
by sorry

end NUMINAMATH_CALUDE_yunas_grandfather_age_l2389_238942


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2389_238907

/-- The ellipse equation -/
def on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The line equation -/
def on_line (x y : ℝ) : Prop := 4*x - 2*y - 3 = 0

/-- Symmetry about the line -/
def symmetric_about_line (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), on_line x₀ y₀ ∧ x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem vector_sum_magnitude (x₁ y₁ x₂ y₂ : ℝ) :
  on_ellipse x₁ y₁ → on_ellipse x₂ y₂ → symmetric_about_line x₁ y₁ x₂ y₂ →
  (x₁ + x₂)^2 + (y₁ + y₂)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2389_238907


namespace NUMINAMATH_CALUDE_marble_game_solution_l2389_238945

/-- Represents a player in the game -/
inductive Player
| A
| B
| C

/-- Represents the game state -/
structure GameState where
  p : ℕ
  q : ℕ
  r : ℕ
  rounds : ℕ
  final_marbles : Player → ℕ
  b_last_round : ℕ

/-- The theorem statement -/
theorem marble_game_solution (g : GameState) 
  (h1 : g.p < g.q ∧ g.q < g.r)
  (h2 : g.rounds ≥ 2)
  (h3 : g.final_marbles Player.A = 20)
  (h4 : g.final_marbles Player.B = 10)
  (h5 : g.final_marbles Player.C = 9)
  (h6 : g.b_last_round = g.r) :
  ∃ (first_round : Player → ℕ), first_round Player.B = 4 := by
  sorry

end NUMINAMATH_CALUDE_marble_game_solution_l2389_238945


namespace NUMINAMATH_CALUDE_checker_in_center_l2389_238970

/-- Represents a square board -/
structure Board :=
  (size : ℕ)

/-- Represents a checker placement on the board -/
structure Placement :=
  (board : Board)
  (positions : Finset (ℕ × ℕ))

/-- Defines symmetry with respect to both main diagonals -/
def is_symmetric (p : Placement) : Prop :=
  ∀ (i j : ℕ), (i, j) ∈ p.positions ↔
    (j, i) ∈ p.positions ∧
    (p.board.size - 1 - i, p.board.size - 1 - j) ∈ p.positions

/-- The central cell of the board -/
def central_cell (b : Board) : ℕ × ℕ :=
  (b.size / 2, b.size / 2)

/-- The main theorem -/
theorem checker_in_center (p : Placement)
  (h_size : p.board.size = 25)
  (h_count : p.positions.card = 25)
  (h_sym : is_symmetric p) :
  central_cell p.board ∈ p.positions :=
sorry

end NUMINAMATH_CALUDE_checker_in_center_l2389_238970


namespace NUMINAMATH_CALUDE_max_value_of_f_in_interval_l2389_238915

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem max_value_of_f_in_interval :
  ∃ (m : ℝ), m = 2 ∧ 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f x ≤ m) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ f x = m) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_in_interval_l2389_238915


namespace NUMINAMATH_CALUDE_correct_total_cost_l2389_238905

/-- The cost of a single sandwich -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda -/
def soda_cost : ℕ := 3

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 4

/-- The number of sodas purchased -/
def num_sodas : ℕ := 5

/-- The total cost of the purchase -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem correct_total_cost : total_cost = 31 := by
  sorry

end NUMINAMATH_CALUDE_correct_total_cost_l2389_238905


namespace NUMINAMATH_CALUDE_isosceles_triangle_angles_l2389_238934

-- Define the triangle
structure IsoscelesTriangle where
  base_angle : Real
  inscribed_square_side : Real
  inscribed_circle_radius : Real

-- Define the conditions
def triangle_conditions (t : IsoscelesTriangle) : Prop :=
  t.inscribed_square_side / t.inscribed_circle_radius = 8 / 5

-- Theorem statement
theorem isosceles_triangle_angles (t : IsoscelesTriangle) 
  (h : triangle_conditions t) : 
  t.base_angle = 2 * Real.arctan (1 / 2) ∧ 
  π - 2 * t.base_angle = π - 4 * Real.arctan (1 / 2) := by
  sorry

#check isosceles_triangle_angles

end NUMINAMATH_CALUDE_isosceles_triangle_angles_l2389_238934


namespace NUMINAMATH_CALUDE_complex_modulus_one_l2389_238932

theorem complex_modulus_one (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : 
  Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l2389_238932


namespace NUMINAMATH_CALUDE_expression_evaluation_l2389_238925

theorem expression_evaluation : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) + 1 / 4 = 37 / 60 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2389_238925


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2389_238975

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 + a*b - 3 = 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → x^2 + x*y - 3 = 0 → 4*x + y ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2389_238975


namespace NUMINAMATH_CALUDE_octal_127_equals_binary_1010111_l2389_238968

def octal_to_decimal (x : ℕ) : ℕ := 
  (x % 10) + 8 * ((x / 10) % 10) + 64 * (x / 100)

def decimal_to_binary (x : ℕ) : List ℕ :=
  if x = 0 then [0]
  else 
    let rec aux (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else aux (n / 2) ((n % 2) :: acc)
    aux x []

theorem octal_127_equals_binary_1010111 : 
  decimal_to_binary (octal_to_decimal 127) = [1, 0, 1, 0, 1, 1, 1] := by
  sorry

#eval octal_to_decimal 127
#eval decimal_to_binary (octal_to_decimal 127)

end NUMINAMATH_CALUDE_octal_127_equals_binary_1010111_l2389_238968


namespace NUMINAMATH_CALUDE_exists_positive_a_leq_inverse_l2389_238978

theorem exists_positive_a_leq_inverse : ∃ a : ℝ, a > 0 ∧ a ≤ 1 / a := by
  sorry

end NUMINAMATH_CALUDE_exists_positive_a_leq_inverse_l2389_238978


namespace NUMINAMATH_CALUDE_sphere_volume_increase_l2389_238921

theorem sphere_volume_increase (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * (2 * r)^3) / (4 / 3 * Real.pi * r^3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_increase_l2389_238921


namespace NUMINAMATH_CALUDE_expression_one_proof_l2389_238902

theorem expression_one_proof : 1 + (-2) + |(-2) - 3| - 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_one_proof_l2389_238902


namespace NUMINAMATH_CALUDE_savings_account_decrease_l2389_238924

theorem savings_account_decrease (initial_balance : ℝ) (increase_percent : ℝ) (final_balance_percent : ℝ) :
  initial_balance = 125 →
  increase_percent = 25 →
  final_balance_percent = 100 →
  let increased_balance := initial_balance * (1 + increase_percent / 100)
  let final_balance := initial_balance * (final_balance_percent / 100)
  let decrease_amount := increased_balance - final_balance
  let decrease_percent := (decrease_amount / increased_balance) * 100
  decrease_percent = 20 := by
sorry

end NUMINAMATH_CALUDE_savings_account_decrease_l2389_238924


namespace NUMINAMATH_CALUDE_only_D_is_valid_assignment_l2389_238906

-- Define what constitutes a valid assignment statement
def is_valid_assignment (s : String) : Prop :=
  ∃ (var : String) (expr : String), 
    s = var ++ "=" ++ expr ∧ 
    var ≠ expr ∧
    var.length = 1 ∧
    var.all Char.isLower

-- Define the given options
def option_A : String := "5=a"
def option_B : String := "a+2=a"
def option_C : String := "a=b=4"
def option_D : String := "a=2*a"

-- Theorem statement
theorem only_D_is_valid_assignment :
  ¬(is_valid_assignment option_A) ∧
  ¬(is_valid_assignment option_B) ∧
  ¬(is_valid_assignment option_C) ∧
  is_valid_assignment option_D :=
sorry

end NUMINAMATH_CALUDE_only_D_is_valid_assignment_l2389_238906


namespace NUMINAMATH_CALUDE_power_difference_solutions_l2389_238999

theorem power_difference_solutions :
  ∀ m n : ℕ+,
  (2^(m : ℕ) - 3^(n : ℕ) = 1 ∧ 3^(n : ℕ) - 2^(m : ℕ) = 1) ↔ 
  ((m = 2 ∧ n = 1) ∨ (m = 1 ∧ n = 1) ∨ (m = 3 ∧ n = 2)) :=
by sorry

end NUMINAMATH_CALUDE_power_difference_solutions_l2389_238999


namespace NUMINAMATH_CALUDE_total_book_price_l2389_238928

/-- Given the following conditions:
  - Total number of books: 90
  - Math books cost: $4 each
  - History books cost: $5 each
  - Number of math books: 60
  Prove that the total price of all books is $390 -/
theorem total_book_price (total_books : Nat) (math_book_price history_book_price : Nat) (math_books : Nat) :
  total_books = 90 →
  math_book_price = 4 →
  history_book_price = 5 →
  math_books = 60 →
  math_books * math_book_price + (total_books - math_books) * history_book_price = 390 := by
  sorry

#check total_book_price

end NUMINAMATH_CALUDE_total_book_price_l2389_238928


namespace NUMINAMATH_CALUDE_largest_n_for_integer_factors_l2389_238913

def polynomial (n : ℤ) (x : ℤ) : ℤ := 3 * x^2 + n * x + 72

def has_integer_linear_factors (n : ℤ) : Prop :=
  ∃ (a b : ℤ), ∀ x, polynomial n x = (3*x + a) * (x + b)

theorem largest_n_for_integer_factors :
  (∃ n : ℤ, has_integer_linear_factors n) ∧
  (∀ m : ℤ, has_integer_linear_factors m → m ≤ 217) ∧
  has_integer_linear_factors 217 :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_integer_factors_l2389_238913


namespace NUMINAMATH_CALUDE_max_area_at_midline_l2389_238962

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a line parallel to AC
def ParallelLine (t : Triangle) (M N : ℝ × ℝ) : Prop :=
  -- Add appropriate condition for parallel lines
  sorry

-- Define the rectangle MNPQ
structure Rectangle (t : Triangle) :=
  (M N P Q : ℝ × ℝ)
  (parallel : ParallelLine t M N)

-- Define the area of a rectangle
def area (r : Rectangle t) : ℝ :=
  sorry

-- Define the midline of a triangle
def Midline (t : Triangle) (M N : ℝ × ℝ) : Prop :=
  -- Add appropriate condition for midline
  sorry

-- Theorem statement
theorem max_area_at_midline (t : Triangle) :
  ∀ (r : Rectangle t), 
    Midline t r.M r.N → 
    ∀ (r' : Rectangle t), area r ≥ area r' :=
sorry

end NUMINAMATH_CALUDE_max_area_at_midline_l2389_238962


namespace NUMINAMATH_CALUDE_mayoral_election_votes_l2389_238914

theorem mayoral_election_votes (Z : ℕ) (hZ : Z = 25000) :
  let Y := (3 / 5 : ℚ) * Z
  let X := (8 / 5 : ℚ) * Y
  X = 24000 := by
  sorry

end NUMINAMATH_CALUDE_mayoral_election_votes_l2389_238914


namespace NUMINAMATH_CALUDE_quadratic_polynomial_proof_l2389_238987

theorem quadratic_polynomial_proof (p q : ℝ) : 
  (∃ a b : ℝ, a + b + p + q = 2 ∧ a * b * p * q = 12 ∧ a + b = -p ∧ a * b = q) →
  p = 3 ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_proof_l2389_238987


namespace NUMINAMATH_CALUDE_max_value_ab_l2389_238948

theorem max_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : (1 : ℝ) * (1 : ℝ) + (2 * a - 1) * (-b) = 0) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 
  (1 : ℝ) * (1 : ℝ) + (2 * x - 1) * (-y) = 0 → 
  x * y ≤ a * b ∧ a * b ≤ (1/8 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_max_value_ab_l2389_238948


namespace NUMINAMATH_CALUDE_coprime_iff_no_common_prime_factor_l2389_238916

theorem coprime_iff_no_common_prime_factor (a b : ℕ) : 
  Nat.gcd a b = 1 ↔ ¬ ∃ (p : ℕ), Nat.Prime p ∧ p ∣ a ∧ p ∣ b := by
  sorry

end NUMINAMATH_CALUDE_coprime_iff_no_common_prime_factor_l2389_238916


namespace NUMINAMATH_CALUDE_airport_distance_l2389_238926

-- Define the problem parameters
def initial_speed : ℝ := 45
def speed_increase : ℝ := 20
def late_time : ℝ := 0.75  -- 45 minutes in hours
def early_time : ℝ := 0.25  -- 15 minutes in hours

-- Define the theorem
theorem airport_distance : ∃ (d : ℝ), d = 241.875 ∧ 
  ∃ (t : ℝ), 
    d = initial_speed * (t + late_time) ∧
    d - initial_speed = (initial_speed + speed_increase) * (t - (1 + early_time)) :=
by
  sorry


end NUMINAMATH_CALUDE_airport_distance_l2389_238926


namespace NUMINAMATH_CALUDE_library_wall_arrangement_l2389_238954

/-- Proves that the maximum number of desk-bookcase pairs on a 15m wall leaves 3m of space --/
theorem library_wall_arrangement (wall_length : ℝ) (desk_length : ℝ) (bookcase_length : ℝ) 
  (space_between : ℝ) (h1 : wall_length = 15) (h2 : desk_length = 2) 
  (h3 : bookcase_length = 1.5) (h4 : space_between = 0.5) : 
  ∃ (n : ℕ) (leftover : ℝ), 
    n * (desk_length + bookcase_length + space_between) + leftover = wall_length ∧ 
    leftover = 3 ∧ 
    ∀ m : ℕ, m > n → m * (desk_length + bookcase_length + space_between) > wall_length := by
  sorry

end NUMINAMATH_CALUDE_library_wall_arrangement_l2389_238954


namespace NUMINAMATH_CALUDE_platform_length_is_605_l2389_238995

/-- Calculates the length of a platform given train movement parameters. -/
def platformLength (
  platformPassTime : Real
) (manPassTime : Real)
  (manDistance : Real)
  (initialSpeed : Real)
  (acceleration : Real) : Real :=
  let trainLength := manPassTime * initialSpeed + 0.5 * acceleration * manPassTime ^ 2 - manDistance
  let platformPassDistance := platformPassTime * initialSpeed + 0.5 * acceleration * platformPassTime ^ 2
  platformPassDistance - trainLength

/-- The length of the platform is 605 meters. -/
theorem platform_length_is_605 :
  platformLength 40 20 5 15 0.5 = 605 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_is_605_l2389_238995


namespace NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l2389_238946

/-- A complex number z is pure imaginary if its real part is zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

/-- The problem statement -/
theorem complex_fraction_pure_imaginary (a : ℝ) :
  is_pure_imaginary ((a + 3 * Complex.I) / (1 + 2 * Complex.I)) → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l2389_238946


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l2389_238908

theorem average_of_a_and_b (a b : ℝ) : 
  (3 + 5 + 7 + a + b) / 5 = 15 → (a + b) / 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l2389_238908


namespace NUMINAMATH_CALUDE_range_of_a_l2389_238909

theorem range_of_a (a : ℝ) : (∃ x : ℝ, Real.exp (2 * x) - (a - 3) * Real.exp x + 4 - 3 * a > 0) → a ≤ 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2389_238909


namespace NUMINAMATH_CALUDE_average_parking_cost_senior_student_l2389_238936

/-- Calculates the average hourly parking cost for a senior citizen or student
    parking for 9 hours on a weekend, given the specified fee structure. -/
theorem average_parking_cost_senior_student (base_cost : ℝ) (additional_hourly_rate : ℝ)
  (weekend_surcharge : ℝ) (discount_rate : ℝ) (parking_duration : ℕ) :
  base_cost = 20 →
  additional_hourly_rate = 1.75 →
  weekend_surcharge = 5 →
  discount_rate = 0.1 →
  parking_duration = 9 →
  let total_cost := base_cost + (parking_duration - 2 : ℕ) * additional_hourly_rate + weekend_surcharge
  let discounted_cost := total_cost * (1 - discount_rate)
  let average_hourly_cost := discounted_cost / parking_duration
  average_hourly_cost = 3.725 := by
sorry

end NUMINAMATH_CALUDE_average_parking_cost_senior_student_l2389_238936


namespace NUMINAMATH_CALUDE_tyrah_pencils_l2389_238950

theorem tyrah_pencils (sarah tim tyrah : ℕ) : 
  tyrah = 6 * sarah →
  tim = 8 * sarah →
  tim = 16 →
  tyrah = 12 :=
by sorry

end NUMINAMATH_CALUDE_tyrah_pencils_l2389_238950


namespace NUMINAMATH_CALUDE_original_quantities_l2389_238933

/-- The original planned quantities of products A and B -/
def original_plan (x y : ℕ) : Prop :=
  ∃ (a b : ℝ), 
    -- Original plan: spend 1500 yuan
    a * x + b * y = 1500 ∧
    -- New scenario 1
    (a + 1.5) * (x - 10) + (b + 1) * y = 1529 ∧
    -- New scenario 2
    (a + 1) * (x - 5) + (b + 1) * y = 1563.5 ∧
    -- Constraint
    205 < 2 * x + y ∧ 2 * x + y < 210

theorem original_quantities : 
  ∃ (x y : ℕ), original_plan x y ∧ x = 76 ∧ y = 55 := by
  sorry

end NUMINAMATH_CALUDE_original_quantities_l2389_238933


namespace NUMINAMATH_CALUDE_largest_integer_less_than_85_remainder_2_mod_6_l2389_238991

theorem largest_integer_less_than_85_remainder_2_mod_6 : 
  ∃ (n : ℤ), n < 85 ∧ n % 6 = 2 ∧ ∀ (m : ℤ), m < 85 ∧ m % 6 = 2 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_85_remainder_2_mod_6_l2389_238991


namespace NUMINAMATH_CALUDE_part_one_part_two_l2389_238967

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- Theorem for part (1)
theorem part_one (a b : ℝ) (ha : a ≠ 0) 
  (h_solution_set : ∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) :
  a = -1 ∧ b = 4 := by sorry

-- Theorem for part (2)
theorem part_two (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_f_1 : f a b 1 = 2) :
  (1 / a + 4 / b) ≥ 9 ∧ ∃ (a b : ℝ), 1 / a + 4 / b = 9 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2389_238967


namespace NUMINAMATH_CALUDE_intersection_perpendicular_l2389_238990

/-- The line y = x - 2 intersects the parabola y^2 = 2x at points A and B. 
    This theorem proves that OA ⊥ OB, where O is the origin (0, 0). -/
theorem intersection_perpendicular (A B : ℝ × ℝ) : 
  (∃ x y : ℝ, A = (x, y) ∧ y = x - 2 ∧ y^2 = 2*x) →
  (∃ x y : ℝ, B = (x, y) ∧ y = x - 2 ∧ y^2 = 2*x) →
  A ≠ B →
  let O : ℝ × ℝ := (0, 0)
  (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 0 :=
by sorry


end NUMINAMATH_CALUDE_intersection_perpendicular_l2389_238990


namespace NUMINAMATH_CALUDE_three_digit_subtraction_problem_l2389_238974

theorem three_digit_subtraction_problem :
  ∀ h t u : ℕ,
  h ≤ 9 ∧ t ≤ 9 ∧ u ≤ 9 →  -- Ensure single-digit numbers
  u = h - 5 →
  (100 * h + 10 * t + u) - (100 * h + 10 * u + t) = 96 →
  h = 5 ∧ t = 9 ∧ u = 0 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_subtraction_problem_l2389_238974


namespace NUMINAMATH_CALUDE_unique_two_digit_multiple_l2389_238951

theorem unique_two_digit_multiple : ∃! t : ℕ, 
  10 ≤ t ∧ t < 100 ∧ (13 * t) % 100 = 42 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_multiple_l2389_238951


namespace NUMINAMATH_CALUDE_max_area_2014_l2389_238961

/-- A polygon drawn on a grid with sides following grid lines -/
structure GridPolygon where
  perimeter : ℕ
  sides_follow_grid : Bool

/-- The maximum area of a grid polygon given its perimeter -/
def max_area (p : GridPolygon) : ℕ :=
  (p.perimeter / 4)^2 - if p.perimeter % 4 == 2 then 1/4 else 0

/-- Theorem stating the maximum area of a grid polygon with perimeter 2014 -/
theorem max_area_2014 :
  ∀ (p : GridPolygon), p.perimeter = 2014 → p.sides_follow_grid → max_area p = 253512 := by
  sorry


end NUMINAMATH_CALUDE_max_area_2014_l2389_238961


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l2389_238994

/-- The line passing through the intersection points of two circles -/
theorem intersection_line_of_circles (x y : ℝ) : 
  (x - 2)^2 + (y + 3)^2 = 8^2 →
  (x + 5)^2 + (y - 7)^2 = 136 →
  x + y = 4.35 :=
by
  sorry


end NUMINAMATH_CALUDE_intersection_line_of_circles_l2389_238994


namespace NUMINAMATH_CALUDE_sum_has_five_digits_l2389_238976

def is_nonzero_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def number_to_digits (n : ℕ) : ℕ := 
  if n = 0 then 1 else (Nat.log 10 n).succ

theorem sum_has_five_digits (A B : ℕ) 
  (hA : is_nonzero_digit A) (hB : is_nonzero_digit B) : 
  number_to_digits (19876 + (10000 * A + 1000 * B + 320) + (200 * B + 1)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_has_five_digits_l2389_238976


namespace NUMINAMATH_CALUDE_continuous_additive_function_is_linear_l2389_238943

-- Define the property of the function
def SatisfiesAdditiveProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

-- State the theorem
theorem continuous_additive_function_is_linear
  (f : ℝ → ℝ)
  (hf_cont : Continuous f)
  (hf_add : SatisfiesAdditiveProperty f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end NUMINAMATH_CALUDE_continuous_additive_function_is_linear_l2389_238943


namespace NUMINAMATH_CALUDE_chess_tournament_players_l2389_238955

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- number of players excluding the lowest 8
  total_players : ℕ := n + 8
  
  -- Each player played exactly one game against each other player
  total_games : ℕ := total_players.choose 2
  
  -- Point distribution condition
  point_distribution : 
    2 * n.choose 2 + 56 = (total_players * (total_players - 1)) / 2

/-- The theorem stating that the total number of players in the tournament is 21 -/
theorem chess_tournament_players : 
  ∀ t : ChessTournament, t.total_players = 21 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l2389_238955


namespace NUMINAMATH_CALUDE_mystery_book_shelves_l2389_238931

theorem mystery_book_shelves (books_per_shelf : ℕ) (picture_book_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 4 →
  picture_book_shelves = 3 →
  total_books = 32 →
  (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf = 5 :=
by sorry

end NUMINAMATH_CALUDE_mystery_book_shelves_l2389_238931


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l2389_238900

theorem quadratic_minimum_value (k : ℝ) : 
  (∀ x y : ℝ, 5*x^2 - 8*k*x*y + (4*k^2 + 3)*y^2 - 10*x - 6*y + 9 ≥ 0) ∧ 
  (∃ x y : ℝ, 5*x^2 - 8*k*x*y + (4*k^2 + 3)*y^2 - 10*x - 6*y + 9 = 0) →
  k = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l2389_238900


namespace NUMINAMATH_CALUDE_abs_rational_inequality_l2389_238920

theorem abs_rational_inequality (x : ℝ) : 
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_abs_rational_inequality_l2389_238920


namespace NUMINAMATH_CALUDE_increase_dimension_theorem_l2389_238947

/-- Represents a rectangle with length and width --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: If increasing both length and width of a rectangle by x feet
    increases its perimeter by 16 feet, then x must be 4 feet --/
theorem increase_dimension_theorem (r : Rectangle) (x : ℝ) :
  perimeter { length := r.length + x, width := r.width + x } - perimeter r = 16 →
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_increase_dimension_theorem_l2389_238947


namespace NUMINAMATH_CALUDE_fraction_simplification_l2389_238996

theorem fraction_simplification :
  (2 * Real.sqrt 3) / (Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5) = 
  (Real.sqrt 6 + 3 - Real.sqrt 15) / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2389_238996


namespace NUMINAMATH_CALUDE_identity_is_increasing_proportional_l2389_238998

/-- A proportional function where y increases as x increases -/
def increasing_proportional_function (f : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧
  (∃ k : ℝ, ∀ x : ℝ, f x = k * x)

/-- The function f(x) = x is an increasing proportional function -/
theorem identity_is_increasing_proportional : increasing_proportional_function (λ x : ℝ => x) := by
  sorry


end NUMINAMATH_CALUDE_identity_is_increasing_proportional_l2389_238998


namespace NUMINAMATH_CALUDE_alice_game_theorem_l2389_238923

/-- The game state, representing the positions of the red and blue beads -/
structure GameState where
  red : ℚ
  blue : ℚ

/-- The move function that updates the game state -/
def move (r : ℚ) (state : GameState) (k : ℤ) (moveRed : Bool) : GameState :=
  if moveRed then
    { red := state.blue + r^k * (state.red - state.blue), blue := state.blue }
  else
    { red := state.red, blue := state.red + r^k * (state.blue - state.red) }

/-- Predicate to check if a rational number is of the form (b+1)/b for 1 ≤ b ≤ 1010 -/
def isValidR (r : ℚ) : Prop :=
  ∃ b : ℕ, 1 ≤ b ∧ b ≤ 1010 ∧ r = (b + 1) / b

/-- Main theorem statement -/
theorem alice_game_theorem (r : ℚ) (hr : r > 1) :
  (∃ (moves : List (ℤ × Bool)), moves.length ≤ 2021 ∧
    (moves.foldl (λ state (k, moveRed) => move r state k moveRed)
      { red := 0, blue := 1 }).red = 1) ↔
  isValidR r :=
sorry

end NUMINAMATH_CALUDE_alice_game_theorem_l2389_238923


namespace NUMINAMATH_CALUDE_circle_symmetry_and_properties_l2389_238966

-- Define the circle C1 and line l
def C1 (m : ℝ) (x y : ℝ) : Prop := (x + 1)^2 + (y - 3*m - 3)^2 = 4*m^2
def l (m : ℝ) (x y : ℝ) : Prop := y = x + m + 2

-- Define the circle C2
def C2 (m : ℝ) (x y : ℝ) : Prop := (x - 2*m - 1)^2 + (y - m - 1)^2 = 4*m^2

-- Define the line on which centers of C2 lie
def centerLine (x y : ℝ) : Prop := x - 2*y + 1 = 0

-- Define the common tangent line
def commonTangent (x y : ℝ) : Prop := y = -3/4 * x + 7/4

theorem circle_symmetry_and_properties 
  (m : ℝ) (h : m ≠ 0) :
  (∀ x y, C2 m x y ↔ 
    ∃ x' y', C1 m x' y' ∧ l m ((x + x') / 2) ((y + y') / 2)) ∧ 
  (∀ m x y, C2 m x y → centerLine x y) ∧
  (∀ m x y, C2 m x y → ∃ x₀ y₀, commonTangent x₀ y₀ ∧ 
    (x₀ - x)^2 + (y₀ - y)^2 = ((x - (2*m + 1))^2 + (y - (m + 1))^2) / 4) :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_and_properties_l2389_238966


namespace NUMINAMATH_CALUDE_group_size_proof_l2389_238904

theorem group_size_proof (total_collection : ℚ) (h1 : total_collection = 92.16) : ∃ n : ℕ, 
  (n : ℚ) * (n : ℚ) / 100 = total_collection ∧ n = 96 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l2389_238904


namespace NUMINAMATH_CALUDE_largest_a_value_l2389_238988

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_triangular (n : ℕ) : Prop := ∃ m : ℕ, n = m * (m + 1) / 2

def valid_phone_number (a b c d e f g h i j : ℕ) : Prop :=
  a > b ∧ b > c ∧
  d > e ∧ e > f ∧
  g > h ∧ h > i ∧ i > j ∧
  is_square d ∧ is_square e ∧ is_square f ∧
  is_triangular g ∧ is_triangular h ∧ is_triangular i ∧ is_triangular j ∧
  a + b + c = 10 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
  g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
  h ≠ i ∧ h ≠ j ∧
  i ≠ j

theorem largest_a_value :
  ∀ a b c d e f g h i j : ℕ,
  valid_phone_number a b c d e f g h i j →
  a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_a_value_l2389_238988
