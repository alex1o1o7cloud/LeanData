import Mathlib

namespace NUMINAMATH_CALUDE_polygon_area_is_1800_l1228_122854

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The vertices of the polygon -/
def vertices : List Point := [
  ⟨0, 0⟩, ⟨15, 0⟩, ⟨45, 30⟩, ⟨45, 45⟩, ⟨30, 45⟩, ⟨0, 15⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vs : List Point) : ℝ :=
  sorry

/-- The theorem stating that the area of the given polygon is 1800 square units -/
theorem polygon_area_is_1800 : polygonArea vertices = 1800 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_is_1800_l1228_122854


namespace NUMINAMATH_CALUDE_f_2_nonneg_necessary_not_sufficient_l1228_122881

/-- A quadratic function f(x) = ax^2 + bx -/
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

/-- f(x) is monotonically increasing on (1, +∞) -/
def monotonically_increasing_on_interval (a b : ℝ) : Prop :=
  ∀ x y, 1 < x ∧ x < y → f a b x < f a b y

/-- f(2) ≥ 0 is a necessary but not sufficient condition for
    f(x) to be monotonically increasing on (1, +∞) -/
theorem f_2_nonneg_necessary_not_sufficient (a b : ℝ) :
  (∀ a b, monotonically_increasing_on_interval a b → f a b 2 ≥ 0) ∧
  ¬(∀ a b, f a b 2 ≥ 0 → monotonically_increasing_on_interval a b) :=
by sorry

end NUMINAMATH_CALUDE_f_2_nonneg_necessary_not_sufficient_l1228_122881


namespace NUMINAMATH_CALUDE_probability_larger_than_40_l1228_122820

def digits : Finset Nat := {1, 2, 3, 4, 5}

def is_valid_selection (a b : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ a ≠ b

def is_larger_than_40 (a b : Nat) : Prop :=
  is_valid_selection a b ∧ 10 * a + b > 40

def total_selections : Nat :=
  digits.card * (digits.card - 1)

def favorable_selections : Nat :=
  (digits.filter (λ x => x ≥ 4)).card * (digits.card - 1)

theorem probability_larger_than_40 :
  (favorable_selections : ℚ) / total_selections = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_larger_than_40_l1228_122820


namespace NUMINAMATH_CALUDE_blue_cube_problem_l1228_122871

theorem blue_cube_problem (n : ℕ) (h : n > 0) :
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 3 :=
by sorry

end NUMINAMATH_CALUDE_blue_cube_problem_l1228_122871


namespace NUMINAMATH_CALUDE_union_equals_interval_l1228_122844

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≥ -1}
def B : Set ℝ := {y : ℝ | y ≥ 1}

-- Define the interval [-1, +∞)
def interval : Set ℝ := {x : ℝ | x ≥ -1}

-- Theorem stating that the union of A and B is equal to the interval [-1, +∞)
theorem union_equals_interval : A ∪ B = interval := by sorry

end NUMINAMATH_CALUDE_union_equals_interval_l1228_122844


namespace NUMINAMATH_CALUDE_profit_ratio_theorem_l1228_122840

/-- Represents the investment of a partner -/
structure Investment where
  amount : ℕ
  duration : ℕ

/-- Calculates the capital-time product of an investment -/
def capitalTimeProduct (i : Investment) : ℕ :=
  i.amount * i.duration

/-- Represents the ratio of two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

theorem profit_ratio_theorem (a b : Investment) 
    (h1 : a.amount = 36000) (h2 : a.duration = 12)
    (h3 : b.amount = 54000) (h4 : b.duration = 4) :
    Ratio.mk (capitalTimeProduct a) (capitalTimeProduct b) = Ratio.mk 2 1 := by
  sorry

end NUMINAMATH_CALUDE_profit_ratio_theorem_l1228_122840


namespace NUMINAMATH_CALUDE_count_special_numbers_l1228_122850

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def leftmost_digit (n : ℕ) : ℕ := n / 1000

def second_digit (n : ℕ) : ℕ := (n / 100) % 10

def third_digit (n : ℕ) : ℕ := (n / 10) % 10

def last_digit (n : ℕ) : ℕ := n % 10

def all_digits_different (n : ℕ) : Prop :=
  leftmost_digit n ≠ second_digit n ∧
  leftmost_digit n ≠ third_digit n ∧
  leftmost_digit n ≠ last_digit n ∧
  second_digit n ≠ third_digit n ∧
  second_digit n ≠ last_digit n ∧
  third_digit n ≠ last_digit n

theorem count_special_numbers :
  ∃ (S : Finset ℕ),
    (∀ n ∈ S,
      is_four_digit n ∧
      leftmost_digit n % 2 = 1 ∧
      leftmost_digit n < 5 ∧
      second_digit n % 2 = 0 ∧
      second_digit n < 6 ∧
      all_digits_different n ∧
      n % 5 = 0) ∧
    S.card = 48 :=
by sorry

end NUMINAMATH_CALUDE_count_special_numbers_l1228_122850


namespace NUMINAMATH_CALUDE_sum_of_fractions_simplification_l1228_122843

theorem sum_of_fractions_simplification 
  (p q r : ℝ) 
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0) 
  (h_sum : p + q + r = 1) :
  1 / (q^2 + r^2 - p^2) + 1 / (p^2 + r^2 - q^2) + 1 / (p^2 + q^2 - r^2) = 3 / (1 - 2*q*r) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_simplification_l1228_122843


namespace NUMINAMATH_CALUDE_incorrect_expression_l1228_122815

theorem incorrect_expression (x : ℝ) : 5 * (x + 7) ≠ 5 * x + 7 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l1228_122815


namespace NUMINAMATH_CALUDE_power_equality_l1228_122899

theorem power_equality (p : ℕ) (h : (81 : ℕ)^6 = 3^p) : p = 24 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1228_122899


namespace NUMINAMATH_CALUDE_part_one_part_two_l1228_122873

-- Define the function f
def f (a : ℝ) (n : ℕ+) (x : ℝ) : ℝ := a * x^n.val * (1 - x)

-- Part 1
theorem part_one (a : ℝ) :
  (∀ x > 0, f a 2 x ≤ 4/27) ∧ (∃ x > 0, f a 2 x = 4/27) → a = 1 := by sorry

-- Part 2
theorem part_two (n : ℕ+) (m : ℝ) :
  (∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ f 1 n x = m ∧ f 1 n y = m) →
  0 < m ∧ m < (n.val ^ n.val : ℝ) / ((n.val + 1 : ℕ) ^ (n.val + 1)) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1228_122873


namespace NUMINAMATH_CALUDE_library_shelves_problem_l1228_122822

/-- Calculates the number of shelves needed to store books -/
def shelves_needed (large_books small_books shelf_capacity : ℕ) : ℕ :=
  let total_units := 2 * large_books + small_books
  (total_units + shelf_capacity - 1) / shelf_capacity

theorem library_shelves_problem :
  let initial_large_books := 18
  let initial_small_books := 18
  let removed_large_books := 4
  let removed_small_books := 2
  let shelf_capacity := 6
  let remaining_large_books := initial_large_books - removed_large_books
  let remaining_small_books := initial_small_books - removed_small_books
  shelves_needed remaining_large_books remaining_small_books shelf_capacity = 8 := by
  sorry

end NUMINAMATH_CALUDE_library_shelves_problem_l1228_122822


namespace NUMINAMATH_CALUDE_lakers_win_probability_l1228_122836

/-- The probability of a team winning a single game in the NBA finals -/
def win_prob : ℚ := 1/4

/-- The number of wins needed to win the NBA finals -/
def wins_needed : ℕ := 4

/-- The total number of games in a 7-game series -/
def total_games : ℕ := 7

/-- The probability of the Lakers winning the NBA finals in exactly 7 games -/
def lakers_win_in_seven : ℚ := 135/4096

theorem lakers_win_probability :
  lakers_win_in_seven = (Nat.choose 6 3 : ℚ) * win_prob^3 * (1 - win_prob)^3 * win_prob :=
by sorry

end NUMINAMATH_CALUDE_lakers_win_probability_l1228_122836


namespace NUMINAMATH_CALUDE_pedro_excess_squares_l1228_122897

-- Define the initial number of squares and multipliers for each player
def jesus_initial : ℕ := 60
def jesus_multiplier : ℕ := 2
def linden_initial : ℕ := 75
def linden_multiplier : ℕ := 3
def pedro_initial : ℕ := 200
def pedro_multiplier : ℕ := 4

-- Calculate the final number of squares for each player
def jesus_final : ℕ := jesus_initial * jesus_multiplier
def linden_final : ℕ := linden_initial * linden_multiplier
def pedro_final : ℕ := pedro_initial * pedro_multiplier

-- Define the theorem to be proved
theorem pedro_excess_squares : 
  pedro_final - (jesus_final + linden_final) = 455 := by
  sorry

end NUMINAMATH_CALUDE_pedro_excess_squares_l1228_122897


namespace NUMINAMATH_CALUDE_calculator_game_result_l1228_122847

def calculator_game (n : Nat) (a b c : Int) : Int :=
  let f1 := fun x => x^3
  let f2 := fun x => x^2
  let f3 := fun x => -x
  (f1^[n] a) + (f2^[n] b) + (f3^[n] c)

theorem calculator_game_result :
  calculator_game 45 1 0 (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculator_game_result_l1228_122847


namespace NUMINAMATH_CALUDE_email_subscription_day_l1228_122864

theorem email_subscription_day :
  ∀ (x : ℕ),
  (x ≤ 30) →
  (20 * x + 25 * (30 - x) = 675) →
  x = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_email_subscription_day_l1228_122864


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1228_122813

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2 - 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - x^2)}

-- State the theorem
theorem intersection_of_M_and_N :
  (M ∩ N : Set ℝ) = {x | -1 ≤ x ∧ x ≤ Real.sqrt 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1228_122813


namespace NUMINAMATH_CALUDE_det_A_plus_three_eq_two_l1228_122861

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 7; 3, 4]

theorem det_A_plus_three_eq_two :
  Matrix.det A + 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_det_A_plus_three_eq_two_l1228_122861


namespace NUMINAMATH_CALUDE_whole_number_between_l1228_122866

theorem whole_number_between : 
  ∀ N : ℤ, (9 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 10) → (N = 37 ∨ N = 38 ∨ N = 39) :=
by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_l1228_122866


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l1228_122806

/-- Distance between centers of two pulleys -/
theorem pulley_centers_distance 
  (r1 : ℝ) (r2 : ℝ) (contact_distance : ℝ)
  (h1 : r1 = 10)
  (h2 : r2 = 6)
  (h3 : contact_distance = 30) :
  ∃ (center_distance : ℝ), 
    center_distance = 2 * Real.sqrt 229 := by
  sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l1228_122806


namespace NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l1228_122800

theorem complex_power_one_minus_i_six :
  let i : ℂ := Complex.I
  (1 - i)^6 = 8*i := by sorry

end NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l1228_122800


namespace NUMINAMATH_CALUDE_stones_partition_exists_l1228_122825

/-- A partition of n into k parts is a list of k positive integers that sum to n. -/
def IsPartition (n k : ℕ) (partition : List ℕ) : Prop :=
  partition.length = k ∧ 
  partition.all (· > 0) ∧
  partition.sum = n

/-- A partition is similar if the maximum value is less than twice the minimum value. -/
def IsSimilarPartition (partition : List ℕ) : Prop :=
  partition.maximum? ≠ none ∧ 
  partition.minimum? ≠ none ∧ 
  (partition.maximum?.get! < 2 * partition.minimum?.get!)

theorem stones_partition_exists : 
  ∃ (partition : List ℕ), IsPartition 660 30 partition ∧ IsSimilarPartition partition := by
  sorry

end NUMINAMATH_CALUDE_stones_partition_exists_l1228_122825


namespace NUMINAMATH_CALUDE_base_7_units_digit_of_sum_l1228_122834

theorem base_7_units_digit_of_sum (a b : ℕ) (ha : a = 156) (hb : b = 97) :
  (a + b) % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_base_7_units_digit_of_sum_l1228_122834


namespace NUMINAMATH_CALUDE_distance_to_big_rock_is_4_l1228_122887

/-- Represents the distance to Big Rock in kilometers -/
def distance_to_big_rock : ℝ := sorry

/-- Rower's speed in still water in km/h -/
def rower_speed : ℝ := 6

/-- Current speed to Big Rock in km/h -/
def current_speed_to : ℝ := 2

/-- Current speed from Big Rock in km/h -/
def current_speed_from : ℝ := 3

/-- Rower's speed from Big Rock in km/h -/
def rower_speed_back : ℝ := 7

/-- Total round trip time in hours -/
def total_time : ℝ := 2

theorem distance_to_big_rock_is_4 :
  distance_to_big_rock = 4 ∧
  (distance_to_big_rock / (rower_speed - current_speed_to) +
   distance_to_big_rock / (rower_speed_back - current_speed_from) = total_time) :=
sorry

end NUMINAMATH_CALUDE_distance_to_big_rock_is_4_l1228_122887


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l1228_122883

theorem quadratic_equation_result : 
  ∀ y : ℝ, (6 * y^2 + 5 = 2 * y + 10) → (12 * y - 5)^2 = 133 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l1228_122883


namespace NUMINAMATH_CALUDE_mackenzie_bought_three_new_cds_l1228_122862

/-- Represents the price of a new CD -/
def new_cd_price : ℚ := 127.92 - 2 * 9.99

/-- Represents the number of new CDs Mackenzie bought -/
def mackenzie_new_cds : ℚ := (133.89 - 8 * 9.99) / (127.92 - 2 * 9.99) * 6

theorem mackenzie_bought_three_new_cds : 
  ⌊mackenzie_new_cds⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_mackenzie_bought_three_new_cds_l1228_122862


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_l1228_122886

theorem consecutive_odd_integers (x y z : ℤ) : 
  (∃ k : ℤ, x = 2*k + 1) →  -- x is odd
  y = x + 2 →               -- y is the next consecutive odd integer
  z = y + 2 →               -- z is the next consecutive odd integer after y
  y + z = x + 17 →          -- sum of last two is 17 more than the first
  x = 11 := by              -- the first integer is 11
sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_l1228_122886


namespace NUMINAMATH_CALUDE_p_suff_not_nec_q_l1228_122804

-- Define propositions p, q, and r
variable (p q r : Prop)

-- Define the conditions
axiom p_suff_not_nec_r : (p → r) ∧ ¬(r → p)
axiom q_nec_r : r → q

-- Theorem to prove
theorem p_suff_not_nec_q : (p → q) ∧ ¬(q → p) := by
  sorry

end NUMINAMATH_CALUDE_p_suff_not_nec_q_l1228_122804


namespace NUMINAMATH_CALUDE_three_subject_average_l1228_122872

theorem three_subject_average (korean_math_avg : ℝ) (english_score : ℝ) :
  korean_math_avg = 86 →
  english_score = 98 →
  (2 * korean_math_avg + english_score) / 3 = 90 := by
sorry

end NUMINAMATH_CALUDE_three_subject_average_l1228_122872


namespace NUMINAMATH_CALUDE_A_intersect_B_l1228_122877

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2 * x - 1}

theorem A_intersect_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1228_122877


namespace NUMINAMATH_CALUDE_no_solution_for_four_divides_sum_of_squares_plus_one_l1228_122831

theorem no_solution_for_four_divides_sum_of_squares_plus_one :
  ∀ (a b : ℤ), ¬(4 ∣ a^2 + b^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_four_divides_sum_of_squares_plus_one_l1228_122831


namespace NUMINAMATH_CALUDE_magnitude_of_z_l1228_122828

open Complex

theorem magnitude_of_z (z : ℂ) (h : i * (1 - z) = 1) : abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1228_122828


namespace NUMINAMATH_CALUDE_sophie_owes_jordan_l1228_122880

theorem sophie_owes_jordan (price_per_window : ℚ) (windows_cleaned : ℚ) :
  price_per_window = 13/3 →
  windows_cleaned = 8/5 →
  price_per_window * windows_cleaned = 104/15 := by
  sorry

end NUMINAMATH_CALUDE_sophie_owes_jordan_l1228_122880


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1228_122884

theorem square_plus_reciprocal_square (m : ℝ) (h : m + 1/m = 6) :
  m^2 + 1/m^2 + 4 = 38 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1228_122884


namespace NUMINAMATH_CALUDE_quasi_pythagorean_prime_divisor_l1228_122848

theorem quasi_pythagorean_prime_divisor 
  (a b c : ℕ+) 
  (h : c.val ^ 2 = a.val ^ 2 + a.val * b.val + b.val ^ 2) : 
  ∃ (p : ℕ), p > 5 ∧ Nat.Prime p ∧ p ∣ c.val :=
sorry

end NUMINAMATH_CALUDE_quasi_pythagorean_prime_divisor_l1228_122848


namespace NUMINAMATH_CALUDE_set_A_is_empty_l1228_122849

theorem set_A_is_empty (a : ℝ) : {x : ℝ | |x - 1| ≤ 2*a - a^2 - 2} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_set_A_is_empty_l1228_122849


namespace NUMINAMATH_CALUDE_youngest_not_first_or_last_l1228_122801

/-- The number of ways to arrange n people in a line -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line with one specific person at the start or end -/
def restrictedArrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- The number of people in the line -/
def n : ℕ := 5

theorem youngest_not_first_or_last :
  totalArrangements n - restrictedArrangements n = 72 := by
  sorry

#eval totalArrangements n - restrictedArrangements n

end NUMINAMATH_CALUDE_youngest_not_first_or_last_l1228_122801


namespace NUMINAMATH_CALUDE_birdseed_mix_l1228_122827

/-- Given two brands of birdseed and their composition, prove the percentage of sunflower in Brand A -/
theorem birdseed_mix (x : ℝ) : 
  (0.4 + x / 100 = 1) →  -- Brand A composition
  (0.65 + 0.35 = 1) →  -- Brand B composition
  (0.6 * x / 100 + 0.4 * 0.35 = 0.5) →  -- Mix composition
  x = 60 := by sorry

end NUMINAMATH_CALUDE_birdseed_mix_l1228_122827


namespace NUMINAMATH_CALUDE_katie_cole_miles_ratio_l1228_122818

theorem katie_cole_miles_ratio :
  ∀ (miles_xavier miles_katie miles_cole : ℕ),
    miles_xavier = 3 * miles_katie →
    miles_xavier = 84 →
    miles_cole = 7 →
    miles_katie / miles_cole = 4 := by
  sorry

end NUMINAMATH_CALUDE_katie_cole_miles_ratio_l1228_122818


namespace NUMINAMATH_CALUDE_sum_of_integers_l1228_122821

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x^2 + y^2 = 100)
  (h2 : x * y = 32) : 
  x + y = 2 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1228_122821


namespace NUMINAMATH_CALUDE_distance_for_boy_problem_l1228_122812

/-- Calculates the distance traveled given time in minutes and speed in meters per second -/
def distance_traveled (time_minutes : ℕ) (speed_meters_per_second : ℕ) : ℕ :=
  time_minutes * 60 * speed_meters_per_second

/-- Theorem: Given 36 minutes and a speed of 4 meters per second, the distance traveled is 8640 meters -/
theorem distance_for_boy_problem : distance_traveled 36 4 = 8640 := by
  sorry

end NUMINAMATH_CALUDE_distance_for_boy_problem_l1228_122812


namespace NUMINAMATH_CALUDE_complex_product_real_l1228_122865

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 3 - 2 * I
  let z₂ : ℂ := 1 + a * I
  (z₁ * z₂).im = 0 → a = 2/3 := by sorry

end NUMINAMATH_CALUDE_complex_product_real_l1228_122865


namespace NUMINAMATH_CALUDE_seventh_degree_equation_reduction_l1228_122852

theorem seventh_degree_equation_reduction (a b : ℝ) :
  ∃ (f : ℝ → ℝ), 
    (∀ x, f x = x^7 - 7*a*x^5 + 14*a^2*x^3 - 7*a^3*x - b) →
    (∃ α β : ℝ, α * β = a ∧ α^7 + β^7 = b ∧ f α = 0 ∧ f β = 0) :=
by sorry

end NUMINAMATH_CALUDE_seventh_degree_equation_reduction_l1228_122852


namespace NUMINAMATH_CALUDE_f_one_root_m_range_l1228_122817

/-- A cubic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

/-- The property of having exactly one real root -/
def has_exactly_one_real_root (g : ℝ → ℝ) : Prop :=
  ∃! x, g x = 0

/-- The theorem stating the range of m for which f has exactly one real root -/
theorem f_one_root_m_range (m : ℝ) :
  has_exactly_one_real_root (f m) ↔ m < -2 ∨ m > 2 := by sorry

end NUMINAMATH_CALUDE_f_one_root_m_range_l1228_122817


namespace NUMINAMATH_CALUDE_oil_depth_relationship_l1228_122863

/-- Represents a right cylindrical tank -/
structure CylindricalTank where
  height : ℝ
  baseDiameter : ℝ

/-- Represents the oil level in the tank -/
structure OilLevel where
  depthWhenFlat : ℝ
  depthWhenUpright : ℝ

/-- The theorem stating the relationship between oil depths -/
theorem oil_depth_relationship (tank : CylindricalTank) (oil : OilLevel) :
  tank.height = 15 ∧ 
  tank.baseDiameter = 6 ∧ 
  oil.depthWhenFlat = 4 →
  oil.depthWhenUpright = 15 := by
  sorry


end NUMINAMATH_CALUDE_oil_depth_relationship_l1228_122863


namespace NUMINAMATH_CALUDE_intersection_of_S_and_T_l1228_122896

open Set

def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | -4 ≤ x ∧ x ≤ 1}

theorem intersection_of_S_and_T : S ∩ T = Ioc (-2) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_S_and_T_l1228_122896


namespace NUMINAMATH_CALUDE_sum_has_even_digit_l1228_122885

def reverse_number (n : List Nat) : List Nat :=
  n.reverse

def sum_digits (n m : List Nat) : List Nat :=
  sorry

theorem sum_has_even_digit (n : List Nat) (h : n.length = 17) :
  ∃ (d : Nat), d ∈ sum_digits n (reverse_number n) ∧ Even d :=
sorry

end NUMINAMATH_CALUDE_sum_has_even_digit_l1228_122885


namespace NUMINAMATH_CALUDE_smallest_common_multiple_l1228_122805

theorem smallest_common_multiple (h : ℕ) (d : ℕ) : 
  (∀ k : ℕ, k > 0 ∧ 10 * k % 15 = 0 → k ≥ 3) ∧ 
  (10 * 3 % 15 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_l1228_122805


namespace NUMINAMATH_CALUDE_quadratic_root_conjugate_l1228_122846

theorem quadratic_root_conjugate (a b c : ℚ) :
  (a ≠ 0) →
  (a * (3 + Real.sqrt 2)^2 + b * (3 + Real.sqrt 2) + c = 0) →
  (a * (3 - Real.sqrt 2)^2 + b * (3 - Real.sqrt 2) + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_conjugate_l1228_122846


namespace NUMINAMATH_CALUDE_intersection_point_is_minus_one_minus_one_l1228_122890

-- Define the two line equations
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y + 7 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Theorem stating that (-1, -1) is the unique intersection point
theorem intersection_point_is_minus_one_minus_one :
  ∃! (x y : ℝ), line1 x y ∧ line2 x y ∧ x = -1 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_point_is_minus_one_minus_one_l1228_122890


namespace NUMINAMATH_CALUDE_f_is_odd_sum_greater_than_two_l1228_122842

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x + x / (x^2 + 1)

-- Theorem 1: f is an odd function
theorem f_is_odd : ∀ x : ℝ, x ≠ 0 → f (-x) = -f x := by sorry

-- Theorem 2: If x₁ > 0, x₂ > 0, x₁ ≠ x₂, and f(x₁) = f(x₂), then x₁ + x₂ > 2
theorem sum_greater_than_two (x₁ x₂ : ℝ) (h1 : x₁ > 0) (h2 : x₂ > 0) (h3 : x₁ ≠ x₂) (h4 : f x₁ = f x₂) : x₁ + x₂ > 2 := by sorry

end NUMINAMATH_CALUDE_f_is_odd_sum_greater_than_two_l1228_122842


namespace NUMINAMATH_CALUDE_special_linear_function_at_two_l1228_122824

/-- A linear function satisfying specific conditions -/
structure SpecialLinearFunction where
  f : ℝ → ℝ
  linear : ∀ x y c : ℝ, f (x + y) = f x + f y ∧ f (c * x) = c * f x
  inverse_relation : ∀ x : ℝ, f x = 3 * f⁻¹ x + 5
  f_one : f 1 = 5

/-- The main theorem stating the value of f(2) for the special linear function -/
theorem special_linear_function_at_two (slf : SpecialLinearFunction) :
  slf.f 2 = 2 * Real.sqrt 3 + (5 * Real.sqrt 3) / (Real.sqrt 3 + 3) := by
  sorry

end NUMINAMATH_CALUDE_special_linear_function_at_two_l1228_122824


namespace NUMINAMATH_CALUDE_units_digit_of_3_power_2004_l1228_122894

theorem units_digit_of_3_power_2004 : 3^2004 % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_3_power_2004_l1228_122894


namespace NUMINAMATH_CALUDE_pet_shop_limbs_l1228_122808

/-- The total number of legs and arms in the pet shop -/
def total_limbs : ℕ :=
  4 * 2 +  -- birds
  6 * 4 +  -- dogs
  5 * 0 +  -- snakes
  2 * 8 +  -- spiders
  3 * 4 +  -- horses
  7 * 4 +  -- rabbits
  2 * 8 +  -- octopuses
  8 * 6 +  -- ants
  1 * 12   -- unique creature

/-- Theorem stating that the total number of legs and arms in the pet shop is 164 -/
theorem pet_shop_limbs : total_limbs = 164 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_limbs_l1228_122808


namespace NUMINAMATH_CALUDE_cube_equation_solution_l1228_122857

theorem cube_equation_solution : ∃ x : ℝ, (x - 1)^3 = 64 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l1228_122857


namespace NUMINAMATH_CALUDE_kitten_growth_l1228_122803

/-- The length of a kitten after doubling twice -/
def kitten_length (initial_length : ℝ) : ℝ :=
  initial_length * 2 * 2

/-- Theorem: A kitten with initial length 4 inches will be 16 inches long after doubling twice -/
theorem kitten_growth : kitten_length 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_kitten_growth_l1228_122803


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l1228_122837

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where C = π/6 and 2acosB = c, prove that A = 5π/12. -/
theorem triangle_angle_proof (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  A + B + C = π →  -- Sum of angles in a triangle
  C = π / 6 →  -- Given condition
  2 * a * Real.cos B = c →  -- Given condition
  A = 5 * π / 12 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l1228_122837


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l1228_122838

theorem isosceles_triangle_angle_measure :
  ∀ (D E F : ℝ),
  -- Triangle DEF is isosceles with angle D congruent to angle F
  D = F →
  -- The measure of angle F is three times the measure of angle E
  F = 3 * E →
  -- The sum of angles in a triangle is 180 degrees
  D + E + F = 180 →
  -- The measure of angle D is 540/7 degrees
  D = 540 / 7 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l1228_122838


namespace NUMINAMATH_CALUDE_mutual_correlation_sign_change_l1228_122832

/-- A stationary stochastic process -/
class StationaryStochasticProcess (X : ℝ → ℝ) : Prop where
  -- Add any necessary properties for a stationary stochastic process

/-- The derivative of a function -/
def derivative (f : ℝ → ℝ) : ℝ → ℝ :=
  fun t => sorry -- Definition of derivative

/-- Mutual correlation function of a process and its derivative -/
def mutualCorrelationFunction (X : ℝ → ℝ) (t₁ t₂ : ℝ) : ℝ :=
  sorry -- Definition of mutual correlation function

/-- Theorem: The mutual correlation function changes sign when arguments are swapped -/
theorem mutual_correlation_sign_change
  (X : ℝ → ℝ) [StationaryStochasticProcess X] (t₁ t₂ : ℝ) :
  mutualCorrelationFunction X t₁ t₂ = -mutualCorrelationFunction X t₂ t₁ :=
by sorry

end NUMINAMATH_CALUDE_mutual_correlation_sign_change_l1228_122832


namespace NUMINAMATH_CALUDE_book_writing_time_l1228_122867

/-- Calculates the number of weeks required to write a book -/
def weeks_to_write_book (pages_per_hour : ℕ) (hours_per_day : ℕ) (total_pages : ℕ) : ℕ :=
  (total_pages / (pages_per_hour * hours_per_day) + 6) / 7

/-- Theorem: It takes 7 weeks to write a 735-page book at 5 pages per hour, 3 hours per day -/
theorem book_writing_time :
  weeks_to_write_book 5 3 735 = 7 := by
  sorry

#eval weeks_to_write_book 5 3 735

end NUMINAMATH_CALUDE_book_writing_time_l1228_122867


namespace NUMINAMATH_CALUDE_apartments_with_one_resident_l1228_122895

theorem apartments_with_one_resident (total : ℕ) (at_least_one_percent : ℚ) (at_least_two_percent : ℚ) :
  total = 120 →
  at_least_one_percent = 85 / 100 →
  at_least_two_percent = 60 / 100 →
  (total * at_least_one_percent - total * at_least_two_percent : ℚ) = 30 := by
sorry

end NUMINAMATH_CALUDE_apartments_with_one_resident_l1228_122895


namespace NUMINAMATH_CALUDE_shampoo_bottles_l1228_122819

theorem shampoo_bottles (medium_capacity : ℕ) (jumbo_capacity : ℕ) (unusable_space : ℕ) :
  medium_capacity = 45 →
  jumbo_capacity = 720 →
  unusable_space = 20 →
  (Nat.ceil ((jumbo_capacity - unusable_space : ℚ) / medium_capacity) : ℕ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_shampoo_bottles_l1228_122819


namespace NUMINAMATH_CALUDE_q_factor_change_l1228_122814

/-- Given a function q defined in terms of w, h, and z, prove that when w is quadrupled,
    h is doubled, and z is tripled, q is multiplied by 5/18. -/
theorem q_factor_change (w h z : ℝ) (q : ℝ → ℝ → ℝ → ℝ) 
    (hq : q w h z = 5 * w / (4 * h * z^2)) :
  q (4*w) (2*h) (3*z) = (5/18) * q w h z := by
  sorry

end NUMINAMATH_CALUDE_q_factor_change_l1228_122814


namespace NUMINAMATH_CALUDE_carries_hourly_rate_l1228_122856

/-- Represents Carrie's cake-making scenario -/
structure CakeScenario where
  hoursPerDay : ℕ
  daysWorked : ℕ
  suppliesCost : ℕ
  profit : ℕ

/-- Calculates Carrie's hourly rate given the scenario -/
def hourlyRate (scenario : CakeScenario) : ℚ :=
  (scenario.profit + scenario.suppliesCost) / (scenario.hoursPerDay * scenario.daysWorked)

/-- Theorem stating that Carrie's hourly rate was $22 -/
theorem carries_hourly_rate :
  let scenario : CakeScenario := {
    hoursPerDay := 2,
    daysWorked := 4,
    suppliesCost := 54,
    profit := 122
  }
  hourlyRate scenario = 22 := by sorry

end NUMINAMATH_CALUDE_carries_hourly_rate_l1228_122856


namespace NUMINAMATH_CALUDE_linear_function_k_value_l1228_122835

/-- Given a linear function y = kx + 1 passing through the point (-1, 0), prove that k = 1 -/
theorem linear_function_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1) → -- The function is linear with equation y = kx + 1
  (0 = k * (-1) + 1) →         -- The graph passes through the point (-1, 0)
  k = 1                        -- Conclusion: k equals 1
:= by sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l1228_122835


namespace NUMINAMATH_CALUDE_nutmeg_amount_l1228_122874

theorem nutmeg_amount (cinnamon : Float) (difference : Float) (nutmeg : Float) : 
  cinnamon = 0.67 → 
  difference = 0.17 →
  cinnamon = nutmeg + difference →
  nutmeg = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_nutmeg_amount_l1228_122874


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1228_122898

theorem absolute_value_equation_solution :
  ∃! x : ℚ, |5 * x - 7| + 2 = 2 ∧ x = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1228_122898


namespace NUMINAMATH_CALUDE_square_measurement_error_l1228_122816

/-- Given a square with side length L, if the length is measured as 1.03L and
    the width is measured as 0.98L, then the percentage error in the calculated
    area is 0.94%. -/
theorem square_measurement_error (L : ℝ) (L_pos : L > 0) :
  let measured_length : ℝ := 1.03 * L
  let measured_width : ℝ := 0.98 * L
  let actual_area : ℝ := L^2
  let measured_area : ℝ := measured_length * measured_width
  let percentage_error : ℝ := (measured_area - actual_area) / actual_area * 100
  percentage_error = 0.94 := by
sorry

end NUMINAMATH_CALUDE_square_measurement_error_l1228_122816


namespace NUMINAMATH_CALUDE_linear_relationship_scaling_l1228_122878

/-- Given a linear relationship between x and y, prove that if an increase of 4 in x
    corresponds to an increase of 10 in y, then an increase of 12 in x
    will result in an increase of 30 in y. -/
theorem linear_relationship_scaling (f : ℝ → ℝ) (h : ∀ x, f (x + 4) - f x = 10) :
  ∀ x, f (x + 12) - f x = 30 := by
sorry

end NUMINAMATH_CALUDE_linear_relationship_scaling_l1228_122878


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1228_122845

theorem polynomial_divisibility (n : ℤ) : 
  ∃ k : ℤ, (n + 7)^2 - n^2 = 7 * k := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1228_122845


namespace NUMINAMATH_CALUDE_part_one_part_two_a_part_two_b_part_two_c_l1228_122833

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 + (3 - 2*a) * x - 6

-- Part 1
theorem part_one (x : ℝ) : f 1 x > 0 ↔ x < -3 ∨ x > 2 := by sorry

-- Part 2
theorem part_two_a (a x : ℝ) (h : a < -3/2) : f a x < 0 ↔ x < -3/a ∨ x > 2 := by sorry

theorem part_two_b (x : ℝ) : f (-3/2) x < 0 ↔ x ≠ 2 := by sorry

theorem part_two_c (a x : ℝ) (h : -3/2 < a ∧ a < 0) : f a x < 0 ↔ x < 2 ∨ x > -3/a := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_a_part_two_b_part_two_c_l1228_122833


namespace NUMINAMATH_CALUDE_tree_planting_problem_l1228_122855

-- Define the types for our numbers
def ThreeDigitNumber := { n : ℕ // n ≥ 100 ∧ n < 1000 }
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n < 100 }

-- Function to reverse digits of a number
def reverseDigits (n : ℕ) : ℕ :=
  let rec aux (n acc : ℕ) : ℕ :=
    if n = 0 then acc
    else aux (n / 10) (acc * 10 + n % 10)
  aux n 0

-- Define our theorem
theorem tree_planting_problem 
  (poplars : ThreeDigitNumber) 
  (lindens : TwoDigitNumber) 
  (h1 : poplars.val + lindens.val = 144)
  (h2 : reverseDigits poplars.val + reverseDigits lindens.val = 603) :
  poplars.val = 105 ∧ lindens.val = 39 := by
  sorry


end NUMINAMATH_CALUDE_tree_planting_problem_l1228_122855


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l1228_122809

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else Real.log x / Real.log a

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) →
  (a ≥ 1/7 ∧ a < 1/3) :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l1228_122809


namespace NUMINAMATH_CALUDE_theorem_1_theorem_2_l1228_122893

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem 1
theorem theorem_1 (a : ℝ) : p a → a ≤ 1 := by sorry

-- Theorem 2
theorem theorem_2 (a : ℝ) : ¬(p a ∧ q a) → a ∈ Set.Ioo (-2) 1 ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_theorem_1_theorem_2_l1228_122893


namespace NUMINAMATH_CALUDE_f_monotonicity_and_range_l1228_122889

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * a - x^2) / Real.exp x

theorem f_monotonicity_and_range (a : ℝ) :
  (a ≤ -1/2 → ∀ x y : ℝ, x < y → f a x < f a y) ∧
  (a > -1/2 → ∀ x y : ℝ,
    ((x < y ∧ y < 1 - Real.sqrt (2 * a + 1)) ∨
     (x > 1 + Real.sqrt (2 * a + 1) ∧ y > x)) →
    f a x < f a y) ∧
  (a > -1/2 → ∀ x y : ℝ,
    (x > 1 - Real.sqrt (2 * a + 1) ∧ y < 1 + Real.sqrt (2 * a + 1) ∧ x < y) →
    f a x > f a y) ∧
  ((∀ x : ℝ, x ≥ 1 → f a x > -1) → a > (1 - Real.exp 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_range_l1228_122889


namespace NUMINAMATH_CALUDE_homework_difference_l1228_122888

def math_pages : ℕ := 5
def reading_pages : ℕ := 2

theorem homework_difference : math_pages - reading_pages = 3 := by
  sorry

end NUMINAMATH_CALUDE_homework_difference_l1228_122888


namespace NUMINAMATH_CALUDE_inequality_proof_l1228_122860

theorem inequality_proof (a b c : ℝ) (h : a + b + c = 3) :
  (1 / (5 * a^2 - 4 * a + 1)) + (1 / (5 * b^2 - 4 * b + 1)) + (1 / (5 * c^2 - 4 * c + 1)) ≤ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1228_122860


namespace NUMINAMATH_CALUDE_max_discount_rate_l1228_122839

/-- The maximum discount rate that can be offered on an item while maintaining a minimum profit margin -/
theorem max_discount_rate (cost_price selling_price min_profit_margin : ℝ) 
  (h1 : cost_price = 4)
  (h2 : selling_price = 5)
  (h3 : min_profit_margin = 0.1)
  (h4 : cost_price > 0)
  (h5 : selling_price > cost_price) :
  ∃ (max_discount : ℝ), 
    max_discount = 12 ∧ 
    ∀ (discount : ℝ), 
      0 ≤ discount → discount ≤ max_discount → 
      (selling_price * (1 - discount / 100) - cost_price) / cost_price ≥ min_profit_margin :=
by sorry

end NUMINAMATH_CALUDE_max_discount_rate_l1228_122839


namespace NUMINAMATH_CALUDE_tony_remaining_money_l1228_122892

/-- Calculates the remaining money after expenses -/
def remaining_money (initial : ℕ) (ticket : ℕ) (hot_dog : ℕ) (soda : ℕ) : ℕ :=
  initial - ticket - hot_dog - soda

/-- Proves that Tony has $26 left after his expenses -/
theorem tony_remaining_money :
  remaining_money 50 15 5 4 = 26 := by
  sorry

#eval remaining_money 50 15 5 4

end NUMINAMATH_CALUDE_tony_remaining_money_l1228_122892


namespace NUMINAMATH_CALUDE_parallelogram_intersection_theorem_l1228_122829

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (J K L M : Point)

/-- Checks if a point is on the extension of a line segment -/
def isOnExtension (A B P : Point) : Prop := sorry

/-- Checks if two line segments intersect at a point -/
def intersectsAt (A B C D Q : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (A B : Point) : ℝ := sorry

theorem parallelogram_intersection_theorem (JKLM : Parallelogram) (P Q R : Point) :
  isOnExtension JKLM.L JKLM.M P →
  intersectsAt JKLM.K P JKLM.L JKLM.J Q →
  intersectsAt JKLM.K P JKLM.J JKLM.M R →
  distance Q R = 40 →
  distance R P = 30 →
  distance JKLM.K Q = 20 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_intersection_theorem_l1228_122829


namespace NUMINAMATH_CALUDE_cubic_polynomial_value_l1228_122870

/-- The given polynomial h -/
def h (x : ℝ) : ℝ := x^3 - x^2 + x - 1

/-- The roots of h -/
def roots_h : Set ℝ := {x | h x = 0}

/-- The theorem statement -/
theorem cubic_polynomial_value (p : ℝ → ℝ) :
  (∃ a b c : ℝ, roots_h = {a, b, c}) →  -- h has three distinct roots
  (∀ x, x ∈ roots_h → x^3 ∈ {y | p y = 0}) →  -- roots of p are cubes of roots of h
  (∀ x, p (p x) = p (p (p x))) →  -- p is a cubic polynomial
  p 1 = 2 →  -- given condition
  p 8 = 1008 := by  -- conclusion to prove
sorry


end NUMINAMATH_CALUDE_cubic_polynomial_value_l1228_122870


namespace NUMINAMATH_CALUDE_l_shape_area_l1228_122810

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The large rectangle -/
def large_rectangle : Rectangle := { length := 10, width := 6 }

/-- The small rectangle to be subtracted -/
def small_rectangle : Rectangle := { length := 4, width := 3 }

/-- The number of small rectangles to be subtracted -/
def num_small_rectangles : ℕ := 2

/-- Theorem: The area of the L-shape is 36 square units -/
theorem l_shape_area : 
  area large_rectangle - num_small_rectangles * area small_rectangle = 36 := by
  sorry

end NUMINAMATH_CALUDE_l_shape_area_l1228_122810


namespace NUMINAMATH_CALUDE_equation_solution_l1228_122802

theorem equation_solution :
  let f (n : ℝ) := (3 - 2*n) / (n + 2) + (3*n - 9) / (3 - 2*n)
  let n₁ := (25 + Real.sqrt 13) / 18
  let n₂ := (25 - Real.sqrt 13) / 18
  f n₁ = 2 ∧ f n₂ = 2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1228_122802


namespace NUMINAMATH_CALUDE_fourth_grid_shaded_fraction_initial_shaded_squares_shaded_squares_arithmetic_l1228_122807

/-- Represents the number of shaded squares in the nth grid -/
def shaded_squares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the total number of squares in the nth grid -/
def total_squares (n : ℕ) : ℕ := n ^ 2

/-- The main theorem stating the fraction of shaded squares in the fourth grid -/
theorem fourth_grid_shaded_fraction :
  (shaded_squares 4 : ℚ) / (total_squares 4 : ℚ) = 7 / 16 := by
  sorry

/-- Verifies that the first three grids have 1, 3, and 5 shaded squares respectively -/
theorem initial_shaded_squares :
  shaded_squares 1 = 1 ∧ shaded_squares 2 = 3 ∧ shaded_squares 3 = 5 := by
  sorry

/-- Verifies that the sequence of shaded squares is arithmetic -/
theorem shaded_squares_arithmetic :
  ∀ n : ℕ, shaded_squares (n + 1) - shaded_squares n = 
           shaded_squares (n + 2) - shaded_squares (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_fourth_grid_shaded_fraction_initial_shaded_squares_shaded_squares_arithmetic_l1228_122807


namespace NUMINAMATH_CALUDE_function_properties_l1228_122879

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / 2) * Real.sin (2 * x) - Real.cos (2 * x)

theorem function_properties (a : ℝ) :
  f a (π / 8) = 0 →
  a = Real.sqrt 2 ∧
  (∀ x : ℝ, f a (x + π) = f a x) ∧
  (∀ x : ℝ, f a x ≤ Real.sqrt 2) ∧
  (∃ x : ℝ, f a x = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1228_122879


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l1228_122891

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O2 (x y r : ℝ) : Prop := (x - 3)^2 + y^2 = r^2

-- Define the tangency condition
def are_tangent (r : ℝ) : Prop := ∃ (x y : ℝ), circle_O1 x y ∧ circle_O2 x y r

-- Theorem statement
theorem tangent_circles_radius (r : ℝ) (h1 : r > 0) (h2 : are_tangent r) : r = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_radius_l1228_122891


namespace NUMINAMATH_CALUDE_gcf_of_75_and_125_l1228_122851

theorem gcf_of_75_and_125 : Nat.gcd 75 125 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_125_l1228_122851


namespace NUMINAMATH_CALUDE_toys_sold_proof_l1228_122876

/-- The number of toys sold by a man -/
def number_of_toys : ℕ := 18

/-- The selling price of the toys -/
def selling_price : ℕ := 23100

/-- The cost price of one toy -/
def cost_price : ℕ := 1100

/-- The gain from the sale -/
def gain : ℕ := 3 * cost_price

theorem toys_sold_proof :
  number_of_toys * cost_price + gain = selling_price :=
by sorry

end NUMINAMATH_CALUDE_toys_sold_proof_l1228_122876


namespace NUMINAMATH_CALUDE_rain_probability_implies_very_likely_l1228_122853

-- Define what "very likely" means in terms of probability
def very_likely (p : ℝ) : Prop := p ≥ 0.7

-- Theorem statement
theorem rain_probability_implies_very_likely (p : ℝ) (h : p = 0.8) : very_likely p := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_implies_very_likely_l1228_122853


namespace NUMINAMATH_CALUDE_infinite_sum_equals_ln2_squared_l1228_122841

/-- The infinite sum of the given series is equal to ln(2)² -/
theorem infinite_sum_equals_ln2_squared :
  ∑' k : ℕ, (3 * Real.log (4 * k + 2) / (4 * k + 2) -
             Real.log (4 * k + 3) / (4 * k + 3) -
             Real.log (4 * k + 4) / (4 * k + 4) -
             Real.log (4 * k + 5) / (4 * k + 5)) = (Real.log 2) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_ln2_squared_l1228_122841


namespace NUMINAMATH_CALUDE_no_valid_base_l1228_122811

theorem no_valid_base : ¬ ∃ (b : ℕ), 0 < b ∧ b^6 ≤ 196 ∧ 196 < b^7 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_base_l1228_122811


namespace NUMINAMATH_CALUDE_polynomial_value_at_n_plus_one_l1228_122826

theorem polynomial_value_at_n_plus_one (n : ℕ) (P : Polynomial ℝ) 
  (h_degree : P.degree ≤ n) 
  (h_values : ∀ k : ℕ, k ≤ n → P.eval (k : ℝ) = k / (k + 1)) :
  P.eval ((n + 1 : ℕ) : ℝ) = (n + 1 + (-1)^(n + 1)) / (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_n_plus_one_l1228_122826


namespace NUMINAMATH_CALUDE_proposition_1_proposition_2_l1228_122859

-- Proposition 1
theorem proposition_1 (x y : ℝ) :
  (xy = 0 → x = 0 ∨ y = 0) ↔
  ((x = 0 ∨ y = 0) → xy = 0) ∧
  (xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧
  ((x ≠ 0 ∧ y ≠ 0) → xy ≠ 0) :=
sorry

-- Proposition 2
theorem proposition_2 (x y : ℝ) :
  ((x > 0 ∧ y > 0) → xy > 0) ↔
  (xy > 0 → x > 0 ∧ y > 0) ∧
  ((x ≤ 0 ∨ y ≤ 0) → xy ≤ 0) ∧
  (xy ≤ 0 → x ≤ 0 ∨ y ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_2_l1228_122859


namespace NUMINAMATH_CALUDE_victoria_money_l1228_122830

/-- The amount of money Victoria was given by her mother -/
def total_money : ℕ := sorry

/-- The cost of one box of pizza -/
def pizza_cost : ℕ := 12

/-- The number of pizza boxes bought -/
def pizza_boxes : ℕ := 2

/-- The cost of one pack of juice drinks -/
def juice_cost : ℕ := 2

/-- The number of juice drink packs bought -/
def juice_packs : ℕ := 2

/-- The amount Victoria should return to her mother -/
def return_amount : ℕ := 22

/-- Theorem stating that the total money Victoria was given equals $50 -/
theorem victoria_money : 
  total_money = pizza_cost * pizza_boxes + juice_cost * juice_packs + return_amount :=
by sorry

end NUMINAMATH_CALUDE_victoria_money_l1228_122830


namespace NUMINAMATH_CALUDE_tangent_line_properties_l1228_122875

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2

theorem tangent_line_properties :
  (∃ x : ℝ, (deriv f) x = 3) ∧
  (∃! t : ℝ, (f t - 2) / t = (deriv f) t) ∧
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧
    (f t₁ - 4) / (t₁ - 1) = (deriv f) t₁ ∧
    (f t₂ - 4) / (t₂ - 1) = (deriv f) t₂ ∧
    ∀ t : ℝ, t ≠ t₁ → t ≠ t₂ →
      (f t - 4) / (t - 1) ≠ (deriv f) t) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_properties_l1228_122875


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1228_122869

theorem rationalize_denominator : (5 : ℝ) / Real.sqrt 125 = Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1228_122869


namespace NUMINAMATH_CALUDE_theta_range_theorem_l1228_122823

-- Define the set of valid θ values
def ValidTheta : Set ℝ := { θ | -Real.pi ≤ θ ∧ θ ≤ Real.pi }

-- Define the inequality condition
def InequalityCondition (θ : ℝ) : Prop :=
  Real.cos (θ + Real.pi / 4) < 3 * (Real.sin θ ^ 5 - Real.cos θ ^ 5)

-- Define the solution set
def SolutionSet : Set ℝ := 
  { θ | (-Real.pi ≤ θ ∧ θ < -3 * Real.pi / 4) ∨ (Real.pi / 4 < θ ∧ θ ≤ Real.pi) }

-- Theorem statement
theorem theta_range_theorem :
  ∀ θ ∈ ValidTheta, InequalityCondition θ ↔ θ ∈ SolutionSet :=
sorry

end NUMINAMATH_CALUDE_theta_range_theorem_l1228_122823


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1228_122882

/-- A point P(x, y) lies on the x-axis if and only if its y-coordinate is 0 -/
def lies_on_x_axis (x y : ℝ) : Prop := y = 0

/-- The theorem states that if the point P(a-4, a+3) lies on the x-axis, then a = -3 -/
theorem point_on_x_axis (a : ℝ) :
  lies_on_x_axis (a - 4) (a + 3) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1228_122882


namespace NUMINAMATH_CALUDE_point_on_hyperbola_l1228_122858

/-- The x-coordinate of point A on the hyperbola y = -4/x with y-coordinate 4 -/
def a : ℝ := sorry

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y = -4 / x

theorem point_on_hyperbola : 
  hyperbola a 4 → a = -1 := by sorry

end NUMINAMATH_CALUDE_point_on_hyperbola_l1228_122858


namespace NUMINAMATH_CALUDE_pencil_price_solution_l1228_122868

def pencil_price_problem (pencil_price notebook_price : ℕ) : Prop :=
  (pencil_price + notebook_price = 950) ∧ 
  (notebook_price = pencil_price + 150)

theorem pencil_price_solution : 
  ∃ (pencil_price notebook_price : ℕ), 
    pencil_price_problem pencil_price notebook_price ∧ 
    pencil_price = 400 := by
  sorry

end NUMINAMATH_CALUDE_pencil_price_solution_l1228_122868
