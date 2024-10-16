import Mathlib

namespace NUMINAMATH_CALUDE_janet_action_figures_l3956_395659

/-- Calculates the final number of action figures Janet has after a series of transactions --/
def final_action_figures (initial : ℕ) (sold : ℕ) (bought : ℕ) : ℕ → ℕ
| 0 => initial - sold + bought
| n + 1 => 2 * (final_action_figures initial sold bought n) + (final_action_figures initial sold bought n)

/-- Theorem stating that Janet ends up with 24 action figures --/
theorem janet_action_figures : final_action_figures 10 6 4 1 = 24 := by
  sorry

end NUMINAMATH_CALUDE_janet_action_figures_l3956_395659


namespace NUMINAMATH_CALUDE_square_area_l3956_395612

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + 5*x + 6

/-- The line function -/
def g (x : ℝ) : ℝ := 10

theorem square_area : ∃ (x₁ x₂ : ℝ), 
  f x₁ = g x₁ ∧ 
  f x₂ = g x₂ ∧ 
  x₁ ≠ x₂ ∧ 
  (x₂ - x₁)^2 = 41 := by
sorry

end NUMINAMATH_CALUDE_square_area_l3956_395612


namespace NUMINAMATH_CALUDE_augmented_matrix_solution_sum_l3956_395618

/-- Given an augmented matrix representing a system of linear equations,
    if the solution exists, then the sum of certain elements in the matrix is determined. -/
theorem augmented_matrix_solution_sum (m n : ℝ) : 
  (∃ x y : ℝ, m * x = 6 ∧ 3 * y = n ∧ x = -3 ∧ y = 4) → m + n = 10 := by
  sorry

end NUMINAMATH_CALUDE_augmented_matrix_solution_sum_l3956_395618


namespace NUMINAMATH_CALUDE_no_real_roots_l3956_395632

theorem no_real_roots : ∀ x : ℝ, 4 * x^2 + 4 * x + (5/4) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3956_395632


namespace NUMINAMATH_CALUDE_two_digit_sum_product_equality_l3956_395695

/-- P(n) is the product of the digits of n -/
def P (n : ℕ) : ℕ := sorry

/-- S(n) is the sum of the digits of n -/
def S (n : ℕ) : ℕ := sorry

/-- A two-digit number can be represented as 10a + b where a ≠ 0 -/
def isTwoDigit (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 10 * a + b

theorem two_digit_sum_product_equality :
  ∀ n : ℕ, isTwoDigit n → (n = P n + S n ↔ ∃ a : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ n = 10 * a + 9) :=
sorry

end NUMINAMATH_CALUDE_two_digit_sum_product_equality_l3956_395695


namespace NUMINAMATH_CALUDE_prob_HHT_fair_coin_l3956_395616

/-- A fair coin has equal probability of heads and tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of a sequence of independent events is the product of their individual probabilities -/
def prob_independent_events (p q r : ℝ) : ℝ := p * q * r

/-- The probability of getting heads on first two flips and tails on third flip for a fair coin -/
theorem prob_HHT_fair_coin (p : ℝ) (h : fair_coin p) : 
  prob_independent_events p p (1 - p) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_prob_HHT_fair_coin_l3956_395616


namespace NUMINAMATH_CALUDE_zain_coin_count_l3956_395693

/-- Represents the count of each coin type --/
structure CoinCount where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ

/-- Calculates the total number of coins --/
def totalCoins (coins : CoinCount) : ℕ :=
  coins.quarters + coins.dimes + coins.nickels

/-- Zain's coin count given Emerie's coin count --/
def zainCoins (emerieCoins : CoinCount) : CoinCount :=
  { quarters := emerieCoins.quarters + 10
  , dimes := emerieCoins.dimes + 10
  , nickels := emerieCoins.nickels + 10 }

theorem zain_coin_count (emerieCoins : CoinCount) 
  (h1 : emerieCoins.quarters = 6)
  (h2 : emerieCoins.dimes = 7)
  (h3 : emerieCoins.nickels = 5) : 
  totalCoins (zainCoins emerieCoins) = 48 := by
  sorry

end NUMINAMATH_CALUDE_zain_coin_count_l3956_395693


namespace NUMINAMATH_CALUDE_min_value_of_f_l3956_395671

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + a

-- State the theorem
theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y a ≤ f x a) ∧
  (f x a ≤ 11) ∧
  (∃ z ∈ Set.Icc (-2 : ℝ) 2, f z a = 11) →
  (∃ w ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f w a ≤ f y a ∧ f w a = -29) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3956_395671


namespace NUMINAMATH_CALUDE_exponent_problem_l3956_395606

theorem exponent_problem (a : ℝ) (x y : ℝ) (h1 : a^x = 2) (h2 : a^y = 3) : a^(2*x + y) = 12 := by
  sorry

end NUMINAMATH_CALUDE_exponent_problem_l3956_395606


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3956_395631

theorem polynomial_factorization (x m : ℝ) : 
  (x^2 + 6*x + 5 = (x+5)*(x+1)) ∧ (m^2 - m - 12 = (m+3)*(m-4)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3956_395631


namespace NUMINAMATH_CALUDE_range_of_m_l3956_395627

theorem range_of_m (m : ℝ) : 
  (m + 4)^(-1/2 : ℝ) < (3 - 2*m)^(-1/2 : ℝ) → 
  -1/3 < m ∧ m < 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3956_395627


namespace NUMINAMATH_CALUDE_union_equals_first_set_l3956_395666

theorem union_equals_first_set (I M N : Set α) : 
  M ⊂ I → N ⊂ I → M ≠ N → M.Nonempty → N.Nonempty → N ∩ (I \ M) = ∅ → M ∪ N = M := by
  sorry

end NUMINAMATH_CALUDE_union_equals_first_set_l3956_395666


namespace NUMINAMATH_CALUDE_distribution_6_boxes_8_floors_l3956_395687

/-- The number of ways to distribute boxes among floors with at least two on the top floor -/
def distributionWays (numBoxes numFloors : ℕ) : ℕ :=
  numFloors^numBoxes - (numFloors - 1)^numBoxes - numBoxes * (numFloors - 1)^(numBoxes - 1)

/-- Theorem: For 6 boxes and 8 floors, the number of distributions with at least 2 boxes on the top floor -/
theorem distribution_6_boxes_8_floors :
  distributionWays 6 8 = 8^6 - 13 * 7^5 := by
  sorry

end NUMINAMATH_CALUDE_distribution_6_boxes_8_floors_l3956_395687


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l3956_395640

theorem regular_polygon_interior_angle_sum 
  (n : ℕ) 
  (h_regular : n ≥ 3) 
  (h_exterior : (360 : ℝ) / n = 40) : 
  (n - 2 : ℝ) * 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l3956_395640


namespace NUMINAMATH_CALUDE_largest_coefficient_binomial_expansion_l3956_395665

theorem largest_coefficient_binomial_expansion :
  let n : ℕ := 7
  let expansion := fun (k : ℕ) => Nat.choose n k
  (∃ k : ℕ, k ≤ n ∧ expansion k = Finset.sup (Finset.range (n + 1)) expansion) →
  Finset.sup (Finset.range (n + 1)) expansion = 35 :=
by sorry

end NUMINAMATH_CALUDE_largest_coefficient_binomial_expansion_l3956_395665


namespace NUMINAMATH_CALUDE_lee_earnings_l3956_395637

/-- Represents Lee's lawn care services and earnings --/
structure LawnCareServices where
  mowing_price : ℕ
  trimming_price : ℕ
  weed_removal_price : ℕ
  mowed_lawns : ℕ
  trimmed_lawns : ℕ
  weed_removed_lawns : ℕ
  mowing_tips : ℕ
  trimming_tips : ℕ
  weed_removal_tips : ℕ

/-- Calculates the total earnings from Lee's lawn care services --/
def total_earnings (s : LawnCareServices) : ℕ :=
  s.mowing_price * s.mowed_lawns +
  s.trimming_price * s.trimmed_lawns +
  s.weed_removal_price * s.weed_removed_lawns +
  s.mowing_tips +
  s.trimming_tips +
  s.weed_removal_tips

/-- Theorem stating that Lee's total earnings for the week were $747 --/
theorem lee_earnings : 
  let s : LawnCareServices := {
    mowing_price := 33,
    trimming_price := 15,
    weed_removal_price := 10,
    mowed_lawns := 16,
    trimmed_lawns := 8,
    weed_removed_lawns := 5,
    mowing_tips := 3 * 10,
    trimming_tips := 2 * 7,
    weed_removal_tips := 1 * 5
  }
  total_earnings s = 747 := by
  sorry


end NUMINAMATH_CALUDE_lee_earnings_l3956_395637


namespace NUMINAMATH_CALUDE_brass_players_count_l3956_395628

/-- Represents the composition of a marching band -/
structure MarchingBand where
  brass : ℕ
  woodwind : ℕ
  percussion : ℕ

/-- The total number of members in the marching band -/
def MarchingBand.total (band : MarchingBand) : ℕ :=
  band.brass + band.woodwind + band.percussion

/-- Theorem: The number of brass players in the marching band is 10 -/
theorem brass_players_count (band : MarchingBand) :
  band.total = 110 →
  band.woodwind = 2 * band.brass →
  band.percussion = 4 * band.woodwind →
  band.brass = 10 := by
  sorry

end NUMINAMATH_CALUDE_brass_players_count_l3956_395628


namespace NUMINAMATH_CALUDE_maximize_product_l3956_395677

theorem maximize_product (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 50) :
  x^3 * y^4 ≤ 30^3 * 20^4 ∧
  (x^3 * y^4 = 30^3 * 20^4 ↔ x = 30 ∧ y = 20) :=
by sorry

end NUMINAMATH_CALUDE_maximize_product_l3956_395677


namespace NUMINAMATH_CALUDE_zoe_pop_albums_l3956_395649

/-- Represents the number of songs in each album -/
def songs_per_album : ℕ := 3

/-- Represents the number of country albums bought -/
def country_albums : ℕ := 3

/-- Represents the total number of songs bought -/
def total_songs : ℕ := 24

/-- Calculates the number of pop albums bought -/
def pop_albums : ℕ := (total_songs - country_albums * songs_per_album) / songs_per_album

theorem zoe_pop_albums : pop_albums = 5 := by
  sorry

end NUMINAMATH_CALUDE_zoe_pop_albums_l3956_395649


namespace NUMINAMATH_CALUDE_area_of_region_l3956_395633

-- Define the curve and line
def curve (x : ℝ) : ℝ → Prop := λ y ↦ y^2 = 2*x
def line (x : ℝ) : ℝ → Prop := λ y ↦ y = x - 4

-- Define the region
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ curve x y ∧ line x y}

-- State the theorem
theorem area_of_region : MeasureTheory.volume region = 18 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l3956_395633


namespace NUMINAMATH_CALUDE_factorization_proof_l3956_395617

theorem factorization_proof (z : ℝ) : 
  75 * z^12 + 162 * z^24 + 27 = 3 * (9 + z^12 * (25 + 54 * z^12)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3956_395617


namespace NUMINAMATH_CALUDE_cube_sum_problem_l3956_395682

theorem cube_sum_problem (p q r : ℝ) 
  (h1 : p + q + r = 5)
  (h2 : p * q + p * r + q * r = 7)
  (h3 : p * q * r = -10) :
  p^3 + q^3 + r^3 = -10 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l3956_395682


namespace NUMINAMATH_CALUDE_hyperbola_condition_l3956_395668

/-- For the equation x²/(2+m) - y²/(m+1) = 1 to represent a hyperbola, 
    m must satisfy: m > -1 or m < -2 -/
theorem hyperbola_condition (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (2 + m) - y^2 / (m + 1) = 1 ∧ 
   (2 + m ≠ 0 ∧ m + 1 ≠ 0)) ↔ (m > -1 ∨ m < -2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l3956_395668


namespace NUMINAMATH_CALUDE_optimal_sapling_positions_l3956_395686

/-- Represents the number of trees planted -/
def num_trees : ℕ := 20

/-- Represents the distance between adjacent trees in meters -/
def tree_spacing : ℕ := 10

/-- Calculates the total distance walked by students for given sapling positions -/
def total_distance (pos1 pos2 : ℕ) : ℕ := sorry

/-- Theorem stating that positions 10 and 11 minimize the total distance -/
theorem optimal_sapling_positions :
  ∀ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ num_trees →
    total_distance 10 11 ≤ total_distance a b :=
by sorry

end NUMINAMATH_CALUDE_optimal_sapling_positions_l3956_395686


namespace NUMINAMATH_CALUDE_page_number_added_twice_l3956_395664

theorem page_number_added_twice 
  (n : ℕ) 
  (h1 : 60 ≤ n ∧ n ≤ 70) 
  (h2 : ∃ k : ℕ, k ≤ n ∧ n * (n + 1) / 2 + k = 2378) : 
  ∃ k : ℕ, k ≤ n ∧ n * (n + 1) / 2 + k = 2378 ∧ k = 32 := by
  sorry

end NUMINAMATH_CALUDE_page_number_added_twice_l3956_395664


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3956_395644

theorem shaded_area_calculation (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) : 
  square_side = 40 →
  triangle_base = 30 →
  triangle_height = 30 →
  square_side * square_side - 2 * (1/2 * triangle_base * triangle_height) = 700 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3956_395644


namespace NUMINAMATH_CALUDE_greatest_divisor_of_sequence_l3956_395674

theorem greatest_divisor_of_sequence :
  ∃ (x : ℕ), x = 18 ∧ 
  (∀ y : ℕ, x ∣ (7^y + 12*y - 1)) ∧
  (∀ z : ℕ, (∀ y : ℕ, z ∣ (7^y + 12*y - 1)) → z ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_of_sequence_l3956_395674


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3956_395635

/-- Proves that the speed of a boat in still water is 25 km/hr, given the speed of the stream
    and the time and distance traveled downstream. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_time = 3)
  (h3 : downstream_distance = 90) :
  (downstream_distance / downstream_time) - stream_speed = 25 := by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3956_395635


namespace NUMINAMATH_CALUDE_cricketer_average_score_l3956_395600

theorem cricketer_average_score (avg1 avg2 overall_avg : ℚ) (n1 n2 : ℕ) : 
  avg1 = 30 → avg2 = 40 → overall_avg = 36 → n1 = 2 → n2 = 3 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = overall_avg →
  n1 + n2 = 5 := by sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l3956_395600


namespace NUMINAMATH_CALUDE_base_subtraction_equality_l3956_395608

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The problem statement -/
theorem base_subtraction_equality : 
  let base_6_num := [5, 2, 3]  -- 325 in base 6 (least significant digit first)
  let base_5_num := [1, 3, 2]  -- 231 in base 5 (least significant digit first)
  (to_base_10 base_6_num 6) - (to_base_10 base_5_num 5) = 59 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_equality_l3956_395608


namespace NUMINAMATH_CALUDE_cos_150_degrees_l3956_395610

theorem cos_150_degrees : Real.cos (150 * π / 180) = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l3956_395610


namespace NUMINAMATH_CALUDE_product_of_numbers_l3956_395613

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 6) (h2 : x^2 + y^2 = 170) : x * y = -67 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3956_395613


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3956_395621

def is_geometric_sequence (a : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

theorem geometric_sequence_sum (a : ℕ → ℝ) (h : is_geometric_sequence a 6) 
  (h1 : a 1 = 1) (h2 : a 2 = 2) : 
  (Finset.range 9).sum (λ i => a (i + 1)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3956_395621


namespace NUMINAMATH_CALUDE_intersection_implies_c_18_l3956_395652

-- Define the functions
def f (x : ℝ) : ℝ := |x - 20| + |x + 18|
def g (c x : ℝ) : ℝ := x + c

-- Define the intersection condition
def unique_intersection (c : ℝ) : Prop :=
  ∃! x, f x = g c x

-- Theorem statement
theorem intersection_implies_c_18 :
  ∀ c : ℝ, unique_intersection c → c = 18 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_c_18_l3956_395652


namespace NUMINAMATH_CALUDE_white_ball_players_l3956_395626

theorem white_ball_players (total : ℕ) (yellow : ℕ) (both : ℕ) (h1 : total = 35) (h2 : yellow = 28) (h3 : both = 19) :
  total = (yellow - both) + (total - yellow + both) :=
by sorry

#check white_ball_players

end NUMINAMATH_CALUDE_white_ball_players_l3956_395626


namespace NUMINAMATH_CALUDE_fibonacci_pair_characterization_l3956_395663

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def is_fibonacci_pair (a b : ℝ) : Prop :=
  ∀ n, ∃ m, a * (fibonacci n) + b * (fibonacci (n + 1)) = fibonacci m

theorem fibonacci_pair_characterization :
  ∀ a b : ℝ, is_fibonacci_pair a b ↔ 
    ((a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 0) ∨ 
     (∃ k : ℕ, a = fibonacci k ∧ b = fibonacci (k + 1))) :=
sorry

end NUMINAMATH_CALUDE_fibonacci_pair_characterization_l3956_395663


namespace NUMINAMATH_CALUDE_square_tiles_count_l3956_395647

theorem square_tiles_count (h s : ℕ) : 
  h + s = 30 →  -- Total number of tiles
  6 * h + 4 * s = 128 →  -- Total number of edges
  s = 26 :=  -- Number of square tiles
by
  sorry

end NUMINAMATH_CALUDE_square_tiles_count_l3956_395647


namespace NUMINAMATH_CALUDE_odd_function_sum_l3956_395646

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def periodic_4 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) = f x

theorem odd_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_periodic : ∀ x, f x + f (4 - x) = 0)
  (h_f_1 : f 1 = 8) :
  f 2010 + f 2011 + f 2012 = -8 := by
sorry

end NUMINAMATH_CALUDE_odd_function_sum_l3956_395646


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3956_395672

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x^2022 + y^2 = 2*y + 2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3956_395672


namespace NUMINAMATH_CALUDE_ramon_twice_loui_in_twenty_years_loui_age_is_23_l3956_395690

/-- Ramon's current age -/
def ramon_current_age : ℕ := 26

/-- Loui's current age -/
def loui_current_age : ℕ := 23

/-- In twenty years, Ramon will be twice as old as Loui today -/
theorem ramon_twice_loui_in_twenty_years :
  ramon_current_age + 20 = 2 * loui_current_age := by sorry

theorem loui_age_is_23 : loui_current_age = 23 := by sorry

end NUMINAMATH_CALUDE_ramon_twice_loui_in_twenty_years_loui_age_is_23_l3956_395690


namespace NUMINAMATH_CALUDE_area_of_problem_l_shape_l3956_395661

/-- Represents an L-shaped figure with given dimensions -/
structure LShape where
  short_width : ℝ
  short_length : ℝ
  long_width : ℝ
  long_length : ℝ

/-- Calculates the area of an L-shaped figure -/
def area_of_l_shape (l : LShape) : ℝ :=
  l.short_width * l.short_length + l.long_width * l.long_length

/-- The specific L-shape from the problem -/
def problem_l_shape : LShape :=
  { short_width := 2
    short_length := 3
    long_width := 5
    long_length := 8 }

/-- Theorem stating that the area of the given L-shape is 46 square units -/
theorem area_of_problem_l_shape :
  area_of_l_shape problem_l_shape = 46 := by
  sorry


end NUMINAMATH_CALUDE_area_of_problem_l_shape_l3956_395661


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3956_395634

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 < x + 6} = {x : ℝ | -2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3956_395634


namespace NUMINAMATH_CALUDE_adam_tickets_left_l3956_395691

/-- The number of tickets Adam had left after riding the ferris wheel -/
def tickets_left (initial_tickets : ℕ) (ticket_cost : ℕ) (spent_on_ride : ℕ) : ℕ :=
  initial_tickets - (spent_on_ride / ticket_cost)

/-- Theorem stating that Adam had 4 tickets left after riding the ferris wheel -/
theorem adam_tickets_left :
  tickets_left 13 9 81 = 4 := by
  sorry

#eval tickets_left 13 9 81

end NUMINAMATH_CALUDE_adam_tickets_left_l3956_395691


namespace NUMINAMATH_CALUDE_investment_value_l3956_395605

theorem investment_value (x : ℝ) : 
  (0.07 * 500 + 0.23 * x = 0.19 * (500 + x)) → x = 1500 := by
  sorry

end NUMINAMATH_CALUDE_investment_value_l3956_395605


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l3956_395630

/-- A right triangle with an inscribed square -/
structure RightTriangleWithSquare where
  -- Sides of the right triangle
  de : ℝ
  ef : ℝ
  df : ℝ
  -- The triangle is right-angled
  is_right : de^2 + ef^2 = df^2
  -- Side lengths are positive
  de_pos : de > 0
  ef_pos : ef > 0
  df_pos : df > 0
  -- The square is inscribed in the triangle
  square_inscribed : True

/-- The side length of the inscribed square -/
def square_side_length (t : RightTriangleWithSquare) : ℝ := sorry

/-- Theorem stating that for a right triangle with sides 6, 8, and 10, 
    the inscribed square has side length 120/37 -/
theorem inscribed_square_side_length :
  let t : RightTriangleWithSquare := {
    de := 6,
    ef := 8,
    df := 10,
    is_right := by norm_num,
    de_pos := by norm_num,
    ef_pos := by norm_num,
    df_pos := by norm_num,
    square_inscribed := trivial
  }
  square_side_length t = 120 / 37 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l3956_395630


namespace NUMINAMATH_CALUDE_power_sum_equals_zero_l3956_395667

theorem power_sum_equals_zero : (-1)^2021 + 1^2022 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_zero_l3956_395667


namespace NUMINAMATH_CALUDE_range_of_m_l3956_395678

def A : Set ℝ := {x | x ≤ -2 ∨ x ≥ -1}
def B (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x ≤ 2*m}

theorem range_of_m (m : ℝ) :
  (A ∩ B m = ∅) → (A ∪ B m = A) → m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3956_395678


namespace NUMINAMATH_CALUDE_right_to_left_equiv_ordinary_l3956_395614

-- Define a function to represent the right-to-left evaluation
def rightToLeftEval (a b c d : ℝ) : ℝ := a * (b + (c - d))

-- Define a function to represent the ordinary algebraic notation
def ordinaryNotation (a b c d : ℝ) : ℝ := a * (b + c - d)

-- Theorem statement
theorem right_to_left_equiv_ordinary (a b c d : ℝ) :
  rightToLeftEval a b c d = ordinaryNotation a b c d := by
  sorry

end NUMINAMATH_CALUDE_right_to_left_equiv_ordinary_l3956_395614


namespace NUMINAMATH_CALUDE_circle_path_in_triangle_l3956_395653

theorem circle_path_in_triangle (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_sides : a = 9 ∧ b = 12 ∧ c = 15) (r : ℝ) (h_radius : r = 2) :
  let p := a + b + c
  let s := (c - 2*r) / c
  (s * p) = 26.4 := by sorry

end NUMINAMATH_CALUDE_circle_path_in_triangle_l3956_395653


namespace NUMINAMATH_CALUDE_range_of_m_when_a_is_one_range_of_a_for_sufficient_condition_l3956_395624

-- Define propositions p and q
def p (m a : ℝ) : Prop := m^2 - 7*a*m + 12*a^2 < 0 ∧ a > 0

def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2/(m-1) + y^2/(6-m) = 1 ∧ 1 < m ∧ m < 6

-- Theorem for part 1
theorem range_of_m_when_a_is_one :
  ∀ m : ℝ, (p m 1 ∧ q m) → (3 < m ∧ m < 7/2) :=
sorry

-- Theorem for part 2
theorem range_of_a_for_sufficient_condition :
  (∀ m a : ℝ, ¬(q m) → ¬(p m a)) ∧ (∃ m a : ℝ, ¬(p m a) ∧ q m) →
  (∀ a : ℝ, 1/3 ≤ a ∧ a ≤ 7/8) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_when_a_is_one_range_of_a_for_sufficient_condition_l3956_395624


namespace NUMINAMATH_CALUDE_triangle_properties_l3956_395609

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

/-- The main theorem -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.c * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.C)
  (h2 : t.a^2 - t.c^2 = 2 * t.b^2)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = 21 * Real.sqrt 3) :
  t.C = π/3 ∧ t.b = 2 * Real.sqrt 7 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l3956_395609


namespace NUMINAMATH_CALUDE_proportion_solution_l3956_395656

theorem proportion_solution (x : ℝ) (h : (0.75 : ℝ) / x = 5 / 8) : x = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l3956_395656


namespace NUMINAMATH_CALUDE_triangle_properties_l3956_395679

/-- Given a triangle ABC with the following properties:
  * The area of the triangle is 3√15
  * b - c = 2, where b and c are sides of the triangle
  * cos A = -1/4, where A is an angle of the triangle
This theorem proves specific values for a, sin C, and cos(2A + π/6) -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (h_area : (1/2) * b * c * Real.sin A = 3 * Real.sqrt 15)
  (h_sides : b - c = 2)
  (h_cos_A : Real.cos A = -1/4) :
  a = 8 ∧ 
  Real.sin C = Real.sqrt 15 / 8 ∧
  Real.cos (2 * A + π/6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3956_395679


namespace NUMINAMATH_CALUDE_theater_ticket_cost_l3956_395654

/-- Proves that the cost of an adult ticket is $7.56 given the conditions of the theater ticket sales. -/
theorem theater_ticket_cost (total_tickets : ℕ) (total_receipts : ℚ) (adult_tickets : ℕ) (child_ticket_cost : ℚ) :
  total_tickets = 130 →
  total_receipts = 840 →
  adult_tickets = 90 →
  child_ticket_cost = 4 →
  ∃ adult_ticket_cost : ℚ,
    adult_ticket_cost * adult_tickets + child_ticket_cost * (total_tickets - adult_tickets) = total_receipts ∧
    adult_ticket_cost = 756 / 100 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_cost_l3956_395654


namespace NUMINAMATH_CALUDE_work_duration_l3956_395657

/-- Given that x does a work in 20 days and x and y together do the same work in 40/3 days,
    prove that y does the work in 40 days. -/
theorem work_duration (x y : ℝ) (h1 : x = 20) (h2 : 1 / x + 1 / y = 3 / 40) : y = 40 := by
  sorry

end NUMINAMATH_CALUDE_work_duration_l3956_395657


namespace NUMINAMATH_CALUDE_line_point_value_l3956_395694

/-- Given a line passing through points (-1, y) and (4, k), with slope equal to k and k = 1, 
    prove that y = -4 -/
theorem line_point_value (y k : ℝ) (h1 : k = 1) 
    (h2 : (k - y) / (4 - (-1)) = k) : y = -4 := by
  sorry

end NUMINAMATH_CALUDE_line_point_value_l3956_395694


namespace NUMINAMATH_CALUDE_triangle_side_length_l3956_395638

/-- Given a triangle ABC with side a = 8, angle B = 30°, and angle C = 105°, 
    prove that the length of side b is equal to 4√2. -/
theorem triangle_side_length (a b : ℝ) (A B C : ℝ) : 
  a = 8 → B = 30 * π / 180 → C = 105 * π / 180 → 
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  b = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3956_395638


namespace NUMINAMATH_CALUDE_equation_solutions_l3956_395685

theorem equation_solutions : 
  (∃ (x₁ x₂ : ℝ), x₁ = 5 + Real.sqrt 35 ∧ x₂ = 5 - Real.sqrt 35 ∧ 
    x₁^2 - 10*x₁ - 10 = 0 ∧ x₂^2 - 10*x₂ - 10 = 0) ∧
  (∃ (y₁ y₂ : ℝ), y₁ = 5 ∧ y₂ = 13/3 ∧ 
    3*(y₁ - 5)^2 = 2*(5 - y₁) ∧ 3*(y₂ - 5)^2 = 2*(5 - y₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3956_395685


namespace NUMINAMATH_CALUDE_equal_diagonal_quadrilateral_multiple_shapes_l3956_395619

/-- A quadrilateral with equal-length diagonals -/
structure EqualDiagonalQuadrilateral where
  /-- The length of the diagonals -/
  diagonal_length : ℝ
  /-- The quadrilateral has positive area -/
  positive_area : ℝ
  area_pos : positive_area > 0

/-- Possible shapes of a quadrilateral -/
inductive QuadrilateralShape
  | Square
  | Rectangle
  | IsoscelesTrapezoid
  | Other

/-- A function that determines if a given shape is possible for an equal-diagonal quadrilateral -/
def is_possible_shape (q : EqualDiagonalQuadrilateral) (shape : QuadrilateralShape) : Prop :=
  ∃ (quad : EqualDiagonalQuadrilateral), quad.diagonal_length = q.diagonal_length ∧ 
    quad.positive_area = q.positive_area ∧ 
    (match shape with
      | QuadrilateralShape.Square => true
      | QuadrilateralShape.Rectangle => true
      | QuadrilateralShape.IsoscelesTrapezoid => true
      | QuadrilateralShape.Other => true)

theorem equal_diagonal_quadrilateral_multiple_shapes (q : EqualDiagonalQuadrilateral) :
  (is_possible_shape q QuadrilateralShape.Square) ∧
  (is_possible_shape q QuadrilateralShape.Rectangle) ∧
  (is_possible_shape q QuadrilateralShape.IsoscelesTrapezoid) :=
sorry

end NUMINAMATH_CALUDE_equal_diagonal_quadrilateral_multiple_shapes_l3956_395619


namespace NUMINAMATH_CALUDE_system_solution_l3956_395639

theorem system_solution (u v w : ℝ) (hu : u ≠ 0) (hv : v ≠ 0) (hw : w ≠ 0)
  (eq1 : 3 / (u * v) + 15 / (v * w) = 2)
  (eq2 : 15 / (v * w) + 5 / (w * u) = 2)
  (eq3 : 5 / (w * u) + 3 / (u * v) = 2) :
  (u = 1 ∧ v = 3 ∧ w = 5) ∨ (u = -1 ∧ v = -3 ∧ w = -5) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3956_395639


namespace NUMINAMATH_CALUDE_inheritance_calculation_l3956_395603

theorem inheritance_calculation (inheritance : ℝ) : 
  inheritance * 0.25 + (inheritance - inheritance * 0.25) * 0.15 = 15000 → 
  inheritance = 41379 := by
sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l3956_395603


namespace NUMINAMATH_CALUDE_f_has_three_zeros_l3956_395611

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then (1/2)^x + 2/x
  else if x > 0 then x * Real.log x - a
  else 0

theorem f_has_three_zeros (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    (∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) ↔
  (-1 / Real.exp 1 < a ∧ a < 0) :=
sorry

end NUMINAMATH_CALUDE_f_has_three_zeros_l3956_395611


namespace NUMINAMATH_CALUDE_sector_chord_length_l3956_395689

/-- Given a circular sector with area 1 cm² and perimeter 4 cm, its chord length is 2sin(1) cm. -/
theorem sector_chord_length (r : ℝ) (α : ℝ) :
  (1/2 * α * r^2 = 1) →  -- Area of sector is 1 cm²
  (2 * r + α * r = 4) →  -- Perimeter of sector is 4 cm
  (2 * r * Real.sin (α/2) = 2 * Real.sin 1) := by
  sorry

end NUMINAMATH_CALUDE_sector_chord_length_l3956_395689


namespace NUMINAMATH_CALUDE_negative_twenty_seven_to_five_thirds_l3956_395607

theorem negative_twenty_seven_to_five_thirds :
  (-27 : ℝ) ^ (5/3) = -243 := by
  sorry

end NUMINAMATH_CALUDE_negative_twenty_seven_to_five_thirds_l3956_395607


namespace NUMINAMATH_CALUDE_simplify_sqrt_product_l3956_395670

theorem simplify_sqrt_product : Real.sqrt 18 * Real.sqrt 72 = 12 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_l3956_395670


namespace NUMINAMATH_CALUDE_tiling_8x2_equals_fib_9_l3956_395645

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of ways to tile a 2 × n rectangle with 1 × 2 dominoes -/
def tiling_ways (n : ℕ) : ℕ := fib (n + 1)

/-- Theorem: The number of ways to tile an 8 × 2 rectangle with 1 × 2 dominoes
    is equal to the 9th Fibonacci number -/
theorem tiling_8x2_equals_fib_9 :
  tiling_ways 8 = fib 9 := by
  sorry

#eval tiling_ways 8  -- Expected output: 34

end NUMINAMATH_CALUDE_tiling_8x2_equals_fib_9_l3956_395645


namespace NUMINAMATH_CALUDE_negative_64_to_four_thirds_equals_256_l3956_395662

theorem negative_64_to_four_thirds_equals_256 : (-64 : ℝ) ^ (4/3) = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_64_to_four_thirds_equals_256_l3956_395662


namespace NUMINAMATH_CALUDE_range_of_m_l3956_395641

/-- Proposition p: For any real number x, mx^2 + mx + 1 > 0 always holds -/
def p (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 + m * x + 1 > 0

/-- Proposition q: The equation x^2 / (m-1) + y^2 / (m-2) = 1 represents a hyperbola with foci on the x-axis -/
def q (m : ℝ) : Prop := m > 1 ∧ m < 2

/-- The set of m values satisfying the conditions -/
def M : Set ℝ := {m : ℝ | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

theorem range_of_m : M = Set.Icc 0 1 ∪ Set.Ico 2 4 := by
  sorry

#check range_of_m

end NUMINAMATH_CALUDE_range_of_m_l3956_395641


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3956_395650

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  (a^2 + 1) / a + (2*b^2 + 1) / b ≥ 4 + 2*Real.sqrt 2 :=
by sorry

theorem min_value_achievable :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 2*b = 1 ∧
  (a^2 + 1) / a + (2*b^2 + 1) / b = 4 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3956_395650


namespace NUMINAMATH_CALUDE_line_equation_l3956_395699

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 5

-- Define the line l
def line_l (x y : ℝ) : Prop := ∃ (m b : ℝ), y = m * x + b

-- Define the center of the circle
def center : ℝ × ℝ := (3, 5)

-- Define that line l passes through the center
def line_through_center (l : ℝ → ℝ → Prop) : Prop :=
  l center.1 center.2

-- Define points A and B on the circle and line
def point_on_circle_and_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  circle_C p.1 p.2 ∧ l p.1 p.2

-- Define point P on y-axis and line
def point_on_y_axis_and_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  p.1 = 0 ∧ l p.1 p.2

-- Define A as midpoint of BP
def A_midpoint_BP (A B P : ℝ × ℝ) : Prop :=
  A.1 = (B.1 + P.1) / 2 ∧ A.2 = (B.2 + P.2) / 2

-- Theorem statement
theorem line_equation :
  ∀ (A B P : ℝ × ℝ) (l : ℝ → ℝ → Prop),
    line_l = l →
    line_through_center l →
    point_on_circle_and_line A l →
    point_on_circle_and_line B l →
    point_on_y_axis_and_line P l →
    A_midpoint_BP A B P →
    (∀ x y, l x y ↔ (2*x - y - 1 = 0 ∨ 2*x + y - 11 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l3956_395699


namespace NUMINAMATH_CALUDE_range_of_a_l3956_395660

-- Define the condition
def inequality_holds (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → 2 * x + 1 / x - a > 0

-- State the theorem
theorem range_of_a (a : ℝ) : inequality_holds a → a < 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3956_395660


namespace NUMINAMATH_CALUDE_holly_insulin_pills_l3956_395658

/-- Represents the number of pills Holly takes per day for each type of medication -/
structure DailyPills where
  insulin : ℕ
  blood_pressure : ℕ
  anticonvulsant : ℕ

/-- Calculates the total number of pills Holly takes in a week -/
def weekly_total (d : DailyPills) : ℕ :=
  7 * (d.insulin + d.blood_pressure + d.anticonvulsant)

/-- Holly's daily pill regimen satisfies the given conditions -/
def holly_pills : DailyPills :=
  { insulin := 2,
    blood_pressure := 3,
    anticonvulsant := 6 }

theorem holly_insulin_pills :
  holly_pills.insulin = 2 ∧
  holly_pills.blood_pressure = 3 ∧
  holly_pills.anticonvulsant = 2 * holly_pills.blood_pressure ∧
  weekly_total holly_pills = 77 := by
  sorry

end NUMINAMATH_CALUDE_holly_insulin_pills_l3956_395658


namespace NUMINAMATH_CALUDE_fraction_sum_difference_l3956_395629

theorem fraction_sum_difference (p q r s : ℚ) 
  (h1 : p / q = 4 / 5) 
  (h2 : r / s = 3 / 7) : 
  (18 / 7) + ((2 * q - p) / (2 * q + p)) - ((3 * s + r) / (3 * s - r)) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_difference_l3956_395629


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3956_395680

/-- Given a hyperbola and a parabola satisfying certain conditions, 
    prove that the hyperbola has a specific equation. -/
theorem hyperbola_equation 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (asymptote : b/a = Real.sqrt 3) 
  (focus_on_directrix : a^2 + b^2 = 36) : 
  a^2 = 9 ∧ b^2 = 27 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3956_395680


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3956_395698

theorem imaginary_part_of_z (i : ℂ) (z : ℂ) : 
  i^2 = -1 → z * (1 + i) = Complex.abs (i + 1) → Complex.im z = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3956_395698


namespace NUMINAMATH_CALUDE_sqrt_neg_five_squared_l3956_395697

theorem sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_five_squared_l3956_395697


namespace NUMINAMATH_CALUDE_count_polynomials_l3956_395636

-- Define a function to check if an expression is a polynomial
def isPolynomial (expr : String) : Bool :=
  match expr with
  | "-7" => true
  | "m" => true
  | "x^3y^2" => true
  | "1/a" => false
  | "2x+3y" => true
  | _ => false

-- Define the list of expressions
def expressions : List String := ["-7", "m", "x^3y^2", "1/a", "2x+3y"]

-- Theorem to prove
theorem count_polynomials :
  (expressions.filter isPolynomial).length = 4 := by sorry

end NUMINAMATH_CALUDE_count_polynomials_l3956_395636


namespace NUMINAMATH_CALUDE_x_value_proof_l3956_395673

theorem x_value_proof (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_x_lt_y : x < y)
  (h_eq1 : Real.sqrt x + Real.sqrt y = 4)
  (h_eq2 : Real.sqrt (x + 2) + Real.sqrt (y + 2) = 5) :
  x = 49 / 36 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l3956_395673


namespace NUMINAMATH_CALUDE_num_triangles_in_circle_l3956_395604

/-- The number of points on the circle -/
def n : ℕ := 9

/-- The number of chords -/
def num_chords : ℕ := n.choose 2

/-- The number of intersection points inside the circle -/
def num_intersections : ℕ := n.choose 4

/-- Theorem: The number of triangles formed by intersection points of chords inside a circle -/
theorem num_triangles_in_circle (n : ℕ) (h : n = 9) : 
  (num_intersections.choose 3) = 315500 :=
sorry

end NUMINAMATH_CALUDE_num_triangles_in_circle_l3956_395604


namespace NUMINAMATH_CALUDE_roberta_shopping_trip_l3956_395601

def shopping_trip (initial_amount bag_price_difference lunch_price_fraction : ℚ) : ℚ :=
  let shoe_price := 45
  let bag_price := shoe_price - bag_price_difference
  let lunch_price := bag_price * lunch_price_fraction
  initial_amount - (shoe_price + bag_price + lunch_price)

theorem roberta_shopping_trip :
  shopping_trip 158 17 (1/4) = 78 := by
  sorry

end NUMINAMATH_CALUDE_roberta_shopping_trip_l3956_395601


namespace NUMINAMATH_CALUDE_circle_ellipse_tangent_l3956_395602

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 5

-- Define the ellipse E
def ellipse_E (x y : ℝ) : Prop := x^2 / 18 + y^2 / 2 = 1

-- Define the line PF₁
def line_PF1 (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define the point A
def point_A : ℝ × ℝ := (3, 1)

-- Define the point P
def point_P : ℝ × ℝ := (4, 4)

-- State the theorem
theorem circle_ellipse_tangent :
  ∃ (m : ℝ),
    m < 3 ∧
    (∃ (a b : ℝ), a > b ∧ b > 0 ∧
      (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ ellipse_E x y)) ∧
    (∃ e : ℝ, e > 1/2 ∧
      (∀ x y : ℝ, ((x - m)^2 + y^2 = 5) ↔ circle_C x y) ∧
      circle_C point_A.1 point_A.2 ∧
      ellipse_E point_A.1 point_A.2 ∧
      line_PF1 point_P.1 point_P.2) :=
sorry

end NUMINAMATH_CALUDE_circle_ellipse_tangent_l3956_395602


namespace NUMINAMATH_CALUDE_extremum_and_solutions_l3956_395669

/-- A function with an extremum at x = 0 -/
noncomputable def f (a b x : ℝ) : ℝ := x^2 + x - Real.log (x + a) + 3*b

/-- The statement to be proved -/
theorem extremum_and_solutions (a b : ℝ) :
  (f a b 0 = 0 ∧ ∀ x, f a b x ≥ f a b 0) →
  (a = 1 ∧ b = 0) ∧
  ∀ m : ℝ, (∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, -1/2 ≤ x ∧ x ≤ 2 ∧ f 1 0 x = m) ↔
    (0 < m ∧ m ≤ -1/4 + Real.log 2) :=
by sorry


end NUMINAMATH_CALUDE_extremum_and_solutions_l3956_395669


namespace NUMINAMATH_CALUDE_cosine_sine_sum_identity_l3956_395681

theorem cosine_sine_sum_identity : 
  Real.cos (43 * π / 180) * Real.cos (77 * π / 180) + 
  Real.sin (43 * π / 180) * Real.cos (167 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_identity_l3956_395681


namespace NUMINAMATH_CALUDE_difference_of_squares_l3956_395615

theorem difference_of_squares : (535 : ℕ)^2 - (465 : ℕ)^2 = 70000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3956_395615


namespace NUMINAMATH_CALUDE_rectangle_length_l3956_395684

/-- The perimeter of a rectangle -/
def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: For a rectangle with perimeter 1200 and width 500, its length is 100 -/
theorem rectangle_length (p w : ℝ) (h1 : p = 1200) (h2 : w = 500) :
  ∃ l : ℝ, perimeter l w = p ∧ l = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l3956_395684


namespace NUMINAMATH_CALUDE_slope_of_line_l3956_395675

theorem slope_of_line (x y : ℝ) : 4 * x + 7 * y = 28 → (y - 4) / x = -4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l3956_395675


namespace NUMINAMATH_CALUDE_symmetric_point_is_correct_l3956_395651

/-- The line of symmetry --/
def line_of_symmetry (x y : ℝ) : Prop := x + 2*y - 10 = 0

/-- The original point --/
def original_point : ℝ × ℝ := (1, 2)

/-- The symmetric point --/
def symmetric_point : ℝ × ℝ := (3, 6)

/-- Checks if two points are symmetric with respect to a line --/
def is_symmetric (p1 p2 : ℝ × ℝ) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  line_of_symmetry midpoint.1 midpoint.2 ∧
  (p2.2 - p1.2) * (1 : ℝ) = (p2.1 - p1.1) * (-2 : ℝ)

theorem symmetric_point_is_correct : 
  is_symmetric original_point symmetric_point :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_is_correct_l3956_395651


namespace NUMINAMATH_CALUDE_water_fraction_after_replacements_l3956_395648

/-- Represents the fraction of water in the radiator mixture -/
def water_fraction (n : ℕ) : ℚ :=
  (3/4 : ℚ) ^ n

/-- The radiator capacity in quarts -/
def radiator_capacity : ℕ := 16

/-- The amount of mixture removed and replaced in each iteration -/
def replacement_amount : ℕ := 4

/-- The number of replacement iterations -/
def num_iterations : ℕ := 4

theorem water_fraction_after_replacements :
  water_fraction num_iterations = 81/256 := by
  sorry

end NUMINAMATH_CALUDE_water_fraction_after_replacements_l3956_395648


namespace NUMINAMATH_CALUDE_spoons_to_knives_ratio_l3956_395688

/-- Represents the initial number of each type of cutlery and the total after adding 2 of each -/
structure Cutlery where
  forks : ℕ
  knives : ℕ
  teaspoons : ℕ
  spoons : ℕ
  total_after_adding : ℕ

/-- The given conditions for the cutlery problem -/
def cutlery_conditions : Cutlery :=
  { forks := 6,
    knives := 6 + 9,
    teaspoons := 6 / 2,
    spoons := 28,  -- This is what we're proving
    total_after_adding := 62 }

/-- Theorem stating that the ratio of spoons to knives is 28:15 given the conditions -/
theorem spoons_to_knives_ratio (c : Cutlery) 
  (h1 : c.forks = 6)
  (h2 : c.knives = c.forks + 9)
  (h3 : c.teaspoons = c.forks / 2)
  (h4 : c.total_after_adding = c.forks + 2 + c.knives + 2 + c.teaspoons + 2 + c.spoons + 2) :
  c.spoons * 15 = 28 * c.knives := by
  sorry

#check spoons_to_knives_ratio

end NUMINAMATH_CALUDE_spoons_to_knives_ratio_l3956_395688


namespace NUMINAMATH_CALUDE_flag_arrangement_remainder_l3956_395622

/-- The number of ways to arrange flags on two poles -/
def M : ℕ := sorry

/-- The total number of flags -/
def total_flags : ℕ := 24

/-- The number of blue flags -/
def blue_flags : ℕ := 14

/-- The number of red flags -/
def red_flags : ℕ := 10

/-- Each flagpole has at least one flag -/
axiom at_least_one_flag : M > 0

/-- Each sequence starts with a blue flag -/
axiom starts_with_blue : True

/-- No two red flags on either pole are adjacent -/
axiom no_adjacent_red : True

theorem flag_arrangement_remainder :
  M % 1000 = 1 := by sorry

end NUMINAMATH_CALUDE_flag_arrangement_remainder_l3956_395622


namespace NUMINAMATH_CALUDE_altitude_to_longest_side_l3956_395692

theorem altitude_to_longest_side (a b c h : ℝ) : 
  a = 8 → b = 15 → c = 17 → 
  a^2 + b^2 = c^2 → 
  h * c = 2 * (1/2 * a * b) → 
  h = 120/17 := by
sorry

end NUMINAMATH_CALUDE_altitude_to_longest_side_l3956_395692


namespace NUMINAMATH_CALUDE_base8_perfect_square_c_not_unique_l3956_395683

/-- Represents a number in base 8 of the form ab5c -/
structure Base8Number where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  b_valid : b < 8
  c_valid : c < 8

/-- Converts a Base8Number to its decimal representation -/
def toDecimal (n : Base8Number) : Nat :=
  512 * n.a + 64 * n.b + 40 + n.c

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

theorem base8_perfect_square_c_not_unique :
  ∃ (n1 n2 : Base8Number),
    n1.a = n2.a ∧ n1.b = n2.b ∧ n1.c ≠ n2.c ∧
    isPerfectSquare (toDecimal n1) ∧
    isPerfectSquare (toDecimal n2) := by
  sorry

end NUMINAMATH_CALUDE_base8_perfect_square_c_not_unique_l3956_395683


namespace NUMINAMATH_CALUDE_complex_power_72_l3956_395620

/-- Prove that (cos 215° + i sin 215°)^72 = 1 -/
theorem complex_power_72 : (Complex.exp (215 * Real.pi / 180 * Complex.I))^72 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_72_l3956_395620


namespace NUMINAMATH_CALUDE_factors_of_45_proportion_l3956_395643

theorem factors_of_45_proportion :
  ∃ (a b c d : ℕ), 
    (a ∣ 45) ∧ (b ∣ 45) ∧ (c ∣ 45) ∧ (d ∣ 45) ∧
    (b = 3 * a) ∧ (d = 3 * c) ∧
    (b : ℚ) / a = (d : ℚ) / c ∧ (b : ℚ) / a = 3 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_45_proportion_l3956_395643


namespace NUMINAMATH_CALUDE_square_sequence_20th_figure_l3956_395676

theorem square_sequence_20th_figure :
  let square_count : ℕ → ℕ := λ n => 2 * n^2 - 2 * n + 1
  (square_count 1 = 1) ∧
  (square_count 2 = 5) ∧
  (square_count 3 = 13) ∧
  (square_count 4 = 25) →
  square_count 20 = 761 :=
by
  sorry

end NUMINAMATH_CALUDE_square_sequence_20th_figure_l3956_395676


namespace NUMINAMATH_CALUDE_digits_of_s_1000_l3956_395625

/-- s(n) is an n-digit number formed by attaching the first n perfect squares in order -/
def s (n : ℕ) : ℕ := sorry

/-- The number of digits in a natural number -/
def num_digits (m : ℕ) : ℕ := sorry

/-- Theorem: The number of digits in s(1000) is 2893 -/
theorem digits_of_s_1000 : num_digits (s 1000) = 2893 := by sorry

end NUMINAMATH_CALUDE_digits_of_s_1000_l3956_395625


namespace NUMINAMATH_CALUDE_unique_solution_exists_l3956_395623

theorem unique_solution_exists : 
  ∃! (x y z : ℝ), x + y = 3 ∧ x * y - z^3 = 0 ∧ x = 1.5 ∧ y = 1.5 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l3956_395623


namespace NUMINAMATH_CALUDE_no_valid_super_sudoku_l3956_395642

-- Define the type for grid entries (1 to 9)
def GridEntry := Fin 9

-- Define the type for the 9x9 grid
def SuperSudokuGrid := Matrix (Fin 9) (Fin 9) GridEntry

-- Define the property that each row contains each number 1-9 exactly once
def validRow (grid : SuperSudokuGrid) (row : Fin 9) : Prop :=
  ∀ n : GridEntry, ∃! col : Fin 9, grid row col = n

-- Define the property that each column contains each number 1-9 exactly once
def validColumn (grid : SuperSudokuGrid) (col : Fin 9) : Prop :=
  ∀ n : GridEntry, ∃! row : Fin 9, grid row col = n

-- Define the property that each 3x3 subsquare contains each number 1-9 exactly once
def validSubsquare (grid : SuperSudokuGrid) (startRow startCol : Fin 3) : Prop :=
  ∀ n : GridEntry, ∃! i j : Fin 3, grid (3 * startRow + i) (3 * startCol + j) = n

-- Define a valid super-sudoku grid
def validSuperSudoku (grid : SuperSudokuGrid) : Prop :=
  (∀ row : Fin 9, validRow grid row) ∧
  (∀ col : Fin 9, validColumn grid col) ∧
  (∀ startRow startCol : Fin 3, validSubsquare grid startRow startCol)

-- Theorem: There are no valid super-sudoku grids
theorem no_valid_super_sudoku : ¬∃ grid : SuperSudokuGrid, validSuperSudoku grid := by
  sorry

end NUMINAMATH_CALUDE_no_valid_super_sudoku_l3956_395642


namespace NUMINAMATH_CALUDE_subset_complement_iff_m_range_l3956_395696

open Set Real

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 28 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | 2*x^2 - (5+m)*x + 5 ≤ 0}

-- State the theorem
theorem subset_complement_iff_m_range (m : ℝ) :
  B m ⊆ (univ \ A) ↔ m < -5 - 2*Real.sqrt 10 ∨ m > -5 + 2*Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_subset_complement_iff_m_range_l3956_395696


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l3956_395655

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 * a 3 - 34 * a 3 + 64 = 0 →
  a 5 * a 5 - 34 * a 5 + 64 = 0 →
  (a 4 = 8 ∨ a 4 = -8) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_l3956_395655
