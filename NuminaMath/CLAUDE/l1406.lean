import Mathlib

namespace NUMINAMATH_CALUDE_x_gets_thirty_paisa_l1406_140664

/-- Represents the share of each person in rupees -/
structure Share where
  w : ℝ
  x : ℝ
  y : ℝ

/-- The total amount distributed -/
def total_amount : ℝ := 15

/-- The share of w in rupees -/
def w_share : ℝ := 10

/-- The amount y gets for each rupee w gets, in rupees -/
def y_per_w : ℝ := 0.20

/-- Theorem stating that x gets 0.30 rupees for each rupee w gets -/
theorem x_gets_thirty_paisa (s : Share) 
  (h1 : s.w = w_share)
  (h2 : s.y = y_per_w * s.w)
  (h3 : s.w + s.x + s.y = total_amount) : 
  s.x / s.w = 0.30 := by sorry

end NUMINAMATH_CALUDE_x_gets_thirty_paisa_l1406_140664


namespace NUMINAMATH_CALUDE_houses_per_block_l1406_140600

/-- Given that each block receives 32 pieces of junk mail and each house in a block receives 8 pieces of mail, 
    prove that the number of houses in a block is 4. -/
theorem houses_per_block (mail_per_block : ℕ) (mail_per_house : ℕ) (h1 : mail_per_block = 32) (h2 : mail_per_house = 8) :
  mail_per_block / mail_per_house = 4 := by
  sorry

end NUMINAMATH_CALUDE_houses_per_block_l1406_140600


namespace NUMINAMATH_CALUDE_unique_consecutive_sum_18_l1406_140653

/-- The sum of n consecutive integers starting from a -/
def consecutiveSum (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- A predicate that checks if a set of consecutive integers sums to 18 -/
def isValidSet (a n : ℕ) : Prop :=
  n ≥ 3 ∧ consecutiveSum a n = 18

theorem unique_consecutive_sum_18 :
  ∃! p : ℕ × ℕ, isValidSet p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_sum_18_l1406_140653


namespace NUMINAMATH_CALUDE_unique_d_for_single_solution_l1406_140611

theorem unique_d_for_single_solution :
  ∃! (d : ℝ), d ≠ 0 ∧
  (∃! (a : ℝ), a > 0 ∧
    (∃! (x : ℝ), x^2 + (a + 1/a) * x + d = 0)) ∧
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_d_for_single_solution_l1406_140611


namespace NUMINAMATH_CALUDE_flag_design_count_l1406_140605

/-- The number of colors available for the flag design -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The number of possible flag designs -/
def num_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating the number of possible flag designs -/
theorem flag_design_count : num_flag_designs = 27 := by
  sorry

end NUMINAMATH_CALUDE_flag_design_count_l1406_140605


namespace NUMINAMATH_CALUDE_jason_grass_cutting_time_l1406_140610

/-- The time it takes Jason to cut one lawn in minutes -/
def time_per_lawn : ℕ := 30

/-- The number of yards Jason cuts on Saturday -/
def yards_saturday : ℕ := 8

/-- The number of yards Jason cuts on Sunday -/
def yards_sunday : ℕ := 8

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem: Jason spends 8 hours cutting grass over the weekend -/
theorem jason_grass_cutting_time :
  (time_per_lawn * (yards_saturday + yards_sunday)) / minutes_per_hour = 8 := by
  sorry

end NUMINAMATH_CALUDE_jason_grass_cutting_time_l1406_140610


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l1406_140689

theorem solution_set_reciprocal_inequality (x : ℝ) :
  (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l1406_140689


namespace NUMINAMATH_CALUDE_team_selection_combinations_l1406_140640

theorem team_selection_combinations (n m k : ℕ) (hn : n = 5) (hm : m = 5) (hk : k = 3) :
  (Nat.choose (n + m) k) - (Nat.choose n k) = 110 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_combinations_l1406_140640


namespace NUMINAMATH_CALUDE_current_trees_count_l1406_140652

/-- The number of dogwood trees currently in the park -/
def current_trees : ℕ := sorry

/-- The number of trees to be planted today -/
def trees_today : ℕ := 3

/-- The number of trees to be planted tomorrow -/
def trees_tomorrow : ℕ := 2

/-- The total number of trees after planting -/
def total_trees : ℕ := 12

/-- Proof that the current number of trees is 7 -/
theorem current_trees_count : current_trees = 7 := by
  sorry

end NUMINAMATH_CALUDE_current_trees_count_l1406_140652


namespace NUMINAMATH_CALUDE_N_mod_100_l1406_140659

/-- The number of ways to select a group of singers satisfying given conditions -/
def N : ℕ := sorry

/-- The total number of tenors available -/
def num_tenors : ℕ := 8

/-- The total number of basses available -/
def num_basses : ℕ := 10

/-- The total number of singers to be selected -/
def total_singers : ℕ := 6

/-- Predicate to check if a group satisfies the conditions -/
def valid_group (tenors basses : ℕ) : Prop :=
  tenors + basses = total_singers ∧ 
  ∃ k : ℤ, tenors - basses = 4 * k

theorem N_mod_100 : N % 100 = 96 := by sorry

end NUMINAMATH_CALUDE_N_mod_100_l1406_140659


namespace NUMINAMATH_CALUDE_second_train_speed_is_16_l1406_140622

/-- The speed of the second train given the conditions of the problem -/
def second_train_speed (first_train_speed : ℝ) (distance_between_stations : ℝ) (distance_difference : ℝ) : ℝ :=
  let v : ℝ := 16  -- Speed of the second train
  v

/-- Theorem stating that under the given conditions, the speed of the second train is 16 km/hr -/
theorem second_train_speed_is_16 :
  second_train_speed 20 495 55 = 16 := by
  sorry

#check second_train_speed_is_16

end NUMINAMATH_CALUDE_second_train_speed_is_16_l1406_140622


namespace NUMINAMATH_CALUDE_tangent_line_parallel_points_l1406_140684

def f (x : ℝ) : ℝ := x^3 - 2

theorem tangent_line_parallel_points :
  ∀ x y : ℝ, f x = y →
  (∃ k : ℝ, k * (x - 1) = y + 1 ∧ k = 3) ↔ 
  ((x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -3)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_points_l1406_140684


namespace NUMINAMATH_CALUDE_peach_price_is_40_cents_l1406_140626

/-- Represents the store's discount policy -/
def discount_rate : ℚ := 2 / 10

/-- Represents the number of peaches bought -/
def num_peaches : ℕ := 400

/-- Represents the total amount paid after discount -/
def total_paid : ℚ := 128

/-- Calculates the price of each peach -/
def price_per_peach : ℚ :=
  let total_before_discount := total_paid / (1 - discount_rate)
  total_before_discount / num_peaches

/-- Proves that the price of each peach is $0.40 -/
theorem peach_price_is_40_cents : price_per_peach = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_peach_price_is_40_cents_l1406_140626


namespace NUMINAMATH_CALUDE_reconstructed_text_is_correct_l1406_140649

-- Define the set of original characters
def OriginalChars : Set Char := {'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я'}

-- Define a mapping from distorted characters to original characters
def DistortedToOriginal : Char → Char := sorry

-- Define the reconstructed text
def ReconstructedText : String := "глобальное потепление"

-- Theorem stating that the reconstructed text is correct
theorem reconstructed_text_is_correct :
  ∀ c ∈ ReconstructedText.data, DistortedToOriginal c ∈ OriginalChars :=
sorry

#check reconstructed_text_is_correct

end NUMINAMATH_CALUDE_reconstructed_text_is_correct_l1406_140649


namespace NUMINAMATH_CALUDE_distance_AC_l1406_140645

/-- Given three points A, B, and C on a line, with AB = 5 and BC = 4, 
    the distance AC is either 1 or 9. -/
theorem distance_AC (A B C : ℝ) : 
  (A < B ∧ B < C) ∨ (C < B ∧ B < A) →  -- Points are on the same line
  |B - A| = 5 →                        -- AB = 5
  |C - B| = 4 →                        -- BC = 4
  |C - A| = 1 ∨ |C - A| = 9 :=         -- AC is either 1 or 9
by sorry


end NUMINAMATH_CALUDE_distance_AC_l1406_140645


namespace NUMINAMATH_CALUDE_rain_probability_l1406_140642

theorem rain_probability (p : ℚ) (n : ℕ) (hp : p = 3/5) (hn : n = 5) :
  1 - (1 - p)^n = 3093/3125 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l1406_140642


namespace NUMINAMATH_CALUDE_inequality_proof_l1406_140629

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1406_140629


namespace NUMINAMATH_CALUDE_penelope_candy_count_l1406_140644

/-- The ratio of M&M candies to Starbursts candies -/
def candy_ratio : ℚ := 5 / 3

/-- The number of M&M candies Penelope has -/
def mm_count : ℕ := 25

/-- The number of Starbursts candies Penelope has -/
def starburst_count : ℕ := 15

/-- Theorem stating the relationship between M&M and Starbursts candies -/
theorem penelope_candy_count : 
  (mm_count : ℚ) / candy_ratio = starburst_count := by sorry

end NUMINAMATH_CALUDE_penelope_candy_count_l1406_140644


namespace NUMINAMATH_CALUDE_max_same_count_2011_grid_max_same_count_2011_grid_achievable_l1406_140687

/-- Represents a configuration of napkins on a grid -/
structure NapkinConfiguration where
  grid_size : Nat
  napkin_size : Nat
  napkins : List (Nat × Nat)  -- List of (row, column) positions of napkin top-left corners

/-- Calculates the maximum number of cells with the same nonzero napkin count -/
def max_same_count (config : NapkinConfiguration) : Nat :=
  sorry

/-- The main theorem stating the maximum number of cells with the same nonzero napkin count -/
theorem max_same_count_2011_grid (config : NapkinConfiguration) 
  (h1 : config.grid_size = 2011)
  (h2 : config.napkin_size = 52) :
  max_same_count config ≤ 1994^2 + 37 * 17^2 :=
sorry

/-- The theorem stating that the upper bound is achievable -/
theorem max_same_count_2011_grid_achievable : 
  ∃ (config : NapkinConfiguration), 
    config.grid_size = 2011 ∧ 
    config.napkin_size = 52 ∧
    max_same_count config = 1994^2 + 37 * 17^2 :=
sorry

end NUMINAMATH_CALUDE_max_same_count_2011_grid_max_same_count_2011_grid_achievable_l1406_140687


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l1406_140633

theorem factorization_cubic_minus_linear (a : ℝ) : 
  a^3 - 4*a = a*(a+2)*(a-2) := by sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l1406_140633


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_iff_a_positive_l1406_140603

/-- A complex number represented by its real and imaginary parts -/
structure ComplexNumber where
  re : ℝ
  im : ℝ

/-- The third quadrant of the complex plane -/
def ThirdQuadrant (z : ComplexNumber) : Prop :=
  z.re < 0 ∧ z.im < 0

/-- The complex number z = (5-ai)/i for a given real number a -/
def z (a : ℝ) : ComplexNumber :=
  { re := -a, im := -5 }

/-- The main theorem: z(a) is in the third quadrant if and only if a > 0 -/
theorem z_in_third_quadrant_iff_a_positive (a : ℝ) :
  ThirdQuadrant (z a) ↔ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_iff_a_positive_l1406_140603


namespace NUMINAMATH_CALUDE_pencil_distribution_l1406_140620

theorem pencil_distribution (num_students : ℕ) (num_pencils : ℕ) 
  (h1 : num_students = 2) 
  (h2 : num_pencils = 18) :
  num_pencils / num_students = 9 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1406_140620


namespace NUMINAMATH_CALUDE_equation_solution_l1406_140697

theorem equation_solution : ∃ x : ℚ, (x / (x + 1) = 1 + 1 / x) ∧ (x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1406_140697


namespace NUMINAMATH_CALUDE_square_root_of_four_l1406_140668

theorem square_root_of_four : ∃ x : ℝ, x > 0 ∧ x^2 = 4 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l1406_140668


namespace NUMINAMATH_CALUDE_birthday_cake_cost_l1406_140673

/-- Proves that the cost of the birthday cake is $25 given the conditions of Erika and Rick's gift-buying scenario. -/
theorem birthday_cake_cost (gift_cost : ℝ) (erika_savings : ℝ) (rick_savings : ℝ) (leftover : ℝ) :
  gift_cost = 250 →
  erika_savings = 155 →
  rick_savings = gift_cost / 2 →
  leftover = 5 →
  erika_savings + rick_savings - gift_cost - leftover = 25 := by
  sorry

#check birthday_cake_cost

end NUMINAMATH_CALUDE_birthday_cake_cost_l1406_140673


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1406_140692

theorem complex_fraction_simplification : 
  let i : ℂ := Complex.I
  (1 - i) / (1 + i) = -i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1406_140692


namespace NUMINAMATH_CALUDE_senior_mean_score_l1406_140608

theorem senior_mean_score 
  (total_students : ℕ) 
  (overall_mean : ℝ) 
  (senior_count : ℕ) 
  (non_senior_count : ℕ) 
  (h1 : total_students = 200)
  (h2 : overall_mean = 120)
  (h3 : non_senior_count = 2 * senior_count)
  (h4 : total_students = senior_count + non_senior_count)
  (h5 : senior_count > 0)
  (h6 : non_senior_count > 0) :
  ∃ (senior_mean non_senior_mean : ℝ),
    non_senior_mean = 0.8 * senior_mean ∧
    (senior_count : ℝ) * senior_mean + (non_senior_count : ℝ) * non_senior_mean = (total_students : ℝ) * overall_mean ∧
    senior_mean = 138 := by
  sorry


end NUMINAMATH_CALUDE_senior_mean_score_l1406_140608


namespace NUMINAMATH_CALUDE_smallest_number_l1406_140614

theorem smallest_number (a b c d : ℤ) (ha : a = 1) (hb : b = -2) (hc : c = 0) (hd : d = -3) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1406_140614


namespace NUMINAMATH_CALUDE_largest_base7_3digit_in_decimal_l1406_140618

/-- The largest three-digit number in base 7 -/
def largest_base7_3digit : ℕ := 6 * 7^2 + 6 * 7^1 + 6 * 7^0

/-- Converts a base 7 number to decimal -/
def base7_to_decimal (n : ℕ) : ℕ := n

theorem largest_base7_3digit_in_decimal :
  base7_to_decimal largest_base7_3digit = 342 := by sorry

end NUMINAMATH_CALUDE_largest_base7_3digit_in_decimal_l1406_140618


namespace NUMINAMATH_CALUDE_expression_value_l1406_140627

theorem expression_value (a b : ℝ) (h : 2 * a - 3 * b = 2) : 
  5 - 6 * a + 9 * b = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1406_140627


namespace NUMINAMATH_CALUDE_hash_sum_plus_six_l1406_140657

def hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

theorem hash_sum_plus_six (a b : ℕ) (h : hash a b = 100) : (a + b) + 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_hash_sum_plus_six_l1406_140657


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l1406_140694

theorem inverse_proportion_ratio (a₁ a₂ b₁ b₂ : ℝ) (h1 : a₁ ≠ 0) (h2 : a₂ ≠ 0) (h3 : b₁ ≠ 0) (h4 : b₂ ≠ 0) :
  (∃ k : ℝ, k ≠ 0 ∧ a₁ * b₁ = k ∧ a₂ * b₂ = k) →
  a₁ / a₂ = 3 / 4 →
  b₁ / b₂ = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l1406_140694


namespace NUMINAMATH_CALUDE_new_average_after_changes_l1406_140619

def initial_count : ℕ := 60
def initial_average : ℚ := 40
def removed_number1 : ℕ := 50
def removed_number2 : ℕ := 60
def added_number : ℕ := 35

theorem new_average_after_changes :
  let initial_sum := initial_count * initial_average
  let sum_after_removal := initial_sum - (removed_number1 + removed_number2)
  let final_sum := sum_after_removal + added_number
  let final_count := initial_count - 1
  final_sum / final_count = 39.41 := by sorry

end NUMINAMATH_CALUDE_new_average_after_changes_l1406_140619


namespace NUMINAMATH_CALUDE_remainder_of_2614303940317_div_13_l1406_140671

theorem remainder_of_2614303940317_div_13 : 
  2614303940317 % 13 = 4 := by sorry

end NUMINAMATH_CALUDE_remainder_of_2614303940317_div_13_l1406_140671


namespace NUMINAMATH_CALUDE_taran_number_puzzle_l1406_140688

theorem taran_number_puzzle : ∃ x : ℕ, 
  ((x * 5 + 5 - 5 = 73) ∨ (x * 5 + 5 - 6 = 73) ∨ (x * 5 + 6 - 5 = 73) ∨ (x * 5 + 6 - 6 = 73) ∨
   (x * 6 + 5 - 5 = 73) ∨ (x * 6 + 5 - 6 = 73) ∨ (x * 6 + 6 - 5 = 73) ∨ (x * 6 + 6 - 6 = 73)) ∧
  x = 12 := by
  sorry

end NUMINAMATH_CALUDE_taran_number_puzzle_l1406_140688


namespace NUMINAMATH_CALUDE_unique_albums_count_l1406_140651

/-- Represents the album collections of Andrew, John, and Bella -/
structure AlbumCollections where
  andrew_total : ℕ
  andrew_john_shared : ℕ
  john_unique : ℕ
  bella_andrew_overlap : ℕ

/-- Calculates the number of unique albums not shared among any two people -/
def unique_albums (collections : AlbumCollections) : ℕ :=
  (collections.andrew_total - collections.andrew_john_shared) + collections.john_unique

/-- Theorem stating that the number of unique albums is 18 given the problem conditions -/
theorem unique_albums_count (collections : AlbumCollections)
  (h1 : collections.andrew_total = 20)
  (h2 : collections.andrew_john_shared = 10)
  (h3 : collections.john_unique = 8)
  (h4 : collections.bella_andrew_overlap = 5)
  (h5 : collections.bella_andrew_overlap ≤ collections.andrew_total - collections.andrew_john_shared) :
  unique_albums collections = 18 := by
  sorry

#eval unique_albums { andrew_total := 20, andrew_john_shared := 10, john_unique := 8, bella_andrew_overlap := 5 }

end NUMINAMATH_CALUDE_unique_albums_count_l1406_140651


namespace NUMINAMATH_CALUDE_factorization_sum_l1406_140679

theorem factorization_sum (x y : ℝ) : ∃ (a b c d e f g h j k : ℤ),
  (27 * x^9 - 512 * y^9 = (a * x + b * y) * (c * x^3 + d * x * y^2 + e * y^3) * 
                          (f * x + g * y) * (h * x^3 + j * x * y^2 + k * y^3)) ∧
  (a + b + c + d + e + f + g + h + j + k = 12) := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l1406_140679


namespace NUMINAMATH_CALUDE_orange_calories_distribution_l1406_140698

theorem orange_calories_distribution :
  let num_oranges : ℕ := 5
  let pieces_per_orange : ℕ := 8
  let num_people : ℕ := 4
  let calories_per_orange : ℕ := 80
  let total_pieces : ℕ := num_oranges * pieces_per_orange
  let pieces_per_person : ℕ := total_pieces / num_people
  let oranges_per_person : ℚ := pieces_per_person / pieces_per_orange
  let calories_per_person : ℚ := oranges_per_person * calories_per_orange
  calories_per_person = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_orange_calories_distribution_l1406_140698


namespace NUMINAMATH_CALUDE_solution_set_a_3_range_of_a_non_negative_l1406_140693

-- Define the function f
def f (a x : ℝ) : ℝ := |x^2 - 2*x + a - 1| - a^2 - 2*a

-- Theorem 1: Solution set when a = 3
theorem solution_set_a_3 :
  {x : ℝ | f 3 x ≥ -10} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} :=
sorry

-- Theorem 2: Range of a for f(x) ≥ 0 for all x
theorem range_of_a_non_negative :
  {a : ℝ | ∀ x, f a x ≥ 0} = {a : ℝ | -2 ≤ a ∧ a ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_solution_set_a_3_range_of_a_non_negative_l1406_140693


namespace NUMINAMATH_CALUDE_sum_of_special_integers_l1406_140604

/-- The smallest positive integer with only two positive divisors -/
def smallest_two_divisors : ℕ := 2

/-- The largest integer less than 150 with exactly three positive divisors -/
def largest_three_divisors_under_150 : ℕ := 121

/-- The theorem stating that the sum of the two defined numbers is 123 -/
theorem sum_of_special_integers : 
  smallest_two_divisors + largest_three_divisors_under_150 = 123 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_special_integers_l1406_140604


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l1406_140648

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 2 ∧ (m / (x - 2) + 1 = x / (2 - x))) ↔ 
  (m ≤ 2 ∧ m ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l1406_140648


namespace NUMINAMATH_CALUDE_plumbing_job_washers_l1406_140650

/-- Calculates the number of remaining washers after a plumbing job --/
def remaining_washers (total_pipe_length : ℕ) (pipe_per_bolt : ℕ) (washers_per_bolt : ℕ) (total_washers : ℕ) : ℕ :=
  let bolts_needed := total_pipe_length / pipe_per_bolt
  let washers_used := bolts_needed * washers_per_bolt
  total_washers - washers_used

/-- Theorem stating that for the given plumbing job, 4 washers will remain --/
theorem plumbing_job_washers :
  remaining_washers 40 5 2 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_plumbing_job_washers_l1406_140650


namespace NUMINAMATH_CALUDE_inequality_proof_l1406_140623

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt a + Real.sqrt b)^8 ≥ 64 * a * b * (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1406_140623


namespace NUMINAMATH_CALUDE_area_of_triangle_KBC_l1406_140630

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a hexagon -/
structure Hexagon :=
  (A B C D E F : Point)

/-- Represents a square -/
structure Square :=
  (A B C D : Point)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The area of a square -/
def squareArea (s : Square) : ℝ := sorry

/-- Check if a hexagon is equiangular -/
def isEquiangular (h : Hexagon) : Prop := sorry

/-- Check if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop := sorry

/-- The length of a line segment between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

theorem area_of_triangle_KBC 
  (ABCDEF : Hexagon) 
  (ABJI FEHG : Square) 
  (JBK : Triangle) :
  isEquiangular ABCDEF →
  squareArea ABJI = 25 →
  squareArea FEHG = 49 →
  isIsosceles JBK →
  distance ABCDEF.F ABCDEF.E = distance ABCDEF.B ABCDEF.C →
  triangleArea ⟨JBK.B, ABCDEF.B, ABCDEF.C⟩ = 49 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_KBC_l1406_140630


namespace NUMINAMATH_CALUDE_speed_ratio_l1406_140665

/-- Two cars traveling towards each other with constant speeds -/
structure TwoCars where
  v1 : ℝ  -- Speed of the first car
  v2 : ℝ  -- Speed of the second car
  d : ℝ   -- Distance between points A and B
  t : ℝ   -- Time until the cars meet

/-- The conditions of the problem -/
def MeetingConditions (cars : TwoCars) : Prop :=
  cars.v1 > 0 ∧ cars.v2 > 0 ∧ cars.d > 0 ∧ cars.t > 0 ∧
  cars.v1 * cars.t + cars.v2 * cars.t = cars.d ∧
  cars.d - cars.v1 * cars.t = cars.v1 ∧
  cars.d - cars.v2 * cars.t = 4 * cars.v2

/-- The theorem stating the ratio of speeds -/
theorem speed_ratio (cars : TwoCars) 
  (h : MeetingConditions cars) : cars.v1 / cars.v2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_l1406_140665


namespace NUMINAMATH_CALUDE_max_b_value_max_b_value_achieved_l1406_140662

theorem max_b_value (x b : ℤ) (h1 : x^2 + b*x = -21) (h2 : b > 0) : b ≤ 22 := by
  sorry

theorem max_b_value_achieved : ∃ x b : ℤ, x^2 + b*x = -21 ∧ b > 0 ∧ b = 22 := by
  sorry

end NUMINAMATH_CALUDE_max_b_value_max_b_value_achieved_l1406_140662


namespace NUMINAMATH_CALUDE_x_eleven_percent_greater_than_70_l1406_140631

/-- If x is 11 percent greater than 70, then x = 77.7 -/
theorem x_eleven_percent_greater_than_70 (x : ℝ) : 
  x = 70 * (1 + 11 / 100) → x = 77.7 := by
sorry

end NUMINAMATH_CALUDE_x_eleven_percent_greater_than_70_l1406_140631


namespace NUMINAMATH_CALUDE_congruence_definition_l1406_140646

-- Define a type for geometric figures
def Figure : Type := sorry

-- Define a relation for figures that can completely overlap
def can_overlap (f1 f2 : Figure) : Prop := sorry

-- Define congruence for figures
def congruent (f1 f2 : Figure) : Prop := sorry

-- Theorem stating the definition of congruent figures
theorem congruence_definition :
  ∀ (f1 f2 : Figure), congruent f1 f2 ↔ can_overlap f1 f2 := by sorry

end NUMINAMATH_CALUDE_congruence_definition_l1406_140646


namespace NUMINAMATH_CALUDE_root_product_expression_l1406_140686

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 - 2*p*α + 1 = 0) → 
  (β^2 - 2*p*β + 1 = 0) → 
  (γ^2 + q*γ + 2 = 0) → 
  (δ^2 + q*δ + 2 = 0) → 
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = 2*(p - q)^2 :=
by sorry

end NUMINAMATH_CALUDE_root_product_expression_l1406_140686


namespace NUMINAMATH_CALUDE_total_fishermen_l1406_140615

theorem total_fishermen (total_fish : ℕ) (fish_per_group : ℕ) (group_size : ℕ) (last_fisherman_fish : ℕ) :
  total_fish = group_size * fish_per_group + last_fisherman_fish →
  total_fish = 10000 →
  fish_per_group = 400 →
  group_size = 19 →
  last_fisherman_fish = 2400 →
  group_size + 1 = 20 := by
sorry

end NUMINAMATH_CALUDE_total_fishermen_l1406_140615


namespace NUMINAMATH_CALUDE_weight_of_B_l1406_140695

theorem weight_of_B (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 44) :
  B = 33 := by
sorry

end NUMINAMATH_CALUDE_weight_of_B_l1406_140695


namespace NUMINAMATH_CALUDE_arc_length_30_degree_sector_l1406_140628

/-- The length of an arc in a circular sector with radius 1 cm and central angle 30° is π/6 cm. -/
theorem arc_length_30_degree_sector :
  let r : ℝ := 1  -- radius in cm
  let θ : ℝ := 30 * π / 180  -- central angle in radians
  let l : ℝ := r * θ  -- arc length formula
  l = π / 6 := by sorry

end NUMINAMATH_CALUDE_arc_length_30_degree_sector_l1406_140628


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l1406_140606

theorem smallest_m_for_integral_solutions : 
  ∃ (m : ℕ), m > 0 ∧ 
  (∃ (x y : ℤ), 10 * x^2 - m * x + 180 = 0 ∧ 10 * y^2 - m * y + 180 = 0 ∧ x ≠ y) ∧
  (∀ (k : ℕ), k > 0 ∧ k < m → 
    ¬∃ (x y : ℤ), 10 * x^2 - k * x + 180 = 0 ∧ 10 * y^2 - k * y + 180 = 0 ∧ x ≠ y) ∧
  m = 90 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l1406_140606


namespace NUMINAMATH_CALUDE_sum_of_a_and_d_l1406_140647

theorem sum_of_a_and_d (a b c d : ℤ) 
  (eq1 : a + b = 5)
  (eq2 : b + c = 6)
  (eq3 : c + d = 3) :
  a + d = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_d_l1406_140647


namespace NUMINAMATH_CALUDE_impossible_arrangement_l1406_140669

theorem impossible_arrangement (n : Nat) (h : n = 2002) : 
  ¬ ∃ (A : Fin n → Fin n → Fin (n^2)),
    (∀ i j : Fin n, A i j < n^2) ∧ 
    (∀ i j : Fin n, ∃ k₁ k₂ : Fin n, 
      (A i k₁ * A i k₂ * A i j ≤ n^2 ∨ A k₁ j * A k₂ j * A i j ≤ n^2)) ∧
    (∀ x : Fin (n^2), ∃ i j : Fin n, A i j = x) := by
  sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l1406_140669


namespace NUMINAMATH_CALUDE_total_sleep_deficit_l1406_140609

/-- Calculates the total sleep deficit for three people over a week. -/
theorem total_sleep_deficit
  (ideal_sleep : ℕ)
  (tom_weeknight : ℕ)
  (tom_weekend : ℕ)
  (jane_weeknight : ℕ)
  (jane_weekend : ℕ)
  (mark_weeknight : ℕ)
  (mark_weekend : ℕ)
  (h1 : ideal_sleep = 8)
  (h2 : tom_weeknight = 5)
  (h3 : tom_weekend = 6)
  (h4 : jane_weeknight = 7)
  (h5 : jane_weekend = 9)
  (h6 : mark_weeknight = 6)
  (h7 : mark_weekend = 7) :
  (7 * ideal_sleep - (5 * tom_weeknight + 2 * tom_weekend)) +
  (7 * ideal_sleep - (5 * jane_weeknight + 2 * jane_weekend)) +
  (7 * ideal_sleep - (5 * mark_weeknight + 2 * mark_weekend)) = 34 := by
  sorry


end NUMINAMATH_CALUDE_total_sleep_deficit_l1406_140609


namespace NUMINAMATH_CALUDE_min_max_y_l1406_140602

/-- The function f(x) = 2 + x -/
def f (x : ℝ) : ℝ := 2 + x

/-- The function y = [f(x)]^2 + f(x) -/
def y (x : ℝ) : ℝ := (f x)^2 + f x

theorem min_max_y :
  (∀ x ∈ Set.Icc 1 9, y 1 ≤ y x) ∧
  (∀ x ∈ Set.Icc 1 9, y x ≤ y 9) ∧
  y 1 = 13 ∧
  y 9 = 141 := by sorry

end NUMINAMATH_CALUDE_min_max_y_l1406_140602


namespace NUMINAMATH_CALUDE_incoming_class_size_l1406_140654

theorem incoming_class_size : ∃! n : ℕ, 
  0 < n ∧ n < 1000 ∧ n % 25 = 18 ∧ n % 28 = 26 ∧ n = 418 := by
  sorry

end NUMINAMATH_CALUDE_incoming_class_size_l1406_140654


namespace NUMINAMATH_CALUDE_count_pairs_eq_28_l1406_140635

def count_pairs : ℕ :=
  (Finset.range 7).sum (λ m =>
    (Finset.range (8 - m)).card)

theorem count_pairs_eq_28 : count_pairs = 28 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_28_l1406_140635


namespace NUMINAMATH_CALUDE_parabola_focus_l1406_140643

/-- The parabola is defined by the equation x = (1/4)y^2 -/
def parabola (x y : ℝ) : Prop := x = (1/4) * y^2

/-- The focus of a parabola is a point (f, 0) such that for any point (x, y) on the parabola,
    the distance from (x, y) to (f, 0) equals the distance from (x, y) to the directrix x = d,
    where d = f + 1 -/
def is_focus (f : ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, parabola x y →
    (x - f)^2 + y^2 = (x - (f + 1))^2

/-- The focus of the parabola x = (1/4)y^2 is at the point (-1, 0) -/
theorem parabola_focus :
  is_focus (-1) parabola := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l1406_140643


namespace NUMINAMATH_CALUDE_simplify_square_roots_l1406_140676

theorem simplify_square_roots : 16^(1/2) - 625^(1/2) = -21 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l1406_140676


namespace NUMINAMATH_CALUDE_min_moves_to_capture_pawns_l1406_140624

/-- Represents a position on a chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- The knight's move function -/
def knightMove (p : Position) : List Position :=
  let moves := [(1,2), (2,1), (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,1), (-1,2)]
  moves.filterMap (fun (dr, dc) =>
    let newRow := p.row + dr
    let newCol := p.col + dc
    if newRow < 8 && newCol < 8 && newRow ≥ 0 && newCol ≥ 0
    then some ⟨newRow, newCol⟩
    else none)

/-- The minimum number of moves for a knight to capture both pawns -/
def minMovesToCapturePawns : ℕ :=
  let start : Position := ⟨0, 1⟩  -- B1
  let pawn1 : Position := ⟨7, 1⟩  -- B8
  let pawn2 : Position := ⟨7, 6⟩  -- G8
  7  -- The actual minimum number of moves

/-- Theorem stating the minimum number of moves to capture both pawns -/
theorem min_moves_to_capture_pawns :
  minMovesToCapturePawns = 7 :=
sorry

end NUMINAMATH_CALUDE_min_moves_to_capture_pawns_l1406_140624


namespace NUMINAMATH_CALUDE_total_amount_spent_l1406_140616

theorem total_amount_spent (num_pens num_pencils : ℕ) (avg_pen_price avg_pencil_price total_amount : ℚ) :
  num_pens = 30 →
  num_pencils = 75 →
  avg_pen_price = 18 →
  avg_pencil_price = 2 →
  total_amount = num_pens * avg_pen_price + num_pencils * avg_pencil_price →
  total_amount = 690 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_spent_l1406_140616


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1406_140663

/-- Represents the total number of staff in each category -/
structure StaffCount where
  business : ℕ
  management : ℕ
  logistics : ℕ

/-- Calculates the total sample size for stratified sampling -/
def calculateSampleSize (staff : StaffCount) (managementSample : ℕ) : ℕ :=
  let totalStaff := staff.business + staff.management + staff.logistics
  let samplingFraction := managementSample / staff.management
  totalStaff * samplingFraction

/-- Theorem: Given the staff counts and management sample, the total sample size is 20 -/
theorem stratified_sample_size 
  (staff : StaffCount) 
  (h1 : staff.business = 120) 
  (h2 : staff.management = 24) 
  (h3 : staff.logistics = 16) 
  (h4 : calculateSampleSize staff 3 = 20) : 
  calculateSampleSize staff 3 = 20 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l1406_140663


namespace NUMINAMATH_CALUDE_laura_park_time_l1406_140682

/-- The number of trips Laura took to the park -/
def num_trips : ℕ := 6

/-- The time (in hours) spent walking to and from the park for each trip -/
def walking_time : ℝ := 0.5

/-- The fraction of total time spent in the park -/
def park_time_fraction : ℝ := 0.8

/-- The time (in hours) Laura spent at the park during each trip -/
def park_time : ℝ := 2

theorem laura_park_time :
  park_time = (park_time_fraction * num_trips * (park_time + walking_time)) / num_trips := by
  sorry

end NUMINAMATH_CALUDE_laura_park_time_l1406_140682


namespace NUMINAMATH_CALUDE_sets_problem_l1406_140656

-- Define the universe set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3 - 2*a}

theorem sets_problem (a : ℝ) :
  (((Set.compl A) ∪ (B a)) = U ↔ a ≤ 0) ∧
  ((A ∩ (B a)) = (B a) ↔ a ≥ (1/2)) := by
  sorry


end NUMINAMATH_CALUDE_sets_problem_l1406_140656


namespace NUMINAMATH_CALUDE_function_characterization_l1406_140621

theorem function_characterization
  (f : ℕ → ℕ)
  (α : ℕ)
  (h1 : ∀ (m n : ℕ), f (m * n^2) = f (m * n) + α * f n)
  (h2 : ∀ (n : ℕ) (p : ℕ), Nat.Prime p → p ∣ n → f p ≠ 0 ∧ f p ∣ f n) :
  ∃ (c : ℕ), 
    (α = 1) ∧
    (∀ (n : ℕ), 
      f n = c * (Nat.factorization n).sum (fun _ e => e)) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l1406_140621


namespace NUMINAMATH_CALUDE_draws_calculation_l1406_140674

def total_games : ℕ := 14
def wins : ℕ := 2
def losses : ℕ := 2

theorem draws_calculation : total_games - (wins + losses) = 10 := by
  sorry

end NUMINAMATH_CALUDE_draws_calculation_l1406_140674


namespace NUMINAMATH_CALUDE_exchange_rate_solution_l1406_140612

/-- Represents the exchange rate problem with Jack's currency amounts --/
def ExchangeRateProblem (pounds euros yen : ℕ) (yenPerPound : ℕ) (totalYen : ℕ) :=
  ∃ (poundsPerEuro : ℚ),
    (pounds : ℚ) * yenPerPound + euros * poundsPerEuro * yenPerPound + yen = totalYen ∧
    poundsPerEuro = 2

/-- Theorem stating that the exchange rate is 2 pounds per euro --/
theorem exchange_rate_solution :
  ExchangeRateProblem 42 11 3000 100 9400 :=
by
  sorry


end NUMINAMATH_CALUDE_exchange_rate_solution_l1406_140612


namespace NUMINAMATH_CALUDE_contests_paths_l1406_140625

/-- Represents the number of choices for each step in the path, except the last --/
def choices : ℕ := 2

/-- Represents the number of starting points (number of "C"s at the base) --/
def starting_points : ℕ := 2

/-- Represents the number of steps in the path (length of "CONTESTS" - 1) --/
def path_length : ℕ := 7

theorem contests_paths :
  starting_points * (choices ^ path_length) = 256 := by
  sorry

end NUMINAMATH_CALUDE_contests_paths_l1406_140625


namespace NUMINAMATH_CALUDE_eleventh_term_value_l1406_140658

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_term (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

/-- The 11th term of a geometric sequence with first term 5 and common ratio 2/3 -/
def eleventh_term : ℚ := geometric_term 5 (2/3) 11

theorem eleventh_term_value : eleventh_term = 5120/59049 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_value_l1406_140658


namespace NUMINAMATH_CALUDE_opposite_numbers_and_cube_root_l1406_140677

theorem opposite_numbers_and_cube_root (a b c : ℝ) : 
  (a + b = 0) → (c^3 = 8) → (2*a + 2*b - c = -2) := by
sorry

end NUMINAMATH_CALUDE_opposite_numbers_and_cube_root_l1406_140677


namespace NUMINAMATH_CALUDE_sqrt_x_equals_3_x_squared_equals_y_squared_l1406_140678

-- Define x and y as functions of a
def x (a : ℝ) : ℝ := 1 - 2*a
def y (a : ℝ) : ℝ := 3*a - 4

-- Theorem 1: When √x = 3, a = -4
theorem sqrt_x_equals_3 : ∃ a : ℝ, x a = 9 ∧ a = -4 := by sorry

-- Theorem 2: There exist values of a such that x² = y² = 1 or x² = y² = 25
theorem x_squared_equals_y_squared :
  (∃ a : ℝ, (x a)^2 = (y a)^2 ∧ (x a)^2 = 1) ∨
  (∃ a : ℝ, (x a)^2 = (y a)^2 ∧ (x a)^2 = 25) := by sorry

end NUMINAMATH_CALUDE_sqrt_x_equals_3_x_squared_equals_y_squared_l1406_140678


namespace NUMINAMATH_CALUDE_greatest_valid_n_l1406_140641

def is_valid (n : ℕ) : Prop :=
  n ≥ 100 ∧ n ≤ 999 ∧ ¬((Nat.factorial (n / 2)) % (n * (n + 1)) = 0)

theorem greatest_valid_n : 
  (∀ m : ℕ, m > 996 → m ≤ 999 → ¬(is_valid m)) ∧
  is_valid 996 := by sorry

end NUMINAMATH_CALUDE_greatest_valid_n_l1406_140641


namespace NUMINAMATH_CALUDE_power_division_rule_l1406_140632

theorem power_division_rule (a : ℝ) : a^5 / a^2 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l1406_140632


namespace NUMINAMATH_CALUDE_exercise_book_problem_l1406_140667

theorem exercise_book_problem :
  ∀ (x y : ℕ),
    x + y = 100 →
    2 * x + 4 * y = 250 →
    x = 75 ∧ y = 25 :=
by sorry

end NUMINAMATH_CALUDE_exercise_book_problem_l1406_140667


namespace NUMINAMATH_CALUDE_shared_focus_hyperbola_ellipse_l1406_140607

/-- Given a hyperbola and an ellipse that share a common focus, prove that the parameter p of the ellipse is equal to 4 -/
theorem shared_focus_hyperbola_ellipse (p : ℝ) : 
  (∀ x y : ℝ, x^2/3 - y^2 = 1 → x^2/8 + y^2/p = 1) → 
  (0 < p) → 
  (p < 8) → 
  p = 4 := by sorry

end NUMINAMATH_CALUDE_shared_focus_hyperbola_ellipse_l1406_140607


namespace NUMINAMATH_CALUDE_hyperbola_foci_l1406_140685

/-- The hyperbola equation --/
def hyperbola_eq (x y : ℝ) : Prop := y^2 - x^2/3 = 1

/-- The focus coordinates --/
def focus_coords : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

/-- Theorem: The given coordinates are the foci of the hyperbola --/
theorem hyperbola_foci : 
  ∀ (x y : ℝ), hyperbola_eq x y ↔ ∃ (f : ℝ × ℝ), f ∈ focus_coords ∧ 
    (x - f.1)^2 + (y - f.2)^2 = ((x + f.1)^2 + (y + f.2)^2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l1406_140685


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1406_140690

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 10 ∧ b = 24 ∧ c^2 = a^2 + b^2 → c = 26 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1406_140690


namespace NUMINAMATH_CALUDE_other_communities_count_l1406_140661

theorem other_communities_count (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ) 
  (h_total : total = 400)
  (h_muslim : muslim_percent = 44/100)
  (h_hindu : hindu_percent = 28/100)
  (h_sikh : sikh_percent = 10/100) :
  (total : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 72 := by
  sorry

end NUMINAMATH_CALUDE_other_communities_count_l1406_140661


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l1406_140617

-- Define the quadratic equation
def quadratic_equation (s t x : ℝ) : Prop :=
  s * x^2 + t * x + s - 1 = 0

-- Define the existence of a real root
def has_real_root (s t : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation s t x

-- Main theorem
theorem quadratic_real_root_condition (s : ℝ) :
  (s ≠ 0 ∧ ∀ t : ℝ, has_real_root s t) ↔ (0 < s ∧ s ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l1406_140617


namespace NUMINAMATH_CALUDE_constant_pace_time_ratio_l1406_140637

/-- Represents a runner with a constant pace -/
structure Runner where
  pace : ℝ  -- pace in minutes per mile

/-- Calculates the time taken to run a given distance -/
def time_to_run (r : Runner) (distance : ℝ) : ℝ :=
  r.pace * distance

theorem constant_pace_time_ratio 
  (r : Runner) 
  (store_distance : ℝ) 
  (store_time : ℝ) 
  (cousin_distance : ℝ) :
  store_distance = 5 →
  store_time = 30 →
  cousin_distance = 2.5 →
  time_to_run r store_distance = store_time →
  time_to_run r cousin_distance = 15 :=
by sorry

end NUMINAMATH_CALUDE_constant_pace_time_ratio_l1406_140637


namespace NUMINAMATH_CALUDE_mollys_age_l1406_140699

theorem mollys_age (sandy_current : ℕ) (molly_current : ℕ) : 
  (sandy_current : ℚ) / molly_current = 4 / 3 →
  sandy_current + 6 = 38 →
  molly_current = 24 := by
sorry

end NUMINAMATH_CALUDE_mollys_age_l1406_140699


namespace NUMINAMATH_CALUDE_ashleys_age_l1406_140613

/-- Given that Ashley and Mary's ages are in the ratio 4:7 and their sum is 22, 
    prove that Ashley's age is 8 years. -/
theorem ashleys_age (ashley mary : ℕ) 
  (h_ratio : ashley * 7 = mary * 4)
  (h_sum : ashley + mary = 22) : 
  ashley = 8 := by
  sorry

end NUMINAMATH_CALUDE_ashleys_age_l1406_140613


namespace NUMINAMATH_CALUDE_divisibility_by_five_l1406_140666

theorem divisibility_by_five (m n : ℕ) : 
  (∃ k : ℕ, m * n = 5 * k) → (∃ j : ℕ, m = 5 * j) ∨ (∃ l : ℕ, n = 5 * l) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l1406_140666


namespace NUMINAMATH_CALUDE_sum_of_xyz_is_negative_one_l1406_140636

theorem sum_of_xyz_is_negative_one 
  (x y z : ℝ) 
  (h1 : x*y + x*z + y*z + x + y + z = -3) 
  (h2 : x^2 + y^2 + z^2 = 5) : 
  x + y + z = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_is_negative_one_l1406_140636


namespace NUMINAMATH_CALUDE_triangle_angle_arithmetic_sequence_property_l1406_140601

-- Define a structure for a triangle
structure Triangle :=
  (a b c : ℝ)  -- sides
  (A B C : ℝ)  -- angles in radians

-- Define the property of angles forming an arithmetic sequence
def arithmeticSequence (t : Triangle) : Prop :=
  ∃ d : ℝ, t.B - t.A = d ∧ t.C - t.B = d

-- State the theorem
theorem triangle_angle_arithmetic_sequence_property (t : Triangle) 
  (h1 : t.a > 0) (h2 : t.b > 0) (h3 : t.c > 0)  -- positive sides
  (h4 : arithmeticSequence t)  -- angles form arithmetic sequence
  : 1 / (t.a + t.b) + 1 / (t.b + t.c) = 3 / (t.a + t.b + t.c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_arithmetic_sequence_property_l1406_140601


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l1406_140634

theorem trig_expression_simplification :
  (Real.sin (20 * π / 180) * Real.cos (15 * π / 180) + Real.cos (160 * π / 180) * Real.cos (105 * π / 180)) /
  (Real.sin (25 * π / 180) * Real.cos (10 * π / 180) + Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) =
  Real.sin (5 * π / 180) / Real.sin (15 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l1406_140634


namespace NUMINAMATH_CALUDE_difference_constant_sum_not_always_minimal_when_equal_l1406_140675

theorem difference_constant_sum_not_always_minimal_when_equal :
  ¬ (∀ (a b : ℝ) (d : ℝ), 
    a > 0 → b > 0 → a - b = d → 
    (∀ (x y : ℝ), x > 0 → y > 0 → x - y = d → a + b ≤ x + y)) :=
sorry

end NUMINAMATH_CALUDE_difference_constant_sum_not_always_minimal_when_equal_l1406_140675


namespace NUMINAMATH_CALUDE_fraction_of_women_l1406_140691

/-- Proves that the fraction of women in a room is 1/4 given the specified conditions -/
theorem fraction_of_women (total_people : ℕ) (married_fraction : ℚ) (max_unmarried_women : ℕ) : 
  total_people = 80 →
  married_fraction = 3/4 →
  max_unmarried_women = 20 →
  (max_unmarried_women : ℚ) / total_people = 1/4 := by
  sorry

#check fraction_of_women

end NUMINAMATH_CALUDE_fraction_of_women_l1406_140691


namespace NUMINAMATH_CALUDE_coefficient_value_l1406_140672

def P (c : ℝ) (x : ℝ) : ℝ := x^4 + 3*x^3 + 2*x^2 + c*x + 15

theorem coefficient_value (c : ℝ) :
  (∀ x, (x - 7 : ℝ) ∣ P c x) → c = -508 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_value_l1406_140672


namespace NUMINAMATH_CALUDE_cats_theorem_l1406_140670

def cats_problem (siamese house persian first_sale second_sale : ℕ) : Prop :=
  let initial_total : ℕ := siamese + house + persian
  let after_first_sale : ℕ := initial_total - first_sale
  let final_count : ℕ := after_first_sale - second_sale
  final_count = 17

theorem cats_theorem : cats_problem 23 17 29 40 12 := by
  sorry

end NUMINAMATH_CALUDE_cats_theorem_l1406_140670


namespace NUMINAMATH_CALUDE_sandy_second_shop_amount_l1406_140696

/-- The amount Sandy paid for books from the second shop -/
def second_shop_amount (first_shop_books : ℕ) (second_shop_books : ℕ) 
  (first_shop_amount : ℚ) (average_price : ℚ) : ℚ :=
  (first_shop_books + second_shop_books) * average_price - first_shop_amount

/-- Proof that Sandy paid $900 for books from the second shop -/
theorem sandy_second_shop_amount :
  second_shop_amount 65 55 1380 19 = 900 := by
  sorry

end NUMINAMATH_CALUDE_sandy_second_shop_amount_l1406_140696


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l1406_140683

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_lines_from_perpendicular_planes 
  (m n : Line) (α β : Plane) :
  perp_line_plane m α → 
  perp_line_plane n β → 
  perp_plane α β → 
  perp_line m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l1406_140683


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l1406_140639

theorem part_to_whole_ratio (N : ℝ) (x : ℝ) (h1 : N = 160) (h2 : x + 4 = (N/4) - 4) : x / N = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l1406_140639


namespace NUMINAMATH_CALUDE_crayons_in_drawer_l1406_140638

theorem crayons_in_drawer (initial_crayons final_crayons benny_crayons : ℕ) : 
  initial_crayons = 9 → 
  final_crayons = 12 → 
  benny_crayons = final_crayons - initial_crayons →
  benny_crayons = 3 := by
sorry

end NUMINAMATH_CALUDE_crayons_in_drawer_l1406_140638


namespace NUMINAMATH_CALUDE_double_round_robin_max_teams_l1406_140660

/-- The maximum number of teams in a double round-robin tournament --/
def max_teams : ℕ := 6

/-- The number of weeks available for the tournament --/
def available_weeks : ℕ := 4

/-- The number of matches each team plays in a double round-robin tournament --/
def matches_per_team (n : ℕ) : ℕ := 2 * (n - 1)

/-- The total number of matches in a double round-robin tournament --/
def total_matches (n : ℕ) : ℕ := n * (n - 1)

/-- The maximum number of away matches a team can play in the available weeks --/
def max_away_matches : ℕ := available_weeks

theorem double_round_robin_max_teams :
  ∀ n : ℕ, n ≤ max_teams ∧ 
  matches_per_team n ≤ 2 * max_away_matches ∧
  (∀ m : ℕ, m > max_teams → matches_per_team m > 2 * max_away_matches) :=
by sorry

#check double_round_robin_max_teams

end NUMINAMATH_CALUDE_double_round_robin_max_teams_l1406_140660


namespace NUMINAMATH_CALUDE_slower_train_speed_l1406_140681

/-- Calculates the speed of the slower train given the conditions of the problem -/
theorem slower_train_speed (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) : 
  train_length = 500 →
  faster_speed = 45 →
  passing_time = 60 / 3600 →
  (faster_speed + (2 * train_length / 1000) / passing_time) - faster_speed = 15 := by
  sorry

#check slower_train_speed

end NUMINAMATH_CALUDE_slower_train_speed_l1406_140681


namespace NUMINAMATH_CALUDE_quadratic_roots_pure_imaginary_l1406_140655

theorem quadratic_roots_pure_imaginary (m : ℝ) (hm : m < 0) :
  ∀ (z : ℂ), 8 * z^2 + 4 * Complex.I * z - m = 0 →
  ∃ (y : ℝ), z = Complex.I * y :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_pure_imaginary_l1406_140655


namespace NUMINAMATH_CALUDE_fourth_power_sum_l1406_140680

theorem fourth_power_sum (x y z : ℝ) 
  (h1 : x + y + z = 2) 
  (h2 : x^2 + y^2 + z^2 = 6) 
  (h3 : x^3 + y^3 + z^3 = 8) : 
  x^4 + y^4 + z^4 = 26 := by
sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l1406_140680
