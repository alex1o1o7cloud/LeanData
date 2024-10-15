import Mathlib

namespace NUMINAMATH_CALUDE_gwen_recycling_points_l2970_297054

/-- Calculates the points earned by recycling bags of cans. -/
def points_earned (total_bags : ℕ) (unrecycled_bags : ℕ) (points_per_bag : ℕ) : ℕ :=
  (total_bags - unrecycled_bags) * points_per_bag

/-- Proves that Gwen earns 16 points given the problem conditions. -/
theorem gwen_recycling_points :
  points_earned 4 2 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_gwen_recycling_points_l2970_297054


namespace NUMINAMATH_CALUDE_irrational_minus_two_implies_irrational_l2970_297059

theorem irrational_minus_two_implies_irrational (a : ℝ) :
  Irrational (a - 2) → Irrational a := by
  sorry

end NUMINAMATH_CALUDE_irrational_minus_two_implies_irrational_l2970_297059


namespace NUMINAMATH_CALUDE_john_good_games_l2970_297002

/-- The number of good games John ended up with -/
def good_games (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (broken_games : ℕ) : ℕ :=
  games_from_friend + games_from_garage_sale - broken_games

/-- Theorem stating that John ended up with 6 good games -/
theorem john_good_games :
  good_games 21 8 23 = 6 := by
  sorry

end NUMINAMATH_CALUDE_john_good_games_l2970_297002


namespace NUMINAMATH_CALUDE_no_real_roots_l2970_297044

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 2) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l2970_297044


namespace NUMINAMATH_CALUDE_cosine_equality_proof_l2970_297092

theorem cosine_equality_proof : ∃! (n : ℤ), 0 ≤ n ∧ n ≤ 360 ∧ Real.cos (n * π / 180) = Real.cos (1234 * π / 180) ∧ n = 154 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_proof_l2970_297092


namespace NUMINAMATH_CALUDE_cassidy_poster_count_l2970_297014

/-- The number of posters Cassidy had two years ago -/
def posters_two_years_ago : ℕ := 14

/-- The number of posters Cassidy will add this summer -/
def posters_to_add : ℕ := 6

/-- Cassidy's current number of posters -/
def current_posters : ℕ := 22

theorem cassidy_poster_count :
  current_posters + posters_to_add = 2 * posters_two_years_ago :=
by sorry

end NUMINAMATH_CALUDE_cassidy_poster_count_l2970_297014


namespace NUMINAMATH_CALUDE_total_tosses_equals_sum_of_heads_and_tails_l2970_297061

/-- Represents the number of times Head came up in the coin tosses -/
def head_count : ℕ := 9

/-- Represents the number of times Tail came up in the coin tosses -/
def tail_count : ℕ := 5

/-- Theorem stating that the total number of coin tosses is the sum of head_count and tail_count -/
theorem total_tosses_equals_sum_of_heads_and_tails :
  head_count + tail_count = 14 := by sorry

end NUMINAMATH_CALUDE_total_tosses_equals_sum_of_heads_and_tails_l2970_297061


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l2970_297046

theorem part_to_whole_ratio (N P : ℚ) 
  (h1 : (1/4) * P = 17)
  (h2 : (2/5) * N = P)
  (h3 : (40/100) * N = 204) :
  P / N = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l2970_297046


namespace NUMINAMATH_CALUDE_max_third_term_geometric_progression_l2970_297073

/-- Given an arithmetic progression of three terms starting with 5, 
    where adding 5 to the second term and 30 to the third term creates a geometric progression, 
    the maximum possible value for the third term of the resulting geometric progression is 45. -/
theorem max_third_term_geometric_progression : 
  ∀ (d : ℝ), 
  let a₁ : ℝ := 5
  let a₂ : ℝ := 5 + d
  let a₃ : ℝ := 5 + 2*d
  let g₁ : ℝ := a₁
  let g₂ : ℝ := a₂ + 5
  let g₃ : ℝ := a₃ + 30
  (g₂^2 = g₁ * g₃) →
  g₃ ≤ 45 :=
by sorry

end NUMINAMATH_CALUDE_max_third_term_geometric_progression_l2970_297073


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_seven_l2970_297025

theorem smallest_n_divisible_by_seven (n : ℕ) : 
  (n > 50000 ∧ 
   (9 * (n - 2)^6 - n^3 + 20*n - 48) % 7 = 0 ∧
   ∀ m, 50000 < m ∧ m < n → (9 * (m - 2)^6 - m^3 + 20*m - 48) % 7 ≠ 0) →
  n = 50001 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_seven_l2970_297025


namespace NUMINAMATH_CALUDE_jacket_markup_percentage_l2970_297088

theorem jacket_markup_percentage 
  (purchase_price : ℝ)
  (selling_price : ℝ)
  (markup_percentage : ℝ)
  (discount_rate : ℝ)
  (gross_profit : ℝ)
  (h1 : purchase_price = 56)
  (h2 : selling_price = purchase_price + markup_percentage * selling_price)
  (h3 : discount_rate = 0.2)
  (h4 : gross_profit = 8)
  (h5 : gross_profit = (1 - discount_rate) * selling_price - purchase_price) :
  markup_percentage = 0.3 := by
sorry

end NUMINAMATH_CALUDE_jacket_markup_percentage_l2970_297088


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l2970_297029

theorem smallest_number_with_conditions : 
  ∃ (n : ℕ), n = 2102 ∧ 
  (11 ∣ n) ∧ 
  (∀ i : ℕ, 3 ≤ i → i ≤ 7 → n % i = 2) ∧
  (∀ m : ℕ, m < n → ¬((11 ∣ m) ∧ (∀ i : ℕ, 3 ≤ i → i ≤ 7 → m % i = 2))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l2970_297029


namespace NUMINAMATH_CALUDE_largest_value_of_P_10_l2970_297026

/-- A quadratic polynomial with real coefficients -/
def QuadraticPolynomial := ℝ → ℝ

/-- The largest possible value of P(10) for a quadratic polynomial P satisfying given conditions -/
theorem largest_value_of_P_10 (P : QuadraticPolynomial) 
  (h1 : P 1 = 20)
  (h2 : P (-1) = 22)
  (h3 : P (P 0) = 400) :
  ∃ (max : ℝ), P 10 ≤ max ∧ max = 2486 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_of_P_10_l2970_297026


namespace NUMINAMATH_CALUDE_emily_marbles_l2970_297087

theorem emily_marbles (E : ℕ) : 
  (3 * E - (3 * E / 2 + 1) = 8) → E = 6 := by
  sorry

end NUMINAMATH_CALUDE_emily_marbles_l2970_297087


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2970_297036

/-- Given a line with slope -3 passing through the point (2, 4), prove that m + b = 7 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = -3 ∧ 4 = m * 2 + b → m + b = 7 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2970_297036


namespace NUMINAMATH_CALUDE_wall_length_calculation_l2970_297032

/-- Calculates the length of a wall given its height, width, brick dimensions, and total number of bricks. -/
theorem wall_length_calculation (wall_height wall_width brick_length brick_width brick_height total_bricks : ℝ) :
  wall_height = 200 ∧ 
  wall_width = 25 ∧ 
  brick_length = 25 ∧ 
  brick_width = 11.25 ∧ 
  brick_height = 6 ∧ 
  total_bricks = 1185.1851851851852 →
  ∃ wall_length : ℝ, 
    wall_length = 400 ∧ 
    total_bricks = (wall_length * wall_height * wall_width) / (brick_length * brick_width * brick_height) :=
by sorry

end NUMINAMATH_CALUDE_wall_length_calculation_l2970_297032


namespace NUMINAMATH_CALUDE_lcm_problem_l2970_297053

theorem lcm_problem (a b c : ℕ+) (ha : a = 72) (hb : b = 108) (hlcm : Nat.lcm (Nat.lcm a b) c = 37800) : c = 175 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2970_297053


namespace NUMINAMATH_CALUDE_joes_remaining_money_l2970_297057

/-- Represents the problem of calculating Joe's remaining money after shopping --/
theorem joes_remaining_money (initial_amount : ℕ) (notebooks : ℕ) (books : ℕ) 
  (notebook_cost : ℕ) (book_cost : ℕ) : 
  initial_amount = 56 →
  notebooks = 7 →
  books = 2 →
  notebook_cost = 4 →
  book_cost = 7 →
  initial_amount - (notebooks * notebook_cost + books * book_cost) = 14 :=
by sorry

end NUMINAMATH_CALUDE_joes_remaining_money_l2970_297057


namespace NUMINAMATH_CALUDE_janna_weekly_sleep_l2970_297095

/-- Represents the number of hours Janna sleeps in a week. -/
def weekly_sleep_hours (weekday_sleep : ℕ) (weekend_sleep : ℕ) : ℕ :=
  5 * weekday_sleep + 2 * weekend_sleep

/-- Proves that Janna sleeps 51 hours in a week. -/
theorem janna_weekly_sleep :
  weekly_sleep_hours 7 8 = 51 :=
by sorry

end NUMINAMATH_CALUDE_janna_weekly_sleep_l2970_297095


namespace NUMINAMATH_CALUDE_max_k_inequality_l2970_297071

theorem max_k_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ k : ℝ, k = 100 ∧ 
  (∀ k' : ℝ, (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → 
    k' * a' * b' * c' / (a' + b' + c') ≤ (a' + b')^2 + (a' + b' + 4*c')^2) → 
  k' ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_max_k_inequality_l2970_297071


namespace NUMINAMATH_CALUDE_scooter_initial_value_l2970_297019

/-- Proves that the initial value of a scooter is 40000 given the depreciation rate and final value after 3 years -/
theorem scooter_initial_value (depreciation_rate : ℚ) (final_value : ℚ) : 
  depreciation_rate = 3/4 →
  final_value = 16875 →
  (depreciation_rate^3 * 40000 : ℚ) = final_value := by
  sorry

end NUMINAMATH_CALUDE_scooter_initial_value_l2970_297019


namespace NUMINAMATH_CALUDE_zero_point_implies_m_range_l2970_297027

theorem zero_point_implies_m_range (m : ℝ) :
  (∃ x₀ ∈ Set.Icc (-2 : ℝ) 0, 3 * m * x₀ - 4 = 0) →
  m ∈ Set.Iic (-2/3 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_zero_point_implies_m_range_l2970_297027


namespace NUMINAMATH_CALUDE_equation_solution_l2970_297066

theorem equation_solution :
  ∃ x : ℝ, x = 4/3 ∧ 
    (3*x^2)/(x-2) - (3*x + 8)/4 + (6 - 9*x)/(x-2) + 2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2970_297066


namespace NUMINAMATH_CALUDE_union_A_B_disjoint_A_B_l2970_297055

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {y | a < y ∧ y ≤ a + 1}

-- Theorem 1: Union of A and B when a = 3/2
theorem union_A_B : A ∪ B (3/2) = {x | 1 < x ∧ x ≤ 5/2} := by sorry

-- Theorem 2: Condition for A and B to be disjoint
theorem disjoint_A_B : ∀ a : ℝ, A ∩ B a = ∅ ↔ a ≥ 2 ∨ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_union_A_B_disjoint_A_B_l2970_297055


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2970_297094

/-- Given an arithmetic sequence with the first three terms x-y, x+y, and x/y,
    the fourth term is (10 - (2√13)/3) / (√13 - 1) -/
theorem arithmetic_sequence_fourth_term (x y : ℝ) (h : x ≠ 0) :
  let a₁ : ℝ := x - y
  let a₂ : ℝ := x + y
  let a₃ : ℝ := x / y
  let d : ℝ := a₂ - a₁
  let a₄ : ℝ := a₃ + d
  a₄ = (10 - (2 * Real.sqrt 13) / 3) / (Real.sqrt 13 - 1) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2970_297094


namespace NUMINAMATH_CALUDE_apples_remaining_l2970_297065

/-- Calculates the number of apples remaining on a tree after three days of picking -/
theorem apples_remaining (total : ℕ) (day1_fraction : ℚ) (day2_multiplier : ℕ) (day3_addition : ℕ) : 
  total = 200 →
  day1_fraction = 1 / 5 →
  day2_multiplier = 2 →
  day3_addition = 20 →
  total - (total * day1_fraction).floor - day2_multiplier * (total * day1_fraction).floor - ((total * day1_fraction).floor + day3_addition) = 20 := by
sorry

end NUMINAMATH_CALUDE_apples_remaining_l2970_297065


namespace NUMINAMATH_CALUDE_jack_marbles_l2970_297033

theorem jack_marbles (initial : ℕ) (shared : ℕ) (final : ℕ) :
  initial = 62 →
  shared = 33 →
  final = initial - shared →
  final = 29 :=
by sorry

end NUMINAMATH_CALUDE_jack_marbles_l2970_297033


namespace NUMINAMATH_CALUDE_johns_trip_cost_l2970_297068

/-- Calculates the total cost of a car rental trip -/
def total_trip_cost (rental_cost : ℚ) (gas_price : ℚ) (gas_needed : ℚ) (mileage_cost : ℚ) (distance : ℚ) : ℚ :=
  rental_cost + gas_price * gas_needed + mileage_cost * distance

/-- Theorem stating that the total cost of John's trip is $338 -/
theorem johns_trip_cost : 
  total_trip_cost 150 3.5 8 0.5 320 = 338 := by
  sorry

end NUMINAMATH_CALUDE_johns_trip_cost_l2970_297068


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2970_297067

theorem arithmetic_calculations :
  ((-20) + 3 + 5 + (-7) = -19) ∧
  (((-32) / 4) * (1 / 4) = -2) ∧
  ((2 / 7 - 1 / 4) * 28 = 1) ∧
  (-(2^4) * (((-3) * (-(2 + 1 + 1/3)) - (-5))) / ((-2/5)^2) = -1500) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2970_297067


namespace NUMINAMATH_CALUDE_train_length_l2970_297058

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 4 → ∃ length : ℝ, abs (length - 66.68) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l2970_297058


namespace NUMINAMATH_CALUDE_cyclists_circular_track_l2970_297010

/-- Given two cyclists on a circular track starting from the same point in opposite directions
    with speeds of 7 m/s and 8 m/s, meeting at the starting point after 20 seconds,
    the circumference of the track is 300 meters. -/
theorem cyclists_circular_track (speed1 speed2 time : ℝ) (circumference : ℝ) : 
  speed1 = 7 → 
  speed2 = 8 → 
  time = 20 → 
  circumference = (speed1 + speed2) * time → 
  circumference = 300 := by sorry

end NUMINAMATH_CALUDE_cyclists_circular_track_l2970_297010


namespace NUMINAMATH_CALUDE_inequality_proof_l2970_297093

theorem inequality_proof (a b c : ℕ+) (h : c ≥ b) :
  (a ^ b.val) * ((a + b) ^ c.val) > (c ^ b.val) * (a ^ c.val) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2970_297093


namespace NUMINAMATH_CALUDE_bisected_right_triangle_angles_l2970_297049

/-- A right triangle with a bisected right angle -/
structure BisectedRightTriangle where
  /-- The measure of the first acute angle -/
  α : Real
  /-- The measure of the second acute angle -/
  β : Real
  /-- The right angle is 90 degrees -/
  right_angle : α + β = 90
  /-- The angle bisector divides the right angle into two 45-degree angles -/
  bisector_angle : Real
  bisector_property : bisector_angle = 45
  /-- The ratio of angles formed by the angle bisector and the hypotenuse is 7:11 -/
  hypotenuse_angles : Real × Real
  hypotenuse_angles_ratio : hypotenuse_angles.1 / hypotenuse_angles.2 = 7 / 11
  hypotenuse_angles_sum : hypotenuse_angles.1 + hypotenuse_angles.2 = 180 - bisector_angle

/-- The theorem stating the angles of the triangle given the conditions -/
theorem bisected_right_triangle_angles (t : BisectedRightTriangle) : 
  t.α = 65 ∧ t.β = 25 ∧ t.α + t.β = 90 := by
  sorry

end NUMINAMATH_CALUDE_bisected_right_triangle_angles_l2970_297049


namespace NUMINAMATH_CALUDE_characterize_g_l2970_297077

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the properties of g
def is_valid_g (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 - 12 * x + 4

-- Theorem statement
theorem characterize_g :
  ∀ g : ℝ → ℝ, is_valid_g g ↔ (∀ x, g x = 3 * x - 2 ∨ g x = -3 * x + 2) :=
by sorry

end NUMINAMATH_CALUDE_characterize_g_l2970_297077


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2970_297030

theorem inequality_system_solution (x : ℝ) : 
  (5 * x - 1 > 3 * (x + 1) ∧ (1/2) * x - 1 ≤ 7 - (3/2) * x) ↔ (2 < x ∧ x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2970_297030


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2970_297038

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) : x^3 + y^3 = 640 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2970_297038


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2970_297099

/-- The sum of roots of a quadratic equation x^2 + (m-1)x + (m+n) = 0 is 1 - m -/
theorem sum_of_roots_quadratic (m n : ℝ) (hm : m ≠ 1) (hn : n ≠ -m) :
  let f : ℝ → ℝ := λ x => x^2 + (m-1)*x + (m+n)
  (∃ r s : ℝ, f r = 0 ∧ f s = 0) → r + s = 1 - m :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2970_297099


namespace NUMINAMATH_CALUDE_square_not_partitionable_into_10deg_isosceles_triangles_l2970_297081

-- Define a square
def Square : Type := Unit

-- Define an isosceles triangle with a 10° vertex angle
def IsoscelesTriangle10Deg : Type := Unit

-- Define a partition of a square
def Partition (s : Square) : Type := List IsoscelesTriangle10Deg

-- Theorem statement
theorem square_not_partitionable_into_10deg_isosceles_triangles :
  ¬∃ (s : Square) (p : Partition s), p.length > 0 := by
  sorry

end NUMINAMATH_CALUDE_square_not_partitionable_into_10deg_isosceles_triangles_l2970_297081


namespace NUMINAMATH_CALUDE_jack_hunting_problem_l2970_297048

theorem jack_hunting_problem (hunts_per_month : ℕ) (season_length : ℚ) 
  (deer_weight : ℕ) (kept_weight_ratio : ℚ) (total_kept_weight : ℕ) :
  hunts_per_month = 6 →
  season_length = 1 / 4 →
  deer_weight = 600 →
  kept_weight_ratio = 1 / 2 →
  total_kept_weight = 10800 →
  (total_kept_weight / kept_weight_ratio / deer_weight) / (hunts_per_month * (season_length * 12)) = 2 := by
sorry

end NUMINAMATH_CALUDE_jack_hunting_problem_l2970_297048


namespace NUMINAMATH_CALUDE_union_of_specific_sets_l2970_297085

theorem union_of_specific_sets :
  let A : Set ℕ := {2, 5, 6}
  let B : Set ℕ := {3, 5}
  A ∪ B = {2, 3, 5, 6} := by
sorry

end NUMINAMATH_CALUDE_union_of_specific_sets_l2970_297085


namespace NUMINAMATH_CALUDE_pyramid_hemisphere_tangency_l2970_297056

theorem pyramid_hemisphere_tangency (h : ℝ) (r : ℝ) (edge_length : ℝ) : 
  h = 8 → r = 3 → 
  (edge_length * edge_length = 2 * ((h * h - r * r) / h * r)^2) →
  edge_length = 24 * Real.sqrt 110 / 55 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_hemisphere_tangency_l2970_297056


namespace NUMINAMATH_CALUDE_sticker_distribution_equivalence_l2970_297043

/-- The number of ways to distribute n identical objects into k distinct containers --/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute s identical stickers among p identical sheets,
    where each sheet must have at least 1 sticker --/
def distribute_stickers (s p : ℕ) : ℕ := stars_and_bars (s - p) p

theorem sticker_distribution_equivalence :
  distribute_stickers 10 5 = stars_and_bars 5 5 :=
by sorry

end NUMINAMATH_CALUDE_sticker_distribution_equivalence_l2970_297043


namespace NUMINAMATH_CALUDE_matrix_not_invertible_iff_l2970_297021

def matrix (x : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![1 + x, 7;
     3 - x, 8]

theorem matrix_not_invertible_iff (x : ℚ) :
  ¬(Matrix.det (matrix x) ≠ 0) ↔ x = 13/15 := by sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_iff_l2970_297021


namespace NUMINAMATH_CALUDE_factorization_equality_l2970_297039

theorem factorization_equality (x : ℝ) : x * (x - 2) + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2970_297039


namespace NUMINAMATH_CALUDE_ribbon_length_difference_l2970_297096

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the ribbon length for the first method -/
def ribbonLengthMethod1 (box : BoxDimensions) : ℝ :=
  2 * box.length + 2 * box.width + 4 * box.height + 24

/-- Calculates the ribbon length for the second method -/
def ribbonLengthMethod2 (box : BoxDimensions) : ℝ :=
  2 * box.length + 4 * box.width + 2 * box.height + 24

/-- Theorem stating that the difference in ribbon lengths equals one side of the box -/
theorem ribbon_length_difference (box : BoxDimensions) 
    (h1 : box.length = 22) 
    (h2 : box.width = 22) 
    (h3 : box.height = 11) : 
  ribbonLengthMethod2 box - ribbonLengthMethod1 box = box.length := by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_difference_l2970_297096


namespace NUMINAMATH_CALUDE_probability_five_green_marbles_l2970_297023

def total_marbles : ℕ := 12
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 4
def num_draws : ℕ := 8
def num_green_draws : ℕ := 5

def prob_green : ℚ := green_marbles / total_marbles
def prob_purple : ℚ := purple_marbles / total_marbles

theorem probability_five_green_marbles :
  (Nat.choose num_draws num_green_draws : ℚ) * 
  (prob_green ^ num_green_draws) * 
  (prob_purple ^ (num_draws - num_green_draws)) = 1792 / 6561 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_green_marbles_l2970_297023


namespace NUMINAMATH_CALUDE_ivan_total_distance_l2970_297035

/-- Represents the distances Ivan ran on each day of the week -/
structure WeeklyRun where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- The conditions of Ivan's running schedule -/
def validWeeklyRun (run : WeeklyRun) : Prop :=
  run.tuesday = 2 * run.monday ∧
  run.wednesday = run.tuesday / 2 ∧
  run.thursday = run.wednesday / 2 ∧
  run.friday = 2 * run.thursday ∧
  min run.monday (min run.tuesday (min run.wednesday (min run.thursday run.friday))) = 5

/-- The theorem stating that the total distance Ivan ran is 55 km -/
theorem ivan_total_distance (run : WeeklyRun) (h : validWeeklyRun run) :
  run.monday + run.tuesday + run.wednesday + run.thursday + run.friday = 55 := by
  sorry


end NUMINAMATH_CALUDE_ivan_total_distance_l2970_297035


namespace NUMINAMATH_CALUDE_product_sequence_l2970_297079

theorem product_sequence (seq : List ℕ) : 
  (∀ i, i + 3 < seq.length → seq[i]! * seq[i+1]! * seq[i+2]! * seq[i+3]! = 120) →
  (∃ i j k, i < j ∧ j < k ∧ k < seq.length ∧ seq[i]! = 2 ∧ seq[j]! = 4 ∧ seq[k]! = 3) →
  (∃ x, x ∈ seq ∧ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_product_sequence_l2970_297079


namespace NUMINAMATH_CALUDE_distance_not_half_radius_l2970_297001

/-- Two circles with radii p and p/2, whose centers are a non-zero distance d apart -/
structure TwoCircles (p : ℝ) where
  d : ℝ
  d_pos : d > 0

/-- Theorem: The distance between the centers cannot be p/2 -/
theorem distance_not_half_radius (p : ℝ) (circles : TwoCircles p) :
  circles.d ≠ p / 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_not_half_radius_l2970_297001


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_application_l2970_297052

theorem chinese_remainder_theorem_application : 
  ∀ x : ℕ, 1000 < x ∧ x < 4000 ∧ 
    x % 11 = 2 ∧ x % 13 = 12 ∧ x % 19 = 18 ↔ 
    x = 1234 ∨ x = 3951 := by sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_application_l2970_297052


namespace NUMINAMATH_CALUDE_hcf_of_48_and_99_l2970_297009

theorem hcf_of_48_and_99 : Nat.gcd 48 99 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_48_and_99_l2970_297009


namespace NUMINAMATH_CALUDE_investment_plans_count_l2970_297050

/-- The number of candidate cities -/
def num_cities : ℕ := 4

/-- The number of projects to invest -/
def num_projects : ℕ := 3

/-- The maximum number of projects allowed in a single city -/
def max_projects_per_city : ℕ := 2

/-- A function that calculates the number of investment plans -/
def investment_plans : ℕ := sorry

/-- Theorem stating that the number of investment plans is 60 -/
theorem investment_plans_count : investment_plans = 60 := by sorry

end NUMINAMATH_CALUDE_investment_plans_count_l2970_297050


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2970_297090

/-- A rectangle with perimeter 40 and area 96 has dimensions (12, 8) or (8, 12) -/
theorem rectangle_dimensions : 
  ∀ a b : ℝ, 
  (2 * a + 2 * b = 40) →  -- perimeter condition
  (a * b = 96) →          -- area condition
  ((a = 12 ∧ b = 8) ∨ (a = 8 ∧ b = 12)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2970_297090


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l2970_297097

theorem arithmetic_simplification : (-18) + (-12) - (-33) + 17 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l2970_297097


namespace NUMINAMATH_CALUDE_selection_ways_l2970_297011

def boys : ℕ := 5
def girls : ℕ := 3
def total_subjects : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))
def permute (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem selection_ways : 
  choose (boys + girls - 2) (total_subjects - 2) * choose (total_subjects - 2) 1 * permute (total_subjects - 2) (total_subjects - 2) = 360 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_l2970_297011


namespace NUMINAMATH_CALUDE_solution_x_volume_l2970_297034

/-- Proves that the volume of solution x is 50 milliliters, given the conditions of the mixing problem. -/
theorem solution_x_volume (x_concentration : Real) (y_concentration : Real) (y_volume : Real) (final_concentration : Real) :
  x_concentration = 0.10 →
  y_concentration = 0.30 →
  y_volume = 150 →
  final_concentration = 0.25 →
  ∃ (x_volume : Real),
    x_volume = 50 ∧
    (x_concentration * x_volume + y_concentration * y_volume) / (x_volume + y_volume) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_solution_x_volume_l2970_297034


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2970_297078

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.sqrt 2 - 3) :
  (1 - 3 / (m + 3)) / (m / (m^2 + 6*m + 9)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2970_297078


namespace NUMINAMATH_CALUDE_swimmers_speed_l2970_297045

/-- Swimmer's speed in still water given time, distance, and current speed -/
theorem swimmers_speed (time : ℝ) (distance : ℝ) (current_speed : ℝ) 
  (h1 : time = 2.5)
  (h2 : distance = 5)
  (h3 : current_speed = 2) :
  ∃ v : ℝ, v = 4 ∧ time = distance / (v - current_speed) :=
by sorry

end NUMINAMATH_CALUDE_swimmers_speed_l2970_297045


namespace NUMINAMATH_CALUDE_gcd_459_357_l2970_297016

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l2970_297016


namespace NUMINAMATH_CALUDE_division_reciprocal_equivalence_l2970_297042

theorem division_reciprocal_equivalence (x : ℝ) (hx : x ≠ 0) :
  1 / x = 1 * (1 / x) :=
by sorry

end NUMINAMATH_CALUDE_division_reciprocal_equivalence_l2970_297042


namespace NUMINAMATH_CALUDE_solution_of_equation_l2970_297024

-- Define the custom operation
def customOp (a b : ℝ) : ℝ := 3 * a - 4 * b

-- State the theorem
theorem solution_of_equation :
  ∃ x : ℝ, customOp 2 (customOp 2 x) = customOp 1 x ∧ x = 21 / 20 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l2970_297024


namespace NUMINAMATH_CALUDE_ball_peak_time_l2970_297080

/-- Given a ball thrown upwards, this theorem proves that with an initial velocity of 1.25 m/s, 
    it takes 6.25 seconds to reach its peak height. -/
theorem ball_peak_time (v : ℝ) (t : ℝ) :
  v = 1.25 → t = 4 * v^2 → t = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_ball_peak_time_l2970_297080


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2970_297062

theorem simplify_and_evaluate : 
  let x : ℚ := -1
  (1 - 1 / (x - 1)) / ((x^2 - 4) / (x^2 - x)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2970_297062


namespace NUMINAMATH_CALUDE_wages_calculation_l2970_297063

def total_budget : ℚ := 3000

def food_fraction : ℚ := 1/3
def supplies_fraction : ℚ := 1/4

def food_expense : ℚ := food_fraction * total_budget
def supplies_expense : ℚ := supplies_fraction * total_budget

def wages_expense : ℚ := total_budget - (food_expense + supplies_expense)

theorem wages_calculation : wages_expense = 1250 := by
  sorry

end NUMINAMATH_CALUDE_wages_calculation_l2970_297063


namespace NUMINAMATH_CALUDE_fraction_difference_equals_one_l2970_297086

theorem fraction_difference_equals_one (x y : ℝ) (h : x * y = x - y) (h_nonzero : x * y ≠ 0) :
  1 / y - 1 / x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_one_l2970_297086


namespace NUMINAMATH_CALUDE_smallest_c_for_inequality_l2970_297041

theorem smallest_c_for_inequality : ∃ c : ℕ, (∀ k : ℕ, 27^k > 3^24 → c ≤ k) ∧ 27^c > 3^24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_for_inequality_l2970_297041


namespace NUMINAMATH_CALUDE_floor_sqrt_27_squared_l2970_297084

theorem floor_sqrt_27_squared : ⌊Real.sqrt 27⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_27_squared_l2970_297084


namespace NUMINAMATH_CALUDE_f_derivative_at_pi_over_4_l2970_297015

noncomputable def f (x : ℝ) : ℝ := Real.sin x / (Real.sin x + Real.cos x)

theorem f_derivative_at_pi_over_4 :
  deriv f (π/4) = 1/2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_pi_over_4_l2970_297015


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l2970_297022

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : Real) 
  (train_speed_kmh : Real) 
  (bridge_length : Real) : 
  train_length = 145 ∧ 
  train_speed_kmh = 45 ∧ 
  bridge_length = 230 → 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l2970_297022


namespace NUMINAMATH_CALUDE_function_expression_l2970_297076

theorem function_expression (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x + 1) = x + 1) :
  ∀ x : ℝ, f x = (1/2) * (x + 1) := by
sorry

end NUMINAMATH_CALUDE_function_expression_l2970_297076


namespace NUMINAMATH_CALUDE_empty_vessel_mass_l2970_297064

/-- The mass of an empty vessel given the masses when filled with kerosene and water, and the densities of kerosene and water. -/
theorem empty_vessel_mass
  (mass_with_kerosene : ℝ)
  (mass_with_water : ℝ)
  (density_water : ℝ)
  (density_kerosene : ℝ)
  (h1 : mass_with_kerosene = 31)
  (h2 : mass_with_water = 33)
  (h3 : density_water = 1000)
  (h4 : density_kerosene = 800) :
  ∃ (empty_mass : ℝ) (volume : ℝ),
    empty_mass = 23 ∧
    mass_with_kerosene = empty_mass + density_kerosene * volume ∧
    mass_with_water = empty_mass + density_water * volume :=
by sorry

end NUMINAMATH_CALUDE_empty_vessel_mass_l2970_297064


namespace NUMINAMATH_CALUDE_circle_equal_circumference_area_diameter_l2970_297017

/-- A circle with numerically equal circumference and area has a diameter of 4 -/
theorem circle_equal_circumference_area_diameter (r : ℝ) (h : r > 0) :
  π * (2 * r) = π * r^2 → 2 * r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equal_circumference_area_diameter_l2970_297017


namespace NUMINAMATH_CALUDE_solution_set_equiv_l2970_297031

theorem solution_set_equiv (x : ℝ) : 
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_equiv_l2970_297031


namespace NUMINAMATH_CALUDE_M_subset_N_l2970_297083

-- Define the sets M and N
def M : Set ℝ := {α | ∃ k : ℤ, α = k * 90 ∨ α = k * 180 + 45}
def N : Set ℝ := {α | ∃ k : ℤ, α = k * 45}

-- Theorem statement
theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l2970_297083


namespace NUMINAMATH_CALUDE_infinite_perfect_squares_in_sequence_l2970_297013

theorem infinite_perfect_squares_in_sequence : 
  ∀ k : ℕ, ∃ n : ℕ, ∃ m : ℕ, 2^n + 4^k = m^2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_perfect_squares_in_sequence_l2970_297013


namespace NUMINAMATH_CALUDE_area_closed_region_l2970_297018

/-- The area of the closed region formed by f(x) and g(x) over one period -/
theorem area_closed_region (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (a * x) + Real.cos (a * x)
  let g : ℝ → ℝ := λ x ↦ Real.sqrt (a^2 + 1)
  let period : ℝ := 2 * Real.pi / a
  ∃ (area : ℝ), area = period * Real.sqrt (a^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_area_closed_region_l2970_297018


namespace NUMINAMATH_CALUDE_cynthia_water_balloons_l2970_297091

/-- The number of water balloons each person has -/
structure WaterBalloons where
  janice : ℕ
  randy : ℕ
  cynthia : ℕ

/-- The conditions of the water balloon distribution -/
def water_balloon_conditions (wb : WaterBalloons) : Prop :=
  wb.janice = 6 ∧
  wb.randy = wb.janice / 2 ∧
  wb.cynthia = 4 * wb.randy

theorem cynthia_water_balloons (wb : WaterBalloons) 
  (h : water_balloon_conditions wb) : wb.cynthia = 12 := by
  sorry

#check cynthia_water_balloons

end NUMINAMATH_CALUDE_cynthia_water_balloons_l2970_297091


namespace NUMINAMATH_CALUDE_factorization_validity_l2970_297075

theorem factorization_validity (x y : ℝ) : 5 * x^2 * y - 10 * x * y^2 = 5 * x * y * (x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_validity_l2970_297075


namespace NUMINAMATH_CALUDE_carpet_shampooing_time_l2970_297082

theorem carpet_shampooing_time 
  (jason_rate : ℝ) 
  (tom_rate : ℝ) 
  (h1 : jason_rate = 1 / 3) 
  (h2 : tom_rate = 1 / 6) : 
  1 / (jason_rate + tom_rate) = 2 := by
  sorry

end NUMINAMATH_CALUDE_carpet_shampooing_time_l2970_297082


namespace NUMINAMATH_CALUDE_sequence_sum_l2970_297069

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => 1  -- First term is 1
  | n + 1 => 
    let k := (n + 1).sqrt  -- k-th group
    if (k * k ≤ n + 1) ∧ (n + 1 < (k + 1) * (k + 1)) then
      if n + 1 = k * k then 1 else 2
    else a n  -- This case should never happen, but Lean needs it for totality

-- Define the sum S_n
def S (n : ℕ) : ℕ := (List.range n).map a |>.sum

-- Theorem statement
theorem sequence_sum :
  S 20 = 36 ∧ S 2017 = 3989 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_l2970_297069


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_factorization_exists_smallest_b_is_85_l2970_297012

theorem smallest_factorization_coefficient (b : ℕ) : 
  (∃ r s : ℤ, x^2 + b*x + 1800 = (x + r) * (x + s)) →
  b ≥ 85 :=
by sorry

theorem factorization_exists : 
  ∃ r s : ℤ, x^2 + 85*x + 1800 = (x + r) * (x + s) :=
by sorry

theorem smallest_b_is_85 : 
  (∃ r s : ℤ, x^2 + 85*x + 1800 = (x + r) * (x + s)) ∧
  (∀ b : ℕ, b < 85 → ¬(∃ r s : ℤ, x^2 + b*x + 1800 = (x + r) * (x + s))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_factorization_exists_smallest_b_is_85_l2970_297012


namespace NUMINAMATH_CALUDE_equation_holds_when_b_plus_c_is_ten_l2970_297007

theorem equation_holds_when_b_plus_c_is_ten (a b c : ℕ) : 
  a > 0 → a < 10 → b > 0 → b < 10 → c > 0 → c < 10 → b + c = 10 → 
  (10 * b + a) * (10 * c + a) = 100 * b * c + 100 * a + a^2 := by
sorry

end NUMINAMATH_CALUDE_equation_holds_when_b_plus_c_is_ten_l2970_297007


namespace NUMINAMATH_CALUDE_anitas_strawberries_l2970_297072

theorem anitas_strawberries (total_cartons : ℕ) (blueberry_cartons : ℕ) (cartons_to_buy : ℕ) 
  (h1 : total_cartons = 26)
  (h2 : blueberry_cartons = 9)
  (h3 : cartons_to_buy = 7) :
  total_cartons - (blueberry_cartons + cartons_to_buy) = 10 := by
  sorry

end NUMINAMATH_CALUDE_anitas_strawberries_l2970_297072


namespace NUMINAMATH_CALUDE_maritime_silk_road_analysis_l2970_297006

/-- Represents the Maritime Silk Road -/
structure MaritimeSilkRoad where
  economic_exchange : Bool
  cultural_exchange : Bool

/-- Represents the discussion method used -/
structure DiscussionMethod where
  theory_of_two_points : Bool
  theory_of_emphasis : Bool

/-- Represents the viewpoints in the discussion -/
inductive Viewpoint
  | economy_first
  | culture_first

/-- Theorem stating the analysis of the Maritime Silk Road discussion -/
theorem maritime_silk_road_analysis 
  (msr : MaritimeSilkRoad) 
  (method : DiscussionMethod) 
  (viewpoints : List Viewpoint) :
  msr.economic_exchange = true →
  msr.cultural_exchange = true →
  method.theory_of_two_points = true →
  method.theory_of_emphasis = true →
  viewpoints.length > 1 →
  (∃ (analysis : Bool), 
    analysis = true ↔ 
      (∃ (social_existence_consciousness : Bool) (culture_economy : Bool),
        social_existence_consciousness = true ∧ 
        culture_economy = true)) :=
by sorry

end NUMINAMATH_CALUDE_maritime_silk_road_analysis_l2970_297006


namespace NUMINAMATH_CALUDE_eight_elevenths_rounded_l2970_297004

/-- Rounds a rational number to the specified number of decimal places -/
def round_to_decimal_places (q : ℚ) (places : ℕ) : ℚ :=
  (⌊q * 10^places + 1/2⌋ : ℚ) / 10^places

/-- Proves that 8/11 rounded to 3 decimal places is equal to 0.727 -/
theorem eight_elevenths_rounded : round_to_decimal_places (8/11) 3 = 727/1000 := by
  sorry

end NUMINAMATH_CALUDE_eight_elevenths_rounded_l2970_297004


namespace NUMINAMATH_CALUDE_triangle_area_l2970_297051

/-- Given a triangle ABC with circumcircle diameter 4√3/3, angle C = 60°, and a + b = ab, 
    the area of the triangle is √3. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  (∃ (R : ℝ), 2 * R = 4 * Real.sqrt 3 / 3) →  -- Circumcircle diameter condition
  C = π / 3 →                                 -- Angle C = 60°
  a + b = a * b →                             -- Given condition
  (∃ (S : ℝ), S = a * b * Real.sin C / 2 ∧ S = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2970_297051


namespace NUMINAMATH_CALUDE_set_operations_proof_l2970_297005

def U := Set ℝ

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3 ∨ 4 < x ∧ x < 6}

def B : Set ℝ := {x | 2 ≤ x ∧ x < 5}

theorem set_operations_proof :
  (Set.compl B = {x : ℝ | x < 2 ∨ x ≥ 5}) ∧
  (A ∩ (Set.compl B) = {x : ℝ | (1 ≤ x ∧ x < 2) ∨ (5 ≤ x ∧ x < 6)}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_proof_l2970_297005


namespace NUMINAMATH_CALUDE_symmetric_sine_value_l2970_297047

/-- Given a function f(x) = 2sin(wx + φ) that is symmetric about x = π/6,
    prove that f(π/6) is either -2 or 2. -/
theorem symmetric_sine_value (w φ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (w * x + φ)
  (∀ x, f (π/6 + x) = f (π/6 - x)) →
  f (π/6) = -2 ∨ f (π/6) = 2 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_sine_value_l2970_297047


namespace NUMINAMATH_CALUDE_first_graders_count_l2970_297037

/-- The number of Kindergartners -/
def kindergartners : ℕ := 101

/-- The cost of an orange shirt for Kindergartners -/
def orange_shirt_cost : ℚ := 29/5

/-- The cost of a yellow shirt for first graders -/
def yellow_shirt_cost : ℚ := 5

/-- The number of second graders -/
def second_graders : ℕ := 107

/-- The cost of a blue shirt for second graders -/
def blue_shirt_cost : ℚ := 28/5

/-- The number of third graders -/
def third_graders : ℕ := 108

/-- The cost of a green shirt for third graders -/
def green_shirt_cost : ℚ := 21/4

/-- The total amount spent by the P.T.O. -/
def total_spent : ℚ := 2317

/-- The number of first graders wearing yellow shirts -/
def first_graders : ℕ := 113

theorem first_graders_count : 
  first_graders * yellow_shirt_cost + 
  kindergartners * orange_shirt_cost + 
  second_graders * blue_shirt_cost + 
  third_graders * green_shirt_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_first_graders_count_l2970_297037


namespace NUMINAMATH_CALUDE_average_marks_equals_85_l2970_297028

def english_marks : ℕ := 86
def math_marks : ℕ := 89
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 87
def biology_marks : ℕ := 81

def total_marks : ℕ := english_marks + math_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

theorem average_marks_equals_85 : (total_marks : ℚ) / num_subjects = 85 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_equals_85_l2970_297028


namespace NUMINAMATH_CALUDE_smallest_integer_gcf_24_is_4_l2970_297070

theorem smallest_integer_gcf_24_is_4 : 
  ∀ n : ℕ, n > 100 → Nat.gcd n 24 = 4 → n ≥ 104 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_gcf_24_is_4_l2970_297070


namespace NUMINAMATH_CALUDE_train_length_calculation_l2970_297020

/-- Calculates the length of a train given its speed, the length of a bridge it passes, and the time it takes to pass the bridge. -/
theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (passing_time : ℝ) : 
  train_speed = 50 * (1000 / 3600) → 
  bridge_length = 140 →
  passing_time = 36 →
  (train_speed * passing_time) - bridge_length = 360 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l2970_297020


namespace NUMINAMATH_CALUDE_f_monotonicity_and_max_k_l2970_297060

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 2

theorem f_monotonicity_and_max_k :
  (∀ a ≤ 0, ∀ x y, x < y → f a x < f a y) ∧
  (∀ a > 0, ∀ x y, x < y → 
    ((x < Real.log a ∧ y < Real.log a → f a x > f a y) ∧
     (x > Real.log a ∧ y > Real.log a → f a x < f a y))) ∧
  (∀ k : ℤ, (∀ x > 0, (x - ↑k) * (Real.exp x - 1) + x + 1 > 0) → k ≤ 2) ∧
  (∀ x > 0, (x - 2) * (Real.exp x - 1) + x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_max_k_l2970_297060


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l2970_297003

theorem fixed_point_exponential_function :
  ∀ (a : ℝ), a > 0 → ((-2 : ℝ)^((-2 : ℝ) + 2) - 3 = -2) := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l2970_297003


namespace NUMINAMATH_CALUDE_baker_cakes_theorem_l2970_297040

def total_cakes (initial : ℕ) (extra : ℕ) : ℕ :=
  initial + extra

theorem baker_cakes_theorem (initial : ℕ) (extra : ℕ) :
  total_cakes initial extra = initial + extra := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_theorem_l2970_297040


namespace NUMINAMATH_CALUDE_order_of_abc_l2970_297098

noncomputable def a : ℝ := 2 + Real.sqrt 3
noncomputable def b : ℝ := 1 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 + Real.sqrt 5

theorem order_of_abc : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l2970_297098


namespace NUMINAMATH_CALUDE_sqrt_sum_quotient_l2970_297074

theorem sqrt_sum_quotient : (Real.sqrt 112 + Real.sqrt 567) / Real.sqrt 175 = 13 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_quotient_l2970_297074


namespace NUMINAMATH_CALUDE_binomial_150_150_l2970_297089

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_l2970_297089


namespace NUMINAMATH_CALUDE_bijective_if_injective_or_surjective_finite_sets_l2970_297000

theorem bijective_if_injective_or_surjective_finite_sets
  {X Y : Type} [Fintype X] [Fintype Y]
  (h_card_eq : Fintype.card X = Fintype.card Y)
  (f : X → Y)
  (h_inj_or_surj : Function.Injective f ∨ Function.Surjective f) :
  Function.Bijective f :=
sorry

end NUMINAMATH_CALUDE_bijective_if_injective_or_surjective_finite_sets_l2970_297000


namespace NUMINAMATH_CALUDE_prob_two_private_teams_prob_distribution_ξ_expectation_ξ_l2970_297008

/-- The number of guided tour teams -/
def guided_teams : ℕ := 6

/-- The number of private tour teams -/
def private_teams : ℕ := 3

/-- The total number of teams -/
def total_teams : ℕ := guided_teams + private_teams

/-- The number of draws with replacement -/
def num_draws : ℕ := 4

/-- The random variable representing the number of private teams drawn -/
def ξ : ℕ → ℝ := sorry

/-- The probability of drawing two private tour teams when selecting two numbers at a time -/
theorem prob_two_private_teams : 
  (Nat.choose private_teams 2 : ℚ) / (Nat.choose total_teams 2) = 1 / 12 := by sorry

/-- The probability distribution of ξ -/
theorem prob_distribution_ξ : 
  (ξ 0 = 16 / 81) ∧ 
  (ξ 1 = 32 / 81) ∧ 
  (ξ 2 = 8 / 27) ∧ 
  (ξ 3 = 8 / 81) ∧ 
  (ξ 4 = 1 / 81) := by sorry

/-- The mathematical expectation of ξ -/
theorem expectation_ξ : 
  (0 * ξ 0 + 1 * ξ 1 + 2 * ξ 2 + 3 * ξ 3 + 4 * ξ 4 : ℝ) = 4 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_two_private_teams_prob_distribution_ξ_expectation_ξ_l2970_297008
