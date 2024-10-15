import Mathlib

namespace NUMINAMATH_CALUDE_number_difference_l2588_258826

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 21780)
  (a_div_5 : a % 5 = 0)
  (b_relation : b * 10 + 5 = a) :
  a - b = 17825 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2588_258826


namespace NUMINAMATH_CALUDE_vector_problem_l2588_258881

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (-1, 7)

theorem vector_problem :
  (a.1 * b.1 + a.2 * b.2 = 25) ∧
  (Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l2588_258881


namespace NUMINAMATH_CALUDE_smallest_sum_of_factors_l2588_258811

theorem smallest_sum_of_factors (r s t : ℕ+) (h : r * s * t = 1230) :
  ∃ (r' s' t' : ℕ+), r' * s' * t' = 1230 ∧ r' + s' + t' = 52 ∧
  ∀ (x y z : ℕ+), x * y * z = 1230 → r' + s' + t' ≤ x + y + z :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_factors_l2588_258811


namespace NUMINAMATH_CALUDE_two_sqrt_two_less_than_three_l2588_258854

theorem two_sqrt_two_less_than_three : 2 * Real.sqrt 2 < 3 := by
  sorry

end NUMINAMATH_CALUDE_two_sqrt_two_less_than_three_l2588_258854


namespace NUMINAMATH_CALUDE_zoo_camels_l2588_258852

theorem zoo_camels (a : ℕ) 
  (h1 : ∃ x y : ℕ, x = y + 10 ∧ x + y = a)
  (h2 : ∃ x y : ℕ, x + 2*y = 55 ∧ x + y = a) : 
  a = 40 := by
sorry

end NUMINAMATH_CALUDE_zoo_camels_l2588_258852


namespace NUMINAMATH_CALUDE_balloon_distribution_l2588_258841

theorem balloon_distribution (total_balloons : ℕ) (friends : ℕ) 
  (h1 : total_balloons = 235) (h2 : friends = 10) : 
  total_balloons % friends = 5 := by
  sorry

end NUMINAMATH_CALUDE_balloon_distribution_l2588_258841


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l2588_258870

theorem quadratic_equation_integer_roots (a : ℚ) :
  (∃ x y : ℤ, a * x^2 + (a + 1) * x + (a - 1) = 0 ∧
               a * y^2 + (a + 1) * y + (a - 1) = 0 ∧
               x ≠ y) →
  (a = 0 ∨ a = -1/7 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l2588_258870


namespace NUMINAMATH_CALUDE_circle_center_l2588_258830

/-- The equation of a circle C in the form x^2 + y^2 + ax + by + c = 0 -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The center of a circle -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a circle C with equation x^2 + y^2 - 2x + y + 1/4 = 0,
    its center is the point (1, -1/2) -/
theorem circle_center (C : Circle) 
  (h : C = { a := -2, b := 1, c := 1/4 }) : 
  ∃ center : Point, center = { x := 1, y := -1/2 } :=
sorry

end NUMINAMATH_CALUDE_circle_center_l2588_258830


namespace NUMINAMATH_CALUDE_football_team_size_l2588_258861

/-- Represents the number of players on a football team -/
def total_players : ℕ := 70

/-- Represents the number of throwers on the team -/
def throwers : ℕ := 52

/-- Represents the total number of right-handed players -/
def right_handed_players : ℕ := 64

/-- States that one third of non-throwers are left-handed -/
axiom one_third_non_throwers_left_handed :
  (total_players - throwers) / 3 = (total_players - throwers - (right_handed_players - throwers))

/-- All throwers are right-handed -/
axiom all_throwers_right_handed :
  throwers ≤ right_handed_players

/-- Theorem stating that the total number of players is 70 -/
theorem football_team_size :
  total_players = 70 :=
sorry

end NUMINAMATH_CALUDE_football_team_size_l2588_258861


namespace NUMINAMATH_CALUDE_abs_one_fifth_set_l2588_258868

theorem abs_one_fifth_set : 
  {x : ℝ | |x| = (1 : ℝ) / 5} = {-(1 : ℝ) / 5, (1 : ℝ) / 5} := by
  sorry

end NUMINAMATH_CALUDE_abs_one_fifth_set_l2588_258868


namespace NUMINAMATH_CALUDE_remainder_987670_div_128_l2588_258823

theorem remainder_987670_div_128 : 987670 % 128 = 22 := by
  sorry

end NUMINAMATH_CALUDE_remainder_987670_div_128_l2588_258823


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2588_258845

/-- The sum of an arithmetic sequence with first term 1, common difference 2, and 20 terms -/
def arithmetic_sum : ℕ → ℕ
  | 0 => 0
  | n + 1 => (n + 1) + arithmetic_sum n

/-- The first term of the sequence -/
def a₁ : ℕ := 1

/-- The common difference of the sequence -/
def d : ℕ := 2

/-- The number of terms in the sequence -/
def n : ℕ := 20

/-- The n-th term of the sequence -/
def aₙ (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_sum :
  arithmetic_sum n = n * (a₁ + aₙ n) / 2 ∧ arithmetic_sum n = 400 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2588_258845


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l2588_258843

theorem fraction_equals_zero (x : ℝ) : (x - 5) / (5 * x - 15) = 0 ↔ x = 5 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l2588_258843


namespace NUMINAMATH_CALUDE_letter_distribution_l2588_258847

/-- The number of ways to distribute n distinct objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- There are 5 distinct letters -/
def num_letters : ℕ := 5

/-- There are 3 distinct mailboxes -/
def num_mailboxes : ℕ := 3

/-- The number of ways to distribute 5 letters into 3 mailboxes is 3^5 -/
theorem letter_distribution : distribute num_letters num_mailboxes = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_letter_distribution_l2588_258847


namespace NUMINAMATH_CALUDE_race_average_time_per_km_l2588_258863

theorem race_average_time_per_km (race_distance : ℝ) (first_half_time second_half_time : ℝ) :
  race_distance = 10 →
  first_half_time = 20 →
  second_half_time = 30 →
  (first_half_time + second_half_time) / race_distance = 5 := by
  sorry

end NUMINAMATH_CALUDE_race_average_time_per_km_l2588_258863


namespace NUMINAMATH_CALUDE_science_club_teams_l2588_258808

theorem science_club_teams (girls : ℕ) (boys : ℕ) :
  girls = 4 → boys = 7 → (girls.choose 3) * (boys.choose 2) = 84 := by
  sorry

end NUMINAMATH_CALUDE_science_club_teams_l2588_258808


namespace NUMINAMATH_CALUDE_jason_initial_cards_l2588_258860

/-- The number of Pokemon cards Jason gave away -/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has left -/
def cards_left : ℕ := 4

/-- The initial number of Pokemon cards Jason had -/
def initial_cards : ℕ := cards_given_away + cards_left

theorem jason_initial_cards : initial_cards = 13 := by
  sorry

end NUMINAMATH_CALUDE_jason_initial_cards_l2588_258860


namespace NUMINAMATH_CALUDE_coin_collection_value_l2588_258819

theorem coin_collection_value (total_coins : ℕ) (sample_coins : ℕ) (sample_value : ℕ) 
  (h1 : total_coins = 20)
  (h2 : sample_coins = 4)
  (h3 : sample_value = 16) :
  (total_coins : ℚ) * (sample_value : ℚ) / (sample_coins : ℚ) = 80 := by
  sorry

end NUMINAMATH_CALUDE_coin_collection_value_l2588_258819


namespace NUMINAMATH_CALUDE_equation_solutions_l2588_258801

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (2 * x₁^2 = 5 * x₁ ∧ x₁ = 0) ∧ (2 * x₂^2 = 5 * x₂ ∧ x₂ = 5/2)) ∧
  (∃ y₁ y₂ : ℝ, (y₁^2 + 3*y₁ = 3 ∧ y₁ = (-3 + Real.sqrt 21) / 2) ∧
               (y₂^2 + 3*y₂ = 3 ∧ y₂ = (-3 - Real.sqrt 21) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2588_258801


namespace NUMINAMATH_CALUDE_percentage_equality_l2588_258829

theorem percentage_equality : (0.75 * 40 : ℝ) = (4/5 : ℝ) * 25 + 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l2588_258829


namespace NUMINAMATH_CALUDE_max_value_of_inverse_sum_l2588_258879

open Real

-- Define the quadratic equation and its roots
def quadratic (t q : ℝ) (x : ℝ) : ℝ := x^2 - t*x + q

-- Define the condition for the roots
def roots_condition (α β : ℝ) : Prop :=
  α + β = α^2 + β^2 ∧ α + β = α^3 + β^3 ∧ α + β = α^4 + β^4 ∧ α + β = α^5 + β^5

-- Theorem statement
theorem max_value_of_inverse_sum (t q α β : ℝ) :
  (∀ x, quadratic t q x = 0 ↔ x = α ∨ x = β) →
  roots_condition α β →
  (∀ γ δ : ℝ, roots_condition γ δ → 1/γ^6 + 1/δ^6 ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_inverse_sum_l2588_258879


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l2588_258848

/-- Given a rhombus with area 64/5 square centimeters and one diagonal 64/9 centimeters,
    prove that the other diagonal is 18/5 centimeters. -/
theorem rhombus_diagonal (area : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ) : 
  area = 64/5 → 
  diagonal1 = 64/9 → 
  area = (diagonal1 * diagonal2) / 2 → 
  diagonal2 = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l2588_258848


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2588_258892

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_eq_one : a + b + c = 1)
  (sum_sq_eq_one : a^2 + b^2 + c^2 = 1)
  (sum_cube_eq_one : a^3 + b^3 + c^3 = 1) : 
  a * b * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2588_258892


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2588_258807

theorem quadratic_inequality_solution_set 
  (α β a b c : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < β) 
  (h3 : ∀ x, a * x^2 + b * x + c > 0 ↔ α < x ∧ x < β) :
  ∀ x, (a + c - b) * x^2 + (b - 2*a) * x + a > 0 ↔ 1 / (1 + β) < x ∧ x < 1 / (1 + α) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2588_258807


namespace NUMINAMATH_CALUDE_birthday_presents_total_l2588_258899

def leonard_wallets : ℕ := 3
def leonard_wallet_price : ℕ := 35
def leonard_sneakers : ℕ := 2
def leonard_sneaker_price : ℕ := 120
def leonard_belt_price : ℕ := 45

def michael_backpack_price : ℕ := 90
def michael_jeans : ℕ := 3
def michael_jeans_price : ℕ := 55
def michael_tie_price : ℕ := 25

def emily_shirts : ℕ := 2
def emily_shirt_price : ℕ := 70
def emily_books : ℕ := 4
def emily_book_price : ℕ := 15

def total_spent : ℕ := 870

theorem birthday_presents_total :
  (leonard_wallets * leonard_wallet_price + 
   leonard_sneakers * leonard_sneaker_price + 
   leonard_belt_price) +
  (michael_backpack_price + 
   michael_jeans * michael_jeans_price + 
   michael_tie_price) +
  (emily_shirts * emily_shirt_price + 
   emily_books * emily_book_price) = total_spent := by
  sorry

end NUMINAMATH_CALUDE_birthday_presents_total_l2588_258899


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l2588_258874

theorem quadrilateral_diagonal_length 
  (offset1 offset2 total_area : ℝ) 
  (h1 : offset1 = 9)
  (h2 : offset2 = 6)
  (h3 : total_area = 180)
  (h4 : total_area = (offset1 + offset2) * diagonal / 2) :
  diagonal = 24 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l2588_258874


namespace NUMINAMATH_CALUDE_opposite_numbers_not_on_hyperbola_l2588_258891

theorem opposite_numbers_not_on_hyperbola (x y : ℝ) : 
  y = 1 / x → x ≠ -y := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_not_on_hyperbola_l2588_258891


namespace NUMINAMATH_CALUDE_work_completion_time_l2588_258853

/-- The number of days it takes for a and b together to complete the work -/
def combined_time : ℝ := 6

/-- The number of days it takes for b alone to complete the work -/
def b_time : ℝ := 11.142857142857144

/-- The number of days it takes for a alone to complete the work -/
def a_time : ℝ := 13

/-- The theorem stating that given the combined time and b's time, a's time is 13 days -/
theorem work_completion_time : 
  (1 / combined_time) = (1 / a_time) + (1 / b_time) :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2588_258853


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2588_258805

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

-- Theorem stating that the given asymptote equation is correct for the hyperbola
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola x y → (∃ x' y' : ℝ, x' ≠ x ∧ y' ≠ y ∧ hyperbola x' y' ∧ asymptote x' y') :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2588_258805


namespace NUMINAMATH_CALUDE_train_speed_l2588_258824

/-- The speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) (h1 : train_length = 240) 
  (h2 : bridge_length = 130) (h3 : time = 26.64) : 
  ∃ (speed : ℝ), abs (speed - 50.004) < 0.001 ∧ 
  speed = (train_length + bridge_length) / time * 3.6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2588_258824


namespace NUMINAMATH_CALUDE_f_difference_l2588_258812

/-- Sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Function f(n) defined as sigma(n) / n -/
def f (n : ℕ+) : ℚ := (sigma n : ℚ) / n

/-- Theorem stating that f(420) - f(360) = 143/20 -/
theorem f_difference : f 420 - f 360 = 143 / 20 := by sorry

end NUMINAMATH_CALUDE_f_difference_l2588_258812


namespace NUMINAMATH_CALUDE_seven_b_equals_ten_l2588_258844

theorem seven_b_equals_ten (a b : ℚ) (h1 : 5 * a + 2 * b = 0) (h2 : b - 2 = a) : 7 * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_seven_b_equals_ten_l2588_258844


namespace NUMINAMATH_CALUDE_system_solution_l2588_258818

theorem system_solution :
  ∃! (x y : ℚ), (2 * x + 3 * y = (7 - 2 * x) + (7 - 3 * y)) ∧
                 (3 * x - 2 * y = (x - 2) + (y - 2)) ∧
                 x = 3 / 4 ∧ y = 11 / 6 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2588_258818


namespace NUMINAMATH_CALUDE_isabel_pop_albums_l2588_258842

/-- The number of country albums Isabel bought -/
def country_albums : ℕ := 6

/-- The number of songs per album -/
def songs_per_album : ℕ := 9

/-- The total number of songs Isabel bought -/
def total_songs : ℕ := 72

/-- The number of pop albums Isabel bought -/
def pop_albums : ℕ := (total_songs - country_albums * songs_per_album) / songs_per_album

theorem isabel_pop_albums : pop_albums = 2 := by
  sorry

end NUMINAMATH_CALUDE_isabel_pop_albums_l2588_258842


namespace NUMINAMATH_CALUDE_sara_marbles_l2588_258897

def marbles_problem (initial_marbles : ℕ) (remaining_marbles : ℕ) : Prop :=
  initial_marbles - remaining_marbles = 7

theorem sara_marbles : marbles_problem 10 3 := by
  sorry

end NUMINAMATH_CALUDE_sara_marbles_l2588_258897


namespace NUMINAMATH_CALUDE_subtract_negatives_l2588_258836

theorem subtract_negatives : (-1) - (-4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negatives_l2588_258836


namespace NUMINAMATH_CALUDE_function_inequality_l2588_258821

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, x > 1 → y > x → f x < f y)
variable (h2 : ∀ x, f (x + 1) = f (-x + 1))

-- State the theorem
theorem function_inequality : f (-1) < f 0 ∧ f 0 < f 4 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2588_258821


namespace NUMINAMATH_CALUDE_multiply_72519_9999_l2588_258865

theorem multiply_72519_9999 : 72519 * 9999 = 725117481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72519_9999_l2588_258865


namespace NUMINAMATH_CALUDE_count_pairs_eq_50_l2588_258850

/-- The number of pairs of positive integers (m,n) satisfying m^2 + mn < 30 -/
def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + p.1 * p.2 < 30) (Finset.product (Finset.range 30) (Finset.range 30))).card

/-- Theorem stating that the count of pairs satisfying the condition is 50 -/
theorem count_pairs_eq_50 : count_pairs = 50 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_50_l2588_258850


namespace NUMINAMATH_CALUDE_matching_probability_abe_bob_l2588_258817

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ := jb.green + jb.red + jb.yellow

/-- Represents the jelly beans held by Abe -/
def abe : JellyBeans := { green := 1, red := 1, yellow := 0 }

/-- Represents the jelly beans held by Bob -/
def bob : JellyBeans := { green := 1, red := 2, yellow := 1 }

/-- Calculates the probability of two people showing the same color jelly bean -/
def matchingProbability (person1 person2 : JellyBeans) : ℚ :=
  let greenProb := (person1.green : ℚ) / person1.total * (person2.green : ℚ) / person2.total
  let redProb := (person1.red : ℚ) / person1.total * (person2.red : ℚ) / person2.total
  greenProb + redProb

theorem matching_probability_abe_bob :
  matchingProbability abe bob = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_matching_probability_abe_bob_l2588_258817


namespace NUMINAMATH_CALUDE_inscribed_squares_area_ratio_l2588_258839

theorem inscribed_squares_area_ratio (r : ℝ) (r_pos : r > 0) :
  let semicircle_square_area := (4 / 5) * r^2
  let equilateral_triangle_side := 2 * r
  let triangle_square_area := r^2
  semicircle_square_area / triangle_square_area = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_area_ratio_l2588_258839


namespace NUMINAMATH_CALUDE_distinct_integer_parts_l2588_258864

theorem distinct_integer_parts (N : ℕ) (h : N > 1) :
  {α : ℝ | (∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ N → ⌊i * α⌋ ≠ ⌊j * α⌋) ∧
           (∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ N → ⌊i / α⌋ ≠ ⌊j / α⌋)} =
  {α : ℝ | (N - 1) / N ≤ α ∧ α ≤ N / (N - 1)} :=
sorry

end NUMINAMATH_CALUDE_distinct_integer_parts_l2588_258864


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_angle_ratio_l2588_258840

/-- A triangle with angles in the ratio 2:2:1 is isosceles -/
theorem isosceles_triangle_from_angle_ratio (A B C : ℝ) 
  (h_sum : A + B + C = 180) 
  (h_ratio : ∃ (k : ℝ), A = 2*k ∧ B = 2*k ∧ C = k) : 
  A = B ∨ B = C ∨ A = C := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_angle_ratio_l2588_258840


namespace NUMINAMATH_CALUDE_l_shaped_area_l2588_258877

/-- The area of an L-shaped region formed by subtracting two smaller squares
    from a larger square -/
theorem l_shaped_area (side_large : ℝ) (side_small1 : ℝ) (side_small2 : ℝ)
    (h1 : side_large = side_small1 + side_small2)
    (h2 : side_small1 = 4)
    (h3 : side_small2 = 2) :
    side_large^2 - (side_small1^2 + side_small2^2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_area_l2588_258877


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_120_3507_l2588_258800

theorem gcd_lcm_sum_120_3507 : 
  Nat.gcd 120 3507 + Nat.lcm 120 3507 = 140283 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_120_3507_l2588_258800


namespace NUMINAMATH_CALUDE_trajectory_eq_sufficient_not_necessary_l2588_258837

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The distance from a point to the x-axis -/
def distToXAxis (p : Point2D) : ℝ := |p.y|

/-- The distance from a point to the y-axis -/
def distToYAxis (p : Point2D) : ℝ := |p.x|

/-- A point has equal distance to both axes -/
def equalDistToAxes (p : Point2D) : Prop :=
  distToXAxis p = distToYAxis p

/-- The trajectory equation y = |x| -/
def trajectoryEq (p : Point2D) : Prop :=
  p.y = |p.x|

/-- Theorem: y = |x| is a sufficient but not necessary condition for equal distance to both axes -/
theorem trajectory_eq_sufficient_not_necessary :
  (∀ p : Point2D, trajectoryEq p → equalDistToAxes p) ∧
  (∃ p : Point2D, equalDistToAxes p ∧ ¬trajectoryEq p) :=
sorry

end NUMINAMATH_CALUDE_trajectory_eq_sufficient_not_necessary_l2588_258837


namespace NUMINAMATH_CALUDE_common_tangent_bisection_l2588_258856

-- Define the basic geometric objects
variable (Circle₁ Circle₂ : Type) [MetricSpace Circle₁] [MetricSpace Circle₂]
variable (A B : ℝ × ℝ)  -- Intersection points of the circles
variable (M N : ℝ × ℝ)  -- Points of tangency on the common tangent

-- Define the property of being a point on a circle
def OnCircle (p : ℝ × ℝ) (circle : Type) [MetricSpace circle] : Prop := sorry

-- Define the property of being a tangent line to a circle
def IsTangent (p q : ℝ × ℝ) (circle : Type) [MetricSpace circle] : Prop := sorry

-- Define the property of a line bisecting another line segment
def Bisects (p q r s : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem common_tangent_bisection 
  (hA₁ : OnCircle A Circle₁) (hA₂ : OnCircle A Circle₂)
  (hB₁ : OnCircle B Circle₁) (hB₂ : OnCircle B Circle₂)
  (hM₁ : OnCircle M Circle₁) (hN₂ : OnCircle N Circle₂)
  (hMN₁ : IsTangent M N Circle₁) (hMN₂ : IsTangent M N Circle₂) :
  Bisects A B M N := by sorry

end NUMINAMATH_CALUDE_common_tangent_bisection_l2588_258856


namespace NUMINAMATH_CALUDE_total_cost_is_3200_cents_l2588_258803

/-- Represents the number of shirt boxes that can be wrapped with one roll of paper -/
def shirt_boxes_per_roll : ℕ := 5

/-- Represents the number of XL boxes that can be wrapped with one roll of paper -/
def xl_boxes_per_roll : ℕ := 3

/-- Represents the number of shirt boxes Harold needs to wrap -/
def total_shirt_boxes : ℕ := 20

/-- Represents the number of XL boxes Harold needs to wrap -/
def total_xl_boxes : ℕ := 12

/-- Represents the cost of one roll of wrapping paper in cents -/
def cost_per_roll : ℕ := 400

/-- Theorem stating that the total cost for Harold to wrap all boxes is $32.00 -/
theorem total_cost_is_3200_cents : 
  (((total_shirt_boxes + shirt_boxes_per_roll - 1) / shirt_boxes_per_roll) + 
   ((total_xl_boxes + xl_boxes_per_roll - 1) / xl_boxes_per_roll)) * 
  cost_per_roll = 3200 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_3200_cents_l2588_258803


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2588_258859

/-- The equation of a line passing through the intersection of two lines and perpendicular to a third line -/
theorem perpendicular_line_equation (a b c d e f g h i j : ℝ) :
  let l₁ : ℝ × ℝ → Prop := λ p => a * p.1 + b * p.2 = 0
  let l₂ : ℝ × ℝ → Prop := λ p => c * p.1 + d * p.2 + e = 0
  let l₃ : ℝ × ℝ → Prop := λ p => f * p.1 + g * p.2 + h = 0
  let l₄ : ℝ × ℝ → Prop := λ p => i * p.1 + j * p.2 + 5 = 0
  (∃! p, l₁ p ∧ l₂ p) →  -- l₁ and l₂ intersect at a unique point
  (∀ p q : ℝ × ℝ, l₃ p ∧ l₃ q → (p.1 - q.1) * (f * (p.1 - q.1) + g * (p.2 - q.2)) + (p.2 - q.2) * (g * (p.1 - q.1) - f * (p.2 - q.2)) = 0) →  -- l₄ is perpendicular to l₃
  (a = 3 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = -2 ∧ f = 2 ∧ g = 1 ∧ h = 3 ∧ i = 1 ∧ j = -2) →
  ∀ p, l₁ p ∧ l₂ p → l₄ p  -- The point of intersection of l₁ and l₂ satisfies l₄
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2588_258859


namespace NUMINAMATH_CALUDE_complement_of_16_51_l2588_258831

/-- Represents an angle in degrees and minutes -/
structure DegreeMinute where
  degrees : ℕ
  minutes : ℕ

/-- Calculates the complement of an angle given in degrees and minutes -/
def complement (angle : DegreeMinute) : DegreeMinute :=
  let totalMinutes := 90 * 60 - (angle.degrees * 60 + angle.minutes)
  { degrees := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem complement_of_16_51 :
  complement { degrees := 16, minutes := 51 } = { degrees := 73, minutes := 9 } := by
  sorry

end NUMINAMATH_CALUDE_complement_of_16_51_l2588_258831


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2588_258809

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x| ≥ 1}
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.univ \ A) ∩ (Set.univ \ B) = {x : ℝ | -1 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2588_258809


namespace NUMINAMATH_CALUDE_rectangle_width_proof_l2588_258883

/-- Proves that the width of a rectangle is 14 cm given specific conditions -/
theorem rectangle_width_proof (length width perimeter : ℝ) (triangle_side : ℝ) : 
  length = 10 →
  perimeter = 2 * (length + width) →
  perimeter = 3 * triangle_side →
  triangle_side = 16 →
  width = 14 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_proof_l2588_258883


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2588_258894

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 5) / 10 = 5 / (x - 10) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2588_258894


namespace NUMINAMATH_CALUDE_zero_point_implies_a_range_l2588_258857

/-- The function f(x) = x^2 + x - 2a has a zero point in the interval (-1, 1) if and only if a ∈ [-1/8, 1) -/
theorem zero_point_implies_a_range (a : ℝ) : 
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 + x - 2*a = 0) ↔ -1/8 ≤ a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_point_implies_a_range_l2588_258857


namespace NUMINAMATH_CALUDE_no_solution_inequality_l2588_258886

theorem no_solution_inequality (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 5| + |x + 3| < a)) → a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_inequality_l2588_258886


namespace NUMINAMATH_CALUDE_tuesday_temperature_l2588_258867

theorem tuesday_temperature
  (temp_tue wed thu fri : ℝ)
  (h1 : (temp_tue + wed + thu) / 3 = 45)
  (h2 : (wed + thu + fri) / 3 = 50)
  (h3 : fri = 53) :
  temp_tue = 38 := by
sorry

end NUMINAMATH_CALUDE_tuesday_temperature_l2588_258867


namespace NUMINAMATH_CALUDE_inverse_proportion_through_point_l2588_258895

/-- An inverse proportion function passing through the point (2,5) has k = 10 -/
theorem inverse_proportion_through_point (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (k / x = 5 ↔ x = 2)) → k = 10 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_through_point_l2588_258895


namespace NUMINAMATH_CALUDE_roses_ratio_l2588_258810

theorem roses_ratio (roses_day1 : ℕ) (roses_day2 : ℕ) (roses_day3 : ℕ) 
  (h1 : roses_day1 = 50)
  (h2 : roses_day2 = roses_day1 + 20)
  (h3 : roses_day1 + roses_day2 + roses_day3 = 220) :
  roses_day3 / roses_day1 = 2 := by
sorry

end NUMINAMATH_CALUDE_roses_ratio_l2588_258810


namespace NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l2588_258846

theorem hcf_from_lcm_and_product (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 750)
  (h_product : a * b = 18750) : 
  Nat.gcd a b = 25 := by
  sorry

end NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l2588_258846


namespace NUMINAMATH_CALUDE_factorial_ratio_simplification_l2588_258825

theorem factorial_ratio_simplification :
  (11 * Nat.factorial 10 * Nat.factorial 7 * Nat.factorial 3) / 
  (Nat.factorial 10 * Nat.factorial 8) = 11 / 56 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_simplification_l2588_258825


namespace NUMINAMATH_CALUDE_line_through_points_l2588_258876

/-- The general form equation of a line passing through two points -/
def general_form_equation (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ (a b c : ℝ), a * x + b * y + c = 0 ∧
    a * x₁ + b * y₁ + c = 0 ∧
    a * x₂ + b * y₂ + c = 0 ∧
    (a ≠ 0 ∨ b ≠ 0)}

/-- Theorem: The general form equation of the line passing through (1, 1) and (-2, 4) is x + y - 2 = 0 -/
theorem line_through_points : 
  general_form_equation 1 1 (-2) 4 = {(x, y) | x + y - 2 = 0} := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2588_258876


namespace NUMINAMATH_CALUDE_negation_existence_statement_l2588_258866

theorem negation_existence_statement :
  (¬ ∃ x : ℝ, x < 0 ∧ x^2 > 0) ↔ (∀ x : ℝ, x < 0 → x^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_existence_statement_l2588_258866


namespace NUMINAMATH_CALUDE_inscribing_square_area_l2588_258816

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x - 6*y + 24 = 0

-- Define the square that inscribes the circle
structure InscribingSquare :=
  (side_length : ℝ)
  (parallel_to_x_axis : Prop)
  (inscribes_circle : Prop)

-- Theorem statement
theorem inscribing_square_area
  (square : InscribingSquare)
  (h_circle : ∀ x y, circle_equation x y ↔ (x - 4)^2 + (y - 3)^2 = 1) :
  square.side_length^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_inscribing_square_area_l2588_258816


namespace NUMINAMATH_CALUDE_container_capacity_sum_l2588_258896

/-- Represents the capacity and fill levels of a container -/
structure Container where
  capacity : ℝ
  initial_fill : ℝ
  final_fill : ℝ
  added_water : ℝ

/-- Calculates the total capacity of three containers -/
def total_capacity (a b c : Container) : ℝ :=
  a.capacity + b.capacity + c.capacity

/-- The problem statement -/
theorem container_capacity_sum : 
  ∃ (a b c : Container),
    a.initial_fill = 0.3 * a.capacity ∧
    a.final_fill = 0.75 * a.capacity ∧
    a.added_water = 36 ∧
    b.initial_fill = 0.4 * b.capacity ∧
    b.final_fill = 0.7 * b.capacity ∧
    b.added_water = 20 ∧
    c.initial_fill = 0.5 * c.capacity ∧
    c.final_fill = 2/3 * c.capacity ∧
    c.added_water = 12 ∧
    total_capacity a b c = 218.6666666666667 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_sum_l2588_258896


namespace NUMINAMATH_CALUDE_valentines_day_theorem_l2588_258873

/-- The number of valentines given on Valentine's Day -/
def valentines_given (male_students female_students : ℕ) : ℕ :=
  male_students * female_students

/-- The total number of students -/
def total_students (male_students female_students : ℕ) : ℕ :=
  male_students + female_students

/-- Theorem stating the number of valentines given -/
theorem valentines_day_theorem (male_students female_students : ℕ) :
  valentines_given male_students female_students = 
  total_students male_students female_students + 22 →
  valentines_given male_students female_students = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_valentines_day_theorem_l2588_258873


namespace NUMINAMATH_CALUDE_area_ratio_of_angle_bisector_l2588_258884

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the lengths of the sides
def side_length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the angle bisector
def is_angle_bisector (X P Y Z : ℝ × ℝ) : Prop := sorry

-- Define the area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

theorem area_ratio_of_angle_bisector (XYZ : Triangle) (P : ℝ × ℝ) :
  side_length XYZ.X XYZ.Y = 20 →
  side_length XYZ.X XYZ.Z = 30 →
  side_length XYZ.Y XYZ.Z = 26 →
  is_angle_bisector XYZ.X P XYZ.Y XYZ.Z →
  (triangle_area XYZ.X XYZ.Y P) / (triangle_area XYZ.X XYZ.Z P) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_angle_bisector_l2588_258884


namespace NUMINAMATH_CALUDE_people_got_on_second_stop_is_two_l2588_258875

/-- The number of people who got on at the second stop of a bus journey -/
def people_got_on_second_stop : ℕ :=
  let initial_people : ℕ := 50
  let first_stop_off : ℕ := 15
  let second_stop_off : ℕ := 8
  let third_stop_off : ℕ := 4
  let third_stop_on : ℕ := 3
  let final_people : ℕ := 28
  initial_people - first_stop_off - second_stop_off + 
    (final_people - (initial_people - first_stop_off - second_stop_off - third_stop_off + third_stop_on))

theorem people_got_on_second_stop_is_two : 
  people_got_on_second_stop = 2 := by sorry

end NUMINAMATH_CALUDE_people_got_on_second_stop_is_two_l2588_258875


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2588_258838

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 + a 8 = 1/2 →
  a 6 * (a 2 + 2 * a 6 + a 10) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2588_258838


namespace NUMINAMATH_CALUDE_binomial_150_150_l2588_258832

theorem binomial_150_150 : (150 : ℕ).choose 150 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_150_150_l2588_258832


namespace NUMINAMATH_CALUDE_perpendicular_condition_l2588_258851

theorem perpendicular_condition (x : ℝ) :
  let a : ℝ × ℝ := (1, 2*x)
  let b : ℝ × ℝ := (4, -x)
  (x = Real.sqrt 2 → a.1 * b.1 + a.2 * b.2 = 0) ∧
  ¬(a.1 * b.1 + a.2 * b.2 = 0 → x = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l2588_258851


namespace NUMINAMATH_CALUDE_intersection_and_union_of_sets_l2588_258849

def A (p : ℝ) : Set ℝ := {x | 2 * x^2 + 3 * p * x + 2 = 0}
def B (q : ℝ) : Set ℝ := {x | 2 * x^2 + x + q = 0}

theorem intersection_and_union_of_sets (p q : ℝ) :
  A p ∩ B q = {1/2} →
  p = -5/3 ∧ q = -1 ∧ A p ∪ B q = {-1, 1/2, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_union_of_sets_l2588_258849


namespace NUMINAMATH_CALUDE_fifty_men_left_l2588_258820

/-- Represents the scenario of a hostel with changing occupancy and food provisions. -/
structure Hostel where
  initialMen : ℕ
  initialDays : ℕ
  finalDays : ℕ

/-- Calculates the number of men who left the hostel based on the change in provision duration. -/
def menWhoLeft (h : Hostel) : ℕ :=
  h.initialMen - (h.initialMen * h.initialDays) / h.finalDays

/-- Theorem stating that in the given hostel scenario, 50 men left. -/
theorem fifty_men_left (h : Hostel)
  (h_initial_men : h.initialMen = 250)
  (h_initial_days : h.initialDays = 36)
  (h_final_days : h.finalDays = 45) :
  menWhoLeft h = 50 := by
  sorry

end NUMINAMATH_CALUDE_fifty_men_left_l2588_258820


namespace NUMINAMATH_CALUDE_rabbit_problem_l2588_258813

/-- The cost price of an Auspicious Rabbit -/
def auspicious_cost : ℝ := 40

/-- The cost price of a Lucky Rabbit -/
def lucky_cost : ℝ := 44

/-- The selling price of an Auspicious Rabbit -/
def auspicious_price : ℝ := 60

/-- The selling price of a Lucky Rabbit -/
def lucky_price : ℝ := 70

/-- The total number of rabbits to be purchased -/
def total_rabbits : ℕ := 200

/-- The minimum required profit -/
def min_profit : ℝ := 4120

/-- The quantity ratio of Lucky Rabbits to Auspicious Rabbits based on the given costs -/
axiom quantity_ratio : (8800 / lucky_cost) = 2 * (4000 / auspicious_cost)

/-- The cost difference between Lucky and Auspicious Rabbits -/
axiom cost_difference : lucky_cost = auspicious_cost + 4

/-- Theorem stating the correct cost prices and minimum number of Lucky Rabbits -/
theorem rabbit_problem :
  (auspicious_cost = 40 ∧ lucky_cost = 44) ∧
  (∀ m : ℕ, m ≥ 20 →
    (lucky_price - lucky_cost) * m + (auspicious_price - auspicious_cost) * (total_rabbits - m) ≥ min_profit) ∧
  (∀ m : ℕ, m < 20 →
    (lucky_price - lucky_cost) * m + (auspicious_price - auspicious_cost) * (total_rabbits - m) < min_profit) :=
sorry

end NUMINAMATH_CALUDE_rabbit_problem_l2588_258813


namespace NUMINAMATH_CALUDE_odd_function_condition_l2588_258898

/-- Given a > 1, f(x) = (a^x / (a^x - 1)) + m is an odd function if and only if m = -1/2 -/
theorem odd_function_condition (a : ℝ) (h : a > 1) :
  ∃ m : ℝ, ∀ x : ℝ, x ≠ 0 →
    (fun x : ℝ => (a^x / (a^x - 1)) + m) x = -((fun x : ℝ => (a^x / (a^x - 1)) + m) (-x)) ↔
    m = -1/2 := by sorry

end NUMINAMATH_CALUDE_odd_function_condition_l2588_258898


namespace NUMINAMATH_CALUDE_second_concert_attendance_l2588_258885

theorem second_concert_attendance 
  (first_concert : Nat) 
  (attendance_increase : Nat) 
  (h1 : first_concert = 65899)
  (h2 : attendance_increase = 119) :
  first_concert + attendance_increase = 66018 := by
  sorry

end NUMINAMATH_CALUDE_second_concert_attendance_l2588_258885


namespace NUMINAMATH_CALUDE_chess_tournament_theorem_l2588_258804

/-- Represents a participant in the chess tournament -/
structure Participant :=
  (id : Nat)

/-- Represents the results of the chess tournament -/
structure TournamentResult :=
  (participants : Finset Participant)
  (white_wins : Participant → Nat)
  (black_wins : Participant → Nat)

/-- Defines the "no weaker than" relation between two participants -/
def no_weaker_than (result : TournamentResult) (a b : Participant) : Prop :=
  result.white_wins a ≥ result.white_wins b ∧ result.black_wins a ≥ result.black_wins b

theorem chess_tournament_theorem :
  ∀ (result : TournamentResult),
    result.participants.card = 20 →
    (∀ p q : Participant, p ∈ result.participants → q ∈ result.participants → p ≠ q →
      result.white_wins p + result.white_wins q = 1 ∧
      result.black_wins p + result.black_wins q = 1) →
    ∃ a b : Participant, a ∈ result.participants ∧ b ∈ result.participants ∧ a ≠ b ∧
      no_weaker_than result a b := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_theorem_l2588_258804


namespace NUMINAMATH_CALUDE_basketball_win_percentage_l2588_258871

theorem basketball_win_percentage (total_games : ℕ) (first_games : ℕ) (first_wins : ℕ) (remaining_games : ℕ) (remaining_wins : ℕ) :
  total_games = first_games + remaining_games →
  first_games = 55 →
  first_wins = 45 →
  remaining_games = 50 →
  remaining_wins = 34 →
  (first_wins + remaining_wins : ℚ) / total_games = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_basketball_win_percentage_l2588_258871


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2588_258833

theorem hyperbola_eccentricity_range (a : ℝ) (h : a > 1) :
  let e := Real.sqrt (1 + ((a + 1) / a) ^ 2)
  Real.sqrt 2 < e ∧ e < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2588_258833


namespace NUMINAMATH_CALUDE_smallest_x_value_l2588_258802

theorem smallest_x_value (x y : ℝ) 
  (hx : 4 < x ∧ x < 8) 
  (hy : 8 < y ∧ y < 12) 
  (h_diff : ∃ (n : ℕ), n = 7 ∧ n = ⌊y - x⌋) : 
  4 < x :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2588_258802


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2588_258835

/-- Prove that for an ellipse with the given properties, its eccentricity is 2/3 -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}
  let F := (Real.sqrt (a^2 - b^2), 0)
  let l := {(x, y) : ℝ × ℝ | y = Real.sqrt 3 * (x - F.1)}
  ∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ C ∧ A ∈ l ∧ B ∈ l ∧
    (A.2 < 0 ∧ B.2 > 0) ∧ 
    (-A.2 = 2 * B.2) →
  (Real.sqrt (a^2 - b^2)) / a = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2588_258835


namespace NUMINAMATH_CALUDE_interest_rate_is_four_percent_l2588_258887

/-- Given a loan with simple interest, prove that the interest rate is 4% per annum -/
theorem interest_rate_is_four_percent
  (P : ℚ) -- Principal amount
  (t : ℚ) -- Time in years
  (I : ℚ) -- Interest amount
  (h1 : P = 250) -- Sum lent is Rs. 250
  (h2 : t = 8) -- Time period is 8 years
  (h3 : I = P - 170) -- Interest is Rs. 170 less than sum lent
  (h4 : I = P * r * t / 100) -- Simple interest formula
  : r = 4 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_four_percent_l2588_258887


namespace NUMINAMATH_CALUDE_inequality_always_true_l2588_258855

theorem inequality_always_true (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) :
  1 / (a * b^2) < 1 / (a^2 * b) := by
sorry

end NUMINAMATH_CALUDE_inequality_always_true_l2588_258855


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2588_258893

theorem quadratic_two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x - a = 0 ∧ y^2 - 2*y - a = 0) ↔ a > -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2588_258893


namespace NUMINAMATH_CALUDE_julia_tag_kids_l2588_258806

/-- The number of kids Julia played tag with on Monday -/
def monday_kids : ℕ := 7

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_kids : ℕ := 13

/-- The total number of kids Julia played tag with -/
def total_kids : ℕ := monday_kids + tuesday_kids

theorem julia_tag_kids : total_kids = 20 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_kids_l2588_258806


namespace NUMINAMATH_CALUDE_cost_of_chips_l2588_258882

/-- The cost of chips when three friends split the bill equally -/
theorem cost_of_chips (num_friends : ℕ) (num_bags : ℕ) (payment_per_friend : ℚ) : 
  num_friends = 3 → num_bags = 5 → payment_per_friend = 5 →
  (num_friends * payment_per_friend) / num_bags = 3 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_chips_l2588_258882


namespace NUMINAMATH_CALUDE_not_all_positive_k_real_roots_not_all_negative_k_nonzero_im_not_all_real_k_not_pure_imaginary_l2588_258834

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (z k : ℂ) : Prop := 10 * z^2 - 7 * i * z - k = 0

-- Statement A is false
theorem not_all_positive_k_real_roots :
  ¬ ∀ (k : ℝ), k > 0 → ∀ (z : ℂ), equation z k → z.im = 0 :=
sorry

-- Statement B is false
theorem not_all_negative_k_nonzero_im :
  ¬ ∀ (k : ℝ), k < 0 → ∀ (z : ℂ), equation z k → z.im ≠ 0 :=
sorry

-- Statement C is false
theorem not_all_real_k_not_pure_imaginary :
  ¬ ∀ (k : ℝ), ∀ (z : ℂ), equation z k → z.re ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_not_all_positive_k_real_roots_not_all_negative_k_nonzero_im_not_all_real_k_not_pure_imaginary_l2588_258834


namespace NUMINAMATH_CALUDE_percentage_increase_l2588_258888

theorem percentage_increase (original : ℝ) (difference : ℝ) (increase : ℝ) : 
  original = 80 →
  original + (increase / 100) * original - (original - 25 / 100 * original) = difference →
  difference = 30 →
  increase = 12.5 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l2588_258888


namespace NUMINAMATH_CALUDE_incorrect_equation_is_false_l2588_258858

/-- Represents the number of 1-yuan stamps purchased -/
def x : ℕ := sorry

/-- The total number of stamps purchased -/
def total_stamps : ℕ := 12

/-- The total amount spent in yuan -/
def total_spent : ℕ := 20

/-- The equation representing the correct relationship between x, total stamps, and total spent -/
def correct_equation : Prop := x + 2 * (total_stamps - x) = total_spent

/-- The incorrect equation to be proven false -/
def incorrect_equation : Prop := 2 * (total_stamps - x) - total_spent = x

theorem incorrect_equation_is_false :
  correct_equation → ¬incorrect_equation := by sorry

end NUMINAMATH_CALUDE_incorrect_equation_is_false_l2588_258858


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2588_258815

theorem complex_number_quadrant (z : ℂ) : z - Complex.I = Complex.abs (1 + 2 * Complex.I) → 
  z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2588_258815


namespace NUMINAMATH_CALUDE_base_prime_rep_360_l2588_258880

def base_prime_representation (n : ℕ) : List ℕ := sorry

theorem base_prime_rep_360 :
  base_prime_representation 360 = [3, 2, 0, 1] :=
by
  sorry

end NUMINAMATH_CALUDE_base_prime_rep_360_l2588_258880


namespace NUMINAMATH_CALUDE_reduced_banana_price_l2588_258872

/-- Given a 60% reduction in banana prices and the ability to obtain 120 more bananas
    for Rs. 150 after the reduction, prove that the reduced price per dozen bananas
    is Rs. 48/17. -/
theorem reduced_banana_price (P : ℚ) : 
  (150 / (0.4 * P) = 150 / P + 120) →
  (12 * (0.4 * P) = 48 / 17) :=
by sorry

end NUMINAMATH_CALUDE_reduced_banana_price_l2588_258872


namespace NUMINAMATH_CALUDE_notebook_statements_l2588_258827

theorem notebook_statements :
  ∃! n : Fin 40, (∀ m : Fin 40, (m.val + 1 = n.val) ↔ (m = n)) ∧ n.val = 39 :=
sorry

end NUMINAMATH_CALUDE_notebook_statements_l2588_258827


namespace NUMINAMATH_CALUDE_skittles_division_l2588_258828

theorem skittles_division (total_skittles : Nat) (num_groups : Nat) (group_size : Nat) :
  total_skittles = 5929 →
  num_groups = 77 →
  total_skittles = num_groups * group_size →
  group_size = 77 := by
sorry

end NUMINAMATH_CALUDE_skittles_division_l2588_258828


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_quadratic_equation_other_root_l2588_258889

/-- The quadratic equation x^2 - 2x + m - 1 = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 - 2*x + m - 1 = 0

theorem quadratic_equation_roots (m : ℝ) :
  (∃ x : ℝ, ∀ y : ℝ, quadratic_equation y m ↔ y = x) →
  m = 2 :=
sorry

theorem quadratic_equation_other_root (m : ℝ) :
  (quadratic_equation 5 m) →
  (∃ x : ℝ, x ≠ 5 ∧ quadratic_equation x m ∧ x = -3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_quadratic_equation_other_root_l2588_258889


namespace NUMINAMATH_CALUDE_jake_lawn_mowing_earnings_l2588_258878

/-- Jake's desired hourly rate in dollars -/
def desired_hourly_rate : ℝ := 20

/-- Time taken to mow the lawn in hours -/
def lawn_mowing_time : ℝ := 1

/-- Time taken to plant flowers in hours -/
def flower_planting_time : ℝ := 2

/-- Total charge for planting flowers in dollars -/
def flower_planting_charge : ℝ := 45

/-- Earnings for mowing the lawn in dollars -/
def lawn_mowing_earnings : ℝ := desired_hourly_rate * lawn_mowing_time

theorem jake_lawn_mowing_earnings :
  lawn_mowing_earnings = 20 := by sorry

end NUMINAMATH_CALUDE_jake_lawn_mowing_earnings_l2588_258878


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2588_258869

theorem trigonometric_identity (α : Real) 
  (h : (1 + Real.tan α) / (1 - Real.tan α) = 2016) : 
  1 / Real.cos (2 * α) + Real.tan (2 * α) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2588_258869


namespace NUMINAMATH_CALUDE_parallel_line_to_plane_parallel_lines_in_intersecting_planes_l2588_258814

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (intersects : Plane → Plane → Line → Prop)

-- Theorem 1
theorem parallel_line_to_plane 
  (α β : Plane) (m : Line) 
  (h1 : parallel_plane α β) 
  (h2 : contains α m) : 
  parallel_plane_line β m :=
sorry

-- Theorem 2
theorem parallel_lines_in_intersecting_planes 
  (α β : Plane) (m n : Line)
  (h1 : parallel_plane_line β m)
  (h2 : contains α m)
  (h3 : intersects α β n) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_parallel_line_to_plane_parallel_lines_in_intersecting_planes_l2588_258814


namespace NUMINAMATH_CALUDE_equilateral_perimeter_is_60_l2588_258862

/-- An equilateral triangle with a side shared with an isosceles triangle -/
structure TrianglePair where
  equilateral_side : ℝ
  isosceles_base : ℝ
  isosceles_perimeter : ℝ
  equilateral_side_positive : 0 < equilateral_side
  isosceles_base_positive : 0 < isosceles_base
  isosceles_perimeter_positive : 0 < isosceles_perimeter

/-- The perimeter of the equilateral triangle in the TrianglePair -/
def equilateral_perimeter (tp : TrianglePair) : ℝ := 3 * tp.equilateral_side

/-- Theorem: The perimeter of the equilateral triangle is 60 -/
theorem equilateral_perimeter_is_60 (tp : TrianglePair)
  (h1 : tp.isosceles_base = 15)
  (h2 : tp.isosceles_perimeter = 55) :
  equilateral_perimeter tp = 60 := by
  sorry

#check equilateral_perimeter_is_60

end NUMINAMATH_CALUDE_equilateral_perimeter_is_60_l2588_258862


namespace NUMINAMATH_CALUDE_line_through_points_l2588_258890

/-- Given three points on a line, find the y-coordinate of a fourth point on the same line -/
theorem line_through_points (x1 y1 x2 y2 x3 y3 x4 : ℝ) (h1 : y2 - y1 = (x2 - x1) * ((y3 - y1) / (x3 - x1))) 
  (h2 : y3 - y2 = (x3 - x2) * ((y3 - y1) / (x3 - x1))) : 
  let t := y1 + (x4 - x1) * ((y3 - y1) / (x3 - x1))
  (x1 = 2 ∧ y1 = 6 ∧ x2 = 5 ∧ y2 = 12 ∧ x3 = 8 ∧ y3 = 18 ∧ x4 = 20) → t = 42 := by
  sorry


end NUMINAMATH_CALUDE_line_through_points_l2588_258890


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2588_258822

theorem unique_solution_for_equation (x r p n : ℕ+) : 
  (x ^ r.val - 1 = p ^ n.val) ∧ 
  (Nat.Prime p.val) ∧ 
  (r.val ≥ 2) ∧ 
  (n.val ≥ 2) → 
  (x = 3 ∧ r = 2 ∧ p = 2 ∧ n = 3) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2588_258822
