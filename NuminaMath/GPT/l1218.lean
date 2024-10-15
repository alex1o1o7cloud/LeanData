import Mathlib

namespace NUMINAMATH_GPT_sock_pairs_l1218_121871

theorem sock_pairs (n : ℕ) (h : ((2 * n) * (2 * n - 1)) / 2 = 90) : n = 10 :=
sorry

end NUMINAMATH_GPT_sock_pairs_l1218_121871


namespace NUMINAMATH_GPT_tetrahedron_cd_length_l1218_121878

theorem tetrahedron_cd_length (a b c d : Type) [MetricSpace a] [MetricSpace b] [MetricSpace c] [MetricSpace d] :
  let ab := 53
  let edge_lengths := [17, 23, 29, 39, 46, 53]
  ∃ cd, cd = 17 :=
by
  sorry

end NUMINAMATH_GPT_tetrahedron_cd_length_l1218_121878


namespace NUMINAMATH_GPT_score_difference_l1218_121858

theorem score_difference (chuck_score red_score : ℕ) (h1 : chuck_score = 95) (h2 : red_score = 76) : chuck_score - red_score = 19 := by
  sorry

end NUMINAMATH_GPT_score_difference_l1218_121858


namespace NUMINAMATH_GPT_irreducible_fraction_l1218_121825

theorem irreducible_fraction {n : ℕ} : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by
  sorry

end NUMINAMATH_GPT_irreducible_fraction_l1218_121825


namespace NUMINAMATH_GPT_correct_expression_l1218_121894

variable (a b : ℝ)

theorem correct_expression : (∃ x, x = 3 * a + b^2) ∧ 
    (x = (3 * a + b)^2 ∨ x = 3 * (a + b)^2 ∨ x = 3 * a + b^2 ∨ x = (a + 3 * b)^2) → 
    x = 3 * a + b^2 := by sorry

end NUMINAMATH_GPT_correct_expression_l1218_121894


namespace NUMINAMATH_GPT_machine_makes_12_shirts_l1218_121886

def shirts_per_minute : ℕ := 2
def minutes_worked : ℕ := 6

def total_shirts_made : ℕ := shirts_per_minute * minutes_worked

theorem machine_makes_12_shirts :
  total_shirts_made = 12 :=
by
  -- proof placeholder
  sorry

end NUMINAMATH_GPT_machine_makes_12_shirts_l1218_121886


namespace NUMINAMATH_GPT_find_pairs_l1218_121811

theorem find_pairs (m n : ℕ) : 
  ∃ x : ℤ, x * x = 2^m * 3^n + 1 ↔ (m = 3 ∧ n = 1) ∨ (m = 4 ∧ n = 1) ∨ (m = 5 ∧ n = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l1218_121811


namespace NUMINAMATH_GPT_find_a_minus_b_l1218_121895

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b * x + 1

theorem find_a_minus_b (a b : ℝ)
  (h1 : deriv (f a b) 1 = -2)
  (h2 : deriv (f a b) (2 / 3) = 0) :
  a - b = 10 :=
sorry

end NUMINAMATH_GPT_find_a_minus_b_l1218_121895


namespace NUMINAMATH_GPT_common_chord_eq_l1218_121807

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 2*y - 40 = 0

-- Define the statement to prove
theorem common_chord_eq (x y : ℝ) : circle1 x y ∧ circle2 x y → 2*x + y - 5 = 0 :=
sorry

end NUMINAMATH_GPT_common_chord_eq_l1218_121807


namespace NUMINAMATH_GPT_distance_between_cities_l1218_121870

theorem distance_between_cities:
    ∃ (x y : ℝ),
    (x = 135) ∧
    (y = 175) ∧
    (7 / 9 * x = 105) ∧
    (x + 7 / 9 * x + y = 415) ∧
    (x = 27 / 35 * y) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_cities_l1218_121870


namespace NUMINAMATH_GPT_find_k_l1218_121804

theorem find_k (a : ℕ → ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h_nz : ∀ n, S n = n ^ 2 - a n) 
  (hSk : 1 < S k ∧ S k < 9) :
  k = 2 := 
sorry

end NUMINAMATH_GPT_find_k_l1218_121804


namespace NUMINAMATH_GPT_part1_part2_l1218_121864

variable {m n : ℤ}

theorem part1 (hm : |m| = 1) (hn : |n| = 4) (hprod : m * n < 0) : m + n = -3 ∨ m + n = 3 := sorry

theorem part2 (hm : |m| = 1) (hn : |n| = 4) : ∃ (k : ℤ), k = 5 ∧ ∀ x, x = m - n → x ≤ k := sorry

end NUMINAMATH_GPT_part1_part2_l1218_121864


namespace NUMINAMATH_GPT_minimum_value_of_x_plus_2y_l1218_121847

-- Definitions for the problem conditions
def isPositive (z : ℝ) : Prop := z > 0

def condition (x y : ℝ) : Prop := 
  isPositive x ∧ isPositive y ∧ (x + 2*y + 2*x*y = 8) 

-- Statement of the problem
theorem minimum_value_of_x_plus_2y (x y : ℝ) (h : condition x y) : x + 2 * y ≥ 4 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_x_plus_2y_l1218_121847


namespace NUMINAMATH_GPT_number_of_games_played_l1218_121883

-- Define our conditions
def teams : ℕ := 14
def games_per_pair : ℕ := 5

-- Define the function to calculate the number of combinations
def combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the expected total games
def total_games : ℕ := 455

-- Statement asserting that given the conditions, the number of games played in the season is total_games
theorem number_of_games_played : (combinations teams 2) * games_per_pair = total_games := 
by 
  sorry

end NUMINAMATH_GPT_number_of_games_played_l1218_121883


namespace NUMINAMATH_GPT_problem_solution_l1218_121844

-- Define the operation otimes
def otimes (x y : ℚ) : ℚ := (x * y) / (x + y / 3)

-- Define the specific values x and y
def x : ℚ := 4
def y : ℚ := 3/2 -- 1.5 in fraction form

-- Prove the mathematical statement
theorem problem_solution : (0.36 : ℚ) * (otimes x y) = 12 / 25 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1218_121844


namespace NUMINAMATH_GPT_find_n_l1218_121823

noncomputable def f (n : ℝ) : ℝ :=
  n ^ (n / 2)

example : f 2 = 2 := sorry

theorem find_n : ∃ n : ℝ, f n = 12 ∧ abs (n - 3.4641) < 0.0001 := sorry

end NUMINAMATH_GPT_find_n_l1218_121823


namespace NUMINAMATH_GPT_cars_with_air_bags_l1218_121882

/--
On a car lot with 65 cars:
- Some have air-bags.
- 30 have power windows.
- 12 have both air-bag and power windows.
- 2 have neither air-bag nor power windows.

Prove that the number of cars with air-bags is 45.
-/
theorem cars_with_air_bags 
    (total_cars : ℕ)
    (cars_with_power_windows : ℕ)
    (cars_with_both : ℕ)
    (cars_with_neither : ℕ)
    (total_cars_eq : total_cars = 65)
    (cars_with_power_windows_eq : cars_with_power_windows = 30)
    (cars_with_both_eq : cars_with_both = 12)
    (cars_with_neither_eq : cars_with_neither = 2) :
    ∃ (A : ℕ), A = 45 :=
by
  sorry

end NUMINAMATH_GPT_cars_with_air_bags_l1218_121882


namespace NUMINAMATH_GPT_evaluate_g_neg_1_l1218_121845

noncomputable def g (x : ℝ) : ℝ := -2 * x^2 + 5 * x - 7

theorem evaluate_g_neg_1 : g (-1) = -14 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_g_neg_1_l1218_121845


namespace NUMINAMATH_GPT_sine_double_angle_l1218_121812

theorem sine_double_angle (theta : ℝ) (h : Real.tan (theta + Real.pi / 4) = 2) : Real.sin (2 * theta) = 3 / 5 :=
sorry

end NUMINAMATH_GPT_sine_double_angle_l1218_121812


namespace NUMINAMATH_GPT_range_of_a_l1218_121838

theorem range_of_a (A B C : Set ℝ) (a : ℝ) :
  A = { x | -1 < x ∧ x < 4 } →
  B = { x | -5 < x ∧ x < (3 / 2) } →
  C = { x | (1 - 2 * a) < x ∧ x < (2 * a) } →
  (C ⊆ (A ∩ B)) →
  a ≤ (3 / 4) :=
by
  intros hA hB hC hSubset
  sorry

end NUMINAMATH_GPT_range_of_a_l1218_121838


namespace NUMINAMATH_GPT_total_animals_on_farm_l1218_121828

theorem total_animals_on_farm :
  let coop1 := 60
  let coop2 := 45
  let coop3 := 55
  let coop4 := 40
  let coop5 := 35
  let coop6 := 20
  let coop7 := 50
  let coop8 := 10
  let coop9 := 10
  let first_shed := 2 * 10
  let second_shed := 10
  let third_shed := 6
  let section1 := 15
  let section2 := 25
  let section3 := 2 * 15
  coop1 + coop2 + coop3 + coop4 + coop5 + coop6 + coop7 + coop8 + coop9 + first_shed + second_shed + third_shed + section1 + section2 + section3 = 431 :=
by
  sorry

end NUMINAMATH_GPT_total_animals_on_farm_l1218_121828


namespace NUMINAMATH_GPT_value_of_a_l1218_121856

noncomputable def A : Set ℝ := {x | x^2 - x - 2 < 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 5}

theorem value_of_a (a : ℝ) (h : A ⊆ B a) : -3 ≤ a ∧ a ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1218_121856


namespace NUMINAMATH_GPT_total_cost_is_correct_l1218_121879

noncomputable def nights : ℕ := 3
noncomputable def cost_per_night : ℕ := 250
noncomputable def discount : ℕ := 100

theorem total_cost_is_correct :
  (nights * cost_per_night) - discount = 650 := by
sorry

end NUMINAMATH_GPT_total_cost_is_correct_l1218_121879


namespace NUMINAMATH_GPT_find_x_l1218_121809

noncomputable def x (n : ℕ) := 6^n + 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_three_prime_divisors (x : ℕ) : Prop :=
  ∃ a b c : ℕ, (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ Prime a ∧ Prime b ∧ Prime c ∧ a * b * c ∣ x ∧ ∀ d, Prime d ∧ d ∣ x → d = a ∨ d = b ∨ d = c

theorem find_x (n : ℕ) (hodd : is_odd n) (hdiv : has_three_prime_divisors (x n)) (hprime : 11 ∣ (x n)) : x n = 7777 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l1218_121809


namespace NUMINAMATH_GPT_samantha_routes_l1218_121865

-- Definitions of the conditions
def blocks_west_to_sw_corner := 3
def blocks_south_to_sw_corner := 2
def blocks_east_to_school := 4
def blocks_north_to_school := 3
def ways_house_to_sw_corner : ℕ := Nat.choose (blocks_west_to_sw_corner + blocks_south_to_sw_corner) blocks_south_to_sw_corner
def ways_through_park : ℕ := 2
def ways_ne_corner_to_school : ℕ := Nat.choose (blocks_east_to_school + blocks_north_to_school) blocks_north_to_school

-- The proof statement
theorem samantha_routes : (ways_house_to_sw_corner * ways_through_park * ways_ne_corner_to_school) = 700 :=
by
  -- Using "sorry" as a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_samantha_routes_l1218_121865


namespace NUMINAMATH_GPT_distribution_ways_l1218_121846

theorem distribution_ways (books students : ℕ) (h_books : books = 6) (h_students : students = 6) :
  ∃ ways : ℕ, ways = 6 * 5^6 ∧ ways = 93750 :=
by
  sorry

end NUMINAMATH_GPT_distribution_ways_l1218_121846


namespace NUMINAMATH_GPT_mark_money_l1218_121816

theorem mark_money (M : ℝ) (h1 : M / 2 + 14 ≤ M) (h2 : M / 3 + 16 ≤ M) :
  M - (M / 2 + 14) - (M / 3 + 16) = 0 → M = 180 := by
  sorry

end NUMINAMATH_GPT_mark_money_l1218_121816


namespace NUMINAMATH_GPT_least_non_lucky_multiple_of_10_l1218_121827

def is_lucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

theorem least_non_lucky_multiple_of_10 : 
  ∃ n : ℕ, n % 10 = 0 ∧ ¬is_lucky n ∧ (∀ m : ℕ, m % 10 = 0 ∧ ¬is_lucky m → m ≥ n) ∧ n = 110 :=
by
  sorry

end NUMINAMATH_GPT_least_non_lucky_multiple_of_10_l1218_121827


namespace NUMINAMATH_GPT_option_C_correct_l1218_121801

theorem option_C_correct (a b : ℝ) : 
  (1 / (b / a) * (a / b) = a^2 / b^2) :=
sorry

end NUMINAMATH_GPT_option_C_correct_l1218_121801


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l1218_121861

theorem hyperbola_asymptotes (a b c : ℝ) (h : a > 0) (h_b_gt_0: b > 0) 
  (eqn1 : b = 2 * Real.sqrt 2 * a)
  (focal_distance : 2 * a = (2 * c)/3)
  (focal_length : c = 3 * a) : 
  (∀ x : ℝ, ∀ y : ℝ, (y = (2 * Real.sqrt 2) * x) ∨ (y = -(2 * Real.sqrt 2) * x)) := by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l1218_121861


namespace NUMINAMATH_GPT_speed_of_stream_l1218_121840

-- Definitions based on conditions
def boat_speed_still_water : ℕ := 24
def travel_time : ℕ := 4
def downstream_distance : ℕ := 112

-- Theorem statement
theorem speed_of_stream : 
  ∀ (v : ℕ), downstream_distance = travel_time * (boat_speed_still_water + v) → v = 4 :=
by
  intros v h
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_speed_of_stream_l1218_121840


namespace NUMINAMATH_GPT_lesser_solution_quadratic_l1218_121806

theorem lesser_solution_quadratic (x : ℝ) :
  x^2 + 9 * x - 22 = 0 → x = -11 ∨ x = 2 :=
sorry

end NUMINAMATH_GPT_lesser_solution_quadratic_l1218_121806


namespace NUMINAMATH_GPT_geometric_seq_prod_l1218_121831

-- Conditions: Geometric sequence and given value of a_1 * a_7 * a_13
variables {a : ℕ → ℝ}
variable (r : ℝ)

-- Definition of a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- The proof problem
theorem geometric_seq_prod (h_geo : geometric_sequence a r) (h_prod : a 1 * a 7 * a 13 = 8) :
  a 3 * a 11 = 4 :=
sorry

end NUMINAMATH_GPT_geometric_seq_prod_l1218_121831


namespace NUMINAMATH_GPT_average_members_remaining_l1218_121839

theorem average_members_remaining :
  let initial_members := [7, 8, 10, 13, 6, 10, 12, 9]
  let members_leaving := [1, 2, 1, 2, 1, 2, 1, 2]
  let remaining_members := List.map (λ (x, y) => x - y) (List.zip initial_members members_leaving)
  let total_remaining := List.foldl Nat.add 0 remaining_members
  let num_families := initial_members.length
  total_remaining / num_families = 63 / 8 := by
    sorry

end NUMINAMATH_GPT_average_members_remaining_l1218_121839


namespace NUMINAMATH_GPT_translation_vector_coords_l1218_121843

-- Definitions according to the given conditions
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 1
def translated_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 1

-- Statement that we need to prove
theorem translation_vector_coords :
  ∃ (a b : ℝ), 
  (∀ x y : ℝ, original_circle x y ↔ translated_circle (x - a) (y - b)) ∧
  (a, b) = (-1, 2) := 
sorry

end NUMINAMATH_GPT_translation_vector_coords_l1218_121843


namespace NUMINAMATH_GPT_estimate_red_balls_l1218_121824

theorem estimate_red_balls (x : ℕ) (drawn_black_balls : ℕ) (total_draws : ℕ) (black_balls : ℕ) 
  (h1 : black_balls = 4) 
  (h2 : total_draws = 100) 
  (h3 : drawn_black_balls = 40) 
  (h4 : (black_balls : ℚ) / (black_balls + x) = drawn_black_balls / total_draws) : 
  x = 6 := 
sorry

end NUMINAMATH_GPT_estimate_red_balls_l1218_121824


namespace NUMINAMATH_GPT_range_of_a_l1218_121887

theorem range_of_a (a : ℚ) (h₀ : 0 < a) (h₁ : ∃ n : ℕ, (2 * n - 1 = 2007) ∧ (-a < n ∧ n < a)) :
  1003 < a ∧ a ≤ 1004 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1218_121887


namespace NUMINAMATH_GPT_total_sales_l1218_121830

noncomputable def sales_in_june : ℕ := 96
noncomputable def sales_in_july : ℕ := sales_in_june * 4 / 3

theorem total_sales (june_sales : ℕ) (july_sales : ℕ) (h1 : june_sales = 96)
                    (h2 : july_sales = june_sales * 4 / 3) :
                    june_sales + july_sales = 224 :=
by
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_total_sales_l1218_121830


namespace NUMINAMATH_GPT_a_2020_equality_l1218_121866

variables (n : ℤ)

def cube (x : ℤ) : ℤ := x * x * x

lemma a_six_n (n : ℤ) :
  cube (n + 1) + cube (n - 1) + cube (-n) + cube (-n) = 6 * n :=
sorry

lemma a_six_n_plus_one (n : ℤ) :
  cube (n + 1) + cube (n - 1) + cube (-n) + cube (-n) + 1 = 6 * n + 1 :=
sorry

lemma a_six_n_minus_one (n : ℤ) :
  cube (n + 1) + cube (n - 1) + cube (-n) + cube (-n) - 1 = 6 * n - 1 :=
sorry

lemma a_six_n_plus_two (n : ℤ) :
  cube n + cube (n - 2) + cube (-n + 1) + cube (-n + 1) + 8 = 6 * n + 2 :=
sorry

lemma a_six_n_minus_two (n : ℤ) :
  cube (n + 2) + cube n + cube (-n - 1) + cube (-n - 1) + (-8) = 6 * n - 2 :=
sorry

lemma a_six_n_plus_three (n : ℤ) :
  cube (n - 3) + cube (n - 5) + cube (-n + 4) + cube (-n + 4) + 27 = 6 * n + 3 :=
sorry

theorem a_2020_equality :
  2020 = cube 339 + cube 337 + cube (-338) + cube (-338) + cube (-2) :=
sorry

end NUMINAMATH_GPT_a_2020_equality_l1218_121866


namespace NUMINAMATH_GPT_law_firm_more_than_two_years_l1218_121821

theorem law_firm_more_than_two_years (p_second p_not_first : ℝ) : 
  p_second = 0.30 →
  p_not_first = 0.60 →
  ∃ p_more_than_two_years : ℝ, p_more_than_two_years = 0.30 :=
by
  intros h1 h2
  use (p_not_first - p_second)
  rw [h1, h2]
  norm_num
  done

end NUMINAMATH_GPT_law_firm_more_than_two_years_l1218_121821


namespace NUMINAMATH_GPT_sum_of_digits_of_gcd_l1218_121873

def gcd_of_differences : ℕ := Int.gcd (Int.gcd 3360 2240) 5600

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem sum_of_digits_of_gcd :
  sum_of_digits gcd_of_differences = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_gcd_l1218_121873


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1218_121849

variable (α : Real)
variable (h : Real.tan α = 1 / 2)

theorem problem_part1 : 
  (2 * Real.cos α - 3 * Real.sin α) / (3 * Real.cos α + 4 * Real.sin α) = 1 / 10 := sorry

theorem problem_part2 : 
  Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α + 4 * Real.cos α ^ 2 = 11 / 5 := sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1218_121849


namespace NUMINAMATH_GPT_find_a_b_find_m_l1218_121890

-- Define the parabola and the points it passes through
def parabola (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- The conditions based on the given problem
def condition1 (a b : ℝ) : Prop := parabola a b 1 = -2
def condition2 (a b : ℝ) : Prop := parabola a b (-2) = 13

-- Part 1: Proof for a and b
theorem find_a_b : ∃ a b : ℝ, condition1 a b ∧ condition2 a b ∧ a = 1 ∧ b = -4 :=
by sorry

-- Part 2: Given y equation and the specific points
def parabola2 (x : ℝ) : ℝ := x^2 - 4 * x + 1

-- Conditions for the second part
def condition3 : Prop := parabola2 5 = 6
def condition4 (m : ℝ) : Prop := parabola2 m = 12 - 6

-- Theorem statement for the second part
theorem find_m : ∃ m : ℝ, condition3 ∧ condition4 m ∧ m = -1 :=
by sorry

end NUMINAMATH_GPT_find_a_b_find_m_l1218_121890


namespace NUMINAMATH_GPT_quad_func_minimum_l1218_121852

def quad_func (x : ℝ) : ℝ := x^2 - 8 * x + 5

theorem quad_func_minimum : ∀ x : ℝ, quad_func x ≥ -11 ∧ quad_func 4 = -11 :=
by
  sorry

end NUMINAMATH_GPT_quad_func_minimum_l1218_121852


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1218_121842

noncomputable def a := 3

theorem simplify_and_evaluate : (a^2 / (a + 1) - 1 / (a + 1)) = 2 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1218_121842


namespace NUMINAMATH_GPT_probability_Q_within_three_units_of_origin_l1218_121867

noncomputable def probability_within_three_units_of_origin :=
  let radius := 3
  let square_side := 10
  let circle_area := Real.pi * radius^2
  let square_area := square_side^2
  circle_area / square_area

theorem probability_Q_within_three_units_of_origin :
  probability_within_three_units_of_origin = 9 * Real.pi / 100 :=
by
  -- Since this proof is not required, we skip it with sorry.
  sorry

end NUMINAMATH_GPT_probability_Q_within_three_units_of_origin_l1218_121867


namespace NUMINAMATH_GPT_min_distance_curves_l1218_121835

theorem min_distance_curves (P Q : ℝ × ℝ) (h1 : P.2 = (1/3) * Real.exp P.1) (h2 : Q.2 = Real.log (3 * Q.1)) :
  ∃ d : ℝ, d = Real.sqrt 2 * (Real.log 3 - 1) ∧ d = |P.1 - Q.1| := sorry

end NUMINAMATH_GPT_min_distance_curves_l1218_121835


namespace NUMINAMATH_GPT_percentage_increase_x_y_l1218_121881

theorem percentage_increase_x_y (Z Y X : ℝ) (h1 : Z = 300) (h2 : Y = 1.20 * Z) (h3 : X = 1110 - Y - Z) :
  ((X - Y) / Y) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_x_y_l1218_121881


namespace NUMINAMATH_GPT_photos_per_album_l1218_121822

theorem photos_per_album
  (n : ℕ) -- number of pages in each album
  (x y : ℕ) -- album numbers
  (h1 : 4 * n * (x - 1) + 17 ≤ 81 ∧ 81 ≤ 4 * n * (x - 1) + 20)
  (h2 : 4 * n * (y - 1) + 9 ≤ 171 ∧ 171 ≤ 4 * n * (y - 1) + 12) :
  4 * n = 32 :=
by 
  sorry

end NUMINAMATH_GPT_photos_per_album_l1218_121822


namespace NUMINAMATH_GPT_max_elephants_l1218_121850

def union_members : ℕ := 28
def non_union_members : ℕ := 37

/-- Given 28 union members and 37 non-union members, where elephants are distributed equally among
each group and each person initially receives at least one elephant, and considering 
the unique distribution constraint, the maximum number of elephants is 2072. -/
theorem max_elephants (n : ℕ) 
  (h1 : n % union_members = 0)
  (h2 : n % non_union_members = 0)
  (h3 : n ≥ union_members * non_union_members) :
  n = 2072 :=
by sorry

end NUMINAMATH_GPT_max_elephants_l1218_121850


namespace NUMINAMATH_GPT_fraction_simplification_l1218_121829

theorem fraction_simplification : 
  (2222 - 2123) ^ 2 / 121 = 81 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1218_121829


namespace NUMINAMATH_GPT_sector_arc_length_l1218_121819

noncomputable def arc_length (R : ℝ) (θ : ℝ) : ℝ :=
  θ / 180 * Real.pi * R

theorem sector_arc_length
  (central_angle : ℝ) (area : ℝ) (arc_length_answer : ℝ)
  (h1 : central_angle = 120)
  (h2 : area = 300 * Real.pi) :
  arc_length_answer = 20 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sector_arc_length_l1218_121819


namespace NUMINAMATH_GPT_line_intersects_circle_l1218_121877

variable {a x_0 y_0 : ℝ}

theorem line_intersects_circle (h1: x_0^2 + y_0^2 > a^2) (h2: a > 0) : 
  ∃ (p : ℝ × ℝ), (p.1 ^ 2 + p.2 ^ 2 = a ^ 2) ∧ (x_0 * p.1 + y_0 * p.2 = a ^ 2) :=
sorry

end NUMINAMATH_GPT_line_intersects_circle_l1218_121877


namespace NUMINAMATH_GPT_unique_a_b_l1218_121832

-- Define the properties of the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * a * x + b else 7 - 2 * x

-- The function satisfies f(f(x)) = x for all x in its domain
theorem unique_a_b (a b : ℝ) (h : ∀ x : ℝ, f a b (f a b x) = x) : a + b = 13 / 4 :=
sorry

end NUMINAMATH_GPT_unique_a_b_l1218_121832


namespace NUMINAMATH_GPT_find_number_thought_of_l1218_121855

theorem find_number_thought_of :
  ∃ x : ℝ, (6 * x^2 - 10) / 3 + 15 = 95 ∧ x = 5 * Real.sqrt 15 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_thought_of_l1218_121855


namespace NUMINAMATH_GPT_find_f_l1218_121837

theorem find_f {f : ℝ → ℝ} (h : ∀ x, f (1/x) = x / (1 - x)) : ∀ x, f x = 1 / (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_find_f_l1218_121837


namespace NUMINAMATH_GPT_beetle_total_distance_l1218_121808

theorem beetle_total_distance 
  (r_outer : ℝ) (r_middle : ℝ) (r_inner : ℝ)
  (r_outer_eq : r_outer = 25)
  (r_middle_eq : r_middle = 15)
  (r_inner_eq : r_inner = 5)
  : (1/3 * 2 * Real.pi * r_middle + (r_outer - r_middle) + 1/2 * 2 * Real.pi * r_inner + 2 * r_outer + (r_middle - r_inner)) = (15 * Real.pi + 70) :=
by
  rw [r_outer_eq, r_middle_eq, r_inner_eq]
  have := Real.pi
  sorry

end NUMINAMATH_GPT_beetle_total_distance_l1218_121808


namespace NUMINAMATH_GPT_fixed_line_of_midpoint_l1218_121885

theorem fixed_line_of_midpoint
  (A B : ℝ × ℝ)
  (H : ∀ (P : ℝ × ℝ), (P = A ∨ P = B) → (P.1^2 / 3 - P.2^2 / 6 = 1))
  (slope_l : (B.2 - A.2) / (B.1 - A.1) = 2)
  (midpoint_lies : (A.1 + B.1) / 2 = (A.2 + B.2) / 2) :
  ∀ (M : ℝ × ℝ), (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) → M.1 - M.2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_fixed_line_of_midpoint_l1218_121885


namespace NUMINAMATH_GPT_sum_of_values_l1218_121813

def f (x : ℝ) : ℝ := x^2 + 2 * x + 2

theorem sum_of_values (z₁ z₂ : ℝ) (h₁ : f (3 * z₁) = 10) (h₂ : f (3 * z₂) = 10) :
  z₁ + z₂ = - (2 / 9) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_values_l1218_121813


namespace NUMINAMATH_GPT_michael_and_emma_dig_time_correct_l1218_121800

noncomputable def michael_and_emma_digging_time : ℝ :=
let father_rate := 4
let father_time := 450
let father_depth := father_rate * father_time
let mother_rate := 5
let mother_time := 300
let mother_depth := mother_rate * mother_time
let michael_desired_depth := 3 * father_depth - 600
let emma_desired_depth := 2 * mother_depth + 300
let desired_depth := max michael_desired_depth emma_desired_depth
let michael_rate := 3
let emma_rate := 6
let combined_rate := michael_rate + emma_rate
desired_depth / combined_rate

theorem michael_and_emma_dig_time_correct :
  michael_and_emma_digging_time = 533.33 := 
sorry

end NUMINAMATH_GPT_michael_and_emma_dig_time_correct_l1218_121800


namespace NUMINAMATH_GPT_max_min_values_of_x_l1218_121836

theorem max_min_values_of_x (x y z : ℝ) (h1 : x + y + z = 0) (h2 : (x - y)^2 + (y - z)^2 + (z - x)^2 ≤ 2) :
  -2/3 ≤ x ∧ x ≤ 2/3 :=
sorry

end NUMINAMATH_GPT_max_min_values_of_x_l1218_121836


namespace NUMINAMATH_GPT_expected_value_coins_heads_l1218_121875

noncomputable def expected_value_cents : ℝ :=
  let values := [1, 5, 10, 25, 50, 100]
  let probability_heads := 1 / 2
  probability_heads * (values.sum : ℝ)

theorem expected_value_coins_heads : expected_value_cents = 95.5 := by
  sorry

end NUMINAMATH_GPT_expected_value_coins_heads_l1218_121875


namespace NUMINAMATH_GPT_remainder_of_sum_mod_13_l1218_121898

theorem remainder_of_sum_mod_13 :
  ∀ (D : ℕ) (k1 k2 : ℕ),
    D = 13 →
    (242 = k1 * D + 8) →
    (698 = k2 * D + 9) →
    (242 + 698) % D = 4 :=
by
  intros D k1 k2 hD h242 h698
  sorry

end NUMINAMATH_GPT_remainder_of_sum_mod_13_l1218_121898


namespace NUMINAMATH_GPT_total_students_l1218_121897

theorem total_students (n x : ℕ) (h1 : 3 * n + 48 = 6 * n) (h2 : 4 * n + x = 2 * n + 2 * x) : n = 16 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l1218_121897


namespace NUMINAMATH_GPT_total_combined_rainfall_l1218_121815

def mondayRainfall := 7 * 1
def tuesdayRainfall := 4 * 2
def wednesdayRate := 2 * 2
def wednesdayRainfall := 2 * wednesdayRate
def totalRainfall := mondayRainfall + tuesdayRainfall + wednesdayRainfall

theorem total_combined_rainfall : totalRainfall = 23 :=
by
  unfold totalRainfall mondayRainfall tuesdayRainfall wednesdayRainfall wednesdayRate
  sorry

end NUMINAMATH_GPT_total_combined_rainfall_l1218_121815


namespace NUMINAMATH_GPT_ones_digit_of_11_pow_46_l1218_121848

theorem ones_digit_of_11_pow_46 : (11 ^ 46) % 10 = 1 :=
by sorry

end NUMINAMATH_GPT_ones_digit_of_11_pow_46_l1218_121848


namespace NUMINAMATH_GPT_find_principal_amount_l1218_121868

theorem find_principal_amount (A R T : ℝ) (P : ℝ) : 
  A = 1680 → R = 0.05 → T = 2.4 → 1.12 * P = 1680 → P = 1500 :=
by
  intros hA hR hT hEq
  sorry

end NUMINAMATH_GPT_find_principal_amount_l1218_121868


namespace NUMINAMATH_GPT_tom_sleep_hours_l1218_121854

-- Define initial sleep hours and increase fraction
def initial_sleep_hours : ℕ := 6
def increase_fraction : ℚ := 1 / 3

-- Define the function to calculate increased sleep
def increased_sleep_hours (initial : ℕ) (fraction : ℚ) : ℚ :=
  initial * fraction

-- Define the function to calculate total sleep hours
def total_sleep_hours (initial : ℕ) (increased : ℚ) : ℚ :=
  initial + increased

-- Theorem stating Tom's total sleep hours per night after the increase
theorem tom_sleep_hours (initial : ℕ) (fraction : ℚ) (increased : ℚ) (total : ℚ) :
  initial = initial_sleep_hours →
  fraction = increase_fraction →
  increased = increased_sleep_hours initial fraction →
  total = total_sleep_hours initial increased →
  total = 8 :=
by
  intros h_init h_frac h_incr h_total
  rw [h_init, h_frac] at h_incr
  rw [h_init, h_incr] at h_total
  sorry

end NUMINAMATH_GPT_tom_sleep_hours_l1218_121854


namespace NUMINAMATH_GPT_sum_of_roots_l1218_121826

theorem sum_of_roots (x1 x2 k c : ℝ) (h1 : 2 * x1^2 - k * x1 = 2 * c) 
  (h2 : 2 * x2^2 - k * x2 = 2 * c) (h3 : x1 ≠ x2) : x1 + x2 = k / 2 := 
sorry

end NUMINAMATH_GPT_sum_of_roots_l1218_121826


namespace NUMINAMATH_GPT_fruit_seller_price_l1218_121859

theorem fruit_seller_price (C : ℝ) (h1 : 1.05 * C = 14.823529411764707) : 
  0.85 * C = 12 := 
sorry

end NUMINAMATH_GPT_fruit_seller_price_l1218_121859


namespace NUMINAMATH_GPT_milk_mixture_l1218_121893

theorem milk_mixture (x : ℝ) : 
  (2.4 + 0.1 * x) / (8 + x) = 0.2 → x = 8 :=
by
  sorry

end NUMINAMATH_GPT_milk_mixture_l1218_121893


namespace NUMINAMATH_GPT_triangle_perimeter_l1218_121880

theorem triangle_perimeter (A r p : ℝ) (hA : A = 75) (hr : r = 2.5) :
  A = r * (p / 2) → p = 60 := by
  intros
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l1218_121880


namespace NUMINAMATH_GPT_find_linear_function_l1218_121896

theorem find_linear_function (f : ℝ → ℝ) (hf_inc : ∀ x y, x < y → f x < f y)
  (hf_lin : ∃ a b, a > 0 ∧ ∀ x, f x = a * x + b)
  (h_comp : ∀ x, f (f x) = 4 * x + 3) :
  ∀ x, f x = 2 * x + 1 :=
by
  sorry

end NUMINAMATH_GPT_find_linear_function_l1218_121896


namespace NUMINAMATH_GPT_eq1_solution_eq2_solution_l1218_121892

theorem eq1_solution (x : ℝ) : (3 * x * (x - 1) = 2 - 2 * x) ↔ (x = 1 ∨ x = -2/3) :=
sorry

theorem eq2_solution (x : ℝ) : (3 * x^2 - 6 * x + 2 = 0) ↔ (x = 1 + (Real.sqrt 3) / 3 ∨ x = 1 - (Real.sqrt 3) / 3) :=
sorry

end NUMINAMATH_GPT_eq1_solution_eq2_solution_l1218_121892


namespace NUMINAMATH_GPT_correct_answer_l1218_121876

theorem correct_answer (a b c : ℝ) : a - (b + c) = a - b - c :=
by sorry

end NUMINAMATH_GPT_correct_answer_l1218_121876


namespace NUMINAMATH_GPT_first_discount_percentage_l1218_121860

noncomputable def saree_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  initial_price * (1 - discount1 / 100) * (1 - discount2 / 100)

theorem first_discount_percentage (x : ℝ) : saree_price 400 x 20 = 240 → x = 25 :=
by sorry

end NUMINAMATH_GPT_first_discount_percentage_l1218_121860


namespace NUMINAMATH_GPT_dark_squares_more_than_light_l1218_121833

/--
A 9 × 9 board is composed of alternating dark and light squares, with the upper-left square being dark.
Prove that there is exactly 1 more dark square than light square.
-/
theorem dark_squares_more_than_light :
  let board_size := 9
  let dark_squares := 5 * 5 + 4 * 4
  let light_squares := 4 * 5 + 5 * 4
  dark_squares - light_squares = 1 :=
by
  let board_size := 9
  let dark_squares := 5 * 5 + 4 * 4
  let light_squares := 4 * 5 + 5 * 4
  show dark_squares - light_squares = 1
  sorry

end NUMINAMATH_GPT_dark_squares_more_than_light_l1218_121833


namespace NUMINAMATH_GPT_nonstudent_ticket_cost_l1218_121817

theorem nonstudent_ticket_cost :
  ∃ x : ℝ, (530 * 2 + (821 - 530) * x = 1933) ∧ x = 3 :=
by 
  sorry

end NUMINAMATH_GPT_nonstudent_ticket_cost_l1218_121817


namespace NUMINAMATH_GPT_problem_statement_l1218_121899

theorem problem_statement (x : ℝ) (hx : x^2 + 1/(x^2) = 2) : x^4 + 1/(x^4) = 2 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1218_121899


namespace NUMINAMATH_GPT_olivia_wallet_l1218_121802

theorem olivia_wallet (initial_amount spent_amount remaining_amount : ℕ)
  (h1 : initial_amount = 78)
  (h2 : spent_amount = 15):
  remaining_amount = initial_amount - spent_amount →
  remaining_amount = 63 :=
sorry

end NUMINAMATH_GPT_olivia_wallet_l1218_121802


namespace NUMINAMATH_GPT_red_paint_intensity_l1218_121874

variable (I : ℝ) -- Intensity of the original paint
variable (P : ℝ) -- Volume of the original paint
variable (fraction_replaced : ℝ := 1) -- Fraction of original paint replaced
variable (new_intensity : ℝ := 20) -- New paint intensity
variable (replacement_intensity : ℝ := 20) -- Replacement paint intensity

theorem red_paint_intensity : new_intensity = replacement_intensity :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_red_paint_intensity_l1218_121874


namespace NUMINAMATH_GPT_gcd_linear_combination_l1218_121857

theorem gcd_linear_combination (a b : ℤ) : 
  Int.gcd (5 * a + 3 * b) (13 * a + 8 * b) = Int.gcd a b := 
sorry

end NUMINAMATH_GPT_gcd_linear_combination_l1218_121857


namespace NUMINAMATH_GPT_max_sum_of_arithmetic_seq_l1218_121851

theorem max_sum_of_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h₁ : a 1 = 11) (h₂ : a 5 = -1) 
  (h₃ : ∀ n, a n = 14 - 3 * (n - 1)) 
  : ∀ n, (S n = (n * (a 1 + a n) / 2)) → max (S n) = 26 :=
sorry

end NUMINAMATH_GPT_max_sum_of_arithmetic_seq_l1218_121851


namespace NUMINAMATH_GPT_range_of_m_for_basis_l1218_121803

open Real

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (m, 3 * m - 2)

theorem range_of_m_for_basis (m : ℝ) :
  vector_a ≠ vector_b m → m ≠ 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_for_basis_l1218_121803


namespace NUMINAMATH_GPT_sum_of_remainders_l1218_121834

theorem sum_of_remainders 
  (a b c : ℕ) 
  (h1 : a % 53 = 37) 
  (h2 : b % 53 = 14) 
  (h3 : c % 53 = 7) : 
  (a + b + c) % 53 = 5 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l1218_121834


namespace NUMINAMATH_GPT_water_flow_speed_l1218_121884

/-- A person rows a boat for 15 li. If he rows at his usual speed,
the time taken to row downstream is 5 hours less than rowing upstream.
If he rows at twice his usual speed, the time taken to row downstream
is only 1 hour less than rowing upstream. 
Prove that the speed of the water flow is 2 li/hour.
-/
theorem water_flow_speed (y x : ℝ)
  (h1 : 15 / (y - x) - 15 / (y + x) = 5)
  (h2 : 15 / (2 * y - x) - 15 / (2 * y + x) = 1) :
  x = 2 := 
sorry

end NUMINAMATH_GPT_water_flow_speed_l1218_121884


namespace NUMINAMATH_GPT_red_grapes_count_l1218_121872

-- Definitions of variables and conditions
variables (G R Ra B P : ℕ)
variables (cond1 : R = 3 * G + 7)
variables (cond2 : Ra = G - 5)
variables (cond3 : B = 4 * Ra)
variables (cond4 : P = (1 / 2) * B + 5)
variables (cond5 : G + R + Ra + B + P = 350)

-- Theorem statement
theorem red_grapes_count : R = 100 :=
by sorry

end NUMINAMATH_GPT_red_grapes_count_l1218_121872


namespace NUMINAMATH_GPT_parabola_vertex_coordinates_l1218_121891

theorem parabola_vertex_coordinates :
  ∀ x : ℝ, (3 * (x - 7) ^ 2 + 5) = 3 * (x - 7) ^ 2 + 5 := by
  sorry

end NUMINAMATH_GPT_parabola_vertex_coordinates_l1218_121891


namespace NUMINAMATH_GPT_part_I_monotonicity_part_II_value_a_l1218_121862

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / (x - 1)

def is_monotonic_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x < f y

def is_monotonic_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f y < f x

theorem part_I_monotonicity :
  (is_monotonic_increasing f {x | 2 < x}) ∧
  ((is_monotonic_decreasing f {x | x < 1}) ∧ (is_monotonic_decreasing f {x | 1 < x ∧ x < 2})) :=
by
  sorry

theorem part_II_value_a (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x → (Real.exp x * (x - 2)) / ((x - 1)^2) ≥ a * (Real.exp x / (x - 1))) → a ∈ Set.Iic 0 :=
by
  sorry

end NUMINAMATH_GPT_part_I_monotonicity_part_II_value_a_l1218_121862


namespace NUMINAMATH_GPT_count_sums_of_fours_and_fives_l1218_121889

theorem count_sums_of_fours_and_fives :
  ∃ n, (∀ x y : ℕ, 4 * x + 5 * y = 1800 ↔ (x = 0 ∨ x ≤ 1800) ∧ (y = 0 ∨ y ≤ 1800)) ∧ n = 201 :=
by
  -- definition and theorem statement is complete. The proof is omitted.
  sorry

end NUMINAMATH_GPT_count_sums_of_fours_and_fives_l1218_121889


namespace NUMINAMATH_GPT_reciprocal_relationship_l1218_121869

theorem reciprocal_relationship (a b : ℚ)
  (h1 : a = (-7 / 8) / (7 / 4 - 7 / 8 - 7 / 12))
  (h2 : b = (7 / 4 - 7 / 8 - 7 / 12) / (-7 / 8)) :
  a = - 1 / b :=
by sorry

end NUMINAMATH_GPT_reciprocal_relationship_l1218_121869


namespace NUMINAMATH_GPT_person_next_to_Boris_arkady_galya_l1218_121841

-- Define the people involved
inductive Person
| Arkady
| Boris
| Vera
| Galya
| Danya
| Egor

open Person

def next_to (p1 p2 : Person) (standing_next : Person → Person → Prop) : Prop :=
standing_next p1 p2 ∨ standing_next p2 p1

noncomputable def position_relationships : Prop :=
  ∃ (standing_next : Person → Person → Prop),
    -- Danya is next to Vera, on Vera's right side
    standing_next Danya Vera ∧
    -- Galya stood opposite Egor
    (∀ p, next_to p Galya standing_next → next_to p Egor standing_next) ∧
    -- Egor is next to Danya
    standing_next Egor Danya ∧
    -- Arkady and Galya did not want to stand next to each other
    ¬ next_to Arkady Galya standing_next ∧
    -- Conclusion: Arkady and Galya are standing next to Boris
    next_to Arkady Boris standing_next ∧ next_to Galya Boris standing_next

theorem person_next_to_Boris_arkady_galya : position_relationships :=
    sorry

end NUMINAMATH_GPT_person_next_to_Boris_arkady_galya_l1218_121841


namespace NUMINAMATH_GPT_price_of_orange_l1218_121888

-- Define relevant conditions
def price_apple : ℝ := 1.50
def morning_apples : ℕ := 40
def morning_oranges : ℕ := 30
def afternoon_apples : ℕ := 50
def afternoon_oranges : ℕ := 40
def total_sales : ℝ := 205

-- Define the proof problem
theorem price_of_orange (O : ℝ) 
  (h : (morning_apples * price_apple + morning_oranges * O) + 
       (afternoon_apples * price_apple + afternoon_oranges * O) = total_sales) : 
  O = 1 :=
by
  sorry

end NUMINAMATH_GPT_price_of_orange_l1218_121888


namespace NUMINAMATH_GPT_Soyun_distance_l1218_121818

theorem Soyun_distance
  (perimeter : ℕ)
  (Soyun_speed : ℕ)
  (Jia_speed : ℕ)
  (meeting_time : ℕ)
  (time_to_meet : perimeter = (Soyun_speed + Jia_speed) * meeting_time) :
  Soyun_speed * meeting_time = 10 :=
by
  sorry

end NUMINAMATH_GPT_Soyun_distance_l1218_121818


namespace NUMINAMATH_GPT_find_playground_side_length_l1218_121820

-- Define the conditions
def playground_side_length (x : ℝ) : Prop :=
  let perimeter_square := 4 * x
  let perimeter_garden := 2 * (12 + 9)
  let total_perimeter := perimeter_square + perimeter_garden
  total_perimeter = 150

-- State the main theorem to prove that the side length of the square fence around the playground is 27 yards
theorem find_playground_side_length : ∃ x : ℝ, playground_side_length x ∧ x = 27 :=
by
  exists 27
  sorry

end NUMINAMATH_GPT_find_playground_side_length_l1218_121820


namespace NUMINAMATH_GPT_determine_coefficients_l1218_121805

variable {α : Type} [Field α]
variables (a a1 a2 a3 : α)

theorem determine_coefficients (h : ∀ x : α, a + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 = x^3) :
  a = 1 ∧ a2 = 3 :=
by
  -- To be proven
  sorry

end NUMINAMATH_GPT_determine_coefficients_l1218_121805


namespace NUMINAMATH_GPT_union_of_A_and_B_l1218_121863

-- Definitions of sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 4}
def B : Set ℝ := {x | -3 ≤ x ∧ x < 1}

-- The theorem we aim to prove
theorem union_of_A_and_B : A ∪ B = { x | -3 ≤ x ∧ x ≤ 4 } :=
sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1218_121863


namespace NUMINAMATH_GPT_range_of_t_l1218_121810

theorem range_of_t (t : ℝ) : 
  (∃ x : ℝ, x^2 - 3 * x + t ≤ 0 ∧ x ≤ t) ↔ (0 ≤ t ∧ t ≤ 9 / 4) := 
sorry

end NUMINAMATH_GPT_range_of_t_l1218_121810


namespace NUMINAMATH_GPT_schoolchildren_number_l1218_121853

theorem schoolchildren_number (n m S : ℕ) 
  (h1 : S = 22 * n + 3)
  (h2 : S = (n - 1) * m)
  (h3 : n ≤ 18)
  (h4 : m ≤ 36) : 
  S = 135 := 
sorry

end NUMINAMATH_GPT_schoolchildren_number_l1218_121853


namespace NUMINAMATH_GPT_problem_statement_l1218_121814

def permutations (n r : ℕ) : ℕ := n.factorial / (n - r).factorial
def combinations (n r : ℕ) : ℕ := n.factorial / (r.factorial * (n - r).factorial)

theorem problem_statement : permutations 4 2 - combinations 4 3 = 8 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1218_121814
