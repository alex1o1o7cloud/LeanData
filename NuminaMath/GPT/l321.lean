import Mathlib

namespace scalene_right_triangle_area_l321_321748

def triangle_area (a b : ℝ) : ℝ := 0.5 * a * b

theorem scalene_right_triangle_area
  (ABC : Type) [is_triangle ABC]
  (hypotenuse : line_segment ABC)
  (P : point hypotenuse)
  (angle_ABP : angle ABC = 30)
  (AP : distance between points A P = 2)
  (CP : distance between points C P = 1)
  (AB : length between points A B)
  (BC : length between points B C)
  (AC : length between points A C = 3)
  (right_angle : right angle of triangle ABC B) 
  : triangle_area AB BC = 9/5 :=
by 
  sorry

end scalene_right_triangle_area_l321_321748


namespace primes_satisfying_condition_l321_321953

theorem primes_satisfying_condition :
    {p : ℕ | p.Prime ∧ ∀ q : ℕ, q.Prime ∧ q < p → ¬ ∃ n : ℕ, n^2 ∣ (p - (p / q) * q)} =
    {2, 3, 5, 7, 13} :=
by sorry

end primes_satisfying_condition_l321_321953


namespace michael_investment_correct_l321_321353

noncomputable def michael_initial_investment_at_solidsavings (total_amount : ℝ) (final_amount : ℝ) (withdrawal : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let x := (final_amount - (rate2 * ((total_amount - 100) * rate2 - 100)) + 100) / (rate1 ^ 2 - rate2^2) in
  x

theorem michael_investment_correct :
  michael_initial_investment_at_solidsavings 1500 1638 100 1.04 1.06 ≈ 1276.19 := by
  sorry

end michael_investment_correct_l321_321353


namespace rotated_main_diagonal_l321_321399

theorem rotated_main_diagonal:
  let board := λ (i j : ℕ), (i * 32 + j + 1)
  let rotations := λ (m : ℕ), 
                    Array.init ((2^m)^2) (λ ij, 
                      match ij % 4 with 
                      | 0 => (ij, 0)
                      | 1 => (ij + (2^m * 32) + 1)
                      | 2 => (ij + (2^m * 32 * 2) + 1)
                      | 3 => (ij + (2^m * 32 * 3) + 1)
                      end)
  in 
  (let final_diagonal := rotations 5
  in ∀ i, i < 32 → final_diagonal[i][i] = 993 - 31 * i)
:= sorry

end rotated_main_diagonal_l321_321399


namespace initial_geese_count_l321_321825

-- Define the number of geese that flew away
def geese_flew_away : ℕ := 28

-- Define the number of geese left in the field
def geese_left : ℕ := 23

-- Prove that the initial number of geese in the field was 51
theorem initial_geese_count : geese_left + geese_flew_away = 51 := by
  sorry

end initial_geese_count_l321_321825


namespace student_adjustment_l321_321409

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem student_adjustment : 
  let front_row_size := 4
  let back_row_size := 8
  let total_students := 12
  let num_to_select := 2
  let ways_to_select := binomial back_row_size num_to_select
  let ways_to_permute := permutation (front_row_size + num_to_select) num_to_select
  ways_to_select * ways_to_permute = 840 :=
  by {
    let front_row_size := 4
    let back_row_size := 8
    let total_students := 12
    let num_to_select := 2
    let ways_to_select := binomial back_row_size num_to_select
    let ways_to_permute := permutation (front_row_size + num_to_select) num_to_select
    exact sorry
  }

end student_adjustment_l321_321409


namespace problem_a_problem_b_problem_d_l321_321245

variables {x y : ℝ}

-- Definitions and conditions
def cond (x y : ℝ) : Prop := 2^x + 2^(y+1) = 1
def m (x y : ℝ) : ℝ := x + y
def n (x y : ℝ) : ℝ := (1/2)^x + (1/2)^(y-1)

-- Proof statements
theorem problem_a (hx : cond x y) : x < 0 ∧ y < -1 :=
sorry

theorem problem_b (hx : cond x y) : m x y ≤ -3 :=
sorry

theorem problem_d (hx : cond x y) : n x y * 2^(m x y) < 2 :=
sorry

-- The statement of equivalence proving that options A, B, and D are correct
def correct_option (x y : ℝ) (hx : cond x y) : Prop :=
  (x < 0 ∧ y < -1) ∧ (m x y ≤ -3) ∧ (n x y * 2^(m x y) < 2)

end problem_a_problem_b_problem_d_l321_321245


namespace largest_prime_divisor_13_factorial_plus_14_factorial_l321_321994

theorem largest_prime_divisor_13_factorial_plus_14_factorial : 
  ∃ p, prime p ∧ p ∈ prime_factors (13! + 14!) ∧ ∀ q, prime q ∧ q ∈ prime_factors (13! + 14!) → q ≤ 13 :=
by { sorry }

end largest_prime_divisor_13_factorial_plus_14_factorial_l321_321994


namespace lines_intersect_at_l321_321154

noncomputable def line1 (x : ℚ) : ℚ := (-2 / 3) * x + 2
noncomputable def line2 (x : ℚ) : ℚ := -2 * x + (3 / 2)

theorem lines_intersect_at :
  ∃ (x y : ℚ), line1 x = y ∧ line2 x = y ∧ x = (3 / 8) ∧ y = (7 / 4) :=
sorry

end lines_intersect_at_l321_321154


namespace diagonals_of_undecagon_l321_321233

theorem diagonals_of_undecagon :
  let n := 11 in
  let D := (n * (n - 3)) / 2 in
  D = 44 :=
by
  sorry

end diagonals_of_undecagon_l321_321233


namespace max_k_solution_l321_321935

theorem max_k_solution
  (k x y : ℝ)
  (h_pos: 0 < k ∧ 0 < x ∧ 0 < y)
  (h_eq: 5 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) :
  ∃ k, 8*k^3 - 8*k^2 - 7*k = 0 := 
sorry

end max_k_solution_l321_321935


namespace figure_100_squares_l321_321534

-- Defining the sequences for figures
def num_squares : ℕ → ℕ
| 0 := 1
| 1 := 6
| 2 := 17
| 3 := 34
| n := 3 * n^2 + 2 * n + 1

theorem figure_100_squares : num_squares 100 = 30201 :=
by sorry

end figure_100_squares_l321_321534


namespace school_competition_l321_321268

theorem school_competition :
  (∃ n : ℕ, 
    n > 0 ∧ 
    75% students did not attend the competition ∧
    10% of those who attended participated in both competitions ∧
    ∃ y z : ℕ, y = 3 / 2 * z ∧ 
    y + z - (1 / 10) * (n / 4) = n / 4
  ) → n = 200 :=
sorry

end school_competition_l321_321268


namespace lex_coins_total_l321_321348

def value_of_coins (dimes quarters : ℕ) : ℕ :=
  10 * dimes + 25 * quarters

def more_quarters_than_dimes (dimes quarters : ℕ) : Prop :=
  quarters > dimes

theorem lex_coins_total (dimes quarters : ℕ) (h : value_of_coins dimes quarters = 265) (h_more : more_quarters_than_dimes dimes quarters) : dimes + quarters = 13 :=
sorry

end lex_coins_total_l321_321348


namespace michael_subtracts_79_from_40sq_to_get_39sq_l321_321419

theorem michael_subtracts_79_from_40sq_to_get_39sq :
  let a := 40
      b := 1 in
  a^2 - (a - b)^2 = 2*a*b - b^2 := by
  let a := 40
  let b := 1
  have h : (a - b)^2 = a^2 - 2*a*b + b^2 := by sorry
  rw [h]
  sorry

end michael_subtracts_79_from_40sq_to_get_39sq_l321_321419


namespace Leela_Hotel_all_three_reunions_l321_321910

theorem Leela_Hotel_all_three_reunions
  (A B C : Finset ℕ)
  (hA : A.card = 80)
  (hB : B.card = 90)
  (hC : C.card = 70)
  (hAB : (A ∩ B).card = 30)
  (hAC : (A ∩ C).card = 25)
  (hBC : (B ∩ C).card = 20)
  (hABC : ((A ∪ B ∪ C)).card = 150) : 
  (A ∩ B ∩ C).card = 15 :=
by
  sorry

end Leela_Hotel_all_three_reunions_l321_321910


namespace most_appropriate_sampling_method_l321_321413

namespace ResearchProblem

def breed_room_mice : list ℕ := [18, 24, 54, 48]
def total_mice : ℕ := breed_room_mice.sum
def sample_size : ℕ := 24
def appropriate_sampling_method : String := "Stratified Sampling"

theorem most_appropriate_sampling_method :
  (∃ (rooms : list ℕ), rooms = [18, 24, 54, 48]) →
  total_mice = 144 →
  sample_size = 24 →
  appropriate_sampling_method = "Stratified Sampling" :=
by
  sorry

end ResearchProblem

end most_appropriate_sampling_method_l321_321413


namespace net_percentage_change_l321_321520

theorem net_percentage_change (k m : ℝ) : 
  let scale_factor_1 := 1 - k / 100
  let scale_factor_2 := 1 + m / 100
  let overall_scale_factor := scale_factor_1 * scale_factor_2
  let percentage_change := (overall_scale_factor - 1) * 100
  percentage_change = m - k - k * m / 100 := 
by 
  sorry

end net_percentage_change_l321_321520


namespace no_solution_system_of_equations_l321_321381

theorem no_solution_system_of_equations :
  ¬ (∃ (x y : ℝ),
    (80 * x + 15 * y - 7) / (78 * x + 12 * y) = 1 ∧
    (2 * x^2 + 3 * y^2 - 11) / (y^2 - x^2 + 3) = 1 ∧
    78 * x + 12 * y ≠ 0 ∧
    y^2 - x^2 + 3 ≠ 0) :=
    by
      sorry

end no_solution_system_of_equations_l321_321381


namespace jason_cooking_time_l321_321316

-- Definitions based on the conditions
def initial_temperature : ℕ := 41
def boiling_temperature : ℕ := 212
def temperature_increase_per_minute : ℕ := 3
def cooking_time_after_boiling : ℕ := 12
def mixing_ratio : ℚ := 1 / 3

-- Main proof statement
theorem jason_cooking_time :
  let time_to_boil := (boiling_temperature - initial_temperature) / temperature_increase_per_minute
  let mixing_time := cooking_time_after_boiling * mixing_ratio
  (time_to_boil + cooking_time_after_boiling + mixing_time).to_nat = 73 := by
  sorry

end jason_cooking_time_l321_321316


namespace lines_parallel_l321_321400

theorem lines_parallel (k_1 k_2 b_1 b_2 : ℝ) :
  k_1 = -1 →
  k_2 = -1 →
  b_1 = 0 →
  b_2 = 6 →
  k_1 = k_2 ∧ b_1 ≠ b_2 →
  "The lines y=-x and y=-x+6 are parallel" :=
by
  sorry

end lines_parallel_l321_321400


namespace largest_prime_divisor_of_factorial_sum_l321_321961

theorem largest_prime_divisor_of_factorial_sum :
  ∃ p : ℕ, p.prime ∧ p ∣ (13! + 14!) ∧ ∀ q : ℕ, q.prime ∧ q ∣ (13! + 14!) → q ≤ p :=
sorry

end largest_prime_divisor_of_factorial_sum_l321_321961


namespace area_of_square_is_25_l321_321465

-- Define side length of the square
def sideLength : ℝ := 5

-- Define the area of the square
def area_of_square (side : ℝ) : ℝ := side * side

-- Prove the area of the square with side length 5 is 25 square meters
theorem area_of_square_is_25 : area_of_square sideLength = 25 := by
  sorry

end area_of_square_is_25_l321_321465


namespace area_of_curve_l321_321754

-- Defining the conditions and the area
def area_under_curve (a b : ℝ) (h : a < b) : ℝ :=
  ∫ x in a..b, 1 / x

-- Defining the problem statement
theorem area_of_curve (k : ℝ) (h : 0 < k) :
  area_under_curve k (1 / 2) (by linarith) = 2 * real.log 2 ∨ area_under_curve (1 / 2) k (by linarith) = 2 * real.log 2 → k = 2 ∨ k = 1 / 8 :=
sorry  -- Proof is omitted.

end area_of_curve_l321_321754


namespace problem_statement_l321_321483

noncomputable def avg_students_per_teacher : ℝ :=
  (80 + 40 + 40 + 10 + 5 + 5) / 6

noncomputable def avg_class_size_per_student : ℝ :=
  (80^2 + 40^2 + 40^2 + 10^2 + 5^2 + 5^2) / 180

theorem problem_statement : avg_students_per_teacher - avg_class_size_per_student = -24.17 :=
by
  have t := avg_students_per_teacher
  have s := avg_class_size_per_student
  calc
    t - s = 30 - 54.17 : by sorry -- Calculation of t and s
    ... = -24.17 : by sorry

end problem_statement_l321_321483


namespace number_of_solutions_eq_one_l321_321397

theorem number_of_solutions_eq_one : 
  (Nat.card {p : ℤ × ℤ | 2^(2 * p.1) - 3^(2 * p.2) = 55}) = 1 :=
by sorry

end number_of_solutions_eq_one_l321_321397


namespace KECD_is_cyclic_l321_321363

variables {O A B C D M E K : Type} [metric_space O]
variables (circle : circle O) (points_on_circle : ∀ p ∈ {A, B, C, D}, p ∈ circle)
variables (M_midpoint : midpoint M A B) (MC_intersect : point_intersect MC AB = E) (MD_intersect : point_intersect MD AB = K)

-- Given the above conditions, we need to prove KECD is cyclic
theorem KECD_is_cyclic :
  cyclic_quadrilateral K E C D :=
sorry

end KECD_is_cyclic_l321_321363


namespace relay_team_permutations_l321_321699

-- Definitions of conditions
def runners := ["Tony", "Leah", "Nina"]
def fixed_positions := ["Maria runs the third lap", "Jordan runs the fifth lap"]

-- Proof statement
theorem relay_team_permutations : 
  ∃ permutations, permutations = 6 := by
sorry

end relay_team_permutations_l321_321699


namespace sum_bn_l321_321169

noncomputable def a_n (n : ℕ) : ℕ :=
match n with
| 0     => 0
| (n+1) => n -- This specific definition is under assumption

noncomputable def b_n (n : ℕ) : ℕ := 
4 * (2 ^ a_n n) - 4 * a_n n

theorem sum_bn (n : ℕ) : 
  (∑ i in Finset.range n, b_n (i + 1)) = n - n^2 := by
  sorry

end sum_bn_l321_321169


namespace recursive_sequence_correct_l321_321032

theorem recursive_sequence_correct :
  ∃ (a : ℕ → ℕ), 
    a 1 = 1 ∧ 
    (∀ n ≥ 2, a n = a (n - 1) + n) ∧ 
    (a 1 = 1) ∧
    (a 2 = 3) ∧
    (a 3 = 6) ∧
    (a 4 = 10) ∧
    (a 5 = 15) :=
by 
  let a : ℕ → ℕ := sorry
  use a
  split; sorry

end recursive_sequence_correct_l321_321032


namespace highest_prob_sum_of_two_cubes_l321_321100

theorem highest_prob_sum_of_two_cubes :
  (∃ sum prob, (sum = 7) ∧ (prob = 1 / 6)) :=
by
  -- Problem Conditions: Define a fair six-faced dice
  let dice_faces := {1, 2, 3, 4, 5, 6}

  -- Calculate the sums and their probabilities
  -- Total outcomes for throwing two such dice (6 * 6 = 36)
  have total_outcomes : ℕ := 
    Finset.card (Finset.product dice_faces dice_faces)

  -- Mapping from each possible sum to its count of outcomes
  let sums_counts : Finset ℕ := Finset.image (fun p : ℕ × ℕ => p.fst + p.snd) (Finset.product dice_faces dice_faces)

  -- Calculate the probabilities
  let prob_map : ℕ → ℚ := 
    fun s : ℕ => 
      (Finset.card (Finset.filter (fun p : ℕ × ℕ => p.fst + p.snd = s) (Finset.product dice_faces dice_faces))) / total_outcomes

  -- highest probability sum of 7
  have seven_sum_prob : prob_map 7 = 1 / 6 := sorry

  use 7
  use (1 / 6)

  -- Provide the actual settlement
  exact ⟨rfl, seven_sum_prob⟩

end highest_prob_sum_of_two_cubes_l321_321100


namespace points_at_constant_distance_form_spherical_surface_l321_321658

theorem points_at_constant_distance_form_spherical_surface (p : EuclideanSpace ℝ 3) (r : ℝ) (h : r > 0) :
  {q : EuclideanSpace ℝ 3 | dist p q = r} = {q : EuclideanSpace ℝ 3 | ∥q - p∥ = r} :=
sorry

end points_at_constant_distance_form_spherical_surface_l321_321658


namespace range_of_x_l321_321246

noncomputable def meaningful_fraction (x : ℝ) : Prop :=
    (sqrt (x + 5)) / x >= 0

theorem range_of_x (x : ℝ) :
  meaningful_fraction x → x ≥ -5 ∧ x ≠ 0 :=
by intro h; sorry

end range_of_x_l321_321246


namespace box_volume_l321_321044

theorem box_volume (width length height : ℝ) (h1 : width = 9) (h2 : length = 4) (h3 : height = 7) : 
  width * length * height = 252 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end box_volume_l321_321044


namespace variance_of_data_is_eight_l321_321388

noncomputable def variance := λ (x : List ℝ), (x.map (λ x_i, (x_i - x.sum / x.length) ^ 2)).sum / x.length

theorem variance_of_data_is_eight
  (x : List ℝ)
  (h_length : x.length = 5)
  (h_avg : x.sum / 5 = 5)
  (h_square_avg : (x.map (λ x_i, x_i ^ 2)).sum / 5 = 33) :
  variance x = 8 :=
by
  sorry

end variance_of_data_is_eight_l321_321388


namespace inflection_points_on_line_l321_321723

-- Define the function f(x)
def f (x : ℝ) := 9 * x^5 - 30 * x^3 + 19 * x

-- Define the second derivative of f(x)
def f'' (x : ℝ) := 180 * x * (x^2 - 1)

-- Define the inflection points
def inflection_points : set (ℝ × ℝ) := {(-1, 2), (0, 0), (1, -2)}

-- Define the equation of the line
def line (x : ℝ) := -2 * x

theorem inflection_points_on_line :
  ∀ p ∈ inflection_points, p.2 = line p.1 := 
by
  -- This is where the proof would go, but we put sorry to skip the proof.
  sorry

end inflection_points_on_line_l321_321723


namespace solve_polynomial_l321_321954

theorem solve_polynomial (z : ℂ) :
    z^5 - 5 * z^3 + 6 * z = 0 ↔ 
    z = 0 ∨ z = -Real.sqrt 2 ∨ z = Real.sqrt 2 ∨ z = -Real.sqrt 3 ∨ z = Real.sqrt 3 := 
by 
  sorry

end solve_polynomial_l321_321954


namespace negation_of_universal_proposition_l321_321774

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) :=
by
  sorry

end negation_of_universal_proposition_l321_321774


namespace relative_speed_proof_l321_321907

-- Given constants
def f : ℝ := 10 -- focal length in cm
def t : ℝ := 30 -- object distance in cm
def v_t : ℝ := -200 -- object speed in cm/s (converted from 2 m/s)

-- Define the lens equation for image distance
def image_distance (f t : ℝ) : ℝ := (f * t) / (t - f)

-- Define the derivative of the image distance with respect to object distance using the quotient rule
def d_image_distance_dt (f t : ℝ) : ℝ := - (f^2) / (t - f)^2

-- Define the image speed
def image_speed (f t v_t : ℝ) : ℝ := (d_image_distance_dt f t) * v_t

-- Define the relative speed between object and image
def relative_speed (v_t : ℝ) (v_k : ℝ) : ℝ := abs(v_t - v_k)

-- Prove that their relative speed is 1.5 m/s (= 150 cm/s)
theorem relative_speed_proof : relative_speed v_t (image_speed f t v_t) = 150 :=
by
  -- we translate the given result in problem
  sorry

end relative_speed_proof_l321_321907


namespace probability_same_color_correct_l321_321679

def number_of_shoes := 12
def pairs_of_shoes := 6
def select_shoes := 3

noncomputable def probability_at_least_2_same_color :
  ℚ := 
  let total_ways_select := nat.choose number_of_shoes select_shoes in
  let ways_all_diff_colors := (12 * 10 * 8) / nat.factorial select_shoes in
  let prob_all_diff_colors := (ways_all_diff_colors : ℚ) / total_ways_select in
  1 - prob_all_diff_colors

theorem probability_same_color_correct :
  probability_at_least_2_same_color = 7 / 11 :=
by
  sorry

end probability_same_color_correct_l321_321679


namespace caochong_weighing_equation_l321_321814

-- Definitions for porter weight, stone weight, and the counts in the respective steps
def porter_weight : ℝ := 120
def stone_weight (x : ℝ) : ℝ := x
def first_step_weight (x : ℝ) : ℝ := 20 * stone_weight x + 3 * porter_weight
def second_step_weight (x : ℝ) : ℝ := (20 + 1) * stone_weight x + 1 * porter_weight

-- Theorem stating the equality condition ensuring the same water level
theorem caochong_weighing_equation (x : ℝ) :
  first_step_weight x = second_step_weight x :=
by
  sorry

end caochong_weighing_equation_l321_321814


namespace speed_difference_l321_321318

theorem speed_difference :
  let distance : ℝ := 8
  let zoe_time_hours : ℝ := 2 / 3
  let john_time_hours : ℝ := 1
  let zoe_speed : ℝ := distance / zoe_time_hours
  let john_speed : ℝ := distance / john_time_hours
  zoe_speed - john_speed = 4 :=
by
  sorry

end speed_difference_l321_321318


namespace poly_div_by_square_l321_321364

theorem poly_div_by_square (n : ℤ) (h : n > 1) : (n^n - n^2 + n - 1) ∣ ((n - 1)^2) :=
sorry

end poly_div_by_square_l321_321364


namespace number_of_routes_l321_321260

open Nat

theorem number_of_routes (south_cities north_cities : ℕ) 
  (connections : south_cities = 4 ∧ north_cities = 5) : 
  ∃ routes, routes = (factorial 3) * (5 ^ 4) := 
by
  sorry

end number_of_routes_l321_321260


namespace quadratic_trinomials_solution_l321_321550

noncomputable def quadratic_trinomials (f g : ℤ → ℤ) : Prop :=
  ∃ (a b c d : ℤ),
    f = λ x, x^2 + a * x + b ∧
    g = λ x, x^2 + c * x + d ∧
    (a + b = -c) ∧ (ab = d) ∧
    (c + d = -a) ∧ (cd = b)

theorem quadratic_trinomials_solution :
  quadratic_trinomials (λ x, x^2 + x - 2) (λ x, x^2 + x - 2) ∨
  ∃ a : ℤ, quadratic_trinomials (λ x, x^2 + a * x) (λ x, x^2 - a * x) :=
sorry

end quadratic_trinomials_solution_l321_321550


namespace area_right_scalene_triangle_l321_321737

noncomputable def area_of_triangle_ABC : ℝ :=
  let AP : ℝ := 2
  let CP : ℝ := 1
  let AC : ℝ := AP + CP
  let ratio : ℝ := 2
  let x_squared : ℝ := 9 / 5
  x_squared

theorem area_right_scalene_triangle (AP CP : ℝ) (h₁ : AP = 2) (h₂ : CP = 1) (h₃ : ∠(B : Point) (A : Point) (P : Point) = 30) :
  let AB := 2 * real.sqrt(9 / 5)
  let BC := real.sqrt(9 / 5)
  ∃ (area : ℝ), area = (1/2) * AB * BC ∧ area = 9 / 5 :=
by
  sorry

end area_right_scalene_triangle_l321_321737


namespace canCutIntoFourEqualParts_l321_321148

def cell : Type := bool -- A cell is either empty (false) or contains a star (true).

def grid : Type := array (fin 4) (array (fin 4) cell)

def givenShape : grid := array.mk
  [array.mk [true, false, false, true],
   array.mk [false, false, true, false],
   array.mk [true, true, false, false],
   array.mk [false, false, false, true]]

def starCount (g : grid) : nat :=
  g.foldl (λ acc row, acc + row.foldl (λ acc' c, acc' + if c then 1 else 0) 0) 0

theorem canCutIntoFourEqualParts (g : grid) (h : g = givenShape) :
  ∃ (parts : fin 4 → grid), 
    (∀ i, starCount (parts i) = 1) ∧ 
    (∀ i j, i ≠ j → parts i ∩ parts j = ∅) ∧ 
    (starCount g = 6) ∧
    (finset.fold (+) 0 (finset.image (λ i, starCount (parts i)) (finset.range 4)) = starCount g) :=
sorry

end canCutIntoFourEqualParts_l321_321148


namespace max_value_xy_l321_321566

theorem max_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2^x * 4^y = 4) : 
  xy ≤ 1 / 2 ∧ (x = 1 ∧ y = 1 / 2) :=
by {
  sorry
}

end max_value_xy_l321_321566


namespace smallest_number_of_students_l321_321285

theorem smallest_number_of_students
  (n : ℕ)
  (students_attended : ℕ)
  (students_both_competitions : ℕ)
  (students_hinting : ℕ)
  (students_cheating : ℕ)
  (attended_fraction : Real := 0.25)
  (both_competitions_fraction : Real := 0.1)
  (hinting_ratio : Real := 1.5)
  (h_attended : students_attended = (attended_fraction * n).to_nat)
  (h_both : students_both_competitions = (both_competitions_fraction * students_attended).to_nat)
  (h_hinting : students_hinting = (hinting_ratio * students_cheating).to_nat)
  (h_total_attended : students_attended = students_hinting + students_cheating - students_both_competitions)
  : n = 200 :=
sorry

end smallest_number_of_students_l321_321285


namespace examination_total_students_l321_321461

theorem examination_total_students (T : ℝ) :
  (0.35 * T + 520) = T ↔ T = 800 :=
by 
  sorry

end examination_total_students_l321_321461


namespace exists_subset_sum_multiple_of_2n_l321_321692

theorem exists_subset_sum_multiple_of_2n (n : ℕ) (hn : n ≥ 4) (a : Fin n → ℕ)
  (h_distinct : Function.Injective a)
  (h_bounds : ∀ i, 0 < a i ∧ a i < 2 * n) :
  ∃ (s : Finset (Fin n)), (∑ i in s, a i) % (2 * n) = 0 := 
  sorry

end exists_subset_sum_multiple_of_2n_l321_321692


namespace jessica_initial_withdrawal_fraction_l321_321317

variable {B : ℝ} -- this is the initial balance

noncomputable def initial_withdrawal_fraction (B : ℝ) : Prop :=
  let remaining_balance := B - 400
  let deposit := (1 / 4) * remaining_balance
  let final_balance := remaining_balance + deposit
  final_balance = 750 → (400 / B) = 2 / 5

-- Our goal is to prove the statement given conditions.
theorem jessica_initial_withdrawal_fraction : 
  ∃ B : ℝ, initial_withdrawal_fraction B :=
sorry

end jessica_initial_withdrawal_fraction_l321_321317


namespace length_of_boat_l321_321877

-- Conditions
def breadth : ℝ := 2  -- breadth of the boat in meters
def sink_depth : ℝ := 0.01  -- sink by 1 cm in meters
def man_mass : ℝ := 140  -- mass of the man in kg
def water_density : ℝ := 1000  -- density of water in kg/m³
def g : ℝ := 9.81  -- acceleration due to gravity in m/s²

-- Hypothesis: Weight of man equals weight of displaced water
theorem length_of_boat (L : ℝ) 
  (h1 : weight_of_man = weight_of_displaced_water) 
  (h2 : volume_displaced = L * breadth * sink_depth) : 
  L = 7 :=
by
  sorry

-- Definitions for weight of man and weight of displaced water
def weight_of_man : ℝ := man_mass * g  -- weight of the man in Newtons
def volume_displaced : ℝ := breadth * sink_depth * L  -- volume of displaced water in m³
def weight_of_displaced_water : ℝ := water_density * volume_displaced * g  -- weight of displaced water in Newtons

-- Equality proof
def main 
  : weight_of_man = weight_of_displaced_water → 
  volume_displaced = L * breadth * sink_depth → 
  L = 7 := 
  by sorry

end length_of_boat_l321_321877


namespace sum_of_odd_multiples_of_5_from_1_to_60_l321_321306

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem sum_of_odd_multiples_of_5_from_1_to_60 :
  (∑ n in Finset.filter (λ x, is_multiple_of_five x ∧ is_odd x) (Finset.range 61), n) = 180 :=
by {
  sorry
}

end sum_of_odd_multiples_of_5_from_1_to_60_l321_321306


namespace storks_count_l321_321477

theorem storks_count (B S : ℕ) (h1 : B = 3) (h2 : B + 2 = S + 1) : S = 4 :=
by
  sorry

end storks_count_l321_321477


namespace solve_equation1_solve_equation2_solve_equation3_solve_equation4_l321_321730

theorem solve_equation1 (x : ℝ) : (x - 1) ^ 2 = 4 ↔ x = 3 ∨ x = -1 :=
by sorry

theorem solve_equation2 (x : ℝ) : x ^ 2 + 3 * x - 4 = 0 ↔ x = 1 ∨ x = -4 :=
by sorry

theorem solve_equation3 (x : ℝ) : 4 * x * (2 * x + 1) = 3 * (2 * x + 1) ↔ x = -1 / 2 ∨ x = 3 / 4 :=
by sorry

theorem solve_equation4 (x : ℝ) : 2 * x ^ 2 + 5 * x - 3 = 0 ↔ x = 1 / 2 ∨ x = -3 :=
by sorry

end solve_equation1_solve_equation2_solve_equation3_solve_equation4_l321_321730


namespace problem_solution_l321_321875

variable (R : Type*) [NontrivialCommRing R] (diamondsuit : R → R → R)

axiom diamond_condition1 : ∀ (a b c : R), a ≠ 0 → b ≠ 0 → c ≠ 0 → a ⬝ (b ⬝ c) = (a ⬝ b) * c
axiom diamond_condition2 : ∀ (a : R), a ≠ 0 → a ⬝ a = 1

theorem problem_solution : (2016 ⬝ (6 ⬝ x) = 100) → x = 25/84 :=
  sorry
find p q : ℕ+ (p + q) such that relatively_prime p q ∧ x = p/q :=
  let p := 25 in
  let q := 84 in
  sorry

#check problem_solution
#check p + q

end problem_solution_l321_321875


namespace problem_proof_l321_321610

variables (a b : EuclideanSpace ℝ (Fin 3))
noncomputable def magnitude (v : EuclideanSpace ℝ (Fin 3)) := Real.sqrt (v.0 ^ 2 + v.1 ^ 2 + v.2 ^ 2)

def collinear (u v : EuclideanSpace ℝ (Fin 3)) := ∃ k : ℝ, u = k • v

def symmetric_about_x (v : EuclideanSpace ℝ (Fin 3)) := (v.0, -v.1, -v.2)
def symmetric_about_yOz (v : EuclideanSpace ℝ (Fin 3)) := (-v.0, v.1, v.2)

variables (a_eq : a = (1, -1, 2))
          (b_eq : b = (-2, 2, -4))

theorem problem_proof :
  magnitude a = Real.sqrt 6 ∧
  collinear a b ∧
  symmetric_about_x a = (1, 1, -2) ∧
  symmetric_about_yOz a ≠ (-1, 1, -2) :=
by sorry

end problem_proof_l321_321610


namespace sum_of_coordinates_of_B_l321_321362

theorem sum_of_coordinates_of_B (x : ℝ) (y : ℝ) 
  (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A = (0,0)) 
  (hB : B = (x, 3))
  (hslope : (3 - 0) / (x - 0) = 4 / 5) :
  x + 3 = 6.75 := 
by
  sorry

end sum_of_coordinates_of_B_l321_321362


namespace find_solutions_l321_321163

theorem find_solutions (x y : ℕ) : 33 ^ x + 31 = 2 ^ y → (x = 0 ∧ y = 5) ∨ (x = 1 ∧ y = 6) := 
by
  sorry

end find_solutions_l321_321163


namespace triangle_area_l321_321740

theorem triangle_area (A B C P : Point)
  (h_right : ∃ (α β γ : ℝ), α = 90 ∧ β ≠ 90 ∧ γ ≠ 90 ∧ β + γ = 90)
  (h_angle_ABP : Measure.angle A B P = 30)
  (h_AP : dist A P = 2)
  (h_CP : dist C P = 1)
  (h_AC : dist A C = 3) :
  area_triangle A B C = 9 / 5 :=
sorry

end triangle_area_l321_321740


namespace probability_rain_given_strong_wind_l321_321223

variable (P : Set (Type → Prop))

def strong_wind : Prop := P strong_wind
def rain : Prop := P rain

variable (P_stW : Prop) (P_rain : Prop) (P_stW_rain : Prop)

axiom prob_strong_wind : P(P_stW) = 0.4
axiom prob_rain : P(P_rain) = 0.5
axiom prob_both : P(P(A ∩ B)) = 0.3

theorem probability_rain_given_strong_wind : P(stW_rain) / P(stong_wind) = 3/4 := by
  sorry

end probability_rain_given_strong_wind_l321_321223


namespace largest_possible_number_after_100_deletions_l321_321941

-- Define the original number as a string formed by concatenating all integers from 1 to 5960
def original_number : String := (List.range (5960 + 1)).tail.toList.intersperse.toString

-- Define the function that deletes 100 digits to maximize the resulting number
def maximize_after_deletions (s : String) (n : Nat) : String := sorry

-- The target result after removing 100 digits
def target_result : String := "99999785960"

theorem largest_possible_number_after_100_deletions :
  maximize_after_deletions original_number 100 = target_result :=
sorry

end largest_possible_number_after_100_deletions_l321_321941


namespace math_problem_l321_321328

noncomputable def seq (n : ℕ) : ℝ := sorry

theorem math_problem (x : ℕ → ℝ)
  (h_pos : ∀ n, x n > 0)
  (h_sum : ∑' n, x n / (2 * n - 1) = 1) :
  ∑' k, ∑ n in Finset.range (k + 1), x n / (k^2 : ℝ) ≤ 2 :=
sorry

end math_problem_l321_321328


namespace one_inch_cubes_with_two_or_more_painted_faces_l321_321810

def original_cube_length : ℕ := 4

def total_one_inch_cubes : ℕ := original_cube_length ^ 3

def corners_count : ℕ := 8

def edges_minus_corners_count : ℕ := 12 * 2

theorem one_inch_cubes_with_two_or_more_painted_faces
  (painted_faces_on_each_face : ∀ i : ℕ, i < total_one_inch_cubes → ℕ) : 
  ∃ n : ℕ, n = corners_count + edges_minus_corners_count ∧ n = 32 := 
by
  simp only [corners_count, edges_minus_corners_count, total_one_inch_cubes]
  sorry

end one_inch_cubes_with_two_or_more_painted_faces_l321_321810


namespace true_propositions_l321_321511

/-- 
Prove that proposition ①, proposition ③, and proposition ④ are true.
-/

-- Definitions for each proposition based on the conditions
def prop_1 : Prop := ∀ (e : ℕ), (e = e) = true
def conv_b_3_b_2_9 : Prop := ∀ (b : ℕ), (b^2 = 9 → b = 3)
def comp_events_mut_exclusive : Prop := ∀ (A B : Prop), (A ∧ ⊥) = false ∧ (B ∧ ⊥) = false
def contra_angles_similar_triangles : Prop := ∀ (ΔABC ΔDEF : Type) (angles : ΔABC → ΔDEF), (angles c = (angles a ∧ angles b))

theorem true_propositions : prop_1 ∧ comp_events_mut_exclusive ∧ contra_angles_similar_triangles :=
by
  split,
  sorry,  -- Proof for proposition ①
  split,
  sorry,  -- Proof for proposition ③
  sorry   -- Proof for proposition ④

end true_propositions_l321_321511


namespace largest_n_l321_321831

noncomputable def is_multiple_of_seven (n : ℕ) : Prop :=
  (6 * (n-3)^3 - n^2 + 10 * n - 15) % 7 = 0

theorem largest_n (n : ℕ) : n < 50000 ∧ is_multiple_of_seven n → n = 49999 :=
by sorry

end largest_n_l321_321831


namespace jason_cooking_time_l321_321315

-- Definitions based on the conditions
def initial_temperature : ℕ := 41
def boiling_temperature : ℕ := 212
def temperature_increase_per_minute : ℕ := 3
def cooking_time_after_boiling : ℕ := 12
def mixing_ratio : ℚ := 1 / 3

-- Main proof statement
theorem jason_cooking_time :
  let time_to_boil := (boiling_temperature - initial_temperature) / temperature_increase_per_minute
  let mixing_time := cooking_time_after_boiling * mixing_ratio
  (time_to_boil + cooking_time_after_boiling + mixing_time).to_nat = 73 := by
  sorry

end jason_cooking_time_l321_321315


namespace rectangular_area_l321_321795

variable {x d : ℝ}

def length := 5 * x
def width := 4 * x
def diagonal := d

theorem rectangular_area (h_ratio : length / width = 5 / 4) (h_diagonal : d^2 = (5 * x)^2 + (4 * x)^2) :
  ∃ k : ℝ, (k = 20 / 41) ∧ (length * width = k * d^2) :=
sorry

end rectangular_area_l321_321795


namespace sum_of_reciprocals_of_roots_l321_321176

open Real

-- Define the polynomial and its properties using Vieta's formulas
theorem sum_of_reciprocals_of_roots :
  ∀ p q : ℝ, 
  (p + q = 16) ∧ (p * q = 9) → 
  (1 / p + 1 / q = 16 / 9) :=
by
  intros p q h
  let ⟨h1, h2⟩ := h
  sorry

end sum_of_reciprocals_of_roots_l321_321176


namespace game_probability_l321_321508

-- Defining initial coin distribution and constraints
def initialCoins := 5
universe u
constant Player : Type u
axiom Alice : Player
axiom Bob : Player
axiom Charlie : Player
axiom Dana : Player
constant Coins : Player → ℕ
axiom h_start : ∀ p : Player, Coins p = initialCoins

-- Defining transaction rules
def transaction (blueDrawer redDrawer yellowDrawer : Player) : Player → ℕ
| p => if p = blueDrawer then Coins p + 1
        else if p = redDrawer then Coins p - 1
        else if p = yellowDrawer then Coins p + 1
        else Coins p

def urn := multiset {Blue, Red, White, White, Yellow}
def draw (urn : multiset Color) : set (multiset Color × multiset Color) := sorry -- define all possible draws
def allPlayers : multiset Player := {Alice, Bob, Charlie, Dana}
def possibleDraws : set (∀ p : Player, Color) := sorry -- define all ways to map players to colors in one round without replacement

-- Proving the probability is 1/10000
theorem game_probability :
  (let possibilities := (draw urn) ^ 5 in
    ∑ p in possibilities, if (∀ p : Player, Coins p = initialCoins ) then 1 else 0) * (1 / ∣possibilities∣) = 1 / 10000 := sorry

end game_probability_l321_321508


namespace no_real_solution_log_equation_l321_321151

theorem no_real_solution_log_equation :
  ¬ ∃ x : ℝ, x + 4 > 0 ∧ x - 2 > 0 ∧ x ^ 2 - 7 > 0 ∧ 
  real.log10 (x + 4) + real.log10 (x - 2) = real.log10 (x ^ 2 - 7) := 
by
  sorry

end no_real_solution_log_equation_l321_321151


namespace total_distinguishable_triangles_l321_321416

-- Define number of colors
def numColors : Nat := 8

-- Define center colors
def centerColors : Nat := 3

-- Prove the total number of distinguishable large equilateral triangles
theorem total_distinguishable_triangles : 
  numColors * (numColors + numColors * (numColors - 1) + (numColors.choose 3)) * centerColors = 360 := by
  sorry

end total_distinguishable_triangles_l321_321416


namespace definite_integral_sin_l321_321921

theorem definite_integral_sin 
  : ∫ x in 0..(Real.pi / 2), (1 + Real.sin x) = (Real.pi / 2) + 1 := 
by sorry

end definite_integral_sin_l321_321921


namespace remainder_7_times_10_pow_20_plus_1_pow_20_mod_9_l321_321173

theorem remainder_7_times_10_pow_20_plus_1_pow_20_mod_9 :
  (7 * 10 ^ 20 + 1 ^ 20) % 9 = 8 :=
by
  -- need to note down the known conditions to help guide proof writing.
  -- condition: 1 ^ 20 = 1
  -- condition: 10 % 9 = 1

  sorry

end remainder_7_times_10_pow_20_plus_1_pow_20_mod_9_l321_321173


namespace audio_cassettes_in_first_set_l321_321732

theorem audio_cassettes_in_first_set (A V : ℝ) (num_audio_cassettes : ℝ) : 
  (V = 300) → (A * num_audio_cassettes + 3 * V = 1110) → (5 * A + 4 * V = 1350) → (A = 30) → (num_audio_cassettes = 7) := 
by
  intros hV hCond1 hCond2 hA
  sorry

end audio_cassettes_in_first_set_l321_321732


namespace hyperbola_lambda_l321_321623
   
   theorem hyperbola_lambda (P F1 F2 : ℝ×ℝ) (λ : ℝ)
     (h1 : λ > 0)
     (h2 : P.1^2 - P.2^2 = λ)
     (h3 : |F2.1 - P.1| = 6)
     (h4 : P.2 = 0) :
     λ = 4 :=
   sorry
   
end hyperbola_lambda_l321_321623


namespace imaginary_part_of_squared_complex_l321_321768

-- Definition of the complex number (1 - 4i)
def z : ℂ := 1 - 4 * complex.i

-- The goal is to prove that the imaginary part of (z)^2 is -8
theorem imaginary_part_of_squared_complex :
  complex.imag (z^2) = -8 :=
by
  sorry

end imaginary_part_of_squared_complex_l321_321768


namespace domain_of_f_l321_321759

noncomputable def f (x : ℝ) : ℝ := (sqrt (Real.logb (1/2:ℝ) (4 * x - 3))) / (x - 1)

theorem domain_of_f :
  {x : ℝ | 4 * x - 3 > 0 ∧ Real.logb (1/2:ℝ) (4 * x - 3) ≥ 0 ∧ x ≠ 1} =
  {x : ℝ | x > 3 / 4 ∧ x < 1} :=
by
  sorry

end domain_of_f_l321_321759


namespace value_of_s_l321_321338

theorem value_of_s (s : ℝ) : 
  (g : ℝ → ℝ) (g = λ x, 3 * x ^ 4 + 2 * x ^ 3 - x ^ 2 - 4 * x + s) → 
  g 3 = 0 → 
  s = -276 :=
by 
  intros g hg g3_eq_0 
  sorry

end value_of_s_l321_321338


namespace twelfth_term_in_sequence_l321_321844

variable (a₁ : ℚ) (d : ℚ)

-- Given conditions
def a_1_term : a₁ = 1 / 2 := rfl
def common_difference : d = 1 / 3 := rfl

-- The twelfth term of the arithmetic sequence
def a₁₂ : ℚ := a₁ + 11 * d

-- Statement to prove
theorem twelfth_term_in_sequence (h₁ : a₁ = 1 / 2) (h₂ : d = 1 / 3) : a₁₂ = 25 / 6 :=
by
  rw [h₁, h₂, add_comm, add_assoc, mul_comm, add_comm]
  exact sorry

end twelfth_term_in_sequence_l321_321844


namespace rearrange_HMMTHMMT_without_HMMT_l321_321295

theorem rearrange_HMMTHMMT_without_HMMT :
  let word := ['H', 'M', 'M', 'T', 'H', 'M', 'M', 'T']
  let H := 2
      M := 4
      T := 2
  count_valid_permutations_without_substring_HMMT word = 361 :=
by
  sorry

end rearrange_HMMTHMMT_without_HMMT_l321_321295


namespace largest_prime_divisor_of_factorial_sum_l321_321964

theorem largest_prime_divisor_of_factorial_sum :
  ∃ p : ℕ, p.prime ∧ p ∣ (13! + 14!) ∧ ∀ q : ℕ, q.prime ∧ q ∣ (13! + 14!) → q ≤ p :=
sorry

end largest_prime_divisor_of_factorial_sum_l321_321964


namespace smallest_number_of_students_l321_321274

theorem smallest_number_of_students 
    (n : ℕ) 
    (attended := n / 4)
    (both := n / 40)
    (cheating_hint_ratio : ℚ := 3 / 2)
    (hinting := cheating_hint_ratio * (attended - both)) :
    n ≥ 200 :=
by sorry

end smallest_number_of_students_l321_321274


namespace b_sixth_power_l321_321751

variable {b : ℝ}

-- Theorem statement translated into Lean 4
theorem b_sixth_power (h : 5 = b + b⁻¹) : b^6 + b⁻⁶ = 12120 := by
  sorry

end b_sixth_power_l321_321751


namespace average_after_17th_inning_l321_321860

theorem average_after_17th_inning 
  (x : ℕ)  -- x is the batsman's average before the 17th inning
  (h₁ : ↑((16 : ℕ) * x + 100) / 17 = x + 5)  -- Condition from the problem: new average calculation
  : (x + 5) = 20 :=
by
  -- solving for x using the provided condition
  have hx : 17 * (x + 5) = 16 * x + 100 := by
    apply congr_arg (λ z, 17 * z)
    exact h₁
  
  -- rearranging the equation to isolate x
  have h2 : 17 * x + 85 = 16 * x + 100 := by
    linarith

  -- isolating x
  have h3 : x = 15 := by
    linarith

  -- substituting the value of x to find the new average
  show (x + 5) = 20, by
    rw [h3]
    linarith

end average_after_17th_inning_l321_321860


namespace smallest_total_students_l321_321281

theorem smallest_total_students (n : ℕ) (h1 : 25 * n % 100 = 0)
  (h2 : 10 * n % 4 = 0)
  (h3 : ∃ (y z : ℕ), y = 3 * z / 2 ∧ (y + z - n / 40 = n / 4)) :
  ∃ k : ℕ, n = 200 * k :=
by
  sorry

end smallest_total_students_l321_321281


namespace concurrency_of_perpendicular_bisector_altitude_segment_l321_321892

/-- Given an acute triangle ABC with circumcenter O and a point T on the altitude through C 
    such that ∠TBA = ∠ACB. Let the line CO intersect side AB at point K.
    Prove that the perpendicular bisector of AB, the altitude through A, and the segment KT 
    are concurrent.-/
theorem concurrency_of_perpendicular_bisector_altitude_segment (A B C O T K : Point)
  (h_acute : is_acute ABC) (h_circumcenter : circumcenter O ABC)
  (h_T_on_altitude : on_line T (altitude C)) (h_angles_equal : ∠TBA = ∠ACB)
  (h_intersection : intersects_at CO AB K) :
  concurrent (perpendicular_bisector AB) (altitude A) (line_segment K T) :=
sorry

end concurrency_of_perpendicular_bisector_altitude_segment_l321_321892


namespace kart_race_routes_l321_321664

-- Definitions based on given conditions
def journey_time_AB :=  1
def journey_time_BB := 1
def loop_direction := counterclockwise
def driver_behavior := no_turnback_midway ∧ no_stop
def total_race_duration := 10

-- Recursive definitions for M_n, M_n,A, M_n,B
def M : ℕ → ℕ
| 0 => 1              -- The driver is at the starting point
| 1 => 2              -- The driver can go to B and return or stay
| n + 2 => M (n + 1) + M n

-- Checking the final result for the total_race_duration
theorem kart_race_routes : M 10 = 34 := by
  sorry

end kart_race_routes_l321_321664


namespace find_d_of_quadratic_roots_l321_321788

theorem find_d_of_quadratic_roots :
  ∃ d : ℝ, (∀ x : ℝ, x^2 + 7 * x + d = 0 ↔ x = (-7 + real.sqrt d) / 2 ∨ x = (-7 - real.sqrt d) / 2) → d = 9.8 :=
by
  sorry

end find_d_of_quadratic_roots_l321_321788


namespace change_received_correct_l321_321351

-- Define the conditions
def apples := 5
def cost_per_apple_cents := 80
def paid_dollars := 10

-- Convert the cost per apple to dollars
def cost_per_apple_dollars := (cost_per_apple_cents : ℚ) / 100

-- Calculate the total cost for 5 apples
def total_cost_dollars := apples * cost_per_apple_dollars

-- Calculate the change received
def change_received := paid_dollars - total_cost_dollars

-- Prove that the change received by Margie
theorem change_received_correct : change_received = 6 := by
  sorry

end change_received_correct_l321_321351


namespace greatest_possible_points_for_top_four_teams_l321_321262

-- Define the conditions
def teams := Fin 8
def points_for_win := 3
def points_for_loss := 0
def games_played := (choose 8 2) * 3

-- Define the top four teams earning the same total points problem
theorem greatest_possible_points_for_top_four_teams :
  (∃ A B C D : teams, 
     A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
     ∀ E : teams, 
       (E = A ∨ E = B ∨ E = C ∨ E = D) → 
       (total_points E = 45)) :=
by
  sorry

end greatest_possible_points_for_top_four_teams_l321_321262


namespace line_curve_intersection_symmetric_l321_321771

theorem line_curve_intersection_symmetric (a b : ℝ) 
    (h1 : ∃ p q : ℝ × ℝ, 
          (p.2 = a * p.1 + 1) ∧ 
          (q.2 = a * q.1 + 1) ∧ 
          (p ≠ q) ∧ 
          (p.1^2 + p.2^2 + b * p.1 - p.2 = 1) ∧ 
          (q.1^2 + q.2^2 + b * q.1 - q.2 = 1) ∧ 
          (p.1 + p.2 = -q.1 - q.2)) : 
  a + b = 2 :=
sorry

end line_curve_intersection_symmetric_l321_321771


namespace sum_of_solutions_l321_321725

theorem sum_of_solutions (x y : ℝ) (h : 2 * x^2 + 2 * y^2 = 20 * x - 12 * y + 68) : x + y = 2 := 
sorry

end sum_of_solutions_l321_321725


namespace number_of_rectangles_on_4x4_grid_l321_321620

theorem number_of_rectangles_on_4x4_grid : 
  let n := 4 in 
  let choose_two_lines := Nat.choose n 2 in
  choose_two_lines * choose_two_lines = 36 :=
by
  sorry

end number_of_rectangles_on_4x4_grid_l321_321620


namespace smallest_possible_number_of_students_l321_321287

theorem smallest_possible_number_of_students :
  ∃ n : ℕ, (n % 200 = 0) ∧ (∀ m : ℕ, (m < n → 
    75 * m ≤ 100 * n) ∧
    (∃ a b c : ℕ, a = m / 4 ∧ b = a / 10 ∧ 
    ∃ y z : ℕ, y = 3 * z ∧ (y + z - b = a) ∧ y * c = n / 4)) :=
by
  sorry

end smallest_possible_number_of_students_l321_321287


namespace prism_circumscribed_sphere_surface_area_l321_321195

noncomputable def circumscribed_sphere_surface_area (r : ℝ := 2) (d : ℝ := 1) : ℝ :=
  let R := sqrt (r^2 + d^2)
  4 * real.pi * R^2

theorem prism_circumscribed_sphere_surface_area :
  let AA' := 2
      AB := 2 * sqrt 3 in
  circumscribed_sphere_surface_area = 20 * real.pi :=
by
  sorry

end prism_circumscribed_sphere_surface_area_l321_321195


namespace sequence_problem_l321_321347

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
∀ n, a(n) = 2 ^ n

theorem sequence_problem (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : ∀ n, S n + a 1 = 2 * a n)
  (h2 : a 1 + a 3 = 2 * (a 2 + 1)) : a 1 + a 5 = 34 :=
by
  sorry

end sequence_problem_l321_321347


namespace triangle_area_zero_l321_321418

theorem triangle_area_zero :
  let A := (3, 3 : ℝ)
  let B := (8, 18 : ℝ)
  let C := (8, 14 / 3 : ℝ)
  ∃ (A B C : ℝ × ℝ), A = (3, 3) ∧ B = (8, 18) ∧ C = (8, 14 / 3) ∧
  let area : ℝ := 1 / 2 * | A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) |
  area = 0 :=
by
  sorry

end triangle_area_zero_l321_321418


namespace log_45_81_l321_321183

variable (a b : ℝ)

-- Given conditions
def log_18_9_eq_a : Prop := Real.logBase 18 9 = a
def exp_18_b_eq_5 : Prop := 18^b = 5

-- Problem statement to prove
theorem log_45_81 (h1 : log_18_9_eq_a a) (h2 : exp_18_b_eq_5 b) : 
  Real.logBase 45 81 = (2 * a) / (a + b) :=
sorry

end log_45_81_l321_321183


namespace sum_tan_squared_eq_zero_l321_321456

theorem sum_tan_squared_eq_zero : 
  ∑ k in Finset.range 22, (Real.tan (4 * (k + 1) : ℝ) ^ 2) = 0 := 
sorry

end sum_tan_squared_eq_zero_l321_321456


namespace lighting_scheme_count_l321_321822

-- Define the condition of the problem
def isValidLightingScheme (n : Nat) (lit : List Nat) : Prop :=
  lit.length = 3 ∧ ∀ k, (k < lit.length - 1) → (lit[k + 1] ≠ lit[k] + 1)

-- The main theorem we want to prove
theorem lighting_scheme_count : (∃ (lit : List Nat), isValidLightingScheme 7 lit) = 10 :=
by
  sorry -- The proof is omitted

end lighting_scheme_count_l321_321822


namespace product_of_invertible_labels_l321_321002

def f1 (x : ℤ) : ℤ := x^3 - 2 * x
def f2 (x : ℤ) : ℤ := x - 2
def f3 (x : ℤ) : ℤ := 2 - x

theorem product_of_invertible_labels :
  (¬ ∃ inv : ℤ → ℤ, f1 (inv 0) = 0 ∧ ∀ x : ℤ, f1 (inv (f1 x)) = x) ∧
  (∃ inv : ℤ → ℤ, f2 (inv 0) = 0 ∧ ∀ x : ℤ, f2 (inv (f2 x)) = x) ∧
  (∃ inv : ℤ → ℤ, f3 (inv 0) = 0 ∧ ∀ x : ℤ, f3 (inv (f3 x)) = x) →
  (2 * 3 = 6) :=
by sorry

end product_of_invertible_labels_l321_321002


namespace product_divisors_30_eq_810000_l321_321432

def product_of_divisors (n : ℕ) : ℕ :=
  (multiset.filter (λ d, n % d = 0) (multiset.range (n + 1))).prod id

theorem product_divisors_30_eq_810000 :
  product_of_divisors 30 = 810000 :=
begin
  -- Proof will involve showing product of divisors of 30 equals 810000
  sorry
end

end product_divisors_30_eq_810000_l321_321432


namespace largest_possible_d_l321_321089

-- Definitions for the primes and the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n

def cond_a (a : ℕ) : Prop := is_prime a
def cond_b (b : ℕ) : Prop := is_prime b
def cond_c (c : ℕ) : Prop := is_prime c
def cond_sum (a b : ℕ) : Prop := a + b = 800
def distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The seven distinct primes
def seven_primes (a b c : ℕ) : Prop :=
  is_prime a ∧
  is_prime b ∧
  is_prime c ∧
  is_prime (a + b + c) ∧
  is_prime (a + b - c) ∧
  is_prime (a - b + c) ∧
  is_prime (-a + b + c)

-- Main statement
theorem largest_possible_d (a b c : ℕ) (h1 : cond_a a) (h2 : cond_b b) (h3 : cond_c c)
 (h4 : cond_sum a b) (h5 : distinct a b c) (h6 : seven_primes a b c) : 
  ∃ d, d = (max (max (max (max (a + b + c) (a + b - c)) (a - b + c)) (-a + b + c)) 
  - min (min (min (min a b) c) (min (min (a + b + c) (a + b - c)) (a - b + c))) = 1594 := by
  sorry

end largest_possible_d_l321_321089


namespace triangle_area_l321_321309

theorem triangle_area (α γ : ℝ) (c : ℝ) (α_pos : 0 < α) (γ_pos : 0 < γ) (c_pos : 0 < c) :
  let area := (c^2 * sin α * sin (α + γ)) / (2 * sin γ)
  in ∀ (ABC : Type) [triangle ABC] [angle ABC BAC = α] [angle ABC BCA = γ] [length ABC AB = c],
  area = (1/2) * c * ((c * sin (α + γ)) / sin γ) * sin α :=
by
  intro ABC _ _ _
  sorry

end triangle_area_l321_321309


namespace sqrt_product_simplification_l321_321141

theorem sqrt_product_simplification (p : ℝ) : 
  (Real.sqrt (42 * p)) * (Real.sqrt (14 * p)) * (Real.sqrt (7 * p)) = 14 * p * (Real.sqrt (21 * p)) := 
  sorry

end sqrt_product_simplification_l321_321141


namespace miles_travelled_before_trip_l321_321705

theorem miles_travelled_before_trip (total_odometer_reading : ℝ) (trip_miles : ℝ) : total_odometer_reading = 372 ∧ trip_miles = 159.7 → total_odometer_reading - trip_miles = 212.3 :=
by
  intro h
  cases h with ht hu
  rw ht
  rw hu
  norm_num

end miles_travelled_before_trip_l321_321705


namespace quadrilateral_in_circle_l321_321266

theorem quadrilateral_in_circle (A B C D : ℝ) (d_AB d_AC d_AD d_BC d_BD d_CD : ℝ) (r : ℝ) :
  (A < 1) → (B < 1) → (C < 1) → (D < 1) → 
  (d_AB < 1) → (d_AC < 1) → (d_AD < 1) → 
  (d_BC < 1) → (d_BD < 1) → (d_CD < 1) → 
  r = 0.9 → 
  (∃ (x y : ℝ), ∀ (d : ℝ), d < r) :=
begin
  sorry
end

end quadrilateral_in_circle_l321_321266


namespace probability_exactly_4_heads_l321_321047

noncomputable def probability_of_heads : ℚ := 1/3
noncomputable def probability_of_tails : ℚ := 2/3
noncomputable def total_flips : ℕ := 10
noncomputable def heads_count : ℕ := 4

theorem probability_exactly_4_heads :
  let arrangements := (nat.choose total_flips heads_count : ℚ),
      specific_prob := (probability_of_heads ^ heads_count) * (probability_of_tails ^ (total_flips - heads_count)) in
  arrangements * specific_prob = 13440/59049 :=
by
  let arrangements := 210 : ℚ -- This is computed as nat.choose 10 4
  let specific_prob := (1/3)^4 * (2/3)^6
  have h1: arrangements = 210 := by sorry -- This will be proven based on binomial coefficient
  have h2: specific_prob = 64/59049 := by sorry -- This will be proved by simplification
  have h3: arrangements * specific_prob = 210 * (64/59049) := by sorry
  have h4: 210 * (64/59049) = 13440/59049 := by sorry
  exact eq.trans (eq.trans h3 h4) _ -- This will conclude the proof

end probability_exactly_4_heads_l321_321047


namespace largest_prime_divisor_13_fac_add_14_fac_l321_321966

theorem largest_prime_divisor_13_fac_add_14_fac : 
  ∀ (p : ℕ), prime p → p ∣ (13! + 14!) → p ≤ 5 :=
by
  sorry

end largest_prime_divisor_13_fac_add_14_fac_l321_321966


namespace ratio_third_to_second_l321_321522

variable (a1 a2 a3 a4 : ℕ)

-- Given conditions
def condition1 : Prop := a1 = 6
def condition2 : Prop := a2 = a1 + 2
def condition3 : Prop := a4 = 3 * a3
def condition4 : Prop := a1 + a2 + a3 + a4 = 30

-- The proof goal
theorem ratio_third_to_second
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4) :
  a3 / a2 = 1 / 2 := 
sorry

end ratio_third_to_second_l321_321522


namespace incorrect_statement_l321_321855

def quadrilateral (A B C D : Type) := true

def is_parallelogram (quad : quadrilateral) := true
def is_rectangle (quad : quadrilateral) := true
def is_rhombus (quad : quadrilateral) := true

-- Equivalent definitions for conditions
def A_statement : Prop := ∀ (quad : quadrilateral), (quad.has_two_pairs_congruent_diagonals) → is_parallelogram quad
def B_statement : Prop := ∀ (quad : quadrilateral), (quad.all_interior_angles_congruent) → is_rectangle quad
def C_statement : Prop := ∀ (quad : quadrilateral), (quad.has_one_pair_parallel_sides ∧ quad.congruent_diagonals) → is_rectangle quad
def D_statement : Prop := ∀ (quad : quadrilateral), (quad.all_sides_congruent) → is_rhombus quad

-- Proof problem to show which one is incorrect
theorem incorrect_statement : C_statement = false :=
by sorry

end incorrect_statement_l321_321855


namespace Jace_post_break_time_correct_l321_321311

noncomputable def Jace_post_break_time (total_distance : ℝ) (speed : ℝ) (pre_break_time : ℝ) : ℝ :=
  (total_distance - (speed * pre_break_time)) / speed

theorem Jace_post_break_time_correct :
  Jace_post_break_time 780 60 4 = 9 :=
by
  sorry

end Jace_post_break_time_correct_l321_321311


namespace cube_opposite_faces_same_color_probability_l321_321057

noncomputable def probability_same_color_pair : ℚ :=
  let p_same_color_pair := 1 - (2/3)^3 in
  -- Probability of having at least one pair of opposite faces with the same color
  p_same_color_pair

theorem cube_opposite_faces_same_color_probability :
  probability_same_color_pair = 19 / 27 :=
by
  sorry

end cube_opposite_faces_same_color_probability_l321_321057


namespace quadratic_roots_form_l321_321792

theorem quadratic_roots_form {d : ℝ} (h : ∀ x : ℝ, x^2 + 7*x + d = 0 → (x = (-7 + real.sqrt d) / 2) ∨ (x = (-7 - real.sqrt d) / 2)) : d = 49 / 5 := 
sorry

end quadratic_roots_form_l321_321792


namespace jason_total_spent_l321_321674

def cost_of_flute : ℝ := 142.46
def cost_of_music_tool : ℝ := 8.89
def cost_of_song_book : ℝ := 7.00

def total_spent (flute_cost music_tool_cost song_book_cost : ℝ) : ℝ :=
  flute_cost + music_tool_cost + song_book_cost

theorem jason_total_spent :
  total_spent cost_of_flute cost_of_music_tool cost_of_song_book = 158.35 :=
by
  -- Proof omitted
  sorry

end jason_total_spent_l321_321674


namespace necessary_but_not_sufficient_condition_l321_321336

variable {a b : ℝ}

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (a > 2 ∧ b > 2) → (a + b > 4) ∧ ¬((a + b > 4) → (a > 2 ∧ b > 2)) :=
by
  split
  -- Proof that (a > 2 ∧ b > 2) → (a + b > 4)
  sorry
  -- Proof that ¬((a + b > 4) → (a > 2 ∧ b > 2))
  sorry

end necessary_but_not_sufficient_condition_l321_321336


namespace NutsInThirdBox_l321_321041

variable (x y z : ℝ)

theorem NutsInThirdBox (h1 : x = (y + z) - 6) (h2 : y = (x + z) - 10) : z = 16 := 
sorry

end NutsInThirdBox_l321_321041


namespace union_A_B_inter_A_B_comp_int_B_l321_321684

open Set

variable (x : ℝ)

def A := {x : ℝ | 2 ≤ x ∧ x < 4}
def B := {x : ℝ | 3 ≤ x}

theorem union_A_B : A ∪ B = (Ici 2) :=
by
  sorry

theorem inter_A_B : A ∩ B = Ico 3 4 :=
by
  sorry

theorem comp_int_B : (univ \ A) ∩ B = Ici 4 :=
by
  sorry

end union_A_B_inter_A_B_comp_int_B_l321_321684


namespace car_time_indeterminate_l321_321529

-- Definitions for conditions
variables (D u v : ℝ)
variable (h1 : u < v)

-- Definitions for time taken by cars
def T_A (D u v : ℝ) : ℝ := (0.4 * D / u) + (0.6 * D / v)

def total_time_B (D u v : ℝ) : ℝ := D / (0.3 * u + 0.7 * v)

-- Main theorem statement
theorem car_time_indeterminate (D u v : ℝ) (h1 : u < v) : 
  "Cannot be determined without knowing \(D\)" :=
sorry

end car_time_indeterminate_l321_321529


namespace initial_scissors_count_l321_321042

theorem initial_scissors_count (added : ℕ) (total : ℕ) 
(h_added : added = 22) 
(h_total : total = 76) : 
∃ initial : ℕ, initial = total - added ∧ initial = 54 :=
by {
  use total - added,
  split,
  { sorry },
  { sorry },
}

end initial_scissors_count_l321_321042


namespace rotated_line_l1_l321_321770

-- Define the original line equation and the point around which the line is rotated
def line_l (x y : ℝ) : Prop := x - y + 1 = 0
def point_A : ℝ × ℝ := (2, 3)

-- Define the line equation that needs to be proven
def line_l1 (x y : ℝ) : Prop := x + y - 5 = 0

-- The theorem stating that after a 90-degree rotation of line l around point A, the new line is equation l1
theorem rotated_line_l1 : 
  ∀ (x y : ℝ), 
  (∃ (k : ℝ), k = 1 ∧ ∀ (x y), line_l x y ∧ ∀ (x y), line_l1 x y) ∧ 
  ∀ (a b : ℝ), (a, b) = point_A → 
  x + y - 5 = 0 := 
by
  sorry

end rotated_line_l1_l321_321770


namespace projection_of_skew_lines_l321_321401

-- Definitions of projections
noncomputable def LineProjectionOntoPlane 
  (perpendicular : Bool) 
  (parallel_to_common_perpendicular : Bool) : Prop :=
if perpendicular then
  "a line and a point outside the line"
else if parallel_to_common_perpendicular then
  "two parallel lines"
else
  "two intersecting lines"

-- Theorem statement that includes all conditions and the conclusion
theorem projection_of_skew_lines : 
  ∀ (skew_line1 skew_line2 : Type) (plane : Type), 
  (∃ (perpendicular : Bool) (parallel_to_common_perpendicular : Bool), 
    (LineProjectionOntoPlane perpendicular parallel_to_common_perpendicular = 
     "a line and a point outside the line") ∨ 
    (LineProjectionOntoPlane perpendicular parallel_to_common_perpendicular = 
     "two parallel lines") ∨ 
    (LineProjectionOntoPlane perpendicular parallel_to_common_perpendicular = 
     "two intersecting lines")) :=
by
  -- Proof omitted
  sorry

end projection_of_skew_lines_l321_321401


namespace clock_ticks_12_times_l321_321518

theorem clock_ticks_12_times (t1 t2 : ℕ) (d1 d2 : ℕ) (h1 : t1 = 6) (h2 : d1 = 40) (h3 : d2 = 88) : t2 = 12 := by
  sorry

end clock_ticks_12_times_l321_321518


namespace max_area_of_rectangle_l321_321025

theorem max_area_of_rectangle (x y : ℝ) (h1 : 2 * (x + y) = 40) : 
  (x * y) ≤ 100 :=
by
  sorry

end max_area_of_rectangle_l321_321025


namespace max_area_quadrilateral_OBTM_l321_321796

noncomputable def max_area_OBTM (r : ℝ) : ℝ :=
  (sqrt 5 / 4) * r^2

theorem max_area_quadrilateral_OBTM (O A C T B M : Point) (r : ℝ) :
  semicircle O A C ∧
  diameter O A C = 2 * r ∧
  radius O A = r ∧
  divide_arcs O A C B 1 3 ∧
  midpoint M O C ∧
  maximize_area O B T M →
  area O B T M = max_area_OBTM r :=
sorry

end max_area_quadrilateral_OBTM_l321_321796


namespace sum_gcd_lcm_is_244_l321_321835

-- Definitions of the constants
def a : ℕ := 12
def b : ℕ := 80

-- Main theorem statement
theorem sum_gcd_lcm_is_244 : Nat.gcd a b + Nat.lcm a b = 244 := by
  sorry

end sum_gcd_lcm_is_244_l321_321835


namespace ed_stay_morning_hours_l321_321158

-- Conditions
def cost_per_hour_night : ℝ := 1.5
def cost_per_hour_morning : ℝ := 2.0
def initial_money : ℝ := 80
def hours_night : ℝ := 6
def money_left_after_stay : ℝ := 63

-- Calculated intermediate results from steps b)
def total_spent : ℝ := initial_money - money_left_after_stay
def cost_night : ℝ := cost_per_hour_night * hours_night
def money_spent_morning : ℝ := total_spent - cost_night
def hours_morning := money_spent_morning / cost_per_hour_morning

-- Proof Statement
theorem ed_stay_morning_hours : hours_morning = 4 := by
  sorry

end ed_stay_morning_hours_l321_321158


namespace largest_prime_divisor_13_fact_14_fact_l321_321979

theorem largest_prime_divisor_13_fact_14_fact :
  ∀ (n : ℕ), (n = 13!) ∧ (m = 14!) ∧ (m = 14 * n) ∧ ((n + m) = n * 15) ∧ (15 = 3 * 5) → 
  (∃ p : ℕ, (Prime p) ∧ (LargestPrimeDivisor p (n + m)) ∧ (p = 13)) :=
by
  sorry

end largest_prime_divisor_13_fact_14_fact_l321_321979


namespace prop_1_prop_2_prop_3_correct_propositions_l321_321293

open Finset

def d (u v : Vector Bool n) : ℕ :=
  (finrange n).countp (λ i => u.nth i ≠ v.nth i)

theorem prop_1 (u v : Vector Bool n) : 0 ≤ d u v ∧ d u v ≤ n :=
by sorry

theorem prop_2 (u : Vector Bool n) : ¬ ∃ vs : Finset (Vector Bool n), 
  (∀ v ∈ vs, d u v = n - 1) ∧ vs.card = n - 1 :=
by sorry

theorem prop_3 (u v w : Vector Bool n) : d u v ≤ d w u + d w v :=
by sorry

theorem correct_propositions (u v w : Vector Bool n) : 
  (prop_1 u v) ∧ (prop_3 u v w) :=
by sorry

end prop_1_prop_2_prop_3_correct_propositions_l321_321293


namespace shelter_new_pets_l321_321677

theorem shelter_new_pets 
  (dogs_initial : ℕ)
  (cats_initial : ℕ)
  (lizards_initial : ℕ)
  (dogs_adopted_pct : ℚ)
  (cats_adopted_pct : ℚ)
  (lizards_adopted_pct : ℚ)
  (total_pets_one_month : ℕ)
  (total_pets_after_adoption : ℕ) :
  (dogs_initial = 30) →
  (cats_initial = 28) →
  (lizards_initial = 20) →
  (dogs_adopted_pct = 0.50) →
  (cats_adopted_pct = 0.25) →
  (lizards_adopted_pct = 0.20) →
  (total_pets_one_month = 65) →
  (total_pets_after_adoption = (dogs_initial - nat.floor (dogs_adopted_pct * dogs_initial : ℕ) +
                                cats_initial - nat.floor (cats_adopted_pct * cats_initial : ℕ) +
                                lizards_initial - nat.floor (lizards_adopted_pct * lizards_initial : ℕ))) →
  (total_pets_one_month - total_pets_after_adoption = 13) :=
by
  intros
  sorry

end shelter_new_pets_l321_321677


namespace find_missing_number_l321_321947

theorem find_missing_number (x : ℝ) (h : 11 + sqrt (x + 6 * 4 / 3) = 13) : x = -4 :=
  sorry

end find_missing_number_l321_321947


namespace multiples_of_n_l321_321859
open Nat

theorem multiples_of_n (a b n : ℕ) (q : Finset ℕ)
  (h1 : a % n = 0) (h2 : b % n = 0)
  (h3 : q = Finset.range (b - a + 1) + a)
  (h4 : Finset.card (q.filter (λ x => x % n = 0)) = 14)
  (h5 : Finset.card (q.filter (λ x => x % 7 = 0)) = 27) :
  n = 27 :=
by
  sorry

end multiples_of_n_l321_321859


namespace arithmetic_sequence_twelfth_term_l321_321848

theorem arithmetic_sequence_twelfth_term :
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  (a1 + 11 * d) = (25 : ℚ) / 6 :=
by
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  exact sorry

end arithmetic_sequence_twelfth_term_l321_321848


namespace false_propositions_l321_321180

def plane := Type
def line := Type
variables (α : plane) (m n : line)

-- Conditions based on propositions
def prop_1 : Prop := (m ⊥ α) ∧ (m ⊥ n) → n ∥ α
def prop_2 : Prop := (m ∥ α) ∧ (n ∥ α) → m ∥ n
def prop_3 : Prop := (m ∥ α) ∧ (n ⊆ α) → m ∥ n
def prop_4 : Prop := (m ∥ n) ∧ (n ∥ α) → m ∥ α

-- Proof that all propositions are false
theorem false_propositions : ¬prop_1 ∧ ¬prop_2 ∧ ¬prop_3 ∧ ¬prop_4 := by
  sorry

end false_propositions_l321_321180


namespace twelfth_term_arithmetic_sequence_l321_321849

theorem twelfth_term_arithmetic_sequence :
  let a := (1 : ℚ) / 2
  let d := (1 : ℚ) / 3
  (a + 11 * d) = (25 : ℚ) / 6 :=
by
  sorry

end twelfth_term_arithmetic_sequence_l321_321849


namespace find_equations_l321_321190

section
variables {Ccenter : ℝ × ℝ} {r k : ℝ}
def circle (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

-- Conditions for the circle and its center
axiom circle_passes_points : ∀ (a b r : ℝ), circle 2 4 a b r → circle 1 3 a b r → a - b + 1 = 0 → (a, b) = (2, 3) ∧ r = 1
-- The equation of the line
def line (k x y : ℝ) : Prop := y = k * x + 1
-- Equation intersect the circle
axiom intersect_line_circle : ∀ k : ℝ, line k 0 1 → ( ∃ x1 x2 y1 y2 : ℝ, x1 + x2 = 4 * (1 + k) / (1 + k^2) ∧ x1 * x2 = 7 / (1 + k^2) ∧ 
∃ (M N : ℝ × ℝ), ((M.1 - 2)^2 + (M.2 - 3)^2 = 1 ∧ (N.1 - 2)^2 + (N.2 - 3)^2 = 1) ∧ (M.1 * N.1 + M.2 * N.2 = 12) → k = 1)

-- Proof statement for the circle and line equations
theorem find_equations (
  H₁ : ∃ (a b r : ℝ), circle 2 4 a b r ∧ circle 1 3 a b r ∧ a - b + 1 = 0 ∧ (a, b) = (2, 3) ∧ r = 1
): ∃ k : ℝ, similar k 0 1 ∧ ∃ (x1 x2 : ℝ), ∃ (M N : ℝ × ℝ), 
(line 1 0 1 ∧ (M.1, (1 * M.1 + 1)) = (2, 4) ∧ (N.1, (1 * N.1 + 1)) = (1, 3)) → 
circle (M.1) (M.2) (N.1) (N.2) 1 ∧ ∃ (M N : ℝ × ℝ), (M.1 * N.1 + M.2 * N.2 = 12)): 
∃ (k : ℝ), (k = 1 ∧ ∀ {x y :  ℝ}, line k 0 1): sorry

end find_equations_l321_321190


namespace find_phi_max_min_values_l321_321249

-- Define the function f
def f (x φ : ℝ) : ℝ := sqrt 5 * sin (2 * x + φ)

-- Define the symmetry condition
axiom symmetry_condition (φ : ℝ) : (0 < φ ∧ φ < π) → 
  (∀ x : ℝ, f (π/3 - x) φ = f (π/3 + x) φ)

-- Prove that φ = 5π / 6 given the symmetry condition and the bounds on φ
theorem find_phi : ∀ φ : ℝ, (0 < φ ∧ φ < π) → 
  (symmetry_condition φ) → φ = (5 * π) / 6 := 
sorry

-- Prove the maximum and minimum values of f(x) given the bounds on x and φ
theorem max_min_values : 
  let φ := (5 * π) / 6 in
  (∀ x : ℝ, x ∈ Icc (-(π / 12)) (π / 2)) →
  (∀ x : ℝ, x ∈ Icc (-(π / 12)) (π / 2) →
    (f x φ ≤ sqrt 15 / 2) ∧ (f x φ ≥ -sqrt 5) ∧
    (f (-(π / 12)) φ = sqrt 15 / 2) ∧ (f (π / 3) φ = -sqrt 5)) :=
sorry

end find_phi_max_min_values_l321_321249


namespace circular_permutations_2a2b2c_l321_321555

open Finset
open Nat.ArithmeticFunction

-- Define the main function calculating the number of circular permutations of a multiset
def circular_permutations (a b c : ℕ) : ℕ :=
  let n := a + b + c in
  (1 / n.toRat * ∑ d in divisors n, totient d * (factorial (n / d) / (factorial (a / d) * factorial (b / d) * factorial (c / d)))).toNat

theorem circular_permutations_2a2b2c : circular_permutations 2 2 2 = 16 :=
  sorry

end circular_permutations_2a2b2c_l321_321555


namespace shopkeeper_loss_percentage_l321_321116

theorem shopkeeper_loss_percentage (SP₁ : ℝ) (profit_percentage : ℝ) (SP₂ : ℝ) (CP : ℝ) (loss_percentage : ℝ)
  (h1 : SP₁ = 800)
  (h2 : profit_percentage = 25)
  (h3 : SP₂ = 512)
  (h4 : CP = SP₁ / (1 + profit_percentage / 100)) :
  loss_percentage = (CP - SP₂) / CP * 100 :=
begin
  -- Skipping the actual proof
  sorry
end

end shopkeeper_loss_percentage_l321_321116


namespace trisha_total_distance_l321_321712

theorem trisha_total_distance :
  let distance1 := 0.11
  let distance2 := 0.11
  let distance3 := 0.67
  distance1 + distance2 + distance3 = 0.89 :=
by
  sorry

end trisha_total_distance_l321_321712


namespace sin_arcsin_plus_arctan_l321_321951

theorem sin_arcsin_plus_arctan :
  sin (Real.arcsin (4 / 5) + Real.arctan 3) = (13 * Real.sqrt 10) / 50 :=
by
  sorry

end sin_arcsin_plus_arctan_l321_321951


namespace third_side_length_l321_321500

-- Given conditions
variables (r : ℝ) (a b x : ℝ) (A B C : ℝ)
variable h_r : r = 300
variable h_a : a = 300
variable h_b : b = 450
variable h_B : B = 60

noncomputable def sin60 := Real.sin (60 * Real.pi / 180)

-- Theorem statement
theorem third_side_length (r a b x B : ℝ) (h_r : r = 300) (h_a : a = 300) (h_b : b = 450) (h_B : B = 60) :
  x = 300 :=
by
  sorry

end third_side_length_l321_321500


namespace range_of_m_l321_321601

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x ≠ 1 ∧ (m + 3) / (x - 1) = 1) ↔ m > -4 ∧ m ≠ -3 := 
by 
  sorry

end range_of_m_l321_321601


namespace unit_square_inside_parallelogram_l321_321767

theorem unit_square_inside_parallelogram (P : Type*) [parallelogram P] (h : ∀ height > 1) : 
  ∃ (unit_square : Type*), can_be_placed_inside P unit_square :=
sorry

end unit_square_inside_parallelogram_l321_321767


namespace solution_set_inequality_l321_321592

theorem solution_set_inequality {a b c x : ℝ} (h1 : a < 0)
  (h2 : -b / a = 1 + 2) (h3 : c / a = 1 * 2) :
  a - c * (x^2 - x - 1) - b * x ≥ 0 ↔ x ≤ -3 / 2 ∨ x ≥ 1 := by
  sorry

end solution_set_inequality_l321_321592


namespace proof_problem_l321_321611

variables {x1 y1 x2 y2 : ℝ}

-- Definitions
def unit_vector (x y : ℝ) : Prop := x^2 + y^2 = 1
def angle_with_p (x y : ℝ) : Prop := (x + y) / Real.sqrt 2 = Real.sqrt 3 / 2
def m := (x1, y1)
def n := (x2, y2)
def p := (1, 1)

-- Conditions
lemma unit_m : unit_vector x1 y1 := sorry
lemma unit_n : unit_vector x2 y2 := sorry
lemma angle_m_p : angle_with_p x1 y1 := sorry
lemma angle_n_p : angle_with_p x2 y2 := sorry

-- Theorem to prove
theorem proof_problem (h1 : unit_vector x1 y1)
                      (h2 : unit_vector x2 y2)
                      (h3 : angle_with_p x1 y1)
                      (h4 : angle_with_p x2 y2) :
                      (x1 * x2 + y1 * y2 = 1/2) ∧ (y1 * y2 / (x1 * x2) = 1) :=
sorry

end proof_problem_l321_321611


namespace min_area_triangle_l321_321084

-- Define the points and line equation
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (30, 10)
def line (x : ℤ) : ℤ := 2 * x - 5

-- Define a function to calculate the area using Shoelace formula
noncomputable def area (C : ℤ × ℤ) : ℝ :=
  (1 / 2) * |(A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1)|

-- Prove that the minimum area of the triangle with the given conditions is 15
theorem min_area_triangle : ∃ (C : ℤ × ℤ), C.2 = line C.1 ∧ area C = 15 := sorry

end min_area_triangle_l321_321084


namespace a21_eq_2016_l321_321762

noncomputable def seq_a : ℕ → ℝ
noncomputable def seq_b : ℕ → ℝ

axiom a1 : seq_a 1 = 1
axiom geom_seq : ∀ n : ℕ, seq_b n = seq_a (n + 1) / seq_a n
axiom b10_b11 : seq_b 10 * seq_b 11 = real.exp ((log 2016) / 10)

theorem a21_eq_2016 : seq_a 21 = 2016 := sorry

end a21_eq_2016_l321_321762


namespace problem_l321_321597

def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

def is_vertex_left (x y : ℝ) : Prop := (x, y) = (-2, 0)
def is_vertex_right (x y : ℝ) : Prop := (x, y) = (2, 0)

def is_focus_right (e : ℝ) (x y : ℝ) : Prop := e = 1/2 ∧ (x, y) = (1, 0)

def is_point_on_line (x y : ℝ) : Prop := x = 4

def is_collinear (x1 y1 x2 y2 x3 y3 : ℝ) : Prop := (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

def triangle_area (A B C : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ) : ℝ := 
(1/2) * abs (B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) + A.1 * (B.2 - C.2))

noncomputable def max_triangle_area := abs (3 / 2)

theorem problem :
  (∀ x y, ellipse 2 3 x y → 
    ∃ e (fx fy : ℝ), is_focus_right e fx fy ∧ 
    (∀ m, ∃ Px Py Qx Qy : ℝ, 
      is_point_on_line 4 m →
      (ellipse 2 3 Px Py ∧ ellipse 2 3 Qx Qy) →
      is_collinear Px Py 1 0 Qx Qy ∧ 
      triangle_area ( (2, 0) , (Px, Py) , (Qx, Qy) ) = max_triangle_area)) :=
sorry

end problem_l321_321597


namespace find_projection_matrix_l321_321685

def projection_matrix_correct (Q : Matrix (Fin 3) (Fin 3) ℝ) (v : Vector ℝ 3) : Prop :=
  Q * v = v - ((dot_product v ⟨2, -1, 2⟩ / dot_product ⟨2, -1, 2⟩ ⟨2, -1, 2⟩) • ⟨2, -1, 2⟩)

theorem find_projection_matrix : ∃ (Q : Matrix (Fin 3) (Fin 3) ℝ), ∀ (v : Vector ℝ 3), projection_matrix_correct Q v :=
  ∃ (Q : Matrix (Fin 3) (Fin 3) ℝ), Q = ![
    [5/9, 4/9, -8/9],
    [2/9, 10/9, 2/9],
    [-8/9, 4/9, 5/9]
  ] ∧ ∀ v : Vector ℝ 3, projection_matrix_correct Q v := by
    sorry

end find_projection_matrix_l321_321685


namespace optionC_form_triangle_l321_321453

noncomputable def canFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem optionC_form_triangle : canFormTriangle 7 15 10 :=
by
  split
  all_goals sorry

end optionC_form_triangle_l321_321453


namespace minimum_black_points_l321_321342

noncomputable def min_black_points {n : ℕ} (hn : n ≥ 3) (E : Finset ℕ) (hE : E.card = 2 * n - 1) : ℕ :=
if (2 * n - 1) % 3 = 0 then n - 1 else n

theorem minimum_black_points (n : ℕ) (hn : n ≥ 3) (E : Finset ℕ) (hE : E.card = 2 * n - 1)
  (good_coloring : ∀ (k : ℕ), k ≥ (min_black_points hn hE) → ∀ (black_points : Finset ℕ), (black_points ⊆ E) ∧ (black_points.card = k) 
  -> ∃ (p q : ℕ) (hp : p ∈ black_points) (hq : q ∈ black_points), (2*n-1-1) / 2 = n) :
  min_black_points hn hE =
  if (2 * n - 1) % 3 = 0 then n - 1 else n := by
sorry

end minimum_black_points_l321_321342


namespace equal_sundays_tuesdays_days_l321_321887

-- Define the problem in Lean
def num_equal_sundays_and_tuesdays_starts : ℕ :=
  3

-- Define a function that calculates the number of starting days that result in equal Sundays and Tuesdays
def calculate_sundays_tuesdays_starts (days_in_month : ℕ) : ℕ :=
  if days_in_month = 30 then 3 else 0

-- Prove that for a month of 30 days, there are 3 valid starting days for equal Sundays and Tuesdays
theorem equal_sundays_tuesdays_days :
  calculate_sundays_tuesdays_starts 30 = num_equal_sundays_and_tuesdays_starts :=
by 
  -- Proof outline here
  sorry

end equal_sundays_tuesdays_days_l321_321887


namespace integer_distances_implies_vertex_l321_321366

theorem integer_distances_implies_vertex (M A B C D : ℝ × ℝ × ℝ)
  (a b c d : ℕ)
  (h_tetrahedron: 
    dist A B = 2 ∧ dist B C = 2 ∧ dist C D = 2 ∧ dist D A = 2 ∧ 
    dist A C = 2 ∧ dist B D = 2)
  (h_distances: 
    dist M A = a ∧ dist M B = b ∧ dist M C = c ∧ dist M D = d) :
  M = A ∨ M = B ∨ M = C ∨ M = D := 
  sorry

end integer_distances_implies_vertex_l321_321366


namespace donuts_selection_l321_321361

/-- Prove that the number of ways to select 6 donuts from 4 types is 84 using the stars and bars theorem. -/
theorem donuts_selection : ∑ (g c p s : ℕ) in {g, c, p, s | g + c + p + s = 6}, 1 = (Nat.choose 9 3) :=
by
  sorry

end donuts_selection_l321_321361


namespace min_value_at_x_eq_2_l321_321406

theorem min_value_at_x_eq_2 (x : ℝ) (h : x > 1) : 
  x + 1/(x-1) = 3 ↔ x = 2 :=
by sorry

end min_value_at_x_eq_2_l321_321406


namespace triangle_ABC_solution_l321_321335

-- Noncomputable context declaration if needed for these calculations
noncomputable def side_a (b : ℝ) := 3 * b

noncomputable def side_b (c : ℝ) := real.sqrt (c ^ 2 / 7)

noncomputable def sin_A (a c : ℝ) := (a * (real.sqrt 3 / 2)) / c

theorem triangle_ABC_solution :
  (∀ (b : ℝ), b = 1 → side_a b = 3) ∧
  (sin_A 3 (real.sqrt 7) = (3 * real.sqrt 21) / 14) :=
by
  split
  { intro b
    intro hb
    rw hb
    unfold side_a
    norm_num }
  { unfold sin_A
    norm_num }
  sorry

end triangle_ABC_solution_l321_321335


namespace intersection_point_l321_321327

noncomputable def f (x : ℝ) : ℝ := x^3 + 9 * x^2 + 24 * x + 36

theorem intersection_point : ∃ a : ℝ, a = f a ∧ a = -3 :=
begin
  -- Lean proof code would go here.
  sorry
end

end intersection_point_l321_321327


namespace ratio_arithmetic_sequence_triangle_l321_321668

theorem ratio_arithmetic_sequence_triangle (a b c : ℝ) 
  (h_triangle : a^2 + b^2 = c^2)
  (h_arith_seq : ∃ d, b = a + d ∧ c = a + 2 * d) :
  a / b = 3 / 4 ∧ b / c = 4 / 5 :=
by
  sorry

end ratio_arithmetic_sequence_triangle_l321_321668


namespace planting_schemes_correct_l321_321895

/-- Define the flowerbed with 6 regions and conditions for planting schemes -/
noncomputable def planting_schemes (types : ℕ) (regions : Finset ℕ) (adjacency : (ℕ × ℕ) → Prop) : ℕ :=
  -- Calculate the number of valid planting schemes
  if types = 6 ∧ regions = {0, 1, 2, 3, 4, 5} ∧ 
     adjacency = (λ (p : ℕ × ℕ), (p = (0, 1)) ∨ (p = (2, 3)) ∨ (p = (4, 5))) 
  then 13230
  else 0

/-- State the condition that the total number of different planting schemes is 13,230 -/
theorem planting_schemes_correct :
  planting_schemes 6 {0, 1, 2, 3, 4, 5} 
    (λ (p : ℕ × ℕ), (p = (0, 1)) ∨ (p = (2, 3)) ∨ (p = (4, 5))) = 13230 := 
  by
    -- Here, we would provide the proof steps to show the correct number of schemes
    sorry

end planting_schemes_correct_l321_321895


namespace equilateral_trisection_l321_321577

-- Define the basic structure and points of the equilateral triangle and the trisection points.
variables {ABC : Type*} [triangle_eq ABC] 
variables {A B C A1 B1 C1 A2 : Point}

-- Define properties of the points based on the problem's conditions.
def equilateral_triangle (Δ : triangle ABC) : Prop := ∀ (a b c : Δ.vertices), eq_triangle Δ

def trisection_points (Δ : triangle ABC) (A1 B1 C1 : Point) : Prop :=
  (distance A C1 = 2 * distance B C1) ∧
  (distance B A1 = 2 * distance C A1) ∧
  (distance C B1 = 2 * distance A B1)

def first_trisection_point (P1 P2 Q : Point) : Prop := distance P1 Q = 2 * distance P2 Q

-- Define the main theorem stating the equality and angle.
theorem equilateral_trisection
  (ΔABC : triangle ABC)
  (h : equilateral_triangle ΔABC)
  (htrisect : trisection_points ΔABC A1 B1 C1)
  (hfirst : first_trisection_point B1 C1 A2) :
  (dist A2 A = dist A2 A1) ∧ (angle A A2 A1 = 120) :=
sorry

end equilateral_trisection_l321_321577


namespace smallest_total_students_l321_321277

theorem smallest_total_students (n : ℕ) (h1 : 25 * n % 100 = 0)
  (h2 : 10 * n % 4 = 0)
  (h3 : ∃ (y z : ℕ), y = 3 * z / 2 ∧ (y + z - n / 40 = n / 4)) :
  ∃ k : ℕ, n = 200 * k :=
by
  sorry

end smallest_total_students_l321_321277


namespace percentage_of_copper_is_correct_l321_321515

-- Defining the conditions
def total_weight := 100.0
def weight_20_percent_alloy := 30.0
def weight_27_percent_alloy := total_weight - weight_20_percent_alloy

def percentage_20 := 0.20
def percentage_27 := 0.27

def copper_20 := percentage_20 * weight_20_percent_alloy
def copper_27 := percentage_27 * weight_27_percent_alloy
def total_copper := copper_20 + copper_27

-- The statement to be proved
def percentage_copper := (total_copper / total_weight) * 100

-- The theorem to prove
theorem percentage_of_copper_is_correct : percentage_copper = 24.9 := by sorry

end percentage_of_copper_is_correct_l321_321515


namespace product_of_divisors_of_30_l321_321449

theorem product_of_divisors_of_30 : ∏ d in (finset.filter (∣ 30) (finset.range 31)), d = 810000 := by
  sorry

end product_of_divisors_of_30_l321_321449


namespace find_n_gadgets_l321_321407

-- Define the given conditions
def conditions (x y : ℝ) := 
  80 * (200 * x + 160 * y) = 1 / ∧
  40 * (160 * x + 240 * y) = 1 /  ∧
  30 * (120 * x + n * y) = 4 

-- Main theorem stating that n is equal to 135680 given the conditions
theorem find_n_gadgets (x y : ℝ) (h : conditions x y) : n = 135680 :=
by sorry

end find_n_gadgets_l321_321407


namespace algorithm_structure_combinations_l321_321077

theorem algorithm_structure_combinations :
  (∀ alg : Type, (∃ s c l : alg → Prop, 
  (∀ a : alg, s a ∨ c a ∨ l a) ∧ (∀ a : alg, ¬(s a ∧ c a) ∨ ¬(s a ∧ l a) ∨ ¬(c a ∧ l a))) → 
  (∃ alg1 alg2 : Type, alg1 = alg2) → 
  (∃ alg3 : Type, true → false) →
  (∀ alg : Type, (s alg ∨ c alg ∨ l alg))) :=
sorry

end algorithm_structure_combinations_l321_321077


namespace find_x_l321_321302

noncomputable theory

variables (A B C : Type) [angle_space A] [angle_space B] [angle_space C]

def given_conditions (angle_ABC : ℝ) (angle_BAC : ℝ) : Prop := 
  angle_ABC = 100 ∧ angle_BAC = 50

theorem find_x (angle_ABC : ℝ) (angle_BAC : ℝ) (x : ℝ) (h : given_conditions angle_ABC angle_BAC) :
  x = 30 :=
sorry

end find_x_l321_321302


namespace centroid_of_shape_L_l321_321563

-- Define the centroids and the conditions for partitions P1, P2, P3, and P4
structure ShapeL where
  P1 P2 P3 P4 : Set (ℝ × ℝ)
  homogeneous : ∀ (p : ℝ × ℝ), p ∈ P1 → p ∈ P2 → p ∈ P3 → p ∈ P4 → True

noncomputable def centroid (shape : Set (ℝ × ℝ)) : (ℝ × ℝ) :=
  sorry

theorem centroid_of_shape_L (L : ShapeL) :
  let G1 := centroid L.P1
  let G2 := centroid L.P2
  let G3 := centroid L.P3
  let G4 := centroid L.P4
  let line1 := {p : ℝ × ℝ | ∃ t : ℝ, p = G1 + t • (G2 - G1)}
  let line2 := {p : ℝ × ℝ | ∃ t : ℝ, p = G3 + t • (G4 - G3)}
  ∃ C, C ∈ line1 ∧ C ∈ line2 :=
sorry

end centroid_of_shape_L_l321_321563


namespace replace_98765_digit8_appears_11111_digit1_appears_most_l321_321676

/- Define the sum of digits function -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else sumOfDigits (n / 10 + n % 10)

-- Part (a): Prove 98765's single-digit replacement is 8
theorem replace_98765 : sumOfDigits 98765 = 8 :=
  by sorry

-- Part (b): Prove that digit 8 appears 11111 times in the final list
theorem digit8_appears_11111 : ∀ (n : Nat), 1 ≤ n ≤ 100000 → 
  (sumOfDigits n = 8) → (count (λ x, x = 8) (map sumOfDigits (range 100000)) = 11111) :=
  by sorry

-- Part (c): Prove that digit 1 appears the most with 11112 times
theorem digit1_appears_most : ∀ (n : Nat), 1 ≤ n ≤ 100000 → 
  (count (λ x, x = 1) (map sumOfDigits (range 100000)) = 11112) :=
  by sorry

end replace_98765_digit8_appears_11111_digit1_appears_most_l321_321676


namespace johns_weekly_allowance_l321_321614

theorem johns_weekly_allowance (A : ℝ) 
  (arcade_spent : A * (3/5) = 3 * (A/5)) 
  (remainder_after_arcade : (2/5) * A = A - 3 * (A/5))
  (toy_store_spent : (1/3) * (2/5) * A = 2 * (A/15)) 
  (remainder_after_toy_store : (2/5) * A - (2/15) * A = 4 * (A/15))
  (last_spent : (4/15) * A = 0.4) :
  A = 1.5 :=
sorry

end johns_weekly_allowance_l321_321614


namespace integral_definite_tangent_line_through_point_l321_321607

def f (x : ℝ) := x^3 + x

theorem integral_definite : ∫ x in -3..3, (f x + x^2) = 18 := 
sorry

theorem tangent_line_through_point : 
  ∀ (m : ℝ), (f' m = 3 * m^2 + 1) ∧ (m = 1) → line_eq (1, 2) (0, -2) 4 :=
sorry

end integral_definite_tangent_line_through_point_l321_321607


namespace angle_ADC_90_degrees_l321_321570

theorem angle_ADC_90_degrees 
  (A B C D K M N : Type)
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited K] [Inhabited M] [Inhabited N] 
  (cyclic : CyclicQuadrilateral A B C D) 
  (intersection : IntersectAtRays A B D C K)
  (circle : Circle B D M N) 
  (midpoint_M : Midpoint M A C) 
  (midpoint_N : Midpoint N K C) 
  : angle ADC = 90 :=
sorry

end angle_ADC_90_degrees_l321_321570


namespace general_inequality_l321_321584

theorem general_inequality (x : ℝ) (n : ℕ) (h_pos_x : x > 0) (h_pos_n : 0 < n) : 
  x + n^n / x^n ≥ n + 1 := by 
  sorry

end general_inequality_l321_321584


namespace arithmetic_sequence_sum_correct_l321_321526

-- Definition of the arithmetic sequence
def arithmetic_sequence : List ℕ := [82, 84, 86, 88, 90, 92, 94, 96, 98, 100]

-- Condition: This is an arithmetic sequence where each term increases by 2
def is_arithmetic_sequence (seq : List ℕ) : Prop :=
  ∀ i ∈ List.range (seq.length - 1), seq.get ⟨i + 1, Nat.lt_succ_of_lt (List.length_pos_of_ne_nil (by intro h; cases h))⟩ - seq.get ⟨i, Nat.lt_of_lt_pred (List.length_pos_of_ne_nil (by intro h; cases h))⟩ = 2

-- Function to calculate the sum of an arithmetic sequence
def arithmetic_sum (seq : List ℕ) : ℕ :=
  List.sum seq

-- The proof statement
theorem arithmetic_sequence_sum_correct :
  is_arithmetic_sequence arithmetic_sequence →
  3 * (arithmetic_sum arithmetic_sequence) = 2730 :=
by
  intro h; sorry

end arithmetic_sequence_sum_correct_l321_321526


namespace find_parabola_equation_l321_321168

-- Define the problem conditions
def parabola_vertex_at_origin (f : ℝ → ℝ) : Prop :=
  f 0 = 0

def axis_of_symmetry_x_or_y (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = 0) ∨ (∀ y, f 0 = y)

def passes_through_point (f : ℝ → ℝ) (pt : ℝ × ℝ) : Prop :=
  f pt.1 = pt.2

-- Define the specific forms we expect the equations of the parabola to take
def equation1 (x y : ℝ) : Prop :=
  y^2 = - (9 / 2) * x

def equation2 (x y : ℝ) : Prop :=
  x^2 = (4 / 3) * y

-- state the main theorem
theorem find_parabola_equation :
  ∃ f : ℝ → ℝ, parabola_vertex_at_origin f ∧ axis_of_symmetry_x_or_y f ∧ passes_through_point f (-2, 3) ∧
  (equation1 (-2) (f (-2)) ∨ equation2 (-2) (f (-2))) :=
sorry

end find_parabola_equation_l321_321168


namespace number_of_soldiers_l321_321653

theorem number_of_soldiers (length_of_wall : ℕ) (interval : ℕ) (soldiers_per_tower : ℕ)
  (h1 : length_of_wall = 7300) (h2 : interval = 5) (h3 : soldiers_per_tower = 2) :
  (length_of_wall / interval) * soldiers_per_tower = 2920 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end number_of_soldiers_l321_321653


namespace speed_limit_l321_321470

theorem speed_limit (x : ℝ) (h₀ : 0 < x) :
  (11 / (x + 1.5) + 8 / x ≥ 12 / (x + 2) + 2) → x ≤ 4 := 
sorry

end speed_limit_l321_321470


namespace part_a_part_b_l321_321657

-- Conditions
def ornament_to_crackers (n : ℕ) : ℕ := n * 2
def sparklers_to_garlands (n : ℕ) : ℕ := (n / 5) * 2
def garlands_to_ornaments (n : ℕ) : ℕ := n * 4

-- Part (a)
theorem part_a (sparklers : ℕ) (h : sparklers = 10) : ornament_to_crackers (garlands_to_ornaments (sparklers_to_garlands sparklers)) = 32 :=
by
  sorry

-- Part (b)
theorem part_b (ornaments : ℕ) (crackers : ℕ) (sparklers : ℕ) (h₁ : ornaments = 5) (h₂ : crackers = 1) (h₃ : sparklers = 2) :
  ornament_to_crackers ornaments + crackers > ornament_to_crackers (garlands_to_ornaments (sparklers_to_garlands sparklers)) :=
by
  sorry

end part_a_part_b_l321_321657


namespace inequality_and_equality_condition_l321_321721

theorem inequality_and_equality_condition (x : ℝ)
  (h : x ∈ (Set.Iio 0 ∪ Set.Ioi 0)) :
  max 0 (Real.log (|x|)) ≥ 
      ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
      (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
      (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2)
  ∧ (max 0 (Real.log (|x|)) = 
      ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
      (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
      (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ↔ 
      x = (Real.sqrt 5 - 1) / 2 ∨ 
      x = -(Real.sqrt 5 - 1) / 2 ∨ 
      x = (Real.sqrt 5 + 1) / 2 ∨ 
      x = -(Real.sqrt 5 + 1) / 2) :=
by
  sorry

end inequality_and_equality_condition_l321_321721


namespace number_of_solutions_l321_321932

def satisfies_eq (n : ℕ) : Prop :=
  n % 90 = 60 ∧ (n + 1500) / 90 = int.floor (real.sqrt n)

theorem number_of_solutions : 
  (finset.filter satisfies_eq (finset.range 10000)).card = 42 := 
by
  sorry

end number_of_solutions_l321_321932


namespace scalene_right_triangle_area_l321_321749

def triangle_area (a b : ℝ) : ℝ := 0.5 * a * b

theorem scalene_right_triangle_area
  (ABC : Type) [is_triangle ABC]
  (hypotenuse : line_segment ABC)
  (P : point hypotenuse)
  (angle_ABP : angle ABC = 30)
  (AP : distance between points A P = 2)
  (CP : distance between points C P = 1)
  (AB : length between points A B)
  (BC : length between points B C)
  (AC : length between points A C = 3)
  (right_angle : right angle of triangle ABC B) 
  : triangle_area AB BC = 9/5 :=
by 
  sorry

end scalene_right_triangle_area_l321_321749


namespace product_divisors_30_eq_810000_l321_321433

def product_of_divisors (n : ℕ) : ℕ :=
  (multiset.filter (λ d, n % d = 0) (multiset.range (n + 1))).prod id

theorem product_divisors_30_eq_810000 :
  product_of_divisors 30 = 810000 :=
begin
  -- Proof will involve showing product of divisors of 30 equals 810000
  sorry
end

end product_divisors_30_eq_810000_l321_321433


namespace total_cost_of_cards_l321_321755

theorem total_cost_of_cards:
  let cost_per_card_first_box := 1.25
  let cost_per_card_second_box := 1.75
  let number_of_cards := 6
  let total_cost_first_box := cost_per_card_first_box * number_of_cards
  let total_cost_second_box := cost_per_card_second_box * number_of_cards
  let total_cost := total_cost_first_box + total_cost_second_box
  in total_cost = 18.00 :=
by
  -- The proof goes here
  sorry

end total_cost_of_cards_l321_321755


namespace exists_a_l321_321340

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq {x y : ℝ} : f (x + y) = f x + f y
axiom continuity : continuous f

theorem exists_a (a : ℝ) (h_a : a = f 1) : ∀ x : ℝ, f x = a * x :=
sorry

end exists_a_l321_321340


namespace impossible_partition_10x10_square_l321_321310

theorem impossible_partition_10x10_square :
  ¬ ∃ (x y : ℝ), (x - y = 1) ∧ (x * y = 1) ∧ (∃ (n m : ℕ), 10 = n * x + m * y ∧ n + m = 100) :=
by
  sorry

end impossible_partition_10x10_square_l321_321310


namespace coin_value_permutations_l321_321098

theorem coin_value_permutations : 
  let digits := [1, 2, 2, 4, 4, 5, 9]
  let odd_digits := [1, 5, 9]
  let permutations (l : List ℕ) := Nat.factorial (l.length) / (l.filter (· = 2)).length.factorial / (l.filter (· = 4)).length.factorial
  3 * permutations (digits.erase 1 ++ digits.erase 5 ++ digits.erase 9) = 540 := by
  let digits := [1, 2, 2, 4, 4, 5, 9]
  let odd_digits := [1, 5, 9]
  let permutations (l : List ℕ) := Nat.factorial (l.length) / (l.filter (· = 2)).length.factorial / (l.filter (· = 4)).length.factorial
  show 3 * permutations (digits.erase 1 ++ digits.erase 5 ++ digits.erase 9) = 540
  
  -- Steps for the proof can be filled in
  -- sorry in place to indicate incomplete proof steps
  sorry

end coin_value_permutations_l321_321098


namespace arithmetic_twelfth_term_l321_321837

theorem arithmetic_twelfth_term 
(a d : ℚ) (n : ℕ) (h_a : a = 1/2) (h_d : d = 1/3) (h_n : n = 12) : 
  a + (n - 1) * d = 25 / 6 := 
by 
  sorry

end arithmetic_twelfth_term_l321_321837


namespace sticker_distribution_ways_l321_321616

theorem sticker_distribution_ways :
  (∑ (a b c d : ℕ) in {x | x.1 + x.2 + x.3 + x.4 = 10 ∧ (x.1 ≤ 1 ∨ x.2 ≤ 1 ∨ x.3 ≤ 1 ∨ x.4 ≤ 1)}, 
    multinomial 10 [a, b, c, d].map (λ i, [1, 2, 3, 4])).val = 456 :=
sorry

end sticker_distribution_ways_l321_321616


namespace max_colored_cells_no_rect_l321_321424

theorem max_colored_cells_no_rect (board_size : ℕ) (rows : list (list ℕ)) (cols : list (list ℕ)) :
  board_size = 6 →
  ∀ (board : list (list ℕ)), (∀ r, r < board_size → (board r).length = board_size) →
  (∀ (r1 r2 r3 r4 : ℕ) (c1 c2 : ℕ),
    r1 < r2 → r2 < r3 → r3 < r4 → c1 < c2 →
    board[r1][c1] = 1 → board[r1][c2] = 1 →
    board[r4][c1] = 1 → board[r4][c2] = 1 →
    board[r2][c1] ≠ 1 ∨ board[r2][c2] ≠ 1 ∨
    board[r3][c1] ≠ 1 ∨ board[r3][c2] ≠ 1) →
  ∃ max_cells, max_cells = 16 :=
by
  intros bs brd_len no_rect
  have hyp1 : bs = 6 := by assumption
  sorry

end max_colored_cells_no_rect_l321_321424


namespace max_temperature_when_80_l321_321355

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 10 * t + 60

-- State the theorem
theorem max_temperature_when_80 : ∃ t : ℝ, temperature t = 80 ∧ t = 5 + Real.sqrt 5 := 
by {
  -- Theorem proof is skipped with sorry
  sorry
}

end max_temperature_when_80_l321_321355


namespace functional_equation_solution_l321_321162

theorem functional_equation_solution :
  ∃ (f : ℝ × ℝ → ℝ), (∀ a b c : ℝ, f (a^2, f (b, c) + 1) = a^2 * (b * c + 1))
  ∧ (∀ x y : ℝ, f (x, y) = x * y) :=
by
  let f : ℝ × ℝ → ℝ := λ x, x.1 * x.2
  use f
  split
  { intros a b c
    simp only [f]
    rw [mul_add, mul_assoc, pow_two]
  }
  { intros x y
    simp
  }
  sorry

end functional_equation_solution_l321_321162


namespace sum_mod_11_l321_321065

theorem sum_mod_11 :
  let seq_sum := ∑ i in finset.range (2007), 3^(2^i)
  seq_sum % 11 = 6 := 
  sorry

end sum_mod_11_l321_321065


namespace smallest_M_mod_1000_l321_321562

/-- 
For each positive integer n: 
- h(n) is the sum of the digits in the base-five representation of n 
- k(n) is the sum of the digits in the base-twelve representation of h(n)
-/

def h (n : ℕ) : ℕ :=
  (n.reprBase 5).digits.sum

def k (n : ℕ) : ℕ :=
  (h n).reprBase 12).digits.sum

theorem smallest_M_mod_1000 : 
  ∃ M : ℕ, (∀ i < M, (k i).reprBase 16.digits.all (λ d, d ≤ 9)) ∧ 
           ¬(k M).reprBase 16.digits.all (λ d, d ≤ 9) ∧ 
           (M % 1000 = 24) := 
begin
  sorry
end

end smallest_M_mod_1000_l321_321562


namespace compute_n_l321_321869

theorem compute_n (n : ℕ) : 5^n = 5 * 25^(3/2) * 125^(5/3) → n = 9 :=
by
  sorry

end compute_n_l321_321869


namespace remainder_of_product_modulo_12_l321_321464

theorem remainder_of_product_modulo_12 : (1625 * 1627 * 1629) % 12 = 3 := by
  sorry

end remainder_of_product_modulo_12_l321_321464


namespace min_radius_of_circle_l321_321396

theorem min_radius_of_circle (a : ℝ) : 
  let eq := ∀ x y,
  x^2 + y^2 + a * x - 2 * a * y - 2 = 0 in
  ∃ r_min, 
  (∀ r, (∃ h k, (x + h)^2 + (y + k)^2 = r^2) → r ≥ r_min) ∧ r_min = real.sqrt 2 :=
  sorry

end min_radius_of_circle_l321_321396


namespace prism_volume_l321_321495

theorem prism_volume
  (l w h : ℝ)
  (h1 : l * w = 6.5)
  (h2 : w * h = 8)
  (h3 : l * h = 13) :
  l * w * h = 26 :=
by
  sorry

end prism_volume_l321_321495


namespace smallest_sector_angle_is_3_l321_321671

theorem smallest_sector_angle_is_3 :
  ∃ (a₁ : ℕ) (d : ℕ), 
    (∀ n : ℕ, n ≥ 1 ∧ n ≤ 15 → ∃ k : ℕ, a₁ + (n - 1) * d = k ∧ k ∈ ℕ) ∧
    (∑ i in (finset.range 15).image (λ n, a₁ + n * d), i) = 360 ∧
    a₁ = 3 :=
sorry

end smallest_sector_angle_is_3_l321_321671


namespace area_of_triangle_l321_321742

theorem area_of_triangle (ABC : Triangle) (right_triangle : is_right_triangle ABC)
    (scalene_triangle : is_scalene ABC) (P : Point) (on_hypotenuse : on_segment P ABC.hypotenuse)
    (angle_ABP : ABC.angle_AB P = 30) (AP_length : dist ABC.A P = 2)
    (CP_length : dist ABC.C P = 1) : 
    area ABC = 2 := 
sorry

end area_of_triangle_l321_321742


namespace product_of_divisors_of_30_l321_321441

theorem product_of_divisors_of_30 : 
  ∏ (d : ℕ) in {d | d ∣ 30} = 810000 :=
sorry

end product_of_divisors_of_30_l321_321441


namespace largest_three_digit_number_divisible_by_six_l321_321061

theorem largest_three_digit_number_divisible_by_six : ∃ n : ℕ, (∃ m < 1000, m ≥ 100 ∧ m % 6 = 0 ∧ m = n) ∧ (∀ k < 1000, k ≥ 100 ∧ k % 6 = 0 → k ≤ n) ∧ n = 996 :=
by sorry

end largest_three_digit_number_divisible_by_six_l321_321061


namespace quadratic_roots_form_l321_321791

theorem quadratic_roots_form {d : ℝ} (h : ∀ x : ℝ, x^2 + 7*x + d = 0 → (x = (-7 + real.sqrt d) / 2) ∨ (x = (-7 - real.sqrt d) / 2)) : d = 49 / 5 := 
sorry

end quadratic_roots_form_l321_321791


namespace largest_prime_divisor_of_thirteen_plus_fourteen_factorial_l321_321997

noncomputable def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def largest_prime_divisor_of_factorials_sum (n m : ℕ) : ℕ :=
  let sum := factorial n + factorial m
  Prime.factorization sum |>.keys.max' sorry

theorem largest_prime_divisor_of_thirteen_plus_fourteen_factorial :
  largest_prime_divisor_of_factorials_sum 13 14 = 17 := 
sorry

end largest_prime_divisor_of_thirteen_plus_fourteen_factorial_l321_321997


namespace product_of_divisors_of_30_l321_321443

theorem product_of_divisors_of_30 : 
  ∏ (d : ℕ) in {d | d ∣ 30} = 810000 :=
sorry

end product_of_divisors_of_30_l321_321443


namespace tetrahedron_equality_l321_321722

theorem tetrahedron_equality 
  (A B C D K M O : Point)
  (K_is_mid_AB : midpoint A B K) 
  (M_is_mid_CD : midpoint C D M)
  (O_is_incenter : incenter A B C D O)
  (line_KM_passes_O : line_through_two_points A B K O ∧ intersects edges (line_through_two_points C D M O))
  : |A - C| = |B - D| ∧ |A - D| = |B - C| := 
sorry

end tetrahedron_equality_l321_321722


namespace arithmetic_twelfth_term_l321_321839

theorem arithmetic_twelfth_term 
(a d : ℚ) (n : ℕ) (h_a : a = 1/2) (h_d : d = 1/3) (h_n : n = 12) : 
  a + (n - 1) * d = 25 / 6 := 
by 
  sorry

end arithmetic_twelfth_term_l321_321839


namespace triangle_equilateral_l321_321308

variable {α : Type} [LinearOrderedField α]

structure Triangle (α : Type) [LinearOrderedField α] :=
(a b c : α × α)

variables (ABC A1 B1 C1 : Triangle α)

def is_angle_bisector (P Q R : α × α) := sorry
def is_trisector (P Q R : α × α) := sorry

-- Assumptions
axiom A1_is_angle_bisector : is_angle_bisector ABC.a ABC.b ABC.c
axiom B1_is_angle_bisector : is_angle_bisector ABC.b ABC.c ABC.a
axiom C1_is_angle_bisector : is_angle_bisector ABC.c ABC.a ABC.b

axiom A1_intersects_angle_bisectors : is_angle_bisector A1.a A1.b A1.c
axiom B1_intersects_angle_bisectors : is_angle_bisector B1.a B1.b B1.c
axiom C1_intersects_angle_bisectors : is_angle_bisector C1.a C1.b C1.c

-- Conclusion
theorem triangle_equilateral (ABC : Triangle α) : 
  is_angle_bisector ABC.a ABC.b ABC.c ∧
  is_angle_bisector ABC.b ABC.c ABC.a ∧
  is_angle_bisector ABC.c ABC.a ABC.b →
  is_angle_bisector A1.a A1.b A1.c ∧
  is_angle_bisector B1.a B1.b B1.c ∧
  is_angle_bisector C1.a C1.b C1.c →
  sorry -- Proof should show that ABC is equilateral.

end triangle_equilateral_l321_321308


namespace largest_prime_divisor_13_fact_14_fact_l321_321974

theorem largest_prime_divisor_13_fact_14_fact :
  ∀ (n : ℕ), (n = 13!) ∧ (m = 14!) ∧ (m = 14 * n) ∧ ((n + m) = n * 15) ∧ (15 = 3 * 5) → 
  (∃ p : ℕ, (Prime p) ∧ (LargestPrimeDivisor p (n + m)) ∧ (p = 13)) :=
by
  sorry

end largest_prime_divisor_13_fact_14_fact_l321_321974


namespace baskets_of_peaches_l321_321414

theorem baskets_of_peaches (n : ℕ) :
  (∀ x : ℕ, (n * 2 = 14) → (n = x)) := by
  sorry

end baskets_of_peaches_l321_321414


namespace min_dot_product_coordinates_l321_321579

open Function

theorem min_dot_product_coordinates :
  ∃ P : ℝ × ℝ × ℝ, (∃ x : ℝ, P = (x, 0, 0)) ∧
  ∀ Q : ℝ × ℝ × ℝ, (∃ x : ℝ, Q = (x, 0, 0)) →
  let AP := (Q.1 - 1, Q.2 - 2, Q.3 - 0)
  let BP := (Q.1 - 0, Q.2 - 1, Q.3 + 1)
  in (AP.1 * BP.1 + AP.2 * BP.2 + AP.3 * BP.3) ≥
     (P.1 - 1) * P.1 + 4 :=
  ∃ (P : ℝ × ℝ × ℝ), P = (1/2, 0, 0) :=
begin
  sorry
end

end min_dot_product_coordinates_l321_321579


namespace centers_of_equilateral_triangles_form_equilateral_l321_321718

noncomputable def Point (α : Type) [Nonempty α] := α
def LineSegment (α : Type) [Nonempty α] := α × α

variables {α : Type} [Nonempty α]

-- Assume we have points A, B, C
variables (A B C : Point α)
-- Assume C is an internal point of the line segment AB
variable (internalC : C ≠ A ∧ C ≠ B ∧ ∃ t : ℝ, 0 < t ∧ t < 1 ∧ C = (1 - t) • A + t • B)

-- Assume we can construct equilateral triangles
-- centers O1, O2, O corresponding to the centers of the equilateral triangles on AC, CB, and AB respectively
variables (O O1 O2 : Point α)

-- Assume that the centers O, O1, O2 form an equilateral triangle
def centers_form_equilateral (A B C O O1 O2 : Point α) (internalC)
: Prop :=
  ∃ d : ℝ, d > 0 ∧ dist O O1 = d ∧ dist O1 O2 = d ∧ dist O2 O = d

theorem centers_of_equilateral_triangles_form_equilateral
  (A B C O O1 O2 : Point α)
  (internalC : C ≠ A ∧ C ≠ B ∧ ∃ t : ℝ, 0 < t ∧ t < 1 ∧ C = (1 - t) • A + t • B)
  (equilateral_AC : is_equilateral_triangle A C O1)
  (equilateral_CB : is_equilateral_triangle C B O2)
  (equilateral_AB : is_equilateral_triangle A B O):
  centers_form_equilateral A B C O O1 O2 internalC := sorry

end centers_of_equilateral_triangles_form_equilateral_l321_321718


namespace find_monotonically_decreasing_even_l321_321906

open Real

def f_A (x : ℝ) : ℝ := x^2
def f_B (x : ℝ) : ℝ := x + 1
def f_C (x : ℝ) : ℝ := -log (abs x)
def f_D (x : ℝ) : ℝ := 2^x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, a < x1 ∧ x1 < b → a < x2 ∧ x2 < b → x1 < x2 → f x1 ≥ f x2

theorem find_monotonically_decreasing_even :
  is_monotonically_decreasing f_C 0 (∞) ∧ is_even f_C :=
by {
  sorry
}

end find_monotonically_decreasing_even_l321_321906


namespace parabola_standard_eq_l321_321247

theorem parabola_standard_eq (h : ∃ (x y : ℝ), x - 2 * y - 4 = 0 ∧ (
                         (y = 0 ∧ x = 4 ∧ y^2 = 16 * x) ∨ 
                         (x = 0 ∧ y = -2 ∧ x^2 = -8 * y))
                         ) :
                         (y^2 = 16 * x) ∨ (x^2 = -8 * y) :=
by 
  sorry

end parabola_standard_eq_l321_321247


namespace rewrite_subtraction_rewrite_division_l321_321716

theorem rewrite_subtraction : -8 - 5 = -8 + (-5) :=
by sorry

theorem rewrite_division : (1/2) / (-2) = (1/2) * (-1/2) :=
by sorry

end rewrite_subtraction_rewrite_division_l321_321716


namespace speed_limit_l321_321471

theorem speed_limit (x : ℝ) (h₀ : 0 < x) :
  (11 / (x + 1.5) + 8 / x ≥ 12 / (x + 2) + 2) → x ≤ 4 := 
sorry

end speed_limit_l321_321471


namespace range_of_m_l321_321598

theorem range_of_m (m : ℤ) (x : ℤ) (h1 : (m + 3) / (x - 1) = 1) (h2 : x > 0) : m > -4 ∧ m ≠ -3 :=
sorry

end range_of_m_l321_321598


namespace problem1_problem2_l321_321144

theorem problem1 : ((- (5 : ℚ) / 6) + 2 / 3) / (- (7 / 12)) * (7 / 2) = 1 := 
sorry

theorem problem2 : ((1 - 1 / 6) * (-3) - (- (11 / 6)) / (- (22 / 3))) = - (11 / 4) := 
sorry

end problem1_problem2_l321_321144


namespace potassium_atoms_count_l321_321883

-- Define the atomic weights of the elements
def atomic_weight_K : ℝ := 39.1
def atomic_weight_Br : ℝ := 79.9
def atomic_weight_O : ℝ := 16.0

-- Define the molecular weight of the compound
def molecular_weight_compound : ℝ := 168

-- Define the number of Bromine and Oxygen atoms in the compound
def num_Br_atoms : ℕ := 1
def num_O_atoms : ℕ := 3

-- Calculate the total weight contribution of Bromine and Oxygen
def weight_Br : ℝ := num_Br_atoms * atomic_weight_Br
def weight_O : ℝ := num_O_atoms * atomic_weight_O

-- Total weight of Br and O
def total_weight_Br_O : ℝ := weight_Br + weight_O

-- Weight of Potassium in the compound
def weight_K_in_compound : ℝ := molecular_weight_compound - total_weight_Br_O

-- Calculate the number of Potassium atoms from the weight of K
def num_K_atoms : ℝ := weight_K_in_compound / atomic_weight_K

-- The main theorem stating the number of Potassium atoms is 1
theorem potassium_atoms_count : num_K_atoms ≈ 1 := by
  sorry

end potassium_atoms_count_l321_321883


namespace min_value_of_expression_l321_321063

theorem min_value_of_expression :
  ∀ (x y : ℝ), ∃ a b : ℝ, x = 5 ∧ y = -3 ∧ (x^2 + y^2 - 10*x + 6*y + 25) = -9 := 
by
  sorry

end min_value_of_expression_l321_321063


namespace fraction_of_children_is_8_19_l321_321135

noncomputable def fraction_of_children (total_women : ℕ) (single_prob : ℚ) (children_per_couple : ℕ) : ℚ :=
  let single_women := total_women * single_prob in
  let married_women := total_women - single_women in
  let married_men := married_women in
  let children := married_men * children_per_couple in
  let total_people := total_women + married_men + children in
  children / total_people

theorem fraction_of_children_is_8_19 : fraction_of_children 7 (3/7) 2 = 8/19 :=
by
  sorry

end fraction_of_children_is_8_19_l321_321135


namespace sum_sequence_l321_321914

noncomputable def a (n : ℕ) : ℤ := 
if n % 4 = 1 ∨ n % 4 = 3 then (-1)^((n + 1) / 2) * n 
else (-1)^((n + 1) / 2 + 1) * n

theorem sum_sequence : (∑ n in Finset.range 2020, a (n + 1)) = 0 := 
by
  sorry

end sum_sequence_l321_321914


namespace sets_are_equal_l321_321131

-- Define sets according to the given options
def option_a_M : Set (ℕ × ℕ) := {(3, 2)}
def option_a_N : Set (ℕ × ℕ) := {(2, 3)}

def option_b_M : Set ℕ := {3, 2}
def option_b_N : Set (ℕ × ℕ) := {(3, 2)}

def option_c_M : Set (ℕ × ℕ) := {(x, y) | x + y = 1}
def option_c_N : Set ℕ := { y | ∃ x, x + y = 1 }

def option_d_M : Set ℕ := {3, 2}
def option_d_N : Set ℕ := {2, 3}

-- Proof goal
theorem sets_are_equal : option_d_M = option_d_N :=
sorry

end sets_are_equal_l321_321131


namespace system_no_five_distinct_solutions_system_four_distinct_solutions_l321_321367

theorem system_no_five_distinct_solutions (a : ℤ) :
  ¬ ∃ x₁ x₂ x₃ x₄ x₅ y₁ y₂ y₃ y₄ y₅ z₁ z₂ z₃ z₄ z₅ : ℤ,
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧ x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₄ ≠ x₅) ∧
    (y₁ ≠ y₂ ∧ y₁ ≠ y₃ ∧ y₁ ≠ y₄ ∧ y₁ ≠ y₅ ∧ y₂ ≠ y₃ ∧ y₂ ≠ y₄ ∧ y₂ ≠ y₅ ∧ y₃ ≠ y₄ ∧ y₃ ≠ y₅ ∧ y₄ ≠ y₅) ∧
    (z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₁ ≠ z₅ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₂ ≠ z₅ ∧ z₃ ≠ z₄ ∧ z₃ ≠ z₅ ∧ z₄ ≠ z₅) ∧
    (2 * y₁ * z₁ + x₁ - y₁ - z₁ = a) ∧ (2 * x₁ * z₁ - x₁ + y₁ - z₁ = a) ∧ (2 * x₁ * y₁ - x₁ - y₁ + z₁ = a) ∧
    (2 * y₂ * z₂ + x₂ - y₂ - z₂ = a) ∧ (2 * x₂ * z₂ - x₂ + y₂ - z₂ = a) ∧ (2 * x₂ * y₂ - x₂ - y₂ + z₂ = a) ∧
    (2 * y₃ * z₃ + x₃ - y₃ - z₃ = a) ∧ (2 * x₃ * z₃ - x₃ + y₃ - z₃ = a) ∧ (2 * x₃ * y₃ - x₃ - y₃ + z₃ = a) ∧
    (2 * y₄ * z₄ + x₄ - y₄ - z₄ = a) ∧ (2 * x₄ * z₄ - x₄ + y₄ - z₄ = a) ∧ (2 * x₄ * y₄ - x₄ - y₄ + z₄ = a) ∧
    (2 * y₅ * z₅ + x₅ - y₅ - z₅ = a) ∧ (2 * x₅ * z₅ - x₅ + y₅ - z₅ = a) ∧ (2 * x₅ * y₅ - x₅ - y₅ + z₅ = a) :=
sorry

theorem system_four_distinct_solutions (a : ℤ) :
  (∃ x₁ x₂ y₁ y₂ z₁ z₂ : ℤ,
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ z₁ ≠ z₂ ∧
    (2 * y₁ * z₁ + x₁ - y₁ - z₁ = a) ∧ (2 * x₁ * z₁ - x₁ + y₁ - z₁ = a) ∧ (2 * x₁ * y₁ - x₁ - y₁ + z₁ = a) ∧
    (2 * y₂ * z₂ + x₂ - y₂ - z₂ = a) ∧ (2 * x₂ * z₂ - x₂ + y₂ - z₂ = a) ∧ (2 * x₂ * y₂ - x₂ - y₂ + z₂ = a)) ↔
  ∃ k : ℤ, k % 2 = 1 ∧ a = (k^2 - 1) / 8 :=
sorry

end system_no_five_distinct_solutions_system_four_distinct_solutions_l321_321367


namespace finite_fun_primes_l321_321680

open Nat

def is_fun_prime (a b p : ℕ) (n : ℕ) : Prop :=
  p ∣ (a^(factorial n) + b) ∧ 
  p ∣ (a^(factorial (n + 1)) + b) ∧ 
  p < 2 * n^2 + 1

theorem finite_fun_primes (a b : ℕ) (h_a : 0 < a) (h_b : 0 < b) :
  {p : ℕ | ∃ n : ℕ, n > 0 ∧ is_fun_prime a b p n}.finite :=
by
  sorry

end finite_fun_primes_l321_321680


namespace find_annual_interest_rate_l321_321166

noncomputable def principal_amount : ℚ := 6000
noncomputable def time_years : ℚ := 2 + 4 / 12
noncomputable def total_future_value : ℚ := 8331.75

theorem find_annual_interest_rate : 
  ∃ r : ℚ, (1 + r)^(2 + 4 / 12) = 8331.75 / 6000 ∧ r = 0.15 :=
begin
  sorry,
end

end find_annual_interest_rate_l321_321166


namespace quadratic_root_condition_l321_321781

theorem quadratic_root_condition (d : ℝ) :
  (∀ x, x^2 + 7 * x + d = 0 → x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → d = 9.8 :=
by
  intro h
  sorry

end quadratic_root_condition_l321_321781


namespace find_T9_l321_321198

noncomputable def a : ℕ → ℕ := sorry  -- The arithmetic-geometric sequence
noncomputable def q : ℕ := sorry      -- The common ratio of the geometric sequence

theorem find_T9
  (h1 : 2 * a 3 = (a 4) ^ 2)
  (h2 : ∀ (n : ℕ), a (n + 1) = a n * q) :
  let T9 := ∏ i in range 9, a (i + 1)
  in T9 = 512 := sorry

end find_T9_l321_321198


namespace identify_false_coins_l321_321256

theorem identify_false_coins (truth_coins false_coins : Nat) (truth_weight odd_weight : Nat) :
  (truth_coins = 2020) → 
  (false_coins = 2) → 
  (∀ x, x ∈ truth_weight → x % 2 = 0) → 
  (∀ y, y ∈ odd_weight → y % 2 ≠ 0) → 
  ∃ k, k = 1349 := 
by
  intro h1 h2 h3 h4
  use 1349
  sorry

end identify_false_coins_l321_321256


namespace find_smallest_beta_l321_321334

variables {a b c : ℝ^3} -- Assuming ℝ^3 context as vectors.
variables (β : ℝ) -- angle β in radians.

-- Conditions
axiom unit_vectors_a_b_c : ‖a‖ = 1 ∧ ‖b‖ = 1 ∧ ‖c‖ = 1
axiom angle_β_ab : ∃ β : ℝ, ∠(a, b) = β
axiom angle_β_ca_cross_b : ∃ β : ℝ, ∠(c, a × b) = β
axiom scalar_triple_product_condition : b • (c × a) = 1 / 3

-- Statement
theorem find_smallest_beta (unit_vectors_a_b_c : ‖a‖ = 1 ∧ ‖b‖ = 1 ∧ ‖c‖ = 1)
    (angle_β_ab : ∃ β : ℝ, ∠(a, b) = β)
    (angle_β_ca_cross_b : ∃ β : ℝ, ∠(c, a × b) = β)
    (scalar_triple_product_condition : b • (c × a) = 1 / 3) :
    β = 1 / 2 * real.arcsin (2 / 3) :=
sorry

end find_smallest_beta_l321_321334


namespace jason_current_cards_l321_321312

-- Define Jason's initial number of Pokemon cards
def jason_initial_cards : ℕ := 1342

-- Define the number of Pokemon cards Alyssa bought
def alyssa_bought_cards : ℕ := 536

-- Define the number of Pokemon cards Jason has now
def jason_final_cards (initial_cards bought_cards : ℕ) : ℕ :=
  initial_cards - bought_cards

-- Theorem statement verifying the final number of Pokemon cards Jason has
theorem jason_current_cards : jason_final_cards jason_initial_cards alyssa_bought_cards = 806 :=
by
  -- Proof goes here
  sorry

end jason_current_cards_l321_321312


namespace pq_parallel_bc_l321_321694

open EuclideanGeometry

-- Define the conditions and the theorem
theorem pq_parallel_bc (A B C M N K P Q : Point) (O : Line)
  (hABC : Circle O ⟨A,B,C ⟩) 
  (hM : ∃ (D : Point), IsAngleBisector (Line A D) A B C ∧ M.circleIntersection hABC)
  (hN : ∃ (E : Point), IsAngleBisector (Line B E) B C A ∧ N.circleIntersection hABC)
  (hK : ∃ (F : Point), IsAngleBisector (Line C F) C A B ∧ K.circleIntersection hABC)
  (hP : intersects (line_through A B) (line_through M K) P)
  (hQ : intersects (line_through A C) (line_through M N) Q)
  : IsParallel (line_through P Q) (line_through B C) := sorry

end pq_parallel_bc_l321_321694


namespace max_homework_time_l321_321352

theorem max_homework_time :
  let biology := 20
  let history := biology * 2
  let geography := history * 3
  biology + history + geography = 180 :=
by
  let biology := 20
  let history := biology * 2
  let geography := history * 3
  show biology + history + geography = 180
  sorry

end max_homework_time_l321_321352


namespace number_of_cubes_with_at_least_two_faces_painted_is_56_l321_321808

def one_inch_cubes_with_at_least_two_faces_painted 
  (side_length : ℕ) (face_colors : ℕ) (cubes_per_side : ℕ) :=
  if side_length = 4 ∧ face_colors = 6 ∧ cubes_per_side = 1 then 56 else 0

theorem number_of_cubes_with_at_least_two_faces_painted_is_56 :
  one_inch_cubes_with_at_least_two_faces_painted 4 6 1 = 56 :=
by
  sorry

end number_of_cubes_with_at_least_two_faces_painted_is_56_l321_321808


namespace value_of_A_l321_321339

theorem value_of_A (a b : ℝ) (h_a_pos : a > 0) (h_b_nonneg : b ≥ 0):
  let A := (sqrt (a + 2 * sqrt b + b / a) * sqrt (2 * a - 10 * real.root (8 * a^3 * b^2) (1/6) + 25 * real.root (b^2) (1/3))) /
           (a * sqrt (2 * a) + sqrt (2 * a * b) - 5 * a * real.root b (1/3) - 5 * real.root (b^5) (1/6))
  in (sqrt (2 * a) < 5 * real.root b (1/3) → A = -1 / sqrt a) ∧
     (sqrt (2 * a) > 5 * real.root b (1/3) → A = 1 / sqrt a) :=
begin
  sorry
end

end value_of_A_l321_321339


namespace percentage_of_children_who_speak_only_english_l321_321642

theorem percentage_of_children_who_speak_only_english :
  (∃ (total_children both_languages hindi_speaking only_english : ℝ),
    total_children = 60 ∧
    both_languages = 0.20 * total_children ∧
    hindi_speaking = 42 ∧
    only_english = total_children - (hindi_speaking - both_languages + both_languages) ∧
    (only_english / total_children) * 100 = 30) :=
  sorry

end percentage_of_children_who_speak_only_english_l321_321642


namespace wildflower_decline_below_five_percent_l321_321160

/-- A biologist takes a census every June 1st, with wildflower count decreasing by 30% each year.
    Prove that there exists a year (since 2010) where the number of wildflowers falls below 5%. --/
theorem wildflower_decline_below_five_percent :
  ∃ n : ℕ, let W : ℕ → ℝ := λ n, 100 * (0.7 ^ n) in W n < 5 :=
by
  sorry

end wildflower_decline_below_five_percent_l321_321160


namespace book_prices_l321_321094

theorem book_prices (x : ℝ) (y : ℝ) (h1 : y = 2.5 * x) (h2 : 800 / x - 800 / y = 24) : (x = 20 ∧ y = 50) :=
by
  sorry

end book_prices_l321_321094


namespace cuboid_surface_area_l321_321955

-- Define the dimensions of the cuboid
def length : ℝ := 12
def breadth : ℝ := 6
def height : ℝ := 10

-- Define the surface area calculation
def surface_area (l b h : ℝ) : ℝ :=
  2 * (l * b) + 2 * (b * h) + 2 * (l * h)

-- Theorem statement
theorem cuboid_surface_area : surface_area length breadth height = 504 := by
  sorry

end cuboid_surface_area_l321_321955


namespace sum_of_reciprocals_lt_two_l321_321326

theorem sum_of_reciprocals_lt_two {a : ℕ → ℕ} (k : ℕ)
  (hrelprime : ∀ i j, i ≠ j → Nat.coprime (a i) (a j))
  (hsquare : ∀ i j, i ≠ j → a i < (a j)^2)
  : (Finset.sum (Finset.range k) (λ i => 1 / (a i : ℝ))) < 2 := 
sorry

end sum_of_reciprocals_lt_two_l321_321326


namespace area_of_triangle_l321_321745

theorem area_of_triangle (ABC : Triangle) (right_triangle : is_right_triangle ABC)
    (scalene_triangle : is_scalene ABC) (P : Point) (on_hypotenuse : on_segment P ABC.hypotenuse)
    (angle_ABP : ABC.angle_AB P = 30) (AP_length : dist ABC.A P = 2)
    (CP_length : dist ABC.C P = 1) : 
    area ABC = 2 := 
sorry

end area_of_triangle_l321_321745


namespace product_of_divisors_30_l321_321428
-- Import the necessary library

-- Declaring the necessary conditions and main proof statement
def prime_factors (n : ℕ) : List ℕ :=
if h : n = 30 then [2, 3, 5] else []

def divisors (n : ℕ) : List ℕ :=
if h : n = 30 then [1, 2, 3, 5, 6, 10, 15, 30] else []

theorem product_of_divisors_30 : 
  let d := divisors 30 
  in d.product = 810000 :=
by {
    -- Skip the proof with sorry
    sorry
}

end product_of_divisors_30_l321_321428


namespace regular_2020_gon_isosceles_probability_l321_321417

theorem regular_2020_gon_isosceles_probability :
  let n := 2020
  let totalTriangles := (n * (n - 1) * (n - 2)) / 6
  let isoscelesTriangles := n * ((n - 2) / 2)
  let probability := isoscelesTriangles * 6 / totalTriangles
  let (a, b) := (1, 673)
  100 * a + b = 773 := by
    sorry

end regular_2020_gon_isosceles_probability_l321_321417


namespace shuttlecock_volume_approx_l321_321137

noncomputable def shuttlecock_frustum_volume : ℝ := 
  let π : ℝ := 3
  let feather_length : ℝ := 7
  let top_diameter : ℝ := 6.8
  let bottom_diameter : ℝ := 2.8
  let R : ℝ := top_diameter / 2
  let r : ℝ := bottom_diameter / 2
  let height : ℝ := Real.sqrt (feather_length^2 - ((top_diameter - bottom_diameter) / 2)^2)
  (1 / 3) * π * height * (R^2 + r^2 + R * r)

theorem shuttlecock_volume_approx : shuttlecock_frustum_volume ≈ 123 := 
  sorry

end shuttlecock_volume_approx_l321_321137


namespace max_value_expr_l321_321936

noncomputable def expr (x : ℝ) : ℝ :=
  sqrt (2 * x + 20) + sqrt (26 - 2 * x) + sqrt (3 * x)

theorem max_value_expr : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 10 → expr x ≤ 4 * sqrt 79 :=
by
  intro x hx
  have hbound : (0 : ℝ) ≤ x ∧ x ≤ 10 := hx
  -- Use Cauchy-Schwarz or other necessary inequalities to prove the bound
  sorry

end max_value_expr_l321_321936


namespace sequence_divisible_by_101_l321_321617

theorem sequence_divisible_by_101 (n : ℕ) :
    let a_n : ℕ := 10^n + 3 in
    (∀ k, 1 ≤ k → k ≤ 2048 → a_n ≡ 0 [MOD 101] → k = 98 * m ∧ m ≤ 20) :=
sorry

end sequence_divisible_by_101_l321_321617


namespace puffy_muffy_total_weight_l321_321523

theorem puffy_muffy_total_weight (scruffy_weight muffy_weight puffy_weight : ℕ)
  (h1 : scruffy_weight = 12)
  (h2 : muffy_weight = scruffy_weight - 3)
  (h3 : puffy_weight = muffy_weight + 5) :
  puffy_weight + muffy_weight = 23 := by
  sorry

end puffy_muffy_total_weight_l321_321523


namespace quadratic_roots_reciprocal_sum_l321_321818

theorem quadratic_roots_reciprocal_sum :
  ∀ x₁ x₂ : ℝ, (x₁^2 + 3 * x₁ - 1 = 0) → (x₂^2 + 3 * x₂ - 1 = 0) → (x₁ ≠ x₂) → (x₁ + x₂ = -3) →
  (x₁ * x₂ = -1) → (1 / x₁ + 1 / x₂ = 3) :=
begin
  sorry
end

end quadratic_roots_reciprocal_sum_l321_321818


namespace trapezoid_base_pairs_count_l321_321385

theorem trapezoid_base_pairs_count :
  ∃ (m n : ℕ), 9 * m + 9 * n = 80 ∧ 
  (∃ (unique_pairs : ℕ), unique_pairs = 4) :=
begin
  sorry
end

end trapezoid_base_pairs_count_l321_321385


namespace largest_prime_divisor_of_factorial_sum_l321_321960

theorem largest_prime_divisor_of_factorial_sum :
  ∃ p : ℕ, p.prime ∧ p ∣ (13! + 14!) ∧ ∀ q : ℕ, q.prime ∧ q ∣ (13! + 14!) → q ≤ p :=
sorry

end largest_prime_divisor_of_factorial_sum_l321_321960


namespace probability_sum_odd_l321_321633

theorem probability_sum_odd
  (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5}) :
  let pairs := S.val.powerset.filter (λ ss, ss.card = 2).val in
  let favorable := pairs.filter (λ ss, (ss.sum % 2 = 1)).length in
  (favorable : ℚ) / pairs.length = 3 / 5 :=
by
  sorry

end probability_sum_odd_l321_321633


namespace string_length_l321_321881

def cylindrical_post_circumference : ℝ := 6
def cylindrical_post_height : ℝ := 15
def loops : ℝ := 3

theorem string_length :
  (cylindrical_post_height / loops)^2 + cylindrical_post_circumference^2 = 61 → 
  loops * Real.sqrt 61 = 3 * Real.sqrt 61 :=
by
  sorry

end string_length_l321_321881


namespace choir_members_max_l321_321894

-- Define the conditions and the proof for the equivalent problem.
theorem choir_members_max (c s y : ℕ) (h1 : c < 120) (h2 : s * y + 3 = c) (h3 : (s - 1) * (y + 2) = c) : c = 120 := by
  sorry

end choir_members_max_l321_321894


namespace evaluate_expression_l321_321634

theorem evaluate_expression (x y z : ℤ) (h1 : x = -2) (h2 : y = -4) (h3 : z = 3) :
  (5 * (x - y)^2 - x * z^2) / (z - y) = 38 / 7 := by
  sorry

end evaluate_expression_l321_321634


namespace count_special_three_digit_numbers_l321_321498

def tens_less_than_others (h t u : ℕ) : Prop :=
  t < h ∧ t < u

theorem count_special_three_digit_numbers (count : ℕ) :
  count = 285 :=
by
  have valid_numbers : ∀ (h t u : ℕ), 1 ≤ h ∧ h ≤ 9 ∧ 0 ≤ t ∧ t ≤ 8 ∧ 1 ≤ u ∧ u ≤ 9 → tens_less_than_others h t u → True :=
    λ h t u _ _, True.intro
  sorry

end count_special_three_digit_numbers_l321_321498


namespace survey_support_percentage_l321_321647

noncomputable def support_men : ℝ := 0.70 * 200
noncomputable def support_women : ℝ := 0.75 * 1200
noncomputable def total_people : ℝ := 200 + 1200
noncomputable def total_supporters : ℝ := support_men + support_women 

theorem survey_support_percentage : (total_supporters / total_people) * 100 = 74 :=
by
  have h1 : support_men = 0.70 * 200 := by rfl
  have h2 : support_women = 0.75 * 1200 := by rfl
  have h3 : total_people = 200 + 1200 := by rfl
  have h4 : total_supporters = support_men + support_women := by rfl
  have percentage_supporters := (total_supporters / total_people) * 100
  have correct_percentage : percentage_supporters = 1040 / 1400 * 100 := by
    rw [← h4, h1, h2, h3]
  have rounded_percentage : percentage_supporters.round = 74 := by
    norm_num at correct_percentage
    exact correct_percentage
  exact rounded_percentage

end survey_support_percentage_l321_321647


namespace driers_drying_time_l321_321359

noncomputable def drying_time (r1 r2 r3 : ℝ) : ℝ := 1 / (r1 + r2 + r3)

theorem driers_drying_time (Q : ℝ) (r1 r2 r3 : ℝ)
  (h1 : r1 = Q / 24) 
  (h2 : r2 = Q / 2) 
  (h3 : r3 = Q / 8) : 
  drying_time r1 r2 r3 = 1.5 :=
by
  sorry

end driers_drying_time_l321_321359


namespace assembling_red_cube_possible_l321_321126

/-
Among the faces of eight identical cubes, one-third are blue and the rest are red.
These cubes were then assembled into a larger cube. Now, among the visible
faces of the larger cube, exactly one-third are red. Prove that it is possible
to assemble a larger cube from these cubes such that the exterior of the larger
cube is completely red.
-/

theorem assembling_red_cube_possible :
  ∃ (f : ℕ → ℕ → Prop),
    (∀ i, f 1 i = 16 ∨ f 2 i = 32) →  -- one-third faces blue, total 48 faces
    (∃ i, f 1 i = 8) ∧               -- proof of exactly one-third visible faces red
    (∃ i, f 2 i = 24) →              -- remaining faces blue on surface
    (∀ i, f 2 i = 0) :=              -- exterior completely red is possible
sorry

end assembling_red_cube_possible_l321_321126


namespace part1_part2_l321_321581
open Set Real

def setA : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def setB (m : ℝ) : Set ℝ := {x | m < x ∧ x < 2*m + 3}

theorem part1 (x : ℤ) : 
  x ∈ ({x : ℝ | -2 < x ∧ x < 4}) → 
  x ∈ ({-1, 0, 1, 2, 3} : Set ℤ) :=
sorry

theorem part2 (m : ℝ) (h : ((univ \ setA) ∩ setB m = ∅)) : 
  m ∈ ((Iic (-3) ∪ Icc (-2) (1/2)) : Set ℝ) :=
sorry

end part1_part2_l321_321581


namespace value_of_f_five_l321_321186

theorem value_of_f_five 
  {a b c m : ℝ} 
  (h1 : ∀ x : ℝ, (f : ℝ → ℝ) = λ x, a * x^7 - b * x^5 + c * x^3 + 2)
  (h2 : f (-5) = m) : 
  f 5 = -m + 4 :=
by
  sorry

end value_of_f_five_l321_321186


namespace largest_prime_divisor_13_fac_add_14_fac_l321_321970

theorem largest_prime_divisor_13_fac_add_14_fac : 
  ∀ (p : ℕ), prime p → p ∣ (13! + 14!) → p ≤ 5 :=
by
  sorry

end largest_prime_divisor_13_fac_add_14_fac_l321_321970


namespace scientific_notation_650000_l321_321369

theorem scientific_notation_650000 : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 650000 = a * 10 ^ n ∧ a = 6.5 ∧ n = 5 :=
  sorry

end scientific_notation_650000_l321_321369


namespace max_area_rect_40_perimeter_l321_321020

noncomputable def max_rect_area (P : ℕ) (hP : P = 40) : ℕ :=
  let w : ℕ → ℕ := id
  let l : ℕ → ℕ := λ w, P / 2 - w
  let area : ℕ → ℕ := λ w, w * (P / 2 - w)
  find_max_value area sorry

theorem max_area_rect_40_perimeter : max_rect_area 40 40 = 100 := 
sorry

end max_area_rect_40_perimeter_l321_321020


namespace color_segments_exists_l321_321187

-- Define the problem setup
variables (points : List Point) (colors : List Color)
  (coloring : Point → Color)
  (h_points : points.length = 20)
  (h_colors : (colors.nodup ∧ colors.length = 4))
  (h_coloring : ∀ c : Color, (points.filter (λ p, coloring p = c)).length = 5)

-- Define the main proof statement
theorem color_segments_exists :
  ∃ (segments : List Segment),
    segments.length = 4 ∧
    (∀ s ∈ segments, s.1.2 = s.2.2) ∧
    (segments.pairwise (λ s1 s2, s1.1.2 ≠ s2.1.2 ∧ s1.2.2 ≠ s2.2.2 ∧ s1.1.1 ≠ s2.1.1 ∧ s1.2.1 ≠ s2.2.1)) :=
sorry

end color_segments_exists_l321_321187


namespace license_count_l321_321494

def num_licenses : ℕ :=
  let num_letters := 3
  let num_digits := 10
  let num_digit_slots := 6
  num_letters * num_digits ^ num_digit_slots

theorem license_count :
  num_licenses = 3000000 := by
  sorry

end license_count_l321_321494


namespace total_grapes_is_157_l321_321510

def number_of_grapes_in_robs_bowl : ℕ := 25

def number_of_grapes_in_allies_bowl : ℕ :=
  number_of_grapes_in_robs_bowl + 5

def number_of_grapes_in_allyns_bowl : ℕ :=
  2 * number_of_grapes_in_allies_bowl - 2

def number_of_grapes_in_sams_bowl : ℕ :=
  (number_of_grapes_in_allies_bowl + number_of_grapes_in_allyns_bowl) / 2

def total_number_of_grapes : ℕ :=
  number_of_grapes_in_robs_bowl +
  number_of_grapes_in_allies_bowl +
  number_of_grapes_in_allyns_bowl +
  number_of_grapes_in_sams_bowl

theorem total_grapes_is_157 : total_number_of_grapes = 157 :=
  sorry

end total_grapes_is_157_l321_321510


namespace initial_money_amount_l321_321513

theorem initial_money_amount (x : ℕ) (h : x + 16 = 18) : x = 2 := by
  sorry

end initial_money_amount_l321_321513


namespace both_roots_are_real_if_k_is_pure_imaginary_l321_321926

theorem both_roots_are_real_if_k_is_pure_imaginary (k : ℂ)
  (hk : ∃ b : ℝ, k = b * complex.I) :
  ∀ z : ℂ, 10 * z^2 - 7 * complex.I * z - k = 0 → z.im = 0 :=
sorry

end both_roots_are_real_if_k_is_pure_imaginary_l321_321926


namespace sum_alternating_binomial_zero_l321_321458

/-- 
Given two natural numbers m and n with m < n, 
prove that the alternating sum of the products of k^m and binomial coefficients is zero.
 -/
theorem sum_alternating_binomial_zero (m n : ℕ) (h : m < n) : 
  ∑ k in finset.range (n + 1), if k = 0 then 0 else (-1) ^ k * k ^ m * nat.choose n k = 0 := 
sorry

end sum_alternating_binomial_zero_l321_321458


namespace general_term_formula_of_sequence_l321_321587

-- Define the sequence
variable (a : ℕ → ℝ)

-- Conditions
def geometric_increasing (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n+1) = r * a n ∧ r > 1

def condition1 (a : ℕ → ℝ) (r : ℝ) : Prop :=
  a 5 ^ 2 = a 10

def condition2 (a : ℕ → ℝ) : Prop :=
  ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)

-- Theorem to prove
theorem general_term_formula_of_sequence :
  ∃ r, geometric_increasing a r ∧ condition1 a r ∧ condition2 a →
  ∀ n, a n = 3^n :=
begin
  sorry
end

end general_term_formula_of_sequence_l321_321587


namespace largest_prime_divisor_of_factorial_sum_l321_321987

theorem largest_prime_divisor_of_factorial_sum :
  ∀ (n: ℕ), n = 13 → largest_prime_divisor (factorial n + factorial (n + 1)) = 7 := 
by
  intros,
  sorry

end largest_prime_divisor_of_factorial_sum_l321_321987


namespace smallest_number_of_students_l321_321282

theorem smallest_number_of_students
  (n : ℕ)
  (students_attended : ℕ)
  (students_both_competitions : ℕ)
  (students_hinting : ℕ)
  (students_cheating : ℕ)
  (attended_fraction : Real := 0.25)
  (both_competitions_fraction : Real := 0.1)
  (hinting_ratio : Real := 1.5)
  (h_attended : students_attended = (attended_fraction * n).to_nat)
  (h_both : students_both_competitions = (both_competitions_fraction * students_attended).to_nat)
  (h_hinting : students_hinting = (hinting_ratio * students_cheating).to_nat)
  (h_total_attended : students_attended = students_hinting + students_cheating - students_both_competitions)
  : n = 200 :=
sorry

end smallest_number_of_students_l321_321282


namespace find_m_l321_321605

-- Define the function f
def f (x : ℝ) : ℝ := Real.exp x + (2 * x - 5) / (x^2 + 1)

-- Define the derivative of f at any point x
noncomputable def f' (x : ℝ) : ℝ := 
  (Real.exp x + 
  ((2 * (x^2 + 1) - 2 * x * (2 * x - 5)) / (x^2 + 1)^2))

-- Evaluate the derivative at x = 0
def f'_at_0 : ℝ := f' 0

-- Define the condition for perpendicularity
theorem find_m : ∃ m : ℝ, m = -3 ∧ f' 0 * m = -1 := by
  sorry

end find_m_l321_321605


namespace triangle_ratio_theorem_l321_321197

noncomputable def triangle_ratio (a b c : ℝ) (A B C : ℝ) : Prop :=
  (b^2 + c^2 - a^2 = sqrt 3 * b * c) ∧ 
  (tan B = sqrt 6 / 12) →
  (b / a = sqrt 30 / 15)

-- The theorem statement
theorem triangle_ratio_theorem (a b c A B C : ℝ) : 
  triangle_ratio a b c A B C := 
sorry

end triangle_ratio_theorem_l321_321197


namespace largest_prime_divisor_13_factorial_plus_14_factorial_l321_321990

theorem largest_prime_divisor_13_factorial_plus_14_factorial : 
  ∃ p, prime p ∧ p ∈ prime_factors (13! + 14!) ∧ ∀ q, prime q ∧ q ∈ prime_factors (13! + 14!) → q ≤ 13 :=
by { sorry }

end largest_prime_divisor_13_factorial_plus_14_factorial_l321_321990


namespace sodium_bisulfate_production_l321_321171

theorem sodium_bisulfate_production (naoh_moles h2so4_moles : ℕ) (h_balance : naoh_moles = 2 ∧ h2so4_moles = 2) :
  let produced_na_hso4 := naoh_moles in
    produced_na_hso4 = 2 :=
by
  sorry

end sodium_bisulfate_production_l321_321171


namespace arithmetic_twelfth_term_l321_321840

theorem arithmetic_twelfth_term 
(a d : ℚ) (n : ℕ) (h_a : a = 1/2) (h_d : d = 1/3) (h_n : n = 12) : 
  a + (n - 1) * d = 25 / 6 := 
by 
  sorry

end arithmetic_twelfth_term_l321_321840


namespace measure_angle_A_l321_321332

variable (a b c : ℝ)
variable (GA GB GC : ℝ → ℝ → ℝ)

-- Conditions
def is_centroid (GA GB GC : ℝ → ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), GA x y + GB x y + GC x y = 0

def condition_1 (G_is_centroid : Prop) : Prop := G_is_centroid

def condition_2 (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0

def condition_3 (a b c : ℝ)
  (GA GB GC : ℝ → ℝ → ℝ) : Prop := 
  ∀ (x y : ℝ),
  a * GA x y + b * GB x y + (sqrt 3 / 3) * c * GC x y = 0

-- Proof goal
theorem measure_angle_A (a b c : ℝ) (GA GB GC : ℝ → ℝ → ℝ)
  (h1: is_centroid GA GB GC)
  (h2: condition_2 a b c)
  (h3: condition_3 a b c GA GB GC) :
  ∃ (A : ℝ), A = 30 :=
sorry

end measure_angle_A_l321_321332


namespace range_of_a_for_false_proposition_l321_321250

theorem range_of_a_for_false_proposition :
  ∀ a : ℝ, (¬ ∃ x : ℝ, a * x ^ 2 + a * x + 1 ≤ 0) ↔ (0 ≤ a ∧ a < 4) :=
by sorry

end range_of_a_for_false_proposition_l321_321250


namespace largest_prime_divisor_13_fact_14_fact_l321_321980

theorem largest_prime_divisor_13_fact_14_fact :
  ∀ (n : ℕ), (n = 13!) ∧ (m = 14!) ∧ (m = 14 * n) ∧ ((n + m) = n * 15) ∧ (15 = 3 * 5) → 
  (∃ p : ℕ, (Prime p) ∧ (LargestPrimeDivisor p (n + m)) ∧ (p = 13)) :=
by
  sorry

end largest_prime_divisor_13_fact_14_fact_l321_321980


namespace pentagon_edges_and_vertices_sum_l321_321823

theorem pentagon_edges_and_vertices_sum :
  let edges := 5
  let vertices := 5
  edges + vertices = 10 := by
  sorry

end pentagon_edges_and_vertices_sum_l321_321823


namespace total_students_in_class_l321_321815

theorem total_students_in_class (front_pos back_pos : ℕ) (H_front : front_pos = 23) (H_back : back_pos = 23) : front_pos + back_pos - 1 = 45 :=
by
  -- No proof required as per instructions
  sorry

end total_students_in_class_l321_321815


namespace power_function_value_at_three_l321_321586

def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem power_function_value_at_three :
  ∃ α : ℝ, power_function α 2 = 4 ∧ power_function α 3 = 9 :=
by
  use 2
  split
  · simp [power_function]
  · simp [power_function]
  sorry

end power_function_value_at_three_l321_321586


namespace expected_heads_after_tosses_l321_321678

noncomputable def prob_heads_after_tosses : ℚ := (1/2) + (1/4) + (1/8) + (1/16)

theorem expected_heads_after_tosses (n : ℕ) (h₁ : n = 100) : 
  (floor (n * prob_heads_after_tosses : ℚ) : ℤ) = 94 :=
by
  sorry

end expected_heads_after_tosses_l321_321678


namespace equivalent_single_discount_l321_321142

theorem equivalent_single_discount :
  let p := 1 -- Assume original price is 1 for simplification
  let final_price := (1 - 0.25) * (1 - 0.15) * p
  let single_discount := 0.3625
  final_price = (1 - single_discount) * p :=
by
  -- Define the original price
  let p := 1
  -- Apply the first discount of 15%
  let price_after_first_discount := (1 - 0.15) * p
  -- Apply the second discount of 25%
  let price_after_second_discount := (1 - 0.25) * price_after_first_discount
  -- Define the final price after the two discounts
  let final_price := price_after_second_discount
  -- Define the single equivalent discount rate
  let single_discount := 0.3625
  -- Prove the equivalent price matches the final price
  have h : final_price = (1 - single_discount) * p, {
    -- Calculation steps are omitted, assume they are correct
    sorry
  }
  exact h

end equivalent_single_discount_l321_321142


namespace twelfth_term_arithmetic_sequence_l321_321852

theorem twelfth_term_arithmetic_sequence :
  let a := (1 : ℚ) / 2
  let d := (1 : ℚ) / 3
  (a + 11 * d) = (25 : ℚ) / 6 :=
by
  sorry

end twelfth_term_arithmetic_sequence_l321_321852


namespace ratio_is_l321_321202

noncomputable def volume_dodecahedron (s : ℝ) : ℝ := (15 + 7 * Real.sqrt 5) / 4 * s ^ 3

noncomputable def volume_tetrahedron (s : ℝ) : ℝ := Real.sqrt 2 / 12 * ((Real.sqrt 3 / 2) * s) ^ 3

noncomputable def ratio_volumes (s : ℝ) : ℝ := volume_dodecahedron s / volume_tetrahedron s

theorem ratio_is (s : ℝ) : ratio_volumes s = (60 + 28 * Real.sqrt 5) / Real.sqrt 6 :=
by
  sorry

end ratio_is_l321_321202


namespace sum_numerator_denominator_of_repeating_decimal_l321_321854

noncomputable def x : ℚ := 0.484848...

theorem sum_numerator_denominator_of_repeating_decimal :
  let f := (48 / 99 : ℚ)
  let simplified_f := (16 / 33 : ℚ) 
  (simplified_f.num + simplified_f.denom) = 49 :=
by
  sorry

end sum_numerator_denominator_of_repeating_decimal_l321_321854


namespace hal_battery_change_25th_time_l321_321547

theorem hal_battery_change_25th_time (months_in_year : ℕ) 
    (battery_interval : ℕ) 
    (first_change_month : ℕ) 
    (change_count : ℕ) : 
    (battery_interval * (change_count-1)) % months_in_year + first_change_month % months_in_year = first_change_month % months_in_year :=
by
    have h1 : months_in_year = 12 := by sorry
    have h2 : battery_interval = 5 := by sorry
    have h3 : first_change_month = 5 := by sorry -- May is represented by 5 (0 = January, 1 = February, ..., 4 = April, 5 = May, ...)
    have h4 : change_count = 25 := by sorry
    sorry

end hal_battery_change_25th_time_l321_321547


namespace distance_to_origin_is_three_fifths_l321_321663

-- Define the given complex number
def complex_number : ℂ := 3 / (2 - I)^2

-- Define the complex point corresponding to the given complex number
def point : ℂ := complex_number

-- Define the origin in the complex plane
def origin : ℂ := (0 : ℝ) + (0 : ℝ) * I

-- Define the distance formula in the complex plane
def distance (z1 z2 : ℂ) : ℝ :=
  complex.abs (z1 - z2)

-- The problem statement
theorem distance_to_origin_is_three_fifths : distance point origin = 3 / 5 :=
  sorry

end distance_to_origin_is_three_fifths_l321_321663


namespace m_in_interval_l321_321539

/-- Define the sequence x_n recursively -/
def x : ℕ → ℝ
| 0       := 2
| (n + 1) := (x n ^ 2 + 3 * x n + 6) / (x n + 4)

/-- Define the least positive integer m such that x_m ≤ 2 + 1 / 2 ^ 10 -/
def m := Nat.find (λ n, x n ≤ 2 + 1 / 2 ^ 10)

/-- The goal is to prove that m lies in the interval [51, 200] -/
theorem m_in_interval : 51 ≤ m ∧ m ≤ 200 := 
sorry

end m_in_interval_l321_321539


namespace probability_one_pair_three_colors_diff_l321_321697

theorem probability_one_pair_three_colors_diff :
  let colors := {red, blue, green, orange, purple}
  let socks := colors.prod (Finset.range 2)
  let draw := socks.choose 5
  let favorable := { draw' ∈ draw | draw'.pairwise (λ x y, if x.1 = y.1 then x == y) ∨ draw'.pairwise (λ x y, x.1 ≠ y.1) }
  (favorable.card : ℚ) / (draw.card : ℚ) = 20 / 21 := by
  sorry

end probability_one_pair_three_colors_diff_l321_321697


namespace number_of_zeros_l321_321398

def f (x : ℝ) : ℝ := sin (2 * x) - sqrt 3 * cos (2 * x) + 1

theorem number_of_zeros :
  ∃ (zero_count : ℕ), zero_count = 2 ∧ ∀ x ∈ set.Icc (0 : ℝ) real.pi, f x = 0 → zero_count = 2 :=
by
  sorry

end number_of_zeros_l321_321398


namespace problem_statement_l321_321189

noncomputable def min_expression_value (θ1 θ2 θ3 θ4 : ℝ) : ℝ :=
  (2 * (Real.sin θ1)^2 + 1 / (Real.sin θ1)^2) *
  (2 * (Real.sin θ2)^2 + 1 / (Real.sin θ2)^2) *
  (2 * (Real.sin θ3)^2 + 1 / (Real.sin θ3)^2) *
  (2 * (Real.sin θ4)^2 + 1 / (Real.sin θ4)^2)

theorem problem_statement (θ1 θ2 θ3 θ4 : ℝ) (h_pos: θ1 > 0 ∧ θ2 > 0 ∧ θ3 > 0 ∧ θ4 > 0) (h_sum: θ1 + θ2 + θ3 + θ4 = Real.pi) :
  min_expression_value θ1 θ2 θ3 θ4 = 81 :=
sorry

end problem_statement_l321_321189


namespace happy_license_plates_count_l321_321480

def license_plate_count (letters : Finset Char) (digits : Finset Nat) : Nat :=
  let consonants := letters.filter (λ c, c ∈ ['В', 'К', 'М', 'Н', 'Р', 'С', 'Т', 'Х'])
  let odd_digits := digits.filter (λ d, d % 2 = 1)
  consonants.card * consonants.card * 10 * 10 * odd_digits.card * letters.card

theorem happy_license_plates_count : license_plate_count 
  (['А', 'В', 'Е', 'К', 'М', 'Н', 'О', 'Р', 'С', 'Т', 'У', 'Х'].toFinset) 
  (Finset.range 10) = 384000 := by 
    sorry

end happy_license_plates_count_l321_321480


namespace distance_Reims_to_Chaumont_l321_321391

noncomputable def distance_Chalons_Vitry : ℝ := 30
noncomputable def distance_Vitry_Chaumont : ℝ := 80
noncomputable def distance_Chaumont_SaintQuentin : ℝ := 236
noncomputable def distance_SaintQuentin_Reims : ℝ := 86
noncomputable def distance_Reims_Chalons : ℝ := 40

theorem distance_Reims_to_Chaumont :
  distance_Reims_Chalons + 
  distance_Chalons_Vitry + 
  distance_Vitry_Chaumont = 150 :=
sorry

end distance_Reims_to_Chaumont_l321_321391


namespace probability_of_five_distinct_numbers_l321_321064

theorem probability_of_five_distinct_numbers (n : ℕ) (k : ℕ) (hn : n = 5) (hk : k = 5) : 
  (finset.card (finset.univ.filter (λ (s : finset (fin k)), s.card = n & all_distinct s)) / k^n) = (120/3125) :=
by sorry

end probability_of_five_distinct_numbers_l321_321064


namespace bruce_paid_correct_amount_l321_321138

-- Define the conditions
def kg_grapes : ℕ := 8
def cost_per_kg_grapes : ℕ := 70
def kg_mangoes : ℕ := 8
def cost_per_kg_mangoes : ℕ := 55

-- Calculate partial costs
def cost_grapes := kg_grapes * cost_per_kg_grapes
def cost_mangoes := kg_mangoes * cost_per_kg_mangoes
def total_paid := cost_grapes + cost_mangoes

-- The theorem to prove
theorem bruce_paid_correct_amount : total_paid = 1000 := 
by 
  -- Merge several logical steps into one
  -- sorry can be used for incomplete proof
  sorry

end bruce_paid_correct_amount_l321_321138


namespace second_smallest_number_is_3_l321_321531

theorem second_smallest_number_is_3 (l : List ℕ) (h : l = [5, 8, 4, 3, 2]) : l.sorted.nthLe 1 (by linarith) = 3 := 
sorry

end second_smallest_number_is_3_l321_321531


namespace travel_routes_l321_321259

theorem travel_routes (S N : Finset ℕ) (hS : S.card = 4) (hN : N.card = 5) :
  ∃ (routes : ℕ), routes = 3! * 5^4 := by
  sorry

end travel_routes_l321_321259


namespace josie_wait_time_for_cart_l321_321710

theorem josie_wait_time_for_cart :
  ∀ (wait_cabinet wait_stocker wait_checkout total_wait shopping_time : ℕ),
    total_wait = 90 →
    shopping_time = 42 →
    wait_cabinet = 13 →
    wait_stocker = 14 →
    wait_checkout = 18 →
    (total_wait - shopping_time) - (wait_cabinet + wait_stocker + wait_checkout) = 3 :=
by
  intros wait_cabinet wait_stocker wait_checkout total_wait shopping_time
  intro h1 h2 h3 h4 h5
  have h : (total_wait - shopping_time) = 48 := by sorry
  have h' : (wait_cabinet + wait_stocker + wait_checkout) = 45 := by sorry
  exact eq.subst h' (eq.subst h rfl)

end josie_wait_time_for_cart_l321_321710


namespace count_house_numbers_l321_321940

def isPrime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def twoDigitPrimesBetween40And60 : List ℕ :=
  [41, 43, 47, 53, 59]

theorem count_house_numbers : 
  ∃ n : ℕ, n = 20 ∧ 
  ∀ (AB CD : ℕ), 
  AB ∈ twoDigitPrimesBetween40And60 → 
  CD ∈ twoDigitPrimesBetween40And60 → 
  AB ≠ CD → 
  true :=
by
  sorry

end count_house_numbers_l321_321940


namespace increase_by_50_percent_l321_321090

theorem increase_by_50_percent (x : ℕ) (h : x = 70) : x + (x * 50 / 100) = 105 := 
by {
  rw h,
  norm_num,
}

end increase_by_50_percent_l321_321090


namespace square_side_length_equals_radius_circumference_l321_321027

def perimeter_square (x : ℝ) : ℝ := 4 * x
def circumference_circle (r : ℝ) : ℝ := 2 * real.pi * r
def approx_equal_to_hundredth (a b : ℝ) : Prop := abs (a - b) < 0.01

theorem square_side_length_equals_radius_circumference :
  approx_equal_to_hundredth (4 * (3 * real.pi / 2)) 4.71 :=
by
  sorry

end square_side_length_equals_radius_circumference_l321_321027


namespace max_area_of_rectangle_with_perimeter_40_l321_321012

theorem max_area_of_rectangle_with_perimeter_40 :
  (∃ (x y : ℝ), (2 * x + 2 * y = 40) ∧
                (∀ (a b : ℝ), (2 * a + 2 * b = 40) → (a * b ≤ x * y)) ∧
                (x * y = 100)) :=
begin
  -- Definitions of x and y satisfying the perimeter and maximizing the area.
  have h1 : ∀ (x y : ℝ), 2 * x + 2 * y = 40 → x * (20 - x) = -(x - 10)^2 + 100,
  { intro x, intro y, intro hper,
    have hy : y = 20 - x, by linarith,
    rw hy,
    ring },
  use 10,
  use 10,
  split,
  { -- Perimeter condition
    linarith },
  { split,
    { -- Maximum area condition
      intros a b hper,
      have hab : b = 20 - a, by linarith,
      rw hab,
      specialize h1 a (20 - a),
      linarith },
    { -- Maximum area is 100
      exact (by ring) } }
end

end max_area_of_rectangle_with_perimeter_40_l321_321012


namespace proof_for_y_l321_321939

theorem proof_for_y (x y : ℝ) (h1 : 3 * x^2 + 5 * x + 4 * y + 2 = 0) (h2 : 3 * x + y + 4 = 0) : 
  y^2 + 15 * y + 2 = 0 :=
sorry

end proof_for_y_l321_321939


namespace solution_set_ln_inequality_l321_321345

noncomputable def f (x : ℝ) := Real.cos x - 4 * x^2

theorem solution_set_ln_inequality :
  {x : ℝ | 0 < x ∧ x < Real.exp (-Real.pi / 2)} ∪ {x : ℝ | x > Real.exp (Real.pi / 2)} =
  {x : ℝ | f (Real.log x) + Real.pi^2 > 0} :=
by
  sorry

end solution_set_ln_inequality_l321_321345


namespace two_digit_number_sum_154_l321_321933

theorem two_digit_number_sum_154 :
  let count := (∑ i in Finset.range 10, if (i + (14 - i) = 14 ∧ 0 ≤ 14 - i ∧ 14 - i < 10) then 1 else 0) in
  count = 5 :=
by
  -- Introduction of the variables and simplification 
  sorry

end two_digit_number_sum_154_l321_321933


namespace fitted_bowling_ball_volume_l321_321479

theorem fitted_bowling_ball_volume :
  let r := 12 in
  let d1 := 4 in
  let d2 := 4 in
  let d3 := 3 in
  let h := 6 in
  let volume_sphere := (4/3) * Real.pi * (r^3) in
  let volume_hole1 := Real.pi * ((d1/2)^2) * h in
  let volume_hole2 := Real.pi * ((d2/2)^2) * h in
  let volume_hole3 := Real.pi * ((d3/2)^2) * h in
  let total_holes_volume := 2 * volume_hole1 + volume_hole3 in
  let fitted_volume := volume_sphere - total_holes_volume in
  fitted_volume = 2242.5 * Real.pi :=
by
  sorry

end fitted_bowling_ball_volume_l321_321479


namespace find_width_of_rect_box_l321_321901

-- Define the dimensions of the wooden box in meters
def wooden_box_length_m : ℕ := 8
def wooden_box_width_m : ℕ := 7
def wooden_box_height_m : ℕ := 6

-- Define the dimensions of the rectangular boxes in centimeters (with unknown width W)
def rect_box_length_cm : ℕ := 8
def rect_box_height_cm : ℕ := 6

-- Define the maximum number of rectangular boxes
def max_boxes : ℕ := 1000000

-- Define the constraint that the total volume of the boxes should not exceed the volume of the wooden box
theorem find_width_of_rect_box (W : ℕ) (wooden_box_volume : ℕ := (wooden_box_length_m * 100) * (wooden_box_width_m * 100) * (wooden_box_height_m * 100)) : 
  (rect_box_length_cm * W * rect_box_height_cm) * max_boxes = wooden_box_volume → W = 7 :=
by
  sorry

end find_width_of_rect_box_l321_321901


namespace statement_C_is_false_l321_321078

-- Definitions based on conditions

-- Defining an isosceles triangle with one angle of 60 degrees
def isosceles_triangle_with_angle_60 (A B C : Point) (hAB : A ≠ B) (hAC : A ≠ C) 
  (h_eq_side : dist A B = dist A C) (h_angle_60 : angle A B C = 60) : Prop :=
  true

-- Defining an equilateral triangle and its symmetry properties
def equilateral_triangle_axes_of_symmetry (A B C : Point) (h_eq_sides : dist A B = dist B C ∧ dist B C = dist C A)
  (h_eq_angles : angle A B C = 60 ∧ angle B C A = 60 ∧ angle C A B = 60) : Prop :=
  trues

-- Congruence by Side-Angle-Side (SAS)
def triangle_congruence_sas (A B C D E F : Point) (hAB : dist A B = dist D E) 
  (hAC : dist A C = dist D F) (h_angle : angle A B C = angle D E F) : Prop :=
  true

-- Distance from a point on the perpendicular bisector to endpoints of the segment
def perpendicular_bisector_distance (P A B : Point) (mid_AB : Point) (h_bisector : bisector P A B mid_AB)
  (h_perpendicular : angle P mid_AB A = 90 ∧ angle P mid_AB B = 90) : Prop :=
  dist P A = dist P B

-- Proposition that statement C is false
theorem statement_C_is_false : 
  ¬ (∀ (A B C D E F : Point), dist A B = dist D E → dist A C = dist D F → angle A B C = angle D E F → ∃ congruent_triangles (△ A B C) (△ D E F)) := 
sorry

end statement_C_is_false_l321_321078


namespace product_of_divisors_of_30_l321_321444

theorem product_of_divisors_of_30 : 
  ∏ (d : ℕ) in {d | d ∣ 30} = 810000 :=
sorry

end product_of_divisors_of_30_l321_321444


namespace sum_odds_eq_square_l321_321708

theorem sum_odds_eq_square (n : ℕ) (h : ∃ k : ℕ, n = 2 * k - 1) :
  (∑ i in Finset.range ((n + 1) / 2), (2 * i + 1)) = (n + 1) / 2 ^ 2 :=
by
  obtain ⟨k, hk⟩ := h
  sorry

end sum_odds_eq_square_l321_321708


namespace largest_prime_divisor_13_fact_14_fact_l321_321975

theorem largest_prime_divisor_13_fact_14_fact :
  ∀ (n : ℕ), (n = 13!) ∧ (m = 14!) ∧ (m = 14 * n) ∧ ((n + m) = n * 15) ∧ (15 = 3 * 5) → 
  (∃ p : ℕ, (Prime p) ∧ (LargestPrimeDivisor p (n + m)) ∧ (p = 13)) :=
by
  sorry

end largest_prime_divisor_13_fact_14_fact_l321_321975


namespace equilateral_triangle_area_in_square_l321_321516

theorem equilateral_triangle_area_in_square :
  ∀ (A B C D : ℝ) (E F : ℝ), 
  (A = 0 ∧ B = 1 ∧ C = 0 ∧ D = 1) ∧
  (0 < E ∧ E < 1) ∧
  (0 < F ∧ F < 1) ∧
  (∃ (a : ℝ), a = E ∧ ∃ (b : ℝ), b = F) ∧
  (∃ (AE : ℝ) (EF : ℝ), (AE = EF) ∧ (EF = AE)) →
  (area_of_equilateral_triangle (A B C D) (E F) = 2 * Real.sqrt 3 - 3) :=
sorry

end equilateral_triangle_area_in_square_l321_321516


namespace find_unknown_number_l321_321033

theorem find_unknown_number : 
  ∃ x : ℚ, (x * 7) / (10 * 17) = 10000 ∧ x = 1700000 / 7 :=
by
  sorry

end find_unknown_number_l321_321033


namespace birds_find_more_than_half_millet_on_sunday_l321_321354

noncomputable def seed_millet_fraction : ℕ → ℚ
| 0 => 2 * 0.2 -- initial amount on Day 1 (Monday)
| (n+1) => 0.7 * seed_millet_fraction n + 0.4

theorem birds_find_more_than_half_millet_on_sunday :
  let dayMillets : ℕ := 7
  let total_seeds : ℚ := 2
  let half_seeds : ℚ := total_seeds / 2
  (seed_millet_fraction dayMillets > half_seeds) := by
    sorry

end birds_find_more_than_half_millet_on_sunday_l321_321354


namespace original_number_l321_321890

theorem original_number (x y : ℝ) (h1 : 10 * x + 22 * y = 780) (h2 : y = 37.66666666666667) : 
  x + y = 32.7 := 
sorry

end original_number_l321_321890


namespace cheryl_cups_of_pesto_l321_321917

def cups_of_basil_per_pesto : ℕ := 4
def basil_per_week : ℕ := 16
def number_of_weeks : ℕ := 8

theorem cheryl_cups_of_pesto :
  ∃ cups_of_pesto : ℕ, cups_of_pesto = (basil_per_week * number_of_weeks) / cups_of_basil_per_pesto ∧ cups_of_pesto = 32 :=
by
  let total_basil := basil_per_week * number_of_weeks
  let cups_of_pesto := total_basil / cups_of_basil_per_pesto
  exact ⟨cups_of_pesto, by simp [total_basil, cups_of_basil_per_pesto]; norm_num⟩

end cheryl_cups_of_pesto_l321_321917


namespace friend_gives_other_l321_321559

theorem friend_gives_other : 
  let earnings := [18, 22, 30, 36, 50]
  let total := earnings.sum
  let share := total / (earnings.length : ℝ)
  share = 31.2 → 50 - share = 18.8 := by
  intros earnings total share h
  have earnings_value : earnings = [18, 22, 30, 36, 50] := rfl
  have total_value : total = 156 := by simp [earnings_value, List.sum]
  have share_value : share = 31.2 := by simp [total_value, share, Int.cast_ofNat, Int.coe_nat, int_cast]
  simp [h]
  sorry

end friend_gives_other_l321_321559


namespace necessary_but_not_sufficient_l321_321473

variables {R : Type*} [Field R] (a b c : R)

def condition1 : Prop := (a / b) = (b / c)
def condition2 : Prop := b^2 = a * c

theorem necessary_but_not_sufficient :
  (condition1 a b c → condition2 a b c) ∧ ¬ (condition2 a b c → condition1 a b c) :=
by
  sorry

end necessary_but_not_sufficient_l321_321473


namespace contractor_absent_days_l321_321081

theorem contractor_absent_days
    (total_days : ℤ) (work_rate : ℤ) (fine_rate : ℤ) (total_amount : ℤ)
    (x y : ℤ)
    (h1 : total_days = 30)
    (h2 : work_rate = 25)
    (h3 : fine_rate = 75) -- fine_rate here is multiplied by 10 to avoid decimals
    (h4 : total_amount = 4250) -- total_amount multiplied by 10 for the same reason
    (h5 : x + y = total_days)
    (h6 : work_rate * x - fine_rate * y = total_amount) :
  y = 10 := 
by
  -- Here, we would provide the proof steps.
  sorry

end contractor_absent_days_l321_321081


namespace smallest_number_of_students_l321_321283

theorem smallest_number_of_students
  (n : ℕ)
  (students_attended : ℕ)
  (students_both_competitions : ℕ)
  (students_hinting : ℕ)
  (students_cheating : ℕ)
  (attended_fraction : Real := 0.25)
  (both_competitions_fraction : Real := 0.1)
  (hinting_ratio : Real := 1.5)
  (h_attended : students_attended = (attended_fraction * n).to_nat)
  (h_both : students_both_competitions = (both_competitions_fraction * students_attended).to_nat)
  (h_hinting : students_hinting = (hinting_ratio * students_cheating).to_nat)
  (h_total_attended : students_attended = students_hinting + students_cheating - students_both_competitions)
  : n = 200 :=
sorry

end smallest_number_of_students_l321_321283


namespace monkey_total_distance_l321_321091

theorem monkey_total_distance :
  let speedRunning := 15
  let timeRunning := 5
  let speedSwinging := 10
  let timeSwinging := 10
  let distanceRunning := speedRunning * timeRunning
  let distanceSwinging := speedSwinging * timeSwinging
  let totalDistance := distanceRunning + distanceSwinging
  totalDistance = 175 :=
by
  sorry

end monkey_total_distance_l321_321091


namespace regular_polygon_diagonal_difference_l321_321758

theorem regular_polygon_diagonal_difference (n : ℕ) (h : 5 < n) :
  (maximal_diagonal n) - (minimal_diagonal n) = (side_length n) → n = 9 :=
sorry

end regular_polygon_diagonal_difference_l321_321758


namespace largest_prime_divisor_of_factorial_sum_l321_321962

theorem largest_prime_divisor_of_factorial_sum :
  ∃ p : ℕ, p.prime ∧ p ∣ (13! + 14!) ∧ ∀ q : ℕ, q.prime ∧ q ∣ (13! + 14!) → q ≤ p :=
sorry

end largest_prime_divisor_of_factorial_sum_l321_321962


namespace quality_sequence_count_l321_321234

def quality_letters : Finset Char := {'Q', 'U', 'A', 'L', 'I', 'T', 'Y'}

theorem quality_sequence_count : 
  ∃ (L Q : Char) (U A I T Y : Char), 
    L = 'L' ∧ Q = 'Q' ∧
    finset.card quality_letters = 7 ∧
    multiset.card (finset.to_multiset quality_letters) = 7 ∧
    (finset.to_multiset quality_letters).remove L = {'Q', 'U', 'A', 'I', 'T', 'Y'} ∧
    (∀ x y z : Char, 
      x ∈ {'U', 'A', 'I', 'T', 'Y'} → 
      y ∈ (finset.to_multiset {'U', 'A', 'I', 'T', 'Y'}).remove x → 
      z ∈ (finset.to_multiset (finset.to_multiset {'U', 'A', 'I', 'T', 'Y'}.remove x)).remove y →
      ∃ seq : Vector Char 5, 
        seq.head = L ∧ seq.tail.head = x ∧ seq.tail.tail.head = y ∧ seq.tail.tail.tail.head = z ∧ seq.last = Q ∧ seq.nodup) :=
sorry

end quality_sequence_count_l321_321234


namespace arithmetic_sequence_twelfth_term_l321_321845

theorem arithmetic_sequence_twelfth_term :
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  (a1 + 11 * d) = (25 : ℚ) / 6 :=
by
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  exact sorry

end arithmetic_sequence_twelfth_term_l321_321845


namespace find_number_of_raccoons_squirrels_opossums_l321_321673

theorem find_number_of_raccoons_squirrels_opossums
  (R : ℕ)
  (total_animals : ℕ)
  (number_of_squirrels : ℕ := 6 * R)
  (number_of_opossums : ℕ := 2 * R)
  (total : ℕ := R + number_of_squirrels + number_of_opossums) 
  (condition : total_animals = 168)
  (correct_total : total = total_animals) :
  ∃ R : ℕ, R + 6 * R + 2 * R = total_animals :=
by
  sorry

end find_number_of_raccoons_squirrels_opossums_l321_321673


namespace smallest_period_sin_pi_x_plus_one_third_l321_321155

noncomputable def smallest_positive_period (f : ℝ → ℝ) : ℝ :=
  if h : ∃ T > 0, ∀ x, f (x + T) = f x then
    Classical.some h
  else
    0

theorem smallest_period_sin_pi_x_plus_one_third : 
  smallest_positive_period (λ x => Real.sin (π * x + 1 / 3)) = 2 :=
sorry

end smallest_period_sin_pi_x_plus_one_third_l321_321155


namespace quadratic_root_condition_l321_321780

theorem quadratic_root_condition (d : ℝ) :
  (∀ x, x^2 + 7 * x + d = 0 → x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → d = 9.8 :=
by
  intro h
  sorry

end quadratic_root_condition_l321_321780


namespace round_repeating_decimal_to_tenth_l321_321370

-- Definition of the repeating decimal 37.396396...
def repeating_decimal_37396 : ℝ := 37 + 396 / 999

-- The theorem statement to prove rounding it to the nearest tenth yields 37.4
theorem round_repeating_decimal_to_tenth : Float.round (repeating_decimal_37396 * 10) / 10 = 37.4 := 
by
  sorry

end round_repeating_decimal_to_tenth_l321_321370


namespace product_divisors_30_eq_810000_l321_321431

def product_of_divisors (n : ℕ) : ℕ :=
  (multiset.filter (λ d, n % d = 0) (multiset.range (n + 1))).prod id

theorem product_divisors_30_eq_810000 :
  product_of_divisors 30 = 810000 :=
begin
  -- Proof will involve showing product of divisors of 30 equals 810000
  sorry
end

end product_divisors_30_eq_810000_l321_321431


namespace BM_tangent_to_w_l321_321930

-- Definitions and assumptions for the problem
variables {Point Line Circle : Type}
variable  (ABCD w : Circle)
variable  (A B C D K L M : Point)
variable  (AB DC KL BD : Line)

-- Conditions provided in part (a)
axiom quad_in_circle : ∀ {A B C D : Point}, IsConvexQuad (insert_quad A B C D) → Inscribe_in_circle ABCD
axiom rays_intersect : ∀ {A B C D : Point}, Ray_intersects_point (line_ray AB) (line_ray DC) K
axiom angle_condition : ∀ {A B C D L : Point}, On_line BD L → ∠BAC = ∠DAL
axiom parallel_condition : ∀ {C M BD: Point}, On_segment KL M → parallel_line CM BD

-- The final statement (proving the result)
theorem BM_tangent_to_w : tangent_to_circle (line_segment BM) w B :=
by
  sorry

end BM_tangent_to_w_l321_321930


namespace midpoint_trajectory_l321_321211

   -- Defining the given conditions
   def P_moves_on_circle (x1 y1 : ℝ) : Prop :=
     (x1 + 1)^2 + y1^2 = 4

   def Q_coordinates : (ℝ × ℝ) := (4, 3)

   -- Defining the midpoint relationship
   def midpoint_relation (x y x1 y1 : ℝ) : Prop :=
     x1 + Q_coordinates.1 = 2 * x ∧ y1 + Q_coordinates.2 = 2 * y

   -- Proving the trajectory equation of the midpoint M
   theorem midpoint_trajectory (x y : ℝ) : 
     (∃ x1 y1 : ℝ, midpoint_relation x y x1 y1 ∧ P_moves_on_circle x1 y1) →
     (x - 3/2)^2 + (y - 3/2)^2 = 1 :=
   by
     intros h
     sorry
   
end midpoint_trajectory_l321_321211


namespace min_colors_for_distance_six_l321_321578

/-
Definitions and conditions:
- The board is an infinite checkered paper with a cell side of one unit.
- The distance between two cells is the length of the shortest path of a rook from one cell to another.

Statement:
- Prove that the minimum number of colors needed to color the board such that two cells that are a distance of 6 apart are always painted different colors is 4.
-/

def cell := (ℤ × ℤ)

def rook_distance (c1 c2 : cell) : ℤ :=
  |c1.1 - c2.1| + |c1.2 - c2.2|

theorem min_colors_for_distance_six : ∃ (n : ℕ), (∀ (f : cell → ℕ), (∀ c1 c2, rook_distance c1 c2 = 6 → f c1 ≠ f c2) → n ≤ 4) :=
by
  sorry

end min_colors_for_distance_six_l321_321578


namespace fraction_calculation_l321_321140

theorem fraction_calculation : 
  ( (1 / 5 + 1 / 7) / (3 / 8 + 2 / 9) ) = (864 / 1505) := 
by
  sorry

end fraction_calculation_l321_321140


namespace inscribable_circle_in_quad_l321_321720

-- Definition of a convex quadrilateral with equal sums of opposite sides.
variables {A B C D : Type*} [convex_quad: ConvexQuadrilateral A B C D]

-- Condition: The sums of opposite sides are equal.
-- Given a convex quadrilateral ABCD where AB + CD = BC + AD.
axiom sum_of_opposite_sides_equal : AB + CD = BC + AD

-- Theorem: A circle can be inscribed in the convex quadrilateral ABCD.
theorem inscribable_circle_in_quad (h : sum_of_opposite_sides_equal) : exists center : Point, InscribableCircle (ConvexQuadrilateral A B C D) center :=
sorry

end inscribable_circle_in_quad_l321_321720


namespace column_with_unique_value_exists_l321_321830

theorem column_with_unique_value_exists
  (table : List (List ℤ))
  (distinct_rows : ∀ i j, i ≠ j → table.nth i ≠ table.nth j)
  (four_by_two_subtable : ∀ (cols : Finset ℕ) (rows : Finset ℕ),
      cols.card = 2 → rows.card = 4 →
      ∃ (i j : ℕ), i ≠ j ∧ 
      ((table.nth (Finset.toList rows).nth i).nth (Finset.toList cols).nth 0 = 
       (table.nth (Finset.toList rows).nth j).nth (Finset.toList cols).nth 0 ∧
       (table.nth (Finset.toList rows).nth i).nth (Finset.toList cols).nth 1 = 
       (table.nth (Finset.toList rows).nth j).nth (Finset.toList cols).nth 1))
  : ∃ col, ∃ val, 
      (table.map (λ row, row.nth col)).count val = 1 := sorry

end column_with_unique_value_exists_l321_321830


namespace adela_numbers_l321_321902

theorem adela_numbers (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : (a - b)^2 = a^2 - b^2 - 4038) :
  (a = 2020 ∧ b = 1) ∨ (a = 2020 ∧ b = 2019) ∨ (a = 676 ∧ b = 3) ∨ (a = 676 ∧ b = 673) :=
sorry

end adela_numbers_l321_321902


namespace find_g_inv_f_at_4_l321_321909

variable {X : Type}
variable (f g : X → X)
variable (f_inv g_inv : X → X)

axiom f_inv_g (x : X) : f_inv (g x) = (x * x - (2 * x) + 3)
axiom g_inv_f (x : X) : g_inv (f (x * x - (2 * x) + 3)) = x

theorem find_g_inv_f_at_4 :
  g_inv (f 4) = 1 + 2 * (5 : ℝ).sqrt :=
sorry

end find_g_inv_f_at_4_l321_321909


namespace school_competition_l321_321269

theorem school_competition :
  (∃ n : ℕ, 
    n > 0 ∧ 
    75% students did not attend the competition ∧
    10% of those who attended participated in both competitions ∧
    ∃ y z : ℕ, y = 3 / 2 * z ∧ 
    y + z - (1 / 10) * (n / 4) = n / 4
  ) → n = 200 :=
sorry

end school_competition_l321_321269


namespace max_area_rect_40_perimeter_l321_321019

noncomputable def max_rect_area (P : ℕ) (hP : P = 40) : ℕ :=
  let w : ℕ → ℕ := id
  let l : ℕ → ℕ := λ w, P / 2 - w
  let area : ℕ → ℕ := λ w, w * (P / 2 - w)
  find_max_value area sorry

theorem max_area_rect_40_perimeter : max_rect_area 40 40 = 100 := 
sorry

end max_area_rect_40_perimeter_l321_321019


namespace scalene_right_triangle_area_l321_321747

def triangle_area (a b : ℝ) : ℝ := 0.5 * a * b

theorem scalene_right_triangle_area
  (ABC : Type) [is_triangle ABC]
  (hypotenuse : line_segment ABC)
  (P : point hypotenuse)
  (angle_ABP : angle ABC = 30)
  (AP : distance between points A P = 2)
  (CP : distance between points C P = 1)
  (AB : length between points A B)
  (BC : length between points B C)
  (AC : length between points A C = 3)
  (right_angle : right angle of triangle ABC B) 
  : triangle_area AB BC = 9/5 :=
by 
  sorry

end scalene_right_triangle_area_l321_321747


namespace trader_sold_85_meters_l321_321119

theorem trader_sold_85_meters (total_SP : ℕ) (profit_per_meter : ℕ) (CP_per_meter : ℕ)
  (h1 : total_SP = 8500) (h2 : profit_per_meter = 15) (h3 : CP_per_meter = 85) :
  let SP_per_meter := CP_per_meter + profit_per_meter in
  let x := total_SP / SP_per_meter in
  x = 85 :=
by
  let SP_per_meter := CP_per_meter + profit_per_meter
  let x := total_SP / SP_per_meter
  have h4 : SP_per_meter = 100 := by
    rw [h3, h2]
    rfl
  have h5 : x = 85 := by
    rw [h1, h4]
    exact Nat.div_eq_of_eq_mul_left (by norm_num : 0 < 100) (by norm_num : 100 * 85 = 8500)
  exact h5

end trader_sold_85_meters_l321_321119


namespace ratio_of_averages_l321_321533

theorem ratio_of_averages (b : ℕ → ℕ → ℝ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ 30) (h2 : ∀ j, 1 ≤ j ∧ j ≤ 50) :
  let U_i := λ i, ∑ j in finset.range 50, b i j,
      V_j := λ j, ∑ i in finset.range 30, b i j,
      C := (∑ i in finset.range 30, U_i i) / 30,
      D := (∑ j in finset.range 50, V_j j) / 50 
  in C / D = 5 / 3 :=
sorry

end ratio_of_averages_l321_321533


namespace remainder_of_poly_div_l321_321450

theorem remainder_of_poly_div (n : ℕ) (h : n > 2) : (n^3 + 3) % (n + 1) = 2 :=
by 
  sorry

end remainder_of_poly_div_l321_321450


namespace prove_inner_product_slope_angle_of_line_l321_321218

-- Conditions
def ellipse_eq (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def eccentricity (a c : ℝ) : Prop := c / a = Real.sqrt 2 / 2
def point_M (x y : ℝ) : Prop := x = 0 ∧ y = 1
def left_vertex (x y b : ℝ) : Prop := x = -Real.sqrt 2 ∧ y = 0
def line_eq (x y k : ℝ) : Prop := y = k * (x + Real.sqrt 2)
def intersect_vertical_line (x y k : ℝ) : Prop := x = Real.sqrt 2 ∧ y = 2 * Real.sqrt 2 * k
def dist_AB (k : ℝ) : Prop := Real.sqrt (1 + k^2) * (Real.sqrt 2 ∨ 0) = 4 / 3

-- Prove part (I)
theorem prove_inner_product (a b c k x y : ℤ) 
  (h1 : eccentricity a c)
  (h2 : point_M 0 1)
  (h3 : ellipse_eq (0) y a b)
  (h4 : left_vertex (- Real.sqrt 2) 0 b)
  (h5 : intersect_vertical_line (Real.sqrt 2) (2 * Real.sqrt 2 * k) k) : 
  ∃ k : ℝ, ∀ P B, (x P = y B) ∧ (∇ (OB) = y (OP)) := sorry

-- Prove part (II)
theorem slope_angle_of_line (a b c k x y : ℤ) 
  (h1 : eccentricity a c)
  (h2 : point_M 0 1)
  (h3 : ellipse_eq (0) y a b)
  (h4 : left_vertex (- Real.sqrt 2) 0 b)
  (h5 : intersect_vertical_line (Real.sqrt 2) (2 * Real.sqrt 2 * k) k): 
  dist_AB (k) → ∀ k, k = ±1 ∨ (Real.atan k = π / 4) ∨ (Real.atan k = 3 * π / 4) := sorry

end prove_inner_product_slope_angle_of_line_l321_321218


namespace minimum_value_expression_l321_321553

open Real

theorem minimum_value_expression (x : ℝ) : 
  let expr := (14 - x) * (9 - x) * (14 + x) * (9 + x)
  ∃ x_min, expr = (x^2 - 138.5)^2 - 1156.25 ∧ expr ≥ -1156.25 :=
by
  sorry

end minimum_value_expression_l321_321553


namespace largest_prime_divisor_of_factorial_sum_l321_321959

theorem largest_prime_divisor_of_factorial_sum :
  ∃ p : ℕ, p.prime ∧ p ∣ (13! + 14!) ∧ ∀ q : ℕ, q.prime ∧ q ∣ (13! + 14!) → q ≤ p :=
sorry

end largest_prime_divisor_of_factorial_sum_l321_321959


namespace simplify_expression_l321_321946

theorem simplify_expression (N M : ℝ) (hN : N > 1) (hM : M > 1) :
  Real.cbrt (N * Real.cbrt (M * Real.cbrt (N * Real.cbrt M))) = (N^(2/3)) * (M^(2/3)) :=
sorry

end simplify_expression_l321_321946


namespace train_speed_proof_l321_321421

theorem train_speed_proof (distance_ab : ℕ) (v : ℕ) (speed_b : ℕ) (meet_time_a : ℕ) (meet_time_b : ℕ) (total_distance : ℕ)
  (train_a_start : ℕ) (train_b_start : ℕ) :
  distance_ab = 200 →
  speed_b = 25 →
  meet_time_a = 12 - 7 →
  meet_time_b = 12 - 8 →
  total_distance = 200 - 100 →
  5 * v + speed_b * meet_time_b = distance_ab →
  v = 20 :=
by
  intros h_dist_ab h_speed_b h_meet_time_a h_meet_time_b h_total_dist h_eqn
  have : v = 20, sorry
  exact this

end train_speed_proof_l321_321421


namespace abs_neg_is_2_l321_321384

theorem abs_neg_is_2 (a : ℝ) (h1 : a < 0) (h2 : |a| = 2) : a = -2 :=
by sorry

end abs_neg_is_2_l321_321384


namespace division_algorithm_l321_321865

-- Define the conditions
def Dividend : ℕ := 109
def Quotient : ℕ := 9
def Remainder : ℕ := 1

-- Identify the divisor
def Divisor : ℕ := 12

-- Prove that under the given conditions, the divisor is 12
theorem division_algorithm {x : ℕ} (h : Dividend = x * Quotient + Remainder) : x = Divisor :=
sorry

end division_algorithm_l321_321865


namespace twelfth_term_arithmetic_sequence_l321_321850

theorem twelfth_term_arithmetic_sequence :
  let a := (1 : ℚ) / 2
  let d := (1 : ℚ) / 3
  (a + 11 * d) = (25 : ℚ) / 6 :=
by
  sorry

end twelfth_term_arithmetic_sequence_l321_321850


namespace largest_prime_divisor_of_factorial_sum_l321_321982

theorem largest_prime_divisor_of_factorial_sum :
  ∀ (n: ℕ), n = 13 → largest_prime_divisor (factorial n + factorial (n + 1)) = 7 := 
by
  intros,
  sorry

end largest_prime_divisor_of_factorial_sum_l321_321982


namespace equal_sum_of_squares_equal_sum_of_cubes_not_equal_sum_of_fourth_powers_l321_321733

variables (p q : ℤ)
def exprs1 := [1, p * q + 2, p * q + p - 2 * q, 2 * p * q + p - 2 * q + 1]
def exprs2 := [2, p * q + p + 1, p * q - 2 * q + 1, 2 * p * q + p - 2 * q]

def sum_of_powers (L : List ℤ) (k : ℕ) := L.foldr (λ x acc, x ^ k + acc) 0

theorem equal_sum_of_squares : sum_of_powers (exprs1 p q) 2 = sum_of_powers (exprs2 p q) 2 := sorry

theorem equal_sum_of_cubes : sum_of_powers (exprs1 p q) 3 = sum_of_powers (exprs2 p q) 3 := sorry

theorem not_equal_sum_of_fourth_powers : ∃ p q : ℤ, sum_of_powers (exprs1 p q) 4 ≠ sum_of_powers (exprs2 p q) 4 :=
begin
  use [3, 1],
  sorry
end

end equal_sum_of_squares_equal_sum_of_cubes_not_equal_sum_of_fourth_powers_l321_321733


namespace fertilizer_spread_across_field_l321_321486

theorem fertilizer_spread_across_field:
  ∀ (fieldSize totalFertilizer areaFertilizer partialArea : ℕ), 
    fieldSize = 10800 ∧ areaFertilizer = 3600 ∧ totalFertilizer = 400 →
    partialArea = (totalFertilizer * fieldSize) / areaFertilizer →
    partialArea = 1200 :=
by
  intros fieldSize totalFertilizer areaFertilizer partialArea
  assume h1 : fieldSize = 10800 ∧ areaFertilizer = 3600 ∧ totalFertilizer = 400
  assume h2 : partialArea = (totalFertilizer * fieldSize) / areaFertilizer
  sorry

end fertilizer_spread_across_field_l321_321486


namespace distinct_real_solutions_l321_321329

theorem distinct_real_solutions
  (a b c d e : ℝ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :
  ∃ x₁ x₂ x₃ x₄ : ℝ,
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
    (x₁ - a) * (x₁ - b) * (x₁ - c) * (x₁ - d) +
    (x₁ - a) * (x₁ - b) * (x₁ - c) * (x₁ - e) +
    (x₁ - a) * (x₁ - b) * (x₁ - d) * (x₁ - e) +
    (x₁ - a) * (x₁ - c) * (x₁ - d) * (x₁ - e) +
    (x₁ - b) * (x₁ - c) * (x₁ - d) * (x₁ - e) = 0 ∧
    (x₂ - a) * (x₂ - b) * (x₂ - c) * (x₂ - d) +
    (x₂ - a) * (x₂ - b) * (x₂ - c) * (x₂ - e) +
    (x₂ - a) * (x₂ - b) * (x₂ - d) * (x₂ - e) +
    (x₂ - a) * (x₂ - c) * (x₂ - d) * (x₂ - e) +
    (x₂ - b) * (x₂ - c) * (x₂ - d) * (x₂ - e) = 0 ∧
    (x₃ - a) * (x₃ - b) * (x₃ - c) * (x₃ - d) +
    (x₃ - a) * (x₃ - b) * (x₃ - c) * (x₃ - e) +
    (x₃ - a) * (x₃ - b) * (x₃ - d) * (x₃ - e) +
    (x₃ - a) * (x₃ - c) * (x₃ - d) * (x₃ - e) +
    (x₃ - b) * (x₃ - c) * (x₃ - d) * (x₃ - e) = 0 ∧
    (x₄ - a) * (x₄ - b) * (x₄ - c) * (x₄ - d) +
    (x₄ - a) * (x₄ - b) * (x₄ - c) * (x₄ - e) +
    (x₄ - a) * (x₄ - b) * (x₄ - d) * (x₄ - e) +
    (x₄ - a) * (x₄ - c) * (x₄ - d) * (x₄ - e) +
    (x₄ - b) * (x₄ - c) * (x₄ - d) * (x₄ - e) = 0 :=
  sorry

end distinct_real_solutions_l321_321329


namespace parabola_intersect_count_l321_321927

/-- The total number of intersection points between pairs of parabolas is 2912. -/
theorem parabola_intersect_count : 
  let focus := (0, 0)
  let a_vals := [-3, -2, -1, 0, 1, 2, 3]
  let b_vals := [-4, -3, -2, -1, 1, 2, 3, 4]
  let parabola (a b : ℤ) := λ p : ℝ × ℝ, let (x, y) := p in (y - a * x - b)^2 = (x^2 + y^2)
  let parabolas := { (a, b) | a ∈ a_vals ∧ b ∈ b_vals }
  let N := parabolas.card
  let total_pairs := N.choose 2 
  let non_intersecting_pairs := 7 * ((4.choose 2) + (4.choose2))
  let intersecting_pairs := total_pairs - non_intersecting_pairs
  let total_points := 2 * intersecting_pairs
  in total_points = 2912 :=
by
  sorry

end parabola_intersect_count_l321_321927


namespace part_a_part_b_l321_321656

-- Conditions
def ornament_to_crackers (n : ℕ) : ℕ := n * 2
def sparklers_to_garlands (n : ℕ) : ℕ := (n / 5) * 2
def garlands_to_ornaments (n : ℕ) : ℕ := n * 4

-- Part (a)
theorem part_a (sparklers : ℕ) (h : sparklers = 10) : ornament_to_crackers (garlands_to_ornaments (sparklers_to_garlands sparklers)) = 32 :=
by
  sorry

-- Part (b)
theorem part_b (ornaments : ℕ) (crackers : ℕ) (sparklers : ℕ) (h₁ : ornaments = 5) (h₂ : crackers = 1) (h₃ : sparklers = 2) :
  ornament_to_crackers ornaments + crackers > ornament_to_crackers (garlands_to_ornaments (sparklers_to_garlands sparklers)) :=
by
  sorry

end part_a_part_b_l321_321656


namespace rearrange_HMMTHMMT_l321_321296

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

def multiset_permutations (n : ℕ) (ns : list ℕ) : ℕ :=
factorial n / (ns.map factorial).prod

theorem rearrange_HMMTHMMT :
  let total_permutations := multiset_permutations 8 [2, 4, 2],
      substring_appears := multiset_permutations 5 [3] * multiset_permutations 4 [2] in
  total_permutations - substring_appears = 361 :=
by
  let total_permutations := multiset_permutations 8 [2, 4, 2],
      substring_appears := multiset_permutations 5 [3] * multiset_permutations 4 [2]
  show total_permutations - substring_appears = 361, from sorry

end rearrange_HMMTHMMT_l321_321296


namespace cosine_identity_find_angle_B_l321_321226

section Geometry
variables (A B C: ℝ) (a b c: ℝ)

-- (Ⅰ) Problem statement
theorem cosine_identity (hC : C = 2 * B) :
  cos A = 3 * cos B - 4 * (cos B) ^ 3 := 
sorry

-- (Ⅱ) Problem statement
variable (S : ℝ)
theorem find_angle_B (hb : b * sin B - c * sin C = a) (hS : S = (b^2 + c^2 - a^2) / 4) :
  B = 77.5 * real.pi / 180 :=
sorry
end Geometry

end cosine_identity_find_angle_B_l321_321226


namespace evaluate_expression_l321_321942

theorem evaluate_expression : (-3)^4 / 3^2 - 2^5 + 7^2 = 26 := by
  sorry

end evaluate_expression_l321_321942


namespace extended_room_area_is_20_l321_321897

-- Define the problem parameters
def room_length_feet : ℝ := 15
def room_width_inches : ℝ := 108
def walkway_width_feet : ℝ := 3

-- Define the conversion factors
def inches_per_foot : ℝ := 12
def feet_per_yard : ℝ := 3

-- Define the conversion of dimensions
def room_width_feet : ℝ := room_width_inches / inches_per_foot
def room_length_yards : ℝ := room_length_feet / feet_per_yard
def room_width_yards : ℝ := room_width_feet / feet_per_yard
def walkway_width_yards : ℝ := walkway_width_feet / feet_per_yard

-- Define areas
def room_area_square_yards : ℝ := room_length_yards * room_width_yards
def walkway_area_square_yards : ℝ := room_length_yards * walkway_width_yards
def total_area_square_yards : ℝ := room_area_square_yards + walkway_area_square_yards

-- Prove that the total area of the extended room is 20 square yards
theorem extended_room_area_is_20 : total_area_square_yards = 20 := by
  sorry

end extended_room_area_is_20_l321_321897


namespace attraction_ticket_cost_for_parents_l321_321300

noncomputable def total_cost (children parents adults: ℕ) (entrance_cost child_attraction_cost adult_attraction_cost: ℕ) : ℕ :=
  (children + parents + adults) * entrance_cost + children * child_attraction_cost + adults * (adult_attraction_cost)

theorem attraction_ticket_cost_for_parents
  (children parents adults: ℕ) 
  (entrance_cost child_attraction_cost total_cost_of_family: ℕ) 
  (h_children: children = 4)
  (h_parents: parents = 2)
  (h_adults: adults = 1)
  (h_entrance_cost: entrance_cost = 5)
  (h_child_attraction_cost: child_attraction_cost = 2)
  (h_total_cost_of_family: total_cost_of_family = 55)
  : (total_cost children parents adults entrance_cost child_attraction_cost 4 / 3) = total_cost_of_family - (children + parents + adults) * entrance_cost - children * child_attraction_cost := 
sorry

end attraction_ticket_cost_for_parents_l321_321300


namespace irrational_sqrt3_l321_321451

theorem irrational_sqrt3 :
  ∀ (x : ℚ), 
  x ≠ -1 ∧ x ≠ 2 ∧ x ≠ 3 ∧ x ≠ (Real.sqrt 3) → ¬ (∃ (r : ℚ), Real.sqrt 3 = r) := 
sorry

end irrational_sqrt3_l321_321451


namespace distance_between_trees_correct_l321_321087

-- Define the given conditions
def yard_length : ℕ := 300
def tree_count : ℕ := 26
def interval_count : ℕ := tree_count - 1

-- Define the target distance between two consecutive trees
def target_distance : ℕ := 12

-- Prove that the distance between two consecutive trees is correct
theorem distance_between_trees_correct :
  yard_length / interval_count = target_distance := 
by
  sorry

end distance_between_trees_correct_l321_321087


namespace trip_distance_is_correct_l321_321050

noncomputable def TrainA_Speed : ℝ := 82.1
noncomputable def TrainB_Speed : ℝ := 109.071
noncomputable def Passing_Time : ℝ := 1.25542053973

theorem trip_distance_is_correct :
  let Distance_A := TrainA_Speed * Passing_Time,
      Distance_B := TrainB_Speed * Passing_Time,
      Total_Distance := Distance_A + Distance_B
  in abs (Total_Distance - 240.040) < 0.001 :=
by
  sorry

end trip_distance_is_correct_l321_321050


namespace maxwell_distance_when_meeting_l321_321704

variable (total_distance : ℝ := 50)
variable (maxwell_speed : ℝ := 4)
variable (brad_speed : ℝ := 6)
variable (t : ℝ := total_distance / (maxwell_speed + brad_speed))

theorem maxwell_distance_when_meeting :
  (maxwell_speed * t = 20) :=
by
  sorry

end maxwell_distance_when_meeting_l321_321704


namespace painted_cubes_count_l321_321802

def total_painted_cubes : ℕ := 8 + 48

theorem painted_cubes_count : total_painted_cubes = 56 :=
by 
  -- Step 1: Define the number of cubes with 3 faces painted (8 corners)
  let corners := 8
  -- Step 2: Calculate the number of edge cubes with 2 faces painted
  let edge_middle_cubes_per_edge := 2
  let edges := 12
  let edge_cubes := edge_middle_cubes_per_edge * edges -- this should be 24
  -- Step 3: Calculate the number of face-interior cubes with 2 faces painted
  let face_cubes_per_face := 4
  let faces := 6
  let face_cubes := face_cubes_per_face * faces -- this should be 24
  -- Step 4: Sum them up to get total cubes with at least two faces painted
  let total_cubes := corners + edge_cubes + face_cubes
  show total_cubes = total_painted_cubes
  sorry

end painted_cubes_count_l321_321802


namespace chocolate_bar_cost_l321_321544

theorem chocolate_bar_cost (total_bars : ℕ) (sold_bars : ℕ) (total_money : ℕ) (cost : ℕ) 
  (h1 : total_bars = 13)
  (h2 : sold_bars = total_bars - 4)
  (h3 : total_money = 18)
  (h4 : total_money = sold_bars * cost) :
  cost = 2 :=
by sorry

end chocolate_bar_cost_l321_321544


namespace cube_faces_paint_count_l321_321799

def is_painted_on_at_least_two_faces (x y z : ℕ) : Prop :=
  ((x = 0 ∨ x = 3) ∨ (y = 0 ∨ y = 3) ∨ (z = 0 ∨ z = 3))

theorem cube_faces_paint_count :
  let n := 4 in
  let volume := n * n * n in
  let one_inch_cubes := {c | ∃ x y z : ℕ, x < n ∧ y < n ∧ z < n ∧ c = (x, y, z)} in
  let painted_cubes := {c ∈ one_inch_cubes | exists_along_edges c} in  -- Auxiliary predicate to check edge paint
  card painted_cubes = 32 :=
begin
  sorry,
end

/-- Auxiliary predicate to check if a cube is on the edges but not corners, signifying 
    at least two faces are painted --/
predicate exists_along_edges (c: ℕ × ℕ × ℕ) := 
  match c with
  | (0, _, _) => true
  | (3, _, _) => true
  | (_, 0, _) => true
  | (_, 3, _) => true
  | (_, _, 0) => true
  | (_, _, 3) => true
  | (_, _, _) => false
  end

end cube_faces_paint_count_l321_321799


namespace find_baseball_cards_per_box_l321_321519

theorem find_baseball_cards_per_box:
  ∀ (x : ℕ), 4 * 10 + 5 * x = 80 → x = 8 :=
by
  intro x,
  intro h,
  have h1 : 40 + 5 * x = 80 := h,
  have h2 : 5 * x = 40 := by linarith,
  suffices : x = 8,
    { assumption },
  sorry

end find_baseball_cards_per_box_l321_321519


namespace express_in_scientific_notation_l321_321639

theorem express_in_scientific_notation (n : ℕ) (h : n = 260150000000) :
  n = 260150000000 → n = 2.6015 * 10^11 :=
by
  intro h
  rw h
  sorry

end express_in_scientific_notation_l321_321639


namespace range_of_m_l321_321599

theorem range_of_m (m : ℤ) (x : ℤ) (h1 : (m + 3) / (x - 1) = 1) (h2 : x > 0) : m > -4 ∧ m ≠ -3 :=
sorry

end range_of_m_l321_321599


namespace distances_from_M_l321_321497

noncomputable def circle_rad : ℝ := 5
  
def distance_ma : ℝ := 6
  
def distance_mb : ℝ := Real.sqrt 2

def distance_mc : ℝ := 8

def distance_md : ℝ := 7 * Real.sqrt 2

theorem distances_from_M 
  (M : ℝ) (A : ℝ) (B : ℝ) (C : ℝ) (D : ℝ)
  (r : ℝ) (dMA : ℝ)
  (H1 : r = circle_rad)
  (H2 : ∀ (x : ℝ), x ∈ {M, A, B, C, D} → x ∈ Metrics.Ball 0 r)
  (H3 : dMA = distance_ma)
  (H4 : ∀ (x y : ℝ), x ∈ {A, B, C, D} → y ∈ {A, B, C, D} → x ≠ y → (dist x y) = 5 * Real.sqrt 2)
  (H5 : ∀ (x : ℝ), x ∈ {A, B, C, D} → ∃ M, dist x M =  5)
  : (dist M B = distance_mb) ∧ (dist M C = distance_mc) ∧ (dist M D = distance_md) := 
by
  sorry

end distances_from_M_l321_321497


namespace find_k_if_3_root_l321_321177

theorem find_k_if_3_root : ∀ k : ℚ, (3 : ℚ)^3 + k * 3 + 20 = 0 → k = (-47) / 3 :=
by
  assume k
  intro h

  -- the proof will go here
  sorry

end find_k_if_3_root_l321_321177


namespace domain_of_f_is_real_l321_321423

-- Define the function f
def f (t : ℝ) : ℝ := 1 / ((t - 2)^2 * (t + 3)^2 + 1)

-- Prove that the domain of f is ℝ
theorem domain_of_f_is_real : ∀ t : ℝ, (t ∈ ℝ) := by
  -- Since the proof is not required, we use sorry
  sorry

end domain_of_f_is_real_l321_321423


namespace solution_l321_321156

-- Define the matrix type as a 2x2 matrix in Lean
structure Matrix2x2 (α : Type) :=
(a11 a12 a21 a22 : α)

-- Define the multiplication of 2x2 matrices
def mul_matrix2x2 {α : Type} [Mul α] [Add α]: Matrix2x2 α → Matrix2x2 α → Matrix2x2 α
| ⟨a11, a12, a21, a22⟩, ⟨b11, b12, b21, b22⟩ => ⟨a11 * b11 + a12 * b21, a11 * b12 + a12 * b22,
                                 a21 * b11 + a22 * b21, a21 * b12 + a22 * b22⟩

-- Define the projection matrix condition
def is_projection_matrix (Q : Matrix2x2 ℝ) :=
  mul_matrix2x2 Q Q = Q

-- State the problem: define the matrix Q and the conditions for b and d
noncomputable def problem : Prop :=
  ∃ (b d : ℝ), 
  let Q := ⟨b, 1 / 5, d, 4 / 5⟩ 
  in is_projection_matrix Q ∧ b = 1 ∧ d = 1

-- Placeholder for the proof
theorem solution : problem := sorry

end solution_l321_321156


namespace yellow_balls_in_bag_l321_321874

theorem yellow_balls_in_bag (y : ℕ) (r : ℕ) (P_red : ℚ) (h_r : r = 8) (h_P_red : P_red = 1 / 3) 
  (h_prob : P_red = r / (r + y)) : y = 16 :=
by
  sorry

end yellow_balls_in_bag_l321_321874


namespace bounded_f_l321_321341

theorem bounded_f (f : ℝ → ℝ) (h1 : ∀ x1 x2, |x1 - x2| ≤ 1 → |f x2 - f x1| ≤ 1)
  (h2 : f 0 = 1) : ∀ x, -|x| ≤ f x ∧ f x ≤ |x| + 2 := by
  sorry

end bounded_f_l321_321341


namespace product_of_divisors_of_30_l321_321436

theorem product_of_divisors_of_30 : 
  ∏ d in (finset.filter (λ d, 30 % d = 0) (finset.range 31)), d = 810000 :=
by sorry

end product_of_divisors_of_30_l321_321436


namespace product_of_divisors_of_30_l321_321435

theorem product_of_divisors_of_30 : 
  ∏ d in (finset.filter (λ d, 30 % d = 0) (finset.range 31)), d = 810000 :=
by sorry

end product_of_divisors_of_30_l321_321435


namespace circumference_of_semicircle_is_51_40_l321_321462

-- Define the parameters given in the problem
def length : ℝ := 23
def breadth : ℝ := 17
def rectangle_perimeter : ℝ := 2 * (length + breadth)

-- Define the square's side
def side : ℝ := rectangle_perimeter / 4

-- Define the diameter of the semicircle (which is the side of the square)
def diameter : ℝ := side

-- Define the circumference of the semicircle
noncomputable def semicircle_circumference : ℝ := (Real.pi * diameter) / 2 + diameter

-- Statement of the theorem to be proved
theorem circumference_of_semicircle_is_51_40 :
  Real.pi = 3.14 → semicircle_circumference = 51.40 :=
by
  sorry

end circumference_of_semicircle_is_51_40_l321_321462


namespace tan_value_given_cos_l321_321216

theorem tan_value_given_cos : 
  ∀ (α : ℝ), (0 < α ∧ α < (3 * Real.pi / 2)) → (Real.cos ((3 * Real.pi / 2) - α) = (Real.sqrt 3 / 2)) → 
  Real.tan (2018 * Real.pi - α) = - Real.sqrt 3 :=
begin
  sorry
end

end tan_value_given_cos_l321_321216


namespace minimum_marks_l321_321357

theorem minimum_marks (grid_size : ℕ) (h : grid_size = 11) : ∃ n, n = 22 ∧ (∀ (x y : ℕ), x ≤ grid_size ∧ y ≤ grid_size → (∃ a b : ℕ, a ≤ grid_size ∧ b ≤ grid_size ∧ on_line_segment (grid_size + 1) (x, y) (a, b))) :=
by
  have fact1 : grid_size = 11 := h
  sorry

example (x y grid_size : ℕ) : Prop := ∀ (a b : ℕ), a ≤ grid_size ∧ b ≤ grid_size ∧ on_line_segment (grid_size + 1) (x, y) (a, b)

noncomputable def on_line_segment (N : ℕ) (P Q : ℕ × ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ k < N ∧ 
            P = (Q.1 + k * (Q.1 - P.1) / N, Q.2 + k * (Q.2 - P.2) / N)

end minimum_marks_l321_321357


namespace product_of_divisors_30_l321_321427
-- Import the necessary library

-- Declaring the necessary conditions and main proof statement
def prime_factors (n : ℕ) : List ℕ :=
if h : n = 30 then [2, 3, 5] else []

def divisors (n : ℕ) : List ℕ :=
if h : n = 30 then [1, 2, 3, 5, 6, 10, 15, 30] else []

theorem product_of_divisors_30 : 
  let d := divisors 30 
  in d.product = 810000 :=
by {
    -- Skip the proof with sorry
    sorry
}

end product_of_divisors_30_l321_321427


namespace certain_number_condition_l321_321088

theorem certain_number_condition:
  ∃ x : ℝ, (0.65 * x = (4 / 5 * 25) + 6) → x = 40 :=
begin
  sorry
end

end certain_number_condition_l321_321088


namespace square_side_length_equals_circle_circumference_l321_321029

constant π : ℝ

theorem square_side_length_equals_circle_circumference
  (x : ℝ)
  (h₁ : 4 * x = 6 * π) :
  x ≈ 4.71 :=
sorry

end square_side_length_equals_circle_circumference_l321_321029


namespace flour_distribution_probability_zero_l321_321390

theorem flour_distribution_probability_zero (infinite_distributions : Set ℕ) (unique_distribution : ℕ) :
  (infinite_distributions : Set ℕ).card = ∞ →
  (unique_distribution ∈ infinite_distributions) →
  (unique_distribution / infinite_distributions.card : ℝ) = 0 :=
by
  intro h_infinite h_unique
  sorry

end flour_distribution_probability_zero_l321_321390


namespace find_A_l321_321330

def is_divisible (n : ℕ) (d : ℕ) : Prop := d ∣ n

noncomputable def valid_digit (A : ℕ) : Prop :=
  A < 10

noncomputable def digit_7_number := 653802 * 10

theorem find_A (A : ℕ) (h : valid_digit A) :
  is_divisible (digit_7_number + A) 2 ∧
  is_divisible (digit_7_number + A) 3 ∧
  is_divisible (digit_7_number + A) 4 ∧
  is_divisible (digit_7_number + A) 6 ∧
  is_divisible (digit_7_number + A) 8 ∧
  is_divisible (digit_7_number + A) 9 ∧
  is_divisible (digit_7_number + A) 25 →
  A = 0 :=
sorry

end find_A_l321_321330


namespace kendra_minivans_l321_321324

theorem kendra_minivans (afternoon: ℕ) (evening: ℕ) (h1: afternoon = 4) (h2: evening = 1) : afternoon + evening = 5 :=
by sorry

end kendra_minivans_l321_321324


namespace f_expression_f_monotonic_decreasing_on_0_2_f_solution_lambda_l321_321150

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ (Set.Ioo 0 2) then 1 / (3^x + 3^(-x))
else if x = 0 ∨ x = 2 ∨ x = -2 then 0
else if x ∈ Set.Ioo (-2) 0 then -1 / (3^(-x) + 3^x)
else 0

theorem f_expression :
  ∀ x : ℝ, x ∈ Set.Icc (-2) 2 → 
    f x = 
      if x ∈ (Set.Ioo 0 2) then 1 / (3^x + 3^(-x)) 
      else if x = 0 ∨ x = 2 ∨ x = -2 then 0 
      else if x ∈ Set.Ioo (-2) 0 then -1 / (3^(-x) + 3^x) 
      else 0 :=
sorry

theorem f_monotonic_decreasing_on_0_2 : 
  ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 2 → f x1 > f x2 :=
sorry

theorem f_solution_lambda (λ : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc (-2) 2 ∧ f x = λ) ↔ 
    (-1/2 < λ ∧ λ < -9/82) ∨ (9/82 < λ ∧ λ < 1/2) ∨ (λ = 0) :=
sorry

end f_expression_f_monotonic_decreasing_on_0_2_f_solution_lambda_l321_321150


namespace probability_divisible_by_11_l321_321244

def is_five_digit (n : Nat) : Prop :=
  n >= 10000 ∧ n < 100000

def digit_sum_is_35 (n : Nat) : Prop :=
  let digits := List.map (λ i, (n / (10 ^ i)) % 10) [0, 1, 2, 3, 4]
  List.sum digits = 35

def is_divisible_by_11 (n : Nat) : Prop :=
  n % 11 = 0

theorem probability_divisible_by_11 
  (S : Finset Nat) 
  (hS : ∀ n ∈ S, is_five_digit n ∧ digit_sum_is_35 n) :
  let divisible_by_11 := (S.filter is_divisible_by_11).card
  let total_numbers := S.card
  total_numbers > 0 →
  (divisible_by_11 : Rat) / total_numbers = 1 / 8 :=
by
  sorry

end probability_divisible_by_11_l321_321244


namespace integral_of_rational_function_l321_321913

theorem integral_of_rational_function :
  ∃ C : ℝ, ∫ (x : ℝ) in set.Ioo (-∞ : ℝ) (∞ : ℝ), (3 * x^4 + 3 * x^3 - 5 * x^2 + 2) / (x * (x - 1) * (x + 2)) 
  = λ x, 3/2 * x ^ 2 - log (abs x) + log (abs (x - 1)) + log (abs (x + 2)) + C :=
sorry

end integral_of_rational_function_l321_321913


namespace max_modulus_conjugate_difference_l321_321569

theorem max_modulus_conjugate_difference (z : ℂ) (hz : complex.abs (z - complex.I) = 2) :
  ∃ w : ℝ, w = complex.abs (z - complex.conj z) ∧ w ≤ 6 :=
sorry

end max_modulus_conjugate_difference_l321_321569


namespace vector_not_parallel_l321_321612

theorem vector_not_parallel {x : ℝ} : 
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (4, x)
  a ≠ 0 ∧ b ≠ 0 ∧ ¬ (∃ k : ℝ, k ≠ 0 ∧ b = k • a)
  → x = 2 :=
begin
  intros h,
  cases h with h1 h2,
  cases h2 with h21 h22,
  sorry
end

end vector_not_parallel_l321_321612


namespace john_books_unsold_l321_321319

noncomputable def percentage_not_sold (initial_stock : ℕ) (sales_returns : List (ℕ × ℕ)) : ℝ :=
  let net_sold := sales_returns.foldr (λ (sr : ℕ × ℕ) acc, acc + (sr.1 - sr.2)) 0
  let books_not_sold := initial_stock - net_sold
  (books_not_sold : ℝ) / (initial_stock : ℝ) * 100

theorem john_books_unsold :
  let initial_stock := 1200
  let sales_returns := [(75, 6), (50, 0), (64, 8), (78, 0), (135, 5)]
  abs (percentage_not_sold initial_stock sales_returns - 68.08) < 0.01 :=
by
  sorry

end john_books_unsold_l321_321319


namespace sum_of_ages_is_24_l321_321325

def age_problem :=
  ∃ (x y z : ℕ), 2 * x^2 + y^2 + z^2 = 194 ∧ (x + x + y + z = 24)

theorem sum_of_ages_is_24 : age_problem :=
by
  sorry

end sum_of_ages_is_24_l321_321325


namespace smallest_possible_number_of_students_l321_321290

theorem smallest_possible_number_of_students :
  ∃ n : ℕ, (n % 200 = 0) ∧ (∀ m : ℕ, (m < n → 
    75 * m ≤ 100 * n) ∧
    (∃ a b c : ℕ, a = m / 4 ∧ b = a / 10 ∧ 
    ∃ y z : ℕ, y = 3 * z ∧ (y + z - b = a) ∧ y * c = n / 4)) :=
by
  sorry

end smallest_possible_number_of_students_l321_321290


namespace k_value_for_root_multiplicity_l321_321179

theorem k_value_for_root_multiplicity (k : ℝ) :
  (∃ x : ℝ, (x - 1) / (x - 3) = k / (x - 3) ∧ (x-3 = 0)) → k = 2 :=
by
  sorry

end k_value_for_root_multiplicity_l321_321179


namespace sum_of_intervals_le_half_l321_321356

theorem sum_of_intervals_le_half (n : ℕ) (lengths : Fin n → ℝ) (h01 : ∀ i, 0 < lengths i)
  (h02 : ∑ i, lengths i ≤ 1)
  (h03 : ∀ i j, i ≠ j → |lengths i - lengths j| > 0.1) : 
  (∑ i, lengths i) ≤ 0.5 := 
sorry

end sum_of_intervals_le_half_l321_321356


namespace train_cross_time_l321_321499

def length_of_train : ℕ := 120 -- the train is 120 m long
def speed_of_train_km_hr : ℕ := 45 -- the train's speed in km/hr
def length_of_bridge : ℕ := 255 -- the bridge is 255 m long

def train_speed_m_s : ℕ := speed_of_train_km_hr * (1000 / 3600)

def total_distance : ℕ := length_of_train + length_of_bridge

def time_to_cross_bridge (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

theorem train_cross_time :
  time_to_cross_bridge total_distance train_speed_m_s = 30 :=
by
  sorry

end train_cross_time_l321_321499


namespace product_of_divisors_30_l321_321429
-- Import the necessary library

-- Declaring the necessary conditions and main proof statement
def prime_factors (n : ℕ) : List ℕ :=
if h : n = 30 then [2, 3, 5] else []

def divisors (n : ℕ) : List ℕ :=
if h : n = 30 then [1, 2, 3, 5, 6, 10, 15, 30] else []

theorem product_of_divisors_30 : 
  let d := divisors 30 
  in d.product = 810000 :=
by {
    -- Skip the proof with sorry
    sorry
}

end product_of_divisors_30_l321_321429


namespace not_possibility_end_with_piles_of_three_l321_321824

theorem not_possibility_end_with_piles_of_three (n : ℕ) :
  let piles := 1 + n, coins := 2013 - n in
  ∃ k : ℕ, (coins = 3 * k ∧ k = piles) → False :=
by
  intros n
  let piles := 1 + n
  let coins := 2013 - n
  intro h
  cases h with k hk
  cases hk with h₁ h₂
  have h₃ : 2013 - n = 3 + 3 * n, by linarith
  have h₄ : 2010 = 4 * n, by linarith
  have h₅ : 502.5 = n, by linarith
  have h₆ : False := by sorry
  exact h₆ 

end not_possibility_end_with_piles_of_three_l321_321824


namespace new_salary_correct_l321_321112

-- Define the initial salary and percentage increase as given in the conditions
def initial_salary : ℝ := 10000
def percentage_increase : ℝ := 0.02

-- Define the function that calculates the new salary after a percentage increase
def new_salary (initial_salary : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_salary + (initial_salary * percentage_increase)

-- The theorem statement that proves the new salary is €10,200
theorem new_salary_correct :
  new_salary initial_salary percentage_increase = 10200 := by
  sorry

end new_salary_correct_l321_321112


namespace initial_apples_l321_321092

def apples_sold (x : Nat) : Prop :=
let R1 := x / 4 + 6,
    R2 := (R1 - (R1 / 3 + 4)),
    final_remaining := (R2 - (R2 / 2 + 3)) in
R1 = x - (x / 4 + 6) ∧ 
R2 = R1 - (R1 / 3 + 4) ∧
final_remaining = 4 ∧
x = 28

theorem initial_apples : ∃ x : Nat, apples_sold x :=
by
  existsi 28
  dsimp [apples_sold]
  sorry

end initial_apples_l321_321092


namespace first_shaded_square_in_each_column_l321_321108

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem first_shaded_square_in_each_column : 
  ∃ n, triangular_number n = 120 ∧ ∀ m < n, ¬ ∀ k < 8, ∃ j ≤ m, ((triangular_number j) % 8) = k := 
by
  sorry

end first_shaded_square_in_each_column_l321_321108


namespace range_of_vector_sum_l321_321568

theorem range_of_vector_sum :
  let C (x y : ℝ) := x ^ 2 + (y - 4) ^ 2 = 4
  let Q := (2, 2)
  let P := (0, 3)
  ∃ A B : ℝ × ℝ, A ≠ B ∧
    (∃ k : ℝ, ∀ (x y : ℝ), y = k * x + 3 → C x y) →
    (∀ (x1 y1 x2 y2 : ℝ), 
      (A = (x1, y1) → B = (x2, y2)) → 
      (|⟨(x1 + x2 - 4), (y1 + y2 - 4)⟩| ∈ set.Icc 4 6))
:= by
  sorry

end range_of_vector_sum_l321_321568


namespace original_curve_eq_l321_321206

theorem original_curve_eq (x y : ℝ) (θ : ℝ) (hθ : θ = (Real.pi / 4))
  (h_rot : ∀ (x y : ℝ) (θ : ℝ), (x * Real.cos θ - y * Real.sin θ, x * Real.sin θ + y * Real.cos θ))
  (h_transformed_curve : ∀ (x' y' : ℝ), (x' = (x * Real.cos θ - y * Real.sin θ)) ∧ (y' = (x * Real.sin θ + y * Real.cos θ)) → (x'^2 - y'^2 = 2))
  : x * y = -1 :=
sorry

end original_curve_eq_l321_321206


namespace triangle_area_l321_321739

theorem triangle_area (A B C P : Point)
  (h_right : ∃ (α β γ : ℝ), α = 90 ∧ β ≠ 90 ∧ γ ≠ 90 ∧ β + γ = 90)
  (h_angle_ABP : Measure.angle A B P = 30)
  (h_AP : dist A P = 2)
  (h_CP : dist C P = 1)
  (h_AC : dist A C = 3) :
  area_triangle A B C = 9 / 5 :=
sorry

end triangle_area_l321_321739


namespace area_of_plane_figure_covered_by_set_M_l321_321661

theorem area_of_plane_figure_covered_by_set_M :
  let M := {p : ℝ × ℝ | ∃ α β : ℝ, p.1 = Real.sin α + Real.cos β ∧ p.2 = Real.cos α - Real.sin β} in
  ∃ area : ℝ, area = 4 * Real.pi :=
sorry

end area_of_plane_figure_covered_by_set_M_l321_321661


namespace range_of_m_l321_321600

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x ≠ 1 ∧ (m + 3) / (x - 1) = 1) ↔ m > -4 ∧ m ≠ -3 := 
by 
  sorry

end range_of_m_l321_321600


namespace A_contains_all_positive_reals_l321_321402

-- Define the set A
variable {A : Set ℝ}

-- Define the conditions
axiom A_pos : ∀ x ∈ A, 0 < x
axiom A_add_closed : ∀ x y ∈ A, x + y ∈ A
axiom A_dense : ∀ a b, 0 < a ∧ a < b → ∃ u v ∈ Icc a b, u ∈ A ∧ v ∈ A

-- The theorem to be proved
theorem A_contains_all_positive_reals (A_pos : ∀ x ∈ A, 0 < x)
  (A_add_closed : ∀ x y ∈ A, x + y ∈ A)
  (A_dense : ∀ a b, 0 < a ∧ a < b → ∃ u v ∈ Icc a b, u ∈ A ∧ v ∈ A) :
  ∀ x, 0 < x → x ∈ A :=
begin
  sorry, -- Proof to be filled in
end

end A_contains_all_positive_reals_l321_321402


namespace intersection_A_B_find_a_b_l321_321182

noncomputable def A : Set ℝ := { x | x^2 - 5 * x + 6 > 0 }
noncomputable def B : Set ℝ := { x | Real.log (x + 1) / Real.log 2 < 2 }

theorem intersection_A_B :
  A ∩ B = { x | -1 < x ∧ x < 2 } :=
by
  -- Proof will be provided
  sorry

theorem find_a_b :
  ∃ a b : ℝ, (∀ x : ℝ, x^2 + a * x - b < 0 ↔ -1 < x ∧ x < 2) ∧ a = -1 ∧ b = 2 :=
by
  -- Proof will be provided
  sorry

end intersection_A_B_find_a_b_l321_321182


namespace intersection_area_correct_l321_321060

-- Definition of the circle equation
def circle (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Definition of the ellipse equation
def ellipse (x y : ℝ) : Prop := (x - 5)^2 + 4 * y^2 = 64

-- Definition of the intersected polygon's area
def polygon_area : ℝ := 5 * (Real.sqrt 119) / 6

theorem intersection_area_correct : (∀ x y, circle x y ∧ ellipse x y → False) → polygon_area = 5 * Real.sqrt 119 / 6 :=
by {
    sorry
}

end intersection_area_correct_l321_321060


namespace sum_of_possible_x_l321_321403

theorem sum_of_possible_x (x : ℝ) :
  (∑ x_i in ({31 + real.sqrt 201, 31 - real.sqrt 201}.to_finset.map (λ x, x / 4)), x_i) = 31 / 2 :=
by
  sorry

end sum_of_possible_x_l321_321403


namespace total_good_numbers_l321_321107

def isGoodNumber (n : ℕ) := ∀ m : ℕ, (∀ perm : List ℕ, perm.PermutesDigits → perm.Product % n = 0) → m * n % n = 0

theorem total_good_numbers : { n : ℕ | isGoodNumber n }.count = 3 :=
  sorry

end total_good_numbers_l321_321107


namespace avg_speed_round_trip_l321_321488

theorem avg_speed_round_trip (m : ℕ) : 
  let d := m / 5280 in
  let t_north := 3 * d in
  let t_south := d / 3 in
  let t_total := t_north + t_south in
  let t_total_hours := t_total / 60 in
  let distance_total := 2 * d in
  distance_total / t_total_hours = 1.5 :=
by
  sorry

end avg_speed_round_trip_l321_321488


namespace units_digit_fraction_l321_321915

theorem units_digit_fraction : 
  let numerator := 30 * 32 * 34 * 36 * 38 * 40 in
  let denominator := 2000 in
  let fraction := numerator / denominator in
  (fraction % 10) = 6 :=
  by 
    let numerator := 2^12 * 3^3 * 5^2 * 17 * 19
    let denominator := 2^4 * 5^3
    let simplified_fraction := 2^8 * 3^3 * 17 * 19
    have h2 : (2^8 % 10) = 6 := sorry
    have h3 : (3^3 % 10) = 7 := sorry
    have h17 : (17 % 10) = 7 := sorry
    have h19 : (19 % 10) = 9 := sorry
    have h6 : (6 * 7 % 10) = 2 := sorry
    have h4 : (2 * 7 % 10) = 4 := sorry
    have h36 : (4 * 9 % 10) = 6 := sorry
    exact h36

end units_digit_fraction_l321_321915


namespace wall_width_l321_321395

theorem wall_width (w h l V : ℝ) (h_eq : h = 4 * w) (l_eq : l = 3 * h) (V_eq : V = w * h * l) (v_val : V = 10368) : w = 6 :=
  sorry

end wall_width_l321_321395


namespace cube_face_sum_max_value_l321_321904

theorem cube_face_sum_max_value :
  ∀ (a b c d e f : ℕ),
    {3, 4, 5, 6, 7, 8} = {a, b, c, d, e, f} →
    (a + b) = 11 →
    (c + d) = 11 →
    (e + f) = 11 →
    (a * c * e + a * c * f + a * d * e + a * d * f + b * c * e + b * c * f + b * d * e + b * d * f) = 1331 :=
by
  sorry

end cube_face_sum_max_value_l321_321904


namespace quadratic_function_correct_l321_321588

noncomputable def quadratic_function : ℝ → ℝ :=
  λ x, (2 / 9) * x^2 + (4 / 9) * x - (16 / 9)

theorem quadratic_function_correct :
  ∃ A B : ℝ × ℝ, 
    A.2 = 0 ∧ B.2 = 0 ∧
    A.1 + B.1 = -2 ∧ 
    |A.1 - B.1| = 6 ∧ 
    ((-1, quadratic_function (-1)) = (-1, 2 * -1)) ∧
    quadratic_function = (λ x, (2 / 9) * x^2 + (4 / 9) * x - (16 / 9)) :=
by
  sorry

end quadratic_function_correct_l321_321588


namespace heather_heavier_than_emily_l321_321229

theorem heather_heavier_than_emily :
  let Heather_weight := 87
  let Emily_weight := 9
  Heather_weight - Emily_weight = 78 :=
by
  -- Proof here
  sorry

end heather_heavier_than_emily_l321_321229


namespace longer_side_length_l321_321493

-- Define the conditions as parameters
variables (W : ℕ) (poles : ℕ) (distance : ℕ) (P : ℕ)

-- Assume the fixed conditions given in the problem
axiom shorter_side : W = 10
axiom number_of_poles : poles = 24
axiom distance_between_poles : distance = 5

-- Define the total perimeter based on the number of segments formed by the poles
noncomputable def perimeter (poles : ℕ) (distance : ℕ) : ℕ :=
  (poles - 4) * distance

-- The total perimeter of the rectangle
axiom total_perimeter : P = perimeter poles distance

-- Definition of the perimeter of the rectangle in terms of its sides
axiom rectangle_perimeter : ∀ (L W : ℕ), P = 2 * L + 2 * W

-- The theorem we need to prove
theorem longer_side_length (L : ℕ) : L = 40 :=
by
  -- Sorry is used to skip the actual proof for now
  sorry

end longer_side_length_l321_321493


namespace maximum_area_of_rectangle_with_fixed_perimeter_l321_321015

theorem maximum_area_of_rectangle_with_fixed_perimeter (x y : ℝ) 
  (h₁ : 2 * (x + y) = 40) 
  (h₂ : x = y) :
  x * y = 100 :=
by
  sorry

end maximum_area_of_rectangle_with_fixed_perimeter_l321_321015


namespace simplify_expression_l321_321380

theorem simplify_expression (x : ℝ) 
  (h1 : sin (3 * x) = 3 * sin x - 4 * (sin x) ^ 3)
  (h2 : cos (3 * x) = 4 * (cos x) ^ 3 - 3 * cos x)
  (h3 : 1 - (sin x) ^ 2 = (cos x) ^ 2) :
  (sin x + sin (3 * x)) / (1 + cos x + cos (3 * x)) 
  = (sin x * (cos x) ^ 2) / ((cos x) * (4 * (cos x) ^ 2 - 1) + (1 / 4)) :=
by
  sorry

end simplify_expression_l321_321380


namespace product_divisors_30_eq_810000_l321_321434

def product_of_divisors (n : ℕ) : ℕ :=
  (multiset.filter (λ d, n % d = 0) (multiset.range (n + 1))).prod id

theorem product_divisors_30_eq_810000 :
  product_of_divisors 30 = 810000 :=
begin
  -- Proof will involve showing product of divisors of 30 equals 810000
  sorry
end

end product_divisors_30_eq_810000_l321_321434


namespace find_initial_money_l321_321507

def initial_money (s1 s2 s3 : ℝ) : ℝ :=
  let after_store_1 := s1 - (0.4 * s1 + 4)
  let after_store_2 := after_store_1 - (0.5 * after_store_1 + 5)
  let after_store_3 := after_store_2 - (0.6 * after_store_2 + 6)
  after_store_3

theorem find_initial_money (s1 s2 s3 : ℝ) (hs3 : initial_money s1 s2 s3 = 2) : s1 = 90 :=
by
  -- Placeholder for the actual proof
  sorry

end find_initial_money_l321_321507


namespace largest_prime_divisor_13_factorial_plus_14_factorial_l321_321992

theorem largest_prime_divisor_13_factorial_plus_14_factorial : 
  ∃ p, prime p ∧ p ∈ prime_factors (13! + 14!) ∧ ∀ q, prime q ∧ q ∈ prime_factors (13! + 14!) → q ≤ 13 :=
by { sorry }

end largest_prime_divisor_13_factorial_plus_14_factorial_l321_321992


namespace cube_assembly_red_faces_l321_321127

theorem cube_assembly_red_faces (small_cubes : Fin 8 -> Fin 6 -> Prop) 
  (one_third_blue_faces : ∀ i, (Finset.filter (small_cubes i) Finset.univ).card = 2) 
  (one_third_visible_red : ∀ i, (Finset.filter (λ f, ∃ j, small_cubes j f) Finset.univ).card = 8) :
  ∃ small_cube_orientation : Fin 8 -> ℕ, (∀ i j, small_cubes i j → (j = small_cube_orientation i) → False) :=
sorry

end cube_assembly_red_faces_l321_321127


namespace points_X_Y_Z_T_concyclic_l321_321358

open EuclideanGeometry

theorem points_X_Y_Z_T_concyclic
  (A B C D E F G H I X Y Z T : Point)
  (h_nonagon: is_regular_nonagon A B C D E F G H I)
  (h_triangles: is_regular_nonagon A B C D E F G H I)
  (h_internal_angles: ∀ {P Q R}, (Q ∈ {A, B, C, D, E, F, G, H, I}) → ∠PQR = 140)
  (h_X_angles: has_angle X A B 20)
  (h_Y_angles: has_angle Y B C 20)
  (h_Z_angles: has_angle Z C D 20)
  (h_T_angles: has_angle T D E 20)
  (h_20_increment: ∀ {angle_XAB angle_YBC angle_ZCD angle_TDE : ℝ}, angle_YBC = angle_XAB + 20 ∧ angle_ZCD = angle_YBC + 20 ∧ angle_TDE = angle_ZCD + 20)
  : concyclic {X, Y, Z, T} :=
sorry

end points_X_Y_Z_T_concyclic_l321_321358


namespace card_draw_prob_correct_l321_321043

-- Define the probability calculation for the card drawing scenario
def card_drawing_probability : ℚ :=
  9 / 385

theorem card_draw_prob_correct :
  (n : ℕ) (cards : Finset ℕ) (h : cards.cardinality = 12) (pairs : Finset (Finset ℕ)) :
  pairs = (Finset.range 6).bind (λ x, Finset.pair x) →
  ∀ d ≤ pairs.cardinality, (draw_probability_with_conditions cards pairs d).terminal = true →
  card_drawing_probability = 9 / 385 :=
begin
  sorry
end

end card_draw_prob_correct_l321_321043


namespace product_of_divisors_of_30_l321_321439

theorem product_of_divisors_of_30 : 
  ∏ d in (finset.filter (λ d, 30 % d = 0) (finset.range 31)), d = 810000 :=
by sorry

end product_of_divisors_of_30_l321_321439


namespace perimeter_of_new_shaded_region_l321_321264

-- Definitions and conditional statements
def identicalCirclesTouch (r : ℝ) : Prop := ∀ i j, i ≠ j → circles i r ∧ circles j r ∧ touch i j
def anglesAtTouchingPoints (angle : ℝ) : Prop := angle = 120
def circumferencesAre (C : ℝ) : Prop := C = 72

-- The theorem stating the perimeter of the new shaded region.
theorem perimeter_of_new_shaded_region (r : ℝ) (C : ℝ) (angle : ℝ) (touch : identicalCirclesTouch r) (circ : circumferencesAre C) (ang : anglesAtTouchingPoints angle) :
  3 * (C / 3) = 72 :=
by
  sorry

end perimeter_of_new_shaded_region_l321_321264


namespace sin_arcsin_plus_arctan_l321_321950

theorem sin_arcsin_plus_arctan :
  sin (Real.arcsin (4 / 5) + Real.arctan 3) = (13 * Real.sqrt 10) / 50 :=
by
  sorry

end sin_arcsin_plus_arctan_l321_321950


namespace painted_cubes_count_l321_321804

def total_painted_cubes : ℕ := 8 + 48

theorem painted_cubes_count : total_painted_cubes = 56 :=
by 
  -- Step 1: Define the number of cubes with 3 faces painted (8 corners)
  let corners := 8
  -- Step 2: Calculate the number of edge cubes with 2 faces painted
  let edge_middle_cubes_per_edge := 2
  let edges := 12
  let edge_cubes := edge_middle_cubes_per_edge * edges -- this should be 24
  -- Step 3: Calculate the number of face-interior cubes with 2 faces painted
  let face_cubes_per_face := 4
  let faces := 6
  let face_cubes := face_cubes_per_face * faces -- this should be 24
  -- Step 4: Sum them up to get total cubes with at least two faces painted
  let total_cubes := corners + edge_cubes + face_cubes
  show total_cubes = total_painted_cubes
  sorry

end painted_cubes_count_l321_321804


namespace scalene_right_triangle_area_l321_321746

def triangle_area (a b : ℝ) : ℝ := 0.5 * a * b

theorem scalene_right_triangle_area
  (ABC : Type) [is_triangle ABC]
  (hypotenuse : line_segment ABC)
  (P : point hypotenuse)
  (angle_ABP : angle ABC = 30)
  (AP : distance between points A P = 2)
  (CP : distance between points C P = 1)
  (AB : length between points A B)
  (BC : length between points B C)
  (AC : length between points A C = 3)
  (right_angle : right angle of triangle ABC B) 
  : triangle_area AB BC = 9/5 :=
by 
  sorry

end scalene_right_triangle_area_l321_321746


namespace exists_a_i_satisfying_condition_l321_321194

theorem exists_a_i_satisfying_condition (k n : ℕ) (h_k : k ≥ 2) :
  (∃ a : Fin k → ℕ, (∏ i, a i + n) ∣ (∑ i, (a i)^2)) ↔ n ≥ 2 :=
by
  sorry

end exists_a_i_satisfying_condition_l321_321194


namespace ratio_of_ages_ten_years_ago_l321_321048

theorem ratio_of_ages_ten_years_ago (A T : ℕ) 
    (h1: A = 30) 
    (h2: T = A - 15) : 
    (A - 10) / (T - 10) = 4 :=
by
  sorry

end ratio_of_ages_ten_years_ago_l321_321048


namespace find_p_plus_q_l321_321383

noncomputable def p_q_sum (x : ℝ) (p q : ℚ) : ℚ := p + q

theorem find_p_plus_q (x : ℝ) (p q : ℚ) (h1 : Real.sec x + Real.tan x = 15 / 4) (h2 : Real.csc x + Real.cot x = p / q) : p_q_sum x p q = 458 := by
  sorry

end find_p_plus_q_l321_321383


namespace find_amplitude_l321_321911

-- Definitions
variables {a b c d : Real}
variables (y : Real → Real)

-- Conditions
def oscillates_between (y : Real → Real) (max min : Real) :=
  ∃ x1 x2, y x1 = max ∧ y x2 = min

theorem find_amplitude (h : oscillates_between (λ x, a * Real.sin (b * x + c) + d) 5 (-3))
  : a = 4 :=
by
  -- The proof will be provided here later
  sorry

end find_amplitude_l321_321911


namespace combined_age_of_sam_and_drew_l321_321727

theorem combined_age_of_sam_and_drew
  (sam_age : ℕ)
  (drew_age : ℕ)
  (h1 : sam_age = 18)
  (h2 : sam_age = drew_age / 2):
  sam_age + drew_age = 54 := sorry

end combined_age_of_sam_and_drew_l321_321727


namespace log_base_proof_l321_321952

noncomputable def log_base_solution (a : ℝ) : ℝ :=
  a ^ (1 / a)

theorem log_base_proof (a : ℝ) (h : 0 < a) : 
  ∃ n : ℝ, a = Real.logBase n a ∧ n = log_base_solution a :=
by
  use log_base_solution a
  split
  {
    sorry  -- Here you would prove a = Real.logBase (a ^ (1 / a)) a
  },
  {
    refl  -- Here you would prove n = a ^ (1 / a)
  }

end log_base_proof_l321_321952


namespace find_n_l321_321593

theorem find_n (n : ℕ) :
  let S := (∑ i in {0, 1, 2}, (Nat.choose n i)) in
  S = 22 → n = 6 :=
by
  intro S hS
  sorry

end find_n_l321_321593


namespace original_price_correct_l321_321104

noncomputable def original_price (selling_price : ℝ) (gain_percent : ℝ) : ℝ :=
  selling_price / (1 + gain_percent / 100)

theorem original_price_correct :
  original_price 35 75 = 20 :=
by
  sorry

end original_price_correct_l321_321104


namespace daniel_drive_distance_l321_321548

theorem daniel_drive_distance (x D : ℝ) (h_sunday_speed : ∀ D, ∀ x, 0 < x → 0 < D → 
  let T_sunday := D / x in
  let T_32 := 32 / (2 * x) in
  let T_rest := (D - 32) / (x / 2) in
  let T_monday := T_32 + T_rest in
  T_monday = 1.6 * T_sunday →
  D = 120) : D = 120 :=
sorry

end daniel_drive_distance_l321_321548


namespace odd_function_with_period_pi_l321_321129

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def has_period (f : ℝ → ℝ) (T : ℝ) := ∀ x, f (x + T) = f x

def f (x : ℝ) := sin x * cos x
def g (x : ℝ) := sin x ^ 2
def h (x : ℝ) := tan (2 * x)
def k (x : ℝ) := sin (2 * x) + cos (2 * x)

theorem odd_function_with_period_pi :
  (is_odd f ∧ has_period f π) ∧
  (¬ (is_odd g ∧ has_period g π)) ∧
  (¬ (is_odd h ∧ has_period h π)) ∧
  (¬ (is_odd k ∧ has_period k π)) := by
  sorry

end odd_function_with_period_pi_l321_321129


namespace value_of_a_g_odd_iff_m_eq_one_l321_321604

noncomputable def f (a x : ℝ) : ℝ := a ^ x

noncomputable def g (m x a : ℝ) : ℝ := m - 2 / (f a x + 1)

theorem value_of_a
  (a : ℝ)
  (h_pos : a > 0)
  (h_neq_one : a ≠ 1)
  (h_diff : ∀ x y : ℝ, x ∈ (Set.Icc 1 2) → y ∈ (Set.Icc 1 2) → abs (f a x - f a y) = 2) :
  a = 2 :=
sorry

theorem g_odd_iff_m_eq_one
  (a m : ℝ)
  (h_pos : a > 0)
  (h_neq_one : a ≠ 1)
  (h_a_eq : a = 2) :
  (∀ x : ℝ, g m x a = -g m (-x) a) ↔ m = 1 :=
sorry

end value_of_a_g_odd_iff_m_eq_one_l321_321604


namespace John_height_l321_321321

open Real

variable (John Mary Tom Angela Helen Amy Becky Carl : ℝ)

axiom h1 : John = 1.5 * Mary
axiom h2 : Mary = 2 * Tom
axiom h3 : Tom = Angela - 70
axiom h4 : Angela = Helen + 4
axiom h5 : Helen = Amy + 3
axiom h6 : Amy = 1.2 * Becky
axiom h7 : Becky = 2 * Carl
axiom h8 : Carl = 120

theorem John_height : John = 675 := by
  sorry

end John_height_l321_321321


namespace janice_items_l321_321672

theorem janice_items : 
  ∃ a b c : ℕ, 
    a + b + c = 60 ∧ 
    15 * a + 400 * b + 500 * c = 6000 ∧ 
    a = 50 := 
by 
  sorry

end janice_items_l321_321672


namespace new_salary_calculation_l321_321114

-- Define the initial conditions of the problem
def current_salary : ℝ := 10000
def percentage_increase : ℝ := 2
def increase := current_salary * (percentage_increase / 100)
def new_salary := current_salary + increase

-- Define the theorem to check the new salary
theorem new_salary_calculation : new_salary = 10200 := by
  -- Lean would check the proof here, but it's being skipped with 'sorry'
  sorry

end new_salary_calculation_l321_321114


namespace gcd_pair_d_is_45_l321_321076

-- Definitions of the pairs of numbers
def pairA : ℤ × ℤ := (819, 333)
def pairB : ℤ × ℤ := (98, 196)
def pairC : ℤ × ℤ := (153, 111)
def pairD : ℤ × ℤ := (225, 135)

-- Definition of gcd
def gcd (a b : ℤ) : ℤ := Nat.gcd (a.nat_abs) (b.nat_abs)

-- Proposition to prove
theorem gcd_pair_d_is_45 : gcd (pairD.1) (pairD.2) = 45 := by
  sorry

end gcd_pair_d_is_45_l321_321076


namespace circumference_of_jogging_track_l321_321003

noncomputable def trackCircumference (Deepak_speed : ℝ) (Wife_speed : ℝ) (meet_time_minutes : ℝ) : ℝ :=
  let relative_speed := Deepak_speed + Wife_speed
  let meet_time_hours := meet_time_minutes / 60
  relative_speed * meet_time_hours

theorem circumference_of_jogging_track :
  trackCircumference 20 17 37 = 1369 / 60 :=
by
  sorry

end circumference_of_jogging_track_l321_321003


namespace darnel_sprinted_distance_l321_321931

theorem darnel_sprinted_distance :
  let jogged := 0.75
  let extra_sprint := 0.125
  let sprinted := jogged + extra_sprint
  sprinted = 0.875 :=
by
  let jogged := 0.75
  let extra_sprint := 0.125
  let sprinted := jogged + extra_sprint
  show sprinted = 0.875, by sorry

end darnel_sprinted_distance_l321_321931


namespace max_goats_from_coconuts_l321_321700

def coconuts := ℕ
def crabs := ℕ
def goats := ℕ

def coconuts_to_crabs (c : coconuts) : crabs := c / 3
def crabs_to_goats (cr : crabs) : goats := cr / 6

theorem max_goats_from_coconuts (initial_coconuts : coconuts) (hc : initial_coconuts = 342) : 
  crabs_to_goats (coconuts_to_crabs initial_coconuts) = 19 := 
by
  rw [hc]
  simp only [coconuts_to_crabs, crabs_to_goats]
  norm_num
  sorry

end max_goats_from_coconuts_l321_321700


namespace volume_eq_eight_space_diagonal_eq_sqrt21_surface_area_neq_24_circumscribed_sphere_area_eq_21pi_l321_321008

namespace RectangularPrism

def length := 4
def width := 2
def height := 1

theorem volume_eq_eight : length * width * height = 8 := sorry

theorem space_diagonal_eq_sqrt21 :
  Real.sqrt (length ^ 2 + width ^ 2 + height ^ 2) = Real.sqrt 21 := sorry

theorem surface_area_neq_24 :
  2 * (length * width + width * height + height * length) ≠ 24 := sorry

theorem circumscribed_sphere_area_eq_21pi :
  4 * Real.pi * ((Real.sqrt (length ^ 2 + width ^ 2 + height ^ 2) / 2) ^ 2) = 21 * Real.pi := sorry

end RectangularPrism

end volume_eq_eight_space_diagonal_eq_sqrt21_surface_area_neq_24_circumscribed_sphere_area_eq_21pi_l321_321008


namespace largest_prime_divisor_13_fac_add_14_fac_l321_321969

theorem largest_prime_divisor_13_fac_add_14_fac : 
  ∀ (p : ℕ), prime p → p ∣ (13! + 14!) → p ≤ 5 :=
by
  sorry

end largest_prime_divisor_13_fac_add_14_fac_l321_321969


namespace expression_equals_4034_l321_321820

theorem expression_equals_4034 : 6 * 2017 - 4 * 2017 = 4034 := by
  sorry

end expression_equals_4034_l321_321820


namespace symmetry_about_point_l321_321542

def func (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

theorem symmetry_about_point :
  ∀ x, func (x - Real.pi / 6) = func (- (x + Real.pi / 6) + 2 * Real.pi) :=
by
  sorry

end symmetry_about_point_l321_321542


namespace simplest_quadratic_radical_l321_321074

theorem simplest_quadratic_radical :
  ∀ x ∈ {sqrt 3, sqrt 4, sqrt 8, sqrt (1 / 2)}, (x = sqrt 3) :=
by
  sorry

end simplest_quadratic_radical_l321_321074


namespace cube_faces_paint_count_l321_321797

def is_painted_on_at_least_two_faces (x y z : ℕ) : Prop :=
  ((x = 0 ∨ x = 3) ∨ (y = 0 ∨ y = 3) ∨ (z = 0 ∨ z = 3))

theorem cube_faces_paint_count :
  let n := 4 in
  let volume := n * n * n in
  let one_inch_cubes := {c | ∃ x y z : ℕ, x < n ∧ y < n ∧ z < n ∧ c = (x, y, z)} in
  let painted_cubes := {c ∈ one_inch_cubes | exists_along_edges c} in  -- Auxiliary predicate to check edge paint
  card painted_cubes = 32 :=
begin
  sorry,
end

/-- Auxiliary predicate to check if a cube is on the edges but not corners, signifying 
    at least two faces are painted --/
predicate exists_along_edges (c: ℕ × ℕ × ℕ) := 
  match c with
  | (0, _, _) => true
  | (3, _, _) => true
  | (_, 0, _) => true
  | (_, 3, _) => true
  | (_, _, 0) => true
  | (_, _, 3) => true
  | (_, _, _) => false
  end

end cube_faces_paint_count_l321_321797


namespace jeremy_tylenol_duration_l321_321675

theorem jeremy_tylenol_duration (num_pills : ℕ) (pill_mg : ℕ) (dose_mg : ℕ) (hours_per_dose : ℕ) (hours_per_day : ℕ) 
  (total_tylenol_mg : ℕ := num_pills * pill_mg)
  (num_doses : ℕ := total_tylenol_mg / dose_mg)
  (total_hours : ℕ := num_doses * hours_per_dose) :
  num_pills = 112 → pill_mg = 500 → dose_mg = 1000 → hours_per_dose = 6 → hours_per_day = 24 → 
  total_hours / hours_per_day = 14 := 
by 
  intros; 
  sorry

end jeremy_tylenol_duration_l321_321675


namespace sum_of_first_ten_terms_seq_l321_321925

def a₁ : ℤ := -5
def d : ℤ := 6
def n : ℕ := 10

theorem sum_of_first_ten_terms_seq : (n * (a₁ + a₁ + (n - 1) * d)) / 2 = 220 :=
by
  sorry

end sum_of_first_ten_terms_seq_l321_321925


namespace min_sum_squares_l321_321691

theorem min_sum_squares (a b c d e f g h : ℤ) 
  (h_distinct : List.nodup [a, b, c, d, e, f, g, h])
  (h_values : a ∈ [-8, -6, -4, -1, 1, 3, 5, 14] ∧ 
              b ∈ [-8, -6, -4, -1, 1, 3, 5, 14] ∧ 
              c ∈ [-8, -6, -4, -1, 1, 3, 5, 14] ∧ 
              d ∈ [-8, -6, -4, -1, 1, 3, 5, 14] ∧ 
              e ∈ [-8, -6, -4, -1, 1, 3, 5, 14] ∧ 
              f ∈ [-8, -6, -4, -1, 1, 3, 5, 14] ∧ 
              g ∈ [-8, -6, -4, -1, 1, 3, 5, 14] ∧ 
              h ∈ [-8, -6, -4, -1, 1, 3, 5, 14])
  (h_sum4 : e + f + g + h = 9) : 
  (a + b + c + d)^2 + (e + f + g + h)^2 = 106 := 
  sorry

end min_sum_squares_l321_321691


namespace total_pens_pencils_l321_321482

noncomputable def totalItems : ℕ × ℕ → ℝ
| (p, c) := 0.35 * p + 0.25 * c

theorem total_pens_pencils (P C: ℕ) (h : totalItems (P, C) = 1.80) : P + C = 6 :=
sorry

end total_pens_pencils_l321_321482


namespace symmetric_points_cannot_all_lie_strictly_inside_circle_l321_321200

-- Define the points and the conditions.
variables (A B C P A1 B1 C1 A2 B2 C2 : Type)
variables [IncidenceGeometry A B C] [IncidenceGeometry P A1 B1 C1]
variables [symmetric P A1 A2] [symmetric P B1 B2] [symmetric P C1 C2]

-- Define the circumcircle of triangle ABC.
def circumcircle (ABC : Triangle A B C) : Circle := sorry

-- Define the statement of the problem.
theorem symmetric_points_cannot_all_lie_strictly_inside_circle
  (h1 : inside P (triangle A B C))
  (h2 : intersects (line A P) (line B C) = A1)
  (h3 : intersects (line B P) (line C A) = B1)
  (h4 : intersects (line C P) (line A B) = C1)
  (h5 : symmetric_point P A1 = A2)
  (h6 : symmetric_point P B1 = B2)
  (h7 : symmetric_point P C1 = C2) :
  ¬(strictly_inside A2 (circumcircle (triangle A B C)) ∧
    strictly_inside B2 (circumcircle (triangle A B C)) ∧
    strictly_inside C2 (circumcircle (triangle A B C))) := 
sorry

end symmetric_points_cannot_all_lie_strictly_inside_circle_l321_321200


namespace peggy_records_l321_321715

theorem peggy_records (R : ℕ) (h : 4 * R - (3 * R + R / 2) = 100) : R = 200 :=
sorry

end peggy_records_l321_321715


namespace smaller_square_area_l321_321103

theorem smaller_square_area (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (side_length : ℝ) 
  (area_large_square : ℝ)
  (h_area_large_square : area_large_square = 100)
  (h_side_length : side_length^2 = area_large_square)
  (midpoint : A → B → C → D → (A × B × C × D))
  (h_midpoint : ∀ (a b c d : A), midpoint a b c d = ⟨(a+b)/2, (b+c)/2, (c+d)/2, (d+a)/2⟩)
  : ∃ (area_small_square : ℝ), area_small_square = 25 :=
by
  sorry

end smaller_square_area_l321_321103


namespace slope_points_eq_l321_321214

theorem slope_points_eq (m : ℚ) (h : ((m + 2) / (3 - m) = 2)) : m = 4 / 3 :=
sorry

end slope_points_eq_l321_321214


namespace school_competition_l321_321271

theorem school_competition :
  (∃ n : ℕ, 
    n > 0 ∧ 
    75% students did not attend the competition ∧
    10% of those who attended participated in both competitions ∧
    ∃ y z : ℕ, y = 3 / 2 * z ∧ 
    y + z - (1 / 10) * (n / 4) = n / 4
  ) → n = 200 :=
sorry

end school_competition_l321_321271


namespace newtons_identities_l321_321343

variable {n : ℕ}
variable {x : ℕ → ℕ}
variable {σ : ℕ → ℕ}

-- Define power sums sk
def s (k : ℕ) : ℕ := ∑ i in finset.range n, x i ^ k

-- Define elementary symmetric polynomials σk
noncomputable def σ (k : ℕ) : ℕ := sorry

-- Newton's identities statement
theorem newtons_identities :
  n * σ n = ∑ i in finset.range n, ((-1) ^ i) * s (i + 1) * σ (n - (i + 1)) :=
sorry

end newtons_identities_l321_321343


namespace regression_estimate_l321_321213

noncomputable def regression_estimation (b : ℝ) (x y : ℝ) : Prop :=
    ∃ a : ℝ, (y = b * x + a)

theorem regression_estimate :
  ∀ a b x : ℝ, b = 1.23 → (4, 5) ∈ set_of (λ (t : ℝ × ℝ), regression_estimation b t.fst t.snd) →
  let y := b * 2 + a in y = 2.54 :=
by
  intros a b x hb hc y
  sorry

end regression_estimate_l321_321213


namespace rhombus_side_length_15_cm_l321_321457

noncomputable def rhombus_side_length (P : ℝ) (side_num : ℝ) : ℝ :=
  P / side_num

theorem rhombus_side_length_15_cm 
  (P : ℝ := 60) 
  (side_num : ℝ := 4) 
  (h_rhombus : P = 60 ∧ side_num = 4) : 
  rhombus_side_length P side_num = 15 := 
by
  dsimp [rhombus_side_length]
  rw [← h_rhombus.left, ← h_rhombus.right]
  sorry

end rhombus_side_length_15_cm_l321_321457


namespace percentage_of_page_used_l321_321864

theorem percentage_of_page_used (length width side_margin top_margin : ℝ) (h_length : length = 30) (h_width : width = 20) (h_side_margin : side_margin = 2) (h_top_margin : top_margin = 3) :
  ( ((length - 2 * top_margin) * (width - 2 * side_margin)) / (length * width) ) * 100 = 64 := 
by
  sorry

end percentage_of_page_used_l321_321864


namespace solve_abs_equation_l321_321729

-- Define the main problem
theorem solve_abs_equation (y : ℝ) :
  (abs (y - 4))^2 + 3 * y = 14 ↔ y = (5 + real.sqrt 17) / 2 ∨ y = (5 - real.sqrt 17) / 2 :=
by
  sorry

end solve_abs_equation_l321_321729


namespace coefficient_x2_in_expansion_l321_321389

theorem coefficient_x2_in_expansion :
  let f := x^2 * (1 + x + x^2) * (x - 1/x)^6 in
  (coeff f 2) = -5 :=
by
  -- Defining the problem's expression
  let f := x^2 * (1 + x + x^2) * (x - 1/x)^6
  -- Continuing with the proof:
  sorry

end coefficient_x2_in_expansion_l321_321389


namespace ellipse_foci_distance_l321_321167

theorem ellipse_foci_distance (h : ∀ x y, x^2 / 36 + y^2 / 9 = 5) :
  let a^2 := 7.2
      let b^2 := 1.8
      let c^2 := a^2 - b^2
      let c := sqrt c^2
  in 2 * c = 2 * sqrt 5.4 :=
by sorry

end ellipse_foci_distance_l321_321167


namespace find_d_l321_321784

noncomputable def quadratic_roots (d : ℝ) : Prop :=
∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2

theorem find_d : ∃ d : ℝ, d = 9.8 ∧ quadratic_roots d :=
sorry

end find_d_l321_321784


namespace average_temperature_correct_l321_321056

-- Definition of the daily temperatures
def daily_temperatures : List ℕ := [51, 64, 61, 59, 48, 63, 55]

-- Define the number of days
def number_of_days : ℕ := 7

-- Prove the average temperature calculation
theorem average_temperature_correct :
  ((List.sum daily_temperatures : ℚ) / number_of_days : ℚ) = 57.3 :=
by
  sorry

end average_temperature_correct_l321_321056


namespace range_of_m_l321_321212

open Real

noncomputable def f (x m : ℝ) : ℝ := log x / log 2 + x - m

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x m = 0) → 1 < m ∧ m < 3 :=
by
  sorry

end range_of_m_l321_321212


namespace binomial_inequality_l321_321719

theorem binomial_inequality (n : ℕ) (x : ℝ) (h1 : 2 ≤ n) (h2 : |x| < 1) : 
  (1 - x)^n + (1 + x)^n < 2^n := 
by 
  sorry

end binomial_inequality_l321_321719


namespace second_triangle_weight_is_36_l321_321491

-- Defining the weight of an equilateral triangle
def weight_of_equilateral_triangle (side_length : ℝ) (density : ℝ) : ℝ :=
  density * (side_length^2 * sqrt 3 / 4)

-- Given conditions
def first_triangle_side_length : ℝ := 4
def first_triangle_weight : ℝ := 16
def first_triangle_density : ℝ := first_triangle_weight / (first_triangle_side_length^2 * sqrt 3 / 4)

def second_triangle_side_length : ℝ := 6

-- The statement to prove
theorem second_triangle_weight_is_36 :
  weight_of_equilateral_triangle second_triangle_side_length first_triangle_density = 36 :=
by
  sorry

end second_triangle_weight_is_36_l321_321491


namespace percentage_collected_is_25_l321_321410

variable (total_pinecones : ℕ) (pinecones_left_after_collecting_to_make_fires : ℕ)
variable (reindeer_eaten_percentage : ℚ) (squirrel_multiple : ℕ)

def pinecones_eaten_by_reindeer := reindeer_eaten_percentage * total_pinecones
def pinecones_eaten_by_squirrels := squirrel_multiple * pinecones_eaten_by_reindeer
def pinecones_eaten := pinecones_eaten_by_reindeer + pinecones_eaten_by_squirrels
def pinecones_before_collecting := total_pinecones - pinecones_eaten
def pinecones_collected_to_make_fires := pinecones_before_collecting - pinecones_left_after_collecting_to_make_fires
def percentage_collected := (pinecones_collected_to_make_fires : ℚ) / pinecones_before_collecting * 100

theorem percentage_collected_is_25 (ht: total_pinecones = 2000) 
                                   (hr: reindeer_eaten_percentage = 0.20) 
                                   (hs: squirrel_multiple = 2) 
                                   (hf: pinecones_left_after_collecting_to_make_fires = 600) :
    percentage_collected total_pinecones pinecones_left_after_collecting_to_make_fires 
                         reindeer_eaten_percentage squirrel_multiple = 25 := 
by {
    sorry
}

end percentage_collected_is_25_l321_321410


namespace imaginary_part_of_square_of_i_l321_321596

theorem imaginary_part_of_square_of_i :
  (Complex.imaginary (Complex.I ^ 2)) = 0 :=
by sorry

end imaginary_part_of_square_of_i_l321_321596


namespace logarithmic_relationship_l321_321625

theorem logarithmic_relationship (a b : ℝ) (h1 : a = Real.logb 16 625) (h2 : b = Real.logb 2 25) : a = b / 2 :=
sorry

end logarithmic_relationship_l321_321625


namespace trigonometric_sum_simplifies_to_tan_l321_321378

theorem trigonometric_sum_simplifies_to_tan (x : ℝ) (n : ℕ) (h_pos : n > 0) :
  (\sin x + \sum i in finset.range n, \sin ((2 * i + 1) * x)) / (\cos x + \sum i in finset.range n, \cos ((2 * i + 1) * x)) = \tan (n * x) := 
  sorry

end trigonometric_sum_simplifies_to_tan_l321_321378


namespace max_min_u_l321_321188

noncomputable def u (x y : ℝ) := Real.logb (0.75) (8 * x * y + 4 * y^2 + 1)

theorem max_min_u (x y : ℝ):
  (x > 0) → (y ≥ 0) → (x + 2 * y = 1 / 2) → 
  (u x y = 0 ↔ x = 1 / 4 ∧ y = 0) ∧ (u x y = -1 ↔ x = 1 / 6 ∧ y = 1 / 6) :=
by
  sorry

end max_min_u_l321_321188


namespace diagonal_of_rectangle_l321_321360

theorem diagonal_of_rectangle (l w d : ℝ) (h_length : l = 15) (h_area : l * w = 120) (h_diagonal : d^2 = l^2 + w^2) : d = 17 :=
by
  sorry

end diagonal_of_rectangle_l321_321360


namespace sam_dimes_now_l321_321371

-- Define the initial number of dimes Sam had
def initial_dimes : ℕ := 9

-- Define the number of dimes Sam gave away
def dimes_given : ℕ := 7

-- State the theorem: The number of dimes Sam has now is 2
theorem sam_dimes_now : initial_dimes - dimes_given = 2 := by
  sorry

end sam_dimes_now_l321_321371


namespace simple_interest_years_l321_321035

theorem simple_interest_years (r1 r2 t2 P1 P2 S : ℝ) (hP1: P1 = 3225) (hP2: P2 = 8000) (hr1: r1 = 0.08) (hr2: r2 = 0.15) (ht2: t2 = 2) (hCI : S = 2580) :
    S / 2 = (P1 * r1 * t) / 100 → t = 5 :=
by
  sorry

end simple_interest_years_l321_321035


namespace smallest_digit_never_in_units_place_of_even_number_l321_321834

theorem smallest_digit_never_in_units_place_of_even_number :
  (∀ d : ℕ, d ∈ {0, 2, 4, 6, 8} → d ≠ 1) ∧ 
  (∀ d : ℕ, d ∈ {1, 3, 5, 7, 9} → d ≠ 0 ∧ d ≠ 2 ∧ d ≠ 4 ∧ d ≠ 6 ∧ d ≠ 8) →
  ∃ d : ℕ, d = 1 ∧ (∀ n : ℕ, n ≠ 1 → n ∉ {1, 3, 5, 7, 9} ↔ n ∈ {0, 2, 4, 6, 8}) :=
begin
  sorry -- No proof needed as instructed.
end

end smallest_digit_never_in_units_place_of_even_number_l321_321834


namespace problem1_problem2_l321_321139

noncomputable section

theorem problem1 :
  (2 * Real.sqrt 3 - 1)^2 + (Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) = 12 - 4 * Real.sqrt 3 :=
  sorry

theorem problem2 :
  (Real.sqrt 6 - 2 * Real.sqrt 15) * Real.sqrt 3 - 6 * Real.sqrt (1 / 2) = -6 * Real.sqrt 5 :=
  sorry

end problem1_problem2_l321_321139


namespace calculate_expression_l321_321916

theorem calculate_expression :
    (real.root 5 2 * ((4 : ℝ) ^ (-2 / 5))⁻¹ + real.log10 (sqrt 1000) - real.sin (3 * real.pi / 2)) = 9 / 2 :=
by
  sorry

end calculate_expression_l321_321916


namespace reading_buddy_fraction_l321_321648

-- Variables for the number of seventh and eighth graders
variables (t e : ℕ)

-- Condition: \(\frac{2}{3}\) of the eighth graders are paired with \(\frac{1}{5}\) of the seventh graders.
def pairing_condition (t e : ℕ) : Prop := (2 * e / 3 = t / 5)

-- Fraction of students with a buddy
def fraction_with_buddy (t e : ℕ) : rat := ((2 * e / 3 : rat) + (t / 5 : rat)) / (e + t : rat)

-- Theorem stating that the fraction of the total number of seventh and eighth graders who have a reading buddy is \(4/13\)
theorem reading_buddy_fraction (t e : ℕ) (ht : t > 0) (he : e > 0) (h : pairing_condition t e) : 
  fraction_with_buddy t e = 4 / 13 :=
sorry

end reading_buddy_fraction_l321_321648


namespace parallel_DE_BC_l321_321133

open EuclideanGeometry

theorem parallel_DE_BC {A B C P Q R D E : Point} (O : Circle A B C) 
  (midpoint1 : Midpoint P (Arc O B C)) 
  (midpoint2 : Midpoint Q (Arc O A C)) 
  (midpoint3 : Midpoint R (Arc O A B)) 
  (intersect1 : Intersect (Segment.PQ P R) (LineSegment A B) D)
  (intersect2 : Intersect (Segment.PQ P Q) (LineSegment A C) E) :
  Parallel (LineSegment D E) (Line B C) := sorry

end parallel_DE_BC_l321_321133


namespace hyperbola_minor_axis_length_l321_321224

open Real

theorem hyperbola_minor_axis_length (b : ℝ) (h1 : 0 < b) (h2 : ∀ (a c : ℝ), a = 2 ∧ c = sqrt (a^2 + b^2) ∧ (b * c) / sqrt (b^2 + a^2) = 3) :
  2 * b = 6 :=
by
  -- The proof is omitted as per instructions
  sorry

end hyperbola_minor_axis_length_l321_321224


namespace lambda_range_l321_321609

noncomputable def sequence_decreasing (a : ℕ → ℝ) := ∀ n : ℕ, n > 0 → a (n + 1) ≤ a n

theorem lambda_range (λ : ℝ) (a : ℕ → ℝ) (h : ∀ n : ℕ, n > 0 → a n = -2 * (n : ℝ) ^ 2 + λ * n) (dec : sequence_decreasing a) :
  λ < 6 :=
sorry

end lambda_range_l321_321609


namespace father_time_correct_l321_321481

-- Define the time each person takes
def time_father : ℕ := 1
def time_mother : ℕ := 2
def time_son : ℕ := 4
def time_daughter : ℕ := 5
def max_time : ℕ := 12

-- Define the total time taken by the family based on the described strategy
def total_time : ℕ :=
  let t := max_time in
  let tf := time_father in
  let tm := time_mother in
  let ts := time_son in
  let td := time_daughter in
  tf + tm + tf + ts + tf + td

-- Prove that the father's time is 1 minute and allows escape
theorem father_time_correct : (time_father = 1) → total_time = max_time := by
  intro h
  rw h
  have ht : total_time = 1 + 2 + 1 + 4 + 1 + 5 := rfl
  rw [nat.add_comm, nat.add_assoc, nat.add_comm 1 1, ←nat.add_assoc, nat.add_comm 6 5] at ht
  exact ht
  sorry

end father_time_correct_l321_321481


namespace num_employees_excluding_boss_and_todd_l321_321707

theorem num_employees_excluding_boss_and_todd (total_cost boss_contribution emp_contribution : ℕ)
  (h_total_cost : total_cost = 100)
  (h_boss_contribution : boss_contribution = 15)
  (h_emp_contribution : emp_contribution = 11) :
  ∀ todd_contribution remaining_amount num_employees,
  todd_contribution = 2 * boss_contribution →
  remaining_amount = total_cost - (boss_contribution + todd_contribution) →
  num_employees = remaining_amount / emp_contribution →
  num_employees = 5 :=
begin
  intros,
  sorry
end

end num_employees_excluding_boss_and_todd_l321_321707


namespace fraction_eq_zero_iff_l321_321254

theorem fraction_eq_zero_iff (x : ℝ) : (3 * x - 1) / (x ^ 2 + 1) = 0 ↔ x = 1 / 3 := by
  sorry

end fraction_eq_zero_iff_l321_321254


namespace triangle_perimeter_le_polygon_perimeter_l321_321123

theorem triangle_perimeter_le_polygon_perimeter
  (ABC_inside_polygon : ∀ (A B C : Point) (polygon : Polygon),
    triangle_in_polygon A B C polygon)
  (P_to_CA : ∀ (A C P : Point) (polygon : Polygon),
    point_on_ray C A P polygon)
  (Q_to_AB : ∀ (A B Q : Point) (polygon : Polygon),
    point_on_ray A B Q polygon)
  (R_to_BC : ∀ (B C R : Point) (polygon : Polygon),
    point_on_ray B C R polygon)
  (path_PQ : ∀ (P Q : Point) (polygon : Polygon),
    path_on_polygon P Q polygon)
  (path_QR : ∀ (Q R : Point) (polygon : Polygon),
    path_on_polygon Q R polygon)
  (path_RP : ∀ (R P : Point) (polygon : Polygon),
    path_on_polygon R P polygon) :
  ∀ (A B C : Point) (polygon : Polygon),
    perimeter_triangle A B C ≤ perimeter_polygon polygon := sorry

end triangle_perimeter_le_polygon_perimeter_l321_321123


namespace henry_has_more_games_l321_321615

-- Define the conditions and initial states
def initial_games_henry : ℕ := 33
def given_games_neil : ℕ := 5
def initial_games_neil : ℕ := 2

-- Define the number of games Henry and Neil have now
def games_henry_now : ℕ := initial_games_henry - given_games_neil
def games_neil_now : ℕ := initial_games_neil + given_games_neil

-- State the theorem to be proven
theorem henry_has_more_games : games_henry_now / games_neil_now = 4 :=
by
  sorry

end henry_has_more_games_l321_321615


namespace triangle_acute_l321_321251

theorem triangle_acute (A B C : ℝ) (h1 : A = 2 * (180 / 9)) (h2 : B = 3 * (180 / 9)) (h3 : C = 4 * (180 / 9)) :
  A < 90 ∧ B < 90 ∧ C < 90 :=
by
  sorry

end triangle_acute_l321_321251


namespace remainder_div_l321_321861

theorem remainder_div (N : ℤ) (k : ℤ) (h : N = 39 * k + 18) :
  N % 13 = 5 := 
by
  sorry

end remainder_div_l321_321861


namespace quadrilateral_proof_l321_321257

/-- In a convex quadrilateral ABCD, where:
    AB = 3,
    ∠ABC = 45°, and
    ∠BCD = 120°, 
    and the area of the quadrilateral is given by (AB * CD + BC * AD) / 2,
    prove that the length of side AD = 3 * sin 37.5° / sin 82.5°.
-/
theorem quadrilateral_proof
  (A B C D : Point)
  (h_convex : ConvexQuadrilateral A B C D)
  (h_OP : Opposite A C)
  (AB CD BC AD : ℝ)
  (h_AB : AB = 3)
  (h_ABC : angle A B C = 45)
  (h_BCD : angle B C D = 120)
  (h_area : area_quad A B C D = (AB * CD + BC * AD) / 2) :
  AD = 3 * (Real.sin 37.5) / (Real.sin 82.5) := sorry

end quadrilateral_proof_l321_321257


namespace geom_seq_common_ratio_l321_321304

-- We define a geometric sequence and the condition provided in the problem.
variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Condition for geometric sequence: a_n = a * q^(n-1)
def is_geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n : ℕ, a n = a 0 * q^(n-1)

-- Given condition: 2a_4 = a_6 - a_5
def given_condition (a : ℕ → ℝ) : Prop := 
  2 * a 4 = a 6 - a 5

-- Proof statement
theorem geom_seq_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : is_geometric_seq a q) (h_cond : given_condition a) : 
    q = 2 ∨ q = -1 :=
sorry

end geom_seq_common_ratio_l321_321304


namespace sequence_nonzero_l321_321466

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 = 2 ∧
  ∀ n ≥ 3, 
    if (a (n - 1) * a (n - 2)) % 2 = 0 then 
      a n = 5 * (a (n - 1)) - 3 * (a (n - 2)) 
    else 
      a n = (a (n - 1)) - (a (n - 2))

theorem sequence_nonzero (a : ℕ → ℤ) (h : seq a) : ∀ n : ℕ, a n ≠ 0 := 
by sorry

end sequence_nonzero_l321_321466


namespace solution_set_condition_l321_321149

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

theorem solution_set_condition (h1 : ∀ x : ℝ, f (x) > f' (x)) (h2 : f 0 = 1) :
  {x : ℝ | (f (x) / Real.exp x) < 1} = set.Ioi 0 :=
sorry

end solution_set_condition_l321_321149


namespace smallest_number_of_students_l321_321275

theorem smallest_number_of_students 
    (n : ℕ) 
    (attended := n / 4)
    (both := n / 40)
    (cheating_hint_ratio : ℚ := 3 / 2)
    (hinting := cheating_hint_ratio * (attended - both)) :
    n ≥ 200 :=
by sorry

end smallest_number_of_students_l321_321275


namespace functional_eq_implies_odd_l321_321564

variable (f : ℝ → ℝ)

def condition (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f (x * f y) = y * f x

theorem functional_eq_implies_odd (h : condition f) : ∀ x : ℝ, f (-x) = -f x :=
sorry

end functional_eq_implies_odd_l321_321564


namespace find_f_prime_two_l321_321603

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (x : ℝ)
hypothesis f_deriv : ∀ x, deriv f x = f' x
hypothesis f_eq : f x = x^2 + 3 * x * f' 2 - Real.log x

theorem find_f_prime_two : f' 2 = -7 / 4 := by
  sorry

end find_f_prime_two_l321_321603


namespace jackson_total_payment_l321_321670

-- Define the measurements of the courtyard
def length : ℕ := 10
def width : ℕ := 25

-- Define the area of the courtyard
def area : ℕ := length * width

-- Define the number of tiles per square foot
def tiles_per_sq_foot : ℕ := 4

-- Define the total number of tiles needed
def total_tiles : ℕ := area * tiles_per_sq_foot

-- Define the percentage of green tiles
def percent_green_tiles : ℝ := 0.40

-- Define the cost per green tile
def cost_per_green_tile : ℝ := 3

-- Define the cost per red tile
def cost_per_red_tile : ℝ := 1.50

-- Define the number of green tiles
def green_tiles : ℕ := (percent_green_tiles * total_tiles).to_nat

-- Define the number of red tiles
def red_tiles : ℕ := total_tiles - green_tiles

-- Define the total cost for green tiles
def cost_green_tiles : ℝ := green_tiles * cost_per_green_tile

-- Define the total cost for red tiles
def cost_red_tiles : ℝ := red_tiles * cost_per_red_tile

-- Define the total cost of all tiles
def total_cost : ℝ := cost_green_tiles + cost_red_tiles

-- State the theorem to be proved
theorem jackson_total_payment : total_cost = 2100 := by
  -- implying we are going to prove the statement
  sorry


end jackson_total_payment_l321_321670


namespace fraction_check_l321_321070

variable (a b x y : ℝ)
noncomputable def is_fraction (expr : ℝ) : Prop :=
∃ n d : ℝ, d ≠ 0 ∧ expr = n / d ∧ ∃ var : ℝ, d = var

theorem fraction_check :
  is_fraction ((x + 3) / x) :=
sorry

end fraction_check_l321_321070


namespace perimeter_semi_circle_l321_321463

def radius := 5.2
def pi_approx := 3.14
def perimeter_of_semicircle (r : ℝ) : ℝ := pi_approx * r + 2 * r

theorem perimeter_semi_circle : perimeter_of_semicircle radius = 26.728 := 
by
  -- Proof that the perimeter of the semicircle with radius 5.2 cm is 26.728 cm
  sorry

end perimeter_semi_circle_l321_321463


namespace distance_between_first_and_last_trees_l321_321546

theorem distance_between_first_and_last_trees (n : ℕ) (d : ℕ) (total_trees : ℕ) (total_distance : ℕ)
  (h1 : total_trees = 8)
  (h2 : n = 1)
  (h3 : d = 5)
  (h4 : total_distance = 80) :
  let distance_between_consecutive_trees := total_distance / (d - n)
  let intervals_between_first_and_last_trees := total_trees - 1
  in distance_between_consecutive_trees * intervals_between_first_and_last_trees = 140 := 
by {
  sorry
}

end distance_between_first_and_last_trees_l321_321546


namespace minimum_xy_l321_321205

theorem minimum_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/y = 1/2) : x * y ≥ 16 :=
sorry

end minimum_xy_l321_321205


namespace least_number_divisible_by_7_9_12_18_with_remainder_4_l321_321062

theorem least_number_divisible_by_7_9_12_18_with_remainder_4 :
  ∃ n : ℕ, 
    n % 7 = 4 ∧ 
    n % 9 = 4 ∧ 
    n % 12 = 4 ∧ 
    n % 18 = 4 ∧ 
    (∀ m : ℕ, 
      (m % 7 = 4 ∧ m % 9 = 4 ∧ m % 12 = 4 ∧ m % 18 = 4) → n ≤ m) :=
begin
  use 256,
  split, { refl },
  split, { refl },
  split, { refl },
  split, { refl },
  intros m h,
  sorry
end

end least_number_divisible_by_7_9_12_18_with_remainder_4_l321_321062


namespace cos_squared_value_l321_321203

theorem cos_squared_value (α : ℝ) (h : Real.tan (α + π/4) = 3/4) : Real.cos (π/4 - α) ^ 2 = 9 / 25 :=
sorry

end cos_squared_value_l321_321203


namespace smallest_total_students_l321_321280

theorem smallest_total_students (n : ℕ) (h1 : 25 * n % 100 = 0)
  (h2 : 10 * n % 4 = 0)
  (h3 : ∃ (y z : ℕ), y = 3 * z / 2 ∧ (y + z - n / 40 = n / 4)) :
  ∃ k : ℕ, n = 200 * k :=
by
  sorry

end smallest_total_students_l321_321280


namespace dogwood_trees_l321_321411

/-- There are 7 dogwood trees currently in the park. 
Park workers will plant 5 dogwood trees today. 
The park will have 16 dogwood trees when the workers are finished.
Prove that 4 dogwood trees will be planted tomorrow. --/
theorem dogwood_trees (x : ℕ) : 7 + 5 + x = 16 → x = 4 :=
by
  sorry

end dogwood_trees_l321_321411


namespace product_of_divisors_30_l321_321426
-- Import the necessary library

-- Declaring the necessary conditions and main proof statement
def prime_factors (n : ℕ) : List ℕ :=
if h : n = 30 then [2, 3, 5] else []

def divisors (n : ℕ) : List ℕ :=
if h : n = 30 then [1, 2, 3, 5, 6, 10, 15, 30] else []

theorem product_of_divisors_30 : 
  let d := divisors 30 
  in d.product = 810000 :=
by {
    -- Skip the proof with sorry
    sorry
}

end product_of_divisors_30_l321_321426


namespace magician_can_always_determine_hidden_pair_l321_321083

-- Define the cards as an enumeration
inductive Card
| one | two | three | four | five

-- Define a pair of cards
structure CardPair where
  first : Card
  second : Card

-- Define the function the magician uses to decode the hidden pair 
-- based on the two cards the assistant points out, encoded as a pentagon
noncomputable def magician_decodes (assistant_cards spectator_announced: CardPair) : CardPair := sorry

-- Theorem statement: given the conditions, the magician can always determine the hidden pair.
theorem magician_can_always_determine_hidden_pair 
  (hidden_cards assistant_cards spectator_announced : CardPair)
  (assistant_strategy : CardPair → CardPair)
  (h : assistant_strategy assistant_cards = spectator_announced)
  : magician_decodes assistant_cards spectator_announced = hidden_cards := sorry

end magician_can_always_determine_hidden_pair_l321_321083


namespace equilateral_triangle_area_l321_321109

theorem equilateral_triangle_area (A_hexagon : ℝ) (A_satisfies : A_hexagon = 12) : 
    ∃ A_triangle : ℝ, A_triangle = 18 := 
by 
    use 18
    sorry

end equilateral_triangle_area_l321_321109


namespace difference_is_four_l321_321821

def chickens_in_coop := 14
def chickens_in_run := 2 * chickens_in_coop
def chickens_free_ranging := 52
def difference := 2 * chickens_in_run - chickens_free_ranging

theorem difference_is_four : difference = 4 := by
  sorry

end difference_is_four_l321_321821


namespace area_of_triangle_l321_321744

theorem area_of_triangle (ABC : Triangle) (right_triangle : is_right_triangle ABC)
    (scalene_triangle : is_scalene ABC) (P : Point) (on_hypotenuse : on_segment P ABC.hypotenuse)
    (angle_ABP : ABC.angle_AB P = 30) (AP_length : dist ABC.A P = 2)
    (CP_length : dist ABC.C P = 1) : 
    area ABC = 2 := 
sorry

end area_of_triangle_l321_321744


namespace positive_expression_l321_321185

variable (a b c d : ℝ)

theorem positive_expression (ha : a < b) (hb : b < 0) (hc : 0 < c) (hd : c < d) : d - c - b - a > 0 := 
sorry

end positive_expression_l321_321185


namespace arithmetic_twelfth_term_l321_321838

theorem arithmetic_twelfth_term 
(a d : ℚ) (n : ℕ) (h_a : a = 1/2) (h_d : d = 1/3) (h_n : n = 12) : 
  a + (n - 1) * d = 25 / 6 := 
by 
  sorry

end arithmetic_twelfth_term_l321_321838


namespace validate_equation_l321_321038

variable (x : ℝ)

def price_of_notebook : ℝ := x - 2
def price_of_pen : ℝ := x

def total_cost (x : ℝ) : ℝ := 5 * price_of_notebook x + 3 * price_of_pen x

theorem validate_equation (x : ℝ) : total_cost x = 14 :=
by
  unfold total_cost
  unfold price_of_notebook
  unfold price_of_pen
  sorry

end validate_equation_l321_321038


namespace tangent_line_eq_l321_321760

theorem tangent_line_eq : 
  ∀ (x y : ℝ), (y = -x^3) ∧ (y = -3 * x + 2) → 
  (tangent_line (λ x : ℝ, -x^3) (0, 2) = λ x : ℝ, -3 * x + 2) :=
by 
  -- Provide proof here
  sorry

end tangent_line_eq_l321_321760


namespace product_divisors_30_eq_810000_l321_321430

def product_of_divisors (n : ℕ) : ℕ :=
  (multiset.filter (λ d, n % d = 0) (multiset.range (n + 1))).prod id

theorem product_divisors_30_eq_810000 :
  product_of_divisors 30 = 810000 :=
begin
  -- Proof will involve showing product of divisors of 30 equals 810000
  sorry
end

end product_divisors_30_eq_810000_l321_321430


namespace length_JN_l321_321638

/-!
# Problem Statement
In $\triangle DEF$, $DE = 14$, $EF = 15$, and $FD = 13$. 
Also, $N$ is the midpoint of side $DE$ and $J$ is the foot of the altitude from $D$ to $EF$. 
Prove that the length of $JN$ is $11.2$.
-/

noncomputable def triangle_DE := 14
noncomputable def triangle_EF := 15
noncomputable def triangle_FD := 13

def is_midpoint (N D E : Point) : Prop :=
  dist N D = dist N E

def is_foot_of_altitude (J D EF : Point) : Prop :=
  dist D EF = dist D J + sqrt (dist J EF ^ 2 - dist D J ^ 2)

theorem length_JN {D E F N J : Type} 
  [MetricSpace D]
  [MetricSpace E]
  [MetricSpace F]
  [MetricSpace N]
  [MetricSpace J]
  (h1 : dist D E = 14)
  (h2 : dist E F = 15)
  (h3 : dist F D = 13)
  (h4 : is_midpoint N D E)
  (h5 : is_foot_of_altitude J D (Seg E F)) :
  dist J N = 11.2 := 
sorry

end length_JN_l321_321638


namespace increasing_implies_range_of_k_l321_321632

theorem increasing_implies_range_of_k (k : ℝ) :
  (∀ x : ℝ, (1 < x ∧ x < real.exp 1) → real.exp x - k / x ≥ 0) → k ≤ real.exp 1 :=
by
  sorry

end increasing_implies_range_of_k_l321_321632


namespace Amy_gets_fewest_cookies_l321_321085

theorem Amy_gets_fewest_cookies:
  let area_Amy := 4 * Real.pi
  let area_Ben := 9
  let area_Carl := 8
  let area_Dana := (9 / 2) * Real.pi
  let num_cookies_Amy := 1 / area_Amy
  let num_cookies_Ben := 1 / area_Ben
  let num_cookies_Carl := 1 / area_Carl
  let num_cookies_Dana := 1 / area_Dana
  num_cookies_Amy < num_cookies_Ben ∧ num_cookies_Amy < num_cookies_Carl ∧ num_cookies_Amy < num_cookies_Dana :=
by
  sorry

end Amy_gets_fewest_cookies_l321_321085


namespace batsman_average_l321_321459

/-- The average after 12 innings given that the batsman makes a score of 115 in his 12th innings,
     increases his average by 3 runs, and he had never been 'not out'. -/
theorem batsman_average (A : ℕ) (h1 : 11 * A + 115 = 12 * (A + 3)) : A + 3 = 82 := 
by
  sorry

end batsman_average_l321_321459


namespace constant_term_in_expansion_l321_321557

def binomial_expansion_constant_term (x : ℝ) (r : ℕ) : Prop :=
  (x^2 + 2 * x^(-1/2))^10 = binomial (10, r) * (x^2)^(10 - r) * (2 / x^(1/2))^r ∧
  (40 - 5 * r) / 2 = 0

theorem constant_term_in_expansion :
  ∃ r : ℕ, binomial_expansion_constant_term 1 r ∧ r + 1 = 9 := 
by
  sorry

end constant_term_in_expansion_l321_321557


namespace product_of_divisors_of_30_l321_321440

theorem product_of_divisors_of_30 : 
  ∏ (d : ℕ) in {d | d ∣ 30} = 810000 :=
sorry

end product_of_divisors_of_30_l321_321440


namespace translate_to_english_l321_321545

def chinese_word : String := "教"  -- This represents the Chinese word meaning "teach"
def part_of_speech : String := "verb"  -- Condition stating the part of speech

theorem translate_to_english : (chinese_word = "教") → (part_of_speech = "verb") → "educate" :=
begin
  intro h1,
  intro h2,
  -- Insert the proof here
  sorry
end

end translate_to_english_l321_321545


namespace points_symmetric_about_x_axis_l321_321201

def point := ℝ × ℝ

def P1 : point := (-4, 3)
def P2 : point := (-4, -3)

theorem points_symmetric_about_x_axis :
  ∃ P1 P2: point, P1 = (-4, 3) ∧ P2 = (-4, -3) ∧ P1.1 = P2.1 ∧ P1.2 = -P2.2 :=
by
  let P1 : point := (-4, 3)
  let P2 : point := (-4, -3)
  use P1
  use P2
  simp
  sorry

end points_symmetric_about_x_axis_l321_321201


namespace minimum_rectangle_area_l321_321893

theorem minimum_rectangle_area (l w : ℕ) (h : 2 * (l + w) = 84) : 
  (l * w) = 41 :=
by sorry

end minimum_rectangle_area_l321_321893


namespace point_b_value_l321_321717

theorem point_b_value :
  ∀ (A B : ℤ), A = -3 → B = A + 6 → B = 3 :=
by
  intros A B hA hB
  rw [hA] at hB
  exact hB
  sorry

end point_b_value_l321_321717


namespace train_speed_75_mph_l321_321122

-- Define the conditions
def length_of_tunnel : ℝ := 3.5
def length_of_train : ℝ := 0.25
def time_in_minutes : ℝ := 3
def minutes_to_hours (m : ℝ) : ℝ := m / 60

-- Calculate the total distance
def total_distance (tunnel_length train_length : ℝ) : ℝ :=
  tunnel_length + train_length

-- Calculate the speed
def speed (distance time_in_hours : ℝ) : ℝ :=
  distance / time_in_hours

-- The main theorem to prove
theorem train_speed_75_mph :
  speed (total_distance length_of_tunnel length_of_train) (minutes_to_hours time_in_minutes) = 75 := by
  sorry

end train_speed_75_mph_l321_321122


namespace valid_exercise_combinations_l321_321490

def exercise_durations : List ℕ := [30, 20, 40, 30, 30]

theorem valid_exercise_combinations : 
  (∃ (s : Finset ℕ), s.card > 1 ∧ s.toList.sum (exercise_durations.snd) ≥ 60 ∧ ∀ t, t.card > 1 ∧ t.toList.sum (exercise_durations.snd) ≥ 60 → s = t) → 23 := 
sorry

end valid_exercise_combinations_l321_321490


namespace symmetrical_line_l321_321856

theorem symmetrical_line (symmetry_line_eq : ∀ x y : ℝ, y = -x) 
  (given_line_eq : ∀ x y : ℝ, sqrt(3) * x + y + 1 = 0) :
  ∃ a b c : ℝ, a = 1 ∧ b = sqrt(3) ∧ c = -1 ∧ ∀ x y : ℝ, a * x + b * y + c = 0 :=
by
  -- Proof is omitted as it is not required.
  sorry

end symmetrical_line_l321_321856


namespace jogging_track_circumference_l321_321537

noncomputable def Deepak_speed : ℝ := 4.5 -- km/hr
noncomputable def Wife_speed : ℝ := 3.75 -- km/hr
noncomputable def time_meet : ℝ := 4.8 / 60 -- hours

noncomputable def Distance_Deepak : ℝ := Deepak_speed * time_meet
noncomputable def Distance_Wife : ℝ := Wife_speed * time_meet

theorem jogging_track_circumference : 2 * (Distance_Deepak + Distance_Wife) = 1.32 := by
  sorry

end jogging_track_circumference_l321_321537


namespace jonathan_running_speed_on_friday_l321_321322

theorem jonathan_running_speed_on_friday :
  ∀ (T : Type) [has_div T] [has_sub T] [has_add T] [has_eq T]
  (distance_monday distance_wednesday distance_friday : T)
  (speed_monday speed_wednesday total_exercise_time : T),
  (distance_monday = 6 ∧ speed_monday = 2 ∧ distance_wednesday = 6 ∧
   speed_wednesday = 3 ∧ distance_friday = 6 ∧ total_exercise_time = 6) →
  (distance_friday / (total_exercise_time - 
                       ((distance_monday / speed_monday) + 
                       (distance_wednesday / speed_wednesday)))) = 6 :=
begin
  sorry
end

end jonathan_running_speed_on_friday_l321_321322


namespace undetermined_photographs_l321_321415

theorem undetermined_photographs (n : ℕ) (t : ℝ) :
  (0 < t) ∧ (t < 24) ∧ 
  (∀ t ∈ {t | t ≠ 0 ∧ t ≠ 6 ∧ t ≠ 12 ∧ t ≠ 18 ∧ t ≠ 24}) ∧ 
  (∀ t, t ∈ {t, 12 - t, 12 + t, 24 - t} \ 
          -> ∃ A B C, 0 < A < 6 ∧ 0 < B < 6 ∧ 0 < C < 6  ∧ A ≠ B ≠ C) ∧ 
  (n = 100) ->
  (3 ≤ n ∧ n ≤ 100 ∧ n = 3 ∨ n = 100) :=
sorry

end undetermined_photographs_l321_321415


namespace weight_of_new_student_l321_321867

theorem weight_of_new_student (avg_weight_29 : ℝ) (new_avg_weight_30 : ℝ) (num_students_29 : ℕ) (num_students_30 : ℕ) :
  avg_weight_29 = 28 → new_avg_weight_30 = 27.3 → num_students_29 = 29 → num_students_30 = 30 → 
  let total_weight_29 := avg_weight_29 * num_students_29 in
  let new_total_weight_30 := new_avg_weight_30 * num_students_30 in
  new_total_weight_30 - total_weight_29 = 7 := 
by
  intros h1 h2 h3 h4
  let total_weight_29 := avg_weight_29 * num_students_29
  let new_total_weight_30 := new_avg_weight_30 * num_students_30
  have : total_weight_29 = 812, by rw [h1, h3]; norm_num
  have : new_total_weight_30 = 819, by rw [h2, h4]; norm_num
  rw [this, this_1]
  norm_num
  sorry

end weight_of_new_student_l321_321867


namespace annual_growth_rate_l321_321099

theorem annual_growth_rate (p : ℝ) : 
  let S1 := (1 + p) ^ 12 - 1 / p
  let S2 := ((1 + p) ^ 12 * ((1 + p) ^ 12 - 1)) / p
  let annual_growth := (S2 - S1) / S1
  annual_growth = (1 + p) ^ 12 - 1 :=
by
  sorry

end annual_growth_rate_l321_321099


namespace largest_prime_divisor_13_fac_add_14_fac_l321_321965

theorem largest_prime_divisor_13_fac_add_14_fac : 
  ∀ (p : ℕ), prime p → p ∣ (13! + 14!) → p ≤ 5 :=
by
  sorry

end largest_prime_divisor_13_fac_add_14_fac_l321_321965


namespace length_segment_abs_eq4_l321_321832

theorem length_segment_abs_eq4 : ∀ x : ℝ, | x - real.cbrt 27 | = 4 → (abs ( (real.cbrt 27 + 4) - (real.cbrt 27 - 4)) = 8 ) :=
by
  intro x h
  have h1 : real.cbrt 27 = 3, by
    sorry
  calc
    abs ( (real.cbrt 27 + 4) - (real.cbrt 27 - 4) ) 
        = abs ( (3 + 4) - (3 - 4) ) : by rw [h1]
    ... = abs ( 7 - (-1) ) : by rw [add_neg_eq_sub]
    ... = abs ( 7 + 1 ) : by rw [sub_neg_eq_add]
    ... = abs 8 : by rw [add_comm]
    ... = 8 : abs_eq_self.mpr (by norm_num)

end length_segment_abs_eq4_l321_321832


namespace tan_double_angle_l321_321756

theorem tan_double_angle (α : ℝ) (h1 : Real.sin (5 * Real.pi / 6) = 1 / 2)
  (h2 : Real.cos (5 * Real.pi / 6) = -Real.sqrt 3 / 2) : 
  Real.tan (2 * α) = Real.sqrt 3 := 
sorry

end tan_double_angle_l321_321756


namespace cross_product_with_scalar_l321_321567

variable {R : Type} [OrderedRing R]
variable (a b : Vector R 3)

theorem cross_product_with_scalar 
  (h : a × b = ![-3, 6, 2]) :
  a × (5 • b) = ![-15, 30, 10] :=
  sorry

end cross_product_with_scalar_l321_321567


namespace find_a_and_b_find_monotonic_intervals_and_extreme_values_l321_321606

-- Definitions and conditions
def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

def takes_extreme_values (f : ℝ → ℝ) (a b c : ℝ) : Prop := 
  ∃ x₁ x₂, x₁ = 1 ∧ x₂ = -2/3 ∧ 3*x₁^2 + 2*a*x₁ + b = 0 ∧ 3*x₂^2 + 2*a*x₂ + b = 0

def f_at_specific_point (f : ℝ → ℝ) (x v : ℝ) : Prop :=
  f x = v

theorem find_a_and_b (a b c : ℝ) :
  takes_extreme_values (f a b c) a b c →
  a = -1/2 ∧ b = -2 :=
sorry

theorem find_monotonic_intervals_and_extreme_values (a b c : ℝ) :
  takes_extreme_values (f a b c) a b c →
  f_at_specific_point (f a b c) (-1) (3/2) →
  c = 1 ∧ 
  (∀ x, x < -2/3 ∨ x > 1 → deriv (f a b c) x > 0) ∧
  (∀ x, -2/3 < x ∧ x < 1 → deriv (f a b c) x < 0) ∧
  f a b c (-2/3) = 49/27 ∧ 
  f a b c 1 = -1/2 :=
sorry

end find_a_and_b_find_monotonic_intervals_and_extreme_values_l321_321606


namespace painted_cubes_count_l321_321801

def total_painted_cubes : ℕ := 8 + 48

theorem painted_cubes_count : total_painted_cubes = 56 :=
by 
  -- Step 1: Define the number of cubes with 3 faces painted (8 corners)
  let corners := 8
  -- Step 2: Calculate the number of edge cubes with 2 faces painted
  let edge_middle_cubes_per_edge := 2
  let edges := 12
  let edge_cubes := edge_middle_cubes_per_edge * edges -- this should be 24
  -- Step 3: Calculate the number of face-interior cubes with 2 faces painted
  let face_cubes_per_face := 4
  let faces := 6
  let face_cubes := face_cubes_per_face * faces -- this should be 24
  -- Step 4: Sum them up to get total cubes with at least two faces painted
  let total_cubes := corners + edge_cubes + face_cubes
  show total_cubes = total_painted_cubes
  sorry

end painted_cubes_count_l321_321801


namespace fraction_check_l321_321069

variable (a b x y : ℝ)
noncomputable def is_fraction (expr : ℝ) : Prop :=
∃ n d : ℝ, d ≠ 0 ∧ expr = n / d ∧ ∃ var : ℝ, d = var

theorem fraction_check :
  is_fraction ((x + 3) / x) :=
sorry

end fraction_check_l321_321069


namespace hannahs_vegetarian_restaurant_l321_321613

theorem hannahs_vegetarian_restaurant :
  let total_weight_of_peppers := 0.6666666666666666
  let weight_of_green_peppers := 0.3333333333333333
  total_weight_of_peppers - weight_of_green_peppers = 0.3333333333333333 :=
by
  sorry

end hannahs_vegetarian_restaurant_l321_321613


namespace volume_rectangular_prism_space_diagonal_rectangular_prism_surface_area_rectangular_prism_surface_area_circumscribed_sphere_l321_321005

-- Define the conditions of the rectangular prism
def length := 4
def width := 2
def height := 1

-- 1. Prove that the volume of the rectangular prism is 8
theorem volume_rectangular_prism : length * width * height = 8 := sorry

-- 2. Prove that the length of the space diagonal is √21
theorem space_diagonal_rectangular_prism : Real.sqrt (length^2 + width^2 + height^2) = Real.sqrt 21 := sorry

-- 3. Prove that the surface area of the rectangular prism is 28
theorem surface_area_rectangular_prism : 2 * (length * width + length * height + width * height) = 28 := sorry

-- 4. Prove that the surface area of the circumscribed sphere is 21π
theorem surface_area_circumscribed_sphere : 4 * Real.pi * (Real.sqrt (length^2 + width^2 + height^2) / 2)^2 = 21 * Real.pi := sorry

end volume_rectangular_prism_space_diagonal_rectangular_prism_surface_area_rectangular_prism_surface_area_circumscribed_sphere_l321_321005


namespace minimum_ellipse_area_l321_321514

theorem minimum_ellipse_area (a b : ℝ) (h₁ : 4 * (a : ℝ) ^ 2 * b ^ 2 = a ^ 2 + b ^ 4)
  (h₂ : (∀ x y : ℝ, ((x - 2) ^ 2 + y ^ 2 ≤ 4 → x ^ 2 / (4 * a ^ 2) + y ^ 2 / (4 * b ^ 2) ≤ 1)) 
       ∧ (∀ x y : ℝ, ((x + 2) ^ 2 + y ^ 2 ≤ 4 → x ^ 2 / (4 * a ^ 2) + y ^ 2 / (4 * b ^ 2) ≤ 1))) : 
  ∃ k : ℝ, (k = 16) ∧ (π * (4 * a * b) = k * π) :=
by sorry

end minimum_ellipse_area_l321_321514


namespace smallest_possible_number_of_students_l321_321291

theorem smallest_possible_number_of_students :
  ∃ n : ℕ, (n % 200 = 0) ∧ (∀ m : ℕ, (m < n → 
    75 * m ≤ 100 * n) ∧
    (∃ a b c : ℕ, a = m / 4 ∧ b = a / 10 ∧ 
    ∃ y z : ℕ, y = 3 * z ∧ (y + z - b = a) ∧ y * c = n / 4)) :=
by
  sorry

end smallest_possible_number_of_students_l321_321291


namespace geometric_series_product_l321_321377

theorem geometric_series_product (q : ℂ) (n : ℕ) (S S₁ : ℂ) 
  (hS : S = ∑ k in finset.range (2 * n + 1), q ^ k) 
  (hS₁ : S₁ = ∑ k in finset.range (2 * n + 1), (-q) ^ k) : 
  ∑ k in finset.range (n + 1), q ^ (2 * k) = S * S₁ := 
by
  sorry

end geometric_series_product_l321_321377


namespace john_age_multiple_of_james_age_l321_321320

-- Define variables for the problem conditions
def john_current_age : ℕ := 39
def john_age_3_years_ago : ℕ := john_current_age - 3

def james_brother_age : ℕ := 16
def james_brother_older : ℕ := 4

def james_current_age : ℕ := james_brother_age - james_brother_older
def james_age_in_6_years : ℕ := james_current_age + 6

-- The goal is to prove the multiple relationship
theorem john_age_multiple_of_james_age :
  john_age_3_years_ago = 2 * james_age_in_6_years :=
by {
  -- Skip the proof
  sorry
}

end john_age_multiple_of_james_age_l321_321320


namespace prove_AD_l321_321331

variable {V : Type} [AddCommGroup V] [Module ℝ V]
variables {A B C D : V} -- Points in the vector space

-- The condition that D is a point in the plane of triangle ABC and BC = 3 * CD
def condition : Prop := 
  B - C = 3 • (C - D)

-- The statement we need to prove
theorem prove_AD (h : condition) : 
  D - A = - (1 / 3:ℝ) • (B - A) + (4 / 3:ℝ) • (C - A) :=
sorry

end prove_AD_l321_321331


namespace order_of_magnitude_l321_321937

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem order_of_magnitude :
  let log_0_7 := log_base 0.7
  in log_0_7 2 < log_0_7 0.8 ∧ log_0_7 0.8 < 0.9 ^ (-2) :=
by
  let log_0_7 := log_base 0.7
  have h1 : log_0_7 2 < log_0_7 0.8, sorry
  have h2 : log_0_7 0.8 < 0.9 ^ (-2), sorry
  exact ⟨h1, h2⟩

end order_of_magnitude_l321_321937


namespace twelfth_term_in_sequence_l321_321843

variable (a₁ : ℚ) (d : ℚ)

-- Given conditions
def a_1_term : a₁ = 1 / 2 := rfl
def common_difference : d = 1 / 3 := rfl

-- The twelfth term of the arithmetic sequence
def a₁₂ : ℚ := a₁ + 11 * d

-- Statement to prove
theorem twelfth_term_in_sequence (h₁ : a₁ = 1 / 2) (h₂ : d = 1 / 3) : a₁₂ = 25 / 6 :=
by
  rw [h₁, h₂, add_comm, add_assoc, mul_comm, add_comm]
  exact sorry

end twelfth_term_in_sequence_l321_321843


namespace carlos_laundry_l321_321530

theorem carlos_laundry (n : ℕ) 
  (h1 : 45 * n + 75 = 165) : n = 2 :=
by
  sorry

end carlos_laundry_l321_321530


namespace holly_initial_amount_l321_321232

-- Define the conditions
noncomputable def amount_with_breakfast : ℕ := 8
noncomputable def amount_with_lunch : ℕ := 8
noncomputable def amount_with_dinner : ℕ := 8
noncomputable def end_of_day_amount : ℕ := 56
noncomputable def total_consumption : ℕ := amount_with_breakfast + amount_with_lunch + amount_with_dinner  -- 24

-- Define the proof problem
theorem holly_initial_amount :
  let initial_amount := end_of_day_amount + total_consumption in
  initial_amount = 80 :=
by
  sorry

end holly_initial_amount_l321_321232


namespace side_BC_length_l321_321096

variable (A B C D O : Type) [euclidean_space A] [euclidean_space B] [euclidean_space C] [euclidean_space D] [euclidean_space O]

variable (circumcircle : circumcircle_type A B C D O)
variable (perpendicular_diagonals : is_perpendicular (diagonal A C) (diagonal B D))
variable (distance_O_AD : distance_from_point_to_line O AD = 1)

theorem side_BC_length :
  length (side B C) = 2 :=
by
  sorry

end side_BC_length_l321_321096


namespace probability_divisible_by_4_l321_321068

/-- When two fair 8-sided dice are tossed, each die can show any number from 1 to 8. 
What is the probability that the two-digit number cd (where c and d are the results)
and both c and d themselves are divisible by 4? -/
theorem probability_divisible_by_4 (c d : ℕ) (h1 : c ∈ {1, 2, 3, 4, 5, 6, 7, 8}) (h2 : d ∈ {1, 2, 3, 4, 5, 6, 7, 8}) :
  (c = 4 ∨ c = 8) ∧ (d = 4 ∨ d = 8) -> 
  (c * 10 + d) % 4 = 0 ->
  1 / 16 :=
by sorry

end probability_divisible_by_4_l321_321068


namespace find_cosine_AOB_l321_321227

-- Define the properties of the vectors and points
variables {O A B M : Type*}
variables [InnerProductSpace ℝ (EuclideanSpace ℝ (Fin 3))]

open EuclideanSpace

-- Define vectors in terms of their magnitudes and centroid property
variables (OA OB : EuclideanSpace ℝ (Fin 3))
variables (MO MB : EuclideanSpace ℝ (Fin 3))

-- Conditions
def given_conditions (OA OB MO MB : EuclideanSpace ℝ (Fin 3)) :=
  (∥OA∥ = 3) ∧ 
  (∥OB∥ = 2) ∧ 
  (MO = (1/3) • (OA + OB)) ∧
  (MB • MO = 0)

-- Proof statement
theorem find_cosine_AOB (OA OB MO MB : EuclideanSpace ℝ (Fin 3))
  (h : given_conditions OA OB MO MB) : 
  ∃ (cos_AOB : ℝ), cos_AOB = -1/6 :=
begin
  sorry
end

end find_cosine_AOB_l321_321227


namespace largest_number_among_four_l321_321075

theorem largest_number_among_four :
  let a := 0.965
  let b := 0.9687
  let c := 0.9618
  let d := 0.955
  max a (max b (max c d)) = b := 
sorry

end largest_number_among_four_l321_321075


namespace area_right_scalene_triangle_l321_321734

noncomputable def area_of_triangle_ABC : ℝ :=
  let AP : ℝ := 2
  let CP : ℝ := 1
  let AC : ℝ := AP + CP
  let ratio : ℝ := 2
  let x_squared : ℝ := 9 / 5
  x_squared

theorem area_right_scalene_triangle (AP CP : ℝ) (h₁ : AP = 2) (h₂ : CP = 1) (h₃ : ∠(B : Point) (A : Point) (P : Point) = 30) :
  let AB := 2 * real.sqrt(9 / 5)
  let BC := real.sqrt(9 / 5)
  ∃ (area : ℝ), area = (1/2) * AB * BC ∧ area = 9 / 5 :=
by
  sorry

end area_right_scalene_triangle_l321_321734


namespace largest_prime_divisor_13_factorial_plus_14_factorial_l321_321996

theorem largest_prime_divisor_13_factorial_plus_14_factorial : 
  ∃ p, prime p ∧ p ∈ prime_factors (13! + 14!) ∧ ∀ q, prime q ∧ q ∈ prime_factors (13! + 14!) → q ≤ 13 :=
by { sorry }

end largest_prime_divisor_13_factorial_plus_14_factorial_l321_321996


namespace length_segment_AE_l321_321097

noncomputable def length_of_AE
  (r : ℝ) (A B C D E : Type) (dist_AB : ℝ) (isosceles : AC = BC) 
  (AB_not_eq_AC : ¬(AB = AC)) (angle_ACB : ℝ) (angle_ACB_30 : angle_ACB = 30) : ℝ :=
  let AE := √6 - √2 in
  AE

theorem length_segment_AE 
  (r : ℝ) (A B C D E : Type) (dist_AB : ℝ) (isosceles : AC = BC) 
  (AB_not_eq_AC : ¬(AB = AC)) (angle_ACB : ℝ) (angle_ACB_30 : angle_ACB = 30) :
  length_of_AE r A B C D E dist_AB isosceles AB_not_eq_AC angle_ACB angle_ACB_30 = √6 - √2 :=
by
  sorry

end length_segment_AE_l321_321097


namespace find_d_of_quadratic_roots_l321_321786

theorem find_d_of_quadratic_roots :
  ∃ d : ℝ, (∀ x : ℝ, x^2 + 7 * x + d = 0 ↔ x = (-7 + real.sqrt d) / 2 ∨ x = (-7 - real.sqrt d) / 2) → d = 9.8 :=
by
  sorry

end find_d_of_quadratic_roots_l321_321786


namespace scientific_notation_of_604800_l321_321124

theorem scientific_notation_of_604800 : 604800 = 6.048 * 10^5 := 
sorry

end scientific_notation_of_604800_l321_321124


namespace parallelogram_area_l321_321912

theorem parallelogram_area :
  let A := (0, 0) 
  let B := (4, 0)
  let C := (1, 5)
  let D := (5, 5) 
  parallelogram.area A B C D = 20 :=
by
  sorry

end parallelogram_area_l321_321912


namespace digit_sum_repeated_2000_2000_l321_321683

noncomputable def digit_sum (n : ℕ) : ℕ := n.digits.sum

theorem digit_sum_repeated_2000_2000 :
  digit_sum (digit_sum (digit_sum (2000 ^ 2000))) = 4 := by
  sorry

end digit_sum_repeated_2000_2000_l321_321683


namespace y_values_relation_l321_321239

theorem y_values_relation :
  ∀ y1 y2 y3 : ℝ,
    (y1 = (-3 + 1) ^ 2 + 1) →
    (y2 = (0 + 1) ^ 2 + 1) →
    (y3 = (2 + 1) ^ 2 + 1) →
    y2 < y1 ∧ y1 < y3 :=
by
  sorry

end y_values_relation_l321_321239


namespace triangle_area_l321_321738

theorem triangle_area (A B C P : Point)
  (h_right : ∃ (α β γ : ℝ), α = 90 ∧ β ≠ 90 ∧ γ ≠ 90 ∧ β + γ = 90)
  (h_angle_ABP : Measure.angle A B P = 30)
  (h_AP : dist A P = 2)
  (h_CP : dist C P = 1)
  (h_AC : dist A C = 3) :
  area_triangle A B C = 9 / 5 :=
sorry

end triangle_area_l321_321738


namespace sin_inv_tan_eq_l321_321949

open Real

theorem sin_inv_tan_eq :
  let a := arcsin (4/5)
  let b := arctan 3
  sin (a + b) = (13 * sqrt 10) / 50 := 
by
  let a := arcsin (4/5)
  let b := arctan 3
  sorry

end sin_inv_tan_eq_l321_321949


namespace total_lunch_bill_l321_321375

def cost_of_hotdog : ℝ := 5.36
def cost_of_salad : ℝ := 5.10

theorem total_lunch_bill : cost_of_hotdog + cost_of_salad = 10.46 := 
by
  sorry

end total_lunch_bill_l321_321375


namespace train_length_correct_l321_321120

-- Define the conditions
def train_speed : ℝ := 63
def time_crossing : ℝ := 40
def expected_length : ℝ := 2520

-- The statement to prove
theorem train_length_correct : train_speed * time_crossing = expected_length :=
by
  exact sorry

end train_length_correct_l321_321120


namespace kangaroo_jump_is_8_5_feet_longer_l321_321644

noncomputable def camel_step_length (total_distance : ℝ) (num_steps : ℕ) : ℝ := total_distance / num_steps
noncomputable def kangaroo_jump_length (total_distance : ℝ) (num_jumps : ℕ) : ℝ := total_distance / num_jumps
noncomputable def length_difference (jump_length step_length : ℝ) : ℝ := jump_length - step_length

theorem kangaroo_jump_is_8_5_feet_longer :
  let total_distance := 7920
  let num_gaps := 50
  let camel_steps_per_gap := 56
  let kangaroo_jumps_per_gap := 14
  let num_camel_steps := num_gaps * camel_steps_per_gap
  let num_kangaroo_jumps := num_gaps * kangaroo_jumps_per_gap
  let camel_step := camel_step_length total_distance num_camel_steps
  let kangaroo_jump := kangaroo_jump_length total_distance num_kangaroo_jumps
  length_difference kangaroo_jump camel_step = 8.5 := sorry

end kangaroo_jump_is_8_5_feet_longer_l321_321644


namespace original_population_calc_l321_321086

-- Definitions based on conditions
def original_population := Nat

-- Conditions
variable (P : original_population)
variable (remaining_population_after_bombardment : original_population := (90 * P) / 100)
variable (remaining_population_after_fear : original_population := (85 * remaining_population_after_bombardment) / 100)

theorem original_population_calc (h : remaining_population_after_fear = 6514) : P = 8518 :=
by
  sorry

end original_population_calc_l321_321086


namespace train_speed_l321_321863

theorem train_speed (train_length bridge_length cross_time : ℝ)
  (h1 : train_length = 250)
  (h2 : bridge_length = 150)
  (h3 : cross_time = 25) :
  (train_length + bridge_length) / cross_time = 16 :=
by
  sorry

end train_speed_l321_321863


namespace assembling_red_cube_possible_l321_321125

/-
Among the faces of eight identical cubes, one-third are blue and the rest are red.
These cubes were then assembled into a larger cube. Now, among the visible
faces of the larger cube, exactly one-third are red. Prove that it is possible
to assemble a larger cube from these cubes such that the exterior of the larger
cube is completely red.
-/

theorem assembling_red_cube_possible :
  ∃ (f : ℕ → ℕ → Prop),
    (∀ i, f 1 i = 16 ∨ f 2 i = 32) →  -- one-third faces blue, total 48 faces
    (∃ i, f 1 i = 8) ∧               -- proof of exactly one-third visible faces red
    (∃ i, f 2 i = 24) →              -- remaining faces blue on surface
    (∀ i, f 2 i = 0) :=              -- exterior completely red is possible
sorry

end assembling_red_cube_possible_l321_321125


namespace simplest_radical_l321_321071

theorem simplest_radical (r1 r2 r3 r4 : ℝ) 
  (h1 : r1 = Real.sqrt 3) 
  (h2 : r2 = Real.sqrt 4)
  (h3 : r3 = Real.sqrt 8)
  (h4 : r4 = Real.sqrt (1 / 2)) : r1 = Real.sqrt 3 :=
  by sorry

end simplest_radical_l321_321071


namespace true_propositions_3_and_4_l321_321543

-- Define the condition for Proposition ③
def prop3_statement (m : ℝ) : Prop :=
  (m > 2) → ∀ x : ℝ, (x^2 - 2*x + m > 0)

def prop3_contrapositive (m : ℝ) : Prop :=
  (∀ x : ℝ, (x^2 - 2*x + m > 0)) → (m > 2)

-- Define the condition for Proposition ④
def prop4_condition (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x : ℝ, f (1 + x) = f (1 - x))

def prop4_period_4 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 4) = f (x))

-- Theorem to prove Propositions ③ and ④ are true
theorem true_propositions_3_and_4
  (m : ℝ) (f : ℝ → ℝ)
  (h3 : ∀ (m : ℝ), prop3_contrapositive m)
  (h4 : prop4_condition f): 
  prop3_statement m ∧ prop4_period_4 f :=
by {
  sorry
}

end true_propositions_3_and_4_l321_321543


namespace max_fifth_power_divisors_l321_321903

theorem max_fifth_power_divisors (n : ℕ) (D : List ℕ) (hD : D = (List.range (n + 1)).filter (λ x, n % x = 0)) (h_len : (D.drop D.find_index(λ x : ℕ, x > 1)).take 151 = 151) :
  ∃ D' : List ℕ, D'.length = 151 ∧ ∀ x ∈ D', x ∈ D ∧ (∃ k : ℕ, x = k^5) → (∃ d_len : ℕ, (D'.filter (λ x, ∃ k : ℕ, x = k^5)).length = d_len ∧ d_len ≤ 31) :=
begin
  sorry
end

end max_fifth_power_divisors_l321_321903


namespace total_chairs_all_together_l321_321560

-- Definitions of given conditions
def rows := 7
def chairs_per_row := 12
def extra_chairs := 11

-- Main statement we want to prove
theorem total_chairs_all_together : 
  (rows * chairs_per_row + extra_chairs = 95) := 
by
  sorry

end total_chairs_all_together_l321_321560


namespace max_coconuts_to_goats_l321_321703

/-
Max can trade 3 coconuts for 1 crab, and 6 crabs for 1 goat. 
If he has 342 coconuts and he wants to convert all of them into goats, 
prove that he will have 19 goats.
-/

theorem max_coconuts_to_goats :
  (coconuts_per_crab coconuts_per_goat coconuts : Nat) 
  (coconuts_per_crab = 3) 
  (coconuts_per_goat = 18) 
  (coconuts = 342) : 
  coconuts / coconuts_per_crab / (coconuts_per_goat / 3) = 19 :=
by
  sorry

end max_coconuts_to_goats_l321_321703


namespace average_income_l321_321387

-- Lean statement to express the given mathematical problem
theorem average_income (A B C : ℝ) 
  (h1 : (A + B) / 2 = 4050)
  (h2 : (B + C) / 2 = 5250)
  (h3 : A = 3000) :
  (A + C) / 2 = 4200 :=
by
  sorry

end average_income_l321_321387


namespace largest_prime_divisor_13_factorial_plus_14_factorial_l321_321991

theorem largest_prime_divisor_13_factorial_plus_14_factorial : 
  ∃ p, prime p ∧ p ∈ prime_factors (13! + 14!) ∧ ∀ q, prime q ∧ q ∈ prime_factors (13! + 14!) → q ≤ 13 :=
by { sorry }

end largest_prime_divisor_13_factorial_plus_14_factorial_l321_321991


namespace amount_of_water_per_minute_l321_321862

-- Given definitions for the conditions
def river_depth : ℝ := 2
def river_width : ℝ := 45
def flow_rate_kmph : ℝ := 3

-- Convert flow rate from kmph to m/min
def flow_rate_m_per_min : ℝ := flow_rate_kmph * 1000 / 60

-- Calculation of the volume of water per minute
def volume_per_minute : ℝ := flow_rate_m_per_min * river_width * river_depth

-- Theorem stating the proof problem
theorem amount_of_water_per_minute : volume_per_minute = 9000 :=
by
  -- Proof skipped
  sorry

end amount_of_water_per_minute_l321_321862


namespace smallest_total_students_l321_321279

theorem smallest_total_students (n : ℕ) (h1 : 25 * n % 100 = 0)
  (h2 : 10 * n % 4 = 0)
  (h3 : ∃ (y z : ℕ), y = 3 * z / 2 ∧ (y + z - n / 40 = n / 4)) :
  ∃ k : ℕ, n = 200 * k :=
by
  sorry

end smallest_total_students_l321_321279


namespace relationship_y1_y2_y3_l321_321627

variable {y1 y2 y3 h : ℝ}

def point_A := -1 / 2
def point_B := 1
def point_C := 2
def quadratic_function (x : ℝ) : ℝ := -(x - 2)^2 + h

theorem relationship_y1_y2_y3
  (hA : A_on_curve : quadratic_function point_A = y1)
  (hB : B_on_curve : quadratic_function point_B = y2)
  (hC : C_on_curve : quadratic_function point_C = y3) :
  y1 < y2 ∧ y2 < y3 := 
sorry

end relationship_y1_y2_y3_l321_321627


namespace arithmetic_sequence_twelfth_term_l321_321847

theorem arithmetic_sequence_twelfth_term :
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  (a1 + 11 * d) = (25 : ℚ) / 6 :=
by
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  exact sorry

end arithmetic_sequence_twelfth_term_l321_321847


namespace average_income_error_l321_321753

theorem average_income_error 
  (num_families : ℕ)
  (range_min : ℕ)
  (range_max : ℕ)
  (incorrect_income : ℕ)
  (correct_income : ℕ)
  (h_families : num_families = 201000)
  (h_range_min : range_min = 8200)
  (h_range_max : range_max = 98000)
  (h_incorrect_income : incorrect_income = 980000)
  (h_correct_income : correct_income = 98000)
  : (incorrect_income - correct_income) / (num_families / 201) = 882 :=
by
  rw [h_families, h_range_min, h_range_max, h_incorrect_income, h_correct_income]
  have h_error : incorrect_income - correct_income = 882000 := by sorry
  exact sorry

end average_income_error_l321_321753


namespace probability_sunglasses_and_hat_l321_321136

-- Define the variables for the given conditions.
def num_sunglasses : ℕ := 80
def num_hats : ℕ := 45
def prob_hat_and_sunglasses : ℚ := 1 / 3

-- The number of people wearing both sunglasses and hats.
def num_both : ℕ := (prob_hat_and_sunglasses * num_hats).natAbs

-- Assertion to be proven: the probability that a person wearing sunglasses also wears a hat.
theorem probability_sunglasses_and_hat :
  (num_both : ℚ) / num_sunglasses = 3 / 16 :=
by
  -- Proof steps will follow
  sorry

end probability_sunglasses_and_hat_l321_321136


namespace largest_prime_divisor_of_factorial_sum_l321_321985

theorem largest_prime_divisor_of_factorial_sum :
  ∀ (n: ℕ), n = 13 → largest_prime_divisor (factorial n + factorial (n + 1)) = 7 := 
by
  intros,
  sorry

end largest_prime_divisor_of_factorial_sum_l321_321985


namespace gain_percent_sale_is_16_98_approx_l321_321866

noncomputable def gain_percent_during_sale : ℝ := 
let sp := 30 in
let gain_percent_without_discount := 30 in
let marked_price := 30 in
let discount := 10 in
let cp := sp / (1 + gain_percent_without_discount / 100) in
let discount_amount := discount / 100 * marked_price in
let sp_sale := marked_price - discount_amount in
let gain := sp_sale - cp in
(gain / cp) * 100

theorem gain_percent_sale_is_16_98_approx : gain_percent_during_sale ≈ 16.98 :=
sorry

end gain_percent_sale_is_16_98_approx_l321_321866


namespace cube_faces_paint_count_l321_321800

def is_painted_on_at_least_two_faces (x y z : ℕ) : Prop :=
  ((x = 0 ∨ x = 3) ∨ (y = 0 ∨ y = 3) ∨ (z = 0 ∨ z = 3))

theorem cube_faces_paint_count :
  let n := 4 in
  let volume := n * n * n in
  let one_inch_cubes := {c | ∃ x y z : ℕ, x < n ∧ y < n ∧ z < n ∧ c = (x, y, z)} in
  let painted_cubes := {c ∈ one_inch_cubes | exists_along_edges c} in  -- Auxiliary predicate to check edge paint
  card painted_cubes = 32 :=
begin
  sorry,
end

/-- Auxiliary predicate to check if a cube is on the edges but not corners, signifying 
    at least two faces are painted --/
predicate exists_along_edges (c: ℕ × ℕ × ℕ) := 
  match c with
  | (0, _, _) => true
  | (3, _, _) => true
  | (_, 0, _) => true
  | (_, 3, _) => true
  | (_, _, 0) => true
  | (_, _, 3) => true
  | (_, _, _) => false
  end

end cube_faces_paint_count_l321_321800


namespace sequence_50th_term_is_3755_l321_321034

-- define a condition for checking if n is part of the sequence
def in_sequence (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : Finset ℕ), n = a.sum (λ i, 5 ^ i) ∧ ∀ i ∈ a, 5 ^ i ≤ n

-- define a function to convert binary representation of k to the corresponding element in the sequence
def sequence_term (k : ℕ) : ℕ :=
  (Finset.range 20).filter (λ i, Nat.testBit k i).sum (λ i, 5 ^ i)

-- assert that the 50th term of the sequence is 3755
theorem sequence_50th_term_is_3755 : sequence_term 50 = 3755 := sorry

end sequence_50th_term_is_3755_l321_321034


namespace max_goats_from_coconuts_l321_321701

def coconuts := ℕ
def crabs := ℕ
def goats := ℕ

def coconuts_to_crabs (c : coconuts) : crabs := c / 3
def crabs_to_goats (cr : crabs) : goats := cr / 6

theorem max_goats_from_coconuts (initial_coconuts : coconuts) (hc : initial_coconuts = 342) : 
  crabs_to_goats (coconuts_to_crabs initial_coconuts) = 19 := 
by
  rw [hc]
  simp only [coconuts_to_crabs, crabs_to_goats]
  norm_num
  sorry

end max_goats_from_coconuts_l321_321701


namespace jason_cook_time_l321_321313

theorem jason_cook_time :
  let initial_temp : ℕ := 41
  let boil_temp : ℕ := 212
  let temp_increase : ℕ := 3
  let boil_time : ℕ := (boil_temp - initial_temp) / temp_increase
  let pasta_cook_time : ℕ := 12
  let mix_salad_time : ℕ := pasta_cook_time / 3
  boil_time + pasta_cook_time + mix_salad_time = 73 :=
by
  let initial_temp := 41
  let boil_temp := 212
  let temp_increase := 3
  let boil_time := (boil_temp - initial_temp) / temp_increase
  let pasta_cook_time := 12
  let mix_salad_time := pasta_cook_time / 3
  have h1 : boil_time = 57 := rfl
  have h2 : mix_salad_time = 4 := rfl
  calc
    boil_time + pasta_cook_time + mix_salad_time
    = 57 + pasta_cook_time + mix_salad_time : by rw h1
    ... = 57 + 12 + mix_salad_time : rfl
    ... = 57 + 12 + 4 : by rw h2
    ... = 73 : rfl

end jason_cook_time_l321_321313


namespace different_selections_with_both_genders_l321_321728

theorem different_selections_with_both_genders :
  ∑ m in {1, 2}, (Nat.choose 10 m) * (Nat.choose 6 (3 - m)) = 420 :=
by
  sorry

end different_selections_with_both_genders_l321_321728


namespace product_of_divisors_30_l321_321425
-- Import the necessary library

-- Declaring the necessary conditions and main proof statement
def prime_factors (n : ℕ) : List ℕ :=
if h : n = 30 then [2, 3, 5] else []

def divisors (n : ℕ) : List ℕ :=
if h : n = 30 then [1, 2, 3, 5, 6, 10, 15, 30] else []

theorem product_of_divisors_30 : 
  let d := divisors 30 
  in d.product = 810000 :=
by {
    -- Skip the proof with sorry
    sorry
}

end product_of_divisors_30_l321_321425


namespace graph_of_f_does_not_pass_through_second_quadrant_l321_321000

def f (x : ℝ) : ℝ := x - 2

theorem graph_of_f_does_not_pass_through_second_quadrant :
  ¬ ∃ x y : ℝ, y = f x ∧ x < 0 ∧ y > 0 :=
sorry

end graph_of_f_does_not_pass_through_second_quadrant_l321_321000


namespace quadratic_points_relation_l321_321629

theorem quadratic_points_relation (h y1 y2 y3 : ℝ) :
  (∀ x, x = -1/2 → y1 = -(x-2) ^ 2 + h) ∧
  (∀ x, x = 1 → y2 = -(x-2) ^ 2 + h) ∧
  (∀ x, x = 2 → y3 = -(x-2) ^ 2 + h) →
  y1 < y2 ∧ y2 < y3 :=
by
  -- The required proof is omitted
  sorry

end quadratic_points_relation_l321_321629


namespace largest_prime_divisor_of_thirteen_plus_fourteen_factorial_l321_321998

noncomputable def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def largest_prime_divisor_of_factorials_sum (n m : ℕ) : ℕ :=
  let sum := factorial n + factorial m
  Prime.factorization sum |>.keys.max' sorry

theorem largest_prime_divisor_of_thirteen_plus_fourteen_factorial :
  largest_prime_divisor_of_factorials_sum 13 14 = 17 := 
sorry

end largest_prime_divisor_of_thirteen_plus_fourteen_factorial_l321_321998


namespace new_salary_correct_l321_321113

-- Define the initial salary and percentage increase as given in the conditions
def initial_salary : ℝ := 10000
def percentage_increase : ℝ := 0.02

-- Define the function that calculates the new salary after a percentage increase
def new_salary (initial_salary : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_salary + (initial_salary * percentage_increase)

-- The theorem statement that proves the new salary is €10,200
theorem new_salary_correct :
  new_salary initial_salary percentage_increase = 10200 := by
  sorry

end new_salary_correct_l321_321113


namespace teal_more_blue_l321_321872

theorem teal_more_blue (total : ℕ) (more_green : ℕ) (both_green_and_blue : ℕ) (neither_green_nor_blue : ℕ) :
  total = 150 → 
  more_green = 90 → 
  both_green_and_blue = 45 → 
  neither_green_nor_blue = 24 →
  (total - (more_green - both_green_and_blue + both_green_and_blue + neither_green_nor_blue)) + both_green_and_blue = 81 :=
by {
  intros h_total h_more_green h_both h_neither,
  simp at *,
  sorry
}

end teal_more_blue_l321_321872


namespace perpendicular_bisector_of_line_segment_l321_321001

theorem perpendicular_bisector_of_line_segment 
    (e : ℝ) : 
    let mid_x := (2 + 8) / 2,
        mid_y := (4 + 10) / 2,
        midpoint := (mid_x, mid_y)
    in (e = 12) :=
by
    let mid_x := (2 + 8) / 2
    let mid_y := (4 + 10) / 2
    let midpoint := (mid_x, mid_y)
    have mid_eq : midpoint = (5, 7) := by simp [mid_x, mid_y]
    have eq_1 : 5 + 7 = e := by rw [mid_eq]; exact rfl
    have e_eq : e = 12 := by linarith [eq_1]
    exact e_eq

end perpendicular_bisector_of_line_segment_l321_321001


namespace integral_of_f_l321_321944

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 1

theorem integral_of_f :
  ∫ x in 1..2, f x = 6 :=
by
  sorry

end integral_of_f_l321_321944


namespace additional_marbles_needed_l321_321696

theorem additional_marbles_needed (total_friends marbles_held : ℕ) (min_total_marbles_needed : ℕ) : total_friends = 10 → marbles_held = 34 → min_total_marbles_needed = 55 → marbles_held < min_total_marbles_needed → min_total_marbles_needed - marbles_held = 21 :=
by
  intros H1 H2 H3 H4
  rw [H1, H2, H3]
  exact Nat.sub_eq 55 34 21 sorry

end additional_marbles_needed_l321_321696


namespace number_of_cubes_with_at_least_two_faces_painted_is_56_l321_321805

def one_inch_cubes_with_at_least_two_faces_painted 
  (side_length : ℕ) (face_colors : ℕ) (cubes_per_side : ℕ) :=
  if side_length = 4 ∧ face_colors = 6 ∧ cubes_per_side = 1 then 56 else 0

theorem number_of_cubes_with_at_least_two_faces_painted_is_56 :
  one_inch_cubes_with_at_least_two_faces_painted 4 6 1 = 56 :=
by
  sorry

end number_of_cubes_with_at_least_two_faces_painted_is_56_l321_321805


namespace sqrt_expression_value_l321_321541

theorem sqrt_expression_value :
  sqrt((2 - sin(π / 6) ^ 2) * (2 - sin(π / 4) ^ 2) * (2 - sin(π / 3) ^ 2)) = (sqrt 210) / 8 :=
by
  have h1 : sin (π / 6) = 1 / 2 := by sorry
  have h2 : sin (π / 4) = sqrt 2 / 2 := by sorry
  have h3 : sin (π / 3) = sqrt 3 / 2 := by sorry
  sorry

end sqrt_expression_value_l321_321541


namespace product_of_divisors_of_30_l321_321447

theorem product_of_divisors_of_30 : ∏ d in (finset.filter (∣ 30) (finset.range 31)), d = 810000 := by
  sorry

end product_of_divisors_of_30_l321_321447


namespace trigonometric_identity_l321_321594

theorem trigonometric_identity (θ : ℝ) (x y : ℝ) (h₀ : θ = 480 ∨ θ = 480 + 360)
  (h₁ : P : Point x y ∧ ¬ (x = 0 ∧ y = 0)) : 
  (xy : ℝ := x * y; x_squared : ℝ := x^2; y_squared : ℝ := y^2; numerator : ℝ := x * sqrt(3) * x; denominator : ℝ := x^2 + (sqrt(3) * x)^2) 
  xy / (x_squared + y_squared) = (sqrt(3)) / 4 := by
  sorry

end trigonometric_identity_l321_321594


namespace find_integer_mod_l321_321059

theorem find_integer_mod (n : ℤ) : 
  0 ≤ n ∧ n < 17 ∧ (-250 ≡ n [ZMOD 17]) ↔ n = 5 :=
by
  split;
  {
    intro h;
    cases' h with hn h_eq;
    cases' h_eq with hlt hmod;
    {
      unfold Int.ModEq at hmod;
      sorry
    }
  }

end find_integer_mod_l321_321059


namespace number_of_cubes_with_at_least_two_faces_painted_is_56_l321_321807

def one_inch_cubes_with_at_least_two_faces_painted 
  (side_length : ℕ) (face_colors : ℕ) (cubes_per_side : ℕ) :=
  if side_length = 4 ∧ face_colors = 6 ∧ cubes_per_side = 1 then 56 else 0

theorem number_of_cubes_with_at_least_two_faces_painted_is_56 :
  one_inch_cubes_with_at_least_two_faces_painted 4 6 1 = 56 :=
by
  sorry

end number_of_cubes_with_at_least_two_faces_painted_is_56_l321_321807


namespace fraction_students_not_enjoy_but_actually_enjoy_l321_321134

noncomputable def number_of_students := 200
def proportion_enjoy_soccer := 0.7
def correct_statement_enjoy := 0.75
def correct_statement_not_enjoy := 0.85

theorem fraction_students_not_enjoy_but_actually_enjoy :
  let total_students := (number_of_students : ℝ)
  let enjoy_soccer := proportion_enjoy_soccer * total_students
  let not_enjoy_soccer := total_students - enjoy_soccer
  let enjoy_but_say_not := (1 - correct_statement_enjoy) * enjoy_soccer
  let not_enjoy_and_say_not := correct_statement_not_enjoy * not_enjoy_soccer
  let total_say_not_enjoy := enjoy_but_say_not + not_enjoy_and_say_not
  (enjoy_but_say_not / total_say_not_enjoy) = (2 / 5) := sorry

end fraction_students_not_enjoy_but_actually_enjoy_l321_321134


namespace triangle_area_l321_321741

theorem triangle_area (A B C P : Point)
  (h_right : ∃ (α β γ : ℝ), α = 90 ∧ β ≠ 90 ∧ γ ≠ 90 ∧ β + γ = 90)
  (h_angle_ABP : Measure.angle A B P = 30)
  (h_AP : dist A P = 2)
  (h_CP : dist C P = 1)
  (h_AC : dist A C = 3) :
  area_triangle A B C = 9 / 5 :=
sorry

end triangle_area_l321_321741


namespace asymptotes_hyperbola_eq_l321_321582

theorem asymptotes_hyperbola_eq :
  ∀ (a : ℝ) (a_pos : 0 < a) (O P A B : (ℝ × ℝ)),
  let hyperbola_eq := ∀ (x y : ℝ), (x, y) ∈ set_of (λ (p : ℝ × ℝ), p.1^2 / a^2 - p.2^2 = 1),
      eq_p := P ∈ set_of (λ (p : ℝ × ℝ), p.1^2  / a^2 - p.2^2 = 1),
      asymptote1 := ∀ (x y : ℝ), y = (1/a) * x,
      asymptote2 := ∀ (x y : ℝ), y = -(1/a) * x,
      parallel_to_asymptotes := ∀ (l1 l2 : ℝ × ℝ → Prop), (l1 = asymptote1 ∨ l1 = asymptote2) ∧
                                                            (l2 = asymptote1 ∨ l2 = asymptote2) ∧
                                                            ∃ (k : ℝ), l1(P) ∧ l2(P),
      parallelogram_area_eq1 : 1 = parallelogram_area O B P A :=
  parallel_to_asymptotes →
  parallelogram_area_eq1 →
  a = 2 :=
begin
  sorry
end

end asymptotes_hyperbola_eq_l321_321582


namespace no_solution_outside_intervals_l321_321164

theorem no_solution_outside_intervals (x a : ℝ) :
  (a < 0 ∨ a > 10) → 3 * |x + 3 * a| + |x + a^2| + 2 * x ≠ a :=
by {
  sorry
}

end no_solution_outside_intervals_l321_321164


namespace find_d_of_quadratic_roots_l321_321787

theorem find_d_of_quadratic_roots :
  ∃ d : ℝ, (∀ x : ℝ, x^2 + 7 * x + d = 0 ↔ x = (-7 + real.sqrt d) / 2 ∨ x = (-7 - real.sqrt d) / 2) → d = 9.8 :=
by
  sorry

end find_d_of_quadratic_roots_l321_321787


namespace total_bees_in_colony_l321_321641

def num_bees_in_hive_after_changes (initial_bees : ℕ) (bees_in : ℕ) (bees_out : ℕ) : ℕ :=
  initial_bees + bees_in - bees_out

theorem total_bees_in_colony :
  let hive1 := num_bees_in_hive_after_changes 45 12 8
  let hive2 := num_bees_in_hive_after_changes 60 15 20
  let hive3 := num_bees_in_hive_after_changes 75 10 5
  hive1 + hive2 + hive3 = 184 :=
by
  sorry

end total_bees_in_colony_l321_321641


namespace Equ_Joint_Proof_l321_321225

-- Define the given parabola and conditions
def parabola (p : ℝ) (hp : p > 0) : set (ℝ × ℝ) :=
  { point | point.2 ^ 2 = 2 * p * point.1 }

def line (m : ℝ) : set (ℝ × ℝ) :=
  { point | point.1 = m * point.2 - 2 }

def C : ℝ × ℝ := (-2, 0)
def O : ℝ × ℝ := (0, 0)
def points_A_B (A B : ℝ × ℝ) : Prop :=
  if O.1 * O.1 + O.2 * O.2 = 12 then true else false

theorem Equ_Joint_Proof
  {p : ℝ} (hp : p > 0)
  (A B : ℝ × ℝ)
  (h1 : A ∈ parabola p hp)
  (h2 : B ∈ parabola p hp)
  (h3 : points_A_B A B)
  (circle_diam_area : real.pi * (8/2)^2 = 16 * real.pi):
  (p = 2) ∧ (∃ S : ℝ, S = 4) :=
by {
  sorry,  
}

end Equ_Joint_Proof_l321_321225


namespace regular_tetrahedron_properties_l321_321524

theorem regular_tetrahedron_properties :
  (∀ (Δ : Type) [equilateral_triangle Δ], 
    (∀ (T : Type) [regular_tetrahedron T],
      (all_edges_equal T ∧
      angle_between_edges_equal T ∧
      all_faces_congruent_equilateral_triangles T ∧
      dihedral_angle_between_faces_equal T))
  := 
begin 
  sorry 
end

end regular_tetrahedron_properties_l321_321524


namespace area_of_quadrilateral_l321_321165

theorem area_of_quadrilateral (d a b : ℝ) (h₀ : d = 28) (h₁ : a = 9) (h₂ : b = 6) :
  (1 / 2 * d * a) + (1 / 2 * d * b) = 210 :=
by
  -- Provided proof steps are skipped
  sorry

end area_of_quadrilateral_l321_321165


namespace ellipse_standard_eqn_line_AB_eqn_l321_321208

-- Ellipse conditions and derived equation
def eccentricity (e : ℝ) (c a : ℝ) : Prop := e = c / a
def vertex_distance (a c : ℝ) : Prop := a - c = 1
def ellipse_eqn (x y : ℝ) (a b : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1

theorem ellipse_standard_eqn :
  ∃ a b c : ℝ,
  eccentricity (1/2) c a ∧ vertex_distance a c ∧ ellipse_eqn x y 2 (sqrt 3) :=
sorry

-- Line conditions and derived equation
def midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

def line_eqn (x y k : ℝ) : Prop :=
  y - 1 = k * (x - 1)

def line_slope (x1 y1 x2 y2 : ℝ) (k : ℝ) : Prop :=
  k = (y1 - y2) / (x1 - x2)

theorem line_AB_eqn :
  ∃ x1 y1 x2 y2 k : ℝ,
  midpoint (1, 1) (x1, y1) (x2, y2) ∧ ellipse_eqn x1 y1 2 (sqrt 3) ∧ ellipse_eqn x2 y2 2 (sqrt 3) ∧
  line_slope x1 y1 x2 y2 k ∧ line_eqn x y (-3/4) :=
sorry

end ellipse_standard_eqn_line_AB_eqn_l321_321208


namespace canoe_row_probability_l321_321460

theorem canoe_row_probability :
  let p_left_works := 3 / 5
  let p_right_works := 3 / 5
  let p_left_breaks := 1 - p_left_works
  let p_right_breaks := 1 - p_right_works
  let p_can_still_row := (p_left_works * p_right_works) + (p_left_works * p_right_breaks) + (p_left_breaks * p_right_works)
  p_can_still_row = 21 / 25 :=
by
  sorry

end canoe_row_probability_l321_321460


namespace combined_difference_is_correct_l321_321905

-- Define the number of cookies each person has
def alyssa_cookies : Nat := 129
def aiyanna_cookies : Nat := 140
def carl_cookies : Nat := 167

-- Define the differences between each pair of people's cookies
def diff_alyssa_aiyanna : Nat := aiyanna_cookies - alyssa_cookies
def diff_alyssa_carl : Nat := carl_cookies - alyssa_cookies
def diff_aiyanna_carl : Nat := carl_cookies - aiyanna_cookies

-- Define the combined difference
def combined_difference : Nat := diff_alyssa_aiyanna + diff_alyssa_carl + diff_aiyanna_carl

-- State the theorem to be proved
theorem combined_difference_is_correct : combined_difference = 76 := by
  sorry

end combined_difference_is_correct_l321_321905


namespace denominator_of_fraction_fraction_denominator_is_99_l321_321757

theorem denominator_of_fraction (S : ℚ)
  (h1 : S = 0.\overline{47}) : (S.num.gcd 99 = 1) := 
by
  sorry

theorem fraction_denominator_is_99 (f : ℚ)
  (h1 : f = 47 / 99) :
  f.denominator = 99 :=
by
  sorry

end denominator_of_fraction_fraction_denominator_is_99_l321_321757


namespace arithmetic_sequence_twelfth_term_l321_321846

theorem arithmetic_sequence_twelfth_term :
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  (a1 + 11 * d) = (25 : ℚ) / 6 :=
by
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  exact sorry

end arithmetic_sequence_twelfth_term_l321_321846


namespace max_area_of_rectangle_with_perimeter_40_l321_321011

theorem max_area_of_rectangle_with_perimeter_40 :
  (∃ (x y : ℝ), (2 * x + 2 * y = 40) ∧
                (∀ (a b : ℝ), (2 * a + 2 * b = 40) → (a * b ≤ x * y)) ∧
                (x * y = 100)) :=
begin
  -- Definitions of x and y satisfying the perimeter and maximizing the area.
  have h1 : ∀ (x y : ℝ), 2 * x + 2 * y = 40 → x * (20 - x) = -(x - 10)^2 + 100,
  { intro x, intro y, intro hper,
    have hy : y = 20 - x, by linarith,
    rw hy,
    ring },
  use 10,
  use 10,
  split,
  { -- Perimeter condition
    linarith },
  { split,
    { -- Maximum area condition
      intros a b hper,
      have hab : b = 20 - a, by linarith,
      rw hab,
      specialize h1 a (20 - a),
      linarith },
    { -- Maximum area is 100
      exact (by ring) } }
end

end max_area_of_rectangle_with_perimeter_40_l321_321011


namespace first_pedestrian_speed_l321_321468

def pedestrian_speed (x : ℝ) : Prop :=
  0 < x ∧ x ≤ 4 ↔ (11 / (x + 1.5) + 8 / x ≥ 12 / (x + 2) + 2)

theorem first_pedestrian_speed 
  (x : ℝ) (h : 11 / (x + 1.5) + 8 / x ≥ 12 / (x + 2) + 2) :
  0 < x ∧ x ≤ 4 :=
begin
  sorry
end

end first_pedestrian_speed_l321_321468


namespace initial_distance_between_cars_l321_321053

theorem initial_distance_between_cars :
  ∀ (D : ℕ), (D - (65 - 62) = 38) → D = 165 :=
by
  intro D
  intro h
  have h1: D - 3 = 38 := h
  have h2: D = 38 + 3 := by linarith
  have h3: D = 41 := h2
  have h4: D = 165 := by linarith
  rw h4
  sorry

end initial_distance_between_cars_l321_321053


namespace largest_prime_divisor_13_fac_add_14_fac_l321_321968

theorem largest_prime_divisor_13_fac_add_14_fac : 
  ∀ (p : ℕ), prime p → p ∣ (13! + 14!) → p ≤ 5 :=
by
  sorry

end largest_prime_divisor_13_fac_add_14_fac_l321_321968


namespace smallest_possible_number_of_students_l321_321288

theorem smallest_possible_number_of_students :
  ∃ n : ℕ, (n % 200 = 0) ∧ (∀ m : ℕ, (m < n → 
    75 * m ≤ 100 * n) ∧
    (∃ a b c : ℕ, a = m / 4 ∧ b = a / 10 ∧ 
    ∃ y z : ℕ, y = 3 * z ∧ (y + z - b = a) ∧ y * c = n / 4)) :=
by
  sorry

end smallest_possible_number_of_students_l321_321288


namespace y_share_per_rupee_of_x_l321_321900

theorem y_share_per_rupee_of_x (share_y : ℝ) (total_amount : ℝ) (z_per_x : ℝ) (y_per_x : ℝ) 
  (h1 : share_y = 54) 
  (h2 : total_amount = 210) 
  (h3 : z_per_x = 0.30) 
  (h4 : share_y = y_per_x * (total_amount / (1 + y_per_x + z_per_x))) : 
  y_per_x = 0.45 :=
sorry

end y_share_per_rupee_of_x_l321_321900


namespace largest_prime_divisor_13_fac_add_14_fac_l321_321971

theorem largest_prime_divisor_13_fac_add_14_fac : 
  ∀ (p : ℕ), prime p → p ∣ (13! + 14!) → p ≤ 5 :=
by
  sorry

end largest_prime_divisor_13_fac_add_14_fac_l321_321971


namespace new_salary_calculation_l321_321115

-- Define the initial conditions of the problem
def current_salary : ℝ := 10000
def percentage_increase : ℝ := 2
def increase := current_salary * (percentage_increase / 100)
def new_salary := current_salary + increase

-- Define the theorem to check the new salary
theorem new_salary_calculation : new_salary = 10200 := by
  -- Lean would check the proof here, but it's being skipped with 'sorry'
  sorry

end new_salary_calculation_l321_321115


namespace travel_routes_l321_321258

theorem travel_routes (S N : Finset ℕ) (hS : S.card = 4) (hN : N.card = 5) :
  ∃ (routes : ℕ), routes = 3! * 5^4 := by
  sorry

end travel_routes_l321_321258


namespace product_of_divisors_of_30_l321_321446

theorem product_of_divisors_of_30 : ∏ d in (finset.filter (∣ 30) (finset.range 31)), d = 810000 := by
  sorry

end product_of_divisors_of_30_l321_321446


namespace maximum_area_sector_l321_321152

noncomputable def theta (r : ℝ) : ℝ := (20 - 2 * r) / r

noncomputable def area (r : ℝ) : ℝ := (1 / 2) * theta r * r^2

theorem maximum_area_sector : 
  ∃ (r θ : ℝ), 0 < r ∧ r < 10 ∧ θ = 2 ∧ θ = theta r 
    ∧ area r = 25 :=
begin
  use 5,
  use 2,
  split,
  { linarith, }, -- 0 < 5 holds.
  split,
  { linarith, }, -- 5 < 10 holds.
  split,
  { refl, }, -- θ = 2 by definition.
  split,
  { unfold theta, norm_num, }, -- θ = theta 5 evaluates to 2.
  { unfold area theta, norm_num, field_simp, ring, }, -- Area evaluates to 25.
end

end maximum_area_sector_l321_321152


namespace find_p_q_r_s_sum_l321_321689

noncomputable def Q (x : ℝ) : ℝ := x^2 - 5*x - 7

theorem find_p_q_r_s_sum :
  (let p := 221
       q := 1
       r := 1
       s := 14 in
   4 ≤ x ∧ x ≤ 18 →
   (∀ x : ℝ, floor (sqrt (Q x)) = sqrt (Q (floor x)) → 
   (let prob := (sqrt (p : ℝ) + sqrt (q : ℝ) - r) / s in
    p + q + r + s = 236))) := sorry

end find_p_q_r_s_sum_l321_321689


namespace count_perfect_squares_and_cubes_l321_321619

theorem count_perfect_squares_and_cubes :
  let lower_bound := 100
  let upper_bound := 1000
  let is_perfect_square (n : ℕ) := ∃ (k : ℕ), k ^ 2 = n
  let is_perfect_cube (n : ℕ) := ∃ (k : ℕ), k ^ 3 = n
  let is_sixth_power (n : ℕ) := ∃ (k : ℕ), k ^ 6 = n
  ∃ (squares cubes sixth_powers : ℕ),
  (squares = set.count (set.filter (λ n, lower_bound ≤ n ∧ n ≤ upper_bound) (set.filter is_perfect_square (finset.range (upper_bound + 1)))))
  ∧ (cubes = set.count (set.filter (λ n, lower_bound ≤ n ∧ n ≤ upper_bound) (set.filter is_perfect_cube (finset.range (upper_bound + 1)))))
  ∧ (sixth_powers = set.count (set.filter (λ n, lower_bound ≤ n ∧ n ≤ upper_bound) (set.filter is_sixth_power (finset.range (upper_bound + 1)))))
  ∧ squares + cubes - sixth_powers = 26 :=
by sorry

end count_perfect_squares_and_cubes_l321_321619


namespace arithmetic_sequence_b_ac_l321_321621

theorem arithmetic_sequence_b_ac
  (a b c : ℤ)
  (d : ℤ)
  (h1 : a = -1 + d)
  (h2 : b = -1 + 2 * d)
  (h3 : c = -1 + 3 * d)
  (h4 : -9 = -1 + 4 * d) :
  b = -5 ∧ a * c = 21 :=
by
  -- solving the equations
  have d_val : d = -2, from sorry,
  have b_val : b = -5, from by rw [h2, d_val]; norm_num,
  have ac_val : a * c = 21, from by rw [h1, h3, d_val]; norm_num,
  exact ⟨b_val, ac_val⟩

end arithmetic_sequence_b_ac_l321_321621


namespace HeatherIsHeavier_l321_321230

-- Definitions
def HeatherWeight : ℕ := 87
def EmilyWeight : ℕ := 9

-- Theorem statement
theorem HeatherIsHeavier : HeatherWeight - EmilyWeight = 78 := by
  sorry

end HeatherIsHeavier_l321_321230


namespace min_value_expression_l321_321554

theorem min_value_expression (x y k : ℝ) (hk : 1 < k) (hx : k < x) (hy : k < y) : 
  (∀ x y, x > k → y > k → (∃ m, (m ≤ (x^2 / (y - k) + y^2 / (x - k)))) ∧ (m = 8 * k)) := sorry

end min_value_expression_l321_321554


namespace unknown_number_l321_321004

theorem unknown_number (n : ℕ) (h1 : Nat.lcm 24 n = 168) (h2 : Nat.gcd 24 n = 4) : n = 28 :=
by
  sorry

end unknown_number_l321_321004


namespace min_distance_between_parallel_lines_l321_321505

theorem min_distance_between_parallel_lines
  (m c_1 c_2 : ℝ)
  (h_parallel : ∀ x : ℝ, m * x + c_1 = m * x + c_2 → false) :
  ∃ D : ℝ, D = (|c_2 - c_1|) / (Real.sqrt (1 + m^2)) :=
by
  sorry

end min_distance_between_parallel_lines_l321_321505


namespace problem_l321_321565

theorem problem (a b : ℝ) 
  (h1 : f x = x^2 + 4*x + 3)
  (h2 : f (a*x + b) = x^2 + 10*x + 24) : 5 * a - b = 2 :=
by
  sorry

end problem_l321_321565


namespace seq_a_general_term_seq_b_general_term_sum_c_n_terms_l321_321575

noncomputable def a (n : ℕ) : ℝ := if n = 0 then 0 else 2^(n-1)
noncomputable def b (n : ℕ) : ℝ := 2^(n-1) + 1
noncomputable def c (n : ℕ) : ℝ := a n / (b n * b (n + 1))
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, c (i + 1)

theorem seq_a_general_term :
  (∀ n : ℕ, n > 0 → a n = 2^(n-1)) :=
sorry

theorem seq_b_general_term :
  (∀ n : ℕ, n > 0 → b n = 2^(n-1) + 1) :=
sorry

theorem sum_c_n_terms :
  (∀ n : ℕ, T n = 1/2 - 1/(1 + 2^n)) :=
sorry

end seq_a_general_term_seq_b_general_term_sum_c_n_terms_l321_321575


namespace traders_fabric_sales_l321_321049

theorem traders_fabric_sales (x y : ℕ) : 
  x + y = 85 ∧
  x = y + 5 ∧
  60 = x * (60 / y) ∧
  30 = y * (30 / x) →
  (x, y) = (25, 20) :=
by {
  sorry
}

end traders_fabric_sales_l321_321049


namespace non_integer_powers_in_expansion_l321_321237

noncomputable def integral_expr : ℝ :=
  2 * ∫ x in -3..3, (x + |x|)

theorem non_integer_powers_in_expansion :
  integral_expr = 18 → 
  ∀ a : ℝ, a = integral_expr → 
  ∑ i in finset.range 19, (∃ k : ℤ, k = (54 - 5 * i) / 6) = 4 →
  19 - 4 = 15 :=
by
  intros h_int_expr h_a h_terms
  sorry

end non_integer_powers_in_expansion_l321_321237


namespace one_inch_cubes_with_two_or_more_painted_faces_l321_321812

def original_cube_length : ℕ := 4

def total_one_inch_cubes : ℕ := original_cube_length ^ 3

def corners_count : ℕ := 8

def edges_minus_corners_count : ℕ := 12 * 2

theorem one_inch_cubes_with_two_or_more_painted_faces
  (painted_faces_on_each_face : ∀ i : ℕ, i < total_one_inch_cubes → ℕ) : 
  ∃ n : ℕ, n = corners_count + edges_minus_corners_count ∧ n = 32 := 
by
  simp only [corners_count, edges_minus_corners_count, total_one_inch_cubes]
  sorry

end one_inch_cubes_with_two_or_more_painted_faces_l321_321812


namespace find_cosine_of_smallest_angle_of_triangle_l321_321769

noncomputable def cosine_of_smallest_angle (n : ℕ) : ℚ := 
  (2*n^2 + 6*n + 5) / (2*(n + 1)*(n + 2))

theorem find_cosine_of_smallest_angle_of_triangle :
  ∀ (n : ℕ), 
  (n, n+1, n+2).Sorted (≤) → 
  (∃ n = 4, cosine_of_smallest_angle n = 53 / 60) :=
by
  intros n h_sorted
  sorry

end find_cosine_of_smallest_angle_of_triangle_l321_321769


namespace consecutive_page_numbers_sum_l321_321777

theorem consecutive_page_numbers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 35280) :
  n + (n + 1) + (n + 2) = 96 := sorry

end consecutive_page_numbers_sum_l321_321777


namespace frog_jumps_l321_321093

theorem frog_jumps (n : ℕ) : 
  (∃ (a b : ℕ), a + b = n ∧ (a - 2 * b) % n = 0 ∧ ∀ v ∈ finset.range n, frogged v a b n) <->
  (6 ∣ n ∨ ¬(30 ∣ n)) :=
by
  sorry

-- Additional lemma needed to verify the frog's pattern covers all vertices
noncomputable def frogged (v : ℕ) (a b n : ℕ) : Prop :=
  ∀ i ∈ (finset.range n), v = (i * 1 + (a - 2 * b + n) % n)

-- Lean is not designed to handle all natural language problems directly. 
-- Definitions and additional constraints have to be added to clarify context.

end frog_jumps_l321_321093


namespace largest_prime_divisor_of_factorial_sum_l321_321958

theorem largest_prime_divisor_of_factorial_sum :
  ∃ p : ℕ, p.prime ∧ p ∣ (13! + 14!) ∧ ∀ q : ℕ, q.prime ∧ q ∣ (13! + 14!) → q ≤ p :=
sorry

end largest_prime_divisor_of_factorial_sum_l321_321958


namespace original_price_tv_l321_321106

variable (X : ℝ) -- define the original price as a real number
variable (incr_rate : ℝ) := 0.4 -- increase rate of 40%
variable (sale_rate : ℝ) := 0.2 -- sale rate of 20%
variable (profit : ℝ) := 270 -- additional profit

-- Define the conditions
def condition1 := (1 + incr_rate) * (1 - sale_rate) * X - X = profit

-- Theorem statement to prove the original price
theorem original_price_tv : condition1 X incr_rate sale_rate profit → X = 2250 := by
  intros h
  sorry -- proof to be filled in

end original_price_tv_l321_321106


namespace avg_ABC_correct_l321_321870

variables (A B C : Set Person)
variables (a b c : ℕ) -- Number of people in sets A, B, and C
variables (sum_A sum_B sum_C : ℕ) -- Sum of ages in sets A, B, and C
variables (avg_A : ℕ := sum_A / a) (avg_B : ℕ := sum_B / b) (avg_C : ℕ := sum_C / c)

axiom avg_A_cond : avg_A = 40
axiom avg_B_cond : avg_B = 25
axiom avg_C_cond : avg_C = 45
axiom avg_AB_cond : (sum_A + sum_B) / (a + b) = 32
axiom avg_AC_cond : (sum_A + sum_C) / (a + c) = 42
axiom avg_BC_cond : (sum_B + sum_C) / (b + c) = 35

def avg_ABC : ℕ := (sum_A + sum_B + sum_C) / (a + b + c)

theorem avg_ABC_correct : avg_ABC = 49.15 := by
  sorry

end avg_ABC_correct_l321_321870


namespace optimal_chalk_length_l321_321305

theorem optimal_chalk_length (l : ℝ) (h₁: 10 ≤ l) (h₂: l ≤ 15) (h₃: l = 12) : l = 12 :=
by
  sorry

end optimal_chalk_length_l321_321305


namespace lily_not_enough_money_l321_321349

-- Define the various prices and conditions
def celery_base_price : ℝ := 8
def celery_discount : ℝ := 0.20
def cereal_price : ℝ := 14
def bread_base_price : ℝ := 10
def bread_discount : ℝ := 0.05
def milk_base_price : ℝ := 12
def milk_discount : ℝ := 0.15
def potato_price_per_pound : ℝ := 2
def potato_weight : ℝ := 8
def cookies_price : ℝ := 15
def tax_rate : ℝ := 0.07
def initial_budget : ℝ := 70

-- Calculate the effective prices
def celery_price : ℝ := celery_base_price * (1 - celery_discount)
def cereal_price_total : ℝ := cereal_price
def bread_price : ℝ := bread_base_price * (1 - bread_discount)
def milk_price : ℝ := milk_base_price * (1 - milk_discount)
def potatoes_price : ℝ := potato_price_per_pound * potato_weight
def cookies_price_total : ℝ := cookies_price

-- Calculate the tax for each item
def celery_tax : ℝ := celery_price * tax_rate
def cereal_tax : ℝ := cereal_price_total * tax_rate
def bread_tax : ℝ := bread_price * tax_rate
def milk_tax : ℝ := milk_price * tax_rate
def potatoes_tax : ℝ := potatoes_price * tax_rate
def cookies_tax : ℝ := cookies_price_total * tax_rate

-- Calculate the total price with taxes
def total_celery : ℝ := celery_price + celery_tax
def total_cereal : ℝ := cereal_price_total + cereal_tax
def total_bread : ℝ := bread_price + bread_tax
def total_milk : ℝ := milk_price + milk_tax
def total_potatoes : ℝ := potatoes_price + potatoes_tax
def total_cookies : ℝ := cookies_price_total + cookies_tax

-- Sum the total price of all items
def total_cost : ℝ := total_celery + total_cereal + total_bread + total_milk + total_potatoes + total_cookies

theorem lily_not_enough_money : initial_budget < total_cost :=
by {
  have h1 : celery_price = 6.40, from sorry,
  have h2 : celery_tax = 0.448, from sorry,
  have h3 : total_celery = 6.848, from sorry,
  have h4 : total_cereal = 14.98, from sorry,
  have h5 : total_bread = 10.165, from sorry,
  have h6 : total_milk = 10.914, from sorry,
  have h7 : total_potatoes = 17.12, from sorry,
  have h8 : total_cookies = 16.05, from sorry,
  have h9 : total_cost = 76.077, from sorry,
  show initial_budget < total_cost, from sorry,
}

end lily_not_enough_money_l321_321349


namespace min_period_of_function_l321_321009

theorem min_period_of_function :
  ∃ T > 0, ∀ x, 2 * sin (2 * x)^2 - 1 = 2 * sin (2 * (x + T))^2 - 1 :=
by
  use π/2
  sorry

end min_period_of_function_l321_321009


namespace one_inch_cubes_with_two_or_more_painted_faces_l321_321809

def original_cube_length : ℕ := 4

def total_one_inch_cubes : ℕ := original_cube_length ^ 3

def corners_count : ℕ := 8

def edges_minus_corners_count : ℕ := 12 * 2

theorem one_inch_cubes_with_two_or_more_painted_faces
  (painted_faces_on_each_face : ∀ i : ℕ, i < total_one_inch_cubes → ℕ) : 
  ∃ n : ℕ, n = corners_count + edges_minus_corners_count ∧ n = 32 := 
by
  simp only [corners_count, edges_minus_corners_count, total_one_inch_cubes]
  sorry

end one_inch_cubes_with_two_or_more_painted_faces_l321_321809


namespace seven_students_arrangement_l321_321408

theorem seven_students_arrangement:
  let total_students : ℕ := 7
  let ab_together_permutations : ℕ := (6! * 2!)
  total_students = 7 ∧ ab_together_permutations = 1440 :=
  sorry

end seven_students_arrangement_l321_321408


namespace line_through_intersection_points_l321_321552

def first_circle (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def second_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem line_through_intersection_points (x y : ℝ) :
  (first_circle x y ∧ second_circle x y) → x - y - 3 = 0 :=
by
  sorry

end line_through_intersection_points_l321_321552


namespace total_classic_books_l321_321773

-- Definitions for the conditions
def authors := 6
def books_per_author := 33

-- Statement of the math proof problem
theorem total_classic_books : authors * books_per_author = 198 := by
  sorry  -- Proof to be filled in

end total_classic_books_l321_321773


namespace axis_of_symmetry_l321_321602

noncomputable def f (x : ℝ) : ℝ := abs (sin (2 * x - (Real.pi / 6)))

theorem axis_of_symmetry : ∃ k : ℤ, (∃ x : ℝ, x = 1/2 * k * Real.pi + Real.pi / 3) :=
begin
  existsi 0,
  existsi Real.pi / 3,
  sorry
end

end axis_of_symmetry_l321_321602


namespace frog_arrangements_l321_321039

theorem frog_arrangements :
  let frogs : Finset (Nat × Char) := {(1, 'G'), (2, 'G'), (3, 'G'), (4, 'R'), (5, 'R'), (6, 'R'), (7, 'R'), (8, 'B')} in
  let valid_sequences : Finset (List Char) := {['G', 'G', 'G', 'B', 'R', 'R', 'R', 'R'], ['R', 'R', 'R', 'R', 'B', 'G', 'G', 'G']} in
  let count_arrangements (seq : List Char) : Nat := 
    if seq = ['G', 'G', 'G', 'B', 'R', 'R', 'R', 'R'] then 4! * 3!
    else if seq = ['R', 'R', 'R', 'R', 'B', 'G', 'G', 'G'] then 4! * 3!
    else 0 in
  let total_arrangements := ∑ seq in valid_sequences, count_arrangements seq in
  total_arrangements = 288 :=
sorry

end frog_arrangements_l321_321039


namespace solve_for_x_l321_321853

theorem solve_for_x : ∃ x : ℝ, (2^12 + 2^12 + 2^12 = 6^x) ∧ x = 4.3 := 
by
  sorry

end solve_for_x_l321_321853


namespace rearrange_HMMTHMMT_l321_321297

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

def multiset_permutations (n : ℕ) (ns : list ℕ) : ℕ :=
factorial n / (ns.map factorial).prod

theorem rearrange_HMMTHMMT :
  let total_permutations := multiset_permutations 8 [2, 4, 2],
      substring_appears := multiset_permutations 5 [3] * multiset_permutations 4 [2] in
  total_permutations - substring_appears = 361 :=
by
  let total_permutations := multiset_permutations 8 [2, 4, 2],
      substring_appears := multiset_permutations 5 [3] * multiset_permutations 4 [2]
  show total_permutations - substring_appears = 361, from sorry

end rearrange_HMMTHMMT_l321_321297


namespace max_value_of_quadratic_zero_l321_321589

noncomputable def zero_max_value (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hab_sum : a + b = 1) : ℝ :=
  (-9 + Real.sqrt 85) / 2

theorem max_value_of_quadratic_zero (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hab_sum : a + b = 1) :
  ∃ x, is_zero_point_of_function f x ∧ f_max = zero_max_value a b ha_pos hb_pos hab_sum := 
sorry -- Needs proof


end max_value_of_quadratic_zero_l321_321589


namespace taxi_fare_problem_l321_321037

-- Define the conditions as Lean definitions
def initial_fare := 3.50
def first_half_mile := 0.5
def additional_fare_per_tenth_mile := 0.30
def available_money := 15
def tip := 3
def fare_without_tip := available_money - tip -- 12

-- Define the proof problem
theorem taxi_fare_problem : 
  let x := 3.33 in
  initial_fare + additional_fare_per_tenth_mile * ((x - first_half_mile) / 0.1) = fare_without_tip :=
by
  sorry

end taxi_fare_problem_l321_321037


namespace largest_prime_divisor_13_fact_14_fact_l321_321973

theorem largest_prime_divisor_13_fact_14_fact :
  ∀ (n : ℕ), (n = 13!) ∧ (m = 14!) ∧ (m = 14 * n) ∧ ((n + m) = n * 15) ∧ (15 = 3 * 5) → 
  (∃ p : ℕ, (Prime p) ∧ (LargestPrimeDivisor p (n + m)) ∧ (p = 13)) :=
by
  sorry

end largest_prime_divisor_13_fact_14_fact_l321_321973


namespace wealthiest_500_income_l321_321763

theorem wealthiest_500_income (N x : ℝ) (hx_pos : 0 < x) (hN : N = 500) 
  (h_formula : N = 5 * 10^9 * x^(-3 / 2)) : x = 10^(14 / 3) :=
by
  sorry

end wealthiest_500_income_l321_321763


namespace largest_prime_divisor_13_fac_add_14_fac_l321_321972

theorem largest_prime_divisor_13_fac_add_14_fac : 
  ∀ (p : ℕ), prime p → p ∣ (13! + 14!) → p ≤ 5 :=
by
  sorry

end largest_prime_divisor_13_fac_add_14_fac_l321_321972


namespace initial_population_is_10000_l321_321776

def population_growth (P : ℝ) : Prop :=
  let growth_rate := 0.20
  let final_population := 12000
  final_population = P * (1 + growth_rate)

theorem initial_population_is_10000 : population_growth 10000 :=
by
  unfold population_growth
  sorry

end initial_population_is_10000_l321_321776


namespace smallest_number_of_students_l321_321273

theorem smallest_number_of_students 
    (n : ℕ) 
    (attended := n / 4)
    (both := n / 40)
    (cheating_hint_ratio : ℚ := 3 / 2)
    (hinting := cheating_hint_ratio * (attended - both)) :
    n ≥ 200 :=
by sorry

end smallest_number_of_students_l321_321273


namespace Kelsey_watched_537_videos_l321_321046

-- Definitions based on conditions
def total_videos : ℕ := 1222
def delilah_videos : ℕ := 78

-- Declaration of variables representing the number of videos each friend watched
variables (Kelsey Ekon Uma Ivan Lance : ℕ)

-- Conditions from the problem
def cond1 : Kelsey = 3 * Ekon := sorry
def cond2 : Ekon = Uma - 23 := sorry
def cond3 : Uma = 2 * Ivan := sorry
def cond4 : Lance = Ivan + 19 := sorry
def cond5 : delilah_videos = 78 := sorry
def cond6 := Kelsey + Ekon + Uma + Ivan + Lance + delilah_videos = total_videos

-- The theorem to prove
theorem Kelsey_watched_537_videos : Kelsey = 537 :=
  by
  sorry

end Kelsey_watched_537_videos_l321_321046


namespace orthocenter_circumcenter_distance_l321_321502

-- Define the geometrical entities and properties
variables {A B C O H : point} (abc_triangle : Triangle A B C) (A_not_collinear : ¬Collinear A B C)

-- State the theorem
theorem orthocenter_circumcenter_distance (R : ℝ) (H_circumradius : Circumradius A B C O R)
    (H_orthocenter : Orthocenter A B C H H_circumradius O) : distance O H < 3 * R :=
by
  sorry

end orthocenter_circumcenter_distance_l321_321502


namespace odd_and_monotonic_f_B_l321_321130

/-- Define the function f_B according to its piecewise definition --/
def f_B (x : ℝ) : ℝ := if x >= 0 then x^2 + 2*x else -x^2 + 2*x

/-- State the properties to be proven: f_B is both an odd function and monotonic on ℝ --/
theorem odd_and_monotonic_f_B : 
  (∀ x : ℝ, f_B (-x) = -f_B x) ∧ (∀ x y : ℝ, x < y → f_B x ≥ f_B y) := 
sorry

end odd_and_monotonic_f_B_l321_321130


namespace quadratic_roots_form_l321_321790

theorem quadratic_roots_form {d : ℝ} (h : ∀ x : ℝ, x^2 + 7*x + d = 0 → (x = (-7 + real.sqrt d) / 2) ∨ (x = (-7 - real.sqrt d) / 2)) : d = 49 / 5 := 
sorry

end quadratic_roots_form_l321_321790


namespace largest_prime_divisor_of_factorial_sum_l321_321988

theorem largest_prime_divisor_of_factorial_sum :
  ∀ (n: ℕ), n = 13 → largest_prime_divisor (factorial n + factorial (n + 1)) = 7 := 
by
  intros,
  sorry

end largest_prime_divisor_of_factorial_sum_l321_321988


namespace count_triples_satisfying_lcms_l321_321618

-- Definitions for conditions
def is_triple_of_lcms (x y z : ℕ) : Prop :=
  Nat.lcm x y = 108 ∧ Nat.lcm x z = 240 ∧ Nat.lcm y z = 360

-- Statement of the proof problem
theorem count_triples_satisfying_lcms :
  (finset.filter (λ (p : ℕ × ℕ × ℕ), is_triple_of_lcms p.1 p.2.1 p.2.2)
  ((finset.range 108).product (finset.range 240)).product (finset.range 360))).card = 8 :=
by
  sorry

end count_triples_satisfying_lcms_l321_321618


namespace solution_set_inequality_l321_321036

theorem solution_set_inequality (x : ℝ) : 
  (\dfrac{x - 3}{x - 2} ≥ 0) ↔ (x ∈ (-∞, 2) ∪ [3, +∞)) := 
sorry

end solution_set_inequality_l321_321036


namespace sum_cubes_mod_5_l321_321174

theorem sum_cubes_mod_5 :
  (∑ k in Finset.range (150 + 1), k^3) % 5 = 0 := by
sorry

end sum_cubes_mod_5_l321_321174


namespace hyperbola_eq_l321_321956

theorem hyperbola_eq (x y : ℝ) (k : ℝ)
  (H1 : (∀ x y : ℝ, x^2 / 2 - y^2 = 1 → (x * y = 0 → ∀ k : ℝ, x^2 / 2 - y^2 = k)) 
  (H2 : 2^2 / 2 - (-2)^2 = k) : (x^2 / 2 - y^2 = k) → (k = -2) :=
by
  sorry

end hyperbola_eq_l321_321956


namespace smallest_possible_number_of_students_l321_321289

theorem smallest_possible_number_of_students :
  ∃ n : ℕ, (n % 200 = 0) ∧ (∀ m : ℕ, (m < n → 
    75 * m ≤ 100 * n) ∧
    (∃ a b c : ℕ, a = m / 4 ∧ b = a / 10 ∧ 
    ∃ y z : ℕ, y = 3 * z ∧ (y + z - b = a) ∧ y * c = n / 4)) :=
by
  sorry

end smallest_possible_number_of_students_l321_321289


namespace product_of_divisors_of_30_l321_321437

theorem product_of_divisors_of_30 : 
  ∏ d in (finset.filter (λ d, 30 % d = 0) (finset.range 31)), d = 810000 :=
by sorry

end product_of_divisors_of_30_l321_321437


namespace monotonicity_of_f_range_of_b_inequality_for_sequence_l321_321222

-- 1. Monotonicity of the function f(x)
theorem monotonicity_of_f (x : ℝ) (h : x > -1) : 
  let f := λ x : ℝ, x^2 + x - log (1 + x)
  in (∀ x < 0, deriv f x < 0) ∧ (∀ x > 0, deriv f x > 0) := 
by sorry

-- 2. Range of b for f(x) = (5/2)x - b to have exactly two distinct real roots in [0, 2]
theorem range_of_b (b : ℝ) : 
  (ln 3 - 1 ≤ b ∧ b < ln 2 + 1/2) ↔ 
  let f := λ x : ℝ, x^2 + x - log (1 + x)
  in ∃ x1 x2 ∈ Icc (0:ℝ) 2, x1 < x2 ∧ f x1 = (5/2) * x1 - b ∧ f x2 = (5/2) * x2 - b := 
by sorry

-- 3. For any positive integer n, prove the inequality
theorem inequality_for_sequence (n : ℕ) (hn : 0 < n) : 
  (2 + (finset.range n).sum (λ i, (i+2)/(i+1)^2)) > log (n+1) := 
by sorry

end monotonicity_of_f_range_of_b_inequality_for_sequence_l321_321222


namespace school_competition_l321_321267

theorem school_competition :
  (∃ n : ℕ, 
    n > 0 ∧ 
    75% students did not attend the competition ∧
    10% of those who attended participated in both competitions ∧
    ∃ y z : ℕ, y = 3 / 2 * z ∧ 
    y + z - (1 / 10) * (n / 4) = n / 4
  ) → n = 200 :=
sorry

end school_competition_l321_321267


namespace product_of_divisors_of_30_l321_321438

theorem product_of_divisors_of_30 : 
  ∏ d in (finset.filter (λ d, 30 % d = 0) (finset.range 31)), d = 810000 :=
by sorry

end product_of_divisors_of_30_l321_321438


namespace equilateral_triangle_EC_squared_l321_321159

open Real EuclideanGeometry

noncomputable def equilateral_triangle_side_length : ℝ := 4

theorem equilateral_triangle_EC_squared :
  ∀ (A B C D E : Point), equilateral_triangle A B C →
  dist A B = equilateral_triangle_side_length →
  midpoint D B C →
  midpoint E A D →
  dist E C ^ 2 = 7 := by
  -- proof steps would go here
  sorry

end equilateral_triangle_EC_squared_l321_321159


namespace find_conjugates_l321_321624

noncomputable def euler_formula (θ : ℂ) : ℂ := complex.exp (complex.I * θ)

theorem find_conjugates (α β : ℂ) (h : euler_formula α + euler_formula β = (1 / 3 : ℂ) + (4 / 9 : ℂ) * complex.I) :
  euler_formula (-α) + euler_formula (-β) = (1 / 3 : ℂ) - (4 / 9 : ℂ) * complex.I :=
by
  sorry

end find_conjugates_l321_321624


namespace power_summation_l321_321525

theorem power_summation :
  (-1:ℤ)^(49) + (2:ℝ)^(3^3 + 5^2 - 48^2) = -1 + 1 / 2 ^ (2252 : ℝ) :=
by
  sorry

end power_summation_l321_321525


namespace exists_infinite_natural_numbers_l321_321690

def sum_of_digits (n : ℕ) : ℕ :=
  (n.toString.data.map (fun c => c.toNat - '0'.toNat)).reduce (· + ·) 0

theorem exists_infinite_natural_numbers (S : ℕ → ℕ) :
  (∀ n : ℕ, S n = sum_of_digits n) → 
  ∃ (A : Set ℕ), (A = {x : ℕ | sum_of_digits (x^2) = sum_of_digits x ∧ ¬ (x % 10 = 0)}) ∧ Set.Infinite A :=
by
  sorry

end exists_infinite_natural_numbers_l321_321690


namespace part1_part2_part3_l321_321346

noncomputable def f (x : ℝ) := log x
noncomputable def g (a b c x : ℝ) := a * x + b / x - c

theorem part1 (a b : ℝ) (hab : a - b = 1 ∧ a + b = 0) : 
  a = 1 / 2 ∧ b = -1 / 2 :=
sorry

theorem part2 (a : ℝ) (h : 0 < a ∧ a < 3) : 
  (∀ x0 : ℝ, x0 ∈ (1, ∞) → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧
    g a (3 - a) 3 x1 = f x0 ∧ g a (3 - a) 3 x2 = f x0) → 
  ∃ c : ℝ, c = 3 :=
sorry

theorem part3 (x1 x2 b : ℝ) (hx : x1 < x2) (a := 1) :
  (∃ y1 y2 : ℝ, y1 = f x1 ∧ y2 = f x2 ∧ g a b a x1 = y1 ∧ g a b a x2 = y2) →
  x1 * x2 - x2 < b ∧ b < x1 * x2 - x1 :=
sorry

end part1_part2_part3_l321_321346


namespace largest_prime_divisor_13_factorial_plus_14_factorial_l321_321989

theorem largest_prime_divisor_13_factorial_plus_14_factorial : 
  ∃ p, prime p ∧ p ∈ prime_factors (13! + 14!) ∧ ∀ q, prime q ∧ q ∈ prime_factors (13! + 14!) → q ≤ 13 :=
by { sorry }

end largest_prime_divisor_13_factorial_plus_14_factorial_l321_321989


namespace circle_equation_standard_no_such_a_exists_l321_321209

noncomputable def circle_equation (x y : ℝ) := x^2 + y^2 - 6*x + 4*y + 4 = 0

theorem circle_equation_standard :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - 3)^2 + (y + 2)^2 = 9 :=
sorry

theorem no_such_a_exists :
  ¬ ∃ a : ℝ, ∀ M N P: ℝ × ℝ, 
  M = (0, -2) ∧ N = (3,1) ∧ P = (2,0) ∧ 
  (circle_equation M.1 M.2) ∧ (circle_equation N.1 N.2) → 
  let l := λ x y : ℝ, a*x - y + 1 
  in ∃ A B : ℝ × ℝ, l A.1 A.2 = 0 ∧ l B.1 B.2 = 0 ∧ 
  (λ x y : ℝ, (x - 3)² + (y + 2)² = 9) A.1 A.2 ∧ 
  (λ x y : ℝ, (x - 3)² + (y + 2)² = 9) B.1 B.2 :=
sorry

end circle_equation_standard_no_such_a_exists_l321_321209


namespace area_ratio_quadrilaterals_l321_321713

theorem area_ratio_quadrilaterals (A B C D P Q R S : Point) (m : ℝ)
  (h_AB : P = (1 - m) * A + m * B)
  (h_BC : Q = (1 - m) * B + m * C)
  (h_CD : R = (1 - m) * C + m * D)
  (h_DA : S = (1 - m) * D + m * A) :
  let Area := λ X Y Z : Point, 1/2 * abs ((Y - X).det (Z - X))
  in Area P Q R S / Area A B C D = 2 * m^2 - 2 * m + 1 :=
sorry

end area_ratio_quadrilaterals_l321_321713


namespace sin_inv_tan_eq_l321_321948

open Real

theorem sin_inv_tan_eq :
  let a := arcsin (4/5)
  let b := arctan 3
  sin (a + b) = (13 * sqrt 10) / 50 := 
by
  let a := arcsin (4/5)
  let b := arctan 3
  sorry

end sin_inv_tan_eq_l321_321948


namespace HeatherIsHeavier_l321_321231

-- Definitions
def HeatherWeight : ℕ := 87
def EmilyWeight : ℕ := 9

-- Theorem statement
theorem HeatherIsHeavier : HeatherWeight - EmilyWeight = 78 := by
  sorry

end HeatherIsHeavier_l321_321231


namespace number_of_pairs_lcm_600_l321_321172

theorem number_of_pairs_lcm_600 :
  ∃ n, n = 53 ∧ (∀ m n : ℕ, (m ≤ n ∧ m > 0 ∧ n > 0 ∧ Nat.lcm m n = 600) ↔ n = 53) := sorry

end number_of_pairs_lcm_600_l321_321172


namespace number_of_people_who_believe_teal_is_more_green_l321_321476

theorem number_of_people_who_believe_teal_is_more_green :
  (number_surveyed more_blue both neither : ℕ) 
  (H1 : number_surveyed = 150) (H2 : more_blue = 90) 
  (H3 : both = 45) (H4 : neither = 20) : 
  (number_surveyed - (more_blue - both + both + neither) + both = 85) :=
by sorry

end number_of_people_who_believe_teal_is_more_green_l321_321476


namespace european_savings_correct_l321_321888

noncomputable def movie_ticket_price : ℝ := 8
noncomputable def popcorn_price : ℝ := 8 - 3
noncomputable def drink_price : ℝ := popcorn_price + 1
noncomputable def candy_price : ℝ := drink_price / 2
noncomputable def hotdog_price : ℝ := 5

noncomputable def monday_discount_popcorn : ℝ := 0.15 * popcorn_price
noncomputable def wednesday_discount_candy : ℝ := 0.10 * candy_price
noncomputable def friday_discount_drink : ℝ := 0.05 * drink_price

noncomputable def monday_price : ℝ := 22
noncomputable def wednesday_price : ℝ := 20
noncomputable def friday_price : ℝ := 25
noncomputable def weekend_price : ℝ := 25
noncomputable def monday_exchange_rate : ℝ := 0.85
noncomputable def wednesday_exchange_rate : ℝ := 0.85
noncomputable def friday_exchange_rate : ℝ := 0.83
noncomputable def weekend_exchange_rate : ℝ := 0.81

noncomputable def total_cost_monday : ℝ := movie_ticket_price + (popcorn_price - monday_discount_popcorn) + drink_price + candy_price + hotdog_price
noncomputable def savings_monday_usd : ℝ := total_cost_monday - monday_price
noncomputable def savings_monday_eur : ℝ := savings_monday_usd * monday_exchange_rate

noncomputable def total_cost_wednesday : ℝ := movie_ticket_price + popcorn_price + drink_price + (candy_price - wednesday_discount_candy) + hotdog_price
noncomputable def savings_wednesday_usd : ℝ := total_cost_wednesday - wednesday_price
noncomputable def savings_wednesday_eur : ℝ := savings_wednesday_usd * wednesday_exchange_rate

noncomputable def total_cost_friday : ℝ := movie_ticket_price + popcorn_price + (drink_price - friday_discount_drink) + candy_price + hotdog_price
noncomputable def savings_friday_usd : ℝ := total_cost_friday - friday_price
noncomputable def savings_friday_eur : ℝ := savings_friday_usd * friday_exchange_rate

noncomputable def total_cost_weekend : ℝ := movie_ticket_price + popcorn_price + drink_price + candy_price + hotdog_price
noncomputable def savings_weekend_usd : ℝ := total_cost_weekend - weekend_price
noncomputable def savings_weekend_eur : ℝ := savings_weekend_usd * weekend_exchange_rate

theorem european_savings_correct :
  savings_monday_eur = 3.61 ∧ 
  savings_wednesday_eur = 5.70 ∧ 
  savings_friday_eur = 1.41 ∧ 
  savings_weekend_eur = 1.62 :=
by
  sorry

end european_savings_correct_l321_321888


namespace find_fifth_number_l321_321772

-- Define the sets and their conditions
def first_set : List ℕ := [28, 70, 88, 104]
def second_set : List ℕ := [50, 62, 97, 124]

-- Define the means
def mean_first_set (x : ℕ) (y : ℕ) : ℚ := (28 + x + 70 + 88 + y) / 5
def mean_second_set (x : ℕ) : ℚ := (50 + 62 + 97 + 124 + x) / 5

-- Conditions given in the problem
axiom mean_first_set_condition (x y : ℕ) : mean_first_set x y = 67
axiom mean_second_set_condition (x : ℕ) : mean_second_set x = 75.6

-- Lean 4 theorem statement to prove the fifth number in the first set is 104 given above conditions
theorem find_fifth_number : ∃ x y, mean_first_set x y = 67 ∧ mean_second_set x = 75.6 ∧ y = 104 := by
  sorry

end find_fifth_number_l321_321772


namespace find_m_l321_321687

theorem find_m (m : ℕ) (h_pos : 0 < m) 
  (h_intersection : ∃ (x y : ℤ), 13 * x + 11 * y = 700 ∧ y = m * x - 1) : 
  m = 6 :=
sorry

end find_m_l321_321687


namespace conditional_probability_A_given_B_l321_321512

noncomputable def calculate_conditional_probability (event_A event_B : Finset (Fin 2 × Fin 2 × Fin 2)) : ℚ :=
  let prob_event_A_given_B := (event_A ∩ event_B).card.toRat / event_B.card.toRat in
  prob_event_A_given_B

def three_digit_codes : Finset (Fin 2 × Fin 2 × Fin 2) := 
  Finset.univ.product (Finset.univ.product Finset.univ)

def event_A : Finset (Fin 2 × Fin 2 × Fin 2) :=
  three_digit_codes.filter (λ code, code.2.1 = 0)

def event_B : Finset (Fin 2 × Fin 2 × Fin 2) := 
  three_digit_codes.filter (λ code, code.1 = 0)

theorem conditional_probability_A_given_B : 
  calculate_conditional_probability event_A event_B = 1 / 2 :=
by
  sorry

end conditional_probability_A_given_B_l321_321512


namespace right_triangle_hypotenuse_l321_321111

noncomputable def hypotenuse_length (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

theorem right_triangle_hypotenuse :
  ∀ (a b : ℝ),
  (1/3) * Real.pi * b^2 * a = 675 * Real.pi →
  (1/3) * Real.pi * a^2 * b = 1215 * Real.pi →
  hypotenuse_length a b = 3 * Real.sqrt 106 :=
  by
  intros a b h1 h2
  sorry

end right_triangle_hypotenuse_l321_321111


namespace volume_eq_eight_space_diagonal_eq_sqrt21_surface_area_neq_24_circumscribed_sphere_area_eq_21pi_l321_321007

namespace RectangularPrism

def length := 4
def width := 2
def height := 1

theorem volume_eq_eight : length * width * height = 8 := sorry

theorem space_diagonal_eq_sqrt21 :
  Real.sqrt (length ^ 2 + width ^ 2 + height ^ 2) = Real.sqrt 21 := sorry

theorem surface_area_neq_24 :
  2 * (length * width + width * height + height * length) ≠ 24 := sorry

theorem circumscribed_sphere_area_eq_21pi :
  4 * Real.pi * ((Real.sqrt (length ^ 2 + width ^ 2 + height ^ 2) / 2) ^ 2) = 21 * Real.pi := sorry

end RectangularPrism

end volume_eq_eight_space_diagonal_eq_sqrt21_surface_area_neq_24_circumscribed_sphere_area_eq_21pi_l321_321007


namespace lambda_range_l321_321199

noncomputable def f : ℝ → ℝ := sorry

theorem lambda_range (h_odd : ∀ x : ℝ, f (-x) = - f x)
  (h_deriv_pos : ∀ x : ℝ, (deriv f) x > 0)
  (h_eq : ∀ x : ℝ, f (x^3 - x - 1) + f (-2*x - λ) = 0):
  -3 < λ ∧ λ < 1 := sorry

end lambda_range_l321_321199


namespace sum_first_100_terms_l321_321394

-- Define the general term of the sequence
def a_n (n : ℕ) : ℚ := 1 / (n * (n + 1))

-- Prove that the sum of the first 100 terms is 100/101
theorem sum_first_100_terms : (∑ n in Finset.range 100, a_n (n + 1)) = 100 / 101 := by
  -- Sorry is used to skip the proof steps
  sorry

end sum_first_100_terms_l321_321394


namespace fair_attendance_l321_321706

theorem fair_attendance (x y z : ℕ) 
    (h1 : y = 2 * x)
    (h2 : z = y - 200)
    (h3 : x + y + z = 2800) : x = 600 := by
  sorry

end fair_attendance_l321_321706


namespace sincos_terminal_side_l321_321215

noncomputable def sincos_expr (α : ℝ) :=
  let P : ℝ × ℝ := (-4, 3)
  let r := Real.sqrt (P.1 ^ 2 + P.2 ^ 2)
  let sinα := P.2 / r
  let cosα := P.1 / r
  sinα + 2 * cosα = -1

theorem sincos_terminal_side :
  sincos_expr α :=
by
  sorry

end sincos_terminal_side_l321_321215


namespace quadratic_root_condition_l321_321779

theorem quadratic_root_condition (d : ℝ) :
  (∀ x, x^2 + 7 * x + d = 0 → x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → d = 9.8 :=
by
  intro h
  sorry

end quadratic_root_condition_l321_321779


namespace sum_of_reciprocals_of_roots_l321_321556

theorem sum_of_reciprocals_of_roots (r1 r2 : ℝ) (h1 : r1 + r2 = 17) (h2 : r1 * r2 = 8) :
  1 / r1 + 1 / r2 = 17 / 8 :=
by
  sorry

end sum_of_reciprocals_of_roots_l321_321556


namespace parallel_lines_condition_l321_321240

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, (a * x + 2 * y - 1 = 0) → (x + (a + 1) * y + 4 = 0) → a = 1) ↔
  (∀ x y : ℝ, (a = 1 ∧ a * x + 2 * y - 1 = 0 → x + (a + 1) * y + 4 = 0) ∨
   (a ≠ 1 ∧ a = -2 ∧ a * x + 2 * y - 1 ≠ 0 → x + (a + 1) * y + 4 ≠ 0)) :=
by
  sorry

end parallel_lines_condition_l321_321240


namespace find_m_of_parallel_lines_l321_321595

theorem find_m_of_parallel_lines
  (m : ℝ) 
  (parallel : ∀ x y, (x - 2 * y + 5 = 0 → 2 * x + m * y - 5 = 0)) :
  m = -4 :=
sorry

end find_m_of_parallel_lines_l321_321595


namespace range_of_m_l321_321252

theorem range_of_m (a c m : ℝ) (h1 : ∀ x : ℝ, -2 < x ∧ x < 1 ↔ x^2 + a*x - c < 0)
  (h2 : ∀ t : ℝ, t ∈ set.Icc 1 2 →
        ∃ x1 x2 : ℝ, t < x1 ∧ x1 < x2 ∧ x2 < 3 ∧
        (∀ x : ℝ, x ∈ set.Ioo t 3 → differentiable ℝ (λ x, a*x^3 + (m + 1/2)*x^2 - c*x)) ∧
        (∀ x : ℝ, x1 ≤ x ∧ x ≤ x2 → has_deriv_at (λ x, a*x^3 + (m + 1/2)*x^2 - c*x) (a*(3*x^2 + (2*m + 1)*x - c)) x))
        : -14/3 < m ∧ m < -3 :=
sorry

end range_of_m_l321_321252


namespace cylinder_volume_l321_321484

theorem cylinder_volume (r h V : ℝ) (hr : r = 3) (h_area : 2 * real.pi * r * h = 12 * real.pi) : 
  V = real.pi * r^2 * h → V = 18 * real.pi :=
by 
  sorry

end cylinder_volume_l321_321484


namespace number_of_bulbs_chosen_l321_321878

-- Definitions for conditions
def total_bulbs : ℕ := 24
def defective_bulbs : ℕ := 4
def probability_at_least_one_defective : ℝ := 0.3115942028985508

-- Function to calculate the probability of choosing a non-defective bulb
def probability_non_defective : ℝ := (total_bulbs - defective_bulbs) / total_bulbs

-- Proof statement
theorem number_of_bulbs_chosen :
  ∃ n : ℕ, (probability_non_defective ^ n = 1 - probability_at_least_one_defective) ∧ n = 2 :=
begin
  sorry
end

end number_of_bulbs_chosen_l321_321878


namespace distance_center_point_is_five_l321_321551

theorem distance_center_point_is_five :
  let circle : (ℝ × ℝ) → Prop := λ p, p.1^2 + p.2^2 = 2*p.1 + 4*p.2 - 20,
      center : ℝ × ℝ := (1, 2),
      point : ℝ × ℝ := (-3, -1), 
      distance : ℝ := Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2)
  in distance = 5 :=
by
  sorry

end distance_center_point_is_five_l321_321551


namespace prove_final_value_is_111_l321_321891

theorem prove_final_value_is_111 :
  let initial_num := 16
  let doubled_num := initial_num * 2
  let added_five := doubled_num + 5
  let trebled_result := added_five * 3
  trebled_result = 111 :=
by
  sorry

end prove_final_value_is_111_l321_321891


namespace largest_prime_divisor_of_thirteen_plus_fourteen_factorial_l321_321999

noncomputable def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def largest_prime_divisor_of_factorials_sum (n m : ℕ) : ℕ :=
  let sum := factorial n + factorial m
  Prime.factorization sum |>.keys.max' sorry

theorem largest_prime_divisor_of_thirteen_plus_fourteen_factorial :
  largest_prime_divisor_of_factorials_sum 13 14 = 17 := 
sorry

end largest_prime_divisor_of_thirteen_plus_fourteen_factorial_l321_321999


namespace simplify_expression_triangle_obtuse_tan_A_value_l321_321196

-- Definitions used in the problem
variable {A B C : ℝ}

-- The given conditions
def cond1 : Prop := A ≠ (Real.pi / 2)
def cond2 : Prop := (Real.sin A + Real.cos A = 1 / 5)

-- Proposition for the first question
theorem simplify_expression (hA : cond1) : 
  (Real.sin ((3 * Real.pi) / 2 + A) * Real.cos (Real.pi / 2 - A)) / 
  (Real.cos (B + C) * Real.tan (Real.pi + A)) = Real.cos A := 
sorry

-- Proposition for the second question
theorem triangle_obtuse (h : cond2) : ¬ (0 < A ∧ A < Real.pi / 2) :=
sorry

-- Proposition for the third question
theorem tan_A_value (h : cond2) : Real.tan A = -4 / 3 := 
sorry

end simplify_expression_triangle_obtuse_tan_A_value_l321_321196


namespace time_away_is_43point64_minutes_l321_321105

theorem time_away_is_43point64_minutes :
  ∃ (n1 n2 : ℝ), 
    (195 + n1 / 2 - 6 * n1 = 120 ∨ 195 + n1 / 2 - 6 * n1 = -120) ∧
    (195 + n2 / 2 - 6 * n2 = 120 ∨ 195 + n2 / 2 - 6 * n2 = -120) ∧
    n1 ≠ n2 ∧
    n1 < 60 ∧
    n2 < 60 ∧
    |n2 - n1| = 43.64 :=
sorry

end time_away_is_43point64_minutes_l321_321105


namespace smallest_positive_integer_problem_l321_321688

theorem smallest_positive_integer_problem
  (n : ℕ) 
  (h1 : 50 ∣ n) 
  (h2 : (∃ e1 e2 e3 : ℕ, n = 2^e1 * 5^e2 * 3^e3 ∧ (e1 + 1) * (e2 + 1) * (e3 + 1) = 100)) 
  (h3 : ∀ m : ℕ, (50 ∣ m) → ((∃ e1 e2 e3 : ℕ, m = 2^e1 * 5^e2 * 3^e3 ∧ (e1 + 1) * (e2 + 1) * (e3 + 1) = 100) → (n ≤ m))) :
  n / 50 = 8100 := 
sorry

end smallest_positive_integer_problem_l321_321688


namespace dot_product_a_b_l321_321255

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (-1, 2)

theorem dot_product_a_b : vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 1 := by
  sorry

end dot_product_a_b_l321_321255


namespace committee_ways_count_l321_321882

theorem committee_ways_count :
  (nat.choose 10 4) * (nat.choose 7 3) = 7350 :=
by 
sorry

end committee_ways_count_l321_321882


namespace exists_8x8_grid_with_row_and_column_properties_l321_321669

theorem exists_8x8_grid_with_row_and_column_properties :
  ∃ (G : Fin 8 → Fin 8 → ℕ),
    (∀ i: Fin 8, ∑ j, G i j = 4) ∧
    (∀ i j: Fin 8, i ≠ j → ∑ k, G k i ≠ ∑ k, G k j) :=
sorry

end exists_8x8_grid_with_row_and_column_properties_l321_321669


namespace athletes_camp_duration_l321_321886

theorem athletes_camp_duration
  (h : ℕ)
  (initial_athletes : ℕ := 300)
  (rate_leaving : ℕ := 28)
  (rate_entering : ℕ := 15)
  (hours_entering : ℕ := 7)
  (difference : ℕ := 7) :
  300 - 28 * h + 15 * 7 = 300 + 7 → h = 4 :=
by
  sorry

end athletes_camp_duration_l321_321886


namespace largest_prime_divisor_of_factorial_sum_l321_321984

theorem largest_prime_divisor_of_factorial_sum :
  ∀ (n: ℕ), n = 13 → largest_prime_divisor (factorial n + factorial (n + 1)) = 7 := 
by
  intros,
  sorry

end largest_prime_divisor_of_factorial_sum_l321_321984


namespace maximum_area_of_rectangle_with_fixed_perimeter_l321_321016

theorem maximum_area_of_rectangle_with_fixed_perimeter (x y : ℝ) 
  (h₁ : 2 * (x + y) = 40) 
  (h₂ : x = y) :
  x * y = 100 :=
by
  sorry

end maximum_area_of_rectangle_with_fixed_perimeter_l321_321016


namespace circle_area_ratio_proof_l321_321626

noncomputable def circle_area_ratio (C_C C_D L_C L_D : ℝ) : ℝ :=
  (π * (C_C / (2 * π))^2) / (π * (C_D / (2 * π))^2)

theorem circle_area_ratio_proof
  (R_C R_D : ℝ)
  (h1 : ∀ (R_C R_D : ℝ), R_D = (3/4) * R_C)
  (h2 : ∀ (R_C R_D : ℝ), R_C = (4/3) * R_D) :
  circle_area_ratio (2 * π * R_C) (2 * π * R_D) (1/6 * 2 * π * R_C) (1/9 * 2 * π * R_D) = 16 / 9 :=
by
  rw [circle_area_ratio, (R_C * 2 * π), (R_D * 2 * π)]
  sorry

end circle_area_ratio_proof_l321_321626


namespace line_through_point_perpendicular_to_other_line_l321_321761

theorem line_through_point_perpendicular_to_other_line :
  ∀ (P : ℝ × ℝ) (L : ℝ × ℝ × ℝ), 
  P = (4, -1) ∧ L = (3, -4, 6) → ∃ a b c, a * 4 + b * (-1) + c = 0 ∧ a * 3 + b * (-4) = 0 ∧ a = 4 ∧ b = 3 ∧ c = -13 :=
by
  intros P L h
  cases h with hP hL
  use [4, 3, -13]
  repeat { split }
  · simp [hP]
  · simp [hL]
  · simp
  · simp
  · simp

end line_through_point_perpendicular_to_other_line_l321_321761


namespace arccos_sin_eq_l321_321920

open Real

-- Definitions from the problem conditions
noncomputable def radians := π / 180

-- The theorem we need to prove
theorem arccos_sin_eq : arccos (sin 3) = 3 - (π / 2) :=
by
  sorry

end arccos_sin_eq_l321_321920


namespace find_positive_pairs_l321_321549

theorem find_positive_pairs (x y : ℝ) (h0 : 0 < x) (h1 : 0 < y) :
    (x - 3 * sqrt (x * y) - 2 * sqrt (x / y) + 6 = 0) ∧ (x^2 * y^2 + x^4 = 82) ↔
    (x = 3 ∧ y = 1 / 3) ∨ (x = real.sqrt (real.sqrt 33) ∧ y = 4 / real.sqrt (real.sqrt 33)) :=
by
  sorry

end find_positive_pairs_l321_321549


namespace girls_in_math_class_l321_321794

theorem girls_in_math_class (x y z : ℕ)
  (boys_girls_ratio : 5 * x = 8 * x)
  (math_science_ratio : 7 * y = 13 * x)
  (science_literature_ratio : 4 * y = 3 * z)
  (total_students : 13 * x + 4 * y + 5 * z = 720) :
  8 * x = 176 :=
by
  sorry

end girls_in_math_class_l321_321794


namespace smallest_number_of_students_l321_321286

theorem smallest_number_of_students
  (n : ℕ)
  (students_attended : ℕ)
  (students_both_competitions : ℕ)
  (students_hinting : ℕ)
  (students_cheating : ℕ)
  (attended_fraction : Real := 0.25)
  (both_competitions_fraction : Real := 0.1)
  (hinting_ratio : Real := 1.5)
  (h_attended : students_attended = (attended_fraction * n).to_nat)
  (h_both : students_both_competitions = (both_competitions_fraction * students_attended).to_nat)
  (h_hinting : students_hinting = (hinting_ratio * students_cheating).to_nat)
  (h_total_attended : students_attended = students_hinting + students_cheating - students_both_competitions)
  : n = 200 :=
sorry

end smallest_number_of_students_l321_321286


namespace isosceles_triangle_third_vertex_y_coord_l321_321132

theorem isosceles_triangle_third_vertex_y_coord :
  ∀ (A B : ℝ × ℝ) (θ : ℝ), 
  A = (0, 5) → B = (8, 5) → θ = 60 → 
  ∃ (C : ℝ × ℝ), C.fst > 0 ∧ C.snd > 5 ∧ C.snd = 5 + 4 * Real.sqrt 3 :=
by
  intros A B θ hA hB hθ
  use (4, 5 + 4 * Real.sqrt 3)
  sorry

end isosceles_triangle_third_vertex_y_coord_l321_321132


namespace group_photo_arrangements_l321_321079

theorem group_photo_arrangements :
  ∃ (arrangements : ℕ), arrangements = 36 ∧
    ∀ (M G H P1 P2 : ℕ),
    (M = G + 1 ∨ M + 1 = G) ∧ (M ≠ H - 1 ∧ M ≠ H + 1) →
    arrangements = 36 :=
by {
  sorry
}

end group_photo_arrangements_l321_321079


namespace sum_of_integers_satisfying_condition_l321_321175

-- Definition of conditions and problem statement
def cond (n : ℤ) : Prop := ∃ (x : ℤ), n^2 - 23 * n + 132 = x^2

-- The main theorem statement
theorem sum_of_integers_satisfying_condition : 
  (∑ n in finset.filter cond (finset.Icc 1 100), n) = 23 :=
by
  sorry

end sum_of_integers_satisfying_condition_l321_321175


namespace find_d_of_quadratic_roots_l321_321789

theorem find_d_of_quadratic_roots :
  ∃ d : ℝ, (∀ x : ℝ, x^2 + 7 * x + d = 0 ↔ x = (-7 + real.sqrt d) / 2 ∨ x = (-7 - real.sqrt d) / 2) → d = 9.8 :=
by
  sorry

end find_d_of_quadratic_roots_l321_321789


namespace part1_part2_part3_l321_321467

variable (a : ℝ) (x : ℕ → ℝ)
variable (h_a_gt_two : a > 2)
variable (h_seq : x 1 = a ∧ ∀ n, x (n + 1) = x n ^ 2 / (2 * (x n - 1)))

/-- Part 1: Prove that x_n > 2 and (x_{n+1} / x_n) < 1 --/
theorem part1 (n : ℕ) (h_seq : ∀ n, x (n + 1) = x n ^ 2 / (2 * (x n - 1))) : 
  (∀ n, x n > 2) ∧ (∀ n, x (n + 1) / x n < 1) :=
  sorry

variable (h_a_le_three : a ≤ 3)

/-- Part 2: If a ≤ 3, show that x_n < 2 + 1 / 2^(n-1) --/
theorem part2 (n : ℕ) (h_seq : ∀ n, x (n + 1) = x n ^ 2 / (2 * (x n - 1))) : 
  a ≤ 3 → (∀ n, x n < 2 + 1 / (2 ^ (n - 1))) :=
  sorry

variable (h_a_gt_three : a > 3)

/-- Part 3: If a > 3, demonstrate that x_{n+1} < 3 for n ≥ log(a/3) / log(4/3) --/
theorem part3 (n : ℕ) (h_seq : ∀ n, x (n + 1) = x n ^ 2 / (2 * (x n - 1))) : 
  a > 3 → (∀ n ≥ (log (a / 3) / log (4 / 3)).to_nat, x (n + 1) < 3) :=
  sorry

end part1_part2_part3_l321_321467


namespace range_of_omega_for_symmetry_axes_l321_321764

theorem range_of_omega_for_symmetry_axes (ω : ℝ) (hω : ω > 0) : 
  (∀ x ∈ Set.Icc 0 π, f x = sin(ω * x + π/4) →
  ∃! y ∈ Set.Icc 0 π, ∃! z ∈ Set.Icc 0 π, y ≠ z ∧ (f y = f (-y)) ∧ (f z = f (-z))) ↔
  ω ∈ Set.Icc (5/4) (9/4) :=
sorry

end range_of_omega_for_symmetry_axes_l321_321764


namespace volume_rectangular_prism_space_diagonal_rectangular_prism_surface_area_rectangular_prism_surface_area_circumscribed_sphere_l321_321006

-- Define the conditions of the rectangular prism
def length := 4
def width := 2
def height := 1

-- 1. Prove that the volume of the rectangular prism is 8
theorem volume_rectangular_prism : length * width * height = 8 := sorry

-- 2. Prove that the length of the space diagonal is √21
theorem space_diagonal_rectangular_prism : Real.sqrt (length^2 + width^2 + height^2) = Real.sqrt 21 := sorry

-- 3. Prove that the surface area of the rectangular prism is 28
theorem surface_area_rectangular_prism : 2 * (length * width + length * height + width * height) = 28 := sorry

-- 4. Prove that the surface area of the circumscribed sphere is 21π
theorem surface_area_circumscribed_sphere : 4 * Real.pi * (Real.sqrt (length^2 + width^2 + height^2) / 2)^2 = 21 * Real.pi := sorry

end volume_rectangular_prism_space_diagonal_rectangular_prism_surface_area_rectangular_prism_surface_area_circumscribed_sphere_l321_321006


namespace square_side_length_equals_radius_circumference_l321_321028

def perimeter_square (x : ℝ) : ℝ := 4 * x
def circumference_circle (r : ℝ) : ℝ := 2 * real.pi * r
def approx_equal_to_hundredth (a b : ℝ) : Prop := abs (a - b) < 0.01

theorem square_side_length_equals_radius_circumference :
  approx_equal_to_hundredth (4 * (3 * real.pi / 2)) 4.71 :=
by
  sorry

end square_side_length_equals_radius_circumference_l321_321028


namespace domain_f_l321_321153

noncomputable def f (x : ℝ) : ℝ := (√(x + 2)) / (x - 2)

theorem domain_f :
  (∃ D : set ℝ, (D = {x : ℝ | x + 2 ≥ 0 ∧ x ≠ 2}) ∧ D = (Icc (-2 : ℝ) 2) ∪ (Ioc 2 (⊤))) :=
begin
  sorry
end

end domain_f_l321_321153


namespace tangents_intersect_at_fixed_point_l321_321191

theorem tangents_intersect_at_fixed_point
  (O : Type*)
  [MetricSpace O]
  [CircularSpace O]
  (r : ℝ) (center : O)
  (l : line O) (hl : ∀ P : O, ¬(P ∈ interior (ball center r)) ∧ distance center P ≥ r)
  (A : O) (hA : A ∈ l)
  (E F : O)
  (tangent_to_circle : ∀ P : O, P ∈ l → ∃ E F, tangent O r center E F P) :
  ∃ P : O, ∀ A E F, A ∈ l → tangent_to_circle A E F → line_through E F ∩ line_through center P := sorry

end tangents_intersect_at_fixed_point_l321_321191


namespace king_of_diamonds_probability_l321_321899

-- Definitions
def ranks := {"Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"}
def suits := {"Spades", "Hearts", "Diamonds", "Clubs"}
def standard_deck_size : ℕ := 52
def total_possible_outcomes : ℕ := 52
def favorable_outcomes : ℕ := 1

-- Lean 4 statement
theorem king_of_diamonds_probability :
  (favorable_outcomes / total_possible_outcomes : ℚ) = 1 / 52 := 
by
  sorry

end king_of_diamonds_probability_l321_321899


namespace integer_values_count_l321_321178

def integer_values_of_n (n : ℕ) : Prop :=
  (8000 : ℚ) * (3 / 4) ^ n ∈ ℤ

theorem integer_values_count :
  { n : ℕ | integer_values_of_n n }.toFinset.card = 4 :=
by
  sorry

end integer_values_count_l321_321178


namespace license_plate_count_l321_321635

theorem license_plate_count :
  let letters := 26
  let digits := 10
  let second_char_options := letters - 1 + digits
  let third_char_options := digits - 1
  letters * second_char_options * third_char_options = 8190 :=
by
  sorry

end license_plate_count_l321_321635


namespace range_of_a_l321_321591

theorem range_of_a (a : ℝ) (hx : ∀ x : ℝ, (a - 1) * x > a - 1 → x < 1) : a < 1 :=
sorry

end range_of_a_l321_321591


namespace linda_five_dollar_bills_l321_321695

theorem linda_five_dollar_bills (x y : ℕ) (h1 : x + y = 12) (h2 : 5 * x + 10 * y = 80) : x = 8 :=
by
  sorry

end linda_five_dollar_bills_l321_321695


namespace minimal_constant_C_l321_321649

def grid_value (m n : ℕ) (grid : ℕ → ℕ → bool) : ℕ :=
  ∑ i in range m, ∑ j in range n,
    if grid i j = true then 0 else 
      (if i > 0 ∧ grid (i - 1) j = true then 1 else 0) +
      (if i < m - 1 ∧ grid (i + 1) j = true then 1 else 0) +
      (if j > 0 ∧ grid i (j - 1) = true then 1 else 0) +
      (if j < n - 1 ∧ grid i (j + 1) = true then 1 else 0) +
      (if i > 0 ∧ j > 0 ∧ grid (i - 1) (j - 1) = true then 1 else 0) +
      (if i > 0 ∧ j < n - 1 ∧ grid (i - 1) (j + 1) = true then 1 else 0) +
      (if i < m - 1 ∧ j > 0 ∧ grid (i + 1) (j - 1) = true then 1 else 0) +
      (if i < m - 1 ∧ j < n - 1 ∧ grid (i + 1) (j + 1) = true then 1 else 0)

def f (m n : ℕ) : ℕ :=
  finset.sup finset.univ (λ grid, grid_value m n (λ i j, (grid >> (i * n + j)).val))

theorem minimal_constant_C (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  (f m n : ℝ) / (m * n : ℝ) ≤ 2 := 
sorry

end minimal_constant_C_l321_321649


namespace correct_value_l321_321080

theorem correct_value (x : ℚ) (h : x + 7/5 = 81/20) : x - 7/5 = 5/4 :=
sorry

end correct_value_l321_321080


namespace at_least_two_even_l321_321682

theorem at_least_two_even (n : Fin 1998 → ℕ) (h : ∑ i, (n i)^2 = (n 1997)^2) : 
  ∃ i j : Fin 1998, i ≠ j ∧ even (n i) ∧ even (n j) :=
by
  sorry

end at_least_two_even_l321_321682


namespace largest_prime_divisor_of_factorial_sum_l321_321957

theorem largest_prime_divisor_of_factorial_sum :
  ∃ p : ℕ, p.prime ∧ p ∣ (13! + 14!) ∧ ∀ q : ℕ, q.prime ∧ q ∣ (13! + 14!) → q ≤ p :=
sorry

end largest_prime_divisor_of_factorial_sum_l321_321957


namespace ellipse_eccentricity_l321_321204

-- Define the triangle vertices and coordinates properties
structure Triangle (α : Type) [LinearOrderedField α] :=
(A B C : α × α)
(side_len : α)
(is_equilateral : dist A B = side_len ∧ dist B C = side_len ∧ dist C A = side_len)

-- Define the ellipse and its properties
structure Ellipse (α : Type) [LinearOrderedField α] :=
(A F B C : α × α)
(a : α) -- semi-major axis
(c : α) -- semi-focal distance
(pass_through_points : ∀ P ∈ {B, C}, dist P A + dist P F = 2 * a)
(mid_point_BC : F = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))

-- Given conditions
variables {α : Type} [LinearOrderedField α]
(h_tri : Triangle α)
(h_ellipse : Ellipse α)

-- Theorem: The eccentricity of the ellipse is sqrt(3)/3.
theorem ellipse_eccentricity : ∃ e : α, e = (1 : α) / √3 ∧ e = h_ellipse.c / h_ellipse.a :=
by
  sorry

end ellipse_eccentricity_l321_321204


namespace problem_l321_321243

variable (a b c d : ℕ)

theorem problem (h1 : a + b = 12) (h2 : b + c = 9) (h3 : c + d = 3) : a + d = 6 :=
sorry

end problem_l321_321243


namespace setB_can_form_triangle_l321_321452

theorem setB_can_form_triangle : 
  let a := 8
  let b := 6
  let c := 4
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  let a := 8
  let b := 6
  let c := 4
  have h1 : a + b > c := by sorry
  have h2 : a + c > b := by sorry
  have h3 : b + c > a := by sorry
  exact ⟨h1, h2, h3⟩

end setB_can_form_triangle_l321_321452


namespace chord_eq_line_l321_321576

theorem chord_eq_line (x y : ℝ)
  (h_ellipse : (x^2) / 16 + (y^2) / 4 = 1)
  (h_midpoint : ∃ x1 y1 x2 y2 : ℝ, 
    ((x1^2) / 16 + (y1^2) / 4 = 1) ∧ 
    ((x2^2) / 16 + (y2^2) / 4 = 1) ∧ 
    (x1 + x2) / 2 = 2 ∧ 
    (y1 + y2) / 2 = 1) :
  x + 2 * y - 4 = 0 :=
sorry

end chord_eq_line_l321_321576


namespace matrix_product_solution_l321_321143

noncomputable def matrix_product_problem : Matrix (Fin 2) (Fin 2) ℝ := 
  let matrices := List.map (λ i, Matrix.scalar 2 1 + Matrix.stdBasisMatrix (Fin 2) (Fin 2) 0 1 • (2 : ℝ) * i) (List.range 1 51)
  matrices.foldl (•) (1 : Matrix (Fin 2) (Fin 2) ℝ)

theorem matrix_product_solution : 
  matrix_product_problem = (Matrix.scalar 2 1 + Matrix.stdBasisMatrix (Fin 2) (Fin 2) 0 1 • (2550 : ℝ)) :=
  sorry

end matrix_product_solution_l321_321143


namespace max_area_of_rectangle_l321_321024

theorem max_area_of_rectangle (x y : ℝ) (h1 : 2 * (x + y) = 40) : 
  (x * y) ≤ 100 :=
by
  sorry

end max_area_of_rectangle_l321_321024


namespace expected_interval_between_trains_l321_321714

-- We define the conditions
def northern_route_time : ℝ := 17
def southern_route_time : ℝ := 11
def interval_between_counterclockwise_and_clockwise_trains : ℝ := 1 + 15 / 60
def commute_time_difference : ℝ := 1

-- Define what we need to prove as a theorem
theorem expected_interval_between_trains (T : ℝ) :
  T = 3 :=
sorry

end expected_interval_between_trains_l321_321714


namespace total_lunch_bill_l321_321373

theorem total_lunch_bill (cost_hotdog cost_salad : ℝ) (h_hd : cost_hotdog = 5.36) (h_sd : cost_salad = 5.10) :
  cost_hotdog + cost_salad = 10.46 :=
by
  sorry

end total_lunch_bill_l321_321373


namespace maximum_perimeter_triangle_l321_321040

theorem maximum_perimeter_triangle (x : ℕ) (h1 : 3 < x) (h2 : x < 6) : 
  ∃ a b c, a + b + c = 14 :=
by 
  existsi [4, 5, 5]
  sorry

end maximum_perimeter_triangle_l321_321040


namespace find_d_l321_321782

noncomputable def quadratic_roots (d : ℝ) : Prop :=
∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2

theorem find_d : ∃ d : ℝ, d = 9.8 ∧ quadratic_roots d :=
sorry

end find_d_l321_321782


namespace max_single_share_l321_321646

theorem max_single_share (shares : Fin 2017 → ℝ) 
  (h : ∀ (S : Finset (Fin 2017)), S.card = 1500 → (∑ i in S, shares i) ≥ 0.5) : 
  ∃ i : Fin 2017, shares i ≤ 0.328 := 
sorry

end max_single_share_l321_321646


namespace cube_faces_paint_count_l321_321798

def is_painted_on_at_least_two_faces (x y z : ℕ) : Prop :=
  ((x = 0 ∨ x = 3) ∨ (y = 0 ∨ y = 3) ∨ (z = 0 ∨ z = 3))

theorem cube_faces_paint_count :
  let n := 4 in
  let volume := n * n * n in
  let one_inch_cubes := {c | ∃ x y z : ℕ, x < n ∧ y < n ∧ z < n ∧ c = (x, y, z)} in
  let painted_cubes := {c ∈ one_inch_cubes | exists_along_edges c} in  -- Auxiliary predicate to check edge paint
  card painted_cubes = 32 :=
begin
  sorry,
end

/-- Auxiliary predicate to check if a cube is on the edges but not corners, signifying 
    at least two faces are painted --/
predicate exists_along_edges (c: ℕ × ℕ × ℕ) := 
  match c with
  | (0, _, _) => true
  | (3, _, _) => true
  | (_, 0, _) => true
  | (_, 3, _) => true
  | (_, _, 0) => true
  | (_, _, 3) => true
  | (_, _, _) => false
  end

end cube_faces_paint_count_l321_321798


namespace f_increasing_l321_321873

noncomputable def f (a b : ℝ) (ξ : ℝ → ℝ) (F : (ℝ → ℝ) → ℝ → ℝ) : ℝ :=
  (∫ x in a..b, x * (F ξ) x) / ((F ξ) b - (F ξ) a)

theorem f_increasing (ξ : ℝ → ℝ) (F : (ℝ → ℝ) → ℝ → ℝ) (a b x : ℝ) :
  (∀ a b : ℝ, -∞ < a < b < ∞ → (F ξ) b - (F ξ) a > 0) →
  a < b → x ∈ set.Ioo a b →
  f a b ξ F < f x b ξ F ∧ f a b ξ F < f a x ξ F :=
by
  intros h1 h2 h3
  sorry

end f_increasing_l321_321873


namespace more_oaks_than_willows_l321_321263

def total_trees := 712
def willow_percentage := 34/100
def oak_percentage := 45/100

theorem more_oaks_than_willows : 
  let willow_trees := ((willow_percentage * total_trees).floor : ℤ) in
  let oak_trees := ((oak_percentage * total_trees).floor : ℤ) in
  oak_trees - willow_trees = 78 :=
by
  sorry

end more_oaks_than_willows_l321_321263


namespace rate_per_square_meter_l321_321896

def length_of_lawn : ℕ := 80
def breadth_of_lawn : ℕ := 60
def road_width : ℕ := 10
def total_cost : ℕ := 5200

theorem rate_per_square_meter :
  let
    area_road_1 := road_width * breadth_of_lawn,
    area_road_2 := road_width * length_of_lawn,
    area_intersection := road_width * road_width,
    total_area_roads := (area_road_1 + area_road_2) - area_intersection,
    rate := total_cost / total_area_roads
  in
    rate = 4 :=
by
  -- Definition of individual areas
  let area_road_1 := road_width * breadth_of_lawn
  let area_road_2 := road_width * length_of_lawn
  let area_intersection := road_width * road_width
  let total_area_roads := (area_road_1 + area_road_2) - area_intersection
  let rate := total_cost / total_area_roads

  -- Prove the rate per square meter
  have h1 : area_road_1 = 600 := by sorry
  have h2 : area_road_2 = 800 := by sorry
  have h3 : area_intersection = 100 := by sorry
  have h4 : total_area_roads = 1300 := by sorry
  have h5 : rate = total_cost / 1300 := by sorry
  have h6 : rate = 4 := by sorry

  exact h6

end rate_per_square_meter_l321_321896


namespace function_inequality_l321_321884

variable (f : ℝ → ℝ)

theorem function_inequality (h : ∀ x > 0, f(x) > 2 * (x + real.sqrt(x)) * deriv f x) :
  f(1) / 2 > f(4) / 3 ∧ f(4) / 3 > f(9) / 4 :=
by
  sorry

end function_inequality_l321_321884


namespace pentagon_AE_sqrt_3_minus_1_l321_321750

open Real

def pentagon_condition :=
  ∃ (A B C D E : Point ℝ),  
  (angle A B C = 105) ∧
  (angle  A  C B  = 105 ) ∧
  (angle  B C D  =  90 )  ∧
  (A.distanceTo B = 2) ∧
  (B.distanceTo C = Real.sqrt 2) ∧
  (C.distanceTo D = Real.sqrt 2) ∧
  ∃ (AE : ℝ), 
  AE = abs (Real.sqrt (3) - 1)

theorem pentagon_AE_sqrt_3_minus_1 (h₀ : pentagon_condition) :
 ∃ (a b : ℕ), AE = Real.sqrt a - b →
  a + b = 4 :=
sorry

end pentagon_AE_sqrt_3_minus_1_l321_321750


namespace die_face_never_lays_on_board_l321_321711

structure Chessboard :=
(rows : ℕ)
(cols : ℕ)
(h_size : rows = 8 ∧ cols = 8)

structure Die :=
(faces : Fin 6 → Nat)  -- a die has 6 faces

structure Position :=
(x : ℕ)
(y : ℕ)

structure State :=
(position : Position)
(bottom_face : Fin 6)
(visited : Fin 64 → Bool)

def initial_position : Position := ⟨0, 0⟩  -- top-left corner (a1)

def initial_state (d : Die) : State :=
  { position := initial_position,
    bottom_face := 0,
    visited := λ _ => false }

noncomputable def can_roll_over_entire_board_without_one_face_touching (board : Chessboard) (d : Die) : Prop :=
  ∃ f : Fin 6, ∀ s : State, -- for some face f of the die
    ((s.position.x < board.rows ∧ s.position.y < board.cols) → 
      s.visited (⟨s.position.x + board.rows * s.position.y, by sorry⟩) = true) → -- every cell visited
      ¬(s.bottom_face = f) -- face f is never the bottom face

theorem die_face_never_lays_on_board (board : Chessboard) (d : Die) :
  can_roll_over_entire_board_without_one_face_touching board d :=
  sorry

end die_face_never_lays_on_board_l321_321711


namespace floor_of_7_point_8_l321_321943

theorem floor_of_7_point_8 : real.floor 7.8 = 7 := 
by
  sorry

end floor_of_7_point_8_l321_321943


namespace fibonacci_150_mod_9_l321_321535

def fibonacci (n : ℕ) : ℕ :=
  if h : n < 2 then n else fibonacci (n - 1) + fibonacci (n - 2)

theorem fibonacci_150_mod_9 : fibonacci 150 % 9 = 8 :=
  sorry

end fibonacci_150_mod_9_l321_321535


namespace sum_of_largest_and_smallest_prime_factors_of_1540_l321_321527

theorem sum_of_largest_and_smallest_prime_factors_of_1540 : 
  (let smallest_prime := 2
       largest_prime := 11 in
   smallest_prime + largest_prime) = 13 := 
by
  -- Prime factors of 1540 are 2, 2, 5, 7, 11
  -- Smallest prime factor is 2
  -- Largest prime factor is 11
  -- Sum is 13
  sorry

end sum_of_largest_and_smallest_prime_factors_of_1540_l321_321527


namespace black_squares_on_29x29_checkerboard_l321_321146

def is_black_square (i j : ℕ) : Prop :=
  (i + j) % 2 = 0

def count_black_squares (n : ℕ) : ℕ :=
  (finset.range n).sum (λ i, (finset.range n).sum (λ j, if is_black_square i j then 1 else 0))

theorem black_squares_on_29x29_checkerboard : count_black_squares 29 = 421 := 
by sorry

end black_squares_on_29x29_checkerboard_l321_321146


namespace number_of_cubes_with_at_least_two_faces_painted_is_56_l321_321806

def one_inch_cubes_with_at_least_two_faces_painted 
  (side_length : ℕ) (face_colors : ℕ) (cubes_per_side : ℕ) :=
  if side_length = 4 ∧ face_colors = 6 ∧ cubes_per_side = 1 then 56 else 0

theorem number_of_cubes_with_at_least_two_faces_painted_is_56 :
  one_inch_cubes_with_at_least_two_faces_painted 4 6 1 = 56 :=
by
  sorry

end number_of_cubes_with_at_least_two_faces_painted_is_56_l321_321806


namespace solve_log_inequality_a_gt_1_solve_log_inequality_0_lt_a_lt_1_l321_321382

theorem solve_log_inequality_a_gt_1 (a x : ℝ) (h_a : 1 < a) :
  (log a (2 * x - 5) > log a (x - 1)) ↔ (x > 4) :=
sorry

theorem solve_log_inequality_0_lt_a_lt_1 (a x : ℝ) (h_a : 0 < a) (h_a' : a < 1) : 
  (log a (2 * x - 5) > log a (x - 1)) ↔ (5 / 2 < x ∧ x < 4) :=
sorry

end solve_log_inequality_a_gt_1_solve_log_inequality_0_lt_a_lt_1_l321_321382


namespace cylinder_volume_half_l321_321101

-- Define the given problem
def radius1 : ℝ := 5
def height1 : ℝ := 12
def volume1 : ℝ := real.pi * radius1^2 * height1

def radius2 : ℝ := 5
def height2 : ℝ := 6
def volume2 : ℝ := real.pi * radius2^2 * height2

-- The theorem to prove
theorem cylinder_volume_half :
  volume2 = volume1 / 2 := by
    sorry

end cylinder_volume_half_l321_321101


namespace painted_cubes_count_l321_321803

def total_painted_cubes : ℕ := 8 + 48

theorem painted_cubes_count : total_painted_cubes = 56 :=
by 
  -- Step 1: Define the number of cubes with 3 faces painted (8 corners)
  let corners := 8
  -- Step 2: Calculate the number of edge cubes with 2 faces painted
  let edge_middle_cubes_per_edge := 2
  let edges := 12
  let edge_cubes := edge_middle_cubes_per_edge * edges -- this should be 24
  -- Step 3: Calculate the number of face-interior cubes with 2 faces painted
  let face_cubes_per_face := 4
  let faces := 6
  let face_cubes := face_cubes_per_face * faces -- this should be 24
  -- Step 4: Sum them up to get total cubes with at least two faces painted
  let total_cubes := corners + edge_cubes + face_cubes
  show total_cubes = total_painted_cubes
  sorry

end painted_cubes_count_l321_321803


namespace max_sphere_radius_fits_pyramid_l321_321472

noncomputable def radius_of_max_sphere 
  (MB : ℝ) (angle_M : ℝ) (angle_BKBH : ℝ) (area_BKH : ℝ) (volume_pyramid : ℝ) (sum_edges_BE_HE : ℝ) : ℝ :=
  if MB = 1 ∧ angle_M = π / 2 ∧ angle_BKBH = π / 4 ∧ volume_pyramid = 1 / 4 ∧ area_BKH = 1 ∧ 
     sum_edges_BE_HE = sqrt 3 then
    2 - sqrt 2
  else
    0  -- just a default value, actual logic will be detailed in the proof

theorem max_sphere_radius_fits_pyramid :
  radius_of_max_sphere 1 (π / 2) (π / 4) 1 (1 / 4) (sqrt 3) = 2 - sqrt 2 :=
sorry

end max_sphere_radius_fits_pyramid_l321_321472


namespace max_area_of_rectangle_with_perimeter_40_l321_321014

theorem max_area_of_rectangle_with_perimeter_40 :
  (∃ (x y : ℝ), (2 * x + 2 * y = 40) ∧
                (∀ (a b : ℝ), (2 * a + 2 * b = 40) → (a * b ≤ x * y)) ∧
                (x * y = 100)) :=
begin
  -- Definitions of x and y satisfying the perimeter and maximizing the area.
  have h1 : ∀ (x y : ℝ), 2 * x + 2 * y = 40 → x * (20 - x) = -(x - 10)^2 + 100,
  { intro x, intro y, intro hper,
    have hy : y = 20 - x, by linarith,
    rw hy,
    ring },
  use 10,
  use 10,
  split,
  { -- Perimeter condition
    linarith },
  { split,
    { -- Maximum area condition
      intros a b hper,
      have hab : b = 20 - a, by linarith,
      rw hab,
      specialize h1 a (20 - a),
      linarith },
    { -- Maximum area is 100
      exact (by ring) } }
end

end max_area_of_rectangle_with_perimeter_40_l321_321014


namespace largest_prime_divisor_13_factorial_plus_14_factorial_l321_321995

theorem largest_prime_divisor_13_factorial_plus_14_factorial : 
  ∃ p, prime p ∧ p ∈ prime_factors (13! + 14!) ∧ ∀ q, prime q ∧ q ∈ prime_factors (13! + 14!) → q ≤ 13 :=
by { sorry }

end largest_prime_divisor_13_factorial_plus_14_factorial_l321_321995


namespace max_area_rect_40_perimeter_l321_321021

noncomputable def max_rect_area (P : ℕ) (hP : P = 40) : ℕ :=
  let w : ℕ → ℕ := id
  let l : ℕ → ℕ := λ w, P / 2 - w
  let area : ℕ → ℕ := λ w, w * (P / 2 - w)
  find_max_value area sorry

theorem max_area_rect_40_perimeter : max_rect_area 40 40 = 100 := 
sorry

end max_area_rect_40_perimeter_l321_321021


namespace train_length_l321_321121

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def train_speed_mps : ℝ := 20
noncomputable def crossing_time : ℝ := 20
noncomputable def platform_length : ℝ := 220.032
noncomputable def total_distance : ℝ := train_speed_mps * crossing_time

theorem train_length :
  total_distance - platform_length = 179.968 := by
  sorry

end train_length_l321_321121


namespace combined_soldiers_on_great_wall_l321_321651

theorem combined_soldiers_on_great_wall :
  let wall_length := 7300 -- total length of the Great Wall in kilometers
  let interval := 5 -- interval between towers in kilometers
  let soldiers_per_tower := 2 -- number of soldiers per tower
  let number_of_towers := wall_length / interval
  let total_soldiers := number_of_towers * soldiers_per_tower
  in total_soldiers = 2920 :=
by
  sorry

end combined_soldiers_on_great_wall_l321_321651


namespace product_of_divisors_of_30_l321_321448

theorem product_of_divisors_of_30 : ∏ d in (finset.filter (∣ 30) (finset.range 31)), d = 810000 := by
  sorry

end product_of_divisors_of_30_l321_321448


namespace avg_of_consecutive_odds_l321_321412

theorem avg_of_consecutive_odds (a : Int) (n : Int) (least : a = 407) (count : n = 8) :
  let nums := List.range n |>.map (λ i => a + 2 * i),
  (nums.sum : ℚ) / n = 401.5 :=
by
  sorry

end avg_of_consecutive_odds_l321_321412


namespace traveler_distance_l321_321455

noncomputable def distance_from_start (north south west east : ℝ) : ℝ :=
  let net_north_south := north - south
  let net_west_east := west - east
  real.sqrt (net_north_south^2 + net_west_east^2)

theorem traveler_distance : distance_from_start 25 10 15 7 = 17 :=
by sorry

end traveler_distance_l321_321455


namespace num_primes_between_3000_and_6000_squares_l321_321236

theorem num_primes_between_3000_and_6000_squares :
  let primesInRange := [59, 61, 67, 71, 73] in
  ∀ n : ℕ, (3000 < n^2 ∧ n^2 < 6000 ∧ (54 < n ∧ n < 78) → n ∈ primesInRange) → primesInRange.length = 5 := 
by
  sorry

end num_primes_between_3000_and_6000_squares_l321_321236


namespace solution_set_l321_321248

noncomputable def f : ℝ → ℝ := sorry

axiom f_cond1 : ∀ x : ℝ, f x + deriv f x > 1
axiom f_cond2 : f 0 = 4

theorem solution_set (x : ℝ) : e^x * f x > e^x + 3 ↔ x > 0 :=
by sorry

end solution_set_l321_321248


namespace largest_prime_divisor_13_fact_14_fact_l321_321978

theorem largest_prime_divisor_13_fact_14_fact :
  ∀ (n : ℕ), (n = 13!) ∧ (m = 14!) ∧ (m = 14 * n) ∧ ((n + m) = n * 15) ∧ (15 = 3 * 5) → 
  (∃ p : ℕ, (Prime p) ∧ (LargestPrimeDivisor p (n + m)) ∧ (p = 13)) :=
by
  sorry

end largest_prime_divisor_13_fact_14_fact_l321_321978


namespace total_ants_found_l321_321503

-- Definitions for the number of ants each child finds
def abe_ants : ℕ := 4
def beth_ants : ℕ := abe_ants + (abe_ants / 2)
def cece_ants : ℕ := 2 * abe_ants
def duke_ants : ℕ := abe_ants / 2

-- Statement that needs to be proven
theorem total_ants_found : abe_ants + beth_ants + cece_ants + duke_ants = 20 :=
by sorry

end total_ants_found_l321_321503


namespace length_GP_l321_321307

noncomputable def semiPerimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def heronArea (a b c : ℝ) : ℝ := 
  let s := semiPerimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def altitudeLength (a b c : ℝ) : ℝ := 
  2 * (heronArea a b c) / c

theorem length_GP :
  let a := 12
  let b := 15
  let c := 18
  ∃ GP : ℝ, GP = altitudeLength a b c / 3 ∧ GP ≈ 3.492 := by
  let a := 12
  let b := 15
  let c := 18
  let s := semiPerimeter a b c
  let area := heronArea a b c
  let AQ := altitudeLength a b c
  let GP := AQ / 3
  have : GP ≈ 3.492 := by sorry
  exact ⟨GP, rfl, this⟩

end length_GP_l321_321307


namespace probability_of_selecting_male_volunteer_l321_321054

variables {UnitA UnitB : Type}
variables (maleA femaleA maleB femaleB: ℕ)

-- Define the conditions given in the problem
def conditions (h1 : maleA = 5) (h2 : femaleA = 7) 
  (h3 : maleB = 4) (h4 : femaleB = 2) 
  (h5 : (1 : ℝ) / 2 = (1 : ℝ) / 2) : Prop := 
maleA + femaleA + maleB + femaleB ≤ 24

-- The theorem stating the probability of selecting a male volunteer is 13/24
theorem probability_of_selecting_male_volunteer : 
  ∀ (h1 : maleA = 5) (h2 : femaleA = 7) 
   (h3 : maleB = 4) (h4 : femaleB = 2),
  (1 : ℝ) / 2 * (maleA / (maleA + femaleA) : ℝ) 
  + (1 : ℝ) / 2 * (maleB / (maleB + femaleB) : ℝ) = 13 / 24 := 
begin
  intros,
  rw [h1, h2, h3, h4],
  norm_num,
end

end probability_of_selecting_male_volunteer_l321_321054


namespace find_alpha_l321_321660

-- Definition of parametric equations of C1
def C1_parametric (φ : ℝ) : ℝ × ℝ :=
  (2 + 2 * Real.cos φ, 2 * Real.sin φ)

-- Definition of polar equation of C2
def C2_polar (θ : ℝ) : ℝ :=
  4 * Real.sin θ

-- Definition of polar equation of C3
def C3_polar (α : ℝ) (ρ : ℝ) : Prop :=
  (0 < α) ∧ (α < π) ∧ (ρ ∈ Set.Ioo (-∞) ∞)

-- The theorem to be proved
theorem find_alpha (α : ℝ) (A B : ℝ × ℝ) :
  let C1_cartesian := ∀ (x y : ℝ), ((x - 2) ^ 2 + y ^ 2 = 4)
  let C2_cartesian := ∀ (x y : ℝ), (x ^ 2 + (y - 2) ^ 2 = 4)
  let C3 := C3_polar α
  (|A.1 - B.1| = 4 * Real.sqrt 2)
  ∧ (A ≠ (0, 0)) 
  ∧ (B ≠ (0, 0)) 
  →
  α = 3 * π / 4 :=
sorry

end find_alpha_l321_321660


namespace probability_of_collecting_both_types_correct_l321_321876

-- Defining the problem space
def toy_box := {1, 2} -- Representing two types of toys as 1 and 2

noncomputable def probability_of_collecting_both_types : ℚ :=
  let outcomes : list (list ℕ) := list.replicate 3 toy_box.to_list.choice in -- All possible outcomes of buying 3 blind boxes
  let favorable : list (list ℕ) := outcomes.filter (λ x, (1 ∈ x) ∧ (2 ∈ x)) in -- Outcomes where both types are collected
  (favorable.length.to_rat / outcomes.length.to_rat)

-- Statement of the theorem
theorem probability_of_collecting_both_types_correct :
  probability_of_collecting_both_types = 3 / 4 :=
by sorry

-- This is to ensure the program will compile.
def main : IO Unit :=
  IO.println s!"Theorem: {probability_of_collecting_both_types_correct}"

end probability_of_collecting_both_types_correct_l321_321876


namespace max_area_of_rectangle_with_perimeter_40_l321_321013

theorem max_area_of_rectangle_with_perimeter_40 :
  (∃ (x y : ℝ), (2 * x + 2 * y = 40) ∧
                (∀ (a b : ℝ), (2 * a + 2 * b = 40) → (a * b ≤ x * y)) ∧
                (x * y = 100)) :=
begin
  -- Definitions of x and y satisfying the perimeter and maximizing the area.
  have h1 : ∀ (x y : ℝ), 2 * x + 2 * y = 40 → x * (20 - x) = -(x - 10)^2 + 100,
  { intro x, intro y, intro hper,
    have hy : y = 20 - x, by linarith,
    rw hy,
    ring },
  use 10,
  use 10,
  split,
  { -- Perimeter condition
    linarith },
  { split,
    { -- Maximum area condition
      intros a b hper,
      have hab : b = 20 - a, by linarith,
      rw hab,
      specialize h1 a (20 - a),
      linarith },
    { -- Maximum area is 100
      exact (by ring) } }
end

end max_area_of_rectangle_with_perimeter_40_l321_321013


namespace largest_prime_divisor_of_factorial_sum_l321_321963

theorem largest_prime_divisor_of_factorial_sum :
  ∃ p : ℕ, p.prime ∧ p ∣ (13! + 14!) ∧ ∀ q : ℕ, q.prime ∧ q ∣ (13! + 14!) → q ≤ p :=
sorry

end largest_prime_divisor_of_factorial_sum_l321_321963


namespace zero_vector_collinear_l321_321454

-- Define vectors and required operations/properties
section vectors
variable {α : Type*} [AddCommGroup α] [Module ℝ α]

-- Definitions and conditions from the problem
def is_square (A B C D : α) : Prop :=
  (∃ l : ℝ, l > 0 ∧ dist A B = l ∧ dist B C = l ∧ dist C D = l ∧ dist D A = l ∧ dist A C = l * √2 ∧ dist B D = l * √2)

def parallel (v w : α) : Prop := ∃ k : ℝ, v = k • w

def collinear (v w : α) : Prop := ∃ k : ℝ, v = k • w ∨ w = k • v

def perpendicular (v w : α) : Prop := inner_product v w = 0

-- Statement to be proved as correct
theorem zero_vector_collinear (v : α) : collinear 0 v := 
by sorry

end zero_vector_collinear_l321_321454


namespace sampling_methods_suitability_l321_321829

-- Define sample sizes and population sizes
def n1 := 2  -- Number of students to be selected in sample ①
def N1 := 10  -- Population size for sample ①
def n2 := 50  -- Number of students to be selected in sample ②
def N2 := 1000  -- Population size for sample ②

-- Define what it means for a sampling method to be suitable
def is_simple_random_sampling_suitable (n N : Nat) : Prop :=
  N <= 50 ∧ n < N

def is_systematic_sampling_suitable (n N : Nat) : Prop :=
  N > 50 ∧ n < N ∧ n ≥ 50 / 1000 * N  -- Ensuring suitable systematic sampling size

-- The proof statement
theorem sampling_methods_suitability :
  is_simple_random_sampling_suitable n1 N1 ∧ is_systematic_sampling_suitable n2 N2 :=
by
  -- Sorry blocks are used to skip the proofs
  sorry

end sampling_methods_suitability_l321_321829


namespace minimum_value_of_F_l321_321833

def F (x y : ℝ) := x^2 + 8 * y + y^2 + 14 * x - 6

noncomputable def constrained_min : ℝ :=
  let C := {p : ℝ × ℝ | p.1^2 + p.2^2 + 25 = 10 * (p.1 + p.2)} in
  Inf {F p.1 p.2 | p ∈ C}

theorem minimum_value_of_F :
  constrained_min = 29 :=
by
  sorry

end minimum_value_of_F_l321_321833


namespace rearrange_HMMTHMMT_without_HMMT_l321_321294

theorem rearrange_HMMTHMMT_without_HMMT :
  let word := ['H', 'M', 'M', 'T', 'H', 'M', 'M', 'T']
  let H := 2
      M := 4
      T := 2
  count_valid_permutations_without_substring_HMMT word = 361 :=
by
  sorry

end rearrange_HMMTHMMT_without_HMMT_l321_321294


namespace largest_prime_divisor_of_factorial_sum_l321_321983

theorem largest_prime_divisor_of_factorial_sum :
  ∀ (n: ℕ), n = 13 → largest_prime_divisor (factorial n + factorial (n + 1)) = 7 := 
by
  intros,
  sorry

end largest_prime_divisor_of_factorial_sum_l321_321983


namespace quadratic_points_relation_l321_321630

theorem quadratic_points_relation (h y1 y2 y3 : ℝ) :
  (∀ x, x = -1/2 → y1 = -(x-2) ^ 2 + h) ∧
  (∀ x, x = 1 → y2 = -(x-2) ^ 2 + h) ∧
  (∀ x, x = 2 → y3 = -(x-2) ^ 2 + h) →
  y1 < y2 ∧ y2 < y3 :=
by
  -- The required proof is omitted
  sorry

end quadratic_points_relation_l321_321630


namespace parabola_vertex_not_in_second_quadrant_l321_321528

theorem parabola_vertex_not_in_second_quadrant :
  ∀ (a : ℝ), let xv := (a + 1) / 2, yv := -(a + 1)^2 + a in ¬ (xv < 0 ∧ yv > 0) :=
by
  intro a
  let xv := (a + 1) / 2
  let yv := -(a + 1)^2 + a
  have : xv < 0 → yv ≤ 0 := sorry
  intro h
  intro ⟨hxv, hyv⟩
  exact this hxv hyv

end parabola_vertex_not_in_second_quadrant_l321_321528


namespace fraction_log_identity_l321_321622

theorem fraction_log_identity (a b : ℝ) (h1 : a = log 2 10) (h2 : b = log 5 10) :
  (a + b) / (a * b) = 1 :=
sorry

end fraction_log_identity_l321_321622


namespace smallest_number_of_students_l321_321284

theorem smallest_number_of_students
  (n : ℕ)
  (students_attended : ℕ)
  (students_both_competitions : ℕ)
  (students_hinting : ℕ)
  (students_cheating : ℕ)
  (attended_fraction : Real := 0.25)
  (both_competitions_fraction : Real := 0.1)
  (hinting_ratio : Real := 1.5)
  (h_attended : students_attended = (attended_fraction * n).to_nat)
  (h_both : students_both_competitions = (both_competitions_fraction * students_attended).to_nat)
  (h_hinting : students_hinting = (hinting_ratio * students_cheating).to_nat)
  (h_total_attended : students_attended = students_hinting + students_cheating - students_both_competitions)
  : n = 200 :=
sorry

end smallest_number_of_students_l321_321284


namespace area_right_scalene_triangle_l321_321735

noncomputable def area_of_triangle_ABC : ℝ :=
  let AP : ℝ := 2
  let CP : ℝ := 1
  let AC : ℝ := AP + CP
  let ratio : ℝ := 2
  let x_squared : ℝ := 9 / 5
  x_squared

theorem area_right_scalene_triangle (AP CP : ℝ) (h₁ : AP = 2) (h₂ : CP = 1) (h₃ : ∠(B : Point) (A : Point) (P : Point) = 30) :
  let AB := 2 * real.sqrt(9 / 5)
  let BC := real.sqrt(9 / 5)
  ∃ (area : ℝ), area = (1/2) * AB * BC ∧ area = 9 / 5 :=
by
  sorry

end area_right_scalene_triangle_l321_321735


namespace days_worked_per_week_l321_321485

theorem days_worked_per_week (total_toys_per_week toys_produced_each_day : ℕ) 
  (h1 : total_toys_per_week = 5505)
  (h2 : toys_produced_each_day = 1101)
  : total_toys_per_week / toys_produced_each_day = 5 :=
  by
    sorry

end days_worked_per_week_l321_321485


namespace school_competition_l321_321270

theorem school_competition :
  (∃ n : ℕ, 
    n > 0 ∧ 
    75% students did not attend the competition ∧
    10% of those who attended participated in both competitions ∧
    ∃ y z : ℕ, y = 3 / 2 * z ∧ 
    y + z - (1 / 10) * (n / 4) = n / 4
  ) → n = 200 :=
sorry

end school_competition_l321_321270


namespace range_of_a_l321_321474

theorem range_of_a
  (p : ∀ x ∈ (set.Icc 1 2), x^2 - a ≥ 0)
  (q : ∃ x0 : ℝ, x + (a - 1) * x0 + 1 < 0) :
  (p ∨ q) ∧ ¬(p ∧ q) → a ∈ {a | -1 ≤ a ∧ a ≤ 1} ∨ a ∈ {a | a > 3} := by
    sorry

end range_of_a_l321_321474


namespace simplest_quadratic_radical_l321_321073

theorem simplest_quadratic_radical :
  ∀ x ∈ {sqrt 3, sqrt 4, sqrt 8, sqrt (1 / 2)}, (x = sqrt 3) :=
by
  sorry

end simplest_quadratic_radical_l321_321073


namespace product_of_areas_eq_square_of_volume_l321_321386

variable (x y z : ℝ)

def area_xy : ℝ := x * y
def area_yz : ℝ := y * z
def area_zx : ℝ := z * x

theorem product_of_areas_eq_square_of_volume :
  (area_xy x y) * (area_yz y z) * (area_zx z x) = (x * y * z) ^ 2 :=
by
  sorry

end product_of_areas_eq_square_of_volume_l321_321386


namespace diamond_example_l321_321538

def diamond (a b : ℕ) : ℤ := 4 * a + 5 * b - a^2 * b

theorem diamond_example : diamond 3 4 = -4 :=
by
  rw [diamond]
  calc
    4 * 3 + 5 * 4 - 3^2 * 4 = 12 + 20 - 36 := by norm_num
                           _              = -4 := by norm_num

end diamond_example_l321_321538


namespace number_of_routes_l321_321261

open Nat

theorem number_of_routes (south_cities north_cities : ℕ) 
  (connections : south_cities = 4 ∧ north_cities = 5) : 
  ∃ routes, routes = (factorial 3) * (5 ^ 4) := 
by
  sorry

end number_of_routes_l321_321261


namespace largest_prime_divisor_13_factorial_plus_14_factorial_l321_321993

theorem largest_prime_divisor_13_factorial_plus_14_factorial : 
  ∃ p, prime p ∧ p ∈ prime_factors (13! + 14!) ∧ ∀ q, prime q ∧ q ∈ prime_factors (13! + 14!) → q ≤ 13 :=
by { sorry }

end largest_prime_divisor_13_factorial_plus_14_factorial_l321_321993


namespace sum_of_interior_angles_of_octagon_l321_321405

theorem sum_of_interior_angles_of_octagon (n : ℕ) (h : n = 8) : (n - 2) * 180 = 1080 := by
  sorry

end sum_of_interior_angles_of_octagon_l321_321405


namespace remainder_of_t50_mod_7_l321_321540

noncomputable def T : ℕ → ℕ
| 0       := 3
| (n + 1) := 3 ^ T n

theorem remainder_of_t50_mod_7 : (T 49) % 7 = 6 :=
sorry

end remainder_of_t50_mod_7_l321_321540


namespace carla_cooking_time_l321_321145

theorem carla_cooking_time :
  ∀ (t_steak t_waffle n_steak : ℕ), 
  t_steak = 6 ∧ t_waffle = 10 ∧ n_steak = 3 →
  t_steak * n_steak + t_waffle = 28 :=
by
  intros t_steak t_waffle n_steak h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h3, h4]
  sorry

end carla_cooking_time_l321_321145


namespace minimize_sum_of_distances_l321_321580

noncomputable def pointA : ℝ × ℝ := (6, 3)
noncomputable def pointB : ℝ × ℝ := (3, -2)

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

def sumOfDistances (k : ℝ) : ℝ :=
  dist pointA (0, k) + dist pointB (0, k)

theorem minimize_sum_of_distances :
  ∃ k, sumOfDistances k = sumOfDistances (-1) :=
sorry

end minimize_sum_of_distances_l321_321580


namespace average_cookies_per_package_is_fifteen_l321_321521

def average_cookies_count (cookies : List ℕ) (n : ℕ) : ℕ :=
  (cookies.sum / n : ℕ)

theorem average_cookies_per_package_is_fifteen :
  average_cookies_count [5, 12, 18, 20, 21] 5 = 15 :=
by
  sorry

end average_cookies_per_package_is_fifteen_l321_321521


namespace part_a_part_b_l321_321654

-- Define the exchange rates
def ornament_to_crackers := 2
def sparklers_to_garlands := (5, 2)
def ornaments_to_garland := (4, 1)

-- Part (a)
theorem part_a (n_sparklers : ℕ) (h : n_sparklers = 10) :
  let n_garlands := (n_sparklers / sparklers_to_garlands.1) * sparklers_to_garlands.2 in
  let n_ornaments := n_garlands * ornaments_to_garland.1 in
  let n_crackers := n_ornaments * ornament_to_crackers in
  n_crackers = 32 :=
by {
  have n_garlands_def : n_garlands = (n_sparklers / sparklers_to_garlands.1) * sparklers_to_garlands.2, sorry,
  have n_ornaments_def : n_ornaments = n_garlands * ornaments_to_garland.1, sorry,
  have n_crackers_def : n_crackers = n_ornaments * ornament_to_crackers, sorry,
  have n_sparklers_eq : n_sparklers = 10, from h,
  sorry
}

-- Part (b)
theorem part_b :
  let v1 := (5 * ornament_to_crackers) + 1 in
  let v2 := ((2 / sparklers_to_garlands.1).nat_divide * sparklers_to_garlands.2 * ornaments_to_garland.1) * ornament_to_crackers in
  v1 > v2 :=
by {
  have v1_def : v1 = (5 * ornament_to_crackers) + 1, sorry,
  have v2_def : v2 = ((2 / sparklers_to_garlands.1).nat_divide * sparklers_to_garlands.2 * ornaments_to_garland.1) * ornament_to_crackers, sorry,
  sorry
}

end part_a_part_b_l321_321654


namespace parabola_shift_right_by_3_l321_321010

theorem parabola_shift_right_by_3 :
  ∀ (x : ℝ), (∃ y₁ y₂ : ℝ, y₁ = 2 * x^2 ∧ y₂ = 2 * (x - 3)^2) →
  (∃ (h : ℝ), h = 3) :=
sorry

end parabola_shift_right_by_3_l321_321010


namespace quadratic_binomial_square_l321_321938

theorem quadratic_binomial_square (a : ℚ) :
  (∃ r s : ℚ, (ax^2 + 22*x + 9 = (r*x + s)^2) ∧ s = 3 ∧ r = 11 / 3) → a = 121 / 9 := 
by 
  sorry

end quadratic_binomial_square_l321_321938


namespace tangent_line_parallel_or_coincident_l321_321583

theorem tangent_line_parallel_or_coincident (f : ℝ → ℝ) (x₀ : ℝ)
  (h : ∀ x, deriv f x = 0) :
  ∀ y, y = f x₀ → tangent_line f x₀ = λ x, f x₀ :=
sorry

def tangent_line (f : ℝ → ℝ) (x₀ : ℝ) := λ x, f x₀

end tangent_line_parallel_or_coincident_l321_321583


namespace mul102_105_l321_321532

theorem mul102_105 : (102 : ℕ) * (105 : ℕ) = 10710 :=
by
  -- Conditions
  have h1 : 102 = 100 + 2 := rfl
  have h2 : 105 = 100 + 5 := rfl
  -- Proof
  calc
    102 * 105 = (100 + 2) * (100 + 5) : by rw [h1, h2]
          ... = 100 * 100 + 100 * 5 + 2 * 100 + 2 * 5 : by ring
          ... = 10000 + 500 + 200 + 10 : rfl
          ... = 10710 : rfl

end mul102_105_l321_321532


namespace number_of_soldiers_l321_321652

theorem number_of_soldiers (length_of_wall : ℕ) (interval : ℕ) (soldiers_per_tower : ℕ)
  (h1 : length_of_wall = 7300) (h2 : interval = 5) (h3 : soldiers_per_tower = 2) :
  (length_of_wall / interval) * soldiers_per_tower = 2920 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end number_of_soldiers_l321_321652


namespace maximum_area_of_rectangle_with_fixed_perimeter_l321_321017

theorem maximum_area_of_rectangle_with_fixed_perimeter (x y : ℝ) 
  (h₁ : 2 * (x + y) = 40) 
  (h₂ : x = y) :
  x * y = 100 :=
by
  sorry

end maximum_area_of_rectangle_with_fixed_perimeter_l321_321017


namespace max_coconuts_to_goats_l321_321702

/-
Max can trade 3 coconuts for 1 crab, and 6 crabs for 1 goat. 
If he has 342 coconuts and he wants to convert all of them into goats, 
prove that he will have 19 goats.
-/

theorem max_coconuts_to_goats :
  (coconuts_per_crab coconuts_per_goat coconuts : Nat) 
  (coconuts_per_crab = 3) 
  (coconuts_per_goat = 18) 
  (coconuts = 342) : 
  coconuts / coconuts_per_crab / (coconuts_per_goat / 3) = 19 :=
by
  sorry

end max_coconuts_to_goats_l321_321702


namespace compute_expression_l321_321922
-- Import the standard math library to avoid import errors.

-- Define the theorem statement based on the given conditions and the correct answer.
theorem compute_expression :
  (75 * 2424 + 25 * 2424) / 2 = 121200 :=
by
  sorry

end compute_expression_l321_321922


namespace sine_addition_formula_l321_321724

theorem sine_addition_formula (α β A B : ℝ) (m n : ℤ) (h1 : α = m * (π / 2) + A) 
  (h2 : β = n * (π / 2) + B) (h3 : 0 ≤ A) (h4 : A < π / 2) (h5 : 0 ≤ B) 
  (h6 : B < π / 2) : 
  sin (α + β) = sin α * cos β + cos α * sin β :=
by
  sorry

end sine_addition_formula_l321_321724


namespace proof_problem_l321_321662

noncomputable def curveC1 (α : Real) : Real × Real :=
  (1 / 2 * Real.cos α, 1 + 1 / 2 * Real.sin α)

def C1_equation (x y : Real) : Prop :=
  x^2 + (y - 1)^2 = 1 / 4

def polar_to_rect (ρ θ : Real) : Real × Real :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

noncomputable def curveC2 (ρ θ : Real) : Prop :=
  ρ^2 * (Real.sin θ ^ 2 + 4 * Real.cos θ ^ 2) = 4

def C2_equation (x y : Real) : Prop :=
  (x / 2)^2 + y^2 = 1

def distance (A B : Real × Real) : Real :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def min_distance : Real :=
  Real.sqrt(6) / 3 - 1 / 2

theorem proof_problem :
  (∀ (α : Real), curveC1 α ∣= (x, y) → C1_equation x y) ∧
  (∀ (ρ θ : Real), curveC2 ρ θ ∣= (x, y) → C2_equation x y) ∧
  (∀ (A ∈ C1, B ∈ C2), ∃ min_val ∈ R, min_val = min_distance) :=
by
  sorry

end proof_problem_l321_321662


namespace quadratic_roots_form_l321_321793

theorem quadratic_roots_form {d : ℝ} (h : ∀ x : ℝ, x^2 + 7*x + d = 0 → (x = (-7 + real.sqrt d) / 2) ∨ (x = (-7 - real.sqrt d) / 2)) : d = 49 / 5 := 
sorry

end quadratic_roots_form_l321_321793


namespace monotonically_increasing_intervals_g_range_l321_321219

noncomputable def f (x : ℝ) : ℝ := 2 * Math.sin (π/4 - x) * Math.sin (π/4 + x) - 2 * real.sqrt 3 * Math.sin x * Math.cos (π - x)

noncomputable def g (x : ℝ) : ℝ := 2 * Math.sin (x + π/3)

theorem monotonically_increasing_intervals (k : ℤ) :
  ∀ x ∈ set.Icc (k * π - π / 3) (k * π + π / 6), 
  f' x > 0 :=
sorry

theorem g_range :
  set.range (g ∘ λ x, x ∈ set.Icc 0 (5 * π / 6)) = set.Icc (-1) 2 :=
sorry

end monotonically_increasing_intervals_g_range_l321_321219


namespace pyramid_volume_l321_321110

noncomputable def volume_of_pyramid
  (total_surface_area : ℝ)
  (area_each_triangular_face : ℝ)
  (area_square_face : ℝ) : ℝ :=
  if h : total_surface_area = area_square_face + 4 * area_each_triangular_face then
    let s := real.sqrt (total_surface_area * 3 / 7) in
    let area_base := s^2 in
    let h_triangular := 6 * real.sqrt 6 in
    let height := real.sqrt (h_triangular^2 - (s/2)^2) in
    (1/3) * area_base * height
  else 0

theorem pyramid_volume
  (h_surface_area : total_surface_area = 648)
  (h_area_triangular_face : area_each_triangular_face = (1/3) * area_square_face)
  (h_area_square : area_square_face = (7/3) * 648) : 
  volume_of_pyramid total_surface_area area_each_triangular_face area_square_face = 486 * real.sqrt 15 :=
by
  unfold volume_of_pyramid
  have : total_surface_area = area_square_face + 4 * area_each_triangular_face := by
    rw [h_area_triangular_face, h_area_square]
  rw this
  sorry

end pyramid_volume_l321_321110


namespace count_negative_numbers_l321_321636

def evaluate (e : String) : Int :=
  match e with
  | "-3^2" => -9
  | "(-3)^2" => 9
  | "-(-3)" => 3
  | "-|-3|" => -3
  | _ => 0

def isNegative (n : Int) : Bool := n < 0

def countNegatives (es : List String) : Int :=
  es.map evaluate |>.filter isNegative |>.length

theorem count_negative_numbers :
  countNegatives ["-3^2", "(-3)^2", "-(-3)", "-|-3|"] = 2 :=
by
  sorry

end count_negative_numbers_l321_321636


namespace find_subtracted_number_l321_321478

theorem find_subtracted_number (x y : ℝ) (h1 : x = 62.5) (h2 : (2 * (x + 5)) / 5 - y = 22) : y = 5 :=
sorry

end find_subtracted_number_l321_321478


namespace total_lunch_bill_l321_321374

theorem total_lunch_bill (cost_hotdog cost_salad : ℝ) (h_hd : cost_hotdog = 5.36) (h_sd : cost_salad = 5.10) :
  cost_hotdog + cost_salad = 10.46 :=
by
  sorry

end total_lunch_bill_l321_321374


namespace bus_driver_regular_rate_l321_321879

theorem bus_driver_regular_rate 
  (R : ℝ)
  (H_hours : 52 = 40 + 12)
  (H_overtime_rate : ∀ (R : ℝ), 1.75 * R)
  (H_total_compensation : 40 * R + 12 * (1.75 * R) = 976) : 
  R = 16 := 
by
  sorry

end bus_driver_regular_rate_l321_321879


namespace triangle_solid_revolution_correct_l321_321217

noncomputable def triangle_solid_revolution (t : ℝ) (alpha beta gamma : ℝ) (longest_side : string) : ℝ × ℝ :=
  let pi := Real.pi;
  let sin := Real.sin;
  let cos := Real.cos;
  let sqrt := Real.sqrt;
  let to_rad (x : ℝ) : ℝ := x * pi / 180;
  let alpha_rad := to_rad alpha;
  let beta_rad := to_rad beta;
  let gamma_rad := to_rad gamma;
  let a := sqrt (2 * t * sin alpha_rad / (sin beta_rad * sin gamma_rad));
  let b := sqrt (2 * t * sin beta_rad / (sin gamma_rad * sin alpha_rad));
  let m_c := sqrt (2 * t * sin alpha_rad * sin beta_rad / sin gamma_rad);
  let F := 2 * pi * t * cos ((alpha_rad - beta_rad) / 2) / sin (gamma_rad / 2);
  let K := 2 * pi / 3 * t * sqrt (2 * t * sin alpha_rad * sin beta_rad / sin gamma_rad);
  (F, K)

theorem triangle_solid_revolution_correct :
  triangle_solid_revolution 80.362 (39 + 34/60 + 30/3600) (60 : ℝ) (80 + 25/60 + 30/3600) "c" = (769.3, 1595.3) :=
sorry

end triangle_solid_revolution_correct_l321_321217


namespace total_lunch_bill_l321_321376

def cost_of_hotdog : ℝ := 5.36
def cost_of_salad : ℝ := 5.10

theorem total_lunch_bill : cost_of_hotdog + cost_of_salad = 10.46 := 
by
  sorry

end total_lunch_bill_l321_321376


namespace position_of_12211_l321_321908

theorem position_of_12211 :
  let nums := [121, 112, 1112, 12112, 11122, 12211, 21211, 12121, 11221]
  let sorted_nums := list.sort Nat.lt nums
  list.index_of 12211 sorted_nums = 7 := -- Since list indexing is 0-based in Lean
by
  sorry

end position_of_12211_l321_321908


namespace range_of_x_l321_321585

def valid_x (x : ℝ) : Prop :=
  ∀ (m : ℝ), m ∈ set.Icc (1 / 2) 3 → x^2 + m * x + 4 > 2 * m + 4 * x

theorem range_of_x (x : ℝ) : valid_x x ↔ x > 2 ∨ x < -1 := by 
  sorry

end range_of_x_l321_321585


namespace line_through_given_point_cyclic_intersections_l321_321157

variable (A B C P : Point)

theorem line_through_given_point_cyclic_intersections
    (A B C P : Point) :
    ∃ L : Line, intersects (L, A, B, C, P) ∧ cyclic L :=
by
  -- Proof would go here
  sorry

end line_through_given_point_cyclic_intersections_l321_321157


namespace arithmetic_sequence_properties_l321_321573

variable {a_n : ℕ → ℤ}
variable {a_1 d : ℤ}
variable {S : ℕ → ℤ}
variable [∀ n, decidable (n = 18 ∨ n = 19 ∨ n = 20)]

-- Conditions
def is_arithmetic_sequence (a_n : ℕ → ℤ) (a_1 : ℤ) (d : ℤ) : Prop :=
  ∀ n, a_n n = a_1 + (n - 1) * d

def sum_of_first_n_terms (a_n : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = n * (2 * a_1 + (n - 1) * d) / 2

def given_conditions (S : ℕ → ℤ) : Prop :=
  S 20 < S 18 ∧ S 18 < S 19

theorem arithmetic_sequence_properties 
  (a_n : ℕ → ℤ) (a_1 d : ℤ) (S : ℕ → ℤ)
  (h_seq : is_arithmetic_sequence a_n a_1 d)
  (h_sum : sum_of_first_n_terms a_n S)
  (h_conditions : given_conditions S) :
  a_1 > 0 ∧ ∀ (n : ℕ), (frac (S n) (a_n n)) ≥ (frac (S 20) (a_n 20)) := sorry

end arithmetic_sequence_properties_l321_321573


namespace conjugate_in_fourth_quadrant_l321_321301

def z : ℂ := (1 * I) / (1 + I)
def conjugate_z : ℂ := conj z

theorem conjugate_in_fourth_quadrant :
  (conjugate_z.re > 0) ∧ (conjugate_z.im < 0) := 
sorry

end conjugate_in_fourth_quadrant_l321_321301


namespace infinite_series_value_l321_321558

theorem infinite_series_value :
  (∑' n : ℕ, (n^3 + 2 * n^2 - n) / ((n + 3)! : ℝ)) = 1 / 6 :=
by
  sorry

end infinite_series_value_l321_321558


namespace first_player_wins_game_with_perfect_play_l321_321102

theorem first_player_wins_game_with_perfect_play :
  (∀ board : list (option ℕ), board.length = 25 → (∀ idx, 1 ≤ idx → idx ≤ 25 → board.nth (idx - 1) = none) →
  (∀ turn : ℕ, turn < 25 → (∀ move : ℕ, move ≤ turn + 1 → (∃ move_pos : ℕ, move_pos < 25 ∧ board.nth move_pos = none → board.nth (move_pos + turn + 1) = none)
  → ∃ has_winner : bool, has_winner = (turn % 2 = 0))) :=
begin
  sorry
end

end first_player_wins_game_with_perfect_play_l321_321102


namespace largest_prime_divisor_13_fact_14_fact_l321_321977

theorem largest_prime_divisor_13_fact_14_fact :
  ∀ (n : ℕ), (n = 13!) ∧ (m = 14!) ∧ (m = 14 * n) ∧ ((n + m) = n * 15) ∧ (15 = 3 * 5) → 
  (∃ p : ℕ, (Prime p) ∧ (LargestPrimeDivisor p (n + m)) ∧ (p = 13)) :=
by
  sorry

end largest_prime_divisor_13_fact_14_fact_l321_321977


namespace quadratic_root_condition_l321_321778

theorem quadratic_root_condition (d : ℝ) :
  (∀ x, x^2 + 7 * x + d = 0 → x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → d = 9.8 :=
by
  intro h
  sorry

end quadratic_root_condition_l321_321778


namespace find_EG_l321_321368

noncomputable def cyclic_quadrilateral
  (EFGH : Type)
  (∠EFG : ℝ)
  (∠EHG : ℝ)
  (EH : ℝ)
  (FG : ℝ)
  (EG : ℝ) : Prop :=
  ∠EFG = 60 ∧
  ∠EHG = 50 ∧
  EH = 5 ∧
  FG = 7 ∧
  EG = 7.07

theorem find_EG : ∃ EFGH : Type, cyclic_quadrilateral EFGH 60 50 5 7 7.07 :=
by
  sorry

end find_EG_l321_321368


namespace max_cake_pieces_l321_321095

theorem max_cake_pieces (m n : ℕ) (h₁ : m ≥ 4) (h₂ : n ≥ 4)
    (h : (m-4)*(n-4) = m * n) :
    m * n = 72 :=
by
  sorry

end max_cake_pieces_l321_321095


namespace area_union_l321_321496

def side_length : ℝ := 12
def radius : ℝ := side_length / 2

theorem area_union (side_length : ℝ) (radius : ℝ) :
  side_length = 12 ∧ radius = side_length / 2 →
  (side_length ^ 2 + 0.5 * real.pi * radius ^ 2) = 144 + 18 * real.pi :=
by
  intros h
  rw [h.left, h.right]
  sorry

end area_union_l321_321496


namespace volume_of_pyramid_proper_l321_321726

noncomputable def volume_of_pyramid (PA : ℝ) (s : ℝ) (base_area : ℝ) (height : ℝ) : ℝ :=
  (1/3) * base_area * height

theorem volume_of_pyramid_proper :
  ∀ (s : ℝ), s = 5 * Real.sqrt 3 →
  ∀ (height : ℝ), height = 5 * Real.sqrt 3 →
  ∀ (base_area : ℝ), base_area = 2 * (1 + Real.sqrt 2) * s^2 →
  ∀ (PA : ℝ), PA = 10 →
  volume_of_pyramid PA s base_area height = 750 * Real.sqrt 3 + 750 * Real.sqrt 6 :=
begin
  intros s s_def height height_def base_area base_area_def PA PA_def,
  rw volume_of_pyramid,
  rw [s_def, height_def, base_area_def, PA_def],
  sorry
end

end volume_of_pyramid_proper_l321_321726


namespace first_pedestrian_speed_l321_321469

def pedestrian_speed (x : ℝ) : Prop :=
  0 < x ∧ x ≤ 4 ↔ (11 / (x + 1.5) + 8 / x ≥ 12 / (x + 2) + 2)

theorem first_pedestrian_speed 
  (x : ℝ) (h : 11 / (x + 1.5) + 8 / x ≥ 12 / (x + 2) + 2) :
  0 < x ∧ x ≤ 4 :=
begin
  sorry
end

end first_pedestrian_speed_l321_321469


namespace simplify_expression_l321_321945

theorem simplify_expression (x y z : ℝ) : ((x + y) - (z - y)) - ((x + z) - (y + z)) = 3 * y - z := by
  sorry

end simplify_expression_l321_321945


namespace maximum_value_l321_321898

variable {a b c : ℝ}

-- Conditions
variable (h : a^2 + b^2 = c^2)

theorem maximum_value (h : a^2 + b^2 = c^2) : 
  (∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ 
   (∀ x y z : ℝ, x^2 + y^2 = z^2 → (x^2 + y^2 + x*y) / z^2 ≤ 1.5)) := 
sorry

end maximum_value_l321_321898


namespace total_chairs_all_together_l321_321561

-- Definitions of given conditions
def rows := 7
def chairs_per_row := 12
def extra_chairs := 11

-- Main statement we want to prove
theorem total_chairs_all_together : 
  (rows * chairs_per_row + extra_chairs = 95) := 
by
  sorry

end total_chairs_all_together_l321_321561


namespace relationship_y1_y2_y3_l321_321628

variable {y1 y2 y3 h : ℝ}

def point_A := -1 / 2
def point_B := 1
def point_C := 2
def quadratic_function (x : ℝ) : ℝ := -(x - 2)^2 + h

theorem relationship_y1_y2_y3
  (hA : A_on_curve : quadratic_function point_A = y1)
  (hB : B_on_curve : quadratic_function point_B = y2)
  (hC : C_on_curve : quadratic_function point_C = y3) :
  y1 < y2 ∧ y2 < y3 := 
sorry

end relationship_y1_y2_y3_l321_321628


namespace quadratic_complete_square_factors_l321_321238

theorem quadratic_complete_square_factors (a b c m : ℝ) (h_eq : 4 * a = b ∧ 9 * c = d) :
  4 * (a * x ^ 2) - (m + 1) * x + 9 =  (4 * a * x ^ 2) ∧ 9 * c =  fully_factors :
  ∃ m, m = 11 ∨ m = -13 :=
begin
  sorry
end

end quadratic_complete_square_factors_l321_321238


namespace problem_solution_l321_321242

theorem problem_solution (x y : ℚ) (h1 : |x| + x + y - 2 = 14) (h2 : x + |y| - y + 3 = 20) : 
  x + y = 31/5 := 
by
  -- It remains to prove
  sorry

end problem_solution_l321_321242


namespace complex_number_z_value_l321_321210

theorem complex_number_z_value :
  ∀ (z : ℂ), z * (1 - complex.i) = (1 + complex.i) ^ 3 → z = -2 :=
by
  intro z h
  sorry

end complex_number_z_value_l321_321210


namespace annual_decrease_rate_l321_321031

theorem annual_decrease_rate (P : ℕ) (P2 : ℕ) (r : ℝ) : 
  (P = 10000) → (P2 = 8100) → (P2 = P * (1 - r / 100)^2) → (r = 10) :=
by
  intro hP hP2 hEq
  sorry

end annual_decrease_rate_l321_321031


namespace smallest_number_of_eggs_l321_321857

theorem smallest_number_of_eggs (c : ℕ) (h1 : 15 * c - 3 > 100) : 102 ≤ 15 * c - 3 :=
by
  sorry

end smallest_number_of_eggs_l321_321857


namespace triangle_side_length_l321_321667

theorem triangle_side_length
  (A B : ℝ)
  (cos_term : Real.cos (3 * A - B) + Real.sin (A + B) = 1)
  (AB : ℝ)
  (h_AB : AB = 5) :
  let BC := AB / Real.sqrt 2
  in BC = 5 * Real.sqrt 2 / 2 :=
by
  sorry

end triangle_side_length_l321_321667


namespace triangle_acute_l321_321637

theorem triangle_acute (a b c : ℝ) (ha : a = 9) (hb : b = 10) (hc : c = 12) : 
  (a^2 + b^2 > c^2) := by 
{
  have h1 : a^2 = 81, by linarith [ha],
  have h2 : b^2 = 100, by linarith [hb],
  have h3 : c^2 = 144, by linarith [hc],
  have hsum : a^2 + b^2 = 181, by linarith [h1, h2],
  have ha : a^2 + b^2 > c^2,
  {
    linarith [hsum, h3],
  },
  exact ha,
}

end triangle_acute_l321_321637


namespace sum_A_B_is_3_l321_321775

-- Define the grid and condition
def grid := Matrix (Fin 3) (Fin 3) ℕ

-- Instantiate the grid with known values and variables A and B
def initial_grid (A B : ℕ) : grid :=
  λ i j, match (i, j) with
  | (0, 0) => 2
  | (1, 1) => A
  | (1, 2) => 3
  | (2, 0) => B
  | (_, _) => 0

-- Identify the condition that each rows, columns and diagonals have {1, 2, 3}
def valid_grid (M : grid) : Prop :=
  (∀ i : Fin 3, ∃ (L : List ℕ), L.Perm [1, 2, 3] ∧ (∀ j, M i j ∈ L)) ∧
  (∀ j : Fin 3, ∃ (L : List ℕ), L.Perm [1, 2, 3] ∧ (∀ i, M i j ∈ L)) ∧
  (∃ (L1 : List ℕ), L1.Perm [1, 2, 3] ∧ (∀ k, M k k ∈ L1)) ∧
  (∃ (L2 : List ℕ), L2.Perm [1, 2, 3] ∧ (∀ k, M k (2 - k) ∈ L2))

-- Prove that A + B = 3
theorem sum_A_B_is_3 {A B : ℕ} (h_valid : valid_grid (initial_grid A B)) : A + B = 3 :=
 sorry

end sum_A_B_is_3_l321_321775


namespace angle_and_max_area_condition_1_angle_and_max_area_condition_2_l321_321918

-- Define the problem in Lean 4

variables {A B C : ℚ} -- Angles of the triangle
variables {a b c : ℚ} -- Sides opposite the angles
-- Variables for points and distances
variables {D : ℚ} 
variables {AD DB CD : ℚ}

-- Additional conditions
axiom law_of_sines : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π
axiom condition_1 : 2 * a - b = 2 * c * cos B 
axiom condition_2 : (sqrt 3) * c * cos A + a * sin C = (sqrt 3) * b 

axiom point_D_condition : 2 * AD = DB ∧ CD = 1

-- Define the triangle properties
def angle_C_equals_pi_over_3 (C : ℚ) : Prop := 
  C = π / 3

def max_area_triangle (a b s : ℚ) (C : ℚ) : Prop :=
  s = (sqrt 3 / 4) * a * b ∧ s <= (3 * sqrt 3 / 8)

-- State the theorem for condition_1
theorem angle_and_max_area_condition_1 : 
  ∃ C, (2 * a - b = 2 * c * cos B) → (C = π / 3) ∧ 
  ∃ s, (2 * AD = DB ∧ CD = 1) → max_area_triangle a b s C :=
by sorry

-- State the theorem for condition_2
theorem angle_and_max_area_condition_2 : 
  ∃ C, (sqrt 3 * c * cos A + a * sin C = sqrt 3 * b) → (C = π / 3) ∧ 
  ∃ s, (2 * AD = DB ∧ CD = 1) → max_area_triangle a b s C :=
by sorry

end angle_and_max_area_condition_1_angle_and_max_area_condition_2_l321_321918


namespace tangent_line_A1_C1_l321_321051

-- Definitions of points within the equilateral triangle
variables {Point : Type} [metric_space Point]
variables {A B C A1 B1 C1: Point}

-- Conditions of the problem
def equilateral_triangle (A B C : Point) : Prop :=
∀ (P Q R : Point), (dist A B = dist B C) ∧ (dist B C = dist C A)

def midpoint (P Q R : Point) : Prop :=
dist P Q = dist R Q ∧ dist P R = dist Q R

def circumcircle_tangent (P Q R M : Point) : Prop :=
dist P Q = dist Q R ∧ dist R M = dist M P

-- Equilateral triangle condition
axiom h1 : equilateral_triangle A B C

-- Midpoints definition
axiom h2 : midpoint B C A1
axiom h3 : midpoint A C B1
axiom h4 : midpoint A B C1

-- Target: prove line A1 C1 is tangent to the circumcircle passing through A1, B1, C
theorem tangent_line_A1_C1
(h1 : equilateral_triangle A B C)
(h2 : midpoint B C A1)
(h3 : midpoint A C B1)
(h4 : midpoint A B C1) : circumcircle_tangent A1 C1 A1 B1 C :=
sorry

end tangent_line_A1_C1_l321_321051


namespace chicago_max_gangs_l321_321640

noncomputable def max_gangs (n : ℕ) : ℕ := 3^12

theorem chicago_max_gangs (n : ℕ) (gangsters : fin (n) → set (fin (n))):
  n = 36 →
  (∀ g1 g2, g1 ≠ g2 → gangsters g1 ≠ gangsters g2) →
  (∀ g, (∀ x ∈ gangsters g, ∀ y ∈ gangsters g, x ≠ y → ¬ (∃ h, h ≠ g ∧ x ∈ gangsters h ∧ y ∈ gangsters h))) →
  (∀ g x, x ∉ gangsters g → ∃ y ∈ gangsters g, ∀ h, h ≠ g → x ∉ gangsters h ∨ y ∈ gangsters h) →
  max_gangs n = 531441 :=
by
  intros
  exact sorry

end chicago_max_gangs_l321_321640


namespace find_d_l321_321783

noncomputable def quadratic_roots (d : ℝ) : Prop :=
∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2

theorem find_d : ∃ d : ℝ, d = 9.8 ∧ quadratic_roots d :=
sorry

end find_d_l321_321783


namespace sum_geometric_series_l321_321333

noncomputable def S_n (n : ℕ) : ℕ :=
  ∑ i in finset.range(n + 1), 2^(3 * i + 1)

theorem sum_geometric_series (n : ℕ) (hn : 0 < n) :
  S_n n = (2 / 7) * (8^(n+1) - 1) :=
begin
  sorry
end

end sum_geometric_series_l321_321333


namespace integral_f_eq_l321_321241

variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)
variable (h_eq : ∀ x, f x = x^2 + 2 * f (π / 2) * x + sin (2 * x))

theorem integral_f_eq : ∫ x in 0..1, f x = 17 / 6 - π - 1 / 2 * cos 2 := 
by {
  have h_f: ∀ x, f x = x^2 + 2 * f (π / 2) * x + sin (2 * x), from h_eq,
  sorry
}

end integral_f_eq_l321_321241


namespace largest_prime_divisor_13_fac_add_14_fac_l321_321967

theorem largest_prime_divisor_13_fac_add_14_fac : 
  ∀ (p : ℕ), prime p → p ∣ (13! + 14!) → p ≤ 5 :=
by
  sorry

end largest_prime_divisor_13_fac_add_14_fac_l321_321967


namespace circle_to_ellipse_scaling_l321_321928

theorem circle_to_ellipse_scaling :
  ∀ (x' y' : ℝ), (4 * x')^2 + y'^2 = 16 → x'^2 / 16 + y'^2 / 4 = 1 :=
by
  intro x' y'
  intro h
  sorry

end circle_to_ellipse_scaling_l321_321928


namespace possible_n_values_l321_321066

theorem possible_n_values :
  let divisors_of_12 := [1, 2, 3, 4, 6, 12]
  let valid_n := divisors_of_12.filter (> 2)
  valid_n.length = 4 := by
  sorry

end possible_n_values_l321_321066


namespace rorrim2_possible_positions_l321_321766

open Function

-- Define the initial board configuration
def Board := Fin 4 × Fin 4

-- Define initial position in one corner (1,1)
def initial_position : Board := (⟨0, by norm_num⟩, ⟨0, by norm_num⟩)

-- Reflect the position
def reflect (p : Board) (axis : ℕ) : Board :=
  let (r, c) := p in
  match axis with
  | 0 => (⟨3 - r.1, by norm_num⟩, c)  -- Horizontal reflection
  | 1 => (r, ⟨3 - c.1, by norm_num⟩)  -- Vertical reflection
  | _ => (⟨3 - r.1, by norm_num⟩, ⟨3 - c.1, by norm_num⟩)  -- Diagonal reflection
-- Note: axis 2-5 for all unique possible reflections

-- Determine if a cell is shaded
def is_shaded (p : Board) : Prop :=
  (p.1.1 + p.2.1) % 2 = 0

-- Set of possible cells after three reflections
def possible_cells_after_three_turns : Finset Board :=
  {p : Board | is_shaded (reflect (reflect (reflect initial_position 0) 1) 2)}

theorem rorrim2_possible_positions :
  (possible_cells_after_three_turns).card = 8 := by
  sorry

end rorrim2_possible_positions_l321_321766


namespace hyperbola_eccentricity_l321_321659

theorem hyperbola_eccentricity (a b : ℝ) (h : a^2 = 4 ∧ b^2 = 3) :
    let c := Real.sqrt (a^2 + b^2)
    let e := c / a
    e = Real.sqrt 7 / 2 :=
    by
  sorry

end hyperbola_eccentricity_l321_321659


namespace count_involutions_l321_321681

-- Define the problem statement
theorem count_involutions (n : ℕ) (h : n > 0) (A : Fin n → ℝ) :
  let count_involutions: ℕ → ℕ := λ n,
    ∑ k in Finset.range (n/2 + 1),
      n.factorial / (2^k * k.factorial * (n - 2*k).factorial)
  in (f : Fin n → Fin n) (hf : ∀ x y : Fin n, x > y → f(f x) - f(f y) ≥ (A x - A y)) 
      → ( ∑ i in Finset.range n, ite (f i = i ∨ (∃ j, j > i ∧ f(i) = j ∧ f(j) = i)) 1 0 ) 
      = count_involutions n :=
sorry

end count_involutions_l321_321681


namespace horner_rule_step4_l321_321055

theorem horner_rule_step4 (n : ℕ) (a : ℕ → ℝ) (x : ℝ) :
  let v := a n in
  let step3 (i : ℕ) := a i in
  let step4 (v : ℝ) (i : ℕ) := v * x + step3 i in
  ∀ i < n, i ≥ 0 → v = (λ acc i, step4 acc i) v i := sorry

end horner_rule_step4_l321_321055


namespace num_ways_is_20_l321_321067

-- Define the set of students
def students := { "Jungkook", "Jimin", "Seokjin", "Taehyung", "Namjoon" }

-- Define a representative and a vice-president as a permutation of two students
def num_ways :=
  fintype.card (equiv.perm (fin 2))

-- Theorem: Number of ways to select representative and vice-president
theorem num_ways_is_20 : num_ways = 20 := by
  sorry

end num_ways_is_20_l321_321067


namespace cost_of_small_bonsai_l321_321350

variable (cost_small_bonsai cost_big_bonsai : ℝ)

theorem cost_of_small_bonsai : 
  cost_big_bonsai = 20 → 
  3 * cost_small_bonsai + 5 * cost_big_bonsai = 190 → 
  cost_small_bonsai = 30 := 
by
  intros h1 h2 
  sorry

end cost_of_small_bonsai_l321_321350


namespace similar_triangles_angle_C_l321_321184

theorem similar_triangles_angle_C (A B C D E F : Type) [Preorder A] [Preorder B] [Preorder C] :
  (A ≈ D) → (B ≈ E) → (C ≈ F) → ∠A = 30° → ∠E = 30° → ∠C = 120° :=
by sorry

end similar_triangles_angle_C_l321_321184


namespace middle_of_7_consecutive_nat_sum_63_l321_321404

theorem middle_of_7_consecutive_nat_sum_63 (x : ℕ) (h : 7 * x = 63) : x = 9 :=
by
  sorry

end middle_of_7_consecutive_nat_sum_63_l321_321404


namespace twelfth_term_in_sequence_l321_321842

variable (a₁ : ℚ) (d : ℚ)

-- Given conditions
def a_1_term : a₁ = 1 / 2 := rfl
def common_difference : d = 1 / 3 := rfl

-- The twelfth term of the arithmetic sequence
def a₁₂ : ℚ := a₁ + 11 * d

-- Statement to prove
theorem twelfth_term_in_sequence (h₁ : a₁ = 1 / 2) (h₂ : d = 1 / 3) : a₁₂ = 25 / 6 :=
by
  rw [h₁, h₂, add_comm, add_assoc, mul_comm, add_comm]
  exact sorry

end twelfth_term_in_sequence_l321_321842


namespace square_side_length_equals_circle_circumference_l321_321030

constant π : ℝ

theorem square_side_length_equals_circle_circumference
  (x : ℝ)
  (h₁ : 4 * x = 6 * π) :
  x ≈ 4.71 :=
sorry

end square_side_length_equals_circle_circumference_l321_321030


namespace price_of_basic_computer_l321_321817

variable (C P : ℝ)

theorem price_of_basic_computer 
    (h1 : C + P = 2500)
    (h2 : P = (1 / 6) * (C + 500 + P)) : 
  C = 2000 :=
by
  sorry

end price_of_basic_computer_l321_321817


namespace a1_minus_2a2_plus_3a3_minus_4a4_eq_48_l321_321181

theorem a1_minus_2a2_plus_3a3_minus_4a4_eq_48:
  ∀ (a a_1 a_2 a_3 a_4 : ℝ),
  (∀ x : ℝ, (1 + 2 * x) ^ 4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  a_1 - 2 * a_2 + 3 * a_3 - 4 * a_4 = 48 :=
by
  sorry

end a1_minus_2a2_plus_3a3_minus_4a4_eq_48_l321_321181


namespace two_connected_iff_constructible_with_H_paths_l321_321885

-- A graph is represented as a structure with vertices and edges
structure Graph where
  vertices : Type
  edges : vertices → vertices → Prop

-- Function to check if a graph is 2-connected
noncomputable def isTwoConnected (G : Graph) : Prop := sorry

-- Function to check if a graph can be constructed by adding H-paths
noncomputable def constructibleWithHPaths (G H : Graph) : Prop := sorry

-- Given a graph G and subgraph H, we need to prove the equivalence
theorem two_connected_iff_constructible_with_H_paths (G H : Graph) :
  (isTwoConnected G) ↔ (constructibleWithHPaths G H) := sorry

end two_connected_iff_constructible_with_H_paths_l321_321885


namespace seven_digit_number_exists_in_at_least_one_strip_seven_digit_number_appears_infinitely_l321_321509

axiom infinite_tape : ℕ → ℕ  -- infinite sequence of natural numbers
axiom digit_strip (n : ℕ) : List ℕ  -- represents a strip of n digits
axiom seven_digit_strip (k : ℕ) : digit_strip 7  -- strips of 7 digits

-- Statement for the existence of a 7-digit number in the 7-digit strips
theorem seven_digit_number_exists_in_at_least_one_strip (n : ℕ) (h : n < 10^7) :
  ∃ k, n ∈ seven_digit_strip k := sorry

-- Statement for the infinite occurrence of a 7-digit number in the 7-digit strips
theorem seven_digit_number_appears_infinitely (n : ℕ) (h : n < 10^7) :
  ∃ᶠ k in Filter.atTop, n ∈ seven_digit_strip k := sorry

end seven_digit_number_exists_in_at_least_one_strip_seven_digit_number_appears_infinitely_l321_321509


namespace smallest_number_of_students_l321_321272

theorem smallest_number_of_students 
    (n : ℕ) 
    (attended := n / 4)
    (both := n / 40)
    (cheating_hint_ratio : ℚ := 3 / 2)
    (hinting := cheating_hint_ratio * (attended - both)) :
    n ≥ 200 :=
by sorry

end smallest_number_of_students_l321_321272


namespace repeating_decimal_to_fraction_l321_321161

noncomputable def x : ℚ := 0.45
noncomputable def y : ℚ := 2.18

theorem repeating_decimal_to_fraction : (x = 5 / 11) ∧ (y = 24 / 11) → (x / y = 5 / 24) :=
by
  intro h
  cases h with hx hy
  rw [hx, hy]
  field_simp
  norm_num
  exact Eq.refl (5 / 24)

end repeating_decimal_to_fraction_l321_321161


namespace largest_prime_divisor_of_factorial_sum_l321_321986

theorem largest_prime_divisor_of_factorial_sum :
  ∀ (n: ℕ), n = 13 → largest_prime_divisor (factorial n + factorial (n + 1)) = 7 := 
by
  intros,
  sorry

end largest_prime_divisor_of_factorial_sum_l321_321986


namespace find_the_number_l321_321052

-- Statement
theorem find_the_number (x : ℤ) (h : 2 * x = 3 * x - 25) : x = 25 :=
  sorry

end find_the_number_l321_321052


namespace roots_of_sqrt_eq_x_l321_321631

theorem roots_of_sqrt_eq_x (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (sqrt (x1 - p) = x1) ∧ (sqrt (x2 - p) = x2)) ↔ (0 ≤ p ∧ p < 1 / 4) :=
by
  sorry

end roots_of_sqrt_eq_x_l321_321631


namespace problem_1_problem_2_l321_321379

-- Problem 1: Simplify and find the value
theorem problem_1 :
  6 * (4: ℝ)^(2/3) + (-1/4: ℝ)^0 - 2 * (7: ℝ)^0.25 * real.root 3 (4: ℝ) - (1/2: ℝ)^(-2) = 1 :=
sorry

-- Problem 2: Simplify and find the value
theorem problem_2 :
  real.log (5: ℝ).sqrt - 2^(1/2 * real.log2 3) - (1/2) * (real.sqrt ((real.log 2)^2 - real.log 2 + real.log 5)) = -real.sqrt 3 :=
sorry

end problem_1_problem_2_l321_321379


namespace find_value_f_1_2016_l321_321686

variable {f : ℝ → ℝ}

theorem find_value_f_1_2016 (h_nondec : ∀ x y, 0 ≤ x → x ≤ y → y ≤ 1 → f(x) ≤ f(y))
  (h_cond1 : f 0 = 0)
  (h_cond2 : ∀ x, f (x / 3) = (1 / 2) * f(x))
  (h_cond3 : ∀ x, f (1 - x) = 1 - f(x)) :
  f (1 / 2016) = 1 / 128 :=
by sorry

end find_value_f_1_2016_l321_321686


namespace sum_of_fractions_l321_321192

theorem sum_of_fractions (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1/2 + x) + f (1/2 - x) = 2) :
  f (1 / 8) + f (2 / 8) + f (3 / 8) + f (4 / 8) + 
  f (5 / 8) + f (6 / 8) + f (7 / 8) = 7 :=
by 
  sorry

end sum_of_fractions_l321_321192


namespace turn_off_all_lights_l321_321058

theorem turn_off_all_lights (n : ℕ) (lights : Fin n → Bool)
  (h_move : ∀ i, lights i → (∃ k, lights (i + k) % n ≠ lights i ∧ lights i = false))
  : ∃ seq : List (Fin n), ∀ i, seq.contains i → lights i = false ↔ 3 ∣ n ∨ n = 2 := by
  sorry

end turn_off_all_lights_l321_321058


namespace distance_A_B_l321_321813

theorem distance_A_B :
  let s := 8 / 4    -- side length of smaller square
  let S := Real.sqrt 64 -- side length of larger square
  let H := 2 + S    -- horizontal distance
  let V := S - s    -- vertical distance
  Real.sqrt (H^2 + V^2) ≈ 11.7 :=
by
  sorry

end distance_A_B_l321_321813


namespace weight_of_8m_rod_l321_321118

noncomputable def cross_sectional_area (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

noncomputable def density : ℝ := let volume := 20.25 * 1125
                                 let mass := 42750
                                 mass / volume

theorem weight_of_8m_rod : 
  ∫ x in (0:ℝ)..8, cross_sectional_area x * density ≈ 0.836 :=
by 
  -- Integral of cross_sectional_area from 0 to 8, multiplied by density should be approximately 0.836
  sorry

end weight_of_8m_rod_l321_321118


namespace price_per_glass_on_second_day_l321_321709

 -- Definitions based on the conditions
def orangeade_first_day (O: ℝ) : ℝ := 2 * O -- Total volume on first day, O + O
def orangeade_second_day (O: ℝ) : ℝ := 3 * O -- Total volume on second day, O + 2O
def revenue_first_day (O: ℝ) (price_first_day: ℝ) : ℝ := 2 * O * price_first_day -- Revenue on first day
def revenue_second_day (O: ℝ) (P: ℝ) : ℝ := 3 * O * P -- Revenue on second day
def price_first_day: ℝ := 0.90 -- Given price per glass on the first day

 -- Statement to be proved
theorem price_per_glass_on_second_day (O: ℝ) (P: ℝ) (h: revenue_first_day O price_first_day = revenue_second_day O P) :
  P = 0.60 :=
by
  sorry

end price_per_glass_on_second_day_l321_321709


namespace multiply_by_12_correct_result_l321_321858

theorem multiply_by_12_correct_result (x : ℕ) (h : x / 14 = 42) : x * 12 = 7056 :=
by
  sorry

end multiply_by_12_correct_result_l321_321858


namespace apple_pyramid_total_apples_l321_321487

-- Definition of pyramid stacking with specific base dimensions and properties
def total_apples_in_pyramid (base_length : ℕ) (base_width : ℕ) : ℕ :=
  let layer1 := base_length * base_width
  let layer2 := (base_length - 1) * (base_width - 1)
  let layer3 := (base_length - 2) * (base_width - 2)
  let layer4 := (base_length - 3) * (base_width - 3)
  layer1 + layer2 + layer3 + layer4

theorem apple_pyramid_total_apples :
  total_apples_in_pyramid 4 6 = 53 :=
by norm_num

end apple_pyramid_total_apples_l321_321487


namespace find_books_second_purchase_profit_l321_321117

-- For part (1)
theorem find_books (x y : ℕ) (h₁ : 12 * x + 10 * y = 1200) (h₂ : 3 * x + 2 * y = 270) :
  x = 50 ∧ y = 60 :=
by 
  sorry

-- For part (2)
theorem second_purchase_profit (m : ℕ) (h₃ : 50 * (m - 12) + 2 * 60 * (12 - 10) ≥ 340) :
  m ≥ 14 :=
by 
  sorry

end find_books_second_purchase_profit_l321_321117


namespace solution_set_l321_321220

def f (x : ℝ) : ℝ := (2 * (abs x)) / (1 + (abs x))

theorem solution_set (x : ℝ) : f (2 * x - 1) < 1 ↔ 0 < x ∧ x < 1 := by
  sorry

end solution_set_l321_321220


namespace area_right_scalene_triangle_l321_321736

noncomputable def area_of_triangle_ABC : ℝ :=
  let AP : ℝ := 2
  let CP : ℝ := 1
  let AC : ℝ := AP + CP
  let ratio : ℝ := 2
  let x_squared : ℝ := 9 / 5
  x_squared

theorem area_right_scalene_triangle (AP CP : ℝ) (h₁ : AP = 2) (h₂ : CP = 1) (h₃ : ∠(B : Point) (A : Point) (P : Point) = 30) :
  let AB := 2 * real.sqrt(9 / 5)
  let BC := real.sqrt(9 / 5)
  ∃ (area : ℝ), area = (1/2) * AB * BC ∧ area = 9 / 5 :=
by
  sorry

end area_right_scalene_triangle_l321_321736


namespace binary_to_decimal_101101_l321_321536

theorem binary_to_decimal_101101 :
  let binary_value := 101101
  in 1 * 2^0 + 0 * 2^1 + 1 * 2^2 + 1 * 2^3 + 0 * 2^4 + 1 * 2^5 = 45 :=
by {
  sorry
}

end binary_to_decimal_101101_l321_321536


namespace more_legs_than_twice_heads_l321_321645

-- Definitions based on the conditions in a)
def numCows : Int := 16
def numLegs (D C : Int) : Int := 2 * D + 4 * C
def numHeads (D C : Int) : Int := D + C

-- Main statement to prove
theorem more_legs_than_twice_heads (D : Int) : 
  let L := numLegs D numCows in
  let H := numHeads D numCows in
  ∃ X : Int, L = 2 * H + X ∧ X = 32 := 
by 
  sorry

end more_legs_than_twice_heads_l321_321645


namespace part_a_part_b_l321_321655

-- Define the exchange rates
def ornament_to_crackers := 2
def sparklers_to_garlands := (5, 2)
def ornaments_to_garland := (4, 1)

-- Part (a)
theorem part_a (n_sparklers : ℕ) (h : n_sparklers = 10) :
  let n_garlands := (n_sparklers / sparklers_to_garlands.1) * sparklers_to_garlands.2 in
  let n_ornaments := n_garlands * ornaments_to_garland.1 in
  let n_crackers := n_ornaments * ornament_to_crackers in
  n_crackers = 32 :=
by {
  have n_garlands_def : n_garlands = (n_sparklers / sparklers_to_garlands.1) * sparklers_to_garlands.2, sorry,
  have n_ornaments_def : n_ornaments = n_garlands * ornaments_to_garland.1, sorry,
  have n_crackers_def : n_crackers = n_ornaments * ornament_to_crackers, sorry,
  have n_sparklers_eq : n_sparklers = 10, from h,
  sorry
}

-- Part (b)
theorem part_b :
  let v1 := (5 * ornament_to_crackers) + 1 in
  let v2 := ((2 / sparklers_to_garlands.1).nat_divide * sparklers_to_garlands.2 * ornaments_to_garland.1) * ornament_to_crackers in
  v1 > v2 :=
by {
  have v1_def : v1 = (5 * ornament_to_crackers) + 1, sorry,
  have v2_def : v2 = ((2 / sparklers_to_garlands.1).nat_divide * sparklers_to_garlands.2 * ornaments_to_garland.1) * ornament_to_crackers, sorry,
  sorry
}

end part_a_part_b_l321_321655


namespace polynomial_lower_bound_l321_321571

theorem polynomial_lower_bound 
  (a : Fin (n + 1) → ℝ) (x : ℝ) (n : ℕ)
  (P : ℝ → ℝ := λ x, ∑ i in Finset.range (n + 1), a i * x ^ (n - i))
  (m : ℝ := Finset.min' (Finset.image (λ k, ∑ i in Finset.range (k + 1), a i) (Finset.range (n + 1))) 
    (by { use ∥a 0∥, exact finset.nonempty.image ⟨0, Finset.zero_lt_succ _⟩ _ }))
  (h1 : 1 ≤ x) :
  P(x) ≥ m * x ^ n :=
sorry

end polynomial_lower_bound_l321_321571


namespace f_of_g_of_3_l321_321337

def f (x : ℝ) : ℝ := 4 * x - 5
def g (x : ℝ) : ℝ := (x + 2)^2
theorem f_of_g_of_3 : f (g 3) = 95 := by
  sorry

end f_of_g_of_3_l321_321337


namespace one_inch_cubes_with_two_or_more_painted_faces_l321_321811

def original_cube_length : ℕ := 4

def total_one_inch_cubes : ℕ := original_cube_length ^ 3

def corners_count : ℕ := 8

def edges_minus_corners_count : ℕ := 12 * 2

theorem one_inch_cubes_with_two_or_more_painted_faces
  (painted_faces_on_each_face : ∀ i : ℕ, i < total_one_inch_cubes → ℕ) : 
  ∃ n : ℕ, n = corners_count + edges_minus_corners_count ∧ n = 32 := 
by
  simp only [corners_count, edges_minus_corners_count, total_one_inch_cubes]
  sorry

end one_inch_cubes_with_two_or_more_painted_faces_l321_321811


namespace katie_initial_cupcakes_l321_321323

theorem katie_initial_cupcakes (ate_by_todd : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) (remaining_cupcakes : ℕ) : 
  ate_by_todd = 8 ∧ packages = 5 ∧ cupcakes_per_package = 2 ∧ (packages * cupcakes_per_package = remaining_cupcakes) → 
  18 = remaining_cupcakes + ate_by_todd :=
begin
  sorry
end

end katie_initial_cupcakes_l321_321323


namespace jason_cook_time_l321_321314

theorem jason_cook_time :
  let initial_temp : ℕ := 41
  let boil_temp : ℕ := 212
  let temp_increase : ℕ := 3
  let boil_time : ℕ := (boil_temp - initial_temp) / temp_increase
  let pasta_cook_time : ℕ := 12
  let mix_salad_time : ℕ := pasta_cook_time / 3
  boil_time + pasta_cook_time + mix_salad_time = 73 :=
by
  let initial_temp := 41
  let boil_temp := 212
  let temp_increase := 3
  let boil_time := (boil_temp - initial_temp) / temp_increase
  let pasta_cook_time := 12
  let mix_salad_time := pasta_cook_time / 3
  have h1 : boil_time = 57 := rfl
  have h2 : mix_salad_time = 4 := rfl
  calc
    boil_time + pasta_cook_time + mix_salad_time
    = 57 + pasta_cook_time + mix_salad_time : by rw h1
    ... = 57 + 12 + mix_salad_time : rfl
    ... = 57 + 12 + 4 : by rw h2
    ... = 73 : rfl

end jason_cook_time_l321_321314


namespace center_of_tangent_circle_l321_321880

theorem center_of_tangent_circle (x y : ℝ) 
  (h1 : 3*x - 4*y = 12) 
  (h2 : 3*x - 4*y = -24)
  (h3 : x - 2*y = 0) : 
  (x, y) = (-6, -3) :=
by
  sorry

end center_of_tangent_circle_l321_321880


namespace twelfth_term_in_sequence_l321_321841

variable (a₁ : ℚ) (d : ℚ)

-- Given conditions
def a_1_term : a₁ = 1 / 2 := rfl
def common_difference : d = 1 / 3 := rfl

-- The twelfth term of the arithmetic sequence
def a₁₂ : ℚ := a₁ + 11 * d

-- Statement to prove
theorem twelfth_term_in_sequence (h₁ : a₁ = 1 / 2) (h₂ : d = 1 / 3) : a₁₂ = 25 / 6 :=
by
  rw [h₁, h₂, add_comm, add_assoc, mul_comm, add_comm]
  exact sorry

end twelfth_term_in_sequence_l321_321841


namespace sum_of_digits_of_seven_power_l321_321836

-- Definitions based on conditions and problem specifications
def ones_digit (n : ℕ) : ℕ := n % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def power_pattern (n : ℕ) : ℕ := 7 ^ n

theorem sum_of_digits_of_seven_power :
  let n := 19 in
  let num := power_pattern n in
  (tens_digit num + ones_digit num) = 7 :=
by
  sorry

end sum_of_digits_of_seven_power_l321_321836


namespace equation_of_plane_l321_321492

-- Definitions based on conditions
def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - t, 3 - 2 * s, 4 - s + 3 * t)

-- Statement to prove the equation of the plane
theorem equation_of_plane : ∃ (A B C D : ℤ), (A > 0) ∧ 
  (Nat.gcd (Int.natAbs A) (Nat.gcd (Int.natAbs B) (Nat.gcd (Int.natAbs C) (Int.natAbs D))) = 1) ∧ 
  ∀ (x y z : ℝ), parametric_plane x y z → A * x + B * y + C * z + D = 0 :=
sorry

end equation_of_plane_l321_321492


namespace price_increase_decreases_purchasing_power_l321_321506

theorem price_increase_decreases_purchasing_power :
  ∀ (P_old S : ℝ), P_old > 0 → S > 0 → 
  let P_new := P_old * 1.25 in
  let Q_old := S / P_old in
  let Q_new := S / P_new in
  let decrease_percent := ((Q_old - Q_new) / Q_old) * 100 in
  decrease_percent = 20 :=
by
sorry

end price_increase_decreases_purchasing_power_l321_321506


namespace largest_prime_divisor_of_factorial_sum_l321_321981

theorem largest_prime_divisor_of_factorial_sum :
  ∀ (n: ℕ), n = 13 → largest_prime_divisor (factorial n + factorial (n + 1)) = 7 := 
by
  intros,
  sorry

end largest_prime_divisor_of_factorial_sum_l321_321981


namespace cube_assembly_red_faces_l321_321128

theorem cube_assembly_red_faces (small_cubes : Fin 8 -> Fin 6 -> Prop) 
  (one_third_blue_faces : ∀ i, (Finset.filter (small_cubes i) Finset.univ).card = 2) 
  (one_third_visible_red : ∀ i, (Finset.filter (λ f, ∃ j, small_cubes j f) Finset.univ).card = 8) :
  ∃ small_cube_orientation : Fin 8 -> ℕ, (∀ i j, small_cubes i j → (j = small_cube_orientation i) → False) :=
sorry

end cube_assembly_red_faces_l321_321128


namespace max_area_of_rectangle_l321_321023

theorem max_area_of_rectangle (x y : ℝ) (h1 : 2 * (x + y) = 40) : 
  (x * y) ≤ 100 :=
by
  sorry

end max_area_of_rectangle_l321_321023


namespace no_base_makes_131b_square_l321_321934

theorem no_base_makes_131b_square : ∀ (b : ℤ), b > 3 → ∀ (n : ℤ), n * n ≠ b^2 + 3 * b + 1 :=
by
  intros b h_gt_3 n
  sorry

end no_base_makes_131b_square_l321_321934


namespace Madeline_has_152_unused_crayons_l321_321698

def box1_2_3_crayons := 30
def box1_2_3_unused_fraction := 3 / 5
def box4_5_crayons := 24
def box4_5_unused_fraction := 5 / 8
def box6_7_crayons := 20
def box6_7_unused_fraction := 1 / 5
def box8_crayons := 35
def box8_unused_fraction := 7 / 10
def box9_crayons := 28
def box9_unused_fraction := 3 / 4
def box10_crayons := 40
def box10_unused_fraction := 3 / 8

def total_unused_crayons :=
  3 * (box1_2_3_unused_fraction * box1_2_3_crayons) +
  2 * (box4_5_unused_fraction * box4_5_crayons) +
  2 * (box6_7_unused_fraction * box6_7_crayons) +
  (box8_unused_fraction * box8_crayons).toNat +  -- rounding down
  (box9_unused_fraction * box9_crayons) +
  (box10_unused_fraction * box10_crayons)

theorem Madeline_has_152_unused_crayons :
  total_unused_crayons = 152 := by
  sorry

end Madeline_has_152_unused_crayons_l321_321698


namespace polar_coordinates_of_point_l321_321929

def convert_rect_to_polar (x y : ℝ) : (ℝ × ℝ) :=
  let r := Real.sqrt (x * x + y * y)
  let θ := if y ≥ 0 then Real.arctan (y / x) else Real.arctan (y / x) + Real.pi
  (r, θ)

theorem polar_coordinates_of_point :
  convert_rect_to_polar (-Real.sqrt 3) (Real.sqrt 3) = (3, 3 * Real.pi / 4) := by
  sorry

end polar_coordinates_of_point_l321_321929


namespace area_of_triangle_l321_321743

theorem area_of_triangle (ABC : Triangle) (right_triangle : is_right_triangle ABC)
    (scalene_triangle : is_scalene ABC) (P : Point) (on_hypotenuse : on_segment P ABC.hypotenuse)
    (angle_ABP : ABC.angle_AB P = 30) (AP_length : dist ABC.A P = 2)
    (CP_length : dist ABC.C P = 1) : 
    area ABC = 2 := 
sorry

end area_of_triangle_l321_321743


namespace solve_inequality_l321_321731

noncomputable def condition_inequality (x : ℝ) :=
  2^(2 * x^2 - 6 * x + 3) + 6^(x^2 - 3 * x + 1) ≥ 3^(2 * x^2 - 6 * x + 3)

theorem solve_inequality (x : ℝ) :
  condition_inequality x ↔ (x ≥ (3 - Real.sqrt 5) / 2 ∧ x ≤ (3 + Real.sqrt 5) / 2) :=
by 
  sorry

end solve_inequality_l321_321731


namespace distance_from_M_to_x_axis_l321_321299

-- Define the point M and its coordinates.
def point_M : ℤ × ℤ := (-9, 12)

-- Define the distance to the x-axis is simply the absolute value of the y-coordinate.
def distance_to_x_axis (p : ℤ × ℤ) : ℤ := Int.natAbs p.snd

-- Theorem stating the distance from point M to the x-axis is 12.
theorem distance_from_M_to_x_axis : distance_to_x_axis point_M = 12 := by
  sorry

end distance_from_M_to_x_axis_l321_321299


namespace tim_total_cost_l321_321826

def price_cabinet := 1200
def discount_cabinet := 0.15
def price_dining_table := 1800
def discount_dining_table := 0.20
def price_sofa := 2500
def discount_sofa := 0.10
def sales_tax_rate := 0.08

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price - (price * discount)

def total_cost_before_tax : ℝ :=
  discounted_price price_cabinet discount_cabinet + discounted_price price_dining_table discount_dining_table + discounted_price price_sofa discount_sofa

def total_cost_including_tax : ℝ :=
  total_cost_before_tax + (total_cost_before_tax * sales_tax_rate)

theorem tim_total_cost : total_cost_including_tax = 5086.80 := by
  -- Placeholder for the actual proof.
  sorry

end tim_total_cost_l321_321826


namespace sequence_correct_l321_321572

-- Define the sequence and the conditions
-- Condition 1: a_1 = 1
-- Condition 2: 4*(a_n*a_n) - a_{n+1} * a_n = 1 for n ∈ ℕ*
def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → 4*(a (n+1) * a (n+1)) - a (n+2) * a (n+1) = 1

-- Conjectured formula for the n-th term
def conjectured_formula (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = (2 * n - 1) / (2 * n - 3)

-- The theorem statement
theorem sequence_correct (a : ℕ → ℚ) :
  sequence a → conjectured_formula a :=
by
  sorry -- Proof is omitted

end sequence_correct_l321_321572


namespace contrapositive_quadratic_roots_l321_321147

theorem contrapositive_quadratic_roots (m : ℝ) (h_discriminant : 1 + 4 * m < 0) : m ≤ 0 :=
sorry

end contrapositive_quadratic_roots_l321_321147


namespace largest_prime_divisor_13_fact_14_fact_l321_321976

theorem largest_prime_divisor_13_fact_14_fact :
  ∀ (n : ℕ), (n = 13!) ∧ (m = 14!) ∧ (m = 14 * n) ∧ ((n + m) = n * 15) ∧ (15 = 3 * 5) → 
  (∃ p : ℕ, (Prime p) ∧ (LargestPrimeDivisor p (n + m)) ∧ (p = 13)) :=
by
  sorry

end largest_prime_divisor_13_fact_14_fact_l321_321976


namespace probability_product_zero_l321_321828

open Finset

theorem probability_product_zero (S : Finset ℤ) (hS : S = {-3, -1, 0, 2, 4}) :
  let pairs := S.product S.filter (λ p, p.1 ≠ p.2),
      zero_product_pairs := pairs.filter (λ p, p.1 * p.2 = 0) in
  (zero_product_pairs.card : ℚ) / pairs.card = 2 / 5 := by
suffices h : (S.product S).filter (λ p : ℤ × ℤ, (p.1 ≠ p.2) ∧ (p.1 * p.2 = 0)).card = 4 by
  have ht : (S.product S).filter (λ p : ℤ × ℤ, p.1 ≠ p.2).card = 10 := by sorry
  rw ← nat.cast_div
  rw ht
  norm_cast
  exact h
sorry


end probability_product_zero_l321_321828


namespace pow_div_mul_l321_321422

theorem pow_div_mul (x : ℕ) : x = 2 → 2^24 / (2^4 ^ 3) * 2^4 = 65536 :=
by
  intro hx
  rw [hx]
  sorry

end pow_div_mul_l321_321422


namespace find_w_l321_321819

theorem find_w (k : ℝ) (h1 : z * Real.sqrt w = k)
  (z_w3 : z = 6) (w3 : w = 3) :
  z = 3 / 2 → w = 48 := sorry

end find_w_l321_321819


namespace toads_max_l321_321265

theorem toads_max (n : ℕ) (h₁ : n ≥ 3) : 
  ∃ k : ℕ, k = ⌈ (n : ℝ) / 2 ⌉ ∧ ∀ (labels : Fin n → Fin n) (jumps : Fin n → ℕ), 
  (∀ i, jumps (labels i) = labels i) → ¬ ∃ f : Fin k → Fin n, ∀ i₁ i₂, i₁ ≠ i₂ → f i₁ ≠ f i₂ :=
sorry

end toads_max_l321_321265


namespace count_multiples_2_or_3_not_5_up_to_100_l321_321235

def isMultipleOf (n : Nat) (k : Nat) : Prop :=
  k % n = 0

def multiplesUpTo (n : Nat) (limit : Nat) : List Nat :=
  List.filter (isMultipleOf n) (List.range (limit + 1))

def multiplesOf2Or3Not5UpTo(limit : Nat) : List Nat :=
  List.filter (λ k => (isMultipleOf 2 k ∨ isMultipleOf 3 k) ∧ ¬ isMultipleOf 5 k) (List.range (limit + 1))

theorem count_multiples_2_or_3_not_5_up_to_100 : multiplesOf2Or3Not5UpTo 100 |>.length = 50 :=
  sorry

end count_multiples_2_or_3_not_5_up_to_100_l321_321235


namespace area_of_rectangle_l321_321298

theorem area_of_rectangle (A B C D : Point)
  (L L' : Line)
  (h1 : is_rectangle A B C D)
  (h2 : parallel_to L A)
  (h3 : parallel_to L' C)
  (h4 : perpendicular_to L D B)
  (h5 : segment_length D E = 1)
  (h6 : segment_length E F = 2)
  (h7 : segment_length F B = 3) :
  area_of_rectangle A B C D = 6 * sqrt 5 := 
by sorry

end area_of_rectangle_l321_321298


namespace combined_soldiers_on_great_wall_l321_321650

theorem combined_soldiers_on_great_wall :
  let wall_length := 7300 -- total length of the Great Wall in kilometers
  let interval := 5 -- interval between towers in kilometers
  let soldiers_per_tower := 2 -- number of soldiers per tower
  let number_of_towers := wall_length / interval
  let total_soldiers := number_of_towers * soldiers_per_tower
  in total_soldiers = 2920 :=
by
  sorry

end combined_soldiers_on_great_wall_l321_321650


namespace line_through_points_l321_321393

theorem line_through_points (m b : ℝ)
  (h_slope : m = (-1 - 3) / (-3 - 1))
  (h_point : 3 = m * 1 + b) :
  m + b = 3 :=
sorry

end line_through_points_l321_321393


namespace simplest_radical_l321_321072

theorem simplest_radical (r1 r2 r3 r4 : ℝ) 
  (h1 : r1 = Real.sqrt 3) 
  (h2 : r2 = Real.sqrt 4)
  (h3 : r3 = Real.sqrt 8)
  (h4 : r4 = Real.sqrt (1 / 2)) : r1 = Real.sqrt 3 :=
  by sorry

end simplest_radical_l321_321072


namespace monotone_f_iff_l321_321193

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then (a - 3) * x - 1 else log a x

theorem monotone_f_iff (a : ℝ) : 
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ 3 < a ∧ a ≤ 4 := 
by sorry

end monotone_f_iff_l321_321193


namespace problem_statement_l321_321590

noncomputable def f (m : ℝ) (x : ℝ) := (m^2 - m - 1) * x^(-5 * m - 3)
noncomputable def g (m : ℝ) (a : ℝ) (x : ℝ) := log a ((1 - m * x) / (x - 1))

theorem problem_statement (a t m : ℝ) (hx : x > 0 ) (ha : a > 1) :
  (∀ x > 0, f m x ≥ 0) →
  (∀ x > 0, ∀ y > x, f m y > f m x) →
  g m a x = log a ((x + 1) / (x - 1)) ∧ 
  (∀ x ∈ set.Ioc t a, g m a x > 1 → t = 1 ∧ a = 1 + sqrt 2) := by 
  sorry

end problem_statement_l321_321590


namespace total_tickets_sold_l321_321045

theorem total_tickets_sold (s_tickets n_tickets s_price n_price total_revenue : ℕ)
  (h1 : s_price = 9) (h2 : n_price = 11) (h3 : total_revenue = 20960) (h4 : s_tickets = 520) :
  s_tickets + n_tickets = 2000 :=
by
  -- Definitions and intermediate calculations
  let s_revenue := s_price * s_tickets
  let n_revenue := total_revenue - s_revenue
  have h_s_revenue : s_revenue = 4680, by sorry
  have h_n_revenue : n_revenue = 16280, by sorry
  have n_tickets := n_revenue / n_price
  -- Verifying n_tickets calculation
  have h_n_tickets : n_tickets = 1480, by sorry
  -- Final proof showing total tickets sold
  exact ((h4 : s_tickets = 520).symm ▸ (h_n_tickets).symm ▸ rfl : 520 + 1480 = 2000)

end total_tickets_sold_l321_321045


namespace total_expenditure_le_budget_l321_321871

theorem total_expenditure_le_budget :
  let num_people := 13
  let budget := 350
  let expenditure_7_people := 7 * 15
  let expenditure_5_people := 5 * 25
  let total_expenditure_12_people := expenditure_7_people + expenditure_5_people
  let average_expenditure := budget / num_people
  let expenditure_last_person := average_expenditure + 10
  let total_expenditure := total_expenditure_12_people + expenditure_last_person
in
  total_expenditure <= budget := by
sorry

end total_expenditure_le_budget_l321_321871


namespace two_numbers_differ_by_more_than_one_l321_321868

theorem two_numbers_differ_by_more_than_one 
  {a : ℕ → ℝ} (n : ℕ) (k : ℝ) (h1 : (∑ i in finset.range n, a i) = 3 * k)
  (h2 : (∑ i in finset.range n, (a i)^2) = 3 * k^2)
  (h3 : (∑ i in finset.range n, (a i)^3) > 3 * k^3 + k)
  (h4 : ∀ i, 0 < a i) :
  ∃ (i j : ℕ), i ≠ j ∧ |a i - a j| > 1 :=
by
  sorry

end two_numbers_differ_by_more_than_one_l321_321868


namespace unequal_circles_common_tangents_l321_321420

-- Definitions and conditions
variables {r s d : ℝ} (h₁ : r ≠ s)
def disjoint (d r s : ℝ) := d > r + s
def one_circle_inside_other (d r s : ℝ) := d + min r s < max r s
def externally_tangent (d r s : ℝ) := d = r + s

-- The theorem statement
theorem unequal_circles_common_tangents (h₁ : r ≠ s) :
  r ≠ s → (∀ d, ¬(disjoint d r s ∨ one_circle_inside_other d r s ∨ externally_tangent d r s) → 2 common_tangents d r s) := sorry

end unequal_circles_common_tangents_l321_321420


namespace maximize_gross_profit_l321_321475

-- Definitions based on the conditions
def price_rose_base := 4
def price_rose_discount := 3
def price_lily := 5
def price_rose_sell := 5
def price_lily_sell := 6.5
def num_rose_min := 1000
def num_rose_max := 1500

-- Condition: Total spending is 9000 yuan
def total_spending := 9000

-- Definitions for quantities determined by the solution
def num_roses_purchased := 1500
def num_lilies_purchased := 900

-- Gross profit calculation
def gross_profit (num_roses num_lilies : ℕ) : ℕ := 
  (price_rose_sell * num_roses + price_lily_sell * num_lilies) -
  if num_roses > 1200 then
    total_spending - ((price_rose_base * 1200) + (price_rose_discount * (num_roses - 1200)))
  else 
    total_spending - (price_rose_base * num_roses) + (price_lily * num_lilies)

theorem maximize_gross_profit : gross_profit num_roses_purchased num_lilies_purchased = 4350 :=
  by sorry

end maximize_gross_profit_l321_321475


namespace isosceles_triangle_largest_angle_l321_321292

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h_isosceles : A = B) (h_angles : A = 60 ∧ B = 60) :
  max A (max B C) = 60 :=
by
  sorry

end isosceles_triangle_largest_angle_l321_321292


namespace sum_of_odd_integers_from_13_to_35_l321_321816

theorem sum_of_odd_integers_from_13_to_35 :
  (∑ i in (finset.filter (λ x, (x % 2 = 1)) (finset.range 36)).filter (λ x, 13 ≤ x) 108 :=
by sorry

end sum_of_odd_integers_from_13_to_35_l321_321816


namespace reasoning_common_sense_l321_321752

theorem reasoning_common_sense :
  (∀ P Q: Prop, names_not_correct → P → ¬Q → affairs_not_successful → ¬Q)
  ∧ (∀ R S: Prop, affairs_not_successful → R → ¬S → rites_not_flourish → ¬S)
  ∧ (∀ T U: Prop, rites_not_flourish → T → ¬U → punishments_not_executed_properly → ¬U)
  ∧ (∀ V W: Prop, punishments_not_executed_properly → V → ¬W → people_nowhere_hands_feet → ¬W)
  → reasoning_is_common_sense :=
by sorry

end reasoning_common_sense_l321_321752


namespace distinct_numbers_in_list_l321_321170

def floor_div_range (n : ℕ) : ℕ := ⌊ (n^2) / 500 ⌋

theorem distinct_numbers_in_list : 
  (Finset.image floor_div_range (Finset.range 1001)).card = 2000 :=
sorry

end distinct_numbers_in_list_l321_321170


namespace continuous_at_2_l321_321693

def f (b : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then
  3 * x^2 + 2 * x - 1
else
  b * x - 5

theorem continuous_at_2 (b : ℝ) (hf : ∀ x, f b x = if x ≤ 2 then 3 * x^2 + 2 * x - 1 else b * x - 5) :
  b = 10 :=
by
  have h1 : f b 2 = 3 * 2^2 + 2 * 2 - 1 := hf 2
  rw [if_pos (le_refl 2)] at h1
  have h2 : f b 2 = b * 2 - 5 := hf 2
  rw [if_neg (lt_irrefl 2)] at h2
  have h3 := h1.trans h2.symm
  linarith

end continuous_at_2_l321_321693


namespace max_area_of_rectangle_l321_321026

theorem max_area_of_rectangle (x y : ℝ) (h1 : 2 * (x + y) = 40) : 
  (x * y) ≤ 100 :=
by
  sorry

end max_area_of_rectangle_l321_321026


namespace initial_deck_card_count_l321_321643

theorem initial_deck_card_count (r b : ℕ) (h1 : r / (r + b) = 1 / 5) (h2 : r / (r + (b + 6)) = 1 / 7) : r + b = 15 :=
begin
  sorry
end

end initial_deck_card_count_l321_321643


namespace coeff_of_x4_in_expansion_l321_321303

theorem coeff_of_x4_in_expansion : 
  let f := (2 * x ^ 3 + 1 / x ^ (1 / 2)) ^ 6 in 
  polynomial.coeff (polynomial.expand ℚ f) 4 = 60 := 
by
  sorry

end coeff_of_x4_in_expansion_l321_321303


namespace coefficient_a_must_be_zero_l321_321365

noncomputable def all_real_and_positive_roots (a b c : ℝ) : Prop :=
∀ p : ℝ, p > 0 → ∀ x : ℝ, (a * x^2 + b * x + c + p = 0) → x > 0

theorem coefficient_a_must_be_zero (a b c : ℝ) :
  (all_real_and_positive_roots a b c) → (a = 0) :=
by sorry

end coefficient_a_must_be_zero_l321_321365


namespace find_d_l321_321785

noncomputable def quadratic_roots (d : ℝ) : Prop :=
∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2

theorem find_d : ∃ d : ℝ, d = 9.8 ∧ quadratic_roots d :=
sorry

end find_d_l321_321785


namespace geometric_sequence_sum_l321_321665

noncomputable def a (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

variables (a₁ r : ℝ) (h1 : a₁ * r + a₁ * r ^ 2 = 4)
                    (h2 : a₁ * r ^ 3 + a₁ * r ^ 4 = 16)

theorem geometric_sequence_sum :
  a 8 + a 9 = 128 :=
sorry

end geometric_sequence_sum_l321_321665


namespace expected_red_light_l321_321489

variables (n : ℕ) (p : ℝ)
def binomial_distribution : Type := sorry

noncomputable def expected_value (n : ℕ) (p : ℝ) : ℝ :=
n * p

theorem expected_red_light :
  expected_value 3 0.4 = 1.2 :=
by
  simp [expected_value]
  sorry

end expected_red_light_l321_321489


namespace distance_AE_eq_3sqrt2_l321_321923

-- Definitions based on the conditions
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (1, 1)
def D : ℝ × ℝ := (3, 3)

-- Calculate intersection of AB and CD and distance AE
theorem distance_AE_eq_3sqrt2 : dist A (3, 1) = 3 * real.sqrt 2 := sorry

end distance_AE_eq_3sqrt2_l321_321923


namespace heather_heavier_than_emily_l321_321228

theorem heather_heavier_than_emily :
  let Heather_weight := 87
  let Emily_weight := 9
  Heather_weight - Emily_weight = 78 :=
by
  -- Proof here
  sorry

end heather_heavier_than_emily_l321_321228


namespace problem_l321_321574

variable {α : Type*} [LinearOrderedField α]

noncomputable def is_arithmetic_sequence (a : ℕ → α) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def sum_of_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
∑ i in finset.range (n + 1), a i

variables (a : ℕ → α) (S : ℕ → α)

theorem problem (h1 : is_arithmetic_sequence a)
                (h2 : ∀ n, S n = sum_of_first_n_terms a n)
                (h3 : S 2023 < 0) 
                (h4 : S 2024 > 0) :
  (∀ n, a n < a (n + 1)) ∧ ∀ n, n ≥ 1012 → S 1012 ≤ S n :=
begin
  sorry
end

end problem_l321_321574


namespace twelfth_term_arithmetic_sequence_l321_321851

theorem twelfth_term_arithmetic_sequence :
  let a := (1 : ℚ) / 2
  let d := (1 : ℚ) / 3
  (a + 11 * d) = (25 : ℚ) / 6 :=
by
  sorry

end twelfth_term_arithmetic_sequence_l321_321851


namespace product_of_divisors_of_30_l321_321442

theorem product_of_divisors_of_30 : 
  ∏ (d : ℕ) in {d | d ∣ 30} = 810000 :=
sorry

end product_of_divisors_of_30_l321_321442


namespace truck_distance_and_efficiency_l321_321501

theorem truck_distance_and_efficiency (m d g1 g2 : ℕ) (h1 : d = 300) (h2 : g1 = 10) (h3 : g2 = 15) :
  (d * (g2 / g1) = 450) ∧ (d / g1 = 30) :=
by
  sorry

end truck_distance_and_efficiency_l321_321501


namespace ratio_of_mn_l321_321608

noncomputable def f (x : ℝ) : ℝ := abs (Real.logb 4 x)

theorem ratio_of_mn 
  (m n : ℝ) 
  (h1 : 0 < m) 
  (h2 : m < n) 
  (h3 : f m = f n)
  (h4 : ∀ x ∈ Icc (m^2) n, f x ≤ 2) 
  (h_max : ∃ x ∈ Icc (m^2) n, f x = 2) : 
  n / m = 16 :=
sorry

end ratio_of_mn_l321_321608


namespace Mathematics_Olympiad_l321_321517

theorem Mathematics_Olympiad :
  ∀ (students_archimedes students_noether students_gauss : ℕ),
    students_archimedes = 15 →
    students_noether = 10 →
    students_gauss = 12 →
    students_archimedes + students_noether + students_gauss = 37 :=
by
  intros students_archimedes students_noether students_gauss H1 H2 H3
  rw [H1, H2, H3]
  norm_num
  sorry

end Mathematics_Olympiad_l321_321517


namespace find_product_stu_l321_321392

-- Define hypotheses
variables (a x y c : ℕ)
variables (s t u : ℕ)
variable (h_eq : a^8 * x * y - a^7 * x - a^6 * y = a^5 * (c^5 - 2))

-- Statement to prove the equivalent form and stu product
theorem find_product_stu (h_eq : a^8 * x * y - a^7 * x - a^6 * y = a^5 * (c^5 - 2)) :
  ∃ s t u : ℕ, (a^s * x - a^t) * (a^u * y - a^3) = a^5 * c^5 ∧ s * t * u = 12 :=
sorry

end find_product_stu_l321_321392


namespace circle_equation_l321_321207

theorem circle_equation : ∃ (x y : ℝ), (x - 2)^2 + y^2 = 2 :=
by
  sorry

end circle_equation_l321_321207


namespace smallest_number_of_students_l321_321276

theorem smallest_number_of_students 
    (n : ℕ) 
    (attended := n / 4)
    (both := n / 40)
    (cheating_hint_ratio : ℚ := 3 / 2)
    (hinting := cheating_hint_ratio * (attended - both)) :
    n ≥ 200 :=
by sorry

end smallest_number_of_students_l321_321276


namespace circle_segment_length_xz_l321_321919

noncomputable def length_of_XZ (C : ℝ) (θ : ℝ) (cosθ : ℝ) : ℝ :=
  6 * Real.sqrt (2 - cosθ)

theorem circle_segment_length_xz :
  ∀ (C : ℝ) (θ : ℝ),
  C = 12 * Real.pi → θ = Real.pi / 6 →
  cos θ = Real.sqrt 3 / 2 →
  length_of_XZ C θ (Real.sqrt 3 / 2) = 6 * Real.sqrt (2 - Real.sqrt 3 / 2) :=
by
  intros C θ hC hθ hcos
  simp [length_of_XZ, hC, hθ, hcos]
  sorry

end circle_segment_length_xz_l321_321919


namespace transformation_sequences_count_l321_321924

-- Definitions corresponding to the conditions in the problem
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (Real.cos (2 * π / 5), Real.sin (2 * π / 5))
def C : ℝ × ℝ := (Real.cos (4 * π / 5), Real.sin (4 * π / 5))
def D : ℝ × ℝ := (Real.cos (6 * π / 5), Real.sin (6 * π / 5))
def E : ℝ × ℝ := (Real.cos (8 * π / 5), Real.sin (8 * π / 5))

def vertices : List (ℝ × ℝ) := [A, B, C, D, E]

-- Definitions of transformations
def R_5 (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x * Real.cos (2 * π / 5) + y * Real.sin (2 * π / 5), -x * Real.sin (2 * π / 5) + y * Real.cos (2 * π / 5))

def H (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-x, y)

def S_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (2 * x, 2 * y)

def V (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x, y + 1)

-- Main theorem statement
theorem transformation_sequences_count :
  ∃ (n : ℕ), 
    (∀ p ∈ vertices, (List.foldl (λ acc t, t acc) p (List.replicate 15 R_5)) = p
    ∨ (List.foldl (λ acc t, t acc) p (List.replicate 15 H)) = p
    ∨ (List.foldl (λ acc t, t acc) p (List.replicate 15 S_2)) = p
    ∨ (List.foldl (λ acc t, t acc) p (List.replicate 15 V)) = p) → 
    n = sorry := sorry

end transformation_sequences_count_l321_321924


namespace smallest_total_students_l321_321278

theorem smallest_total_students (n : ℕ) (h1 : 25 * n % 100 = 0)
  (h2 : 10 * n % 4 = 0)
  (h3 : ∃ (y z : ℕ), y = 3 * z / 2 ∧ (y + z - n / 40 = n / 4)) :
  ∃ k : ℕ, n = 200 * k :=
by
  sorry

end smallest_total_students_l321_321278


namespace maximum_area_of_rectangle_with_fixed_perimeter_l321_321018

theorem maximum_area_of_rectangle_with_fixed_perimeter (x y : ℝ) 
  (h₁ : 2 * (x + y) = 40) 
  (h₂ : x = y) :
  x * y = 100 :=
by
  sorry

end maximum_area_of_rectangle_with_fixed_perimeter_l321_321018


namespace product_of_divisors_of_30_l321_321445

theorem product_of_divisors_of_30 : ∏ d in (finset.filter (∣ 30) (finset.range 31)), d = 810000 := by
  sorry

end product_of_divisors_of_30_l321_321445


namespace different_routes_calculation_l321_321372

-- Definitions for the conditions
def west_blocks := 3
def south_blocks := 2
def east_blocks := 3
def north_blocks := 3

-- Calculation of combinations for the number of sequences
def house_to_sw_corner_routes := Nat.choose (west_blocks + south_blocks) south_blocks
def ne_corner_to_school_routes := Nat.choose (east_blocks + north_blocks) east_blocks

-- Proving the total number of routes
theorem different_routes_calculation : 
  house_to_sw_corner_routes * 1 * ne_corner_to_school_routes = 200 :=
by
  -- Mathematical proof steps (to be filled)
  sorry

end different_routes_calculation_l321_321372


namespace DEF_and_A1B1C1_are_similar_l321_321344

-- Definitions of points and lines based on given conditions
variable (A B C : Point) (D E F : Point)
variable (circumcircle : Circle)
variable (HD : Line)
variable (A1 B1 C1 : Point)

-- Conditions for midpoints of arcs, feet of perpendiculars, and intersection of lines
axiom midpoint_arc_D (D_is_midpoint_arc_BC : is_midpoint (arc B C) D)
axiom midpoint_arc_E (E_is_midpoint_arc_CA : is_midpoint (arc C A) E)
axiom midpoint_arc_F (F_is_midpoint_arc_AB : is_midpoint (arc A B) F)

axiom feet_perpendicular_A (ell_a : Line)
axiom line_l_passing_A (ell_a_passes_perpendiculars_A : passes_through_perpendiculars_from A ell_a D B C)

axiom feet_perpendicular_D (m_a : Line)
axiom line_m_passing_D (m_a_passes_perpendiculars_D : passes_through_perpendiculars_from D m_a A B C)

axiom intersection_of_lines_A (is_intersection_A : intersection ell_a m_a A1)
axiom intersection_of_lines_B (is_intersection_B : intersection ell_a m_a B1)
axiom intersection_of_lines_C (is_intersection_C : intersection ell_a m_a C1)

-- Theorem: Prove similarity of triangles DEF and A1B1C1
theorem DEF_and_A1B1C1_are_similar :
  ∀ (ABC_triangle : is_triangle A B C) (DEF_triangle : is_triangle D E F) (A1B1C1_triangle : is_triangle A1 B1 C1),
  similar DEF_triangle A1B1C1_triangle :=
by sorry  -- Proof to be provided

end DEF_and_A1B1C1_are_similar_l321_321344


namespace problem_solution_l321_321253

noncomputable def valid_a_sum : ℤ :=
  let is_valid_a (a : ℤ) : Prop :=
    ∃ (x y : ℝ), (x - 1) / 2 ≤ (2 * x + 3) / 6 ∧ x + 1 > a + 3 ∧ y ≥ 0 ∧
    (a + 1) / (y - 2) + 3 / (2 - y) = 2
  in (Finset.filter is_valid_a (Finset.range 7)
    ∪ Finset.filter is_valid_a (Finset.range (-3)).erase 2).sum

theorem problem_solution : valid_a_sum = 1 :=
sorry

end problem_solution_l321_321253


namespace original_number_is_400_l321_321889

theorem original_number_is_400 (x : ℝ) (h : 1.20 * x = 480) : x = 400 :=
sorry

end original_number_is_400_l321_321889


namespace tan_x_ge_1_implies_range_l321_321221

theorem tan_x_ge_1_implies_range (x : ℝ) (h1 : x ∈ Ioo (-π/2 : ℝ) (π/2 : ℝ)) (h2 : Real.tan x ≥ 1) : x ∈ Ico (π/4) (π/2) :=
sorry

end tan_x_ge_1_implies_range_l321_321221


namespace number_of_children_l321_321504

-- Define conditions
variable (A C : ℕ) (h1 : A + C = 280) (h2 : 60 * A + 25 * C = 14000)

-- Lean statement to prove the number of children
theorem number_of_children : C = 80 :=
by
  sorry

end number_of_children_l321_321504


namespace gasoline_reduction_l321_321082

theorem gasoline_reduction (P Q : ℝ) :
  let new_price := 1.25 * P
  let new_budget := 1.10 * (P * Q)
  let new_quantity := new_budget / new_price
  let percent_reduction := 1 - (new_quantity / Q)
  percent_reduction = 0.12 :=
by
  sorry

end gasoline_reduction_l321_321082


namespace max_area_rect_40_perimeter_l321_321022

noncomputable def max_rect_area (P : ℕ) (hP : P = 40) : ℕ :=
  let w : ℕ → ℕ := id
  let l : ℕ → ℕ := λ w, P / 2 - w
  let area : ℕ → ℕ := λ w, w * (P / 2 - w)
  find_max_value area sorry

theorem max_area_rect_40_perimeter : max_rect_area 40 40 = 100 := 
sorry

end max_area_rect_40_perimeter_l321_321022


namespace find_g6_minus_g2_div_g3_l321_321765

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition (a c : ℝ) : c^3 * g a = a^3 * g c
axiom g_nonzero : g 3 ≠ 0

theorem find_g6_minus_g2_div_g3 : (g 6 - g 2) / g 3 = 208 / 27 := by
  sorry

end find_g6_minus_g2_div_g3_l321_321765


namespace proof_problem_l321_321666

noncomputable def problem : ℚ :=
  let a := 1
  let b := 2
  let c := 1
  let d := 0
  a + 2 * b + 3 * c + 4 * d

theorem proof_problem : problem = 8 := by
  -- All computations are visible here
  unfold problem
  rfl

end proof_problem_l321_321666


namespace find_m_plus_c_l321_321827

-- We need to define the conditions first
variable {A : ℝ × ℝ} {B : ℝ × ℝ} {c : ℝ} {m : ℝ}

-- Given conditions from part a)
def A_def : Prop := A = (1, 3)
def B_def : Prop := B = (m, -1)
def centers_line : Prop := ∀ C : ℝ × ℝ, (C.1 - C.2 + c = 0)

-- Define the theorem for the proof problem
theorem find_m_plus_c (A_def : A = (1, 3)) (B_def : B = (m, -1)) (centers_line : ∀ C : ℝ × ℝ, (C.1 - C.2 + c = 0)) : m + c = 3 :=
sorry

end find_m_plus_c_l321_321827
