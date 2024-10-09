import Mathlib

namespace mean_profit_first_15_days_l2365_236516

-- Definitions and conditions
def mean_daily_profit_entire_month : ℝ := 350
def total_days_in_month : ℕ := 30
def mean_daily_profit_last_15_days : ℝ := 445

-- Proof statement
theorem mean_profit_first_15_days : 
  (mean_daily_profit_entire_month * (total_days_in_month : ℝ) 
   - mean_daily_profit_last_15_days * 15) / 15 = 255 :=
by
  sorry

end mean_profit_first_15_days_l2365_236516


namespace only_solutions_l2365_236542

theorem only_solutions (m n : ℕ) (p : ℕ) (prime_p : Nat.Prime p) (condition : (Nat.choose m 2) - 1 = p^n) :
  (m = 5 ∧ n = 2 ∧ p = 3) ∨ (m = 8 ∧ n = 3 ∧ p = 3) :=
by
  sorry

end only_solutions_l2365_236542


namespace power_of_two_plus_one_is_power_of_integer_l2365_236517

theorem power_of_two_plus_one_is_power_of_integer (n : ℕ) (hn : 0 < n) (a k : ℕ) (ha : 2^n + 1 = a^k) (hk : 1 < k) : n = 3 :=
by
  sorry

end power_of_two_plus_one_is_power_of_integer_l2365_236517


namespace average_roots_of_quadratic_l2365_236560

open Real

theorem average_roots_of_quadratic (a b : ℝ) (h_eq : ∃ x1 x2 : ℝ, a * x1^2 - 2 * a * x1 + b = 0 ∧ a * x2^2 - 2 * a * x2 + b = 0):
  (b = b) → (a ≠ 0) → (h_discriminant : (2 * a)^2 - 4 * a * b ≥ 0) → (x1 + x2) / 2 = 1 :=
by
  sorry

end average_roots_of_quadratic_l2365_236560


namespace smallest_number_divisible_l2365_236566

theorem smallest_number_divisible (x : ℕ) :
  (∃ n : ℕ, x = n * 5 + 24) ∧
  (∃ n : ℕ, x = n * 10 + 24) ∧
  (∃ n : ℕ, x = n * 15 + 24) ∧
  (∃ n : ℕ, x = n * 20 + 24) →
  x = 84 :=
by
  sorry

end smallest_number_divisible_l2365_236566


namespace find_years_ago_twice_age_l2365_236563

-- Definitions of given conditions
def age_sum (H J : ℕ) : Prop := H + J = 43
def henry_age : ℕ := 27
def jill_age : ℕ := 16

-- Definition of the problem to be proved
theorem find_years_ago_twice_age (X : ℕ) 
  (h1 : age_sum henry_age jill_age) 
  (h2 : henry_age = 27) 
  (h3 : jill_age = 16) : (27 - X = 2 * (16 - X)) → X = 5 := 
by 
  sorry

end find_years_ago_twice_age_l2365_236563


namespace average_score_is_8_9_l2365_236538

-- Define the scores and their frequencies
def scores : List ℝ := [7.5, 8.5, 9, 10]
def frequencies : List ℕ := [2, 2, 3, 3]

-- Express the condition that the total number of shots is 10
def total_shots : ℕ := frequencies.sum

-- Calculate the weighted sum of the scores
def weighted_sum (scores : List ℝ) (frequencies : List ℕ) : ℝ :=
  (List.zip scores frequencies).foldl (λ acc (sc, freq) => acc + (sc * freq)) 0

-- Prove that the average score is 8.9
theorem average_score_is_8_9 :
  total_shots = 10 →
  weighted_sum scores frequencies / total_shots = 8.9 :=
by
  intros h_total_shots
  sorry

end average_score_is_8_9_l2365_236538


namespace sin_B_value_cos_A_minus_cos_C_value_l2365_236520

variables {A B C : ℝ} {a b c : ℝ}

theorem sin_B_value (h₁ : 4 * b * (Real.sin A) = Real.sqrt 7 * a) : Real.sin B = Real.sqrt 7 / 4 := 
sorry

theorem cos_A_minus_cos_C_value (h₁ : 4 * b * (Real.sin A) = Real.sqrt 7 * a) (h₂ : 2 * b = a + c) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 := 
sorry

end sin_B_value_cos_A_minus_cos_C_value_l2365_236520


namespace remainder_5310_mod8_l2365_236559

theorem remainder_5310_mod8 : (53 ^ 10) % 8 = 1 := 
by 
  sorry

end remainder_5310_mod8_l2365_236559


namespace combined_6th_grade_percentage_l2365_236586

noncomputable def percentage_of_6th_graders 
  (parkPercent : Fin 7 → ℚ) 
  (riversidePercent : Fin 7 → ℚ) 
  (totalParkside : ℕ) 
  (totalRiverside : ℕ) 
  : ℚ := 
    let num6thParkside := parkPercent 6 * totalParkside
    let num6thRiverside := riversidePercent 6 * totalRiverside
    let total6thGraders := num6thParkside + num6thRiverside
    let totalStudents := totalParkside + totalRiverside
    (total6thGraders / totalStudents) * 100

theorem combined_6th_grade_percentage :
  let parkPercent := ![(14.0 : ℚ) / 100, 13 / 100, 16 / 100, 15 / 100, 12 / 100, 15 / 100, 15 / 100]
  let riversidePercent := ![(13.0 : ℚ) / 100, 16 / 100, 13 / 100, 15 / 100, 14 / 100, 15 / 100, 14 / 100]
  percentage_of_6th_graders parkPercent riversidePercent 150 250 = 15 := 
  by
  sorry

end combined_6th_grade_percentage_l2365_236586


namespace total_books_is_correct_l2365_236592

-- Definitions based on the conditions
def initial_books_benny : Nat := 24
def books_given_to_sandy : Nat := 10
def books_tim : Nat := 33

-- Definition based on the computation in the solution
def books_benny_now := initial_books_benny - books_given_to_sandy
def total_books : Nat := books_benny_now + books_tim

-- The statement to be proven
theorem total_books_is_correct : total_books = 47 := by
  sorry

end total_books_is_correct_l2365_236592


namespace students_drawn_in_sample_l2365_236555

def total_people : ℕ := 1600
def number_of_teachers : ℕ := 100
def sample_size : ℕ := 80
def number_of_students : ℕ := total_people - number_of_teachers
def expected_students_sample : ℕ := 75

theorem students_drawn_in_sample : (sample_size * number_of_students) / total_people = expected_students_sample :=
by
  -- The proof steps would go here
  sorry

end students_drawn_in_sample_l2365_236555


namespace nancy_ate_3_apples_l2365_236505

theorem nancy_ate_3_apples
  (mike_apples : ℝ)
  (keith_apples : ℝ)
  (apples_left : ℝ)
  (mike_apples_eq : mike_apples = 7.0)
  (keith_apples_eq : keith_apples = 6.0)
  (apples_left_eq : apples_left = 10.0) :
  mike_apples + keith_apples - apples_left = 3.0 := 
by
  rw [mike_apples_eq, keith_apples_eq, apples_left_eq]
  norm_num

end nancy_ate_3_apples_l2365_236505


namespace distance_min_value_l2365_236531

theorem distance_min_value (a b c d : ℝ) 
  (h₁ : |b - (Real.log a) / a| + |c - d + 2| = 0) : 
  (a - c)^2 + (b - d)^2 = 9 / 2 :=
by {
  sorry
}

end distance_min_value_l2365_236531


namespace simplify_expression_l2365_236532

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : 
  (3 * x - 1 - 5 * x) / 3 = -(2 / 3) * x - (1 / 3) := 
by
  sorry

end simplify_expression_l2365_236532


namespace perfect_square_condition_l2365_236590

-- Definitions from conditions
def is_integer (x : ℝ) : Prop := ∃ k : ℤ, x = k

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m^2

-- Theorem statement
theorem perfect_square_condition (n : ℤ) (h1 : 0 < n) (h2 : is_integer (2 + 2 * Real.sqrt (1 + 12 * (n: ℝ)^2))) : 
  is_perfect_square n :=
by
  sorry

end perfect_square_condition_l2365_236590


namespace jake_pure_alcohol_l2365_236510

-- Definitions based on the conditions
def shots : ℕ := 8
def ounces_per_shot : ℝ := 1.5
def vodka_purity : ℝ := 0.5
def friends : ℕ := 2

-- Statement to prove the amount of pure alcohol Jake drank
theorem jake_pure_alcohol : (shots * ounces_per_shot * vodka_purity) / friends = 3 := by
  sorry

end jake_pure_alcohol_l2365_236510


namespace min_value_of_x_plus_y_l2365_236578

theorem min_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y)  
  (h : 19 / x + 98 / y = 1) : x + y ≥ 203 :=
sorry

end min_value_of_x_plus_y_l2365_236578


namespace shaded_fraction_is_one_fourth_l2365_236567

def quilt_block_shaded_fraction : ℚ :=
  let total_unit_squares := 16
  let triangles_per_unit_square := 2
  let shaded_triangles := 8
  let shaded_unit_squares := shaded_triangles / triangles_per_unit_square
  shaded_unit_squares / total_unit_squares

theorem shaded_fraction_is_one_fourth :
  quilt_block_shaded_fraction = 1 / 4 :=
sorry

end shaded_fraction_is_one_fourth_l2365_236567


namespace perfectCubesCount_l2365_236519

theorem perfectCubesCount (a b : Nat) (h₁ : 50 < a ∧ a ^ 3 > 50) (h₂ : b ^ 3 < 2000 ∧ b < 2000) :
  let n := b - a + 1
  n = 9 := by
  sorry

end perfectCubesCount_l2365_236519


namespace minimum_value_x_plus_3y_plus_6z_l2365_236535

theorem minimum_value_x_plus_3y_plus_6z 
  (x y z : ℝ) 
  (h1 : 0 < x) 
  (h2 : 0 < y) 
  (h3 : 0 < z) 
  (h4 : x * y * z = 18) : 
  x + 3 * y + 6 * z ≥ 3 * (2 * Real.sqrt 6 + 1) :=
sorry

end minimum_value_x_plus_3y_plus_6z_l2365_236535


namespace triangle_ABC_is_right_triangle_l2365_236595

theorem triangle_ABC_is_right_triangle (A B C : ℝ) (hA : A = 68) (hB : B = 22) :
  A + B + C = 180 → C = 90 :=
by
  intro hABC
  sorry

end triangle_ABC_is_right_triangle_l2365_236595


namespace calculate_difference_l2365_236501

def g (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 5

theorem calculate_difference (x h : ℝ) : g (x + h) - g x = h * (6 * x + 3 * h + 4) :=
by
  sorry

end calculate_difference_l2365_236501


namespace polynomial_evaluation_l2365_236550

theorem polynomial_evaluation :
  101^4 - 4 * 101^3 + 6 * 101^2 - 4 * 101 + 1 = 100000000 := sorry

end polynomial_evaluation_l2365_236550


namespace liters_per_bottle_l2365_236583

-- Condition statements
def price_per_liter : ℕ := 1
def total_cost : ℕ := 12
def num_bottles : ℕ := 6

-- Desired result statement
theorem liters_per_bottle : (total_cost / price_per_liter) / num_bottles = 2 := by
  sorry

end liters_per_bottle_l2365_236583


namespace martha_black_butterflies_l2365_236515

theorem martha_black_butterflies (total_butterflies blue_butterflies yellow_butterflies : ℕ)
  (h1 : total_butterflies = 11)
  (h2 : blue_butterflies = 4)
  (h3 : blue_butterflies = 2 * yellow_butterflies) :
  ∃ black_butterflies : ℕ, black_butterflies = total_butterflies - blue_butterflies - yellow_butterflies :=
sorry

end martha_black_butterflies_l2365_236515


namespace christina_has_three_snakes_l2365_236518

def snake_lengths : List ℕ := [24, 16, 10]

def total_length : ℕ := 50

theorem christina_has_three_snakes
  (lengths : List ℕ)
  (total : ℕ)
  (h_lengths : lengths = snake_lengths)
  (h_total : total = total_length)
  : lengths.length = 3 :=
by
  sorry

end christina_has_three_snakes_l2365_236518


namespace probability_blue_then_red_l2365_236539

/--
A box contains 15 balls, of which 5 are blue and 10 are red.
Two balls are drawn sequentially from the box without returning the first ball to the box.
Prove that the probability that the first ball drawn is blue and the second ball is red is 5 / 21.
-/
theorem probability_blue_then_red :
  let total_balls := 15
  let blue_balls := 5
  let red_balls := 10
  let first_is_blue := (blue_balls : ℚ) / total_balls
  let second_is_red_given_blue := (red_balls : ℚ) / (total_balls - 1)
  first_is_blue * second_is_red_given_blue = 5 / 21 := by
  sorry

end probability_blue_then_red_l2365_236539


namespace vector_calculation_l2365_236556

def v1 : ℝ × ℝ := (3, -5)
def v2 : ℝ × ℝ := (-1, 6)
def v3 : ℝ × ℝ := (2, -1)

theorem vector_calculation :
  (5:ℝ) • v1 - (3:ℝ) • v2 + v3 = (20, -44) :=
by
  sorry

end vector_calculation_l2365_236556


namespace distance_problem_l2365_236537

theorem distance_problem (x y n : ℝ) (h1 : y = 15) (h2 : Real.sqrt ((x - 2) ^ 2 + (15 - 7) ^ 2) = 13) (h3 : x > 2) :
  n = Real.sqrt ((2 + Real.sqrt 105) ^ 2 + 15 ^ 2) := by
  sorry

end distance_problem_l2365_236537


namespace distance_between_cars_after_third_checkpoint_l2365_236546

theorem distance_between_cars_after_third_checkpoint
  (initial_distance : ℝ)
  (initial_speed : ℝ)
  (speed_after_first : ℝ)
  (speed_after_second : ℝ)
  (speed_after_third : ℝ)
  (distance_travelled : ℝ) :
  initial_distance = 100 →
  initial_speed = 60 →
  speed_after_first = 80 →
  speed_after_second = 100 →
  speed_after_third = 120 →
  distance_travelled = 200 :=
by
  sorry

end distance_between_cars_after_third_checkpoint_l2365_236546


namespace necessarily_positive_l2365_236506

-- Definitions based on given conditions
variables {x y z : ℝ}

-- Stating the problem
theorem necessarily_positive : (0 < x ∧ x < 1) → (-2 < y ∧ y < 0) → (0 < z ∧ z < 1) → (x + y^2 > 0) :=
by
  intros hx hy hz
  sorry

end necessarily_positive_l2365_236506


namespace tangent_line_at_1_1_is_5x_plus_y_minus_6_l2365_236588

noncomputable def f : ℝ → ℝ :=
  λ x => x^3 - 4*x^2 + 4

def tangent_line_equation (x₀ y₀ m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => y - y₀ = m * (x - x₀)

theorem tangent_line_at_1_1_is_5x_plus_y_minus_6 : 
  tangent_line_equation 1 1 (-5) = (λ x y => 5 * x + y - 6 = 0) := 
by
  sorry

end tangent_line_at_1_1_is_5x_plus_y_minus_6_l2365_236588


namespace tuesday_snow_correct_l2365_236533

-- Define the snowfall amounts as given in the conditions
def monday_snow : ℝ := 0.32
def total_snow : ℝ := 0.53

-- Define the amount of snow on Tuesday as per the question to be proved
def tuesday_snow : ℝ := total_snow - monday_snow

-- State the theorem to prove that the snowfall on Tuesday is 0.21 inches
theorem tuesday_snow_correct : tuesday_snow = 0.21 := by
  -- Proof skipped with sorry
  sorry

end tuesday_snow_correct_l2365_236533


namespace hours_worked_l2365_236526

theorem hours_worked (w e : ℝ) (hw : w = 6.75) (he : e = 67.5) 
  : e / w = 10 := by
  sorry

end hours_worked_l2365_236526


namespace unique_angles_sum_l2365_236509

theorem unique_angles_sum (a1 a2 a3 a4 e4 e5 e6 e7 : ℝ) 
  (h_abcd: a1 + a2 + a3 + a4 = 360) 
  (h_efgh: e4 + e5 + e6 + e7 = 360) 
  (h_shared: a4 = e4) : 
  a1 + a2 + a3 + e4 + e5 + e6 + e7 - a4 = 360 := 
by 
  sorry

end unique_angles_sum_l2365_236509


namespace a_2019_value_l2365_236512

noncomputable def a_sequence (n : ℕ) : ℝ :=
  if n = 0 then 0  -- not used, a_0 is irrelevant
  else if n = 1 then 1 / 2
  else a_sequence (n - 1) + 1 / (2 ^ (n - 1))

theorem a_2019_value :
  a_sequence 2019 = 3 / 2 - 1 / (2 ^ 2018) :=
by
  sorry

end a_2019_value_l2365_236512


namespace simplify_expression_l2365_236572

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  ((x + y) ^ 2 - (x - y) ^ 2) / (4 * x * y) = 1 := 
by sorry

end simplify_expression_l2365_236572


namespace quadratic_min_value_l2365_236523

theorem quadratic_min_value (f : ℕ → ℚ) (n : ℕ)
  (h₁ : f n = 6)
  (h₂ : f (n + 1) = 5)
  (h₃ : f (n + 2) = 5) :
  ∃ c : ℚ, c = 39 / 8 ∧ ∀ x : ℕ, f x ≥ c :=
by
  sorry

end quadratic_min_value_l2365_236523


namespace anya_age_l2365_236527

theorem anya_age (n : ℕ) (h : 110 ≤ (n * (n + 1)) / 2 ∧ (n * (n + 1)) / 2 ≤ 130) : n = 15 :=
sorry

end anya_age_l2365_236527


namespace num_children_got_off_l2365_236584

-- Define the original number of children on the bus
def original_children : ℕ := 43

-- Define the number of children left after some got off the bus
def children_left : ℕ := 21

-- Define the number of children who got off the bus as the difference between original_children and children_left
def children_got_off : ℕ := original_children - children_left

-- State the theorem that the number of children who got off the bus is 22
theorem num_children_got_off : children_got_off = 22 :=
by
  -- Proof steps would go here, but are omitted
  sorry

end num_children_got_off_l2365_236584


namespace person_income_l2365_236565

theorem person_income 
    (income expenditure savings : ℕ) 
    (h1 : income = 3 * (income / 3)) 
    (h2 : expenditure = 2 * (income / 3)) 
    (h3 : savings = 7000) 
    (h4 : income = expenditure + savings) : 
    income = 21000 := 
by 
  sorry

end person_income_l2365_236565


namespace max_net_income_is_50000_l2365_236575

def tax_rate (y : ℝ) : ℝ :=
  10 * y ^ 2

def net_income (y : ℝ) : ℝ :=
  1000 * y - tax_rate y

theorem max_net_income_is_50000 :
  ∃ y : ℝ, (net_income y = 25000 ∧ 1000 * y = 50000) :=
by
  use 50
  sorry

end max_net_income_is_50000_l2365_236575


namespace solve_system_equations_l2365_236574

theorem solve_system_equations (x y : ℝ) :
  x + y = 0 ∧ 2 * x + 3 * y = 3 → x = -3 ∧ y = 3 :=
by {
  -- Leave the proof as a placeholder with "sorry".
  sorry
}

end solve_system_equations_l2365_236574


namespace volume_of_polyhedron_l2365_236589

open Real

-- Define the conditions
def square_side : ℝ := 100  -- in cm, equivalent to 1 meter
def rectangle_length : ℝ := 40  -- in cm
def rectangle_width : ℝ := 20  -- in cm
def trapezoid_leg_length : ℝ := 130  -- in cm

-- Define the question as a theorem statement
theorem volume_of_polyhedron :
  ∃ V : ℝ, V = 552 :=
sorry

end volume_of_polyhedron_l2365_236589


namespace skyler_total_songs_skyler_success_breakdown_l2365_236530

noncomputable def skyler_songs : ℕ :=
  let hit_songs := 25
  let top_100_songs := hit_songs + 10
  let unreleased_songs := hit_songs - 5
  let duets_total := 12
  let duets_top_20 := duets_total / 2
  let duets_not_top_200 := duets_total / 2
  let soundtracks_total := 18
  let soundtracks_extremely := 3
  let soundtracks_moderate := 8
  let soundtracks_lukewarm := 7
  let projects_total := 22
  let projects_global := 1
  let projects_regional := 7
  let projects_overlooked := 14
  hit_songs + top_100_songs + unreleased_songs + duets_total + soundtracks_total + projects_total

theorem skyler_total_songs : skyler_songs = 132 := by
  sorry

theorem skyler_success_breakdown :
  let extremely_successful := 25 + 1
  let successful := 35 + 6 + 3
  let moderately_successful := 8 + 7
  let less_successful := 7 + 14 + 6
  let unreleased := 20
  (extremely_successful, successful, moderately_successful, less_successful, unreleased) =
  (26, 44, 15, 27, 20) := by
  sorry

end skyler_total_songs_skyler_success_breakdown_l2365_236530


namespace find_wrong_observation_value_l2365_236562

-- Defining the given conditions
def original_mean : ℝ := 36
def corrected_mean : ℝ := 36.5
def num_observations : ℕ := 50
def correct_value : ℝ := 30

-- Defining the given sums based on means
def original_sum : ℝ := num_observations * original_mean
def corrected_sum : ℝ := num_observations * corrected_mean

-- The wrong value can be calculated based on the difference
def wrong_value : ℝ := correct_value + (corrected_sum - original_sum)

-- The theorem to prove
theorem find_wrong_observation_value (h : original_sum = 1800) (h' : corrected_sum = 1825) :
  wrong_value = 55 :=
sorry

end find_wrong_observation_value_l2365_236562


namespace part_a_l2365_236500

theorem part_a (x : ℝ) : (6 - x) / x = 3 / 6 → x = 4 := by
  sorry

end part_a_l2365_236500


namespace alex_buys_17_1_pounds_of_corn_l2365_236543

-- Definitions based on conditions
def corn_cost_per_pound : ℝ := 1.20
def bean_cost_per_pound : ℝ := 0.50
def total_pounds : ℝ := 30
def total_cost : ℝ := 27.00

-- Define the variables
variables (c b : ℝ)

-- Theorem statement to prove the number of pounds of corn Alex buys
theorem alex_buys_17_1_pounds_of_corn (h1 : b + c = total_pounds) (h2 : bean_cost_per_pound * b + corn_cost_per_pound * c = total_cost) :
  c = 17.1 :=
sorry

end alex_buys_17_1_pounds_of_corn_l2365_236543


namespace weight_of_a_is_75_l2365_236507

theorem weight_of_a_is_75 (a b c d e : ℕ) 
  (h1 : (a + b + c) / 3 = 84) 
  (h2 : (a + b + c + d) / 4 = 80) 
  (h3 : e = d + 3) 
  (h4 : (b + c + d + e) / 4 = 79) : 
  a = 75 :=
by
  -- Proof omitted
  sorry

end weight_of_a_is_75_l2365_236507


namespace total_cards_square_l2365_236552

theorem total_cards_square (s : ℕ) (h_perim : 4 * s - 4 = 240) : s * s = 3721 := by
  sorry

end total_cards_square_l2365_236552


namespace equal_intercepts_condition_l2365_236587

theorem equal_intercepts_condition (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) :
  (a = b ∨ c = 0) ↔ (c = 0 ∨ (c ≠ 0 ∧ a = b)) :=
by sorry

end equal_intercepts_condition_l2365_236587


namespace polygon_sides_l2365_236573

theorem polygon_sides (n : ℕ) 
  (h1 : (n-2) * 180 = 3 * 360) 
  (h2 : n > 2) : n = 8 :=
sorry

end polygon_sides_l2365_236573


namespace mandy_pieces_eq_fifteen_l2365_236585

-- Define the initial chocolate pieces
def total_pieces := 60

-- Define Michael's share
def michael_share := total_pieces / 2

-- Define the remainder after Michael's share
def remainder_after_michael := total_pieces - michael_share

-- Define Paige's share
def paige_share := remainder_after_michael / 2

-- Define the remainder after Paige's share
def mandy_share := remainder_after_michael - paige_share

-- Theorem to prove Mandy gets 15 pieces
theorem mandy_pieces_eq_fifteen : mandy_share = 15 :=
by
  sorry

end mandy_pieces_eq_fifteen_l2365_236585


namespace fraction_product_l2365_236597

theorem fraction_product : 
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_l2365_236597


namespace problem_statement_l2365_236503

-- Given conditions
variable (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k)

-- Hypothesis configuration for inductive proof and goal statement
theorem problem_statement : ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
by sorry

end problem_statement_l2365_236503


namespace least_distance_between_ticks_l2365_236522

theorem least_distance_between_ticks (x : ℚ) :
  (∀ n : ℕ, ∃ k : ℤ, k = n * 11 ∨ k = n * 13) →
  x = 1 / 143 :=
by
  sorry

end least_distance_between_ticks_l2365_236522


namespace simplify_fraction_expression_l2365_236540

theorem simplify_fraction_expression : 
  (18 / 42 - 3 / 8 - 1 / 12 : ℚ) = -5 / 168 :=
by
  sorry

end simplify_fraction_expression_l2365_236540


namespace max_value_trig_expr_exists_angle_for_max_value_l2365_236508

theorem max_value_trig_expr : ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5 :=
sorry

theorem exists_angle_for_max_value : ∃ x : ℝ, 3 * Real.cos x + 4 * Real.sin x = 5 :=
sorry

end max_value_trig_expr_exists_angle_for_max_value_l2365_236508


namespace find_angle_four_l2365_236577

theorem find_angle_four (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle1 + angle2 = 180)
  (h2 : angle1 + angle3 + 60 = 180)
  (h3 : angle3 = angle4) :
  angle4 = 60 :=
by sorry

end find_angle_four_l2365_236577


namespace no_integral_points_on_AB_l2365_236561

theorem no_integral_points_on_AB (k m n : ℤ) (h1: ((m^3 - m)^2 + (n^3 - n)^2 > (3*k + 1)^2)) :
  ¬ ∃ (x y : ℤ), (m^3 - m) * x + (n^3 - n) * y = (3*k + 1)^2 :=
by {
  sorry
}

end no_integral_points_on_AB_l2365_236561


namespace circumscribe_quadrilateral_a_circumscribe_quadrilateral_b_l2365_236528

theorem circumscribe_quadrilateral_a : 
  ∃ (x : ℝ), 2 * x + 4 * x + 5 * x + 3 * x = 360 
          ∧ (2 * x + 5 * x = 180) 
          ∧ (4 * x + 3 * x = 180) := sorry

theorem circumscribe_quadrilateral_b : 
  ∃ (x : ℝ), 5 * x + 7 * x + 8 * x + 9 * x = 360 
          ∧ (5 * x + 8 * x ≠ 180) 
          ∧ (7 * x + 9 * x ≠ 180) := sorry

end circumscribe_quadrilateral_a_circumscribe_quadrilateral_b_l2365_236528


namespace arithmetic_sequence_20th_term_l2365_236564

-- Definitions for the first term and common difference
def first_term : ℤ := 8
def common_difference : ℤ := -3

-- Define the general term for an arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℤ := first_term + (n - 1) * common_difference

-- The specific property we seek to prove: the 20th term is -49
theorem arithmetic_sequence_20th_term : arithmetic_sequence 20 = -49 := by
  -- Proof is omitted, filled with sorry
  sorry

end arithmetic_sequence_20th_term_l2365_236564


namespace watermelon_sales_correct_l2365_236580

def total_watermelons_sold 
  (customers_one_melon : ℕ) 
  (customers_three_melons : ℕ) 
  (customers_two_melons : ℕ) : ℕ :=
  (customers_one_melon * 1) + (customers_three_melons * 3) + (customers_two_melons * 2)

theorem watermelon_sales_correct :
  total_watermelons_sold 17 3 10 = 46 := by
  sorry

end watermelon_sales_correct_l2365_236580


namespace chess_program_ratio_l2365_236596

theorem chess_program_ratio {total_students chess_program_absent : ℕ}
  (h_total : total_students = 24)
  (h_absent : chess_program_absent = 4)
  (h_half : chess_program_absent * 2 = chess_program_absent + chess_program_absent) :
  (chess_program_absent * 2 : ℚ) / total_students = 1 / 3 :=
by
  sorry

end chess_program_ratio_l2365_236596


namespace range_a_ineq_value_of_a_plus_b_l2365_236557

open Real

def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 3)
def g (a x : ℝ) : ℝ := a - abs (x - 2)

noncomputable def range_a (a : ℝ) : Prop :=
  ∃ x : ℝ, f x < g a x

theorem range_a_ineq (a : ℝ) : range_a a ↔ 4 < a := sorry

def solution_set (b : ℝ) : Prop :=
  ∀ x : ℝ, f x < g ((13/2) : ℝ) x ↔ (b < x ∧ x < 7/2)

theorem value_of_a_plus_b (b : ℝ) (h : solution_set b) : (13/2) + b = 6 := sorry

end range_a_ineq_value_of_a_plus_b_l2365_236557


namespace intersection_of_sets_l2365_236598

def setA : Set ℝ := { x | x^2 - 4*x + 3 < 0 }
def setB : Set ℝ := { x | 2*x - 3 > 0 }

theorem intersection_of_sets : setA ∩ setB = { x : ℝ | x > 3/2 ∧ x < 3 } :=
  by sorry

end intersection_of_sets_l2365_236598


namespace alpha_beta_power_eq_sum_power_for_large_p_l2365_236582

theorem alpha_beta_power_eq_sum_power_for_large_p (α β : ℂ) (p : ℕ) (hp : p ≥ 5)
  (hαβ : ∀ x : ℂ, 2 * x^4 - 6 * x^3 + 11 * x^2 - 6 * x - 4 = 0 → x = α ∨ x = β) :
  α^p + β^p = (α + β)^p :=
sorry

end alpha_beta_power_eq_sum_power_for_large_p_l2365_236582


namespace janet_initial_action_figures_l2365_236549

theorem janet_initial_action_figures (x : ℕ) :
  (x - 2 + 2 * (x - 2) = 24) -> x = 10 := 
by
  sorry

end janet_initial_action_figures_l2365_236549


namespace twenty_five_question_test_l2365_236545

def not_possible_score (score total_questions correct_points unanswered_points incorrect_points : ℕ) : Prop :=
  ∀ correct unanswered incorrect : ℕ,
    correct + unanswered + incorrect = total_questions →
    correct * correct_points + unanswered * unanswered_points + incorrect * incorrect_points ≠ score

theorem twenty_five_question_test :
  not_possible_score 96 25 4 2 0 :=
by
  sorry

end twenty_five_question_test_l2365_236545


namespace rajeev_share_of_profit_l2365_236571

open Nat

theorem rajeev_share_of_profit (profit : ℕ) (ramesh_xyz_ratio1 ramesh_xyz_ratio2 xyz_rajeev_ratio1 xyz_rajeev_ratio2 : ℕ) (rajeev_ratio_part : ℕ) (total_parts : ℕ) (individual_part_value : ℕ) :
  profit = 36000 →
  ramesh_xyz_ratio1 = 5 →
  ramesh_xyz_ratio2 = 4 →
  xyz_rajeev_ratio1 = 8 →
  xyz_rajeev_ratio2 = 9 →
  rajeev_ratio_part = 9 →
  total_parts = ramesh_xyz_ratio1 * (xyz_rajeev_ratio1 / ramesh_xyz_ratio2) + xyz_rajeev_ratio1 + xyz_rajeev_ratio2 →
  individual_part_value = profit / total_parts →
  rajeev_ratio_part * individual_part_value = 12000 := 
sorry

end rajeev_share_of_profit_l2365_236571


namespace lloyd_hourly_rate_l2365_236544

variable (R : ℝ)  -- Lloyd's regular hourly rate

-- Conditions
def lloyd_works_regular_hours_per_day : Prop := R > 0
def lloyd_earns_excess_rate : Prop := 1.5 * R > 0
def lloyd_worked_hours : Prop := 10.5 > 7.5
def lloyd_earned_amount : Prop := 7.5 * R + 3 * 1.5 * R = 66

-- Theorem statement
theorem lloyd_hourly_rate (hr_pos : lloyd_works_regular_hours_per_day R)
                           (excess_rate : lloyd_earns_excess_rate R)
                           (worked_hours : lloyd_worked_hours)
                           (earned_amount : lloyd_earned_amount R) : 
    R = 5.5 :=
by sorry

end lloyd_hourly_rate_l2365_236544


namespace largest_possible_A_l2365_236593

theorem largest_possible_A (A B C : ℕ) (h1 : 10 = A * B + C) (h2 : B = C) : A ≤ 9 :=
by sorry

end largest_possible_A_l2365_236593


namespace f_is_odd_l2365_236548

def f (x : ℝ) : ℝ := |x + 1| - |x - 1|

theorem f_is_odd : ∀ x, f (-x) = -f x := by
  sorry

end f_is_odd_l2365_236548


namespace correct_average_wrong_reading_l2365_236581

theorem correct_average_wrong_reading
  (initial_average : ℕ) (list_length : ℕ) (wrong_number : ℕ) (correct_number : ℕ) (correct_average : ℕ) 
  (h1 : initial_average = 18)
  (h2 : list_length = 10)
  (h3 : wrong_number = 26)
  (h4 : correct_number = 66)
  (h5 : correct_average = 22) :
  correct_average = ((initial_average * list_length) - wrong_number + correct_number) / list_length :=
sorry

end correct_average_wrong_reading_l2365_236581


namespace solve_system_of_equations_l2365_236521

theorem solve_system_of_equations (x y : ℝ) :
  (3 * x^2 + 4 * x * y + 12 * y^2 + 16 * y = -6) ∧
  (x^2 - 12 * x * y + 4 * y^2 - 10 * x + 12 * y = -7) →
  (x = 1 / 2) ∧ (y = -3 / 4) :=
by
  sorry

end solve_system_of_equations_l2365_236521


namespace roses_problem_l2365_236551

variable (R B C : ℕ)

theorem roses_problem
    (h1 : R = B + 10)
    (h2 : C = 10)
    (h3 : 16 - 6 = C)
    (h4 : B = R - C):
  R = B + 10 ∧ R - C = B := 
by 
  have hC: C = 10 := by linarith
  have hR: R = B + 10 := by linarith
  have hRC: R - C = B := by linarith
  exact ⟨hR, hRC⟩

end roses_problem_l2365_236551


namespace selection_methods_l2365_236502

theorem selection_methods (females males : Nat) (h_females : females = 3) (h_males : males = 2):
  females + males = 5 := 
  by 
    -- We add sorry here to skip the proof
    sorry

end selection_methods_l2365_236502


namespace fifth_term_arithmetic_sequence_l2365_236579

theorem fifth_term_arithmetic_sequence (a d : ℤ) 
  (h_twentieth : a + 19 * d = 12) 
  (h_twenty_first : a + 20 * d = 16) : 
  a + 4 * d = -48 := 
by sorry

end fifth_term_arithmetic_sequence_l2365_236579


namespace correct_statement_l2365_236558

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + (Real.pi / 2))
noncomputable def g (x : ℝ) : ℝ := Real.cos (x + (3 * Real.pi / 2))

theorem correct_statement (x : ℝ) : f (x - (Real.pi / 2)) = g x :=
by sorry

end correct_statement_l2365_236558


namespace solve_for_y_l2365_236536

theorem solve_for_y (y : ℕ) (h : 5 * (2 ^ y) = 320) : y = 6 := 
by 
  sorry

end solve_for_y_l2365_236536


namespace students_move_bricks_l2365_236594

variable (a b c : ℕ)

theorem students_move_bricks (h : a * b * c ≠ 0) : 
  (by let efficiency := (c : ℚ) / (a * b);
      let total_work := (a : ℚ);
      let required_time := total_work / efficiency;
      exact required_time = (a^2 * b) / (c^2)) := sorry

end students_move_bricks_l2365_236594


namespace problem_1_problem_2_l2365_236524

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x - 3|

theorem problem_1 (a x : ℝ) (h1 : a < 3) (h2 : (∀ x, f x a >= 4 ↔ x ≤ 1/2 ∨ x ≥ 9/2)) : 
  a = 2 :=
sorry

theorem problem_2 (a : ℝ) (h1 : ∀ x : ℝ, f x a + |x - 3| ≥ 1) : 
  a ≤ 2 :=
sorry

end problem_1_problem_2_l2365_236524


namespace range_of_a_l2365_236547

variable (a : ℝ)

-- Definitions of propositions p and q
def p := ∀ x : ℝ, x^2 - 2*x - a ≥ 0
def q := ∃ x : ℝ, x^2 + x + 2*a - 1 ≤ 0

-- Lean 4 statement of the proof problem
theorem range_of_a : ¬ p a ∧ q a → -1 < a ∧ a ≤ 5/8 := by
  sorry

end range_of_a_l2365_236547


namespace problem_b_c_constants_l2365_236534

theorem problem_b_c_constants (b c : ℝ) (h : ∀ x : ℝ, (x + 2) * (x + b) = x^2 + c * x + 6) : c = 5 := 
by sorry

end problem_b_c_constants_l2365_236534


namespace decimal_to_base9_l2365_236591

theorem decimal_to_base9 (n : ℕ) (h : n = 1729) : 
  (2 * 9^3 + 3 * 9^2 + 3 * 9^1 + 1 * 9^0) = n :=
by sorry

end decimal_to_base9_l2365_236591


namespace smallest_number_of_students_l2365_236541

theorem smallest_number_of_students 
  (n : ℕ) 
  (h1 : 4 * 80 + (n - 4) * 50 ≤ 65 * n) :
  n = 8 :=
by sorry

end smallest_number_of_students_l2365_236541


namespace q_investment_time_l2365_236513

-- Definitions from the conditions
def investment_ratio_p_q : ℚ := 7 / 5
def profit_ratio_p_q : ℚ := 7 / 13
def time_p : ℕ := 5

-- Problem statement
theorem q_investment_time
  (investment_ratio_p_q : ℚ)
  (profit_ratio_p_q : ℚ)
  (time_p : ℕ)
  (hpq_inv : investment_ratio_p_q = 7 / 5)
  (hpq_profit : profit_ratio_p_q = 7 / 13)
  (ht_p : time_p = 5) : 
  ∃ t_q : ℕ, 35 * t_q = 455 :=
sorry

end q_investment_time_l2365_236513


namespace find_m_range_a_l2365_236553

noncomputable def f (x m : ℝ) : ℝ :=
  m - |x - 3|

theorem find_m (m : ℝ) (h : ∀ x, 2 < f x m ↔ 2 < x ∧ x < 4) : m = 3 :=
  sorry

theorem range_a (a : ℝ) (h : ∀ x, |x - a| ≥ f x 3) : a ≤ 0 ∨ 6 ≤ a :=
  sorry

end find_m_range_a_l2365_236553


namespace price_of_pants_l2365_236570

theorem price_of_pants (P : ℝ) (h1 : 4 * 33 = 132) (h2 : 2 * P + 132 = 240) : P = 54 :=
sorry

end price_of_pants_l2365_236570


namespace base6_sub_base9_to_base10_l2365_236554

theorem base6_sub_base9_to_base10 :
  (3 * 6^2 + 2 * 6^1 + 5 * 6^0) - (2 * 9^2 + 1 * 9^1 + 5 * 9^0) = -51 :=
by
  sorry

end base6_sub_base9_to_base10_l2365_236554


namespace degree_measure_of_supplement_of_complement_of_35_degree_angle_l2365_236569

def complement (α : ℝ) : ℝ := 90 - α
def supplement (β : ℝ) : ℝ := 180 - β

theorem degree_measure_of_supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 :=
by
  sorry

end degree_measure_of_supplement_of_complement_of_35_degree_angle_l2365_236569


namespace tom_dimes_count_l2365_236525

def originalDimes := 15
def dimesFromDad := 33
def dimesSpent := 11

theorem tom_dimes_count : originalDimes + dimesFromDad - dimesSpent = 37 := by
  sorry

end tom_dimes_count_l2365_236525


namespace perfect_number_divisibility_l2365_236576

theorem perfect_number_divisibility (P : ℕ) (h1 : P > 28) (h2 : Nat.Perfect P) (h3 : 7 ∣ P) : 49 ∣ P := 
sorry

end perfect_number_divisibility_l2365_236576


namespace calculate_expression_l2365_236568

theorem calculate_expression : 1^345 + 5^10 / 5^7 = 126 := by
  sorry

end calculate_expression_l2365_236568


namespace gibbs_inequality_l2365_236514

noncomputable section

open BigOperators

variable {r : ℕ} (p q : Fin r → ℝ)

/-- (p_i) is a probability distribution -/
def isProbabilityDistribution (p : Fin r → ℝ) : Prop :=
  (∀ i, 0 ≤ p i) ∧ (∑ i, p i = 1)

/-- -\sum_{i=1}^{r} p_i \ln p_i \leqslant -\sum_{i=1}^{r} p_i \ln q_i for probability distributions p and q -/
theorem gibbs_inequality
  (hp : isProbabilityDistribution p)
  (hq : isProbabilityDistribution q) :
  -∑ i, p i * Real.log (p i) ≤ -∑ i, p i * Real.log (q i) := 
by
  sorry

end gibbs_inequality_l2365_236514


namespace determine_k_l2365_236511

noncomputable def k_value (k : ℤ) : Prop :=
  let m := (-2 - 2) / (3 - 1)
  let b := 2 - m * 1
  let y := m * 4 + b
  let point := (4, k / 3)
  point.2 = y

theorem determine_k :
  ∃ k : ℤ, k_value k ∧ k = -12 :=
by
  use -12
  sorry

end determine_k_l2365_236511


namespace largest_cube_edge_from_cone_l2365_236504

theorem largest_cube_edge_from_cone : 
  ∀ (s : ℝ), 
  (s = 2) → 
  ∃ (x : ℝ), x = 3 * Real.sqrt 2 - 2 * Real.sqrt 3 :=
by
  sorry

end largest_cube_edge_from_cone_l2365_236504


namespace marbles_problem_l2365_236599

theorem marbles_problem
  (cindy_original : ℕ)
  (lisa_original : ℕ)
  (h1 : cindy_original = 20)
  (h2 : cindy_original = lisa_original + 5)
  (marbles_given : ℕ)
  (h3 : marbles_given = 12) :
  (lisa_original + marbles_given) - (cindy_original - marbles_given) = 19 :=
by
  sorry

end marbles_problem_l2365_236599


namespace exists_constant_not_geometric_l2365_236529

-- Definitions for constant and geometric sequences
def is_constant_sequence (seq : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, seq n = c

def is_geometric_sequence (seq : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, seq (n + 1) = r * seq n

-- The negation problem statement
theorem exists_constant_not_geometric :
  ∃ seq : ℕ → ℝ, is_constant_sequence seq ∧ ¬is_geometric_sequence seq :=
sorry

end exists_constant_not_geometric_l2365_236529
