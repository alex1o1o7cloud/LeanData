import Mathlib

namespace NUMINAMATH_GPT_binary_add_mul_l718_71878

def x : ℕ := 0b101010
def y : ℕ := 0b11010
def z : ℕ := 0b1110
def result : ℕ := 0b11000000000

theorem binary_add_mul : ((x + y) * z) = result := by
  sorry

end NUMINAMATH_GPT_binary_add_mul_l718_71878


namespace NUMINAMATH_GPT_eval_expression_l718_71849

theorem eval_expression :
    (727 * 727) - (726 * 728) = 1 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l718_71849


namespace NUMINAMATH_GPT_inequality_proof_l718_71895

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l718_71895


namespace NUMINAMATH_GPT_probability_playing_one_instrument_l718_71890

noncomputable def total_people : ℕ := 800
noncomputable def fraction_playing_instruments : ℚ := 1 / 5
noncomputable def number_playing_two_or_more : ℕ := 32

theorem probability_playing_one_instrument :
  let number_playing_at_least_one := (fraction_playing_instruments * total_people)
  let number_playing_exactly_one := number_playing_at_least_one - number_playing_two_or_more
  (number_playing_exactly_one / total_people) = 1 / 6.25 :=
by 
  let number_playing_at_least_one := (fraction_playing_instruments * total_people)
  let number_playing_exactly_one := number_playing_at_least_one - number_playing_two_or_more
  have key : (number_playing_exactly_one / total_people) = 1 / 6.25 := sorry
  exact key

end NUMINAMATH_GPT_probability_playing_one_instrument_l718_71890


namespace NUMINAMATH_GPT_cat_birds_total_l718_71846

def day_birds : ℕ := 8
def night_birds : ℕ := 2 * day_birds
def total_birds : ℕ := day_birds + night_birds

theorem cat_birds_total : total_birds = 24 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_cat_birds_total_l718_71846


namespace NUMINAMATH_GPT_three_dice_prime_probability_l718_71887

noncomputable def rolling_three_dice_prime_probability : ℚ :=
  sorry

theorem three_dice_prime_probability : rolling_three_dice_prime_probability = 1 / 24 :=
  sorry

end NUMINAMATH_GPT_three_dice_prime_probability_l718_71887


namespace NUMINAMATH_GPT_sub_number_l718_71813

theorem sub_number : 600 - 333 = 267 := by
  sorry

end NUMINAMATH_GPT_sub_number_l718_71813


namespace NUMINAMATH_GPT_appropriate_grouping_43_neg78_27_neg52_l718_71858

theorem appropriate_grouping_43_neg78_27_neg52 :
  let a := 43
  let b := -78
  let c := 27
  let d := -52
  (a + c) + (b + d) = -60 :=
by
  let a := 43
  let b := -78
  let c := 27
  let d := -52
  sorry

end NUMINAMATH_GPT_appropriate_grouping_43_neg78_27_neg52_l718_71858


namespace NUMINAMATH_GPT_tabitha_color_start_l718_71834

def add_color_each_year (n : ℕ) : ℕ := n + 1

theorem tabitha_color_start 
  (age_start age_now future_colors years_future current_colors : ℕ)
  (h1 : age_start = 15)
  (h2 : age_now = 18)
  (h3 : years_future = 3)
  (h4 : age_now + years_future = 21)
  (h5 : future_colors = 8)
  (h6 : future_colors - years_future = current_colors + 3)
  (h7 : current_colors = 5)
  : age_start + (current_colors - (age_now - age_start)) = 3 := 
by
  sorry

end NUMINAMATH_GPT_tabitha_color_start_l718_71834


namespace NUMINAMATH_GPT_multiple_of_75_with_36_divisors_l718_71835

theorem multiple_of_75_with_36_divisors (n : ℕ) (h1 : n % 75 = 0) (h2 : ∃ (a b c : ℕ), a ≥ 1 ∧ b ≥ 2 ∧ n = 3^a * 5^b * (2^c) ∧ (a+1)*(b+1)*(c+1) = 36) : n / 75 = 24 := 
sorry

end NUMINAMATH_GPT_multiple_of_75_with_36_divisors_l718_71835


namespace NUMINAMATH_GPT_extreme_value_sum_l718_71880

noncomputable def f (m n x : ℝ) : ℝ := x^3 + 3 * m * x^2 + n * x + m^2

theorem extreme_value_sum (m n : ℝ) (h1 : f m n (-1) = 0) (h2 : (deriv (f m n)) (-1) = 0) : m + n = 11 := 
sorry

end NUMINAMATH_GPT_extreme_value_sum_l718_71880


namespace NUMINAMATH_GPT_power_of_seven_l718_71882

theorem power_of_seven : 
  (7 : ℝ) ^ (1 / 4) / (7 ^ (1 / 7)) = (7 ^ (3 / 28)) :=
by
  sorry

end NUMINAMATH_GPT_power_of_seven_l718_71882


namespace NUMINAMATH_GPT_pq_eq_real_nums_l718_71894

theorem pq_eq_real_nums (p q r x y z : ℝ) 
  (h1 : x / p + q / y = 1) 
  (h2 : y / q + r / z = 1) : 
  p * q * r + x * y * z = 0 := 
by 
  sorry

end NUMINAMATH_GPT_pq_eq_real_nums_l718_71894


namespace NUMINAMATH_GPT_three_more_than_seven_in_pages_l718_71899

theorem three_more_than_seven_in_pages : 
  ∀ (pages : List Nat), (∀ n, n ∈ pages → 1 ≤ n ∧ n ≤ 530) ∧ (List.length pages = 530) →
  ((List.count 3 (pages.bind (λ n => Nat.digits 10 n))) - (List.count 7 (pages.bind (λ n => Nat.digits 10 n)))) = 100 :=
by
  intros pages h
  sorry

end NUMINAMATH_GPT_three_more_than_seven_in_pages_l718_71899


namespace NUMINAMATH_GPT_daniel_age_l718_71804

def isAgeSet (s : Set ℕ) : Prop :=
  s = {4, 6, 8, 10, 12, 14}

def sumTo18 (s : Set ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ a + b = 18 ∧ a ≠ b

def youngerThan11 (s : Set ℕ) : Prop :=
  ∀ (a : ℕ), a ∈ s → a < 11

def staysHome (DanielAge : ℕ) (s : Set ℕ) : Prop :=
  6 ∈ s ∧ DanielAge ∈ s

theorem daniel_age :
  ∀ (ages : Set ℕ) (DanielAge : ℕ),
    isAgeSet ages →
    (∃ s, sumTo18 s ∧ s ⊆ ages) →
    (∃ s, youngerThan11 s ∧ s ⊆ ages ∧ 6 ∉ s) →
    staysHome DanielAge ages →
    DanielAge = 12 :=
by
  intros ages DanielAge isAgeSetAges sumTo18Ages youngerThan11Ages staysHomeDaniel
  sorry

end NUMINAMATH_GPT_daniel_age_l718_71804


namespace NUMINAMATH_GPT_quinton_total_fruit_trees_l718_71896

-- Define the given conditions
def num_apple_trees := 2
def width_apple_tree_ft := 10
def space_between_apples_ft := 12
def width_peach_tree_ft := 12
def space_between_peaches_ft := 15
def total_space_ft := 71

-- Definition that calculates the total number of fruit trees Quinton wants to plant
def total_fruit_trees : ℕ := 
  let space_apple_trees := num_apple_trees * width_apple_tree_ft + space_between_apples_ft
  let space_remaining_for_peaches := total_space_ft - space_apple_trees
  1 + space_remaining_for_peaches / (width_peach_tree_ft + space_between_peaches_ft) + num_apple_trees

-- The statement to prove
theorem quinton_total_fruit_trees : total_fruit_trees = 4 := by
  sorry

end NUMINAMATH_GPT_quinton_total_fruit_trees_l718_71896


namespace NUMINAMATH_GPT_tan_alpha_beta_l718_71841

noncomputable def tan_alpha := -1 / 3
noncomputable def cos_beta := (Real.sqrt 5) / 5
noncomputable def beta := (1:ℝ) -- Dummy representation for being in first quadrant

theorem tan_alpha_beta (h1 : tan_alpha = -1 / 3) 
                       (h2 : cos_beta = (Real.sqrt 5) / 5) 
                       (h3 : 0 < beta ∧ beta < Real.pi / 2) : 
  Real.tan (α + β) = 1 := 
sorry

end NUMINAMATH_GPT_tan_alpha_beta_l718_71841


namespace NUMINAMATH_GPT_proof_m_cd_value_l718_71840

theorem proof_m_cd_value (a b c d m : ℝ) 
  (H1 : a + b = 0) (H2 : c * d = 1) (H3 : |m| = 3) : 
  m + c * d - (a + b) / (m ^ 2) = 4 ∨ m + c * d - (a + b) / (m ^ 2) = -2 :=
by
  sorry

end NUMINAMATH_GPT_proof_m_cd_value_l718_71840


namespace NUMINAMATH_GPT_greatest_multiple_of_30_less_than_1000_l718_71837

theorem greatest_multiple_of_30_less_than_1000 : ∃ (n : ℕ), n < 1000 ∧ n % 30 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 30 = 0 → m ≤ n := 
by 
  use 990
  sorry

end NUMINAMATH_GPT_greatest_multiple_of_30_less_than_1000_l718_71837


namespace NUMINAMATH_GPT_pairs_with_green_shirts_l718_71873

theorem pairs_with_green_shirts (r g t p rr_pairs gg_pairs : ℕ)
  (h1 : r = 60)
  (h2 : g = 90)
  (h3 : t = 150)
  (h4 : p = 75)
  (h5 : rr_pairs = 28)
  : gg_pairs = 43 := 
sorry

end NUMINAMATH_GPT_pairs_with_green_shirts_l718_71873


namespace NUMINAMATH_GPT_problem_solution_l718_71838

noncomputable def find_a3_and_sum (a0 a1 a2 a3 a4 a5 : ℝ) : Prop :=
  (∀ x : ℝ, x^5 = a0 + a1 * (x + 2) + a2 * (x + 2)^2 + a3 * (x + 2)^3 + a4 * (x + 2)^4 + a5 * (x + 2)^5) →
  (a3 = 40 ∧ a0 + a1 + a2 + a4 + a5 = -41)

theorem problem_solution {a0 a1 a2 a3 a4 a5 : ℝ} :
  find_a3_and_sum a0 a1 a2 a3 a4 a5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_problem_solution_l718_71838


namespace NUMINAMATH_GPT_range_of_a_real_root_l718_71811

theorem range_of_a_real_root :
  (∀ x : ℝ, x^2 - a * x + 4 = 0 → ∃ x : ℝ, (x^2 - a * x + 4 = 0 ∧ (a ≥ 4 ∨ a ≤ -4))) ∨
  (∀ x : ℝ, x^2 + (a-2) * x + 4 = 0 → ∃ x : ℝ, (x^2 + (a-2) * x + 4 = 0 ∧ (a ≥ 6 ∨ a ≤ -2))) ∨
  (∀ x : ℝ, x^2 + 2 * a * x + a^2 + 1 = 0 → False) →
  (a ≥ 4 ∨ a ≤ -2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_real_root_l718_71811


namespace NUMINAMATH_GPT_determinant_range_l718_71889

theorem determinant_range (x : ℝ) : 
  (2 * x - (3 - x) > 0) ↔ (x > 1) :=
by
  sorry

end NUMINAMATH_GPT_determinant_range_l718_71889


namespace NUMINAMATH_GPT_A_more_than_B_l718_71892

variable (A B C : ℝ)

-- Conditions
def condition1 : Prop := A = (1/3) * (B + C)
def condition2 : Prop := B = (2/7) * (A + C)
def condition3 : Prop := A + B + C = 1080

-- Conclusion
theorem A_more_than_B (A B C : ℝ) (h1 : condition1 A B C) (h2 : condition2 A B C) (h3 : condition3 A B C) :
  A - B = 30 :=
sorry

end NUMINAMATH_GPT_A_more_than_B_l718_71892


namespace NUMINAMATH_GPT_f_500_l718_71809

-- Define a function f on positive integers
def f (n : ℕ) : ℕ := sorry

-- Assume the given conditions
axiom f_mul (x y : ℕ) (hx : x > 0) (hy : y > 0) : f (x * y) = f x + f y
axiom f_10 : f 10 = 14
axiom f_40 : f 40 = 20

-- Prove the required result
theorem f_500 : f 500 = 39 := by
  sorry

end NUMINAMATH_GPT_f_500_l718_71809


namespace NUMINAMATH_GPT_equation_of_line_passing_through_points_l718_71863

-- Definition of the points
def point1 : ℝ × ℝ := (-2, -3)
def point2 : ℝ × ℝ := (4, 7)

-- The statement to prove
theorem equation_of_line_passing_through_points :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ (forall (x y : ℝ), 
  y + 3 = (5 / 3) * (x + 2) → 3 * y - 5 * x = 1) := sorry

end NUMINAMATH_GPT_equation_of_line_passing_through_points_l718_71863


namespace NUMINAMATH_GPT_negation_of_proposition_l718_71831

variables (a b : ℕ)

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def both_even (a b : ℕ) : Prop := is_even a ∧ is_even b

def sum_even (a b : ℕ) : Prop := is_even (a + b)

theorem negation_of_proposition : ¬ (both_even a b → sum_even a b) ↔ ¬both_even a b ∨ ¬sum_even a b :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l718_71831


namespace NUMINAMATH_GPT_seeds_per_can_l718_71823

theorem seeds_per_can (total_seeds : Float) (cans : Float) (h1 : total_seeds = 54.0) (h2 : cans = 9.0) : total_seeds / cans = 6.0 :=
by
  sorry

end NUMINAMATH_GPT_seeds_per_can_l718_71823


namespace NUMINAMATH_GPT_number_of_Slurpees_l718_71876

theorem number_of_Slurpees
  (total_money : ℕ)
  (cost_per_Slurpee : ℕ)
  (change : ℕ)
  (spent_money := total_money - change)
  (number_of_Slurpees := spent_money / cost_per_Slurpee)
  (h1 : total_money = 20)
  (h2 : cost_per_Slurpee = 2)
  (h3 : change = 8) :
  number_of_Slurpees = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_Slurpees_l718_71876


namespace NUMINAMATH_GPT_range_of_sum_l718_71888

theorem range_of_sum (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b + 1 / a + 9 / b = 10) : 2 ≤ a + b ∧ a + b ≤ 8 :=
sorry

end NUMINAMATH_GPT_range_of_sum_l718_71888


namespace NUMINAMATH_GPT_bottles_produced_l718_71833

def machine_rate (total_machines : ℕ) (total_bottles_per_minute : ℕ) : ℕ :=
  total_bottles_per_minute / total_machines

def total_bottles (total_machines : ℕ) (bottles_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  total_machines * bottles_per_minute * minutes

theorem bottles_produced (machines1 machines2 minutes : ℕ) (bottles1 : ℕ) :
  machine_rate machines1 bottles1 = bottles1 / machines1 →
  total_bottles machines2 (bottles1 / machines1) minutes = 2160 :=
by
  intros machine_rate_eq
  sorry

end NUMINAMATH_GPT_bottles_produced_l718_71833


namespace NUMINAMATH_GPT_cat_mouse_position_after_moves_l718_71872

-- Define the total number of moves
def total_moves : ℕ := 360

-- Define cat's cycle length and position calculation
def cat_cycle_length : ℕ := 5
def cat_final_position := total_moves % cat_cycle_length

-- Define mouse's cycle length and actual moves per cycle
def mouse_cycle_length : ℕ := 10
def mouse_effective_moves_per_cycle : ℕ := 9
def total_mouse_effective_moves := (total_moves / mouse_cycle_length) * mouse_effective_moves_per_cycle
def mouse_final_position := total_mouse_effective_moves % mouse_cycle_length

theorem cat_mouse_position_after_moves :
  cat_final_position = 0 ∧ mouse_final_position = 4 :=
by
  sorry

end NUMINAMATH_GPT_cat_mouse_position_after_moves_l718_71872


namespace NUMINAMATH_GPT_Jazmin_strip_width_l718_71857

theorem Jazmin_strip_width (a b c : ℕ) (ha : a = 44) (hb : b = 33) (hc : c = 55) : Nat.gcd (Nat.gcd a b) c = 11 := by
  sorry

end NUMINAMATH_GPT_Jazmin_strip_width_l718_71857


namespace NUMINAMATH_GPT_fraction_simplification_l718_71852

-- We define the given fractions
def a := 3 / 7
def b := 2 / 9
def c := 5 / 12
def d := 1 / 4

-- We state the main theorem
theorem fraction_simplification : (a - b) / (c + d) = 13 / 42 := by
  -- Skipping proof for the equivalence problem
  sorry

end NUMINAMATH_GPT_fraction_simplification_l718_71852


namespace NUMINAMATH_GPT_simplify_fractions_l718_71839

theorem simplify_fractions :
  (20 / 19) * (15 / 28) * (76 / 45) = 95 / 84 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fractions_l718_71839


namespace NUMINAMATH_GPT_volunteer_count_change_l718_71843

theorem volunteer_count_change :
  let x := 1
  let fall_increase := 1.09
  let winter_increase := 1.15
  let spring_decrease := 0.81
  let summer_increase := 1.12
  let summer_end_decrease := 0.95
  let final_ratio := x * fall_increase * winter_increase * spring_decrease * summer_increase * summer_end_decrease
  (final_ratio - x) / x * 100 = 19.13 :=
by
  sorry

end NUMINAMATH_GPT_volunteer_count_change_l718_71843


namespace NUMINAMATH_GPT_quadratic_real_roots_range_l718_71871

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, (x^2 - x - m = 0)) ↔ m ≥ -1 / 4 :=
by sorry

end NUMINAMATH_GPT_quadratic_real_roots_range_l718_71871


namespace NUMINAMATH_GPT_courses_selection_l718_71879

-- Definition of the problem
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total number of ways person A can choose 2 courses from 4
def total_ways : ℕ := C 4 2 * C 4 2

-- Number of ways both choose exactly the same courses
def same_ways : ℕ := C 4 2

-- Prove the number of ways they can choose such that there is at least one course different
theorem courses_selection :
  total_ways - same_ways = 30 := by
  sorry

end NUMINAMATH_GPT_courses_selection_l718_71879


namespace NUMINAMATH_GPT_inequality_cube_l718_71801

theorem inequality_cube (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end NUMINAMATH_GPT_inequality_cube_l718_71801


namespace NUMINAMATH_GPT_flight_duration_is_four_hours_l718_71850

def convert_to_moscow_time (local_time : ℕ) (time_difference : ℕ) : ℕ :=
  (local_time - time_difference) % 24

def flight_duration (departure_time arrival_time : ℕ) : ℕ :=
  (arrival_time - departure_time) % 24

def duration_per_flight (total_flight_time : ℕ) (number_of_flights : ℕ) : ℕ :=
  total_flight_time / number_of_flights

theorem flight_duration_is_four_hours :
  let MoscowToBishkekTimeDifference := 3
  let departureMoscowTime := 12
  let arrivalBishkekLocalTime := 18
  let departureBishkekLocalTime := 8
  let arrivalMoscowTime := 10
  let outboundArrivalMoscowTime := convert_to_moscow_time arrivalBishkekLocalTime MoscowToBishkekTimeDifference
  let returnDepartureMoscowTime := convert_to_moscow_time departureBishkekLocalTime MoscowToBishkekTimeDifference
  let outboundDuration := flight_duration departureMoscowTime outboundArrivalMoscowTime
  let returnDuration := flight_duration returnDepartureMoscowTime arrivalMoscowTime
  let totalFlightTime := outboundDuration + returnDuration
  duration_per_flight totalFlightTime 2 = 4 := by
  sorry

end NUMINAMATH_GPT_flight_duration_is_four_hours_l718_71850


namespace NUMINAMATH_GPT_four_digit_number_exists_l718_71868

theorem four_digit_number_exists :
  ∃ (A B C D : ℕ), 
  B = 3 * A ∧ 
  C = A + B ∧ 
  D = 3 * B ∧ 
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ 
  1000 * A + 100 * B + 10 * C + D = 1349 :=
by {
  sorry 
}

end NUMINAMATH_GPT_four_digit_number_exists_l718_71868


namespace NUMINAMATH_GPT_solve_for_y_l718_71825

theorem solve_for_y (y : ℝ) : (5:ℝ)^(2*y + 3) = (625:ℝ)^y → y = 3/2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_y_l718_71825


namespace NUMINAMATH_GPT_length_of_stone_slab_l718_71836

theorem length_of_stone_slab 
  (num_slabs : ℕ) 
  (total_area : ℝ) 
  (h_num_slabs : num_slabs = 30) 
  (h_total_area : total_area = 50.7): 
  ∃ l : ℝ, l = 1.3 ∧ l * l * num_slabs = total_area := 
by 
  sorry

end NUMINAMATH_GPT_length_of_stone_slab_l718_71836


namespace NUMINAMATH_GPT_faster_speed_l718_71859

theorem faster_speed (x : ℝ) (h1 : 10 ≠ 0) (h2 : 5 * 10 = 50) (h3 : 50 + 20 = 70) (h4 : 5 = 70 / x) : x = 14 :=
by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_faster_speed_l718_71859


namespace NUMINAMATH_GPT_sum_of_angles_is_55_l718_71815

noncomputable def arc_BR : ℝ := 60
noncomputable def arc_RS : ℝ := 50
noncomputable def arc_AC : ℝ := 0
noncomputable def arc_BS := arc_BR + arc_RS
noncomputable def angle_P := (arc_BS - arc_AC) / 2
noncomputable def angle_R := arc_AC / 2
noncomputable def sum_of_angles := angle_P + angle_R

theorem sum_of_angles_is_55 :
  sum_of_angles = 55 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_angles_is_55_l718_71815


namespace NUMINAMATH_GPT_molly_age_l718_71861

theorem molly_age (S M : ℕ) (h1 : S / M = 4 / 3) (h2 : S + 6 = 34) : M = 21 :=
by
  sorry

end NUMINAMATH_GPT_molly_age_l718_71861


namespace NUMINAMATH_GPT_boys_meeting_problem_l718_71821

theorem boys_meeting_problem (d : ℝ) (t : ℝ)
  (speed1 speed2 : ℝ)
  (h1 : speed1 = 6) 
  (h2 : speed2 = 8) 
  (h3 : t > 0)
  (h4 : ∀ n : ℤ, n * (speed1 + speed2) * t ≠ d) : 
  0 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_boys_meeting_problem_l718_71821


namespace NUMINAMATH_GPT_exists_four_distinct_numbers_with_equal_half_sum_l718_71853

theorem exists_four_distinct_numbers_with_equal_half_sum (S : Finset ℕ) (h_card : S.card = 10) (h_range : ∀ x ∈ S, x ≤ 23) :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S) ∧ (a + b = c + d) :=
by
  sorry

end NUMINAMATH_GPT_exists_four_distinct_numbers_with_equal_half_sum_l718_71853


namespace NUMINAMATH_GPT_count_diff_squares_not_representable_1_to_1000_l718_71829

def num_not_diff_squares (n : ℕ) : ℕ :=
  (n + 1) / 4 * (if (n + 1) % 4 >= 2 then 1 else 0)

theorem count_diff_squares_not_representable_1_to_1000 :
  num_not_diff_squares 999 = 250 := 
sorry

end NUMINAMATH_GPT_count_diff_squares_not_representable_1_to_1000_l718_71829


namespace NUMINAMATH_GPT_range_of_m_l718_71884

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3*x + 2

theorem range_of_m (m : ℝ) :
  (∀ θ : ℝ, f (3 + 2 * Real.sin θ) < m) → m > 4 :=
sorry

end NUMINAMATH_GPT_range_of_m_l718_71884


namespace NUMINAMATH_GPT_distance_point_line_l718_71807

theorem distance_point_line (m : ℝ) : 
  abs (m + 1) = 2 ↔ (m = 1 ∨ m = -3) := by
  sorry

end NUMINAMATH_GPT_distance_point_line_l718_71807


namespace NUMINAMATH_GPT_difference_in_x_coordinates_is_constant_l718_71883

variable {a x₀ y₀ k : ℝ}

-- Define the conditions
def point_on_x_axis (a : ℝ) : Prop := true

def passes_through_fixed_point_and_tangent (a : ℝ) : Prop :=
  a = 1

def equation_of_curve_C (x y : ℝ) : Prop :=
  y^2 = 4 * x

def tangent_condition (a x₀ y₀ : ℝ) (k : ℝ) : Prop :=
  a > 2 ∧ y₀ > 0 ∧ y₀^2 = 4 * x₀ ∧ 
  (4 * x₀ - 2 * y₀ * y₀ + y₀^2 = 0)

-- The statement
theorem difference_in_x_coordinates_is_constant (a x₀ y₀ k : ℝ) :
  point_on_x_axis a →
  passes_through_fixed_point_and_tangent a →
  equation_of_curve_C x₀ y₀ →
  tangent_condition a x₀ y₀ k → 
  a - x₀ = 2 :=
by
  intro h1 h2 h3 h4 
  sorry

end NUMINAMATH_GPT_difference_in_x_coordinates_is_constant_l718_71883


namespace NUMINAMATH_GPT_area_transformed_function_l718_71818

noncomputable def area_g : ℝ := 15

noncomputable def area_4g_shifted : ℝ :=
  4 * area_g

theorem area_transformed_function :
  area_4g_shifted = 60 := by
  sorry

end NUMINAMATH_GPT_area_transformed_function_l718_71818


namespace NUMINAMATH_GPT_Moe_has_least_amount_of_money_l718_71851

variables (Money : Type) [LinearOrder Money]
variables (Bo Coe Flo Jo Moe Zoe : Money)
variables (Bo_lt_Flo : Bo < Flo) (Jo_lt_Flo : Jo < Flo)
variables (Moe_lt_Bo : Moe < Bo) (Moe_lt_Coe : Moe < Coe)
variables (Moe_lt_Jo : Moe < Jo) (Jo_lt_Bo : Jo < Bo)
variables (Moe_lt_Zoe : Moe < Zoe) (Zoe_lt_Jo : Zoe < Jo)

theorem Moe_has_least_amount_of_money : ∀ x, x ≠ Moe → Moe < x := by
  sorry

end NUMINAMATH_GPT_Moe_has_least_amount_of_money_l718_71851


namespace NUMINAMATH_GPT_simplify_expression_l718_71810

-- Define the constants and variables with required conditions
variables {x y z p q r : ℝ}

-- Assume the required distinctness conditions
axiom h1 : x ≠ p 
axiom h2 : y ≠ q 
axiom h3 : z ≠ r 

-- State the theorem to be proven
theorem simplify_expression (h : p ≠ q ∧ q ≠ r ∧ r ≠ p) : 
  (2 * (x - p) / (3 * (r - z))) * (2 * (y - q) / (3 * (p - x))) * (2 * (z - r) / (3 * (q - y))) = -8 / 27 :=
  sorry

end NUMINAMATH_GPT_simplify_expression_l718_71810


namespace NUMINAMATH_GPT_negation_proof_equivalence_l718_71814

theorem negation_proof_equivalence : 
  ¬(∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
sorry

end NUMINAMATH_GPT_negation_proof_equivalence_l718_71814


namespace NUMINAMATH_GPT_regular_pyramid_sufficient_condition_l718_71866

-- Define the basic structure of a pyramid
structure Pyramid :=
  (lateral_face_is_equilateral_triangle : Prop)  
  (base_is_square : Prop)  
  (apex_angles_of_lateral_face_are_45_deg : Prop)  
  (projection_of_vertex_at_intersection_of_base_diagonals : Prop)
  (is_regular : Prop)

-- Define the hypothesis conditions
variables 
  (P : Pyramid)
  (h1 : P.lateral_face_is_equilateral_triangle)
  (h2 : P.base_is_square)
  (h3 : P.apex_angles_of_lateral_face_are_45_deg)
  (h4 : P.projection_of_vertex_at_intersection_of_base_diagonals)

-- Define the statement of the proof
theorem regular_pyramid_sufficient_condition :
  (P.lateral_face_is_equilateral_triangle → P.is_regular) ∧ 
  (¬(P.lateral_face_is_equilateral_triangle) → ¬P.is_regular) ↔
  (P.lateral_face_is_equilateral_triangle ∧ ¬P.base_is_square ∧ ¬P.apex_angles_of_lateral_face_are_45_deg ∧ ¬P.projection_of_vertex_at_intersection_of_base_diagonals) := 
by { sorry }


end NUMINAMATH_GPT_regular_pyramid_sufficient_condition_l718_71866


namespace NUMINAMATH_GPT_conference_center_distance_l718_71891

variables (d t: ℝ)

theorem conference_center_distance
  (h1: ∃ t: ℝ, d = 45 * (t + 1.5))
  (h2: ∃ t: ℝ, d - 45 = 55 * (t - 1.25)):
  d = 478.125 :=
by
  sorry

end NUMINAMATH_GPT_conference_center_distance_l718_71891


namespace NUMINAMATH_GPT_imaginary_unit_sum_l718_71806

-- Define that i is the imaginary unit, which satisfies \(i^2 = -1\)
def is_imaginary_unit (i : ℂ) := i^2 = -1

-- The theorem to be proven: i + i^2 + i^3 + i^4 = 0 given that i is the imaginary unit
theorem imaginary_unit_sum (i : ℂ) (h : is_imaginary_unit i) : 
  i + i^2 + i^3 + i^4 = 0 := 
sorry

end NUMINAMATH_GPT_imaginary_unit_sum_l718_71806


namespace NUMINAMATH_GPT_sum_of_number_and_reverse_l718_71856

theorem sum_of_number_and_reverse (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) 
(h : (10 * a + b) - (10 * b + a) = 3 * (a + b)) : 
  (10 * a + b) + (10 * b + a) = 33 := 
sorry

end NUMINAMATH_GPT_sum_of_number_and_reverse_l718_71856


namespace NUMINAMATH_GPT_simplify_polynomial_l718_71865

theorem simplify_polynomial :
  (2 * x^6 + x^5 + 3 * x^4 + 7 * x^2 + 2 * x + 25) - (x^6 + 2 * x^5 + x^4 + x^3 + 8 * x^2 + 15) = 
  (x^6 - x^5 + 2 * x^4 - x^3 - x^2 + 2 * x + 10) :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l718_71865


namespace NUMINAMATH_GPT_computation_l718_71822

theorem computation :
  4.165 * 4.8 + 4.165 * 6.7 - 4.165 / (2 / 3) = 41.65 :=
by
  sorry

end NUMINAMATH_GPT_computation_l718_71822


namespace NUMINAMATH_GPT_find_a_l718_71816

theorem find_a (m c a b : ℝ) (h_m : m < 0) (h_radius : (m^2 + 3) = 4) 
  (h_c : c = 1 ∨ c = -3) (h_focus : c > 0) (h_ellipse : b^2 = 3) 
  (h_focus_eq : c^2 = a^2 - b^2) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l718_71816


namespace NUMINAMATH_GPT_find_angle_l718_71893

theorem find_angle :
  ∃ (x : ℝ), (90 - x = 0.4 * (180 - x)) → x = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_l718_71893


namespace NUMINAMATH_GPT_haley_number_of_shirts_l718_71877

-- Define the given information
def washing_machine_capacity : ℕ := 7
def total_loads : ℕ := 5
def number_of_sweaters : ℕ := 33
def number_of_shirts := total_loads * washing_machine_capacity - number_of_sweaters

-- The statement that needs to be proven
theorem haley_number_of_shirts : number_of_shirts = 2 := by
  sorry

end NUMINAMATH_GPT_haley_number_of_shirts_l718_71877


namespace NUMINAMATH_GPT_solve_system_of_inequalities_l718_71885

theorem solve_system_of_inequalities (x : ℝ) 
  (h1 : -3 * x^2 + 7 * x + 6 > 0) 
  (h2 : 4 * x - 4 * x^2 > -3) : 
  -1/2 < x ∧ x < 3/2 :=
sorry

end NUMINAMATH_GPT_solve_system_of_inequalities_l718_71885


namespace NUMINAMATH_GPT_simplify_expression_l718_71862

theorem simplify_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^(2/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l718_71862


namespace NUMINAMATH_GPT_centripetal_accel_v_r_centripetal_accel_omega_r_centripetal_accel_T_r_l718_71826

-- Defining the variables involved
variables {a v r ω T : ℝ}

-- Main theorem statements representing the problem
theorem centripetal_accel_v_r (v r : ℝ) (h₁ : 0 < r) : a = v^2 / r :=
sorry

theorem centripetal_accel_omega_r (ω r : ℝ) (h₁ : 0 < r) : a = r * ω^2 :=
sorry

theorem centripetal_accel_T_r (T r : ℝ) (h₁ : 0 < r) (h₂ : 0 < T) : a = 4 * π^2 * r / T^2 :=
sorry

end NUMINAMATH_GPT_centripetal_accel_v_r_centripetal_accel_omega_r_centripetal_accel_T_r_l718_71826


namespace NUMINAMATH_GPT_intersection_distance_to_pole_l718_71844

theorem intersection_distance_to_pole (rho theta : ℝ) (h1 : rho > 0) (h2 : rho = 2 * theta + 1) (h3 : rho * theta = 1) : rho = 2 :=
by
  -- We replace "sorry" with actual proof steps, if necessary.
  sorry

end NUMINAMATH_GPT_intersection_distance_to_pole_l718_71844


namespace NUMINAMATH_GPT_students_in_grades_2_and_3_l718_71847

theorem students_in_grades_2_and_3 (boys_2nd : ℕ) (girls_2nd : ℕ) (third_grade_factor : ℕ) 
  (h_boys_2nd : boys_2nd = 20) (h_girls_2nd : girls_2nd = 11) (h_third_grade_factor : third_grade_factor = 2) :
  (boys_2nd + girls_2nd) + ((boys_2nd + girls_2nd) * third_grade_factor) = 93 := by
  sorry

end NUMINAMATH_GPT_students_in_grades_2_and_3_l718_71847


namespace NUMINAMATH_GPT_negation_of_p_is_neg_p_l718_71802

-- Define the proposition p
def p : Prop := ∀ n : ℕ, 3^n ≥ n^2 + 1

-- Define the negation of p
def neg_p : Prop := ∃ n_0 : ℕ, 3^n_0 < n_0^2 + 1

-- The proof statement
theorem negation_of_p_is_neg_p : ¬p ↔ neg_p :=
by sorry

end NUMINAMATH_GPT_negation_of_p_is_neg_p_l718_71802


namespace NUMINAMATH_GPT_onions_left_on_shelf_l718_71897

def initial_onions : ℕ := 98
def sold_onions : ℕ := 65
def remaining_onions : ℕ := initial_onions - sold_onions

theorem onions_left_on_shelf : remaining_onions = 33 :=
by 
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_onions_left_on_shelf_l718_71897


namespace NUMINAMATH_GPT_min_value_of_expression_l718_71803

theorem min_value_of_expression
  (x y : ℝ) 
  (h : x + y = 1) : 
  ∃ (m : ℝ), m = 2 * x^2 + 3 * y^2 ∧ m = 6 / 5 := 
sorry

end NUMINAMATH_GPT_min_value_of_expression_l718_71803


namespace NUMINAMATH_GPT_cody_money_l718_71830

theorem cody_money (a b c d : ℕ) (h₁ : a = 45) (h₂ : b = 9) (h₃ : c = 19) (h₄ : d = a + b - c) : d = 35 :=
by
  rw [h₁, h₂, h₃] at h₄
  simp at h₄
  exact h₄

end NUMINAMATH_GPT_cody_money_l718_71830


namespace NUMINAMATH_GPT_find_a_l718_71817

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.sqrt x

theorem find_a (a : ℝ) (h_intersect : ∃ x₀, f a x₀ = g x₀) (h_tangent : ∃ x₀, (f a x₀) = g x₀ ∧ (1/x₀ * a = 1/ (2 * Real.sqrt x₀))):
  a = Real.exp 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l718_71817


namespace NUMINAMATH_GPT_soccer_balls_per_class_l718_71848

-- Definitions for all conditions in the problem
def elementary_classes_per_school : ℕ := 4
def middle_school_classes_per_school : ℕ := 5
def number_of_schools : ℕ := 2
def total_soccer_balls_donated : ℕ := 90

-- The total number of classes in one school
def classes_per_school : ℕ := elementary_classes_per_school + middle_school_classes_per_school

-- The total number of classes in both schools
def total_classes : ℕ := classes_per_school * number_of_schools

-- Prove that the number of soccer balls donated per class is 5
theorem soccer_balls_per_class : total_soccer_balls_donated / total_classes = 5 :=
  by sorry

end NUMINAMATH_GPT_soccer_balls_per_class_l718_71848


namespace NUMINAMATH_GPT_quadratic_ineq_solution_set_l718_71881

theorem quadratic_ineq_solution_set (a b c : ℝ) (h1 : a < 0) (h2 : ∀ x : ℝ, 3 < x → x < 6 → ax^2 + bx + c > 0) :
  ∀ x : ℝ, x < (1 / 6) ∨ x > (1 / 3) → cx^2 + bx + a < 0 := by 
  sorry

end NUMINAMATH_GPT_quadratic_ineq_solution_set_l718_71881


namespace NUMINAMATH_GPT_values_of_m_l718_71832

theorem values_of_m (m : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + (2 - m) * x + 12 = 0)) ↔ (m = -10 ∨ m = 14) := 
by
  sorry

end NUMINAMATH_GPT_values_of_m_l718_71832


namespace NUMINAMATH_GPT_highest_probability_of_red_ball_l718_71870

theorem highest_probability_of_red_ball (red yellow white blue : ℕ) (H1 : red = 5) (H2 : yellow = 4) (H3 : white = 1) (H4 : blue = 3) :
  (red : ℚ) / (red + yellow + white + blue) > (yellow : ℚ) / (red + yellow + white + blue) ∧
  (red : ℚ) / (red + yellow + white + blue) > (white : ℚ) / (red + yellow + white + blue) ∧
  (red : ℚ) / (red + yellow + white + blue) > (blue : ℚ) / (red + yellow + white + blue) := 
by {
  sorry
}

end NUMINAMATH_GPT_highest_probability_of_red_ball_l718_71870


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l718_71860

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l718_71860


namespace NUMINAMATH_GPT_clear_board_possible_l718_71842

def operation (board : Array (Array Nat)) (op_type : String) (index : Fin 8) : Array (Array Nat) :=
  match op_type with
  | "column" => board.map (λ row => row.modify index fun x => x - 1)
  | "row" => board.modify index fun row => row.map (λ x => 2 * x)
  | _ => board

def isZeroBoard (board : Array (Array Nat)) : Prop :=
  board.all (λ row => row.all (λ x => x = 0))

theorem clear_board_possible (initial_board : Array (Array Nat)) : 
  ∃ (ops : List (String × Fin 8)), 
    isZeroBoard (ops.foldl (λ b ⟨t, i⟩ => operation b t i) initial_board) :=
sorry

end NUMINAMATH_GPT_clear_board_possible_l718_71842


namespace NUMINAMATH_GPT_find_y_interval_l718_71875

theorem find_y_interval (y : ℝ) (h : y^2 - 8 * y + 12 < 0) : 2 < y ∧ y < 6 :=
sorry

end NUMINAMATH_GPT_find_y_interval_l718_71875


namespace NUMINAMATH_GPT_not_divisible_by_n_plus_4_l718_71898

theorem not_divisible_by_n_plus_4 (n : ℕ) : ¬ ∃ k : ℕ, n^2 + 8*n + 15 = k * (n + 4) :=
sorry

end NUMINAMATH_GPT_not_divisible_by_n_plus_4_l718_71898


namespace NUMINAMATH_GPT_sin_vertex_angle_isosceles_triangle_l718_71867

theorem sin_vertex_angle_isosceles_triangle (α β : ℝ) (h_isosceles : β = 2 * α) (tan_base_angle : Real.tan α = 2 / 3) :
  Real.sin β = 12 / 13 := 
sorry

end NUMINAMATH_GPT_sin_vertex_angle_isosceles_triangle_l718_71867


namespace NUMINAMATH_GPT_triangle_max_perimeter_l718_71874

noncomputable def max_perimeter_triangle_ABC (a b c : ℝ) (A B C : ℝ) (h1 : B = 60) (h2 : b = 2 * Real.sqrt 3) : ℝ := 
  a + b + c

theorem triangle_max_perimeter (a b c A B C : ℝ) (h1 : B = 60) (h2 : b = 2 * Real.sqrt 3) :
  max_perimeter_triangle_ABC a b c A B C h1 h2 ≤ 6 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_triangle_max_perimeter_l718_71874


namespace NUMINAMATH_GPT_valid_outfit_selections_l718_71869

-- Definitions based on the given conditions
def num_shirts : ℕ := 6
def num_pants : ℕ := 5
def num_hats : ℕ := 6
def num_colors : ℕ := 6

-- The total number of outfits without restrictions
def total_outfits : ℕ := num_shirts * num_pants * num_hats

-- The theorem statement to prove the final answer
theorem valid_outfit_selections : total_outfits = 150 :=
by
  have h1 : total_outfits = 6 * 5 * 6 := rfl
  have h2 : 6 * 5 * 6 = 180 := by norm_num
  have h3 : 180 = 150 := sorry -- Here you need to differentiate the invalid outfits using provided restrictions
  exact h3

end NUMINAMATH_GPT_valid_outfit_selections_l718_71869


namespace NUMINAMATH_GPT_remainder_when_n_plus_2947_divided_by_7_l718_71820

theorem remainder_when_n_plus_2947_divided_by_7 (n : ℤ) (h : n % 7 = 3) : (n + 2947) % 7 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_n_plus_2947_divided_by_7_l718_71820


namespace NUMINAMATH_GPT_quadratic_complete_square_l718_71805

theorem quadratic_complete_square:
  ∃ (a b c : ℝ), (∀ (x : ℝ), 3 * x^2 + 9 * x - 81 = a * (x + b) * (x + b) + c) ∧ a + b + c = -83.25 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_complete_square_l718_71805


namespace NUMINAMATH_GPT_tommy_house_price_l718_71800

variable (P : ℝ)

theorem tommy_house_price 
  (h1 : 1.25 * P = 125000) : 
  P = 100000 :=
by
  sorry

end NUMINAMATH_GPT_tommy_house_price_l718_71800


namespace NUMINAMATH_GPT_amount_lent_by_A_to_B_l718_71855

theorem amount_lent_by_A_to_B
  (P : ℝ)
  (H1 : P * 0.115 * 3 - P * 0.10 * 3 = 1125) :
  P = 25000 :=
by
  sorry

end NUMINAMATH_GPT_amount_lent_by_A_to_B_l718_71855


namespace NUMINAMATH_GPT_gcd_power_sub_one_l718_71812

theorem gcd_power_sub_one (a b : ℕ) (h1 : b = a + 30) : 
  Nat.gcd (2^a - 1) (2^b - 1) = 2^30 - 1 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_power_sub_one_l718_71812


namespace NUMINAMATH_GPT_true_weight_third_object_proof_l718_71864

noncomputable def true_weight_third_object (A a B b C : ℝ) : ℝ :=
  let h := Real.sqrt ((a - b) / (A - B))
  let k := (b * A - a * B) / ((A - B) * (h + 1))
  h * C + k

theorem true_weight_third_object_proof (A a B b C : ℝ) (h := Real.sqrt ((a - b) / (A - B))) (k := (b * A - a * B) / ((A - B) * (h + 1))) :
  true_weight_third_object A a B b C = h * C + k := by
  sorry

end NUMINAMATH_GPT_true_weight_third_object_proof_l718_71864


namespace NUMINAMATH_GPT_set_subset_find_m_l718_71808

open Set

def A (m : ℝ) : Set ℝ := {1, 3, 2 * m + 3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem set_subset_find_m (m : ℝ) : (B m ⊆ A m) → (m = 1 ∨ m = 3) :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_set_subset_find_m_l718_71808


namespace NUMINAMATH_GPT_percentage_reduction_is_correct_l718_71854

-- Definitions and initial conditions
def initial_price_per_model := 100
def models_for_kindergarten := 2
def models_for_elementary := 2 * models_for_kindergarten
def total_models := models_for_kindergarten + models_for_elementary
def total_cost_without_reduction := total_models * initial_price_per_model
def total_cost_paid := 570

-- Goal statement in Lean 4
theorem percentage_reduction_is_correct :
  (total_models > 5) →
  total_cost_paid = 570 →
  models_for_kindergarten = 2 →
  (total_cost_without_reduction - total_cost_paid) / total_models / initial_price_per_model * 100 = 5 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_percentage_reduction_is_correct_l718_71854


namespace NUMINAMATH_GPT_mike_baseball_cards_l718_71886

theorem mike_baseball_cards (initial_cards birthday_cards traded_cards : ℕ)
  (h1 : initial_cards = 64) 
  (h2 : birthday_cards = 18) 
  (h3 : traded_cards = 20) :
  initial_cards + birthday_cards - traded_cards = 62 :=
by 
  -- assumption:
  sorry

end NUMINAMATH_GPT_mike_baseball_cards_l718_71886


namespace NUMINAMATH_GPT_min_a_3b_l718_71828

theorem min_a_3b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (1 / (a + 3) + 1 / (b + 3) = 1 / 4)) : 
  a + 3*b ≥ 12 + 16*Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_min_a_3b_l718_71828


namespace NUMINAMATH_GPT_abs_inequality_equiv_l718_71827

theorem abs_inequality_equiv (x : ℝ) : 1 ≤ |x - 2| ∧ |x - 2| ≤ 7 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x ≤ 9) :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_equiv_l718_71827


namespace NUMINAMATH_GPT_pet_store_animals_l718_71824

theorem pet_store_animals (cats dogs birds : ℕ) 
    (ratio_cats_dogs_birds : 2 * birds = 4 * cats ∧ 3 * cats = 2 * dogs) 
    (num_cats : cats = 20) : dogs = 30 ∧ birds = 40 :=
by 
  -- This is where the proof would go, but we can skip it for this problem statement.
  sorry

end NUMINAMATH_GPT_pet_store_animals_l718_71824


namespace NUMINAMATH_GPT_length_of_greater_segment_l718_71819

-- Definitions based on conditions
variable (shorter longer : ℝ)
variable (h1 : longer = shorter + 2)
variable (h2 : (longer^2) - (shorter^2) = 32)

-- Proof goal
theorem length_of_greater_segment : longer = 9 :=
by
  sorry

end NUMINAMATH_GPT_length_of_greater_segment_l718_71819


namespace NUMINAMATH_GPT_student_ticket_price_l718_71845

theorem student_ticket_price
  (S : ℕ)
  (num_tickets : ℕ := 2000)
  (num_student_tickets : ℕ := 520)
  (price_non_student : ℕ := 11)
  (total_revenue : ℕ := 20960)
  (h : 520 * S + (2000 - 520) * 11 = 20960) :
  S = 9 :=
sorry

end NUMINAMATH_GPT_student_ticket_price_l718_71845
