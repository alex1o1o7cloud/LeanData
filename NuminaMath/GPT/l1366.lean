import Mathlib

namespace NUMINAMATH_GPT_derivative_u_l1366_136646

noncomputable def u (x : ℝ) : ℝ :=
  let z := Real.sin x
  let y := x^2
  Real.exp (z - 2 * y)

theorem derivative_u (x : ℝ) :
  deriv u x = Real.exp (Real.sin x - 2 * x^2) * (Real.cos x - 4 * x) :=
by
  sorry

end NUMINAMATH_GPT_derivative_u_l1366_136646


namespace NUMINAMATH_GPT_evaluate_expression_at_neg_two_l1366_136616

noncomputable def complex_expression (a : ℝ) : ℝ :=
  (1 - (a / (a + 1))) / (1 / (1 - a^2))

theorem evaluate_expression_at_neg_two :
  complex_expression (-2) = sorry :=
sorry

end NUMINAMATH_GPT_evaluate_expression_at_neg_two_l1366_136616


namespace NUMINAMATH_GPT_determine_BD_l1366_136680

variable (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AB BC CD DA : ℝ)
variables (BD : ℝ)

-- Setting up the conditions:
axiom AB_eq_5 : AB = 5
axiom BC_eq_17 : BC = 17
axiom CD_eq_5 : CD = 5
axiom DA_eq_9 : DA = 9
axiom BD_is_integer : ∃ (n : ℤ), BD = n

theorem determine_BD : BD = 13 :=
by
  sorry

end NUMINAMATH_GPT_determine_BD_l1366_136680


namespace NUMINAMATH_GPT_parallel_lines_l1366_136602

theorem parallel_lines (a : ℝ) : (∀ x y : ℝ, 2 * x + a * y + 1 = 0 ↔ x - 4 * y - 1 = 0) → a = -8 :=
by
  intro h -- Introduce the hypothesis that lines are parallel
  sorry -- Skip the proof

end NUMINAMATH_GPT_parallel_lines_l1366_136602


namespace NUMINAMATH_GPT_journey_time_difference_journey_time_difference_in_minutes_l1366_136629

-- Define the constant speed of the bus
def speed : ℕ := 60

-- Define distances of journeys
def distance_1 : ℕ := 360
def distance_2 : ℕ := 420

-- Define the time calculation function
def time (d : ℕ) (s : ℕ) : ℕ := d / s

-- State the theorem
theorem journey_time_difference :
  time distance_2 speed - time distance_1 speed = 1 :=
by
  sorry

-- Convert the time difference from hours to minutes
theorem journey_time_difference_in_minutes :
  (time distance_2 speed - time distance_1 speed) * 60 = 60 :=
by
  sorry

end NUMINAMATH_GPT_journey_time_difference_journey_time_difference_in_minutes_l1366_136629


namespace NUMINAMATH_GPT_real_to_fraction_l1366_136600

noncomputable def real_num : ℚ := 3.675

theorem real_to_fraction : real_num = 147 / 40 :=
by
  -- convert 3.675 to a mixed number
  have h1 : real_num = 3 + 675 / 1000 := by sorry
  -- find gcd of 675 and 1000
  have h2 : Nat.gcd 675 1000 = 25 := by sorry
  -- simplify 675/1000 to 27/40
  have h3 : 675 / 1000 = 27 / 40 := by sorry
  -- convert mixed number to improper fraction 147/40
  have h4 : 3 + 27 / 40 = 147 / 40 := by sorry
  -- combine the results to prove the required equality
  exact sorry

end NUMINAMATH_GPT_real_to_fraction_l1366_136600


namespace NUMINAMATH_GPT_cost_of_toys_target_weekly_price_l1366_136630

-- First proof problem: Cost of Plush Toy and Metal Ornament
theorem cost_of_toys (x : ℝ) (hx : 6400 / x = 2 * (4000 / (x + 20))) : 
  x = 80 :=
by sorry

-- Second proof problem: Price to achieve target weekly profit
theorem target_weekly_price (y : ℝ) (hy : (y - 80) * (10 + (150 - y) / 5) = 720) :
  y = 140 :=
by sorry

end NUMINAMATH_GPT_cost_of_toys_target_weekly_price_l1366_136630


namespace NUMINAMATH_GPT_solution_set_inequality_l1366_136620

theorem solution_set_inequality : {x : ℝ | (x + 3) * (1 - x) ≥ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1366_136620


namespace NUMINAMATH_GPT_complex_root_of_unity_prod_l1366_136670

theorem complex_root_of_unity_prod (r : ℂ) (h₁ : r^6 = 1) (h₂ : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 6 :=
by
  sorry

end NUMINAMATH_GPT_complex_root_of_unity_prod_l1366_136670


namespace NUMINAMATH_GPT_inequality_proof_l1366_136638

-- Given conditions
variables {a b : ℝ} (ha_lt_b : a < b) (hb_lt_0 : b < 0)

-- Question statement we want to prove
theorem inequality_proof : ab < 0 → a < b → b < 0 → ab > b^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1366_136638


namespace NUMINAMATH_GPT_find_matrix_N_l1366_136666

def matrix2x2 := ℚ × ℚ × ℚ × ℚ

def apply_matrix (M : matrix2x2) (v : ℚ × ℚ) : ℚ × ℚ :=
  let (a, b, c, d) := M;
  let (x, y) := v;
  (a * x + b * y, c * x + d * y)

theorem find_matrix_N : ∃ (N : matrix2x2), 
  apply_matrix N (3, 1) = (5, -1) ∧ 
  apply_matrix N (1, -2) = (0, 6) ∧ 
  N = (10/7, 5/7, 4/7, -19/7) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_matrix_N_l1366_136666


namespace NUMINAMATH_GPT_problem1_proof_problem2_proof_l1366_136619

section Problems

variable {x a : ℝ}

-- Problem 1
theorem problem1_proof : 3 * x^2 * x^4 - (-x^3)^2 = 2 * x^6 := by
  sorry

-- Problem 2
theorem problem2_proof : a^3 * a + (-a^2)^3 / a^2 = 0 := by
  sorry

end Problems

end NUMINAMATH_GPT_problem1_proof_problem2_proof_l1366_136619


namespace NUMINAMATH_GPT_arrangement_ways_count_l1366_136623

theorem arrangement_ways_count:
  let n := 10
  let k := 4
  (Nat.choose n k) = 210 :=
by
  sorry

end NUMINAMATH_GPT_arrangement_ways_count_l1366_136623


namespace NUMINAMATH_GPT_range_of_x_for_function_l1366_136607

theorem range_of_x_for_function :
  ∀ x : ℝ, (2 - x ≥ 0 ∧ x - 1 ≠ 0) ↔ (x ≤ 2 ∧ x ≠ 1) := by
  sorry

end NUMINAMATH_GPT_range_of_x_for_function_l1366_136607


namespace NUMINAMATH_GPT_sale_price_correct_l1366_136649

variable (x : ℝ)

-- Conditions
def decreased_price (x : ℝ) : ℝ :=
  0.9 * x

def final_sale_price (decreased_price : ℝ) : ℝ :=
  0.7 * decreased_price

-- Proof statement
theorem sale_price_correct : final_sale_price (decreased_price x) = 0.63 * x := by
  sorry

end NUMINAMATH_GPT_sale_price_correct_l1366_136649


namespace NUMINAMATH_GPT_land_area_of_each_section_l1366_136694

theorem land_area_of_each_section (n : ℕ) (total_area : ℕ) (h1 : n = 3) (h2 : total_area = 7305) :
  total_area / n = 2435 :=
by {
  sorry
}

end NUMINAMATH_GPT_land_area_of_each_section_l1366_136694


namespace NUMINAMATH_GPT_northton_time_capsule_depth_l1366_136693

def southton_depth : ℕ := 15

def northton_depth : ℕ := 4 * southton_depth + 12

theorem northton_time_capsule_depth : northton_depth = 72 := by
  sorry

end NUMINAMATH_GPT_northton_time_capsule_depth_l1366_136693


namespace NUMINAMATH_GPT_polynomial_factorization_l1366_136685

noncomputable def factorize_polynomial (a b : ℝ) : ℝ :=
  -3 * a^3 * b + 6 * a^2 * b^2 - 3 * a * b^3

theorem polynomial_factorization (a b : ℝ) : 
  factorize_polynomial a b = -3 * a * b * (a - b)^2 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_factorization_l1366_136685


namespace NUMINAMATH_GPT_perpendicular_slope_solution_l1366_136604

theorem perpendicular_slope_solution (a : ℝ) :
  (∀ x y : ℝ, ax + (3 - a) * y + 1 = 0) →
  (∀ x y : ℝ, x - 2 * y = 0) →
  (l1_perp_l2 : ∀ x y : ℝ, ax + (3 - a) * y + 1 = 0 → x - 2 * y = 0 → False) →
  a = 2 :=
sorry

end NUMINAMATH_GPT_perpendicular_slope_solution_l1366_136604


namespace NUMINAMATH_GPT_amount_deducted_from_third_l1366_136687

theorem amount_deducted_from_third
  (x : ℝ) 
  (h1 : ((x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) / 10 = 16)) 
  (h2 : (( (x - 9) + ((x + 1) - 8) + ((x + 2) - d) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9) ) / 10 = 11.5)) :
  d = 13.5 :=
by
  sorry

end NUMINAMATH_GPT_amount_deducted_from_third_l1366_136687


namespace NUMINAMATH_GPT_equal_candies_l1366_136692

theorem equal_candies
  (sweet_math_per_box : ℕ := 12)
  (geometry_nuts_per_box : ℕ := 15)
  (sweet_math_boxes : ℕ := 5)
  (geometry_nuts_boxes : ℕ := 4) :
  sweet_math_boxes * sweet_math_per_box = geometry_nuts_boxes * geometry_nuts_per_box := 
  by
  sorry

end NUMINAMATH_GPT_equal_candies_l1366_136692


namespace NUMINAMATH_GPT_point_below_line_l1366_136665

theorem point_below_line {a : ℝ} (h : 2 * a - 3 < 3) : a < 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_point_below_line_l1366_136665


namespace NUMINAMATH_GPT_max_value_of_expression_l1366_136635

theorem max_value_of_expression (x y : ℝ) 
  (h : Real.sqrt (x * y) + Real.sqrt ((1 - x) * (1 - y)) = Real.sqrt (7 * x * (1 - y)) + (Real.sqrt (y * (1 - x)) / Real.sqrt 7)) :
  x + 7 * y ≤ 57 / 8 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l1366_136635


namespace NUMINAMATH_GPT_part1_arithmetic_sequence_part2_minimum_value_Sn_l1366_136672

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end NUMINAMATH_GPT_part1_arithmetic_sequence_part2_minimum_value_Sn_l1366_136672


namespace NUMINAMATH_GPT_pos_integers_divisible_by_2_3_5_7_less_than_300_l1366_136601

theorem pos_integers_divisible_by_2_3_5_7_less_than_300 : 
  ∃ n : ℕ, n = 1 ∧ ∀ k : ℕ, k < 300 → 2 ∣ k ∧ 3 ∣ k ∧ 5 ∣ k ∧ 7 ∣ k → k = n * (210 : ℕ) :=
by
  sorry

end NUMINAMATH_GPT_pos_integers_divisible_by_2_3_5_7_less_than_300_l1366_136601


namespace NUMINAMATH_GPT_largest_possible_three_day_success_ratio_l1366_136625

noncomputable def beta_max_success_ratio : ℝ :=
  let (a : ℕ) := 33
  let (b : ℕ) := 50
  let (c : ℕ) := 225
  let (d : ℕ) := 300
  let (e : ℕ) := 100
  let (f : ℕ) := 200
  a / b + c / d + e / f

theorem largest_possible_three_day_success_ratio :
  beta_max_success_ratio = (358 / 600 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_three_day_success_ratio_l1366_136625


namespace NUMINAMATH_GPT_negation_equivalence_l1366_136644

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_equivalence_l1366_136644


namespace NUMINAMATH_GPT_solve_abs_eqn_l1366_136676

theorem solve_abs_eqn (y : ℝ) : (|y - 4| + 3 * y = 15) ↔ (y = 19 / 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_abs_eqn_l1366_136676


namespace NUMINAMATH_GPT_range_of_a_l1366_136647

variables {f : ℝ → ℝ} (a : ℝ)

-- Even function definition
def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

-- Monotonically increasing on (-∞, 0)
def mono_increasing_on_neg (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → y < 0 → f x ≤ f y

-- Problem statement
theorem range_of_a
  (h_even : even_function f)
  (h_mono_neg : mono_increasing_on_neg f)
  (h_inequality : f (2 ^ |a - 1|) > f 4) :
  -1 < a ∧ a < 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1366_136647


namespace NUMINAMATH_GPT_simplify_expression_to_fraction_l1366_136673

theorem simplify_expression_to_fraction : 
  (1 / (1 / (1/2)^2 + 1 / (1/2)^3 + 1 / (1/2)^4 + 1 / (1/2)^5)) = 1/60 :=
by 
  have h1 : 1 / (1/2)^2 = 4 := by sorry
  have h2 : 1 / (1/2)^3 = 8 := by sorry
  have h3 : 1 / (1/2)^4 = 16 := by sorry
  have h4 : 1 / (1/2)^5 = 32 := by sorry
  have h5 : 4 + 8 + 16 + 32 = 60 := by sorry
  have h6 : 1 / 60 = 1/60 := by sorry
  sorry

end NUMINAMATH_GPT_simplify_expression_to_fraction_l1366_136673


namespace NUMINAMATH_GPT_simpsons_paradox_example_l1366_136682

theorem simpsons_paradox_example :
  ∃ n1 n2 a1 a2 b1 b2,
    n1 = 10 ∧ a1 = 3 ∧ b1 = 2 ∧
    n2 = 90 ∧ a2 = 45 ∧ b2 = 488 ∧
    ((a1 : ℝ) / n1 > (b1 : ℝ) / n1) ∧
    ((a2 : ℝ) / n2 > (b2 : ℝ) / n2) ∧
    ((a1 + a2 : ℝ) / (n1 + n2) < (b1 + b2 : ℝ) / (n1 + n2)) :=
by
  use 10, 90, 3, 45, 2, 488
  simp
  sorry

end NUMINAMATH_GPT_simpsons_paradox_example_l1366_136682


namespace NUMINAMATH_GPT_one_plane_halves_rect_prism_l1366_136674

theorem one_plane_halves_rect_prism :
  ∀ (T : Type) (a b c : ℝ)
  (x y z : ℝ) 
  (black_prisms_volume white_prisms_volume : ℝ),
  (black_prisms_volume = (x * y * z + x * (b - y) * (c - z) + (a - x) * y * (c - z) + (a - x) * (b - y) * z)) ∧
  (white_prisms_volume = ((a - x) * (b - y) * (c - z) + (a - x) * y * z + x * (b - y) * z + x * y * (c - z))) ∧
  (black_prisms_volume = white_prisms_volume) →
  (x = a / 2 ∨ y = b / 2 ∨ z = c / 2) :=
by
  sorry

end NUMINAMATH_GPT_one_plane_halves_rect_prism_l1366_136674


namespace NUMINAMATH_GPT_range_of_m_l1366_136677

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, f x < |m - 2|) ↔ m < 0 ∨ m > 4 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1366_136677


namespace NUMINAMATH_GPT_smallest_disk_cover_count_l1366_136643

theorem smallest_disk_cover_count (D : ℝ) (r : ℝ) (n : ℕ) 
  (hD : D = 1) (hr : r = 1 / 2) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_smallest_disk_cover_count_l1366_136643


namespace NUMINAMATH_GPT_surveys_completed_total_l1366_136621

variable (regular_rate cellphone_rate total_earnings cellphone_surveys total_surveys : ℕ)
variable (h_regular_rate : regular_rate = 10)
variable (h_cellphone_rate : cellphone_rate = 13) -- 30% higher than regular_rate
variable (h_total_earnings : total_earnings = 1180)
variable (h_cellphone_surveys : cellphone_surveys = 60)
variable (h_total_surveys : total_surveys = cellphone_surveys + (total_earnings - (cellphone_surveys * cellphone_rate)) / regular_rate)

theorem surveys_completed_total :
  total_surveys = 100 :=
by
  sorry

end NUMINAMATH_GPT_surveys_completed_total_l1366_136621


namespace NUMINAMATH_GPT_total_kids_receive_macarons_l1366_136659

theorem total_kids_receive_macarons :
  let mitch_good := 18
  let joshua := 26 -- 20 + 6
  let joshua_good := joshua - 3
  let miles := joshua * 2
  let miles_good := miles
  let renz := (3 * miles) / 4 - 1
  let renz_good := renz - 4
  let leah_good := 35 - 5
  let total_good := mitch_good + joshua_good + miles_good + renz_good + leah_good 
  let kids_with_3_macarons := 10
  let macaron_per_3 := kids_with_3_macarons * 3
  let remaining_macarons := total_good - macaron_per_3
  let kids_with_2_macarons := remaining_macarons / 2
  kids_with_3_macarons + kids_with_2_macarons = 73 :=
by 
  sorry

end NUMINAMATH_GPT_total_kids_receive_macarons_l1366_136659


namespace NUMINAMATH_GPT_verify_addition_by_subtraction_l1366_136627

theorem verify_addition_by_subtraction (a b c : ℤ) (h : a + b = c) : (c - a = b) ∧ (c - b = a) :=
by
  sorry

end NUMINAMATH_GPT_verify_addition_by_subtraction_l1366_136627


namespace NUMINAMATH_GPT_sum_of_three_consecutive_odd_integers_l1366_136658

-- Define the variables and conditions
variables (a : ℤ) (h1 : (a + (a + 4) = 100))

-- Define the statement that needs to be proved
theorem sum_of_three_consecutive_odd_integers (ha : a = 48) : a + (a + 2) + (a + 4) = 150 := by
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_odd_integers_l1366_136658


namespace NUMINAMATH_GPT_songs_downloaded_later_l1366_136642

-- Definition that each song has a size of 5 MB
def song_size : ℕ := 5

-- Definition that the new songs will occupy 140 MB of memory space
def total_new_song_memory : ℕ := 140

-- Prove that the number of songs Kira downloaded later on that day is 28
theorem songs_downloaded_later (x : ℕ) (h : song_size * x = total_new_song_memory) : x = 28 :=
by
  sorry

end NUMINAMATH_GPT_songs_downloaded_later_l1366_136642


namespace NUMINAMATH_GPT_find_value_less_than_twice_l1366_136614

def value_less_than_twice_another (x y v : ℕ) : Prop :=
  y = 2 * x - v ∧ x + y = 51 ∧ y = 33

theorem find_value_less_than_twice (x y v : ℕ) (h : value_less_than_twice_another x y v) : v = 3 := by
  sorry

end NUMINAMATH_GPT_find_value_less_than_twice_l1366_136614


namespace NUMINAMATH_GPT_cost_of_childrens_ticket_l1366_136633

theorem cost_of_childrens_ticket (x : ℝ) 
  (h1 : ∀ A C : ℝ, A = 2 * C) 
  (h2 : 152 = 2 * 76)
  (h3 : ∀ A C : ℝ, 5.50 * A + x * C = 1026) 
  (h4 : 152 = 152) : 
  x = 2.50 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_childrens_ticket_l1366_136633


namespace NUMINAMATH_GPT_toys_lost_l1366_136679

theorem toys_lost (initial_toys found_in_closet total_after_finding : ℕ) 
  (h1 : initial_toys = 40) 
  (h2 : found_in_closet = 9) 
  (h3 : total_after_finding = 43) : 
  initial_toys - (total_after_finding - found_in_closet) = 9 :=
by 
  sorry

end NUMINAMATH_GPT_toys_lost_l1366_136679


namespace NUMINAMATH_GPT_not_sufficient_nor_necessary_condition_l1366_136695

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

def is_increasing_for_nonpositive (f : ℝ → ℝ) : Prop :=
∀ x y, x ≤ 0 → y ≤ 0 → x < y → f x < f y

theorem not_sufficient_nor_necessary_condition
  {f : ℝ → ℝ}
  (hf_even : is_even_function f)
  (hf_incr : is_increasing_for_nonpositive f)
  (x : ℝ) :
  (6/5 < x ∧ x < 2) → ¬((1 < x ∧ x < 7/4) ↔ (f (Real.log (2 * x - 2) / Real.log 2) > f (Real.log (2 / 3) / Real.log (1 / 2)))) :=
sorry

end NUMINAMATH_GPT_not_sufficient_nor_necessary_condition_l1366_136695


namespace NUMINAMATH_GPT_outlets_per_room_l1366_136640

theorem outlets_per_room
  (rooms : ℕ)
  (total_outlets : ℕ)
  (h1 : rooms = 7)
  (h2 : total_outlets = 42) :
  total_outlets / rooms = 6 :=
by sorry

end NUMINAMATH_GPT_outlets_per_room_l1366_136640


namespace NUMINAMATH_GPT_ball_maximum_height_l1366_136699
-- Import necessary libraries

-- Define the height function
def ball_height (t : ℝ) : ℝ := -5 * t^2 + 50 * t + 20

-- Proposition asserting that the maximum height of the ball is 145 meters
theorem ball_maximum_height : ∃ t : ℝ, ball_height t = 145 :=
  sorry

end NUMINAMATH_GPT_ball_maximum_height_l1366_136699


namespace NUMINAMATH_GPT_hike_distance_l1366_136668

theorem hike_distance :
  let total_distance := 0.7
  let car_to_stream := 0.2
  let meadow_to_campsite := 0.1
  let stream_to_meadow := total_distance - (car_to_stream + meadow_to_campsite)
  stream_to_meadow = 0.4 :=
by
  let total_distance := 0.7
  let car_to_stream := 0.2
  let meadow_to_campsite := 0.1
  let stream_to_meadow := total_distance - (car_to_stream + meadow_to_campsite)
  show stream_to_meadow = 0.4
  sorry

end NUMINAMATH_GPT_hike_distance_l1366_136668


namespace NUMINAMATH_GPT_expansion_coefficient_a2_l1366_136613

theorem expansion_coefficient_a2 (z x : ℂ) 
  (h : z = 1 + I) : 
  ∃ a_0 a_1 a_2 a_3 a_4 : ℂ,
    (z + x)^4 = a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0
    ∧ a_2 = 12 * I :=
by
  sorry

end NUMINAMATH_GPT_expansion_coefficient_a2_l1366_136613


namespace NUMINAMATH_GPT_find_principal_amount_l1366_136626

variable (P : ℝ)

def interestA_to_B (P : ℝ) : ℝ := P * 0.10 * 3
def interestB_from_C (P : ℝ) : ℝ := P * 0.115 * 3
def gain_B (P : ℝ) : ℝ := interestB_from_C P - interestA_to_B P

theorem find_principal_amount (h : gain_B P = 45) : P = 1000 := by
  sorry

end NUMINAMATH_GPT_find_principal_amount_l1366_136626


namespace NUMINAMATH_GPT_sum_of_first_4_terms_arithmetic_sequence_l1366_136609

variable {a : ℕ → ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d a1, (∀ n, a n = a1 + n * d) ∧ (a 3 - a 1 = 2) ∧ (a 5 = 5)

-- Define the sum S4 for the first 4 terms of the sequence
def sum_first_4_terms (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3

-- Define the Lean statement for the problem
theorem sum_of_first_4_terms_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a → sum_first_4_terms a = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_4_terms_arithmetic_sequence_l1366_136609


namespace NUMINAMATH_GPT_horner_value_x_neg2_l1366_136698

noncomputable def horner (x : ℝ) : ℝ :=
  (((((x - 5) * x + 6) * x + 0) * x + 1) * x + 0.3) * x + 2

theorem horner_value_x_neg2 : horner (-2) = -40 :=
by
  sorry

end NUMINAMATH_GPT_horner_value_x_neg2_l1366_136698


namespace NUMINAMATH_GPT_Frank_has_four_one_dollar_bills_l1366_136661

noncomputable def Frank_one_dollar_bills : ℕ :=
  let total_money := 4 * 5 + 2 * 10 + 20 -- Money from five, ten, and twenty dollar bills
  let peanuts_cost := 10 - 4 -- Cost of peanuts (given $10 and received $4 in change)
  let one_dollar_bills_value := 54 - total_money -- Total money Frank has - money from large bills
  (one_dollar_bills_value : ℕ)

theorem Frank_has_four_one_dollar_bills 
   (five_dollar_bills : ℕ := 4) 
   (ten_dollar_bills : ℕ := 2)
   (twenty_dollar_bills : ℕ := 1)
   (peanut_price : ℚ := 3)
   (change : ℕ := 4)
   (total_money : ℕ := 50)
   (total_money_incl_change : ℚ := 54):
   Frank_one_dollar_bills = 4 := by
  sorry

end NUMINAMATH_GPT_Frank_has_four_one_dollar_bills_l1366_136661


namespace NUMINAMATH_GPT_angle_F_after_decrease_l1366_136611

theorem angle_F_after_decrease (D E F : ℝ) (h1 : D = 60) (h2 : E = 60) (h3 : F = 60) (h4 : E = D) :
  F - 20 = 40 := by
  simp [h3]
  sorry

end NUMINAMATH_GPT_angle_F_after_decrease_l1366_136611


namespace NUMINAMATH_GPT_sum_of_largest_smallest_angles_l1366_136690

noncomputable section

def sides_ratio (a b c : ℝ) : Prop := a / 5 = b / 7 ∧ b / 7 = c / 8

theorem sum_of_largest_smallest_angles (a b c : ℝ) (θA θB θC : ℝ) 
  (h1 : sides_ratio a b c) 
  (h2 : a^2 + b^2 - c^2 = 2 * a * b * Real.cos θC)
  (h3 : b^2 + c^2 - a^2 = 2 * b * c * Real.cos θA)
  (h4 : c^2 + a^2 - b^2 = 2 * c * a * Real.cos θB)
  (h5 : θA + θB + θC = 180) :
  θA + θC = 120 :=
sorry

end NUMINAMATH_GPT_sum_of_largest_smallest_angles_l1366_136690


namespace NUMINAMATH_GPT_labourer_savings_l1366_136667

theorem labourer_savings
  (monthly_expenditure_first_6_months : ℕ)
  (monthly_expenditure_next_4_months : ℕ)
  (monthly_income : ℕ)
  (total_expenditure_first_6_months : ℕ)
  (total_income_first_6_months : ℕ)
  (debt_incurred : ℕ)
  (total_expenditure_next_4_months : ℕ)
  (total_income_next_4_months : ℕ)
  (money_saved : ℕ) :
  monthly_expenditure_first_6_months = 85 →
  monthly_expenditure_next_4_months = 60 →
  monthly_income = 78 →
  total_expenditure_first_6_months = 6 * monthly_expenditure_first_6_months →
  total_income_first_6_months = 6 * monthly_income →
  debt_incurred = total_expenditure_first_6_months - total_income_first_6_months →
  total_expenditure_next_4_months = 4 * monthly_expenditure_next_4_months →
  total_income_next_4_months = 4 * monthly_income →
  money_saved = total_income_next_4_months - (total_expenditure_next_4_months + debt_incurred) →
  money_saved = 30 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_labourer_savings_l1366_136667


namespace NUMINAMATH_GPT_dot_product_theorem_l1366_136697

open Real

namespace VectorProof

-- Define the vectors m and n
def m := (2, 5)
def n (t : ℝ) := (-5, t)

-- Define the condition that m is perpendicular to n
def perpendicular (t : ℝ) : Prop := (2 * -5) + (5 * t) = 0

-- Function to calculate the dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define the vectors m+n and m-2n
def vector_add (t : ℝ) : ℝ × ℝ := (m.1 + (n t).1, m.2 + (n t).2)
def vector_sub (t : ℝ) : ℝ × ℝ := (m.1 - 2 * (n t).1, m.2 - 2 * (n t).2)

-- The theorem to prove
theorem dot_product_theorem : ∀ (t : ℝ), perpendicular t → dot_product (vector_add t) (vector_sub t) = -29 :=
by
  intros t ht
  sorry

end VectorProof

end NUMINAMATH_GPT_dot_product_theorem_l1366_136697


namespace NUMINAMATH_GPT_max_value_of_quadratic_l1366_136663

theorem max_value_of_quadratic :
  ∀ (x : ℝ), ∃ y : ℝ, y = -3 * x^2 + 18 ∧
  (∀ x' : ℝ, -3 * x'^2 + 18 ≤ y) := by
  sorry

end NUMINAMATH_GPT_max_value_of_quadratic_l1366_136663


namespace NUMINAMATH_GPT_proof_problem_l1366_136660

noncomputable def problem_statement (m : ℕ) : Prop :=
  ∀ pairs : List (ℕ × ℕ),
  (∀ (x y : ℕ), (x, y) ∈ pairs ↔ x^2 - 3 * y^2 + 2 = 16 * m ∧ 2 * y ≤ x - 1) →
  pairs.length % 2 = 0 ∨ pairs.length = 0

theorem proof_problem (m : ℕ) (hm : m > 0) : problem_statement m :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1366_136660


namespace NUMINAMATH_GPT_radius_of_circle_l1366_136681

-- Define the given circle equation as a condition
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 7 = 0

theorem radius_of_circle : ∀ x y : ℝ, circle_equation x y → ∃ r : ℝ, r = 3 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_l1366_136681


namespace NUMINAMATH_GPT_find_angle_MON_l1366_136612

-- Definitions of conditions
variables {A B O C M N : Type} -- Points in a geometric space
variables (angle_AOB : ℝ) (ray_OC : Prop) (bisects_OM : Prop) (bisects_ON : Prop)
variables (angle_MOB : ℝ) (angle_MON : ℝ)

-- Conditions
-- Angle AOB is 90 degrees
def angle_AOB_90 (angle_AOB : ℝ) : Prop := angle_AOB = 90

-- OC is a ray (using a placeholder property for ray, as Lean may not have geometric entities)
def OC_is_ray (ray_OC : Prop) : Prop := ray_OC

-- OM bisects angle BOC
def OM_bisects_BOC (bisects_OM : Prop) : Prop := bisects_OM

-- ON bisects angle AOC
def ON_bisects_AOC (bisects_ON : Prop) : Prop := bisects_ON

-- The problem statement as a theorem in Lean
theorem find_angle_MON
  (h1 : angle_AOB_90 angle_AOB)
  (h2 : OC_is_ray ray_OC)
  (h3 : OM_bisects_BOC bisects_OM)
  (h4 : ON_bisects_AOC bisects_ON) :
  angle_MON = 45 ∨ angle_MON = 135 :=
sorry

end NUMINAMATH_GPT_find_angle_MON_l1366_136612


namespace NUMINAMATH_GPT_remainder_sum_of_numbers_l1366_136641

theorem remainder_sum_of_numbers :
  ((123450 + 123451 + 123452 + 123453 + 123454 + 123455) % 7) = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_sum_of_numbers_l1366_136641


namespace NUMINAMATH_GPT_jack_helped_hours_l1366_136691

-- Definitions based on the problem's conditions
def sam_rate : ℕ := 6  -- Sam assembles 6 widgets per hour
def tony_rate : ℕ := 2  -- Tony assembles 2 widgets per hour
def jack_rate : ℕ := sam_rate  -- Jack assembles at the same rate as Sam
def total_widgets : ℕ := 68  -- The total number of widgets assembled by all three

-- Statement to prove
theorem jack_helped_hours : 
  ∃ h : ℕ, (sam_rate * h) + (tony_rate * h) + (jack_rate * h) = total_widgets ∧ h = 4 := 
  by
  -- The proof is not necessary; we only need the statement
  sorry

end NUMINAMATH_GPT_jack_helped_hours_l1366_136691


namespace NUMINAMATH_GPT_intersection_complement_R_M_and_N_l1366_136636

open Set

def universalSet := ℝ
def M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def complementR (S : Set ℝ) := {x : ℝ | x ∉ S}
def N := {x : ℝ | x < 1}

theorem intersection_complement_R_M_and_N:
  (complementR M ∩ N) = {x : ℝ | x < -2} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_R_M_and_N_l1366_136636


namespace NUMINAMATH_GPT_no_valid_C_for_2C4_multiple_of_5_l1366_136624

theorem no_valid_C_for_2C4_multiple_of_5 :
  ¬ (∃ C : ℕ, C < 10 ∧ (2 * 100 + C * 10 + 4) % 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_valid_C_for_2C4_multiple_of_5_l1366_136624


namespace NUMINAMATH_GPT_range_of_f_l1366_136671

noncomputable def f (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_f :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → (f x ≥ (Real.pi / 2 - Real.arctan 2) ∧ f x ≤ (Real.pi / 2 + Real.arctan 2)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l1366_136671


namespace NUMINAMATH_GPT_airplane_seats_l1366_136662

theorem airplane_seats (s : ℝ)
  (h1 : 0.30 * s = 0.30 * s)
  (h2 : (3 / 5) * s = (3 / 5) * s)
  (h3 : 36 + 0.30 * s + (3 / 5) * s = s) : s = 360 :=
by
  sorry

end NUMINAMATH_GPT_airplane_seats_l1366_136662


namespace NUMINAMATH_GPT_three_digit_number_with_units5_and_hundreds3_divisible_by_9_l1366_136654

theorem three_digit_number_with_units5_and_hundreds3_divisible_by_9 :
  ∃ n : ℕ, ∃ x : ℕ, n = 305 + 10 * x ∧ (n % 9) = 0 ∧ n = 315 := by
sorry

end NUMINAMATH_GPT_three_digit_number_with_units5_and_hundreds3_divisible_by_9_l1366_136654


namespace NUMINAMATH_GPT_sum_geq_three_implies_one_geq_two_l1366_136651

theorem sum_geq_three_implies_one_geq_two (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 :=
by { sorry }

end NUMINAMATH_GPT_sum_geq_three_implies_one_geq_two_l1366_136651


namespace NUMINAMATH_GPT_range_of_m_l1366_136608

noncomputable def p (m : ℝ) : Prop :=
  (m > 2)

noncomputable def q (m : ℝ) : Prop :=
  (m > 1)

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → (1 < m ∧ m ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1366_136608


namespace NUMINAMATH_GPT_peggy_records_l1366_136678

theorem peggy_records (R : ℕ) (h : 4 * R - (3 * R + R / 2) = 100) : R = 200 :=
sorry

end NUMINAMATH_GPT_peggy_records_l1366_136678


namespace NUMINAMATH_GPT_mean_computation_l1366_136605

theorem mean_computation (x y : ℝ) 
  (h1 : (28 + x + 70 + 88 + 104) / 5 = 67)
  (h2 : (if x < 50 ∧ x < 62 then if y < 62 then ((28 + y) / 2 = 81) else ((62 + x) / 2 = 81) else if y < 50 then ((y + 50) / 2 = 81) else if y < 62 then ((50 + y) / 2 = 81) else ((50 + x) / 2 = 81)) -- conditions for median can be simplified and expanded as necessary
) : (50 + 62 + 97 + 124 + x + y) / 6 = 82.5 :=
sorry

end NUMINAMATH_GPT_mean_computation_l1366_136605


namespace NUMINAMATH_GPT_smallest_digit_to_make_divisible_by_9_l1366_136645

theorem smallest_digit_to_make_divisible_by_9 : ∃ d : ℕ, d < 10 ∧ (5 + 2 + 8 + d + 4 + 6) % 9 = 0 ∧ ∀ d' : ℕ, d' < d → (5 + 2 + 8 + d' + 4 + 6) % 9 ≠ 0 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_digit_to_make_divisible_by_9_l1366_136645


namespace NUMINAMATH_GPT_emma_still_missing_fraction_l1366_136610

variable (x : ℕ)  -- Total number of coins Emma received 

-- Conditions
def emma_lost_half (x : ℕ) : ℕ := x / 2
def emma_found_four_fifths (lost : ℕ) : ℕ := 4 * lost / 5

-- Question to prove
theorem emma_still_missing_fraction :
  (x - (x / 2 + emma_found_four_fifths (emma_lost_half x))) / x = 1 / 10 := 
by
  sorry

end NUMINAMATH_GPT_emma_still_missing_fraction_l1366_136610


namespace NUMINAMATH_GPT_smallest_number_of_pencils_l1366_136656

theorem smallest_number_of_pencils
  (P : ℕ)
  (h5 : P % 5 = 2)
  (h9 : P % 9 = 2)
  (h11 : P % 11 = 2)
  (hP_gt2 : P > 2) :
  P = 497 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_pencils_l1366_136656


namespace NUMINAMATH_GPT_oranges_to_juice_l1366_136618

theorem oranges_to_juice (oranges: ℕ) (juice: ℕ) (h: oranges = 18 ∧ juice = 27): 
  ∃ x, (juice / oranges) = (9 / x) ∧ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_oranges_to_juice_l1366_136618


namespace NUMINAMATH_GPT_find_x_l1366_136648

theorem find_x (a x : ℝ) (ha : 1 < a) (hx : 0 < x)
  (h : (3 * x)^(Real.log 3 / Real.log a) - (4 * x)^(Real.log 4 / Real.log a) = 0) : 
  x = 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l1366_136648


namespace NUMINAMATH_GPT_problem_l1366_136696

variables {A B C A1 B1 C1 A0 B0 C0 : Type}

-- Define the acute triangle and constructions
axiom acute_triangle (ABC : Type) : Prop
axiom circumcircle (ABC : Type) (A1 B1 C1 : Type) : Prop
axiom extended_angle_bisectors (ABC : Type) (A0 B0 C0 : Type) : Prop

-- Define the points according to the problem statement
axiom intersections_A0 (ABC : Type) (A0 : Type) : Prop
axiom intersections_B0 (ABC : Type) (B0 : Type) : Prop
axiom intersections_C0 (ABC : Type) (C0 : Type) : Prop

-- Define the areas of triangles and hexagon
axiom area_triangle_A0B0C0 (ABC : Type) (A0 B0 C0 : Type) : ℝ
axiom area_hexagon_AC1B_A1CB1 (ABC : Type) (A1 B1 C1 : Type) : ℝ
axiom area_triangle_ABC (ABC : Type) : ℝ

-- Problem: Prove the area relationships
theorem problem
  (ABC: Type)
  (h1 : acute_triangle ABC)
  (h2 : circumcircle ABC A1 B1 C1)
  (h3 : extended_angle_bisectors ABC A0 B0 C0)
  (h4 : intersections_A0 ABC A0)
  (h5 : intersections_B0 ABC B0)
  (h6 : intersections_C0 ABC C0):
  area_triangle_A0B0C0 ABC A0 B0 C0 = 2 * area_hexagon_AC1B_A1CB1 ABC A1 B1 C1 ∧
  area_triangle_A0B0C0 ABC A0 B0 C0 ≥ 4 * area_triangle_ABC ABC :=
sorry

end NUMINAMATH_GPT_problem_l1366_136696


namespace NUMINAMATH_GPT_lines_parallel_iff_l1366_136686

theorem lines_parallel_iff (a : ℝ) : (∀ x y : ℝ, x + 2*a*y - 1 = 0 ∧ (2*a - 1)*x - a*y - 1 = 0 → x = 1 ∧ x = -1 ∨ ∃ (slope : ℝ), slope = - (1 / (2 * a)) ∧ slope = (2 * a - 1) / a) ↔ (a = 0 ∨ a = 1/4) :=
by
  sorry

end NUMINAMATH_GPT_lines_parallel_iff_l1366_136686


namespace NUMINAMATH_GPT_smallest_n_l1366_136617

theorem smallest_n (n : ℕ) (h₁ : ∃ k₁ : ℕ, 5 * n = k₁ ^ 2) (h₂ : ∃ k₂ : ℕ, 4 * n = k₂ ^ 3) : n = 1600 :=
sorry

end NUMINAMATH_GPT_smallest_n_l1366_136617


namespace NUMINAMATH_GPT_monikaTotalSpending_l1366_136688

-- Define the conditions as constants
def mallSpent : ℕ := 250
def movieCost : ℕ := 24
def movieCount : ℕ := 3
def beanCost : ℚ := 1.25
def beanCount : ℕ := 20

-- Define the theorem to prove the total spending
theorem monikaTotalSpending : mallSpent + (movieCost * movieCount) + (beanCost * beanCount) = 347 :=
by
  sorry

end NUMINAMATH_GPT_monikaTotalSpending_l1366_136688


namespace NUMINAMATH_GPT_expected_prize_money_l1366_136639

theorem expected_prize_money :
  let a1 := 1 / 7
  let prob1 := a1
  let prob2 := 2 * a1
  let prob3 := 4 * a1
  let prize1 := 700
  let prize2 := 700 - 140
  let prize3 := 700 - 140 * 2
  let expected_money := prize1 * prob1 + prize2 * prob2 + prize3 * prob3
  expected_money = 500 := 
by
  -- Definitions
  let a1 := 1 / 7
  let prob1 := a1
  let prob2 := 2 * a1
  let prob3 := 4 * a1
  let prize1 := 700
  let prize2 := 700 - 140
  let prize3 := 700 - 140 * 2
  let expected_money := prize1 * prob1 + prize2 * prob2 + prize3 * prob3

  -- Calculate
  sorry -- Proof to show expected_money equals 500

end NUMINAMATH_GPT_expected_prize_money_l1366_136639


namespace NUMINAMATH_GPT_range_of_abs_function_l1366_136637

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 1)

theorem range_of_abs_function : Set.range f = Set.Ici 2 := by
  sorry

end NUMINAMATH_GPT_range_of_abs_function_l1366_136637


namespace NUMINAMATH_GPT_snakes_in_cage_l1366_136669

theorem snakes_in_cage (snakes_hiding : Nat) (snakes_not_hiding : Nat) (total_snakes : Nat) 
  (h : snakes_hiding = 64) (nh : snakes_not_hiding = 31) : 
  total_snakes = snakes_hiding + snakes_not_hiding := by
  sorry

end NUMINAMATH_GPT_snakes_in_cage_l1366_136669


namespace NUMINAMATH_GPT_son_age_l1366_136634

theorem son_age (S F : ℕ) (h1 : F = S + 30) (h2 : F + 2 = 2 * (S + 2)) : S = 28 :=
by
  sorry

end NUMINAMATH_GPT_son_age_l1366_136634


namespace NUMINAMATH_GPT_angle_B_in_triangle_l1366_136603

theorem angle_B_in_triangle (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) :
  B = 3 * Real.pi / 10 :=
sorry

end NUMINAMATH_GPT_angle_B_in_triangle_l1366_136603


namespace NUMINAMATH_GPT_largest_two_digit_number_divisible_by_6_ending_in_4_l1366_136653

theorem largest_two_digit_number_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ n = 84 :=
by
  existsi 84
  sorry

end NUMINAMATH_GPT_largest_two_digit_number_divisible_by_6_ending_in_4_l1366_136653


namespace NUMINAMATH_GPT_minimum_strips_cover_circle_l1366_136615

theorem minimum_strips_cover_circle (l R : ℝ) (hl : l > 0) (hR : R > 0) :
  ∃ (k : ℕ), (k : ℝ) * l ≥ 2 * R ∧ ((k - 1 : ℕ) : ℝ) * l < 2 * R :=
sorry

end NUMINAMATH_GPT_minimum_strips_cover_circle_l1366_136615


namespace NUMINAMATH_GPT_faith_earnings_correct_l1366_136684

variable (pay_per_hour : ℝ) (regular_hours_per_day : ℝ) (work_days_per_week : ℝ) (overtime_hours_per_day : ℝ)
variable (overtime_rate_multiplier : ℝ)

def total_earnings (pay_per_hour : ℝ) (regular_hours_per_day : ℝ) (work_days_per_week : ℝ) 
                   (overtime_hours_per_day : ℝ) (overtime_rate_multiplier : ℝ) : ℝ :=
  let regular_hours := regular_hours_per_day * work_days_per_week
  let overtime_hours := overtime_hours_per_day * work_days_per_week
  let overtime_pay_rate := pay_per_hour * overtime_rate_multiplier
  let regular_earnings := pay_per_hour * regular_hours
  let overtime_earnings := overtime_pay_rate * overtime_hours
  regular_earnings + overtime_earnings

theorem faith_earnings_correct : 
  total_earnings 13.5 8 5 2 1.5 = 742.50 :=
by
  -- This is where the proof would go, but it's omitted as per the instructions
  sorry

end NUMINAMATH_GPT_faith_earnings_correct_l1366_136684


namespace NUMINAMATH_GPT_imaginary_part_of_complex_number_l1366_136628

def imaginary_unit (i : ℂ) : Prop := i * i = -1

def complex_number (z : ℂ) (i : ℂ) : Prop := z = i * (1 - 3 * i)

theorem imaginary_part_of_complex_number (i z : ℂ) (h1 : imaginary_unit i) (h2 : complex_number z i) : z.im = 1 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_complex_number_l1366_136628


namespace NUMINAMATH_GPT_general_formula_for_sequence_l1366_136664

def sequence_terms (a : ℕ → ℚ) : Prop :=
∀ n : ℕ, n > 0 → a n = 1 / (n * (n + 1))

def seq_conditions (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
a 1 = 1 / 2 ∧ (∀ n : ℕ, n > 0 → S n = n^2 * a n)

theorem general_formula_for_sequence :
  ∃ a S : ℕ → ℚ, seq_conditions a S ∧ sequence_terms a := by
  sorry

end NUMINAMATH_GPT_general_formula_for_sequence_l1366_136664


namespace NUMINAMATH_GPT_large_block_volume_correct_l1366_136622

def normal_block_volume (w d l : ℝ) : ℝ := w * d * l

def large_block_volume (w d l : ℝ) : ℝ := (2 * w) * (2 * d) * (3 * l)

theorem large_block_volume_correct (w d l : ℝ) (h : normal_block_volume w d l = 3) :
  large_block_volume w d l = 36 :=
by sorry

end NUMINAMATH_GPT_large_block_volume_correct_l1366_136622


namespace NUMINAMATH_GPT_percentage_decrease_l1366_136689

theorem percentage_decrease (original_price new_price decrease: ℝ) (h₁: original_price = 2400) (h₂: new_price = 1200) (h₃: decrease = original_price - new_price): 
  decrease / original_price * 100 = 50 :=
by
  rw [h₁, h₂] at h₃ -- Update the decrease according to given prices
  sorry -- Left as a placeholder for the actual proof

end NUMINAMATH_GPT_percentage_decrease_l1366_136689


namespace NUMINAMATH_GPT_max_candy_received_l1366_136655

theorem max_candy_received (students : ℕ) (candies : ℕ) (min_candy_per_student : ℕ) 
    (h_students : students = 40) (h_candies : candies = 200) (h_min_candy : min_candy_per_student = 2) :
    ∃ max_candy : ℕ, max_candy = 122 := by
  sorry

end NUMINAMATH_GPT_max_candy_received_l1366_136655


namespace NUMINAMATH_GPT_smaller_angle_at_9_15_l1366_136683

theorem smaller_angle_at_9_15 (h_degree : ℝ) (m_degree : ℝ) (smaller_angle : ℝ) :
  (h_degree = 277.5) → (m_degree = 90) → (smaller_angle = 172.5) :=
by
  sorry

end NUMINAMATH_GPT_smaller_angle_at_9_15_l1366_136683


namespace NUMINAMATH_GPT_find_remainder_l1366_136632

-- Definitions based on given conditions
def dividend := 167
def divisor := 18
def quotient := 9

-- Statement to prove
theorem find_remainder : dividend = (divisor * quotient) + 5 :=
by
  -- Definitions used in the problem
  unfold dividend divisor quotient
  sorry

end NUMINAMATH_GPT_find_remainder_l1366_136632


namespace NUMINAMATH_GPT_calculate_expression_l1366_136675

theorem calculate_expression : (50 - (5020 - 520) + (5020 - (520 - 50))) = 100 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1366_136675


namespace NUMINAMATH_GPT_iggy_total_time_correct_l1366_136631

noncomputable def total_time_iggy_spends : ℕ :=
  let monday_time := 3 * (10 + 1)
  let tuesday_time := 4 * (9 + 1)
  let wednesday_time := 6 * 12
  let thursday_time := 8 * (8 + 2)
  let friday_time := 3 * 10
  monday_time + tuesday_time + wednesday_time + thursday_time + friday_time

theorem iggy_total_time_correct : total_time_iggy_spends = 255 :=
by
  -- sorry at the end indicates the skipping of the actual proof elaboration.
  sorry

end NUMINAMATH_GPT_iggy_total_time_correct_l1366_136631


namespace NUMINAMATH_GPT_number_of_federal_returns_sold_l1366_136606

/-- Given conditions for revenue calculations at the Kwik-e-Tax Center -/
structure TaxCenter where
  price_federal : ℕ
  price_state : ℕ
  price_quarterly : ℕ
  num_state : ℕ
  num_quarterly : ℕ
  total_revenue : ℕ

/-- The specific instance of the TaxCenter for this problem -/
def KwikETaxCenter : TaxCenter :=
{ price_federal := 50,
  price_state := 30,
  price_quarterly := 80,
  num_state := 20,
  num_quarterly := 10,
  total_revenue := 4400 }

/-- Proof statement for the number of federal returns sold -/
theorem number_of_federal_returns_sold (F : ℕ) :
  KwikETaxCenter.price_federal * F + 
  KwikETaxCenter.price_state * KwikETaxCenter.num_state + 
  KwikETaxCenter.price_quarterly * KwikETaxCenter.num_quarterly = 
  KwikETaxCenter.total_revenue → 
  F = 60 :=
by
  intro h
  /- Proof is skipped -/
  sorry

end NUMINAMATH_GPT_number_of_federal_returns_sold_l1366_136606


namespace NUMINAMATH_GPT_regular_vs_diet_sodas_l1366_136657

theorem regular_vs_diet_sodas :
  let regular_cola := 67
  let regular_lemon := 45
  let regular_orange := 23
  let diet_cola := 9
  let diet_lemon := 32
  let diet_orange := 12
  let regular_sodas := regular_cola + regular_lemon + regular_orange
  let diet_sodas := diet_cola + diet_lemon + diet_orange
  regular_sodas - diet_sodas = 82 := sorry

end NUMINAMATH_GPT_regular_vs_diet_sodas_l1366_136657


namespace NUMINAMATH_GPT_solve_for_y_l1366_136652

theorem solve_for_y (x y : ℝ) (h : (x + y)^5 - x^5 + y = 0) : y = 0 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l1366_136652


namespace NUMINAMATH_GPT_inscribed_triangle_area_is_12_l1366_136650

noncomputable def area_of_triangle_in_inscribed_circle 
  (a b c : ℝ) 
  (h_ratio : a = 2 * b ∧ c = 2 * a) 
  (h_radius : ∀ R, R = 4) 
  (h_inscribed : ∃ x, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x ∧ c = 2 * 4) : 
  ℝ := 
1 / 2 * (2 * (4 / 2)) * (3 * (4 / 2))

theorem inscribed_triangle_area_is_12 
  (a b c : ℝ) 
  (h_ratio : a = 2 * b ∧ c = 2 * a) 
  (h_radius : ∀ R, R = 4) 
  (h_inscribed : ∃ x, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x ∧ c = 2 * 4) :
  area_of_triangle_in_inscribed_circle a b c h_ratio h_radius h_inscribed = 12 :=
sorry

end NUMINAMATH_GPT_inscribed_triangle_area_is_12_l1366_136650
