import Mathlib

namespace question1_question2_l310_310240

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) - abs (2 * x - 1)

theorem question1 (x : ℝ) :
  ∀ a, a = 2 → (f x 2 + 3 ≥ 0 ↔ -4 ≤ x ∧ x ≤ 2) := by
sorry

theorem question2 (a : ℝ) :
  (∀ x, 1 ≤ x → x ≤ 3 → f x a ≤ 3) ↔ (-3 ≤ a ∧ a ≤ 5) := by
sorry

end question1_question2_l310_310240


namespace coordinates_of_A_l310_310433

-- Definition of the point A with coordinates (-1, 3)
def point_A : ℝ × ℝ := (-1, 3)

-- Statement that the coordinates of point A with respect to the origin are (-1, 3)
theorem coordinates_of_A : point_A = (-1, 3) := by
  sorry

end coordinates_of_A_l310_310433


namespace smallest_x_for_cubic_1890_l310_310824

theorem smallest_x_for_cubic_1890 (x : ℕ) (N : ℕ) (hx : 1890 * x = N ^ 3) : x = 4900 :=
sorry

end smallest_x_for_cubic_1890_l310_310824


namespace solve_equation_l310_310598

theorem solve_equation (x : ℝ) (hx : 0 ≤ x) : 2021 * (x^2020)^(1/202) - 1 = 2020 * x → x = 1 :=
by
  sorry

end solve_equation_l310_310598


namespace probability_four_squares_form_square_l310_310861

noncomputable def probability_form_square (n k : ℕ) :=
  if (k = 4) ∧ (n = 6) then (1 / 561 : ℚ) else 0

theorem probability_four_squares_form_square :
  probability_form_square 6 4 = (1 / 561 : ℚ) :=
by
  -- Here we would usually include the detailed proof
  -- corresponding to the solution steps from the problem,
  -- but we leave it as sorry for now.
  sorry

end probability_four_squares_form_square_l310_310861


namespace product_gt_one_l310_310157

theorem product_gt_one 
  (m : ℚ) (b : ℚ)
  (hm : m = 3 / 4)
  (hb : b = 5 / 2) :
  m * b > 1 := 
by
  sorry

end product_gt_one_l310_310157


namespace factor_polynomial_l310_310065

theorem factor_polynomial (x y : ℝ) : 
  2*x^2 - x*y - 15*y^2 = (2*x - 5*y) * (x - 3*y) :=
sorry

end factor_polynomial_l310_310065


namespace find_sum_of_abc_l310_310580

theorem find_sum_of_abc
  (a b c x y : ℕ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a^2 + b^2 + c^2 = 2011)
  (h3 : Nat.gcd a (Nat.gcd b c) = x)
  (h4 : Nat.lcm a (Nat.lcm b c) = y)
  (h5 : x + y = 388)
  :
  a + b + c = 61 :=
sorry

end find_sum_of_abc_l310_310580


namespace min_troublemakers_l310_310802

theorem min_troublemakers (students : Finset ℕ) (table : List ℕ) (is_liar : ℕ → Bool) :
  students.card = 29 ∧
  (∀ s ∈ students, 
    (∃ i j : ℕ, 
      table.nth i = some s ∧ table.nth ((i + 1) % 29) = some j ∧
      (∃ k : ℕ, 
        (is_liar s = false → count (/=is_liar k liar) = 0 ∧
        is_liar s = true → count (/=is_liar k liar) = 2))) → 
  (∃! l ∈ students, count (/=is_liar l liar) = 1))) → 
  ∃ t : ℕ, t = 10 :=
sorry

end min_troublemakers_l310_310802


namespace bill_sun_vs_sat_l310_310287

theorem bill_sun_vs_sat (B_Sat B_Sun J_Sun : ℕ) 
  (h1 : B_Sun = 6)
  (h2 : J_Sun = 2 * B_Sun)
  (h3 : B_Sat + B_Sun + J_Sun = 20) : 
  B_Sun - B_Sat = 4 :=
by
  sorry

end bill_sun_vs_sat_l310_310287


namespace floor_square_of_sqrt_50_eq_49_l310_310885

theorem floor_square_of_sqrt_50_eq_49 : (Int.floor (Real.sqrt 50))^2 = 49 := 
by
  sorry

end floor_square_of_sqrt_50_eq_49_l310_310885


namespace alice_wins_probability_is_5_over_6_l310_310381

noncomputable def probability_Alice_wins : ℚ := 
  let total_pairs := 36
  let losing_pairs := 6
  1 - (losing_pairs / total_pairs)

theorem alice_wins_probability_is_5_over_6 : 
  let winning_probability := probability_Alice_wins
  winning_probability = 5 / 6 :=
by
  sorry

end alice_wins_probability_is_5_over_6_l310_310381


namespace P_projection_matrix_P_not_invertible_l310_310281

noncomputable def v : ℝ × ℝ := (4, -1)

noncomputable def norm_v : ℝ := Real.sqrt (4^2 + (-1)^2)

noncomputable def u : ℝ × ℝ := (4 / norm_v, -1 / norm_v)

noncomputable def P : ℝ × ℝ × ℝ × ℝ :=
((4 * 4) / norm_v^2, (4 * -1) / norm_v^2, 
 (-1 * 4) / norm_v^2, (-1 * -1) / norm_v^2)

theorem P_projection_matrix :
  P = (16 / 17, -4 / 17, -4 / 17, 1 / 17) := by
  sorry

theorem P_not_invertible :
  ¬(∃ Q : ℝ × ℝ × ℝ × ℝ, P = Q) := by
  sorry

end P_projection_matrix_P_not_invertible_l310_310281


namespace sum_of_valid_n_l310_310012

theorem sum_of_valid_n :
  (∑ n in { n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13 }, n) = 13 :=
by
  sorry

end sum_of_valid_n_l310_310012


namespace quadratic_solution_l310_310087

theorem quadratic_solution (m n x : ℝ)
  (h1 : (x - m)^2 + n = 0) 
  (h2 : ∃ (a b : ℝ), a ≠ b ∧ (x = a ∨ x = b) ∧ (a - m)^2 + n = 0 ∧ (b - m)^2 + n = 0
    ∧ (a = -1 ∨ a = 3) ∧ (b = -1 ∨ b = 3)) :
  x = -3 ∨ x = 1 :=
by {
  sorry
}

end quadratic_solution_l310_310087


namespace joggers_meet_again_at_correct_time_l310_310047

-- Define the joggers and their lap times
def bob_lap_time := 3
def carol_lap_time := 5
def ted_lap_time := 8

-- Calculate the Least Common Multiple (LCM) of their lap times
def lcm_joggers := Nat.lcm (Nat.lcm bob_lap_time carol_lap_time) ted_lap_time

-- Start time is 9:00 AM
def start_time := 9 * 60  -- in minutes

-- The time (in minutes) we get back together is start_time plus the LCM
def earliest_meeting_time := start_time + lcm_joggers

-- Convert the meeting time to hours and minutes
def hours := earliest_meeting_time / 60
def minutes := earliest_meeting_time % 60

-- Define an expected result
def expected_meeting_hour := 11
def expected_meeting_minute := 0

-- Prove that all joggers will meet again at the correct time
theorem joggers_meet_again_at_correct_time :
  hours = expected_meeting_hour ∧ minutes = expected_meeting_minute :=
by
  -- Here you would provide the proof, but we'll use sorry for brevity
  sorry

end joggers_meet_again_at_correct_time_l310_310047


namespace rope_cutting_impossible_l310_310514

/-- 
Given a rope initially cut into 5 pieces, and then some of these pieces were each cut into 
5 parts, with this process repeated several times, it is not possible for the total 
number of pieces to be exactly 2019.
-/ 
theorem rope_cutting_impossible (n : ℕ) : 5 + 4 * n ≠ 2019 := 
sorry

end rope_cutting_impossible_l310_310514


namespace evaluate_expression_l310_310393

theorem evaluate_expression : (3 / (1 - (2 / 5))) = 5 := by
  sorry

end evaluate_expression_l310_310393


namespace quadratic_real_roots_l310_310725

theorem quadratic_real_roots (k : ℝ) (h1 : k ≠ 0) : (4 + 4 * k) ≥ 0 ↔ k ≥ -1 := 
by 
  sorry

end quadratic_real_roots_l310_310725


namespace cos_value_l310_310237

variable (α : ℝ)

theorem cos_value (h : Real.sin (Real.pi / 6 + α) = 1 / 3) : Real.cos (2 * Real.pi / 3 - 2 * α) = -7 / 9 :=
by
  sorry

end cos_value_l310_310237


namespace population_net_increase_l310_310265

theorem population_net_increase
  (birth_rate : ℕ) (death_rate : ℕ) (T : ℕ)
  (h1 : birth_rate = 7) (h2 : death_rate = 3) (h3 : T = 86400) :
  (birth_rate - death_rate) * (T / 2) = 172800 :=
by
  sorry

end population_net_increase_l310_310265


namespace complex_solution_count_l310_310680

theorem complex_solution_count : 
  ∃ (s : Finset ℂ), (∀ z ∈ s, (z^3 - 8) / (z^2 - 3 * z + 2) = 0) ∧ s.card = 2 := 
by
  sorry

end complex_solution_count_l310_310680


namespace height_of_platform_l310_310817

variable (h l w : ℕ)

-- Define the conditions as hypotheses
def measured_length_first_configuration : Prop := l + h - w = 40
def measured_length_second_configuration : Prop := w + h - l = 34

-- The goal is to prove that the height is 37 inches
theorem height_of_platform
  (h l w : ℕ)
  (config1 : measured_length_first_configuration h l w)
  (config2 : measured_length_second_configuration h l w) : 
  h = 37 := 
sorry

end height_of_platform_l310_310817


namespace platform_length_l310_310028

theorem platform_length
    (train_length : ℕ)
    (time_to_cross_tree : ℕ)
    (speed : ℕ)
    (time_to_pass_platform : ℕ)
    (platform_length : ℕ) :
    train_length = 1200 →
    time_to_cross_tree = 120 →
    speed = train_length / time_to_cross_tree →
    time_to_pass_platform = 150 →
    speed * time_to_pass_platform = train_length + platform_length →
    platform_length = 300 :=
by
  intros h_train_length h_time_to_cross_tree h_speed h_time_to_pass_platform h_pass_platform_eq
  sorry

end platform_length_l310_310028


namespace avg_speed_in_mph_l310_310998

/-- 
Given conditions:
1. The man travels 10,000 feet due north.
2. He travels 6,000 feet due east in 1/4 less time than he took heading north, traveling at 3 miles per minute.
3. He returns to his starting point by traveling south at 1 mile per minute.
4. He travels back west at the same speed as he went east.
We aim to prove that the average speed for the entire trip is 22.71 miles per hour.
-/
theorem avg_speed_in_mph :
  let distance_north_feet := 10000
  let distance_east_feet := 6000
  let speed_east_miles_per_minute := 3
  let speed_south_miles_per_minute := 1
  let feet_per_mile := 5280
  let distance_north_mil := (distance_north_feet / feet_per_mile : ℝ)
  let distance_east_mil := (distance_east_feet / feet_per_mile : ℝ)
  let time_north_min := distance_north_mil / (1 / 3)
  let time_east_min := time_north_min * 0.75
  let time_south_min := distance_north_mil / speed_south_miles_per_minute
  let time_west_min := time_east_min
  let total_time_hr := (time_north_min + time_east_min + time_south_min + time_west_min) / 60
  let total_distance_miles := 2 * (distance_north_mil + distance_east_mil)
  let avg_speed_mph := total_distance_miles / total_time_hr
  avg_speed_mph = 22.71 := by
sorry

end avg_speed_in_mph_l310_310998


namespace birdseed_mix_percentage_l310_310034

theorem birdseed_mix_percentage (x : ℝ) :
  (0.40 * x + 0.65 * (100 - x) = 50) → x = 60 :=
by
  sorry

end birdseed_mix_percentage_l310_310034


namespace basketball_team_count_l310_310670

theorem basketball_team_count :
  (∃ n : ℕ, n = (Nat.choose 13 4) ∧ n = 715) :=
by
  sorry

end basketball_team_count_l310_310670


namespace find_k_l310_310626

theorem find_k (m n k : ℝ) (h1 : m = 2 * n + 3) (h2 : m + 2 = 2 * (n + k) + 3) : k = 1 :=
by
  -- Proof is omitted
  sorry

end find_k_l310_310626


namespace logan_snowfall_total_l310_310588

theorem logan_snowfall_total (wednesday thursday friday : ℝ) :
  wednesday = 0.33 → thursday = 0.33 → friday = 0.22 → wednesday + thursday + friday = 0.88 :=
by
  intros hw ht hf
  rw [hw, ht, hf]
  exact (by norm_num : (0.33 : ℝ) + 0.33 + 0.22 = 0.88)

end logan_snowfall_total_l310_310588


namespace rectangle_area_from_square_l310_310648

theorem rectangle_area_from_square {s a : ℝ} (h1 : s^2 = 36) (h2 : a = 3 * s) :
    s * a = 108 :=
by
  -- The proof goes here
  sorry

end rectangle_area_from_square_l310_310648


namespace total_cost_shorts_tshirt_boots_shinguards_l310_310311

variable (x : ℝ)

-- Definitions provided in the problem statement.
def cost_shorts : ℝ := x
def cost_shorts_and_tshirt : ℝ := 2 * x
def cost_shorts_and_boots : ℝ := 5 * x
def cost_shorts_and_shinguards : ℝ := 3 * x

-- The proof goal to verify:
theorem total_cost_shorts_tshirt_boots_shinguards : 
  (cost_shorts x + (cost_shorts_and_tshirt x - cost_shorts x) + 
   (cost_shorts_and_boots x - cost_shorts x) + 
   (cost_shorts_and_shinguards x - cost_shorts x)) = 8 * x := by 
  sorry

end total_cost_shorts_tshirt_boots_shinguards_l310_310311


namespace fraction_sum_is_one_l310_310444

theorem fraction_sum_is_one
    (a b c d w x y z : ℝ)
    (h1 : 17 * w + b * x + c * y + d * z = 0)
    (h2 : a * w + 29 * x + c * y + d * z = 0)
    (h3 : a * w + b * x + 37 * y + d * z = 0)
    (h4 : a * w + b * x + c * y + 53 * z = 0)
    (a_ne_17 : a ≠ 17)
    (b_ne_29 : b ≠ 29)
    (c_ne_37 : c ≠ 37)
    (wxyz_nonzero : w ≠ 0 ∨ x ≠ 0 ∨ y ≠ 0) :
    (a / (a - 17)) + (b / (b - 29)) + (c / (c - 37)) + (d / (d - 53)) = 1 := 
sorry

end fraction_sum_is_one_l310_310444


namespace minimum_a2_plus_4b2_l310_310400

theorem minimum_a2_plus_4b2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 / a + 1 / b = 1) : 
  a^2 + 4 * b^2 ≥ 32 :=
sorry

end minimum_a2_plus_4b2_l310_310400


namespace find_f3_l310_310403

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f3 (h1 : ∀ x : ℝ, f (x + 1) = f (-x - 1))
                (h2 : ∀ x : ℝ, f (2 - x) = -f x) :
  f 3 = 0 := 
sorry

end find_f3_l310_310403


namespace angles_equal_l310_310549

theorem angles_equal (α θ γ : Real) (hα : 0 < α ∧ α < π / 2) (hθ : 0 < θ ∧ θ < π / 2) (hγ : 0 < γ ∧ γ < π / 2)
  (h : Real.sin (α + γ) * Real.tan α = Real.sin (θ + γ) * Real.tan θ) : α = θ :=
by
  sorry

end angles_equal_l310_310549


namespace product_modulo_seven_l310_310215

/-- 2021 is congruent to 6 modulo 7 -/
def h1 : 2021 % 7 = 6 := rfl

/-- 2022 is congruent to 0 modulo 7 -/
def h2 : 2022 % 7 = 0 := rfl

/-- 2023 is congruent to 1 modulo 7 -/
def h3 : 2023 % 7 = 1 := rfl

/-- 2024 is congruent to 2 modulo 7 -/
def h4 : 2024 % 7 = 2 := rfl

/-- The product 2021 * 2022 * 2023 * 2024 is congruent to 0 modulo 7 -/
theorem product_modulo_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
  by sorry

end product_modulo_seven_l310_310215


namespace matrix_determinant_is_zero_l310_310388

variable (a b : ℝ)

theorem matrix_determinant_is_zero :
  Matrix.det ![
    ![1, Real.cos (a - b), Real.cos a], 
    ![Real.cos (a - b), 1, Real.cos b], 
    ![Real.cos a, Real.cos b, 1]
  ] = 0 := 
sorry

end matrix_determinant_is_zero_l310_310388


namespace min_troublemakers_l310_310790

theorem min_troublemakers (n : ℕ) (students : ℕ → Prop) 
  (h : n = 29)
  (condition1 : ∀ i, students i → (students ((i - 1) % n) ↔ ¬ students ((i + 1) % n)))
  (condition2 : ∀ i, ¬ students i → (students ((i - 1) % n) ∧ students ((i + 1) % n)))
  : ∃ L : ℕ, (L ≤ 29 ∧ L ≥ 10) :=
by sorry

end min_troublemakers_l310_310790


namespace minimum_possible_value_of_Box_l310_310257

theorem minimum_possible_value_of_Box :
  ∃ a b : ℤ, a ≠ b ∧ a * b = 45 ∧ 
    (∀ c d : ℤ, c * d = 45 → c^2 + d^2 ≥ 106) ∧ a^2 + b^2 = 106 :=
by
  sorry

end minimum_possible_value_of_Box_l310_310257


namespace circle_arrangement_l310_310267

theorem circle_arrangement (n : ℕ) (h : n = 100) :
  ∃ k l : ℕ, k! * 2^l = 49! * 2^49 :=
by
  use (50 - 1)! -- 49!
  use (50 - 1) -- 49
  sorry

end circle_arrangement_l310_310267


namespace frog_problem_l310_310352

theorem frog_problem 
  (N : ℕ) 
  (h1 : N < 50) 
  (h2 : N % 2 = 1) 
  (h3 : N % 3 = 1) 
  (h4 : N % 4 = 1) 
  (h5 : N % 5 = 0) : 
  N = 25 := 
  sorry

end frog_problem_l310_310352


namespace sqrt_floor_squared_50_l310_310869

noncomputable def sqrt_floor_squared (n : ℕ) : ℕ :=
  (Int.floor (Real.sqrt n))^2

theorem sqrt_floor_squared_50 : sqrt_floor_squared 50 = 49 := 
  by
  sorry

end sqrt_floor_squared_50_l310_310869


namespace base_number_is_two_l310_310942

theorem base_number_is_two (x : ℝ) (n : ℕ) (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^18) (h2 : n = 17) : x = 2 :=
by sorry

end base_number_is_two_l310_310942


namespace hidden_prime_average_correct_l310_310228

noncomputable def hidden_prime_average : ℚ :=
  (13 + 17 + 59) / 3

theorem hidden_prime_average_correct :
  hidden_prime_average = 29.6 :=
by
  sorry

end hidden_prime_average_correct_l310_310228


namespace shaded_rectangle_area_l310_310119

-- Define the square PQRS and its properties
def is_square (s : ℝ) := ∃ (PQ QR RS SP : ℝ), PQ = s ∧ QR = s ∧ RS = s ∧ SP = s

-- Define the conditions for the side lengths and segments
def side_length := 11
def top_left_height := 6
def top_right_height := 2
def width_bottom_right := 11 - 10
def width_top_right := 8

-- Calculate necessary dimensions
def shaded_rectangle_height := top_left_height - top_right_height
def shaded_rectangle_width := width_top_right - width_bottom_right

-- Proof statement
theorem shaded_rectangle_area (s : ℝ) (h1 : is_square s)
  (h2 : s = side_length)
  (h3 : shaded_rectangle_height = 4)
  (h4 : shaded_rectangle_width = 7) :
  4 * 7 = 28 := by
  sorry

end shaded_rectangle_area_l310_310119


namespace slope_range_PA2_l310_310399

-- Define the given conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

def A1 : ℝ × ℝ := (-2, 0)
def A2 : ℝ × ℝ := (2, 0)
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.fst P.snd

-- Define the range of the slope of line PA1
def slope_range_PA1 (k_PA1 : ℝ) : Prop := -2 ≤ k_PA1 ∧ k_PA1 ≤ -1

-- Main theorem
theorem slope_range_PA2 (x0 y0 k_PA1 k_PA2 : ℝ) (h1 : on_ellipse (x0, y0)) (h2 : slope_range_PA1 k_PA1) :
  k_PA1 = (y0 / (x0 + 2)) →
  k_PA2 = (y0 / (x0 - 2)) →
  - (3 / 4) = k_PA1 * k_PA2 →
  (3 / 8) ≤ k_PA2 ∧ k_PA2 ≤ (3 / 4) :=
by
  sorry

end slope_range_PA2_l310_310399


namespace find_a_l310_310412

noncomputable def A : Set ℝ := {1, 2, 3}
noncomputable def B (a : ℝ) : Set ℝ := { x | x^2 - (a + 1) * x + a = 0 }

theorem find_a (a : ℝ) (h : A ∪ B a = A) : a = 1 ∨ a = 2 ∨ a = 3 :=
by
  sorry

end find_a_l310_310412


namespace sqrt_floor_squared_50_l310_310868

noncomputable def sqrt_floor_squared (n : ℕ) : ℕ :=
  (Int.floor (Real.sqrt n))^2

theorem sqrt_floor_squared_50 : sqrt_floor_squared 50 = 49 := 
  by
  sorry

end sqrt_floor_squared_50_l310_310868


namespace value_of_a_l310_310719

noncomputable def A : Set ℝ := { x | abs x = 1 }
def B (a : ℝ) : Set ℝ := { x | a * x = 1 }
def is_superset (A B : Set ℝ) : Prop := ∀ x, x ∈ B → x ∈ A

theorem value_of_a (a : ℝ) (h : is_superset A (B a)) : a = 1 ∨ a = 0 ∨ a = -1 :=
  sorry

end value_of_a_l310_310719


namespace min_liars_needed_l310_310800

namespace Problem

/-
  We introduce our problem using Lean definitions and statements.
-/

def students_sitting_at_table : ℕ := 29

structure Student :=
(is_liar : Bool)

def student_truth_statement (s: ℕ) (next1: Student) (next2: Student) : Bool :=
  if s then (next1.is_liar && !next2.is_liar) || (!next1.is_liar && next2.is_liar)
  else next1.is_liar && next2.is_liar

def minimum_liars_class (sitting_students: ℕ) : ℕ :=
  sitting_students / 3 + (if sitting_students % 3 = 0 then 0 else 1)

theorem min_liars_needed (students : List Student) (H_total : students.length = students_sitting_at_table)
  (H_truth_teller_claim : ∀ (i : ℕ), i < students.length → student_truth_statement i (students.get ⟨(i+1) % students.length, sorry⟩) (students.get ⟨(i+2) % students.length, sorry⟩)) :
  (students.count (λ s => s.is_liar)) ≥ minimum_liars_class students_sitting_at_table :=
sorry

end Problem

end min_liars_needed_l310_310800


namespace remainder_when_divided_by_20_l310_310750

theorem remainder_when_divided_by_20
  (a b : ℤ) 
  (h1 : a % 60 = 49)
  (h2 : b % 40 = 29) :
  (a + b) % 20 = 18 :=
by
  sorry

end remainder_when_divided_by_20_l310_310750


namespace michael_has_16_blocks_l310_310965

-- Define the conditions
def number_of_boxes : ℕ := 8
def blocks_per_box : ℕ := 2

-- Define the expected total number of blocks
def total_blocks : ℕ := 16

-- State the theorem
theorem michael_has_16_blocks (n_boxes blocks_per_b : ℕ) :
  n_boxes = number_of_boxes → 
  blocks_per_b = blocks_per_box → 
  n_boxes * blocks_per_b = total_blocks :=
by intros h1 h2; rw [h1, h2]; sorry

end michael_has_16_blocks_l310_310965


namespace expression_undefined_iff_l310_310906

theorem expression_undefined_iff (a : ℝ) : (a^2 - 9 = 0) ↔ (a = 3 ∨ a = -3) :=
sorry

end expression_undefined_iff_l310_310906


namespace mother_age_twice_xiaoming_in_18_years_l310_310831

-- Definitions based on conditions
def xiaoming_age_now : ℕ := 6
def mother_age_now : ℕ := 30

theorem mother_age_twice_xiaoming_in_18_years : 
    ∀ (n : ℕ), xiaoming_age_now + n = 24 → mother_age_now + n = 2 * (xiaoming_age_now + n) → n = 18 :=
by
  intro n hn hm
  sorry

end mother_age_twice_xiaoming_in_18_years_l310_310831


namespace hexagon_label_count_l310_310463

def hexagon_label (s : Finset ℕ) (a b c d e f g : ℕ) : Prop :=
  s = Finset.range 8 ∧ 
  (a ∈ s) ∧ (b ∈ s) ∧ (c ∈ s) ∧ (d ∈ s) ∧ (e ∈ s) ∧ (f ∈ s) ∧ (g ∈ s) ∧
  a + b + c + d + e + f + g = 28 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ 
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g ∧
  a + g + d = b + g + e ∧ b + g + e = c + g + f

theorem hexagon_label_count : ∃ s a b c d e f g, hexagon_label s a b c d e f g ∧ 
  (s.card = 8) ∧ (a + g + d = 10) ∧ (b + g + e = 10) ∧ (c + g + f = 10) ∧ 
  144 = 3 * 48 :=
sorry

end hexagon_label_count_l310_310463


namespace four_distinct_real_roots_l310_310078

theorem four_distinct_real_roots (m : ℝ) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = x^2 - 4 * |x| + 5 - m) ∧ ∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0) ↔ (1 < m ∧ m < 5) :=
by
  sorry

end four_distinct_real_roots_l310_310078


namespace athlete_runs_entire_track_in_44_seconds_l310_310781

noncomputable def time_to_complete_track (flags : ℕ) (time_to_4th_flag : ℕ) : ℕ :=
  let distances_between_flags := flags - 1
  let distances_to_4th_flag := 4 - 1
  let time_per_distance := time_to_4th_flag / distances_to_4th_flag
  distances_between_flags * time_per_distance

theorem athlete_runs_entire_track_in_44_seconds :
  time_to_complete_track 12 12 = 44 :=
by
  sorry

end athlete_runs_entire_track_in_44_seconds_l310_310781


namespace tom_initial_money_l310_310988

theorem tom_initial_money (spent_on_game : ℕ) (toy_cost : ℕ) (number_of_toys : ℕ)
    (total_spent : ℕ) (h1 : spent_on_game = 49) (h2 : toy_cost = 4)
    (h3 : number_of_toys = 2) (h4 : total_spent = spent_on_game + number_of_toys * toy_cost) :
  total_spent = 57 := by
  sorry

end tom_initial_money_l310_310988


namespace calculate_actual_distance_l310_310142

-- Definitions corresponding to the conditions
def map_scale : ℕ := 6000000
def map_distance_cm : ℕ := 5

-- The theorem statement corresponding to the proof problem
theorem calculate_actual_distance :
  (map_distance_cm * map_scale / 100000) = 300 := 
by
  sorry

end calculate_actual_distance_l310_310142


namespace ways_to_place_books_in_bins_l310_310560

theorem ways_to_place_books_in_bins :
  ∃ (S: ℕ → ℕ → ℕ), S 5 3 = 25 :=
by
  use fun (n k : ℕ) => Stirling.second_kind n k
  simp [Stirling.second_kind]
  sorry

end ways_to_place_books_in_bins_l310_310560


namespace triangle_sides_from_rhombus_l310_310513

variable (m p q : ℝ)

def is_triangle_side_lengths (BC AC AB : ℝ) :=
  (BC = p + q) ∧
  (AC = m * (p + q) / p) ∧
  (AB = m * (p + q) / q)

theorem triangle_sides_from_rhombus :
  ∃ BC AC AB : ℝ, is_triangle_side_lengths m p q BC AC AB :=
by
  use p + q
  use m * (p + q) / p
  use m * (p + q) / q
  sorry

end triangle_sides_from_rhombus_l310_310513


namespace geometric_sequence_divisibility_l310_310453

theorem geometric_sequence_divisibility 
  (a1 : ℚ) (h1 : a1 = 1 / 2) 
  (a2 : ℚ) (h2 : a2 = 10) 
  (n : ℕ) :
  ∃ (n : ℕ), a_n = (a1 * 20^(n - 1)) ∧ (n ≥ 4) ∧ (5000 ∣ a_n) :=
by
  sorry

end geometric_sequence_divisibility_l310_310453


namespace min_value_g_range_of_m_l310_310402

section
variable (x : ℝ)
noncomputable def g (x : ℝ) := Real.exp x - x

theorem min_value_g :
  (∀ x : ℝ, g x ≥ g 0) ∧ g 0 = 1 := 
by 
  sorry

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (2 * x - m) / g x > x) → m < Real.log 2 ^ 2 := 
by 
  sorry
end

end min_value_g_range_of_m_l310_310402


namespace smallest_perfect_square_divisible_by_4_and_5_l310_310617

theorem smallest_perfect_square_divisible_by_4_and_5 : 
  ∃ n : ℕ, n > 0 ∧ n ∣ 4 ∧ n ∣ 5 ∧ is_square n ∧ 
  ∀ m : ℕ, (m > 0 ∧ m ∣ 4 ∧ m ∣ 5 ∧ is_square m) → n ≤ m :=
sorry

end smallest_perfect_square_divisible_by_4_and_5_l310_310617


namespace quadratic_real_roots_implies_k_range_l310_310723

theorem quadratic_real_roots_implies_k_range (k : ℝ) 
  (h : ∃ x : ℝ, k * x^2 + 2 * x - 1 = 0)
  (hk : k ≠ 0) : k ≥ -1 ∧ k ≠ 0 :=
sorry

end quadratic_real_roots_implies_k_range_l310_310723


namespace A_eq_D_l310_310283

def A := {θ : ℝ | 0 < θ ∧ θ < 90}
def D := {θ : ℝ | 0 < θ ∧ θ < 90}

theorem A_eq_D : A = D :=
by
  sorry

end A_eq_D_l310_310283


namespace eval_floor_sqrt_50_square_l310_310878

theorem eval_floor_sqrt_50_square:
    (int.floor (real.sqrt 50))^2 = 49 :=
by
  have h1 : real.sqrt 49 < real.sqrt 50 := by norm_num [real.sqrt]
  have h2 : real.sqrt 50 < real.sqrt 64 := by norm_num [real.sqrt]
  have floor_sqrt_50 : int.floor (real.sqrt 50) = 7 :=
    by linarith [h1, h2]
  rw [floor_sqrt_50]
  norm_num

end eval_floor_sqrt_50_square_l310_310878


namespace sin_6_cos_6_theta_proof_l310_310126

noncomputable def sin_6_cos_6_theta (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) : ℝ :=
  Real.sin θ ^ 6 + Real.cos θ ^ 6

theorem sin_6_cos_6_theta_proof (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) : 
  sin_6_cos_6_theta θ h = 19 / 64 :=
by
  sorry

end sin_6_cos_6_theta_proof_l310_310126


namespace total_paintable_area_l310_310139

-- Define the dimensions of a bedroom
def bedroom_length : ℕ := 10
def bedroom_width : ℕ := 12
def bedroom_height : ℕ := 9

-- Define the non-paintable area per bedroom
def non_paintable_area_per_bedroom : ℕ := 74

-- Number of bedrooms
def number_of_bedrooms : ℕ := 4

-- The total paintable area that we need to prove
theorem total_paintable_area : 
  4 * (2 * (bedroom_length * bedroom_height) + 2 * (bedroom_width * bedroom_height) - non_paintable_area_per_bedroom) = 1288 := 
by
  sorry

end total_paintable_area_l310_310139


namespace polynomial_evaluation_l310_310420

theorem polynomial_evaluation (x : ℤ) (h : x = 2) : 3 * x^2 + 5 * x - 2 = 20 := by
  sorry

end polynomial_evaluation_l310_310420


namespace find_number_l310_310335

theorem find_number (x : ℤ) (h : x - 27 = 49) : x = 76 := by
  sorry

end find_number_l310_310335


namespace nested_expression_evaluation_l310_310204

theorem nested_expression_evaluation : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 :=
by
  sorry

end nested_expression_evaluation_l310_310204


namespace remainder_problem_l310_310117

theorem remainder_problem (n m q1 q2 : ℤ) (h1 : n = 11 * q1 + 1) (h2 : m = 17 * q2 + 3) :
  ∃ r : ℤ, (r = (5 * n + 3 * m) % 11) ∧ (r = (7 * q2 + 3) % 11) :=
by
  sorry

end remainder_problem_l310_310117


namespace square_window_side_length_is_24_l310_310246

noncomputable def side_length_square_window
  (num_panes_per_row : ℕ) (pane_height_ratio : ℝ) (border_width : ℝ) (x : ℝ) : ℝ :=
  num_panes_per_row * x + (num_panes_per_row + 1) * border_width

theorem square_window_side_length_is_24
  (num_panes_per_row : ℕ)
  (pane_height_ratio : ℝ)
  (border_width : ℝ) 
  (pane_width : ℝ)
  (pane_height : ℝ)
  (window_side_length : ℝ) : 
  (num_panes_per_row = 3) →
  (pane_height_ratio = 3) →
  (border_width = 3) →
  (pane_height = pane_height_ratio * pane_width) →
  (window_side_length = side_length_square_window num_panes_per_row pane_height_ratio border_width pane_width) →
  (window_side_length = 24) :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end square_window_side_length_is_24_l310_310246


namespace train_passes_man_in_15_seconds_l310_310041

theorem train_passes_man_in_15_seconds
  (length_of_train : ℝ)
  (speed_of_train : ℝ)
  (speed_of_man : ℝ)
  (direction_opposite : Bool)
  (h1 : length_of_train = 275)
  (h2 : speed_of_train = 60)
  (h3 : speed_of_man = 6)
  (h4 : direction_opposite = true) : 
  ∃ t : ℝ, t = 15 :=
by
  sorry

end train_passes_man_in_15_seconds_l310_310041


namespace madeline_hours_left_over_l310_310286

theorem madeline_hours_left_over :
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleep_hours_per_day := 8
  let sleep_hours_per_week := sleep_hours_per_day * 7
  let part_time_hours := 20
  let total_hours_in_week := 168
  total_hours_in_week - (class_hours + homework_hours_per_week + sleep_hours_per_week + part_time_hours) = 46 :=
by
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleep_hours_per_day := 8
  let sleep_hours_per_week := sleep_hours_per_day * 7
  let part_time_hours := 20
  let total_hours_in_week := 168
  calc
    total_hours_in_week - (class_hours + homework_hours_per_week + sleep_hours_per_week + part_time_hours)
        = 168 - (18 + 4 * 7 + 8 * 7 + 20) : by rfl
    ... = 168 - (18 + 28 + 56 + 20) : by rfl
    ... = 168 - 122 : by rfl
    ... = 46 : by rfl

end madeline_hours_left_over_l310_310286


namespace value_of_expression_l310_310780

theorem value_of_expression : (5^2 - 4^2 + 3^2) = 18 := 
by
  sorry

end value_of_expression_l310_310780


namespace undefined_denominator_values_l310_310901

theorem undefined_denominator_values (a : ℝ) : a = 3 ∨ a = -3 ↔ ∃ b : ℝ, (a - b) * (a + b) = 0 := by
  sorry

end undefined_denominator_values_l310_310901


namespace fair_attendance_l310_310000

theorem fair_attendance :
  let this_year := 600
  let next_year := 2 * this_year
  let total_people := 2800
  let last_year := total_people - this_year - next_year
  (1200 - last_year = 200) ∧ (last_year = 1000) := by
  sorry

end fair_attendance_l310_310000


namespace speed_of_j_l310_310627

theorem speed_of_j (j p : ℝ) 
  (h_faster : j > p)
  (h_distance_j : 24 / j = 24 / j)
  (h_distance_p : 24 / p = 24 / p)
  (h_sum_speeds : j + p = 7)
  (h_sum_times : 24 / j + 24 / p = 14) : j = 4 := 
sorry

end speed_of_j_l310_310627


namespace rectangle_area_l310_310645

theorem rectangle_area (a w l : ℝ) (h_square_area : a = 36) 
    (h_rect_width : w * w = a) 
    (h_rect_length : l = 3 * w) : w * l = 108 := 
sorry

end rectangle_area_l310_310645


namespace mod_2021_2022_2023_2024_eq_zero_mod_7_l310_310213

theorem mod_2021_2022_2023_2024_eq_zero_mod_7 :
  (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end mod_2021_2022_2023_2024_eq_zero_mod_7_l310_310213


namespace chebyshev_birth_year_l310_310271

theorem chebyshev_birth_year :
  ∃ (a b : ℕ),
  a > b ∧ 
  a + b = 3 ∧ 
  (1821 = 1800 + 10 * a + 1 * b) ∧
  (1821 + 73) < 1900 :=
by sorry

end chebyshev_birth_year_l310_310271


namespace calculate_power_expr_l310_310668

theorem calculate_power_expr :
  let a := (-8 : ℝ)
  let b := (0.125 : ℝ)
  a^2023 * b^2024 = -0.125 :=
by
  sorry

end calculate_power_expr_l310_310668


namespace unique_positive_integers_exists_l310_310178

theorem unique_positive_integers_exists (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) : 
  ∃! m n : ℕ, m^2 = n * (n + p) ∧ m = (p^2 - 1) / 2 ∧ n = (p - 1)^2 / 4 := by
  sorry

end unique_positive_integers_exists_l310_310178


namespace find_M_l310_310109

theorem find_M :
  ∃ (M : ℕ), 1001 + 1003 + 1005 + 1007 + 1009 = 5100 - M ∧ M = 75 :=
by
  sorry

end find_M_l310_310109


namespace perpendicular_planes_l310_310708

-- Definitions for lines and planes and their relationships
variable {a b : Line}
variable {α β : Plane}

-- Given conditions for the problem
axiom line_perpendicular (l1 l2 : Line) : Prop -- l1 ⊥ l2
axiom line_parallel (l1 l2 : Line) : Prop -- l1 ∥ l2
axiom line_plane_perpendicular (l : Line) (p : Plane) : Prop -- l ⊥ p
axiom line_plane_parallel (l : Line) (p : Plane) : Prop -- l ∥ p
axiom plane_perpendicular (p1 p2 : Plane) : Prop -- p1 ⊥ p2

-- Problem statement
theorem perpendicular_planes (h1 : line_perpendicular a b)
                            (h2 : line_plane_perpendicular a α)
                            (h3 : line_plane_perpendicular b β) :
                            plane_perpendicular α β :=
sorry

end perpendicular_planes_l310_310708


namespace find_coefficients_l310_310704

theorem find_coefficients
  (a b c : ℝ)
  (hA : ∀ x : ℝ, (x = -3 ∨ x = 4) ↔ (x^2 + a * x - 12 = 0))
  (hB : ∀ x : ℝ, (x = -3 ∨ x = 1) ↔ (x^2 + b * x + c = 0))
  (hAnotB : ¬ (∀ x, (x^2 + a * x - 12 = 0) ↔ (x^2 + b * x + c = 0)))
  (hA_inter_B : ∀ x, x = -3 ↔ (x^2 + a * x - 12 = 0) ∧ (x^2 + b * x + c = 0))
  (hA_union_B : ∀ x, (x = -3 ∨ x = 1 ∨ x = 4) ↔ (x^2 + a * x - 12 = 0) ∨ (x^2 + b * x + c = 0)):
  a = -1 ∧ b = 2 ∧ c = -3 :=
sorry

end find_coefficients_l310_310704


namespace marathon_distance_l310_310354

theorem marathon_distance (d_1 : ℕ) (n : ℕ) (h1 : d_1 = 3) (h2 : n = 5): 
  (2 ^ (n - 1)) * d_1 = 48 :=
by
  sorry

end marathon_distance_l310_310354


namespace product_of_last_two_digits_div_by_6_and_sum_15_l310_310258

theorem product_of_last_two_digits_div_by_6_and_sum_15
  (n : ℕ)
  (h1 : n % 6 = 0)
  (A B : ℕ)
  (h2 : n % 100 = 10 * A + B)
  (h3 : A + B = 15)
  (h4 : B % 2 = 0) : 
  A * B = 54 := 
sorry

end product_of_last_two_digits_div_by_6_and_sum_15_l310_310258


namespace Madeline_hours_left_over_l310_310285

theorem Madeline_hours_left_over :
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleeping_hours_per_day := 8
  let sleeping_hours_per_week := sleeping_hours_per_day * 7
  let work_hours := 20
  let total_busy_hours := class_hours + homework_hours_per_week + sleeping_hours_per_week + work_hours
  let total_hours_per_week := 24 * 7
  total_hours_per_week - total_busy_hours = 46 :=
by
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleeping_hours_per_day := 8
  let sleeping_hours_per_week := sleeping_hours_per_day * 7
  let work_hours := 20
  let total_busy_hours := class_hours + homework_hours_per_week + sleeping_hours_per_week + work_hours
  let total_hours_per_week := 24 * 7
  have : total_hours_per_week - total_busy_hours = 168 - 122 := by rfl
  have : 168 - 122 = 46 := by rfl
  exact this

end Madeline_hours_left_over_l310_310285


namespace perfect_squares_100_to_400_l310_310099

theorem perfect_squares_100_to_400 :
  {n : ℕ | 100 ≤ n^2 ∧ n^2 ≤ 400}.card = 11 :=
by {
  sorry
}

end perfect_squares_100_to_400_l310_310099


namespace sqrt_prime_irrational_l310_310447

theorem sqrt_prime_irrational (p : ℕ) (hp : Nat.Prime p) : Irrational (Real.sqrt p) :=
by
  sorry

end sqrt_prime_irrational_l310_310447


namespace solve_equation_l310_310350

theorem solve_equation (x : ℝ) : 4 * (x - 1) ^ 2 = 9 ↔ x = 5 / 2 ∨ x = -1 / 2 := 
by 
  sorry

end solve_equation_l310_310350


namespace probability_of_interval_l310_310338

-- Define the random variable ξ and its probability distribution P(ξ = k)
variables (ξ : ℕ → ℝ) (P : ℕ → ℝ)

-- Define a constant a
noncomputable def a : ℝ := 5/4

-- Given conditions
axiom condition1 : ∀ k, k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 → P k = a / (k * (k + 1))
axiom condition2 : P 1 + P 2 + P 3 + P 4 = 1

-- Statement to prove
theorem probability_of_interval : P 1 + P 2 = 5/6 :=
by sorry

end probability_of_interval_l310_310338


namespace number_of_students_l310_310712

theorem number_of_students (groups : ℕ) (students_per_group : ℕ) (minutes_per_student : ℕ) (minutes_per_group : ℕ) :
    groups = 3 →
    minutes_per_student = 4 →
    minutes_per_group = 24 →
    minutes_per_group = students_per_group * minutes_per_student →
    18 = groups * students_per_group :=
by
  intros h_groups h_minutes_per_student h_minutes_per_group h_relation
  sorry

end number_of_students_l310_310712


namespace system_of_equations_solution_l310_310973

theorem system_of_equations_solution (x y z : ℤ) :
  x^2 - 9 * y^2 - z^2 = 0 ∧ z = x - 3 * y ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (∃ k : ℤ, x = 3 * k ∧ y = k ∧ z = 0) := 
by
  sorry

end system_of_equations_solution_l310_310973


namespace karlson_word_count_l310_310743

def single_word_count : Nat := 9
def ten_to_nineteen_count : Nat := 10
def two_word_count (num_tens_units : Nat) : Nat := 2 * num_tens_units

def count_words_1_to_99 : Nat :=
  let single_word := single_word_count + ten_to_nineteen_count
  let two_word := two_word_count (99 - (single_word_count + ten_to_nineteen_count))
  single_word + two_word

def prefix_hundred (count_1_to_99 : Nat) : Nat := 9 * count_1_to_99
def extra_prefix (num_two_word_transformed : Nat) : Nat := 9 * num_two_word_transformed

def total_words : Nat :=
  let first_99 := count_words_1_to_99
  let nine_hundreds := prefix_hundred count_words_1_to_99 + extra_prefix 72
  first_99 + nine_hundreds + 37

theorem karlson_word_count : total_words = 2611 :=
  by
    sorry

end karlson_word_count_l310_310743


namespace stapler_machines_l310_310033

theorem stapler_machines (x : ℝ) :
  (∃ (x : ℝ), x > 0) ∧
  ((∀ r1 r2 : ℝ, (r1 = 800 / 6) → (r2 = 800 / x) → (r1 + r2 = 800 / 3)) ↔
    (1 / 6 + 1 / x = 1 / 3)) :=
by sorry

end stapler_machines_l310_310033


namespace zero_not_in_range_of_g_l310_310129

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈1 / (x + 3)⌉
  else ⌊1 / (x + 3)⌋

theorem zero_not_in_range_of_g : ∀ x : ℝ, x ≠ -3 → g x ≠ 0 := by
  sorry

end zero_not_in_range_of_g_l310_310129


namespace distance_between_parallel_lines_l310_310489

theorem distance_between_parallel_lines 
  (r : ℝ) (d : ℝ) 
  (h1 : 3 * (2 * r^2) = 722 + (19 / 4) * d^2) 
  (h2 : 3 * (2 * r^2) = 578 + (153 / 4) * d^2) : 
  d = 6 :=
by
  sorry

end distance_between_parallel_lines_l310_310489


namespace num_perfect_squares_l310_310101

theorem num_perfect_squares (a b : ℤ) (h₁ : a = 100) (h₂ : b = 400) : 
  ∃ n : ℕ, (100 < n^2) ∧ (n^2 < 400) ∧ (n = 9) :=
by
  sorry

end num_perfect_squares_l310_310101


namespace smallest_positive_t_l310_310683

theorem smallest_positive_t (x_1 x_2 x_3 x_4 x_5 t : ℝ) :
  (x_1 + x_3 = 2 * t * x_2) →
  (x_2 + x_4 = 2 * t * x_3) →
  (x_3 + x_5 = 2 * t * x_4) →
  (0 ≤ x_1) →
  (0 ≤ x_2) →
  (0 ≤ x_3) →
  (0 ≤ x_4) →
  (0 ≤ x_5) →
  (x_1 ≠ 0 ∨ x_2 ≠ 0 ∨ x_3 ≠ 0 ∨ x_4 ≠ 0 ∨ x_5 ≠ 0) →
  t = 1 / Real.sqrt 2 → 
  ∃ t, (0 < t) ∧ (x_1 + x_3 = 2 * t * x_2) ∧ (x_2 + x_4 = 2 * t * x_3) ∧ (x_3 + x_5 = 2 * t * x_4)
:=
sorry

end smallest_positive_t_l310_310683


namespace least_positive_three_digit_multiple_of_9_l310_310821

   theorem least_positive_three_digit_multiple_of_9 : ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ 9 ∣ n ∧ n = 108 :=
   by
     sorry
   
end least_positive_three_digit_multiple_of_9_l310_310821


namespace position_of_99_l310_310664

-- Define a function that describes the position of an odd number in the 5-column table.
def position_in_columns (n : ℕ) : ℕ := sorry  -- position in columns is defined by some rule

-- Now, state the theorem regarding the position of 99.
theorem position_of_99 : position_in_columns 99 = 3 := 
by 
  sorry  -- Proof goes here

end position_of_99_l310_310664


namespace hyperbola_solution_l310_310927

noncomputable def hyperbola_eq (x y : ℝ) : Prop :=
  y^2 - x^2 / 3 = 1

theorem hyperbola_solution :
  ∃ x y : ℝ,
    (∃ c : ℝ, c = 2) ∧
    (∃ a : ℝ, a = 1) ∧
    (∃ n : ℝ, n = 1) ∧
    (∃ b : ℝ, b^2 = 3) ∧
    (∃ m : ℝ, m = -3) ∧
    hyperbola_eq x y := sorry

end hyperbola_solution_l310_310927


namespace savings_if_together_l310_310658

def price_per_window : ℕ := 150

def discount_offer (n : ℕ) : ℕ := n - n / 7

def cost (n : ℕ) : ℕ := price_per_window * discount_offer n

def alice_windows : ℕ := 9
def bob_windows : ℕ := 10

def separate_cost : ℕ := cost alice_windows + cost bob_windows

def total_windows : ℕ := alice_windows + bob_windows

def together_cost : ℕ := cost total_windows

def savings : ℕ := separate_cost - together_cost

theorem savings_if_together : savings = 150 := by
  sorry

end savings_if_together_l310_310658


namespace arithmetic_sequence_value_l310_310699

theorem arithmetic_sequence_value (a : ℝ) 
  (h1 : 2 * (2 * a + 1) = (a - 1) + (a + 4)) : a = 1 / 2 := 
by 
  sorry

end arithmetic_sequence_value_l310_310699


namespace minutes_spent_calling_clients_l310_310751

theorem minutes_spent_calling_clients
    (C : ℕ)
    (H1 : 7 * C + C = 560) :
    C = 70 :=
sorry

end minutes_spent_calling_clients_l310_310751


namespace A_wins_probability_is_3_over_4_l310_310635

def parity (n : ℕ) : Bool := n % 2 == 0

def number_of_dice_outcomes : ℕ := 36

def same_parity_outcome : ℕ := 18

def probability_A_wins : ℕ → ℕ → ℕ → ℚ
| total_outcomes, same_parity, different_parity =>
  (same_parity / total_outcomes : ℚ) * 1 + (different_parity / total_outcomes : ℚ) * (1 / 2)

theorem A_wins_probability_is_3_over_4 :
  probability_A_wins number_of_dice_outcomes same_parity_outcome (number_of_dice_outcomes - same_parity_outcome) = 3/4 :=
by
  sorry

end A_wins_probability_is_3_over_4_l310_310635


namespace fraction_lost_l310_310387

-- Definitions of the given conditions
def initial_pencils : ℕ := 30
def lost_pencils_initially : ℕ := 6
def current_pencils : ℕ := 16

-- Statement of the proof problem
theorem fraction_lost (initial_pencils lost_pencils_initially current_pencils : ℕ) :
  let remaining_pencils := initial_pencils - lost_pencils_initially
  let lost_remaining_pencils := remaining_pencils - current_pencils
  (lost_remaining_pencils : ℚ) / remaining_pencils = 1 / 3 :=
by
  let remaining_pencils := initial_pencils - lost_pencils_initially
  let lost_remaining_pencils := remaining_pencils - current_pencils
  sorry

end fraction_lost_l310_310387


namespace simplify_expression_l310_310386

theorem simplify_expression (a b : ℚ) (h : a ≠ b) : 
  a^2 / (a - b) + (2 * a * b - b^2) / (b - a) = a - b :=
by
  sorry

end simplify_expression_l310_310386


namespace point_in_quadrants_l310_310929

theorem point_in_quadrants (x y : ℝ) (h1 : 4 * x + 7 * y = 28) (h2 : |x| = |y|) :
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
by
  sorry

end point_in_quadrants_l310_310929


namespace y1_y2_positive_l310_310692

theorem y1_y2_positive 
  (x1 x2 x3 : ℝ)
  (y1 y2 y3 : ℝ)
  (h_line1 : y1 = -2 * x1 + 3)
  (h_line2 : y2 = -2 * x2 + 3)
  (h_line3 : y3 = -2 * x3 + 3)
  (h_order : x1 < x2 ∧ x2 < x3)
  (h_product_neg : x2 * x3 < 0) :
  y1 * y2 > 0 :=
by
  sorry

end y1_y2_positive_l310_310692


namespace find_number_l310_310506

theorem find_number (x : ℝ) (h : 0.20 * x = 0.20 * 650 + 190) : x = 1600 := by 
  sorry

end find_number_l310_310506


namespace volume_of_quadrilateral_pyramid_l310_310474

theorem volume_of_quadrilateral_pyramid (m α : ℝ) : 
  ∃ (V : ℝ), V = (2 / 3) * m^3 * (Real.cos α) * (Real.sin (2 * α)) :=
by
  sorry

end volume_of_quadrilateral_pyramid_l310_310474


namespace andrea_fewer_apples_l310_310177

theorem andrea_fewer_apples {total_apples given_to_zenny kept_by_yanna given_to_andrea : ℕ} 
  (h1 : total_apples = 60) 
  (h2 : given_to_zenny = 18) 
  (h3 : kept_by_yanna = 36) 
  (h4 : given_to_andrea = total_apples - kept_by_yanna - given_to_zenny) : 
  (given_to_andrea + 12 = given_to_zenny) := 
sorry

end andrea_fewer_apples_l310_310177


namespace sum_six_digit_odd_and_multiples_of_3_l310_310280

-- Definitions based on conditions
def num_six_digit_odd_numbers : Nat := 9 * (10 ^ 4) * 5

def num_six_digit_multiples_of_3 : Nat := 900000 / 3

-- Proof statement
theorem sum_six_digit_odd_and_multiples_of_3 : 
  num_six_digit_odd_numbers + num_six_digit_multiples_of_3 = 750000 := 
by 
  sorry

end sum_six_digit_odd_and_multiples_of_3_l310_310280


namespace cost_of_purchase_l310_310318

theorem cost_of_purchase (x : ℝ) (T_shirt boots shin_guards : ℝ) 
  (h1 : x + T_shirt = 2 * x)
  (h2 : x + boots = 5 * x)
  (h3 : x + shin_guards = 3 * x) :
  x + T_shirt + boots + shin_guards = 8 * x :=
begin
  sorry
end

end cost_of_purchase_l310_310318


namespace units_digit_of_m_power2_plus_power_of_2_l310_310746

theorem units_digit_of_m_power2_plus_power_of_2 (m : ℕ) (h : m = 2023^2 + 2^2023) : 
  (m^2 + 2^m) % 10 = 7 :=
by
  -- Enter the proof here
  sorry

end units_digit_of_m_power2_plus_power_of_2_l310_310746


namespace binomial_sum_eq_728_l310_310225

theorem binomial_sum_eq_728 :
  (Nat.choose 6 1) * 2^1 +
  (Nat.choose 6 2) * 2^2 +
  (Nat.choose 6 3) * 2^3 +
  (Nat.choose 6 4) * 2^4 +
  (Nat.choose 6 5) * 2^5 +
  (Nat.choose 6 6) * 2^6 = 728 :=
by
  sorry

end binomial_sum_eq_728_l310_310225


namespace AJHSMETL_19892_reappears_on_line_40_l310_310475
-- Import the entire Mathlib library

-- Define the conditions
def cycleLengthLetters : ℕ := 8
def cycleLengthDigits : ℕ := 5
def lcm_cycles : ℕ := Nat.lcm cycleLengthLetters cycleLengthDigits

-- Problem statement with proof to be filled in later
theorem AJHSMETL_19892_reappears_on_line_40 :
  lcm_cycles = 40 := 
by
  sorry

end AJHSMETL_19892_reappears_on_line_40_l310_310475


namespace lollipop_count_l310_310568

theorem lollipop_count (total_cost : ℝ) (cost_per_lollipop : ℝ) (h1 : total_cost = 90) (h2 : cost_per_lollipop = 0.75) : 
  total_cost / cost_per_lollipop = 120 :=
by 
  sorry

end lollipop_count_l310_310568


namespace solve_w_from_system_of_equations_l310_310150

open Real

variables (w x y z : ℝ)

theorem solve_w_from_system_of_equations
  (h1 : 2 * w + x + y + z = 1)
  (h2 : w + 2 * x + y + z = 2)
  (h3 : w + x + 2 * y + z = 2)
  (h4 : w + x + y + 2 * z = 1) :
  w = -1 / 5 :=
by
  sorry

end solve_w_from_system_of_equations_l310_310150


namespace trees_planted_l310_310982

def initial_trees : ℕ := 150
def total_trees_after_planting : ℕ := 225

theorem trees_planted (number_of_trees_planted : ℕ) : 
  number_of_trees_planted = total_trees_after_planting - initial_trees → number_of_trees_planted = 75 :=
by 
  sorry

end trees_planted_l310_310982


namespace complement_A_eq_interval_l310_310414

-- Define the universal set U as the set of all real numbers.
def U : Set ℝ := Set.univ

-- Define the set A using the condition x^2 - 2x - 3 > 0.
def A : Set ℝ := { x | x^2 - 2 * x - 3 > 0 }

-- Define the complement of A with respect to U.
def A_complement : Set ℝ := { x | -1 <= x ∧ x <= 3 }

theorem complement_A_eq_interval : A_complement = { x | -1 <= x ∧ x <= 3 } :=
by
  sorry

end complement_A_eq_interval_l310_310414


namespace possible_values_of_a_l310_310815

theorem possible_values_of_a :
  ∃ (a b c : ℤ), ∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c) → a = 3 ∨ a = 7 :=
by
  sorry

end possible_values_of_a_l310_310815


namespace birds_never_gather_44_l310_310361

theorem birds_never_gather_44 :
    ∀ (position : Fin 44 → Nat), 
    (∀ (i : Fin 44), position i ≤ 44) →
    (∀ (i j : Fin 44), position i ≠ position j) →
    ∃ (S : Nat), S % 4 = 2 →
    ∀ (moves : (Fin 44 → Fin 44) → (Fin 44 → Fin 44)),
    ¬(∃ (tree : Nat), ∀ (i : Fin 44), position i = tree) := 
sorry

end birds_never_gather_44_l310_310361


namespace area_of_triangle_with_medians_l310_310048

theorem area_of_triangle_with_medians
  (s_a s_b s_c : ℝ) :
  (∃ t : ℝ, t = (1 / 3 : ℝ) * ((s_a + s_b + s_c) * (s_b + s_c - s_a) * (s_a + s_c - s_b) * (s_a + s_b - s_c)).sqrt) :=
sorry

end area_of_triangle_with_medians_l310_310048


namespace total_cost_is_eight_times_l310_310299

theorem total_cost_is_eight_times (cost_shorts cost_shirt cost_boots cost_shin_guards : ℕ) :
  (cost_shorts + cost_shirt = 2 * cost_shorts) →
  (cost_shorts + cost_boots = 5 * cost_shorts) →
  (cost_shorts + cost_shin_guards = 3 * cost_shorts) →
  cost_shorts + cost_shirt + cost_boots + cost_shin_guards = 8 * cost_shorts := 
by 
  intros h_shirt h_boots h_shin_guards 
  sorry

end total_cost_is_eight_times_l310_310299


namespace min_value_of_f_inequality_with_conditions_l310_310702

-- Definition of f
def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (5 - x)

-- Minimum value of f
theorem min_value_of_f : ∃ m, m = (9:ℝ) / 2 ∧ ∀ x, f(x) ≥ m :=
by
  let m := (9:ℝ) / 2
  use m
  split
  . rfl -- This establishes that the value is indeed 9/2
  . sorry -- Here we would prove the actual inequality for all x

-- Given constraints, prove the inequality
theorem inequality_with_conditions (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 3) :
  1 / (a + 1) + 1 / (b + 2) ≥ 2 / 3 :=
by
  have h_sum : (a + 1) + (b + 2) = 6 := by
    linarith -- Simplify the condition
  sorry -- Here we would complete the proof using properties of reciprocals and inequalities

end min_value_of_f_inequality_with_conditions_l310_310702


namespace second_group_men_count_l310_310715

-- Define the conditions given in the problem
def men1 := 8
def days1 := 80
def days2 := 32

-- The question we need to answer
theorem second_group_men_count : 
  ∃ (men2 : ℕ), men1 * days1 = men2 * days2 ∧ men2 = 20 :=
by
  sorry

end second_group_men_count_l310_310715


namespace find_a4_l310_310587

noncomputable def geometric_sequence (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n * q

theorem find_a4 (a_n : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a_n q →
  a_n 1 + a_n 2 = -1 →
  a_n 1 - a_n 3 = -3 →
  a_n 4 = -8 :=
by 
  sorry

end find_a4_l310_310587


namespace exists_triangle_with_sides_l2_l3_l4_l310_310922

theorem exists_triangle_with_sides_l2_l3_l4
  (a1 a2 a3 a4 d : ℝ)
  (h_arith_seq : a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d)
  (h_pos : a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a4 > 0)
  (h_d_pos : d > 0) :
  a2 + a3 > a4 ∧ a3 + a4 > a2 ∧ a4 + a2 > a3 :=
by
  sorry

end exists_triangle_with_sides_l2_l3_l4_l310_310922


namespace samara_oil_spent_l310_310661

theorem samara_oil_spent (O : ℕ) (A_total : ℕ) (S_tires : ℕ) (S_detailing : ℕ) (diff : ℕ) (S_total : ℕ) :
  A_total = 2457 →
  S_tires = 467 →
  S_detailing = 79 →
  diff = 1886 →
  S_total = O + S_tires + S_detailing →
  A_total = S_total + diff →
  O = 25 :=
by
  sorry

end samara_oil_spent_l310_310661


namespace draw_probability_l310_310603

theorem draw_probability (P_A_win : ℝ) (P_A_not_lose : ℝ) (h1 : P_A_win = 0.3) (h2 : P_A_not_lose = 0.8) : 
  ∃ P_draw : ℝ, P_draw = 0.5 := 
by
  sorry

end draw_probability_l310_310603


namespace find_smallest_n_l310_310960

theorem find_smallest_n (k : ℕ) (hk: 0 < k) :
        ∃ n : ℕ, (∀ (s : Finset ℤ), s.card = n → 
        ∃ (x y : ℤ), x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ (x + y) % (2 * k) = 0 ∨ (x - y) % (2 * k) = 0) 
        ∧ n = k + 2 :=
sorry

end find_smallest_n_l310_310960


namespace coincide_foci_of_parabola_and_hyperbola_l310_310930

theorem coincide_foci_of_parabola_and_hyperbola (p : ℝ) (hpos : p > 0) :
  (∃ x y : ℝ, (x, y) = (4, 0) ∧ y^2 = 2 * p * x) →
  (∃ x y : ℝ, (x, y) = (4, 0) ∧ (x^2 / 12) - (y^2 / 4) = 1) →
  p = 8 := 
sorry

end coincide_foci_of_parabola_and_hyperbola_l310_310930


namespace probability_X_equals_Y_l310_310383

noncomputable def prob_X_equals_Y : ℚ :=
  let count_intersections : ℚ := 15
  let total_possibilities : ℚ := 15 * 15
  count_intersections / total_possibilities

theorem probability_X_equals_Y :
  (∀ (x y : ℝ), -15 * Real.pi ≤ x ∧ x ≤ 15 * Real.pi ∧ -15 * Real.pi ≤ y ∧ y ≤ 15 * Real.pi →
    (Real.cos (Real.cos x) = Real.cos (Real.cos y)) →
    prob_X_equals_Y = 1/15) :=
sorry

end probability_X_equals_Y_l310_310383


namespace part_I_part_II_l310_310092

noncomputable def f (x : ℝ) : ℝ :=
  |x - (1/2)| + |x + (1/2)|

def solutionSetM : Set ℝ :=
  { x : ℝ | -1 < x ∧ x < 1 }

theorem part_I :
  { x : ℝ | f x < 2 } = solutionSetM := 
sorry

theorem part_II (a b : ℝ) (ha : a ∈ solutionSetM) (hb : b ∈ solutionSetM) :
  |a + b| < |1 + a * b| :=
sorry

end part_I_part_II_l310_310092


namespace molecular_weight_K3AlC2O4_3_l310_310202

noncomputable def molecularWeightOfCompound : ℝ :=
  let potassium_weight : ℝ := 39.10
  let aluminum_weight  : ℝ := 26.98
  let carbon_weight    : ℝ := 12.01
  let oxygen_weight    : ℝ := 16.00
  let total_potassium_weight : ℝ := 3 * potassium_weight
  let total_aluminum_weight  : ℝ := aluminum_weight
  let total_carbon_weight    : ℝ := 3 * 2 * carbon_weight
  let total_oxygen_weight    : ℝ := 3 * 4 * oxygen_weight
  total_potassium_weight + total_aluminum_weight + total_carbon_weight + total_oxygen_weight

theorem molecular_weight_K3AlC2O4_3 : molecularWeightOfCompound = 408.34 := by
  sorry

end molecular_weight_K3AlC2O4_3_l310_310202


namespace bowlfuls_per_box_l310_310967

def clusters_per_spoonful : ℕ := 4
def spoonfuls_per_bowl : ℕ := 25
def clusters_per_box : ℕ := 500

theorem bowlfuls_per_box : clusters_per_box / (clusters_per_spoonful * spoonfuls_per_bowl) = 5 :=
by
  sorry

end bowlfuls_per_box_l310_310967


namespace books_sold_in_february_l310_310636

theorem books_sold_in_february (F : ℕ) 
  (h_avg : (15 + F + 17) / 3 = 16): 
  F = 16 := 
by 
  sorry

end books_sold_in_february_l310_310636


namespace inequality_equivalence_l310_310971

theorem inequality_equivalence (a : ℝ) : a < -1 ↔ a + 1 < 0 :=
by
  sorry

end inequality_equivalence_l310_310971


namespace largest_even_number_l310_310941

theorem largest_even_number (n : ℤ) 
    (h1 : (n-6) % 2 = 0) 
    (h2 : (n+6) = 3 * (n-6)) :
    (n + 6) = 18 :=
by
  sorry

end largest_even_number_l310_310941


namespace arithmetic_progression_complete_iff_divides_l310_310641

-- Definitions from the conditions
def complete_sequence (s : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, s n ≠ 0) ∧ (∀ m : ℤ, m ≠ 0 → ∃ n : ℕ, s n = m)

-- Arithmetic progression definition
def arithmetic_progression (a r : ℤ) (n : ℕ) : ℤ :=
  a + n * r

-- Lean theorem statement
theorem arithmetic_progression_complete_iff_divides (a r : ℤ) :
  (complete_sequence (arithmetic_progression a r)) ↔ (r ∣ a) := by
  sorry

end arithmetic_progression_complete_iff_divides_l310_310641


namespace equation_infinitely_many_solutions_l310_310531

theorem equation_infinitely_many_solutions (a : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - 2 * a) = 3 * (4 * x + 18)) ↔ a = -27 / 4 :=
sorry

end equation_infinitely_many_solutions_l310_310531


namespace sachin_age_l310_310764

theorem sachin_age {Sachin_age Rahul_age : ℕ} (h1 : Sachin_age + 14 = Rahul_age) (h2 : Sachin_age * 9 = Rahul_age * 7) : Sachin_age = 49 := by
sorry

end sachin_age_l310_310764


namespace sage_can_determine_weight_l310_310369

theorem sage_can_determine_weight : 
  ∃ (wei : ℕ → ℕ), 
    (∀ i, i ∈ {0, 1, 2, 3, 4, 5, 6} → wei i ∈ {7, 8, 9, 10, 11, 12, 13}) →  -- weights per coin in each bag
    (∀ B : ℕ, B ∈ {0, 1, 2, 3, 4, 5, 6} →  -- B indicates the specified bag
      ∃ f : ℕ → {0, 1, ..., 6} → Prop,
        f 10 = (λ i, 7 + 8 + 9 + 10 + 11 + 12 + 13 ≤ 70 * wei B) ∧
        ( (70 * wei (f 10) = 70 * wei B) ∨
          ((70 * wei (f 10) < 70 * wei B ∧ 70 * wei B ≤ 80 * wei (f 1)) ∨
           (80 * wei (f 1) < 70 * wei B ∧ 70 * wei B ≤ 90 * wei (f 2)) ∨
           (90 * wei (f 2) < 70 * wei B ∧ 70 * wei B ≤ 100 * wei (f 3)) ∨
           (100 * wei (f 3) < 70 * wei B ∧ 70 * wei B ≤ 110 * wei (f 4)) ∨
           (110 * wei (f 4) < 70 * wei B ∧ 70 * wei B ≤ 120 * wei (f 5)) ∨
           (120 * wei (f 5) < 70 * wei B ∧ 70 * wei B ≤ 130 * wei (f 6))))) :=
sorry

end sage_can_determine_weight_l310_310369


namespace average_number_of_problems_per_day_l310_310769

theorem average_number_of_problems_per_day (P D : ℕ) (hP : P = 161) (hD : D = 7) : (P / D) = 23 :=
  by sorry

end average_number_of_problems_per_day_l310_310769


namespace sum_is_18_l310_310434

/-- Define the distinct non-zero digits, Hen, Xin, Chun, satisfying the given equation. -/
theorem sum_is_18 (Hen Xin Chun : ℕ) (h1 : Hen ≠ Xin) (h2 : Xin ≠ Chun) (h3 : Hen ≠ Chun)
  (h4 : 1 ≤ Hen ∧ Hen ≤ 9) (h5 : 1 ≤ Xin ∧ Xin ≤ 9) (h6 : 1 ≤ Chun ∧ Chun ≤ 9) :
  Hen + Xin + Chun = 18 :=
sorry

end sum_is_18_l310_310434


namespace horizontal_asymptote_of_rational_function_l310_310256

theorem horizontal_asymptote_of_rational_function :
  (∃ y, y = (10 * x ^ 4 + 3 * x ^ 3 + 7 * x ^ 2 + 6 * x + 4) / (2 * x ^ 4 + 5 * x ^ 3 + 4 * x ^ 2 + 2 * x + 1) → y = 5) := sorry

end horizontal_asymptote_of_rational_function_l310_310256


namespace sqrt_floor_square_l310_310891

theorem sqrt_floor_square (h1 : 7 < Real.sqrt 50) (h2 : Real.sqrt 50 < 8) :
  Int.floor (Real.sqrt 50) ^ 2 = 49 := by
  sorry

end sqrt_floor_square_l310_310891


namespace range_of_m_l310_310056

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x + m) * (2 - x) < 1) ↔ (-4 < m ∧ m < 0) :=
sorry

end range_of_m_l310_310056


namespace evaluate_expression_simplified_l310_310466

theorem evaluate_expression_simplified (x : ℝ) (h : x = Real.sqrt 2) : 
  (x + 3) ^ 2 + (x + 2) * (x - 2) - x * (x + 6) = 7 := by
  rw [h]
  sorry

end evaluate_expression_simplified_l310_310466


namespace fraction_of_selected_color_l310_310188

theorem fraction_of_selected_color (x y : ℕ) (hx : x > 0) :
  let bw_films := 20 * x
  let color_films := 8 * y
  let selected_bw_films := (y / x) * bw_films / 100
  let selected_color_films := color_films
  let total_selected_films := selected_bw_films + selected_color_films
  (selected_color_films / total_selected_films) = 40 / 41 :=
by
  have h_bw_selected : selected_bw_films = y / 5, by sorry
  have h_fractions : (selected_color_films : ℚ) / ((selected_bw_films + selected_color_films) : ℚ) = 40 / 41, by sorry
  exact h_fractions

end fraction_of_selected_color_l310_310188


namespace original_square_side_length_l310_310462

-- Defining the variables and conditions
variables (x : ℝ) (h₁ : 1.2 * x * (x - 2) = x * x)

-- Theorem statement to prove the side length of the original square is 12 cm
theorem original_square_side_length : x = 12 :=
by
  sorry

end original_square_side_length_l310_310462


namespace total_cost_is_eight_times_l310_310298

theorem total_cost_is_eight_times (cost_shorts cost_shirt cost_boots cost_shin_guards : ℕ) :
  (cost_shorts + cost_shirt = 2 * cost_shorts) →
  (cost_shorts + cost_boots = 5 * cost_shorts) →
  (cost_shorts + cost_shin_guards = 3 * cost_shorts) →
  cost_shorts + cost_shirt + cost_boots + cost_shin_guards = 8 * cost_shorts := 
by 
  intros h_shirt h_boots h_shin_guards 
  sorry

end total_cost_is_eight_times_l310_310298


namespace find_length_QT_l310_310576

noncomputable def length_RS : ℝ := 75
noncomputable def length_PQ : ℝ := 36
noncomputable def length_PT : ℝ := 12

theorem find_length_QT :
  ∀ (PQRS : Type)
  (P Q R S T : PQRS)
  (h_RS_perp_PQ : true)
  (h_PQ_perp_RS : true)
  (h_PT_perpendicular_to_PR : true),
  QT = 24 :=
by
  sorry

end find_length_QT_l310_310576


namespace positive_real_solution_l310_310222

theorem positive_real_solution (x : ℝ) (h : 0 < x)
  (h_eq : (1/3) * (2 * x^2 + 3) = (x^2 - 40 * x - 8) * (x^2 + 20 * x + 4)) :
  x = 20 + Real.sqrt 409 :=
sorry

end positive_real_solution_l310_310222


namespace total_cost_is_eight_times_short_cost_l310_310290

variables (x : ℝ)
def cost_shorts := x
def cost_shirt := x
def cost_boots := 4 * x
def cost_shin_guards := 2 * x

theorem total_cost_is_eight_times_short_cost
    (cond_shirt : cost_shorts x + cost_shirt x = 2*x)
    (cond_boots : cost_shorts x + cost_boots x = 5*x)
    (cond_shin_guards : cost_shorts x + cost_shin_guards x = 3*x) :
    cost_shorts x + cost_shirt x + cost_boots x + cost_shin_guards x = 8*x :=
by
  sorry

end total_cost_is_eight_times_short_cost_l310_310290


namespace george_coin_distribution_l310_310688

theorem george_coin_distribution (a b c : ℕ) (h₁ : a = 1050) (h₂ : b = 1260) (h₃ : c = 210) :
  Nat.gcd (Nat.gcd a b) c = 210 :=
by
  sorry

end george_coin_distribution_l310_310688


namespace average_marks_of_class_l310_310182

theorem average_marks_of_class :
  (∀ (students total_students: ℕ) (marks95 marks0: ℕ) (avg_remaining: ℕ),
    total_students = 25 →
    students = 3 →
    marks95 = 95 →
    students = 5 →
    marks0 = 0 →
    (total_students - students - students) = 17 →
    avg_remaining = 45 →
    ((students * marks95 + students * marks0 + (total_students - students - students) * avg_remaining) / total_students) = 42)
:= sorry

end average_marks_of_class_l310_310182


namespace ship_length_in_emilys_steps_l310_310061

variable (L E S : ℝ)

-- Conditions from the problem:
variable (cond1 : 240 * E = L + 240 * S)
variable (cond2 : 60 * E = L - 60 * S)

-- Theorem to prove:
theorem ship_length_in_emilys_steps (cond1 : 240 * E = L + 240 * S) (cond2 : 60 * E = L - 60 * S) : 
  L = 96 * E := 
sorry

end ship_length_in_emilys_steps_l310_310061


namespace total_cost_is_eight_times_short_cost_l310_310292

variables (x : ℝ)
def cost_shorts := x
def cost_shirt := x
def cost_boots := 4 * x
def cost_shin_guards := 2 * x

theorem total_cost_is_eight_times_short_cost
    (cond_shirt : cost_shorts x + cost_shirt x = 2*x)
    (cond_boots : cost_shorts x + cost_boots x = 5*x)
    (cond_shin_guards : cost_shorts x + cost_shin_guards x = 3*x) :
    cost_shorts x + cost_shirt x + cost_boots x + cost_shin_guards x = 8*x :=
by
  sorry

end total_cost_is_eight_times_short_cost_l310_310292


namespace total_cost_is_eight_times_l310_310321

theorem total_cost_is_eight_times (x : ℝ) 
  (h1 : ∀ t, x + t = 2 * x)
  (h2 : ∀ b, x + b = 5 * x)
  (h3 : ∀ s, x + s = 3 * x) :
  ∃ t b s, x + t + b + s = 8 * x :=
by
  sorry

end total_cost_is_eight_times_l310_310321


namespace increasing_intervals_g_l310_310541

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

noncomputable def g (x : ℝ) : ℝ := f (2 - x^2)

theorem increasing_intervals_g : 
  (∀ x ∈ Set.Icc (-1 : ℝ) (0 : ℝ), ∀ y ∈ Set.Icc (-1 : ℝ) (0 : ℝ), x ≤ y → g x ≤ g y) ∧
  (∀ x ∈ Set.Ici (1 : ℝ), ∀ y ∈ Set.Ici (1 : ℝ), x ≤ y → g x ≤ g y) := 
sorry

end increasing_intervals_g_l310_310541


namespace undefined_denominator_values_l310_310902

theorem undefined_denominator_values (a : ℝ) : a = 3 ∨ a = -3 ↔ ∃ b : ℝ, (a - b) * (a + b) = 0 := by
  sorry

end undefined_denominator_values_l310_310902


namespace pizza_problem_l310_310248

theorem pizza_problem :
  ∃ (x : ℕ), x = 20 ∧ (3 * x ^ 2 = 3 * 14 ^ 2 * 2 + 49) :=
by
  let small_pizza_side := 14
  let large_pizza_cost := 20
  let pool_cost := 60
  let individually_cost := 30
  have total_individual_area := 2 * 3 * (small_pizza_side ^ 2)
  have extra_area := 49
  sorry

end pizza_problem_l310_310248


namespace number_of_packs_l310_310008

theorem number_of_packs (total_towels towels_per_pack : ℕ) (h1 : total_towels = 27) (h2 : towels_per_pack = 3) :
  total_towels / towels_per_pack = 9 :=
by
  sorry

end number_of_packs_l310_310008


namespace container_ratio_l310_310044

theorem container_ratio (V1 V2 V3 : ℝ)
  (h1 : (3 / 4) * V1 = (5 / 8) * V2)
  (h2 : (5 / 8) * V2 = (1 / 2) * V3) :
  V1 / V3 = 1 / 2 :=
by
  sorry

end container_ratio_l310_310044


namespace calculate_xy_yz_zx_l310_310135

variable (x y z : ℝ)

theorem calculate_xy_yz_zx (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
    (h1 : x^2 + x * y + y^2 = 75)
    (h2 : y^2 + y * z + z^2 = 49)
    (h3 : z^2 + z * x + x^2 = 124) : 
    x * y + y * z + z * x = 70 :=
sorry

end calculate_xy_yz_zx_l310_310135


namespace divisible_by_five_l310_310460

theorem divisible_by_five {x y z : ℤ} (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  5 ∣ ((x - y)^5 + (y - z)^5 + (z - x)^5) :=
sorry

end divisible_by_five_l310_310460


namespace min_troublemakers_in_class_l310_310812

noncomputable def min_troublemakers : ℕ :=
  10

theorem min_troublemakers_in_class :
  (∃ t l : ℕ, t + l = 29 ∧ t + l - 1 = 29 ∧
   (∀ i : ℕ, i < 29 → (i % 3 = 0 → ∃ t : ℕ, t = 1) ∧ 
   (i % 3 ≠ 0 → ∃ t : ℕ, t = 2))) →
   min_troublemakers = 10 :=
by
  sorry

end min_troublemakers_in_class_l310_310812


namespace num_perfect_squares_l310_310102

theorem num_perfect_squares (a b : ℤ) (h₁ : a = 100) (h₂ : b = 400) : 
  ∃ n : ℕ, (100 < n^2) ∧ (n^2 < 400) ∧ (n = 9) :=
by
  sorry

end num_perfect_squares_l310_310102


namespace proposition_p_q_true_l310_310545

def represents_hyperbola (m : ℝ) : Prop := (1 - m) * (m + 2) < 0

def represents_ellipse (m : ℝ) : Prop := (2 * m > 2 - m) ∧ (2 - m > 0)

theorem proposition_p_q_true (m : ℝ) :
  represents_hyperbola m ∧ represents_ellipse m → (1 < m ∧ m < 2) :=
by
  sorry

end proposition_p_q_true_l310_310545


namespace total_cartons_accepted_l310_310443

theorem total_cartons_accepted (total_cartons : ℕ) (customers : ℕ) (damaged_cartons_per_customer : ℕ) (initial_cartons_per_customer accepted_cartons_per_customer total_accepted_cartons : ℕ) :
    total_cartons = 400 →
    customers = 4 →
    damaged_cartons_per_customer = 60 →
    initial_cartons_per_customer = total_cartons / customers →
    accepted_cartons_per_customer = initial_cartons_per_customer - damaged_cartons_per_customer →
    total_accepted_cartons = accepted_cartons_per_customer * customers →
    total_accepted_cartons = 160 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_cartons_accepted_l310_310443


namespace square_perimeter_of_N_l310_310151

theorem square_perimeter_of_N (area_M : ℝ) (area_N : ℝ) (side_N : ℝ) (perimeter_N : ℝ)
  (h1 : area_M = 100)
  (h2 : area_N = 4 * area_M)
  (h3 : area_N = side_N * side_N)
  (h4 : perimeter_N = 4 * side_N) :
  perimeter_N = 80 := 
sorry

end square_perimeter_of_N_l310_310151


namespace price_per_ticket_is_six_l310_310123

-- Definition of the conditions
def total_tickets (friends_tickets extra_tickets : ℕ) : ℕ :=
  friends_tickets + extra_tickets

def total_cost (tickets price_per_ticket : ℕ) : ℕ :=
  tickets * price_per_ticket

-- Given conditions
def friends_tickets : ℕ := 8
def extra_tickets : ℕ := 2
def total_spent : ℕ := 60

-- Formulate the problem to prove the price per ticket
theorem price_per_ticket_is_six :
  ∃ (price_per_ticket : ℕ), price_per_ticket = 6 ∧ 
  total_cost (total_tickets friends_tickets extra_tickets) price_per_ticket = total_spent :=
by
  -- The proof is not required; we assume its correctness here.
  sorry

end price_per_ticket_is_six_l310_310123


namespace exists_triangle_with_sides_l2_l3_l4_l310_310921

theorem exists_triangle_with_sides_l2_l3_l4
  (a1 a2 a3 a4 d : ℝ)
  (h_arith_seq : a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d)
  (h_pos : a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a4 > 0)
  (h_d_pos : d > 0) :
  a2 + a3 > a4 ∧ a3 + a4 > a2 ∧ a4 + a2 > a3 :=
by
  sorry

end exists_triangle_with_sides_l2_l3_l4_l310_310921


namespace eighth_term_matchstick_count_l310_310098

def matchstick_sequence (n : ℕ) : ℕ := (n + 1) * 3

theorem eighth_term_matchstick_count : matchstick_sequence 8 = 27 :=
by
  -- the proof will go here
  sorry

end eighth_term_matchstick_count_l310_310098


namespace marbles_leftover_l310_310464

theorem marbles_leftover (r p j : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 7) (hj : j % 8 = 2) : (r + p + j) % 8 = 6 := 
sorry

end marbles_leftover_l310_310464


namespace find_c_l310_310544

theorem find_c (x y c : ℝ) (h1 : 7^(3 * x - 1) * 3^(4 * y - 3) = c^x * 27^y)
  (h2 : x + y = 4) : c = 49 :=
by
  sorry

end find_c_l310_310544


namespace problem_condition_sufficient_not_necessary_l310_310630

open real

theorem problem_condition_sufficient_not_necessary :
  (∃ φ, φ = π ∧ ∀ x, sin (2 * x + φ) = 0 → x = 0) ∧ (∃ φ, φ ≠ π ∧ ∀ x, sin (2 * x + φ) = 0 → x = 0) :=
by
  sorry

end problem_condition_sufficient_not_necessary_l310_310630


namespace counterexample_exists_l310_310055

-- Define prime predicate
def is_prime (n : ℕ) : Prop :=
∀ m, m ∣ n → m = 1 ∨ m = n

def counterexample_to_statement (n : ℕ) : Prop :=
  is_prime n ∧ ¬ is_prime (n + 2)

theorem counterexample_exists : ∃ n ∈ [3, 5, 11, 17, 23], is_prime n ∧ ¬ is_prime (n + 2) :=
by
  sorry

end counterexample_exists_l310_310055


namespace equivalent_proof_l310_310456

noncomputable def find_m_n : ℕ :=
  let boxes := [4, 5, 6] in
  let totalWays := Nat.choose 15 4 * Nat.choose (15 - 4) 5 * Nat.choose (15 - 4 - 5) 6 in
  let mathIn4 := Nat.choose 11 5 * Nat.choose (11 - 5) 6 in
  let mathIn5 := Nat.choose 11 1 * Nat.choose 10 4 * Nat.choose 6 6 in
  let mathIn6 := Nat.choose 11 2 * Nat.choose 9 4 * Nat.choose 5 5 in
  let totalMathWays := mathIn4 + mathIn5 + mathIn6 in
  let prob := totalMathWays * totalWays in
  let gcd_val := Nat.gcd (totalMathWays * totalWays) totalWays in
  let m := (totalMathWays * totalWays) / gcd_val in
  let n := totalWays / gcd_val in
  m + n

theorem equivalent_proof : 
  ∃ m n : ℕ, 
    (Nat.gcd m n = 1) ∧
    (m / Nat.gcd m n + n / Nat.gcd m n = find_m_n) :=
by
  sorry

end equivalent_proof_l310_310456


namespace square_side_length_tangent_circle_l310_310367

theorem square_side_length_tangent_circle (r s : ℝ) :
  (∃ (O : ℝ × ℝ) (A : ℝ × ℝ) (AB : ℝ) (AD : ℝ),
    AB = AD ∧
    O = (r, r) ∧
    A = (0, 0) ∧
    dist O A = r * Real.sqrt 2 ∧
    s = dist (O.fst, 0) A ∧
    s = dist (0, O.snd) A ∧
    ∀ x y, (O = (x, y) → x = r ∧ y = r)) → s = 2 * r :=
by
  sorry

end square_side_length_tangent_circle_l310_310367


namespace prove_f_of_increasing_l310_310407

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def strictly_increasing_on_positives (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0

theorem prove_f_of_increasing {f : ℝ → ℝ}
  (h_odd : odd_function f)
  (h_incr : strictly_increasing_on_positives f) :
  f (-3) > f (-5) :=
by
  sorry

end prove_f_of_increasing_l310_310407


namespace sum_of_ages_five_years_ago_l310_310663

-- Definitions from the conditions
variables (A B : ℕ) -- Angela's current age and Beth's current age

-- Conditions
def angela_is_four_times_as_old_as_beth := A = 4 * B
def angela_will_be_44_in_five_years := A + 5 = 44

-- Theorem statement to prove the sum of their ages five years ago
theorem sum_of_ages_five_years_ago (h1 : angela_is_four_times_as_old_as_beth A B) (h2 : angela_will_be_44_in_five_years A) : 
  (A - 5) + (B - 5) = 39 :=
by sorry

end sum_of_ages_five_years_ago_l310_310663


namespace find_line_and_intersection_l310_310734

def direct_proportion_function (k : ℝ) (x : ℝ) : ℝ :=
  k * x

def shifted_function (k : ℝ) (x b : ℝ) : ℝ :=
  k * x + b

theorem find_line_and_intersection
  (k : ℝ) (b : ℝ) (h₀ : direct_proportion_function k 1 = 2) (h₁ : b = 5) :
  (shifted_function k 1 b = 7) ∧ (shifted_function k (-5/2) b = 0) :=
by
  -- This is just a placeholder to indicate where the proof would go
  sorry

end find_line_and_intersection_l310_310734


namespace petrol_expense_l310_310848

theorem petrol_expense 
  (rent milk groceries education misc savings petrol total_salary : ℝ)
  (H1 : rent = 5000)
  (H2 : milk = 1500)
  (H3 : groceries = 4500)
  (H4 : education = 2500)
  (H5 : misc = 6100)
  (H6 : savings = 2400)
  (H7 : total_salary = savings / 0.10)
  (H8 : total_salary = rent + milk + groceries + education + misc + petrol + savings) :
  petrol = 2000 :=
by
  sorry

end petrol_expense_l310_310848


namespace total_cost_is_eight_times_l310_310319

theorem total_cost_is_eight_times (x : ℝ) 
  (h1 : ∀ t, x + t = 2 * x)
  (h2 : ∀ b, x + b = 5 * x)
  (h3 : ∀ s, x + s = 3 * x) :
  ∃ t b s, x + t + b + s = 8 * x :=
by
  sorry

end total_cost_is_eight_times_l310_310319


namespace total_cost_shorts_tshirt_boots_shinguards_l310_310312

variable (x : ℝ)

-- Definitions provided in the problem statement.
def cost_shorts : ℝ := x
def cost_shorts_and_tshirt : ℝ := 2 * x
def cost_shorts_and_boots : ℝ := 5 * x
def cost_shorts_and_shinguards : ℝ := 3 * x

-- The proof goal to verify:
theorem total_cost_shorts_tshirt_boots_shinguards : 
  (cost_shorts x + (cost_shorts_and_tshirt x - cost_shorts x) + 
   (cost_shorts_and_boots x - cost_shorts x) + 
   (cost_shorts_and_shinguards x - cost_shorts x)) = 8 * x := by 
  sorry

end total_cost_shorts_tshirt_boots_shinguards_l310_310312


namespace collinear_points_b_value_l310_310573

theorem collinear_points_b_value :
  ∃ b : ℝ, (3 - (-2)) * (11 - b) = (8 - 3) * (1 - b) → b = -9 :=
by
  sorry

end collinear_points_b_value_l310_310573


namespace perfect_squares_between_100_and_400_l310_310107

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def count_perfect_squares_between (a b : ℕ) : ℕ :=
  (finset.Ico a b).filter is_perfect_square .card

theorem perfect_squares_between_100_and_400 : count_perfect_squares_between 101 400 = 9 :=
by
  -- The space for the proof is intentionally left as a placeholder
  sorry

end perfect_squares_between_100_and_400_l310_310107


namespace Josiah_spent_on_cookies_l310_310897

theorem Josiah_spent_on_cookies :
  let cookies_per_day := 2
  let cost_per_cookie := 16
  let days_in_march := 31
  2 * days_in_march * cost_per_cookie = 992 := 
by
  sorry

end Josiah_spent_on_cookies_l310_310897


namespace combined_value_l310_310112

theorem combined_value (a b : ℝ) (h1 : 0.005 * a = 95 / 100) (h2 : b = 3 * a - 50) : a + b = 710 := by
  sorry

end combined_value_l310_310112


namespace sqrt_floor_squared_l310_310872

theorem sqrt_floor_squared (h1 : 7^2 = 49) (h2 : 8^2 = 64) (h3 : 7 < Real.sqrt 50) (h4 : Real.sqrt 50 < 8) : (Int.floor (Real.sqrt 50))^2 = 49 :=
by
  sorry

end sqrt_floor_squared_l310_310872


namespace maximum_mass_difference_l310_310288

theorem maximum_mass_difference (m1 m2 : ℝ) (h1 : 19.7 ≤ m1 ∧ m1 ≤ 20.3) (h2 : 19.7 ≤ m2 ∧ m2 ≤ 20.3) :
  abs (m1 - m2) ≤ 0.6 :=
by
  sorry

end maximum_mass_difference_l310_310288


namespace tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l310_310547

-- Conditions
variables {O : ℝ × ℝ} (A : ℝ × ℝ) (B : ℝ × ℝ)
          {P Q : ℝ × ℝ} (p : ℝ)
          (hp : 0 < p)
          (hA : A.1 ^ 2 = 2 * p * A.2)
          (hB : B = (0, -1))
          (hP : P.2 = P.1 ^ 2 / (2 * p))
          (hQ : Q.2 = Q.1 ^ 2 / (2 * p))

-- Proof problem statements
theorem tangent_line_AB
  (hAB_tangent : ∀ x : ℝ, x ^ 2 / (2 * p) = 2 * x - 1 → x = 1) : true :=
by sorry

theorem op_oq_leq_oa_squared 
  (h_op_oq_leq : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + (P.1 ^ 2 / (2 * p)) ^ 2) * (Q.1 ^ 2 + (Q.1 ^ 2 / (2 * p)) ^ 2) ≤ 2) : true :=
by sorry

theorem bp_bq_gt_ba_squared 
  ( h_bp_bq_gt : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + ((P.1 ^ 2 / (2 * p)) + 1) ^ 2) * (Q.1 ^ 2 + ((Q.1 ^ 2 / (2 * p)) +1 ) ^ 2) > 5 ) : true :=
by sorry

end tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l310_310547


namespace triangle_inequality_l310_310332

theorem triangle_inequality (a b c : ℝ) (h : a + b > c ∧ a + c > b ∧ b + c > a) :
  1 < a / (b + c) + b / (c + a) + c / (a + b) ∧ a / (b + c) + b / (c + a) + c / (a + b) < 2 := 
sorry

end triangle_inequality_l310_310332


namespace digits_base_d_l310_310336

theorem digits_base_d (d A B : ℕ) (h₀ : d > 7) (h₁ : A < d) (h₂ : B < d) 
  (h₃ : A * d + B + B * d + A = 2 * d^2 + 2) : A - B = 2 :=
by
  sorry

end digits_base_d_l310_310336


namespace twelfth_term_of_arithmetic_sequence_l310_310497

/-- Condition: a_1 = 1/2 -/
def a1 : ℚ := 1 / 2

/-- Condition: common difference d = 1/3 -/
def d : ℚ := 1 / 3

/-- Prove that the 12th term in the arithmetic sequence is 25/6 given the conditions. -/
theorem twelfth_term_of_arithmetic_sequence : a1 + 11 * d = 25 / 6 := by
  sorry

end twelfth_term_of_arithmetic_sequence_l310_310497


namespace problem_1_problem_2_l310_310553

open Real
open Set

noncomputable def y (x : ℝ) : ℝ := (2 * sin x - cos x ^ 2) / (1 + sin x)

theorem problem_1 :
  { x : ℝ | y x = 1 ∧ sin x ≠ -1 } = { x | ∃ (k : ℤ), x = 2 * k * π + (π / 2) } :=
by
  sorry

theorem problem_2 : 
  ∃ x, y x = 1 ∧ ∀ x', y x' ≤ 1 :=
by
  sorry

end problem_1_problem_2_l310_310553


namespace log_sqrt2_bounds_l310_310829

theorem log_sqrt2_bounds :
  10^3 = 1000 →
  10^4 = 10000 →
  2^11 = 2048 →
  2^12 = 4096 →
  2^13 = 8192 →
  2^14 = 16384 →
  3 / 22 < Real.log 2 / Real.log 10 / 2 ∧ Real.log 2 / Real.log 10 / 2 < 1 / 7 :=
by
  sorry

end log_sqrt2_bounds_l310_310829


namespace son_l310_310370

theorem son's_present_age
  (S F : ℤ)
  (h1 : F = S + 45)
  (h2 : F + 10 = 4 * (S + 10))
  (h3 : S + 15 = 2 * S) :
  S = 15 :=
by
  sorry

end son_l310_310370


namespace unique_7_tuple_count_l310_310075

theorem unique_7_tuple_count :
  ∃! (x : ℕ → ℝ) (zero_le_x : (∀ i, 0 ≤ i → i ≤ 6 → true)),
  (2 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1 / 8 :=
by
  sorry

end unique_7_tuple_count_l310_310075


namespace Chloe_second_round_points_l310_310051

-- Conditions
def firstRoundPoints : ℕ := 40
def lastRoundPointsLost : ℕ := 4
def totalPoints : ℕ := 86
def secondRoundPoints : ℕ := 50

-- Statement to prove: Chloe scored 50 points in the second round
theorem Chloe_second_round_points :
  firstRoundPoints + secondRoundPoints - lastRoundPointsLost = totalPoints :=
by {
  -- Proof (not required, skipping with sorry)
  sorry
}

end Chloe_second_round_points_l310_310051


namespace petya_purchase_cost_l310_310328

theorem petya_purchase_cost (x : ℝ) 
  (h1 : ∃ shirt_cost : ℝ, x + shirt_cost = 2 * x)
  (h2 : ∃ boots_cost : ℝ, x + boots_cost = 5 * x)
  (h3 : ∃ shin_guards_cost : ℝ, x + shin_guards_cost = 3 * x) :
  ∃ total_cost : ℝ, total_cost = 8 * x :=
by 
  sorry

end petya_purchase_cost_l310_310328


namespace frog_climbing_time_is_correct_l310_310639

noncomputable def frog_climb_out_time : Nat :=
  let well_depth := 12
  let climb_up := 3
  let slip_down := 1
  let net_gain := climb_up - slip_down
  let total_cycles := (well_depth - 3) / net_gain + 1
  let total_time := total_cycles * 3
  let extra_time := 6
  total_time + extra_time

theorem frog_climbing_time_is_correct :
  frog_climb_out_time = 22 := by
  sorry

end frog_climbing_time_is_correct_l310_310639


namespace monotonically_increasing_f_l310_310849

open Set Filter Topology

noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

theorem monotonically_increasing_f : MonotoneOn f (Ioi 0) :=
sorry

end monotonically_increasing_f_l310_310849


namespace no_triangle_formed_l310_310096

def line1 (x y : ℝ) := 2 * x - 3 * y + 1 = 0
def line2 (x y : ℝ) := 4 * x + 3 * y + 5 = 0
def line3 (m : ℝ) (x y : ℝ) := m * x - y - 1 = 0

theorem no_triangle_formed (m : ℝ) :
  (∀ x y, line1 x y → line3 m x y) ∨
  (∀ x y, line2 x y → line3 m x y) ∨
  (∃ x y, line1 x y ∧ line2 x y ∧ line3 m x y) ↔
  (m = -4/3 ∨ m = 2/3 ∨ m = 4/3) :=
sorry -- Proof to be provided

end no_triangle_formed_l310_310096


namespace total_cost_is_eight_times_short_cost_l310_310289

variables (x : ℝ)
def cost_shorts := x
def cost_shirt := x
def cost_boots := 4 * x
def cost_shin_guards := 2 * x

theorem total_cost_is_eight_times_short_cost
    (cond_shirt : cost_shorts x + cost_shirt x = 2*x)
    (cond_boots : cost_shorts x + cost_boots x = 5*x)
    (cond_shin_guards : cost_shorts x + cost_shin_guards x = 3*x) :
    cost_shorts x + cost_shirt x + cost_boots x + cost_shin_guards x = 8*x :=
by
  sorry

end total_cost_is_eight_times_short_cost_l310_310289


namespace find_a1_l310_310244

theorem find_a1 (a : ℕ → ℝ) (h1 : ∀ n : ℕ, n > 0 → a (n + 1) = 1 / (1 - a n)) (h2 : a 8 = 2)
: a 1 = 1 / 2 :=
sorry

end find_a1_l310_310244


namespace minimum_f_l310_310552

def f (x : ℝ) : ℝ := |3 - x| + |x - 7|

theorem minimum_f : ∀ x : ℝ, min (f x) = 4 := sorry

end minimum_f_l310_310552


namespace moles_of_AgOH_formed_l310_310396

theorem moles_of_AgOH_formed (moles_AgNO3 : ℕ) (moles_NaOH : ℕ) 
  (reaction : moles_AgNO3 + moles_NaOH = 2) : moles_AgNO3 + 2 = 2 :=
by
  sorry

end moles_of_AgOH_formed_l310_310396


namespace min_troublemakers_29_l310_310809

noncomputable def min_troublemakers (n : ℕ) : ℕ :=
sorry

theorem min_troublemakers_29 : min_troublemakers 29 = 10 := 
sorry

end min_troublemakers_29_l310_310809


namespace Oates_reunion_l310_310493

-- Declare the conditions as variables
variables (total_guests both_reunions yellow_reunion : ℕ)
variables (H1 : total_guests = 100)
variables (H2 : both_reunions = 7)
variables (H3 : yellow_reunion = 65)

-- The proof problem statement
theorem Oates_reunion (O : ℕ) (H4 : total_guests = O + yellow_reunion - both_reunions) : O = 42 :=
sorry

end Oates_reunion_l310_310493


namespace solution_set_of_inequality_l310_310446

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x^2 - x else Real.log (x + 1) / Real.log 2

theorem solution_set_of_inequality :
  { x : ℝ | f x ≥ 2 } = { x : ℝ | x ∈ Set.Iic (-1) } ∪ { x : ℝ | x ∈ Set.Ici 3 } :=
by
  sorry

end solution_set_of_inequality_l310_310446


namespace corn_plants_multiple_of_nine_l310_310713

theorem corn_plants_multiple_of_nine 
  (num_sunflowers : ℕ) (num_tomatoes : ℕ) (num_corn : ℕ) (max_plants_per_row : ℕ)
  (h1 : num_sunflowers = 45) (h2 : num_tomatoes = 63) (h3 : max_plants_per_row = 9)
  : ∃ k : ℕ, num_corn = 9 * k :=
by
  sorry

end corn_plants_multiple_of_nine_l310_310713


namespace floor_sqrt_50_squared_l310_310886

theorem floor_sqrt_50_squared :
  (\lfloor real.sqrt 50 \rfloor)^2 = 49 := 
by
  sorry

end floor_sqrt_50_squared_l310_310886


namespace number_problem_l310_310363

theorem number_problem (x : ℤ) (h1 : (x - 5) / 7 = 7) : (x - 24) / 10 = 3 := by
  sorry

end number_problem_l310_310363


namespace min_troublemakers_in_class_l310_310810

noncomputable def min_troublemakers : ℕ :=
  10

theorem min_troublemakers_in_class :
  (∃ t l : ℕ, t + l = 29 ∧ t + l - 1 = 29 ∧
   (∀ i : ℕ, i < 29 → (i % 3 = 0 → ∃ t : ℕ, t = 1) ∧ 
   (i % 3 ≠ 0 → ∃ t : ℕ, t = 2))) →
   min_troublemakers = 10 :=
by
  sorry

end min_troublemakers_in_class_l310_310810


namespace find_f_6_l310_310423

def f : ℕ → ℕ := sorry

lemma f_equality (x : ℕ) : f (x + 1) = x := sorry

theorem find_f_6 : f 6 = 5 :=
by
  -- the proof would go here
  sorry

end find_f_6_l310_310423


namespace proposition_C_is_true_l310_310021

theorem proposition_C_is_true :
  (∀ θ : ℝ, 90 < θ ∧ θ < 180 → θ > 90) :=
by
  sorry

end proposition_C_is_true_l310_310021


namespace min_troublemakers_l310_310797

theorem min_troublemakers (n : ℕ) (truth_teller liar : ℕ → Prop) 
  (total_students : n = 29)
  (exists_one_liar : ∀ i : ℕ, liar i = true → ((liar (i - 1) = true ∨ liar (i + 1) = true))) -- assume cyclic seating
  (exists_two_liar : ∀ i : ℕ, truth_teller i = true → (liar (i - 1) = true ∧ liar (i + 1) = true)) :
  (∃ k : ℕ, k = 10 ∧ (∀ i : ℕ, i < 29 -> liar i = true -> k ≥ i)) :=
sorry

end min_troublemakers_l310_310797


namespace cost_per_mile_l310_310966

theorem cost_per_mile (m x : ℝ) (h_cost_eq : 2.50 + x * m = 2.50 + 5.00 + x * 14) : 
  x = 5 / 14 :=
by
  sorry

end cost_per_mile_l310_310966


namespace maximize_revenue_l310_310637

theorem maximize_revenue (p : ℝ) (hp : p ≤ 30) :
  (p = 12 ∨ p = 13) → (∀ p : ℤ, p ≤ 30 → 200 * p - 8 * p * p ≤ 1248) :=
by
  intros h1 h2
  sorry

end maximize_revenue_l310_310637


namespace distribution_schemes_count_l310_310391

def students : Finset (Fin 4) := {0, 1, 2, 3}
def villages : Finset (Fin 3) := {0, 1, 2}

theorem distribution_schemes_count (h : ∀ village, 1 ≤ (students.filter (λ student, ∃ v ∈ villages, true)).card) :
  students.card = 4 ∧ villages.card = 3 →
  ∃ n, n = 36 :=
by
  intro h_card
  have h_proof : ∀ village, 1 ≤ (students.filter (λ student, ∃ v, v ∈ villages)).card, from sorry
  exact ⟨36, rfl⟩

end distribution_schemes_count_l310_310391


namespace range_of_q_l310_310530

variable (x : ℝ)

def q (x : ℝ) := (3 * x^2 + 1)^2

theorem range_of_q : ∀ y, (∃ x : ℝ, x ≥ 0 ∧ y = q x) ↔ y ≥ 1 := by
  sorry

end range_of_q_l310_310530


namespace new_avg_weight_of_boxes_l310_310219

theorem new_avg_weight_of_boxes :
  ∀ (x y : ℕ), x + y = 30 → (10 * x + 20 * y) / 30 = 18 → (10 * x + 20 * (y - 18)) / 12 = 15 :=
by
  intro x y h1 h2
  sorry

end new_avg_weight_of_boxes_l310_310219


namespace fill_pipe_half_cistern_time_l310_310623

theorem fill_pipe_half_cistern_time (time_to_fill_half : ℕ) 
  (H : time_to_fill_half = 10) : 
  time_to_fill_half = 10 := 
by
  -- Proof is omitted
  sorry

end fill_pipe_half_cistern_time_l310_310623


namespace opposite_of_neg_2023_l310_310160

theorem opposite_of_neg_2023 : -( -2023 ) = 2023 := by
  sorry

end opposite_of_neg_2023_l310_310160


namespace graph_passes_through_quadrants_l310_310242

theorem graph_passes_through_quadrants (k : ℝ) (h : k < 0) :
  ∀ (x y : ℝ), (y = k * x - k) → 
    ((0 < x ∧ 0 < y) ∨ (x < 0 ∧ 0 < y) ∨ (x < 0 ∧ y < 0)) :=
by
  sorry

end graph_passes_through_quadrants_l310_310242


namespace min_max_expr_l310_310673

noncomputable def expr (a b c : ℝ) : ℝ :=
  (1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1)) *
  (a^2 / (a^2 + 1) + b^2 / (b^2 + 1) + c^2 / (c^2 + 1))

theorem min_max_expr (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h_cond : a * b + b * c + c * a = 1) :
  27 / 16 ≤ expr a b c ∧ expr a b c ≤ 2 :=
sorry

end min_max_expr_l310_310673


namespace minimum_number_of_circles_l310_310358

-- Define the problem conditions
def conditions_of_problem (circles : ℕ) (n : ℕ) (highlighted_lines : ℕ) (sides_of_regular_2011_gon : ℕ) : Prop :=
  circles ≥ n ∧ highlighted_lines = sides_of_regular_2011_gon

-- The main theorem we need to prove
theorem minimum_number_of_circles :
  ∀ (n circles highlighted_lines sides_of_regular_2011_gon : ℕ),
    sides_of_regular_2011_gon = 2011 ∧ (highlighted_lines = sides_of_regular_2011_gon * 2) ∧ conditions_of_problem circles n highlighted_lines sides_of_regular_2011_gon → n = 504 :=
by
  sorry

end minimum_number_of_circles_l310_310358


namespace colored_pencils_count_l310_310486

-- Given conditions
def bundles := 7
def pencils_per_bundle := 10
def extra_colored_pencils := 3

-- Calculations based on conditions
def total_pencils : ℕ := bundles * pencils_per_bundle
def total_colored_pencils : ℕ := total_pencils + extra_colored_pencils

-- Statement to be proved
theorem colored_pencils_count : total_colored_pencils = 73 := by
  sorry

end colored_pencils_count_l310_310486


namespace number_of_boys_l310_310349

noncomputable def numGirls : Nat := 46
noncomputable def numGroups : Nat := 8
noncomputable def groupSize : Nat := 9
noncomputable def totalMembers : Nat := numGroups * groupSize
noncomputable def numBoys : Nat := totalMembers - numGirls

theorem number_of_boys :
  numBoys = 26 := by
  sorry

end number_of_boys_l310_310349


namespace oliver_cycling_distance_l310_310968

/-- Oliver has a training loop for his weekend cycling. He starts by cycling due north for 3 miles. 
  Then he cycles northeast, making a 30° angle with the north for 2 miles, followed by cycling 
  southeast, making a 60° angle with the south for 2 miles. He completes his loop by cycling 
  directly back to the starting point. Prove that the distance of this final segment of his ride 
  is √(11 + 6√3) miles. -/
theorem oliver_cycling_distance :
  let north_displacement : ℝ := 3
  let northeast_displacement : ℝ := 2
  let northeast_angle : ℝ := 30
  let southeast_displacement : ℝ := 2
  let southeast_angle : ℝ := 60
  let north_northeast : ℝ := northeast_displacement * Real.cos (northeast_angle * Real.pi / 180)
  let east_northeast : ℝ := northeast_displacement * Real.sin (northeast_angle * Real.pi / 180)
  let south_southeast : ℝ := southeast_displacement * Real.cos (southeast_angle * Real.pi / 180)
  let east_southeast : ℝ := southeast_displacement * Real.sin (southeast_angle * Real.pi / 180)
  let total_north : ℝ := north_displacement + north_northeast - south_southeast
  let total_east : ℝ := east_northeast + east_southeast
  total_north = 2 + Real.sqrt 3 ∧ total_east = 1 + Real.sqrt 3
  → Real.sqrt (total_north^2 + total_east^2) = Real.sqrt (11 + 6 * Real.sqrt 3) :=
by
  sorry

end oliver_cycling_distance_l310_310968


namespace simplify_expression_l310_310179

variable (x : ℝ)

theorem simplify_expression : (5 * x + 2 * (4 + x)) = (7 * x + 8) := 
by
  sorry

end simplify_expression_l310_310179


namespace cost_per_load_is_25_cents_l310_310080

def washes_per_bottle := 80
def price_per_bottle_on_sale := 20
def bottles := 2
def total_cost := bottles * price_per_bottle_on_sale -- 2 * 20 = 40
def total_loads := bottles * washes_per_bottle -- 2 * 80 = 160
def cost_per_load_in_dollars := total_cost / total_loads -- 40 / 160 = 0.25
def cost_per_load_in_cents := cost_per_load_in_dollars * 100

theorem cost_per_load_is_25_cents :
  cost_per_load_in_cents = 25 :=
by 
  sorry

end cost_per_load_is_25_cents_l310_310080


namespace tank_capacity_l310_310846

theorem tank_capacity (T : ℕ) (h1 : T > 0) 
    (h2 : (2 * T) / 5 + 15 + 20 = T - 25) : 
    T = 100 := 
  by 
    sorry

end tank_capacity_l310_310846


namespace min_troublemakers_l310_310803

theorem min_troublemakers (students : Finset ℕ) (table : List ℕ) (is_liar : ℕ → Bool) :
  students.card = 29 ∧
  (∀ s ∈ students, 
    (∃ i j : ℕ, 
      table.nth i = some s ∧ table.nth ((i + 1) % 29) = some j ∧
      (∃ k : ℕ, 
        (is_liar s = false → count (/=is_liar k liar) = 0 ∧
        is_liar s = true → count (/=is_liar k liar) = 2))) → 
  (∃! l ∈ students, count (/=is_liar l liar) = 1))) → 
  ∃ t : ℕ, t = 10 :=
sorry

end min_troublemakers_l310_310803


namespace calc_is_a_pow4_l310_310020

theorem calc_is_a_pow4 (a : ℕ) : (a^2)^2 = a^4 := 
by 
  sorry

end calc_is_a_pow4_l310_310020


namespace sum_of_x_for_ggg_eq_neg2_l310_310640

noncomputable def g (x : ℝ) := (x^2) / 3 + x - 2

theorem sum_of_x_for_ggg_eq_neg2 : (∃ x1 x2 : ℝ, (g (g (g x1)) = -2 ∧ g (g (g x2)) = -2 ∧ x1 ≠ x2)) ∧ (x1 + x2 = 0) :=
by
  sorry

end sum_of_x_for_ggg_eq_neg2_l310_310640


namespace seventh_monomial_l310_310593

noncomputable def sequence_monomial (n : ℕ) (x : ℝ) : ℝ :=
  (-1)^n * 2^(n-1) * x^(n-1)

theorem seventh_monomial (x : ℝ) : sequence_monomial 7 x = -64 * x^6 := by
  sorry

end seventh_monomial_l310_310593


namespace simplify_expression_l310_310450

variable (c d : ℝ)
variable (hc : 0 < c)
variable (hd : 0 < d)
variable (h : c^3 + d^3 = 3 * (c + d))

theorem simplify_expression : (c / d) + (d / c) - (3 / (c * d)) = 1 := by
  sorry

end simplify_expression_l310_310450


namespace quadratic_nonneg_iff_m_in_range_l310_310260

theorem quadratic_nonneg_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 2 * m + 5 ≥ 0) ↔ (-2 : ℝ) ≤ m ∧ m ≤ 10 :=
by sorry

end quadratic_nonneg_iff_m_in_range_l310_310260


namespace acid_solution_mix_l310_310816

theorem acid_solution_mix (x : ℝ) (h₁ : 0.2 * x + 50 = 0.35 * (100 + x)) : x = 100 :=
by
  sorry

end acid_solution_mix_l310_310816


namespace undefined_denominator_values_l310_310900

theorem undefined_denominator_values (a : ℝ) : a = 3 ∨ a = -3 ↔ ∃ b : ℝ, (a - b) * (a + b) = 0 := by
  sorry

end undefined_denominator_values_l310_310900


namespace range_m_l310_310088

theorem range_m (m : ℝ) :
  (∀ x : ℝ, (1 / 3 < x ∧ x < 1 / 2) ↔ abs (x - m) < 1) →
  -1 / 2 ≤ m ∧ m ≤ 4 / 3 :=
by
  intro h
  sorry

end range_m_l310_310088


namespace quadratic_real_roots_l310_310726

theorem quadratic_real_roots (k : ℝ) (h1 : k ≠ 0) : (4 + 4 * k) ≥ 0 ↔ k ≥ -1 := 
by 
  sorry

end quadratic_real_roots_l310_310726


namespace fraction_meaningful_l310_310771

theorem fraction_meaningful (x : ℝ) : (x + 2 ≠ 0) ↔ x ≠ -2 := by
  sorry

end fraction_meaningful_l310_310771


namespace otimes_square_neq_l310_310865

noncomputable def otimes (a b : ℝ) : ℝ :=
  if a > b then a else b

theorem otimes_square_neq (a b : ℝ) (h : a ≠ b) : (otimes a b) ^ 2 ≠ otimes (a ^ 2) (b ^ 2) := by
  sorry

end otimes_square_neq_l310_310865


namespace four_thirds_eq_36_l310_310912

theorem four_thirds_eq_36 (x : ℝ) (h : (4 / 3) * x = 36) : x = 27 := by
  sorry

end four_thirds_eq_36_l310_310912


namespace condition_for_all_real_solutions_l310_310469

theorem condition_for_all_real_solutions (c : ℝ) :
  (∀ x : ℝ, x^2 + x + c > 0) ↔ c > 1 / 4 :=
sorry

end condition_for_all_real_solutions_l310_310469


namespace motorist_routes_birmingham_to_sheffield_l310_310371

-- Definitions for the conditions
def routes_bristol_to_birmingham : ℕ := 6
def routes_sheffield_to_carlisle : ℕ := 2
def total_routes_bristol_to_carlisle : ℕ := 36

-- The proposition that should be proven
theorem motorist_routes_birmingham_to_sheffield : 
  ∃ x : ℕ, routes_bristol_to_birmingham * x * routes_sheffield_to_carlisle = total_routes_bristol_to_carlisle ∧ x = 3 :=
sorry

end motorist_routes_birmingham_to_sheffield_l310_310371


namespace total_cost_is_eight_x_l310_310304

-- Definitions of cost variables based on conditions
variable (x : ℝ) -- Cost of shorts

-- Cost conditions
variable (shirt_cost : ℝ) (boot_cost : ℝ) (shin_guard_cost : ℝ)
variable (c1 : x + shirt_cost = 2 * x)
variable (c2 : x + boot_cost = 5 * x)
variable (c3 : x + shin_guard_cost = 3 * x)

-- To prove that the total cost is 8 times the cost of shorts
theorem total_cost_is_eight_x
  (c1 : x + shirt_cost = 2 * x)
  (c2 : x + boot_cost = 5 * x)
  (c3 : x + shin_guard_cost = 3 * x) :
  x + shirt_cost + boot_cost + shin_guard_cost = 8 * x := 
by
  sorry

end total_cost_is_eight_x_l310_310304


namespace total_cost_is_eight_times_l310_310300

theorem total_cost_is_eight_times (cost_shorts cost_shirt cost_boots cost_shin_guards : ℕ) :
  (cost_shorts + cost_shirt = 2 * cost_shorts) →
  (cost_shorts + cost_boots = 5 * cost_shorts) →
  (cost_shorts + cost_shin_guards = 3 * cost_shorts) →
  cost_shorts + cost_shirt + cost_boots + cost_shin_guards = 8 * cost_shorts := 
by 
  intros h_shirt h_boots h_shin_guards 
  sorry

end total_cost_is_eight_times_l310_310300


namespace power_of_power_rule_l310_310524

theorem power_of_power_rule (h : 128 = 2^7) : (128: ℝ)^(4/7) = 16 := by
  sorry

end power_of_power_rule_l310_310524


namespace trains_total_distance_l310_310009

theorem trains_total_distance (speedA_kmph speedB_kmph time_min : ℕ)
                             (hA : speedA_kmph = 70)
                             (hB : speedB_kmph = 90)
                             (hT : time_min = 15) :
    let speedA_kmpm := (speedA_kmph : ℝ) / 60
    let speedB_kmpm := (speedB_kmph : ℝ) / 60
    let distanceA := speedA_kmpm * (time_min : ℝ)
    let distanceB := speedB_kmpm * (time_min : ℝ)
    distanceA + distanceB = 40 := 
by 
  sorry

end trains_total_distance_l310_310009


namespace sqrt_equation_solution_l310_310536

theorem sqrt_equation_solution (x : ℝ) :
  (Real.sqrt ((2 + Real.sqrt 3) ^ x) + Real.sqrt ((2 - Real.sqrt 3) ^ x) = 4) ↔ (x = 2 ∨ x = -2) := 
by
  sorry

end sqrt_equation_solution_l310_310536


namespace balloon_arrangement_count_l310_310529

theorem balloon_arrangement_count :
  let n := 7
  let l := 2
  let o := 2
  n.factorial / (l.factorial * o.factorial) = 1260 :=
by
  sorry

end balloon_arrangement_count_l310_310529


namespace probability_at_least_one_visits_guangzhou_l310_310152

-- Define the probabilities of visiting for persons A, B, and C
def p_A : ℚ := 2 / 3
def p_B : ℚ := 1 / 4
def p_C : ℚ := 3 / 5

-- Calculate the probability that no one visits
def p_not_A : ℚ := 1 - p_A
def p_not_B : ℚ := 1 - p_B
def p_not_C : ℚ := 1 - p_C

-- Calculate the probability that at least one person visits
def p_none_visit : ℚ := p_not_A * p_not_B * p_not_C
def p_at_least_one_visit : ℚ := 1 - p_none_visit

-- The statement we need to prove
theorem probability_at_least_one_visits_guangzhou : p_at_least_one_visit = 9 / 10 :=
by 
  sorry

end probability_at_least_one_visits_guangzhou_l310_310152


namespace floor_square_of_sqrt_50_eq_49_l310_310884

theorem floor_square_of_sqrt_50_eq_49 : (Int.floor (Real.sqrt 50))^2 = 49 := 
by
  sorry

end floor_square_of_sqrt_50_eq_49_l310_310884


namespace petya_purchase_cost_l310_310330

theorem petya_purchase_cost (x : ℝ) 
  (h1 : ∃ shirt_cost : ℝ, x + shirt_cost = 2 * x)
  (h2 : ∃ boots_cost : ℝ, x + boots_cost = 5 * x)
  (h3 : ∃ shin_guards_cost : ℝ, x + shin_guards_cost = 3 * x) :
  ∃ total_cost : ℝ, total_cost = 8 * x :=
by 
  sorry

end petya_purchase_cost_l310_310330


namespace total_cost_is_eight_times_l310_310320

theorem total_cost_is_eight_times (x : ℝ) 
  (h1 : ∀ t, x + t = 2 * x)
  (h2 : ∀ b, x + b = 5 * x)
  (h3 : ∀ s, x + s = 3 * x) :
  ∃ t b s, x + t + b + s = 8 * x :=
by
  sorry

end total_cost_is_eight_times_l310_310320


namespace determine_no_conditionals_l310_310551

def problem_requires_conditionals (n : ℕ) : Prop :=
  n = 3 ∨ n = 4

theorem determine_no_conditionals :
  problem_requires_conditionals 1 = false ∧
  problem_requires_conditionals 2 = false ∧
  problem_requires_conditionals 3 = true ∧
  problem_requires_conditionals 4 = true :=
by sorry

end determine_no_conditionals_l310_310551


namespace train_speed_approx_l310_310376

noncomputable def distance_in_kilometers (d : ℝ) : ℝ :=
d / 1000

noncomputable def time_in_hours (t : ℝ) : ℝ :=
t / 3600

noncomputable def speed_in_kmh (d : ℝ) (t : ℝ) : ℝ :=
distance_in_kilometers d / time_in_hours t

theorem train_speed_approx (d t : ℝ) (h_d : d = 200) (h_t : t = 5.80598713393251) :
  abs (speed_in_kmh d t - 124.019) < 1e-3 :=
by
  rw [h_d, h_t]
  simp only [distance_in_kilometers, time_in_hours, speed_in_kmh]
  norm_num
  -- We're using norm_num to deal with numerical approximations and constants
  -- The actual calculations can be verified through manual checks or external tools but in Lean we skip this step.
  sorry

end train_speed_approx_l310_310376


namespace students_sampled_from_second_grade_l310_310036

def arithmetic_sequence (a d : ℕ) : Prop :=
  3 * a - d = 1200

def stratified_sampling (total students second_grade : ℕ) : ℕ :=
  (second_grade * students) / total

theorem students_sampled_from_second_grade 
  (total students : ℕ)
  (h1 : total = 1200)
  (h2 : students = 48)
  (a d : ℕ)
  (h3 : arithmetic_sequence a d)
: stratified_sampling total students a = 16 :=
by
  rw [h1, h2]
  sorry

end students_sampled_from_second_grade_l310_310036


namespace ann_boxes_less_than_n_l310_310137

-- Define the total number of boxes n
def n : ℕ := 12

-- Define the number of boxes Mark sold
def mark_sold : ℕ := n - 11

-- Define a condition on the number of boxes Ann sold
def ann_sold (A : ℕ) : Prop := 1 ≤ A ∧ A < n - mark_sold

-- The statement to prove
theorem ann_boxes_less_than_n : ∃ A : ℕ, ann_sold A ∧ n - A = 2 :=
by
  sorry

end ann_boxes_less_than_n_l310_310137


namespace find_d_l310_310480

theorem find_d (a b c d : ℝ) 
  (h : a^2 + b^2 + 2 * c^2 + 4 = 2 * d + Real.sqrt (a^2 + b^2 + c - d)) :
  d = 1/2 :=
sorry

end find_d_l310_310480


namespace find_n_with_divisors_sum_l310_310448

theorem find_n_with_divisors_sum (n : ℕ) (d1 d2 d3 d4 : ℕ)
  (h1 : d1 = 1) (h2 : d2 = 2) (h3 : d3 = 5) (h4 : d4 = 10) 
  (hd : n = 130) : d1^2 + d2^2 + d3^2 + d4^2 = n :=
sorry

end find_n_with_divisors_sum_l310_310448


namespace condition_nonzero_neither_zero_l310_310614

theorem condition_nonzero_neither_zero (a b : ℝ) (h : a^2 + b^2 ≠ 0) : ¬(a = 0 ∧ b = 0) :=
sorry

end condition_nonzero_neither_zero_l310_310614


namespace part_I_part_II_l310_310241

noncomputable def f (a x : ℝ) : ℝ :=
  (Real.log (a * x + 1 / 2)) + (2 / (2 * x + 1))

theorem part_I (a : ℝ) (h_a_pos : a > 0) : (∀ x > 0, (1 / ((2 * x + 1) * (a * (2 * x + 1) - (2 * (a * x + 1) / 2))) ≥ 0) ↔ a ≥ 2) :=
sorry

theorem part_II : ∃ a : ℝ, (∀ x > 0, (Real.log (a * x + 1 / 2)) + (2 / (2 * x + 1)) ≥ 1) ∧ (Real.log (a * (Real.sqrt ((2 - a) / (4 * a))) + 1 / 2) + (2 / (2 * (Real.sqrt ((2 - a) / (4 * a))) + 1)) = 1) ∧ a = 1 :=
sorry

end part_I_part_II_l310_310241


namespace incorrect_statement_l310_310932

-- Definition of the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Definition of set M
def M : Set ℕ := {1, 2}

-- Definition of set N
def N : Set ℕ := {2, 4}

-- Complement of set in a universal set
def complement (S : Set ℕ) : Set ℕ := U \ S

-- Statement that D is incorrect
theorem incorrect_statement :
  M ∩ complement N ≠ {1, 2, 3} :=
by
  sorry

end incorrect_statement_l310_310932


namespace convex_polyhedron_faces_same_edges_l310_310761

theorem convex_polyhedron_faces_same_edges (n : ℕ) (f : Fin n → ℕ) 
  (n_ge_4 : 4 ≤ n)
  (h : ∀ i : Fin n, 3 ≤ f i ∧ f i ≤ n - 1) : 
  ∃ (i j : Fin n), i ≠ j ∧ f i = f j := 
by
  sorry

end convex_polyhedron_faces_same_edges_l310_310761


namespace prob_A_exactly_once_l310_310437

theorem prob_A_exactly_once (P : ℚ) (h : 1 - (1 - P)^3 = 63 / 64) : 
  (3 * P * (1 - P)^2 = 9 / 64) :=
by
  sorry

end prob_A_exactly_once_l310_310437


namespace clock_hand_overlaps_in_24_hours_l310_310711

-- Define the number of revolutions of the hour hand in 24 hours.
def hour_hand_revolutions_24_hours : ℕ := 2

-- Define the number of revolutions of the minute hand in 24 hours.
def minute_hand_revolutions_24_hours : ℕ := 24

-- Define the number of overlaps as a constant.
def number_of_overlaps (hour_rev : ℕ) (minute_rev : ℕ) : ℕ :=
  minute_rev - hour_rev

-- The theorem we want to prove:
theorem clock_hand_overlaps_in_24_hours :
  number_of_overlaps hour_hand_revolutions_24_hours minute_hand_revolutions_24_hours = 22 :=
sorry

end clock_hand_overlaps_in_24_hours_l310_310711


namespace rotate_parabola_180deg_l310_310333

theorem rotate_parabola_180deg (x y : ℝ) :
  (∀ x, y = 2 * x^2 - 12 * x + 16) →
  (∀ x, y = -2 * x^2 + 12 * x - 20) :=
sorry

end rotate_parabola_180deg_l310_310333


namespace total_cost_is_eight_times_l310_310295

theorem total_cost_is_eight_times (cost_shorts cost_shirt cost_boots cost_shin_guards : ℕ) :
  (cost_shorts + cost_shirt = 2 * cost_shorts) →
  (cost_shorts + cost_boots = 5 * cost_shorts) →
  (cost_shorts + cost_shin_guards = 3 * cost_shorts) →
  cost_shorts + cost_shirt + cost_boots + cost_shin_guards = 8 * cost_shorts := 
by 
  intros h_shirt h_boots h_shin_guards 
  sorry

end total_cost_is_eight_times_l310_310295


namespace cows_in_group_l310_310574

theorem cows_in_group (c h : ℕ) (L H: ℕ) 
  (legs_eq : L = 4 * c + 2 * h)
  (heads_eq : H = c + h)
  (legs_heads_relation : L = 2 * H + 14) 
  : c = 7 :=
by
  sorry

end cows_in_group_l310_310574


namespace original_card_deck_count_l310_310121

theorem original_card_deck_count (r b : ℕ) :
  (r : ℚ) / (r + b : ℚ) = 2 / 5 →
  (r + 3 : ℚ) / (r + b + 3 : ℚ) = 1 / 2 →
  r + b = 15 :=
by
  sorry

end original_card_deck_count_l310_310121


namespace maryann_time_spent_calling_clients_l310_310754

theorem maryann_time_spent_calling_clients (a c : ℕ) 
  (h1 : a + c = 560) 
  (h2 : a = 7 * c) : c = 70 := 
by 
  sorry

end maryann_time_spent_calling_clients_l310_310754


namespace rectangle_area_l310_310652

theorem rectangle_area (square_area : ℝ) (rectangle_length_ratio : ℝ) (square_area_eq : square_area = 36)
  (rectangle_length_ratio_eq : rectangle_length_ratio = 3) :
  ∃ (rectangle_area : ℝ), rectangle_area = 108 :=
by
  -- Extract the side length of the square from its area
  let side_length := real.sqrt square_area
  have side_length_eq : side_length = 6, from calc
    side_length = real.sqrt 36 : by rw [square_area_eq]
    ... = 6 : real.sqrt_eq 6 (by norm_num)
  -- Calculate the rectangle's width
  let width := side_length
  -- Calculate the rectangle's length
  let length := rectangle_length_ratio * width
  -- Calculate the area of the rectangle
  let area := width * length
  -- Prove the area is 108
  use area
  calc area
    = 6 * (3 * 6) : by rw [side_length_eq, rectangle_length_ratio_eq]
    ... = 108 : by norm_num

end rectangle_area_l310_310652


namespace pau_total_ordered_correct_l310_310278

-- Define the initial pieces of fried chicken ordered by Kobe
def kobe_order : ℝ := 5

-- Define Pau's initial order as twice Kobe's order plus 2.5 pieces
def pau_initial_order : ℝ := (2 * kobe_order) + 2.5

-- Define Shaquille's initial order as 50% more than Pau's initial order
def shaq_initial_order : ℝ := pau_initial_order * 1.5

-- Define the total pieces of chicken Pau will have eaten by the end
def pau_total_ordered : ℝ := 2 * pau_initial_order

-- Prove that Pau will have eaten 25 pieces of fried chicken by the end
theorem pau_total_ordered_correct : pau_total_ordered = 25 := by
  sorry

end pau_total_ordered_correct_l310_310278


namespace find_k_l310_310703

-- Define the vectors a, b, and c
def vecA : ℝ × ℝ := (2, -1)
def vecB : ℝ × ℝ := (1, 1)
def vecC : ℝ × ℝ := (-5, 1)

-- Define the condition for two vectors being parallel
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 - u.2 * v.1 = 0

-- Define the target statement to be proven
theorem find_k : ∃ k : ℝ, parallel (vecA.1 + k * vecB.1, vecA.2 + k * vecB.2) vecC ∧ k = 1/2 := 
sorry

end find_k_l310_310703


namespace selected_number_in_14th_group_is_272_l310_310355

-- Definitions based on conditions
def total_students : ℕ := 400
def sample_size : ℕ := 20
def first_selected_number : ℕ := 12
def sampling_interval : ℕ := total_students / sample_size
def target_group : ℕ := 14

-- Correct answer definition
def selected_number_in_14th_group : ℕ := first_selected_number + (target_group - 1) * sampling_interval

-- Theorem stating the correct answer is 272
theorem selected_number_in_14th_group_is_272 :
  selected_number_in_14th_group = 272 :=
sorry

end selected_number_in_14th_group_is_272_l310_310355


namespace program_output_l310_310993

theorem program_output :
  let a := 1
  let b := 3
  let a := a + b
  let b := b * a
  a = 4 ∧ b = 12 :=
by
  sorry

end program_output_l310_310993


namespace money_spent_on_ferris_wheel_l310_310024

-- Conditions
def initial_tickets : ℕ := 6
def remaining_tickets : ℕ := 3
def ticket_cost : ℕ := 9

-- Prove that the money spent during the ferris wheel ride is 27 dollars
theorem money_spent_on_ferris_wheel : (initial_tickets - remaining_tickets) * ticket_cost = 27 := by
  sorry

end money_spent_on_ferris_wheel_l310_310024


namespace fractions_product_l310_310667

theorem fractions_product :
  (8 / 4) * (10 / 25) * (20 / 10) * (15 / 45) * (40 / 20) * (24 / 8) * (30 / 15) * (35 / 7) = 64 := by
  sorry

end fractions_product_l310_310667


namespace max_volume_is_correct_l310_310735

noncomputable def max_volume_of_inscribed_sphere (AB BC AA₁ : ℝ) (h₁ : AB = 6) (h₂ : BC = 8) (h₃ : AA₁ = 3) : ℝ :=
  let AC := Real.sqrt ((6 : ℝ) ^ 2 + (8 : ℝ) ^ 2)
  let r := (AB + BC - AC) / 2
  let sphere_radius := AA₁ / 2
  (4/3) * Real.pi * sphere_radius ^ 3

theorem max_volume_is_correct : max_volume_of_inscribed_sphere 6 8 3 (by rfl) (by rfl) (by rfl) = 9 * Real.pi / 2 := by
  sorry

end max_volume_is_correct_l310_310735


namespace average_median_eq_l310_310023

theorem average_median_eq (a b c : ℤ) (h1 : (a + b + c) / 3 = 4 * b)
  (h2 : a < b) (h3 : b < c) (h4 : a = 0) : c / b = 11 := 
by
  sorry

end average_median_eq_l310_310023


namespace petya_purchase_cost_l310_310325

theorem petya_purchase_cost (x : ℝ) 
  (h1 : ∃ shirt_cost : ℝ, x + shirt_cost = 2 * x)
  (h2 : ∃ boots_cost : ℝ, x + boots_cost = 5 * x)
  (h3 : ∃ shin_guards_cost : ℝ, x + shin_guards_cost = 3 * x) :
  ∃ total_cost : ℝ, total_cost = 8 * x :=
by 
  sorry

end petya_purchase_cost_l310_310325


namespace geometric_sequence_m_value_l310_310578

theorem geometric_sequence_m_value 
  (a : ℕ → ℝ) (q : ℝ) (m : ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n, a n = a 1 * q^(n-1))
  (h3 : |q| ≠ 1) 
  (h4 : a m = a 1 * a 2 * a 3 * a 4 * a 5) : 
  m = 11 := by
  sorry

end geometric_sequence_m_value_l310_310578


namespace third_team_pieces_l310_310231

theorem third_team_pieces (total_pieces : ℕ) (first_team : ℕ) (second_team : ℕ) (third_team : ℕ) : 
  total_pieces = 500 → first_team = 189 → second_team = 131 → third_team = total_pieces - first_team - second_team → third_team = 180 :=
by
  intros h_total h_first h_second h_third
  rw [h_total, h_first, h_second] at h_third
  exact h_third

end third_team_pieces_l310_310231


namespace number_of_pies_l310_310972

-- Definitions based on the conditions
def box_weight : ℕ := 120
def weight_for_applesauce : ℕ := box_weight / 2
def weight_per_pie : ℕ := 4
def remaining_weight : ℕ := box_weight - weight_for_applesauce

-- The proof problem statement
theorem number_of_pies : (remaining_weight / weight_per_pie) = 15 :=
by
  sorry

end number_of_pies_l310_310972


namespace min_value_sum_reciprocal_squares_l310_310548

open Real

theorem min_value_sum_reciprocal_squares 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :  
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ 27 := 
sorry

end min_value_sum_reciprocal_squares_l310_310548


namespace sum_excluded_values_domain_l310_310223

theorem sum_excluded_values_domain (x : ℝ) :
  (3 * x^2 - 9 * x + 6 = 0) → (x = 1 ∨ x = 2) ∧ (1 + 2 = 3) :=
by {
  -- given that 3x² - 9x + 6 = 0, we need to show that x = 1 or x = 2, and that their sum is 3
  sorry
}

end sum_excluded_values_domain_l310_310223


namespace num_solutions_l310_310557

theorem num_solutions :
  ∃ n, (∀ a b c : ℤ, (|a + b| + c = 21 ∧ a * b + |c| = 85) ↔ n = 12) :=
sorry

end num_solutions_l310_310557


namespace smallest_number_of_coins_l310_310494

theorem smallest_number_of_coins :
  ∃ (n : ℕ), (∀ (a : ℕ), 5 ≤ a ∧ a < 100 → 
    ∃ (c : ℕ → ℕ), (a = 5 * c 0 + 10 * c 1 + 25 * c 2) ∧ 
    (c 0 + c 1 + c 2 = n)) ∧ n = 9 :=
by
  sorry

end smallest_number_of_coins_l310_310494


namespace total_cost_is_eight_x_l310_310301

-- Definitions of cost variables based on conditions
variable (x : ℝ) -- Cost of shorts

-- Cost conditions
variable (shirt_cost : ℝ) (boot_cost : ℝ) (shin_guard_cost : ℝ)
variable (c1 : x + shirt_cost = 2 * x)
variable (c2 : x + boot_cost = 5 * x)
variable (c3 : x + shin_guard_cost = 3 * x)

-- To prove that the total cost is 8 times the cost of shorts
theorem total_cost_is_eight_x
  (c1 : x + shirt_cost = 2 * x)
  (c2 : x + boot_cost = 5 * x)
  (c3 : x + shin_guard_cost = 3 * x) :
  x + shirt_cost + boot_cost + shin_guard_cost = 8 * x := 
by
  sorry

end total_cost_is_eight_x_l310_310301


namespace sqrt_floor_square_l310_310890

theorem sqrt_floor_square (h1 : 7 < Real.sqrt 50) (h2 : Real.sqrt 50 < 8) :
  Int.floor (Real.sqrt 50) ^ 2 = 49 := by
  sorry

end sqrt_floor_square_l310_310890


namespace area_of_triangle_ABC_l310_310171

structure Point := (x y : ℝ)

def A := Point.mk 2 3
def B := Point.mk 9 3
def C := Point.mk 4 12

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * ((B.x - A.x) * (C.y - A.y))

theorem area_of_triangle_ABC :
  area_of_triangle A B C = 31.5 :=
by
  -- Proof is omitted
  sorry

end area_of_triangle_ABC_l310_310171


namespace four_thirds_of_number_is_36_l310_310913

theorem four_thirds_of_number_is_36 (x : ℝ) (h : (4 / 3) * x = 36) : x = 27 :=
  sorry

end four_thirds_of_number_is_36_l310_310913


namespace prove_s90_zero_l310_310077

variable {a : ℕ → ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 0) + (n * (n - 1) * (a 1 - a 0)) / 2)

theorem prove_s90_zero (a : ℕ → ℕ) (h_arith : is_arithmetic_sequence a) (h : sum_of_first_n_terms a 30 = sum_of_first_n_terms a 60) :
  sum_of_first_n_terms a 90 = 0 :=
sorry

end prove_s90_zero_l310_310077


namespace problem_proof_l310_310238

theorem problem_proof (a b x y : ℝ) (h1 : a + b = 0) (h2 : x * y = 1) : 5 * |a + b| - 5 * (x * y) = -5 :=
by
  sorry

end problem_proof_l310_310238


namespace greatest_consecutive_integers_sum_55_l310_310495

theorem greatest_consecutive_integers_sum_55 :
  ∃ N a : ℤ, (N * (2 * a + N - 1)) = 110 ∧ (∀ M a' : ℤ, (M * (2 * a' + M - 1)) = 110 → N ≥ M) :=
sorry

end greatest_consecutive_integers_sum_55_l310_310495


namespace total_players_count_l310_310027

def kabadi_players : ℕ := 10
def kho_kho_only_players : ℕ := 35
def both_games_players : ℕ := 5

theorem total_players_count : kabadi_players + kho_kho_only_players - both_games_players = 40 :=
by
  sorry

end total_players_count_l310_310027


namespace orthogonal_matrix_property_l310_310054

open Matrix

variables {a b c d : ℝ}
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![a, b], ![c, d]]
def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem orthogonal_matrix_property (hA : Aᵀ = A⁻¹) (k : ℝ) (hk : k = 1) :
  ((a + 1)^2 + b^2 = 1) ∧ (c^2 + (d + 1)^2 = 1) → (a^2 + b^2 + c^2 + d^2) = 2 :=
by {
  sorry
}

end orthogonal_matrix_property_l310_310054


namespace complement_intersection_l310_310505

open Set

variable (U : Set ℕ) (A B : Set ℕ)

theorem complement_intersection :
  U = {1, 2, 3, 4, 5} →
  A = {1, 2, 3} →
  B = {2, 3, 5} →
  U \ (A ∩ B) = {1, 4, 5} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  sorry

end complement_intersection_l310_310505


namespace sum_over_positive_reals_nonnegative_l310_310690

theorem sum_over_positive_reals_nonnegative (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (b + c - 2 * a) / (a^2 + b * c) + 
  (c + a - 2 * b) / (b^2 + c * a) + 
  (a + b - 2 * c) / (c^2 + a * b) ≥ 0 :=
sorry

end sum_over_positive_reals_nonnegative_l310_310690


namespace petya_purchase_cost_l310_310326

theorem petya_purchase_cost (x : ℝ) 
  (h1 : ∃ shirt_cost : ℝ, x + shirt_cost = 2 * x)
  (h2 : ∃ boots_cost : ℝ, x + boots_cost = 5 * x)
  (h3 : ∃ shin_guards_cost : ℝ, x + shin_guards_cost = 3 * x) :
  ∃ total_cost : ℝ, total_cost = 8 * x :=
by 
  sorry

end petya_purchase_cost_l310_310326


namespace trigonometric_identity_l310_310563

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -3) : 
  (Real.cos α - Real.sin α) / (Real.cos α + Real.sin α) = -2 :=
by 
  sorry

end trigonometric_identity_l310_310563


namespace probability_of_blue_buttons_l310_310740

theorem probability_of_blue_buttons
  (orig_red_A : ℕ) (orig_blue_A : ℕ)
  (removed_red : ℕ) (removed_blue : ℕ)
  (target_ratio : ℚ)
  (final_red_A : ℕ) (final_blue_A : ℕ)
  (final_red_B : ℕ) (final_blue_B : ℕ)
  (orig_buttons_A : orig_red_A + orig_blue_A = 16)
  (removed_buttons : removed_red = 3 ∧ removed_blue = 5)
  (final_buttons_A : final_red_A + final_blue_A = 8)
  (buttons_ratio : target_ratio = 2 / 3)
  (final_ratio_A : final_red_A + final_blue_A = target_ratio * 16)
  (red_in_A : final_red_A = orig_red_A - removed_red)
  (blue_in_A : final_blue_A = orig_blue_A - removed_blue)
  (red_in_B : final_red_B = removed_red)
  (blue_in_B : final_blue_B = removed_blue):
  (final_blue_A / (final_red_A + final_blue_A)) * (final_blue_B / (final_red_B + final_blue_B)) = 25 / 64 := 
by
  sorry

end probability_of_blue_buttons_l310_310740


namespace fourth_house_number_l310_310970

theorem fourth_house_number (sum: ℕ) (k x: ℕ) (h1: sum = 78) (h2: k ≥ 4)
  (h3: (k+1) * (x + k) = 78) : x + 6 = 14 :=
by
  sorry

end fourth_house_number_l310_310970


namespace repeating_decimal_as_fraction_l310_310827

theorem repeating_decimal_as_fraction :
  (∃ y : ℚ, y = 737910 ∧ 0.73 + 864 / 999900 = y / 999900) :=
by
  -- proof omitted
  sorry

end repeating_decimal_as_fraction_l310_310827


namespace determine_a_values_l310_310718

theorem determine_a_values (a : ℝ) (A : Set ℝ) (B : Set ℝ)
  (hA : A = { x | abs x = 1 }) 
  (hB : B = { x | a * x = 1 }) 
  (h_superset : A ⊇ B) :
  a = -1 ∨ a = 0 ∨ a = 1 :=
sorry

end determine_a_values_l310_310718


namespace integer_multiplication_for_ones_l310_310615

theorem integer_multiplication_for_ones :
  ∃ x : ℤ, (10^9 - 1) * x = (10^81 - 1) / 9 :=
by
  sorry

end integer_multiplication_for_ones_l310_310615


namespace total_cost_is_eight_times_l310_310297

theorem total_cost_is_eight_times (cost_shorts cost_shirt cost_boots cost_shin_guards : ℕ) :
  (cost_shorts + cost_shirt = 2 * cost_shorts) →
  (cost_shorts + cost_boots = 5 * cost_shorts) →
  (cost_shorts + cost_shin_guards = 3 * cost_shorts) →
  cost_shorts + cost_shirt + cost_boots + cost_shin_guards = 8 * cost_shorts := 
by 
  intros h_shirt h_boots h_shin_guards 
  sorry

end total_cost_is_eight_times_l310_310297


namespace min_max_value_is_zero_l310_310066

def max_at_x (x : ℝ) (y : ℝ) : ℝ := |x^2 - 2 * x * y|

theorem min_max_value_is_zero :
  ∃ y ∈ set.univ, min (set.univ) (λ y, real.sup (set.Icc 0 2) (λ x, max_at_x x y)) = 0 :=
sorry

end min_max_value_is_zero_l310_310066


namespace Dean_handled_100_transactions_l310_310759

-- Definitions for the given conditions
def Mabel_transactions : ℕ := 90
def Anthony_transactions : ℕ := (9 * Mabel_transactions) / 10 + Mabel_transactions
def Cal_transactions : ℕ := (2 * Anthony_transactions) / 3
def Jade_transactions : ℕ := Cal_transactions + 14
def Dean_transactions : ℕ := (Jade_transactions * 25) / 100 + Jade_transactions

-- Define the theorem we need to prove
theorem Dean_handled_100_transactions : Dean_transactions = 100 :=
by
  -- Statement to skip the actual proof
  sorry

end Dean_handled_100_transactions_l310_310759


namespace find_q_revolutions_per_minute_l310_310525

variable (p_rpm : ℕ) (q_rpm : ℕ) (t : ℕ)

def revolutions_per_minute_q : Prop :=
  (p_rpm = 10) → (t = 4) → (q_rpm = (10 / 60 * 4 + 2) * 60 / 4) → (q_rpm = 120)

theorem find_q_revolutions_per_minute (p_rpm q_rpm t : ℕ) :
  revolutions_per_minute_q p_rpm q_rpm t :=
by
  unfold revolutions_per_minute_q
  sorry

end find_q_revolutions_per_minute_l310_310525


namespace brick_length_l310_310029

theorem brick_length 
  (width : ℝ) (height : ℝ) (num_bricks : ℕ)
  (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ)
  (brick_vol : ℝ) :
  width = 10 →
  height = 7.5 →
  num_bricks = 27000 →
  wall_length = 27 →
  wall_width = 2 →
  wall_height = 0.75 →
  brick_vol = width * height * (20:ℝ) →
  wall_length * wall_width * wall_height * 1000000 = num_bricks * brick_vol :=
by
  intros
  sorry

end brick_length_l310_310029


namespace cost_of_each_soda_l310_310496

theorem cost_of_each_soda (total_cost sandwiches_cost : ℝ) (number_of_sodas : ℕ)
  (h_total_cost : total_cost = 6.46)
  (h_sandwiches_cost : sandwiches_cost = 2 * 1.49) :
  total_cost - sandwiches_cost = 4 * 0.87 := by
  sorry

end cost_of_each_soda_l310_310496


namespace sqrt_floor_squared_eq_49_l310_310881

theorem sqrt_floor_squared_eq_49 : (⌊real.sqrt 50⌋)^2 = 49 :=
by sorry

end sqrt_floor_squared_eq_49_l310_310881


namespace total_leaves_on_farm_l310_310380

noncomputable def number_of_branches : ℕ := 10
noncomputable def sub_branches_per_branch : ℕ := 40
noncomputable def leaves_per_sub_branch : ℕ := 60
noncomputable def number_of_trees : ℕ := 4

theorem total_leaves_on_farm :
  number_of_branches * sub_branches_per_branch * leaves_per_sub_branch * number_of_trees = 96000 :=
by
  sorry

end total_leaves_on_farm_l310_310380


namespace division_of_fractions_l310_310170

theorem division_of_fractions : (5 / 6) / (1 + 3 / 9) = 5 / 8 := by
  sorry

end division_of_fractions_l310_310170


namespace max_tan_y_l310_310748

noncomputable def tan_y_upper_bound (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) 
    (h : Real.sin y = 2005 * Real.cos (x + y) * Real.sin x) : Real :=
  Real.tan y

theorem max_tan_y (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) 
    (h : Real.sin y = 2005 * Real.cos (x + y) * Real.sin x) : 
    tan_y_upper_bound x y hx hy h = 2005 * Real.sqrt 2006 / 4012 := 
by 
  sorry

end max_tan_y_l310_310748


namespace determine_weight_two_weighings_l310_310368

theorem determine_weight_two_weighings :
  ∃ (x : ℝ), (∃ n : ℕ, n ≤ 2 ∧ ∀ b1 b2 : list ℝ, perform_weighings b1 b2 x n)
  → (∃ w ∈ {7, 8, 9, 10, 11, 12, 13}, balance_weighted_bag b1 b2 w) :=
sorry

end determine_weight_two_weighings_l310_310368


namespace xiaofang_final_score_l310_310503

def removeHighestLowestScores (scores : List ℕ) : List ℕ :=
  let max_score := scores.maximum.getD 0
  let min_score := scores.minimum.getD 0
  scores.erase max_score |>.erase min_score

def average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem xiaofang_final_score :
  let scores := [95, 94, 91, 88, 91, 90, 94, 93, 91, 92]
  average (removeHighestLowestScores scores) = 92 := by
  sorry

end xiaofang_final_score_l310_310503


namespace four_thirds_of_number_is_36_l310_310914

theorem four_thirds_of_number_is_36 (x : ℝ) (h : (4 / 3) * x = 36) : x = 27 :=
  sorry

end four_thirds_of_number_is_36_l310_310914


namespace connie_blue_markers_l310_310860

theorem connie_blue_markers :
  ∀ (total_markers red_markers blue_markers : ℕ),
    total_markers = 105 →
    red_markers = 41 →
    blue_markers = total_markers - red_markers →
    blue_markers = 64 :=
by
  intros total_markers red_markers blue_markers htotal hred hblue
  rw [htotal, hred] at hblue
  exact hblue

end connie_blue_markers_l310_310860


namespace plane_equation_l310_310909

noncomputable def equation_of_plane (x y z : ℝ) :=
  3 * x + 2 * z - 1

theorem plane_equation :
  ∀ (x y z : ℝ), 
    (∃ (p : ℝ × ℝ × ℝ), p = (1, 2, -1) ∧ 
                         (∃ (n : ℝ × ℝ × ℝ), n = (3, 0, 2) ∧ 
                                              equation_of_plane x y z = 0)) :=
by
  -- The statement setup is done. The proof is not included as per instructions.
  sorry

end plane_equation_l310_310909


namespace length_of_garden_l310_310259

-- Definitions based on conditions
def P : ℕ := 600
def b : ℕ := 200

-- Theorem statement
theorem length_of_garden : ∃ L : ℕ, 2 * (L + b) = P ∧ L = 100 :=
by
  existsi 100
  simp
  sorry

end length_of_garden_l310_310259


namespace tan_x_plus_pi_over_4_l310_310251

theorem tan_x_plus_pi_over_4 (x : ℝ) (hx : Real.tan x = 2) : Real.tan (x + Real.pi / 4) = -3 :=
by
  sorry

end tan_x_plus_pi_over_4_l310_310251


namespace find_h_l310_310745

-- Define the polynomial f(x)
def f (x : ℤ) := x^4 - 2 * x^3 + x - 1

-- Define the condition that f(x) + h(x) = 3x^2 + 5x - 4
def condition (f h : ℤ → ℤ) := ∀ x, f x + h x = 3 * x^2 + 5 * x - 4

-- Define the solution for h(x) to be proved
def h_solution (x : ℤ) := -x^4 + 2 * x^3 + 3 * x^2 + 4 * x - 3

-- State the theorem to be proved
theorem find_h (h : ℤ → ℤ) (H : condition f h) : h = h_solution :=
by
  sorry

end find_h_l310_310745


namespace resulting_shape_is_cone_l310_310501

-- Assume we have a right triangle
structure right_triangle (α β γ : ℝ) : Prop :=
  (is_right : γ = π / 2)
  (sum_of_angles : α + β + γ = π)
  (acute_angles : α < π / 2 ∧ β < π / 2)

-- Assume we are rotating around one of the legs
def rotate_around_leg (α β : ℝ) : Prop := sorry

theorem resulting_shape_is_cone (α β γ : ℝ) (h : right_triangle α β γ) :
  ∃ (shape : Type), rotate_around_leg α β → shape = cone :=
by
  sorry

end resulting_shape_is_cone_l310_310501


namespace total_cost_is_eight_x_l310_310302

-- Definitions of cost variables based on conditions
variable (x : ℝ) -- Cost of shorts

-- Cost conditions
variable (shirt_cost : ℝ) (boot_cost : ℝ) (shin_guard_cost : ℝ)
variable (c1 : x + shirt_cost = 2 * x)
variable (c2 : x + boot_cost = 5 * x)
variable (c3 : x + shin_guard_cost = 3 * x)

-- To prove that the total cost is 8 times the cost of shorts
theorem total_cost_is_eight_x
  (c1 : x + shirt_cost = 2 * x)
  (c2 : x + boot_cost = 5 * x)
  (c3 : x + shin_guard_cost = 3 * x) :
  x + shirt_cost + boot_cost + shin_guard_cost = 8 * x := 
by
  sorry

end total_cost_is_eight_x_l310_310302


namespace new_ratio_boarders_to_day_students_l310_310001

-- Given conditions
def initial_ratio_boarders_to_day_students : ℚ := 2 / 5
def initial_boarders : ℕ := 120
def new_boarders : ℕ := 30

-- Derived definitions
def initial_day_students : ℕ :=
  (initial_boarders * (5 : ℕ)) / 2

def total_boarders : ℕ := initial_boarders + new_boarders
def total_day_students : ℕ := initial_day_students

-- Theorem to prove the new ratio
theorem new_ratio_boarders_to_day_students : total_boarders / total_day_students = 1 / 2 :=
  sorry

end new_ratio_boarders_to_day_students_l310_310001


namespace value_of_expression_l310_310826

theorem value_of_expression : (15 + 5)^2 - (15^2 + 5^2) = 150 := by
  sorry

end value_of_expression_l310_310826


namespace find_a_values_for_eccentricity_l310_310340

theorem find_a_values_for_eccentricity (a : ℝ) : 
  ( ∃ a : ℝ, ((∀ x y : ℝ, (x^2 / (a+8) + y^2 / 9 = 1)) ∧ (e = 1/2) ) 
  → (a = 4 ∨ a = -5/4)) := 
sorry

end find_a_values_for_eccentricity_l310_310340


namespace cost_of_purchase_l310_310316

theorem cost_of_purchase (x : ℝ) (T_shirt boots shin_guards : ℝ) 
  (h1 : x + T_shirt = 2 * x)
  (h2 : x + boots = 5 * x)
  (h3 : x + shin_guards = 3 * x) :
  x + T_shirt + boots + shin_guards = 8 * x :=
begin
  sorry
end

end cost_of_purchase_l310_310316


namespace geometric_seq_a6_l310_310953

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ q, ∀ n, a (n + 1) = a n * q

theorem geometric_seq_a6 {a : ℕ → ℝ} (h : geometric_sequence a) (h1 : a 1 * a 3 = 4) (h2 : a 4 = 4) : a 6 = 8 :=
sorry

end geometric_seq_a6_l310_310953


namespace quadratic_properties_l310_310243

def quadratic_function (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 3

theorem quadratic_properties :
  -- 1. The parabola opens downwards.
  (∀ x : ℝ, quadratic_function x < quadratic_function (x + 1) → false) ∧
  -- 2. The axis of symmetry is x = 1.
  (∀ x : ℝ, ∃ y : ℝ, quadratic_function x = quadratic_function y → x = y ∨ x + y = 2) ∧
  -- 3. The vertex coordinates are (1, 5).
  (quadratic_function 1 = 5) ∧
  -- 4. y decreases for x > 1.
  (∀ x : ℝ, x > 1 → quadratic_function x < quadratic_function (x - 1)) :=
by
  sorry

end quadratic_properties_l310_310243


namespace cost_of_purchase_l310_310314

theorem cost_of_purchase (x : ℝ) (T_shirt boots shin_guards : ℝ) 
  (h1 : x + T_shirt = 2 * x)
  (h2 : x + boots = 5 * x)
  (h3 : x + shin_guards = 3 * x) :
  x + T_shirt + boots + shin_guards = 8 * x :=
begin
  sorry
end

end cost_of_purchase_l310_310314


namespace gcd_1248_585_l310_310978

theorem gcd_1248_585 : Nat.gcd 1248 585 = 39 := by
  sorry

end gcd_1248_585_l310_310978


namespace sqrt_floor_squared_l310_310873

theorem sqrt_floor_squared (h1 : 7^2 = 49) (h2 : 8^2 = 64) (h3 : 7 < Real.sqrt 50) (h4 : Real.sqrt 50 < 8) : (Int.floor (Real.sqrt 50))^2 = 49 :=
by
  sorry

end sqrt_floor_squared_l310_310873


namespace sarah_photos_l310_310671

theorem sarah_photos (photos_Cristina photos_John photos_Clarissa total_slots : ℕ)
  (hCristina : photos_Cristina = 7)
  (hJohn : photos_John = 10)
  (hClarissa : photos_Clarissa = 14)
  (hTotal : total_slots = 40) :
  ∃ photos_Sarah, photos_Sarah = total_slots - (photos_Cristina + photos_John + photos_Clarissa) ∧ photos_Sarah = 9 :=
by
  sorry

end sarah_photos_l310_310671


namespace marvin_birthday_friday_l310_310455

open Nat

def is_leap_year (year : ℕ) : Prop :=
  year % 4 = 0 ∧ (year % 100 ≠ 0 ∨ year % 400 = 0)

noncomputable def day_of_week (year month day : ℕ) : ℕ :=
  Date.civil_to_gregorian_transformed (mk_civil year month day) mod 7

theorem marvin_birthday_friday (year : ℕ) (by2013 : day_of_week 2013 5 27 = 1) :
  ∃ year, year > 2013 ∧ day_of_week year 5 27 = 5 :=
by
  have day_increment : ∀ n, day_of_week (2013 + n) 5 27 = (day_of_week 2013 5 27 + finset.range (n).sum (λ i, if is_leap_year (2013 + i) then 2 else 1)) % 7 :=
    λ n, sorry  -- method to calculate each year's increment

  existsi 2016
  split
  · linarith
  · simp [day_of_week, by2013, day_increment]
    sorry -- proof that the specific day of week for May 27, 2016 is a Friday

end marvin_birthday_friday_l310_310455


namespace sum_of_fractions_l310_310991

theorem sum_of_fractions : (1 / 1) + (2 / 2) + (3 / 3) = 3 := 
by 
  norm_num

end sum_of_fractions_l310_310991


namespace trail_mix_total_weight_l310_310853

noncomputable def peanuts : ℝ := 0.16666666666666666
noncomputable def chocolate_chips : ℝ := 0.16666666666666666
noncomputable def raisins : ℝ := 0.08333333333333333

theorem trail_mix_total_weight :
  peanuts + chocolate_chips + raisins = 0.41666666666666663 :=
by
  unfold peanuts chocolate_chips raisins
  sorry

end trail_mix_total_weight_l310_310853


namespace total_cost_is_eight_times_l310_310324

theorem total_cost_is_eight_times (x : ℝ) 
  (h1 : ∀ t, x + t = 2 * x)
  (h2 : ∀ b, x + b = 5 * x)
  (h3 : ∀ s, x + s = 3 * x) :
  ∃ t b s, x + t + b + s = 8 * x :=
by
  sorry

end total_cost_is_eight_times_l310_310324


namespace three_secretaries_project_l310_310468

theorem three_secretaries_project (t1 t2 t3 : ℕ) 
  (h1 : t1 / t2 = 1 / 2) 
  (h2 : t1 / t3 = 1 / 5) 
  (h3 : t3 = 75) : 
  t1 + t2 + t3 = 120 := 
  by 
    sorry

end three_secretaries_project_l310_310468


namespace fuel_consumption_l310_310987

-- Define the initial conditions based on the problem
variable (s Q : ℝ)

-- Distance and fuel data points
def data_points : List (ℝ × ℝ) := [(0, 50), (100, 42), (200, 34), (300, 26), (400, 18)]

-- Define the function Q and required conditions
theorem fuel_consumption :
  (∀ p ∈ data_points, ∃ k b, Q = k * s + b ∧
    ((p.1 = 0 → b = 50) ∧
     (p.1 = 100 → Q = 42 → k = -0.08))) :=
by
  sorry

end fuel_consumption_l310_310987


namespace point_on_line_l310_310161

theorem point_on_line : ∀ (t : ℤ), 
  (∃ m : ℤ, (6 - 2) * m = 20 - 8 ∧ (10 - 6) * m = 32 - 20) →
  (∃ b : ℤ, 8 - 2 * m = b) →
  t = m * 35 + b → t = 107 :=
by
  sorry

end point_on_line_l310_310161


namespace general_formula_S_gt_a_l310_310959

variable {n : ℕ}
variable {a S : ℕ → ℤ}
variable {d : ℤ}

-- Definitions of the arithmetic sequence and sums
def a_n (n : ℕ) : ℤ := a n
def S_n (n : ℕ) : ℤ := (n * (2 * a 1 + (n - 1) * d)) / 2

-- Conditions from the problem statement
axiom condition_1 : a_n 3 = S_n 5
axiom condition_2 : a_n 2 * a_n 4 = S_n 4

-- Problem 1: General formula for the sequence
theorem general_formula : (∀ n, a_n n = 2 * n - 6) := by
  sorry

-- Problem 2: Smallest value of n for which S_n > a_n
theorem S_gt_a : ∃ n ≥ 7, S_n n > a_n n := by
  sorry

end general_formula_S_gt_a_l310_310959


namespace correct_propositions_l310_310710

-- Definitions of the conditions in the Math problem

variable (triangle_outside_plane : Prop)
variable (triangle_side_intersections_collinear : Prop)
variable (parallel_lines_coplanar : Prop)
variable (noncoplanar_points_planes : Prop)

-- Math proof problem statement
theorem correct_propositions :
  (triangle_outside_plane ∧ 
   parallel_lines_coplanar ∧ 
   ¬noncoplanar_points_planes) →
  2 = 2 :=
by
  sorry

end correct_propositions_l310_310710


namespace factorize_expression_l310_310892

theorem factorize_expression (a x y : ℤ) : a^2 * (x - y) + 4 * (y - x) = (x - y) * (a + 2) * (a - 2) :=
by
  sorry

end factorize_expression_l310_310892


namespace suresh_investment_correct_l310_310974

noncomputable def suresh_investment
  (ramesh_investment : ℝ)
  (total_profit : ℝ)
  (ramesh_profit_share : ℝ)
  : ℝ := sorry

theorem suresh_investment_correct
  (ramesh_investment : ℝ := 40000)
  (total_profit : ℝ := 19000)
  (ramesh_profit_share : ℝ := 11875)
  : suresh_investment ramesh_investment total_profit ramesh_profit_share = 24000 := sorry

end suresh_investment_correct_l310_310974


namespace proof_inequality_l310_310694

noncomputable def problem (a b c d : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a * b * c * d = 1 → a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d

theorem proof_inequality (a b c d : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a * b * c * d = 1) :
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
sorry

end proof_inequality_l310_310694


namespace gcd_sum_abcde_edcba_l310_310342

-- Definition to check if digits are consecutive
def consecutive_digits (a b c d e : ℤ) : Prop :=
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4

-- Definition of the five-digit number in the form abcde
def abcde (a b c d e : ℤ) : ℤ :=
  10000 * a + 1000 * b + 100 * c + 10 * d + e

-- Definition of the five-digit number in the form edcba
def edcba (a b c d e : ℤ) : ℤ :=
  10000 * e + 1000 * d + 100 * c + 10 * b + a

-- Definition which sums both abcde and edcba
def sum_abcde_edcba (a b c d e : ℤ) : ℤ :=
  abcde a b c d e + edcba a b c d e

-- Lean theorem statement for the problem
theorem gcd_sum_abcde_edcba (a b c d e : ℤ) (h : consecutive_digits a b c d e) :
  Int.gcd (sum_abcde_edcba a b c d e) 11211 = 11211 :=
by
  sorry

end gcd_sum_abcde_edcba_l310_310342


namespace max_chords_intersecting_line_l310_310508

theorem max_chords_intersecting_line (A : Fin 2017 → Type) :
  ∃ k : ℕ, (k ≤ 2016 ∧ ∃ m : ℕ, (m = k * (2016 - k) + 2016) ∧ m = 1018080) :=
sorry

end max_chords_intersecting_line_l310_310508


namespace min_troublemakers_l310_310795

theorem min_troublemakers (n : ℕ) (truth_teller liar : ℕ → Prop) 
  (total_students : n = 29)
  (exists_one_liar : ∀ i : ℕ, liar i = true → ((liar (i - 1) = true ∨ liar (i + 1) = true))) -- assume cyclic seating
  (exists_two_liar : ∀ i : ℕ, truth_teller i = true → (liar (i - 1) = true ∧ liar (i + 1) = true)) :
  (∃ k : ℕ, k = 10 ∧ (∀ i : ℕ, i < 29 -> liar i = true -> k ≥ i)) :=
sorry

end min_troublemakers_l310_310795


namespace asia_paid_140_l310_310665

noncomputable def original_price : ℝ := 350
noncomputable def discount_percentage : ℝ := 0.60
noncomputable def discount_amount : ℝ := original_price * discount_percentage
noncomputable def final_price : ℝ := original_price - discount_amount

theorem asia_paid_140 : final_price = 140 := by
  unfold final_price
  unfold discount_amount
  unfold original_price
  unfold discount_percentage
  sorry

end asia_paid_140_l310_310665


namespace no_minimum_of_f_over_M_l310_310025

/-- Define the domain M for the function y = log(3 - 4x + x^2) -/
def domain_M (x : ℝ) : Prop := (x > 3 ∨ x < 1)

/-- Define the function f(x) = 2x + 2 - 3 * 4^x -/
noncomputable def f (x : ℝ) : ℝ := 2 * x + 2 - 3 * 4^x

/-- The theorem statement:
    Prove that f(x) does not have a minimum value for x in the domain M -/
theorem no_minimum_of_f_over_M : ¬ ∃ x ∈ {x | domain_M x}, ∀ y ∈ {x | domain_M x}, f x ≤ f y := sorry

end no_minimum_of_f_over_M_l310_310025


namespace line_through_intersection_parallel_to_y_axis_l310_310052

theorem line_through_intersection_parallel_to_y_axis:
  ∃ x, (∃ y, 3 * x + 2 * y - 5 = 0 ∧ x - 3 * y + 2 = 0) ∧
       (x = 1) :=
sorry

end line_through_intersection_parallel_to_y_axis_l310_310052


namespace sandy_money_l310_310147

theorem sandy_money (X : ℝ) (h1 : 0.70 * X = 224) : X = 320 := 
by {
  sorry
}

end sandy_money_l310_310147


namespace find_annual_interest_rate_l310_310678

noncomputable def compound_interest_problem : Prop :=
  ∃ (r : ℝ),
    let P := 8000
    let CI := 3109
    let t := 2.3333
    let A := 11109
    let n := 1
    A = P * (1 + r/n)^(n*t) ∧ r = 0.1505

theorem find_annual_interest_rate : compound_interest_problem :=
by sorry

end find_annual_interest_rate_l310_310678


namespace cleaning_time_ratio_l310_310136

/-- 
Given that Lilly and Fiona together take a total of 480 minutes to clean a room and Fiona
was cleaning for 360 minutes, prove that the ratio of the time Lilly spent cleaning 
to the total time spent cleaning the room is 1:4.
-/
theorem cleaning_time_ratio (total_time minutes Fiona_time : ℕ) 
  (h1 : total_time = 480)
  (h2 : Fiona_time = 360) : 
  (total_time - Fiona_time) / total_time = 1 / 4 :=
by
  sorry

end cleaning_time_ratio_l310_310136


namespace rectangle_area_l310_310654

theorem rectangle_area (square_area : ℝ) (rectangle_length_ratio : ℝ) (square_area_eq : square_area = 36)
  (rectangle_length_ratio_eq : rectangle_length_ratio = 3) :
  ∃ (rectangle_area : ℝ), rectangle_area = 108 :=
by
  -- Extract the side length of the square from its area
  let side_length := real.sqrt square_area
  have side_length_eq : side_length = 6, from calc
    side_length = real.sqrt 36 : by rw [square_area_eq]
    ... = 6 : real.sqrt_eq 6 (by norm_num)
  -- Calculate the rectangle's width
  let width := side_length
  -- Calculate the rectangle's length
  let length := rectangle_length_ratio * width
  -- Calculate the area of the rectangle
  let area := width * length
  -- Prove the area is 108
  use area
  calc area
    = 6 * (3 * 6) : by rw [side_length_eq, rectangle_length_ratio_eq]
    ... = 108 : by norm_num

end rectangle_area_l310_310654


namespace cows_number_l310_310428

theorem cows_number (D C : ℕ) (L H : ℕ) 
  (h1 : L = 2 * D + 4 * C)
  (h2 : H = D + C)
  (h3 : L = 2 * H + 12) 
  : C = 6 := 
by
  sorry

end cows_number_l310_310428


namespace long_furred_brown_dogs_l310_310427

-- Definitions based on given conditions
def T : ℕ := 45
def L : ℕ := 36
def B : ℕ := 27
def N : ℕ := 8

-- The number of long-furred brown dogs (LB) that needs to be proved
def LB : ℕ := 26

-- Lean 4 statement to prove LB
theorem long_furred_brown_dogs :
  L + B - LB = T - N :=
by 
  unfold T L B N LB -- we unfold definitions to simplify the theorem
  sorry

end long_furred_brown_dogs_l310_310427


namespace ball_bounce_l310_310507

theorem ball_bounce :
  ∃ b : ℕ, 324 * (3 / 4) ^ b < 40 ∧ b = 8 :=
by
  have : (3 / 4 : ℝ) < 1 := by norm_num
  have h40_324 : (40 : ℝ) / 324 = 10 / 81 := by norm_num
  sorry

end ball_bounce_l310_310507


namespace quadratic_inequality_a_value_l310_310076

theorem quadratic_inequality_a_value (a t : ℝ)
  (h_a1 : ∀ x : ℝ, t * x ^ 2 - 6 * x + t ^ 2 = 0 → (x = a ∨ x = 1))
  (h_t : t < 0) :
  a = -3 :=
by
  sorry

end quadratic_inequality_a_value_l310_310076


namespace probability_of_selecting_green_ball_l310_310527

-- Declare the probability of selecting each container
def prob_of_selecting_container := (1 : ℚ) / 4

-- Declare the number of balls in each container
def balls_in_container_A := 10
def balls_in_container_B := 14
def balls_in_container_C := 14
def balls_in_container_D := 10

-- Declare the number of green balls in each container
def green_balls_in_A := 6
def green_balls_in_B := 6
def green_balls_in_C := 6
def green_balls_in_D := 7

-- Calculate the probability of drawing a green ball from each container
def prob_green_from_A := (green_balls_in_A : ℚ) / balls_in_container_A
def prob_green_from_B := (green_balls_in_B : ℚ) / balls_in_container_B
def prob_green_from_C := (green_balls_in_C : ℚ) / balls_in_container_C
def prob_green_from_D := (green_balls_in_D : ℚ) / balls_in_container_D

-- Calculate the total probability of drawing a green ball
def total_prob_green :=
  prob_of_selecting_container * prob_green_from_A +
  prob_of_selecting_container * prob_green_from_B +
  prob_of_selecting_container * prob_green_from_C +
  prob_of_selecting_container * prob_green_from_D

theorem probability_of_selecting_green_ball : total_prob_green = 13 / 28 :=
by sorry

end probability_of_selecting_green_ball_l310_310527


namespace min_troublemakers_l310_310804

theorem min_troublemakers (students : Finset ℕ) (table : List ℕ) (is_liar : ℕ → Bool) :
  students.card = 29 ∧
  (∀ s ∈ students, 
    (∃ i j : ℕ, 
      table.nth i = some s ∧ table.nth ((i + 1) % 29) = some j ∧
      (∃ k : ℕ, 
        (is_liar s = false → count (/=is_liar k liar) = 0 ∧
        is_liar s = true → count (/=is_liar k liar) = 2))) → 
  (∃! l ∈ students, count (/=is_liar l liar) = 1))) → 
  ∃ t : ℕ, t = 10 :=
sorry

end min_troublemakers_l310_310804


namespace triangle_side_length_l310_310739

variable (A C : ℝ) (a c b : ℝ)

theorem triangle_side_length (h1 : c = 48) (h2 : a = 27) (h3 : C = 3 * A) : b = 35 := by
  sorry

end triangle_side_length_l310_310739


namespace product_of_two_numbers_l310_310625

theorem product_of_two_numbers (x y : ℕ) (h₁ : x + y = 16) (h₂ : x^2 + y^2 = 200) : x * y = 28 :=
by
  sorry

end product_of_two_numbers_l310_310625


namespace bisection_method_third_interval_l310_310174

noncomputable def bisection_method_interval (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : (ℝ × ℝ) :=
  sorry  -- Definition of the interval using bisection method, but this is not necessary.

theorem bisection_method_third_interval (f : ℝ → ℝ) :
  (bisection_method_interval f (-2) 4 3) = (-1/2, 1) :=
sorry

end bisection_method_third_interval_l310_310174


namespace distinct_solutions_diff_l310_310747

theorem distinct_solutions_diff (r s : ℝ) 
  (h1 : r ≠ s) 
  (h2 : (5*r - 15)/(r^2 + 3*r - 18) = r + 3) 
  (h3 : (5*s - 15)/(s^2 + 3*s - 18) = s + 3) 
  (h4 : r > s) : 
  r - s = 13 :=
sorry

end distinct_solutions_diff_l310_310747


namespace area_of_rectangle_is_108_l310_310657

theorem area_of_rectangle_is_108 (s w l : ℕ) (h₁ : s * s = 36) (h₂ : w = s) (h₃ : l = 3 * w) : w * l = 108 :=
by
  -- This is a placeholder for a detailed proof.
  sorry

end area_of_rectangle_is_108_l310_310657


namespace trees_not_pine_trees_l310_310983

theorem trees_not_pine_trees
  (total_trees : ℕ)
  (percentage_pine : ℝ)
  (number_pine : ℕ)
  (number_not_pine : ℕ)
  (h_total : total_trees = 350)
  (h_percentage : percentage_pine = 0.70)
  (h_pine : number_pine = percentage_pine * total_trees)
  (h_not_pine : number_not_pine = total_trees - number_pine)
  : number_not_pine = 105 :=
sorry

end trees_not_pine_trees_l310_310983


namespace negate_proposition_l310_310158

def p (x : ℝ) : Prop := x^2 + x - 6 > 0
def q (x : ℝ) : Prop := x > 2 ∨ x < -3

def neg_p (x : ℝ) : Prop := x^2 + x - 6 ≤ 0
def neg_q (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 2

theorem negate_proposition (x : ℝ) :
  (¬ (p x → q x)) ↔ (neg_p x → neg_q x) :=
by unfold p q neg_p neg_q; apply sorry

end negate_proposition_l310_310158


namespace last_day_of_third_quarter_l310_310611

def is_common_year (year: Nat) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0) 

def days_in_month (year: Nat) (month: Nat) : Nat :=
  if month = 2 then 28
  else if month = 4 ∨ month = 6 ∨ month = 9 ∨ month = 11 then 30
  else 31

def last_day_of_month (year: Nat) (month: Nat) : Nat :=
  days_in_month year month

theorem last_day_of_third_quarter (year: Nat) (h : is_common_year year) : last_day_of_month year 9 = 30 :=
by
  sorry

end last_day_of_third_quarter_l310_310611


namespace perfect_squares_between_100_and_400_l310_310106

theorem perfect_squares_between_100_and_400 :
  let n := 11
  let m := 19
  list.count (list.map (λ x, x * x) (list.range (m - n + 1) + (fun c => c + n))) = 9 := by
    sorry  -- Proof omitted

end perfect_squares_between_100_and_400_l310_310106


namespace min_troublemakers_l310_310805

theorem min_troublemakers (students : Finset ℕ) (table : List ℕ) (is_liar : ℕ → Bool) :
  students.card = 29 ∧
  (∀ s ∈ students, 
    (∃ i j : ℕ, 
      table.nth i = some s ∧ table.nth ((i + 1) % 29) = some j ∧
      (∃ k : ℕ, 
        (is_liar s = false → count (/=is_liar k liar) = 0 ∧
        is_liar s = true → count (/=is_liar k liar) = 2))) → 
  (∃! l ∈ students, count (/=is_liar l liar) = 1))) → 
  ∃ t : ℕ, t = 10 :=
sorry

end min_troublemakers_l310_310805


namespace cos_product_equals_one_over_128_l310_310686

theorem cos_product_equals_one_over_128 :
  (Real.cos (Real.pi / 15)) *
  (Real.cos (2 * Real.pi / 15)) *
  (Real.cos (3 * Real.pi / 15)) *
  (Real.cos (4 * Real.pi / 15)) *
  (Real.cos (5 * Real.pi / 15)) *
  (Real.cos (6 * Real.pi / 15)) *
  (Real.cos (7 * Real.pi / 15))
  = 1 / 128 := 
sorry

end cos_product_equals_one_over_128_l310_310686


namespace correct_distribution_l310_310264

-- Define the conditions
def num_students : ℕ := 40
def ratio_A_to_B : ℚ := 0.8
def ratio_C_to_B : ℚ := 1.2

-- Definitions for the number of students earning each grade
def num_B (x : ℕ) : ℕ := x
def num_A (x : ℕ) : ℕ := Nat.floor (ratio_A_to_B * x)
def num_C (x : ℕ) : ℕ := Nat.ceil (ratio_C_to_B * x)

-- Prove the distribution is correct
theorem correct_distribution :
  ∃ x : ℕ, num_A x + num_B x + num_C x = num_students ∧ 
           num_A x = 10 ∧ num_B x = 14 ∧ num_C x = 16 :=
by
  sorry

end correct_distribution_l310_310264


namespace rate_of_pipe_B_l310_310459

-- Definitions based on conditions
def tank_capacity : ℕ := 850
def pipe_A_rate : ℕ := 40
def pipe_C_rate : ℕ := 20
def cycle_time : ℕ := 3
def full_time : ℕ := 51

-- Prove that the rate of pipe B is 30 liters per minute
theorem rate_of_pipe_B (B : ℕ) : 
  (17 * (B + 20) = 850) → B = 30 := 
by 
  introv h1
  sorry

end rate_of_pipe_B_l310_310459


namespace determine_xyz_l310_310672

variables {x y z : ℝ}

theorem determine_xyz (h : (x - y - 3)^2 + (y - z)^2 + (x - z)^2 = 3) : 
  x = z + 1 ∧ y = z - 1 := 
sorry

end determine_xyz_l310_310672


namespace distance_per_trip_l310_310755

--  Define the conditions as assumptions
variables (total_distance : ℝ) (num_trips : ℝ)
axiom h_total_distance : total_distance = 120
axiom h_num_trips : num_trips = 4

-- Define the question converted into a statement to be proven
theorem distance_per_trip : total_distance / num_trips = 30 :=
by
  -- Placeholder for the actual proof
  sorry

end distance_per_trip_l310_310755


namespace remainder_3x_minus_6_divides_P_l310_310172

def P(x : ℝ) : ℝ := 5 * x^8 - 3 * x^7 + 2 * x^6 - 8 * x^4 + 3 * x^3 - 5
def D(x : ℝ) : ℝ := 3 * x - 6

theorem remainder_3x_minus_6_divides_P :
  P 2 = 915 :=
by
  sorry

end remainder_3x_minus_6_divides_P_l310_310172


namespace geometric_sequence_term_l310_310435

/-
Prove that the 303rd term in a geometric sequence with the first term a1 = 5 and the second term a2 = -10 is 5 * 2^302.
-/

theorem geometric_sequence_term :
  let a1 := 5
  let a2 := -10
  let r := a2 / a1
  let n := 303
  let a_n := a1 * r^(n-1)
  a_n = 5 * 2^302 :=
by
  let a1 := 5
  let a2 := -10
  let r := a2 / a1
  let n := 303
  have h1 : a1 * r^(n-1) = 5 * 2^302 := sorry
  exact h1

end geometric_sequence_term_l310_310435


namespace area_of_region_l310_310359

theorem area_of_region : 
  (∀ x y : ℝ, x^2 + y^2 - 8*x + 6*y = 0 → 
     let a := (x - 4)^2 + (y + 3)^2 
     (a = 25) ∧ ∃ r : ℝ, r = 5 ∧ (π * r^2 = 25 * π)) := 
sorry

end area_of_region_l310_310359


namespace perfect_squares_in_range_100_400_l310_310104

theorem perfect_squares_in_range_100_400 : ∃ n : ℕ, (∀ m, 100 ≤ m^2 → m^2 ≤ 400 → m^2 = (m - 10 + 1)^2) ∧ n = 9 := 
by
  sorry

end perfect_squares_in_range_100_400_l310_310104


namespace mean_height_is_approx_correct_l310_310482

def heights : List ℕ := [120, 123, 127, 132, 133, 135, 140, 142, 145, 148, 152, 155, 158, 160]

def mean_height : ℚ := heights.sum / heights.length

theorem mean_height_is_approx_correct : 
  abs (mean_height - 140.71) < 0.01 := 
by
  sorry

end mean_height_is_approx_correct_l310_310482


namespace intersection_M_N_l310_310555

def M : Set ℝ := { x : ℝ | x^2 > 4 }
def N : Set ℝ := { x : ℝ | x = -3 ∨ x = -2 ∨ x = 2 ∨ x = 3 ∨ x = 4 }

theorem intersection_M_N : M ∩ N = { x : ℝ | x = -3 ∨ x = 3 ∨ x = 4 } :=
by
  sorry

end intersection_M_N_l310_310555


namespace commute_times_variance_l310_310193

theorem commute_times_variance (x y : ℝ) :
  (x + y + 10 + 11 + 9) / 5 = 10 ∧
  ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2 →
  |x - y| = 4 :=
by
  sorry

end commute_times_variance_l310_310193


namespace right_triangle_angles_l310_310950

theorem right_triangle_angles (a b S : ℝ) (hS : S = 1 / 2 * a * b) (h : (a + b) ^ 2 = 8 * S) :
  ∃ θ₁ θ₂ θ₃ : ℝ, θ₁ = 45 ∧ θ₂ = 45 ∧ θ₃ = 90 :=
by {
  sorry
}

end right_triangle_angles_l310_310950


namespace laundry_loads_l310_310081

theorem laundry_loads (usual_price : ℝ) (sale_price : ℝ) (cost_per_load : ℝ) (total_loads_2_bottles : ℝ) :
  usual_price = 25 ∧ sale_price = 20 ∧ cost_per_load = 0.25 ∧ total_loads_2_bottles = (2 * sale_price) / cost_per_load →
  (total_loads_2_bottles / 2) = 80 :=
by
  sorry

end laundry_loads_l310_310081


namespace sum_of_valid_ns_l310_310016

-- Define a function for binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def condition1 (n : ℕ) : Prop := binom 25 n + binom 25 12 = binom 26 13

-- Define the theorem stating the sum of all valid n satisfying condition1 is 24
theorem sum_of_valid_ns : (∑ n in Finset.filter condition1 (Finset.range 26), n) = 24 := by
  sorry

end sum_of_valid_ns_l310_310016


namespace susan_betsy_ratio_l310_310852

theorem susan_betsy_ratio (betsy_wins : ℕ) (helen_wins : ℕ) (susan_wins : ℕ) (total_wins : ℕ)
  (h1 : betsy_wins = 5)
  (h2 : helen_wins = 2 * betsy_wins)
  (h3 : betsy_wins + helen_wins + susan_wins = total_wins)
  (h4 : total_wins = 30) :
  susan_wins / betsy_wins = 3 := by
  sorry

end susan_betsy_ratio_l310_310852


namespace application_methods_count_l310_310610

theorem application_methods_count (n_graduates m_universities : ℕ) (h_graduates : n_graduates = 5) (h_universities : m_universities = 3) :
  (m_universities ^ n_graduates) = 243 :=
by
  rw [h_graduates, h_universities]
  show 3 ^ 5 = 243
  sorry

end application_methods_count_l310_310610


namespace find_abc_value_l310_310115

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom h1 : a + 1 / b = 5
axiom h2 : b + 1 / c = 2
axiom h3 : c + 1 / a = 9 / 4

theorem find_abc_value : a * b * c = (7 + Real.sqrt 21) / 8 :=
by
  sorry

end find_abc_value_l310_310115


namespace intersection_of_circle_and_line_l310_310410

theorem intersection_of_circle_and_line 
  (α : ℝ) 
  (x y : ℝ)
  (h1 : x = Real.cos α) 
  (h2 : y = 1 + Real.sin α) 
  (h3 : y = 1) :
  (x, y) = (1, 1) :=
by
  sorry

end intersection_of_circle_and_line_l310_310410


namespace total_cost_is_eight_x_l310_310306

-- Definitions of cost variables based on conditions
variable (x : ℝ) -- Cost of shorts

-- Cost conditions
variable (shirt_cost : ℝ) (boot_cost : ℝ) (shin_guard_cost : ℝ)
variable (c1 : x + shirt_cost = 2 * x)
variable (c2 : x + boot_cost = 5 * x)
variable (c3 : x + shin_guard_cost = 3 * x)

-- To prove that the total cost is 8 times the cost of shorts
theorem total_cost_is_eight_x
  (c1 : x + shirt_cost = 2 * x)
  (c2 : x + boot_cost = 5 * x)
  (c3 : x + shin_guard_cost = 3 * x) :
  x + shirt_cost + boot_cost + shin_guard_cost = 8 * x := 
by
  sorry

end total_cost_is_eight_x_l310_310306


namespace exist_a_b_not_triangle_l310_310674

theorem exist_a_b_not_triangle (h₁ : ∀ a b : ℕ, (a > 1000) → (b > 1000) →
  ∃ c : ℕ, (∃ (k : ℕ), c = k * k) →
  ¬ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a))) :
  ∃ (a b : ℕ), (a > 1000 ∧ b > 1000) ∧ 
  ∀ c : ℕ, (∃ (k : ℕ), c = k * k) →
  ¬ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
sorry

end exist_a_b_not_triangle_l310_310674


namespace polynomial_simplification_l310_310830

variable (x : ℝ)

theorem polynomial_simplification :
  (3 * x^2 + 5 * x + 9) * (x + 2) - (x + 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x + 2) * (x + 4) =
  6 * x^3 - 28 * x^2 - 59 * x + 42 :=
by
  sorry

end polynomial_simplification_l310_310830


namespace quadratic_sum_l310_310262

theorem quadratic_sum (x : ℝ) :
  ∃ a h k : ℝ, (5*x^2 - 10*x - 3 = a*(x - h)^2 + k) ∧ (a + h + k = -2) :=
sorry

end quadratic_sum_l310_310262


namespace slope_of_tangent_line_at_x_2_l310_310682

noncomputable def curve (x : ℝ) : ℝ := x^2 + 3*x

theorem slope_of_tangent_line_at_x_2 : (deriv curve 2) = 7 := by
  sorry

end slope_of_tangent_line_at_x_2_l310_310682


namespace number_of_subsets_of_three_element_set_l310_310159

theorem number_of_subsets_of_three_element_set :
  ∃ (S : Finset ℕ), S.card = 3 ∧ S.powerset.card = 8 :=
sorry

end number_of_subsets_of_three_element_set_l310_310159


namespace product_mod_7_l310_310209

theorem product_mod_7 : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
by
  have h1 : 2021 % 7 = 6 := by sorry
  have h2 : 2022 % 7 = 0 := by sorry
  have h3 : 2023 % 7 = 1 := by sorry
  have h4 : 2024 % 7 = 2 := by sorry
  sorry

end product_mod_7_l310_310209


namespace part1_f0_f1_part1_f_neg1_f2_part1_f_neg2_f3_part2_conjecture_l310_310452

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem part1_f0_f1 : f 0 + f 1 = Real.sqrt 3 / 3 := sorry

theorem part1_f_neg1_f2 : f (-1) + f 2 = Real.sqrt 3 / 3 := sorry

theorem part1_f_neg2_f3 : f (-2) + f 3 = Real.sqrt 3 / 3 := sorry

theorem part2_conjecture (x1 x2 : ℝ) (h : x1 + x2 = 1) : f x1 + f x2 = Real.sqrt 3 / 3 := sorry

end part1_f0_f1_part1_f_neg1_f2_part1_f_neg2_f3_part2_conjecture_l310_310452


namespace random_variable_point_of_increase_l310_310512

-- Assuming Real numbers, Probability space and measurable function details
variable {μ : MeasureTheory.Measure ℝ}

-- F is the distribution function
def is_distribution_function (F : ℝ → ℝ) : Prop :=
  ∀ x : ℝ , (0 ≤ F x) ∧ (F x ≤ 1) ∧ monotone F ∧ (Filter.atTop.tendsto F (𝓝 1))

def point_of_increase (F : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ ε > 0, F (x - ε) < F (x + ε)

noncomputable def ξ : MeasureTheory.ProbabilityMeasure ℝ := sorry

theorem random_variable_point_of_increase (F : ℝ → ℝ) (ξ : MeasureTheory.ProbabilityMeasure ℝ) 
  (hF : is_distribution_function F) : 
  (μ { ω : ℝ | ∃ε > 0, F (ω - ε) < F (ω + ε) }) = 1 := sorry

end random_variable_point_of_increase_l310_310512


namespace cost_of_game_l310_310762

theorem cost_of_game
  (number_of_ice_creams : ℕ) 
  (price_per_ice_cream : ℕ)
  (total_sold : number_of_ice_creams = 24)
  (price : price_per_ice_cream = 5) :
  (number_of_ice_creams * price_per_ice_cream) / 2 = 60 :=
by
  sorry

end cost_of_game_l310_310762


namespace fraction_of_difference_l310_310037

theorem fraction_of_difference (A_s A_l : ℝ) (h_total : A_s + A_l = 500) (h_smaller : A_s = 225) :
  (A_l - A_s) / ((A_s + A_l) / 2) = 1 / 5 :=
by
  -- Proof goes here
  sorry

end fraction_of_difference_l310_310037


namespace calculate_expression_l310_310521

theorem calculate_expression :
  ((7 / 9) - (5 / 6) + (5 / 18)) * 18 = 4 :=
by
  -- proof to be filled in later.
  sorry

end calculate_expression_l310_310521


namespace minimum_number_of_troublemakers_l310_310786

theorem minimum_number_of_troublemakers (n m : ℕ) (students : Fin n → Prop) 
  (truthtellers : Fin n → Prop) (liars : Fin n → Prop) (round_table : List (Fin n))
  (h_all_students : n = 29)
  (h_truth_or_liar : ∀ s, students s → (truthtellers s ∨ liars s))
  (h_truth_tellers_conditions : ∀ s, truthtellers s → 
    (count_lies_next_to s = 1 ∨ count_lies_next_to s = 2))
  (h_liars_conditions : ∀ s, liars s → 
    (count_lies_next_to s ≠ 1 ∧ count_lies_next_to s ≠ 2))
  (h_round_table_conditions : round_table.length = n 
    ∧ (∀ i, i < n → round_table.nth i ≠ none)):
  m = 10 :=
by
  sorry

end minimum_number_of_troublemakers_l310_310786


namespace find_A_in_triangle_l310_310273

theorem find_A_in_triangle
  (a b : ℝ) (B A : ℝ)
  (h₀ : a = Real.sqrt 3)
  (h₁ : b = Real.sqrt 2)
  (h₂ : B = Real.pi / 4)
  (h₃ : a / Real.sin A = b / Real.sin B) :
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end find_A_in_triangle_l310_310273


namespace tom_age_ratio_l310_310167

-- Define the variables and conditions
variables (T N : ℕ)

-- Condition 1: Tom's current age is twice the sum of his children's ages
def children_sum_current : ℤ := T / 2

-- Condition 2: Tom's age N years ago was three times the sum of their ages then
def children_sum_past : ℤ := (T / 2) - 2 * N

-- Main theorem statement proving the ratio T/N = 10 assuming given conditions
theorem tom_age_ratio (h1 : T = 2 * (T / 2)) 
                      (h2 : T - N = 3 * ((T / 2) - 2 * N)) : 
                      T / N = 10 :=
sorry

end tom_age_ratio_l310_310167


namespace fact_division_example_l310_310859

theorem fact_division_example : (50! / 48!) = 2450 := 
by sorry

end fact_division_example_l310_310859


namespace total_cost_shorts_tshirt_boots_shinguards_l310_310309

variable (x : ℝ)

-- Definitions provided in the problem statement.
def cost_shorts : ℝ := x
def cost_shorts_and_tshirt : ℝ := 2 * x
def cost_shorts_and_boots : ℝ := 5 * x
def cost_shorts_and_shinguards : ℝ := 3 * x

-- The proof goal to verify:
theorem total_cost_shorts_tshirt_boots_shinguards : 
  (cost_shorts x + (cost_shorts_and_tshirt x - cost_shorts x) + 
   (cost_shorts_and_boots x - cost_shorts x) + 
   (cost_shorts_and_shinguards x - cost_shorts x)) = 8 * x := by 
  sorry

end total_cost_shorts_tshirt_boots_shinguards_l310_310309


namespace clothing_store_profit_l310_310996

theorem clothing_store_profit 
  (cost_price selling_price : ℕ)
  (initial_items_per_day items_increment items_reduction : ℕ)
  (initial_profit_per_day : ℕ) :
  -- Conditions
  cost_price = 50 ∧
  selling_price = 90 ∧
  initial_items_per_day = 20 ∧
  items_increment = 2 ∧
  items_reduction = 1 ∧
  initial_profit_per_day = 1200 →
  -- Question
  exists x, 
  (selling_price - x - cost_price) * (initial_items_per_day + items_increment * x) = initial_profit_per_day ∧
  x = 20 := 
sorry

end clothing_store_profit_l310_310996


namespace arithmetic_expression_evaluation_l310_310389

theorem arithmetic_expression_evaluation :
  (3 + 9) ^ 2 + (3 ^ 2) * (9 ^ 2) = 873 :=
by
  -- Proof is skipped, using sorry for now.
  sorry

end arithmetic_expression_evaluation_l310_310389


namespace value_of_expression_l310_310572

theorem value_of_expression (x : ℤ) (h : x^2 = 1369) : (x + 1) * (x - 1) = 1368 := 
by 
  sorry

end value_of_expression_l310_310572


namespace min_troublemakers_l310_310794

theorem min_troublemakers (n : ℕ) (truth_teller liar : ℕ → Prop) 
  (total_students : n = 29)
  (exists_one_liar : ∀ i : ℕ, liar i = true → ((liar (i - 1) = true ∨ liar (i + 1) = true))) -- assume cyclic seating
  (exists_two_liar : ∀ i : ℕ, truth_teller i = true → (liar (i - 1) = true ∧ liar (i + 1) = true)) :
  (∃ k : ℕ, k = 10 ∧ (∀ i : ℕ, i < 29 -> liar i = true -> k ≥ i)) :=
sorry

end min_troublemakers_l310_310794


namespace actual_distance_is_correct_l310_310141

def scale := 6000000
def map_distance := 5 -- in cm

def actual_distance := map_distance * scale / 100000 -- conversion factor from cm to km

theorem actual_distance_is_correct :
  actual_distance = 300 :=
by
  simp [actual_distance, map_distance, scale]
  exact sorry

end actual_distance_is_correct_l310_310141


namespace neg_exists_lt_1000_l310_310095

open Nat

theorem neg_exists_lt_1000 : (¬ ∃ n : ℕ, 2^n < 1000) = ∀ n : ℕ, 2^n ≥ 1000 := by
  sorry

end neg_exists_lt_1000_l310_310095


namespace half_angle_in_second_quadrant_l310_310562

theorem half_angle_in_second_quadrant (α : ℝ) (h : 180 < α ∧ α < 270) : 90 < α / 2 ∧ α / 2 < 135 := 
by
  sorry

end half_angle_in_second_quadrant_l310_310562


namespace fractional_identity_l310_310404

theorem fractional_identity (m n r t : ℚ) 
  (h₁ : m / n = 5 / 2) 
  (h₂ : r / t = 8 / 5) : 
  (2 * m * r - 3 * n * t) / (5 * n * t - 4 * m * r) = -5 / 11 :=
by 
  sorry

end fractional_identity_l310_310404


namespace min_troublemakers_l310_310793

theorem min_troublemakers (n : ℕ) (students : ℕ → Prop) 
  (h : n = 29)
  (condition1 : ∀ i, students i → (students ((i - 1) % n) ↔ ¬ students ((i + 1) % n)))
  (condition2 : ∀ i, ¬ students i → (students ((i - 1) % n) ∧ students ((i + 1) % n)))
  : ∃ L : ℕ, (L ≤ 29 ∧ L ≥ 10) :=
by sorry

end min_troublemakers_l310_310793


namespace sector_central_angle_l310_310408

noncomputable def sector_radius (r l : ℝ) : Prop :=
2 * r + l = 10

noncomputable def sector_area (r l : ℝ) : Prop :=
(1 / 2) * l * r = 4

noncomputable def central_angle (α r l : ℝ) : Prop :=
α = l / r

theorem sector_central_angle (r l α : ℝ) 
  (h1 : sector_radius r l) 
  (h2 : sector_area r l) 
  (h3 : central_angle α r l) : 
  α = 1 / 2 := 
by
  sorry

end sector_central_angle_l310_310408


namespace product_modulo_seven_l310_310214

/-- 2021 is congruent to 6 modulo 7 -/
def h1 : 2021 % 7 = 6 := rfl

/-- 2022 is congruent to 0 modulo 7 -/
def h2 : 2022 % 7 = 0 := rfl

/-- 2023 is congruent to 1 modulo 7 -/
def h3 : 2023 % 7 = 1 := rfl

/-- 2024 is congruent to 2 modulo 7 -/
def h4 : 2024 % 7 = 2 := rfl

/-- The product 2021 * 2022 * 2023 * 2024 is congruent to 0 modulo 7 -/
theorem product_modulo_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
  by sorry

end product_modulo_seven_l310_310214


namespace minimize_maximum_absolute_value_expression_l310_310069

theorem minimize_maximum_absolute_value_expression : 
  (∀ y : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2) →
  ∃ y : ℝ, (y = 2) ∧ (min_value = 0) :=
sorry -- Proof goes here

end minimize_maximum_absolute_value_expression_l310_310069


namespace sum_evaluation_l310_310203

theorem sum_evaluation : 5 * 399 + 4 * 399 + 3 * 399 + 398 = 5186 :=
by
  sorry

end sum_evaluation_l310_310203


namespace ratio_cost_to_marked_price_l310_310515

variables (x : ℝ) (marked_price : ℝ) (selling_price : ℝ) (cost_price : ℝ)

theorem ratio_cost_to_marked_price :
  (selling_price = marked_price - 1/4 * marked_price) →
  (cost_price = 2/3 * selling_price) →
  (cost_price / marked_price = 1/2) :=
by
  sorry

end ratio_cost_to_marked_price_l310_310515


namespace part1_part2_l310_310701

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |5 - x|

theorem part1 : ∃ m, m = 9 / 2 ∧ ∀ x, f x ≥ m :=
sorry

theorem part2 (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 3) : 
  (1 / (a + 1) + 1 / (b + 2)) ≥ 2 / 3 :=
sorry

end part1_part2_l310_310701


namespace find_range_l310_310981

open Real

def f (x : ℝ) : ℝ := exp x + x^2 - x

theorem find_range : 
  ∃ (a b : ℝ), (a ≤ b) ∧ 
    (∀ x ∈ Icc (-1 : ℝ) 1, a ≤ f x ∧ f x ≤ b) ∧ 
    a = 1 ∧ b = exp 1 :=
by
  sorry

end find_range_l310_310981


namespace maximum_combined_power_l310_310005

theorem maximum_combined_power (x1 x2 x3 : ℝ) (hx : x1 < 1 ∧ x2 < 1 ∧ x3 < 1) 
    (hcond : 2 * (x1 + x2 + x3) + 4 * (x1 * x2 * x3) = 3 * (x1 * x2 + x1 * x3 + x2 * x3) + 1) : 
    x1 + x2 + x3 ≤ 3 / 4 := 
sorry

end maximum_combined_power_l310_310005


namespace minimum_number_of_troublemakers_l310_310788

theorem minimum_number_of_troublemakers (n m : ℕ) (students : Fin n → Prop) 
  (truthtellers : Fin n → Prop) (liars : Fin n → Prop) (round_table : List (Fin n))
  (h_all_students : n = 29)
  (h_truth_or_liar : ∀ s, students s → (truthtellers s ∨ liars s))
  (h_truth_tellers_conditions : ∀ s, truthtellers s → 
    (count_lies_next_to s = 1 ∨ count_lies_next_to s = 2))
  (h_liars_conditions : ∀ s, liars s → 
    (count_lies_next_to s ≠ 1 ∧ count_lies_next_to s ≠ 2))
  (h_round_table_conditions : round_table.length = n 
    ∧ (∀ i, i < n → round_table.nth i ≠ none)):
  m = 10 :=
by
  sorry

end minimum_number_of_troublemakers_l310_310788


namespace arithmetic_sequence_properties_l310_310958

theorem arithmetic_sequence_properties
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (d : ℤ)
  (h_arith: ∀ n, a (n + 1) = a n + d)
  (hS: ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2)
  (h_a3_eq_S5: a 3 = S 5)
  (h_a2a4_eq_S4: a 2 * a 4 = S 4) :
  (∀ n, a n = 2 * n - 6) ∧ (∃ n, S n > a n ∧ ∀ m < n, ¬(S m > a m)) :=
begin
  sorry
end

end arithmetic_sequence_properties_l310_310958


namespace pow_evaluation_l310_310675

theorem pow_evaluation (h : 81 = 3^4) : 81^(5/4) = 243 := by
  sorry

end pow_evaluation_l310_310675


namespace sqrt_floor_square_l310_310889

theorem sqrt_floor_square (h1 : 7 < Real.sqrt 50) (h2 : Real.sqrt 50 < 8) :
  Int.floor (Real.sqrt 50) ^ 2 = 49 := by
  sorry

end sqrt_floor_square_l310_310889


namespace Ludwig_daily_salary_l310_310964

theorem Ludwig_daily_salary 
(D : ℝ)
(h_weekly_earnings : 4 * D + (3 / 2) * D = 55) :
D = 10 := 
by
  sorry

end Ludwig_daily_salary_l310_310964


namespace solution_set_of_inequality_l310_310926

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem solution_set_of_inequality :
  { x : ℝ | f (x - 2) + f (x^2 - 4) < 0 } = Set.Ioo (-3 : ℝ) 2 :=
by
  sorry

end solution_set_of_inequality_l310_310926


namespace eval_x_squared_minus_y_squared_l310_310083

theorem eval_x_squared_minus_y_squared (x y : ℝ) (h1 : 3 * x + 2 * y = 30) (h2 : 4 * x + 2 * y = 34) : x^2 - y^2 = -65 :=
by
  sorry

end eval_x_squared_minus_y_squared_l310_310083


namespace trail_length_proof_l310_310756

theorem trail_length_proof (x1 x2 x3 x4 x5 : ℝ)
  (h1 : x1 + x2 = 28)
  (h2 : x2 + x3 = 30)
  (h3 : x3 + x4 + x5 = 42)
  (h4 : x1 + x4 = 30) :
  x1 + x2 + x3 + x4 + x5 = 70 := by
  sorry

end trail_length_proof_l310_310756


namespace smallest_perfect_square_divisible_by_4_and_5_l310_310618

theorem smallest_perfect_square_divisible_by_4_and_5 : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (m : ℕ), n = m * m) ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ (n = 400) := 
by
  sorry

end smallest_perfect_square_divisible_by_4_and_5_l310_310618


namespace age_of_son_l310_310180

theorem age_of_son (S F : ℕ) (h1 : F = S + 28) (h2 : F + 2 = 2 * (S + 2)) : S = 26 := 
by
  -- skip the proof
  sorry

end age_of_son_l310_310180


namespace longest_side_AB_l310_310430

-- Definitions of angles in the quadrilateral
def angle_ABC := 65
def angle_BCD := 70
def angle_CDA := 60

/-- In a quadrilateral ABCD with angles as specified, prove that AB is the longest side. -/
theorem longest_side_AB (AB BC CD DA : ℝ) : 
  (angle_ABC = 65 ∧ angle_BCD = 70 ∧ angle_CDA = 60) → 
  AB > DA ∧ AB > BC ∧ AB > CD :=
by
  intros h
  sorry

end longest_side_AB_l310_310430


namespace intersection_of_M_and_N_l310_310422

def M := {x : ℝ | abs x ≤ 2}
def N := {x : ℝ | x^2 - 3 * x = 0}

theorem intersection_of_M_and_N : M ∩ N = {0} :=
by
  sorry

end intersection_of_M_and_N_l310_310422


namespace inheritance_amount_l310_310279

def federalTax (x : ℝ) : ℝ := 0.25 * x
def remainingAfterFederalTax (x : ℝ) : ℝ := x - federalTax x
def stateTax (x : ℝ) : ℝ := 0.15 * remainingAfterFederalTax x
def totalTaxes (x : ℝ) : ℝ := federalTax x + stateTax x

theorem inheritance_amount (x : ℝ) (h : totalTaxes x = 15000) : x = 41379 :=
by
  sorry

end inheritance_amount_l310_310279


namespace line_through_two_points_l310_310556

theorem line_through_two_points :
  ∀ (A_1 B_1 A_2 B_2 : ℝ),
    (2 * A_1 + 3 * B_1 = 1) →
    (2 * A_2 + 3 * B_2 = 1) →
    (∀ (x y : ℝ), (2 * x + 3 * y = 1) → (x * (B_2 - B_1) + y * (A_1 - A_2) = A_1 * B_2 - A_2 * B_1)) :=
by 
  intros A_1 B_1 A_2 B_2 h1 h2 x y hxy
  sorry

end line_through_two_points_l310_310556


namespace cheetahs_pandas_ratio_l310_310266

-- Let C denote the number of cheetahs 5 years ago.
-- Let P denote the number of pandas 5 years ago.
-- The conditions given are:
-- 1. The ratio of cheetahs to pandas 5 years ago was the same as it is now.
-- 2. The number of cheetahs has increased by 2.
-- 3. The number of pandas has increased by 6.
-- We need to prove that the current ratio of cheetahs to pandas is C / P.

theorem cheetahs_pandas_ratio
  (C P : ℕ)
  (h1 : C / P = (C + 2) / (P + 6)) :
  (C + 2) / (P + 6) = C / P :=
by sorry

end cheetahs_pandas_ratio_l310_310266


namespace tank_filling_time_l310_310169

noncomputable def fill_time (R1 R2 R3 : ℚ) : ℚ :=
  1 / (R1 + R2 + R3)

theorem tank_filling_time :
  let R1 := 1 / 18
  let R2 := 1 / 30
  let R3 := -1 / 45
  fill_time R1 R2 R3 = 15 :=
by
  intros
  unfold fill_time
  sorry

end tank_filling_time_l310_310169


namespace abs_sub_sqrt5_l310_310153

theorem abs_sub_sqrt5 :
  |2 - real.sqrt 5| = real.sqrt 5 - 2 :=
by sorry

end abs_sub_sqrt5_l310_310153


namespace problem1_problem2_l310_310631

-- Problem 1
theorem problem1 : (-1 : ℤ) ^ 2024 + (1 / 3 : ℝ) ^ (-2 : ℤ) - (3.14 - Real.pi) ^ 0 = 9 := 
sorry

-- Problem 2
theorem problem2 (x : ℤ) (y : ℤ) (hx : x = 2) (hy : y = 3) : 
  x * (x + 2 * y) - (x + 1) ^ 2 + 2 * x = 11 :=
sorry

end problem1_problem2_l310_310631


namespace range_of_a_l310_310409

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.exp x - 2 * a * x - a ^ 2 + 3

theorem range_of_a (h : ∀ x, x ≥ 0 → f x a - x ^ 2 ≥ 0) :
  -Real.sqrt 5 ≤ a ∧ a ≤ 3 - Real.log 3 := sorry

end range_of_a_l310_310409


namespace minimum_number_of_troublemakers_l310_310789

theorem minimum_number_of_troublemakers (n m : ℕ) (students : Fin n → Prop) 
  (truthtellers : Fin n → Prop) (liars : Fin n → Prop) (round_table : List (Fin n))
  (h_all_students : n = 29)
  (h_truth_or_liar : ∀ s, students s → (truthtellers s ∨ liars s))
  (h_truth_tellers_conditions : ∀ s, truthtellers s → 
    (count_lies_next_to s = 1 ∨ count_lies_next_to s = 2))
  (h_liars_conditions : ∀ s, liars s → 
    (count_lies_next_to s ≠ 1 ∧ count_lies_next_to s ≠ 2))
  (h_round_table_conditions : round_table.length = n 
    ∧ (∀ i, i < n → round_table.nth i ≠ none)):
  m = 10 :=
by
  sorry

end minimum_number_of_troublemakers_l310_310789


namespace quadratic_real_roots_implies_k_range_l310_310724

theorem quadratic_real_roots_implies_k_range (k : ℝ) 
  (h : ∃ x : ℝ, k * x^2 + 2 * x - 1 = 0)
  (hk : k ≠ 0) : k ≥ -1 ∧ k ≠ 0 :=
sorry

end quadratic_real_roots_implies_k_range_l310_310724


namespace trig_identity_cos_add_l310_310236

open Real

theorem trig_identity_cos_add (x : ℝ) (h1 : sin (π / 3 - x) = 3 / 5) (h2 : π / 2 < x ∧ x < π) :
  cos (x + π / 6) = 3 / 5 :=
by
  sorry

end trig_identity_cos_add_l310_310236


namespace total_leaves_on_farm_l310_310379

noncomputable def number_of_branches : ℕ := 10
noncomputable def sub_branches_per_branch : ℕ := 40
noncomputable def leaves_per_sub_branch : ℕ := 60
noncomputable def number_of_trees : ℕ := 4

theorem total_leaves_on_farm :
  number_of_branches * sub_branches_per_branch * leaves_per_sub_branch * number_of_trees = 96000 :=
by
  sorry

end total_leaves_on_farm_l310_310379


namespace extremum_at_x1_l310_310700

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_at_x1 (a b : ℝ) (h1 : (3*1^2 + 2*a*1 + b) = 0) (h2 : 1^3 + a*1^2 + b*1 + a^2 = 10) :
  a = 4 :=
by
  sorry

end extremum_at_x1_l310_310700


namespace calculate_expression_l310_310205

def seq (k : Nat) : Nat := 2^k + 3^k

def product_seq : Nat :=
  (2 + 3) * (2^3 + 3^3) * (2^6 + 3^6) * (2^12 + 3^12) * (2^24 + 3^24)

theorem calculate_expression :
  product_seq = (3^47 - 2^47) :=
sorry

end calculate_expression_l310_310205


namespace initial_card_count_l310_310520

theorem initial_card_count (x : ℕ) (h1 : (3 * (1/2) * ((x / 3) + (4 / 3))) = 34) : x = 64 :=
  sorry

end initial_card_count_l310_310520


namespace inequality_proof_l310_310451

theorem inequality_proof (a b : ℝ) (c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : c < 0) :
  (c / a > c / b) ∧ (a^c < b^c) ∧ (Real.log (a - c) / Real.log b > Real.log (b - c) / Real.log a) := 
sorry

end inequality_proof_l310_310451


namespace measured_diagonal_in_quadrilateral_l310_310851

-- Defining the conditions (side lengths and diagonals)
def valid_diagonal (side1 side2 side3 side4 diagonal : ℝ) : Prop :=
  side1 + side2 > diagonal ∧ side1 + side3 > diagonal ∧ side1 + side4 > diagonal ∧ 
  side2 + side3 > diagonal ∧ side2 + side4 > diagonal ∧ side3 + side4 > diagonal

theorem measured_diagonal_in_quadrilateral :
  let sides := [1, 2, 2.8, 5]
  let diagonal1 := 7.5
  let diagonal2 := 2.8
  (valid_diagonal 1 2 2.8 5 diagonal2) :=
sorry

end measured_diagonal_in_quadrilateral_l310_310851


namespace roman_numeral_sketching_l310_310199

/-- Roman numeral sketching problem. -/
theorem roman_numeral_sketching (n : ℕ) (k : ℕ) (students : ℕ) 
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ i / 1 = i) 
  (h2 : ∀ i : ℕ, i > n → i = n - (i - n)) 
  (h3 : k = 7) 
  (h4 : ∀ r : ℕ, r = (k * n)) : students = 350 :=
by
  sorry

end roman_numeral_sketching_l310_310199


namespace largest_n_satisfying_conditions_l310_310074

theorem largest_n_satisfying_conditions :
  ∃ n : ℤ, n = 181 ∧
    (∃ m : ℤ, n^2 = (m + 1)^3 - m^3) ∧
    ∃ k : ℤ, 2 * n + 79 = k^2 :=
by
  sorry

end largest_n_satisfying_conditions_l310_310074


namespace effective_discount_l310_310196

theorem effective_discount (original_price sale_price price_after_coupon : ℝ) :
  sale_price = 0.4 * original_price →
  price_after_coupon = 0.7 * sale_price →
  (original_price - price_after_coupon) / original_price * 100 = 72 :=
by
  intros h1 h2
  sorry

end effective_discount_l310_310196


namespace total_cost_is_eight_times_l310_310322

theorem total_cost_is_eight_times (x : ℝ) 
  (h1 : ∀ t, x + t = 2 * x)
  (h2 : ∀ b, x + b = 5 * x)
  (h3 : ∀ s, x + s = 3 * x) :
  ∃ t b s, x + t + b + s = 8 * x :=
by
  sorry

end total_cost_is_eight_times_l310_310322


namespace average_speed_to_first_summit_l310_310591

theorem average_speed_to_first_summit 
  (time_first_summit : ℝ := 3)
  (time_descend_partially : ℝ := 1)
  (time_second_uphill : ℝ := 2)
  (time_descend_back : ℝ := 2)
  (avg_speed_whole_journey : ℝ := 3) :
  avg_speed_whole_journey = 3 →
  time_first_summit = 3 →
  avg_speed_whole_journey * (time_first_summit + time_descend_partially + time_second_uphill + time_descend_back) = 24 →
  avg_speed_whole_journey = 3 := 
by
  intros h_avg_speed h_time_first_summit h_total_distance
  sorry

end average_speed_to_first_summit_l310_310591


namespace ages_correct_l310_310277

-- Definitions of the given conditions
def john_age : ℕ := 42
def tim_age : ℕ := 79
def james_age : ℕ := 30
def lisa_age : ℚ := 54.5
def kate_age : ℕ := 34
def michael_age : ℚ := 61.5
def anna_age : ℚ := 54.5

-- Mathematically equivalent proof problem
theorem ages_correct :
  (james_age = 30) ∧
  (lisa_age = 54.5) ∧
  (kate_age = 34) ∧
  (michael_age = 61.5) ∧
  (anna_age = 54.5) :=
by {
  sorry  -- Proof to be filled in
}

end ages_correct_l310_310277


namespace complement_of_A_l310_310413

open Set

theorem complement_of_A (U : Set ℕ) (A : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {3, 4, 5}) :
  (U \ A) = {1, 2, 6} :=
by
  sorry

end complement_of_A_l310_310413


namespace case_m_eq_neg_1_case_m_gt_neg_1_case_m_lt_neg_1_l310_310684

noncomputable def solution_set (m x : ℝ) : Prop :=
  x^2 + (m-1) * x - m > 0

theorem case_m_eq_neg_1 (x : ℝ) :
  solution_set (-1) x ↔ x ≠ 1 :=
sorry

theorem case_m_gt_neg_1 (m x : ℝ) (hm : m > -1) :
  solution_set m x ↔ (x < -m ∨ x > 1) :=
sorry

theorem case_m_lt_neg_1 (m x : ℝ) (hm : m < -1) :
  solution_set m x ↔ (x < 1 ∨ x > -m) :=
sorry

end case_m_eq_neg_1_case_m_gt_neg_1_case_m_lt_neg_1_l310_310684


namespace evaluate_expression_l310_310062

theorem evaluate_expression : 3 ^ 123 + 9 ^ 5 / 9 ^ 3 = 3 ^ 123 + 81 :=
by
  -- we add sorry as the proof is not required
  sorry

end evaluate_expression_l310_310062


namespace find_multiple_of_y_l310_310727

noncomputable def multiple_of_y (q m : ℝ) : Prop :=
  ∀ x y : ℝ, (x = 5 - q) → (y = m * q - 1) → (q = 1) → (x = 3 * y) → (m = 7 / 3)

theorem find_multiple_of_y :
  multiple_of_y 1 (7 / 3) :=
by
  sorry

end find_multiple_of_y_l310_310727


namespace hyperbola_asymptote_l310_310094

theorem hyperbola_asymptote (a : ℝ) (h : a > 0)
  (has_asymptote : ∀ x : ℝ, abs (9 / a * x) = abs (3 * x))
  : a = 3 :=
sorry

end hyperbola_asymptote_l310_310094


namespace sqrt_floor_square_eq_49_l310_310876

theorem sqrt_floor_square_eq_49 : (⌊Real.sqrt 50⌋)^2 = 49 :=
by
  have h1 : 7 < Real.sqrt 50, from (by norm_num : 7 < Real.sqrt 50),
  have h2 : Real.sqrt 50 < 8, from (by norm_num : Real.sqrt 50 < 8),
  have floor_sqrt_50_eq_7 : ⌊Real.sqrt 50⌋ = 7, from Int.floor_eq_iff.mpr ⟨h1, h2⟩,
  calc
    (⌊Real.sqrt 50⌋)^2 = (7)^2 : by rw [floor_sqrt_50_eq_7]
                  ... = 49 : by norm_num,
  sorry -- omit the actual proof

end sqrt_floor_square_eq_49_l310_310876


namespace evaluate_functions_l310_310716

def f (x : ℝ) := x + 2
def g (x : ℝ) := 2 * x^2 - 4
def h (x : ℝ) := x + 1

theorem evaluate_functions : f (g (h 3)) = 30 := by
  sorry

end evaluate_functions_l310_310716


namespace sequence_general_term_l310_310436

def recurrence_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n / (1 + a n)

theorem sequence_general_term :
  ∀ a : ℕ → ℚ, recurrence_sequence a → ∀ n : ℕ, n ≥ 1 → a n = 2 / (2 * n - 1) :=
by
  intro a h n hn
  sorry

end sequence_general_term_l310_310436


namespace line_through_circle_center_slope_one_eq_l310_310607

theorem line_through_circle_center_slope_one_eq (x y : ℝ) :
  (∃ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 4 ∧ y = 2) →
  (∃ m : ℝ, m = 1 ∧ (x + 1) = m * (y - 2)) →
  (x - y + 3 = 0) :=
sorry

end line_through_circle_center_slope_one_eq_l310_310607


namespace cost_of_plastering_is_334_point_8_l310_310181

def tank_length : ℝ := 25
def tank_width : ℝ := 12
def tank_depth : ℝ := 6
def cost_per_sq_meter : ℝ := 0.45

def bottom_area : ℝ := tank_length * tank_width
def long_wall_area : ℝ := 2 * (tank_length * tank_depth)
def short_wall_area : ℝ := 2 * (tank_width * tank_depth)
def total_surface_area : ℝ := bottom_area + long_wall_area + short_wall_area
def total_cost : ℝ := total_surface_area * cost_per_sq_meter

theorem cost_of_plastering_is_334_point_8 :
  total_cost = 334.8 :=
by
  sorry

end cost_of_plastering_is_334_point_8_l310_310181


namespace abs_two_minus_sqrt_five_l310_310154

noncomputable def sqrt_5 : ℝ := Real.sqrt 5

theorem abs_two_minus_sqrt_five : |2 - sqrt_5| = sqrt_5 - 2 := by
  sorry

end abs_two_minus_sqrt_five_l310_310154


namespace no_distinct_natural_numbers_eq_sum_and_cubes_eq_l310_310465

theorem no_distinct_natural_numbers_eq_sum_and_cubes_eq:
  ∀ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 
  → a^3 + b^3 = c^3 + d^3
  → a + b = c + d
  → false := 
by
  intros
  sorry

end no_distinct_natural_numbers_eq_sum_and_cubes_eq_l310_310465


namespace find_a_l310_310570

theorem find_a (a : ℝ) :
  (∀ x, x < 2 → 0 < a - 3 * x) ↔ (a = 6) :=
by
  sorry

end find_a_l310_310570


namespace count_squares_3x3_grid_count_squares_5x5_grid_l310_310629

/-- Define a mathematical problem: 
  Prove that the number of squares with all four vertices on the dots in a 3x3 grid is 4.
  Prove that the number of squares with all four vertices on the dots in a 5x5 grid is 50.
-/

def num_squares_3x3 : Nat := 4
def num_squares_5x5 : Nat := 50

theorem count_squares_3x3_grid : 
  ∀ (grid_size : Nat), grid_size = 3 → (∃ (dots_on_square : Bool), ∀ (distance_between_dots : Real), (dots_on_square = true → num_squares_3x3 = 4)) := 
by 
  intros grid_size h1
  exists true
  intros distance_between_dots 
  sorry

theorem count_squares_5x5_grid : 
  ∀ (grid_size : Nat), grid_size = 5 → (∃ (dots_on_square : Bool), ∀ (distance_between_dots : Real), (dots_on_square = true → num_squares_5x5 = 50)) :=
by 
  intros grid_size h1
  exists true
  intros distance_between_dots 
  sorry

end count_squares_3x3_grid_count_squares_5x5_grid_l310_310629


namespace real_roots_exist_l310_310461

noncomputable def cubic_equation (x : ℝ) := x^3 - x^2 - 2*x + 1

theorem real_roots_exist : ∃ (a b : ℝ), 
  cubic_equation a = 0 ∧ cubic_equation b = 0 ∧ a - a * b = 1 := 
by
  sorry

end real_roots_exist_l310_310461


namespace silverware_probability_l310_310416

-- Definitions based on the problem conditions
def total_silverware : ℕ := 8 + 10 + 7
def total_combinations : ℕ := Nat.choose total_silverware 4

def fork_combinations : ℕ := Nat.choose 8 2
def spoon_combinations : ℕ := Nat.choose 10 1
def knife_combinations : ℕ := Nat.choose 7 1

def favorable_combinations : ℕ := fork_combinations * spoon_combinations * knife_combinations
def specific_combination_probability : ℚ := favorable_combinations / total_combinations

-- The statement to prove the given probability
theorem silverware_probability :
  specific_combination_probability = 392 / 2530 :=
by
  sorry

end silverware_probability_l310_310416


namespace min_troublemakers_in_class_l310_310813

noncomputable def min_troublemakers : ℕ :=
  10

theorem min_troublemakers_in_class :
  (∃ t l : ℕ, t + l = 29 ∧ t + l - 1 = 29 ∧
   (∀ i : ℕ, i < 29 → (i % 3 = 0 → ∃ t : ℕ, t = 1) ∧ 
   (i % 3 ≠ 0 → ∃ t : ℕ, t = 2))) →
   min_troublemakers = 10 :=
by
  sorry

end min_troublemakers_in_class_l310_310813


namespace product_mod_7_l310_310208

theorem product_mod_7 : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
by
  have h1 : 2021 % 7 = 6 := by sorry
  have h2 : 2022 % 7 = 0 := by sorry
  have h3 : 2023 % 7 = 1 := by sorry
  have h4 : 2024 % 7 = 2 := by sorry
  sorry

end product_mod_7_l310_310208


namespace cost_of_purchase_l310_310317

theorem cost_of_purchase (x : ℝ) (T_shirt boots shin_guards : ℝ) 
  (h1 : x + T_shirt = 2 * x)
  (h2 : x + boots = 5 * x)
  (h3 : x + shin_guards = 3 * x) :
  x + T_shirt + boots + shin_guards = 8 * x :=
begin
  sorry
end

end cost_of_purchase_l310_310317


namespace smallest_perfect_square_div_by_4_and_5_l310_310619

theorem smallest_perfect_square_div_by_4_and_5 : 
  ∃ n : ℕ, (∃ m : ℕ, n = m^2) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (∀ k : ℕ, (∃ l : ℕ, k = l^2) ∧ (4 ∣ k) ∧ (5 ∣ k) → n ≤ k) :=
begin
  let n := 400,
  use n,
  split,
  { use 20, -- 400 is 20^2
    refl },
  split,
  { exact dvd.intro 100 rfl }, -- 400 = 4 * 100
  split,
  { exact dvd.intro 80 rfl }, -- 400 = 5 * 80
  { 
    intros k hk,
    obtain ⟨l, hl⟩ := hk.left,
    obtain ⟨_h4⟩ := hk.right.left  -- k divisible by 4
    obtain ⟨_h5⟩ := hk.right.right -- k divisible by 5
    rw hl,
    sorry  -- This is where the rest of the proof would go.
  }
end

end smallest_perfect_square_div_by_4_and_5_l310_310619


namespace count_CONES_paths_l310_310053

def diagram : List (List Char) :=
  [[' ', ' ', 'C', ' ', ' ', ' '],
   [' ', 'C', 'O', 'C', ' ', ' '],
   ['C', 'O', 'N', 'O', 'C', ' '],
   [' ', 'N', 'E', 'N', ' ', ' '],
   [' ', ' ', 'S', ' ', ' ', ' ']]

def is_adjacent (pos1 pos2 : (Nat × Nat)) : Bool :=
  (pos1.1 = pos2.1 ∨ pos1.1 + 1 = pos2.1 ∨ pos1.1 = pos2.1 + 1) ∧
  (pos1.2 = pos2.2 ∨ pos1.2 + 1 = pos2.2 ∨ pos1.2 = pos2.2 + 1)

def valid_paths (diagram : List (List Char)) : Nat :=
  -- Implementation of counting paths that spell "CONES" skipped
  sorry

theorem count_CONES_paths (d : List (List Char)) 
  (h : d = [[' ', ' ', 'C', ' ', ' ', ' '],
            [' ', 'C', 'O', 'C', ' ', ' '],
            ['C', 'O', 'N', 'O', 'C', ' '],
            [' ', 'N', 'E', 'N', ' ', ' '],
            [' ', ' ', 'S', ' ', ' ', ' ']]): valid_paths d = 6 := 
by
  sorry

end count_CONES_paths_l310_310053


namespace marly_100_bills_l310_310590

-- Define the number of each type of bill Marly has
def num_20_bills := 10
def num_10_bills := 8
def num_5_bills := 4

-- Define the values of the bills
def value_20_bill := 20
def value_10_bill := 10
def value_5_bill := 5

-- Define the total amount of money Marly has
def total_amount := num_20_bills * value_20_bill + num_10_bills * value_10_bill + num_5_bills * value_5_bill

-- Define the value of a $100 bill
def value_100_bill := 100

-- Now state the main theorem
theorem marly_100_bills : total_amount / value_100_bill = 3 := by
  sorry

end marly_100_bills_l310_310590


namespace find_xyz_l310_310132

theorem find_xyz (x y z : ℝ)
  (h1 : x > 4)
  (h2 : y > 4)
  (h3 : z > 4)
  (h4 : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 42) :
  (x, y, z) = (11, 9, 7) :=
by {
  sorry
}

end find_xyz_l310_310132


namespace minimum_liars_l310_310783

noncomputable def minimum_troublemakers (n : ℕ) := 10

theorem minimum_liars (n : ℕ) (students : ℕ → bool) 
  (seated_at_round_table : ∀ i, students (i % 29) = tt ∨ students (i % 29) = ff)
  (one_liar_next_to : ∀ i, students i = tt → (students ((i + 1) % 29) = ff ∨ students ((i + 28) % 29) = ff))
  (two_liars_next_to : ∀ i, students i = ff → ((students ((i + 1) % 29) = ff) ∧ students ((i + 28) % 29) = ff)) :
  ∃ m, m = minimum_troublemakers 29 ∧ (m ≤ n) :=
by
  let liars_count := 10
  have : liars_count = 10 := rfl
  use liars_count
  exact ⟨rfl, sorry⟩

end minimum_liars_l310_310783


namespace problem_a_proof_l310_310362

variables {A B C D M K : Point}
variables {triangle_ABC : Triangle A B C}
variables {incircle : Circle} (ht : touches incircle AC D) 
variables (hdm : diameter incircle D M) 
variables (bm_line : Line B M) (intersect_bm_ac : intersects bm_line AC K)

theorem problem_a_proof : 
  AK = DC :=
sorry

end problem_a_proof_l310_310362


namespace tan_half_angle_of_sin_alpha_is_given_l310_310406

theorem tan_half_angle_of_sin_alpha_is_given 
  (α : Real) 
  (h1 : Real.sin α = 4 / 5)
  (h2 : π / 2 < α ∧ α < π) : Real.tan (α / 2) = 2 := 
by 
  sorry

end tan_half_angle_of_sin_alpha_is_given_l310_310406


namespace minimum_liars_l310_310785

noncomputable def minimum_troublemakers (n : ℕ) := 10

theorem minimum_liars (n : ℕ) (students : ℕ → bool) 
  (seated_at_round_table : ∀ i, students (i % 29) = tt ∨ students (i % 29) = ff)
  (one_liar_next_to : ∀ i, students i = tt → (students ((i + 1) % 29) = ff ∨ students ((i + 28) % 29) = ff))
  (two_liars_next_to : ∀ i, students i = ff → ((students ((i + 1) % 29) = ff) ∧ students ((i + 28) % 29) = ff)) :
  ∃ m, m = minimum_troublemakers 29 ∧ (m ≤ n) :=
by
  let liars_count := 10
  have : liars_count = 10 := rfl
  use liars_count
  exact ⟨rfl, sorry⟩

end minimum_liars_l310_310785


namespace ratio_of_kids_l310_310728

theorem ratio_of_kids (k2004 k2005 k2006 : ℕ) 
  (h2004: k2004 = 60) 
  (h2005: k2005 = k2004 / 2)
  (h2006: k2006 = 20) :
  (k2006 : ℚ) / k2005 = 2 / 3 :=
by
  sorry

end ratio_of_kids_l310_310728


namespace necessary_but_not_sufficient_l310_310331

variable (p q : Prop)
-- Condition p: The base of a right prism is a rhombus.
def base_of_right_prism_is_rhombus := p
-- Condition q: A prism is a right rectangular prism.
def prism_is_right_rectangular := q

-- Proof: p is a necessary but not sufficient condition for q.
theorem necessary_but_not_sufficient (p q : Prop) 
  (h1 : base_of_right_prism_is_rhombus p)
  (h2 : prism_is_right_rectangular q) : 
  (q → p) ∧ ¬ (p → q) :=
sorry

end necessary_but_not_sufficient_l310_310331


namespace polynomial_remainder_division_l310_310896

theorem polynomial_remainder_division :
  ∀ (x : ℝ), (x^4 + 2 * x^2 - 3) % (x^2 + 3 * x + 2) = -21 * x - 21 := 
by
  sorry

end polynomial_remainder_division_l310_310896


namespace Vasya_has_larger_amount_l310_310357

-- Defining the conditions and given data
variables (V P : ℝ)

-- Vasya's profit calculation
def Vasya_profit (V : ℝ) : ℝ := 0.20 * V

-- Petya's profit calculation considering exchange rate increase
def Petya_profit (P : ℝ) : ℝ := 0.2045 * P

-- Proof statement
theorem Vasya_has_larger_amount (h : Vasya_profit V = Petya_profit P) : V > P :=
sorry

end Vasya_has_larger_amount_l310_310357


namespace a_and_c_can_complete_in_20_days_l310_310833

-- Define the work rates for the pairs given in the conditions.
variables {A B C : ℚ}

-- a and b together can complete the work in 12 days
axiom H1 : A + B = 1 / 12

-- b and c together can complete the work in 15 days
axiom H2 : B + C = 1 / 15

-- a, b, and c together can complete the work in 10 days
axiom H3 : A + B + C = 1 / 10

-- We aim to prove that a and c together can complete the work in 20 days,
-- hence their combined work rate should be 1 / 20.
theorem a_and_c_can_complete_in_20_days : A + C = 1 / 20 :=
by
  -- sorry will be used to skip the proof
  sorry

end a_and_c_can_complete_in_20_days_l310_310833


namespace die_top_face_odd_probability_l310_310594

-- Define the standard die faces and the number of dots on each face
def die_faces : Fin 6 → ℕ
| 0 => 1
| 1 => 2
| 2 => 3
| 3 => 4
| 4 => 5
| 5 => 6

-- Define the total number of dots on the die
def total_dots : ℕ := 21

-- Define the probability that the top face has an odd number of dots
def prob_top_odd_dots : ℝ := 11 / 21

theorem die_top_face_odd_probability :
  (1 / 6) * ( 
    (1 - 1 / 21) + 
    (2 / 21) + 
    (1 - 3 / 21) + 
    (4 / 21) + 
    (1 - 5 / 21) + 
    (6 / 21)
  ) = prob_top_odd_dots := by
  simp
  norm_num
  sorry

end die_top_face_odd_probability_l310_310594


namespace value_of_x2_plus_9y2_l310_310419

theorem value_of_x2_plus_9y2 (x y : ℝ) (h1 : x + 3 * y = 9) (h2 : x * y = -15) : x^2 + 9 * y^2 = 171 :=
sorry

end value_of_x2_plus_9y2_l310_310419


namespace polynomial_divisible_by_seven_l310_310898

-- Define the theorem
theorem polynomial_divisible_by_seven (n : ℤ) : 7 ∣ (n + 7)^2 - n^2 :=
by sorry

end polynomial_divisible_by_seven_l310_310898


namespace total_leaves_on_farm_l310_310377

theorem total_leaves_on_farm : 
  (branches_per_tree subbranches_per_branch leaves_per_subbranch trees_on_farm : ℕ)
  (h1 : branches_per_tree = 10)
  (h2 : subbranches_per_branch = 40)
  (h3 : leaves_per_subbranch = 60)
  (h4 : trees_on_farm = 4) :
  (trees_on_farm * branches_per_tree * subbranches_per_branch * leaves_per_subbranch = 96000) :=
by
  sorry

end total_leaves_on_farm_l310_310377


namespace perfect_squares_between_100_and_400_l310_310108

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def count_perfect_squares_between (a b : ℕ) : ℕ :=
  (finset.Ico a b).filter is_perfect_square .card

theorem perfect_squares_between_100_and_400 : count_perfect_squares_between 101 400 = 9 :=
by
  -- The space for the proof is intentionally left as a placeholder
  sorry

end perfect_squares_between_100_and_400_l310_310108


namespace composite_numbers_l310_310778

theorem composite_numbers (n : ℕ) (hn : n > 0) :
  (∃ p q, p > 1 ∧ q > 1 ∧ 2 * 2^(2^n) + 1 = p * q) ∧ 
  (∃ p q, p > 1 ∧ q > 1 ∧ 3 * 2^(2*n) + 1 = p * q) :=
sorry

end composite_numbers_l310_310778


namespace fibonacci_factorial_sum_l310_310825

def factorial_last_two_digits(n: ℕ) : ℕ :=
  if n > 10 then 0 else 
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | 4 => 24
  | 5 => 120 % 100
  | 6 => 720 % 100
  | 7 => 5040 % 100
  | 8 => 40320 % 100
  | 9 => 362880 % 100
  | 10 => 3628800 % 100
  | _ => 0

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 144]

noncomputable def sum_last_two_digits (l: List ℕ) : ℕ :=
  l.map factorial_last_two_digits |>.sum

theorem fibonacci_factorial_sum:
  sum_last_two_digits fibonacci_factorial_series = 50 := by
  sorry

end fibonacci_factorial_sum_l310_310825


namespace initial_students_count_l310_310604

theorem initial_students_count (n : ℕ) (T T' : ℚ)
    (h1 : T = n * 61.5)
    (h2 : T' = T - 24)
    (h3 : T' = (n - 1) * 64) :
  n = 16 :=
by
  sorry

end initial_students_count_l310_310604


namespace current_average_is_35_l310_310189

noncomputable def cricket_avg (A : ℝ) : Prop :=
  let innings := 10
  let next_runs := 79
  let increase := 4
  (innings * A + next_runs = (A + increase) * (innings + 1))

theorem current_average_is_35 : cricket_avg 35 :=
by
  unfold cricket_avg
  simp only
  sorry

end current_average_is_35_l310_310189


namespace valid_range_and_difference_l310_310579

/- Assume side lengths as given expressions -/
def BC (x : ℝ) : ℝ := x + 11
def AC (x : ℝ) : ℝ := x + 6
def AB (x : ℝ) : ℝ := 3 * x + 2

/- Define the inequalities representing the triangle inequalities and largest angle condition -/
def triangle_inequality1 (x : ℝ) : Prop := AB x + AC x > BC x
def triangle_inequality2 (x : ℝ) : Prop := AB x + BC x > AC x
def triangle_inequality3 (x : ℝ) : Prop := AC x + BC x > AB x
def largest_angle_condition (x : ℝ) : Prop := BC x > AB x

/- Define the combined condition for x, ensuring all relevant conditions are met -/
def valid_x_range (x : ℝ) : Prop :=
  1 < x ∧ x < 4.5 ∧ triangle_inequality1 x ∧ triangle_inequality2 x ∧ triangle_inequality3 x ∧ largest_angle_condition x

/- Compute n - m for the interval (m, n) where x lies -/
def n_minus_m : ℝ :=
  4.5 - 1

/- Main theorem stating the final result -/
theorem valid_range_and_difference :
  (∃ x : ℝ, valid_x_range x) ∧ (n_minus_m = 7 / 2) :=
by
  sorry

end valid_range_and_difference_l310_310579


namespace range_of_f_l310_310895

noncomputable def f (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_f :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → (f x ≥ (Real.pi / 2 - Real.arctan 2) ∧ f x ≤ (Real.pi / 2 + Real.arctan 2)) :=
by
  sorry

end range_of_f_l310_310895


namespace melindas_math_textbooks_probability_l310_310457

def total_ways_to_arrange_textbooks : ℕ :=
  (Nat.choose 15 4) * (Nat.choose 11 5) * (Nat.choose 6 6)

def favorable_ways (b : ℕ) : ℕ :=
  match b with
  | 4 => (Nat.choose 11 0) * (Nat.choose 11 5) * (Nat.choose 6 6)
  | 5 => (Nat.choose 11 1) * (Nat.choose 10 4) * (Nat.choose 6 6)
  | 6 => (Nat.choose 11 2) * (Nat.choose 9 4) * (Nat.choose 5 5)
  | _ => 0

def total_favorable_ways : ℕ :=
  favorable_ways 4 + favorable_ways 5 + favorable_ways 6

theorem melindas_math_textbooks_probability :
  let m := 1
  let n := 143
  Nat.Gcd m n = 1 ∧ total_ways_to_arrange_textbooks = 1387386 ∧ total_favorable_ways = 9702
  → m + n = 144 := by
sory

end melindas_math_textbooks_probability_l310_310457


namespace eval_floor_sqrt_50_square_l310_310879

theorem eval_floor_sqrt_50_square:
    (int.floor (real.sqrt 50))^2 = 49 :=
by
  have h1 : real.sqrt 49 < real.sqrt 50 := by norm_num [real.sqrt]
  have h2 : real.sqrt 50 < real.sqrt 64 := by norm_num [real.sqrt]
  have floor_sqrt_50 : int.floor (real.sqrt 50) = 7 :=
    by linarith [h1, h2]
  rw [floor_sqrt_50]
  norm_num

end eval_floor_sqrt_50_square_l310_310879


namespace cos_identity_l310_310250

theorem cos_identity (α : ℝ) (h : Real.cos (π / 3 - α) = 3 / 5) : 
  Real.cos (2 * π / 3 + α) = -3 / 5 :=
by
  sorry

end cos_identity_l310_310250


namespace min_troublemakers_l310_310796

theorem min_troublemakers (n : ℕ) (truth_teller liar : ℕ → Prop) 
  (total_students : n = 29)
  (exists_one_liar : ∀ i : ℕ, liar i = true → ((liar (i - 1) = true ∨ liar (i + 1) = true))) -- assume cyclic seating
  (exists_two_liar : ∀ i : ℕ, truth_teller i = true → (liar (i - 1) = true ∧ liar (i + 1) = true)) :
  (∃ k : ℕ, k = 10 ∧ (∀ i : ℕ, i < 29 -> liar i = true -> k ≥ i)) :=
sorry

end min_troublemakers_l310_310796


namespace minutes_spent_calling_clients_l310_310752

theorem minutes_spent_calling_clients
    (C : ℕ)
    (H1 : 7 * C + C = 560) :
    C = 70 :=
sorry

end minutes_spent_calling_clients_l310_310752


namespace max_profit_at_90_l310_310837

-- Definitions for conditions
def fixed_cost : ℝ := 5
def price_per_unit : ℝ := 100

noncomputable def variable_cost (x : ℕ) : ℝ :=
  if h : x < 80 then
    0.5 * x^2 + 40 * x
  else
    101 * x + 8100 / x - 2180

-- Definition of the profit function
noncomputable def profit (x : ℕ) : ℝ :=
  if h : x < 80 then
    -0.5 * x^2 + 60 * x - fixed_cost
  else
    1680 - x - 8100 / x

-- Maximum profit occurs at x = 90
theorem max_profit_at_90 : ∀ x : ℕ, profit 90 ≥ profit x := 
by {
  sorry
}

end max_profit_at_90_l310_310837


namespace relationship_abc_l310_310401

noncomputable def a : ℝ := 5 ^ (Real.log 3.4 / Real.log 3)
noncomputable def b : ℝ := 5 ^ (Real.log 3.6 / Real.log 3)
noncomputable def c : ℝ := (1 / 5) ^ (Real.log 0.5 / Real.log 3)

theorem relationship_abc : b > a ∧ a > c := by 
  sorry

end relationship_abc_l310_310401


namespace lollipop_count_l310_310566

theorem lollipop_count (total_cost one_lollipop_cost : ℚ) (h1 : total_cost = 90) (h2 : one_lollipop_cost = 0.75) : total_cost / one_lollipop_cost = 120 :=
by
  sorry

end lollipop_count_l310_310566


namespace variance_transformed_variable_l310_310116

noncomputable def X (ω : Ω) : ℕ := sorry  -- Assume we have a random variable X

axiom X_is_binomial_10_0_6 : X ~ binomial 10 0.6

theorem variance_transformed_variable : variance (λ ω, 3 * X ω + 9) = 21.6 :=
by
  -- Definitions for the transformed variable and binomial variance
  have variance_X : variance X = 10 * 0.6 * (1 - 0.6) := sorry
  have linear_transformation_variance : ∀ (a b : ℝ), variance (λ ω, a * X ω + b) = a^2 * variance X := sorry
  -- Conclude the proof using the given facts
  sorry

end variance_transformed_variable_l310_310116


namespace floor_sqrt_50_squared_l310_310888

theorem floor_sqrt_50_squared :
  (\lfloor real.sqrt 50 \rfloor)^2 = 49 := 
by
  sorry

end floor_sqrt_50_squared_l310_310888


namespace trapezoid_EFBA_area_l310_310185

theorem trapezoid_EFBA_area {a : ℚ} (AE BF : ℚ) (area_ABCD : ℚ) (column_areas : List ℚ)
  (h_grid : column_areas = [a, 2 * a, 4 * a, 8 * a])
  (h_total_area : 3 * (a + 2 * a + 4 * a + 8 * a) = 48)
  (h_AE : AE = 2)
  (h_BF : BF = 4) :
  let AFGB_area := 15 * a
  let triangle_EF_area := 7 * a
  let total_trapezoid_area := AFGB_area + (triangle_EF_area / 2)
  total_trapezoid_area = 352 / 15 :=
by
  sorry

end trapezoid_EFBA_area_l310_310185


namespace sqrt_floor_squared_50_l310_310870

noncomputable def sqrt_floor_squared (n : ℕ) : ℕ :=
  (Int.floor (Real.sqrt n))^2

theorem sqrt_floor_squared_50 : sqrt_floor_squared 50 = 49 := 
  by
  sorry

end sqrt_floor_squared_50_l310_310870


namespace permutation_combination_example_l310_310632

-- Definition of permutation (A) and combination (C) in Lean
def permutation (n k : ℕ): ℕ := Nat.factorial n / Nat.factorial (n - k)
def combination (n k : ℕ): ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The Lean statement of the proof problem
theorem permutation_combination_example : 
3 * permutation 3 2 + 2 * combination 4 2 = 30 := 
by 
  sorry

end permutation_combination_example_l310_310632


namespace fermat_prime_divisibility_l310_310133

def F (k : ℕ) : ℕ := 2 ^ 2 ^ k + 1

theorem fermat_prime_divisibility {m n : ℕ} (hmn : m > n) : F n ∣ (F m - 2) :=
sorry

end fermat_prime_divisibility_l310_310133


namespace certain_number_is_1_l310_310478

theorem certain_number_is_1 (z : ℕ) (hz : z % 4 = 0) :
  ∃ n : ℕ, (z * (6 + z) + n) % 2 = 1 ∧ n = 1 :=
by
  sorry

end certain_number_is_1_l310_310478


namespace total_leaves_on_farm_l310_310378

theorem total_leaves_on_farm : 
  (branches_per_tree subbranches_per_branch leaves_per_subbranch trees_on_farm : ℕ)
  (h1 : branches_per_tree = 10)
  (h2 : subbranches_per_branch = 40)
  (h3 : leaves_per_subbranch = 60)
  (h4 : trees_on_farm = 4) :
  (trees_on_farm * branches_per_tree * subbranches_per_branch * leaves_per_subbranch = 96000) :=
by
  sorry

end total_leaves_on_farm_l310_310378


namespace triangle_inequality_a2_a3_a4_l310_310920

variables {a1 a2 a3 a4 d : ℝ}

def is_arithmetic_sequence (a1 a2 a3 a4 : ℝ) (d : ℝ) : Prop :=
  a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d

def positive_terms (a1 a2 a3 a4 : ℝ) : Prop :=
  0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 0 < a4

theorem triangle_inequality_a2_a3_a4 (h1: positive_terms a1 a2 a3 a4)
  (h2: is_arithmetic_sequence a1 a2 a3 a4 d) (h3: d > 0) :
  (a2 + a3 > a4) ∧ (a2 + a4 > a3) ∧ (a3 + a4 > a2) :=
sorry

end triangle_inequality_a2_a3_a4_l310_310920


namespace parabola_directrix_symmetry_l310_310976

theorem parabola_directrix_symmetry:
  (∃ (d : ℝ), (∀ x : ℝ, x = d ↔ 
  (∃ y : ℝ, y^2 = (1 / 2) * x) ∧
  (∀ y : ℝ, x = (1 / 8)) → x = - (1 / 8))) :=
sorry

end parabola_directrix_symmetry_l310_310976


namespace ellipse_hyperbola_foci_l310_310776

theorem ellipse_hyperbola_foci (a b : ℝ) 
    (h1 : b^2 - a^2 = 25) 
    (h2 : a^2 + b^2 = 49) : 
    |a * b| = 2 * Real.sqrt 111 := 
by 
  -- proof omitted 
  sorry

end ellipse_hyperbola_foci_l310_310776


namespace min_troublemakers_in_class_l310_310811

noncomputable def min_troublemakers : ℕ :=
  10

theorem min_troublemakers_in_class :
  (∃ t l : ℕ, t + l = 29 ∧ t + l - 1 = 29 ∧
   (∀ i : ℕ, i < 29 → (i % 3 = 0 → ∃ t : ℕ, t = 1) ∧ 
   (i % 3 ≠ 0 → ∃ t : ℕ, t = 2))) →
   min_troublemakers = 10 :=
by
  sorry

end min_troublemakers_in_class_l310_310811


namespace total_number_of_water_filled_jars_l310_310835

theorem total_number_of_water_filled_jars : 
  ∃ (x : ℕ), 28 = x * (1/4 + 1/2 + 1) ∧ 3 * x = 48 :=
by
  sorry

end total_number_of_water_filled_jars_l310_310835


namespace total_cost_is_eight_times_l310_310323

theorem total_cost_is_eight_times (x : ℝ) 
  (h1 : ∀ t, x + t = 2 * x)
  (h2 : ∀ b, x + b = 5 * x)
  (h3 : ∀ s, x + s = 3 * x) :
  ∃ t b s, x + t + b + s = 8 * x :=
by
  sorry

end total_cost_is_eight_times_l310_310323


namespace age_of_youngest_child_l310_310483

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 60) : 
  x = 6 :=
sorry

end age_of_youngest_child_l310_310483


namespace no_solution_for_x_l310_310595

noncomputable def proof_problem : Prop :=
  ∀ x : ℝ, ⌊x⌋ + ⌊2*x⌋ + ⌊4*x⌋ + ⌊8*x⌋ + ⌊16*x⌋ + ⌊32*x⌋ ≠ 12345

theorem no_solution_for_x : proof_problem :=
  by
    intro x
    sorry

end no_solution_for_x_l310_310595


namespace find_bounds_l310_310124

open Set

variable {U : Type} [TopologicalSpace U]

def A := {x : ℝ | 3 ≤ x ∧ x ≤ 4}
def C_UA := {x : ℝ | x > 4 ∨ x < 3}

theorem find_bounds (T : Type) [TopologicalSpace T] : 3 = 3 ∧ 4 = 4 := 
 by sorry

end find_bounds_l310_310124


namespace train_cross_time_l310_310375

-- Define the given conditions
def train_length : ℕ := 100
def train_speed_kmph : ℕ := 45
def total_length : ℕ := 275
def seconds_in_hour : ℕ := 3600
def meters_in_km : ℕ := 1000

-- Convert the speed from km/hr to m/s
noncomputable def train_speed_mps : ℚ := (train_speed_kmph * meters_in_km) / seconds_in_hour

-- The time to cross the bridge
noncomputable def time_to_cross (train_length total_length : ℕ) (train_speed_mps : ℚ) : ℚ :=
  total_length / train_speed_mps

-- The statement we want to prove
theorem train_cross_time : time_to_cross train_length total_length train_speed_mps = 30 :=
by
  sorry

end train_cross_time_l310_310375


namespace min_max_value_of_expr_l310_310963

theorem min_max_value_of_expr (p q r s : ℝ)
  (h1 : p + q + r + s = 10)
  (h2 : p^2 + q^2 + r^2 + s^2 = 20) :
  ∃ m M : ℝ, m = 2 ∧ M = 0 ∧ ∀ x, (x = 3 * (p^3 + q^3 + r^3 + s^3) - 2 * (p^4 + q^4 + r^4 + s^4)) → m ≤ x ∧ x ≤ M :=
sorry

end min_max_value_of_expr_l310_310963


namespace product_mod_7_l310_310210

theorem product_mod_7 : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
by
  have h1 : 2021 % 7 = 6 := by sorry
  have h2 : 2022 % 7 = 0 := by sorry
  have h3 : 2023 % 7 = 1 := by sorry
  have h4 : 2024 % 7 = 2 := by sorry
  sorry

end product_mod_7_l310_310210


namespace cone_volume_l310_310609

theorem cone_volume (S : ℝ) (hPos : S > 0) : 
  let R := Real.sqrt (S / 7)
  let H := Real.sqrt (5 * S)
  let V := (π * S * (Real.sqrt (5 * S))) / 21
  (π * R * R * H / 3) = V := 
sorry

end cone_volume_l310_310609


namespace lollipop_count_l310_310569

theorem lollipop_count (total_cost : ℝ) (cost_per_lollipop : ℝ) (h1 : total_cost = 90) (h2 : cost_per_lollipop = 0.75) : 
  total_cost / cost_per_lollipop = 120 :=
by 
  sorry

end lollipop_count_l310_310569


namespace area_of_rectangular_field_l310_310365

def length (L : ℝ) : Prop := L > 0
def breadth (L : ℝ) (B : ℝ) : Prop := B = 0.6 * L
def perimeter (L : ℝ) (B : ℝ) : Prop := 2 * L + 2 * B = 800
def area (L : ℝ) (B : ℝ) (A : ℝ) : Prop := A = L * B

theorem area_of_rectangular_field (L B A : ℝ) 
  (h1 : breadth L B) 
  (h2 : perimeter L B) : 
  area L B 37500 :=
sorry

end area_of_rectangular_field_l310_310365


namespace _l310_310605

example : coeff (x^3) ((1 + 2*x)^6) = 160 :=
by
  -- We use the binomial theorem here
  have h := nat.choose 6 3
  -- Simplification step
  suffices h' : 2^3 * h = 160
  { exact h' }
  have h_binom : h = 6 * 5 * 4 / (3 * 2 * 1) := nat.choose_eq_factorial_div_factorial 6 3
  have h_calculate : 6 * 5 * 4 / (3 * 2 * 1) = 20 := by norm_num
  rw h_binom at h_calculate
  have h_pow : 2^3 = 8 := by norm_num
  rw [←h_pow, ←h_calculate]
  norm_num
  length 4 sorry

end _l310_310605


namespace combined_selling_price_correct_l310_310192

noncomputable def cost_A : ℝ := 500
noncomputable def cost_B : ℝ := 800
noncomputable def profit_A_perc : ℝ := 0.10
noncomputable def profit_B_perc : ℝ := 0.15
noncomputable def tax_perc : ℝ := 0.05
noncomputable def packaging_fee : ℝ := 50

-- Calculating selling prices before tax and fees
noncomputable def selling_price_A_before_tax_fees : ℝ := cost_A * (1 + profit_A_perc)
noncomputable def selling_price_B_before_tax_fees : ℝ := cost_B * (1 + profit_B_perc)

-- Calculating taxes
noncomputable def tax_A : ℝ := selling_price_A_before_tax_fees * tax_perc
noncomputable def tax_B : ℝ := selling_price_B_before_tax_fees * tax_perc

-- Adding tax to selling prices
noncomputable def selling_price_A_incl_tax : ℝ := selling_price_A_before_tax_fees + tax_A
noncomputable def selling_price_B_incl_tax : ℝ := selling_price_B_before_tax_fees + tax_B

-- Adding packaging and shipping fees
noncomputable def final_selling_price_A : ℝ := selling_price_A_incl_tax + packaging_fee
noncomputable def final_selling_price_B : ℝ := selling_price_B_incl_tax + packaging_fee

-- Combined selling price
noncomputable def combined_selling_price : ℝ := final_selling_price_A + final_selling_price_B

theorem combined_selling_price_correct : 
  combined_selling_price = 1643.5 := by
  sorry

end combined_selling_price_correct_l310_310192


namespace triangle_inequality_a2_a3_a4_l310_310919

variables {a1 a2 a3 a4 d : ℝ}

def is_arithmetic_sequence (a1 a2 a3 a4 : ℝ) (d : ℝ) : Prop :=
  a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d

def positive_terms (a1 a2 a3 a4 : ℝ) : Prop :=
  0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 0 < a4

theorem triangle_inequality_a2_a3_a4 (h1: positive_terms a1 a2 a3 a4)
  (h2: is_arithmetic_sequence a1 a2 a3 a4 d) (h3: d > 0) :
  (a2 + a3 > a4) ∧ (a2 + a4 > a3) ∧ (a3 + a4 > a2) :=
sorry

end triangle_inequality_a2_a3_a4_l310_310919


namespace max_men_with_all_amenities_marrried_l310_310006

theorem max_men_with_all_amenities_marrried :
  let total_men := 100
  let married_men := 85
  let men_with_TV := 75
  let men_with_radio := 85
  let men_with_AC := 70
  (∀ s : Finset ℕ, s.card ≤ total_men) →
  (∀ s : Finset ℕ, s.card ≤ married_men) →
  (∀ s : Finset ℕ, s.card ≤ men_with_TV) →
  (∀ s : Finset ℕ, s.card ≤ men_with_radio) →
  (∀ s : Finset ℕ, s.card ≤ men_with_AC) →
  (∀ s : Finset ℕ, s.card ≤ min married_men (min men_with_TV (min men_with_radio men_with_AC))) :=
by
  intros
  sorry

end max_men_with_all_amenities_marrried_l310_310006


namespace total_goals_l310_310533

def first_period_goals (k: ℕ) : ℕ :=
  k

def second_period_goals (k: ℕ) : ℕ :=
  2 * k

def spiders_first_period_goals (k: ℕ) : ℕ :=
  k / 2

def spiders_second_period_goals (s1: ℕ) : ℕ :=
  s1 * s1

def third_period_goals (k1 k2: ℕ) : ℕ :=
  2 * (k1 + k2)

def spiders_third_period_goals (s2: ℕ) : ℕ :=
  s2

def apply_bonus (goals: ℕ) (multiple: ℕ) : ℕ :=
  if goals % multiple = 0 then goals + 1 else goals

theorem total_goals (k1 k2 s1 s2 k3 s3 : ℕ) :
  first_period_goals 2 = k1 →
  second_period_goals k1 = k2 →
  spiders_first_period_goals k1 = s1 →
  spiders_second_period_goals s1 = s2 →
  third_period_goals k1 k2 = k3 →
  apply_bonus k3 3 = k3 + 1 →
  apply_bonus s2 2 = s2 →
  spiders_third_period_goals s2 = s3 →
  apply_bonus s3 2 = s3 →
  2 + k2 + (k3 + 1) + (s1 + s2 + s3) = 22 :=
by
  sorry

end total_goals_l310_310533


namespace arithmetic_sequence_common_difference_l310_310925

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
    (h1 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2)
    (h2 : (S 2017) / 2017 - (S 17) / 17 = 100) :
    d = 1/10 := 
by sorry

end arithmetic_sequence_common_difference_l310_310925


namespace cost_of_purchasing_sandwiches_and_sodas_l310_310517

def sandwich_price : ℕ := 4
def soda_price : ℕ := 1
def num_sandwiches : ℕ := 6
def num_sodas : ℕ := 5
def total_cost : ℕ := 29

theorem cost_of_purchasing_sandwiches_and_sodas :
  (num_sandwiches * sandwich_price + num_sodas * soda_price) = total_cost :=
by
  sorry

end cost_of_purchasing_sandwiches_and_sodas_l310_310517


namespace degree_of_divisor_l310_310038

theorem degree_of_divisor (f d q r : Polynomial ℝ) 
  (hf : f.degree = 15) 
  (hq : q.degree = 9) 
  (hr : r.degree = 4) 
  (hr_poly : r = (Polynomial.C 5) * (Polynomial.X^4) + (Polynomial.C 6) * (Polynomial.X^3) - (Polynomial.C 2) * (Polynomial.X) + (Polynomial.C 7)) 
  (hdiv : f = d * q + r) : 
  d.degree = 6 := 
sorry

end degree_of_divisor_l310_310038


namespace area_of_original_triangle_l310_310940

theorem area_of_original_triangle (a : Real) (S_intuitive : Real) : 
  a = 2 -> S_intuitive = (Real.sqrt 3) -> (S_intuitive / (Real.sqrt 2 / 4)) = 2 * Real.sqrt 6 := 
by
  sorry

end area_of_original_triangle_l310_310940


namespace min_bn_of_arithmetic_sequence_l310_310916

theorem min_bn_of_arithmetic_sequence :
  (∃ n : ℕ, 1 ≤ n ∧ b_n = n + 1 + 7 / n ∧ (∀ m : ℕ, 1 ≤ m → b_m ≥ b_n)) :=
sorry

def a_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

def S_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else n * (n + 1) / 2

def b_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else (2 * S_n n + 7) / n

end min_bn_of_arithmetic_sequence_l310_310916


namespace number_of_students_on_wednesday_l310_310969

-- Define the problem conditions
variables (W T : ℕ)

-- Define the given conditions
def condition1 : Prop := T = W - 9
def condition2 : Prop := W + T = 65

-- Define the theorem to prove
theorem number_of_students_on_wednesday (h1 : condition1 W T) (h2 : condition2 W T) : W = 37 :=
by
  sorry

end number_of_students_on_wednesday_l310_310969


namespace math_books_count_l310_310613

theorem math_books_count (M H : ℕ) :
  M + H = 90 →
  4 * M + 5 * H = 396 →
  H = 90 - M →
  M = 54 :=
by
  intro h1 h2 h3
  sorry

end math_books_count_l310_310613


namespace min_troublemakers_29_l310_310808

noncomputable def min_troublemakers (n : ℕ) : ℕ :=
sorry

theorem min_troublemakers_29 : min_troublemakers 29 = 10 := 
sorry

end min_troublemakers_29_l310_310808


namespace perfect_squares_100_to_400_l310_310100

theorem perfect_squares_100_to_400 :
  {n : ℕ | 100 ≤ n^2 ∧ n^2 ≤ 400}.card = 11 :=
by {
  sorry
}

end perfect_squares_100_to_400_l310_310100


namespace cat_finishes_food_on_sunday_l310_310146

-- Define the constants and parameters
def daily_morning_consumption : ℚ := 2 / 5
def daily_evening_consumption : ℚ := 1 / 5
def total_food : ℕ := 8
def days_in_week : ℕ := 7

-- Define the total daily consumption
def total_daily_consumption : ℚ := daily_morning_consumption + daily_evening_consumption

-- Define the sum of consumptions over each day until the day when all food is consumed
def food_remaining_after_days (days : ℕ) : ℚ := total_food - days * total_daily_consumption

-- Proposition that the food is finished on Sunday
theorem cat_finishes_food_on_sunday :
  ∃ days : ℕ, (food_remaining_after_days days ≤ 0) ∧ days ≡ 7 [MOD days_in_week] :=
sorry

end cat_finishes_food_on_sunday_l310_310146


namespace domain_of_sqrt_ln_l310_310975

def domain_function (x : ℝ) : Prop := x - 1 ≥ 0 ∧ 2 - x > 0

theorem domain_of_sqrt_ln (x : ℝ) : domain_function x ↔ 1 ≤ x ∧ x < 2 := by
  sorry

end domain_of_sqrt_ln_l310_310975


namespace geometric_sequence_product_l310_310577

theorem geometric_sequence_product {a : ℕ → ℝ} (q : ℝ) (h1 : |q| ≠ 1) :
  a 1 = 1 → (∀ n, a n = a 1 * (q ^ (n - 1))) → a 11 = a 1 * a 2 * a 3 * a 4 * a 5 :=
by {
  intros h2 h3,
  sorry
}

end geometric_sequence_product_l310_310577


namespace proof_problem_l310_310128

def f (x : ℤ) : ℤ := 3 * x + 5
def g (x : ℤ) : ℤ := 4 * x - 3

theorem proof_problem : 
  (f (g (f (g 3)))) / (g (f (g (f 3)))) = (380 / 653) := 
  by 
    sorry

end proof_problem_l310_310128


namespace solution_eq1_solution_eq2_l310_310601

-- Definitions corresponding to the conditions of the problem.
def eq1 (x : ℝ) : Prop := 16 * x^2 = 49
def eq2 (x : ℝ) : Prop := (x - 2)^2 = 64

-- Statements for the proof problem.
theorem solution_eq1 (x : ℝ) : eq1 x → (x = 7 / 4 ∨ x = - (7 / 4)) :=
by
  intro h
  sorry

theorem solution_eq2 (x : ℝ) : eq2 x → (x = 10 ∨ x = -6) :=
by
  intro h
  sorry

end solution_eq1_solution_eq2_l310_310601


namespace fraction_simplification_l310_310622

theorem fraction_simplification : 
  (1/5 - 1/6) / (1/3 - 1/4) = 2/5 := 
by 
  sorry

end fraction_simplification_l310_310622


namespace smallest_b_value_l310_310767

theorem smallest_b_value (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a - b = 7) 
    (h₄ : (Nat.gcd ((a^3 + b^3) / (a + b)) (a^2 * b)) = 12) : b = 6 :=
by
    -- proof goes here
    sorry

end smallest_b_value_l310_310767


namespace ride_cost_l310_310155

theorem ride_cost (joe_age_over_18 : Prop)
                   (joe_brother_age : Nat)
                   (joe_entrance_fee : ℝ)
                   (brother_entrance_fee : ℝ)
                   (total_spending : ℝ)
                   (rides_per_person : Nat)
                   (total_persons : Nat)
                   (total_entrance_fee : ℝ)
                   (amount_spent_on_rides : ℝ)
                   (total_rides : Nat) :
  joe_entrance_fee = 6 →
  brother_entrance_fee = 5 →
  total_spending = 20.5 →
  rides_per_person = 3 →
  total_persons = 3 →
  total_entrance_fee = 16 →
  amount_spent_on_rides = (total_spending - total_entrance_fee) →
  total_rides = (rides_per_person * total_persons) →
  (amount_spent_on_rides / total_rides) = 0.50 :=
by
  sorry

end ride_cost_l310_310155


namespace perpendicular_lines_k_value_l310_310866

theorem perpendicular_lines_k_value (k : ℚ) : (∀ x y : ℚ, y = 3 * x + 7) ∧ (∀ x y : ℚ, 4 * y + k * x = 4) → k = 4 / 3 :=
by
  sorry

end perpendicular_lines_k_value_l310_310866


namespace integer_solution_exists_l310_310394

theorem integer_solution_exists : ∃ n : ℤ, (⌊(n^2 : ℚ) / 3⌋ - ⌊(n : ℚ) / 2⌋^2 = 3) ∧ n = 6 := by
  sorry

end integer_solution_exists_l310_310394


namespace floor_sqrt_50_squared_l310_310887

theorem floor_sqrt_50_squared :
  (\lfloor real.sqrt 50 \rfloor)^2 = 49 := 
by
  sorry

end floor_sqrt_50_squared_l310_310887


namespace q_one_eq_five_l310_310384

variable (q : ℝ → ℝ)
variable (h : q 1 = 5)

theorem q_one_eq_five : q 1 = 5 :=
by sorry

end q_one_eq_five_l310_310384


namespace mod_2021_2022_2023_2024_eq_zero_mod_7_l310_310211

theorem mod_2021_2022_2023_2024_eq_zero_mod_7 :
  (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end mod_2021_2022_2023_2024_eq_zero_mod_7_l310_310211


namespace quadratic_eq_real_roots_l310_310945

theorem quadratic_eq_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + 2*m = 0) → m ≤ 9 / 8 :=
by
  sorry

end quadratic_eq_real_roots_l310_310945


namespace book_pages_l310_310956

-- Define the number of pages read each day
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5
def pages_tomorrow : ℕ := 35

-- Total number of pages in the book
def total_pages : ℕ := pages_yesterday + pages_today + pages_tomorrow

-- Proof that the total number of pages is 100
theorem book_pages : total_pages = 100 := by
  -- Skip the detailed proof
  sorry

end book_pages_l310_310956


namespace undefined_values_l310_310904

theorem undefined_values (a : ℝ) : a = -3 ∨ a = 3 ↔ (a^2 - 9 = 0) := sorry

end undefined_values_l310_310904


namespace part_a_part_b_part_c_l310_310165

namespace EulerianBridgeProblem

-- Graph definition with islands and bridges
noncomputable def Graph := Type

-- Vertex definition as islands
variable (Island : Type)

-- Context of Eulerian path or walk
structure EulerianPath (G : Graph) :=
(vertices : Finset Island)
(edges : Finset (Island × Island))
(adj : ∀ {u v : Island}, (u, v) ∈ edges ∨ (v, u) ∈ edges)
(degree : Island → Nat)
(path : List Island)
(is_eulerian : ∀ e ∈ edges, ∃! i, (path.nth i = some (e.1) ∧ path.nth (i+1) = some (e.2))
 ∨ (path.nth i = some (e.2) ∧ path.nth (i+1) = some (e.1)))

variable {G : Graph}
variable {T : Island}

-- Part (a): How many bridges lead from Troekratny if the tourist did not start and did not finish at this island?
theorem part_a (h : EulerianPath G) (start_ne_T : h.path.head ≠ some T) (end_ne_T : h.path.last ≠ some T) :
  h.degree T = 6 := sorry

-- Part (b): How many bridges lead from Troekratny if the tourist started but did not finish at this island?
theorem part_b (h : EulerianPath G) (start_T : h.path.head = some T) (end_ne_T : h.path.last ≠ some T) :
  h.degree T = 5 := sorry

-- Part (c): How many bridges lead from Troekratny if the tourist started and finished at this island?
theorem part_c (h : EulerianPath G) (start_T : h.path.head = some T) (end_T : h.path.last = some T) :
  h.degree T = 4 := sorry

end EulerianBridgeProblem

end part_a_part_b_part_c_l310_310165


namespace find_a_of_inequality_solution_set_l310_310571

theorem find_a_of_inequality_solution_set
  (a : ℝ)
  (h : ∀ x : ℝ, |a * x + 2| < 6 ↔ -1 < x ∧ x < 2) :
  a = -4 :=
sorry

end find_a_of_inequality_solution_set_l310_310571


namespace John_can_lift_now_l310_310122

def originalWeight : ℕ := 135
def trainingIncrease : ℕ := 265
def bracerIncreaseFactor : ℕ := 6

def newWeight : ℕ := originalWeight + trainingIncrease
def bracerIncrease : ℕ := newWeight * bracerIncreaseFactor
def totalWeight : ℕ := newWeight + bracerIncrease

theorem John_can_lift_now :
  totalWeight = 2800 :=
by
  -- proof steps go here
  sorry

end John_can_lift_now_l310_310122


namespace B_and_D_know_their_grades_l310_310659

-- Define the students and their respective grades
inductive Grade : Type
| excellent : Grade
| good : Grade

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student

-- Define the information given in the problem regarding which student sees whose grade
def sees (s1 s2 : Student) : Prop :=
  (s1 = Student.A ∧ (s2 = Student.B ∨ s2 = Student.C)) ∨
  (s1 = Student.B ∧ s2 = Student.C) ∨
  (s1 = Student.D ∧ s2 = Student.A)

-- Define the condition that there are 2 excellent and 2 good grades
def grade_distribution (gA gB gC gD : Grade) : Prop :=
  gA ≠ gB → (gC = gA ∨ gC = gB) ∧ (gD = gA ∨ gD = gB) ∧
  (gA = Grade.excellent ∧ (gB = Grade.good ∨ gC = Grade.good ∨ gD = Grade.good)) ∧
  (gA = Grade.good ∧ (gB = Grade.excellent ∨ gC = Grade.excellent ∨ gD = Grade.excellent))

-- Student A's statement after seeing B and C's grades
def A_statement (gA gB gC : Grade) : Prop :=
  (gB = gA ∨ gC = gA) ∨ (gB ≠ gA ∧ gC ≠ gA)

-- Formal proof goal: Prove that B and D can know their own grades based on the information provided
theorem B_and_D_know_their_grades (gA gB gC gD : Grade)
  (h1 : grade_distribution gA gB gC gD)
  (h2 : A_statement gA gB gC)
  (h3 : sees Student.A Student.B)
  (h4 : sees Student.A Student.C)
  (h5 : sees Student.B Student.C)
  (h6 : sees Student.D Student.A) :
  (gB ≠ Grade.excellent ∨ gB ≠ Grade.good) ∧ (gD ≠ Grade.excellent ∨ gD ≠ Grade.good) :=
by sorry

end B_and_D_know_their_grades_l310_310659


namespace hcf_of_12_and_15_l310_310770

-- Definitions of LCM and HCF
def LCM (a b : ℕ) : ℕ := sorry  -- Placeholder for actual LCM definition
def HCF (a b : ℕ) : ℕ := sorry  -- Placeholder for actual HCF definition

theorem hcf_of_12_and_15 :
  LCM 12 15 = 60 → HCF 12 15 = 3 :=
by
  sorry

end hcf_of_12_and_15_l310_310770


namespace factorial_difference_l310_310856

theorem factorial_difference :
  10! - 9! = 3265920 :=
by
  sorry

end factorial_difference_l310_310856


namespace find_triple_sum_l310_310421

theorem find_triple_sum (x y z : ℝ) 
  (h1 : y + z = 20 - 4 * x)
  (h2 : x + z = 1 - 4 * y)
  (h3 : x + y = -12 - 4 * z) :
  3 * x + 3 * y + 3 * z = 9 / 2 := 
sorry

end find_triple_sum_l310_310421


namespace total_cost_shorts_tshirt_boots_shinguards_l310_310307

variable (x : ℝ)

-- Definitions provided in the problem statement.
def cost_shorts : ℝ := x
def cost_shorts_and_tshirt : ℝ := 2 * x
def cost_shorts_and_boots : ℝ := 5 * x
def cost_shorts_and_shinguards : ℝ := 3 * x

-- The proof goal to verify:
theorem total_cost_shorts_tshirt_boots_shinguards : 
  (cost_shorts x + (cost_shorts_and_tshirt x - cost_shorts x) + 
   (cost_shorts_and_boots x - cost_shorts x) + 
   (cost_shorts_and_shinguards x - cost_shorts x)) = 8 * x := by 
  sorry

end total_cost_shorts_tshirt_boots_shinguards_l310_310307


namespace tiffany_lives_problem_l310_310492

/-- Tiffany's lives problem -/
theorem tiffany_lives_problem (L : ℤ) (h1 : 43 - L + 27 = 56) : L = 14 :=
by {
  sorry
}

end tiffany_lives_problem_l310_310492


namespace map_length_to_reality_l310_310840

def scale : ℝ := 500
def length_map : ℝ := 7.2
def length_actual : ℝ := 3600

theorem map_length_to_reality : length_actual = length_map * scale :=
by
  sorry

end map_length_to_reality_l310_310840


namespace total_cost_shorts_tshirt_boots_shinguards_l310_310308

variable (x : ℝ)

-- Definitions provided in the problem statement.
def cost_shorts : ℝ := x
def cost_shorts_and_tshirt : ℝ := 2 * x
def cost_shorts_and_boots : ℝ := 5 * x
def cost_shorts_and_shinguards : ℝ := 3 * x

-- The proof goal to verify:
theorem total_cost_shorts_tshirt_boots_shinguards : 
  (cost_shorts x + (cost_shorts_and_tshirt x - cost_shorts x) + 
   (cost_shorts_and_boots x - cost_shorts x) + 
   (cost_shorts_and_shinguards x - cost_shorts x)) = 8 * x := by 
  sorry

end total_cost_shorts_tshirt_boots_shinguards_l310_310308


namespace geometric_sequence_properties_l310_310939

theorem geometric_sequence_properties (a b c : ℝ) (r : ℝ)
    (h1 : a = -2 * r)
    (h2 : b = a * r)
    (h3 : c = b * r)
    (h4 : -8 = c * r) :
    b = -4 ∧ a * c = 16 :=
by
  sorry

end geometric_sequence_properties_l310_310939


namespace combined_value_l310_310113

theorem combined_value (a b : ℝ) (h1 : 0.005 * a = 95 / 100) (h2 : b = 3 * a - 50) : a + b = 710 := by
  sorry

end combined_value_l310_310113


namespace determine_k_l310_310057

theorem determine_k (k : ℝ) : (1 - 3 * k * (-2/3) = 7 * 3) → k = 10 :=
by
  intro h
  sorry

end determine_k_l310_310057


namespace trigonometric_identity_l310_310226

theorem trigonometric_identity :
  7 * 6 * (1 / Real.tan (2 * Real.pi * 10 / 360) + Real.tan (2 * Real.pi * 5 / 360)) 
  = 7 * 6 * (1 / Real.sin (2 * Real.pi * 10 / 360)) := 
sorry

end trigonometric_identity_l310_310226


namespace quadratic_real_roots_leq_l310_310943

theorem quadratic_real_roots_leq (m : ℝ) :
  ∃ x : ℝ, x^2 - 3 * x + 2 * m = 0 → m ≤ 9 / 8 :=
by
  sorry

end quadratic_real_roots_leq_l310_310943


namespace parabola_hyperbola_tangent_l310_310058

noncomputable def parabola : ℝ → ℝ := λ x => x^2 + 5

noncomputable def hyperbola (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => y^2 - m * x^2 = 1

theorem parabola_hyperbola_tangent (m : ℝ) : 
  (m = 10 + 4*Real.sqrt 6 ∨ m = 10 - 4*Real.sqrt 6) →
  ∃ x y, parabola x = y ∧ hyperbola m x y ∧ 
    ∃ c b a, a * y^2 + b * y + c = 0 ∧ a = 1 ∧ c = 5 * m - 1 ∧ b = -m ∧ b^2 - 4*a*c = 0 :=
by
  sorry

end parabola_hyperbola_tangent_l310_310058


namespace ratio_of_red_to_total_simplified_l310_310638

def number_of_red_haired_children := 9
def total_number_of_children := 48

theorem ratio_of_red_to_total_simplified:
  (number_of_red_haired_children: ℚ) / (total_number_of_children: ℚ) = (3 : ℚ) / (16 : ℚ) := 
by
  sorry

end ratio_of_red_to_total_simplified_l310_310638


namespace expression_undefined_iff_l310_310908

theorem expression_undefined_iff (a : ℝ) : (a^2 - 9 = 0) ↔ (a = 3 ∨ a = -3) :=
sorry

end expression_undefined_iff_l310_310908


namespace find_a4_b4_l310_310220

theorem find_a4_b4 :
  ∃ (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ),
    a₁ * b₁ + a₂ * b₃ = 1 ∧
    a₁ * b₂ + a₂ * b₄ = 0 ∧
    a₃ * b₁ + a₄ * b₃ = 0 ∧
    a₃ * b₂ + a₄ * b₄ = 1 ∧
    a₂ * b₃ = 7 ∧
    a₄ * b₄ = -6 :=
by
  sorry

end find_a4_b4_l310_310220


namespace group_total_people_l310_310229

theorem group_total_people (k : ℕ) (h1 : k = 7) (h2 : ((n - k) / n : ℝ) - (k / n : ℝ) = 0.30000000000000004) : n = 20 :=
  sorry

end group_total_people_l310_310229


namespace prove_two_minus_a_l310_310936

theorem prove_two_minus_a (a b : ℚ) 
  (h1 : 2 * a + 3 = 5 - b) 
  (h2 : 5 + 2 * b = 10 + a) : 
  2 - a = 11 / 5 := 
by 
  sorry

end prove_two_minus_a_l310_310936


namespace petya_purchase_cost_l310_310329

theorem petya_purchase_cost (x : ℝ) 
  (h1 : ∃ shirt_cost : ℝ, x + shirt_cost = 2 * x)
  (h2 : ∃ boots_cost : ℝ, x + boots_cost = 5 * x)
  (h3 : ∃ shin_guards_cost : ℝ, x + shin_guards_cost = 3 * x) :
  ∃ total_cost : ℝ, total_cost = 8 * x :=
by 
  sorry

end petya_purchase_cost_l310_310329


namespace no_representation_of_expr_l310_310276

theorem no_representation_of_expr :
  ¬ ∃ f g : ℝ → ℝ, (∀ x y : ℝ, 1 + x ^ 2016 * y ^ 2016 = f x * g y) :=
by
  sorry

end no_representation_of_expr_l310_310276


namespace real_roots_range_l310_310470

theorem real_roots_range (k : ℝ) : 
  (∃ x : ℝ, k*x^2 - 6*x + 9 = 0) ↔ k ≤ 1 :=
sorry

end real_roots_range_l310_310470


namespace square_equiv_l310_310049

theorem square_equiv (x : ℝ) : 
  (7 - (x^3 - 49)^(1/3))^2 = 
  49 - 14 * (x^3 - 49)^(1/3) + ((x^3 - 49)^(1/3))^2 := 
by 
  sorry

end square_equiv_l310_310049


namespace sqrt_floor_square_eq_49_l310_310875

theorem sqrt_floor_square_eq_49 : (⌊Real.sqrt 50⌋)^2 = 49 :=
by
  have h1 : 7 < Real.sqrt 50, from (by norm_num : 7 < Real.sqrt 50),
  have h2 : Real.sqrt 50 < 8, from (by norm_num : Real.sqrt 50 < 8),
  have floor_sqrt_50_eq_7 : ⌊Real.sqrt 50⌋ = 7, from Int.floor_eq_iff.mpr ⟨h1, h2⟩,
  calc
    (⌊Real.sqrt 50⌋)^2 = (7)^2 : by rw [floor_sqrt_50_eq_7]
                  ... = 49 : by norm_num,
  sorry -- omit the actual proof

end sqrt_floor_square_eq_49_l310_310875


namespace quadratic_eq_real_roots_l310_310946

theorem quadratic_eq_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + 2*m = 0) → m ≤ 9 / 8 :=
by
  sorry

end quadratic_eq_real_roots_l310_310946


namespace total_cost_is_eight_times_l310_310296

theorem total_cost_is_eight_times (cost_shorts cost_shirt cost_boots cost_shin_guards : ℕ) :
  (cost_shorts + cost_shirt = 2 * cost_shorts) →
  (cost_shorts + cost_boots = 5 * cost_shorts) →
  (cost_shorts + cost_shin_guards = 3 * cost_shorts) →
  cost_shorts + cost_shirt + cost_boots + cost_shin_guards = 8 * cost_shorts := 
by 
  intros h_shirt h_boots h_shin_guards 
  sorry

end total_cost_is_eight_times_l310_310296


namespace length_of_first_train_is_correct_l310_310198

noncomputable def length_of_first_train 
  (speed_first_train_kmph : ℝ)
  (length_second_train_m : ℝ)
  (speed_second_train_kmph : ℝ)
  (time_crossing_s : ℝ) : ℝ :=
  let speed_first_train_mps := (speed_first_train_kmph * 1000) / 3600
  let speed_second_train_mps := (speed_second_train_kmph * 1000) / 3600
  let relative_speed_mps := speed_first_train_mps + speed_second_train_mps
  let total_distance_m := relative_speed_mps * time_crossing_s
  total_distance_m - length_second_train_m

theorem length_of_first_train_is_correct :
  length_of_first_train 50 112 82 6 = 108.02 :=
by
  sorry

end length_of_first_train_is_correct_l310_310198


namespace english_class_students_l310_310951

variables (e f s u v w : ℕ)

theorem english_class_students
  (h1 : e + u + v + w + f + s + 2 = 40)
  (h2 : e + u + v = 3 * (f + w))
  (h3 : e + u + w = 2 * (s + v)) : 
  e = 30 := 
sorry

end english_class_students_l310_310951


namespace probability_of_y_gt_2x_l310_310669

noncomputable def probability_y_gt_2x : ℝ := 
  (∫ x in (0:ℝ)..(1000:ℝ), ∫ y in (2*x)..(2000:ℝ), (1 / (1000 * 2000) : ℝ)) * (1000 * 2000)

theorem probability_of_y_gt_2x : probability_y_gt_2x = 0.5 := sorry

end probability_of_y_gt_2x_l310_310669


namespace minimum_value_of_f_l310_310254

open Real

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 + sin x

theorem minimum_value_of_f (x : ℝ) (h : abs x ≤ π / 4) : 
  ∃ m : ℝ, (∀ y : ℝ, f y ≥ m) ∧ m = 1 / 2 - sqrt 2 / 2 :=
sorry

end minimum_value_of_f_l310_310254


namespace susan_fraction_apples_given_out_l310_310079

theorem susan_fraction_apples_given_out (frank_apples : ℕ) (frank_sold_fraction : ℚ) 
  (total_remaining_apples : ℕ) (susan_multiple : ℕ) 
  (H1 : frank_apples = 36) 
  (H2 : susan_multiple = 3) 
  (H3 : frank_sold_fraction = 1 / 3) 
  (H4 : total_remaining_apples = 78) :
  let susan_apples := susan_multiple * frank_apples
  let frank_sold_apples := frank_sold_fraction * frank_apples
  let frank_remaining_apples := frank_apples - frank_sold_apples
  let total_before_susan_gave_out := susan_apples + frank_remaining_apples
  let susan_gave_out := total_before_susan_gave_out - total_remaining_apples
  let susan_gave_fraction := susan_gave_out / susan_apples
  susan_gave_fraction = 1 / 2 :=
by
  sorry

end susan_fraction_apples_given_out_l310_310079


namespace probability_top_three_same_color_l310_310195

/-- 
  A theorem stating the probability that the top three cards from a shuffled 
  standard deck of 52 cards are all of the same color is \(\frac{12}{51}\).
-/
theorem probability_top_three_same_color : 
  let deck := 52
  let colors := 2
  let cards_per_color := 26
  let favorable_outcomes := 2 * 26 * 25 * 24
  let total_outcomes := 52 * 51 * 50
  favorable_outcomes / total_outcomes = 12 / 51 :=
by
  sorry

end probability_top_three_same_color_l310_310195


namespace least_positive_three_digit_multiple_of_9_l310_310820

theorem least_positive_three_digit_multiple_of_9 : 
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ n % 9 = 0 ∧ ∀ m : ℕ, (m >= 100 ∧ m < 1000 ∧ m % 9 = 0) → n ≤ m :=
begin
  use 108,
  split,
  { exact nat.le_refl 108 },
  split,
  { exact nat.lt_of_lt_of_le (nat.succ_pos 9) (nat.succ_le_succ (nat.le_refl 99)) },
  split,
  { exact nat.mod_eq_zero_of_mk (nat.zero_of_succ_pos 12) },
  { intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    exact nat.le_of_eq ((nat.mod_eq_zero_of_dvd (nat.gcd_eq_gcd_ab (12) (8) (1)))),
  },
  sorry
end

end least_positive_three_digit_multiple_of_9_l310_310820


namespace cricket_team_matches_played_in_august_l310_310510

theorem cricket_team_matches_played_in_august
    (M : ℕ)
    (h1 : ∃ W : ℕ, W = 24 * M / 100)
    (h2 : ∃ W : ℕ, W + 70 = 52 * (M + 70) / 100) :
    M = 120 :=
sorry

end cricket_team_matches_played_in_august_l310_310510


namespace sector_radian_measure_l310_310089

theorem sector_radian_measure {r l : ℝ} 
  (h1 : 2 * r + l = 12) 
  (h2 : (1/2) * l * r = 8) : 
  (l / r = 1) ∨ (l / r = 4) :=
sorry

end sector_radian_measure_l310_310089


namespace arithmetic_sequence_a5_l310_310962

theorem arithmetic_sequence_a5
  (a : ℕ → ℤ) -- a is the arithmetic sequence function
  (S : ℕ → ℤ) -- S is the sum of the first n terms of the sequence
  (h1 : S 5 = 2 * S 4) -- Condition S_5 = 2S_4
  (h2 : a 2 + a 4 = 8) -- Condition a_2 + a_4 = 8
  (hS : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) -- Definition of S_n
  (ha : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) -- Definition of a_n
: a 5 = 10 := 
by
  -- proof
  sorry

end arithmetic_sequence_a5_l310_310962


namespace problem_1_problem_2_l310_310554

-- Problem 1 statement
theorem problem_1 (a x : ℝ) (m : ℝ) (h_pos_a : a > 0) (h_cond_a : a = 1/4) (h_cond_q : (1 : ℝ) / 2 < x ∧ x < 1) (h_cond_p : a < x ∧ x < 3 * a): 1 / 2 < x ∧ x < 3 / 4 :=
by sorry

-- Problem 2 statement
theorem problem_2 (a x : ℝ) (m : ℝ) (h_pos_a : a > 0) (h_neg_p : ¬(a < x ∧ x < 3 * a)) (h_neg_q : ¬((1 / (2 : ℝ))^(m - 1) < x ∧ x < 1)): 1 / 3 ≤ a ∧ a ≤ 1 / 2 :=
by sorry

end problem_1_problem_2_l310_310554


namespace money_distribution_l310_310628

theorem money_distribution (p q r : ℝ) 
  (h1 : p + q + r = 9000) 
  (h2 : r = (2/3) * (p + q)) : 
  r = 3600 := 
by 
  sorry

end money_distribution_l310_310628


namespace portions_of_milk_l310_310558

theorem portions_of_milk (liters_to_ml : ℕ) (total_liters : ℕ) (portion : ℕ) (total_volume_ml : ℕ) (num_portions : ℕ) :
  liters_to_ml = 1000 →
  total_liters = 2 →
  portion = 200 →
  total_volume_ml = total_liters * liters_to_ml →
  num_portions = total_volume_ml / portion →
  num_portions = 10 := by
  sorry

end portions_of_milk_l310_310558


namespace geometric_concepts_cases_l310_310073

theorem geometric_concepts_cases :
  (∃ x y, x = "rectangle" ∧ y = "rhombus") ∧ 
  (∃ x y z, x = "right_triangle" ∧ y = "isosceles_triangle" ∧ z = "acute_triangle") ∧ 
  (∃ x y z u, x = "parallelogram" ∧ y = "rectangle" ∧ z = "square" ∧ u = "acute_angled_rhombus") ∧ 
  (∃ x y z u t, x = "polygon" ∧ y = "triangle" ∧ z = "isosceles_triangle" ∧ u = "equilateral_triangle" ∧ t = "right_triangle") ∧ 
  (∃ x y z u, x = "right_triangle" ∧ y = "isosceles_triangle" ∧ z = "obtuse_triangle" ∧ u = "scalene_triangle") :=
by {
  sorry
}

end geometric_concepts_cases_l310_310073


namespace sum_binom_solutions_l310_310013

theorem sum_binom_solutions :
  (12 + 13 = 25) ∧ ∀ n : ℕ, (choose 25 n + choose 25 12 = choose 26 13) ↔ (n = 12 ∨ n = 13) :=
by
  sorry

end sum_binom_solutions_l310_310013


namespace cornelia_european_countries_l310_310390

def total_countries : Nat := 42
def south_american_countries : Nat := 10
def asian_countries : Nat := 6

def non_european_countries : Nat :=
  south_american_countries + 2 * asian_countries

def european_countries : Nat :=
  total_countries - non_european_countries

theorem cornelia_european_countries :
  european_countries = 20 := by
  sorry

end cornelia_european_countries_l310_310390


namespace min_max_value_is_zero_l310_310067

def max_at_x (x : ℝ) (y : ℝ) : ℝ := |x^2 - 2 * x * y|

theorem min_max_value_is_zero :
  ∃ y ∈ set.univ, min (set.univ) (λ y, real.sup (set.Icc 0 2) (λ x, max_at_x x y)) = 0 :=
sorry

end min_max_value_is_zero_l310_310067


namespace mutually_exclusive_B_C_l310_310687

-- Define the events A, B, C
def event_A (x y : ℕ → Bool) : Prop := ¬(x 1 = false ∨ x 2 = false)
def event_B (x y : ℕ → Bool) : Prop := x 1 = false ∧ x 2 = false
def event_C (x y : ℕ → Bool) : Prop := ¬(x 1 = false ∧ x 2 = false)

-- Prove that event B and event C are mutually exclusive
theorem mutually_exclusive_B_C (x y : ℕ → Bool) :
  (event_B x y ∧ event_C x y) ↔ false := sorry

end mutually_exclusive_B_C_l310_310687


namespace least_three_digit_multiple_of_9_eq_108_l310_310819

/--
What is the least positive three-digit multiple of 9?
-/
theorem least_three_digit_multiple_of_9_eq_108 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 9 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 9 = 0 → n ≤ m :=
  ∃ n : ℕ, n = 108 :=
begin
  sorry
end

end least_three_digit_multiple_of_9_eq_108_l310_310819


namespace area_of_rectangle_l310_310649

--- Define the problem's conditions
def square_area : ℕ := 36
def rectangle_width := (square_side : ℕ) (h : square_area = square_side * square_side) : ℕ := square_side
def rectangle_length := (width : ℕ) : ℕ := 3 * width

--- State the theorem using the defined conditions
theorem area_of_rectangle (square_side : ℕ) 
  (h1 : square_area = square_side * square_side)
  (width := rectangle_width square_side h1)
  (length := rectangle_length width) :
  width * length = 108 := by
    sorry

end area_of_rectangle_l310_310649


namespace positive_integers_ab_divides_asq_bsq_implies_a_eq_b_l310_310134

theorem positive_integers_ab_divides_asq_bsq_implies_a_eq_b
  (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hdiv : a * b ∣ a^2 + b^2) : a = b := by
  sorry

end positive_integers_ab_divides_asq_bsq_implies_a_eq_b_l310_310134


namespace volume_of_tetrahedron_eq_20_l310_310270

noncomputable def volume_tetrahedron (a b c : ℝ) : ℝ :=
  1 / 3 * a * b * c

theorem volume_of_tetrahedron_eq_20 {x y z : ℝ} (h1 : x^2 + y^2 = 25) (h2 : y^2 + z^2 = 41) (h3 : z^2 + x^2 = 34) :
  volume_tetrahedron 3 4 5 = 20 :=
by
  sorry

end volume_of_tetrahedron_eq_20_l310_310270


namespace converse_statement_l310_310938

theorem converse_statement (x : ℝ) :
  x^2 + 3 * x - 2 < 0 → x < 1 :=
sorry

end converse_statement_l310_310938


namespace work_completion_days_l310_310504

theorem work_completion_days (x : ℕ) (h_ratio : 5 * 18 = 3 * 30) : 30 = 30 :=
by {
    sorry
}

end work_completion_days_l310_310504


namespace inequality_negatives_l310_310084

theorem inequality_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : (b / a) < 1 :=
by
  sorry

end inequality_negatives_l310_310084


namespace no_real_solutions_l310_310559

theorem no_real_solutions :
  ∀ y : ℝ, ( (-2 * y + 7)^2 + 2 = -2 * |y| ) → false := by
  sorry

end no_real_solutions_l310_310559


namespace triangle_angles_21_equal_triangles_around_square_l310_310744

theorem triangle_angles_21_equal_triangles_around_square
    (theta alpha beta gamma : ℝ)
    (h1 : 4 * theta + 90 = 360)
    (h2 : alpha + beta + 90 = 180)
    (h3 : alpha + beta + gamma = 180)
    (h4 : gamma + 90 = 180)
    : theta = 67.5 ∧ alpha = 67.5 ∧ beta = 22.5 ∧ gamma = 90 :=
by
  sorry

end triangle_angles_21_equal_triangles_around_square_l310_310744


namespace sum_of_consecutive_even_numbers_l310_310351

theorem sum_of_consecutive_even_numbers (n : ℤ) 
  (h : n + 4 = 14) : n + (n + 2) + (n + 4) + (n + 6) = 52 :=
by
  sorry

end sum_of_consecutive_even_numbers_l310_310351


namespace eval_floor_sqrt_50_square_l310_310877

theorem eval_floor_sqrt_50_square:
    (int.floor (real.sqrt 50))^2 = 49 :=
by
  have h1 : real.sqrt 49 < real.sqrt 50 := by norm_num [real.sqrt]
  have h2 : real.sqrt 50 < real.sqrt 64 := by norm_num [real.sqrt]
  have floor_sqrt_50 : int.floor (real.sqrt 50) = 7 :=
    by linarith [h1, h2]
  rw [floor_sqrt_50]
  norm_num

end eval_floor_sqrt_50_square_l310_310877


namespace shaded_square_percentage_l310_310498

theorem shaded_square_percentage (total_squares shaded_squares : ℕ) (h_total: total_squares = 25) (h_shaded: shaded_squares = 13) : 
(shaded_squares * 100) / total_squares = 52 := 
by
  sorry

end shaded_square_percentage_l310_310498


namespace brenda_age_correct_l310_310382

open Nat

noncomputable def brenda_age_proof : Prop :=
  ∃ (A B J : ℚ), 
  (A = 4 * B) ∧ 
  (J = B + 8) ∧ 
  (A = J) ∧ 
  (B = 8 / 3)

theorem brenda_age_correct : brenda_age_proof := 
  sorry

end brenda_age_correct_l310_310382


namespace negation_proposition_l310_310979

theorem negation_proposition (x : ℝ) : ¬ (x ≥ 1 → x^2 - 4 * x + 2 ≥ -1) ↔ (x < 1 ∧ x^2 - 4 * x + 2 < -1) :=
by
  sorry

end negation_proposition_l310_310979


namespace ten_fact_minus_nine_fact_l310_310857

-- Definitions corresponding to the conditions
def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Condition for 9!
def nine_factorial : ℕ := 362880

-- 10! can be expressed in terms of 9!
noncomputable def ten_factorial : ℕ := 10 * nine_factorial

-- Proof statement we need to show
theorem ten_fact_minus_nine_fact : ten_factorial - nine_factorial = 3265920 :=
by
  unfold ten_factorial
  unfold nine_factorial
  sorry

end ten_fact_minus_nine_fact_l310_310857


namespace solve_equation_l310_310600

theorem solve_equation (x : ℝ) (hx : 0 ≤ x) : 2021 * x^10 - 2020 * x - 1 = 0 ↔ x = 1 := 
by 
  sorry

end solve_equation_l310_310600


namespace fill_time_without_leak_l310_310187

theorem fill_time_without_leak (F L : ℝ)
  (h1 : (F - L) * 12 = 1)
  (h2 : L * 24 = 1) :
  1 / F = 8 := 
sorry

end fill_time_without_leak_l310_310187


namespace valid_parametrizations_l310_310143

-- Define the line as a function
def line (x : ℝ) : ℝ := -2 * x + 7

-- Define vectors and their properties
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def on_line (v : Vector2D) : Prop :=
  v.y = line v.x

def direction_vector (v1 v2 : Vector2D) : Vector2D :=
  ⟨v2.x - v1.x, v2.y - v1.y⟩

def is_multiple (v1 v2 : Vector2D) : Prop :=
  ∃ k : ℝ, v2.x = k * v1.x ∧ v2.y = k * v1.y

-- Define the given parameterizations
def param_A (t : ℝ) : Vector2D := ⟨0 + t * 5, 7 + t * 10⟩
def param_B (t : ℝ) : Vector2D := ⟨2 + t * 1, 3 + t * -2⟩
def param_C (t : ℝ) : Vector2D := ⟨7 + t * 4, 0 + t * -8⟩
def param_D (t : ℝ) : Vector2D := ⟨-1 + t * 2, 9 + t * 4⟩
def param_E (t : ℝ) : Vector2D := ⟨3 + t * 2, 1 + t * 0⟩

-- Define the theorem
theorem valid_parametrizations :
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨0, 7⟩ (param_A t)) ∧ on_line (param_A t) → False) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨2, 3⟩ (param_B t)) ∧ on_line (param_B t)) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨7, 0⟩ (param_C t)) ∧ on_line (param_C t)) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨-1, 9⟩ (param_D t)) ∧ on_line (param_D t) → False) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨3, 1⟩ (param_E t)) ∧ on_line (param_E t) → False) :=
by
  sorry

end valid_parametrizations_l310_310143


namespace f_2000_equals_1499001_l310_310227

noncomputable def f (x : ℕ) : ℝ → ℝ := sorry

axiom f_initial : f 0 = 1

axiom f_recursive (x : ℕ) : f (x + 4) = f x + 3 * x + 4

theorem f_2000_equals_1499001 : f 2000 = 1499001 :=
by sorry

end f_2000_equals_1499001_l310_310227


namespace total_accepted_cartons_l310_310441

theorem total_accepted_cartons 
  (total_cartons : ℕ) 
  (customers : ℕ) 
  (damaged_cartons : ℕ)
  (h1 : total_cartons = 400)
  (h2 : customers = 4)
  (h3 : damaged_cartons = 60)
  : total_cartons / customers * (customers - (damaged_cartons / (total_cartons / customers))) = 160 := by
  sorry

end total_accepted_cartons_l310_310441


namespace amount_of_second_alloy_used_l310_310733

variable (x : ℝ)

-- Conditions
def chromium_in_first_alloy : ℝ := 0.10 * 15
def chromium_in_second_alloy (x : ℝ) : ℝ := 0.06 * x
def total_weight (x : ℝ) : ℝ := 15 + x
def chromium_in_third_alloy (x : ℝ) : ℝ := 0.072 * (15 + x)

-- Proof statement
theorem amount_of_second_alloy_used :
  1.5 + 0.06 * x = 0.072 * (15 + x) → x = 35 := by
  sorry

end amount_of_second_alloy_used_l310_310733


namespace yearly_water_consumption_correct_l310_310499

def monthly_water_consumption : ℝ := 182.88
def months_in_a_year : ℕ := 12
def yearly_water_consumption : ℝ := monthly_water_consumption * (months_in_a_year : ℝ)

theorem yearly_water_consumption_correct :
  yearly_water_consumption = 2194.56 :=
by
  sorry

end yearly_water_consumption_correct_l310_310499


namespace DVDs_per_season_l310_310758

theorem DVDs_per_season (total_DVDs : ℕ) (seasons : ℕ) (h1 : total_DVDs = 40) (h2 : seasons = 5) : total_DVDs / seasons = 8 :=
by
  sorry

end DVDs_per_season_l310_310758


namespace div_by_eleven_l310_310252

theorem div_by_eleven (n : ℤ) : 11 ∣ ((n + 11)^2 - n^2) :=
by
  sorry

end div_by_eleven_l310_310252


namespace mary_needs_to_add_6_25_more_cups_l310_310138

def total_flour_needed : ℚ := 8.5
def flour_already_added : ℚ := 2.25
def flour_to_add : ℚ := total_flour_needed - flour_already_added

theorem mary_needs_to_add_6_25_more_cups :
  flour_to_add = 6.25 :=
sorry

end mary_needs_to_add_6_25_more_cups_l310_310138


namespace magician_hat_probability_l310_310842

def total_arrangements : ℕ := Nat.choose 6 2
def favorable_arrangements : ℕ := Nat.choose 5 1
def probability_red_chips_drawn_first : ℚ := favorable_arrangements / total_arrangements

theorem magician_hat_probability :
  probability_red_chips_drawn_first = 1 / 3 :=
by
  sorry

end magician_hat_probability_l310_310842


namespace length_of_bridge_l310_310624

theorem length_of_bridge
  (walking_speed_km_hr : ℝ) (time_minutes : ℝ) (length_bridge : ℝ) 
  (h1 : walking_speed_km_hr = 5) 
  (h2 : time_minutes = 15) 
  (h3 : length_bridge = 1250) : 
  length_bridge = (walking_speed_km_hr * 1000 / 60) * time_minutes := 
by 
  sorry

end length_of_bridge_l310_310624


namespace probability_red_given_spade_or_king_l310_310565

def num_cards := 52
def num_spades := 13
def num_kings := 4
def num_red_kings := 2

def num_non_spade_kings := num_kings - 1
def num_spades_or_kings := num_spades + num_non_spade_kings

theorem probability_red_given_spade_or_king :
  (num_red_kings : ℚ) / num_spades_or_kings = 1 / 8 :=
sorry

end probability_red_given_spade_or_king_l310_310565


namespace max_sum_of_squares_eq_7_l310_310485

theorem max_sum_of_squares_eq_7 :
  ∃ (x y : ℤ), (x^2 + y^2 = 25 ∧ x + y = 7) ∧
  (∀ x' y' : ℤ, (x'^2 + y'^2 = 25 → x' + y' ≤ 7)) := by
sorry

end max_sum_of_squares_eq_7_l310_310485


namespace min_Box_value_l310_310418

/-- The conditions are given as:
  1. (ax + b)(bx + a) = 24x^2 + Box * x + 24
  2. a, b, Box are distinct integers
  The task is to find the minimum possible value of Box.
-/
theorem min_Box_value :
  ∃ (a b Box : ℤ), a ≠ b ∧ a ≠ Box ∧ b ≠ Box ∧ (∀ x : ℤ, (a * x + b) * (b * x + a) = 24 * x^2 + Box * x + 24) ∧ Box = 52 := sorry

end min_Box_value_l310_310418


namespace value_of_expression_l310_310714

theorem value_of_expression (x y : ℝ) (hy : y > 0) (h : x = 3 * y) :
  (x^y * y^x) / (y^y * x^x) = 3^(-2 * y) := by
  sorry

end value_of_expression_l310_310714


namespace quadratic_coefficients_sum_l310_310844

-- Definition of the quadratic function and the conditions
def quadraticFunction (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Conditions
def vertexCondition (a b c : ℝ) : Prop :=
  quadraticFunction a b c 2 = 3
  
def pointCondition (a b c : ℝ) : Prop :=
  quadraticFunction a b c 3 = 2

-- The theorem to prove
theorem quadratic_coefficients_sum (a b c : ℝ)
  (hv : vertexCondition a b c)
  (hp : pointCondition a b c):
  a + b + 2 * c = 2 :=
sorry

end quadratic_coefficients_sum_l310_310844


namespace rectangle_area_l310_310643

theorem rectangle_area (a w l : ℝ) (h_square_area : a = 36) 
    (h_rect_width : w * w = a) 
    (h_rect_length : l = 3 * w) : w * l = 108 := 
sorry

end rectangle_area_l310_310643


namespace petya_purchase_cost_l310_310327

theorem petya_purchase_cost (x : ℝ) 
  (h1 : ∃ shirt_cost : ℝ, x + shirt_cost = 2 * x)
  (h2 : ∃ boots_cost : ℝ, x + boots_cost = 5 * x)
  (h3 : ∃ shin_guards_cost : ℝ, x + shin_guards_cost = 3 * x) :
  ∃ total_cost : ℝ, total_cost = 8 * x :=
by 
  sorry

end petya_purchase_cost_l310_310327


namespace parallel_line_slope_l310_310823

theorem parallel_line_slope (x y : ℝ) : (∃ (c : ℝ), 3 * x - 6 * y = c) → (1 / 2) = 1 / 2 :=
by sorry

end parallel_line_slope_l310_310823


namespace grade_more_problems_l310_310043

theorem grade_more_problems (worksheets_total problems_per_worksheet worksheets_graded: ℕ)
  (h1 : worksheets_total = 9)
  (h2 : problems_per_worksheet = 4)
  (h3 : worksheets_graded = 5):
  (worksheets_total - worksheets_graded) * problems_per_worksheet = 16 :=
by
  sorry

end grade_more_problems_l310_310043


namespace mason_courses_not_finished_l310_310140

-- Each necessary condition is listed as a definition.
def coursesPerWall := 6
def bricksPerCourse := 10
def numOfWalls := 4
def totalBricksUsed := 220

-- Creating an entity to store the problem and prove it.
theorem mason_courses_not_finished : 
  (numOfWalls * coursesPerWall * bricksPerCourse - totalBricksUsed) / bricksPerCourse = 2 := 
by
  sorry

end mason_courses_not_finished_l310_310140


namespace divisible_sum_l310_310234

theorem divisible_sum (k : ℕ) (n : ℕ) (h : n = 2^(k-1)) : 
  ∀ (S : Finset ℕ), S.card = 2*n - 1 → ∃ T ⊆ S, T.card = n ∧ T.sum id % n = 0 :=
by
  sorry

end divisible_sum_l310_310234


namespace modulus_z_eq_sqrt_10_l310_310721

noncomputable def z : ℂ := (1 + 7 * Complex.I) / (2 + Complex.I)

theorem modulus_z_eq_sqrt_10 : Complex.abs z = Real.sqrt 10 := sorry

end modulus_z_eq_sqrt_10_l310_310721


namespace sufficient_but_not_necessary_condition_l310_310445

variables (a b : ℝ)

def p : Prop := a > b ∧ b > 1
def q : Prop := a - b < a^2 - b^2

theorem sufficient_but_not_necessary_condition (h : p a b) : q a b :=
  sorry

end sufficient_but_not_necessary_condition_l310_310445


namespace original_number_l310_310372

theorem original_number (x y : ℝ) (h1 : 10 * x + 22 * y = 780) (h2 : y = 34) : x + y = 37.2 :=
sorry

end original_number_l310_310372


namespace rectangle_area_l310_310653

theorem rectangle_area (square_area : ℝ) (rectangle_length_ratio : ℝ) (square_area_eq : square_area = 36)
  (rectangle_length_ratio_eq : rectangle_length_ratio = 3) :
  ∃ (rectangle_area : ℝ), rectangle_area = 108 :=
by
  -- Extract the side length of the square from its area
  let side_length := real.sqrt square_area
  have side_length_eq : side_length = 6, from calc
    side_length = real.sqrt 36 : by rw [square_area_eq]
    ... = 6 : real.sqrt_eq 6 (by norm_num)
  -- Calculate the rectangle's width
  let width := side_length
  -- Calculate the rectangle's length
  let length := rectangle_length_ratio * width
  -- Calculate the area of the rectangle
  let area := width * length
  -- Prove the area is 108
  use area
  calc area
    = 6 * (3 * 6) : by rw [side_length_eq, rectangle_length_ratio_eq]
    ... = 108 : by norm_num

end rectangle_area_l310_310653


namespace problem_l310_310697

theorem problem (x : ℝ) (h : x + 1/x = 10) :
  (x^2 + 1/x^2 = 98) ∧ (x^3 + 1/x^3 = 970) :=
by
  sorry

end problem_l310_310697


namespace necessary_condition_for_inequality_l310_310937

theorem necessary_condition_for_inequality (m : ℝ) :
  (∀ x : ℝ, (x^2 - 3 * x + 2 < 0) → (x > m)) ∧ (∃ x : ℝ, (x > m) ∧ ¬(x^2 - 3 * x + 2 < 0)) → m ≤ 1 := 
by
  sorry

end necessary_condition_for_inequality_l310_310937


namespace meeting_probability_l310_310757

theorem meeting_probability :
  let steps := 8
  let total_paths := 2^steps
  let intersection_count := ∑ i in (Finset.range (steps + 1)), (Nat.choose steps i) ^ 2
  (intersection_count / total_paths ^ 2) = (6435 : ℚ) / 65536 :=
by
  sorry

end meeting_probability_l310_310757


namespace tile_ratio_l310_310862

theorem tile_ratio (original_black_tiles : ℕ) (original_white_tiles : ℕ) (original_width : ℕ) (original_height : ℕ) (border_width : ℕ) (border_height : ℕ) :
  original_black_tiles = 10 ∧ original_white_tiles = 22 ∧ original_width = 8 ∧ original_height = 4 ∧ border_width = 2 ∧ border_height = 2 →
  (original_black_tiles + ( (original_width + 2 * border_width) * (original_height + 2 * border_height) - original_width * original_height ) ) / original_white_tiles = 19 / 11 :=
by
  -- sorry to skip the proof
  sorry

end tile_ratio_l310_310862


namespace sufficient_not_necessary_condition_l310_310131

theorem sufficient_not_necessary_condition (x y : ℝ) : 
  (x - y) * x^4 < 0 → x < y ∧ ¬(x < y → (x - y) * x^4 < 0) := 
sorry

end sufficient_not_necessary_condition_l310_310131


namespace min_y_value_l310_310432

noncomputable def y (x : ℝ) : ℝ :=
  (x - 6.5)^2 + (x - 5.9)^2 + (x - 6.0)^2 + (x - 6.7)^2 + (x - 4.5)^2

theorem min_y_value : 
  ∃ x : ℝ, (∀ ε > 0, ∃ δ > 0, ∀ x' : ℝ, abs (x' - 5.92) < δ → abs (y x' - y 5.92) < ε) :=
sorry

end min_y_value_l310_310432


namespace undefined_values_l310_310903

theorem undefined_values (a : ℝ) : a = -3 ∨ a = 3 ↔ (a^2 - 9 = 0) := sorry

end undefined_values_l310_310903


namespace geom_seq_prop_l310_310398

variable (b : ℕ → ℝ) (r : ℝ) (s t : ℕ)
variable (h : s ≠ t)
variable (h1 : s > 0) (h2 : t > 0)
variable (h3 : b 1 = 1)
variable (h4 : ∀ n, b (n + 1) = b n * r)

theorem geom_seq_prop : s ≠ t → s > 0 → t > 0 → b 1 = 1 → (∀ n, b (n + 1) = b n * r) → (b t)^(s - 1) / (b s)^(t - 1) = 1 :=
by
  intros h h1 h2 h3 h4
  sorry

end geom_seq_prop_l310_310398


namespace equilateral_sector_area_l310_310720

noncomputable def area_of_equilateral_sector (r : ℝ) : ℝ :=
  if h : r = r then (1/2) * r^2 * 1 else 0

theorem equilateral_sector_area (r : ℝ) : r = 2 → area_of_equilateral_sector r = 2 :=
by
  intros hr
  rw [hr]
  unfold area_of_equilateral_sector
  split_ifs
  · norm_num
  · contradiction

end equilateral_sector_area_l310_310720


namespace max_operations_l310_310612

def arithmetic_mean (a b : ℕ) := (a + b) / 2

theorem max_operations (b : ℕ) (hb : b < 2002) (heven : (2002 + b) % 2 = 0) :
  ∃ n, n = 10 ∧ (2002 - b) / 2^n = 1 :=
by
  sorry

end max_operations_l310_310612


namespace maryann_time_spent_calling_clients_l310_310753

theorem maryann_time_spent_calling_clients (a c : ℕ) 
  (h1 : a + c = 560) 
  (h2 : a = 7 * c) : c = 70 := 
by 
  sorry

end maryann_time_spent_calling_clients_l310_310753


namespace domain_of_sqrt_2_cos_x_minus_1_l310_310395

theorem domain_of_sqrt_2_cos_x_minus_1 :
  {x : ℝ | ∃ k : ℤ, - (Real.pi / 3) + 2 * k * Real.pi ≤ x ∧ x ≤ (Real.pi / 3) + 2 * k * Real.pi } =
  {x : ℝ | 2 * Real.cos x - 1 ≥ 0 } :=
sorry

end domain_of_sqrt_2_cos_x_minus_1_l310_310395


namespace product_of_integers_l310_310004

theorem product_of_integers :
  ∃ (a b c d e : ℤ), 
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} = {2, 6, 10, 10, 12, 14, 16, 18, 20, 24}) ∧
  (a * b * c * d * e = -3003) :=
begin
  sorry
end

end product_of_integers_l310_310004


namespace max_ab_value_l310_310127

theorem max_ab_value {a b : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : 6 * a + 8 * b = 72) : ab = 27 :=
by {
  sorry
}

end max_ab_value_l310_310127


namespace solution_set_of_inequality_l310_310481

theorem solution_set_of_inequality (x : ℝ) (h : 2 * x + 3 ≤ 1) : x ≤ -1 :=
sorry

end solution_set_of_inequality_l310_310481


namespace monotonic_increasing_interval_l310_310346

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (x^2 - 4)

theorem monotonic_increasing_interval :
  ∀ x : ℝ, 2 < x → (f x < f (x + 1)) :=
by
  intros x h
  sorry

end monotonic_increasing_interval_l310_310346


namespace fraction_sum_l310_310070

theorem fraction_sum : (1 / 3 : ℚ) + (2 / 7) + (3 / 8) = 167 / 168 := by
  sorry

end fraction_sum_l310_310070


namespace bicycle_final_price_l310_310995

theorem bicycle_final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (h1 : original_price = 200) (h2 : discount1 = 0.4) (h3 : discount2 = 0.2) :
  (original_price * (1 - discount1) * (1 - discount2)) = 96 :=
by
  -- sorry proof here
  sorry

end bicycle_final_price_l310_310995


namespace smallest_two_digit_palindrome_l310_310173

def is_palindrome {α : Type} [DecidableEq α] (xs : List α) : Prop :=
  xs = xs.reverse

-- A number is a two-digit palindrome in base 5 if it has the form ab5 where a and b are digits 0-4
def two_digit_palindrome_base5 (n : ℕ) : Prop :=
  ∃ a b : ℕ, a < 5 ∧ b < 5 ∧ a ≠ 0 ∧ n = a * 5 + b ∧ is_palindrome [a, b]

-- A number is a three-digit palindrome in base 2 if it has the form abc2 where a = c and b can vary (0-1)
def three_digit_palindrome_base2 (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a < 2 ∧ b < 2 ∧ c < 2 ∧ a = c ∧ n = a * 4 + b * 2 + c ∧ is_palindrome [a, b, c]

theorem smallest_two_digit_palindrome :
  ∃ n, two_digit_palindrome_base5 n ∧ three_digit_palindrome_base2 n ∧
       (∀ m, two_digit_palindrome_base5 m ∧ three_digit_palindrome_base2 m → n ≤ m) :=
sorry

end smallest_two_digit_palindrome_l310_310173


namespace sqrt_floor_squared_eq_49_l310_310880

theorem sqrt_floor_squared_eq_49 : (⌊real.sqrt 50⌋)^2 = 49 :=
by sorry

end sqrt_floor_squared_eq_49_l310_310880


namespace cost_of_article_is_308_l310_310255

theorem cost_of_article_is_308 
  (C G : ℝ) 
  (h1 : 348 = C + G)
  (h2 : 350 = C + G + 0.05 * G) : 
  C = 308 :=
by
  sorry

end cost_of_article_is_308_l310_310255


namespace find_extrema_of_S_l310_310689

theorem find_extrema_of_S (x y z : ℚ) (h1 : 3 * x + 2 * y + z = 5) (h2 : x + y - z = 2) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  2 ≤ 2 * x + y - z ∧ 2 * x + y - z ≤ 3 :=
by
  sorry

end find_extrema_of_S_l310_310689


namespace floor_square_of_sqrt_50_eq_49_l310_310883

theorem floor_square_of_sqrt_50_eq_49 : (Int.floor (Real.sqrt 50))^2 = 49 := 
by
  sorry

end floor_square_of_sqrt_50_eq_49_l310_310883


namespace candy_probability_difference_l310_310003

theorem candy_probability_difference :
  let total := 2004
  let total_ways := Nat.choose total 2
  let different_ways := 2002 * 1002 / 2
  let same_ways := 1002 * 1001 / 2 + 1002 * 1001 / 2
  let q := (different_ways : ℚ) / total_ways
  let p := (same_ways : ℚ) / total_ways
  q - p = 1 / 2003 :=
by sorry

end candy_probability_difference_l310_310003


namespace smiths_bakery_multiple_l310_310766

theorem smiths_bakery_multiple (x : ℤ) (mcgee_pies : ℤ) (smith_pies : ℤ) 
  (h1 : smith_pies = x * mcgee_pies + 6)
  (h2 : mcgee_pies = 16)
  (h3 : smith_pies = 70) : x = 4 :=
by
  sorry

end smiths_bakery_multiple_l310_310766


namespace ten_numbers_property_l310_310584

theorem ten_numbers_property (x : ℕ → ℝ) (h : ∀ i : ℕ, 1 ≤ i → i ≤ 9 → x i + 2 * x (i + 1) = 1) : 
  x 1 + 512 * x 10 = 171 :=
by
  sorry

end ten_numbers_property_l310_310584


namespace power_calculation_l310_310523

theorem power_calculation : (128 : ℝ) ^ (4/7) = 16 :=
by {
  have factorization : (128 : ℝ) = 2 ^ 7 := by {
    norm_num,
  },
  rw factorization,
  have power_rule : (2 ^ 7 : ℝ) ^ (4/7) = 2 ^ 4 := by {
    norm_num,
  },
  rw power_rule,
  norm_num,
  sorry
}

end power_calculation_l310_310523


namespace length_CF_is_7_l310_310954

noncomputable def CF_length
  (ABCD_rectangle : Prop)
  (triangle_ABE_right : Prop)
  (triangle_CDF_right : Prop)
  (area_triangle_ABE : ℝ)
  (length_AE length_DF : ℝ)
  (h1 : ABCD_rectangle)
  (h2 : triangle_ABE_right)
  (h3 : triangle_CDF_right)
  (h4 : area_triangle_ABE = 150)
  (h5 : length_AE = 15)
  (h6 : length_DF = 24) :
  ℝ :=
7

theorem length_CF_is_7
  (ABCD_rectangle : Prop)
  (triangle_ABE_right : Prop)
  (triangle_CDF_right : Prop)
  (area_triangle_ABE : ℝ)
  (length_AE length_DF : ℝ)
  (h1 : ABCD_rectangle)
  (h2 : triangle_ABE_right)
  (h3 : triangle_CDF_right)
  (h4 : area_triangle_ABE = 150)
  (h5 : length_AE = 15)
  (h6 : length_DF = 24) :
  CF_length ABCD_rectangle triangle_ABE_right triangle_CDF_right area_triangle_ABE length_AE length_DF h1 h2 h3 h4 h5 h6 = 7 :=
by
  sorry

end length_CF_is_7_l310_310954


namespace cat_food_weight_l310_310458

theorem cat_food_weight (x : ℝ) :
  let bags_of_cat_food := 2
  let bags_of_dog_food := 2
  let ounces_per_pound := 16
  let total_ounces_of_pet_food := 256
  let dog_food_extra_weight := 2
  (ounces_per_pound * (bags_of_cat_food * x + bags_of_dog_food * (x + dog_food_extra_weight))) = total_ounces_of_pet_food
  → x = 3 :=
by
  sorry

end cat_food_weight_l310_310458


namespace trader_profit_l310_310197

noncomputable def original_price (P : ℝ) : ℝ := P
noncomputable def purchase_price (P : ℝ) : ℝ := 0.8 * P
noncomputable def depreciation1 (P : ℝ) : ℝ := 0.04 * P
noncomputable def depreciation2 (P : ℝ) : ℝ := 0.038 * P
noncomputable def value_after_depreciation (P : ℝ) : ℝ := 0.722 * P
noncomputable def taxes (P : ℝ) : ℝ := 0.024 * P
noncomputable def insurance (P : ℝ) : ℝ := 0.032 * P
noncomputable def maintenance (P : ℝ) : ℝ := 0.01 * P
noncomputable def total_cost (P : ℝ) : ℝ := value_after_depreciation P + taxes P + insurance P + maintenance P
noncomputable def selling_price (P : ℝ) : ℝ := 1.70 * total_cost P
noncomputable def profit (P : ℝ) : ℝ := selling_price P - original_price P
noncomputable def profit_percent (P : ℝ) : ℝ := (profit P / original_price P) * 100

theorem trader_profit (P : ℝ) : profit_percent P = 33.96 :=
  by
    sorry

end trader_profit_l310_310197


namespace single_digit_pairs_l310_310059

theorem single_digit_pairs:
  ∃ x y: ℕ, x ≠ 1 ∧ x ≠ 9 ∧ y ≠ 1 ∧ y ≠ 9 ∧ x < 10 ∧ y < 10 ∧ 
  (x * y < 100 ∧ ((x * y) % 10 + (x * y) / 10 == x ∨ (x * y) % 10 + (x * y) / 10 == y))
  → (x, y) ∈ [(3, 4), (3, 7), (6, 4), (6, 7)] :=
by
  sorry

end single_digit_pairs_l310_310059


namespace min_value_of_m_l310_310928

theorem min_value_of_m (m : ℝ) : (∀ x : ℝ, 0 < x → x ≠ ⌊x⌋ → mx < Real.log x) ↔ m = (1 / 2) * Real.log 2 :=
by
  sorry

end min_value_of_m_l310_310928


namespace table_covered_area_l310_310910

-- Definitions based on conditions
def length := 12
def width := 1
def number_of_strips := 4
def overlapping_strips := 3

-- Calculating the area of one strip
def area_of_one_strip := length * width

-- Calculating total area assuming no overlaps
def total_area_no_overlap := number_of_strips * area_of_one_strip

-- Calculating the total overlap area
def overlap_area := overlapping_strips * (width * width)

-- Final area after subtracting overlaps
def final_covered_area := total_area_no_overlap - overlap_area

-- Theorem stating the proof problem
theorem table_covered_area : final_covered_area = 45 :=
by
  sorry

end table_covered_area_l310_310910


namespace total_cost_is_eight_x_l310_310305

-- Definitions of cost variables based on conditions
variable (x : ℝ) -- Cost of shorts

-- Cost conditions
variable (shirt_cost : ℝ) (boot_cost : ℝ) (shin_guard_cost : ℝ)
variable (c1 : x + shirt_cost = 2 * x)
variable (c2 : x + boot_cost = 5 * x)
variable (c3 : x + shin_guard_cost = 3 * x)

-- To prove that the total cost is 8 times the cost of shorts
theorem total_cost_is_eight_x
  (c1 : x + shirt_cost = 2 * x)
  (c2 : x + boot_cost = 5 * x)
  (c3 : x + shin_guard_cost = 3 * x) :
  x + shirt_cost + boot_cost + shin_guard_cost = 8 * x := 
by
  sorry

end total_cost_is_eight_x_l310_310305


namespace solution_set_of_inequality_l310_310164

theorem solution_set_of_inequality :
  {x : ℝ | 9 * x^2 + 6 * x + 1 ≤ 0} = {-1 / 3} :=
by {
  sorry -- Proof goes here
}

end solution_set_of_inequality_l310_310164


namespace lollipop_count_l310_310567

theorem lollipop_count (total_cost one_lollipop_cost : ℚ) (h1 : total_cost = 90) (h2 : one_lollipop_cost = 0.75) : total_cost / one_lollipop_cost = 120 :=
by
  sorry

end lollipop_count_l310_310567


namespace set_intersection_l310_310915
noncomputable def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 2 }
noncomputable def B : Set ℝ := {x : ℝ | x ≥ 1 }

theorem set_intersection (x : ℝ) : x ∈ A ∩ B ↔ x ∈ A := sorry

end set_intersection_l310_310915


namespace correct_quotient_is_243_l310_310949

-- Define the given conditions
def mistaken_divisor : ℕ := 121
def mistaken_quotient : ℕ := 432
def correct_divisor : ℕ := 215
def remainder : ℕ := 0

-- Calculate the dividend based on mistaken values
def dividend : ℕ := mistaken_divisor * mistaken_quotient + remainder

-- State the theorem for the correct quotient
theorem correct_quotient_is_243
  (h_dividend : dividend = mistaken_divisor * mistaken_quotient + remainder)
  (h_divisible : dividend % correct_divisor = remainder) :
  dividend / correct_divisor = 243 :=
sorry

end correct_quotient_is_243_l310_310949


namespace angle_C_in_triangle_l310_310272

open Real

noncomputable def determine_angle_C (A B C: ℝ) (AB BC: ℝ) : Prop :=
  A = (3 * π) / 4 ∧ BC = sqrt 2 * AB → C = π / 6

-- Define the theorem to state the problem
theorem angle_C_in_triangle (A B C : ℝ) (AB BC : ℝ) :
  determine_angle_C A B C AB BC := 
by
  -- Step to indicate where the proof would be
  sorry

end angle_C_in_triangle_l310_310272


namespace min_value_f_l310_310345

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 1/2 * Real.cos (2 * x) - 1

theorem min_value_f : ∃ x : ℝ, f x = -5/2 := sorry

end min_value_f_l310_310345


namespace speed_conversion_l310_310839

theorem speed_conversion (speed_kmh : ℝ) (conversion_factor : ℝ) :
  speed_kmh = 1.3 → conversion_factor = (1000 / 3600) → speed_kmh * conversion_factor = 0.3611 :=
by
  intros h_speed h_factor
  rw [h_speed, h_factor]
  norm_num
  sorry

end speed_conversion_l310_310839


namespace sum_of_natural_numbers_eq_4005_l310_310666

theorem sum_of_natural_numbers_eq_4005 :
  ∃ n : ℕ, (n * (n + 1)) / 2 = 4005 ∧ n = 89 :=
by
  sorry

end sum_of_natural_numbers_eq_4005_l310_310666


namespace total_cost_is_eight_times_short_cost_l310_310294

variables (x : ℝ)
def cost_shorts := x
def cost_shirt := x
def cost_boots := 4 * x
def cost_shin_guards := 2 * x

theorem total_cost_is_eight_times_short_cost
    (cond_shirt : cost_shorts x + cost_shirt x = 2*x)
    (cond_boots : cost_shorts x + cost_boots x = 5*x)
    (cond_shin_guards : cost_shorts x + cost_shin_guards x = 3*x) :
    cost_shorts x + cost_shirt x + cost_boots x + cost_shin_guards x = 8*x :=
by
  sorry

end total_cost_is_eight_times_short_cost_l310_310294


namespace quadratic_has_exactly_one_solution_l310_310532

theorem quadratic_has_exactly_one_solution (k : ℚ) :
  (3 * x^2 - 8 * x + k = 0) → ((-8)^2 - 4 * 3 * k = 0) → k = 16 / 3 :=
by
  sorry

end quadratic_has_exactly_one_solution_l310_310532


namespace point_B_coordinates_l310_310660

def move_up (x y : Int) (units : Int) : Int := y + units
def move_left (x y : Int) (units : Int) : Int := x - units

theorem point_B_coordinates :
  let A : Int × Int := (1, -1)
  let B : Int × Int := (move_left A.1 A.2 3, move_up A.1 A.2 2)
  B = (-2, 1) := 
by
  -- This is where the proof would go, but we omit it with "sorry"
  sorry

end point_B_coordinates_l310_310660


namespace polynomial_ascending_l310_310045

theorem polynomial_ascending (x : ℝ) :
  (x^2 - 2 - 5*x^4 + 3*x^3) = (-2 + x^2 + 3*x^3 - 5*x^4) :=
by sorry

end polynomial_ascending_l310_310045


namespace geometric_sequence_smallest_n_l310_310190

def geom_seq (n : ℕ) (r : ℝ) (b₁ : ℝ) : ℝ := 
  b₁ * r^(n-1)

theorem geometric_sequence_smallest_n 
  (b₁ b₂ b₃ : ℝ) (r : ℝ)
  (h₁ : b₁ = 2)
  (h₂ : b₂ = 6)
  (h₃ : b₃ = 18)
  (h_seq : ∀ n, bₙ = geom_seq n r b₁) :
  ∃ n, n = 5 ∧ geom_seq n r 2 = 324 :=
by
  sorry

end geometric_sequence_smallest_n_l310_310190


namespace floor_problem_2020_l310_310899

-- Define the problem statement
theorem floor_problem_2020:
  2020 ^ 2021 - (Int.floor ((2020 ^ 2021 : ℝ) / 2021) * 2021) = 2020 :=
sorry

end floor_problem_2020_l310_310899


namespace distance_between_foci_of_hyperbola_l310_310679

theorem distance_between_foci_of_hyperbola :
  ∀ x y : ℝ, (x^2 - 8 * x - 16 * y^2 - 16 * y = 48) → (∃ c : ℝ, 2 * c = 2 * Real.sqrt 63.75) :=
by
  sorry

end distance_between_foci_of_hyperbola_l310_310679


namespace min_liars_needed_l310_310798

namespace Problem

/-
  We introduce our problem using Lean definitions and statements.
-/

def students_sitting_at_table : ℕ := 29

structure Student :=
(is_liar : Bool)

def student_truth_statement (s: ℕ) (next1: Student) (next2: Student) : Bool :=
  if s then (next1.is_liar && !next2.is_liar) || (!next1.is_liar && next2.is_liar)
  else next1.is_liar && next2.is_liar

def minimum_liars_class (sitting_students: ℕ) : ℕ :=
  sitting_students / 3 + (if sitting_students % 3 = 0 then 0 else 1)

theorem min_liars_needed (students : List Student) (H_total : students.length = students_sitting_at_table)
  (H_truth_teller_claim : ∀ (i : ℕ), i < students.length → student_truth_statement i (students.get ⟨(i+1) % students.length, sorry⟩) (students.get ⟨(i+2) % students.length, sorry⟩)) :
  (students.count (λ s => s.is_liar)) ≥ minimum_liars_class students_sitting_at_table :=
sorry

end Problem

end min_liars_needed_l310_310798


namespace volume_of_red_tetrahedron_l310_310031

def volume_of_cube (side_length : ℕ) : ℕ :=
  side_length^3

def volume_of_tetrahedron (base_area : ℕ) (height : ℕ) : ℚ :=
  (1/3 : ℚ) * base_area * height

def smaller_tetrahedron_volume (side_length : ℕ) : ℚ :=
  volume_of_tetrahedron ((1/2 : ℚ) * side_length^2) side_length

theorem volume_of_red_tetrahedron :
  let cube_side_length := 8 in
  let cube_volume := volume_of_cube cube_side_length in
  let smaller_tetrahedrons_volume := 4 * smaller_tetrahedron_volume cube_side_length in
  cube_volume - smaller_tetrahedrons_volume = 512 / 3 :=
by
  sorry

end volume_of_red_tetrahedron_l310_310031


namespace sum_of_valid_n_l310_310011

theorem sum_of_valid_n :
  ∃ n : ℤ, (n = 13) ∧ (∑ n in {k : ℤ | (nat.choose 25 k + nat.choose 25 12 = nat.choose 26 13)}, n = 13) :=
by
  sorry

end sum_of_valid_n_l310_310011


namespace larger_number_ratio_l310_310168

theorem larger_number_ratio (x : ℕ) (a b : ℕ) (h1 : a = 3 * x) (h2 : b = 8 * x) 
(h3 : (a - 24) * 9 = (b - 24) * 4) : b = 192 :=
sorry

end larger_number_ratio_l310_310168


namespace height_of_triangle_l310_310948

noncomputable def height_to_side_BC (a b c : ℝ) (angle_B : ℝ) : ℝ :=
  let S := (1 / 2) * a * c * real.sin angle_B in (2 * S) / a

theorem height_of_triangle (a b : ℝ) (angle_B : ℝ) (h : ℝ) (h_val : h = 3 * real.sqrt 3 / 2) : 
  a = 2 ∧ b = real.sqrt 7 ∧ angle_B = real.pi / 3 → height_to_side_BC a b 3 angle_B = h :=
by
  -- Add proof steps here
  sorry

end height_of_triangle_l310_310948


namespace perimeter_ADEF_is_56_l310_310947

noncomputable def ADEF_perimeter (A B C D E F : Point) (AB AC BC : ℝ) (hA : A = (0, 0)) (hB : B = (a, 0)) (hC : C = (b, 0)) 
  (h_AB : AB = 28) (h_AC : AC = 28) (h_BC : BC = 20) 
  (hD_on_AB : D ∈ line_through A B) (hE_on_BC : E ∈ line_through B C) (hF_on_AC : F ∈ line_through A C) 
  (hDE_parallel_AC : parallel line_through D E (line_through A C)) (hEF_parallel_AB : parallel line_through E F (line_through A B)) : ℝ :=
  56

-- Main theorem statement
theorem perimeter_ADEF_is_56 (A B C D E F : Point) (AB AC BC : ℝ) 
  (hA : A = (0, 0)) (hB : B = (a, 0)) (hC : C = (b, 0)) 
  (h_AB : AB = 28) (h_AC : AC = 28) (h_BC : 20)
  (hD_on_AB : D ∈ line_through A B) (hE_on_BC : E ∈ line_through B C) (hF_on_AC : F ∈ line_through A C) 
  (hDE_parallel_AC : parallel (line_through D E) (line_through A C)) (hEF_parallel_AB : parallel (line_through E F) (line_through A B)) :
  ADEF_perimeter A B C D E F AB AC BC hA hB hC h_AB h_AC h_BC hD_on_AB hE_on_BC hF_on_AC hDE_parallel_AC hEF_parallel_AB = 56 := 
  sorry

end perimeter_ADEF_is_56_l310_310947


namespace Chloe_wins_l310_310206

theorem Chloe_wins (C M : ℕ) (h_ratio : 8 * M = 3 * C) (h_Max : M = 9) : C = 24 :=
by {
    sorry
}

end Chloe_wins_l310_310206


namespace total_cost_is_eight_times_short_cost_l310_310293

variables (x : ℝ)
def cost_shorts := x
def cost_shirt := x
def cost_boots := 4 * x
def cost_shin_guards := 2 * x

theorem total_cost_is_eight_times_short_cost
    (cond_shirt : cost_shorts x + cost_shirt x = 2*x)
    (cond_boots : cost_shorts x + cost_boots x = 5*x)
    (cond_shin_guards : cost_shorts x + cost_shin_guards x = 3*x) :
    cost_shorts x + cost_shirt x + cost_boots x + cost_shin_guards x = 8*x :=
by
  sorry

end total_cost_is_eight_times_short_cost_l310_310293


namespace combined_value_of_a_and_b_l310_310110

theorem combined_value_of_a_and_b :
  (∃ a b : ℝ,
    0.005 * a = 95 / 100 ∧
    b = 3 * a - 50 ∧
    a + b = 710) :=
sorry

end combined_value_of_a_and_b_l310_310110


namespace trips_and_weights_l310_310385

theorem trips_and_weights (x : ℕ) (w : ℕ) (trips_Bill Jean_total limit_total: ℕ)
  (h1 : x + (x + 6) = 40)
  (h2 : trips_Bill = x)
  (h3 : Jean_total = x + 6)
  (h4 : w = 7850)
  (h5 : limit_total = 8000)
  : 
  trips_Bill = 17 ∧ 
  Jean_total = 23 ∧ 
  (w : ℝ) / 40 = 196.25 := 
by 
  sorry

end trips_and_weights_l310_310385


namespace rectangle_area_l310_310644

theorem rectangle_area (a w l : ℝ) (h_square_area : a = 36) 
    (h_rect_width : w * w = a) 
    (h_rect_length : l = 3 * w) : w * l = 108 := 
sorry

end rectangle_area_l310_310644


namespace two_digit_number_l310_310847

theorem two_digit_number (x y : ℕ) (h1 : x + y = 11) (h2 : 10 * y + x = 10 * x + y + 63) : 10 * x + y = 29 := 
by 
  sorry

end two_digit_number_l310_310847


namespace sale_in_fifth_month_l310_310035

theorem sale_in_fifth_month
  (s1 s2 s3 s4 s6 : ℕ)
  (avg : ℕ)
  (h1 : s1 = 5435)
  (h2 : s2 = 5927)
  (h3 : s3 = 5855)
  (h4 : s4 = 6230)
  (h6 : s6 = 3991)
  (hav : avg = 5500) :
  ∃ s5 : ℕ, s1 + s2 + s3 + s4 + s5 + s6 = avg * 6 ∧ s5 = 5562 := 
by
  sorry

end sale_in_fifth_month_l310_310035


namespace proof_problem_l310_310695

noncomputable def a {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def b {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def c {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def d {α : Type*} [LinearOrderedField α] : α := sorry

theorem proof_problem (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
(hprod : a * b * c * d = 1) : 
a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
sorry

end proof_problem_l310_310695


namespace coach_A_spent_less_l310_310166

-- Definitions of costs and discounts for coaches purchases
def total_cost_before_discount_A : ℝ := 10 * 29 + 5 * 15
def total_cost_before_discount_B : ℝ := 14 * 2.50 + 1 * 18 + 4 * 25 + 1 * 72
def total_cost_before_discount_C : ℝ := 8 * 32 + 12 * 12

def discount_A : ℝ := 0.05 * total_cost_before_discount_A
def discount_B : ℝ := 0.10 * total_cost_before_discount_B
def discount_C : ℝ := 0.07 * total_cost_before_discount_C

def total_cost_after_discount_A : ℝ := total_cost_before_discount_A - discount_A
def total_cost_after_discount_B : ℝ := total_cost_before_discount_B - discount_B
def total_cost_after_discount_C : ℝ := total_cost_before_discount_C - discount_C

def combined_cost_B_C : ℝ := total_cost_after_discount_B + total_cost_after_discount_C
def difference_A_BC : ℝ := total_cost_after_discount_A - combined_cost_B_C

theorem coach_A_spent_less : difference_A_BC = -227.75 := by
  sorry

end coach_A_spent_less_l310_310166


namespace function_intersects_y_axis_at_0_neg4_l310_310596

theorem function_intersects_y_axis_at_0_neg4 :
  (∃ x y : ℝ, y = 4 * x - 4 ∧ x = 0 ∧ y = -4) :=
sorry

end function_intersects_y_axis_at_0_neg4_l310_310596


namespace election_winner_margin_l310_310985

theorem election_winner_margin (V : ℝ) 
    (hV: V = 3744 / 0.52) 
    (w_votes: ℝ := 3744) 
    (l_votes: ℝ := 0.48 * V) :
    w_votes - l_votes = 288 := by
  sorry

end election_winner_margin_l310_310985


namespace measure_of_angle_y_l310_310736

theorem measure_of_angle_y (m n : ℝ) (A B C D F G H : ℝ) :
  (m = n) → (A = 40) → (B = 90) → (B = 40) → (y = 80) :=
by
  -- proof steps to be filled in
  sorry

end measure_of_angle_y_l310_310736


namespace average_marks_is_25_l310_310183

variable (M P C : ℕ)

def average_math_chemistry (M C : ℕ) : ℕ :=
  (M + C) / 2

theorem average_marks_is_25 (M P C : ℕ) 
  (h₁ : M + P = 30)
  (h₂ : C = P + 20) : 
  average_math_chemistry M C = 25 :=
by
  sorry

end average_marks_is_25_l310_310183


namespace software_package_cost_l310_310040

theorem software_package_cost 
  (devices : ℕ) 
  (cost_first : ℕ) 
  (devices_covered_first : ℕ) 
  (devices_covered_second : ℕ) 
  (savings : ℕ)
  (total_cost_first : ℕ := (devices / devices_covered_first) * cost_first)
  (total_cost_second : ℕ := total_cost_first - savings)
  (num_packages_second : ℕ := devices / devices_covered_second)
  (cost_second : ℕ := total_cost_second / num_packages_second) :
  devices = 50 ∧ cost_first = 40 ∧ devices_covered_first = 5 ∧ devices_covered_second = 10 ∧ savings = 100 →
  cost_second = 60 := 
by
  sorry

end software_package_cost_l310_310040


namespace exist_n_for_all_k_l310_310535

theorem exist_n_for_all_k (k : ℕ) (h_k : k > 1) : 
  ∃ n : ℕ, 
    (n > 0 ∧ ((n.choose k) % n = 0) ∧ (∀ m : ℕ, (2 ≤ m ∧ m < k) → ((n.choose m) % n ≠ 0))) :=
sorry

end exist_n_for_all_k_l310_310535


namespace length_chord_AB_l310_310855

-- Given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 1 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 3 = 0

-- Prove the length of the chord AB
theorem length_chord_AB : 
  (∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle2 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 B.1 B.2 ∧ A ≠ B) →
  (∃ (length : ℝ), length = 2*Real.sqrt 2) :=
by
  sorry

end length_chord_AB_l310_310855


namespace airplane_seats_theorem_l310_310516

def airplane_seats_proof : Prop :=
  ∀ (s : ℝ),
  (∃ (first_class business_class economy premium_economy : ℝ),
    first_class = 30 ∧
    business_class = 0.4 * s ∧
    economy = 0.6 * s ∧
    premium_economy = s - (first_class + business_class + economy)) →
  s = 150

theorem airplane_seats_theorem : airplane_seats_proof :=
sorry

end airplane_seats_theorem_l310_310516


namespace product_modulo_seven_l310_310216

/-- 2021 is congruent to 6 modulo 7 -/
def h1 : 2021 % 7 = 6 := rfl

/-- 2022 is congruent to 0 modulo 7 -/
def h2 : 2022 % 7 = 0 := rfl

/-- 2023 is congruent to 1 modulo 7 -/
def h3 : 2023 % 7 = 1 := rfl

/-- 2024 is congruent to 2 modulo 7 -/
def h4 : 2024 % 7 = 2 := rfl

/-- The product 2021 * 2022 * 2023 * 2024 is congruent to 0 modulo 7 -/
theorem product_modulo_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
  by sorry

end product_modulo_seven_l310_310216


namespace last_two_digits_of_sum_l310_310224

theorem last_two_digits_of_sum : 
  (6.factorial + 9.factorial + 12.factorial + 15.factorial + 18.factorial + 21.factorial +
   24.factorial + 27.factorial + 30.factorial + 33.factorial + 36.factorial + 39.factorial + 
   42.factorial + 45.factorial + 48.factorial + 51.factorial + 54.factorial + 57.factorial + 
   60.factorial + 63.factorial + 66.factorial + 69.factorial + 72.factorial + 75.factorial + 
   78.factorial + 81.factorial + 84.factorial + 87.factorial + 90.factorial + 93.factorial +
   96.factorial) % 100 = 20 :=
begin
  sorry
end

end last_two_digits_of_sum_l310_310224


namespace point_in_plane_region_l310_310175

-- Defining the condition that the inequality represents a region on the plane
def plane_region (x y : ℝ) : Prop := x + 2 * y - 1 > 0

-- Stating that the point (0, 1) lies within the plane region represented by the inequality
theorem point_in_plane_region : plane_region 0 1 :=
by {
    sorry
}

end point_in_plane_region_l310_310175


namespace simplify_expression_l310_310149

theorem simplify_expression (i : ℂ) (h : i^2 = -1) : 3 * (2 - i) + i * (3 + 2 * i) = 4 :=
by
  sorry

end simplify_expression_l310_310149


namespace number_of_pints_of_paint_l310_310114

-- Statement of the problem
theorem number_of_pints_of_paint (A B : ℝ) (N : ℕ) 
  (large_cube_paint : ℝ) (hA : A = 4) (hB : B = 2) (hN : N = 125) 
  (large_cube_paint_condition : large_cube_paint = 1) : 
  (N * (B / A) ^ 2 * large_cube_paint = 31.25) :=
by {
  -- Given the conditions
  sorry
}

end number_of_pints_of_paint_l310_310114


namespace sqrt_floor_square_eq_49_l310_310874

theorem sqrt_floor_square_eq_49 : (⌊Real.sqrt 50⌋)^2 = 49 :=
by
  have h1 : 7 < Real.sqrt 50, from (by norm_num : 7 < Real.sqrt 50),
  have h2 : Real.sqrt 50 < 8, from (by norm_num : Real.sqrt 50 < 8),
  have floor_sqrt_50_eq_7 : ⌊Real.sqrt 50⌋ = 7, from Int.floor_eq_iff.mpr ⟨h1, h2⟩,
  calc
    (⌊Real.sqrt 50⌋)^2 = (7)^2 : by rw [floor_sqrt_50_eq_7]
                  ... = 49 : by norm_num,
  sorry -- omit the actual proof

end sqrt_floor_square_eq_49_l310_310874


namespace trigonometric_identity_proof_l310_310832

theorem trigonometric_identity_proof (α : ℝ) :
  3.3998 * (Real.cos α) ^ 4 - 4 * (Real.cos α) ^ 3 - 8 * (Real.cos α) ^ 2 + 3 * Real.cos α + 1 =
  -2 * Real.sin (7 * α / 2) * Real.sin (α / 2) :=
by
  sorry

end trigonometric_identity_proof_l310_310832


namespace arc_length_of_sector_l310_310347

theorem arc_length_of_sector : 
  ∀ (r : ℝ) (theta: ℝ), r = 1 ∧ theta = 30 * (Real.pi / 180) → (theta * r = Real.pi / 6) :=
by
  sorry

end arc_length_of_sector_l310_310347


namespace arithmetic_sequence_sum_l310_310405

theorem arithmetic_sequence_sum (a : ℕ → Int) (a1 a2017 : Int)
  (h1 : a 1 = a1) 
  (h2017 : a 2017 = a2017)
  (roots_eq : ∀ x, x^2 - 10 * x + 16 = 0 → (x = a1 ∨ x = a2017))
  (arith_seq : ∀ n m, a (n + 1) - a n = a (m + 1) - a m) :
  a 2 + a 1009 + a 2016 = 15 :=
by
  sorry

end arithmetic_sequence_sum_l310_310405


namespace bob_distance_walked_l310_310364

theorem bob_distance_walked
    (dist : ℕ)
    (yolanda_rate : ℕ)
    (bob_rate : ℕ)
    (hour_diff : ℕ)
    (meet_time_bob: ℕ) :

    dist = 31 → yolanda_rate = 1 → bob_rate = 2 → hour_diff = 1 → meet_time_bob = 10 →
    (bob_rate * meet_time_bob) = 20 :=
by
  intros
  sorry

end bob_distance_walked_l310_310364


namespace arithmetic_sequence_fifth_term_l310_310341

theorem arithmetic_sequence_fifth_term (x y : ℚ) 
  (h1 : a₁ = x + y) 
  (h2 : a₂ = x - y) 
  (h3 : a₃ = x * y) 
  (h4 : a₄ = x / y) 
  (h5 : a₂ - a₁ = -2 * y) 
  (h6 : a₃ - a₂ = -2 * y) 
  (h7 : a₄ - a₃ = -2 * y) 
  (hx : x = -9 / 8)
  (hy : y = -3 / 5) : 
  a₅ = 123 / 40 :=
by
  sorry

end arithmetic_sequence_fifth_term_l310_310341


namespace existential_proposition_l310_310162

theorem existential_proposition :
  (∃ x y : ℝ, x + y > 1) ∧ (∀ P : Prop, (∃ x y : ℝ, x + y > 1 → P) → P) :=
sorry

end existential_proposition_l310_310162


namespace trigonometric_identity_solution_l310_310360

theorem trigonometric_identity_solution (x : ℝ) (k : ℤ) :
  8.469 * (Real.sin x)^4 + 2 * (Real.cos x)^3 + 2 * (Real.sin x)^2 - Real.cos x + 1 = 0 ↔
  ∃ (k : ℤ), x = Real.pi + 2 * Real.pi * k := by
  sorry

end trigonometric_identity_solution_l310_310360


namespace crt_solution_l310_310072

/-- Congruences from the conditions -/
def congruences : Prop :=
  ∃ x : ℤ, 
    (x % 2 = 1) ∧
    (x % 3 = 2) ∧
    (x % 5 = 3) ∧
    (x % 7 = 4)

/-- The target result from the Chinese Remainder Theorem -/
def target_result : Prop :=
  ∃ x : ℤ, 
    (x % 210 = 53)

/-- The proof problem stating that the given conditions imply the target result -/
theorem crt_solution : congruences → target_result :=
by
  sorry

end crt_solution_l310_310072


namespace _l310_310546

noncomputable def is_tangent_to_parabola (x1 y1 p k : ℝ) : Prop :=
  let y := k * x1 - 1
  let rhs := x1^2
  rhs = y

noncomputable def leans_theorem_prover (O A B : (ℝ × ℝ)) : Prop :=
  -- Definitions of points
  let O := (0,0)
  let A := (1,1)
  let B := (0,-1)
  -- Value of p from point A on parabola C: x^2 = 2py
  let p := 1 / 2  -- as obtained by solving 1^2 = 2p * 1
  -- Checking option A: directrix is y = -1 is false
  let directrix := - p / 2
  (directrix ≠ -1) ∧
  -- Checking option B: tangent condition
  let slope_AB := (1 - (-1)) / (1 - 0)
  let tangent := is_tangent_to_parabola 1 1 p slope_AB
  tangent ∧
  -- Option C: |OP| * |OQ| = |OA|^2 is false
  let |OA|² := 2  -- obtained from the calculation |OA| = sqrt(1^2 + 1^2)
  ∀ (k > 2), k² ≠ |OA|² ∧
  -- Option D: |BP| * |BQ| > |BA|^2 is true
  let |BA|² := 5 -- obtained from the calculation |BA| = sqrt(1^2 + 4)
  ∀ (x1 x2) (hx1 : x1 + x2 = k) (hx2 : x1 * x2 = 1),
  let |BP| := sqrt(x1^2 + (x1^2 + 2x1 + 1))
  let |BQ| := sqrt(x2^2 + (x2^2 + 2x2 + 1))
  |BP| * |BQ| > |BA|²

example : leans_theorem_prover (0, 0) (1, 1) (0, -1) :=
by sorry  -- Proof is omitted, to be completed by Lean theorem prover

end _l310_310546


namespace machine_present_value_l310_310484

theorem machine_present_value
  (depreciation_rate : ℝ)
  (selling_price : ℝ)
  (profit : ℝ)
  (dep_years : ℕ)
  (value_after_depreciation : ℝ)
  (present_value : ℝ) :

  depreciation_rate = 0.8 →
  selling_price = 118000.00000000001 →
  profit = 22000 →
  dep_years = 2 →
  value_after_depreciation = (selling_price - profit) →
  value_after_depreciation = 96000.00000000001 →
  present_value * (depreciation_rate ^ dep_years) = value_after_depreciation →
  present_value = 150000.00000000002 :=
by sorry

end machine_present_value_l310_310484


namespace square_side_percentage_increase_l310_310608

theorem square_side_percentage_increase (s : ℝ) (p : ℝ) :
  (s * (1 + p / 100)) ^ 2 = 1.44 * s ^ 2 → p = 20 :=
by
  sorry

end square_side_percentage_increase_l310_310608


namespace ellipse_hyperbola_foci_l310_310775

theorem ellipse_hyperbola_foci (a b : ℝ) 
    (h1 : b^2 - a^2 = 25) 
    (h2 : a^2 + b^2 = 49) : 
    |a * b| = 2 * Real.sqrt 111 := 
by 
  -- proof omitted 
  sorry

end ellipse_hyperbola_foci_l310_310775


namespace minimum_value_of_function_l310_310894

noncomputable def y (x : ℝ) : ℝ := 4 * x + 25 / x

theorem minimum_value_of_function : ∃ x > 0, y x = 20 :=
by
  sorry

end minimum_value_of_function_l310_310894


namespace binom_sum_l310_310018

open Nat

theorem binom_sum (n : ℕ) (h : binom 25 n + binom 25 12 = binom 26 13) :
    n = 13 :=
by
  sorry

end binom_sum_l310_310018


namespace find_ab_value_l310_310774

-- Definitions from the conditions
def ellipse_eq (a b : ℝ) : Prop := b^2 - a^2 = 25
def hyperbola_eq (a b : ℝ) : Prop := a^2 + b^2 = 49

-- Main theorem statement
theorem find_ab_value {a b : ℝ} (h_ellipse : ellipse_eq a b) (h_hyperbola : hyperbola_eq a b) : 
  |a * b| = 2 * Real.sqrt 111 :=
by
  -- Proof goes here
  sorry

end find_ab_value_l310_310774


namespace bottle_caps_per_child_l310_310534

-- Define the conditions
def num_children : ℕ := 9
def total_bottle_caps : ℕ := 45

-- State the theorem that needs to be proved: each child has 5 bottle caps
theorem bottle_caps_per_child : (total_bottle_caps / num_children) = 5 := by
  sorry

end bottle_caps_per_child_l310_310534


namespace count_integers_l310_310247

theorem count_integers (n : ℕ) (h : n = 33000) :
  ∃ k : ℕ, k = 1600 ∧
  (∀ x, 1 ≤ x ∧ x ≤ n → (x % 11 = 0 → (x % 3 ≠ 0 ∧ x % 5 ≠ 0) → x ≤ x)) :=
by 
  sorry

end count_integers_l310_310247


namespace initial_percentage_reduction_l310_310479

theorem initial_percentage_reduction (x : ℝ) :
  (1 - x / 100) * 1.17649 = 1 → x = 15 :=
by
  sorry

end initial_percentage_reduction_l310_310479


namespace fraction_of_historical_fiction_new_releases_l310_310518

theorem fraction_of_historical_fiction_new_releases (total_books : ℕ) (p1 p2 p3 : ℕ) (frac_hist_fic : Rat) (frac_new_hist_fic : Rat) (frac_new_non_hist_fic : Rat) 
  (h1 : total_books > 0) (h2 : frac_hist_fic = 40 / 100) (h3 : frac_new_hist_fic = 40 / 100) (h4 : frac_new_non_hist_fic = 40 / 100) 
  (h5 : p1 = frac_hist_fic * total_books) (h6 : p2 = frac_new_hist_fic * p1) (h7 : p3 = frac_new_non_hist_fic * (total_books - p1)) :
  p2 / (p2 + p3) = 2 / 5 :=
by
  sorry

end fraction_of_historical_fiction_new_releases_l310_310518


namespace intersection_S_T_eq_interval_l310_310706

-- Define the sets S and T
def S : Set ℝ := {x | x ≥ 2}
def T : Set ℝ := {x | x ≤ 5}

-- Prove the intersection of S and T is [2, 5]
theorem intersection_S_T_eq_interval : S ∩ T = {x | 2 ≤ x ∧ x ≤ 5} :=
by
  sorry

end intersection_S_T_eq_interval_l310_310706


namespace arithmetic_sequence_transformation_l310_310543

theorem arithmetic_sequence_transformation (a : ℕ → ℝ) (d c : ℝ) (h : ∀ n, a (n + 1) = a n + d) (hc : c ≠ 0) :
  ∀ n, (c * a (n + 1)) - (c * a n) = c * d := 
by
  sorry

end arithmetic_sequence_transformation_l310_310543


namespace find_A_l310_310344

theorem find_A (J : ℤ := 15)
  (JAVA_pts : ℤ := 50)
  (AJAX_pts : ℤ := 53)
  (AXLE_pts : ℤ := 40)
  (L : ℤ := 12)
  (JAVA_eq : ∀ A V : ℤ, 2 * A + V + J = JAVA_pts)
  (AJAX_eq : ∀ A X : ℤ, 2 * A + X + J = AJAX_pts)
  (AXLE_eq : ∀ A X E : ℤ, A + X + L + E = AXLE_pts) : A = 21 :=
sorry

end find_A_l310_310344


namespace coefficient_of_x_in_expansion_l310_310528

noncomputable def binomial_expansion_term (r : ℕ) : ℤ :=
  (-1)^r * (2^(5-r)) * Nat.choose 5 r

theorem coefficient_of_x_in_expansion :
  binomial_expansion_term 3 = -40 := by
  sorry

end coefficient_of_x_in_expansion_l310_310528


namespace model2_best_fit_l310_310431
-- Import necessary tools from Mathlib

-- Define the coefficients of determination for the four models
def R2_model1 : ℝ := 0.75
def R2_model2 : ℝ := 0.90
def R2_model3 : ℝ := 0.28
def R2_model4 : ℝ := 0.55

-- Define the best fitting model
def best_fitting_model (R2_1 R2_2 R2_3 R2_4 : ℝ) : Prop :=
  R2_2 > R2_1 ∧ R2_2 > R2_3 ∧ R2_2 > R2_4

-- Statement to prove
theorem model2_best_fit : best_fitting_model R2_model1 R2_model2 R2_model3 R2_model4 :=
  by
  -- Proof goes here
  sorry

end model2_best_fit_l310_310431


namespace strange_number_l310_310602

theorem strange_number (x : ℤ) (h : (x - 7) * 7 = (x - 11) * 11) : x = 18 :=
sorry

end strange_number_l310_310602


namespace set_intersection_correct_l310_310249

def set_A := {x : ℝ | x + 1 > 0}
def set_B := {x : ℝ | x - 3 < 0}
def set_intersection := {x : ℝ | -1 < x ∧ x < 3}

theorem set_intersection_correct : (set_A ∩ set_B) = set_intersection :=
by
  sorry

end set_intersection_correct_l310_310249


namespace probability_one_white_one_black_l310_310426

-- Define the basic setup of the problem
def setup : Type :=
  { red : ℕ // red = 1 } ×
  { white : ℕ // white = 2 } ×
  { black : ℕ // black = 3 }

-- Statement of the problem
theorem probability_one_white_one_black (s : setup) :
  let total_balls := 6
      total_combinations := (total_balls.choose 2)
      favourable_combinations := (s.2.1.val * s.2.2.1.val)
  in (favourable_combinations : ℚ) / (total_combinations : ℚ) = 2/5 :=
by
  sorry

end probability_one_white_one_black_l310_310426


namespace quadratic_real_roots_leq_l310_310944

theorem quadratic_real_roots_leq (m : ℝ) :
  ∃ x : ℝ, x^2 - 3 * x + 2 * m = 0 → m ≤ 9 / 8 :=
by
  sorry

end quadratic_real_roots_leq_l310_310944


namespace minimum_number_of_troublemakers_l310_310787

theorem minimum_number_of_troublemakers (n m : ℕ) (students : Fin n → Prop) 
  (truthtellers : Fin n → Prop) (liars : Fin n → Prop) (round_table : List (Fin n))
  (h_all_students : n = 29)
  (h_truth_or_liar : ∀ s, students s → (truthtellers s ∨ liars s))
  (h_truth_tellers_conditions : ∀ s, truthtellers s → 
    (count_lies_next_to s = 1 ∨ count_lies_next_to s = 2))
  (h_liars_conditions : ∀ s, liars s → 
    (count_lies_next_to s ≠ 1 ∧ count_lies_next_to s ≠ 2))
  (h_round_table_conditions : round_table.length = n 
    ∧ (∀ i, i < n → round_table.nth i ≠ none)):
  m = 10 :=
by
  sorry

end minimum_number_of_troublemakers_l310_310787


namespace population_of_missing_village_l310_310477

theorem population_of_missing_village 
  (pop1 pop2 pop3 pop4 pop5 pop6 : ℕ) 
  (avg_pop : ℕ) 
  (h1 : pop1 = 803)
  (h2 : pop2 = 900)
  (h3 : pop3 = 1023)
  (h4 : pop4 = 945)
  (h5 : pop5 = 980)
  (h6 : pop6 = 1249)
  (h_avg : avg_pop = 1000) :
  ∃ (pop_missing : ℕ), pop_missing = 1100 := 
by
  -- Placeholder for proof
  sorry

end population_of_missing_village_l310_310477


namespace rectangle_area_from_square_l310_310646

theorem rectangle_area_from_square {s a : ℝ} (h1 : s^2 = 36) (h2 : a = 3 * s) :
    s * a = 108 :=
by
  -- The proof goes here
  sorry

end rectangle_area_from_square_l310_310646


namespace geometric_sequence_value_of_m_l310_310397

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_value_of_m (r : ℝ) (hr : r ≠ 1) 
    (h1 : is_geometric_sequence a r)
    (h2 : a 5 * a 6 + a 4 * a 7 = 18) 
    (h3 : a 1 * a m = 9) :
  m = 10 :=
by
  sorry

end geometric_sequence_value_of_m_l310_310397


namespace exchange_rate_l310_310581

def jackPounds : ℕ := 42
def jackEuros : ℕ := 11
def jackYen : ℕ := 3000
def poundsPerYen : ℕ := 100
def totalYen : ℕ := 9400

theorem exchange_rate :
  ∃ (x : ℕ), 100 * jackPounds + 100 * jackEuros * x + jackYen = totalYen ∧ x = 2 :=
by
  sorry

end exchange_rate_l310_310581


namespace find_k1_over_k2_plus_k2_over_k1_l310_310961

theorem find_k1_over_k2_plus_k2_over_k1 (p q k k1 k2 : ℚ)
  (h1 : k * (p^2) - (2 * k - 3) * p + 7 = 0)
  (h2 : k * (q^2) - (2 * k - 3) * q + 7 = 0)
  (h3 : p ≠ 0)
  (h4 : q ≠ 0)
  (h5 : k ≠ 0)
  (h6 : k1 ≠ 0)
  (h7 : k2 ≠ 0)
  (h8 : p / q + q / p = 6 / 7)
  (h9 : (p + q) = (2 * k - 3) / k)
  (h10 : p * q = 7 / k)
  (h11 : k1 + k2 = 6)
  (h12 : k1 * k2 = 9 / 4) :
  (k1 / k2 + k2 / k1 = 14) :=
  sorry

end find_k1_over_k2_plus_k2_over_k1_l310_310961


namespace total_ticket_cost_l310_310986

theorem total_ticket_cost (adult_tickets student_tickets : ℕ) 
    (price_adult price_student : ℕ) 
    (total_tickets : ℕ) (n_adult_tickets : adult_tickets = 410) 
    (n_student_tickets : student_tickets = 436) 
    (p_adult : price_adult = 6) 
    (p_student : price_student = 3) 
    (total_tickets_sold : total_tickets = 846) : 
    (adult_tickets * price_adult + student_tickets * price_student) = 3768 :=
by
  sorry

end total_ticket_cost_l310_310986


namespace f_a1_a3_a5_positive_l310_310086

theorem f_a1_a3_a5_positive (f : ℝ → ℝ) (a : ℕ → ℝ)
  (hf_odd : ∀ x, f (-x) = - f x)
  (hf_mono : ∀ x y, x < y → f x < f y)
  (ha_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (ha3_pos : 0 < a 3) :
  0 < f (a 1) + f (a 3) + f (a 5) :=
sorry

end f_a1_a3_a5_positive_l310_310086


namespace P_has_common_root_l310_310473

def P (x : ℝ) (p : ℝ) (q : ℝ) : ℝ := x^2 + p * x + q

theorem P_has_common_root (p q : ℝ) (t : ℝ) (h : P t p q = 0) :
  P 0 p q * P 1 p q = 0 :=
by
  sorry

end P_has_common_root_l310_310473


namespace number_of_pairs_l310_310698

open Nat

theorem number_of_pairs :
  ∃ n, n = 9 ∧
    (∃ x y : ℕ,
      x > 0 ∧ y > 0 ∧
      x + y = 150 ∧
      x % 3 = 0 ∧
      y % 5 = 0 ∧
      (∃! (x y : ℕ), x + y = 150 ∧ x % 3 = 0 ∧ y % 5 = 0 ∧ x > 0 ∧ y > 0)) := sorry

end number_of_pairs_l310_310698


namespace initial_pipes_count_l310_310838

theorem initial_pipes_count (n : ℕ) (r : ℝ) :
  n * r = 1 / 16 → (n + 15) * r = 1 / 4 → n = 5 :=
by
  intro h1 h2
  sorry

end initial_pipes_count_l310_310838


namespace smaller_molds_radius_l310_310511

theorem smaller_molds_radius (r : ℝ) : 
  (∀ V_large V_small : ℝ, 
     V_large = (2/3) * π * (2:ℝ)^3 ∧
     V_small = (2/3) * π * r^3 ∧
     8 * V_small = V_large) → r = 1 := by
  sorry

end smaller_molds_radius_l310_310511


namespace jennifer_total_discount_is_28_l310_310438

-- Define the conditions in the Lean context

def initial_whole_milk_cans : ℕ := 40 
def mark_whole_milk_cans : ℕ := 30 
def mark_skim_milk_cans : ℕ := 15 
def almond_milk_per_3_whole_milk : ℕ := 2 
def whole_milk_per_5_skim_milk : ℕ := 4 
def discount_per_10_whole_milk : ℕ := 4 
def discount_per_7_almond_milk : ℕ := 3 
def discount_per_3_almond_milk : ℕ := 1

def jennifer_additional_almond_milk := (mark_whole_milk_cans / 3) * almond_milk_per_3_whole_milk
def jennifer_additional_whole_milk := (mark_skim_milk_cans / 5) * whole_milk_per_5_skim_milk

def jennifer_whole_milk_cans := initial_whole_milk_cans + jennifer_additional_whole_milk
def jennifer_almond_milk_cans := jennifer_additional_almond_milk

def jennifer_whole_milk_discount := (jennifer_whole_milk_cans / 10) * discount_per_10_whole_milk
def jennifer_almond_milk_discount := 
  (jennifer_almond_milk_cans / 7) * discount_per_7_almond_milk + 
  ((jennifer_almond_milk_cans % 7) / 3) * discount_per_3_almond_milk

def total_jennifer_discount := jennifer_whole_milk_discount + jennifer_almond_milk_discount

-- Theorem stating the total discount 
theorem jennifer_total_discount_is_28 : total_jennifer_discount = 28 := by
  sorry

end jennifer_total_discount_is_28_l310_310438


namespace minimum_liars_l310_310782

noncomputable def minimum_troublemakers (n : ℕ) := 10

theorem minimum_liars (n : ℕ) (students : ℕ → bool) 
  (seated_at_round_table : ∀ i, students (i % 29) = tt ∨ students (i % 29) = ff)
  (one_liar_next_to : ∀ i, students i = tt → (students ((i + 1) % 29) = ff ∨ students ((i + 28) % 29) = ff))
  (two_liars_next_to : ∀ i, students i = ff → ((students ((i + 1) % 29) = ff) ∧ students ((i + 28) % 29) = ff)) :
  ∃ m, m = minimum_troublemakers 29 ∧ (m ≤ n) :=
by
  let liars_count := 10
  have : liars_count = 10 := rfl
  use liars_count
  exact ⟨rfl, sorry⟩

end minimum_liars_l310_310782


namespace problem_equiv_conditions_l310_310957

theorem problem_equiv_conditions (n : ℕ) :
  (∀ a : ℕ, n ∣ a^n - a) ↔ (∀ p : ℕ, p ∣ n → Prime p → ¬ p^2 ∣ n ∧ (p - 1) ∣ (n - 1)) :=
sorry

end problem_equiv_conditions_l310_310957


namespace sin_cos_power_equality_l310_310125

theorem sin_cos_power_equality (θ : ℝ) (h : cos (2 * θ) = 1 / 4) : (sin θ) ^ 6 + (cos θ) ^ 6 = 19 / 64 :=
by
  sorry

end sin_cos_power_equality_l310_310125


namespace total_bill_is_correct_l310_310191

-- Given conditions
def hourly_rate := 45
def parts_cost := 225
def hours_worked := 5

-- Total bill calculation
def labor_cost := hourly_rate * hours_worked
def total_bill := labor_cost + parts_cost

-- Prove that the total bill is equal to 450 dollars
theorem total_bill_is_correct : total_bill = 450 := by
  sorry

end total_bill_is_correct_l310_310191


namespace expression_undefined_iff_l310_310907

theorem expression_undefined_iff (a : ℝ) : (a^2 - 9 = 0) ↔ (a = 3 ∨ a = -3) :=
sorry

end expression_undefined_iff_l310_310907


namespace forum_posting_total_l310_310194

theorem forum_posting_total (num_members : ℕ) (num_answers_per_question : ℕ) (num_questions_per_hour : ℕ) (hours_per_day : ℕ) :
  num_members = 1000 ->
  num_answers_per_question = 5 ->
  num_questions_per_hour = 7 ->
  hours_per_day = 24 ->
  ((num_questions_per_hour * hours_per_day * num_members) + (num_answers_per_question * num_questions_per_hour * hours_per_day * num_members)) = 1008000 :=
by
  intros
  sorry

end forum_posting_total_l310_310194


namespace evaluate_expr_l310_310992

theorem evaluate_expr :
  (150^2 - 12^2) / (90^2 - 21^2) * ((90 + 21) * (90 - 21)) / ((150 + 12) * (150 - 12)) = 2 :=
by sorry

end evaluate_expr_l310_310992


namespace sides_imply_angles_l310_310425

noncomputable theory

open Real

theorem sides_imply_angles {a b A B C : ℝ}
  (h_triangle: a > 0 ∧ b > 0 ∧ C > 0)
  (h_sides_opposite: ∠ABC = A ∧ ∠BAC = B)
  (h_law_of_sines: a / sin A = b / sin B) :
  (a > b ↔ sin A > sin B) :=
  sorry

end sides_imply_angles_l310_310425


namespace probability_of_matching_pair_l310_310217

/-!
# Probability of Selecting a Matching Pair of Shoes

Given:
- 12 pairs of sneakers, each with a 4% probability of being chosen.
- 15 pairs of boots, each with a 3% probability of being chosen.
- 18 pairs of dress shoes, each with a 2% probability of being chosen.

If two shoes are selected from the warehouse without replacement, prove that the probability 
of selecting a matching pair of shoes is 52.26%.
-/

namespace ShoeWarehouse

def prob_sneakers_first : ℝ := 0.48
def prob_sneakers_second : ℝ := 0.44
def prob_boots_first : ℝ := 0.45
def prob_boots_second : ℝ := 0.42
def prob_dress_first : ℝ := 0.36
def prob_dress_second : ℝ := 0.34

theorem probability_of_matching_pair :
  (prob_sneakers_first * prob_sneakers_second) +
  (prob_boots_first * prob_boots_second) +
  (prob_dress_first * prob_dress_second) = 0.5226 :=
sorry

end ShoeWarehouse

end probability_of_matching_pair_l310_310217


namespace simplify_expression_l310_310621

theorem simplify_expression : 1 + (1 / (1 + (1 / (2 + 1)))) = 7 / 4 :=
by
  sorry

end simplify_expression_l310_310621


namespace sum_of_squares_not_perfect_square_l310_310144

theorem sum_of_squares_not_perfect_square (n : ℕ) (h : n > 4) :
  ¬ (∃ k : ℕ, 10 * n^2 + 10 * n + 85 = k^2) :=
sorry

end sum_of_squares_not_perfect_square_l310_310144


namespace four_digit_number_divisible_by_36_l310_310232

theorem four_digit_number_divisible_by_36 (n : ℕ) (h₁ : ∃ k : ℕ, 6130 + n = 36 * k) 
  (h₂ : ∃ k : ℕ, 130 + n = 4 * k) 
  (h₃ : ∃ k : ℕ, (10 + n) = 9 * k) : n = 6 :=
sorry

end four_digit_number_divisible_by_36_l310_310232


namespace basic_computer_price_l310_310184

variables (C P : ℕ)

theorem basic_computer_price (h1 : C + P = 2500)
                            (h2 : C + 500 + P = 6 * P) : C = 2000 :=
by
  sorry

end basic_computer_price_l310_310184


namespace find_second_number_l310_310634

theorem find_second_number (A B : ℝ) (h1 : A = 6400) (h2 : 0.05 * A = 0.2 * B + 190) : B = 650 :=
by
  sorry

end find_second_number_l310_310634


namespace find_sum_A_B_C_l310_310156

theorem find_sum_A_B_C (A B C : ℤ)
  (h1 : ∀ x > 4, (x^2 : ℝ) / (A * x^2 + B * x + C) > 0.4)
  (h2 : A * (-2)^2 + B * (-2) + C = 0)
  (h3 : A * (3)^2 + B * (3) + C = 0)
  (h4 : 0.4 < 1 / (A : ℝ) ∧ 1 / (A : ℝ) < 1) :
  A + B + C = -12 :=
by
  sorry

end find_sum_A_B_C_l310_310156


namespace angle_in_quadrants_l310_310561

theorem angle_in_quadrants (α : ℝ) (hα : 0 < α ∧ α < π / 2) (k : ℤ) :
  (∃ i : ℤ, k = 2 * i + 1 ∧ π < (2 * i + 1) * π + α ∧ (2 * i + 1) * π + α < 3 * π / 2) ∨
  (∃ i : ℤ, k = 2 * i ∧ 0 < 2 * i * π + α ∧ 2 * i * π + α < π / 2) :=
sorry

end angle_in_quadrants_l310_310561


namespace probability_of_triangle_l310_310487

/-- There are 12 figures in total: 4 squares, 5 triangles, and 3 rectangles.
    Prove that the probability of choosing a triangle is 5/12. -/
theorem probability_of_triangle (total_figures : ℕ) (num_squares : ℕ) (num_triangles : ℕ) (num_rectangles : ℕ)
  (h1 : total_figures = 12)
  (h2 : num_squares = 4)
  (h3 : num_triangles = 5)
  (h4 : num_rectangles = 3) :
  num_triangles / total_figures = 5 / 12 :=
sorry

end probability_of_triangle_l310_310487


namespace infinite_geometric_series_sum_l310_310064

theorem infinite_geometric_series_sum :
  let a := (5 : ℚ) / 3
  let r := -(3 : ℚ) / 4
  ∑' n : ℕ, a * r ^ n = 20 / 21 := by
  sorry

end infinite_geometric_series_sum_l310_310064


namespace some_students_are_not_club_members_l310_310200

variable (U : Type) -- U represents the universe of students and club members
variables (Student ClubMember StudyLate : U → Prop)

-- Conditions derived from the problem
axiom h1 : ∃ s, Student s ∧ ¬ StudyLate s -- Some students do not study late
axiom h2 : ∀ c, ClubMember c → StudyLate c -- All club members study late

theorem some_students_are_not_club_members :
  ∃ s, Student s ∧ ¬ ClubMember s :=
by
  sorry

end some_students_are_not_club_members_l310_310200


namespace range_of_x_l310_310239

noncomputable def f (x : ℝ) : ℝ :=
  Real.log (Real.exp x + Real.exp (-x)) + x^2

theorem range_of_x (x : ℝ) : 
  (∃ y z : ℝ, y = 2 * x - 1 ∧ f x > f y ∧ x > 1 / 3 ∧ x < 1) :=
sorry

end range_of_x_l310_310239


namespace initial_reading_times_per_day_l310_310955

-- Definitions based on the conditions

/-- Number of pages Jessy plans to read initially in each session is 6. -/
def session_pages : ℕ := 6

/-- Jessy needs to read 140 pages in one week. -/
def total_pages : ℕ := 140

/-- Jessy reads an additional 2 pages per day to achieve her goal. -/
def additional_daily_pages : ℕ := 2

/-- Days in a week -/
def days_in_week : ℕ := 7

-- Proving Jessy's initial plan for reading times per day
theorem initial_reading_times_per_day (x : ℕ) (h : days_in_week * (session_pages * x + additional_daily_pages) = total_pages) : 
    x = 3 := by
  -- skipping the proof itself
  sorry

end initial_reading_times_per_day_l310_310955


namespace find_u_plus_v_l310_310564

theorem find_u_plus_v (u v : ℚ) 
  (h₁ : 3 * u + 7 * v = 17) 
  (h₂ : 5 * u - 3 * v = 9) : 
  u + v = 43 / 11 :=
sorry

end find_u_plus_v_l310_310564


namespace range_of_m_for_distinct_real_roots_of_quadratic_l310_310722

theorem range_of_m_for_distinct_real_roots_of_quadratic (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 4*x1 - m = 0 ∧ x2^2 + 4*x2 - m = 0) ↔ m > -4 :=
by
  sorry

end range_of_m_for_distinct_real_roots_of_quadratic_l310_310722


namespace and_or_distrib_left_or_and_distrib_right_l310_310366

theorem and_or_distrib_left (A B C : Prop) : A ∧ (B ∨ C) ↔ (A ∧ B) ∨ (A ∧ C) :=
sorry

theorem or_and_distrib_right (A B C : Prop) : A ∨ (B ∧ C) ↔ (A ∨ B) ∧ (A ∨ C) :=
sorry

end and_or_distrib_left_or_and_distrib_right_l310_310366


namespace negation_of_p_l310_310411

-- Define the original proposition p
def p : Prop := ∀ x : ℝ, x ≥ 2

-- State the proof problem as a Lean theorem
theorem negation_of_p : (∀ x : ℝ, x ≥ 2) → ∃ x₀ : ℝ, x₀ < 2 :=
by
  intro h
  -- Define how the proof would generally proceed
  -- as the negation of a universal statement is an existential statement.
  sorry

end negation_of_p_l310_310411


namespace min_troublemakers_29_l310_310806

noncomputable def min_troublemakers (n : ℕ) : ℕ :=
sorry

theorem min_troublemakers_29 : min_troublemakers 29 = 10 := 
sorry

end min_troublemakers_29_l310_310806


namespace triangle_inequality_l310_310230

theorem triangle_inequality (a b c : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
  (h_triangle : (a^2 + b^2 > c^2) ∧ (b^2 + c^2 > a^2) ∧ (c^2 + a^2 > b^2)) :
  (a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) ≥ 4 * (a^6 + b^6 + c^6) :=
sorry

end triangle_inequality_l310_310230


namespace rotating_right_triangle_results_in_cone_l310_310500

theorem rotating_right_triangle_results_in_cone (T : Triangle) (h : isRightTriangle T) (leg : Side T) :
  ¬(isHypotenuse leg) → 
  resultingShapeFromRotation T leg = Shape.cone :=
by
  sorry

end rotating_right_triangle_results_in_cone_l310_310500


namespace no_two_ways_for_z_l310_310867

theorem no_two_ways_for_z (z : ℤ) (x y x' y' : ℕ) 
  (hx : x ≤ y) (hx' : x' ≤ y') : ¬ (z = x! + y! ∧ z = x'! + y'! ∧ (x ≠ x' ∨ y ≠ y')) :=
by
  sorry

end no_two_ways_for_z_l310_310867


namespace area_of_rectangle_is_108_l310_310655

theorem area_of_rectangle_is_108 (s w l : ℕ) (h₁ : s * s = 36) (h₂ : w = s) (h₃ : l = 3 * w) : w * l = 108 :=
by
  -- This is a placeholder for a detailed proof.
  sorry

end area_of_rectangle_is_108_l310_310655


namespace dice_probability_l310_310828

theorem dice_probability :
  ∃ (p : ℚ), 
    (∀ (die1 die2 : ℕ), die1 ∈ finset.Icc 1 6 → 
                        die2 ∈ finset.Icc 1 6 → 
                        die1 ≠ die2 → 
                        (die1 = 3 ∨ die2 = 3) → 
                        p = 1/3) :=
begin
  sorry
end

end dice_probability_l310_310828


namespace fraction_subtraction_l310_310071

theorem fraction_subtraction :
  (8 / 23) - (5 / 46) = 11 / 46 := by
  sorry

end fraction_subtraction_l310_310071


namespace remainder_when_added_then_divided_l310_310822

def num1 : ℕ := 2058167
def num2 : ℕ := 934
def divisor : ℕ := 8

theorem remainder_when_added_then_divided :
  (num1 + num2) % divisor = 5 := 
sorry

end remainder_when_added_then_divided_l310_310822


namespace min_troublemakers_l310_310792

theorem min_troublemakers (n : ℕ) (students : ℕ → Prop) 
  (h : n = 29)
  (condition1 : ∀ i, students i → (students ((i - 1) % n) ↔ ¬ students ((i + 1) % n)))
  (condition2 : ∀ i, ¬ students i → (students ((i - 1) % n) ∧ students ((i + 1) % n)))
  : ∃ L : ℕ, (L ≤ 29 ∧ L ≥ 10) :=
by sorry

end min_troublemakers_l310_310792


namespace scientific_notation_14000000_l310_310606

theorem scientific_notation_14000000 :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 14000000 = a * 10 ^ n ∧ a = 1.4 ∧ n = 7 :=
by
  sorry

end scientific_notation_14000000_l310_310606


namespace total_accepted_cartons_l310_310440

theorem total_accepted_cartons 
  (total_cartons : ℕ) 
  (customers : ℕ) 
  (damaged_cartons : ℕ)
  (h1 : total_cartons = 400)
  (h2 : customers = 4)
  (h3 : damaged_cartons = 60)
  : total_cartons / customers * (customers - (damaged_cartons / (total_cartons / customers))) = 160 := by
  sorry

end total_accepted_cartons_l310_310440


namespace volume_of_red_tetrahedron_in_colored_cube_l310_310032

noncomputable def red_tetrahedron_volume (side_length : ℝ) : ℝ :=
  let cube_volume := side_length ^ 3
  let clear_tetrahedron_volume := (1/3) * (1/2 * side_length * side_length) * side_length
  let red_tetrahedron_volume := (cube_volume - 4 * clear_tetrahedron_volume)
  red_tetrahedron_volume

theorem volume_of_red_tetrahedron_in_colored_cube 
: red_tetrahedron_volume 8 = 512 / 3 := by
  sorry

end volume_of_red_tetrahedron_in_colored_cube_l310_310032


namespace min_people_like_mozart_bach_not_beethoven_l310_310760

-- Define the initial conditions
variables {n a b c : ℕ}
variables (total_people := 150)
variables (likes_mozart := 120)
variables (likes_bach := 105)
variables (likes_beethoven := 45)

theorem min_people_like_mozart_bach_not_beethoven : 
  ∃ (x : ℕ), 
    total_people = 150 ∧ 
    likes_mozart = 120 ∧ 
    likes_bach = 105 ∧ 
    likes_beethoven = 45 ∧ 
    x = (likes_mozart + likes_bach - total_people) := 
    sorry

end min_people_like_mozart_bach_not_beethoven_l310_310760


namespace product_of_solutions_of_x_squared_eq_49_l310_310681

theorem product_of_solutions_of_x_squared_eq_49 : 
  (∀ x, x^2 = 49 → x = 7 ∨ x = -7) → (7 * (-7) = -49) :=
by
  intros
  sorry

end product_of_solutions_of_x_squared_eq_49_l310_310681


namespace number_of_goats_l310_310488

-- Mathematical definitions based on the conditions
def number_of_hens : ℕ := 10
def total_cost : ℤ := 2500
def price_per_hen : ℤ := 50
def price_per_goat : ℤ := 400

-- Prove the number of goats
theorem number_of_goats (G : ℕ) : 
  number_of_hens * price_per_hen + G * price_per_goat = total_cost ↔ G = 5 := 
by
  sorry

end number_of_goats_l310_310488


namespace min_liars_needed_l310_310801

namespace Problem

/-
  We introduce our problem using Lean definitions and statements.
-/

def students_sitting_at_table : ℕ := 29

structure Student :=
(is_liar : Bool)

def student_truth_statement (s: ℕ) (next1: Student) (next2: Student) : Bool :=
  if s then (next1.is_liar && !next2.is_liar) || (!next1.is_liar && next2.is_liar)
  else next1.is_liar && next2.is_liar

def minimum_liars_class (sitting_students: ℕ) : ℕ :=
  sitting_students / 3 + (if sitting_students % 3 = 0 then 0 else 1)

theorem min_liars_needed (students : List Student) (H_total : students.length = students_sitting_at_table)
  (H_truth_teller_claim : ∀ (i : ℕ), i < students.length → student_truth_statement i (students.get ⟨(i+1) % students.length, sorry⟩) (students.get ⟨(i+2) % students.length, sorry⟩)) :
  (students.count (λ s => s.is_liar)) ≥ minimum_liars_class students_sitting_at_table :=
sorry

end Problem

end min_liars_needed_l310_310801


namespace shopkeeper_percentage_above_cost_l310_310642

theorem shopkeeper_percentage_above_cost (CP MP SP : ℚ) 
  (h1 : CP = 100) 
  (h2 : SP = CP * 1.02)
  (h3 : SP = MP * 0.85) : 
  (MP - CP) / CP * 100 = 20 :=
by sorry

end shopkeeper_percentage_above_cost_l310_310642


namespace factorize_correct_l310_310676
noncomputable def factorize_expression (a b : ℝ) : ℝ :=
  (a - b)^4 + (a + b)^4 + (a + b)^2 * (a - b)^2

theorem factorize_correct (a b : ℝ) :
  factorize_expression a b = (3 * a^2 + b^2) * (a^2 + 3 * b^2) :=
by
  sorry

end factorize_correct_l310_310676


namespace meet_days_l310_310269

-- Definition of conditions
def person_a_days : ℕ := 5
def person_b_days : ℕ := 7
def person_b_early_departure : ℕ := 2

-- Definition of the number of days after A's start that they meet
variable {x : ℕ}

-- Statement to be proven
theorem meet_days (x : ℕ) : (x + 2 : ℚ) / person_b_days + x / person_a_days = 1 := sorry

end meet_days_l310_310269


namespace rose_can_afford_l310_310763

noncomputable def total_cost_before_discount : ℝ :=
  2.40 + 9.20 + 6.50 + 12.25 + 4.75

noncomputable def discount : ℝ :=
  0.15 * total_cost_before_discount

noncomputable def total_cost_after_discount : ℝ :=
  total_cost_before_discount - discount

noncomputable def budget : ℝ :=
  30.00

noncomputable def remaining_budget : ℝ :=
  budget - total_cost_after_discount

theorem rose_can_afford :
  remaining_budget = 0.165 :=
by
  -- proof goes here
  sorry

end rose_can_afford_l310_310763


namespace tan_10pi_minus_theta_l310_310550

open Real

theorem tan_10pi_minus_theta (θ : ℝ) (h1 : π < θ) (h2 : θ < 2 * π) (h3 : cos (θ - 9 * π) = -3 / 5) : 
  tan (10 * π - θ) = -4 / 3 := 
sorry

end tan_10pi_minus_theta_l310_310550


namespace pushups_count_l310_310864

theorem pushups_count :
  ∀ (David Zachary Hailey : ℕ),
    David = 44 ∧ (David = Zachary + 9) ∧ (Zachary = 2 * Hailey) ∧ (Hailey = 27) →
      (David = 63 ∧ Zachary = 54 ∧ Hailey = 27) :=
by
  intros David Zachary Hailey
  intro conditions
  obtain ⟨hDavid44, hDavid9Zachary, hZachary2Hailey, hHailey27⟩ := conditions
  sorry

end pushups_count_l310_310864


namespace ladybugs_calculation_l310_310467

def total_ladybugs : ℕ := 67082
def ladybugs_with_spots : ℕ := 12170
def ladybugs_without_spots : ℕ := 54912

theorem ladybugs_calculation :
  total_ladybugs - ladybugs_with_spots = ladybugs_without_spots :=
by
  sorry

end ladybugs_calculation_l310_310467


namespace digit_205_of_14_div_360_l310_310818

noncomputable def decimal_expansion_of_fraction (n d : ℕ) : ℕ → ℕ := sorry

theorem digit_205_of_14_div_360 : 
  decimal_expansion_of_fraction 14 360 205 = 8 :=
sorry

end digit_205_of_14_div_360_l310_310818


namespace distributive_addition_over_multiplication_not_hold_l310_310843

def complex_add (z1 z2 : ℝ × ℝ) : ℝ × ℝ :=
(z1.1 + z2.1, z1.2 + z2.2)

def complex_mul (z1 z2 : ℝ × ℝ) : ℝ × ℝ :=
(z1.1 * z2.1 - z1.2 * z2.2, z1.1 * z2.2 + z1.2 * z2.1)

theorem distributive_addition_over_multiplication_not_hold (x y x1 y1 x2 y2 : ℝ) :
  complex_add (x, y) (complex_mul (x1, y1) (x2, y2)) ≠
    complex_mul (complex_add (x, y) (x1, y1)) (complex_add (x, y) (x2, y2)) :=
sorry

end distributive_addition_over_multiplication_not_hold_l310_310843


namespace rectangle_dimensions_l310_310415

theorem rectangle_dimensions (l w : ℝ) (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 2880) :
  (l = 86.833 ∧ w = 33.167) ∨ (l = 33.167 ∧ w = 86.833) :=
by
  sorry

end rectangle_dimensions_l310_310415


namespace perfect_squares_between_100_and_400_l310_310105

theorem perfect_squares_between_100_and_400 :
  let n := 11
  let m := 19
  list.count (list.map (λ x, x * x) (list.range (m - n + 1) + (fun c => c + n))) = 9 := by
    sorry  -- Proof omitted

end perfect_squares_between_100_and_400_l310_310105


namespace fraction_comparison_l310_310472

noncomputable def one_seventh : ℚ := 1 / 7
noncomputable def decimal_0_point_14285714285 : ℚ := 14285714285 / 10^11
noncomputable def eps_1 : ℚ := 1 / (7 * 10^11)
noncomputable def eps_2 : ℚ := 1 / (7 * 10^12)

theorem fraction_comparison :
  one_seventh = decimal_0_point_14285714285 + eps_1 :=
sorry

end fraction_comparison_l310_310472


namespace integral_abs_x_minus_two_l310_310050

theorem integral_abs_x_minus_two : ∫ x in (0:ℝ)..4, |x - 2| = 4 := 
by
  sorry

end integral_abs_x_minus_two_l310_310050


namespace nearest_integer_to_3_plus_sqrt2_pow_four_l310_310616

open Real

theorem nearest_integer_to_3_plus_sqrt2_pow_four : 
  (∃ n : ℤ, abs (n - (3 + (sqrt 2))^4) < 0.5) ∧ 
  (abs (382 - (3 + (sqrt 2))^4) < 0.5) := 
by 
  sorry

end nearest_integer_to_3_plus_sqrt2_pow_four_l310_310616


namespace xiao_yang_correct_answers_l310_310575

noncomputable def problems_group_a : ℕ := 5
noncomputable def points_per_problem_group_a : ℕ := 8
noncomputable def problems_group_b : ℕ := 12
noncomputable def points_per_problem_group_b_correct : ℕ := 5
noncomputable def points_per_problem_group_b_incorrect : ℤ := -2
noncomputable def total_score : ℕ := 71
noncomputable def correct_answers_group_a : ℕ := 2 -- minimum required
noncomputable def correct_answers_total : ℕ := 13 -- provided correct result by the problem

theorem xiao_yang_correct_answers : correct_answers_total = 13 := by
  sorry

end xiao_yang_correct_answers_l310_310575


namespace intersection_M_N_l310_310705

-- Definitions for the sets M and N based on the given conditions
def M : Set ℝ := { x | x < 2 }
def N : Set ℝ := { x | x^2 - 3 * x - 4 ≤ 0 }

-- The statement we need to prove
theorem intersection_M_N : M ∩ N = { x | -1 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end intersection_M_N_l310_310705


namespace largest_proper_subset_size_l310_310583

universe u

open Finset -- Open the Finset namespace

noncomputable def X (n : ℕ) : Finset (Fin (n+1) → ℕ) :=
  univ.filter (λ s, ∀ i, s i ∈ finset.range (i + 1))

def join (s t : Fin (n+1) → ℕ) : Fin (n+1) → ℕ :=
  λ i, max (s i) (t i)

def meet (s t : Fin (n+1) → ℕ) : Fin (n+1) → ℕ :=
  λ i, min (s i) (t i)

theorem largest_proper_subset_size (n : ℕ) (hn : n ≥ 2) :
  ∃ A ⊂ X n, (∀ s t ∈ A, join s t ∈ A ∧ meet s t ∈ A) ∧ |A| = (n + 1)! - (n - 1)! :=
begin
  sorry
end

end largest_proper_subset_size_l310_310583


namespace mixed_sum_in_range_l310_310854

def mixed_to_improper (a : ℕ) (b c : ℕ) : ℚ := a + b / c

def mixed_sum (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℕ) : ℚ :=
  (mixed_to_improper a1 b1 c1) + (mixed_to_improper a2 b2 c2) + (mixed_to_improper a3 b3 c3)

theorem mixed_sum_in_range :
  11 < mixed_sum 1 4 6 3 1 2 8 3 21 ∧ mixed_sum 1 4 6 3 1 2 8 3 21 < 12 :=
by { sorry }

end mixed_sum_in_range_l310_310854


namespace subtract_digits_value_l310_310984

theorem subtract_digits_value (A B : ℕ) (h1 : A ≠ B) (h2 : 2 * 1000 + A * 100 + 3 * 10 + 2 - (B * 100 + B * 10 + B) = 1 * 1000 + B * 100 + B * 10 + B) :
  B - A = 3 :=
by
  sorry

end subtract_digits_value_l310_310984


namespace slope_of_line_l310_310163

theorem slope_of_line {x y : ℝ} : 
  (∃ (x y : ℝ), 0 = 3 * x + 4 * y + 12) → ∀ (m : ℝ), m = -3/4 :=
by
  sorry

end slope_of_line_l310_310163


namespace solution_set_of_quadratic_inequality_l310_310779

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 3} :=
sorry

end solution_set_of_quadratic_inequality_l310_310779


namespace gcd_of_a_and_b_lcm_of_a_and_b_l310_310540

def a : ℕ := 2 * 3 * 7
def b : ℕ := 2 * 3 * 3 * 5

theorem gcd_of_a_and_b : Nat.gcd a b = 6 := by
  sorry

theorem lcm_of_a_and_b : Nat.lcm a b = 630 := by
  sorry

end gcd_of_a_and_b_lcm_of_a_and_b_l310_310540


namespace probability_of_fx_leq_zero_is_3_over_10_l310_310091

noncomputable def fx (x : ℝ) : ℝ := -x + 2

def in_interval (x : ℝ) (a b : ℝ) : Prop := a ≤ x ∧ x ≤ b

def probability_fx_leq_zero : ℚ :=
  let interval_start := -5
  let interval_end := 5
  let fx_leq_zero_start := 2
  let fx_leq_zero_end := 5
  (fx_leq_zero_end - fx_leq_zero_start) / (interval_end - interval_start)

theorem probability_of_fx_leq_zero_is_3_over_10 :
  probability_fx_leq_zero = 3 / 10 :=
sorry

end probability_of_fx_leq_zero_is_3_over_10_l310_310091


namespace undefined_values_l310_310905

theorem undefined_values (a : ℝ) : a = -3 ∨ a = 3 ↔ (a^2 - 9 = 0) := sorry

end undefined_values_l310_310905


namespace mod_2021_2022_2023_2024_eq_zero_mod_7_l310_310212

theorem mod_2021_2022_2023_2024_eq_zero_mod_7 :
  (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end mod_2021_2022_2023_2024_eq_zero_mod_7_l310_310212


namespace arithmetic_sequence_sum_10_l310_310691

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

noncomputable def a_n (a1 d : α) (n : ℕ) : α :=
a1 + (n - 1) • d

def sequence_sum (a1 d : α) (n : ℕ) : α :=
n • a1 + (n • (n - 1) / 2) • d

theorem arithmetic_sequence_sum_10 
  (a1 d : ℤ)
  (h1 : a_n a1 d 2 + a_n a1 d 4 = 4)
  (h2 : a_n a1 d 3 + a_n a1 d 5 = 10) :
  sequence_sum a1 d 10 = 95 :=
by
  sorry

end arithmetic_sequence_sum_10_l310_310691


namespace fraction_sum_l310_310814

theorem fraction_sum (n : ℕ) (a : ℚ) (sum_fraction : a = 1/12) (number_of_fractions : n = 450) : 
  ∀ (f : ℚ), (n * f = a) → (f = 1/5400) :=
by
  intros f H
  sorry

end fraction_sum_l310_310814


namespace ratio_of_discretionary_income_l310_310060

theorem ratio_of_discretionary_income
  (net_monthly_salary : ℝ) 
  (vacation_fund_pct : ℝ) 
  (savings_pct : ℝ) 
  (socializing_pct : ℝ) 
  (gifts_amt : ℝ)
  (D : ℝ) 
  (ratio : ℝ)
  (salary : net_monthly_salary = 3700)
  (vacation_fund : vacation_fund_pct = 0.30)
  (savings : savings_pct = 0.20)
  (socializing : socializing_pct = 0.35)
  (gifts : gifts_amt = 111)
  (discretionary_income : D = gifts_amt / 0.15)
  (net_salary_ratio : ratio = D / net_monthly_salary) :
  ratio = 1 / 5 := sorry

end ratio_of_discretionary_income_l310_310060


namespace combined_value_of_a_and_b_l310_310111

theorem combined_value_of_a_and_b :
  (∃ a b : ℝ,
    0.005 * a = 95 / 100 ∧
    b = 3 * a - 50 ∧
    a + b = 710) :=
sorry

end combined_value_of_a_and_b_l310_310111


namespace comb_product_l310_310526

theorem comb_product :
  (Nat.choose 10 3) * (Nat.choose 8 3) * 2 = 13440 :=
by
  sorry

end comb_product_l310_310526


namespace slope_of_line_m_equals_neg_2_l310_310002

theorem slope_of_line_m_equals_neg_2
  (m : ℝ)
  (h : (3 * m - 6) / (1 + m) = 12) :
  m = -2 :=
sorry

end slope_of_line_m_equals_neg_2_l310_310002


namespace missing_digit_is_4_l310_310977

theorem missing_digit_is_4 (x : ℕ) (hx : 7385 = 7380 + x + 5)
  (hdiv : (7 + 3 + 8 + x + 5) % 9 = 0) : x = 4 :=
by
  sorry

end missing_digit_is_4_l310_310977


namespace no_zero_in_range_l310_310130

noncomputable def g (x : ℝ) : ℤ :=
if x > -3 then ⌈1 / (x + 3)⌉
else if x < -3 then ⌊1 / (x + 3)⌋
else 0 -- This value is arbitrary as g(x) is not defined at x = -3

theorem no_zero_in_range : ¬ ∃ x : ℝ, g x = 0 :=
begin
  unfold g,
  intros h,
  cases h with x hx,
  rw if_neg (by linarith) at hx, -- Excludes the case x = -3
  cases (lt_or_gt_of_ne (ne_of_gt (by linarith.min)))
  case inl => rw if_neg (by linarith) at hx -- Case when x < -3
  case inr => rw if_pos (by linarith) at hx -- Case when x > -3
  -- Both cases imply contradiction as shown in the problem solution
  contradiction,
end

end no_zero_in_range_l310_310130


namespace proof_inequality_l310_310693

noncomputable def problem (a b c d : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a * b * c * d = 1 → a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d

theorem proof_inequality (a b c d : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a * b * c * d = 1) :
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
sorry

end proof_inequality_l310_310693


namespace sum_of_n_values_l310_310014

theorem sum_of_n_values :
  (∑ n in {n : ℕ | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13}, n) = 24 :=
by {sorry}

end sum_of_n_values_l310_310014


namespace intersection_A_complementB_l310_310707

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1, 3, 5, 7}
def complementB := U \ B

theorem intersection_A_complementB :
  A ∩ complementB = {2, 4, 6} := 
by
  sorry

end intersection_A_complementB_l310_310707


namespace rodney_lift_l310_310145

theorem rodney_lift :
  ∃ (Ry : ℕ), 
  (∃ (Re R Ro : ℕ), 
  Re + Ry + R + Ro = 450 ∧
  Ry = 2 * R ∧
  R = Ro + 5 ∧
  Re = 3 * Ro - 20 ∧
  20 ≤ Ry ∧ Ry ≤ 200 ∧
  20 ≤ R ∧ R ≤ 200 ∧
  20 ≤ Ro ∧ Ro ≤ 200 ∧
  20 ≤ Re ∧ Re ≤ 200) ∧
  Ry = 140 :=
by
  sorry

end rodney_lift_l310_310145


namespace quadratic_completing_square_l310_310850

theorem quadratic_completing_square
  (a : ℤ) (b : ℤ) (c : ℤ)
  (h1 : a > 0)
  (h2 : 64 * a^2 * x^2 - 96 * x - 48 = 64 * x^2 - 96 * x - 48)
  (h3 : (a * x + b)^2 = c) :
  a + b + c = 86 :=
sorry

end quadratic_completing_square_l310_310850


namespace exponent_fraction_equals_five_fourths_l310_310019

theorem exponent_fraction_equals_five_fourths :
  (3^2016 + 3^2014) / (3^2016 - 3^2014) = 5 / 4 :=
by
  sorry

end exponent_fraction_equals_five_fourths_l310_310019


namespace FruitKeptForNextWeek_l310_310454

/-- Define the variables and conditions -/
def total_fruit : ℕ := 10
def fruit_eaten : ℕ := 5
def fruit_brought_on_friday : ℕ := 3

/-- Define what we need to prove -/
theorem FruitKeptForNextWeek : 
  ∃ k, total_fruit - fruit_eaten - fruit_brought_on_friday = k ∧ k = 2 :=
by
  sorry

end FruitKeptForNextWeek_l310_310454


namespace a_squared_divisible_by_b_l310_310235

theorem a_squared_divisible_by_b (a b : ℕ) (h1 : a < 1000) (h2 : b > 0) 
    (h3 : ∃ k, a ^ 21 = b ^ 10 * k) : ∃ m, a ^ 2 = b * m := 
by
  sorry

end a_squared_divisible_by_b_l310_310235


namespace range_of_a1_l310_310334

theorem range_of_a1 (a1 : ℝ) :
  (∃ (a2 a3 : ℝ), 
    ((a2 = 2 * a1 - 12) ∨ (a2 = a1 / 2 + 12)) ∧
    ((a3 = 2 * a2 - 12) ∨ (a3 = a2 / 2 + 12)) ) →
  ((a3 > a1) ↔ ((a1 ≤ 12) ∨ (24 ≤ a1))) :=
by
  sorry

end range_of_a1_l310_310334


namespace p_necessary_for_q_l310_310233

def p (x : ℝ) := x ≠ 1
def q (x : ℝ) := x ≥ 2

theorem p_necessary_for_q : ∀ x, q x → p x :=
by
  intro x
  intro hqx
  rw [q] at hqx
  rw [p]
  sorry

end p_necessary_for_q_l310_310233


namespace curve_transformation_l310_310738

theorem curve_transformation :
  (∀ (x y : ℝ), 2 * (5 * x)^2 + 8 * (3 * y)^2 = 1) → (∀ (x y : ℝ), 50 * x^2 + 72 * y^2 = 1) :=
by
  intros h x y
  have h1 : 2 * (5 * x)^2 + 8 * (3 * y)^2 = 1 := h x y
  sorry

end curve_transformation_l310_310738


namespace carla_drank_total_amount_l310_310207

-- Define the conditions
def carla_water : ℕ := 15
def carla_soda := 3 * carla_water - 6
def total_liquid := carla_water + carla_soda

-- State the theorem
theorem carla_drank_total_amount : total_liquid = 54 := by
  sorry

end carla_drank_total_amount_l310_310207


namespace simplify_expression_l310_310063

theorem simplify_expression (x y z : ℝ) :
  3 * (x - (2 * y - 3 * z)) - 2 * ((3 * x - 2 * y) - 4 * z) = -3 * x - 2 * y + 17 * z :=
by
  sorry

end simplify_expression_l310_310063


namespace evaluate_expression_at_x_eq_2_l310_310990

theorem evaluate_expression_at_x_eq_2 : (3 * 2 + 4)^2 - 10 * 2 = 80 := by
  sorry

end evaluate_expression_at_x_eq_2_l310_310990


namespace inequality_proof_l310_310449

theorem inequality_proof (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l310_310449


namespace Joan_video_game_expense_l310_310582

theorem Joan_video_game_expense : 
  let basketball_price := 5.20
  let racing_price := 4.23
  let action_price := 7.12
  let discount_rate := 0.10
  let sales_tax_rate := 0.06
  let discounted_basketball_price := basketball_price * (1 - discount_rate)
  let discounted_racing_price := racing_price * (1 - discount_rate)
  let discounted_action_price := action_price * (1 - discount_rate)
  let total_cost_before_tax := discounted_basketball_price + discounted_racing_price + discounted_action_price
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost := total_cost_before_tax + sales_tax
  total_cost = 15.79 :=
by
  sorry

end Joan_video_game_expense_l310_310582


namespace alice_wins_with_optimal_strategy_l310_310662

theorem alice_wins_with_optimal_strategy :
  (∀ (N : ℕ) (X Y : ℕ), N = 270000 → N = X * Y → gcd X Y ≠ 1 → 
    (∃ (alice : ℕ → ℕ → Prop), ∀ N, ∃ (X Y : ℕ), alice N (X * Y) → gcd X Y ≠ 1) ∧
    (∀ (bob : ℕ → ℕ → ℕ → Prop), ∀ N X Y, bob N X Y → gcd X Y ≠ 1)) →
  (N : ℕ) → N = 270000 → gcd N 1 ≠ 1 :=
by
  sorry

end alice_wins_with_optimal_strategy_l310_310662


namespace find_n_l310_310537

theorem find_n (x y : ℝ) (n : ℝ) (h1 : x / (2 * y) = 3 / n) (h2 : (7 * x + 2 * y) / (x - 2 * y) = 23) : n = 2 := by
  sorry

end find_n_l310_310537


namespace sin_cos_eq_one_l310_310221

theorem sin_cos_eq_one (x : ℝ) (hx0 : 0 ≤ x) (hx2pi : x < 2 * Real.pi) :
  (Real.sin x - Real.cos x = 1) ↔ (x = Real.pi / 2 ∨ x = Real.pi) :=
by
  sorry

end sin_cos_eq_one_l310_310221


namespace Granger_payment_correct_l310_310097

noncomputable def Granger_total_payment : ℝ :=
  let spam_per_can := 3.0
  let peanut_butter_per_jar := 5.0
  let bread_per_loaf := 2.0
  let spam_quantity := 12
  let peanut_butter_quantity := 3
  let bread_quantity := 4
  let spam_dis := 0.1
  let peanut_butter_tax := 0.05
  let spam_cost := spam_quantity * spam_per_can
  let peanut_butter_cost := peanut_butter_quantity * peanut_butter_per_jar
  let bread_cost := bread_quantity * bread_per_loaf
  let spam_discount := spam_dis * spam_cost
  let peanut_butter_tax_amount := peanut_butter_tax * peanut_butter_cost
  let spam_final_cost := spam_cost - spam_discount
  let peanut_butter_final_cost := peanut_butter_cost + peanut_butter_tax_amount
  let total := spam_final_cost + peanut_butter_final_cost + bread_cost
  total

theorem Granger_payment_correct :
  Granger_total_payment = 56.15 :=
by
  sorry

end Granger_payment_correct_l310_310097


namespace valid_sentence_count_is_208_l310_310337

def four_words := ["splargh", "glumph", "amr", "flark"]

def valid_sentence (sentence : List String) : Prop :=
  ¬(sentence.contains "glumph amr")

def count_valid_sentences : Nat :=
  let total_sentences := 4^4
  let invalid_sentences := 3 * 4 * 4
  total_sentences - invalid_sentences

theorem valid_sentence_count_is_208 :
  count_valid_sentences = 208 := by
  sorry

end valid_sentence_count_is_208_l310_310337


namespace hiker_total_distance_l310_310834

theorem hiker_total_distance :
  let day1_distance := 18
  let day1_speed := 3
  let day2_speed := day1_speed + 1
  let day1_time := day1_distance / day1_speed
  let day2_time := day1_time - 1
  let day2_distance := day2_speed * day2_time
  let day3_speed := 5
  let day3_time := 3
  let day3_distance := day3_speed * day3_time
  let total_distance := day1_distance + day2_distance + day3_distance
  total_distance = 53 :=
by
  sorry

end hiker_total_distance_l310_310834


namespace perfect_squares_in_range_100_400_l310_310103

theorem perfect_squares_in_range_100_400 : ∃ n : ℕ, (∀ m, 100 ≤ m^2 → m^2 ≤ 400 → m^2 = (m - 10 + 1)^2) ∧ n = 9 := 
by
  sorry

end perfect_squares_in_range_100_400_l310_310103


namespace john_initial_clean_jerk_weight_l310_310742

def initial_snatch_weight : ℝ := 50
def increase_rate : ℝ := 1.8
def total_new_lifting_capacity : ℝ := 250

theorem john_initial_clean_jerk_weight :
  ∃ (C : ℝ), 2 * C + (increase_rate * initial_snatch_weight) = total_new_lifting_capacity ∧ C = 80 := by
  sorry

end john_initial_clean_jerk_weight_l310_310742


namespace intersection_of_sets_l310_310917

def A : Set ℕ := {1, 2, 5}
def B : Set ℕ := {1, 3, 5}

theorem intersection_of_sets : A ∩ B = {1, 5} :=
by
  sorry

end intersection_of_sets_l310_310917


namespace sqrt_floor_squared_l310_310871

theorem sqrt_floor_squared (h1 : 7^2 = 49) (h2 : 8^2 = 64) (h3 : 7 < Real.sqrt 50) (h4 : Real.sqrt 50 < 8) : (Int.floor (Real.sqrt 50))^2 = 49 :=
by
  sorry

end sqrt_floor_squared_l310_310871


namespace inradius_of_triangle_l310_310980

/-- Given conditions for the triangle -/
def perimeter : ℝ := 32
def area : ℝ := 40

/-- The theorem to prove the inradius of the triangle -/
theorem inradius_of_triangle (h : area = (r * perimeter) / 2) : r = 2.5 :=
by
  sorry

end inradius_of_triangle_l310_310980


namespace keep_oranges_per_day_l310_310934

def total_oranges_harvested (sacks_per_day : ℕ) (oranges_per_sack : ℕ) : ℕ :=
  sacks_per_day * oranges_per_sack

def oranges_discarded (discarded_sacks : ℕ) (oranges_per_sack : ℕ) : ℕ :=
  discarded_sacks * oranges_per_sack

def oranges_kept_per_day (total_oranges : ℕ) (discarded_oranges : ℕ) : ℕ :=
  total_oranges - discarded_oranges

theorem keep_oranges_per_day 
  (sacks_per_day : ℕ)
  (oranges_per_sack : ℕ)
  (discarded_sacks : ℕ)
  (h1 : sacks_per_day = 76)
  (h2 : oranges_per_sack = 50)
  (h3 : discarded_sacks = 64) :
  oranges_kept_per_day (total_oranges_harvested sacks_per_day oranges_per_sack) 
  (oranges_discarded discarded_sacks oranges_per_sack) = 600 :=
by
  sorry

end keep_oranges_per_day_l310_310934


namespace hexagon_overlapping_area_l310_310039

theorem hexagon_overlapping_area:
  ∀ (s α : ℝ),
  s = 1 →
  0 < α ∧ α < π / 3 →
  real.cos α = real.sqrt 3 / 2 →
  let area := (3 * real.sqrt 3) / 2
  in 
  (s ^ 2 * area / 3) = real.sqrt 3 / 2 :=
by 
  intros s α h_s h_α h_cos g 
  let area := (3 * real.sqrt 3) / 2
  sorry

end hexagon_overlapping_area_l310_310039


namespace like_terms_constants_l310_310176

theorem like_terms_constants :
  ∀ (a b : ℚ), a = 1/2 → b = -1/3 → (a = 1/2 ∧ b = -1/3) → a + b = 1/2 + -1/3 :=
by
  intros a b ha hb h
  sorry

end like_terms_constants_l310_310176


namespace domain_f_a_5_abs_inequality_ab_l310_310772

-- Definition for the domain of f(x) when a=5
def domain_of_f_a_5 (x : ℝ) : Prop := |x + 1| + |x + 2| - 5 ≥ 0

-- The theorem to find the domain A of the function f(x) when a=5.
theorem domain_f_a_5 (x : ℝ) : domain_of_f_a_5 x ↔ (x ≤ -4 ∨ x ≥ 1) :=
by
  sorry

-- Theorem to prove the inequality for a, b ∈ (-1, 1)
theorem abs_inequality_ab (a b : ℝ) (ha : -1 < a ∧ a < 1) (hb : -1 < b ∧ b < 1) :
  |a + b| / 2 < |1 + a * b / 4| :=
by
  sorry

end domain_f_a_5_abs_inequality_ab_l310_310772


namespace find_x_solutions_l310_310999

theorem find_x_solutions :
  ∀ {x : ℝ}, (x = (1/x) + (-x)^2 + 3) → (x = -1 ∨ x = 1) :=
by
  sorry

end find_x_solutions_l310_310999


namespace cost_of_purchase_l310_310313

theorem cost_of_purchase (x : ℝ) (T_shirt boots shin_guards : ℝ) 
  (h1 : x + T_shirt = 2 * x)
  (h2 : x + boots = 5 * x)
  (h3 : x + shin_guards = 3 * x) :
  x + T_shirt + boots + shin_guards = 8 * x :=
begin
  sorry
end

end cost_of_purchase_l310_310313


namespace area_of_rectangle_l310_310651

--- Define the problem's conditions
def square_area : ℕ := 36
def rectangle_width := (square_side : ℕ) (h : square_area = square_side * square_side) : ℕ := square_side
def rectangle_length := (width : ℕ) : ℕ := 3 * width

--- State the theorem using the defined conditions
theorem area_of_rectangle (square_side : ℕ) 
  (h1 : square_area = square_side * square_side)
  (width := rectangle_width square_side h1)
  (length := rectangle_length width) :
  width * length = 108 := by
    sorry

end area_of_rectangle_l310_310651


namespace triangle_cos_half_angle_l310_310274

noncomputable theory

-- Assume \( \triangle ABC \) with \( A, B, C \) angles such that \( A = 2x \), \( B = 3x \), \( C = 4x \)
variables {A B C a b c : ℝ}

-- Assume the angles in the conditions
axiom angle_ratios (x : ℝ) : A = 2 * x ∧ B = 3 * x ∧ C = 4 * x ∧ A + B + C = 180

-- Assume \( \cos \frac{A}{2} \) identity
axiom cos_half_angle_formula (A : ℝ):
  cos (A / 2) = sqrt ((1 + cos A) / 2)

-- Assume relationship between sides using Law of Sines
axiom law_of_sines :
  ∀ {a b c A B C : ℝ}, (a / sin A = b / sin B) ∧ (b / sin B = c / sin C)

-- Assume relationships between side lengths using addition
axiom sine_values (a b c : ℝ) :
  sin 60 = sqrt (3) / 2 ∧ sin 40 = real.sin (40 * real.pi / 180) ∧ sin 80 = real.sin (80 * real.pi / 180)

theorem triangle_cos_half_angle (x a b c : ℝ) (h1 : A = 2 * x)
  (h2 : B = 3 * x) (h3 : C = 4 * x) (h4: A + B + C = 180) :
  cos (A/2) = (a + c) / (2 * b) :=
by
  sorry

end triangle_cos_half_angle_l310_310274


namespace greatest_possible_value_of_x_l310_310768

theorem greatest_possible_value_of_x (x : ℕ) (h₁ : x % 4 = 0) (h₂ : x > 0) (h₃ : x^3 < 8000) :
  x ≤ 16 := by
  apply sorry

end greatest_possible_value_of_x_l310_310768


namespace maximum_area_of_triangle_l310_310085

theorem maximum_area_of_triangle :
  ∃ (b c : ℝ), (a = 2) ∧ (A = 60 * Real.pi / 180) ∧
  (∀ S : ℝ, S = (1/2) * b * c * Real.sin A → S ≤ Real.sqrt 3) :=
by sorry

end maximum_area_of_triangle_l310_310085


namespace choir_members_max_l310_310030

theorem choir_members_max (x r m : ℕ) 
  (h1 : r * x + 3 = m)
  (h2 : (r - 3) * (x + 2) = m) 
  (h3 : m < 150) : 
  m = 759 :=
sorry

end choir_members_max_l310_310030


namespace quadratic_intersects_x_axis_if_and_only_if_k_le_four_l310_310093

-- Define the quadratic function
def quadratic_function (k x : ℝ) : ℝ :=
  (k - 3) * x^2 + 2 * x + 1

-- Theorem stating the relationship between the function intersecting the x-axis and k ≤ 4
theorem quadratic_intersects_x_axis_if_and_only_if_k_le_four
  (k : ℝ) :
  (∃ x : ℝ, quadratic_function k x = 0) ↔ k ≤ 4 :=
sorry

end quadratic_intersects_x_axis_if_and_only_if_k_le_four_l310_310093


namespace sqrt_floor_squared_eq_49_l310_310882

theorem sqrt_floor_squared_eq_49 : (⌊real.sqrt 50⌋)^2 = 49 :=
by sorry

end sqrt_floor_squared_eq_49_l310_310882


namespace prob_at_least_one_palindrome_correct_l310_310429

-- Define a function to represent the probability calculation.
def probability_at_least_one_palindrome : ℚ :=
  let prob_digit_palindrome : ℚ := 1 / 100
  let prob_letter_palindrome : ℚ := 1 / 676
  let prob_both_palindromes : ℚ := (1 / 100) * (1 / 676)
  (prob_digit_palindrome + prob_letter_palindrome - prob_both_palindromes)

-- The theorem we are stating based on the given problem and solution:
theorem prob_at_least_one_palindrome_correct : probability_at_least_one_palindrome = 427 / 2704 :=
by
  -- We assume this step for now as we are just stating the theorem
  sorry

end prob_at_least_one_palindrome_correct_l310_310429


namespace min_liars_needed_l310_310799

namespace Problem

/-
  We introduce our problem using Lean definitions and statements.
-/

def students_sitting_at_table : ℕ := 29

structure Student :=
(is_liar : Bool)

def student_truth_statement (s: ℕ) (next1: Student) (next2: Student) : Bool :=
  if s then (next1.is_liar && !next2.is_liar) || (!next1.is_liar && next2.is_liar)
  else next1.is_liar && next2.is_liar

def minimum_liars_class (sitting_students: ℕ) : ℕ :=
  sitting_students / 3 + (if sitting_students % 3 = 0 then 0 else 1)

theorem min_liars_needed (students : List Student) (H_total : students.length = students_sitting_at_table)
  (H_truth_teller_claim : ∀ (i : ℕ), i < students.length → student_truth_statement i (students.get ⟨(i+1) % students.length, sorry⟩) (students.get ⟨(i+2) % students.length, sorry⟩)) :
  (students.count (λ s => s.is_liar)) ≥ minimum_liars_class students_sitting_at_table :=
sorry

end Problem

end min_liars_needed_l310_310799


namespace loss_of_30_yuan_is_minus_30_yuan_l310_310717

def profit (p : ℤ) : Prop := p = 20
def loss (l : ℤ) : Prop := l = -30

theorem loss_of_30_yuan_is_minus_30_yuan (p : ℤ) (l : ℤ) (h : profit p) : loss l :=
by
  sorry

end loss_of_30_yuan_is_minus_30_yuan_l310_310717


namespace tetrahedron_volume_from_cube_l310_310997

theorem tetrahedron_volume_from_cube {s : ℝ} (h : s = 8) :
  let cube_volume := s^3
  let smaller_tetrahedron_volume := (1/3) * (1/2) * s * s * s
  let total_smaller_tetrahedron_volume := 4 * smaller_tetrahedron_volume
  let tetrahedron_volume := cube_volume - total_smaller_tetrahedron_volume
  tetrahedron_volume = 170.6666 :=
by
  sorry

end tetrahedron_volume_from_cube_l310_310997


namespace jaime_can_buy_five_apples_l310_310732

theorem jaime_can_buy_five_apples :
  ∀ (L M : ℝ),
  (L = M / 2 + 1 / 2) →
  (M / 3 = L / 4 + 1 / 2) →
  (15 / M = 5) :=
by
  intros L M h1 h2
  sorry

end jaime_can_buy_five_apples_l310_310732


namespace proof_problem_l310_310696

noncomputable def a {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def b {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def c {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def d {α : Type*} [LinearOrderedField α] : α := sorry

theorem proof_problem (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
(hprod : a * b * c * d = 1) : 
a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
sorry

end proof_problem_l310_310696


namespace knights_in_company_l310_310730

theorem knights_in_company :
  ∃ k : ℕ, (k = 0 ∨ k = 6) ∧ k ≤ 39 ∧
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 39) →
    (∃ i : ℕ, (1 ≤ i ∧ i ≤ 39) ∧ n * k = 1 + (i - 1) * k) →
    ∃ i : ℕ, ∃ nk : ℕ, (nk = i * k ∧ nk ≤ 39 ∧ (nk ∣ k → i = 1 + (i - 1))) :=
by
  sorry

end knights_in_company_l310_310730


namespace extremum_f_at_1_max_t_l310_310586

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * m * x^2 - 2 * x + real.log(x + 1)

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m x - real.log(x + 1) + x^3

theorem extremum_f_at_1 (m : ℝ) : 
  ((deriv (f m)) 1 = 0) ↔ (m = 3/2) := 
by sorry

theorem max_t (m : ℝ) (hm : m ∈ Icc (-4: ℝ) (-1: ℝ)) : 
  ∃ t > 1, ∀ x ∈ Icc (1: ℝ) t, g m x ≤ g m 1 ∧ t ≤ (1 + real.sqrt 13) / 2 := 
by sorry

end extremum_f_at_1_max_t_l310_310586


namespace odd_function_condition_l310_310923

-- Definitions for real numbers and absolute value function
def f (x a b : ℝ) : ℝ := (x + a) * |x + b|

-- Theorem statement
theorem odd_function_condition (a b : ℝ) (h1 : ∀ x : ℝ, f x a b = (x + a) * |x + b|) :
  (∀ x : ℝ, f x a b = -f (-x) a b) ↔ (a = 0 ∨ b = 0) :=
by
  sorry

end odd_function_condition_l310_310923


namespace how_many_eyes_do_I_see_l310_310417

def boys : ℕ := 23
def eyes_per_boy : ℕ := 2
def total_eyes : ℕ := boys * eyes_per_boy

theorem how_many_eyes_do_I_see : total_eyes = 46 := by
  sorry

end how_many_eyes_do_I_see_l310_310417


namespace no_arithmetic_progression_40_terms_l310_310218

noncomputable def is_arith_prog (f : ℕ → ℕ) (a : ℕ) (b : ℕ) : Prop :=
∀ n : ℕ, ∃ k : ℕ, f n = a + n * b

noncomputable def in_form_2m_3n (x : ℕ) : Prop :=
∃ m n : ℕ, x = 2^m + 3^n

theorem no_arithmetic_progression_40_terms :
  ¬ (∃ (a b : ℕ), ∀ n, n < 40 → in_form_2m_3n (a + n * b)) :=
sorry

end no_arithmetic_progression_40_terms_l310_310218


namespace acute_triangle_l310_310348

theorem acute_triangle (r R : ℝ) (h : R < r * (Real.sqrt 2 + 1)) : 
  ∃ (α β γ : ℝ), α + β + γ = π ∧ (0 < α) ∧ (0 < β) ∧ (0 < γ) ∧ (α < π / 2) ∧ (β < π / 2) ∧ (γ < π / 2) := 
sorry

end acute_triangle_l310_310348


namespace max_intersecting_chords_through_A1_l310_310509

theorem max_intersecting_chords_through_A1 
  (n : ℕ) (h_n : n = 2017) 
  (A : Fin n → α) 
  (line_through_A1 : α) 
  (no_other_intersection : ∀ i : Fin n, i ≠ 0 → A i ≠ line_through_A1) :
  ∃ k : ℕ, k * (2016 - k) + 2016 = 1018080 := 
sorry

end max_intersecting_chords_through_A1_l310_310509


namespace evaluate_expression_l310_310392

theorem evaluate_expression : ((3 ^ 2) ^ 3) - ((2 ^ 3) ^ 2) = 665 := by
  sorry

end evaluate_expression_l310_310392


namespace unique_solution_l310_310677

theorem unique_solution (x : ℝ) : (3 : ℝ)^x + (4 : ℝ)^x + (5 : ℝ)^x = (6 : ℝ)^x ↔ x = 3 := by
  sorry

end unique_solution_l310_310677


namespace domain_range_sum_l310_310339

def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem domain_range_sum (m n : ℝ) (hmn : ∀ x, m ≤ x ∧ x ≤ n → (f x = 3 * x)) : m + n = -1 :=
by
  sorry

end domain_range_sum_l310_310339


namespace ratio_frank_to_others_l310_310538

theorem ratio_frank_to_others:
  (Betty_oranges : ℕ) (Bill_oranges : ℕ) (total_oranges_picked_by_philip : ℕ) :
  Betty_oranges = 15 → Bill_oranges = 12 →
  (exists m : ℕ, total_oranges_picked_by_philip = 270 * m ∧ 270 * m / 27 = 3) →
  (total_oranges_picked_by_philip / 27 / (Betty_oranges + Bill_oranges / 27) = 3) :=
by
  intros Betty_oranges Bill_oranges total_oranges_picked_by_philip
  assume hB : Betty_oranges = 15,
  assume hBi : Bill_oranges = 12,
  assume hE : ∃ m : ℕ, total_oranges_picked_by_philip = 270 * m ∧ 270 * m / 27 = 3,
  sorry

end ratio_frank_to_others_l310_310538


namespace area_of_rectangle_is_108_l310_310656

theorem area_of_rectangle_is_108 (s w l : ℕ) (h₁ : s * s = 36) (h₂ : w = s) (h₃ : l = 3 * w) : w * l = 108 :=
by
  -- This is a placeholder for a detailed proof.
  sorry

end area_of_rectangle_is_108_l310_310656


namespace max_area_rectangular_playground_l310_310491

theorem max_area_rectangular_playground (l w : ℝ) 
  (h_perimeter : 2 * l + 2 * w = 360) 
  (h_length : l ≥ 90) 
  (h_width : w ≥ 50) : 
  (l * w) ≤ 8100 :=
by
  sorry

end max_area_rectangular_playground_l310_310491


namespace stratified_sampling_correct_l310_310841

-- Define the conditions
def num_freshmen : ℕ := 900
def num_sophomores : ℕ := 1200
def num_seniors : ℕ := 600
def total_sample_size : ℕ := 135
def total_students := num_freshmen + num_sophomores + num_seniors

-- Proportions
def proportion_freshmen := (num_freshmen : ℚ) / total_students
def proportion_sophomores := (num_sophomores : ℚ) / total_students
def proportion_seniors := (num_seniors : ℚ) / total_students

-- Expected samples count
def expected_freshmen_samples := (total_sample_size : ℚ) * proportion_freshmen
def expected_sophomores_samples := (total_sample_size : ℚ) * proportion_sophomores
def expected_seniors_samples := (total_sample_size : ℚ) * proportion_seniors

-- Statement to be proven
theorem stratified_sampling_correct :
  expected_freshmen_samples = (45 : ℚ) ∧
  expected_sophomores_samples = (60 : ℚ) ∧
  expected_seniors_samples = (30 : ℚ) := by
  -- Provide the necessary proof or calculation
  sorry

end stratified_sampling_correct_l310_310841


namespace find_a_squared_plus_b_squared_l310_310522

theorem find_a_squared_plus_b_squared (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) : a^2 + b^2 = 7 := 
by
  sorry

end find_a_squared_plus_b_squared_l310_310522


namespace white_ball_probability_l310_310263

noncomputable def prob_white_ball : ℚ :=
  let initial_combinations := [{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}] in
  let add_white_ball (c : set ℕ) := c.insert 1 in
  let prob := λ c, ((add_white_ball c).filter (λ x, x = 1)).card.toRat / (add_white_ball c).card.toRat in
  (initial_combinations.map prob).sum / initial_combinations.length

theorem white_ball_probability : prob_white_ball = 5 / 8 :=
  sorry

end white_ball_probability_l310_310263


namespace blankets_first_day_l310_310539

-- Definition of the conditions
def num_people := 15
def blankets_day_three := 22
def total_blankets := 142

-- The problem statement
theorem blankets_first_day (B : ℕ) : 
  (num_people * B) + (3 * (num_people * B)) + blankets_day_three = total_blankets → 
  B = 2 :=
by sorry

end blankets_first_day_l310_310539


namespace fraction_eq_zero_iff_x_eq_2_l310_310424

theorem fraction_eq_zero_iff_x_eq_2 (x : ℝ) : (x - 2) / (x + 2) = 0 ↔ x = 2 := by sorry

end fraction_eq_zero_iff_x_eq_2_l310_310424


namespace initial_profit_price_reduction_for_target_profit_l310_310186

-- Define given conditions
def purchase_price : ℝ := 280
def initial_selling_price : ℝ := 360
def items_sold_per_month : ℕ := 60
def target_profit : ℝ := 7200
def increment_per_reduced_yuan : ℕ := 5

-- Problem 1: Prove the initial profit per month before the price reduction
theorem initial_profit : 
  items_sold_per_month * (initial_selling_price - purchase_price) = 4800 := by
sorry

-- Problem 2: Prove that reducing the price by 60 yuan achieves the target profit
theorem price_reduction_for_target_profit : 
  ∃ x : ℝ, 
    ((initial_selling_price - x) - purchase_price) * (items_sold_per_month + (increment_per_reduced_yuan * x)) = target_profit ∧
    x = 60 := by
sorry

end initial_profit_price_reduction_for_target_profit_l310_310186


namespace expr_value_l310_310924

variable (x y m n a : ℝ)
variable (hxy : x = -y) (hmn : m * n = 1) (ha : |a| = 3)

theorem expr_value : (a / (m * n) + 2018 * (x + y)) = a := sorry

end expr_value_l310_310924


namespace initial_number_l310_310620

theorem initial_number (N : ℤ) 
  (h : (N + 3) % 24 = 0) : N = 21 := 
sorry

end initial_number_l310_310620


namespace sum_of_ns_l310_310015

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem sum_of_ns :
  ∑ n in {n : ℕ | binomial 25 n + binomial 25 12 = binomial 26 13} , n = 25 :=
by
  sorry

end sum_of_ns_l310_310015


namespace sum_fourth_powers_eq_t_l310_310989

theorem sum_fourth_powers_eq_t (a b t : ℝ) (h1 : a + b = t) (h2 : a^2 + b^2 = t) (h3 : a^3 + b^3 = t) : 
  a^4 + b^4 = t := 
by
  sorry

end sum_fourth_powers_eq_t_l310_310989


namespace second_and_third_shooters_cannot_win_or_lose_simultaneously_l310_310007

-- Define the conditions C1, C2, and C3
variables (C1 C2 C3 : Prop)

-- The first shooter bets that at least one of the second or third shooters will miss
def first_shooter_bet : Prop := ¬ (C2 ∧ C3)

-- The second shooter bets that if the first shooter hits, then at least one of the remaining shooters will miss
def second_shooter_bet : Prop := C1 → ¬ (C2 ∧ C3)

-- The third shooter bets that all three will hit the target on the first attempt
def third_shooter_bet : Prop := C1 ∧ C2 ∧ C3

-- Prove that it is impossible for both the second and third shooters to either win or lose their bets concurrently
theorem second_and_third_shooters_cannot_win_or_lose_simultaneously :
  ¬ ((second_shooter_bet C1 C2 C3 ∧ third_shooter_bet C1 C2 C3) ∨ (¬ second_shooter_bet C1 C2 C3 ∧ ¬ third_shooter_bet C1 C2 C3)) :=
by
  sorry

end second_and_third_shooters_cannot_win_or_lose_simultaneously_l310_310007


namespace f_2009_equals_4_l310_310343

open Real

def f (x α β : ℝ) : ℝ := sin (π * x + α) + cos (π * x + β) + 3

theorem f_2009_equals_4 (α β : ℝ) (h : f 2008 α β = 2) : f 2009 α β = 4 :=
  sorry

end f_2009_equals_4_l310_310343


namespace minimize_maximum_absolute_value_expression_l310_310068

theorem minimize_maximum_absolute_value_expression : 
  (∀ y : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2) →
  ∃ y : ℝ, (y = 2) ∧ (min_value = 0) :=
sorry -- Proof goes here

end minimize_maximum_absolute_value_expression_l310_310068


namespace hyperbola_focus_distance_l310_310918

theorem hyperbola_focus_distance :
  let F := (Real.sqrt 6, 0)
  let asymptote := λ (x : ℝ), -x
  let distance (p : ℝ × ℝ) (l : ℝ → ℝ) := 
    Real.abs (p.1 + p.2) / Real.sqrt 2
  distance F asymptote = Real.sqrt 3 :=
by
  sorry

end hyperbola_focus_distance_l310_310918


namespace infinite_non_prime_seq_l310_310148

-- Let's state the theorem in Lean
theorem infinite_non_prime_seq (k : ℕ) : 
  ∃ᶠ n in at_top, ∀ i : ℕ, (1 ≤ i ∧ i ≤ k) → ¬ Nat.Prime (n + i) := 
sorry

end infinite_non_prime_seq_l310_310148


namespace trajectory_equation_of_P_l310_310933

variable {x y : ℝ}
variable (A B P : ℝ × ℝ)

def in_line_through (a b : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  let k := (p.2 - a.2) / (p.1 - a.1)
  (b.2 - a.2) / (b.1 - a.1) = k

theorem trajectory_equation_of_P
  (hA : A = (-1, 0)) (hB : B = (1, 0)) (hP : in_line_through A B P)
  (slope_product : (P.2 / (P.1 + 1)) * (P.2 / (P.1 - 1)) = -1) :
  P.1 ^ 2 + P.2 ^ 2 = 1 ∧ P.1 ≠ 1 ∧ P.1 ≠ -1 := 
sorry

end trajectory_equation_of_P_l310_310933


namespace find_x_plus_y_l310_310090

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2010) (h2 : x + 2010 * Real.cos y + 3 * Real.sin y = 2005) (h3 : 0 ≤ y ∧ y ≤ Real.pi) : 
  x + y = 2009 + Real.pi / 2 := 
sorry

end find_x_plus_y_l310_310090


namespace find_diameter_endpoint_l310_310731

def circle_center : ℝ × ℝ := (4, 1)
def diameter_endpoint_1 : ℝ × ℝ := (1, 5)

theorem find_diameter_endpoint :
  let (h, k) := circle_center
  let (x1, y1) := diameter_endpoint_1
  (2 * h - x1, 2 * k - y1) = (7, -3) :=
by
  let (h, k) := circle_center
  let (x1, y1) := diameter_endpoint_1
  sorry

end find_diameter_endpoint_l310_310731


namespace find_ab_value_l310_310773

-- Definitions from the conditions
def ellipse_eq (a b : ℝ) : Prop := b^2 - a^2 = 25
def hyperbola_eq (a b : ℝ) : Prop := a^2 + b^2 = 49

-- Main theorem statement
theorem find_ab_value {a b : ℝ} (h_ellipse : ellipse_eq a b) (h_hyperbola : hyperbola_eq a b) : 
  |a * b| = 2 * Real.sqrt 111 :=
by
  -- Proof goes here
  sorry

end find_ab_value_l310_310773


namespace min_troublemakers_29_l310_310807

noncomputable def min_troublemakers (n : ℕ) : ℕ :=
sorry

theorem min_troublemakers_29 : min_troublemakers 29 = 10 := 
sorry

end min_troublemakers_29_l310_310807


namespace not_possible_f_g_l310_310275

theorem not_possible_f_g (f g : ℝ → ℝ) :
  ¬(∀ x y : ℝ, 1 + x^2016 * y^2016 = f(x) * g(y)) :=
by
  sorry

end not_possible_f_g_l310_310275


namespace setB_is_empty_l310_310022

noncomputable def setB := {x : ℝ | x^2 + 1 = 0}

theorem setB_is_empty : setB = ∅ :=
by
  sorry

end setB_is_empty_l310_310022


namespace sum_of_integer_values_l310_310017

def binom_sum_condition (n : ℕ) : Prop :=
  nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13

theorem sum_of_integer_values (h : ∀ n, binom_sum_condition n → n = 12 ∨ n = 13) : 
  12 + 13 = 25 :=
by
  sorry

end sum_of_integer_values_l310_310017


namespace union_complement_eq_l310_310931

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}
def complement (U A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

theorem union_complement_eq :
  (complement U A ∪ B) = {2, 3, 4} :=
by
  sorry

end union_complement_eq_l310_310931


namespace boys_contributions_l310_310353

theorem boys_contributions (x y z : ℝ) (h1 : z = x + 6.4) (h2 : (1 / 2) * x = (1 / 3) * y) (h3 : (1 / 2) * x = (1 / 4) * z) :
  x = 6.4 ∧ y = 9.6 ∧ z = 12.8 :=
by
  -- This is where the proof would go
  sorry

end boys_contributions_l310_310353


namespace min_troublemakers_l310_310791

theorem min_troublemakers (n : ℕ) (students : ℕ → Prop) 
  (h : n = 29)
  (condition1 : ∀ i, students i → (students ((i - 1) % n) ↔ ¬ students ((i + 1) % n)))
  (condition2 : ∀ i, ¬ students i → (students ((i - 1) % n) ∧ students ((i + 1) % n)))
  : ∃ L : ℕ, (L ≤ 29 ∧ L ≥ 10) :=
by sorry

end min_troublemakers_l310_310791


namespace area_of_rectangle_l310_310650

--- Define the problem's conditions
def square_area : ℕ := 36
def rectangle_width := (square_side : ℕ) (h : square_area = square_side * square_side) : ℕ := square_side
def rectangle_length := (width : ℕ) : ℕ := 3 * width

--- State the theorem using the defined conditions
theorem area_of_rectangle (square_side : ℕ) 
  (h1 : square_area = square_side * square_side)
  (width := rectangle_width square_side h1)
  (length := rectangle_length width) :
  width * length = 108 := by
    sorry

end area_of_rectangle_l310_310650


namespace sum_of_cube_faces_l310_310765

theorem sum_of_cube_faces (a d b e c f : ℕ) (h1: a > 0) (h2: d > 0) (h3: b > 0) (h4: e > 0) (h5: c > 0) (h6: f > 0)
(h7 : a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 1491) :
  a + d + b + e + c + f = 41 := 
sorry

end sum_of_cube_faces_l310_310765


namespace total_cartons_accepted_l310_310442

theorem total_cartons_accepted (total_cartons : ℕ) (customers : ℕ) (damaged_cartons_per_customer : ℕ) (initial_cartons_per_customer accepted_cartons_per_customer total_accepted_cartons : ℕ) :
    total_cartons = 400 →
    customers = 4 →
    damaged_cartons_per_customer = 60 →
    initial_cartons_per_customer = total_cartons / customers →
    accepted_cartons_per_customer = initial_cartons_per_customer - damaged_cartons_per_customer →
    total_accepted_cartons = accepted_cartons_per_customer * customers →
    total_accepted_cartons = 160 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_cartons_accepted_l310_310442


namespace rectangle_area_from_square_l310_310647

theorem rectangle_area_from_square {s a : ℝ} (h1 : s^2 = 36) (h2 : a = 3 * s) :
    s * a = 108 :=
by
  -- The proof goes here
  sorry

end rectangle_area_from_square_l310_310647


namespace joe_saves_6000_l310_310741

-- Definitions based on the conditions
def flight_cost : ℕ := 1200
def hotel_cost : ℕ := 800
def food_cost : ℕ := 3000
def money_left : ℕ := 1000

-- Total expenses
def total_expenses : ℕ := flight_cost + hotel_cost + food_cost

-- Total savings
def total_savings : ℕ := total_expenses + money_left

-- The proof statement
theorem joe_saves_6000 : total_savings = 6000 := by
  -- Proof goes here
  sorry

end joe_saves_6000_l310_310741


namespace four_thirds_eq_36_l310_310911

theorem four_thirds_eq_36 (x : ℝ) (h : (4 / 3) * x = 36) : x = 27 := by
  sorry

end four_thirds_eq_36_l310_310911


namespace ratio_of_x_and_y_l310_310709

theorem ratio_of_x_and_y {x y a b : ℝ} (h1 : (2 * a - x) / (3 * b - y) = 3) (h2 : a / b = 4.5) : x / y = 3 :=
sorry

end ratio_of_x_and_y_l310_310709


namespace gcd_f_50_51_l310_310284

-- Define f(x)
def f (x : ℤ) : ℤ := x^3 - x^2 + 2 * x + 2000

-- State the problem: Prove gcd(f(50), f(51)) = 8
theorem gcd_f_50_51 : Int.gcd (f 50) (f 51) = 8 := by
  sorry

end gcd_f_50_51_l310_310284


namespace find_n_l310_310994

theorem find_n (n : ℕ) : 
  Nat.lcm n 12 = 48 ∧ Nat.gcd n 12 = 8 → n = 32 := 
by 
  sorry

end find_n_l310_310994


namespace find_n_tan_l310_310893

theorem find_n_tan (n : ℤ) (hn : -90 < n ∧ n < 90) (htan : Real.tan (n * Real.pi / 180) = Real.tan (312 * Real.pi / 180)) : 
  n = -48 := 
sorry

end find_n_tan_l310_310893


namespace crayons_per_box_l310_310749

-- Define the conditions
def crayons : ℕ := 80
def boxes : ℕ := 10

-- State the proof problem
theorem crayons_per_box : (crayons / boxes) = 8 := by
  sorry

end crayons_per_box_l310_310749


namespace rectangle_right_triangle_max_area_and_hypotenuse_l310_310476

theorem rectangle_right_triangle_max_area_and_hypotenuse (x y h : ℝ) (h_triangle : h^2 = x^2 + y^2) (h_perimeter : 2 * (x + y) = 60) :
  (x * y ≤ 225) ∧ (x = 15) ∧ (y = 15) ∧ (h = 15 * Real.sqrt 2) :=
by
  sorry

end rectangle_right_triangle_max_area_and_hypotenuse_l310_310476


namespace sequence_term_sum_max_value_sum_equality_l310_310952

noncomputable def a (n : ℕ) : ℝ := -2 * n + 6

def S (n : ℕ) : ℝ := -n^2 + 5 * n

theorem sequence_term (n : ℕ) : ∀ n, a n = 4 + (n - 1) * (-2) :=
by sorry

theorem sum_max_value (n : ℕ) : ∃ n, S n = 6 :=
by sorry

theorem sum_equality : S 2 = 6 ∧ S 3 = 6 :=
by sorry

end sequence_term_sum_max_value_sum_equality_l310_310952


namespace bowling_ball_weight_l310_310592

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 9 * b = 6 * c) 
  (h2 : 4 * c = 120) : 
  b = 20 :=
by 
  sorry

end bowling_ball_weight_l310_310592


namespace distance_between_parallel_lines_l310_310490

theorem distance_between_parallel_lines (r : ℝ) (d : ℝ) (h1 : ∃ (C D E F : ℝ), CD = 38 ∧ EF = 38 ∧ DE = 34) :
  d = 6 :=
begin
  sorry
end

end distance_between_parallel_lines_l310_310490


namespace factorial_difference_l310_310858

theorem factorial_difference :
  10! - 9! = 3265920 :=
by
  sorry

end factorial_difference_l310_310858


namespace tangent_lines_to_two_circles_l310_310935

noncomputable def center_radius (a b c d e : ℝ) : (ℝ × ℝ) × ℝ :=
let x₀ := -a/2,
    y₀ := -d/2,
    r := real.sqrt (x₀^2 + y₀^2 - e)
in ((x₀, y₀), r)

theorem tangent_lines_to_two_circles :
  let C1 := (center_radius 2 2 4 (-4) 7).1,
      r1 := (center_radius 2 2 4 (-4) 7).2,
      C2 := (center_radius 2 2 (-4) (-10) 13).1,
      r2 := (center_radius 2 2 (-4) (-10) 13).2,
      dist_centers := real.sqrt ((C1.1 - C2.1)^2 + (C1.2 - C2.2)^2)
  in dist_centers = r1 + r2 → 
     (∃(n : ℕ), n = 3 ∧ tangent_lines C1 C2 r1 r2 n) :=
by { sorry }

end tangent_lines_to_two_circles_l310_310935


namespace total_cost_shorts_tshirt_boots_shinguards_l310_310310

variable (x : ℝ)

-- Definitions provided in the problem statement.
def cost_shorts : ℝ := x
def cost_shorts_and_tshirt : ℝ := 2 * x
def cost_shorts_and_boots : ℝ := 5 * x
def cost_shorts_and_shinguards : ℝ := 3 * x

-- The proof goal to verify:
theorem total_cost_shorts_tshirt_boots_shinguards : 
  (cost_shorts x + (cost_shorts_and_tshirt x - cost_shorts x) + 
   (cost_shorts_and_boots x - cost_shorts x) + 
   (cost_shorts_and_shinguards x - cost_shorts x)) = 8 * x := by 
  sorry

end total_cost_shorts_tshirt_boots_shinguards_l310_310310


namespace total_cost_is_eight_x_l310_310303

-- Definitions of cost variables based on conditions
variable (x : ℝ) -- Cost of shorts

-- Cost conditions
variable (shirt_cost : ℝ) (boot_cost : ℝ) (shin_guard_cost : ℝ)
variable (c1 : x + shirt_cost = 2 * x)
variable (c2 : x + boot_cost = 5 * x)
variable (c3 : x + shin_guard_cost = 3 * x)

-- To prove that the total cost is 8 times the cost of shorts
theorem total_cost_is_eight_x
  (c1 : x + shirt_cost = 2 * x)
  (c2 : x + boot_cost = 5 * x)
  (c3 : x + shin_guard_cost = 3 * x) :
  x + shirt_cost + boot_cost + shin_guard_cost = 8 * x := 
by
  sorry

end total_cost_is_eight_x_l310_310303


namespace meridian_students_l310_310519

theorem meridian_students
  (eighth_to_seventh_ratio : Nat → Nat → Prop)
  (seventh_to_sixth_ratio : Nat → Nat → Prop)
  (r1 : ∀ a b, eighth_to_seventh_ratio a b ↔ 7 * b = 4 * a)
  (r2 : ∀ b c, seventh_to_sixth_ratio b c ↔ 10 * c = 9 * b) :
  ∃ a b c, eighth_to_seventh_ratio a b ∧ seventh_to_sixth_ratio b c ∧ a + b + c = 73 :=
by
  sorry

end meridian_students_l310_310519


namespace length_of_second_train_is_correct_l310_310836

-- Define the known values and conditions
def speed_train1_kmph := 120
def speed_train2_kmph := 80
def length_train1_m := 280
def crossing_time_s := 9

-- Convert speeds from km/h to m/s
def kmph_to_mps (kmph : ℕ) : ℚ := kmph * 1000 / 3600

def speed_train1_mps := kmph_to_mps speed_train1_kmph
def speed_train2_mps := kmph_to_mps speed_train2_kmph

-- Calculate relative speed
def relative_speed_mps := speed_train1_mps + speed_train2_mps

-- Calculate total distance covered when crossing
def total_distance_m := relative_speed_mps * crossing_time_s

-- The length of the second train
def length_train2_m := total_distance_m - length_train1_m

-- Prove the length of the second train
theorem length_of_second_train_is_correct : length_train2_m = 219.95 := by {
  sorry
}

end length_of_second_train_is_correct_l310_310836


namespace sticker_price_of_smartphone_l310_310863

theorem sticker_price_of_smartphone (p : ℝ)
  (h1 : 0.90 * p - 100 = 0.80 * p - 20) : p = 800 :=
sorry

end sticker_price_of_smartphone_l310_310863


namespace kocourkov_coins_l310_310729

theorem kocourkov_coins :
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 
  (∀ n > 53, ∃ x y : ℕ, n = x * a + y * b) ∧ 
  ¬ (∃ x y : ℕ, 53 = x * a + y * b) ∧
  ((a = 2 ∧ b = 55) ∨ (a = 3 ∧ b = 28)) :=
by {
  sorry
}

end kocourkov_coins_l310_310729


namespace angle_in_third_quadrant_l310_310026

theorem angle_in_third_quadrant (θ : ℤ) (hθ : θ = -510) : 
  (210 % 360 > 180 ∧ 210 % 360 < 270) := 
by
  have h : 210 % 360 = 210 := by norm_num
  sorry

end angle_in_third_quadrant_l310_310026


namespace gcd_bc_eq_one_l310_310282

theorem gcd_bc_eq_one (a b c x y : ℕ)
  (h1 : Nat.gcd a b = 120)
  (h2 : Nat.gcd a c = 1001)
  (hb : b = 120 * x)
  (hc : c = 1001 * y) :
  Nat.gcd b c = 1 :=
by
  sorry

end gcd_bc_eq_one_l310_310282


namespace track_width_l310_310373

theorem track_width (r : ℝ) (h1 : 4 * π * r - 2 * π * r = 16 * π) (h2 : 2 * r = r + r) : 2 * r - r = 8 :=
by
  sorry

end track_width_l310_310373


namespace sin_C_value_l310_310120

theorem sin_C_value (A B C : ℝ) (a b c : ℝ) 
  (h_a : a = 1) 
  (h_b : b = 1/2) 
  (h_cos_A : Real.cos A = (Real.sqrt 3) / 2) 
  (h_angles : A + B + C = Real.pi) 
  (h_sides : Real.sin A / a = Real.sin B / b) :
  Real.sin C = (Real.sqrt 15 + Real.sqrt 3) / 8 :=
by 
  sorry

end sin_C_value_l310_310120


namespace minimum_liars_l310_310784

noncomputable def minimum_troublemakers (n : ℕ) := 10

theorem minimum_liars (n : ℕ) (students : ℕ → bool) 
  (seated_at_round_table : ∀ i, students (i % 29) = tt ∨ students (i % 29) = ff)
  (one_liar_next_to : ∀ i, students i = tt → (students ((i + 1) % 29) = ff ∨ students ((i + 28) % 29) = ff))
  (two_liars_next_to : ∀ i, students i = ff → ((students ((i + 1) % 29) = ff) ∧ students ((i + 28) % 29) = ff)) :
  ∃ m, m = minimum_troublemakers 29 ∧ (m ≤ n) :=
by
  let liars_count := 10
  have : liars_count = 10 := rfl
  use liars_count
  exact ⟨rfl, sorry⟩

end minimum_liars_l310_310784


namespace enrollment_difference_l310_310356

theorem enrollment_difference 
  (Varsity_enrollment : ℕ)
  (Northwest_enrollment : ℕ)
  (Central_enrollment : ℕ)
  (Greenbriar_enrollment : ℕ) 
  (h1 : Varsity_enrollment = 1300) 
  (h2 : Northwest_enrollment = 1500)
  (h3 : Central_enrollment = 1800)
  (h4 : Greenbriar_enrollment = 1600) : 
  Varsity_enrollment < Northwest_enrollment ∧ 
  Northwest_enrollment < Greenbriar_enrollment ∧ 
  Greenbriar_enrollment < Central_enrollment → 
    (Greenbriar_enrollment - Varsity_enrollment = 300) :=
by
  sorry

end enrollment_difference_l310_310356


namespace chromium_percentage_in_new_alloy_l310_310268

theorem chromium_percentage_in_new_alloy :
  ∀ (weight1 weight2 chromium1 chromium2: ℝ),
  weight1 = 15 → weight2 = 35 → chromium1 = 0.12 → chromium2 = 0.08 →
  (chromium1 * weight1 + chromium2 * weight2) / (weight1 + weight2) * 100 = 9.2 :=
by
  intros weight1 weight2 chromium1 chromium2 hweight1 hweight2 hchromium1 hchromium2
  sorry

end chromium_percentage_in_new_alloy_l310_310268


namespace solve_equation_l310_310599

theorem solve_equation (x : ℝ) (hx : 0 ≤ x) : 2021 * x^10 - 2020 * x - 1 = 0 ↔ x = 1 := 
by 
  sorry

end solve_equation_l310_310599


namespace infinite_series_sum_l310_310685

theorem infinite_series_sum :
  (∑' n : ℕ, n * (1/5)^n) = 5/16 :=
by sorry

end infinite_series_sum_l310_310685


namespace true_propositions_l310_310471

noncomputable theory
open_locale classical

-- Define each proposition
def prop1 : Prop :=
∀ (p : Point) (l : Line), ¬(p ∈ l) → ∃! m : Line, m ∥ l ∧ p ∈ m

def prop2 : Prop :=
∀ (p : Point) (l : Line), ∃! m : Line, m ⟂ l ∧ p ∈ m

def prop3 : Prop :=
∀ (∠α ∠β : Angle), supplementary (∠α, ∠β) → 
∃ (θα θβ : Line), angle_bisector ∠α θα ∧ angle_bisector ∠β θβ ∧ θα ⟂ θβ

def prop4 : Prop :=
∀ (a b c : Line), (a ∥ b) ∧ (b ⟂ c) → a ⟂ c

-- Mathematical equivalent proof problem
def math_proof_problem : Prop :=
prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4

-- Lean statement
theorem true_propositions :
math_proof_problem :=
by
  sorry

end true_propositions_l310_310471


namespace cost_of_purchase_l310_310315

theorem cost_of_purchase (x : ℝ) (T_shirt boots shin_guards : ℝ) 
  (h1 : x + T_shirt = 2 * x)
  (h2 : x + boots = 5 * x)
  (h3 : x + shin_guards = 3 * x) :
  x + T_shirt + boots + shin_guards = 8 * x :=
begin
  sorry
end

end cost_of_purchase_l310_310315


namespace prob_defective_l310_310046

/-- Assume there are two boxes of components. 
    The first box contains 10 pieces, including 2 defective ones; 
    the second box contains 20 pieces, including 3 defective ones. --/
def box1_total : ℕ := 10
def box1_defective : ℕ := 2
def box2_total : ℕ := 20
def box2_defective : ℕ := 3

/-- Randomly select one box from the two boxes, 
    and then randomly pick 1 component from that box. --/
def prob_select_box : ℚ := 1 / 2

/-- Probability of selecting a defective component given that box 1 was selected. --/
def prob_defective_given_box1 : ℚ := box1_defective / box1_total

/-- Probability of selecting a defective component given that box 2 was selected. --/
def prob_defective_given_box2 : ℚ := box2_defective / box2_total

/-- The probability of selecting a defective component is 7/40. --/
theorem prob_defective :
  prob_select_box * prob_defective_given_box1 + prob_select_box * prob_defective_given_box2 = 7 / 40 :=
sorry

end prob_defective_l310_310046


namespace elephants_at_WePreserveForFuture_l310_310777

theorem elephants_at_WePreserveForFuture (E : ℕ) 
  (h1 : ∀ gest : ℕ, gest = 3 * E)
  (h2 : ∀ total : ℕ, total = E + 3 * E) 
  (h3 : total = 280) : 
  E = 70 := 
by
  sorry

end elephants_at_WePreserveForFuture_l310_310777


namespace total_cost_is_eight_times_short_cost_l310_310291

variables (x : ℝ)
def cost_shorts := x
def cost_shirt := x
def cost_boots := 4 * x
def cost_shin_guards := 2 * x

theorem total_cost_is_eight_times_short_cost
    (cond_shirt : cost_shorts x + cost_shirt x = 2*x)
    (cond_boots : cost_shorts x + cost_boots x = 5*x)
    (cond_shin_guards : cost_shorts x + cost_shin_guards x = 3*x) :
    cost_shorts x + cost_shirt x + cost_boots x + cost_shin_guards x = 8*x :=
by
  sorry

end total_cost_is_eight_times_short_cost_l310_310291


namespace school_team_selection_l310_310845

theorem school_team_selection : 
  (Nat.choose 8 4) * (Nat.choose 10 4) = 14700 := by
  sorry

end school_team_selection_l310_310845


namespace mul_powers_same_base_l310_310502

theorem mul_powers_same_base : 2^2 * 2^3 = 2^5 :=
by sorry

end mul_powers_same_base_l310_310502


namespace min_value_of_f_five_l310_310118

open Real

def f (x a : ℝ) : ℝ := abs (x + 1) + 2 * abs (x - a)

theorem min_value_of_f_five (a : ℝ) :
  (∃ x, f x a = 5) → (a = -6 ∨ a = 4) :=
sorry

end min_value_of_f_five_l310_310118


namespace max_distinct_dance_counts_l310_310633

theorem max_distinct_dance_counts (B G : ℕ) (hB : B = 29) (hG : G = 15) 
  (dance_with : ℕ → ℕ → Prop)
  (h_dance_limit : ∀ b g, dance_with b g → b ≤ B ∧ g ≤ G) :
  ∃ max_counts : ℕ, max_counts = 29 :=
by
  -- The statement of the theorem. Proof is omitted.
  sorry

end max_distinct_dance_counts_l310_310633


namespace most_reasonable_sampling_method_is_stratified_l310_310374

def population_has_significant_differences 
    (grades : List String)
    (understanding : String → ℕ)
    : Prop := sorry -- This would be defined based on the details of "significant differences"

theorem most_reasonable_sampling_method_is_stratified
    (grades : List String)
    (understanding : String → ℕ)
    (h : population_has_significant_differences grades understanding)
    : (method : String) → (method = "Stratified sampling") :=
sorry

end most_reasonable_sampling_method_is_stratified_l310_310374


namespace mean_value_of_pentagon_interior_angles_l310_310010

theorem mean_value_of_pentagon_interior_angles :
  let n := 5
  let sum_of_interior_angles := (n - 2) * 180
  let mean_value := sum_of_interior_angles / n
  mean_value = 108 :=
by
  sorry

end mean_value_of_pentagon_interior_angles_l310_310010


namespace find_third_term_l310_310261

theorem find_third_term :
  ∃ (a : ℕ → ℝ), a 0 = 5 ∧ a 4 = 2025 ∧ (∀ n, a (n + 1) = a n * r) ∧ a 2 = 225 :=
by
  sorry

end find_third_term_l310_310261


namespace range_of_a_l310_310082

section
variables (a : ℝ)
def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (h : p a ∧ q a) : a = 1 ∨ a ≤ -2 :=
sorry
end

end range_of_a_l310_310082


namespace proof_problem_l310_310253

def x := 3
def y := 4

theorem proof_problem : 3 * x - 2 * y = 1 := by
  -- We will rely on these definitions and properties of arithmetic to show the result.
  -- The necessary proof steps would follow here, but are skipped for now.
  sorry

end proof_problem_l310_310253


namespace cost_of_two_burritos_and_five_quesadillas_l310_310439

theorem cost_of_two_burritos_and_five_quesadillas
  (b q : ℝ)
  (h1 : b + 4 * q = 3.50)
  (h2 : 4 * b + q = 4.10) :
  2 * b + 5 * q = 5.02 := 
sorry

end cost_of_two_burritos_and_five_quesadillas_l310_310439


namespace cubic_sum_l310_310542

theorem cubic_sum (x : ℝ) (h : x + 1/x = 4) : x^3 + 1/x^3 = 52 :=
by 
  sorry

end cubic_sum_l310_310542


namespace lucas_fraction_of_money_left_l310_310589

theorem lucas_fraction_of_money_left (m p n : ℝ) (h1 : (1 / 4) * m = (1 / 2) * n * p) :
  (m - n * p) / m = 1 / 2 :=
by 
  -- Sorry is used to denote that we are skipping the proof
  sorry

end lucas_fraction_of_money_left_l310_310589


namespace silenos_time_l310_310201

theorem silenos_time :
  (∃ x : ℝ, ∃ b: ℝ, (x - 2 = x / 2) ∧ (b = x / 3)) → (∃ x : ℝ, x = 3) :=
by sorry

end silenos_time_l310_310201


namespace solve_equation_l310_310597

theorem solve_equation (x : ℝ) (hx : 0 ≤ x) : 2021 * (x^2020)^(1/202) - 1 = 2020 * x → x = 1 :=
by
  sorry

end solve_equation_l310_310597


namespace lines_intersect_at_l310_310245

theorem lines_intersect_at :
  ∃ t u : ℝ, (∃ (x y : ℝ),
    (x = 2 + 3 * t ∧ y = 4 - 2 * t) ∧
    (x = -1 + 6 * u ∧ y = 5 + u) ∧
    (x = 1/5 ∧ y = 26/5)) :=
by
  sorry

end lines_intersect_at_l310_310245


namespace common_difference_range_l310_310737

theorem common_difference_range (a : ℕ → ℝ) (d : ℝ) (h : a 3 = 2) (h_pos : ∀ n, a n > 0) (h_arith : ∀ n, a (n + 1) = a n + d) : 0 ≤ d ∧ d < 1 :=
by
  sorry

end common_difference_range_l310_310737


namespace train_speed_is_100_kmph_l310_310042

noncomputable def speed_of_train (length_of_train : ℝ) (time_to_cross_pole : ℝ) : ℝ :=
  (length_of_train / time_to_cross_pole) * 3.6

theorem train_speed_is_100_kmph :
  speed_of_train 100 3.6 = 100 :=
by
  sorry

end train_speed_is_100_kmph_l310_310042


namespace min_dSigma_correct_l310_310585

noncomputable def min_dSigma {a r : ℝ} (h : a > r) : ℝ :=
  (a - r) / 2

theorem min_dSigma_correct (a r : ℝ) (h : a > r) :
  min_dSigma h = (a - r) / 2 :=
by 
  unfold min_dSigma
  sorry

end min_dSigma_correct_l310_310585
