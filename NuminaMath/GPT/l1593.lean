import Mathlib

namespace NUMINAMATH_GPT_even_quadratic_increasing_l1593_159386

theorem even_quadratic_increasing (m : ℝ) (h : ∀ x : ℝ, (m-1)*x^2 + 2*m*x + 1 = (m-1)*(-x)^2 + 2*m*(-x) + 1) :
  ∀ x1 x2 : ℝ, x1 < x2 ∧ x2 ≤ 0 → ((m-1)*x1^2 + 2*m*x1 + 1) < ((m-1)*x2^2 + 2*m*x2 + 1) :=
sorry

end NUMINAMATH_GPT_even_quadratic_increasing_l1593_159386


namespace NUMINAMATH_GPT_reconstruct_right_triangle_l1593_159305

theorem reconstruct_right_triangle (c d : ℝ) (hc : c > 0) (hd : d > 0) :
  ∃ A X Y: ℝ, (A ≠ X ∧ A ≠ Y ∧ X ≠ Y) ∧ 
  -- Right triangle with hypotenuse c
  (A - X) ^ 2 + (Y - X) ^ 2 = c ^ 2 ∧ 
  -- Difference of legs is d
  ∃ AY XY: ℝ, ((AY = abs (A - Y)) ∧ (XY = abs (Y - X)) ∧ (abs (AY - XY) = d)) := 
by
  sorry

end NUMINAMATH_GPT_reconstruct_right_triangle_l1593_159305


namespace NUMINAMATH_GPT_determine_constants_l1593_159371

theorem determine_constants (P Q R : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (-2 * x^2 + 5 * x - 7) / (x^3 - x) = P / x + (Q * x + R) / (x^2 - 1)) ↔
    (P = 7 ∧ Q = -9 ∧ R = 5) :=
by
  sorry

end NUMINAMATH_GPT_determine_constants_l1593_159371


namespace NUMINAMATH_GPT_solve_for_lambda_l1593_159335

def vector_dot_product : (ℤ × ℤ) → (ℤ × ℤ) → ℤ
| (x1, y1), (x2, y2) => x1 * x2 + y1 * y2

theorem solve_for_lambda
  (a : ℤ × ℤ) (b : ℤ × ℤ) (lambda : ℤ)
  (h1 : a = (3, -2))
  (h2 : b = (1, 2))
  (h3 : vector_dot_product (a.1 + lambda * b.1, a.2 + lambda * b.2) a = 0) :
  lambda = 13 :=
sorry

end NUMINAMATH_GPT_solve_for_lambda_l1593_159335


namespace NUMINAMATH_GPT_bowling_average_before_last_match_l1593_159329

theorem bowling_average_before_last_match
  (wickets_before_last : ℕ)
  (wickets_last_match : ℕ)
  (runs_last_match : ℕ)
  (decrease_in_average : ℝ)
  (average_before_last : ℝ) :

  wickets_before_last = 115 →
  wickets_last_match = 6 →
  runs_last_match = 26 →
  decrease_in_average = 0.4 →
  (average_before_last - decrease_in_average) = 
  ((wickets_before_last * average_before_last + runs_last_match) / 
  (wickets_before_last + wickets_last_match)) →
  average_before_last = 12.4 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_bowling_average_before_last_match_l1593_159329


namespace NUMINAMATH_GPT_divisor_is_three_l1593_159341

theorem divisor_is_three (n d q p : ℕ) (h1 : n = d * q + 3) (h2 : n^2 = d * p + 3) : d = 3 := 
sorry

end NUMINAMATH_GPT_divisor_is_three_l1593_159341


namespace NUMINAMATH_GPT_slope_of_line_l1593_159309

-- Definition of the line equation
def lineEquation (x y : ℝ) : Prop := 4 * x - 7 * y = 14

-- The statement that we need to prove
theorem slope_of_line : ∀ x y, lineEquation x y → ∃ m, m = 4 / 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_slope_of_line_l1593_159309


namespace NUMINAMATH_GPT_angle_P_of_extended_sides_l1593_159311

noncomputable def regular_pentagon_angle_sum : ℕ := 540

noncomputable def internal_angle_regular_pentagon (n : ℕ) (h : 5 = n) : ℕ :=
  regular_pentagon_angle_sum / n

def interior_angle_pentagon : ℕ := 108

theorem angle_P_of_extended_sides (ABCDE : Prop) (h1 : interior_angle_pentagon = 108)
  (P : Prop) (h3 : 72 + 72 = 144) : 180 - 144 = 36 := by 
  sorry

end NUMINAMATH_GPT_angle_P_of_extended_sides_l1593_159311


namespace NUMINAMATH_GPT_problem1_problem2_l1593_159306

-- Problem 1: Proving the given equation under specified conditions
theorem problem1 (x y : ℝ) (h : x + y ≠ 0) : ((2 * x + 3 * y) / (x + y)) - ((x + 2 * y) / (x + y)) = 1 :=
sorry

-- Problem 2: Proving the given equation under specified conditions
theorem problem2 (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ 1) : ((a^2 - 1) / (a^2 - 4 * a + 4)) / ((a - 1) / (a - 2)) = (a + 1) / (a - 2) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1593_159306


namespace NUMINAMATH_GPT_cubes_with_red_face_l1593_159366

theorem cubes_with_red_face :
  let totalCubes := 10 * 10 * 10
  let innerCubes := (10 - 2) * (10 - 2) * (10 - 2)
  let redFaceCubes := totalCubes - innerCubes
  redFaceCubes = 488 :=
by
  let totalCubes := 10 * 10 * 10
  let innerCubes := (10 - 2) * (10 - 2) * (10 - 2)
  let redFaceCubes := totalCubes - innerCubes
  sorry

end NUMINAMATH_GPT_cubes_with_red_face_l1593_159366


namespace NUMINAMATH_GPT_sandy_gave_puppies_l1593_159389

theorem sandy_gave_puppies 
  (original_puppies : ℕ) 
  (puppies_with_spots : ℕ) 
  (puppies_left : ℕ) 
  (h1 : original_puppies = 8) 
  (h2 : puppies_with_spots = 3) 
  (h3 : puppies_left = 4) : 
  original_puppies - puppies_left = 4 := 
by {
  -- This is a placeholder for the proof.
  sorry
}

end NUMINAMATH_GPT_sandy_gave_puppies_l1593_159389


namespace NUMINAMATH_GPT_cameron_answers_l1593_159376

theorem cameron_answers (q_per_tourist : ℕ := 2) 
  (group_1 : ℕ := 6) 
  (group_2 : ℕ := 11) 
  (group_3 : ℕ := 8) 
  (group_3_inquisitive : ℕ := 1) 
  (group_4 : ℕ := 7) :
  (q_per_tourist * group_1) +
  (q_per_tourist * group_2) +
  (q_per_tourist * (group_3 - group_3_inquisitive)) +
  (q_per_tourist * 3 * group_3_inquisitive) +
  (q_per_tourist * group_4) = 68 :=
by
  sorry

end NUMINAMATH_GPT_cameron_answers_l1593_159376


namespace NUMINAMATH_GPT_vectors_perpendicular_l1593_159360

open Real

def vector := ℝ × ℝ

def dot_product (v w : vector) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def perpendicular (v w : vector) : Prop :=
  dot_product v w = 0

def vector_sub (v w : vector) : vector :=
  (v.1 - w.1, v.2 - w.2)

theorem vectors_perpendicular :
  let a : vector := (2, 0)
  let b : vector := (1, 1)
  perpendicular (vector_sub a b) b :=
by
  sorry

end NUMINAMATH_GPT_vectors_perpendicular_l1593_159360


namespace NUMINAMATH_GPT_probability_AEMC9_is_1_over_84000_l1593_159372

-- Define possible symbols for each category.
def vowels : List Char := ['A', 'E', 'I', 'O', 'U']
def nonVowels : List Char := ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
def digits : List Char := ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

-- Define the total number of possible license plates.
def totalLicensePlates : Nat := 
  (vowels.length) * (vowels.length - 1) * 
  (nonVowels.length) * (nonVowels.length - 1) * 
  (digits.length)

-- Define the number of favorable outcomes.
def favorableOutcomes : Nat := 1

-- Define the probability calculation.
noncomputable def probabilityAEMC9 : ℚ := favorableOutcomes / totalLicensePlates

-- The theorem to prove.
theorem probability_AEMC9_is_1_over_84000 :
  probabilityAEMC9 = 1 / 84000 := by
  sorry

end NUMINAMATH_GPT_probability_AEMC9_is_1_over_84000_l1593_159372


namespace NUMINAMATH_GPT_lives_per_player_l1593_159316

theorem lives_per_player (num_players total_lives : ℕ) (h1 : num_players = 8) (h2 : total_lives = 64) :
  total_lives / num_players = 8 := by
  sorry

end NUMINAMATH_GPT_lives_per_player_l1593_159316


namespace NUMINAMATH_GPT_not_divisor_of_44_l1593_159340

theorem not_divisor_of_44 (m j : ℤ) (H1 : m = j * (j + 1) * (j + 2) * (j + 3))
  (H2 : 11 ∣ m) : ¬ (∀ j : ℤ, 44 ∣ j * (j + 1) * (j + 2) * (j + 3)) :=
by
  sorry

end NUMINAMATH_GPT_not_divisor_of_44_l1593_159340


namespace NUMINAMATH_GPT_oldest_child_age_l1593_159315

open Nat

def avg_age (a b c d : ℕ) := (a + b + c + d) / 4

theorem oldest_child_age 
  (h_avg : avg_age 5 8 11 x = 9) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_oldest_child_age_l1593_159315


namespace NUMINAMATH_GPT_minimum_sum_of_natural_numbers_with_lcm_2012_l1593_159387

/-- 
Prove that the minimum sum of seven natural numbers whose least common multiple is 2012 is 512.
-/

theorem minimum_sum_of_natural_numbers_with_lcm_2012 : 
  ∃ (a b c d e f g : ℕ), Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm a b) c) d) e) f) g = 2012 ∧ (a + b + c + d + e + f + g) = 512 :=
sorry

end NUMINAMATH_GPT_minimum_sum_of_natural_numbers_with_lcm_2012_l1593_159387


namespace NUMINAMATH_GPT_smallest_m_exists_l1593_159395

theorem smallest_m_exists : ∃ (m : ℕ), (∀ n : ℕ, (n > 0) → ((10000 * n % 53 = 0) → (m ≤ n))) ∧ (10000 * m % 53 = 0) :=
by
  sorry

end NUMINAMATH_GPT_smallest_m_exists_l1593_159395


namespace NUMINAMATH_GPT_polygon_sides_given_interior_angle_l1593_159362

theorem polygon_sides_given_interior_angle
  (h : ∀ (n : ℕ), (n > 2) → ((n - 2) * 180 = n * 140)): n = 9 := by
  sorry

end NUMINAMATH_GPT_polygon_sides_given_interior_angle_l1593_159362


namespace NUMINAMATH_GPT_find_c_l1593_159348

theorem find_c (a b c : ℝ) (h1 : a + b = 5) (h2 : c^2 = a * b + b - 9) : c = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1593_159348


namespace NUMINAMATH_GPT_paperboy_delivery_sequences_l1593_159393

noncomputable def D : ℕ → ℕ
| 0       => 1  -- D_0 is a dummy value to facilitate indexing
| 1       => 2
| 2       => 4
| 3       => 7
| (n + 4) => D (n + 3) + D (n + 2) + D (n + 1)

theorem paperboy_delivery_sequences : D 11 = 927 := by
  sorry

end NUMINAMATH_GPT_paperboy_delivery_sequences_l1593_159393


namespace NUMINAMATH_GPT_first_two_digits_of_52x_l1593_159332

-- Define the digit values that would make 52x divisible by 6.
def digit_values (x : Nat) : Prop :=
  x = 2 ∨ x = 5 ∨ x = 8

-- The main theorem to prove the first two digits are 52 given the conditions.
theorem first_two_digits_of_52x (x : Nat) (h : digit_values x) : (52 * 10 + x) / 10 = 52 :=
by sorry

end NUMINAMATH_GPT_first_two_digits_of_52x_l1593_159332


namespace NUMINAMATH_GPT_radius_of_circle_l1593_159310

noncomputable def circle_radius (x y : ℝ) : ℝ := 
  let lhs := x^2 - 8 * x + y^2 - 4 * y + 16
  if lhs = 0 then 2 else 0

theorem radius_of_circle : circle_radius 0 0 = 2 :=
sorry

end NUMINAMATH_GPT_radius_of_circle_l1593_159310


namespace NUMINAMATH_GPT_liars_count_l1593_159328

inductive Person
| Knight
| Liar
| Eccentric

open Person

def isLiarCondition (p : Person) (right : Person) : Prop :=
  match p with
  | Knight => right = Liar
  | Liar => right ≠ Liar
  | Eccentric => True

theorem liars_count (people : Fin 100 → Person) (h : ∀ i, isLiarCondition (people i) (people ((i + 1) % 100))) :
  (∃ n : ℕ, n = 0 ∨ n = 50) :=
sorry

end NUMINAMATH_GPT_liars_count_l1593_159328


namespace NUMINAMATH_GPT_second_pipe_fills_in_15_minutes_l1593_159399

theorem second_pipe_fills_in_15_minutes :
  ∀ (x : ℝ),
  (∀ (x : ℝ), (1 / 2 + (7.5 / x)) = 1 → x = 15) :=
by
  intros
  sorry

end NUMINAMATH_GPT_second_pipe_fills_in_15_minutes_l1593_159399


namespace NUMINAMATH_GPT_fill_up_minivans_l1593_159322

theorem fill_up_minivans (service_cost : ℝ) (fuel_cost_per_liter : ℝ) (total_cost : ℝ)
  (mini_van_liters : ℝ) (truck_percent_bigger : ℝ) (num_trucks : ℕ) (num_minivans : ℕ) :
  service_cost = 2.3 ∧ fuel_cost_per_liter = 0.7 ∧ total_cost = 396 ∧
  mini_van_liters = 65 ∧ truck_percent_bigger = 1.2 ∧ num_trucks = 2 →
  num_minivans = 4 :=
by
  sorry

end NUMINAMATH_GPT_fill_up_minivans_l1593_159322


namespace NUMINAMATH_GPT_existence_of_indices_l1593_159381

theorem existence_of_indices 
  (a1 a2 a3 a4 a5 : ℝ) 
  (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) (h4 : 0 < a4) (h5 : 0 < a5) : 
  ∃ (i j k l : Fin 5), 
    (i ≠ j) ∧ (i ≠ k) ∧ (i ≠ l) ∧ (j ≠ k) ∧ (j ≠ l) ∧ (k ≠ l) ∧ 
    |(a1 / a2) - (a3 / a4)| < 1/2 :=
by 
  sorry

end NUMINAMATH_GPT_existence_of_indices_l1593_159381


namespace NUMINAMATH_GPT_geese_count_l1593_159346

-- Define the number of ducks in the marsh
def number_of_ducks : ℕ := 37

-- Define the total number of birds in the marsh
def total_number_of_birds : ℕ := 95

-- Define the number of geese in the marsh
def number_of_geese : ℕ := total_number_of_birds - number_of_ducks

-- Theorem stating the number of geese in the marsh is 58
theorem geese_count : number_of_geese = 58 := by
  sorry

end NUMINAMATH_GPT_geese_count_l1593_159346


namespace NUMINAMATH_GPT_elmer_saves_21_875_percent_l1593_159380

noncomputable def old_car_efficiency (x : ℝ) := x
noncomputable def new_car_efficiency (x : ℝ) := 1.6 * x

noncomputable def gasoline_cost (c : ℝ) := c
noncomputable def diesel_cost (c : ℝ) := 1.25 * c

noncomputable def trip_distance := 1000

noncomputable def old_car_fuel_consumption (x : ℝ) := trip_distance / x
noncomputable def new_car_fuel_consumption (x : ℝ) := trip_distance / (new_car_efficiency x)

noncomputable def old_car_trip_cost (x c : ℝ) := (trip_distance / x) * c
noncomputable def new_car_trip_cost (x c : ℝ) := (trip_distance / (new_car_efficiency x)) * (diesel_cost c)

noncomputable def savings (x c : ℝ) := old_car_trip_cost x c - new_car_trip_cost x c
noncomputable def percentage_savings (x c : ℝ) := (savings x c) / (old_car_trip_cost x c) * 100

theorem elmer_saves_21_875_percent (x c : ℝ) : percentage_savings x c = 21.875 := 
sorry

end NUMINAMATH_GPT_elmer_saves_21_875_percent_l1593_159380


namespace NUMINAMATH_GPT_Chad_savings_l1593_159343

theorem Chad_savings :
  let earnings_mowing := 600
  let earnings_birthday := 250
  let earnings_video_games := 150
  let earnings_odd_jobs := 150
  let total_earnings := earnings_mowing + earnings_birthday + earnings_video_games + earnings_odd_jobs
  let savings_rate := 0.40
  let savings := savings_rate * total_earnings
  savings = 460 :=
by
  -- Definitions
  let earnings_mowing : ℤ := 600
  let earnings_birthday : ℤ := 250
  let earnings_video_games : ℤ := 150
  let earnings_odd_jobs : ℤ := 150
  let total_earnings : ℤ := earnings_mowing + earnings_birthday + earnings_video_games + earnings_odd_jobs
  let savings_rate := (40:ℚ) / 100
  let savings : ℚ := savings_rate * total_earnings
  -- Proof (to be completed by sorry)
  exact sorry

end NUMINAMATH_GPT_Chad_savings_l1593_159343


namespace NUMINAMATH_GPT_cheesecake_needs_more_eggs_l1593_159300

def chocolate_eggs_per_cake := 3
def cheesecake_eggs_per_cake := 8
def num_chocolate_cakes := 5
def num_cheesecakes := 9

theorem cheesecake_needs_more_eggs :
  cheesecake_eggs_per_cake * num_cheesecakes - chocolate_eggs_per_cake * num_chocolate_cakes = 57 :=
by
  sorry

end NUMINAMATH_GPT_cheesecake_needs_more_eggs_l1593_159300


namespace NUMINAMATH_GPT_impossible_to_half_boys_sit_with_girls_l1593_159304

theorem impossible_to_half_boys_sit_with_girls:
  ∀ (g b : ℕ), 
  (g + b = 30) → 
  (∃ k, g = 2 * k) →
  (∀ (d : ℕ), 2 * d = g) →
  ¬ ∃ m, (b = 2 * m) ∧ (∀ (d : ℕ), 2 * d = b) :=
by
  sorry

end NUMINAMATH_GPT_impossible_to_half_boys_sit_with_girls_l1593_159304


namespace NUMINAMATH_GPT_speed_of_first_train_l1593_159392

theorem speed_of_first_train
  (length_train1 length_train2 : ℕ)
  (speed_train2 : ℕ)
  (time_seconds : ℝ)
  (distance_km : ℝ := (length_train1 + length_train2) / 1000)
  (time_hours : ℝ := time_seconds / 3600)
  (relative_speed : ℝ := distance_km / time_hours) :
  length_train1 = 111 →
  length_train2 = 165 →
  speed_train2 = 120 →
  time_seconds = 4.516002356175142 →
  relative_speed = 220 →
  speed_train2 + 100 = relative_speed :=
by
  intros
  sorry

end NUMINAMATH_GPT_speed_of_first_train_l1593_159392


namespace NUMINAMATH_GPT_grace_earnings_in_september_l1593_159350

theorem grace_earnings_in_september
  (hours_mowing : ℕ) (hours_pulling_weeds : ℕ) (hours_putting_mulch : ℕ)
  (rate_mowing : ℕ) (rate_pulling_weeds : ℕ) (rate_putting_mulch : ℕ)
  (total_hours_mowing : hours_mowing = 63) (total_hours_pulling_weeds : hours_pulling_weeds = 9) (total_hours_putting_mulch : hours_putting_mulch = 10)
  (rate_for_mowing : rate_mowing = 6) (rate_for_pulling_weeds : rate_pulling_weeds = 11) (rate_for_putting_mulch : rate_putting_mulch = 9) :
  hours_mowing * rate_mowing + hours_pulling_weeds * rate_pulling_weeds + hours_putting_mulch * rate_putting_mulch = 567 :=
by
  intros
  sorry

end NUMINAMATH_GPT_grace_earnings_in_september_l1593_159350


namespace NUMINAMATH_GPT_smallest_positive_multiple_of_6_and_5_l1593_159345

theorem smallest_positive_multiple_of_6_and_5 : ∃ (n : ℕ), (n > 0) ∧ (n % 6 = 0) ∧ (n % 5 = 0) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 6 = 0) ∧ (m % 5 = 0) → n ≤ m) :=
  sorry

end NUMINAMATH_GPT_smallest_positive_multiple_of_6_and_5_l1593_159345


namespace NUMINAMATH_GPT_sqrt_36_eq_6_cube_root_neg_a_125_l1593_159382

theorem sqrt_36_eq_6 : ∀ (x : ℝ), 0 ≤ x ∧ x^2 = 36 → x = 6 :=
by sorry

theorem cube_root_neg_a_125 : ∀ (a y : ℝ), y^3 = - a / 125 → y = - (a^(1/3)) / 5 :=
by sorry

end NUMINAMATH_GPT_sqrt_36_eq_6_cube_root_neg_a_125_l1593_159382


namespace NUMINAMATH_GPT_proof_problem_l1593_159359

-- Define the propositions p and q.
def p (a : ℝ) : Prop := a < -1/2 

def q (a b : ℝ) : Prop := a > b → (1 / (a + 1)) < (1 / (b + 1))

-- Define the final proof problem: proving that "p or q" is true.
theorem proof_problem (a b : ℝ) : (p a) ∨ (q a b) := by
  sorry

end NUMINAMATH_GPT_proof_problem_l1593_159359


namespace NUMINAMATH_GPT_no_real_solutions_for_eqn_l1593_159363

theorem no_real_solutions_for_eqn :
  ¬ ∃ x : ℝ, (x + 4) ^ 2 = 3 * (x - 2) := 
by 
  sorry

end NUMINAMATH_GPT_no_real_solutions_for_eqn_l1593_159363


namespace NUMINAMATH_GPT_cos_double_angle_zero_l1593_159327

variable (θ : ℝ)

-- Conditions
def tan_eq_one : Prop := Real.tan θ = 1

-- Objective
theorem cos_double_angle_zero (h : tan_eq_one θ) : Real.cos (2 * θ) = 0 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_zero_l1593_159327


namespace NUMINAMATH_GPT_f_at_1_is_neg7007_l1593_159351

variable (a b c : ℝ)

def g (x : ℝ) := x^3 + a * x^2 + x + 10
def f (x : ℝ) := x^4 + x^3 + b * x^2 + 100 * x + c

theorem f_at_1_is_neg7007
  (a b c : ℝ)
  (h1 : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ g a (r1) = 0 ∧ g a (r2) = 0 ∧ g a (r3) = 0)
  (h2 : ∀ x, f x = 0 → g x = 0) :
  f 1 = -7007 := 
sorry

end NUMINAMATH_GPT_f_at_1_is_neg7007_l1593_159351


namespace NUMINAMATH_GPT_vector_triangle_c_solution_l1593_159333

theorem vector_triangle_c_solution :
  let a : ℝ × ℝ := (1, -3)
  let b : ℝ × ℝ := (-2, 4)
  let c : ℝ × ℝ := (4, -6)
  (4 • a + (3 • b - 2 • a) + c = (0, 0)) →
  c = (4, -6) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_vector_triangle_c_solution_l1593_159333


namespace NUMINAMATH_GPT_increase_in_tire_radius_l1593_159358

theorem increase_in_tire_radius
  (r : ℝ)
  (d1 d2 : ℝ)
  (conv_factor : ℝ)
  (original_radius : r = 16)
  (odometer_reading_outbound : d1 = 500)
  (odometer_reading_return : d2 = 485)
  (conversion_factor : conv_factor = 63360) :
  ∃ Δr : ℝ, Δr = 0.33 :=
by
  sorry

end NUMINAMATH_GPT_increase_in_tire_radius_l1593_159358


namespace NUMINAMATH_GPT_barbara_wins_gameA_l1593_159384

noncomputable def gameA_winning_strategy : Prop :=
∃ (has_winning_strategy : (ℤ → ℝ) → Prop),
  has_winning_strategy (fun n => n : ℤ → ℝ)

theorem barbara_wins_gameA :
  gameA_winning_strategy := sorry

end NUMINAMATH_GPT_barbara_wins_gameA_l1593_159384


namespace NUMINAMATH_GPT_hyperbola_represents_l1593_159342

theorem hyperbola_represents (k : ℝ) : 
  (k - 2) * (5 - k) < 0 ↔ (k < 2 ∨ k > 5) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_represents_l1593_159342


namespace NUMINAMATH_GPT_non_adjacent_arrangement_l1593_159312

-- Define the number of people
def numPeople : ℕ := 8

-- Define the number of specific people who must not be adjacent
def numSpecialPeople : ℕ := 3

-- Define the number of general people who are not part of the specific group
def numGeneralPeople : ℕ := numPeople - numSpecialPeople

-- Permutations calculation for general people
def permuteGeneralPeople : ℕ := Nat.factorial numGeneralPeople

-- Number of gaps available after arranging general people
def numGaps : ℕ := numGeneralPeople + 1

-- Permutations calculation for special people placed in the gaps
def permuteSpecialPeople : ℕ := Nat.descFactorial numGaps numSpecialPeople

-- Total permutations
def totalPermutations : ℕ := permuteSpecialPeople * permuteGeneralPeople

theorem non_adjacent_arrangement :
  totalPermutations = Nat.descFactorial 6 3 * Nat.factorial 5 := by
  sorry

end NUMINAMATH_GPT_non_adjacent_arrangement_l1593_159312


namespace NUMINAMATH_GPT_difference_of_two_distinct_members_sum_of_two_distinct_members_l1593_159396

theorem difference_of_two_distinct_members (S : Set ℕ) (h : S = {n | n ∈ Finset.range 20 ∧ 1 ≤ n ∧ n ≤ 20}) :
  (∃ N, N = 19 ∧ (∀ n, 1 ≤ n ∧ n ≤ N → ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ n = a - b)) :=
by
  sorry

theorem sum_of_two_distinct_members (S : Set ℕ) (h : S = {n | n ∈ Finset.range 20 ∧ 1 ≤ n ∧ n ≤ 20}) :
  (∃ M, M = 37 ∧ (∀ m, 3 ≤ m ∧ m ≤ 39 → ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ m = a + b)) :=
by
  sorry

end NUMINAMATH_GPT_difference_of_two_distinct_members_sum_of_two_distinct_members_l1593_159396


namespace NUMINAMATH_GPT_seeds_in_small_gardens_l1593_159375

theorem seeds_in_small_gardens 
  (total_seeds : ℕ)
  (planted_seeds : ℕ)
  (small_gardens : ℕ)
  (remaining_seeds := total_seeds - planted_seeds) 
  (seeds_per_garden := remaining_seeds / small_gardens) :
  total_seeds = 101 → planted_seeds = 47 → small_gardens = 9 → seeds_per_garden = 6 := by
  sorry

end NUMINAMATH_GPT_seeds_in_small_gardens_l1593_159375


namespace NUMINAMATH_GPT_monotonic_increasing_intervals_l1593_159377

noncomputable def f (x : ℝ) : ℝ := (Real.cos (x - Real.pi / 6))^2

theorem monotonic_increasing_intervals (k : ℤ) : 
  ∃ t : Set ℝ, t = Set.Ioo (-Real.pi / 3 + k * Real.pi) (Real.pi / 6 + k * Real.pi) ∧ 
    ∀ x y, x ∈ t → y ∈ t → x ≤ y → f x ≤ f y :=
sorry

end NUMINAMATH_GPT_monotonic_increasing_intervals_l1593_159377


namespace NUMINAMATH_GPT_total_cans_l1593_159398

theorem total_cans (c o : ℕ) (h1 : c = 8) (h2 : o = 2 * c) : c + o = 24 := by
  sorry

end NUMINAMATH_GPT_total_cans_l1593_159398


namespace NUMINAMATH_GPT_mass_of_sodium_acetate_formed_l1593_159391

-- Define the reaction conditions and stoichiometry
def initial_moles_acetic_acid : ℝ := 3
def initial_moles_sodium_hydroxide : ℝ := 4
def initial_reaction_moles_acetic_acid_with_sodium_carbonate : ℝ := 2
def initial_reaction_moles_sodium_carbonate : ℝ := 1
def product_moles_sodium_acetate_from_step1 : ℝ := 2
def remaining_moles_acetic_acid : ℝ := initial_moles_acetic_acid - initial_reaction_moles_acetic_acid_with_sodium_carbonate
def product_moles_sodium_acetate_from_step2 : ℝ := remaining_moles_acetic_acid
def total_moles_sodium_acetate : ℝ := product_moles_sodium_acetate_from_step1 + product_moles_sodium_acetate_from_step2
def molar_mass_sodium_acetate : ℝ := 82.04

-- Translate to the equivalent proof problem
theorem mass_of_sodium_acetate_formed :
  total_moles_sodium_acetate * molar_mass_sodium_acetate = 246.12 :=
by
  -- The detailed proof steps would go here
  sorry

end NUMINAMATH_GPT_mass_of_sodium_acetate_formed_l1593_159391


namespace NUMINAMATH_GPT_ming_dynasty_wine_problem_l1593_159302

theorem ming_dynasty_wine_problem (x y : ℕ) (h1 : x + y = 19) (h2 : 3 * x + y / 3 = 33 ) : 
  (x = 10 ∧ y = 9) :=
by {
  sorry
}

end NUMINAMATH_GPT_ming_dynasty_wine_problem_l1593_159302


namespace NUMINAMATH_GPT_initially_working_machines_l1593_159353

theorem initially_working_machines (N R x : ℝ) 
  (h1 : N * R = x / 3) 
  (h2 : 45 * R = x / 2) : 
  N = 30 := by
  sorry

end NUMINAMATH_GPT_initially_working_machines_l1593_159353


namespace NUMINAMATH_GPT_number_of_three_digit_numbers_is_48_l1593_159330

-- Define the problem: the cards and their constraints
def card1 := (1, 2)
def card2 := (3, 4)
def card3 := (5, 6)

-- The condition given is that 6 cannot be used as 9

-- Define the function to compute the number of different three-digit numbers
def number_of_three_digit_numbers : Nat := 6 * 4 * 2

/- Prove that the number of different three-digit numbers that can be formed is 48 -/
theorem number_of_three_digit_numbers_is_48 : number_of_three_digit_numbers = 48 :=
by
  -- We skip the proof here
  sorry

end NUMINAMATH_GPT_number_of_three_digit_numbers_is_48_l1593_159330


namespace NUMINAMATH_GPT_tan_eq_2sqrt3_over_3_l1593_159394

theorem tan_eq_2sqrt3_over_3 (θ : ℝ) (h : 2 * Real.cos (θ - Real.pi / 3) = 3 * Real.cos θ) : 
  Real.tan θ = 2 * Real.sqrt 3 / 3 :=
by 
  sorry -- Proof is omitted as per the instructions

end NUMINAMATH_GPT_tan_eq_2sqrt3_over_3_l1593_159394


namespace NUMINAMATH_GPT_total_workers_construction_l1593_159339

def number_of_monkeys : Nat := 239
def number_of_termites : Nat := 622
def total_workers (m : Nat) (t : Nat) : Nat := m + t

theorem total_workers_construction : total_workers number_of_monkeys number_of_termites = 861 := by
  sorry

end NUMINAMATH_GPT_total_workers_construction_l1593_159339


namespace NUMINAMATH_GPT_ratio_third_to_second_is_one_l1593_159334

variable (x y : ℕ)

-- The second throw skips 2 more times than the first throw
def second_throw := x + 2
-- The third throw skips y times
def third_throw := y
-- The fourth throw skips 3 fewer times than the third throw
def fourth_throw := y - 3
-- The fifth throw skips 1 more time than the fourth throw
def fifth_throw := (y - 3) + 1

-- The fifth throw skipped 8 times
axiom fifth_throw_condition : fifth_throw y = 8
-- The total number of skips between all throws is 33
axiom total_skips_condition : x + second_throw x + y + fourth_throw y + fifth_throw y = 33

-- Prove the ratio of skips in third throw to the second throw is 1:1
theorem ratio_third_to_second_is_one : (third_throw y) / (second_throw x) = 1 := sorry

end NUMINAMATH_GPT_ratio_third_to_second_is_one_l1593_159334


namespace NUMINAMATH_GPT_cost_comparison_cost_effectiveness_47_l1593_159324

section
variable (x : ℕ)

-- Conditions
def price_teapot : ℕ := 25
def price_teacup : ℕ := 5
def quantity_teapots : ℕ := 4
def discount_scheme_2 : ℝ := 0.94

-- Total cost for Scheme 1
def cost_scheme_1 (x : ℕ) : ℕ :=
  (quantity_teapots * price_teapot) + (price_teacup * (x - quantity_teapots))

-- Total cost for Scheme 2
def cost_scheme_2 (x : ℕ) : ℝ :=
  (quantity_teapots * price_teapot + price_teacup * x : ℝ) * discount_scheme_2

-- The proof problem
theorem cost_comparison (x : ℕ) (h : x ≥ 4) :
  cost_scheme_1 x = 5 * x + 80 ∧ cost_scheme_2 x = 4.7 * x + 94 :=
sorry

-- When x = 47
theorem cost_effectiveness_47 : cost_scheme_2 47 < cost_scheme_1 47 :=
sorry

end

end NUMINAMATH_GPT_cost_comparison_cost_effectiveness_47_l1593_159324


namespace NUMINAMATH_GPT_negation_of_proposition_l1593_159336

theorem negation_of_proposition (x y : ℝ) :
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1593_159336


namespace NUMINAMATH_GPT_full_price_tickets_count_l1593_159369

def num_tickets_reduced := 5400
def total_tickets := 25200
def num_tickets_full := 5 * num_tickets_reduced

theorem full_price_tickets_count :
  num_tickets_reduced + num_tickets_full = total_tickets → num_tickets_full = 27000 :=
by
  sorry

end NUMINAMATH_GPT_full_price_tickets_count_l1593_159369


namespace NUMINAMATH_GPT_solve_for_x_l1593_159307

noncomputable def avg (a b : ℝ) := (a + b) / 2

noncomputable def B (t : List ℝ) : List ℝ :=
  match t with
  | [a, b, c, d, e] => [avg a b, avg b c, avg c d, avg d e]
  | _ => []

noncomputable def B_iter (m : ℕ) (t : List ℝ) : List ℝ :=
  match m with
  | 0 => t
  | k + 1 => B (B_iter k t)

theorem solve_for_x (x : ℝ) (h1 : 0 < x) (h2 : B_iter 4 [1, x, x^2, x^3, x^4] = [1/4]) :
  x = Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1593_159307


namespace NUMINAMATH_GPT_constant_term_is_19_l1593_159383

theorem constant_term_is_19 (x y C : ℝ) 
  (h1 : 7 * x + y = C) 
  (h2 : x + 3 * y = 1) 
  (h3 : 2 * x + y = 5) : 
  C = 19 :=
sorry

end NUMINAMATH_GPT_constant_term_is_19_l1593_159383


namespace NUMINAMATH_GPT_salary_percentage_difference_l1593_159338

theorem salary_percentage_difference (A B : ℝ) (h : A = 0.8 * B) :
  (B - A) / A * 100 = 25 :=
sorry

end NUMINAMATH_GPT_salary_percentage_difference_l1593_159338


namespace NUMINAMATH_GPT_trajectory_of_M_is_ellipse_l1593_159320

def circle_eq (x y : ℝ) : Prop := ((x + 3)^2 + y^2 = 100)

def point_B (x y : ℝ) : Prop := (x = 3 ∧ y = 0)

def point_on_circle (P : ℝ × ℝ) : Prop :=
  ∃ x y, P = (x, y) ∧ circle_eq x y

def perpendicular_bisector_intersects_CQ_at_M (B P M : ℝ × ℝ) : Prop :=
  (B.fst = 3 ∧ B.snd = 0) ∧
  point_on_circle P ∧
  ∃ r : ℝ, (P.fst + B.fst) / 2 = M.fst ∧ r = (M.snd - P.snd) / (M.fst - P.fst) ∧ 
  r = -(P.fst - B.fst) / (P.snd - B.snd)

theorem trajectory_of_M_is_ellipse (M : ℝ × ℝ) 
  (hC : ∀ x y, circle_eq x y)
  (hB : point_B 3 0)
  (hP : ∃ P : ℝ × ℝ, point_on_circle P)
  (hM : ∃ B P : ℝ × ℝ, perpendicular_bisector_intersects_CQ_at_M B P M) 
: (M.fst^2 / 25 + M.snd^2 / 16 = 1) := 
sorry

end NUMINAMATH_GPT_trajectory_of_M_is_ellipse_l1593_159320


namespace NUMINAMATH_GPT_distinct_strings_after_operations_l1593_159323

def valid_strings (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else valid_strings (n-1) + valid_strings (n-2)

theorem distinct_strings_after_operations :
  valid_strings 10 = 144 := by
  sorry

end NUMINAMATH_GPT_distinct_strings_after_operations_l1593_159323


namespace NUMINAMATH_GPT_equilateral_triangle_area_decrease_l1593_159347

theorem equilateral_triangle_area_decrease (A : ℝ) (A' : ℝ) (s s' : ℝ) 
  (h1 : A = 121 * Real.sqrt 3) 
  (h2 : A = (s^2 * Real.sqrt 3) / 4) 
  (h3 : s' = s - 8) 
  (h4 : A' = (s'^2 * Real.sqrt 3) / 4) :
  A - A' = 72 * Real.sqrt 3 := 
by sorry

end NUMINAMATH_GPT_equilateral_triangle_area_decrease_l1593_159347


namespace NUMINAMATH_GPT_revenue_fraction_large_cups_l1593_159331

theorem revenue_fraction_large_cups (total_cups : ℕ) (price_small : ℚ) (price_large : ℚ)
  (h1 : price_large = (7 / 6) * price_small) 
  (h2 : (1 / 5 : ℚ) * total_cups = total_cups - (4 / 5 : ℚ) * total_cups) :
  ((4 / 5 : ℚ) * (7 / 6 * price_small) * total_cups) / 
  (((1 / 5 : ℚ) * price_small + (4 / 5 : ℚ) * (7 / 6 * price_small)) * total_cups) = (14 / 17 : ℚ) :=
by
  intros
  have h_total_small := (1 / 5 : ℚ) * total_cups
  have h_total_large := (4 / 5 : ℚ) * total_cups
  have revenue_small := h_total_small * price_small
  have revenue_large := h_total_large * price_large
  have total_revenue := revenue_small + revenue_large
  have revenue_large_frac := revenue_large / total_revenue
  have target_frac := (14 / 17 : ℚ)
  have target := revenue_large_frac = target_frac
  sorry

end NUMINAMATH_GPT_revenue_fraction_large_cups_l1593_159331


namespace NUMINAMATH_GPT_multiplication_result_l1593_159337

theorem multiplication_result :
  3^2 * 5^2 * 7 * 11^2 = 190575 :=
by sorry

end NUMINAMATH_GPT_multiplication_result_l1593_159337


namespace NUMINAMATH_GPT_find_a_if_odd_function_l1593_159313

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + Real.sqrt (a + x^2))

theorem find_a_if_odd_function (a : ℝ) :
  (∀ x : ℝ, f (-x) a = - f x a) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_if_odd_function_l1593_159313


namespace NUMINAMATH_GPT_Sara_team_wins_l1593_159361

theorem Sara_team_wins (total_games losses wins : ℕ) (h1 : total_games = 12) (h2 : losses = 4) (h3 : wins = total_games - losses) :
  wins = 8 :=
by
  sorry

end NUMINAMATH_GPT_Sara_team_wins_l1593_159361


namespace NUMINAMATH_GPT_smallest_solution_of_equation_l1593_159365

theorem smallest_solution_of_equation :
  ∃ x : ℝ, (x^4 - 26 * x^2 + 169 = 0) ∧ x = -Real.sqrt 13 :=
by
  sorry

end NUMINAMATH_GPT_smallest_solution_of_equation_l1593_159365


namespace NUMINAMATH_GPT_find_kgs_of_apples_l1593_159314

def cost_of_apples_per_kg : ℝ := 2
def num_packs_of_sugar : ℝ := 3
def cost_of_sugar_per_pack : ℝ := cost_of_apples_per_kg - 1
def weight_walnuts_kg : ℝ := 0.5
def cost_of_walnuts_per_kg : ℝ := 6
def cost_of_walnuts : ℝ := cost_of_walnuts_per_kg * weight_walnuts_kg
def total_cost : ℝ := 16

theorem find_kgs_of_apples (A : ℝ) :
  2 * A + (num_packs_of_sugar * cost_of_sugar_per_pack) + cost_of_walnuts = total_cost →
  A = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_kgs_of_apples_l1593_159314


namespace NUMINAMATH_GPT_sector_radius_l1593_159349

theorem sector_radius (A L : ℝ) (hA : A = 240 * Real.pi) (hL : L = 20 * Real.pi) : 
  ∃ r : ℝ, r = 24 :=
by
  sorry

end NUMINAMATH_GPT_sector_radius_l1593_159349


namespace NUMINAMATH_GPT_total_money_raised_l1593_159355

def tickets_sold : ℕ := 25
def price_per_ticket : ℝ := 2.0
def donation_count : ℕ := 2
def donation_amount : ℝ := 15.0
def additional_donation : ℝ := 20.0

theorem total_money_raised :
  (tickets_sold * price_per_ticket) + (donation_count * donation_amount) + additional_donation = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_money_raised_l1593_159355


namespace NUMINAMATH_GPT_trig_identity_theorem_l1593_159378

noncomputable def trig_identity_proof : Prop :=
  (1 + Real.cos (Real.pi / 9)) * 
  (1 + Real.cos (2 * Real.pi / 9)) * 
  (1 + Real.cos (4 * Real.pi / 9)) * 
  (1 + Real.cos (5 * Real.pi / 9)) = 
  (1 / 2) * (Real.sin (Real.pi / 9))^4

#check trig_identity_proof

theorem trig_identity_theorem : trig_identity_proof := by
  sorry

end NUMINAMATH_GPT_trig_identity_theorem_l1593_159378


namespace NUMINAMATH_GPT_stratified_sampling_BA3_count_l1593_159344

-- Defining the problem parameters
def num_Om_BA1 : ℕ := 60
def num_Om_BA2 : ℕ := 20
def num_Om_BA3 : ℕ := 40
def total_sample_size : ℕ := 30

-- Proving using stratified sampling
theorem stratified_sampling_BA3_count : 
  (total_sample_size * num_Om_BA3 / (num_Om_BA1 + num_Om_BA2 + num_Om_BA3)) = 10 :=
by
  -- Since Lean doesn't handle reals and integers simplistically,
  -- we need to translate the division and multiplication properly.
  sorry

end NUMINAMATH_GPT_stratified_sampling_BA3_count_l1593_159344


namespace NUMINAMATH_GPT_value_of_x_squared_plus_reciprocal_squared_l1593_159364

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (h : 45 = x^4 + 1 / x^4) : 
  x^2 + 1 / x^2 = Real.sqrt 47 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_reciprocal_squared_l1593_159364


namespace NUMINAMATH_GPT_pythagorean_triangle_product_divisible_by_60_l1593_159397

theorem pythagorean_triangle_product_divisible_by_60 : 
  ∀ (a b c : ℕ),
  (∃ m n : ℕ,
  m > n ∧ (m % 2 = 0 ∨ n % 2 = 0) ∧ m.gcd n = 1 ∧
  a = m^2 - n^2 ∧ b = 2 * m * n ∧ c = m^2 + n^2 ∧ a^2 + b^2 = c^2) →
  60 ∣ (a * b * c) :=
sorry

end NUMINAMATH_GPT_pythagorean_triangle_product_divisible_by_60_l1593_159397


namespace NUMINAMATH_GPT_initial_nickels_l1593_159354

variable (q0 n0 : Nat)
variable (d_nickels : Nat := 3) -- His dad gave him 3 nickels
variable (final_nickels : Nat := 12) -- Tim has now 12 nickels

theorem initial_nickels (q0 : Nat) (n0 : Nat) (d_nickels : Nat) (final_nickels : Nat) :
  final_nickels = n0 + d_nickels → n0 = 9 :=
by
  sorry

end NUMINAMATH_GPT_initial_nickels_l1593_159354


namespace NUMINAMATH_GPT_five_b_value_l1593_159357

theorem five_b_value (a b : ℚ) (h1 : 3 * a + 4 * b = 2) (h2 : a = 2 * b - 3) : 5 * b = 5.5 := 
by
  sorry

end NUMINAMATH_GPT_five_b_value_l1593_159357


namespace NUMINAMATH_GPT_convert_radian_to_degree_part1_convert_radian_to_degree_part2_convert_radian_to_degree_part3_convert_degree_to_radian_part1_convert_degree_to_radian_part2_l1593_159379

noncomputable def pi_deg : ℝ := 180 -- Define pi in degrees
notation "°" => pi_deg -- Define a notation for degrees

theorem convert_radian_to_degree_part1 : (π / 12) * (180 / π) = 15 := 
by
  sorry

theorem convert_radian_to_degree_part2 : (13 * π / 6) * (180 / π) = 390 := 
by
  sorry

theorem convert_radian_to_degree_part3 : -(5 / 12) * π * (180 / π) = -75 := 
by
  sorry

theorem convert_degree_to_radian_part1 : 36 * (π / 180) = (π / 5) := 
by
  sorry

theorem convert_degree_to_radian_part2 : -105 * (π / 180) = -(7 * π / 12) := 
by
  sorry

end NUMINAMATH_GPT_convert_radian_to_degree_part1_convert_radian_to_degree_part2_convert_radian_to_degree_part3_convert_degree_to_radian_part1_convert_degree_to_radian_part2_l1593_159379


namespace NUMINAMATH_GPT_distance_between_points_l1593_159368

theorem distance_between_points : abs (3 - (-2)) = 5 := 
by
  sorry

end NUMINAMATH_GPT_distance_between_points_l1593_159368


namespace NUMINAMATH_GPT_complete_square_correct_l1593_159367

theorem complete_square_correct (x : ℝ) : x^2 - 4 * x - 1 = 0 → (x - 2)^2 = 5 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_complete_square_correct_l1593_159367


namespace NUMINAMATH_GPT_simplify_expression_l1593_159374

variable (m : ℕ) (h1 : m ≠ 2) (h2 : m ≠ 3)

theorem simplify_expression : 
  (m - 3) / (2 * m - 4) / (m + 2 - 5 / (m - 2)) = 1 / (2 * m + 6) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1593_159374


namespace NUMINAMATH_GPT_surface_area_of_cube_l1593_159303

theorem surface_area_of_cube (V : ℝ) (H : V = 125) : ∃ A : ℝ, A = 25 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_cube_l1593_159303


namespace NUMINAMATH_GPT_expected_number_of_sixes_l1593_159388

-- Define the probability of not rolling a 6 on one die
def prob_not_six : ℚ := 5 / 6

-- Define the probability of rolling zero 6's on three dice
def prob_zero_six : ℚ := prob_not_six ^ 3

-- Define the probability of rolling exactly one 6 among the three dice
def prob_one_six (n : ℕ) : ℚ := n * (1 / 6) * (prob_not_six ^ (n - 1))

-- Calculate the probabilities of each specific outcomes
def prob_exactly_zero_six : ℚ := prob_zero_six
def prob_exactly_one_six : ℚ := prob_one_six 3 * (prob_not_six ^ 2)
def prob_exactly_two_six : ℚ := prob_one_six 3 * (1 / 6) * prob_not_six
def prob_exactly_three_six : ℚ := (1 / 6) ^ 3

-- Define the expected value calculation
noncomputable def expected_value : ℚ :=
  0 * prob_exactly_zero_six
  + 1 * prob_exactly_one_six
  + 2 * prob_exactly_two_six
  + 3 * prob_exactly_three_six

-- Prove that the expected value equals to 1/2
theorem expected_number_of_sixes : expected_value = 1 / 2 :=
  by
    sorry

end NUMINAMATH_GPT_expected_number_of_sixes_l1593_159388


namespace NUMINAMATH_GPT_remainders_are_distinct_l1593_159326

theorem remainders_are_distinct (a : ℕ → ℕ) (H1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 100 → a i ≠ a (i % 100 + 1))
  (H2 : ∃ r1 r2 : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ 100 → a i % a (i % 100 + 1) = r1 ∨ a i % a (i % 100 + 1) = r2) :
  ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ 100 → (a (i % 100 + 1) % a i) ≠ (a (j % 100 + 1) % a j) :=
by
  sorry

end NUMINAMATH_GPT_remainders_are_distinct_l1593_159326


namespace NUMINAMATH_GPT_total_number_of_feet_l1593_159319

theorem total_number_of_feet 
  (H C F : ℕ)
  (h1 : H + C = 44)
  (h2 : H = 24)
  (h3 : F = 2 * H + 4 * C) : 
  F = 128 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_feet_l1593_159319


namespace NUMINAMATH_GPT_natasha_dimes_l1593_159356

theorem natasha_dimes (n : ℕ) (h1 : 10 < n) (h2 : n < 100) (h3 : n % 3 = 1) (h4 : n % 4 = 1) (h5 : n % 5 = 1) : n = 61 :=
sorry

end NUMINAMATH_GPT_natasha_dimes_l1593_159356


namespace NUMINAMATH_GPT_correct_average_is_18_l1593_159318

theorem correct_average_is_18 (incorrect_avg : ℕ) (incorrect_num : ℕ) (true_num : ℕ) (n : ℕ) 
  (h1 : incorrect_avg = 16) (h2 : incorrect_num = 25) (h3 : true_num = 45) (h4 : n = 10) : 
  (incorrect_avg * n + (true_num - incorrect_num)) / n = 18 :=
by
  sorry

end NUMINAMATH_GPT_correct_average_is_18_l1593_159318


namespace NUMINAMATH_GPT_binom_arithmetic_sequence_l1593_159321

noncomputable def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_arithmetic_sequence {n : ℕ} (h : 2 * binom n 5 = binom n 4 + binom n 6) (n_eq : n = 14) : binom n 12 = 91 := by
  sorry

end NUMINAMATH_GPT_binom_arithmetic_sequence_l1593_159321


namespace NUMINAMATH_GPT_parallelogram_area_increase_l1593_159308

theorem parallelogram_area_increase (b h : ℕ) :
  let A1 := b * h
  let b' := 2 * b
  let h' := 2 * h
  let A2 := b' * h'
  (A2 - A1) * 100 / A1 = 300 :=
by
  let A1 := b * h
  let b' := 2 * b
  let h' := 2 * h
  let A2 := b' * h'
  sorry

end NUMINAMATH_GPT_parallelogram_area_increase_l1593_159308


namespace NUMINAMATH_GPT_graph_of_y_eq_neg2x_passes_quadrant_II_IV_l1593_159352

-- Definitions
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x

def is_in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0

def is_in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- The main statement
theorem graph_of_y_eq_neg2x_passes_quadrant_II_IV :
  ∀ (x : ℝ), (is_in_quadrant_II x (linear_function (-2) x) ∨ 
               is_in_quadrant_IV x (linear_function (-2) x)) :=
by
  sorry

end NUMINAMATH_GPT_graph_of_y_eq_neg2x_passes_quadrant_II_IV_l1593_159352


namespace NUMINAMATH_GPT_no_whole_numbers_satisfy_eqn_l1593_159373

theorem no_whole_numbers_satisfy_eqn :
  ¬ ∃ (x y z : ℤ), (x - y) ^ 3 + (y - z) ^ 3 + (z - x) ^ 3 = 2021 :=
by
  sorry

end NUMINAMATH_GPT_no_whole_numbers_satisfy_eqn_l1593_159373


namespace NUMINAMATH_GPT_lemons_for_lemonade_l1593_159390

theorem lemons_for_lemonade (lemons_gallons_ratio : 30 / 25 = x / 10) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_lemons_for_lemonade_l1593_159390


namespace NUMINAMATH_GPT_sum_of_abs_values_l1593_159385

-- Define the problem conditions
variable (a b c d m : ℤ)
variable (h1 : a + b + c + d = 1)
variable (h2 : a * b + a * c + a * d + b * c + b * d + c * d = 0)
variable (h3 : a * b * c + a * b * d + a * c * d + b * c * d = -4023)
variable (h4 : a * b * c * d = m)

-- Prove the required sum of absolute values
theorem sum_of_abs_values : |a| + |b| + |c| + |d| = 621 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_abs_values_l1593_159385


namespace NUMINAMATH_GPT_thomas_total_drawings_l1593_159301

theorem thomas_total_drawings :
  let colored_pencil_drawings := 14
  let blending_marker_drawings := 7
  let charcoal_drawings := 4
  colored_pencil_drawings + blending_marker_drawings + charcoal_drawings = 25 := 
by
  sorry

end NUMINAMATH_GPT_thomas_total_drawings_l1593_159301


namespace NUMINAMATH_GPT_guitar_center_discount_is_correct_l1593_159317

-- Define the suggested retail price
def retail_price : ℕ := 1000

-- Define the shipping fee of Guitar Center
def shipping_fee : ℕ := 100

-- Define the discount percentage offered by Sweetwater
def sweetwater_discount_rate : ℕ := 10

-- Define the amount saved by buying from the cheaper store
def savings : ℕ := 50

-- Define the discount offered by Guitar Center
def guitar_center_discount : ℕ :=
  retail_price - ((retail_price * (100 - sweetwater_discount_rate) / 100) + savings - shipping_fee)

-- Theorem: Prove that the discount offered by Guitar Center is $150
theorem guitar_center_discount_is_correct : guitar_center_discount = 150 :=
  by
    -- The proof will be filled in based on the given conditions
    sorry

end NUMINAMATH_GPT_guitar_center_discount_is_correct_l1593_159317


namespace NUMINAMATH_GPT_equation_of_line_projection_l1593_159325

theorem equation_of_line_projection (x y : ℝ) (m : ℝ) (x1 x2 : ℝ) (d : ℝ)
  (h1 : (5, 3) ∈ {(x, y) | y = 3 + m * (x - 5)})
  (h2 : x1 = (16 + 20 * m - 12) / (4 * m + 3))
  (h3 : x2 = (1 + 20 * m - 12) / (4 * m + 3))
  (h4 : abs (x1 - x2) = 1) :
  (y = 3 * x - 12 ∨ y = -4.5 * x + 25.5) :=
sorry

end NUMINAMATH_GPT_equation_of_line_projection_l1593_159325


namespace NUMINAMATH_GPT_selling_price_of_book_l1593_159370

   theorem selling_price_of_book
     (cost_price : ℝ)
     (profit_rate : ℝ)
     (profit := (profit_rate / 100) * cost_price)
     (selling_price := cost_price + profit)
     (hp : cost_price = 50)
     (hr : profit_rate = 60) :
     selling_price = 80 := sorry
   
end NUMINAMATH_GPT_selling_price_of_book_l1593_159370
