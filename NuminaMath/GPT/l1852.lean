import Mathlib

namespace sandy_gave_puppies_l1852_185237

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

end sandy_gave_puppies_l1852_185237


namespace Cameron_task_completion_l1852_185296

theorem Cameron_task_completion (C : ℝ) (h1 : ∃ x, x = 9 / C) (h2 : ∃ y, y = 1 / 2) (total_work : ∃ z, z = 1):
  9 - 9 / C + 1/2 = 1 -> C = 18 := by
  sorry

end Cameron_task_completion_l1852_185296


namespace p_distance_300_l1852_185273

-- Assume q's speed is v meters per second, and the race ends in a tie
variables (v : ℝ) (t : ℝ)
variable (d : ℝ)

-- Conditions
def q_speed : ℝ := v
def p_speed : ℝ := 1.25 * v
def q_distance : ℝ := d
def p_distance : ℝ := d + 60

-- Time equations
def q_time_eq : Prop := d = v * t
def p_time_eq : Prop := d + 60 = (1.25 * v) * t

-- Given the conditions, prove that p ran 300 meters in the race
theorem p_distance_300
  (v_pos : v > 0) 
  (t_pos : t > 0)
  (q_time : q_time_eq v d t)
  (p_time : p_time_eq v d t) :
  p_distance d = 300 :=
by
  sorry

end p_distance_300_l1852_185273


namespace tan_eq_2sqrt3_over_3_l1852_185215

theorem tan_eq_2sqrt3_over_3 (θ : ℝ) (h : 2 * Real.cos (θ - Real.pi / 3) = 3 * Real.cos θ) : 
  Real.tan θ = 2 * Real.sqrt 3 / 3 :=
by 
  sorry -- Proof is omitted as per the instructions

end tan_eq_2sqrt3_over_3_l1852_185215


namespace total_cans_l1852_185210

theorem total_cans (c o : ℕ) (h1 : c = 8) (h2 : o = 2 * c) : c + o = 24 := by
  sorry

end total_cans_l1852_185210


namespace circle_radius_5_l1852_185245

theorem circle_radius_5 (k : ℝ) : 
  (∃ x y : ℝ, x^2 + 14 * x + y^2 + 8 * y - k = 0) ↔ k = -40 :=
by
  sorry

end circle_radius_5_l1852_185245


namespace average_speed_l1852_185295

theorem average_speed (v1 v2 : ℝ) (h1 : v1 = 110) (h2 : v2 = 88) : 
  (2 * v1 * v2) / (v1 + v2) = 97.78 := 
by sorry

end average_speed_l1852_185295


namespace function_satisfies_equation_l1852_185297

noncomputable def f (x : ℝ) : ℝ := x + 1 / x + 1 / (x - 1)

theorem function_satisfies_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  f ((x - 1) / x) + f (1 / (1 - x)) = 2 - 2 * x := by
  sorry

end function_satisfies_equation_l1852_185297


namespace existence_of_indices_l1852_185242

theorem existence_of_indices 
  (a1 a2 a3 a4 a5 : ℝ) 
  (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) (h4 : 0 < a4) (h5 : 0 < a5) : 
  ∃ (i j k l : Fin 5), 
    (i ≠ j) ∧ (i ≠ k) ∧ (i ≠ l) ∧ (j ≠ k) ∧ (j ≠ l) ∧ (k ≠ l) ∧ 
    |(a1 / a2) - (a3 / a4)| < 1/2 :=
by 
  sorry

end existence_of_indices_l1852_185242


namespace cube_faces_one_third_blue_l1852_185240

theorem cube_faces_one_third_blue (n : ℕ) (h1 : ∃ n, n > 0 ∧ (6 * n^2) / (6 * n^3) = 1 / 3) : n = 3 := by
  sorry

end cube_faces_one_third_blue_l1852_185240


namespace elmer_saves_21_875_percent_l1852_185241

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

end elmer_saves_21_875_percent_l1852_185241


namespace intersection_points_count_l1852_185258

-- Definition of the two equations as conditions
def eq1 (x y : ℝ) : Prop := y = 3 * x^2
def eq2 (x y : ℝ) : Prop := y^2 - 6 * y + 8 = x^2

-- The theorem stating that the number of intersection points of the two graphs is exactly 4
theorem intersection_points_count : 
  ∃ (points : Finset (ℝ × ℝ)), (∀ p : ℝ × ℝ, p ∈ points ↔ eq1 p.1 p.2 ∧ eq2 p.1 p.2) ∧ points.card = 4 :=
by
  sorry

end intersection_points_count_l1852_185258


namespace find_x_plus_y_l1852_185233

theorem find_x_plus_y (x y : ℝ) (h1 : |x| - x + y = 13) (h2 : x - |y| + y = 7) : x + y = 20 := 
by
  sorry

end find_x_plus_y_l1852_185233


namespace downstream_speed_l1852_185252

noncomputable def V_b : ℝ := 7
noncomputable def V_up : ℝ := 4
noncomputable def V_s : ℝ := V_b - V_up

theorem downstream_speed :
  V_b + V_s = 10 := sorry

end downstream_speed_l1852_185252


namespace ratio_of_boxes_sold_l1852_185214

-- Definitions for conditions
variables (T W Tu : ℕ)

-- Define the conditions as hypotheses
def conditions : Prop :=
  W = 2 * T ∧
  Tu = 2 * W ∧
  T = 1200

-- The statement to prove the ratio Tu / W = 2
theorem ratio_of_boxes_sold (T W Tu : ℕ) (h : conditions T W Tu) :
  Tu / W = 2 :=
by
  sorry

end ratio_of_boxes_sold_l1852_185214


namespace seeds_in_small_gardens_l1852_185208

theorem seeds_in_small_gardens 
  (total_seeds : ℕ)
  (planted_seeds : ℕ)
  (small_gardens : ℕ)
  (remaining_seeds := total_seeds - planted_seeds) 
  (seeds_per_garden := remaining_seeds / small_gardens) :
  total_seeds = 101 → planted_seeds = 47 → small_gardens = 9 → seeds_per_garden = 6 := by
  sorry

end seeds_in_small_gardens_l1852_185208


namespace solve_eq_log_base_l1852_185227

theorem solve_eq_log_base (x : ℝ) : (9 : ℝ)^(x+8) = (10 : ℝ)^x → x = Real.logb (10 / 9) ((9 : ℝ)^8) := by
  intro h
  sorry

end solve_eq_log_base_l1852_185227


namespace find_ratio_l1852_185292

open Nat

def sequence_def (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 →
    (a ((n + 2)) / a ((n + 1))) - (a ((n + 1)) / a n) = d

def geometric_difference_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3 ∧ sequence_def a 2

theorem find_ratio (a : ℕ → ℕ) (h : geometric_difference_sequence a) :
  a 12 / a 10 = 399 := sorry

end find_ratio_l1852_185292


namespace least_possible_faces_two_dice_l1852_185270

noncomputable def least_possible_sum_of_faces (a b : ℕ) : ℕ :=
(a + b)

theorem least_possible_faces_two_dice (a b : ℕ) (h1 : 8 ≤ a) (h2 : 8 ≤ b)
  (h3 : ∃ k, 9 * k = 2 * (11 * k)) 
  (h4 : ∃ m, 9 * m = a * b) : 
  least_possible_sum_of_faces a b = 22 :=
sorry

end least_possible_faces_two_dice_l1852_185270


namespace series_sum_eq_half_l1852_185276

theorem series_sum_eq_half : ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end series_sum_eq_half_l1852_185276


namespace contrapositive_of_quadratic_l1852_185293

theorem contrapositive_of_quadratic (m : ℝ) :
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ↔ (¬∃ x : ℝ, x^2 + x - m = 0 → m ≤ 0) :=
by
  sorry

end contrapositive_of_quadratic_l1852_185293


namespace tan_beta_minus_2alpha_l1852_185249

theorem tan_beta_minus_2alpha
  (α β : ℝ)
  (h1 : Real.tan α = 1/2)
  (h2 : Real.tan (α - β) = -1/3) :
  Real.tan (β - 2 * α) = -1/7 := 
sorry

end tan_beta_minus_2alpha_l1852_185249


namespace minimum_sum_of_natural_numbers_with_lcm_2012_l1852_185219

/-- 
Prove that the minimum sum of seven natural numbers whose least common multiple is 2012 is 512.
-/

theorem minimum_sum_of_natural_numbers_with_lcm_2012 : 
  ∃ (a b c d e f g : ℕ), Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm a b) c) d) e) f) g = 2012 ∧ (a + b + c + d + e + f + g) = 512 :=
sorry

end minimum_sum_of_natural_numbers_with_lcm_2012_l1852_185219


namespace selling_price_of_book_l1852_185243

   theorem selling_price_of_book
     (cost_price : ℝ)
     (profit_rate : ℝ)
     (profit := (profit_rate / 100) * cost_price)
     (selling_price := cost_price + profit)
     (hp : cost_price = 50)
     (hr : profit_rate = 60) :
     selling_price = 80 := sorry
   
end selling_price_of_book_l1852_185243


namespace seventy_five_percent_of_number_l1852_185253

variable (N : ℝ)

theorem seventy_five_percent_of_number :
  (1 / 8) * (3 / 5) * (4 / 7) * (5 / 11) * N - (1 / 9) * (2 / 3) * (3 / 4) * (5 / 8) * N = 30 →
  0.75 * N = -1476 :=
by
  sorry

end seventy_five_percent_of_number_l1852_185253


namespace cameron_answers_l1852_185209

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

end cameron_answers_l1852_185209


namespace find_PF2_l1852_185299

open Real

noncomputable def hyperbola_equation (x y : ℝ) := (x^2 / 16) - (y^2 / 20) = 1

noncomputable def distance (P F : ℝ × ℝ) : ℝ := 
  let (px, py) := P
  let (fx, fy) := F
  sqrt ((px - fx)^2 + (py - fy)^2)

theorem find_PF2
  (P : ℝ × ℝ)
  (F1 F2 : ℝ × ℝ)
  (on_hyperbola : hyperbola_equation P.1 P.2)
  (foci_F1_F2 : F1 = (-6, 0) ∧ F2 = (6, 0))
  (distance_PF1 : distance P F1 = 9) : 
  distance P F2 = 17 := 
by
  sorry

end find_PF2_l1852_185299


namespace frisbee_price_l1852_185298

theorem frisbee_price 
  (total_frisbees : ℕ)
  (frisbees_at_3 : ℕ)
  (price_x_frisbees : ℕ)
  (total_revenue : ℕ) 
  (min_frisbees_at_x : ℕ)
  (price_at_3 : ℕ) 
  (n_min_at_x : ℕ)
  (h1 : total_frisbees = 60)
  (h2 : price_at_3 = 3)
  (h3 : total_revenue = 200)
  (h4 : n_min_at_x = 20)
  (h5 : min_frisbees_at_x >= n_min_at_x)
  : price_x_frisbees = 4 :=
by
  sorry

end frisbee_price_l1852_185298


namespace tickets_savings_percentage_l1852_185277

theorem tickets_savings_percentage (P S : ℚ) (h : 8 * S = 5 * P) :
  (12 * P - 12 * S) / (12 * P) * 100 = 37.5 :=
by 
  sorry

end tickets_savings_percentage_l1852_185277


namespace min_distance_parabola_l1852_185256

open Real

theorem min_distance_parabola {P : ℝ × ℝ} (hP : P.2^2 = 4 * P.1) : ∃ m : ℝ, m = 2 * sqrt 3 ∧ ∀ Q : ℝ × ℝ, Q = (4, 0) → dist P Q ≥ m :=
by sorry

end min_distance_parabola_l1852_185256


namespace total_days_on_jury_duty_l1852_185261

-- Definitions based on conditions
def jurySelectionDays := 2
def juryDeliberationDays := 6
def deliberationHoursPerDay := 16
def trialDurationMultiplier := 4

-- Calculate the total number of hours spent in deliberation
def totalDeliberationHours := juryDeliberationDays * 24

-- Calculate the number of days spent in deliberation based on hours per day
def deliberationDays := totalDeliberationHours / deliberationHoursPerDay

-- Calculate the trial days based on trial duration multiplier
def trialDays := jurySelectionDays * trialDurationMultiplier

-- Calculate the total days on jury duty
def totalJuryDutyDays := jurySelectionDays + trialDays + deliberationDays

theorem total_days_on_jury_duty : totalJuryDutyDays = 19 := by
  sorry

end total_days_on_jury_duty_l1852_185261


namespace probability_AEMC9_is_1_over_84000_l1852_185221

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

end probability_AEMC9_is_1_over_84000_l1852_185221


namespace distance_between_lines_l1852_185279

-- Define lines l1 and l2
def line_l1 (x y : ℝ) := x + y + 1 = 0
def line_l2 (x y : ℝ) := 2 * x + 2 * y + 3 = 0

-- Proof statement for the distance between parallel lines
theorem distance_between_lines :
  let a := 1
  let b := 1
  let c1 := 1
  let c2 := 3 / 2
  let distance := |c2 - c1| / (Real.sqrt (a^2 + b^2))
  distance = Real.sqrt 2 / 4 :=
by
  sorry

end distance_between_lines_l1852_185279


namespace second_pipe_fills_in_15_minutes_l1852_185204

theorem second_pipe_fills_in_15_minutes :
  ∀ (x : ℝ),
  (∀ (x : ℝ), (1 / 2 + (7.5 / x)) = 1 → x = 15) :=
by
  intros
  sorry

end second_pipe_fills_in_15_minutes_l1852_185204


namespace sum_of_intercepts_l1852_185286

theorem sum_of_intercepts (x₀ y₀ : ℕ) (hx₀ : 4 * x₀ ≡ 2 [MOD 25]) (hy₀ : 5 * y₀ ≡ 23 [MOD 25]) 
  (hx_cond : x₀ < 25) (hy_cond : y₀ < 25) : x₀ + y₀ = 28 :=
  sorry

end sum_of_intercepts_l1852_185286


namespace mrs_jackson_decorations_l1852_185202

theorem mrs_jackson_decorations (boxes decorations_in_each_box decorations_used : Nat) 
  (h1 : boxes = 4) 
  (h2 : decorations_in_each_box = 15) 
  (h3 : decorations_used = 35) :
  boxes * decorations_in_each_box - decorations_used = 25 := 
  by
  sorry

end mrs_jackson_decorations_l1852_185202


namespace distance_between_x_intercepts_l1852_185262

theorem distance_between_x_intercepts :
  ∀ (x1 x2 : ℝ),
  (∀ x, x1 = 8 → x2 = 20 → 20 = 4 * (x - 8)) → 
  (∀ x, x1 = 8 → x2 = 20 → 20 = 7 * (x - 8)) → 
  abs ((3 : ℝ) - (36 / 7)) = (15 / 7) :=
by
  intros x1 x2 h1 h2
  sorry

end distance_between_x_intercepts_l1852_185262


namespace initial_bananas_proof_l1852_185207

noncomputable def initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) : ℕ :=
  (extra_bananas * (total_children - absent_children)) / (total_children - extra_bananas)

theorem initial_bananas_proof
  (total_children : ℕ)
  (absent_children : ℕ)
  (extra_bananas : ℕ)
  (h_total : total_children = 640)
  (h_absent : absent_children = 320)
  (h_extra : extra_bananas = 2) : initial_bananas_per_child total_children absent_children extra_bananas = 2 :=
by
  sorry

end initial_bananas_proof_l1852_185207


namespace blankets_warmth_increase_l1852_185290

-- Conditions
def blankets_in_closet : ℕ := 14
def blankets_used : ℕ := blankets_in_closet / 2
def degree_per_blanket : ℕ := 3

-- Goal: Prove that the total temperature increase is 21 degrees.
theorem blankets_warmth_increase : blankets_used * degree_per_blanket = 21 :=
by
  sorry

end blankets_warmth_increase_l1852_185290


namespace sum_sequence_up_to_2015_l1852_185288

def sequence_val (n : ℕ) : ℕ :=
  if n % 288 = 0 then 7 
  else if n % 224 = 0 then 9
  else if n % 63 = 0 then 32
  else 0

theorem sum_sequence_up_to_2015 : 
  (Finset.range 2016).sum sequence_val = 1106 :=
by
  sorry

end sum_sequence_up_to_2015_l1852_185288


namespace fort_blocks_count_l1852_185255

noncomputable def volume_of_blocks (l w h : ℕ) (wall_thickness floor_thickness top_layer_volume : ℕ) : ℕ :=
  let interior_length := l - 2 * wall_thickness
  let interior_width := w - 2 * wall_thickness
  let interior_height := h - floor_thickness
  let volume_original := l * w * h
  let volume_interior := interior_length * interior_width * interior_height
  volume_original - volume_interior + top_layer_volume

theorem fort_blocks_count : volume_of_blocks 15 12 7 2 1 180 = 912 :=
by
  sorry

end fort_blocks_count_l1852_185255


namespace horner_evaluation_l1852_185234

-- Define the polynomial function
def f (x : ℝ) : ℝ := 4 * x^4 + 3 * x^3 - 6 * x^2 + x - 1

-- The theorem that we need to prove
theorem horner_evaluation : f (-1) = -5 :=
  by
  -- This is the statement without the proof steps
  sorry

end horner_evaluation_l1852_185234


namespace constant_term_is_19_l1852_185213

theorem constant_term_is_19 (x y C : ℝ) 
  (h1 : 7 * x + y = C) 
  (h2 : x + 3 * y = 1) 
  (h3 : 2 * x + y = 5) : 
  C = 19 :=
sorry

end constant_term_is_19_l1852_185213


namespace length_of_train_l1852_185291

theorem length_of_train
  (speed_kmph : ℝ)
  (platform_length : ℝ)
  (crossing_time : ℝ)
  (train_speed_mps : ℝ := speed_kmph * (1000 / 3600))
  (total_distance : ℝ := train_speed_mps * crossing_time)
  (train_length : ℝ := total_distance - platform_length)
  (h_speed : speed_kmph = 72)
  (h_platform : platform_length = 260)
  (h_time : crossing_time = 26)
  : train_length = 260 := by
  sorry

end length_of_train_l1852_185291


namespace find_x_plus_y_l1852_185266

theorem find_x_plus_y (x y : ℝ) (h1 : |x| + x + y = 16) (h2 : x + |y| - y = 18) : x + y = 6 := 
sorry

end find_x_plus_y_l1852_185266


namespace four_consecutive_integers_plus_one_is_square_l1852_185282

theorem four_consecutive_integers_plus_one_is_square (n : ℤ) : 
  (n - 1) * n * (n + 1) * (n + 2) + 1 = (n ^ 2 + n - 1) ^ 2 := 
by 
  sorry

end four_consecutive_integers_plus_one_is_square_l1852_185282


namespace work_completion_days_l1852_185275

theorem work_completion_days (A B C : ℕ) 
  (hA : A = 4) (hB : B = 8) (hC : C = 8) : 
  2 = 1 / (1 / A + 1 / B + 1 / C) :=
by
  -- skip the proof for now
  sorry

end work_completion_days_l1852_185275


namespace factor_polynomial_l1852_185283

theorem factor_polynomial : 
  (x : ℝ) → (x^2 - 6 * x + 9 - 49 * x^4) = (-7 * x^2 + x - 3) * (7 * x^2 + x - 3) :=
by
  sorry

end factor_polynomial_l1852_185283


namespace new_person_weight_l1852_185260

-- Define the total number of persons and their average weight increase
def num_persons : ℕ := 9
def avg_increase : ℝ := 1.5

-- Define the weight of the person being replaced
def weight_of_replaced_person : ℝ := 65

-- Define the total increase in weight
def total_increase_in_weight : ℝ := num_persons * avg_increase

-- Define the weight of the new person
def weight_of_new_person : ℝ := weight_of_replaced_person + total_increase_in_weight

-- Theorem to prove the weight of the new person is 78.5 kg
theorem new_person_weight : weight_of_new_person = 78.5 := by
  -- proof is omitted
  sorry

end new_person_weight_l1852_185260


namespace mass_of_sodium_acetate_formed_l1852_185231

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

end mass_of_sodium_acetate_formed_l1852_185231


namespace simplify_expression_l1852_185205

variable (m : ℕ) (h1 : m ≠ 2) (h2 : m ≠ 3)

theorem simplify_expression : 
  (m - 3) / (2 * m - 4) / (m + 2 - 5 / (m - 2)) = 1 / (2 * m + 6) :=
by sorry

end simplify_expression_l1852_185205


namespace max_square_area_in_rhombus_l1852_185257

noncomputable def side_length_triangle := 10
noncomputable def height_triangle := Real.sqrt (side_length_triangle^2 - (side_length_triangle / 2)^2)
noncomputable def diag_long := 2 * height_triangle
noncomputable def diag_short := side_length_triangle
noncomputable def side_square := diag_short / Real.sqrt 2
noncomputable def area_square := side_square^2

theorem max_square_area_in_rhombus :
  area_square = 50 := by sorry

end max_square_area_in_rhombus_l1852_185257


namespace scientific_notation_of_number_l1852_185289

theorem scientific_notation_of_number :
  1214000 = 1.214 * 10^6 :=
by
  sorry

end scientific_notation_of_number_l1852_185289


namespace train_pass_platform_time_l1852_185239

theorem train_pass_platform_time (l v t : ℝ) (h1 : v = l / t) (h2 : l > 0) (h3 : t > 0) :
  ∃ T : ℝ, T = 3.5 * t := by
  sorry

end train_pass_platform_time_l1852_185239


namespace kittens_more_than_twice_puppies_l1852_185216

-- Define the number of puppies
def num_puppies : ℕ := 32

-- Define the number of kittens
def num_kittens : ℕ := 78

-- Define the problem statement
theorem kittens_more_than_twice_puppies :
  num_kittens = 2 * num_puppies + 14 :=
by sorry

end kittens_more_than_twice_puppies_l1852_185216


namespace find_a5_l1852_185281

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (a1 : ℝ)

-- Geometric sequence definition
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
∀ (n : ℕ), a (n + 1) = a1 * q^n

-- Given conditions
def condition1 (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
a 1 + a 3 = 10

def condition2 (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
a 2 + a 4 = -30

-- Theorem to prove
theorem find_a5 (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ)
  (h1 : geometric_sequence a a1 q)
  (h2 : condition1 a a1 q)
  (h3 : condition2 a a1 q) :
  a 5 = 81 := by
  sorry

end find_a5_l1852_185281


namespace magnitude_of_a_plus_b_l1852_185218

open Real

noncomputable def magnitude (x y : ℝ) : ℝ :=
  sqrt (x^2 + y^2)

theorem magnitude_of_a_plus_b (m : ℝ) (a b : ℝ × ℝ)
  (h₁ : a = (m+2, 1))
  (h₂ : b = (1, -2*m))
  (h₃ : (a.1 * b.1 + a.2 * b.2 = 0)) :
  magnitude (a.1 + b.1) (a.2 + b.2) = sqrt 34 :=
by
  sorry

end magnitude_of_a_plus_b_l1852_185218


namespace household_waste_per_day_l1852_185217

theorem household_waste_per_day (total_waste_4_weeks : ℝ) (h : total_waste_4_weeks = 30.8) : 
  (total_waste_4_weeks / 4 / 7) = 1.1 :=
by
  sorry

end household_waste_per_day_l1852_185217


namespace complete_square_correct_l1852_185244

theorem complete_square_correct (x : ℝ) : x^2 - 4 * x - 1 = 0 → (x - 2)^2 = 5 :=
by 
  intro h
  sorry

end complete_square_correct_l1852_185244


namespace inequality_solution_set_l1852_185294

theorem inequality_solution_set (x : ℝ) :
  (1 / |x - 1| > 3 / 2) ↔ (1 / 3 < x ∧ x < 5 / 3 ∧ x ≠ 1) :=
by
  sorry

end inequality_solution_set_l1852_185294


namespace monotonic_increasing_intervals_l1852_185203

noncomputable def f (x : ℝ) : ℝ := (Real.cos (x - Real.pi / 6))^2

theorem monotonic_increasing_intervals (k : ℤ) : 
  ∃ t : Set ℝ, t = Set.Ioo (-Real.pi / 3 + k * Real.pi) (Real.pi / 6 + k * Real.pi) ∧ 
    ∀ x y, x ∈ t → y ∈ t → x ≤ y → f x ≤ f y :=
sorry

end monotonic_increasing_intervals_l1852_185203


namespace find_value_of_s_l1852_185259

variable {r s : ℝ}

theorem find_value_of_s (hr : r > 1) (hs : s > 1) (h1 : 1/r + 1/s = 1) (h2 : r * s = 9) :
  s = (9 + 3 * Real.sqrt 5) / 2 :=
sorry

end find_value_of_s_l1852_185259


namespace junior_average_score_l1852_185274

def total_students : ℕ := 20
def proportion_juniors : ℝ := 0.2
def proportion_seniors : ℝ := 0.8
def average_class_score : ℝ := 78
def average_senior_score : ℝ := 75

theorem junior_average_score :
  let num_juniors := total_students * proportion_juniors
  let num_seniors := total_students * proportion_seniors
  let total_score := total_students * average_class_score
  let total_senior_score := num_seniors * average_senior_score
  let total_junior_score := total_score - total_senior_score
  total_junior_score / num_juniors = 90 := 
by
  sorry

end junior_average_score_l1852_185274


namespace find_number_l1852_185211

theorem find_number (x : ℕ) (h : x / 3 = 3) : x = 9 :=
sorry

end find_number_l1852_185211


namespace boundary_length_is_25_point_7_l1852_185248

-- Define the side length derived from the given area.
noncomputable def sideLength (area : ℝ) : ℝ :=
  Real.sqrt area

-- Define the length of each segment when the square's side is divided into four equal parts.
noncomputable def segmentLength (side : ℝ) : ℝ :=
  side / 4

-- Define the total boundary length, which includes the circumference of the quarter-circle arcs and the straight segments.
noncomputable def totalBoundaryLength (area : ℝ) : ℝ :=
  let side := sideLength area
  let segment := segmentLength side
  let arcsLength := 2 * Real.pi * segment  -- the full circle's circumference
  let straightLength := 4 * segment
  arcsLength + straightLength

-- State the theorem that the total boundary length is approximately 25.7 units.
theorem boundary_length_is_25_point_7 :
  totalBoundaryLength 100 = 5 * Real.pi + 10 :=
by sorry

end boundary_length_is_25_point_7_l1852_185248


namespace jerrys_age_l1852_185264

theorem jerrys_age (M J : ℕ) (h1 : M = 3 * J - 4) (h2 : M = 14) : J = 6 :=
by 
  sorry

end jerrys_age_l1852_185264


namespace speed_of_first_train_l1852_185232

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

end speed_of_first_train_l1852_185232


namespace smallest_of_three_consecutive_l1852_185278

theorem smallest_of_three_consecutive (x : ℤ) (h : x + (x + 1) + (x + 2) = 90) : x = 29 :=
by
  sorry

end smallest_of_three_consecutive_l1852_185278


namespace geometric_sequence_general_term_l1852_185246

noncomputable def a_n (n : ℕ) : ℝ := 1 * (2:ℝ)^(n-1)

theorem geometric_sequence_general_term : 
  ∀ (n : ℕ), 
  (∀ (n : ℕ), 0 < a_n n) ∧ a_n 1 = 1 ∧ (a_n 1 + a_n 2 + a_n 3 = 7) → 
  a_n n = 2^(n-1) :=
by
  sorry

end geometric_sequence_general_term_l1852_185246


namespace cubes_with_red_face_l1852_185230

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

end cubes_with_red_face_l1852_185230


namespace sum_of_coefficients_l1852_185267

theorem sum_of_coefficients (a : Fin 7 → ℕ) (x : ℕ) : 
  (1 - x) ^ 6 = (a 0) + (a 1) * x + (a 2) * x^2 + (a 3) * x^3 + (a 4) * x^4 + (a 5) * x^5 + (a 6) * x^6 → 
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 0 := 
by
  intro h
  by_cases hx : x = 1
  · rw [hx] at h
    sorry
  · sorry

end sum_of_coefficients_l1852_185267


namespace pipe_fill_rate_l1852_185254

variable (R_A R_B : ℝ)

theorem pipe_fill_rate :
  R_A = 1 / 32 →
  R_A + R_B = 1 / 6.4 →
  R_B / R_A = 4 :=
by
  intros hRA hSum
  have hRA_pos : R_A ≠ 0 := by linarith
  sorry

end pipe_fill_rate_l1852_185254


namespace office_person_count_l1852_185206

theorem office_person_count
    (N : ℕ)
    (avg_age_all : ℕ)
    (num_5 : ℕ)
    (avg_age_5 : ℕ)
    (num_9 : ℕ)
    (avg_age_9 : ℕ)
    (age_15th : ℕ)
    (h1 : avg_age_all = 15)
    (h2 : num_5 = 5)
    (h3 : avg_age_5 = 14)
    (h4 : num_9 = 9)
    (h5 : avg_age_9 = 16)
    (h6 : age_15th = 86)
    (h7 : 15 * N = (num_5 * avg_age_5) + (num_9 * avg_age_9) + age_15th) :
    N = 20 :=
by
    -- Proof will be provided here
    sorry

end office_person_count_l1852_185206


namespace expected_number_of_sixes_l1852_185236

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

end expected_number_of_sixes_l1852_185236


namespace even_function_a_value_l1852_185200

theorem even_function_a_value (a : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = (x + 1) * (x - a))
  (h_even : ∀ x, f x = f (-x)) : a = -1 :=
by
  sorry

end even_function_a_value_l1852_185200


namespace sqrt_36_eq_6_cube_root_neg_a_125_l1852_185212

theorem sqrt_36_eq_6 : ∀ (x : ℝ), 0 ≤ x ∧ x^2 = 36 → x = 6 :=
by sorry

theorem cube_root_neg_a_125 : ∀ (a y : ℝ), y^3 = - a / 125 → y = - (a^(1/3)) / 5 :=
by sorry

end sqrt_36_eq_6_cube_root_neg_a_125_l1852_185212


namespace bulbs_on_perfect_squares_l1852_185271

def is_on (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k * k

theorem bulbs_on_perfect_squares (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 100) :
  (∀ i : ℕ, 1 ≤ i → i ≤ 100 → ∃ j : ℕ, i = j * j ↔ is_on i) := sorry

end bulbs_on_perfect_squares_l1852_185271


namespace no_all_blue_possible_l1852_185220

-- Define initial counts of chameleons
def initial_red : ℕ := 25
def initial_green : ℕ := 12
def initial_blue : ℕ := 8

-- Define the invariant condition
def invariant (r g : ℕ) : Prop := (r - g) % 3 = 1

-- Define the main theorem statement
theorem no_all_blue_possible : ¬∃ r g, r = 0 ∧ g = 0 ∧ invariant r g :=
by {
  sorry
}

end no_all_blue_possible_l1852_185220


namespace segments_form_quadrilateral_l1852_185201

theorem segments_form_quadrilateral (a d : ℝ) (h_pos : a > 0 ∧ d > 0) (h_sum : 4 * a + 6 * d = 3) : 
  (∃ s1 s2 s3 s4 : ℝ, s1 + s2 + s3 > s4 ∧ s1 + s2 + s4 > s3 ∧ s1 + s3 + s4 > s2 ∧ s2 + s3 + s4 > s1) :=
sorry

end segments_form_quadrilateral_l1852_185201


namespace trig_identity_theorem_l1852_185223

noncomputable def trig_identity_proof : Prop :=
  (1 + Real.cos (Real.pi / 9)) * 
  (1 + Real.cos (2 * Real.pi / 9)) * 
  (1 + Real.cos (4 * Real.pi / 9)) * 
  (1 + Real.cos (5 * Real.pi / 9)) = 
  (1 / 2) * (Real.sin (Real.pi / 9))^4

#check trig_identity_proof

theorem trig_identity_theorem : trig_identity_proof := by
  sorry

end trig_identity_theorem_l1852_185223


namespace four_digit_number_l1852_185251

def digit_constraint (A B C D : ℕ) : Prop :=
  A = B / 3 ∧ C = A + B ∧ D = 3 * B

theorem four_digit_number 
  (A B C D : ℕ) 
  (h₁ : A = B / 3) 
  (h₂ : C = A + B) 
  (h₃ : D = 3 * B)
  (hA_digit : A < 10) 
  (hB_digit : B < 10)
  (hC_digit : C < 10)
  (hD_digit : D < 10) :
  1000 * A + 100 * B + 10 * C + D = 1349 := 
sorry

end four_digit_number_l1852_185251


namespace math_problem_l1852_185250

-- Conditions
def ellipse_eq (a b : ℝ) : Prop := ∀ x y : ℝ, x^2 / (a^2) + y^2 / (b^2) = 1
def eccentricity (a c : ℝ) : Prop := c / a = (Real.sqrt 2) / 2
def major_axis_length (a : ℝ) : Prop := 2 * a = 6 * Real.sqrt 2

-- Equations and properties to be proven
def ellipse_equation : Prop := ∃ a b : ℝ, a = 3 * Real.sqrt 2 ∧ b = 3 ∧ ellipse_eq a b
def length_AB (θ : ℝ) : Prop := ∃ AB : ℝ, AB = (6 * Real.sqrt 2) / (1 + (Real.sin θ)^2)
def min_AB_CD : Prop := ∃ θ : ℝ, (Real.sin (2 * θ) = 1) ∧ (6 * Real.sqrt 2) / (1 + (Real.sin θ)^2) + (6 * Real.sqrt 2) / (1 + (Real.cos θ)^2) = 8 * Real.sqrt 2

-- The complete proof problem
theorem math_problem : ellipse_equation ∧
                       (∀ θ : ℝ, length_AB θ) ∧
                       min_AB_CD := by
  sorry

end math_problem_l1852_185250


namespace prove_by_contradiction_l1852_185263

-- Statement: To prove "a > b" by contradiction, assuming the negation "a ≤ b".
theorem prove_by_contradiction (a b : ℝ) (h : a ≤ b) : false := sorry

end prove_by_contradiction_l1852_185263


namespace f_five_l1852_185226

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = - f x
axiom f_one : f 1 = 1 / 2
axiom functional_equation : ∀ x : ℝ, f (x + 2) = f x + f 2

theorem f_five : f 5 = 5 / 2 :=
by sorry

end f_five_l1852_185226


namespace lena_nicole_candy_difference_l1852_185265

variables (L K N : ℕ)

theorem lena_nicole_candy_difference
  (hL : L = 16)
  (hLK : L + 5 = 3 * K)
  (hKN : K = N - 4) :
  L - N = 5 :=
sorry

end lena_nicole_candy_difference_l1852_185265


namespace lemons_for_lemonade_l1852_185238

theorem lemons_for_lemonade (lemons_gallons_ratio : 30 / 25 = x / 10) : x = 12 :=
by
  sorry

end lemons_for_lemonade_l1852_185238


namespace difference_of_two_distinct_members_sum_of_two_distinct_members_l1852_185228

theorem difference_of_two_distinct_members (S : Set ℕ) (h : S = {n | n ∈ Finset.range 20 ∧ 1 ≤ n ∧ n ≤ 20}) :
  (∃ N, N = 19 ∧ (∀ n, 1 ≤ n ∧ n ≤ N → ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ n = a - b)) :=
by
  sorry

theorem sum_of_two_distinct_members (S : Set ℕ) (h : S = {n | n ∈ Finset.range 20 ∧ 1 ≤ n ∧ n ≤ 20}) :
  (∃ M, M = 37 ∧ (∀ m, 3 ≤ m ∧ m ≤ 39 → ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ m = a + b)) :=
by
  sorry

end difference_of_two_distinct_members_sum_of_two_distinct_members_l1852_185228


namespace paperboy_delivery_sequences_l1852_185235

noncomputable def D : ℕ → ℕ
| 0       => 1  -- D_0 is a dummy value to facilitate indexing
| 1       => 2
| 2       => 4
| 3       => 7
| (n + 4) => D (n + 3) + D (n + 2) + D (n + 1)

theorem paperboy_delivery_sequences : D 11 = 927 := by
  sorry

end paperboy_delivery_sequences_l1852_185235


namespace min_sum_of_dimensions_l1852_185272

theorem min_sum_of_dimensions (a b c : ℕ) (h1 : a * b * c = 1645) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : 
  a + b + c ≥ 129 :=
sorry

end min_sum_of_dimensions_l1852_185272


namespace convert_radian_to_degree_part1_convert_radian_to_degree_part2_convert_radian_to_degree_part3_convert_degree_to_radian_part1_convert_degree_to_radian_part2_l1852_185224

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

end convert_radian_to_degree_part1_convert_radian_to_degree_part2_convert_radian_to_degree_part3_convert_degree_to_radian_part1_convert_degree_to_radian_part2_l1852_185224


namespace hotel_floors_l1852_185284

/-- Given:
  - Each floor has 10 identical rooms.
  - The last floor is unavailable for guests.
  - Hans could be checked into 90 different rooms.
  - There are no other guests.
 - Prove that the total number of floors in the hotel is 10.
--/
theorem hotel_floors :
  (∃ n : ℕ, n ≥ 1 ∧ 10 * (n - 1) = 90) → n = 10 :=
by 
  sorry

end hotel_floors_l1852_185284


namespace arctan_sum_eq_pi_div_two_l1852_185280

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2 :=
by
  sorry

end arctan_sum_eq_pi_div_two_l1852_185280


namespace alpha_beta_square_inequality_l1852_185225

theorem alpha_beta_square_inequality
  (α β : ℝ)
  (h1 : α ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h2 : β ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h3 : α * Real.sin α - β * Real.sin β > 0) :
  α^2 > β^2 :=
by
  sorry

end alpha_beta_square_inequality_l1852_185225


namespace perimeter_of_structure_l1852_185287

noncomputable def structure_area : ℝ := 576
noncomputable def num_squares : ℕ := 9
noncomputable def square_area : ℝ := structure_area / num_squares
noncomputable def side_length : ℝ := Real.sqrt square_area
noncomputable def perimeter (side_length : ℝ) : ℝ := 8 * side_length

theorem perimeter_of_structure : perimeter side_length = 64 := by
  -- proof will follow here
  sorry

end perimeter_of_structure_l1852_185287


namespace emily_cleaning_time_l1852_185269

noncomputable def total_time : ℝ := 8 -- total time in hours
noncomputable def lilly_fiona_time : ℝ := 1/4 * total_time -- Lilly and Fiona's combined time in hours
noncomputable def jack_time : ℝ := 1/3 * total_time -- Jack's time in hours
noncomputable def emily_time : ℝ := total_time - lilly_fiona_time - jack_time -- Emily's time in hours
noncomputable def emily_time_minutes : ℝ := emily_time * 60 -- Emily's time in minutes

theorem emily_cleaning_time :
  emily_time_minutes = 200 := by
  sorry

end emily_cleaning_time_l1852_185269


namespace no_whole_numbers_satisfy_eqn_l1852_185222

theorem no_whole_numbers_satisfy_eqn :
  ¬ ∃ (x y z : ℤ), (x - y) ^ 3 + (y - z) ^ 3 + (z - x) ^ 3 = 2021 :=
by
  sorry

end no_whole_numbers_satisfy_eqn_l1852_185222


namespace negation_proposition_l1852_185247

theorem negation_proposition :
  ¬(∃ x_0 : ℝ, x_0^2 + x_0 - 2 < 0) ↔ ∀ x_0 : ℝ, x_0^2 + x_0 - 2 ≥ 0 :=
by
  sorry

end negation_proposition_l1852_185247


namespace reflection_correct_l1852_185285

/-- Definition of reflection across the line y = -x -/
def reflection_across_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

/-- Given points C and D, and their images C' and D' respectively, under reflection,
    prove the transformation is correct. -/
theorem reflection_correct :
  (reflection_across_y_eq_neg_x (-3, 2) = (3, -2)) ∧ (reflection_across_y_eq_neg_x (-2, 5) = (2, -5)) :=
  by
    sorry

end reflection_correct_l1852_185285


namespace pythagorean_triangle_product_divisible_by_60_l1852_185229

theorem pythagorean_triangle_product_divisible_by_60 : 
  ∀ (a b c : ℕ),
  (∃ m n : ℕ,
  m > n ∧ (m % 2 = 0 ∨ n % 2 = 0) ∧ m.gcd n = 1 ∧
  a = m^2 - n^2 ∧ b = 2 * m * n ∧ c = m^2 + n^2 ∧ a^2 + b^2 = c^2) →
  60 ∣ (a * b * c) :=
sorry

end pythagorean_triangle_product_divisible_by_60_l1852_185229


namespace solve_for_2a_2d_l1852_185268

noncomputable def f (a b c d x : ℝ) : ℝ :=
  (2 * a * x + b) / (c * x + 2 * d)

theorem solve_for_2a_2d (a b c d : ℝ) (habcd_ne_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h : ∀ x, f a b c d (f a b c d x) = x) : 2 * a + 2 * d = 0 :=
sorry

end solve_for_2a_2d_l1852_185268
