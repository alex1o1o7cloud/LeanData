import Mathlib

namespace train_crossing_time_l179_17989

noncomputable def length_of_train : ℝ := 120 -- meters
noncomputable def speed_of_train_kmh : ℝ := 27 -- kilometers per hour
noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmh * (1000 / 3600) -- converted to meters per second
noncomputable def time_to_cross : ℝ := length_of_train / speed_of_train_mps

theorem train_crossing_time :
  time_to_cross = 16 :=
by
  -- proof goes here
  sorry

end train_crossing_time_l179_17989


namespace max_rock_value_l179_17952

/-- Carl discovers a cave with three types of rocks:
    - 6-pound rocks worth $16 each,
    - 3-pound rocks worth $9 each,
    - 2-pound rocks worth $3 each.
    There are at least 15 of each type.
    He can carry a maximum of 20 pounds and no more than 5 rocks in total.
    Prove that the maximum value, in dollars, of the rocks he can carry is $52. -/
theorem max_rock_value :
  ∃ (max_value: ℕ),
  (∀ (c6 c3 c2: ℕ),
    (c6 + c3 + c2 ≤ 5) ∧
    (6 * c6 + 3 * c3 + 2 * c2 ≤ 20) →
    max_value ≥ 16 * c6 + 9 * c3 + 3 * c2) ∧
  max_value = 52 :=
by
  sorry

end max_rock_value_l179_17952


namespace can_construct_segment_l179_17971

noncomputable def constructSegment (Ω₁ Ω₂ : Set (ℝ × ℝ)) (P : ℝ × ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A ∈ Ω₁ ∧ B ∈ Ω₂ ∧ (A + B) / 2 = P

theorem can_construct_segment (Ω₁ Ω₂ : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ Ω₁ ∧ B ∈ Ω₂ ∧ (A + B) / 2 = P) :=
sorry

end can_construct_segment_l179_17971


namespace find_a10_l179_17954

theorem find_a10 (a_n : ℕ → ℤ) (d : ℤ) (h1 : ∀ n, a_n n = a_n 1 + (n - 1) * d)
  (h2 : 5 * a_n 3 = a_n 3 ^ 2)
  (h3 : (a_n 3 + 2 * d) ^ 2 = (a_n 3 - d) * (a_n 3 + 11 * d))
  (h_nonzero : d ≠ 0) :
  a_n 10 = 23 :=
sorry

end find_a10_l179_17954


namespace split_cost_evenly_l179_17957

noncomputable def cupcake_cost : ℝ := 1.50
noncomputable def number_of_cupcakes : ℝ := 12
noncomputable def total_cost : ℝ := number_of_cupcakes * cupcake_cost
noncomputable def total_people : ℝ := 2

theorem split_cost_evenly : (total_cost / total_people) = 9 :=
by
  -- Skipping the proof for now
  sorry

end split_cost_evenly_l179_17957


namespace angle_ABD_30_degrees_l179_17940

theorem angle_ABD_30_degrees (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  (AB BD : ℝ) (angle_DBC : ℝ)
  (h1 : BD = AB * (Real.sqrt 3 / 2))
  (h2 : angle_DBC = 90) : 
  ∃ angle_ABD, angle_ABD = 30 :=
by
  sorry

end angle_ABD_30_degrees_l179_17940


namespace percentage_stock_sold_l179_17990

/-!
# Problem Statement
Given:
1. The cash realized on selling a certain percentage stock is Rs. 109.25.
2. The brokerage is 1/4%.
3. The cash after deducting the brokerage is Rs. 109.

Prove:
The percentage of the stock sold is 100%.
-/

noncomputable def brokerage_fee (S : ℝ) : ℝ :=
  S * 0.0025

noncomputable def selling_price (realized_cash : ℝ) (fee : ℝ) : ℝ :=
  realized_cash + fee

theorem percentage_stock_sold (S : ℝ) (realized_cash : ℝ) (cash_after_brokerage : ℝ)
  (h1 : realized_cash = 109.25)
  (h2 : cash_after_brokerage = 109)
  (h3 : brokerage_fee S = S * 0.0025) :
  S = 109.25 :=
by
  sorry

end percentage_stock_sold_l179_17990


namespace solve_inequality_l179_17920

theorem solve_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) :=
by
  sorry

end solve_inequality_l179_17920


namespace remaining_games_win_percent_l179_17918

variable (totalGames : ℕ) (firstGames : ℕ) (firstWinPercent : ℕ) (seasonWinPercent : ℕ)

-- Given conditions expressed as assumptions:
-- The total number of games played in a season is 40
axiom total_games_condition : totalGames = 40
-- The number of first games played is 30
axiom first_games_condition : firstGames = 30
-- The team won 40% of the first 30 games
axiom first_win_percent_condition : firstWinPercent = 40
-- The team won 50% of all its games in the season
axiom season_win_percent_condition : seasonWinPercent = 50

-- We need to prove that the percentage of the remaining games that the team won is 80%
theorem remaining_games_win_percent {remainingWinPercent : ℕ} :
  totalGames = 40 →
  firstGames = 30 →
  firstWinPercent = 40 →
  seasonWinPercent = 50 →
  remainingWinPercent = 80 :=
by
  intros
  sorry

end remaining_games_win_percent_l179_17918


namespace circle_inscribed_radius_l179_17953

theorem circle_inscribed_radius (R α : ℝ) (hα : α < Real.pi) : 
  ∃ x : ℝ, x = R * (Real.sin (α / 4))^2 :=
sorry

end circle_inscribed_radius_l179_17953


namespace mixed_number_sum_l179_17917

theorem mixed_number_sum : (2 + (1 / 10 : ℝ)) + (3 + (11 / 100 : ℝ)) = 5.21 := by
  sorry

end mixed_number_sum_l179_17917


namespace ending_time_proof_l179_17905

def starting_time_seconds : ℕ := (1 * 3600) + (57 * 60) + 58
def glow_interval : ℕ := 13
def total_glow_count : ℕ := 382
def total_glow_duration : ℕ := total_glow_count * glow_interval
def ending_time_seconds : ℕ := starting_time_seconds + total_glow_duration

theorem ending_time_proof : 
ending_time_seconds = (3 * 3600) + (14 * 60) + 4 := by
  -- Proof starts here
  sorry

end ending_time_proof_l179_17905


namespace smallest_number_increased_by_3_divisible_by_divisors_l179_17962

theorem smallest_number_increased_by_3_divisible_by_divisors
  (n : ℕ)
  (d1 d2 d3 d4 : ℕ)
  (h1 : d1 = 27)
  (h2 : d2 = 35)
  (h3 : d3 = 25)
  (h4 : d4 = 21) :
  (n + 3) % d1 = 0 →
  (n + 3) % d2 = 0 →
  (n + 3) % d3 = 0 →
  (n + 3) % d4 = 0 →
  n = 4722 :=
by
  sorry

end smallest_number_increased_by_3_divisible_by_divisors_l179_17962


namespace moon_speed_kmh_l179_17951

theorem moon_speed_kmh (speed_kms : ℝ) (h : speed_kms = 0.9) : speed_kms * 3600 = 3240 :=
by
  rw [h]
  norm_num

end moon_speed_kmh_l179_17951


namespace tom_age_ratio_l179_17955

theorem tom_age_ratio (T N : ℕ) (h1 : T = 2 * (T / 2)) (h2 : T - N = 3 * ((T / 2) - 3 * N)) : T / N = 16 :=
  sorry

end tom_age_ratio_l179_17955


namespace part1_part2_part3_l179_17909

-- Part 1
theorem part1 :
  ∀ x : ℝ, (4 * x - 3 = 1) → (x = 1) ↔ 
    (¬(x - 3 > 3 * x - 1) ∧ (4 * (x - 1) ≤ 2) ∧ (x + 2 > 0 ∧ 3 * x - 3 ≤ 1)) :=
by sorry

-- Part 2
theorem part2 :
  ∀ (m n q : ℝ), (m + 2 * n = 6) → (2 * m + n = 3 * q) → (m + n > 1) → q > -1 :=
by sorry

-- Part 3
theorem part3 :
  ∀ (k m n : ℝ), (k < 3) → (∃ x : ℝ, (3 * (x - 1) = k) ∧ (4 * x + n < x + 2 * m)) → 
    (m + n ≥ 0) → (∃! n : ℝ, ∀ x : ℝ, (2 ≤ m ∧ m < 5 / 2)) :=
by sorry

end part1_part2_part3_l179_17909


namespace inequality_for_positive_real_l179_17941

theorem inequality_for_positive_real (x : ℝ) (h : 0 < x) : x + 1/x ≥ 2 :=
by
  sorry

end inequality_for_positive_real_l179_17941


namespace min_f_a_eq_1_min_f_a_le_neg1_min_f_neg1_lt_a_lt_0_l179_17921

-- Define the quadratic function
def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 5

-- Prove the minimum value for a = 1 and x in [-1, 0]
theorem min_f_a_eq_1 : ∀ x : ℝ, x ∈ Set.Icc (-1) 0 → f 1 x ≥ 5 :=
by
  sorry

-- Prove the minimum value for a < 0 and x in [-1, 0], when a ≤ -1
theorem min_f_a_le_neg1 (h : ∀ a : ℝ, a ≤ -1) : ∀ x : ℝ, x ∈ Set.Icc (-1) 0 → f a (-1) ≤ f a x :=
by
  sorry

-- Prove the minimum value for a < 0 and x in [-1, 0], when -1 < a < 0
theorem min_f_neg1_lt_a_lt_0 (h : ∀ a : ℝ, -1 < a ∧ a < 0) : ∀ x : ℝ, x ∈ Set.Icc (-1) 0 → f a a ≤ f a x :=
by
  sorry

end min_f_a_eq_1_min_f_a_le_neg1_min_f_neg1_lt_a_lt_0_l179_17921


namespace total_sheep_l179_17902

-- Define the conditions as hypotheses
variables (Aaron_sheep Beth_sheep : ℕ)
def condition1 := Aaron_sheep = 7 * Beth_sheep
def condition2 := Aaron_sheep = 532
def condition3 := Beth_sheep = 76

-- Assert that under these conditions, the total number of sheep is 608.
theorem total_sheep
  (h1 : condition1 Aaron_sheep Beth_sheep)
  (h2 : condition2 Aaron_sheep)
  (h3 : condition3 Beth_sheep) :
  Aaron_sheep + Beth_sheep = 608 :=
by sorry

end total_sheep_l179_17902


namespace opposite_of_neg_2023_l179_17965

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l179_17965


namespace first_percentage_increase_l179_17986

theorem first_percentage_increase (x : ℝ) :
  (1 + x / 100) * 1.4 = 1.82 → x = 30 := 
by 
  intro h
  -- start your proof here
  sorry

end first_percentage_increase_l179_17986


namespace train_speed_l179_17988

def distance : ℕ := 500
def time : ℕ := 10
def conversion_factor : ℝ := 3.6

theorem train_speed :
  (distance / time : ℝ) * conversion_factor = 180 :=
by
  sorry

end train_speed_l179_17988


namespace solve_system_l179_17925

theorem solve_system (x y z u : ℝ) :
  x^3 * y^2 * z = 2 ∧
  z^3 * u^2 * x = 32 ∧
  y^3 * z^2 * u = 8 ∧
  u^3 * x^2 * y = 8 →
  (x = 1 ∧ y = 1 ∧ z = 2 ∧ u = 2) ∨
  (x = 1 ∧ y = -1 ∧ z = 2 ∧ u = -2) ∨
  (x = -1 ∧ y = 1 ∧ z = -2 ∧ u = 2) ∨
  (x = -1 ∧ y = -1 ∧ z = -2 ∧ u = -2) :=
sorry

end solve_system_l179_17925


namespace apples_prepared_l179_17958

variables (n_x n_l : ℕ)

theorem apples_prepared (hx : 3 * n_x = 5 * n_l - 12) (hs : 6 * n_l = 72) : n_x = 12 := 
by sorry

end apples_prepared_l179_17958


namespace sum_reciprocals_l179_17970

theorem sum_reciprocals (a b α β : ℝ) (h1: 7 * a^2 + 2 * a + 6 = 0) (h2: 7 * b^2 + 2 * b + 6 = 0) 
  (h3: α = 1 / a) (h4: β = 1 / b) (h5: a + b = -2/7) (h6: a * b = 6/7) : 
  α + β = -1/3 :=
by
  sorry

end sum_reciprocals_l179_17970


namespace integral_eval_l179_17984

theorem integral_eval : ∫ x in (1:ℝ)..(2:ℝ), (2*x + 1/x) = 3 + Real.log 2 := by
  sorry

end integral_eval_l179_17984


namespace melted_mixture_weight_l179_17916

theorem melted_mixture_weight (Z C : ℝ) (ratio : 9 / 11 = Z / C) (zinc_weight : Z = 28.8) : Z + C = 64 :=
by
  sorry

end melted_mixture_weight_l179_17916


namespace closest_to_9_l179_17966

noncomputable def optionA : ℝ := 10.01
noncomputable def optionB : ℝ := 9.998
noncomputable def optionC : ℝ := 9.9
noncomputable def optionD : ℝ := 9.01
noncomputable def target : ℝ := 9

theorem closest_to_9 : 
  abs (optionD - target) < abs (optionA - target) ∧ 
  abs (optionD - target) < abs (optionB - target) ∧ 
  abs (optionD - target) < abs (optionC - target) := 
by
  sorry

end closest_to_9_l179_17966


namespace solution_set_l179_17924

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)
axiom monotone_decreasing_f : ∀ {a b : ℝ}, 0 ≤ a → a ≤ b → f b ≤ f a
axiom f_half_eq_zero : f (1 / 2) = 0

theorem solution_set :
  { x : ℝ | f (Real.log x / Real.log (1 / 4)) < 0 } = 
  { x : ℝ | 0 < x ∧ x < 1 / 2 } ∪ { x : ℝ | 2 < x } :=
by
  sorry

end solution_set_l179_17924


namespace enclosed_polygons_l179_17937

theorem enclosed_polygons (n : ℕ) :
  (∃ α β : ℝ, (15 * β) = 360 ∧ β = 180 - α ∧ (15 * α) = 180 * (n - 2) / n) ↔ n = 15 :=
by sorry

end enclosed_polygons_l179_17937


namespace school_total_payment_l179_17996

theorem school_total_payment
  (price : ℕ)
  (kindergarten_models : ℕ)
  (elementary_library_multiplier : ℕ)
  (model_reduction_percentage : ℚ)
  (total_models : ℕ)
  (reduced_price : ℚ)
  (total_payment : ℚ)
  (h1 : price = 100)
  (h2 : kindergarten_models = 2)
  (h3 : elementary_library_multiplier = 2)
  (h4 : model_reduction_percentage = 0.05)
  (h5 : total_models = kindergarten_models + (kindergarten_models * elementary_library_multiplier))
  (h6 : total_models > 5)
  (h7 : reduced_price = price - (price * model_reduction_percentage))
  (h8 : total_payment = total_models * reduced_price) :
  total_payment = 570 := 
by
  sorry

end school_total_payment_l179_17996


namespace ratio_y_to_x_l179_17936

theorem ratio_y_to_x (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : y / x = 13 / 2 :=
by
  sorry

end ratio_y_to_x_l179_17936


namespace y_less_than_z_by_40_percent_l179_17930

variable {x y z : ℝ}

theorem y_less_than_z_by_40_percent (h1 : x = 1.3 * y) (h2 : x = 0.78 * z) : y = 0.6 * z :=
by
  -- The proof will be provided here
  -- We are demonstrating that y = 0.6 * z is a consequence of h1 and h2
  sorry

end y_less_than_z_by_40_percent_l179_17930


namespace cars_meet_and_crush_fly_l179_17960

noncomputable def time_to_meet (L v_A v_B : ℝ) : ℝ := L / (v_A + v_B)

theorem cars_meet_and_crush_fly :
  ∀ (L v_A v_B v_fly : ℝ), L = 300 → v_A = 50 → v_B = 100 → v_fly = 150 → time_to_meet L v_A v_B = 2 :=
by
  intros L v_A v_B v_fly L_eq v_A_eq v_B_eq v_fly_eq
  rw [L_eq, v_A_eq, v_B_eq]
  simp [time_to_meet]
  norm_num

end cars_meet_and_crush_fly_l179_17960


namespace sum_of_squares_expressible_l179_17976

theorem sum_of_squares_expressible (a b c : ℕ) (h1 : c^2 = a^2 + b^2) : 
  ∃ x y : ℕ, x^2 + y^2 = c^2 + a*b ∧ ∃ u v : ℕ, u^2 + v^2 = c^2 - a*b :=
by
  sorry

end sum_of_squares_expressible_l179_17976


namespace cube_volume_given_surface_area_l179_17923

theorem cube_volume_given_surface_area (A : ℝ) (V : ℝ) :
  A = 96 → V = 64 :=
by
  sorry

end cube_volume_given_surface_area_l179_17923


namespace unique_cd_exists_l179_17908

open Real

theorem unique_cd_exists (h₀ : 0 < π / 2):
  ∃! (c d : ℝ), (0 < c) ∧ (c < π / 2) ∧ (0 < d) ∧ (d < π / 2) ∧ (c < d) ∧ 
  (sin (cos c) = c) ∧ (cos (sin d) = d) := sorry

end unique_cd_exists_l179_17908


namespace general_formula_for_sequence_l179_17904

noncomputable def S := ℕ → ℚ
noncomputable def a := ℕ → ℚ

theorem general_formula_for_sequence (a : a) (S : S) (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, S (n + 1) = (2 / 3) * a (n + 1) + 1 / 3) :
  ∀ n : ℕ, a n = 
  if n = 1 then 2 
  else -5 * (-2)^(n-2) := 
by 
  sorry

end general_formula_for_sequence_l179_17904


namespace total_distance_driven_l179_17906

def renaldo_distance : ℕ := 15
def ernesto_distance : ℕ := 7 + (renaldo_distance / 3)

theorem total_distance_driven :
  renaldo_distance + ernesto_distance = 27 :=
sorry

end total_distance_driven_l179_17906


namespace plane_speed_in_still_air_l179_17900

theorem plane_speed_in_still_air (P W : ℝ) 
  (h1 : (P + W) * 3 = 900) 
  (h2 : (P - W) * 4 = 900) 
  : P = 262.5 :=
by
  sorry

end plane_speed_in_still_air_l179_17900


namespace boys_variance_greater_than_girls_l179_17969

def boys_scores : List ℝ := [86, 94, 88, 92, 90]
def girls_scores : List ℝ := [88, 93, 93, 88, 93]

noncomputable def variance (scores : List ℝ) : ℝ :=
  let n := scores.length
  let mean := (scores.sum / n)
  let squared_diff := scores.map (λ x => (x - mean) ^ 2)
  (squared_diff.sum) / n

theorem boys_variance_greater_than_girls :
  variance boys_scores > variance girls_scores :=
by
  sorry

end boys_variance_greater_than_girls_l179_17969


namespace correct_quadratic_equation_l179_17927

-- Definitions based on conditions
def root_sum (α β : ℝ) := α + β = 8
def root_product (α β : ℝ) := α * β = 24

-- Main statement to be proven
theorem correct_quadratic_equation (α β : ℝ) (h1 : root_sum 5 3) (h2 : root_product (-6) (-4)) :
    (α - 5) * (α - 3) = 0 ∧ (α + 6) * (α + 4) = 0 → α * α - 8 * α + 24 = 0 :=
sorry

end correct_quadratic_equation_l179_17927


namespace slower_train_speed_l179_17956

noncomputable def speed_of_slower_train (v_f : ℕ) (l1 l2 : ℚ) (t : ℚ) : ℚ :=
  let total_distance := l1 + l2
  let time_in_hours := t / 3600
  let relative_speed := total_distance / time_in_hours
  relative_speed - v_f

theorem slower_train_speed :
  speed_of_slower_train 210 (11 / 10) (9 / 10) 24 = 90 := by
  sorry

end slower_train_speed_l179_17956


namespace complementary_angles_positive_difference_l179_17932

theorem complementary_angles_positive_difference
  (x : ℝ)
  (h1 : 3 * x + x = 90): 
  |(3 * x) - x| = 45 := 
by
  -- Proof would go here (details skipped)
  sorry

end complementary_angles_positive_difference_l179_17932


namespace car_speed_l179_17992

-- Definitions from conditions
def distance : ℝ := 360
def time : ℝ := 4.5

-- Statement to prove
theorem car_speed : (distance / time) = 80 := by
  sorry

end car_speed_l179_17992


namespace perimeter_of_square_l179_17931

theorem perimeter_of_square (area : ℝ) (h : area = 392) : 
  ∃ (s : ℝ), 4 * s = 56 * Real.sqrt 2 :=
by 
  use (Real.sqrt 392)
  sorry

end perimeter_of_square_l179_17931


namespace isosceles_triangle_third_vertex_y_coord_l179_17974

theorem isosceles_triangle_third_vertex_y_coord :
  ∀ (A B : ℝ × ℝ) (θ : ℝ), 
  A = (0, 5) → B = (8, 5) → θ = 60 → 
  ∃ (C : ℝ × ℝ), C.fst > 0 ∧ C.snd > 5 ∧ C.snd = 5 + 4 * Real.sqrt 3 :=
by
  intros A B θ hA hB hθ
  use (4, 5 + 4 * Real.sqrt 3)
  sorry

end isosceles_triangle_third_vertex_y_coord_l179_17974


namespace numbers_difference_l179_17981

theorem numbers_difference (A B C : ℝ) (h1 : B = 10) (h2 : B - A = C - B) (h3 : A * B = 85) (h4 : B * C = 115) : 
  B - A = 1.5 ∧ C - B = 1.5 :=
by
  sorry

end numbers_difference_l179_17981


namespace range_of_omega_l179_17922

theorem range_of_omega (ω : ℝ) (hω : ω > 2/3) :
  (∀ x : ℝ, x = (k : ℤ) * π / ω + 3 * π / (4 * ω) → (x ≤ π ∨ x ≥ 2 * π) ) →
  ω ∈ Set.Icc (3/4 : ℝ) (7/8 : ℝ) :=
by
  sorry

end range_of_omega_l179_17922


namespace largest_multiple_of_9_less_than_100_l179_17939

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n * 9 < 100 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ n * 9 :=
by
  sorry

end largest_multiple_of_9_less_than_100_l179_17939


namespace math_vs_english_time_difference_l179_17946

-- Definitions based on the conditions
def english_total_questions : ℕ := 30
def math_total_questions : ℕ := 15
def english_total_time_minutes : ℕ := 60 -- 1 hour = 60 minutes
def math_total_time_minutes : ℕ := 90 -- 1.5 hours = 90 minutes

noncomputable def time_per_english_question : ℕ :=
  english_total_time_minutes / english_total_questions

noncomputable def time_per_math_question : ℕ :=
  math_total_time_minutes / math_total_questions

-- Theorem based on the question and correct answer
theorem math_vs_english_time_difference :
  (time_per_math_question - time_per_english_question) = 4 :=
by
  -- Proof here
  sorry

end math_vs_english_time_difference_l179_17946


namespace ordering_9_8_4_12_3_16_l179_17929

theorem ordering_9_8_4_12_3_16 : (4 ^ 12 < 9 ^ 8) ∧ (9 ^ 8 = 3 ^ 16) :=
by {
  sorry
}

end ordering_9_8_4_12_3_16_l179_17929


namespace sin_double_angle_l179_17998

theorem sin_double_angle (α : ℝ) (h : Real.cos (Real.pi / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 :=
  sorry

end sin_double_angle_l179_17998


namespace ratio_p_q_is_minus_one_l179_17926

theorem ratio_p_q_is_minus_one (p q : ℤ) (h : (25 / 7 : ℝ) + ((2 * q - p) / (2 * q + p) : ℝ) = 4) : (p / q : ℝ) = -1 := 
sorry

end ratio_p_q_is_minus_one_l179_17926


namespace problem_solution_l179_17980

theorem problem_solution (x y : ℝ) (h1 : x * y = 6) (h2 : x^2 * y + x * y^2 + x + y = 63) : x^2 + y^2 = 69 :=
by
  sorry

end problem_solution_l179_17980


namespace length_of_platform_l179_17945

-- Given conditions
def train_length : ℝ := 100
def time_pole : ℝ := 15
def time_platform : ℝ := 40

-- Theorem to prove the length of the platform
theorem length_of_platform (L : ℝ) 
    (h_train_length : train_length = 100)
    (h_time_pole : time_pole = 15)
    (h_time_platform : time_platform = 40)
    (h_speed : (train_length / time_pole) = (100 + L) / time_platform) : 
    L = 500 / 3 :=
by
  sorry

end length_of_platform_l179_17945


namespace tournament_committees_count_l179_17950

-- Definitions corresponding to the conditions
def num_teams : ℕ := 4
def team_size : ℕ := 8
def members_selected_by_winning_team : ℕ := 3
def members_selected_by_other_teams : ℕ := 2

-- Binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Counting the number of possible committees
def total_committees : ℕ :=
  let num_ways_winning_team := binom team_size members_selected_by_winning_team
  let num_ways_other_teams := binom team_size members_selected_by_other_teams
  num_teams * num_ways_winning_team * (num_ways_other_teams ^ (num_teams - 1))

-- The statement to be proved
theorem tournament_committees_count : total_committees = 4917248 := by
  sorry

end tournament_committees_count_l179_17950


namespace probability_composite_is_correct_l179_17935

noncomputable def probability_composite : ℚ :=
  1 - (25 / (8^6))

theorem probability_composite_is_correct :
  probability_composite = 262119 / 262144 :=
by
  sorry

end probability_composite_is_correct_l179_17935


namespace number_of_tangents_l179_17907

-- Define the points and conditions
variable (A B : ℝ × ℝ)
variable (dist_AB : dist A B = 8)
variable (radius_A : ℝ := 3)
variable (radius_B : ℝ := 2)

-- The goal
theorem number_of_tangents (dist_condition : dist A B = 8) : 
  ∃ n, n = 2 :=
by
  -- skipping the proof
  sorry

end number_of_tangents_l179_17907


namespace complement_U_B_eq_D_l179_17993

def B (x : ℝ) : Prop := x^2 - 3 * x + 2 < 0
def U : Set ℝ := Set.univ
def complement_U_B : Set ℝ := U \ {x | B x}

theorem complement_U_B_eq_D : complement_U_B = {x | x ≤ 1 ∨ x ≥ 2} := by
  sorry

end complement_U_B_eq_D_l179_17993


namespace largest_x_63_over_8_l179_17994

theorem largest_x_63_over_8 (x : ℝ) (h1 : ⌊x⌋ / x = 8 / 9) : x = 63 / 8 :=
by
  sorry

end largest_x_63_over_8_l179_17994


namespace evaluate_expression_l179_17913

theorem evaluate_expression (x z : ℤ) (h1 : x = 2) (h2 : z = 1) : z * (z - 4 * x) = -7 :=
by
  rw [h1, h2]
  sorry

end evaluate_expression_l179_17913


namespace spherical_to_rectangular_coords_l179_17977

noncomputable def sphericalToRectangular (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * (Real.sin phi) * (Real.cos theta), 
   rho * (Real.sin phi) * (Real.sin theta), 
   rho * (Real.cos phi))

theorem spherical_to_rectangular_coords :
  sphericalToRectangular 3 (3 * Real.pi / 2) (Real.pi / 3) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  sorry

end spherical_to_rectangular_coords_l179_17977


namespace cos_C_sin_B_area_l179_17982

noncomputable def triangle_conditions (A B C a b c : ℝ) : Prop :=
  (A + B + C = Real.pi) ∧
  (b / c = 2 * Real.sqrt 3 / 3) ∧
  (A + 3 * C = Real.pi)

theorem cos_C (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) :
  Real.cos C = Real.sqrt 3 / 3 :=
sorry

theorem sin_B (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) :
  Real.sin B = 2 * Real.sqrt 2 / 3 :=
sorry

theorem area (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) (hb : b = 3 * Real.sqrt 3) :
  (1 / 2) * b * c * Real.sin A = 9 * Real.sqrt 2 / 4 :=
sorry

end cos_C_sin_B_area_l179_17982


namespace no_15_students_with_unique_colors_l179_17942

-- Conditions as definitions
def num_students : Nat := 30
def num_colors : Nat := 15

-- The main statement
theorem no_15_students_with_unique_colors
  (students : Fin num_students → (Fin num_colors × Fin num_colors)) :
  ¬ ∃ (subset : Fin 15 → Fin num_students),
    ∀ i j (hi : i ≠ j), (students (subset i)).1 ≠ (students (subset j)).1 ∧
                         (students (subset i)).2 ≠ (students (subset j)).2 :=
by sorry

end no_15_students_with_unique_colors_l179_17942


namespace circle_a_lt_8_tangent_lines_perpendicular_circle_intersection_l179_17979

-- Problem (1)
theorem circle_a_lt_8 (x y a : ℝ) (h : x^2 + y^2 - 4*x - 4*y + a = 0) : 
  a < 8 :=
by
  sorry

-- Problem (2)
theorem tangent_lines (a : ℝ) (h : a = -17) : 
  ∃ (k : ℝ), k * 7 - 6 - 7 * k = 0 ∧
  ((39 * k + 80 * (-7) - 207 = 0) ∨ (k = 7)) :=
by
  sorry

-- Problem (3)
theorem perpendicular_circle_intersection (x1 x2 y1 y2 a : ℝ) 
  (h1: 2 * x1 - y1 - 3 = 0) 
  (h2: 2 * x2 - y2 - 3 = 0) 
  (h3: x1 * x2 + y1 * y2 = 0) 
  (hpoly : 5 * x1 * x2 - 6 * (x1 + x2) + 9 = 0): 
  a = -6 / 5 :=
by
  sorry

end circle_a_lt_8_tangent_lines_perpendicular_circle_intersection_l179_17979


namespace polynomial_prime_is_11_l179_17914

def P (a : ℕ) : ℕ := a^4 - 4 * a^3 + 15 * a^2 - 30 * a + 27

theorem polynomial_prime_is_11 (a : ℕ) (hp : Nat.Prime (P a)) : P a = 11 := 
by {
  sorry
}

end polynomial_prime_is_11_l179_17914


namespace quadratic_reciprocal_squares_l179_17910

theorem quadratic_reciprocal_squares :
  (∃ p q : ℝ, (∀ x : ℝ, 3*x^2 - 5*x + 2 = 0 → (x = p ∨ x = q)) ∧ (1 / p^2 + 1 / q^2 = 13 / 4)) :=
by
  have quadratic_eq : (∀ x : ℝ, 3*x^2 - 5*x + 2 = 0 → (x = 1 ∨ x = 2 / 3)) := sorry
  have identity_eq : 1 / (1:ℝ)^2 + 1 / (2 / 3)^2 = 13 / 4 := sorry
  exact ⟨1, 2 / 3, quadratic_eq, identity_eq⟩

end quadratic_reciprocal_squares_l179_17910


namespace amusement_park_people_l179_17912

theorem amusement_park_people (students adults free : ℕ) (total_people paid : ℕ) :
  students = 194 →
  adults = 235 →
  free = 68 →
  total_people = students + adults →
  paid = total_people - free →
  paid - free = 293 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end amusement_park_people_l179_17912


namespace solution_set_of_inequality_l179_17997

theorem solution_set_of_inequality:
  {x : ℝ | x^2 > x} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1} :=
by
  sorry

end solution_set_of_inequality_l179_17997


namespace find_original_numbers_l179_17919

theorem find_original_numbers (x y : ℕ) (hx : x + y = 2022) 
  (hy : (x - 5) / 10 + 10 * y + 1 = 2252) : x = 1815 ∧ y = 207 :=
by sorry

end find_original_numbers_l179_17919


namespace total_cars_l179_17944

theorem total_cars (Tommy_cars Jessie_cars : ℕ) (older_brother_cars : ℕ) 
  (h1 : Tommy_cars = 3) 
  (h2 : Jessie_cars = 3)
  (h3 : older_brother_cars = Tommy_cars + Jessie_cars + 5) : 
  Tommy_cars + Jessie_cars + older_brother_cars = 17 := by
  sorry

end total_cars_l179_17944


namespace garden_strawberry_area_l179_17963

variable (total_garden_area : Real) (fruit_fraction : Real) (strawberry_fraction : Real)
variable (h1 : total_garden_area = 64)
variable (h2 : fruit_fraction = 1 / 2)
variable (h3 : strawberry_fraction = 1 / 4)

theorem garden_strawberry_area : 
  let fruit_area := total_garden_area * fruit_fraction
  let strawberry_area := fruit_area * strawberry_fraction
  strawberry_area = 8 :=
by
  sorry

end garden_strawberry_area_l179_17963


namespace sqrt_expression_equal_cos_half_theta_l179_17967

noncomputable def sqrt_half_plus_sqrt_half_cos2theta_minus_sqrt_one_minus_sintheta (θ : Real) : Real :=
  Real.sqrt (1 / 2 + 1 / 2 * Real.sqrt (1 / 2 + 1 / 2 * Real.cos (2 * θ))) - Real.sqrt (1 - Real.sin θ)

theorem sqrt_expression_equal_cos_half_theta (θ : Real) (h : π < θ) (h2 : θ < 3 * π / 2)
  (h3 : Real.cos θ < 0) (h4 : 0 < Real.sin (θ / 2)) (h5 : Real.cos (θ / 2) < 0) :
  sqrt_half_plus_sqrt_half_cos2theta_minus_sqrt_one_minus_sintheta θ = Real.cos (θ / 2) :=
by
  sorry

end sqrt_expression_equal_cos_half_theta_l179_17967


namespace irrational_b_eq_neg_one_l179_17947

theorem irrational_b_eq_neg_one
  (a : ℝ) (b : ℝ)
  (h_irrational : ¬ ∃ q : ℚ, a = (q : ℝ))
  (h_eq : ab + a - b = 1) :
  b = -1 :=
sorry

end irrational_b_eq_neg_one_l179_17947


namespace man_l179_17964

noncomputable def man_saves (S : ℝ) : ℝ :=
0.20 * S

noncomputable def initial_expenses (S : ℝ) : ℝ :=
0.80 * S

noncomputable def new_expenses (S : ℝ) : ℝ :=
1.10 * (0.80 * S)

noncomputable def said_savings (S : ℝ) : ℝ :=
S - new_expenses S

theorem man's_monthly_salary (S : ℝ) (h : said_savings S = 500) : S = 4166.67 :=
by
  sorry

end man_l179_17964


namespace definite_integral_solution_l179_17933

noncomputable def integral_problem : ℝ := 
  by 
    sorry

theorem definite_integral_solution :
  integral_problem = (1/6 : ℝ) + Real.log 2 - Real.log 3 := 
by
  sorry

end definite_integral_solution_l179_17933


namespace extreme_point_f_l179_17961

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x - 1)

theorem extreme_point_f :
  ∃ x : ℝ, (∀ y : ℝ, y ≠ 0 → (Real.exp y * y < 0 ↔ y < x)) ∧ x = 0 :=
by
  sorry

end extreme_point_f_l179_17961


namespace total_legs_arms_proof_l179_17978

/-
There are 4 birds, each with 2 legs.
There are 6 dogs, each with 4 legs.
There are 5 snakes, each with no legs.
There are 2 spiders, each with 8 legs.
There are 3 horses, each with 4 legs.
There are 7 rabbits, each with 4 legs.
There are 2 octopuses, each with 8 arms.
There are 8 ants, each with 6 legs.
There is 1 unique creature with 12 legs.
We need to prove that the total number of legs and arms is 164.
-/

def total_legs_arms : Nat := 
  (4 * 2) + (6 * 4) + (5 * 0) + (2 * 8) + (3 * 4) + (7 * 4) + (2 * 8) + (8 * 6) + (1 * 12)

theorem total_legs_arms_proof : total_legs_arms = 164 := by
  sorry

end total_legs_arms_proof_l179_17978


namespace num_subsets_of_abc_eq_eight_l179_17983

theorem num_subsets_of_abc_eq_eight : 
  (∃ (s : Finset ℕ), s = {1, 2, 3} ∧ s.powerset.card = 8) :=
sorry

end num_subsets_of_abc_eq_eight_l179_17983


namespace Nancy_hourly_wage_l179_17975

def tuition_cost := 22000
def parents_coverage := tuition_cost / 2
def scholarship := 3000
def loan := 2 * scholarship
def working_hours := 200
def remaining_tuition := tuition_cost - parents_coverage - scholarship - loan
def hourly_wage_required := remaining_tuition / working_hours

theorem Nancy_hourly_wage : hourly_wage_required = 10 := by
  sorry

end Nancy_hourly_wage_l179_17975


namespace find_a_l179_17949

noncomputable def f (a x : ℝ) := 3*x^3 - 9*x + a
noncomputable def f' (x : ℝ) : ℝ := 9*x^2 - 9

theorem find_a (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) :
  a = 6 ∨ a = -6 :=
by sorry

end find_a_l179_17949


namespace find_m_l179_17903

theorem find_m (A B : Set ℝ) (m : ℝ) (hA: A = {2, m}) (hB: B = {1, m^2}) (hU: A ∪ B = {1, 2, 3, 9}) : m = 3 :=
by 
  sorry

end find_m_l179_17903


namespace seashells_given_to_brothers_l179_17995

theorem seashells_given_to_brothers :
  ∃ B : ℕ, 180 - 40 - B = 2 * 55 ∧ B = 30 := by
  sorry

end seashells_given_to_brothers_l179_17995


namespace evelyn_found_caps_l179_17928

theorem evelyn_found_caps (start_caps end_caps found_caps : ℕ) 
    (h1 : start_caps = 18) 
    (h2 : end_caps = 81) 
    (h3 : found_caps = end_caps - start_caps) :
  found_caps = 63 := by
  sorry

end evelyn_found_caps_l179_17928


namespace first_day_reduction_percentage_l179_17968

variables (P x : ℝ)

theorem first_day_reduction_percentage (h : P * (1 - x / 100) * 0.90 = 0.81 * P) : x = 10 :=
sorry

end first_day_reduction_percentage_l179_17968


namespace x_positive_implies_abs_positive_abs_positive_not_necessiarily_x_positive_x_positive_is_sufficient_but_not_necessary_l179_17973

variable (x : ℝ)

theorem x_positive_implies_abs_positive (hx : x > 0) : |x| > 0 := sorry

theorem abs_positive_not_necessiarily_x_positive : (∃ x : ℝ, |x| > 0 ∧ ¬(x > 0)) := sorry

theorem x_positive_is_sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 0 → |x| > 0) ∧ 
  (∃ x : ℝ, |x| > 0 ∧ ¬(x > 0)) := 
  ⟨x_positive_implies_abs_positive, abs_positive_not_necessiarily_x_positive⟩

end x_positive_implies_abs_positive_abs_positive_not_necessiarily_x_positive_x_positive_is_sufficient_but_not_necessary_l179_17973


namespace four_digit_numbers_count_eq_l179_17999

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l179_17999


namespace range_of_m_l179_17948

theorem range_of_m (m : ℝ) (x : ℝ) :
  (¬ (|1 - (x - 1) / 3| ≤ 2) → ¬ (x^2 - 2 * x + (1 - m^2) ≤ 0)) → 
  (|m| ≥ 9) :=
by
  sorry

end range_of_m_l179_17948


namespace model2_best_fit_l179_17934
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

end model2_best_fit_l179_17934


namespace average_score_l179_17959

theorem average_score (avg1 avg2 : ℕ) (n1 n2 total_matches : ℕ) (total_avg : ℕ) 
  (h1 : avg1 = 60) 
  (h2 : avg2 = 70) 
  (h3 : n1 = 10) 
  (h4 : n2 = 15) 
  (h5 : total_matches = 25) 
  (h6 : total_avg = 66) :
  (( (avg1 * n1) + (avg2 * n2) ) / total_matches = total_avg) :=
by
  sorry

end average_score_l179_17959


namespace trackball_mice_count_l179_17985

theorem trackball_mice_count 
  (total_mice wireless_mice optical_mice trackball_mice : ℕ)
  (h1 : total_mice = 80)
  (h2 : wireless_mice = total_mice / 2)
  (h3 : optical_mice = total_mice / 4)
  (h4 : trackball_mice = total_mice - (wireless_mice + optical_mice)) :
  trackball_mice = 20 := by 
  sorry

end trackball_mice_count_l179_17985


namespace directly_above_156_is_133_l179_17972

def row_numbers (k : ℕ) : ℕ := 2 * k - 1

def total_numbers_up_to_row (k : ℕ) : ℕ := k * k

def find_row (n : ℕ) : ℕ :=
  Nat.sqrt (n + 1)

def position_in_row (n k : ℕ) : ℕ :=
  n - (total_numbers_up_to_row (k - 1)) + 1

def number_directly_above (n : ℕ) : ℕ :=
  let k := find_row n
  let pos := position_in_row n k
  (total_numbers_up_to_row (k - 1) - row_numbers (k - 1)) + pos + 1

theorem directly_above_156_is_133 : number_directly_above 156 = 133 := 
  by
  sorry

end directly_above_156_is_133_l179_17972


namespace bunches_with_new_distribution_l179_17943

-- Given conditions
def bunches_initial := 8
def flowers_per_bunch_initial := 9
def total_flowers := bunches_initial * flowers_per_bunch_initial

-- New condition and proof requirement
def flowers_per_bunch_new := 12
def bunches_new := total_flowers / flowers_per_bunch_new

theorem bunches_with_new_distribution : bunches_new = 6 := by
  sorry

end bunches_with_new_distribution_l179_17943


namespace find_a_n_plus_b_n_l179_17938

noncomputable def a (n : ℕ) : ℕ := 
  if n = 1 then 1 
  else if n = 2 then 3 
  else sorry -- Placeholder for proper recursive implementation

noncomputable def b (n : ℕ) : ℕ := 
  if n = 1 then 5
  else sorry -- Placeholder for proper recursive implementation

theorem find_a_n_plus_b_n (n : ℕ) (i j k l : ℕ) (h1 : a 1 = 1) (h2 : a 2 = 3) (h3 : b 1 = 5) 
  (h4 : i + j = k + l) (h5 : a i + b j = a k + b l) : a n + b n = 4 * n + 2 := 
by
  sorry

end find_a_n_plus_b_n_l179_17938


namespace harmonic_arithmetic_sequence_common_difference_l179_17915

theorem harmonic_arithmetic_sequence_common_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) : 
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * d)) →
  (∀ n, a n = a 1 + (n - 1) * d) →
  (a 1 = 1) →
  (d ≠ 0) →
  (∃ k, ∀ n, S n / S (2 * n) = k) →
  d = 2 :=
by
  sorry

end harmonic_arithmetic_sequence_common_difference_l179_17915


namespace exists_x0_l179_17901

noncomputable def f (x a : ℝ) : ℝ := x^2 + (Real.log (3 * x))^2 - 2 * a * (x + 3 * Real.log (3 * x)) + 10 * a^2

theorem exists_x0 (a : ℝ) (h : a = 1 / 30) : ∃ x0 : ℝ, f x0 a ≤ 1 / 10 := 
by
  sorry

end exists_x0_l179_17901


namespace isabella_hair_length_l179_17991

theorem isabella_hair_length (h : ℕ) (g : h + 4 = 22) : h = 18 := by
  sorry

end isabella_hair_length_l179_17991


namespace abc_zero_l179_17911

theorem abc_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = (a * b * c)^3) 
  : a * b * c = 0 := by
  sorry

end abc_zero_l179_17911


namespace sum_of_roots_l179_17987

theorem sum_of_roots (a b c : ℚ) (h_eq : 6 * a^3 + 7 * a^2 - 12 * a = 0) (h_eq_b : 6 * b^3 + 7 * b^2 - 12 * b = 0) (h_eq_c : 6 * c^3 + 7 * c^2 - 12 * c = 0) : 
  a + b + c = -7/6 := 
by
  -- Insert proof steps here
  sorry

end sum_of_roots_l179_17987
