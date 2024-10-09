import Mathlib

namespace integral_rational_term_expansion_l325_32507

theorem integral_rational_term_expansion :
  ∫ x in 0.0..1.0, x ^ (1/6 : ℝ) = 6/7 := by
  sorry

end integral_rational_term_expansion_l325_32507


namespace functional_equation_solution_l325_32597

theorem functional_equation_solution (f : ℚ → ℕ) :
  (∀ (x y : ℚ) (hx : 0 < x) (hy : 0 < y),
    f (x * y) * Nat.gcd (f x * f y) (f (x⁻¹) * f (y⁻¹)) = (x * y) * f (x⁻¹) * f (y⁻¹))
  → (∀ (x : ℚ) (hx : 0 < x), f x = x.num) :=
sorry

end functional_equation_solution_l325_32597


namespace average_one_half_one_fourth_one_eighth_l325_32577

theorem average_one_half_one_fourth_one_eighth : 
  ((1 / 2.0 + 1 / 4.0 + 1 / 8.0) / 3.0) = 7 / 24 := 
by sorry

end average_one_half_one_fourth_one_eighth_l325_32577


namespace greatest_integer_less_than_neg_eight_over_three_l325_32521

theorem greatest_integer_less_than_neg_eight_over_three :
  ∃ (z : ℤ), (z < -8 / 3) ∧ ∀ w : ℤ, (w < -8 / 3) → w ≤ z := by
  sorry

end greatest_integer_less_than_neg_eight_over_three_l325_32521


namespace positive_number_eq_576_l325_32533

theorem positive_number_eq_576 (x : ℝ) (h : 0 < x) (h_eq : (2 / 3) * x = (25 / 216) * (1 / x)) : x = 5.76 := 
by 
  sorry

end positive_number_eq_576_l325_32533


namespace find_varphi_l325_32576

theorem find_varphi
  (ϕ : ℝ)
  (h : ∃ k : ℤ, ϕ = (π / 8) + (k * π / 2)) :
  ϕ = π / 8 :=
by
  sorry

end find_varphi_l325_32576


namespace value_of_c_minus_a_l325_32513

variables (a b c : ℝ)

theorem value_of_c_minus_a (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 60) : (c - a) = 30 :=
by
  have h3 : a + b = 90 := by sorry
  have h4 : b + c = 120 := by sorry
  -- now we have the required form of the problem statement
  -- c - a = 120 - 90
  sorry

end value_of_c_minus_a_l325_32513


namespace framed_painting_ratio_l325_32502

-- Definitions and conditions
def painting_width : ℕ := 20
def painting_height : ℕ := 30
def frame_side_width (x : ℕ) : ℕ := x
def frame_top_bottom_width (x : ℕ) : ℕ := 3 * x

-- Overall dimensions of the framed painting
def framed_painting_width (x : ℕ) : ℕ := painting_width + 2 * frame_side_width x
def framed_painting_height (x : ℕ) : ℕ := painting_height + 2 * frame_top_bottom_width x

-- Area of the painting
def painting_area : ℕ := painting_width * painting_height

-- Area of the frame
def frame_area (x : ℕ) : ℕ := framed_painting_width x * framed_painting_height x - painting_area

-- Condition that frame area equals painting area
def frame_area_condition (x : ℕ) : Prop := frame_area x = painting_area

-- Theoretical ratio of smaller to larger dimension of the framed painting
def dimension_ratio (x : ℕ) : ℚ := (framed_painting_width x : ℚ) / (framed_painting_height x)

-- The mathematical problem to prove
theorem framed_painting_ratio : ∃ x : ℕ, frame_area_condition x ∧ dimension_ratio x = (4 : ℚ) / 7 :=
by
  sorry

end framed_painting_ratio_l325_32502


namespace beef_weight_before_processing_l325_32515

-- Define the initial weight of the beef.
def W_initial := 1070.5882

-- Define the loss percentages.
def loss1 := 0.20
def loss2 := 0.15
def loss3 := 0.25

-- Define the final weight after all losses.
def W_final := 546.0

-- The main proof goal: show that W_initial results in W_final after considering the weight losses.
theorem beef_weight_before_processing (W_initial W_final : ℝ) (loss1 loss2 loss3 : ℝ) :
  W_final = (1 - loss3) * (1 - loss2) * (1 - loss1) * W_initial :=
by
  sorry

end beef_weight_before_processing_l325_32515


namespace explicit_formula_solution_set_l325_32519

noncomputable def f : ℝ → ℝ 
| x => if 0 < x ∧ x ≤ 4 then Real.log x / Real.log 2 else
       if -4 ≤ x ∧ x < 0 then Real.log (-x) / Real.log 2 else
       0

theorem explicit_formula (x : ℝ) :
  f x = if 0 < x ∧ x ≤ 4 then Real.log x / Real.log 2 else
        if -4 ≤ x ∧ x < 0 then Real.log (-x) / Real.log 2 else
        0 := 
by 
  sorry 

theorem solution_set (x : ℝ) : 
  (0 < x ∧ x < 1 ∨ -4 < x ∧ x < -1) ↔ x * f x < 0 := 
by
  sorry

end explicit_formula_solution_set_l325_32519


namespace animath_extortion_l325_32567

noncomputable def max_extortion (n : ℕ) : ℕ :=
2^n - n - 1 

theorem animath_extortion (n : ℕ) :
  ∃ steps : ℕ, steps < (2^n - n - 1) :=
sorry

end animath_extortion_l325_32567


namespace exists_positive_b_l325_32568

theorem exists_positive_b (m p : ℕ) (hm : 0 < m) (hp : Prime p)
  (h1 : m^2 ≡ 2 [MOD p])
  (ha : ∃ a : ℕ, 0 < a ∧ a^2 ≡ 2 - m [MOD p]) :
  ∃ b : ℕ, 0 < b ∧ b^2 ≡ m + 2 [MOD p] := 
  sorry

end exists_positive_b_l325_32568


namespace barrels_oil_difference_l325_32579

/--
There are two barrels of oil, A and B.
1. $\frac{1}{3}$ of the oil is poured from barrel A into barrel B.
2. $\frac{1}{5}$ of the oil is poured from barrel B back into barrel A.
3. Each barrel contains 24kg of oil after the transfers.

Prove that originally, barrel A had 6 kg more oil than barrel B.
-/
theorem barrels_oil_difference :
  ∃ (x y : ℝ), (y = 48 - x) ∧
  (24 = (2 / 3) * x + (1 / 5) * (48 - x + (1 / 3) * x)) ∧
  (24 = (48 - x + (1 / 3) * x) * (4 / 5)) ∧
  (x - y = 6) :=
by
  sorry

end barrels_oil_difference_l325_32579


namespace well_depth_l325_32573

variable (d : ℝ)

-- Conditions
def total_time (t₁ t₂ : ℝ) : Prop := t₁ + t₂ = 8.5
def stone_fall (t₁ : ℝ) : Prop := d = 16 * t₁^2 
def sound_travel (t₂ : ℝ) : Prop := t₂ = d / 1100

theorem well_depth : 
  ∃ t₁ t₂ : ℝ, total_time t₁ t₂ ∧ stone_fall d t₁ ∧ sound_travel d t₂ → d = 918.09 := 
by
  sorry

end well_depth_l325_32573


namespace bird_cost_l325_32504

variable (scost bcost : ℕ)

theorem bird_cost (h1 : bcost = 2 * scost)
                  (h2 : (5 * bcost + 3 * scost) = (3 * bcost + 5 * scost) + 20) :
                  scost = 10 ∧ bcost = 20 :=
by {
  sorry
}

end bird_cost_l325_32504


namespace heximal_to_binary_k_value_l325_32531

theorem heximal_to_binary_k_value (k : ℕ) (h : 10 * (6^3) + k * 6 + 5 = 239) : 
  k = 3 :=
by
  sorry

end heximal_to_binary_k_value_l325_32531


namespace george_run_speed_l325_32546

theorem george_run_speed (usual_distance : ℝ) (usual_speed : ℝ) (today_first_distance : ℝ) (today_first_speed : ℝ)
  (remaining_distance : ℝ) (expected_time : ℝ) :
  usual_distance = 1.5 →
  usual_speed = 3 →
  today_first_distance = 1 →
  today_first_speed = 2.5 →
  remaining_distance = 0.5 →
  expected_time = usual_distance / usual_speed →
  today_first_distance / today_first_speed + remaining_distance / (remaining_distance / (expected_time - today_first_distance / today_first_speed)) = expected_time →
  remaining_distance / (expected_time - today_first_distance / today_first_speed) = 5 :=
by sorry

end george_run_speed_l325_32546


namespace distance_between_foci_of_ellipse_l325_32593

theorem distance_between_foci_of_ellipse :
  let center : (ℝ × ℝ) := (8, 2)
  let a : ℝ := 16 / 2 -- half the length of the major axis
  let b : ℝ := 4 / 2  -- half the length of the minor axis
  let c : ℝ := Real.sqrt (a^2 - b^2) -- distance from the center to each focus
  2 * c = 4 * Real.sqrt 15 :=
by
  let center : (ℝ × ℝ) := (8, 2)
  let a : ℝ := 16 / 2 -- half the length of the major axis
  let b : ℝ := 4 / 2  -- half the length of the minor axis
  let c : ℝ := Real.sqrt (a^2 - b^2) -- distance from the center to each focus
  show 2 * c = 4 * Real.sqrt 15
  sorry

end distance_between_foci_of_ellipse_l325_32593


namespace problem1_problem2_l325_32586

-- Problem 1
theorem problem1 (a b : ℝ) (h : a ≠ b) : 
  (a / (a - b)) + (b / (b - a)) = 1 := 
sorry

-- Problem 2
theorem problem2 (m : ℝ) : 
  (m^2 - 4) / (4 + 4 * m + m^2) / ((m - 2) / (2 * m - 2)) * ((m + 2) / (m - 1)) = 2 := 
sorry

end problem1_problem2_l325_32586


namespace valid_pin_count_l325_32543

def total_pins : ℕ := 10^5

def restricted_pins (seq : List ℕ) : ℕ :=
  if seq = [3, 1, 4, 1] then 10 else 0

def valid_pins (seq : List ℕ) : ℕ :=
  total_pins - restricted_pins seq

theorem valid_pin_count :
  valid_pins [3, 1, 4, 1] = 99990 :=
by
  sorry

end valid_pin_count_l325_32543


namespace sugar_needed_287_163_l325_32508

theorem sugar_needed_287_163 :
  let sugar_stored := 287
  let additional_sugar_needed := 163
  sugar_stored + additional_sugar_needed = 450 :=
by
  let sugar_stored := 287
  let additional_sugar_needed := 163
  sorry

end sugar_needed_287_163_l325_32508


namespace invitational_tournament_l325_32523

theorem invitational_tournament (x : ℕ) (h : 2 * (x * (x - 1) / 2) = 56) : x = 8 :=
by
  sorry

end invitational_tournament_l325_32523


namespace find_ab_cd_l325_32584

variables (a b c d : ℝ)

def special_eq (x : ℝ) := 
  (min (20 * x + 19) (19 * x + 20) = (a * x + b) - abs (c * x + d))

theorem find_ab_cd (h : ∀ x : ℝ, special_eq a b c d x) :
  a * b + c * d = 380 := 
sorry

end find_ab_cd_l325_32584


namespace age_ratio_in_six_years_l325_32553

-- Definitions for Claire's and Pete's current ages
variables (c p : ℕ)

-- Conditions given in the problem
def condition1 : Prop := c - 3 = 2 * (p - 3)
def condition2 : Prop := p - 7 = (1 / 4) * (c - 7)

-- The proof problem statement
theorem age_ratio_in_six_years (c p : ℕ) (h1 : condition1 c p) (h2 : condition2 c p) : 
  (c + 6) = 3 * (p + 6) :=
sorry

end age_ratio_in_six_years_l325_32553


namespace kylie_daisies_l325_32532

theorem kylie_daisies :
  let initial_daisies := 5
  let additional_daisies := 9
  let total_daisies := initial_daisies + additional_daisies
  let daisies_left := total_daisies / 2
  daisies_left = 7 :=
by
  sorry

end kylie_daisies_l325_32532


namespace cube_plane_intersection_distance_l325_32524

theorem cube_plane_intersection_distance :
  let vertices := [(0, 0, 0), (0, 0, 6), (0, 6, 0), (0, 6, 6), (6, 0, 0), (6, 0, 6), (6, 6, 0), (6, 6, 6)]
  let P := (0, 3, 0)
  let Q := (2, 0, 0)
  let R := (2, 6, 6)
  let plane_equation := 3 * x - 2 * y - 2 * z + 6 = 0
  let S := (2, 0, 6)
  let T := (0, 6, 3)
  dist S T = 7 := sorry

end cube_plane_intersection_distance_l325_32524


namespace cost_of_five_dozen_apples_l325_32598

theorem cost_of_five_dozen_apples 
  (cost_four_dozen : ℝ) 
  (cost_one_dozen : ℝ) 
  (cost_five_dozen : ℝ) 
  (h1 : cost_four_dozen = 31.20) 
  (h2 : cost_one_dozen = cost_four_dozen / 4) 
  (h3 : cost_five_dozen = 5 * cost_one_dozen)
  : cost_five_dozen = 39.00 :=
sorry

end cost_of_five_dozen_apples_l325_32598


namespace perimeter_circumradius_ratio_neq_l325_32510

-- Define the properties for the equilateral triangle
def Triangle (A K R P : ℝ) : Prop :=
  P = 3 * A ∧ K = A^2 * Real.sqrt 3 / 4 ∧ R = A * Real.sqrt 3 / 3

-- Define the properties for the square
def Square (b k r p : ℝ) : Prop :=
  p = 4 * b ∧ k = b^2 ∧ r = b * Real.sqrt 2 / 2

-- Main statement to prove
theorem perimeter_circumradius_ratio_neq 
  (A b K R P k r p : ℝ)
  (hT : Triangle A K R P) 
  (hS : Square b k r p) :
  P / p ≠ R / r := 
by
  rcases hT with ⟨hP, hK, hR⟩
  rcases hS with ⟨hp, hk, hr⟩
  sorry

end perimeter_circumradius_ratio_neq_l325_32510


namespace elvis_writing_time_per_song_l325_32500

-- Define the conditions based on the problem statement
def total_studio_time_minutes := 300   -- 5 hours converted to minutes
def songs := 10
def recording_time_per_song := 12
def total_editing_time := 30

-- Define the total recording time
def total_recording_time := songs * recording_time_per_song

-- Define the total time available for writing songs
def total_writing_time := total_studio_time_minutes - total_recording_time - total_editing_time

-- Define the time to write each song
def time_per_song_writing := total_writing_time / songs

-- State the proof goal
theorem elvis_writing_time_per_song : time_per_song_writing = 15 := by
  sorry

end elvis_writing_time_per_song_l325_32500


namespace base_conversion_min_sum_l325_32527

theorem base_conversion_min_sum (a b : ℕ) (h : 3 * a + 5 = 5 * b + 3)
    (h_mod: 3 * a - 2 ≡ 0 [MOD 5])
    (valid_base_a : a >= 2)
    (valid_base_b : b >= 2):
  a + b = 14 := sorry

end base_conversion_min_sum_l325_32527


namespace certain_number_approximation_l325_32589

theorem certain_number_approximation (h1 : 2994 / 14.5 = 177) (h2 : 29.94 / x = 17.7) : x = 2.57455 := by
  sorry

end certain_number_approximation_l325_32589


namespace coin_arrangements_l325_32590

/-- We define the conditions for Robert's coin arrangement problem. -/
def gold_coins := 5
def silver_coins := 5
def total_coins := gold_coins + silver_coins

/-- We define the number of ways to arrange 5 gold coins and 5 silver coins in 10 positions,
using the binomial coefficient. -/
def arrangements_colors : ℕ := Nat.choose total_coins gold_coins

/-- We define the number of possible configurations for the orientation of the coins
such that no two adjacent coins are face to face. -/
def arrangements_orientation : ℕ := 11

/-- The total number of distinguishable arrangements of the coins. -/
def total_arrangements : ℕ := arrangements_colors * arrangements_orientation

theorem coin_arrangements : total_arrangements = 2772 := by
  -- The proof is omitted.
  sorry

end coin_arrangements_l325_32590


namespace power_of_four_l325_32529

theorem power_of_four (x : ℕ) (h : 5^29 * 4^x = 2 * 10^29) : x = 15 := by
  sorry

end power_of_four_l325_32529


namespace orchard_harvest_l325_32575

theorem orchard_harvest (sacks_per_section : ℕ) (sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 45 → sections = 8 → total_sacks = sacks_per_section * sections → total_sacks = 360 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end orchard_harvest_l325_32575


namespace simplify_and_evaluate_l325_32537

-- Defining the variables with given values
def a : ℚ := 1 / 2
def b : ℚ := -2

-- Expression to be simplified and evaluated
def expression : ℚ := (2 * a + b) ^ 2 - (2 * a - b) * (a + b) - 2 * (a - 2 * b) * (a + 2 * b)

-- The main theorem
theorem simplify_and_evaluate : expression = 37 := by
  sorry

end simplify_and_evaluate_l325_32537


namespace necessary_condition_for_q_implies_m_in_range_neg_p_or_neg_q_false_implies_x_in_range_l325_32585

-- Proof Problem 1
theorem necessary_condition_for_q_implies_m_in_range (m : ℝ) (h1 : 0 < m) :
  (∀ x : ℝ, 2 - m ≤ x ∧ x ≤ 2 + m → -2 ≤ x ∧ x ≤ 6) →
  0 < m ∧ m ≤ 4 :=
by
  sorry

-- Proof Problem 2
theorem neg_p_or_neg_q_false_implies_x_in_range (m : ℝ) (x : ℝ)
  (h2 : m = 2)
  (h3 : (x + 2) * (x - 6) ≤ 0)
  (h4 : 2 - m ≤ x ∧ x ≤ 2 + m)
  (h5 : ¬ ((x + 2) * (x - 6) > 0 ∨ x < 2 - m ∨ x > 2 + m)) :
  0 ≤ x ∧ x ≤ 4 :=
by
  sorry

end necessary_condition_for_q_implies_m_in_range_neg_p_or_neg_q_false_implies_x_in_range_l325_32585


namespace problem_solution_l325_32526

variable (a : ℝ)
def ellipse_p (a : ℝ) : Prop := (0 < a) ∧ (a < 5)
def quadratic_q (a : ℝ) : Prop := (-3 ≤ a) ∧ (a ≤ 3)
def p_or_q (a : ℝ) : Prop := ((0 < a ∧ a < 5) ∨ ((-3 ≤ a) ∧ (a ≤ 3)))
def p_and_q (a : ℝ) : Prop := ((0 < a ∧ a < 5) ∧ ((-3 ≤ a) ∧ (a ≤ 3)))

theorem problem_solution (a : ℝ) :
  (ellipse_p a → 0 < a ∧ a < 5) ∧ 
  (¬(ellipse_p a) ∧ quadratic_q a → -3 ≤ a ∧ a ≤ 0) ∧
  (p_or_q a ∧ ¬(p_and_q a) → 3 < a ∧ a < 5 ∨ (-3 ≤ a ∧ a ≤ 0)) :=
  by
  sorry

end problem_solution_l325_32526


namespace set_complement_l325_32569

variable {U : Set ℝ} (A : Set ℝ)

theorem set_complement :
  (U = {x : ℝ | x > 1}) →
  (A ⊆ U) →
  (U \ A = {x : ℝ | x > 9}) →
  (A = {x : ℝ | 1 < x ∧ x ≤ 9}) :=
by
  intros hU hA hC
  sorry

end set_complement_l325_32569


namespace find_g_at_3_l325_32503

theorem find_g_at_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 2) = 4 * x + 1) : g 3 = 23 / 3 :=
by
  sorry

end find_g_at_3_l325_32503


namespace largest_angle_is_176_l325_32564

-- Define the angles of the pentagon
def angle1 (y : ℚ) : ℚ := y
def angle2 (y : ℚ) : ℚ := 2 * y + 2
def angle3 (y : ℚ) : ℚ := 3 * y - 3
def angle4 (y : ℚ) : ℚ := 4 * y + 4
def angle5 (y : ℚ) : ℚ := 5 * y - 5

-- Define the function to calculate the largest angle
def largest_angle (y : ℚ) : ℚ := 5 * y - 5

-- Problem statement: Prove that the largest angle in the pentagon is 176 degrees
theorem largest_angle_is_176 (y : ℚ) (h : angle1 y + angle2 y + angle3 y + angle4 y + angle5 y = 540) :
  largest_angle y = 176 :=
by sorry

end largest_angle_is_176_l325_32564


namespace extra_flowers_l325_32512

-- Definitions from the conditions
def tulips : Nat := 57
def roses : Nat := 73
def daffodils : Nat := 45
def sunflowers : Nat := 35
def used_flowers : Nat := 181

-- Statement to prove
theorem extra_flowers : (tulips + roses + daffodils + sunflowers) - used_flowers = 29 := by
  sorry

end extra_flowers_l325_32512


namespace correct_operation_l325_32541

variable (a b : ℝ)

theorem correct_operation : 3 * a^2 * b - b * a^2 = 2 * a^2 * b := 
sorry

end correct_operation_l325_32541


namespace reciprocal_of_neg3_l325_32580

theorem reciprocal_of_neg3 : (1 / (-3) = -1 / 3) :=
by
  sorry

end reciprocal_of_neg3_l325_32580


namespace count_valid_ys_l325_32557

theorem count_valid_ys : 
  ∃ ys : Finset ℤ, ys.card = 4 ∧ ∀ y ∈ ys, (y - 3 > 0) ∧ ((y + 3) * (y - 3) * (y^2 + 9) < 2000) :=
by
  sorry

end count_valid_ys_l325_32557


namespace M_greater_than_N_l325_32540

variable (a : ℝ)

def M := 2 * a^2 - 4 * a
def N := a^2 - 2 * a - 3

theorem M_greater_than_N : M a > N a := by
  sorry

end M_greater_than_N_l325_32540


namespace parallel_vectors_sufficiency_l325_32592

noncomputable def parallel_vectors_sufficiency_problem (a b : ℝ × ℝ) (x : ℝ) : Prop :=
a = (1, x) ∧ b = (x, 4) →
(x = 2 → ∃ k : ℝ, k • a = b) ∧ (∃ k : ℝ, k • a = b → x = 2 ∨ x = -2)

theorem parallel_vectors_sufficiency (x : ℝ) :
  parallel_vectors_sufficiency_problem (1, x) (x, 4) x :=
sorry

end parallel_vectors_sufficiency_l325_32592


namespace geom_sum_3m_l325_32516

variable (a_n : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (m : ℕ)

axiom geom_sum_m : S m = 10
axiom geom_sum_2m : S (2 * m) = 30

theorem geom_sum_3m : S (3 * m) = 70 :=
by
  sorry

end geom_sum_3m_l325_32516


namespace three_configuration_m_separable_l325_32535

theorem three_configuration_m_separable
  {n m : ℕ} (A : Finset (Fin n)) (h : m ≥ n / 2) :
  ∀ (C : Finset (Fin n)), C.card = 3 → ∃ B : Finset (Fin n), B.card = m ∧ (∀ c ∈ C, ∃ b ∈ B, c ≠ b) :=
by
  sorry

end three_configuration_m_separable_l325_32535


namespace correct_product_exists_l325_32566

variable (a b : ℕ)

theorem correct_product_exists
  (h1 : a < 100)
  (h2 : 10 * (a % 10) + a / 10 = 14)
  (h3 : 14 * b = 182) : a * b = 533 := sorry

end correct_product_exists_l325_32566


namespace waiter_tables_l325_32517

theorem waiter_tables (total_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (tables : ℕ) :
  total_customers = 62 →
  left_customers = 17 →
  people_per_table = 9 →
  remaining_customers = total_customers - left_customers →
  tables = remaining_customers / people_per_table →
  tables = 5 := by
  sorry

end waiter_tables_l325_32517


namespace matrix_product_l325_32547

-- Define matrix A
def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, -1, 3], ![0, 3, 2], ![1, -3, 4]]

-- Define matrix B
def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 3, 0], ![2, 0, 4], ![3, 0, 1]]

-- Define the expected result matrix C
def C : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![9, 6, -1], ![12, 0, 14], ![7, 3, -8]]

-- The statement to prove
theorem matrix_product : A * B = C :=
by
  sorry

end matrix_product_l325_32547


namespace num_ways_to_choose_starting_lineup_l325_32542

-- Define conditions as Lean definitions
def team_size : ℕ := 12
def outfield_players : ℕ := 4

-- Define the function to compute the number of ways to choose the starting lineup
def choose_starting_lineup (team_size : ℕ) (outfield_players : ℕ) : ℕ :=
  team_size * Nat.choose (team_size - 1) outfield_players

-- The theorem to prove that the number of ways to choose the lineup is 3960
theorem num_ways_to_choose_starting_lineup : choose_starting_lineup team_size outfield_players = 3960 :=
  sorry

end num_ways_to_choose_starting_lineup_l325_32542


namespace arith_seq_formula_l325_32538

noncomputable def arith_seq (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n + a (n + 2) = 4 * n + 6

theorem arith_seq_formula (a : ℕ → ℤ) (h : arith_seq a) : ∀ n : ℕ, a n = 2 * n + 1 :=
by
  intros
  sorry

end arith_seq_formula_l325_32538


namespace find_relationship_l325_32520

noncomputable def log_equation (c d : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 1 → 6 * (Real.log (x) / Real.log (c))^2 + 5 * (Real.log (x) / Real.log (d))^2 = 12 * (Real.log (x))^2 / (Real.log (c) * Real.log (d))

theorem find_relationship (c d : ℝ) :
  log_equation c d → 
    (d = c ^ (5 / (6 + Real.sqrt 6)) ∨ d = c ^ (5 / (6 - Real.sqrt 6))) :=
by
  sorry

end find_relationship_l325_32520


namespace minutes_between_bathroom_visits_l325_32591

-- Definition of the conditions
def movie_duration_hours : ℝ := 2.5
def bathroom_uses : ℕ := 3
def minutes_per_hour : ℝ := 60

-- Theorem statement for the proof
theorem minutes_between_bathroom_visits :
  let total_movie_minutes := movie_duration_hours * minutes_per_hour
  let intervals := bathroom_uses + 1
  total_movie_minutes / intervals = 37.5 :=
by
  sorry

end minutes_between_bathroom_visits_l325_32591


namespace area_ratio_BDF_FDCE_l325_32562

-- Define the vertices of the triangle
variables {A B C : Point}
-- Define the points on the sides and midpoints
variables {E D F : Point}
-- Define angles and relevant properties
variables (angle_CBA : Angle B C A = 72)
variables (midpoint_E : Midpoint E A C)
variables (ratio_D : RatioSegment B D D C = 2)
-- Define intersection point F
variables (intersect_F : IntersectLineSegments (LineSegment A D) (LineSegment B E) = F)

theorem area_ratio_BDF_FDCE (h_angle : angle_CBA = 72) 
  (h_midpoint_E : midpoint_E) (h_ratio_D : ratio_D) (h_intersect_F : intersect_F)
  : area_ratio (Triangle.area B D F) (Quadrilateral.area F D C E) = 1 / 5 :=
sorry

end area_ratio_BDF_FDCE_l325_32562


namespace system_of_equations_soln_l325_32509

theorem system_of_equations_soln :
  {p : ℝ × ℝ | ∃ a : ℝ, (a * p.1 + p.2 = 2 * a + 3) ∧ (p.1 - a * p.2 = a + 4)} =
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 5} \ {⟨2, -1⟩} :=
by
  sorry

end system_of_equations_soln_l325_32509


namespace nonneg_int_repr_l325_32539

theorem nonneg_int_repr (n : ℕ) : ∃ (a b c : ℕ), (0 < a ∧ a < b ∧ b < c) ∧ n = a^2 + b^2 - c^2 :=
sorry

end nonneg_int_repr_l325_32539


namespace solve_for_y_l325_32581

theorem solve_for_y (x y : ℝ) (h : 2 * x - y = 6) : y = 2 * x - 6 :=
by
  sorry

end solve_for_y_l325_32581


namespace packets_for_dollars_l325_32514

variable (P R C : ℕ)

theorem packets_for_dollars :
  let dimes := 10 * C
  let taxable_dimes := 9 * C
  ∃ x, x = taxable_dimes * P / R :=
sorry

end packets_for_dollars_l325_32514


namespace caps_production_l325_32560

def caps1 : Int := 320
def caps2 : Int := 400
def caps3 : Int := 300

def avg_caps (caps1 caps2 caps3 : Int) : Int := (caps1 + caps2 + caps3) / 3

noncomputable def total_caps_after_four_weeks : Int :=
  caps1 + caps2 + caps3 + avg_caps caps1 caps2 caps3

theorem caps_production : total_caps_after_four_weeks = 1360 :=
by
  sorry

end caps_production_l325_32560


namespace f_monotonicity_l325_32548

noncomputable def f (a x : ℝ) : ℝ := a^x + x^2 - x * Real.log a

theorem f_monotonicity (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x : ℝ, x > 0 → deriv (f a) x > 0) ∧ (∀ x : ℝ, x < 0 → deriv (f a) x < 0) :=
by
  sorry

end f_monotonicity_l325_32548


namespace parity_of_expression_l325_32505

theorem parity_of_expression (a b c : ℤ) (h : (a + b + c) % 2 = 1) : (a^2 + b^2 - c^2 + 2*a*b) % 2 = 1 :=
by
sorry

end parity_of_expression_l325_32505


namespace extra_interest_is_correct_l325_32595

def principal : ℝ := 5000
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def time : ℝ := 2

def simple_interest (P R T : ℝ) : ℝ := P * R * T

def interest1 : ℝ := simple_interest principal rate1 time
def interest2 : ℝ := simple_interest principal rate2 time

def extra_interest : ℝ := interest1 - interest2

theorem extra_interest_is_correct : extra_interest = 600 := by
  sorry

end extra_interest_is_correct_l325_32595


namespace solve_combination_eq_l325_32506

theorem solve_combination_eq (x : ℕ) (h : x ≥ 3) : 
  (Nat.choose x 3 + Nat.choose x 2 = 12 * (x - 1)) ↔ (x = 9) := 
by
  sorry

end solve_combination_eq_l325_32506


namespace min_value_f_l325_32501

noncomputable def f (x : ℝ) : ℝ :=
  7 * (Real.sin x)^2 + 5 * (Real.cos x)^2 + 2 * Real.sin x

theorem min_value_f : ∃ x : ℝ, f x = 4.5 :=
  sorry

end min_value_f_l325_32501


namespace inequalities_proof_l325_32555

variables (x y z : ℝ)

def p := x + y + z
def q := x * y + y * z + z * x
def r := x * y * z

theorem inequalities_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (p x y z) ^ 2 ≥ 3 * (q x y z) ∧
  (p x y z) ^ 3 ≥ 27 * (r x y z) ∧
  (p x y z) * (q x y z) ≥ 9 * (r x y z) ∧
  (q x y z) ^ 2 ≥ 3 * (p x y z) * (r x y z) ∧
  (p x y z) ^ 2 * (q x y z) + 3 * (p x y z) * (r x y z) ≥ 4 * (q x y z) ^ 2 ∧
  (p x y z) ^ 3 + 9 * (r x y z) ≥ 4 * (p x y z) * (q x y z) ∧
  (p x y z) * (q x y z) ^ 2 ≥ 2 * (p x y z) ^ 2 * (r x y z) + 3 * (q x y z) * (r x y z) ∧
  (p x y z) * (q x y z) ^ 2 + 3 * (q x y z) * (r x y z) ≥ 4 * (p x y z) ^ 2 * (r x y z) ∧
  2 * (q x y z) ^ 3 + 9 * (r x y z) ^ 2 ≥ 7 * (p x y z) * (q x y z) * (r x y z) ∧
  (p x y z) ^ 4 + 4 * (q x y z) ^ 2 + 6 * (p x y z) * (r x y z) ≥ 5 * (p x y z) ^ 2 * (q x y z) :=
by sorry

end inequalities_proof_l325_32555


namespace sum_of_digits_smallest_N_l325_32530

/-- Define the probability Q(N) -/
def Q (N : ℕ) : ℚ :=
  ((2 * N) / 3 + 1) / (N + 1)

/-- Main mathematical statement to be proven in Lean 4 -/

theorem sum_of_digits_smallest_N (N : ℕ) (h1 : N > 9) (h2 : N % 6 = 0) (h3 : Q N < 7 / 10) : 
  (N.digits 10).sum = 3 :=
  sorry

end sum_of_digits_smallest_N_l325_32530


namespace contrapositive_even_statement_l325_32551

-- Translate the conditions to Lean 4 definitions
def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem contrapositive_even_statement (a b : Int) :
  (¬ is_even (a + b) → ¬ (is_even a ∧ is_even b)) ↔ 
  (is_even a ∧ is_even b → is_even (a + b)) :=
by sorry

end contrapositive_even_statement_l325_32551


namespace fixed_point_of_function_l325_32588

theorem fixed_point_of_function (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : 
  ∃ P : ℝ × ℝ, P = (1, 1) ∧ ∀ x : ℝ, (x = 1 → a^(x-1) = 1) :=
by
  sorry

end fixed_point_of_function_l325_32588


namespace num_of_possible_outcomes_l325_32522

def participants : Fin 6 := sorry  -- Define the participants as elements of Fin 6

theorem num_of_possible_outcomes : (6 * 5 * 4 = 120) :=
by {
  -- Prove this mathematical statement
  rfl
}

end num_of_possible_outcomes_l325_32522


namespace round_table_chairs_l325_32578

theorem round_table_chairs :
  ∃ x : ℕ, (2 * x + 2 * 7 = 26) ∧ x = 6 :=
by
  sorry

end round_table_chairs_l325_32578


namespace Robie_l325_32525

def initial_bags (X : ℕ) := (X - 2) + 3 = 4

theorem Robie's_initial_bags (X : ℕ) (h : initial_bags X) : X = 3 :=
by
  unfold initial_bags at h
  sorry

end Robie_l325_32525


namespace determine_OP_l325_32554

theorem determine_OP
  (a b c d e : ℝ)
  (h_dist_OA : a > 0)
  (h_dist_OB : b > 0)
  (h_dist_OC : c > 0)
  (h_dist_OD : d > 0)
  (h_dist_OE : e > 0)
  (h_c_le_d : c ≤ d)
  (P : ℝ)
  (hP : c ≤ P ∧ P ≤ d)
  (h_ratio : ∀ (P : ℝ) (hP : c ≤ P ∧ P ≤ d), (a - P) / (P - e) = (c - P) / (P - d)) :
  P = (ce - ad) / (a - c + e - d) :=
sorry

end determine_OP_l325_32554


namespace hyperbola_m_value_l325_32587

noncomputable def m_value : ℝ := 2 * (Real.sqrt 2 - 1)

theorem hyperbola_m_value (a : ℝ) (m : ℝ) (AF_2 AF_1 BF_2 BF_1 : ℝ)
  (h1 : a = 1)
  (h2 : AF_2 = m)
  (h3 : AF_1 = 2 + AF_2)
  (h4 : AF_1 = m + BF_2)
  (h5 : BF_2 = 2)
  (h6 : BF_1 = 4)
  (h7 : BF_1 = Real.sqrt 2 * AF_1) :
  m = m_value :=
by
  sorry

end hyperbola_m_value_l325_32587


namespace triangle_longest_side_l325_32596

theorem triangle_longest_side (y : ℝ) (h₁ : 8 + (y + 5) + (3 * y + 2) = 45) : 
  ∃ s1 s2 s3, s1 = 8 ∧ s2 = y + 5 ∧ s3 = 3 * y + 2 ∧ (s1 + s2 + s3 = 45) ∧ (s3 = 24.5) := 
by
  sorry

end triangle_longest_side_l325_32596


namespace inequality_example_l325_32545

theorem inequality_example (a b c : ℝ) (habc_pos : 0 < a ∧ 0 < b ∧ 0 < c) (habc_sum : a + b + c = 3) :
  18 * ((1 / ((3 - a) * (4 - a))) + (1 / ((3 - b) * (4 - b))) + (1 / ((3 - c) * (4 - c)))) + 2 * (a * b + b * c + c * a) ≥ 15 :=
by
  sorry

end inequality_example_l325_32545


namespace find_x_solution_l325_32556

theorem find_x_solution (x : ℝ) 
  (h : ∑' n:ℕ, ((-1)^(n+1)) * (2 * n + 1) * x^n = 16) : 
  x = -15/16 :=
sorry

end find_x_solution_l325_32556


namespace negation_of_P_l325_32536

-- Define the proposition P
def P : Prop := ∀ x : ℝ, x > Real.sin x

-- Formulate the negation of P
def neg_P : Prop := ∃ x : ℝ, x ≤ Real.sin x

-- State the theorem to be proved
theorem negation_of_P (hP : P) : neg_P :=
sorry

end negation_of_P_l325_32536


namespace license_plates_count_l325_32534

/-
Problem:
I want to choose a license plate that is 4 characters long,
where the first character is a letter,
the last two characters are either a letter or a digit,
and the second character can be a letter or a digit 
but must be the same as either the first or the third character.
Additionally, the fourth character must be different from the first three characters.
-/

def is_letter (c : Char) : Prop := c.isAlpha
def is_digit_or_letter (c : Char) : Prop := c.isAlpha || c.isDigit
noncomputable def count_license_plates : ℕ :=
  let first_char_options := 26
  let third_char_options := 36
  let second_char_options := 2
  let fourth_char_options := 34
  first_char_options * third_char_options * second_char_options * fourth_char_options

theorem license_plates_count : count_license_plates = 59904 := by
  sorry

end license_plates_count_l325_32534


namespace no_leopards_in_circus_l325_32550

theorem no_leopards_in_circus (L T : ℕ) (N : ℕ) (h₁ : L = N / 5) (h₂ : T = 5 * (N - T)) : 
  ∀ A, A = L + N → A = T + (N - T) → ¬ ∃ x, x ≠ L ∧ x ≠ T ∧ x ≠ (N - L - T) :=
by
  sorry

end no_leopards_in_circus_l325_32550


namespace joan_seashells_l325_32511

/-- Prove that Joan has 36 seashells given the initial conditions. -/
theorem joan_seashells :
  let initial_seashells := 79
  let given_mike := 63
  let found_more := 45
  let traded_seashells := 20
  let lost_seashells := 5
  (initial_seashells - given_mike + found_more - traded_seashells - lost_seashells) = 36 :=
by
  sorry

end joan_seashells_l325_32511


namespace speed_of_stream_l325_32599

theorem speed_of_stream (v : ℝ) 
    (h1 : ∀ (v : ℝ), v ≠ 0 → (80 / (36 + v) = 40 / (36 - v))) : 
    v = 12 := 
by 
    sorry

end speed_of_stream_l325_32599


namespace average_daily_sales_after_10_yuan_reduction_price_reduction_for_1200_yuan_profit_l325_32549

-- Conditions from the problem statement
def initial_daily_sales : ℕ := 20
def profit_per_box : ℕ := 40
def additional_sales_per_yuan_reduction : ℕ := 2

-- Part 1: New average daily sales after a 10 yuan reduction
theorem average_daily_sales_after_10_yuan_reduction :
  (initial_daily_sales + 10 * additional_sales_per_yuan_reduction) = 40 :=
  sorry

-- Part 2: Price reduction needed to achieve a daily sales profit of 1200 yuan
theorem price_reduction_for_1200_yuan_profit :
  ∃ (x : ℕ), 
  (profit_per_box - x) * (initial_daily_sales + x * additional_sales_per_yuan_reduction) = 1200 ∧ x = 20 :=
  sorry

end average_daily_sales_after_10_yuan_reduction_price_reduction_for_1200_yuan_profit_l325_32549


namespace largest_trailing_zeros_l325_32565

def count_trailing_zeros (n : Nat) : Nat :=
  if n = 0 then 0
  else Nat.min (Nat.factorial (n / 10)) (Nat.factorial (n / 5))

theorem largest_trailing_zeros :
  (count_trailing_zeros (4^3 * 5^6 * 6^5) > count_trailing_zeros (2^5 * 3^4 * 5^6)) ∧
  (count_trailing_zeros (4^3 * 5^6 * 6^5) > count_trailing_zeros (2^4 * 3^4 * 5^5)) ∧
  (count_trailing_zeros (4^3 * 5^6 * 6^5) > count_trailing_zeros (4^2 * 5^4 * 6^3)) :=
  sorry

end largest_trailing_zeros_l325_32565


namespace fibonacci_recurrence_l325_32561

theorem fibonacci_recurrence (f : ℕ → ℝ) (a b : ℝ) 
  (h₀ : f 0 = 1) 
  (h₁ : f 1 = 1) 
  (h₂ : ∀ n, f (n + 2) = f (n + 1) + f n)
  (h₃ : a + b = 1) 
  (h₄ : a * b = -1) 
  (h₅ : a > b) 
  : ∀ n, f n = (a ^ (n + 1) - b ^ (n + 1)) / Real.sqrt 5 := by
  sorry

end fibonacci_recurrence_l325_32561


namespace smallest_pos_mult_of_31_mod_97_l325_32563

theorem smallest_pos_mult_of_31_mod_97 {k : ℕ} (h : 31 * k % 97 = 6) : 31 * k = 2015 :=
sorry

end smallest_pos_mult_of_31_mod_97_l325_32563


namespace inequality_l325_32558

theorem inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) : 
  (a / b) + (b / c) + (c / a) + (b / a) + (a / c) + (c / b) + 6 ≥ 
  2 * Real.sqrt 2 * (Real.sqrt ((1 - a) / a) + Real.sqrt ((1 - b) / b) + Real.sqrt ((1 - c) / c)) :=
sorry

end inequality_l325_32558


namespace domain_of_function_l325_32552

noncomputable def is_domain_of_function (x : ℝ) : Prop :=
  (4 - x^2 ≥ 0) ∧ (x ≠ 1)

theorem domain_of_function :
  {x : ℝ | is_domain_of_function x} = {x : ℝ | -2 ≤ x ∧ x < 1} ∪ {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_function_l325_32552


namespace correct_inequality_l325_32559

theorem correct_inequality :
  1.6 ^ 0.3 > 0.9 ^ 3.1 :=
sorry

end correct_inequality_l325_32559


namespace Janet_pages_per_day_l325_32574

variable (J : ℕ)

-- Conditions
def belinda_pages_per_day : ℕ := 30
def janet_extra_pages_per_6_weeks : ℕ := 2100
def days_in_6_weeks : ℕ := 42

-- Prove that Janet reads 80 pages a day
theorem Janet_pages_per_day (h : J * days_in_6_weeks = (belinda_pages_per_day * days_in_6_weeks) + janet_extra_pages_per_6_weeks) : J = 80 := 
by sorry

end Janet_pages_per_day_l325_32574


namespace unique_solution_c_exceeds_s_l325_32528

-- Problem Conditions
def steers_cost : ℕ := 35
def cows_cost : ℕ := 40
def total_budget : ℕ := 1200

-- Definition of the solution conditions
def valid_purchase (s c : ℕ) : Prop := 
  steers_cost * s + cows_cost * c = total_budget ∧ s > 0 ∧ c > 0

-- Statement to prove
theorem unique_solution_c_exceeds_s :
  ∃ s c : ℕ, valid_purchase s c ∧ c > s ∧ ∀ (s' c' : ℕ), valid_purchase s' c' → s' = 8 ∧ c' = 17 :=
sorry

end unique_solution_c_exceeds_s_l325_32528


namespace marsh_ducks_l325_32582

theorem marsh_ducks (D : ℕ) (h1 : 58 = D + 21) : D = 37 := 
by {
  sorry
}

end marsh_ducks_l325_32582


namespace jerry_task_duration_l325_32544

def earnings_per_task : ℕ := 40
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7
def total_earnings : ℕ := 1400

theorem jerry_task_duration :
  (10 * 7 = 70) →
  (1400 / 40 = 35) →
  (70 / 35 = 2) →
  (total_earnings / earnings_per_task = (hours_per_day * days_per_week) / h) →
  h = 2 :=
by
  intros h1 h2 h3 h4
  -- proof steps (omitted)
  sorry

end jerry_task_duration_l325_32544


namespace distinct_shading_patterns_l325_32571

/-- How many distinct patterns can be made by shading exactly three of the sixteen squares 
    in a 4x4 grid, considering that patterns which can be matched by flips and/or turns are 
    not considered different? The answer is 8. -/
theorem distinct_shading_patterns : 
  (number_of_distinct_patterns : ℕ) = 8 :=
by
  /- Define the 4x4 Grid and the condition of shading exactly three squares, considering 
     flips and turns -/
  sorry

end distinct_shading_patterns_l325_32571


namespace possible_value_m_l325_32572

theorem possible_value_m (x m : ℝ) (h : ∃ x : ℝ, 2 * x^2 + 5 * x - m = 0) : m ≥ -25 / 8 := sorry

end possible_value_m_l325_32572


namespace last_digit_fifth_power_l325_32583

theorem last_digit_fifth_power (R : ℤ) : (R^5 - R) % 10 = 0 := 
sorry

end last_digit_fifth_power_l325_32583


namespace impossible_to_equalize_numbers_l325_32594

theorem impossible_to_equalize_numbers (nums : Fin 6 → ℤ) :
  ¬ (∃ n : ℤ, ∀ i : Fin 6, nums i = n) :=
sorry

end impossible_to_equalize_numbers_l325_32594


namespace max_area_rect_l325_32518

theorem max_area_rect (x y : ℝ) (h_perimeter : 2 * x + 2 * y = 40) : 
  x * y ≤ 100 :=
by
  sorry

end max_area_rect_l325_32518


namespace peach_count_l325_32570

theorem peach_count (n : ℕ) : n % 4 = 2 ∧ n % 6 = 4 ∧ n % 8 = 6 ∧ 120 ≤ n ∧ n ≤ 150 → n = 142 :=
sorry

end peach_count_l325_32570
