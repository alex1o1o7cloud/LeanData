import Mathlib

namespace smallest_consecutive_even_sum_560_l525_52527

theorem smallest_consecutive_even_sum_560 (n : ℕ) (h : 7 * n + 42 = 560) : n = 74 :=
  by
    sorry

end smallest_consecutive_even_sum_560_l525_52527


namespace evaluate_expression_l525_52558

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem evaluate_expression : (nabla (nabla 2 3) 4) = 16777219 :=
by sorry

end evaluate_expression_l525_52558


namespace time_addition_sum_l525_52596

/-- Given the start time of 3:15:20 PM and adding a duration of 
    305 hours, 45 minutes, and 56 seconds, the resultant hour, 
    minute, and second values sum to 26. -/
theorem time_addition_sum : 
  let current_hour := 15
  let current_minute := 15
  let current_second := 20
  let added_hours := 305
  let added_minutes := 45
  let added_seconds := 56
  let final_hour := ((current_hour + (added_hours % 12) + ((current_minute + added_minutes) / 60) + ((current_second + added_seconds) / 3600)) % 12)
  let final_minute := ((current_minute + added_minutes + ((current_second + added_seconds) / 60)) % 60)
  let final_second := ((current_second + added_seconds) % 60)
  final_hour + final_minute + final_second = 26 := 
  sorry

end time_addition_sum_l525_52596


namespace solve_fractional_equation_l525_52576

theorem solve_fractional_equation : 
  ∃ x : ℝ, (x - 1) / 2 = 1 - (3 * x + 2) / 5 ↔ x = 1 := 
sorry

end solve_fractional_equation_l525_52576


namespace sue_received_votes_l525_52568

theorem sue_received_votes (total_votes : ℕ) (sue_percentage : ℚ) (h1 : total_votes = 1000) (h2 : sue_percentage = 35 / 100) :
  (sue_percentage * total_votes) = 350 := by
  sorry

end sue_received_votes_l525_52568


namespace winning_lottery_ticket_is_random_l525_52598

-- Definitions of the events
inductive Event
| certain : Event
| impossible : Event
| random : Event

open Event

-- Conditions
def boiling_water_event : Event := certain
def lottery_ticket_event : Event := random
def athlete_running_30mps_event : Event := impossible
def draw_red_ball_event : Event := impossible

-- Problem Statement
theorem winning_lottery_ticket_is_random : 
    lottery_ticket_event = random :=
sorry

end winning_lottery_ticket_is_random_l525_52598


namespace arithmetic_expression_evaluation_l525_52542

theorem arithmetic_expression_evaluation : 
  ∃ (a b c d e f : Float),
  a - b * c / d + e = 0 ∧
  a = 5 ∧ b = 4 ∧ c = 3 ∧ d = 2 ∧ e = 1 := sorry

end arithmetic_expression_evaluation_l525_52542


namespace ab_cd_l525_52501

theorem ab_cd {a b c d : ℕ} {w x y z : ℕ}
  (hw : Prime w) (hx : Prime x) (hy : Prime y) (hz : Prime z)
  (horder : w < x ∧ x < y ∧ y < z)
  (hprod : w^a * x^b * y^c * z^d = 660) :
  (a + b) - (c + d) = 1 :=
by
  sorry

end ab_cd_l525_52501


namespace complete_square_transform_l525_52552

theorem complete_square_transform :
  ∀ x : ℝ, x^2 - 4 * x - 6 = 0 → (x - 2)^2 = 10 :=
by
  intros x h
  sorry

end complete_square_transform_l525_52552


namespace johns_drawings_l525_52569

theorem johns_drawings (total_pictures : ℕ) (back_pictures : ℕ) 
  (h1 : total_pictures = 15) (h2 : back_pictures = 9) : total_pictures - back_pictures = 6 := by
  -- proof goes here
  sorry

end johns_drawings_l525_52569


namespace range_of_a_l525_52545

theorem range_of_a (a : ℝ) (h1 : a > 0)
  (h2 : ∃ x : ℝ, abs (Real.sin x) > a)
  (h3 : ∀ x : ℝ, x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) → (Real.sin x)^2 + a * Real.sin x - 1 ≥ 0) :
  a ∈ Set.Ico (Real.sqrt 2 / 2) 1 :=
sorry

end range_of_a_l525_52545


namespace TruckloadsOfSand_l525_52507

theorem TruckloadsOfSand (S : ℝ) (totalMat dirt cement : ℝ) 
  (h1 : totalMat = 0.67) 
  (h2 : dirt = 0.33) 
  (h3 : cement = 0.17) 
  (h4 : totalMat = S + dirt + cement) : 
  S = 0.17 := 
  by 
    sorry

end TruckloadsOfSand_l525_52507


namespace dice_circle_probability_l525_52555

theorem dice_circle_probability :
  ∀ (d : ℕ), (2 ≤ d ∧ d ≤ 432) ∧
  ((∃ (x y : ℕ), (1 ≤ x ∧ x ≤ 6) ∧ (1 ≤ y ∧ y <= 6) ∧ d = x^3 + y^3)) →
  ((d * (d - 4) < 0) ↔ (d = 2)) →
  (∃ (P : ℚ), P = 1 / 36) :=
by
  sorry

end dice_circle_probability_l525_52555


namespace intersection_A_B_union_A_compB_l525_52503

-- Define the sets A and B
def A : Set ℝ := { x | x^2 + 3 * x - 10 < 0 }
def B : Set ℝ := { x | x^2 - 2 * x - 3 ≥ 0 }

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define the complement of B in the universal set
def comp_B : Set ℝ := { x | ¬ B x }

-- 1. Prove that A ∩ B = {x | -5 < x ∧ x ≤ -1}
theorem intersection_A_B :
  A ∩ B = { x | -5 < x ∧ x ≤ -1 } :=
by 
  sorry

-- 2. Prove that A ∪ (complement of B) = {x | -5 < x ∧ x < 3}
theorem union_A_compB :
  A ∪ comp_B = { x | -5 < x ∧ x < 3 } :=
by 
  sorry

end intersection_A_B_union_A_compB_l525_52503


namespace cases_in_1990_l525_52540

theorem cases_in_1990 (cases_1970 cases_2000 : ℕ) (linear_decrease : ℕ → ℝ) :
  cases_1970 = 300000 →
  cases_2000 = 600 →
  (∀ t, linear_decrease t = cases_1970 - (cases_1970 - cases_2000) * t / 30) →
  linear_decrease 20 = 100400 :=
by
  intros h1 h2 h3
  sorry

end cases_in_1990_l525_52540


namespace fraction_sum_eq_five_fourths_l525_52588

theorem fraction_sum_eq_five_fourths (a b c : ℚ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4) :
  (a + b) / c = 5 / 4 :=
by
  sorry

end fraction_sum_eq_five_fourths_l525_52588


namespace evaluate_expressions_l525_52543

theorem evaluate_expressions : (∀ (a b c d : ℤ), a = -(-3) → b = -(|-3|) → c = -(-(3^2)) → d = ((-3)^2) → b < 0) :=
by
  sorry

end evaluate_expressions_l525_52543


namespace percent_difference_l525_52550

theorem percent_difference :
  (0.90 * 40) - ((4 / 5) * 25) = 16 :=
by sorry

end percent_difference_l525_52550


namespace solve_equation_1_solve_equation_2_l525_52531

theorem solve_equation_1 (x : ℝ) : (2 * x - 1) ^ 2 - 25 = 0 ↔ x = 3 ∨ x = -2 := 
sorry

theorem solve_equation_2 (x : ℝ) : (1 / 3) * (x + 3) ^ 3 - 9 = 0 ↔ x = 0 := 
sorry

end solve_equation_1_solve_equation_2_l525_52531


namespace distance_interval_l525_52591

theorem distance_interval (d : ℝ) (h1 : ¬(d ≥ 8)) (h2 : ¬(d ≤ 7)) (h3 : ¬(d ≤ 6 → north)):
  7 < d ∧ d < 8 :=
by
  have h_d8 : d < 8 := by linarith
  have h_d7 : d > 7 := by linarith
  exact ⟨h_d7, h_d8⟩

end distance_interval_l525_52591


namespace area_of_rhombus_l525_52559

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 4) (h2 : d2 = 4) :
    (d1 * d2) / 2 = 8 := by
  sorry

end area_of_rhombus_l525_52559


namespace value_of_a2012_l525_52511

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) - a n = 2 * n

theorem value_of_a2012 (a : ℕ → ℤ) (h : seq a) : a 2012 = 2012 * 2011 :=
by 
  sorry

end value_of_a2012_l525_52511


namespace verify_sum_of_new_rates_proof_l525_52589

-- Given conditions and initial setup
variable (k : ℕ)
variable (h_initial : ℕ := 5 * k) -- Hanhan's initial hourly rate
variable (x_initial : ℕ := 4 * k) -- Xixi's initial hourly rate
variable (increment : ℕ := 20)    -- Increment in hourly rates

-- New rates after increment
variable (h_new : ℕ := h_initial + increment) -- Hanhan's new hourly rate
variable (x_new : ℕ := x_initial + increment) -- Xixi's new hourly rate

-- Given ratios
variable (initial_ratio : h_initial / x_initial = 5 / 4) 
variable (new_ratio : h_new / x_new = 6 / 5)

-- Target sum of the new hourly rates
def sum_of_new_rates_proof : Prop :=
  h_new + x_new = 220

theorem verify_sum_of_new_rates_proof : sum_of_new_rates_proof k :=
by
  sorry

end verify_sum_of_new_rates_proof_l525_52589


namespace ezekiel_painted_faces_l525_52534

noncomputable def cuboid_faces_painted (num_cuboids : ℕ) (faces_per_cuboid : ℕ) : ℕ :=
num_cuboids * faces_per_cuboid

theorem ezekiel_painted_faces :
  cuboid_faces_painted 8 6 = 48 := 
by
  sorry

end ezekiel_painted_faces_l525_52534


namespace sin_cos_product_l525_52521

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l525_52521


namespace find_ordered_pairs_of_b_c_l525_52522

theorem find_ordered_pairs_of_b_c : 
  ∃! (pairs : ℕ × ℕ), 
    (pairs.1 > 0 ∧ pairs.2 > 0) ∧ 
    (pairs.1 * pairs.1 = 4 * pairs.2) ∧ 
    (pairs.2 * pairs.2 = 4 * pairs.1) :=
sorry

end find_ordered_pairs_of_b_c_l525_52522


namespace ice_cream_stack_order_l525_52592

theorem ice_cream_stack_order (scoops : Finset ℕ) (h_scoops : scoops.card = 5) :
  (scoops.prod id) = 120 :=
by
  sorry

end ice_cream_stack_order_l525_52592


namespace tablecloth_covers_table_l525_52554

theorem tablecloth_covers_table
(length_ellipse : ℝ) (width_ellipse : ℝ) (length_tablecloth : ℝ) (width_tablecloth : ℝ)
(h1 : length_ellipse = 160)
(h2 : width_ellipse = 100)
(h3 : length_tablecloth = 140)
(h4 : width_tablecloth = 130) :
length_tablecloth >= width_ellipse ∧ width_tablecloth >= width_ellipse ∧
(length_tablecloth ^ 2 + width_tablecloth ^ 2) >= (length_ellipse ^ 2 + width_ellipse ^ 2) :=
by
  sorry

end tablecloth_covers_table_l525_52554


namespace min_value_frac_2_over_a_plus_3_over_b_l525_52593

theorem min_value_frac_2_over_a_plus_3_over_b 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hline : 2 * a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 25 :=
sorry

end min_value_frac_2_over_a_plus_3_over_b_l525_52593


namespace average_fixed_points_of_permutation_l525_52524

open Finset

noncomputable def average_fixed_points (n : ℕ) : ℕ :=
  1

theorem average_fixed_points_of_permutation (n : ℕ) :
  ∀ (σ : (Fin n) → (Fin n)), 
  (1: ℚ) = (1: ℕ) :=
by
  sorry

end average_fixed_points_of_permutation_l525_52524


namespace cartons_per_stack_l525_52556

-- Declare the variables and conditions
def total_cartons := 799
def stacks := 133

-- State the theorem
theorem cartons_per_stack : (total_cartons / stacks) = 6 := by
  sorry

end cartons_per_stack_l525_52556


namespace max_xy_l525_52551

theorem max_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 1) : xy <= 1 / 12 :=
by
  sorry

end max_xy_l525_52551


namespace remainder_divisible_by_4_l525_52570

theorem remainder_divisible_by_4 (z : ℕ) (h : z % 4 = 0) : ((z * (2 + 4 + z) + 3) % 2) = 1 :=
by
  sorry

end remainder_divisible_by_4_l525_52570


namespace initial_oranges_l525_52504

variable (x : ℕ)
variable (total_oranges : ℕ := 8)
variable (oranges_from_joyce : ℕ := 3)

theorem initial_oranges (h : total_oranges = x + oranges_from_joyce) : x = 5 := by
  sorry

end initial_oranges_l525_52504


namespace frank_candy_bags_l525_52510

theorem frank_candy_bags (total_candies : ℕ) (candies_per_bag : ℕ) (bags : ℕ) 
  (h1 : total_candies = 22) (h2 : candies_per_bag = 11) : bags = 2 :=
by
  sorry

end frank_candy_bags_l525_52510


namespace total_length_of_segments_in_new_figure_l525_52581

-- Defining the given conditions.
def left_side := 10
def top_side := 3
def right_side := 8
def segments_removed_from_bottom := [2, 1, 2] -- List of removed segments from the bottom.

-- This is the theorem statement that confirms the total length of the new figure's sides.
theorem total_length_of_segments_in_new_figure :
  (left_side + top_side + right_side) = 21 :=
by
  -- This is where the proof would be written.
  sorry

end total_length_of_segments_in_new_figure_l525_52581


namespace geometric_sum_ratio_l525_52553

theorem geometric_sum_ratio (a₁ q : ℝ) (h₁ : q ≠ 1) (h₂ : (1 - q^4) / (1 - q^2) = 5) :
  (1 - q^8) / (1 - q^4) = 17 := 
by
  sorry

end geometric_sum_ratio_l525_52553


namespace unique_number_encoding_l525_52518

-- Defining participants' score ranges 
def score_range := {x : ℕ // x ≤ 5}

-- Defining total score
def total_score (s1 s2 s3 s4 s5 s6 : score_range) : ℕ := 
  s1.val + s2.val + s3.val + s4.val + s5.val + s6.val

-- Main statement to encode participant's scores into a unique number
theorem unique_number_encoding (s1 s2 s3 s4 s5 s6 : score_range) :
  ∃ n : ℕ, ∃ s : ℕ, 
    s = total_score s1 s2 s3 s4 s5 s6 ∧ 
    n = s * 10^6 + s1.val * 10^5 + s2.val * 10^4 + s3.val * 10^3 + s4.val * 10^2 + s5.val * 10 + s6.val := 
sorry

end unique_number_encoding_l525_52518


namespace range_of_x_l525_52546

theorem range_of_x (a b x : ℝ) (h1 : a + b = 1) (h2 : 0 < a) (h3 : 0 < b) :
  (1 / a + 4 / b ≥ |2 * x - 1| - |x + 1|) → (-7 ≤ x ∧ x ≤ 11) :=
by
  -- we provide the exact statement we aim to prove.
  sorry

end range_of_x_l525_52546


namespace kimberly_peanuts_per_visit_l525_52541

theorem kimberly_peanuts_per_visit 
  (trips : ℕ) (total_peanuts : ℕ) 
  (h1 : trips = 3) 
  (h2 : total_peanuts = 21) : 
  total_peanuts / trips = 7 :=
by
  sorry

end kimberly_peanuts_per_visit_l525_52541


namespace ribeye_steak_cost_l525_52547

/-- Define the conditions in Lean -/
def appetizer_cost : ℕ := 8
def wine_cost : ℕ := 3
def wine_glasses : ℕ := 2
def dessert_cost : ℕ := 6
def total_spent : ℕ := 38
def tip_percentage : ℚ := 0.20

/-- Proving the cost of the ribeye steak before the discount -/
theorem ribeye_steak_cost (S : ℚ) (h : 20 + (S / 2) + (tip_percentage * (20 + S)) = total_spent) : S = 20 :=
by
  sorry

end ribeye_steak_cost_l525_52547


namespace product_modulo_l525_52532

theorem product_modulo : ∃ m : ℕ, 0 ≤ m ∧ m < 30 ∧ (33 * 77 * 99) % 30 = m := 
  sorry

end product_modulo_l525_52532


namespace initial_pencils_count_l525_52529

-- Define the conditions
def students : ℕ := 25
def pencils_per_student : ℕ := 5

-- Statement of the proof problem
theorem initial_pencils_count : students * pencils_per_student = 125 :=
by
  sorry

end initial_pencils_count_l525_52529


namespace map_representation_l525_52582

-- Defining the conditions
noncomputable def map_scale : ℝ := 28 -- 1 inch represents 28 miles

-- Defining the specific instance provided in the problem
def inches_represented : ℝ := 13.7
def miles_represented : ℝ := 383.6

-- Statement of the problem
theorem map_representation (D : ℝ) : (D / map_scale) = (D : ℝ) / 28 := 
by
  -- Prove the statement
  sorry

end map_representation_l525_52582


namespace eddie_games_l525_52586

-- Define the study block duration in minutes
def study_block_duration : ℕ := 60

-- Define the homework time in minutes
def homework_time : ℕ := 25

-- Define the time for one game in minutes
def game_time : ℕ := 5

-- Define the total time Eddie can spend playing games
noncomputable def time_for_games : ℕ := study_block_duration - homework_time

-- Define the number of games Eddie can play
noncomputable def number_of_games : ℕ := time_for_games / game_time

-- Theorem stating the number of games Eddie can play while completing his homework
theorem eddie_games : number_of_games = 7 := by
  sorry

end eddie_games_l525_52586


namespace domain_of_function_l525_52575

open Real

theorem domain_of_function : 
  ∀ x, 
    (x + 1 ≠ 0) ∧ 
    (-x^2 - 3 * x + 4 > 0) ↔ 
    (-4 < x ∧ x < -1) ∨ ( -1 < x ∧ x < 1) := 
by 
  sorry

end domain_of_function_l525_52575


namespace integer_solutions_l525_52535

theorem integer_solutions (m n : ℤ) (h1 : m * (m + n) = n * 12) (h2 : n * (m + n) = m * 3) :
  (m = 4 ∧ n = 2) :=
by sorry

end integer_solutions_l525_52535


namespace fraction_of_air_conditioned_rooms_rented_l525_52557

variable (R : ℚ)
variable (h1 : R > 0)
variable (rented_rooms : ℚ := (3/4) * R)
variable (air_conditioned_rooms : ℚ := (3/5) * R)
variable (not_rented_rooms : ℚ := (1/4) * R)
variable (air_conditioned_not_rented_rooms : ℚ := (4/5) * not_rented_rooms)
variable (air_conditioned_rented_rooms : ℚ := air_conditioned_rooms - air_conditioned_not_rented_rooms)
variable (fraction_air_conditioned_rented : ℚ := air_conditioned_rented_rooms / air_conditioned_rooms)

theorem fraction_of_air_conditioned_rooms_rented :
  fraction_air_conditioned_rented = (2/3) := by
  sorry

end fraction_of_air_conditioned_rooms_rented_l525_52557


namespace brokerage_percentage_calculation_l525_52515

theorem brokerage_percentage_calculation
  (face_value : ℝ)
  (discount_percentage : ℝ)
  (cost_price : ℝ)
  (h_face_value : face_value = 100)
  (h_discount_percentage : discount_percentage = 6)
  (h_cost_price : cost_price = 94.2) :
  ((cost_price - (face_value - (discount_percentage / 100 * face_value))) / cost_price * 100) = 0.2124 := 
by
  sorry

end brokerage_percentage_calculation_l525_52515


namespace solution_set_of_quadratic_inequality_l525_52533

theorem solution_set_of_quadratic_inequality (x : ℝ) :
  (x^2 - 2*x - 3 > 0) ↔ (x > 3 ∨ x < -1) := 
sorry

end solution_set_of_quadratic_inequality_l525_52533


namespace find_b_value_l525_52595

-- Define the conditions: line equation and given range for b
def line_eq (x : ℝ) (b : ℝ) : ℝ := b - x

-- Define the points P, Q, S
def P (b : ℝ) : ℝ × ℝ := ⟨0, b⟩
def Q (b : ℝ) : ℝ × ℝ := ⟨b, 0⟩
def S (b : ℝ) : ℝ × ℝ := ⟨6, b - 6⟩

-- Define the area ratio condition
def area_ratio_condition (b : ℝ) : Prop :=
  (0 < b ∧ b < 6) ∧ ((6 - b) / b) ^ 2 = 4 / 25

-- Define the main theorem to prove
theorem find_b_value (b : ℝ) : area_ratio_condition b → b = 4.3 := by
  sorry

end find_b_value_l525_52595


namespace polynomial_divisibility_l525_52587

theorem polynomial_divisibility (P : Polynomial ℝ) (h_nonconstant : ∃ n : ℕ, P.degree = n ∧ n ≥ 1)
  (h_div : ∀ x : ℝ, P.eval (x^3 + 8) = 0 → P.eval (x^2 - 2*x + 4) = 0) :
  ∃ a : ℝ, ∃ n : ℕ, a ≠ 0 ∧ P = Polynomial.C a * Polynomial.X ^ n :=
sorry

end polynomial_divisibility_l525_52587


namespace inverse_function_f_l525_52561

noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)

noncomputable def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 2 - 1

theorem inverse_function_f : ∀ x > 0, f_inv (f x) = x :=
by
  intro x hx
  dsimp [f, f_inv]
  sorry

end inverse_function_f_l525_52561


namespace trader_profit_l525_52580

theorem trader_profit (P : ℝ) :
  let buy_price := 0.80 * P
  let sell_price := 1.20 * P
  sell_price - P = 0.20 * P := 
by
  sorry

end trader_profit_l525_52580


namespace camryn_flute_practice_interval_l525_52528

theorem camryn_flute_practice_interval (x : ℕ) 
  (h1 : ∃ n : ℕ, n * 11 = 33) 
  (h2 : x ∣ 33) 
  (h3 : x < 11) 
  (h4 : x > 1) 
  : x = 3 := 
sorry

end camryn_flute_practice_interval_l525_52528


namespace probability_of_vowel_initials_l525_52564

/-- In a class with 26 students, each student has unique initials that are double letters
    (i.e., AA, BB, ..., ZZ). If the vowels are A, E, I, O, U, and W, then the probability of
    randomly picking a student whose initials are vowels is 3/13. -/
theorem probability_of_vowel_initials :
  let total_students := 26
  let vowels := ['A', 'E', 'I', 'O', 'U', 'W']
  let num_vowels := 6
  let probability := num_vowels / total_students
  probability = 3 / 13 :=
by
  sorry

end probability_of_vowel_initials_l525_52564


namespace f_zero_f_odd_range_of_x_l525_52590

variable {f : ℝ → ℝ}

axiom func_property (x y : ℝ) : f (x + y) = f x + f y
axiom f_third : f (1 / 3) = 1
axiom f_positive (x : ℝ) : x > 0 → f x > 0

-- Part (1)
theorem f_zero : f 0 = 0 :=
sorry

-- Part (2)
theorem f_odd (x : ℝ) : f (-x) = -f x :=
sorry

-- Part (3)
theorem range_of_x (x : ℝ) : f x + f (2 + x) < 2 → x < -2 / 3 :=
sorry

end f_zero_f_odd_range_of_x_l525_52590


namespace smallest_n_for_coloring_l525_52536

theorem smallest_n_for_coloring (n : ℕ) : n = 4 :=
sorry

end smallest_n_for_coloring_l525_52536


namespace range_of_m_l525_52597

theorem range_of_m (m : ℝ) :
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ x - (m^2 - 2 * m + 4) * y + 6 > 0) →
  -1 < m ∧ m < 3 :=
by
  intros h
  rcases h with ⟨x, y, hx, hy, hineq⟩
  rw [hx, hy] at hineq
  sorry

end range_of_m_l525_52597


namespace percentage_increase_l525_52514

theorem percentage_increase (N M P : ℝ) (h : M = N * (1 + P / 100)) : ((M - N) / N) * 100 = P :=
by
  sorry

end percentage_increase_l525_52514


namespace initial_birds_was_one_l525_52583

def initial_birds (b : Nat) : Prop :=
  b + 4 = 5

theorem initial_birds_was_one : ∃ b, initial_birds b ∧ b = 1 :=
by
  use 1
  unfold initial_birds
  sorry

end initial_birds_was_one_l525_52583


namespace solve_for_x_l525_52594

theorem solve_for_x (x : ℝ) (hx₁ : x ≠ 3) (hx₂ : x ≠ -2) 
  (h : (x + 5) / (x - 3) = (x - 2) / (x + 2)) : x = -1 / 3 :=
by
  sorry

end solve_for_x_l525_52594


namespace integer_part_sqrt_sum_l525_52573

theorem integer_part_sqrt_sum {a b c : ℤ} 
  (h_a : |a| = 4) 
  (h_b_sqrt : b^2 = 9) 
  (h_c_cubert : c^3 = -8) 
  (h_order : a > b ∧ b > c) 
  : (⌊ Real.sqrt (a + b + c) ⌋) = 2 := 
by 
  sorry

end integer_part_sqrt_sum_l525_52573


namespace garden_contains_53_33_percent_tulips_l525_52585

theorem garden_contains_53_33_percent_tulips :
  (∃ (flowers : ℕ) (yellow tulips flowers_in_garden : ℕ) (yellow_flowers blue_flowers yellow_tulips blue_tulips : ℕ),
    flowers_in_garden = yellow_flowers + blue_flowers ∧
    yellow_flowers = 4 * flowers / 5 ∧
    blue_flowers = 1 * flowers / 5 ∧
    yellow_tulips = yellow_flowers / 2 ∧
    blue_tulips = 2 * blue_flowers / 3 ∧
    (yellow_tulips + blue_tulips) = 8 * flowers / 15) →
    0.5333 ∈ ([46.67, 53.33, 60, 75, 80] : List ℝ) := sorry

end garden_contains_53_33_percent_tulips_l525_52585


namespace find_number_l525_52513

theorem find_number :
  ∃ x : ℚ, x * (-1/2) = 1 ↔ x = -2 := 
sorry

end find_number_l525_52513


namespace minimum_area_of_rectangle_l525_52539

theorem minimum_area_of_rectangle (x y : ℝ) (h1 : x = 3) (h2 : y = 4) : 
  (min_area : ℝ) = (2.3 * 3.3) :=
by
  have length_min := x - 0.7
  have width_min := y - 0.7
  have min_area := length_min * width_min
  sorry

end minimum_area_of_rectangle_l525_52539


namespace batsman_new_average_l525_52519

-- Let A be the average score before the 16th inning
def avg_before (A : ℝ) : Prop :=
  ∃ total_runs: ℝ, total_runs = 15 * A

-- Condition 1: The batsman makes 64 runs in the 16th inning
def score_in_16th_inning := 64

-- Condition 2: This increases his average by 3 runs
def avg_increase (A : ℝ) : Prop :=
  A + 3 = (15 * A + score_in_16th_inning) / 16

theorem batsman_new_average (A : ℝ) (h1 : avg_before A) (h2 : avg_increase A) :
  (A + 3) = 19 :=
sorry

end batsman_new_average_l525_52519


namespace bob_work_days_per_week_l525_52560

theorem bob_work_days_per_week (daily_hours : ℕ) (monthly_hours : ℕ) (average_days_per_month : ℕ) (days_per_week : ℕ)
  (h1 : daily_hours = 10)
  (h2 : monthly_hours = 200)
  (h3 : average_days_per_month = 30)
  (h4 : days_per_week = 7) :
  (monthly_hours / daily_hours) / (average_days_per_month / days_per_week) = 5 := by
  -- Now we will skip the proof itself. The focus here is on the structure.
  sorry

end bob_work_days_per_week_l525_52560


namespace max_period_initial_phase_function_l525_52574

theorem max_period_initial_phase_function 
  (A ω ϕ : ℝ) 
  (f : ℝ → ℝ)
  (h1 : A = 1/2) 
  (h2 : ω = 6) 
  (h3 : ϕ = π/4) 
  (h4 : ∀ x, f x = A * Real.sin (ω * x + ϕ)) : 
  ∀ x, f x = (1/2) * Real.sin (6 * x + (π/4)) :=
by
  sorry

end max_period_initial_phase_function_l525_52574


namespace find_circle_parameter_l525_52565

theorem find_circle_parameter (c : ℝ) :
  (∃ x y : ℝ, x^2 + 8 * x + y^2 - 2 * y + c = 0 ∧ ((x + 4)^2 + (y - 1)^2 = 25)) → c = -8 :=
by
  sorry

end find_circle_parameter_l525_52565


namespace correct_average_l525_52577

theorem correct_average (n : Nat) (incorrect_avg correct_mark incorrect_mark : ℝ) 
  (h1 : n = 30) (h2 : incorrect_avg = 60) (h3 : correct_mark = 15) (h4 : incorrect_mark = 90) :
  (incorrect_avg * n - incorrect_mark + correct_mark) / n = 57.5 :=
by
  sorry

end correct_average_l525_52577


namespace symmetric_line_equation_l525_52563

theorem symmetric_line_equation (x y : ℝ) :
  (∃ x y : ℝ, 3 * x + 4 * y = 2) →
  (4 * x + 3 * y = 2) :=
by
  intros h
  sorry

end symmetric_line_equation_l525_52563


namespace intersection_eq_interval_l525_52584

def P : Set ℝ := {x | x * (x - 3) < 0}
def Q : Set ℝ := {x | |x| < 2}

theorem intersection_eq_interval : P ∩ Q = {x | 0 < x ∧ x < 2} :=
by
  sorry

end intersection_eq_interval_l525_52584


namespace factorial_ratio_l525_52572

theorem factorial_ratio : Nat.factorial 16 / (Nat.factorial 6 * Nat.factorial 10) = 5120 := by
  sorry

end factorial_ratio_l525_52572


namespace sum_of_reciprocals_eq_one_l525_52526

theorem sum_of_reciprocals_eq_one {x y : ℝ} (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : x + y = (x * y) ^ 2) : (1/x) + (1/y) = 1 :=
sorry

end sum_of_reciprocals_eq_one_l525_52526


namespace num_dimes_is_3_l525_52578

noncomputable def num_dimes (pennies nickels dimes quarters : ℕ) : ℕ :=
  dimes

theorem num_dimes_is_3 (h_total_coins : pennies + nickels + dimes + quarters = 11)
  (h_total_value : pennies + 5 * nickels + 10 * dimes + 25 * quarters = 118)
  (h_at_least_one_each : 0 < pennies ∧ 0 < nickels ∧ 0 < dimes ∧ 0 < quarters) :
  num_dimes pennies nickels dimes quarters = 3 :=
sorry

end num_dimes_is_3_l525_52578


namespace least_number_to_add_l525_52544

theorem least_number_to_add (x : ℕ) (h : 1056 % 23 = 21) : (1056 + x) % 23 = 0 ↔ x = 2 :=
by {
    sorry
}

end least_number_to_add_l525_52544


namespace percent_round_trip_tickets_is_100_l525_52579

noncomputable def percent_round_trip_tickets (P : ℕ) (x : ℚ) : ℚ :=
  let R := x / 0.20
  R

theorem percent_round_trip_tickets_is_100
  (P : ℕ)
  (x : ℚ)
  (h : 20 * x = P) :
  percent_round_trip_tickets P (x / P) = 100 :=
by
  sorry

end percent_round_trip_tickets_is_100_l525_52579


namespace find_negative_integer_l525_52549

theorem find_negative_integer (M : ℤ) (h_neg : M < 0) (h_eq : M^2 + M = 12) : M = -4 :=
sorry

end find_negative_integer_l525_52549


namespace smaller_angle_at_8_15_l525_52500

noncomputable def hour_hand_position (h m : ℕ) : ℝ := (↑h % 12) * 30 + (↑m / 60) * 30

noncomputable def minute_hand_position (m : ℕ) : ℝ := ↑m / 60 * 360

noncomputable def angle_between_hands (h m : ℕ) : ℝ :=
  let θ := |hour_hand_position h m - minute_hand_position m|
  min θ (360 - θ)

theorem smaller_angle_at_8_15 : angle_between_hands 8 15 = 157.5 := by
  sorry

end smaller_angle_at_8_15_l525_52500


namespace range_of_a_l525_52562

-- Definitions for the conditions
def p (x : ℝ) := x ≤ 2
def q (x : ℝ) (a : ℝ) := x < a + 2

-- Theorem statement
theorem range_of_a (a : ℝ) : (∀ x : ℝ, q x a → p x) → a ≤ 0 := by
  sorry

end range_of_a_l525_52562


namespace negation_of_p_l525_52599

noncomputable def p : Prop := ∀ x : ℝ, x > 0 → 2 * x^2 + 1 > 0

theorem negation_of_p : (∃ x : ℝ, x > 0 ∧ 2 * x^2 + 1 ≤ 0) ↔ ¬p :=
by
  sorry

end negation_of_p_l525_52599


namespace Nick_riding_speed_l525_52509

theorem Nick_riding_speed (Alan_speed Maria_ratio Nick_ratio : ℝ) 
(h1 : Alan_speed = 6) (h2 : Maria_ratio = 3/4) (h3 : Nick_ratio = 4/3) : 
Nick_ratio * (Maria_ratio * Alan_speed) = 6 := 
by 
  sorry

end Nick_riding_speed_l525_52509


namespace cost_price_to_marked_price_l525_52525

theorem cost_price_to_marked_price (MP CP SP : ℝ)
  (h1 : SP = MP * 0.87)
  (h2 : SP = CP * 1.359375) :
  (CP / MP) * 100 = 64 := by
  sorry

end cost_price_to_marked_price_l525_52525


namespace union_of_sets_l525_52502

open Set

theorem union_of_sets (M N : Set ℝ) (hM : M = {x | -3 < x ∧ x < 1}) (hN : N = {x | x ≤ -3}) :
  M ∪ N = {x | x < 1} := by
  sorry

end union_of_sets_l525_52502


namespace sin_double_angle_l525_52508

theorem sin_double_angle (θ : ℝ) (h : Real.tan θ + 1 / Real.tan θ = 4) : Real.sin (2 * θ) = 1 / 2 :=
by
  sorry

end sin_double_angle_l525_52508


namespace problem_1_problem_2_problem_3_l525_52517

-- First proof statement
theorem problem_1 : 2017^2 - 2016 * 2018 = 1 :=
by
  sorry

-- Definitions for the second problem
variables {a b : ℤ}

-- Second proof statement
theorem problem_2 (h1 : a + b = 7) (h2 : a * b = -1) : (a + b)^2 = 49 :=
by
  sorry

-- Third proof statement (part of the second problem)
theorem problem_3 (h1 : a + b = 7) (h2 : a * b = -1) : a^2 - 3 * a * b + b^2 = 54 :=
by
  sorry

end problem_1_problem_2_problem_3_l525_52517


namespace sequence_count_zeros_ones_15_l525_52537

-- Definition of the problem
def count_sequences (n : Nat) : Nat := sorry -- Function calculating the number of valid sequences

-- The theorem stating that for sequence length 15, the number of such sequences is 266
theorem sequence_count_zeros_ones_15 : count_sequences 15 = 266 := 
by {
  sorry -- Proof goes here
}

end sequence_count_zeros_ones_15_l525_52537


namespace tan_cos_solution_count_l525_52512

theorem tan_cos_solution_count : 
  ∃ (n : ℕ), n = 5 ∧ ∀ x ∈ Set.Icc 0 (2 * Real.pi), Real.tan (2 * x) = Real.cos (x / 2) → x ∈ Set.Icc 0 (2 * Real.pi) :=
sorry

end tan_cos_solution_count_l525_52512


namespace volume_of_given_wedge_l525_52505

noncomputable def volume_of_wedge (d : ℝ) (angle : ℝ) : ℝ := 
  let r := d / 2
  let height := d
  let cos_angle := Real.cos angle
  (r^2 * height * Real.pi / 2) * cos_angle

theorem volume_of_given_wedge :
  volume_of_wedge 20 (Real.pi / 6) = 1732 * Real.pi :=
by {
  -- The proof logic will go here.
  sorry
}

end volume_of_given_wedge_l525_52505


namespace ratio_of_still_lifes_to_portraits_l525_52548

noncomputable def total_paintings : ℕ := 80
noncomputable def portraits : ℕ := 16
noncomputable def still_lifes : ℕ := total_paintings - portraits
axiom still_lifes_is_multiple_of_portraits : ∃ k : ℕ, still_lifes = k * portraits

theorem ratio_of_still_lifes_to_portraits : still_lifes / portraits = 4 := by
  -- proof would go here
  sorry

end ratio_of_still_lifes_to_portraits_l525_52548


namespace total_ages_is_32_l525_52567

variable (a b c : ℕ)
variable (h_b : b = 12)
variable (h_a : a = b + 2)
variable (h_c : b = 2 * c)

theorem total_ages_is_32 (h_b : b = 12) (h_a : a = b + 2) (h_c : b = 2 * c) : a + b + c = 32 :=
by
  sorry

end total_ages_is_32_l525_52567


namespace triangle_to_initial_position_l525_52566

-- Definitions for triangle vertices
structure Point where
  x : Int
  y : Int

def p1 : Point := { x := 0, y := 0 }
def p2 : Point := { x := 6, y := 0 }
def p3 : Point := { x := 0, y := 4 }

-- Definitions for transformations
def rotate90 (p : Point) : Point := { x := -p.y, y := p.x }
def rotate180 (p : Point) : Point := { x := -p.x, y := -p.y }
def rotate270 (p : Point) : Point := { x := p.y, y := -p.x }
def reflect_y_eq_x (p : Point) : Point := { x := p.y, y := p.x }
def reflect_y_eq_neg_x (p : Point) : Point := { x := -p.y, y := -p.x }

-- Definitions for combination of transformations
-- This part defines how to combine transformations, e.g., as a sequence of three transformations.
def transform (fs : List (Point → Point)) (p : Point) : Point :=
  fs.foldl (fun acc f => f acc) p

-- The total number of valid sequences that return the triangle to its original position
def valid_sequences_count : Int := 6

-- Lean 4 statement
theorem triangle_to_initial_position : valid_sequences_count = 6 := by
  sorry

end triangle_to_initial_position_l525_52566


namespace finish_work_in_time_l525_52520

noncomputable def work_in_days_A (DA : ℕ) := DA
noncomputable def work_in_days_B (DA : ℕ) := DA / 2
noncomputable def combined_work_rate (DA : ℕ) : ℚ := 1 / work_in_days_A DA + 2 / work_in_days_A DA

theorem finish_work_in_time (DA : ℕ) (h_combined_rate : combined_work_rate DA = 0.25) : DA = 12 :=
sorry

end finish_work_in_time_l525_52520


namespace salary_increase_is_57point35_percent_l525_52530

variable (S : ℝ)

-- Assume Mr. Blue receives a 12% raise every year.
def annualRaise : ℝ := 1.12

-- After four years
theorem salary_increase_is_57point35_percent (h : annualRaise ^ 4 = 1.5735):
  ((annualRaise ^ 4 - 1) * S) / S = 0.5735 :=
by
  sorry

end salary_increase_is_57point35_percent_l525_52530


namespace polynomial_evaluation_l525_52516

theorem polynomial_evaluation :
  102^5 - 5 * 102^4 + 10 * 102^3 - 10 * 102^2 + 5 * 102 - 1 = 101^5 :=
by
  sorry

end polynomial_evaluation_l525_52516


namespace total_doughnuts_made_l525_52538

def num_doughnuts_per_box : ℕ := 10
def num_boxes_sold : ℕ := 27
def doughnuts_given_away : ℕ := 30

theorem total_doughnuts_made :
  num_boxes_sold * num_doughnuts_per_box + doughnuts_given_away = 300 :=
by
  sorry

end total_doughnuts_made_l525_52538


namespace average_cost_price_per_meter_l525_52506

noncomputable def average_cost_per_meter (total_cost total_meters : ℝ) : ℝ :=
  total_cost / total_meters

theorem average_cost_price_per_meter :
  let silk_cost := 416.25
  let silk_meters := 9.25
  let cotton_cost := 337.50
  let cotton_meters := 7.5
  let wool_cost := 378.0
  let wool_meters := 6.0
  let total_cost := silk_cost + cotton_cost + wool_cost
  let total_meters := silk_meters + cotton_meters + wool_meters
  average_cost_per_meter total_cost total_meters = 49.75 := by
  sorry

end average_cost_price_per_meter_l525_52506


namespace deg_d_eq_6_l525_52571

theorem deg_d_eq_6
  (f d q : Polynomial ℝ)
  (r : Polynomial ℝ)
  (hf : f.degree = 15)
  (hdq : (d * q + r) = f)
  (hq : q.degree = 9)
  (hr : r.degree = 4) :
  d.degree = 6 :=
by sorry

end deg_d_eq_6_l525_52571


namespace factorization_correct_l525_52523

theorem factorization_correct (x : ℤ) :
  (3 * (x + 3) * (x + 4) * (x + 7) * (x + 8) - 2 * x^2) =
  ((3 * x^2 + 35 * x + 72) * (x + 3) * (x + 6)) :=
by sorry

end factorization_correct_l525_52523
