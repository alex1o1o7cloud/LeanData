import Mathlib

namespace rational_square_plus_one_positive_l149_149330

theorem rational_square_plus_one_positive (x : ℚ) : x^2 + 1 > 0 :=
sorry

end rational_square_plus_one_positive_l149_149330


namespace not_sufficient_nor_necessary_condition_l149_149478

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

end not_sufficient_nor_necessary_condition_l149_149478


namespace speed_of_student_B_l149_149787

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l149_149787


namespace expression_value_l149_149869

-- Step c: Definitions based on conditions
def base1 : ℤ := -2
def exponent1 : ℕ := 4^2
def base2 : ℕ := 1
def exponent2 : ℕ := 3^3

-- The Lean statement for the problem
theorem expression_value :
  base1 ^ exponent1 + base2 ^ exponent2 = 65537 := by
  sorry

end expression_value_l149_149869


namespace average_weight_estimate_l149_149506

noncomputable def average_weight (female_students male_students : ℕ) (avg_weight_female avg_weight_male : ℕ) : ℝ :=
  (female_students / (female_students + male_students) : ℝ) * avg_weight_female +
  (male_students / (female_students + male_students) : ℝ) * avg_weight_male

theorem average_weight_estimate :
  average_weight 504 596 49 57 = (504 / 1100 : ℝ) * 49 + (596 / 1100 : ℝ) * 57 :=
by
  sorry

end average_weight_estimate_l149_149506


namespace smallest_x_with_18_factors_and_factors_18_21_exists_l149_149714

def has_18_factors (x : ℕ) : Prop :=
(x.factors.length == 18)

def is_factor (a b : ℕ) : Prop :=
(b % a == 0)

theorem smallest_x_with_18_factors_and_factors_18_21_exists :
  ∃ x : ℕ, has_18_factors x ∧ is_factor 18 x ∧ is_factor 21 x ∧ ∀ y : ℕ, has_18_factors y ∧ is_factor 18 y ∧ is_factor 21 y → y ≥ x :=
sorry

end smallest_x_with_18_factors_and_factors_18_21_exists_l149_149714


namespace bicycle_speed_B_l149_149766

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l149_149766


namespace face_value_of_share_l149_149431

theorem face_value_of_share (FV : ℝ) (dividend_percent : ℝ) (interest_percent : ℝ) (market_value : ℝ) :
  dividend_percent = 0.09 → 
  interest_percent = 0.12 →
  market_value = 33 →
  (0.09 * FV = 0.12 * 33) → FV = 44 :=
by
  intros
  sorry

end face_value_of_share_l149_149431


namespace gerbil_weights_l149_149579

theorem gerbil_weights
  (puffy muffy scruffy fluffy tuffy : ℕ)
  (h1 : puffy = 2 * muffy)
  (h2 : muffy = scruffy - 3)
  (h3 : scruffy = 12)
  (h4 : fluffy = muffy + tuffy)
  (h5 : fluffy = puffy / 2)
  (h6 : tuffy = puffy / 2) :
  puffy + muffy + tuffy = 36 := by
  sorry

end gerbil_weights_l149_149579


namespace number_to_match_l149_149961

def twenty_five_percent_less (x: ℕ) : ℕ := 3 * x / 4

def one_third_more (n: ℕ) : ℕ := 4 * n / 3

theorem number_to_match (n : ℕ) (x : ℕ) 
  (h1 : x = 80) 
  (h2 : one_third_more n = twenty_five_percent_less x) : n = 45 :=
by
  -- Proof is skipped as per the instruction
  sorry

end number_to_match_l149_149961


namespace triangle_area_l149_149194

theorem triangle_area (l1 l2 : ℝ → ℝ → Prop)
  (h1 : ∀ x y, l1 x y ↔ 3 * x - y + 12 = 0)
  (h2 : ∀ x y, l2 x y ↔ 3 * x + 2 * y - 6 = 0) :
  ∃ A : ℝ, A = 9 :=
by
  sorry

end triangle_area_l149_149194


namespace martin_distance_l149_149228

-- Define the given conditions
def speed : ℝ := 12.0
def time : ℝ := 6.0

-- State the theorem we want to prove
theorem martin_distance : speed * time = 72.0 := by
  sorry

end martin_distance_l149_149228


namespace tree_leaves_not_shed_l149_149151

-- Definitions of conditions based on the problem.
variable (initial_leaves : ℕ) (shed_week1 shed_week2 shed_week3 shed_week4 shed_week5 remaining_leaves : ℕ)

-- Setting the conditions
def conditions :=
  initial_leaves = 5000 ∧
  shed_week1 = initial_leaves / 5 ∧
  shed_week2 = 30 * (initial_leaves - shed_week1) / 100 ∧
  shed_week3 = 60 * shed_week2 / 100 ∧
  shed_week4 = 50 * (initial_leaves - shed_week1 - shed_week2 - shed_week3) / 100 ∧
  shed_week5 = 2 * shed_week3 / 3 ∧
  remaining_leaves = initial_leaves - shed_week1 - shed_week2 - shed_week3 - shed_week4 - shed_week5

-- The proof statement
theorem tree_leaves_not_shed (h : conditions initial_leaves shed_week1 shed_week2 shed_week3 shed_week4 shed_week5 remaining_leaves) :
  remaining_leaves = 560 :=
sorry

end tree_leaves_not_shed_l149_149151


namespace pearJuicePercentageCorrect_l149_149044

-- Define the conditions
def dozen : ℕ := 12
def pears := dozen
def oranges := dozen
def pearJuiceFrom3Pears : ℚ := 8
def orangeJuiceFrom2Oranges : ℚ := 10
def juiceBlendPears : ℕ := 4
def juiceBlendOranges : ℕ := 4
def pearJuicePerPear : ℚ := pearJuiceFrom3Pears / 3
def orangeJuicePerOrange : ℚ := orangeJuiceFrom2Oranges / 2
def totalPearJuice : ℚ := juiceBlendPears * pearJuicePerPear
def totalOrangeJuice : ℚ := juiceBlendOranges * orangeJuicePerOrange
def totalJuice : ℚ := totalPearJuice + totalOrangeJuice

-- Prove that the percentage of pear juice in the blend is 34.78%
theorem pearJuicePercentageCorrect : 
  (totalPearJuice / totalJuice) * 100 = 34.78 := by
  sorry

end pearJuicePercentageCorrect_l149_149044


namespace discount_percentage_l149_149227

theorem discount_percentage (sale_price original_price : ℝ) (h1 : sale_price = 480) (h2 : original_price = 600) : 
  100 * (original_price - sale_price) / original_price = 20 := by 
  sorry

end discount_percentage_l149_149227


namespace sum_of_7a_and_3b_l149_149744

theorem sum_of_7a_and_3b (a b : ℤ) (h : a + b = 1998) : 7 * a + 3 * b ≠ 6799 :=
by sorry

end sum_of_7a_and_3b_l149_149744


namespace root_magnitude_conditions_l149_149495

theorem root_magnitude_conditions (p : ℝ) (h : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 + r2 = -p) ∧ (r1 * r2 = -12)) :
  (∃ r1 r2 : ℝ, (r1 ≠ r2) ∧ |r1| > 2 ∨ |r2| > 2) ∧ (∀ r1 r2 : ℝ, (r1 + r2 = -p) ∧ (r1 * r2 = -12) → |r1| * |r2| ≤ 14) :=
by
  -- Proof of the theorem goes here
  sorry

end root_magnitude_conditions_l149_149495


namespace no_solutions_in_domain_l149_149395

-- Define the function g
def g (x : ℝ) : ℝ := -0.5 * x^2 + x + 3

-- Define the condition on the domain of g
def in_domain (x : ℝ) : Prop := x ≥ -3 ∧ x ≤ 3

-- State the theorem to be proved
theorem no_solutions_in_domain :
  ∀ x : ℝ, in_domain x → ¬ (g (g x) = 3) :=
by
  -- Provide a placeholder for the proof
  sorry

end no_solutions_in_domain_l149_149395


namespace bicycle_speed_B_l149_149767

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l149_149767


namespace dad_contribution_is_correct_l149_149443

noncomputable def carl_savings_weekly : ℕ := 25
noncomputable def savings_duration_weeks : ℕ := 6
noncomputable def coat_cost : ℕ := 170

-- Total savings after 6 weeks
noncomputable def total_savings : ℕ := carl_savings_weekly * savings_duration_weeks

-- Amount used to pay bills in the seventh week
noncomputable def bills_payment : ℕ := total_savings / 3

-- Money left after paying bills
noncomputable def remaining_savings : ℕ := total_savings - bills_payment

-- Amount needed from Dad
noncomputable def dad_contribution : ℕ := coat_cost - remaining_savings

theorem dad_contribution_is_correct : dad_contribution = 70 := by
  sorry

end dad_contribution_is_correct_l149_149443


namespace min_value_inverse_sum_l149_149534

theorem min_value_inverse_sum (a m n : ℝ) (a_pos : 0 < a) (a_ne_one : a ≠ 1) (mn_pos : 0 < m * n) :
  (a^(1-1) = 1) ∧ (m + n = 1) → (1/m + 1/n) = 4 :=
by
  sorry

end min_value_inverse_sum_l149_149534


namespace solution_set_of_floor_eqn_l149_149675

theorem solution_set_of_floor_eqn:
  ∀ x y : ℝ, 
  (⌊x⌋ * ⌊x⌋ + ⌊y⌋ * ⌊y⌋ = 4) ↔ 
  ((2 ≤ x ∧ x < 3 ∧ 0 ≤ y ∧ y < 1) ∨
   (0 ≤ x ∧ x < 1 ∧ 2 ≤ y ∧ y < 3) ∨
   (-2 ≤ x ∧ x < -1 ∧ 0 ≤ y ∧ y < 1) ∨
   (0 ≤ x ∧ x < 1 ∧ -2 ≤ y ∧ y < -1)) :=
by
  sorry

end solution_set_of_floor_eqn_l149_149675


namespace flute_player_count_l149_149383

-- Define the total number of people in the orchestra
def total_people : Nat := 21

-- Define the number of people in each section
def sebastian : Nat := 1
def brass : Nat := 4 + 2 + 1
def strings : Nat := 3 + 1 + 1
def woodwinds_excluding_flutes : Nat := 3
def maestro : Nat := 1

-- Calculate the number of accounted people
def accounted_people : Nat := sebastian + brass + strings + woodwinds_excluding_flutes + maestro

-- State the number of flute players
def flute_players : Nat := total_people - accounted_people

-- The theorem stating the number of flute players
theorem flute_player_count : flute_players = 4 := by
  unfold flute_players accounted_people total_people sebastian brass strings woodwinds_excluding_flutes maestro
  -- Need to evaluate the expressions step by step to reach the final number 4.
  -- (Or simply "sorry" since we are skipping the proof steps)
  sorry

end flute_player_count_l149_149383


namespace mirror_area_l149_149059

theorem mirror_area (frame_length frame_width frame_border_length : ℕ) (mirror_area : ℕ)
  (h_frame_length : frame_length = 100)
  (h_frame_width : frame_width = 130)
  (h_frame_border_length : frame_border_length = 15)
  (h_mirror_area : mirror_area = (frame_length - 2 * frame_border_length) * (frame_width - 2 * frame_border_length)) :
  mirror_area = 7000 := by 
    sorry

end mirror_area_l149_149059


namespace expression_meaning_l149_149539

variable (m n : ℤ) -- Assuming m and n are integers for the context.

theorem expression_meaning : 2 * (m - n) = 2 * (m - n) := 
by
  -- It simply follows from the definition of the expression
  sorry

end expression_meaning_l149_149539


namespace outer_boundary_diameter_l149_149433

theorem outer_boundary_diameter (d_pond : ℝ) (w_picnic : ℝ) (w_track : ℝ)
  (h_pond_diam : d_pond = 16) (h_picnic_width : w_picnic = 10) (h_track_width : w_track = 4) :
  2 * (d_pond / 2 + w_picnic + w_track) = 44 :=
by
  -- We avoid the entire proof, we only assert the statement in Lean
  sorry

end outer_boundary_diameter_l149_149433


namespace more_birds_than_storks_l149_149273

def initial_storks : ℕ := 5
def initial_birds : ℕ := 3
def additional_birds : ℕ := 4

def total_birds : ℕ := initial_birds + additional_birds

def stork_vs_bird_difference : ℕ := total_birds - initial_storks

theorem more_birds_than_storks : stork_vs_bird_difference = 2 := by
  sorry

end more_birds_than_storks_l149_149273


namespace find_speed_B_l149_149862

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l149_149862


namespace least_n_satisfies_inequality_l149_149637

theorem least_n_satisfies_inequality : ∃ n : ℕ, (1 : ℚ) / n - (1 : ℚ) / (n + 1) < 1 / 15 ∧ ∀ m : ℕ, (1 : ℚ) / m - (1 : ℚ) / (m + 1) < 1 / 15 -> ¬ (m < n) := 
sorry

end least_n_satisfies_inequality_l149_149637


namespace polygon_max_sides_l149_149500

theorem polygon_max_sides (n : ℕ) (h : (n - 2) * 180 < 2005) : n ≤ 13 :=
by {
  sorry
}

end polygon_max_sides_l149_149500


namespace problem_solution_l149_149188

variable (U : Set Real) (a b : Real) (t : Real)
variable (A B : Set Real)

-- Conditions
def condition1 : U = Set.univ := sorry

def condition2 : ∀ x, a ≠ 0 → ax^2 + 2 * x + b > 0 ↔ x ≠ -1 / a := sorry

def condition3 : a > b := sorry

def condition4 : t = (a^2 + b^2) / (a - b) := sorry

def condition5 : ∀ m, (∀ x, |x + 1| - |x - 3| ≤ m^2 - 3 * m) → m ∈ B := sorry

-- To Prove
theorem problem_solution : A ∩ (Set.univ \ B) = {m : Real | 2 * Real.sqrt 2 ≤ m ∧ m < 4} := sorry

end problem_solution_l149_149188


namespace johns_groceries_cost_l149_149215

noncomputable def calculate_total_cost : ℝ := 
  let bananas_cost := 6 * 2
  let bread_cost := 2 * 3
  let butter_cost := 3 * 5
  let cereal_cost := 4 * (6 - 0.25 * 6)
  let subtotal := bananas_cost + bread_cost + butter_cost + cereal_cost
  if subtotal >= 50 then
    subtotal - 10
  else
    subtotal

-- The statement to prove
theorem johns_groceries_cost : calculate_total_cost = 41 := by
  sorry

end johns_groceries_cost_l149_149215


namespace hyperbola_eccentricity_l149_149484

-- Definitions based on conditions
def hyperbola (x y : ℝ) (a : ℝ) := x^2 / a^2 - y^2 / 5 = 1

-- Main theorem
theorem hyperbola_eccentricity (a : ℝ) (c : ℝ) (h_focus : c = 3) (h_hyperbola : hyperbola 0 0 a) (focus_condition : c^2 = a^2 + 5) :
  c / a = 3 / 2 :=
by
  sorry

end hyperbola_eccentricity_l149_149484


namespace plywood_cut_difference_l149_149105

/-- A proof problem to determine the positive difference between the greatest possible
perimeter and the least possible perimeter of congruent pieces resulting from cutting 
a 6-foot by 9-foot rectangular plywood into 6 congruent rectangles with no wood leftover 
or lost --/
theorem plywood_cut_difference :
  ∃ (perimeter_max perimeter_min : ℕ), 
  let piece1 := 1 * 9
  let piece2 := 1 * 6
  let piece3 := 2 * 3
  let perimeter1 := 2 * (1 + 9)
  let perimeter2 := 2 * (1 + 6)
  let perimeter3 := 2 * (2 + 3)
  perimeter_max = perimeter1 ∧
  perimeter_min = perimeter3 ∧
  (perimeter_max - perimeter_min) = 10 :=
sorry

end plywood_cut_difference_l149_149105


namespace total_players_l149_149980

-- Definitions of the given conditions
def K : ℕ := 10
def Kho_only : ℕ := 40
def Both : ℕ := 5

-- The lean statement that captures the problem of proving the total number of players equals 50
theorem total_players : (K - Both) + Kho_only + Both = 50 :=
by
  -- Placeholder for the proof
  sorry

end total_players_l149_149980


namespace plywood_perimeter_difference_l149_149135

noncomputable theory

open classical

theorem plywood_perimeter_difference :
  ∃ (rect1 rect2 : ℕ) (a b : ℕ),
  (rect1 = 6 ∧ rect2 = 9 ∧ rect1 % 6 = 0 ∧ rect2 % 6 = 0) ∧ 
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≤ 20) ∧
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≥ 10) ∧ 
  (20 - 10 = 10) :=
by
  exists (6, 9, 6, 9)
  sorry

end plywood_perimeter_difference_l149_149135


namespace polynomial_value_at_2_l149_149159

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

-- Define the transformation rules for each v_i according to Horner's Rule
def v0 : ℝ := 1
def v1 (x : ℝ) : ℝ := (v0 * x) - 12
def v2 (x : ℝ) : ℝ := (v1 x * x) + 60
def v3 (x : ℝ) : ℝ := (v2 x * x) - 160

-- State the theorem to be proven
theorem polynomial_value_at_2 : v3 2 = -80 := 
by 
  -- Since this is just a Lean 4 statement, we include sorry to defer proof
  sorry

end polynomial_value_at_2_l149_149159


namespace dollars_saved_is_correct_l149_149955

noncomputable def blender_in_store_price : ℝ := 120
noncomputable def juicer_in_store_price : ℝ := 80
noncomputable def blender_tv_price : ℝ := 4 * 28 + 12
noncomputable def total_in_store_price_with_discount : ℝ := (blender_in_store_price + juicer_in_store_price) * 0.90
noncomputable def dollars_saved : ℝ := total_in_store_price_with_discount - blender_tv_price

theorem dollars_saved_is_correct :
  dollars_saved = 56 := by
  sorry

end dollars_saved_is_correct_l149_149955


namespace length_of_bridge_l149_149732

/-- What is the length of a bridge (in meters), which a train 156 meters long and travelling at 45 km/h can cross in 40 seconds? -/
theorem length_of_bridge (train_length: ℕ) (train_speed_kmh: ℕ) (time_seconds: ℕ) (bridge_length: ℕ) :
  train_length = 156 →
  train_speed_kmh = 45 →
  time_seconds = 40 →
  bridge_length = 344 :=
by {
  sorry
}

end length_of_bridge_l149_149732


namespace number_to_add_l149_149415

theorem number_to_add (a b n : ℕ) (h_a : a = 425897) (h_b : b = 456) (h_n : n = 47) : 
  (a + n) % b = 0 :=
by
  rw [h_a, h_b, h_n]
  sorry

end number_to_add_l149_149415


namespace above_265_is_234_l149_149559

namespace PyramidArray

-- Definition of the pyramid structure and identifying important properties
def is_number_in_pyramid (n : ℕ) : Prop :=
  ∃ k : ℕ, (k^2 - (k - 1)^2) / 2 ≥ n ∧ (k^2 - (k - 1)^2) / 2 < n + (2 * k - 1)

def row_start (k : ℕ) : ℕ :=
  (k - 1)^2 + 1

def row_end (k : ℕ) : ℕ :=
  k^2

def number_above (n : ℕ) (r : ℕ) : ℕ :=
  row_start r + ((n - row_start (r + 1)) % (2 * (r + 1) - 1))

theorem above_265_is_234 : 
  (number_above 265 16) = 234 := 
sorry

end PyramidArray

end above_265_is_234_l149_149559


namespace least_n_l149_149629

theorem least_n (n : ℕ) (h_pos : n > 0) (h_ineq : 1 / n - 1 / (n + 1) < 1 / 15) : n = 4 :=
sorry

end least_n_l149_149629


namespace minimum_people_who_like_both_l149_149522

open Nat

theorem minimum_people_who_like_both (total : ℕ) (mozart : ℕ) (bach : ℕ)
  (h_total: total = 100) (h_mozart: mozart = 87) (h_bach: bach = 70) :
  ∃ x, x = mozart + bach - total ∧ x ≥ 57 :=
by
  sorry

end minimum_people_who_like_both_l149_149522


namespace sqrt_exp_sum_eq_eight_sqrt_two_l149_149024

theorem sqrt_exp_sum_eq_eight_sqrt_two : 
  (Real.sqrt ((5 - 4 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 4 * Real.sqrt 2) ^ 2) = 8 * Real.sqrt 2) :=
by
  sorry

end sqrt_exp_sum_eq_eight_sqrt_two_l149_149024


namespace monotone_increasing_solve_inequality_l149_149925

section MathProblem

variable {f : ℝ → ℝ}

theorem monotone_increasing (h₁ : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x + f y) 
(h₂ : ∀ x : ℝ, 1 < x → 0 < f x) : 
∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ := sorry

theorem solve_inequality (h₃ : f 2 = 1) (h₄ : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x + f y) 
(h₅ : ∀ x : ℝ, 1 < x → 0 < f x) :
∀ x : ℝ, 0 < x → f x + f (x - 3) ≤ 2 → 3 < x ∧ x ≤ 4 := sorry

end MathProblem

end monotone_increasing_solve_inequality_l149_149925


namespace smallest_n_l149_149086

theorem smallest_n (n : ℕ) (h1 : ∃ k1, 4 * n = k1^2) (h2 : ∃ k2, 3 * n = k2^3) : n = 144 :=
sorry

end smallest_n_l149_149086


namespace smallest_n_l149_149084

theorem smallest_n (n : ℕ) (h₁ : ∃ k1 : ℕ, 4 * n = k1 ^ 2) (h₂ : ∃ k2 : ℕ, 3 * n = k2 ^ 3) : n = 18 :=
sorry

end smallest_n_l149_149084


namespace least_n_l149_149606

theorem least_n (n : ℕ) (h : n > 0) : 
  (1 / (n : ℝ) - 1 / (n + 1) < 1 / 15) ↔ n ≥ 4 := 
begin
  sorry
end

end least_n_l149_149606


namespace perimeters_positive_difference_l149_149122

theorem perimeters_positive_difference (orig_length orig_width : ℝ) (num_pieces : ℕ)
  (congruent_division : ∃ (length width : ℝ), length * width = (orig_length * orig_width) / num_pieces)
  (greatest_perimeter least_perimeter : ℝ)
  (h1 : greatest_perimeter = 2 * (1.5 + 9))
  (h2 : least_perimeter = 2 * (1 + 6)) :
  abs (greatest_perimeter - least_perimeter) = 7 := 
sorry

end perimeters_positive_difference_l149_149122


namespace find_x_l149_149966

theorem find_x (x : ℤ) (h : 2 * x = (26 - x) + 19) : x = 15 :=
by
  sorry

end find_x_l149_149966


namespace projected_increase_in_attendance_l149_149717

variable (A P : ℝ)

theorem projected_increase_in_attendance :
  (0.8 * A = 0.64 * (A + (P / 100) * A)) → P = 25 :=
by
  intro h
  -- Proof omitted
  sorry

end projected_increase_in_attendance_l149_149717


namespace Harry_bought_five_packets_of_chili_pepper_l149_149142

noncomputable def price_pumpkin : ℚ := 2.50
noncomputable def price_tomato : ℚ := 1.50
noncomputable def price_chili_pepper : ℚ := 0.90
noncomputable def packets_pumpkin : ℕ := 3
noncomputable def packets_tomato : ℕ := 4
noncomputable def total_spent : ℚ := 18
noncomputable def packets_chili_pepper (p : ℕ) := price_pumpkin * packets_pumpkin + price_tomato * packets_tomato + price_chili_pepper * p = total_spent

theorem Harry_bought_five_packets_of_chili_pepper :
  ∃ p : ℕ, packets_chili_pepper p ∧ p = 5 :=
by 
  sorry

end Harry_bought_five_packets_of_chili_pepper_l149_149142


namespace problem_statement_l149_149049

open Real

variable (a b c : ℝ)

theorem problem_statement
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h_cond : a + b + c + a * b * c = 4) :
  (1 + a / b + c * a) * (1 + b / c + a * b) * (1 + c / a + b * c) ≥ 27 := 
by
  sorry

end problem_statement_l149_149049


namespace bus_departure_interval_l149_149260

theorem bus_departure_interval
  (v : ℝ) -- speed of B (per minute)
  (t_A : ℝ := 10) -- A is overtaken every 10 minutes
  (t_B : ℝ := 6) -- B is overtaken every 6 minutes
  (v_A : ℝ := 3 * v) -- speed of A
  (d_A : ℝ := v_A * t_A) -- distance covered by A in 10 minutes
  (d_B : ℝ := v * t_B) -- distance covered by B in 6 minutes
  (v_bus_minus_vA : ℝ := d_A / t_A) -- bus speed relative to A
  (v_bus_minus_vB : ℝ := d_B / t_B) -- bus speed relative to B) :
  (t : ℝ) -- time interval between bus departures
  : t = 5 := sorry

end bus_departure_interval_l149_149260


namespace student_b_speed_l149_149846

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l149_149846


namespace remainder_17_pow_63_mod_7_l149_149551

theorem remainder_17_pow_63_mod_7 : (17^63) % 7 = 6 := 
by
  sorry

end remainder_17_pow_63_mod_7_l149_149551


namespace least_n_l149_149609

theorem least_n (n : ℕ) (h : n > 0) : 
  (1 / (n : ℝ) - 1 / (n + 1) < 1 / 15) ↔ n ≥ 4 := 
begin
  sorry
end

end least_n_l149_149609


namespace least_n_satisfies_inequality_l149_149636

theorem least_n_satisfies_inequality : ∃ n : ℕ, (1 : ℚ) / n - (1 : ℚ) / (n + 1) < 1 / 15 ∧ ∀ m : ℕ, (1 : ℚ) / m - (1 : ℚ) / (m + 1) < 1 / 15 -> ¬ (m < n) := 
sorry

end least_n_satisfies_inequality_l149_149636


namespace student_B_speed_l149_149781

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l149_149781


namespace area_of_quadrilateral_l149_149032

theorem area_of_quadrilateral (A B C D H : Type) (AB BC : Real)
    (angle_ABC angle_ADC : Real) (BH h : Real)
    (H1 : AB = BC) (H2 : angle_ABC = 90 ∧ angle_ADC = 90)
    (H3 : BH = h) :
    (∃ area : Real, area = h^2) :=
by
  sorry

end area_of_quadrilateral_l149_149032


namespace white_ball_probability_l149_149346

theorem white_ball_probability (m : ℕ) 
  (initial_black : ℕ := 6) 
  (initial_white : ℕ := 10) 
  (added_white := 14) 
  (probability := 0.8) :
  (10 + added_white) / (16 + added_white) = probability :=
by
  -- no proof required
  sorry

end white_ball_probability_l149_149346


namespace remaining_shape_perimeter_l149_149989

def rectangle_perimeter (L W : ℕ) : ℕ := 2 * (L + W)

theorem remaining_shape_perimeter (L W S : ℕ) (hL : L = 12) (hW : W = 5) (hS : S = 2) :
  rectangle_perimeter L W = 34 :=
by
  rw [hL, hW]
  rfl

end remaining_shape_perimeter_l149_149989


namespace student_B_speed_l149_149792

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l149_149792


namespace percentage_saved_l149_149148

theorem percentage_saved (saved spent : ℝ) (h_saved : saved = 3) (h_spent : spent = 27) : 
  (saved / (saved + spent)) * 100 = 10 := by
  sorry

end percentage_saved_l149_149148


namespace escalator_rate_l149_149285

theorem escalator_rate
  (length_escalator : ℕ) 
  (person_speed : ℕ) 
  (time_taken : ℕ) 
  (total_length : length_escalator = 112) 
  (person_speed_rate : person_speed = 4)
  (time_taken_rate : time_taken = 8) :
  ∃ v : ℕ, (person_speed + v) * time_taken = length_escalator ∧ v = 10 :=
by
  sorry

end escalator_rate_l149_149285


namespace speed_of_student_B_l149_149828

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l149_149828


namespace new_ratio_alcohol_water_l149_149716

theorem new_ratio_alcohol_water (alcohol water: ℕ) (initial_ratio: alcohol * 3 = water * 4) 
  (extra_water: ℕ) (extra_water_added: extra_water = 4) (alcohol_given: alcohol = 20):
  20 * 19 = alcohol * (water + extra_water) :=
by
  sorry

end new_ratio_alcohol_water_l149_149716


namespace max_height_of_table_l149_149678

theorem max_height_of_table (BC CA AB : ℕ) (h : ℝ) :
  BC = 24 →
  CA = 28 →
  AB = 32 →
  h ≤ (49 * Real.sqrt 60) / 19 :=
by
  intros
  sorry

end max_height_of_table_l149_149678


namespace power_mean_inequality_l149_149050

theorem power_mean_inequality
  (n : ℕ) (hn : 0 < n) (x1 x2 : ℝ) :
  (x1^n + x2^n)^(n+1) / (x1^(n-1) + x2^(n-1))^n ≤ (x1^(n+1) + x2^(n+1))^n / (x1^n + x2^n)^(n-1) :=
by
  sorry

end power_mean_inequality_l149_149050


namespace find_speed_of_B_l149_149802

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l149_149802


namespace speed_of_student_B_l149_149836

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l149_149836


namespace forgotten_angles_correct_l149_149445

theorem forgotten_angles_correct (n : ℕ) (h1 : (n - 2) * 180 = 2520) (h2 : 2345 + 175 = 2520) : 
  ∃ a b : ℕ, a + b = 175 :=
by
  sorry

end forgotten_angles_correct_l149_149445


namespace student_b_speed_l149_149853

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l149_149853


namespace george_correct_possible_change_sum_l149_149311

noncomputable def george_possible_change_sum : ℕ :=
if h : ∃ (change : ℕ), change < 100 ∧
  ((change % 25 == 7) ∨ (change % 25 == 32) ∨ (change % 25 == 57) ∨ (change % 25 == 82)) ∧
  ((change % 10 == 2) ∨ (change % 10 == 12) ∨ (change % 10 == 22) ∨
   (change % 10 == 32) ∨ (change % 10 == 42) ∨ (change % 10 == 52) ∨
   (change % 10 == 62) ∨ (change % 10 == 72) ∨ (change % 10 == 82) ∨ (change % 10 == 92)) ∧
  ((change % 5 == 9) ∨ (change % 5 == 14) ∨ (change % 5 == 19) ∨
   (change % 5 == 24) ∨ (change % 5 == 29) ∨ (change % 5 == 34) ∨
   (change % 5 == 39) ∨ (change % 5 == 44) ∨ (change % 5 == 49) ∨
   (change % 5 == 54) ∨ (change % 5 == 59) ∨ (change % 5 == 64) ∨
   (change % 5 == 69) ∨ (change % 5 == 74) ∨ (change % 5 == 79) ∨
   (change % 5 == 84) ∨ (change % 5 == 89) ∨ (change % 5 == 94) ∨ (change % 5 == 99)) then
  114
else 0

theorem george_correct_possible_change_sum :
  george_possible_change_sum = 114 :=
by
  sorry

end george_correct_possible_change_sum_l149_149311


namespace min_marked_cells_in_7x7_grid_l149_149734

noncomputable def min_marked_cells : Nat :=
  12

theorem min_marked_cells_in_7x7_grid :
  ∀ (grid : Matrix Nat Nat Nat), (∀ (r c : Nat), r < 7 → c < 7 → (∃ i : Fin 4, grid[[r, i % 4 + c]] = 1) ∨ (∃ j : Fin 4, grid[[j % 4 + r, c]] = 1)) → 
  (∃ m, m = min_marked_cells) :=
sorry

end min_marked_cells_in_7x7_grid_l149_149734


namespace perimeters_positive_difference_l149_149120

theorem perimeters_positive_difference (orig_length orig_width : ℝ) (num_pieces : ℕ)
  (congruent_division : ∃ (length width : ℝ), length * width = (orig_length * orig_width) / num_pieces)
  (greatest_perimeter least_perimeter : ℝ)
  (h1 : greatest_perimeter = 2 * (1.5 + 9))
  (h2 : least_perimeter = 2 * (1 + 6)) :
  abs (greatest_perimeter - least_perimeter) = 7 := 
sorry

end perimeters_positive_difference_l149_149120


namespace no_such_polyhedron_l149_149873

theorem no_such_polyhedron (n : ℕ) (S : Fin n → ℝ) (H : ∀ i j : Fin n, i ≠ j → S i ≥ 2 * S j) : False :=
by
  sorry

end no_such_polyhedron_l149_149873


namespace min_value_problem1_min_value_problem2_l149_149272

-- Problem 1: Prove that the minimum value of the function y = x + 4/(x + 1) + 6 is 9 given x > -1
theorem min_value_problem1 (x : ℝ) (h : x > -1) : (x + 4 / (x + 1) + 6) ≥ 9 := 
sorry

-- Problem 2: Prove that the minimum value of the function y = (x^2 + 8) / (x - 1) is 8 given x > 1
theorem min_value_problem2 (x : ℝ) (h : x > 1) : ((x^2 + 8) / (x - 1)) ≥ 8 :=
sorry

end min_value_problem1_min_value_problem2_l149_149272


namespace largest_k_inequality_l149_149461

noncomputable def k : ℚ := 39 / 2

theorem largest_k_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a + b + c)^3 ≥ (5 / 2) * (a^3 + b^3 + c^3) + k * a * b * c := 
sorry

end largest_k_inequality_l149_149461


namespace seventeen_power_sixty_three_mod_seven_l149_149552

theorem seventeen_power_sixty_three_mod_seven : (17^63) % 7 = 6 := by
  -- Here you would write the actual proof demonstrating the equivalence:
  -- 1. 17 ≡ 3 (mod 7)
  -- 2. Calculate 3^63 (mod 7)
  sorry

end seventeen_power_sixty_three_mod_seven_l149_149552


namespace four_digit_perfect_square_exists_l149_149456

theorem four_digit_perfect_square_exists (x y : ℕ) (h1 : 10 ≤ x ∧ x < 100) (h2 : 10 ≤ y ∧ y < 100) (h3 : 101 * x + 100 = y^2) : 
  ∃ n, n = 8281 ∧ n = y^2 ∧ (((n / 100) : ℕ) = ((n % 100) : ℕ) + 1) :=
by 
  sorry

end four_digit_perfect_square_exists_l149_149456


namespace student_B_speed_l149_149778

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l149_149778


namespace geometric_sequence_first_term_l149_149450

theorem geometric_sequence_first_term (a b c : ℕ) 
    (h1 : 16 = a * (2^3)) 
    (h2 : 32 = a * (2^4)) : 
    a = 2 := 
sorry

end geometric_sequence_first_term_l149_149450


namespace find_speed_B_l149_149856

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l149_149856


namespace student_b_speed_l149_149845

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l149_149845


namespace area_of_similar_rectangle_l149_149162

theorem area_of_similar_rectangle:
  ∀ (R1 : ℝ → ℝ → Prop) (R2 : ℝ → ℝ → Prop),
  (∀ a b, R1 a b → a = 3 ∧ a * b = 18) →
  (∀ a b c d, R1 a b → R2 c d → c / d = a / b) →
  (∀ a b, R2 a b → a^2 + b^2 = 400) →
  ∃ areaR2, (∀ a b, R2 a b → a * b = areaR2) ∧ areaR2 = 160 :=
by
  intros R1 R2 hR1 h_similar h_diagonal
  use 160
  sorry

end area_of_similar_rectangle_l149_149162


namespace eggs_broken_l149_149226

theorem eggs_broken (brown_eggs white_eggs total_pre total_post broken_eggs : ℕ) 
  (h1 : brown_eggs = 10)
  (h2 : white_eggs = 3 * brown_eggs)
  (h3 : total_pre = brown_eggs + white_eggs)
  (h4 : total_post = 20)
  (h5 : broken_eggs = total_pre - total_post) : broken_eggs = 20 :=
by
  sorry

end eggs_broken_l149_149226


namespace average_increase_l149_149217

def scores : List ℕ := [92, 85, 90, 95]

def initial_average (s : List ℕ) : ℚ := (s.take 3).sum / 3

def new_average (s : List ℕ) : ℚ := s.sum / s.length

theorem average_increase :
  initial_average scores + 1.5 = new_average scores := 
by
  sorry

end average_increase_l149_149217


namespace value_of_expression_l149_149503

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 2 * x + 5 = 9) : 3 * x^2 + 3 * x - 7 = -1 :=
by
  -- The proof would go here
  sorry

end value_of_expression_l149_149503


namespace congruent_semicircles_span_diameter_l149_149028

theorem congruent_semicircles_span_diameter (N : ℕ) (r : ℝ) 
  (h1 : 2 * N * r = 2 * (N * r)) 
  (h2 : (N * (π * r^2 / 2)) / ((N^2 * (π * r^2 / 2)) - (N * (π * r^2 / 2))) = 1/4) 
  : N = 5 :=
by
  sorry

end congruent_semicircles_span_diameter_l149_149028


namespace min_abs_sum_of_diffs_l149_149222

theorem min_abs_sum_of_diffs (x : ℝ) (α β : ℝ)
  (h₁ : α * α - 6 * α + 5 = 0)
  (h₂ : β * β - 6 * β + 5 = 0)
  (h_ne : α ≠ β) :
  ∃ m, ∀ x, m = min (|x - α| + |x - β|) :=
by
  use (4)
  sorry

end min_abs_sum_of_diffs_l149_149222


namespace range_of_x_l149_149187

-- Let p and q be propositions regarding the range of x:
def p (x : ℝ) : Prop := x^2 - 5 * x + 6 ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4

-- Main theorem statement
theorem range_of_x 
  (h1 : ∀ x : ℝ, p x ∨ q x)
  (h2 : ∀ x : ℝ, ¬ q x) :
  ∀ x : ℝ, (x ≤ 0 ∨ x ≥ 4) := by
  sorry

end range_of_x_l149_149187


namespace exam_student_count_l149_149740

theorem exam_student_count (N T T_5 T_remaining : ℕ)
  (h1 : T = 70 * N)
  (h2 : T_5 = 50 * 5)
  (h3 : T_remaining = 90 * (N - 5))
  (h4 : T = T_5 + T_remaining) :
  N = 10 :=
by
  sorry

end exam_student_count_l149_149740


namespace shift_parabola_3_right_4_up_l149_149910

theorem shift_parabola_3_right_4_up (x : ℝ) : 
  let y := x^2 in
  (shifted_y : ℝ) = ((x - 3)^2 + 4) :=
begin
  sorry
end

end shift_parabola_3_right_4_up_l149_149910


namespace hawks_first_half_score_l149_149504

variable (H1 H2 E : ℕ)

theorem hawks_first_half_score (H1 H2 E : ℕ) 
  (h1 : H1 + H2 + E = 120)
  (h2 : E = H1 + H2 + 16)
  (h3 : H2 = H1 + 8) :
  H1 = 22 :=
by
  sorry

end hawks_first_half_score_l149_149504


namespace remainder_17_pow_63_mod_7_l149_149549

theorem remainder_17_pow_63_mod_7 : (17^63) % 7 = 6 := 
by
  sorry

end remainder_17_pow_63_mod_7_l149_149549


namespace man_older_than_son_l149_149572

variables (S M : ℕ)

theorem man_older_than_son (h1 : S = 32) (h2 : M + 2 = 2 * (S + 2)) : M - S = 34 :=
by
  sorry

end man_older_than_son_l149_149572


namespace new_supervisor_salary_l149_149741

theorem new_supervisor_salary
  (W S1 S2 : ℝ)
  (avg_old : (W + S1) / 9 = 430)
  (S1_val : S1 = 870)
  (avg_new : (W + S2) / 9 = 410) :
  S2 = 690 :=
by
  sorry

end new_supervisor_salary_l149_149741


namespace max_value_proof_l149_149705

theorem max_value_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 - 3 * x * y + 5 * y^2 = 9) : 
  (∃ a b c d : ℕ, a = 315 ∧ b = 297 ∧ c = 5 ∧ d = 55 ∧ (x^2 + 3 * x * y + 5 * y^2 = (315 + 297 * Real.sqrt 5) / 55) ∧ (a + b + c + d = 672)) :=
by
  sorry

end max_value_proof_l149_149705


namespace three_times_x_greater_than_four_l149_149300

theorem three_times_x_greater_than_four (x : ℝ) : 3 * x > 4 := by
  sorry

end three_times_x_greater_than_four_l149_149300


namespace find_speed_of_B_l149_149763

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l149_149763


namespace arccos_sin_1_5_eq_pi_over_2_minus_1_5_l149_149870

-- Define the problem statement in Lean 4.
theorem arccos_sin_1_5_eq_pi_over_2_minus_1_5 : 
  Real.arccos (Real.sin 1.5) = (Real.pi / 2) - 1.5 :=
by
  sorry

end arccos_sin_1_5_eq_pi_over_2_minus_1_5_l149_149870


namespace preceding_integer_binary_l149_149905

theorem preceding_integer_binary (M : ℕ) (h : M = 0b110101) : 
  (M - 1) = 0b110100 :=
by
  sorry

end preceding_integer_binary_l149_149905


namespace plywood_perimeter_difference_l149_149100

theorem plywood_perimeter_difference :
  ∀ (length width : ℕ) (n : ℕ), 
    length = 6 ∧ width = 9 ∧ n = 6 → 
    ∃ (max_perimeter min_perimeter : ℕ), 
      (max_perimeter - min_perimeter = 10) ∧
      max_perimeter = 20 ∧ 
      min_perimeter = 10 :=
by
  sorry

end plywood_perimeter_difference_l149_149100


namespace find_f_of_7_over_3_l149_149220

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the odd function f

-- Hypothesis: f is an odd function
axiom odd_function (x : ℝ) : f (-x) = -f x

-- Hypothesis: f(1 + x) = f(-x) for all x in ℝ
axiom functional_equation (x : ℝ) : f (1 + x) = f (-x)

-- Hypothesis: f(-1/3) = 1/3
axiom initial_condition : f (-1 / 3) = 1 / 3

-- The statement we need to prove
theorem find_f_of_7_over_3 : f (7 / 3) = - (1 / 3) :=
by
  sorry -- Proof to be provided

end find_f_of_7_over_3_l149_149220


namespace nancy_other_albums_count_l149_149698

-- Definitions based on the given conditions
def total_pictures : ℕ := 51
def pics_in_first_album : ℕ := 11
def pics_per_other_album : ℕ := 5

-- Theorem to prove the question's answer
theorem nancy_other_albums_count : 
  (total_pictures - pics_in_first_album) / pics_per_other_album = 8 := by
  sorry

end nancy_other_albums_count_l149_149698


namespace speed_of_student_B_l149_149839

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l149_149839


namespace gcd_of_11121_and_12012_l149_149308

def gcd_problem : Prop :=
  gcd 11121 12012 = 1

theorem gcd_of_11121_and_12012 : gcd_problem :=
by
  -- Proof omitted
  sorry

end gcd_of_11121_and_12012_l149_149308


namespace rationalize_denominator_l149_149376

theorem rationalize_denominator : 
  (∃ (x y : ℝ), x = real.sqrt 12 + real.sqrt 5 ∧ y = real.sqrt 3 + real.sqrt 5 
    ∧ (x / y) = (real.sqrt 15 - 1) / 2) :=
begin
  use [real.sqrt 12 + real.sqrt 5, real.sqrt 3 + real.sqrt 5],
  split,
  { refl },
  split,
  { refl },
  { sorry }
end

end rationalize_denominator_l149_149376


namespace least_n_inequality_l149_149614

theorem least_n_inequality (n : ℕ) (hn : 0 < n) (ineq: 1 / n - 1 / (n + 1) < 1 / 15) :
  4 ≤ n :=
begin
  simp [lt_div_iff] at ineq,
  rw [sub_eq_add_neg, add_div_eq_mul_add_neg_div, ←lt_div_iff], 
  exact ineq
end

end least_n_inequality_l149_149614


namespace number_142857_has_property_l149_149263

noncomputable def has_desired_property (n : ℕ) : Prop :=
∀ m ∈ [1, 2, 3, 4, 5, 6], ∀ d ∈ (Nat.digits 10 (n * m)), d ∈ (Nat.digits 10 n)

theorem number_142857_has_property : has_desired_property 142857 :=
sorry

end number_142857_has_property_l149_149263


namespace find_candy_bars_per_week_l149_149924

-- Define the conditions
variables (x : ℕ)

-- Condition: Kim's dad buys Kim x candy bars each week
def candies_bought := 16 * x

-- Condition: Kim eats one candy bar every 4 weeks
def candies_eaten := 16 / 4

-- Condition: After 16 weeks, Kim has saved 28 candy bars
def saved_candies := 28

-- The theorem we want to prove
theorem find_candy_bars_per_week : (16 * x - (16 / 4) = 28) → x = 2 := by
  -- We will skip the actual proof for now.
  sorry

end find_candy_bars_per_week_l149_149924


namespace ratio_of_scores_l149_149259

theorem ratio_of_scores 
  (u v : ℝ) 
  (h1 : u > v) 
  (h2 : u - v = (u + v) / 2) 
  : v / u = 1 / 3 :=
sorry

end ratio_of_scores_l149_149259


namespace max_gcd_of_consecutive_terms_l149_149733

def sequence_b (n : ℕ) : ℕ := n.factorial + 2 * n

theorem max_gcd_of_consecutive_terms (n : ℕ) (h : n ≥ 1) : gcd (sequence_b n) (sequence_b (n + 1)) = 2 := by
  sorry

end max_gcd_of_consecutive_terms_l149_149733


namespace range_of_k_for_ellipse_l149_149204

def represents_ellipse (x y k : ℝ) : Prop :=
  (k^2 - 3 > 0) ∧ 
  (k - 1 > 0) ∧ 
  (k - 1 ≠ k^2 - 3)

theorem range_of_k_for_ellipse (k : ℝ) : 
  represents_ellipse x y k → k ∈ Set.Ioo (-Real.sqrt 3) (-1) ∪ Set.Ioo (-1) 1 :=
by
  sorry

end range_of_k_for_ellipse_l149_149204


namespace dayan_sequence_20th_term_l149_149527

theorem dayan_sequence_20th_term (a : ℕ → ℕ) (h1 : a 0 = 0)
    (h2 : a 1 = 2) (h3 : a 2 = 4) (h4 : a 3 = 8) (h5 : a 4 = 12)
    (h6 : a 5 = 18) (h7 : a 6 = 24) (h8 : a 7 = 32) (h9 : a 8 = 40) (h10 : a 9 = 50)
    (h_even : ∀ n : ℕ, a (2 * n) = 2 * n^2) :
  a 20 = 200 :=
  sorry

end dayan_sequence_20th_term_l149_149527


namespace opposite_of_fraction_l149_149951

theorem opposite_of_fraction : - (11 / 2022 : ℚ) = -(11 / 2022) := 
by
  sorry

end opposite_of_fraction_l149_149951


namespace cost_per_page_first_time_l149_149069

-- Definitions based on conditions
variables (num_pages : ℕ) (rev_once_pages : ℕ) (rev_twice_pages : ℕ)
variables (rev_cost : ℕ) (total_cost : ℕ)
variables (first_time_cost : ℕ)

-- Conditions
axiom h1 : num_pages = 100
axiom h2 : rev_once_pages = 35
axiom h3 : rev_twice_pages = 15
axiom h4 : rev_cost = 4
axiom h5 : total_cost = 860

-- Proof statement: Prove that the cost per page for the first time a page is typed is $6
theorem cost_per_page_first_time : first_time_cost = 6 :=
sorry

end cost_per_page_first_time_l149_149069


namespace pow_mod_seventeen_l149_149546

theorem pow_mod_seventeen sixty_three :
  17^63 % 7 = 6 := by
  have h : 17 % 7 = 3 := by norm_num
  have h1 : 17^63 % 7 = 3^63 % 7 := by rw [pow_mod_eq_of_mod_eq h] 
  norm_num at h1
  rw [h1]
  sorry

end pow_mod_seventeen_l149_149546


namespace number_of_tiles_l149_149990

noncomputable def tile_count (room_length : ℝ) (room_width : ℝ) (tile_length : ℝ) (tile_width : ℝ) :=
  let room_area := room_length * room_width
  let tile_area := tile_length * tile_width
  room_area / tile_area

theorem number_of_tiles :
  tile_count 10 15 (1 / 4) (5 / 12) = 1440 := by
  sorry

end number_of_tiles_l149_149990


namespace order_of_numbers_l149_149872

variable (a b c : ℝ)
variable (h₁ : a = (1 / 2) ^ (1 / 3))
variable (h₂ : b = (1 / 2) ^ (2 / 3))
variable (h₃ : c = (1 / 5) ^ (2 / 3))

theorem order_of_numbers (a b c : ℝ) (h₁ : a = (1 / 2) ^ (1 / 3)) (h₂ : b = (1 / 2) ^ (2 / 3)) (h₃ : c = (1 / 5) ^ (2 / 3)) :
  c < b ∧ b < a := 
by
  sorry

end order_of_numbers_l149_149872


namespace no_distinct_positive_integers_2007_l149_149235

theorem no_distinct_positive_integers_2007 (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) : 
  ¬ (x^2007 + y! = y^2007 + x!) :=
by
  sorry

end no_distinct_positive_integers_2007_l149_149235


namespace actual_area_l149_149348

open Real

theorem actual_area
  (scale : ℝ)
  (mapped_area_cm2 : ℝ)
  (actual_area_cm2 : ℝ)
  (actual_area_m2 : ℝ)
  (h_scale : scale = 1 / 50000)
  (h_mapped_area : mapped_area_cm2 = 100)
  (h_proportion : mapped_area_cm2 / actual_area_cm2 = scale ^ 2)
  : actual_area_m2 = 2.5 * 10^7 :=
by
  sorry

end actual_area_l149_149348


namespace car_distance_kilometers_l149_149577

theorem car_distance_kilometers (d_amar : ℝ) (d_car : ℝ) (ratio : ℝ) (total_d_amar : ℝ) :
  d_amar = 24 ->
  d_car = 60 ->
  ratio = 2 / 5 ->
  total_d_amar = 880 ->
  (d_car / d_amar) = 5 / 2 ->
  (total_d_amar * 5 / 2) / 1000 = 2.2 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end car_distance_kilometers_l149_149577


namespace modulus_squared_of_complex_l149_149193

theorem modulus_squared_of_complex :
  (complex.abs (complex.div (3 - 2 * complex.I) (1 - complex.I)))^2 = 13 / 2 :=
by
  sorry

end modulus_squared_of_complex_l149_149193


namespace quadratic_roots_value_l149_149397

theorem quadratic_roots_value (d : ℝ) 
  (h : ∀ x : ℝ, x^2 + 7 * x + d = 0 → x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) : 
  d = 9.8 :=
by 
  sorry

end quadratic_roots_value_l149_149397


namespace rectangle_diagonal_length_l149_149711

theorem rectangle_diagonal_length (L W : ℝ) (h1 : L * W = 20) (h2 : L + W = 9) :
  (L^2 + W^2) = 41 :=
by
  sorry

end rectangle_diagonal_length_l149_149711


namespace real_solution_unique_l149_149171

theorem real_solution_unique (x : ℝ) (h : x^4 + (2 - x)^4 + 2 * x = 34) : x = 0 :=
sorry

end real_solution_unique_l149_149171


namespace determine_identity_l149_149290

-- Define the types for human and vampire
inductive Being
| human
| vampire

-- Define the responses for sanity questions
def claims_sanity (b : Being) : Prop :=
  match b with
  | Being.human   => true
  | Being.vampire => false

-- Proof statement: Given that a human always claims sanity and a vampire always claims insanity,
-- asking "Are you sane?" will determine their identity. 
theorem determine_identity (b : Being) (h : b = Being.human ↔ claims_sanity b = true) : 
  ((claims_sanity b = true) → b = Being.human) ∧ ((claims_sanity b = false) → b = Being.vampire) :=
sorry

end determine_identity_l149_149290


namespace captain_age_l149_149208

theorem captain_age
  (C W : ℕ)
  (avg_team_age : ℤ)
  (avg_remaining_players_age : ℤ)
  (total_team_age : ℤ)
  (total_remaining_players_age : ℤ)
  (remaining_players_count : ℕ)
  (total_team_count : ℕ)
  (total_team_age_eq : total_team_age = total_team_count * avg_team_age)
  (remaining_players_age_eq : total_remaining_players_age = remaining_players_count * avg_remaining_players_age)
  (total_team_eq : total_team_count = 11)
  (remaining_players_eq : remaining_players_count = 9)
  (avg_team_age_eq : avg_team_age = 23)
  (avg_remaining_players_age_eq : avg_remaining_players_age = avg_team_age - 1)
  (age_diff : W = C + 5)
  (players_age_sum : total_team_age = total_remaining_players_age + C + W) :
  C = 25 :=
by
  sorry

end captain_age_l149_149208


namespace unique_function_l149_149592

theorem unique_function (f : ℝ → ℝ) (hf : ∀ x : ℝ, 0 ≤ x → 0 ≤ f x)
  (cond1 : ∀ x : ℝ, 0 ≤ x → 4 * f x ≥ 3 * x)
  (cond2 : ∀ x : ℝ, 0 ≤ x → f (4 * f x - 3 * x) = x) :
  ∀ x : ℝ, 0 ≤ x → f x = x :=
by
  sorry

end unique_function_l149_149592


namespace least_n_l149_149600

theorem least_n (n : ℕ) (h : (1 : ℝ) / n - (1 / (n + 1)) < (1 / 15)) : n = 4 :=
sorry

end least_n_l149_149600


namespace union_A_B_l149_149894

-- Define set A
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 0}

-- Define set B
def B : Set ℝ := {x | x^2 > 1}

-- Prove the union of A and B is the expected result
theorem union_A_B : A ∪ B = {x | x ≤ 0 ∨ x > 1} :=
by
  sorry

end union_A_B_l149_149894


namespace find_k_square_binomial_l149_149088

theorem find_k_square_binomial (k : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 - 16 * x + k = (x + b)^2) ↔ k = 64 :=
by
  sorry

end find_k_square_binomial_l149_149088


namespace sum_of_ages_of_cousins_l149_149922

noncomputable def is_valid_age_group (a b c d : ℕ) : Prop :=
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) ∧
  (1 ≤ a) ∧ (a ≤ 9) ∧ (1 ≤ b) ∧ (b ≤ 9) ∧ (1 ≤ c) ∧ (c ≤ 9) ∧ (1 ≤ d) ∧ (d ≤ 9)

theorem sum_of_ages_of_cousins :
  ∃ (a b c d : ℕ), is_valid_age_group a b c d ∧ (a * b = 40) ∧ (c * d = 36) ∧ (a + b + c + d = 26) := 
sorry

end sum_of_ages_of_cousins_l149_149922


namespace student_b_speed_l149_149824

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l149_149824


namespace perimeter_difference_l149_149127

-- Definitions as per conditions
def plywood_width : ℕ := 6
def plywood_height : ℕ := 9
def rectangles_count : ℕ := 6

-- The perimeter difference to be proved
theorem perimeter_difference : 
  ∃ (max_perimeter min_perimeter : ℕ), 
  max_perimeter = 22 ∧ min_perimeter = 12 ∧ (max_perimeter - min_perimeter = 10) :=
by
  sorry

end perimeter_difference_l149_149127


namespace remainder_17_pow_63_mod_7_l149_149543

theorem remainder_17_pow_63_mod_7 : (17 ^ 63) % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l149_149543


namespace find_speed_B_l149_149861

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l149_149861


namespace student_b_speed_l149_149852

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l149_149852


namespace find_all_functions_satisfying_func_eq_l149_149877

-- Given a function f : ℝ → ℝ that satisfies a certain functional equation:
def satisfies_func_eq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (floor x * y) = f x * floor (f y)

-- The main proposition to prove:
theorem find_all_functions_satisfying_func_eq :
  ∀ f : ℝ → ℝ, satisfies_func_eq f → (f = (λ x, 0) ∨ (∃ c : ℝ, 1 ≤ c ∧ c < 2 ∧ f = (λ _, c))) :=
by
  -- Proof goes here
  sorry

end find_all_functions_satisfying_func_eq_l149_149877


namespace goat_age_l149_149713

theorem goat_age : 26 + 42 = 68 := 
by 
  -- Since we only need the statement,
  -- we add sorry to skip the proof.
  sorry

end goat_age_l149_149713


namespace polar_to_rectangular_l149_149163

theorem polar_to_rectangular (r θ : ℝ) (h₁ : r = 5) (h₂ : θ = 5 * Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (5 / 2, -5 * Real.sqrt 3 / 2) :=
by sorry

end polar_to_rectangular_l149_149163


namespace fraction_students_above_eight_l149_149914

theorem fraction_students_above_eight (total_students S₈ : ℕ) (below_eight_percent : ℝ)
    (num_below_eight : total_students * below_eight_percent = 10) 
    (total_equals : total_students = 50) 
    (students_eight : S₈ = 24) :
    (total_students - (total_students * below_eight_percent + S₈)) / S₈ = 2 / 3 := 
by 
  -- Solution steps can go here 
  sorry

end fraction_students_above_eight_l149_149914


namespace find_x_l149_149331

theorem find_x (x : ℚ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 :=
by
  intro h,
  sorry

end find_x_l149_149331


namespace evaluate_expression_l149_149454

theorem evaluate_expression (x y : ℝ) (h1 : x = 3) (h2 : y = 0) : y * (y - 3 * x) = 0 :=
by sorry

end evaluate_expression_l149_149454


namespace find_p_l149_149316

theorem find_p (n : ℝ) (p : ℝ) (h1 : p = 4 * n * (1 / (2 ^ 2009)) ^ Real.log 1) (h2 : n = 9 / 4) : p = 9 :=
by
  sorry

end find_p_l149_149316


namespace range_of_a_l149_149649

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := if x ≥ 1 then x * Real.log x - a * x^2 else a^x

theorem range_of_a (a : ℝ) (f_decreasing : ∀ x y : ℝ, x ≤ y → f x a ≥ f y a) : 
  1/2 ≤ a ∧ a < 1 :=
by
  sorry

end range_of_a_l149_149649


namespace exists_convex_quadrilateral_with_ratio_of_sides_eq_ratio_of_angles_l149_149237

theorem exists_convex_quadrilateral_with_ratio_of_sides_eq_ratio_of_angles
    (a b c d α β γ δ: ℝ) (h_conv: a < b + c + d ∧ b < a + c + d ∧ c < a + b + d ∧ d < a + b + c)
    (h_angles: α < β + γ + δ ∧ β < α + γ + δ ∧ γ < α + β + δ ∧ δ < α + β + γ) :
    ∃ (a' b' c' d' α' β' γ' δ' : ℝ),
      (a' / b' = α / β) ∧ (b' / c' = β / γ) ∧ (c' / d' = γ / δ) ∧ (d' / a' = δ / α) ∧
      (a' < b' + c' + d') ∧ (b' < a' + c' + d') ∧ (c' < a' + b' + d') ∧ (d' < a' + b' + c') ∧
      (α' < β' + γ' + δ') ∧ (β' < α' + γ' + δ') ∧ (γ' < α' + β' + δ') ∧ (δ' < α' + β' + γ') :=
  sorry

end exists_convex_quadrilateral_with_ratio_of_sides_eq_ratio_of_angles_l149_149237


namespace intersection_proof_l149_149656

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def N : Set ℕ := { x | Real.sqrt (2^x - 1) < 5 }
def expected_intersection : Set ℕ := {1, 2, 3, 4}

theorem intersection_proof : M ∩ N = expected_intersection := by
  sorry

end intersection_proof_l149_149656


namespace num_routes_A_to_B_l149_149581

-- Define the cities as an inductive type
inductive City
| A | B | C | D | E | F
deriving DecidableEq, Inhabited

open City

-- Define the roads as a set of pairs of cities
def roads : set (City × City) :=
  { (A, B), (A, D), (A, E),
    (B, A), (B, C), (B, D),
    (C, B), (C, D),
    (D, A), (D, B), (D, C), (D, E),
    (E, A), (E, D), (E, F),
    (F, E) }

-- Define what it means to be a valid route from A to B that uses each road exactly once
def valid_route (p: list (City × City)) : Prop :=
  (p.head = some (A, _)) ∧ (p.last = some (_, B)) ∧
  (p.nodup) ∧ (∀ e ∈ p, e ∈ roads) ∧ (roads ⊆ p.to_finset)

-- The theorem stating the number of valid routes
theorem num_routes_A_to_B : {p : list (City × City) // valid_route p}.card = 12 :=
by 
  sorry  -- Proof omitted

end num_routes_A_to_B_l149_149581


namespace least_n_inequality_l149_149612

theorem least_n_inequality (n : ℕ) (hn : 0 < n) (ineq: 1 / n - 1 / (n + 1) < 1 / 15) :
  4 ≤ n :=
begin
  simp [lt_div_iff] at ineq,
  rw [sub_eq_add_neg, add_div_eq_mul_add_neg_div, ←lt_div_iff], 
  exact ineq
end

end least_n_inequality_l149_149612


namespace find_pairs_l149_149221

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ q r : ℕ, a^2 + b^2 = (a + b) * q + r ∧ q^2 + r = 1977) →
  (a, b) = (50, 37) ∨ (a, b) = (37, 50) ∨ (a, b) = (50, 7) ∨ (a, b) = (7, 50) :=
by
  sorry

end find_pairs_l149_149221


namespace least_n_l149_149607

theorem least_n (n : ℕ) (h : n > 0) : 
  (1 / (n : ℝ) - 1 / (n + 1) < 1 / 15) ↔ n ≥ 4 := 
begin
  sorry
end

end least_n_l149_149607


namespace other_root_of_quadratic_l149_149001

theorem other_root_of_quadratic (m : ℝ) (h : ∀ x : ℝ, x^2 + m*x - 20 = 0 → (x = -4)) 
: ∃ t : ℝ, t = 5 := 
by
  existsi 5
  sorry

end other_root_of_quadratic_l149_149001


namespace speed_of_student_B_l149_149835

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l149_149835


namespace smallest_positive_integer_solution_l149_149298

theorem smallest_positive_integer_solution (x : ℤ) 
  (hx : |5 * x - 8| = 47) : x = 11 :=
by
  sorry

end smallest_positive_integer_solution_l149_149298


namespace modulus_remainder_l149_149167

namespace Proof

def a (n : ℕ) : ℕ := 88134 + n

theorem modulus_remainder :
  (2 * ((a 0)^2 + (a 1)^2 + (a 2)^2 + (a 3)^2 + (a 4)^2 + (a 5)^2)) % 11 = 3 := by
  sorry

end Proof

end modulus_remainder_l149_149167


namespace doubled_radius_and_arc_length_invariant_l149_149912

theorem doubled_radius_and_arc_length_invariant (r l : ℝ) : (l / r) = (2 * l / (2 * r)) :=
by
  sorry

end doubled_radius_and_arc_length_invariant_l149_149912


namespace least_n_inequality_l149_149611

theorem least_n_inequality (n : ℕ) (hn : 0 < n) (ineq: 1 / n - 1 / (n + 1) < 1 / 15) :
  4 ≤ n :=
begin
  simp [lt_div_iff] at ineq,
  rw [sub_eq_add_neg, add_div_eq_mul_add_neg_div, ←lt_div_iff], 
  exact ineq
end

end least_n_inequality_l149_149611


namespace percentage_students_left_in_classroom_l149_149347

def total_students : ℕ := 250
def fraction_painting : ℚ := 3 / 10
def fraction_field : ℚ := 2 / 10
def fraction_science : ℚ := 1 / 5

theorem percentage_students_left_in_classroom :
  let gone_painting := total_students * fraction_painting
  let gone_field := total_students * fraction_field
  let gone_science := total_students * fraction_science
  let students_gone := gone_painting + gone_field + gone_science
  let students_left := total_students - students_gone
  (students_left / total_students) * 100 = 30 :=
by sorry

end percentage_students_left_in_classroom_l149_149347


namespace find_other_parallel_side_l149_149174

variable (a b h : ℝ) (Area : ℝ)

-- Conditions
axiom h_pos : h = 13
axiom a_val : a = 18
axiom area_val : Area = 247
axiom area_formula : Area = (1 / 2) * (a + b) * h

-- Theorem (to be proved by someone else)
theorem find_other_parallel_side (a b h : ℝ) 
  (h_pos : h = 13) 
  (a_val : a = 18) 
  (area_val : Area = 247) 
  (area_formula : Area = (1 / 2) * (a + b) * h) : 
  b = 20 :=
by
  sorry

end find_other_parallel_side_l149_149174


namespace tiles_painted_in_15_minutes_l149_149874

open Nat

theorem tiles_painted_in_15_minutes:
  let don_rate := 3
  let ken_rate := don_rate + 2
  let laura_rate := 2 * ken_rate
  let kim_rate := laura_rate - 3
  don_rate + ken_rate + laura_rate + kim_rate == 25 → 
  15 * (don_rate + ken_rate + laura_rate + kim_rate) = 375 :=
by
  intros
  sorry

end tiles_painted_in_15_minutes_l149_149874


namespace proof_problem_l149_149315

-- Definitions of the function and conditions:
def f : ℝ → ℝ := sorry
axiom odd_f : ∀ x, f (-x) = -f x
axiom periodicity_f : ∀ x, f (x + 2) = -f x
axiom f_def_on_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2^x - 1

-- The theorem statement:
theorem proof_problem :
  f 6 < f (11 / 2) ∧ f (11 / 2) < f (-7) :=
by
  sorry

end proof_problem_l149_149315


namespace student_B_speed_l149_149797

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l149_149797


namespace jake_present_weight_l149_149973

theorem jake_present_weight (J S : ℕ) 
  (h1 : J - 32 = 2 * S) 
  (h2 : J + S = 212) : 
  J = 152 := 
by 
  sorry

end jake_present_weight_l149_149973


namespace plywood_perimeter_difference_l149_149136

noncomputable theory

open classical

theorem plywood_perimeter_difference :
  ∃ (rect1 rect2 : ℕ) (a b : ℕ),
  (rect1 = 6 ∧ rect2 = 9 ∧ rect1 % 6 = 0 ∧ rect2 % 6 = 0) ∧ 
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≤ 20) ∧
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≥ 10) ∧ 
  (20 - 10 = 10) :=
by
  exists (6, 9, 6, 9)
  sorry

end plywood_perimeter_difference_l149_149136


namespace speed_of_student_B_l149_149785

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l149_149785


namespace number_of_girls_l149_149029

theorem number_of_girls
  (total_pupils : ℕ)
  (boys : ℕ)
  (teachers : ℕ)
  (girls : ℕ)
  (h1 : total_pupils = 626)
  (h2 : boys = 318)
  (h3 : teachers = 36)
  (h4 : girls = total_pupils - boys - teachers) :
  girls = 272 :=
by
  rw [h1, h2, h3] at h4
  exact h4

-- Proof is not required, hence 'sorry' can be used for practical purposes
-- exact sorry

end number_of_girls_l149_149029


namespace product_of_two_numbers_l149_149669

theorem product_of_two_numbers :
  ∃ x y : ℝ, x + y = 16 ∧ x^2 + y^2 = 200 ∧ x * y = 28 :=
by
  sorry

end product_of_two_numbers_l149_149669


namespace maximum_modest_number_l149_149501

-- Definitions and Conditions
def is_modest (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
  5 * a = b + c + d ∧
  d % 2 = 0

def G (a b c d : ℕ) : ℕ :=
  (1000 * a + 100 * b + 10 * c + d - (1000 * c + 100 * d + 10 * a + b)) / 99

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def is_divisible_by_3 (abc : ℕ) : Prop :=
  abc % 3 = 0

-- Theorem statement
theorem maximum_modest_number :
  ∃ a b c d : ℕ, is_modest a b c d ∧ is_divisible_by_11 (G a b c d) ∧ is_divisible_by_3 (100 * a + 10 * b + c) ∧ 
  (1000 * a + 100 * b + 10 * c + d) = 3816 := 
sorry

end maximum_modest_number_l149_149501


namespace student_B_speed_l149_149815

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l149_149815


namespace number_of_men_l149_149439

theorem number_of_men (M W C : ℕ) 
  (h1 : M + W + C = 10000)
  (h2 : C = 2500)
  (h3 : C = 5 * W) : 
  M = 7000 := 
by
  sorry

end number_of_men_l149_149439


namespace find_a_l149_149363

theorem find_a (a x1 x2 : ℝ)
  (h1: 4 * x1 ^ 2 - 4 * (a + 2) * x1 + a ^ 2 + 11 = 0)
  (h2: 4 * x2 ^ 2 - 4 * (a + 2) * x2 + a ^ 2 + 11 = 0)
  (h3: x1 - x2 = 3) : a = 4 := sorry

end find_a_l149_149363


namespace initial_workers_l149_149673

/--
In a factory, some workers were employed, and then 25% more workers have just been hired.
There are now 1065 employees in the factory. Prove that the number of workers initially employed is 852.
-/
theorem initial_workers (x : ℝ) (h1 : x + 0.25 * x = 1065) : x = 852 :=
sorry

end initial_workers_l149_149673


namespace positive_irrational_less_than_one_l149_149436

theorem positive_irrational_less_than_one : 
  ∃! (x : ℝ), 
    (x = (Real.sqrt 6) / 3 ∧ Irrational x ∧ 0 < x ∧ x < 1) ∨ 
    (x = -(Real.sqrt 3) / 3 ∧ Irrational x ∧ x < 0) ∨ 
    (x = 1 / 3 ∧ ¬Irrational x ∧ 0 < x ∧ x < 1) ∨ 
    (x = Real.pi / 3 ∧ Irrational x ∧ x > 1) :=
by
  sorry

end positive_irrational_less_than_one_l149_149436


namespace gcd_of_polynomial_and_multiple_l149_149494

theorem gcd_of_polynomial_and_multiple (b : ℕ) (hb : 714 ∣ b) : 
  Nat.gcd (5 * b^3 + 2 * b^2 + 6 * b + 102) b = 102 := by
  sorry

end gcd_of_polynomial_and_multiple_l149_149494


namespace systematic_sampling_example_l149_149939

theorem systematic_sampling_example : 
  ∃ (a : ℕ → ℕ), (∀ i : ℕ, 5 ≤ i ∧ i ≤ 5 → a i = 5 + 10 * (i - 1)) ∧ 
  ∀ i : ℕ, 1 ≤ i ∧ i < 6 → a i - a (i - 1) = a (i + 1) - a i :=
sorry

end systematic_sampling_example_l149_149939


namespace janina_cover_expenses_l149_149514

theorem janina_cover_expenses : 
  ∀ (rent supplies price_per_pancake : ℕ), 
    rent = 30 → 
    supplies = 12 → 
    price_per_pancake = 2 → 
    (rent + supplies) / price_per_pancake = 21 := 
by 
  intros rent supplies price_per_pancake h_rent h_supplies h_price_per_pancake 
  rw [h_rent, h_supplies, h_price_per_pancake]
  sorry

end janina_cover_expenses_l149_149514


namespace speed_of_student_B_l149_149842

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l149_149842


namespace find_min_value_c_l149_149047

theorem find_min_value_c (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 2010) :
  (∃ x y : ℤ, 3 * x + y = 3005 ∧ y = abs (x - a) + abs (x - 2 * b) + abs (x - c) ∧
   (∀ x' y' : ℤ, 3 * x' + y' = 3005 → y' = abs (x' - a) + abs (x' - 2 * b) + abs (x' - c) → x = x' ∧ y = y')) →
  c ≥ 1014 :=
by
  sorry

end find_min_value_c_l149_149047


namespace least_n_satisfies_condition_l149_149617

theorem least_n_satisfies_condition :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (m < n) → ¬(1 / (m : ℝ) - 1 / (m + 1) < 1 / 15)) ∧
    (1 / (n : ℝ) - 1 / (n + 1) < 1 / 15) :=
sorry

end least_n_satisfies_condition_l149_149617


namespace binomial_square_value_l149_149168

theorem binomial_square_value (c : ℝ) : (∃ d : ℝ, 16 * x^2 + 40 * x + c = (4 * x + d) ^ 2) → c = 25 :=
by
  sorry

end binomial_square_value_l149_149168


namespace plywood_perimeter_difference_l149_149102

theorem plywood_perimeter_difference :
  ∀ (length width : ℕ) (n : ℕ), 
    length = 6 ∧ width = 9 ∧ n = 6 → 
    ∃ (max_perimeter min_perimeter : ℕ), 
      (max_perimeter - min_perimeter = 10) ∧
      max_perimeter = 20 ∧ 
      min_perimeter = 10 :=
by
  sorry

end plywood_perimeter_difference_l149_149102


namespace inequality_property_l149_149890

variable {a b : ℝ} (h : a > b) (c : ℝ)

theorem inequality_property : a * |c| ≥ b * |c| :=
sorry

end inequality_property_l149_149890


namespace small_gate_width_l149_149867

-- Bob's garden dimensions
def garden_length : ℝ := 225
def garden_width : ℝ := 125

-- Total fencing needed, including the gates
def total_fencing : ℝ := 687

-- Width of the large gate
def large_gate_width : ℝ := 10

-- Perimeter of the garden without gates
def garden_perimeter : ℝ := 2 * (garden_length + garden_width)

-- Width of the small gate
theorem small_gate_width :
  2 * (garden_length + garden_width) + small_gate + large_gate_width = total_fencing → small_gate = 3 :=
by
  sorry

end small_gate_width_l149_149867


namespace student_b_speed_l149_149820

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l149_149820


namespace manuscript_fee_3800_l149_149949

theorem manuscript_fee_3800 (tax_fee manuscript_fee : ℕ) 
  (h1 : tax_fee = 420) 
  (h2 : (0 < manuscript_fee) ∧ 
        (manuscript_fee ≤ 4000) → 
        tax_fee = (14 * (manuscript_fee - 800)) / 100) 
  (h3 : (manuscript_fee > 4000) → 
        tax_fee = (11 * manuscript_fee) / 100) : manuscript_fee = 3800 :=
by
  sorry

end manuscript_fee_3800_l149_149949


namespace John_has_30_boxes_l149_149354

noncomputable def Stan_boxes : ℕ := 100
noncomputable def Joseph_boxes (S : ℕ) : ℕ := S - (S * 80 / 100)
noncomputable def Jules_boxes (J1 : ℕ) : ℕ := J1 + 5
noncomputable def John_boxes (J2 : ℕ) : ℕ := J2 + (J2 * 20 / 100)

theorem John_has_30_boxes :
  let S := Stan_boxes in
  let J1 := Joseph_boxes S in
  let J2 := Jules_boxes J1 in
  let J3 := John_boxes J2 in
  J3 = 30 :=
by
  sorry

end John_has_30_boxes_l149_149354


namespace find_speed_of_B_l149_149761

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l149_149761


namespace stratified_sampling_l149_149275

-- Define the known quantities
def total_products := 2000
def sample_size := 200
def workshop_production := 250

-- Define the main theorem to prove
theorem stratified_sampling:
  (workshop_production / total_products) * sample_size = 25 := by
  sorry

end stratified_sampling_l149_149275


namespace student_B_speed_l149_149798

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l149_149798


namespace sqrt_sum_of_roots_l149_149941

theorem sqrt_sum_of_roots :
  (36 + 14 * Real.sqrt 6 + 14 * Real.sqrt 5 + 6 * Real.sqrt 30).sqrt
  = (Real.sqrt 15 + Real.sqrt 10 + Real.sqrt 8 + Real.sqrt 3) :=
by
  sorry

end sqrt_sum_of_roots_l149_149941


namespace inequality_solution_set_l149_149647

theorem inequality_solution_set (m : ℝ) : 
  (∀ (x : ℝ), m * x^2 - (1 - m) * x + m ≥ 0) ↔ m ≥ 1/3 := 
sorry

end inequality_solution_set_l149_149647


namespace sqrt_arithmetic_l149_149178

theorem sqrt_arithmetic : 
  sqrt 1.21 / sqrt 0.81 + 
  sqrt 1.44 / sqrt 0.49 = 
  2 + 59 / 63 := by
  sorry

end sqrt_arithmetic_l149_149178


namespace triangle_area_range_l149_149895

theorem triangle_area_range (x₁ x₂ : ℝ) (h₀ : 0 < x₁) (h₁ : x₁ < 1) (h₂ : 1 < x₂) (h₃ : x₁ * x₂ = 1) :
  0 < (2 / (x₁ + 1 / x₁)) ∧ (2 / (x₁ + 1 / x₁)) < 1 :=
by
  sorry

end triangle_area_range_l149_149895


namespace molecular_weight_of_one_mole_l149_149082

theorem molecular_weight_of_one_mole 
  (molicular_weight_9_moles : ℕ) 
  (weight_9_moles : ℕ)
  (h : molicular_weight_9_moles = 972 ∧ weight_9_moles = 9) : 
  molicular_weight_9_moles / weight_9_moles = 108 := 
  by
    sorry

end molecular_weight_of_one_mole_l149_149082


namespace simplify_expression_l149_149098

variable (a : ℝ)

theorem simplify_expression (a : ℝ) : (3 * a) ^ 2 * a ^ 5 = 9 * a ^ 7 :=
by sorry

end simplify_expression_l149_149098


namespace count_rel_prime_21_between_10_and_100_l149_149326

def between (a b : ℕ) (x : ℕ) : Prop := a < x ∧ x < b
def rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem count_rel_prime_21_between_10_and_100 :
  (∑ n in Finset.filter (λ (x : ℕ), between 10 100 x ∧ rel_prime x 21) (Finset.range 100), (1 : ℕ)) = 51 :=
sorry

end count_rel_prime_21_between_10_and_100_l149_149326


namespace speed_of_student_B_l149_149832

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l149_149832


namespace least_n_l149_149632

theorem least_n (n : ℕ) (h_pos : n > 0) (h_ineq : 1 / n - 1 / (n + 1) < 1 / 15) : n = 4 :=
sorry

end least_n_l149_149632


namespace speed_of_student_B_l149_149833

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l149_149833


namespace plywood_perimeter_difference_l149_149139

noncomputable theory

open classical

theorem plywood_perimeter_difference :
  ∃ (rect1 rect2 : ℕ) (a b : ℕ),
  (rect1 = 6 ∧ rect2 = 9 ∧ rect1 % 6 = 0 ∧ rect2 % 6 = 0) ∧ 
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≤ 20) ∧
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≥ 10) ∧ 
  (20 - 10 = 10) :=
by
  exists (6, 9, 6, 9)
  sorry

end plywood_perimeter_difference_l149_149139


namespace boy_speed_l149_149421

theorem boy_speed (d : ℝ) (v₁ v₂ : ℝ) (t₁ t₂ l e : ℝ) :
  d = 2 ∧ v₂ = 8 ∧ l = 7 / 60 ∧ e = 8 / 60 ∧ t₁ = d / v₁ ∧ t₂ = d / v₂ ∧ t₁ - t₂ = l + e → v₁ = 4 :=
by
  sorry

end boy_speed_l149_149421


namespace min_value_75_l149_149488

def min_value (x y z : ℝ) := x^2 + y^2 + z^2

theorem min_value_75 
  (x y z : ℝ) 
  (h1 : (x + 5) * (y - 5) = 0) 
  (h2 : (y + 5) * (z - 5) = 0) 
  (h3 : (z + 5) * (x - 5) = 0) :
  min_value x y z = 75 := 
sorry

end min_value_75_l149_149488


namespace possible_values_of_b_l149_149670

-- Set up the basic definitions and conditions
variable (a b c : ℝ)
variable (A B C : ℝ)

-- Assuming the conditions provided in the problem
axiom cond1 : a * (1 - Real.cos B) = b * Real.cos A
axiom cond2 : c = 3
axiom cond3 : 1 / 2 * a * c * Real.sin B = 2 * Real.sqrt 2

-- The theorem expressing the question and the correct answer
theorem possible_values_of_b : b = 2 ∨ b = 4 * Real.sqrt 2 := sorry

end possible_values_of_b_l149_149670


namespace value_of_2_pow_a_l149_149018

theorem value_of_2_pow_a (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
(h1 : (2^a)^b = 2^2) (h2 : 2^a * 2^b = 8): 2^a = 2 := 
by
  sorry

end value_of_2_pow_a_l149_149018


namespace focus_of_parabola_x2_eq_neg_4y_l149_149531

theorem focus_of_parabola_x2_eq_neg_4y :
  (∀ x y : ℝ, x^2 = -4 * y → focus = (0, -1)) := 
sorry

end focus_of_parabola_x2_eq_neg_4y_l149_149531


namespace least_n_l149_149626

theorem least_n (n : ℕ) (h : 0 < n ∧ (1 / n - 1 / (n + 1) < 1 / 15)) : n ≥ 4 :=
by sorry

end least_n_l149_149626


namespace max_value_of_a_l149_149908

theorem max_value_of_a (a : ℝ) : (∀ x : ℝ, x^2 + |2 * x - 6| ≥ a) → a ≤ 5 :=
by sorry

end max_value_of_a_l149_149908


namespace choir_membership_l149_149535

theorem choir_membership (n : ℕ) (h1 : n % 7 = 4) (h2 : n % 8 = 3) (h3 : n ≥ 100) (h4 : n ≤ 200) :
  n = 123 ∨ n = 179 :=
by
  sorry

end choir_membership_l149_149535


namespace josh_remaining_marbles_l149_149923

theorem josh_remaining_marbles : 
  let initial_marbles := 19 
  let lost_marbles := 11
  initial_marbles - lost_marbles = 8 := by
  sorry

end josh_remaining_marbles_l149_149923


namespace roots_of_quadratic_eq_l149_149398

theorem roots_of_quadratic_eq : ∀ x : ℝ, (x^2 = 9) → (x = 3 ∨ x = -3) :=
by
  sorry

end roots_of_quadratic_eq_l149_149398


namespace seventeen_power_sixty_three_mod_seven_l149_149554

theorem seventeen_power_sixty_three_mod_seven : (17^63) % 7 = 6 := by
  -- Here you would write the actual proof demonstrating the equivalence:
  -- 1. 17 ≡ 3 (mod 7)
  -- 2. Calculate 3^63 (mod 7)
  sorry

end seventeen_power_sixty_three_mod_seven_l149_149554


namespace neg_p_sufficient_but_not_necessary_for_q_l149_149472

variable {x : ℝ}

def p (x : ℝ) : Prop := (1 - x) * (x + 3) < 0
def q (x : ℝ) : Prop := 5 * x - 6 ≤ x^2

theorem neg_p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, ¬ p x → q x) ∧ ¬ (∀ x : ℝ, q x → ¬ p x) :=
by
  sorry

end neg_p_sufficient_but_not_necessary_for_q_l149_149472


namespace harmonic_mean_average_of_x_is_11_l149_149248

theorem harmonic_mean_average_of_x_is_11 :
  let h := (2 * 1008) / (2 + 1008)
  ∃ (x : ℕ), (h + x) / 2 = 11 → x = 18 := by
  sorry

end harmonic_mean_average_of_x_is_11_l149_149248


namespace student_b_speed_l149_149826

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l149_149826


namespace train_length_l149_149150

theorem train_length :
  ∃ L : ℝ, 
    (∀ V : ℝ, V = L / 24 ∧ V = (L + 650) / 89) → 
    L = 240 :=
by
  sorry

end train_length_l149_149150


namespace distance_swim_against_current_l149_149434

-- Definitions based on problem conditions
def swimmer_speed_still_water : ℝ := 4 -- km/h
def water_current_speed : ℝ := 1 -- km/h
def time_swimming_against_current : ℝ := 2 -- hours

-- Calculation of effective speed against the current
def effective_speed_against_current : ℝ :=
  swimmer_speed_still_water - water_current_speed

-- Proof statement
theorem distance_swim_against_current :
  effective_speed_against_current * time_swimming_against_current = 6 :=
by
  -- By substituting values from the problem,
  -- effective_speed_against_current * time_swimming_against_current = 3 * 2
  -- which equals 6.
  sorry

end distance_swim_against_current_l149_149434


namespace find_speed_of_B_l149_149760

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l149_149760


namespace intersection_M_N_l149_149196

-- Definitions of the sets M and N
def M : Set ℤ := {-3, -2, -1}
def N : Set ℤ := { x | -2 < x ∧ x < 3 }

-- The theorem stating that the intersection of M and N is {-1}
theorem intersection_M_N : M ∩ N = {-1} := by
  sorry

end intersection_M_N_l149_149196


namespace elderly_in_sample_l149_149423

variable (A E M : ℕ)
variable (total_employees : ℕ)
variable (total_young : ℕ)
variable (sample_size_young : ℕ)
variable (sampling_ratio : ℚ)
variable (sample_elderly : ℕ)

axiom condition_1 : total_young = 160
axiom condition_2 : total_employees = 430
axiom condition_3 : M = 2 * E
axiom condition_4 : A + M + E = total_employees
axiom condition_5 : sampling_ratio = sample_size_young / total_young
axiom sampling : sample_size_young = 32
axiom elderly_employees : sample_elderly = 18

theorem elderly_in_sample : sample_elderly = sampling_ratio * E := by
  -- Proof steps are not provided
  sorry

end elderly_in_sample_l149_149423


namespace find_CB_l149_149035

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V)

-- Given condition
-- D divides AB in the ratio 1:3 such that CA = a and CD = b

def D_divides_AB (A B D : V) : Prop := ∃ (k : ℝ), k = 1 / 4 ∧ A + k • (B - A) = D

theorem find_CB (CA CD : V) (A B D : V) (h1 : CA = A) (h2 : CD = B)
  (h3 : D_divides_AB A B D) : (B - A) = -3 • CA + 4 • CD :=
sorry

end find_CB_l149_149035


namespace find_a_l149_149170

noncomputable def exists_nonconstant_function (a : ℝ) : Prop :=
  ∃ f : ℝ → ℝ, (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 ≠ f x2) ∧ 
  (∀ x : ℝ, f (a * x) = a^2 * f x) ∧
  (∀ x : ℝ, f (f x) = a * f x)

theorem find_a :
  ∀ (a : ℝ), exists_nonconstant_function a → (a = 0 ∨ a = 1) :=
by
  sorry

end find_a_l149_149170


namespace cost_price_of_one_ball_is_48_l149_149412

-- Define the cost price of one ball
def costPricePerBall (x : ℝ) : Prop :=
  let totalCostPrice20Balls := 20 * x
  let sellingPrice20Balls := 720
  let loss := 5 * x
  totalCostPrice20Balls = sellingPrice20Balls + loss

-- Define the main proof problem
theorem cost_price_of_one_ball_is_48 (x : ℝ) (h : costPricePerBall x) : x = 48 :=
by
  sorry

end cost_price_of_one_ball_is_48_l149_149412


namespace find_a_n_l149_149186

theorem find_a_n (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n, S n = 3^n + 2) :
  ∀ n, a n = if n = 1 then 5 else 2 * 3^(n - 1) := by
  sorry

end find_a_n_l149_149186


namespace other_root_of_quadratic_l149_149002

theorem other_root_of_quadratic (m : ℝ) (h : ∀ x : ℝ, x^2 + m*x - 20 = 0 → (x = -4)) 
: ∃ t : ℝ, t = 5 := 
by
  existsi 5
  sorry

end other_root_of_quadratic_l149_149002


namespace tiffany_total_bags_l149_149727

theorem tiffany_total_bags (monday_bags next_day_bags : ℕ) (h1 : monday_bags = 4) (h2 : next_day_bags = 8) :
  monday_bags + next_day_bags = 12 :=
by
  sorry

end tiffany_total_bags_l149_149727


namespace find_xy_yz_xz_l149_149270

-- Define the conditions given in the problem
variables (x y z : ℝ)
variable (hxyz_pos : x > 0 ∧ y > 0 ∧ z > 0)
variable (h1 : x^2 + x * y + y^2 = 12)
variable (h2 : y^2 + y * z + z^2 = 16)
variable (h3 : z^2 + z * x + x^2 = 28)

-- State the theorem to be proved
theorem find_xy_yz_xz : x * y + y * z + x * z = 16 :=
by {
    -- Proof will be done here
    sorry
}

end find_xy_yz_xz_l149_149270


namespace normal_line_at_x0_is_correct_l149_149460

noncomputable def curve (x : ℝ) : ℝ := x^(2/3) - 20

def x0 : ℝ := -8

def normal_line_equation (x : ℝ) : ℝ := 3 * x + 8

theorem normal_line_at_x0_is_correct : 
  ∃ y0 : ℝ, curve x0 = y0 ∧ y0 = curve x0 ∧ normal_line_equation x0 = y0 :=
sorry

end normal_line_at_x0_is_correct_l149_149460


namespace line_always_passes_through_fixed_point_l149_149485

theorem line_always_passes_through_fixed_point (k : ℝ) : 
  ∀ x y, y + 2 = k * (x + 1) → (x = -1 ∧ y = -2) :=
by
  sorry

end line_always_passes_through_fixed_point_l149_149485


namespace plywood_perimeter_difference_l149_149137

noncomputable theory

open classical

theorem plywood_perimeter_difference :
  ∃ (rect1 rect2 : ℕ) (a b : ℕ),
  (rect1 = 6 ∧ rect2 = 9 ∧ rect1 % 6 = 0 ∧ rect2 % 6 = 0) ∧ 
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≤ 20) ∧
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≥ 10) ∧ 
  (20 - 10 = 10) :=
by
  exists (6, 9, 6, 9)
  sorry

end plywood_perimeter_difference_l149_149137


namespace plywood_cut_difference_l149_149107

/-- A proof problem to determine the positive difference between the greatest possible
perimeter and the least possible perimeter of congruent pieces resulting from cutting 
a 6-foot by 9-foot rectangular plywood into 6 congruent rectangles with no wood leftover 
or lost --/
theorem plywood_cut_difference :
  ∃ (perimeter_max perimeter_min : ℕ), 
  let piece1 := 1 * 9
  let piece2 := 1 * 6
  let piece3 := 2 * 3
  let perimeter1 := 2 * (1 + 9)
  let perimeter2 := 2 * (1 + 6)
  let perimeter3 := 2 * (2 + 3)
  perimeter_max = perimeter1 ∧
  perimeter_min = perimeter3 ∧
  (perimeter_max - perimeter_min) = 10 :=
sorry

end plywood_cut_difference_l149_149107


namespace steps_to_Madison_eq_991_l149_149286

variable (steps_down steps_to_Madison : ℕ)

def total_steps (steps_down steps_to_Madison : ℕ) : ℕ :=
  steps_down + steps_to_Madison

theorem steps_to_Madison_eq_991 (h1 : steps_down = 676) (h2 : steps_to_Madison = 315) :
  total_steps steps_down steps_to_Madison = 991 :=
by
  sorry

end steps_to_Madison_eq_991_l149_149286


namespace BB_digit_value_in_5BB3_l149_149392

theorem BB_digit_value_in_5BB3 (B : ℕ) (h : 2 * B + 8 % 9 = 0) : B = 5 :=
sorry

end BB_digit_value_in_5BB3_l149_149392


namespace evaluate_expression_l149_149590

theorem evaluate_expression (x y z : ℚ) 
    (hx : x = 1 / 4) 
    (hy : y = 1 / 3) 
    (hz : z = -6) : 
    x^2 * y^3 * z^2 = 1 / 12 :=
by
  sorry

end evaluate_expression_l149_149590


namespace remaining_black_cards_l149_149420

theorem remaining_black_cards 
  (total_cards : ℕ)
  (black_cards : ℕ)
  (red_cards : ℕ)
  (cards_taken_out : ℕ)
  (h1 : total_cards = 52)
  (h2 : black_cards = 26)
  (h3 : red_cards = 26)
  (h4 : cards_taken_out = 5) :
  black_cards - cards_taken_out = 21 := 
by {
  sorry
}

end remaining_black_cards_l149_149420


namespace student_B_speed_l149_149813

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l149_149813


namespace distinguishable_large_triangles_l149_149076

def num_of_distinguishable_large_eq_triangles : Nat :=
  let colors := 8
  let pairs := 7 + Nat.choose 7 2
  colors * pairs

theorem distinguishable_large_triangles : num_of_distinguishable_large_eq_triangles = 224 := by
  sorry

end distinguishable_large_triangles_l149_149076


namespace find_speed_of_B_l149_149762

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l149_149762


namespace expand_expression_l149_149876

theorem expand_expression (y : ℝ) : (7 * y + 12) * 3 * y = 21 * y ^ 2 + 36 * y := by
  sorry

end expand_expression_l149_149876


namespace plywood_cut_perimeter_difference_l149_149116

theorem plywood_cut_perimeter_difference :
  (∃ (l w : ℕ), (l * w = 54) ∧ (9 % w = 0) ∧ (6 % l = 0) ∧ (6 / l) * (9 / w) = 6) →
  10 =
  let p := λ l w, 2 * (l + w) in
  let perimeters := [
    p 1 9,
    p 1 6,
    p 2 3,
    p 3 2
  ]
  in (list.max precedence perimeters - list.min precedence perimeters) :=
begin
  sorry
end

end plywood_cut_perimeter_difference_l149_149116


namespace iron_wire_left_l149_149986

-- Given conditions as variables
variable (initial_usage : ℚ) (additional_usage : ℚ)

-- Conditions as hypotheses
def conditions := initial_usage = 2 / 9 ∧ additional_usage = 3 / 9

-- The goal to prove
theorem iron_wire_left (h : conditions initial_usage additional_usage):
  1 - initial_usage - additional_usage = 4 / 9 :=
by
  -- Insert proof here
  sorry

end iron_wire_left_l149_149986


namespace student_B_speed_l149_149812

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l149_149812


namespace hyperbola_standard_equation_l149_149676

theorem hyperbola_standard_equation
  (passes_through : ∀ {x y : ℝ}, (x, y) = (1, 1) → 2 * x + y = 0 ∨ 2 * x - y = 0)
  (asymptote1 : ∀ {x y : ℝ}, 2 * x + y = 0 → y = -2 * x)
  (asymptote2 : ∀ {x y : ℝ}, 2 * x - y = 0 → y = 2 * x) :
  ∃ a b : ℝ, a = 4 / 3 ∧ b = 1 / 3 ∧ ∀ x y : ℝ, (x, y) = (1, 1) → (x^2 / a - y^2 / b = 1) := 
sorry

end hyperbola_standard_equation_l149_149676


namespace student_B_speed_l149_149796

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l149_149796


namespace number_of_homework_situations_l149_149538

theorem number_of_homework_situations (teachers students : ℕ) (homework_options : students = 4 ∧ teachers = 3) :
  teachers ^ students = 81 :=
by
  sorry

end number_of_homework_situations_l149_149538


namespace correct_removal_of_parentheses_l149_149409

theorem correct_removal_of_parentheses (x : ℝ) : (1/3) * (6 * x - 3) = 2 * x - 1 :=
by sorry

end correct_removal_of_parentheses_l149_149409


namespace factorize_square_difference_l149_149302

theorem factorize_square_difference (x: ℝ):
  x^2 - 4 = (x + 2) * (x - 2) := by
  -- Using the difference of squares formula a^2 - b^2 = (a + b)(a - b)
  sorry

end factorize_square_difference_l149_149302


namespace find_speed_of_B_l149_149801

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l149_149801


namespace simplify_expression_l149_149097

variable (a : ℝ)

theorem simplify_expression (a : ℝ) : (3 * a) ^ 2 * a ^ 5 = 9 * a ^ 7 :=
by sorry

end simplify_expression_l149_149097


namespace student_b_speed_l149_149822

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l149_149822


namespace rationalize_denominator_l149_149373

theorem rationalize_denominator :
  (∃ x y z w : ℂ, x = sqrt 12 ∧ y = sqrt 5 ∧
  z = sqrt 3 ∧ w = sqrt 5 ∧
  (x + y) / (z + w) = (sqrt 15 - 1) / 2) :=
by
  use [√12, √5, √3, √5]
  sorry

end rationalize_denominator_l149_149373


namespace sum_of_abs_coeffs_l149_149014

theorem sum_of_abs_coeffs (a : ℕ → ℤ) :
  (∀ x : ℤ, (1 - x)^5 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5) →
  |a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| = 32 := 
by
  sorry

end sum_of_abs_coeffs_l149_149014


namespace slope_probability_l149_149358

noncomputable def probability_of_slope_gte (x y : ℝ) (Q : ℝ × ℝ) : ℝ :=
  if y - 1 / 4 ≥ (2 / 3) * (x - 3 / 4) then 1 else 0

theorem slope_probability :
  let unit_square_area := 1  -- the area of the unit square
  let valid_area := (1 / 2) * (5 / 8) * (5 / 12) -- area of the triangle above the line
  valid_area / unit_square_area = 25 / 96 :=
sorry

end slope_probability_l149_149358


namespace least_n_l149_149630

theorem least_n (n : ℕ) (h_pos : n > 0) (h_ineq : 1 / n - 1 / (n + 1) < 1 / 15) : n = 4 :=
sorry

end least_n_l149_149630


namespace geometric_seq_a6_value_l149_149210

theorem geometric_seq_a6_value 
    (a : ℕ → ℝ) 
    (q : ℝ) 
    (h_q_pos : q > 0)
    (h_a_pos : ∀ n, a n > 0)
    (h_a2 : a 2 = 1)
    (h_a8_eq : a 8 = a 6 + 2 * a 4) : 
    a 6 = 4 := 
by 
  sorry

end geometric_seq_a6_value_l149_149210


namespace max_sin_sin2x_l149_149040

theorem max_sin_sin2x (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) : 
  ∃ y, y = sin x * sin (2 * x) ∧ y ≤ 4 * Real.sqrt 3 / 9 :=
sorry

end max_sin_sin2x_l149_149040


namespace find_m_l149_149023

theorem find_m (m : ℝ) : (∀ x : ℝ, x - m > 5 ↔ x > 2) → m = -3 :=
by
  sorry

end find_m_l149_149023


namespace point_P_path_length_l149_149969

/-- A rectangle PQRS in the plane with points P Q R S, where PQ = RS = 2 and QR = SP = 6. 
    The rectangle is rotated 90 degrees twice: first about point R and then 
    about the new position of point S after the first rotation. 
    The goal is to prove that the length of the path P travels is (3 + sqrt 10) * pi. -/
theorem point_P_path_length :
  ∀ (P Q R S : ℝ × ℝ), 
    dist P Q = 2 ∧ dist Q R = 6 ∧ dist R S = 2 ∧ dist S P = 6 →
    ∃ path_length : ℝ, path_length = (3 + Real.sqrt 10) * Real.pi :=
by
  sorry

end point_P_path_length_l149_149969


namespace probability_same_flips_l149_149405

-- Define the probability of getting the first head on the nth flip
def prob_first_head_on_nth_flip (n : ℕ) : ℚ :=
  (1 / 2) ^ n

-- Define the probability that all three get the first head on the nth flip
def prob_all_three_first_head_on_nth_flip (n : ℕ) : ℚ :=
  (prob_first_head_on_nth_flip n) ^ 3

-- Define the total probability considering all n
noncomputable def total_prob_all_three_same_flips : ℚ :=
  ∑' n, prob_all_three_first_head_on_nth_flip (n + 1)

-- The statement to prove
theorem probability_same_flips : total_prob_all_three_same_flips = 1 / 7 :=
by sorry

end probability_same_flips_l149_149405


namespace rationalize_denominator_l149_149371

theorem rationalize_denominator : 
  (√12 + √5) / (√3 + √5) = (√15 - 1) / 2 :=
by sorry

end rationalize_denominator_l149_149371


namespace part1_part2_l149_149654

noncomputable def f (x : ℝ) : ℝ := |x - 2|
noncomputable def g (x m : ℝ) : ℝ := -|x + 3| + m

def solution_set_ineq_1 (a : ℝ) : Set ℝ :=
  if a = 1 then {x | x < 2 ∨ x > 2}
  else if a > 1 then Set.univ
  else {x | x < 1 + a ∨ x > 3 - a}

theorem part1 (a : ℝ) : 
  ∃ S : Set ℝ, S = solution_set_ineq_1 a ∧ ∀ x : ℝ, (f x + a - 1 > 0) ↔ x ∈ S := sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, f x ≥ g x m) ↔ m < 5 := sorry

end part1_part2_l149_149654


namespace polar_to_cartesian_l149_149898

-- Define the conditions
def polar_eq (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define the goal as a theorem
theorem polar_to_cartesian : ∀ (x y : ℝ), 
  (∃ θ : ℝ, polar_eq (Real.sqrt (x^2 + y^2)) θ ∧ x = (Real.sqrt (x^2 + y^2)) * Real.cos θ 
  ∧ y = (Real.sqrt (x^2 + y^2)) * Real.sin θ) → (x-1)^2 + y^2 = 1 :=
by
  intro x y
  intro h
  sorry

end polar_to_cartesian_l149_149898


namespace rower_trip_time_to_Big_Rock_l149_149147

noncomputable def row_trip_time (rowing_speed_in_still_water : ℝ) (river_speed : ℝ) (distance_to_destination : ℝ) : ℝ :=
  let speed_upstream := rowing_speed_in_still_water - river_speed
  let speed_downstream := rowing_speed_in_still_water + river_speed
  let time_upstream := distance_to_destination / speed_upstream
  let time_downstream := distance_to_destination / speed_downstream
  time_upstream + time_downstream

theorem rower_trip_time_to_Big_Rock :
  row_trip_time 7 2 3.2142857142857144 = 1 :=
by
  sorry

end rower_trip_time_to_Big_Rock_l149_149147


namespace transformed_system_solution_l149_149320

theorem transformed_system_solution (a b : ℝ) (x : ℝ) (h1 : a < 0) (h2 : b < 0) :
  ((49^(x + 1) - 50 * 7^x + 1 < 0) ∧ (log (x + 5 / 2) (abs (x + 1 / 2)) < 0)) ↔
  ((-2 < x ∧ x < -3 / 2) ∨ (-3 / 2 < x ∧ x < -1 / 2) ∨ (-1 / 2 < x ∧ x ≤ 0)) :=
sorry

end transformed_system_solution_l149_149320


namespace value_of_x_l149_149339

theorem value_of_x (x : ℝ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 :=
by
  intro h
  sorry

end value_of_x_l149_149339


namespace polynomial_bound_l149_149470

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem polynomial_bound (a b c d : ℝ) (hP : ∀ x : ℝ, |x| < 1 → |P x a b c d| ≤ 1) : 
  |a| + |b| + |c| + |d| ≤ 7 := 
sorry

end polynomial_bound_l149_149470


namespace TV_height_l149_149864

theorem TV_height (Area Width Height : ℝ) (h_area : Area = 21) (h_width : Width = 3) (h_area_def : Area = Width * Height) : Height = 7 := 
by
  sorry

end TV_height_l149_149864


namespace negation_of_proposition_l149_149950

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), a > b → a^2 > b^2) ↔ ∃ (a b : ℝ), a ≤ b ∧ a^2 ≤ b^2 :=
sorry

end negation_of_proposition_l149_149950


namespace coeff_x2y2_in_expansion_l149_149064

-- Define the coefficient of a specific term in the binomial expansion
def coeff_binom (n k : ℕ) (a b : ℤ) (x y : ℕ) : ℤ :=
  (Nat.choose n k) * (a ^ (n - k)) * (b ^ k)

theorem coeff_x2y2_in_expansion : coeff_binom 4 2 1 (-2) 2 2 = 24 := by
  sorry

end coeff_x2y2_in_expansion_l149_149064


namespace diagonal_of_square_l149_149391

theorem diagonal_of_square (d : ℝ) (s : ℝ) (h : d = 2) (h_eq : s * Real.sqrt 2 = d) : s = Real.sqrt 2 :=
by sorry

end diagonal_of_square_l149_149391


namespace probability_correct_l149_149915

noncomputable def probability_B1_eq_5_given_WB : ℚ :=
  let P_B1_eq_5 : ℚ := 1 / 8
  let P_WB : ℚ := 1 / 5
  let P_WB_given_B1_eq_5 : ℚ := 1 / 16 + 369 / 2048
  (P_B1_eq_5 * P_WB_given_B1_eq_5) / P_WB

theorem probability_correct :
  probability_B1_eq_5_given_WB = 115 / 1024 :=
by
  sorry

end probability_correct_l149_149915


namespace range_of_m_l149_149888

theorem range_of_m (m : ℝ) (x0 : ℝ)
  (h : (4^(-x0) - m * 2^(-x0 + 1)) = -(4^x0 - m * 2^(x0 + 1))) :
  m ≥ 1/2 :=
sorry

end range_of_m_l149_149888


namespace rationalize_denominator_l149_149378

noncomputable def sqrt_12 := Real.sqrt 12
noncomputable def sqrt_5 := Real.sqrt 5
noncomputable def sqrt_3 := Real.sqrt 3
noncomputable def sqrt_15 := Real.sqrt 15

theorem rationalize_denominator :
  (sqrt_12 + sqrt_5) / (sqrt_3 + sqrt_5) = (-1 / 2) + (sqrt_15 / 2) :=
sorry

end rationalize_denominator_l149_149378


namespace parabola_transformation_l149_149909

theorem parabola_transformation :
  ∀ (x y : ℝ), (y = x^2) →
  let shiftedRight := (x - 3)
  let shiftedUp := (shiftedRight)^2 + 4
  y = shiftedUp :=
by
  intros x y,
  intro hyp,
  let shiftedRight := (x - 3),
  let shiftedUp := (shiftedRight)^2 + 4,
  have eq_shifted : y = shiftedRight^2 := by sorry,
  have eq_final : y = shiftedUp := by sorry,
  exact eq_final
  
end parabola_transformation_l149_149909


namespace total_fault_line_movement_l149_149998

-- Define the movements in specific years.
def movement_past_year : ℝ := 1.25
def movement_year_before : ℝ := 5.25

-- Theorem stating the total movement of the fault line over the two years.
theorem total_fault_line_movement : movement_past_year + movement_year_before = 6.50 :=
by
  -- Proof is omitted.
  sorry

end total_fault_line_movement_l149_149998


namespace student_B_speed_l149_149780

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l149_149780


namespace number_of_rows_with_exactly_7_students_l149_149671

theorem number_of_rows_with_exactly_7_students 
  (total_students : ℕ) (rows_with_6_students rows_with_7_students : ℕ) 
  (total_students_eq : total_students = 53)
  (seats_condition : total_students = 6 * rows_with_6_students + 7 * rows_with_7_students) 
  (no_seat_unoccupied : rows_with_6_students + rows_with_7_students = rows_with_6_students + rows_with_7_students) :
  rows_with_7_students = 5 := by
  sorry

end number_of_rows_with_exactly_7_students_l149_149671


namespace inequality_proof_l149_149701

theorem inequality_proof (a b c d : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : d > 0)
    (h_cond : 2 * (a + b + c + d) ≥ a * b * c * d) : (a^2 + b^2 + c^2 + d^2) ≥ (a * b * c * d) :=
by
  sorry

end inequality_proof_l149_149701


namespace simplify_expression_l149_149201

theorem simplify_expression (y : ℝ) : (y - 2) ^ 2 + 2 * (y - 2) * (4 + y) + (4 + y) ^ 2 = 4 * (y + 1) ^ 2 := 
by 
  sorry

end simplify_expression_l149_149201


namespace determine_angle_B_l149_149213

noncomputable def problem_statement (A B C : ℝ) (a b c : ℝ) : Prop :=
  (2 * (Real.cos ((A - B) / 2))^2 * Real.cos B - Real.sin (A - B) * Real.sin B + Real.cos (A + C) = -3 / 5)
  ∧ (a = 8)
  ∧ (b = Real.sqrt 3)

theorem determine_angle_B (A B C : ℝ) (a b c : ℝ)
  (h : problem_statement A B C a b c) : 
  B = Real.arcsin (Real.sqrt 3 / 10) :=
by 
  sorry

end determine_angle_B_l149_149213


namespace parabola_shift_l149_149911

-- Define the initial equation of the parabola
def initial_parabola (x : ℝ) : ℝ := x^2

-- Define the shift function for shifting the parabola right by 3 units
def shift_right (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x - a)

-- Define the shift function for shifting the parabola up by 4 units
def shift_up (f : ℝ → ℝ) (b : ℝ) (y : ℝ) : ℝ := y + b

-- Define the transformed parabola
def transformed_parabola (x : ℝ) : ℝ := shift_up (shift_right initial_parabola 3) 4 (initial_parabola x)

-- Goal: Prove that the transformed parabola is y = (x - 3)^2 + 4
theorem parabola_shift (x : ℝ) : transformed_parabola x = (x - 3)^2 + 4 := sorry

end parabola_shift_l149_149911


namespace Marty_combination_count_l149_149520

theorem Marty_combination_count :
  let num_colors := 4
  let num_methods := 3
  num_colors * num_methods = 12 :=
by
  let num_colors := 4
  let num_methods := 3
  sorry

end Marty_combination_count_l149_149520


namespace speed_of_student_B_l149_149782

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l149_149782


namespace least_number_of_colors_needed_l149_149731

-- Define the tessellation of hexagons
structure HexagonalTessellation :=
(adjacent : (ℕ × ℕ) → (ℕ × ℕ) → Prop)
(symm : ∀ {a b : ℕ × ℕ}, adjacent a b → adjacent b a)
(irrefl : ∀ a : ℕ × ℕ, ¬ adjacent a a)
(hex_property : ∀ a : ℕ × ℕ, ∃ b1 b2 b3 b4 b5 b6,
  adjacent a b1 ∧ adjacent a b2 ∧ adjacent a b3 ∧ adjacent a b4 ∧ adjacent a b5 ∧ adjacent a b6)

-- Define a coloring function for a HexagonalTessellation
def coloring (T : HexagonalTessellation) (colors : ℕ) :=
(∀ (a b : ℕ × ℕ), T.adjacent a b → a ≠ b → colors ≥ 1 → colors ≤ 3)

-- Statement to prove the minimum number of colors required
theorem least_number_of_colors_needed (T : HexagonalTessellation) :
  ∃ colors, coloring T colors ∧ colors = 3 :=
sorry

end least_number_of_colors_needed_l149_149731


namespace symmetry_about_origin_l149_149190

-- Define the conditions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = g x

-- Define the function v based on f and g
def v (f g : ℝ → ℝ) (x : ℝ) : ℝ := f x * |g x|

-- The theorem statement
theorem symmetry_about_origin (f g : ℝ → ℝ) (h_odd : is_odd f) (h_even : is_even g) : 
  ∀ x : ℝ, v f g (-x) = -v f g x := 
by
  sorry

end symmetry_about_origin_l149_149190


namespace calc_625_to_4_div_5_l149_149158

theorem calc_625_to_4_div_5 :
  (625 : ℝ)^(4/5) = 238 :=
sorry

end calc_625_to_4_div_5_l149_149158


namespace percent_other_sales_l149_149706

-- Define the given conditions
def s_brushes : ℝ := 0.45
def s_paints : ℝ := 0.28

-- Define the proof goal in Lean
theorem percent_other_sales :
  1 - (s_brushes + s_paints) = 0.27 := by
-- Adding the conditions to the proof environment
  sorry

end percent_other_sales_l149_149706


namespace sin_alpha_sol_cos_2alpha_pi4_sol_l149_149466

open Real

-- Define the main problem conditions
def cond1 (α : ℝ) := sin (α + π / 3) + sin α = 9 * sqrt 7 / 14
def range (α : ℝ) := 0 < α ∧ α < π / 3

-- Define the statement for the first problem
theorem sin_alpha_sol (α : ℝ) (h1 : cond1 α) (h2 : range α) : sin α = 2 * sqrt 7 / 7 := 
sorry

-- Define the statement for the second problem
theorem cos_2alpha_pi4_sol (α : ℝ) (h1 : cond1 α) (h2 : range α) (h3 : sin α = 2 * sqrt 7 / 7) : 
  cos (2 * α - π / 4) = (4 * sqrt 6 - sqrt 2) / 14 := 
sorry

end sin_alpha_sol_cos_2alpha_pi4_sol_l149_149466


namespace math_problem_proof_l149_149641

noncomputable def least_n : ℕ := 4

theorem math_problem_proof : 
  ∃ n ≥ 1, ((1 : ℝ) / n) - ((1 : ℝ) / (n + 1)) < 1 / 15 ∧ 
           ∀ m ≥ 1, (((1 : ℝ) / m - (1 : ℝ) / (m + 1)) < 1 / 15) → 
           least_n ≤ m := 
by
  use least_n
  sorry

end math_problem_proof_l149_149641


namespace one_tail_in_three_tosses_l149_149562

open Probability

-- Define the fairness of the coin
def fair_coin : ProbSpace :=
{ space := bool,
  prob := λ b, 1/2 }

-- Define the experiment of tossing the coin 3 times
def coin_toss_experiment : ProbSpace :=
  vector_measure_of (λ _, fair_coin) 3

-- Define the specific event occurrence of exactly one tail and rest heads in 3 tosses
def one_tail_and_rest_heads : set (vector bool 3) :=
  {v | (v.to_list.filter id).length = 2}

-- State the theorem to prove the desired probability
theorem one_tail_in_three_tosses (S : ProbSpace) :
  (@measure S _ coin_toss_experiment one_tail_and_rest_heads) = 3/8 :=
sorry

end one_tail_in_three_tosses_l149_149562


namespace fixed_point_of_line_l149_149715

theorem fixed_point_of_line :
  ∀ m : ℝ, ∀ x y : ℝ, (y - 2 = m * (x + 1)) → (x = -1 ∧ y = 2) :=
by sorry

end fixed_point_of_line_l149_149715


namespace binomial_coeff_sum_l149_149477

theorem binomial_coeff_sum 
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ)
  (h1 : (1 - 2 * 0 : ℝ)^(7) = a_0 + a_1 * 0 + a_2 * 0^2 + a_3 * 0^3 + a_4 * 0^4 + a_5 * 0^5 + a_6 * 0^6 + a_7 * 0^7)
  (h2 : (1 - 2 * 1 : ℝ)^(7) = a_0 + a_1 * 1 + a_2 * 1^2 + a_3 * 1^3 + a_4 * 1^4 + a_5 * 1^5 + a_6 * 1^6 + a_7 * 1^7) :
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = -2 := 
sorry

end binomial_coeff_sum_l149_149477


namespace men_in_club_l149_149749

-- Definitions
variables (M W : ℕ) -- Number of men and women

-- Conditions
def club_members := M + W = 30
def event_participation := W / 3 + M = 18

-- Goal
theorem men_in_club : club_members M W → event_participation M W → M = 12 :=
sorry

end men_in_club_l149_149749


namespace arth_seq_val_a7_l149_149887

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem arth_seq_val_a7 {a : ℕ → ℝ} 
  (h_arith : arithmetic_sequence a)
  (h_positive : ∀ n : ℕ, 0 < a n)
  (h_eq : 2 * a 6 + 2 * a 8 = (a 7) ^ 2) :
  a 7 = 4 := 
by sorry

end arth_seq_val_a7_l149_149887


namespace find_b_l149_149563

-- Definitions based on the given conditions
def good_point (a b : ℝ) (φ : ℝ) : Prop :=
  a + (b - a) * φ = 2.382 ∨ b - (b - a) * φ = 2.382

theorem find_b (b : ℝ) (φ : ℝ := 0.618) :
  good_point 2 b φ → b = 2.618 ∨ b = 3 :=
by
  sorry

end find_b_l149_149563


namespace local_minimum_at_2_l149_149364

noncomputable def f (x : ℝ) : ℝ := (2 / x) + Real.log x

theorem local_minimum_at_2 : ∃ δ > 0, ∀ y, abs (y - 2) < δ → f y ≥ f 2 := by
  sorry

end local_minimum_at_2_l149_149364


namespace area_of_rectangle_l149_149754

theorem area_of_rectangle (A G Y : ℝ) 
  (hG : G = 0.15 * A) 
  (hY : Y = 21) 
  (hG_plus_Y : G + Y = 0.5 * A) : 
  A = 60 := 
by 
  -- proof goes here
  sorry

end area_of_rectangle_l149_149754


namespace randolph_age_l149_149054

theorem randolph_age (R Sy S : ℕ) 
  (h1 : R = Sy + 5) 
  (h2 : Sy = 2 * S) 
  (h3 : S = 25) : 
  R = 55 :=
by 
  sorry

end randolph_age_l149_149054


namespace surjective_injective_eq_l149_149928

theorem surjective_injective_eq (f g : ℕ → ℕ) 
  (hf : Function.Surjective f) 
  (hg : Function.Injective g) 
  (h : ∀ n : ℕ, f n ≥ g n) : 
  ∀ n : ℕ, f n = g n := 
by
  sorry

end surjective_injective_eq_l149_149928


namespace log_defined_for_powers_of_a_if_integer_exponents_log_undefined_if_only_positive_indices_l149_149090

variable (a : ℝ) (b : ℝ)

-- Conditions
axiom base_pos (h : a > 0) : a ≠ 1
axiom integer_exponents_only (h : ∃ n : ℤ, b = a^n) : True
axiom positive_indices_only (h : ∃ n : ℕ, b = a^n) : 0 < b ∧ b < 1 → False

-- Theorem: If we only knew integer exponents, the logarithm of any number b in base a is defined for powers of a.
theorem log_defined_for_powers_of_a_if_integer_exponents (h : ∃ n : ℤ, b = a^n) : True :=
by sorry

-- Theorem: If we only knew positive exponents, the logarithm of any number b in base a is undefined for all 0 < b < 1
theorem log_undefined_if_only_positive_indices : (∃ n : ℕ, b = a^n) → (0 < b ∧ b < 1 → False) :=
by sorry

end log_defined_for_powers_of_a_if_integer_exponents_log_undefined_if_only_positive_indices_l149_149090


namespace relationship_xy_l149_149197

def M (x : ℤ) : Prop := ∃ m : ℤ, x = 3 * m + 1
def N (y : ℤ) : Prop := ∃ n : ℤ, y = 3 * n + 2

theorem relationship_xy (x y : ℤ) (hx : M x) (hy : N y) : N (x * y) ∧ ¬ M (x * y) :=
by
  sorry

end relationship_xy_l149_149197


namespace least_n_inequality_l149_149616

theorem least_n_inequality (n : ℕ) (hn : 0 < n) (ineq: 1 / n - 1 / (n + 1) < 1 / 15) :
  4 ≤ n :=
begin
  simp [lt_div_iff] at ineq,
  rw [sub_eq_add_neg, add_div_eq_mul_add_neg_div, ←lt_div_iff], 
  exact ineq
end

end least_n_inequality_l149_149616


namespace least_n_l149_149627

theorem least_n (n : ℕ) (h : 0 < n ∧ (1 / n - 1 / (n + 1) < 1 / 15)) : n ≥ 4 :=
by sorry

end least_n_l149_149627


namespace total_trees_in_gray_areas_l149_149427

theorem total_trees_in_gray_areas (white_region_first : ℕ) (white_region_second : ℕ)
    (total_first : ℕ) (total_second : ℕ)
    (h1 : white_region_first = 82) (h2 : white_region_second = 82)
    (h3 : total_first = 100) (h4 : total_second = 90) :
  (total_first - white_region_first) + (total_second - white_region_second) = 26 := by
  sorry

end total_trees_in_gray_areas_l149_149427


namespace P_gt_Q_l149_149312

theorem P_gt_Q (a : ℝ) : 
  let P := a^2 + 2*a
  let Q := 3*a - 1
  P > Q :=
by
  sorry

end P_gt_Q_l149_149312


namespace inequality_condition_l149_149067

theorem inequality_condition (x : ℝ) :
  ((x + 3) * (x - 2) < 0 ↔ -3 < x ∧ x < 2) →
  ((-3 < x ∧ x < 0) → (x + 3) * (x - 2) < 0) →
  ∃ p q : Prop, (p → q) ∧ ¬(q → p) ∧
  p = ((x + 3) * (x - 2) < 0) ∧ q = (-3 < x ∧ x < 0) := by
  sorry

end inequality_condition_l149_149067


namespace bridge_length_l149_149414

noncomputable def length_of_bridge (length_of_train : ℕ) (speed_of_train_kmh : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_of_train_ms := (speed_of_train_kmh * 1000) / 3600
  let total_distance := speed_of_train_ms * time_seconds
  total_distance - length_of_train

theorem bridge_length (length_of_train : ℕ) (speed_of_train_kmh : ℕ) (time_seconds : ℕ) (h1 : length_of_train = 170) (h2 : speed_of_train_kmh = 45) (h3 : time_seconds = 30) :
  length_of_bridge length_of_train speed_of_train_kmh time_seconds = 205 :=
by 
  rw [h1, h2, h3]
  unfold length_of_bridge
  simp
  sorry

end bridge_length_l149_149414


namespace evaluate_expression_l149_149663

theorem evaluate_expression (x : ℤ) (h : x = 4) : 3 * x + 5 = 17 :=
by
  sorry

end evaluate_expression_l149_149663


namespace fraction_eq_zero_has_solution_l149_149205

theorem fraction_eq_zero_has_solution :
  ∀ (x : ℝ), x^2 - x - 2 = 0 ∧ x + 1 ≠ 0 → x = 2 :=
by
  sorry

end fraction_eq_zero_has_solution_l149_149205


namespace number_of_girls_joined_l149_149258

-- Define the initial conditions
def initial_girls := 18
def initial_boys := 15
def boys_quit := 4
def total_children_after_changes := 36

-- Define the changes
def boys_after_quit := initial_boys - boys_quit
def girls_after_changes := total_children_after_changes - boys_after_quit
def girls_joined := girls_after_changes - initial_girls

-- State the theorem
theorem number_of_girls_joined :
  girls_joined = 7 :=
by
  sorry

end number_of_girls_joined_l149_149258


namespace expression_simplify_l149_149907

theorem expression_simplify
  (a b : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : b ≠ 0)
  (h₃ : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ((3 * a)⁻¹ - (b / 3)⁻¹) = - (1 / (a * b)) :=
by
  sorry

end expression_simplify_l149_149907


namespace inequality_abc_d_l149_149051

theorem inequality_abc_d (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (H1 : d ≥ a) (H2 : d ≥ b) (H3 : d ≥ c) : a * (d - b) + b * (d - c) + c * (d - a) ≤ d^2 :=
by
  sorry

end inequality_abc_d_l149_149051


namespace max_even_integers_for_odd_product_l149_149281

theorem max_even_integers_for_odd_product (a b c d e f g : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : 0 < f) (h7 : 0 < g) 
  (h_prod_odd : a * b * c * d * e * f * g % 2 = 1) : a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ e % 2 = 1 ∧ f % 2 = 1 ∧ g % 2 = 1 :=
sorry

end max_even_integers_for_odd_product_l149_149281


namespace solve_for_x_l149_149336

theorem solve_for_x (x : ℚ) (h : (3 * x + 5) / 7 = 13) : x = 86 / 3 :=
sorry

end solve_for_x_l149_149336


namespace boys_running_speed_l149_149422
-- Import the necessary libraries

-- Define the input conditions:
def side_length : ℝ := 50
def time_seconds : ℝ := 80
def conversion_factor_meters_to_kilometers : ℝ := 1000
def conversion_factor_seconds_to_hours : ℝ := 3600

-- Define the theorem:
theorem boys_running_speed :
  let perimeter := 4 * side_length
  let distance_kilometers := perimeter / conversion_factor_meters_to_kilometers
  let time_hours := time_seconds / conversion_factor_seconds_to_hours
  distance_kilometers / time_hours = 9 :=
by
  sorry

end boys_running_speed_l149_149422


namespace student_B_speed_l149_149773

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l149_149773


namespace least_value_of_g_l149_149448

noncomputable def g (x : ℝ) : ℝ := 3 * x ^ 2 + 6 * x + 1

theorem least_value_of_g : ∃ x : ℝ, ∀ y : ℝ, g y ≥ g x ∧ g x = -2 := by
  sorry

end least_value_of_g_l149_149448


namespace person_speed_l149_149155

namespace EscalatorProblem

/-- The speed of the person v_p walking on the moving escalator is 3 ft/sec given the conditions -/
theorem person_speed (v_p : ℝ) 
  (escalator_speed : ℝ := 12) 
  (escalator_length : ℝ := 150) 
  (time_taken : ℝ := 10) :
  escalator_length = (v_p + escalator_speed) * time_taken → v_p = 3 := 
by sorry

end EscalatorProblem

end person_speed_l149_149155


namespace find_speed_of_B_l149_149800

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l149_149800


namespace remainder_97_pow_103_mul_7_mod_17_l149_149262

theorem remainder_97_pow_103_mul_7_mod_17 :
  (97 ^ 103 * 7) % 17 = 13 := by
  have h1 : 97 % 17 = -3 % 17 := by sorry
  have h2 : 9 % 17 = -8 % 17 := by sorry
  have h3 : 64 % 17 = 13 % 17 := by sorry
  have h4 : -21 % 17 = 13 % 17 := by sorry
  sorry

end remainder_97_pow_103_mul_7_mod_17_l149_149262


namespace cauliflower_sales_l149_149695

namespace WeeklyMarket

def broccoliPrice := 3
def totalEarnings := 520
def broccolisSold := 19

def carrotPrice := 2
def spinachPrice := 4
def spinachWeight := 8 -- This is derived from solving $4S = 2S + $16 

def broccoliEarnings := broccolisSold * broccoliPrice
def carrotEarnings := spinachWeight * carrotPrice -- This is twice copied

def spinachEarnings : ℕ := spinachWeight * spinachPrice
def tomatoEarnings := broccoliEarnings + spinachEarnings

def otherEarnings : ℕ := broccoliEarnings + carrotEarnings + spinachEarnings + tomatoEarnings

def cauliflowerEarnings : ℕ := totalEarnings - otherEarnings -- This directly from subtraction of earnings

theorem cauliflower_sales : cauliflowerEarnings = 310 :=
  by
    -- only the statement part, no actual proof needed
    sorry

end WeeklyMarket

end cauliflower_sales_l149_149695


namespace plywood_cut_difference_l149_149112

theorem plywood_cut_difference :
  let original_width := 6
  let original_height := 9
  let total_area := original_width * original_height
  let num_pieces := 6
  let area_per_piece := total_area / num_pieces
  -- Let possible perimeters based on given conditions
  let max_perimeter := 20
  let min_perimeter := 15
  max_perimeter - min_perimeter = 5 :=
by
  sorry

end plywood_cut_difference_l149_149112


namespace sequence_a_2024_l149_149953

theorem sequence_a_2024 (a : ℕ → ℝ) (h₀ : a 1 = 2) (h₁ : ∀ n : ℕ, n > 0 → a (n + 1) = 1 - 1 / a n) : a 2024 = 1 / 2 :=
by
  sorry

end sequence_a_2024_l149_149953


namespace non_congruent_non_square_rectangles_count_l149_149278

theorem non_congruent_non_square_rectangles_count :
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (x : ℕ × ℕ), x ∈ S → 2 * (x.1 + x.2) = 80) ∧
    S.card = 19 ∧
    (∀ (x : ℕ × ℕ), x ∈ S → x.1 ≠ x.2) ∧
    (∀ (x y : ℕ × ℕ), x ∈ S → y ∈ S → x ≠ y → x.1 = y.2 → x.2 = y.1) :=
sorry

end non_congruent_non_square_rectangles_count_l149_149278


namespace fractional_sides_l149_149499

variable {F : ℕ} -- Number of fractional sides
variable {D : ℕ} -- Number of diagonals

theorem fractional_sides (h1 : D = 2 * F) (h2 : D = F * (F - 3) / 2) : F = 7 :=
by
  sorry

end fractional_sides_l149_149499


namespace randolph_age_l149_149055

theorem randolph_age (R Sy S : ℕ) 
  (h1 : R = Sy + 5) 
  (h2 : Sy = 2 * S) 
  (h3 : S = 25) : 
  R = 55 :=
by 
  sorry

end randolph_age_l149_149055


namespace option_d_correct_factorization_l149_149533

theorem option_d_correct_factorization (x : ℝ) : 
  -8 * x ^ 2 + 8 * x - 2 = -2 * (2 * x - 1) ^ 2 :=
by 
  sorry

end option_d_correct_factorization_l149_149533


namespace metallic_sheet_width_l149_149573

-- Defining the conditions
def sheet_length := 48
def cut_square_side := 8
def box_volume := 5632

-- Main theorem statement
theorem metallic_sheet_width 
    (L : ℕ := sheet_length)
    (s : ℕ := cut_square_side)
    (V : ℕ := box_volume) :
    (32 * (w - 2 * s) * s = V) → (w = 38) := by
  intros h1
  sorry

end metallic_sheet_width_l149_149573


namespace food_duration_l149_149037

theorem food_duration (mom_meals_per_day : ℕ) (mom_cups_per_meal : ℚ)
                      (puppy_count : ℕ) (puppy_meals_per_day : ℕ) (puppy_cups_per_meal : ℚ)
                      (total_food : ℚ)
                      (H_mom : mom_meals_per_day = 3) 
                      (H_mom_cups : mom_cups_per_meal = 3/2)
                      (H_puppies : puppy_count = 5) 
                      (H_puppy_meals : puppy_meals_per_day = 2) 
                      (H_puppy_cups : puppy_cups_per_meal = 1/2) 
                      (H_total_food : total_food = 57) : 
  (total_food / ((mom_meals_per_day * mom_cups_per_meal) + (puppy_count * puppy_meals_per_day * puppy_cups_per_meal))) = 6 := 
by
  sorry

end food_duration_l149_149037


namespace solve_congruence_l149_149239

theorem solve_congruence (n : ℤ) : 15 * n ≡ 9 [ZMOD 47] → n ≡ 18 [ZMOD 47] :=
by
  sorry

end solve_congruence_l149_149239


namespace cost_of_25kg_l149_149578

-- Definitions and conditions
def price_33kg (l q : ℕ) : Prop := 30 * l + 3 * q = 360
def price_36kg (l q : ℕ) : Prop := 30 * l + 6 * q = 420

-- Theorem statement
theorem cost_of_25kg (l q : ℕ) (h1 : 30 * l + 3 * q = 360) (h2 : 30 * l + 6 * q = 420) : 25 * l = 250 :=
by
  sorry

end cost_of_25kg_l149_149578


namespace triangle_area_l149_149513

theorem triangle_area
  (a b : ℝ)
  (C : ℝ)
  (h₁ : a = 2)
  (h₂ : b = 3)
  (h₃ : C = π / 3) :
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_area_l149_149513


namespace student_B_speed_l149_149809

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l149_149809


namespace julia_internet_speed_l149_149682

theorem julia_internet_speed
  (songs : ℕ) (song_size : ℕ) (time_sec : ℕ)
  (h_songs : songs = 7200)
  (h_song_size : song_size = 5)
  (h_time_sec : time_sec = 1800) :
  songs * song_size / time_sec = 20 := by
  sorry

end julia_internet_speed_l149_149682


namespace amount_of_bill_correct_l149_149724

noncomputable def TD : ℝ := 360
noncomputable def BD : ℝ := 421.7142857142857
noncomputable def computeFV (TD BD : ℝ) := (TD * BD) / (BD - TD)

theorem amount_of_bill_correct :
  computeFV TD BD = 2460 := 
sorry

end amount_of_bill_correct_l149_149724


namespace find_positive_real_solution_l149_149305

theorem find_positive_real_solution (x : ℝ) (h₁ : 0 < x) (h₂ : 1/2 * (4 * x ^ 2 - 4) = (x ^ 2 - 40 * x - 8) * (x ^ 2 + 20 * x + 4)) :
  x = 20 + Real.sqrt 410 :=
by
  sorry

end find_positive_real_solution_l149_149305


namespace bicycle_speed_B_l149_149765

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l149_149765


namespace cosine_third_angle_of_triangle_l149_149920

theorem cosine_third_angle_of_triangle (X Y Z : ℝ)
  (sinX_eq : Real.sin X = 4/5)
  (cosY_eq : Real.cos Y = 12/13)
  (triangle_sum : X + Y + Z = Real.pi) :
  Real.cos Z = -16/65 :=
by
  -- proof will be filled in
  sorry

end cosine_third_angle_of_triangle_l149_149920


namespace simplify_sqrt_sum_l149_149238

noncomputable def sqrt_72 : ℝ := Real.sqrt 72
noncomputable def sqrt_32 : ℝ := Real.sqrt 32
noncomputable def sqrt_27 : ℝ := Real.sqrt 27
noncomputable def result : ℝ := 10 * Real.sqrt 2 + 3 * Real.sqrt 3

theorem simplify_sqrt_sum :
  sqrt_72 + sqrt_32 + sqrt_27 = result :=
by
  sorry

end simplify_sqrt_sum_l149_149238


namespace osmanthus_trees_variance_l149_149349

variable (n : Nat) (p : ℚ)

def variance_binomial_distribution (n : Nat) (p : ℚ) : ℚ :=
  n * p * (1 - p)

theorem osmanthus_trees_variance (n : Nat) (p : ℚ) (h₁ : n = 4) (h₂ : p = 4 / 5) :
  variance_binomial_distribution n p = 16 / 25 := by
  sorry

end osmanthus_trees_variance_l149_149349


namespace domain_of_f_l149_149297

open Real

noncomputable def f (x : ℝ) : ℝ := log (log x)

theorem domain_of_f : { x : ℝ | 1 < x } = { x : ℝ | ∃ y > 1, x = y } :=
by
  sorry

end domain_of_f_l149_149297


namespace nth_row_equation_l149_149521

theorem nth_row_equation (n : ℕ) : 2 * n + 1 = (n + 1) ^ 2 - n ^ 2 := 
sorry

end nth_row_equation_l149_149521


namespace number_of_packages_sold_l149_149437

noncomputable def supplier_charges (P : ℕ) : ℕ :=
  if P ≤ 10 then 25 * P
  else 250 + 20 * (P - 10)

theorem number_of_packages_sold
  (supplier_received : ℕ)
  (percent_to_X : ℕ)
  (percent_to_Y : ℕ)
  (percent_to_Z : ℕ)
  (per_package_price : ℕ)
  (discount_percent : ℕ)
  (discount_threshold : ℕ)
  (P : ℕ)
  (h_received : supplier_received = 1340)
  (h_to_X : percent_to_X = 15)
  (h_to_Y : percent_to_Y = 15)
  (h_to_Z : percent_to_Z = 70)
  (h_full_price : per_package_price = 25)
  (h_discount : discount_percent = 4 * per_package_price / 5)
  (h_threshold : discount_threshold = 10)
  (h_calculation : supplier_charges P = supplier_received) : P = 65 := 
sorry

end number_of_packages_sold_l149_149437


namespace surface_area_of_cube_l149_149917

-- Definition of the problem in Lean 4
theorem surface_area_of_cube (a : ℝ) (s : ℝ) (h : s * Real.sqrt 3 = a) : 6 * (s^2) = 2 * a^2 :=
by
  sorry

end surface_area_of_cube_l149_149917


namespace student_b_speed_l149_149821

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l149_149821


namespace find_amplitude_l149_149287

theorem find_amplitude (A D : ℝ) (h1 : D + A = 5) (h2 : D - A = -3) : A = 4 :=
by
  sorry

end find_amplitude_l149_149287


namespace dealer_gross_profit_l149_149982

theorem dealer_gross_profit
  (purchase_price : ℝ)
  (markup_rate : ℝ)
  (discount_rate : ℝ)
  (initial_selling_price : ℝ)
  (final_selling_price : ℝ)
  (gross_profit : ℝ)
  (h0 : purchase_price = 150)
  (h1 : markup_rate = 0.5)
  (h2 : discount_rate = 0.2)
  (h3 : initial_selling_price = purchase_price + markup_rate * initial_selling_price)
  (h4 : final_selling_price = initial_selling_price - discount_rate * initial_selling_price)
  (h5 : gross_profit = final_selling_price - purchase_price) :
  gross_profit = 90 :=
sorry

end dealer_gross_profit_l149_149982


namespace product_of_roots_l149_149359

noncomputable def Q : Polynomial ℚ := Polynomial.Cubic 1 0 -6 -12

theorem product_of_roots : Polynomial.root_product Q = 12 :=
by sorry

end product_of_roots_l149_149359


namespace plywood_perimeter_difference_l149_149133

/--
Given a 6-foot by 9-foot rectangular piece of plywood cut into 6 congruent rectangles 
with no wood left over and no wood lost due to the cuts,
prove that the positive difference between the greatest and the least perimeter of a single piece is 11 feet.
-/
theorem plywood_perimeter_difference :
  ∃ (rectangles : List (ℕ × ℕ)), 
  (∀ r ∈ rectangles, r.fst * r.snd = 9 * 6 / 6) ∧
  (Greatest (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) - 
  (Least (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) = 11 :=
by
  sorry

end plywood_perimeter_difference_l149_149133


namespace blue_shoes_in_warehouse_l149_149254

theorem blue_shoes_in_warehouse (total blue purple green : ℕ) (h1 : total = 1250) (h2 : green = purple) (h3 : purple = 355) :
    blue = total - (green + purple) := by
  sorry

end blue_shoes_in_warehouse_l149_149254


namespace speed_of_student_B_l149_149786

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l149_149786


namespace student_B_speed_l149_149799

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l149_149799


namespace pie_split_l149_149156

theorem pie_split (initial_pie : ℚ) (number_of_people : ℕ) (amount_taken_by_each : ℚ) 
  (h1 : initial_pie = 5/6) (h2 : number_of_people = 4) : amount_taken_by_each = 5/24 :=
by
  sorry

end pie_split_l149_149156


namespace find_numbers_l149_149065

theorem find_numbers (x y z : ℝ) 
  (h1 : x = 280)
  (h2 : y = 200)
  (h3 : z = 220) :
  (x = 1.4 * y) ∧
  (x / z = 14 / 11) ∧
  (z - y = 0.125 * (x + y) - 40) :=
by
  sorry

end find_numbers_l149_149065


namespace g_neg_2_eq_3_l149_149200

def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

theorem g_neg_2_eq_3 : g (-2) = 3 :=
by
  sorry

end g_neg_2_eq_3_l149_149200


namespace smallest_sum_of_pairwise_distinct_squares_l149_149927

theorem smallest_sum_of_pairwise_distinct_squares :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  ∃ x y z : ℕ, a + b = x^2 ∧ b + c = z^2 ∧ c + a = y^2 ∧ a + b + c = 55 :=
sorry

end smallest_sum_of_pairwise_distinct_squares_l149_149927


namespace find_polynomial_q_l149_149387

theorem find_polynomial_q (q : ℝ → ℝ) :
  (∀ x : ℝ, q x + (x^6 + 4*x^4 + 8*x^2 + 7*x) = (12*x^4 + 30*x^3 + 40*x^2 + 10*x + 2)) →
  (∀ x : ℝ, q x = -x^6 + 8*x^4 + 30*x^3 + 32*x^2 + 3*x + 2) :=
by 
  sorry

end find_polynomial_q_l149_149387


namespace initial_dimes_l149_149367

theorem initial_dimes (dimes_received_from_dad : ℕ) (dimes_received_from_mom : ℕ) (total_dimes_now : ℕ) : 
  dimes_received_from_dad = 8 → dimes_received_from_mom = 4 → total_dimes_now = 19 → 
  total_dimes_now - (dimes_received_from_dad + dimes_received_from_mom) = 7 :=
by
  intros
  sorry

end initial_dimes_l149_149367


namespace total_flags_l149_149169

theorem total_flags (x : ℕ) (hx1 : 4 * x + 20 > 8 * (x - 1)) (hx2 : 4 * x + 20 < 8 * x) : 4 * 6 + 20 = 44 :=
by sorry

end total_flags_l149_149169


namespace average_of_first_13_even_numbers_l149_149743

-- Definition of the first 13 even numbers
def first_13_even_numbers := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]

-- The sum of the first 13 even numbers
def sum_of_first_13_even_numbers : ℕ := 182

-- The number of these even numbers
def number_of_even_numbers : ℕ := 13

-- The average of the first 13 even numbers
theorem average_of_first_13_even_numbers : (sum_of_first_13_even_numbers / number_of_even_numbers) = 14 := by
  sorry

end average_of_first_13_even_numbers_l149_149743


namespace Alan_has_eight_pine_trees_l149_149152

noncomputable def number_of_pine_trees (total_pine_cones_per_tree : ℕ) (percentage_on_roof : ℚ) 
                                       (weight_per_pine_cone : ℚ) (total_weight_on_roof : ℚ) : ℚ :=
  total_weight_on_roof / (total_pine_cones_per_tree * percentage_on_roof * weight_per_pine_cone)

theorem Alan_has_eight_pine_trees :
  number_of_pine_trees 200 (30 / 100) 4 1920 = 8 :=
by
  sorry

end Alan_has_eight_pine_trees_l149_149152


namespace student_B_speed_l149_149779

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l149_149779


namespace maximum_illuminated_surfaces_l149_149144

noncomputable def optimal_position (r R d : ℝ) (h : d > r + R) : ℝ :=
  d / (1 + Real.sqrt (R^3 / r^3))

theorem maximum_illuminated_surfaces (r R d : ℝ) (h : d > r + R) (h1 : r ≤ optimal_position r R d h) (h2 : optimal_position r R d h ≤ d - R) :
  (optimal_position r R d h = d / (1 + Real.sqrt (R^3 / r^3))) ∨ (optimal_position r R d h = r) :=
sorry

end maximum_illuminated_surfaces_l149_149144


namespace inverse_five_eq_two_l149_149322

-- Define the function f(x) = x^2 + 1 for x >= 0
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the condition x >= 0
def nonneg (x : ℝ) : Prop := x ≥ 0

-- State the problem: proving that the inverse function f⁻¹(5) = 2
theorem inverse_five_eq_two : ∃ x : ℝ, nonneg x ∧ f x = 5 ∧ x = 2 :=
by
  sorry

end inverse_five_eq_two_l149_149322


namespace tom_seashells_now_l149_149959

def original_seashells : ℕ := 5
def given_seashells : ℕ := 2

theorem tom_seashells_now : original_seashells - given_seashells = 3 :=
by
  sorry

end tom_seashells_now_l149_149959


namespace sum_of_integers_product_neg17_l149_149251

theorem sum_of_integers_product_neg17 (a b c : ℤ) (h : a * b * c = -17) : a + b + c = -15 ∨ a + b + c = 17 :=
sorry

end sum_of_integers_product_neg17_l149_149251


namespace find_y_l149_149010

theorem find_y (x y : ℤ) (h1 : x + 2 * y = 100) (h2 : x = 50) : y = 25 :=
by
  sorry

end find_y_l149_149010


namespace correct_answer_is_B_l149_149866

-- Define what it means to be a quadratic equation in one variable
def is_quadratic_in_one_variable (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ (a * x ^ 2 + b * x + c = 0)

-- Conditions:
def eqA (x : ℝ) : Prop := 2 * x + 1 = 0
def eqB (x : ℝ) : Prop := x ^ 2 + 1 = 0
def eqC (x y : ℝ) : Prop := y ^ 2 + x = 1
def eqD (x : ℝ) : Prop := 1 / x + x ^ 2 = 1

-- Theorem statement: Prove which equation is a quadratic equation in one variable
theorem correct_answer_is_B : is_quadratic_in_one_variable eqB :=
sorry  -- Proof is not required as per the instructions

end correct_answer_is_B_l149_149866


namespace current_price_of_soda_l149_149995

theorem current_price_of_soda (C S : ℝ) (h1 : 1.25 * C = 15) (h2 : C + S = 16) : 1.5 * S = 6 :=
by
  sorry

end current_price_of_soda_l149_149995


namespace factorize_square_difference_l149_149303

theorem factorize_square_difference (x: ℝ):
  x^2 - 4 = (x + 2) * (x - 2) := by
  -- Using the difference of squares formula a^2 - b^2 = (a + b)(a - b)
  sorry

end factorize_square_difference_l149_149303


namespace bell_rings_before_geography_l149_149038

def number_of_bell_rings : Nat :=
  let assembly_start := 1
  let assembly_end := 1
  let maths_start := 1
  let maths_end := 1
  let history_start := 1
  let history_end := 1
  let quiz_start := 1
  let quiz_end := 1
  let geography_start := 1
  assembly_start + assembly_end + maths_start + maths_end + 
  history_start + history_end + quiz_start + quiz_end + 
  geography_start

theorem bell_rings_before_geography : number_of_bell_rings = 9 := 
by
  -- Proof omitted
  sorry

end bell_rings_before_geography_l149_149038


namespace find_y_value_l149_149886

theorem find_y_value (y : ℝ) (h : 1 / (3 + 1 / (3 + 1 / (3 - y))) = 0.30337078651685395) : y = 0.3 :=
sorry

end find_y_value_l149_149886


namespace cos_C_l149_149679

-- Define the data and conditions of the problem
variables {A B C : ℝ}
variables (triangle_ABC : Prop)
variable (h_sinA : Real.sin A = 4 / 5)
variable (h_cosB : Real.cos B = 12 / 13)

-- Statement of the theorem
theorem cos_C (h1 : triangle_ABC)
  (h2 : Real.sin A = 4 / 5)
  (h3 : Real.cos B = 12 / 13) :
  Real.cos C = -16 / 65 :=
sorry

end cos_C_l149_149679


namespace hyperbola_through_C_l149_149919

noncomputable def equation_of_hyperbola_passing_through_C : Prop :=
  let A := (-1/2, 1/4)
  let B := (2, 4)
  let C := (-1/2, 4)
  ∃ (k : ℝ), k = -2 ∧ (∀ x : ℝ, x ≠ 0 → x * (4) = k)

theorem hyperbola_through_C :
  equation_of_hyperbola_passing_through_C :=
by
  sorry

end hyperbola_through_C_l149_149919


namespace find_x_l149_149296

def diamond (x y : ℤ) : ℤ := 3 * x - y^2

theorem find_x (x : ℤ) (h : diamond x 7 = 20) : x = 23 :=
sorry

end find_x_l149_149296


namespace remainder_17_pow_63_mod_7_l149_149550

theorem remainder_17_pow_63_mod_7 : (17^63) % 7 = 6 := 
by
  sorry

end remainder_17_pow_63_mod_7_l149_149550


namespace find_m_l149_149902

noncomputable def vector_a : ℝ × ℝ := (1, -3)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, 2)
noncomputable def vector_sum (m : ℝ) : ℝ × ℝ := (1 + m, -1)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem find_m (m : ℝ) : dot_product vector_a (vector_sum m) = 0 → m = -4 :=
by
  sorry

end find_m_l149_149902


namespace cuboid_properties_l149_149723

-- Given definitions from conditions
variables (l w h : ℝ)
variables (h_edge_length : 4 * (l + w + h) = 72)
variables (h_ratio : l / w = 3 / 2 ∧ w / h = 2 / 1)

-- Define the surface area and volume based on the given conditions
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)
def volume (l w h : ℝ) : ℝ := l * w * h

-- Theorem statement
theorem cuboid_properties :
  surface_area l w h = 198 ∧ volume l w h = 162 :=
by
  -- Code to provide the proof goes here
  sorry

end cuboid_properties_l149_149723


namespace proper_sampling_method_l149_149256

-- Definitions for conditions
def large_bulbs : ℕ := 120
def medium_bulbs : ℕ := 60
def small_bulbs : ℕ := 20
def sample_size : ℕ := 25

-- Definition for the proper sampling method to use
def sampling_method : String := "Stratified sampling"

-- Theorem statement to prove the sampling method
theorem proper_sampling_method :
  ∃ method : String, 
  method = sampling_method ∧
  sampling_method = "Stratified sampling" := by
    sorry

end proper_sampling_method_l149_149256


namespace canoe_upstream_speed_l149_149274

theorem canoe_upstream_speed (C : ℝ) (stream_speed downstream_speed : ℝ) 
  (h_stream : stream_speed = 2) (h_downstream : downstream_speed = 12) 
  (h_equation : C + stream_speed = downstream_speed) :
  C - stream_speed = 8 := 
by 
  sorry

end canoe_upstream_speed_l149_149274


namespace equilateral_triangle_circumradius_ratio_l149_149729

variables (B b S s : ℝ)

-- Given two equilateral triangles with side lengths B and b, and respectively circumradii S and s
-- B and b are not equal
-- Prove that S / s = B / b
theorem equilateral_triangle_circumradius_ratio (hBneqb : B ≠ b)
  (hS : S = B * Real.sqrt 3 / 3)
  (hs : s = b * Real.sqrt 3 / 3) : S / s = B / b :=
by
  sorry

end equilateral_triangle_circumradius_ratio_l149_149729


namespace chinese_number_representation_l149_149509

theorem chinese_number_representation :
  ∀ (祝 贺 华 杯 赛 : ℕ),
  祝 = 4 → 贺 = 8 → 
  华 ≠ 杯 ∧ 华 ≠ 赛 ∧ 杯 ≠ 赛 ∧ 华 ≠ 祝 ∧ 华 ≠ 贺 ∧ 杯 ≠ 祝 ∧ 杯 ≠ 贺 ∧ 赛 ≠ 祝 ∧ 赛 ≠ 贺 → 
  华 ≥ 1 ∧ 华 ≤ 9 → 杯 ≥ 1 ∧ 杯 ≤ 9 → 赛 ≥ 1 ∧ 赛 ≤ 9 → 
  华 * 100 + 杯 * 10 + 赛 = 7632 :=
begin
  sorry
end

end chinese_number_representation_l149_149509


namespace train_passing_time_l149_149971

-- conditions
def train_length := 490 -- in meters
def train_speed_kmh := 63 -- in kilometers per hour
def conversion_factor := 1000 / 3600 -- to convert km/hr to m/s

-- conversion
def train_speed_ms := train_speed_kmh * conversion_factor -- speed in meters per second

-- expected correct answer
def expected_time := 28 -- in seconds

-- Theorem statement
theorem train_passing_time :
  train_length / train_speed_ms = expected_time :=
by
  sorry

end train_passing_time_l149_149971


namespace only_pair_2_2_satisfies_l149_149458

theorem only_pair_2_2_satisfies :
  ∀ a b : ℕ, (∀ n : ℕ, ∃ c : ℕ, a ^ n + b ^ n = c ^ (n + 1)) → (a = 2 ∧ b = 2) :=
by sorry

end only_pair_2_2_satisfies_l149_149458


namespace solve_equation_l149_149746

theorem solve_equation {x : ℝ} (hx : x = 1) : 9 - 3 / x / 3 + 3 = 3 := by
  rw [hx] -- Substitute x = 1
  norm_num -- Simplify the numerical expression
  sorry -- to be proved

end solve_equation_l149_149746


namespace men_per_table_correct_l149_149994

def tables := 6
def women_per_table := 3
def total_customers := 48
def total_women := women_per_table * tables
def total_men := total_customers - total_women
def men_per_table := total_men / tables

theorem men_per_table_correct : men_per_table = 5 := by
  sorry

end men_per_table_correct_l149_149994


namespace intersection_of_A_and_B_l149_149486

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := { y | ∃ x ∈ A, y = x + 1 }

theorem intersection_of_A_and_B :
  A ∩ B = {2, 3, 4} :=
sorry

end intersection_of_A_and_B_l149_149486


namespace arccos_sin_three_l149_149871

theorem arccos_sin_three : Real.arccos (Real.sin 3) = 3 - Real.pi / 2 :=
by
  sorry

end arccos_sin_three_l149_149871


namespace perimeter_difference_l149_149126

-- Definitions as per conditions
def plywood_width : ℕ := 6
def plywood_height : ℕ := 9
def rectangles_count : ℕ := 6

-- The perimeter difference to be proved
theorem perimeter_difference : 
  ∃ (max_perimeter min_perimeter : ℕ), 
  max_perimeter = 22 ∧ min_perimeter = 12 ∧ (max_perimeter - min_perimeter = 10) :=
by
  sorry

end perimeter_difference_l149_149126


namespace value_f2_f5_l149_149342

variable {α : Type} [AddGroup α]

noncomputable def f : α → ℤ := sorry

axiom func_eq : ∀ x y, f (x + y) = f x + f y + 7 * x * y + 4

axiom f_one : f 1 = 4

theorem value_f2_f5 :
  f 2 + f 5 = 125 :=
sorry

end value_f2_f5_l149_149342


namespace hundredth_digit_l149_149963

theorem hundredth_digit (n : ℕ) (h : n = 100) :
  let recurring_seq := 03
  let digit := recurring_seq[(n - 1) % recurring_seq.length]
  \(\frac{21}{700}\) = 0.\overline{03} → digit = '3' :=
by sorry

end hundredth_digit_l149_149963


namespace expand_and_simplify_l149_149299

theorem expand_and_simplify :
  ∀ (x : ℝ), 5 * (6 * x^3 - 3 * x^2 + 4 * x - 2) = 30 * x^3 - 15 * x^2 + 20 * x - 10 :=
by
  intro x
  sorry

end expand_and_simplify_l149_149299


namespace brick_length_correct_l149_149571

-- Define the constants
def courtyard_length_meters : ℝ := 25
def courtyard_width_meters : ℝ := 18
def courtyard_area_meters : ℝ := courtyard_length_meters * courtyard_width_meters
def bricks_number : ℕ := 22500
def brick_width_cm : ℕ := 10

-- We want to prove the length of each brick
def brick_length_cm : ℕ := 20

-- Convert courtyard area to square centimeters
def courtyard_area_cm : ℝ := courtyard_area_meters * 10000

-- Define the proof statement
theorem brick_length_correct :
  courtyard_area_cm = (brick_length_cm * brick_width_cm) * bricks_number :=
by
  sorry

end brick_length_correct_l149_149571


namespace student_b_speed_l149_149850

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l149_149850


namespace area_under_f_l149_149650

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3 * x

noncomputable def f' (x : ℝ) : ℝ := 1 / x + 2 * x - 3

theorem area_under_f' : 
  - ∫ x in (1/2 : ℝ)..1, f' x = (3 / 4) - Real.log 2 := 
by
  sorry

end area_under_f_l149_149650


namespace plywood_cut_perimeter_difference_l149_149119

theorem plywood_cut_perimeter_difference :
  (∃ (l w : ℕ), (l * w = 54) ∧ (9 % w = 0) ∧ (6 % l = 0) ∧ (6 / l) * (9 / w) = 6) →
  10 =
  let p := λ l w, 2 * (l + w) in
  let perimeters := [
    p 1 9,
    p 1 6,
    p 2 3,
    p 3 2
  ]
  in (list.max precedence perimeters - list.min precedence perimeters) :=
begin
  sorry
end

end plywood_cut_perimeter_difference_l149_149119


namespace intersection_A_B_l149_149685

def A : Set ℝ := { x | ∃ y, y = Real.sqrt (x^2 - 2*x - 3) }
def B : Set ℝ := { x | ∃ y, y = Real.log x }

theorem intersection_A_B : A ∩ B = {x | x ∈ Set.Ici 3} :=
by
  sorry

end intersection_A_B_l149_149685


namespace first_player_wins_l149_149034

-- Define the set of points S
def S : Set (ℤ × ℤ) := { p | ∃ x y : ℤ, p = (x, y) ∧ x^2 + y^2 ≤ 1010 }

-- Define the game properties and conditions
def game_property :=
  ∀ (p : ℤ × ℤ), p ∈ S →
  ∀ (q : ℤ × ℤ), q ∈ S →
  p ≠ q →
  -- Forbidden to move to a point symmetric to the current one relative to the origin
  q ≠ (-p.fst, -p.snd) →
  -- Distances of moves must strictly increase
  dist p q > dist q (q.fst, q.snd)

-- The first player always guarantees a win
theorem first_player_wins : game_property → true :=
by
  sorry

end first_player_wins_l149_149034


namespace perimeters_positive_difference_l149_149121

theorem perimeters_positive_difference (orig_length orig_width : ℝ) (num_pieces : ℕ)
  (congruent_division : ∃ (length width : ℝ), length * width = (orig_length * orig_width) / num_pieces)
  (greatest_perimeter least_perimeter : ℝ)
  (h1 : greatest_perimeter = 2 * (1.5 + 9))
  (h2 : least_perimeter = 2 * (1 + 6)) :
  abs (greatest_perimeter - least_perimeter) = 7 := 
sorry

end perimeters_positive_difference_l149_149121


namespace num_solutions_20_l149_149231

def num_solutions (n : ℕ) : ℕ :=
  4 * n

theorem num_solutions_20 : num_solutions 20 = 80 := by
  sorry

end num_solutions_20_l149_149231


namespace calculate_expression_l149_149660

variable (y : ℝ) (π : ℝ) (Q : ℝ)

theorem calculate_expression (h : 5 * (3 * y - 7 * π) = Q) : 
  10 * (6 * y - 14 * π) = 4 * Q := by
  sorry

end calculate_expression_l149_149660


namespace find_x_l149_149203

theorem find_x (x : ℕ) : (x % 6 = 0) ∧ (x^2 > 200) ∧ (x < 30) → (x = 18 ∨ x = 24) :=
by
  intros
  sorry

end find_x_l149_149203


namespace student_B_speed_l149_149775

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l149_149775


namespace dark_squares_exceed_light_squares_by_one_l149_149424

theorem dark_squares_exceed_light_squares_by_one 
  (m n : ℕ) (h_m : m = 9) (h_n : n = 9) (h_total_squares : m * n = 81) :
  let dark_squares := 5 * 5 + 4 * 4
  let light_squares := 5 * 4 + 4 * 5
  dark_squares - light_squares = 1 :=
by {
  sorry
}

end dark_squares_exceed_light_squares_by_one_l149_149424


namespace speed_of_student_B_l149_149830

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l149_149830


namespace find_x_l149_149661

theorem find_x 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (dot_product : ℝ)
  (ha : a = (1, 2)) 
  (hb : b = (x, 3)) 
  (hdot : a.1 * b.1 + a.2 * b.2 = dot_product) 
  (hdot_val : dot_product = 4) : 
  x = -2 :=
by 
  sorry

end find_x_l149_149661


namespace cole_time_to_work_is_90_minutes_l149_149265

noncomputable def cole_drive_time_to_work (D : ℝ) : ℝ := D / 30

def cole_trip_proof : Prop :=
  ∃ (D : ℝ), (D / 30) + (D / 90) = 2 ∧ cole_drive_time_to_work D * 60 = 90

theorem cole_time_to_work_is_90_minutes : cole_trip_proof :=
  sorry

end cole_time_to_work_is_90_minutes_l149_149265


namespace domain_and_monotone_l149_149009

noncomputable def f (x : ℝ) : ℝ := (1 + x^2) / (1 - x^2)

theorem domain_and_monotone :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → ∃ y, f x = y) ∧
  ∀ x1 x2 : ℝ, 1 < x1 ∧ x1 < x2 → f x1 < f x2 :=
by
  sorry

end domain_and_monotone_l149_149009


namespace rationalize_denominator_l149_149372

theorem rationalize_denominator :
  (∃ x y z w : ℂ, x = sqrt 12 ∧ y = sqrt 5 ∧
  z = sqrt 3 ∧ w = sqrt 5 ∧
  (x + y) / (z + w) = (sqrt 15 - 1) / 2) :=
by
  use [√12, √5, √3, √5]
  sorry

end rationalize_denominator_l149_149372


namespace range_neg_square_l149_149952

theorem range_neg_square (x : ℝ) (hx : -3 ≤ x ∧ x ≤ 1) : 
  -9 ≤ -x^2 ∧ -x^2 ≤ 0 :=
sorry

end range_neg_square_l149_149952


namespace total_students_l149_149432

theorem total_students (S : ℕ) (h1 : S / 2 / 2 = 250) : S = 1000 :=
by
  sorry

end total_students_l149_149432


namespace all_tell_truth_at_same_time_l149_149446

-- Define the probabilities of each person telling the truth.
def prob_Alice := 0.7
def prob_Bob := 0.6
def prob_Carol := 0.8
def prob_David := 0.5

-- Prove that the probability that all four tell the truth at the same time is 0.168.
theorem all_tell_truth_at_same_time :
  prob_Alice * prob_Bob * prob_Carol * prob_David = 0.168 :=
by
  sorry

end all_tell_truth_at_same_time_l149_149446


namespace fraction_sum_eq_l149_149289

variable {x : ℝ}

theorem fraction_sum_eq (h : x ≠ -1) : 
  (x / (x + 1) ^ 2) + (1 / (x + 1) ^ 2) = 1 / (x + 1) := 
by
  sorry

end fraction_sum_eq_l149_149289


namespace golden_ratio_problem_l149_149916

theorem golden_ratio_problem
  (m n : ℝ) (sin cos : ℝ → ℝ)
  (h1 : m = 2 * sin (Real.pi / 10))
  (h2 : m ^ 2 + n = 4)
  (sin63 : sin (7 * Real.pi / 18) ≠ 0) :
  (m + Real.sqrt n) / (sin (7 * Real.pi / 18)) = 2 * Real.sqrt 2 := by
  sorry

end golden_ratio_problem_l149_149916


namespace bicycle_speed_B_l149_149764

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l149_149764


namespace greatest_second_term_l149_149400

-- Definitions and Conditions
def is_arithmetic_sequence (a d : ℕ) : Bool := (a > 0) && (d > 0)
def sum_four_terms (a d : ℕ) : Bool := (4 * a + 6 * d = 80)
def integer_d (a d : ℕ) : Bool := ((40 - 2 * a) % 3 = 0)

-- Theorem statement to prove
theorem greatest_second_term : ∃ a d : ℕ, is_arithmetic_sequence a d ∧ sum_four_terms a d ∧ integer_d a d ∧ (a + d = 19) :=
sorry

end greatest_second_term_l149_149400


namespace john_boxes_l149_149355

theorem john_boxes
  (stan_boxes : ℕ)
  (joseph_boxes : ℕ)
  (jules_boxes : ℕ)
  (john_boxes : ℕ)
  (h1 : stan_boxes = 100)
  (h2 : joseph_boxes = stan_boxes - 80 * stan_boxes / 100)
  (h3 : jules_boxes = joseph_boxes + 5)
  (h4 : john_boxes = jules_boxes + 20 * jules_boxes / 100) :
  john_boxes = 30 :=
by
  -- Proof will go here
  sorry

end john_boxes_l149_149355


namespace pow_mod_seventeen_l149_149548

theorem pow_mod_seventeen sixty_three :
  17^63 % 7 = 6 := by
  have h : 17 % 7 = 3 := by norm_num
  have h1 : 17^63 % 7 = 3^63 % 7 := by rw [pow_mod_eq_of_mod_eq h] 
  norm_num at h1
  rw [h1]
  sorry

end pow_mod_seventeen_l149_149548


namespace range_of_m_l149_149245

noncomputable def quadraticExpr (m : ℝ) (x : ℝ) : ℝ :=
  m * x^2 + 4 * m * x + m + 3

theorem range_of_m :
  (∀ x : ℝ, quadraticExpr m x ≥ 0) ↔ 0 ≤ m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l149_149245


namespace fermat_prime_sum_not_possible_l149_149048

-- Definitions of the conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, (m ∣ p) → (m = 1 ∨ m = p)

-- The Lean statement
theorem fermat_prime_sum_not_possible 
  (n : ℕ) (x y z : ℤ) (p : ℕ) 
  (h_odd : is_odd n) 
  (h_gt_one : n > 1) 
  (h_prime : is_prime p)
  (h_sum: x + y = ↑p) :
  ¬ (x ^ n + y ^ n = z ^ n) :=
by
  sorry


end fermat_prime_sum_not_possible_l149_149048


namespace lines_are_skew_l149_149459

def line1 (a t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 3 * t, 3 + 4 * t, a + 5 * t)

def line2 (u : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 6 * u, 2 + 2 * u, 1 + 2 * u)

theorem lines_are_skew (a : ℝ) :
  ¬(∃ t u : ℝ, line1 a t = line2 u) ↔ a ≠ 5 / 3 :=
sorry

end lines_are_skew_l149_149459


namespace stamps_in_album_l149_149570

theorem stamps_in_album (n : ℕ) : 
  n % 2 = 1 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ 
  n % 6 = 5 ∧ n % 7 = 6 ∧ n % 8 = 7 ∧ n % 9 = 8 ∧ 
  n % 10 = 9 ∧ n < 3000 → n = 2519 :=
by
  sorry

end stamps_in_album_l149_149570


namespace plywood_cut_difference_l149_149109

/-- A proof problem to determine the positive difference between the greatest possible
perimeter and the least possible perimeter of congruent pieces resulting from cutting 
a 6-foot by 9-foot rectangular plywood into 6 congruent rectangles with no wood leftover 
or lost --/
theorem plywood_cut_difference :
  ∃ (perimeter_max perimeter_min : ℕ), 
  let piece1 := 1 * 9
  let piece2 := 1 * 6
  let piece3 := 2 * 3
  let perimeter1 := 2 * (1 + 9)
  let perimeter2 := 2 * (1 + 6)
  let perimeter3 := 2 * (2 + 3)
  perimeter_max = perimeter1 ∧
  perimeter_min = perimeter3 ∧
  (perimeter_max - perimeter_min) = 10 :=
sorry

end plywood_cut_difference_l149_149109


namespace interval_solution_l149_149304

theorem interval_solution (x : ℝ) : 2 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 7 ↔ (35 / 13 : ℝ) < x ∧ x ≤ 10 / 3 :=
by
  sorry

end interval_solution_l149_149304


namespace boys_camp_problem_l149_149505

noncomputable def total_boys_in_camp : ℝ :=
  let schoolA_fraction := 0.20
  let science_fraction := 0.30
  let non_science_boys := 63
  let non_science_fraction := 1 - science_fraction
  let schoolA_boys := (non_science_boys / non_science_fraction)
  schoolA_boys / schoolA_fraction

theorem boys_camp_problem : total_boys_in_camp = 450 := 
by
  sorry

end boys_camp_problem_l149_149505


namespace average_weight_increase_l149_149241

theorem average_weight_increase
 (num_persons : ℕ) (weight_increase : ℝ) (replacement_weight : ℝ) (new_weight : ℝ) (weight_difference : ℝ) (avg_weight_increase : ℝ)
 (cond1 : num_persons = 10)
 (cond2 : replacement_weight = 65)
 (cond3 : new_weight = 90)
 (cond4 : weight_difference = new_weight - replacement_weight)
 (cond5 : weight_difference = weight_increase)
 (cond6 : avg_weight_increase = weight_increase / num_persons) :
avg_weight_increase = 2.5 :=
by
  sorry

end average_weight_increase_l149_149241


namespace sum_cubic_polynomial_l149_149426

noncomputable def q : ℤ → ℤ := sorry  -- We use a placeholder definition for q

theorem sum_cubic_polynomial :
  q 3 = 2 ∧ q 8 = 22 ∧ q 12 = 10 ∧ q 17 = 32 →
  (q 2 + q 3 + q 4 + q 5 + q 6 + q 7 + q 8 + q 9 + q 10 + q 11 + q 12 + q 13 + q 14 + q 15 + q 16 + q 17 + q 18) = 272 :=
sorry

end sum_cubic_polynomial_l149_149426


namespace sin_cos_value_sin_plus_cos_value_l149_149467

noncomputable def given_condition (θ : ℝ) : Prop := 
  (Real.tan θ + 1 / Real.tan θ = 2)

theorem sin_cos_value (θ : ℝ) (h : given_condition θ) : 
  Real.sin θ * Real.cos θ = 1 / 2 :=
sorry

theorem sin_plus_cos_value (θ : ℝ) (h : given_condition θ) : 
  Real.sin θ + Real.cos θ = Real.sqrt 2 ∨ Real.sin θ + Real.cos θ = -Real.sqrt 2 :=
sorry

end sin_cos_value_sin_plus_cos_value_l149_149467


namespace initial_number_of_persons_l149_149243

theorem initial_number_of_persons (n : ℕ) 
  (w_increase : ∀ (k : ℕ), k = 4) 
  (old_weight new_weight : ℕ) 
  (h_old : old_weight = 58) 
  (h_new : new_weight = 106) 
  (h_difference : new_weight - old_weight = 48) 
  : n = 12 := 
by
  sorry

end initial_number_of_persons_l149_149243


namespace z_amount_per_rupee_l149_149993

theorem z_amount_per_rupee (x y z : ℝ) 
  (h1 : ∀ rupees_x, y = 0.45 * rupees_x)
  (h2 : y = 36)
  (h3 : x + y + z = 156)
  (h4 : ∀ rupees_x, x = rupees_x) :
  ∃ a : ℝ, z = a * x ∧ a = 0.5 := 
by
  -- Placeholder for the actual proof
  sorry

end z_amount_per_rupee_l149_149993


namespace student_B_speed_l149_149794

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l149_149794


namespace routes_from_A_to_B_l149_149582

-- Definitions based on conditions given in the problem
variables (A B C D E F : Type)
variables (AB AD AE BC BD CD DE EF : Prop) 

-- Theorem statement
theorem routes_from_A_to_B (route_criteria : AB ∧ AD ∧ AE ∧ BC ∧ BD ∧ CD ∧ DE ∧ EF)
  : ∃ n : ℕ, n = 16 :=
sorry

end routes_from_A_to_B_l149_149582


namespace not_enough_space_in_cube_l149_149737

-- Define the edge length of the cube in kilometers.
def cube_edge_length_km : ℝ := 3

-- Define the global population exceeding threshold.
def global_population : ℝ := 7 * 10^9

-- Define the function to calculate the volume of a cube given its edge length in kilometers.
def cube_volume_km (edge_length: ℝ) : ℝ := edge_length^3

-- Define the conversion from kilometers to meters.
def km_to_m (distance_km: ℝ) : ℝ := distance_km * 1000

-- Define the function to calculate the volume of the cube in cubic meters.
def cube_volume_m (edge_length_km: ℝ) : ℝ := (km_to_m edge_length_km)^3

-- Statement: The entire population and all buildings and structures will not fit inside the cube.
theorem not_enough_space_in_cube :
  cube_volume_m cube_edge_length_km < global_population * (some_constant_value_to_account_for_buildings_and_structures) :=
sorry

end not_enough_space_in_cube_l149_149737


namespace trapezium_other_side_length_l149_149173

theorem trapezium_other_side_length 
  (side1 : ℝ) (distance : ℝ) (area : ℝ) (side2 : ℝ)
  (h_side1 : side1 = 18)
  (h_distance : distance = 13)
  (h_area : area = 247)
  (h_area_formula : area = 0.5 * (side1 + side2) * distance) :
  side2 = 20 :=
by
  rw [h_side1, h_distance, h_area] at h_area_formula
  sorry

end trapezium_other_side_length_l149_149173


namespace mul_mod_remainder_l149_149083

theorem mul_mod_remainder (a b m : ℕ)
  (h₁ : a ≡ 8 [MOD 9])
  (h₂ : b ≡ 1 [MOD 9]) :
  (a * b) % 9 = 8 := 
  sorry

def main : IO Unit :=
  IO.println "The theorem statement has been defined."

end mul_mod_remainder_l149_149083


namespace monotonicity_increasing_when_a_nonpos_monotonicity_increasing_decreasing_when_a_pos_range_of_a_for_f_less_than_zero_l149_149365

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

-- Define the problem stating that when a <= 0, f(x) is increasing on (0, +∞)
theorem monotonicity_increasing_when_a_nonpos (a : ℝ) (h : a ≤ 0) :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f a x < f a y :=
sorry

-- Define the problem stating that when a > 0, f(x) is increasing on (0, 1/a) and decreasing on (1/a, +∞)
theorem monotonicity_increasing_decreasing_when_a_pos (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → x < (1 / a) → y < (1 / a) → f a x < f a y) ∧
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → (1 / a) < x → (1 / a) < y → f a y < f a x) :=
sorry

-- Define the problem for the range of a such that f(x) < 0 for all x in (0, +∞)
theorem range_of_a_for_f_less_than_zero (a : ℝ) :
  (∀ x : ℝ, 0 < x → f a x < 0) ↔ a ∈ Set.Ioi (1 / Real.exp 1) :=
sorry

end monotonicity_increasing_when_a_nonpos_monotonicity_increasing_decreasing_when_a_pos_range_of_a_for_f_less_than_zero_l149_149365


namespace find_digit_B_in_5BB3_l149_149393

theorem find_digit_B_in_5BB3 (B : ℕ) (h : 5BB3 / 10^3 = 5 + 100*B + 10*B + 3) (divby9 : (5 + B + B + 3) % 9 = 0) : B = 5 := 
  by 
    sorry

end find_digit_B_in_5BB3_l149_149393


namespace fraction_identity_l149_149491

variable {n : ℕ}

theorem fraction_identity
  (h1 : ∀ n : ℕ, (n ≠ 0 → n ≠ 1 → 1 / (n * (n + 1)) = 1 / n - 1 / (n + 1)))
  (h2 : ∀ n : ℕ, (n ≠ 0 → n ≠ 1 → n ≠ 2 → 1 / (n * (n + 1) * (n + 2)) = 1 / (2 * n * (n + 1)) - 1 / (2 * (n + 1) * (n + 2))))
  : 1 / (n * (n + 1) * (n + 2) * (n + 3)) = 1 / (3 * n * (n + 1) * (n + 2)) - 1 / (3 * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end fraction_identity_l149_149491


namespace illumination_ways_l149_149026

def ways_to_illuminate_traffic_lights (n : ℕ) : ℕ :=
  3^n

theorem illumination_ways (n : ℕ) : ways_to_illuminate_traffic_lights n = 3 ^ n :=
by
  sorry

end illumination_ways_l149_149026


namespace sum_of_a_and_b_l149_149442

theorem sum_of_a_and_b (a b : ℕ) (h1 : a > 0) (h2 : b > 1) (h3 : a^b < 500) (h_max : ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a'^b' ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l149_149442


namespace speed_of_student_B_l149_149841

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l149_149841


namespace number_of_pieces_of_paper_l149_149444

def three_digit_number_with_unique_digits (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ n / 100 ≠ (n / 10) % 10 ∧ n / 100 ≠ n % 10 ∧ (n / 10) % 10 ≠ n % 10

theorem number_of_pieces_of_paper (n : ℕ) (k : ℕ) (h1 : three_digit_number_with_unique_digits n) (h2 : 2331 = k * n) : k = 9 :=
by
  sorry

end number_of_pieces_of_paper_l149_149444


namespace fifth_root_of_unity_l149_149061

noncomputable def expression (x : ℂ) := 
  2 * x + 1 / (1 + x) + x / (1 + x^2) + x^2 / (1 + x^3) + x^3 / (1 + x^4)

theorem fifth_root_of_unity (x : ℂ) (hx : x^5 = 1) : 
  (expression x = 4) ∨ (expression x = -1 + Real.sqrt 5) ∨ (expression x = -1 - Real.sqrt 5) :=
sorry

end fifth_root_of_unity_l149_149061


namespace angle_B_of_triangle_l149_149889

theorem angle_B_of_triangle {A B C a b c : ℝ} (h1 : b^2 = a * c) (h2 : Real.sin A + Real.sin C = 2 * Real.sin B) : 
  B = Real.pi / 3 :=
sorry

end angle_B_of_triangle_l149_149889


namespace sequence_general_formula_l149_149185

theorem sequence_general_formula :
  ∀ (a : ℕ → ℝ),
  (a 1 = 1) →
  (∀ n : ℕ, n > 0 → a n - a (n + 1) = 2 * a n * a (n + 1) / (n * (n + 1))) →
  ∀ n : ℕ, n > 0 → a n = n / (3 * n - 2) :=
by
  intros a h1 h_rec n hn
  sorry

end sequence_general_formula_l149_149185


namespace minimum_combined_horses_ponies_l149_149988

noncomputable def ranch_min_total (P H : ℕ) : ℕ :=
  P + H

theorem minimum_combined_horses_ponies (P H : ℕ) 
  (h1 : ∃ k : ℕ, P = 16 * k ∧ k ≥ 1)
  (h2 : H = P + 3) 
  (h3 : P = 80) 
  (h4 : H = 83) :
  ranch_min_total P H = 163 :=
by
  sorry

end minimum_combined_horses_ponies_l149_149988


namespace michael_max_correct_answers_l149_149984

theorem michael_max_correct_answers (c w b : ℕ) 
  (h1 : c + w + b = 30) 
  (h2 : 4 * c - 3 * w = 72) : 
  c ≤ 21 := 
sorry

end michael_max_correct_answers_l149_149984


namespace inequality_transitivity_l149_149198

theorem inequality_transitivity (a b c : ℝ) (h : a > b) : 
  a + c > b + c :=
sorry

end inequality_transitivity_l149_149198


namespace non_zero_const_c_l149_149247

theorem non_zero_const_c (a b c x1 x2 : ℝ) (h1 : x1 ≠ 0) (h2 : x2 ≠ 0) 
(h3 : (a - 1) * x1 ^ 2 + b * x1 + c = 0) 
(h4 : (a - 1) * x2 ^ 2 + b * x2 + c = 0)
(h5 : x1 * x2 = -1) 
(h6 : x1 ≠ x2) 
(h7 : x1 * x2 < 0): c ≠ 0 :=
sorry

end non_zero_const_c_l149_149247


namespace batsman_average_after_17th_inning_l149_149747

theorem batsman_average_after_17th_inning
  (A : ℝ)
  (h1 : A + 10 = (16 * A + 200) / 17)
  : (A = 30 ∧ (A + 10) = 40) :=
by
  sorry

end batsman_average_after_17th_inning_l149_149747


namespace smallest_w_l149_149017

theorem smallest_w (w : ℕ) (hw : w > 0) (h1 : ∃ k1, 936 * w = 2^5 * k1) (h2 : ∃ k2, 936 * w = 3^3 * k2) (h3 : ∃ k3, 936 * w = 10^2 * k3) : 
  w = 300 :=
by
  sorry

end smallest_w_l149_149017


namespace track_length_l149_149157

theorem track_length (L : ℕ)
  (h1 : ∃ B S : ℕ, B = 120 ∧ (L - B) = S ∧ (S + 200) - B = (L + 80) - B)
  (h2 : L + 80 = 440 - L) : L = 180 := 
  by
    sorry

end track_length_l149_149157


namespace group_allocation_minimizes_time_total_duration_after_transfer_l149_149099

theorem group_allocation_minimizes_time :
  ∃ x y : ℕ,
  x + y = 52 ∧
  (x = 20 ∧ y = 32) ∧
  (min (60 / x) (100 / y) = 25 / 8) := sorry

theorem total_duration_after_transfer (x y x' y' : ℕ) (H : x = 20) (H1 : y = 32) (H2 : x' = x - 6) (H3 : y' = y + 6) :
  min ((100 * (2/5)) / x') ((152 * (2/3)) / y') = 27 / 7 := sorry

end group_allocation_minimizes_time_total_duration_after_transfer_l149_149099


namespace find_speed_of_B_l149_149808

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l149_149808


namespace part_I_solution_set_part_II_min_value_l149_149651

-- Define the function f(x)
def f (x : ℝ) : ℝ := x + 1 + |3 - x|

-- Prove the solution set of the inequality f(x) ≤ 6 for x ≥ -1 is -1 ≤ x ≤ 4
theorem part_I_solution_set (x : ℝ) (h1 : x ≥ -1) : f x ≤ 6 ↔ (-1 ≤ x ∧ x ≤ 4) :=
by
  sorry

-- Define the condition for the minimum value of f(x)
def min_f := 4

-- Prove the minimum value of 2a + b under the given constraints
theorem part_II_min_value (a b : ℝ) (h2 : a > 0 ∧ b > 0) (h3 : 8 * a * b = a + 2 * b) : 2 * a + b ≥ 9 / 8 :=
by
  sorry

end part_I_solution_set_part_II_min_value_l149_149651


namespace find_speed_B_l149_149854

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l149_149854


namespace population_factor_proof_l149_149719

-- Define the conditions given in the problem
variables (N x y z : ℕ)

theorem population_factor_proof :
  (N = x^2) ∧ (N + 100 = y^2 + 1) ∧ (N + 200 = z^2) → (7 ∣ N) :=
by sorry

end population_factor_proof_l149_149719


namespace student_B_speed_l149_149795

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l149_149795


namespace square_side_length_l149_149388

theorem square_side_length (d s : ℝ) (h_diag : d = 2) (h_rel : d = s * Real.sqrt 2) : s = Real.sqrt 2 :=
sorry

end square_side_length_l149_149388


namespace tangent_line_to_curve_at_point_l149_149307

theorem tangent_line_to_curve_at_point :
  ∀ (x y : ℝ),
  (y = 2 * Real.log x) →
  (x = 2) →
  (y = 2 * Real.log 2) →
  (x - y + 2 * Real.log 2 - 2 = 0) := by
  sorry

end tangent_line_to_curve_at_point_l149_149307


namespace zero_in_interval_l149_149253

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 + x - 2

theorem zero_in_interval : f 1 < 0 ∧ f 2 > 0 → ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 := 
by
  intros h
  sorry

end zero_in_interval_l149_149253


namespace simplify_expr_l149_149704

variable (x y : ℝ)

def expr (x y : ℝ) := (x + y) * (x - y) - y * (2 * x - y)

theorem simplify_expr :
  let x := Real.sqrt 2
  let y := Real.sqrt 3
  expr x y = 2 - 2 * Real.sqrt 6 := by
  sorry

end simplify_expr_l149_149704


namespace production_average_l149_149309

theorem production_average (n : ℕ) (P : ℕ) (h1 : P / n = 50) (h2 : (P + 90) / (n + 1) = 54) : n = 9 :=
sorry

end production_average_l149_149309


namespace right_triangle_segments_l149_149033

open Real

theorem right_triangle_segments 
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (h_ab : a > b)
  (P Q : ℝ × ℝ) (P_on_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (Q_on_ellipse : Q.1^2 / a^2 + Q.2^2 / b^2 = 1)
  (Q_in_first_quad : Q.1 > 0 ∧ Q.2 > 0)
  (OQ_parallel_AP : ∃ k : ℝ, Q.1 = k * P.1 ∧ Q.2 = k * P.2)
  (M : ℝ × ℝ) (M_midpoint : M = ((P.1 + 0) / 2, (P.2 + 0) / 2))
  (R : ℝ × ℝ) (R_on_ellipse : R.1^2 / a^2 + R.2^2 / b^2 = 1)
  (OM_intersects_R : ∃ k : ℝ, R = (k * M.1, k * M.2))
: dist (0,0) Q ≠ 0 →
  dist (0,0) R ≠ 0 →
  dist (Q, R) ≠ 0 →
  dist (0,0) Q ^ 2 + dist (0,0) R ^ 2 = dist ((-a), (b)) ((a), (b)) ^ 2 :=
by
  sorry

end right_triangle_segments_l149_149033


namespace inequality_proof_equality_condition_l149_149926

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / c) + (c / b) ≥ (4 * a) / (a + b) := 
by
  sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / c) + (c / b) = (4 * a) / (a + b) ↔ a = b ∧ b = c :=
by
  sorry

end inequality_proof_equality_condition_l149_149926


namespace convert_spherical_to_rectangular_l149_149294

noncomputable def spherical_to_rectangular (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin phi * Real.cos theta,
   rho * Real.sin phi * Real.sin theta,
   rho * Real.cos phi)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 4) = (2 * Real.sqrt 3, Real.sqrt 2, 2 * Real.sqrt 2) :=
by
  -- Define the spherical coordinates
  let rho := 4
  let theta := Real.pi / 6
  let phi := Real.pi / 4

  -- Calculate x, y, z using conversion formulas
  sorry

end convert_spherical_to_rectangular_l149_149294


namespace variance_inequality_construct_gaussian_variables_l149_149269

noncomputable def exchangeable_random_variables (X : ℕ → ℝ) : Prop :=
∀ (σ : Fin.perm (Fin n)), 
  (∀ i : Fin n, X i) 
  = (∀ i : Fin n, X (σ i))

noncomputable def covariance (X Y : ℝ) : ℝ :=
∫ x, (X - ∫ y, X) * (Y - ∫ y, Y)

noncomputable def variance (X : ℝ) : ℝ :=
covariance X X

theorem variance_inequality
  (n : ℕ)
  (X : ℕ → ℝ)
  (h_exchangeable : exchangeable_random_variables X) :
  variance (X 1) ≥
  if covariance (X 1) (X 2) < 0 then
    (n - 1) * |covariance (X 1) (X 2)|
  else
    covariance (X 1) (X 2) :=
sorry

theorem construct_gaussian_variables
  (n : ℕ)
  (ρ σ2 : ℝ)
  (h_ρ_neg : ρ < 0)
  (h_ineq_neg : σ2 + (n - 1) * ρ ≥ 0)
  (h_ρ_nonneg : ρ ≥ 0)
  (h_ineq_nonneg : σ2 ≥ ρ) :
  ∃ (X : ℕ → ℝ),
    exchangeable_random_variables X ∧
    (∫ x, X x = 0) ∧
    (variance (X 1) = σ2) ∧
    (covariance (X 1) (X 2) = ρ) :=
sorry

end variance_inequality_construct_gaussian_variables_l149_149269


namespace alicia_taxes_l149_149153

theorem alicia_taxes:
  let w := 20 -- Alicia earns 20 dollars per hour
  let r := 1.45 / 100 -- The local tax rate is 1.45%
  let wage_in_cents := w * 100 -- Convert dollars to cents
  let tax_deduction := wage_in_cents * r -- Calculate tax deduction in cents
  tax_deduction = 29 := 
by 
  sorry

end alicia_taxes_l149_149153


namespace difference_between_numbers_l149_149712

theorem difference_between_numbers 
  (L S : ℕ) 
  (hL : L = 1584) 
  (hDiv : L = 6 * S + 15) : 
  L - S = 1323 := 
by
  sorry

end difference_between_numbers_l149_149712


namespace trekking_adults_l149_149276

theorem trekking_adults
  (A : ℕ)
  (C : ℕ)
  (meal_for_adults : ℕ)
  (meal_for_children : ℕ)
  (remaining_food_children : ℕ) :
  C = 70 →
  meal_for_adults = 70 →
  meal_for_children = 90 →
  remaining_food_children = 72 →
  A - 14 = (meal_for_adults - 14) →
  A = 56 :=
sorry

end trekking_adults_l149_149276


namespace minimum_value_inequality_l149_149219

open Real

theorem minimum_value_inequality
  (a b c : ℝ)
  (ha : 2 ≤ a) 
  (hb : a ≤ b)
  (hc : b ≤ c)
  (hd : c ≤ 5) :
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (5 / c - 1)^2 = 4 * (sqrt 5 ^ (1 / 4) - 1)^2 :=
sorry

end minimum_value_inequality_l149_149219


namespace find_number_l149_149665

noncomputable def least_common_multiple (a b : ℕ) : ℕ := Nat.lcm a b

theorem find_number (n : ℕ) (h1 : least_common_multiple (least_common_multiple n 16) (least_common_multiple 18 24) = 144) : n = 9 :=
sorry

end find_number_l149_149665


namespace percentage_of_first_solution_l149_149739

theorem percentage_of_first_solution (P : ℕ) 
  (h1 : 28 * P / 100 + 12 * 80 / 100 = 40 * 45 / 100) : 
  P = 30 :=
sorry

end percentage_of_first_solution_l149_149739


namespace find_numbers_l149_149073

theorem find_numbers : ∃ x y : ℕ, x + y = 2016 ∧ (∃ d : ℕ, d < 10 ∧ (x = 10 * y + d) ∧ x = 1833 ∧ y = 183) :=
by 
  sorry

end find_numbers_l149_149073


namespace probability_at_least_one_girl_correct_l149_149575

def probability_at_least_one_girl (total_students boys girls selections : ℕ) : Rat :=
  let total_ways := Nat.choose total_students selections
  let at_least_one_girl_ways := (Nat.choose boys 1) * (Nat.choose girls 1) + Nat.choose girls 2
  at_least_one_girl_ways / total_ways

theorem probability_at_least_one_girl_correct :
  probability_at_least_one_girl 5 3 2 2 = 7 / 10 := 
by
  -- sorry to skip the proof
  sorry

end probability_at_least_one_girl_correct_l149_149575


namespace no_int_satisfies_both_congruences_l149_149368

theorem no_int_satisfies_both_congruences :
  ¬ ∃ n : ℤ, (n ≡ 5 [ZMOD 6]) ∧ (n ≡ 1 [ZMOD 21]) :=
sorry

end no_int_satisfies_both_congruences_l149_149368


namespace solve_for_m_l149_149497

theorem solve_for_m : 
  ∀ m : ℝ, (3 * (-2) + 5 = -2 - m) → m = -1 :=
by
  intros m h
  sorry

end solve_for_m_l149_149497


namespace part1_part2_l149_149081

def star (a b c d : ℝ) : ℝ := a * c - b * d

-- Part (1)
theorem part1 : star (-4) 3 2 (-6) = 10 := by
  sorry

-- Part (2)
theorem part2 (m : ℝ) (h : ∀ x : ℝ, star x (2 * x - 1) (m * x + 1) m = 0 → (m ≠ 0 → (((1 - 2 * m) ^ 2 - 4 * m * m) ≥ 0))) :
  (m ≤ 1 / 4 ∨ m < 0) ∧ m ≠ 0 := by
  sorry

end part1_part2_l149_149081


namespace max_days_to_be_cost_effective_is_8_l149_149977

-- Definitions
def cost_of_hiring (₥ : ℕ) := 50000
def cost_of_materials (₥ : ℕ) := 20000
def husbands_daily_wage (₥ : ℕ) := 2000
def wifes_daily_wage (₥ : ℕ) := 1500

-- Total daily wage
def total_daily_wage := husbands_daily_wage 0 + wifes_daily_wage 0

-- Cost difference
def cost_difference := cost_of_hiring 0 - cost_of_materials 0

-- Maximum number of days
def max_days_to_be_cost_effective := cost_difference / total_daily_wage

-- Prove that the maximum number of days is 8
theorem max_days_to_be_cost_effective_is_8 : max_days_to_be_cost_effective = 8 := by
  sorry

end max_days_to_be_cost_effective_is_8_l149_149977


namespace speed_of_train_is_correct_l149_149091

-- Given conditions
def length_of_train : ℝ := 250
def length_of_bridge : ℝ := 120
def time_to_cross_bridge : ℝ := 20

-- Derived definition
def total_distance : ℝ := length_of_train + length_of_bridge

-- Goal to be proved
theorem speed_of_train_is_correct : total_distance / time_to_cross_bridge = 18.5 := 
by
  sorry

end speed_of_train_is_correct_l149_149091


namespace interval_of_y_l149_149985

theorem interval_of_y (y : ℝ) (h : y = (1 / y) * (-y) - 5) : -6 ≤ y ∧ y ≤ -4 :=
by sorry

end interval_of_y_l149_149985


namespace least_n_l149_149631

theorem least_n (n : ℕ) (h_pos : n > 0) (h_ineq : 1 / n - 1 / (n + 1) < 1 / 15) : n = 4 :=
sorry

end least_n_l149_149631


namespace max_value_sqrt_expr_l149_149689

open Real

theorem max_value_sqrt_expr (x y z : ℝ)
  (h1 : x + y + z = 1)
  (h2 : x ≥ -1/3)
  (h3 : y ≥ -1)
  (h4 : z ≥ -5/3) :
  (sqrt (3 * x + 1) + sqrt (3 * y + 3) + sqrt (3 * z + 5)) ≤ 6 :=
  sorry

end max_value_sqrt_expr_l149_149689


namespace kyro_percentage_paid_l149_149997

theorem kyro_percentage_paid
    (aryan_debt : ℕ) -- Aryan owes Fernanda $1200
    (kyro_debt : ℕ) -- Kyro owes Fernanda
    (aryan_debt_twice_kyro_debt : aryan_debt = 2 * kyro_debt) -- Aryan's debt is twice what Kyro owes
    (aryan_payment : ℕ) -- Aryan's payment
    (aryan_payment_percentage : aryan_payment = 60 * aryan_debt / 100) -- Aryan pays 60% of her debt
    (initial_savings : ℕ) -- Initial savings in Fernanda's account
    (final_savings : ℕ) -- Final savings in Fernanda's account
    (initial_savings_cond : initial_savings = 300) -- Fernanda's initial savings is $300
    (final_savings_cond : final_savings = 1500) -- Fernanda's final savings is $1500
    : kyro_payment = 80 * kyro_debt / 100 := -- Kyro paid 80% of her debt
by {
    sorry
}

end kyro_percentage_paid_l149_149997


namespace solve_sqrt_eq_l149_149879

theorem solve_sqrt_eq (x : ℝ) :
  (Real.sqrt ((3 + 2 * Real.sqrt 2)^x) + Real.sqrt ((3 - 2 * Real.sqrt 2)^x) = 5) ↔ (x = 2 ∨ x = -2) := by
  sorry

end solve_sqrt_eq_l149_149879


namespace least_n_satisfies_inequality_l149_149639

theorem least_n_satisfies_inequality : ∃ n : ℕ, (1 : ℚ) / n - (1 : ℚ) / (n + 1) < 1 / 15 ∧ ∀ m : ℕ, (1 : ℚ) / m - (1 : ℚ) / (m + 1) < 1 / 15 -> ¬ (m < n) := 
sorry

end least_n_satisfies_inequality_l149_149639


namespace fibonacci_recurrence_l149_149976

def F : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => F (n + 1) + F n

theorem fibonacci_recurrence (n : ℕ) (h: n ≥ 2) : 
  F n = F (n-1) + F (n-2) := by
 {
 sorry
 }

end fibonacci_recurrence_l149_149976


namespace student_B_speed_l149_149811

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l149_149811


namespace find_num_oranges_l149_149943

def num_oranges (O : ℝ) (x : ℕ) : Prop :=
  6 * 0.21 + O * (x : ℝ) = 1.77 ∧ 2 * 0.21 + 5 * O = 1.27
  ∧ 0.21 = 0.21

theorem find_num_oranges (O : ℝ) (x : ℕ) (h : num_oranges O x) : x = 3 :=
  sorry

end find_num_oranges_l149_149943


namespace total_candies_l149_149567

theorem total_candies (Linda_candies Chloe_candies : ℕ) (h1 : Linda_candies = 34) (h2 : Chloe_candies = 28) :
  Linda_candies + Chloe_candies = 62 := by
  sorry

end total_candies_l149_149567


namespace eval_g_l149_149447

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem eval_g : 3 * g 2 + 4 * g (-4) = 327 := 
by
  sorry

end eval_g_l149_149447


namespace gerald_total_pieces_eq_672_l149_149180

def pieces_per_table : Nat := 12
def pieces_per_chair : Nat := 8
def num_tables : Nat := 24
def num_chairs : Nat := 48

def total_pieces : Nat := pieces_per_table * num_tables + pieces_per_chair * num_chairs

theorem gerald_total_pieces_eq_672 : total_pieces = 672 :=
by
  sorry

end gerald_total_pieces_eq_672_l149_149180


namespace find_line_eq_l149_149648

theorem find_line_eq (x y : ℝ) (h : x^2 + y^2 - 4 * x - 5 = 0) 
(mid_x mid_y : ℝ) (mid_point : mid_x = 3 ∧ mid_y = 1) : 
x + y - 4 = 0 := 
sorry

end find_line_eq_l149_149648


namespace A_time_to_complete_work_l149_149341

-- Definitions of work rates for A, B, and C.
variables (A_work B_work C_work : ℚ)

-- Conditions
axiom cond1 : A_work = 3 * B_work
axiom cond2 : B_work = 2 * C_work
axiom cond3 : A_work + B_work + C_work = 1 / 15

-- Proof statement: The time taken by A alone to do the work is 22.5 days.
theorem A_time_to_complete_work : 1 / A_work = 22.5 :=
by {
  sorry
}

end A_time_to_complete_work_l149_149341


namespace least_n_satisfies_inequality_l149_149635

theorem least_n_satisfies_inequality : ∃ n : ℕ, (1 : ℚ) / n - (1 : ℚ) / (n + 1) < 1 / 15 ∧ ∀ m : ℕ, (1 : ℚ) / m - (1 : ℚ) / (m + 1) < 1 / 15 -> ¬ (m < n) := 
sorry

end least_n_satisfies_inequality_l149_149635


namespace rectangle_area_l149_149511

theorem rectangle_area (AB AD AE : ℝ) (S_trapezoid S_triangle : ℝ) (perim_triangle perim_trapezoid : ℝ)
  (h1 : AD - AB = 9)
  (h2 : S_trapezoid = 5 * S_triangle)
  (h3 : perim_triangle + 68 = perim_trapezoid)
  (h4 : S_trapezoid + S_triangle = S_triangle * 6)
  (h5 : perim_triangle = AB + AE + (AE - AB))
  (h6 : perim_trapezoid = AB + AD + AE + (2 * (AD - AE))) :
  AD * AB = 3060 := by
  sorry

end rectangle_area_l149_149511


namespace age_of_other_man_l149_149529

-- Definitions of the given conditions
def average_age_increase (avg_men : ℕ → ℝ) (men_removed women_avg : ℕ) : Prop :=
  avg_men 8 + 2 = avg_men 6 + women_avg / 2

def one_man_age : ℕ := 24
def women_avg : ℕ := 30

-- Statement of the problem to prove
theorem age_of_other_man (avg_men : ℕ → ℝ) (other_man : ℕ) :
  average_age_increase avg_men 24 women_avg →
  other_man = 20 :=
sorry

end age_of_other_man_l149_149529


namespace inequality_solution_set_l149_149022

theorem inequality_solution_set (a b x : ℝ) (h1 : a > 0) (h2 : b = a) : 
  ((a * x + b) * (x - 3) > 0 ↔ x < -1 ∨ x > 3) :=
by
  sorry

end inequality_solution_set_l149_149022


namespace triangle_evaluation_l149_149218

def triangle (a b : ℤ) : ℤ := a^2 - 2 * b

theorem triangle_evaluation : triangle (-2) (triangle 3 2) = -6 := by
  sorry

end triangle_evaluation_l149_149218


namespace plywood_cut_perimeter_difference_l149_149117

theorem plywood_cut_perimeter_difference :
  (∃ (l w : ℕ), (l * w = 54) ∧ (9 % w = 0) ∧ (6 % l = 0) ∧ (6 / l) * (9 / w) = 6) →
  10 =
  let p := λ l w, 2 * (l + w) in
  let perimeters := [
    p 1 9,
    p 1 6,
    p 2 3,
    p 3 2
  ]
  in (list.max precedence perimeters - list.min precedence perimeters) :=
begin
  sorry
end

end plywood_cut_perimeter_difference_l149_149117


namespace haley_laundry_loads_l149_149659

theorem haley_laundry_loads (shirts sweaters pants socks : ℕ) 
    (machine_capacity total_pieces : ℕ)
    (sum_of_clothing : 6 + 28 + 10 + 9 = total_pieces)
    (machine_capacity_eq : machine_capacity = 5) :
  ⌈(total_pieces:ℚ) / machine_capacity⌉ = 11 :=
by
  sorry

end haley_laundry_loads_l149_149659


namespace number_of_cows_in_farm_l149_149672

-- Definitions relating to the conditions
def total_bags_consumed := 20
def bags_per_cow := 1
def days := 20

-- Question and proof of the answer
theorem number_of_cows_in_farm : (total_bags_consumed / bags_per_cow) = 20 := by
  -- proof goes here
  sorry

end number_of_cows_in_farm_l149_149672


namespace salt_concentration_l149_149748

theorem salt_concentration (volume_water volume_solution concentration_solution : ℝ)
  (h1 : volume_water = 1)
  (h2 : volume_solution = 0.5)
  (h3 : concentration_solution = 0.45) :
  (volume_solution * concentration_solution) / (volume_water + volume_solution) = 0.15 :=
by
  sorry

end salt_concentration_l149_149748


namespace max_triangles_in_graph_l149_149597

def points : Finset Point := sorry
def no_coplanar (points : Finset Point) : Prop := sorry
def no_tetrahedron (points : Finset Point) : Prop := sorry
def triangles (points : Finset Point) : ℕ := sorry

theorem max_triangles_in_graph (points : Finset Point) 
  (H1 : points.card = 9) 
  (H2 : no_coplanar points) 
  (H3 : no_tetrahedron points) : 
  triangles points ≤ 27 := 
sorry

end max_triangles_in_graph_l149_149597


namespace least_n_l149_149602

theorem least_n (n : ℕ) (h : (1 : ℝ) / n - (1 / (n + 1)) < (1 / 15)) : n = 4 :=
sorry

end least_n_l149_149602


namespace max_even_integers_for_odd_product_l149_149280

theorem max_even_integers_for_odd_product (a b c d e f g : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : 0 < f) (h7 : 0 < g) 
  (h_prod_odd : a * b * c * d * e * f * g % 2 = 1) : a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ e % 2 = 1 ∧ f % 2 = 1 ∧ g % 2 = 1 :=
sorry

end max_even_integers_for_odd_product_l149_149280


namespace ellipse_eq_derive_AF2_BF2_eq_C2_trajectory_separation_l149_149321

-- Defining the conditions of the ellipse C1
variables {a b : ℝ}
def ellipse_C1 (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Given conditions for a and b
axiom a_gt_b_gt_zero : a > 0 ∧ b > 0 ∧ a > b

-- Proving the equation of the ellipse
theorem ellipse_eq_derive :
  (4 * a = 4 * real.sqrt 2 → a = real.sqrt 2) →
  ((2 * real.sqrt 2 / b^2) = 2 * real.sqrt 2 → b = 1) →
  ellipse_C1 x y ↔ (x^2 / 2 + y^2 = 1) :=
sorry

-- Proving |AF2| + |BF2| = 2√2 |AF2||BF2| for all α in [0, π)
theorem AF2_BF2_eq (α : ℝ) (hα : 0 ≤ α ∧ α < real.pi) :
  |AF2| + |BF2| = 2 * real.sqrt 2 * |AF2| * |BF2| :=
sorry

-- Proving the equation of the trajectory of E and its separation
theorem C2_trajectory_separation :
  (OC_perp OD) →
  (perp_through_O_intersects_l2_at_E) →
  let E_traj_eq := x^2 + y^2 = (2 / 3)
  (equation_of_trajectory E E_traj_eq) →
  (directrix_of_C1 = x = 1 ∨ directrix_of_C1 = x = -1) →
  separated (x^2 + y^2 = 2 / 3) (x = 1 ∨ x = -1) :=
sorry

end ellipse_eq_derive_AF2_BF2_eq_C2_trajectory_separation_l149_149321


namespace max_abs_diff_f_l149_149897

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2 * Real.exp x

theorem max_abs_diff_f {k x1 x2 : ℝ} (hk : -3 ≤ k ∧ k ≤ -1) 
    (hx1 : k ≤ x1 ∧ x1 ≤ k + 2) (hx2 : k ≤ x2 ∧ x2 ≤ k + 2) : 
    |f x1 - f x2| ≤ 4 * Real.exp 1 := 
sorry

end max_abs_diff_f_l149_149897


namespace number_of_arrangements_is_48_l149_149419

noncomputable def number_of_arrangements (students : List String) (boy_not_at_ends : String) (adjacent_girls : List String) : Nat :=
  sorry

theorem number_of_arrangements_is_48 : number_of_arrangements ["A", "B1", "B2", "G1", "G2", "G3"] "B1" ["G1", "G2", "G3"] = 48 :=
by
  sorry

end number_of_arrangements_is_48_l149_149419


namespace speed_of_student_B_l149_149838

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l149_149838


namespace trig_identity_equiv_l149_149181

theorem trig_identity_equiv (α : ℝ) (h : Real.sin (Real.pi - α) = -2 * Real.cos (-α)) : 
  Real.sin (2 * α) - Real.cos α ^ 2 = -1 :=
by
  sorry

end trig_identity_equiv_l149_149181


namespace distance_from_M0_to_plane_is_sqrt77_l149_149306

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def M1 : Point3D := ⟨1, 0, 2⟩
def M2 : Point3D := ⟨1, 2, -1⟩
def M3 : Point3D := ⟨2, -2, 1⟩
def M0 : Point3D := ⟨-5, -9, 1⟩

noncomputable def distance_to_plane (P : Point3D) (A B C : Point3D) : ℝ := sorry

theorem distance_from_M0_to_plane_is_sqrt77 : 
  distance_to_plane M0 M1 M2 M3 = Real.sqrt 77 := sorry

end distance_from_M0_to_plane_is_sqrt77_l149_149306


namespace remainder_tiling_8x1_l149_149568

theorem remainder_tiling_8x1 (N : ℕ) :
  (∃ (m : ℕ → ℕ), (∀ i, m i > 0) ∧ (∑ i in finset.range 8, m i) = 8 ∧
  (∀ i, ∃ color : fin (3), true) ∧ -- there are red, blue, or green tiles
  (∃ r_count b_count g_count, r_count > 0 ∧ b_count > 0 ∧ g_count > 0 ∧ r_count + b_count + g_count = 8))
  → N ≡ 179 [MOD 1000] :=
begin
  sorry
end

end remainder_tiling_8x1_l149_149568


namespace question_proof_l149_149662

theorem question_proof (x y : ℝ) (h : x * (x + y) = x^2 + y + 12) : xy + y^2 = y^2 + y + 12 :=
by
  sorry

end question_proof_l149_149662


namespace person2_speed_l149_149257

variables (v_1 : ℕ) (v_2 : ℕ)

def meet_time := 4
def catch_up_time := 16

def meet_equation : Prop := v_1 + v_2 = 22
def catch_up_equation : Prop := v_2 - v_1 = 4

theorem person2_speed :
  meet_equation v_1 v_2 → catch_up_equation v_1 v_2 →
  v_1 = 6 → v_2 = 10 :=
by
  intros h1 h2 h3
  sorry

end person2_speed_l149_149257


namespace badminton_members_count_l149_149212

def total_members := 30
def neither_members := 2
def both_members := 6

def members_play_badminton_and_tennis (B T : ℕ) : Prop :=
  B + T - both_members = total_members - neither_members

theorem badminton_members_count (B T : ℕ) (hbt : B = T) :
  members_play_badminton_and_tennis B T → B = 17 :=
by
  intros h
  sorry

end badminton_members_count_l149_149212


namespace whisky_replacement_l149_149970

variable (x : ℝ) -- Original quantity of whisky in the jar
variable (y : ℝ) -- Quantity of whisky replaced

-- Condition: A jar full of whisky contains 40% alcohol
-- Condition: After replacement, the percentage of alcohol is 24%
theorem whisky_replacement (h : 0 < x) : 
  0.40 * x - 0.40 * y + 0.19 * y = 0.24 * x → y = (16 / 21) * x :=
by
  intro h_eq
  -- Sorry for the proof
  sorry

end whisky_replacement_l149_149970


namespace problem1_problem2_l149_149182

def f (x : ℝ) : ℝ := abs (x - 1) - abs (x + 3)

-- Proof Problem 1
theorem problem1 (x : ℝ) (h : f x > 2) : x < -2 := sorry

-- Proof Problem 2
theorem problem2 (k : ℝ) (h : ∀ x : ℝ, -3 ≤ x ∧ x ≤ -1 → f x ≤ k * x + 1) : k ≤ -1 := sorry

end problem1_problem2_l149_149182


namespace find_speed_of_B_l149_149803

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l149_149803


namespace find_x_l149_149332

theorem find_x (x : ℚ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 :=
by
  intro h,
  sorry

end find_x_l149_149332


namespace slope_point_on_line_l149_149690

theorem slope_point_on_line (b : ℝ) (h1 : ∃ x, x + b = 30) (h2 : (b / (30 - b)) = 4) : b = 24 :=
  sorry

end slope_point_on_line_l149_149690


namespace find_speed_B_l149_149860

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l149_149860


namespace triangle_relation_l149_149913

theorem triangle_relation (A B C a b : ℝ) (h : 4 * A = B ∧ B = C) (hABC : A + B + C = 180) : 
  a^3 + b^3 = 3 * a * b^2 := 
by 
  sorry

end triangle_relation_l149_149913


namespace maximum_even_integers_of_odd_product_l149_149282

theorem maximum_even_integers_of_odd_product (a b c d e f g : ℕ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: d > 0) (h5: e > 0) (h6: f > 0) (h7: g > 0) (hprod : a * b * c * d * e * f * g % 2 = 1): 
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (d % 2 = 1) ∧ (e % 2 = 1) ∧ (f % 2 = 1) ∧ (g % 2 = 1) :=
sorry

end maximum_even_integers_of_odd_product_l149_149282


namespace remainder_of_17_pow_63_mod_7_l149_149557

theorem remainder_of_17_pow_63_mod_7 :
  17^63 % 7 = 6 :=
by {
  -- Condition: 17 ≡ 3 (mod 7)
  have h : 17 % 7 = 3 := by norm_num,
  -- Use the periodicity established in the powers of 3 modulo 7 to prove the statement
  -- Note: Leaving the proof part out as instructed
  sorry
}

end remainder_of_17_pow_63_mod_7_l149_149557


namespace inequality_proof_l149_149692

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
(h : Real.sqrt a + Real.sqrt b + Real.sqrt c = 3) :
  (a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) ≥ 3 / 2 := 
sorry

end inequality_proof_l149_149692


namespace least_n_l149_149625

theorem least_n (n : ℕ) (h : 0 < n ∧ (1 / n - 1 / (n + 1) < 1 / 15)) : n ≥ 4 :=
by sorry

end least_n_l149_149625


namespace remainder_349_div_13_l149_149561

theorem remainder_349_div_13 : 349 % 13 = 11 := 
by 
  sorry

end remainder_349_div_13_l149_149561


namespace correct_calculation_l149_149408

theorem correct_calculation (x y : ℝ) : -x^2 * y + 3 * x^2 * y = 2 * x^2 * y :=
by
  sorry

end correct_calculation_l149_149408


namespace max_D_n_l149_149402

-- Define the properties for each block
structure Block where
  shape : ℕ -- 1 for Square, 2 for Circular
  color : ℕ -- 1 for Red, 2 for Yellow
  city  : ℕ -- 1 for Nanchang, 2 for Beijing

-- The 8 blocks
def blocks : List Block := [
  { shape := 1, color := 1, city := 1 },
  { shape := 2, color := 1, city := 1 },
  { shape := 2, color := 2, city := 1 },
  { shape := 1, color := 2, city := 1 },
  { shape := 1, color := 1, city := 2 },
  { shape := 2, color := 1, city := 2 },
  { shape := 2, color := 2, city := 2 },
  { shape := 1, color := 2, city := 2 }
]

-- Define D_n counting function (to be implemented)
noncomputable def D_n (n : ℕ) : ℕ := sorry

-- Define the required proof
theorem max_D_n : 2 ≤ n → n ≤ 8 → ∃ k : ℕ, 2 ≤ k ∧ k ≤ 8 ∧ D_n k = 240 := sorry

end max_D_n_l149_149402


namespace smallest_blocks_required_l149_149750

theorem smallest_blocks_required (L H : ℕ) (block_height block_long block_short : ℕ) 
  (vert_joins_staggered : Prop) (consistent_end_finish : Prop) : 
  L = 120 → H = 10 → block_height = 1 → block_long = 3 → block_short = 1 → 
  (vert_joins_staggered) → (consistent_end_finish) → 
  ∃ n, n = 415 :=
by
  sorry

end smallest_blocks_required_l149_149750


namespace seunghyeon_pizza_diff_l149_149525

theorem seunghyeon_pizza_diff (S Y : ℕ) (h : S - 2 = Y + 7) : S - Y = 9 :=
by {
  sorry
}

end seunghyeon_pizza_diff_l149_149525


namespace inradius_of_triangle_l149_149350

theorem inradius_of_triangle (A p s r : ℝ) (h1 : A = 2 * p) (h2 : A = r * s) (h3 : p = 2 * s) : r = 4 :=
by
  sorry

end inradius_of_triangle_l149_149350


namespace isosceles_triangle_circles_distance_l149_149066

theorem isosceles_triangle_circles_distance (h α : ℝ) (hα : α ≤ π / 6) :
    let R := h / (2 * (Real.cos α)^2)
    let r := h * (Real.tan α) * (Real.tan (π / 4 - α / 2))
    let OO1 := h * (1 - 1 / (2 * (Real.cos α)^2) - (Real.tan α) * (Real.tan (π / 4 - α / 2)))
    OO1 = (2 * h * Real.sin (π / 12 - α / 2) * Real.cos (π / 12 + α / 2)) / (Real.cos α)^2 :=
    sorry

end isosceles_triangle_circles_distance_l149_149066


namespace student_b_speed_l149_149847

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l149_149847


namespace rationalize_denominator_l149_149377

theorem rationalize_denominator : 
  (∃ (x y : ℝ), x = real.sqrt 12 + real.sqrt 5 ∧ y = real.sqrt 3 + real.sqrt 5 
    ∧ (x / y) = (real.sqrt 15 - 1) / 2) :=
begin
  use [real.sqrt 12 + real.sqrt 5, real.sqrt 3 + real.sqrt 5],
  split,
  { refl },
  split,
  { refl },
  { sorry }
end

end rationalize_denominator_l149_149377


namespace plywood_perimeter_difference_l149_149104

theorem plywood_perimeter_difference :
  ∀ (length width : ℕ) (n : ℕ), 
    length = 6 ∧ width = 9 ∧ n = 6 → 
    ∃ (max_perimeter min_perimeter : ℕ), 
      (max_perimeter - min_perimeter = 10) ∧
      max_perimeter = 20 ∧ 
      min_perimeter = 10 :=
by
  sorry

end plywood_perimeter_difference_l149_149104


namespace bicycle_speed_B_l149_149772

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l149_149772


namespace total_time_spent_l149_149353

noncomputable def time_per_round : ℕ := 30
noncomputable def saturday_rounds : ℕ := 1 + 10
noncomputable def sunday_rounds : ℕ := 15
noncomputable def total_rounds : ℕ := saturday_rounds + sunday_rounds
noncomputable def total_time : ℕ := total_rounds * time_per_round

theorem total_time_spent :
  total_time = 780 := by sorry

end total_time_spent_l149_149353


namespace compound_proposition_l149_149224

theorem compound_proposition (Sn P Q : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → Sn n = 2 * n^2 + 3 * n + 1) →
  (∀ n : ℕ, n > 0 → Sn n = 2 * P n + 1) →
  (¬(∀ n, n > 0 → ∃ d, (P (n + 1) - P n) = d)) ∧ (∀ n, n > 0 → P n = Q (n - 1)) :=
by
  sorry

end compound_proposition_l149_149224


namespace initial_percentage_of_alcohol_l149_149418

theorem initial_percentage_of_alcohol 
  (P: ℝ)
  (h_condition1 : 18 * P / 100 = 21 * 17.14285714285715 / 100) : 
  P = 20 :=
by 
  sorry

end initial_percentage_of_alcohol_l149_149418


namespace at_least_one_ge_two_l149_149362

theorem at_least_one_ge_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let a := x + 1 / y
  let b := y + 1 / z
  let c := z + 1 / x
  a + b + c ≥ 6 → (a ≥ 2 ∨ b ≥ 2 ∨ c ≥ 2) :=
by
  intros
  let a := x + 1 / y
  let b := y + 1 / z
  let c := z + 1 / x
  sorry

end at_least_one_ge_two_l149_149362


namespace least_n_inequality_l149_149613

theorem least_n_inequality (n : ℕ) (hn : 0 < n) (ineq: 1 / n - 1 / (n + 1) < 1 / 15) :
  4 ≤ n :=
begin
  simp [lt_div_iff] at ineq,
  rw [sub_eq_add_neg, add_div_eq_mul_add_neg_div, ←lt_div_iff], 
  exact ineq
end

end least_n_inequality_l149_149613


namespace weight_of_7th_person_l149_149726

/--
There are 6 people in the elevator with an average weight of 152 lbs.
Another person enters the elevator, increasing the average weight to 151 lbs.
Prove that the weight of the 7th person is 145 lbs.
-/
theorem weight_of_7th_person
  (W : ℕ) (X : ℕ) (h1 : W / 6 = 152) (h2 : (W + X) / 7 = 151) :
  X = 145 :=
sorry

end weight_of_7th_person_l149_149726


namespace necessary_but_not_sufficient_condition_l149_149095

variable (a b : ℝ) (lna lnb : ℝ)

theorem necessary_but_not_sufficient_condition (h1 : lna < lnb) (h2 : lna = Real.log a) (h3 : lnb = Real.log b) :
  (a > 0 ∧ b > 0 ∧ a < b ∧ a ^ 3 < b ^ 3) ∧ ¬(a ^ 3 < b ^ 3 → 0 < a ∧ a < b ∧ 0 < b) :=
by {
  sorry
}

end necessary_but_not_sufficient_condition_l149_149095


namespace dog_catches_sheep_in_20_seconds_l149_149229

variable (v_sheep v_dog : ℕ) (d : ℕ)

def relative_speed (v_dog v_sheep : ℕ) := v_dog - v_sheep

def time_to_catch (d v_sheep v_dog : ℕ) : ℕ := d / (relative_speed v_dog v_sheep)

theorem dog_catches_sheep_in_20_seconds
  (h1 : v_sheep = 16)
  (h2 : v_dog = 28)
  (h3 : d = 240) :
  time_to_catch d v_sheep v_dog = 20 := by {
  sorry
}

end dog_catches_sheep_in_20_seconds_l149_149229


namespace James_bought_3_CDs_l149_149036

theorem James_bought_3_CDs :
  ∃ (cd1 cd2 cd3 : ℝ), cd1 = 1.5 ∧ cd2 = 1.5 ∧ cd3 = 2 * cd1 ∧ cd1 + cd2 + cd3 = 6 ∧ 3 = 3 :=
by
  sorry

end James_bought_3_CDs_l149_149036


namespace apple_order_for_month_l149_149580

def Chandler_apples (week : ℕ) : ℕ :=
  23 + 2 * week

def Lucy_apples (week : ℕ) : ℕ :=
  19 - week

def Ross_apples : ℕ :=
  15

noncomputable def total_apples : ℕ :=
  (Chandler_apples 0 + Chandler_apples 1 + Chandler_apples 2 + Chandler_apples 3) +
  (Lucy_apples 0 + Lucy_apples 1 + Lucy_apples 2 + Lucy_apples 3) +
  (Ross_apples * 4)

theorem apple_order_for_month : total_apples = 234 := by
  sorry

end apple_order_for_month_l149_149580


namespace sqrt_condition_l149_149016

theorem sqrt_condition (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 1)) ↔ x ≥ 1 := by
  sorry

end sqrt_condition_l149_149016


namespace rectangular_prism_diagonals_l149_149145

theorem rectangular_prism_diagonals (length width height : ℕ) (length_eq : length = 4) (width_eq : width = 3) (height_eq : height = 2) : 
  ∃ (total_diagonals : ℕ), total_diagonals = 16 :=
by
  let face_diagonals := 12
  let space_diagonals := 4
  let total_diagonals := face_diagonals + space_diagonals
  use total_diagonals
  sorry

end rectangular_prism_diagonals_l149_149145


namespace plywood_cut_difference_l149_149114

theorem plywood_cut_difference :
  let original_width := 6
  let original_height := 9
  let total_area := original_width * original_height
  let num_pieces := 6
  let area_per_piece := total_area / num_pieces
  -- Let possible perimeters based on given conditions
  let max_perimeter := 20
  let min_perimeter := 15
  max_perimeter - min_perimeter = 5 :=
by
  sorry

end plywood_cut_difference_l149_149114


namespace increase_by_percentage_proof_l149_149417

def initial_number : ℕ := 150
def percentage_increase : ℝ := 0.4
def final_number : ℕ := 210

theorem increase_by_percentage_proof :
  initial_number + (percentage_increase * initial_number) = final_number :=
by
  sorry

end increase_by_percentage_proof_l149_149417


namespace circle_center_l149_149530

theorem circle_center : 
  ∃ (h k : ℝ), (h, k) = (1, -2) ∧ 
    ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y - 4 = 0 ↔ (x - h)^2 + (y - k)^2 = 9 :=
by
  sorry

end circle_center_l149_149530


namespace rotor_permutations_l149_149451

-- Define the factorial function for convenience
def fact : Nat → Nat
| 0     => 1
| (n + 1) => (n + 1) * fact n

-- The main statement to prove
theorem rotor_permutations : (fact 5) / ((fact 2) * (fact 2)) = 30 := by
  sorry

end rotor_permutations_l149_149451


namespace negation_of_exists_l149_149195

theorem negation_of_exists (p : Prop) : 
  (∃ (x₀ : ℝ), x₀ > 0 ∧ |x₀| ≤ 2018) ↔ 
  ¬(∀ (x : ℝ), x > 0 → |x| > 2018) :=
by sorry

end negation_of_exists_l149_149195


namespace sum_of_terms_l149_149975

-- Defining the arithmetic progression
def arithmetic_progression (a d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

-- Given conditions
theorem sum_of_terms (a d : ℕ) (h : (a + 3 * d) + (a + 11 * d) = 20) :
  12 * (a + 11 * d) / 2 = 60 :=
by
  sorry

end sum_of_terms_l149_149975


namespace championship_titles_l149_149465

theorem championship_titles {S T : ℕ} (h_S : S = 4) (h_T : T = 3) : S^T = 64 := by
  rw [h_S, h_T]
  norm_num

end championship_titles_l149_149465


namespace find_b_when_a_is_1600_l149_149938

theorem find_b_when_a_is_1600 :
  ∀ (a b : ℝ), (a * b = 400) ∧ ((2 * a) * b = 600) → (1600 * b = 600) → b = 0.375 :=
by
  intro a b
  intro h
  sorry

end find_b_when_a_is_1600_l149_149938


namespace arithmetic_sequence_value_l149_149677

variable (a : ℕ → ℤ) (d : ℤ)
variable (h1 : a 1 + a 4 + a 7 = 39)
variable (h2 : a 2 + a 5 + a 8 = 33)
variable (h_arith : ∀ n, a (n + 1) = a n + d)

theorem arithmetic_sequence_value : a 5 + a 8 + a 11 = 15 := by
  sorry

end arithmetic_sequence_value_l149_149677


namespace jogger_ahead_of_train_l149_149752

theorem jogger_ahead_of_train (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (time_to_pass : ℝ) 
  (h1 : jogger_speed = 9) 
  (h2 : train_speed = 45) 
  (h3 : train_length = 100) 
  (h4 : time_to_pass = 34) : 
  ∃ d : ℝ, d = 240 :=
by
  sorry

end jogger_ahead_of_train_l149_149752


namespace BoatWorks_total_canoes_by_April_l149_149288

def BoatWorksCanoes : ℕ → ℕ
| 0 => 5
| (n+1) => 2 * BoatWorksCanoes n

theorem BoatWorks_total_canoes_by_April : (BoatWorksCanoes 0) + (BoatWorksCanoes 1) + (BoatWorksCanoes 2) + (BoatWorksCanoes 3) = 75 :=
by
  sorry

end BoatWorks_total_canoes_by_April_l149_149288


namespace time_calculation_correct_l149_149214

theorem time_calculation_correct :
  let start_hour := 3
  let start_minute := 0
  let start_second := 0
  let hours_to_add := 158
  let minutes_to_add := 55
  let seconds_to_add := 32
  let total_seconds := seconds_to_add + minutes_to_add * 60 + hours_to_add * 3600
  let new_hour := (start_hour + (total_seconds / 3600) % 12) % 12
  let new_minute := (start_minute + (total_seconds / 60) % 60) % 60
  let new_second := (start_second + total_seconds % 60) % 60
  let A := new_hour
  let B := new_minute
  let C := new_second
  A + B + C = 92 :=
by
  sorry

end time_calculation_correct_l149_149214


namespace proportional_function_range_l149_149004

theorem proportional_function_range (m : ℝ) (h : ∀ x : ℝ, (x < 0 → (1 - m) * x > 0) ∧ (x > 0 → (1 - m) * x < 0)) : m > 1 :=
by sorry

end proportional_function_range_l149_149004


namespace least_value_in_valid_set_l149_149691

open Finset

def is_valid_set (T : Finset ℕ) : Prop :=
    T.card = 7 ∧
    (∀ {x y : ℕ}, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0)) ∧
    (3 ≤ (card (filter Nat.prime T)))

theorem least_value_in_valid_set : ∀ (T : Finset ℕ), is_valid_set T → ∃ m, m ∈ T ∧ m = 3 :=
by
  intro T hT
  sorry

end least_value_in_valid_set_l149_149691


namespace student_b_speed_l149_149823

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l149_149823


namespace student_B_speed_l149_149810

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l149_149810


namespace stratified_sampling_class2_l149_149161

theorem stratified_sampling_class2 (students_class1 : ℕ) (students_class2 : ℕ) (total_samples : ℕ) (h1 : students_class1 = 36) (h2 : students_class2 = 42) (h_tot : total_samples = 13) : 
  (students_class2 / (students_class1 + students_class2) * total_samples = 7) :=
by
  sorry

end stratified_sampling_class2_l149_149161


namespace find_a_and_vertices_find_y_range_find_a_range_l149_149892

noncomputable def quadratic_function (x a : ℝ) : ℝ :=
  x^2 - 6 * a * x + 9

theorem find_a_and_vertices (a : ℝ) :
  quadratic_function 2 a = 7 →
  a = 1 / 2 ∧
  (3 * a, quadratic_function (3 * a) a) = (3 / 2, 27 / 4) :=
sorry

theorem find_y_range (x a : ℝ) :
  a = 1 / 2 →
  -1 ≤ x ∧ x < 3 →
  27 / 4 ≤ quadratic_function x a ∧ quadratic_function x a ≤ 13 :=
sorry

theorem find_a_range (a : ℝ) (x1 x2 : ℝ) :
  (3 * a - 2 ≤ x1 ∧ x1 ≤ 5 ∧ 3 * a - 2 ≤ x2 ∧ x2 ≤ 5) →
  (x1 ≥ 3 ∧ x2 ≥ 3 → quadratic_function x1 a - quadratic_function x2 a ≤ 9 * a^2 + 20) →
  1 / 6 ≤ a ∧ a ≤ 1 :=
sorry

end find_a_and_vertices_find_y_range_find_a_range_l149_149892


namespace ratio_of_liquid_p_to_q_initial_l149_149569

noncomputable def initial_ratio_of_p_to_q : ℚ :=
  let p := 20
  let q := 15
  p / q

theorem ratio_of_liquid_p_to_q_initial
  (p q : ℚ)
  (h1 : p + q = 35)
  (h2 : p / (q + 13) = 5 / 7) :
  p / q = 4 / 3 := by
  sorry

end ratio_of_liquid_p_to_q_initial_l149_149569


namespace remainder_17_pow_63_mod_7_l149_149544

theorem remainder_17_pow_63_mod_7 : (17 ^ 63) % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l149_149544


namespace range_of_a_l149_149344

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x^3 - 3 * a^2 * x + 1 ≠ 3)) 
  → (-1 < a ∧ a < 1) := 
by
  sorry

end range_of_a_l149_149344


namespace tank_filled_in_96_minutes_l149_149233

-- conditions
def pipeA_fill_time : ℝ := 6
def pipeB_empty_time : ℝ := 24
def time_with_both_pipes_open : ℝ := 96

-- rate computations and final proof
noncomputable def pipeA_fill_rate : ℝ := 1 / pipeA_fill_time
noncomputable def pipeB_empty_rate : ℝ := 1 / pipeB_empty_time
noncomputable def net_fill_rate : ℝ := pipeA_fill_rate - pipeB_empty_rate
noncomputable def tank_filled_in_time_with_both : ℝ := time_with_both_pipes_open * net_fill_rate

theorem tank_filled_in_96_minutes (HA : pipeA_fill_time = 6) (HB : pipeB_empty_time = 24)
  (HT : time_with_both_pipes_open = 96) : tank_filled_in_time_with_both = 1 :=
by
  sorry

end tank_filled_in_96_minutes_l149_149233


namespace todd_money_left_l149_149079

def candy_bar_cost : ℝ := 2.50
def chewing_gum_cost : ℝ := 1.50
def soda_cost : ℝ := 3
def discount : ℝ := 0.20
def initial_money : ℝ := 50
def number_of_candy_bars : ℕ := 7
def number_of_chewing_gum : ℕ := 5
def number_of_soda : ℕ := 3

noncomputable def total_candy_bar_cost : ℝ := number_of_candy_bars * candy_bar_cost
noncomputable def total_chewing_gum_cost : ℝ := number_of_chewing_gum * chewing_gum_cost
noncomputable def total_soda_cost : ℝ := number_of_soda * soda_cost
noncomputable def discount_amount : ℝ := total_soda_cost * discount
noncomputable def discounted_soda_cost : ℝ := total_soda_cost - discount_amount
noncomputable def total_cost : ℝ := total_candy_bar_cost + total_chewing_gum_cost + discounted_soda_cost
noncomputable def money_left : ℝ := initial_money - total_cost

theorem todd_money_left : money_left = 17.80 :=
by sorry

end todd_money_left_l149_149079


namespace min_value_fraction_sum_l149_149686

theorem min_value_fraction_sum (m n : ℝ) (h₁ : 0 < m) (h₂ : 0 < n) (h₃ : 2 * m + n = 1) : 
  (1 / m + 2 / n) ≥ 8 :=
sorry

end min_value_fraction_sum_l149_149686


namespace max_partitioned_test_plots_is_78_l149_149981

def field_length : ℕ := 52
def field_width : ℕ := 24
def total_fence : ℕ := 1994
def gcd_field_dimensions : ℕ := Nat.gcd field_length field_width

-- Since gcd_field_dimensions divides both 52 and 24 and gcd_field_dimensions = 4
def possible_side_lengths : List ℕ := [1, 2, 4]

noncomputable def max_square_plots : ℕ :=
  let max_plots (a : ℕ) : ℕ := (field_length / a) * (field_width / a)
  let valid_fence (a : ℕ) : Bool :=
    let vertical_fence := (field_length / a - 1) * field_width
    let horizontal_fence := (field_width / a - 1) * field_length
    vertical_fence + horizontal_fence ≤ total_fence
  let valid_lengths := possible_side_lengths.filter valid_fence
  valid_lengths.map max_plots |>.maximum? |>.getD 0

theorem max_partitioned_test_plots_is_78 : max_square_plots = 78 := by
  sorry

end max_partitioned_test_plots_is_78_l149_149981


namespace events_A_B_mutually_exclusive_events_A_C_independent_l149_149077

-- Definitions for events A, B, and C
def event_A (x y : ℕ) : Prop := x + y = 7
def event_B (x y : ℕ) : Prop := (x * y) % 2 = 1
def event_C (x : ℕ) : Prop := x > 3

-- Proof problems to decide mutual exclusivity and independence
theorem events_A_B_mutually_exclusive :
  ∀ (x y : ℕ), event_A x y → ¬ event_B x y := 
by sorry

theorem events_A_C_independent :
  ∀ (x y : ℕ), (event_A x y) ↔ ∀ x y, event_C x ↔ event_A x y ∧ event_C x := 
by sorry

end events_A_B_mutually_exclusive_events_A_C_independent_l149_149077


namespace number_is_minus_three_l149_149020

variable (x a : ℝ)

theorem number_is_minus_three (h1 : a = 0.5) (h2 : x / (a - 3) = 3 / (a + 2)) : x = -3 :=
by
  sorry

end number_is_minus_three_l149_149020


namespace problem1_problem2_l149_149317

-- Definitions for the conditions
variables {A B C : ℝ}
variables {a b c S : ℝ}

-- Problem 1: Proving the value of side "a" given certain conditions
theorem problem1 (h₁ : S = (1 / 2) * a * b * Real.sin C) (h₂ : a^2 = 4 * Real.sqrt 3 * S)
  (h₃ : C = Real.pi / 3) (h₄ : b = 1) : a = 3 := by
  sorry

-- Problem 2: Proving the measure of angle "A" given certain conditions
theorem problem2 (h₁ : S = (1 / 2) * a * b * Real.sin C) (h₂ : a^2 = 4 * Real.sqrt 3 * S)
  (h₃ : c / b = 2 + Real.sqrt 3) : A = Real.pi / 3 := by
  sorry

end problem1_problem2_l149_149317


namespace brother_pays_correct_amount_l149_149930

-- Definition of constants and variables
def friend_per_day := 5
def cousin_per_day := 4
def total_amount_collected := 119
def days := 7
def brother_per_day := 8

-- Statement of the theorem to be proven
theorem brother_pays_correct_amount :
  friend_per_day * days + cousin_per_day * days + brother_per_day * days = total_amount_collected :=
by {
  sorry
}

end brother_pays_correct_amount_l149_149930


namespace diagonal_length_l149_149074

noncomputable def length_of_diagonal (a b c : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 + c^2)

theorem diagonal_length
  (a b c : ℝ)
  (h1 : 2 * (a * b + a * c + b * c) = 11)
  (h2 : 4 * (a + b + c) = 24) :
  length_of_diagonal a b c = 5 := by
  sorry

end diagonal_length_l149_149074


namespace largest_y_coordinate_on_graph_l149_149584

theorem largest_y_coordinate_on_graph :
  ∀ x y : ℝ, (x / 7) ^ 2 + ((y - 3) / 5) ^ 2 = 0 → y ≤ 3 := 
by
  intro x y h
  sorry

end largest_y_coordinate_on_graph_l149_149584


namespace renovation_days_l149_149978

/-
Conditions:
1. Cost to hire a company: 50000 rubles
2. Cost of buying materials: 20000 rubles
3. Husband's daily wage: 2000 rubles
4. Wife's daily wage: 1500 rubles
Question:
How many workdays can they spend on the renovation to make it more cost-effective?
-/

theorem renovation_days (cost_hire_company cost_materials : ℕ) 
  (husband_daily_wage wife_daily_wage : ℕ) 
  (more_cost_effective_days : ℕ) :
  cost_hire_company = 50000 → 
  cost_materials = 20000 → 
  husband_daily_wage = 2000 → 
  wife_daily_wage = 1500 → 
  more_cost_effective_days = 8 :=
by
  intros
  sorry

end renovation_days_l149_149978


namespace geometric_sequence_angle_count_l149_149166

theorem geometric_sequence_angle_count :
  (∃ θs : Finset ℝ, (∀ θ ∈ θs, 0 < θ ∧ θ < 2 * π ∧ ¬ ∃ k : ℕ, θ = k * (π / 2)) 
                    ∧ θs.card = 4
                    ∧ ∀ θ ∈ θs, ∃ a b c : ℝ, (a, b, c) = (Real.sin θ, Real.cos θ, Real.tan θ) 
                                             ∨ (a, b) = (Real.sin θ, Real.tan θ) 
                                             ∨ (a, b) = (Real.cos θ, Real.tan θ)
                                             ∧ b = a * c) :=
sorry

end geometric_sequence_angle_count_l149_149166


namespace color_of_217th_marble_l149_149991

-- Definitions of conditions
def total_marbles := 240
def pattern_length := 15
def red_marbles := 6
def blue_marbles := 5
def green_marbles := 4
def position := 217

-- Lean 4 statement
theorem color_of_217th_marble :
  (position % pattern_length ≤ red_marbles) :=
by sorry

end color_of_217th_marble_l149_149991


namespace closed_pipe_length_l149_149903

def speed_of_sound : ℝ := 333
def fundamental_frequency : ℝ := 440

theorem closed_pipe_length :
  ∃ l : ℝ, l = 0.189 ∧ fundamental_frequency = speed_of_sound / (4 * l) :=
by
  sorry

end closed_pipe_length_l149_149903


namespace solve_conjugate_l149_149479
open Complex

-- Problem definition:
def Z (a : ℝ) : ℂ := ⟨a, 1⟩  -- Z = a + i

def conj_Z (a : ℝ) : ℂ := ⟨a, -1⟩  -- conjugate of Z

theorem solve_conjugate (a : ℝ) (h : Z a + conj_Z a = 4) : conj_Z 2 = 2 - I := by
  sorry

end solve_conjugate_l149_149479


namespace seq_20_eq_5_over_7_l149_149323

theorem seq_20_eq_5_over_7 :
  ∃ (a : ℕ → ℚ), 
    a 1 = 6 / 7 ∧ 
    (∀ n, (0 ≤ a n ∧ a n < 1) → 
      (a (n + 1) = if a n < 1 / 2 then 2 * a n else 2 * a n - 1)) ∧ 
    a 20 = 5 / 7 := 
sorry

end seq_20_eq_5_over_7_l149_149323


namespace trajectory_of_square_is_line_l149_149918

open Complex

theorem trajectory_of_square_is_line (z : ℂ) (h : z.re = z.im) : ∃ c : ℝ, z^2 = Complex.I * (c : ℂ) :=
by
  sorry

end trajectory_of_square_is_line_l149_149918


namespace profit_percentage_l149_149666

theorem profit_percentage (SP : ℝ) (h : SP > 0) (CP : ℝ) (h1 : CP = 0.96 * SP) :
  (SP - CP) / CP * 100 = 4.17 :=
by
  sorry

end profit_percentage_l149_149666


namespace plywood_cut_difference_l149_149106

/-- A proof problem to determine the positive difference between the greatest possible
perimeter and the least possible perimeter of congruent pieces resulting from cutting 
a 6-foot by 9-foot rectangular plywood into 6 congruent rectangles with no wood leftover 
or lost --/
theorem plywood_cut_difference :
  ∃ (perimeter_max perimeter_min : ℕ), 
  let piece1 := 1 * 9
  let piece2 := 1 * 6
  let piece3 := 2 * 3
  let perimeter1 := 2 * (1 + 9)
  let perimeter2 := 2 * (1 + 6)
  let perimeter3 := 2 * (2 + 3)
  perimeter_max = perimeter1 ∧
  perimeter_min = perimeter3 ∧
  (perimeter_max - perimeter_min) = 10 :=
sorry

end plywood_cut_difference_l149_149106


namespace student_b_speed_l149_149849

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l149_149849


namespace sum_of_interior_angles_n_plus_2_l149_149710

-- Define the sum of the interior angles formula for a convex polygon
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the degree measure of the sum of the interior angles of a convex polygon with n sides being 1800
def sum_of_n_sides_is_1800 (n : ℕ) : Prop := sum_of_interior_angles n = 1800

-- Translate the proof problem as a theorem statement in Lean
theorem sum_of_interior_angles_n_plus_2 (n : ℕ) (h: sum_of_n_sides_is_1800 n) : 
  sum_of_interior_angles (n + 2) = 2160 :=
sorry

end sum_of_interior_angles_n_plus_2_l149_149710


namespace problem_AD_l149_149473

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.sin x + Real.cos x

open Real

theorem problem_AD :
  (∀ x, 0 < x ∧ x < π / 4 → f x < f (x + 0.01) ∧ g x < g (x + 0.01)) ∧
  (∃ x, x = π / 4 ∧ f x + g x = 1 / 2 + sqrt 2) :=
by
  sorry

end problem_AD_l149_149473


namespace find_a2019_l149_149192

-- Arithmetic sequence
def a (n : ℕ) : ℤ := sorry -- to be defined later

-- Given conditions
def sum_first_five_terms (a: ℕ → ℤ) : Prop := a 1 + a 2 + a 3 + a 4 + a 5 = 15
def term_six (a: ℕ → ℤ) : Prop := a 6 = 6

-- Question (statement to be proved)
def term_2019 (a: ℕ → ℤ) : Prop := a 2019 = 2019

-- Main theorem to be proved
theorem find_a2019 (a: ℕ → ℤ) 
  (h1 : sum_first_five_terms a)
  (h2 : term_six a) : 
  term_2019 a := 
by
  sorry

end find_a2019_l149_149192


namespace probability_of_seeing_red_light_l149_149149

def red_light_duration : ℝ := 30
def yellow_light_duration : ℝ := 5
def green_light_duration : ℝ := 40

def total_cycle_duration : ℝ := red_light_duration + yellow_light_duration + green_light_duration

theorem probability_of_seeing_red_light :
  (red_light_duration / total_cycle_duration) = 30 / 75 := by
  sorry

end probability_of_seeing_red_light_l149_149149


namespace gumballs_difference_l149_149972

theorem gumballs_difference :
  ∃ (x_min x_max : ℕ), 
    19 ≤ (16 + 12 + x_min) / 3 ∧ (16 + 12 + x_min) / 3 ≤ 25 ∧
    19 ≤ (16 + 12 + x_max) / 3 ∧ (16 + 12 + x_max) / 3 ≤ 25 ∧
    (x_max - x_min = 18) :=
by
  sorry

end gumballs_difference_l149_149972


namespace plywood_perimeter_difference_l149_149130

/--
Given a 6-foot by 9-foot rectangular piece of plywood cut into 6 congruent rectangles 
with no wood left over and no wood lost due to the cuts,
prove that the positive difference between the greatest and the least perimeter of a single piece is 11 feet.
-/
theorem plywood_perimeter_difference :
  ∃ (rectangles : List (ℕ × ℕ)), 
  (∀ r ∈ rectangles, r.fst * r.snd = 9 * 6 / 6) ∧
  (Greatest (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) - 
  (Least (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) = 11 :=
by
  sorry

end plywood_perimeter_difference_l149_149130


namespace math_problem_proof_l149_149644

noncomputable def least_n : ℕ := 4

theorem math_problem_proof : 
  ∃ n ≥ 1, ((1 : ℝ) / n) - ((1 : ℝ) / (n + 1)) < 1 / 15 ∧ 
           ∀ m ≥ 1, (((1 : ℝ) / m - (1 : ℝ) / (m + 1)) < 1 / 15) → 
           least_n ≤ m := 
by
  use least_n
  sorry

end math_problem_proof_l149_149644


namespace determine_machines_in_first_group_l149_149945

noncomputable def machines_in_first_group (x r : ℝ) : Prop :=
  (x * r * 6 = 1) ∧ (12 * r * 4 = 1)

theorem determine_machines_in_first_group (x r : ℝ) (h : machines_in_first_group x r) :
  x = 8 :=
by
  sorry

end determine_machines_in_first_group_l149_149945


namespace output_of_program_l149_149407

def loop_until (i S : ℕ) : ℕ :=
if i < 9 then S
else loop_until (i - 1) (S * i)

theorem output_of_program : loop_until 11 1 = 990 :=
sorry

end output_of_program_l149_149407


namespace seashells_found_l149_149164

theorem seashells_found (C B : ℤ) (h1 : 9 * B = 7 * C) (h2 : B = C - 12) : C = 54 :=
by
  sorry

end seashells_found_l149_149164


namespace find_speed_B_l149_149859

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l149_149859


namespace particle_probability_at_2_3_after_5_moves_l149_149277

theorem particle_probability_at_2_3_after_5_moves:
  ∃ (C : ℕ), C = Nat.choose 5 2 ∧
  (1/2 ^ 5 * C) = (Nat.choose 5 2) * ((1/2: ℝ) ^ 5) := by
sorry

end particle_probability_at_2_3_after_5_moves_l149_149277


namespace neither_necessary_nor_sufficient_l149_149416

theorem neither_necessary_nor_sufficient (x : ℝ) :
  ¬ ((-1 < x ∧ x < 2) → (|x - 2| < 1)) ∧ ¬ ((|x - 2| < 1) → (-1 < x ∧ x < 2)) :=
by
  sorry

end neither_necessary_nor_sufficient_l149_149416


namespace find_speed_of_B_l149_149755

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l149_149755


namespace least_n_l149_149603

theorem least_n (n : ℕ) (h : (1 : ℝ) / n - (1 / (n + 1)) < (1 / 15)) : n = 4 :=
sorry

end least_n_l149_149603


namespace identity_completion_factorize_polynomial_equilateral_triangle_l149_149382

-- Statement 1: Prove that a^3 - b^3 + a^2 b - ab^2 = (a - b)(a + b)^2 
theorem identity_completion (a b : ℝ) : a^3 - b^3 + a^2 * b - a * b^2 = (a - b) * (a + b)^2 :=
sorry

-- Statement 2: Prove that 4x^2 - 2x - y^2 - y = (2x + y)(2x - y - 1)
theorem factorize_polynomial (x y : ℝ) : 4 * x^2 - 2 * x - y^2 - y = (2 * x + y) * (2 * x - y - 1) :=
sorry

-- Statement 3: Given a^2 + b^2 + 2c^2 - 2ac - 2bc = 0, Prove that triangle ABC is equilateral
theorem equilateral_triangle (a b c : ℝ) (h : a^2 + b^2 + 2 * c^2 - 2 * a * c - 2 * b * c = 0) : a = b ∧ b = c :=
sorry

end identity_completion_factorize_polynomial_equilateral_triangle_l149_149382


namespace total_distance_joseph_ran_l149_149356

-- Defining the conditions
def distance_per_day : ℕ := 900
def days_run : ℕ := 3

-- The proof problem statement
theorem total_distance_joseph_ran :
  (distance_per_day * days_run) = 2700 :=
by
  sorry

end total_distance_joseph_ran_l149_149356


namespace plywood_perimeter_difference_l149_149134

/--
Given a 6-foot by 9-foot rectangular piece of plywood cut into 6 congruent rectangles 
with no wood left over and no wood lost due to the cuts,
prove that the positive difference between the greatest and the least perimeter of a single piece is 11 feet.
-/
theorem plywood_perimeter_difference :
  ∃ (rectangles : List (ℕ × ℕ)), 
  (∀ r ∈ rectangles, r.fst * r.snd = 9 * 6 / 6) ∧
  (Greatest (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) - 
  (Least (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) = 11 :=
by
  sorry

end plywood_perimeter_difference_l149_149134


namespace acute_angle_probability_l149_149933

/-- 
  Given a clock with two hands (the hour and the minute hand) and assuming:
  1. The hour hand is always pointing at 12 o'clock.
  2. The angle between the hands is acute if the minute hand is either in the first quadrant 
     (between 12 and 3 o'clock) or in the fourth quadrant (between 9 and 12 o'clock).

  Prove that the probability that the angle between the hands is acute is 1/2.
-/
theorem acute_angle_probability : 
  let total_intervals := 12
  let favorable_intervals := 6
  (favorable_intervals / total_intervals : ℝ) = (1 / 2 : ℝ) :=
by
  sorry

end acute_angle_probability_l149_149933


namespace max_sum_sqrt_expr_max_sum_sqrt_expr_attained_l149_149039

open Real

theorem max_sum_sqrt_expr (a b c : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h_sum : a + b + c = 8) :
  sqrt (3 * a^2 + 1) + sqrt (3 * b^2 + 1) + sqrt (3 * c^2 + 1) ≤ sqrt 201 :=
  sorry

theorem max_sum_sqrt_expr_attained : sqrt (3 * (8/3)^2 + 1) + sqrt (3 * (8/3)^2 + 1) + sqrt (3 * (8/3)^2 + 1) = sqrt 201 :=
  sorry

end max_sum_sqrt_expr_max_sum_sqrt_expr_attained_l149_149039


namespace find_speed_of_B_l149_149756

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l149_149756


namespace mrs_santiago_more_roses_l149_149931

theorem mrs_santiago_more_roses :
  58 - 24 = 34 :=
by 
  sorry

end mrs_santiago_more_roses_l149_149931


namespace binary_to_decimal_110101_l149_149293

theorem binary_to_decimal_110101 :
  (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 53) :=
by
  sorry

end binary_to_decimal_110101_l149_149293


namespace xy_square_sum_l149_149940

theorem xy_square_sum (x y : ℝ) (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 132) : x^2 + y^2 = 1336 :=
by
  sorry

end xy_square_sum_l149_149940


namespace max_min_sum_l149_149072

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log (x + 1) / Real.log 2

theorem max_min_sum : 
  (f 0 + f 1) = 4 := 
by
  sorry

end max_min_sum_l149_149072


namespace math_problem_proof_l149_149645

noncomputable def least_n : ℕ := 4

theorem math_problem_proof : 
  ∃ n ≥ 1, ((1 : ℝ) / n) - ((1 : ℝ) / (n + 1)) < 1 / 15 ∧ 
           ∀ m ≥ 1, (((1 : ℝ) / m - (1 : ℝ) / (m + 1)) < 1 / 15) → 
           least_n ≤ m := 
by
  use least_n
  sorry

end math_problem_proof_l149_149645


namespace remainder_of_17_pow_63_mod_7_l149_149556

theorem remainder_of_17_pow_63_mod_7 :
  17^63 % 7 = 6 :=
by {
  -- Condition: 17 ≡ 3 (mod 7)
  have h : 17 % 7 = 3 := by norm_num,
  -- Use the periodicity established in the powers of 3 modulo 7 to prove the statement
  -- Note: Leaving the proof part out as instructed
  sorry
}

end remainder_of_17_pow_63_mod_7_l149_149556


namespace value_depletion_rate_l149_149430

theorem value_depletion_rate (P F : ℝ) (t : ℝ) (r : ℝ) (h₁ : P = 1100) (h₂ : F = 891) (h₃ : t = 2) (decay_formula : F = P * (1 - r) ^ t) : r = 0.1 :=
by 
  sorry

end value_depletion_rate_l149_149430


namespace adult_tickets_l149_149404

theorem adult_tickets (A C : ℕ) (h1 : A + C = 130) (h2 : 12 * A + 4 * C = 840) : A = 40 :=
by {
  -- Proof omitted
  sorry
}

end adult_tickets_l149_149404


namespace factor_expression_l149_149455

theorem factor_expression (a b c : ℝ) :
  ((a^4 - b^4)^3 + (b^4 - c^4)^3 + (c^4 - a^4)^3) /
  ((a^2 - b^2)^3 + (b^2 - c^2)^3 + (c^2 - a^2)^3) =
  (a^2 + b^2) * (b^2 + c^2) * (c^2 + a^2) :=
by
  sorry

end factor_expression_l149_149455


namespace number_of_permissible_sandwiches_l149_149947

theorem number_of_permissible_sandwiches (b m c : ℕ) (h : b = 5) (me : m = 7) (ch : c = 6) 
  (no_ham_cheddar : ∀ bread, ¬(bread = ham ∧ cheese = cheddar))
  (no_turkey_swiss : ∀ bread, ¬(bread = turkey ∧ cheese = swiss)) : 
  5 * 7 * 6 - (5 * 1 * 1) - (5 * 1 * 1) = 200 := 
by 
  sorry

end number_of_permissible_sandwiches_l149_149947


namespace student_b_speed_l149_149819

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l149_149819


namespace least_n_satisfies_condition_l149_149621

theorem least_n_satisfies_condition :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (m < n) → ¬(1 / (m : ℝ) - 1 / (m + 1) < 1 / 15)) ∧
    (1 / (n : ℝ) - 1 / (n + 1) < 1 / 15) :=
sorry

end least_n_satisfies_condition_l149_149621


namespace election_threshold_l149_149093

theorem election_threshold (total_votes geoff_percent_more_votes : ℕ) (geoff_vote_percent : ℚ) (geoff_votes_needed extra_votes_needed : ℕ) (threshold_percent : ℚ) :
  total_votes = 6000 → 
  geoff_vote_percent = 0.5 → 
  geoff_votes_needed = (geoff_vote_percent / 100) * total_votes →
  extra_votes_needed = 3000 → 
  (geoff_votes_needed + extra_votes_needed) / total_votes * 100 = threshold_percent →
  threshold_percent = 50.5 := 
by
  intros total_votes_eq geoff_vote_percent_eq geoff_votes_needed_eq extra_votes_needed_eq threshold_eq
  sorry

end election_threshold_l149_149093


namespace speed_of_student_B_l149_149834

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l149_149834


namespace base7_to_base10_proof_l149_149946

theorem base7_to_base10_proof (c d : ℕ) (h1 : 764 = 4 * 100 + c * 10 + d) : (c * d) / 20 = 6 / 5 :=
by
  sorry

end base7_to_base10_proof_l149_149946


namespace count_permutations_l149_149357

theorem count_permutations : 
  (∃ (b : Fin 10 → Fin 11), 
    (∀ i, b i ∈ Finset.univ) ∧ 
    b 3 = 0 ∧ 
    (∀ i, i < 3 → b i > b (i + 1)) ∧ 
    (∀ j, j > 3 → b (j - 1) < b j)) → 
  (Finset.card {b : Fin 10 → Fin 11 | 
    (∀ i, b i ∈ Finset.univ) ∧ 
    b 3 = 0 ∧ 
    (∀ i, i < 3 → b i > b (i + 1)) ∧ 
    (∀ j, j > 3 → b (j - 1) < b j)} = 84) :=
by 
  sorry

end count_permutations_l149_149357


namespace speed_of_student_B_l149_149844

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l149_149844


namespace maximize_profit_l149_149588

-- Define the variables
variables (x y a b : ℝ)
variables (P : ℝ)

-- Define the conditions and the proof goal
theorem maximize_profit
  (h1 : x + 3 * y = 240)
  (h2 : 2 * x + y = 130)
  (h3 : a + b = 100)
  (h4 : a ≥ 4 * b)
  (ha : a = 80)
  (hb : b = 20) :
  x = 30 ∧ y = 70 ∧ P = (40 * a + 90 * b) - (30 * a + 70 * b) := 
by
  -- We assume the solution steps are solved correctly as provided
  sorry

end maximize_profit_l149_149588


namespace student_B_speed_l149_149774

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l149_149774


namespace compute_f_g_f_3_l149_149361

def f (x : ℤ) : ℤ := 5 * x + 5
def g (x : ℤ) : ℤ := 6 * x + 4

theorem compute_f_g_f_3 : f (g (f 3)) = 625 := sorry

end compute_f_g_f_3_l149_149361


namespace find_speed_of_B_l149_149805

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l149_149805


namespace tickets_total_l149_149345

theorem tickets_total (T : ℝ) (h1 : T / 2 + (T / 2) / 4 = 3600) : T = 5760 :=
by
  sorry

end tickets_total_l149_149345


namespace spring_work_compression_l149_149019

theorem spring_work_compression :
  ∀ (k : ℝ) (F : ℝ) (x : ℝ), 
  (F = 10) → (x = 1 / 100) → (k = F / x) → (W = 5) :=
by
sorry

end spring_work_compression_l149_149019


namespace discount_calculation_l149_149295

-- Definitions based on the given conditions
def cost_magazine : Float := 0.85
def cost_pencil : Float := 0.50
def amount_spent : Float := 1.00

-- Define the total cost before discount
def total_cost_before_discount : Float := cost_magazine + cost_pencil

-- Goal: Prove that the discount is $0.35
theorem discount_calculation : total_cost_before_discount - amount_spent = 0.35 := by
  -- Proof (to be filled in later)
  sorry

end discount_calculation_l149_149295


namespace parabola_int_x_axis_for_all_m_l149_149655

theorem parabola_int_x_axis_for_all_m {n : ℝ} :
  (∀ m : ℝ, (9 * m^2 - 4 * m - 4 * n) ≥ 0) → (n ≤ -1 / 9) :=
by
  intro h
  sorry

end parabola_int_x_axis_for_all_m_l149_149655


namespace initial_number_of_eggs_l149_149403

theorem initial_number_of_eggs (eggs_taken harry_eggs eggs_left initial_eggs : ℕ)
    (h1 : harry_eggs = 5)
    (h2 : eggs_left = 42)
    (h3 : initial_eggs = eggs_left + harry_eggs) : 
    initial_eggs = 47 := by
  sorry

end initial_number_of_eggs_l149_149403


namespace speed_of_student_B_l149_149829

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l149_149829


namespace clock_angle_acute_probability_l149_149935

noncomputable def probability_acute_angle : ℚ := 1 / 2

theorem clock_angle_acute_probability :
  ∀ (hour minute : ℕ), (hour >= 0 ∧ hour < 12) →
  (minute >= 0 ∧ minute < 60) →
  (let angle := min (60 * hour - 11 * minute) (720 - (60 * hour - 11 * minute)) in angle < 90 ↔ probability_acute_angle = 1 / 2) :=
sorry

end clock_angle_acute_probability_l149_149935


namespace loss_percentage_is_11_l149_149709

-- Constants for the given problem conditions
def cost_price : ℝ := 1500
def selling_price : ℝ := 1335

-- Formulation of the proof problem
theorem loss_percentage_is_11 :
  ((cost_price - selling_price) / cost_price) * 100 = 11 := by
  sorry

end loss_percentage_is_11_l149_149709


namespace janina_cover_expenses_l149_149515

noncomputable def rent : ℝ := 30
noncomputable def supplies : ℝ := 12
noncomputable def price_per_pancake : ℝ := 2
noncomputable def total_expenses : ℝ := rent + supplies

theorem janina_cover_expenses : total_expenses / price_per_pancake = 21 := 
by
  calc
    total_expenses / price_per_pancake 
    = (rent + supplies) / price_per_pancake : by rfl
    ... = 42 / 2 : by norm_num
    ... = 21 : by norm_num

end janina_cover_expenses_l149_149515


namespace hyperbola_equation_l149_149191

theorem hyperbola_equation (c a b : ℝ) (ecc : ℝ) (h_c : c = 3) (h_ecc : ecc = 3 / 2) (h_a : a = 2) (h_b : b^2 = c^2 - a^2) :
    (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ↔ (x^2 / 4 - y^2 / 5 = 1)) :=
by
  sorry

end hyperbola_equation_l149_149191


namespace rationalization_correct_l149_149381

noncomputable def rationalize_denominator : Prop :=
  (∃ (a b : ℝ), a = (12:ℝ).sqrt + (5:ℝ).sqrt ∧ b = (3:ℝ).sqrt + (5:ℝ).sqrt ∧
                (a / b) = (((15:ℝ).sqrt - 1) / 2))

theorem rationalization_correct : rationalize_denominator :=
by {
  sorry
}

end rationalization_correct_l149_149381


namespace sufficient_and_necessary_condition_l149_149223

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem sufficient_and_necessary_condition (a b : ℝ) : (a + b > 0) ↔ (f a + f b > 0) :=
by sorry

end sufficient_and_necessary_condition_l149_149223


namespace area_of_shape_l149_149469

theorem area_of_shape (x y : ℝ) (α : ℝ) (P : ℝ × ℝ) :
  (x - 2 * Real.cos α)^2 + (y - 2 * Real.sin α)^2 = 16 →
  ∃ A : ℝ, A = 32 * Real.pi :=
by
  sorry

end area_of_shape_l149_149469


namespace second_intersection_points_circle_l149_149471

noncomputable theory

variables {Sphere : Type*} {Circle : Type*} {Point : Type*}

-- Define the conditions
structure sphere (S : Type*) :=
(center : Point)
(radius : ℝ)

structure circle (S : Type*) :=
(center : Point)
(radius : ℝ)
(inside : sphere S)

-- Given conditions
variables (sphere1 : sphere Sphere) (circle_S : circle Sphere) (P : Point)
(hP : P ≠ sphere1.center)

-- Statement to prove
theorem second_intersection_points_circle :
  ∀ Q ∈ circle_S, ∃ circle' : circle Sphere,
  (∃ X Y : Point, line_through P X ∧ line_through P Y ∧ 
   X ∈ sphere1 ∧ Y ∈ sphere1 ∧ Q ∈ circle_S ∧
   Y ≠ Q ∧ Y ∈ circle'.inside) :=
sorry

end second_intersection_points_circle_l149_149471


namespace integer_triplets_satisfy_eq_l149_149165

theorem integer_triplets_satisfy_eq {x y z : ℤ} : 
  x^2 + y^2 + z^2 - x * y - y * z - z * x = 3 ↔ 
  (∃ k : ℤ, (x = k + 2 ∧ y = k + 1 ∧ z = k) ∨ (x = k - 2 ∧ y = k - 1 ∧ z = k)) := 
by
  sorry

end integer_triplets_satisfy_eq_l149_149165


namespace value_of_m_l149_149667

theorem value_of_m 
    (x : ℝ) (m : ℝ) 
    (h : 0 < x)
    (h_eq : (2 / (x - 2)) - ((2 * x - m) / (2 - x)) = 3) : 
    m = 6 := 
sorry

end value_of_m_l149_149667


namespace largest_three_digit_number_l149_149730

theorem largest_three_digit_number :
  ∃ n k m : ℤ, 100 ≤ n ∧ n < 1000 ∧ n = 7 * k + 2 ∧ n = 4 * m + 1 ∧ n = 989 :=
by
  sorry

end largest_three_digit_number_l149_149730


namespace find_speed_B_l149_149857

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l149_149857


namespace combined_time_is_45_l149_149053

-- Definitions based on conditions
def Pulsar_time : ℕ := 10
def Polly_time : ℕ := 3 * Pulsar_time
def Petra_time : ℕ := (1 / 6 ) * Polly_time

-- Total combined time
def total_time : ℕ := Pulsar_time + Polly_time + Petra_time

-- Theorem to prove
theorem combined_time_is_45 : total_time = 45 := by
  sorry

end combined_time_is_45_l149_149053


namespace cans_purchased_l149_149386

theorem cans_purchased (S Q E : ℕ) (hQ : Q ≠ 0) :
  (∃ x : ℕ, x = (5 * S * E) / Q) := by
  sorry

end cans_purchased_l149_149386


namespace dots_not_visible_l149_149310

def total_dots_on_die : Nat := 21
def number_of_dice : Nat := 4
def total_dots : Nat := number_of_dice * total_dots_on_die
def visible_faces : List Nat := [1, 2, 2, 3, 3, 5, 6]
def sum_visible_faces : Nat := visible_faces.sum

theorem dots_not_visible : total_dots - sum_visible_faces = 62 := by
  sorry

end dots_not_visible_l149_149310


namespace N_subset_M_l149_149487

-- Definitions of sets M and N
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | x * x - x < 0 }

-- Proof statement: N is a subset of M
theorem N_subset_M : N ⊆ M :=
sorry

end N_subset_M_l149_149487


namespace least_possible_value_l149_149092

theorem least_possible_value (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 4 * x = 5 * y ∧ 5 * y = 6 * z) : x + y + z = 37 :=
by
  sorry

end least_possible_value_l149_149092


namespace B_fraction_l149_149523

theorem B_fraction (A_s B_s C_s : ℕ) (h1 : A_s = 600) (h2 : A_s = (2 / 5) * (B_s + C_s))
  (h3 : A_s + B_s + C_s = 1800) :
  B_s / (A_s + C_s) = 1 / 6 :=
by
  sorry

end B_fraction_l149_149523


namespace pow_mod_seventeen_l149_149547

theorem pow_mod_seventeen sixty_three :
  17^63 % 7 = 6 := by
  have h : 17 % 7 = 3 := by norm_num
  have h1 : 17^63 % 7 = 3^63 % 7 := by rw [pow_mod_eq_of_mod_eq h] 
  norm_num at h1
  rw [h1]
  sorry

end pow_mod_seventeen_l149_149547


namespace number_of_arrangements_l149_149875

noncomputable def arrangements_nonadjacent_teachers (A : ℕ → ℕ → ℕ) : ℕ :=
  let students_arrangements := A 8 8
  let gaps_count := 9
  let teachers_arrangements := A gaps_count 2
  students_arrangements * teachers_arrangements

theorem number_of_arrangements (A : ℕ → ℕ → ℕ) :
  arrangements_nonadjacent_teachers A = A 8 8 * A 9 2 := 
  sorry

end number_of_arrangements_l149_149875


namespace customers_at_start_l149_149435

def initial_customers (X : ℕ) : Prop :=
  let first_hour := X + 3
  let second_hour := first_hour - 6
  second_hour = 12

theorem customers_at_start {X : ℕ} : initial_customers X → X = 15 :=
by
  sorry

end customers_at_start_l149_149435


namespace monotonically_increasing_intervals_exists_a_decreasing_l149_149007

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x - a * x - 1

theorem monotonically_increasing_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x : ℝ, 0 ≤ Real.exp x - a) ∧
  (a > 0 → ∀ x : ℝ, x ≥ Real.log a → 0 ≤ Real.exp x - a) :=
by sorry

theorem exists_a_decreasing (a : ℝ) :
  (a ≥ Real.exp 3) ↔ ∀ x : ℝ, -2 < x ∧ x < 3 → Real.exp x - a ≤ 0 :=
by sorry

end monotonically_increasing_intervals_exists_a_decreasing_l149_149007


namespace total_purchase_cost_l149_149863

variable (kg_nuts : ℝ) (kg_dried_fruits : ℝ)
variable (cost_per_kg_nuts : ℝ) (cost_per_kg_dried_fruits : ℝ)

-- Define the quantities
def cost_nuts := kg_nuts * cost_per_kg_nuts
def cost_dried_fruits := kg_dried_fruits * cost_per_kg_dried_fruits

-- The total cost can be expressed as follows
def total_cost := cost_nuts + cost_dried_fruits

theorem total_purchase_cost (h1 : kg_nuts = 3) (h2 : kg_dried_fruits = 2.5)
  (h3 : cost_per_kg_nuts = 12) (h4 : cost_per_kg_dried_fruits = 8) :
  total_cost kg_nuts kg_dried_fruits cost_per_kg_nuts cost_per_kg_dried_fruits = 56 := by
  sorry

end total_purchase_cost_l149_149863


namespace hyejin_math_score_l149_149012

theorem hyejin_math_score :
  let ethics := 82
  let korean_language := 90
  let science := 88
  let social_studies := 84
  let avg_score := 88
  let total_subjects := 5
  ∃ (M : ℕ), (ethics + korean_language + science + social_studies + M) / total_subjects = avg_score := by
    sorry

end hyejin_math_score_l149_149012


namespace tangent_line_eq_l149_149652

variable {R : Type} [LinearOrderedField R] [Algebra R R] {x : R}

def f (x : R) : R := x^3 + 2 * deriv f 1 * x^2

def deriv_f_eq : deriv f x = 3 * x^2 + 4 * deriv f 1 * x := sorry

def deriv_f_at_1_eq_minus_1 : deriv f 1 = -1 :=
by
  calc
    4 * deriv f 1 = deriv f 1 - 3 := by sorry
    3 * deriv f 1 = -3 := by sorry
    deriv f 1 = -1 := by sorry

def f_at_1 : f 1 = -1 :=
by
  calc
    f 1 = 1^3 - 2 * 1^2 := by sorry
    _ = -1 := by sorry

theorem tangent_line_eq (x R : Type) [LinearOrderedField R] [Algebra R R] : 
  (∀ x : R, (y : R) x = -x) :=
sorry

end tangent_line_eq_l149_149652


namespace bus_speed_l149_149140

def distance : ℝ := 350.028
def time : ℝ := 10
def speed_kmph : ℝ := 126.01

theorem bus_speed :
  (distance / time) * 3.6 = speed_kmph := 
sorry

end bus_speed_l149_149140


namespace notebook_price_l149_149043

theorem notebook_price (x : ℝ) 
  (h1 : 3 * x + 1.50 + 1.70 = 6.80) : 
  x = 1.20 :=
by 
  sorry

end notebook_price_l149_149043


namespace speed_of_student_B_l149_149784

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l149_149784


namespace students_not_in_biology_l149_149906

theorem students_not_in_biology (S : ℕ) (f : ℚ) (hS : S = 840) (hf : f = 0.35) :
  S - (f * S) = 546 :=
by
  sorry

end students_not_in_biology_l149_149906


namespace rationalize_fraction_l149_149375

-- Define the terms of the problem
def expr1 := (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5)
def expected := (Real.sqrt 15 - 1) / 2

-- State the theorem
theorem rationalize_fraction :
  expr1 = expected :=
by
  sorry

end rationalize_fraction_l149_149375


namespace student_B_speed_l149_149793

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l149_149793


namespace Nord_Stream_pipeline_payment_l149_149058

/-- Suppose Russia, Germany, and France decided to build the "Nord Stream 2" pipeline,
     which is 1200 km long, agreeing to finance this project equally.
     Russia built 650 kilometers of the pipeline.
     Germany built 550 kilometers of the pipeline.
     France contributed its share in money and did not build any kilometers.
     Germany received 1.2 billion euros from France.
     Prove that Russia should receive 2 billion euros from France.
--/
theorem Nord_Stream_pipeline_payment
  (total_km : ℝ)
  (russia_km : ℝ)
  (germany_km : ℝ)
  (total_countries : ℝ)
  (payment_to_germany : ℝ)
  (germany_additional_payment : ℝ)
  (france_km : ℝ)
  (france_payment_ratio : ℝ)
  (russia_payment : ℝ) :
  total_km = 1200 ∧
  russia_km = 650 ∧
  germany_km = 550 ∧
  total_countries = 3 ∧
  payment_to_germany = 1.2 ∧
  france_km = 0 ∧
  germany_additional_payment = germany_km - (total_km / total_countries) ∧
  france_payment_ratio = 5 / 3 ∧
  russia_payment = payment_to_germany * (5 / 3) →
  russia_payment = 2 := by sorry

end Nord_Stream_pipeline_payment_l149_149058


namespace number_of_good_subsets_of_1_to_13_l149_149225

open Finset

theorem number_of_good_subsets_of_1_to_13 : 
  let n := 13
  let S := range (n+1) \ 0
  have cardinality_of_S : S.card = n := sorry
  have sum_S_is_odd : (S.sum id) % 2 = 1 := sorry
  ∃ (s : Finset ℕ), s ⊂ S ∧ (s.sum id) % 2 = 0 ∧ s ≠ ∅ → 
    (∃ (k : ℕ), k = 2 ^ (n - 1) - 1)
:=
begin
  sorry
end

end number_of_good_subsets_of_1_to_13_l149_149225


namespace sqrt6_eq_l149_149979

theorem sqrt6_eq (r : Real) (h : r = Real.sqrt 2 + Real.sqrt 3) : Real.sqrt 6 = (r ^ 2 - 5) / 2 :=
by
  sorry

end sqrt6_eq_l149_149979


namespace find_original_number_l149_149974

/-- The difference between a number increased by 18.7% and the same number decreased by 32.5% is 45. -/
theorem find_original_number (w : ℝ) (h : 1.187 * w - 0.675 * w = 45) : w = 45 / 0.512 :=
by
  sorry

end find_original_number_l149_149974


namespace speed_of_student_B_l149_149843

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l149_149843


namespace max_cut_strings_preserving_net_l149_149576

-- Define the conditions of the problem
def volleyball_net_width : ℕ := 50
def volleyball_net_height : ℕ := 600

-- The vertices count is calculated as (width + 1) * (height + 1)
def vertices_count : ℕ := (volleyball_net_width + 1) * (volleyball_net_height + 1)

-- The total edges count is the sum of vertical and horizontal edges
def total_edges_count : ℕ := volleyball_net_width * (volleyball_net_height + 1) + (volleyball_net_width + 1) * volleyball_net_height

-- The edges needed to keep the graph connected (number of vertices - 1)
def edges_in_tree : ℕ := vertices_count - 1

-- The maximum removable edges (total edges - edges needed in tree)
def max_removable_edges : ℕ := total_edges_count - edges_in_tree

-- Define the theorem to prove
theorem max_cut_strings_preserving_net : max_removable_edges = 30000 := by
  sorry

end max_cut_strings_preserving_net_l149_149576


namespace translated_parabola_correct_l149_149728

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := x^2 + 2

-- Theorem stating that translating the original parabola up by 2 units results in the translated parabola
theorem translated_parabola_correct (x : ℝ) :
  translated_parabola x = original_parabola x + 2 :=
by
  sorry

end translated_parabola_correct_l149_149728


namespace average_height_corrected_l149_149063

-- Defining the conditions as functions and constants
def incorrect_average_height : ℝ := 175
def number_of_students : ℕ := 30
def incorrect_height : ℝ := 151
def actual_height : ℝ := 136

-- The target average height to prove
def target_actual_average_height : ℝ := 174.5

-- Main theorem stating the problem
theorem average_height_corrected : 
  (incorrect_average_height * number_of_students - (incorrect_height - actual_height)) / number_of_students = target_actual_average_height :=
by
  sorry

end average_height_corrected_l149_149063


namespace good_fractions_expression_l149_149598

def is_good_fraction (n : ℕ) (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = n

theorem good_fractions_expression (n : ℕ) (a b : ℕ) :
  n > 1 →
  (∀ a b, b < n → is_good_fraction n a b → ∃ x y, x + y = a / b ∨ x - y = a / b) ↔
  Nat.Prime n :=
by
  sorry

end good_fractions_expression_l149_149598


namespace x_coordinate_second_point_l149_149512

theorem x_coordinate_second_point (m n : ℝ) 
(h₁ : m = 2 * n + 5)
(h₂ : m + 2 = 2 * (n + 1) + 5) : 
  (m + 2) = 2 * n + 7 :=
by sorry

end x_coordinate_second_point_l149_149512


namespace student_b_speed_l149_149851

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l149_149851


namespace avg_weight_increase_l149_149242

theorem avg_weight_increase
  (A : ℝ) -- Initial average weight
  (n : ℕ) -- Initial number of people
  (w_old : ℝ) -- Weight of the person being replaced
  (w_new : ℝ) -- Weight of the new person
  (h_n : n = 8) -- Initial number of people is 8
  (h_w_old : w_old = 85) -- Weight of the replaced person is 85
  (h_w_new : w_new = 105) -- Weight of the new person is 105
  : ((8 * A + (w_new - w_old)) / 8) - A = 2.5 := 
sorry

end avg_weight_increase_l149_149242


namespace greatest_savings_by_choosing_boat_l149_149696

/-- Given the transportation costs:
     - plane cost: $600.00
     - boat cost: $254.00
     - helicopter cost: $850.00
    Prove that the greatest amount of money saved by choosing the boat over the other options is $596.00. -/
theorem greatest_savings_by_choosing_boat :
  let plane_cost := 600
  let boat_cost := 254
  let helicopter_cost := 850
  max (plane_cost - boat_cost) (helicopter_cost - boat_cost) = 596 :=
by
  sorry

end greatest_savings_by_choosing_boat_l149_149696


namespace part2_l149_149482

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x

theorem part2 (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : f 1 x1 = f 1 x2) : x1 + x2 > 2 := by
  have f_x1 := h2
  sorry

end part2_l149_149482


namespace sam_investment_time_l149_149236

theorem sam_investment_time (P r : ℝ) (n A t : ℕ) (hP : P = 8000) (hr : r = 0.10) (hn : n = 2) (hA : A = 8820) :
  A = P * (1 + r / n) ^ (n * t) → t = 1 :=
by
  sorry

end sam_investment_time_l149_149236


namespace math_problem_proof_l149_149643

noncomputable def least_n : ℕ := 4

theorem math_problem_proof : 
  ∃ n ≥ 1, ((1 : ℝ) / n) - ((1 : ℝ) / (n + 1)) < 1 / 15 ∧ 
           ∀ m ≥ 1, (((1 : ℝ) / m - (1 : ℝ) / (m + 1)) < 1 / 15) → 
           least_n ≤ m := 
by
  use least_n
  sorry

end math_problem_proof_l149_149643


namespace part_one_part_one_equality_part_two_l149_149891

-- Given constants and their properties
variables (a b c d : ℝ)

-- Statement for the first problem
theorem part_one : a^6 + b^6 + c^6 + d^6 - 6 * a * b * c * d ≥ -2 :=
sorry

-- Statement for the equality condition in the first problem
theorem part_one_equality (h : |a| = 1 ∧ |b| = 1 ∧ |c| = 1 ∧ |d| = 1) : 
  a^6 + b^6 + c^6 + d^6 - 6 * a * b * c * d = -2 :=
sorry

-- Statement for the second problem (existence of Mk for k >= 4 and odd)
theorem part_two (k : ℕ) (hk1 : 4 ≤ k) (hk2 : k % 2 = 1) : ∃ Mk : ℝ, ∀ a b c d : ℝ, a^k + b^k + c^k + d^k - k * a * b * c * d ≥ Mk :=
sorry

end part_one_part_one_equality_part_two_l149_149891


namespace model_height_l149_149453

noncomputable def H_actual : ℝ := 50
noncomputable def A_actual : ℝ := 25
noncomputable def A_model : ℝ := 0.025

theorem model_height : 
  let ratio := (A_actual / A_model)
  ∃ h : ℝ, h = H_actual / (Real.sqrt ratio) ∧ h = 5 * Real.sqrt 10 := 
by 
  sorry

end model_height_l149_149453


namespace blending_marker_drawings_correct_l149_149957

-- Define the conditions
def total_drawings : ℕ := 25
def colored_pencil_drawings : ℕ := 14
def charcoal_drawings : ℕ := 4

-- Define the target proof statement
def blending_marker_drawings : ℕ := total_drawings - (colored_pencil_drawings + charcoal_drawings)

-- Proof goal
theorem blending_marker_drawings_correct : blending_marker_drawings = 7 := by
  sorry

end blending_marker_drawings_correct_l149_149957


namespace graph_symmetric_about_x_2_l149_149464

variables {D : Set ℝ} {f : ℝ → ℝ}

theorem graph_symmetric_about_x_2 (h : ∀ x ∈ D, f (x + 1) = f (-x + 3)) : 
  ∀ x ∈ D, f (x) = f (4 - x) :=
by
  sorry

end graph_symmetric_about_x_2_l149_149464


namespace gum_pieces_per_package_l149_149703

theorem gum_pieces_per_package (packages : ℕ) (extra : ℕ) (total : ℕ) (pieces_per_package : ℕ) :
    packages = 43 → extra = 8 → total = 997 → 43 * pieces_per_package + extra = total → pieces_per_package = 23 :=
by
  intros hpkg hextra htotal htotal_eq
  sorry

end gum_pieces_per_package_l149_149703


namespace sqrt_31_estimate_l149_149589

theorem sqrt_31_estimate : 5 < Real.sqrt 31 ∧ Real.sqrt 31 < 6 := 
by
  sorry

end sqrt_31_estimate_l149_149589


namespace smallest_value_of_k_l149_149593

theorem smallest_value_of_k (k : ℝ) :
  (∃ x : ℝ, x^2 - 4 * x + k = 5) ↔ k >= 9 := 
sorry

end smallest_value_of_k_l149_149593


namespace speed_of_student_B_l149_149837

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l149_149837


namespace speed_of_student_B_l149_149840

theorem speed_of_student_B (d : ℕ) (vA : ℕ → ℕ) (vB : ℕ → ℕ) (t_diff : ℕ → ℚ)
  (h1 : d = 12) 
  (h2 : ∀ x, vA x = 6/5 * x)
  (h3 : ∀ x, vB x = x)
  (h4 : t_diff = 1/6) :
  ∃ (x : ℚ), vB x = 12 := 
begin
  sorry
end

end speed_of_student_B_l149_149840


namespace nth_term_is_4037_l149_149246

noncomputable def arithmetic_sequence_nth_term (n : ℕ) : ℤ :=
7 + (n - 1) * 6

theorem nth_term_is_4037 {n : ℕ} : arithmetic_sequence_nth_term 673 = 4037 :=
by
  sorry

end nth_term_is_4037_l149_149246


namespace plywood_cut_difference_l149_149108

/-- A proof problem to determine the positive difference between the greatest possible
perimeter and the least possible perimeter of congruent pieces resulting from cutting 
a 6-foot by 9-foot rectangular plywood into 6 congruent rectangles with no wood leftover 
or lost --/
theorem plywood_cut_difference :
  ∃ (perimeter_max perimeter_min : ℕ), 
  let piece1 := 1 * 9
  let piece2 := 1 * 6
  let piece3 := 2 * 3
  let perimeter1 := 2 * (1 + 9)
  let perimeter2 := 2 * (1 + 6)
  let perimeter3 := 2 * (2 + 3)
  perimeter_max = perimeter1 ∧
  perimeter_min = perimeter3 ∧
  (perimeter_max - perimeter_min) = 10 :=
sorry

end plywood_cut_difference_l149_149108


namespace sum_of_squares_l149_149954

variable {x y : ℝ}

theorem sum_of_squares (h1 : x + y = 20) (h2 : x * y = 100) : x^2 + y^2 = 200 :=
sorry

end sum_of_squares_l149_149954


namespace number_of_roots_in_right_half_plane_is_one_l149_149883

def Q5 (z : ℂ) : ℂ := z^5 + z^4 + 2*z^3 - 8*z - 1

theorem number_of_roots_in_right_half_plane_is_one :
  (∃ n, ∀ z, Q5 z = 0 ∧ z.re > 0 ↔ n = 1) := 
sorry

end number_of_roots_in_right_half_plane_is_one_l149_149883


namespace least_n_l149_149624

theorem least_n (n : ℕ) (h : 0 < n ∧ (1 / n - 1 / (n + 1) < 1 / 15)) : n ≥ 4 :=
by sorry

end least_n_l149_149624


namespace factorize_square_difference_l149_149301

theorem factorize_square_difference (x: ℝ):
  x^2 - 4 = (x + 2) * (x - 2) := by
  -- Using the difference of squares formula a^2 - b^2 = (a + b)(a - b)
  sorry

end factorize_square_difference_l149_149301


namespace age_of_B_l149_149062

theorem age_of_B (a b c d : ℕ) 
  (h1: a + b + c + d = 112)
  (h2: a + c = 58)
  (h3: 2 * b + 3 * d = 135)
  (h4: b + d = 54) :
  b = 27 :=
by
  sorry

end age_of_B_l149_149062


namespace number_of_rel_prime_to_21_in_range_l149_149328

def is_rel_prime (a b : ℕ) : Prop := gcd a b = 1

noncomputable def count_rel_prime_in_range (a b g : ℕ) : ℕ :=
  ((b - a + 1) : ℕ) - ((b / 3 - (a - 1) / 3) + (b / 7 - (a - 1) / 7) - (b / 21 - (a - 1) / 21))

theorem number_of_rel_prime_to_21_in_range :
  count_rel_prime_in_range 11 99 21 = 51 :=
by 
  sorry

end number_of_rel_prime_to_21_in_range_l149_149328


namespace triangle_area_l149_149261

open Real

def line1 (x y : ℝ) : Prop := y = 6
def line2 (x y : ℝ) : Prop := y = 2 + x
def line3 (x y : ℝ) : Prop := y = 2 - x

def is_vertex (x y : ℝ) (l1 l2 : ℝ → ℝ → Prop) : Prop := l1 x y ∧ l2 x y

def vertices (v1 v2 v3 : ℝ × ℝ) : Prop :=
  is_vertex v1.1 v1.2 line1 line2 ∧
  is_vertex v2.1 v2.2 line1 line3 ∧
  is_vertex v3.1 v3.2 line2 line3

def area_triangle (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  0.5 * abs ((v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v1.2) -
             (v2.1 * v1.2 + v3.1 * v2.2 + v1.1 * v3.2))

theorem triangle_area : vertices (4, 6) (-4, 6) (0, 2) → area_triangle (4, 6) (-4, 6) (0, 2) = 8 :=
by
  sorry

end triangle_area_l149_149261


namespace plywood_cut_perimeter_difference_l149_149115

theorem plywood_cut_perimeter_difference :
  (∃ (l w : ℕ), (l * w = 54) ∧ (9 % w = 0) ∧ (6 % l = 0) ∧ (6 / l) * (9 / w) = 6) →
  10 =
  let p := λ l w, 2 * (l + w) in
  let perimeters := [
    p 1 9,
    p 1 6,
    p 2 3,
    p 3 2
  ]
  in (list.max precedence perimeters - list.min precedence perimeters) :=
begin
  sorry
end

end plywood_cut_perimeter_difference_l149_149115


namespace find_speed_of_B_l149_149758

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l149_149758


namespace sum_inverse_one_minus_roots_eq_half_l149_149006

noncomputable def cubic_eq_roots (x : ℝ) : ℝ := 10 * x^3 - 25 * x^2 + 8 * x - 1

theorem sum_inverse_one_minus_roots_eq_half
  {p q s : ℝ} (hpqseq : cubic_eq_roots p = 0 ∧ cubic_eq_roots q = 0 ∧ cubic_eq_roots s = 0)
  (hpospq : 0 < p ∧ 0 < q ∧ 0 < s) (hlespq : p < 1 ∧ q < 1 ∧ s < 1) :
  (1 / (1 - p)) + (1 / (1 - q)) + (1 / (1 - s)) = 1 / 2 :=
sorry

end sum_inverse_one_minus_roots_eq_half_l149_149006


namespace rationalize_denominator_l149_149379

noncomputable def sqrt_12 := Real.sqrt 12
noncomputable def sqrt_5 := Real.sqrt 5
noncomputable def sqrt_3 := Real.sqrt 3
noncomputable def sqrt_15 := Real.sqrt 15

theorem rationalize_denominator :
  (sqrt_12 + sqrt_5) / (sqrt_3 + sqrt_5) = (-1 / 2) + (sqrt_15 / 2) :=
sorry

end rationalize_denominator_l149_149379


namespace find_m_l149_149325

variable (a : ℝ × ℝ := (2, 3))
variable (b : ℝ × ℝ := (-1, 2))

def isCollinear (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 - u.2 * v.1 = 0

theorem find_m (m : ℝ) (h : isCollinear (2 * m - 4, 3 * m + 8) (4, -1)) : m = -2 :=
by {
  sorry
}

end find_m_l149_149325


namespace opposite_seven_is_minus_seven_l149_149536

theorem opposite_seven_is_minus_seven :
  ∃ x : ℤ, 7 + x = 0 ∧ x = -7 := 
sorry

end opposite_seven_is_minus_seven_l149_149536


namespace more_girls_than_boys_l149_149674

def initial_girls : ℕ := 632
def initial_boys : ℕ := 410
def new_girls_joined : ℕ := 465
def total_girls : ℕ := initial_girls + new_girls_joined

theorem more_girls_than_boys :
  total_girls - initial_boys = 687 :=
by
  -- Proof goes here
  sorry


end more_girls_than_boys_l149_149674


namespace parabola_x_intercepts_count_l149_149011

theorem parabola_x_intercepts_count : 
  let equation := fun y : ℝ => -3 * y^2 + 2 * y + 3
  ∃! x : ℝ, ∃ y : ℝ, y = 0 ∧ x = equation y :=
by
  sorry

end parabola_x_intercepts_count_l149_149011


namespace ratio_smaller_to_larger_dimension_of_framed_painting_l149_149574

-- Definitions
def painting_width : ℕ := 16
def painting_height : ℕ := 20
def side_frame_width (x : ℝ) : ℝ := x
def top_frame_width (x : ℝ) : ℝ := 1.5 * x
def total_frame_area (x : ℝ) : ℝ := (painting_width + 2 * side_frame_width x) * (painting_height + 2 * top_frame_width x) - painting_width * painting_height
def frame_area_eq_painting_area (x : ℝ) : Prop := total_frame_area x = painting_width * painting_height

-- Lean statement
theorem ratio_smaller_to_larger_dimension_of_framed_painting :
  ∃ x : ℝ, frame_area_eq_painting_area x → 
  ((painting_width + 2 * side_frame_width x) / (painting_height + 2 * top_frame_width x)) = (3 / 4) :=
by
  sorry

end ratio_smaller_to_larger_dimension_of_framed_painting_l149_149574


namespace common_chord_length_of_intersecting_circles_l149_149324

noncomputable def commonChordLength (C₁ C₂ : Circle) : ℝ :=
  2 * real.sqrt 5

theorem common_chord_length_of_intersecting_circles :
  ∀ (C₁ C₂ : Circle), 
  (C₁ = {center := (2, 1), radius := real.sqrt 10}) → 
  (C₂ = {center := (-6, -3), radius := real.sqrt 50}) →
  C₁.intersects C₂ →
  commonChordLength C₁ C₂ = 2 * real.sqrt 5 :=
by
  intros C₁ C₂ hC₁ hC₂ hIntersects
  sorry

end common_chord_length_of_intersecting_circles_l149_149324


namespace perimeters_positive_difference_l149_149124

theorem perimeters_positive_difference (orig_length orig_width : ℝ) (num_pieces : ℕ)
  (congruent_division : ∃ (length width : ℝ), length * width = (orig_length * orig_width) / num_pieces)
  (greatest_perimeter least_perimeter : ℝ)
  (h1 : greatest_perimeter = 2 * (1.5 + 9))
  (h2 : least_perimeter = 2 * (1 + 6)) :
  abs (greatest_perimeter - least_perimeter) = 7 := 
sorry

end perimeters_positive_difference_l149_149124


namespace contest_sum_l149_149594

theorem contest_sum 
(A B C D E : ℕ) 
(h_sum : A + B + C + D + E = 35)
(h_right_E : B + C + D + E = 13)
(h_right_D : C + D + E = 31)
(h_right_A : B + C + D + E = 21)
(h_right_C : C + D + E = 7)
: D + B = 11 :=
sorry

end contest_sum_l149_149594


namespace plywood_perimeter_difference_l149_149103

theorem plywood_perimeter_difference :
  ∀ (length width : ℕ) (n : ℕ), 
    length = 6 ∧ width = 9 ∧ n = 6 → 
    ∃ (max_perimeter min_perimeter : ℕ), 
      (max_perimeter - min_perimeter = 10) ∧
      max_perimeter = 20 ∧ 
      min_perimeter = 10 :=
by
  sorry

end plywood_perimeter_difference_l149_149103


namespace rectangular_coordinates_from_polar_l149_149143

theorem rectangular_coordinates_from_polar (x y r θ : ℝ) (h1 : r * Real.cos θ = x) (h2 : r * Real.sin θ = y) :
    r = 10 ∧ θ = Real.arctan (6 / 8) ∧ (2 * r, 3 * θ) = (20, 3 * Real.arctan (6 / 8)) →
    (20 * Real.cos (3 * Real.arctan (6 / 8)), 20 * Real.sin (3 * Real.arctan (6 / 8))) = (-7.04, 18.72) :=
by
  intros
  -- We need to prove that the statement holds
  sorry

end rectangular_coordinates_from_polar_l149_149143


namespace rationalize_fraction_l149_149374

-- Define the terms of the problem
def expr1 := (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5)
def expected := (Real.sqrt 15 - 1) / 2

-- State the theorem
theorem rationalize_fraction :
  expr1 = expected :=
by
  sorry

end rationalize_fraction_l149_149374


namespace average_of_remaining_two_numbers_l149_149742

theorem average_of_remaining_two_numbers 
  (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 6.40)
  (h2 : (a + b) / 2 = 6.2)
  (h3 : (c + d) / 2 = 6.1) :
  ((e + f) / 2) = 6.9 :=
by
  sorry

end average_of_remaining_two_numbers_l149_149742


namespace large_cube_painted_blue_l149_149983

theorem large_cube_painted_blue (n : ℕ) (hp : 1 ≤ n) 
  (hc : (6 * n^2) = (1 / 3) * 6 * n^3) : n = 3 := by
  have hh := hc
  sorry

end large_cube_painted_blue_l149_149983


namespace simplify_polynomial_l149_149526

theorem simplify_polynomial (r : ℝ) :
  (2 * r ^ 3 + 5 * r ^ 2 - 4 * r + 8) - (r ^ 3 + 9 * r ^ 2 - 2 * r - 3)
  = r ^ 3 - 4 * r ^ 2 - 2 * r + 11 :=
by sorry

end simplify_polynomial_l149_149526


namespace M_union_N_eq_M_l149_149519

def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | abs (p.1 * p.2) = 1 ∧ p.1 > 0}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | Real.arctan p.1 + Real.arctan p.2 = Real.pi}

theorem M_union_N_eq_M : M ∪ N = M := by
  sorry

end M_union_N_eq_M_l149_149519


namespace solve_bx2_ax_1_lt_0_l149_149657

noncomputable def quadratic_inequality_solution (a b : ℝ) (x : ℝ) : Prop :=
  x^2 + a * x + b > 0

theorem solve_bx2_ax_1_lt_0 (a b : ℝ) :
  (∀ x : ℝ, quadratic_inequality_solution a b x ↔ (x < -2 ∨ x > -1/2)) →
  (∀ x : ℝ, (x = -2 ∨ x = -1/2) → x^2 + a * x + b = 0) →
  (b * x^2 + a * x + 1 < 0) ↔ (-2 < x ∧ x < -1/2) :=
by
  sorry

end solve_bx2_ax_1_lt_0_l149_149657


namespace second_caterer_cheaper_l149_149232

theorem second_caterer_cheaper (x : ℕ) :
  (150 + 18 * x > 250 + 14 * x) → x ≥ 26 :=
by
  intro h
  sorry

end second_caterer_cheaper_l149_149232


namespace solve_congruence_l149_149944

theorem solve_congruence :
  ∃ n : ℤ, 19 * n ≡ 13 [ZMOD 47] ∧ n ≡ 25 [ZMOD 47] :=
by
  sorry

end solve_congruence_l149_149944


namespace fraction_pattern_l149_149045

theorem fraction_pattern (n m k : ℕ) (h : n / m = k * n / (k * m)) : (n + m) / m = (k * n + k * m) / (k * m) := by
  sorry

end fraction_pattern_l149_149045


namespace solve_for_x_l149_149334

theorem solve_for_x (x : ℚ) (h : (3 * x + 5) / 7 = 13) : x = 86 / 3 :=
sorry

end solve_for_x_l149_149334


namespace least_n_satisfies_condition_l149_149622

theorem least_n_satisfies_condition :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (m < n) → ¬(1 / (m : ℝ) - 1 / (m + 1) < 1 / 15)) ∧
    (1 / (n : ℝ) - 1 / (n + 1) < 1 / 15) :=
sorry

end least_n_satisfies_condition_l149_149622


namespace remainder_17_pow_63_mod_7_l149_149545

theorem remainder_17_pow_63_mod_7 : (17 ^ 63) % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l149_149545


namespace square_side_length_l149_149389

theorem square_side_length (d s : ℝ) (h_diag : d = 2) (h_rel : d = s * Real.sqrt 2) : s = Real.sqrt 2 :=
sorry

end square_side_length_l149_149389


namespace trigonometric_identity_l149_149189

theorem trigonometric_identity (x : ℝ) (h₁ : Real.sin x = 4 / 5) (h₂ : π / 2 ≤ x ∧ x ≤ π) :
  Real.cos x = -3 / 5 ∧ (Real.cos (-x) / (Real.sin (π / 2 - x) - Real.sin (2 * π - x)) = -3) := 
by
  sorry

end trigonometric_identity_l149_149189


namespace max_value_of_expr_l149_149929

theorem max_value_of_expr  
  (a b c : ℝ) 
  (h₀ : 0 ≤ a)
  (h₁ : 0 ≤ b)
  (h₂ : 0 ≤ c)
  (h₃ : a + 2 * b + 3 * c = 1) :
  a + b^3 + c^4 ≤ 0.125 := 
sorry

end max_value_of_expr_l149_149929


namespace extremum_of_function_l149_149532

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x

theorem extremum_of_function :
  (∀ x, f x ≥ -Real.exp 1) ∧ (f 1 = -Real.exp 1) ∧ (∀ M, ∃ x, f x > M) :=
by
  sorry

end extremum_of_function_l149_149532


namespace probability_abs_diff_gt_half_is_7_over_16_l149_149056

noncomputable def probability_abs_diff_gt_half : ℚ :=
  let p_tail := (1 : ℚ) / (2 : ℚ)   -- Probability of flipping tails
  let p_head := (1 : ℚ) / (2 : ℚ)   -- Probability of flipping heads
  let p_x_tail_y_tail := p_tail * p_tail   -- Both first flips tails
  let p_x1_y_tail := p_head * p_tail / 2     -- x = 1, y flip tails
  let p_x_tail_y0 := p_tail * p_head / 2     -- x flip tails, y = 0
  let p_x1_y0 := p_head * p_head / 4         -- x = 1, y = 0
  -- Individual probabilities for x − y > 1/2
  let p_x_tail_y_tail_diff := (1 : ℚ) / (8 : ℚ) * p_x_tail_y_tail
  let p_x1_y_tail_diff := (1 : ℚ) / (2 : ℚ) * p_x1_y_tail
  let p_x_tail_y0_diff := (1 : ℚ) / (2 : ℚ) * p_x_tail_y0
  let p_x1_y0_diff := (1 : ℚ) * p_x1_y0
  -- Combined probability for x − y > 1/2
  let p_x_y_diff_gt_half := p_x_tail_y_tail_diff +
                            p_x1_y_tail_diff +
                            p_x_tail_y0_diff +
                            p_x1_y0_diff
  -- Final probability for |x − y| > 1/2 is twice of x − y > 1/2
  2 * p_x_y_diff_gt_half

theorem probability_abs_diff_gt_half_is_7_over_16 :
  probability_abs_diff_gt_half = (7 : ℚ) / 16 := 
  sorry

end probability_abs_diff_gt_half_is_7_over_16_l149_149056


namespace common_ratio_arith_geo_sequence_l149_149252

theorem common_ratio_arith_geo_sequence (a : ℕ → ℝ) (d : ℝ) (q : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_geo : (a 1 + 2) * q = a 5 + 5) 
  (h_geo' : (a 5 + 5) * q = a 9 + 8) :
  q = 1 :=
by
  sorry

end common_ratio_arith_geo_sequence_l149_149252


namespace circles_intersect_l149_149896

theorem circles_intersect (m : ℝ) 
  (h₁ : ∃ x y, x^2 + y^2 = m) 
  (h₂ : ∃ x y, x^2 + y^2 + 6*x - 8*y + 21 = 0) : 
  9 < m ∧ m < 49 :=
by sorry

end circles_intersect_l149_149896


namespace plywood_perimeter_difference_l149_149131

/--
Given a 6-foot by 9-foot rectangular piece of plywood cut into 6 congruent rectangles 
with no wood left over and no wood lost due to the cuts,
prove that the positive difference between the greatest and the least perimeter of a single piece is 11 feet.
-/
theorem plywood_perimeter_difference :
  ∃ (rectangles : List (ℕ × ℕ)), 
  (∀ r ∈ rectangles, r.fst * r.snd = 9 * 6 / 6) ∧
  (Greatest (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) - 
  (Least (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) = 11 :=
by
  sorry

end plywood_perimeter_difference_l149_149131


namespace student_B_speed_l149_149816

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l149_149816


namespace bicycle_speed_B_l149_149768

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l149_149768


namespace right_triangle_inequality_l149_149507

theorem right_triangle_inequality (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : b > a) (h3 : b / a < 2) :
  a^2 / (b^2 + c^2) + b^2 / (a^2 + c^2) > 4 / 9 :=
by
  sorry

end right_triangle_inequality_l149_149507


namespace least_n_l149_149605

theorem least_n (n : ℕ) (h : n > 0) : 
  (1 / (n : ℝ) - 1 / (n + 1) < 1 / 15) ↔ n ≥ 4 := 
begin
  sorry
end

end least_n_l149_149605


namespace evaluate_expression_l149_149329

theorem evaluate_expression (x y : ℕ) (hx : x = 3) (hy : y = 5) :
  3 * x^4 + 2 * y^2 + 10 = 8 * 37 + 7 := 
by
  sorry

end evaluate_expression_l149_149329


namespace A_minus_B_l149_149352

theorem A_minus_B (x y m n A B : ℤ) (hx : x > y) (hx1 : x + y = 7) (hx2 : x * y = 12)
                  (hm : m > n) (hm1 : m + n = 13) (hm2 : m^2 + n^2 = 97)
                  (hA : A = x - y) (hB : B = m - n) :
                  A - B = -4 := by
  sorry

end A_minus_B_l149_149352


namespace find_sum_of_bounds_l149_149688

variable (x y z : ℝ)

theorem find_sum_of_bounds (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) : 
  let m := min x (min y z)
  let M := max x (max y z)
  m + M = 8 / 3 :=
sorry

end find_sum_of_bounds_l149_149688


namespace apple_price_theorem_l149_149384

-- Given conditions
def apple_counts : List Nat := [20, 40, 60, 80, 100, 120, 140]

-- Helper function to calculate revenue for a given apple count.
def revenue (apples : Nat) (price_per_batch : Nat) (price_per_leftover : Nat) (batch_size : Nat) : Nat :=
  (apples / batch_size) * price_per_batch + (apples % batch_size) * price_per_leftover

-- Theorem stating that the price per 7 apples is 1 cent and 3 cents per leftover apple ensures equal revenue.
theorem apple_price_theorem : 
  ∀ seller ∈ apple_counts, 
  revenue seller 1 3 7 = 20 :=
by
  intros seller h_seller
  -- Proof will follow here
  sorry

end apple_price_theorem_l149_149384


namespace coin_flip_probability_l149_149524

theorem coin_flip_probability :
  let p : ℚ := 3
  let q : ℚ := 5 in
  (p / q).denom = q ∧ (p / q).num = p ∧ p.gcd q = 1 →
  p + q = 8 := 
by
  intros
  sorry

end coin_flip_probability_l149_149524


namespace find_x_l149_149333

theorem find_x (x : ℚ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 :=
by
  intro h,
  sorry

end find_x_l149_149333


namespace rationalization_correct_l149_149380

noncomputable def rationalize_denominator : Prop :=
  (∃ (a b : ℝ), a = (12:ℝ).sqrt + (5:ℝ).sqrt ∧ b = (3:ℝ).sqrt + (5:ℝ).sqrt ∧
                (a / b) = (((15:ℝ).sqrt - 1) / 2))

theorem rationalization_correct : rationalize_denominator :=
by {
  sorry
}

end rationalization_correct_l149_149380


namespace range_of_a_l149_149483

variable (a : ℝ)

def a_n (n : ℕ) : ℝ :=
if n = 1 then a else 4 * ↑n + (-1 : ℝ) ^ n * (8 - 2 * a)

theorem range_of_a (h : ∀ n : ℕ, n > 0 → a_n a n < a_n a (n + 1)) : 3 < a ∧ a < 5 :=
by
  sorry

end range_of_a_l149_149483


namespace range_of_sqrt_x_minus_1_meaningful_l149_149015

theorem range_of_sqrt_x_minus_1_meaningful (x : ℝ) (h : 0 ≤ x - 1) : 1 ≤ x := 
sorry

end range_of_sqrt_x_minus_1_meaningful_l149_149015


namespace fraction_meaningful_l149_149078

-- Define the condition about the denominator not being zero.
def denominator_condition (x : ℝ) : Prop := x + 2 ≠ 0

-- The proof problem statement.
theorem fraction_meaningful (x : ℝ) : denominator_condition x ↔ x ≠ -2 :=
by
  -- Ensure that the Lean environment is aware this is a theorem statement.
  sorry -- Proof is omitted as instructed.

end fraction_meaningful_l149_149078


namespace solve_for_k_l149_149664

-- Definition and conditions
def ellipse_eq (k : ℝ) : Prop := ∀ x y, k * x^2 + 5 * y^2 = 5

-- Problem: Prove k = 1 given the above definitions
theorem solve_for_k (k : ℝ) :
  (exists (x y : ℝ), ellipse_eq k ∧ x = 2 ∧ y = 0) -> k = 1 :=
sorry

end solve_for_k_l149_149664


namespace geometric_sequence_a8_l149_149209

theorem geometric_sequence_a8 {a : ℕ → ℝ} (h1 : a 1 * a 3 = 4) (h9 : a 9 = 256) :
  a 8 = 128 ∨ a 8 = -128 :=
sorry

end geometric_sequence_a8_l149_149209


namespace largest_of_three_l149_149564

theorem largest_of_three (a b c : ℕ) (h1 : a = 5) (h2 : b = 8) (h3 : c = 4) : max a (max b c) = 8 := 
sorry

end largest_of_three_l149_149564


namespace new_person_weight_l149_149413

theorem new_person_weight (average_increase : ℝ) (num_persons : ℕ) (replaced_weight : ℝ) (new_weight : ℝ) 
  (h1 : num_persons = 10) 
  (h2 : average_increase = 3.2) 
  (h3 : replaced_weight = 65) : 
  new_weight = 97 :=
by
  sorry

end new_person_weight_l149_149413


namespace student_B_speed_l149_149777

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l149_149777


namespace least_n_l149_149634

theorem least_n (n : ℕ) (h_pos : n > 0) (h_ineq : 1 / n - 1 / (n + 1) < 1 / 15) : n = 4 :=
sorry

end least_n_l149_149634


namespace student_B_speed_l149_149791

theorem student_B_speed (d t : ℝ) (speed_A_relative speed_B time_difference : ℝ)
  (h1 : d = 12) 
  (h2 : speed_A_relative = 1.2)
  (h3 : time_difference = (1 : ℝ) / 6)
  (h4 : ∀ s_B : ℝ, (d / s_B - time_difference = d / (speed_A_relative * s_B)) → s_B = 12) :
  t = 12 :=
by
  let speed_B := t
  have h := h4 speed_B
  exact h (h1 ▸ h3 ▸ rfl)

end student_B_speed_l149_149791


namespace pencils_remaining_in_drawer_l149_149725

-- Definitions of the conditions
def total_pencils_initially : ℕ := 34
def pencils_taken : ℕ := 22

-- The theorem statement with the correct answer
theorem pencils_remaining_in_drawer : total_pencils_initially - pencils_taken = 12 :=
by
  sorry

end pencils_remaining_in_drawer_l149_149725


namespace plywood_perimeter_difference_l149_149101

theorem plywood_perimeter_difference :
  ∀ (length width : ℕ) (n : ℕ), 
    length = 6 ∧ width = 9 ∧ n = 6 → 
    ∃ (max_perimeter min_perimeter : ℕ), 
      (max_perimeter - min_perimeter = 10) ∧
      max_perimeter = 20 ∧ 
      min_perimeter = 10 :=
by
  sorry

end plywood_perimeter_difference_l149_149101


namespace smallest_n_l149_149085

theorem smallest_n (n : ℕ) (h₁ : ∃ k1 : ℕ, 4 * n = k1 ^ 2) (h₂ : ∃ k2 : ℕ, 3 * n = k2 ^ 3) : n = 18 :=
sorry

end smallest_n_l149_149085


namespace find_other_parallel_side_l149_149175

variable (a b h : ℝ) (Area : ℝ)

-- Conditions
axiom h_pos : h = 13
axiom a_val : a = 18
axiom area_val : Area = 247
axiom area_formula : Area = (1 / 2) * (a + b) * h

-- Theorem (to be proved by someone else)
theorem find_other_parallel_side (a b h : ℝ) 
  (h_pos : h = 13) 
  (a_val : a = 18) 
  (area_val : Area = 247) 
  (area_formula : Area = (1 / 2) * (a + b) * h) : 
  b = 20 :=
by
  sorry

end find_other_parallel_side_l149_149175


namespace function_no_extrema_k_equals_one_l149_149343

theorem function_no_extrema_k_equals_one (k : ℝ) (h : ∀ x : ℝ, ¬ ∃ m, (k - 1) * x^2 - 4 * x + 5 - k = m) : k = 1 :=
sorry

end function_no_extrema_k_equals_one_l149_149343


namespace complex_division_l149_149476

theorem complex_division (i : ℂ) (h : i ^ 2 = -1) : (3 - 4 * i) / i = -4 - 3 * i :=
by
  sorry

end complex_division_l149_149476


namespace prob_A_two_qualified_l149_149206

noncomputable def prob_qualified (p : ℝ) : ℝ := p * p

def qualified_rate : ℝ := 0.8

theorem prob_A_two_qualified : prob_qualified qualified_rate = 0.64 :=
by
  sorry

end prob_A_two_qualified_l149_149206


namespace proof_problem_l149_149474

def p : Prop := ∃ x : ℝ, x^2 - x + 1 ≥ 0
def q : Prop := ∀ (a b : ℝ), (a^2 < b^2) → (a < b)

theorem proof_problem (h₁ : p) (h₂ : ¬ q) : p ∧ ¬ q := by
  exact ⟨h₁, h₂⟩

end proof_problem_l149_149474


namespace angle_perpendicular_vectors_l149_149313

theorem angle_perpendicular_vectors (α : ℝ) (h1 : 0 < α) (h2 : α < π)
  (h3 : (1 : ℝ) * Real.sin α + Real.cos α * (1 : ℝ) = 0) : α = 3 * Real.pi / 4 :=
sorry

end angle_perpendicular_vectors_l149_149313


namespace speed_of_student_B_l149_149827

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l149_149827


namespace remainder_17_pow_63_mod_7_l149_149541

theorem remainder_17_pow_63_mod_7 :
  (17 ^ 63) % 7 = 6 :=
by {
  -- Given that 17 ≡ 3 (mod 7)
  have h1 : 17 % 7 = 3 := by norm_num,
  
  -- We need to show that (3 ^ 63) % 7 = 6.
  have h2 : (17 ^ 63) % 7 = (3 ^ 63) % 7 := by {
    rw ← h1,
    exact pow_mod_eq_mod_pow _ _ _
  },
  
  -- Now it suffices to show that (3 ^ 63) % 7 = 6
  have h3 : (3 ^ 63) % 7 = 6 := by {
    rw pow_eq_pow_mod 6, -- 63 = 6 * 10 + 3, so 3^63 = (3^6)^10 * 3^3
    have : 3 ^ 6 % 7 = 1 := by norm_num,
    rw [this, one_pow, one_mul, pow_mod_eq_pow_mod],
    exact_pow [exact_mod [norm_num]],
    exact rfl,
  },
  
  -- Combine both results
  exact h2 ▸ h3
}

end remainder_17_pow_63_mod_7_l149_149541


namespace student_b_speed_l149_149848

theorem student_b_speed :
  ∃ (x : ℝ), x = 12 ∧
  (∀ (A_speed B_speed : ℝ), B_speed = x → A_speed = 1.2 * B_speed → 
    (∀ distance time_difference : ℝ, 
      distance = 12 ∧ time_difference = 1/6 →
      ((distance / B_speed) - (distance / A_speed) = time_difference))) := 
begin
  use 12,
  split,
  {
    refl,
  },
  intros A_speed B_speed B_speed_def A_speed_def distance time_difference dist_def,
  rw [B_speed_def, A_speed_def, dist_def, dist_def.right],
  norm_num,
  sorry,
end

end student_b_speed_l149_149848


namespace no_geometric_progression_l149_149385

theorem no_geometric_progression (r s t : ℕ) (h1 : r < s) (h2 : s < t) :
  ¬ ∃ (b : ℂ), (3^r - 2^r) * b^(s - r) = 3^s - 2^s ∧ (3^s - 2^s) * b^(t - s) = 3^t - 2^t := by
  sorry

end no_geometric_progression_l149_149385


namespace passengers_at_18_max_revenue_l149_149707

noncomputable def P (t : ℝ) : ℝ :=
if 10 ≤ t ∧ t < 20 then 500 - 4 * (20 - t)^2 else
if 20 ≤ t ∧ t ≤ 30 then 500 else 0

noncomputable def Q (t : ℝ) : ℝ :=
if 10 ≤ t ∧ t < 20 then -8 * t - (1800 / t) + 320 else
if 20 ≤ t ∧ t ≤ 30 then 1400 / t else 0

-- 1. Prove P(18) = 484
theorem passengers_at_18 : P 18 = 484 := sorry

-- 2. Prove that Q(t) is maximized at t = 15 with a maximum value of 80
theorem max_revenue : ∃ t, Q t = 80 ∧ t = 15 := sorry

end passengers_at_18_max_revenue_l149_149707


namespace smallest_solution_proof_l149_149177

noncomputable def smallest_solution (x : ℝ) : ℝ :=
  if x = (1 - Real.sqrt 65) / 4 then x else x

theorem smallest_solution_proof :
  ∃ x : ℝ, (2 * x / (x - 2) + (2 * x^2 - 24) / x = 11) ∧
           (∀ y : ℝ, 2 * y / (y - 2) + (2 * y^2 - 24) / y = 11 → y ≥ (1 - Real.sqrt 65) / 4) ∧
           x = (1 - Real.sqrt 65) /4 :=
sorry

end smallest_solution_proof_l149_149177


namespace main_l149_149094

-- Definition for part (a)
def part_a : Prop :=
  ∀ (a b : ℕ), a = 300 ∧ b = 200 → 3^b > 2^a

-- Definition for part (b)
def part_b : Prop :=
  ∀ (c d : ℕ), c = 40 ∧ d = 28 → 3^d > 2^c

-- Definition for part (c)
def part_c : Prop :=
  ∀ (e f : ℕ), e = 44 ∧ f = 53 → 4^f > 5^e

-- Main conjecture proving all parts
theorem main : part_a ∧ part_b ∧ part_c :=
by
  sorry

end main_l149_149094


namespace simplify_expression_to_polynomial_l149_149410

theorem simplify_expression_to_polynomial :
    (3 * x^2 + 4 * x + 8) * (2 * x + 1) - 
    (2 * x + 1) * (x^2 + 5 * x - 72) + 
    (4 * x - 15) * (2 * x + 1) * (x + 6) = 
    12 * x^3 + 22 * x^2 - 12 * x - 10 :=
by
    sorry

end simplify_expression_to_polynomial_l149_149410


namespace problem_solution_l149_149736

noncomputable def sqrt_3_simplest : Prop :=
  let A := Real.sqrt 3
  let B := Real.sqrt 0.5
  let C := Real.sqrt 8
  let D := Real.sqrt (1 / 3)
  ∀ (x : ℝ), x = A ∨ x = B ∨ x = C ∨ x = D → x = A → 
    (x = Real.sqrt 0.5 ∨ x = Real.sqrt 8 ∨ x = Real.sqrt (1 / 3)) ∧ 
    ¬(x = Real.sqrt 0.5 ∨ x = 2 * Real.sqrt 2 ∨ x = Real.sqrt (1 / 3))

theorem problem_solution : sqrt_3_simplest :=
by
  sorry

end problem_solution_l149_149736


namespace sum_first_12_terms_l149_149893

-- Defining the basic sequence recurrence relation
def seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) + (-1 : ℝ) ^ n * a n = 2 * (n : ℝ) - 1

-- Theorem statement: Sum of the first 12 terms of the given sequence is 78
theorem sum_first_12_terms (a : ℕ → ℝ) (h : seq a) : 
  (Finset.range 12).sum a = 78 := 
sorry

end sum_first_12_terms_l149_149893


namespace angle_value_l149_149558

theorem angle_value (y : ℝ) (h1 : 2 * y + 140 = 360) : y = 110 :=
by {
  -- Proof will be written here
  sorry
}

end angle_value_l149_149558


namespace least_n_l149_149599

theorem least_n (n : ℕ) (h : (1 : ℝ) / n - (1 / (n + 1)) < (1 / 15)) : n = 4 :=
sorry

end least_n_l149_149599


namespace find_speed_of_B_l149_149759

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l149_149759


namespace plywood_perimeter_difference_l149_149132

/--
Given a 6-foot by 9-foot rectangular piece of plywood cut into 6 congruent rectangles 
with no wood left over and no wood lost due to the cuts,
prove that the positive difference between the greatest and the least perimeter of a single piece is 11 feet.
-/
theorem plywood_perimeter_difference :
  ∃ (rectangles : List (ℕ × ℕ)), 
  (∀ r ∈ rectangles, r.fst * r.snd = 9 * 6 / 6) ∧
  (Greatest (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) - 
  (Least (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) = 11 :=
by
  sorry

end plywood_perimeter_difference_l149_149132


namespace parallel_line_through_point_l149_149271

theorem parallel_line_through_point (x y c : ℝ) (h1 : c = -1) :
  ∃ c, (x-2*y+c = 0 ∧ x = 1 ∧ y = 0) ∧ ∃ k b, k = 1 ∧ b = -2 ∧ k*x-2*y+b=0 → c = -1 := by
  sorry

end parallel_line_through_point_l149_149271


namespace plywood_cut_difference_l149_149111

theorem plywood_cut_difference :
  let original_width := 6
  let original_height := 9
  let total_area := original_width * original_height
  let num_pieces := 6
  let area_per_piece := total_area / num_pieces
  -- Let possible perimeters based on given conditions
  let max_perimeter := 20
  let min_perimeter := 15
  max_perimeter - min_perimeter = 5 :=
by
  sorry

end plywood_cut_difference_l149_149111


namespace complement_U_A_l149_149658

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}

theorem complement_U_A : U \ A = {2, 4, 5} :=
by
  sorry

end complement_U_A_l149_149658


namespace yoongi_average_score_l149_149738

/-- 
Yoongi's average score on the English test taken in August and September was 86, and his English test score in October was 98. 
Prove that the average score of the English test for 3 months is 90.
-/
theorem yoongi_average_score 
  (avg_aug_sep : ℕ)
  (score_oct : ℕ)
  (hp1 : avg_aug_sep = 86)
  (hp2 : score_oct = 98) :
  ((avg_aug_sep * 2 + score_oct) / 3) = 90 :=
by
  sorry

end yoongi_average_score_l149_149738


namespace impossibility_of_sum_sixteen_l149_149284

open Nat

def max_roll_value : ℕ := 6
def sum_of_two_rolls (a b : ℕ) : ℕ := a + b

theorem impossibility_of_sum_sixteen :
  ∀ a b : ℕ, (1 ≤ a ∧ a ≤ max_roll_value) ∧ (1 ≤ b ∧ b ≤ max_roll_value) → sum_of_two_rolls a b ≠ 16 :=
by
  intros a b h
  sorry

end impossibility_of_sum_sixteen_l149_149284


namespace perimeter_difference_l149_149125

-- Definitions as per conditions
def plywood_width : ℕ := 6
def plywood_height : ℕ := 9
def rectangles_count : ℕ := 6

-- The perimeter difference to be proved
theorem perimeter_difference : 
  ∃ (max_perimeter min_perimeter : ℕ), 
  max_perimeter = 22 ∧ min_perimeter = 12 ∧ (max_perimeter - min_perimeter = 10) :=
by
  sorry

end perimeter_difference_l149_149125


namespace melted_mixture_weight_l149_149565

theorem melted_mixture_weight
    (Z C : ℝ)
    (ratio_eq : Z / C = 9 / 11)
    (zinc_weight : Z = 33.3) :
    Z + C = 74 :=
by
  sorry

end melted_mixture_weight_l149_149565


namespace unique_solution_l149_149292

noncomputable def solve_system (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ) (x1 x2 x3 : ℝ) : Prop :=
  (a11 * x1 + a12 * x2 + a13 * x3 = 0) ∧
  (a21 * x1 + a22 * x2 + a23 * x3 = 0) ∧
  (a31 * x1 + a32 * x2 + a33 * x3 = 0)

theorem unique_solution 
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ)
  (h1 : 0 < a11) (h2 : 0 < a22) (h3 : 0 < a33)
  (h4 : a12 < 0) (h5 : a13 < 0) (h6 : a21 < 0)
  (h7 : a23 < 0) (h8 : a31 < 0) (h9 : a32 < 0)
  (h10 : 0 < a11 + a12 + a13) (h11 : 0 < a21 + a22 + a23) (h12 : 0 < a31 + a32 + a33) :
  ∀ (x1 x2 x3 : ℝ), solve_system a11 a12 a13 a21 a22 a23 a31 a32 a33 x1 x2 x3 → (x1 = 0 ∧ x2 = 0 ∧ x3 = 0) :=
by
  sorry

end unique_solution_l149_149292


namespace solve_for_x_l149_149335

theorem solve_for_x (x : ℚ) (h : (3 * x + 5) / 7 = 13) : x = 86 / 3 :=
sorry

end solve_for_x_l149_149335


namespace trapezium_other_side_length_l149_149172

theorem trapezium_other_side_length 
  (side1 : ℝ) (distance : ℝ) (area : ℝ) (side2 : ℝ)
  (h_side1 : side1 = 18)
  (h_distance : distance = 13)
  (h_area : area = 247)
  (h_area_formula : area = 0.5 * (side1 + side2) * distance) :
  side2 = 20 :=
by
  rw [h_side1, h_distance, h_area] at h_area_formula
  sorry

end trapezium_other_side_length_l149_149172


namespace avg_rate_of_change_interval_1_2_l149_149708

def f (x : ℝ) : ℝ := 2 * x + 1

theorem avg_rate_of_change_interval_1_2 : 
  (f 2 - f 1) / (2 - 1) = 2 :=
by sorry

end avg_rate_of_change_interval_1_2_l149_149708


namespace log_comparison_l149_149694

noncomputable def logBase (a x : ℝ) := Real.log x / Real.log a

theorem log_comparison
  (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) 
  (m : ℝ) (hm : m = logBase a (a^2 + 1))
  (n : ℝ) (hn : n = logBase a (a + 1))
  (p : ℝ) (hp : p = logBase a (2 * a)) :
  p > m ∧ m > n :=
by
  sorry

end log_comparison_l149_149694


namespace candle_ratio_proof_l149_149683

noncomputable def candle_height_ratio := 
  ∃ (x y : ℝ), 
    (x / 6) * 3 = x / 2 ∧
    (y / 8) * 3 = 3 * y / 8 ∧
    (x / 2) = (5 * y / 8) →
    x / y = 5 / 4

theorem candle_ratio_proof : candle_height_ratio :=
by sorry

end candle_ratio_proof_l149_149683


namespace ratio_of_x_to_y_l149_149587

theorem ratio_of_x_to_y (x y : ℤ) (h : (12 * x - 5 * y) / (17 * x - 3 * y) = 5 / 7) : x / y = -20 :=
by
  sorry

end ratio_of_x_to_y_l149_149587


namespace sum_of_digits_B_l149_149595

/- 
  Let A be the natural number formed by concatenating integers from 1 to 100.
  Let B be the smallest possible natural number formed by removing 100 digits from A.
  We need to prove that the sum of the digits of B equals 486.
-/
def A : ℕ := sorry -- construct the natural number 1234567891011121314...99100

def sum_of_digits (n : ℕ) : ℕ := sorry -- function to calculate the sum of digits of a natural number

def B : ℕ := sorry -- construct the smallest possible number B by removing 100 digits from A

theorem sum_of_digits_B : sum_of_digits B = 486 := sorry

end sum_of_digits_B_l149_149595


namespace quadratic_coeffs_l149_149586

theorem quadratic_coeffs (x : ℝ) :
  (x - 1)^2 = 3 * x - 2 → ∃ b c, (x^2 + b * x + c = 0 ∧ b = -5 ∧ c = 3) :=
by
  sorry

end quadratic_coeffs_l149_149586


namespace seventeen_power_sixty_three_mod_seven_l149_149553

theorem seventeen_power_sixty_three_mod_seven : (17^63) % 7 = 6 := by
  -- Here you would write the actual proof demonstrating the equivalence:
  -- 1. 17 ≡ 3 (mod 7)
  -- 2. Calculate 3^63 (mod 7)
  sorry

end seventeen_power_sixty_three_mod_seven_l149_149553


namespace math_problem_proof_l149_149642

noncomputable def least_n : ℕ := 4

theorem math_problem_proof : 
  ∃ n ≥ 1, ((1 : ℝ) / n) - ((1 : ℝ) / (n + 1)) < 1 / 15 ∧ 
           ∀ m ≥ 1, (((1 : ℝ) / m - (1 : ℝ) / (m + 1)) < 1 / 15) → 
           least_n ≤ m := 
by
  use least_n
  sorry

end math_problem_proof_l149_149642


namespace plywood_cut_difference_l149_149110

theorem plywood_cut_difference :
  let original_width := 6
  let original_height := 9
  let total_area := original_width * original_height
  let num_pieces := 6
  let area_per_piece := total_area / num_pieces
  -- Let possible perimeters based on given conditions
  let max_perimeter := 20
  let min_perimeter := 15
  max_perimeter - min_perimeter = 5 :=
by
  sorry

end plywood_cut_difference_l149_149110


namespace regular_polygon_perimeter_l149_149146

theorem regular_polygon_perimeter (n : ℕ) (exterior_angle : ℝ) (side_length : ℝ) 
  (h1 : 360 / exterior_angle = n) (h2 : 20 = exterior_angle)
  (h3 : 10 = side_length) : 180 = n * side_length :=
by
  sorry

end regular_polygon_perimeter_l149_149146


namespace perimeters_positive_difference_l149_149123

theorem perimeters_positive_difference (orig_length orig_width : ℝ) (num_pieces : ℕ)
  (congruent_division : ∃ (length width : ℝ), length * width = (orig_length * orig_width) / num_pieces)
  (greatest_perimeter least_perimeter : ℝ)
  (h1 : greatest_perimeter = 2 * (1.5 + 9))
  (h2 : least_perimeter = 2 * (1 + 6)) :
  abs (greatest_perimeter - least_perimeter) = 7 := 
sorry

end perimeters_positive_difference_l149_149123


namespace clock_angle_acute_probability_l149_149934

-- Given the condition that a clock stops randomly at any moment,
-- and defining the probability of forming an acute angle between the hour and minute hands,
-- prove that this probability is 1/2.

theorem clock_angle_acute_probability : 
  (probability (\theta : ℝ, is_acute ⟨θ % 360, 0 ≤ θ % 360 < 360⟩) = 1/2) :=
-- Definitions and conditions.
sorry

end clock_angle_acute_probability_l149_149934


namespace set_representation_l149_149996

theorem set_representation : 
  { x : ℕ | x < 5 } = {0, 1, 2, 3, 4} :=
sorry

end set_representation_l149_149996


namespace remainder_of_17_pow_63_mod_7_l149_149555

theorem remainder_of_17_pow_63_mod_7 :
  17^63 % 7 = 6 :=
by {
  -- Condition: 17 ≡ 3 (mod 7)
  have h : 17 % 7 = 3 := by norm_num,
  -- Use the periodicity established in the powers of 3 modulo 7 to prove the statement
  -- Note: Leaving the proof part out as instructed
  sorry
}

end remainder_of_17_pow_63_mod_7_l149_149555


namespace number_of_rectangles_in_5x5_grid_l149_149492

-- Number of ways to choose k elements from a set of n elements
def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Given conditions
def points_in_each_direction : ℕ := 5
def number_of_rectangles : ℕ :=
  binomial points_in_each_direction 2 * binomial points_in_each_direction 2

-- Lean statement to prove the problem
theorem number_of_rectangles_in_5x5_grid :
  number_of_rectangles = 100 :=
by
  -- begin Lean proof
  sorry

end number_of_rectangles_in_5x5_grid_l149_149492


namespace speed_of_student_B_l149_149788

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l149_149788


namespace value_of_x_l149_149338

theorem value_of_x (x : ℝ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 :=
by
  intro h
  sorry

end value_of_x_l149_149338


namespace probability_first_genuine_on_third_test_l149_149005

noncomputable def probability_of_genuine : ℚ := 3 / 4
noncomputable def probability_of_defective : ℚ := 1 / 4
noncomputable def probability_X_eq_3 := probability_of_defective * probability_of_defective * probability_of_genuine

theorem probability_first_genuine_on_third_test :
  probability_X_eq_3 = 3 / 64 :=
by
  sorry

end probability_first_genuine_on_third_test_l149_149005


namespace age_difference_l149_149396

variables (X Y Z : ℕ)

theorem age_difference (h : X + Y = Y + Z + 12) : X - Z = 12 :=
sorry

end age_difference_l149_149396


namespace find_ax5_plus_by5_l149_149475

variable (a b x y : ℝ)

-- Conditions
axiom h1 : a * x + b * y = 3
axiom h2 : a * x^2 + b * y^2 = 7
axiom h3 : a * x^3 + b * y^3 = 16
axiom h4 : a * x^4 + b * y^4 = 42

-- Theorem (what we need to prove)
theorem find_ax5_plus_by5 : a * x^5 + b * y^5 = 20 :=
sorry

end find_ax5_plus_by5_l149_149475


namespace vertices_of_cube_l149_149314

-- Given condition: geometric shape is a cube
def is_cube (x : Type) : Prop := true -- This is a placeholder declaration that x is a cube.

-- Question: How many vertices does a cube have?
-- Proof problem: Prove that the number of vertices of a cube is 8.
theorem vertices_of_cube (x : Type) (h : is_cube x) : true := 
  sorry

end vertices_of_cube_l149_149314


namespace volume_to_surface_area_ratio_l149_149992

-- Definitions based on the conditions
def unit_cube_volume : ℕ := 1
def num_unit_cubes : ℕ := 7
def unit_cube_total_volume : ℕ := num_unit_cubes * unit_cube_volume

def surface_area_of_central_cube : ℕ := 0
def exposed_faces_per_surrounding_cube : ℕ := 5
def num_surrounding_cubes : ℕ := 6
def total_surface_area : ℕ := num_surrounding_cubes * exposed_faces_per_surrounding_cube

-- Mathematical proof statement
theorem volume_to_surface_area_ratio : 
  (unit_cube_total_volume : ℚ) / (total_surface_area : ℚ) = 7 / 30 :=
by sorry

end volume_to_surface_area_ratio_l149_149992


namespace smallest_n_l149_149087

theorem smallest_n (n : ℕ) (h1 : ∃ k1, 4 * n = k1^2) (h2 : ∃ k2, 3 * n = k2^3) : n = 144 :=
sorry

end smallest_n_l149_149087


namespace vinegar_solution_concentration_l149_149340

theorem vinegar_solution_concentration
  (original_volume : ℝ) (water_volume : ℝ)
  (original_concentration : ℝ)
  (h1 : original_volume = 12)
  (h2 : water_volume = 50)
  (h3 : original_concentration = 36.166666666666664) :
  original_concentration / 100 * original_volume / (original_volume + water_volume) = 0.07 :=
by
  sorry

end vinegar_solution_concentration_l149_149340


namespace plan_Y_cheaper_l149_149865

theorem plan_Y_cheaper (y : ℤ) :
  (15 * (y : ℚ) > 2500 + 8 * (y : ℚ)) ↔ y > 358 :=
by
  sorry

end plan_Y_cheaper_l149_149865


namespace least_n_l149_149623

theorem least_n (n : ℕ) (h : 0 < n ∧ (1 / n - 1 / (n + 1) < 1 / 15)) : n ≥ 4 :=
by sorry

end least_n_l149_149623


namespace min_f_value_inequality_solution_l149_149042

theorem min_f_value (x : ℝ) : |x+7| + |x-1| ≥ 8 := by
  sorry

theorem inequality_solution (x : ℝ) (m : ℝ) (h : m = 8) : |x-3| - 2*x ≤ 2*m - 12 ↔ x ≥ -1/3 := by
  sorry

end min_f_value_inequality_solution_l149_149042


namespace common_non_integer_root_eq_l149_149234

theorem common_non_integer_root_eq (p1 p2 q1 q2 : ℤ) 
  (x : ℝ) (hx1 : x^2 + p1 * x + q1 = 0) (hx2 : x^2 + p2 * x + q2 = 0) 
  (hnint : ¬ ∃ (n : ℤ), x = n) : p1 = p2 ∧ q1 = q2 :=
sorry

end common_non_integer_root_eq_l149_149234


namespace unique_solution_eq_condition_l149_149452

theorem unique_solution_eq_condition (p q : ℝ) :
  (∃! x : ℝ, (2 * x - 2 * p + q) / (2 * x - 2 * p - q) = (2 * q + p + x) / (2 * q - p - x)) ↔ (p = 3 * q / 4 ∧ q ≠ 0) :=
  sorry

end unique_solution_eq_condition_l149_149452


namespace inequality_transform_l149_149184

theorem inequality_transform (x y : ℝ) (h : x > y) : 2 * x + 1 > 2 * y + 1 := 
by {
  sorry
}

end inequality_transform_l149_149184


namespace number_of_disconnected_regions_l149_149255

theorem number_of_disconnected_regions (n : ℕ) (h : 2 ≤ n) : 
  ∀ R : ℕ → ℕ, (R 1 = 2) → 
  (∀ k, R k = k^2 - k + 2 → R (k + 1) = (k + 1)^2 - (k + 1) + 2) → 
  R n = n^2 - n + 2 :=
sorry

end number_of_disconnected_regions_l149_149255


namespace speed_of_student_B_l149_149790

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l149_149790


namespace janina_must_sell_21_pancakes_l149_149516

/-- The daily rent cost for Janina. -/
def daily_rent := 30

/-- The daily supply cost for Janina. -/
def daily_supplies := 12

/-- The cost of a single pancake. -/
def pancake_price := 2

/-- The total daily expenses for Janina. -/
def total_daily_expenses := daily_rent + daily_supplies

/-- The required number of pancakes Janina needs to sell each day to cover her expenses. -/
def required_pancakes := total_daily_expenses / pancake_price

theorem janina_must_sell_21_pancakes :
  required_pancakes = 21 :=
sorry

end janina_must_sell_21_pancakes_l149_149516


namespace triangle_perimeter_l149_149249

theorem triangle_perimeter (side1 side2 side3 : ℕ) (h1 : side1 = 40) (h2 : side2 = 50) (h3 : side3 = 70) : 
  side1 + side2 + side3 = 160 :=
by 
  sorry

end triangle_perimeter_l149_149249


namespace value_of_x_l149_149337

theorem value_of_x (x : ℝ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 :=
by
  intro h
  sorry

end value_of_x_l149_149337


namespace num_people_got_on_bus_l149_149441

-- Definitions based on the conditions
def initialNum : ℕ := 4
def currentNum : ℕ := 17
def peopleGotOn (initial : ℕ) (current : ℕ) : ℕ := current - initial

-- Theorem statement
theorem num_people_got_on_bus : peopleGotOn initialNum currentNum = 13 := 
by {
  sorry -- Placeholder for the proof
}

end num_people_got_on_bus_l149_149441


namespace number_added_is_59_l149_149202

theorem number_added_is_59 (x : ℤ) (h1 : -2 < 0) (h2 : -3 < 0) (h3 : -2 * -3 + x = 65) : x = 59 :=
by sorry

end number_added_is_59_l149_149202


namespace math_problem_proof_l149_149646

noncomputable def least_n : ℕ := 4

theorem math_problem_proof : 
  ∃ n ≥ 1, ((1 : ℝ) / n) - ((1 : ℝ) / (n + 1)) < 1 / 15 ∧ 
           ∀ m ≥ 1, (((1 : ℝ) / m - (1 : ℝ) / (m + 1)) < 1 / 15) → 
           least_n ≤ m := 
by
  use least_n
  sorry

end math_problem_proof_l149_149646


namespace total_percentage_change_l149_149411

theorem total_percentage_change (X : ℝ) (fall_increase : X' = 1.08 * X) (spring_decrease : X'' = 0.8748 * X) :
  ((X'' - X) / X) * 100 = -12.52 := 
by
  sorry

end total_percentage_change_l149_149411


namespace acute_angle_probability_is_half_l149_149932

noncomputable def probability_acute_angle (h_mins : ℕ) (m_mins : ℕ) : ℝ :=
  if m_mins < 15 ∨ m_mins > 45 then 1 / 2 else 0

theorem acute_angle_probability_is_half (h_mins : ℕ) (m_mins : ℕ) :
  let P := probability_acute_angle h_mins m_mins in
  0 ≤ m_mins ∧ m_mins < 60 →
  P = 1 / 2 :=
sorry

end acute_angle_probability_is_half_l149_149932


namespace sum_of_coordinates_of_B_l149_149936

def point := (ℝ × ℝ)

noncomputable def point_A : point := (0, 0)

def line_y_equals_6 (B : point) : Prop := B.snd = 6

def slope_AB (A B : point) (m : ℝ) : Prop := (B.snd - A.snd) / (B.fst - A.fst) = m

theorem sum_of_coordinates_of_B (B : point) 
  (h1 : B.snd = 6)
  (h2 : slope_AB point_A B (3/5)) :
  B.fst + B.snd = 16 :=
sorry

end sum_of_coordinates_of_B_l149_149936


namespace find_speed_of_B_l149_149757

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l149_149757


namespace least_n_satisfies_condition_l149_149619

theorem least_n_satisfies_condition :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (m < n) → ¬(1 / (m : ℝ) - 1 / (m + 1) < 1 / 15)) ∧
    (1 / (n : ℝ) - 1 / (n + 1) < 1 / 15) :=
sorry

end least_n_satisfies_condition_l149_149619


namespace binom_12_9_eq_220_l149_149291

noncomputable def binom (n k : ℕ) : ℕ :=
  n.choose k

theorem binom_12_9_eq_220 : binom 12 9 = 220 :=
sorry

end binom_12_9_eq_220_l149_149291


namespace perimeter_difference_l149_149129

-- Definitions as per conditions
def plywood_width : ℕ := 6
def plywood_height : ℕ := 9
def rectangles_count : ℕ := 6

-- The perimeter difference to be proved
theorem perimeter_difference : 
  ∃ (max_perimeter min_perimeter : ℕ), 
  max_perimeter = 22 ∧ min_perimeter = 12 ∧ (max_perimeter - min_perimeter = 10) :=
by
  sorry

end perimeter_difference_l149_149129


namespace trey_uses_47_nails_l149_149960

variable (D : ℕ) -- total number of decorations
variable (nails thumbtacks sticky_strips : ℕ)

-- Conditions
def uses_nails := nails = (5 * D) / 8
def uses_thumbtacks := thumbtacks = (9 * D) / 80
def uses_sticky_strips := sticky_strips = 20
def total_decorations := (21 * D) / 80 = 20

-- Question: Prove that Trey uses 47 nails when the conditions hold
theorem trey_uses_47_nails (D : ℕ) (h1 : uses_nails D nails) (h2 : uses_thumbtacks D thumbtacks) (h3 : uses_sticky_strips sticky_strips) (h4 : total_decorations D) : nails = 47 :=  
by
  sorry

end trey_uses_47_nails_l149_149960


namespace pool_cleaning_l149_149921

theorem pool_cleaning (full_capacity_liters : ℕ) (percent_full : ℕ) (loss_per_jump_ml : ℕ) 
    (full_capacity : full_capacity_liters = 2000) (trigger_clean : percent_full = 80) 
    (loss_per_jump : loss_per_jump_ml = 400) : 
    let trigger_capacity_liters := (full_capacity_liters * percent_full) / 100
    let splash_out_capacity_liters := full_capacity_liters - trigger_capacity_liters
    let splash_out_capacity_ml := splash_out_capacity_liters * 1000
    (splash_out_capacity_ml / loss_per_jump_ml) = 1000 :=
by {
    sorry
}

end pool_cleaning_l149_149921


namespace loan_period_l149_149753

theorem loan_period (principal : ℝ) (rate_A rate_C gain_B : ℝ) (n : ℕ) 
  (h1 : principal = 3150)
  (h2 : rate_A = 0.08)
  (h3 : rate_C = 0.125)
  (h4 : gain_B = 283.5) :
  (gain_B = (rate_C * principal - rate_A * principal) * n) → n = 2 := by
  sorry

end loan_period_l149_149753


namespace diagonal_of_square_l149_149390

theorem diagonal_of_square (d : ℝ) (s : ℝ) (h : d = 2) (h_eq : s * Real.sqrt 2 = d) : s = Real.sqrt 2 :=
by sorry

end diagonal_of_square_l149_149390


namespace possible_values_x_plus_y_l149_149693

theorem possible_values_x_plus_y (x y : ℝ) (h1 : x = y * (3 - y)^2) (h2 : y = x * (3 - x)^2) :
  x + y = 0 ∨ x + y = 3 ∨ x + y = 4 ∨ x + y = 5 ∨ x + y = 8 :=
sorry

end possible_values_x_plus_y_l149_149693


namespace distance_between_centers_of_tangent_circles_l149_149489

theorem distance_between_centers_of_tangent_circles
  (R r d : ℝ) (h1 : R = 8) (h2 : r = 3) (h3 : d = R + r) : d = 11 :=
by
  -- Insert proof here
  sorry

end distance_between_centers_of_tangent_circles_l149_149489


namespace remainder_17_pow_63_mod_7_l149_149542

theorem remainder_17_pow_63_mod_7 :
  (17 ^ 63) % 7 = 6 :=
by {
  -- Given that 17 ≡ 3 (mod 7)
  have h1 : 17 % 7 = 3 := by norm_num,
  
  -- We need to show that (3 ^ 63) % 7 = 6.
  have h2 : (17 ^ 63) % 7 = (3 ^ 63) % 7 := by {
    rw ← h1,
    exact pow_mod_eq_mod_pow _ _ _
  },
  
  -- Now it suffices to show that (3 ^ 63) % 7 = 6
  have h3 : (3 ^ 63) % 7 = 6 := by {
    rw pow_eq_pow_mod 6, -- 63 = 6 * 10 + 3, so 3^63 = (3^6)^10 * 3^3
    have : 3 ^ 6 % 7 = 1 := by norm_num,
    rw [this, one_pow, one_mul, pow_mod_eq_pow_mod],
    exact_pow [exact_mod [norm_num]],
    exact rfl,
  },
  
  -- Combine both results
  exact h2 ▸ h3
}

end remainder_17_pow_63_mod_7_l149_149542


namespace find_speed_B_l149_149855

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l149_149855


namespace monotonic_increasing_iff_l149_149481

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + 1 / x

theorem monotonic_increasing_iff (a : ℝ) :
  (∀ x : ℝ, 1 < x → f a x ≥ f a 1) ↔ a ≥ 1 :=
by
  sorry

end monotonic_increasing_iff_l149_149481


namespace find_speed_of_B_l149_149804

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l149_149804


namespace pythagorean_triangle_exists_l149_149942

theorem pythagorean_triangle_exists (a : ℤ) (h : a ≥ 5) : 
  ∃ (b c : ℤ), c ≥ b ∧ b ≥ a ∧ a^2 + b^2 = c^2 :=
by {
  sorry
}

end pythagorean_triangle_exists_l149_149942


namespace smallest_b_for_factoring_l149_149884

theorem smallest_b_for_factoring (b : ℕ) (p q : ℕ) (h1 : p * q = 1800) (h2 : p + q = b) : b = 85 :=
by
  sorry

end smallest_b_for_factoring_l149_149884


namespace find_a_l149_149360

theorem find_a (a : ℤ) (h₀ : 0 ≤ a ∧ a ≤ 13) (h₁ : 13 ∣ (51 ^ 2016 - a)) : a = 1 := sorry

end find_a_l149_149360


namespace gcd_36_60_l149_149881

theorem gcd_36_60 : Int.gcd 36 60 = 12 := by
  sorry

end gcd_36_60_l149_149881


namespace inequality_solution_l149_149502

variable {α : Type*} [LinearOrderedField α]
variable (a b x : α)

theorem inequality_solution (h1 : a < 0) (h2 : b = -a) :
  0 < x ∧ x < 1 ↔ ax^2 + bx > 0 :=
by sorry

end inequality_solution_l149_149502


namespace percentage_of_invalid_votes_calculation_l149_149030

theorem percentage_of_invalid_votes_calculation
  (total_votes_poled : ℕ)
  (valid_votes_B : ℕ)
  (additional_percent_votes_A : ℝ)
  (Vb : ℝ)
  (total_valid_votes : ℝ)
  (P : ℝ) :
  total_votes_poled = 8720 →
  valid_votes_B = 2834 →
  additional_percent_votes_A = 0.15 →
  Vb = valid_votes_B →
  total_valid_votes = (2 * Vb) + (additional_percent_votes_A * total_votes_poled) →
  total_valid_votes / total_votes_poled = 1 - P/100 →
  P = 20 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end percentage_of_invalid_votes_calculation_l149_149030


namespace g_six_l149_149394

noncomputable def g : ℝ → ℝ := sorry

axiom g_func_eq (x y : ℝ) : g (x - y) = g x * g y
axiom g_nonzero (x : ℝ) : g x ≠ 0
axiom g_double (x : ℝ) : g (2 * x) = g x ^ 2
axiom g_value : g 6 = 1

theorem g_six : g 6 = 1 := by
  exact g_value

end g_six_l149_149394


namespace c_d_not_true_l149_149268

variables (Beatles_haircut : Type → Prop) (hooligan : Type → Prop) (rude : Type → Prop)

-- Conditions
axiom a : ∃ x, Beatles_haircut x ∧ hooligan x
axiom b : ∀ y, hooligan y → rude y

-- Prove there is a rude hooligan with a Beatles haircut
theorem c : ∃ z, rude z ∧ Beatles_haircut z ∧ hooligan z :=
sorry

-- Disprove every rude hooligan having a Beatles haircut
theorem d_not_true : ¬(∀ w, rude w ∧ hooligan w → Beatles_haircut w) :=
sorry

end c_d_not_true_l149_149268


namespace inequality_solution_l149_149021

theorem inequality_solution (x y z : ℝ) (h1 : x + 3 * y + 2 * z = 6) :
  (z = 3 - 1/2 * x - 3/2 * y) ∧ (x - 2)^2 + (3 * y - 2)^2 ≤ 4 ∧ 0 ≤ x ∧ x ≤ 4 :=
sorry

end inequality_solution_l149_149021


namespace least_n_satisfies_condition_l149_149620

theorem least_n_satisfies_condition :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (m < n) → ¬(1 / (m : ℝ) - 1 / (m + 1) < 1 / 15)) ∧
    (1 / (n : ℝ) - 1 / (n + 1) < 1 / 15) :=
sorry

end least_n_satisfies_condition_l149_149620


namespace find_k_l149_149668

theorem find_k (k : ℝ) : 
  (1 / 2) * |k| * |k / 2| = 4 → (k = 4 ∨ k = -4) := 
sorry

end find_k_l149_149668


namespace smallest_class_number_l149_149751

theorem smallest_class_number (sum_classes : ℕ) (n_classes interval number_of_classes : ℕ) 
                              (h_sum : sum_classes = 87) (h_n_classes : n_classes = 30) 
                              (h_interval : interval = 5) (h_number_of_classes : number_of_classes = 6) : 
                              ∃ x, x + (interval + x) + (2 * interval + x) + (3 * interval + x) 
                              + (4 * interval + x) + (5 * interval + x) = sum_classes ∧ x = 2 :=
by {
  use 2,
  sorry
}

end smallest_class_number_l149_149751


namespace factorize_quartic_l149_149937

-- Specify that p and q are real numbers (ℝ)
variables {p q : ℝ}

-- Statement: For any real numbers p and q, the polynomial x^4 + p x^2 + q can always be factored into two quadratic polynomials.
theorem factorize_quartic (p q : ℝ) : 
  ∃ a b c d e f : ℝ, (x^2 + a * x + b) * (x^2 + c * x + d) = x^4 + p * x^2 + q :=
sorry

end factorize_quartic_l149_149937


namespace sqrt8_same_type_as_sqrt2_l149_149967

theorem sqrt8_same_type_as_sqrt2 :
  (∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 8) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 4) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 6) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 10) :=
by
  sorry

end sqrt8_same_type_as_sqrt2_l149_149967


namespace measure_AX_l149_149510

-- Definitions based on conditions
def circle_radii (r_A r_B r_C : ℝ) : Prop :=
  r_A - r_B = 6 ∧
  r_A - r_C = 5 ∧
  r_B + r_C = 9

-- Theorem statement
theorem measure_AX (r_A r_B r_C : ℝ) (h : circle_radii r_A r_B r_C) : r_A = 10 :=
by
  sorry

end measure_AX_l149_149510


namespace smallest_w_l149_149498

def fact_936 : ℕ := 2^3 * 3^1 * 13^1

theorem smallest_w (w : ℕ) (h_w_pos : 0 < w) :
  (2^5 ∣ 936 * w) ∧ (3^3 ∣ 936 * w) ∧ (12^2 ∣ 936 * w) → w = 36 :=
by
  sorry

end smallest_w_l149_149498


namespace max_sum_of_abc_l149_149351

theorem max_sum_of_abc (A B C : ℕ) (h1 : A * B * C = 1386) (h2 : A ≠ B) (h3 : A ≠ C) (h4 : B ≠ C) : 
  A + B + C ≤ 88 :=
sorry

end max_sum_of_abc_l149_149351


namespace student_B_speed_l149_149814

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l149_149814


namespace factorial_computation_l149_149583

theorem factorial_computation : (11! / (7! * 4!)) = 660 := by
  sorry

end factorial_computation_l149_149583


namespace alicia_tax_deduction_l149_149154

theorem alicia_tax_deduction :
  let hourly_wage_dollars := 20
  let hourly_wage_cents := hourly_wage_dollars * 100
  let tax_rate := 0.0145
  let tax_deduction_cents := tax_rate * hourly_wage_cents
  in tax_deduction_cents = 29 := by
sorry

end alicia_tax_deduction_l149_149154


namespace least_n_satisfies_condition_l149_149618

theorem least_n_satisfies_condition :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (m < n) → ¬(1 / (m : ℝ) - 1 / (m + 1) < 1 / 15)) ∧
    (1 / (n : ℝ) - 1 / (n + 1) < 1 / 15) :=
sorry

end least_n_satisfies_condition_l149_149618


namespace exists_q_r_polynomials_l149_149566

theorem exists_q_r_polynomials (n : ℕ) (p : Polynomial ℝ) 
  (h_deg : p.degree = n) 
  (h_monic : p.leadingCoeff = 1) :
  ∃ q r : Polynomial ℝ, 
    q.degree = n ∧ r.degree = n ∧ 
    (∀ x : ℝ, q.eval x = 0 → r.eval x = 0) ∧
    (∀ y : ℝ, r.eval y = 0 → q.eval y = 0) ∧
    q.leadingCoeff = 1 ∧ r.leadingCoeff = 1 ∧ 
    p = (q + r) / 2 := 
sorry

end exists_q_r_polynomials_l149_149566


namespace vandermonde_identity_combinatorial_identity_l149_149745

open Nat

-- Problem 1: Vandermonde Identity
theorem vandermonde_identity (m n k : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 0 < k) (h4 : m + n ≥ k) :
  (Finset.range (k + 1)).sum (λ i => Nat.choose m i * Nat.choose n (k - i)) = Nat.choose (m + n) k :=
sorry

-- Problem 2:
theorem combinatorial_identity (p q n : ℕ) (h1 : 0 < p) (h2 : 0 < q) (h3 : 0 < n) :
  (Finset.range (p + 1)).sum (λ k => Nat.choose p k * Nat.choose q k * Nat.choose (n + k) (p + q)) =
  Nat.choose n p * Nat.choose n q :=
sorry

end vandermonde_identity_combinatorial_identity_l149_149745


namespace interval_of_increase_inequality_for_large_x_l149_149008

open Real

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + log x

theorem interval_of_increase :
  ∀ x > 0, ∀ y > x, f y > f x :=
by
  sorry

theorem inequality_for_large_x (x : ℝ) (hx : x > 1) :
  (1/2) * x^2 + log x < (2/3) * x^3 :=
by
  sorry

end interval_of_increase_inequality_for_large_x_l149_149008


namespace rationalize_denominator_l149_149370

theorem rationalize_denominator : 
  (√12 + √5) / (√3 + √5) = (√15 - 1) / 2 :=
by sorry

end rationalize_denominator_l149_149370


namespace sum_of_prime_factors_eq_28_l149_149885

-- Define 2310 as a constant
def n : ℕ := 2310

-- Define the prime factors of 2310
def prime_factors : List ℕ := [2, 3, 5, 7, 11]

-- The sum of the prime factors
def sum_prime_factors : ℕ := prime_factors.sum

-- State the theorem
theorem sum_of_prime_factors_eq_28 : sum_prime_factors = 28 :=
by 
  sorry

end sum_of_prime_factors_eq_28_l149_149885


namespace bicycle_speed_B_l149_149770

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l149_149770


namespace share_of_a_is_240_l149_149096

theorem share_of_a_is_240 (A B C : ℝ) 
  (h1 : A = (2/3) * (B + C)) 
  (h2 : B = (2/3) * (A + C)) 
  (h3 : A + B + C = 600) : 
  A = 240 := 
by sorry

end share_of_a_is_240_l149_149096


namespace plywood_cut_difference_l149_149113

theorem plywood_cut_difference :
  let original_width := 6
  let original_height := 9
  let total_area := original_width * original_height
  let num_pieces := 6
  let area_per_piece := total_area / num_pieces
  -- Let possible perimeters based on given conditions
  let max_perimeter := 20
  let min_perimeter := 15
  max_perimeter - min_perimeter = 5 :=
by
  sorry

end plywood_cut_difference_l149_149113


namespace quadratic_general_form_coeffs_l149_149585

theorem quadratic_general_form_coeffs :
  ∀ (x : ℝ), (x - 1)^2 = 3 * x - 2 → (∃ a b c : ℝ, a = 1 ∧ b = -5 ∧ c = 3 ∧ a * x^2 + b * x + c = 0) :=
by
  intro x h,
  use [1, -5, 3],
  constructor; refl,
  constructor; refl,
  constructor; refl,
  rw [h],
  sorry

end quadratic_general_form_coeffs_l149_149585


namespace fraction_of_pelicans_moved_l149_149463

-- Conditions
variables (P : ℕ)
variables (n_Sharks : ℕ := 60) -- Number of sharks in Pelican Bay
variables (n_Pelicans_original : ℕ := 2 * P) -- Twice the original number of Pelicans in Shark Bite Cove
variables (n_Pelicans_remaining : ℕ := 20) -- Number of remaining Pelicans in Shark Bite Cove

-- Proof to show fraction that moved
theorem fraction_of_pelicans_moved (h : 2 * P = n_Sharks) : (P - n_Pelicans_remaining) / P = 1 / 3 :=
by {
  sorry
}

end fraction_of_pelicans_moved_l149_149463


namespace order_of_three_numbers_l149_149199

theorem order_of_three_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a * b) / (a + b) ≤ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≤ (a + b) / 2 :=
by
  sorry

end order_of_three_numbers_l149_149199


namespace max_binomial_term_l149_149868

theorem max_binomial_term (k : ℕ) :
  ∃ k : ℕ, k = 165 ∧
  ∀ m : ℕ, (m ≠ 165 → 
  (nat.choose 214 m * (real.sqrt 11) ^ m) < 
  (nat.choose 214 165 * (real.sqrt 11) ^ 165)) := 
sorry

end max_binomial_term_l149_149868


namespace cos_equivalence_l149_149013

open Real

theorem cos_equivalence (α : ℝ) (h : cos (π / 8 - α) = 1 / 6) : 
  cos (3 * π / 4 + 2 * α) = 17 / 18 :=
by
  sorry

end cos_equivalence_l149_149013


namespace pow_congr_mod_eight_l149_149702

theorem pow_congr_mod_eight (n : ℕ) : (5^n + 2 * 3^(n-1) + 1) % 8 = 0 := sorry

end pow_congr_mod_eight_l149_149702


namespace smallest_value_of_a_b_l149_149518

theorem smallest_value_of_a_b :
  ∃ (a b : ℤ), (∀ x : ℤ, ((x^2 + a*x + 20) = 0 ∨ (x^2 + 17*x + b) = 0) → x < 0) ∧ a + b = -5 :=
sorry

end smallest_value_of_a_b_l149_149518


namespace bicycle_speed_B_l149_149769

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l149_149769


namespace inverse_g_neg138_l149_149653

def g (x : ℝ) : ℝ := 5 * x^3 - 3

theorem inverse_g_neg138 :
  g (-3) = -138 :=
by
  sorry

end inverse_g_neg138_l149_149653


namespace inequality_transform_l149_149183

theorem inequality_transform (x y : ℝ) (h : x > y) : 2 * x + 1 > 2 * y + 1 := 
by {
  sorry
}

end inequality_transform_l149_149183


namespace student_b_speed_l149_149825

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l149_149825


namespace speed_of_student_B_l149_149831

-- Let x be the speed of student B in km/h.
-- Given that A's speed is 1.2 times B's speed and A arrives 10 minutes earlier than B 
-- for a distance of 12 km, we need to prove that B's speed is 2 * sqrt(3) km/h.

theorem speed_of_student_B (x : ℝ) (h1 : x > 0) :
  (∀ x, 1.2 * x > 0 ∧ ∀ h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6 → x = 2 * sqrt 3) :=
begin
  assume x h1,
  have h2 : ∀ t : ℕ, (12 / x) - (12 / (1.2 * x)) = 1 / 6,
  sorry,
end

end speed_of_student_B_l149_149831


namespace solve_equation_l149_149240

theorem solve_equation (x : ℝ) (floor : ℝ → ℤ) 
  (h_floor : ∀ y, floor y ≤ y ∧ y < floor y + 1) :
  (floor (20 * x + 23) = 20 + 23 * x) ↔ 
  (∃ n : ℤ, 20 ≤ n ∧ n ≤ 43 ∧ x = (n - 23) / 20) := 
by
  sorry

end solve_equation_l149_149240


namespace least_n_satisfies_inequality_l149_149638

theorem least_n_satisfies_inequality : ∃ n : ℕ, (1 : ℚ) / n - (1 : ℚ) / (n + 1) < 1 / 15 ∧ ∀ m : ℕ, (1 : ℚ) / m - (1 : ℚ) / (m + 1) < 1 / 15 -> ¬ (m < n) := 
sorry

end least_n_satisfies_inequality_l149_149638


namespace find_n_l149_149882

theorem find_n (n : ℤ) (hn : -180 ≤ n ∧ n ≤ 180) (hsin : Real.sin (n * Real.pi / 180) = Real.sin (750 * Real.pi / 180)) :
  n = 30 ∨ n = 150 ∨ n = -30 ∨ n = -150 :=
by
  sorry

end find_n_l149_149882


namespace price_of_first_candy_l149_149068

theorem price_of_first_candy (P: ℝ) 
  (total_weight: ℝ) (price_per_lb_mixture: ℝ) 
  (weight_first: ℝ) (weight_second: ℝ) 
  (price_per_lb_second: ℝ) :
  total_weight = 30 →
  price_per_lb_mixture = 3 →
  weight_first = 20 →
  weight_second = 10 →
  price_per_lb_second = 3.1 →
  20 * P + 10 * price_per_lb_second = total_weight * price_per_lb_mixture →
  P = 2.95 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end price_of_first_candy_l149_149068


namespace angle_B_in_arithmetic_sequence_l149_149207

theorem angle_B_in_arithmetic_sequence (A B C : ℝ) (h_triangle_sum : A + B + C = 180) (h_arithmetic_sequence : 2 * B = A + C) : B = 60 := 
by 
  -- proof omitted
  sorry

end angle_B_in_arithmetic_sequence_l149_149207


namespace angle_lateral_face_base_plane_l149_149401

-- Definitions of the conditions
def truncated_triangular_pyramid (P1 P2 A1 A2 B1 B2 C1 C2: Point) :=
  -- Geometry relationships to be filled
  sorry

def insphere (O: Point) (R: ℝ) : Sphere :=
  -- Geometry relationships to be filled
  sorry

axiom geometric_relationships : 
  ∀ (P1 P2 A1 A2 B1 B2 C1 C2 D1 D2 O: Point) (a b R: ℝ),
  truncated_triangular_pyramid P1 P2 A1 A2 B1 B2 C1 C2 → 
  insphere O R →
  midpoint D1 B1 C1 →
  midpoint D2 B2 C2 →
  -- Additional geometry conditions
  sorry

-- The given ratio condition
axiom surface_area_ratio : 
  ∀ (S_pyramid S_sphere: ℝ),
  S_sphere/S_pyramid = π / (6 * sqrt 3)

-- The problem to prove in Lean
theorem angle_lateral_face_base_plane (A1 A2 B1 B2 C1 C2 O D1 D2: Point) (a b R: ℝ) 
  (h_pyramid : truncated_triangular_pyramid A1 A2 B1 B2 C1 C2)
  (h_insphere : insphere O R)
  (h_midpoint1 : midpoint D1 B1 C1)
  (h_midpoint2 : midpoint D2 B2 C2)
  (h_surface_area_ratio : surface_area_ratio (surface_area_pyramid h_pyramid) (surface_area_insphere h_insphere))
  : angle_between (lateral_face A1 B1 C1 B2 C2) (base_plane A2 B2 C2) = arctan 2 :=
sorry

end angle_lateral_face_base_plane_l149_149401


namespace dot_product_expression_max_value_of_dot_product_l149_149490

variable (x : ℝ)
variable (k : ℤ)
variable (a : ℝ × ℝ := (Real.cos x, -1 + Real.sin x))
variable (b : ℝ × ℝ := (2 * Real.cos x, Real.sin x))

theorem dot_product_expression :
  (a.1 * b.1 + a.2 * b.2) = 2 - 3 * (Real.sin x)^2 - (Real.sin x) := 
sorry

theorem max_value_of_dot_product :
  ∃ (x : ℝ), 2 - 3 * (Real.sin x)^2 - (Real.sin x) = 9 / 4 ∧ 
  (Real.sin x = -1/2 ∧ 
  (x = 7 * Real.pi / 6 + 2 * k * Real.pi ∨ x = 11 * Real.pi / 6 + 2 * k * Real.pi)) := 
sorry

end dot_product_expression_max_value_of_dot_product_l149_149490


namespace alcohol_to_water_ratio_l149_149406

theorem alcohol_to_water_ratio (V p q : ℝ) (hV : V > 0) (hp : p > 0) (hq : q > 0) :
  let alcohol_first_jar := (p / (p + 1)) * V
  let water_first_jar   := (1 / (p + 1)) * V
  let alcohol_second_jar := (2 * q / (q + 1)) * V
  let water_second_jar   := (2 / (q + 1)) * V
  let total_alcohol := alcohol_first_jar + alcohol_second_jar
  let total_water := water_first_jar + water_second_jar
  (total_alcohol / total_water) = ((p * (q + 1) + 2 * p + 2 * q) / (q + 1 + 2 * p + 2)) :=
by
  sorry

end alcohol_to_water_ratio_l149_149406


namespace price_after_two_months_l149_149721

noncomputable def initial_price : ℝ := 1000

noncomputable def first_month_discount_rate : ℝ := 0.10
noncomputable def second_month_discount_rate : ℝ := 0.20

noncomputable def first_month_price (P0 : ℝ) (discount_rate1 : ℝ) : ℝ :=
  P0 - discount_rate1 * P0

noncomputable def second_month_price (P1 : ℝ) (discount_rate2 : ℝ) : ℝ :=
  P1 - discount_rate2 * P1

theorem price_after_two_months :
  let P0 := initial_price in
  let P1 := first_month_price P0 first_month_discount_rate in
  let P2 := second_month_price P1 second_month_discount_rate in
  P2 = 720 :=
by
  sorry

end price_after_two_months_l149_149721


namespace value_of_x_l149_149025

theorem value_of_x (x : ℝ) (h : x = 52 * (1 + 20 / 100)) : x = 62.4 :=
by sorry

end value_of_x_l149_149025


namespace polynomial_system_solution_l149_149999

variable {x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ}

theorem polynomial_system_solution (
  h1 : x₁ + 3 * x₂ + 5 * x₃ + 7 * x₄ + 9 * x₅ + 11 * x₆ + 13 * x₇ = 3)
  (h2 : 3 * x₁ + 5 * x₂ + 7 * x₃ + 9 * x₄ + 11 * x₅ + 13 * x₆ + 15 * x₇ = 15)
  (h3 : 5 * x₁ + 7 * x₂ + 9 * x₃ + 11 * x₄ + 13 * x₅ + 15 * x₆ + 17 * x₇ = 85) :
  7 * x₁ + 9 * x₂ + 11 * x₃ + 13 * x₄ + 15 * x₅ + 17 * x₆ + 19 * x₇ = 213 :=
sorry

end polynomial_system_solution_l149_149999


namespace sufficient_not_necessary_condition_l149_149399

theorem sufficient_not_necessary_condition (x : ℝ) : x - 1 > 0 → (x > 2) ∧ (¬ (x - 1 > 0 → x > 2)) :=
by
  sorry

end sufficient_not_necessary_condition_l149_149399


namespace product_of_roots_l149_149176

theorem product_of_roots :
  (let a := 36
   let b := -24
   let c := -120
   a ≠ 0) →
  let roots_product := c / a
  roots_product = -10/3 :=
by
  sorry

end product_of_roots_l149_149176


namespace probability_laurent_greater_chloe_l149_149160

open Probability Theory

noncomputable def chloe_distribution : ProbabilityMeasure ℝ := uniform 0 3000

noncomputable def laurent_distribution : ProbabilityMeasure ℝ := uniform 0 4000

theorem probability_laurent_greater_chloe :
  P(y > x) = 5 / 8 :=
by
  let x := random_variable chloe_distribution
  let y := random_variable laurent_distribution
  have h_uniform_x: x ~ U(0, 3000) := sorry
  have h_uniform_y: y ~ U(0, 4000) := sorry
  sorry

end probability_laurent_greater_chloe_l149_149160


namespace other_root_of_quadratic_l149_149000

theorem other_root_of_quadratic (m : ℝ) :
  (∃ t : ℝ, (x^2 + m * x - 20 = 0) ∧ (x = -4 ∨ x = t)) → (t = 5) :=
by
  sorry

end other_root_of_quadratic_l149_149000


namespace least_n_l149_149608

theorem least_n (n : ℕ) (h : n > 0) : 
  (1 / (n : ℝ) - 1 / (n + 1) < 1 / 15) ↔ n ≥ 4 := 
begin
  sorry
end

end least_n_l149_149608


namespace find_speed_of_B_l149_149806

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l149_149806


namespace number_of_smaller_cubes_l149_149429

theorem number_of_smaller_cubes 
  (volume_large_cube : ℝ)
  (volume_small_cube : ℝ)
  (surface_area_difference : ℝ)
  (h1 : volume_large_cube = 216)
  (h2 : volume_small_cube = 1)
  (h3 : surface_area_difference = 1080) :
  ∃ n : ℕ, n * 6 - 6 * (volume_large_cube^(1/3))^2 = surface_area_difference ∧ n = 216 :=
by
  sorry

end number_of_smaller_cubes_l149_149429


namespace number_of_plastic_bottles_l149_149075

-- Define the weights of glass and plastic bottles
variables (G P : ℕ)

-- Define the number of plastic bottles in the second scenario
variable (x : ℕ)

-- Define the conditions
def condition_1 := 3 * G = 600
def condition_2 := G = P + 150
def condition_3 := 4 * G + x * P = 1050

-- Proof that x is equal to 5 given the conditions
theorem number_of_plastic_bottles (h1 : condition_1 G) (h2 : condition_2 G P) (h3 : condition_3 G P x) : x = 5 :=
sorry

end number_of_plastic_bottles_l149_149075


namespace ratio_AD_DC_l149_149700

-- Definitions based on conditions
variable (A B C D : Point)
variable (AB BC AD DB : ℝ)
variable (h1 : AB = 2 * BC)
variable (h2 : AD = 3 / 5 * AB)
variable (h3 : DB = 2 / 5 * AB)

-- Lean statement for the problem
theorem ratio_AD_DC (h1 : AB = 2 * BC) (h2 : AD = 3 / 5 * AB) (h3 : DB = 2 / 5 * AB) :
  AD / (DB + BC) = 2 / 3 := 
by
  sorry

end ratio_AD_DC_l149_149700


namespace find_f_prime_at_1_l149_149468

def f (x : ℝ) (f_prime_at_1 : ℝ) : ℝ := x^2 + 3 * x * f_prime_at_1

theorem find_f_prime_at_1 (f_prime_at_1 : ℝ) :
  (∀ x, deriv (λ x => f x f_prime_at_1) x = 2 * x + 3 * f_prime_at_1) → 
  deriv (λ x => f x f_prime_at_1) 1 = -1 := 
by
exact sorry

end find_f_prime_at_1_l149_149468


namespace final_price_correct_l149_149720

-- Define the initial price of the iPhone
def initial_price : ℝ := 1000

-- Define the discount rates for the first and second month
def first_month_discount : ℝ := 0.10
def second_month_discount : ℝ := 0.20

-- Calculate the price after the first month's discount
def price_after_first_month (price : ℝ) : ℝ := price * (1 - first_month_discount)

-- Calculate the price after the second month's discount
def price_after_second_month (price : ℝ) : ℝ := price * (1 - second_month_discount)

-- Final price calculation after both discounts
def final_price : ℝ := price_after_second_month (price_after_first_month initial_price)

-- Proof statement
theorem final_price_correct : final_price = 720 := by
  sorry

end final_price_correct_l149_149720


namespace perpendicular_condition_centroid_coordinates_l149_149480

structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := -1, y := 0}
def B : Point := {x := 4, y := 0}
def C (c : ℝ) : Point := {x := 0, y := c}

def vec (P Q : Point) : Point :=
  {x := Q.x - P.x, y := Q.y - P.y}

def dot_product (P Q : Point) : ℝ :=
  P.x * Q.x + P.y * Q.y

theorem perpendicular_condition (c : ℝ) (h : dot_product (vec A (C c)) (vec B (C c)) = 0) :
  c = 2 ∨ c = -2 :=
by
  -- proof to be filled in
  sorry

theorem centroid_coordinates (c : ℝ) (h : c = 2 ∨ c = -2) :
  (c = 2 → Point.mk 1 (2 / 3) = Point.mk 1 (2 / 3)) ∧
  (c = -2 → Point.mk 1 (-2 / 3) = Point.mk 1 (-2 / 3)) :=
by
  -- proof to be filled in
  sorry

end perpendicular_condition_centroid_coordinates_l149_149480


namespace total_reading_materials_l149_149684

def reading_materials (magazines newspapers books pamphlets : ℕ) : ℕ :=
  magazines + newspapers + books + pamphlets

theorem total_reading_materials:
  reading_materials 425 275 150 75 = 925 := by
  sorry

end total_reading_materials_l149_149684


namespace wheels_motion_is_rotation_l149_149250

def motion_wheel_car := "rotation"
def question_wheels_motion := "What is the type of motion exhibited by the wheels of a moving car?"

theorem wheels_motion_is_rotation :
  (question_wheels_motion = "What is the type of motion exhibited by the wheels of a moving car?" ∧ 
   motion_wheel_car = "rotation") → motion_wheel_car = "rotation" :=
by
  sorry

end wheels_motion_is_rotation_l149_149250


namespace number_of_solutions_l149_149462

noncomputable def f (x : ℝ) : ℝ := x / 50
noncomputable def g (x : ℝ) : ℝ := Real.cos x

theorem number_of_solutions : 
  (Set.filter (λ x : ℝ, f x = g x) (Set.Icc (-50 : ℝ) 50)).card = 31 :=
by sorry

end number_of_solutions_l149_149462


namespace smallest_integer_2023m_54321n_l149_149965

theorem smallest_integer_2023m_54321n : ∃ (m n : ℤ), 2023 * m + 54321 * n = 1 :=
sorry

end smallest_integer_2023m_54321n_l149_149965


namespace student_b_speed_l149_149818

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l149_149818


namespace max_sin_sin2x_l149_149041

open Real

theorem max_sin_sin2x (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  ∃ x : ℝ, (0 < x ∧ x < π / 2) ∧ (sin x * sin (2 * x) = 4 * sqrt 3 / 9) := 
sorry

end max_sin_sin2x_l149_149041


namespace least_n_l149_149604

theorem least_n (n : ℕ) (h : (1 : ℝ) / n - (1 / (n + 1)) < (1 / 15)) : n = 4 :=
sorry

end least_n_l149_149604


namespace line_segment_intersection_range_l149_149319

theorem line_segment_intersection_range (P Q : ℝ × ℝ) (m : ℝ)
  (hP : P = (-1, 1)) (hQ : Q = (2, 2)) :
  ∃ m : ℝ, (x + m * y + m = 0) ∧ (-3 < m ∧ m < -2/3) := 
sorry

end line_segment_intersection_range_l149_149319


namespace finding_breadth_and_length_of_floor_l149_149266

noncomputable def length_of_floor (b : ℝ) := 3 * b
noncomputable def area_of_floor (b : ℝ) := (length_of_floor b) * b

theorem finding_breadth_and_length_of_floor
  (breadth : ℝ)
  (length : ℝ := length_of_floor breadth)
  (area : ℝ := area_of_floor breadth)
  (painting_cost : ℝ)
  (cost_per_sqm : ℝ)
  (h1 : painting_cost = 100)
  (h2 : cost_per_sqm = 2)
  (h3 : area = painting_cost / cost_per_sqm) :
  length = Real.sqrt 150 :=
by
  sorry

end finding_breadth_and_length_of_floor_l149_149266


namespace least_n_l149_149601

theorem least_n (n : ℕ) (h : (1 : ℝ) / n - (1 / (n + 1)) < (1 / 15)) : n = 4 :=
sorry

end least_n_l149_149601


namespace room_dimension_l149_149211

theorem room_dimension {a : ℝ} (h1 : a > 0) 
  (h2 : 4 = 2^2) 
  (h3 : 14 = 2 * (7)) 
  (h4 : 2 * a = 14) :
  (a + 2 * a - 2 = 19) :=
sorry

end room_dimension_l149_149211


namespace periodic_function_of_f_l149_149052

theorem periodic_function_of_f (f : ℝ → ℝ) (c : ℝ) (h : ∀ x, f (x + c) = (2 / (1 + f x)) - 1) : ∀ x, f (x + 2 * c) = f x :=
sorry

end periodic_function_of_f_l149_149052


namespace total_cups_l149_149699

theorem total_cups (n : ℤ) (h_rainy_days : n = 8) :
  let tea_cups := 6 * 3
  let total_cups := tea_cups + n
  total_cups = 26 :=
by
  let tea_cups := 6 * 3
  let total_cups := tea_cups + n
  exact sorry

end total_cups_l149_149699


namespace initial_worth_of_wears_l149_149230

theorem initial_worth_of_wears (W : ℝ) 
  (h1 : W + 2/5 * W = 1.4 * W)
  (h2 : 0.85 * (W + 2/5 * W) = W + 95) : 
  W = 500 := 
by 
  sorry

end initial_worth_of_wears_l149_149230


namespace least_n_satisfies_inequality_l149_149640

theorem least_n_satisfies_inequality : ∃ n : ℕ, (1 : ℚ) / n - (1 : ℚ) / (n + 1) < 1 / 15 ∧ ∀ m : ℕ, (1 : ℚ) / m - (1 : ℚ) / (m + 1) < 1 / 15 -> ¬ (m < n) := 
sorry

end least_n_satisfies_inequality_l149_149640


namespace solve_equation_l149_149449

theorem solve_equation (a b : ℤ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a^2 = b * (b + 7) ↔ (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
by
  sorry

end solve_equation_l149_149449


namespace find_speed_of_B_l149_149807

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l149_149807


namespace speed_of_student_B_l149_149789

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l149_149789


namespace least_n_l149_149610

theorem least_n (n : ℕ) (h : n > 0) : 
  (1 / (n : ℝ) - 1 / (n + 1) < 1 / 15) ↔ n ≥ 4 := 
begin
  sorry
end

end least_n_l149_149610


namespace count_positive_values_x_l149_149904

theorem count_positive_values_x :
  let valid_x := {x : ℕ | 25 ≤ x ∧ x ≤ 33} in
  set.finite valid_x ∧ set.count valid_x = 9 :=
by
  sorry

end count_positive_values_x_l149_149904


namespace total_winnings_l149_149958

theorem total_winnings (x : ℝ)
  (h1 : x / 4 = first_person_share)
  (h2 : x / 7 = second_person_share)
  (h3 : third_person_share = 17)
  (h4 : first_person_share + second_person_share + third_person_share = x) :
  x = 28 := 
by sorry

end total_winnings_l149_149958


namespace coin_change_problem_l149_149687

theorem coin_change_problem (d q h : ℕ) (n : ℕ) 
  (h1 : 2 * d + 5 * q + 10 * h = 240)
  (h2 : d ≥ 1)
  (h3 : q ≥ 1)
  (h4 : h ≥ 1) :
  n = 275 := 
sorry

end coin_change_problem_l149_149687


namespace all_terms_are_integers_l149_149537

open Nat

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 143 ∧ ∀ n ≥ 2, a (n + 1) = 5 * (Finset.range n).sum a / n

theorem all_terms_are_integers (a : ℕ → ℕ) (h : seq a) : ∀ n : ℕ, 1 ≤ n → ∃ k : ℕ, a n = k := 
by
  sorry

end all_terms_are_integers_l149_149537


namespace common_factor_extraction_l149_149089

-- Define the polynomial
def poly (a b c : ℝ) := 8 * a^3 * b^2 + 12 * a^3 * b * c - 4 * a^2 * b

-- Define the common factor
def common_factor (a b : ℝ) := 4 * a^2 * b

-- State the theorem
theorem common_factor_extraction (a b c : ℝ) :
  ∃ p : ℝ, poly a b c = common_factor a b * p := by
  sorry

end common_factor_extraction_l149_149089


namespace razorback_shop_tshirts_l149_149528

theorem razorback_shop_tshirts (T : ℕ) (h : 215 * T = 4300) : T = 20 :=
by sorry

end razorback_shop_tshirts_l149_149528


namespace water_percentage_in_fresh_mushrooms_l149_149179

theorem water_percentage_in_fresh_mushrooms
  (fresh_mushrooms_mass : ℝ)
  (dried_mushrooms_mass : ℝ)
  (dried_mushrooms_water_percentage : ℝ)
  (dried_mushrooms_non_water_mass : ℝ)
  (fresh_mushrooms_dry_percentage : ℝ)
  (fresh_mushrooms_water_percentage : ℝ)
  (h1 : fresh_mushrooms_mass = 22)
  (h2 : dried_mushrooms_mass = 2.5)
  (h3 : dried_mushrooms_water_percentage = 12 / 100)
  (h4 : dried_mushrooms_non_water_mass = dried_mushrooms_mass * (1 - dried_mushrooms_water_percentage))
  (h5 : fresh_mushrooms_dry_percentage = dried_mushrooms_non_water_mass / fresh_mushrooms_mass * 100)
  (h6 : fresh_mushrooms_water_percentage = 100 - fresh_mushrooms_dry_percentage) :
  fresh_mushrooms_water_percentage = 90 := 
by
  sorry

end water_percentage_in_fresh_mushrooms_l149_149179


namespace sqrt_same_type_as_sqrt_2_l149_149968

theorem sqrt_same_type_as_sqrt_2 (a b : ℝ) :
  ((sqrt a)^2 = 8) ↔ (sqrt 2) * (sqrt 2) = 2 * (sqrt 2) * (sqrt 2)  :=
sorry

end sqrt_same_type_as_sqrt_2_l149_149968


namespace student_B_speed_l149_149817

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l149_149817


namespace smallest_positive_x_l149_149560

theorem smallest_positive_x
  (x : ℕ)
  (h1 : x % 3 = 2)
  (h2 : x % 7 = 6)
  (h3 : x % 8 = 7) : x = 167 :=
by
  sorry

end smallest_positive_x_l149_149560


namespace James_age_is_47_5_l149_149440

variables (James_Age Mara_Age : ℝ)

def condition1 : Prop := James_Age = 3 * Mara_Age - 20
def condition2 : Prop := James_Age + Mara_Age = 70

theorem James_age_is_47_5 (h1 : condition1 James_Age Mara_Age) (h2 : condition2 James_Age Mara_Age) : James_Age = 47.5 :=
by
  sorry

end James_age_is_47_5_l149_149440


namespace plywood_perimeter_difference_l149_149138

noncomputable theory

open classical

theorem plywood_perimeter_difference :
  ∃ (rect1 rect2 : ℕ) (a b : ℕ),
  (rect1 = 6 ∧ rect2 = 9 ∧ rect1 % 6 = 0 ∧ rect2 % 6 = 0) ∧ 
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≤ 20) ∧
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≥ 10) ∧ 
  (20 - 10 = 10) :=
by
  exists (6, 9, 6, 9)
  sorry

end plywood_perimeter_difference_l149_149138


namespace speed_of_student_B_l149_149783

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end speed_of_student_B_l149_149783


namespace bicycle_speed_B_l149_149771

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l149_149771


namespace least_n_l149_149628

theorem least_n (n : ℕ) (h : 0 < n ∧ (1 / n - 1 / (n + 1) < 1 / 15)) : n ≥ 4 :=
by sorry

end least_n_l149_149628


namespace least_n_l149_149633

theorem least_n (n : ℕ) (h_pos : n > 0) (h_ineq : 1 / n - 1 / (n + 1) < 1 / 15) : n = 4 :=
sorry

end least_n_l149_149633


namespace nat_count_rel_prime_21_l149_149327
open Nat

def is_relatively_prime_to_21 (n : Nat) : Prop :=
  gcd n 21 = 1

theorem nat_count_rel_prime_21 : (∃ (N : Nat), N = 53 ∧ ∀ (n : Nat), 10 < n ∧ n < 100 ∧ is_relatively_prime_to_21 n → N = 53) :=
by {
  use 53,
  split,
  {
    refl,  -- 53 is the correct count given by the conditions
  },
  {
    intros n h1 h2 h3,
    sorry  -- proof skipped
  }
}

end nat_count_rel_prime_21_l149_149327


namespace unique_two_digit_factors_l149_149493

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def factors (n : ℕ) (a b : ℕ) : Prop := a * b = n

theorem unique_two_digit_factors : 
  ∃! (a b : ℕ), is_two_digit a ∧ is_two_digit b ∧ factors 1950 a b :=
by sorry

end unique_two_digit_factors_l149_149493


namespace corresponding_angles_equal_l149_149722

-- Definition of corresponding angles (this should be previously defined, so here we assume it is just a predicate)
def CorrespondingAngles (a b : Angle) : Prop := sorry

-- The main theorem to be proven
theorem corresponding_angles_equal (a b : Angle) (h : CorrespondingAngles a b) : a = b := 
sorry

end corresponding_angles_equal_l149_149722


namespace wechat_group_member_count_l149_149046

theorem wechat_group_member_count :
  (∃ x : ℕ, x * (x - 1) / 2 = 72) → ∃ x : ℕ, x = 9 :=
by
  sorry

end wechat_group_member_count_l149_149046


namespace candidate_1_fails_by_40_marks_l149_149141

-- Definitions based on the conditions
def total_marks (T : ℕ) := T
def passing_marks (pass : ℕ) := pass = 160
def candidate_1_failed_by (marks_failed_by : ℕ) := ∃ (T : ℕ), (0.4 : ℝ) * T = 0.4 * T ∧ (0.6 : ℝ) * T - 20 = 160

-- Theorem to prove the first candidate fails by 40 marks
theorem candidate_1_fails_by_40_marks (marks_failed_by : ℕ) : candidate_1_failed_by marks_failed_by → marks_failed_by = 40 :=
by
  sorry

end candidate_1_fails_by_40_marks_l149_149141


namespace fraction_special_phone_numbers_l149_149438

def valid_phone_numbers : ℕ := (7 * 9 * 10^5)

def special_phone_numbers : ℕ := 10^5

theorem fraction_special_phone_numbers :
  (special_phone_numbers : ℚ) / valid_phone_numbers = 1 / 63 := 
by
  sorry

end fraction_special_phone_numbers_l149_149438


namespace probability_abs_diff_gt_half_l149_149057

noncomputable def fair_coin : Probability :=
by sorry  -- Placeholder for fair coin flip definition

noncomputable def choose_real (using_coin : Probability) : Probability :=
by sorry  -- Placeholder for real number choice based on coin flip

def prob_abs_diff_gt_half : Probability :=
probability (abs (choose_real fair_coin - choose_real fair_coin) > 1 / 2)

theorem probability_abs_diff_gt_half :
  prob_abs_diff_gt_half = 7 / 16 := sorry

end probability_abs_diff_gt_half_l149_149057


namespace min_marks_required_l149_149735

-- Definitions and conditions
def grid_size := 7
def strip_size := 4

-- Question and answer as a proof statement
theorem min_marks_required (n : ℕ) (h : grid_size = 2 * n - 1) : 
  (∃ marks : ℕ, 
    (∀ row col : ℕ, 
      row < grid_size → col < grid_size → 
      (∃ i j : ℕ, 
        i < strip_size → j < strip_size → 
        (marks ≥ 12)))) :=
sorry

end min_marks_required_l149_149735


namespace functional_eq_solution_l149_149457

-- Define the conditions
variables (f g : ℕ → ℕ)

-- Define the main theorem
theorem functional_eq_solution :
  (∀ n : ℕ, f n + f (n + g n) = f (n + 1)) →
  ( (∀ n, f n = 0) ∨ 
    (∃ (n₀ c : ℕ), 
      (∀ n < n₀, f n = 0) ∧ 
      (∀ n ≥ n₀, f n = c * 2^(n - n₀)) ∧
      (∀ n < n₀ - 1, ∃ ck : ℕ, g n = ck) ∧
      g (n₀ - 1) = 1 ∧
      ∀ n ≥ n₀, g n = 0 ) ) := 
by
  intro h
  /- Proof goes here -/
  sorry

end functional_eq_solution_l149_149457


namespace find_huabei_number_l149_149508

theorem find_huabei_number :
  ∃ (hua bei sai : ℕ), 
    (hua ≠ 4 ∧ hua ≠ 8 ∧ bei ≠ 4 ∧ bei ≠ 8 ∧ sai ≠ 4 ∧ sai ≠ 8) ∧
    (hua ≠ bei ∧ hua ≠ sai ∧ bei ≠ sai) ∧
    (1 ≤ hua ∧ hua ≤ 9 ∧ 1 ≤ bei ∧ bei ≤ 9 ∧ 1 ≤ sai ∧ sai ≤ 9) ∧
    ((100 * hua + 10 * bei + sai) = 7632) :=
sorry

end find_huabei_number_l149_149508


namespace sqrt_div_value_l149_149071

open Real

theorem sqrt_div_value (n x : ℝ) (h1 : n = 3600) (h2 : sqrt n / x = 4) : x = 15 :=
by
  sorry

end sqrt_div_value_l149_149071


namespace hexagonal_pyramid_cross_section_distance_l149_149080

theorem hexagonal_pyramid_cross_section_distance
  (A1 A2 : ℝ) (distance_between_planes : ℝ)
  (A1_area : A1 = 125 * Real.sqrt 3)
  (A2_area : A2 = 500 * Real.sqrt 3)
  (distance_between_planes_eq : distance_between_planes = 10) :
  ∃ h : ℝ, h = 20 :=
by
  sorry

end hexagonal_pyramid_cross_section_distance_l149_149080


namespace max_whole_number_n_l149_149964

theorem max_whole_number_n (n : ℕ) : (1/2 + n/9 < 1) → n ≤ 4 :=
by
  sorry

end max_whole_number_n_l149_149964


namespace quadrilateral_ratio_l149_149517

theorem quadrilateral_ratio (AB CD AD BC IA IB IC ID : ℝ)
  (h_tangential : AB + CD = AD + BC)
  (h_IA : IA = 5)
  (h_IB : IB = 7)
  (h_IC : IC = 4)
  (h_ID : ID = 9) :
  AB / CD = 35 / 36 :=
by
  -- Proof will be provided here
  sorry

end quadrilateral_ratio_l149_149517


namespace find_p_l149_149899

theorem find_p
  (A B C r s p q : ℝ)
  (h1 : A ≠ 0)
  (h2 : r + s = -B / A)
  (h3 : r * s = C / A)
  (h4 : r^3 + s^3 = -p) :
  p = (B^3 - 3 * A * B * C) / A^3 :=
by {
  sorry
}

end find_p_l149_149899


namespace problem_l149_149318

variable (p q : Prop)

theorem problem (h : ¬ (¬ p ∨ ¬ q)) : (p ∧ q) ∧ (p ∨ q) :=
by
  sorry

end problem_l149_149318


namespace remainder_17_pow_63_mod_7_l149_149540

theorem remainder_17_pow_63_mod_7 :
  (17 ^ 63) % 7 = 6 :=
by {
  -- Given that 17 ≡ 3 (mod 7)
  have h1 : 17 % 7 = 3 := by norm_num,
  
  -- We need to show that (3 ^ 63) % 7 = 6.
  have h2 : (17 ^ 63) % 7 = (3 ^ 63) % 7 := by {
    rw ← h1,
    exact pow_mod_eq_mod_pow _ _ _
  },
  
  -- Now it suffices to show that (3 ^ 63) % 7 = 6
  have h3 : (3 ^ 63) % 7 = 6 := by {
    rw pow_eq_pow_mod 6, -- 63 = 6 * 10 + 3, so 3^63 = (3^6)^10 * 3^3
    have : 3 ^ 6 % 7 = 1 := by norm_num,
    rw [this, one_pow, one_mul, pow_mod_eq_pow_mod],
    exact_pow [exact_mod [norm_num]],
    exact rfl,
  },
  
  -- Combine both results
  exact h2 ▸ h3
}

end remainder_17_pow_63_mod_7_l149_149540


namespace problem_part1_problem_part2_l149_149003

def ellipse_condition (m : ℝ) : Prop :=
  m + 1 > 4 - m ∧ 4 - m > 0

def circle_condition (m : ℝ) : Prop :=
  m^2 - 4 > 0

theorem problem_part1 (m : ℝ) :
  ellipse_condition m → (3 / 2 < m ∧ m < 4) :=
sorry

theorem problem_part2 (m : ℝ) :
  ellipse_condition m ∧ circle_condition m → (2 < m ∧ m < 4) :=
sorry

end problem_part1_problem_part2_l149_149003


namespace maximum_even_integers_of_odd_product_l149_149283

theorem maximum_even_integers_of_odd_product (a b c d e f g : ℕ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: d > 0) (h5: e > 0) (h6: f > 0) (h7: g > 0) (hprod : a * b * c * d * e * f * g % 2 = 1): 
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (d % 2 = 1) ∧ (e % 2 = 1) ∧ (f % 2 = 1) ∧ (g % 2 = 1) :=
sorry

end maximum_even_integers_of_odd_product_l149_149283


namespace miyoung_largest_square_side_l149_149031

theorem miyoung_largest_square_side :
  ∃ (G : ℕ), G > 0 ∧ ∀ (a b : ℕ), (a = 32) → (b = 74) → (gcd a b = G) → (G = 2) :=
by {
  sorry
}

end miyoung_largest_square_side_l149_149031


namespace plywood_cut_perimeter_difference_l149_149118

theorem plywood_cut_perimeter_difference :
  (∃ (l w : ℕ), (l * w = 54) ∧ (9 % w = 0) ∧ (6 % l = 0) ∧ (6 / l) * (9 / w) = 6) →
  10 =
  let p := λ l w, 2 * (l + w) in
  let perimeters := [
    p 1 9,
    p 1 6,
    p 2 3,
    p 3 2
  ]
  in (list.max precedence perimeters - list.min precedence perimeters) :=
begin
  sorry
end

end plywood_cut_perimeter_difference_l149_149118


namespace annual_growth_rate_l149_149070

-- definitions based on the conditions in the problem
def FirstYear : ℝ := 400
def ThirdYear : ℝ := 625
def n : ℕ := 2

-- the main statement to prove the corresponding equation
theorem annual_growth_rate (x : ℝ) : 400 * (1 + x)^2 = 625 :=
sorry

end annual_growth_rate_l149_149070


namespace intersection_points_of_parametric_curve_l149_149244

def parametric_curve_intersection_points (t : ℝ) : Prop :=
  let x := t - 1
  let y := t + 2
  (x = -3 ∧ y = 0) ∨ (x = 0 ∧ y = 3)

theorem intersection_points_of_parametric_curve :
  ∃ t1 t2 : ℝ, parametric_curve_intersection_points t1 ∧ parametric_curve_intersection_points t2 := 
by
  sorry

end intersection_points_of_parametric_curve_l149_149244


namespace perimeter_difference_l149_149128

-- Definitions as per conditions
def plywood_width : ℕ := 6
def plywood_height : ℕ := 9
def rectangles_count : ℕ := 6

-- The perimeter difference to be proved
theorem perimeter_difference : 
  ∃ (max_perimeter min_perimeter : ℕ), 
  max_perimeter = 22 ∧ min_perimeter = 12 ∧ (max_perimeter - min_perimeter = 10) :=
by
  sorry

end perimeter_difference_l149_149128


namespace tan_double_angle_sub_l149_149596

theorem tan_double_angle_sub (α β : ℝ) 
  (h1 : Real.tan α = 1 / 2)
  (h2 : Real.tan (α - β) = 1 / 5) : Real.tan (2 * α - β) = 7 / 9 :=
by
  sorry

end tan_double_angle_sub_l149_149596


namespace student_B_speed_l149_149776

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l149_149776


namespace inscribed_rectangle_area_l149_149279

theorem inscribed_rectangle_area (A S x : ℝ) (hA : A = 18) (hS : S = (x * x) * 2) (hx : x = 2):
  S = 8 :=
by
  -- The proofs steps will go here
  sorry

end inscribed_rectangle_area_l149_149279


namespace exists_face_with_fewer_than_six_sides_l149_149369

theorem exists_face_with_fewer_than_six_sides
  (N K M : ℕ) 
  (h_euler : N - K + M = 2)
  (h_vertices : M ≤ 2 * K / 3) : 
  ∃ n_i : ℕ, n_i < 6 :=
by
  sorry

end exists_face_with_fewer_than_six_sides_l149_149369


namespace number_division_l149_149060

theorem number_division (x : ℤ) (h : x - 17 = 55) : x / 9 = 8 :=
by 
  sorry

end number_division_l149_149060


namespace jorge_total_spent_l149_149681

-- Definitions based on the problem conditions
def price_adult_ticket : ℝ := 10
def price_child_ticket : ℝ := 5
def num_adult_tickets : ℕ := 12
def num_child_tickets : ℕ := 12
def discount_adult : ℝ := 0.40
def discount_child : ℝ := 0.30
def extra_discount : ℝ := 0.10

-- The desired statement to prove
theorem jorge_total_spent :
  let total_adult_cost := num_adult_tickets * price_adult_ticket
  let total_child_cost := num_child_tickets * price_child_ticket
  let discounted_adult := total_adult_cost * (1 - discount_adult)
  let discounted_child := total_child_cost * (1 - discount_child)
  let total_cost_before_extra_discount := discounted_adult + discounted_child
  let final_cost := total_cost_before_extra_discount * (1 - extra_discount)
  final_cost = 102.60 :=
by 
  sorry

end jorge_total_spent_l149_149681


namespace urn_problem_l149_149264

noncomputable def probability_of_two_black_balls : ℚ := (10 / 15) * (9 / 14)

theorem urn_problem : probability_of_two_black_balls = 3 / 7 := 
by
  sorry

end urn_problem_l149_149264


namespace max_point_diff_l149_149027

theorem max_point_diff (n : ℕ) : ∃ max_diff, max_diff = 2 :=
by
  -- Conditions from (a)
  -- - \( n \) teams participate in a football tournament.
  -- - Each team plays against every other team exactly once.
  -- - The winning team is awarded 2 points.
  -- - A draw gives -1 point to each team.
  -- - The losing team gets 0 points.
  -- Correct Answer from (b)
  -- - The maximum point difference between teams that are next to each other in the ranking is 2.
  sorry

end max_point_diff_l149_149027


namespace prove_inequalities_l149_149901

variable {a b c R r_a r_b r_c : ℝ}

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def has_circumradius (a b c R : ℝ) : Prop :=
  ∃ S : ℝ, S = a * b * c / (4 * R)

def has_exradii (a b c r_a r_b r_c : ℝ) : Prop :=
  ∃ S : ℝ, 
    r_a = 2 * S / (b + c - a) ∧
    r_b = 2 * S / (a + c - b) ∧
    r_c = 2 * S / (a + b - c)

theorem prove_inequalities
  (h_triangle : is_triangle a b c)
  (h_circumradius : has_circumradius a b c R)
  (h_exradii : has_exradii a b c r_a r_b r_c)
  (h_two_R_le_r_a : 2 * R ≤ r_a) :
  a > b ∧ a > c ∧ 2 * R > r_b ∧ 2 * R > r_c := 
sorry

end prove_inequalities_l149_149901


namespace find_2xy2_l149_149591

theorem find_2xy2 (x y : ℤ) (h : y^2 + 2 * x^2 * y^2 = 20 * x^2 + 412) : 2 * x * y^2 = 288 :=
sorry

end find_2xy2_l149_149591


namespace total_produce_of_mangoes_is_400_l149_149697

variable (A M O : ℕ)  -- Defines variables for total produce of apples, mangoes, and oranges respectively
variable (P : ℕ := 50)  -- Price per kg
variable (R : ℕ := 90000)  -- Total revenue

-- Definition of conditions
def apples_total_produce := 2 * M
def oranges_total_produce := M + 200
def total_weight_of_fruits := apples_total_produce + M + oranges_total_produce

-- Statement to prove
theorem total_produce_of_mangoes_is_400 :
  (total_weight_of_fruits = R / P) → (M = 400) :=
by
  sorry

end total_produce_of_mangoes_is_400_l149_149697


namespace intersection_of_A_and_B_l149_149900

def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x < 1} := 
by
  sorry

end intersection_of_A_and_B_l149_149900


namespace find_functions_l149_149878

noncomputable def satisfies_condition (f : ℝ → ℝ) :=
  ∀ (p q r s : ℝ), p > 0 → q > 0 → r > 0 → s > 0 →
  (p * q = r * s) →
  (f p ^ 2 + f q ^ 2) / (f (r ^ 2) + f (s ^ 2)) = 
  (p ^ 2 + q ^ 2) / (r ^ 2 + s ^ 2)

theorem find_functions :
  ∀ (f : ℝ → ℝ),
  (satisfies_condition f) → 
  (∀ x : ℝ, x > 0 → f x = x ∨ f x = 1 / x) :=
by
  sorry

end find_functions_l149_149878


namespace total_weight_of_fish_l149_149428

theorem total_weight_of_fish (fry : ℕ) (survival_rate : ℚ) 
  (first_catch : ℕ) (first_avg_weight : ℚ) 
  (second_catch : ℕ) (second_avg_weight : ℚ)
  (third_catch : ℕ) (third_avg_weight : ℚ)
  (total_weight : ℚ) :
  fry = 100000 ∧ 
  survival_rate = 0.95 ∧ 
  first_catch = 40 ∧ 
  first_avg_weight = 2.5 ∧ 
  second_catch = 25 ∧ 
  second_avg_weight = 2.2 ∧ 
  third_catch = 35 ∧ 
  third_avg_weight = 2.8 ∧ 
  total_weight = fry * survival_rate * 
    ((first_catch * first_avg_weight + 
      second_catch * second_avg_weight + 
      third_catch * third_avg_weight) / 100) / 10000 →
  total_weight = 24 :=
by
  sorry

end total_weight_of_fish_l149_149428


namespace find_x_l149_149496

theorem find_x
  (x : ℕ)
  (h1 : x % 7 = 0)
  (h2 : x > 0)
  (h3 : x^2 > 144)
  (h4 : x < 25) : x = 14 := 
  sorry

end find_x_l149_149496


namespace total_oranges_l149_149956

theorem total_oranges (a b c : ℕ) 
  (h₁ : a = 22) 
  (h₂ : b = a + 17) 
  (h₃ : c = b - 11) : 
  a + b + c = 89 := 
by
  sorry

end total_oranges_l149_149956


namespace cricket_player_average_increase_l149_149425

theorem cricket_player_average_increase (total_innings initial_innings next_run : ℕ) (initial_average desired_increase : ℕ) 
(h1 : initial_innings = 10) (h2 : initial_average = 32) (h3 : next_run = 76) : desired_increase = 4 :=
by
  sorry

end cricket_player_average_increase_l149_149425


namespace range_of_m_l149_149948

open Real

noncomputable def f : ℝ → ℝ :=
  λ x, 2 * sin (π / 4 + x) - sqrt 3 * cos (2 * x)

theorem range_of_m (t : ℝ) (m : ℝ) :
  (0 < t ∧ t < π) →
  (∀ x, (π / 4 ≤ x ∧ x ≤ π / 2) → abs (f x - m) < 3) →
  -1 < m ∧ m < 4 :=
by
  sorry

end range_of_m_l149_149948


namespace resulting_solution_percentage_l149_149987

theorem resulting_solution_percentage (w_original: ℝ) (w_replaced: ℝ) (c_original: ℝ) (c_new: ℝ) :
  c_original = 0.9 → w_replaced = 0.7142857142857143 → c_new = 0.2 →
  (0.2571428571428571 + 0.14285714285714285) / (0.2857142857142857 + 0.7142857142857143) * 100 = 40 := 
by
  intros h1 h2 h3
  sorry

end resulting_solution_percentage_l149_149987


namespace martins_travel_time_l149_149366

-- Declare the necessary conditions from the problem
variables (speed : ℝ) (distance : ℝ)
-- Define the conditions
def martin_speed := speed = 12 -- Martin's speed is 12 miles per hour
def martin_distance := distance = 72 -- Martin drove 72 miles

-- State the theorem to prove the time taken is 6 hours
theorem martins_travel_time (h1 : martin_speed speed) (h2 : martin_distance distance) : distance / speed = 6 :=
by
  -- To complete the problem statement, insert sorry to skip the actual proof
  sorry

end martins_travel_time_l149_149366


namespace simple_interest_true_discount_l149_149267

theorem simple_interest_true_discount (P R T : ℝ) 
  (h1 : 85 = (P * R * T) / 100)
  (h2 : 80 = (85 * P) / (P + 85)) : P = 1360 :=
sorry

end simple_interest_true_discount_l149_149267


namespace katrina_tax_deduction_l149_149216

variable (hourlyWage : ℚ) (taxRate : ℚ)

def wageInCents (wage : ℚ) : ℚ := wage * 100
def taxInCents (wageInCents : ℚ) (rate : ℚ) : ℚ := wageInCents * rate / 100

theorem katrina_tax_deduction : 
  hourlyWage = 25 ∧ taxRate = 2.5 → taxInCents (wageInCents hourlyWage) taxRate = 62.5 := 
by 
  sorry

end katrina_tax_deduction_l149_149216


namespace find_triples_l149_149880

theorem find_triples (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x^2 + y^2 = 3 * 2016^z + 77) :
  (x = 4 ∧ y = 8 ∧ z = 0) ∨ (x = 8 ∧ y = 4 ∧ z = 0) ∨
  (x = 14 ∧ y = 77 ∧ z = 1) ∨ (x = 77 ∧ y = 14 ∧ z = 1) ∨
  (x = 35 ∧ y = 70 ∧ z = 1) ∨ (x = 70 ∧ y = 35 ∧ z = 1) :=
sorry

end find_triples_l149_149880


namespace boxes_with_neither_l149_149680

def total_boxes : ℕ := 15
def boxes_with_stickers : ℕ := 9
def boxes_with_stamps : ℕ := 5
def boxes_with_both : ℕ := 3

theorem boxes_with_neither
  (total_boxes : ℕ)
  (boxes_with_stickers : ℕ)
  (boxes_with_stamps : ℕ)
  (boxes_with_both : ℕ) :
  total_boxes - ((boxes_with_stickers + boxes_with_stamps) - boxes_with_both) = 4 :=
by
  sorry

end boxes_with_neither_l149_149680


namespace find_rectangle_dimensions_area_30_no_rectangle_dimensions_area_32_l149_149962

noncomputable def length_width_rectangle_area_30 : Prop :=
∃ (x y : ℝ), x * y = 30 ∧ 2 * (x + y) = 22 ∧ x = 6 ∧ y = 5

noncomputable def impossible_rectangle_area_32 : Prop :=
¬(∃ (x y : ℝ), x * y = 32 ∧ 2 * (x + y) = 22)

-- Proof statements (without proofs)
theorem find_rectangle_dimensions_area_30 : length_width_rectangle_area_30 :=
sorry

theorem no_rectangle_dimensions_area_32 : impossible_rectangle_area_32 :=
sorry

end find_rectangle_dimensions_area_30_no_rectangle_dimensions_area_32_l149_149962


namespace least_n_inequality_l149_149615

theorem least_n_inequality (n : ℕ) (hn : 0 < n) (ineq: 1 / n - 1 / (n + 1) < 1 / 15) :
  4 ≤ n :=
begin
  simp [lt_div_iff] at ineq,
  rw [sub_eq_add_neg, add_div_eq_mul_add_neg_div, ←lt_div_iff], 
  exact ineq
end

end least_n_inequality_l149_149615


namespace polynomial_divisibility_l149_149718

theorem polynomial_divisibility (C D : ℝ) (h : ∀ x : ℂ, x^2 + x + 1 = 0 → x^104 + C * x + D = 0) :
  C + D = 2 := 
sorry

end polynomial_divisibility_l149_149718


namespace find_speed_B_l149_149858

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l149_149858
