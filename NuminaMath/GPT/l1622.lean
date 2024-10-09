import Mathlib

namespace necessary_but_not_sufficient_l1622_162264

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 ≥ 1) → (¬(x ≥ 1) ∨ (x ≥ 1)) :=
by
  sorry

end necessary_but_not_sufficient_l1622_162264


namespace evaluate_magnitude_l1622_162249

noncomputable def mag1 : ℂ := 3 * Real.sqrt 2 - 3 * Complex.I
noncomputable def mag2 : ℂ := Real.sqrt 5 + 5 * Complex.I
noncomputable def mag3 : ℂ := 2 - 2 * Complex.I

theorem evaluate_magnitude :
  Complex.abs (mag1 * mag2 * mag3) = 18 * Real.sqrt 10 :=
by
  sorry

end evaluate_magnitude_l1622_162249


namespace square_with_12_sticks_square_with_15_sticks_l1622_162263

-- Definitions for problem conditions
def sum_of_first_n_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def can_form_square (total_length : ℕ) : Prop :=
  total_length % 4 = 0

-- Given n = 12, check if breaking 2 sticks is required to form a square
theorem square_with_12_sticks : (n = 12) → ¬ can_form_square (sum_of_first_n_natural_numbers 12) → true :=
by
  intros
  sorry

-- Given n = 15, check if it is possible to form a square without breaking any sticks
theorem square_with_15_sticks : (n = 15) → can_form_square (sum_of_first_n_natural_numbers 15) → true :=
by
  intros
  sorry

end square_with_12_sticks_square_with_15_sticks_l1622_162263


namespace range_of_a_l1622_162284

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 - a * x - a ≤ -3) ↔ (a ≤ -6 ∨ a ≥ 2) :=
by
  sorry

end range_of_a_l1622_162284


namespace train_speed_l1622_162255

theorem train_speed
  (distance: ℝ) (time_in_minutes : ℝ) (time_in_hours : ℝ) (speed: ℝ)
  (h1 : distance = 20)
  (h2 : time_in_minutes = 10)
  (h3 : time_in_hours = time_in_minutes / 60)
  (h4 : speed = distance / time_in_hours)
  : speed = 120 := 
by
  sorry

end train_speed_l1622_162255


namespace crayon_division_l1622_162202

theorem crayon_division (total_crayons : ℕ) (crayons_each : ℕ) (Fred Benny Jason : ℕ) 
  (h_total : total_crayons = 24) (h_each : crayons_each = 8) 
  (h_division : Fred = crayons_each ∧ Benny = crayons_each ∧ Jason = crayons_each) : 
  Fred + Benny + Jason = total_crayons :=
by
  sorry

end crayon_division_l1622_162202


namespace educated_employees_count_l1622_162217

def daily_wages_decrease (illiterate_avg_before illiterate_avg_after illiterate_count : ℕ) : ℕ :=
  (illiterate_avg_before - illiterate_avg_after) * illiterate_count

def total_employees (total_decreased total_avg_decreased : ℕ) : ℕ :=
  total_decreased / total_avg_decreased

theorem educated_employees_count :
  ∀ (illiterate_avg_before illiterate_avg_after illiterate_count total_avg_decreased : ℕ),
    illiterate_avg_before = 25 →
    illiterate_avg_after = 10 →
    illiterate_count = 20 →
    total_avg_decreased = 10 →
    total_employees (daily_wages_decrease illiterate_avg_before illiterate_avg_after illiterate_count) total_avg_decreased - illiterate_count = 10 :=
by
  intros
  sorry

end educated_employees_count_l1622_162217


namespace problem_a_eq_2_problem_a_real_pos_problem_a_real_zero_problem_a_real_neg_l1622_162244

theorem problem_a_eq_2 (x : ℝ) : (12 * x^2 - 2 * x > 4) ↔ (x < -1 / 2 ∨ x > 2 / 3) := sorry

theorem problem_a_real_pos (a x : ℝ) (h : a > 0) : (12 * x^2 - a * x > a^2) ↔ (x < -a / 4 ∨ x > a / 3) := sorry

theorem problem_a_real_zero (x : ℝ) : (12 * x^2 > 0) ↔ (x ≠ 0) := sorry

theorem problem_a_real_neg (a x : ℝ) (h : a < 0) : (12 * x^2 - a * x > a^2) ↔ (x < a / 3 ∨ x > -a / 4) := sorry

end problem_a_eq_2_problem_a_real_pos_problem_a_real_zero_problem_a_real_neg_l1622_162244


namespace number_of_clothes_hangers_l1622_162270

noncomputable def total_money : ℝ := 60
noncomputable def spent_on_tissues : ℝ := 34.8
noncomputable def price_per_hanger : ℝ := 1.6

theorem number_of_clothes_hangers : 
  let remaining_money := total_money - spent_on_tissues
  let hangers := remaining_money / price_per_hanger
  Int.floor hangers = 15 := 
by
  sorry

end number_of_clothes_hangers_l1622_162270


namespace parabola_inequality_l1622_162203

theorem parabola_inequality {y1 y2 : ℝ} :
  (∀ x1 x2 : ℝ, x1 = -5 → x2 = 2 →
  y1 = x1^2 + 2 * x1 + 3 ∧ y2 = x2^2 + 2 * x2 + 3) → (y1 > y2) :=
by
  intros h
  sorry

end parabola_inequality_l1622_162203


namespace sum_of_a_values_l1622_162252

theorem sum_of_a_values : 
  (∀ (a x : ℝ), (a + x) / 2 ≥ x - 2 ∧ x / 3 - (x - 2) > 2 / 3 ∧ 
  (x - 1) / (4 - x) + (a + 5) / (x - 4) = -4 ∧ x < 2 ∧ (∃ n : ℤ, x = n ∧ 0 < n)) →
  ∃ I : ℤ, I = 12 :=
by
  sorry

end sum_of_a_values_l1622_162252


namespace spent_on_burgers_l1622_162204

noncomputable def money_spent_on_burgers (total_allowance : ℝ) (movie_fraction music_fraction ice_cream_fraction : ℝ) : ℝ :=
  let movie_expense := (movie_fraction * total_allowance)
  let music_expense := (music_fraction * total_allowance)
  let ice_cream_expense := (ice_cream_fraction * total_allowance)
  total_allowance - (movie_expense + music_expense + ice_cream_expense)

theorem spent_on_burgers : 
  money_spent_on_burgers 50 (1/4) (3/10) (2/5) = 2.5 :=
by sorry

end spent_on_burgers_l1622_162204


namespace difference_of_squares_l1622_162278

theorem difference_of_squares (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by
  sorry

end difference_of_squares_l1622_162278


namespace phase_shift_cos_l1622_162243

theorem phase_shift_cos (b c : ℝ) (h_b : b = 2) (h_c : c = π / 2) :
  (-c / b) = -π / 4 :=
by
  rw [h_b, h_c]
  sorry

end phase_shift_cos_l1622_162243


namespace negation_of_proposition_l1622_162248

-- Define the original proposition and its negation
def original_proposition (x : ℝ) : Prop := x^2 - 3*x + 3 > 0
def negated_proposition (x : ℝ) : Prop := x^2 - 3*x + 3 ≤ 0

-- The theorem about the negation of the original proposition
theorem negation_of_proposition :
  ¬ (∀ x : ℝ, original_proposition x) ↔ ∃ x : ℝ, negated_proposition x :=
by
  sorry

end negation_of_proposition_l1622_162248


namespace arithmetic_seq_problem_l1622_162246

theorem arithmetic_seq_problem
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a (n + 1) - a n = d)
  (h2 : d > 0)
  (h3 : a 1 + a 2 + a 3 = 15)
  (h4 : a 1 * a 2 * a 3 = 80) :
  a 11 + a 12 + a 13 = 105 :=
sorry

end arithmetic_seq_problem_l1622_162246


namespace find_k_l1622_162212

-- Definitions of given vectors and the condition that the vectors are parallel.
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 4)

-- Condition for vectors to be parallel in 2D is that their cross product is zero.
def parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem find_k : ∀ k : ℝ, parallel vector_a (vector_b k) → k = -2 :=
by
  intro k
  intro h
  sorry

end find_k_l1622_162212


namespace trays_needed_to_fill_ice_cubes_l1622_162210

-- Define the initial conditions
def ice_cubes_in_glass : Nat := 8
def multiplier_for_pitcher : Nat := 2
def spaces_per_tray : Nat := 12

-- Define the total ice cubes used
def total_ice_cubes_used : Nat := ice_cubes_in_glass + multiplier_for_pitcher * ice_cubes_in_glass

-- State the Lean theorem to be proven: The number of trays needed
theorem trays_needed_to_fill_ice_cubes : 
  total_ice_cubes_used / spaces_per_tray = 2 :=
  by 
  sorry

end trays_needed_to_fill_ice_cubes_l1622_162210


namespace calculate_expression_l1622_162274

theorem calculate_expression (b : ℝ) (hb : b ≠ 0) : 
  (1 / 25) * b^0 + (1 / (25 * b))^0 - 81^(-1 / 4 : ℝ) - (-27)^(-1 / 3 : ℝ) = 26 / 25 :=
by sorry

end calculate_expression_l1622_162274


namespace necessary_and_sufficient_condition_for_f_to_be_odd_l1622_162299

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def f (a b x : ℝ) : ℝ :=
  x * abs (x + a) + b

theorem necessary_and_sufficient_condition_for_f_to_be_odd (a b : ℝ) :
  is_odd_function (f a b) ↔ sorry :=
by
  -- This is where the proof would go.
  sorry

end necessary_and_sufficient_condition_for_f_to_be_odd_l1622_162299


namespace total_items_on_shelf_l1622_162259

-- Given conditions
def initial_action_figures : Nat := 4
def initial_books : Nat := 22
def initial_video_games : Nat := 10

def added_action_figures : Nat := 6
def added_video_games : Nat := 3
def removed_books : Nat := 5

-- Definitions based on conditions
def final_action_figures : Nat := initial_action_figures + added_action_figures
def final_books : Nat := initial_books - removed_books
def final_video_games : Nat := initial_video_games + added_video_games

-- Claim to prove
theorem total_items_on_shelf : final_action_figures + final_books + final_video_games = 40 := by
  sorry

end total_items_on_shelf_l1622_162259


namespace continuous_stripe_encircling_tetrahedron_probability_l1622_162289

noncomputable def tetrahedron_continuous_stripe_probability : ℚ :=
  let total_combinations := 3^4
  let favorable_combinations := 2 
  favorable_combinations / total_combinations

theorem continuous_stripe_encircling_tetrahedron_probability :
  tetrahedron_continuous_stripe_probability = 2 / 81 :=
by
  -- the proof would be here
  sorry

end continuous_stripe_encircling_tetrahedron_probability_l1622_162289


namespace compute_100p_plus_q_l1622_162224

-- Given constants p, q under the provided conditions,
-- prove the result: 100p + q = 430 / 3.
theorem compute_100p_plus_q (p q : ℚ) 
  (h1 : ∀ x : ℚ, (x + p) * (x + q) * (x + 20) = 0 → x ≠ -4)
  (h2 : ∀ x : ℚ, (x + 3 * p) * (x + 4) * (x + 10) = 0 → (x = -4 ∨ x ≠ -4)) :
  100 * p + q = 430 / 3 := 
by 
  sorry

end compute_100p_plus_q_l1622_162224


namespace solve_perimeter_l1622_162220

noncomputable def ellipse_perimeter_proof : Prop :=
  let a := 4
  let b := Real.sqrt 7
  let c := 3
  let F1 := (-c, 0)
  let F2 := (c, 0)
  let ellipse_eq (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 7) = 1
  ∀ (A B : ℝ×ℝ), 
    (ellipse_eq A.1 A.2) ∧ (ellipse_eq B.1 B.2) ∧ (∃ l : ℝ, l ≠ 0 ∧ ∀ t : ℝ, (A = (F1.1 + t * l, F1.2 + t * l)) ∨ (B = (F1.1 + t * l, F1.2 + t * l))) 
    → ∃ P : ℝ, P = 16

theorem solve_perimeter : ellipse_perimeter_proof := sorry

end solve_perimeter_l1622_162220


namespace ratio_height_radius_l1622_162273

variable (V r h : ℝ)

theorem ratio_height_radius (h_eq_2r : h = 2 * r) (volume_eq : π * r^2 * h = V) : h / r = 2 :=
by
  sorry

end ratio_height_radius_l1622_162273


namespace complex_multiplication_l1622_162261

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (1 + i) = -1 + i :=
by
  sorry

end complex_multiplication_l1622_162261


namespace iron_balls_molded_l1622_162254

noncomputable def volume_of_iron_bar (l w h : ℝ) : ℝ :=
  l * w * h

theorem iron_balls_molded (l w h n : ℝ) (volume_of_ball : ℝ) 
  (h_l : l = 12) (h_w : w = 8) (h_h : h = 6) (h_n : n = 10) (h_ball_volume : volume_of_ball = 8) :
  (n * volume_of_iron_bar l w h) / volume_of_ball = 720 :=
by 
  rw [h_l, h_w, h_h, h_n, h_ball_volume]
  rw [volume_of_iron_bar]
  sorry

end iron_balls_molded_l1622_162254


namespace evaluate_expression_at_4_l1622_162294

theorem evaluate_expression_at_4 :
  ∀ x : ℝ, x = 4 → (x^2 - 3 * x - 10) / (x - 5) = 6 :=
by
  intro x
  intro hx
  sorry

end evaluate_expression_at_4_l1622_162294


namespace infinite_integer_solutions_l1622_162226

theorem infinite_integer_solutions (a b c k : ℤ) (D : ℤ) 
  (hD : D = b^2 - 4 * a * c) (hD_pos : D > 0) (hD_non_square : ¬ ∃ (n : ℤ), n^2 = D) 
  (hk_non_zero : k ≠ 0) :
  (∃ (x₀ y₀ : ℤ), a * x₀^2 + b * x₀ * y₀ + c * y₀^2 = k) →
  ∃ (f : ℤ → ℤ × ℤ), ∀ n : ℤ, a * (f n).1^2 + b * (f n).1 * (f n).2 + c * (f n).2^2 = k :=
by
  sorry

end infinite_integer_solutions_l1622_162226


namespace percentage_of_women_in_study_group_l1622_162229

theorem percentage_of_women_in_study_group
  (W : ℝ)
  (H1 : 0 ≤ W ∧ W ≤ 1)
  (H2 : 0.60 * W = 0.54) :
  W = 0.9 :=
sorry

end percentage_of_women_in_study_group_l1622_162229


namespace altitude_length_l1622_162214

theorem altitude_length 
    {A B C : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
    (AB BC AC : ℝ) (hAC : 𝕜) 
    (h₀ : AB = 8)
    (h₁ : BC = 7)
    (h₂ : AC = 5) :
  h = (5 * Real.sqrt 3) / 2 :=
sorry

end altitude_length_l1622_162214


namespace desired_digit_set_l1622_162247

noncomputable def prob_digit (d : ℕ) : ℝ := if d > 0 then Real.log (d + 1) - Real.log d else 0

theorem desired_digit_set : 
  (prob_digit 5 = (1 / 2) * (prob_digit 5 + prob_digit 6 + prob_digit 7 + prob_digit 8)) ↔
  {d | d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8} = {5, 6, 7, 8} :=
by
  sorry

end desired_digit_set_l1622_162247


namespace difference_increased_decreased_l1622_162290

theorem difference_increased_decreased (x : ℝ) (hx : x = 80) : 
  ((x * 1.125) - (x * 0.75)) = 30 := by
  have h1 : x * 1.125 = 90 := by rw [hx]; norm_num
  have h2 : x * 0.75 = 60 := by rw [hx]; norm_num
  rw [h1, h2]
  norm_num
  done

end difference_increased_decreased_l1622_162290


namespace p_suff_not_necess_q_l1622_162201

def proposition_p (a : ℝ) : Prop := ∀ (x : ℝ), x > 0 → (3*a - 1)^x < 1
def proposition_q (a : ℝ) : Prop := a > (1 / 3)

theorem p_suff_not_necess_q : 
  (∀ (a : ℝ), proposition_p a → proposition_q a) ∧ (¬∀ (a : ℝ), proposition_q a → proposition_p a) :=
  sorry

end p_suff_not_necess_q_l1622_162201


namespace symmetry_construction_complete_l1622_162262

-- Conditions: The word and the chosen axis of symmetry
def word : String := "ГЕОМЕТРИя"

inductive Axis
| horizontal
| vertical

-- The main theorem which states that a symmetrical figure can be constructed for the given word and axis
theorem symmetry_construction_complete (axis : Axis) : ∃ (symmetrical : String), 
  (axis = Axis.horizontal ∨ axis = Axis.vertical) → 
   symmetrical = "яИРТЕМОЕГ" := 
by
  sorry

end symmetry_construction_complete_l1622_162262


namespace min_ab_value_l1622_162291

theorem min_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1 / a + 4 / b = Real.sqrt (a * b)) :
  a * b = 4 :=
  sorry

end min_ab_value_l1622_162291


namespace unique_flavors_l1622_162222

theorem unique_flavors (x y : ℕ) (h₀ : x = 5) (h₁ : y = 4) : 
  (∃ f : ℕ, f = 17) :=
sorry

end unique_flavors_l1622_162222


namespace part1_part2_part3_max_part3_min_l1622_162227

noncomputable def f : ℝ → ℝ := sorry

-- Given Conditions
axiom f_add (x y : ℝ) : f (x + y) = f x + f y
axiom f_neg (x : ℝ) : x > 0 → f x < 0
axiom f_one : f 1 = -2

-- Prove that f(0) = 0
theorem part1 : f 0 = 0 := sorry

-- Prove that f(x) is an odd function
theorem part2 : ∀ x : ℝ, f (-x) = -f x := sorry

-- Prove the maximum and minimum values of f(x) on [-3,3]
theorem part3_max : f (-3) = 6 := sorry
theorem part3_min : f 3 = -6 := sorry

end part1_part2_part3_max_part3_min_l1622_162227


namespace canal_cross_section_area_l1622_162257

theorem canal_cross_section_area
  (a b h : ℝ)
  (H1 : a = 12)
  (H2 : b = 8)
  (H3 : h = 84) :
  (1 / 2) * (a + b) * h = 840 :=
by
  rw [H1, H2, H3]
  sorry

end canal_cross_section_area_l1622_162257


namespace bucket_weight_l1622_162205

variable {p q x y : ℝ}

theorem bucket_weight (h1 : x + (1 / 4) * y = p) (h2 : x + (3 / 4) * y = q) :
  x + y = - (1 / 2) * p + (3 / 2) * q := by
  sorry

end bucket_weight_l1622_162205


namespace find_k_shelf_life_at_11_22_l1622_162272

noncomputable def food_shelf_life (k b x : ℝ) : ℝ := Real.exp (k * x + b)

-- Given conditions
def condition1 : food_shelf_life k b 0 = 192 := by sorry
def condition2 : food_shelf_life k b 33 = 24 := by sorry

-- Prove that k = - (Real.log 2) / 11
theorem find_k (k b : ℝ) (h1 : food_shelf_life k b 0 = 192) (h2 : food_shelf_life k b 33 = 24) : 
  k = - (Real.log 2) / 11 :=
by sorry

-- Use the found value of k to determine the shelf life at 11°C and 22°C
theorem shelf_life_at_11_22 (k b : ℝ) (h1 : food_shelf_life k b 0 = 192) (h2 : food_shelf_life k b 33 = 24) 
  (hk : k = - (Real.log 2) / 11) : 
  food_shelf_life k b 11 = 96 ∧ food_shelf_life k b 22 = 48 :=
by sorry

end find_k_shelf_life_at_11_22_l1622_162272


namespace calc_1_calc_2_l1622_162253

variable (x y : ℝ)

theorem calc_1 : (-x^2)^4 = x^8 := 
sorry

theorem calc_2 : (-x^2 * y)^3 = -x^6 * y^3 := 
sorry

end calc_1_calc_2_l1622_162253


namespace base_any_number_l1622_162268

open Nat

theorem base_any_number (n k : ℕ) (h1 : k ≥ 0) (h2 : (30 ^ k) ∣ 929260) (h3 : n ^ k - k ^ 3 = 1) : true :=
by
  sorry

end base_any_number_l1622_162268


namespace sufficient_not_necessary_condition_l1622_162207

theorem sufficient_not_necessary_condition (a b : ℝ) (h : (a - b) * a^2 > 0) : a > b ∧ a ≠ 0 :=
by {
  sorry
}

end sufficient_not_necessary_condition_l1622_162207


namespace closest_point_to_origin_l1622_162200

theorem closest_point_to_origin : 
  ∃ x y : ℝ, x > 0 ∧ y = x + 1/x ∧ (x, y) = (1/(2^(1/4)), (1 + 2^(1/2))/(2^(1/4))) :=
by
  sorry

end closest_point_to_origin_l1622_162200


namespace smallest_n_property_l1622_162286

theorem smallest_n_property (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
 (hxy : x ∣ y^3) (hyz : y ∣ z^3) (hzx : z ∣ x^3) : 
  x * y * z ∣ (x + y + z) ^ 13 := 
by sorry

end smallest_n_property_l1622_162286


namespace circle_centers_connection_line_eq_l1622_162232

-- Define the first circle equation
def circle1 (x y : ℝ) := (x^2 + y^2 - 4*x + 6*y = 0)

-- Define the second circle equation
def circle2 (x y : ℝ) := (x^2 + y^2 - 6*x = 0)

-- Given the centers of the circles, prove the equation of the line connecting them
theorem circle_centers_connection_line_eq (x y : ℝ) :
  (∀ (x y : ℝ), circle1 x y → (x = 2 ∧ y = -3)) →
  (∀ (x y : ℝ), circle2 x y → (x = 3 ∧ y = 0)) →
  (3 * x - y - 9 = 0) :=
by
  -- Here we would sketch the proof but skip it with sorry
  sorry

end circle_centers_connection_line_eq_l1622_162232


namespace pow_mod_remainder_l1622_162233

theorem pow_mod_remainder : (3 ^ 304) % 11 = 4 := by
  sorry

end pow_mod_remainder_l1622_162233


namespace project_completion_days_l1622_162295

-- A's work rate per day
def A_work_rate : ℚ := 1 / 20

-- B's work rate per day
def B_work_rate : ℚ := 1 / 30

-- Combined work rate per day
def combined_work_rate : ℚ := A_work_rate + B_work_rate

-- Work done by B alone in the last 5 days
def B_alone_work : ℚ := 5 * B_work_rate

-- Let variable x represent the number of days A and B work together
def x (x_days : ℚ) := x_days / combined_work_rate + B_alone_work = 1

theorem project_completion_days (x_days : ℚ) (total_days : ℚ) :
  A_work_rate = 1 / 20 → B_work_rate = 1 / 30 → combined_work_rate = 1 / 12 → x_days / 12 + 1 / 6 = 1 → x_days = 10 → total_days = x_days + 5 → total_days = 15 :=
by
  intros _ _ _ _ _ _
  sorry

end project_completion_days_l1622_162295


namespace distance_to_x_axis_l1622_162285

theorem distance_to_x_axis (x y : ℝ) :
  (x^2 / 9 - y^2 / 16 = 1) →
  (x^2 + y^2 = 25) →
  abs y = 16 / 5 :=
by
  -- Conditions: x^2 / 9 - y^2 / 16 = 1, x^2 + y^2 = 25
  -- Conclusion: abs y = 16 / 5 
  intro h1 h2
  sorry

end distance_to_x_axis_l1622_162285


namespace weight_of_new_person_l1622_162251

-- Definitions
variable (W : ℝ) -- total weight of original 15 people
variable (x : ℝ) -- weight of the new person
variable (n : ℕ) (avr_increase : ℝ) (original_person_weight : ℝ)
variable (total_increase : ℝ) -- total weight increase

-- Given constants
axiom n_value : n = 15
axiom avg_increase_value : avr_increase = 8
axiom original_person_weight_value : original_person_weight = 45
axiom total_increase_value : total_increase = n * avr_increase

-- Equation stating the condition
axiom weight_replace : W - original_person_weight + x = W + total_increase

-- Theorem (problem translated)
theorem weight_of_new_person : x = 165 := by
  sorry

end weight_of_new_person_l1622_162251


namespace intersection_a_eq_1_parallel_lines_value_of_a_l1622_162282

-- Define lines
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y - a + 2 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := 2 * a * x + (a + 3) * y + a - 5 = 0

-- Part 1: Prove intersection point for a = 1
theorem intersection_a_eq_1 :
  line1 1 (-4) 3 ∧ line2 1 (-4) 3 :=
by sorry

-- Part 2: Prove value of a for which lines are parallel
theorem parallel_lines_value_of_a :
  ∃ a : ℝ, ∀ x y : ℝ, line1 a x y ∧ line2 a x y →
  (2 * a^2 - a - 3 = 0 ∧ a ≠ -1 ∧ a = 3/2) :=
by sorry

end intersection_a_eq_1_parallel_lines_value_of_a_l1622_162282


namespace minimum_value_18_sqrt_3_minimum_value_at_x_3_l1622_162236

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 12*x + 81 / x^3

theorem minimum_value_18_sqrt_3 (x : ℝ) (hx : x > 0) :
  f x ≥ 18 * Real.sqrt 3 :=
by
  sorry

theorem minimum_value_at_x_3 : f 3 = 18 * Real.sqrt 3 :=
by
  sorry

end minimum_value_18_sqrt_3_minimum_value_at_x_3_l1622_162236


namespace Molly_age_now_l1622_162281

/- Definitions -/
def Sandy_curr_age : ℕ := 60
def Molly_curr_age (S : ℕ) : ℕ := 3 * S / 4
def Sandy_age_in_6_years (S : ℕ) : ℕ := S + 6

/- Theorem to prove -/
theorem Molly_age_now 
  (ratio_condition : ∀ S M : ℕ, S / M = 4 / 3 → M = 3 * S / 4)
  (age_condition : Sandy_age_in_6_years Sandy_curr_age = 66) : 
  Molly_curr_age Sandy_curr_age = 45 :=
by
  sorry

end Molly_age_now_l1622_162281


namespace correct_negation_l1622_162266

-- Define a triangle with angles A, B, and C
variables (α β γ : ℝ)

-- Define properties of the angles
def is_triangle (α β γ : ℝ) : Prop := α + β + γ = 180
def is_right_angle (angle : ℝ) : Prop := angle = 90
def is_acute_angle (angle : ℝ) : Prop := angle > 0 ∧ angle < 90

-- Original statement to be negated
def original_statement (α β γ : ℝ) : Prop := 
  is_triangle α β γ ∧ is_right_angle γ → is_acute_angle α ∧ is_acute_angle β

-- Negation of the original statement
def negated_statement (α β γ : ℝ) : Prop := 
  is_triangle α β γ ∧ ¬ is_right_angle γ → ¬ (is_acute_angle α ∧ is_acute_angle β)

-- Proof statement: prove that the negated statement is the correct negation
theorem correct_negation (α β γ : ℝ) :
  negated_statement α β γ = ¬ original_statement α β γ :=
sorry

end correct_negation_l1622_162266


namespace problem1_problem2_problem3_l1622_162267

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_periodic : ∀ x, f (x - 4) = -f x)
variable (h_increasing : ∀ x y : ℝ, 0 ≤ x → x ≤ 2 → x ≤ y → y ≤ 2 → f x ≤ f y)

-- Problem statements
theorem problem1 : f 2012 = 0 := sorry

theorem problem2 : ∀ x, f (4 - x) = -f (4 + x) := sorry

theorem problem3 : f (-25) < f 80 ∧ f 80 < f 11 := sorry

end problem1_problem2_problem3_l1622_162267


namespace part1_part2_i_part2_ii_l1622_162208

theorem part1 :
  ¬ ∃ x : ℝ, - (4 / x) = x := 
sorry

theorem part2_i (a c : ℝ) (ha : a ≠ 0) :
  (∃! x : ℝ, x = a * (x^2) + 6 * x + c ∧ x = 5 / 2) ↔ (a = -1 ∧ c = -25 / 4) :=
sorry

theorem part2_ii (m : ℝ) :
  (∃ (a c : ℝ), a = -1 ∧ c = - 25 / 4 ∧
    ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → - (x^2) + 6 * x - 25 / 4 + 1/4 ≥ -1 ∧ - (x^2) + 6 * x - 25 / 4 + 1/4 ≤ 3) ↔
    (3 ≤ m ∧ m ≤ 5) :=
sorry

end part1_part2_i_part2_ii_l1622_162208


namespace find_b_l1622_162237

noncomputable def f (x : ℝ) : ℝ := -1 / x

theorem find_b (a b : ℝ) (h1 : f a = -1 / 3) (h2 : f (a * b) = 1 / 6) : b = -2 := 
by
  sorry

end find_b_l1622_162237


namespace speed_of_A_is_24_speed_of_A_is_18_l1622_162277

-- Definitions for part 1
def speed_of_B (x : ℝ) := x
def speed_of_A_1 (x : ℝ) := 1.2 * x
def distance_AB := 30 -- kilometers
def distance_B_rides_first := 2 -- kilometers
def time_A_catches_up := 0.5 -- hours

theorem speed_of_A_is_24 (x : ℝ) (h1 : 0.6 * x = 2 + 0.5 * x) : speed_of_A_1 x = 24 := by
  sorry

-- Definitions for part 2
def speed_of_A_2 (y : ℝ) := 1.2 * y
def time_B_rides_first := 1/3 -- hours
def time_difference := 1/3 -- hours

theorem speed_of_A_is_18 (y : ℝ) (h2 : (30 / y) - (30 / (1.2 * y)) = 1/3) : speed_of_A_2 y = 18 := by
  sorry

end speed_of_A_is_24_speed_of_A_is_18_l1622_162277


namespace cost_of_ice_cream_l1622_162241

theorem cost_of_ice_cream 
  (meal_cost : ℕ)
  (number_of_people : ℕ)
  (total_money : ℕ)
  (total_cost : ℕ := meal_cost * number_of_people) 
  (remaining_money : ℕ := total_money - total_cost) 
  (ice_cream_cost_per_scoop : ℕ := remaining_money / number_of_people) :
  meal_cost = 10 ∧ number_of_people = 3 ∧ total_money = 45 →
  ice_cream_cost_per_scoop = 5 :=
by
  intros
  sorry

end cost_of_ice_cream_l1622_162241


namespace min_value_expression_l1622_162298

theorem min_value_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
    (h3 : ∀ x y : ℝ, x + y + a = 0 → (x - b)^2 + (y - 1)^2 = 2) : 
    (∃ c : ℝ,  c = 4 ∧ ∀ a b : ℝ, (0 < a → 0 < b → x + y + a = 0 → (x - b)^2 + (y - 1)^2 = 2 →  (3 - 2 * b)^2 / (2 * a) ≥ c)) :=
by
  sorry

end min_value_expression_l1622_162298


namespace tan_alpha_value_l1622_162271

theorem tan_alpha_value (α : ℝ) (h1 : Real.sin α = 3 / 5) (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) : Real.tan α = -3 / 4 := 
sorry

end tan_alpha_value_l1622_162271


namespace max_weight_l1622_162213

-- Define the weights
def weight1 := 2
def weight2 := 5
def weight3 := 10

-- Theorem stating that the heaviest single item that can be weighed using any combination of these weights is 17 lb
theorem max_weight : ∃ x, (x = weight1 + weight2 + weight3) ∧ x = 17 :=
by
  sorry

end max_weight_l1622_162213


namespace gcd_power_minus_one_l1622_162230

theorem gcd_power_minus_one (a b : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) : gcd (2^a - 1) (2^b - 1) = 2^(gcd a b) - 1 :=
by
  sorry

end gcd_power_minus_one_l1622_162230


namespace Sandy_pumpkins_l1622_162231

-- Definitions from the conditions
def Mike_pumpkins : ℕ := 23
def Total_pumpkins : ℕ := 74

-- Theorem to prove the number of pumpkins Sandy grew
theorem Sandy_pumpkins : ∃ (n : ℕ), n + Mike_pumpkins = Total_pumpkins :=
by
  existsi 51
  sorry

end Sandy_pumpkins_l1622_162231


namespace find_number_l1622_162276

-- Define the number x that satisfies the given condition
theorem find_number (x : ℤ) (h : x + 12 - 27 = 24) : x = 39 :=
by {
  -- This is where the proof steps will go, but we'll use sorry to indicate it's incomplete
  sorry
}

end find_number_l1622_162276


namespace probability_both_hit_l1622_162235

-- Conditions
def prob_A_hits : ℝ := 0.9
def prob_B_hits : ℝ := 0.8

-- Question and proof problem
theorem probability_both_hit : prob_A_hits * prob_B_hits = 0.72 :=
by
  sorry

end probability_both_hit_l1622_162235


namespace room_width_l1622_162279

theorem room_width (length : ℝ) (cost : ℝ) (rate : ℝ) (h_length : length = 5.5)
                    (h_cost : cost = 16500) (h_rate : rate = 800) : 
                    (cost / rate / length = 3.75) :=
by 
  sorry

end room_width_l1622_162279


namespace problem_solution_l1622_162239

noncomputable def solve_system : List (ℝ × ℝ × ℝ) :=
[(0, 1, -2), (-3/2, 5/2, -1/2)]

theorem problem_solution (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h1 : x^2 + y^2 = -x + 3*y + z)
  (h2 : y^2 + z^2 = x + 3*y - z)
  (h3 : z^2 + x^2 = 2*x + 2*y - z) :
  (x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = -3/2 ∧ y = 5/2 ∧ z = -1/2) :=
sorry

end problem_solution_l1622_162239


namespace arccos_one_eq_zero_l1622_162211

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l1622_162211


namespace songs_in_each_album_l1622_162265

variable (X : ℕ)

theorem songs_in_each_album (h : 6 * X + 2 * X = 72) : X = 9 :=
by sorry

end songs_in_each_album_l1622_162265


namespace quadratic_roots_ratio_l1622_162275

noncomputable def value_of_m (m : ℚ) : Prop :=
  ∃ r s : ℚ, r ≠ 0 ∧ s ≠ 0 ∧ (r / s = 3) ∧ (r + s = -9) ∧ (r * s = m)

theorem quadratic_roots_ratio (m : ℚ) (h : value_of_m m) : m = 243 / 16 :=
by
  sorry

end quadratic_roots_ratio_l1622_162275


namespace triangle_problem_l1622_162219

noncomputable def find_b (a b c : ℝ) : Prop :=
  let B : ℝ := 60 * Real.pi / 180 -- converting 60 degrees to radians
  b = 2 * Real.sqrt 2

theorem triangle_problem
  (a b c : ℝ)
  (h_area : (1 / 2) * a * c * Real.sin (60 * Real.pi / 180) = Real.sqrt 3)
  (h_cosine : a^2 + c^2 = 3 * a * c) : find_b a b c :=
by
  -- The proof would go here, but we're skipping it as per the instructions.
  sorry

end triangle_problem_l1622_162219


namespace num_nat_numbers_l1622_162216

theorem num_nat_numbers (n : ℕ) (h1 : n ≥ 1) (h2 : n ≤ 1992)
  (h3 : ∃ k3, n = 3 * k3)
  (h4 : ¬ (∃ k2, n = 2 * k2))
  (h5 : ¬ (∃ k5, n = 5 * k5)) : ∃ (m : ℕ), m = 266 :=
by
  sorry

end num_nat_numbers_l1622_162216


namespace shaded_area_in_rectangle_is_correct_l1622_162228

noncomputable def percentage_shaded_area : ℝ :=
  let side_length_congruent_squares := 10
  let side_length_small_square := 5
  let rect_length := 20
  let rect_width := 15
  let rect_area := rect_length * rect_width
  let overlap_congruent_squares := side_length_congruent_squares * rect_width
  let overlap_small_square := (side_length_small_square / 2) * side_length_small_square
  let total_shaded_area := overlap_congruent_squares + overlap_small_square
  (total_shaded_area / rect_area) * 100

theorem shaded_area_in_rectangle_is_correct :
  percentage_shaded_area = 54.17 :=
sorry

end shaded_area_in_rectangle_is_correct_l1622_162228


namespace investment_scientific_notation_l1622_162223

def is_scientific_notation (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ (1650000000 = a * 10^n)

theorem investment_scientific_notation :
  ∃ a n, is_scientific_notation a n ∧ a = 1.65 ∧ n = 9 :=
sorry

end investment_scientific_notation_l1622_162223


namespace mrs_franklin_needs_more_valentines_l1622_162234

theorem mrs_franklin_needs_more_valentines (valentines_have : ℝ) (students : ℝ) : valentines_have = 58 ∧ students = 74 → students - valentines_have = 16 :=
by
  sorry

end mrs_franklin_needs_more_valentines_l1622_162234


namespace n_plus_5_divisible_by_6_l1622_162260

theorem n_plus_5_divisible_by_6 (n : ℕ) (h1 : (n + 2) % 3 = 0) (h2 : (n + 3) % 4 = 0) : (n + 5) % 6 = 0 := 
sorry

end n_plus_5_divisible_by_6_l1622_162260


namespace sum_of_three_numbers_l1622_162287

theorem sum_of_three_numbers : ∃ (a b c : ℝ), a ≤ b ∧ b ≤ c ∧ b = 8 ∧ 
  (a + b + c) / 3 = a + 8 ∧ (a + b + c) / 3 = c - 20 ∧ a + b + c = 60 :=
sorry

end sum_of_three_numbers_l1622_162287


namespace michael_pets_kangaroos_l1622_162269

theorem michael_pets_kangaroos :
  let total_pets := 24
  let fraction_dogs := 1 / 8
  let fraction_not_cows := 3 / 4
  let fraction_not_cats := 2 / 3
  let num_dogs := fraction_dogs * total_pets
  let num_cows := (1 - fraction_not_cows) * total_pets
  let num_cats := (1 - fraction_not_cats) * total_pets
  let num_kangaroos := total_pets - num_dogs - num_cows - num_cats
  num_kangaroos = 7 :=
by
  sorry

end michael_pets_kangaroos_l1622_162269


namespace list_price_proof_l1622_162221

-- Define the list price of the item
noncomputable def list_price : ℝ := 33

-- Define the selling price and commission for Alice
def alice_selling_price (x : ℝ) : ℝ := x - 15
def alice_commission (x : ℝ) : ℝ := 0.15 * alice_selling_price x

-- Define the selling price and commission for Charles
def charles_selling_price (x : ℝ) : ℝ := x - 18
def charles_commission (x : ℝ) : ℝ := 0.18 * charles_selling_price x

-- The main theorem: proving the list price given Alice and Charles receive the same commission
theorem list_price_proof (x : ℝ) (h : alice_commission x = charles_commission x) : x = list_price :=
by 
  sorry

end list_price_proof_l1622_162221


namespace is_quadratic_equation_l1622_162283

open Real

-- Define the candidate equations as statements in Lean 4
def equation_A (x : ℝ) : Prop := 3 * x^2 = 1 - 1 / (3 * x)
def equation_B (x m : ℝ) : Prop := (m - 2) * x^2 - m * x + 3 = 0
def equation_C (x : ℝ) : Prop := (x^2 - 3) * (x - 1) = 0
def equation_D (x : ℝ) : Prop := x^2 = 2

-- Prove that among the given equations, equation_D is the only quadratic equation
theorem is_quadratic_equation (x : ℝ) :
  (∃ a b c : ℝ, a ≠ 0 ∧ equation_A x = (a * x^2 + b * x + c = 0)) ∨
  (∃ m a b c : ℝ, a ≠ 0 ∧ equation_B x m = (a * x^2 + b * x + c = 0)) ∨
  (∃ a b c : ℝ, a ≠ 0 ∧ equation_C x = (a * x^2 + b * x + c = 0)) ∨
  (∃ a b c : ℝ, a ≠ 0 ∧ equation_D x = (a * x^2 + b * x + c = 0)) := by
  sorry

end is_quadratic_equation_l1622_162283


namespace miles_per_book_l1622_162206

theorem miles_per_book (total_miles : ℝ) (books_read : ℝ) (miles_per_book : ℝ) : 
  total_miles = 6760 ∧ books_read = 15 → miles_per_book = 450.67 := 
by
  sorry

end miles_per_book_l1622_162206


namespace solve_equation_l1622_162256

theorem solve_equation {x : ℝ} (h : x ≠ -2) : (6 * x) / (x + 2) - 4 / (x + 2) = 2 / (x + 2) → x = 1 :=
by
  intro h_eq
  -- proof steps would go here
  sorry

end solve_equation_l1622_162256


namespace solve_abs_inequality_l1622_162209

theorem solve_abs_inequality (x : ℝ) : abs ((7 - 2 * x) / 4) < 3 ↔ -2.5 < x ∧ x < 9.5 := by
  sorry

end solve_abs_inequality_l1622_162209


namespace probability_sum_8_9_10_l1622_162288

/-- The faces of the first die -/
def first_die := [2, 2, 3, 3, 5, 5]

/-- The faces of the second die -/
def second_die := [1, 3, 4, 5, 6, 7]

/-- Predicate that checks if the sum of two numbers is either 8, 9, or 10 -/
def valid_sum (a b : ℕ) : Prop := a + b = 8 ∨ a + b = 9 ∨ a + b = 10

/-- Calculate the probability of a sum being 8, 9, or 10 according to the given dice setup -/
def calc_probability : ℚ := 
  let valid_pairs := [(2, 6), (2, 7), (3, 5), (3, 6), (3, 7), (5, 3), (5, 4), (5, 5)] 
  (valid_pairs.length : ℚ) / (first_die.length * second_die.length : ℚ)

theorem probability_sum_8_9_10 : calc_probability = 4 / 9 :=
by
  sorry

end probability_sum_8_9_10_l1622_162288


namespace factorize_expression_l1622_162240

-- Lean 4 statement for the proof problem
theorem factorize_expression (a b : ℝ) : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end factorize_expression_l1622_162240


namespace base4_more_digits_than_base9_l1622_162293

def base4_digits_1234 : ℕ := 6
def base9_digits_1234 : ℕ := 4

theorem base4_more_digits_than_base9 :
  base4_digits_1234 - base9_digits_1234 = 2 :=
by
  sorry

end base4_more_digits_than_base9_l1622_162293


namespace ratio_calc_l1622_162296

theorem ratio_calc :
  (14^4 + 484) * (26^4 + 484) * (38^4 + 484) * (50^4 + 484) * (62^4 + 484) /
  ((8^4 + 484) * (20^4 + 484) * (32^4 + 484) * (44^4 + 484) * (56^4 + 484)) = -423 := 
by
  sorry

end ratio_calc_l1622_162296


namespace cookies_flour_and_eggs_l1622_162225

theorem cookies_flour_and_eggs (c₁ c₂ : ℕ) (f₁ f₂ : ℕ) (e₁ e₂ : ℕ) 
  (h₁ : c₁ = 40) (h₂ : f₁ = 3) (h₃ : e₁ = 2) (h₄ : c₂ = 120) :
  f₂ = f₁ * (c₂ / c₁) ∧ e₂ = e₁ * (c₂ / c₁) :=
by
  sorry

end cookies_flour_and_eggs_l1622_162225


namespace trent_bus_blocks_to_library_l1622_162250

-- Define the given conditions
def total_distance := 22
def walking_distance := 4

-- Define the function to determine bus block distance
def bus_ride_distance (total: ℕ) (walk: ℕ) : ℕ :=
  (total - (walk * 2)) / 2

-- The theorem we need to prove
theorem trent_bus_blocks_to_library : 
  bus_ride_distance total_distance walking_distance = 7 := by
  sorry

end trent_bus_blocks_to_library_l1622_162250


namespace molecular_weight_C8H10N4O6_eq_258_22_l1622_162280

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

def number_C : ℕ := 8
def number_H : ℕ := 10
def number_N : ℕ := 4
def number_O : ℕ := 6

def molecular_weight : ℝ :=
    (number_C * atomic_weight_C) +
    (number_H * atomic_weight_H) +
    (number_N * atomic_weight_N) +
    (number_O * atomic_weight_O)

theorem molecular_weight_C8H10N4O6_eq_258_22 :
  molecular_weight = 258.22 :=
  by
    sorry

end molecular_weight_C8H10N4O6_eq_258_22_l1622_162280


namespace problem_solution_l1622_162292

open Nat

def sum_odd (n : ℕ) : ℕ :=
  n ^ 2

def sum_even (n : ℕ) : ℕ :=
  n * (n + 1)

theorem problem_solution : 
  sum_odd 1010 - sum_even 1009 = 1010 :=
by
  -- Here the proof would go
  sorry

end problem_solution_l1622_162292


namespace sacks_after_6_days_l1622_162297

theorem sacks_after_6_days (sacks_per_day : ℕ) (days : ℕ) 
  (h1 : sacks_per_day = 83) (h2 : days = 6) : 
  sacks_per_day * days = 498 :=
by
  sorry

end sacks_after_6_days_l1622_162297


namespace subscription_total_amount_l1622_162242

theorem subscription_total_amount 
  (A B C : ℝ)
  (profit_C profit_total : ℝ)
  (subscription_A subscription_B subscription_C : ℝ)
  (subscription_total : ℝ)
  (hA : subscription_A = subscription_B + 4000)
  (hB : subscription_B = subscription_C + 5000)
  (hc_share : profit_C = 8400)
  (total_profit : profit_total = 35000)
  (h_ratio : profit_C / profit_total = subscription_C / subscription_total)
  (h_subs : subscription_total = subscription_A + subscription_B + subscription_C)
  : subscription_total = 50000 := 
sorry

end subscription_total_amount_l1622_162242


namespace union_A_B_inter_A_B_diff_U_A_U_B_subset_A_C_l1622_162258

universe u

open Set

def U := @univ ℝ
def A := { x : ℝ | 3 ≤ x ∧ x < 10 }
def B := { x : ℝ | 2 < x ∧ x ≤ 7 }
def C (a : ℝ) := { x : ℝ | x > a }

theorem union_A_B : A ∪ B = { x : ℝ | 2 < x ∧ x < 10 } :=
by sorry

theorem inter_A_B : A ∩ B = { x : ℝ | 3 ≤ x ∧ x ≤ 7 } :=
by sorry

theorem diff_U_A_U_B : (U \ A) ∩ (U \ B) = { x : ℝ | x ≤ 2 } ∪ { x : ℝ | 10 ≤ x } :=
by sorry

theorem subset_A_C (a : ℝ) (h : A ⊆ C a) : a < 3 :=
by sorry

end union_A_B_inter_A_B_diff_U_A_U_B_subset_A_C_l1622_162258


namespace basis_v_l1622_162218

variable {V : Type*} [AddCommGroup V] [Module ℝ V]  -- specifying V as a real vector space
variables (a b c : V)

-- Assume a, b, and c are linearly independent, forming a basis
axiom linear_independent_a_b_c : LinearIndependent ℝ ![a, b, c]

-- The main theorem which we need to prove
theorem basis_v (h : LinearIndependent ℝ ![a, b, c]) :
  LinearIndependent ℝ ![c, a + b, a - b] :=
sorry

end basis_v_l1622_162218


namespace percentage_profit_without_discount_l1622_162238

variable (CP : ℝ) (discountRate profitRate noDiscountProfitRate : ℝ)

theorem percentage_profit_without_discount 
  (hCP : CP = 100)
  (hDiscount : discountRate = 0.04)
  (hProfit : profitRate = 0.26)
  (hNoDiscountProfit : noDiscountProfitRate = 0.3125) :
  let SP := CP * (1 + profitRate)
  let MP := SP / (1 - discountRate)
  noDiscountProfitRate = (MP - CP) / CP :=
by
  sorry

end percentage_profit_without_discount_l1622_162238


namespace height_pillar_D_correct_l1622_162215

def height_of_pillar_at_D (h_A h_B h_C : ℕ) (side_length : ℕ) : ℕ :=
17

theorem height_pillar_D_correct :
  height_of_pillar_at_D 15 10 12 10 = 17 := 
by sorry

end height_pillar_D_correct_l1622_162215


namespace compare_fx_l1622_162245

noncomputable def f (a x : ℝ) := a * x ^ 2 + 2 * a * x + 4

theorem compare_fx (a x1 x2 : ℝ) (h₁ : -3 < a) (h₂ : a < 0) (h₃ : x1 < x2) (h₄ : x1 + x2 ≠ 1 + a) :
  f a x1 > f a x2 :=
sorry

end compare_fx_l1622_162245
