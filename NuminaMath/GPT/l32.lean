import Mathlib

namespace number_of_participants_l32_3233

theorem number_of_participants (total_gloves : ℕ) (gloves_per_participant : ℕ)
  (h : total_gloves = 126) (h' : gloves_per_participant = 2) : 
  (total_gloves / gloves_per_participant = 63) :=
by
  sorry

end number_of_participants_l32_3233


namespace allison_uploads_480_hours_in_june_l32_3264

noncomputable def allison_upload_total_hours : Nat :=
  let before_june_16 := 10 * 15
  let from_june_16_to_23 := 15 * 8
  let from_june_24_to_end := 30 * 7
  before_june_16 + from_june_16_to_23 + from_june_24_to_end

theorem allison_uploads_480_hours_in_june :
  allison_upload_total_hours = 480 := by
  sorry

end allison_uploads_480_hours_in_june_l32_3264


namespace calculate_x_one_minus_f_l32_3207

noncomputable def x := (2 + Real.sqrt 3) ^ 500
noncomputable def n := Int.floor x
noncomputable def f := x - n

theorem calculate_x_one_minus_f : x * (1 - f) = 1 := by
  sorry

end calculate_x_one_minus_f_l32_3207


namespace trig_expression_identity_l32_3242

theorem trig_expression_identity (a : ℝ) (h : 2 * Real.sin a = 3 * Real.cos a) : 
  (4 * Real.sin a + Real.cos a) / (5 * Real.sin a - 2 * Real.cos a) = 14 / 11 :=
by
  sorry

end trig_expression_identity_l32_3242


namespace seating_chart_example_l32_3271

def seating_chart_representation (a b : ℕ) : String :=
  s!"{a} columns {b} rows"

theorem seating_chart_example :
  seating_chart_representation 4 3 = "4 columns 3 rows" :=
by
  sorry

end seating_chart_example_l32_3271


namespace factor_problem_l32_3215

theorem factor_problem (x y m : ℝ) (h : (1 - 2 * x + y) ∣ (4 * x * y - 4 * x^2 - y^2 - m)) :
  m = -1 :=
by
  sorry

end factor_problem_l32_3215


namespace problem_statement_l32_3219

def star (x y : Nat) : Nat :=
  match x, y with
  | 1, 1 => 4 | 1, 2 => 3 | 1, 3 => 2 | 1, 4 => 1
  | 2, 1 => 1 | 2, 2 => 4 | 2, 3 => 3 | 2, 4 => 2
  | 3, 1 => 2 | 3, 2 => 1 | 3, 3 => 4 | 3, 4 => 3
  | 4, 1 => 3 | 4, 2 => 2 | 4, 3 => 1 | 4, 4 => 4
  | _, _ => 0  -- This line handles unexpected inputs.

theorem problem_statement : star (star 3 2) (star 2 1) = 4 := by
  sorry

end problem_statement_l32_3219


namespace fraction_of_widgets_second_shift_l32_3232

theorem fraction_of_widgets_second_shift (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let first_shift_widgets := x * y
  let second_shift_widgets := (2 / 3) * x * (4 / 3) * y
  let total_widgets := first_shift_widgets + second_shift_widgets
  let fraction_second_shift := second_shift_widgets / total_widgets
  fraction_second_shift = 8 / 17 :=
by
  sorry

end fraction_of_widgets_second_shift_l32_3232


namespace original_cost_of_statue_l32_3214

theorem original_cost_of_statue (sale_price : ℝ) (profit_percent : ℝ) (original_cost : ℝ) 
  (h1 : sale_price = 620) 
  (h2 : profit_percent = 0.25) 
  (h3 : sale_price = (1 + profit_percent) * original_cost) : 
  original_cost = 496 :=
by
  sorry

end original_cost_of_statue_l32_3214


namespace range_of_x_for_acute_angle_l32_3263

theorem range_of_x_for_acute_angle (x : ℝ) (h₁ : (x, 2*x) ≠ (0, 0)) (h₂ : (x+1, x+3) ≠ (0, 0)) (h₃ : (3*x^2 + 7*x > 0)) : 
  x < -7/3 ∨ (0 < x ∧ x < 1) ∨ x > 1 :=
by {
  -- This theorem asserts the given range of x given the dot product solution.
  sorry
}

end range_of_x_for_acute_angle_l32_3263


namespace problem_correct_answer_l32_3286

theorem problem_correct_answer (x y : ℕ) (h1 : y > 3) (h2 : x^2 + y^4 = 2 * ((x - 6)^2 + (y + 1)^2)) : x^2 + y^4 = 1994 :=
  sorry

end problem_correct_answer_l32_3286


namespace combined_bus_rides_length_l32_3254

theorem combined_bus_rides_length :
  let v := 0.62
  let z := 0.5
  let a := 0.72
  v + z + a = 1.84 :=
by
  let v := 0.62
  let z := 0.5
  let a := 0.72
  show v + z + a = 1.84
  sorry

end combined_bus_rides_length_l32_3254


namespace sum_of_x_and_y_l32_3257

theorem sum_of_x_and_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
(hx15 : x < 15) (hy15 : y < 15) (h : x + y + x * y = 119) : x + y = 21 ∨ x + y = 20 := 
by
  sorry

end sum_of_x_and_y_l32_3257


namespace automobile_distance_2_minutes_l32_3299

theorem automobile_distance_2_minutes (a : ℝ) :
  let acceleration := a / 12
  let time_minutes := 2
  let time_seconds := time_minutes * 60
  let distance_feet := (1 / 2) * acceleration * time_seconds^2
  let distance_yards := distance_feet / 3
  distance_yards = 200 * a := 
by sorry

end automobile_distance_2_minutes_l32_3299


namespace george_collected_50_marbles_l32_3228

theorem george_collected_50_marbles (w y g r total : ℕ)
  (hw : w = total / 2)
  (hy : y = 12)
  (hg : g = y / 2)
  (hr : r = 7)
  (htotal : total = w + y + g + r) :
  total = 50 := by
  sorry

end george_collected_50_marbles_l32_3228


namespace find_range_of_a_l32_3203

def have_real_roots (a : ℝ) : Prop := a^2 - 16 ≥ 0

def is_increasing_on_interval (a : ℝ) : Prop := a ≥ -12

theorem find_range_of_a (a : ℝ) : ((have_real_roots a ∨ is_increasing_on_interval a) ∧ ¬(have_real_roots a ∧ is_increasing_on_interval a)) → (a < -12 ∨ (-4 < a ∧ a < 4)) :=
by 
  sorry

end find_range_of_a_l32_3203


namespace solve_real_equation_l32_3244

theorem solve_real_equation (x : ℝ) (h : x^4 + (3 - x)^4 = 82) : x = 2.5 ∨ x = 0.5 :=
sorry

end solve_real_equation_l32_3244


namespace non_degenerate_ellipse_condition_l32_3256

theorem non_degenerate_ellipse_condition (x y k a : ℝ) :
  (3 * x^2 + 9 * y^2 - 12 * x + 27 * y = k) ∧
  (∃ h : ℝ, 3 * (x - h)^2 + 9 * (y + 3/2)^2 = k + 129/4) ∧
  (k > a) ↔ (a = -129 / 4) :=
by
  sorry

end non_degenerate_ellipse_condition_l32_3256


namespace find_youngest_age_l32_3209

noncomputable def youngest_child_age 
  (meal_cost_mother : ℝ) 
  (meal_cost_per_year : ℝ) 
  (total_bill : ℝ) 
  (triplets_count : ℕ) := 
  {y : ℝ // 
    (∃ t : ℝ, 
      meal_cost_mother + meal_cost_per_year * (triplets_count * t + y) = total_bill ∧ y = 2 ∨ y = 5)}

theorem find_youngest_age : 
  youngest_child_age 3.75 0.50 12.25 3 := 
sorry

end find_youngest_age_l32_3209


namespace percentage_problem_l32_3279

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 400) : 1.20 * x = 2400 :=
by
  sorry

end percentage_problem_l32_3279


namespace minimum_value_f_minimum_value_abc_l32_3258

noncomputable def f (x : ℝ) : ℝ := abs (x - 4) + abs (x - 3)

theorem minimum_value_f : ∃ m : ℝ, m = 1 ∧ ∀ x : ℝ, f x ≥ m := 
by
  let m := 1
  existsi m
  sorry

theorem minimum_value_abc (a b c : ℝ) (h : a + 2 * b + 3 * c = 1) : ∃ n : ℝ, n = 1/14 ∧ a^2 + b^2 + c^2 ≥ n :=
by
  let n := 1 / 14
  existsi n
  sorry

end minimum_value_f_minimum_value_abc_l32_3258


namespace ratio_of_a_to_b_l32_3210

-- Given conditions
variables {a b x : ℝ}
-- a and b are positive real numbers distinct from 1
variables (h1 : a > 0) (h2 : a ≠ 1) (h3 : b > 0) (h4 : b ≠ 1)
-- Given equation involving logarithms
variables (h5 : 5 * (Real.log x / Real.log a) ^ 2 + 7 * (Real.log x / Real.log b) ^ 2 = 10 * (Real.log x) ^ 2)

-- Prove that the ratio of a to b is a^(sqrt(7/5))
theorem ratio_of_a_to_b (h1 : a > 0) (h2 : a ≠ 1) (h3 : b > 0) (h4 : b ≠ 1) (h5 : 5 * (Real.log x / Real.log a) ^ 2 + 7 * (Real.log x / Real.log b) ^ 2 = 10 * (Real.log x) ^ 2) :
  b = a ^ Real.sqrt (7 / 5) :=
sorry

end ratio_of_a_to_b_l32_3210


namespace swimmer_upstream_distance_l32_3217

theorem swimmer_upstream_distance (v : ℝ) (c : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) 
                                   (downstream_speed : ℝ) (upstream_time : ℝ) : 
  c = 4.5 →
  downstream_distance = 55 →
  downstream_time = 5 →
  downstream_speed = downstream_distance / downstream_time →
  v + c = downstream_speed →
  upstream_time = 5 →
  (v - c) * upstream_time = 10 := 
by
  intro h_c
  intro h_downstream_distance
  intro h_downstream_time
  intro h_downstream_speed
  intro h_effective_downstream
  intro h_upstream_time
  sorry

end swimmer_upstream_distance_l32_3217


namespace expand_polynomial_l32_3252

variable {x y z : ℝ}

theorem expand_polynomial : (x + 10 * z + 5) * (2 * y + 15) = 2 * x * y + 20 * y * z + 15 * x + 10 * y + 150 * z + 75 :=
  sorry

end expand_polynomial_l32_3252


namespace quadratic_value_l32_3285

theorem quadratic_value (a b c : ℤ) (a_pos : a > 0) (h_eq : ∀ x : ℝ, (a * x + b)^2 = 49 * x^2 + 70 * x + c) : a + b + c = -134 :=
by
  -- Proof starts here
  sorry

end quadratic_value_l32_3285


namespace girls_count_l32_3289

-- Define the constants according to the conditions
def boys_on_team : ℕ := 28
def groups : ℕ := 8
def members_per_group : ℕ := 4

-- Calculate the total number of members
def total_members : ℕ := groups * members_per_group

-- Calculate the number of girls by subtracting the number of boys from the total members
def girls_on_team : ℕ := total_members - boys_on_team

-- The proof statement: prove that the number of girls on the team is 4
theorem girls_count : girls_on_team = 4 := by
  -- Skip the proof, completing the statement
  sorry

end girls_count_l32_3289


namespace find_S_l32_3284

variable {R S T c : ℝ}

theorem find_S
  (h1 : R = 2)
  (h2 : T = 1/2)
  (h3 : S = 4)
  (h4 : R = c * S / T)
  (h5 : R = 8)
  (h6 : T = 1/3) :
  S = 32 / 3 :=
by
  sorry

end find_S_l32_3284


namespace triangle_inequality_l32_3229

variables {a b c : ℝ}

def sides_of_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality (h : sides_of_triangle a b c) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧ 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
sorry

end triangle_inequality_l32_3229


namespace tangency_point_l32_3262

def parabola1 (x : ℝ) : ℝ := x^2 + 10 * x + 18
def parabola2 (y : ℝ) : ℝ := y^2 + 60 * y + 910

theorem tangency_point (x y : ℝ) (h1 : y = parabola1 x) (h2 : x = parabola2 y) :
  x = -9 / 2 ∧ y = -59 / 2 :=
by
  sorry

end tangency_point_l32_3262


namespace complement_intersection_eq_l32_3204

-- Definitions of sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 5, 8}
def B : Set ℕ := {1, 3, 5, 7}

-- Definition of complement of A in U
def complement_U_A : Set ℕ := U \ A

-- The main statement to prove
theorem complement_intersection_eq :
  (complement_U_A ∩ B) = {1, 3, 7} :=
by sorry

end complement_intersection_eq_l32_3204


namespace initial_dogwood_trees_in_park_l32_3249

def num_added_trees := 5 + 4
def final_num_trees := 16
def initial_num_trees (x : ℕ) := x

theorem initial_dogwood_trees_in_park (x : ℕ) 
  (h1 : num_added_trees = 9) 
  (h2 : final_num_trees = 16) : 
  initial_num_trees x + num_added_trees = final_num_trees → 
  x = 7 := 
by 
  intro h3
  rw [initial_num_trees, num_added_trees] at h3
  linarith

end initial_dogwood_trees_in_park_l32_3249


namespace s_l32_3274

def cost_per_chocolate_bar : ℝ := 1.50
def sections_per_chocolate_bar : ℕ := 3
def scouts : ℕ := 15
def total_money_spent : ℝ := 15.0

theorem s'mores_per_scout :
  (total_money_spent / cost_per_chocolate_bar * sections_per_chocolate_bar) / scouts = 2 :=
by
  sorry

end s_l32_3274


namespace zeros_at_end_of_quotient_factorial_l32_3278

def count_factors_of_five (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625

theorem zeros_at_end_of_quotient_factorial :
  count_factors_of_five 2018 - count_factors_of_five 30 - count_factors_of_five 11 = 493 :=
by
  sorry

end zeros_at_end_of_quotient_factorial_l32_3278


namespace rhinos_horn_segment_area_l32_3213

theorem rhinos_horn_segment_area :
  let full_circle_area (r : ℝ) := π * r^2
  let quarter_circle_area (r : ℝ) := (1 / 4) * full_circle_area r
  let half_circle_area (r : ℝ) := (1 / 2) * full_circle_area r
  let larger_quarter_circle_area := quarter_circle_area 4
  let smaller_half_circle_area := half_circle_area 2
  let rhinos_horn_segment_area := larger_quarter_circle_area - smaller_half_circle_area
  rhinos_horn_segment_area = 2 * π := 
by sorry 

end rhinos_horn_segment_area_l32_3213


namespace carpet_rate_l32_3240

theorem carpet_rate (length breadth cost area: ℝ) (h₁ : length = 13) (h₂ : breadth = 9) (h₃ : cost = 1872) (h₄ : area = length * breadth) :
  cost / area = 16 := by
  sorry

end carpet_rate_l32_3240


namespace trigonometric_identity_l32_3221

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -2) : 
  2 * Real.sin α * Real.cos α - (Real.cos α)^2 = -1 := 
by
  sorry

end trigonometric_identity_l32_3221


namespace no_nat_solutions_l32_3282
-- Import the Mathlib library

-- Lean statement for the proof problem
theorem no_nat_solutions (x : ℕ) : ¬ (19 * x^2 + 97 * x = 1997) :=
by {
  -- Solution omitted
  sorry
}

end no_nat_solutions_l32_3282


namespace first_day_exceeds_200_l32_3246

def bacteria_count (n : ℕ) : ℕ := 4 * 3^n

def exceeds_200 (n : ℕ) : Prop := bacteria_count n > 200

theorem first_day_exceeds_200 : ∃ n, exceeds_200 n ∧ ∀ m < n, ¬ exceeds_200 m :=
by sorry

end first_day_exceeds_200_l32_3246


namespace train_length_l32_3283

noncomputable def length_of_train (t : ℝ) (v_train_kmh : ℝ) (v_man_kmh : ℝ) : ℝ :=
  let v_relative_kmh := v_train_kmh - v_man_kmh
  let v_relative_ms := v_relative_kmh * 1000 / 3600
  v_relative_ms * t

theorem train_length : length_of_train 30.99752019838413 80 8 = 619.9504039676826 := 
  by simp [length_of_train]; sorry

end train_length_l32_3283


namespace sequence_inequality_l32_3269

theorem sequence_inequality
  (a : ℕ → ℝ)
  (h_cond : ∀ k m : ℕ, |a (k + m) - a k - a m| ≤ 1) :
  ∀ p q : ℕ, |a p / p - a q / q| < 1 / p + 1 / q :=
by
  intros p q
  sorry

end sequence_inequality_l32_3269


namespace intersection_complement_eq_singleton_l32_3277

open Set

def U : Set ℤ := {-1, 0, 1, 2, 3, 4}
def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {-1, 0, 2}
def CU_A : Set ℤ := U \ A

theorem intersection_complement_eq_singleton : B ∩ CU_A = {0} := 
by
  sorry

end intersection_complement_eq_singleton_l32_3277


namespace total_votes_cast_l32_3296

variable (total_votes : ℕ)
variable (emily_votes : ℕ)
variable (emily_share : ℚ := 4 / 15)
variable (dexter_share : ℚ := 1 / 3)

theorem total_votes_cast :
  emily_votes = 48 → 
  emily_share * total_votes = emily_votes → 
  total_votes = 180 := by
  intro h_emily_votes
  intro h_emily_share
  sorry

end total_votes_cast_l32_3296


namespace num_square_free_odds_l32_3255

noncomputable def is_square_free (m : ℕ) : Prop :=
  ∀ n : ℕ, n^2 ∣ m → n = 1

noncomputable def count_square_free_odds : ℕ :=
  (199 - 1) / 2 - (11 + 4 + 2 + 1 + 1 + 1)

theorem num_square_free_odds : count_square_free_odds = 79 := by
  sorry

end num_square_free_odds_l32_3255


namespace females_on_police_force_l32_3216

theorem females_on_police_force (H : ∀ (total_female_officers total_officers_on_duty female_officers_on_duty : ℕ), 
  total_officers_on_duty = 500 ∧ female_officers_on_duty = total_officers_on_duty / 2 ∧ female_officers_on_duty = total_female_officers / 4) :
  ∃ total_female_officers : ℕ, total_female_officers = 1000 := 
by {
  sorry
}

end females_on_police_force_l32_3216


namespace minimum_value_x_2y_l32_3227

theorem minimum_value_x_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = x * y) : x + 2 * y = 8 :=
sorry

end minimum_value_x_2y_l32_3227


namespace geometric_seq_problem_l32_3250

-- Definitions to capture the geometric sequence and the known condition
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

variables (a : ℕ → ℝ)

-- Given the condition a_1 * a_8^3 * a_15 = 243
axiom geom_seq_condition : a 1 * (a 8)^3 * a 15 = 243

theorem geometric_seq_problem 
  (h : is_geometric_sequence a) : (a 9)^3 / (a 11) = 9 :=
sorry

end geometric_seq_problem_l32_3250


namespace find_side_difference_l32_3218

def triangle_ABC : Type := ℝ
def angle_B := 20
def angle_C := 40
def length_AD := 2

theorem find_side_difference (ABC : triangle_ABC) (B : ℝ) (C : ℝ) (AD : ℝ) (BC AB : ℝ) :
  B = angle_B → C = angle_C → AD = length_AD → BC - AB = 2 :=
by 
  sorry

end find_side_difference_l32_3218


namespace pens_distribution_l32_3220

theorem pens_distribution (friends : ℕ) (pens : ℕ) (at_least_one : ℕ) 
  (h1 : friends = 4) (h2 : pens = 10) (h3 : at_least_one = 1) 
  (h4 : ∀ f : ℕ, f < friends → at_least_one ≤ f) :
  ∃ ways : ℕ, ways = 142 := 
sorry

end pens_distribution_l32_3220


namespace intersection_A_B_l32_3268

-- Define sets A and B based on given conditions
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 1 < x ∧ x < 4}

-- Prove the intersection of A and B equals (2,4)
theorem intersection_A_B : A ∩ B = {x | 2 < x ∧ x < 4} := 
by
  sorry

end intersection_A_B_l32_3268


namespace scientific_notation_equivalent_l32_3247

theorem scientific_notation_equivalent : ∃ a n, (3120000 : ℝ) = a * 10^n ∧ a = 3.12 ∧ n = 6 :=
by
  exists 3.12
  exists 6
  sorry

end scientific_notation_equivalent_l32_3247


namespace cost_of_pet_snake_l32_3206

theorem cost_of_pet_snake (original_amount : ℕ) (amount_left : ℕ) (cost : ℕ) 
  (h1 : original_amount = 73) (h2 : amount_left = 18) : cost = 55 :=
by
  sorry

end cost_of_pet_snake_l32_3206


namespace inequality_solution_l32_3212

theorem inequality_solution (a b c : ℝ) :
  (∀ x : ℝ, -4 < x ∧ x < 7 → a * x^2 + b * x + c > 0) →
  (∀ x : ℝ, (x < -1/7 ∨ x > 1/4) ↔ c * x^2 - b * x + a > 0) :=
by
  sorry

end inequality_solution_l32_3212


namespace find_f3_l32_3236

noncomputable def f : ℝ → ℝ := sorry

theorem find_f3 (h : ∀ x : ℝ, x ≠ 0 → f x - 2 * f (1 / x) = 3 ^ x) : f 3 = -11 :=
sorry

end find_f3_l32_3236


namespace minimum_m_l32_3208

theorem minimum_m (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 24 * m = n ^ 4) : m ≥ 54 :=
sorry

end minimum_m_l32_3208


namespace find_x_l32_3293

noncomputable def x_value : ℝ :=
  let x := 24
  x

theorem find_x (x : ℝ) (h : 7 * x + 3 * x + 4 * x + x = 360) : x = 24 := by
  sorry

end find_x_l32_3293


namespace negation_proposition_correct_l32_3291

theorem negation_proposition_correct : 
  (∀ x : ℝ, 0 < x → x + 4 / x ≥ 4) :=
by
  intro x hx
  sorry

end negation_proposition_correct_l32_3291


namespace inverse_proportional_ratios_l32_3261

variables {x y x1 x2 y1 y2 : ℝ}
variables (h_inv_prop : ∀ (x y : ℝ), x * y = 1) (hx1_ne : x1 ≠ 0) (hx2_ne : x2 ≠ 0) (hy1_ne : y1 ≠ 0) (hy2_ne : y2 ≠ 0)

theorem inverse_proportional_ratios 
  (h1 : x1 * y1 = x2 * y2)
  (h2 : (x1 / x2) = (3 / 4)) : 
  (y1 / y2) = (4 / 3) :=
by 

sorry

end inverse_proportional_ratios_l32_3261


namespace students_sampled_from_second_grade_l32_3201

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

end students_sampled_from_second_grade_l32_3201


namespace exp_values_l32_3241

variable {a x y : ℝ}

theorem exp_values (hx : a^x = 3) (hy : a^y = 2) :
  a^(x - y) = 3 / 2 ∧ a^(2 * x + y) = 18 :=
by
  sorry

end exp_values_l32_3241


namespace boys_from_pine_l32_3238

/-- 
Given the following conditions:
1. There are 150 students at the camp.
2. There are 90 boys at the camp.
3. There are 60 girls at the camp.
4. There are 70 students from Maple High School.
5. There are 80 students from Pine High School.
6. There are 20 girls from Oak High School.
7. There are 30 girls from Maple High School.

Prove that the number of boys from Pine High School is 70.
--/
theorem boys_from_pine (total_students boys girls maple_high pine_high oak_girls maple_girls : ℕ)
  (H1 : total_students = 150)
  (H2 : boys = 90)
  (H3 : girls = 60)
  (H4 : maple_high = 70)
  (H5 : pine_high = 80)
  (H6 : oak_girls = 20)
  (H7 : maple_girls = 30) : 
  ∃ pine_boys : ℕ, pine_boys = 70 :=
by
  -- Proof goes here
  sorry

end boys_from_pine_l32_3238


namespace cube_inscribed_circumscribed_volume_ratio_l32_3251

theorem cube_inscribed_circumscribed_volume_ratio
  (S_1 S_2 V_1 V_2 : ℝ)
  (h : S_1 / S_2 = (1 / Real.sqrt 2) ^ 2) :
  V_1 / V_2 = (Real.sqrt 3 / 3) ^ 3 :=
sorry

end cube_inscribed_circumscribed_volume_ratio_l32_3251


namespace liquid_flow_problem_l32_3253

variables (x y z : ℝ)

theorem liquid_flow_problem 
    (h1 : 1/x + 1/y + 1/z = 1/6) 
    (h2 : y = 0.75 * x) 
    (h3 : z = y + 10) : 
    x = 56/3 ∧ y = 14 ∧ z = 24 :=
sorry

end liquid_flow_problem_l32_3253


namespace sample_size_is_100_l32_3205

-- Define the number of students selected for the sample.
def num_students_sampled : ℕ := 100

-- The statement that the sample size is equal to the number of students sampled.
theorem sample_size_is_100 : num_students_sampled = 100 := 
by {
  -- Proof goes here
  sorry
}

end sample_size_is_100_l32_3205


namespace value_of_expression_l32_3211

theorem value_of_expression (x : ℝ) (h : x^2 + x + 1 = 8) : 4 * x^2 + 4 * x + 9 = 37 :=
by
  sorry

end value_of_expression_l32_3211


namespace trapezium_area_l32_3275

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) : 
  (1/2) * (a + b) * h = 285 :=
by
  rw [ha, hb, hh]
  norm_num

end trapezium_area_l32_3275


namespace four_digit_integer_l32_3248

theorem four_digit_integer (a b c d : ℕ) 
(h1: a + b + c + d = 14) (h2: b + c = 9) (h3: a - d = 1)
(h4: (a - b + c - d) % 11 = 0) : 1000 * a + 100 * b + 10 * c + d = 3542 :=
by
  sorry

end four_digit_integer_l32_3248


namespace find_a_l32_3226

noncomputable def f (x a : ℝ) : ℝ := x / (x^2 + a)

theorem find_a (a : ℝ) (h_positive : a > 0) (h_max : ∀ x, x ∈ Set.Ici 1 → f x a ≤ f 1 a) :
  a = Real.sqrt 3 - 1 := by
  sorry

end find_a_l32_3226


namespace am_gm_inequality_l32_3235

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c) / 3) :=
sorry

end am_gm_inequality_l32_3235


namespace find_a_b_max_min_values_l32_3245

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (1/3) * x^3 + a * x^2 + b * x

noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^2 + 2 * a * x + b

theorem find_a_b (a b : ℝ) :
  f' (-3) a b = 0 ∧ f (-3) a b = 9 → a = 1 ∧ b = -3 :=
  by sorry

theorem max_min_values (a b : ℝ) (h₁ : a = 1) (h₂ : b = -3):
  ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x a b ≥ -5 / 3 ∧ f x a b ≤ 9 :=
  by sorry

end find_a_b_max_min_values_l32_3245


namespace percentage_problem_l32_3294

theorem percentage_problem (p x : ℝ) (h1 : (p / 100) * x = 400) (h2 : (120 / 100) * x = 2400) : p = 20 := by
  sorry

end percentage_problem_l32_3294


namespace minimum_rebate_rate_l32_3280

open Real

noncomputable def rebate_rate (s p_M p_N p: ℝ) : ℝ := 100 * (p_M + p_N - p) / s

theorem minimum_rebate_rate 
  (s p_M p_N p : ℝ)
  (h_M : 0.19 * 0.4 * s ≤ p_M ∧ p_M ≤ 0.24 * 0.4 * s)
  (h_N : 0.29 * 0.6 * s ≤ p_N ∧ p_N ≤ 0.34 * 0.6 * s)
  (h_total : 0.10 * s ≤ p ∧ p ≤ 0.15 * s) :
  ∃ r : ℝ, r = rebate_rate s p_M p_N p ∧ 0.1 ≤ r ∧ r ≤ 0.2 :=
sorry

end minimum_rebate_rate_l32_3280


namespace vertical_complementary_perpendicular_l32_3266

theorem vertical_complementary_perpendicular (α β : ℝ) (l1 l2 : ℝ) :
  (α = β ∧ α + β = 90) ∧ l1 = l2 -> l1 + l2 = 90 := by
  sorry

end vertical_complementary_perpendicular_l32_3266


namespace tan_alpha_beta_l32_3273

theorem tan_alpha_beta (α β : ℝ) (h : 2 * Real.sin β = Real.sin (2 * α + β)) :
  Real.tan (α + β) = 3 * Real.tan α := 
sorry

end tan_alpha_beta_l32_3273


namespace area_of_triangle_PQR_is_correct_l32_3287

noncomputable def calculate_area_of_triangle_PQR : ℝ := 
  let side_length := 4
  let altitude := 8
  let WO := (side_length * Real.sqrt 2) / 2
  let center_to_vertex_distance := Real.sqrt (WO^2 + altitude^2)
  let WP := (1 / 4) * WO
  let YQ := (1 / 2) * WO
  let XR := (3 / 4) * WO
  let area := (1 / 2) * (YQ - WP) * (XR - YQ)
  area

theorem area_of_triangle_PQR_is_correct :
  calculate_area_of_triangle_PQR = 2.25 := sorry

end area_of_triangle_PQR_is_correct_l32_3287


namespace height_difference_l32_3292

variable (H_A H_B : ℝ)

-- Conditions
axiom B_is_66_67_percent_more_than_A : H_B = H_A * 1.6667

-- Proof statement
theorem height_difference (H_A H_B : ℝ) (h : H_B = H_A * 1.6667) : 
  (H_B - H_A) / H_B * 100 = 40 := by
sorry

end height_difference_l32_3292


namespace coins_division_remainder_l32_3231

theorem coins_division_remainder
  (n : ℕ)
  (h1 : n % 6 = 4)
  (h2 : n % 5 = 3)
  (h3 : n = 28) :
  n % 7 = 0 :=
by
  sorry

end coins_division_remainder_l32_3231


namespace motel_total_rent_l32_3276

theorem motel_total_rent (R₅₀ R₆₀ : ℕ) 
  (h₁ : ∀ x y : ℕ, 50 * x + 60 * y = 50 * (x + 10) + 60 * (y - 10) + 100)
  (h₂ : ∀ x y : ℕ, 25 * (50 * x + 60 * y) = 10000) : 
  50 * R₅₀ + 60 * R₆₀ = 400 :=
by
  sorry

end motel_total_rent_l32_3276


namespace number_division_l32_3200

theorem number_division (x : ℤ) (h : x - 17 = 55) : x / 9 = 8 :=
by 
  sorry

end number_division_l32_3200


namespace last_four_digits_of_5_pow_9000_l32_3259

theorem last_four_digits_of_5_pow_9000 (h : 5^300 ≡ 1 [MOD 1250]) : 
  5^9000 ≡ 1 [MOD 1250] :=
sorry

end last_four_digits_of_5_pow_9000_l32_3259


namespace f_is_odd_and_increasing_l32_3260

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_is_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = - f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end f_is_odd_and_increasing_l32_3260


namespace tangent_line_at_point_is_correct_l32_3298

theorem tangent_line_at_point_is_correct :
  ∀ (x y : ℝ), (y = x^2 + 2 * x) → (x = 1) → (y = 3) → (4 * x - y - 1 = 0) :=
by
  intros x y h_curve h_x h_y
  -- Here would be the proof
  sorry

end tangent_line_at_point_is_correct_l32_3298


namespace cube_difference_divisible_by_16_l32_3270

theorem cube_difference_divisible_by_16 (a b : ℤ) : 
  16 ∣ ((2 * a + 1)^3 - (2 * b + 1)^3 + 8) :=
by
  sorry

end cube_difference_divisible_by_16_l32_3270


namespace num_diagonals_29_sides_l32_3295

-- Define the number of sides
def n : Nat := 29

-- Calculate the combination (binomial coefficient) of selecting 2 vertices from n vertices
def binom (n k : Nat) : Nat := Nat.choose n k

-- Define the number of diagonals in a polygon with n sides
def num_diagonals (n : Nat) : Nat := binom n 2 - n

-- State the theorem to prove the number of diagonals for a polygon with 29 sides is 377
theorem num_diagonals_29_sides : num_diagonals 29 = 377 :=
by
  sorry

end num_diagonals_29_sides_l32_3295


namespace eq_triangle_perimeter_l32_3225

theorem eq_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end eq_triangle_perimeter_l32_3225


namespace shaded_percentage_of_large_square_l32_3224

theorem shaded_percentage_of_large_square
  (side_length_small_square : ℕ)
  (side_length_large_square : ℕ)
  (total_border_squares : ℕ)
  (shaded_border_squares : ℕ)
  (central_region_shaded_fraction : ℚ)
  (total_area_large_square : ℚ)
  (shaded_area_border_squares : ℚ)
  (shaded_area_central_region : ℚ) :
  side_length_small_square = 1 →
  side_length_large_square = 5 →
  total_border_squares = 16 →
  shaded_border_squares = 8 →
  central_region_shaded_fraction = 3 / 4 →
  total_area_large_square = 25 →
  shaded_area_border_squares = 8 →
  shaded_area_central_region = (3 / 4) * 9 →
  (shaded_area_border_squares + shaded_area_central_region) / total_area_large_square = 0.59 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end shaded_percentage_of_large_square_l32_3224


namespace max_value_on_interval_l32_3288

noncomputable def f (x : ℝ) := 2 * x ^ 3 - 6 * x ^ 2 + 10

theorem max_value_on_interval :
  (∀ x ∈ Set.Icc (1 : ℝ) 3, f 2 <= f x) → 
  ∃ y ∈ Set.Icc (1 : ℝ) 3, ∀ z ∈ Set.Icc (1 : ℝ) 3, f y >= f z :=
by
  sorry

end max_value_on_interval_l32_3288


namespace roots_positive_range_no_negative_roots_opposite_signs_range_l32_3222

theorem roots_positive_range (m : ℝ) : (8 * x^2 - m * x + (m - 6) = 0) → (6 < m ∧ m ≤ 8 ∨ m ≥ 24) :=
sorry

theorem no_negative_roots (m : ℝ) : (8 * x^2 - m * x + (m - 6) = 0) → ¬ (∀ α β, (α < 0 ∧ β < 0)) :=
sorry

theorem opposite_signs_range (m : ℝ) : (8 * x^2 - m * x + (m - 6) = 0) → m < 6 :=
sorry

end roots_positive_range_no_negative_roots_opposite_signs_range_l32_3222


namespace find_g2_l32_3281

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (x - 1)
noncomputable def g (f : ℝ → ℝ) (y : ℝ) : ℝ := f⁻¹ y

variable (a : ℝ)
variable (h_inv : ∀ (x : ℝ), g (f a) (f a x) = x)
variable (h_g4 : g (f a) 4 = 2)

theorem find_g2 : g (f a) 2 = 3 / 2 :=
by sorry

end find_g2_l32_3281


namespace equation_solution_l32_3272

theorem equation_solution (x : ℝ) : (3 : ℝ)^(x-1) = 1/9 ↔ x = -1 :=
by sorry

end equation_solution_l32_3272


namespace second_polygon_sides_l32_3237

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0) : 
  ∀ (P1 P2 : ℝ) (n1 n2 : ℕ), 
    n1 = 50 →
    P1 = n1 * (3 * s) →
    P2 = n2 * s →
    P1 = P2 →
    n2 = 150 :=
by
  intros P1 P2 n1 n2 hn1 hp1 hp2 hp
  sorry

end second_polygon_sides_l32_3237


namespace sin_alpha_value_l32_3239

open Real

theorem sin_alpha_value (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : cos (α + π / 4) = 1 / 3) :
  sin α = (4 - sqrt 2) / 6 :=
sorry

end sin_alpha_value_l32_3239


namespace dice_sum_not_possible_l32_3202

theorem dice_sum_not_possible (a b c d : ℕ) (h₁ : 1 ≤ a ∧ a ≤ 6) (h₂ : 1 ≤ b ∧ b ≤ 6) 
(h₃ : 1 ≤ c ∧ c ≤ 6) (h₄ : 1 ≤ d ∧ d ≤ 6) (h_product : a * b * c * d = 216) : 
(a + b + c + d ≠ 15) ∧ (a + b + c + d ≠ 16) ∧ (a + b + c + d ≠ 18) :=
sorry

end dice_sum_not_possible_l32_3202


namespace sum_of_decimals_l32_3265

theorem sum_of_decimals : (1 / 10) + (9 / 100) + (9 / 1000) + (7 / 10000) = 0.1997 := 
sorry

end sum_of_decimals_l32_3265


namespace marys_income_percent_of_juans_income_l32_3234

variables (M T J : ℝ)

theorem marys_income_percent_of_juans_income (h1 : M = 1.40 * T) (h2 : T = 0.60 * J) : M = 0.84 * J :=
by
  sorry

end marys_income_percent_of_juans_income_l32_3234


namespace range_of_x_in_function_l32_3223

theorem range_of_x_in_function (x : ℝ) : (y = 1/(x + 3) → x ≠ -3) :=
sorry

end range_of_x_in_function_l32_3223


namespace amount_of_bill_is_720_l32_3267

-- Definitions and conditions
def TD : ℝ := 360
def BD : ℝ := 428.21

-- The relationship between TD, BD, and FV
axiom relationship (FV : ℝ) : BD = TD + (TD * BD) / (FV - TD)

-- The main theorem to prove
theorem amount_of_bill_is_720 : ∃ FV : ℝ, BD = TD + (TD * BD) / (FV - TD) ∧ FV = 720 :=
by
  use 720
  sorry

end amount_of_bill_is_720_l32_3267


namespace initial_money_proof_l32_3230

-- Definition: Dan's initial money, the money spent, and the money left.
def initial_money : ℝ := sorry
def spent_money : ℝ := 1.0
def left_money : ℝ := 2.0

-- Theorem: Prove that Dan's initial money is the sum of the money spent and the money left.
theorem initial_money_proof : initial_money = spent_money + left_money :=
sorry

end initial_money_proof_l32_3230


namespace off_the_rack_suit_cost_l32_3290

theorem off_the_rack_suit_cost (x : ℝ)
  (h1 : ∀ y, y = 3 * x + 200)
  (h2 : ∀ y, x + y = 1400) :
  x = 300 :=
by
  sorry

end off_the_rack_suit_cost_l32_3290


namespace cos_double_angle_l32_3243

open Real

theorem cos_double_angle (α β : ℝ) 
    (h1 : sin α = 2 * sin β) 
    (h2 : tan α = 3 * tan β) :
  cos (2 * α) = -1 / 4 ∨ cos (2 * α) = 1 := 
sorry

end cos_double_angle_l32_3243


namespace expression_evaluation_l32_3297

theorem expression_evaluation : 3 * 257 + 4 * 257 + 2 * 257 + 258 = 2571 := by
  sorry

end expression_evaluation_l32_3297
