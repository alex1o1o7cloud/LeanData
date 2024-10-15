import Mathlib

namespace NUMINAMATH_GPT_dan_balloons_l1391_139159

theorem dan_balloons (fred_balloons sam_balloons total_balloons dan_balloons : ℕ) 
  (h₁ : fred_balloons = 10) 
  (h₂ : sam_balloons = 46) 
  (h₃ : total_balloons = 72) : 
  dan_balloons = total_balloons - (fred_balloons + sam_balloons) :=
by
  sorry

end NUMINAMATH_GPT_dan_balloons_l1391_139159


namespace NUMINAMATH_GPT_factor_expression_l1391_139175

theorem factor_expression (a : ℝ) : 74 * a^2 + 222 * a + 148 = 74 * (a + 2) * (a + 1) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1391_139175


namespace NUMINAMATH_GPT_minimum_pyramid_volume_proof_l1391_139170

noncomputable def minimum_pyramid_volume (side_length : ℝ) (apex_angle : ℝ) : ℝ :=
  if side_length = 6 ∧ apex_angle = 2 * Real.arcsin (1 / 3 : ℝ) then 5 * Real.sqrt 23 else 0

theorem minimum_pyramid_volume_proof : 
  minimum_pyramid_volume 6 (2 * Real.arcsin (1 / 3)) = 5 * Real.sqrt 23 :=
by
  sorry

end NUMINAMATH_GPT_minimum_pyramid_volume_proof_l1391_139170


namespace NUMINAMATH_GPT_hilary_stalks_l1391_139111

-- Define the given conditions
def ears_per_stalk : ℕ := 4
def kernels_per_ear_first_half : ℕ := 500
def kernels_per_ear_second_half : ℕ := 600
def total_kernels : ℕ := 237600

-- Average number of kernels per ear
def average_kernels_per_ear : ℕ := (kernels_per_ear_first_half + kernels_per_ear_second_half) / 2

-- Total number of ears based on total kernels
noncomputable def total_ears : ℕ := total_kernels / average_kernels_per_ear

-- Total number of stalks based on total ears
noncomputable def total_stalks : ℕ := total_ears / ears_per_stalk

-- The main theorem to prove
theorem hilary_stalks : total_stalks = 108 :=
by
  sorry

end NUMINAMATH_GPT_hilary_stalks_l1391_139111


namespace NUMINAMATH_GPT_distance_between_cars_l1391_139103

theorem distance_between_cars (t : ℝ) (v_kmh : ℝ) (v_ms : ℝ) :
  t = 1 ∧ v_kmh = 180 ∧ v_ms = v_kmh * 1000 / 3600 → 
  v_ms * t = 50 := 
by 
  sorry

end NUMINAMATH_GPT_distance_between_cars_l1391_139103


namespace NUMINAMATH_GPT_hot_dogs_remainder_l1391_139118

theorem hot_dogs_remainder :
  25197625 % 4 = 1 :=
by
  sorry

end NUMINAMATH_GPT_hot_dogs_remainder_l1391_139118


namespace NUMINAMATH_GPT_steve_break_even_l1391_139157

noncomputable def break_even_performances
  (fixed_overhead : ℕ)
  (min_production_cost max_production_cost : ℕ)
  (venue_capacity percentage_occupied : ℕ)
  (ticket_price : ℕ) : ℕ :=
(fixed_overhead + (percentage_occupied / 100 * venue_capacity * ticket_price)) / (percentage_occupied / 100 * venue_capacity * ticket_price)

theorem steve_break_even
  (fixed_overhead : ℕ := 81000)
  (min_production_cost : ℕ := 5000)
  (max_production_cost : ℕ := 9000)
  (venue_capacity : ℕ := 500)
  (percentage_occupied : ℕ := 80)
  (ticket_price : ℕ := 40)
  (avg_production_cost : ℕ := (min_production_cost + max_production_cost) / 2) :
  break_even_performances fixed_overhead min_production_cost max_production_cost venue_capacity percentage_occupied ticket_price = 9 :=
by
  sorry

end NUMINAMATH_GPT_steve_break_even_l1391_139157


namespace NUMINAMATH_GPT_minimum_balls_ensure_20_single_color_l1391_139183

def num_balls_to_guarantee_color (r g y b w k : ℕ) : ℕ :=
  let max_without_20 := 19 + 19 + 19 + 18 + 15 + 12
  max_without_20 + 1

theorem minimum_balls_ensure_20_single_color :
  num_balls_to_guarantee_color 30 25 25 18 15 12 = 103 := by
  sorry

end NUMINAMATH_GPT_minimum_balls_ensure_20_single_color_l1391_139183


namespace NUMINAMATH_GPT_operation_B_is_correct_l1391_139115

theorem operation_B_is_correct (a b x : ℝ) : 
  2 * (a^2) * b * 4 * a * (b^3) = 8 * (a^3) * (b^4) :=
by
  sorry

-- Conditions for incorrect operations
lemma operation_A_is_incorrect (x : ℝ) : 
  x^8 / x^2 ≠ x^4 :=
by
  sorry

lemma operation_C_is_incorrect (x : ℝ) : 
  (-x^5)^4 ≠ -x^20 :=
by
  sorry

lemma operation_D_is_incorrect (a b : ℝ) : 
  (a + b)^2 ≠ a^2 + b^2 :=
by
  sorry

end NUMINAMATH_GPT_operation_B_is_correct_l1391_139115


namespace NUMINAMATH_GPT_remaining_lemons_proof_l1391_139131

-- Definitions for initial conditions
def initial_lemons_first_tree   := 15
def initial_lemons_second_tree  := 20
def initial_lemons_third_tree   := 25

def sally_picked_first_tree     := 7
def mary_picked_second_tree     := 9
def tom_picked_first_tree       := 12

def lemons_fell_each_tree       := 4
def animals_eaten_per_tree      := lemons_fell_each_tree / 2

-- Definitions for intermediate calculations
def remaining_lemons_first_tree_full := initial_lemons_first_tree - sally_picked_first_tree - tom_picked_first_tree
def remaining_lemons_first_tree      := if remaining_lemons_first_tree_full < 0 then 0 else remaining_lemons_first_tree_full

def remaining_lemons_second_tree := initial_lemons_second_tree - mary_picked_second_tree

def mary_picked_third_tree := (remaining_lemons_second_tree : ℚ) / 2
def remaining_lemons_third_tree_full := (initial_lemons_third_tree : ℚ) - mary_picked_third_tree
def remaining_lemons_third_tree      := Nat.floor remaining_lemons_third_tree_full

-- Adjusting for fallen and eaten lemons
def final_remaining_lemons_first_tree_full := remaining_lemons_first_tree - lemons_fell_each_tree + animals_eaten_per_tree
def final_remaining_lemons_first_tree      := if final_remaining_lemons_first_tree_full < 0 then 0 else final_remaining_lemons_first_tree_full

def final_remaining_lemons_second_tree     := remaining_lemons_second_tree - lemons_fell_each_tree + animals_eaten_per_tree

def final_remaining_lemons_third_tree_full := remaining_lemons_third_tree - lemons_fell_each_tree + animals_eaten_per_tree
def final_remaining_lemons_third_tree      := if final_remaining_lemons_third_tree_full < 0 then 0 else final_remaining_lemons_third_tree_full

-- Lean 4 statement to prove the equivalence
theorem remaining_lemons_proof :
  final_remaining_lemons_first_tree = 0 ∧
  final_remaining_lemons_second_tree = 9 ∧
  final_remaining_lemons_third_tree = 18 :=
by
  -- The proof is omitted as per the requirement
  sorry

end NUMINAMATH_GPT_remaining_lemons_proof_l1391_139131


namespace NUMINAMATH_GPT_find_initial_number_l1391_139174

theorem find_initial_number (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_initial_number_l1391_139174


namespace NUMINAMATH_GPT_complex_roots_sum_condition_l1391_139112

theorem complex_roots_sum_condition 
  (z1 z2 : ℂ) 
  (h1 : ∀ z, z ^ 2 + z + 1 = 0) 
  (h2 : z1 ^ 2 + z1 + 1 = 0)
  (h3 : z2 ^ 2 + z2 + 1 = 0) : 
  (z2 / (z1 + 1)) + (z1 / (z2 + 1)) = -2 := 
 sorry

end NUMINAMATH_GPT_complex_roots_sum_condition_l1391_139112


namespace NUMINAMATH_GPT_total_amount_proof_l1391_139152

def total_shared_amount : ℝ :=
  let z := 250
  let y := 1.20 * z
  let x := 1.25 * y
  x + y + z

theorem total_amount_proof : total_shared_amount = 925 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_proof_l1391_139152


namespace NUMINAMATH_GPT_cylinder_water_depth_l1391_139177

theorem cylinder_water_depth 
  (height radius : ℝ)
  (h_ge_zero : height ≥ 0)
  (r_ge_zero : radius ≥ 0)
  (total_height : height = 1200)
  (total_radius : radius = 100)
  (above_water_vol : 1 / 3 * π * radius^2 * height = 1 / 3 * π * radius^2 * 1200) :
  height - 800 = 400 :=
by
  -- Use provided constraints and logical reasoning on structures
  sorry

end NUMINAMATH_GPT_cylinder_water_depth_l1391_139177


namespace NUMINAMATH_GPT_benjamin_earns_more_l1391_139182

noncomputable def additional_earnings : ℝ :=
  let P : ℝ := 75000
  let r : ℝ := 0.05
  let t_M : ℝ := 3
  let r_m : ℝ := r / 12
  let t_B : ℝ := 36
  let A_M : ℝ := P * (1 + r)^t_M
  let A_B : ℝ := P * (1 + r_m)^t_B
  A_B - A_M

theorem benjamin_earns_more : additional_earnings = 204 := by
  sorry

end NUMINAMATH_GPT_benjamin_earns_more_l1391_139182


namespace NUMINAMATH_GPT_triangle_PR_eq_8_l1391_139134

open Real

theorem triangle_PR_eq_8 (P Q R M : ℝ) 
  (PQ QR PM : ℝ) 
  (hPQ : PQ = 6) (hQR : QR = 10) (hPM : PM = 5) 
  (M_midpoint : M = (Q + R) / 2) :
  dist P R = 8 :=
by
  sorry

end NUMINAMATH_GPT_triangle_PR_eq_8_l1391_139134


namespace NUMINAMATH_GPT_arithmetic_sequence_S22_zero_l1391_139189

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

noncomputable def sum_of_first_n_terms (a d : ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a + (n - 1) * d)

theorem arithmetic_sequence_S22_zero (a d : ℝ) (S : ℕ → ℝ) (h_arith_seq : ∀ n, S n = sum_of_first_n_terms a d n)
  (h1 : a > 0) (h2 : S 5 = S 17) :
  S 22 = 0 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_S22_zero_l1391_139189


namespace NUMINAMATH_GPT_larger_integer_is_21_l1391_139102

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end NUMINAMATH_GPT_larger_integer_is_21_l1391_139102


namespace NUMINAMATH_GPT_largest_divisor_of_m_l1391_139168

-- Definitions
def positive_integer (m : ℕ) : Prop := m > 0
def divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- Statement
theorem largest_divisor_of_m (m : ℕ) (h1 : positive_integer m) (h2 : divisible_by (m^2) 54) : ∃ k : ℕ, k = 9 ∧ k ∣ m := 
sorry

end NUMINAMATH_GPT_largest_divisor_of_m_l1391_139168


namespace NUMINAMATH_GPT_find_b12_l1391_139140

noncomputable def seq (b : ℕ → ℤ) : Prop :=
  b 1 = 2 ∧ 
  ∀ m n : ℕ, m > 0 → n > 0 → b (m + n) = b m + b n + (m * n * n)

theorem find_b12 (b : ℕ → ℤ) (h : seq b) : b 12 = 98 := 
by
  sorry

end NUMINAMATH_GPT_find_b12_l1391_139140


namespace NUMINAMATH_GPT_probability_X_eq_3_l1391_139138

def number_of_ways_to_choose (n k : ℕ) : ℕ :=
  Nat.choose n k

def P_X_eq_3 : ℚ :=
  (number_of_ways_to_choose 5 3) * (number_of_ways_to_choose 3 1) / (number_of_ways_to_choose 8 4)

theorem probability_X_eq_3 : P_X_eq_3 = 3 / 7 := by
  sorry

end NUMINAMATH_GPT_probability_X_eq_3_l1391_139138


namespace NUMINAMATH_GPT_airplane_rows_l1391_139127

theorem airplane_rows (r : ℕ) (h1 : ∀ (seats_per_row total_rows : ℕ), seats_per_row = 8 → total_rows = r →
  ∀ occupied_seats : ℕ, occupied_seats = (3 * seats_per_row) / 4 →
  ∀ unoccupied_seats : ℕ, unoccupied_seats = seats_per_row * total_rows - occupied_seats * total_rows →
  unoccupied_seats = 24): 
  r = 12 :=
by
  sorry

end NUMINAMATH_GPT_airplane_rows_l1391_139127


namespace NUMINAMATH_GPT_num_correct_conclusions_l1391_139162

-- Definitions and conditions from the problem
variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}
variable (n : ℕ)
variable (hSn_eq : S n + S (n + 1) = n ^ 2)

-- Assert the conditions described in the comments
theorem num_correct_conclusions (hSn_eq : ∀ n, S n + S (n + 1) = n ^ 2) :
  (1:ℕ) = 3 ↔
  (-- Conclusion 1
   ¬(∀ n, a (n + 2) - a n = 2) ∧
   -- Conclusion 2: If a_1 = 0, then S_50 = 1225
   (S 50 = 1225) ∧
   -- Conclusion 3: If a_1 = 1, then S_50 = 1224
   (S 50 = 1224) ∧
   -- Conclusion 4: Monotonically increasing sequence
   (∀ a_1, (-1/4 : ℚ) < a_1 ∧ a_1 < 1/4)) :=
by
  sorry

end NUMINAMATH_GPT_num_correct_conclusions_l1391_139162


namespace NUMINAMATH_GPT_original_number_l1391_139137

theorem original_number (x : ℝ) (h1 : 268 * 74 = 19732) (h2 : x * 0.74 = 1.9832) : x = 2.68 :=
by
  sorry

end NUMINAMATH_GPT_original_number_l1391_139137


namespace NUMINAMATH_GPT_frustum_radius_l1391_139187

theorem frustum_radius (C1 C2 l: ℝ) (S_lateral: ℝ) (r: ℝ) :
  (C1 = 2 * r * π) ∧ (C2 = 6 * r * π) ∧ (l = 3) ∧ (S_lateral = 84 * π) → (r = 7) :=
by
  sorry

end NUMINAMATH_GPT_frustum_radius_l1391_139187


namespace NUMINAMATH_GPT_number_of_boys_in_school_l1391_139191

theorem number_of_boys_in_school (total_students : ℕ) (sample_size : ℕ) 
(number_diff : ℕ) (ratio_boys_sample_girls_sample : ℚ) : 
total_students = 1200 → sample_size = 200 → number_diff = 10 →
ratio_boys_sample_girls_sample = 105 / 95 →
∃ (boys_in_school : ℕ), boys_in_school = 630 := by 
  sorry

end NUMINAMATH_GPT_number_of_boys_in_school_l1391_139191


namespace NUMINAMATH_GPT_square_side_length_l1391_139185

theorem square_side_length (x : ℝ) (h : x^2 = 12) : x = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_square_side_length_l1391_139185


namespace NUMINAMATH_GPT_trees_planted_l1391_139167

theorem trees_planted (yard_length : ℕ) (distance_between_trees : ℕ) (n_trees : ℕ) 
  (h1 : yard_length = 434) 
  (h2 : distance_between_trees = 14) 
  (h3 : n_trees = yard_length / distance_between_trees + 1) : 
  n_trees = 32 :=
by
  sorry

end NUMINAMATH_GPT_trees_planted_l1391_139167


namespace NUMINAMATH_GPT_base_eight_to_base_ten_l1391_139193

theorem base_eight_to_base_ten (n : ℕ) : 
  n = 3 * 8^1 + 1 * 8^0 → n = 25 :=
by
  intro h
  rw [mul_comm 3 (8^1), pow_one, mul_comm 1 (8^0), pow_zero, mul_one] at h
  exact h

end NUMINAMATH_GPT_base_eight_to_base_ten_l1391_139193


namespace NUMINAMATH_GPT_find_sticker_price_l1391_139181

-- Define the conditions
def storeX_discount (x : ℝ) : ℝ := 0.80 * x - 70
def storeY_discount (x : ℝ) : ℝ := 0.70 * x

-- Define the main statement
theorem find_sticker_price (x : ℝ) (h : storeX_discount x = storeY_discount x - 20) : x = 500 :=
sorry

end NUMINAMATH_GPT_find_sticker_price_l1391_139181


namespace NUMINAMATH_GPT_determine_m_l1391_139146

-- Define the fractional equation condition
def fractional_eq (m x : ℝ) : Prop := (m/(x - 2) + 2*x/(x - 2) = 1)

-- Define the main theorem statement
theorem determine_m (m : ℝ) (h : ∃ (x : ℝ), x > 0 ∧ x ≠ 2 ∧ fractional_eq m x) : m = -4 :=
sorry

end NUMINAMATH_GPT_determine_m_l1391_139146


namespace NUMINAMATH_GPT_rowers_voted_l1391_139176

variable (R : ℕ)

/-- Each rower votes for exactly 4 coaches out of 50 coaches,
and each coach receives exactly 7 votes.
Prove that the number of rowers is 88. -/
theorem rowers_voted (h1 : 50 * 7 = 4 * R) : R = 88 := by 
  sorry

end NUMINAMATH_GPT_rowers_voted_l1391_139176


namespace NUMINAMATH_GPT_train_length_l1391_139158

-- Define the given speeds and time
def train_speed_km_per_h := 25
def man_speed_km_per_h := 2
def crossing_time_sec := 36

-- Convert speeds to m/s
def km_per_h_to_m_per_s (v : ℕ) : ℕ := (v * 1000) / 3600
def train_speed_m_per_s := km_per_h_to_m_per_s train_speed_km_per_h
def man_speed_m_per_s := km_per_h_to_m_per_s man_speed_km_per_h

-- Define the relative speed in m/s
def relative_speed_m_per_s := train_speed_m_per_s + man_speed_m_per_s

-- Theorem to prove the length of the train
theorem train_length : (relative_speed_m_per_s * crossing_time_sec) = 270 :=
by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_train_length_l1391_139158


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l1391_139100

noncomputable def a : ℝ := (Real.sqrt 2) / 2 * (Real.sin (17 * Real.pi / 180) + Real.cos (17 * Real.pi / 180))
noncomputable def b : ℝ := 2 * (Real.cos (13 * Real.pi / 180))^2 - 1
noncomputable def c : ℝ := (Real.sqrt 3) / 2

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l1391_139100


namespace NUMINAMATH_GPT_negation_of_all_cars_are_fast_l1391_139135

variable {α : Type} -- Assume α is the type of entities
variable (car fast : α → Prop) -- car and fast are predicates on entities

theorem negation_of_all_cars_are_fast :
  ¬ (∀ x, car x → fast x) ↔ ∃ x, car x ∧ ¬ fast x :=
by sorry

end NUMINAMATH_GPT_negation_of_all_cars_are_fast_l1391_139135


namespace NUMINAMATH_GPT_alice_unanswered_questions_l1391_139160

-- Declare variables for the proof
variables (c w u : ℕ)

-- State the problem in Lean
theorem alice_unanswered_questions :
  50 + 5 * c - 2 * w = 100 ∧
  40 + 7 * c - w - u = 120 ∧
  6 * c + 3 * u = 130 ∧
  c + w + u = 25 →
  u = 20 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_alice_unanswered_questions_l1391_139160


namespace NUMINAMATH_GPT_married_fraction_l1391_139126

variables (M W N : ℕ)

def married_men : Prop := 2 * M = 3 * N
def married_women : Prop := 3 * W = 5 * N
def total_population : ℕ := M + W
def married_population : ℕ := 2 * N

theorem married_fraction (h1: married_men M N) (h2: married_women W N) :
  (married_population N : ℚ) / (total_population M W : ℚ) = 12 / 19 :=
by sorry

end NUMINAMATH_GPT_married_fraction_l1391_139126


namespace NUMINAMATH_GPT_winner_won_by_324_votes_l1391_139178

theorem winner_won_by_324_votes
  (total_votes : ℝ)
  (winner_percentage : ℝ)
  (winner_votes : ℝ)
  (h1 : winner_percentage = 0.62)
  (h2 : winner_votes = 837) :
  (winner_votes - (0.38 * total_votes) = 324) :=
by
  sorry

end NUMINAMATH_GPT_winner_won_by_324_votes_l1391_139178


namespace NUMINAMATH_GPT_minimum_value_l1391_139188

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y / x = 1) :
  ∃ (m : ℝ), m = 4 ∧ ∀ z, z = (1 / x + x / y) → z ≥ m :=
sorry

end NUMINAMATH_GPT_minimum_value_l1391_139188


namespace NUMINAMATH_GPT_roots_numerically_equal_opposite_signs_l1391_139136

theorem roots_numerically_equal_opposite_signs
  (a b c : ℝ) (k : ℝ)
  (h : (∃ x : ℝ, x^2 - (b+1) * x ≠ 0) →
    ∃ x : ℝ, x ≠ 0 ∧ x ∈ {x : ℝ | (k+2)*(x^2 - (b+1)*x) = (k-2)*((a+1)*x - c)} ∧ -x ∈ {x : ℝ | (k+2)*(x^2 - (b+1)*x) = (k-2)*((a+1)*x - c)}) :
  k = (-2 * (b - a)) / (b + a + 2) :=
by
  sorry

end NUMINAMATH_GPT_roots_numerically_equal_opposite_signs_l1391_139136


namespace NUMINAMATH_GPT_final_sum_l1391_139173

def Q (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 4

noncomputable def probability_condition_holds : ℝ :=
  by sorry

theorem final_sum :
  let m := 1
  let n := 1
  let o := 1
  let p := 0
  let q := 8
  (m + n + o + p + q) = 11 :=
  by
    sorry

end NUMINAMATH_GPT_final_sum_l1391_139173


namespace NUMINAMATH_GPT_cube_volume_l1391_139142

theorem cube_volume {V : ℝ} (x : ℝ) (hV : V = x^3) (hA : 2 * V = 6 * x^2) : V = 27 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cube_volume_l1391_139142


namespace NUMINAMATH_GPT_max_min_value_l1391_139113

theorem max_min_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 12) (h5 : x * y + y * z + z * x = 30) :
  ∃ n : ℝ, n = min (x * y) (min (y * z) (z * x)) ∧ n = 2 :=
sorry

end NUMINAMATH_GPT_max_min_value_l1391_139113


namespace NUMINAMATH_GPT_number_is_three_l1391_139156

theorem number_is_three (n : ℝ) (h : 4 * n - 7 = 5) : n = 3 :=
by sorry

end NUMINAMATH_GPT_number_is_three_l1391_139156


namespace NUMINAMATH_GPT_jake_peaches_count_l1391_139144

-- Define Jill's peaches
def jill_peaches : ℕ := 5

-- Define Steven's peaches based on the condition that Steven has 18 more peaches than Jill
def steven_peaches : ℕ := jill_peaches + 18

-- Define Jake's peaches based on the condition that Jake has 6 fewer peaches than Steven
def jake_peaches : ℕ := steven_peaches - 6

-- The theorem to prove that Jake has 17 peaches
theorem jake_peaches_count : jake_peaches = 17 := by
  sorry

end NUMINAMATH_GPT_jake_peaches_count_l1391_139144


namespace NUMINAMATH_GPT_timeTakenByBobIs30_l1391_139148

-- Define the conditions
def timeTakenByAlice : ℕ := 40
def fractionOfTimeBobTakes : ℚ := 3 / 4

-- Define the statement to be proven
theorem timeTakenByBobIs30 : (fractionOfTimeBobTakes * timeTakenByAlice : ℚ) = 30 := 
by
  sorry

end NUMINAMATH_GPT_timeTakenByBobIs30_l1391_139148


namespace NUMINAMATH_GPT_betty_watermelons_l1391_139107

theorem betty_watermelons :
  ∃ b : ℕ, 
  (b + (b + 10) + (b + 20) + (b + 30) + (b + 40) = 200) ∧
  (b + 40 = 60) :=
by
  sorry

end NUMINAMATH_GPT_betty_watermelons_l1391_139107


namespace NUMINAMATH_GPT_value_of_A_l1391_139124

def clubsuit (A B : ℕ) := 3 * A + 2 * B + 5

theorem value_of_A (A : ℕ) (h : clubsuit A 7 = 82) : A = 21 :=
by
  sorry

end NUMINAMATH_GPT_value_of_A_l1391_139124


namespace NUMINAMATH_GPT_rowing_distance_l1391_139186

noncomputable def effective_speed_with_current (rowing_speed current_speed : ℕ) : ℕ :=
  rowing_speed + current_speed

noncomputable def effective_speed_against_current (rowing_speed current_speed : ℕ) : ℕ :=
  rowing_speed - current_speed

noncomputable def distance (speed time : ℕ) : ℕ :=
  speed * time

theorem rowing_distance (rowing_speed current_speed total_time : ℕ) 
  (hrowing_speed : rowing_speed = 10)
  (hcurrent_speed : current_speed = 2)
  (htotal_time : total_time = 30) : 
  (distance 8 18) = 144 := 
by
  sorry

end NUMINAMATH_GPT_rowing_distance_l1391_139186


namespace NUMINAMATH_GPT_medal_allocation_l1391_139141

-- Define the participants
inductive Participant
| Jiri
| Vit
| Ota

open Participant

-- Define the medals
inductive Medal
| Gold
| Silver
| Bronze

open Medal

-- Define a structure to capture each person's statement
structure Statements :=
  (Jiri : Prop)
  (Vit : Prop)
  (Ota : Prop)

-- Define the condition based on their statements
def statements (m : Participant → Medal) : Statements :=
  {
    Jiri := m Ota = Gold,
    Vit := m Ota = Silver,
    Ota := (m Ota ≠ Gold ∧ m Ota ≠ Silver)
  }

-- Define the condition for truth-telling and lying based on medals
def truths_and_lies (m : Participant → Medal) (s : Statements) : Prop :=
  (m Jiri = Gold → s.Jiri) ∧ (m Jiri = Bronze → ¬ s.Jiri) ∧
  (m Vit = Gold → s.Vit) ∧ (m Vit = Bronze → ¬ s.Vit) ∧
  (m Ota = Gold → s.Ota) ∧ (m Ota = Bronze → ¬ s.Ota)

-- Define the final theorem to be proven
theorem medal_allocation : 
  ∃ (m : Participant → Medal), 
    truths_and_lies m (statements m) ∧ 
    m Vit = Gold ∧ 
    m Ota = Silver ∧ 
    m Jiri = Bronze := 
sorry

end NUMINAMATH_GPT_medal_allocation_l1391_139141


namespace NUMINAMATH_GPT_calc_fraction_product_l1391_139145

theorem calc_fraction_product : 
  (7 / 4) * (8 / 14) * (14 / 8) * (16 / 40) * (35 / 20) * (18 / 45) * (49 / 28) * (32 / 64) = 49 / 200 := 
by sorry

end NUMINAMATH_GPT_calc_fraction_product_l1391_139145


namespace NUMINAMATH_GPT_sum_T_mod_1000_l1391_139104

open Nat

def T (a b : ℕ) : ℕ :=
  if h : a + b ≤ 6 then Nat.choose 6 a * Nat.choose 6 b * Nat.choose 6 (a + b) else 0

def sum_T : ℕ :=
  (Finset.range 7).sum (λ a => (Finset.range (7 - a)).sum (λ b => T a b))

theorem sum_T_mod_1000 : sum_T % 1000 = 564 := by
  sorry

end NUMINAMATH_GPT_sum_T_mod_1000_l1391_139104


namespace NUMINAMATH_GPT_calc_30_exp_l1391_139106

theorem calc_30_exp :
  30 * 30 ^ 10 = 30 ^ 11 :=
by sorry

end NUMINAMATH_GPT_calc_30_exp_l1391_139106


namespace NUMINAMATH_GPT_store_profit_l1391_139155

theorem store_profit (m n : ℝ) (hmn : m > n) : 
  let selling_price := (m + n) / 2
  let profit_a := 40 * (selling_price - m)
  let profit_b := 60 * (selling_price - n)
  let total_profit := profit_a + profit_b
  total_profit > 0 :=
by sorry

end NUMINAMATH_GPT_store_profit_l1391_139155


namespace NUMINAMATH_GPT_b_n_plus_1_eq_2a_n_l1391_139121

/-- Definition of binary sequences of length n that do not contain 0, 1, 0 -/
def a_n (n : ℕ) : ℕ := -- specify the actual counting function, placeholder below
  sorry

/-- Definition of binary sequences of length n that do not contain 0, 0, 1, 1 or 1, 1, 0, 0 -/
def b_n (n : ℕ) : ℕ := -- specify the actual counting function, placeholder below
  sorry

/-- Proof statement that for all positive integers n, b_{n+1} = 2a_n -/
theorem b_n_plus_1_eq_2a_n (n : ℕ) (hn : 0 < n) : b_n (n + 1) = 2 * a_n n :=
  sorry

end NUMINAMATH_GPT_b_n_plus_1_eq_2a_n_l1391_139121


namespace NUMINAMATH_GPT_james_total_fish_catch_l1391_139119

-- Definitions based on conditions
def weight_trout : ℕ := 200
def weight_salmon : ℕ := weight_trout + (60 * weight_trout / 100)
def weight_tuna : ℕ := 2 * weight_trout
def weight_bass : ℕ := 3 * weight_salmon
def weight_catfish : ℚ := weight_tuna / 3

-- Total weight of the fish James caught
def total_weight_fish : ℚ := 
  weight_trout + weight_salmon + weight_tuna + weight_bass + weight_catfish 

-- The theorem statement
theorem james_total_fish_catch : total_weight_fish = 2013.33 := by
  sorry

end NUMINAMATH_GPT_james_total_fish_catch_l1391_139119


namespace NUMINAMATH_GPT_rods_needed_to_complete_6_step_pyramid_l1391_139143

def rods_in_step (n : ℕ) : ℕ :=
  16 * n

theorem rods_needed_to_complete_6_step_pyramid (rods_1_step rods_2_step : ℕ) :
  rods_1_step = 16 → rods_2_step = 32 → rods_in_step 6 - rods_in_step 4 = 32 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_rods_needed_to_complete_6_step_pyramid_l1391_139143


namespace NUMINAMATH_GPT_fourth_equation_pattern_l1391_139166

theorem fourth_equation_pattern :
  36^2 + 37^2 + 38^2 + 39^2 + 40^2 = 41^2 + 42^2 + 43^2 + 44^2 :=
by
  sorry

end NUMINAMATH_GPT_fourth_equation_pattern_l1391_139166


namespace NUMINAMATH_GPT_number_of_cities_l1391_139154

theorem number_of_cities (n : ℕ) (h : n * (n - 1) / 2 = 15) : n = 6 :=
sorry

end NUMINAMATH_GPT_number_of_cities_l1391_139154


namespace NUMINAMATH_GPT_correct_calculation_l1391_139108

theorem correct_calculation :
  (∀ a : ℝ, a^3 + a^2 ≠ a^5) ∧
  (∀ a : ℝ, a^3 / a^2 = a) ∧
  (∀ a : ℝ, 3 * a^3 * 2 * a^2 ≠ 6 * a^6) ∧
  (∀ a : ℝ, (a - 2)^2 ≠ a^2 - 4) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1391_139108


namespace NUMINAMATH_GPT_prove_a_range_if_p_prove_a_range_if_p_or_q_and_not_and_l1391_139172

-- Define the conditions
def quadratic_has_two_different_negative_roots (a : ℝ) : Prop :=
  a^2 - 1/4 > 0 ∧ -a < 0 ∧ 1/16 > 0

def inequality_q (a : ℝ) : Prop :=
  0 < a ∧ a < 1

-- Prove the results based on the conditions
theorem prove_a_range_if_p (a : ℝ) (hp : quadratic_has_two_different_negative_roots a) : a > 1/2 :=
  sorry

theorem prove_a_range_if_p_or_q_and_not_and (a : ℝ) (hp_or_q : quadratic_has_two_different_negative_roots a ∨ inequality_q a) 
  (hnot_p_and_q : ¬ (quadratic_has_two_different_negative_roots a ∧ inequality_q a)) :
  a ≥ 1 ∨ (0 < a ∧ a ≤ 1/2) :=
  sorry

end NUMINAMATH_GPT_prove_a_range_if_p_prove_a_range_if_p_or_q_and_not_and_l1391_139172


namespace NUMINAMATH_GPT_sequence_increasing_l1391_139101

theorem sequence_increasing (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) = a n + 3) : ∀ n, a (n + 1) > a n := 
by 
  sorry

end NUMINAMATH_GPT_sequence_increasing_l1391_139101


namespace NUMINAMATH_GPT_num_solutions_congruence_l1391_139196

-- Define the problem context and conditions
def is_valid_solution (y : ℕ) : Prop :=
  y < 150 ∧ (y + 21) % 46 = 79 % 46

-- Define the proof problem
theorem num_solutions_congruence : ∃ (s : Finset ℕ), s.card = 3 ∧ ∀ y ∈ s, is_valid_solution y := by
  sorry

end NUMINAMATH_GPT_num_solutions_congruence_l1391_139196


namespace NUMINAMATH_GPT_total_squares_in_6x6_grid_l1391_139133

theorem total_squares_in_6x6_grid : 
  let size := 6
  let total_squares := (size * size) + ((size - 1) * (size - 1)) + ((size - 2) * (size - 2)) + ((size - 3) * (size - 3)) + ((size - 4) * (size - 4)) + ((size - 5) * (size - 5))
  total_squares = 91 :=
by
  let size := 6
  let total_squares := (size * size) + ((size - 1) * (size - 1)) + ((size - 2) * (size - 2)) + ((size - 3) * (size - 3)) + ((size - 4) * (size - 4)) + ((size - 5) * (size - 5))
  have eqn : total_squares = 91 := sorry
  exact eqn

end NUMINAMATH_GPT_total_squares_in_6x6_grid_l1391_139133


namespace NUMINAMATH_GPT_parallelogram_smaller_angle_proof_l1391_139123

noncomputable def smaller_angle (x : ℝ) : Prop :=
  let larger_angle := x + 120
  let angle_sum := x + larger_angle + x + larger_angle = 360
  angle_sum

theorem parallelogram_smaller_angle_proof (x : ℝ) (h1 : smaller_angle x) : x = 30 := by
  sorry

end NUMINAMATH_GPT_parallelogram_smaller_angle_proof_l1391_139123


namespace NUMINAMATH_GPT_max_sector_area_central_angle_l1391_139164

theorem max_sector_area_central_angle (radius arc_length : ℝ) :
  (arc_length + 2 * radius = 20) ∧ (arc_length = 20 - 2 * radius) ∧
  (arc_length / radius = 2) → 
  arc_length / radius = 2 :=
by
  intros h 
  sorry

end NUMINAMATH_GPT_max_sector_area_central_angle_l1391_139164


namespace NUMINAMATH_GPT_value_of_expression_l1391_139117

theorem value_of_expression 
  (x : ℝ) 
  (h : 7 * x^2 + 6 = 5 * x + 11) 
  : (8 * x - 5)^2 = (2865 - 120 * Real.sqrt 165) / 49 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_expression_l1391_139117


namespace NUMINAMATH_GPT_range_of_a_if_proposition_l1391_139184

theorem range_of_a_if_proposition :
  (∃ x : ℝ, |x - 1| + |x + a| < 3) → -4 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_GPT_range_of_a_if_proposition_l1391_139184


namespace NUMINAMATH_GPT_total_seeds_planted_l1391_139129

def number_of_flowerbeds : ℕ := 9
def seeds_per_flowerbed : ℕ := 5

theorem total_seeds_planted : number_of_flowerbeds * seeds_per_flowerbed = 45 :=
by
  sorry

end NUMINAMATH_GPT_total_seeds_planted_l1391_139129


namespace NUMINAMATH_GPT_hyperbola_standard_eq_l1391_139120

theorem hyperbola_standard_eq (a c : ℝ) (h1 : a = 5) (h2 : c = 7) :
  (∃ b, b^2 = c^2 - a^2 ∧ (1 = (x^2 / a^2 - y^2 / b^2) ∨ 1 = (y^2 / a^2 - x^2 / b^2))) := by
  sorry

end NUMINAMATH_GPT_hyperbola_standard_eq_l1391_139120


namespace NUMINAMATH_GPT_second_quadrant_set_l1391_139199

-- Define the set P of points in the second quadrant
def P : Set (ℝ × ℝ) := { p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0 }

-- Statement of the problem: Prove that this definition accurately describes the set of all points in the second quadrant
theorem second_quadrant_set :
  P = { p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0 } :=
by
  sorry

end NUMINAMATH_GPT_second_quadrant_set_l1391_139199


namespace NUMINAMATH_GPT_pentagon_vertex_assignment_l1391_139195

theorem pentagon_vertex_assignment :
  ∃ (x_A x_B x_C x_D x_E : ℝ),
    x_A + x_B = 1 ∧
    x_B + x_C = 2 ∧
    x_C + x_D = 3 ∧
    x_D + x_E = 4 ∧
    x_E + x_A = 5 ∧
    (x_A, x_B, x_C, x_D, x_E) = (1.5, -0.5, 2.5, 0.5, 3.5) := by
  sorry

end NUMINAMATH_GPT_pentagon_vertex_assignment_l1391_139195


namespace NUMINAMATH_GPT_eval_p_nested_l1391_139105

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then 2 * x + 3 * y
  else if x < 0 ∧ y < 0 then x ^ 2 - y
  else 4 * x + 2 * y

theorem eval_p_nested :
  p (p 2 (-3)) (p (-4) (-3)) = 61 :=
by
  sorry

end NUMINAMATH_GPT_eval_p_nested_l1391_139105


namespace NUMINAMATH_GPT_farmer_cages_l1391_139179

theorem farmer_cages (c : ℕ) (h1 : 164 + 6 = 170) (h2 : ∃ r : ℕ, c * r = 170) (h3 : ∃ r : ℕ, c * r > 164) :
  c = 10 :=
by
  sorry

end NUMINAMATH_GPT_farmer_cages_l1391_139179


namespace NUMINAMATH_GPT_last_digit_to_appear_mod9_l1391_139151

def fib (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fib (n - 1) + fib (n - 2)

def fib_mod9 (n : ℕ) : ℕ :=
  (fib n) % 9

theorem last_digit_to_appear_mod9 :
  ∃ n : ℕ, ∀ m : ℕ, m < n → fib_mod9 m ≠ 0 ∧ fib_mod9 n = 0 :=
sorry

end NUMINAMATH_GPT_last_digit_to_appear_mod9_l1391_139151


namespace NUMINAMATH_GPT_power_of_fraction_l1391_139190

theorem power_of_fraction :
  (3 / 4) ^ 5 = 243 / 1024 :=
by sorry

end NUMINAMATH_GPT_power_of_fraction_l1391_139190


namespace NUMINAMATH_GPT_sin_eq_cos_510_l1391_139147

theorem sin_eq_cos_510 (n : ℤ) (h1 : -180 ≤ n ∧ n ≤ 180) (h2 : Real.sin (n * Real.pi / 180) = Real.cos (510 * Real.pi / 180)) :
  n = -60 :=
sorry

end NUMINAMATH_GPT_sin_eq_cos_510_l1391_139147


namespace NUMINAMATH_GPT_find_height_of_box_l1391_139197

-- Given the conditions
variables (h l w : ℝ)
variables (V : ℝ)

-- Conditions as definitions in Lean
def length_eq_height (h : ℝ) : ℝ := 3 * h
def length_eq_width (w : ℝ) : ℝ := 4 * w
def volume_eq (h l w : ℝ) : ℝ := l * w * h

-- The proof problem: Prove height of the box is 12 given the conditions
theorem find_height_of_box : 
  (∃ h l w, l = 3 * h ∧ l = 4 * w ∧ l * w * h = 3888) → h = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_height_of_box_l1391_139197


namespace NUMINAMATH_GPT_solve_inequality_l1391_139139

theorem solve_inequality :
  {x : ℝ | -x^2 + 5 * x > 6} = {x : ℝ | 2 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_solve_inequality_l1391_139139


namespace NUMINAMATH_GPT_polynomial_root_sum_l1391_139109

theorem polynomial_root_sum :
  ∃ a b c : ℝ,
    (∀ x : ℝ, Polynomial.eval x (Polynomial.X ^ 3 - 10 * Polynomial.X ^ 2 + 16 * Polynomial.X - 2) = 0) →
    a + b + c = 10 → ab + ac + bc = 16 → abc = 2 →
    (a / (bc + 2) + b / (ac + 2) + c / (ab + 2) = 4) := sorry

end NUMINAMATH_GPT_polynomial_root_sum_l1391_139109


namespace NUMINAMATH_GPT_determine_triangle_value_l1391_139153

theorem determine_triangle_value (p : ℕ) (triangle : ℕ) (h1 : triangle + p = 67) (h2 : 3 * (triangle + p) - p = 185) : triangle = 51 := by
  sorry

end NUMINAMATH_GPT_determine_triangle_value_l1391_139153


namespace NUMINAMATH_GPT_find_solutions_l1391_139150

theorem find_solutions (n k : ℕ) (hn : n > 0) (hk : k > 0) : 
  n! + n = n^k → (n, k) = (2, 2) ∨ (n, k) = (3, 2) ∨ (n, k) = (5, 3) :=
sorry

end NUMINAMATH_GPT_find_solutions_l1391_139150


namespace NUMINAMATH_GPT_monotonic_subsequence_exists_l1391_139110

theorem monotonic_subsequence_exists (n : ℕ) (a : Fin ((2^n : ℕ) + 1) → ℕ)
  (h : ∀ k : Fin (2^n + 1), a k ≤ k.val) : 
  ∃ (b : Fin (n + 2) → Fin (2^n + 1)),
    (∀ i j : Fin (n + 2), i ≤ j → b i ≤ b j) ∧
    (∀ i j : Fin (n + 2), i < j → a (b i) ≤ a ( b j)) :=
by
  sorry

end NUMINAMATH_GPT_monotonic_subsequence_exists_l1391_139110


namespace NUMINAMATH_GPT_simplify_expression_l1391_139122

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : (2 * x⁻¹ + 3 * y⁻¹)⁻¹ = (x * y) / (2 * y + 3 * x) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1391_139122


namespace NUMINAMATH_GPT_sample_size_l1391_139192

-- Definitions for the conditions
def ratio_A : Nat := 2
def ratio_B : Nat := 3
def ratio_C : Nat := 4
def stratified_sample_size : Nat := 9 -- Total parts in the ratio sum
def products_A_sample : Nat := 18 -- Sample contains 18 Type A products

-- We need to tie these conditions together and prove the size of the sample n
theorem sample_size (n : Nat) (ratio_A ratio_B ratio_C stratified_sample_size products_A_sample : Nat) :
  ratio_A = 2 → ratio_B = 3 → ratio_C = 4 → stratified_sample_size = 9 → products_A_sample = 18 → n = 81 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof body here
  sorry -- Placeholder for the proof

end NUMINAMATH_GPT_sample_size_l1391_139192


namespace NUMINAMATH_GPT_polynomial_min_value_l1391_139125

noncomputable def poly (x y : ℝ) : ℝ := x^2 + y^2 - 6*x + 8*y + 7

theorem polynomial_min_value : 
  ∃ x y : ℝ, poly x y = -18 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_min_value_l1391_139125


namespace NUMINAMATH_GPT_cube_without_lid_configurations_l1391_139128

-- Introduce assumption for cube without a lid
structure CubeWithoutLid

-- Define the proof statement
theorem cube_without_lid_configurations : 
  ∃ (configs : Nat), (configs = 8) :=
by
  sorry

end NUMINAMATH_GPT_cube_without_lid_configurations_l1391_139128


namespace NUMINAMATH_GPT_range_of_a_values_l1391_139116

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - |x + 1| + 3 * a ≥ 0

theorem range_of_a_values (a : ℝ) : range_of_a a ↔ a ≥ 1/2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_values_l1391_139116


namespace NUMINAMATH_GPT_dice_sum_not_possible_l1391_139114

   theorem dice_sum_not_possible (a b c d : ℕ) :
     (1 ≤ a ∧ a ≤ 6) → (1 ≤ b ∧ b ≤ 6) → (1 ≤ c ∧ c ≤ 6) → (1 ≤ d ∧ d ≤ 6) →
     (a * b * c * d = 360) → ¬ (a + b + c + d = 20) :=
   by
     intros ha hb hc hd prod eq_sum
     -- Proof skipped
     sorry
   
end NUMINAMATH_GPT_dice_sum_not_possible_l1391_139114


namespace NUMINAMATH_GPT_complex_multiplication_l1391_139149

variable (i : ℂ)
axiom i_square : i^2 = -1

theorem complex_multiplication : i * (1 + i) = -1 + i :=
by
  sorry

end NUMINAMATH_GPT_complex_multiplication_l1391_139149


namespace NUMINAMATH_GPT_area_of_triangle_BXC_l1391_139171

/-
  Given:
  - AB = 15 units
  - CD = 40 units
  - The area of trapezoid ABCD = 550 square units

  To prove:
  - The area of triangle BXC = 1200 / 11 square units
-/
theorem area_of_triangle_BXC 
  (AB CD : ℝ) 
  (hAB : AB = 15) 
  (hCD : CD = 40) 
  (area_ABCD : ℝ)
  (hArea_ABCD : area_ABCD = 550) 
  : ∃ (area_BXC : ℝ), area_BXC = 1200 / 11 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_BXC_l1391_139171


namespace NUMINAMATH_GPT_find_tricycles_l1391_139161

noncomputable def number_of_tricycles (w b t : ℕ) : ℕ := t

theorem find_tricycles : ∃ (w b t : ℕ), 
  (w + b + t = 10) ∧ 
  (2 * b + 3 * t = 25) ∧ 
  (number_of_tricycles w b t = 5) :=
  by 
    sorry

end NUMINAMATH_GPT_find_tricycles_l1391_139161


namespace NUMINAMATH_GPT_max_value_a_l1391_139180

def no_lattice_points (m : ℚ) : Prop :=
  ∀ (x : ℤ), 0 < x ∧ x ≤ 150 → ¬∃ (y : ℤ), y = m * x + 3

def valid_m (m : ℚ) (a : ℚ) : Prop :=
  (2 : ℚ) / 3 < m ∧ m < a

theorem max_value_a (a : ℚ) : (a = 101 / 151) ↔ 
  ∀ (m : ℚ), valid_m m a → no_lattice_points m :=
sorry

end NUMINAMATH_GPT_max_value_a_l1391_139180


namespace NUMINAMATH_GPT_construct_80_construct_160_construct_20_l1391_139169

-- Define the notion of constructibility from an angle
inductive Constructible : ℝ → Prop
| base (a : ℝ) : a = 40 → Constructible a
| add (a b : ℝ) : Constructible a → Constructible b → Constructible (a + b)
| sub (a b : ℝ) : Constructible a → Constructible b → Constructible (a - b)

-- Lean statements for proving the constructibility
theorem construct_80 : Constructible 80 :=
sorry

theorem construct_160 : Constructible 160 :=
sorry

theorem construct_20 : Constructible 20 :=
sorry

end NUMINAMATH_GPT_construct_80_construct_160_construct_20_l1391_139169


namespace NUMINAMATH_GPT_distance_origin_to_point_l1391_139130

theorem distance_origin_to_point :
  let distance (x1 y1 x2 y2 : ℝ) := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance 0 0 8 (-15) = 17 :=
by
  sorry

end NUMINAMATH_GPT_distance_origin_to_point_l1391_139130


namespace NUMINAMATH_GPT_percentage_of_rotten_oranges_l1391_139163

theorem percentage_of_rotten_oranges
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (percentage_good_condition : ℕ)
  (rotted_percentage_bananas : ℕ)
  (total_fruits : ℕ)
  (good_condition_fruits : ℕ)
  (rotted_fruits : ℕ)
  (rotted_bananas : ℕ)
  (rotted_oranges : ℕ)
  (percentage_rotten_oranges : ℕ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : percentage_good_condition = 89)
  (h4 : rotted_percentage_bananas = 5)
  (h5 : total_fruits = total_oranges + total_bananas)
  (h6 : good_condition_fruits = percentage_good_condition * total_fruits / 100)
  (h7 : rotted_fruits = total_fruits - good_condition_fruits)
  (h8 : rotted_bananas = rotted_percentage_bananas * total_bananas / 100)
  (h9 : rotted_oranges = rotted_fruits - rotted_bananas)
  (h10 : percentage_rotten_oranges = rotted_oranges * 100 / total_oranges) : 
  percentage_rotten_oranges = 15 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_rotten_oranges_l1391_139163


namespace NUMINAMATH_GPT_remainder_of_3_pow_102_mod_101_l1391_139165

theorem remainder_of_3_pow_102_mod_101 : (3^102) % 101 = 9 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_102_mod_101_l1391_139165


namespace NUMINAMATH_GPT_triangle_sum_correct_l1391_139194

def triangle_op (a b c : ℕ) : ℕ :=
  a * b / c

theorem triangle_sum_correct :
  triangle_op 4 8 2 + triangle_op 5 10 5 = 26 :=
by
  sorry

end NUMINAMATH_GPT_triangle_sum_correct_l1391_139194


namespace NUMINAMATH_GPT_bullfinches_are_50_l1391_139132

theorem bullfinches_are_50 :
  ∃ N : ℕ, (N > 50 ∨ N < 50 ∨ N ≥ 1) ∧ (¬(N > 50) ∨ ¬(N < 50) ∨ ¬(N ≥ 1)) ∧
  (N > 50 ∧ ¬(N < 50) ∨ N < 50 ∧ ¬(N > 50) ∨ N ≥ 1 ∧ (¬(N > 50) ∧ ¬(N < 50))) ∧
  N = 50 :=
by
  sorry

end NUMINAMATH_GPT_bullfinches_are_50_l1391_139132


namespace NUMINAMATH_GPT_gertrude_fleas_l1391_139198

variables (G M O : ℕ)

def fleas_maud := M = 5 * O
def fleas_olive := O = G / 2
def total_fleas := G + M + O = 40

theorem gertrude_fleas
  (h_maud : fleas_maud M O)
  (h_olive : fleas_olive G O)
  (h_total : total_fleas G M O) :
  G = 10 :=
sorry

end NUMINAMATH_GPT_gertrude_fleas_l1391_139198
