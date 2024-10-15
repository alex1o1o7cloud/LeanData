import Mathlib

namespace NUMINAMATH_GPT_flour_needed_for_one_batch_l701_70109

theorem flour_needed_for_one_batch (F : ℝ) (h1 : 8 * F + 8 * 1.5 = 44) : F = 4 := 
by
    sorry

end NUMINAMATH_GPT_flour_needed_for_one_batch_l701_70109


namespace NUMINAMATH_GPT_scientific_notation_of_3930_billion_l701_70120

theorem scientific_notation_of_3930_billion :
  (3930 * 10^9) = 3.93 * 10^12 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_3930_billion_l701_70120


namespace NUMINAMATH_GPT_ones_digit_of_6_pow_52_l701_70163

theorem ones_digit_of_6_pow_52 : (6 ^ 52) % 10 = 6 := by
  -- we'll put the proof here
  sorry

end NUMINAMATH_GPT_ones_digit_of_6_pow_52_l701_70163


namespace NUMINAMATH_GPT_area_covered_by_three_layers_l701_70111

theorem area_covered_by_three_layers (A B C : ℕ) (total_wallpaper : ℕ := 300)
  (wall_area : ℕ := 180) (two_layer_coverage : ℕ := 30) :
  A + 2 * B + 3 * C = total_wallpaper ∧ B + C = total_wallpaper - wall_area ∧ B = two_layer_coverage → 
  C = 90 :=
by
  sorry

end NUMINAMATH_GPT_area_covered_by_three_layers_l701_70111


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l701_70161

theorem quadratic_inequality_solution (a b : ℝ)
  (h : ∀ x : ℝ, -1/2 < x ∧ x < 1/3 → ax^2 + bx + 2 > 0) :
  a + b = -14 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l701_70161


namespace NUMINAMATH_GPT_defective_percentage_is_0_05_l701_70100

-- Define the problem conditions as Lean definitions
def total_meters : ℕ := 4000
def defective_meters : ℕ := 2

-- Define the percentage calculation function
def percentage_defective (defective total : ℕ) : ℚ :=
  (defective : ℚ) / (total : ℚ) * 100

-- Rewrite the proof statement using these definitions
theorem defective_percentage_is_0_05 :
  percentage_defective defective_meters total_meters = 0.05 :=
by
  sorry

end NUMINAMATH_GPT_defective_percentage_is_0_05_l701_70100


namespace NUMINAMATH_GPT_unique_solution_l701_70183

theorem unique_solution : ∀ (x y z : ℕ), 
  x > 0 → y > 0 → z > 0 → 
  x^2 = 2 * (y + z) → 
  x^6 = y^6 + z^6 + 31 * (y^2 + z^2) → 
  (x, y, z) = (2, 1, 1) :=
by sorry

end NUMINAMATH_GPT_unique_solution_l701_70183


namespace NUMINAMATH_GPT_max_C_usage_l701_70196

-- Definition of variables (concentration percentages and weights)
def A_conc := 3 / 100
def B_conc := 8 / 100
def C_conc := 11 / 100

def target_conc := 7 / 100
def total_weight := 100

def max_A := 50
def max_B := 70
def max_C := 60

-- Equation to satisfy
def conc_equation (x y : ℝ) : Prop :=
  C_conc * x + B_conc * y + A_conc * (total_weight - x - y) = target_conc * total_weight

-- Definition with given constraints
def within_constraints (x y : ℝ) : Prop :=
  x ≤ max_C ∧ y ≤ max_B ∧ (total_weight - x - y) ≤ max_A

-- The theorem that needs to be proved
theorem max_C_usage (x y : ℝ) : within_constraints x y ∧ conc_equation x y → x ≤ 50 :=
by
  sorry

end NUMINAMATH_GPT_max_C_usage_l701_70196


namespace NUMINAMATH_GPT_mul_digits_example_l701_70122

theorem mul_digits_example (A B C D : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D)
  (h4 : B ≠ C) (h5 : B ≠ D) (h6 : C ≠ D) (h7 : C = 2) (h8 : D = 5) : A + B = 2 := by
  sorry

end NUMINAMATH_GPT_mul_digits_example_l701_70122


namespace NUMINAMATH_GPT_negation_correct_l701_70124

-- Definitions needed from the conditions:
def is_positive (m : ℝ) : Prop := m > 0
def square (m : ℝ) : ℝ := m * m

-- The original proposition
def original_proposition (m : ℝ) : Prop := is_positive m → square m > 0

-- The negation of the proposition
def negated_proposition (m : ℝ) : Prop := ¬is_positive m → ¬(square m > 0)

-- The theorem to prove that the negated proposition is the negation of the original proposition
theorem negation_correct (m : ℝ) : (original_proposition m) ↔ (negated_proposition m) :=
by
  sorry

end NUMINAMATH_GPT_negation_correct_l701_70124


namespace NUMINAMATH_GPT_find_a_l701_70147

noncomputable def f (x : ℝ) : ℝ := x + 1 + Real.log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

theorem find_a (a : ℝ) : 
  (∃ x : ℝ, g a x = 2 * x) ∧ (deriv f 1 = 2) ∧ f 1 = 2 → a = 4 :=
by
  -- Math proof goes here
  sorry

end NUMINAMATH_GPT_find_a_l701_70147


namespace NUMINAMATH_GPT_part1_part2_l701_70123

-- Defining the function f(x) and the given conditions
def f (x a : ℝ) := x^2 - a * x + 2 * a - 2

-- Given conditions
variables (a : ℝ)
axiom f_condition : ∀ (x : ℝ), f (2 + x) a * f (2 - x) a = 4
axiom a_gt_0 : a > 0
axiom fx_bounds : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 4 → 1 ≤ f x a ∧ f x a ≤ 3

-- To prove (part 1)
theorem part1 (h : f 2 a + f 3 a = 6) : a = 2 := sorry

-- To prove (part 2)
theorem part2 : (4 - (2 * Real.sqrt 6) / 3) ≤ a ∧ a ≤ 5 / 2 := sorry

end NUMINAMATH_GPT_part1_part2_l701_70123


namespace NUMINAMATH_GPT_johns_donation_l701_70160

theorem johns_donation (A : ℝ) (T : ℝ) (J : ℝ) (h1 : A + 0.5 * A = 75) (h2 : T = 3 * A) 
                       (h3 : (T + J) / 4 = 75) : J = 150 := by
  sorry

end NUMINAMATH_GPT_johns_donation_l701_70160


namespace NUMINAMATH_GPT_probability_of_one_defective_l701_70128

theorem probability_of_one_defective :
  (2 : ℕ) ≤ 5 → (0 : ℕ) ≤ 2 → (0 : ℕ) ≤ 3 →
  let total_outcomes := Nat.choose 5 2
  let favorable_outcomes := Nat.choose 3 1 * Nat.choose 2 1
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = (3 / 5 : ℚ) :=
by
  intros h1 h2 h3
  let total_outcomes := Nat.choose 5 2
  let favorable_outcomes := Nat.choose 3 1 * Nat.choose 2 1
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  have : total_outcomes = 10 := by sorry
  have : favorable_outcomes = 6 := by sorry
  have : probability = (6 / 10 : ℚ) := by sorry
  have : (6 / 10 : ℚ) = (3 / 5 : ℚ) := by sorry
  exact this

end NUMINAMATH_GPT_probability_of_one_defective_l701_70128


namespace NUMINAMATH_GPT_carpet_area_proof_l701_70179

noncomputable def carpet_area (main_room_length_ft : ℕ) (main_room_width_ft : ℕ)
  (corridor_length_ft : ℕ) (corridor_width_ft : ℕ) (feet_per_yard : ℕ) : ℚ :=
  let main_room_length_yd := main_room_length_ft / feet_per_yard
  let main_room_width_yd := main_room_width_ft / feet_per_yard
  let corridor_length_yd := corridor_length_ft / feet_per_yard
  let corridor_width_yd := corridor_width_ft / feet_per_yard
  let main_room_area_yd2 := main_room_length_yd * main_room_width_yd
  let corridor_area_yd2 := corridor_length_yd * corridor_width_yd
  main_room_area_yd2 + corridor_area_yd2

theorem carpet_area_proof : carpet_area 15 12 10 3 3 = 23.33 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_carpet_area_proof_l701_70179


namespace NUMINAMATH_GPT_emma_age_proof_l701_70148

theorem emma_age_proof (Inez Zack Jose Emma : ℕ)
  (hJose : Jose = 20)
  (hZack : Zack = Jose + 4)
  (hInez : Inez = Zack - 12)
  (hEmma : Emma = Jose + 5) :
  Emma = 25 :=
by
  sorry

end NUMINAMATH_GPT_emma_age_proof_l701_70148


namespace NUMINAMATH_GPT_bus_children_l701_70167

theorem bus_children (X : ℕ) (initial_children : ℕ) (got_on : ℕ) (total_children_after : ℕ) 
  (h1 : initial_children = 28) 
  (h2 : got_on = 82) 
  (h3 : total_children_after = 30) 
  (h4 : initial_children + got_on - X = total_children_after) : 
  got_on - X = 2 :=
by 
  -- h1, h2, h3, and h4 are conditions from the problem
  sorry

end NUMINAMATH_GPT_bus_children_l701_70167


namespace NUMINAMATH_GPT_find_value_l701_70192

theorem find_value (a b : ℝ) (h : a^2 + b^2 - 2 * a + 6 * b + 10 = 0) : 2 * a^100 - 3 * b⁻¹ = 3 :=
by sorry

end NUMINAMATH_GPT_find_value_l701_70192


namespace NUMINAMATH_GPT_election_threshold_l701_70199

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

end NUMINAMATH_GPT_election_threshold_l701_70199


namespace NUMINAMATH_GPT_files_deleted_l701_70185

theorem files_deleted 
  (orig_files : ℕ) (final_files : ℕ) (deleted_files : ℕ) 
  (h_orig : orig_files = 24) 
  (h_final : final_files = 21) : 
  deleted_files = orig_files - final_files :=
by
  rw [h_orig, h_final]
  sorry

end NUMINAMATH_GPT_files_deleted_l701_70185


namespace NUMINAMATH_GPT_sum_of_faces_edges_vertices_rectangular_prism_l701_70127

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end NUMINAMATH_GPT_sum_of_faces_edges_vertices_rectangular_prism_l701_70127


namespace NUMINAMATH_GPT_jane_buys_4_bagels_l701_70146

theorem jane_buys_4_bagels (b m : ℕ) (h1 : b + m = 7) (h2 : (80 * b + 60 * m) % 100 = 0) : b = 4 := 
by sorry

end NUMINAMATH_GPT_jane_buys_4_bagels_l701_70146


namespace NUMINAMATH_GPT_students_like_both_l701_70166

theorem students_like_both {total students_apple_pie students_chocolate_cake students_none students_at_least_one students_both : ℕ} 
  (h_total : total = 50)
  (h_apple : students_apple_pie = 22)
  (h_chocolate : students_chocolate_cake = 20)
  (h_none : students_none = 17)
  (h_least_one : students_at_least_one = total - students_none)
  (h_union : students_at_least_one = students_apple_pie + students_chocolate_cake - students_both) :
  students_both = 9 :=
by
  sorry

end NUMINAMATH_GPT_students_like_both_l701_70166


namespace NUMINAMATH_GPT_maximize_angle_distance_l701_70102

noncomputable def f (x : ℝ) : ℝ :=
  40 * x / (x * x + 500)

theorem maximize_angle_distance :
  ∃ x : ℝ, x = 10 * Real.sqrt 5 ∧ ∀ y : ℝ, y ≠ x → f y < f x :=
sorry

end NUMINAMATH_GPT_maximize_angle_distance_l701_70102


namespace NUMINAMATH_GPT_broken_pieces_correct_l701_70114

variable (pieces_transported : ℕ)
variable (shipping_cost_per_piece : ℝ)
variable (compensation_per_broken_piece : ℝ)
variable (total_profit : ℝ)
variable (broken_pieces : ℕ)

def logistics_profit (pieces_transported : ℕ) (shipping_cost_per_piece : ℝ) 
                     (compensation_per_broken_piece : ℝ) (broken_pieces : ℕ) : ℝ :=
  shipping_cost_per_piece * (pieces_transported - broken_pieces) - compensation_per_broken_piece * broken_pieces

theorem broken_pieces_correct :
  pieces_transported = 2000 →
  shipping_cost_per_piece = 0.2 →
  compensation_per_broken_piece = 2.3 →
  total_profit = 390 →
  logistics_profit pieces_transported shipping_cost_per_piece compensation_per_broken_piece broken_pieces = total_profit →
  broken_pieces = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_broken_pieces_correct_l701_70114


namespace NUMINAMATH_GPT_problem_statement_l701_70153

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
def T : Set ℝ := {x | x < 2}

def set_otimes (A B : Set ℝ) : Set ℝ := {x | x ∈ (A ∪ B) ∧ x ∉ (A ∩ B)}

theorem problem_statement : set_otimes M T = {x | x < -1 ∨ (2 ≤ x ∧ x ≤ 4)} :=
by sorry

end NUMINAMATH_GPT_problem_statement_l701_70153


namespace NUMINAMATH_GPT_root_in_interval_2_3_l701_70165

noncomputable def f (x : ℝ) : ℝ := -|x - 5| + 2^(x - 1)

theorem root_in_interval_2_3 :
  (f 2) * (f 3) < 0 → ∃ c, 2 < c ∧ c < 3 ∧ f c = 0 := by sorry

end NUMINAMATH_GPT_root_in_interval_2_3_l701_70165


namespace NUMINAMATH_GPT_max_squares_covered_l701_70137

theorem max_squares_covered 
    (board_square_side : ℝ) 
    (card_side : ℝ) 
    (n : ℕ) 
    (h1 : board_square_side = 1) 
    (h2 : card_side = 2) 
    (h3 : ∀ x y : ℝ, (x*x + y*y ≤ card_side*card_side) → card_side*card_side ≤ 4) :
    n ≤ 9 := sorry

end NUMINAMATH_GPT_max_squares_covered_l701_70137


namespace NUMINAMATH_GPT_Robinson_age_l701_70171

theorem Robinson_age (R : ℕ)
    (brother : ℕ := R + 2)
    (sister : ℕ := R + 6)
    (mother : ℕ := R + 20)
    (avg_age_yesterday : ℕ := 39)
    (total_age_yesterday : ℕ := 156)
    (eq : (R - 1) + (brother - 1) + (sister - 1) + (mother - 1) = total_age_yesterday) :
  R = 33 :=
by
  sorry

end NUMINAMATH_GPT_Robinson_age_l701_70171


namespace NUMINAMATH_GPT_range_of_m_l701_70182

theorem range_of_m {m : ℝ} : 
  (¬ ∃ x : ℝ, (1 / 2 ≤ x ∧ x ≤ 2 ∧ x^2 - 2 * x - m ≤ 0)) → m < -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l701_70182


namespace NUMINAMATH_GPT_part_a_possible_final_number_l701_70138

theorem part_a_possible_final_number :
  ∃ (n : ℕ), n = 97 ∧ 
  (∃ f : {x // x ≠ 0} → ℕ → ℕ, 
    f ⟨1, by decide⟩ 0 = 1 ∧ 
    f ⟨2, by decide⟩ 1 = 2 ∧ 
    f ⟨4, by decide⟩ 2 = 4 ∧ 
    f ⟨8, by decide⟩ 3 = 8 ∧ 
    f ⟨16, by decide⟩ 4 = 16 ∧ 
    f ⟨32, by decide⟩ 5 = 32 ∧ 
    f ⟨64, by decide⟩ 6 = 64 ∧ 
    f ⟨128, by decide⟩ 7 = 128 ∧ 
    ∀ i j : {x // x ≠ 0}, f i j = (f i j - f i j)) := sorry

end NUMINAMATH_GPT_part_a_possible_final_number_l701_70138


namespace NUMINAMATH_GPT_number_of_girls_l701_70156

theorem number_of_girls (boys girls : ℕ) (h1 : boys = 337) (h2 : girls = boys + 402) : girls = 739 := by
  sorry

end NUMINAMATH_GPT_number_of_girls_l701_70156


namespace NUMINAMATH_GPT_ratio_of_sizes_l701_70195

-- Defining Anna's size
def anna_size : ℕ := 2

-- Defining Becky's size as three times Anna's size
def becky_size : ℕ := 3 * anna_size

-- Defining Ginger's size
def ginger_size : ℕ := 8

-- Defining the goal statement
theorem ratio_of_sizes : (ginger_size : ℕ) / (becky_size : ℕ) = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_sizes_l701_70195


namespace NUMINAMATH_GPT_pancakes_eaten_by_older_is_12_l701_70107

/-- Pancake problem conditions -/
def initial_pancakes : ℕ := 19
def final_pancakes : ℕ := 11
def younger_eats_per_cycle : ℕ := 1
def older_eats_per_cycle : ℕ := 3
def grandma_bakes_per_cycle : ℕ := 2
def net_reduction_per_cycle := younger_eats_per_cycle + older_eats_per_cycle - grandma_bakes_per_cycle
def total_pancakes_eaten_by_older (cycles : ℕ) := older_eats_per_cycle * cycles

/-- Calculate the cycles based on net reduction -/
def cycles : ℕ := (initial_pancakes - final_pancakes) / net_reduction_per_cycle

/-- Prove the number of pancakes the older grandchild eats is 12 based on given conditions --/
theorem pancakes_eaten_by_older_is_12 : total_pancakes_eaten_by_older cycles = 12 := by
  sorry

end NUMINAMATH_GPT_pancakes_eaten_by_older_is_12_l701_70107


namespace NUMINAMATH_GPT_find_larger_number_l701_70112

variable (L S : ℕ)

theorem find_larger_number 
  (h1 : L - S = 1355) 
  (h2 : L = 6 * S + 15) : 
  L = 1623 := 
sorry

end NUMINAMATH_GPT_find_larger_number_l701_70112


namespace NUMINAMATH_GPT_geometric_sequence_sum_l701_70151

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = a n * r)
    (h2 : r = 2) (h3 : a 1 * 2 + a 3 * 8 + a 5 * 32 = 3) :
    a 4 * 16 + a 6 * 64 + a 8 * 256 = 24 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l701_70151


namespace NUMINAMATH_GPT_find_peaches_l701_70178

theorem find_peaches (A P : ℕ) (h1 : A + P = 15) (h2 : 1000 * A + 2000 * P = 22000) : P = 7 := sorry

end NUMINAMATH_GPT_find_peaches_l701_70178


namespace NUMINAMATH_GPT_original_weight_calculation_l701_70155

-- Conditions
variable (postProcessingWeight : ℝ) (originalWeight : ℝ)
variable (lostPercentage : ℝ)

-- Problem Statement
theorem original_weight_calculation
  (h1 : postProcessingWeight = 240)
  (h2 : lostPercentage = 0.40) :
  originalWeight = 400 :=
sorry

end NUMINAMATH_GPT_original_weight_calculation_l701_70155


namespace NUMINAMATH_GPT_f_of_1789_l701_70125

-- Definitions as per conditions
def f : ℕ → ℕ := sorry -- This will be the function definition satisfying the conditions

axiom f_f_n (n : ℕ) (h : n > 0) : f (f n) = 4 * n + 9
axiom f_2_k (k : ℕ) : f (2^k) = 2^(k+1) + 3

-- Prove f(1789) = 3581 given the conditions.
theorem f_of_1789 : f 1789 = 3581 := 
sorry

end NUMINAMATH_GPT_f_of_1789_l701_70125


namespace NUMINAMATH_GPT_weight_of_fourth_dog_l701_70187

theorem weight_of_fourth_dog (y x : ℝ) : 
  (25 + 31 + 35 + x) / 4 = (25 + 31 + 35 + x + y) / 5 → 
  x = -91 - 5 * y :=
by
  sorry

end NUMINAMATH_GPT_weight_of_fourth_dog_l701_70187


namespace NUMINAMATH_GPT_bus_stop_time_l701_70180

noncomputable def time_stopped_per_hour (excl_speed incl_speed : ℕ) : ℕ :=
  60 * (excl_speed - incl_speed) / excl_speed

theorem bus_stop_time:
  time_stopped_per_hour 54 36 = 20 :=
by
  sorry

end NUMINAMATH_GPT_bus_stop_time_l701_70180


namespace NUMINAMATH_GPT_grunters_win_all_6_games_l701_70144

-- Define the probability of the Grunters winning a single game
def probability_win_single_game : ℚ := 3 / 5

-- Define the number of games
def number_of_games : ℕ := 6

-- Calculate the probability of winning all games (all games are independent)
def probability_win_all_games (p : ℚ) (n : ℕ) : ℚ := p ^ n

-- Prove that the probability of the Grunters winning all 6 games is exactly 729/15625
theorem grunters_win_all_6_games :
  probability_win_all_games probability_win_single_game number_of_games = 729 / 15625 :=
by
  sorry

end NUMINAMATH_GPT_grunters_win_all_6_games_l701_70144


namespace NUMINAMATH_GPT_cube_of_square_of_third_smallest_prime_l701_70176

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- Theorem to prove the cube of the square of the third smallest prime number
theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime^2)^3 = 15625 := by
  sorry

end NUMINAMATH_GPT_cube_of_square_of_third_smallest_prime_l701_70176


namespace NUMINAMATH_GPT_remainder_mod_7_l701_70133

theorem remainder_mod_7 : (4 * 6^24 + 3^48) % 7 = 5 := by
  sorry

end NUMINAMATH_GPT_remainder_mod_7_l701_70133


namespace NUMINAMATH_GPT_find_speed_l701_70118

noncomputable def circumference := 15 / 5280 -- miles
noncomputable def increased_speed (r : ℝ) := r + 5 -- miles per hour
noncomputable def reduced_time (t : ℝ) := t - 1 / 10800 -- hours
noncomputable def original_distance (r t : ℝ) := r * t
noncomputable def new_distance (r t : ℝ) := increased_speed r * reduced_time t

theorem find_speed (r t : ℝ) (h1 : original_distance r t = circumference) 
(h2 : new_distance r t = circumference) : r = 13.5 := by
  sorry

end NUMINAMATH_GPT_find_speed_l701_70118


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l701_70198

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (k - 1 ≠ 0 ∧ 8 - 4 * k > 0) ↔ (k < 2 ∧ k ≠ 1) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l701_70198


namespace NUMINAMATH_GPT_union_A_B_l701_70115

def A : Set ℕ := {2, 3}
def B : Set ℕ := {1, 2}

theorem union_A_B : A ∪ B = {1, 2, 3} := 
by
  sorry

end NUMINAMATH_GPT_union_A_B_l701_70115


namespace NUMINAMATH_GPT_expected_value_of_difference_l701_70168

noncomputable def expected_difference (num_days : ℕ) : ℝ :=
  let p_prime := 3 / 4
  let p_composite := 1 / 4
  let p_no_reroll := 2 / 3
  let expected_unsweetened_days := p_prime * p_no_reroll * num_days
  let expected_sweetened_days := p_composite * p_no_reroll * num_days
  expected_unsweetened_days - expected_sweetened_days

theorem expected_value_of_difference :
  expected_difference 365 = 121.667 := by
  sorry

end NUMINAMATH_GPT_expected_value_of_difference_l701_70168


namespace NUMINAMATH_GPT_problem_l701_70197

noncomputable def a := Real.log 2 / Real.log 3
noncomputable def b := Real.log (1/8) / Real.log 2
noncomputable def c := Real.sqrt 2

theorem problem : c > a ∧ a > b := 
by
  sorry

end NUMINAMATH_GPT_problem_l701_70197


namespace NUMINAMATH_GPT_maximize_GDP_investment_l701_70169

def invest_A_B_max_GDP : Prop :=
  ∃ (A B : ℝ), 
  A + B ≤ 30 ∧
  20000 * A + 40000 * B ≤ 1000000 ∧
  24 * A + 32 * B ≥ 800 ∧
  A = 20 ∧ B = 10

theorem maximize_GDP_investment : invest_A_B_max_GDP :=
by
  sorry

end NUMINAMATH_GPT_maximize_GDP_investment_l701_70169


namespace NUMINAMATH_GPT_snowflake_stamps_count_l701_70149

theorem snowflake_stamps_count (S : ℕ) (truck_stamps : ℕ) (rose_stamps : ℕ) :
  truck_stamps = S + 9 →
  rose_stamps = S + 9 - 13 →
  S + truck_stamps + rose_stamps = 38 →
  S = 11 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_snowflake_stamps_count_l701_70149


namespace NUMINAMATH_GPT_beanie_babies_total_l701_70139

theorem beanie_babies_total
  (Lori_beanie_babies : ℕ) (Sydney_beanie_babies : ℕ)
  (h1 : Lori_beanie_babies = 15 * Sydney_beanie_babies)
  (h2 : Lori_beanie_babies = 300) :
  Lori_beanie_babies + Sydney_beanie_babies = 320 :=
sorry

end NUMINAMATH_GPT_beanie_babies_total_l701_70139


namespace NUMINAMATH_GPT_y_intercept_of_line_l701_70119

theorem y_intercept_of_line :
  ∃ y : ℝ, (∃ x : ℝ, x = 0 ∧ 2 * x - 3 * y = 6) ∧ y = -2 :=
sorry

end NUMINAMATH_GPT_y_intercept_of_line_l701_70119


namespace NUMINAMATH_GPT_number_of_initial_cans_l701_70141

theorem number_of_initial_cans (n : ℕ) (T : ℝ)
  (h1 : T = n * 36.5)
  (h2 : T - (2 * 49.5) = (n - 2) * 30) :
  n = 6 :=
sorry

end NUMINAMATH_GPT_number_of_initial_cans_l701_70141


namespace NUMINAMATH_GPT_largest_value_of_m_exists_l701_70105

theorem largest_value_of_m_exists (a b c : ℝ) (h₁ : a + b + c = 12) (h₂ : a * b + b * c + c * a = 30) (h₃ : 0 < a) (h₄ : 0 < b) (h₅ : 0 < c) : 
  ∃ m : ℝ, (m = min (a * b) (min (b * c) (c * a))) ∧ (m = 2) := sorry

end NUMINAMATH_GPT_largest_value_of_m_exists_l701_70105


namespace NUMINAMATH_GPT_parabola_relationship_l701_70172

theorem parabola_relationship (a : ℝ) (h : a < 0) :
  let y1 := 24 * a - 7
  let y2 := -7
  let y3 := 3 * a - 7
  y1 < y3 ∧ y3 < y2 :=
by
  let y1 := 24 * a - 7
  let y2 := -7
  let y3 := 3 * a - 7
  sorry

end NUMINAMATH_GPT_parabola_relationship_l701_70172


namespace NUMINAMATH_GPT_proposition_incorrect_l701_70174

theorem proposition_incorrect :
  ¬(∀ x : ℝ, x^2 + 3 * x + 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_proposition_incorrect_l701_70174


namespace NUMINAMATH_GPT_bulbs_in_bathroom_and_kitchen_l701_70142

theorem bulbs_in_bathroom_and_kitchen
  (bedroom_bulbs : Nat)
  (basement_bulbs : Nat)
  (garage_bulbs : Nat)
  (bulbs_per_pack : Nat)
  (packs_needed : Nat)
  (total_bulbs : Nat)
  (H1 : bedroom_bulbs = 2)
  (H2 : basement_bulbs = 4)
  (H3 : garage_bulbs = basement_bulbs / 2)
  (H4 : bulbs_per_pack = 2)
  (H5 : packs_needed = 6)
  (H6 : total_bulbs = packs_needed * bulbs_per_pack) :
  (total_bulbs - (bedroom_bulbs + basement_bulbs + garage_bulbs) = 4) :=
by
  sorry

end NUMINAMATH_GPT_bulbs_in_bathroom_and_kitchen_l701_70142


namespace NUMINAMATH_GPT_mutually_exclusive_pairs_l701_70181

/-- Define the events for shooting rings and drawing balls. -/
inductive ShootEvent
| hits_7th_ring : ShootEvent
| hits_8th_ring : ShootEvent

inductive PersonEvent
| at_least_one_hits : PersonEvent
| A_hits_B_does_not : PersonEvent

inductive BallEvent
| at_least_one_black : BallEvent
| both_red : BallEvent
| no_black : BallEvent
| one_red : BallEvent

/-- Define mutually exclusive events. -/
def mutually_exclusive (e1 e2 : Prop) : Prop := e1 ∧ e2 → False

/-- Prove the pairs of events that are mutually exclusive. -/
theorem mutually_exclusive_pairs :
  mutually_exclusive (ShootEvent.hits_7th_ring = ShootEvent.hits_7th_ring) (ShootEvent.hits_8th_ring = ShootEvent.hits_8th_ring) ∧
  ¬mutually_exclusive (PersonEvent.at_least_one_hits = PersonEvent.at_least_one_hits) (PersonEvent.A_hits_B_does_not = PersonEvent.A_hits_B_does_not) ∧
  mutually_exclusive (BallEvent.at_least_one_black = BallEvent.at_least_one_black) (BallEvent.both_red = BallEvent.both_red) ∧
  mutually_exclusive (BallEvent.no_black = BallEvent.no_black) (BallEvent.one_red = BallEvent.one_red) :=
by {
  sorry
}

end NUMINAMATH_GPT_mutually_exclusive_pairs_l701_70181


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l701_70154

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, m * x^2 + m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l701_70154


namespace NUMINAMATH_GPT_fixed_point_l701_70164

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1) + 2

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a (-1) = 3 :=
by
  unfold f
  sorry

end NUMINAMATH_GPT_fixed_point_l701_70164


namespace NUMINAMATH_GPT_converse_statement_l701_70189

theorem converse_statement (a : ℝ) : (a > 2018 → a > 2017) ↔ (a > 2017 → a > 2018) :=
by
  sorry

end NUMINAMATH_GPT_converse_statement_l701_70189


namespace NUMINAMATH_GPT_find_abc_value_l701_70113

open Real

/- Defining the conditions -/
variables (a b c : ℝ)
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variables (h4 : a * (b + c) = 156) (h5 : b * (c + a) = 168) (h6 : c * (a + b) = 176)

/- Prove the value of abc -/
theorem find_abc_value :
  a * b * c = 754 :=
sorry

end NUMINAMATH_GPT_find_abc_value_l701_70113


namespace NUMINAMATH_GPT_calc_pairs_count_l701_70108

theorem calc_pairs_count :
  ∃! (ab : ℤ × ℤ), (ab.1 + ab.2 = ab.1 * ab.2) :=
by
  sorry

end NUMINAMATH_GPT_calc_pairs_count_l701_70108


namespace NUMINAMATH_GPT_temp_difference_l701_70104

theorem temp_difference
  (temp_beijing : ℤ) 
  (temp_hangzhou : ℤ) 
  (h_beijing : temp_beijing = -10) 
  (h_hangzhou : temp_hangzhou = -1) : 
  temp_beijing - temp_hangzhou = -9 := 
by 
  rw [h_beijing, h_hangzhou] 
  sorry

end NUMINAMATH_GPT_temp_difference_l701_70104


namespace NUMINAMATH_GPT_subtract_largest_unit_fraction_l701_70194

theorem subtract_largest_unit_fraction
  (a b n : ℕ) (ha : a > 0) (hb : b > a) (hn : 1 ≤ b * n ∧ b * n <= a * n + b): 
  (a * n - b < a) := by
  sorry

end NUMINAMATH_GPT_subtract_largest_unit_fraction_l701_70194


namespace NUMINAMATH_GPT_convert_to_cylindrical_l701_70190

noncomputable def cylindricalCoordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arccos (x / r)
  if y / r < 0 then (r, 2 * Real.pi - θ, z) else (r, θ, z)

theorem convert_to_cylindrical :
  cylindricalCoordinates 3 (-3 * Real.sqrt 3) 4 = (6, 5 * Real.pi / 3, 4) :=
by
  sorry

end NUMINAMATH_GPT_convert_to_cylindrical_l701_70190


namespace NUMINAMATH_GPT_minimum_sum_dimensions_l701_70170

def is_product (a b c : ℕ) (v : ℕ) : Prop :=
  a * b * c = v

def sum (a b c : ℕ) : ℕ :=
  a + b + c

theorem minimum_sum_dimensions : 
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ is_product a b c 3003 ∧ sum a b c = 45 :=
by
  sorry

end NUMINAMATH_GPT_minimum_sum_dimensions_l701_70170


namespace NUMINAMATH_GPT_man_distance_from_start_l701_70131

noncomputable def distance_from_start (west_distance north_distance : ℝ) : ℝ :=
  Real.sqrt (west_distance^2 + north_distance^2)

theorem man_distance_from_start :
  distance_from_start 10 10 = Real.sqrt 200 :=
by
  sorry

end NUMINAMATH_GPT_man_distance_from_start_l701_70131


namespace NUMINAMATH_GPT_polynomial_factorization_l701_70143

variable (x y : ℝ)

theorem polynomial_factorization (m : ℝ) :
  (∃ (a b : ℝ), 6 * x^2 - 5 * x * y - 4 * y^2 - 11 * x + 22 * y + m = (3 * x - 4 * y + a) * (2 * x + y + b)) →
  m = -10 :=
sorry

end NUMINAMATH_GPT_polynomial_factorization_l701_70143


namespace NUMINAMATH_GPT_completing_the_square_solution_correct_l701_70184

theorem completing_the_square_solution_correct (x : ℝ) :
  (x^2 + 8 * x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
by
  sorry

end NUMINAMATH_GPT_completing_the_square_solution_correct_l701_70184


namespace NUMINAMATH_GPT_least_positive_integer_l701_70193
  
theorem least_positive_integer 
  (x : ℕ) (d n : ℕ) (p : ℕ) 
  (h_eq : x = 10^p * d + n) 
  (h_ratio : n = x / 17) 
  (h_cond1 : 1 ≤ d) 
  (h_cond2 : d ≤ 9)
  (h_nonzero : n > 0) : 
  x = 10625 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_l701_70193


namespace NUMINAMATH_GPT_range_of_theta_l701_70152

theorem range_of_theta (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
    (h_ineq : 3 * (Real.sin θ ^ 5 + Real.cos (2 * θ) ^ 5) > 5 * (Real.sin θ ^ 3 + Real.cos (2 * θ) ^ 3)) :
    θ ∈ Set.Ico (7 * Real.pi / 6) (11 * Real.pi / 6) :=
sorry

end NUMINAMATH_GPT_range_of_theta_l701_70152


namespace NUMINAMATH_GPT_parallel_lines_parallel_lines_solution_l701_70140

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) → a = -1 ∨ a = 2 :=
sorry

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) ∧ 
  ((a = -1 → ∀ x y : ℝ, a * x + 2 * y + 6 = 0 ∧ (x + (a - 1) * y + (a^2 - 1) = 0)) ∨ 
  (a = 2 → ∀ x y : ℝ, a * x + 2 * y + 6 = 0 ∧ (x + (a - 1) * y + (a^2 - 1) = 0))) :=
sorry

end NUMINAMATH_GPT_parallel_lines_parallel_lines_solution_l701_70140


namespace NUMINAMATH_GPT_range_of_a_l701_70150

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) → 0 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l701_70150


namespace NUMINAMATH_GPT_mean_score_is_76_l701_70177

noncomputable def mean_stddev_problem := 
  ∃ (M SD : ℝ), (M - 2 * SD = 60) ∧ (M + 3 * SD = 100) ∧ (M = 76)

theorem mean_score_is_76 : mean_stddev_problem :=
sorry

end NUMINAMATH_GPT_mean_score_is_76_l701_70177


namespace NUMINAMATH_GPT_vlad_taller_than_sister_l701_70162

def height_vlad_meters : ℝ := 1.905
def height_sister_cm : ℝ := 86.36

theorem vlad_taller_than_sister :
  (height_vlad_meters * 100 - height_sister_cm = 104.14) :=
by 
  sorry

end NUMINAMATH_GPT_vlad_taller_than_sister_l701_70162


namespace NUMINAMATH_GPT_probability_no_coinciding_sides_l701_70132

theorem probability_no_coinciding_sides :
  let total_triangles := Nat.choose 10 3
  let unfavorable_outcomes := 60 + 10
  let favorable_outcomes := total_triangles - unfavorable_outcomes
  favorable_outcomes / total_triangles = 5 / 12 := by
  sorry

end NUMINAMATH_GPT_probability_no_coinciding_sides_l701_70132


namespace NUMINAMATH_GPT_crayons_given_to_friends_l701_70129

def initial_crayons : ℕ := 440
def lost_crayons : ℕ := 106
def remaining_crayons : ℕ := 223

theorem crayons_given_to_friends :
  initial_crayons - remaining_crayons - lost_crayons = 111 := 
by
  sorry

end NUMINAMATH_GPT_crayons_given_to_friends_l701_70129


namespace NUMINAMATH_GPT_percentage_earth_fresh_water_l701_70110

theorem percentage_earth_fresh_water :
  let portion_land := 3 / 10
  let portion_water := 1 - portion_land
  let percent_salt_water := 97 / 100
  let percent_fresh_water := 1 - percent_salt_water
  100 * (portion_water * percent_fresh_water) = 2.1 :=
by
  sorry

end NUMINAMATH_GPT_percentage_earth_fresh_water_l701_70110


namespace NUMINAMATH_GPT_distance_between_points_l701_70117

noncomputable def distance (A B : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := A
  let (x2, y2, z2) := B
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

theorem distance_between_points :
  distance (-3, 4, 0) (2, -1, 6) = Real.sqrt 86 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_points_l701_70117


namespace NUMINAMATH_GPT_value_of_g_at_2_l701_70191

def g (x : ℝ) : ℝ := x^2 - 4

theorem value_of_g_at_2 : g 2 = 0 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_value_of_g_at_2_l701_70191


namespace NUMINAMATH_GPT_compare_neg_fractions_l701_70121

theorem compare_neg_fractions : (-3 / 4) > (-5 / 6) :=
sorry

end NUMINAMATH_GPT_compare_neg_fractions_l701_70121


namespace NUMINAMATH_GPT_geom_inequality_l701_70126

noncomputable def geom_seq_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then n * a1 else a1 * (1 - q^n) / (1 - q)

theorem geom_inequality (a1 q : ℝ) (h_q : q ≠ 0) :
  (a1 * (a1 * q^2)) > 0 :=
by
  sorry

end NUMINAMATH_GPT_geom_inequality_l701_70126


namespace NUMINAMATH_GPT_mul_large_numbers_l701_70101

theorem mul_large_numbers : 300000 * 300000 * 3 = 270000000000 := by
  sorry

end NUMINAMATH_GPT_mul_large_numbers_l701_70101


namespace NUMINAMATH_GPT_original_curve_equation_l701_70134

theorem original_curve_equation (x y : ℝ) (θ : ℝ) (hθ : θ = π / 4)
  (h : (∃ P : ℝ × ℝ, P = (x, y) ∧ (∃ P' : ℝ × ℝ, P' = (x * Real.cos θ - y * Real.sin θ, x * Real.sin θ + y * Real.cos θ) ∧ ((P'.fst)^2 - (P'.snd)^2 = 2)))) :
  x * y = -1 :=
sorry

end NUMINAMATH_GPT_original_curve_equation_l701_70134


namespace NUMINAMATH_GPT_culture_growth_l701_70145

/-- Define the initial conditions and growth rates of the bacterial culture -/
def initial_cells : ℕ := 5

def growth_rate1 : ℕ := 3
def growth_rate2 : ℕ := 2

def cycle_duration : ℕ := 3
def first_phase_duration : ℕ := 6
def second_phase_duration : ℕ := 6

def total_duration : ℕ := 12

/-- Define the hypothesis that calculates the number of cells at any point in time based on the given rules -/
theorem culture_growth : 
    (initial_cells * growth_rate1^ (first_phase_duration / cycle_duration) 
    * growth_rate2^ (second_phase_duration / cycle_duration)) = 180 := 
sorry

end NUMINAMATH_GPT_culture_growth_l701_70145


namespace NUMINAMATH_GPT_maximum_a_value_condition_l701_70188

theorem maximum_a_value_condition (x a : ℝ) :
  (∀ x, (x^2 - 2 * x - 3 > 0 → x < a)) ↔ a ≤ -1 :=
by sorry

end NUMINAMATH_GPT_maximum_a_value_condition_l701_70188


namespace NUMINAMATH_GPT_problem_l701_70173

variable {a b c : ℝ}

theorem problem (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + 4 / b ≤ -4 ∨ b + 4 / c ≤ -4 ∨ c + 4 / a ≤ -4 := 
sorry

end NUMINAMATH_GPT_problem_l701_70173


namespace NUMINAMATH_GPT_quad_function_one_zero_l701_70106

theorem quad_function_one_zero (m : ℝ) :
  (∃ x : ℝ, m * x^2 - 6 * x + 1 = 0 ∧ (∀ x1 x2 : ℝ, m * x1^2 - 6 * x1 + 1 = 0 ∧ m * x2^2 - 6 * x2 + 1 = 0 → x1 = x2)) ↔ (m = 0 ∨ m = 9) :=
by
  sorry

end NUMINAMATH_GPT_quad_function_one_zero_l701_70106


namespace NUMINAMATH_GPT_expression_independent_of_a_l701_70136

theorem expression_independent_of_a (a : ℝ) :
  7 + a - (8 * a - (a + 5 - (4 - 6 * a))) = 8 :=
by sorry

end NUMINAMATH_GPT_expression_independent_of_a_l701_70136


namespace NUMINAMATH_GPT_integral_of_2x_minus_1_over_x_sq_l701_70186

theorem integral_of_2x_minus_1_over_x_sq:
  ∫ x in (1 : ℝ)..3, (2 * x - (1 / x^2)) = 26 / 3 := by
  sorry

end NUMINAMATH_GPT_integral_of_2x_minus_1_over_x_sq_l701_70186


namespace NUMINAMATH_GPT_intersection_M_N_l701_70130

def M (x : ℝ) : Prop := abs (x - 1) ≥ 2

def N (x : ℝ) : Prop := x^2 - 4 * x ≥ 0

def P (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 4

theorem intersection_M_N (x : ℝ) : (M x ∧ N x) → P x :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l701_70130


namespace NUMINAMATH_GPT_percent_students_at_trip_l701_70159

variable (total_students : ℕ)
variable (students_taking_more_than_100 : ℕ := (14 * total_students) / 100)
variable (students_not_taking_more_than_100 : ℕ := (75 * total_students) / 100)
variable (students_who_went_to_trip := (students_taking_more_than_100 * 100) / 25)

/--
  If 14 percent of the students at a school went to a camping trip and took more than $100,
  and 75 percent of the students who went to the camping trip did not take more than $100,
  then 56 percent of the students at the school went to the camping trip.
-/
theorem percent_students_at_trip :
    (students_who_went_to_trip * 100) / total_students = 56 :=
sorry

end NUMINAMATH_GPT_percent_students_at_trip_l701_70159


namespace NUMINAMATH_GPT_marts_income_percentage_l701_70158

variable (J T M : ℝ)

theorem marts_income_percentage (h1 : M = 1.40 * T) (h2 : T = 0.60 * J) : M = 0.84 * J :=
by
  sorry

end NUMINAMATH_GPT_marts_income_percentage_l701_70158


namespace NUMINAMATH_GPT_two_digit_factors_of_2_pow_18_minus_1_l701_70135

-- Define the main problem statement: 
-- How many two-digit factors does 2^18 - 1 have?

theorem two_digit_factors_of_2_pow_18_minus_1 : 
  ∃ n : ℕ, n = 5 ∧ ∀ f : ℕ, (f ∣ (2^18 - 1) ∧ 10 ≤ f ∧ f < 100) ↔ (f ∣ (2^18 - 1) ∧ 10 ≤ f ∧ f < 100 ∧ ∃ k : ℕ, (2^18 - 1) = k * f) :=
by sorry

end NUMINAMATH_GPT_two_digit_factors_of_2_pow_18_minus_1_l701_70135


namespace NUMINAMATH_GPT_jerry_feathers_count_l701_70175

noncomputable def hawk_feathers : ℕ := 6
noncomputable def eagle_feathers : ℕ := 17 * hawk_feathers
noncomputable def total_feathers : ℕ := hawk_feathers + eagle_feathers
noncomputable def remaining_feathers_after_sister : ℕ := total_feathers - 10
noncomputable def jerry_feathers_left : ℕ := remaining_feathers_after_sister / 2

theorem jerry_feathers_count : jerry_feathers_left = 49 :=
  by
  sorry

end NUMINAMATH_GPT_jerry_feathers_count_l701_70175


namespace NUMINAMATH_GPT_gcd_12a_18b_l701_70116

theorem gcd_12a_18b (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a.gcd b = 15) : (12 * a).gcd (18 * b) = 90 :=
by sorry

end NUMINAMATH_GPT_gcd_12a_18b_l701_70116


namespace NUMINAMATH_GPT_calculate_correct_subtraction_l701_70103

theorem calculate_correct_subtraction (x : ℤ) (h : x - 63 = 24) : x - 36 = 51 :=
by
  sorry

end NUMINAMATH_GPT_calculate_correct_subtraction_l701_70103


namespace NUMINAMATH_GPT_complex_fraction_value_l701_70157

theorem complex_fraction_value (a b : ℝ) (h : (i - 2) / (1 + i) = a + b * i) : a + b = 1 :=
by
  sorry

end NUMINAMATH_GPT_complex_fraction_value_l701_70157
