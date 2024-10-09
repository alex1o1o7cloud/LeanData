import Mathlib

namespace function_is_even_with_period_pi_div_2_l1225_122532

noncomputable def f (x : ℝ) : ℝ := (1 + Real.cos (2 * x)) * (Real.sin x) ^ 2

theorem function_is_even_with_period_pi_div_2 : 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, f (x + (π / 2)) = f x) :=
by
  sorry

end function_is_even_with_period_pi_div_2_l1225_122532


namespace gcd_20586_58768_l1225_122538

theorem gcd_20586_58768 : Int.gcd 20586 58768 = 2 := by
  sorry

end gcd_20586_58768_l1225_122538


namespace value_of_number_l1225_122597

theorem value_of_number (number y : ℝ) 
  (h1 : (number + 5) * (y - 5) = 0) 
  (h2 : ∀ n m : ℝ, (n + 5) * (m - 5) = 0 → n^2 + m^2 ≥ 25) 
  (h3 : number^2 + y^2 = 25) : number = -5 :=
sorry

end value_of_number_l1225_122597


namespace train_speed_l1225_122560

theorem train_speed (length_of_train : ℝ) (time_to_cross : ℝ) (speed_of_man_km_hr : ℝ) 
  (h_length : length_of_train = 420)
  (h_time : time_to_cross = 62.99496040316775)
  (h_man_speed : speed_of_man_km_hr = 6) :
  ∃ speed_of_train_km_hr : ℝ, speed_of_train_km_hr = 30 :=
by
  sorry

end train_speed_l1225_122560


namespace quadratic_has_solution_l1225_122594

theorem quadratic_has_solution (a b : ℝ) : ∃ x : ℝ, (a^6 - b^6) * x^2 + 2 * (a^5 - b^5) * x + (a^4 - b^4) = 0 :=
  by sorry

end quadratic_has_solution_l1225_122594


namespace students_in_hollow_square_are_160_l1225_122566

-- Define the problem conditions
def hollow_square_formation (outer_layer : ℕ) (inner_layer : ℕ) : Prop :=
  outer_layer = 52 ∧ inner_layer = 28

-- Define the total number of students in the group based on the given condition
def total_students (n : ℕ) : Prop := n = 160

-- Prove that the total number of students is 160 given the hollow square formation conditions
theorem students_in_hollow_square_are_160 : ∀ (outer_layer inner_layer : ℕ),
  hollow_square_formation outer_layer inner_layer → total_students 160 :=
by
  intros outer_layer inner_layer h
  sorry

end students_in_hollow_square_are_160_l1225_122566


namespace pastries_count_l1225_122576

def C : ℕ := 19
def P : ℕ := C + 112

theorem pastries_count : P = 131 := by
  -- P = 19 + 112
  -- P = 131
  sorry

end pastries_count_l1225_122576


namespace margaret_mean_score_l1225_122549

theorem margaret_mean_score : 
  let all_scores_sum := 832
  let cyprian_scores_count := 5
  let margaret_scores_count := 4
  let cyprian_mean_score := 92
  let cyprian_scores_sum := cyprian_scores_count * cyprian_mean_score
  (all_scores_sum - cyprian_scores_sum) / margaret_scores_count = 93 := by
  sorry

end margaret_mean_score_l1225_122549


namespace no_hot_dogs_l1225_122562

def hamburgers_initial := 9.0
def hamburgers_additional := 3.0
def hamburgers_total := 12.0

theorem no_hot_dogs (h1 : hamburgers_initial + hamburgers_additional = hamburgers_total) : 0 = 0 :=
by
  sorry

end no_hot_dogs_l1225_122562


namespace fifth_equation_l1225_122509

theorem fifth_equation
: 1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2 :=
by
  sorry

end fifth_equation_l1225_122509


namespace total_weight_kg_l1225_122588

def envelope_weight_grams : ℝ := 8.5
def num_envelopes : ℝ := 800

theorem total_weight_kg : (envelope_weight_grams * num_envelopes) / 1000 = 6.8 :=
by
  sorry

end total_weight_kg_l1225_122588


namespace perpendicular_line_directional_vector_l1225_122586

theorem perpendicular_line_directional_vector
  (l1 : ℝ → ℝ → Prop)
  (l2 : ℝ → ℝ → Prop)
  (perpendicular : ∀ x y, l1 x y ↔ l2 y (-x))
  (l2_eq : ∀ x y, l2 x y ↔ 2 * x + 5 * y = 1) :
  ∃ d1 d2, (d1, d2) = (5, -2) ∧ (d1 * 2 + d2 * 5 = 0) :=
by
  sorry

end perpendicular_line_directional_vector_l1225_122586


namespace largest_additional_license_plates_l1225_122568

theorem largest_additional_license_plates :
  let original_first_set := 5
  let original_second_set := 3
  let original_third_set := 4
  let original_total := original_first_set * original_second_set * original_third_set

  let new_set_case1 := original_first_set * (original_second_set + 2) * original_third_set
  let new_set_case2 := original_first_set * (original_second_set + 1) * (original_third_set + 1)

  let new_total := max new_set_case1 new_set_case2

  new_total - original_total = 40 :=
by
  let original_first_set := 5
  let original_second_set := 3
  let original_third_set := 4
  let original_total := original_first_set * original_second_set * original_third_set

  let new_set_case1 := original_first_set * (original_second_set + 2) * original_third_set
  let new_set_case2 := original_first_set * (original_second_set + 1) * (original_third_set + 1)

  let new_total := max new_set_case1 new_set_case2

  sorry

end largest_additional_license_plates_l1225_122568


namespace length_of_platform_l1225_122543

theorem length_of_platform (length_train speed_train time_crossing speed_train_mps distance_train_cross : ℝ)
  (h1 : length_train = 120)
  (h2 : speed_train = 60)
  (h3 : time_crossing = 20)
  (h4 : speed_train_mps = 16.67)
  (h5 : distance_train_cross = speed_train_mps * time_crossing):
  (distance_train_cross = length_train + 213.4) :=
by
  sorry

end length_of_platform_l1225_122543


namespace cone_to_prism_volume_ratio_l1225_122593

noncomputable def ratio_of_volumes (a h : ℝ) (pos_a : 0 < a) (pos_h : 0 < h) : ℝ :=
  let r := a / 2
  let V_cone := (1/3) * Real.pi * r^2 * h
  let V_prism := a * (2 * a) * h
  V_cone / V_prism

theorem cone_to_prism_volume_ratio (a h : ℝ) (pos_a : 0 < a) (pos_h : 0 < h) :
  ratio_of_volumes a h pos_a pos_h = Real.pi / 24 := by
  sorry

end cone_to_prism_volume_ratio_l1225_122593


namespace problem_statement_l1225_122522

-- Definition of the conditions
variables {a : ℝ} (h₀ : a > 0) (h₁ : a ≠ 1)

-- The Lean 4 statement for the problem
theorem problem_statement (h : 0 < a ∧ a < 1) : 
  (∀ x y : ℝ, x < y → a^x > a^y) → 
  (∀ x : ℝ, (2 - a) * x^3 > 0) ∧ 
  (∀ x : ℝ, (2 - a) * x^3 > 0 → 0 < a ∧ a < 2 ∧ (∀ x y : ℝ, x < y → a^x > a^y) → False) :=
by
  intros
  sorry

end problem_statement_l1225_122522


namespace treaty_of_versailles_original_day_l1225_122556

-- Define the problem in Lean terms
def treatySignedDay : Nat -> Nat -> String
| 1919, 6 => "Saturday"
| _, _ => "Unknown"

-- Theorem statement
theorem treaty_of_versailles_original_day :
  treatySignedDay 1919 6 = "Saturday" :=
sorry

end treaty_of_versailles_original_day_l1225_122556


namespace seating_arrangements_l1225_122561

/-
Given:
1. There are 8 students.
2. Four different classes: (1), (2), (3), and (4).
3. Each class has 2 students.
4. There are 2 cars, Car A and Car B, each with a capacity for 4 students.
5. The two students from Class (1) (twin sisters) must ride in the same car.

Prove:
The total number of ways to seat the students such that exactly 2 students from the same class are in Car A is 24.
-/

theorem seating_arrangements : 
  ∃ (arrangements : ℕ), arrangements = 24 :=
sorry

end seating_arrangements_l1225_122561


namespace radius_for_visibility_l1225_122530

theorem radius_for_visibility (r : ℝ) (h₁ : r > 0)
  (h₂ : ∃ o : ℝ, ∀ (s : ℝ), s = 3 → o = 0):
  (∃ p : ℝ, p = 1/3) ∧ (r = 3.6) :=
sorry

end radius_for_visibility_l1225_122530


namespace arithmetic_sequence_first_term_l1225_122550

theorem arithmetic_sequence_first_term
  (a : ℕ) -- First term of the arithmetic sequence
  (d : ℕ := 3) -- Common difference, given as 3
  (n : ℕ := 20) -- Number of terms, given as 20
  (S : ℕ := 650) -- Sum of the sequence, given as 650
  (h : S = (n / 2) * (2 * a + (n - 1) * d)) : a = 4 := 
by
  sorry

end arithmetic_sequence_first_term_l1225_122550


namespace rectangle_area_l1225_122595

theorem rectangle_area (L B : ℕ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 266) : L * B = 4290 :=
by
  sorry

end rectangle_area_l1225_122595


namespace fence_remaining_l1225_122554

noncomputable def totalFence : Float := 150.0
noncomputable def ben_whitewashed : Float := 20.0

-- Remaining fence after Ben's contribution
noncomputable def remaining_after_ben : Float := totalFence - ben_whitewashed

noncomputable def billy_fraction : Float := 1.0 / 5.0
noncomputable def billy_whitewashed : Float := billy_fraction * remaining_after_ben

-- Remaining fence after Billy's contribution
noncomputable def remaining_after_billy : Float := remaining_after_ben - billy_whitewashed

noncomputable def johnny_fraction : Float := 1.0 / 3.0
noncomputable def johnny_whitewashed : Float := johnny_fraction * remaining_after_billy

-- Remaining fence after Johnny's contribution
noncomputable def remaining_after_johnny : Float := remaining_after_billy - johnny_whitewashed

noncomputable def timmy_percentage : Float := 15.0 / 100.0
noncomputable def timmy_whitewashed : Float := timmy_percentage * remaining_after_johnny

-- Remaining fence after Timmy's contribution
noncomputable def remaining_after_timmy : Float := remaining_after_johnny - timmy_whitewashed

noncomputable def alice_fraction : Float := 1.0 / 8.0
noncomputable def alice_whitewashed : Float := alice_fraction * remaining_after_timmy

-- Remaining fence after Alice's contribution
noncomputable def remaining_fence : Float := remaining_after_timmy - alice_whitewashed

theorem fence_remaining : remaining_fence = 51.56 :=
by
    -- Placeholder for actual proof
    sorry

end fence_remaining_l1225_122554


namespace find_a_l1225_122507

-- Define the function f
def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

-- Define the derivative of f
def f_prime (x a b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem find_a (a b : ℝ) (h1 : f_prime 1 a b = 0) (h2 : f 1 a b = 10) : a = 4 :=
by
  sorry

end find_a_l1225_122507


namespace num_of_winnable_players_l1225_122584

noncomputable def num_players := 2 ^ 2013

def can_win_if (x y : Nat) : Prop := x ≤ y + 3

def single_elimination_tournament (players : Nat) : Nat :=
  -- Function simulating the single elimination based on the specified can_win_if condition
  -- Assuming the given conditions and returning the number of winnable players directly
  6038

theorem num_of_winnable_players : single_elimination_tournament num_players = 6038 :=
  sorry

end num_of_winnable_players_l1225_122584


namespace ammonium_iodide_molecular_weight_l1225_122533

theorem ammonium_iodide_molecular_weight :
  let N := 14.01
  let H := 1.008
  let I := 126.90
  let NH4I_weight := (1 * N) + (4 * H) + (1 * I)
  NH4I_weight = 144.942 :=
by
  -- The proof will go here
  sorry

end ammonium_iodide_molecular_weight_l1225_122533


namespace ratio_of_spots_to_wrinkles_l1225_122545

-- Definitions
def E : ℕ := 3
def W : ℕ := 3 * E
def S : ℕ := E + W - 69

-- Theorem
theorem ratio_of_spots_to_wrinkles : S / W = 7 :=
by
  sorry

end ratio_of_spots_to_wrinkles_l1225_122545


namespace disjoint_polynomial_sets_l1225_122581

theorem disjoint_polynomial_sets (A B : ℤ) : 
  ∃ C : ℤ, ∀ x1 x2 : ℤ, x1^2 + A * x1 + B ≠ 2 * x2^2 + 2 * x2 + C :=
by
  sorry

end disjoint_polynomial_sets_l1225_122581


namespace z_in_second_quadrant_l1225_122514

open Complex

-- Given the condition
def satisfies_eqn (z : ℂ) : Prop := z * (1 - I) = 4 * I

-- Define the second quadrant condition
def in_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

theorem z_in_second_quadrant (z : ℂ) (h : satisfies_eqn z) : in_second_quadrant z :=
  sorry

end z_in_second_quadrant_l1225_122514


namespace elliot_storeroom_blocks_l1225_122557

def storeroom_volume (length: ℕ) (width: ℕ) (height: ℕ) : ℕ :=
  length * width * height

def inner_volume (length: ℕ) (width: ℕ) (height: ℕ) (thickness: ℕ) : ℕ :=
  (length - 2 * thickness) * (width - 2 * thickness) * (height - thickness)

def blocks_needed (outer_volume: ℕ) (inner_volume: ℕ) : ℕ :=
  outer_volume - inner_volume

theorem elliot_storeroom_blocks :
  let length := 15
  let width := 12
  let height := 8
  let thickness := 2
  let outer_volume := storeroom_volume length width height
  let inner_volume := inner_volume length width height thickness
  let required_blocks := blocks_needed outer_volume inner_volume
  required_blocks = 912 :=
by {
  -- Definitions and calculations as per conditions
  sorry
}

end elliot_storeroom_blocks_l1225_122557


namespace equal_probabilities_hearts_clubs_l1225_122553

/-- Define the total number of cards in a standard deck including two Jokers -/
def total_cards := 52 + 2

/-- Define the counts of specific card types -/
def num_jokers := 2
def num_spades := 13
def num_tens := 4
def num_hearts := 13
def num_clubs := 13

/-- Define the probabilities of drawing specific card types -/
def prob_joker := num_jokers / total_cards
def prob_spade := num_spades / total_cards
def prob_ten := num_tens / total_cards
def prob_heart := num_hearts / total_cards
def prob_club := num_clubs / total_cards

theorem equal_probabilities_hearts_clubs :
  prob_heart = prob_club :=
by
  sorry

end equal_probabilities_hearts_clubs_l1225_122553


namespace candle_height_problem_l1225_122539

/-- Define the height functions of the two candles. -/
def h1 (t : ℚ) : ℚ := 1 - t / 5
def h2 (t : ℚ) : ℚ := 1 - t / 4

/-- The main theorem stating the time t when the first candle is three times the height of the second candle. -/
theorem candle_height_problem : 
  (∀ t : ℚ, h1 t = 3 * h2 t) → t = (40 : ℚ) / 11 :=
by
  sorry

end candle_height_problem_l1225_122539


namespace valentine_count_initial_l1225_122521

def valentines_given : ℕ := 42
def valentines_left : ℕ := 16
def valentines_initial := valentines_given + valentines_left

theorem valentine_count_initial :
  valentines_initial = 58 :=
by
  sorry

end valentine_count_initial_l1225_122521


namespace compute_fractions_product_l1225_122565

theorem compute_fractions_product :
  (2 * (2^4 - 1) / (2 * (2^4 + 1))) *
  (2 * (3^4 - 1) / (2 * (3^4 + 1))) *
  (2 * (4^4 - 1) / (2 * (4^4 + 1))) *
  (2 * (5^4 - 1) / (2 * (5^4 + 1))) *
  (2 * (6^4 - 1) / (2 * (6^4 + 1))) *
  (2 * (7^4 - 1) / (2 * (7^4 + 1)))
  = 4400 / 135 := by
sorry

end compute_fractions_product_l1225_122565


namespace geometric_sequence_a7_l1225_122582

noncomputable def a_n (a_1 q : ℝ) (n : ℕ) := a_1 * q^(n - 1)

theorem geometric_sequence_a7 
  (a1 q : ℝ)
  (a1_neq_zero : a1 ≠ 0)
  (a9_eq_256 : a_n a1 q 9 = 256)
  (a1_a3_eq_4 : a_n a1 q 1 * a_n a1 q 3 = 4) :
  a_n a1 q 7 = 64 := 
sorry

end geometric_sequence_a7_l1225_122582


namespace wind_velocity_l1225_122512

theorem wind_velocity (P A V : ℝ) (k : ℝ := 1/200) :
  (P = k * A * V^2) →
  (P = 2) → (A = 1) → (V = 20) →
  ∀ (P' A' : ℝ), P' = 128 → A' = 4 → ∃ V' : ℝ, V'^2 = 6400 :=
by
  intros h1 h2 h3 h4 P' A' h5 h6
  use 80
  linarith

end wind_velocity_l1225_122512


namespace perpendicular_lines_a_value_l1225_122567

theorem perpendicular_lines_a_value :
  ∀ (a : ℝ), (∀ x y : ℝ, 2 * x - y = 0) -> (∀ x y : ℝ, a * x - 2 * y - 1 = 0) ->    
  (∀ m1 m2 : ℝ, m1 = 2 -> m2 = a / 2 -> m1 * m2 = -1) -> a = -1 :=
sorry

end perpendicular_lines_a_value_l1225_122567


namespace find_rosy_age_l1225_122525

-- Definitions and conditions
def rosy_current_age (R : ℕ) : Prop :=
  ∃ D : ℕ,
    (D = R + 18) ∧ -- David is 18 years older than Rosy
    (D + 6 = 2 * (R + 6)) -- In 6 years, David will be twice as old as Rosy

-- Proof statement: Rosy's current age is 12
theorem find_rosy_age : rosy_current_age 12 :=
  sorry

end find_rosy_age_l1225_122525


namespace quadratic_root_product_l1225_122518

theorem quadratic_root_product (a b : ℝ) (m p r : ℝ)
  (h1 : a * b = 3)
  (h2 : ∀ x, x^2 - mx + 3 = 0 → x = a ∨ x = b)
  (h3 : ∀ x, x^2 - px + r = 0 → x = a + 2 / b ∨ x = b + 2 / a) :
  r = 25 / 3 := by
  sorry

end quadratic_root_product_l1225_122518


namespace mean_weight_participants_l1225_122596

def weights_120s := [123, 125]
def weights_130s := [130, 132, 133, 135, 137, 138]
def weights_140s := [141, 145, 145, 149, 149]
def weights_150s := [150, 152, 153, 155, 158]
def weights_160s := [164, 167, 167, 169]

def total_weights := weights_120s ++ weights_130s ++ weights_140s ++ weights_150s ++ weights_160s

def total_sum : ℕ := total_weights.sum
def total_count : ℕ := total_weights.length

theorem mean_weight_participants : (total_sum : ℚ) / total_count = 3217 / 22 := by
  sorry -- Proof goes here, but we're skipping it

end mean_weight_participants_l1225_122596


namespace solve_cubic_equation_l1225_122599

theorem solve_cubic_equation (x y z : ℤ) (h : x^3 - 3*y^3 - 9*z^3 = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end solve_cubic_equation_l1225_122599


namespace sweets_distribution_l1225_122515

theorem sweets_distribution (S X : ℕ) (h1 : S = 112 * X) (h2 : S = 80 * (X + 6)) :
  X = 15 := 
by
  sorry

end sweets_distribution_l1225_122515


namespace daniel_age_is_correct_l1225_122528

open Nat

-- Define Uncle Ben's age
def uncleBenAge : ℕ := 50

-- Define Edward's age as two-thirds of Uncle Ben's age
def edwardAge : ℚ := (2 / 3) * uncleBenAge

-- Define that Daniel is 7 years younger than Edward
def danielAge : ℚ := edwardAge - 7

-- Assert that Daniel's age is 79/3 years old
theorem daniel_age_is_correct : danielAge = 79 / 3 := by
  sorry

end daniel_age_is_correct_l1225_122528


namespace max_length_of_third_side_of_triangle_l1225_122535

noncomputable def max_third_side_length (D E F : ℝ) (a b : ℝ) : ℝ :=
  let c_square := a^2 + b^2 - 2 * a * b * Real.cos (90 * Real.pi / 180)
  Real.sqrt c_square

theorem max_length_of_third_side_of_triangle (D E F : ℝ) (a b : ℝ) (h₁ : Real.cos (2 * D) + Real.cos (2 * E) + Real.cos (2 * F) = 1)
    (h₂ : a = 8) (h₃ : b = 15) : 
    max_third_side_length D E F a b = 17 := 
by
  sorry

end max_length_of_third_side_of_triangle_l1225_122535


namespace solve_for_x_l1225_122516

theorem solve_for_x
  (x y : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h1 : 6 * x^3 + 12 * x * y = 2 * x^4 + 3 * x^3 * y)
  (h2 : y = x^2) :
  x = (-1 + Real.sqrt 55) / 3 := 
by
  sorry

end solve_for_x_l1225_122516


namespace total_snakes_among_pet_owners_l1225_122579

theorem total_snakes_among_pet_owners :
  let owns_only_snakes := 15
  let owns_cats_and_snakes := 7
  let owns_dogs_and_snakes := 10
  let owns_birds_and_snakes := 2
  let owns_snakes_and_hamsters := 3
  let owns_cats_dogs_and_snakes := 4
  let owns_cats_snakes_and_hamsters := 2
  let owns_all_categories := 1
  owns_only_snakes + owns_cats_and_snakes + owns_dogs_and_snakes + owns_birds_and_snakes + owns_snakes_and_hamsters + owns_cats_dogs_and_snakes + owns_cats_snakes_and_hamsters + owns_all_categories = 44 :=
by
  sorry

end total_snakes_among_pet_owners_l1225_122579


namespace exists_mn_coprime_l1225_122519

theorem exists_mn_coprime (a b : ℤ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_gcd : Int.gcd a b = 1) :
  ∃ (m n : ℕ), 1 ≤ m ∧ 1 ≤ n ∧ (a^m + b^n) % (a * b) = 1 % (a * b) :=
sorry

end exists_mn_coprime_l1225_122519


namespace div_30_prime_ge_7_l1225_122505

theorem div_30_prime_ge_7 (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_7 : p ≥ 7) : 30 ∣ (p^2 - 1) := 
sorry

end div_30_prime_ge_7_l1225_122505


namespace smallest_n_l1225_122575

theorem smallest_n (n : ℕ) : 634 * n ≡ 1275 * n [MOD 30] ↔ n = 30 :=
by
  sorry

end smallest_n_l1225_122575


namespace intersection_M_N_l1225_122564

def M (x : ℝ) : Prop := x^2 ≥ x

def N (x : ℝ) (y : ℝ) : Prop := y = 3^x + 1

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | ∃ y : ℝ, N x y ∧ y > 1} = {x : ℝ | x > 1} :=
by {
  sorry
}

end intersection_M_N_l1225_122564


namespace speed_of_stream_l1225_122540

theorem speed_of_stream (c v : ℝ) (h1 : c - v = 8) (h2 : c + v = 12) : v = 2 :=
by {
  -- proof will go here
  sorry
}

end speed_of_stream_l1225_122540


namespace probability_of_4_rainy_days_out_of_6_l1225_122551

noncomputable def probability_of_rain_on_given_day : ℝ := 0.5

noncomputable def probability_of_rain_on_exactly_k_days (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

theorem probability_of_4_rainy_days_out_of_6 :
  probability_of_rain_on_exactly_k_days 6 4 probability_of_rain_on_given_day = 0.234375 :=
by
  sorry

end probability_of_4_rainy_days_out_of_6_l1225_122551


namespace problem_equivalence_l1225_122511

noncomputable def f (a b x : ℝ) : ℝ := a ^ x + b

theorem problem_equivalence (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : f a b 0 = -2) (h4 : f a b 2 = 0) :
    a = Real.sqrt 3 ∧ b = -3 ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 4, (-8 / 3 : ℝ) ≤ f a b x ∧ f a b x ≤ 6) :=
sorry

end problem_equivalence_l1225_122511


namespace distance_between_cities_l1225_122585

noncomputable def distance_A_to_B : ℕ := 180
noncomputable def distance_B_to_A : ℕ := 150
noncomputable def total_distance : ℕ := distance_A_to_B + distance_B_to_A

theorem distance_between_cities : total_distance = 330 := by
  sorry

end distance_between_cities_l1225_122585


namespace product_nonzero_except_cases_l1225_122598

theorem product_nonzero_except_cases (n : ℤ) (h : n ≠ 5 ∧ n ≠ 17 ∧ n ≠ 257) : 
  (n - 5) * (n - 17) * (n - 257) ≠ 0 :=
by
  sorry

end product_nonzero_except_cases_l1225_122598


namespace negation_proposition_l1225_122573

theorem negation_proposition : ∀ (a : ℝ), (a > 3) → (a^2 ≥ 9) :=
by
  intros a ha
  sorry

end negation_proposition_l1225_122573


namespace john_avg_increase_l1225_122500

theorem john_avg_increase (a b c d : ℝ) (h₁ : a = 90) (h₂ : b = 85) (h₃ : c = 92) (h₄ : d = 95) :
    let initial_avg := (a + b + c) / 3
    let new_avg := (a + b + c + d) / 4
    new_avg - initial_avg = 1.5 :=
by
  sorry

end john_avg_increase_l1225_122500


namespace oliver_bumper_cars_proof_l1225_122590

def rides_of_bumper_cars (total_tickets : ℕ) (tickets_per_ride : ℕ) (rides_ferris_wheel : ℕ) : ℕ :=
  (total_tickets - rides_ferris_wheel * tickets_per_ride) / tickets_per_ride

def oliver_bumper_car_rides : Prop :=
  rides_of_bumper_cars 30 3 7 = 3

theorem oliver_bumper_cars_proof : oliver_bumper_car_rides :=
by
  sorry

end oliver_bumper_cars_proof_l1225_122590


namespace inequality_of_function_inequality_l1225_122572

noncomputable def f (x : ℝ) : ℝ := (Real.log (x + Real.sqrt (x^2 + 1))) + 2 * x + Real.sin x

theorem inequality_of_function_inequality (x1 x2 : ℝ) (h : f x1 + f x2 > 0) : x1 + x2 > 0 :=
sorry

end inequality_of_function_inequality_l1225_122572


namespace prob_A_second_day_is_correct_l1225_122517

-- Definitions for the problem conditions
def prob_first_day_A : ℝ := 0.5
def prob_A_given_A : ℝ := 0.6
def prob_first_day_B : ℝ := 0.5
def prob_A_given_B : ℝ := 0.5

-- Calculate the probability of going to A on the second day
def prob_A_second_day : ℝ :=
  prob_first_day_A * prob_A_given_A + prob_first_day_B * prob_A_given_B

-- The theorem statement
theorem prob_A_second_day_is_correct : 
  prob_A_second_day = 0.55 :=
by
  unfold prob_A_second_day prob_first_day_A prob_A_given_A prob_first_day_B prob_A_given_B
  sorry

end prob_A_second_day_is_correct_l1225_122517


namespace solve_problem_1_solve_problem_2_l1225_122574

/-
Problem 1:
Given the equation 2(x - 1)^2 = 18, prove that x = 4 or x = -2.
-/
theorem solve_problem_1 (x : ℝ) : 2 * (x - 1)^2 = 18 → (x = 4 ∨ x = -2) :=
by
  sorry

/-
Problem 2:
Given the equation x^2 - 4x - 3 = 0, prove that x = 2 + √7 or x = 2 - √7.
-/
theorem solve_problem_2 (x : ℝ) : x^2 - 4 * x - 3 = 0 → (x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) :=
by
  sorry

end solve_problem_1_solve_problem_2_l1225_122574


namespace skaters_total_hours_l1225_122548

-- Define the practice hours based on the conditions
def hannah_weekend_hours := 8
def hannah_weekday_extra_hours := 17
def sarah_weekday_hours := 12
def sarah_weekend_hours := 6
def emma_weekday_hour_multiplier := 2
def emma_weekend_hour_extra := 5

-- Hannah's total hours
def hannah_weekday_hours := hannah_weekend_hours + hannah_weekday_extra_hours
def hannah_total_hours := hannah_weekend_hours + hannah_weekday_hours

-- Sarah's total hours
def sarah_total_hours := sarah_weekday_hours + sarah_weekend_hours

-- Emma's total hours
def emma_weekday_hours := emma_weekday_hour_multiplier * sarah_weekday_hours
def emma_weekend_hours := sarah_weekend_hours + emma_weekend_hour_extra
def emma_total_hours := emma_weekday_hours + emma_weekend_hours

-- Total hours for all three skaters combined
def total_hours := hannah_total_hours + sarah_total_hours + emma_total_hours

-- Lean statement version only, no proof required
theorem skaters_total_hours : total_hours = 86 := by
  sorry

end skaters_total_hours_l1225_122548


namespace total_investment_amount_l1225_122563

-- Define the initial conditions
def amountAt8Percent : ℝ := 3000
def interestAt8Percent (amount : ℝ) : ℝ := amount * 0.08
def interestAt10Percent (amount : ℝ) : ℝ := amount * 0.10
def totalAmount (x y : ℝ) : ℝ := x + y

-- State the theorem
theorem total_investment_amount : 
    let x := 2400
    totalAmount amountAt8Percent x = 5400 :=
by
  sorry

end total_investment_amount_l1225_122563


namespace inclination_line_eq_l1225_122546

theorem inclination_line_eq (l : ℝ → ℝ) (h1 : ∃ x, l x = 2 ∧ ∃ y, l y = 2) (h2 : ∃ θ, θ = 135) :
  ∃ a b c, a = 1 ∧ b = 1 ∧ c = -4 ∧ ∀ x y, y = l x → a * x + b * y + c = 0 :=
by 
  sorry

end inclination_line_eq_l1225_122546


namespace cos_double_angle_l1225_122552

open Real

theorem cos_double_angle {α β : ℝ} (h1 : sin α = sqrt 5 / 5)
                         (h2 : sin (α - β) = - sqrt 10 / 10)
                         (h3 : 0 < α ∧ α < π / 2)
                         (h4 : 0 < β ∧ β < π / 2) :
  cos (2 * β) = 0 :=
  sorry

end cos_double_angle_l1225_122552


namespace sequence_a7_l1225_122569

/-- 
  Given a sequence {a_n} such that a_1 + a_{2n-1} = 4n - 6, 
  prove that a_7 = 11 
-/
theorem sequence_a7 (a : ℕ → ℤ)
  (h : ∀ n : ℕ, a 1 + a (2 * n - 1) = 4 * n - 6) : a 7 = 11 :=
by
  sorry

end sequence_a7_l1225_122569


namespace correct_choice_l1225_122527

def proposition_p : Prop := ∀ (x : ℝ), 2^x > x^2
def proposition_q : Prop := ∃ (x_0 : ℝ), x_0 - 2 > 0

theorem correct_choice : ¬proposition_p ∧ proposition_q :=
by
  sorry

end correct_choice_l1225_122527


namespace sin2θ_over_1pluscos2θ_eq_sqrt3_l1225_122547

theorem sin2θ_over_1pluscos2θ_eq_sqrt3 {θ : ℝ} (h : Real.tan θ = Real.sqrt 3) :
  (Real.sin (2 * θ)) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 :=
sorry

end sin2θ_over_1pluscos2θ_eq_sqrt3_l1225_122547


namespace dad_strawberry_weight_l1225_122523

theorem dad_strawberry_weight :
  ∀ (T L M D : ℕ), T = 36 → L = 8 → M = 12 → (D = T - L - M) → D = 16 :=
by
  intros T L M D hT hL hM hD
  rw [hT, hL, hM] at hD
  exact hD

end dad_strawberry_weight_l1225_122523


namespace car_distance_covered_l1225_122571

def distance_covered_by_car (time : ℝ) (speed : ℝ) : ℝ :=
  speed * time

theorem car_distance_covered :
  distance_covered_by_car (3 + 1/5 : ℝ) 195 = 624 :=
by
  sorry

end car_distance_covered_l1225_122571


namespace min_value_x_plus_y_l1225_122520

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 / x + 1 / y = 1 / 2) : x + y ≥ 18 := sorry

end min_value_x_plus_y_l1225_122520


namespace cost_of_toilet_paper_roll_l1225_122534

-- Definitions of the problem's conditions
def num_toilet_paper_rolls : Nat := 10
def num_paper_towel_rolls : Nat := 7
def num_tissue_boxes : Nat := 3

def cost_per_paper_towel : Real := 2
def cost_per_tissue_box : Real := 2

def total_cost : Real := 35

-- The function to prove
def cost_per_toilet_paper_roll (x : Real) :=
  num_toilet_paper_rolls * x + 
  num_paper_towel_rolls * cost_per_paper_towel + 
  num_tissue_boxes * cost_per_tissue_box = total_cost

-- Statement to prove
theorem cost_of_toilet_paper_roll : 
  cost_per_toilet_paper_roll 1.5 := 
by
  simp [num_toilet_paper_rolls, num_paper_towel_rolls, num_tissue_boxes, cost_per_paper_towel, cost_per_tissue_box, total_cost]
  sorry

end cost_of_toilet_paper_roll_l1225_122534


namespace problem_solution_l1225_122587

theorem problem_solution (a b c : ℝ) (h1 : a^2 - b^2 = 5) (h2 : a * b = 2) (h3 : a^2 + b^2 + c^2 = 8) : 
  a^4 + b^4 + c^4 = 38 :=
sorry

end problem_solution_l1225_122587


namespace tom_distance_before_karen_wins_l1225_122503

theorem tom_distance_before_karen_wins 
    (karen_speed : ℕ)
    (tom_speed : ℕ) 
    (karen_late_start : ℚ) 
    (karen_additional_distance : ℕ) 
    (T : ℚ) 
    (condition1 : karen_speed = 60) 
    (condition2 : tom_speed = 45)
    (condition3 : karen_late_start = 4 / 60)
    (condition4 : karen_additional_distance = 4)
    (condition5 : 60 * T = 45 * T + 8) :
    (45 * (8 / 15) = 24) :=
by
    sorry 

end tom_distance_before_karen_wins_l1225_122503


namespace triangle_inequality_l1225_122510

theorem triangle_inequality (a b c : ℝ) (h : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a / (b + c) + b / (c + a) + c / (a + b) ≤ 2 :=
sorry

end triangle_inequality_l1225_122510


namespace number_of_tshirts_sold_l1225_122506

theorem number_of_tshirts_sold 
    (original_price discounted_price revenue : ℕ)
    (discount : ℕ) 
    (no_of_tshirts: ℕ)
    (h1 : original_price = 51)
    (h2 : discount = 8)
    (h3 : discounted_price = original_price - discount)
    (h4 : revenue = 5590)
    (h5 : revenue = no_of_tshirts * discounted_price) : 
    no_of_tshirts = 130 :=
by
  sorry

end number_of_tshirts_sold_l1225_122506


namespace sum_of_ages_l1225_122524

variable (M E : ℝ)
variable (h1 : M = E + 9)
variable (h2 : M + 5 = 3 * (E - 3))

theorem sum_of_ages : M + E = 32 :=
by
  sorry

end sum_of_ages_l1225_122524


namespace dot_product_parallel_vectors_l1225_122544

variable (x : ℝ)
def a : ℝ × ℝ := (x, x - 1)
def b : ℝ × ℝ := (1, 2)
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 / b.1 = a.2 / b.2

theorem dot_product_parallel_vectors
  (h_parallel : are_parallel (a x) b)
  (h_x : x = -1) :
  (a x).1 * (b).1 + (a x).2 * (b).2 = -5 :=
by
  sorry

end dot_product_parallel_vectors_l1225_122544


namespace find_s_at_3_l1225_122529

def t (x : ℝ) : ℝ := 4 * x - 9
def s (y : ℝ) : ℝ := y^2 - (y + 12)

theorem find_s_at_3 : s 3 = -6 :=
by
  sorry

end find_s_at_3_l1225_122529


namespace mult_469158_9999_l1225_122591

theorem mult_469158_9999 : 469158 * 9999 = 4691176842 := 
by sorry

end mult_469158_9999_l1225_122591


namespace no_intersection_points_l1225_122578

-- Define f(x) and g(x)
def f (x : ℝ) : ℝ := abs (3 * x + 6)
def g (x : ℝ) : ℝ := -abs (4 * x - 3)

-- The main theorem to prove the number of intersection points is zero
theorem no_intersection_points : ∀ x : ℝ, f x ≠ g x := by
  intro x
  sorry -- Proof goes here

end no_intersection_points_l1225_122578


namespace dividend_is_correct_l1225_122558

-- Definitions of the given conditions.
def divisor : ℕ := 17
def quotient : ℕ := 4
def remainder : ℕ := 8

-- Define the dividend using the given formula.
def dividend : ℕ := (divisor * quotient) + remainder

-- The theorem to prove.
theorem dividend_is_correct : dividend = 76 := by
  -- The following line contains a placeholder for the actual proof.
  sorry

end dividend_is_correct_l1225_122558


namespace ludwig_weekly_salary_is_55_l1225_122526

noncomputable def daily_salary : ℝ := 10
noncomputable def full_days : ℕ := 4
noncomputable def half_days : ℕ := 3
noncomputable def half_day_salary := daily_salary / 2

theorem ludwig_weekly_salary_is_55 :
  (full_days * daily_salary + half_days * half_day_salary = 55) := by
  sorry

end ludwig_weekly_salary_is_55_l1225_122526


namespace negative_expressions_l1225_122555

-- Define the approximated values for P, Q, R, S, and T
def P : ℝ := 3.5
def Q : ℝ := 1.1
def R : ℝ := -0.1
def S : ℝ := 0.9
def T : ℝ := 1.5

-- State the theorem to be proved
theorem negative_expressions : 
  (R / (P * Q) < 0) ∧ ((S + T) / R < 0) :=
by
  sorry

end negative_expressions_l1225_122555


namespace bead_bracelet_problem_l1225_122541

-- Define the condition Bead A and Bead B are always next to each other
def adjacent (A B : ℕ) (l : List ℕ) : Prop :=
  ∃ (l1 l2 : List ℕ), l = l1 ++ A :: B :: l2 ∨ l = l1 ++ B :: A :: l2

-- Define the context and translate the problem
def bracelet_arrangements (n : ℕ) : ℕ :=
  if n = 8 then 720 else 0

theorem bead_bracelet_problem : bracelet_arrangements 8 = 720 :=
by {
  -- Place proof here
  sorry 
}

end bead_bracelet_problem_l1225_122541


namespace roger_piles_of_quarters_l1225_122501

theorem roger_piles_of_quarters (Q : ℕ) 
  (h₀ : ∃ Q : ℕ, True) 
  (h₁ : ∀ p, (p = Q) → True)
  (h₂ : ∀ c, (c = 7) → True) 
  (h₃ : Q * 14 = 42) : 
  Q = 3 := 
sorry

end roger_piles_of_quarters_l1225_122501


namespace time_per_bone_l1225_122559

theorem time_per_bone (total_hours : ℕ) (total_bones : ℕ) (h1 : total_hours = 1030) (h2 : total_bones = 206) :
  (total_hours / total_bones = 5) :=
by {
  sorry
}

end time_per_bone_l1225_122559


namespace min_distance_feasible_region_line_l1225_122589

def point (x y : ℝ) : Type := ℝ × ℝ 

theorem min_distance_feasible_region_line :
  ∃ (M N : ℝ × ℝ),
    (2 * M.1 + M.2 - 4 >= 0) ∧
    (M.1 - M.2 - 2 <= 0) ∧
    (M.2 - 3 <= 0) ∧
    (N.2 = -2 * N.1 + 2) ∧
    (dist M N = (2 * Real.sqrt 5)/5) :=
by 
  sorry

end min_distance_feasible_region_line_l1225_122589


namespace difference_of_squares_example_product_calculation_factorization_by_completing_square_l1225_122502

/-
  Theorem: The transformation in the step \(195 \times 205 = 200^2 - 5^2\) uses the difference of squares formula.
-/

theorem difference_of_squares_example : 
  (195 * 205 = (200 - 5) * (200 + 5)) ∧ ((200 - 5) * (200 + 5) = 200^2 - 5^2) :=
  sorry

/-
  Theorem: Calculate \(9 \times 11 \times 101 \times 10001\) using a simple method.
-/

theorem product_calculation : 
  9 * 11 * 101 * 10001 = 99999999 :=
  sorry

/-
  Theorem: Factorize \(a^2 - 6a + 8\) using the completing the square method.
-/

theorem factorization_by_completing_square (a : ℝ) :
  a^2 - 6 * a + 8 = (a - 2) * (a - 4) :=
  sorry

end difference_of_squares_example_product_calculation_factorization_by_completing_square_l1225_122502


namespace last_digit_101_pow_100_l1225_122531

theorem last_digit_101_pow_100 :
  (101^100) % 10 = 1 :=
by
  sorry

end last_digit_101_pow_100_l1225_122531


namespace find_white_balls_l1225_122577

-- Define the number of red balls
def red_balls : ℕ := 4

-- Define the probability of drawing a red ball
def prob_red : ℚ := 1 / 4

-- Define the number of white balls
def white_balls : ℕ := 12

theorem find_white_balls (x : ℕ) (h1 : (red_balls : ℚ) / (red_balls + x) = prob_red) : x = white_balls :=
by
  -- Proof is omitted
  sorry

end find_white_balls_l1225_122577


namespace betty_total_blue_and_green_beads_l1225_122542

theorem betty_total_blue_and_green_beads (r b g : ℕ) (h1 : 5 * b = 3 * r) (h2 : 5 * g = 2 * r) (h3 : r = 50) : b + g = 50 :=
by
  sorry

end betty_total_blue_and_green_beads_l1225_122542


namespace max_initial_jars_l1225_122513

theorem max_initial_jars (w_B w_C a : ℤ) (h1 : w_C = 13 * w_B) (h2 : w_C - a = 8 * (w_B + a)) : 
  ∃ (n : ℤ), n ≤ 23 ∧ ∀ (k : ℤ), w_B = 9 * k ∧ w_C = 117 * k := 
  by 
  sorry

end max_initial_jars_l1225_122513


namespace select_students_l1225_122580

-- Definitions for the conditions
variables (A B C D E : Prop)

-- Conditions
def condition1 : Prop := A → B ∧ ¬E
def condition2 : Prop := (B ∨ E) → ¬D
def condition3 : Prop := C ∨ D

-- The main theorem
theorem select_students (hA : A) (h1 : condition1 A B E) (h2 : condition2 B E D) (h3 : condition3 C D) : B ∧ C :=
by 
  sorry

end select_students_l1225_122580


namespace farmer_land_l1225_122504

theorem farmer_land (A : ℝ) (h1 : 0.9 * A = A_cleared) (h2 : 0.3 * A_cleared = A_soybeans) 
  (h3 : 0.6 * A_cleared = A_wheat) (h4 : 0.1 * A_cleared = 540) : A = 6000 :=
by
  sorry

end farmer_land_l1225_122504


namespace percentage_third_year_students_l1225_122570

-- Define the conditions as given in the problem
variables (T : ℝ) (T_3 : ℝ) (S_2 : ℝ)

-- Conditions
def cond1 : Prop := S_2 = 0.10 * T
def cond2 : Prop := (0.10 * T) / (T - T_3) = 1 / 7

-- Define the proof goal
theorem percentage_third_year_students (h1 : cond1 T S_2) (h2 : cond2 T T_3) : T_3 = 0.30 * T :=
sorry

end percentage_third_year_students_l1225_122570


namespace hua_luogeng_optimal_selection_l1225_122508

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l1225_122508


namespace brownies_count_l1225_122537

variable (total_people : Nat) (pieces_per_person : Nat) (cookies : Nat) (candy : Nat) (brownies : Nat)

def total_dessert_needed : Nat := total_people * pieces_per_person

def total_pieces_have : Nat := cookies + candy

def total_brownies_needed : Nat := total_dessert_needed total_people pieces_per_person - total_pieces_have cookies candy

theorem brownies_count (h1 : total_people = 7)
                       (h2 : pieces_per_person = 18)
                       (h3 : cookies = 42)
                       (h4 : candy = 63) :
                       total_brownies_needed total_people pieces_per_person cookies candy = 21 :=
by
  rw [h1, h2, h3, h4]
  sorry

end brownies_count_l1225_122537


namespace vector_dot_product_problem_l1225_122583

theorem vector_dot_product_problem :
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (-1, 3)
  let C : ℝ × ℝ := (2, 1)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
  let BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
  let dot_prod := AB.1 * (2 * AC.1 + BC.1) + AB.2 * (2 * AC.2 + BC.2)
  dot_prod = -14 :=
by
  sorry

end vector_dot_product_problem_l1225_122583


namespace snowboard_price_after_discounts_l1225_122592

theorem snowboard_price_after_discounts
  (original_price : ℝ) (friday_discount_rate : ℝ) (monday_discount_rate : ℝ) 
  (sales_tax_rate : ℝ) (price_after_all_adjustments : ℝ) :
  original_price = 200 →
  friday_discount_rate = 0.40 →
  monday_discount_rate = 0.20 →
  sales_tax_rate = 0.05 →
  price_after_all_adjustments = 100.80 :=
by
  intros
  sorry

end snowboard_price_after_discounts_l1225_122592


namespace inverse_value_at_2_l1225_122536

noncomputable def f (x : ℝ) : ℝ := x / (2 * x + 1)

noncomputable def f_inv (x : ℝ) : ℝ := x / (1 - 2 * x)

theorem inverse_value_at_2 :
  f_inv 2 = -2/3 := by
  sorry

end inverse_value_at_2_l1225_122536
