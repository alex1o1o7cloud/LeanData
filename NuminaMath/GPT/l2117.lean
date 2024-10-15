import Mathlib

namespace NUMINAMATH_GPT_quadratic_has_two_distinct_roots_l2117_211758

theorem quadratic_has_two_distinct_roots (a b c α : ℝ) (h : a * (a * α^2 + b * α + c) < 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a*x1^2 + b*x1 + c = 0) ∧ (a*x2^2 + b*x2 + c = 0) ∧ x1 < α ∧ x2 > α :=
sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_roots_l2117_211758


namespace NUMINAMATH_GPT_DE_plus_FG_equals_19_div_6_l2117_211774

theorem DE_plus_FG_equals_19_div_6
    (AB AC : ℝ)
    (BC : ℝ)
    (h_isosceles : AB = 2 ∧ AC = 2 ∧ BC = 1.5)
    (D E G F : ℝ)
    (h_parallel_DE_BC : D = E)
    (h_parallel_FG_BC : F = G)
    (h_same_perimeter : 2 + D = 2 + F ∧ 2 + F = 5.5 - F) :
    D + F = 19 / 6 := by
  sorry

end NUMINAMATH_GPT_DE_plus_FG_equals_19_div_6_l2117_211774


namespace NUMINAMATH_GPT_largest_partner_share_l2117_211767

-- Definitions for the conditions
def total_profit : ℕ := 48000
def ratio_parts : List ℕ := [2, 4, 5, 3, 6]
def total_ratio_parts : ℕ := ratio_parts.sum
def value_per_part : ℕ := total_profit / total_ratio_parts
def largest_share : ℕ := 6 * value_per_part

-- Statement of the proof problem
theorem largest_partner_share : largest_share = 14400 := by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_largest_partner_share_l2117_211767


namespace NUMINAMATH_GPT_total_songs_sung_l2117_211726

def total_minutes := 80
def intermission_minutes := 10
def long_song_minutes := 10
def short_song_minutes := 5

theorem total_songs_sung : 
  (total_minutes - intermission_minutes - long_song_minutes) / short_song_minutes + 1 = 13 := 
by 
  sorry

end NUMINAMATH_GPT_total_songs_sung_l2117_211726


namespace NUMINAMATH_GPT_infinite_coprime_binom_l2117_211744

theorem infinite_coprime_binom (k l : ℕ) (hk : k > 0) (hl : l > 0) : 
  ∃ᶠ m in atTop, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1 := by
sorry

end NUMINAMATH_GPT_infinite_coprime_binom_l2117_211744


namespace NUMINAMATH_GPT_not_p_and_not_p_and_q_implies_not_p_or_q_l2117_211771

theorem not_p_and_not_p_and_q_implies_not_p_or_q (p q : Prop) (h1 : ¬p) (h2 : ¬(p ∧ q)) : ¬(p ∨ q) :=
sorry

end NUMINAMATH_GPT_not_p_and_not_p_and_q_implies_not_p_or_q_l2117_211771


namespace NUMINAMATH_GPT_sin_2012_equals_neg_sin_32_l2117_211717

theorem sin_2012_equals_neg_sin_32 : Real.sin (2012 * Real.pi / 180) = -Real.sin (32 * Real.pi / 180) := by
  sorry

end NUMINAMATH_GPT_sin_2012_equals_neg_sin_32_l2117_211717


namespace NUMINAMATH_GPT_closest_clock_to_16_is_C_l2117_211776

noncomputable def closestTo16InMirror (clock : Char) : Bool :=
  clock = 'C'

theorem closest_clock_to_16_is_C : 
  (closestTo16InMirror 'A' = False) ∧ 
  (closestTo16InMirror 'B' = False) ∧ 
  (closestTo16InMirror 'C' = True) ∧ 
  (closestTo16InMirror 'D' = False) := 
by
  sorry

end NUMINAMATH_GPT_closest_clock_to_16_is_C_l2117_211776


namespace NUMINAMATH_GPT_solve_for_m_l2117_211780

def power_function_monotonic (m : ℝ) : Prop :=
  (m^2 - m - 1 = 1) ∧ (m^2 - 2 * m - 3 < 0)

theorem solve_for_m (m : ℝ) (h : power_function_monotonic m) : m = 2 :=
sorry

end NUMINAMATH_GPT_solve_for_m_l2117_211780


namespace NUMINAMATH_GPT_certain_number_is_4_l2117_211720

theorem certain_number_is_4 (x y C : ℝ) (h1 : 2 * x - y = C) (h2 : 6 * x - 3 * y = 12) : C = 4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_certain_number_is_4_l2117_211720


namespace NUMINAMATH_GPT_find_x_l2117_211769

noncomputable def a (x : ℝ) : ℝ × ℝ :=
  (Real.cos (3 * x / 2), Real.sin (3 * x / 2))

noncomputable def b (x : ℝ) : ℝ × ℝ :=
  (Real.cos (x / 2), -Real.sin (x / 2))

noncomputable def norm_sq (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

theorem find_x (x : ℝ) :
  (0 ≤ x ∧ x ≤ Real.pi)
  ∧ (norm_sq (a x) + norm_sq (b x) + 2 * ((a x).1 * (b x).1 + (a x).2 * (b x).2) = 1)
  → (x = Real.pi / 3 ∨ x = 2 * Real.pi / 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l2117_211769


namespace NUMINAMATH_GPT_game_remaining_sprite_color_l2117_211745

theorem game_remaining_sprite_color (m n : ℕ) : 
  (∀ m n : ℕ, ∃ sprite : String, sprite = if n % 2 = 0 then "Red" else "Blue") :=
by sorry

end NUMINAMATH_GPT_game_remaining_sprite_color_l2117_211745


namespace NUMINAMATH_GPT_petya_equals_vasya_l2117_211766

def petya_word_count (m : ℕ) : ℕ :=
  sorry -- The actual count of m-letter words with equal T's and O's using letters T, O, W, and N.

def vasya_word_count (m : ℕ) : ℕ :=
  sorry -- The actual count of 2m-letter words with equal T's and O's using only letters T and O.

theorem petya_equals_vasya (m : ℕ) : petya_word_count m = vasya_word_count m :=
  sorry

end NUMINAMATH_GPT_petya_equals_vasya_l2117_211766


namespace NUMINAMATH_GPT_largest_two_digit_with_remainder_2_l2117_211738

theorem largest_two_digit_with_remainder_2 (n : ℕ) :
  10 ≤ n ∧ n ≤ 99 ∧ n % 13 = 2 → n = 93 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_largest_two_digit_with_remainder_2_l2117_211738


namespace NUMINAMATH_GPT_minimum_xy_l2117_211788

theorem minimum_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/y = 1/2) : x * y ≥ 16 :=
sorry

end NUMINAMATH_GPT_minimum_xy_l2117_211788


namespace NUMINAMATH_GPT_common_number_in_sequences_l2117_211768

theorem common_number_in_sequences (n m: ℕ) (a : ℕ)
    (h1 : a = 3 + 8 * n)
    (h2 : a = 5 + 9 * m)
    (h3 : 1 ≤ a ∧ a ≤ 200) : a = 131 :=
by
  sorry

end NUMINAMATH_GPT_common_number_in_sequences_l2117_211768


namespace NUMINAMATH_GPT_intersection_single_point_max_PA_PB_l2117_211718

-- Problem (1)
theorem intersection_single_point (a : ℝ) :
  (∀ x : ℝ, 2 * a = |x - a| - 1 → x = a) → a = -1 / 2 :=
sorry

-- Problem (2)
theorem max_PA_PB (m : ℝ) (P : ℝ × ℝ) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 3)
  P ≠ A ∧ P ≠ B ∧ (P.1 + m * P.2 = 0) ∧ (m * P.1 - P.2 - m + 3 = 0) →
  |dist P A| * |dist P B| ≤ 5 :=
sorry

end NUMINAMATH_GPT_intersection_single_point_max_PA_PB_l2117_211718


namespace NUMINAMATH_GPT_train_crosses_tunnel_in_45_sec_l2117_211715

/-- Given the length of the train, the length of the platform, the length of the tunnel, 
and the time taken to cross the platform, prove the time taken for the train to cross the tunnel is 45 seconds. -/
theorem train_crosses_tunnel_in_45_sec (l_train : ℕ) (l_platform : ℕ) (t_platform : ℕ) (l_tunnel : ℕ)
  (h_train_length : l_train = 330)
  (h_platform_length : l_platform = 180)
  (h_time_platform : t_platform = 15)
  (h_tunnel_length : l_tunnel = 1200) :
  (l_train + l_tunnel) / ((l_train + l_platform) / t_platform) = 45 :=
by
  -- placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_train_crosses_tunnel_in_45_sec_l2117_211715


namespace NUMINAMATH_GPT_lana_goal_is_20_l2117_211739

def muffins_sold_morning := 12
def muffins_sold_afternoon := 4
def muffins_needed_to_goal := 4
def total_muffins_sold := muffins_sold_morning + muffins_sold_afternoon
def lana_goal := total_muffins_sold + muffins_needed_to_goal

theorem lana_goal_is_20 : lana_goal = 20 := by
  sorry

end NUMINAMATH_GPT_lana_goal_is_20_l2117_211739


namespace NUMINAMATH_GPT_miss_tree_class_children_count_l2117_211753

noncomputable def number_of_children (n: ℕ) : ℕ := 7 * n + 2

theorem miss_tree_class_children_count (n : ℕ) :
  (20 < number_of_children n) ∧ (number_of_children n < 30) ∧ 7 * n + 2 = 23 :=
by {
  sorry
}

end NUMINAMATH_GPT_miss_tree_class_children_count_l2117_211753


namespace NUMINAMATH_GPT_expand_expression_l2117_211785

theorem expand_expression : ∀ (x : ℝ), (1 + x^3) * (1 - x^4 + x^5) = 1 + x^3 - x^4 + x^5 - x^7 + x^8 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_expand_expression_l2117_211785


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l2117_211723

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℝ) (d a1 : ℝ) (h1 : a 3 = a1 + 2 * d) (h2 : a 5 = a1 + 4 * d)
  (h3 : a 7 = a1 + 6 * d) (h4 : a 10 = a1 + 9 * d) (h5 : a 13 = a1 + 12 * d) (h6 : (a 3) + (a 5) = 2) (h7 : (a 7) + (a 10) + (a 13) = 9) :
  d = (1 / 3) := by
  sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l2117_211723


namespace NUMINAMATH_GPT_functional_equation_solution_l2117_211712

theorem functional_equation_solution (f : ℕ+ → ℕ+) :
  (∀ n : ℕ+, f (f (f n)) + f (f n) + f n = 3 * n) →
  ∀ n : ℕ+, f n = n :=
by
  intro h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l2117_211712


namespace NUMINAMATH_GPT_mean_median_difference_is_correct_l2117_211748

noncomputable def mean_median_difference (scores : List ℕ) (percentages : List ℚ) : ℚ := sorry

theorem mean_median_difference_is_correct :
  mean_median_difference [60, 75, 85, 90, 100] [15/100, 20/100, 25/100, 30/100, 10/100] = 2.75 :=
sorry

end NUMINAMATH_GPT_mean_median_difference_is_correct_l2117_211748


namespace NUMINAMATH_GPT_wine_consumption_correct_l2117_211710

-- Definitions based on conditions
def drank_after_first_pound : ℚ := 1
def drank_after_second_pound : ℚ := 1
def drank_after_third_pound : ℚ := 1 / 2
def drank_after_fourth_pound : ℚ := 1 / 4
def drank_after_fifth_pound : ℚ := 1 / 8
def drank_after_sixth_pound : ℚ := 1 / 16

-- Total wine consumption
def total_wine_consumption : ℚ :=
  drank_after_first_pound + drank_after_second_pound +
  drank_after_third_pound + drank_after_fourth_pound +
  drank_after_fifth_pound + drank_after_sixth_pound

-- Theorem statement
theorem wine_consumption_correct :
  total_wine_consumption = 47 / 16 :=
by
  sorry

end NUMINAMATH_GPT_wine_consumption_correct_l2117_211710


namespace NUMINAMATH_GPT_circles_tangent_l2117_211782

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 16*y - 48 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 44 = 0

theorem circles_tangent :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y :=
sorry

end NUMINAMATH_GPT_circles_tangent_l2117_211782


namespace NUMINAMATH_GPT_meaningful_sqrt_range_l2117_211700

theorem meaningful_sqrt_range (x : ℝ) (h : 2 * x - 3 ≥ 0) : x ≥ 3 / 2 :=
sorry

end NUMINAMATH_GPT_meaningful_sqrt_range_l2117_211700


namespace NUMINAMATH_GPT_smallest_m_for_integral_solutions_l2117_211713

theorem smallest_m_for_integral_solutions (p q : ℤ) (h : p * q = 42) (h0 : p + q = m / 15) : 
  0 < m ∧ 15 * p * p - m * p + 630 = 0 ∧ 15 * q * q - m * q + 630 = 0 →
  m = 195 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_m_for_integral_solutions_l2117_211713


namespace NUMINAMATH_GPT_twenty_four_is_eighty_percent_of_what_number_l2117_211792

theorem twenty_four_is_eighty_percent_of_what_number (x : ℝ) (hx : 24 = 0.8 * x) : x = 30 :=
  sorry

end NUMINAMATH_GPT_twenty_four_is_eighty_percent_of_what_number_l2117_211792


namespace NUMINAMATH_GPT_least_k_for_sum_divisible_l2117_211790

theorem least_k_for_sum_divisible (n : ℕ) (hn : n > 0) : 
  (∃ k : ℕ, (∀ (xs : List ℕ), (xs.length = k) → (∃ ys : List ℕ, (ys.length % 2 = 0) ∧ (ys.sum % n = 0))) ∧ 
    (k = if n % 2 = 1 then 2 * n else n + 1)) :=
sorry

end NUMINAMATH_GPT_least_k_for_sum_divisible_l2117_211790


namespace NUMINAMATH_GPT_athletes_meet_time_number_of_overtakes_l2117_211747

-- Define the speeds of the athletes
def speed1 := 155 -- m/min
def speed2 := 200 -- m/min
def speed3 := 275 -- m/min

-- Define the total length of the track
def track_length := 400 -- meters

-- Prove the minimum time for the athletes to meet again is 80/3 minutes
theorem athletes_meet_time (speed1 speed2 speed3 track_length : ℕ) (h1 : speed1 = 155) (h2 : speed2 = 200) (h3 : speed3 = 275) (h4 : track_length = 400) :
  ∃ t : ℚ, t = (80 / 3 : ℚ) :=
by
  sorry

-- Prove the number of overtakes during this time is 13
theorem number_of_overtakes (speed1 speed2 speed3 track_length t : ℕ) (h1 : speed1 = 155) (h2 : speed2 = 200) (h3 : speed3 = 275) (h4 : track_length = 400) (h5 : t = 80 / 3) :
  ∃ n : ℕ, n = 13 :=
by
  sorry

end NUMINAMATH_GPT_athletes_meet_time_number_of_overtakes_l2117_211747


namespace NUMINAMATH_GPT_find_triangle_side_value_find_triangle_tan_value_l2117_211732

noncomputable def triangle_side_value (A B C : ℝ) (a b c : ℝ) : Prop :=
  C = 2 * Real.pi / 3 ∧
  c = 5 ∧
  a = Real.sqrt 5 * b * Real.sin A ∧
  b = 2 * Real.sqrt 15 / 3

noncomputable def triangle_tan_value (B : ℝ) : Prop :=
  Real.tan (B + Real.pi / 4) = 3

theorem find_triangle_side_value (A B C a b c : ℝ) :
  triangle_side_value A B C a b c := by sorry

theorem find_triangle_tan_value (B : ℝ) :
  triangle_tan_value B := by sorry

end NUMINAMATH_GPT_find_triangle_side_value_find_triangle_tan_value_l2117_211732


namespace NUMINAMATH_GPT_projection_correct_l2117_211743

theorem projection_correct :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-1, 3)
  -- Definition of dot product for 2D vectors
  let dot (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2
  -- Definition of projection of a onto b
  let proj := (dot a b / (b.1^2 + b.2^2)) • b
  proj = (-1 / 2, 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_projection_correct_l2117_211743


namespace NUMINAMATH_GPT_sum_digits_l2117_211731

def distinct_digits (a b c d : ℕ) : Prop :=
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d)

def valid_equation (Y E M T : ℕ) : Prop :=
  ∃ (YE ME TTT : ℕ),
    YE = Y * 10 + E ∧
    ME = M * 10 + E ∧
    TTT = T * 111 ∧
    YE < ME ∧
    YE * ME = TTT ∧
    distinct_digits Y E M T

theorem sum_digits (Y E M T : ℕ) :
  valid_equation Y E M T → Y + E + M + T = 21 := 
sorry

end NUMINAMATH_GPT_sum_digits_l2117_211731


namespace NUMINAMATH_GPT_pascals_triangle_contains_47_once_l2117_211746

theorem pascals_triangle_contains_47_once (n : ℕ) : 
  (∃ k, k ≤ n ∧ Nat.choose n k = 47) ↔ n = 47 := by
  sorry

end NUMINAMATH_GPT_pascals_triangle_contains_47_once_l2117_211746


namespace NUMINAMATH_GPT_proportion_terms_l2117_211786

theorem proportion_terms (x v y z : ℤ) (a b c : ℤ)
  (h1 : x + v = y + z + a)
  (h2 : x^2 + v^2 = y^2 + z^2 + b)
  (h3 : x^4 + v^4 = y^4 + z^4 + c)
  (ha : a = 7) (hb : b = 21) (hc : c = 2625) :
  (x = -3 ∧ v = 8 ∧ y = -6 ∧ z = 4) :=
by
  sorry

end NUMINAMATH_GPT_proportion_terms_l2117_211786


namespace NUMINAMATH_GPT_birds_after_changes_are_235_l2117_211722

-- Define initial conditions for the problem
def initial_cages : Nat := 15
def parrots_per_cage : Nat := 3
def parakeets_per_cage : Nat := 8
def canaries_per_cage : Nat := 5
def parrots_sold : Nat := 5
def canaries_sold : Nat := 2
def parakeets_added : Nat := 2


-- Define the function to count total birds after the changes
def total_birds_after_changes (initial_cages parrots_per_cage parakeets_per_cage canaries_per_cage parrots_sold canaries_sold parakeets_added : Nat) : Nat :=
  let initial_parrots := initial_cages * parrots_per_cage
  let initial_parakeets := initial_cages * parakeets_per_cage
  let initial_canaries := initial_cages * canaries_per_cage
  
  let final_parrots := initial_parrots - parrots_sold
  let final_parakeets := initial_parakeets + parakeets_added
  let final_canaries := initial_canaries - canaries_sold
  
  final_parrots + final_parakeets + final_canaries

-- Prove that the total number of birds is 235
theorem birds_after_changes_are_235 : total_birds_after_changes 15 3 8 5 5 2 2 = 235 :=
  by 
    -- Proof is omitted as per the instructions
    sorry

end NUMINAMATH_GPT_birds_after_changes_are_235_l2117_211722


namespace NUMINAMATH_GPT_meeting_day_correct_l2117_211761

noncomputable def smallest_meeting_day :=
  ∀ (players courts : ℕ)
    (initial_reimu_court initial_marisa_court : ℕ),
    players = 2016 →
    courts = 1008 →
    initial_reimu_court = 123 →
    initial_marisa_court = 876 →
    ∀ (winner_moves_to court : ℕ → ℕ),
      (∀ (i : ℕ), 2 ≤ i ∧ i ≤ courts → winner_moves_to i = i - 1) →
      (winner_moves_to 1 = 1) →
      ∀ (loser_moves_to court : ℕ → ℕ),
        (∀ (j : ℕ), 1 ≤ j ∧ j ≤ courts - 1 → loser_moves_to j = j + 1) →
        (loser_moves_to courts = courts) →
        ∃ (n : ℕ), n = 1139

theorem meeting_day_correct : smallest_meeting_day :=
  sorry

end NUMINAMATH_GPT_meeting_day_correct_l2117_211761


namespace NUMINAMATH_GPT_probability_of_ace_ten_king_l2117_211760

noncomputable def probability_first_ace_second_ten_third_king : ℚ :=
  (4/52) * (4/51) * (4/50)

theorem probability_of_ace_ten_king :
  probability_first_ace_second_ten_third_king = 2/16575 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_ace_ten_king_l2117_211760


namespace NUMINAMATH_GPT_volume_cube_box_for_pyramid_l2117_211740

theorem volume_cube_box_for_pyramid (h_pyramid : height_of_pyramid = 18) 
  (base_side_pyramid : side_of_square_base = 15) : 
  volume_of_box = 18^3 :=
by
  sorry

end NUMINAMATH_GPT_volume_cube_box_for_pyramid_l2117_211740


namespace NUMINAMATH_GPT_zero_in_interval_l2117_211795

noncomputable def f (x : ℝ) : ℝ := Real.logb 3 x + 2 * x - 8

theorem zero_in_interval : (f 3 < 0) ∧ (f 4 > 0) → ∃ c, 3 < c ∧ c < 4 ∧ f c = 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_in_interval_l2117_211795


namespace NUMINAMATH_GPT_logan_passengers_count_l2117_211714

noncomputable def passengers_used_Kennedy_Airport : ℝ := (1 / 3) * 38.3
noncomputable def passengers_used_Miami_Airport : ℝ := (1 / 2) * passengers_used_Kennedy_Airport
noncomputable def passengers_used_Logan_Airport : ℝ := passengers_used_Miami_Airport / 4

theorem logan_passengers_count : abs (passengers_used_Logan_Airport - 1.6) < 0.01 := by
  sorry

end NUMINAMATH_GPT_logan_passengers_count_l2117_211714


namespace NUMINAMATH_GPT_translate_sin_eq_cos_l2117_211716

theorem translate_sin_eq_cos (φ : ℝ) (hφ : 0 ≤ φ ∧ φ < 2 * Real.pi) :
  (∀ x, Real.cos (x - Real.pi / 6) = Real.sin (x + φ)) → φ = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_translate_sin_eq_cos_l2117_211716


namespace NUMINAMATH_GPT_problem_statement_l2117_211736

noncomputable def nonnegative_reals : Type := {x : ℝ // 0 ≤ x}

theorem problem_statement (x : nonnegative_reals) :
  x.1^(3/2) + 6*x.1^(5/4) + 8*x.1^(3/4) ≥ 15*x.1 ∧
  (x.1^(3/2) + 6*x.1^(5/4) + 8*x.1^(3/4) = 15*x.1 ↔ (x.1 = 0 ∨ x.1 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2117_211736


namespace NUMINAMATH_GPT_max_mean_BC_l2117_211783

theorem max_mean_BC (A_n B_n C_n A_total_weight B_total_weight C_total_weight : ℕ)
    (hA_mean : A_total_weight = 45 * A_n)
    (hB_mean : B_total_weight = 55 * B_n)
    (hAB_mean : (A_total_weight + B_total_weight) / (A_n + B_n) = 48)
    (hAC_mean : (A_total_weight + C_total_weight) / (A_n + C_n) = 50) :
    ∃ m : ℤ, m = 66 := by
  sorry

end NUMINAMATH_GPT_max_mean_BC_l2117_211783


namespace NUMINAMATH_GPT_aaron_earnings_l2117_211708

def monday_hours : ℚ := 7 / 4
def tuesday_hours : ℚ := 1 + 10 / 60
def wednesday_hours : ℚ := 3 + 15 / 60
def friday_hours : ℚ := 45 / 60

def total_hours_worked : ℚ := monday_hours + tuesday_hours + wednesday_hours + friday_hours
def hourly_rate : ℚ := 4

def total_earnings : ℚ := total_hours_worked * hourly_rate

theorem aaron_earnings : total_earnings = 27 := by
  sorry

end NUMINAMATH_GPT_aaron_earnings_l2117_211708


namespace NUMINAMATH_GPT_percent_increase_between_maintenance_checks_l2117_211719

theorem percent_increase_between_maintenance_checks (original_time new_time : ℕ) (h_orig : original_time = 50) (h_new : new_time = 60) :
  ((new_time - original_time : ℚ) / original_time) * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_percent_increase_between_maintenance_checks_l2117_211719


namespace NUMINAMATH_GPT_hexagon_perimeter_sum_l2117_211756

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def perimeter : ℝ := 
  distance 0 0 1 2 +
  distance 1 2 3 3 +
  distance 3 3 5 3 +
  distance 5 3 6 1 +
  distance 6 1 4 (-1) +
  distance 4 (-1) 0 0

theorem hexagon_perimeter_sum :
  perimeter = 3 * Real.sqrt 5 + 2 + 2 * Real.sqrt 2 + Real.sqrt 17 := 
sorry

end NUMINAMATH_GPT_hexagon_perimeter_sum_l2117_211756


namespace NUMINAMATH_GPT_solve_linear_equation_l2117_211777

theorem solve_linear_equation : ∀ x : ℝ, 4 * (2 * x - 1) = 1 - 3 * (x + 2) → x = -1 / 11 :=
by
  intro x h
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_solve_linear_equation_l2117_211777


namespace NUMINAMATH_GPT_number_of_women_l2117_211798

theorem number_of_women (w1 w2: ℕ) (m1 m2 d1 d2: ℕ)
    (h1: w2 = 5) (h2: m2 = 100) (h3: d2 = 1) 
    (h4: d1 = 3) (h5: m1 = 360)
    (h6: w1 * d1 = m1 * d2 / m2 * w2) : w1 = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_women_l2117_211798


namespace NUMINAMATH_GPT_compute_expression_l2117_211791

theorem compute_expression :
  18 * (216 / 3 + 36 / 6 + 4 / 9 + 2 + 1 / 18) = 1449 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l2117_211791


namespace NUMINAMATH_GPT_average_of_first_12_even_is_13_l2117_211759

-- Define the first 12 even numbers
def first_12_even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

-- Define the sum of the first 12 even numbers
def sum_first_12_even : ℕ := first_12_even_numbers.sum

-- Define the number of values
def num_vals : ℕ := first_12_even_numbers.length

-- Define the average calculation
def average_first_12_even : ℕ := sum_first_12_even / num_vals

-- The theorem we want to prove
theorem average_of_first_12_even_is_13 : average_first_12_even = 13 := by
  sorry

end NUMINAMATH_GPT_average_of_first_12_even_is_13_l2117_211759


namespace NUMINAMATH_GPT_find_correct_four_digit_number_l2117_211729

theorem find_correct_four_digit_number (N : ℕ) (misspelledN : ℕ) (misspelled_unit_digit_correction : ℕ) 
  (h1 : misspelledN = (N / 10) * 10 + 6)
  (h2 : N - misspelled_unit_digit_correction = (N / 10) * 10 - 7 + 9)
  (h3 : misspelledN - 57 = 1819) : N = 1879 :=
  sorry


end NUMINAMATH_GPT_find_correct_four_digit_number_l2117_211729


namespace NUMINAMATH_GPT_fraction_of_earnings_spent_on_candy_l2117_211724

theorem fraction_of_earnings_spent_on_candy :
  let candy_bars_cost := 2 * 0.75
  let lollipops_cost := 4 * 0.25
  let total_candy_cost := candy_bars_cost + lollipops_cost
  let earnings_per_driveway := 1.5
  let total_earnings := 10 * earnings_per_driveway
  total_candy_cost / total_earnings = 1 / 6 :=
by
  let candy_bars_cost := 2 * 0.75
  let lollipops_cost := 4 * 0.25
  let total_candy_cost := candy_bars_cost + lollipops_cost
  let earnings_per_driveway := 1.5
  let total_earnings := 10 * earnings_per_driveway
  have h : total_candy_cost / total_earnings = 1 / 6 := by sorry
  exact h

end NUMINAMATH_GPT_fraction_of_earnings_spent_on_candy_l2117_211724


namespace NUMINAMATH_GPT_number_of_problems_l2117_211727

theorem number_of_problems (Terry_score : ℤ) (points_right : ℤ) (points_wrong : ℤ) (wrong_ans : ℤ) 
  (h_score : Terry_score = 85) (h_points_right : points_right = 4) 
  (h_points_wrong : points_wrong = -1) (h_wrong_ans : wrong_ans = 3) : 
  ∃ (total_problems : ℤ), total_problems = 25 :=
by
  sorry

end NUMINAMATH_GPT_number_of_problems_l2117_211727


namespace NUMINAMATH_GPT_exists_point_on_graph_of_quadratic_l2117_211752

-- Define the condition for the discriminant to be zero
def is_single_root (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c = 0

-- Define a function representing a quadratic polynomial
def quadratic_poly (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- The main statement
theorem exists_point_on_graph_of_quadratic (b c : ℝ) 
  (h : is_single_root 1 b c) :
  ∃ (p q : ℝ), q = (p^2) / 4 ∧ is_single_root 1 p q :=
sorry

end NUMINAMATH_GPT_exists_point_on_graph_of_quadratic_l2117_211752


namespace NUMINAMATH_GPT_fraction_of_B_amount_equals_third_of_A_amount_l2117_211772

variable (A B : ℝ)
variable (x : ℝ)

theorem fraction_of_B_amount_equals_third_of_A_amount
  (h1 : A + B = 1210)
  (h2 : B = 484)
  (h3 : (1 / 3) * A = x * B) : 
  x = 1 / 2 :=
sorry

end NUMINAMATH_GPT_fraction_of_B_amount_equals_third_of_A_amount_l2117_211772


namespace NUMINAMATH_GPT_ratio_of_A_to_B_l2117_211707

-- Definitions of the conditions.
def amount_A : ℕ := 200
def total_amount : ℕ := 600
def amount_B : ℕ := total_amount - amount_A

-- The proof statement.
theorem ratio_of_A_to_B :
  amount_A / amount_B = 1 / 2 := 
sorry

end NUMINAMATH_GPT_ratio_of_A_to_B_l2117_211707


namespace NUMINAMATH_GPT_convert_A03_to_decimal_l2117_211787

theorem convert_A03_to_decimal :
  let A := 10
  let hex_value := A * 16^2 + 0 * 16^1 + 3 * 16^0
  hex_value = 2563 :=
by
  let A := 10
  let hex_value := A * 16^2 + 0 * 16^1 + 3 * 16^0
  have : hex_value = 2563 := sorry
  exact this

end NUMINAMATH_GPT_convert_A03_to_decimal_l2117_211787


namespace NUMINAMATH_GPT_ThreePowFifteenModFive_l2117_211709

def rem_div_3_pow_15_by_5 : ℕ :=
  let base := 3
  let mod := 5
  let exp := 15
  
  base^exp % mod

theorem ThreePowFifteenModFive (h1: 3^4 ≡ 1 [MOD 5]) : rem_div_3_pow_15_by_5 = 2 := by
  sorry

end NUMINAMATH_GPT_ThreePowFifteenModFive_l2117_211709


namespace NUMINAMATH_GPT_breaststroke_hours_correct_l2117_211749

namespace Swimming

def total_required_hours : ℕ := 1500
def backstroke_hours : ℕ := 50
def butterfly_hours : ℕ := 121
def monthly_freestyle_sidestroke_hours : ℕ := 220
def months : ℕ := 6

def calculated_total_hours : ℕ :=
  backstroke_hours + butterfly_hours + (monthly_freestyle_sidestroke_hours * months)

def remaining_hours_to_breaststroke : ℕ :=
  total_required_hours - calculated_total_hours

theorem breaststroke_hours_correct :
  remaining_hours_to_breaststroke = 9 :=
by
  sorry

end Swimming

end NUMINAMATH_GPT_breaststroke_hours_correct_l2117_211749


namespace NUMINAMATH_GPT_cost_of_dozen_pens_l2117_211754

theorem cost_of_dozen_pens
  (cost_three_pens_five_pencils : ℝ)
  (cost_one_pen : ℝ)
  (pen_to_pencil_ratio : ℝ)
  (h1 : 3 * cost_one_pen + 5 * (cost_three_pens_five_pencils / 8) = 260)
  (h2 : cost_one_pen = 65)
  (h3 : cost_one_pen / (cost_three_pens_five_pencils / 8) = 5/1)
  : 12 * cost_one_pen = 780 := by
    sorry

end NUMINAMATH_GPT_cost_of_dozen_pens_l2117_211754


namespace NUMINAMATH_GPT_single_elimination_games_needed_l2117_211779

theorem single_elimination_games_needed (n : ℕ) (n_pos : n > 0) :
  (number_of_games_needed : ℕ) = n - 1 :=
by
  sorry

end NUMINAMATH_GPT_single_elimination_games_needed_l2117_211779


namespace NUMINAMATH_GPT_find_other_number_l2117_211703

theorem find_other_number (lcm_ab hcf_ab : ℕ) (A : ℕ) (h_lcm: Nat.lcm A (B) = lcm_ab)
  (h_hcf : Nat.gcd A (B) = hcf_ab) (h_a : A = 48) (h_lcm_value: lcm_ab = 192) (h_hcf_value: hcf_ab = 16) :
  B = 64 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l2117_211703


namespace NUMINAMATH_GPT_sum_of_first_half_of_numbers_l2117_211701

theorem sum_of_first_half_of_numbers 
  (avg_total : ℝ) 
  (total_count : ℕ) 
  (avg_second_half : ℝ) 
  (sum_total : ℝ)
  (sum_second_half : ℝ)
  (sum_first_half : ℝ) 
  (h1 : total_count = 8)
  (h2 : avg_total = 43.1)
  (h3 : avg_second_half = 46.6)
  (h4 : sum_total = avg_total * total_count)
  (h5 : sum_second_half = 4 * avg_second_half)
  (h6 : sum_first_half = sum_total - sum_second_half)
  :
  sum_first_half = 158.4 := 
sorry

end NUMINAMATH_GPT_sum_of_first_half_of_numbers_l2117_211701


namespace NUMINAMATH_GPT_largest_expression_l2117_211770

noncomputable def x : ℝ := 10 ^ (-2024 : ℤ)

theorem largest_expression :
  let a := 5 + x
  let b := 5 - x
  let c := 5 * x
  let d := 5 / x
  let e := x / 5
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by
  sorry

end NUMINAMATH_GPT_largest_expression_l2117_211770


namespace NUMINAMATH_GPT_fencing_cost_per_foot_is_3_l2117_211742

-- Definitions of the constants given in the problem
def side_length : ℕ := 9
def back_length : ℕ := 18
def total_cost : ℕ := 72
def neighbor_behind_rate : ℚ := 1/2
def neighbor_left_rate : ℚ := 1/3

-- The statement to be proved
theorem fencing_cost_per_foot_is_3 : 
  (total_cost / ((2 * side_length + back_length) - 
                (neighbor_behind_rate * back_length) -
                (neighbor_left_rate * side_length))) = 3 := 
by
  sorry

end NUMINAMATH_GPT_fencing_cost_per_foot_is_3_l2117_211742


namespace NUMINAMATH_GPT_find_a_l2117_211721

noncomputable def hyperbola_eccentricity (a : ℝ) : ℝ := (Real.sqrt (a^2 + 3)) / a

theorem find_a (a : ℝ) (h : a > 0) (hexp : hyperbola_eccentricity a = 2) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2117_211721


namespace NUMINAMATH_GPT_profit_without_discount_l2117_211750

noncomputable def cost_price : ℝ := 100
noncomputable def profit_percentage_with_discount : ℝ := 44
noncomputable def discount : ℝ := 4

theorem profit_without_discount (CP MP SP : ℝ) (h_CP : CP = cost_price) (h_pwpd : profit_percentage_with_discount = 44) (h_discount : discount = 4) (h_SP : SP = CP * (1 + profit_percentage_with_discount / 100)) (h_MP : SP = MP * (1 - discount / 100)) :
  ((MP - CP) / CP * 100) = 50 :=
by
  sorry

end NUMINAMATH_GPT_profit_without_discount_l2117_211750


namespace NUMINAMATH_GPT_sum_of_numbers_facing_up_is_4_probability_l2117_211735

-- Definition of a uniform dice with faces numbered 1 to 6
def dice_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Definition of the sample space when the dice is thrown twice
def sample_space : Finset (ℕ × ℕ) := Finset.product dice_faces dice_faces

-- Definition of the event where the sum of the numbers is 4
def event_sum_4 : Finset (ℕ × ℕ) := sample_space.filter (fun pair => pair.1 + pair.2 = 4)

-- The number of favorable outcomes
def favorable_outcomes : ℕ := event_sum_4.card

-- The total number of possible outcomes
def total_outcomes : ℕ := sample_space.card

-- The probability of the event
def probability_event_sum_4 : ℚ := favorable_outcomes / total_outcomes

theorem sum_of_numbers_facing_up_is_4_probability :
  probability_event_sum_4 = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_facing_up_is_4_probability_l2117_211735


namespace NUMINAMATH_GPT_min_cost_to_win_l2117_211773

theorem min_cost_to_win (n : ℕ) : 
  (∀ m : ℕ, m = 0 →
  (∀ cents : ℕ, 
  (n = 5 * m ∨ n = m + 1) ∧ n > 2008 ∧ n % 100 = 42 → 
  cents = 35)) :=
sorry

end NUMINAMATH_GPT_min_cost_to_win_l2117_211773


namespace NUMINAMATH_GPT_find_integer_n_l2117_211704

theorem find_integer_n (n : ℤ) : (⌊(n^2 : ℤ) / 4⌋ - ⌊n / 2⌋ ^ 2 = 3) → n = 7 :=
by sorry

end NUMINAMATH_GPT_find_integer_n_l2117_211704


namespace NUMINAMATH_GPT_typhoon_tree_survival_l2117_211781

def planted_trees : Nat := 150
def died_trees : Nat := 92
def slightly_damaged_trees : Nat := 15

def total_trees_affected : Nat := died_trees + slightly_damaged_trees
def trees_survived_without_damages : Nat := planted_trees - total_trees_affected
def more_died_than_survived : Nat := died_trees - trees_survived_without_damages

theorem typhoon_tree_survival :
  more_died_than_survived = 49 :=
by
  -- Define the necessary computations and assertions
  let total_trees_affected := 92 + 15
  let trees_survived_without_damages := 150 - total_trees_affected
  let more_died_than_survived := 92 - trees_survived_without_damages
  -- Prove the statement
  have : total_trees_affected = 107 := rfl
  have : trees_survived_without_damages = 43 := rfl
  have : more_died_than_survived = 49 := rfl
  exact this

end NUMINAMATH_GPT_typhoon_tree_survival_l2117_211781


namespace NUMINAMATH_GPT_ratio_of_second_to_first_l2117_211711

theorem ratio_of_second_to_first:
  ∀ (x y z : ℕ), 
  (y = 90) → 
  (z = 4 * y) → 
  ((x + y + z) / 3 = 165) → 
  (y / x = 2) := 
by 
  intros x y z h1 h2 h3
  sorry

end NUMINAMATH_GPT_ratio_of_second_to_first_l2117_211711


namespace NUMINAMATH_GPT_ratio_proof_l2117_211796

theorem ratio_proof (a b c : ℝ) (h1 : b / a = 4) (h2 : c / b = 5) : (a + 2 * b) / (3 * b + c) = 9 / 32 :=
by
  sorry

end NUMINAMATH_GPT_ratio_proof_l2117_211796


namespace NUMINAMATH_GPT_probability_distribution_xi_l2117_211728

theorem probability_distribution_xi (a : ℝ) (ξ : ℕ → ℝ) (h1 : ξ 1 = a / (1 * 2))
  (h2 : ξ 2 = a / (2 * 3)) (h3 : ξ 3 = a / (3 * 4)) (h4 : ξ 4 = a / (4 * 5))
  (h5 : (ξ 1) + (ξ 2) + (ξ 3) + (ξ 4) = 1) :
  ξ 1 + ξ 2 = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_probability_distribution_xi_l2117_211728


namespace NUMINAMATH_GPT_initial_average_is_16_l2117_211733

def average_of_six_observations (A : ℝ) : Prop :=
  ∃ s : ℝ, s = 6 * A

def new_observation (A : ℝ) (new_obs : ℝ := 9) : Prop :=
  ∃ t : ℝ, t = 7 * (A - 1)

theorem initial_average_is_16 (A : ℝ) (new_obs : ℝ := 9) :
  (average_of_six_observations A) → (new_observation A new_obs) → A = 16 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_initial_average_is_16_l2117_211733


namespace NUMINAMATH_GPT_minimum_value_abs_a_plus_2_abs_b_l2117_211775

open Real

theorem minimum_value_abs_a_plus_2_abs_b 
  (a b : ℝ)
  (f : ℝ → ℝ)
  (x₁ x₂ x₃ : ℝ)
  (f_def : ∀ x, f x = x^3 + a*x^2 + b*x)
  (roots_cond : x₁ + 1 ≤ x₂ ∧ x₂ ≤ x₃ - 1)
  (equal_values : f x₁ = f x₂ ∧ f x₂ = f x₃) :
  ∃ minimum, minimum = (sqrt 3) ∧ (∀ (a b : ℝ), |a| + 2*|b| ≥ sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_abs_a_plus_2_abs_b_l2117_211775


namespace NUMINAMATH_GPT_parts_per_hour_l2117_211730

theorem parts_per_hour (x y : ℝ) (h₁ : 90 / x = 120 / y) (h₂ : x + y = 35) : x = 15 ∧ y = 20 :=
by
  sorry

end NUMINAMATH_GPT_parts_per_hour_l2117_211730


namespace NUMINAMATH_GPT_initial_workers_l2117_211799

theorem initial_workers (W : ℕ) (work1 : ℕ) (work2 : ℕ) :
  (work1 = W * 8 * 30) →
  (work2 = (W + 35) * 6 * 40) →
  (work1 / 30 = work2 / 40) →
  W = 105 :=
by
  intros hwork1 hwork2 hprop
  sorry

end NUMINAMATH_GPT_initial_workers_l2117_211799


namespace NUMINAMATH_GPT_polynomial_solution_l2117_211765

theorem polynomial_solution (x : ℝ) (h : (2 * x - 1) ^ 2 = 9) : x = 2 ∨ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_solution_l2117_211765


namespace NUMINAMATH_GPT_number_exceeds_twenty_percent_by_forty_l2117_211741

theorem number_exceeds_twenty_percent_by_forty (x : ℝ) (h : x = 0.20 * x + 40) : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_number_exceeds_twenty_percent_by_forty_l2117_211741


namespace NUMINAMATH_GPT_non_negative_dot_product_l2117_211757

theorem non_negative_dot_product
  (a b c d e f g h : ℝ) :
  (a * c + b * d ≥ 0) ∨ (a * e + b * f ≥ 0) ∨ (a * g + b * h ≥ 0) ∨
  (c * e + d * f ≥ 0) ∨ (c * g + d * h ≥ 0) ∨ (e * g + f * h ≥ 0) :=
sorry

end NUMINAMATH_GPT_non_negative_dot_product_l2117_211757


namespace NUMINAMATH_GPT_sum_of_legs_of_similar_larger_triangle_l2117_211797

-- Define the conditions for the problem
def smaller_triangle_area : ℝ := 10
def larger_triangle_area : ℝ := 400
def smaller_triangle_hypotenuse : ℝ := 10

-- Define the correct answer (sum of the lengths of the legs of the larger triangle)
def sum_of_legs_of_larger_triangle : ℝ := 88.55

-- State the Lean theorem
theorem sum_of_legs_of_similar_larger_triangle :
  (∀ (A B C a b c : ℝ), 
    a * b / 2 = smaller_triangle_area ∧ 
    c = smaller_triangle_hypotenuse ∧
    C * C / 4 = larger_triangle_area / smaller_triangle_area ∧
    A / a = B / b ∧ 
    A^2 + B^2 = C^2 → 
    A + B = sum_of_legs_of_larger_triangle) :=
  by sorry

end NUMINAMATH_GPT_sum_of_legs_of_similar_larger_triangle_l2117_211797


namespace NUMINAMATH_GPT_absolute_value_of_h_l2117_211725

theorem absolute_value_of_h {h : ℝ} :
  (∀ x : ℝ, (x^2 + 2 * h * x = 3) → (∃ r s : ℝ, r + s = -2 * h ∧ r * s = -3 ∧ r^2 + s^2 = 10)) →
  |h| = 1 :=
by
  sorry

end NUMINAMATH_GPT_absolute_value_of_h_l2117_211725


namespace NUMINAMATH_GPT_octahedron_plane_pairs_l2117_211762

-- A regular octahedron has 12 edges.
def edges_octahedron : ℕ := 12

-- Each edge determines a plane with 8 other edges.
def pairs_with_each_edge : ℕ := 8

-- The number of unordered pairs of edges that determine a plane
theorem octahedron_plane_pairs : (edges_octahedron * pairs_with_each_edge) / 2 = 48 :=
by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_octahedron_plane_pairs_l2117_211762


namespace NUMINAMATH_GPT_missing_fraction_l2117_211794

-- Definitions for the given fractions
def a := 1 / 3
def b := 1 / 2
def c := 1 / 5
def d := 1 / 4
def e := -9 / 20
def f := -2 / 15
def target_sum := 2 / 15 -- because 0.13333333333333333 == 2 / 15

-- Main theorem statement for the problem
theorem missing_fraction : a + b + c + d + e + f + -17 / 30 = target_sum :=
by
  simp [a, b, c, d, e, f, target_sum]
  sorry

end NUMINAMATH_GPT_missing_fraction_l2117_211794


namespace NUMINAMATH_GPT_p_arithmetic_fibonacci_term_correct_l2117_211706

noncomputable def p_arithmetic_fibonacci_term (p : ℕ) : ℝ :=
  5 ^ ((p - 1) / 2)

theorem p_arithmetic_fibonacci_term_correct (p : ℕ) : p_arithmetic_fibonacci_term p = 5 ^ ((p - 1) / 2) := 
by 
  rfl -- direct application of the definition

#check p_arithmetic_fibonacci_term_correct

end NUMINAMATH_GPT_p_arithmetic_fibonacci_term_correct_l2117_211706


namespace NUMINAMATH_GPT_find_sp_l2117_211705

theorem find_sp (s p : ℝ) (t x y : ℝ) (h1 : x = 3 + 5 * t) (h2 : y = 3 + p * t) 
  (h3 : y = 4 * x - 9) : 
  s = 3 ∧ p = 20 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_sp_l2117_211705


namespace NUMINAMATH_GPT_plates_count_l2117_211764

theorem plates_count (n : ℕ)
  (h1 : 500 < n)
  (h2 : n < 600)
  (h3 : n % 10 = 7)
  (h4 : n % 12 = 7) : n = 547 :=
sorry

end NUMINAMATH_GPT_plates_count_l2117_211764


namespace NUMINAMATH_GPT_adam_books_l2117_211778

theorem adam_books (before_books total_shelves books_per_shelf after_books leftover_books bought_books : ℕ)
  (h_before: before_books = 56)
  (h_shelves: total_shelves = 4)
  (h_books_per_shelf: books_per_shelf = 20)
  (h_leftover: leftover_books = 2)
  (h_after: after_books = (total_shelves * books_per_shelf) + leftover_books)
  (h_difference: bought_books = after_books - before_books) :
  bought_books = 26 :=
by
  sorry

end NUMINAMATH_GPT_adam_books_l2117_211778


namespace NUMINAMATH_GPT_Kim_has_4_cousins_l2117_211734

noncomputable def pieces_per_cousin : ℕ := 5
noncomputable def total_pieces : ℕ := 20
noncomputable def cousins : ℕ := total_pieces / pieces_per_cousin

theorem Kim_has_4_cousins : cousins = 4 := 
by
  show cousins = 4
  sorry

end NUMINAMATH_GPT_Kim_has_4_cousins_l2117_211734


namespace NUMINAMATH_GPT_prove_inequality_l2117_211702

noncomputable def inequality_problem :=
  ∀ (x y z : ℝ),
    0 < x ∧ 0 < y ∧ 0 < z ∧ x^2 + y^2 + z^2 = 3 → 
      (x ^ 2009 - 2008 * (x - 1)) / (y + z) + 
      (y ^ 2009 - 2008 * (y - 1)) / (x + z) + 
      (z ^ 2009 - 2008 * (z - 1)) / (x + y) ≥ 
      (x + y + z) / 2

theorem prove_inequality : inequality_problem := 
  by 
    sorry

end NUMINAMATH_GPT_prove_inequality_l2117_211702


namespace NUMINAMATH_GPT_scientific_notation_correct_l2117_211789

def original_number : ℕ := 31900

def scientific_notation_option_A : ℝ := 3.19 * 10^2
def scientific_notation_option_B : ℝ := 0.319 * 10^3
def scientific_notation_option_C : ℝ := 3.19 * 10^4
def scientific_notation_option_D : ℝ := 0.319 * 10^5

theorem scientific_notation_correct :
  original_number = 31900 ∧ scientific_notation_option_C = 3.19 * 10^4 ∧ (original_number : ℝ) = scientific_notation_option_C := 
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l2117_211789


namespace NUMINAMATH_GPT_ryan_learning_hours_l2117_211763

theorem ryan_learning_hours :
  ∃ hours : ℕ, 
    (∀ e_hrs : ℕ, e_hrs = 2) → 
    (∃ c_hrs : ℕ, c_hrs = hours) → 
    (∀ s_hrs : ℕ, s_hrs = 4) → 
    hours = 4 + 1 :=
by
  sorry

end NUMINAMATH_GPT_ryan_learning_hours_l2117_211763


namespace NUMINAMATH_GPT_compute_five_fold_application_l2117_211755

def f (x : ℤ) : ℤ :=
  if x >= 0 then -(x^3) else x + 10

theorem compute_five_fold_application : f (f (f (f (f 2)))) = -8 := by
  sorry

end NUMINAMATH_GPT_compute_five_fold_application_l2117_211755


namespace NUMINAMATH_GPT_determine_p_l2117_211751

theorem determine_p (p x1 x2 : ℝ) 
  (h_eq : ∀ x, x^2 + p * x + 3 = 0)
  (h_root_relation : x2 = 3 * x1)
  (h_vieta1 : x1 + x2 = -p)
  (h_vieta2 : x1 * x2 = 3) :
  p = 4 ∨ p = -4 := 
sorry

end NUMINAMATH_GPT_determine_p_l2117_211751


namespace NUMINAMATH_GPT_bridge_crossing_possible_l2117_211784

/-- 
  There are four people A, B, C, and D. 
  The time it takes for each of them to cross the bridge is 2, 4, 6, and 8 minutes respectively.
  No more than two people can be on the bridge at the same time.
  Prove that it is possible for all four people to cross the bridge in 10 minutes.
--/
theorem bridge_crossing_possible : 
  ∃ (cross : ℕ → ℕ), 
  cross 1 = 2 ∧ cross 2 = 4 ∧ cross 3 = 6 ∧ cross 4 = 8 ∧
  (∀ (t : ℕ), t ≤ 2 → cross 1 + cross 2 + cross 3 + cross 4 = 10) :=
by
  sorry

end NUMINAMATH_GPT_bridge_crossing_possible_l2117_211784


namespace NUMINAMATH_GPT_evaluate_expression_l2117_211793

theorem evaluate_expression : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3 / 3) := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2117_211793


namespace NUMINAMATH_GPT_phantom_additional_money_needed_l2117_211737

theorem phantom_additional_money_needed
  (given_money : ℕ)
  (black_inks_cost : ℕ)
  (red_inks_cost : ℕ)
  (yellow_inks_cost : ℕ)
  (blue_inks_cost : ℕ)
  (total_money_needed : ℕ)
  (additional_money_needed : ℕ) :
  given_money = 50 →
  black_inks_cost = 3 * 12 →
  red_inks_cost = 4 * 16 →
  yellow_inks_cost = 3 * 14 →
  blue_inks_cost = 2 * 17 →
  total_money_needed = black_inks_cost + red_inks_cost + yellow_inks_cost + blue_inks_cost →
  additional_money_needed = total_money_needed - given_money →
  additional_money_needed = 126 :=
by
  intros h_given_money h_black h_red h_yellow h_blue h_total h_additional
  sorry

end NUMINAMATH_GPT_phantom_additional_money_needed_l2117_211737
