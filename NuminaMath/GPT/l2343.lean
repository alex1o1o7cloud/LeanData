import Mathlib

namespace NUMINAMATH_GPT_identical_digits_has_37_factor_l2343_234306

theorem identical_digits_has_37_factor (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) : 37 ∣ (100 * a + 10 * a + a) :=
by
  sorry

end NUMINAMATH_GPT_identical_digits_has_37_factor_l2343_234306


namespace NUMINAMATH_GPT_initial_percentage_of_water_is_12_l2343_234312

noncomputable def initial_percentage_of_water (initial_volume : ℕ) (added_water : ℕ) (final_percentage : ℕ) : ℕ :=
  let final_volume := initial_volume + added_water
  let final_water_amount := (final_percentage * final_volume) / 100
  let initial_water_amount := final_water_amount - added_water
  (initial_water_amount * 100) / initial_volume

theorem initial_percentage_of_water_is_12 :
  initial_percentage_of_water 20 2 20 = 12 :=
by
  sorry

end NUMINAMATH_GPT_initial_percentage_of_water_is_12_l2343_234312


namespace NUMINAMATH_GPT_volume_after_increasing_edges_l2343_234365

-- Defining the initial conditions and the theorem to prove regarding the volume.
theorem volume_after_increasing_edges {a b c : ℝ} 
  (h1 : a * b * c = 8) 
  (h2 : (a + 1) * (b + 1) * (c + 1) = 27) : 
  (a + 2) * (b + 2) * (c + 2) = 64 :=
sorry

end NUMINAMATH_GPT_volume_after_increasing_edges_l2343_234365


namespace NUMINAMATH_GPT_range_of_a_for_monotonically_decreasing_function_l2343_234395

theorem range_of_a_for_monotonically_decreasing_function {a : ℝ} :
    (∀ x y : ℝ, (x > 2 → y > 2 → (ax^2 + x - 1) ≤ (a*y^2 + y - 1)) ∧
                (x ≤ 2 → y ≤ 2 → (-x + 1) ≤ (-y + 1)) ∧
                (x > 2 → y ≤ 2 → (ax^2 + x - 1) ≤ (-y + 1)) ∧
                (x ≤ 2 → y > 2 → (-x + 1) ≤ (a*y^2 + y - 1))) →
    (a < 0 ∧ - (1 / (2 * a)) ≤ 2 ∧ 4 * a + 1 ≤ -1) →
    a ≤ -1 / 2 :=
by
  intro hmonotone hconditions
  sorry

end NUMINAMATH_GPT_range_of_a_for_monotonically_decreasing_function_l2343_234395


namespace NUMINAMATH_GPT_total_points_scored_l2343_234310

theorem total_points_scored (m1 m2 m3 m4 m5 m6 j1 j2 j3 j4 j5 j6 : ℕ) :
  m1 = 5 → j1 = m1 + 2 →
  m2 = 7 → j2 = m2 - 3 →
  m3 = 10 → j3 = m3 / 2 →
  m4 = 12 → j4 = m4 * 2 →
  m5 = 6 → j5 = m5 →
  j6 = 8 → m6 = j6 + 4 →
  m1 + m2 + m3 + m4 + m5 + m6 + j1 + j2 + j3 + j4 + j5 + j6 = 106 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_points_scored_l2343_234310


namespace NUMINAMATH_GPT_time_A_to_complete_work_alone_l2343_234383

theorem time_A_to_complete_work_alone :
  ∃ (x : ℝ), (1 / x) + (1 / 20) = (1 / 8.571428571428571) ∧ x = 15 :=
by
  sorry

end NUMINAMATH_GPT_time_A_to_complete_work_alone_l2343_234383


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_and_sum_max_l2343_234326

-- Definitions and conditions
def a1 : ℤ := 4
def d : ℤ := -2
def a (n : ℕ) : ℤ := a1 + (n - 1) * d
def Sn (n : ℕ) : ℤ := n * (a1 + (a n)) / 2

-- Prove the general term formula and maximum value
theorem arithmetic_sequence_general_term_and_sum_max :
  (∀ n, a n = -2 * n + 6) ∧ (∃ n, Sn n = 6) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_and_sum_max_l2343_234326


namespace NUMINAMATH_GPT_book_donation_growth_rate_l2343_234355

theorem book_donation_growth_rate (x : ℝ) : 
  400 + 400 * (1 + x) + 400 * (1 + x)^2 = 1525 :=
sorry

end NUMINAMATH_GPT_book_donation_growth_rate_l2343_234355


namespace NUMINAMATH_GPT_problem_equivalent_l2343_234305

theorem problem_equivalent :
  500 * 2019 * 0.0505 * 20 = 2019^2 :=
by
  sorry

end NUMINAMATH_GPT_problem_equivalent_l2343_234305


namespace NUMINAMATH_GPT_lakshmi_share_annual_gain_l2343_234321

theorem lakshmi_share_annual_gain (x : ℝ) (annual_gain : ℝ) (Raman_inv_months : ℝ) (Lakshmi_inv_months : ℝ) (Muthu_inv_months : ℝ) (Gowtham_inv_months : ℝ) (Pradeep_inv_months : ℝ)
  (total_inv_months : ℝ) (lakshmi_share : ℝ) :
  Raman_inv_months = x * 12 →
  Lakshmi_inv_months = 2 * x * 6 →
  Muthu_inv_months = 3 * x * 4 →
  Gowtham_inv_months = 4 * x * 9 →
  Pradeep_inv_months = 5 * x * 1 →
  total_inv_months = Raman_inv_months + Lakshmi_inv_months + Muthu_inv_months + Gowtham_inv_months + Pradeep_inv_months →
  annual_gain = 58000 →
  lakshmi_share = (Lakshmi_inv_months / total_inv_months) * annual_gain →
  lakshmi_share = 9350.65 :=
by
  sorry

end NUMINAMATH_GPT_lakshmi_share_annual_gain_l2343_234321


namespace NUMINAMATH_GPT_fraction_zero_condition_l2343_234330

theorem fraction_zero_condition (x : ℝ) (h : (abs x - 2) / (2 - x) = 0) : x = -2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_zero_condition_l2343_234330


namespace NUMINAMATH_GPT_stack_map_A_front_view_l2343_234356

def column1 : List ℕ := [3, 1]
def column2 : List ℕ := [2, 2, 1]
def column3 : List ℕ := [1, 4, 2]
def column4 : List ℕ := [5]

def tallest (l : List ℕ) : ℕ :=
  l.foldl max 0

theorem stack_map_A_front_view :
  [tallest column1, tallest column2, tallest column3, tallest column4] = [3, 2, 4, 5] := by
  sorry

end NUMINAMATH_GPT_stack_map_A_front_view_l2343_234356


namespace NUMINAMATH_GPT_A_work_days_l2343_234396

variables (r_A r_B r_C : ℝ) (h1 : r_A + r_B = (1 / 3)) (h2 : r_B + r_C = (1 / 3)) (h3 : r_A + r_C = (5 / 24))

theorem A_work_days :
  1 / r_A = 9.6 := 
sorry

end NUMINAMATH_GPT_A_work_days_l2343_234396


namespace NUMINAMATH_GPT_largest_of_five_numbers_l2343_234337

theorem largest_of_five_numbers : ∀ (a b c d e : ℝ), 
  a = 0.938 → b = 0.9389 → c = 0.93809 → d = 0.839 → e = 0.893 → b = max a (max b (max c (max d e))) :=
by
  intros a b c d e ha hb hc hd he
  rw [ha, hb, hc, hd, he]
  sorry

end NUMINAMATH_GPT_largest_of_five_numbers_l2343_234337


namespace NUMINAMATH_GPT_point_to_line_distance_l2343_234362

theorem point_to_line_distance :
  let circle_center : ℝ×ℝ := (0, 1)
  let A : ℝ := -1
  let B : ℝ := 1
  let C : ℝ := -2
  let line_eq (x y : ℝ) := A * x + B * y + C == 0
  ∀ (x0 : ℝ) (y0 : ℝ),
    circle_center = (x0, y0) →
    (|A * x0 + B * y0 + C| / Real.sqrt (A^2 + B^2)) = (Real.sqrt 2 / 2) := 
by 
  intros
  -- Proof goes here
  sorry -- Placeholder for the proof.

end NUMINAMATH_GPT_point_to_line_distance_l2343_234362


namespace NUMINAMATH_GPT_flash_catches_ace_l2343_234309

theorem flash_catches_ace (v : ℝ) (x : ℝ) (y : ℝ) (hx : x > 1) :
  let t := y / (v * (x - 1))
  let ace_distance := v * t
  let flash_distance := x * v * t
  flash_distance = (xy / (x - 1)) :=
by
  let t := y / (v * (x - 1))
  let ace_distance := v * t
  let flash_distance := x * v * t
  have h1 : x * v * t = xy / (x - 1) := sorry
  exact h1

end NUMINAMATH_GPT_flash_catches_ace_l2343_234309


namespace NUMINAMATH_GPT_complex_exponential_sum_identity_l2343_234301

theorem complex_exponential_sum_identity :
    12 * Complex.exp (Real.pi * Complex.I / 7) + 12 * Complex.exp (19 * Real.pi * Complex.I / 14) =
    24 * Real.cos (5 * Real.pi / 28) * Complex.exp (3 * Real.pi * Complex.I / 4) :=
sorry

end NUMINAMATH_GPT_complex_exponential_sum_identity_l2343_234301


namespace NUMINAMATH_GPT_paint_cans_for_25_rooms_l2343_234324

theorem paint_cans_for_25_rooms (cans rooms : ℕ) (H1 : cans * 30 = rooms) (H2 : cans * 25 = rooms - 5 * cans) :
  cans = 15 :=
by
  sorry

end NUMINAMATH_GPT_paint_cans_for_25_rooms_l2343_234324


namespace NUMINAMATH_GPT_range_of_years_of_service_l2343_234347

theorem range_of_years_of_service : 
  let years := [15, 10, 9, 17, 6, 3, 14, 16]
  ∃ min max, (min ∈ years ∧ max ∈ years ∧ (max - min = 14)) :=
by 
  let years := [15, 10, 9, 17, 6, 3, 14, 16]
  use 3, 17 
  sorry

end NUMINAMATH_GPT_range_of_years_of_service_l2343_234347


namespace NUMINAMATH_GPT_part1_part2_l2343_234350

-- Part 1: Inequality solution
theorem part1 (x : ℝ) :
  (1 / 3 * x - (3 * x + 4) / 6 ≤ 2 / 3) → (x ≥ -8) := 
by
  intro h
  sorry

-- Part 2: System of inequalities solution
theorem part2 (x : ℝ) :
  (4 * (x + 1) ≤ 7 * x + 13) ∧ ((x + 2) / 3 - x / 2 > 1) → (-3 ≤ x ∧ x < -2) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_part1_part2_l2343_234350


namespace NUMINAMATH_GPT_max_candies_takeable_l2343_234344

theorem max_candies_takeable : 
  ∃ (max_take : ℕ), max_take = 159 ∧
  ∀ (boxes: Fin 5 → ℕ), 
    boxes 0 = 11 → 
    boxes 1 = 22 → 
    boxes 2 = 33 → 
    boxes 3 = 44 → 
    boxes 4 = 55 →
    (∀ (i : Fin 5), 
      ∀ (new_boxes : Fin 5 → ℕ),
      (new_boxes i = boxes i - 4) ∧ 
      (∀ (j : Fin 5), j ≠ i → new_boxes j = boxes j + 1) →
      boxes i = 0 → max_take = new_boxes i) :=
sorry

end NUMINAMATH_GPT_max_candies_takeable_l2343_234344


namespace NUMINAMATH_GPT_max_african_team_wins_max_l2343_234382

-- Assume there are n African teams and (n + 9) European teams.
-- Each pair of teams plays exactly once.
-- European teams won nine times as many matches as African teams.
-- Prove that the maximum number of matches that a single African team might have won is 11.

theorem max_african_team_wins_max (n : ℕ) (k : ℕ) (n_african_wins : ℕ) (n_european_wins : ℕ)
  (h1 : n_african_wins = (n * (n - 1)) / 2) 
  (h2 : n_european_wins = ((n + 9) * (n + 8)) / 2 + k)
  (h3 : n_european_wins = 9 * (n_african_wins + (n * (n + 9) - k))) :
  ∃ max_wins, max_wins = 11 := by
  sorry

end NUMINAMATH_GPT_max_african_team_wins_max_l2343_234382


namespace NUMINAMATH_GPT_percentage_x_minus_y_l2343_234381

variable (x y : ℝ)

theorem percentage_x_minus_y (P : ℝ) :
  P / 100 * (x - y) = 20 / 100 * (x + y) ∧ y = 20 / 100 * x → P = 30 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_percentage_x_minus_y_l2343_234381


namespace NUMINAMATH_GPT_domain_of_f_l2343_234368

noncomputable def f (x : ℝ) := Real.log x / Real.log 6

noncomputable def g (x : ℝ) := Real.log x / Real.log 5

noncomputable def h (x : ℝ) := Real.log x / Real.log 3

open Set

theorem domain_of_f :
  (∀ x, x > 7776 → ∃ y, y = (h ∘ g ∘ f) x) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l2343_234368


namespace NUMINAMATH_GPT_number_is_18_l2343_234314

theorem number_is_18 (x : ℝ) (h : (7 / 3) * x = 42) : x = 18 :=
sorry

end NUMINAMATH_GPT_number_is_18_l2343_234314


namespace NUMINAMATH_GPT_Kiera_envelopes_total_l2343_234385

-- Define variables for different colored envelopes
def E_b : ℕ := 120
def E_y : ℕ := E_b - 25
def E_g : ℕ := 5 * E_y
def E_r : ℕ := (E_b + E_y) / 2  -- integer division in lean automatically rounds down
def E_p : ℕ := E_r + 71
def E_total : ℕ := E_b + E_y + E_g + E_r + E_p

-- The statement to be proven
theorem Kiera_envelopes_total : E_total = 975 := by
  -- intentionally put the sorry to mark the proof as unfinished
  sorry

end NUMINAMATH_GPT_Kiera_envelopes_total_l2343_234385


namespace NUMINAMATH_GPT_probability_one_from_each_l2343_234327

-- Define the total number of cards
def total_cards : ℕ := 10

-- Define the number of cards from Amelia's name
def amelia_cards : ℕ := 6

-- Define the number of cards from Lucas's name
def lucas_cards : ℕ := 4

-- Define the probability that one letter is from each person's name
theorem probability_one_from_each : (amelia_cards / total_cards) * (lucas_cards / (total_cards - 1)) +
                                    (lucas_cards / total_cards) * (amelia_cards / (total_cards - 1)) = 8 / 15 :=
by
  sorry

end NUMINAMATH_GPT_probability_one_from_each_l2343_234327


namespace NUMINAMATH_GPT_statement_A_statement_D_l2343_234313

variable (a b c d : ℝ)

-- Statement A: If ac² > bc², then a > b
theorem statement_A (h1 : a * c^2 > b * c^2) (h2 : c ≠ 0) : a > b := by
  sorry

-- Statement D: If a > b > 0, then a + 1/b > b + 1/a
theorem statement_D (h1 : a > b) (h2 : b > 0) : a + 1 / b > b + 1 / a := by
  sorry

end NUMINAMATH_GPT_statement_A_statement_D_l2343_234313


namespace NUMINAMATH_GPT_Liz_team_deficit_l2343_234367

theorem Liz_team_deficit :
  ∀ (initial_deficit liz_free_throws liz_three_pointers liz_jump_shots opponent_points : ℕ),
    initial_deficit = 20 →
    liz_free_throws = 5 →
    liz_three_pointers = 3 →
    liz_jump_shshots = 4 →
    opponent_points = 10 →
    (initial_deficit - (liz_free_throws * 1 + liz_three_pointers * 3 + liz_jump_shshots * 2 - opponent_points)) = 8 := by
  intros initial_deficit liz_free_throws liz_three_pointers liz_jump_shots opponent_points
  intros h_initial_deficit h_liz_free_throws h_liz_three_pointers h_liz_jump_shots h_opponent_points
  sorry

end NUMINAMATH_GPT_Liz_team_deficit_l2343_234367


namespace NUMINAMATH_GPT_value_of_f_at_pi_over_12_l2343_234360

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - ω * Real.pi)

theorem value_of_f_at_pi_over_12 (ω : ℝ) (hω_pos : ω > 0) 
(h_period : ∀ x, f ω (x + Real.pi) = f ω x) : 
  f ω (Real.pi / 12) = 1 / 2 := 
sorry

end NUMINAMATH_GPT_value_of_f_at_pi_over_12_l2343_234360


namespace NUMINAMATH_GPT_volume_ratio_l2343_234393

theorem volume_ratio (a b : ℝ) (h : a^2 / b^2 = 9 / 25) : b^3 / a^3 = 125 / 27 :=
by
  -- Skipping the proof by adding 'sorry'
  sorry

end NUMINAMATH_GPT_volume_ratio_l2343_234393


namespace NUMINAMATH_GPT_initial_hamburgers_correct_l2343_234379

-- Define the initial problem conditions
def initial_hamburgers (H : ℝ) : Prop := H + 3.0 = 12

-- State the proof problem
theorem initial_hamburgers_correct (H : ℝ) (h : initial_hamburgers H) : H = 9.0 :=
sorry

end NUMINAMATH_GPT_initial_hamburgers_correct_l2343_234379


namespace NUMINAMATH_GPT_total_pens_l2343_234372

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end NUMINAMATH_GPT_total_pens_l2343_234372


namespace NUMINAMATH_GPT_find_a_l2343_234398
-- Import necessary Lean libraries

-- Define the function and its maximum value condition
def f (a x : ℝ) := -x^2 + 2*a*x + 1 - a

def has_max_value (f : ℝ → ℝ) (M : ℝ) (interval : Set ℝ) : Prop :=
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = M

theorem find_a (a : ℝ) :
  has_max_value (f a) 2 (Set.Icc 0 1) → (a = -1 ∨ a = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2343_234398


namespace NUMINAMATH_GPT_factorize_a_cubed_minus_a_l2343_234302

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end NUMINAMATH_GPT_factorize_a_cubed_minus_a_l2343_234302


namespace NUMINAMATH_GPT_passengers_got_off_l2343_234363

theorem passengers_got_off :
  ∀ (initial_boarded new_boarded final_left got_off : ℕ),
    initial_boarded = 28 →
    new_boarded = 7 →
    final_left = 26 →
    got_off = initial_boarded + new_boarded - final_left →
    got_off = 9 :=
by
  intros initial_boarded new_boarded final_left got_off h_initial h_new h_final h_got_off
  rw [h_initial, h_new, h_final] at h_got_off
  exact h_got_off

end NUMINAMATH_GPT_passengers_got_off_l2343_234363


namespace NUMINAMATH_GPT_z_share_per_rupee_x_l2343_234397

-- Definitions according to the conditions
def x_gets (r : ℝ) : ℝ := r
def y_gets_for_x (r : ℝ) : ℝ := 0.45 * r
def y_share : ℝ := 18
def total_amount : ℝ := 78

-- Problem statement to prove z gets 0.5 rupees for each rupee x gets.
theorem z_share_per_rupee_x (r : ℝ) (hx : x_gets r = 40) (hy : y_gets_for_x r = 18) (ht : total_amount = 78) :
  (total_amount - (x_gets r + y_share)) / x_gets r = 0.5 := by
  sorry

end NUMINAMATH_GPT_z_share_per_rupee_x_l2343_234397


namespace NUMINAMATH_GPT_second_polygon_sides_l2343_234389

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0)
  (perimeter_eq : ∀ (p1 p2 : ℝ), p1 = p2)
  (first_sides : ℕ) (second_sides : ℕ)
  (first_polygon_side_length : ℝ) (second_polygon_side_length : ℝ)
  (first_sides_eq : first_sides = 50)
  (side_length_relation : first_polygon_side_length = 3 * second_polygon_side_length)
  (perimeter_relation : first_sides * first_polygon_side_length = second_sides * second_polygon_side_length) :
  second_sides = 150 := by
  sorry

end NUMINAMATH_GPT_second_polygon_sides_l2343_234389


namespace NUMINAMATH_GPT_cos_arccos_minus_arctan_eq_l2343_234343

noncomputable def cos_arccos_minus_arctan: Real :=
  Real.cos (Real.arccos (4 / 5) - Real.arctan (1 / 2))

theorem cos_arccos_minus_arctan_eq : cos_arccos_minus_arctan = (11 * Real.sqrt 5) / 25 := by
  sorry

end NUMINAMATH_GPT_cos_arccos_minus_arctan_eq_l2343_234343


namespace NUMINAMATH_GPT_female_athletes_in_sample_l2343_234303

theorem female_athletes_in_sample (M F S : ℕ) (hM : M = 56) (hF : F = 42) (hS : S = 28) :
  (F * (S / (M + F))) = 12 :=
by
  rw [hM, hF, hS]
  norm_num
  sorry

end NUMINAMATH_GPT_female_athletes_in_sample_l2343_234303


namespace NUMINAMATH_GPT_combined_weight_of_jake_and_sister_l2343_234340

theorem combined_weight_of_jake_and_sister
  (J : ℕ) (S : ℕ)
  (h₁ : J = 113)
  (h₂ : J - 33 = 2 * S)
  : J + S = 153 :=
sorry

end NUMINAMATH_GPT_combined_weight_of_jake_and_sister_l2343_234340


namespace NUMINAMATH_GPT_cosine_relationship_l2343_234338

open Real

noncomputable def functional_relationship (x y : ℝ) : Prop :=
  y = -(4 / 5) * sqrt (1 - x ^ 2) + (3 / 5) * x

theorem cosine_relationship (α β : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2)
  (h5 : cos (α + β) = - 4 / 5) (h6 : sin β = x) (h7 : cos α = y) (h8 : 4 / 5 < x) (h9 : x < 1) :
  functional_relationship x y :=
sorry

end NUMINAMATH_GPT_cosine_relationship_l2343_234338


namespace NUMINAMATH_GPT_volume_OABC_is_l2343_234358

noncomputable def volume_tetrahedron_ABC (a b c : ℝ) (hx : a^2 + b^2 = 36) (hy : b^2 + c^2 = 25) (hz : c^2 + a^2 = 16) : ℝ :=
  1 / 6 * a * b * c

theorem volume_OABC_is (a b c : ℝ) (hx : a^2 + b^2 = 36) (hy : b^2 + c^2 = 25) (hz : c^2 + a^2 = 16) :
  volume_tetrahedron_ABC a b c hx hy hz = (5 / 6) * Real.sqrt 30.375 :=
by
  sorry

end NUMINAMATH_GPT_volume_OABC_is_l2343_234358


namespace NUMINAMATH_GPT_fred_has_18_stickers_l2343_234316

def jerry_stickers := 36
def george_stickers (jerry : ℕ) := jerry / 3
def fred_stickers (george : ℕ) := george + 6

theorem fred_has_18_stickers :
  let j := jerry_stickers
  let g := george_stickers j 
  fred_stickers g = 18 :=
by
  sorry

end NUMINAMATH_GPT_fred_has_18_stickers_l2343_234316


namespace NUMINAMATH_GPT_sum_and_times_l2343_234323

theorem sum_and_times 
  (a : ℕ) (ha : a = 99) 
  (b : ℕ) (hb : b = 301) 
  (c : ℕ) (hc : c = 200) : 
  a + b = 2 * c :=
by 
  -- skipping proof 
  sorry

end NUMINAMATH_GPT_sum_and_times_l2343_234323


namespace NUMINAMATH_GPT_opposite_signs_add_same_signs_sub_l2343_234311

-- Definitions based on the conditions
variables {a b : ℤ}

-- 1. Case when a and b have opposite signs
theorem opposite_signs_add (h₁ : |a| = 4) (h₂ : |b| = 3) (h₃ : a * b < 0) :
  a + b = 1 ∨ a + b = -1 := 
sorry

-- 2. Case when a and b have the same sign
theorem same_signs_sub (h₁ : |a| = 4) (h₂ : |b| = 3) (h₃ : a * b > 0) :
  a - b = 1 ∨ a - b = -1 := 
sorry

end NUMINAMATH_GPT_opposite_signs_add_same_signs_sub_l2343_234311


namespace NUMINAMATH_GPT_find_m_l2343_234318

-- Define vectors as tuples
def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, -1)
def c (m : ℝ) : ℝ × ℝ := (4, m)

-- Define vector subtraction
def sub_vect (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

-- Define dot product
def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove the condition that (a - b) ⊥ c implies m = 4
theorem find_m (m : ℝ) (h : dot_prod (sub_vect a (b m)) (c m) = 0) : m = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l2343_234318


namespace NUMINAMATH_GPT_avg10_students_correct_l2343_234390

-- Definitions for the conditions
def avg15_students : ℝ := 70
def num15_students : ℕ := 15
def num10_students : ℕ := 10
def num25_students : ℕ := num15_students + num10_students
def avg25_students : ℝ := 80

-- Total percentage calculation based on conditions
def total_perc25_students := num25_students * avg25_students
def total_perc15_students := num15_students * avg15_students

-- The average percent of the 10 students, based on the conditions and given average for 25 students.
theorem avg10_students_correct : 
  (total_perc25_students - total_perc15_students) / (num10_students : ℝ) = 95 := by
  sorry

end NUMINAMATH_GPT_avg10_students_correct_l2343_234390


namespace NUMINAMATH_GPT_unit_digit_of_12_pow_100_l2343_234369

def unit_digit_pow (a: ℕ) (n: ℕ) : ℕ :=
  (a ^ n) % 10

theorem unit_digit_of_12_pow_100 : unit_digit_pow 12 100 = 6 := by
  sorry

end NUMINAMATH_GPT_unit_digit_of_12_pow_100_l2343_234369


namespace NUMINAMATH_GPT_farmer_has_42_cows_left_l2343_234332

-- Define the conditions
def initial_cows := 51
def added_cows := 5
def sold_fraction := 1 / 4

-- Lean statement to prove the number of cows left
theorem farmer_has_42_cows_left :
  (initial_cows + added_cows) - (sold_fraction * (initial_cows + added_cows)) = 42 :=
by
  -- skipping the proof part
  sorry

end NUMINAMATH_GPT_farmer_has_42_cows_left_l2343_234332


namespace NUMINAMATH_GPT_minimum_keys_needed_l2343_234300

theorem minimum_keys_needed (total_cabinets : ℕ) (boxes_per_cabinet : ℕ)
(boxes_needed : ℕ) (boxes_per_cabinet : ℕ) 
(warehouse_key : ℕ) (boxes_per_cabinet: ℕ)
(h1 : total_cabinets = 8)
(h2 : boxes_per_cabinet = 4)
(h3 : (boxes_needed = 52))
(h4 : boxes_per_cabinet = 4)
(h5 : warehouse_key = 1):
    6 + 2 + 1 = 9 := 
    sorry

end NUMINAMATH_GPT_minimum_keys_needed_l2343_234300


namespace NUMINAMATH_GPT_find_foci_l2343_234370

def hyperbolaFoci : Prop :=
  let eq := ∀ x y, 2 * x^2 - 3 * y^2 + 8 * x - 12 * y - 23 = 0
  ∃ foci : ℝ × ℝ, foci = (-2 - Real.sqrt (5 / 6), -2) ∨ foci = (-2 + Real.sqrt (5 / 6), -2)

theorem find_foci : hyperbolaFoci :=
by
  sorry

end NUMINAMATH_GPT_find_foci_l2343_234370


namespace NUMINAMATH_GPT_ellipse_AB_length_l2343_234320

theorem ellipse_AB_length :
  ∀ (F1 F2 A B : ℝ × ℝ) (x y : ℝ),
  (x^2 / 25 + y^2 / 9 = 1) →
  (F1 = (5, 0) ∨ F1 = (-5, 0)) →
  (F2 = (if F1 = (5, 0) then (-5, 0) else (5, 0))) →
  ({p : ℝ × ℝ | p.1^2 / 25 + p.2^2 / 9 = 1} A ∨ {p : ℝ × ℝ | p.1^2 / 25 + p.2^2 / 9 = 1} B) →
  ((A = F1) ∨ (B = F1)) →
  (abs (F2.1 - A.1) + abs (F2.2 - A.2) + abs (F2.1 - B.1) + abs (F2.2 - B.2) = 12) →
  abs (A.1 - B.1) + abs (A.2 - B.2) = 8 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_AB_length_l2343_234320


namespace NUMINAMATH_GPT_two_connected_iff_constructible_with_H_paths_l2343_234378

-- A graph is represented as a structure with vertices and edges
structure Graph where
  vertices : Type
  edges : vertices → vertices → Prop

-- Function to check if a graph is 2-connected
noncomputable def isTwoConnected (G : Graph) : Prop := sorry

-- Function to check if a graph can be constructed by adding H-paths
noncomputable def constructibleWithHPaths (G H : Graph) : Prop := sorry

-- Given a graph G and subgraph H, we need to prove the equivalence
theorem two_connected_iff_constructible_with_H_paths (G H : Graph) :
  (isTwoConnected G) ↔ (constructibleWithHPaths G H) := sorry

end NUMINAMATH_GPT_two_connected_iff_constructible_with_H_paths_l2343_234378


namespace NUMINAMATH_GPT_distance_between_hyperbola_vertices_l2343_234342

theorem distance_between_hyperbola_vertices :
  ∀ (x y : ℝ), (x^2 / 121 - y^2 / 49 = 1) → (22 = 2 * 11) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_hyperbola_vertices_l2343_234342


namespace NUMINAMATH_GPT_find_b_of_expression_l2343_234315

theorem find_b_of_expression (y : ℝ) (b : ℝ) (hy : y > 0)
  (h : (7 / 10) * y = (8 * y) / b + (3 * y) / 10) : b = 20 :=
sorry

end NUMINAMATH_GPT_find_b_of_expression_l2343_234315


namespace NUMINAMATH_GPT_largest_angle_of_consecutive_interior_angles_pentagon_l2343_234339

theorem largest_angle_of_consecutive_interior_angles_pentagon (x : ℕ)
  (h1 : (x - 3) + (x - 2) + (x - 1) + x + (x + 1) = 540) :
  x + 1 = 110 := sorry

end NUMINAMATH_GPT_largest_angle_of_consecutive_interior_angles_pentagon_l2343_234339


namespace NUMINAMATH_GPT_div_eq_210_over_79_l2343_234366

def a_at_b (a b : ℕ) : ℤ := a^2 * b - a * (b^2)
def a_hash_b (a b : ℕ) : ℤ := a^2 + b^2 - a * b

theorem div_eq_210_over_79 : (a_at_b 10 3) / (a_hash_b 10 3) = 210 / 79 :=
by
  -- This is a placeholder and needs to be filled with the actual proof.
  sorry

end NUMINAMATH_GPT_div_eq_210_over_79_l2343_234366


namespace NUMINAMATH_GPT_first_discount_percentage_l2343_234345

-- Definitions based on the conditions provided
def listed_price : ℝ := 400
def final_price : ℝ := 334.4
def additional_discount : ℝ := 5

-- The equation relating these quantities
theorem first_discount_percentage (D : ℝ) (h : listed_price * (1 - D / 100) * (1 - additional_discount / 100) = final_price) : D = 12 :=
sorry

end NUMINAMATH_GPT_first_discount_percentage_l2343_234345


namespace NUMINAMATH_GPT_f_three_l2343_234335

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom even_f_succ : ∀ x : ℝ, f (x + 1) = f (-x + 1)
axiom f_one : f 1 = 1 

-- Goal
theorem f_three : f 3 = -1 :=
by
  -- The proof will be provided here
  sorry

end NUMINAMATH_GPT_f_three_l2343_234335


namespace NUMINAMATH_GPT_aiyanna_cookies_l2343_234357

theorem aiyanna_cookies (a b : ℕ) (h₁ : a = 129) (h₂ : b = a + 11) : b = 140 := by
  sorry

end NUMINAMATH_GPT_aiyanna_cookies_l2343_234357


namespace NUMINAMATH_GPT_range_of_a_monotonically_decreasing_l2343_234325

-- Definitions
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

-- Lean statement
theorem range_of_a_monotonically_decreasing {a : ℝ} : 
  (∀ x y : ℝ, -2 ≤ x → x ≤ 4 → -2 ≤ y → y ≤ 4 → x < y → f a y < f a x) ↔ a ≤ -3 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_a_monotonically_decreasing_l2343_234325


namespace NUMINAMATH_GPT_car_distance_l2343_234359

variable (T_initial : ℕ) (T_new : ℕ) (S : ℕ) (D : ℕ)

noncomputable def calculate_distance (T_initial T_new S : ℕ) : ℕ :=
  S * T_new

theorem car_distance :
  T_initial = 6 →
  T_new = (3 / 2) * T_initial →
  S = 16 →
  D = calculate_distance T_initial T_new S →
  D = 144 :=
by
  sorry

end NUMINAMATH_GPT_car_distance_l2343_234359


namespace NUMINAMATH_GPT_least_product_xy_l2343_234388

theorem least_product_xy : ∀ (x y : ℕ), 0 < x → 0 < y →
  (1 : ℚ) / x + (1 : ℚ) / (3 * y) = 1 / 6 → x * y = 48 :=
by
  intros x y x_pos y_pos h
  sorry

end NUMINAMATH_GPT_least_product_xy_l2343_234388


namespace NUMINAMATH_GPT_chocolate_bars_squares_l2343_234376

theorem chocolate_bars_squares
  (gerald_bars : ℕ)
  (teacher_rate : ℕ)
  (students : ℕ)
  (squares_per_student : ℕ)
  (total_squares : ℕ)
  (total_bars : ℕ)
  (squares_per_bar : ℕ)
  (h1 : gerald_bars = 7)
  (h2 : teacher_rate = 2)
  (h3 : students = 24)
  (h4 : squares_per_student = 7)
  (h5 : total_squares = students * squares_per_student)
  (h6 : total_bars = gerald_bars + teacher_rate * gerald_bars)
  (h7 : squares_per_bar = total_squares / total_bars)
  : squares_per_bar = 8 := by 
  sorry

end NUMINAMATH_GPT_chocolate_bars_squares_l2343_234376


namespace NUMINAMATH_GPT_books_assigned_total_l2343_234353

-- Definitions for the conditions.
def Mcgregor_books := 34
def Floyd_books := 32
def remaining_books := 23

-- The total number of books assigned.
def total_books := Mcgregor_books + Floyd_books + remaining_books

-- The theorem that needs to be proven.
theorem books_assigned_total : total_books = 89 :=
by
  sorry

end NUMINAMATH_GPT_books_assigned_total_l2343_234353


namespace NUMINAMATH_GPT_remainder_of_h_x10_div_h_x_l2343_234328

noncomputable def h (x : ℤ) : ℤ := x^5 - x^4 + x^3 - x^2 + x - 1

theorem remainder_of_h_x10_div_h_x (x : ℤ) : (h (x ^ 10)) % (h x) = -6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_h_x10_div_h_x_l2343_234328


namespace NUMINAMATH_GPT_jeff_total_cabinets_l2343_234387

def initial_cabinets : ℕ := 3
def cabinets_per_counter : ℕ := 2 * initial_cabinets
def total_cabinets_installed : ℕ := 3 * cabinets_per_counter + 5
def total_cabinets (initial : ℕ) (installed : ℕ) : ℕ := initial + installed

theorem jeff_total_cabinets : total_cabinets initial_cabinets total_cabinets_installed = 26 :=
by
  sorry

end NUMINAMATH_GPT_jeff_total_cabinets_l2343_234387


namespace NUMINAMATH_GPT_kids_difference_l2343_234384

def kidsPlayedOnMonday : Nat := 11
def kidsPlayedOnTuesday : Nat := 12

theorem kids_difference :
  kidsPlayedOnTuesday - kidsPlayedOnMonday = 1 := by
  sorry

end NUMINAMATH_GPT_kids_difference_l2343_234384


namespace NUMINAMATH_GPT_fraction_between_stops_l2343_234394

/-- Prove that the fraction of the remaining distance traveled between Maria's first and second stops is 1/4. -/
theorem fraction_between_stops (total_distance first_stop_distance remaining_distance final_leg_distance : ℝ)
  (h_total : total_distance = 400)
  (h_first_stop : first_stop_distance = total_distance / 2)
  (h_remaining : remaining_distance = total_distance - first_stop_distance)
  (h_final_leg : final_leg_distance = 150)
  (h_second_leg : remaining_distance - final_leg_distance = 50) :
  50 / remaining_distance = 1 / 4 :=
by
  { sorry }

end NUMINAMATH_GPT_fraction_between_stops_l2343_234394


namespace NUMINAMATH_GPT_red_marbles_in_A_l2343_234319

-- Define the number of marbles in baskets A, B, and C
variables (R : ℕ)
def basketA := R + 2 -- Basket A: R red, 2 yellow
def basketB := 6 + 1 -- Basket B: 6 green, 1 yellow
def basketC := 3 + 9 -- Basket C: 3 white, 9 yellow

-- Define the greatest difference condition
def greatest_difference (A B C : ℕ) := max (max (A - B) (B - C)) (max (A - C) (C - B))

-- Define the hypothesis based on the conditions
axiom H1 : greatest_difference 3 9 0 = 6

-- The theorem we need to prove: The number of red marbles in Basket A is 8
theorem red_marbles_in_A : R = 8 := 
by {
  -- The proof would go here, but we'll use sorry to skip it
  sorry
}

end NUMINAMATH_GPT_red_marbles_in_A_l2343_234319


namespace NUMINAMATH_GPT_find_x_in_sequence_l2343_234373

theorem find_x_in_sequence :
  ∃ x y z : Int, (z + 3 = 5) ∧ (y + z = 5) ∧ (x + y = 2) ∧ (x = -1) :=
by
  use -1, 3, 2
  sorry

end NUMINAMATH_GPT_find_x_in_sequence_l2343_234373


namespace NUMINAMATH_GPT_determinant_example_l2343_234361

def det_2x2 (a b c d : ℤ) : ℤ := a * d - b * c

theorem determinant_example : det_2x2 7 (-2) (-3) 6 = 36 := by
  sorry

end NUMINAMATH_GPT_determinant_example_l2343_234361


namespace NUMINAMATH_GPT_Clare_has_more_pencils_than_Jeanine_l2343_234349

def Jeanine_initial_pencils : ℕ := 250
def Clare_initial_pencils : ℤ := (-3 : ℤ) * Jeanine_initial_pencils / 5
def Jeanine_pencils_given_Abby : ℕ := (2 : ℕ) * Jeanine_initial_pencils / 7
def Jeanine_pencils_given_Lea : ℕ := (5 : ℕ) * Jeanine_initial_pencils / 11
def Clare_pencils_after_squaring : ℤ := Clare_initial_pencils ^ 2
def Clare_pencils_after_Jeanine_share : ℤ := Clare_pencils_after_squaring + (-1) * Jeanine_initial_pencils / 4

def Jeanine_final_pencils : ℕ := Jeanine_initial_pencils - Jeanine_pencils_given_Abby - Jeanine_pencils_given_Lea

theorem Clare_has_more_pencils_than_Jeanine :
  Clare_pencils_after_Jeanine_share - Jeanine_final_pencils = 22372 :=
sorry

end NUMINAMATH_GPT_Clare_has_more_pencils_than_Jeanine_l2343_234349


namespace NUMINAMATH_GPT_no_such_b_c_exist_l2343_234336

theorem no_such_b_c_exist :
  ¬ ∃ (b c : ℝ), (∃ (k l : ℤ), (k ≠ l ∧ (k ^ 2 + b * ↑k + c = 0) ∧ (l ^ 2 + b * ↑l + c = 0))) ∧
                  (∃ (m n : ℤ), (m ≠ n ∧ (2 * (m ^ 2) + (b + 1) * ↑m + (c + 1) = 0) ∧ 
                                        (2 * (n ^ 2) + (b + 1) * ↑n + (c + 1) = 0))) :=
sorry

end NUMINAMATH_GPT_no_such_b_c_exist_l2343_234336


namespace NUMINAMATH_GPT_value_of_mn_l2343_234331

theorem value_of_mn (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_eq : m^4 - n^4 = 3439) : m * n = 90 := 
by sorry

end NUMINAMATH_GPT_value_of_mn_l2343_234331


namespace NUMINAMATH_GPT_smaller_angle_between_clock_hands_3_40_pm_l2343_234352

theorem smaller_angle_between_clock_hands_3_40_pm :
  let minute_hand_angle : ℝ := 240
  let hour_hand_angle : ℝ := 110
  let angle_between_hands : ℝ := |minute_hand_angle - hour_hand_angle|
  angle_between_hands = 130.0 :=
by
  sorry

end NUMINAMATH_GPT_smaller_angle_between_clock_hands_3_40_pm_l2343_234352


namespace NUMINAMATH_GPT_tiling_impossible_l2343_234351

theorem tiling_impossible (T2 T14 : ℕ) :
  let S_before := 2 * T2
  let S_after := 2 * (T2 - 1) + 1 
  S_after ≠ S_before :=
sorry

end NUMINAMATH_GPT_tiling_impossible_l2343_234351


namespace NUMINAMATH_GPT_determine_numbers_l2343_234374

theorem determine_numbers (n : ℕ) (m : ℕ) (x y z u v : ℕ) (h₁ : 10000 <= n ∧ n < 100000)
(h₂ : n = 10000 * x + 1000 * y + 100 * z + 10 * u + v)
(h₃ : m = 1000 * x + 100 * y + 10 * u + v)
(h₄ : x ≠ 0)
(h₅ : n % m = 0) :
∃ a : ℕ, (10 <= a ∧ a <= 99 ∧ n = a * 1000) :=
sorry

end NUMINAMATH_GPT_determine_numbers_l2343_234374


namespace NUMINAMATH_GPT_angle_Z_90_l2343_234364

-- Definitions and conditions from step a)
def Triangle (X Y Z : ℝ) : Prop :=
  X + Y + Z = 180

def in_triangle_XYZ (X Y Z : ℝ) : Prop :=
  Triangle X Y Z ∧ (X + Y = 90)

-- Proof problem from step c)
theorem angle_Z_90 (X Y Z : ℝ) (h : in_triangle_XYZ X Y Z) : Z = 90 :=
  by
  sorry

end NUMINAMATH_GPT_angle_Z_90_l2343_234364


namespace NUMINAMATH_GPT_fraction_of_number_l2343_234371

theorem fraction_of_number : (7 / 8) * 48 = 42 := 
by sorry

end NUMINAMATH_GPT_fraction_of_number_l2343_234371


namespace NUMINAMATH_GPT_new_total_lines_l2343_234334

-- Definitions and conditions
variable (L : ℕ)
def increased_lines : ℕ := L + 60
def percentage_increase := (60 : ℚ) / L = 1 / 3

-- Theorem statement
theorem new_total_lines : percentage_increase L → increased_lines L = 240 :=
by
  sorry

end NUMINAMATH_GPT_new_total_lines_l2343_234334


namespace NUMINAMATH_GPT_problem_l2343_234375

theorem problem (θ : ℝ) (htan : Real.tan θ = 1 / 3) : Real.cos θ ^ 2 + 2 * Real.sin θ = 6 / 5 := 
by
  sorry

end NUMINAMATH_GPT_problem_l2343_234375


namespace NUMINAMATH_GPT_number_of_trips_l2343_234399

theorem number_of_trips (bags_per_trip : ℕ) (weight_per_bag : ℕ) (total_weight : ℕ)
  (h1 : bags_per_trip = 10)
  (h2 : weight_per_bag = 50)
  (h3 : total_weight = 10000) : 
  total_weight / (bags_per_trip * weight_per_bag) = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_trips_l2343_234399


namespace NUMINAMATH_GPT_find_distance_between_stripes_l2343_234346

-- Define the problem conditions
def parallel_curbs (a b : ℝ) := ∀ g : ℝ, g * a = b
def crosswalk_conditions (curb_distance curb_length stripe_length : ℝ) := 
  curb_distance = 60 ∧ curb_length = 22 ∧ stripe_length = 65

-- State the theorem
theorem find_distance_between_stripes (curb_distance curb_length stripe_length : ℝ) 
  (h : ℝ) (H : crosswalk_conditions curb_distance curb_length stripe_length) :
  h = 264 / 13 :=
sorry

end NUMINAMATH_GPT_find_distance_between_stripes_l2343_234346


namespace NUMINAMATH_GPT_parabola_vertex_coordinates_l2343_234392

theorem parabola_vertex_coordinates :
  ∀ (x y : ℝ), y = -3 * (x + 1)^2 - 2 → (x, y) = (-1, -2) := by
  sorry

end NUMINAMATH_GPT_parabola_vertex_coordinates_l2343_234392


namespace NUMINAMATH_GPT_average_mark_excluded_students_l2343_234380

variables (N A E A_R A_E : ℕ)

theorem average_mark_excluded_students:
    N = 56 → A = 80 → E = 8 → A_R = 90 →
    N * A = E * A_E + (N - E) * A_R →
    A_E = 20 :=
by
  intros hN hA hE hAR hEquation
  rw [hN, hA, hE, hAR] at hEquation
  have h : 4480 = 8 * A_E + 4320 := hEquation
  sorry

end NUMINAMATH_GPT_average_mark_excluded_students_l2343_234380


namespace NUMINAMATH_GPT_time_to_pay_back_l2343_234377

-- Definitions for conditions
def initial_cost : ℕ := 25000
def monthly_revenue : ℕ := 4000
def monthly_expenses : ℕ := 1500
def monthly_profit : ℕ := monthly_revenue - monthly_expenses

-- Theorem statement
theorem time_to_pay_back : initial_cost / monthly_profit = 10 := by
  -- Skipping the proof here
  sorry

end NUMINAMATH_GPT_time_to_pay_back_l2343_234377


namespace NUMINAMATH_GPT_bob_daily_earnings_l2343_234322

-- Define Sally's daily earnings
def Sally_daily_earnings : ℝ := 6

-- Define the total savings after a year for both Sally and Bob
def total_savings : ℝ := 1825

-- Define the number of days in a year
def days_in_year : ℝ := 365

-- Define Bob's daily earnings
variable (B : ℝ)

-- Define the proof statement
theorem bob_daily_earnings : (3 + B / 2) * days_in_year = total_savings → B = 4 :=
by
  sorry

end NUMINAMATH_GPT_bob_daily_earnings_l2343_234322


namespace NUMINAMATH_GPT_sin_product_identity_l2343_234341

noncomputable def sin_15_deg := Real.sin (15 * Real.pi / 180)
noncomputable def sin_30_deg := Real.sin (30 * Real.pi / 180)
noncomputable def sin_75_deg := Real.sin (75 * Real.pi / 180)

theorem sin_product_identity :
  sin_15_deg * sin_30_deg * sin_75_deg = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_sin_product_identity_l2343_234341


namespace NUMINAMATH_GPT_factorize_expr1_factorize_expr2_l2343_234308

variable (x y a b : ℝ)

theorem factorize_expr1 : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := sorry

theorem factorize_expr2 : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b) := sorry

end NUMINAMATH_GPT_factorize_expr1_factorize_expr2_l2343_234308


namespace NUMINAMATH_GPT_ch_sub_ch_add_sh_sub_sh_add_l2343_234329

noncomputable def sh (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2
noncomputable def ch (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2

theorem ch_sub (x y : ℝ) : ch (x - y) = ch x * ch y - sh x * sh y := sorry
theorem ch_add (x y : ℝ) : ch (x + y) = ch x * ch y + sh x * sh y := sorry
theorem sh_sub (x y : ℝ) : sh (x - y) = sh x * ch y - ch x * sh y := sorry
theorem sh_add (x y : ℝ) : sh (x + y) = sh x * ch y + ch x * sh y := sorry

end NUMINAMATH_GPT_ch_sub_ch_add_sh_sub_sh_add_l2343_234329


namespace NUMINAMATH_GPT_marching_band_formations_l2343_234307

theorem marching_band_formations :
  (∃ (s t : ℕ), s * t = 240 ∧ 8 ≤ t ∧ t ≤ 30) →
  ∃ (z : ℕ), z = 4 := sorry

end NUMINAMATH_GPT_marching_band_formations_l2343_234307


namespace NUMINAMATH_GPT_actual_speed_of_valentin_l2343_234391

theorem actual_speed_of_valentin
  (claimed_speed : ℕ := 50) -- Claimed speed in m/min
  (wrong_meter : ℕ := 60)   -- Valentin thought 1 meter = 60 cm
  (wrong_minute : ℕ := 100) -- Valentin thought 1 minute = 100 seconds
  (correct_speed : ℕ := 18) -- The actual speed in m/min
  : (claimed_speed * wrong_meter / wrong_minute) * 60 / 100 = correct_speed :=
by
  sorry

end NUMINAMATH_GPT_actual_speed_of_valentin_l2343_234391


namespace NUMINAMATH_GPT_exists_ratios_eq_l2343_234348

theorem exists_ratios_eq (a b z : ℕ) (ha : 0 < a) (hb : 0 < b) (hz : 0 < z) (h : a * b = z^2 + 1) :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ (a : ℚ) / b = (x^2 + 1) / (y^2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_exists_ratios_eq_l2343_234348


namespace NUMINAMATH_GPT_cos_120_eq_neg_half_l2343_234386

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_120_eq_neg_half_l2343_234386


namespace NUMINAMATH_GPT_poly_eq_l2343_234333

-- Definition of the polynomials f(x) and g(x)
def f (x : ℝ) := x^4 + 4*x^3 + 8*x
def g (x : ℝ) := 10*x^4 + 30*x^3 + 29*x^2 + 2*x + 5

-- Define p(x) as a function that satisfies the given condition
def p (x : ℝ) := 9*x^4 + 26*x^3 + 29*x^2 - 6*x + 5

-- Prove that the function p(x) satisfies the equation
theorem poly_eq : ∀ x : ℝ, p x + f x = g x :=
by
  intro x
  -- Add a marker to indicate that this is where the proof would go
  sorry

end NUMINAMATH_GPT_poly_eq_l2343_234333


namespace NUMINAMATH_GPT_ratio_markus_age_son_age_l2343_234317

variable (M S G : ℕ)

theorem ratio_markus_age_son_age (h1 : G = 20) (h2 : S = 2 * G) (h3 : M + S + G = 140) : M / S = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_markus_age_son_age_l2343_234317


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2343_234304

open Nat

theorem arithmetic_sequence_sum :
  let a := 50
  let d := 3
  let l := 98
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  3 * S = 3774 := 
by
  let a := 50
  let d := 3
  let l := 98
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2343_234304


namespace NUMINAMATH_GPT_min_value_ratio_l2343_234354

noncomputable def min_ratio (a : ℝ) (h : a > 0) : ℝ :=
  let x_A := 4^(-a)
  let x_B := 4^(a)
  let x_C := 4^(- (18 / (2*a + 1)))
  let x_D := 4^((18 / (2*a + 1)))
  let m := abs (x_A - x_C)
  let n := abs (x_B - x_D)
  n / m

theorem min_value_ratio (a : ℝ) (h : a > 0) : 
  ∃ c : ℝ, c = 2^11 := sorry

end NUMINAMATH_GPT_min_value_ratio_l2343_234354
