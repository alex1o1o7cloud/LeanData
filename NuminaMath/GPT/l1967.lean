import Mathlib

namespace NUMINAMATH_GPT_distance_at_40_kmph_l1967_196733

theorem distance_at_40_kmph (x : ℝ) (h1 : x / 40 + (250 - x) / 60 = 5) : x = 100 := 
by
  sorry

end NUMINAMATH_GPT_distance_at_40_kmph_l1967_196733


namespace NUMINAMATH_GPT_garden_remaining_area_is_250_l1967_196786

open Nat

-- Define the dimensions of the rectangular garden
def garden_length : ℕ := 18
def garden_width : ℕ := 15
-- Define the dimensions of the square cutouts
def cutout1_side : ℕ := 4
def cutout2_side : ℕ := 2

-- Calculate areas based on the definitions
def garden_area : ℕ := garden_length * garden_width
def cutout1_area : ℕ := cutout1_side * cutout1_side
def cutout2_area : ℕ := cutout2_side * cutout2_side

-- Calculate total area excluding the cutouts
def remaining_area : ℕ := garden_area - cutout1_area - cutout2_area

-- Prove that the remaining area is 250 square feet
theorem garden_remaining_area_is_250 : remaining_area = 250 :=
by
  sorry

end NUMINAMATH_GPT_garden_remaining_area_is_250_l1967_196786


namespace NUMINAMATH_GPT_sum_first_15_odd_from_5_l1967_196782

theorem sum_first_15_odd_from_5 : 
  let a₁ := 5 
  let d := 2 
  let n := 15 
  let a₁₅ := a₁ + (n - 1) * d 
  let S := n * (a₁ + a₁₅) / 2 
  S = 285 := by 
  sorry

end NUMINAMATH_GPT_sum_first_15_odd_from_5_l1967_196782


namespace NUMINAMATH_GPT_find_selling_price_l1967_196791

-- Define the cost price of the article
def cost_price : ℝ := 47

-- Define the profit when the selling price is Rs. 54
def profit : ℝ := 54 - cost_price

-- Assume that the profit is the same as the loss
axiom profit_equals_loss : profit = 7

-- Define the selling price that yields the same loss as the profit
def selling_price_loss : ℝ := cost_price - profit

-- Now state the theorem to prove that the selling price for loss is Rs. 40
theorem find_selling_price : selling_price_loss = 40 :=
sorry

end NUMINAMATH_GPT_find_selling_price_l1967_196791


namespace NUMINAMATH_GPT_david_marks_in_english_l1967_196794

theorem david_marks_in_english
  (math phys chem bio : ℕ)
  (avg subs : ℕ) 
  (h_math : math = 95) 
  (h_phys : phys = 82) 
  (h_chem : chem = 97) 
  (h_bio : bio = 95) 
  (h_avg : avg = 93)
  (h_subs : subs = 5) :
  ∃ E : ℕ, (avg * subs = E + math + phys + chem + bio) ∧ E = 96 :=
by
  sorry

end NUMINAMATH_GPT_david_marks_in_english_l1967_196794


namespace NUMINAMATH_GPT_pure_imaginary_number_l1967_196741

theorem pure_imaginary_number (a : ℝ) (ha : (1 + a) / (1 + a^2) = 0) : a = -1 :=
sorry

end NUMINAMATH_GPT_pure_imaginary_number_l1967_196741


namespace NUMINAMATH_GPT_A_work_days_l1967_196796

theorem A_work_days (x : ℝ) (H : 3 * (1 / x + 1 / 20) = 0.35) : x = 15 := 
by
  sorry

end NUMINAMATH_GPT_A_work_days_l1967_196796


namespace NUMINAMATH_GPT_machine_value_depletion_rate_l1967_196726

theorem machine_value_depletion_rate :
  ∃ r : ℝ, 700 * (1 - r)^2 = 567 ∧ r = 0.1 := 
by
  sorry

end NUMINAMATH_GPT_machine_value_depletion_rate_l1967_196726


namespace NUMINAMATH_GPT_basketball_shots_l1967_196707

theorem basketball_shots (total_points total_3pt_shots: ℕ) 
  (h1: total_points = 26) 
  (h2: total_3pt_shots = 4) 
  (h3: ∀ points_from_3pt_shots, points_from_3pt_shots = 3 * total_3pt_shots) :
  let points_from_3pt_shots := 3 * total_3pt_shots
  let points_from_2pt_shots := total_points - points_from_3pt_shots
  let total_2pt_shots := points_from_2pt_shots / 2
  total_2pt_shots + total_3pt_shots = 11 :=
by
  sorry

end NUMINAMATH_GPT_basketball_shots_l1967_196707


namespace NUMINAMATH_GPT_arrangement_with_A_in_middle_arrangement_with_A_at_end_B_not_at_end_arrangement_with_A_B_adjacent_not_adjacent_to_C_l1967_196738

-- Proof Problem 1
theorem arrangement_with_A_in_middle (products : Finset ℕ) (A : ℕ) (hA : A ∈ products) (arrangements : Finset (Fin 5 → ℕ)) :
  5 ∈ products ∧ (∀ a ∈ arrangements, a (Fin.mk 2 sorry) = A) →
  arrangements.card = 24 :=
by sorry

-- Proof Problem 2
theorem arrangement_with_A_at_end_B_not_at_end (products : Finset ℕ) (A B : ℕ) (hA : A ∈ products) (hB : B ∈ products) (arrangements : Finset (Fin 5 → ℕ)) :
  (5 ∈ products ∧ (∀ a ∈ arrangements, (a 0 = A ∨ a 4 = A) ∧ (a 1 ≠ B ∧ a 2 ≠ B ∧ a 3 ≠ B))) →
  arrangements.card = 36 :=
by sorry

-- Proof Problem 3
theorem arrangement_with_A_B_adjacent_not_adjacent_to_C (products : Finset ℕ) (A B C : ℕ) (hA : A ∈ products) (hB : B ∈ products) (hC : C ∈ products) (arrangements : Finset (Fin 5 → ℕ)) :
  (5 ∈ products ∧ (∀ a ∈ arrangements, ((a 0 = A ∧ a 1 = B) ∨ (a 1 = A ∧ a 2 = B) ∨ (a 2 = A ∧ a 3 = B) ∨ (a 3 = A ∧ a 4 = B)) ∧
   (a 0 ≠ A ∧ a 1 ≠ B ∧ a 2 ≠ C))) →
  arrangements.card = 36 :=
by sorry

end NUMINAMATH_GPT_arrangement_with_A_in_middle_arrangement_with_A_at_end_B_not_at_end_arrangement_with_A_B_adjacent_not_adjacent_to_C_l1967_196738


namespace NUMINAMATH_GPT_swap_square_digit_l1967_196714

theorem swap_square_digit (n : ℕ) (h1 : n ≥ 10 ∧ n < 100) : 
  ∃ (x y : ℕ), n = 10 * x + y ∧ (x < 10 ∧ y < 10) ∧ (y * 100 + x * 10 + y^2 + 20 * x * y - 1) = n * n + 2 * n + 1 :=
by 
    sorry

end NUMINAMATH_GPT_swap_square_digit_l1967_196714


namespace NUMINAMATH_GPT_remainder_of_N_eq_4101_l1967_196795

noncomputable def N : ℕ :=
  20 + 3^(3^(3+1) - 13)

theorem remainder_of_N_eq_4101 : N % 10000 = 4101 := by
  sorry

end NUMINAMATH_GPT_remainder_of_N_eq_4101_l1967_196795


namespace NUMINAMATH_GPT_value_is_85_over_3_l1967_196762

theorem value_is_85_over_3 (a b : ℚ)  (h1 : 3 * a + 6 * b = 48) (h2 : 8 * a + 4 * b = 84) : 2 * a + 3 * b = 85 / 3 := 
by {
  -- Proof will go here
  sorry
}

end NUMINAMATH_GPT_value_is_85_over_3_l1967_196762


namespace NUMINAMATH_GPT_feed_mixture_hay_calculation_l1967_196720

theorem feed_mixture_hay_calculation
  (hay_Stepan_percent oats_Pavel_percent corn_mixture_percent : ℝ)
  (hay_Stepan_mass_Stepan hay_Pavel_mass_Pavel total_mixture_mass : ℝ):
  hay_Stepan_percent = 0.4 ∧
  oats_Pavel_percent = 0.26 ∧
  (∃ (x : ℝ), 
  x > 0 ∧ 
  hay_Pavel_percent =  0.74 - x ∧ 
  0.15 * x + 0.25 * x = 0.3 * total_mixture_mass ∧
  hay_Stepan_mass_Stepan = 0.40 * 150 ∧
  hay_Pavel_mass_Pavel = (0.74 - x) * 250 ∧ 
  total_mixture_mass = 150 + 250) → 
  hay_Stepan_mass_Stepan + hay_Pavel_mass_Pavel = 170 := 
by
  intro h
  obtain ⟨h1, h2, ⟨x, hx1, hx2, hx3, hx4, hx5, hx6⟩⟩ := h
  /- proof -/
  sorry

end NUMINAMATH_GPT_feed_mixture_hay_calculation_l1967_196720


namespace NUMINAMATH_GPT_bob_more_than_alice_l1967_196767

-- Definitions for conditions
def initial_investment_alice : ℕ := 10000
def initial_investment_bob : ℕ := 10000
def multiple_alice : ℕ := 3
def multiple_bob : ℕ := 7

-- Derived conditions based on the investment multiples
def final_amount_alice : ℕ := initial_investment_alice * multiple_alice
def final_amount_bob : ℕ := initial_investment_bob * multiple_bob

-- Statement of the problem
theorem bob_more_than_alice : final_amount_bob - final_amount_alice = 40000 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_bob_more_than_alice_l1967_196767


namespace NUMINAMATH_GPT_acute_angle_at_3_16_l1967_196788

def angle_between_clock_hands (hour minute : ℕ) : ℝ :=
  let minute_angle := (minute / 60) * 360
  let hour_angle := (hour % 12) * 30 + (minute / 60) * 30
  |hour_angle - minute_angle|

theorem acute_angle_at_3_16 : angle_between_clock_hands 3 16 = 2 := 
sorry

end NUMINAMATH_GPT_acute_angle_at_3_16_l1967_196788


namespace NUMINAMATH_GPT_probability_three_different_suits_l1967_196752

noncomputable def pinochle_deck := 48
noncomputable def total_cards := 48
noncomputable def different_suits_probability := (36 / 47) * (23 / 46)

theorem probability_three_different_suits :
  different_suits_probability = 414 / 1081 :=
sorry

end NUMINAMATH_GPT_probability_three_different_suits_l1967_196752


namespace NUMINAMATH_GPT_periodic_function_implies_rational_ratio_l1967_196798

noncomputable def g (i : ℕ) (a ω θ x : ℝ) : ℝ := 
  a * Real.sin (ω * x + θ)

theorem periodic_function_implies_rational_ratio 
  (a1 a2 ω1 ω2 θ1 θ2 : ℝ) (h1 : a1 * ω1 ≠ 0) (h2 : a2 * ω2 ≠ 0)
  (h3 : |ω1| ≠ |ω2|) 
  (hf_periodic : ∃ T : ℝ, ∀ x : ℝ, g 1 a1 ω1 θ1 (x + T) + g 2 a2 ω2 θ2 (x + T) = g 1 a1 ω1 θ1 x + g 2 a2 ω2 θ2 x) :
  ∃ m n : ℤ, n ≠ 0 ∧ ω1 / ω2 = m / n :=
sorry

end NUMINAMATH_GPT_periodic_function_implies_rational_ratio_l1967_196798


namespace NUMINAMATH_GPT_one_over_m_add_one_over_n_l1967_196701

theorem one_over_m_add_one_over_n (m n : ℕ) (h_sum : m + n = 80) (h_hcf : Nat.gcd m n = 6) (h_lcm : Nat.lcm m n = 210) : 
  1 / (m:ℚ) + 1 / (n:ℚ) = 1 / 15.75 :=
by
  sorry

end NUMINAMATH_GPT_one_over_m_add_one_over_n_l1967_196701


namespace NUMINAMATH_GPT_proposition_false_at_9_l1967_196753

theorem proposition_false_at_9 (P : ℕ → Prop) 
  (h : ∀ k : ℕ, k ≥ 1 → P k → P (k + 1))
  (hne10 : ¬ P 10) : ¬ P 9 :=
by
  intro hp9
  have hp10 : P 10 := h _ (by norm_num) hp9
  contradiction

end NUMINAMATH_GPT_proposition_false_at_9_l1967_196753


namespace NUMINAMATH_GPT_base_height_l1967_196700

-- Define the height of the sculpture and the combined height.
def sculpture_height : ℚ := 2 + 10 / 12
def total_height : ℚ := 3 + 2 / 3

-- We want to prove that the base height is 5/6 feet.
theorem base_height :
  total_height - sculpture_height = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_base_height_l1967_196700


namespace NUMINAMATH_GPT_chess_tournament_games_l1967_196783

def players : ℕ := 12

def games_per_pair : ℕ := 2

theorem chess_tournament_games (n : ℕ) (h : n = players) : 
  (n * (n - 1) * games_per_pair) = 264 := by
  sorry

end NUMINAMATH_GPT_chess_tournament_games_l1967_196783


namespace NUMINAMATH_GPT_B_share_is_correct_l1967_196717

open Real

noncomputable def total_money : ℝ := 10800
noncomputable def ratio_A : ℝ := 0.5
noncomputable def ratio_B : ℝ := 1.5
noncomputable def ratio_C : ℝ := 2.25
noncomputable def ratio_D : ℝ := 3.5
noncomputable def ratio_E : ℝ := 4.25
noncomputable def total_ratio : ℝ := ratio_A + ratio_B + ratio_C + ratio_D + ratio_E
noncomputable def value_per_part : ℝ := total_money / total_ratio
noncomputable def B_share : ℝ := ratio_B * value_per_part

theorem B_share_is_correct : B_share = 1350 := by 
  sorry

end NUMINAMATH_GPT_B_share_is_correct_l1967_196717


namespace NUMINAMATH_GPT_min_weight_of_automobile_l1967_196725

theorem min_weight_of_automobile (ferry_weight_tons: ℝ) (auto_max_weight: ℝ) 
  (max_autos: ℝ) (ferry_weight_pounds: ℝ) (min_auto_weight: ℝ) : 
  ferry_weight_tons = 50 → 
  auto_max_weight = 3200 → 
  max_autos = 62.5 → 
  ferry_weight_pounds = ferry_weight_tons * 2000 → 
  min_auto_weight = ferry_weight_pounds / max_autos → 
  min_auto_weight = 1600 :=
by
  intros
  sorry

end NUMINAMATH_GPT_min_weight_of_automobile_l1967_196725


namespace NUMINAMATH_GPT_distribution_schemes_36_l1967_196784

def num_distribution_schemes (total_students english_excellent computer_skills : ℕ) : ℕ :=
  if total_students = 8 ∧ english_excellent = 2 ∧ computer_skills = 3 then 36 else 0

theorem distribution_schemes_36 :
  num_distribution_schemes 8 2 3 = 36 :=
by
 sorry

end NUMINAMATH_GPT_distribution_schemes_36_l1967_196784


namespace NUMINAMATH_GPT_linear_condition_l1967_196704

theorem linear_condition (a : ℝ) : a ≠ 0 ↔ ∃ (x y : ℝ), ax + y = -1 :=
by
  sorry

end NUMINAMATH_GPT_linear_condition_l1967_196704


namespace NUMINAMATH_GPT_spectators_count_l1967_196716

theorem spectators_count (total_wristbands : ℕ) (wristbands_per_person : ℕ) (h1 : total_wristbands = 250) (h2 : wristbands_per_person = 2) : (total_wristbands / wristbands_per_person = 125) :=
by
  sorry

end NUMINAMATH_GPT_spectators_count_l1967_196716


namespace NUMINAMATH_GPT_team_with_at_least_one_girl_l1967_196729

noncomputable def choose (n m : ℕ) := Nat.factorial n / (Nat.factorial m * Nat.factorial (n - m))

theorem team_with_at_least_one_girl (total_boys total_girls select : ℕ) (h_boys : total_boys = 5) (h_girls : total_girls = 5) (h_select : select = 3) :
  (choose (total_boys + total_girls) select) - (choose total_boys select) = 110 := 
by
  sorry

end NUMINAMATH_GPT_team_with_at_least_one_girl_l1967_196729


namespace NUMINAMATH_GPT_perpendicular_vectors_t_values_l1967_196780

variable (t : ℝ)
def a := (t, 0, -1)
def b := (2, 5, t^2)

theorem perpendicular_vectors_t_values (h : (2 * t + 0 * 5 + -1 * t^2) = 0) : t = 0 ∨ t = 2 :=
by sorry

end NUMINAMATH_GPT_perpendicular_vectors_t_values_l1967_196780


namespace NUMINAMATH_GPT_animal_population_l1967_196718

def total_population (L P E : ℕ) : ℕ :=
L + P + E

theorem animal_population 
    (L P E : ℕ) 
    (h1 : L = 2 * P) 
    (h2 : E = (L + P) / 2) 
    (h3 : L = 200) : 
  total_population L P E = 450 := 
  by 
    sorry

end NUMINAMATH_GPT_animal_population_l1967_196718


namespace NUMINAMATH_GPT_find_f_of_one_half_l1967_196781

def g (x : ℝ) : ℝ := 1 - 2 * x

noncomputable def f (x : ℝ) : ℝ := (1 - x ^ 2) / x ^ 2

theorem find_f_of_one_half :
  f (g (1 / 2)) = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_one_half_l1967_196781


namespace NUMINAMATH_GPT_area_of_quadrilateral_EFGH_l1967_196745

-- Define the properties of rectangle ABCD and the areas
def rectangle (A B C D : Type) := 
  ∃ (area : ℝ), area = 48

-- Define the positions of the points E, G, F, H
def points_positions (A D C B E G F H : Type) :=
  ∃ (one_third : ℝ) (two_thirds : ℝ), one_third = 1/3 ∧ two_thirds = 2/3

-- Define the area calculation for quadrilateral EFGH
def area_EFGH (area_ABCD : ℝ) (one_third : ℝ) : ℝ :=
  (one_third * one_third) * area_ABCD

-- The proof statement that area of EFGH is 5 1/3 square meters
theorem area_of_quadrilateral_EFGH 
  (A B C D E F G H : Type)
  (area_ABCD : ℝ)
  (one_third : ℝ) :
  rectangle A B C D →
  points_positions A D C B E G F H →
  area_ABCD = 48 →
  one_third = 1/3 →
  area_EFGH area_ABCD one_third = 16/3 :=
by
  intros h1 h2 h3 h4
  have h5 : area_EFGH area_ABCD one_third = 16/3 :=
  sorry
  exact h5

end NUMINAMATH_GPT_area_of_quadrilateral_EFGH_l1967_196745


namespace NUMINAMATH_GPT_number_of_sides_l1967_196792

def side_length : ℕ := 16
def perimeter : ℕ := 80

theorem number_of_sides (h1: side_length = 16) (h2: perimeter = 80) : (perimeter / side_length = 5) :=
by
  -- Proof should be inserted here.
  sorry

end NUMINAMATH_GPT_number_of_sides_l1967_196792


namespace NUMINAMATH_GPT_grade_distribution_sum_l1967_196756

theorem grade_distribution_sum (a b c d : ℝ) (ha : a = 0.6) (hb : b = 0.25) (hc : c = 0.1) (hd : d = 0.05) :
  a + b + c + d = 1.0 :=
by
  -- Introduce the hypothesis
  rw [ha, hb, hc, hd]
  -- Now the goal simplifies to: 0.6 + 0.25 + 0.1 + 0.05 = 1.0
  sorry

end NUMINAMATH_GPT_grade_distribution_sum_l1967_196756


namespace NUMINAMATH_GPT_equidistant_point_quadrants_l1967_196731

theorem equidistant_point_quadrants (x y : ℝ) (h : 4 * x + 3 * y = 12) :
  (x > 0 ∧ y = 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end NUMINAMATH_GPT_equidistant_point_quadrants_l1967_196731


namespace NUMINAMATH_GPT_number_of_pines_l1967_196742

theorem number_of_pines (total_trees : ℕ) (T B S : ℕ) (h_total : total_trees = 101)
  (h_poplar_spacing : ∀ T1 T2, T1 ≠ T2 → (T2 - T1) > 1)
  (h_birch_spacing : ∀ B1 B2, B1 ≠ B2 → (B2 - B1) > 2)
  (h_pine_spacing : ∀ S1 S2, S1 ≠ S2 → (S2 - S1) > 3) :
  S = 25 ∨ S = 26 :=
sorry

end NUMINAMATH_GPT_number_of_pines_l1967_196742


namespace NUMINAMATH_GPT_prob_A_wins_correct_l1967_196715

noncomputable def prob_A_wins : ℚ :=
  let outcomes : ℕ := 3^3
  let win_one_draw_two : ℕ := 3
  let win_two_other : ℕ := 6
  let win_all : ℕ := 1
  let total_wins : ℕ := win_one_draw_two + win_two_other + win_all
  total_wins / outcomes

theorem prob_A_wins_correct :
  prob_A_wins = 10/27 :=
by
  sorry

end NUMINAMATH_GPT_prob_A_wins_correct_l1967_196715


namespace NUMINAMATH_GPT_box_cost_is_550_l1967_196744

noncomputable def cost_of_dryer_sheets (loads_per_week : ℕ) (sheets_per_load : ℕ)
                                        (sheets_per_box : ℕ) (annual_savings : ℝ) : ℝ :=
  let sheets_per_week := loads_per_week * sheets_per_load
  let sheets_per_year := sheets_per_week * 52
  let boxes_per_year := sheets_per_year / sheets_per_box
  annual_savings / boxes_per_year

theorem box_cost_is_550 (h1 : 4 = 4)
                        (h2 : 1 = 1)
                        (h3 : 104 = 104)
                        (h4 : 11 = 11) :
  cost_of_dryer_sheets 4 1 104 11 = 5.50 :=
by
  sorry

end NUMINAMATH_GPT_box_cost_is_550_l1967_196744


namespace NUMINAMATH_GPT_sum_of_a_and_b_l1967_196737

def otimes (x y : ℝ) : ℝ := x * (1 - y)

variable (a b : ℝ)

theorem sum_of_a_and_b :
  ({ x : ℝ | (x - a) * (1 - (x - b)) > 0 } = { x : ℝ | 2 < x ∧ x < 3 }) →
  a + b = 4 :=
by
  intro h
  have h_eq : ∀ x, (x - a) * ((1 : ℝ) - (x - b)) = (x - a) * (x - (b + 1)) := sorry
  have h_ineq : ∀ x, (x - a) * (x - (b + 1)) > 0 ↔ 2 < x ∧ x < 3 := sorry
  have h_set_eq : { x | (x - a) * ((1 : ℝ) - (x - b)) > 0 } = { x | 2 < x ∧ x < 3 } := sorry
  have h_roots_2_3 : (2 - a) * (2 - (b + 1)) = 0 ∧ (3 - a) * (3 - (b + 1)) = 0 := sorry
  have h_2_eq : 2 - a = 0 ∨ 2 - (b + 1) = 0 := sorry
  have h_3_eq : 3 - a = 0 ∨ 3 - (b + 1) = 0 := sorry
  have h_a_2 : a = 2 ∨ b + 1 = 2 := sorry
  have h_b_2 : b = 2 - 1 := sorry
  have h_a_3 : a = 3 ∨ b + 1 = 3 := sorry
  have h_b_3 : b = 3 - 1 := sorry
  sorry

end NUMINAMATH_GPT_sum_of_a_and_b_l1967_196737


namespace NUMINAMATH_GPT_dual_colored_numbers_l1967_196785

theorem dual_colored_numbers (table : Matrix (Fin 10) (Fin 20) ℕ)
  (distinct_numbers : ∀ (i j k l : Fin 10) (m n : Fin 20), 
    (i ≠ k ∨ m ≠ n) → table i m ≠ table k n)
  (row_red : ∀ (i : Fin 10), ∃ r₁ r₂ : Fin 20, r₁ ≠ r₂ ∧ 
    (∀ (j : Fin 20), table i j ≤ table i r₁ ∨ table i j ≤ table i r₂))
  (col_blue : ∀ (j : Fin 20), ∃ b₁ b₂ : Fin 10, b₁ ≠ b₂ ∧ 
    (∀ (i : Fin 10), table i j ≤ table b₁ j ∨ table i j ≤ table b₂ j)) : 
  ∃ i₁ i₂ i₃ : Fin 10, ∃ j₁ j₂ j₃ : Fin 20, 
    i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₂ ≠ i₃ ∧ j₁ ≠ j₂ ∧ j₁ ≠ j₃ ∧ j₂ ≠ j₃ ∧ 
    ((table i₁ j₁ ≤ table i₁ j₂ ∨ table i₁ j₁ ≤ table i₃ j₂) ∧ 
     (table i₂ j₂ ≤ table i₂ j₁ ∨ table i₂ j₂ ≤ table i₃ j₁) ∧ 
     (table i₃ j₃ ≤ table i₃ j₁ ∨ table i₃ j₃ ≤ table i₂ j₁)) := 
  sorry

end NUMINAMATH_GPT_dual_colored_numbers_l1967_196785


namespace NUMINAMATH_GPT_total_logs_in_stack_l1967_196766

theorem total_logs_in_stack : 
  ∀ (a_1 a_n : ℕ) (n : ℕ), 
  a_1 = 5 → a_n = 15 → n = a_n - a_1 + 1 → 
  (a_1 + a_n) * n / 2 = 110 :=
by
  intros a_1 a_n n h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_logs_in_stack_l1967_196766


namespace NUMINAMATH_GPT_smallest_number_groups_l1967_196739

theorem smallest_number_groups :
  ∃ x : ℕ, (∀ y : ℕ, (y % 12 = 0 ∧ y % 20 = 0 ∧ y % 6 = 0) → y ≥ x) ∧ 
           (x % 12 = 0 ∧ x % 20 = 0 ∧ x % 6 = 0) ∧ x = 60 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_groups_l1967_196739


namespace NUMINAMATH_GPT_equation_transformation_correct_l1967_196702

theorem equation_transformation_correct :
  ∀ (x : ℝ), 
  6 * ((x - 1) / 2 - 1) = 6 * ((3 * x + 1) / 3) → 
  (3 * (x - 1) - 6 = 2 * (3 * x + 1)) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_equation_transformation_correct_l1967_196702


namespace NUMINAMATH_GPT_odd_function_b_value_f_monotonically_increasing_l1967_196724

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := Real.log (Real.sqrt (4 * x^2 + b) + 2 * x)

-- part (1): Prove that if y = f(x) is an odd function, then b = 1
theorem odd_function_b_value :
  (∀ x : ℝ, f x b + f (-x) b = 0) → b = 1 := sorry

-- part (2): Prove that y = f(x) is monotonically increasing for all x in ℝ given b = 1
theorem f_monotonically_increasing (b : ℝ) :
  b = 1 → ∀ x1 x2 : ℝ, x1 < x2 → f x1 b < f x2 b := sorry

end NUMINAMATH_GPT_odd_function_b_value_f_monotonically_increasing_l1967_196724


namespace NUMINAMATH_GPT_largest_class_students_l1967_196755

theorem largest_class_students (x : ℕ) (h1 : x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = 105) : x = 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_class_students_l1967_196755


namespace NUMINAMATH_GPT_tree_planting_equation_l1967_196710

theorem tree_planting_equation (x : ℝ) (h : x > 0) : 
  180 / x - 180 / (1.5 * x) = 2 :=
sorry

end NUMINAMATH_GPT_tree_planting_equation_l1967_196710


namespace NUMINAMATH_GPT_rectangle_perimeter_l1967_196747

-- Definitions of the conditions
def side_length_square : ℕ := 75  -- side length of the square in mm
def height_sum (x y z : ℕ) : Prop := x + y + z = side_length_square  -- sum of heights of the rectangles

-- Perimeter definition
def perimeter (h : ℕ) (w : ℕ) : ℕ := 2 * (h + w)

-- Statement of the problem
theorem rectangle_perimeter (x y z : ℕ) (h_sum : height_sum x y z)
  (h1 : perimeter x side_length_square = (perimeter y side_length_square + perimeter z side_length_square) / 2)
  : perimeter x side_length_square = 200 := by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1967_196747


namespace NUMINAMATH_GPT_smallest_four_digit_multiple_of_18_l1967_196740

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℕ), 999 < n ∧ n < 10000 ∧ 18 ∣ n ∧ ∀ m : ℕ, 999 < m ∧ m < 10000 ∧ 18 ∣ m → n ≤ m ∧ n = 1008 := 
sorry

end NUMINAMATH_GPT_smallest_four_digit_multiple_of_18_l1967_196740


namespace NUMINAMATH_GPT_num_ordered_quadruples_l1967_196751

theorem num_ordered_quadruples (n : ℕ) :
  ∃ (count : ℕ), count = (1 / 3 : ℚ) * (n + 1) * (2 * n^2 + 4 * n + 3) ∧
  (∀ (k1 k2 k3 k4 : ℕ), k1 ≤ n ∧ k2 ≤ n ∧ k3 ≤ n ∧ k4 ≤ n → 
    ((k1 + k3) / 2 = (k2 + k4) / 2) → 
    count = (1 / 3 : ℚ) * (n + 1) * (2 * n^2 + 4 * n + 3)) :=
by sorry

end NUMINAMATH_GPT_num_ordered_quadruples_l1967_196751


namespace NUMINAMATH_GPT_vector_dot_product_l1967_196735

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def sin_deg (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def cos_deg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

theorem vector_dot_product :
  let a := (sin_deg 55, sin_deg 35)
  let b := (sin_deg 25, sin_deg 65)
  dot_product a b = (Real.sqrt 3) / 2 :=
by
  sorry

end NUMINAMATH_GPT_vector_dot_product_l1967_196735


namespace NUMINAMATH_GPT_polynomial_remainder_division_l1967_196774

theorem polynomial_remainder_division :
  ∀ (x : ℝ), (3 * x^7 + 2 * x^5 - 5 * x^3 + x^2 - 9) % (x^2 + 2 * x + 1) = 14 * x - 16 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_division_l1967_196774


namespace NUMINAMATH_GPT_exist_n_for_all_k_l1967_196769

theorem exist_n_for_all_k (k : ℕ) (h_k : k > 1) : 
  ∃ n : ℕ, 
    (n > 0 ∧ ((n.choose k) % n = 0) ∧ (∀ m : ℕ, (2 ≤ m ∧ m < k) → ((n.choose m) % n ≠ 0))) :=
sorry

end NUMINAMATH_GPT_exist_n_for_all_k_l1967_196769


namespace NUMINAMATH_GPT_total_inheritance_money_l1967_196775

-- Defining the conditions
def number_of_inheritors : ℕ := 5
def amount_per_person : ℕ := 105500

-- The proof problem
theorem total_inheritance_money :
  number_of_inheritors * amount_per_person = 527500 :=
by sorry

end NUMINAMATH_GPT_total_inheritance_money_l1967_196775


namespace NUMINAMATH_GPT_bruce_total_cost_l1967_196787

def cost_of_grapes : ℕ := 8 * 70
def cost_of_mangoes : ℕ := 11 * 55
def cost_of_oranges : ℕ := 5 * 45
def cost_of_apples : ℕ := 3 * 90
def cost_of_cherries : ℕ := (45 / 10) * 120  -- use rational division and then multiplication

def total_cost : ℕ :=
  cost_of_grapes + cost_of_mangoes + cost_of_oranges + cost_of_apples + cost_of_cherries

theorem bruce_total_cost : total_cost = 2200 := by
  sorry

end NUMINAMATH_GPT_bruce_total_cost_l1967_196787


namespace NUMINAMATH_GPT_Ksyusha_time_to_school_l1967_196728

/-- Ksyusha runs twice as fast as she walks (both speeds are constant).
On Tuesday, Ksyusha walked a distance twice the distance she ran, and she reached school in 30 minutes.
On Wednesday, Ksyusha ran twice the distance she walked.
Prove that it took her 24 minutes to get from home to school on Wednesday. -/
theorem Ksyusha_time_to_school
  (S v : ℝ)                  -- S for distance unit, v for walking speed
  (htuesday : 2 * (2 * S / v) + S / (2 * v) = 30) -- Condition on Tuesday's travel
  :
  (S / v + 2 * S / (2 * v) = 24)                 -- Claim about Wednesday's travel time
:=
  sorry

end NUMINAMATH_GPT_Ksyusha_time_to_school_l1967_196728


namespace NUMINAMATH_GPT_pipeA_fill_time_l1967_196734

variable (t : ℕ) -- t is the time in minutes for Pipe A to fill the tank

-- Conditions
def pipeA_duration (t : ℕ) : Prop :=
  t > 0

def pipeB_duration (t : ℕ) : Prop :=
  t / 3 > 0

def combined_rate (t : ℕ) : Prop :=
  3 * (1 / (4 / t)) = t

-- Problem
theorem pipeA_fill_time (h1 : pipeA_duration t) (h2 : pipeB_duration t) (h3 : combined_rate t) : t = 12 :=
sorry

end NUMINAMATH_GPT_pipeA_fill_time_l1967_196734


namespace NUMINAMATH_GPT_find_cost_price_l1967_196764

def selling_price : ℝ := 150
def profit_percentage : ℝ := 25

theorem find_cost_price (cost_price : ℝ) (h : profit_percentage = ((selling_price - cost_price) / cost_price) * 100) : 
  cost_price = 120 := 
sorry

end NUMINAMATH_GPT_find_cost_price_l1967_196764


namespace NUMINAMATH_GPT_incorrect_conclusion_l1967_196768

theorem incorrect_conclusion :
  ∃ (a x y : ℝ), 
  (x + 3 * y = 4 - a ∧ x - y = 3 * a) ∧ 
  (∀ (xa ya : ℝ), (xa = 2) → (x = 2 * xa + 1) ∧ (y = 1 - xa) → ¬ (xa + ya = 4 - xa)) :=
sorry

end NUMINAMATH_GPT_incorrect_conclusion_l1967_196768


namespace NUMINAMATH_GPT_min_value_expression_l1967_196719

open Real

theorem min_value_expression : ∀ x y : ℝ, x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 :=
by
  intro x y
  sorry

end NUMINAMATH_GPT_min_value_expression_l1967_196719


namespace NUMINAMATH_GPT_power_neg_two_inverse_l1967_196760

theorem power_neg_two_inverse : (-2 : ℤ) ^ (-2 : ℤ) = (1 : ℚ) / (4 : ℚ) := by
  -- Condition: a^{-n} = 1 / a^n for any non-zero number a and any integer n
  have h: ∀ (a : ℚ) (n : ℤ), a ≠ 0 → a ^ (-n) = 1 / a ^ n := sorry
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_power_neg_two_inverse_l1967_196760


namespace NUMINAMATH_GPT_total_pumped_volume_l1967_196722

def powerJetA_rate : ℕ := 360
def powerJetB_rate : ℕ := 540
def powerJetA_time : ℕ := 30
def powerJetB_time : ℕ := 45

def pump_volume (rate : ℕ) (minutes : ℕ) : ℕ :=
  rate * (minutes / 60)

theorem total_pumped_volume : 
  pump_volume powerJetA_rate powerJetA_time + pump_volume powerJetB_rate powerJetB_time = 585 := 
by
  sorry

end NUMINAMATH_GPT_total_pumped_volume_l1967_196722


namespace NUMINAMATH_GPT_cart_total_books_l1967_196797

theorem cart_total_books (fiction non_fiction autobiographies picture: ℕ) 
  (h1: fiction = 5)
  (h2: non_fiction = fiction + 4)
  (h3: autobiographies = 2 * fiction)
  (h4: picture = 11)
  : fiction + non_fiction + autobiographies + picture = 35 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_cart_total_books_l1967_196797


namespace NUMINAMATH_GPT_sqrt_eq_pm_four_l1967_196793

theorem sqrt_eq_pm_four (a : ℤ) : (a * a = 16) ↔ (a = 4 ∨ a = -4) :=
by sorry

end NUMINAMATH_GPT_sqrt_eq_pm_four_l1967_196793


namespace NUMINAMATH_GPT_exists_x0_l1967_196705

noncomputable def f (x : Real) (a : Real) : Real :=
  Real.exp x - a * Real.sin x

theorem exists_x0 (a : Real) (h : a = 1) :
  ∃ x0 ∈ Set.Ioo (-Real.pi / 2) 0, 1 < f x0 a ∧ f x0 a < Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_exists_x0_l1967_196705


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1967_196771

variable (a x : ℝ)

theorem quadratic_inequality_solution (h : 0 < a ∧ a < 1) : (x - a) * (x - (1 / a)) > 0 ↔ (x < a ∨ x > 1 / a) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1967_196771


namespace NUMINAMATH_GPT_number_of_seats_in_nth_row_l1967_196708

theorem number_of_seats_in_nth_row (n : ℕ) :
    ∃ m : ℕ, m = 3 * n + 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_seats_in_nth_row_l1967_196708


namespace NUMINAMATH_GPT_part_one_part_two_l1967_196790

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2

theorem part_one (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m * n > 1) : f m >= 0 ∨ f n >= 0 :=
sorry

theorem part_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (hf : f a = f b) : a + b < 4 / 3 :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l1967_196790


namespace NUMINAMATH_GPT_certain_positive_integer_value_l1967_196779

-- Define factorial
def fact : ℕ → ℕ 
| 0     => 1
| (n+1) => (n+1) * fact n

-- Statement of the problem
theorem certain_positive_integer_value (i k m a : ℕ) :
  (fact 8 = 2^i * 3^k * 5^m * 7^a) ∧ (i + k + m + a = 11) → a = 1 :=
by 
  sorry

end NUMINAMATH_GPT_certain_positive_integer_value_l1967_196779


namespace NUMINAMATH_GPT_greatest_integer_le_x_squared_div_50_l1967_196743

-- Define the conditions as given in the problem
def trapezoid (b h : ℝ) (x : ℝ) : Prop :=
  let baseDifference := 50
  let longerBase := b + baseDifference
  let midline := (b + longerBase) / 2
  let heightRatioFactor := 2
  let xSquared := 6875
  let regionAreaRatio := 2 / 1 -- represented as 2
  (let areaRatio := (b + midline) / (b + baseDifference / 2)
   areaRatio = regionAreaRatio) ∧
  (x = Real.sqrt xSquared) ∧
  (b = 50)

-- Define the theorem that captures the question
theorem greatest_integer_le_x_squared_div_50 (b h x : ℝ) (h_trapezoid : trapezoid b h x) :
  ⌊ (x^2) / 50 ⌋ = 137 :=
by sorry

end NUMINAMATH_GPT_greatest_integer_le_x_squared_div_50_l1967_196743


namespace NUMINAMATH_GPT_find_w_l1967_196789

variable (x y z w : ℝ)

theorem find_w (h : (x + y + z) / 3 = (y + z + w) / 3 + 10) : w = x - 30 := by 
  sorry

end NUMINAMATH_GPT_find_w_l1967_196789


namespace NUMINAMATH_GPT_male_students_count_l1967_196736

theorem male_students_count :
  ∃ (N M : ℕ), 
  (N % 4 = 2) ∧ 
  (N % 5 = 1) ∧ 
  (N = M + 15) ∧ 
  (15 > M) ∧ 
  (M = 11) :=
sorry

end NUMINAMATH_GPT_male_students_count_l1967_196736


namespace NUMINAMATH_GPT_polygon_diagonals_l1967_196763

theorem polygon_diagonals (n : ℕ) (h : 20 = n) : (n * (n - 3)) / 2 = 170 :=
by
  sorry

end NUMINAMATH_GPT_polygon_diagonals_l1967_196763


namespace NUMINAMATH_GPT_smallest_positive_integer_remainder_l1967_196723

theorem smallest_positive_integer_remainder
  (b : ℕ) (h1 : b % 4 = 3) (h2 : b % 6 = 5) :
  b = 11 := by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_remainder_l1967_196723


namespace NUMINAMATH_GPT_lily_account_balance_l1967_196777

def initial_balance : ℕ := 55

def shirt_cost : ℕ := 7

def second_spend_multiplier : ℕ := 3

def first_remaining_balance (initial_balance shirt_cost: ℕ) : ℕ :=
  initial_balance - shirt_cost

def second_spend (shirt_cost second_spend_multiplier: ℕ) : ℕ :=
  shirt_cost * second_spend_multiplier

def final_remaining_balance (first_remaining_balance second_spend: ℕ) : ℕ :=
  first_remaining_balance - second_spend

theorem lily_account_balance :
  final_remaining_balance (first_remaining_balance initial_balance shirt_cost) (second_spend shirt_cost second_spend_multiplier) = 27 := by
    sorry

end NUMINAMATH_GPT_lily_account_balance_l1967_196777


namespace NUMINAMATH_GPT_interest_rate_correct_l1967_196759

-- Definitions based on the problem conditions
def P : ℝ := 7000 -- Principal investment amount
def A : ℝ := 8470 -- Future value of the investment
def n : ℕ := 1 -- Number of times interest is compounded per year
def t : ℕ := 2 -- Number of years

-- The interest rate r to be proven
def r : ℝ := 0.1 -- Annual interest rate

-- Statement of the problem that needs to be proven in Lean
theorem interest_rate_correct :
  A = P * (1 + r / n)^(n * t) :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_correct_l1967_196759


namespace NUMINAMATH_GPT_daps_equivalent_to_dips_l1967_196750

theorem daps_equivalent_to_dips (daps dops dips : ℕ) 
  (h1 : 4 * daps = 3 * dops) 
  (h2 : 2 * dops = 7 * dips) :
  35 * dips = 20 * daps :=
by
  sorry

end NUMINAMATH_GPT_daps_equivalent_to_dips_l1967_196750


namespace NUMINAMATH_GPT_divisor_of_136_l1967_196730

theorem divisor_of_136 (d : ℕ) (h : 136 = 9 * d + 1) : d = 15 := 
by {
  -- Since the solution steps are skipped, we use sorry to indicate a placeholder.
  sorry
}

end NUMINAMATH_GPT_divisor_of_136_l1967_196730


namespace NUMINAMATH_GPT_fraction_of_total_amount_l1967_196713

theorem fraction_of_total_amount (p q r : ℕ) (h1 : p + q + r = 4000) (h2 : r = 1600) :
  r / (p + q + r) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_total_amount_l1967_196713


namespace NUMINAMATH_GPT_fraction_quaduple_l1967_196773

variable (b a : ℤ)

theorem fraction_quaduple (h₁ : a ≠ 0) : (2 * b) / (a / 2) = 4 * (b / a) :=
by
  sorry

end NUMINAMATH_GPT_fraction_quaduple_l1967_196773


namespace NUMINAMATH_GPT_find_a3_l1967_196748

-- Definitions from conditions
def arithmetic_sum (a1 a3 : ℕ) := (3 / 2) * (a1 + a3)
def common_difference := 2
def S3 := 12

-- Theorem to prove that a3 = 6
theorem find_a3 (a1 a3 : ℕ) (h₁ : arithmetic_sum a1 a3 = S3) (h₂ : a3 = a1 + common_difference * 2) : a3 = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_a3_l1967_196748


namespace NUMINAMATH_GPT_hexagon_side_equality_l1967_196776

variables {A B C D E F : Type} [AddCommGroup A] [Module ℝ A] [AddCommGroup B] [Module ℝ B]
          [AddCommGroup C] [Module ℝ C] [AddCommGroup D] [Module ℝ D]
          [AddCommGroup E] [Module ℝ E] [AddCommGroup F] [Module ℝ F]

def parallel (x y : A) : Prop := ∀ r : ℝ, x = r • y
noncomputable def length_eq (x y : A) : Prop := ∃ r : ℝ, r • x = y

variables (AB DE BC EF CD FA : A)
variables (h1 : parallel AB DE)
variables (h2 : parallel BC EF)
variables (h3 : parallel CD FA)
variables (h4 : length_eq AB DE)

theorem hexagon_side_equality :
  length_eq BC EF ∧ length_eq CD FA :=
by
  sorry

end NUMINAMATH_GPT_hexagon_side_equality_l1967_196776


namespace NUMINAMATH_GPT_simplify_expression_find_value_a_m_2n_l1967_196758

-- Proof Problem 1
theorem simplify_expression : ( (-2 : ℤ) * x )^3 * x^2 + ( (3 : ℤ) * x^4 )^2 / x^3 = x^5 := by
  sorry

-- Proof Problem 2
theorem find_value_a_m_2n (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + 2*n) = 18 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_find_value_a_m_2n_l1967_196758


namespace NUMINAMATH_GPT_fruits_given_away_l1967_196799

-- Definitions based on the conditions
def initial_pears := 10
def initial_oranges := 20
def initial_apples := 2 * initial_pears
def initial_fruits := initial_pears + initial_oranges + initial_apples
def fruits_left := 44

-- Theorem to prove the total number of fruits given to her sister
theorem fruits_given_away : initial_fruits - fruits_left = 6 := by
  sorry

end NUMINAMATH_GPT_fruits_given_away_l1967_196799


namespace NUMINAMATH_GPT_bc_over_ad_l1967_196711

-- Define the rectangular prism
structure RectangularPrism :=
(length width height : ℝ)

-- Define the problem parameters
def B : RectangularPrism := ⟨2, 4, 5⟩

-- Define the volume form of S(r)
def volume (a b c d : ℝ) (r : ℝ) : ℝ := a * r^3 + b * r^2 + c * r + d

-- Prove that the relationship holds
theorem bc_over_ad (a b c d : ℝ) (r : ℝ) (h_a : a = (4 * π) / 3) (h_b : b = 11 * π) (h_c : c = 76) (h_d : d = 40) :
  (b * c) / (a * d) = 15.67 := by
  sorry

end NUMINAMATH_GPT_bc_over_ad_l1967_196711


namespace NUMINAMATH_GPT_parabola_focus_l1967_196732

theorem parabola_focus (a : ℝ) : (∀ x : ℝ, y = a * x^2) ∧ ∃ f : ℝ × ℝ, f = (0, 1) → a = (1/4) := 
sorry

end NUMINAMATH_GPT_parabola_focus_l1967_196732


namespace NUMINAMATH_GPT_students_after_joining_l1967_196772

theorem students_after_joining (N : ℕ) (T : ℕ)
  (h1 : T = 48 * N)
  (h2 : 120 * 32 / (N + 120) + (T / (N + 120)) = 44)
  : N + 120 = 480 :=
by
  sorry

end NUMINAMATH_GPT_students_after_joining_l1967_196772


namespace NUMINAMATH_GPT_f_decreasing_interval_triangle_abc_l1967_196727

noncomputable def f (x : Real) : Real := 2 * (Real.sin x)^2 + Real.cos ((Real.pi) / 3 - 2 * x)

theorem f_decreasing_interval :
  ∃ (a b : Real), a = Real.pi / 3 ∧ b = 5 * Real.pi / 6 ∧ 
  ∀ x y, (a ≤ x ∧ x < y ∧ y ≤ b) → f y ≤ f x := 
sorry

variables {a b c : Real} (A B C : Real) 

theorem triangle_abc (h1 : A = Real.pi / 3) 
    (h2 : f A = 2)
    (h3 : a = 2 * b)
    (h4 : Real.sin C = 2 * Real.sin B):
  a / b = Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_f_decreasing_interval_triangle_abc_l1967_196727


namespace NUMINAMATH_GPT_solution_A_l1967_196721

def P : Set ℕ := {1, 2, 3, 4}

theorem solution_A (A : Set ℕ) (h1 : A ⊆ P) 
  (h2 : ∀ x ∈ A, 2 * x ∉ A) 
  (h3 : ∀ x ∈ (P \ A), 2 * x ∉ (P \ A)): 
    A = {2} ∨ A = {1, 4} ∨ A = {2, 3} ∨ A = {1, 3, 4} :=
sorry

end NUMINAMATH_GPT_solution_A_l1967_196721


namespace NUMINAMATH_GPT_larger_of_two_numbers_l1967_196778

theorem larger_of_two_numbers (x y : ℕ) (h1 : x * y = 24) (h2 : x + y = 11) : max x y = 8 :=
sorry

end NUMINAMATH_GPT_larger_of_two_numbers_l1967_196778


namespace NUMINAMATH_GPT_largest_value_x_l1967_196749

-- Definition of the conditions
def equation (x : ℚ) : Prop :=
  (16 * x^2 - 40 * x + 15) / (4 * x - 3) + 7 * x = 8 * x - 2

-- Statement of the proof 
theorem largest_value_x : ∀ x : ℚ, equation x → x ≤ 9 / 4 := sorry

end NUMINAMATH_GPT_largest_value_x_l1967_196749


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l1967_196706

-- Problem 1
theorem problem1_solution (x y : ℝ) (h1 : y = 2 * x - 3) (h2 : 2 * x + y = 5) : 
  x = 2 ∧ y = 1 :=
  sorry

-- Problem 2
theorem problem2_solution (x y : ℝ) (h1 : 3 * x + 4 * y = 5) (h2 : 5 * x - 2 * y = 17) : 
  x = 3 ∧ y = -1 :=
  sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l1967_196706


namespace NUMINAMATH_GPT_chandler_tickets_total_cost_l1967_196746

theorem chandler_tickets_total_cost :
  let movie_ticket_cost := 30
  let num_movie_tickets := 8
  let num_football_tickets := 5
  let num_concert_tickets := 3
  let num_theater_tickets := 4
  let theater_ticket_cost := 40
  let discount := 0.10
  let total_movie_cost := num_movie_tickets * movie_ticket_cost
  let football_ticket_cost := total_movie_cost / 2
  let total_football_cost := num_football_tickets * football_ticket_cost
  let concert_ticket_cost := football_ticket_cost - 10
  let total_concert_cost := num_concert_tickets * concert_ticket_cost
  let discounted_theater_ticket_cost := theater_ticket_cost * (1 - discount)
  let total_theater_cost := num_theater_tickets * discounted_theater_ticket_cost
  let total_cost := total_movie_cost + total_football_cost + total_concert_cost + total_theater_cost
  total_cost = 1314 := by
  sorry

end NUMINAMATH_GPT_chandler_tickets_total_cost_l1967_196746


namespace NUMINAMATH_GPT_petya_maximum_margin_l1967_196757

def max_margin_votes (total_votes first_period_margin last_period_margin : ℕ) (petya_vasaya_margin : ℕ) :=
  ∀ (P1 P2 V1 V2 : ℕ),
    (P1 + P2 + V1 + V2 = total_votes) →
    (P1 = V1 + first_period_margin) →
    (V2 = P2 + last_period_margin) →
    (P1 + P2 > V1 + V2) →
    petya_vasaya_margin = P1 + P2 - (V1 + V2)

theorem petya_maximum_margin
  (total_votes first_period_margin last_period_margin : ℕ)
  (h_total_votes: total_votes = 27)
  (h_first_period_margin: first_period_margin = 9)
  (h_last_period_margin: last_period_margin = 9):
  ∃ (petya_vasaya_margin : ℕ), max_margin_votes total_votes first_period_margin last_period_margin petya_vasaya_margin ∧ petya_vasaya_margin = 9 :=
by {
    sorry
}

end NUMINAMATH_GPT_petya_maximum_margin_l1967_196757


namespace NUMINAMATH_GPT_geometric_sequence_a5_l1967_196754

theorem geometric_sequence_a5 (α : Type) [LinearOrderedField α] (a : ℕ → α)
  (h1 : ∀ n, a (n + 1) = a n * 2)
  (h2 : ∀ n, a n > 0)
  (h3 : a 3 * a 11 = 16) :
  a 5 = 1 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a5_l1967_196754


namespace NUMINAMATH_GPT_range_of_ω_l1967_196703

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ :=
  Real.cos (ω * x + ϕ)

theorem range_of_ω :
  ∀ (ω : ℝ) (ϕ : ℝ),
    (0 < ω) →
    (-π ≤ ϕ) →
    (ϕ ≤ 0) →
    (∀ x, f x ω ϕ = -f (-x) ω ϕ) →
    (∀ x1 x2, (x1 < x2) → (-π/4 ≤ x1 ∧ x1 ≤ 3*π/16) ∧ (-π/4 ≤ x2 ∧ x2 ≤ 3*π/16) → f x1 ω ϕ ≤ f x2 ω ϕ) →
    (0 < ω ∧ ω ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_ω_l1967_196703


namespace NUMINAMATH_GPT_min_value_l1967_196761

theorem min_value (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 2) : 
  (∃ x y z, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2 ∧ (1/3 * x^3 + y^2 + z = 13/12)) :=
sorry

end NUMINAMATH_GPT_min_value_l1967_196761


namespace NUMINAMATH_GPT_gcd_lcm_sum_l1967_196712

-- Define the necessary components: \( A \) as the greatest common factor and \( B \) as the least common multiple of 16, 32, and 48
def A := Int.gcd (Int.gcd 16 32) 48
def B := Int.lcm (Int.lcm 16 32) 48

-- Statement that needs to be proved
theorem gcd_lcm_sum : A + B = 112 := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_sum_l1967_196712


namespace NUMINAMATH_GPT_problem_solution_l1967_196765

theorem problem_solution
  (a d : ℝ)
  (h : (∀ x : ℝ, (x - 3) * (x + a) = x^2 + d * x - 18)) :
  d = 3 := 
sorry

end NUMINAMATH_GPT_problem_solution_l1967_196765


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_min_value_l1967_196770

theorem arithmetic_geometric_sequence_min_value (x y a b c d : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (arithmetic_seq : a = (x + y) / 2) (geometric_seq : c * d = x * y) :
  ( (a + b) ^ 2 ) / (c * d) ≥ 4 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_min_value_l1967_196770


namespace NUMINAMATH_GPT_quadratic_equal_real_roots_l1967_196709

theorem quadratic_equal_real_roots (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x + 1 = 0 ∧ (x = a*x / 2)) ↔ a = 2 ∨ a = -2 :=
by sorry

end NUMINAMATH_GPT_quadratic_equal_real_roots_l1967_196709
