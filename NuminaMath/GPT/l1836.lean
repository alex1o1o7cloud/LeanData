import Mathlib

namespace set_intersection_l1836_183682
noncomputable def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 2 }
noncomputable def B : Set ℝ := {x : ℝ | x ≥ 1 }

theorem set_intersection (x : ℝ) : x ∈ A ∩ B ↔ x ∈ A := sorry

end set_intersection_l1836_183682


namespace actual_total_discount_discount_difference_l1836_183634

variable {original_price : ℝ}
variable (first_discount second_discount claimed_discount actual_discount : ℝ)

-- Definitions based on the problem conditions
def discount_1 (p : ℝ) : ℝ := (1 - first_discount) * p
def discount_2 (p : ℝ) : ℝ := (1 - second_discount) * discount_1 first_discount p

-- Statements we need to prove
theorem actual_total_discount (original_price : ℝ)
  (first_discount : ℝ := 0.40) (second_discount : ℝ := 0.30) (claimed_discount : ℝ := 0.70) :
  actual_discount = 1 - discount_2 first_discount second_discount original_price := 
by 
  sorry

theorem discount_difference (original_price : ℝ)
  (first_discount : ℝ := 0.40) (second_discount : ℝ := 0.30) (claimed_discount : ℝ := 0.70)
  (actual_discount : ℝ := 0.58) :
  claimed_discount - actual_discount = 0.12 := 
by 
  sorry

end actual_total_discount_discount_difference_l1836_183634


namespace intersection_A_complement_B_l1836_183681

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x - 3 > 0}
def comR (S : Set ℝ) : Set ℝ := {x | ¬ (x ∈ S)}

theorem intersection_A_complement_B : A ∩ (comR B) = {x | -1 < x ∧ x ≤ 3} := 
by
  sorry

end intersection_A_complement_B_l1836_183681


namespace triangle_inradius_l1836_183674

theorem triangle_inradius (p A : ℝ) (h_p : p = 20) (h_A : A = 30) : 
  ∃ r : ℝ, r = 3 ∧ A = r * p / 2 :=
by
  sorry

end triangle_inradius_l1836_183674


namespace olivia_packs_of_basketball_cards_l1836_183633

-- Definitions for the given conditions
def pack_cost : ℕ := 3
def deck_cost : ℕ := 4
def number_of_decks : ℕ := 5
def total_money : ℕ := 50
def change_received : ℕ := 24

-- Statement to be proved
theorem olivia_packs_of_basketball_cards (x : ℕ) (hx : pack_cost * x + deck_cost * number_of_decks = total_money - change_received) : x = 2 :=
by 
  sorry

end olivia_packs_of_basketball_cards_l1836_183633


namespace steve_speed_back_home_l1836_183662

-- Define a structure to hold the given conditions:
structure Conditions where
  home_to_work_distance : Float := 35 -- km
  v  : Float -- speed on the way to work in km/h
  additional_stop_time : Float := 0.25 -- hours
  total_weekly_time : Float := 30 -- hours

-- Define the main proposition:
theorem steve_speed_back_home (c: Conditions)
  (h1 : 5 * ((c.home_to_work_distance / c.v) + (c.home_to_work_distance / (2 * c.v))) + 3 * c.additional_stop_time = c.total_weekly_time) :
  2 * c.v = 18 := by
  sorry

end steve_speed_back_home_l1836_183662


namespace cos_b4_b6_l1836_183672

theorem cos_b4_b6 (a b : ℕ → ℝ) (d : ℝ) 
  (ha_geom : ∀ n, a (n + 1) / a n = a 1)
  (hb_arith : ∀ n, b (n + 1) = b n + d)
  (ha_prod : a 1 * a 5 * a 9 = -8)
  (hb_sum : b 2 + b 5 + b 8 = 6 * Real.pi) : 
  Real.cos ((b 4 + b 6) / (1 - a 3 * a 7)) = -1 / 2 :=
sorry

end cos_b4_b6_l1836_183672


namespace sin_cos_sum_identity_l1836_183697

noncomputable def trigonometric_identity (x y z w : ℝ) := 
  (Real.sin x * Real.cos y + Real.sin z * Real.cos w) = Real.sqrt 2 / 2

theorem sin_cos_sum_identity :
  trigonometric_identity 347 148 77 58 :=
by sorry

end sin_cos_sum_identity_l1836_183697


namespace janet_total_miles_run_l1836_183658

/-- Janet was practicing for a marathon. She practiced for 9 days, running 8 miles each day.
Prove that Janet ran 72 miles in total. -/
theorem janet_total_miles_run (days_practiced : ℕ) (miles_per_day : ℕ) (total_miles : ℕ) 
  (h1 : days_practiced = 9) (h2 : miles_per_day = 8) : total_miles = 72 := by
  sorry

end janet_total_miles_run_l1836_183658


namespace distance_between_foci_of_ellipse_l1836_183661

theorem distance_between_foci_of_ellipse : 
  let a := 5
  let b := 3
  2 * Real.sqrt (a^2 - b^2) = 8 := by
  let a := 5
  let b := 3
  sorry

end distance_between_foci_of_ellipse_l1836_183661


namespace probability_of_one_of_each_color_l1836_183690

-- Definitions based on the conditions
def total_marbles : ℕ := 12
def marbles_of_each_color : ℕ := 3
def number_of_selected_marbles : ℕ := 4

-- Calculation based on problem requirements
def total_ways_to_choose_marbles : ℕ := Nat.choose total_marbles number_of_selected_marbles
def favorable_ways_to_choose : ℕ := marbles_of_each_color ^ number_of_selected_marbles

-- The main theorem to prove the probability
theorem probability_of_one_of_each_color :
  (favorable_ways_to_choose : ℚ) / total_ways_to_choose = 9 / 55 := by
  sorry

end probability_of_one_of_each_color_l1836_183690


namespace equation_of_line_l_l1836_183651

theorem equation_of_line_l (P : ℝ × ℝ) (hP : P = (1, -1)) (θ₁ θ₂ : ℕ) (hθ₁ : θ₁ = 45) (hθ₂ : θ₂ = θ₁ * 2) (hθ₂_90 : θ₂ = 90) : 
  ∃ l : ℝ → ℝ, (∀ x, l x = l (P.fst)) := 
sorry

end equation_of_line_l_l1836_183651


namespace largest_angle_of_convex_hexagon_l1836_183621

theorem largest_angle_of_convex_hexagon (a d : ℕ) (h_seq : ∀ i, a + i * d < 180 ∧ a + i * d > 0)
  (h_sum : 6 * a + 15 * d = 720)
  (h_seq_arithmetic : ∀ (i j : ℕ), (a + i * d) < (a + j * d) ↔ i < j) :
  ∃ m : ℕ, (m = a + 5 * d ∧ m = 175) :=
by
  sorry

end largest_angle_of_convex_hexagon_l1836_183621


namespace consecutive_page_numbers_sum_l1836_183664

theorem consecutive_page_numbers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 35280) :
  n + (n + 1) + (n + 2) = 96 := sorry

end consecutive_page_numbers_sum_l1836_183664


namespace angle_not_45_or_135_l1836_183641

variable {a b S : ℝ}
variable {C : ℝ} (h : S = (1/2) * a * b * Real.cos C)

theorem angle_not_45_or_135 (h : S = (1/2) * a * b * Real.cos C) : ¬ (C = 45 ∨ C = 135) :=
sorry

end angle_not_45_or_135_l1836_183641


namespace five_digit_odd_and_multiples_of_5_sum_l1836_183606

theorem five_digit_odd_and_multiples_of_5_sum :
  let A := 9 * 10^3 * 5
  let B := 9 * 10^3 * 1
  A + B = 45000 := by
sorry

end five_digit_odd_and_multiples_of_5_sum_l1836_183606


namespace division_of_fractions_l1836_183684

theorem division_of_fractions : (5 / 6) / (1 + 3 / 9) = 5 / 8 := by
  sorry

end division_of_fractions_l1836_183684


namespace frequency_not_equal_probability_l1836_183689

theorem frequency_not_equal_probability
  (N : ℕ) -- Total number of trials
  (N1 : ℕ) -- Number of times student A is selected
  (hN : N > 0) -- Ensure the number of trials is positive
  (rand_int_gen : ℕ → ℕ) -- A function generating random integers from 1 to 6
  (h_gen : ∀ n, 1 ≤ rand_int_gen n ∧ rand_int_gen n ≤ 6) -- Generator produces numbers between 1 to 6
: (N1/N : ℚ) ≠ (1/6 : ℚ) := 
sorry

end frequency_not_equal_probability_l1836_183689


namespace percentage_j_of_k_theorem_l1836_183683

noncomputable def percentage_j_of_k 
  (j k l m : ℝ) (x : ℝ) 
  (h1 : j * (x / 100) = k * (25 / 100))
  (h2 : k * (150 / 100) = l * (50 / 100))
  (h3 : l * (175 / 100) = m * (75 / 100))
  (h4 : m * (20 / 100) = j * (700 / 100)) : Prop :=
  x = 500

theorem percentage_j_of_k_theorem 
  (j k l m : ℝ) (x : ℝ)
  (h1 : j * (x / 100) = k * (25 / 100))
  (h2 : k * (150 / 100) = l * (50 / 100))
  (h3 : l * (175 / 100) = m * (75 / 100))
  (h4 : m * (20 / 100) = j * (700 / 100)) : percentage_j_of_k j k l m x h1 h2 h3 h4 :=
by 
  sorry

end percentage_j_of_k_theorem_l1836_183683


namespace compound_interest_correct_l1836_183671

variables (SI : ℚ) (R : ℚ) (T : ℕ) (P : ℚ)

def calculate_principal (SI R T : ℚ) : ℚ := SI * 100 / (R * T)

def calculate_compound_interest (P R : ℚ) (T : ℕ) : ℚ :=
  P * ((1 + R / 100)^T - 1)

theorem compound_interest_correct (h1: SI = 52) (h2: R = 5) (h3: T = 2) :
  calculate_compound_interest (calculate_principal SI R T) R T = 53.30 :=
by
  sorry

end compound_interest_correct_l1836_183671


namespace prime_condition_l1836_183615

theorem prime_condition (p : ℕ) [Fact (Nat.Prime p)] :
  (∀ (a : ℕ), (1 < a ∧ a < p / 2) → (∃ (b : ℕ), (p / 2 < b ∧ b < p) ∧ p ∣ (a * b - 1))) ↔ (p = 5 ∨ p = 7 ∨ p = 13) := by
  sorry

end prime_condition_l1836_183615


namespace tom_gaming_system_value_l1836_183609

theorem tom_gaming_system_value
    (V : ℝ) 
    (h1 : 0.80 * V + 80 - 10 = 160 + 30) 
    : V = 150 :=
by
  -- Logical steps for the proof will be added here.
  sorry

end tom_gaming_system_value_l1836_183609


namespace f_greater_than_fp_3_2_l1836_183649

noncomputable def f (x : ℝ) (a : ℝ) := a * (x - Real.log x) + (2 * x - 1) / (x ^ 2)
noncomputable def f' (x : ℝ) (a : ℝ) := (a * x^3 - a * x^2 + 2 - 2*x) / x^3

theorem f_greater_than_fp_3_2 (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) :
  f x 1 > f' x 1 + 3 / 2 := sorry

end f_greater_than_fp_3_2_l1836_183649


namespace wind_power_in_scientific_notation_l1836_183645

theorem wind_power_in_scientific_notation :
  (56 * 10^6) = (5.6 * 10^7) :=
by
  sorry

end wind_power_in_scientific_notation_l1836_183645


namespace series_sum_eq_half_l1836_183695

theorem series_sum_eq_half :
  ∑' (n : ℕ), 2^n / (3^(2^n) + 1) = 1 / 2 :=
sorry

end series_sum_eq_half_l1836_183695


namespace triangle_medians_inequality_l1836_183643

-- Define the parameters
variables {a b c t_a t_b t_c D : ℝ}

-- Assume the sides and medians of the triangle and the diameter of the circumcircle
axiom sides_of_triangle (a b c : ℝ) : Prop
axiom medians_of_triangle (t_a t_b t_c : ℝ) : Prop
axiom diameter_of_circumcircle (D : ℝ) : Prop

-- The theorem to prove
theorem triangle_medians_inequality
  (h_sides : sides_of_triangle a b c)
  (h_medians : medians_of_triangle t_a t_b t_c)
  (h_diameter : diameter_of_circumcircle D)
  : (a^2 + b^2) / t_c + (b^2 + c^2) / t_a + (c^2 + a^2) / t_b ≤ 6 * D :=
sorry -- proof omitted

end triangle_medians_inequality_l1836_183643


namespace mens_wages_l1836_183678

theorem mens_wages
  (M : ℝ) (WW : ℝ) (B : ℝ)
  (h1 : 5 * M = WW)
  (h2 : WW = 8 * B)
  (h3 : 5 * M + WW + 8 * B = 60) :
  5 * M = 30 :=
by
  sorry

end mens_wages_l1836_183678


namespace ratio_of_ages_in_two_years_l1836_183627

theorem ratio_of_ages_in_two_years
  (S M : ℕ) 
  (h1 : M = S + 22)
  (h2 : S = 20)
  (h3 : ∃ k : ℕ, M + 2 = k * (S + 2)) :
  (M + 2) / (S + 2) = 2 :=
by
  sorry

end ratio_of_ages_in_two_years_l1836_183627


namespace carB_distance_traveled_l1836_183610

-- Define the initial conditions
def initial_separation : ℝ := 150
def distance_carA_main_road : ℝ := 25
def distance_between_cars : ℝ := 38

-- Define the question as a theorem where we need to show the distance Car B traveled
theorem carB_distance_traveled (initial_separation distance_carA_main_road distance_between_cars : ℝ) :
  initial_separation - (distance_carA_main_road + distance_between_cars) = 87 :=
  sorry

end carB_distance_traveled_l1836_183610


namespace mork_effective_tax_rate_theorem_mindy_effective_tax_rate_theorem_combined_effective_tax_rate_theorem_l1836_183657

noncomputable def Mork_base_income (M : ℝ) : ℝ := M
noncomputable def Mindy_base_income (M : ℝ) : ℝ := 4 * M
noncomputable def Mork_total_income (M : ℝ) : ℝ := 1.5 * M
noncomputable def Mindy_total_income (M : ℝ) : ℝ := 6 * M

noncomputable def Mork_total_tax (M : ℝ) : ℝ :=
  0.4 * M + 0.5 * 0.5 * M
noncomputable def Mindy_total_tax (M : ℝ) : ℝ :=
  0.3 * 4 * M + 0.35 * 2 * M

noncomputable def Mork_effective_tax_rate (M : ℝ) : ℝ :=
  (Mork_total_tax M) / (Mork_total_income M)

noncomputable def Mindy_effective_tax_rate (M : ℝ) : ℝ :=
  (Mindy_total_tax M) / (Mindy_total_income M)

noncomputable def combined_effective_tax_rate (M : ℝ) : ℝ :=
  (Mork_total_tax M + Mindy_total_tax M) / (Mork_total_income M + Mindy_total_income M)

theorem mork_effective_tax_rate_theorem (M : ℝ) : Mork_effective_tax_rate M = 43.33 / 100 := sorry
theorem mindy_effective_tax_rate_theorem (M : ℝ) : Mindy_effective_tax_rate M = 31.67 / 100 := sorry
theorem combined_effective_tax_rate_theorem (M : ℝ) : combined_effective_tax_rate M = 34 / 100 := sorry

end mork_effective_tax_rate_theorem_mindy_effective_tax_rate_theorem_combined_effective_tax_rate_theorem_l1836_183657


namespace residue_of_5_pow_2023_mod_11_l1836_183654

theorem residue_of_5_pow_2023_mod_11 : (5 ^ 2023) % 11 = 4 := by
  sorry

end residue_of_5_pow_2023_mod_11_l1836_183654


namespace sum_remainder_l1836_183656

theorem sum_remainder (a b c : ℕ) (h1 : a % 53 = 33) (h2 : b % 53 = 14) (h3 : c % 53 = 9) : 
  (a + b + c) % 53 = 3 := 
by 
  sorry

end sum_remainder_l1836_183656


namespace ball_probability_l1836_183677

theorem ball_probability :
  ∀ (total_balls red_balls white_balls : ℕ),
  total_balls = 10 → red_balls = 6 → white_balls = 4 →
  -- Given conditions: Total balls, red balls, and white balls.
  -- First ball drawn is red
  ∀ (first_ball_red : true),
  -- Prove that the probability of the second ball being red is 5/9.
  (red_balls - 1) / (total_balls - 1) = 5/9 :=
by
  intros total_balls red_balls white_balls h_total h_red h_white first_ball_red
  sorry

end ball_probability_l1836_183677


namespace field_trip_total_l1836_183669

-- Define the conditions
def vans := 2
def buses := 3
def people_per_van := 8
def people_per_bus := 20

-- The total number of people
def total_people := (vans * people_per_van) + (buses * people_per_bus)

theorem field_trip_total : total_people = 76 :=
by
  -- skip the proof here
  sorry

end field_trip_total_l1836_183669


namespace students_not_enrolled_in_course_l1836_183600

def total_students : ℕ := 150
def french_students : ℕ := 61
def german_students : ℕ := 32
def spanish_students : ℕ := 45
def french_and_german : ℕ := 15
def french_and_spanish : ℕ := 12
def german_and_spanish : ℕ := 10
def all_three_courses : ℕ := 5

theorem students_not_enrolled_in_course : total_students - 
    (french_students + german_students + spanish_students - 
     french_and_german - french_and_spanish - german_and_spanish + 
     all_three_courses) = 44 := by
  sorry

end students_not_enrolled_in_course_l1836_183600


namespace solve_for_n_l1836_183640

theorem solve_for_n (n : ℕ) (h : 2^n * 8^n = 64^(n - 30)) : n = 90 :=
by {
  sorry
}

end solve_for_n_l1836_183640


namespace sacred_k_words_n10_k4_l1836_183611

/- Definitions for the problem -/
def sacred_k_words_count (n k : ℕ) (hk : k < n / 2) : ℕ :=
  n * Nat.choose (n - k - 1) (k - 1) * (Nat.factorial k / k)

theorem sacred_k_words_n10_k4 : sacred_k_words_count 10 4 (by norm_num : 4 < 10 / 2) = 600 := by
  sorry

end sacred_k_words_n10_k4_l1836_183611


namespace tan_22_5_eq_half_l1836_183644

noncomputable def tan_h_LHS (θ : Real) := Real.tan θ / (1 - Real.tan θ ^ 2)

theorem tan_22_5_eq_half :
    tan_h_LHS (Real.pi / 8) = 1 / 2 :=
  sorry

end tan_22_5_eq_half_l1836_183644


namespace function_range_is_interval_l1836_183668

theorem function_range_is_interval :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ (256 * x^9 - 576 * x^7 + 432 * x^5 - 120 * x^3 + 9 * x) ∧ 
  (256 * x^9 - 576 * x^7 + 432 * x^5 - 120 * x^3 + 9 * x) ≤ 1 := 
by
  sorry

end function_range_is_interval_l1836_183668


namespace winnie_balloons_remainder_l1836_183601

theorem winnie_balloons_remainder :
  let red_balloons := 20
  let white_balloons := 40
  let green_balloons := 70
  let chartreuse_balloons := 90
  let violet_balloons := 15
  let friends := 10
  let total_balloons := red_balloons + white_balloons + green_balloons + chartreuse_balloons + violet_balloons
  total_balloons % friends = 5 :=
by
  sorry

end winnie_balloons_remainder_l1836_183601


namespace earnings_from_cauliflower_correct_l1836_183673

-- Define the earnings from each vegetable
def earnings_from_broccoli : ℕ := 57
def earnings_from_carrots : ℕ := 2 * earnings_from_broccoli
def earnings_from_spinach : ℕ := (earnings_from_carrots / 2) + 16
def total_earnings : ℕ := 380

-- Define the total earnings from vegetables other than cauliflower
def earnings_from_others : ℕ := earnings_from_broccoli + earnings_from_carrots + earnings_from_spinach

-- Define the earnings from cauliflower
def earnings_from_cauliflower : ℕ := total_earnings - earnings_from_others

-- Theorem to prove the earnings from cauliflower
theorem earnings_from_cauliflower_correct : earnings_from_cauliflower = 136 :=
by
  sorry

end earnings_from_cauliflower_correct_l1836_183673


namespace ethan_coconut_oil_per_candle_l1836_183637

noncomputable def ounces_of_coconut_oil_per_candle (candles: ℕ) (total_weight: ℝ) (beeswax_per_candle: ℝ) : ℝ :=
(total_weight - candles * beeswax_per_candle) / candles

theorem ethan_coconut_oil_per_candle :
  ounces_of_coconut_oil_per_candle 7 63 8 = 1 :=
by
  sorry

end ethan_coconut_oil_per_candle_l1836_183637


namespace sum_of_first_15_terms_l1836_183617

theorem sum_of_first_15_terms (a d : ℝ) 
  (h : (a + 3 * d) + (a + 11 * d) = 24) : 
  (15 / 2) * (2 * a + 14 * d) = 180 :=
by
  sorry

end sum_of_first_15_terms_l1836_183617


namespace vegan_menu_fraction_suitable_l1836_183616

theorem vegan_menu_fraction_suitable (vegan_dishes total_dishes vegan_dishes_with_gluten_or_dairy : ℕ)
  (h1 : vegan_dishes = 9)
  (h2 : vegan_dishes = 3 * total_dishes / 10)
  (h3 : vegan_dishes_with_gluten_or_dairy = 7) :
  (vegan_dishes - vegan_dishes_with_gluten_or_dairy) / total_dishes = 1 / 15 := by
  sorry

end vegan_menu_fraction_suitable_l1836_183616


namespace sin_cos_term_side_l1836_183688

theorem sin_cos_term_side (a : ℝ) (ha : a ≠ 0) :
  ∃ k : ℝ, (k = 2 * (if a > 0 then -3/5 else 3/5) + (if a > 0 then 4/5 else -4/5)) ∧ (k = 2/5 ∨ k = -2/5) := by
  sorry

end sin_cos_term_side_l1836_183688


namespace polynomial_divisibility_l1836_183625

theorem polynomial_divisibility (
  p q r s : ℝ
) :
  (x^5 + 5 * x^4 + 10 * p * x^3 + 10 * q * x^2 + 5 * r * x + s) % (x^4 + 4 * x^3 + 6 * x^2 + 4 * x + 1) = 0 ->
  (p + q + r) * s = -2 :=
by {
  sorry
}

end polynomial_divisibility_l1836_183625


namespace division_identity_l1836_183698

theorem division_identity :
  (0.0048 * 3.5) / (0.05 * 0.1 * 0.004) = 840 :=
by
  -- TODO: Provide the proof here
  sorry

end division_identity_l1836_183698


namespace determine_xyz_l1836_183603

variables {x y z : ℝ}

theorem determine_xyz (h : (x - y - 3)^2 + (y - z)^2 + (x - z)^2 = 3) : 
  x = z + 1 ∧ y = z - 1 := 
sorry

end determine_xyz_l1836_183603


namespace tangent_line_at_point_l1836_183630

theorem tangent_line_at_point (x y : ℝ) (h : y = x / (x - 2)) (hx : x = 1) (hy : y = -1) : y = -2 * x + 1 :=
sorry

end tangent_line_at_point_l1836_183630


namespace playground_area_l1836_183667

theorem playground_area :
  ∃ (l w : ℝ), 2 * l + 2 * w = 84 ∧ l = 3 * w ∧ l * w = 330.75 :=
by
  sorry

end playground_area_l1836_183667


namespace shaded_area_ratio_l1836_183679

-- Definitions based on conditions
def large_square_area : ℕ := 16
def shaded_components : ℕ := 4
def component_fraction : ℚ := 1 / 2
def shaded_square_area : ℚ := shaded_components * component_fraction
def large_square_area_q : ℚ := large_square_area

-- Goal statement
theorem shaded_area_ratio : (shaded_square_area / large_square_area_q) = (1 / 8) :=
by sorry

end shaded_area_ratio_l1836_183679


namespace problem_statement_l1836_183699

-- Define A as the number of four-digit odd numbers
def A : ℕ := 4500

-- Define B as the number of four-digit multiples of 3
def B : ℕ := 3000

-- The main theorem stating the sum A + B equals 7500
theorem problem_statement : A + B = 7500 := by
  -- The exact proof is omitted using sorry
  sorry

end problem_statement_l1836_183699


namespace geom_seq_a4_l1836_183642

theorem geom_seq_a4 (a1 a2 a3 a4 r : ℝ)
  (h1 : a1 + a2 + a3 = 7)
  (h2 : a1 * a2 * a3 = 8)
  (h3 : a1 > 0)
  (h4 : r > 1)
  (h5 : a2 = a1 * r)
  (h6 : a3 = a1 * r^2)
  (h7 : a4 = a1 * r^3) : 
  a4 = 8 :=
sorry

end geom_seq_a4_l1836_183642


namespace initial_wage_illiterate_l1836_183694

variable (I : ℕ) -- initial daily average wage of illiterate employees

theorem initial_wage_illiterate (h1 : 20 * I - 20 * 10 = 300) : I = 25 :=
by
  simp at h1
  sorry

end initial_wage_illiterate_l1836_183694


namespace range_a_l1836_183628

open Set Real

-- Define the predicate p: real number x satisfies x^2 - 4ax + 3a^2 < 0, where a < 0
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0

-- Define the predicate q: real number x satisfies x^2 - x - 6 ≤ 0, or x^2 + 2x - 8 > 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0

-- Define the complement sets
def not_p_set (a : ℝ) : Set ℝ := {x | ¬p x a}
def not_q_set : Set ℝ := {x | ¬q x}

-- Define p as necessary but not sufficient condition for q
def necessary_but_not_sufficient (a : ℝ) : Prop := 
  (not_q_set ⊆ not_p_set a) ∧ ¬(not_p_set a ⊆ not_q_set)

-- The main theorem to prove
theorem range_a : {a : ℝ | necessary_but_not_sufficient a} = {a : ℝ | -4 ≤ a ∧ a < 0 ∨ a ≤ -4} :=
by
  sorry

end range_a_l1836_183628


namespace circle_line_bisect_l1836_183631

theorem circle_line_bisect (a : ℝ) :
    (∀ x y : ℝ, (x - (-1))^2 + (y - 2)^2 = 5 → 3 * x + y + a = 0) → a = 1 :=
sorry

end circle_line_bisect_l1836_183631


namespace sarah_initial_money_l1836_183607

def initial_money 
  (cost_toy_car : ℕ)
  (cost_scarf : ℕ)
  (cost_beanie : ℕ)
  (remaining_money : ℕ)
  (number_of_toy_cars : ℕ) : ℕ :=
  remaining_money + cost_beanie + cost_scarf + number_of_toy_cars * cost_toy_car

theorem sarah_initial_money : 
  (initial_money 11 10 14 7 2) = 53 :=
by
  rfl 

end sarah_initial_money_l1836_183607


namespace pyramid_surface_area_l1836_183693

-- Definitions for the conditions
structure Rectangle where
  length : ℝ
  width : ℝ

structure Pyramid where
  base : Rectangle
  height : ℝ

-- Create instances representing the given conditions
noncomputable def givenRectangle : Rectangle := {
  length := 8,
  width := 6
}

noncomputable def givenPyramid : Pyramid := {
  base := givenRectangle,
  height := 15
}

-- Statement to prove the surface area of the pyramid
theorem pyramid_surface_area
  (rect: Rectangle)
  (length := rect.length)
  (width := rect.width)
  (height: ℝ)
  (hy1: length = 8)
  (hy2: width = 6)
  (hy3: height = 15) :
  let base_area := length * width
  let slant_height := Real.sqrt (height^2 + (length / 2)^2)
  let lateral_area := 2 * ((length * slant_height) / 2 + (width * slant_height) / 2)
  let total_surface_area := base_area + lateral_area 
  total_surface_area = 48 + 7 * Real.sqrt 241 := 
  sorry

end pyramid_surface_area_l1836_183693


namespace sequence_eq_l1836_183613

-- Define the sequence and the conditions
def is_sequence (a : ℕ → ℕ) :=
  (∀ i, a i > 0) ∧ (∀ i j, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j)

-- The theorem we want to prove: for all i, a_i = i
theorem sequence_eq (a : ℕ → ℕ) (h : is_sequence a) : ∀ i, a i = i :=
by
  sorry

end sequence_eq_l1836_183613


namespace find_m_l1836_183686

theorem find_m
  (θ : Real)
  (m : Real)
  (h_sin_cos_roots : ∀ x : Real, 4 * x^2 + 2 * m * x + m = 0 → x = Real.sin θ ∨ x = Real.cos θ)
  (h_real_roots : ∃ x : Real, 4 * x^2 + 2 * m * x + m = 0) :
  m = 1 - Real.sqrt 5 :=
sorry

end find_m_l1836_183686


namespace number_of_roots_l1836_183638

-- Definitions for the conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_monotonic_in_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → y ≤ a → f x ≤ f y

-- Main theorem to prove
theorem number_of_roots (f : ℝ → ℝ) (a : ℝ) (h1 : 0 < a) 
  (h2 : is_even_function f) (h3 : is_monotonic_in_interval f a) 
  (h4 : f 0 * f a < 0) : ∃ x0 > 0, f x0 = 0 ∧ ∃ x1 < 0, f x1 = 0 :=
sorry

end number_of_roots_l1836_183638


namespace find_sinα_and_tanα_l1836_183619

open Real 

noncomputable def vectors (α : ℝ) := (Real.cos α, 1)

noncomputable def vectors_perpendicular (α : ℝ) := (Real.sin α, -2)

theorem find_sinα_and_tanα (α: ℝ) (hα: π < α ∧ α < 3 * π / 2)
  (h_perp: vectors_perpendicular α = (Real.sin α, -2) ∧ vectors α = (Real.cos α, 1) ∧ (vectors α).1 * (vectors_perpendicular α).1 + (vectors α).2 * (vectors_perpendicular α).2 = 0):
  (Real.sin α = - (2 * Real.sqrt 5) / 5) ∧ 
  (Real.tan (α + π / 4) = -3) := 
sorry 

end find_sinα_and_tanα_l1836_183619


namespace total_votes_l1836_183676

theorem total_votes (V : ℝ) 
  (h1 : 0.5 / 100 * V = 0.005 * V) 
  (h2 : 50.5 / 100 * V = 0.505 * V) 
  (h3 : 0.505 * V - 0.005 * V = 3000) : 
  V = 6000 := 
by
  sorry

end total_votes_l1836_183676


namespace shoe_price_on_monday_l1836_183636

theorem shoe_price_on_monday
  (price_on_thursday : ℝ)
  (price_increase : ℝ)
  (discount : ℝ)
  (price_on_friday : ℝ := price_on_thursday * (1 + price_increase))
  (price_on_monday : ℝ := price_on_friday * (1 - discount))
  (price_on_thursday_eq : price_on_thursday = 50)
  (price_increase_eq : price_increase = 0.2)
  (discount_eq : discount = 0.15) :
  price_on_monday = 51 :=
by
  sorry

end shoe_price_on_monday_l1836_183636


namespace max_value_of_f_l1836_183639

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

theorem max_value_of_f :
  ∃ x_max : ℝ, (∀ x : ℝ, f x ≤ f x_max) ∧ f 1 = 1 / Real.exp 1 := 
by {
  sorry
}

end max_value_of_f_l1836_183639


namespace baseball_games_in_season_l1836_183659

def games_per_month : ℕ := 7
def months_in_season : ℕ := 2
def total_games_in_season : ℕ := games_per_month * months_in_season

theorem baseball_games_in_season : total_games_in_season = 14 := by
  sorry

end baseball_games_in_season_l1836_183659


namespace neg_P_l1836_183692

/-
Proposition: There exists a natural number n such that 2^n > 1000.
-/
def P : Prop := ∃ n : ℕ, 2^n > 1000

/-
Theorem: The negation of the above proposition P is:
For all natural numbers n, 2^n ≤ 1000.
-/
theorem neg_P : ¬ P ↔ ∀ n : ℕ, 2^n ≤ 1000 :=
by
  sorry

end neg_P_l1836_183692


namespace schedule_arrangement_count_l1836_183629

-- Given subjects
inductive Subject
| Chinese
| Mathematics
| Politics
| English
| PhysicalEducation
| Art

open Subject

-- Define a function to get the total number of different arrangements
def arrangement_count : Nat := 192

-- The proof statement (problem restated in Lean 4)
theorem schedule_arrangement_count :
  arrangement_count = 192 :=
by
  sorry

end schedule_arrangement_count_l1836_183629


namespace range_f_contained_in_0_1_l1836_183670

theorem range_f_contained_in_0_1 (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > y → (f x)^2 ≤ f y) : 
  ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 := 
by {
  sorry
}

end range_f_contained_in_0_1_l1836_183670


namespace hyperbola_center_l1836_183650

theorem hyperbola_center :
  ∀ (x y : ℝ), 
  (4 * x + 8)^2 / 36 - (3 * y - 6)^2 / 25 = 1 → (x, y) = (-2, 2) :=
by
  intros x y h
  sorry

end hyperbola_center_l1836_183650


namespace possible_values_for_abc_l1836_183612

theorem possible_values_for_abc (a b c : ℝ)
  (h : ∀ x y z : ℤ, (a * x + b * y + c * z) ∣ (b * x + c * y + a * z)) :
  (a, b, c) = (1, 0, 0) ∨ (a, b, c) = (0, 1, 0) ∨ (a, b, c) = (0, 0, 1) ∨
  (a, b, c) = (-1, 0, 0) ∨ (a, b, c) = (0, -1, 0) ∨ (a, b, c) = (0, 0, -1) :=
sorry

end possible_values_for_abc_l1836_183612


namespace simplify_and_evaluate_l1836_183622

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) :
  (1 / (x - 1) - 2 / (x ^ 2 - 1)) = -1 := by
  sorry

end simplify_and_evaluate_l1836_183622


namespace platform_length_l1836_183653

theorem platform_length
  (train_length : ℤ)
  (speed_kmph : ℤ)
  (time_sec : ℤ)
  (speed_mps : speed_kmph * 1000 / 3600 = 20)
  (distance_eq : (train_length + 220) = (20 * time_sec))
  (train_length_val : train_length = 180)
  (time_sec_val : time_sec = 20) :
  220 = 220 := by
  sorry

end platform_length_l1836_183653


namespace necessary_and_sufficient_condition_l1836_183605

def line1 (a : ℝ) (x y : ℝ) := 2 * x - a * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) := (a - 1) * x - y + a = 0
def parallel (a : ℝ) : Prop := ∀ x y : ℝ, line1 a x y = line2 a x y

theorem necessary_and_sufficient_condition (a : ℝ) : 
  (a = 2 ↔ parallel a) :=
sorry

end necessary_and_sufficient_condition_l1836_183605


namespace seq_a_2012_value_l1836_183675

theorem seq_a_2012_value :
  ∀ (a : ℕ → ℕ),
  (a 1 = 0) →
  (∀ n : ℕ, a (n + 1) = a n + 2 * n) →
  a 2012 = 2011 * 2012 :=
by
  intros a h₁ h₂
  sorry

end seq_a_2012_value_l1836_183675


namespace battery_change_month_battery_change_in_november_l1836_183691

theorem battery_change_month :
  (119 % 12) = 11 := by
  sorry

theorem battery_change_in_november (n : Nat) (h1 : n = 18) :
  let month := ((n - 1) * 7) % 12
  month = 11 := by
  sorry

end battery_change_month_battery_change_in_november_l1836_183691


namespace number_of_triangles_in_decagon_l1836_183687

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l1836_183687


namespace problem_statement_l1836_183696

theorem problem_statement 
  (p q r x y z a b c : ℝ)
  (h1 : p / x = q / y ∧ q / y = r / z)
  (h2 : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1) :
  p^2 / a^2 + q^2 / b^2 + r^2 / c^2 = (p^2 + q^2 + r^2) / (x^2 + y^2 + z^2) :=
sorry  -- Proof omitted

end problem_statement_l1836_183696


namespace right_triangle_sides_l1836_183648

theorem right_triangle_sides (r R : ℝ) (a b c : ℝ) 
    (r_eq : r = 8)
    (R_eq : R = 41)
    (right_angle : a^2 + b^2 = c^2)
    (inradius : 2*r = a + b - c)
    (circumradius : 2*R = c) :
    (a = 18 ∧ b = 80 ∧ c = 82) ∨ (a = 80 ∧ b = 18 ∧ c = 82) :=
by
  sorry

end right_triangle_sides_l1836_183648


namespace find_d_l1836_183618

def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3
def h (x : ℝ) (c : ℝ) (d : ℝ) : Prop := f (g x c) c = 15 * x + d

theorem find_d (c d : ℝ) (h : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry

end find_d_l1836_183618


namespace max_value_of_a_l1836_183685

theorem max_value_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 3 > 0 → x < a) ∧
  (∃ x : ℝ, x < a ∧ ¬(x^2 - 2*x - 3 > 0)) →
  a = -1 :=
by
  sorry

end max_value_of_a_l1836_183685


namespace percentage_peanut_clusters_is_64_l1836_183655

def total_chocolates := 50
def caramels := 3
def nougats := 2 * caramels
def truffles := caramels + 6
def other_chocolates := caramels + nougats + truffles
def peanut_clusters := total_chocolates - other_chocolates
def percentage_peanut_clusters := (peanut_clusters * 100) / total_chocolates

theorem percentage_peanut_clusters_is_64 :
  percentage_peanut_clusters = 64 := by
  sorry

end percentage_peanut_clusters_is_64_l1836_183655


namespace library_hospital_community_center_bells_ring_together_l1836_183608

theorem library_hospital_community_center_bells_ring_together :
  ∀ (library hospital community : ℕ), 
    (library = 18) → (hospital = 24) → (community = 30) → 
    (∀ t, (t = 0) ∨ (∃ n₁ n₂ n₃ : ℕ, 
      t = n₁ * library ∧ t = n₂ * hospital ∧ t = n₃ * community)) → 
    true :=
by
  intros
  sorry

end library_hospital_community_center_bells_ring_together_l1836_183608


namespace annual_rent_per_square_foot_l1836_183660

theorem annual_rent_per_square_foot
  (length width : ℕ) (monthly_rent : ℕ) (h_length : length = 10)
  (h_width : width = 8) (h_monthly_rent : monthly_rent = 2400) :
  (monthly_rent * 12) / (length * width) = 360 := 
by 
  -- We assume the theorem is true.
  sorry

end annual_rent_per_square_foot_l1836_183660


namespace quadratic_roots_eccentricities_l1836_183646

theorem quadratic_roots_eccentricities :
  (∃ x y : ℝ, 3 * x^2 - 4 * x + 1 = 0 ∧ 3 * y^2 - 4 * y + 1 = 0 ∧ 
              (0 ≤ x ∧ x < 1) ∧ y = 1) :=
by
  -- Proof would go here
  sorry

end quadratic_roots_eccentricities_l1836_183646


namespace adah_practiced_total_hours_l1836_183623

theorem adah_practiced_total_hours :
  let minutes_per_day := 86
  let days_practiced := 2
  let minutes_other_days := 278
  let total_minutes := (minutes_per_day * days_practiced) + minutes_other_days
  let total_hours := total_minutes / 60
  total_hours = 7.5 :=
by
  sorry

end adah_practiced_total_hours_l1836_183623


namespace ryan_hours_english_is_6_l1836_183614

def hours_chinese : Nat := 2

def hours_english (C : Nat) : Nat := C + 4

theorem ryan_hours_english_is_6 (C : Nat) (hC : C = hours_chinese) : hours_english C = 6 :=
by
  sorry

end ryan_hours_english_is_6_l1836_183614


namespace carrots_total_l1836_183665

theorem carrots_total (sandy_carrots : Nat) (sam_carrots : Nat) (h1 : sandy_carrots = 6) (h2 : sam_carrots = 3) :
  sandy_carrots + sam_carrots = 9 :=
by
  sorry

end carrots_total_l1836_183665


namespace fred_balloons_l1836_183604

theorem fred_balloons (T S D F : ℕ) (hT : T = 72) (hS : S = 46) (hD : D = 16) (hTotal : T = F + S + D) : F = 10 := 
by
  sorry

end fred_balloons_l1836_183604


namespace paving_rate_correct_l1836_183647

-- Define the constants
def length (L : ℝ) := L = 5.5
def width (W : ℝ) := W = 4
def cost (C : ℝ) := C = 15400
def area (A : ℝ) := A = 22

-- Given the definitions above, prove the rate per sq. meter
theorem paving_rate_correct (L W C A : ℝ) (hL : length L) (hW : width W) (hC : cost C) (hA : area A) :
  C / A = 700 := 
sorry

end paving_rate_correct_l1836_183647


namespace smallest_possible_w_l1836_183666

theorem smallest_possible_w 
  (h1 : 936 = 2^3 * 3 * 13)
  (h2 : 2^5 = 32)
  (h3 : 3^3 = 27)
  (h4 : 14^2 = 196) :
  ∃ w : ℕ, (w > 0) ∧ (936 * w) % 32 = 0 ∧ (936 * w) % 27 = 0 ∧ (936 * w) % 196 = 0 ∧ w = 1764 :=
sorry

end smallest_possible_w_l1836_183666


namespace sum_of_squares_eq_three_l1836_183635

theorem sum_of_squares_eq_three
  (a b s : ℝ)
  (h₀ : a ≠ b)
  (h₁ : a * s^2 + b * s + b = 0)
  (h₂ : a * (1 / s)^2 + a * (1 / s) + b = 0)
  (h₃ : s * (1 / s) = 1) :
  s^2 + (1 / s)^2 = 3 := 
sorry

end sum_of_squares_eq_three_l1836_183635


namespace distinct_license_plates_l1836_183624

noncomputable def license_plates : ℕ :=
  let digits_possibilities := 10^5
  let letters_possibilities := 26^3
  let positions := 6
  positions * digits_possibilities * letters_possibilities

theorem distinct_license_plates : 
  license_plates = 105456000 := by
  sorry

end distinct_license_plates_l1836_183624


namespace nh4cl_formed_l1836_183663

theorem nh4cl_formed :
  (∀ (nh3 hcl nh4cl : ℝ), nh3 = 1 ∧ hcl = 1 → nh3 + hcl = nh4cl → nh4cl = 1) :=
by
  intros nh3 hcl nh4cl
  sorry

end nh4cl_formed_l1836_183663


namespace triangle_orthocenter_example_l1836_183626

open Real EuclideanGeometry

def point_3d := (ℝ × ℝ × ℝ)

def orthocenter (A B C : point_3d) : point_3d := sorry

theorem triangle_orthocenter_example :
  orthocenter (2, 4, 6) (6, 5, 3) (4, 6, 7) = (4/5, 38/5, 59/5) := sorry

end triangle_orthocenter_example_l1836_183626


namespace solve_custom_eq_l1836_183652

namespace CustomProof

def custom_mul (a b : ℕ) : ℕ := a * b + a + b

theorem solve_custom_eq (x : ℕ) (h : custom_mul 3 x = 31) : x = 7 := 
by
  sorry

end CustomProof

end solve_custom_eq_l1836_183652


namespace range_of_k_l1836_183680

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x > k → (3 / (x + 1) < 1)) ↔ k ≥ 2 := sorry

end range_of_k_l1836_183680


namespace find_constant_l1836_183632

-- Define the conditions
def is_axles (x : ℕ) : Prop := x = 5
def toll_for_truck (t : ℝ) : Prop := t = 4

-- Define the formula for the toll
def toll_formula (t : ℝ) (constant : ℝ) (x : ℕ) : Prop :=
  t = 2.50 + constant * (x - 2)

-- Proof problem statement
theorem find_constant : ∃ (constant : ℝ), 
  ∀ x : ℕ, is_axles x → toll_for_truck 4 →
  toll_formula 4 constant x → constant = 0.50 :=
sorry

end find_constant_l1836_183632


namespace perimeter_of_shaded_region_correct_l1836_183620

noncomputable def perimeter_of_shaded_region : ℝ :=
  let r := 7
  let perimeter := 2 * r + (3 / 4) * (2 * Real.pi * r)
  perimeter

theorem perimeter_of_shaded_region_correct :
  perimeter_of_shaded_region = 14 + 10.5 * Real.pi :=
by
  sorry

end perimeter_of_shaded_region_correct_l1836_183620


namespace shorter_side_of_quilt_l1836_183602

theorem shorter_side_of_quilt :
  ∀ (x : ℕ), (∃ y : ℕ, 24 * y = 144) -> x = 6 :=
by
  intros x h
  sorry

end shorter_side_of_quilt_l1836_183602
