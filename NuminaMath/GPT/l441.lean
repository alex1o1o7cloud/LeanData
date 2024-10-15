import Mathlib

namespace NUMINAMATH_GPT_jana_height_l441_44104

theorem jana_height (jess_height : ℕ) (kelly_height : ℕ) (jana_height : ℕ) 
  (h1 : kelly_height = jess_height - 3) 
  (h2 : jana_height = kelly_height + 5) 
  (h3 : jess_height = 72) : 
  jana_height = 74 := 
by
  sorry

end NUMINAMATH_GPT_jana_height_l441_44104


namespace NUMINAMATH_GPT_ending_number_of_range_divisible_by_five_l441_44136

theorem ending_number_of_range_divisible_by_five
  (first_number : ℕ)
  (number_of_terms : ℕ)
  (h_first : first_number = 15)
  (h_terms : number_of_terms = 10)
  : ∃ ending_number : ℕ, ending_number = first_number + 5 * (number_of_terms - 1) := 
by
  sorry

end NUMINAMATH_GPT_ending_number_of_range_divisible_by_five_l441_44136


namespace NUMINAMATH_GPT_no_snow_five_days_l441_44122

-- Define the probability of snow on a given day
def prob_snow : ℚ := 2 / 3

-- Define the probability of no snow on a given day
def prob_not_snow : ℚ := 1 - prob_snow

-- Define the probability of no snow for five consecutive days
def prob_no_snow_five_days : ℚ := (prob_not_snow)^5

-- Statement to prove the probability of no snow for the next five days is 1/243
theorem no_snow_five_days : prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end NUMINAMATH_GPT_no_snow_five_days_l441_44122


namespace NUMINAMATH_GPT_factorize_expression_l441_44139

variable (m n : ℤ)

theorem factorize_expression : 2 * m * n^2 - 12 * m * n + 18 * m = 2 * m * (n - 3)^2 := by
  sorry

end NUMINAMATH_GPT_factorize_expression_l441_44139


namespace NUMINAMATH_GPT_evaluate_composite_l441_44187

def f (x : ℕ) : ℕ := 2 * x + 5
def g (x : ℕ) : ℕ := 3 * x + 4

theorem evaluate_composite : f (g (f 3)) = 79 := by
  sorry

end NUMINAMATH_GPT_evaluate_composite_l441_44187


namespace NUMINAMATH_GPT_Sam_needs_16_more_hours_l441_44146

noncomputable def Sam_hourly_rate : ℝ :=
  460 / 23

noncomputable def Sam_earnings_Sep_to_Feb : ℝ :=
  8 * Sam_hourly_rate

noncomputable def Sam_total_earnings : ℝ :=
  460 + Sam_earnings_Sep_to_Feb

noncomputable def Sam_remaining_money : ℝ :=
  Sam_total_earnings - 340

noncomputable def Sam_needed_money : ℝ :=
  600 - Sam_remaining_money

noncomputable def Sam_additional_hours_needed : ℝ :=
  Sam_needed_money / Sam_hourly_rate

theorem Sam_needs_16_more_hours : Sam_additional_hours_needed = 16 :=
by 
  sorry

end NUMINAMATH_GPT_Sam_needs_16_more_hours_l441_44146


namespace NUMINAMATH_GPT_find_initial_children_l441_44191

variables (x y : ℕ)

-- Defining the conditions 
def initial_children_on_bus (x : ℕ) : Prop :=
  ∃ y : ℕ, x - 68 + y = 12 ∧ 68 - y = 24 + y

-- Theorem statement
theorem find_initial_children : initial_children_on_bus x → x = 58 :=
by
  -- Skipping the proof for now
  sorry

end NUMINAMATH_GPT_find_initial_children_l441_44191


namespace NUMINAMATH_GPT_factorize_binomial_square_l441_44114

theorem factorize_binomial_square (x y : ℝ) : x^2 + 2*x*y + y^2 = (x + y)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_binomial_square_l441_44114


namespace NUMINAMATH_GPT_seashells_total_l441_44156

theorem seashells_total (s m : Nat) (hs : s = 18) (hm : m = 47) : s + m = 65 := 
by
  -- We are just specifying the theorem statement here
  sorry

end NUMINAMATH_GPT_seashells_total_l441_44156


namespace NUMINAMATH_GPT_greatest_possible_sum_l441_44134

theorem greatest_possible_sum (x y : ℤ) (h : x^2 + y^2 = 100) : x + y ≤ 14 :=
sorry

end NUMINAMATH_GPT_greatest_possible_sum_l441_44134


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_not_necessary_l441_44144

theorem sufficient_but_not_necessary (a b : ℝ) (h : b > a ∧ a > 0) : (a * (b + 1) > a^2) :=
sorry

theorem not_necessary (a b : ℝ) : (a * (b + 1) > a^2 → b > a ∧ a > 0) → false :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_not_necessary_l441_44144


namespace NUMINAMATH_GPT_base8_difference_divisible_by_7_l441_44170

theorem base8_difference_divisible_by_7 (A B : ℕ) (h₁ : A < 8) (h₂ : B < 8) (h₃ : A ≠ B) : 
  ∃ k : ℕ, k * 7 = (if 8 * A + B > 8 * B + A then 8 * A + B - (8 * B + A) else 8 * B + A - (8 * A + B)) :=
by
  sorry

end NUMINAMATH_GPT_base8_difference_divisible_by_7_l441_44170


namespace NUMINAMATH_GPT_evan_ivan_kara_total_weight_eq_432_l441_44194

variable (weight_evan : ℕ) (weight_ivan : ℕ) (weight_kara_cat : ℕ)

-- Conditions
def evans_dog_weight : Prop := weight_evan = 63
def ivans_dog_weight : Prop := weight_evan = 7 * weight_ivan
def karas_cat_weight : Prop := weight_kara_cat = 5 * (weight_evan + weight_ivan)

-- Mathematical equivalence
def total_weight : Prop := weight_evan + weight_ivan + weight_kara_cat = 432

theorem evan_ivan_kara_total_weight_eq_432 :
  evans_dog_weight weight_evan →
  ivans_dog_weight weight_evan weight_ivan →
  karas_cat_weight weight_evan weight_ivan weight_kara_cat →
  total_weight weight_evan weight_ivan weight_kara_cat :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_evan_ivan_kara_total_weight_eq_432_l441_44194


namespace NUMINAMATH_GPT_range_of_x_l441_44174

theorem range_of_x (x p : ℝ) (hp : 0 ≤ p ∧ p ≤ 4) :
  (x^2 + p*x > 4*x + p - 3) ↔ (x > 3 ∨ x < -1) :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_x_l441_44174


namespace NUMINAMATH_GPT_ratio_of_B_to_C_l441_44142

variables (A B C : ℕ)

-- Conditions from the problem
axiom h1 : A = B + 2
axiom h2 : A + B + C = 12
axiom h3 : B = 4

-- Goal: Prove that the ratio of B's age to C's age is 2
theorem ratio_of_B_to_C : B / C = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_of_B_to_C_l441_44142


namespace NUMINAMATH_GPT_second_divisor_l441_44116

theorem second_divisor (x : ℕ) : (282 % 31 = 3) ∧ (282 % x = 3) → x = 9 :=
by
  sorry

end NUMINAMATH_GPT_second_divisor_l441_44116


namespace NUMINAMATH_GPT_min_value_3x_4y_l441_44101

noncomputable def minValue (x y : ℝ) : ℝ := 3 * x + 4 * y

theorem min_value_3x_4y 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : x + 3 * y = 5 * x * y) : 
  minValue x y ≥ 5 :=
sorry

end NUMINAMATH_GPT_min_value_3x_4y_l441_44101


namespace NUMINAMATH_GPT_hexagon_inequality_l441_44138

variables {Point : Type} [MetricSpace Point]

-- Definitions of points and distances
variables (A B C D E F G H : Point) 
variables (dist : Point → Point → ℝ)
variables (angle : Point → Point → Point → ℝ)

-- Conditions
variables (hABCDEF : ConvexHexagon A B C D E F)
variables (hAB_BC_CD : dist A B = dist B C ∧ dist B C = dist C D)
variables (hDE_EF_FA : dist D E = dist E F ∧ dist E F = dist F A)
variables (hBCD_60 : angle B C D = 60)
variables (hEFA_60 : angle E F A = 60)
variables (hAGB_120 : angle A G B = 120)
variables (hDHE_120 : angle D H E = 120)

-- Objective statement
theorem hexagon_inequality : 
  dist A G + dist G B + dist G H + dist D H + dist H E ≥ dist C F :=
sorry

end NUMINAMATH_GPT_hexagon_inequality_l441_44138


namespace NUMINAMATH_GPT_minimum_revenue_maximum_marginal_cost_minimum_profit_l441_44140

noncomputable def R (x : ℕ) : ℝ := x^2 + 16 / x^2 + 40
noncomputable def C (x : ℕ) : ℝ := 10 * x + 40 / x
noncomputable def MC (x : ℕ) : ℝ := C (x + 1) - C x
noncomputable def z (x : ℕ) : ℝ := R x - C x

theorem minimum_revenue :
  ∀ x : ℕ, 1 ≤ x → x ≤ 10 → R x ≥ 72 :=
sorry

theorem maximum_marginal_cost :
  ∀ x : ℕ, 1 ≤ x → x ≤ 9 → MC x ≤ 86 / 9 :=
sorry

theorem minimum_profit :
  ∀ x : ℕ, 1 ≤ x → x ≤ 10 → (x = 1 ∨ x = 4) → z x ≥ 7 :=
sorry

end NUMINAMATH_GPT_minimum_revenue_maximum_marginal_cost_minimum_profit_l441_44140


namespace NUMINAMATH_GPT_find_solutions_l441_44128

def is_solution (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ Int.gcd (Int.gcd a b) c = 1 ∧
  (a + b + c) ∣ (a^12 + b^12 + c^12) ∧
  (a + b + c) ∣ (a^23 + b^23 + c^23) ∧
  (a + b + c) ∣ (a^11004 + b^11004 + c^11004)

theorem find_solutions :
  (is_solution 1 1 1) ∧ (is_solution 1 1 4) ∧ 
  (∀ a b c : ℕ, is_solution a b c → 
    (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 1 ∧ c = 4)) := 
sorry

end NUMINAMATH_GPT_find_solutions_l441_44128


namespace NUMINAMATH_GPT_hagrid_divisible_by_three_l441_44141

def distinct_digits (n : ℕ) : Prop :=
  n < 10

theorem hagrid_divisible_by_three (H A G R I D : ℕ) (H_dist A_dist G_dist R_dist I_dist D_dist : distinct_digits H ∧ distinct_digits A ∧ distinct_digits G ∧ distinct_digits R ∧ distinct_digits I ∧ distinct_digits D)
  (distinct_letters: H ≠ A ∧ H ≠ G ∧ H ≠ R ∧ H ≠ I ∧ H ≠ D ∧ A ≠ G ∧ A ≠ R ∧ A ≠ I ∧ A ≠ D ∧ G ≠ R ∧ G ≠ I ∧ G ≠ D ∧ R ≠ I ∧ R ≠ D ∧ I ≠ D) :
  3 ∣ (H * 100000 + A * 10000 + G * 1000 + R * 100 + I * 10 + D) * H * A * G * R * I * D :=
sorry

end NUMINAMATH_GPT_hagrid_divisible_by_three_l441_44141


namespace NUMINAMATH_GPT_total_votes_cast_l441_44178

theorem total_votes_cast (V : ℕ) (C R : ℕ) 
  (hC : C = 30 * V / 100) 
  (hR1 : R = C + 4000) 
  (hR2 : R = 70 * V / 100) : 
  V = 10000 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_cast_l441_44178


namespace NUMINAMATH_GPT_square_side_length_l441_44151

-- Define the conditions
def rectangle_width : ℝ := 4
def rectangle_length : ℝ := 4
def area_rectangle : ℝ := rectangle_width * rectangle_length
def area_square : ℝ := area_rectangle

-- Prove the side length of the square
theorem square_side_length :
  ∃ s : ℝ, s * s = area_square ∧ s = 4 := 
  by {
    -- Here you'd write the proof step, but it's omitted as per instructions
    sorry
  }

end NUMINAMATH_GPT_square_side_length_l441_44151


namespace NUMINAMATH_GPT_interest_rate_B_lent_to_C_l441_44126

noncomputable def principal : ℝ := 1500
noncomputable def rate_A : ℝ := 10
noncomputable def time : ℝ := 3
noncomputable def gain_B : ℝ := 67.5
noncomputable def interest_paid_by_B_to_A : ℝ := principal * rate_A * time / 100
noncomputable def interest_received_by_B_from_C : ℝ := interest_paid_by_B_to_A + gain_B
noncomputable def expected_rate : ℝ := 11.5

theorem interest_rate_B_lent_to_C :
  interest_received_by_B_from_C = principal * (expected_rate) * time / 100 := 
by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_interest_rate_B_lent_to_C_l441_44126


namespace NUMINAMATH_GPT_polynomial_range_open_interval_l441_44145

theorem polynomial_range_open_interval :
  ∀ (k : ℝ), k > 0 → ∃ (x y : ℝ), (1 - x * y)^2 + x^2 = k :=
by
  sorry

end NUMINAMATH_GPT_polynomial_range_open_interval_l441_44145


namespace NUMINAMATH_GPT_megan_initial_acorns_l441_44167

def initial_acorns (given_away left: ℕ) : ℕ := 
  given_away + left

theorem megan_initial_acorns :
  initial_acorns 7 9 = 16 := 
by 
  unfold initial_acorns
  rfl

end NUMINAMATH_GPT_megan_initial_acorns_l441_44167


namespace NUMINAMATH_GPT_partnership_investment_l441_44149

theorem partnership_investment
  (a_investment : ℕ := 30000)
  (b_investment : ℕ)
  (c_investment : ℕ := 50000)
  (c_profit_share : ℕ := 36000)
  (total_profit : ℕ := 90000)
  (total_investment := a_investment + b_investment + c_investment)
  (c_defined_share : ℚ := 2/5)
  (profit_proportionality : (c_profit_share : ℚ) / total_profit = (c_investment : ℚ) / total_investment) :
  b_investment = 45000 :=
by
  sorry

end NUMINAMATH_GPT_partnership_investment_l441_44149


namespace NUMINAMATH_GPT_cos_A_sin_B_eq_l441_44120

theorem cos_A_sin_B_eq (A B : ℝ) (hA1 : 0 < A) (hA2 : A < π / 2) (hB1 : 0 < B) (hB2 : B < π / 2)
    (h : (4 + (Real.tan A)^2) * (5 + (Real.tan B)^2) = Real.sqrt 320 * Real.tan A * Real.tan B) :
    Real.cos A * Real.sin B = 1 / Real.sqrt 6 := sorry

end NUMINAMATH_GPT_cos_A_sin_B_eq_l441_44120


namespace NUMINAMATH_GPT_percent_of_whole_l441_44168

theorem percent_of_whole (Part Whole : ℝ) (Percent : ℝ) (hPart : Part = 160) (hWhole : Whole = 50) :
  Percent = (Part / Whole) * 100 → Percent = 320 :=
by
  rw [hPart, hWhole]
  sorry

end NUMINAMATH_GPT_percent_of_whole_l441_44168


namespace NUMINAMATH_GPT_calc_residue_modulo_l441_44169

theorem calc_residue_modulo :
  let a := 320
  let b := 16
  let c := 28
  let d := 5
  let e := 7
  let n := 14
  (a * b - c * d + e) % n = 3 :=
by
  sorry

end NUMINAMATH_GPT_calc_residue_modulo_l441_44169


namespace NUMINAMATH_GPT_part1_part2_part3_l441_44166

-- Part 1: There exists a real number a such that a + 1/a ≤ 2
theorem part1 : ∃ a : ℝ, a + 1/a ≤ 2 := sorry

-- Part 2: For all positive real numbers a and b, b/a + a/b ≥ 2
theorem part2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : b / a + a / b ≥ 2 := sorry

-- Part 3: For positive real numbers x and y such that x + 2y = 1, then 2/x + 1/y ≥ 8
theorem part3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 2 / x + 1 / y ≥ 8 := sorry

end NUMINAMATH_GPT_part1_part2_part3_l441_44166


namespace NUMINAMATH_GPT_modified_determinant_l441_44133

def determinant_2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem modified_determinant (x y z w : ℝ)
  (h : determinant_2x2 x y z w = 6) :
  determinant_2x2 x (5 * x + 4 * y) z (5 * z + 4 * w) = 24 := by
  sorry

end NUMINAMATH_GPT_modified_determinant_l441_44133


namespace NUMINAMATH_GPT_logarithmic_relationship_l441_44164

theorem logarithmic_relationship (a b : ℝ) (h1 : a = Real.logb 16 625) (h2 : b = Real.logb 2 25) : a = b / 2 :=
sorry

end NUMINAMATH_GPT_logarithmic_relationship_l441_44164


namespace NUMINAMATH_GPT_focus_of_parabola_l441_44162

theorem focus_of_parabola (p : ℝ) :
  (∃ p, x ^ 2 = 4 * p * y ∧ x ^ 2 = 4 * 1 * y) → (0, p) = (0, 1) :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l441_44162


namespace NUMINAMATH_GPT_fraction_division_l441_44127

theorem fraction_division : (3 / 4) / (5 / 8) = 6 / 5 := 
by
  sorry

end NUMINAMATH_GPT_fraction_division_l441_44127


namespace NUMINAMATH_GPT_black_cards_taken_out_l441_44115

theorem black_cards_taken_out (total_black_cards remaining_black_cards : ℕ)
  (h1 : total_black_cards = 26) (h2 : remaining_black_cards = 21) :
  total_black_cards - remaining_black_cards = 5 :=
by
  sorry

end NUMINAMATH_GPT_black_cards_taken_out_l441_44115


namespace NUMINAMATH_GPT_find_n_l441_44176

theorem find_n (n : ℕ) (h : (2 * n + 1) / 3 = 2022) : n = 3033 :=
sorry

end NUMINAMATH_GPT_find_n_l441_44176


namespace NUMINAMATH_GPT_vertex_x_coordinate_of_quadratic_l441_44111

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 8 * x + 15

-- Define the x-coordinate of the vertex
def vertex_x_coordinate (f : ℝ → ℝ) : ℝ := 4

-- The theorem to prove
theorem vertex_x_coordinate_of_quadratic :
  vertex_x_coordinate quadratic_function = 4 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_vertex_x_coordinate_of_quadratic_l441_44111


namespace NUMINAMATH_GPT_frequency_of_middle_group_l441_44143

theorem frequency_of_middle_group :
  ∃ m : ℝ, m + (1/3) * m = 200 ∧ (1/3) * m = 50 :=
by
  sorry

end NUMINAMATH_GPT_frequency_of_middle_group_l441_44143


namespace NUMINAMATH_GPT_track_extension_needed_l441_44157

noncomputable def additional_track_length (r : ℝ) (g1 g2 : ℝ) : ℝ :=
  let l1 := r / g1
  let l2 := r / g2
  l2 - l1

theorem track_extension_needed :
  additional_track_length 800 0.04 0.015 = 33333 :=
by
  sorry

end NUMINAMATH_GPT_track_extension_needed_l441_44157


namespace NUMINAMATH_GPT_general_term_seq_l441_44147

theorem general_term_seq 
  (a : ℕ → ℚ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 5/3) 
  (h_rec : ∀ n, n > 0 → a (n + 2) = (5 / 3) * a (n + 1) - (2 / 3) * a n) : 
  ∀ n, a n = 2 - (3 / 2) * (2 / 3)^n :=
by
  sorry

end NUMINAMATH_GPT_general_term_seq_l441_44147


namespace NUMINAMATH_GPT_intersection_point_exists_l441_44181

theorem intersection_point_exists :
  ∃ (x y z t : ℝ), (x = 1 - 2 * t) ∧ (y = 2 + t) ∧ (z = -1 - t) ∧
                   (x - 2 * y + 5 * z + 17 = 0) ∧ 
                   (x = -1) ∧ (y = 3) ∧ (z = -2) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_exists_l441_44181


namespace NUMINAMATH_GPT_find_k_l441_44124

noncomputable def k := 3

theorem find_k :
  (∀ x : ℝ, (Real.sin x ^ k) * (Real.sin (k * x)) + (Real.cos x ^ k) * (Real.cos (k * x)) = Real.cos (2 * x) ^ k) ↔ k = 3 :=
sorry

end NUMINAMATH_GPT_find_k_l441_44124


namespace NUMINAMATH_GPT_exists_same_color_points_at_distance_one_l441_44150

theorem exists_same_color_points_at_distance_one (coloring : ℝ × ℝ → Fin 3) :
  ∃ (p q : ℝ × ℝ), (coloring p = coloring q) ∧ (dist p q = 1) := sorry

end NUMINAMATH_GPT_exists_same_color_points_at_distance_one_l441_44150


namespace NUMINAMATH_GPT_pedestrian_walking_time_in_interval_l441_44196

noncomputable def bus_departure_interval : ℕ := 5  -- Condition 1: Buses depart every 5 minutes
noncomputable def buses_same_direction : ℕ := 11  -- Condition 2: 11 buses passed him going the same direction
noncomputable def buses_opposite_direction : ℕ := 13  -- Condition 3: 13 buses came from opposite direction
noncomputable def bus_speed_factor : ℕ := 8  -- Condition 4: Bus speed is 8 times the pedestrian's speed
noncomputable def min_walking_time : ℚ := 57 + 1 / 7 -- Correct Answer: Minimum walking time
noncomputable def max_walking_time : ℚ := 62 + 2 / 9 -- Correct Answer: Maximum walking time

theorem pedestrian_walking_time_in_interval (t : ℚ)
  (h1 : bus_departure_interval = 5)
  (h2 : buses_same_direction = 11)
  (h3 : buses_opposite_direction = 13)
  (h4 : bus_speed_factor = 8) :
  min_walking_time ≤ t ∧ t ≤ max_walking_time :=
sorry

end NUMINAMATH_GPT_pedestrian_walking_time_in_interval_l441_44196


namespace NUMINAMATH_GPT_inverse_proportional_example_l441_44165

variable (x y : ℝ)

def inverse_proportional (x y : ℝ) := y = 8 / (x - 1)

theorem inverse_proportional_example
  (h1 : y = 4)
  (h2 : x = 3) :
  inverse_proportional x y :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportional_example_l441_44165


namespace NUMINAMATH_GPT_min_cubes_l441_44119

theorem min_cubes (a b c : ℕ) (h₁ : (a - 1) * (b - 1) * (c - 1) = 240) : a * b * c = 385 :=
  sorry

end NUMINAMATH_GPT_min_cubes_l441_44119


namespace NUMINAMATH_GPT_number_of_triangles_in_polygon_l441_44192

theorem number_of_triangles_in_polygon {n : ℕ} (h : n > 0) :
  let vertices := (2 * n + 1)
  ∃ triangles_containing_center : ℕ, triangles_containing_center = (n * (n + 1) * (2 * n + 1)) / 6 :=
sorry

end NUMINAMATH_GPT_number_of_triangles_in_polygon_l441_44192


namespace NUMINAMATH_GPT_arithmetic_seq_inequality_l441_44112

-- Definition for the sum of the first n terms of an arithmetic sequence
def sum_arith_seq (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_seq_inequality (a₁ : ℕ) (d : ℕ) (n : ℕ) (h : d > 0) :
  sum_arith_seq a₁ d n + sum_arith_seq a₁ d (3 * n) > 2 * sum_arith_seq a₁ d (2 * n) := by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_inequality_l441_44112


namespace NUMINAMATH_GPT_curve_is_circle_l441_44109

theorem curve_is_circle (r θ : ℝ) (h : r = 1 / (Real.sin θ + Real.cos θ)) : 
  ∃ k : ℝ, ∀ x y : ℝ, (x^2 + y^2 = k^2) → 
    (r^2 = x^2 + y^2 ∧ ∃ (θ : ℝ), x/r = Real.cos θ ∧ y/r = Real.sin θ) :=
sorry

end NUMINAMATH_GPT_curve_is_circle_l441_44109


namespace NUMINAMATH_GPT_john_moves_540kg_l441_44148

-- Conditions
def used_to_back_squat : ℝ := 200
def increased_by : ℝ := 50
def front_squat_ratio : ℝ := 0.8
def triple_ratio : ℝ := 0.9

-- Definitions based on conditions
def current_back_squat : ℝ := used_to_back_squat + increased_by
def current_front_squat : ℝ := front_squat_ratio * current_back_squat
def one_triple : ℝ := triple_ratio * current_front_squat
def three_triples : ℝ := 3 * one_triple

-- The proof statement
theorem john_moves_540kg : three_triples = 540 := by
  sorry

end NUMINAMATH_GPT_john_moves_540kg_l441_44148


namespace NUMINAMATH_GPT_tv_price_reduction_l441_44199

theorem tv_price_reduction (x : ℝ) (Q : ℝ) (P : ℝ) (h1 : Q > 0) (h2 : P > 0) (h3 : P*(1 - x/100) * 1.85 * Q = 1.665 * P * Q) : x = 10 :=
by 
  sorry

end NUMINAMATH_GPT_tv_price_reduction_l441_44199


namespace NUMINAMATH_GPT_caleb_hamburgers_total_l441_44188

def total_spent : ℝ := 66.50
def cost_single : ℝ := 1.00
def cost_double : ℝ := 1.50
def num_double : ℕ := 33

theorem caleb_hamburgers_total : 
  ∃ n : ℕ,  n = 17 + num_double ∧ 
            (num_double * cost_double) + (n - num_double) * cost_single = total_spent := by
sorry

end NUMINAMATH_GPT_caleb_hamburgers_total_l441_44188


namespace NUMINAMATH_GPT_max_gcd_15n_plus_4_8n_plus_1_l441_44158

theorem max_gcd_15n_plus_4_8n_plus_1 (n : ℕ) (h : n > 0) : 
  ∃ g, g = gcd (15 * n + 4) (8 * n + 1) ∧ g ≤ 17 :=
sorry

end NUMINAMATH_GPT_max_gcd_15n_plus_4_8n_plus_1_l441_44158


namespace NUMINAMATH_GPT_bluegrass_percentage_l441_44179

-- Define the problem conditions
def seed_mixture_X_ryegrass_percentage : ℝ := 40
def seed_mixture_Y_ryegrass_percentage : ℝ := 25
def seed_mixture_Y_fescue_percentage : ℝ := 75
def mixture_X_Y_ryegrass_percentage : ℝ := 30
def mixture_weight_percentage_X : ℝ := 33.33333333333333

-- Prove that the percentage of bluegrass in seed mixture X is 60%
theorem bluegrass_percentage (X_ryegrass : ℝ) (Y_ryegrass : ℝ) (Y_fescue : ℝ) (mixture_ryegrass : ℝ) (weight_percentage_X : ℝ) :
  X_ryegrass = seed_mixture_X_ryegrass_percentage →
  Y_ryegrass = seed_mixture_Y_ryegrass_percentage →
  Y_fescue = seed_mixture_Y_fescue_percentage →
  mixture_ryegrass = mixture_X_Y_ryegrass_percentage →
  weight_percentage_X = mixture_weight_percentage_X →
  (100 - X_ryegrass) = 60 :=
by
  intro hX_ryegrass hY_ryegrass hY_fescue hmixture_ryegrass hweight_X
  rw [hX_ryegrass]
  sorry

end NUMINAMATH_GPT_bluegrass_percentage_l441_44179


namespace NUMINAMATH_GPT_tin_to_copper_ratio_l441_44152

theorem tin_to_copper_ratio (L_A T_A T_B C_B : ℝ) 
  (h_total_mass_A : L_A + T_A = 90)
  (h_ratio_A : L_A / T_A = 3 / 4)
  (h_total_mass_B : T_B + C_B = 140)
  (h_total_tin : T_A + T_B = 91.42857142857143) :
  T_B / C_B = 2 / 5 :=
sorry

end NUMINAMATH_GPT_tin_to_copper_ratio_l441_44152


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l441_44195

theorem isosceles_triangle_base_length
  (b : ℕ)
  (congruent_side : ℕ)
  (perimeter : ℕ)
  (h1 : congruent_side = 8)
  (h2 : perimeter = 25)
  (h3 : 2 * congruent_side + b = perimeter) :
  b = 9 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l441_44195


namespace NUMINAMATH_GPT_find_n_l441_44123

-- Define the operation €
def operation (x y : ℕ) : ℕ := 2 * x * y

-- State the theorem
theorem find_n (n : ℕ) (h : operation 8 (operation 4 n) = 640) : n = 5 :=
  by
  sorry

end NUMINAMATH_GPT_find_n_l441_44123


namespace NUMINAMATH_GPT_find_constants_u_v_l441_44160

theorem find_constants_u_v
  (n p r1 r2 : ℝ)
  (h1 : r1 + r2 = n)
  (h2 : r1 * r2 = p) :
  ∃ u v, (r1^4 + r2^4 = -u) ∧ (r1^4 * r2^4 = v) ∧ u = -(n^4 - 4*p*n^2 + 2*p^2) ∧ v = p^4 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_u_v_l441_44160


namespace NUMINAMATH_GPT_pqr_value_l441_44161

theorem pqr_value (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (h1 : p + q + r = 24)
  (h2 : (1 / p : ℚ) + (1 / q) + (1 / r) + 240 / (p * q * r) = 1): 
  p * q * r = 384 :=
by
  sorry

end NUMINAMATH_GPT_pqr_value_l441_44161


namespace NUMINAMATH_GPT_necessary_sufficient_condition_l441_44135

theorem necessary_sufficient_condition (n : ℕ) (hn : n > 0) :
  (∃ x : ℚ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ n = 1 :=
sorry

end NUMINAMATH_GPT_necessary_sufficient_condition_l441_44135


namespace NUMINAMATH_GPT_ending_number_of_range_l441_44155

theorem ending_number_of_range (n : ℕ) (h : ∃ k, 1 ≤ k ∧ k ≤ 5 ∧ n = 29 + 11 * k) : n = 77 := by
  sorry

end NUMINAMATH_GPT_ending_number_of_range_l441_44155


namespace NUMINAMATH_GPT_inequality_proof_l441_44105

variable {a b c d : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  64 * (abcd + 1) / (a + b + c + d)^2 ≤ a^2 + b^2 + c^2 + d^2 + 1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l441_44105


namespace NUMINAMATH_GPT_encoded_base5_to_base10_l441_44193

-- Given definitions
def base5_to_int (d1 d2 d3 : ℕ) : ℕ := d1 * 25 + d2 * 5 + d3

def V := 2
def W := 0
def X := 4
def Y := 1
def Z := 3

-- Prove that the base-10 expression for the integer coded as XYZ is 108
theorem encoded_base5_to_base10 :
  base5_to_int X Y Z = 108 :=
sorry

end NUMINAMATH_GPT_encoded_base5_to_base10_l441_44193


namespace NUMINAMATH_GPT_number_of_good_colorings_l441_44153

theorem number_of_good_colorings (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) : 
  ∃ (good_colorings : ℕ), good_colorings = 6 * (2^n - 4 + 4 * 2^(m-2)) :=
sorry

end NUMINAMATH_GPT_number_of_good_colorings_l441_44153


namespace NUMINAMATH_GPT_englishman_land_earnings_l441_44129

noncomputable def acres_to_square_yards (acres : ℝ) : ℝ := acres * 4840
noncomputable def square_yards_to_square_meters (sq_yards : ℝ) : ℝ := sq_yards * (0.9144 ^ 2)
noncomputable def square_meters_to_hectares (sq_meters : ℝ) : ℝ := sq_meters / 10000
noncomputable def cost_of_land (hectares : ℝ) (price_per_hectare : ℝ) : ℝ := hectares * price_per_hectare

theorem englishman_land_earnings
  (acres_owned : ℝ)
  (price_per_hectare : ℝ)
  (acre_to_yard : ℝ)
  (yard_to_meter : ℝ)
  (hectare_to_meter : ℝ)
  (h1 : acres_owned = 2)
  (h2 : price_per_hectare = 500000)
  (h3 : acre_to_yard = 4840)
  (h4 : yard_to_meter = 0.9144)
  (h5 : hectare_to_meter = 10000)
  : cost_of_land (square_meters_to_hectares (square_yards_to_square_meters (acres_to_square_yards acres_owned))) price_per_hectare = 404685.6 := sorry

end NUMINAMATH_GPT_englishman_land_earnings_l441_44129


namespace NUMINAMATH_GPT_M_subset_N_cond_l441_44107

theorem M_subset_N_cond (a : ℝ) (h : 0 < a) :
  (∀ p : ℝ × ℝ, p ∈ {p : ℝ × ℝ | p.fst^2 + p.snd^2 = a^2} → p ∈ {p : ℝ × ℝ | |p.fst + p.snd| + |p.fst - p.snd| ≤ 2}) ↔ (0 < a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_GPT_M_subset_N_cond_l441_44107


namespace NUMINAMATH_GPT_nancy_total_savings_l441_44190

noncomputable def total_savings : ℝ :=
  let cost_this_month := 9 * 5
  let cost_last_month := 8 * 4
  let cost_next_month := 7 * 6
  let discount_this_month := 0.20 * cost_this_month
  let discount_last_month := 0.20 * cost_last_month
  let discount_next_month := 0.20 * cost_next_month
  discount_this_month + discount_last_month + discount_next_month

theorem nancy_total_savings : total_savings = 23.80 :=
by
  sorry

end NUMINAMATH_GPT_nancy_total_savings_l441_44190


namespace NUMINAMATH_GPT_number_of_zeros_of_f_l441_44113

noncomputable def f (a x : ℝ) := x * Real.log x - a * x^2 - x

theorem number_of_zeros_of_f (a : ℝ) (h : |a| ≥ 1 / (2 * Real.exp 1)) :
  ∃! x, f a x = 0 :=
sorry

end NUMINAMATH_GPT_number_of_zeros_of_f_l441_44113


namespace NUMINAMATH_GPT_coffee_price_decrease_is_37_5_l441_44163

-- Define the initial and new prices
def initial_price_per_packet := 12 / 3
def new_price_per_packet := 10 / 4

-- Define the calculation of the percent decrease
def percent_decrease (initial_price : ℚ) (new_price : ℚ) : ℚ :=
  ((initial_price - new_price) / initial_price) * 100

-- The theorem statement
theorem coffee_price_decrease_is_37_5 :
  percent_decrease initial_price_per_packet new_price_per_packet = 37.5 := by
  sorry

end NUMINAMATH_GPT_coffee_price_decrease_is_37_5_l441_44163


namespace NUMINAMATH_GPT_solve_for_z_l441_44189

theorem solve_for_z :
  ∃ z : ℤ, (∀ x y : ℤ, x = 11 → y = 8 → 2 * x + 3 * z = 5 * y) → z = 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_z_l441_44189


namespace NUMINAMATH_GPT_hexagon_perimeter_is_24_l441_44137

-- Conditions given in the problem
def AB : ℝ := 3
def EF : ℝ := 3
def BE : ℝ := 4
def AF : ℝ := 4
def CD : ℝ := 5
def DF : ℝ := 5

-- Statement to show that the perimeter is 24 units
theorem hexagon_perimeter_is_24 :
  AB + BE + CD + DF + EF + AF = 24 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_perimeter_is_24_l441_44137


namespace NUMINAMATH_GPT_track_length_l441_44175

theorem track_length (x : ℝ) (b_speed s_speed : ℝ) (b_dist1 s_dist1 s_dist2 : ℝ)
  (h1 : b_dist1 = 80)
  (h2 : s_dist1 = x / 2 - 80)
  (h3 : s_dist2 = s_dist1 + 180)
  (h4 : x / 4 * b_speed = (x / 2 - 80) * s_speed)
  (h5 : x / 4 * ((x / 2) - 100) = (x / 2 + 100) * s_speed) :
  x = 520 := 
sorry

end NUMINAMATH_GPT_track_length_l441_44175


namespace NUMINAMATH_GPT_sequence_is_increasing_l441_44102

def S (n : ℕ) : ℤ :=
  n^2 + 2 * n - 2

def a : ℕ → ℤ
| 0       => 0
| 1       => 1
| n + 1   => S (n + 1) - S n

theorem sequence_is_increasing : ∀ n m : ℕ, n < m → a n < a m :=
  sorry

end NUMINAMATH_GPT_sequence_is_increasing_l441_44102


namespace NUMINAMATH_GPT_sum_of_coefficients_l441_44121

noncomputable def coeff_sum (x y z : ℝ) : ℝ :=
  let p := (x + 2*y - z)^8  
  -- extract and sum coefficients where exponent of x is 2 and exponent of y is not 1
  sorry

theorem sum_of_coefficients (x y z : ℝ) :
  coeff_sum x y z = 364 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l441_44121


namespace NUMINAMATH_GPT_subtract_29_after_46_l441_44177

theorem subtract_29_after_46 (x : ℤ) (h : x - 46 = 15) : x - 29 = 32 :=
sorry

end NUMINAMATH_GPT_subtract_29_after_46_l441_44177


namespace NUMINAMATH_GPT_find_function_l441_44182

noncomputable def solution_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = x + f y → ∃ c : ℝ, ∀ x : ℝ, f x = x + c

theorem find_function (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = x + f y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
sorry

end NUMINAMATH_GPT_find_function_l441_44182


namespace NUMINAMATH_GPT_correct_answer_is_B_l441_44184

def is_permutation_problem (desc : String) : Prop :=
  desc = "Permutation"

def check_problem_A : Prop :=
  ¬ is_permutation_problem "Selecting 2 out of 8 students to participate in a knowledge competition"

def check_problem_B : Prop :=
  is_permutation_problem "If 10 people write letters to each other once, how many letters are written in total"

def check_problem_C : Prop :=
  ¬ is_permutation_problem "There are 5 points on a plane, with no three points collinear, what is the maximum number of lines that can be determined by these 5 points"

def check_problem_D : Prop :=
  ¬ is_permutation_problem "From the numbers 1, 2, 3, 4, choose any two numbers to multiply, how many different results are there"

theorem correct_answer_is_B : check_problem_A ∧ check_problem_B ∧ check_problem_C ∧ check_problem_D → 
  ("B" = "B") := by
  sorry

end NUMINAMATH_GPT_correct_answer_is_B_l441_44184


namespace NUMINAMATH_GPT_max_partial_sum_l441_44131

variable (a_n : ℕ → ℤ) (a_1 : ℤ) (d : ℤ)
variable (S : ℕ → ℤ)

-- Define the arithmetic sequence and the conditions given
def arithmetic_sequence (a_n : ℕ → ℤ) (a_1 : ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a_n n = a_1 + n * d

def condition1 (a_1 : ℤ) : Prop := a_1 > 0

def condition2 (a_n : ℕ → ℤ) (d : ℤ) : Prop := 3 * (a_n 8) = 5 * (a_n 13)

-- Define the partial sum of the arithmetic sequence
def partial_sum (S : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
∀ n : ℕ, S n = n * (a_n 1 + a_n n) / 2

-- Define the main problem: Prove that S_20 is the greatest
theorem max_partial_sum (a_n : ℕ → ℤ) (a_1 : ℤ) (d : ℤ) (S : ℕ → ℤ) :
  arithmetic_sequence a_n a_1 d →
  condition1 a_1 →
  condition2 a_n d →
  partial_sum S a_n →
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → S 20 ≥ S n := by
  sorry

end NUMINAMATH_GPT_max_partial_sum_l441_44131


namespace NUMINAMATH_GPT_sum_of_numbers_l441_44125

theorem sum_of_numbers : 
  5678 + 6785 + 7856 + 8567 = 28886 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l441_44125


namespace NUMINAMATH_GPT_train_speed_l441_44171

-- Definitions to capture the conditions
def length_of_train : ℝ := 100
def length_of_bridge : ℝ := 300
def time_to_cross_bridge : ℝ := 36

-- The speed of the train calculated according to the condition
def total_distance : ℝ := length_of_train + length_of_bridge

theorem train_speed : total_distance / time_to_cross_bridge = 11.11 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l441_44171


namespace NUMINAMATH_GPT_max_value_of_y_l441_44117

theorem max_value_of_y (x : ℝ) (h : x < 5/4) : 
  ∃ y : ℝ, y = 4 * x - 2 + 1 / (4 * x - 5) ∧ y ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_value_of_y_l441_44117


namespace NUMINAMATH_GPT_max_value_x_2y_2z_l441_44106

theorem max_value_x_2y_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) : x + 2*y + 2*z ≤ 15 :=
sorry

end NUMINAMATH_GPT_max_value_x_2y_2z_l441_44106


namespace NUMINAMATH_GPT_leos_time_is_1230_l441_44186

theorem leos_time_is_1230
  (theo_watch_slow: Int)
  (theo_watch_fast_belief: Int)
  (leo_watch_fast: Int)
  (leo_watch_slow_belief: Int)
  (theo_thinks_time: Int):
  theo_watch_slow = 10 ∧
  theo_watch_fast_belief = 5 ∧
  leo_watch_fast = 5 ∧
  leo_watch_slow_belief = 10 ∧
  theo_thinks_time = 720
  → leo_thinks_time = 750 :=
by
  sorry

end NUMINAMATH_GPT_leos_time_is_1230_l441_44186


namespace NUMINAMATH_GPT_price_of_thermometer_l441_44108

noncomputable def thermometer_price : ℝ := 2

theorem price_of_thermometer
  (T : ℝ)
  (price_hot_water_bottle : ℝ := 6)
  (hot_water_bottles_sold : ℕ := 60)
  (total_sales : ℝ := 1200)
  (thermometers_sold : ℕ := 7 * hot_water_bottles_sold)
  (thermometers_sales : ℝ := total_sales - (price_hot_water_bottle * hot_water_bottles_sold)) :
  T = thermometer_price :=
by
  sorry

end NUMINAMATH_GPT_price_of_thermometer_l441_44108


namespace NUMINAMATH_GPT_cliff_collection_has_180_rocks_l441_44118

noncomputable def cliffTotalRocks : ℕ :=
  let shiny_igneous_rocks := 40
  let total_igneous_rocks := shiny_igneous_rocks * 3 / 2
  let total_sedimentary_rocks := total_igneous_rocks * 2
  total_igneous_rocks + total_sedimentary_rocks

theorem cliff_collection_has_180_rocks :
  let shiny_igneous_rocks := 40
  let total_igneous_rocks := shiny_igneous_rocks * 3 / 2
  let total_sedimentary_rocks := total_igneous_rocks * 2
  total_igneous_rocks + total_sedimentary_rocks = 180 := sorry

end NUMINAMATH_GPT_cliff_collection_has_180_rocks_l441_44118


namespace NUMINAMATH_GPT_shelves_used_l441_44154

def coloring_books := 87
def sold_books := 33
def books_per_shelf := 6

theorem shelves_used (h1: coloring_books - sold_books = 54) : 54 / books_per_shelf = 9 :=
by
  sorry

end NUMINAMATH_GPT_shelves_used_l441_44154


namespace NUMINAMATH_GPT_speed_of_man_rowing_upstream_l441_44185

theorem speed_of_man_rowing_upstream (V_m V_downstream V_upstream V_s : ℝ) 
  (h1 : V_m = 25) 
  (h2 : V_downstream = 38) :
  V_upstream = V_m - (V_downstream - V_m) :=
by
  sorry

end NUMINAMATH_GPT_speed_of_man_rowing_upstream_l441_44185


namespace NUMINAMATH_GPT_simplify_expr_l441_44180

open Real

theorem simplify_expr (x : ℝ) (hx : x ≠ 0) : 
  sqrt (1 + ( (x^6 - 2) / (3 * x^3) )^2) = sqrt (x^12 + 5 * x^6 + 4) / (3 * x^3) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l441_44180


namespace NUMINAMATH_GPT_school_xx_percentage_increase_l441_44159

theorem school_xx_percentage_increase
  (X Y : ℕ) -- denote the number of students at school XX and YY last year
  (H_Y : Y = 2400) -- condition: school YY had 2400 students last year
  (H_total : X + Y = 4000) -- condition: total number of students last year was 4000
  (H_increase_YY : YY_increase = (3 * Y) / 100) -- condition: 3 percent increase at school YY
  (H_difference : XX_increase = YY_increase + 40) -- condition: school XX grew by 40 more students than YY
  : (XX_increase * 100) / X = 7 :=
by
  sorry

end NUMINAMATH_GPT_school_xx_percentage_increase_l441_44159


namespace NUMINAMATH_GPT_find_speed_of_stream_l441_44198

-- Define the given conditions
def boat_speed_still_water : ℝ := 14
def distance_downstream : ℝ := 72
def time_downstream : ℝ := 3.6

-- Define the speed of the stream (to be proven)
def speed_of_stream : ℝ := 6

-- The statement of the problem
theorem find_speed_of_stream 
  (h1 : boat_speed_still_water = 14)
  (h2 : distance_downstream = 72)
  (h3 : time_downstream = 3.6)
  (speed_of_stream_eq : boat_speed_still_water + speed_of_stream = distance_downstream / time_downstream) :
  speed_of_stream = 6 := 
by 
  sorry

end NUMINAMATH_GPT_find_speed_of_stream_l441_44198


namespace NUMINAMATH_GPT_range_of_a_l441_44103

noncomputable def f (x : ℝ) (a : ℝ) := Real.log (3 * x + a / x - 2)

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → f x a ≤ f y a) ↔ (-1 < a ∧ a ≤ 3) := 
sorry

end NUMINAMATH_GPT_range_of_a_l441_44103


namespace NUMINAMATH_GPT_sums_of_squares_divisibility_l441_44130

theorem sums_of_squares_divisibility :
  (∀ n : ℤ, (3 * n^2 + 2) % 3 ≠ 0) ∧ (∃ n : ℤ, (3 * n^2 + 2) % 11 = 0) := 
by
  sorry

end NUMINAMATH_GPT_sums_of_squares_divisibility_l441_44130


namespace NUMINAMATH_GPT_arithmetic_seq_term_six_l441_44132

theorem arithmetic_seq_term_six {a : ℕ → ℝ} (a1 : ℝ) (S3 : ℝ) (h1 : a1 = 2) (h2 : S3 = 12) :
  a 6 = 12 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_term_six_l441_44132


namespace NUMINAMATH_GPT_neg_cube_squared_l441_44100

theorem neg_cube_squared (x : ℝ) : (-x^3) ^ 2 = x ^ 6 :=
by
  sorry

end NUMINAMATH_GPT_neg_cube_squared_l441_44100


namespace NUMINAMATH_GPT_sum_of_numbers_on_cards_l441_44197

-- Define the natural numbers condition
variables {a b c d e f g h : ℕ}

-- The theorem statement
theorem sum_of_numbers_on_cards (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) :
  a + b + c + d + e + f + g + h = 21 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_on_cards_l441_44197


namespace NUMINAMATH_GPT_factorize_diff_of_squares_factorize_common_factor_diff_of_squares_l441_44173

-- Problem 1: Prove the factorization of 4x^2 - 25y^2
theorem factorize_diff_of_squares (x y : ℝ) : 4 * x^2 - 25 * y^2 = (2 * x + 5 * y) * (2 * x - 5 * y) := 
sorry

-- Problem 2: Prove the factorization of -3xy^3 + 27x^3y
theorem factorize_common_factor_diff_of_squares (x y : ℝ) : 
  -3 * x * y^3 + 27 * x^3 * y = -3 * x * y * (y + 3 * x) * (y - 3 * x) := 
sorry

end NUMINAMATH_GPT_factorize_diff_of_squares_factorize_common_factor_diff_of_squares_l441_44173


namespace NUMINAMATH_GPT_cars_pass_same_order_l441_44183

theorem cars_pass_same_order (num_cars : ℕ) (num_points : ℕ)
    (cities_speeds speeds_outside_cities : Fin num_cars → ℝ) :
    num_cars = 10 → num_points = 2011 → 
    ∃ (p1 p2 : Fin num_points), p1 ≠ p2 ∧ (∀ i j : Fin num_cars, (i < j) → 
    (cities_speeds i) / (cities_speeds i + speeds_outside_cities i) = 
    (cities_speeds j) / (cities_speeds j + speeds_outside_cities j) → p1 = p2 ) :=
by
  sorry

end NUMINAMATH_GPT_cars_pass_same_order_l441_44183


namespace NUMINAMATH_GPT_students_like_both_l441_44110

variable (total_students : ℕ) 
variable (students_like_sea : ℕ) 
variable (students_like_mountains : ℕ) 
variable (students_like_neither : ℕ) 

theorem students_like_both (h1 : total_students = 500)
                           (h2 : students_like_sea = 337)
                           (h3 : students_like_mountains = 289)
                           (h4 : students_like_neither = 56) :
  (students_like_sea + students_like_mountains - (total_students - students_like_neither)) = 182 :=
sorry

end NUMINAMATH_GPT_students_like_both_l441_44110


namespace NUMINAMATH_GPT_lara_total_space_larger_by_1500_square_feet_l441_44172

theorem lara_total_space_larger_by_1500_square_feet :
  let length_rect := 30
  let width_rect := 50
  let area_rect := length_rect * width_rect
  let side_square := width_rect
  let area_square := side_square * side_square
  let total_area := area_rect + area_square
  total_area - area_square = 1500 :=
by
  -- Definitions
  let length_rect := 30
  let width_rect := 50
  let area_rect := length_rect * width_rect
  let side_square := width_rect
  let area_square := side_square * side_square
  let total_area := area_rect + area_square
  
  -- Calculation
  have h_area_rect : area_rect = 1500 := by
    norm_num [area_rect, length_rect, width_rect]

  have h_area_square : area_square = 2500 := by
    norm_num [area_square, side_square]

  have h_total_area : total_area = 4000 := by
    norm_num [total_area, h_area_rect, h_area_square]

  -- Final comparison
  have h_difference : total_area - area_square = 1500 := by
    norm_num [total_area, area_square, h_area_square]

  exact h_difference

end NUMINAMATH_GPT_lara_total_space_larger_by_1500_square_feet_l441_44172
