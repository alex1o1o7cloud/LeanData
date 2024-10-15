import Mathlib

namespace NUMINAMATH_GPT_k_value_l1292_129228

noncomputable def find_k (AB BC AC BD : ℝ) (h_AB : AB = 3) (h_BC : BC = 4) (h_AC : AC = 5) (h_BD : BD = (12/7) * Real.sqrt 2) : ℝ :=
  12 / 7

theorem k_value (AB BC AC BD : ℝ) (h_AB : AB = 3) (h_BC : BC = 4) (h_AC : AC = 5) (h_BD : BD = (12/7) * Real.sqrt 2) : 
  find_k AB BC AC BD h_AB h_BC h_AC h_BD = 12 / 7 :=
by
  sorry

end NUMINAMATH_GPT_k_value_l1292_129228


namespace NUMINAMATH_GPT_odd_and_periodic_function_l1292_129246

noncomputable def f : ℝ → ℝ := sorry

lemma given_conditions (x : ℝ) : 
  (f (10 + x) = f (10 - x)) ∧ (f (20 - x) = -f (20 + x)) :=
  sorry

theorem odd_and_periodic_function (x : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f (x + 40) = f x) :=
  sorry

end NUMINAMATH_GPT_odd_and_periodic_function_l1292_129246


namespace NUMINAMATH_GPT_hunter_ants_l1292_129236

variable (spiders : ℕ) (ladybugs_before : ℕ) (ladybugs_flew : ℕ) (total_insects : ℕ)

theorem hunter_ants (h1 : spiders = 3)
                    (h2 : ladybugs_before = 8)
                    (h3 : ladybugs_flew = 2)
                    (h4 : total_insects = 21) :
  ∃ ants : ℕ, ants = total_insects - (spiders + (ladybugs_before - ladybugs_flew)) ∧ ants = 12 :=
by
  sorry

end NUMINAMATH_GPT_hunter_ants_l1292_129236


namespace NUMINAMATH_GPT_unique_function_l1292_129262

theorem unique_function (f : ℝ → ℝ) (hf : ∀ x : ℝ, 0 ≤ x → 0 ≤ f x)
  (cond1 : ∀ x : ℝ, 0 ≤ x → 4 * f x ≥ 3 * x)
  (cond2 : ∀ x : ℝ, 0 ≤ x → f (4 * f x - 3 * x) = x) :
  ∀ x : ℝ, 0 ≤ x → f x = x :=
by
  sorry

end NUMINAMATH_GPT_unique_function_l1292_129262


namespace NUMINAMATH_GPT_imaginary_number_m_l1292_129265

theorem imaginary_number_m (m : ℝ) : 
  (∀ Z, Z = (m + 2 * Complex.I) / (1 + Complex.I) → Z.im = 0 → Z.re = 0) → m = -2 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_number_m_l1292_129265


namespace NUMINAMATH_GPT_sqrt_neg_sq_eq_two_l1292_129267

theorem sqrt_neg_sq_eq_two : Real.sqrt ((-2 : ℝ)^2) = 2 := by
  -- Proof intentionally omitted.
  sorry

end NUMINAMATH_GPT_sqrt_neg_sq_eq_two_l1292_129267


namespace NUMINAMATH_GPT_value_of_x_plus_y_l1292_129280

theorem value_of_x_plus_y (x y : ℚ) (h1 : 1 / x + 1 / y = 5) (h2 : 1 / x - 1 / y = -9) : x + y = -5 / 14 := sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l1292_129280


namespace NUMINAMATH_GPT_certain_number_is_2_l1292_129206

theorem certain_number_is_2 
    (X : ℕ) 
    (Y : ℕ) 
    (h1 : X = 15) 
    (h2 : 0.40 * (X : ℝ) = 0.80 * 5 + (Y : ℝ)) : 
    Y = 2 := 
  sorry

end NUMINAMATH_GPT_certain_number_is_2_l1292_129206


namespace NUMINAMATH_GPT_coprime_odd_sum_of_floors_l1292_129278

theorem coprime_odd_sum_of_floors (p q : ℕ) (hp : p % 2 = 1) (hq : q % 2 = 1) (h_coprime : Nat.gcd p q = 1) : 
  (List.sum (List.map (λ i => Nat.floor ((i • q : ℚ) / p)) ((List.range (p / 2 + 1)).tail)) +
   List.sum (List.map (λ i => Nat.floor ((i • p : ℚ) / q)) ((List.range (q / 2 + 1)).tail))) =
  (p - 1) * (q - 1) / 4 :=
by
  sorry

end NUMINAMATH_GPT_coprime_odd_sum_of_floors_l1292_129278


namespace NUMINAMATH_GPT_unit_prices_min_selling_price_l1292_129238

-- Problem 1: Unit price determination
theorem unit_prices (x y : ℕ) (hx : 3600 / x * 2 = 5400 / y) (hy : y = x - 5) : x = 20 ∧ y = 15 := 
by 
  sorry

-- Problem 2: Minimum selling price for 50% profit margin
theorem min_selling_price (a : ℕ) (hx : 3600 / 20 = 180) (hy : 180 * 2 = 360) (hz : 540 * a ≥ 13500) : a ≥ 25 := 
by 
  sorry

end NUMINAMATH_GPT_unit_prices_min_selling_price_l1292_129238


namespace NUMINAMATH_GPT_original_time_to_complete_book_l1292_129258

-- Define the problem based on the given conditions
variables (n : ℕ) (T : ℚ)

-- Define the conditions
def condition1 : Prop := 
  ∃ (n T : ℚ), 
  n / T = (n + 3) / (0.75 * T) ∧
  n / T = (n - 3) / (T + 5 / 6)

-- State the theorem with the correct answer
theorem original_time_to_complete_book : condition1 → T = 5 / 3 :=
by sorry

end NUMINAMATH_GPT_original_time_to_complete_book_l1292_129258


namespace NUMINAMATH_GPT_min_value_expr_l1292_129219

noncomputable def find_min_value (a b c d : ℝ) (x y : ℝ) : ℝ :=
  x / c^2 + y^2 / d^2

theorem min_value_expr (a b c d : ℝ) (h : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ) :
  ∃ x y : ℝ, find_min_value a b c d x y = -abs a / c^2 := 
sorry

end NUMINAMATH_GPT_min_value_expr_l1292_129219


namespace NUMINAMATH_GPT_correct_equation_l1292_129292

def initial_count_A : ℕ := 54
def initial_count_B : ℕ := 48
def new_count_A (x : ℕ) : ℕ := initial_count_A + x
def new_count_B (x : ℕ) : ℕ := initial_count_B - x

theorem correct_equation (x : ℕ) : new_count_A x = 2 * new_count_B x := 
sorry

end NUMINAMATH_GPT_correct_equation_l1292_129292


namespace NUMINAMATH_GPT_internet_usage_minutes_l1292_129256

-- Define the given conditions
variables (M P E : ℕ)

-- Problem statement
theorem internet_usage_minutes (h : P ≠ 0) : 
  (∀ M P E : ℕ, ∃ y : ℕ, y = (100 * E * M) / P) :=
by {
  sorry
}

end NUMINAMATH_GPT_internet_usage_minutes_l1292_129256


namespace NUMINAMATH_GPT_prove_p_or_q_l1292_129286

-- Define propositions p and q
def p : Prop := ∃ n : ℕ, 0 = 2 * n
def q : Prop := ∃ m : ℕ, 3 = 2 * m

-- The Lean statement to prove
theorem prove_p_or_q : p ∨ q := by
  sorry

end NUMINAMATH_GPT_prove_p_or_q_l1292_129286


namespace NUMINAMATH_GPT_total_participants_l1292_129270

theorem total_participants (freshmen sophomores : ℕ) (h1 : freshmen = 8) (h2 : sophomores = 5 * freshmen) : freshmen + sophomores = 48 := 
by
  sorry

end NUMINAMATH_GPT_total_participants_l1292_129270


namespace NUMINAMATH_GPT_algebra_expression_evaluation_l1292_129253

theorem algebra_expression_evaluation (a b : ℝ) (h : a + 3 * b = 4) : 2 * a + 6 * b - 1 = 7 := by
  sorry

end NUMINAMATH_GPT_algebra_expression_evaluation_l1292_129253


namespace NUMINAMATH_GPT_average_speed_approx_l1292_129264

noncomputable def average_speed : ℝ :=
  let distance1 := 7
  let speed1 := 10
  let distance2 := 10
  let speed2 := 7
  let distance3 := 5
  let speed3 := 12
  let distance4 := 8
  let speed4 := 6
  let total_distance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let time4 := distance4 / speed4
  let total_time := time1 + time2 + time3 + time4
  total_distance / total_time

theorem average_speed_approx : abs (average_speed - 7.73) < 0.01 := by
  -- The necessary definitions fulfill the conditions and hence we put sorry here
  sorry

end NUMINAMATH_GPT_average_speed_approx_l1292_129264


namespace NUMINAMATH_GPT_value_of_x2_minus_y2_l1292_129204

theorem value_of_x2_minus_y2 (x y : ℚ) (h1 : x + y = 8 / 15) (h2 : x - y = 2 / 15) : x^2 - y^2 = 16 / 225 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x2_minus_y2_l1292_129204


namespace NUMINAMATH_GPT_solve_for_x_l1292_129291

theorem solve_for_x (x y : ℝ) (h1 : 3 * x - 2 * y = 8) (h2 : 2 * x + 3 * y = 1) : x = 2 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1292_129291


namespace NUMINAMATH_GPT_circle_ratio_increase_l1292_129212

theorem circle_ratio_increase (r : ℝ) (h : r + 2 ≠ 0) : 
  (2 * Real.pi * (r + 2)) / (2 * (r + 2)) = Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circle_ratio_increase_l1292_129212


namespace NUMINAMATH_GPT_spherical_coordinates_standard_equivalence_l1292_129293

def std_spherical_coords (ρ θ φ: ℝ) : Prop :=
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi

theorem spherical_coordinates_standard_equivalence :
  std_spherical_coords 5 (11 * Real.pi / 6) (2 * Real.pi - 5 * Real.pi / 3) :=
by
  sorry

end NUMINAMATH_GPT_spherical_coordinates_standard_equivalence_l1292_129293


namespace NUMINAMATH_GPT_first_year_payment_l1292_129276

theorem first_year_payment (x : ℝ) 
  (second_year : ℝ := x + 2)
  (third_year : ℝ := x + 5)
  (fourth_year : ℝ := x + 9)
  (total_payment : ℝ := x + second_year + third_year + fourth_year)
  (h : total_payment = 96) : x = 20 := 
by
  sorry

end NUMINAMATH_GPT_first_year_payment_l1292_129276


namespace NUMINAMATH_GPT_minimum_people_who_like_both_l1292_129277

theorem minimum_people_who_like_both
    (total_people : ℕ)
    (vivaldi_likers : ℕ)
    (chopin_likers : ℕ)
    (people_surveyed : total_people = 150)
    (like_vivaldi : vivaldi_likers = 120)
    (like_chopin : chopin_likers = 90) :
    ∃ (both_likers : ℕ), both_likers = 60 ∧
                            vivaldi_likers + chopin_likers - both_likers ≤ total_people :=
by 
  sorry

end NUMINAMATH_GPT_minimum_people_who_like_both_l1292_129277


namespace NUMINAMATH_GPT_clover_walk_distance_l1292_129298

theorem clover_walk_distance (total_distance days walks_per_day : ℝ) (h1 : total_distance = 90) (h2 : days = 30) (h3 : walks_per_day = 2) :
  (total_distance / days / walks_per_day = 1.5) :=
by
  sorry

end NUMINAMATH_GPT_clover_walk_distance_l1292_129298


namespace NUMINAMATH_GPT_find_b_value_l1292_129231

theorem find_b_value (a b c A B C : ℝ) 
  (h1 : a = 1)
  (h2 : B = 120 * (π / 180))
  (h3 : c = b * Real.cos C + c * Real.cos B)
  (h4 : c = 1) : 
  b = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_b_value_l1292_129231


namespace NUMINAMATH_GPT_chicago_denver_temperature_l1292_129226

def temperature_problem (C D : ℝ) (N : ℝ) : Prop :=
  (C = D - N) ∧ (abs ((D - N + 4) - (D - 2)) = 1)

theorem chicago_denver_temperature (C D N : ℝ) (h : temperature_problem C D N) :
  N = 5 ∨ N = 7 → (5 * 7 = 35) :=
by sorry

end NUMINAMATH_GPT_chicago_denver_temperature_l1292_129226


namespace NUMINAMATH_GPT_find_percent_defective_l1292_129214

def percent_defective (D : ℝ) : Prop :=
  (0.04 * D = 0.32)

theorem find_percent_defective : ∃ D, percent_defective D ∧ D = 8 := by
  sorry

end NUMINAMATH_GPT_find_percent_defective_l1292_129214


namespace NUMINAMATH_GPT_average_weight_of_Arun_l1292_129252

theorem average_weight_of_Arun :
  ∃ avg_weight : Real,
    (avg_weight = (65 + 68) / 2) ∧
    ∀ w : Real, (65 < w ∧ w < 72) ∧ (60 < w ∧ w < 70) ∧ (w ≤ 68) → avg_weight = 66.5 :=
by
  -- we will fill the details of the proof here
  sorry

end NUMINAMATH_GPT_average_weight_of_Arun_l1292_129252


namespace NUMINAMATH_GPT_arithmetic_and_geometric_sequence_l1292_129296

-- Definitions based on given conditions
def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

-- Main statement to prove
theorem arithmetic_and_geometric_sequence :
  ∀ (x y a b c : ℝ), 
  is_arithmetic_sequence 1 x y 4 →
  is_geometric_sequence (-2) a b c (-8) →
  (y - x) / b = -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_and_geometric_sequence_l1292_129296


namespace NUMINAMATH_GPT_sum_s_r_values_l1292_129247

def r_values : List ℤ := [-2, -1, 0, 1, 3]
def r_range : List ℤ := [-1, 0, 1, 3, 5]

def s (x : ℤ) : ℤ := if 1 ≤ x then 2 * x + 1 else 0

theorem sum_s_r_values :
  (s 1) + (s 3) + (s 5) = 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_s_r_values_l1292_129247


namespace NUMINAMATH_GPT_find_integer_l1292_129225

theorem find_integer (a b c d : ℕ) (h1 : a + b + c + d = 18) 
  (h2 : b + c = 11) (h3 : a - d = 3) (h4 : (10^3 * a + 10^2 * b + 10 * c + d) % 9 = 0) :
  10^3 * a + 10^2 * b + 10 * c + d = 5262 ∨ 10^3 * a + 10^2 * b + 10 * c + d = 5622 := 
by
  sorry

end NUMINAMATH_GPT_find_integer_l1292_129225


namespace NUMINAMATH_GPT_distance_between_points_l1292_129282

open Complex Real

def joe_point : ℂ := 2 + 3 * I
def gracie_point : ℂ := -2 + 2 * I

theorem distance_between_points : abs (joe_point - gracie_point) = sqrt 17 := by
  sorry

end NUMINAMATH_GPT_distance_between_points_l1292_129282


namespace NUMINAMATH_GPT_range_of_z_l1292_129209

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  12 ≤ x^2 + 4*y^2 ∧ x^2 + 4*y^2 ≤ 20 :=
by
  sorry

end NUMINAMATH_GPT_range_of_z_l1292_129209


namespace NUMINAMATH_GPT_solve_for_x_l1292_129268

theorem solve_for_x (q r x : ℚ)
  (h1 : 5 / 6 = q / 90)
  (h2 : 5 / 6 = (q + r) / 102)
  (h3 : 5 / 6 = (x - r) / 150) :
  x = 135 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l1292_129268


namespace NUMINAMATH_GPT_parabola_y_relation_l1292_129242

-- Conditions of the problem
def parabola (x : ℝ) (c : ℝ) : ℝ := 2 * x^2 - 4 * x + c

-- The proof problem statement
theorem parabola_y_relation (c y1 y2 y3 : ℝ) :
  parabola (-4) c = y1 →
  parabola (-2) c = y2 →
  parabola (1 / 2) c = y3 →
  y1 > y2 ∧ y2 > y3 :=
by
  sorry

end NUMINAMATH_GPT_parabola_y_relation_l1292_129242


namespace NUMINAMATH_GPT_number_of_morse_code_symbols_l1292_129257

-- Define the number of sequences for different lengths
def sequences_of_length (n : Nat) : Nat :=
  2 ^ n

theorem number_of_morse_code_symbols : 
  (sequences_of_length 1) + (sequences_of_length 2) + (sequences_of_length 3) + (sequences_of_length 4) + (sequences_of_length 5) = 62 := by
  sorry

end NUMINAMATH_GPT_number_of_morse_code_symbols_l1292_129257


namespace NUMINAMATH_GPT_number_of_blue_candles_l1292_129269

-- Conditions
def grandfather_age : ℕ := 79
def yellow_candles : ℕ := 27
def red_candles : ℕ := 14
def total_candles : ℕ := grandfather_age
def yellow_red_candles : ℕ := yellow_candles + red_candles
def blue_candles : ℕ := total_candles - yellow_red_candles

-- Proof statement
theorem number_of_blue_candles : blue_candles = 38 :=
by
  -- sorry indicates the proof is omitted
  sorry

end NUMINAMATH_GPT_number_of_blue_candles_l1292_129269


namespace NUMINAMATH_GPT_sum_c_eq_l1292_129260

-- Definitions and conditions
def a_n : ℕ → ℝ := λ n => 2 ^ n
def b_n : ℕ → ℝ := λ n => 2 * n
def c_n (n : ℕ) : ℝ := a_n n * b_n n

-- Sum of the first n terms of sequence {c_n}
def sum_c (n : ℕ) : ℝ := (Finset.range n).sum c_n

-- Theorem statement
theorem sum_c_eq (n : ℕ) : sum_c n = (n - 1) * 2 ^ (n + 2) + 4 :=
sorry

end NUMINAMATH_GPT_sum_c_eq_l1292_129260


namespace NUMINAMATH_GPT_six_identities_l1292_129240

theorem six_identities :
    (∀ x, (2 * x - 1) * (x - 3) = 2 * x^2 - 7 * x + 3) ∧
    (∀ x, (2 * x + 1) * (x + 3) = 2 * x^2 + 7 * x + 3) ∧
    (∀ x, (2 - x) * (1 - 3 * x) = 2 - 7 * x + 3 * x^2) ∧
    (∀ x, (2 + x) * (1 + 3 * x) = 2 + 7 * x + 3 * x^2) ∧
    (∀ x y, (2 * x - y) * (x - 3 * y) = 2 * x^2 - 7 * x * y + 3 * y^2) ∧
    (∀ x y, (2 * x + y) * (x + 3 * y) = 2 * x^2 + 7 * x * y + 3 * y^2) →
    6 = 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_six_identities_l1292_129240


namespace NUMINAMATH_GPT_three_x_plus_three_y_plus_three_z_l1292_129237

theorem three_x_plus_three_y_plus_three_z (x y z : ℝ) 
  (h1 : y + z = 20 - 5 * x)
  (h2 : x + z = -18 - 5 * y)
  (h3 : x + y = 10 - 5 * z) :
  3 * x + 3 * y + 3 * z = 36 / 7 := by
  sorry

end NUMINAMATH_GPT_three_x_plus_three_y_plus_three_z_l1292_129237


namespace NUMINAMATH_GPT_middle_number_of_consecutive_numbers_sum_of_squares_eq_2030_l1292_129259

theorem middle_number_of_consecutive_numbers_sum_of_squares_eq_2030 :
  ∃ n : ℕ, n^2 + (n+1)^2 + (n+2)^2 = 2030 ∧ (n + 1) = 26 :=
by sorry

end NUMINAMATH_GPT_middle_number_of_consecutive_numbers_sum_of_squares_eq_2030_l1292_129259


namespace NUMINAMATH_GPT_solve_for_a_l1292_129266

theorem solve_for_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 13) (h3 : 13 ∣ 51^2016 - a) : a = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_a_l1292_129266


namespace NUMINAMATH_GPT_incorrect_statement_l1292_129211

def geom_seq (a r : ℝ) : ℕ → ℝ
| 0       => a
| (n + 1) => r * geom_seq a r n

theorem incorrect_statement
  (a : ℝ) (r : ℝ) (S6 : ℝ)
  (h1 : r = 1 / 2)
  (h2 : S6 = a * (1 - (1 / 2) ^ 6) / (1 - 1 / 2))
  (h3 : S6 = 378) :
  geom_seq a r 2 / S6 ≠ 1 / 8 :=
by 
  have h4 : a = 192 := by sorry
  have h5 : geom_seq 192 (1 / 2) 2 = 192 * (1 / 2) ^ 2 := by sorry
  exact sorry

end NUMINAMATH_GPT_incorrect_statement_l1292_129211


namespace NUMINAMATH_GPT_lewis_speed_l1292_129223

theorem lewis_speed
  (v : ℕ)
  (john_speed : ℕ := 40)
  (distance_AB : ℕ := 240)
  (meeting_distance : ℕ := 160)
  (time_john_to_meeting : ℕ := meeting_distance / john_speed)
  (distance_lewis_traveled : ℕ := distance_AB + (distance_AB - meeting_distance))
  (v_eq : v = distance_lewis_traveled / time_john_to_meeting) :
  v = 80 :=
by
  sorry

end NUMINAMATH_GPT_lewis_speed_l1292_129223


namespace NUMINAMATH_GPT_heather_total_distance_l1292_129221

theorem heather_total_distance :
  let d1 := 0.3333333333333333
  let d2 := 0.3333333333333333
  let d3 := 0.08333333333333333
  d1 + d2 + d3 = 0.75 :=
by
  sorry

end NUMINAMATH_GPT_heather_total_distance_l1292_129221


namespace NUMINAMATH_GPT_cone_radius_correct_l1292_129283

noncomputable def cone_radius (CSA l : ℝ) : ℝ := CSA / (Real.pi * l)

theorem cone_radius_correct :
  cone_radius 1539.3804002589986 35 = 13.9 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_cone_radius_correct_l1292_129283


namespace NUMINAMATH_GPT_certain_number_is_7000_l1292_129227

theorem certain_number_is_7000 (x : ℕ) (h1 : 1 / 10 * (1 / 100 * x) = x / 1000)
    (h2 : 1 / 10 * x = x / 10)
    (h3 : x / 10 - x / 1000 = 693) : 
  x = 7000 := 
sorry

end NUMINAMATH_GPT_certain_number_is_7000_l1292_129227


namespace NUMINAMATH_GPT_correct_survey_method_l1292_129250

-- Definitions for the conditions
def visionStatusOfMiddleSchoolStudentsNationwide := "Comprehensive survey is impractical for this large population."
def batchFoodContainsPreservatives := "Comprehensive survey is unnecessary, sampling survey would suffice."
def airQualityOfCity := "Comprehensive survey is impractical due to vast area, sampling survey is appropriate."
def passengersCarryProhibitedItems := "Comprehensive survey is necessary for security reasons."

-- Theorem stating that option C is the correct and reasonable choice
theorem correct_survey_method : airQualityOfCity = "Comprehensive survey is impractical due to vast area, sampling survey is appropriate." := by
  sorry

end NUMINAMATH_GPT_correct_survey_method_l1292_129250


namespace NUMINAMATH_GPT_cookie_sales_l1292_129294

theorem cookie_sales (n : ℕ) (h1 : 1 ≤ n - 11) (h2 : 1 ≤ n - 2) (h3 : (n - 11) + (n - 2) < n) : n = 12 :=
sorry

end NUMINAMATH_GPT_cookie_sales_l1292_129294


namespace NUMINAMATH_GPT_exponentiation_rule_l1292_129248

theorem exponentiation_rule (a : ℝ) : (a^2)^3 = a^6 :=
by sorry

end NUMINAMATH_GPT_exponentiation_rule_l1292_129248


namespace NUMINAMATH_GPT_Danielle_rooms_is_6_l1292_129213

-- Definitions for the problem conditions
def Heidi_rooms (Danielle_rooms : ℕ) : ℕ := 3 * Danielle_rooms
def Grant_rooms (Heidi_rooms : ℕ) : ℕ := Heidi_rooms / 9
def Grant_rooms_value : ℕ := 2

-- Theorem statement
theorem Danielle_rooms_is_6 (h : Grant_rooms_value = Grant_rooms (Heidi_rooms d)) : d = 6 :=
by
  sorry

end NUMINAMATH_GPT_Danielle_rooms_is_6_l1292_129213


namespace NUMINAMATH_GPT_functional_equation_solution_l1292_129201

variable (f : ℝ → ℝ)

-- Declare the conditions as hypotheses
axiom cond1 : ∀ x : ℝ, 0 < x → 0 < f x
axiom cond2 : f 1 = 1
axiom cond3 : ∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2

-- State the theorem to be proved
theorem functional_equation_solution : ∀ x : ℝ, f x = x :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l1292_129201


namespace NUMINAMATH_GPT_rows_seating_exactly_10_people_exists_l1292_129273

theorem rows_seating_exactly_10_people_exists :
  ∃ y x : ℕ, 73 = 10 * y + 9 * x ∧ (73 - 10 * y) % 9 = 0 := 
sorry

end NUMINAMATH_GPT_rows_seating_exactly_10_people_exists_l1292_129273


namespace NUMINAMATH_GPT_find_divisor_l1292_129244

theorem find_divisor (d x k j : ℤ) (h₁ : x = k * d + 5) (h₂ : 7 * x = j * d + 8) : d = 11 :=
sorry

end NUMINAMATH_GPT_find_divisor_l1292_129244


namespace NUMINAMATH_GPT_mr_rainwater_chickens_l1292_129284

theorem mr_rainwater_chickens :
  ∃ (Ch : ℕ), (∀ (C G : ℕ), C = 9 ∧ G = 4 * C ∧ G = 2 * Ch → Ch = 18) :=
by
  sorry

end NUMINAMATH_GPT_mr_rainwater_chickens_l1292_129284


namespace NUMINAMATH_GPT_total_savings_correct_l1292_129203

-- Definitions of savings per day and days saved for Josiah, Leah, and Megan
def josiah_saving_per_day : ℝ := 0.25
def josiah_days : ℕ := 24

def leah_saving_per_day : ℝ := 0.50
def leah_days : ℕ := 20

def megan_saving_per_day : ℝ := 1.00
def megan_days : ℕ := 12

-- Definition to calculate total savings for each child
def total_saving (saving_per_day : ℝ) (days : ℕ) : ℝ :=
  saving_per_day * days

-- Total amount saved by Josiah, Leah, and Megan
def total_savings : ℝ :=
  total_saving josiah_saving_per_day josiah_days +
  total_saving leah_saving_per_day leah_days +
  total_saving megan_saving_per_day megan_days

-- Theorem to prove the total savings is $28
theorem total_savings_correct : total_savings = 28 := by
  sorry

end NUMINAMATH_GPT_total_savings_correct_l1292_129203


namespace NUMINAMATH_GPT_integer_solution_abs_lt_sqrt2_l1292_129288

theorem integer_solution_abs_lt_sqrt2 (x : ℤ) (h : |x| < Real.sqrt 2) : x = -1 ∨ x = 0 ∨ x = 1 :=
sorry

end NUMINAMATH_GPT_integer_solution_abs_lt_sqrt2_l1292_129288


namespace NUMINAMATH_GPT_evaluate_expression_when_c_eq_4_and_k_eq_2_l1292_129222

theorem evaluate_expression_when_c_eq_4_and_k_eq_2 :
  ( (4^4 - 4 * (4 - 1)^4 + 2) ^ 4 ) = 18974736 :=
by
  -- Definitions
  let c := 4
  let k := 2
  -- Evaluations
  let a := c^c
  let b := c * (c - 1)^c
  let expression := (a - b + k)^c
  -- Proof
  have result : expression = 18974736 := sorry
  exact result

end NUMINAMATH_GPT_evaluate_expression_when_c_eq_4_and_k_eq_2_l1292_129222


namespace NUMINAMATH_GPT_quadratic_function_inequality_l1292_129215

theorem quadratic_function_inequality
  (x1 x2 : ℝ) (y1 y2 : ℝ)
  (hx1_pos : 0 < x1)
  (hx2_pos : x1 < x2)
  (hy1 : y1 = x1^2 - 1)
  (hy2 : y2 = x2^2 - 1) :
  y1 < y2 := 
sorry

end NUMINAMATH_GPT_quadratic_function_inequality_l1292_129215


namespace NUMINAMATH_GPT_new_student_weight_l1292_129254

theorem new_student_weight : 
  ∀ (w_new : ℕ), 
    (∀ (sum_weight: ℕ), 80 + sum_weight - w_new = sum_weight - 18) → 
      w_new = 62 := 
by
  intros w_new h
  sorry

end NUMINAMATH_GPT_new_student_weight_l1292_129254


namespace NUMINAMATH_GPT_range_of_function_l1292_129205

open Real

noncomputable def f (x : ℝ) : ℝ := -cos x ^ 2 - 4 * sin x + 6

theorem range_of_function : 
  ∀ y, (∃ x, y = f x) ↔ 2 ≤ y ∧ y ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_range_of_function_l1292_129205


namespace NUMINAMATH_GPT_find_a5_l1292_129297

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

-- Given conditions
variable (a : ℕ → ℝ)
variable (h_arith : arithmetic_sequence a)
variable (h_a1 : a 0 = 2)
variable (h_sum : a 1 + a 3 = 8)

-- The target question
theorem find_a5 : a 4 = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_a5_l1292_129297


namespace NUMINAMATH_GPT_cost_of_chairs_l1292_129290

-- Given conditions
def total_spent : ℕ := 56
def cost_of_table : ℕ := 34
def number_of_chairs : ℕ := 2

-- The target definition
def cost_of_one_chair : ℕ := 11

-- Statement to prove
theorem cost_of_chairs (N : ℕ) (T : ℕ) (C : ℕ) (x : ℕ) (hN : N = total_spent) (hT : T = cost_of_table) (hC : C = number_of_chairs) : x = cost_of_one_chair ↔ N - T = C * x :=
by
  sorry

end NUMINAMATH_GPT_cost_of_chairs_l1292_129290


namespace NUMINAMATH_GPT_range_of_m_l1292_129299

theorem range_of_m (α β m : ℝ) (hαβ : 0 < α ∧ α < 1 ∧ 1 < β ∧ β < 2)
  (h_eq : ∀ x, x^2 - 2*(m-1)*x + (m-1) = 0 ↔ (x = α ∨ x = β)) :
  2 < m ∧ m < 7 / 3 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l1292_129299


namespace NUMINAMATH_GPT_find_a_l1292_129261

-- Definitions
def parabola (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c
def vertex_property (a b c : ℤ) := 
  ∃ x y, x = 2 ∧ y = 5 ∧ y = parabola a b c x
def point_on_parabola (a b c : ℤ) := 
  ∃ x y, x = 1 ∧ y = 2 ∧ y = parabola a b c x

-- The main statement
theorem find_a {a b c : ℤ} (h_vertex : vertex_property a b c) (h_point : point_on_parabola a b c) : a = -3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_l1292_129261


namespace NUMINAMATH_GPT_find_c_plus_one_over_b_l1292_129239

theorem find_c_plus_one_over_b 
  (a b c : ℝ) 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) 
  (h1 : a * b * c = 1) 
  (h2 : a + 1 / c = 8) 
  (h3 : b + 1 / a = 20) : 
  c + 1 / b = 10 / 53 := 
sorry

end NUMINAMATH_GPT_find_c_plus_one_over_b_l1292_129239


namespace NUMINAMATH_GPT_division_result_l1292_129216

theorem division_result : 203515 / 2015 = 101 := 
by sorry

end NUMINAMATH_GPT_division_result_l1292_129216


namespace NUMINAMATH_GPT_adam_clothing_ratio_l1292_129275

-- Define the initial amount of clothing Adam took out
def initial_clothing_adam : ℕ := 4 + 4 + 8 + 20

-- Define the number of friends donating the same amount of clothing as Adam
def number_of_friends : ℕ := 3

-- Define the total number of clothes being donated
def total_donated_clothes : ℕ := 126

-- Define the ratio of the clothes Adam is keeping to the clothes he initially took out
def ratio_kept_to_initial (initial_clothing: ℕ) (total_donated: ℕ) (kept: ℕ) : Prop :=
  kept * initial_clothing = 0

-- Theorem statement
theorem adam_clothing_ratio :
  ratio_kept_to_initial initial_clothing_adam total_donated_clothes 0 :=
by 
  sorry

end NUMINAMATH_GPT_adam_clothing_ratio_l1292_129275


namespace NUMINAMATH_GPT_identity_function_uniq_l1292_129217

theorem identity_function_uniq (f g h : ℝ → ℝ)
    (hg : ∀ x, g x = x + 1)
    (hh : ∀ x, h x = x^2)
    (H1 : ∀ x, f (g x) = g (f x))
    (H2 : ∀ x, f (h x) = h (f x)) :
  ∀ x, f x = x :=
by
  sorry

end NUMINAMATH_GPT_identity_function_uniq_l1292_129217


namespace NUMINAMATH_GPT_find_cost_price_per_meter_l1292_129224

noncomputable def cost_price_per_meter
  (total_cloth : ℕ)
  (selling_price : ℕ)
  (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_cloth) / total_cloth

theorem find_cost_price_per_meter :
  cost_price_per_meter 75 4950 15 = 51 :=
by
  unfold cost_price_per_meter
  sorry

end NUMINAMATH_GPT_find_cost_price_per_meter_l1292_129224


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l1292_129251

def ellipse_touching_hyperbola (a b : ℝ) :=
  ∀ x y : ℝ, ( (x / a) ^ 2 + (y / b) ^ 2 = 1 ∧ y = 1 / x → False )

  theorem relationship_between_a_and_b (a b : ℝ) :
  ellipse_touching_hyperbola a b →
  a * b = 2 :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l1292_129251


namespace NUMINAMATH_GPT_yulia_profit_l1292_129229

-- Assuming the necessary definitions in the problem
def lemonade_revenue : ℕ := 47
def babysitting_revenue : ℕ := 31
def expenses : ℕ := 34
def profit : ℕ := lemonade_revenue + babysitting_revenue - expenses

-- The proof statement to prove Yulia's profit
theorem yulia_profit : profit = 44 := by
  sorry -- Proof is skipped

end NUMINAMATH_GPT_yulia_profit_l1292_129229


namespace NUMINAMATH_GPT_factor_54x5_135x9_l1292_129255

theorem factor_54x5_135x9 (x : ℝ) :
  54 * x ^ 5 - 135 * x ^ 9 = -27 * x ^ 5 * (5 * x ^ 4 - 2) :=
by 
  sorry

end NUMINAMATH_GPT_factor_54x5_135x9_l1292_129255


namespace NUMINAMATH_GPT_parabola_points_count_l1292_129263

theorem parabola_points_count :
  ∃ n : ℕ, n = 8 ∧ 
    (∀ x y : ℕ, (y = -((x^2 : ℤ) / 3) + 7 * (x : ℤ) + 54) → 1 ≤ x ∧ x ≤ 26 ∧ x % 3 = 0) :=
by
  sorry

end NUMINAMATH_GPT_parabola_points_count_l1292_129263


namespace NUMINAMATH_GPT_find_g_3_l1292_129233

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g_3 (h : ∀ x : ℝ, g (2 * x - 5) = 3 * x + 9) : g 3 = 21 :=
by
  sorry

end NUMINAMATH_GPT_find_g_3_l1292_129233


namespace NUMINAMATH_GPT_trapezoid_area_ratio_l1292_129295

theorem trapezoid_area_ratio (AD AO OB BC AB DO OC : ℝ) (h_eq1 : AD = 15) (h_eq2 : AO = 15) (h_eq3 : OB = 15) (h_eq4 : BC = 15)
  (h_eq5 : AB = 20) (h_eq6 : DO = 20) (h_eq7 : OC = 20) (is_trapezoid : true) (OP_perp_to_AB : true) 
  (X_mid_AD : true) (Y_mid_BC : true) : (5 + 7 = 12) :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_ratio_l1292_129295


namespace NUMINAMATH_GPT_sum_of_numbers_l1292_129243

theorem sum_of_numbers (a b c : ℕ) 
  (h1 : a ≤ b ∧ b ≤ c) 
  (h2 : b = 10) 
  (h3 : (a + b + c) / 3 = a + 15) 
  (h4 : (a + b + c) / 3 = c - 20) 
  (h5 : c = 2 * a)
  : a + b + c = 115 := by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1292_129243


namespace NUMINAMATH_GPT_tan_five_pi_over_four_l1292_129218

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
by
  sorry

end NUMINAMATH_GPT_tan_five_pi_over_four_l1292_129218


namespace NUMINAMATH_GPT_determine_m_l1292_129210

def f (x m : ℝ) : ℝ := x^2 - 3*x + m
def g (x m : ℝ) : ℝ := x^2 - 3*x + 5*m

theorem determine_m (m : ℝ) : 3 * f 5 m = 2 * g 5 m → m = 10 / 7 := 
by sorry

end NUMINAMATH_GPT_determine_m_l1292_129210


namespace NUMINAMATH_GPT_seq_a_n_value_l1292_129220

theorem seq_a_n_value (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n, S n = n^2)
  (h2 : ∀ n ≥ 2, a n = S n - S (n-1)) :
  a 10 = 19 :=
sorry

end NUMINAMATH_GPT_seq_a_n_value_l1292_129220


namespace NUMINAMATH_GPT_first_two_digits_of_1666_l1292_129202

/-- Lean 4 statement for the given problem -/
theorem first_two_digits_of_1666 (y k : ℕ) (H_nonzero_k : k ≠ 0) (H_nonzero_y : y ≠ 0) (H_y_six : y = 6) :
  (1666 / 100) = 16 := by
  sorry

end NUMINAMATH_GPT_first_two_digits_of_1666_l1292_129202


namespace NUMINAMATH_GPT_right_triangle_x_value_l1292_129287

theorem right_triangle_x_value (x Δ : ℕ) (h₁ : x > 0) (h₂ : Δ > 0) :
  ((x + 2 * Δ)^2 = x^2 + (x + Δ)^2) → 
  x = (Δ * (-1 + 2 * Real.sqrt 7)) / 2 := 
sorry

end NUMINAMATH_GPT_right_triangle_x_value_l1292_129287


namespace NUMINAMATH_GPT_part1_part2_l1292_129281

-- Define the predicate for the inequality
def prop (x m : ℝ) : Prop := x^2 - 2 * m * x - 3 * m^2 < 0

-- Define the set A
def A (m : ℝ) : Prop := m < -2 ∨ m > 2 / 3

-- Define the predicate for the other inequality
def prop_B (x a : ℝ) : Prop := x^2 - 2 * a * x + a^2 - 1 < 0

-- Define the set B in terms of a
def B (x a : ℝ) : Prop := a - 1 < x ∧ x < a + 1

-- Define the propositions required in the problem
theorem part1 (m : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → prop x m) ↔ A m :=
sorry

theorem part2 (a : ℝ) :
  (∀ x, B x a → A x) ∧ (∃ x, A x ∧ ¬ B x a) ↔ (a ≤ -3 ∨ a ≥ 5 / 3) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1292_129281


namespace NUMINAMATH_GPT_madison_classes_l1292_129285

/-- Madison's classes -/
def total_bell_rings : ℕ := 9

/-- Each class requires two bell rings (one to start, one to end) -/
def bell_rings_per_class : ℕ := 2

/-- The number of classes Madison has on Monday -/
theorem madison_classes (total_bell_rings bell_rings_per_class : ℕ) (last_class_start_only : total_bell_rings % bell_rings_per_class = 1) : 
  (total_bell_rings - 1) / bell_rings_per_class + 1 = 5 :=
by
  sorry

end NUMINAMATH_GPT_madison_classes_l1292_129285


namespace NUMINAMATH_GPT_MeganSavingsExceed500_l1292_129272

theorem MeganSavingsExceed500 :
  ∃ n : ℕ, n ≥ 7 ∧ ((3^n - 1) / 2 > 500) :=
sorry

end NUMINAMATH_GPT_MeganSavingsExceed500_l1292_129272


namespace NUMINAMATH_GPT_negation_of_forall_inequality_l1292_129274

theorem negation_of_forall_inequality :
  (¬ (∀ x : ℝ, x > 0 → x * Real.sin x < 2^x - 1)) ↔ (∃ x : ℝ, x > 0 ∧ x * Real.sin x ≥ 2^x - 1) :=
by sorry

end NUMINAMATH_GPT_negation_of_forall_inequality_l1292_129274


namespace NUMINAMATH_GPT_mike_total_earning_l1292_129235

theorem mike_total_earning 
  (first_job : ℕ := 52)
  (hours : ℕ := 12)
  (wage_per_hour : ℕ := 9) :
  first_job + (hours * wage_per_hour) = 160 :=
by
  sorry

end NUMINAMATH_GPT_mike_total_earning_l1292_129235


namespace NUMINAMATH_GPT_ant_impossibility_l1292_129249

-- Define the vertices and edges of a cube
structure Cube :=
(vertices : Finset ℕ) -- Representing a finite set of vertices
(edges : Finset (ℕ × ℕ)) -- Representing a finite set of edges between vertices
(valid_edge : ∀ e ∈ edges, ∃ v1 v2, (v1, v2) = e ∨ (v2, v1) = e)
(starting_vertex : ℕ)

-- Ant behavior on the cube
structure AntOnCube (C : Cube) :=
(is_path_valid : List ℕ → Prop) -- A property that checks the path is valid

-- Problem conditions translated: 
-- No retracing and specific visit numbers
noncomputable def ant_problem (C : Cube) (A : AntOnCube C) : Prop :=
  ∀ (path : List ℕ), A.is_path_valid path → ¬ (
    (path.count C.starting_vertex = 25) ∧ 
    (∀ v ∈ C.vertices, v ≠ C.starting_vertex → path.count v = 20)
  )

-- The final theorem statement
theorem ant_impossibility (C : Cube) (A : AntOnCube C) : ant_problem C A :=
by
  -- providing the theorem framework; proof omitted with sorry
  sorry

end NUMINAMATH_GPT_ant_impossibility_l1292_129249


namespace NUMINAMATH_GPT_sum_of_two_squares_l1292_129241

theorem sum_of_two_squares (a b : ℝ) : 2 * a^2 + 2 * b^2 = (a + b)^2 + (a - b)^2 :=
by sorry

end NUMINAMATH_GPT_sum_of_two_squares_l1292_129241


namespace NUMINAMATH_GPT_amount_C_l1292_129234

theorem amount_C (A B C : ℕ) 
  (h₁ : A + B + C = 900) 
  (h₂ : A + C = 400) 
  (h₃ : B + C = 750) : 
  C = 250 :=
sorry

end NUMINAMATH_GPT_amount_C_l1292_129234


namespace NUMINAMATH_GPT_combined_rate_last_year_l1292_129207

noncomputable def combine_effective_rate_last_year (r_increased: ℝ) (r_this_year: ℝ) : ℝ :=
  r_this_year / r_increased

theorem combined_rate_last_year
  (compounding_frequencies : List String)
  (r_increased : ℝ)
  (r_this_year : ℝ)
  (combined_interest_rate_this_year : r_this_year = 0.11)
  (interest_rate_increase : r_increased = 1.10) :
  combine_effective_rate_last_year r_increased r_this_year = 0.10 :=
by
  sorry

end NUMINAMATH_GPT_combined_rate_last_year_l1292_129207


namespace NUMINAMATH_GPT_length_of_DE_in_triangle_l1292_129232

noncomputable def triangle_length_DE (BC : ℝ) (C_deg: ℝ) (DE : ℝ) : Prop :=
  BC = 24 * Real.sqrt 2 ∧ C_deg = 45 ∧ DE = 12 * Real.sqrt 2

theorem length_of_DE_in_triangle :
  ∀ (BC : ℝ) (C_deg: ℝ) (DE : ℝ), (BC = 24 * Real.sqrt 2 ∧ C_deg = 45) → DE = 12 * Real.sqrt 2 :=
by
  intros BC C_deg DE h_cond
  have h_length := h_cond.2
  sorry

end NUMINAMATH_GPT_length_of_DE_in_triangle_l1292_129232


namespace NUMINAMATH_GPT_lizzy_final_amount_l1292_129279

-- Define constants
def m : ℕ := 80   -- cents from mother
def f : ℕ := 40   -- cents from father
def s : ℕ := 50   -- cents spent on candy
def u : ℕ := 70   -- cents from uncle
def t : ℕ := 90   -- cents for the toy
def c : ℕ := 110  -- cents change she received

-- Define the final amount calculation
def final_amount : ℕ := m + f - s + u - t + c

-- Prove the final amount is 160
theorem lizzy_final_amount : final_amount = 160 := by
  sorry

end NUMINAMATH_GPT_lizzy_final_amount_l1292_129279


namespace NUMINAMATH_GPT_per_minute_charge_after_6_minutes_l1292_129208

noncomputable def cost_plan_a (x : ℝ) (t : ℝ) : ℝ :=
  if t <= 6 then 0.60 else 0.60 + (t - 6) * x

noncomputable def cost_plan_b (t : ℝ) : ℝ :=
  t * 0.08

theorem per_minute_charge_after_6_minutes :
  ∃ (x : ℝ), cost_plan_a x 12 = cost_plan_b 12 ∧ x = 0.06 :=
by
  use 0.06
  simp [cost_plan_a, cost_plan_b]
  sorry

end NUMINAMATH_GPT_per_minute_charge_after_6_minutes_l1292_129208


namespace NUMINAMATH_GPT_workshop_total_workers_l1292_129245

noncomputable def average_salary_of_all (W : ℕ) : ℝ := 8000
noncomputable def average_salary_of_technicians : ℝ := 12000
noncomputable def average_salary_of_non_technicians : ℝ := 6000

theorem workshop_total_workers
    (W : ℕ)
    (T : ℕ := 7)
    (N : ℕ := W - T)
    (h1 : (T + N) = W)
    (h2 : average_salary_of_all W = 8000)
    (h3 : average_salary_of_technicians = 12000)
    (h4 : average_salary_of_non_technicians = 6000)
    (h5 : (7 * 12000) + (N * 6000) = (7 + N) * 8000) :
  W = 21 :=
by
  sorry


end NUMINAMATH_GPT_workshop_total_workers_l1292_129245


namespace NUMINAMATH_GPT_square_land_plot_area_l1292_129289

theorem square_land_plot_area (side_length : ℕ) (h1 : side_length = 40) : side_length * side_length = 1600 :=
by
  sorry

end NUMINAMATH_GPT_square_land_plot_area_l1292_129289


namespace NUMINAMATH_GPT_probability_y_gt_x_l1292_129271

-- Define the uniform distribution and the problem setup
def uniform_distribution (a b : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ b }

-- Define the variables
variables (x : ℝ) (hx : x ∈ uniform_distribution 0 3000) (y : ℝ) (hy : y ∈ uniform_distribution 0 6000)

-- Define the probability calculation function (assuming some proper definition for probability)
noncomputable def probability_event (E : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the event that Laurent's number is greater than Chloe's number
def event_y_gt_x : Set (ℝ × ℝ) := {p | p.2 > p.1}

-- State the theorem
theorem probability_y_gt_x (x : ℝ) (hx : x ∈ uniform_distribution 0 3000) (y : ℝ) (hy : y ∈ uniform_distribution 0 6000) :
  probability_event event_y_gt_x = 3/4 :=
sorry

end NUMINAMATH_GPT_probability_y_gt_x_l1292_129271


namespace NUMINAMATH_GPT_stratified_sampling_sophomores_selected_l1292_129230

theorem stratified_sampling_sophomores_selected 
  (total_freshmen : ℕ) (total_sophomores : ℕ) (total_seniors : ℕ) 
  (freshmen_selected : ℕ) (selection_ratio : ℕ) :
  total_freshmen = 210 →
  total_sophomores = 270 →
  total_seniors = 300 →
  freshmen_selected = 7 →
  selection_ratio = total_freshmen / freshmen_selected →
  selection_ratio = 30 →
  total_sophomores / selection_ratio = 9 :=
by sorry

end NUMINAMATH_GPT_stratified_sampling_sophomores_selected_l1292_129230


namespace NUMINAMATH_GPT_find_ordered_pair_l1292_129200

theorem find_ordered_pair (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  18 * m * n = 72 - 9 * m - 4 * n ↔ (m = 8 ∧ n = 36) := 
by 
  sorry

end NUMINAMATH_GPT_find_ordered_pair_l1292_129200
