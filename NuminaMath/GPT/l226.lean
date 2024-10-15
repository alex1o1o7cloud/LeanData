import Mathlib

namespace NUMINAMATH_GPT_problem1_problem2_l226_22623

-- Problem 1
theorem problem1 : (Real.sqrt 8 - Real.sqrt 27 - (4 * Real.sqrt (1 / 2) + Real.sqrt 12)) = -5 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem2 : ((Real.sqrt 6 + Real.sqrt 12) * (2 * Real.sqrt 3 - Real.sqrt 6) - 3 * Real.sqrt 32 / (Real.sqrt 2 / 2)) = -18 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l226_22623


namespace NUMINAMATH_GPT_integer_roots_and_composite_l226_22698

theorem integer_roots_and_composite (a b : ℤ) (h1 : ∃ x1 x2 : ℤ, x1 * x2 = 1 - b ∧ x1 + x2 = -a) (h2 : b ≠ 1) : 
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ m * n = (a^2 + b^2) := 
sorry

end NUMINAMATH_GPT_integer_roots_and_composite_l226_22698


namespace NUMINAMATH_GPT_gcd_largest_of_forms_l226_22659

theorem gcd_largest_of_forms (a b : ℕ) (h1 : a ≠ b) (h2 : a < 10) (h3 : b < 10) :
  Nat.gcd (100 * a + 11 * b) (101 * b + 10 * a) = 45 :=
by
  sorry

end NUMINAMATH_GPT_gcd_largest_of_forms_l226_22659


namespace NUMINAMATH_GPT_flowers_per_set_l226_22625

variable (totalFlowers : ℕ) (numSets : ℕ)

theorem flowers_per_set (h1 : totalFlowers = 270) (h2 : numSets = 3) : totalFlowers / numSets = 90 :=
by
  sorry

end NUMINAMATH_GPT_flowers_per_set_l226_22625


namespace NUMINAMATH_GPT_soap_remaining_days_l226_22603

theorem soap_remaining_days 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0)
  (daily_consumption : ℝ)
  (h4 : daily_consumption = a * b * c / 8) 
  (h5 : ∀ t : ℝ, t > 0 → t ≤ 7 → daily_consumption = (a * b * c - (a * b * c) * (1 / 8))) :
  ∃ t : ℝ, t = 1 :=
by 
  sorry

end NUMINAMATH_GPT_soap_remaining_days_l226_22603


namespace NUMINAMATH_GPT_elder_child_age_l226_22686

theorem elder_child_age (x : ℕ) (h : x + (x + 4) + (x + 8) + (x + 12) = 48) : (x + 12) = 18 :=
by
  sorry

end NUMINAMATH_GPT_elder_child_age_l226_22686


namespace NUMINAMATH_GPT_solve_rational_inequality_l226_22614

theorem solve_rational_inequality :
  {x : ℝ | (9*x^2 + 18 * x - 60) / ((3 * x - 4) * (x + 5)) < 4} =
  {x : ℝ | (-10 < x ∧ x < -5) ∨ (2/3 < x ∧ x < 4/3) ∨ (4/3 < x)} :=
by
  sorry

end NUMINAMATH_GPT_solve_rational_inequality_l226_22614


namespace NUMINAMATH_GPT_rotated_point_coordinates_l226_22675

noncomputable def A : ℝ × ℝ := (1, 2)

def rotate_90_counterclockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.snd, p.fst)

theorem rotated_point_coordinates :
  rotate_90_counterclockwise A = (-2, 1) :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_rotated_point_coordinates_l226_22675


namespace NUMINAMATH_GPT_sqrt_addition_l226_22656

theorem sqrt_addition : Real.sqrt 8 + Real.sqrt 2 = 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_addition_l226_22656


namespace NUMINAMATH_GPT_sheila_weekly_earnings_l226_22607

-- Variables
variables {hours_mon_wed_fri hours_tue_thu rate_per_hour : ℕ}

-- Conditions
def sheila_works_mwf : hours_mon_wed_fri = 8 := by sorry
def sheila_works_tue_thu : hours_tue_thu = 6 := by sorry
def sheila_rate : rate_per_hour = 11 := by sorry

-- Main statement to prove
theorem sheila_weekly_earnings : 
  3 * hours_mon_wed_fri + 2 * hours_tue_thu = 36 →
  rate_per_hour = 11 →
  (3 * hours_mon_wed_fri + 2 * hours_tue_thu) * rate_per_hour = 396 :=
by
  intros h_hours h_rate
  sorry

end NUMINAMATH_GPT_sheila_weekly_earnings_l226_22607


namespace NUMINAMATH_GPT_prime_divisor_greater_than_p_l226_22660

theorem prime_divisor_greater_than_p (p q : ℕ) (hp : Prime p) 
    (hq : Prime q) (hdiv : q ∣ 2^p - 1) : p < q := 
by
  sorry

end NUMINAMATH_GPT_prime_divisor_greater_than_p_l226_22660


namespace NUMINAMATH_GPT_correct_calculation_for_b_l226_22651

theorem correct_calculation_for_b (a : ℝ) : (a^2)^3 = a^6 := 
by sorry

end NUMINAMATH_GPT_correct_calculation_for_b_l226_22651


namespace NUMINAMATH_GPT_soccer_minimum_wins_l226_22629

/-
Given that a soccer team has won 60% of 45 matches played so far, 
prove that the minimum number of matches that the team still needs to win to reach a winning percentage of 75% is 27.
-/
theorem soccer_minimum_wins 
  (initial_matches : ℕ)                 -- the initial number of matches
  (initial_win_rate : ℚ)                -- the initial win rate (as a percentage)
  (desired_win_rate : ℚ)                -- the desired win rate (as a percentage)
  (initial_wins : ℕ)                    -- the initial number of wins

  -- Given conditions
  (h1 : initial_matches = 45)
  (h2 : initial_win_rate = 0.60)
  (h3 : desired_win_rate = 0.75)
  (h4 : initial_wins = 27):
  
  -- To prove: the minimum number of additional matches that need to be won is 27
  ∃ (n : ℕ), (initial_wins + n) / (initial_matches + n) = desired_win_rate ∧ 
                  n = 27 :=
by 
  sorry

end NUMINAMATH_GPT_soccer_minimum_wins_l226_22629


namespace NUMINAMATH_GPT_rectangle_midpoints_sum_l226_22682

theorem rectangle_midpoints_sum (A B C D M N O P : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (4, 0))
  (hC : C = (4, 3))
  (hD : D = (0, 3))
  (hM : M = (2, 0))
  (hN : N = (4, 1.5))
  (hO : O = (2, 3))
  (hP : P = (0, 1.5)) :
  (Real.sqrt ((2 - 0) ^ 2 + (0 - 0) ^ 2) + 
  Real.sqrt ((4 - 0) ^ 2 + (1.5 - 0) ^ 2) + 
  Real.sqrt ((2 - 0) ^ 2 + (3 - 0) ^ 2) + 
  Real.sqrt ((0 - 0) ^ 2 + (1.5 - 0) ^ 2)) = 11.38 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_midpoints_sum_l226_22682


namespace NUMINAMATH_GPT_mul_inv_mod_35_l226_22699

theorem mul_inv_mod_35 : (8 * 22) % 35 = 1 := 
  sorry

end NUMINAMATH_GPT_mul_inv_mod_35_l226_22699


namespace NUMINAMATH_GPT_pq_eq_neg72_l226_22617

theorem pq_eq_neg72 {p q : ℝ} (h : ∀ x, (x - 7) * (3 * x + 11) = x ^ 2 - 20 * x + 63 →
(p = x ∨ q = x) ∧ p ≠ q) : 
(p + 2) * (q + 2) = -72 :=
sorry

end NUMINAMATH_GPT_pq_eq_neg72_l226_22617


namespace NUMINAMATH_GPT_general_term_sequence_l226_22690

noncomputable def a (t : ℝ) (n : ℕ) : ℝ :=
if h : t ≠ 1 then (2 * (t^n - 1) / n) - 1 else 0

theorem general_term_sequence (t : ℝ) (n : ℕ) (hn : n ≠ 0) (h : t ≠ 1) :
  a t (n+1) = (2 * (t^(n+1) - 1) / (n+1)) - 1 := 
sorry

end NUMINAMATH_GPT_general_term_sequence_l226_22690


namespace NUMINAMATH_GPT_average_age_after_person_leaves_l226_22652

theorem average_age_after_person_leaves
  (average_age_seven : ℕ := 28)
  (num_people_initial : ℕ := 7)
  (person_leaves : ℕ := 20) :
  (average_age_seven * num_people_initial - person_leaves) / (num_people_initial - 1) = 29 := by
  sorry

end NUMINAMATH_GPT_average_age_after_person_leaves_l226_22652


namespace NUMINAMATH_GPT_stations_between_l226_22664

theorem stations_between (n : ℕ) (h : n * (n - 1) / 2 = 306) : n - 2 = 25 := 
by
  sorry

end NUMINAMATH_GPT_stations_between_l226_22664


namespace NUMINAMATH_GPT_MNPQ_is_rectangle_l226_22648

variable {Point : Type}
variable {A B C D M N P Q : Point}

def is_parallelogram (A B C D : Point) : Prop := sorry
def altitude (X Y : Point) : Prop := sorry
def rectangle (M N P Q : Point) : Prop := sorry

theorem MNPQ_is_rectangle 
  (h_parallelogram : is_parallelogram A B C D)
  (h_alt1 : altitude B M)
  (h_alt2 : altitude B N)
  (h_alt3 : altitude D P)
  (h_alt4 : altitude D Q) :
  rectangle M N P Q :=
sorry

end NUMINAMATH_GPT_MNPQ_is_rectangle_l226_22648


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l226_22661

theorem point_in_fourth_quadrant (a b : ℝ) (h1 : -4 * a < 0) (h2 : 2 + b < 0) : 
  (a > 0) ∧ (b < -2) → (a > 0) ∧ (b < 0) := 
by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l226_22661


namespace NUMINAMATH_GPT_symmetric_line_eq_l226_22650

theorem symmetric_line_eq (a b : ℝ) (ha : a ≠ 0) : 
  (∃ k m : ℝ, (∀ x: ℝ, ax + b = (k * ( -x)) + m ∧ (k = 1/a ∧ m = b/a )))  := 
sorry

end NUMINAMATH_GPT_symmetric_line_eq_l226_22650


namespace NUMINAMATH_GPT_max_tan_A_minus_B_l226_22672

open Real

-- Given conditions
variables {A B C a b c : ℝ}

-- Assume the triangle ABC with sides a, b, c opposite to angles A, B, C respectively
-- and the equation a * cos B - b * cos A = (3 / 5) * c holds.
def condition (a b c A B C : ℝ) : Prop :=
  a * cos B - b * cos A = (3 / 5) * c

-- Prove that the maximum value of tan(A - B) is 3/4
theorem max_tan_A_minus_B (a b c A B C : ℝ) (h : condition a b c A B C) :
  ∃ t : ℝ, t = tan (A - B) ∧ 0 ≤ t ∧ t ≤ 3 / 4 :=
sorry

end NUMINAMATH_GPT_max_tan_A_minus_B_l226_22672


namespace NUMINAMATH_GPT_uneaten_pancakes_time_l226_22610

theorem uneaten_pancakes_time:
  ∀ (production_rate_dad production_rate_mom consumption_rate_petya consumption_rate_vasya : ℕ) (k : ℕ),
    production_rate_dad = 70 →
    production_rate_mom = 100 →
    consumption_rate_petya = 10 * 4 → -- 10 pancakes in 15 minutes -> (10/15) * 60 = 40 per hour
    consumption_rate_vasya = 2 * consumption_rate_petya →
    k * ((production_rate_dad + production_rate_mom) / 60 - (consumption_rate_petya + consumption_rate_vasya) / 60) ≥ 20 →
    k ≥ 24 := 
by
  intros production_rate_dad production_rate_mom consumption_rate_petya consumption_rate_vasya k
  sorry

end NUMINAMATH_GPT_uneaten_pancakes_time_l226_22610


namespace NUMINAMATH_GPT_general_formula_a_n_general_formula_b_n_sum_c_n_T_n_l226_22677

open Classical

axiom S_n : ℕ → ℝ
axiom a_n : ℕ → ℝ
axiom b_n : ℕ → ℝ
axiom c_n : ℕ → ℝ
axiom T_n : ℕ → ℝ

noncomputable def general_a_n (n : ℕ) : ℝ :=
  sorry

axiom h1 : ∀ n, S_n n + a_n n = 2

theorem general_formula_a_n : ∀ n, a_n n = 1 / 2^(n-1) :=
  sorry

axiom h2 : b_n 1 = a_n 1
axiom h3 : ∀ n ≥ 2, b_n n = 3 * b_n (n-1) / (b_n (n-1) + 3)

theorem general_formula_b_n : ∀ n, b_n n = 3 / (n + 2) ∧
  (∀ n, 1 / b_n n = 1 + (n - 1) / 3) :=
  sorry

axiom h4 : ∀ n, c_n n = a_n n / b_n n

theorem sum_c_n_T_n : ∀ n, T_n n = 8 / 3 - (n + 4) / (3 * 2^(n-1)) :=
  sorry

end NUMINAMATH_GPT_general_formula_a_n_general_formula_b_n_sum_c_n_T_n_l226_22677


namespace NUMINAMATH_GPT_pointA_in_second_quadrant_l226_22632

def pointA : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem pointA_in_second_quadrant : isSecondQuadrant pointA :=
by
  sorry

end NUMINAMATH_GPT_pointA_in_second_quadrant_l226_22632


namespace NUMINAMATH_GPT_SUVs_purchased_l226_22630

theorem SUVs_purchased (x : ℕ) (hToyota : ℕ) (hHonda : ℕ) (hNissan : ℕ) 
  (hRatioToyota : hToyota = 7 * x) 
  (hRatioHonda : hHonda = 5 * x) 
  (hRatioNissan : hNissan = 3 * x) 
  (hToyotaSUV : ℕ) (hHondaSUV : ℕ) (hNissanSUV : ℕ) 
  (hToyotaSUV_num : hToyotaSUV = (50 * hToyota) / 100) 
  (hHondaSUV_num : hHondaSUV = (40 * hHonda) / 100) 
  (hNissanSUV_num : hNissanSUV = (30 * hNissan) / 100) : 
  hToyotaSUV + hHondaSUV + hNissanSUV = 64 := 
by
  sorry

end NUMINAMATH_GPT_SUVs_purchased_l226_22630


namespace NUMINAMATH_GPT_no_prime_p_such_that_22p2_plus_23_is_prime_l226_22609

theorem no_prime_p_such_that_22p2_plus_23_is_prime :
  ∀ p : ℕ, Prime p → ¬ Prime (22 * p ^ 2 + 23) :=
by
  sorry

end NUMINAMATH_GPT_no_prime_p_such_that_22p2_plus_23_is_prime_l226_22609


namespace NUMINAMATH_GPT_min_sum_of_factors_l226_22620

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 1800) : 
  a + b + c = 64 :=
sorry

end NUMINAMATH_GPT_min_sum_of_factors_l226_22620


namespace NUMINAMATH_GPT_sum_x_y_m_l226_22684

theorem sum_x_y_m (x y m : ℕ) (h1 : x >= 10 ∧ x < 100) (h2 : y >= 10 ∧ y < 100) 
  (h3 : ∃ a b : ℕ, x = 10 * a + b ∧ y = 10 * b + a ∧ a < 10 ∧ b < 10) 
  (h4 : x^2 - y^2 = 4 * m^2) : 
  x + y + m = 105 := 
sorry

end NUMINAMATH_GPT_sum_x_y_m_l226_22684


namespace NUMINAMATH_GPT_expenditure_may_to_july_l226_22685

theorem expenditure_may_to_july (spent_by_may : ℝ) (spent_by_july : ℝ) (h_may : spent_by_may = 0.8) (h_july : spent_by_july = 3.5) :
  spent_by_july - spent_by_may = 2.7 :=
by
  sorry

end NUMINAMATH_GPT_expenditure_may_to_july_l226_22685


namespace NUMINAMATH_GPT_average_first_two_numbers_l226_22640

theorem average_first_two_numbers (a1 a2 a3 a4 a5 a6 : ℝ)
  (h1 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 3.95)
  (h2 : (a3 + a4) / 2 = 3.85)
  (h3 : (a5 + a6) / 2 = 4.200000000000001) :
  (a1 + a2) / 2 = 3.8 :=
by
  sorry

end NUMINAMATH_GPT_average_first_two_numbers_l226_22640


namespace NUMINAMATH_GPT_num_factors_m_l226_22621

noncomputable def m : ℕ := 2^5 * 3^6 * 5^7 * 6^8

theorem num_factors_m : ∃ (k : ℕ), k = 1680 ∧ ∀ d : ℕ, d ∣ m ↔ ∃ (a b c : ℕ), 0 ≤ a ∧ a ≤ 13 ∧ 0 ≤ b ∧ b ≤ 14 ∧ 0 ≤ c ∧ c ≤ 7 ∧ d = 2^a * 3^b * 5^c :=
by 
sorry

end NUMINAMATH_GPT_num_factors_m_l226_22621


namespace NUMINAMATH_GPT_units_digit_17_mul_27_l226_22673

theorem units_digit_17_mul_27 : 
  let u17 := (17 % 10)
  let u27 := (27 % 10)
  let units_product := (u17 * u27) % 10
  units_product = 9 := by
  let u17 := (17 % 10)
  let u27 := (27 % 10)
  let units_product := (u17 * u27) % 10
  sorry

end NUMINAMATH_GPT_units_digit_17_mul_27_l226_22673


namespace NUMINAMATH_GPT_determine_x_l226_22605

theorem determine_x (x y : ℝ) (h : x / (x - 2) = (y^2 + 3 * y - 2) / (y^2 + 3 * y + 1)) : 
  x = 2 * y^2 + 6 * y + 4 := 
by
  sorry

end NUMINAMATH_GPT_determine_x_l226_22605


namespace NUMINAMATH_GPT_right_triangle_counterexample_l226_22644

def is_acute_angle (α : ℝ) : Prop := 0 < α ∧ α < 90

def is_right_angle (α : ℝ) : Prop := α = 90

def is_triangle (α β γ : ℝ) : Prop := α + β + γ = 180

def is_acute_triangle (α β γ : ℝ) : Prop := is_acute_angle α ∧ is_acute_angle β ∧ is_acute_angle γ

def is_right_triangle (α β γ : ℝ) : Prop := 
  (is_right_angle α ∧ is_acute_angle β ∧ is_acute_angle γ) ∨ 
  (is_acute_angle α ∧ is_right_angle β ∧ is_acute_angle γ) ∨ 
  (is_acute_angle α ∧ is_acute_angle β ∧ is_right_angle γ)

theorem right_triangle_counterexample (α β γ : ℝ) : 
  is_triangle α β γ → is_right_triangle α β γ → ¬ is_acute_triangle α β γ :=
by
  intro htri hrt hacute
  sorry

end NUMINAMATH_GPT_right_triangle_counterexample_l226_22644


namespace NUMINAMATH_GPT_irwins_family_hike_total_distance_l226_22669

theorem irwins_family_hike_total_distance
    (d1 d2 d3 : ℝ)
    (h1 : d1 = 0.2)
    (h2 : d2 = 0.4)
    (h3 : d3 = 0.1)
    :
    d1 + d2 + d3 = 0.7 :=
by
  rw [h1, h2, h3]
  norm_num
  done

end NUMINAMATH_GPT_irwins_family_hike_total_distance_l226_22669


namespace NUMINAMATH_GPT_gina_snake_mice_in_decade_l226_22696

-- Definitions based on the conditions in a)
def weeks_per_mouse : ℕ := 4
def weeks_per_year : ℕ := 52
def years_per_decade : ℕ := 10

-- The final statement to prove
theorem gina_snake_mice_in_decade : 
  (weeks_per_year / weeks_per_mouse) * years_per_decade = 130 :=
by
  sorry

end NUMINAMATH_GPT_gina_snake_mice_in_decade_l226_22696


namespace NUMINAMATH_GPT_volume_of_63_ounces_l226_22631

variable {V W : ℝ}
variable (k : ℝ)

def directly_proportional (V W : ℝ) (k : ℝ) : Prop :=
  V = k * W

theorem volume_of_63_ounces (h1 : directly_proportional 48 112 k)
                            (h2 : directly_proportional V 63 k) :
  V = 27 := by
  sorry

end NUMINAMATH_GPT_volume_of_63_ounces_l226_22631


namespace NUMINAMATH_GPT_march_volume_expression_l226_22619

variable (x : ℝ) (y : ℝ)

def initial_volume : ℝ := 500
def growth_rate_volumes (x : ℝ) (initial_volume : ℝ) : ℝ := initial_volume * (1 + x)
def calculate_march_volume (x : ℝ) (initial_volume : ℝ) : ℝ := initial_volume * (1 + x)^2

theorem march_volume_expression :
  y = calculate_march_volume x initial_volume :=
sorry

end NUMINAMATH_GPT_march_volume_expression_l226_22619


namespace NUMINAMATH_GPT_initial_books_gathered_l226_22681

-- Conditions
def total_books_now : Nat := 59
def books_found : Nat := 26

-- Proof problem
theorem initial_books_gathered : total_books_now - books_found = 33 :=
by
  sorry -- Proof to be provided later

end NUMINAMATH_GPT_initial_books_gathered_l226_22681


namespace NUMINAMATH_GPT_geometric_sequence_product_l226_22671

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (a_geometric : ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r)
  (root_condition : ∃ x y : ℝ, x * y = 16 ∧ x + y = 10 ∧ a 1 = x ∧ a 19 = y) :
  a 8 * a 10 * a 12 = 64 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_product_l226_22671


namespace NUMINAMATH_GPT_terminal_side_of_angle_y_eq_neg_one_l226_22618
/-
Given that the terminal side of angle θ lies on the line y = -x,
prove that y = -1 where y = sin θ / |sin θ| + |cos θ| / cos θ + tan θ / |tan θ|.
-/


noncomputable def y (θ : ℝ) : ℝ :=
  (Real.sin θ / |Real.sin θ|) + (|Real.cos θ| / Real.cos θ) + (Real.tan θ / |Real.tan θ|)

theorem terminal_side_of_angle_y_eq_neg_one (θ : ℝ) (k : ℤ) (h : θ = k * Real.pi - (Real.pi / 4)) :
  y θ = -1 :=
by
  sorry

end NUMINAMATH_GPT_terminal_side_of_angle_y_eq_neg_one_l226_22618


namespace NUMINAMATH_GPT_cannot_be_sum_of_four_consecutive_even_integers_l226_22611

-- Define what it means to be the sum of four consecutive even integers
def sum_of_four_consecutive_even_integers (n : ℤ) : Prop :=
  ∃ m : ℤ, n = 4 * m + 12 ∧ m % 2 = 0

-- State the problem in Lean 4
theorem cannot_be_sum_of_four_consecutive_even_integers :
  ¬ sum_of_four_consecutive_even_integers 32 ∧
  ¬ sum_of_four_consecutive_even_integers 80 ∧
  ¬ sum_of_four_consecutive_even_integers 104 ∧
  ¬ sum_of_four_consecutive_even_integers 200 :=
by
  sorry

end NUMINAMATH_GPT_cannot_be_sum_of_four_consecutive_even_integers_l226_22611


namespace NUMINAMATH_GPT_determine_d_minus_r_l226_22606

theorem determine_d_minus_r :
  ∃ d r: ℕ, (∀ n ∈ [2023, 2459, 3571], n % d = r) ∧ (1 < d) ∧ (d - r = 1) :=
sorry

end NUMINAMATH_GPT_determine_d_minus_r_l226_22606


namespace NUMINAMATH_GPT_find_A_l226_22622

theorem find_A (A B C : ℕ) (h1 : A ≠ B ∧ A ≠ C ∧ B ≠ C) 
  (h2 : 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9)
  (h3 : A * 10 + B + B * 10 + C = B * 100 + C * 10 + B) : 
  A = 9 :=
  sorry

end NUMINAMATH_GPT_find_A_l226_22622


namespace NUMINAMATH_GPT_sum_common_elements_ap_gp_l226_22602

noncomputable def sum_of_first_10_common_elements : ℕ := 20 * (4^10 - 1) / (4 - 1)

theorem sum_common_elements_ap_gp :
  sum_of_first_10_common_elements = 6990500 :=
by
  unfold sum_of_first_10_common_elements
  sorry

end NUMINAMATH_GPT_sum_common_elements_ap_gp_l226_22602


namespace NUMINAMATH_GPT_a_n_formula_T_n_formula_l226_22691

variable (a : Nat → Int) (b : Nat → Int)
variable (S : Nat → Int) (T : Nat → Int)
variable (d a_1 : Int)

-- Conditions:
axiom a_seq_arith : ∀ n, a (n + 1) = a n + d
axiom S_arith : ∀ n, S n = n * (a 1 + a n) / 2
axiom S_10 : S 10 = 110
axiom geo_seq : (a 2) ^ 2 = a 1 * a 4
axiom b_def : ∀ n, b n = 1 / ((a n - 1) * (a n + 1))

-- Goals: 
-- 1. Find the general formula for the terms of sequence {a_n}
theorem a_n_formula : ∀ n, a n = 2 * n := sorry

-- 2. Find the sum of the first n terms T_n of the sequence {b_n} given b_n
theorem T_n_formula : ∀ n, T n = 1 / 2 - 1 / (4 * n + 2) := sorry

end NUMINAMATH_GPT_a_n_formula_T_n_formula_l226_22691


namespace NUMINAMATH_GPT_max_tan_y_l226_22643

noncomputable def tan_y_upper_bound (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) 
    (h : Real.sin y = 2005 * Real.cos (x + y) * Real.sin x) : Real :=
  Real.tan y

theorem max_tan_y (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) 
    (h : Real.sin y = 2005 * Real.cos (x + y) * Real.sin x) : 
    tan_y_upper_bound x y hx hy h = 2005 * Real.sqrt 2006 / 4012 := 
by 
  sorry

end NUMINAMATH_GPT_max_tan_y_l226_22643


namespace NUMINAMATH_GPT_desired_annual_profit_is_30500000_l226_22665

noncomputable def annual_fixed_costs : ℝ := 50200000
noncomputable def average_cost_per_car : ℝ := 5000
noncomputable def number_of_cars : ℕ := 20000
noncomputable def selling_price_per_car : ℝ := 9035

noncomputable def total_revenue : ℝ :=
  selling_price_per_car * number_of_cars

noncomputable def total_variable_costs : ℝ :=
  average_cost_per_car * number_of_cars

noncomputable def total_costs : ℝ :=
  annual_fixed_costs + total_variable_costs

noncomputable def desired_annual_profit : ℝ :=
  total_revenue - total_costs

theorem desired_annual_profit_is_30500000:
  desired_annual_profit = 30500000 := by
  sorry

end NUMINAMATH_GPT_desired_annual_profit_is_30500000_l226_22665


namespace NUMINAMATH_GPT_problem_statement_l226_22663

variable (a b : ℝ)

theorem problem_statement (h1 : a > b) (h2 : b > 1/a) (h3 : 1/a > 0) :
  (a + b > 2) ∧ (a > 1) ∧ (a - 1/b > b - 1/a) :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l226_22663


namespace NUMINAMATH_GPT_ineq_pos_xy_l226_22647

theorem ineq_pos_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + y) / Real.sqrt (x * y) ≤ x / y + y / x := 
sorry

end NUMINAMATH_GPT_ineq_pos_xy_l226_22647


namespace NUMINAMATH_GPT_primes_div_conditions_unique_l226_22680

theorem primes_div_conditions_unique (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (p ∣ q + 6) ∧ (q ∣ p + 7) → (p = 19 ∧ q = 13) :=
sorry

end NUMINAMATH_GPT_primes_div_conditions_unique_l226_22680


namespace NUMINAMATH_GPT_find_factor_l226_22613

theorem find_factor (x f : ℕ) (hx : x = 110) (h : x * f - 220 = 110) : f = 3 :=
sorry

end NUMINAMATH_GPT_find_factor_l226_22613


namespace NUMINAMATH_GPT_geometric_sequence_seventh_term_l226_22649

theorem geometric_sequence_seventh_term (a r : ℝ) 
  (h4 : a * r^3 = 16) 
  (h9 : a * r^8 = 2) : 
  a * r^6 = 8 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_seventh_term_l226_22649


namespace NUMINAMATH_GPT_bus_distance_l226_22627

theorem bus_distance (w r : ℝ) (h1 : w = 0.17) (h2 : r = w + 3.67) : r = 3.84 :=
by
  sorry

end NUMINAMATH_GPT_bus_distance_l226_22627


namespace NUMINAMATH_GPT_product_of_irwins_baskets_l226_22615

theorem product_of_irwins_baskets 
  (baskets_scored : Nat)
  (point_value : Nat)
  (total_baskets : baskets_scored = 2)
  (value_per_basket : point_value = 11) : 
  point_value * baskets_scored = 22 := 
by 
  sorry

end NUMINAMATH_GPT_product_of_irwins_baskets_l226_22615


namespace NUMINAMATH_GPT_find_root_l226_22633

theorem find_root (y : ℝ) (h : y - 9 / (y - 4) = 2 - 9 / (y - 4)) : y = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_root_l226_22633


namespace NUMINAMATH_GPT_markus_more_marbles_than_mara_l226_22687

variable (mara_bags : Nat) (mara_marbles_per_bag : Nat)
variable (markus_bags : Nat) (markus_marbles_per_bag : Nat)

theorem markus_more_marbles_than_mara :
  mara_bags = 12 →
  mara_marbles_per_bag = 2 →
  markus_bags = 2 →
  markus_marbles_per_bag = 13 →
  (markus_bags * markus_marbles_per_bag) - (mara_bags * mara_marbles_per_bag) = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_markus_more_marbles_than_mara_l226_22687


namespace NUMINAMATH_GPT_sufficient_necessary_condition_l226_22678

theorem sufficient_necessary_condition (a : ℝ) :
  (∃ x : ℝ, 2 * x + 1 = a ∧ x > 2) ↔ a > 5 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_necessary_condition_l226_22678


namespace NUMINAMATH_GPT_min_value_f_l226_22679

noncomputable def f (x : ℝ) : ℝ := |x - 4| + |x - 6|

theorem min_value_f : ∃ x : ℝ, f x ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_f_l226_22679


namespace NUMINAMATH_GPT_find_missing_score_l226_22645

noncomputable def total_points (mean : ℝ) (games : ℕ) : ℝ :=
  mean * games

noncomputable def sum_of_scores (scores : List ℝ) : ℝ :=
  scores.sum

theorem find_missing_score
  (scores : List ℝ)
  (mean : ℝ)
  (games : ℕ)
  (total_points_value : ℝ)
  (sum_of_recorded_scores : ℝ)
  (missing_score : ℝ) :
  scores = [81, 73, 86, 73] →
  mean = 79.2 →
  games = 5 →
  total_points_value = total_points mean games →
  sum_of_recorded_scores = sum_of_scores scores →
  missing_score = total_points_value - sum_of_recorded_scores →
  missing_score = 83 :=
by
  intros
  exact sorry

end NUMINAMATH_GPT_find_missing_score_l226_22645


namespace NUMINAMATH_GPT_clinton_shoes_count_l226_22641

def num_hats : ℕ := 5
def num_belts : ℕ := num_hats + 2
def num_shoes : ℕ := 2 * num_belts

theorem clinton_shoes_count : num_shoes = 14 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_clinton_shoes_count_l226_22641


namespace NUMINAMATH_GPT_find_m_l226_22697

noncomputable def m_value (m : ℝ) := 
  ((m ^ 2) - m - 1, (m ^ 2) - 2 * m - 1)

theorem find_m (m : ℝ) (h1 : (m ^ 2) - m - 1 = 1) (h2 : (m ^ 2) - 2 * m - 1 < 0) : 
  m = 2 :=
by sorry

end NUMINAMATH_GPT_find_m_l226_22697


namespace NUMINAMATH_GPT_wilsons_theorem_l226_22667

theorem wilsons_theorem (p : ℕ) (hp : p ≥ 2) : Nat.Prime p ↔ (Nat.factorial (p - 1) + 1) % p = 0 := 
sorry

end NUMINAMATH_GPT_wilsons_theorem_l226_22667


namespace NUMINAMATH_GPT_sqrt_D_irrational_l226_22668

variable (k : ℤ)

def a := 3 * k
def b := 3 * k + 3
def c := a k + b k
def D := a k * a k + b k * b k + c k * c k

theorem sqrt_D_irrational : ¬ ∃ (r : ℚ), r * r = D k := 
by sorry

end NUMINAMATH_GPT_sqrt_D_irrational_l226_22668


namespace NUMINAMATH_GPT_neither_biology_nor_chemistry_l226_22635

def science_club_total : ℕ := 80
def biology_members : ℕ := 50
def chemistry_members : ℕ := 40
def both_members : ℕ := 25

theorem neither_biology_nor_chemistry :
  (science_club_total -
  ((biology_members - both_members) +
  (chemistry_members - both_members) +
  both_members)) = 15 := by
  sorry

end NUMINAMATH_GPT_neither_biology_nor_chemistry_l226_22635


namespace NUMINAMATH_GPT_find_m_l226_22636

variables (a b m : ℝ)

def f (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

def f' (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem find_m (h1 : f m = 0) (h2 : f' m = 0) (h3 : m ≠ 0)
    (h4 : ∃ x, f' x = 0 ∧ ∀ y, x ≤ y → f x ≥ f y ∧ f x = 1/2) :
    m = 3/2 :=
sorry

end NUMINAMATH_GPT_find_m_l226_22636


namespace NUMINAMATH_GPT_smallest_winning_N_and_digit_sum_l226_22658

-- Definitions of operations
def B (x : ℕ) : ℕ := 3 * x
def S (x : ℕ) : ℕ := x + 100

/-- The main theorem confirming the smallest winning number and sum of its digits -/
theorem smallest_winning_N_and_digit_sum :
  ∃ (N : ℕ), 0 ≤ N ∧ N ≤ 999 ∧ (900 ≤ 9 * N + 400 ∧ 9 * N + 400 < 1000) ∧ (N = 56) ∧ (5 + 6 = 11) :=
by {
  -- Proof skipped
  sorry
}

end NUMINAMATH_GPT_smallest_winning_N_and_digit_sum_l226_22658


namespace NUMINAMATH_GPT_sixth_graders_bought_more_pencils_23_l226_22676

open Int

-- Conditions
def pencils_cost_whole_number_cents : Prop := ∃ n : ℕ, n > 0
def seventh_graders_total_cents := 165
def sixth_graders_total_cents := 234
def number_of_sixth_graders := 30

-- The number of sixth graders who bought more pencils than seventh graders
theorem sixth_graders_bought_more_pencils_23 :
  (seventh_graders_total_cents / 3 = 55) ∧
  (sixth_graders_total_cents / 3 = 78) →
  78 - 55 = 23 :=
by
  sorry

end NUMINAMATH_GPT_sixth_graders_bought_more_pencils_23_l226_22676


namespace NUMINAMATH_GPT_least_pennies_l226_22639

theorem least_pennies : 
  ∃ (a : ℕ), a % 5 = 1 ∧ a % 3 = 2 ∧ a = 11 :=
by
  sorry

end NUMINAMATH_GPT_least_pennies_l226_22639


namespace NUMINAMATH_GPT_single_discount_eq_l226_22634

/--
A jacket is originally priced at $50. It is on sale for 25% off. After applying the sale discount, 
John uses a coupon that gives an additional 10% off of the discounted price. If there is a 5% sales 
tax on the final price, what single percent discount (before taxes) is equivalent to these series 
of discounts followed by the tax? --/
theorem single_discount_eq :
  let P0 := 50
  let discount1 := 0.25
  let discount2 := 0.10
  let tax := 0.05
  let discounted_price := P0 * (1 - discount1) * (1 - discount2)
  let after_tax_price := discounted_price * (1 + tax)
  let single_discount := (P0 - discounted_price) / P0
  single_discount * 100 = 32.5 :=
by
  sorry

end NUMINAMATH_GPT_single_discount_eq_l226_22634


namespace NUMINAMATH_GPT_pyramid_base_edge_length_l226_22600

theorem pyramid_base_edge_length
  (hemisphere_radius : ℝ) (pyramid_height : ℝ) (slant_height : ℝ) (is_tangent: Prop) :
  hemisphere_radius = 3 ∧ pyramid_height = 8 ∧ slant_height = 10 ∧ is_tangent →
  ∃ (base_edge_length : ℝ), base_edge_length = 6 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_base_edge_length_l226_22600


namespace NUMINAMATH_GPT_side_length_of_square_l226_22655

theorem side_length_of_square : 
  ∀ (L : ℝ), L = 28 → (L / 4) = 7 :=
by
  intro L h
  rw [h]
  norm_num

end NUMINAMATH_GPT_side_length_of_square_l226_22655


namespace NUMINAMATH_GPT_min_value_parabola_l226_22674

theorem min_value_parabola : 
  ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 4 ∧ (-x^2 + 4 * x - 2) = -2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_parabola_l226_22674


namespace NUMINAMATH_GPT_distance_light_in_50_years_l226_22646

/-- The distance light travels in one year, given in scientific notation -/
def distance_light_per_year : ℝ := 9.4608 * 10^12

/-- The distance light travels in 50 years is calculated -/
theorem distance_light_in_50_years :
  distance_light_per_year * 50 = 4.7304 * 10^14 :=
by
  -- the proof is not demanded, so we use sorry
  sorry

end NUMINAMATH_GPT_distance_light_in_50_years_l226_22646


namespace NUMINAMATH_GPT_no_real_solution_x_squared_minus_2x_plus_3_eq_zero_l226_22601

theorem no_real_solution_x_squared_minus_2x_plus_3_eq_zero :
  ∀ x : ℝ, x^2 - 2 * x + 3 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_x_squared_minus_2x_plus_3_eq_zero_l226_22601


namespace NUMINAMATH_GPT_sum_p_q_l226_22628

theorem sum_p_q (p q : ℚ) (g : ℚ → ℚ) (h : g = λ x => (x + 2) / (x^2 + p * x + q))
  (h_asymp1 : ∀ {x}, x = -1 → (x^2 + p * x + q) = 0)
  (h_asymp2 : ∀ {x}, x = 3 → (x^2 + p * x + q) = 0) :
  p + q = -5 := by
  sorry

end NUMINAMATH_GPT_sum_p_q_l226_22628


namespace NUMINAMATH_GPT_seminar_attendees_l226_22604

theorem seminar_attendees (a b c d attendees_not_from_companies : ℕ)
  (h1 : a = 30)
  (h2 : b = 2 * a)
  (h3 : c = a + 10)
  (h4 : d = c - 5)
  (h5 : attendees_not_from_companies = 20) :
  a + b + c + d + attendees_not_from_companies = 185 := by
  sorry

end NUMINAMATH_GPT_seminar_attendees_l226_22604


namespace NUMINAMATH_GPT_solve_system_l226_22695

theorem solve_system :
  ∃ x y : ℤ, (x - 3 * y = 7) ∧ (5 * x + 2 * y = 1) ∧ (x = 1) ∧ (y = -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l226_22695


namespace NUMINAMATH_GPT_rice_difference_l226_22657

theorem rice_difference:
  ∀ (total : ℕ) (kept_fraction : ℚ) (kept : ℕ) (given : ℕ),
  total = 50 →
  kept_fraction = (7 : ℚ) / 10 →
  kept = total * kept_fraction →
  given = total - kept →
  (kept - given = 20) :=
by
  intros total kept_fraction kept given
  intros h_total h_fraction h_kept h_given
  sorry

end NUMINAMATH_GPT_rice_difference_l226_22657


namespace NUMINAMATH_GPT_solution_set_of_inequality_l226_22653

theorem solution_set_of_inequality {x : ℝ} :
  {x : ℝ | |2 - 3 * x| ≥ 4} = {x : ℝ | x ≤ -2 / 3 ∨ 2 ≤ x} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l226_22653


namespace NUMINAMATH_GPT_complex_equation_l226_22654

theorem complex_equation (m n : ℝ) (i : ℂ)
  (hi : i^2 = -1)
  (h1 : m * (1 + i) = 1 + n * i) :
  ( (m + n * i) / (m - n * i) )^2 = -1 :=
sorry

end NUMINAMATH_GPT_complex_equation_l226_22654


namespace NUMINAMATH_GPT_inequality_solution_condition_necessary_but_not_sufficient_l226_22662

theorem inequality_solution_condition (a : ℝ) :
  (∃ x : ℝ, x^2 + 2 * a * x - a ≤ 0) ↔ (a ≥ 0 ∨ a ≤ -1) := sorry

theorem necessary_but_not_sufficient (a : ℝ) :
  (a > 0 ∨ a < -1) → (∃ x : ℝ, x^2 + 2 * a * x - a ≤ 0) ∧ ¬(∃ x : ℝ, x^2 + 2 * a * x - a ≤ 0 → (a > 0 ∨ a < -1)) := sorry

end NUMINAMATH_GPT_inequality_solution_condition_necessary_but_not_sufficient_l226_22662


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l226_22688

theorem geometric_series_common_ratio
    (a : ℝ) (S : ℝ) (r : ℝ)
    (h_a : a = 512)
    (h_S : S = 3072)
    (h_sum : S = a / (1 - r)) : 
    r = 5 / 6 :=
by 
  rw [h_a] at h_sum
  rw [h_S] at h_sum
  sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l226_22688


namespace NUMINAMATH_GPT_card_game_impossible_l226_22626

theorem card_game_impossible : 
  ∀ (students : ℕ) (initial_cards : ℕ) (cards_distribution : ℕ → ℕ), 
  students = 2018 → 
  initial_cards = 2018 →
  (∀ n, n < students → (if n = 0 then cards_distribution n = initial_cards else cards_distribution n = 0)) →
  (¬ ∃ final_distribution : ℕ → ℕ, (∀ n, n < students → final_distribution n = 1)) :=
by
  intros students initial_cards cards_distribution stu_eq init_eq init_dist final_dist
  -- Sorry can be used here as the proof is not required
  sorry

end NUMINAMATH_GPT_card_game_impossible_l226_22626


namespace NUMINAMATH_GPT_area_enclosed_by_region_l226_22612

open Real

def condition (x y : ℝ) := abs (2 * x + 2 * y) + abs (2 * x - 2 * y) ≤ 8

theorem area_enclosed_by_region : 
  (∃ u v : ℝ, condition u v) → ∃ A : ℝ, A = 16 := 
sorry

end NUMINAMATH_GPT_area_enclosed_by_region_l226_22612


namespace NUMINAMATH_GPT_trigonometric_identity_l226_22693

theorem trigonometric_identity :
  1 / Real.sin (70 * Real.pi / 180) - Real.sqrt 2 / Real.cos (70 * Real.pi / 180) = 
  -2 * (Real.sin (25 * Real.pi / 180) / Real.sin (40 * Real.pi / 180)) :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l226_22693


namespace NUMINAMATH_GPT_common_ratio_geometric_series_l226_22692

theorem common_ratio_geometric_series :
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  (b / a) = - (10 : ℚ) / 21 :=
by
  -- definitions
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  -- assertion
  have ratio := b / a
  sorry

end NUMINAMATH_GPT_common_ratio_geometric_series_l226_22692


namespace NUMINAMATH_GPT_quadrilateral_not_parallelogram_l226_22616

-- Definitions based on the given conditions
structure Quadrilateral :=
  (a b c d : ℝ) -- sides of the quadrilateral
  (parallel : Prop) -- one pair of parallel sides
  (equal_sides : Prop) -- another pair of equal sides

-- Problem statement
theorem quadrilateral_not_parallelogram (q : Quadrilateral) 
  (h1 : q.parallel) 
  (h2 : q.equal_sides) : 
  ¬ (∃ p : Quadrilateral, p = q) :=
sorry

end NUMINAMATH_GPT_quadrilateral_not_parallelogram_l226_22616


namespace NUMINAMATH_GPT_inequality_proof_l226_22637

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_cond : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l226_22637


namespace NUMINAMATH_GPT_points_on_single_circle_l226_22608

theorem points_on_single_circle (n : ℕ) (points : Fin n → ℝ × ℝ)
  (h : ∀ i j : Fin n, ∃ f : ℝ × ℝ → ℝ × ℝ, (∀ p, f p ≠ p) ∧ f (points i) = points j ∧ 
        (∀ k : Fin n, ∃ p, points k = f p)) :
  ∃ (O : ℝ × ℝ) (r : ℝ), ∀ i : Fin n, dist (points i) O = r := sorry

end NUMINAMATH_GPT_points_on_single_circle_l226_22608


namespace NUMINAMATH_GPT_cosine_double_angle_l226_22666

theorem cosine_double_angle (α : ℝ) (h : Real.sin α = 1 / 3) : Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cosine_double_angle_l226_22666


namespace NUMINAMATH_GPT_find_number_l226_22638

theorem find_number (x : ℚ) (h : 15 + 3 * x = 6 * x - 10) : x = 25 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l226_22638


namespace NUMINAMATH_GPT_determine_price_reduction_l226_22683

noncomputable def initial_cost_price : ℝ := 220
noncomputable def initial_selling_price : ℝ := 280
noncomputable def initial_daily_sales_volume : ℕ := 30
noncomputable def price_reduction_increase_rate : ℝ := 3

variable (x : ℝ)

noncomputable def daily_sales_volume (x : ℝ) : ℝ := initial_daily_sales_volume + price_reduction_increase_rate * x
noncomputable def profit_per_item (x : ℝ) : ℝ := (initial_selling_price - x) - initial_cost_price

theorem determine_price_reduction (x : ℝ) 
    (h1 : daily_sales_volume x = initial_daily_sales_volume + price_reduction_increase_rate * x)
    (h2 : profit_per_item x = 60 - x) : 
    (30 + 3 * x) * (60 - x) = 3600 → x = 30 :=
by 
  sorry

end NUMINAMATH_GPT_determine_price_reduction_l226_22683


namespace NUMINAMATH_GPT_diagonal_AC_length_l226_22694

theorem diagonal_AC_length (AB BC CD DA : ℝ) (angle_ADC : ℝ) (h_AB : AB = 12) (h_BC : BC = 12) 
(h_CD : CD = 13) (h_DA : DA = 13) (h_angle_ADC : angle_ADC = 60) : 
  AC = 13 := 
sorry

end NUMINAMATH_GPT_diagonal_AC_length_l226_22694


namespace NUMINAMATH_GPT_zero_point_condition_l226_22624

-- Define the function f(x) = ax + 3
def f (a x : ℝ) : ℝ := a * x + 3

-- Define that a > 2 is necessary but not sufficient condition
theorem zero_point_condition (a : ℝ) (h : a > 2) : 
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 0 → f a x = 0) ↔ (a ≥ 3) := 
sorry

end NUMINAMATH_GPT_zero_point_condition_l226_22624


namespace NUMINAMATH_GPT_complement_intersection_l226_22689

/-- Given the universal set U={1,2,3,4,5},
    A={2,3,4}, and B={1,2,3}, 
    Prove the complement of (A ∩ B) in U is {1,4,5}. -/
theorem complement_intersection 
    (U : Set ℕ) (A : Set ℕ) (B : Set ℕ) 
    (hU : U = {1, 2, 3, 4, 5})
    (hA : A = {2, 3, 4})
    (hB : B = {1, 2, 3}) :
    U \ (A ∩ B) = {1, 4, 5} :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_complement_intersection_l226_22689


namespace NUMINAMATH_GPT_inequality_problem_l226_22642

variable (a b c d : ℝ)

theorem inequality_problem (h1 : b < 0) (h2 : 0 < a) (h3 : d < c) (h4 : c < 0) : 
  a + c > b + d := sorry

end NUMINAMATH_GPT_inequality_problem_l226_22642


namespace NUMINAMATH_GPT_total_ticket_cost_l226_22670

theorem total_ticket_cost 
  (young_discount : ℝ := 0.55) 
  (old_discount : ℝ := 0.30) 
  (full_price : ℝ := 10)
  (num_young : ℕ := 2) 
  (num_middle : ℕ := 2) 
  (num_old : ℕ := 2) 
  (grandma_ticket_cost : ℝ := 7) :
  2 * (full_price * young_discount) + 2 * full_price + 2 * grandma_ticket_cost = 43 :=
by 
  sorry

end NUMINAMATH_GPT_total_ticket_cost_l226_22670
