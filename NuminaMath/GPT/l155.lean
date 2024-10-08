import Mathlib

namespace car_a_speed_l155_155557

theorem car_a_speed (d_gap : ℕ) (v_B : ℕ) (t : ℕ) (d_ahead : ℕ) (v_A : ℕ) 
  (h1 : d_gap = 24) (h2 : v_B = 50) (h3 : t = 4) (h4 : d_ahead = 8)
  (h5 : v_A = (d_gap + v_B * t + d_ahead) / t) : v_A = 58 :=
by {
  exact (sorry : v_A = 58)
}

end car_a_speed_l155_155557


namespace geometric_series_sum_l155_155778

theorem geometric_series_sum :
  let a := (1 / 2 : ℝ)
  let r := (1 / 2 : ℝ)
  let n := 6
  (a * (1 - r^n) / (1 - r)) = (63 / 64 : ℝ) := 
by 
  sorry

end geometric_series_sum_l155_155778


namespace sufficient_but_not_necessary_l155_155209

-- Definitions for lines and planes
def line : Type := ℝ × ℝ × ℝ
def plane : Type := ℝ × ℝ × ℝ × ℝ

-- Predicate for perpendicularity of a line to a plane
def perp_to_plane (l : line) (α : plane) : Prop := sorry

-- Predicate for parallelism of two planes
def parallel_planes (α β : plane) : Prop := sorry

-- Predicate for perpendicularity of two lines
def perp_lines (l m : line) : Prop := sorry

-- Predicate for a line being parallel to a plane
def parallel_to_plane (m : line) (β : plane) : Prop := sorry

-- Given conditions
variable (l : line)
variable (m : line)
variable (alpha : plane)
variable (beta : plane)
variable (H1 : perp_to_plane l alpha) -- l ⊥ α
variable (H2 : parallel_to_plane m beta) -- m ∥ β

-- Theorem statement
theorem sufficient_but_not_necessary :
  (parallel_planes alpha beta → perp_lines l m) ∧ ¬(perp_lines l m → parallel_planes alpha beta) :=
sorry

end sufficient_but_not_necessary_l155_155209


namespace common_tangent_at_point_l155_155777

theorem common_tangent_at_point (x₀ b : ℝ) 
  (h₁ : 6 * x₀^2 = 6 * x₀) 
  (h₂ : 1 + 2 * x₀^3 = 3 * x₀^2 - b) :
  b = 0 ∨ b = -1 :=
sorry

end common_tangent_at_point_l155_155777


namespace variance_of_binomial_distribution_l155_155388

def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem variance_of_binomial_distribution :
  binomial_variance 10 (2/5) = 12 / 5 :=
by
  sorry

end variance_of_binomial_distribution_l155_155388


namespace semiperimeter_inequality_l155_155637

theorem semiperimeter_inequality (p R r : ℝ) (hp : p ≥ 0) (hR : R ≥ 0) (hr : r ≥ 0) :
  p ≥ (3 / 2) * Real.sqrt (6 * R * r) :=
sorry

end semiperimeter_inequality_l155_155637


namespace cookie_sheet_perimeter_l155_155965

theorem cookie_sheet_perimeter :
  let width_in_inches := 15.2
  let length_in_inches := 3.7
  let conversion_factor := 2.54
  let width_in_cm := width_in_inches * conversion_factor
  let length_in_cm := length_in_inches * conversion_factor
  2 * (width_in_cm + length_in_cm) = 96.012 :=
by
  sorry

end cookie_sheet_perimeter_l155_155965


namespace consecutive_numbers_expression_l155_155171

theorem consecutive_numbers_expression (x y z : ℤ) (h1 : x = y + 1) (h2 : z = y - 1) (h3 : z = 2) :
  2 * x + 3 * y + 3 * z = 8 * y - 1 :=
by
  -- substitute the conditions and simplify
  sorry

end consecutive_numbers_expression_l155_155171


namespace distance_equal_axes_l155_155919

theorem distance_equal_axes (m : ℝ) :
  (abs (3 * m + 1) = abs (2 * m - 5)) ↔ (m = -6 ∨ m = 4 / 5) :=
by 
  sorry

end distance_equal_axes_l155_155919


namespace average_first_20_multiples_of_17_l155_155092

theorem average_first_20_multiples_of_17 :
  (20 / 2 : ℝ) * (17 + 17 * 20) / 20 = 178.5 := by
  sorry

end average_first_20_multiples_of_17_l155_155092


namespace flag_arrangement_remainder_l155_155031

theorem flag_arrangement_remainder :
  let num_blue := 12
  let num_green := 10
  let div := 1000
  let M := 13 * Nat.choose (num_blue + 2) num_green - 2 * Nat.choose (num_blue + 1) num_green
  M % div = 441 := 
by
  -- Definitions
  let num_blue := 12
  let num_green := 10
  let div := 1000
  let M := 13 * Nat.choose (num_blue + 2) num_green - 2 * Nat.choose (num_blue + 1) num_green
  -- Proof
  sorry

end flag_arrangement_remainder_l155_155031


namespace find_n_between_50_and_150_l155_155840

theorem find_n_between_50_and_150 :
  ∃ (n : ℤ), 50 ≤ n ∧ n ≤ 150 ∧
  n % 7 = 0 ∧ 
  n % 9 = 3 ∧ 
  n % 6 = 3 ∧ 
  n % 4 = 1 ∧
  n = 105 :=
by
  sorry

end find_n_between_50_and_150_l155_155840


namespace curve_crosses_itself_at_point_l155_155274

theorem curve_crosses_itself_at_point :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
  (2 * t₁^2 + 1 = 2 * t₂^2 + 1) ∧ 
  (2 * t₁^3 - 6 * t₁^2 + 8 = 2 * t₂^3 - 6 * t₂^2 + 8) ∧ 
  2 * t₁^2 + 1 = 1 ∧ 2 * t₁^3 - 6 * t₁^2 + 8 = 8 :=
by
  sorry

end curve_crosses_itself_at_point_l155_155274


namespace simple_interest_rate_l155_155734

theorem simple_interest_rate (P R T A : ℝ) (h_double: A = 2 * P) (h_si: A = P + P * R * T / 100) (h_T: T = 5) : R = 20 :=
by
  have h1: A = 2 * P := h_double
  have h2: A = P + P * R * T / 100 := h_si
  have h3: T = 5 := h_T
  sorry

end simple_interest_rate_l155_155734


namespace part1_part2_l155_155498

noncomputable def f (a : ℝ) (x : ℝ) := (a - 1/2) * x^2 + Real.log x
noncomputable def g (a : ℝ) (x : ℝ) := f a x - 2 * a * x

theorem part1 (x : ℝ) (hxe : Real.exp (-1) ≤ x ∧ x ≤ Real.exp (1)) : 
    f (-1/2) x ≤ -1/2 - 1/2 * Real.log 2 ∧ f (-1/2) x ≥ 1 - Real.exp 2 := sorry

theorem part2 (h : ∀ x > 2, g a x < 0) : a ≤ 1/2 := sorry

end part1_part2_l155_155498


namespace correct_option_l155_155000

def option_A_1 : ℤ := (-2) ^ 2
def option_A_2 : ℤ := -(2 ^ 2)
def option_B_1 : ℤ := (|-2|) ^ 2
def option_B_2 : ℤ := -(2 ^ 2)
def option_C_1 : ℤ := (-2) ^ 3
def option_C_2 : ℤ := -(2 ^ 3)
def option_D_1 : ℤ := (|-2|) ^ 3
def option_D_2 : ℤ := -(2 ^ 3)

theorem correct_option : option_C_1 = option_C_2 ∧ 
  (option_A_1 ≠ option_A_2) ∧ 
  (option_B_1 ≠ option_B_2) ∧ 
  (option_D_1 ≠ option_D_2) :=
by
  sorry

end correct_option_l155_155000


namespace solution_y_chemical_A_percentage_l155_155098

def percent_chemical_A_in_x : ℝ := 0.30
def percent_chemical_A_in_mixture : ℝ := 0.32
def percent_solution_x_in_mixture : ℝ := 0.80
def percent_solution_y_in_mixture : ℝ := 0.20

theorem solution_y_chemical_A_percentage
  (P : ℝ) 
  (h : percent_solution_x_in_mixture * percent_chemical_A_in_x + percent_solution_y_in_mixture * P = percent_chemical_A_in_mixture) :
  P = 0.40 :=
sorry

end solution_y_chemical_A_percentage_l155_155098


namespace fresh_water_needed_l155_155950

noncomputable def mass_of_seawater : ℝ := 30
noncomputable def initial_salt_concentration : ℝ := 0.05
noncomputable def desired_salt_concentration : ℝ := 0.015

theorem fresh_water_needed :
  ∃ (fresh_water_mass : ℝ), 
    fresh_water_mass = 70 ∧ 
    (mass_of_seawater * initial_salt_concentration) / (mass_of_seawater + fresh_water_mass) = desired_salt_concentration :=
by
  sorry

end fresh_water_needed_l155_155950


namespace find_r_l155_155261

theorem find_r (r : ℝ) (h_curve : r = -2 * r^2 + 5 * r - 2) : r = 1 :=
sorry

end find_r_l155_155261


namespace consecutive_days_probability_l155_155693

noncomputable def probability_of_consecutive_days : ℚ :=
  let total_days := 5
  let combinations := Nat.choose total_days 2
  let consecutive_pairs := 4
  consecutive_pairs / combinations

theorem consecutive_days_probability :
  probability_of_consecutive_days = 2 / 5 :=
by
  sorry

end consecutive_days_probability_l155_155693


namespace range_of_function_x_geq_0_l155_155402

theorem range_of_function_x_geq_0 :
  ∀ (x : ℝ), x ≥ 0 → ∃ (y : ℝ), y ≥ 3 ∧ (y = x^2 + 2 * x + 3) :=
by
  sorry

end range_of_function_x_geq_0_l155_155402


namespace total_sections_l155_155453

theorem total_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 288) : 
  (boys / Nat.gcd boys girls) + (girls / Nat.gcd boys girls) = 29 :=
by
  sorry

end total_sections_l155_155453


namespace simplify_expression_l155_155922

variable (x y : ℝ)

theorem simplify_expression : 2 * x^2 * y - 4 * x * y^2 - (-3 * x * y^2 + x^2 * y) = x^2 * y - x * y^2 :=
by
  sorry

end simplify_expression_l155_155922


namespace minimum_value_l155_155118

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem minimum_value (a m n : ℝ)
    (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1)
    (h_a_on_graph : ∀ x, log_a a (x + 3) - 1 = 0 → x = -2)
    (h_on_line : 2 * m + n = 2)
    (h_mn_pos : m * n > 0) :
    (1 / m) + (2 / n) = 4 :=
by
  sorry

end minimum_value_l155_155118


namespace tangent_sum_formula_application_l155_155597

-- Define the problem's parameters and statement
noncomputable def thirty_three_degrees_radian := Real.pi * 33 / 180
noncomputable def seventeen_degrees_radian := Real.pi * 17 / 180
noncomputable def twenty_eight_degrees_radian := Real.pi * 28 / 180

theorem tangent_sum_formula_application :
  Real.tan seventeen_degrees_radian + Real.tan twenty_eight_degrees_radian + Real.tan seventeen_degrees_radian * Real.tan twenty_eight_degrees_radian = 1 := 
sorry

end tangent_sum_formula_application_l155_155597


namespace sum_of_n_with_unformable_postage_120_equals_43_l155_155869

theorem sum_of_n_with_unformable_postage_120_equals_43 :
  ∃ n1 n2 : ℕ, n1 = 21 ∧ n2 = 22 ∧ 
  (∀ k : ℕ, k > 120 → ∃ a b c : ℕ, k = 7 * a + n1 * b + (n1 + 1) * c) ∧ 
  (∀ k : ℕ, k > 120 → ∃ a b c : ℕ, k = 7 * a + n2 * b + (n2 + 1) * c) ∧ 
  (120 = 7 * a + n1 * b + (n1 + 1) * c → a = 0 ∧ b = 0 ∧ c = 0) ∧
  (120 = 7 * a + n2 * b + (n2 + 1) * c → a = 0 ∧ b = 0 ∧ c = 0) ∧
  (n1 + n2 = 43) :=
by
  sorry

end sum_of_n_with_unformable_postage_120_equals_43_l155_155869


namespace balloons_in_each_bag_of_round_balloons_l155_155589

variable (x : ℕ)

-- Definitions based on the problem's conditions
def totalRoundBalloonsBought := 5 * x
def totalLongBalloonsBought := 4 * 30
def remainingRoundBalloons := totalRoundBalloonsBought x - 5
def totalRemainingBalloons := remainingRoundBalloons x + totalLongBalloonsBought

-- Theorem statement based on the question and derived from the conditions and correct answer
theorem balloons_in_each_bag_of_round_balloons : totalRemainingBalloons x = 215 → x = 20 := by
  -- We acknowledge that the proof steps will follow here (omitted as per instructions)
  sorry

end balloons_in_each_bag_of_round_balloons_l155_155589


namespace max_marks_l155_155363

theorem max_marks (marks_secured : ℝ) (percentage : ℝ) (max_marks : ℝ) 
  (h1 : marks_secured = 332) 
  (h2 : percentage = 83) 
  (h3 : percentage = (marks_secured / max_marks) * 100) 
  : max_marks = 400 :=
by
  sorry

end max_marks_l155_155363


namespace kostyas_table_prime_l155_155619

theorem kostyas_table_prime (n : ℕ) (h₁ : n > 3) 
    (h₂ : ¬ ∃ r s : ℕ, r ≥ 3 ∧ s ≥ 3 ∧ n = r * s - (r + s)) : 
    Prime (n + 1) := 
sorry

end kostyas_table_prime_l155_155619


namespace number_of_friends_shared_with_l155_155726

-- Conditions and given data
def doughnuts_samuel : ℕ := 2 * 12
def doughnuts_cathy : ℕ := 3 * 12
def total_doughnuts : ℕ := doughnuts_samuel + doughnuts_cathy
def each_person_doughnuts : ℕ := 6
def total_people := total_doughnuts / each_person_doughnuts
def samuel_and_cathy : ℕ := 2

-- Statement to prove - Number of friends they shared with
theorem number_of_friends_shared_with : (total_people - samuel_and_cathy) = 8 := by
  sorry

end number_of_friends_shared_with_l155_155726


namespace mother_daughter_ages_l155_155427

theorem mother_daughter_ages :
  ∃ (x y : ℕ), (y = x + 22) ∧ (2 * x = (x + 22) - x) ∧ (x = 11) ∧ (y = 33) :=
by
  sorry

end mother_daughter_ages_l155_155427


namespace line_equation_l155_155908

theorem line_equation (P : ℝ × ℝ) (slope : ℝ) (hP : P = (-2, 0)) (hSlope : slope = 3) :
    ∃ (a b : ℝ), ∀ x y : ℝ, y = a * x + b ↔ P.1 = -2 ∧ P.2 = 0 ∧ slope = 3 ∧ y = 3 * x + 6 :=
by
  sorry

end line_equation_l155_155908


namespace smallest_x_l155_155521

theorem smallest_x (x : ℝ) (h : |4 * x + 12| = 40) : x = -13 :=
sorry

end smallest_x_l155_155521


namespace no_real_solutions_for_g_g_x_l155_155621

theorem no_real_solutions_for_g_g_x (d : ℝ) :
  ¬ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 4 * x1 + d)^2 + 4 * (x1^2 + 4 * x1 + d) + d = 0 ∧
                                (x2^2 + 4 * x2 + d)^2 + 4 * (x2^2 + 4 * x2 + d) + d = 0 :=
by
  sorry

end no_real_solutions_for_g_g_x_l155_155621


namespace mnp_sum_correct_l155_155889

noncomputable def mnp_sum : ℕ :=
  let m := 1032
  let n := 40
  let p := 3
  m + n + p

theorem mnp_sum_correct : mnp_sum = 1075 := by
  -- Given the conditions, the established value for m, n, and p should sum to 1075
  sorry

end mnp_sum_correct_l155_155889


namespace jan_skips_in_5_minutes_l155_155985

theorem jan_skips_in_5_minutes 
  (original_speed : ℕ)
  (time_in_minutes : ℕ)
  (doubled : ℕ)
  (new_speed : ℕ)
  (skips_in_5_minutes : ℕ) : 
  original_speed = 70 →
  doubled = 2 →
  new_speed = original_speed * doubled →
  time_in_minutes = 5 →
  skips_in_5_minutes = new_speed * time_in_minutes →
  skips_in_5_minutes = 700 :=
by
  intros 
  sorry

end jan_skips_in_5_minutes_l155_155985


namespace find_breadth_of_rectangle_l155_155936

theorem find_breadth_of_rectangle
  (L R S : ℝ)
  (h1 : L = (2 / 5) * R)
  (h2 : R = S)
  (h3 : S ^ 2 = 625)
  (A : ℝ := 100)
  (h4 : A = L * B) :
  B = 10 := sorry

end find_breadth_of_rectangle_l155_155936


namespace unique_pair_exists_l155_155896

theorem unique_pair_exists (n : ℕ) (hn : n > 0) : 
  ∃! (k l : ℕ), n = k * (k - 1) / 2 + l ∧ 0 ≤ l ∧ l < k :=
sorry

end unique_pair_exists_l155_155896


namespace prob_geometry_given_algebra_l155_155858

variable (algebra geometry : ℕ) (total : ℕ)

/-- Proof of the probability of selecting a geometry question on the second draw,
    given that an algebra question is selected on the first draw. -/
theorem prob_geometry_given_algebra : 
  algebra = 3 ∧ geometry = 2 ∧ total = 5 →
  (algebra / (total : ℚ)) * (geometry / (total - 1 : ℚ)) = 1 / 2 :=
by
  intro h
  sorry

end prob_geometry_given_algebra_l155_155858


namespace volume_of_circumscribed_polyhedron_l155_155337

theorem volume_of_circumscribed_polyhedron (R : ℝ) (V : ℝ) (S_n : ℝ) (h : Π (F_i : ℝ), V = (1/3) * S_n * R) : V = (1/3) * S_n * R :=
sorry

end volume_of_circumscribed_polyhedron_l155_155337


namespace find_expression_for_a_n_l155_155030

noncomputable def a_sequence (a : ℕ → ℕ) :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 2^n

theorem find_expression_for_a_n (a : ℕ → ℕ) (h : a_sequence a) (initial : a 1 = 1) :
  ∀ n : ℕ, a (n + 1) = 2^(n + 1) - 1 :=
by
  sorry

end find_expression_for_a_n_l155_155030


namespace system1_solution_system2_solution_l155_155170

theorem system1_solution :
  ∃ x y : ℝ, 3 * x + 4 * y = 16 ∧ 5 * x - 8 * y = 34 ∧ x = 6 ∧ y = -1/2 :=
by
  sorry

theorem system2_solution :
  ∃ x y : ℝ, (x - 1) / 2 + (y + 1) / 3 = 1 ∧ x + y = 4 ∧ x = -1 ∧ y = 5 :=
by
  sorry

end system1_solution_system2_solution_l155_155170


namespace min_pairs_l155_155305

-- Define the types for knights and liars
inductive Residents
| Knight : Residents
| Liar : Residents

def total_residents : ℕ := 200
def knights : ℕ := 100
def liars : ℕ := 100

-- Additional conditions
def conditions (friend_claims_knights friend_claims_liars : ℕ) : Prop :=
  friend_claims_knights = 100 ∧
  friend_claims_liars = 100 ∧
  knights + liars = total_residents

-- Minimum number of knight-liar pairs to prove
def min_knight_liar_pairs : ℕ := 50

theorem min_pairs {friend_claims_knights friend_claims_liars : ℕ} (h : conditions friend_claims_knights friend_claims_liars) :
    min_knight_liar_pairs = 50 :=
sorry

end min_pairs_l155_155305


namespace P_intersection_Q_is_singleton_l155_155332

theorem P_intersection_Q_is_singleton :
  {p : ℝ × ℝ | p.1 + p.2 = 3} ∩ {p : ℝ × ℝ | p.1 - p.2 = 5} = { (4, -1) } :=
by
  -- The proof steps would go here.
  sorry

end P_intersection_Q_is_singleton_l155_155332


namespace denis_fourth_board_score_l155_155577

theorem denis_fourth_board_score :
  ∀ (darts_per_board points_first_board points_second_board points_third_board points_total_boards : ℕ),
    darts_per_board = 3 →
    points_first_board = 30 →
    points_second_board = 38 →
    points_third_board = 41 →
    points_total_boards = (points_first_board + points_second_board + points_third_board) / 2 →
    points_total_boards = 34 :=
by
  intros darts_per_board points_first_board points_second_board points_third_board points_total_boards h1 h2 h3 h4 h5
  sorry

end denis_fourth_board_score_l155_155577


namespace smallest_sum_of_big_in_circle_l155_155061

theorem smallest_sum_of_big_in_circle (arranged_circle : Fin 8 → ℕ) (h_circle : ∀ n, arranged_circle n ∈ Finset.range (9) ∧ arranged_circle n > 0) :
  (∀ n, (arranged_circle n > arranged_circle (n + 1) % 8 ∧ arranged_circle n > arranged_circle (n + 7) % 8) ∨ (arranged_circle n < arranged_circle (n + 1) % 8 ∧ arranged_circle n < arranged_circle (n + 7) % 8)) →
  ∃ big_indices : Finset (Fin 8), big_indices.card = 4 ∧ big_indices.sum arranged_circle = 23 :=
by
  sorry

end smallest_sum_of_big_in_circle_l155_155061


namespace find_max_m_l155_155901

-- We define real numbers a, b, c that satisfy the given conditions
variable (a b c m : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_sum : a + b + c = 12)
variable (h_prod_sum : a * b + b * c + c * a = 30)
variable (m_def : m = min (a * b) (min (b * c) (c * a)))

-- We state the main theorem to be proved
theorem find_max_m : m ≤ 2 :=
by
  sorry

end find_max_m_l155_155901


namespace sqrt_23_parts_xy_diff_l155_155401

-- Problem 1: Integer and decimal parts of sqrt(23)
theorem sqrt_23_parts : ∃ (integer_part : ℕ) (decimal_part : ℝ), 
  integer_part = 4 ∧ decimal_part = Real.sqrt 23 - 4 ∧
  (integer_part : ℝ) + decimal_part = Real.sqrt 23 :=
by
  sorry

-- Problem 2: x - y for 9 + sqrt(3) = x + y with given conditions
theorem xy_diff : 
  ∀ (x y : ℝ), x = 10 → y = Real.sqrt 3 - 1 → x - y = 11 - Real.sqrt 3 :=
by
  sorry

end sqrt_23_parts_xy_diff_l155_155401


namespace drink_exactly_five_bottles_last_day_l155_155696

/-- 
Robin bought 617 bottles of water and needs to purchase 4 additional bottles on the last day 
to meet her daily water intake goal. 
Prove that Robin will drink exactly 5 bottles on the last day.
-/
theorem drink_exactly_five_bottles_last_day : 
  ∀ (bottles_bought : ℕ) (extra_bottles : ℕ), bottles_bought = 617 → extra_bottles = 4 → 
  ∃ x : ℕ, 621 = x * 617 + 4 ∧ x + 4 = 5 :=
by
  intros bottles_bought extra_bottles bottles_bought_eq extra_bottles_eq
  -- The proof would follow here
  sorry

end drink_exactly_five_bottles_last_day_l155_155696


namespace man_speed_l155_155140

theorem man_speed (distance_meters time_minutes : ℝ) (h_distance : distance_meters = 1250) (h_time : time_minutes = 15) :
  (distance_meters / 1000) / (time_minutes / 60) = 5 :=
by
  sorry

end man_speed_l155_155140


namespace price_of_AC_l155_155709

theorem price_of_AC (x : ℝ) (price_car price_ac : ℝ)
  (h1 : price_car = 3 * x) 
  (h2 : price_ac = 2 * x) 
  (h3 : price_car = price_ac + 500) : 
  price_ac = 1000 := sorry

end price_of_AC_l155_155709


namespace sufficient_condition_ab_greater_than_1_l155_155434

theorem sufficient_condition_ab_greater_than_1 (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) : ab > 1 := 
  sorry

end sufficient_condition_ab_greater_than_1_l155_155434


namespace gcd_of_given_lcm_and_ratio_l155_155714

theorem gcd_of_given_lcm_and_ratio (C D : ℕ) (h1 : Nat.lcm C D = 200) (h2 : C * 5 = D * 2) : Nat.gcd C D = 5 :=
sorry

end gcd_of_given_lcm_and_ratio_l155_155714


namespace coin_toss_probability_l155_155776

-- Define the sample space of the coin toss
inductive Coin
| heads : Coin
| tails : Coin

-- Define the probability function
def probability (outcome : Coin) : ℝ :=
  match outcome with
  | Coin.heads => 0.5
  | Coin.tails => 0.5

-- The theorem to be proved: In a fair coin toss, the probability of getting "heads" or "tails" is 0.5
theorem coin_toss_probability (outcome : Coin) : probability outcome = 0.5 :=
sorry

end coin_toss_probability_l155_155776


namespace k5_possibility_l155_155966

noncomputable def possible_k5 : Prop :=
  ∃ (intersections : Fin 5 → Fin 5 × Fin 10), 
    ∀ i j : Fin 5, i ≠ j → intersections i ≠ intersections j

theorem k5_possibility : possible_k5 := 
by
  sorry

end k5_possibility_l155_155966


namespace fill_tank_time_l155_155781

theorem fill_tank_time (hA : ∀ t : Real, t > 0 → (t / 10) = 1) 
                       (hB : ∀ t : Real, t > 0 → (t / 20) = 1) 
                       (hC : ∀ t : Real, t > 0 → (t / 30) = 1) : 
                       (60 / 7 : Real) = 60 / 7 :=
by
    sorry

end fill_tank_time_l155_155781


namespace abs_iff_neg_one_lt_x_lt_one_l155_155804

theorem abs_iff_neg_one_lt_x_lt_one (x : ℝ) : |x| < 1 ↔ -1 < x ∧ x < 1 :=
by
  sorry

end abs_iff_neg_one_lt_x_lt_one_l155_155804


namespace kids_on_Monday_l155_155450

-- Defining the conditions
def kidsOnTuesday : ℕ := 10
def difference : ℕ := 8

-- Formulating the theorem to prove the number of kids Julia played with on Monday
theorem kids_on_Monday : kidsOnTuesday + difference = 18 := by
  sorry

end kids_on_Monday_l155_155450


namespace binomial_log_inequality_l155_155660

theorem binomial_log_inequality (n : ℤ) :
  n * Real.log 2 ≤ Real.log (Nat.choose (2 * n.natAbs) n.natAbs) ∧ 
  Real.log (Nat.choose (2 * n.natAbs) n.natAbs) ≤ n * Real.log 4 :=
by sorry

end binomial_log_inequality_l155_155660


namespace seats_with_middle_empty_l155_155571

-- Define the parameters
def chairs := 5
def people := 4
def middle_empty := 3

-- Define the function to calculate seating arrangements
def number_of_ways (people : ℕ) (chairs : ℕ) (middle_empty : ℕ) : ℕ := 
  if chairs < people + 1 then 0
  else (chairs - 1) * (chairs - 2) * (chairs - 3) * (chairs - 4)

-- The theorem to prove the number of ways given the conditions
theorem seats_with_middle_empty : number_of_ways 4 5 3 = 24 := by
  sorry

end seats_with_middle_empty_l155_155571


namespace remaining_half_speed_l155_155014

-- Define the given conditions
def total_time : ℕ := 11
def first_half_distance : ℕ := 150
def first_half_speed : ℕ := 30
def total_distance : ℕ := 300

-- Prove the speed for the remaining half of the distance
theorem remaining_half_speed :
  ∃ v : ℕ, v = 25 ∧
  (total_distance = 2 * first_half_distance) ∧
  (first_half_distance / first_half_speed = 5) ∧
  (total_time = 5 + (first_half_distance / v)) :=
by
  -- Proof omitted
  sorry

end remaining_half_speed_l155_155014


namespace g_neg_one_add_g_one_l155_155708

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f (x - y) = f x * g y - f y * g x
axiom f_one_ne_zero : f 1 ≠ 0
axiom f_one_eq_f_two : f 1 = f 2

theorem g_neg_one_add_g_one : g (-1) + g 1 = 1 := by
  sorry

end g_neg_one_add_g_one_l155_155708


namespace BothNormal_l155_155208

variable (Normal : Type) (Person : Type) (MrA MrsA : Person)
variables (isNormal : Person → Prop)

-- Conditions given in the problem
axiom MrA_statement : ∀ p : Person, p = MrsA → isNormal MrA → isNormal MrsA
axiom MrsA_statement : ∀ p : Person, p = MrA → isNormal MrsA → isNormal MrA

-- Question (translated to proof problem): 
-- prove that Mr. A and Mrs. A are both normal persons
theorem BothNormal : isNormal MrA ∧ isNormal MrsA := 
  by 
    sorry -- proof is omitted

end BothNormal_l155_155208


namespace right_triangle_max_value_l155_155019

theorem right_triangle_max_value (a b c : ℝ) (h : a^2 + b^2 = c^2) :
    (a + b) / (ab / c) ≤ 2 * Real.sqrt 2 := sorry

end right_triangle_max_value_l155_155019


namespace range_of_m_l155_155248

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - m| > 4) ↔ m > 3 ∨ m < -5 := 
sorry

end range_of_m_l155_155248


namespace solve_for_x_l155_155164

theorem solve_for_x (x : ℝ) (h : 1 / 4 - 1 / 6 = 4 / x) : x = 48 := 
sorry

end solve_for_x_l155_155164


namespace subset_m_values_l155_155667

theorem subset_m_values
  {A B : Set ℝ}
  (hA : A = { x | x^2 + x - 6 = 0 })
  (hB : ∃ m, B = { x | m * x + 1 = 0 })
  (h_subset : ∀ {x}, x ∈ B → x ∈ A) :
  (∃ m, m = -1/2 ∨ m = 0 ∨ m = 1/3) :=
sorry

end subset_m_values_l155_155667


namespace zero_in_set_l155_155964

theorem zero_in_set : 0 ∈ ({0, 1, 2} : Set Nat) := 
sorry

end zero_in_set_l155_155964


namespace sum_of_roots_l155_155074

theorem sum_of_roots (x : ℝ) :
  (3 * x - 2) * (x - 3) + (3 * x - 2) * (2 * x - 8) = 0 ->
  x = 2 / 3 ∨ x = 11 / 3 ->
  (2 / 3) + (11 / 3) = 13 / 3 :=
by
  sorry

end sum_of_roots_l155_155074


namespace card_draw_sequential_same_suit_l155_155055

theorem card_draw_sequential_same_suit : 
  let hearts := 13
  let diamonds := 13
  let total_suits := hearts + diamonds
  ∃ ways : ℕ, ways = total_suits * (hearts - 1) :=
by
  sorry

end card_draw_sequential_same_suit_l155_155055


namespace exists_odd_a_b_and_positive_k_l155_155071

theorem exists_odd_a_b_and_positive_k (m : ℤ) :
  ∃ (a b : ℤ) (k : ℕ), a % 2 = 1 ∧ b % 2 = 1 ∧ k > 0 ∧ 2 * m = a^5 + b^5 + k * 2^100 := 
sorry

end exists_odd_a_b_and_positive_k_l155_155071


namespace triangle_possible_side_lengths_l155_155276

theorem triangle_possible_side_lengths (x : ℕ) (hx : x > 0) (h1 : x^2 + 9 > 12) (h2 : x^2 + 12 > 9) (h3 : 9 + 12 > x^2) : x = 2 ∨ x = 3 ∨ x = 4 :=
by
  sorry

end triangle_possible_side_lengths_l155_155276


namespace smallest_positive_debt_resolved_l155_155410

theorem smallest_positive_debt_resolved : ∃ (D : ℕ), D > 0 ∧ (∃ (p g : ℤ), D = 400 * p + 240 * g) ∧ D = 80 := by
  sorry

end smallest_positive_debt_resolved_l155_155410


namespace ice_cream_stacks_l155_155021

theorem ice_cream_stacks :
  let ice_cream_flavors := ["vanilla", "chocolate", "strawberry", "cherry", "banana"]
  let ways_to_stack := Nat.factorial ice_cream_flavors.length
  ways_to_stack = 120 :=
by
  let ice_cream_flavors := ["vanilla", "chocolate", "strawberry", "cherry", "banana"]
  let ways_to_stack := Nat.factorial ice_cream_flavors.length
  show (ways_to_stack = 120)
  sorry

end ice_cream_stacks_l155_155021


namespace pencils_count_l155_155632

theorem pencils_count (initial_pencils additional_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : additional_pencils = 100) : initial_pencils + additional_pencils = 215 :=
by sorry

end pencils_count_l155_155632


namespace coordinates_of_D_l155_155991

-- Definitions of the points and translation conditions
def A : (ℝ × ℝ) := (-1, 4)
def B : (ℝ × ℝ) := (-4, -1)
def C : (ℝ × ℝ) := (4, 7)

theorem coordinates_of_D :
  ∃ (D : ℝ × ℝ), D = (1, 2) ∧
  ∀ (translate : ℝ × ℝ), translate = (C.1 - A.1, C.2 - A.2) → 
  D = (B.1 + translate.1, B.2 + translate.2) :=
by
  sorry

end coordinates_of_D_l155_155991


namespace ratio_a6_b6_l155_155752

noncomputable def a_n (n : ℕ) : ℝ := sorry -- Define the nth term of sequence a
noncomputable def b_n (n : ℕ) : ℝ := sorry -- Define the nth term of sequence b
noncomputable def S_n (n : ℕ) : ℝ := sorry -- Define the sum of the first n terms of sequence a
noncomputable def T_n (n : ℕ) : ℝ := sorry -- Define the sum of the first n terms of sequence b

axiom condition (n : ℕ) : S_n n / T_n n = (2 * n) / (3 * n + 1)

theorem ratio_a6_b6 : a_n 6 / b_n 6 = 11 / 17 :=
by
  sorry

end ratio_a6_b6_l155_155752


namespace repeated_process_pure_alcohol_l155_155894

theorem repeated_process_pure_alcohol : 
  ∃ n : ℕ, n ≥ 4 ∧ ∀ m < 4, 2 * (1 / 2 : ℝ)^(m : ℝ) ≥ 0.2 := by
  sorry

end repeated_process_pure_alcohol_l155_155894


namespace sixth_root_binomial_expansion_l155_155182

theorem sixth_root_binomial_expansion :
  (2748779069441 = 1 * 150^6 + 6 * 150^5 + 15 * 150^4 + 20 * 150^3 + 15 * 150^2 + 6 * 150 + 1) →
  (2748779069441 = Nat.choose 6 6 * 150^6 + Nat.choose 6 5 * 150^5 + Nat.choose 6 4 * 150^4 + Nat.choose 6 3 * 150^3 + Nat.choose 6 2 * 150^2 + Nat.choose 6 1 * 150 + Nat.choose 6 0) →
  (Real.sqrt (2748779069441 : ℝ) = 151) :=
by
  intros h1 h2
  sorry

end sixth_root_binomial_expansion_l155_155182


namespace card_dealing_probability_l155_155112

-- Define the events and their probabilities
def prob_first_card_ace : ℚ := 4 / 52
def prob_second_card_ten_given_ace : ℚ := 4 / 51
def prob_third_card_jack_given_ace_and_ten : ℚ := 2 / 25

-- Define the overall probability
def overall_probability : ℚ :=
  prob_first_card_ace * 
  prob_second_card_ten_given_ace *
  prob_third_card_jack_given_ace_and_ten

-- State the problem
theorem card_dealing_probability :
  overall_probability = 8 / 16575 := by
  sorry

end card_dealing_probability_l155_155112


namespace students_spring_outing_l155_155842

theorem students_spring_outing (n : ℕ) (h1 : n = 5) : 2^n = 32 :=
  by {
    sorry
  }

end students_spring_outing_l155_155842


namespace shortest_distance_from_curve_to_line_l155_155788

noncomputable def curve (x : ℝ) : ℝ := Real.log (2 * x - 1)

def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

theorem shortest_distance_from_curve_to_line : 
  ∃ (x y : ℝ), y = curve x ∧ line x y ∧ 
  (∀ (x₀ y₀ : ℝ), y₀ = curve x₀ → ∃ (x₀ y₀ : ℝ), 
    y₀ = curve x₀ ∧ d = Real.sqrt 5) :=
sorry

end shortest_distance_from_curve_to_line_l155_155788


namespace carbonate_weight_l155_155263

namespace MolecularWeight

def molecular_weight_Al2_CO3_3 : ℝ := 234
def molecular_weight_Al : ℝ := 26.98
def num_Al_atoms : ℕ := 2

theorem carbonate_weight :
  molecular_weight_Al2_CO3_3 - (num_Al_atoms * molecular_weight_Al) = 180.04 :=
sorry

end MolecularWeight

end carbonate_weight_l155_155263


namespace people_in_room_after_2019_minutes_l155_155452

theorem people_in_room_after_2019_minutes :
  ∀ (P : Nat → Int), 
    P 0 = 0 -> 
    (∀ t, P (t+1) = P t + 2 ∨ P (t+1) = P t - 1) -> 
    P 2019 ≠ 2018 :=
by
  intros P hP0 hP_changes
  sorry

end people_in_room_after_2019_minutes_l155_155452


namespace find_k_value_l155_155675

-- Define the condition that point A(3, -5) lies on the graph of the function y = k / x
def point_on_inverse_proportion (k : ℝ) : Prop :=
  (3 : ℝ) ≠ 0 ∧ (-5) = k / (3 : ℝ)

-- The theorem to prove that k = -15 given the point on the graph
theorem find_k_value (k : ℝ) (h : point_on_inverse_proportion k) : k = -15 :=
by
  sorry

end find_k_value_l155_155675


namespace smallest_n_l155_155137

theorem smallest_n (n : ℕ) (h1 : ∃ k1, 4 * n = k1^2) (h2 : ∃ k2, 3 * n = k2^3) : n = 144 :=
sorry

end smallest_n_l155_155137


namespace simplify_expression_l155_155339

theorem simplify_expression (x : ℤ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 + 24 = 45 * x + 42 := 
by 
  -- proof steps
  sorry

end simplify_expression_l155_155339


namespace kids_wearing_shoes_l155_155065

-- Definitions based on the problem's conditions
def total_kids := 22
def kids_with_socks := 12
def kids_with_both := 6
def barefoot_kids := 8

-- Theorem statement
theorem kids_wearing_shoes :
  (∃ (kids_with_shoes : ℕ), 
     (kids_with_shoes = (total_kids - barefoot_kids) - (kids_with_socks - kids_with_both) + kids_with_both) ∧ 
     kids_with_shoes = 8) :=
by
  sorry

end kids_wearing_shoes_l155_155065


namespace find_base_k_l155_155634

theorem find_base_k : ∃ k : ℕ, 6 * k^2 + 6 * k + 4 = 340 ∧ k = 7 := 
by 
  sorry

end find_base_k_l155_155634


namespace initial_thickness_of_blanket_l155_155888

theorem initial_thickness_of_blanket (T : ℝ)
  (h : ∀ n, n = 4 → T * 2^n = 48) : T = 3 :=
by
  have h4 := h 4 rfl
  sorry

end initial_thickness_of_blanket_l155_155888


namespace monster_ratio_l155_155254

theorem monster_ratio (r : ℝ) :
  (121 + 121 * r + 121 * r^2 = 847) → r = 2 :=
by
  intros h
  sorry

end monster_ratio_l155_155254


namespace compute_p2_q2_compute_p3_q3_l155_155011

variables (p q : ℝ)

theorem compute_p2_q2 (h1 : p * q = 15) (h2 : p + q = 8) : p^2 + q^2 = 34 :=
sorry

theorem compute_p3_q3 (h1 : p * q = 15) (h2 : p + q = 8) : p^3 + q^3 = 152 :=
sorry

end compute_p2_q2_compute_p3_q3_l155_155011


namespace multiplicative_inverse_CD_mod_1000000_l155_155834

theorem multiplicative_inverse_CD_mod_1000000 :
  let C := 123456
  let D := 166666
  let M := 48
  M * (C * D) % 1000000 = 1 := by
  sorry

end multiplicative_inverse_CD_mod_1000000_l155_155834


namespace daniel_practices_total_minutes_in_week_l155_155131

theorem daniel_practices_total_minutes_in_week :
  let school_minutes_per_day := 15
  let school_days := 5
  let weekend_minutes_per_day := 2 * school_minutes_per_day
  let weekend_days := 2
  let total_school_week_minutes := school_minutes_per_day * school_days
  let total_weekend_minutes := weekend_minutes_per_day * weekend_days
  total_school_week_minutes + total_weekend_minutes = 135 :=
by
  sorry

end daniel_practices_total_minutes_in_week_l155_155131


namespace james_writing_hours_per_week_l155_155018

variables (pages_per_hour : ℕ) (pages_per_day_per_person : ℕ) (people : ℕ) (days_per_week : ℕ)

theorem james_writing_hours_per_week
  (h1 : pages_per_hour = 10)
  (h2 : pages_per_day_per_person = 5)
  (h3 : people = 2)
  (h4 : days_per_week = 7) :
  (pages_per_day_per_person * people * days_per_week) / pages_per_hour = 7 :=
by
  sorry

end james_writing_hours_per_week_l155_155018


namespace enemies_left_undefeated_l155_155125

theorem enemies_left_undefeated (points_per_enemy : ℕ) (total_enemies : ℕ) (total_points_earned : ℕ) 
  (h1: points_per_enemy = 9) (h2: total_enemies = 11) (h3: total_points_earned = 72):
  total_enemies - (total_points_earned / points_per_enemy) = 3 :=
by
  sorry

end enemies_left_undefeated_l155_155125


namespace arithmetic_sequence_common_difference_l155_155106

theorem arithmetic_sequence_common_difference
  (a1 a4 : ℤ) (d : ℤ) 
  (h1 : a1 + (a1 + 4 * d) = 10)
  (h2 : a1 + 3 * d = 7) : 
  d = 2 :=
sorry

end arithmetic_sequence_common_difference_l155_155106


namespace tickets_to_be_sold_l155_155865

theorem tickets_to_be_sold : 
  let total_tickets := 200
  let jude_tickets := 16
  let andrea_tickets := 4 * jude_tickets
  let sandra_tickets := 2 * jude_tickets + 8
  total_tickets - (jude_tickets + andrea_tickets + sandra_tickets) = 80 := by
  sorry

end tickets_to_be_sold_l155_155865


namespace tournament_committees_l155_155139

-- Assuming each team has 7 members
def team_members : Nat := 7

-- There are 5 teams
def total_teams : Nat := 5

-- The host team selects 3 members including at least one woman
def select_host_team_members (w m : Nat) : ℕ :=
  let total_combinations := Nat.choose team_members 3
  let all_men_combinations := Nat.choose (team_members - 1) 3
  total_combinations - all_men_combinations

-- Each non-host team selects 2 members including at least one woman
def select_non_host_team_members (w m : Nat) : ℕ :=
  let total_combinations := Nat.choose team_members 2
  let all_men_combinations := Nat.choose (team_members - 1) 2
  total_combinations - all_men_combinations

-- Total number of committees when one team is the host
def one_team_host_total_combinations (w m : Nat) : ℕ :=
  select_host_team_members w m * (select_non_host_team_members w m) ^ (total_teams - 1)

-- Total number of possible 11-member tournament committees
def total_committees (w m : Nat) : ℕ :=
  one_team_host_total_combinations w m * total_teams

theorem tournament_committees (w m : Nat) (hw : w ≥ 1) (hm : m ≤ 6) :
  total_committees w m = 97200 :=
by
  sorry

end tournament_committees_l155_155139


namespace range_of_m_l155_155202

theorem range_of_m (a b m : ℝ) (h1 : 2 * b = 2 * a + b) (h2 : b * b = a * a * b) (h3 : 0 < Real.log b / Real.log m) (h4 : Real.log b / Real.log m < 1) : m > 8 :=
sorry

end range_of_m_l155_155202


namespace triangle_angle_contradiction_l155_155723

theorem triangle_angle_contradiction (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A > 60) (h₃ : B > 60) (h₄ : C > 60) :
  false :=
by
  sorry

end triangle_angle_contradiction_l155_155723


namespace express_function_as_chain_of_equalities_l155_155361

theorem express_function_as_chain_of_equalities (x : ℝ) : 
  ∃ (u : ℝ), (u = 2 * x - 5) ∧ ((2 * x - 5) ^ 10 = u ^ 10) :=
by 
  sorry

end express_function_as_chain_of_equalities_l155_155361


namespace initial_number_of_quarters_l155_155321

theorem initial_number_of_quarters 
  (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (half_dollars : ℕ) (dollar_coins : ℕ) 
  (two_dollar_coins : ℕ) (quarters : ℕ)
  (cost_per_sundae : ℝ) 
  (special_topping_cost : ℝ)
  (featured_flavor_discount : ℝ)
  (members_with_special_topping : ℕ)
  (members_with_featured_flavor : ℕ)
  (left_over : ℝ)
  (expected_quarters : ℕ) :
  pennies = 123 ∧
  nickels = 85 ∧
  dimes = 35 ∧
  half_dollars = 15 ∧
  dollar_coins = 5 ∧
  quarters = expected_quarters ∧
  two_dollar_coins = 4 ∧
  cost_per_sundae = 5.25 ∧
  special_topping_cost = 0.50 ∧
  featured_flavor_discount = 0.25 ∧
  members_with_special_topping = 3 ∧
  members_with_featured_flavor = 5 ∧
  left_over = 0.97 →
  expected_quarters = 54 :=
  by
  sorry

end initial_number_of_quarters_l155_155321


namespace trains_crossing_time_l155_155154

theorem trains_crossing_time :
  let length_of_each_train := 120 -- in meters
  let speed_of_each_train := 12 -- in km/hr
  let total_distance := length_of_each_train * 2
  let relative_speed := (speed_of_each_train * 1000 / 3600 * 2) -- in m/s
  total_distance / relative_speed = 36 := 
by
  -- Since we only need to state the theorem, the proof is omitted.
  sorry

end trains_crossing_time_l155_155154


namespace philip_paints_2_per_day_l155_155962

def paintings_per_day (initial_paintings total_paintings days : ℕ) : ℕ :=
  (total_paintings - initial_paintings) / days

theorem philip_paints_2_per_day :
  paintings_per_day 20 80 30 = 2 :=
by
  sorry

end philip_paints_2_per_day_l155_155962


namespace medium_stores_to_select_l155_155520

-- Definitions based on conditions in a)
def total_stores := 1500
def ratio_large := 1
def ratio_medium := 5
def ratio_small := 9
def sample_size := 30
def medium_proportion := ratio_medium / (ratio_large + ratio_medium + ratio_small)

-- Main theorem to prove
theorem medium_stores_to_select : (sample_size * medium_proportion) = 10 :=
by sorry

end medium_stores_to_select_l155_155520


namespace find_m_over_n_l155_155130

noncomputable
def ellipse_intersection_midpoint (m n : ℝ) (P : ℝ × ℝ) : Prop :=
  let M := (P.1, 1 - P.1)
  let N := (1 - P.2, P.2)
  let midpoint_MN := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  P = midpoint_MN

noncomputable
def ellipse_condition (m n : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

noncomputable
def line_condition (x y : ℝ) : Prop :=
  x + y = 1

noncomputable
def slope_OP_condition (P : ℝ × ℝ) : Prop :=
  P.2 / P.1 = (Real.sqrt 2 / 2)

theorem find_m_over_n
  (m n : ℝ)
  (P : ℝ × ℝ)
  (h1 : ellipse_condition m n P.1 P.2)
  (h2 : line_condition P.1 P.2)
  (h3 : slope_OP_condition P)
  (h4 : ellipse_intersection_midpoint m n P) :
  (m / n = 1) :=
sorry

end find_m_over_n_l155_155130


namespace factor_of_polynomial_l155_155251

theorem factor_of_polynomial :
  (x^4 + 4 * x^2 + 16) % (x^2 + 4) = 0 :=
sorry

end factor_of_polynomial_l155_155251


namespace fair_game_x_value_l155_155914

theorem fair_game_x_value (x : ℕ) (h : x + 2 * x + 2 * x = 15) : x = 3 := 
by sorry

end fair_game_x_value_l155_155914


namespace monthly_rent_calc_l155_155301

def monthly_rent (length width annual_rent_per_sq_ft : ℕ) : ℕ :=
  (length * width * annual_rent_per_sq_ft) / 12

theorem monthly_rent_calc :
  monthly_rent 10 8 360 = 2400 := 
  sorry

end monthly_rent_calc_l155_155301


namespace triangle_ratio_l155_155564

-- Given conditions:
-- a: one side of the triangle
-- h_a: height corresponding to side a
-- r: inradius of the triangle
-- p: semiperimeter of the triangle

theorem triangle_ratio (a h_a r p : ℝ) (area_formula_1 : p * r = 1 / 2 * a * h_a) :
  (2 * p) / a = h_a / r :=
by {
  sorry
}

end triangle_ratio_l155_155564


namespace alicia_taxes_l155_155920

theorem alicia_taxes:
  let w := 20 -- Alicia earns 20 dollars per hour
  let r := 1.45 / 100 -- The local tax rate is 1.45%
  let wage_in_cents := w * 100 -- Convert dollars to cents
  let tax_deduction := wage_in_cents * r -- Calculate tax deduction in cents
  tax_deduction = 29 := 
by 
  sorry

end alicia_taxes_l155_155920


namespace remainder_problem_l155_155372

theorem remainder_problem : (9^5 + 8^6 + 7^7) % 7 = 5 := by
  sorry

end remainder_problem_l155_155372


namespace perpendicular_condition_sufficient_not_necessary_l155_155479

theorem perpendicular_condition_sufficient_not_necessary (m : ℝ) :
  (∀ x y : ℝ, m * x + (2 * m - 1) * y + 1 = 0) →
  (∀ x y : ℝ, 3 * x + m * y + 3 = 0) →
  (∀ a b : ℝ, m = -1 → (∃ c d : ℝ, 3 / a = 1 / b)) →
  (m = -1 → (m = -1 → (3 / (-m / (2 * m - 1)) * m) / 2 - (3 / m) = -1)) :=
by sorry

end perpendicular_condition_sufficient_not_necessary_l155_155479


namespace prove_p_l155_155988

variables {m n p : ℝ}

/-- Given points (m, n) and (m + p, n + 4) lie on the line 
   x = y / 2 - 2 / 5, prove p = 2.
-/
theorem prove_p (hmn : m = n / 2 - 2 / 5)
                (hmpn4 : m + p = (n + 4) / 2 - 2 / 5) : p = 2 := 
by
  sorry

end prove_p_l155_155988


namespace circle_integer_solution_max_sum_l155_155231

theorem circle_integer_solution_max_sum : ∀ (x y : ℤ), (x - 1)^2 + (y + 2)^2 = 16 → x + y ≤ 3 :=
by
  sorry

end circle_integer_solution_max_sum_l155_155231


namespace mary_gave_becky_green_crayons_l155_155738

-- Define the initial conditions
def initial_green_crayons : Nat := 5
def initial_blue_crayons : Nat := 8
def given_blue_crayons : Nat := 1
def remaining_crayons : Nat := 9

-- Define the total number of crayons initially
def total_initial_crayons : Nat := initial_green_crayons + initial_blue_crayons

-- Define the number of crayons given away
def given_crayons : Nat := total_initial_crayons - remaining_crayons

-- The crux of the problem
def given_green_crayons : Nat :=
  given_crayons - given_blue_crayons

-- Formal statement of the theorem
theorem mary_gave_becky_green_crayons
  (h_initial_green : initial_green_crayons = 5)
  (h_initial_blue : initial_blue_crayons = 8)
  (h_given_blue : given_blue_crayons = 1)
  (h_remaining : remaining_crayons = 9) :
  given_green_crayons = 3 :=
by {
  -- This should be the body of the proof, but we'll skip it for now
  sorry
}

end mary_gave_becky_green_crayons_l155_155738


namespace fraction_subtraction_l155_155377

theorem fraction_subtraction : (9 / 23) - (5 / 69) = 22 / 69 :=
by
  sorry

end fraction_subtraction_l155_155377


namespace marble_problem_l155_155600

-- Define the given conditions
def ratio (red blue green : ℕ) : Prop := red * 3 * 4 = blue * 2 * 4 ∧ blue * 2 * 4 = green * 2 * 3

-- The total number of marbles
def total_marbles (red blue green : ℕ) : ℕ := red + blue + green

-- The number of green marbles is given
def green_marbles : ℕ := 36

-- Proving the number of marbles and number of red marbles
theorem marble_problem
  (red blue green : ℕ)
  (h_ratio : ratio red blue green)
  (h_green : green = green_marbles) :
  total_marbles red blue green = 81 ∧ red = 18 :=
by
  sorry

end marble_problem_l155_155600


namespace cistern_water_depth_l155_155423

theorem cistern_water_depth
  (length width : ℝ) 
  (wet_surface_area : ℝ)
  (h : ℝ) 
  (hl : length = 7)
  (hw : width = 4)
  (ha : wet_surface_area = 55.5)
  (h_eq : 28 + 22 * h = wet_surface_area) 
  : h = 1.25 := 
  by 
  sorry

end cistern_water_depth_l155_155423


namespace quadratic_has_exactly_one_root_l155_155563

noncomputable def discriminant (b c : ℝ) : ℝ :=
b^2 - 4 * c

noncomputable def f (x b c : ℝ) : ℝ :=
x^2 + b * x + c

noncomputable def transformed_f (x b c : ℝ) : ℝ :=
(x - 2020)^2 + b * (x - 2020) + c

theorem quadratic_has_exactly_one_root (b c : ℝ)
  (h_discriminant : discriminant b c = 2020) :
  ∃! x : ℝ, f (x - 2020) b c + f x b c = 0 :=
sorry

end quadratic_has_exactly_one_root_l155_155563


namespace expression_undefined_at_9_l155_155236

theorem expression_undefined_at_9 (x : ℝ) : (3 * x ^ 3 - 5) / (x ^ 2 - 18 * x + 81) = 0 → x = 9 :=
by sorry

end expression_undefined_at_9_l155_155236


namespace num_factors_34848_l155_155262

/-- Define the number 34848 and its prime factorization -/
def n : ℕ := 34848
def p_factors : List (ℕ × ℕ) := [(2, 5), (3, 2), (11, 2)]

/-- Helper function to calculate the number of divisors from prime factors -/
def num_divisors (factors : List (ℕ × ℕ)) : ℕ := 
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.2 + 1)) 1

/-- Formal statement of the problem -/
theorem num_factors_34848 : num_divisors p_factors = 54 :=
by
  -- Proof that 34848 has the prime factorization 3^2 * 2^5 * 11^2 
  -- and that the number of factors is 54 would go here.
  sorry

end num_factors_34848_l155_155262


namespace pieces_from_sister_calculation_l155_155336

-- Definitions for the conditions
def pieces_from_neighbors : ℕ := 5
def pieces_per_day : ℕ := 9
def duration : ℕ := 2

-- Definition to calculate the total number of pieces Emily ate
def total_pieces : ℕ := pieces_per_day * duration

-- Proof Problem: Prove Emily received 13 pieces of candy from her older sister
theorem pieces_from_sister_calculation :
  ∃ (pieces_from_sister : ℕ), pieces_from_sister = total_pieces - pieces_from_neighbors ∧ pieces_from_sister = 13 :=
by
  sorry

end pieces_from_sister_calculation_l155_155336


namespace smallest_positive_multiple_of_45_l155_155285

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l155_155285


namespace number_of_triangles_fitting_in_square_l155_155753

-- Define the conditions for the right triangle and the square
def right_triangle_height := 2
def right_triangle_width := 2
def square_side := 2

-- Define the areas
def area_triangle := (1 / 2) * right_triangle_height * right_triangle_width
def area_square := square_side * square_side

-- Define the proof statement to show the number of right triangles fitting in the square is 2
theorem number_of_triangles_fitting_in_square : (area_square / area_triangle) = 2 := by
  sorry

end number_of_triangles_fitting_in_square_l155_155753


namespace problem_l155_155356

theorem problem (x : ℝ) (h : x + 1/x = 10) :
  (x^2 + 1/x^2 = 98) ∧ (x^3 + 1/x^3 = 970) :=
by
  sorry

end problem_l155_155356


namespace clownfish_display_tank_l155_155004

theorem clownfish_display_tank
  (C B : ℕ)
  (h1 : C = B)
  (h2 : C + B = 100)
  (h3 : ∀ dC dB : ℕ, dC = dB → C - dC = 24)
  (h4 : ∀ b : ℕ, b = (1 / 3) * 24): 
  C - (1 / 3 * 24) = 16 := sorry

end clownfish_display_tank_l155_155004


namespace faye_age_l155_155682
open Nat

theorem faye_age :
  ∃ (C D E F : ℕ), 
    (D = E - 3) ∧ 
    (E = C + 4) ∧ 
    (F = C + 3) ∧ 
    (D = 14) ∧ 
    (F = 16) :=
by
  sorry

end faye_age_l155_155682


namespace parallelogram_base_length_l155_155976

theorem parallelogram_base_length (A : ℕ) (h b : ℕ) (h1 : A = b * h) (h2 : h = 2 * b) (h3 : A = 200) : b = 10 :=
by {
  sorry
}

end parallelogram_base_length_l155_155976


namespace B_pow_16_eq_I_l155_155649

noncomputable def B : Matrix (Fin 4) (Fin 4) ℝ := 
  ![
    ![Real.cos (Real.pi / 4), -Real.sin (Real.pi / 4), 0 , 0],
    ![Real.sin (Real.pi / 4), Real.cos (Real.pi / 4), 0 , 0],
    ![0, 0, Real.cos (Real.pi / 4), Real.sin (Real.pi / 4)],
    ![0, 0, -Real.sin (Real.pi / 4), Real.cos (Real.pi / 4)]
  ]

theorem B_pow_16_eq_I : B^16 = 1 := by
  sorry

end B_pow_16_eq_I_l155_155649


namespace exists_not_odd_l155_155891

variable (f : ℝ → ℝ)

-- Define the condition that f is not an odd function
def not_odd_function := ¬ (∀ x : ℝ, f (-x) = -f x)

-- Lean statement to prove the correct answer
theorem exists_not_odd (h : not_odd_function f) : ∃ x : ℝ, f (-x) ≠ -f x :=
sorry

end exists_not_odd_l155_155891


namespace range_of_a_l155_155149

noncomputable def my_function (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + (x + 1) ^ 2

theorem range_of_a (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₁ > x₂ → my_function a x₁ - my_function a x₂ ≥ 4 * (x₁ - x₂)) → a ≥ 1 / 2 :=
by
  sorry

end range_of_a_l155_155149


namespace difference_of_squares_550_450_l155_155568

theorem difference_of_squares_550_450 : (550 ^ 2 - 450 ^ 2) = 100000 := 
by
  sorry

end difference_of_squares_550_450_l155_155568


namespace cartesian_equation_of_parametric_l155_155732

variable (t : ℝ) (x y : ℝ)

open Real

theorem cartesian_equation_of_parametric 
  (h1 : x = sqrt t)
  (h2 : y = 2 * sqrt (1 - t))
  (h3 : 0 ≤ t ∧ t ≤ 1) :
  (x^2 / 1) + (y^2 / 4) = 1 := by 
  sorry

end cartesian_equation_of_parametric_l155_155732


namespace coefficients_sum_binomial_coefficients_sum_l155_155541

theorem coefficients_sum (x : ℝ) (h : (x + 2 / x)^6 = coeff_sum) : coeff_sum = 729 := 
sorry

theorem binomial_coefficients_sum (x : ℝ) (h : (x + 2 / x)^6 = binom_coeff_sum) : binom_coeff_sum = 64 := 
sorry

end coefficients_sum_binomial_coefficients_sum_l155_155541


namespace tiles_cover_the_floor_l155_155979

theorem tiles_cover_the_floor
  (n : ℕ)
  (h : 2 * n - 1 = 101)
  : n ^ 2 = 2601 := sorry

end tiles_cover_the_floor_l155_155979


namespace quadratic_eq_real_roots_l155_155848

theorem quadratic_eq_real_roots (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 4 * x + 2 = 0) →
  (∃ y : ℝ, a * y^2 - 4 * y + 2 = 0) →
  a ≤ 2 ∧ a ≠ 0 :=
by sorry

end quadratic_eq_real_roots_l155_155848


namespace vector_parallel_addition_l155_155053

theorem vector_parallel_addition 
  (x : ℝ)
  (a : ℝ × ℝ := (2, 1))
  (b : ℝ × ℝ := (x, -2)) 
  (h_parallel : 2 / x = 1 / -2) :
  a + b = (-2, -1) := 
by
  -- While the proof is omitted, the statement is complete and correct.
  sorry

end vector_parallel_addition_l155_155053


namespace roger_trips_required_l155_155548

variable (carry_trays_per_trip total_trays : ℕ)

theorem roger_trips_required (h1 : carry_trays_per_trip = 4) (h2 : total_trays = 12) : total_trays / carry_trays_per_trip = 3 :=
by
  -- proof follows
  sorry

end roger_trips_required_l155_155548


namespace systematic_sample_first_segment_number_l155_155802

theorem systematic_sample_first_segment_number :
  ∃ a_1 : ℕ, ∀ d k : ℕ, k = 5 → a_1 + (59 - 1) * k = 293 → a_1 = 3 :=
by
  sorry

end systematic_sample_first_segment_number_l155_155802


namespace range_of_x_l155_155996

theorem range_of_x (x : ℝ) : (2 : ℝ)^(3 - 2 * x) < (2 : ℝ)^(3 * x - 4) → x > 7 / 5 := by
  sorry

end range_of_x_l155_155996


namespace directrix_of_parabola_l155_155744

theorem directrix_of_parabola :
  ∀ (x : ℝ), (∃ k : ℝ, y = (x^2 - 8 * x + 16) / 8 → k = -2) :=
by
  sorry

end directrix_of_parabola_l155_155744


namespace greenville_height_of_boxes_l155_155350

theorem greenville_height_of_boxes:
  ∃ h : ℝ, 
    (20 * 20 * h) * (2160000 / (20 * 20 * h)) * 0.40 = 180 ∧ 
    400 * h = 2160000 / (2160000 / (20 * 20 * h)) ∧
    400 * h = 5400 ∧
    h = 12 :=
    sorry

end greenville_height_of_boxes_l155_155350


namespace find_n_l155_155493

theorem find_n (x k m n : ℤ) 
  (h1 : x = 82 * k + 5)
  (h2 : x + n = 41 * m + 18) :
  n = 5 :=
by
  sorry

end find_n_l155_155493


namespace Mateen_garden_area_l155_155113

theorem Mateen_garden_area :
  ∀ (L W : ℝ), (50 * L = 2000) ∧ (20 * (2 * L + 2 * W) = 2000) → (L * W = 400) :=
by
  intros L W h
  -- We have two conditions based on the problem:
  -- 1. Mateen must walk the length 50 times to cover 2000 meters.
  -- 2. Mateen must walk the perimeter 20 times to cover 2000 meters.
  have h1 : 50 * L = 2000 := h.1
  have h2 : 20 * (2 * L + 2 * W) = 2000 := h.2
  -- We can use these conditions to derive the area of the garden
  sorry

end Mateen_garden_area_l155_155113


namespace max_sin_a_l155_155945

theorem max_sin_a (a b : ℝ) (h : Real.sin (a + b) = Real.sin a + Real.sin b) : Real.sin a ≤ 1 :=
by
  sorry

end max_sin_a_l155_155945


namespace cost_per_book_l155_155611

theorem cost_per_book
  (books_sold_each_time : ℕ)
  (people_bought : ℕ)
  (income_per_book : ℕ)
  (profit : ℕ)
  (total_income : ℕ := books_sold_each_time * people_bought * income_per_book)
  (total_cost : ℕ := total_income - profit)
  (total_books : ℕ := books_sold_each_time * people_bought)
  (cost_per_book : ℕ := total_cost / total_books) :
  books_sold_each_time = 2 ->
  people_bought = 4 ->
  income_per_book = 20 ->
  profit = 120 ->
  cost_per_book = 5 :=
  by intros; sorry

end cost_per_book_l155_155611


namespace problem_solution_l155_155157

theorem problem_solution (a b c d : ℝ) 
  (h1 : a + b = 14) 
  (h2 : b + c = 9) 
  (h3 : c + d = 3) : 
  a + d = 8 := 
by 
  sorry

end problem_solution_l155_155157


namespace range_of_a_l155_155307

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 0 then (x - a) ^ 2 else x + (1 / x) + a

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ f 0 a) ↔ 0 ≤ a ∧ a ≤ 2 := 
by
  sorry

end range_of_a_l155_155307


namespace number_of_students_in_line_l155_155692

-- Definitions for the conditions
def yoojung_last (n : ℕ) : Prop :=
  n = 14

def eunjung_position : ℕ := 5

def students_between (n : ℕ) : Prop :=
  n = 8

noncomputable def total_students : ℕ := 14

-- The theorem to be proven
theorem number_of_students_in_line 
  (last : yoojung_last total_students) 
  (eunjung_pos : eunjung_position = 5) 
  (between : students_between 8) :
  total_students = 14 := by
  sorry

end number_of_students_in_line_l155_155692


namespace additional_savings_l155_155561

def window_price : ℕ := 100

def special_offer (windows_purchased : ℕ) : ℕ :=
  windows_purchased + windows_purchased / 6 * 2

def dave_windows : ℕ := 10

def doug_windows : ℕ := 12

def total_windows := dave_windows + doug_windows

def calculate_windows_cost (windows_needed : ℕ) : ℕ :=
  if windows_needed % 8 = 0 then (windows_needed / 8) * 6 * window_price
  else ((windows_needed / 8) * 6 + (windows_needed % 8)) * window_price

def separate_savings : ℕ :=
  window_price * (dave_windows + doug_windows) - (calculate_windows_cost dave_windows + calculate_windows_cost doug_windows)

def combined_savings : ℕ :=
  window_price * total_windows - calculate_windows_cost total_windows

theorem additional_savings :
  separate_savings + 200 = combined_savings :=
sorry

end additional_savings_l155_155561


namespace prime_consecutive_fraction_equivalence_l155_155814

theorem prime_consecutive_fraction_equivalence (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hq_p_consec : p + 1 ≤ q ∧ Nat.Prime (p + 1) -> p + 1 = q) (hpq : p < q) (frac_eq : p / q = 4 / 5) :
  25 / 7 + (2 * q - p) / (2 * q + p) = 4 := sorry

end prime_consecutive_fraction_equivalence_l155_155814


namespace largest_number_of_four_consecutive_whole_numbers_l155_155730

theorem largest_number_of_four_consecutive_whole_numbers 
  (a : ℕ) (h1 : a + (a + 1) + (a + 2) = 184)
  (h2 : a + (a + 1) + (a + 3) = 201)
  (h3 : a + (a + 2) + (a + 3) = 212)
  (h4 : (a + 1) + (a + 2) + (a + 3) = 226) : 
  a + 3 = 70 := 
by sorry

end largest_number_of_four_consecutive_whole_numbers_l155_155730


namespace expected_value_coin_flip_l155_155923

-- Definitions based on conditions
def P_heads : ℚ := 2 / 3
def P_tails : ℚ := 1 / 3
def win_heads : ℚ := 4
def lose_tails : ℚ := -9

-- Expected value calculation
def expected_value : ℚ :=
  P_heads * win_heads + P_tails * lose_tails

-- Theorem statement to be proven
theorem expected_value_coin_flip : expected_value = -1 / 3 :=
by sorry

end expected_value_coin_flip_l155_155923


namespace trigonometric_inequality_for_tan_l155_155135

open Real

theorem trigonometric_inequality_for_tan (x : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) : 
  1 + tan x < 1 / (1 - sin x) :=
sorry

end trigonometric_inequality_for_tan_l155_155135


namespace room_volume_l155_155134

theorem room_volume (b l h : ℝ) (h1 : l = 3 * b) (h2 : h = 2 * b) (h3 : l * b = 12) :
  l * b * h = 48 :=
by sorry

end room_volume_l155_155134


namespace function_bounded_in_interval_l155_155859

variables {f : ℝ → ℝ}

theorem function_bounded_in_interval (h : ∀ x y : ℝ, x > y → f x ^ 2 ≤ f y) : ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 :=
by
  sorry

end function_bounded_in_interval_l155_155859


namespace parabola_vertex_in_fourth_quadrant_l155_155371

theorem parabola_vertex_in_fourth_quadrant (a c : ℝ) (h : -a > 0 ∧ c < 0) :
  a < 0 ∧ c < 0 :=
by
  sorry

end parabola_vertex_in_fourth_quadrant_l155_155371


namespace sabrina_total_leaves_l155_155367

-- Definitions based on conditions
def basil_leaves := 12
def twice_the_sage_leaves (sages : ℕ) := 2 * sages = basil_leaves
def five_fewer_than_verbena (sages verbenas : ℕ) := sages + 5 = verbenas

-- Statement to prove
theorem sabrina_total_leaves (sages verbenas : ℕ) 
    (h1 : twice_the_sage_leaves sages) 
    (h2 : five_fewer_than_verbena sages verbenas) :
    basil_leaves + sages + verbenas = 29 :=
sorry

end sabrina_total_leaves_l155_155367


namespace chess_games_l155_155294

theorem chess_games (n : ℕ) (total_games : ℕ) (players : ℕ) (games_per_player : ℕ)
  (h1 : players = 9)
  (h2 : total_games = 36)
  (h3 : ∀ i : ℕ, i < players → games_per_player = players - 1)
  (h4 : 2 * total_games = players * games_per_player) :
  games_per_player = 1 :=
by
  rw [h1, h2] at h4
  sorry

end chess_games_l155_155294


namespace mike_changed_64_tires_l155_155575

def total_tires_mike_changed (motorcycles : ℕ) (cars : ℕ) (tires_per_motorcycle : ℕ) (tires_per_car : ℕ) : ℕ :=
  motorcycles * tires_per_motorcycle + cars * tires_per_car

theorem mike_changed_64_tires :
  total_tires_mike_changed 12 10 2 4 = 64 :=
by
  sorry

end mike_changed_64_tires_l155_155575


namespace find_a_plus_b_l155_155674

def satisfies_conditions (a b : ℝ) :=
  ∀ x : ℝ, 3 * (a * x + b) - 8 = 4 * x + 7

theorem find_a_plus_b (a b : ℝ) (h : satisfies_conditions a b) : a + b = 19 / 3 :=
  sorry

end find_a_plus_b_l155_155674


namespace find_circle_center_l155_155782

theorem find_circle_center :
  ∀ x y : ℝ,
  (x^2 + 4*x + y^2 - 6*y = 20) →
  (x + 2, y - 3) = (-2, 3) := by
  sorry

end find_circle_center_l155_155782


namespace vasya_filling_time_l155_155034

-- Definition of conditions
def hose_filling_time (x : ℝ) : Prop :=
  ∀ (first_hose_mult second_hose_mult : ℝ), 
    first_hose_mult = x ∧
    second_hose_mult = 5 * x ∧
    (5 * second_hose_mult - 5 * first_hose_mult) = 1

-- Conclusion
theorem vasya_filling_time (x : ℝ) (first_hose_mult second_hose_mult : ℝ) :
  hose_filling_time x → 25 * x = 1 * (60 + 15) := sorry

end vasya_filling_time_l155_155034


namespace days_required_for_C_l155_155614

noncomputable def rate_A (r_A r_B r_C : ℝ) : Prop := r_A + r_B = 1 / 3
noncomputable def rate_B (r_A r_B r_C : ℝ) : Prop := r_B + r_C = 1 / 6
noncomputable def rate_C (r_A r_B r_C : ℝ) : Prop := r_C + r_A = 1 / 4
noncomputable def days_for_C (r_C : ℝ) : ℝ := 1 / r_C

theorem days_required_for_C
  (r_A r_B r_C : ℝ)
  (h1 : rate_A r_A r_B r_C)
  (h2 : rate_B r_A r_B r_C)
  (h3 : rate_C r_A r_B r_C) :
  days_for_C r_C = 4.8 :=
sorry

end days_required_for_C_l155_155614


namespace min_total_penalty_l155_155174

noncomputable def min_penalty (B W R : ℕ) : ℕ :=
  min (B * W) (min (2 * W * R) (3 * R * B))

theorem min_total_penalty (B W R : ℕ) :
  min_penalty B W R = min (B * W) (min (2 * W * R) (3 * R * B)) := by
  sorry

end min_total_penalty_l155_155174


namespace correct_inequality_l155_155298

theorem correct_inequality :
  (1 / 2)^(2 / 3) < (1 / 2)^(1 / 3) ∧ (1 / 2)^(1 / 3) < 1 :=
by sorry

end correct_inequality_l155_155298


namespace inequality_neg_mul_l155_155036

theorem inequality_neg_mul (a b : ℝ) (h : a < b) : -3 * a > -3 * b := by
    sorry

end inequality_neg_mul_l155_155036


namespace exponentiation_product_l155_155152

theorem exponentiation_product (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (3 ^ a) ^ b = 3 ^ 3) : 3 ^ a * 3 ^ b = 3 ^ 4 :=
by
  sorry

end exponentiation_product_l155_155152


namespace cost_of_article_l155_155877

theorem cost_of_article (C G1 G2 : ℝ) (h1 : G1 = 380 - C) (h2 : G2 = 450 - C) (h3 : G2 = 1.10 * G1) : 
  C = 320 :=
by
  sorry

end cost_of_article_l155_155877


namespace divisor_of_condition_l155_155122

theorem divisor_of_condition {d z : ℤ} (h1 : ∃ k : ℤ, z = k * d + 6)
  (h2 : ∃ m : ℤ, (z + 3) = d * m) : d = 9 := 
sorry

end divisor_of_condition_l155_155122


namespace circle_radius_d_l155_155990

theorem circle_radius_d (d : ℝ) : ∀ (x y : ℝ), (x^2 + 8 * x + y^2 + 2 * y + d = 0) → (∃ r : ℝ, r = 5) → d = -8 :=
by
  sorry

end circle_radius_d_l155_155990


namespace time_to_walk_without_walkway_l155_155594

theorem time_to_walk_without_walkway 
  (vp vw : ℝ) 
  (h1 : (vp + vw) * 40 = 80) 
  (h2 : (vp - vw) * 120 = 80) : 
  80 / vp = 60 :=
by
  sorry

end time_to_walk_without_walkway_l155_155594


namespace opponent_choice_is_random_l155_155720

-- Define the possible outcomes in the game
inductive Outcome
| rock
| paper
| scissors

-- Defining the opponent's choice set
def opponent_choice := {outcome : Outcome | outcome = Outcome.rock ∨ outcome = Outcome.paper ∨ outcome = Outcome.scissors}

-- The event where the opponent chooses "scissors"
def event_opponent_chooses_scissors := Outcome.scissors ∈ opponent_choice

-- Proving that the event of opponent choosing "scissors" is a random event
theorem opponent_choice_is_random : ¬(∀outcome ∈ opponent_choice, outcome = Outcome.scissors) ∧ (∃ outcome ∈ opponent_choice, outcome = Outcome.scissors) → event_opponent_chooses_scissors := 
sorry

end opponent_choice_is_random_l155_155720


namespace total_distance_AD_l155_155096

theorem total_distance_AD :
  let d_AB := 100
  let d_BC := d_AB + 50
  let d_CD := 2 * d_BC
  d_AB + d_BC + d_CD = 550 := by
  sorry

end total_distance_AD_l155_155096


namespace sum_of_80_consecutive_integers_l155_155215

-- Definition of the problem using the given conditions
theorem sum_of_80_consecutive_integers (n : ℤ) (h : (80 * (n + (n + 79))) / 2 = 40) : n = -39 := by
  sorry

end sum_of_80_consecutive_integers_l155_155215


namespace polynomial_at_five_l155_155458

theorem polynomial_at_five (P : ℝ → ℝ) 
  (hP_degree : ∃ (a b c d : ℝ), ∀ x : ℝ, P x = a*x^3 + b*x^2 + c*x + d)
  (hP1 : P 1 = 1 / 3)
  (hP2 : P 2 = 1 / 7)
  (hP3 : P 3 = 1 / 13)
  (hP4 : P 4 = 1 / 21) :
  P 5 = -3 / 91 :=
sorry

end polynomial_at_five_l155_155458


namespace problem_statement_l155_155478

noncomputable def percent_of_y (y : ℝ) (z : ℂ) : ℝ :=
  ((6 * y + 3 * z * Complex.I) / 20 + (3 * y + 4 * z * Complex.I) / 10).re

theorem problem_statement (y : ℝ) (z : ℂ) (hy : y > 0) : percent_of_y y z = 0.6 * y :=
by
  sorry

end problem_statement_l155_155478


namespace A_minus_B_l155_155767

def A : ℕ := 3^7 + Nat.choose 7 2 * 3^5 + Nat.choose 7 4 * 3^3 + Nat.choose 7 6 * 3
def B : ℕ := Nat.choose 7 1 * 3^6 + Nat.choose 7 3 * 3^4 + Nat.choose 7 5 * 3^2 + 1

theorem A_minus_B : A - B = 128 := by
  sorry

end A_minus_B_l155_155767


namespace task_probabilities_l155_155697

theorem task_probabilities (P1_on_time : ℚ) (P2_on_time : ℚ) 
  (h1 : P1_on_time = 2/3) (h2 : P2_on_time = 3/5) : 
  P1_on_time * (1 - P2_on_time) = 4/15 := 
by
  -- proof is omitted
  sorry

end task_probabilities_l155_155697


namespace tunnel_length_l155_155489

-- Definitions as per the conditions
def train_length : ℚ := 2  -- 2 miles
def train_speed : ℚ := 40  -- 40 miles per hour

def speed_in_miles_per_minute (speed_mph : ℚ) : ℚ :=
  speed_mph / 60  -- Convert speed from miles per hour to miles per minute

def time_travelled_in_minutes : ℚ := 5  -- 5 minutes

-- Theorem statement to prove the length of the tunnel
theorem tunnel_length (h1 : train_length = 2) (h2 : train_speed = 40) :
  (speed_in_miles_per_minute train_speed * time_travelled_in_minutes) - train_length = 4 / 3 :=
by
  sorry  -- Proof not included

end tunnel_length_l155_155489


namespace book_cost_price_l155_155407

theorem book_cost_price
  (C : ℝ) (P : ℝ) (SP : ℝ)
  (h1 : SP = 1.25 * C)
  (h2 : 0.95 * P = SP)
  (h3 : SP = 62.5) : 
  C = 50 := 
by
  sorry

end book_cost_price_l155_155407


namespace smallest_n_condition_l155_155644

noncomputable def distance_origin_to_point (n : ℕ) : ℝ := Real.sqrt (n)

noncomputable def radius_Bn (n : ℕ) : ℝ := distance_origin_to_point n - 1

def condition_Bn_contains_point_with_coordinate_greater_than_2 (n : ℕ) : Prop :=
  radius_Bn n > 2

theorem smallest_n_condition : ∃ n : ℕ, n ≥ 10 ∧ condition_Bn_contains_point_with_coordinate_greater_than_2 n :=
  sorry

end smallest_n_condition_l155_155644


namespace wizard_viable_combinations_l155_155810

def wizard_combination_problem : Prop :=
  let total_combinations := 4 * 6
  let incompatible_combinations := 3
  let viable_combinations := total_combinations - incompatible_combinations
  viable_combinations = 21

theorem wizard_viable_combinations : wizard_combination_problem :=
by
  sorry

end wizard_viable_combinations_l155_155810


namespace books_left_after_sale_l155_155924

theorem books_left_after_sale (initial_books sold_books books_left : ℕ)
    (h1 : initial_books = 33)
    (h2 : sold_books = 26)
    (h3 : books_left = initial_books - sold_books) :
    books_left = 7 := by
  sorry

end books_left_after_sale_l155_155924


namespace remove_terms_sum_equals_one_l155_155653

theorem remove_terms_sum_equals_one :
  let seq := [1/3, 1/6, 1/9, 1/12, 1/15, 1/18]
  let remove := [1/12, 1/15]
  (seq.sum - remove.sum) = 1 :=
by
  sorry

end remove_terms_sum_equals_one_l155_155653


namespace zero_in_interval_l155_155235

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := -2 * x + 3
noncomputable def h (x : ℝ) : ℝ := f x + 2 * x - 3

theorem zero_in_interval : ∃ x : ℝ, x ∈ Set.Ioo (1/2 : ℝ) 1 ∧ h x = 0 := 
sorry

end zero_in_interval_l155_155235


namespace part_a_l155_155436

-- Lean 4 statement equivalent to Part (a)
theorem part_a (n : ℕ) (x : ℝ) (hn : 0 < n) (hx : n^2 ≤ x) : 
  n * Real.sqrt (x - n^2) ≤ x / 2 := 
sorry

-- Lean 4 statement equivalent to Part (b)
noncomputable def find_xyz : ℕ × ℕ × ℕ :=
  ((2, 8, 18) : ℕ × ℕ × ℕ)

end part_a_l155_155436


namespace real_solutions_quadratic_l155_155425

theorem real_solutions_quadratic (a : ℝ) : 
  (∃ x : ℝ, x^2 - 4 * x + a = 0) ↔ a ≤ 4 :=
by sorry

end real_solutions_quadratic_l155_155425


namespace quadratic_value_at_point_l155_155511

theorem quadratic_value_at_point :
  ∃ a b c, 
    (∃ y, y = a * 2^2 + b * 2 + c ∧ y = 7) ∧
    (∃ y, y = a * 0^2 + b * 0 + c ∧ y = -7) ∧
    (∃ y, y = a * 5^2 + b * 5 + c ∧ y = -24.5) := 
sorry

end quadratic_value_at_point_l155_155511


namespace greatest_possible_xy_value_l155_155879

-- Define the conditions
variables (a b c d x y : ℕ)
variables (h1 : a < b) (h2 : b < c) (h3 : c < d)
variables (sums : Finset ℕ) (hsums : sums = {189, 320, 287, 234, x, y})

-- Define the goal statement to prove
theorem greatest_possible_xy_value : x + y = 791 :=
sorry

end greatest_possible_xy_value_l155_155879


namespace number_of_common_tangents_l155_155904

noncomputable def circle1_center : ℝ × ℝ := (-3, 0)
noncomputable def circle1_radius : ℝ := 4

noncomputable def circle2_center : ℝ × ℝ := (0, 3)
noncomputable def circle2_radius : ℝ := 6

theorem number_of_common_tangents 
  (center1 center2 : ℝ × ℝ)
  (radius1 radius2 : ℝ)
  (h_center1: center1 = (-3, 0))
  (h_radius1: radius1 = 4)
  (h_center2: center2 = (0, 3))
  (h_radius2: radius2 = 6) :
  -- The sought number of common tangents between the two circles
  2 = 2 :=
by
  sorry

end number_of_common_tangents_l155_155904


namespace ladder_wood_sufficiency_l155_155145

theorem ladder_wood_sufficiency
  (total_wood : ℝ)
  (rung_length_in: ℝ)
  (rung_distance_in: ℝ)
  (ladder_height_ft: ℝ)
  (total_wood_ft : total_wood = 300)
  (rung_length_ft : rung_length_in = 18 / 12)
  (rung_distance_ft : rung_distance_in = 6 / 12)
  (ladder_height_ft : ladder_height_ft = 50) :
  (∃ wood_needed : ℝ, wood_needed ≤ total_wood ∧ total_wood - wood_needed = 162.5) :=
sorry

end ladder_wood_sufficiency_l155_155145


namespace number_of_students_exclusively_in_math_l155_155178

variable (T M F K : ℕ)
variable (students_in_math students_in_foreign_language students_only_music : ℕ)
variable (students_not_in_music total_students_only_non_music : ℕ)

theorem number_of_students_exclusively_in_math (hT: T = 120) (hM: M = 82)
    (hF: F = 71) (hK: K = 20) :
    T - K = 100 →
    (M + F - 53 = T - K) →
    M - 53 = 29 :=
by
  intros
  sorry

end number_of_students_exclusively_in_math_l155_155178


namespace percent_games_lost_l155_155483

def games_ratio (won lost : ℕ) : Prop :=
  won * 3 = lost * 7

def total_games (won lost : ℕ) : Prop :=
  won + lost = 50

def percentage_lost (lost total : ℕ) : ℕ :=
  lost * 100 / total

theorem percent_games_lost (won lost : ℕ) (h1 : games_ratio won lost) (h2 : total_games won lost) : 
  percentage_lost lost 50 = 30 := 
by
  sorry

end percent_games_lost_l155_155483


namespace find_l_find_C3_l155_155288

-- Circle definitions
def C1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 2*y + 7 = 0

-- Given line passes through common points of C1 and C2
theorem find_l (x y : ℝ) (h1 : C1 x y) (h2 : C2 x y) : x = 1 := by
  sorry

-- Circle C3 passes through intersection points of C1 and C2, and its center lies on y = x
def C3 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def on_line_y_eq_x (x y : ℝ) : Prop := y = x

theorem find_C3 (x y : ℝ) (hx : C3 x y) (hy : on_line_y_eq_x x y) : (x - 1)^2 + (y - 1)^2 = 1 := by
  sorry

end find_l_find_C3_l155_155288


namespace focus_of_hyperbola_l155_155259

theorem focus_of_hyperbola (m : ℝ) :
  let focus_parabola := (0, 4)
  let focus_hyperbola_upper := (0, 4)
  ∃ focus_parabola, ∃ focus_hyperbola_upper, 
    (focus_parabola = (0, 4)) ∧ (focus_hyperbola_upper = (0, 4)) ∧ 
    (3 + m = 16) → m = 13 :=
by
  sorry

end focus_of_hyperbola_l155_155259


namespace no_base_450_odd_last_digit_l155_155986

theorem no_base_450_odd_last_digit :
  ¬ ∃ b : ℕ, b^3 ≤ 450 ∧ 450 < b^4 ∧ (450 % b) % 2 = 1 :=
sorry

end no_base_450_odd_last_digit_l155_155986


namespace part1_part2_l155_155530

-- Definition of the function
def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * x - 1

-- Theorem for part (1)
theorem part1 
  (m n : ℝ)
  (h1 : ∀ x : ℝ, f x m < 0 ↔ -2 < x ∧ x < n) : 
  m = 3 / 2 ∧ n = 1 / 2 :=
sorry

-- Theorem for part (2)
theorem part2 
  (m : ℝ)
  (h2 : ∀ x : ℝ, m ≤ x ∧ x ≤ m + 1 → f x m < 0) : 
  -Real.sqrt 2 / 2 < m ∧ m < 0 :=
sorry

end part1_part2_l155_155530


namespace actual_distance_between_city_centers_l155_155223

-- Define the conditions
def map_distance_cm : ℝ := 45
def scale_cm_to_km : ℝ := 10

-- Define the proof statement
theorem actual_distance_between_city_centers
  (md : ℝ := map_distance_cm)
  (scale : ℝ := scale_cm_to_km) :
  md * scale = 450 :=
by
  sorry

end actual_distance_between_city_centers_l155_155223


namespace quadratic_function_proof_l155_155867

theorem quadratic_function_proof (a c : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * x^2 - 4 * x + c)
  (h_sol_set : ∀ x, f x < 0 → (-1 < x ∧ x < 5)) :
  (a = 1 ∧ c = -5) ∧ (∀ x, 0 ≤ x ∧ x ≤ 3 → -9 ≤ f x ∧ f x ≤ -5) :=
by
  sorry

end quadratic_function_proof_l155_155867


namespace city_population_l155_155636

theorem city_population (p : ℝ) (hp : 0.85 * (p + 2000) = p + 2050) : p = 2333 :=
by
  sorry

end city_population_l155_155636


namespace solution_set_ineq_l155_155153

theorem solution_set_ineq (x : ℝ) : (1 < x ∧ x ≤ 3) ↔ (x - 3) / (x - 1) ≤ 0 := sorry

end solution_set_ineq_l155_155153


namespace vector_addition_example_l155_155527

def vector_addition (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)

theorem vector_addition_example : vector_addition (1, -1) (-1, 2) = (0, 1) := 
by 
  unfold vector_addition 
  simp
  sorry

end vector_addition_example_l155_155527


namespace minimum_d_exists_l155_155189

open Nat

theorem minimum_d_exists :
  ∃ (a b c d e f g h i k : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ k ∧
                                b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ k ∧
                                c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ k ∧
                                d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ k ∧
                                e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ k ∧
                                f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ k ∧
                                g ≠ h ∧ g ≠ i ∧ g ≠ k ∧
                                h ≠ i ∧ h ≠ k ∧
                                i ≠ k ∧
                                d = a + 3 * (e + h) + k ∧
                                d = 20 :=
by
  sorry

end minimum_d_exists_l155_155189


namespace exists_2013_distinct_numbers_l155_155707

theorem exists_2013_distinct_numbers : 
  ∃ (a : ℕ → ℕ), 
    (∀ m n, m ≠ n → m < 2013 ∧ n < 2013 → (a m + a n) % (a m - a n) = 0) ∧
    (∀ k l, k < 2013 ∧ l < 2013 → (a k) ≠ (a l)) :=
sorry

end exists_2013_distinct_numbers_l155_155707


namespace hall_breadth_l155_155639

theorem hall_breadth (l : ℝ) (w_s l_s b : ℝ) (n : ℕ)
  (hall_length : l = 36)
  (stone_width : w_s = 0.4)
  (stone_length : l_s = 0.5)
  (num_stones : n = 2700)
  (area_paving : l * b = n * (w_s * l_s)) :
  b = 15 := by
  sorry

end hall_breadth_l155_155639


namespace vector_dot_product_problem_l155_155567

variables {a b : ℝ}

theorem vector_dot_product_problem (h1 : a + 2 * b = 0) (h2 : (a + b) * a = 2) : a * b = -2 :=
sorry

end vector_dot_product_problem_l155_155567


namespace side_length_of_largest_square_l155_155296

theorem side_length_of_largest_square (S : ℝ) 
  (h1 : 2 * (S / 2)^2 + 2 * (S / 4)^2 = 810) : S = 36 :=
by
  -- proof steps go here
  sorry

end side_length_of_largest_square_l155_155296


namespace union_complement_B_A_equals_a_values_l155_155275

namespace ProofProblem

-- Define the universal set R as real numbers
def R := Set ℝ

-- Define set A and set B as per the conditions
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- Complement of B in R
def complement_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 9}

-- Union of complement of B with A
def union_complement_B_A : Set ℝ := complement_B ∪ A

-- The first statement to be proven
theorem union_complement_B_A_equals : 
  union_complement_B_A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} :=
by
  sorry

-- Define set C as per the conditions
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- The second statement to be proven
theorem a_values (a : ℝ) (h : C a ⊆ B) : 
  2 ≤ a ∧ a ≤ 8 :=
by
  sorry

end ProofProblem

end union_complement_B_A_equals_a_values_l155_155275


namespace Iggy_Tuesday_Run_l155_155586

def IggyRunsOnTuesday (total_miles : ℕ) (monday_miles : ℕ) (wednesday_miles : ℕ) (thursday_miles : ℕ) (friday_miles : ℕ) : ℕ :=
  total_miles - (monday_miles + wednesday_miles + thursday_miles + friday_miles)

theorem Iggy_Tuesday_Run :
  let monday_miles := 3
  let wednesday_miles := 6
  let thursday_miles := 8
  let friday_miles := 3
  let total_miles := 240 / 10
  IggyRunsOnTuesday total_miles monday_miles wednesday_miles thursday_miles friday_miles = 4 :=
by
  sorry

end Iggy_Tuesday_Run_l155_155586


namespace man_owns_fraction_of_business_l155_155918

theorem man_owns_fraction_of_business
  (x : ℚ)
  (H1 : (3 / 4) * (x * 90000) = 45000)
  (H2 : x * 90000 = y) : 
  x = 2 / 3 := 
by
  sorry

end man_owns_fraction_of_business_l155_155918


namespace harry_geckos_count_l155_155656

theorem harry_geckos_count 
  (G : ℕ)
  (iguanas : ℕ := 2)
  (snakes : ℕ := 4)
  (cost_snake : ℕ := 10)
  (cost_iguana : ℕ := 5)
  (cost_gecko : ℕ := 15)
  (annual_cost : ℕ := 1140) :
  12 * (snakes * cost_snake + iguanas * cost_iguana + G * cost_gecko) = annual_cost → 
  G = 3 := 
by 
  intros h
  sorry

end harry_geckos_count_l155_155656


namespace donald_paul_ratio_l155_155791

-- Let P be the number of bottles Paul drinks in one day.
-- Let D be the number of bottles Donald drinks in one day.
def paul_bottles (P : ℕ) := P = 3
def donald_bottles (D : ℕ) := D = 9

theorem donald_paul_ratio (P D : ℕ) (hP : paul_bottles P) (hD : donald_bottles D) : D / P = 3 :=
by {
  -- Insert proof steps here using the conditions.
  sorry
}

end donald_paul_ratio_l155_155791


namespace slope_of_given_line_l155_155743

def slope_of_line (l : String) : Real :=
  -- Assuming that we have a function to parse the line equation
  -- and extract its slope. Normally, this would be a complex parsing function.
  1 -- Placeholder, as the slope calculation logic is trivial here.

theorem slope_of_given_line : slope_of_line "x - y - 1 = 0" = 1 := by
  sorry

end slope_of_given_line_l155_155743


namespace cougar_sleep_hours_l155_155198

-- Definitions
def total_sleep_hours (C Z : Nat) : Prop :=
  C + Z = 70

def zebra_cougar_difference (C Z : Nat) : Prop :=
  Z = C + 2

-- Theorem statement
theorem cougar_sleep_hours :
  ∃ C : Nat, ∃ Z : Nat, zebra_cougar_difference C Z ∧ total_sleep_hours C Z ∧ C = 34 :=
sorry

end cougar_sleep_hours_l155_155198


namespace tagged_fish_ratio_l155_155470

theorem tagged_fish_ratio (tagged_first_catch : ℕ) (total_second_catch : ℕ) (tagged_second_catch : ℕ) 
  (approx_total_fish : ℕ) (h1 : tagged_first_catch = 60) 
  (h2 : total_second_catch = 50) 
  (h3 : tagged_second_catch = 2) 
  (h4 : approx_total_fish = 1500) :
  tagged_second_catch / total_second_catch = 1 / 25 := by
  sorry

end tagged_fish_ratio_l155_155470


namespace find_c_exactly_two_common_points_l155_155419

theorem find_c_exactly_two_common_points (c : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^3 - 3*x1 + c = 0) ∧ (x2^3 - 3*x2 + c = 0)) ↔ (c = -2 ∨ c = 2) := 
sorry

end find_c_exactly_two_common_points_l155_155419


namespace AM_GM_inequality_l155_155462

theorem AM_GM_inequality (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) :
  (a^n + b^n) / 2 ≥ ((a + b) / 2) ^ n :=
by
  sorry

end AM_GM_inequality_l155_155462


namespace largest_angle_in_pentagon_l155_155267

-- Define the angles and sum condition
variables (x : ℝ) {P Q R S T : ℝ}

-- Conditions
def angle_P : P = 90 := sorry
def angle_Q : Q = 70 := sorry
def angle_R : R = x := sorry
def angle_S : S = x := sorry
def angle_T : T = 2*x + 20 := sorry
def sum_of_angles : P + Q + R + S + T = 540 := sorry

-- Prove the largest angle
theorem largest_angle_in_pentagon (hP : P = 90) (hQ : Q = 70)
    (hR : R = x) (hS : S = x) (hT : T = 2*x + 20) 
    (h_sum : P + Q + R + S + T = 540) : T = 200 :=
by
  sorry

end largest_angle_in_pentagon_l155_155267


namespace smallest_value_of_diff_l155_155238

-- Definitions of the side lengths from the conditions
def XY (x : ℝ) := x + 6
def XZ (x : ℝ) := 4 * x - 1
def YZ (x : ℝ) := x + 10

-- Conditions derived from the problem
noncomputable def valid_x (x : ℝ) := x > 5 / 3 ∧ x < 11 / 3

-- The proof statement
theorem smallest_value_of_diff : 
  ∀ (x : ℝ), valid_x x → (YZ x - XY x) = 4 :=
by
  intros x hx
  -- Proof goes here
  sorry

end smallest_value_of_diff_l155_155238


namespace part_one_part_two_l155_155579

variable (α : Real) (h : Real.tan α = 2)

theorem part_one (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 6 / 11 := 
by
  sorry

theorem part_two (h : Real.tan α = 2) : 
  (1 / 4 * Real.sin α ^ 2 + 1 / 3 * Real.sin α * Real.cos α + 1 / 2 * Real.cos α ^ 2 + 1) = 43 / 30 := 
by
  sorry

end part_one_part_two_l155_155579


namespace rita_bought_four_pounds_l155_155737

def initial_amount : ℝ := 70
def cost_per_pound : ℝ := 8.58
def left_amount : ℝ := 35.68

theorem rita_bought_four_pounds :
  (initial_amount - left_amount) / cost_per_pound = 4 :=
by
  sorry

end rita_bought_four_pounds_l155_155737


namespace find_f_inv_128_l155_155308

noncomputable def f : ℕ → ℕ := sorry

axiom f_at_5 : f 5 = 2
axiom f_doubling : ∀ x : ℕ, f (2 * x) = 2 * f x

theorem find_f_inv_128 : f 320 = 128 :=
by sorry

end find_f_inv_128_l155_155308


namespace price_of_each_brownie_l155_155384

variable (B : ℝ)

theorem price_of_each_brownie (h : 4 * B + 10 + 28 = 50) : B = 3 := by
  -- proof steps would go here
  sorry

end price_of_each_brownie_l155_155384


namespace maximum_squares_formation_l155_155451

theorem maximum_squares_formation (total_matchsticks : ℕ) (triangles : ℕ) (used_for_triangles : ℕ) (remaining_matchsticks : ℕ) (squares : ℕ):
  total_matchsticks = 24 →
  triangles = 6 →
  used_for_triangles = 13 →
  remaining_matchsticks = total_matchsticks - used_for_triangles →
  squares = remaining_matchsticks / 4 →
  squares = 4 :=
by
  sorry

end maximum_squares_formation_l155_155451


namespace q_f_digit_div_36_l155_155216

theorem q_f_digit_div_36 (q f : ℕ) (hq : q ≠ f) (hq_digit: q < 10) (hf_digit: f < 10) :
    (457 * 10000 + q * 1000 + 89 * 10 + f) % 36 = 0 → q + f = 6 :=
sorry

end q_f_digit_div_36_l155_155216


namespace average_of_multiples_of_6_l155_155702

def first_n_multiples_sum (n : ℕ) : ℕ :=
  (n * (6 + 6 * n)) / 2

def first_n_multiples_avg (n : ℕ) : ℕ :=
  (first_n_multiples_sum n) / n

theorem average_of_multiples_of_6 (n : ℕ) : first_n_multiples_avg n = 66 → n = 11 := by
  sorry

end average_of_multiples_of_6_l155_155702


namespace symmetric_point_to_origin_l155_155929

theorem symmetric_point_to_origin (a b : ℝ) :
  (∃ (a b : ℝ), (a / 2) - 2 * (b / 2) + 2 = 0 ∧ (b / a) * (1 / 2) = -1) →
  (a = -4 / 5 ∧ b = 8 / 5) :=
sorry

end symmetric_point_to_origin_l155_155929


namespace temperature_at_midnight_l155_155552

theorem temperature_at_midnight 
  (morning_temp : ℝ) 
  (afternoon_rise : ℝ) 
  (midnight_drop : ℝ)
  (h1 : morning_temp = 30)
  (h2 : afternoon_rise = 1)
  (h3 : midnight_drop = 7) 
  : morning_temp + afternoon_rise - midnight_drop = 24 :=
by
  -- Convert all conditions into the correct forms
  rw [h1, h2, h3]
  -- Perform the arithmetic operations
  norm_num

end temperature_at_midnight_l155_155552


namespace cos_angle_identity_l155_155219

theorem cos_angle_identity (α : ℝ) (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 17 * π / 12) = 1 / 3 :=
by
  sorry

end cos_angle_identity_l155_155219


namespace top_width_of_channel_l155_155370

theorem top_width_of_channel (b : ℝ) (A : ℝ) (h : ℝ) (w : ℝ) : 
  b = 8 ∧ A = 700 ∧ h = 70 ∧ (A = (1/2) * (w + b) * h) → w = 12 := 
by 
  intro h1
  sorry

end top_width_of_channel_l155_155370


namespace find_room_length_l155_155798

theorem find_room_length (w : ℝ) (A : ℝ) (h_w : w = 8) (h_A : A = 96) : (A / w = 12) :=
by
  rw [h_w, h_A]
  norm_num

end find_room_length_l155_155798


namespace solution_pairs_l155_155291

theorem solution_pairs (x y : ℝ) : 
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) := 
by 
  sorry

end solution_pairs_l155_155291


namespace linda_age_13_l155_155718

variable (J L : ℕ)

-- Conditions: 
-- 1. Linda is 3 more than 2 times the age of Jane.
-- 2. In five years, the sum of their ages will be 28.
def conditions (J L : ℕ) : Prop :=
  L = 2 * J + 3 ∧ (J + 5) + (L + 5) = 28

-- Question/answer to prove: Linda's current age is 13.
theorem linda_age_13 (J L : ℕ) (h : conditions J L) : L = 13 :=
by
  sorry

end linda_age_13_l155_155718


namespace continuity_at_x_2_l155_155967

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x > 2 then x + 4 else 3 * x + b

theorem continuity_at_x_2 (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) ↔ b = 0 :=
by
  sorry

end continuity_at_x_2_l155_155967


namespace mary_rental_hours_l155_155266

def ocean_bike_fixed_fee := 17
def ocean_bike_hourly_rate := 7
def total_paid := 80

def calculate_hours (fixed_fee : Nat) (hourly_rate : Nat) (total_amount : Nat) : Nat :=
  (total_amount - fixed_fee) / hourly_rate

theorem mary_rental_hours :
  calculate_hours ocean_bike_fixed_fee ocean_bike_hourly_rate total_paid = 9 :=
by
  sorry

end mary_rental_hours_l155_155266


namespace problem1_problem2_l155_155128

-- Proof Problem 1:

theorem problem1 : (5 / 3) ^ 2004 * (3 / 5) ^ 2003 = 5 / 3 := by
  sorry

-- Proof Problem 2:

theorem problem2 (x : ℝ) (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 := by
  sorry

end problem1_problem2_l155_155128


namespace triangle_hypotenuse_and_area_l155_155878

theorem triangle_hypotenuse_and_area 
  (A B C D : Type) 
  (CD : ℝ) 
  (angle_A : ℝ) 
  (hypotenuse_AC : ℝ) 
  (area_ABC : ℝ) 
  (h1 : CD = 1) 
  (h2 : angle_A = 45) : 
  hypotenuse_AC = Real.sqrt 2 
  ∧ 
  area_ABC = 1 / 2 := 
by
  sorry

end triangle_hypotenuse_and_area_l155_155878


namespace bank_balance_after_2_years_l155_155073

noncomputable def compound_interest (P₀ : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P₀ * (1 + r)^n

theorem bank_balance_after_2_years :
  compound_interest 100 0.10 2 = 121 := 
  by
  sorry

end bank_balance_after_2_years_l155_155073


namespace sum_of_variables_l155_155704

theorem sum_of_variables (x y z w : ℤ) 
(h1 : x - y + z = 7) 
(h2 : y - z + w = 8) 
(h3 : z - w + x = 4) 
(h4 : w - x + y = 3) : 
x + y + z + w = 11 := 
sorry

end sum_of_variables_l155_155704


namespace initial_volume_of_mixture_l155_155876

-- Define the initial condition volumes for p and q
def initial_volumes (x : ℕ) : ℕ × ℕ := (3 * x, 2 * x)

-- Define the final condition volumes for p and q after adding 2 liters of q
def final_volumes (x : ℕ) : ℕ × ℕ := (3 * x, 2 * x + 2)

-- Define the initial total volume of the mixture
def initial_volume (x : ℕ) : ℕ := 5 * x

-- The theorem stating the solution
theorem initial_volume_of_mixture (x : ℕ) (h : 3 * x / (2 * x + 2) = 5 / 4) : 5 * x = 25 := 
by sorry

end initial_volume_of_mixture_l155_155876


namespace problem_ineq_l155_155813

theorem problem_ineq (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
(h4 : x * y * z = 1) :
    (x^3 / ((1 + y)*(1 + z)) + y^3 / ((1 + z)*(1 + x)) + z^3 / ((1 + x)*(1 + y))) ≥ 3 / 4 := 
sorry

end problem_ineq_l155_155813


namespace integer_roots_of_quadratic_l155_155750

theorem integer_roots_of_quadratic (a : ℤ) : 
  (∃ x : ℤ , x^2 + a * x + a = 0) ↔ (a = 0 ∨ a = 4) := 
sorry

end integer_roots_of_quadratic_l155_155750


namespace algebra_expression_value_l155_155870

theorem algebra_expression_value
  (x y : ℝ)
  (h : x - 2 * y + 2 = 5) : 4 * y - 2 * x + 1 = -5 :=
by sorry

end algebra_expression_value_l155_155870


namespace total_valid_arrangements_l155_155711

-- Define the students and schools
inductive Student
| G1 | G2 | B1 | B2 | B3 | BA
deriving DecidableEq

inductive School
| A | B | C
deriving DecidableEq

-- Define the condition that any two students cannot be in the same school
def is_valid_arrangement (arr : School → Student → Bool) : Bool :=
  (arr School.A Student.G1 ≠ arr School.A Student.G2) ∧ 
  (arr School.B Student.G1 ≠ arr School.B Student.G2) ∧
  (arr School.C Student.G1 ≠ arr School.C Student.G2) ∧
  ¬ arr School.C Student.G1 ∧
  ¬ arr School.C Student.G2 ∧
  ¬ arr School.A Student.BA

-- The theorem to prove the total number of different valid arrangements
theorem total_valid_arrangements : 
  ∃ n : ℕ, n = 18 ∧ ∃ arr : (School → Student → Bool), is_valid_arrangement arr := 
sorry

end total_valid_arrangements_l155_155711


namespace locus_of_Q_max_area_of_triangle_OPQ_l155_155595

open Real

theorem locus_of_Q (x y : ℝ) (x_0 y_0 : ℝ) :
  (x_0 / 4)^2 + (y_0 / 3)^2 = 1 ∧
  x = 3 * x_0 ∧ y = 4 * y_0 →
  (x / 6)^2 + (y / 4)^2 = 1 :=
sorry

theorem max_area_of_triangle_OPQ (S : ℝ) (x_0 y_0 : ℝ) :
  (x_0 / 4)^2 + (y_0 / 3)^2 = 1 ∧
  x_0 > 0 ∧ y_0 > 0 →
  S <= sqrt 3 / 2 :=
sorry

end locus_of_Q_max_area_of_triangle_OPQ_l155_155595


namespace A_contribution_is_500_l155_155598

-- Define the contributions
variables (A B C : ℕ)

-- Total amount spent
def total_contribution : ℕ := 820

-- Given ratios
def ratio_A_to_B : ℕ × ℕ := (5, 2)
def ratio_B_to_C : ℕ × ℕ := (5, 3)

-- Condition stating the sum of contributions
axiom sum_contribution : A + B + C = total_contribution

-- Conditions stating the ratios
axiom ratio_A_B : 5 * B = 2 * A
axiom ratio_B_C : 5 * C = 3 * B

-- The statement to prove
theorem A_contribution_is_500 : A = 500 :=
by
  sorry

end A_contribution_is_500_l155_155598


namespace simplify_fraction_l155_155602

noncomputable def simplify_expression (x : ℂ) : Prop :=
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) =
  (x - 3) / (x^2 - 6*x + 8)

theorem simplify_fraction (x : ℂ) : simplify_expression x :=
by
  sorry

end simplify_fraction_l155_155602


namespace alarm_prob_l155_155809

theorem alarm_prob (pA pB : ℝ) (hA : pA = 0.80) (hB : pB = 0.90) : 
  (1 - (1 - pA) * (1 - pB)) = 0.98 :=
by 
  sorry

end alarm_prob_l155_155809


namespace total_price_of_bananas_and_oranges_l155_155695

variable (price_orange price_pear price_banana : ℝ)

axiom total_cost_orange_pear : price_orange + price_pear = 120
axiom cost_pear : price_pear = 90
axiom diff_orange_pear_banana : price_orange - price_pear = price_banana

theorem total_price_of_bananas_and_oranges :
  let num_bananas := 200
  let num_oranges := 2 * num_bananas
  let cost_bananas := num_bananas * price_banana
  let cost_oranges := num_oranges * price_orange
  cost_bananas + cost_oranges = 24000 :=
by
  sorry

end total_price_of_bananas_and_oranges_l155_155695


namespace minimum_filtrations_needed_l155_155931

theorem minimum_filtrations_needed (I₀ I_n : ℝ) (n : ℕ) (h1 : I₀ = 0.02) (h2 : I_n ≤ 0.001) (h3 : I_n = I₀ * 0.5 ^ n) :
  n = 8 := by
sorry

end minimum_filtrations_needed_l155_155931


namespace min_positive_integer_expression_l155_155079

theorem min_positive_integer_expression : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → (m: ℝ) / 3 + 27 / (m: ℝ) ≥ (n: ℝ) / 3 + 27 / (n: ℝ)) ∧ (n / 3 + 27 / n = 6) :=
sorry

end min_positive_integer_expression_l155_155079


namespace factorize_expression_l155_155883

theorem factorize_expression (x y : ℝ) : 25 * x - x * y ^ 2 = x * (5 + y) * (5 - y) := by
  sorry

end factorize_expression_l155_155883


namespace percentage_first_less_third_l155_155002

variable (A B C : ℝ)

theorem percentage_first_less_third :
  B = 0.58 * C → B = 0.8923076923076923 * A → (100 - (A / C * 100)) = 35 :=
by
  intros h₁ h₂
  sorry

end percentage_first_less_third_l155_155002


namespace petya_cannot_have_equal_coins_l155_155946

theorem petya_cannot_have_equal_coins
  (transact : ℕ → ℕ)
  (initial_two_kopeck : ℕ)
  (total_operations : ℕ)
  (insertion_machine : ℕ)
  (by_insert_two : ℕ)
  (by_insert_ten : ℕ)
  (odd : ℕ)
  :
  (initial_two_kopeck = 1) ∧ 
  (by_insert_two = 5) ∧ 
  (by_insert_ten = 5) ∧
  (∀ n, transact n = 1 + 4 * n) →
  (odd % 2 = 1) →
  (total_operations = transact insertion_machine) →
  (total_operations % 2 = 1) →
  (∀ x y, (x + y = total_operations) → (x = y) → False) :=
sorry

end petya_cannot_have_equal_coins_l155_155946


namespace only_valid_M_l155_155094

def digit_sum (n : ℕ) : ℕ :=
  -- definition of digit_sum as a function summing up digits of n
  sorry 

def is_valid_M (M : ℕ) := 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ M → digit_sum (M * k) = digit_sum M

theorem only_valid_M (M : ℕ) :
  is_valid_M M ↔ ∃ n : ℕ, ∀ m : ℕ, M = 10^n - 1 :=
by
  sorry

end only_valid_M_l155_155094


namespace decreasing_interval_l155_155603

noncomputable def f (x : ℝ) : ℝ := x / 2 + Real.cos x

theorem decreasing_interval : ∀ x ∈ Set.Ioo (Real.pi / 6) (5 * Real.pi / 6), 
  (1 / 2 - Real.sin x) < 0 := sorry

end decreasing_interval_l155_155603


namespace N_vector_3_eq_result_vector_l155_155821

noncomputable def matrix_N : Matrix (Fin 2) (Fin 2) ℝ :=
-- The matrix N is defined such that:
-- N * (vector 3 -2) = (vector 4 1)
-- N * (vector -2 3) = (vector 1 2)
sorry

def vector_1 : Fin 2 → ℝ := fun | ⟨0,_⟩ => 3 | ⟨1,_⟩ => -2
def vector_2 : Fin 2 → ℝ := fun | ⟨0,_⟩ => -2 | ⟨1,_⟩ => 3
def vector_3 : Fin 2 → ℝ := fun | ⟨0,_⟩ => 7 | ⟨1,_⟩ => 0
def result_vector : Fin 2 → ℝ := fun | ⟨0,_⟩ => 14 | ⟨1,_⟩ => 7

theorem N_vector_3_eq_result_vector :
  matrix_N.mulVec vector_3 = result_vector := by
  -- Given conditions:
  -- matrix_N.mulVec vector_1 = vector_4
  -- and matrix_N.mulVec vector_2 = vector_5
  sorry

end N_vector_3_eq_result_vector_l155_155821


namespace minimum_value_of_quadratic_expression_l155_155721

theorem minimum_value_of_quadratic_expression (x y z : ℝ)
  (h : x + y + z = 2) : 
  x^2 + 2 * y^2 + z^2 ≥ 4 / 3 :=
sorry

end minimum_value_of_quadratic_expression_l155_155721


namespace avg_scores_relation_l155_155759

variables (class_avg top8_avg other32_avg : ℝ)

theorem avg_scores_relation (h1 : 40 = 40) 
  (h2 : top8_avg = class_avg + 3) :
  other32_avg = top8_avg - 3.75 :=
sorry

end avg_scores_relation_l155_155759


namespace inequality_solution_l155_155494

theorem inequality_solution (x : ℝ) (h : (x + 1) / 2 ≥ x / 3) : x ≥ -3 :=
by
  sorry

end inequality_solution_l155_155494


namespace FirstCandidatePercentage_l155_155244

noncomputable def percentage_of_first_candidate_marks (PassingMarks TotalMarks MarksFirstCandidate : ℝ) :=
  (MarksFirstCandidate / TotalMarks) * 100

theorem FirstCandidatePercentage 
  (PassingMarks TotalMarks MarksFirstCandidate : ℝ)
  (h1 : PassingMarks = 200)
  (h2 : 0.45 * TotalMarks = PassingMarks + 25)
  (h3 : MarksFirstCandidate = PassingMarks - 50)
  : percentage_of_first_candidate_marks PassingMarks TotalMarks MarksFirstCandidate = 30 :=
sorry

end FirstCandidatePercentage_l155_155244


namespace system_solution_l155_155416

theorem system_solution (x y : ℝ) 
  (h1 : 0 < x + y) 
  (h2 : x + y ≠ 1) 
  (h3 : 2 * x - y ≠ 0)
  (eq1 : (x + y) * 2^(y - 2 * x) = 6.25) 
  (eq2 : (x + y) * (1 / (2 * x - y)) = 5) :
x = 9 ∧ y = 16 := 
sorry

end system_solution_l155_155416


namespace petya_time_comparison_l155_155650

theorem petya_time_comparison (V : ℝ) (a : ℝ) (hV_pos : 0 < V) (ha_pos : 0 < a) :
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  T_real > T_planned :=
by
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  have T_real_eq : T_real = (a / (1.25 * V)) + (a / (0.8 * V)) := by sorry
  have T_real_simplified : T_real = (2 * a / 5 / V) + (5 * a / 8 / V) := by sorry
  have T_real_combined : T_real = (41 * a) / (40 * V) := by sorry
  have planned_vs_real : T_real > a / V := by sorry
  exact planned_vs_real

end petya_time_comparison_l155_155650


namespace graph_not_in_first_quadrant_l155_155706

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2)^x - 2

-- Prove that the graph of f(x) does not pass through the first quadrant
theorem graph_not_in_first_quadrant : ∀ (x : ℝ), x > 0 → f x ≤ 0 := by
  intro x hx
  sorry

end graph_not_in_first_quadrant_l155_155706


namespace Sally_next_birthday_age_l155_155862

variables (a m s d : ℝ)

def Adam_older_than_Mary := a = 1.3 * m
def Mary_younger_than_Sally := m = 0.75 * s
def Sally_younger_than_Danielle := s = 0.8 * d
def Sum_ages := a + m + s + d = 60

theorem Sally_next_birthday_age (a m s d : ℝ) 
  (H1 : Adam_older_than_Mary a m)
  (H2 : Mary_younger_than_Sally m s)
  (H3 : Sally_younger_than_Danielle s d)
  (H4 : Sum_ages a m s d) : 
  s + 1 = 16 := 
by sorry

end Sally_next_birthday_age_l155_155862


namespace pollen_particle_diameter_in_scientific_notation_l155_155177

theorem pollen_particle_diameter_in_scientific_notation :
  0.0000078 = 7.8 * 10^(-6) :=
by
  sorry

end pollen_particle_diameter_in_scientific_notation_l155_155177


namespace polynomial_remainder_correct_l155_155366

noncomputable def remainder_polynomial (x : ℝ) : ℝ := x ^ 100

def divisor_polynomial (x : ℝ) : ℝ := x ^ 2 - 3 * x + 2

def polynomial_remainder (x : ℝ) : ℝ := 2 ^ 100 * (x - 1) - (x - 2)

theorem polynomial_remainder_correct : ∀ x : ℝ, (remainder_polynomial x) % (divisor_polynomial x) = polynomial_remainder x := by
  sorry

end polynomial_remainder_correct_l155_155366


namespace calculate_sum_of_squares_l155_155756

theorem calculate_sum_of_squares :
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 268 := 
  sorry

end calculate_sum_of_squares_l155_155756


namespace find_some_number_l155_155758

theorem find_some_number (some_number : ℝ) :
  (0.0077 * some_number) / (0.04 * 0.1 * 0.007) = 990.0000000000001 → 
  some_number = 3.6 :=
by
  intro h
  sorry

end find_some_number_l155_155758


namespace repeating_decimal_to_fraction_l155_155526

theorem repeating_decimal_to_fraction : ∀ (x : ℝ), x = 0.7 + 0.08 / (1-0.1) → x = 71 / 90 :=
by
  intros x hx
  sorry

end repeating_decimal_to_fraction_l155_155526


namespace largest_multiple_of_15_who_negation_greater_than_neg_150_l155_155735

theorem largest_multiple_of_15_who_negation_greater_than_neg_150 : 
  ∃ (x : ℤ), x % 15 = 0 ∧ -x > -150 ∧ ∀ (y : ℤ), y % 15 = 0 ∧ -y > -150 → x ≥ y :=
by
  sorry

end largest_multiple_of_15_who_negation_greater_than_neg_150_l155_155735


namespace polynomial_evaluation_l155_155193

-- Given the value of y
def y : ℤ := 4

-- Our goal is to prove this mathematical statement
theorem polynomial_evaluation : (3 * (y ^ 2) + 4 * y + 2 = 66) := 
by 
    sorry

end polynomial_evaluation_l155_155193


namespace eddys_climbing_rate_l155_155884

def base_camp_ft := 5000
def departure_time := 6 -- in hours: 6:00 AM
def hillary_climbing_rate := 800 -- ft/hr
def stopping_distance_ft := 1000 -- ft short of summit
def hillary_descending_rate := 1000 -- ft/hr
def passing_time := 12 -- in hours: 12:00 PM

theorem eddys_climbing_rate :
  ∀ (base_ft departure hillary_rate stop_dist descend_rate pass_time : ℕ),
    base_ft = base_camp_ft →
    departure = departure_time →
    hillary_rate = hillary_climbing_rate →
    stop_dist = stopping_distance_ft →
    descend_rate = hillary_descending_rate →
    pass_time = passing_time →
    (pass_time - departure) * hillary_rate - descend_rate * (pass_time - (departure + (base_ft - stop_dist) / hillary_rate)) = 6 * 500 :=
by
  intros
  sorry

end eddys_climbing_rate_l155_155884


namespace yield_difference_correct_l155_155196

noncomputable def tomato_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)
noncomputable def corn_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)
noncomputable def onion_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)
noncomputable def carrot_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)

theorem yield_difference_correct :
  let tomato_initial := 2073
  let corn_initial := 4112
  let onion_initial := 985
  let carrot_initial := 6250
  let tomato_growth := 12
  let corn_growth := 15
  let onion_growth := 8
  let carrot_growth := 10
  let tomato_total := tomato_yield tomato_initial tomato_growth
  let corn_total := corn_yield corn_initial corn_growth
  let onion_total := onion_yield onion_initial onion_growth
  let carrot_total := carrot_yield carrot_initial carrot_growth
  let highest_yield := max (max tomato_total corn_total) (max onion_total carrot_total)
  let lowest_yield := min (min tomato_total corn_total) (min onion_total carrot_total)
  highest_yield - lowest_yield = 5811.2 := by
  sorry

end yield_difference_correct_l155_155196


namespace container_dimensions_l155_155833

theorem container_dimensions (a b c : ℝ) 
  (h1 : a * b * 16 = 2400)
  (h2 : a * c * 10 = 2400)
  (h3 : b * c * 9.6 = 2400) :
  a = 12 ∧ b = 12.5 ∧ c = 20 :=
by
  sorry

end container_dimensions_l155_155833


namespace min_value_abs_expr_l155_155560

noncomputable def minExpr (a b : ℝ) : ℝ :=
  |a + b| + |(1 / (a + 1)) - b|

theorem min_value_abs_expr (a b : ℝ) (h₁ : a ≠ -1) : minExpr a b ≥ 1 ∧ (minExpr a b = 1 ↔ a = 0) :=
by
  sorry

end min_value_abs_expr_l155_155560


namespace janina_spend_on_supplies_each_day_l155_155351

theorem janina_spend_on_supplies_each_day 
  (rent : ℝ)
  (p : ℝ)
  (n : ℕ)
  (H1 : rent = 30)
  (H2 : p = 2)
  (H3 : n = 21) :
  (n : ℝ) * p - rent = 12 := 
by
  sorry

end janina_spend_on_supplies_each_day_l155_155351


namespace correct_negation_statement_l155_155722

def Person : Type := sorry

def is_adult (p : Person) : Prop := sorry
def is_teenager (p : Person) : Prop := sorry
def is_responsible (p : Person) : Prop := sorry
def is_irresponsible (p : Person) : Prop := sorry

axiom all_adults_responsible : ∀ p, is_adult p → is_responsible p
axiom some_adults_responsible : ∃ p, is_adult p ∧ is_responsible p
axiom no_teenagers_responsible : ∀ p, is_teenager p → ¬is_responsible p
axiom all_teenagers_irresponsible : ∀ p, is_teenager p → is_irresponsible p
axiom exists_irresponsible_teenager : ∃ p, is_teenager p ∧ is_irresponsible p
axiom all_teenagers_responsible : ∀ p, is_teenager p → is_responsible p

theorem correct_negation_statement
: (∃ p, is_teenager p ∧ ¬is_responsible p) ↔ 
  (∃ p, is_teenager p ∧ is_irresponsible p) :=
sorry

end correct_negation_statement_l155_155722


namespace solve_for_k_l155_155475

theorem solve_for_k (x y k : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h : (1 / 2)^(25 * x) * (1 / 81)^k = 1 / (18 ^ (25 * y))) :
  k = 25 * y / 2 :=
by
  sorry

end solve_for_k_l155_155475


namespace eval_expression_l155_155403

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l155_155403


namespace possible_values_x2_y2_z2_l155_155437

theorem possible_values_x2_y2_z2 {x y z : ℤ}
    (h1 : x + y + z = 3)
    (h2 : x^3 + y^3 + z^3 = 3) : (x^2 + y^2 + z^2 = 3) ∨ (x^2 + y^2 + z^2 = 57) :=
by sorry

end possible_values_x2_y2_z2_l155_155437


namespace exists_a_b_l155_155381

theorem exists_a_b (S : Finset ℕ) (hS : S.card = 43) :
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ (a^2 - b^2) % 100 = 0 := 
by
  sorry

end exists_a_b_l155_155381


namespace domain_of_function_l155_155811

/-- The domain of the function \( y = \lg (12 + x - x^2) \) is the interval \(-3 < x < 4\). -/
theorem domain_of_function :
  {x : ℝ | 12 + x - x^2 > 0} = {x : ℝ | -3 < x ∧ x < 4} :=
sorry

end domain_of_function_l155_155811


namespace arithmetic_sequence_sum_condition_l155_155620

noncomputable def sum_first_n_terms (a_1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a_1 + (n * (n - 1)) / 2 * d

theorem arithmetic_sequence_sum_condition (a_1 d : ℤ) :
  sum_first_n_terms a_1 d 3 = 3 →
  sum_first_n_terms a_1 d 6 = 15 →
  (a_1 + 9 * d) + (a_1 + 10 * d) + (a_1 + 11 * d) = 30 :=
by
  intros h1 h2
  sorry

end arithmetic_sequence_sum_condition_l155_155620


namespace integral_value_l155_155239

theorem integral_value : ∫ x in (1:ℝ)..(2:ℝ), (x^2 + 1) / x = (3 / 2) + Real.log 2 :=
by sorry

end integral_value_l155_155239


namespace part1_l155_155047

def A (a : ℝ) : Set ℝ := {x | 2 * a - 3 < x ∧ x < a + 1}
def B : Set ℝ := {x | (x + 1) * (x - 3) < 0}

theorem part1 (a : ℝ) (h : a = 0) : A a ∩ B = {x | -1 < x ∧ x < 1} :=
by
  -- Proof here
  sorry

end part1_l155_155047


namespace smaller_number_is_25_l155_155873

theorem smaller_number_is_25 (x y : ℕ) (h1 : x + y = 62) (h2 : y = x + 12) : x = 25 :=
by sorry

end smaller_number_is_25_l155_155873


namespace smallest_positive_m_l155_155104

theorem smallest_positive_m (m : ℕ) :
  (∃ (r s : ℤ), 18 * r * s = 252 ∧ m = 18 * (r + s) ∧ r ≠ s) ∧ m > 0 →
  m = 162 := 
sorry

end smallest_positive_m_l155_155104


namespace variance_scaled_data_l155_155826

noncomputable def variance (data : List ℝ) : ℝ :=
  let n := data.length
  let mean := data.sum / n
  (data.map (λ x => (x - mean) ^ 2)).sum / n

theorem variance_scaled_data (data : List ℝ) (h_len : data.length > 0) (h_var : variance data = 4) :
  variance (data.map (λ x => 2 * x)) = 16 :=
by
  sorry

end variance_scaled_data_l155_155826


namespace reciprocal_inverse_proportional_l155_155747

variable {x y k c : ℝ}

-- Given condition: x * y = k
axiom inverse_proportional (h : x * y = k) : ∃ c, (1/x) * (1/y) = c

theorem reciprocal_inverse_proportional (h : x * y = k) :
  ∃ c, (1/x) * (1/y) = c :=
inverse_proportional h

end reciprocal_inverse_proportional_l155_155747


namespace minimum_ladder_rungs_l155_155024

theorem minimum_ladder_rungs (a b : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b): ∃ n, n = a + b - 1 :=
by
    sorry

end minimum_ladder_rungs_l155_155024


namespace range_of_k_for_distinct_real_roots_l155_155310

theorem range_of_k_for_distinct_real_roots (k : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (k - 1) * x1^2 - 2 * x1 + 1 = 0 ∧ (k - 1) * x2^2 - 2 * x2 + 1 = 0) →
    k < 2 ∧ k ≠ 1 :=
by
  sorry

end range_of_k_for_distinct_real_roots_l155_155310


namespace coordinates_of_P_l155_155913

structure Point (α : Type) [LinearOrderedField α] :=
  (x : α)
  (y : α)

def in_fourth_quadrant {α : Type} [LinearOrderedField α] (P : Point α) : Prop :=
  P.x > 0 ∧ P.y < 0

def distance_to_axes_is_4 {α : Type} [LinearOrderedField α] (P : Point α) : Prop :=
  abs P.x = 4 ∧ abs P.y = 4

theorem coordinates_of_P {α : Type} [LinearOrderedField α] (P : Point α) :
  in_fourth_quadrant P ∧ distance_to_axes_is_4 P → P = ⟨4, -4⟩ :=
by
  sorry

end coordinates_of_P_l155_155913


namespace quadratic_solution_property_l155_155015

theorem quadratic_solution_property :
  (∃ p q : ℝ, 3 * p^2 + 7 * p - 6 = 0 ∧ 3 * q^2 + 7 * q - 6 = 0 ∧ (p - 2) * (q - 2) = 6) :=
by
  sorry

end quadratic_solution_property_l155_155015


namespace ellipse_foci_on_y_axis_l155_155221

theorem ellipse_foci_on_y_axis (k : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (∀ x y, x^2 + k * y^2 = 2 ↔ x^2/a^2 + y^2/b^2 = 1) ∧ b^2 > a^2)
  → (0 < k ∧ k < 1) :=
sorry

end ellipse_foci_on_y_axis_l155_155221


namespace batsman_average_46_innings_l155_155997

variable (A : ℕ) (highest_score : ℕ) (lowest_score : ℕ) (average_excl : ℕ)
variable (n_innings n_without_highest_lowest : ℕ)

theorem batsman_average_46_innings
  (h_diff: highest_score - lowest_score = 190)
  (h_avg_excl: average_excl = 58)
  (h_highest: highest_score = 199)
  (h_innings: n_innings = 46)
  (h_innings_excl: n_without_highest_lowest = 44) :
  A = (44 * 58 + 199 + 9) / 46 := by
  sorry

end batsman_average_46_innings_l155_155997


namespace dorothy_will_be_twice_as_old_l155_155315

-- Define some variables
variables (D S Y : ℕ)

-- Hypothesis
def dorothy_age_condition (D S : ℕ) : Prop := D = 3 * S
def dorothy_current_age (D : ℕ) : Prop := D = 15

-- Theorems we want to prove
theorem dorothy_will_be_twice_as_old (D S Y : ℕ) 
  (h1 : dorothy_age_condition D S)
  (h2 : dorothy_current_age D)
  (h3 : D = 15)
  (h4 : S = 5)
  (h5 : D + Y = 2 * (S + Y)) : Y = 5 := 
sorry

end dorothy_will_be_twice_as_old_l155_155315


namespace complex_sum_l155_155618

noncomputable def omega : ℂ := sorry
axiom omega_power_five : omega^5 = 1
axiom omega_not_one : omega ≠ 1

theorem complex_sum :
  (omega^20 + omega^25 + omega^30 + omega^35 + omega^40 + omega^45 + omega^50 + omega^55 + omega^60 + omega^65 + omega^70) = 11 :=
by
  sorry

end complex_sum_l155_155618


namespace reflect_triangle_final_position_l155_155476

variables {x1 x2 x3 y1 y2 y3 : ℝ}

-- Definition of reflection in x-axis and y-axis
def reflect_x (x y : ℝ) : ℝ × ℝ := (x, -y)
def reflect_y (x y : ℝ) : ℝ × ℝ := (-x, y)

theorem reflect_triangle_final_position (x1 x2 x3 y1 y2 y3 : ℝ) :
  (reflect_y (reflect_x x1 y1).1 (reflect_x x1 y1).2) = (-x1, -y1) ∧
  (reflect_y (reflect_x x2 y2).1 (reflect_x x2 y2).2) = (-x2, -y2) ∧
  (reflect_y (reflect_x x3 y3).1 (reflect_x x3 y3).2) = (-x3, -y3) :=
by
  sorry

end reflect_triangle_final_position_l155_155476


namespace smallest_four_digit_in_pascal_l155_155368

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l155_155368


namespace rectangle_length_reduction_l155_155587

theorem rectangle_length_reduction:
  ∀ (L W : ℝ) (X : ℝ),
  W > 0 →
  L > 0 →
  (L * (1 - X / 100) * (4 / 3)) * W = L * W →
  X = 25 :=
by
  intros L W X hW hL hEq
  sorry

end rectangle_length_reduction_l155_155587


namespace greatest_natural_number_l155_155316

theorem greatest_natural_number (n q r : ℕ) (h1 : n = 91 * q + r)
  (h2 : r = q^2) (h3 : r < 91) : n = 900 :=
sorry

end greatest_natural_number_l155_155316


namespace arithmetic_sequence_a1a6_eq_l155_155283

noncomputable def a_1 : ℤ := 2
noncomputable def d : ℤ := 1
noncomputable def a_n (n : ℕ) : ℤ := a_1 + (n - 1) * d

theorem arithmetic_sequence_a1a6_eq :
  (a_1 * a_n 6) = 14 := by 
  sorry

end arithmetic_sequence_a1a6_eq_l155_155283


namespace change_from_fifteen_dollars_l155_155052

theorem change_from_fifteen_dollars : 
  ∀ (cost_eggs cost_pancakes cost_mug_cocoa num_mugs tax additional_pancakes additional_mug paid : ℕ),
  cost_eggs = 3 →
  cost_pancakes = 2 →
  cost_mug_cocoa = 2 →
  num_mugs = 2 →
  tax = 1 →
  additional_pancakes = 2 →
  additional_mug = 2 →
  paid = 15 →
  paid - (cost_eggs + cost_pancakes + (num_mugs * cost_mug_cocoa) + tax + additional_pancakes + additional_mug) = 1 :=
by
  intros cost_eggs cost_pancakes cost_mug_cocoa num_mugs tax additional_pancakes additional_mug paid
  sorry

end change_from_fifteen_dollars_l155_155052


namespace sin_pi_minus_alpha_l155_155676

theorem sin_pi_minus_alpha (α : ℝ) (h : Real.sin (Real.pi - α) = -1/3) : Real.sin α = -1/3 :=
sorry

end sin_pi_minus_alpha_l155_155676


namespace distance_from_origin_to_midpoint_l155_155314

theorem distance_from_origin_to_midpoint :
  ∀ (x1 y1 x2 y2 : ℝ), (x1 = 10) → (y1 = 20) → (x2 = -10) → (y2 = -20) → 
  dist (0 : ℝ × ℝ) ((x1 + x2) / 2, (y1 + y2) / 2) = 0 := 
by
  intros x1 y1 x2 y2 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- remaining proof goes here
  sorry

end distance_from_origin_to_midpoint_l155_155314


namespace range_of_a_l155_155596

theorem range_of_a (a : ℝ) : 
  (∃ n : ℕ, (∀ x : ℕ, 1 ≤ x → x ≤ 5 → x < a) ∧ (∀ y : ℕ, x ≥ 1 → y ≥ 6 → y ≥ a)) ↔ (5 < a ∧ a < 6) :=
by
  sorry

end range_of_a_l155_155596


namespace n_minus_m_l155_155313

theorem n_minus_m (m n : ℤ) (h_m : m - 2 = 3) (h_n : n + 1 = 2) : n - m = -4 := sorry

end n_minus_m_l155_155313


namespace num_3_digit_multiples_l155_155234

def is_3_digit (n : Nat) : Prop := 100 ≤ n ∧ n ≤ 999
def multiple_of (k n : Nat) : Prop := ∃ m : Nat, n = m * k

theorem num_3_digit_multiples (count_35_not_70 : Nat) (h : count_35_not_70 = 13) :
  let count_multiples_35 := (980 / 35) - (105 / 35) + 1
  let count_multiples_70 := (980 / 70) - (140 / 70) + 1
  count_multiples_35 - count_multiples_70 = count_35_not_70 := sorry

end num_3_digit_multiples_l155_155234


namespace stormi_additional_money_needed_l155_155912

noncomputable def earnings_from_jobs : ℝ :=
  let washing_cars := 5 * 8.50
  let walking_dogs := 4 * 6.75
  let mowing_lawns := 3 * 12.25
  let gardening := 2 * 7.40
  washing_cars + walking_dogs + mowing_lawns + gardening

noncomputable def discounted_prices : ℝ :=
  let bicycle := 150.25 * (1 - 0.15)
  let helmet := 35.75 - 5.00
  let lock := 24.50
  bicycle + helmet + lock

noncomputable def total_cost_after_tax : ℝ :=
  let cost_before_tax := discounted_prices
  cost_before_tax * 1.05

noncomputable def amount_needed : ℝ :=
  total_cost_after_tax - earnings_from_jobs

theorem stormi_additional_money_needed : amount_needed = 71.06 := by
  sorry

end stormi_additional_money_needed_l155_155912


namespace additional_miles_needed_l155_155056

theorem additional_miles_needed :
  ∀ (h : ℝ), (25 + 75 * h) / (5 / 8 + h) = 60 → 75 * h = 62.5 := 
by
  intros h H
  -- the rest of the proof goes here
  sorry

end additional_miles_needed_l155_155056


namespace determine_x_l155_155958

theorem determine_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^3) (h3 : x / 9 = 9 * y) :
  x = 243 * Real.sqrt 3 ∨ x = -243 * Real.sqrt 3 :=
by
  sorry

end determine_x_l155_155958


namespace students_passed_in_dixon_lecture_l155_155793

theorem students_passed_in_dixon_lecture :
  let ratio_collins := 18 / 30
  let students_dixon := 45
  ∃ y, ratio_collins = y / students_dixon ∧ y = 27 :=
by
  sorry

end students_passed_in_dixon_lecture_l155_155793


namespace binomial_square_l155_155467

variable (c : ℝ)

theorem binomial_square (h : ∃ a : ℝ, (x^2 - 164 * x + c) = (x + a)^2) : c = 6724 := sorry

end binomial_square_l155_155467


namespace max_elevation_l155_155426

def particle_elevation (t : ℝ) : ℝ := 200 * t - 20 * t^2 + 50

theorem max_elevation : ∃ t : ℝ, particle_elevation t = 550 :=
by {
  sorry
}

end max_elevation_l155_155426


namespace simplifies_to_minus_18_point_5_l155_155281

theorem simplifies_to_minus_18_point_5 (x y : ℝ) (h_x : x = 1/2) (h_y : y = -2) :
  ((2 * x + y)^2 - (2 * x - y) * (x + y) - 2 * (x - 2 * y) * (x + 2 * y)) / y = -18.5 :=
by
  -- Let's replace x and y with their values
  -- Expand and simplify the expression
  -- Divide the expression by y
  -- Prove the final result is equal to -18.5
  sorry

end simplifies_to_minus_18_point_5_l155_155281


namespace son_age_l155_155839

theorem son_age {S M : ℕ} 
  (h1 : M = S + 37)
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 35 :=
by sorry

end son_age_l155_155839


namespace geometric_sequence_sum_x_l155_155853

variable {α : Type*} [Field α]

theorem geometric_sequence_sum_x (a : ℕ → α) (S : ℕ → α) (x : α) 
  (h₁ : ∀ n, S n = x * (3:α)^n + 1)
  (h₂ : ∀ n, a n = S n - S (n - 1)) :
  ∃ x, x = -1 :=
by
  let a1 := S 1
  let a2 := S 2 - S 1
  let a3 := S 3 - S 2
  have ha1 : a1 = 3 * x + 1 := sorry
  have ha2 : a2 = 6 * x := sorry
  have ha3 : a3 = 18 * x := sorry
  have h_geom : (6 * x)^2 = (3 * x + 1) * 18 * x := sorry
  have h_solve : 18 * x * (x + 1) = 0 := sorry
  have h_x_neg1 : x = 0 ∨ x = -1 := sorry
  exact ⟨-1, sorry⟩

end geometric_sequence_sum_x_l155_155853


namespace sugar_amount_l155_155487

theorem sugar_amount (S F B : ℕ) (h1 : S = 5 * F / 4) (h2 : F = 10 * B) (h3 : F = 8 * (B + 60)) : S = 3000 := by
  sorry

end sugar_amount_l155_155487


namespace binom_eight_four_l155_155032

theorem binom_eight_four : (Nat.choose 8 4) = 70 :=
by
  sorry

end binom_eight_four_l155_155032


namespace triangle_angle_tangent_ratio_triangle_tan_A_minus_B_maximum_l155_155129

theorem triangle_angle_tangent_ratio (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos B - b * Real.cos A = 3 / 5 * c) :
  Real.tan A / Real.tan B = 4 := sorry

theorem triangle_tan_A_minus_B_maximum (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos B - b * Real.cos A = 3 / 5 * c)
  (h2 : Real.tan A / Real.tan B = 4) : Real.tan (A - B) ≤ 3 / 4 := sorry

end triangle_angle_tangent_ratio_triangle_tan_A_minus_B_maximum_l155_155129


namespace solve_system1_l155_155062

structure SystemOfEquations :=
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ)

def system1 : SystemOfEquations :=
  { a₁ := 1, b₁ := -3, c₁ := 4,
    a₂ := 2, b₂ := -1, c₂ := 3 }

theorem solve_system1 :
  ∃ x y : ℝ, x - 3 * y = 4 ∧ 2 * x - y = 3 ∧ x = 1 ∧ y = -1 :=
by
  sorry

end solve_system1_l155_155062


namespace cloth_sales_worth_l155_155971

theorem cloth_sales_worth 
  (commission : ℝ) 
  (commission_rate : ℝ) 
  (commission_received : ℝ) 
  (commission_rate_of_sales : commission_rate = 2.5)
  (commission_received_rs : commission_received = 21) 
  : (commission_received / (commission_rate / 100)) = 840 :=
by
  sorry

end cloth_sales_worth_l155_155971


namespace normal_cost_of_car_wash_l155_155359

-- Conditions
variables (C : ℝ) (H1 : 20 * C > 0) (H2 : 0.60 * (20 * C) = 180)

-- Theorem to be proved
theorem normal_cost_of_car_wash (C : ℝ) (H1 : 20 * C > 0) (H2 : 0.60 * (20 * C) = 180) : C = 15 :=
by
  -- proof omitted
  sorry

end normal_cost_of_car_wash_l155_155359


namespace math_problem_l155_155625

noncomputable def m : ℕ := 294
noncomputable def n : ℕ := 81
noncomputable def d : ℕ := 3

axiom circle_radius (r : ℝ) : r = 42
axiom chords_length (l : ℝ) : l = 78
axiom intersection_distance (d : ℝ) : d = 18

theorem math_problem :
  let m := 294
  let n := 81
  let d := 3
  m + n + d = 378 :=
by {
  -- Proof omitted
  sorry
}

end math_problem_l155_155625


namespace find_age_of_second_person_l155_155442

variable (T A X : ℝ)

def average_original_group (T A : ℝ) : Prop :=
  T = 7 * A

def average_with_39 (T A : ℝ) : Prop :=
  T + 39 = 8 * (A + 2)

def average_with_second_person (T A X : ℝ) : Prop :=
  T + X = 8 * (A - 1) 

theorem find_age_of_second_person (T A X : ℝ) 
  (h1 : average_original_group T A)
  (h2 : average_with_39 T A)
  (h3 : average_with_second_person T A X) :
  X = 15 :=
sorry

end find_age_of_second_person_l155_155442


namespace jordan_meets_emily_after_total_time_l155_155673

noncomputable def meet_time
  (initial_distance : ℝ)
  (speed_ratio : ℝ)
  (decrease_rate : ℝ)
  (time_until_break : ℝ)
  (break_duration : ℝ)
  (total_meet_time : ℝ) : Prop :=
  initial_distance = 30 ∧
  speed_ratio = 2 ∧
  decrease_rate = 2 ∧
  time_until_break = 10 ∧
  break_duration = 5 ∧
  total_meet_time = 17

theorem jordan_meets_emily_after_total_time :
  meet_time 30 2 2 10 5 17 := 
by {
  -- The conditions directly state the requirements needed for the proof.
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩ -- This line confirms that all inputs match the given conditions.
}

end jordan_meets_emily_after_total_time_l155_155673


namespace more_girls_than_boys_l155_155921

theorem more_girls_than_boys (girls boys total_pupils : ℕ) (h1 : girls = 692) (h2 : total_pupils = 926) (h3 : boys = total_pupils - girls) : girls - boys = 458 :=
by
  sorry

end more_girls_than_boys_l155_155921


namespace inequality_proof_l155_155827

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_geq : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3 * b^2 + 5 * c^2 ≤ 1 := 
sorry

end inequality_proof_l155_155827


namespace Annie_total_cookies_l155_155698

theorem Annie_total_cookies :
  let monday_cookies := 5
  let tuesday_cookies := 2 * monday_cookies 
  let wednesday_cookies := 1.4 * tuesday_cookies
  monday_cookies + tuesday_cookies + wednesday_cookies = 29 :=
by
  sorry

end Annie_total_cookies_l155_155698


namespace solve_quadratic_l155_155534

theorem solve_quadratic (x : ℝ) (h_pos : x > 0) (h_eq : 6 * x^2 + 9 * x - 24 = 0) : x = 4 / 3 :=
by
  sorry

end solve_quadratic_l155_155534


namespace tricycle_wheels_l155_155264

theorem tricycle_wheels (T : ℕ) 
  (h1 : 3 * 2 = 6) 
  (h2 : 7 * 1 = 7) 
  (h3 : 6 + 7 + 4 * T = 25) : T = 3 :=
sorry

end tricycle_wheels_l155_155264


namespace polynomial_is_perfect_square_trinomial_l155_155227

-- The definition of a perfect square trinomial
def isPerfectSquareTrinomial (a b c m : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a * b = c ∧ 4 * a * a + m * b = 4 * a * b * b

-- The main theorem to prove that if the polynomial is a perfect square trinomial, then m = 20
theorem polynomial_is_perfect_square_trinomial (a b : ℝ) (h : isPerfectSquareTrinomial 2 1 5 25) :
  ∀ x, (4 * x * x + 20 * x + 25 = (2 * x + 5) * (2 * x + 5)) :=
by
  sorry

end polynomial_is_perfect_square_trinomial_l155_155227


namespace ten_row_triangle_total_l155_155570

theorem ten_row_triangle_total:
  let rods := 3 * (Finset.range 10).sum id
  let connectors := (Finset.range 11).sum (fun n => n + 1)
  rods + connectors = 231 :=
by
  let rods := 3 * (Finset.range 10).sum id
  let connectors := (Finset.range 11).sum (fun n => n + 1)
  sorry

end ten_row_triangle_total_l155_155570


namespace find_minimum_x2_x1_l155_155538

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.log x + 1 / 2

theorem find_minimum_x2_x1 (x1 : ℝ) :
  ∃ x2 : {r : ℝ // 0 < r}, f x1 = g x2 → (x2 - x1) ≥ 1 + Real.log 2 / 2 :=
by
  -- Proof
  sorry

end find_minimum_x2_x1_l155_155538


namespace part1_part2_l155_155245

-- Part (1)
theorem part1 (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < -1 / 2 → (ax - 1) * (x + 1) > 0) →
  a = -2 :=
sorry

-- Part (2)
theorem part2 (a : ℝ) :
  (∀ x : ℝ,
    ((a < -1 ∧ -1 < x ∧ x < 1/a) ∨
     (a = -1 ∧ ∀ x : ℝ, false) ∨
     (-1 < a ∧ a < 0 ∧ 1/a < x ∧ x < -1) ∨
     (a = 0 ∧ x < -1) ∨
     (a > 0 ∧ (x < -1 ∨ x > 1/a))) →
    (ax - 1) * (x + 1) > 0) :=
sorry

end part1_part2_l155_155245


namespace find_P_l155_155448

variable (a b c d P : ℝ)

theorem find_P 
  (h1 : (a + b + c + d) / 4 = 8) 
  (h2 : (a + b + c + d + P) / 5 = P) : 
  P = 8 := 
by 
  sorry

end find_P_l155_155448


namespace gauravi_walks_4500m_on_tuesday_l155_155374

def initial_distance : ℕ := 500
def increase_per_day : ℕ := 500
def target_distance : ℕ := 4500

def distance_after_days (n : ℕ) : ℕ :=
  initial_distance + n * increase_per_day

def day_of_week_after (start_day : ℕ) (n : ℕ) : ℕ :=
  (start_day + n) % 7

def monday : ℕ := 0 -- Represent Monday as 0

theorem gauravi_walks_4500m_on_tuesday :
  distance_after_days 8 = target_distance ∧ day_of_week_after monday 8 = 2 :=
by 
  sorry

end gauravi_walks_4500m_on_tuesday_l155_155374


namespace possible_last_three_digits_product_l155_155952

def lastThreeDigits (n : ℕ) : ℕ := n % 1000

theorem possible_last_three_digits_product (a b c : ℕ) (ha : a > 1000) (hb : b > 1000) (hc : c > 1000)
  (h1 : (a + b) % 10 = c % 10)
  (h2 : (a + c) % 10 = b % 10)
  (h3 : (b + c) % 10 = a % 10) :
  lastThreeDigits (a * b * c) = 0 ∨ lastThreeDigits (a * b * c) = 250 ∨ lastThreeDigits (a * b * c) = 500 ∨ lastThreeDigits (a * b * c) = 750 := 
sorry

end possible_last_three_digits_product_l155_155952


namespace lowest_price_per_component_l155_155028

def production_cost_per_component : ℝ := 80
def shipping_cost_per_component : ℝ := 6
def fixed_monthly_costs : ℝ := 16500
def components_per_month : ℕ := 150

theorem lowest_price_per_component (price_per_component : ℝ) :
  let total_cost_per_component := production_cost_per_component + shipping_cost_per_component
  let total_production_and_shipping_cost := total_cost_per_component * components_per_month
  let total_cost := total_production_and_shipping_cost + fixed_monthly_costs
  price_per_component = total_cost / components_per_month → price_per_component = 196 :=
by
  sorry

end lowest_price_per_component_l155_155028


namespace factor_expression_l155_155399

theorem factor_expression (y : ℤ) : 
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 + 9) = 6 * (2 * y^6 + 7 * y^4 - 3) :=
by 
  sorry

end factor_expression_l155_155399


namespace sum_of_angles_l155_155300

theorem sum_of_angles (p q r s t u v w x y : ℝ)
  (H1 : p + r + t + v + x = 360)
  (H2 : q + s + u + w + y = 360) :
  p + q + r + s + t + u + v + w + x + y = 720 := 
by sorry

end sum_of_angles_l155_155300


namespace travel_distance_l155_155207

-- Define the conditions
def distance_10_gallons := 300 -- 300 miles on 10 gallons of fuel
def gallons_10 := 10 -- 10 gallons

-- Given the distance per gallon, calculate the distance for 15 gallons
def distance_per_gallon := distance_10_gallons / gallons_10

def gallons_15 := 15 -- 15 gallons

def distance_15_gallons := distance_per_gallon * gallons_15

-- Proof statement
theorem travel_distance (d_10 : distance_10_gallons = 300)
                        (g_10 : gallons_10 = 10)
                        (g_15 : gallons_15 = 15) :
  distance_15_gallons = 450 :=
  by
  -- The actual proof goes here
  sorry

end travel_distance_l155_155207


namespace max_visible_sum_l155_155222

-- Definitions for the problem conditions

def numbers : List ℕ := [1, 3, 6, 12, 24, 48]

def num_faces (cubes : List ℕ) : Prop :=
  cubes.length = 18 -- since each of 3 cubes has 6 faces, we expect 18 numbers in total.

def is_valid_cube (cube : List ℕ) : Prop :=
  ∀ n ∈ cube, n ∈ numbers

def are_cubes (cubes : List (List ℕ)) : Prop :=
  cubes.length = 3 ∧ ∀ cube ∈ cubes, is_valid_cube cube ∧ cube.length = 6

-- The main theorem stating the maximum possible sum of the visible numbers
theorem max_visible_sum (cubes : List (List ℕ)) (h : are_cubes cubes) : ∃ s, s = 267 :=
by
  sorry

end max_visible_sum_l155_155222


namespace each_person_gets_9_apples_l155_155203

-- Define the initial number of apples and the number of apples given to Jack's father
def initial_apples : ℕ := 55
def apples_given_to_father : ℕ := 10

-- Define the remaining apples after giving to Jack's father
def remaining_apples : ℕ := initial_apples - apples_given_to_father

-- Define the number of people sharing the remaining apples
def number_of_people : ℕ := 1 + 4

-- Define the number of apples each person will get
def apples_per_person : ℕ := remaining_apples / number_of_people

-- Prove that each person gets 9 apples
theorem each_person_gets_9_apples (h₁ : initial_apples = 55) 
                                  (h₂ : apples_given_to_father = 10) 
                                  (h₃ : number_of_people = 5) 
                                  (h₄ : remaining_apples = initial_apples - apples_given_to_father) 
                                  (h₅ : apples_per_person = remaining_apples / number_of_people) : 
  apples_per_person = 9 :=
by sorry

end each_person_gets_9_apples_l155_155203


namespace number_of_8_digit_increasing_integers_mod_1000_l155_155404

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem number_of_8_digit_increasing_integers_mod_1000 :
  let M := choose 9 8
  M % 1000 = 9 :=
by
  let M := choose 9 8
  show M % 1000 = 9
  sorry

end number_of_8_digit_increasing_integers_mod_1000_l155_155404


namespace geometric_prog_y_90_common_ratio_l155_155037

theorem geometric_prog_y_90_common_ratio :
  ∀ (y : ℝ), y = 90 → ∃ r : ℝ, r = (90 + y) / (30 + y) ∧ r = (180 + y) / (90 + y) ∧ r = 3 / 2 :=
by
  intros
  sorry

end geometric_prog_y_90_common_ratio_l155_155037


namespace Lavinia_daughter_age_difference_l155_155010

-- Define the ages of the individuals involved
variables (Ld Ls Kd : ℕ)

-- Conditions given in the problem
variables (H1 : Kd = 12)
variables (H2 : Ls = 2 * Kd)
variables (H3 : Ls = Ld + 22)

-- Statement we need to prove
theorem Lavinia_daughter_age_difference(Ld Ls Kd : ℕ) (H1 : Kd = 12) (H2 : Ls = 2 * Kd) (H3 : Ls = Ld + 22) : 
  Kd - Ld = 10 :=
sorry

end Lavinia_daughter_age_difference_l155_155010


namespace largest_integer_chosen_l155_155699

-- Define the sequence of operations and establish the resulting constraints
def transformed_value (x : ℤ) : ℤ :=
  2 * (4 * x - 30) - 10

theorem largest_integer_chosen : 
  ∃ (x : ℤ), (10 : ℤ) ≤ transformed_value x ∧ transformed_value x ≤ (99 : ℤ) ∧ x = 21 :=
by
  sorry

end largest_integer_chosen_l155_155699


namespace evaluate_neg_64_pow_two_thirds_l155_155358

theorem evaluate_neg_64_pow_two_thirds 
  (h : -64 = (-4)^3) : (-64)^(2/3) = 16 := 
by 
  -- sorry added to skip the proof.
  sorry  

end evaluate_neg_64_pow_two_thirds_l155_155358


namespace orange_juice_production_correct_l155_155935

noncomputable def orangeJuiceProduction (total_oranges : Float) (export_percent : Float) (juice_percent : Float) : Float :=
  let remaining_oranges := total_oranges * (1 - export_percent / 100)
  let juice_oranges := remaining_oranges * (juice_percent / 100)
  Float.round (juice_oranges * 10) / 10

theorem orange_juice_production_correct :
  orangeJuiceProduction 8.2 30 40 = 2.3 := by
  sorry

end orange_juice_production_correct_l155_155935


namespace pat_kate_mark_ratio_l155_155033

variables (P K M r : ℚ) 

theorem pat_kate_mark_ratio (h1 : P + K + M = 189) 
                            (h2 : P = r * K) 
                            (h3 : P = (1 / 3) * M) 
                            (h4 : M = K + 105) :
  r = 4 / 3 :=
sorry

end pat_kate_mark_ratio_l155_155033


namespace value_to_subtract_l155_155519

theorem value_to_subtract (N x : ℕ) 
  (h1 : (N - x) / 7 = 7) 
  (h2 : (N - 2) / 13 = 4) : x = 5 :=
by 
  sorry

end value_to_subtract_l155_155519


namespace binomial_probability_p_l155_155228

noncomputable def binomial_expected_value (n p : ℝ) := n * p
noncomputable def binomial_variance (n p : ℝ) := n * p * (1 - p)

theorem binomial_probability_p (n p : ℝ) (h1: binomial_expected_value n p = 2) (h2: binomial_variance n p = 1) : 
  p = 0.5 :=
by
  sorry

end binomial_probability_p_l155_155228


namespace sally_initial_peaches_l155_155068

section
variables 
  (peaches_after : ℕ)
  (peaches_picked : ℕ)
  (initial_peaches : ℕ)

theorem sally_initial_peaches 
    (h1 : peaches_picked = 42)
    (h2 : peaches_after = 55)
    (h3 : peaches_after = initial_peaches + peaches_picked) : 
    initial_peaches = 13 := 
by 
  sorry
end

end sally_initial_peaches_l155_155068


namespace seven_points_unit_distance_l155_155395

theorem seven_points_unit_distance :
  ∃ (A B C D E F G : ℝ × ℝ),
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧
     D ≠ E ∧ D ≠ F ∧ D ≠ G ∧
     E ≠ F ∧ E ≠ G ∧
     F ≠ G) ∧
    (∀ (P Q R : ℝ × ℝ),
      (P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E ∨ P = F ∨ P = G) →
      (Q = A ∨ Q = B ∨ Q = C ∨ Q = D ∨ Q = E ∨ Q = F ∨ Q = G) →
      (R = A ∨ R = B ∨ R = C ∨ R = D ∨ R = E ∨ R = F ∨ R = G) →
      P ≠ Q → P ≠ R → Q ≠ R →
      (dist P Q = 1 ∨ dist P R = 1 ∨ dist Q R = 1)) :=
sorry

end seven_points_unit_distance_l155_155395


namespace digits_of_result_l155_155789

theorem digits_of_result 
  (u1 u2 t1 t2 h1 h2 : ℕ) 
  (hu_condition : u1 = u2 + 6)
  (units_column : u1 - u2 = 5)
  (tens_column : t1 - t2 = 9)
  (no_borrowing : u2 < u1) 
  : (h1, u1 - u2) = (4, 5) := 
sorry

end digits_of_result_l155_155789


namespace burritos_in_each_box_l155_155445

theorem burritos_in_each_box (B : ℕ) (h1 : 3 * B - B - 30 = 10) : B = 20 :=
by
  sorry

end burritos_in_each_box_l155_155445


namespace sum_of_number_and_reverse_l155_155998

theorem sum_of_number_and_reverse (a b : Nat) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 := by
  sorry

end sum_of_number_and_reverse_l155_155998


namespace sum_smallest_and_largest_prime_between_1_and_50_l155_155506

noncomputable def smallest_prime_between_1_and_50 : ℕ := 2
noncomputable def largest_prime_between_1_and_50 : ℕ := 47

theorem sum_smallest_and_largest_prime_between_1_and_50 : 
  smallest_prime_between_1_and_50 + largest_prime_between_1_and_50 = 49 := 
by
  sorry

end sum_smallest_and_largest_prime_between_1_and_50_l155_155506


namespace solve_system_l155_155349

theorem solve_system 
    (x y z : ℝ) 
    (h1 : x + y - 2 + 4 * x * y = 0) 
    (h2 : y + z - 2 + 4 * y * z = 0) 
    (h3 : z + x - 2 + 4 * z * x = 0) :
    (x = -1 ∧ y = -1 ∧ z = -1) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
sorry

end solve_system_l155_155349


namespace not_integer_fraction_l155_155631

theorem not_integer_fraction (a b : ℤ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (hrelprime : Nat.gcd a.natAbs b.natAbs = 1) : 
  ¬(∃ (k : ℤ), 2 * a * (a^2 + b^2) = k * (a^2 - b^2)) :=
  sorry

end not_integer_fraction_l155_155631


namespace fair_people_ratio_l155_155725

def next_year_ratio (this_year next_year last_year : ℕ) (total : ℕ) :=
  this_year = 600 ∧
  last_year = next_year - 200 ∧
  this_year + last_year + next_year = total → 
  next_year = 2 * this_year

theorem fair_people_ratio :
  ∀ (next_year : ℕ),
  next_year_ratio 600 next_year (next_year - 200) 2800 → next_year = 2 * 600 := by
sorry

end fair_people_ratio_l155_155725


namespace exists_k_for_inequality_l155_155017

noncomputable def C : ℕ := sorry -- C is a positive integer > 0
def a : ℕ → ℝ := sorry -- a sequence of positive real numbers

axiom C_pos : 0 < C
axiom a_pos : ∀ n : ℕ, 0 < a n
axiom recurrence_relation : ∀ n : ℕ, a (n + 1) = n / a n + C

theorem exists_k_for_inequality :
  ∃ k : ℕ, ∀ n : ℕ, n ≥ k → a (n + 2) > a n :=
  sorry

end exists_k_for_inequality_l155_155017


namespace vasya_fraction_l155_155615

-- Define the variables for distances and total distance
variables {a b c d s : ℝ}

-- Define conditions
def anton_distance (a b : ℝ) : Prop := a = b / 2
def sasha_distance (c a d : ℝ) : Prop := c = a + d
def dima_distance (d s : ℝ) : Prop := d = s / 10
def total_distance (a b c d s : ℝ) : Prop := a + b + c + d = s

-- The main theorem 
theorem vasya_fraction (a b c d s : ℝ) (h1 : anton_distance a b) 
  (h2 : sasha_distance c a d) (h3 : dima_distance d s)
  (h4 : total_distance a b c d s) : b / s = 0.4 :=
sorry

end vasya_fraction_l155_155615


namespace how_many_halves_to_sum_one_and_one_half_l155_155054

theorem how_many_halves_to_sum_one_and_one_half : 
  (3 / 2) / (1 / 2) = 3 := 
by 
  sorry

end how_many_halves_to_sum_one_and_one_half_l155_155054


namespace volume_of_cone_formed_by_sector_l155_155138

theorem volume_of_cone_formed_by_sector :
  let radius := 6
  let sector_fraction := (5:ℝ) / 6
  let circumference := 2 * Real.pi * radius
  let cone_base_circumference := sector_fraction * circumference
  let cone_base_radius := cone_base_circumference / (2 * Real.pi)
  let slant_height := radius
  let cone_height := Real.sqrt (slant_height^2 - cone_base_radius^2)
  let volume := (1:ℝ) / 3 * Real.pi * (cone_base_radius^2) * cone_height
  volume = 25 / 3 * Real.pi * Real.sqrt 11 :=
by sorry

end volume_of_cone_formed_by_sector_l155_155138


namespace folded_triangle_sqrt_equals_l155_155418

noncomputable def folded_triangle_length_squared (s : ℕ) (d : ℕ) : ℚ :=
  let x := (2 * s * s - 2 * d * s)/(2 * d)
  let y := (2 * s * s - 2 * (s - d) * s)/(2 * (s - d))
  x * x - x * y + y * y

theorem folded_triangle_sqrt_equals :
  folded_triangle_length_squared 15 11 = (60118.9025 / 1681 : ℚ) := sorry

end folded_triangle_sqrt_equals_l155_155418


namespace arithmetic_expression_value_l155_155322

def mixed_to_frac (a b c : ℕ) : ℚ := a + b / c

theorem arithmetic_expression_value :
  ( ( (mixed_to_frac 5 4 45 - mixed_to_frac 4 1 6) / mixed_to_frac 5 8 15 ) / 
    ( (mixed_to_frac 4 2 3 + 3 / 4) * mixed_to_frac 3 9 13 ) * mixed_to_frac 34 2 7 + 
    (3 / 10 / (1 / 100) / 70) + 2 / 7 ) = 1 :=
by
  -- We need to convert the mixed numbers to fractions using mixed_to_frac
  -- Then, we simplify step-by-step as in the problem solution, but for now we just use sorry
  sorry

end arithmetic_expression_value_l155_155322


namespace max_distinct_colorings_5x5_l155_155680

theorem max_distinct_colorings_5x5 (n : ℕ) :
  ∃ N, N ≤ (n^25 + 4 * n^15 + n^13 + 2 * n^7) / 8 :=
sorry

end max_distinct_colorings_5x5_l155_155680


namespace find_ab_l155_155133

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
by
  sorry

end find_ab_l155_155133


namespace rehabilitation_centers_total_l155_155255

noncomputable def jane_visits (han_visits : ℕ) : ℕ := 2 * han_visits + 6
noncomputable def han_visits (jude_visits : ℕ) : ℕ := 2 * jude_visits - 2
noncomputable def jude_visits (lisa_visits : ℕ) : ℕ := lisa_visits / 2
def lisa_visits : ℕ := 6

def total_visits (jane_visits han_visits jude_visits lisa_visits : ℕ) : ℕ :=
  jane_visits + han_visits + jude_visits + lisa_visits

theorem rehabilitation_centers_total :
  total_visits (jane_visits (han_visits (jude_visits lisa_visits))) 
               (han_visits (jude_visits lisa_visits))
               (jude_visits lisa_visits) 
               lisa_visits = 27 :=
by
  sorry

end rehabilitation_centers_total_l155_155255


namespace f_at_1_over_11_l155_155329

noncomputable def f : (ℝ → ℝ) := sorry

axiom f_domain : ∀ x, 0 < x → 0 < f x

axiom f_eq : ∀ x y, 0 < x → 0 < y → 10 * ((x + y) / (x * y)) = (f x) * (f y) - f (x * y) - 90

theorem f_at_1_over_11 : f (1 / 11) = 21 := by
  -- proof is omitted
  sorry

end f_at_1_over_11_l155_155329


namespace volume_tetrahedral_region_is_correct_l155_155961

noncomputable def volume_of_tetrahedral_region (a : ℝ) : ℝ :=
  (81 - 8 * Real.pi) * a^3 / 486

theorem volume_tetrahedral_region_is_correct (a : ℝ) :
  volume_of_tetrahedral_region a = (81 - 8 * Real.pi) * a^3 / 486 :=
by
  sorry

end volume_tetrahedral_region_is_correct_l155_155961


namespace different_books_l155_155973

-- Definitions for the conditions
def tony_books : ℕ := 23
def dean_books : ℕ := 12
def breanna_books : ℕ := 17
def common_books_tony_dean : ℕ := 3
def common_books_all : ℕ := 1

-- Prove the total number of different books they have read is 47
theorem different_books : (tony_books + dean_books - common_books_tony_dean + breanna_books - 2 * common_books_all) = 47 :=
by
  sorry

end different_books_l155_155973


namespace playground_girls_count_l155_155542

theorem playground_girls_count (boys : ℕ) (total_children : ℕ) 
  (h_boys : boys = 35) (h_total : total_children = 63) : 
  ∃ girls : ℕ, girls = 28 ∧ girls = total_children - boys := 
by 
  sorry

end playground_girls_count_l155_155542


namespace length_of_courtyard_l155_155588

-- Given conditions

def width_of_courtyard : ℝ := 14
def brick_length : ℝ := 0.25
def brick_width : ℝ := 0.15
def total_bricks : ℝ := 8960

-- To be proven
theorem length_of_courtyard : brick_length * brick_width * total_bricks / width_of_courtyard = 24 := 
by sorry

end length_of_courtyard_l155_155588


namespace alligators_not_hiding_l155_155184

-- Definitions derived from conditions
def total_alligators : ℕ := 75
def hiding_alligators : ℕ := 19

-- Theorem statement matching the mathematically equivalent proof problem.
theorem alligators_not_hiding : (total_alligators - hiding_alligators) = 56 := by
  -- Sorry skips the proof. Replace with actual proof if required.
  sorry

end alligators_not_hiding_l155_155184


namespace ratio_future_age_l155_155379

variable (S M : ℕ)

theorem ratio_future_age (h1 : (S : ℝ) / M = 7 / 2) (h2 : S - 6 = 78) : 
  ((S + 16) : ℝ) / (M + 16) = 5 / 2 := 
by
  sorry

end ratio_future_age_l155_155379


namespace average_marks_class_l155_155446

theorem average_marks_class (total_students : ℕ)
  (students_98 : ℕ) (score_98 : ℕ)
  (students_0 : ℕ) (score_0 : ℕ)
  (remaining_avg : ℝ)
  (h1 : total_students = 40)
  (h2 : students_98 = 6)
  (h3 : score_98 = 98)
  (h4 : students_0 = 9)
  (h5 : score_0 = 0)
  (h6 : remaining_avg = 57) :
  ( (( students_98 * score_98) + (students_0 * score_0) + ((total_students - students_98 - students_0) * remaining_avg)) / total_students ) = 50.325 :=
by 
  -- This is where the proof steps would go
  sorry

end average_marks_class_l155_155446


namespace part1_part2_part3_l155_155531

-- Part 1
theorem part1 : (1 > -1) ∧ (1 < 2) ∧ (-(1/2) > -1) ∧ (-(1/2) < 2) := 
  by sorry

-- Part 2
theorem part2 (k : Real) : (3 < k) ∧ (k ≤ 4) := 
  by sorry

-- Part 3
theorem part3 (m : Real) : (2 < m) ∧ (m ≤ 3) := 
  by sorry

end part1_part2_part3_l155_155531


namespace original_purchase_price_first_commodity_l155_155457

theorem original_purchase_price_first_commodity (x y : ℝ) 
  (h1 : 1.07 * (x + y) = 827) 
  (h2 : x = y + 127) : 
  x = 450.415 :=
  sorry

end original_purchase_price_first_commodity_l155_155457


namespace expression_evaluation_l155_155341

def eval_expression : Int := 
  let a := -2 ^ 3
  let b := abs (2 - 3)
  let c := -2 * (-1) ^ 2023
  a + b + c

theorem expression_evaluation :
  eval_expression = -5 :=
by
  sorry

end expression_evaluation_l155_155341


namespace area_of_woods_l155_155072

def width := 8 -- the width in miles
def length := 3 -- the length in miles
def area (w : Nat) (l : Nat) : Nat := w * l -- the area function for a rectangle

theorem area_of_woods : area width length = 24 := by
  sorry

end area_of_woods_l155_155072


namespace find_number_l155_155716

theorem find_number (x : ℤ) (h : x - (28 - (37 - (15 - 16))) = 55) : x = 65 :=
sorry

end find_number_l155_155716


namespace shaina_keeps_chocolate_l155_155824

theorem shaina_keeps_chocolate :
  let total_chocolate := (60 : ℚ) / 7
  let number_of_piles := 5
  let weight_per_pile := total_chocolate / number_of_piles
  let given_weight_back := (1 / 2) * weight_per_pile
  let kept_weight := weight_per_pile - given_weight_back
  kept_weight = 6 / 7 :=
by
  sorry

end shaina_keeps_chocolate_l155_155824


namespace incorrect_games_less_than_three_fourths_l155_155916

/-- In a round-robin chess tournament, each participant plays against every other participant exactly once.
A win earns one point, a draw earns half a point, and a loss earns zero points.
We will call a game incorrect if the player who won the game ends up with fewer total points than the player who lost.

1. Prove that incorrect games make up less than 3/4 of the total number of games in the tournament.
2. Prove that in part (1), the number 3/4 cannot be replaced with a smaller number.
--/
theorem incorrect_games_less_than_three_fourths {n : ℕ} (h : n > 1) :
  ∃ m, (∃ (incorrect_games total_games : ℕ), m = incorrect_games ∧ total_games = (n * (n - 1)) / 2 
    ∧ (incorrect_games : ℚ) / total_games < 3 / 4) 
    ∧ (∀ m' : ℚ, m' ≥ 0 → m = incorrect_games ∧ (incorrect_games : ℚ) / total_games < m' → m' ≥ 3 / 4) :=
sorry

end incorrect_games_less_than_three_fourths_l155_155916


namespace bhanu_house_rent_l155_155220

theorem bhanu_house_rent (I : ℝ) 
  (h1 : 0.30 * I = 300) 
  (h2 : 210 = 210) : 
  210 / (I - 300) = 0.30 := 
by 
  sorry

end bhanu_house_rent_l155_155220


namespace find_value_of_squares_l155_155823

-- Defining the conditions
variable (a b c : ℝ)
variable (h1 : a^2 + 3 * b = 10)
variable (h2 : b^2 + 5 * c = 0)
variable (h3 : c^2 + 7 * a = -21)

-- Stating the theorem to prove the desired result
theorem find_value_of_squares : a^2 + b^2 + c^2 = 83 / 4 :=
   sorry

end find_value_of_squares_l155_155823


namespace maximum_xy_l155_155592

variable {a b c x y : ℝ}

theorem maximum_xy 
  (h1 : a * x + b * y + 2 * c = 0)
  (h2 : c ≠ 0)
  (h3 : a * b - c^2 ≥ 0) :
  ∃ (m : ℝ), m = x * y ∧ m ≤ 1 :=
sorry

end maximum_xy_l155_155592


namespace bowling_ball_weight_l155_155304

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 9 * b = 6 * c) 
  (h2 : 4 * c = 120) : 
  b = 20 :=
by 
  sorry

end bowling_ball_weight_l155_155304


namespace line_intersects_iff_sufficient_l155_155580

noncomputable def sufficient_condition (b : ℝ) : Prop :=
b > 1

noncomputable def condition (b : ℝ) : Prop :=
b > 0

noncomputable def line_intersects_hyperbola (b : ℝ) : Prop :=
b > 2 / 3

theorem line_intersects_iff_sufficient (b : ℝ) (h : condition b) : 
  (sufficient_condition b) → (line_intersects_hyperbola b) ∧ ¬(line_intersects_hyperbola b) → (sufficient_condition b) :=
by {
  sorry
}

end line_intersects_iff_sufficient_l155_155580


namespace find_matrix_A_l155_155485

theorem find_matrix_A (a b c d : ℝ) 
  (h1 : a - 3 * b = -1)
  (h2 : c - 3 * d = 3)
  (h3 : a + b = 3)
  (h4 : c + d = 3) :
  a = 2 ∧ b = 1 ∧ c = 3 ∧ d = 0 := by
  sorry

end find_matrix_A_l155_155485


namespace lawrence_walked_total_distance_l155_155362

noncomputable def distance_per_day : ℝ := 4.0
noncomputable def number_of_days : ℝ := 3.0
noncomputable def total_distance_walked (distance_per_day : ℝ) (number_of_days : ℝ) : ℝ :=
  distance_per_day * number_of_days

theorem lawrence_walked_total_distance :
  total_distance_walked distance_per_day number_of_days = 12.0 :=
by
  -- The detailed proof is omitted as per the instructions.
  sorry

end lawrence_walked_total_distance_l155_155362


namespace relatively_prime_sequence_l155_155173

theorem relatively_prime_sequence (k : ℤ) (hk : k > 1) :
  ∃ (a b : ℤ) (x : ℕ → ℤ),
    a > 0 ∧ b > 0 ∧
    (∀ n, x (n + 2) = x (n + 1) + x n) ∧
    x 0 = a ∧ x 1 = b ∧ ∀ n, gcd (x n) (4 * k^2 - 5) = 1 :=
by
  sorry

end relatively_prime_sequence_l155_155173


namespace deepak_investment_l155_155635

theorem deepak_investment (D : ℝ) (A : ℝ) (P : ℝ) (Dp : ℝ) (Ap : ℝ) 
  (hA : A = 22500)
  (hP : P = 13800)
  (hDp : Dp = 5400)
  (h_ratio : Dp / P = D / (A + D)) :
  D = 15000 := by
  sorry

end deepak_investment_l155_155635


namespace total_tickets_sold_l155_155378

theorem total_tickets_sold :
  ∃(S : ℕ), 4 * S + 6 * 388 = 2876 ∧ S + 388 = 525 :=
by
  sorry

end total_tickets_sold_l155_155378


namespace smallest_invariant_number_l155_155481

def operation (n : ℕ) : ℕ :=
  let q := n / 10
  let r := n % 10
  q + 2 * r

def is_invariant (n : ℕ) : Prop :=
  operation n = n

theorem smallest_invariant_number : ∃ n : ℕ, is_invariant n ∧ n = 10^99 + 1 :=
by
  sorry

end smallest_invariant_number_l155_155481


namespace part_I_extreme_value_part_II_range_of_a_l155_155994

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + Real.log x + 1

theorem part_I_extreme_value (a : ℝ) (h1 : a = -1/4) :
  (∀ x > 0, f a x ≤ f a 2) ∧ f a 2 = 3/4 + Real.log 2 :=
sorry

theorem part_II_range_of_a (a : ℝ) :
  (∀ x ≥ 1, f a x ≤ x) ↔ a ≤ 0 :=
sorry

end part_I_extreme_value_part_II_range_of_a_l155_155994


namespace infinite_solutions_eq_l155_155559

/-
Proving that the equation x - y + z = 1 has infinite solutions under the conditions:
1. x, y, z are distinct positive integers.
2. The product of any two numbers is divisible by the third one.
-/
theorem infinite_solutions_eq (x y z : ℕ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) 
(h4 : ∃ m n k : ℕ, x = m * n ∧ y = n * k ∧ z = m * k)
(h5 : (x*y) % z = 0) (h6 : (y*z) % x = 0) (h7 : (z*x) % y = 0) : 
∃ (m : ℕ), x - y + z = 1 ∧ x > 0 ∧ y > 0 ∧ z > 0 :=
by sorry

end infinite_solutions_eq_l155_155559


namespace triangle_area_l155_155347

theorem triangle_area : 
  let p1 := (0, 0)
  let p2 := (0, 6)
  let p3 := (8, 15)
  let base := 6
  let height := 8
  0.5 * base * height = 24.0 :=
by
  let p1 := (0, 0)
  let p2 := (0, 6)
  let p3 := (8, 15)
  let base := 6
  let height := 8
  sorry

end triangle_area_l155_155347


namespace initial_workers_number_l155_155081

-- Define the initial problem
variables {W : ℕ} -- Number of initial workers
variables (Work1 : ℕ := W * 8) -- Work done for the first hole
variables (Work2 : ℕ := (W + 65) * 6) -- Work done for the second hole
variables (Depth1 : ℕ := 30) -- Depth of the first hole
variables (Depth2 : ℕ := 55) -- Depth of the second hole

-- Expressing the conditions and question
theorem initial_workers_number : 8 * W * 55 = 30 * (W + 65) * 6 → W = 45 :=
by
  sorry

end initial_workers_number_l155_155081


namespace equivalent_conditions_l155_155051

open Real

theorem equivalent_conditions (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / x + 1 / y + 1 / z ≤ 1) ↔
  (∀ a b c d : ℝ, a + b + c > d → a^2 * x + b^2 * y + c^2 * z > d^2) :=
by
  sorry

end equivalent_conditions_l155_155051


namespace find_u_v_l155_155070

theorem find_u_v (u v : ℤ) (huv_pos : 0 < v ∧ v < u) (area_eq : u^2 + 3 * u * v = 615) : 
  u + v = 45 :=
sorry

end find_u_v_l155_155070


namespace vendor_has_1512_liters_of_sprite_l155_155357

-- Define the conditions
def liters_of_maaza := 60
def liters_of_pepsi := 144
def least_number_of_cans := 143
def gcd_maaza_pepsi := Nat.gcd liters_of_maaza liters_of_pepsi --let Lean compute GCD

-- Define the liters per can as the GCD of Maaza and Pepsi
def liters_per_can := gcd_maaza_pepsi

-- Define the number of cans for Maaza and Pepsi respectively
def cans_of_maaza := liters_of_maaza / liters_per_can
def cans_of_pepsi := liters_of_pepsi / liters_per_can

-- Define total cans for Maaza and Pepsi
def total_cans_for_maaza_and_pepsi := cans_of_maaza + cans_of_pepsi

-- Define the number of cans for Sprite
def cans_of_sprite := least_number_of_cans - total_cans_for_maaza_and_pepsi

-- The total liters of Sprite the vendor has
def liters_of_sprite := cans_of_sprite * liters_per_can

-- Statement to prove
theorem vendor_has_1512_liters_of_sprite : 
  liters_of_sprite = 1512 :=
by
  -- solution omitted 
  sorry

end vendor_has_1512_liters_of_sprite_l155_155357


namespace probability_odd_sum_is_one_half_probability_2x_plus_y_less_than_10_is_seven_eighteenths_l155_155820

def num_faces : ℕ := 6
def possible_outcomes : ℕ := num_faces * num_faces

def count_odd_sum_outcomes : ℕ := 18 -- From solution steps
def probability_odd_sum : ℚ := count_odd_sum_outcomes / possible_outcomes

def count_2x_plus_y_less_than_10 : ℕ := 14 -- From solution steps
def probability_2x_plus_y_less_than_10 : ℚ := count_2x_plus_y_less_than_10 / possible_outcomes

theorem probability_odd_sum_is_one_half :
  probability_odd_sum = 1 / 2 :=
sorry

theorem probability_2x_plus_y_less_than_10_is_seven_eighteenths :
  probability_2x_plus_y_less_than_10 = 7 / 18 :=
sorry

end probability_odd_sum_is_one_half_probability_2x_plus_y_less_than_10_is_seven_eighteenths_l155_155820


namespace prime_divisors_of_n_congruent_to_1_mod_4_l155_155309

theorem prime_divisors_of_n_congruent_to_1_mod_4
  (x y n : ℕ)
  (hx : x ≥ 3)
  (hn : n ≥ 2)
  (h_eq : x^2 + 5 = y^n) :
  ∀ p : ℕ, Prime p → p ∣ n → p ≡ 1 [MOD 4] :=
by
  sorry

end prime_divisors_of_n_congruent_to_1_mod_4_l155_155309


namespace factor_adjustment_l155_155406

theorem factor_adjustment (a b : ℝ) (h : a * b = 65.08) : a / 100 * (100 * b) = 65.08 :=
by
  sorry

end factor_adjustment_l155_155406


namespace max_wx_plus_xy_plus_yz_l155_155948

theorem max_wx_plus_xy_plus_yz (w x y z : ℝ) (h1 : w ≥ 0) (h2 : x ≥ 0) (h3 : y ≥ 0) (h4 : z ≥ 0) (h_sum : w + x + y + z = 200) : wx + xy + yz ≤ 10000 :=
sorry

end max_wx_plus_xy_plus_yz_l155_155948


namespace solve_for_a_b_c_d_l155_155191

theorem solve_for_a_b_c_d :
  ∃ a b c d : ℕ, (a + b + c + d) * (a^2 + b^2 + c^2 + d^2)^2 = 2023 ∧ a^3 + b^3 + c^3 + d^3 = 43 := 
by
  sorry

end solve_for_a_b_c_d_l155_155191


namespace additional_bureaus_needed_correct_l155_155126

-- The number of bureaus the company has
def total_bureaus : ℕ := 192

-- The number of offices
def total_offices : ℕ := 36

-- The additional bureaus needed to ensure each office gets an equal number
def additional_bureaus_needed (bureaus : ℕ) (offices : ℕ) : ℕ :=
  let bureaus_per_office := bureaus / offices
  let rounded_bureaus_per_office := bureaus_per_office + if bureaus % offices = 0 then 0 else 1
  let total_bureaus_needed := offices * rounded_bureaus_per_office
  total_bureaus_needed - bureaus

-- Problem Statement: Prove that at least 24 more bureaus are needed
theorem additional_bureaus_needed_correct : 
  additional_bureaus_needed total_bureaus total_offices = 24 := 
by
  sorry

end additional_bureaus_needed_correct_l155_155126


namespace find_x_to_print_800_leaflets_in_3_minutes_l155_155844

theorem find_x_to_print_800_leaflets_in_3_minutes (x : ℝ) :
  (800 / 12 + 800 / x = 800 / 3) → (1 / 12 + 1 / x = 1 / 3) :=
by
  intro h
  have h1 : 800 / 12 = 200 / 3 := by norm_num
  have h2 : 800 / 3 = 800 / 3 := by norm_num
  sorry

end find_x_to_print_800_leaflets_in_3_minutes_l155_155844


namespace prod_ab_eq_three_l155_155477

theorem prod_ab_eq_three (a b : ℝ) (h₁ : a - b = 5) (h₂ : a^2 + b^2 = 31) : a * b = 3 := 
sorry

end prod_ab_eq_three_l155_155477


namespace fraction_of_integers_divisible_by_3_and_4_up_to_100_eq_2_over_25_l155_155860

def positive_integers_up_to (n : ℕ) : List ℕ :=
  List.range' 1 n

def divisible_by_lcm (lcm : ℕ) (lst : List ℕ) : List ℕ :=
  lst.filter (λ x => x % lcm = 0)

noncomputable def fraction_divisible_by_both (n a b : ℕ) : ℚ :=
  let lcm_ab := Nat.lcm a b
  let elems := positive_integers_up_to n
  let divisible_elems := divisible_by_lcm lcm_ab elems
  divisible_elems.length / n

theorem fraction_of_integers_divisible_by_3_and_4_up_to_100_eq_2_over_25 :
  fraction_divisible_by_both 100 3 4 = (2 : ℚ) / 25 :=
by
  sorry

end fraction_of_integers_divisible_by_3_and_4_up_to_100_eq_2_over_25_l155_155860


namespace cost_price_of_radio_l155_155035

theorem cost_price_of_radio (C : ℝ) 
  (overhead_expenses : ℝ := 20) 
  (selling_price : ℝ := 300) 
  (profit_percent : ℝ := 22.448979591836732) :
  C = 228.57 :=
by
  sorry

end cost_price_of_radio_l155_155035


namespace students_in_class_l155_155480

theorem students_in_class (x : ℕ) (S : ℕ)
  (h1 : S = 3 * (S / x) + 24)
  (h2 : S = 4 * (S / x) - 26) : 3 * x + 24 = 4 * x - 26 :=
by
  sorry

end students_in_class_l155_155480


namespace compute_cos_l155_155352

noncomputable def angle1 (A C B : ℝ) : Prop := A + C = 2 * B
noncomputable def angle2 (A C B : ℝ) : Prop := 1 / Real.cos A + 1 / Real.cos C = -Real.sqrt 2 / Real.cos B

theorem compute_cos (A B C : ℝ) (h1 : angle1 A C B) (h2 : angle2 A C B) : 
  Real.cos ((A - C) / 2) = Real.sqrt 2 / 2 :=
sorry

end compute_cos_l155_155352


namespace pancakes_needed_l155_155835

theorem pancakes_needed (initial_pancakes : ℕ) (num_people : ℕ) (pancakes_left : ℕ) :
  initial_pancakes = 12 → num_people = 8 → pancakes_left = initial_pancakes - num_people →
  (num_people - pancakes_left) = 4 :=
by
  intros initial_pancakes_eq num_people_eq pancakes_left_eq
  sorry

end pancakes_needed_l155_155835


namespace dividend_is_686_l155_155626

theorem dividend_is_686 (divisor quotient remainder : ℕ) (h1 : divisor = 36) (h2 : quotient = 19) (h3 : remainder = 2) :
  divisor * quotient + remainder = 686 :=
by
  sorry

end dividend_is_686_l155_155626


namespace binomial_product_result_l155_155044

-- Defining the combination (binomial coefficient) formula
def combination (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- Lean theorem statement to prove the problem
theorem binomial_product_result : combination 10 3 * combination 8 3 = 6720 := by
  sorry

end binomial_product_result_l155_155044


namespace stamps_problem_l155_155837

def largest_common_divisor (a b c : ℕ) : ℕ :=
  gcd (gcd a b) c

theorem stamps_problem :
  largest_common_divisor 1020 1275 1350 = 15 :=
by
  sorry

end stamps_problem_l155_155837


namespace problem_statement_l155_155444

noncomputable def f (x : ℝ) : ℝ := (1 + x) / (2 - x)

noncomputable def f_iter : ℕ → ℝ → ℝ
| 0, x => x
| n + 1, x => f (f_iter n x)

variable (x : ℝ)

theorem problem_statement
  (h : f_iter 13 x = f_iter 31 x) :
  f_iter 16 x = (x - 1) / x :=
by
  sorry

end problem_statement_l155_155444


namespace product_of_three_3_digits_has_four_zeros_l155_155013

noncomputable def has_four_zeros_product : Prop :=
  ∃ (a b c: ℕ),
    (100 ≤ a ∧ a < 1000) ∧
    (100 ≤ b ∧ b < 1000) ∧
    (100 ≤ c ∧ c < 1000) ∧
    (∃ (da db dc: Finset ℕ), (da ∪ db ∪ dc = Finset.range 10) ∧
    (∀ x ∈ da, x = a / 10^(x%10) % 10) ∧
    (∀ x ∈ db, x = b / 10^(x%10) % 10) ∧
    (∀ x ∈ dc, x = c / 10^(x%10) % 10)) ∧
    (a * b * c % 10000 = 0)

theorem product_of_three_3_digits_has_four_zeros : has_four_zeros_product := sorry

end product_of_three_3_digits_has_four_zeros_l155_155013


namespace sum_of_largest_and_smallest_odd_numbers_is_16_l155_155938

-- Define odd numbers between 5 and 12
def odd_numbers_set := {n | 5 ≤ n ∧ n ≤ 12 ∧ n % 2 = 1}

-- Define smallest odd number from the set
def min_odd := 5

-- Define largest odd number from the set
def max_odd := 11

-- The main theorem stating that the sum of the smallest and largest odd numbers is 16
theorem sum_of_largest_and_smallest_odd_numbers_is_16 :
  min_odd + max_odd = 16 := by
  sorry

end sum_of_largest_and_smallest_odd_numbers_is_16_l155_155938


namespace students_exam_percentage_l155_155465

theorem students_exam_percentage 
  (total_students : ℕ) 
  (avg_assigned_day : ℚ) 
  (avg_makeup_day : ℚ)
  (overall_avg : ℚ) 
  (h_total : total_students = 100)
  (h_avg_assigned_day : avg_assigned_day = 0.60) 
  (h_avg_makeup_day : avg_makeup_day = 0.80) 
  (h_overall_avg : overall_avg = 0.66) : 
  ∃ x : ℚ, x = 70 / 100 :=
by
  sorry

end students_exam_percentage_l155_155465


namespace find_multiplier_l155_155503

theorem find_multiplier (x y n : ℤ) (h1 : 3 * x + y = 40) (h2 : 2 * x - y = 20) (h3 : y^2 = 16) :
  n * y^2 = 48 :=
by 
  -- proof goes here
  sorry

end find_multiplier_l155_155503


namespace min_AB_distance_l155_155218

theorem min_AB_distance : 
  ∀ (A B : ℝ × ℝ), 
  A ≠ B → 
  ((∃ (m : ℝ), A.2 = m * (A.1 - 1) + 1 ∧ B.2 = m * (B.1 - 1) + 1) ∧ 
    ((A.1 - 2)^2 + (A.2 - 3)^2 = 9) ∧ 
    ((B.1 - 2)^2 + (B.2 - 3)^2 = 9)) → 
  dist A B = 4 :=
sorry

end min_AB_distance_l155_155218


namespace total_money_spent_l155_155495

/-- Erika, Elizabeth, Emma, and Elsa went shopping on Wednesday.
Emma spent $58.
Erika spent $20 more than Emma.
Elsa spent twice as much as Emma.
Elizabeth spent four times as much as Elsa.
Erika received a 10% discount on what she initially spent.
Elizabeth had to pay a 6% tax on her purchases.
Prove that the total amount of money they spent is $736.04.
-/
theorem total_money_spent :
  let emma_spent := 58
  let erika_initial_spent := emma_spent + 20
  let erika_discount := 0.10 * erika_initial_spent
  let erika_final_spent := erika_initial_spent - erika_discount
  let elsa_spent := 2 * emma_spent
  let elizabeth_initial_spent := 4 * elsa_spent
  let elizabeth_tax := 0.06 * elizabeth_initial_spent
  let elizabeth_final_spent := elizabeth_initial_spent + elizabeth_tax
  let total_spent := emma_spent + erika_final_spent + elsa_spent + elizabeth_final_spent
  total_spent = 736.04 := by
  sorry

end total_money_spent_l155_155495


namespace judy_shopping_trip_l155_155569

-- Define the quantities and prices of the items
def num_carrots : ℕ := 5
def price_carrot : ℕ := 1
def num_milk : ℕ := 4
def price_milk : ℕ := 3
def num_pineapples : ℕ := 2
def price_pineapple : ℕ := 4
def num_flour : ℕ := 2
def price_flour : ℕ := 5
def price_ice_cream : ℕ := 7

-- Define the promotion conditions
def pineapple_promotion : ℕ := num_pineapples / 2

-- Define the coupon condition
def coupon_threshold : ℕ := 40
def coupon_value : ℕ := 10

-- Define the total cost without coupon
def total_cost : ℕ := 
  (num_carrots * price_carrot) + 
  (num_milk * price_milk) +
  (pineapple_promotion * price_pineapple) +
  (num_flour * price_flour) +
  price_ice_cream

-- Define the final cost considering the coupon condition
def final_cost : ℕ :=
  if total_cost < coupon_threshold then total_cost else total_cost - coupon_value

-- The theorem to be proven
theorem judy_shopping_trip : final_cost = 38 := by
  sorry

end judy_shopping_trip_l155_155569


namespace find_a_l155_155027

theorem find_a (a : ℝ) (x_values y_values : List ℝ)
  (h_y : ∀ x, List.getD y_values x 0 = 2.1 * List.getD x_values x 1 - 0.3) :
  a = 10 :=
by
  have h_mean_x : (1 + 2 + 3 + 4 + 5) / 5 = 3 := by norm_num
  have h_sum_y : (2 + 3 + 7 + 8 + a) / 5 = (2.1 * 3 - 0.3) := by sorry
  sorry

end find_a_l155_155027


namespace equation_has_real_roots_for_all_K_l155_155963

open Real

noncomputable def original_equation (K x : ℝ) : ℝ :=
  x - K^3 * (x - 1) * (x - 3)

theorem equation_has_real_roots_for_all_K :
  ∀ K : ℝ, ∃ x : ℝ, original_equation K x = 0 :=
sorry

end equation_has_real_roots_for_all_K_l155_155963


namespace find_x_plus_y_l155_155435

variables {x y : ℝ}

def f (t : ℝ) : ℝ := t^2003 + 2002 * t

theorem find_x_plus_y (hx : f (x - 1) = -1) (hy : f (y - 2) = 1) : x + y = 3 :=
by
  sorry

end find_x_plus_y_l155_155435


namespace polygon_sides_eq_six_l155_155546

theorem polygon_sides_eq_six (n : ℕ) (h1 : (n - 2) * 180 = 2 * 360) : n = 6 :=
by
  sorry

end polygon_sides_eq_six_l155_155546


namespace find_constant_k_l155_155861

theorem find_constant_k 
  (k : ℝ)
  (h : ∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4)) : 
  k = -16 :=
sorry

end find_constant_k_l155_155861


namespace min_value_64_l155_155770

noncomputable def min_value_expr (a b c d e f g h : ℝ) : ℝ :=
  (a * e) ^ 2 + (b * f) ^ 2 + (c * g) ^ 2 + (d * h) ^ 2

theorem min_value_64 
  (a b c d e f g h : ℝ) 
  (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16)
  (h3 : a + b + c + d = e * f * g) :
  min_value_expr a b c d e f g h = 64 := 
sorry

end min_value_64_l155_155770


namespace log_ordering_l155_155042

theorem log_ordering 
  (a b c : ℝ) 
  (ha: a = Real.log 3 / Real.log 2) 
  (hb: b = Real.log 2 / Real.log 3) 
  (hc: c = Real.log 0.5 / Real.log 10) : 
  a > b ∧ b > c := 
by 
  sorry

end log_ordering_l155_155042


namespace acme_profit_l155_155006

-- Define the given problem conditions
def initial_outlay : ℝ := 12450
def cost_per_set : ℝ := 20.75
def selling_price_per_set : ℝ := 50
def num_sets : ℝ := 950

-- Define the total revenue and total manufacturing costs
def total_revenue : ℝ := num_sets * selling_price_per_set
def total_cost : ℝ := initial_outlay + (cost_per_set * num_sets)

-- State the profit calculation and the expected result
def profit : ℝ := total_revenue - total_cost

theorem acme_profit : profit = 15337.50 := by
  -- Proof goes here
  sorry

end acme_profit_l155_155006


namespace intersection_is_expected_result_l155_155729

def set_A : Set ℝ := { x | x * (x + 1) > 0 }
def set_B : Set ℝ := { x | ∃ y, y = Real.sqrt (x - 1) }
def expected_result : Set ℝ := { x | x ≥ 1 }

theorem intersection_is_expected_result : set_A ∩ set_B = expected_result := by
  sorry

end intersection_is_expected_result_l155_155729


namespace prob1_prob2_l155_155669

-- Define lines l1 and l2
def l1 (x y m : ℝ) : Prop := x + m * y + 1 = 0
def l2 (x y m : ℝ) : Prop := (m - 3) * x - 2 * y + (13 - 7 * m) = 0

-- Perpendicular condition
def perp_cond (m : ℝ) : Prop := 1 * (m - 3) - 2 * m = 0

-- Parallel condition
def parallel_cond (m : ℝ) : Prop := m * (m - 3) + 2 = 0

-- Distance between parallel lines when m = 1
def distance_between_parallel_lines (d : ℝ) : Prop := d = 2 * Real.sqrt 2

-- Problem 1: Prove that if l1 ⊥ l2, then m = -3
theorem prob1 (m : ℝ) (h : perp_cond m) : m = -3 := sorry

-- Problem 2: Prove that if l1 ∥ l2, the distance d is 2√2
theorem prob2 (m : ℝ) (h1 : parallel_cond m) (d : ℝ) (h2 : m = 1 ∨ m = -2) (h3 : m = 1) (h4 : distance_between_parallel_lines d) : d = 2 * Real.sqrt 2 := sorry

end prob1_prob2_l155_155669


namespace no_nonzero_integers_satisfy_conditions_l155_155409

theorem no_nonzero_integers_satisfy_conditions :
  ¬ ∃ a b x y : ℤ, (a ≠ 0 ∧ b ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0) ∧ (a * x - b * y = 16) ∧ (a * y + b * x = 1) :=
by
  sorry

end no_nonzero_integers_satisfy_conditions_l155_155409


namespace pipe_fill_time_without_leakage_l155_155334

theorem pipe_fill_time_without_leakage (t : ℕ) (h1 : 7 * t * (1/t - 1/70) = 1) : t = 60 :=
by
  sorry

end pipe_fill_time_without_leakage_l155_155334


namespace puzzles_sold_correct_l155_155103

def science_kits_sold : ℕ := 45
def puzzles_sold : ℕ := science_kits_sold - 9

theorem puzzles_sold_correct : puzzles_sold = 36 := by
  -- Proof will be provided here
  sorry

end puzzles_sold_correct_l155_155103


namespace inequalities_for_m_gt_n_l155_155026

open Real

theorem inequalities_for_m_gt_n (m n : ℕ) (hmn : m > n) : 
  (1 + 1 / (m : ℝ)) ^ m > (1 + 1 / (n : ℝ)) ^ n ∧ 
  (1 + 1 / (m : ℝ)) ^ (m + 1) < (1 + 1 / (n : ℝ)) ^ (n + 1) := 
by
  sorry

end inequalities_for_m_gt_n_l155_155026


namespace black_ball_on_second_draw_given_white_ball_on_first_draw_l155_155257

def num_white_balls : ℕ := 4
def num_black_balls : ℕ := 5
def total_balls : ℕ := num_white_balls + num_black_balls

def P_A : ℚ := num_white_balls / total_balls
def P_AB : ℚ := (num_white_balls * num_black_balls) / (total_balls * (total_balls - 1))
def P_B_given_A : ℚ := P_AB / P_A

theorem black_ball_on_second_draw_given_white_ball_on_first_draw : P_B_given_A = 5 / 8 :=
by
  sorry

end black_ball_on_second_draw_given_white_ball_on_first_draw_l155_155257


namespace compute_expression_l155_155333

theorem compute_expression : 85 * 1305 - 25 * 1305 + 100 = 78400 := by
  sorry

end compute_expression_l155_155333


namespace factorization_correct_l155_155492

variable (a b : ℝ)

theorem factorization_correct :
  12 * a ^ 3 * b - 12 * a ^ 2 * b + 3 * a * b = 3 * a * b * (2 * a - 1) ^ 2 :=
by 
  sorry

end factorization_correct_l155_155492


namespace farmer_harvested_correctly_l155_155175

def estimated_harvest : ℕ := 213489
def additional_harvest : ℕ := 13257
def total_harvest : ℕ := 226746

theorem farmer_harvested_correctly :
  estimated_harvest + additional_harvest = total_harvest :=
by
  sorry

end farmer_harvested_correctly_l155_155175


namespace equivalent_fractions_l155_155466

variable {x y a c : ℝ}

theorem equivalent_fractions (h_nonzero_c : c ≠ 0) (h_transform : x = (a / c) * y) :
  (x + a) / (y + c) = a / c :=
by
  sorry

end equivalent_fractions_l155_155466


namespace range_of_a_l155_155642

noncomputable def matrix_det_2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem range_of_a : 
  {a : ℝ | matrix_det_2x2 (a^2) 1 3 2 < matrix_det_2x2 a 0 4 1} = {a : ℝ | -1 < a ∧ a < 3/2} :=
by
  sorry

end range_of_a_l155_155642


namespace percentage_of_students_attend_chess_class_l155_155454

-- Definitions based on the conditions
def total_students : ℕ := 1000
def swimming_students : ℕ := 125
def chess_to_swimming_ratio : ℚ := 1 / 2

-- Problem statement
theorem percentage_of_students_attend_chess_class :
  ∃ P : ℚ, (P / 100) * total_students / 2 = swimming_students → P = 25 := by
  sorry

end percentage_of_students_attend_chess_class_l155_155454


namespace excircle_problem_l155_155773

-- Define the data structure for a triangle with incenter and excircle properties
structure TriangleWithIncenterAndExcircle (α : Type) [LinearOrderedField α] :=
  (A B C I X : α)
  (is_incenter : Boolean)  -- condition for point I being the incenter
  (is_excircle_center_opposite_A : Boolean)  -- condition for point X being the excircle center opposite A
  (I_A_I : I ≠ A)
  (X_A_X : X ≠ A)

-- Define the problem statement
theorem excircle_problem
  (α : Type) [LinearOrderedField α]
  (T : TriangleWithIncenterAndExcircle α)
  (h_incenter : T.is_incenter)
  (h_excircle_center : T.is_excircle_center_opposite_A)
  (h_not_eq_I : T.I ≠ T.A)
  (h_not_eq_X : T.X ≠ T.A)
  : 
    (T.I * T.X = T.A * T.B) ∧ 
    (T.I * (T.B * T.C) = T.X * (T.B * T.C)) :=
by
  sorry

end excircle_problem_l155_155773


namespace gcd_factorials_l155_155185

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  -- Proof steps would go here
  sorry

end gcd_factorials_l155_155185


namespace mn_value_l155_155581
open Real

-- Define the conditions
def L_1_scenario_1 (m n : ℝ) : Prop :=
  ∃ (θ₁ θ₂ : ℝ), θ₁ = 2 * θ₂ ∧ m = tan θ₁ ∧ n = tan θ₂ ∧ m = 4 * n

-- State the theorem
theorem mn_value (m n : ℝ) (hL1 : L_1_scenario_1 m n) (hm : m ≠ 0) : m * n = 2 :=
  sorry

end mn_value_l155_155581


namespace mother_stickers_given_l155_155686

-- Definitions based on the conditions
def initial_stickers : ℝ := 20.0
def bought_stickers : ℝ := 26.0
def birthday_stickers : ℝ := 20.0
def sister_stickers : ℝ := 6.0
def total_stickers : ℝ := 130.0

-- Statement of the problem to be proved in Lean 4.
theorem mother_stickers_given :
  initial_stickers + bought_stickers + birthday_stickers + sister_stickers + 58.0 = total_stickers :=
by
  sorry

end mother_stickers_given_l155_155686


namespace value_of_x_l155_155582

variable {x y z : ℤ}

theorem value_of_x
  (h1 : x + y = 31)
  (h2 : y + z = 47)
  (h3 : x + z = 52)
  (h4 : y + z = x + 16) :
  x = 31 := by
  sorry

end value_of_x_l155_155582


namespace infinitely_many_positive_integers_l155_155857

theorem infinitely_many_positive_integers (k : ℕ) (m := 13 * k + 1) (h : m ≠ 8191) :
  8191 = 2 ^ 13 - 1 → ∃ (m : ℕ), ∀ k : ℕ, (13 * k + 1) ≠ 8191 ∧ ∃ (t : ℕ), (2 ^ (13 * k) - 1) = 8191 * m * t := by
  intros
  sorry

end infinitely_many_positive_integers_l155_155857


namespace expansion_coeff_sum_l155_155076

theorem expansion_coeff_sum
  (a : ℕ → ℤ)
  (h : ∀ x y : ℤ, (x - 2 * y) ^ 5 * (x + 3 * y) ^ 4 = 
    a 9 * x ^ 9 + 
    a 8 * x ^ 8 * y + 
    a 7 * x ^ 7 * y ^ 2 + 
    a 6 * x ^ 6 * y ^ 3 + 
    a 5 * x ^ 5 * y ^ 4 + 
    a 4 * x ^ 4 * y ^ 5 + 
    a 3 * x ^ 3 * y ^ 6 + 
    a 2 * x ^ 2 * y ^ 7 + 
    a 1 * x * y ^ 8 + 
    a 0 * y ^ 9) :
  a 0 + a 8 = -2602 := by
  sorry

end expansion_coeff_sum_l155_155076


namespace angle_terminal_side_eq_l155_155181

noncomputable def has_same_terminal_side (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * k * Real.pi

theorem angle_terminal_side_eq (k : ℤ) :
  has_same_terminal_side (- (Real.pi / 3)) (5 * Real.pi / 3) :=
by
  use 1
  sorry

end angle_terminal_side_eq_l155_155181


namespace P_zero_value_l155_155624

noncomputable def P (x b c : ℚ) : ℚ := x ^ 2 + b * x + c

theorem P_zero_value (b c : ℚ)
  (h1 : P (P 1 b c) b c = 0)
  (h2 : P (P (-2) b c) b c = 0)
  (h3 : P 1 b c ≠ P (-2) b c) :
  P 0 b c = -5 / 2 :=
sorry

end P_zero_value_l155_155624


namespace lily_remaining_milk_l155_155197

def initial_milk : ℚ := (11 / 2)
def given_away : ℚ := (17 / 4)
def remaining_milk : ℚ := initial_milk - given_away

theorem lily_remaining_milk : remaining_milk = 5 / 4 :=
by
  -- Here, we would provide the proof steps, but we can use sorry to skip it.
  exact sorry

end lily_remaining_milk_l155_155197


namespace tan_alpha_minus_pi_over_4_l155_155591

theorem tan_alpha_minus_pi_over_4 (α : Real) (h1 : 0 < α) (h2 : α < π) (h3 : Real.sin α = 3 / 5) :
  (Real.tan (α - π / 4) = -1/7) ∨ (Real.tan (α - π / 4) = -7) :=
by
  sorry

end tan_alpha_minus_pi_over_4_l155_155591


namespace rate_of_change_area_at_t4_l155_155795

variable (t : ℝ)

def a (t : ℝ) : ℝ := 2 * t + 1

def b (t : ℝ) : ℝ := 3 * t + 2

def S (t : ℝ) : ℝ := a t * b t

theorem rate_of_change_area_at_t4 :
  (deriv S 4) = 55 := by
  sorry

end rate_of_change_area_at_t4_l155_155795


namespace sqrt_meaningful_range_l155_155211

theorem sqrt_meaningful_range (x : ℝ): x + 2 ≥ 0 ↔ x ≥ -2 := by
  sorry

end sqrt_meaningful_range_l155_155211


namespace tan_alpha_is_neg_5_over_12_l155_155852

variables (α : ℝ) (h1 : Real.sin α = 5/13) (h2 : π/2 < α ∧ α < π)

theorem tan_alpha_is_neg_5_over_12 : Real.tan α = -5/12 :=
by
  sorry

end tan_alpha_is_neg_5_over_12_l155_155852


namespace cost_of_math_books_l155_155224

theorem cost_of_math_books (M : ℕ) : 
  (∃ (total_books math_books history_books total_cost : ℕ),
    total_books = 90 ∧
    math_books = 60 ∧
    history_books = total_books - math_books ∧
    history_books * 5 + math_books * M = total_cost ∧
    total_cost = 390) → 
  M = 4 :=
by
  -- We provide the assumed conditions
  intro h
  -- We will skip the proof with sorry
  sorry

end cost_of_math_books_l155_155224


namespace non_real_roots_interval_l155_155057

theorem non_real_roots_interval (b : ℝ) : (b^2 < 64) ↔ (b > -8 ∧ b < 8) :=
by
  sorry

end non_real_roots_interval_l155_155057


namespace distinct_intersection_points_l155_155784

theorem distinct_intersection_points :
  let S1 := { p : ℝ × ℝ | (p.1 + p.2 - 7) * (2 * p.1 - 3 * p.2 + 9) = 0 }
  let S2 := { p : ℝ × ℝ | (p.1 - p.2 - 2) * (4 * p.1 + 3 * p.2 - 18) = 0 }
  ∃! (p1 p2 p3 : ℝ × ℝ), p1 ∈ S1 ∧ p1 ∈ S2 ∧ p2 ∈ S1 ∧ p2 ∈ S2 ∧ p3 ∈ S1 ∧ p3 ∈ S2 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 :=
sorry

end distinct_intersection_points_l155_155784


namespace find_c_for_root_ratio_l155_155572

theorem find_c_for_root_ratio :
  ∃ c : ℝ, (∀ x1 x2 : ℝ, (4 * x1^2 - 5 * x1 + c = 0) ∧ (x1 / x2 = -3 / 4)) → c = -75 := 
by {
  sorry
}

end find_c_for_root_ratio_l155_155572


namespace janice_initial_sentences_l155_155417

theorem janice_initial_sentences:
  ∀ (r t1 t2 t3 t4: ℕ), 
  r = 6 → 
  t1 = 20 → 
  t2 = 15 → 
  t3 = 40 → 
  t4 = 18 → 
  (t1 * r + t2 * r + t4 * r - t3 = 536 - 258) → 
  536 - (t1 * r + t2 * r + t4 * r - t3) = 258 := by
  intros
  sorry

end janice_initial_sentences_l155_155417


namespace four_distinct_real_solutions_l155_155095

noncomputable def polynomial (a b c d e x : ℝ) : ℝ :=
  (x - a) * (x - b) * (x - c) * (x - d) * (x - e)

noncomputable def derivative (a b c d e x : ℝ) : ℝ :=
  (x - b) * (x - c) * (x - d) * (x - e) + 
  (x - a) * (x - c) * (x - d) * (x - e) + 
  (x - a) * (x - b) * (x - d) * (x - e) +
  (x - a) * (x - b) * (x - c) * (x - e) +
  (x - a) * (x - b) * (x - c) * (x - d)

theorem four_distinct_real_solutions (a b c d e : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :
  ∃ x1 x2 x3 x4 : ℝ, 
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
    (derivative a b c d e x1 = 0 ∧ derivative a b c d e x2 = 0 ∧ derivative a b c d e x3 = 0 ∧ derivative a b c d e x4 = 0) :=
sorry

end four_distinct_real_solutions_l155_155095


namespace find_a_b_find_max_m_l155_155460

-- Define the function
def f (x : ℝ) : ℝ := abs (2 * x + 1) + abs (3 * x - 2)

-- Conditions
def solution_set_condition (x a : ℝ) : Prop := (-4 * a / 5 ≤ x ∧ x ≤ 3 * a / 5)
def eq_five_condition (x : ℝ) : Prop := f x ≤ 5

-- Prove the values of a and b
theorem find_a_b (a b : ℝ) : (∀ x : ℝ, eq_five_condition x ↔ solution_set_condition x a) → (a = 1 ∧ b = 2) :=
by
  sorry

-- Prove that |x - a| + |x + b| >= m^2 - 3m and find the maximum value of m
theorem find_max_m (a b m : ℝ) : (a = 1 ∧ b = 2) →
  (∀ x : ℝ, abs (x - a) + abs (x + b) ≥ m^2 - 3 * m) →
  m ≤ (3 + Real.sqrt 21) / 2 :=
by
  sorry


end find_a_b_find_max_m_l155_155460


namespace smallest_integer_l155_155905

theorem smallest_integer (M : ℕ) :
  (M % 4 = 3) ∧ (M % 5 = 4) ∧ (M % 6 = 5) ∧ (M % 7 = 6) ∧
  (M % 8 = 7) ∧ (M % 9 = 8) → M = 2519 :=
by sorry

end smallest_integer_l155_155905


namespace students_at_end_of_year_l155_155420

-- Define the initial number of students
def initial_students : Nat := 10

-- Define the number of students who left during the year
def students_left : Nat := 4

-- Define the number of new students who arrived during the year
def new_students : Nat := 42

-- Proof problem: the number of students at the end of the year
theorem students_at_end_of_year : initial_students - students_left + new_students = 48 := by
  sorry

end students_at_end_of_year_l155_155420


namespace fraction_of_book_finished_l155_155794

variables (x y : ℝ)

theorem fraction_of_book_finished (h1 : x = y + 90) (h2 : x + y = 270) : x / 270 = 2 / 3 :=
by sorry

end fraction_of_book_finished_l155_155794


namespace length_of_box_l155_155249

theorem length_of_box (v : ℝ) (w : ℝ) (h : ℝ) (l : ℝ) (conversion_factor : ℝ) (v_gallons : ℝ)
  (h_inch : ℝ) (conversion_inches_feet : ℝ) :
  v_gallons / conversion_factor = v → 
  h_inch / conversion_inches_feet = h →
  v = l * w * h →
  w = 25 →
  v_gallons = 4687.5 →
  conversion_factor = 7.5 →
  h_inch = 6 →
  conversion_inches_feet = 12 →
  l = 50 :=
by
  sorry

end length_of_box_l155_155249


namespace boat_speed_l155_155080

theorem boat_speed (b s : ℝ) (h1 : b + s = 7) (h2 : b - s = 5) : b = 6 := 
by
  sorry

end boat_speed_l155_155080


namespace quadratic_inequality_l155_155616

theorem quadratic_inequality
  (a b c : ℝ)
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : c ≠ 0)
  (h : ∀ x : ℝ, a * x^2 + b * x + c > c * x) :
  ∀ x : ℝ, c * x^2 - b * x + a > c * x - b :=
by
  sorry

end quadratic_inequality_l155_155616


namespace total_money_l155_155001

def Billy_money (S : ℕ) := 3 * S - 150
def Lila_money (B S : ℕ) := B - S

theorem total_money (S B L : ℕ) (h1 : B = Billy_money S) (h2 : S = 200) (h3 : L = Lila_money B S) : 
  S + B + L = 900 :=
by
  -- The proof would go here.
  sorry

end total_money_l155_155001


namespace school_spent_440_l155_155518

-- Definition based on conditions listed in part a)
def cost_of_pencils (cartons_pencils : ℕ) (boxes_per_carton_pencils : ℕ) (cost_per_box_pencils : ℕ) : ℕ := 
  cartons_pencils * boxes_per_carton_pencils * cost_per_box_pencils

def cost_of_markers (cartons_markers : ℕ) (cost_per_carton_markers : ℕ) : ℕ := 
  cartons_markers * cost_per_carton_markers

noncomputable def total_cost (cartons_pencils cartons_markers boxes_per_carton_pencils cost_per_box_pencils cost_per_carton_markers : ℕ) : ℕ := 
  cost_of_pencils cartons_pencils boxes_per_carton_pencils cost_per_box_pencils + 
  cost_of_markers cartons_markers cost_per_carton_markers

-- Theorem statement to prove the total cost is $440 given the conditions
theorem school_spent_440 : total_cost 20 10 10 2 4 = 440 := by 
  sorry

end school_spent_440_l155_155518


namespace solution_set_of_inequality_l155_155455

theorem solution_set_of_inequality (x : ℝ) :
  |x^2 - 2| < 2 ↔ (-2 < x ∧ x < 0) ∨ (0 < x ∧ x < 2) :=
sorry

end solution_set_of_inequality_l155_155455


namespace equivalent_statements_l155_155433

variable (P Q : Prop)

theorem equivalent_statements :
  (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by
  sorry

end equivalent_statements_l155_155433


namespace f_of_3_l155_155628

theorem f_of_3 (f : ℕ → ℕ) (h : ∀ x, f (x + 1) = 2 * x + 3) : f 3 = 7 := 
sorry

end f_of_3_l155_155628


namespace rectangle_area_l155_155325

theorem rectangle_area (x : ℝ) (h : (2*x - 3) * (3*x + 4) = 20 * x - 12) : x = 7 / 2 :=
sorry

end rectangle_area_l155_155325


namespace adams_father_total_amount_l155_155658

noncomputable def annual_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

noncomputable def total_interest (annual_interest : ℝ) (years : ℝ) : ℝ :=
  annual_interest * years

noncomputable def total_amount (principal : ℝ) (total_interest : ℝ) : ℝ :=
  principal + total_interest

theorem adams_father_total_amount :
  let principal := 2000
  let rate := 0.08
  let years := 2.5
  let annualInterest := annual_interest principal rate
  let interest := total_interest annualInterest years
  let amount := total_amount principal interest
  amount = 2400 :=
by sorry

end adams_father_total_amount_l155_155658


namespace sqrt_arithmetic_identity_l155_155887

theorem sqrt_arithmetic_identity : 4 * (Real.sqrt 2) * (Real.sqrt 3) - (Real.sqrt 12) / (Real.sqrt 2) + (Real.sqrt 24) = 5 * (Real.sqrt 6) := by
  sorry

end sqrt_arithmetic_identity_l155_155887


namespace sum_of_coefficients_l155_155348

-- Define the polynomial expansion and the target question
theorem sum_of_coefficients
  (x : ℝ)
  (b_6 b_5 b_4 b_3 b_2 b_1 b_0 : ℝ)
  (h : (5 * x - 2) ^ 6 = b_6 * x ^ 6 + b_5 * x ^ 5 + b_4 * x ^ 4 + 
                        b_3 * x ^ 3 + b_2 * x ^ 2 + b_1 * x + b_0) :
  (b_6 + b_5 + b_4 + b_3 + b_2 + b_1 + b_0) = 729 :=
by {
  -- We substitute x = 1 and show that the polynomial equals 729
  sorry
}

end sum_of_coefficients_l155_155348


namespace exists_nat_expressed_as_sum_of_powers_l155_155299

theorem exists_nat_expressed_as_sum_of_powers 
  (P : Finset ℕ) (hP : ∀ p ∈ P, Nat.Prime p) :
  ∃ x : ℕ, (∀ p ∈ P, ∃ a b : ℕ, x = a^p + b^p) ∧ (∀ p : ℕ, Nat.Prime p → p ∉ P → ¬∃ a b : ℕ, x = a^p + b^p) :=
by
  let x := 2^(P.val.prod + 1)
  use x
  sorry

end exists_nat_expressed_as_sum_of_powers_l155_155299


namespace sandwich_cost_proof_l155_155229

/-- Definitions of ingredient costs and quantities. --/
def bread_cost : ℝ := 0.15
def ham_cost : ℝ := 0.25
def cheese_cost : ℝ := 0.35
def mayo_cost : ℝ := 0.10
def lettuce_cost : ℝ := 0.05
def tomato_cost : ℝ := 0.08

def num_bread_slices : ℕ := 2
def num_ham_slices : ℕ := 2
def num_cheese_slices : ℕ := 2
def num_mayo_tbsp : ℕ := 1
def num_lettuce_leaf : ℕ := 1
def num_tomato_slices : ℕ := 2

/-- Calculation of the total cost in dollars and conversion to cents. --/
def sandwich_cost_in_dollars : ℝ :=
  (num_bread_slices * bread_cost) + 
  (num_ham_slices * ham_cost) + 
  (num_cheese_slices * cheese_cost) + 
  (num_mayo_tbsp * mayo_cost) + 
  (num_lettuce_leaf * lettuce_cost) + 
  (num_tomato_slices * tomato_cost)

def sandwich_cost_in_cents : ℝ :=
  sandwich_cost_in_dollars * 100

/-- Prove that the cost of the sandwich in cents is 181. --/
theorem sandwich_cost_proof : sandwich_cost_in_cents = 181 := by
  sorry

end sandwich_cost_proof_l155_155229


namespace middle_digit_zero_l155_155855

theorem middle_digit_zero (a b c M : ℕ) (h1 : M = 36 * a + 6 * b + c) (h2 : M = 64 * a + 8 * b + c) (ha : 0 ≤ a ∧ a < 6) (hb : 0 ≤ b ∧ b < 6) (hc : 0 ≤ c ∧ c < 6) : 
  b = 0 := 
  by sorry

end middle_digit_zero_l155_155855


namespace arith_seq_common_diff_l155_155380

/-
Given:
- an arithmetic sequence {a_n} with common difference d,
- the sum of the first n terms S_n = n * a_1 + n * (n - 1) / 2 * d,
- b_n = S_n / n,

Prove that the common difference of the sequence {a_n - b_n} is d/2.
-/

theorem arith_seq_common_diff (a b : ℕ → ℚ) (a1 d : ℚ) 
  (h1 : ∀ n, a n = a1 + n * d) 
  (h2 : ∀ n, b n = (a1 + n - 1 * d + n * (n - 1) / 2 * d) / n) : 
  ∀ n, (a n - b n) - (a (n + 1) - b (n + 1)) = d / 2 := 
    sorry

end arith_seq_common_diff_l155_155380


namespace expression_evaluation_l155_155630

def a : ℚ := 8 / 9
def b : ℚ := 5 / 6
def c : ℚ := 2 / 3
def d : ℚ := -5 / 18
def lhs : ℚ := (a - b + c) / d
def rhs : ℚ := -13 / 5

theorem expression_evaluation : lhs = rhs := by
  sorry

end expression_evaluation_l155_155630


namespace min_value_18_solve_inequality_l155_155797

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (1/a^3) + (1/b^3) + (1/c^3) + 27 * a * b * c

theorem min_value_18 (a b c : ℝ) (h : a > 0) (h' : b > 0) (h'' : c > 0) :
  min_value a b c ≥ 18 :=
by sorry

theorem solve_inequality (x : ℝ) :
  abs (x + 1) - 2 * x < 18 ↔ x > -(19/3) :=
by sorry

end min_value_18_solve_inequality_l155_155797


namespace count_negative_numbers_l155_155736

theorem count_negative_numbers : 
  (List.filter (λ x => x < (0:ℚ)) [-14, 7, 0, -2/3, -5/16]).length = 3 := 
by
  sorry

end count_negative_numbers_l155_155736


namespace div_fractions_eq_l155_155927

theorem div_fractions_eq : (3/7) / (5/2) = 6/35 := 
by sorry

end div_fractions_eq_l155_155927


namespace sum_of_consecutive_integers_product_384_l155_155771

theorem sum_of_consecutive_integers_product_384 :
  ∃ (a : ℤ), a * (a + 1) * (a + 2) = 384 ∧ a + (a + 1) + (a + 2) = 24 :=
by
  sorry

end sum_of_consecutive_integers_product_384_l155_155771


namespace orange_slices_needed_l155_155746

theorem orange_slices_needed (total_slices containers_capacity leftover_slices: ℕ) 
(h1 : containers_capacity = 4) 
(h2 : total_slices = 329) 
(h3 : leftover_slices = 1) :
    containers_capacity - leftover_slices = 3 :=
by
  sorry

end orange_slices_needed_l155_155746


namespace sum_last_two_digits_pow_mod_eq_zero_l155_155864

/-
Given condition: 
Sum of the last two digits of \( 9^{25} + 11^{25} \)
-/
theorem sum_last_two_digits_pow_mod_eq_zero : 
  let a := 9
  let b := 11
  let n := 25 
  (a ^ n + b ^ n) % 100 = 0 :=
by
  sorry

end sum_last_two_digits_pow_mod_eq_zero_l155_155864


namespace inequality_always_holds_l155_155502

noncomputable def range_for_inequality (k : ℝ) : Prop :=
  0 < k ∧ k ≤ 2 * Real.sqrt (2 + Real.sqrt 5)

theorem inequality_always_holds (x y k : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y = k) :
  (x + 1/x) * (y + 1/y) ≥ (k/2 + 2/k)^2 ↔ range_for_inequality k :=
sorry

end inequality_always_holds_l155_155502


namespace kiera_fruit_cups_l155_155280

def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3
def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2
def kiera_muffins : ℕ := 2
def total_cost : ℕ := 17

theorem kiera_fruit_cups : ∃ kiera_fruit_cups : ℕ, muffin_cost * kiera_muffins + fruit_cup_cost * kiera_fruit_cups = total_cost - (muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups) :=
by
  let francis_cost := muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups
  let remaining_cost := total_cost - francis_cost
  let kiera_fruit_cups := remaining_cost / fruit_cup_cost
  exact ⟨kiera_fruit_cups, by sorry⟩

end kiera_fruit_cups_l155_155280


namespace time_to_save_for_vehicle_l155_155977

def monthly_earnings : ℕ := 4000
def saving_factor : ℚ := 1 / 2
def vehicle_cost : ℕ := 16000

theorem time_to_save_for_vehicle : (vehicle_cost / (monthly_earnings * saving_factor)) = 8 := by
  sorry

end time_to_save_for_vehicle_l155_155977


namespace intersection_equiv_l155_155500

-- Define the sets M and N based on the given conditions
def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {x | x < 0 ∨ x > 2}

-- The main proof statement
theorem intersection_equiv : M ∩ N = {-1, 3} :=
by
  sorry -- proof goes here

end intersection_equiv_l155_155500


namespace prob_A_eq_prob_B_l155_155022

-- Define the number of students and the number of tickets
def num_students : ℕ := 56
def num_tickets : ℕ := 56
def prize_tickets : ℕ := 1

-- Define the probability of winning the prize for a given student (A for first student, B for last student)
def prob_A := prize_tickets / num_tickets
def prob_B := prize_tickets / num_tickets

-- Statement to prove
theorem prob_A_eq_prob_B : prob_A = prob_B :=
by 
  -- We provide the statement to prove without the proof steps
  sorry

end prob_A_eq_prob_B_l155_155022


namespace shaded_area_l155_155376

theorem shaded_area (r₁ r₂ r₃ : ℝ) (h₁ : 0 < r₁) (h₂ : 0 < r₂) (h₃ : 0 < r₃) (h₁₂ : r₁ < r₂) (h₂₃ : r₂ < r₃)
    (area_shaded_div_area_unshaded : (r₁^2 * π) + (r₂^2 * π) + (r₃^2 * π) = 77 * π)
    (shaded_by_unshaded_ratio : ∀ S U : ℝ, S = (3 / 7) * U) :
    ∃ S : ℝ, S = (1617 * π) / 70 :=
by
  sorry

end shaded_area_l155_155376


namespace art_club_artworks_l155_155533

theorem art_club_artworks (students : ℕ) (artworks_per_student_per_quarter : ℕ)
  (quarters_per_year : ℕ) (years : ℕ) :
  students = 15 → artworks_per_student_per_quarter = 2 → 
  quarters_per_year = 4 → years = 2 → 
  (students * artworks_per_student_per_quarter * quarters_per_year * years) = 240 :=
by
  intros
  sorry

end art_club_artworks_l155_155533


namespace arithmetic_sequence_sum_l155_155613

theorem arithmetic_sequence_sum : 
  ∀ (a : ℕ → ℝ) (d : ℝ), (a 1 = 2 ∨ a 1 = 8) → (a 2017 = 2 ∨ a 2017 = 8) → 
  (∀ n : ℕ, a (n + 1) = a n + d) →
  a 2 + a 1009 + a 2016 = 15 := 
by
  intro a d h1 h2017 ha
  sorry

end arithmetic_sequence_sum_l155_155613


namespace sales_on_second_day_l155_155831

variable (m : ℕ)

-- Define the condition for sales on the first day
def first_day_sales : ℕ := m

-- Define the condition for sales on the second day
def second_day_sales : ℕ := 2 * first_day_sales m - 3

-- The proof statement
theorem sales_on_second_day (m : ℕ) : second_day_sales m = 2 * m - 3 := by
  -- provide the actual proof here
  sorry

end sales_on_second_day_l155_155831


namespace solution_set_of_inequality_l155_155830

theorem solution_set_of_inequality (x : ℝ) (h : (2 * x - 1) / x < 0) : 0 < x ∧ x < 1 / 2 :=
by
  sorry

end solution_set_of_inequality_l155_155830


namespace solve_equation_1_solve_equation_2_l155_155499

theorem solve_equation_1 (x : ℝ) (h : 0.5 * x + 1.1 = 6.5 - 1.3 * x) : x = 3 :=
  by sorry

theorem solve_equation_2 (x : ℝ) (h : (1 / 6) * (3 * x - 9) = (2 / 5) * x - 3) : x = -15 :=
  by sorry

end solve_equation_1_solve_equation_2_l155_155499


namespace sine_triangle_sides_l155_155992

variable {α β γ : ℝ}

-- Given conditions: α, β, γ are angles of a triangle.
def is_triangle_angles (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi ∧ 0 < α ∧ α < Real.pi ∧
  0 < β ∧ β < Real.pi ∧ 0 < γ ∧ γ < Real.pi

-- The proof statement: Prove that there exists a triangle with sides sin α, sin β, sin γ
theorem sine_triangle_sides (h : is_triangle_angles α β γ) :
  ∃ (x y z : ℝ), x = Real.sin α ∧ y = Real.sin β ∧ z = Real.sin γ ∧
  (x + y > z) ∧ (x + z > y) ∧ (y + z > x) := sorry

end sine_triangle_sides_l155_155992


namespace Ann_age_is_39_l155_155694

def current_ages (A B : ℕ) : Prop :=
  A + B = 52 ∧ (B = 2 * B - A / 3) ∧ (A = 3 * B)

theorem Ann_age_is_39 : ∃ A B : ℕ, current_ages A B ∧ A = 39 :=
by
  sorry

end Ann_age_is_39_l155_155694


namespace julia_miles_l155_155365

theorem julia_miles (total_miles darius_miles julia_miles : ℕ) 
  (h1 : darius_miles = 679)
  (h2 : total_miles = 1677)
  (h3 : total_miles = darius_miles + julia_miles) :
  julia_miles = 998 :=
by
  sorry

end julia_miles_l155_155365


namespace max_sides_of_convex_polygon_with_4_obtuse_l155_155838

theorem max_sides_of_convex_polygon_with_4_obtuse (n : ℕ) (hn : n ≥ 3) :
  (∃ k : ℕ, k = 4 ∧
    ∀ θ : Fin n → ℝ, 
      (∀ p, θ p > 90 ∧ ∃ t, θ t = 180 ∨ θ t < 90 ∨ θ t = 90) →
      4 = k →
      n ≤ 7
  ) :=
sorry

end max_sides_of_convex_polygon_with_4_obtuse_l155_155838


namespace georgia_coughs_5_times_per_minute_l155_155195

-- Definitions
def georgia_coughs_per_minute (G : ℕ) := true
def robert_coughs_per_minute (G : ℕ) := 2 * G
def total_coughs (G : ℕ) := 20 * (G + 2 * G) = 300

-- Theorem to prove
theorem georgia_coughs_5_times_per_minute (G : ℕ) 
  (h1 : georgia_coughs_per_minute G) 
  (h2 : robert_coughs_per_minute G = 2 * G) 
  (h3 : total_coughs G) : G = 5 := 
sorry

end georgia_coughs_5_times_per_minute_l155_155195


namespace cakes_difference_l155_155424

theorem cakes_difference :
  let bought := 154
  let sold := 91
  bought - sold = 63 :=
by
  let bought := 154
  let sold := 91
  show bought - sold = 63
  sorry

end cakes_difference_l155_155424


namespace sqrt_div_val_l155_155172

theorem sqrt_div_val (n : ℕ) (h : n = 3600) : (Nat.sqrt n) / 15 = 4 := by 
  sorry

end sqrt_div_val_l155_155172


namespace correct_calculation_l155_155167

theorem correct_calculation (x y : ℝ) : 
  ¬(2 * x^2 + 3 * x^2 = 6 * x^2) ∧ 
  ¬(x^4 * x^2 = x^8) ∧ 
  ¬(x^6 / x^2 = x^3) ∧ 
  ((x * y^2)^2 = x^2 * y^4) :=
by
  sorry

end correct_calculation_l155_155167


namespace solve_eq1_solve_eq2_l155_155319

theorem solve_eq1 (x : ℝ) : x^2 - 6*x - 7 = 0 → x = 7 ∨ x = -1 :=
by
  sorry

theorem solve_eq2 (x : ℝ) : 3*x^2 - 1 = 2*x → x = 1 ∨ x = -1/3 :=
by
  sorry

end solve_eq1_solve_eq2_l155_155319


namespace cos_A_and_sin_2B_minus_A_l155_155386

variable (A B C a b c : ℝ)
variable (h1 : a * Real.sin A = 4 * b * Real.sin B)
variable (h2 : a * c = Real.sqrt 5 * (a^2 - b^2 - c^2))

theorem cos_A_and_sin_2B_minus_A :
  Real.cos A = -Real.sqrt 5 / 5 ∧ Real.sin (2 * B - A) = -2 * Real.sqrt 5 / 5 :=
by
  sorry

end cos_A_and_sin_2B_minus_A_l155_155386


namespace card_draw_probability_l155_155545

theorem card_draw_probability : 
  let P1 := (12 / 52 : ℚ) * (4 / 51 : ℚ) * (13 / 50 : ℚ)
  let P2 := (1 / 52 : ℚ) * (3 / 51 : ℚ) * (13 / 50 : ℚ)
  P1 + P2 = (63 / 107800 : ℚ) :=
by
  sorry

end card_draw_probability_l155_155545


namespace least_pos_int_N_l155_155012

theorem least_pos_int_N :
  ∃ N : ℕ, (N > 0) ∧ (N % 4 = 3) ∧ (N % 5 = 4) ∧ (N % 6 = 5) ∧ (N % 7 = 6) ∧ 
  (∀ m : ℕ, (m > 0) ∧ (m % 4 = 3) ∧ (m % 5 = 4) ∧ (m % 6 = 5) ∧ (m % 7 = 6) → N ≤ m) ∧ N = 419 :=
by
  sorry

end least_pos_int_N_l155_155012


namespace find_percentage_l155_155713

variable (dollars_1 dollars_2 dollars_total interest_total percentage_unknown : ℝ)
variable (investment_1 investment_rest interest_2 : ℝ)
variable (P : ℝ)

-- Assuming given conditions
axiom H1 : dollars_total = 12000
axiom H2 : dollars_1 = 5500
axiom H3 : interest_total = 970
axiom H4 : investment_rest = dollars_total - dollars_1
axiom H5 : interest_2 = investment_rest * 0.09
axiom H6 : interest_total = dollars_1 * P + interest_2

-- Prove that P = 0.07
theorem find_percentage : P = 0.07 :=
by
  -- Placeholder for the proof that needs to be filled in
  sorry

end find_percentage_l155_155713


namespace positive_difference_of_solutions_l155_155608

theorem positive_difference_of_solutions:
  ∀ (s : ℝ), s ≠ -3 → (s^2 - 5*s - 24) / (s + 3) = 3*s + 10 →
  abs (-1 - (-27)) = 26 :=
by
  sorry

end positive_difference_of_solutions_l155_155608


namespace min_value_frac_inv_l155_155944

theorem min_value_frac_inv {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  (∃ m, (∀ x y, x > 0 ∧ y > 0 ∧ x + y = 2 → m ≤ (1 / x + 1 / y)) ∧ (m = 2)) :=
by
  sorry

end min_value_frac_inv_l155_155944


namespace sum_S16_over_S4_l155_155903

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a q : α) (n : ℕ) := a * q^n

def sum_of_first_n_terms (a q : α) (n : ℕ) : α :=
if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem sum_S16_over_S4
  (a q : α)
  (hq : q ≠ 1)
  (h8_over_4 : sum_of_first_n_terms a q 8 / sum_of_first_n_terms a q 4 = 3) :
  sum_of_first_n_terms a q 16 / sum_of_first_n_terms a q 4 = 15 :=
sorry

end sum_S16_over_S4_l155_155903


namespace find_number_l155_155075

theorem find_number : ∃ n : ℕ, (∃ x : ℕ, x / 15 = 4 ∧ x^2 = n) ∧ n = 3600 := 
by
  sorry

end find_number_l155_155075


namespace john_cards_l155_155225

theorem john_cards (C : ℕ) (h1 : 15 * 2 + C * 2 = 70) : C = 20 :=
by
  sorry

end john_cards_l155_155225


namespace part_a_part_b_l155_155148

variable (a b c : ℤ)
variable (h : a + b + c = 0)

theorem part_a : (a^4 + b^4 + c^4) % (a^2 + b^2 + c^2) = 0 :=
by
  -- proof goes here
  sorry

theorem part_b : (a^100 + b^100 + c^100) % (a^2 + b^2 + c^2) = 0 :=
by
  -- proof goes here
  sorry

end part_a_part_b_l155_155148


namespace simple_interest_rate_l155_155989

theorem simple_interest_rate 
  (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ)
  (hSI : SI = 250) (hP : P = 1500) (hT : T = 5)
  (hSIFormula : SI = (P * R * T) / 100) :
  R = 3.33 := 
by 
  sorry

end simple_interest_rate_l155_155989


namespace efficiency_ratio_l155_155524

theorem efficiency_ratio (A B : ℝ) (h1 : A + B = 1 / 26) (h2 : B = 1 / 39) : A / B = 1 / 2 := 
by
  sorry

end efficiency_ratio_l155_155524


namespace smallest_k_l155_155719

theorem smallest_k :
  ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 :=
by
  sorry

end smallest_k_l155_155719


namespace license_plates_possible_l155_155606

open Function Nat

theorem license_plates_possible :
  let characters := ['B', 'C', 'D', '1', '2', '2', '5']
  let license_plate_length := 4
  let plate_count_with_two_twos := (choose 4 2) * (choose 5 2 * 2!)
  let plate_count_with_one_two := (choose 4 1) * (choose 5 3 * 3!)
  let plate_count_with_no_twos := (choose 5 4) * 4!
  let plate_count_with_three_twos := (choose 4 3) * (choose 4 1)
  plate_count_with_two_twos + plate_count_with_one_two + plate_count_with_no_twos + plate_count_with_three_twos = 496 := 
  sorry

end license_plates_possible_l155_155606


namespace circle_diameter_eq_l155_155846

-- Definitions
def line (x y : ℝ) : Prop := 3 * x - 4 * y + 12 = 0
def point_A (x y : ℝ) : Prop := x = 0 ∧ y = 3
def point_B (x y : ℝ) : Prop := x = -4 ∧ y = 0
def midpoint_AB (x y : ℝ) : Prop := x = -2 ∧ y = 3 / 2 -- Midpoint of A(0,3) and B(-4,0)
def diameter_AB : ℝ := 5

-- The equation of the circle with diameter AB
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 3 * y = 0

-- The proof statement
theorem circle_diameter_eq :
  (∃ A B : ℝ × ℝ, point_A A.1 A.2 ∧ point_B B.1 B.2 ∧ 
                   midpoint_AB ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) ∧ diameter_AB = 5) →
  (∀ x y : ℝ, circle_eq x y) :=
sorry

end circle_diameter_eq_l155_155846


namespace how_many_did_not_play_l155_155978

def initial_players : ℕ := 40
def first_half_starters : ℕ := 11
def first_half_substitutions : ℕ := 4
def second_half_extra_substitutions : ℕ := (first_half_substitutions * 3) / 4 -- 75% more substitutions
def injury_substitution : ℕ := 1
def total_second_half_substitutions : ℕ := first_half_substitutions + second_half_extra_substitutions + injury_substitution
def total_players_played : ℕ := first_half_starters + first_half_substitutions + total_second_half_substitutions
def players_did_not_play : ℕ := initial_players - total_players_played

theorem how_many_did_not_play : players_did_not_play = 17 := by
  sorry

end how_many_did_not_play_l155_155978


namespace cuboid_third_face_area_l155_155212

-- Problem statement in Lean
theorem cuboid_third_face_area (l w h : ℝ) (A₁ A₂ V : ℝ) 
  (hw1 : l * w = 120)
  (hw2 : w * h = 60)
  (hw3 : l * w * h = 720) : 
  l * h = 72 :=
sorry

end cuboid_third_face_area_l155_155212


namespace base_b_not_divisible_by_5_l155_155107

theorem base_b_not_divisible_by_5 (b : ℕ) : b = 4 ∨ b = 7 ∨ b = 8 → ¬ (5 ∣ (2 * b^2 * (b - 1))) :=
by
  sorry

end base_b_not_divisible_by_5_l155_155107


namespace f_expression_when_x_gt_1_l155_155868

variable (f : ℝ → ℝ)

-- conditions
def f_even : Prop := ∀ x, f (x + 1) = f (-x + 1)
def f_defn_when_x_lt_1 : Prop := ∀ x, x < 1 → f x = x ^ 2 + 1

-- theorem to prove
theorem f_expression_when_x_gt_1 (h_even : f_even f) (h_defn : f_defn_when_x_lt_1 f) : 
  ∀ x, x > 1 → f x = x ^ 2 - 4 * x + 5 := 
by
  sorry

end f_expression_when_x_gt_1_l155_155868


namespace standard_equation_of_circle_l155_155654

theorem standard_equation_of_circle :
  (∃ a r, r^2 = (a + 1)^2 + (a - 1)^2 ∧ r^2 = (a - 1)^2 + (a - 3)^2 ∧ a = 1 ∧ r^2 = 4) →
  ∃ r, (x - 1)^2 + (y - 1)^2 = r^2 :=
by
  intro h
  sorry

end standard_equation_of_circle_l155_155654


namespace students_prob_red_light_l155_155169

noncomputable def probability_red_light_encountered (p1 p2 p3 : ℚ) : ℚ :=
  1 - ((1 - p1) * (1 - p2) * (1 - p3))

theorem students_prob_red_light :
  probability_red_light_encountered (1/2) (1/3) (1/4) = 3/4 :=
by
  sorry

end students_prob_red_light_l155_155169


namespace sequence_sum_l155_155911

theorem sequence_sum :
  (∀ r: ℝ, 
    (∀ x y: ℝ,
      r = 1 / 4 → 
      x = 256 * r → 
      y = x * r → 
      x + y = 80)) :=
by sorry

end sequence_sum_l155_155911


namespace wage_increase_percentage_l155_155523

theorem wage_increase_percentage (new_wage old_wage : ℝ) (h1 : new_wage = 35) (h2 : old_wage = 25) : 
  ((new_wage - old_wage) / old_wage) * 100 = 40 := 
by
  sorry

end wage_increase_percentage_l155_155523


namespace value_of_x_squared_minus_y_squared_l155_155114

theorem value_of_x_squared_minus_y_squared (x y : ℝ) 
  (h₁ : x + y = 20) 
  (h₂ : x - y = 6) :
  x^2 - y^2 = 120 := 
by 
  sorry

end value_of_x_squared_minus_y_squared_l155_155114


namespace minimum_value_of_quadratic_l155_155668

theorem minimum_value_of_quadratic (p q : ℝ) (hp : 0 < p) (hq : 0 < q) : 
  ∃ x : ℝ, x = -p / 2 ∧ (∀ y : ℝ, (y - x) ^ 2 + 2*q ≥ (x ^ 2 + p * x + 2*q)) :=
by
  sorry

end minimum_value_of_quadratic_l155_155668


namespace Juliska_correct_l155_155007

-- Definitions according to the conditions in a)
def has_three_rum_candy (candies : List String) : Prop :=
  ∀ (selected_triplet : List String), selected_triplet.length = 3 → "rum" ∈ selected_triplet

def has_three_coffee_candy (candies : List String) : Prop :=
  ∀ (selected_triplet : List String), selected_triplet.length = 3 → "coffee" ∈ selected_triplet

-- Proof problem statement
theorem Juliska_correct 
  (candies : List String) 
  (h_rum : has_three_rum_candy candies)
  (h_coffee : has_three_coffee_candy candies) : 
  (∀ (selected_triplet : List String), selected_triplet.length = 3 → "walnut" ∈ selected_triplet) :=
sorry

end Juliska_correct_l155_155007


namespace profit_ratio_l155_155260

def praveen_initial_capital : ℝ := 3500
def hari_initial_capital : ℝ := 9000.000000000002
def total_months : ℕ := 12
def months_hari_invested : ℕ := total_months - 5

def effective_capital (initial_capital : ℝ) (months : ℕ) : ℝ :=
  initial_capital * months

theorem profit_ratio :
  effective_capital praveen_initial_capital total_months / effective_capital hari_initial_capital months_hari_invested 
  = 2 / 3 :=
by
  sorry

end profit_ratio_l155_155260


namespace sages_success_l155_155355

-- Assume we have a finite type representing our 1000 colors
inductive Color
| mk : Fin 1000 → Color

open Color

-- Define the sages
def Sage : Type := Fin 11

-- Define the problem conditions into a Lean structure
structure Problem :=
  (sages : Fin 11)
  (colors : Fin 1000)
  (assignments : Sage → Color)
  (strategies : Sage → (Fin 1024 → Fin 2))

-- Define the success condition
def success (p : Problem) : Prop :=
  ∃ (strategies : Sage → (Fin 1024 → Fin 2)),
    ∀ (assignment : Sage → Color),
      ∃ (color_guesses : Sage → Color),
        (∀ s, color_guesses s = assignment s)

-- The sages will succeed in determining the colors of their hats.
theorem sages_success : ∀ (p : Problem), success p := by
  sorry

end sages_success_l155_155355


namespace student_correct_answers_l155_155629

theorem student_correct_answers 
(C W : ℕ) 
(h1 : C + W = 80) 
(h2 : 4 * C - W = 120) : 
C = 40 :=
by
  sorry 

end student_correct_answers_l155_155629


namespace imo_42_problem_l155_155627

theorem imo_42_problem {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * a * c) + c / Real.sqrt (c^2 + 8 * a * b) >= 1 :=
sorry

end imo_42_problem_l155_155627


namespace no_real_roots_contradiction_l155_155748

open Real

variables (a b : ℝ)

theorem no_real_roots_contradiction (h : ∀ x : ℝ, a * x^3 + a * x + b ≠ 0) : false :=
by
  sorry

end no_real_roots_contradiction_l155_155748


namespace value_of_fraction_power_series_l155_155700

theorem value_of_fraction_power_series (x : ℕ) (h : x = 3) :
  (x^3 * x^5 * x^7 * x^9 * x^11 * x^13 * x^15 * x^17 * x^19 * x^21) /
  (x^4 * x^8 * x^12 * x^16 * x^20 * x^24) = 3^36 :=
by
  subst h
  sorry

end value_of_fraction_power_series_l155_155700


namespace chores_minutes_proof_l155_155205

-- Definitions based on conditions
def minutes_of_cartoon_per_hour := 60
def cartoon_watched_hours := 2
def cartoon_watched_minutes := cartoon_watched_hours * minutes_of_cartoon_per_hour
def ratio_of_cartoon_to_chores := 10 / 8

-- Definition based on the question
def chores_minutes (cartoon_minutes : ℕ) : ℕ := (8 * cartoon_minutes) / 10

theorem chores_minutes_proof : chores_minutes cartoon_watched_minutes = 96 := 
by sorry 

end chores_minutes_proof_l155_155205


namespace find_a_in_terms_of_x_l155_155832

variable (a b x : ℝ)

theorem find_a_in_terms_of_x (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = 3 * x) : a = 3 * x :=
sorry

end find_a_in_terms_of_x_l155_155832


namespace speed_downstream_l155_155136

def speed_in_still_water := 12 -- man in still water
def speed_of_stream := 6  -- speed of stream
def speed_upstream := 6  -- rowing upstream

theorem speed_downstream : 
  speed_in_still_water + speed_of_stream = 18 := 
by 
  sorry

end speed_downstream_l155_155136


namespace new_person_weight_l155_155491

theorem new_person_weight
  (avg_increase : ℝ) (original_person_weight : ℝ) (num_people : ℝ) (new_weight : ℝ)
  (h1 : avg_increase = 2.5)
  (h2 : original_person_weight = 85)
  (h3 : num_people = 8)
  (h4 : num_people * avg_increase = new_weight - original_person_weight):
    new_weight = 105 :=
by
  sorry

end new_person_weight_l155_155491


namespace solution_set_Inequality_l155_155093

theorem solution_set_Inequality : {x : ℝ | abs (1 + x + x^2 / 2) < 1} = {x : ℝ | -2 < x ∧ x < 0} :=
sorry

end solution_set_Inequality_l155_155093


namespace upper_bound_of_third_inequality_l155_155751

variable (x : ℤ)

theorem upper_bound_of_third_inequality : (3 < x ∧ x < 10) →
                                          (5 < x ∧ x < 18) →
                                          (∃ n, n > x ∧ x > -2) →
                                          (0 < x ∧ x < 8) →
                                          (x + 1 < 9) →
                                          x < 8 :=
by { sorry }

end upper_bound_of_third_inequality_l155_155751


namespace car_speed_l155_155176

theorem car_speed (v : ℝ) (hv : (1 / v * 3600) = (1 / 40 * 3600) + 10) : v = 36 := 
by
  sorry

end car_speed_l155_155176


namespace certain_number_is_two_l155_155646

variable (x : ℕ)  -- x is the certain number

-- Condition: Given that adding 6 incorrectly results in 8
axiom h1 : x + 6 = 8

-- The mathematically equivalent proof problem Lean statement
theorem certain_number_is_two : x = 2 :=
by
  sorry

end certain_number_is_two_l155_155646


namespace parabola_min_y1_y2_squared_l155_155666

theorem parabola_min_y1_y2_squared (x1 x2 y1 y2 : ℝ) :
  (y1^2 = 4 * x1) ∧
  (y2^2 = 4 * x2) ∧
  (x1 * x2 = 16) →
  (y1^2 + y2^2 ≥ 32) :=
by
  intro h
  sorry

end parabola_min_y1_y2_squared_l155_155666


namespace larry_final_channels_l155_155430

def initial_channels : Int := 150
def removed_channels : Int := 20
def replacement_channels : Int := 12
def reduced_channels : Int := 10
def sports_package_channels : Int := 8
def supreme_sports_package_channels : Int := 7

theorem larry_final_channels :
  initial_channels 
  - removed_channels 
  + replacement_channels 
  - reduced_channels 
  + sports_package_channels 
  + supreme_sports_package_channels 
  = 147 := by
  rfl  -- Reflects the direct computation as per the problem

end larry_final_channels_l155_155430


namespace angle_in_third_quadrant_l155_155025

theorem angle_in_third_quadrant (α : ℝ) (h : α = 2023) : 180 < α % 360 ∧ α % 360 < 270 := by
  sorry

end angle_in_third_quadrant_l155_155025


namespace new_average_after_doubling_l155_155968

theorem new_average_after_doubling (n : ℕ) (avg : ℝ) (h_n : n = 12) (h_avg : avg = 50) :
  2 * avg = 100 :=
by
  sorry

end new_average_after_doubling_l155_155968


namespace division_of_expressions_l155_155893

theorem division_of_expressions : 
  (2 * 3 + 4) / (2 + 3) = 2 :=
by
  sorry

end division_of_expressions_l155_155893


namespace shortest_player_height_correct_l155_155469

def tallest_player_height : Real := 77.75
def height_difference : Real := 9.5
def shortest_player_height : Real := 68.25

theorem shortest_player_height_correct :
  tallest_player_height - height_difference = shortest_player_height :=
by
  sorry

end shortest_player_height_correct_l155_155469


namespace intervals_union_l155_155373

open Set

noncomputable def I (a b : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < b}

theorem intervals_union {I1 I2 I3 : Set ℝ} (h1 : ∃ (a1 b1 : ℝ), I1 = I a1 b1)
  (h2 : ∃ (a2 b2 : ℝ), I2 = I a2 b2) (h3 : ∃ (a3 b3 : ℝ), I3 = I a3 b3)
  (h_non_empty : (I1 ∩ I2 ∩ I3).Nonempty) (h_not_contained : ¬ (I1 ⊆ I2) ∧ ¬ (I1 ⊆ I3) ∧ ¬ (I2 ⊆ I1) ∧ ¬ (I2 ⊆ I3) ∧ ¬ (I3 ⊆ I1) ∧ ¬ (I3 ⊆ I2)) :
  I1 ⊆ (I2 ∪ I3) ∨ I2 ⊆ (I1 ∪ I3) ∨ I3 ⊆ (I1 ∪ I2) :=
sorry

end intervals_union_l155_155373


namespace value_of_3Y5_l155_155330

def Y (a b : ℤ) : ℤ := b + 10 * a - a^2 - b^2

theorem value_of_3Y5 : Y 3 5 = 1 := sorry

end value_of_3Y5_l155_155330


namespace ordering_of_powers_l155_155761

theorem ordering_of_powers :
  (3:ℕ)^15 < 10^9 ∧ 10^9 < (5:ℕ)^13 :=
by
  sorry

end ordering_of_powers_l155_155761


namespace percentage_of_boy_scouts_with_signed_permission_slips_l155_155970

noncomputable def total_scouts : ℕ := 100 -- assume 100 scouts
noncomputable def total_signed_permission_slips : ℕ := 70 -- 70% of 100
noncomputable def boy_scouts : ℕ := 60 -- 60% of 100
noncomputable def girl_scouts : ℕ := 40 -- total_scouts - boy_scouts 

noncomputable def girl_scouts_signed_permission_slips : ℕ := girl_scouts * 625 / 1000 

theorem percentage_of_boy_scouts_with_signed_permission_slips :
  (boy_scouts * 75 / 100) = (total_signed_permission_slips - girl_scouts_signed_permission_slips) :=
by
  sorry

end percentage_of_boy_scouts_with_signed_permission_slips_l155_155970


namespace part1_part2a_part2b_l155_155808

-- Definitions and conditions
def vector_a : ℝ × ℝ := (1, -2)
def vector_b : ℝ × ℝ := (-3, 2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def scalar_mul (k : ℝ) (u : ℝ × ℝ) : ℝ × ℝ := (k * u.1, k * u.2)
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def collinear (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- Proof statements

-- Part 1: Verify the dot product computation
theorem part1 : dot_product (vector_add vector_a vector_b) (vector_sub vector_a vector_b) = -8 := by
  sorry

-- Part 2a: Verify the value of k for parallel vectors
theorem part2a : collinear (vector_add (scalar_mul (1/3) vector_a) vector_b) (vector_sub vector_a (scalar_mul 3 vector_b)) := by
  sorry

-- Part 2b: Verify antiparallel direction
theorem part2b : collinear (vector_add (scalar_mul (1/3) vector_a) vector_b) (scalar_mul (-1) (vector_sub vector_a (scalar_mul 3 vector_b))) := by
  sorry

end part1_part2a_part2b_l155_155808


namespace range_of_a_l155_155847

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (a - 1) * x < a - 1 ↔ x > 1) : a < 1 := 
sorry

end range_of_a_l155_155847


namespace maria_total_money_l155_155703

theorem maria_total_money (Rene Florence Isha : ℕ) (hRene : Rene = 300)
  (hFlorence : Florence = 3 * Rene) (hIsha : Isha = Florence / 2) :
  Isha + Florence + Rene = 1650 := by
  sorry

end maria_total_money_l155_155703


namespace number_of_tables_l155_155647

theorem number_of_tables (c t : ℕ) (h1 : c = 8 * t) (h2 : 4 * c + 3 * t = 759) : t = 22 := by
  sorry

end number_of_tables_l155_155647


namespace f_has_two_zeros_l155_155141

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + (a - 2) * x - Real.log x

theorem f_has_two_zeros (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0 := sorry

end f_has_two_zeros_l155_155141


namespace value_set_of_t_l155_155147

theorem value_set_of_t (t : ℝ) :
  (1 > 2 * (1) + 1 - t) ∧ (∀ x : ℝ, x^2 + (2*t-4)*x + 4 > 0) → 3 < t ∧ t < 4 :=
by
  intros h
  sorry

end value_set_of_t_l155_155147


namespace complex_solution_l155_155108

theorem complex_solution (z : ℂ) (i : ℂ) (h : i^2 = -1) (hz : (3 - 4 * i) * z = 5 * i) : z = (4 / 5) + (3 / 5) * i :=
by {
  sorry
}

end complex_solution_l155_155108


namespace problem_statement_l155_155459

variables (a b c : ℝ)

-- The conditions given in the problem
def condition1 : Prop := a^2 + b^2 - 4 * a ≤ 1
def condition2 : Prop := b^2 + c^2 - 8 * b ≤ -3
def condition3 : Prop := c^2 + a^2 - 12 * c ≤ -26

-- The theorem we need to prove
theorem problem_statement (h1 : condition1 a b) (h2 : condition2 b c) (h3 : condition3 c a) : (a + b) ^ c = 27 :=
by sorry

end problem_statement_l155_155459


namespace decreasing_interval_implies_a_ge_two_l155_155766

-- The function f is given
def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x - 3

-- Defining the condition for f(x) being decreasing in the interval (-8, 2)
def is_decreasing_in_interval (a : ℝ) : Prop :=
  ∀ x y : ℝ, (-8 < x ∧ x < y ∧ y < 2) → f x a > f y a

-- The proof statement
theorem decreasing_interval_implies_a_ge_two (a : ℝ) (h : is_decreasing_in_interval a) : a ≥ 2 :=
sorry

end decreasing_interval_implies_a_ge_two_l155_155766


namespace range_of_m_for_line_to_intersect_ellipse_twice_l155_155179

theorem range_of_m_for_line_to_intersect_ellipse_twice (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
   (A.2 = 4 * A.1 + m) ∧
   (B.2 = 4 * B.1 + m) ∧
   ((A.1 ^ 2) / 4 + (A.2 ^ 2) / 3 = 1) ∧
   ((B.1 ^ 2) / 4 + (B.2 ^ 2) / 3 = 1) ∧
   (A.1 + B.1) / 2 = 0 ∧ 
   (A.2 + B.2) / 2 = 4 * 0 + m) ↔
   - (2 * Real.sqrt 13) / 13 < m ∧ m < (2 * Real.sqrt 13) / 13
 :=
sorry

end range_of_m_for_line_to_intersect_ellipse_twice_l155_155179


namespace sufficient_not_necessary_condition_l155_155851

theorem sufficient_not_necessary_condition (a : ℝ) : (a = 2 → (a^2 - a) * 1 + 1 = 0) ∧ (¬ ((a^2 - a) * 1 + 1 = 0 → a = 2)) :=
by sorry

end sufficient_not_necessary_condition_l155_155851


namespace train_length_l155_155617

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (h1 : speed_kmh = 90) (h2 : time_s = 12) : 
  ∃ length_m : ℕ, length_m = 300 := 
by
  sorry

end train_length_l155_155617


namespace average_after_discard_l155_155507

theorem average_after_discard (avg : ℝ) (n : ℕ) (a b : ℝ) (new_avg : ℝ) :
  avg = 62 →
  n = 50 →
  a = 45 →
  b = 55 →
  new_avg = 62.5 →
  (avg * n - (a + b)) / (n - 2) = new_avg := 
by
  intros h_avg h_n h_a h_b h_new_avg
  rw [h_avg, h_n, h_a, h_b, h_new_avg]
  sorry

end average_after_discard_l155_155507


namespace increasing_exponential_function_range_l155_155290

theorem increasing_exponential_function_range (a : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ (x : ℝ), f x = a ^ x) 
    (h2 : a > 0)
    (h3 : a ≠ 1)
    (h4 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) : a > 1 := 
sorry

end increasing_exponential_function_range_l155_155290


namespace problem_I_solution_set_l155_155464

def f1 (x : ℝ) : ℝ := |2 * x| + |x - 1| -- since a = -1

theorem problem_I_solution_set :
  {x : ℝ | f1 x ≤ 4} = Set.Icc (-1 : ℝ) ((5 : ℝ) / 3) :=
sorry

end problem_I_solution_set_l155_155464


namespace completing_square_transformation_l155_155102

theorem completing_square_transformation (x : ℝ) :
  x^2 - 2 * x - 5 = 0 -> (x - 1)^2 = 6 :=
by {
  sorry -- Proof to be completed
}

end completing_square_transformation_l155_155102


namespace solve_for_x_l155_155925

theorem solve_for_x (x : ℝ) (h : 10 - x = 15) : x = -5 :=
by
  sorry

end solve_for_x_l155_155925


namespace sum_products_roots_l155_155105

theorem sum_products_roots :
  (∃ p q r : ℂ, (3 * p^3 - 5 * p^2 + 12 * p - 10 = 0) ∧
                  (3 * q^3 - 5 * q^2 + 12 * q - 10 = 0) ∧
                  (3 * r^3 - 5 * r^2 + 12 * r - 10 = 0) ∧
                  (p ≠ q) ∧ (q ≠ r) ∧ (p ≠ r)) →
  ∀ p q r : ℂ, (3 * p) * (q * r) + (3 * q) * (r * p) + (3 * r) * (p * q) =
    (3 * p * q * r) :=
sorry

end sum_products_roots_l155_155105


namespace ascending_order_l155_155405

theorem ascending_order : (3 / 8 : ℝ) < 0.75 ∧ 
                          0.75 < (1 + 2 / 5 : ℝ) ∧ 
                          (1 + 2 / 5 : ℝ) < 1.43 ∧
                          1.43 < (13 / 8 : ℝ) :=
by
  sorry

end ascending_order_l155_155405


namespace flavoring_ratio_comparison_l155_155256

theorem flavoring_ratio_comparison (f_st cs_st w_st : ℕ) (f_sp cs_sp w_sp : ℕ) :
  f_st = 1 → cs_st = 12 → w_st = 30 →
  w_sp = 75 → cs_sp = 5 →
  f_sp / w_sp = f_st / (2 * w_st) →
  (f_st / cs_st) * 3 = f_sp / cs_sp :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end flavoring_ratio_comparison_l155_155256


namespace vector_subtraction_result_l155_155449

-- definition of vectors as pairs of integers
def OA : ℝ × ℝ := (1, -2)
def OB : ℝ × ℝ := (-3, 1)

-- definition of vector subtraction for pairs of reals
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- definition of the vector AB as the subtraction of OB and OA
def AB : ℝ × ℝ := vector_sub OB OA

-- statement to assert the expected result
theorem vector_subtraction_result : AB = (-4, 3) :=
by
  -- this is where the proof would go, but we use sorry to skip it
  sorry

end vector_subtraction_result_l155_155449


namespace arabella_total_learning_time_l155_155836

-- Define the conditions
def arabella_first_step_time := 30 -- in minutes
def arabella_second_step_time := arabella_first_step_time / 2 -- half the time of the first step
def arabella_third_step_time := arabella_first_step_time + arabella_second_step_time -- sum of the first and second steps

-- Define the total time spent
def arabella_total_time := arabella_first_step_time + arabella_second_step_time + arabella_third_step_time

-- The theorem to prove
theorem arabella_total_learning_time : arabella_total_time = 90 := 
  sorry

end arabella_total_learning_time_l155_155836


namespace janet_saves_minutes_l155_155972

theorem janet_saves_minutes 
  (key_search_time : ℕ := 8) 
  (complaining_time : ℕ := 3) 
  (days_in_week : ℕ := 7) : 
  (key_search_time + complaining_time) * days_in_week = 77 := 
by
  sorry

end janet_saves_minutes_l155_155972


namespace problem_solution_l155_155907

theorem problem_solution {n : ℕ} :
  (∀ x y z : ℤ, x + y + z = 0 → ∃ k : ℤ, (x^n + y^n + z^n) / 2 = k^2) ↔ n = 1 ∨ n = 4 :=
by
  sorry

end problem_solution_l155_155907


namespace sum_of_triangle_ops_l155_155928

def triangle_op (a b c : ℕ) : ℕ := 2 * a + b - c 

theorem sum_of_triangle_ops : 
  triangle_op 1 2 3 + triangle_op 4 6 5 + triangle_op 2 7 1 = 20 :=
by
  sorry

end sum_of_triangle_ops_l155_155928


namespace polynomial_abs_sum_eq_81_l155_155408

theorem polynomial_abs_sum_eq_81 
  (a a_1 a_2 a_3 a_4 : ℝ) 
  (h : (1 - 2 * x)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4)
  (ha : a > 0) 
  (ha_2 : a_2 > 0) 
  (ha_4 : a_4 > 0) 
  (ha_1 : a_1 < 0) 
  (ha_3 : a_3 < 0): 
  |a| + |a_1| + |a_2| + |a_3| + |a_4| = 81 := 
by 
  sorry

end polynomial_abs_sum_eq_81_l155_155408


namespace sum_two_angles_greater_third_l155_155121

-- Definitions of the angles and the largest angle condition
variables {P A B C} -- Points defining the trihedral angle
variables {α β γ : ℝ} -- Angles α, β, γ
variables (h1 : γ ≥ α) (h2 : γ ≥ β)

-- Statement of the theorem
theorem sum_two_angles_greater_third (P A B C : Type*) (α β γ : ℝ)
  (h1 : γ ≥ α) (h2 : γ ≥ β) : α + β > γ :=
sorry  -- Proof is omitted

end sum_two_angles_greater_third_l155_155121


namespace decorations_cost_correct_l155_155200

def cost_of_decorations (num_tables : ℕ) (cost_tablecloth per_tablecloth : ℕ) (num_place_settings per_table : ℕ) (cost_place_setting per_setting : ℕ) (num_roses per_centerpiece : ℕ) (cost_rose per_rose : ℕ) (num_lilies per_centerpiece : ℕ) (cost_lily per_lily : ℕ) : ℕ :=
  let cost_roses := cost_rose * num_roses
  let cost_lilies := cost_lily * num_lilies
  let cost_settings := cost_place_setting * num_place_settings
  let cost_per_table := cost_roses + cost_lilies + cost_settings + cost_tablecloth
  num_tables * cost_per_table

theorem decorations_cost_correct :
  cost_of_decorations 20 25 4 10 10 5 15 4 = 3500 :=
by
  sorry

end decorations_cost_correct_l155_155200


namespace identify_solids_with_identical_views_l155_155681

def has_identical_views (s : Type) : Prop := sorry

def sphere : Type := sorry
def triangular_pyramid : Type := sorry
def cube : Type := sorry
def cylinder : Type := sorry

theorem identify_solids_with_identical_views :
  (has_identical_views sphere) ∧
  (¬ has_identical_views triangular_pyramid) ∧
  (has_identical_views cube) ∧
  (¬ has_identical_views cylinder) :=
sorry

end identify_solids_with_identical_views_l155_155681


namespace element_in_set_l155_155932

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def complement_U_M : Set ℕ := {1, 2}

-- The main statement to prove
theorem element_in_set (M : Set ℕ) (h1 : U = {1, 2, 3, 4, 5}) (h2 : U \ M = complement_U_M) : 3 ∈ M := 
sorry

end element_in_set_l155_155932


namespace domain_of_g_l155_155473

theorem domain_of_g (x y : ℝ) : 
  (∃ g : ℝ, g = 1 / (x^2 + (x - y)^2 + y^2)) ↔ (x, y) ≠ (0, 0) :=
by sorry

end domain_of_g_l155_155473


namespace tan_2A_cos_pi3_minus_A_l155_155206

variable (A : ℝ)

def line_equation (A : ℝ) : Prop :=
  (4 * Real.tan A = 3)

theorem tan_2A : line_equation A → Real.tan (2 * A) = -24 / 7 :=
by
  intro h 
  sorry

theorem cos_pi3_minus_A : (0 < A ∧ A < Real.pi) →
    Real.tan A = 4 / 3 →
    Real.cos (Real.pi / 3 - A) = (3 + 4 * Real.sqrt 3) / 10 :=
by
  intro h1 h2
  sorry

end tan_2A_cos_pi3_minus_A_l155_155206


namespace segment_length_l155_155279

theorem segment_length (x y : ℝ) (A B : ℝ × ℝ) 
  (h1 : A.2^2 = 4 * A.1) 
  (h2 : B.2^2 = 4 * B.1) 
  (h3 : A.2 = 2 * A.1 - 2)
  (h4 : B.2 = 2 * B.1 - 2)
  (h5 : A ≠ B) :
  dist A B = 5 :=
sorry

end segment_length_l155_155279


namespace find_r_l155_155237

theorem find_r 
  (r s : ℝ)
  (h1 : 9 * (r * r) * s = -6)
  (h2 : r * r + 2 * r * s = -16 / 3)
  (h3 : 2 * r + s = 2 / 3)
  (polynomial_condition : ∀ x : ℝ, 9 * x^3 - 6 * x^2 - 48 * x + 54 = 9 * (x - r)^2 * (x - s)) 
: r = -2 / 3 :=
sorry

end find_r_l155_155237


namespace proposition_1_proposition_3_l155_155690

variables {Line Plane : Type}
variables (m n : Line) (α β γ : Plane)

-- Condition predicates
def parallel (p q : Plane) : Prop := sorry -- parallelism of p and q
def perpendicular (p q : Plane) : Prop := sorry -- perpendicularly of p and q
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry -- parallelism of line and plane
def line_perpendicular_plane (l : Line) (p : Plane) : Prop := sorry -- perpendicularity of line and plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry -- line is in the plane

-- Proposition ①
theorem proposition_1 (h1 : parallel α β) (h2 : parallel α γ) : parallel β γ := sorry

-- Proposition ③
theorem proposition_3 (h1 : line_perpendicular_plane m α) (h2 : line_parallel_plane m β) : perpendicular α β := sorry

end proposition_1_proposition_3_l155_155690


namespace max_value_fraction_l155_155949

theorem max_value_fraction (a b : ℝ)
  (h1 : a + b - 2 ≥ 0)
  (h2 : b - a - 1 ≤ 0)
  (h3 : a ≤ 1) :
  (a ≠ 0) → (b ≠ 0) →
  ∃ m, m = (a + 2 * b) / (2 * a + b) ∧ m ≤ 7 / 5 :=
by
  sorry

end max_value_fraction_l155_155949


namespace shoveling_problem_l155_155863

variable (S : ℝ) -- Wayne's son's shoveling rate (driveways per hour)
variable (W : ℝ) -- Wayne's shoveling rate (driveways per hour)
variable (T : ℝ) -- Time it takes for Wayne's son to shovel the driveway alone (hours)

theorem shoveling_problem 
  (h1 : W = 6 * S)
  (h2 : (S + W) * 3 = 1) : T = 21 := 
by
  sorry

end shoveling_problem_l155_155863


namespace pencils_purchased_l155_155003

theorem pencils_purchased (n : ℕ) (h1: n ≤ 10) 
  (h2: 2 ≤ 10) 
  (h3: (10 - 2) / 10 * (10 - 2 - 1) / (10 - 1) * (10 - 2 - 2) / (10 - 2) = 0.4666666666666667) :
  n = 3 :=
sorry

end pencils_purchased_l155_155003


namespace min_value_polynomial_expression_at_k_eq_1_is_0_l155_155792

-- Definition of the polynomial expression
def polynomial_expression (k x y : ℝ) : ℝ :=
  3 * x^2 - 4 * k * x * y + (2 * k^2 + 1) * y^2 - 6 * x - 2 * y + 4

-- Proof statement
theorem min_value_polynomial_expression_at_k_eq_1_is_0 :
  (∀ x y : ℝ, polynomial_expression 1 x y ≥ 0) ∧ (∃ x y : ℝ, polynomial_expression 1 x y = 0) :=
by
  -- Expected proof here. For now, we indicate sorry to skip the proof.
  sorry

end min_value_polynomial_expression_at_k_eq_1_is_0_l155_155792


namespace groupC_is_all_polyhedra_l155_155590

inductive GeometricBody
| TriangularPrism : GeometricBody
| QuadrangularPyramid : GeometricBody
| Sphere : GeometricBody
| Cone : GeometricBody
| Cube : GeometricBody
| TruncatedCone : GeometricBody
| HexagonalPyramid : GeometricBody
| Hemisphere : GeometricBody

def isPolyhedron : GeometricBody → Prop
| GeometricBody.TriangularPrism => true
| GeometricBody.QuadrangularPyramid => true
| GeometricBody.Sphere => false
| GeometricBody.Cone => false
| GeometricBody.Cube => true
| GeometricBody.TruncatedCone => false
| GeometricBody.HexagonalPyramid => true
| GeometricBody.Hemisphere => false

def groupA := [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid, GeometricBody.Sphere, GeometricBody.Cone]
def groupB := [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid, GeometricBody.Cube, GeometricBody.TruncatedCone]
def groupC := [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid, GeometricBody.Cube, GeometricBody.HexagonalPyramid]
def groupD := [GeometricBody.Cone, GeometricBody.TruncatedCone, GeometricBody.Sphere, GeometricBody.Hemisphere]

def allPolyhedra (group : List GeometricBody) : Prop :=
  ∀ b, b ∈ group → isPolyhedron b

theorem groupC_is_all_polyhedra : 
  allPolyhedra groupC ∧
  ¬ allPolyhedra groupA ∧
  ¬ allPolyhedra groupB ∧
  ¬ allPolyhedra groupD :=
by
  sorry

end groupC_is_all_polyhedra_l155_155590


namespace find_sale_month_4_l155_155247

-- Definitions based on the given conditions
def avg_sale_per_month : ℕ := 6500
def num_months : ℕ := 6
def sale_month_1 : ℕ := 6435
def sale_month_2 : ℕ := 6927
def sale_month_3 : ℕ := 6855
def sale_month_5 : ℕ := 6562
def sale_month_6 : ℕ := 4991

theorem find_sale_month_4 : 
  (avg_sale_per_month * num_months) - (sale_month_1 + sale_month_2 + sale_month_3 + sale_month_5 + sale_month_6) = 7230 :=
by
  -- The proof will be provided below
  sorry

end find_sale_month_4_l155_155247


namespace alexander_has_more_pencils_l155_155505

-- Definitions based on conditions
def asaf_age := 50
def total_age := 140
def total_pencils := 220

-- Auxiliary definitions based on conditions
def alexander_age := total_age - asaf_age
def age_difference := alexander_age - asaf_age
def asaf_pencils := 2 * age_difference
def alexander_pencils := total_pencils - asaf_pencils

-- Statement to prove
theorem alexander_has_more_pencils :
  (alexander_pencils - asaf_pencils) = 60 := sorry

end alexander_has_more_pencils_l155_155505


namespace find_r_l155_155078

theorem find_r (r : ℝ) (cone1_radius cone2_radius cone3_radius : ℝ) (sphere_radius : ℝ)
  (cone_height_eq : cone1_radius = 2 * r ∧ cone2_radius = 3 * r ∧ cone3_radius = 10 * r)
  (sphere_touch : sphere_radius = 2)
  (center_eq_dist : ∀ {P Q : ℝ}, dist P Q = 2 → dist Q r = 2) :
  r = 1 := 
sorry

end find_r_l155_155078


namespace circle_radius_zero_l155_155556

theorem circle_radius_zero (x y : ℝ) : 2*x^2 - 8*x + 2*y^2 + 4*y + 10 = 0 → (x - 2)^2 + (y + 1)^2 = 0 :=
by
  intro h
  sorry

end circle_radius_zero_l155_155556


namespace line_intersects_parabola_at_one_point_l155_155039
   
   theorem line_intersects_parabola_at_one_point (k : ℝ) :
     (∃ y : ℝ, (x = 3 * y^2 - 7 * y + 2 ∧ x = k) → x = k) ↔ k = (-25 / 12) :=
   by
     -- your proof goes here
     sorry
   
end line_intersects_parabola_at_one_point_l155_155039


namespace find_a_l155_155841

variable (a : ℝ)

def average_condition (a : ℝ) : Prop :=
  ((2 * a + 16) + (3 * a - 8)) / 2 = 74

theorem find_a (h: average_condition a) : a = 28 :=
  sorry

end find_a_l155_155841


namespace polynomial_is_quadratic_l155_155268

theorem polynomial_is_quadratic (m : ℤ) (h : (m - 2 ≠ 0) ∧ (|m| = 2)) : m = -2 :=
by sorry

end polynomial_is_quadratic_l155_155268


namespace volume_related_to_area_l155_155645

theorem volume_related_to_area (x y z : ℝ) 
  (bottom_area_eq : 3 * x * y = 3 * x * y)
  (front_area_eq : 2 * y * z = 2 * y * z)
  (side_area_eq : 3 * x * z = 3 * x * z) :
  (3 * x * y) * (2 * y * z) * (3 * x * z) = 18 * (x * y * z) ^ 2 := 
by sorry

end volume_related_to_area_l155_155645


namespace largest_multiple_of_7_less_than_neg_30_l155_155909

theorem largest_multiple_of_7_less_than_neg_30 (m : ℤ) (h1 : m % 7 = 0) (h2 : m < -30) : m = -35 :=
sorry

end largest_multiple_of_7_less_than_neg_30_l155_155909


namespace solution_of_inequalities_l155_155585

theorem solution_of_inequalities (x : ℝ) :
  (2 * x / 5 ≤ 3 + x) ∧ (3 + x < -3 * (1 + x)) ↔ (-5 ≤ x ∧ x < -3 / 2) := by
  sorry

end solution_of_inequalities_l155_155585


namespace circle_represents_real_l155_155400

theorem circle_represents_real
  (a : ℝ)
  (h : ∀ x y : ℝ, x^2 + y^2 + 2*y + 2*a - 1 = 0 → ∃ r : ℝ, r > 0) : 
  a < 1 := 
sorry

end circle_represents_real_l155_155400


namespace minimum_n_for_factorable_polynomial_l155_155717

theorem minimum_n_for_factorable_polynomial :
  ∃ n : ℤ, (∀ A B : ℤ, 5 * A = 48 → 5 * B + A = n) ∧
  (∀ k : ℤ, (∀ A B : ℤ, 5 * A * B = 48 → 5 * B + A = k) → n ≤ k) :=
by
  sorry

end minimum_n_for_factorable_polynomial_l155_155717


namespace A_odot_B_correct_l155_155549

open Set

def A : Set ℝ := { x | x ≥ 1 }
def B : Set ℝ := { x | x < 0 ∨ x > 2 }
def A_union_B : Set ℝ := A ∪ B
def A_inter_B : Set ℝ := A ∩ B
def A_odot_B : Set ℝ := { x | x ∈ A_union_B ∧ x ∉ A_inter_B }

theorem A_odot_B_correct : A_odot_B = (Iio 0) ∪ Icc 1 2 :=
by
  sorry

end A_odot_B_correct_l155_155549


namespace determine_GH_l155_155981

-- Define a structure for a Tetrahedron with edge lengths as given conditions
structure Tetrahedron :=
  (EF FG EH FH EG GH : ℕ)

-- Instantiate the Tetrahedron with the given edge lengths
def tetrahedron_EFGH := Tetrahedron.mk 42 14 37 19 28 14

-- State the theorem
theorem determine_GH (t : Tetrahedron) (hEF : t.EF = 42) :
  t.GH = 14 :=
sorry

end determine_GH_l155_155981


namespace arithmetic_sequence_nth_term_l155_155882

theorem arithmetic_sequence_nth_term (a b c n : ℕ) (x: ℕ)
  (h1: a = 3*x - 4)
  (h2: b = 6*x - 17)
  (h3: c = 4*x + 5)
  (h4: b - a = c - b)
  (h5: a + (n - 1) * (b - a) = 4021) : 
  n = 502 :=
by 
  sorry

end arithmetic_sequence_nth_term_l155_155882


namespace arithmetic_sequence_geometric_mean_l155_155514

theorem arithmetic_sequence_geometric_mean (d : ℝ) (k : ℕ) (a : ℕ → ℝ)
  (h1 : d ≠ 0)
  (h2 : a 1 = 9 * d)
  (h3 : a (k + 1) = a 1 + k * d)
  (h4 : a (2 * k + 1) = a 1 + (2 * k) * d)
  (h_gm : (a k) ^ 2 = a 1 * a (2 * k)) :
  k = 4 :=
sorry

end arithmetic_sequence_geometric_mean_l155_155514


namespace sunil_investment_l155_155463

noncomputable def total_amount (P : ℝ) : ℝ :=
  let r1 := 0.025  -- 5% per annum compounded semi-annually
  let r2 := 0.03   -- 6% per annum compounded semi-annually
  let A2 := P * (1 + r1) * (1 + r1)
  let A3 := (A2 + 0.5 * P) * (1 + r2)
  let A4 := A3 * (1 + r2)
  A4

theorem sunil_investment (P : ℝ) : total_amount P = 1.645187625 * P :=
by
  sorry

end sunil_investment_l155_155463


namespace solution_range_l155_155083

-- Define the polynomial function and given values at specific points
def polynomial (x : ℝ) (b : ℝ) : ℝ := x^2 - b * x - 5

-- Given conditions as values of the polynomial at specific points
axiom h1 : ∀ b : ℝ, polynomial (-2) b = 5
axiom h2 : ∀ b : ℝ, polynomial (-1) b = -1
axiom h3 : ∀ b : ℝ, polynomial 4 b = -1
axiom h4 : ∀ b : ℝ, polynomial 5 b = 5

-- The range of solutions for the polynomial equation
theorem solution_range (b : ℝ) : 
  (∃ x : ℝ, -2 < x ∧ x < -1 ∧ polynomial x b = 0) ∨ 
  (∃ x : ℝ, 4 < x ∧ x < 5 ∧ polynomial x b = 0) :=
sorry

end solution_range_l155_155083


namespace determine_words_per_page_l155_155340

noncomputable def wordsPerPage (totalPages : ℕ) (wordsPerPage : ℕ) (totalWordsMod : ℕ) : ℕ :=
if totalPages * wordsPerPage % 250 = totalWordsMod ∧ wordsPerPage <= 200 then wordsPerPage else 0

theorem determine_words_per_page :
  wordsPerPage 150 198 137 = 198 :=
by 
  sorry

end determine_words_per_page_l155_155340


namespace overall_percentage_good_fruits_l155_155049

theorem overall_percentage_good_fruits
  (oranges_bought : ℕ)
  (bananas_bought : ℕ)
  (apples_bought : ℕ)
  (pears_bought : ℕ)
  (oranges_rotten_percent : ℝ)
  (bananas_rotten_percent : ℝ)
  (apples_rotten_percent : ℝ)
  (pears_rotten_percent : ℝ)
  (h_oranges : oranges_bought = 600)
  (h_bananas : bananas_bought = 400)
  (h_apples : apples_bought = 800)
  (h_pears : pears_bought = 200)
  (h_oranges_rotten : oranges_rotten_percent = 0.15)
  (h_bananas_rotten : bananas_rotten_percent = 0.03)
  (h_apples_rotten : apples_rotten_percent = 0.12)
  (h_pears_rotten : pears_rotten_percent = 0.25) :
  let total_fruits := oranges_bought + bananas_bought + apples_bought + pears_bought
  let rotten_oranges := oranges_rotten_percent * oranges_bought
  let rotten_bananas := bananas_rotten_percent * bananas_bought
  let rotten_apples := apples_rotten_percent * apples_bought
  let rotten_pears := pears_rotten_percent * pears_bought
  let good_oranges := oranges_bought - rotten_oranges
  let good_bananas := bananas_bought - rotten_bananas
  let good_apples := apples_bought - rotten_apples
  let good_pears := pears_bought - rotten_pears
  let total_good_fruits := good_oranges + good_bananas + good_apples + good_pears
  (total_good_fruits / total_fruits) * 100 = 87.6 :=
by
  sorry

end overall_percentage_good_fruits_l155_155049


namespace trig_inequality_l155_155576

open Real

theorem trig_inequality (a b c : ℝ) (h₁ : a = sin (2 * π / 7))
  (h₂ : b = cos (2 * π / 7)) (h₃ : c = tan (2 * π / 7)) :
  c > a ∧ a > b :=
by 
  sorry

end trig_inequality_l155_155576


namespace point_in_fourth_quadrant_l155_155390

open Complex

theorem point_in_fourth_quadrant (z : ℂ) (h : (3 + 4 * I) * z = 25) : 
  Complex.arg z > -π / 2 ∧ Complex.arg z < 0 := 
by
  sorry

end point_in_fourth_quadrant_l155_155390


namespace icosahedron_edge_probability_l155_155806

theorem icosahedron_edge_probability :
  let vertices := 12
  let total_pairs := vertices * (vertices - 1) / 2
  let edges := 30
  let probability := edges.toFloat / total_pairs.toFloat
  probability = 5 / 11 :=
by
  sorry

end icosahedron_edge_probability_l155_155806


namespace find_integer_solutions_l155_155146

theorem find_integer_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 + 1 / (x: ℝ)) * (1 + 1 / (y: ℝ)) * (1 + 1 / (z: ℝ)) = 2 ↔ (x = 2 ∧ y = 4 ∧ z = 15) ∨ (x = 2 ∧ y = 5 ∧ z = 9) ∨ (x = 2 ∧ y = 6 ∧ z = 7) ∨ (x = 3 ∧ y = 3 ∧ z = 8) ∨ (x = 3 ∧ y = 4 ∧ z = 5) := sorry

end find_integer_solutions_l155_155146


namespace sum_of_first_and_fourth_l155_155822

theorem sum_of_first_and_fourth (x : ℤ) (h : x + (x + 6) = 156) : (x + 2) = 77 :=
by {
  -- This block represents the assumptions and goal as expressed above,
  -- but the proof steps are omitted.
  sorry
}

end sum_of_first_and_fourth_l155_155822


namespace problem_solution_l155_155915

theorem problem_solution
  (x y : ℝ)
  (h : 5 * x^2 - 4 * x * y + y^2 - 2 * x + 1 = 0) :
  (x - y) ^ 2007 = -1 := by
  sorry

end problem_solution_l155_155915


namespace athletes_same_color_probability_l155_155119

theorem athletes_same_color_probability :
  let colors := ["red", "white", "blue"]
  let total_ways := 3 * 3
  let same_color_ways := 3
  total_ways > 0 → 
  (same_color_ways : ℚ) / (total_ways : ℚ) = 1 / 3 :=
by
  sorry

end athletes_same_color_probability_l155_155119


namespace find_a_l155_155886

theorem find_a (x y a : ℝ) (h1 : x + 2 * y = 2) (h2 : 2 * x + y = a) (h3 : x + y = 5) : a = 13 := by
  sorry

end find_a_l155_155886


namespace problem_l155_155522

def f (u : ℝ) : ℝ := u^2 - 2

theorem problem : f 3 = 7 := 
by sorry

end problem_l155_155522


namespace correct_simplification_l155_155331

theorem correct_simplification (x y : ℝ) (hy : y ≠ 0):
  3 * x^4 * y / (x^2 * y) = 3 * x^2 :=
by
  sorry

end correct_simplification_l155_155331


namespace sequence_ratio_l155_155038

variable (a : ℕ → ℝ) -- Define the sequence a_n
variable (q : ℝ) (h_q : q > 0) -- q is the common ratio and it is positive

-- Define the conditions
axiom geom_seq_pos : ∀ n : ℕ, 0 < a n
axiom geom_seq_def : ∀ n : ℕ, a (n + 1) = q * a n
axiom arith_seq_def : 2 * (1 / 2 * a 3) = 3 * a 1 + 2 * a 2

theorem sequence_ratio : (a 11 + a 13) / (a 8 + a 10) = 27 := 
by
  sorry

end sequence_ratio_l155_155038


namespace find_a_tangent_to_curve_l155_155866

theorem find_a_tangent_to_curve (a : ℝ) :
  (∃ (x₀ : ℝ), y = x - 1 ∧ y = e^(x + a) ∧ (e^(x₀ + a) = 1)) → a = -2 :=
by
  sorry

end find_a_tangent_to_curve_l155_155866


namespace pictures_at_museum_l155_155397

variable (M : ℕ)

-- Definitions from conditions
def pictures_at_zoo : ℕ := 50
def pictures_deleted : ℕ := 38
def pictures_left : ℕ := 20

-- Theorem to prove the total number of pictures taken including the museum pictures
theorem pictures_at_museum :
  pictures_at_zoo + M - pictures_deleted = pictures_left → M = 8 :=
by
  sorry

end pictures_at_museum_l155_155397


namespace lines_positional_relationship_l155_155651

-- Defining basic geometric entities and their properties
structure Line :=
  (a b : ℝ)
  (point_on_line : ∃ x, a * x + b = 0)

-- Defining skew lines (two lines that do not intersect and are not parallel)
def skew_lines (l1 l2 : Line) : Prop :=
  ¬(∀ x, l1.a * x + l1.b = l2.a * x + l2.b) ∧ ¬(l1.a = l2.a)

-- Defining intersecting lines
def intersect (l1 l2 : Line) : Prop :=
  ∃ x, l1.a * x + l1.b = l2.a * x + l2.b

-- Main theorem to prove
theorem lines_positional_relationship (l1 l2 k m : Line) 
  (hl1: intersect l1 k) (hl2: intersect l2 k) (hk: skew_lines l1 m) (hm: skew_lines l2 m) :
  (intersect l1 l2) ∨ (skew_lines l1 l2) :=
sorry

end lines_positional_relationship_l155_155651


namespace fraction_value_l155_155155

theorem fraction_value (p q r s : ℝ) (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -4 / 7 :=
by
  sorry

end fraction_value_l155_155155


namespace polygon_sides_l155_155326

theorem polygon_sides (n : ℕ) 
  (H : (n * (n - 3)) / 2 = 3 * n) : n = 9 := 
sorry

end polygon_sides_l155_155326


namespace net_wealth_after_transactions_l155_155515

-- Define initial values and transactions
def initial_cash_A : ℕ := 15000
def initial_cash_B : ℕ := 20000
def initial_house_value : ℕ := 15000
def first_transaction_price : ℕ := 20000
def depreciation_rate : ℝ := 0.15

-- Post-depreciation house value
def depreciated_house_value : ℝ := initial_house_value * (1 - depreciation_rate)

-- Final amounts after transactions
def final_cash_A : ℝ := (initial_cash_A + first_transaction_price) - depreciated_house_value
def final_cash_B : ℝ := depreciated_house_value

-- Net changes in wealth
def net_change_wealth_A : ℝ := final_cash_A + depreciated_house_value - (initial_cash_A + initial_house_value)
def net_change_wealth_B : ℝ := final_cash_B - initial_cash_B

-- Our proof goal
theorem net_wealth_after_transactions :
  net_change_wealth_A = 5000 ∧ net_change_wealth_B = -7250 :=
by
  sorry

end net_wealth_after_transactions_l155_155515


namespace geometric_sequence_properties_l155_155292

theorem geometric_sequence_properties (a : ℕ → ℝ) (n : ℕ) (q : ℝ) 
  (h_geom : ∀ (m k : ℕ), a (m + k) = a m * q ^ k) 
  (h_sum : a 1 + a n = 66) 
  (h_prod : a 3 * a (n - 2) = 128) 
  (h_s_n : (a 1 * (1 - q ^ n)) / (1 - q) = 126) : 
  n = 6 ∧ (q = 2 ∨ q = 1/2) :=
sorry

end geometric_sequence_properties_l155_155292


namespace product_of_roots_of_quadratics_l155_155871

noncomputable def product_of_roots : ℝ :=
  let r1 := 2021 / 2020
  let r2 := 2020 / 2019
  let r3 := 2019
  r1 * r2 * r3

theorem product_of_roots_of_quadratics (b : ℝ) 
  (h1 : ∃ x1 x2 : ℝ, 2020 * x1 * x1 + b * x1 + 2021 = 0 ∧ 2020 * x2 * x2 + b * x2 + 2021 = 0) 
  (h2 : ∃ y1 y2 : ℝ, 2019 * y1 * y1 + b * y1 + 2020 = 0 ∧ 2019 * y2 * y2 + b * y2 + 2020 = 0) 
  (h3 : ∃ z1 z2 : ℝ, z1 * z1 + b * z1 + 2019 = 0 ∧ z1 * z1 + b * z2 + 2019 = 0) :
  product_of_roots = 2021 :=
by
  sorry

end product_of_roots_of_quadratics_l155_155871


namespace binary_to_octal_101110_l155_155828

theorem binary_to_octal_101110 : 
  ∀ (binary_to_octal : ℕ → ℕ), 
  binary_to_octal 0b101110 = 0o56 :=
by
  sorry

end binary_to_octal_101110_l155_155828


namespace binary_subtraction_result_l155_155757

theorem binary_subtraction_result :
  let x := 0b1101101 -- binary notation for 109
  let y := 0b11101   -- binary notation for 29
  let z := 0b101010  -- binary notation for 42
  let product := x * y
  let result := product - z
  result = 0b10000010001 := -- binary notation for 3119
by
  sorry

end binary_subtraction_result_l155_155757


namespace no_solutions_l155_155490

theorem no_solutions {x y : ℤ} :
  (x ≠ 1) → (y ≠ 1) →
  ((x^7 - 1) / (x - 1) = y^5 - 1) →
  false :=
by sorry

end no_solutions_l155_155490


namespace inning_count_l155_155612

-- Definition of the conditions
variables {n T H L : ℕ}
variables (avg_total : ℕ) (avg_excl : ℕ) (diff : ℕ) (high_score : ℕ)

-- Define the conditions
def conditions :=
  avg_total = 62 ∧
  high_score = 225 ∧
  diff = 150 ∧
  avg_excl = 58

-- Proving the main theorem
theorem inning_count (avg_total := 62) (high_score := 225) (diff := 150) (avg_excl := 58) :
   conditions avg_total avg_excl diff high_score →
   n = 104 :=
sorry

end inning_count_l155_155612


namespace pipe_length_l155_155724

theorem pipe_length (L_short : ℕ) (hL_short : L_short = 59) : 
    L_short + 2 * L_short = 177 := by
  sorry

end pipe_length_l155_155724


namespace wuxi_GDP_scientific_notation_l155_155346

theorem wuxi_GDP_scientific_notation :
  14800 = 1.48 * 10^4 :=
sorry

end wuxi_GDP_scientific_notation_l155_155346


namespace factorize_expression_l155_155066

variable {a b : ℕ}

theorem factorize_expression (a b : ℕ) : 9 * a - 6 * b = 3 * (3 * a - 2 * b) :=
by
  sorry

end factorize_expression_l155_155066


namespace father_son_age_relationship_l155_155438

theorem father_son_age_relationship 
    (F S X : ℕ) 
    (h1 : F = 27) 
    (h2 : F = 3 * S + 3) 
    : X = 11 ∧ F + X > 2 * (S + X) :=
by
  sorry

end father_son_age_relationship_l155_155438


namespace problem_statement_l155_155930

theorem problem_statement :
  ∃ p q r : ℤ,
    (∀ x : ℝ, (x^2 + 19*x + 88 = (x + p) * (x + q)) ∧ (x^2 - 23*x + 132 = (x - q) * (x - r))) →
      p + q + r = 31 :=
sorry

end problem_statement_l155_155930


namespace green_balls_count_l155_155818

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def yellow_balls : ℕ := 2
def red_balls : ℕ := 15
def purple_balls : ℕ := 3
def probability_neither_red_nor_purple : ℝ := 0.7

theorem green_balls_count (G : ℕ) :
  (white_balls + G + yellow_balls) / total_balls = probability_neither_red_nor_purple →
  G = 18 := 
by
  sorry

end green_balls_count_l155_155818


namespace area_unpainted_region_l155_155428

theorem area_unpainted_region
  (width_board_1 : ℝ)
  (width_board_2 : ℝ)
  (cross_angle_degrees : ℝ)
  (unpainted_area : ℝ)
  (h1 : width_board_1 = 5)
  (h2 : width_board_2 = 7)
  (h3 : cross_angle_degrees = 45)
  (h4 : unpainted_area = (49 * Real.sqrt 2) / 2) : 
  unpainted_area = (width_board_2 * ((width_board_1 * Real.sqrt 2) / 2)) / 2 :=
sorry

end area_unpainted_region_l155_155428


namespace total_lives_correct_l155_155954

-- Define the initial number of friends
def initial_friends : ℕ := 16

-- Define the number of lives each player has
def lives_per_player : ℕ := 10

-- Define the number of additional players that joined
def additional_players : ℕ := 4

-- Define the initial total number of lives
def initial_lives : ℕ := initial_friends * lives_per_player

-- Define the additional lives from the new players
def additional_lives : ℕ := additional_players * lives_per_player

-- Define the final total number of lives
def total_lives : ℕ := initial_lives + additional_lives

-- The proof goal
theorem total_lives_correct : total_lives = 200 :=
by
  -- This is where the proof would be written, but it is omitted.
  sorry

end total_lives_correct_l155_155954


namespace cakes_served_today_l155_155657

def lunch_cakes := 6
def dinner_cakes := 9
def total_cakes := lunch_cakes + dinner_cakes

theorem cakes_served_today : total_cakes = 15 := by
  sorry

end cakes_served_today_l155_155657


namespace fraction_subtraction_equals_one_l155_155553

theorem fraction_subtraction_equals_one (x : ℝ) (h : x ≠ 1) : (x / (x - 1)) - (1 / (x - 1)) = 1 := 
by sorry

end fraction_subtraction_equals_one_l155_155553


namespace solve_equation_solutions_count_l155_155086

open Real

theorem solve_equation_solutions_count :
  (∃ (x_plural : List ℝ), (∀ x ∈ x_plural, 2 * sqrt 2 * (sin (π * x / 4)) ^ 3 = cos (π / 4 * (1 - x)) ∧ 0 ≤ x ∧ x ≤ 2020) ∧ x_plural.length = 505) :=
sorry

end solve_equation_solutions_count_l155_155086


namespace inequality_abc_l155_155166

theorem inequality_abc
  (a b c : ℝ)
  (ha : 0 ≤ a) (ha_le : a ≤ 1)
  (hb : 0 ≤ b) (hb_le : b ≤ 1)
  (hc : 0 ≤ c) (hc_le : c ≤ 1) :
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
sorry

end inequality_abc_l155_155166


namespace theta_digit_l155_155090

theorem theta_digit (Θ : ℕ) (h : Θ ≠ 0) (h1 : 252 / Θ = 10 * 4 + Θ + Θ) : Θ = 5 :=
  sorry

end theta_digit_l155_155090


namespace find_ordered_pair_l155_155387

theorem find_ordered_pair {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : ∀ x, x^2 + a * x + b = 0 ↔ x = -2 * a ∨ x = b)
  (h2 : b = -2 * -2 * a) : (a, b) = (-1/2, -1/2) :=
by
  sorry

end find_ordered_pair_l155_155387


namespace negation_of_p_l155_155312

variable (x : ℝ)

def p : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 := by
  sorry

end negation_of_p_l155_155312


namespace arithmetic_sequence_problem_l155_155550

theorem arithmetic_sequence_problem 
    (a : ℕ → ℝ)  -- Define the arithmetic sequence as a function from natural numbers to reals
    (a1 : ℝ)  -- Represent a₁ as a1
    (a8 : ℝ)  -- Represent a₈ as a8
    (a9 : ℝ)  -- Represent a₉ as a9
    (a10 : ℝ)  -- Represent a₁₀ as a10
    (a15 : ℝ)  -- Represent a₁₅ as a15
    (h1 : a 1 = a1)  -- Hypothesis that a(1) is represented by a1
    (h8 : a 8 = a8)  -- Hypothesis that a(8) is represented by a8
    (h9 : a 9 = a9)  -- Hypothesis that a(9) is represented by a9
    (h10 : a 10 = a10)  -- Hypothesis that a(10) is represented by a10
    (h15 : a 15 = a15)  -- Hypothesis that a(15) is represented by a15
    (h_condition : a1 + 2 * a8 + a15 = 96)  -- Condition of the problem
    : 2 * a9 - a10 = 24 := 
sorry

end arithmetic_sequence_problem_l155_155550


namespace feet_perpendiculars_concyclic_l155_155441

variables {S A B C D O M N P Q : Type} 

-- Given conditions
variables (is_convex_quadrilateral : convex_quadrilateral A B C D)
variables (diagonals_perpendicular : ∀ (AC BD : Line), perpendicular AC BD)
variables (foot_perpendicular : ∀ (O : Point), intersection_point O = foot (perpendicular_from S (base_quadrilateral A B C D)))

-- Define the proof statement
theorem feet_perpendiculars_concyclic
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : diagonals_perpendicular AC BD)
  (h3 : foot_perpendicular O) :
  concyclic (feet_perpendicular_pts O (face S A B)) (feet_perpendicular_pts O (face S B C)) 
            (feet_perpendicular_pts O (face S C D)) (feet_perpendicular_pts O (face S D A)) := sorry

end feet_perpendiculars_concyclic_l155_155441


namespace ratio_of_distances_l155_155780

/-- 
  Given two points A and B moving along intersecting lines with constant,
  but different velocities v_A and v_B respectively, prove that there exists a 
  point P such that at any moment in time, the ratio of distances AP to BP equals 
  the ratio of their velocities.
-/
theorem ratio_of_distances (A B : ℝ → ℝ × ℝ) (v_A v_B : ℝ)
  (intersecting_lines : ∃ t, A t = B t)
  (diff_velocities : v_A ≠ v_B) :
  ∃ P : ℝ × ℝ, ∀ t, (dist P (A t) / dist P (B t)) = v_A / v_B := 
sorry

end ratio_of_distances_l155_155780


namespace total_dolls_l155_155677

-- Definitions based on the given conditions
def grandmother_dolls : Nat := 50
def sister_dolls : Nat := grandmother_dolls + 2
def rene_dolls : Nat := 3 * sister_dolls

-- Statement we want to prove
theorem total_dolls : grandmother_dolls + sister_dolls + rene_dolls = 258 := by
  sorry

end total_dolls_l155_155677


namespace line_parallel_to_plane_l155_155816

-- Defining conditions
def vector_a : ℝ × ℝ × ℝ := (1, -1, 3)
def vector_n : ℝ × ℝ × ℝ := (0, 3, 1)

-- Lean theorem statement
theorem line_parallel_to_plane : 
  let ⟨a1, a2, a3⟩ := vector_a;
  let ⟨n1, n2, n3⟩ := vector_n;
  a1 * n1 + a2 * n2 + a3 * n3 = 0 :=
by 
  -- Proof omitted
  sorry

end line_parallel_to_plane_l155_155816


namespace tangent_line_slope_l155_155250

theorem tangent_line_slope (x₀ y₀ k : ℝ)
    (h_tangent_point : y₀ = x₀ + Real.exp (-x₀))
    (h_tangent_line : y₀ = k * x₀) :
    k = 1 - Real.exp 1 := 
sorry

end tangent_line_slope_l155_155250


namespace circumference_of_smaller_circle_l155_155252

theorem circumference_of_smaller_circle (r R : ℝ)
  (h1 : 4 * R^2 = 784) 
  (h2 : R = (7/3) * r) :
  2 * Real.pi * r = 12 * Real.pi := 
by {
  sorry
}

end circumference_of_smaller_circle_l155_155252


namespace three_digit_minuends_count_l155_155324

theorem three_digit_minuends_count :
  ∀ a b c : ℕ, a - c = 4 ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
  (∃ n : ℕ, n = 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c - 396 = 100 * c + 10 * b + a) →
  ∃ count : ℕ, count = 50 :=
by
  sorry

end three_digit_minuends_count_l155_155324


namespace total_votes_polled_l155_155772

theorem total_votes_polled (V: ℝ) (h: 0 < V) (h1: 0.70 * V - 0.30 * V = 320) : V = 800 :=
sorry

end total_votes_polled_l155_155772


namespace third_side_length_not_4_l155_155664

theorem third_side_length_not_4 (x : ℕ) : 
  (5 < x + 9) ∧ (9 < x + 5) ∧ (x + 5 < 14) → ¬ (x = 4) := 
by
  intros h
  sorry

end third_side_length_not_4_l155_155664


namespace projection_v_w_l155_155947

noncomputable def vector_v : ℝ × ℝ := (3, 4)
noncomputable def vector_w : ℝ × ℝ := (2, -1)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := dot_product u v / dot_product v v
  (scalar * v.1, scalar * v.2)

theorem projection_v_w :
  proj vector_v vector_w = (4/5, -2/5) :=
sorry

end projection_v_w_l155_155947


namespace total_peaches_l155_155974

variable {n m : ℕ}

-- conditions
def equal_subgroups (n : ℕ) := (n % 3 = 0)

def condition_1 (n m : ℕ) := (m - 27) % n = 0 ∧ (m - 27) / n = 5

def condition_2 (n m : ℕ) : Prop := 
  ∃ x : ℕ, 0 < x ∧ x < 7 ∧ (m - x) % n = 0 ∧ ((m - x) / n = 7) 

-- theorem to be proved
theorem total_peaches (n m : ℕ) (h1 : equal_subgroups n) (h2 : condition_1 n m) (h3 : condition_2 n m) : m = 102 := 
sorry

end total_peaches_l155_155974


namespace total_amount_l155_155953

theorem total_amount (A N J : ℕ) (h1 : A = N - 5) (h2 : J = 4 * N) (h3 : J = 48) : A + N + J = 67 :=
by
  -- Proof will be constructed here
  sorry

end total_amount_l155_155953


namespace product_less_by_nine_times_l155_155525

theorem product_less_by_nine_times (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : y < 10) : 
  (x * y) * 10 - x * y = 9 * (x * y) := 
by
  sorry

end product_less_by_nine_times_l155_155525


namespace diagonals_in_25_sided_polygon_l155_155740

-- Define a function to calculate the number of specific diagonals in a convex polygon with n sides
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 5) / 2

-- Theorem stating the number of diagonals for a convex polygon with 25 sides with the given condition
theorem diagonals_in_25_sided_polygon : number_of_diagonals 25 = 250 := 
sorry

end diagonals_in_25_sided_polygon_l155_155740


namespace exponent_on_right_side_l155_155983

theorem exponent_on_right_side (n : ℕ) (k : ℕ) (h : n = 25) :
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^k → k = 26 :=
by
  sorry

end exponent_on_right_side_l155_155983


namespace probability_not_equal_genders_l155_155194

noncomputable def probability_more_grandsons_or_more_granddaughters : ℚ :=
  let total_ways := 2 ^ 12
  let equal_distribution_ways := (Nat.choose 12 6)
  let probability_equal := (equal_distribution_ways : ℚ) / (total_ways : ℚ)
  1 - probability_equal

theorem probability_not_equal_genders (n : ℕ) (p : ℚ) (hp : p = 1 / 2) (hn : n = 12) :
  probability_more_grandsons_or_more_granddaughters = 793 / 1024 :=
by
  sorry

end probability_not_equal_genders_l155_155194


namespace total_number_of_workers_l155_155939

theorem total_number_of_workers (W N : ℕ) 
    (avg_all : ℝ) 
    (avg_technicians : ℝ) 
    (avg_non_technicians : ℝ)
    (h1 : avg_all = 8000)
    (h2 : avg_technicians = 20000)
    (h3 : avg_non_technicians = 6000)
    (h4 : 7 * avg_technicians + N * avg_non_technicians = (7 + N) * avg_all) :
  W = 49 := by
  sorry

end total_number_of_workers_l155_155939


namespace existence_of_subset_A_l155_155415

def M : Set ℚ := {x : ℚ | 0 < x ∧ x < 1}

theorem existence_of_subset_A :
  ∃ A ⊆ M, ∀ m ∈ M, ∃! (S : Finset ℚ), (∀ a ∈ S, a ∈ A) ∧ (S.sum id = m) :=
sorry

end existence_of_subset_A_l155_155415


namespace largest_positive_integer_n_l155_155661

 

theorem largest_positive_integer_n (n : ℕ) :
  (∀ p : ℕ, Nat.Prime p ∧ 2 < p ∧ p < n → Nat.Prime (n - p)) →
  ∀ m : ℕ, (∀ q : ℕ, Nat.Prime q ∧ 2 < q ∧ q < m → Nat.Prime (m - q)) → n ≥ m → n = 10 :=
by
  sorry

end largest_positive_integer_n_l155_155661


namespace total_coins_l155_155995

theorem total_coins (q1 q2 q3 q4 : Nat) (d1 d2 d3 : Nat) (n1 n2 : Nat) (p1 p2 p3 p4 p5 : Nat) :
  q1 = 8 → q2 = 6 → q3 = 7 → q4 = 5 →
  d1 = 7 → d2 = 5 → d3 = 9 →
  n1 = 4 → n2 = 6 →
  p1 = 10 → p2 = 3 → p3 = 8 → p4 = 2 → p5 = 13 →
  q1 + q2 + q3 + q4 + d1 + d2 + d3 + n1 + n2 + p1 + p2 + p3 + p4 + p5 = 93 :=
by
  intros
  sorry

end total_coins_l155_155995


namespace evaluate_triangle_l155_155204

def triangle_op (a b : Int) : Int :=
  a * b - a - b + 1

theorem evaluate_triangle :
  triangle_op (-3) 4 = -12 :=
by
  sorry

end evaluate_triangle_l155_155204


namespace percentage_volume_taken_by_cubes_l155_155043

def volume_of_box (l w h : ℝ) : ℝ := l * w * h

def volume_of_cube (side : ℝ) : ℝ := side ^ 3

noncomputable def total_cubes_fit (l w h side : ℝ) : ℝ := 
  (l / side) * (w / side) * (h / side)

theorem percentage_volume_taken_by_cubes (l w h side : ℝ) (hl : l = 12) (hw : w = 6) (hh : h = 9) (hside : side = 3) :
  volume_of_box l w h ≠ 0 → 
  (total_cubes_fit l w h side * volume_of_cube side / volume_of_box l w h) * 100 = 100 :=
by
  intros
  rw [hl, hw, hh, hside]
  simp only [volume_of_box, volume_of_cube, total_cubes_fit]
  sorry

end percentage_volume_taken_by_cubes_l155_155043


namespace solve_quadratic_equations_l155_155655

noncomputable def E1 := ∀ x : ℝ, x^2 - 14 * x + 21 = 0 ↔ (x = 7 + 2 * Real.sqrt 7 ∨ x = 7 - 2 * Real.sqrt 7)

noncomputable def E2 := ∀ x : ℝ, x^2 - 3 * x + 2 = 0 ↔ (x = 1 ∨ x = 2)

theorem solve_quadratic_equations :
  (E1) ∧ (E2) :=
by
  sorry

end solve_quadratic_equations_l155_155655


namespace problem1_problem2_l155_155045

-- Problem (1) proof statement
theorem problem1 (a : ℝ) (h : a ≠ 0) : 
  3 * a^2 * a^3 + a^7 / a^2 = 4 * a^5 :=
by
  sorry

-- Problem (2) proof statement
theorem problem2 (x : ℝ) : 
  (x - 1)^2 - x * (x + 1) + (-2023)^0 = -3 * x + 2 :=
by
  sorry

end problem1_problem2_l155_155045


namespace yellow_marbles_count_l155_155796

theorem yellow_marbles_count 
  (total_marbles red_marbles blue_marbles : ℕ) 
  (h_total : total_marbles = 85) 
  (h_red : red_marbles = 14) 
  (h_blue : blue_marbles = 3 * red_marbles) :
  (total_marbles - (red_marbles + blue_marbles)) = 29 :=
by
  sorry

end yellow_marbles_count_l155_155796


namespace repeating_decimals_difference_l155_155980

theorem repeating_decimals_difference :
  let x := 234 / 999
  let y := 567 / 999
  let z := 891 / 999
  x - y - z = -408 / 333 :=
by
  sorry

end repeating_decimals_difference_l155_155980


namespace designer_suit_size_l155_155672

theorem designer_suit_size : ∀ (waist_in_inches : ℕ) (comfort_in_inches : ℕ) 
  (inches_per_foot : ℕ) (cm_per_foot : ℝ), 
  waist_in_inches = 34 →
  comfort_in_inches = 2 →
  inches_per_foot = 12 →
  cm_per_foot = 30.48 →
  (((waist_in_inches + comfort_in_inches) / inches_per_foot : ℝ) * cm_per_foot) = 91.4 :=
by
  intros waist_in_inches comfort_in_inches inches_per_foot cm_per_foot
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_cast
  norm_num
  sorry

end designer_suit_size_l155_155672


namespace cos_product_inequality_l155_155960

theorem cos_product_inequality : (1 / 8 : ℝ) < (Real.cos (20 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) * Real.cos (70 * Real.pi / 180)) ∧
    (Real.cos (20 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) * Real.cos (70 * Real.pi / 180)) < (1 / 4 : ℝ) :=
by
  sorry

end cos_product_inequality_l155_155960


namespace limonia_largest_none_providable_amount_l155_155638

def is_achievable (n : ℕ) (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), x = a * (6 * n + 1) + b * (6 * n + 4) + c * (6 * n + 7) + d * (6 * n + 10)

theorem limonia_largest_none_providable_amount (n : ℕ) : 
  ∃ s, ¬ is_achievable n s ∧ (∀ t, t > s → is_achievable n t) ∧ s = 12 * n^2 + 14 * n - 1 :=
by
  sorry

end limonia_largest_none_providable_amount_l155_155638


namespace amount_paid_Y_l155_155344

theorem amount_paid_Y (X Y : ℝ) (h1 : X + Y = 330) (h2 : X = 1.2 * Y) : Y = 150 := 
by
  sorry

end amount_paid_Y_l155_155344


namespace variance_of_remaining_scores_l155_155942

def scores : List ℕ := [91, 89, 91, 96, 94, 95, 94]

def remaining_scores : List ℕ := [91, 91, 94, 95, 94]

def mean (l : List ℕ) : ℚ := l.sum / l.length

def variance (l : List ℕ) : ℚ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length

theorem variance_of_remaining_scores :
  variance remaining_scores = 2.8 := by
  sorry

end variance_of_remaining_scores_l155_155942


namespace units_digit_of_pow_sum_is_correct_l155_155984

theorem units_digit_of_pow_sum_is_correct : 
  (24^3 + 42^3) % 10 = 2 := by
  -- Start the proof block
  sorry

end units_digit_of_pow_sum_is_correct_l155_155984


namespace a_greater_than_1_and_b_less_than_1_l155_155892

theorem a_greater_than_1_and_b_less_than_1
  (a b c : ℝ) 
  (h1 : a ≥ b)
  (h2 : b ≥ c)
  (h3 : c > 0)
  (h4 : a * b * c = 1)
  (h5 : a + b + c > 1/a + 1/b + 1/c) : a > 1 ∧ b < 1 :=
by
  sorry

end a_greater_than_1_and_b_less_than_1_l155_155892


namespace clothes_in_total_l155_155652

-- Define the conditions as constants since they are fixed values
def piecesInOneLoad : Nat := 17
def numberOfSmallLoads : Nat := 5
def piecesPerSmallLoad : Nat := 6

-- Noncomputable for definition involving calculation
noncomputable def totalClothes : Nat :=
  piecesInOneLoad + (numberOfSmallLoads * piecesPerSmallLoad)

-- The theorem to prove Luke had 47 pieces of clothing in total
theorem clothes_in_total : totalClothes = 47 := by
  sorry

end clothes_in_total_l155_155652


namespace exists_integers_a_b_part_a_l155_155354

theorem exists_integers_a_b_part_a : 
  ∃ a b : ℤ, (∀ x : ℝ, x^2 + a * x + b ≠ 0) ∧ (∃ x : ℝ, ⌊x^2⌋ + a * x + (b : ℝ) = 0) := 
sorry

end exists_integers_a_b_part_a_l155_155354


namespace triangle_area_eq_l155_155484

theorem triangle_area_eq :
  let y_intercept1 := 3
  let y_intercept2 := 15 / 2
  let base := y_intercept2 - y_intercept1
  let inter_x := 1
  let height := inter_x
  let area := (1 / 2) * base * height
  area = 9 / 4 :=
by
  let y_intercept1 := 3
  let y_intercept2 := 15 / 2
  let base := y_intercept2 - y_intercept1
  let inter_x := 1
  let height := inter_x
  let area := (1 / 2) * base * height
  sorry

end triangle_area_eq_l155_155484


namespace repeating_decimal_product_l155_155343

theorem repeating_decimal_product 
  (x : ℚ) 
  (h1 : x = (0.0126 : ℚ)) 
  (h2 : 9999 * x = 126) 
  (h3 : x = 14 / 1111) : 
  14 * 1111 = 15554 := 
by
  sorry

end repeating_decimal_product_l155_155343


namespace boat_distance_downstream_is_68_l155_155069

variable (boat_speed : ℕ) (stream_speed : ℕ) (time_hours : ℕ)

-- Given conditions
def effective_speed_downstream (boat_speed stream_speed : ℕ) : ℕ := boat_speed + stream_speed
def distance_downstream (speed time : ℕ) : ℕ := speed * time

theorem boat_distance_downstream_is_68 
  (h1 : boat_speed = 13) 
  (h2 : stream_speed = 4) 
  (h3 : time_hours = 4) : 
  distance_downstream (effective_speed_downstream boat_speed stream_speed) time_hours = 68 := 
by 
  sorry

end boat_distance_downstream_is_68_l155_155069


namespace total_oranges_l155_155555

theorem total_oranges (joan_oranges : ℕ) (sara_oranges : ℕ) 
                      (h1 : joan_oranges = 37) 
                      (h2 : sara_oranges = 10) :
  joan_oranges + sara_oranges = 47 := by
  sorry

end total_oranges_l155_155555


namespace time_to_cross_platform_l155_155411

-- Definition of the given conditions
def length_of_train : ℕ := 1500 -- in meters
def time_to_cross_tree : ℕ := 120 -- in seconds
def length_of_platform : ℕ := 500 -- in meters
def speed : ℚ := length_of_train / time_to_cross_tree -- speed in meters per second

-- Definition of the total distance to cross the platform
def total_distance : ℕ := length_of_train + length_of_platform

-- Theorem to prove the time taken to cross the platform
theorem time_to_cross_platform : (total_distance / speed) = 160 :=
by
  -- Placeholder for the proof
  sorry

end time_to_cross_platform_l155_155411


namespace value_of_x_plus_y_l155_155230

theorem value_of_x_plus_y 
  (x y : ℝ)
  (h1 : |x| = 3)
  (h2 : |y| = 2)
  (h3 : x > y) :
  x + y = 5 ∨ x + y = 1 := 
  sorry

end value_of_x_plus_y_l155_155230


namespace greatest_possible_x_exists_greatest_x_l155_155392

theorem greatest_possible_x (x : ℤ) (h1 : 6.1 * (10 : ℝ) ^ x < 620) : x ≤ 2 :=
sorry

theorem exists_greatest_x : ∃ x : ℤ, 6.1 * (10 : ℝ) ^ x < 620 ∧ x = 2 :=
sorry

end greatest_possible_x_exists_greatest_x_l155_155392


namespace cylinder_volume_ratio_l155_155803

noncomputable def volume_of_cylinder (height : ℝ) (circumference : ℝ) : ℝ := 
  let r := circumference / (2 * Real.pi)
  Real.pi * r^2 * height

theorem cylinder_volume_ratio :
  let V1 := volume_of_cylinder 10 6
  let V2 := volume_of_cylinder 6 10
  max V1 V2 / min V1 V2 = 5 / 3 :=
by
  let V1 := volume_of_cylinder 10 6
  let V2 := volume_of_cylinder 6 10
  have hV1 : V1 = 90 / Real.pi := sorry
  have hV2 : V2 = 150 / Real.pi := sorry
  sorry

end cylinder_volume_ratio_l155_155803


namespace correct_weight_of_misread_boy_l155_155775

variable (num_boys : ℕ) (avg_weight_incorrect : ℝ) (misread_weight : ℝ) (avg_weight_correct : ℝ)

theorem correct_weight_of_misread_boy
  (h1 : num_boys = 20)
  (h2 : avg_weight_incorrect = 58.4)
  (h3 : misread_weight = 56)
  (h4 : avg_weight_correct = 58.6) : 
  misread_weight + (num_boys * avg_weight_correct - num_boys * avg_weight_incorrect) / num_boys = 60 := 
by 
  -- skipping proof
  sorry

end correct_weight_of_misread_boy_l155_155775


namespace distance_after_12_seconds_time_to_travel_380_meters_l155_155243

def distance_travelled (t : ℝ) : ℝ := 9 * t + 0.5 * t^2

theorem distance_after_12_seconds : distance_travelled 12 = 180 :=
by 
  sorry

theorem time_to_travel_380_meters : ∃ t : ℝ, distance_travelled t = 380 ∧ t = 20 :=
by 
  sorry

end distance_after_12_seconds_time_to_travel_380_meters_l155_155243


namespace last_two_digits_of_sum_l155_155429

-- Define factorial, and factorials up to 50 specifically for our problem.
def fac : ℕ → ℕ
  | 0     => 1
  | (n+1) => (n+1) * fac n

-- Sum the last two digits of factorials from 1 to 50
def last_two_digits_sum : ℕ :=
  (fac 1 % 100 + fac 2 % 100 + fac 3 % 100 + fac 4 % 100 + fac 5 % 100 + 
   fac 6 % 100 + fac 7 % 100 + fac 8 % 100 + fac 9 % 100) % 100

theorem last_two_digits_of_sum : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_l155_155429


namespace max_area_curves_intersection_l155_155993

open Real

def C₁ (x : ℝ) : ℝ := x^3 - x
def C₂ (x a : ℝ) : ℝ := (x - a)^3 - (x - a)

theorem max_area_curves_intersection (a : ℝ) (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ C₁ x₁ = C₂ x₁ a ∧ C₁ x₂ = C₂ x₂ a) :
  ∃ A_max : ℝ, A_max = 3 / 4 :=
by
  -- TODO: Provide the proof here
  sorry

end max_area_curves_intersection_l155_155993


namespace average_marks_correct_l155_155517

/-- Define the marks scored by Shekar in different subjects -/
def marks_math : ℕ := 76
def marks_science : ℕ := 65
def marks_social_studies : ℕ := 82
def marks_english : ℕ := 67
def marks_biology : ℕ := 55

/-- Define the total marks scored by Shekar -/
def total_marks : ℕ := marks_math + marks_science + marks_social_studies + marks_english + marks_biology

/-- Define the number of subjects -/
def num_subjects : ℕ := 5

/-- Define the average marks scored by Shekar -/
def average_marks : ℕ := total_marks / num_subjects

theorem average_marks_correct : average_marks = 69 := by
  -- We need to show that the average marks is 69
  sorry

end average_marks_correct_l155_155517


namespace incorrect_proposition_l155_155641

theorem incorrect_proposition (p q : Prop) :
  ¬(¬(p ∧ q) → ¬p ∧ ¬q) := sorry

end incorrect_proposition_l155_155641


namespace find_b_l155_155762

theorem find_b (a b c : ℝ) (h1 : a + b + c = 120) (h2 : a + 5 = b - 5) (h3 : b - 5 = c^2) : b = 61.25 :=
by {
  sorry
}

end find_b_l155_155762


namespace smallest_prime_with_digit_sum_23_l155_155008

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l155_155008


namespace greatest_integer_of_negative_fraction_l155_155431

-- Define the original fraction
def original_fraction : ℚ := -19 / 5

-- Define the greatest integer function
def greatest_integer_less_than (q : ℚ) : ℤ :=
  Int.floor q

-- The proof problem statement:
theorem greatest_integer_of_negative_fraction :
  greatest_integer_less_than original_fraction = -4 :=
sorry

end greatest_integer_of_negative_fraction_l155_155431


namespace find_value_l155_155412

theorem find_value : (100 + (20 / 90)) * 90 = 120 := by
  sorry

end find_value_l155_155412


namespace minimum_stamps_l155_155393

theorem minimum_stamps (c f : ℕ) (h : 3 * c + 4 * f = 50) : c + f = 13 :=
sorry

end minimum_stamps_l155_155393


namespace sum_of_two_integers_l155_155297

theorem sum_of_two_integers (a b : ℕ) (h1 : a * b + a + b = 113) (h2 : Nat.gcd a b = 1) (h3 : a < 25) (h4 : b < 25) : a + b = 23 := by
  sorry

end sum_of_two_integers_l155_155297


namespace valid_reasonings_l155_155265

-- Define the conditions as hypotheses
def analogical_reasoning (R1 : Prop) : Prop := R1
def inductive_reasoning (R2 R4 : Prop) : Prop := R2 ∧ R4
def invalid_generalization (R3 : Prop) : Prop := ¬R3

-- Given the conditions, prove that the valid reasonings are (1), (2), and (4)
theorem valid_reasonings
  (R1 : Prop) (R2 : Prop) (R3 : Prop) (R4 : Prop)
  (h1 : analogical_reasoning R1) 
  (h2 : inductive_reasoning R2 R4) 
  (h3 : invalid_generalization R3) : 
  R1 ∧ R2 ∧ R4 :=
by 
  sorry

end valid_reasonings_l155_155265


namespace sector_area_eq_4cm2_l155_155159

variable (α : ℝ) (l : ℝ) (R : ℝ)
variable (h_alpha : α = 2) (h_l : l = 4) (h_R : R = l / α)

theorem sector_area_eq_4cm2
    (h_alpha : α = 2)
    (h_l : l = 4)
    (h_R : R = l / α) :
    (1/2 * l * R) = 4 := by
  sorry

end sector_area_eq_4cm2_l155_155159


namespace smaller_angle_at_8_15_l155_155701

def angle_minute_hand_at_8_15: ℝ := 90
def angle_hour_hand_at_8: ℝ := 240
def additional_angle_hour_hand_at_8_15: ℝ := 7.5
def total_angle_hour_hand_at_8_15 := angle_hour_hand_at_8 + additional_angle_hour_hand_at_8_15

theorem smaller_angle_at_8_15 :
  min (abs (total_angle_hour_hand_at_8_15 - angle_minute_hand_at_8_15))
      (abs (360 - (total_angle_hour_hand_at_8_15 - angle_minute_hand_at_8_15))) = 157.5 :=
by
  sorry

end smaller_angle_at_8_15_l155_155701


namespace calculate_total_feet_in_garden_l155_155214

-- Define the entities in the problem
def dogs := 6
def feet_per_dog := 4

def ducks := 2
def feet_per_duck := 2

-- Define the total number of feet in the garden
def total_feet_in_garden : Nat :=
  (dogs * feet_per_dog) + (ducks * feet_per_duck)

-- Theorem to state the total number of feet in the garden
theorem calculate_total_feet_in_garden :
  total_feet_in_garden = 28 :=
by
  sorry

end calculate_total_feet_in_garden_l155_155214


namespace solve_fra_eq_l155_155284

theorem solve_fra_eq : ∀ x : ℝ, (x - 2) / (x + 2) + 4 / (x^2 - 4) = 1 → x = 3 :=
by 
  -- Proof steps go here
  sorry

end solve_fra_eq_l155_155284


namespace prove_f_3_equals_11_l155_155623

-- Assuming the given function definition as condition
def f (y : ℝ) : ℝ := sorry

-- The condition provided: f(x - 1/x) = x^2 + 1/x^2.
axiom function_definition (x : ℝ) (h : x ≠ 0): f (x - 1 / x) = x^2 + 1 / x^2

-- The goal is to prove that f(3) = 11
theorem prove_f_3_equals_11 : f 3 = 11 :=
by
  sorry

end prove_f_3_equals_11_l155_155623


namespace cos_inequality_for_triangle_l155_155413

theorem cos_inequality_for_triangle (A B C : ℝ) (h : A + B + C = π) :
  (1 / 3) * (Real.cos A + Real.cos B + Real.cos C) ≤ (1 / 2) ∧
  (1 / 2) ≤ Real.sqrt ((1 / 3) * (Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2)) :=
by
  sorry

end cos_inequality_for_triangle_l155_155413


namespace geom_sequence_a_n_l155_155934

variable {a : ℕ → ℝ}

-- Given conditions
def is_geom_seq (a : ℕ → ℝ) : Prop :=
  |a 1| = 1 ∧ a 5 = -8 * a 2 ∧ a 5 > a 2

-- Statement to prove
theorem geom_sequence_a_n (h : is_geom_seq a) : ∀ n, a n = (-2) ^ (n - 1) := 
by
  sorry

end geom_sequence_a_n_l155_155934


namespace part_I_solution_part_II_solution_l155_155046

-- Part (I) proof problem: Prove the solution set for a specific inequality
theorem part_I_solution (x : ℝ) : -6 < x ∧ x < 10 / 3 → |2 * x - 2| + x + 1 < 9 :=
by
  sorry

-- Part (II) proof problem: Prove the range of 'a' for a given inequality to hold
theorem part_II_solution (a : ℝ) : (-3 ≤ a ∧ a ≤ 17 / 3) →
  (∀ x : ℝ, x ≥ 2 → |a * x + a - 4| + x + 1 ≤ (x + 2)^2) :=
by
  sorry

end part_I_solution_part_II_solution_l155_155046


namespace algebraic_expression_value_l155_155486

theorem algebraic_expression_value
  (x : ℝ)
  (h : 2 * x^2 + 3 * x + 1 = 10) :
  4 * x^2 + 6 * x + 1 = 19 := 
by
  sorry

end algebraic_expression_value_l155_155486


namespace exercise_l155_155933

theorem exercise (n : ℕ) : 900 ∣ (6 ^ (2 * (n + 1)) - 2 ^ (n + 3) * 3 ^ (n + 2) + 36) :=
sorry

end exercise_l155_155933


namespace trucks_needed_l155_155872

-- Definitions of the conditions
def total_apples : ℕ := 80
def apples_transported : ℕ := 56
def truck_capacity : ℕ := 4

-- Definition to calculate the remaining apples
def remaining_apples : ℕ := total_apples - apples_transported

-- The theorem statement
theorem trucks_needed : remaining_apples / truck_capacity = 6 := by
  sorry

end trucks_needed_l155_155872


namespace people_remaining_on_bus_l155_155755

theorem people_remaining_on_bus
  (students_left : ℕ) (students_right : ℕ) (students_back : ℕ)
  (students_aisle : ℕ) (teachers : ℕ) (bus_driver : ℕ) 
  (students_off1 : ℕ) (teachers_off1 : ℕ)
  (students_off2 : ℕ) (teachers_off2 : ℕ)
  (students_off3 : ℕ) :
  students_left = 42 ∧ students_right = 38 ∧ students_back = 5 ∧
  students_aisle = 15 ∧ teachers = 2 ∧ bus_driver = 1 ∧
  students_off1 = 14 ∧ teachers_off1 = 1 ∧
  students_off2 = 18 ∧ teachers_off2 = 1 ∧
  students_off3 = 5 →
  (students_left + students_right + students_back + students_aisle + teachers + bus_driver) -
  (students_off1 + teachers_off1 + students_off2 + teachers_off2 + students_off3) = 64 :=
by {
  sorry
}

end people_remaining_on_bus_l155_155755


namespace sum_of_digits_l155_155318

theorem sum_of_digits (d : ℕ) (h1 : d % 5 = 0) (h2 : 3 * d - 75 = d) : 
  (d / 10 + d % 10) = 11 :=
by {
  -- Placeholder for the proof
  sorry
}

end sum_of_digits_l155_155318


namespace xy_condition_l155_155269

variable (x y : ℝ) -- This depends on the problem context specifying real numbers.

theorem xy_condition (h : x ≠ 0 ∧ y ≠ 0) : (x + y = 0 ↔ y / x + x / y = -2) :=
  sorry

end xy_condition_l155_155269


namespace root_expr_value_eq_175_div_11_l155_155850

noncomputable def root_expr_value (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + bc + ca = 25) (h3 : abc = 10) : ℝ :=
  (a / (1 / a + b * c)) + (b / (1 / b + c * a)) + (c / (1 / c + a * b))

theorem root_expr_value_eq_175_div_11 (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : ab + bc + ca = 25) 
  (h3 : abc = 10) : 
  root_expr_value a b c h1 h2 h3 = 175 / 11 := 
sorry

end root_expr_value_eq_175_div_11_l155_155850


namespace simplify_fraction_l155_155584

theorem simplify_fraction :
  (3 - 6 + 12 - 24 + 48 - 96) / (6 - 12 + 24 - 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end simplify_fraction_l155_155584


namespace geometric_series_sum_l155_155287

theorem geometric_series_sum : 
  let a := 1 
  let r := 2 
  let n := 11 
  let S_n := (a * (1 - r^n)) / (1 - r)
  S_n = 2047 := by
  -- The proof steps would normally go here.
  sorry

end geometric_series_sum_l155_155287


namespace candy_bar_calories_unit_l155_155161

-- Definitions based on conditions
def calories_unit := "calories per candy bar"

-- There are 4 units of calories in a candy bar
def units_per_candy_bar : ℕ := 4

-- There are 2016 calories in 42 candy bars
def total_calories : ℕ := 2016
def number_of_candy_bars : ℕ := 42

-- The statement to prove
theorem candy_bar_calories_unit : (total_calories / number_of_candy_bars = 48) → calories_unit = "calories per candy bar" :=
by
  sorry

end candy_bar_calories_unit_l155_155161


namespace production_days_l155_155111

variable (n : ℕ) (average_past : ℝ := 50) (production_today : ℝ := 115) (new_average : ℝ := 55)

theorem production_days (h1 : average_past * n + production_today = new_average * (n + 1)) : 
    n = 12 := 
by 
  sorry

end production_days_l155_155111


namespace markup_percent_based_on_discounted_price_l155_155461

-- Defining the conditions
def original_price : ℝ := 1
def discount_percent : ℝ := 0.2
def discounted_price : ℝ := original_price * (1 - discount_percent)

-- The proof problem statement
theorem markup_percent_based_on_discounted_price :
  (original_price - discounted_price) / discounted_price = 0.25 :=
sorry

end markup_percent_based_on_discounted_price_l155_155461


namespace correct_answer_is_C_l155_155217

structure Point where
  x : ℤ
  y : ℤ

def inSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

def A : Point := ⟨1, -1⟩
def B : Point := ⟨0, 2⟩
def C : Point := ⟨-3, 2⟩
def D : Point := ⟨4, 0⟩

theorem correct_answer_is_C : inSecondQuadrant C := sorry

end correct_answer_is_C_l155_155217


namespace curve_C_is_circle_area_AOB_constant_find_valid_a_and_curve_eq_l155_155163

-- Define the equation of the curve C
def curve_C (a x y : ℝ) : Prop := a * x^2 + a * y^2 - 2 * a^2 * x - 4 * y = 0

-- Prove that curve C is a circle
theorem curve_C_is_circle (a : ℝ) (h : a ≠ 0) :
  ∃ (h_c : ℝ), ∃ (k : ℝ), ∃ (r : ℝ), r > 0 ∧ ∀ (x y : ℝ), curve_C a x y ↔ (x - h_c)^2 + (y - k)^2 = r^2
:= sorry

-- Prove that the area of triangle AOB is constant
theorem area_AOB_constant (a : ℝ) (h : a ≠ 0) :
  ∃ (A B : ℝ × ℝ), (A = (2 * a, 0) ∧ B = (0, 4 / a)) ∧ 1/2 * (2 * a) * (4 / a) = 4
:= sorry

-- Find valid a and equation of curve C given conditions of line l and points M, N
theorem find_valid_a_and_curve_eq (a : ℝ) (h : a ≠ 0) :
  ∀ (M N : ℝ × ℝ), (|M.1 - 0| = |N.1 - 0| ∧ |M.2 - 0| = |N.2 - 0|) → (M.1 = N.1 ∧ M.2 = N.2) →
  y = -2 * x + 4 →  a = 2 ∧ ∀ (x y : ℝ), curve_C 2 x y ↔ x^2 + y^2 - 4 * x - 2 * y = 0
:= sorry

end curve_C_is_circle_area_AOB_constant_find_valid_a_and_curve_eq_l155_155163


namespace sequence_general_term_l155_155396

theorem sequence_general_term (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n, n ≥ 1 → a (n + 1) = a n + 2) : 
  ∀ n, a n = 2 * n - 1 := 
by 
  sorry

end sequence_general_term_l155_155396


namespace percentage_in_biology_is_correct_l155_155375

/-- 
There are 840 students at a college.
546 students are not enrolled in a biology class.
We need to show what percentage of students are enrolled in biology classes.
--/

def num_students := 840
def not_in_biology := 546

def percentage_in_biology : ℕ := 
  ((num_students - not_in_biology) * 100) / num_students

theorem percentage_in_biology_is_correct : percentage_in_biology = 35 := 
  by
    -- proof is skipped
    sorry

end percentage_in_biology_is_correct_l155_155375


namespace ratio_of_chickens_in_run_to_coop_l155_155110

def chickens_in_coop : ℕ := 14
def free_ranging_chickens : ℕ := 52
def run_condition (R : ℕ) : Prop := 2 * R - 4 = 52

theorem ratio_of_chickens_in_run_to_coop (R : ℕ) (hR : run_condition R) :
  R / chickens_in_coop = 2 :=
by
  sorry

end ratio_of_chickens_in_run_to_coop_l155_155110


namespace product_of_roots_l155_155745

theorem product_of_roots :
  let a := 24
  let c := -216
  ∀ x : ℝ, (24 * x^2 + 36 * x - 216 = 0) → (c / a = -9) :=
by
  intros
  sorry

end product_of_roots_l155_155745


namespace simplify_expr1_simplify_expr2_l155_155951

noncomputable section

-- Problem 1: Simplify the given expression
theorem simplify_expr1 (a b : ℝ) : 4 * a^2 + 2 * (3 * a * b - 2 * a^2) - (7 * a * b - 1) = -a * b + 1 := 
by sorry

-- Problem 2: Simplify the given expression
theorem simplify_expr2 (x y : ℝ) : 3 * (x^2 * y - 1/2 * x * y^2) - 1/2 * (4 * x^2 * y - 3 * x * y^2) = x^2 * y :=
by sorry

end simplify_expr1_simplify_expr2_l155_155951


namespace chameleons_color_change_l155_155120

theorem chameleons_color_change (total_chameleons : ℕ) (initial_blue_red_ratio : ℕ) 
  (final_blue_ratio : ℕ) (final_red_ratio : ℕ) (initial_total : ℕ) (final_total : ℕ) 
  (initial_red : ℕ) (final_red : ℕ) (blues_decrease_by : ℕ) (reds_increase_by : ℕ) 
  (x : ℕ) :
  total_chameleons = 140 →
  initial_blue_red_ratio = 5 →
  final_blue_ratio = 1 →
  final_red_ratio = 3 →
  initial_total = 140 →
  final_total = 140 →
  initial_red = 140 - 5 * x →
  final_red = 3 * (140 - 5 * x) →
  total_chameleons = initial_total →
  (total_chameleons = final_total) →
  initial_red + x = 3 * (140 - 5 * x) →
  blues_decrease_by = (initial_blue_red_ratio - final_blue_ratio) * x →
  reds_increase_by = final_red_ratio * (140 - initial_blue_red_ratio * x) →
  blues_decrease_by = 4 * x →
  x = 20 →
  blues_decrease_by = 80 :=
by
  sorry

end chameleons_color_change_l155_155120


namespace gretel_hansel_salary_difference_l155_155213

theorem gretel_hansel_salary_difference :
  let hansel_initial_salary := 30000
  let hansel_raise_percentage := 10
  let gretel_initial_salary := 30000
  let gretel_raise_percentage := 15
  let hansel_new_salary := hansel_initial_salary + (hansel_raise_percentage / 100 * hansel_initial_salary)
  let gretel_new_salary := gretel_initial_salary + (gretel_raise_percentage / 100 * gretel_initial_salary)
  gretel_new_salary - hansel_new_salary = 1500 := sorry

end gretel_hansel_salary_difference_l155_155213


namespace difference_of_interests_l155_155829

def investment_in_funds (X Y : ℝ) (total_investment : ℝ) : ℝ := X + Y
def interest_earned (investment_rate : ℝ) (amount : ℝ) : ℝ := investment_rate * amount

variable (X : ℝ) (Y : ℝ)
variable (total_investment : ℝ) (rate_X : ℝ) (rate_Y : ℝ)
variable (investment_X : ℝ) 

axiom h1 : total_investment = 100000
axiom h2 : rate_X = 0.23
axiom h3 : rate_Y = 0.17
axiom h4 : investment_X = 42000
axiom h5 : investment_in_funds X Y total_investment = total_investment - investment_X

-- We need to show the difference in interest is 200
theorem difference_of_interests : 
  let interest_X := interest_earned rate_X investment_X
  let investment_Y := total_investment - investment_X
  let interest_Y := interest_earned rate_Y investment_Y
  interest_Y - interest_X = 200 :=
by
  sorry

end difference_of_interests_l155_155829


namespace sum_le_xyz_plus_two_l155_155910

theorem sum_le_xyz_plus_two (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : 
  x + y + z ≤ xyz + 2 := 
sorry

end sum_le_xyz_plus_two_l155_155910


namespace evaluate_expression_l155_155327

theorem evaluate_expression (c : ℕ) (h : c = 4) : (c^c - c * (c - 1)^(c - 1))^c = 148^4 := 
by 
  sorry

end evaluate_expression_l155_155327


namespace sum_of_cosines_dihedral_angles_l155_155605

-- Define the conditions of the problem
def sum_of_plane_angles_trihederal (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- Define the problem statement
theorem sum_of_cosines_dihedral_angles (α β γ : ℝ) (d1 d2 d3 : ℝ)
  (h : sum_of_plane_angles_trihederal α β γ) : 
  d1 + d2 + d3 = 1 :=
  sorry

end sum_of_cosines_dihedral_angles_l155_155605


namespace region_area_l155_155270

theorem region_area (x y : ℝ) : (x^2 + y^2 + 6*x - 4*y - 11 = 0) → (∃ (A : ℝ), A = 24 * Real.pi) :=
by
  sorry

end region_area_l155_155270


namespace quadratic_root_in_interval_l155_155982

variable (a b c : ℝ)

theorem quadratic_root_in_interval 
  (h : 2 * a + 3 * b + 6 * c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
by
  sorry

end quadratic_root_in_interval_l155_155982


namespace jim_loan_inequality_l155_155029

noncomputable def A (t : ℕ) : ℝ := 1500 * (1.06 ^ t)

theorem jim_loan_inequality : ∃ t : ℕ, A t > 3000 ∧ ∀ t' : ℕ, t' < t → A t' ≤ 3000 :=
by
  sorry

end jim_loan_inequality_l155_155029


namespace total_number_of_seats_l155_155067

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end total_number_of_seats_l155_155067


namespace unique_solution_value_l155_155562

theorem unique_solution_value (k : ℝ) :
  (∃ x : ℝ, x^2 = 2 * x + k ∧ ∀ y : ℝ, y^2 = 2 * y + k → y = x) ↔ k = -1 := 
by
  sorry

end unique_solution_value_l155_155562


namespace problem_statement_l155_155955

noncomputable def g (x : ℝ) : ℝ :=
  sorry

theorem problem_statement : (∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x^3 + x^2) → g 3 = -201 / 8 :=
by
  intro h
  sorry

end problem_statement_l155_155955


namespace range_of_a_l155_155443

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then |x - 2 * a| else x + 1 / (x - 2) + a

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f a 2 ≤ f a x) : 1 ≤ a ∧ a ≤ 6 := 
sorry

end range_of_a_l155_155443


namespace num_hens_in_caravan_l155_155662

variable (H G C K : ℕ)  -- number of hens, goats, camels, keepers
variable (total_heads total_feet : ℕ)

-- Defining the conditions
def num_goats := 35
def num_camels := 6
def num_keepers := 10
def heads := H + G + C + K
def feet := 2 * H + 4 * G + 4 * C + 2 * K
def relation := feet = heads + 193

theorem num_hens_in_caravan :
  G = num_goats → C = num_camels → K = num_keepers → relation → 
  H = 60 :=
by 
  intros _ _ _ _
  sorry

end num_hens_in_caravan_l155_155662


namespace measure_AX_l155_155335

-- Definitions based on conditions
def circle_radii (r_A r_B r_C : ℝ) : Prop :=
  r_A - r_B = 6 ∧
  r_A - r_C = 5 ∧
  r_B + r_C = 9

-- Theorem statement
theorem measure_AX (r_A r_B r_C : ℝ) (h : circle_radii r_A r_B r_C) : r_A = 10 :=
by
  sorry

end measure_AX_l155_155335


namespace find_angle_C_find_area_triangle_l155_155769

open Real

-- Let the angles and sides of the triangle be defined as follows
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
axiom condition1 : (a^2 + b^2 - c^2) * (tan C) = sqrt 2 * a * b
axiom condition2 : c = 2
axiom condition3 : b = 2 * sqrt 2

-- Proof statements
theorem find_angle_C :
  C = pi / 4 ∨ C = 3 * pi / 4 :=
sorry

theorem find_area_triangle :
  C = pi / 4 → a = 2 → (1 / 2) * a * b * sin C = 2 :=
sorry

end find_angle_C_find_area_triangle_l155_155769


namespace cyclic_sum_inequality_l155_155447

theorem cyclic_sum_inequality (x y z : ℝ) (hp : x > 0 ∧ y > 0 ∧ z > 0) (h : x + y + z = 3) : 
  (y^2 * z^2 + z^2 * x^2 + x^2 * y^2) < (3 + x * y + y * z + z * x) := by
  sorry

end cyclic_sum_inequality_l155_155447


namespace player_reach_wingspan_l155_155691

theorem player_reach_wingspan :
  ∀ (rim_height player_height jump_height reach_above_rim reach_with_jump reach_wingspan : ℕ),
  rim_height = 120 →
  player_height = 72 →
  jump_height = 32 →
  reach_above_rim = 6 →
  reach_with_jump = player_height + jump_height →
  reach_wingspan = (rim_height + reach_above_rim) - reach_with_jump →
  reach_wingspan = 22 :=
by
  intros rim_height player_height jump_height reach_above_rim reach_with_jump reach_wingspan
  intros h_rim_height h_player_height h_jump_height h_reach_above_rim h_reach_with_jump h_reach_wingspan
  rw [h_rim_height, h_player_height, h_jump_height, h_reach_above_rim] at *
  simp at *
  sorry

end player_reach_wingspan_l155_155691


namespace pos_int_fraction_iff_l155_155101

theorem pos_int_fraction_iff (p : ℕ) (hp : p > 0) : (∃ k : ℕ, 4 * p + 11 = k * (2 * p - 7)) ↔ (p = 4 ∨ p = 5) := 
sorry

end pos_int_fraction_iff_l155_155101


namespace arithmetic_sequence_sum_l155_155610

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_roots : (a 3) * (a 10) - 3 * (a 3 + a 10) - 5 = 0) : a 5 + a 8 = 3 :=
sorry

end arithmetic_sequence_sum_l155_155610


namespace mark_percentage_increase_l155_155005

-- Given a game with the following conditions:
-- Condition 1: Samanta has 8 more points than Mark
-- Condition 2: Eric has 6 points
-- Condition 3: The total points of Samanta, Mark, and Eric is 32

theorem mark_percentage_increase (S M : ℕ) (h1 : S = M + 8) (h2 : 6 + S + M = 32) : 
  (M - 6) / 6 * 100 = 50 :=
sorry

end mark_percentage_increase_l155_155005


namespace heal_time_l155_155659

theorem heal_time (x : ℝ) (hx_pos : 0 < x) (h_total : 2.5 * x = 10) : x = 4 := 
by {
  -- Lean proof will be here
  sorry
}

end heal_time_l155_155659


namespace vector_n_value_l155_155077

theorem vector_n_value {n : ℤ} (hAB : (2, 4) = (2, 4)) (hBC : (-2, n) = (-2, n)) (hAC : (0, 2) = (2 + -2, 4 + n)) : n = -2 :=
by
  sorry

end vector_n_value_l155_155077


namespace moral_of_saying_l155_155323

/-!
  Comrade Mao Zedong said: "If you want to know the taste of a pear, you must change the pear and taste it yourself." 
  Prove that this emphasizes "Practice is the source of knowledge" (option C) over the other options.
-/

def question := "What is the moral of Comrade Mao Zedong's famous saying about tasting a pear?"

def options := ["Knowledge is the driving force behind the development of practice", 
                "Knowledge guides practice", 
                "Practice is the source of knowledge", 
                "Practice has social and historical characteristics"]

def correct_answer := "Practice is the source of knowledge"

theorem moral_of_saying : (question, options[2]) ∈ [("What is the moral of Comrade Mao Zedong's famous saying about tasting a pear?", 
                                                      "Practice is the source of knowledge")] := by 
  sorry

end moral_of_saying_l155_155323


namespace kolya_is_wrong_l155_155501

def pencils_problem_statement (at_least_four_blue : Prop) 
                              (at_least_five_green : Prop) 
                              (at_least_three_blue_and_four_green : Prop) 
                              (at_least_four_blue_and_four_green : Prop) : 
                              Prop :=
  ∃ (B G : ℕ), -- B represents the number of blue pencils, G represents the number of green pencils
    ((B ≥ 4) ∧ (G ≥ 4)) ∧ -- Vasya's statement (at least 4 blue), Petya's and Misha's combined statement (at least 4 green)
    at_least_four_blue ∧ -- Vasya's statement (there are at least 4 blue pencils)
    (at_least_five_green ↔ G ≥ 5) ∧ -- Kolya's statement (there are at least 5 green pencils)
    at_least_three_blue_and_four_green ∧ -- Petya's statement (at least 3 blue and 4 green)
    at_least_four_blue_and_four_green -- Misha's statement (at least 4 blue and 4 green)

theorem kolya_is_wrong (at_least_four_blue : Prop) 
                        (at_least_five_green : Prop) 
                        (at_least_three_blue_and_four_green : Prop) 
                        (at_least_four_blue_and_four_green : Prop) : 
                        pencils_problem_statement at_least_four_blue 
                                                  at_least_five_green 
                                                  at_least_three_blue_and_four_green 
                                                  at_least_four_blue_and_four_green :=
sorry

end kolya_is_wrong_l155_155501


namespace elsie_money_l155_155509

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem elsie_money : 
  compound_interest 2500 0.04 20 = 5477.81 :=
by 
  sorry

end elsie_money_l155_155509


namespace car_travel_distance_l155_155059

theorem car_travel_distance :
  let a := 36
  let d := -12
  let n := 4
  let S := (n / 2) * (2 * a + (n - 1) * d)
  S = 72 := by
    sorry

end car_travel_distance_l155_155059


namespace f_inequality_l155_155289

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem f_inequality (x : ℝ) : f (3^x) ≥ f (2^x) := 
by 
  sorry

end f_inequality_l155_155289


namespace prime_q_exists_l155_155786

theorem prime_q_exists (p : ℕ) (pp : Nat.Prime p) : 
  ∃ q, Nat.Prime q ∧ (∀ n, n > 0 → ¬ q ∣ n ^ p - p) := 
sorry

end prime_q_exists_l155_155786


namespace no_500_good_trinomials_l155_155226

def is_good_quadratic_trinomial (a b c : ℤ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (b^2 - 4 * a * c) > 0

theorem no_500_good_trinomials (S : Finset ℤ) (hS: S.card = 10)
  (hs_pos: ∀ x ∈ S, x > 0) : ¬(∃ T : Finset (ℤ × ℤ × ℤ), 
  T.card = 500 ∧ (∀ (a b c : ℤ), (a, b, c) ∈ T → is_good_quadratic_trinomial a b c)) :=
by
  sorry

end no_500_good_trinomials_l155_155226


namespace distance_third_day_l155_155201

theorem distance_third_day (total_distance : ℝ) (days : ℕ) (first_day_factor : ℝ) (halve_factor : ℝ) (third_day_distance : ℝ) :
  total_distance = 378 ∧ days = 6 ∧ first_day_factor = 4 ∧ halve_factor = 0.5 →
  third_day_distance = 48 := sorry

end distance_third_day_l155_155201


namespace min_value_expression_l155_155183

theorem min_value_expression : ∃ (x y : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 8*x + 6*y + 25 ≥ 0) ∧ (∀ (x y : ℝ), x = 4 ∧ y = -3 → x^2 + y^2 - 8*x + 6*y + 25 = 0) :=
sorry

end min_value_expression_l155_155183


namespace find_m_values_l155_155488

def has_unique_solution (m : ℝ) (A : Set ℝ) : Prop :=
  ∀ x1 x2, x1 ∈ A → x2 ∈ A → x1 = x2

theorem find_m_values :
  {m : ℝ | ∃ A : Set ℝ, has_unique_solution m A ∧ (A = {x | m * x^2 + 2 * x + 3 = 0})} = {0, 1/3} :=
by
  sorry

end find_m_values_l155_155488


namespace geom_seq_prop_l155_155742

variable (b : ℕ → ℝ) (r : ℝ) (s t : ℕ)
variable (h : s ≠ t)
variable (h1 : s > 0) (h2 : t > 0)
variable (h3 : b 1 = 1)
variable (h4 : ∀ n, b (n + 1) = b n * r)

theorem geom_seq_prop : s ≠ t → s > 0 → t > 0 → b 1 = 1 → (∀ n, b (n + 1) = b n * r) → (b t)^(s - 1) / (b s)^(t - 1) = 1 :=
by
  intros h h1 h2 h3 h4
  sorry

end geom_seq_prop_l155_155742


namespace decreasing_function_a_leq_zero_l155_155394

theorem decreasing_function_a_leq_zero (a : ℝ) :
  (∀ x y : ℝ, x < y → ax^3 - x ≥ ay^3 - y) → a ≤ 0 :=
by
  sorry

end decreasing_function_a_leq_zero_l155_155394


namespace solve_for_y_l155_155760

theorem solve_for_y (y : ℝ) (h : 3 * y ^ (1 / 4) - 3 * y ^ (1 / 2) / y ^ (1 / 4) = 13 - 2 * y ^ (1 / 4)) :
  y = (13 / 2) ^ 4 :=
by sorry

end solve_for_y_l155_155760


namespace ratio_students_above_8_to_8_years_l155_155497

-- Definitions of the problem's known conditions
def total_students : ℕ := 125
def students_below_8_years : ℕ := 25
def students_of_8_years : ℕ := 60

-- Main proof inquiry
theorem ratio_students_above_8_to_8_years :
  ∃ (A : ℕ), students_below_8_years + students_of_8_years + A = total_students ∧
             A * 3 = students_of_8_years * 2 := 
sorry

end ratio_students_above_8_to_8_years_l155_155497


namespace sequence_third_term_l155_155508

theorem sequence_third_term (a m : ℤ) (h_a_neg : a < 0) (h_a1 : a + m = 2) (h_a2 : a^2 + m = 4) :
  (a^3 + m = 2) :=
by
  sorry

end sequence_third_term_l155_155508


namespace town_population_growth_l155_155320

noncomputable def populationAfterYears (population : ℝ) (year1Increase : ℝ) (year2Increase : ℝ) : ℝ :=
  let populationAfterFirstYear := population * (1 + year1Increase)
  let populationAfterSecondYear := populationAfterFirstYear * (1 + year2Increase)
  populationAfterSecondYear

theorem town_population_growth :
  ∀ (initialPopulation : ℝ) (year1Increase : ℝ) (year2Increase : ℝ),
    initialPopulation = 1000 → year1Increase = 0.10 → year2Increase = 0.20 →
      populationAfterYears initialPopulation year1Increase year2Increase = 1320 :=
by
  intros initialPopulation year1Increase year2Increase h1 h2 h3
  rw [h1, h2, h3]
  have h4 : populationAfterYears 1000 0.10 0.20 = 1320 := sorry
  exact h4

end town_population_growth_l155_155320


namespace find_a7_l155_155064

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, ∃ r, a (n + m) = (a n) * (r ^ m)

def sequence_properties (a : ℕ → ℝ) : Prop :=
geometric_sequence a ∧ a 3 = 3 ∧ a 11 = 27

theorem find_a7 (a : ℕ → ℝ) (h : sequence_properties a) : a 7 = 9 := 
sorry

end find_a7_l155_155064


namespace min_abs_sum_l155_155124

theorem min_abs_sum (x : ℝ) : (∃ x : ℝ, ∀ y : ℝ, (|y - 2| + |y - 47| ≥ |x - 2| + |x - 47|)) → (|x - 2| + |x - 47| = 45) :=
by
  sorry

end min_abs_sum_l155_155124


namespace product_of_two_numbers_l155_155819

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 6) (h2 : x^2 + y^2 = 170) : x * y = -67 := 
by 
  sorry

end product_of_two_numbers_l155_155819


namespace arithmetic_sequence_sum_l155_155210

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℕ), a 1 = 2 ∧ a 2 + a 3 = 13 → a 4 + a 5 + a 6 = 42 :=
by
  intro a
  intro h
  sorry

end arithmetic_sequence_sum_l155_155210


namespace find_q_l155_155016

-- Define the roots of the polynomial 2x^2 - 6x + 1 = 0
def roots_of_first_poly (a b : ℝ) : Prop :=
    2 * a^2 - 6 * a + 1 = 0 ∧ 2 * b^2 - 6 * b + 1 = 0

-- Conditions from Vieta's formulas for the first polynomial
def sum_of_roots (a b : ℝ) : Prop := a + b = 3
def product_of_roots (a b : ℝ) : Prop := a * b = 0.5

-- Define the roots of the second polynomial x^2 + px + q = 0
def roots_of_second_poly (a b : ℝ) (p q : ℝ) : Prop :=
    (λ x => x^2 + p * x + q) (3 * a - 1) = 0 ∧ 
    (λ x => x^2 + p * x + q) (3 * b - 1) = 0

-- Proof that q = -0.5 given the conditions
theorem find_q (a b p q : ℝ) (h1 : roots_of_first_poly a b) (h2 : sum_of_roots a b)
    (h3 : product_of_roots a b) (h4 : roots_of_second_poly a b p q) : q = -0.5 :=
by
  sorry

end find_q_l155_155016


namespace biker_bob_east_distance_l155_155715

noncomputable def distance_between_towns : ℝ := 28.30194339616981
noncomputable def distance_west : ℝ := 30
noncomputable def distance_north_1 : ℝ := 6
noncomputable def distance_north_2 : ℝ := 18
noncomputable def total_distance_north : ℝ := distance_north_1 + distance_north_2
noncomputable def unknown_distance_east : ℝ := 45.0317 -- Expected distance east

theorem biker_bob_east_distance :
  ∃ (E : ℝ), (total_distance_north ^ 2 + (-distance_west + E) ^ 2 = distance_between_towns ^ 2) ∧ E = unknown_distance_east :=
by 
  sorry

end biker_bob_east_distance_l155_155715


namespace four_thirds_of_number_is_36_l155_155440

theorem four_thirds_of_number_is_36 (x : ℝ) (h : (4 / 3) * x = 36) : x = 27 :=
  sorry

end four_thirds_of_number_is_36_l155_155440


namespace trevor_spends_more_l155_155369

theorem trevor_spends_more (T R Q : ℕ) 
  (hT : T = 80) 
  (hR : R = 2 * Q) 
  (hTotal : 4 * (T + R + Q) = 680) : 
  T = R + 20 :=
by
  sorry

end trevor_spends_more_l155_155369


namespace find_a7_l155_155583

-- Define the arithmetic sequence
def a (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Conditions
def S5 : ℤ := 25
def a2 : ℤ := 3

-- Main Goal: Find a_7
theorem find_a7 (a1 d : ℤ) (h1 : sum_first_n_terms a1 d 5 = S5)
                     (h2 : a a1 d 2 = a2) :
  a a1 d 7 = 13 := 
sorry

end find_a7_l155_155583


namespace ratio_c_d_l155_155278

theorem ratio_c_d (a b c d : ℝ) (h_eq : ∀ x, a * x^3 + b * x^2 + c * x + d = 0) 
    (h_roots : ∀ r, r = 2 ∨ r = 4 ∨ r = 5 ↔ (a * r^3 + b * r^2 + c * r + d = 0)) :
    c / d = 19 / 20 :=
by
  sorry

end ratio_c_d_l155_155278


namespace certain_number_is_48_l155_155117

theorem certain_number_is_48 (x : ℕ) (h : x = 4) : 36 + 3 * x = 48 := by
  sorry

end certain_number_is_48_l155_155117


namespace function_property_l155_155943

theorem function_property (k a b : ℝ) (h : a ≠ b) (h_cond : k * a + 1 / b = k * b + 1 / a) : 
  k * (a * b) + 1 = 0 := 
by 
  sorry

end function_property_l155_155943


namespace next_birthday_monday_l155_155536
open Nat

-- Define the basic structure and parameters of our problem
def is_leap_year (year : ℕ) : Prop := 
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def day_of_week (start_day : ℕ) (year_diff : ℕ) (is_leap : ℕ → Prop) : ℕ :=
  (start_day + year_diff + (year_diff / 4) - (year_diff / 100) + (year_diff / 400)) % 7

-- Specify problem conditions
def initial_year := 2009
def initial_day := 5 -- 2009-06-18 is Friday, which is 5 if we start counting from Sunday as 0
def end_day := 1 -- target day is Monday, which is 1

-- Main theorem
theorem next_birthday_monday : ∃ year, year > initial_year ∧
  day_of_week initial_day (year - initial_year) is_leap_year = end_day := by
  use 2017
  -- The proof would go here, skipping with sorry
  sorry

end next_birthday_monday_l155_155536


namespace problem1421_part1_problem1421_part2_problem1421_part3_problem1421_part4_l155_155790

theorem problem1421_part1 (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ)
  (h_total : total_balls = 20) (h_red : red_balls = 5) (h_yellow : yellow_balls = 15) :
  (red_balls < yellow_balls) := by 
  sorry  -- Solution Proof for Part 1

theorem problem1421_part2 (total_balls : ℕ) (red_balls : ℕ) (h_total : total_balls = 20) 
  (h_red : red_balls = 5) :
  (red_balls / total_balls = 1 / 4) := by 
  sorry  -- Solution Proof for Part 2

theorem problem1421_part3 (red_balls total_balls m : ℕ) (h_red : red_balls = 5) 
  (h_total : total_balls = 20) :
  ((red_balls + m) / (total_balls + m) = 3 / 4) → (m = 40) := by 
  sorry  -- Solution Proof for Part 3

theorem problem1421_part4 (total_balls red_balls additional_balls x : ℕ) 
  (h_total : total_balls = 20) (h_red : red_balls = 5) (h_additional : additional_balls = 18):
  (total_balls + additional_balls = 38) → ((red_balls + x) / 38 = 1 / 2) → 
  (x = 14) ∧ ((additional_balls - x) = 4) := by 
  sorry  -- Solution Proof for Part 4

end problem1421_part1_problem1421_part2_problem1421_part3_problem1421_part4_l155_155790


namespace smallest_base10_num_exists_l155_155162

noncomputable def A_digit_range := {n : ℕ // n < 6}
noncomputable def B_digit_range := {n : ℕ // n < 8}

theorem smallest_base10_num_exists :
  ∃ (A : A_digit_range) (B : B_digit_range), 7 * A.val = 9 * B.val ∧ 7 * A.val = 63 :=
by
  sorry

end smallest_base10_num_exists_l155_155162


namespace sum_of_possible_values_of_g1_l155_155328

def g (x : ℝ) : ℝ := sorry

axiom g_prop : ∀ x y : ℝ, g (g (x - y)) = g x * g y - g x + g y - x^2 * y^2

theorem sum_of_possible_values_of_g1 : g 1 = -1 := by sorry

end sum_of_possible_values_of_g1_l155_155328


namespace mean_age_correct_l155_155353

def children_ages : List ℕ := [6, 6, 9, 12]

def number_of_children : ℕ := 4

def sum_of_ages (ages : List ℕ) : ℕ := ages.sum

def mean_age (ages : List ℕ) (num_children : ℕ) : ℚ :=
  sum_of_ages ages / num_children

theorem mean_age_correct :
  mean_age children_ages number_of_children = 8.25 := by
  sorry

end mean_age_correct_l155_155353


namespace first_train_speed_l155_155414

noncomputable def speed_of_first_train (length_train1 : ℕ) (speed_train2 : ℕ) (length_train2 : ℕ) (time_cross : ℕ) : ℕ :=
  let relative_speed_m_s := (500 : ℕ) / time_cross
  let relative_speed_km_h := relative_speed_m_s * 18 / 5
  relative_speed_km_h - speed_train2

theorem first_train_speed :
  speed_of_first_train 270 80 230 9 = 920 := by
  sorry

end first_train_speed_l155_155414


namespace total_sign_up_methods_l155_155815

theorem total_sign_up_methods (n : ℕ) (k : ℕ) (h1 : n = 4) (h2 : k = 2) :
  k ^ n = 16 :=
by
  rw [h1, h2]
  norm_num

end total_sign_up_methods_l155_155815


namespace transformer_minimum_load_l155_155063

-- Define the conditions as hypotheses
def running_current_1 := 40
def running_current_2 := 60
def running_current_3 := 25

def start_multiplier_1 := 2
def start_multiplier_2 := 3
def start_multiplier_3 := 4

def units_1 := 3
def units_2 := 2
def units_3 := 1

def starting_current_1 := running_current_1 * start_multiplier_1
def starting_current_2 := running_current_2 * start_multiplier_2
def starting_current_3 := running_current_3 * start_multiplier_3

def total_starting_current_1 := starting_current_1 * units_1
def total_starting_current_2 := starting_current_2 * units_2
def total_starting_current_3 := starting_current_3 * units_3

def total_combined_minimum_current_load := 
  total_starting_current_1 + total_starting_current_2 + total_starting_current_3

-- The theorem to prove that the total combined minimum current load is 700A
theorem transformer_minimum_load : total_combined_minimum_current_load = 700 := by
  sorry

end transformer_minimum_load_l155_155063


namespace ratio_of_volumes_l155_155151
-- Import the necessary library

-- Define the edge lengths of the cubes
def small_cube_edge : ℕ := 4
def large_cube_edge : ℕ := 24

-- Define the volumes of the cubes
def volume (edge : ℕ) : ℕ := edge ^ 3

-- Define the volumes
def V_small := volume small_cube_edge
def V_large := volume large_cube_edge

-- State the main theorem
theorem ratio_of_volumes : V_small / V_large = 1 / 216 := 
by 
  -- we skip the proof here
  sorry

end ratio_of_volumes_l155_155151


namespace eccentricity_is_two_l155_155342

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 3) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem eccentricity_is_two (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 3) : 
  eccentricity_of_hyperbola a b h1 h2 h3 = 2 := 
  sorry

end eccentricity_is_two_l155_155342


namespace quadratic_range_m_l155_155874

theorem quadratic_range_m (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m * x + 1 = 0 ∧ y^2 + m * y + 1 = 0) ↔ (m < -2 ∨ m > 2) :=
by 
  sorry

end quadratic_range_m_l155_155874


namespace window_treatments_total_cost_l155_155528

def sheers_cost_per_pair := 40
def drapes_cost_per_pair := 60
def number_of_windows := 3

theorem window_treatments_total_cost :
  (number_of_windows * sheers_cost_per_pair) + (number_of_windows * drapes_cost_per_pair) = 300 :=
by 
  -- calculations omitted
  sorry

end window_treatments_total_cost_l155_155528


namespace jane_started_babysitting_at_age_18_l155_155565

-- Define the age Jane started babysitting
def jane_starting_age := 18

-- State Jane's current age
def jane_current_age : ℕ := 34

-- State the years since Jane stopped babysitting
def years_since_jane_stopped := 12

-- Calculate Jane's age when she stopped babysitting
def jane_age_when_stopped : ℕ := jane_current_age - years_since_jane_stopped

-- State the current age of the oldest person she could have babysat
def current_oldest_child_age : ℕ := 25

-- Calculate the age of the oldest child when Jane stopped babysitting
def age_oldest_child_when_stopped : ℕ := current_oldest_child_age - years_since_jane_stopped

-- State the condition that the child was no more than half her age at the time
def child_age_condition (jane_age : ℕ) (child_age : ℕ) : Prop := child_age ≤ jane_age / 2

-- The theorem to prove the age Jane started babysitting
theorem jane_started_babysitting_at_age_18
  (jane_current : jane_current_age = 34)
  (years_stopped : years_since_jane_stopped = 12)
  (current_oldest : current_oldest_child_age = 25)
  (age_when_stopped : jane_age_when_stopped = 22)
  (child_when_stopped : age_oldest_child_when_stopped = 13)
  (child_condition : ∀ {j : ℕ}, child_age_condition j age_oldest_child_when_stopped → False) :
  jane_starting_age = 18 :=
sorry

end jane_started_babysitting_at_age_18_l155_155565


namespace find_m_of_hyperbola_l155_155727

theorem find_m_of_hyperbola (m : ℝ) (h : mx^2 + y^2 = 1) (s : ∃ x : ℝ, x = 2) : m = -4 := 
by
  sorry

end find_m_of_hyperbola_l155_155727


namespace vershoks_per_arshin_l155_155801

theorem vershoks_per_arshin (plank_length_arshins : ℝ) (plank_width_vershoks : ℝ) 
    (room_side_length_arshins : ℝ) (total_planks : ℕ) (n : ℝ)
    (h1 : plank_length_arshins = 6) (h2 : plank_width_vershoks = 6)
    (h3 : room_side_length_arshins = 12) (h4 : total_planks = 64) 
    (h5 : (total_planks : ℝ) * (plank_length_arshins * (plank_width_vershoks / n)) = room_side_length_arshins^2) :
    n = 16 :=
by {
  sorry
}

end vershoks_per_arshin_l155_155801


namespace ratio_income_to_expenditure_l155_155805

theorem ratio_income_to_expenditure (I E S : ℕ) 
  (h1 : I = 10000) 
  (h2 : S = 3000) 
  (h3 : S = I - E) : I / Nat.gcd I E = 10 ∧ E / Nat.gcd I E = 7 := by 
  sorry

end ratio_income_to_expenditure_l155_155805


namespace repeating_decimal_exceeds_decimal_l155_155578

noncomputable def repeating_decimal_to_fraction : ℚ := 9 / 11
noncomputable def decimal_to_fraction : ℚ := 3 / 4

theorem repeating_decimal_exceeds_decimal :
  repeating_decimal_to_fraction - decimal_to_fraction = 3 / 44 :=
by
  sorry

end repeating_decimal_exceeds_decimal_l155_155578


namespace correct_statements_l155_155282

-- Define the statements
def statement_1 := true
def statement_2 := false
def statement_3 := true
def statement_4 := true

-- Define a function to count the number of true statements
def num_correct_statements (s1 s2 s3 s4 : Bool) : Nat :=
  [s1, s2, s3, s4].countP id

-- Define the theorem to prove that the number of correct statements is 3
theorem correct_statements :
  num_correct_statements statement_1 statement_2 statement_3 statement_4 = 3 :=
by
  -- You can use sorry to skip the proof
  sorry

end correct_statements_l155_155282


namespace arith_seq_sum_ratio_l155_155599

theorem arith_seq_sum_ratio 
  (S : ℕ → ℝ) 
  (a1 d : ℝ) 
  (h1 : S 1 = 1) 
  (h2 : (S 4) / (S 2) = 4) :
  (S 6) / (S 4) = 9 / 4 :=
sorry

end arith_seq_sum_ratio_l155_155599


namespace worm_length_l155_155593

theorem worm_length (l1 l2 : ℝ) (h1 : l1 = 0.8) (h2 : l2 = l1 + 0.7) : l1 = 0.8 :=
by
  exact h1

end worm_length_l155_155593


namespace domain_sqrt_quot_l155_155432

noncomputable def domain_of_function (f : ℝ → ℝ) : Set ℝ := {x : ℝ | f x ≠ 0}

theorem domain_sqrt_quot (x : ℝ) :
  (x + 1 ≥ 0) ∧ (x ≠ 0) ↔ (x ∈ {x : ℝ | -1 ≤ x ∧ x < 0} ∪ {x : ℝ | x > 0}) :=
by
  sorry

end domain_sqrt_quot_l155_155432


namespace value_of_expression_l155_155898

theorem value_of_expression : (2^4 - 2) / (2^3 - 1) = 2 := by
  sorry

end value_of_expression_l155_155898


namespace base7_to_base10_of_645_l155_155115

theorem base7_to_base10_of_645 :
  (6 * 7^2 + 4 * 7^1 + 5 * 7^0) = 327 := 
by 
  sorry

end base7_to_base10_of_645_l155_155115


namespace knight_min_moves_l155_155150

theorem knight_min_moves (n : ℕ) (h : n ≥ 4) : 
  ∃ k : ℕ, k = 2 * (Nat.floor ((n + 1 : ℚ) / 3)) ∧
  (∀ m, (3 * m) ≥ (2 * (n - 1)) → ∃ l, l = 2 * m ∧ l ≥ k) :=
by
  sorry

end knight_min_moves_l155_155150


namespace total_balloons_after_gift_l155_155272

-- Definitions for conditions
def initial_balloons := 26
def additional_balloons := 34

-- Proposition for the total number of balloons
theorem total_balloons_after_gift : initial_balloons + additional_balloons = 60 := 
by
  -- Proof omitted, adding sorry
  sorry

end total_balloons_after_gift_l155_155272


namespace cans_of_beans_is_two_l155_155787

-- Define the problem parameters
variable (C B T : ℕ)

-- Conditions based on the problem statement
axiom chili_can : C = 1
axiom tomato_to_bean_ratio : T = 3 * B / 2
axiom quadruple_batch_cans : 4 * (C + B + T) = 24

-- Prove the number of cans of beans is 2
theorem cans_of_beans_is_two : B = 2 :=
by
  -- Include conditions
  have h1 : C = 1 := by sorry
  have h2 : T = 3 * B / 2 := by sorry
  have h3 : 4 * (C + B + T) = 24 := by sorry
  -- Derive the answer (Proof omitted)
  sorry

end cans_of_beans_is_two_l155_155787


namespace solve_equation1_solve_equation2_l155_155764

theorem solve_equation1 (x : ℝ) (h : 4 * x^2 - 81 = 0) : x = 9/2 ∨ x = -9/2 := 
sorry

theorem solve_equation2 (x : ℝ) (h : 8 * (x + 1)^3 = 27) : x = 1/2 := 
sorry

end solve_equation1_solve_equation2_l155_155764


namespace triangles_congruence_l155_155293

theorem triangles_congruence (A_1 B_1 C_1 A_2 B_2 C_2 : ℝ)
  (angle_A1 angle_B1 angle_C1 angle_A2 angle_B2 angle_C2 : ℝ)
  (h_side1 : A_1 = A_2) 
  (h_side2 : B_1 = B_2)
  (h_angle1 : angle_A1 = angle_A2)
  (h_angle2 : angle_B1 = angle_B2)
  (h_angle3 : angle_C1 = angle_C2) : 
  ¬((A_1 = C_1) ∧ (B_1 = C_2) ∧ (angle_A1 = angle_B2) ∧ (angle_B1 = angle_A2) ∧ (angle_C1 = angle_B2) → 
     (A_1 = A_2) ∧ (B_1 = B_2) ∧ (C_1 = C_2)) :=
by {
  sorry
}

end triangles_congruence_l155_155293


namespace santiago_stay_in_australia_l155_155881

/-- Santiago leaves his home country in the month of January,
    stays in Australia for a few months,
    and returns on the same date in the month of December.
    Prove that Santiago stayed in Australia for 11 months. -/
theorem santiago_stay_in_australia :
  ∃ (months : ℕ), months = 11 ∧
  (months = if (departure_month = 1) ∧ (return_month = 12) then 11 else 0) :=
by sorry

end santiago_stay_in_australia_l155_155881


namespace sequence_2007th_number_l155_155648

-- Defining the sequence according to the given rule
def a (n : ℕ) : ℕ := 2 ^ n

theorem sequence_2007th_number : a 2007 = 2 ^ 2007 :=
by
  -- Proof is omitted
  sorry

end sequence_2007th_number_l155_155648


namespace find_the_number_l155_155472

theorem find_the_number (x : ℤ) (h : 2 + x = 6) : x = 4 :=
sorry

end find_the_number_l155_155472


namespace tuesday_pairs_of_boots_l155_155241

theorem tuesday_pairs_of_boots (S B : ℝ) (x : ℤ) 
  (h1 : 22 * S + 16 * B = 460)
  (h2 : 8 * S + x * B = 560)
  (h3 : B = S + 15) : 
  x = 24 :=
sorry

end tuesday_pairs_of_boots_l155_155241


namespace find_distance_between_sides_of_trapezium_l155_155385

variable (side1 side2 h area : ℝ)
variable (h1 : side1 = 20)
variable (h2 : side2 = 18)
variable (h3 : area = 228)
variable (trapezium_area : area = (1 / 2) * (side1 + side2) * h)

theorem find_distance_between_sides_of_trapezium : h = 12 := by
  sorry

end find_distance_between_sides_of_trapezium_l155_155385


namespace xy_sum_l155_155754

theorem xy_sum (x y : ℝ) (h1 : 2 / x + 3 / y = 4) (h2 : 2 / x - 3 / y = -2) : x + y = 3 := by
  sorry

end xy_sum_l155_155754


namespace molecular_weight_al_fluoride_l155_155345

/-- Proving the molecular weight of Aluminum fluoride calculation -/
theorem molecular_weight_al_fluoride (x : ℕ) (h : 10 * x = 840) : x = 84 :=
by sorry

end molecular_weight_al_fluoride_l155_155345


namespace total_number_of_squares_up_to_50th_ring_l155_155271

def number_of_squares_up_to_50th_ring : Nat :=
  let central_square := 1
  let sum_rings := (50 * (50 + 1)) * 4  -- Using the formula for arithmetic series sum where a = 8 and d = 8 and n = 50
  central_square + sum_rings

theorem total_number_of_squares_up_to_50th_ring : number_of_squares_up_to_50th_ring = 10201 :=
  by  -- This statement means we believe the theorem is true and will be proven.
    sorry                                                      -- Proof omitted, will need to fill this in later

end total_number_of_squares_up_to_50th_ring_l155_155271


namespace find_P_l155_155902

noncomputable def P (x : ℝ) : ℝ :=
  4 * x^3 - 6 * x^2 - 12 * x

theorem find_P (a b c : ℝ) (h_root : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_roots : ∀ x, x^3 - 2 * x^2 - 4 * x - 1 = 0 ↔ x = a ∨ x = b ∨ x = c)
  (h_Pa : P a = b + 2 * c)
  (h_Pb : P b = 2 * a + c)
  (h_Pc : P c = a + 2 * b)
  (h_Psum : P (a + b + c) = -20) :
  ∀ x, P x = 4 * x^3 - 6 * x^2 - 12 * x :=
by
  sorry

end find_P_l155_155902


namespace concert_ticket_to_motorcycle_ratio_l155_155456

theorem concert_ticket_to_motorcycle_ratio (initial_amount spend_motorcycle remaining_amount : ℕ)
  (h_initial : initial_amount = 5000)
  (h_spend_motorcycle : spend_motorcycle = 2800)
  (amount_left := initial_amount - spend_motorcycle)
  (h_remaining : remaining_amount = 825)
  (h_amount_left : ∃ C : ℕ, amount_left - C - (1/4 : ℚ) * (amount_left - C) = remaining_amount) :
  ∃ C : ℕ, (C / amount_left) = (1 / 2 : ℚ) := sorry

end concert_ticket_to_motorcycle_ratio_l155_155456


namespace students_use_red_color_l155_155242

theorem students_use_red_color
  (total_students : ℕ)
  (students_use_green : ℕ)
  (students_use_both : ℕ)
  (total_students_eq : total_students = 70)
  (students_use_green_eq : students_use_green = 52)
  (students_use_both_eq : students_use_both = 38) :
  ∃ (students_use_red : ℕ), students_use_red = 56 :=
by
  -- We will skip the proof part as specified
  sorry

end students_use_red_color_l155_155242


namespace windmere_zoo_two_legged_birds_l155_155807

theorem windmere_zoo_two_legged_birds (b m u : ℕ) (head_count : b + m + u = 300) (leg_count : 2 * b + 4 * m + 3 * u = 710) : b = 230 :=
sorry

end windmere_zoo_two_legged_birds_l155_155807


namespace expected_number_of_returns_l155_155156

noncomputable def expected_returns_to_zero : ℝ :=
  let p_move := 1 / 3
  let expected_value := -1 + (3 / (Real.sqrt 5))
  expected_value

theorem expected_number_of_returns : expected_returns_to_zero = (3 * Real.sqrt 5 - 5) / 5 :=
  by sorry

end expected_number_of_returns_l155_155156


namespace problem1_problem2_l155_155109

noncomputable def A : Set ℝ := Set.Icc 1 4
noncomputable def B (a : ℝ) : Set ℝ := Set.Iio a

-- Problem 1
theorem problem1 (A := A) (B := B 4) : A ∩ B = Set.Icc 1 4 := by
  sorry 

-- Problem 2
theorem problem2 (A := A) : ∀ a : ℝ, (A ⊆ B a) → (4 ≤ a) := by
  sorry

end problem1_problem2_l155_155109


namespace students_played_both_l155_155382

theorem students_played_both (C B X total : ℕ) (hC : C = 500) (hB : B = 600) (hTotal : total = 880) (hInclusionExclusion : C + B - X = total) : X = 220 :=
by
  rw [hC, hB, hTotal] at hInclusionExclusion
  sorry

end students_played_both_l155_155382


namespace equivalent_problem_l155_155957

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

def is_geometric_sequence (b : ℕ → ℝ) :=
  ∀ n m : ℕ, b (n + 1) / b n = b (m + 1) / b m

theorem equivalent_problem
  (ha : is_arithmetic_sequence a)
  (hb : is_geometric_sequence b)
  (h1 : a 1 + a 3 + a 5 + a 7 + a 9 = 50)
  (h2 : b 4 * b 6 * b 14 * b 16 = 625) :
  (a 2 + a 8) / b 10 = 4 ∨ (a 2 + a 8) / b 10 = -4 := 
sorry

end equivalent_problem_l155_155957


namespace range_of_m_l155_155665

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := -x^2 + x + m + 2

theorem range_of_m (m : ℝ) : 
  (∃! x : ℤ, f x m ≥ |x|) ↔ -2 ≤ m ∧ m < -1 :=
by
  sorry

end range_of_m_l155_155665


namespace problem1_solutions_problem2_solutions_l155_155875

-- Problem 1: Solve x² - 7x + 6 = 0

theorem problem1_solutions (x : ℝ) : 
  x^2 - 7 * x + 6 = 0 ↔ (x = 1 ∨ x = 6) := by
  sorry

-- Problem 2: Solve (2x + 3)² = (x - 3)² 

theorem problem2_solutions (x : ℝ) : 
  (2 * x + 3)^2 = (x - 3)^2 ↔ (x = 0 ∨ x = -6) := by
  sorry

end problem1_solutions_problem2_solutions_l155_155875


namespace gratuity_calculation_correct_l155_155180

noncomputable def tax_rate (item: String): ℝ :=
  if item = "NY Striploin" then 0.10
  else if item = "Glass of wine" then 0.15
  else if item = "Dessert" then 0.05
  else if item = "Bottle of water" then 0.00
  else 0

noncomputable def base_price (item: String): ℝ :=
  if item = "NY Striploin" then 80
  else if item = "Glass of wine" then 10
  else if item = "Dessert" then 12
  else if item = "Bottle of water" then 3
  else 0

noncomputable def total_price_with_tax (item: String): ℝ :=
  base_price item + base_price item * tax_rate item

noncomputable def gratuity (item: String): ℝ :=
  total_price_with_tax item * 0.20

noncomputable def total_gratuity: ℝ :=
  gratuity "NY Striploin" + gratuity "Glass of wine" + gratuity "Dessert" + gratuity "Bottle of water"

theorem gratuity_calculation_correct :
  total_gratuity = 23.02 :=
by
  sorry

end gratuity_calculation_correct_l155_155180


namespace number_of_buckets_after_reduction_l155_155303

def initial_buckets : ℕ := 25
def reduction_factor : ℚ := 2 / 5

theorem number_of_buckets_after_reduction :
  (initial_buckets : ℚ) * (1 / reduction_factor) = 63 := by
  sorry

end number_of_buckets_after_reduction_l155_155303


namespace quadratic_has_two_distinct_real_roots_l155_155710

theorem quadratic_has_two_distinct_real_roots (k : ℝ) :
  (∃ a b c : ℝ, a = k - 2 ∧ b = -2 ∧ c = 1 / 2 ∧ a ≠ 0 ∧ b ^ 2 - 4 * a * c > 0) ↔ (k < 4 ∧ k ≠ 2) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l155_155710


namespace loaned_books_l155_155160

theorem loaned_books (initial_books : ℕ) (returned_percent : ℝ)
  (end_books : ℕ) (damaged_books : ℕ) (L : ℝ) :
  initial_books = 150 ∧
  returned_percent = 0.85 ∧
  end_books = 135 ∧
  damaged_books = 5 ∧
  0.85 * L + 5 + (initial_books - L) = end_books →
  L = 133 :=
by
  intros h
  rcases h with ⟨hb, hr, he, hd, hsum⟩
  repeat { sorry }

end loaned_books_l155_155160


namespace bricklayer_hours_l155_155880

theorem bricklayer_hours
  (B E : ℝ)
  (h1 : B + E = 90)
  (h2 : 12 * B + 16 * E = 1350) :
  B = 22.5 :=
by
  sorry

end bricklayer_hours_l155_155880


namespace compute_expression_l155_155295

theorem compute_expression :
  ( ((15 ^ 15) / (15 ^ 10)) ^ 3 * 5 ^ 6 ) / (25 ^ 2) = 3 ^ 15 * 5 ^ 17 :=
by
  -- We'll use sorry here as proof is not required
  sorry

end compute_expression_l155_155295


namespace abc_gt_16_abc_geq_3125_div_108_l155_155338

variables {a b c α β : ℝ}

-- Define the conditions
def conditions (a b c α β : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ b > 0 ∧
  (a * α^2 + b * α - c = 0) ∧
  (a * β^2 + b * β - c = 0) ∧
  (α ≠ β) ∧
  (α^3 + b * α^2 + a * α - c = 0) ∧
  (β^3 + b * β^2 + a * β - c = 0)

-- State the first proof problem
theorem abc_gt_16 (h : conditions a b c α β) : a * b * c > 16 :=
sorry

-- State the second proof problem
theorem abc_geq_3125_div_108 (h : conditions a b c α β) : a * b * c ≥ 3125 / 108 :=
sorry

end abc_gt_16_abc_geq_3125_div_108_l155_155338


namespace find_m_l155_155540

theorem find_m (m : ℚ) : 
  (∃ m, (∀ x y z : ℚ, ((x, y) = (2, 9) ∨ (x, y) = (15, m) ∨ (x, y) = (35, 4)) ∧ 
  (∀ a b c d e f : ℚ, ((a, b) = (2, 9) ∨ (a, b) = (15, m) ∨ (a, b) = (35, 4)) → 
  ((b - d) / (a - c) = (f - d) / (e - c))) → m = 232 / 33)) :=
sorry

end find_m_l155_155540


namespace tan_sixty_eq_sqrt_three_l155_155551

theorem tan_sixty_eq_sqrt_three : Real.tan (Real.pi / 3) = Real.sqrt 3 := 
by
  sorry

end tan_sixty_eq_sqrt_three_l155_155551


namespace find_general_formula_sum_b_n_less_than_two_l155_155048

noncomputable def a_n (n : ℕ) : ℕ := n

noncomputable def S_n (n : ℕ) : ℚ := (n^2 + n) / 2

noncomputable def b_n (n : ℕ) : ℚ := 1 / S_n n

theorem find_general_formula (n : ℕ) : b_n n = 2 / (n^2 + n) := by 
  sorry

theorem sum_b_n_less_than_two (n : ℕ) :
  Finset.sum (Finset.range n) (λ k => b_n (k + 1)) < 2 :=
by 
  sorry

end find_general_formula_sum_b_n_less_than_two_l155_155048


namespace sum_coordinates_l155_155474

theorem sum_coordinates (x : ℝ) : 
  let C := (x, 8)
  let D := (-x, 8)
  (C.1 + C.2 + D.1 + D.2) = 16 := 
by
  sorry

end sum_coordinates_l155_155474


namespace total_people_going_to_zoo_and_amusement_park_l155_155186

theorem total_people_going_to_zoo_and_amusement_park :
  (7.0 * 45.0) + (5.0 * 56.0) = 595.0 :=
by
  sorry

end total_people_going_to_zoo_and_amusement_park_l155_155186


namespace find_n_l155_155258

noncomputable def cube_probability_solid_color (num_cubes edge_length num_corner num_edge num_face_center num_center : ℕ)
  (corner_prob edge_prob face_center_prob center_prob : ℚ) : ℚ :=
  have total_corner_prob := corner_prob ^ num_corner
  have total_edge_prob := edge_prob ^ num_edge
  have total_face_center_prob := face_center_prob ^ num_face_center
  have total_center_prob := center_prob ^ num_center
  2 * (total_corner_prob * total_edge_prob * total_face_center_prob * total_center_prob)

theorem find_n : ∃ n : ℕ, cube_probability_solid_color 27 3 8 12 6 1
  (1/8) (1/4) (1/2) 1 = (1 / (2 : ℚ) ^ n) ∧ n = 53 := by
  use 53
  simp only [cube_probability_solid_color]
  sorry

end find_n_l155_155258


namespace polygon_interior_angle_sum_l155_155917

theorem polygon_interior_angle_sum (n : ℕ) (h : (n-1) * 180 = 2400 + 120) : n = 16 :=
by
  sorry

end polygon_interior_angle_sum_l155_155917


namespace neon_signs_blink_together_l155_155800

-- Define the time intervals for the blinks
def blink_interval1 : ℕ := 7
def blink_interval2 : ℕ := 11
def blink_interval3 : ℕ := 13

-- Define the least common multiple function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- State the theorem
theorem neon_signs_blink_together : Nat.lcm (Nat.lcm blink_interval1 blink_interval2) blink_interval3 = 1001 := by
  sorry

end neon_signs_blink_together_l155_155800


namespace sam_has_75_dollars_l155_155739

variable (S B : ℕ)

def condition1 := B = 2 * S - 25
def condition2 := S + B = 200

theorem sam_has_75_dollars (h1 : condition1 S B) (h2 : condition2 S B) : S = 75 := by
  sorry

end sam_has_75_dollars_l155_155739


namespace jungkook_colored_paper_count_l155_155311

theorem jungkook_colored_paper_count :
  (3 * 10) + 8 = 38 :=
by sorry

end jungkook_colored_paper_count_l155_155311


namespace find_abc_sum_l155_155496

theorem find_abc_sum :
  ∀ (a b c : ℝ),
    2 * |a + 3| + 4 - b = 0 →
    c^2 + 4 * b - 4 * c - 12 = 0 →
    a + b + c = 5 :=
by
  intros a b c h1 h2
  sorry

end find_abc_sum_l155_155496


namespace profit_share_difference_l155_155041

theorem profit_share_difference
    (P_A P_B P_C P_D : ℕ) (R_A R_B R_C R_D parts_A parts_B parts_C parts_D : ℕ) (profit_B : ℕ)
    (h1 : P_A = 8000) (h2 : P_B = 10000) (h3 : P_C = 12000) (h4 : P_D = 15000)
    (h5 : R_A = 3) (h6 : R_B = 5) (h7 : R_C = 6) (h8 : R_D = 7)
    (h9: profit_B = 2000) :
    profit_B / R_B = 400 ∧ P_C * R_C / R_B - P_A * R_A / R_B = 1200 :=
by
  sorry

end profit_share_difference_l155_155041


namespace number_of_blue_parrots_l155_155689

-- Defining the known conditions
def total_parrots : ℕ := 120
def fraction_red : ℚ := 2 / 3
def fraction_green : ℚ := 1 / 6

-- Proving the number of blue parrots given the conditions
theorem number_of_blue_parrots : (1 - (fraction_red + fraction_green)) * total_parrots = 20 := by
  sorry

end number_of_blue_parrots_l155_155689


namespace triangle_angles_l155_155849

theorem triangle_angles (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 45) : B = 90 ∧ C = 45 :=
sorry

end triangle_angles_l155_155849


namespace sum_of_first_6n_integers_l155_155622

theorem sum_of_first_6n_integers (n : ℕ) (h1 : (5 * n * (5 * n + 1)) / 2 = (n * (n + 1)) / 2 + 200) :
  (6 * n * (6 * n + 1)) / 2 = 300 :=
by
  sorry

end sum_of_first_6n_integers_l155_155622


namespace largest_multiple_5_6_lt_1000_is_990_l155_155543

theorem largest_multiple_5_6_lt_1000_is_990 : ∃ n, (n < 1000) ∧ (n % 5 = 0) ∧ (n % 6 = 0) ∧ n = 990 :=
by 
  -- Needs to follow the procedures to prove it step-by-step
  sorry

end largest_multiple_5_6_lt_1000_is_990_l155_155543


namespace number_of_girls_in_school_l155_155926

-- Variables representing the population and the sample.
variables (total_students sample_size boys_sample girls_sample : ℕ)

-- Initial conditions.
def initial_conditions := 
  total_students = 1600 ∧ 
  sample_size = 200 ∧
  girls_sample = 90 ∧
  boys_sample = 110 ∧
  (girls_sample + 20 = boys_sample)

-- Statement to prove.
theorem number_of_girls_in_school (x: ℕ) 
  (h : initial_conditions total_students sample_size boys_sample girls_sample) :
  x = 720 :=
by {
  -- Obligatory proof omitted.
  sorry
}

end number_of_girls_in_school_l155_155926


namespace final_quantity_of_milk_l155_155890

-- Initially, a vessel is filled with 45 litres of pure milk
def initial_milk : Nat := 45

-- First operation: removing 9 litres of milk and replacing with water
def first_operation_milk(initial_milk : Nat) : Nat := initial_milk - 9
def first_operation_water : Nat := 9

-- Second operation: removing 9 litres of the mixture and replacing with water
def milk_fraction_mixture(milk : Nat) (total : Nat) : Rat := milk / total
def water_fraction_mixture(water : Nat) (total : Nat) : Rat := water / total

def second_operation_milk(milk : Nat) (total : Nat) (removed : Nat) : Rat := 
  milk - (milk_fraction_mixture milk total) * removed
def second_operation_water(water : Nat) (total : Nat) (removed : Nat) : Rat := 
  water - (water_fraction_mixture water total) * removed + removed

-- Prove the final quantity of milk
theorem final_quantity_of_milk : second_operation_milk 36 45 9 = 28.8 := by
  sorry

end final_quantity_of_milk_l155_155890


namespace total_songs_l155_155897

theorem total_songs (h : ℕ) (m : ℕ) (a : ℕ) (t : ℕ) (P : ℕ)
  (Hh : h = 6) (Hm : m = 3) (Ha : a = 5) 
  (Htotal : P = (h + m + a + t) / 3) 
  (Hdiv : (h + m + a + t) % 3 = 0) : P = 6 := by
  sorry

end total_songs_l155_155897


namespace orange_shells_correct_l155_155987

def total_shells : Nat := 65
def purple_shells : Nat := 13
def pink_shells : Nat := 8
def yellow_shells : Nat := 18
def blue_shells : Nat := 12
def orange_shells : Nat := total_shells - (purple_shells + pink_shells + yellow_shells + blue_shells)

theorem orange_shells_correct : orange_shells = 14 :=
by
  sorry

end orange_shells_correct_l155_155987


namespace number_of_cartons_of_pencils_l155_155843

theorem number_of_cartons_of_pencils (P E : ℕ) 
  (h1 : P + E = 100) 
  (h2 : 6 * P + 3 * E = 360) : 
  P = 20 := 
by
  sorry

end number_of_cartons_of_pencils_l155_155843


namespace joan_total_money_l155_155364

-- Define the number of each type of coin found
def dimes_jacket : ℕ := 15
def dimes_shorts : ℕ := 4
def nickels_shorts : ℕ := 7
def quarters_jeans : ℕ := 12
def pennies_jeans : ℕ := 2
def nickels_backpack : ℕ := 8
def pennies_backpack : ℕ := 23

-- Calculate the total number of each type of coin
def total_dimes : ℕ := dimes_jacket + dimes_shorts
def total_nickels : ℕ := nickels_shorts + nickels_backpack
def total_quarters : ℕ := quarters_jeans
def total_pennies : ℕ := pennies_jeans + pennies_backpack

-- Calculate the total value of each type of coin
def value_dimes : ℝ := total_dimes * 0.10
def value_nickels : ℝ := total_nickels * 0.05
def value_quarters : ℝ := total_quarters * 0.25
def value_pennies : ℝ := total_pennies * 0.01

-- Calculate the total amount of money found
def total_money : ℝ := value_dimes + value_nickels + value_quarters + value_pennies

-- Proof statement
theorem joan_total_money : total_money = 5.90 := by
  sorry

end joan_total_money_l155_155364


namespace zoo_sea_lions_l155_155233

variable (S P : ℕ)

theorem zoo_sea_lions (h1 : S / P = 4 / 11) (h2 : P = S + 84) : S = 48 := 
sorry

end zoo_sea_lions_l155_155233


namespace geometric_sequence_common_ratio_l155_155082

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * q)
  (h_arith : -a 5 + a 6 = 2 * a 4) :
  q = -1 ∨ q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l155_155082


namespace number_of_puppies_l155_155050

theorem number_of_puppies (P K : ℕ) (h1 : K = 2 * P + 14) (h2 : K = 78) : P = 32 :=
by sorry

end number_of_puppies_l155_155050


namespace q_at_2_l155_155306

noncomputable def q (x : ℝ) : ℝ :=
  Real.sign (3 * x - 2) * |3 * x - 2|^(1/4) +
  2 * Real.sign (3 * x - 2) * |3 * x - 2|^(1/6) +
  |3 * x - 2|^(1/8)

theorem q_at_2 : q 2 = 4 := by
  -- Proof attempt needed
  sorry

end q_at_2_l155_155306


namespace remainder_of_65_power_65_plus_65_mod_97_l155_155240

theorem remainder_of_65_power_65_plus_65_mod_97 :
  (65^65 + 65) % 97 = 33 :=
by
  sorry

end remainder_of_65_power_65_plus_65_mod_97_l155_155240


namespace interest_earned_l155_155471

theorem interest_earned :
  let P : ℝ := 1500
  let r : ℝ := 0.02
  let n : ℕ := 3
  let A : ℝ := P * (1 + r) ^ n
  let interest : ℝ := A - P
  interest = 92 := 
by
  sorry

end interest_earned_l155_155471


namespace unique_geometric_sequence_l155_155360

theorem unique_geometric_sequence (a : ℝ) (q : ℝ) (a_n b_n : ℕ → ℝ) 
    (h1 : a > 0) 
    (h2 : a_n 1 = a) 
    (h3 : b_n 1 - a_n 1 = 1) 
    (h4 : b_n 2 - a_n 2 = 2) 
    (h5 : b_n 3 - a_n 3 = 3) 
    (h6 : ∀ n, a_n (n + 1) = a_n n * q) 
    (h7 : ∀ n, b_n (n + 1) = b_n n * q) : 
    a = 1 / 3 := sorry

end unique_geometric_sequence_l155_155360


namespace minimum_boxes_needed_l155_155683

theorem minimum_boxes_needed (small_box_capacity medium_box_capacity large_box_capacity : ℕ)
    (max_small_boxes max_medium_boxes max_large_boxes : ℕ)
    (total_dozens: ℕ) :
  small_box_capacity = 2 → 
  medium_box_capacity = 3 → 
  large_box_capacity = 4 → 
  max_small_boxes = 6 → 
  max_medium_boxes = 5 → 
  max_large_boxes = 4 → 
  total_dozens = 40 → 
  ∃ (small_boxes_needed medium_boxes_needed large_boxes_needed : ℕ), 
    small_boxes_needed = 5 ∧ 
    medium_boxes_needed = 5 ∧ 
    large_boxes_needed = 4 := 
by
  sorry

end minimum_boxes_needed_l155_155683


namespace C_paisa_for_A_rupee_l155_155513

variable (A B C : ℝ)
variable (C_share : ℝ) (total_sum : ℝ)
variable (B_per_A : ℝ)

noncomputable def C_paisa_per_A_rupee (A B C C_share total_sum B_per_A : ℝ) : ℝ :=
  let C_paisa := C_share * 100
  C_paisa / A

theorem C_paisa_for_A_rupee : C_share = 32 ∧ total_sum = 164 ∧ B_per_A = 0.65 → 
  C_paisa_per_A_rupee A B C C_share total_sum B_per_A = 40 := by
  sorry

end C_paisa_for_A_rupee_l155_155513


namespace inequality_positive_reals_l155_155273

theorem inequality_positive_reals (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) :
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧ 
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 2 / 2) :=
sorry

end inequality_positive_reals_l155_155273


namespace dwarfs_truthful_count_l155_155089

theorem dwarfs_truthful_count (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) :
    x = 4 ∧ y = 6 :=
by
  sorry

end dwarfs_truthful_count_l155_155089


namespace order_b_gt_c_gt_a_l155_155084

noncomputable def a : ℝ := Real.log 2.6
def b : ℝ := 0.5 * 1.8^2
noncomputable def c : ℝ := 1.1^5

theorem order_b_gt_c_gt_a : b > c ∧ c > a := by
  sorry

end order_b_gt_c_gt_a_l155_155084


namespace parallel_lines_condition_l155_155937

variable {a : ℝ}

theorem parallel_lines_condition (a_is_2 : a = 2) :
  (∀ x y : ℝ, a * x + 2 * y = 0 → x + y = 1) ∧ (∀ x y : ℝ, x + y = 1 → a * x + 2 * y = 0) :=
by
  sorry

end parallel_lines_condition_l155_155937


namespace infinite_series_equals_l155_155685

noncomputable def infinite_series : Real :=
  ∑' n, if h : (n : ℕ) ≥ 2 then (n^4 + 2 * n^3 + 8 * n^2 + 8 * n + 8) / (2^n * (n^4 + 4)) else 0

theorem infinite_series_equals : infinite_series = 11 / 10 :=
  sorry

end infinite_series_equals_l155_155685


namespace centroid_plane_distance_l155_155040

theorem centroid_plane_distance :
  ∀ (α β γ : ℝ) (p q r : ℝ),
    (1 / α^2 + 1 / β^2 + 1 / γ^2 = 1 / 4) →
    (p = α / 3) →
    (q = β / 3) →
    (r = γ / 3) →
    (1 / p^2 + 1 / q^2 + 1 / r^2 = 9 / 4) :=
by sorry

end centroid_plane_distance_l155_155040


namespace sum_first_3m_terms_l155_155547

variable (m : ℕ) (a₁ d : ℕ)

def S (n : ℕ) := n * a₁ + (n * (n - 1)) / 2 * d

-- Given conditions
axiom sum_first_m_terms : S m = 0
axiom sum_first_2m_terms : S (2 * m) = 0

-- Theorem to be proved
theorem sum_first_3m_terms : S (3 * m) = 210 :=
by
  sorry

end sum_first_3m_terms_l155_155547


namespace total_distance_l155_155529

theorem total_distance (D : ℕ) 
  (h1 : (1 / 2 * D : ℝ) + (1 / 4 * (1 / 2 * D : ℝ)) + 105 = D) : 
  D = 280 :=
by
  sorry

end total_distance_l155_155529


namespace mean_equality_l155_155088

theorem mean_equality (y z : ℝ)
  (h : (14 + y + z) / 3 = (8 + 15 + 21) / 3)
  (hyz : y = z) :
  y = 15 ∧ z = 15 :=
by sorry

end mean_equality_l155_155088


namespace length_of_second_platform_is_correct_l155_155741

-- Define the constants
def lt : ℕ := 70  -- Length of the train
def l1 : ℕ := 170  -- Length of the first platform
def t1 : ℕ := 15  -- Time to cross the first platform
def t2 : ℕ := 20  -- Time to cross the second platform

-- Calculate the speed of the train
def v : ℕ := (lt + l1) / t1

-- Define the length of the second platform
def l2 : ℕ := 250

-- The proof statement
theorem length_of_second_platform_is_correct : lt + l2 = v * t2 := sorry

end length_of_second_platform_is_correct_l155_155741


namespace fraction_problem_l155_155783

-- Definitions given in the conditions
variables {p q r s : ℚ}
variables (h₁ : p / q = 8)
variables (h₂ : r / q = 5)
variables (h₃ : r / s = 3 / 4)

-- Statement to prove
theorem fraction_problem : s / p = 5 / 6 :=
by
  sorry

end fraction_problem_l155_155783


namespace ratio_x_y_l155_155232

theorem ratio_x_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : x / y = 13 / 2 :=
by
  sorry

end ratio_x_y_l155_155232


namespace inequality_proof_l155_155607

variable (a b c : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a + b + c + a * b + b * c + c * a + a * b * c = 7)

theorem inequality_proof : 
  (Real.sqrt (a ^ 2 + b ^ 2 + 2) + Real.sqrt (b ^ 2 + c ^ 2 + 2) + Real.sqrt (c ^ 2 + a ^ 2 + 2)) ≥ 6 := by
  sorry

end inequality_proof_l155_155607


namespace reassemble_into_square_conditions_l155_155854

noncomputable def graph_paper_figure : Type := sorry
noncomputable def is_cuttable_into_parts (figure : graph_paper_figure) (parts : ℕ) : Prop := sorry
noncomputable def all_parts_are_triangles (figure : graph_paper_figure) (parts : ℕ) : Prop := sorry
noncomputable def can_reassemble_to_square (figure : graph_paper_figure) : Prop := sorry

theorem reassemble_into_square_conditions :
  ∀ (figure : graph_paper_figure), 
  (is_cuttable_into_parts figure 4 ∧ can_reassemble_to_square figure) ∧ 
  (is_cuttable_into_parts figure 5 ∧ all_parts_are_triangles figure 5 ∧ can_reassemble_to_square figure) :=
sorry

end reassemble_into_square_conditions_l155_155854


namespace inequality_solution_set_l155_155468

open Set

noncomputable def rational_expression (x : ℝ) : ℝ := (x^2 - 16) / (x^2 + 10*x + 25)

theorem inequality_solution_set :
  {x : ℝ | rational_expression x < 0} = Ioo (-4 : ℝ) 4 :=
by
  sorry

end inequality_solution_set_l155_155468


namespace correct_letter_is_P_l155_155421

variable (x : ℤ)

-- Conditions
def date_behind_C := x
def date_behind_A := x + 2
def date_behind_B := x + 11
def date_behind_P := x + 13 -- Based on problem setup
def date_behind_Q := x + 14 -- Continuous sequence assumption
def date_behind_R := x + 15 -- Continuous sequence assumption
def date_behind_S := x + 16 -- Continuous sequence assumption

-- Proof statement
theorem correct_letter_is_P :
  ∃ y, (y = date_behind_P ∧ x + y = date_behind_A + date_behind_B) := by
  sorry

end correct_letter_is_P_l155_155421


namespace number_of_sandwiches_l155_155535

-- Defining the conditions
def kinds_of_meat := 12
def kinds_of_cheese := 11
def kinds_of_bread := 5

-- Combinations calculation
def choose_one (n : Nat) := n
def choose_three (n : Nat) := Nat.choose n 3

-- Proof statement to show that the total number of sandwiches is 9900
theorem number_of_sandwiches : (choose_one kinds_of_meat) * (choose_three kinds_of_cheese) * (choose_one kinds_of_bread) = 9900 := by
  sorry

end number_of_sandwiches_l155_155535


namespace product_modulo_25_l155_155192

theorem product_modulo_25 : 
  (123 ≡ 3 [MOD 25]) → 
  (456 ≡ 6 [MOD 25]) → 
  (789 ≡ 14 [MOD 25]) → 
  (123 * 456 * 789 ≡ 2 [MOD 25]) := 
by 
  intros h1 h2 h3 
  sorry

end product_modulo_25_l155_155192


namespace find_hourly_charge_l155_155895

variable {x : ℕ}

--Assumptions and conditions
def fixed_charge := 17
def total_paid := 80
def rental_hours := 9

-- Proof problem
theorem find_hourly_charge (h : fixed_charge + rental_hours * x = total_paid) : x = 7 :=
sorry

end find_hourly_charge_l155_155895


namespace solve_inequality_solution_set_l155_155999

def solution_set (x : ℝ) : Prop := -x^2 + 5 * x > 6

theorem solve_inequality_solution_set :
  { x : ℝ | solution_set x } = { x : ℝ | 2 < x ∧ x < 3 } :=
sorry

end solve_inequality_solution_set_l155_155999


namespace cos_C_eq_3_5_l155_155670

theorem cos_C_eq_3_5 (A B C : ℝ) (hABC : A^2 + B^2 = C^2) (hRight : B ^ 2 + C ^ 2 = A ^ 2) (hTan : B / C = 4 / 3) : B / A = 3 / 5 :=
by
  sorry

end cos_C_eq_3_5_l155_155670


namespace guesthouse_rolls_probability_l155_155812

theorem guesthouse_rolls_probability :
  let rolls := 12
  let guests := 3
  let types := 4
  let rolls_per_guest := 3
  let total_probability : ℚ := (12 / 12) * (9 / 11) * (6 / 10) * (3 / 9) *
                               (8 / 8) * (6 / 7) * (4 / 6) * (2 / 5) *
                               1
  let simplified_probability : ℚ := 24 / 1925
  total_probability = simplified_probability := sorry

end guesthouse_rolls_probability_l155_155812


namespace inequality_abc_ad_bc_bd_cd_l155_155482

theorem inequality_abc_ad_bc_bd_cd (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (1 / (a * b) + 1 / (a * c) + 1 / (a * d) + 1 / (b * c) + 1 / (b * d) + 1 / (c * d)) 
  ≤ (3 / 8) * (1 / a + 1 / b + 1 / c + 1 / d) ^ 2 := sorry

end inequality_abc_ad_bc_bd_cd_l155_155482


namespace average_value_of_series_l155_155906

theorem average_value_of_series (z : ℤ) :
  let series := [0^2, (2*z)^2, (4*z)^2, (8*z)^2]
  let sum_series := series.sum
  let n := series.length
  sum_series / n = 21 * z^2 :=
by
  let series := [0^2, (2*z)^2, (4*z)^2, (8*z)^2]
  let sum_series := series.sum
  let n := series.length
  sorry

end average_value_of_series_l155_155906


namespace part_i_part_ii_l155_155116

variable {b c : ℤ}

theorem part_i (hb : b ≠ 0) (hc : c ≠ 0) 
    (cond1 : ∃ m n : ℤ, m ≠ n ∧ (m + n = -b) ∧ (m * n = c))
    (cond2 : ∃ r s : ℤ, r ≠ s ∧ (r + s = -b) ∧ (r * s = -c)) :
    ∃ (p q : ℤ), p > 0 ∧ q > 0 ∧ p ≠ q ∧ 2 * b ^ 2 = p ^ 2 + q ^ 2 :=
sorry

theorem part_ii (hb : b ≠ 0) (hc : c ≠ 0) 
    (cond1 : ∃ m n : ℤ, m ≠ n ∧ (m + n = -b) ∧ (m * n = c))
    (cond2 : ∃ r s : ℤ, r ≠ s ∧ (r + s = -b) ∧ (r * s = -c)) :
    ∃ (r s : ℤ), r > 0 ∧ s > 0 ∧ r ≠ s ∧ b ^ 2 = r ^ 2 + s ^ 2 :=
sorry

end part_i_part_ii_l155_155116


namespace points_connected_l155_155765

theorem points_connected (m l : ℕ) (h1 : l < m) (h2 : Even (l * m)) :
  ∃ points : Finset (ℕ × ℕ), ∀ p ∈ points, (∃ q, q ∈ points ∧ (p ≠ q → p.snd = q.snd → p.fst = q.fst)) :=
sorry

end points_connected_l155_155765


namespace elements_in_M_l155_155422

def is_element_of_M (x y : ℕ) : Prop :=
  x + y ≤ 1

def M : Set (ℕ × ℕ) :=
  {p | is_element_of_M p.fst p.snd}

theorem elements_in_M :
  M = { (0,0), (0,1), (1,0) } :=
by
  -- Proof would go here
  sorry

end elements_in_M_l155_155422


namespace courtyard_length_l155_155100

/-- Given the following conditions:
  1. The width of the courtyard is 16.5 meters.
  2. 66 paving stones are required.
  3. Each paving stone measures 2.5 meters by 2 meters.
  Prove that the length of the rectangular courtyard is 20 meters. -/
theorem courtyard_length :
  ∃ L : ℝ, L = 20 ∧ 
           (∃ W : ℝ, W = 16.5) ∧ 
           (∃ n : ℕ, n = 66) ∧ 
           (∃ A : ℝ, A = 2.5 * 2) ∧
           n * A = L * W :=
by
  sorry

end courtyard_length_l155_155100


namespace cistern_length_l155_155439

variable (L : ℝ) (width water_depth total_area : ℝ)

theorem cistern_length
  (h_width : width = 8)
  (h_water_depth : water_depth = 1.5)
  (h_total_area : total_area = 134) :
  11 * L + 24 = total_area → L = 10 :=
by
  intro h_eq
  have h_eq1 : 11 * L = 110 := by
    linarith
  have h_L : L = 10 := by
    linarith
  exact h_L

end cistern_length_l155_155439


namespace james_profit_l155_155684

def cattle_profit (num_cattle : ℕ) (purchase_price total_feed_increase : ℝ)
    (weight_per_cattle : ℝ) (selling_price_per_pound : ℝ) : ℝ :=
  let feed_cost := purchase_price * (1 + total_feed_increase)
  let total_cost := purchase_price + feed_cost
  let revenue_per_cattle := weight_per_cattle * selling_price_per_pound
  let total_revenue := revenue_per_cattle * num_cattle
  total_revenue - total_cost

theorem james_profit : cattle_profit 100 40000 0.20 1000 2 = 112000 := by
  sorry

end james_profit_l155_155684


namespace calculate_expression_l155_155679

theorem calculate_expression :
  let s1 := 3 + 6 + 9
  let s2 := 4 + 8 + 12
  s1 = 18 → s2 = 24 → (s1 / s2 + s2 / s1) = 25 / 12 :=
by
  intros
  sorry

end calculate_expression_l155_155679


namespace isosceles_triangle_perimeter_l155_155516

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 6 * x + 8 = 0

-- Define the roots based on factorization of the given equation
def root1 := 2
def root2 := 4

-- Define the perimeter of the isosceles triangle given the roots
def triangle_perimeter := root2 + root2 + root1

-- Prove that the perimeter of the isosceles triangle is 10
theorem isosceles_triangle_perimeter : triangle_perimeter = 10 :=
by
  -- We need to verify the solution without providing the steps explicitly
  sorry

end isosceles_triangle_perimeter_l155_155516


namespace part1_part2_l155_155643

theorem part1 : (2 / 9 - 1 / 6 + 1 / 18) * (-18) = -2 := 
by
  sorry

theorem part2 : 54 * (3 / 4 + 1 / 2 - 1 / 4) = 54 := 
by
  sorry

end part1_part2_l155_155643


namespace BC_work_time_l155_155389

-- Definitions
def rateA : ℚ := 1 / 4 -- A's rate of work
def rateB : ℚ := 1 / 4 -- B's rate of work
def rateAC : ℚ := 1 / 3 -- A and C's combined rate of work

-- To prove
theorem BC_work_time : 1 / (rateB + (rateAC - rateA)) = 3 := by
  sorry

end BC_work_time_l155_155389


namespace geometric_sequence_min_n_l155_155604

theorem geometric_sequence_min_n (n : ℕ) (h : 2^(n + 1) - 2 - n > 1020) : n ≥ 10 :=
sorry

end geometric_sequence_min_n_l155_155604


namespace three_digit_number_l155_155712

theorem three_digit_number (a b c : ℕ) (h1 : a * (b + c) = 33) (h2 : b * (a + c) = 40) : 
  100 * a + 10 * b + c = 347 :=
by
  sorry

end three_digit_number_l155_155712


namespace kate_candy_l155_155187

variable (K : ℕ)
variable (R : ℕ) (B : ℕ) (M : ℕ)

-- Define the conditions
def robert_pieces := R = K + 2
def mary_pieces := M = R + 2
def bill_pieces := B = M - 6
def total_pieces := K + R + M + B = 20

-- The theorem to prove
theorem kate_candy :
  ∃ (K : ℕ), robert_pieces K R ∧ mary_pieces R M ∧ bill_pieces M B ∧ total_pieces K R M B ∧ K = 4 :=
sorry

end kate_candy_l155_155187


namespace base7_to_base10_l155_155020

theorem base7_to_base10 (a b : ℕ) (h : 235 = 49 * 2 + 7 * 3 + 5) (h_ab : 100 + 10 * a + b = 124) : 
  (a + b) / 7 = 6 / 7 :=
by
  sorry

end base7_to_base10_l155_155020


namespace james_total_room_area_l155_155009

theorem james_total_room_area :
  let original_length := 13
  let original_width := 18
  let increase := 2
  let new_length := original_length + increase
  let new_width := original_width + increase
  let area_of_one_room := new_length * new_width
  let number_of_rooms := 4
  let area_of_four_rooms := area_of_one_room * number_of_rooms
  let area_of_larger_room := area_of_one_room * 2
  let total_area := area_of_four_rooms + area_of_larger_room
  total_area = 1800
  := sorry

end james_total_room_area_l155_155009


namespace find_sum_of_x_and_reciprocal_l155_155671

theorem find_sum_of_x_and_reciprocal (x : ℝ) (hx_condition : x^3 + 1/x^3 = 110) : x + 1/x = 5 := 
sorry

end find_sum_of_x_and_reciprocal_l155_155671


namespace relationship_among_x_y_z_l155_155532

variable (a b c d : ℝ)

-- Conditions
variables (h1 : a < b)
variables (h2 : b < c)
variables (h3 : c < d)

-- Definitions of x, y, z
def x : ℝ := (a + b) * (c + d)
def y : ℝ := (a + c) * (b + d)
def z : ℝ := (a + d) * (b + c)

-- Theorem: Prove the relationship among x, y, z
theorem relationship_among_x_y_z (h1 : a < b) (h2 : b < c) (h3 : c < d) : x a b c d < y a b c d ∧ y a b c d < z a b c d := by
  sorry

end relationship_among_x_y_z_l155_155532


namespace vertices_of_square_l155_155856

-- Define lattice points as points with integer coordinates
structure LatticePoint where
  x : ℤ
  y : ℤ

-- Define the distance between two lattice points
def distance (P Q : LatticePoint) : ℤ :=
  (P.x - Q.x) * (P.x - Q.x) + (P.y - Q.y) * (P.y - Q.y)

-- Define the area of a triangle formed by three lattice points using the determinant method
def area (P Q R : LatticePoint) : ℤ :=
  (Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x)

-- Prove that three distinct lattice points form the vertices of a square given the condition
theorem vertices_of_square (P Q R : LatticePoint) (h₀ : P ≠ Q) (h₁ : Q ≠ R) (h₂ : P ≠ R)
    (h₃ : (distance P Q + distance Q R) < 8 * (area P Q R) + 1) :
    ∃ S : LatticePoint, S ≠ P ∧ S ≠ Q ∧ S ≠ R ∧
    (distance P Q = distance Q R ∧ distance Q R = distance R S ∧ distance R S = distance S P) := 
by sorry

end vertices_of_square_l155_155856


namespace mary_younger_than_albert_l155_155510

variable (A M B : ℕ)

noncomputable def albert_age := 4 * B
noncomputable def mary_age := A / 2
noncomputable def betty_age := 4

theorem mary_younger_than_albert (h1 : A = 2 * M) (h2 : A = 4 * 4) (h3 : 4 = 4) :
  A - M = 8 :=
sorry

end mary_younger_than_albert_l155_155510


namespace arc_length_sector_l155_155544

theorem arc_length_sector (r : ℝ) (θ : ℝ) (h_r : r = 3) (h_θ : θ = 150 * Real.pi / 180) :
  θ * r = 5 * Real.pi / 2 :=
by
  rw [h_r, h_θ]
  sorry

end arc_length_sector_l155_155544


namespace solve_for_x_l155_155091

theorem solve_for_x (x : ℕ) (h : 2^12 = 64^x) : x = 2 :=
by {
  sorry
}

end solve_for_x_l155_155091


namespace a4_value_l155_155774

axiom a_n : ℕ → ℝ
axiom S_n : ℕ → ℝ
axiom q : ℝ

-- Conditions
axiom a1_eq_1 : a_n 1 = 1
axiom S6_eq_4S3 : S_n 6 = 4 * S_n 3
axiom q_ne_1 : q ≠ 1

-- Arithmetic Sequence Sum Formula
axiom sum_formula : ∀ n, S_n n = (1 - q^n) / (1 - q)

-- nth-term Formula
axiom nth_term_formula : ∀ n, a_n n = a_n 1 * q^(n - 1)

-- Prove the value of the 4th term
theorem a4_value : a_n 4 = 3 := sorry

end a4_value_l155_155774


namespace total_flour_l155_155158

def cups_of_flour (flour_added : ℕ) (flour_needed : ℕ) : ℕ :=
  flour_added + flour_needed

theorem total_flour :
  ∀ (flour_added flour_needed : ℕ), flour_added = 3 → flour_needed = 6 → cups_of_flour flour_added flour_needed = 9 :=
by 
  intros flour_added flour_needed h_added h_needed
  rw [h_added, h_needed]
  rfl

end total_flour_l155_155158


namespace compute_expression_l155_155687

-- Define the conditions as specific values and operations within the theorem itself
theorem compute_expression : 5 + 7 * (2 - 9)^2 = 348 := 
  by
  sorry

end compute_expression_l155_155687


namespace length_PX_l155_155302

theorem length_PX (CX DP PW PX : ℕ) (hCX : CX = 60) (hDP : DP = 20) (hPW : PW = 40)
  (parallel_CD_WX : true)  -- We use a boolean to denote the parallel condition for simplicity
  (h1 : DP + PW = CX)  -- The sum of the segments from point C through P to point X
  (h2 : DP * 2 = PX)  -- The ratio condition derived from the similarity of triangles
  : PX = 40 := 
by
  -- using the given conditions and h2 to solve for PX
  sorry

end length_PX_l155_155302


namespace triangle_def_ef_value_l155_155573

theorem triangle_def_ef_value (E F D : ℝ) (DE DF EF : ℝ) (h1 : E = 45)
  (h2 : DE = 100) (h3 : DF = 100 * Real.sqrt 2) :
  EF = Real.sqrt (30000 + 5000*(Real.sqrt 6 - Real.sqrt 2)) := 
sorry 

end triangle_def_ef_value_l155_155573


namespace pricePerRedStamp_l155_155554

namespace StampCollection

-- Definitions for the conditions
def totalRedStamps : ℕ := 20
def soldRedStamps : ℕ := 20
def totalBlueStamps : ℕ := 80
def soldBlueStamps : ℕ := 80
def pricePerBlueStamp : ℝ := 0.8
def totalYellowStamps : ℕ := 7
def pricePerYellowStamp : ℝ := 2
def totalTargetEarnings : ℝ := 100

-- Derived definitions from conditions
def earningsFromBlueStamps : ℝ := soldBlueStamps * pricePerBlueStamp
def earningsFromYellowStamps : ℝ := totalYellowStamps * pricePerYellowStamp
def earningsRequiredFromRedStamps : ℝ := totalTargetEarnings - (earningsFromBlueStamps + earningsFromYellowStamps)

-- The statement asserting the main proof obligation
theorem pricePerRedStamp :
  (earningsRequiredFromRedStamps / soldRedStamps) = 1.10 :=
sorry

end StampCollection

end pricePerRedStamp_l155_155554


namespace part_a_part_b_part_c_l155_155165

-- The conditions for quadrilateral ABCD
variables (a b c d e f m n S : ℝ)
variables (S_nonneg : 0 ≤ S)

-- Prove Part (a)
theorem part_a (a b c d e f : ℝ) (S : ℝ) (h : S ≤ 1/4 * (e^2 + f^2)) : S <= 1/4 * (e^2 + f^2) :=
by 
  exact h

-- Prove Part (b)
theorem part_b (a b c d e f m n S: ℝ) (h : S ≤ 1/2 * (m^2 + n^2)) : S <= 1/2 * (m^2 + n^2) :=
by 
  exact h

-- Prove Part (c)
theorem part_c (a b c d e f m n S: ℝ) (h : S ≤ 1/4 * (a + c) * (b + d)) : S <= 1/4 * (a + c) * (b + d) :=
by 
  exact h

#eval "This Lean code defines the correctness statement of each part of the problem."

end part_a_part_b_part_c_l155_155165


namespace y_intercept_range_l155_155398

-- Define the points A and B
def pointA : ℝ × ℝ := (-1, -2)
def pointB : ℝ × ℝ := (2, 3)

-- We define the predicate for the line intersection condition
def line_intersects_segment (c : ℝ) : Prop :=
  let x_val_a := -1
  let y_val_a := -2
  let x_val_b := 2
  let y_val_b := 3
  -- Line equation at point A
  let eqn_a := x_val_a + y_val_a - c
  -- Line equation at point B
  let eqn_b := x_val_b + y_val_b - c
  -- We assert that the line must intersect the segment AB
  eqn_a ≤ 0 ∧ eqn_b ≥ 0 ∨ eqn_a ≥ 0 ∧ eqn_b ≤ 0

-- The main theorem to prove the range of c
theorem y_intercept_range : 
  ∃ c_min c_max : ℝ, c_min = -3 ∧ c_max = 5 ∧
  ∀ c, line_intersects_segment c ↔ c_min ≤ c ∧ c ≤ c_max :=
by
  existsi -3
  existsi 5
  sorry

end y_intercept_range_l155_155398


namespace trigonometric_identity_l155_155941

theorem trigonometric_identity :
  (1 / Real.cos (80 * (Real.pi / 180)) - Real.sqrt 3 / Real.sin (80 * (Real.pi / 180)) = 4) :=
by
  sorry

end trigonometric_identity_l155_155941


namespace point_P_position_l155_155900

variable {a b c d : ℝ}
variable (h1: a ≠ b) (h2: a ≠ c) (h3: a ≠ d) (h4: b ≠ c) (h5: b ≠ d) (h6: c ≠ d)

theorem point_P_position (P : ℝ) (hP: b < P ∧ P < c) (hRatio: (|a - P| / |P - d|) = (|b - P| / |P - c|)) : 
  P = (a * c - b * d) / (a - b + c - d) := 
by
  sorry

end point_P_position_l155_155900


namespace percentage_of_men_attended_picnic_l155_155190

variable (E : ℝ) (W M P : ℝ)
variable (H1 : M = 0.5 * E)
variable (H2 : W = 0.5 * E)
variable (H3 : 0.4 * W = 0.2 * E)
variable (H4 : 0.3 * E = P * M + 0.2 * E)

theorem percentage_of_men_attended_picnic : P = 0.2 :=
by sorry

end percentage_of_men_attended_picnic_l155_155190


namespace jordan_running_time_l155_155566

-- Define the conditions given in the problem
variables (time_steve : ℕ) (distance_steve distance_jordan_1 distance_jordan_2 distance_jordan_3 : ℕ)

-- Assign the known values
axiom time_steve_def : time_steve = 24
axiom distance_steve_def : distance_steve = 3
axiom distance_jordan_1_def : distance_jordan_1 = 2
axiom distance_jordan_2_def : distance_jordan_2 = 1
axiom distance_jordan_3_def : distance_jordan_3 = 5

axiom half_time_condition : ∀ t_2, t_2 = time_steve / 2

-- The proof problem
theorem jordan_running_time : ∀ t_j1 t_j2 t_j3, 
  (t_j1 = time_steve / 2 ∧ 
   t_j2 = t_j1 / 2 ∧ 
   t_j3 = t_j2 * 5) →
  t_j3 = 30 := 
by
  intros t_j1 t_j2 t_j3 h
  sorry

end jordan_running_time_l155_155566


namespace impossible_piles_of_three_l155_155539

theorem impossible_piles_of_three (n : ℕ) (h1 : n = 1001)
  (h2 : ∀ p : ℕ, p > 1 → ∃ a b : ℕ, a + b = p - 1 ∧ a ≤ b) : 
  ¬ (∃ piles : List ℕ, ∀ pile ∈ piles, pile = 3 ∧ (piles.sum = n + piles.length)) :=
by
  sorry

end impossible_piles_of_three_l155_155539


namespace people_left_line_l155_155749

-- Definitions based on the conditions given in the problem
def initial_people := 7
def new_people := 8
def final_people := 11

-- Proof statement
theorem people_left_line (L : ℕ) (h : 7 - L + 8 = 11) : L = 4 :=
by
  -- Adding the proof steps directly skips to the required proof
  sorry

end people_left_line_l155_155749


namespace selling_price_40_percent_profit_l155_155253

variable (C L : ℝ)

-- Condition: the profit earned by selling at $832 is equal to the loss incurred when selling at some price "L".
axiom eq_profit_loss : 832 - C = C - L

-- Condition: the desired profit price for a 40% profit on the cost price is $896.
axiom forty_percent_profit : 1.40 * C = 896

-- Theorem: the selling price for making a 40% profit is $896.
theorem selling_price_40_percent_profit : 1.40 * C = 896 :=
by
  sorry

end selling_price_40_percent_profit_l155_155253


namespace find_number_l155_155087

theorem find_number (x : ℝ) (h : 75 = 0.6 * x) : x = 125 :=
sorry

end find_number_l155_155087


namespace second_group_persons_l155_155168

open Nat

theorem second_group_persons
  (P : ℕ)
  (work_first_group : 39 * 24 * 5 = 4680)
  (work_second_group : P * 26 * 6 = 4680) :
  P = 30 :=
by
  sorry

end second_group_persons_l155_155168


namespace intersection_of_A_and_B_l155_155688

-- Definitions of sets A and B based on the conditions
def A : Set ℝ := {x | 0 < x}
def B : Set ℝ := {0, 1, 2}

-- Theorem statement to prove A ∩ B = {1, 2}
theorem intersection_of_A_and_B : A ∩ B = {1, 2} := 
  sorry

end intersection_of_A_and_B_l155_155688


namespace mass_percentage_C_in_CuCO3_l155_155640

def molar_mass_Cu := 63.546 -- g/mol
def molar_mass_C := 12.011 -- g/mol
def molar_mass_O := 15.999 -- g/mol
def molar_mass_CuCO3 := molar_mass_Cu + molar_mass_C + 3 * molar_mass_O

theorem mass_percentage_C_in_CuCO3 : 
  (molar_mass_C / molar_mass_CuCO3) * 100 = 9.72 :=
by
  sorry

end mass_percentage_C_in_CuCO3_l155_155640


namespace sin_theta_plus_2cos_theta_eq_zero_l155_155940

theorem sin_theta_plus_2cos_theta_eq_zero (θ : ℝ) (h : Real.sin θ + 2 * Real.cos θ = 0) :
  (1 + Real.sin (2 * θ)) / (Real.cos θ)^2 = 1 :=
  sorry

end sin_theta_plus_2cos_theta_eq_zero_l155_155940


namespace members_count_l155_155127

theorem members_count (n : ℕ) (h : n * n = 2025) : n = 45 :=
sorry

end members_count_l155_155127


namespace thickness_of_wall_l155_155099

theorem thickness_of_wall 
    (brick_length cm : ℝ)
    (brick_width cm : ℝ)
    (brick_height cm : ℝ)
    (num_bricks : ℝ)
    (wall_length cm : ℝ)
    (wall_height cm : ℝ)
    (wall_thickness cm : ℝ) :
    brick_length = 25 → 
    brick_width = 11.25 → 
    brick_height = 6 →
    num_bricks = 7200 → 
    wall_length = 900 → 
    wall_height = 600 →
    wall_length * wall_height * wall_thickness = num_bricks * (brick_length * brick_width * brick_height) →
    wall_thickness = 22.5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end thickness_of_wall_l155_155099


namespace cosine_of_angle_in_second_quadrant_l155_155885

theorem cosine_of_angle_in_second_quadrant
  (α : ℝ)
  (h1 : Real.sin α = 1 / 3)
  (h2 : π / 2 < α ∧ α < π) :
  Real.cos α = - (2 * Real.sqrt 2) / 3 :=
by
  sorry

end cosine_of_angle_in_second_quadrant_l155_155885


namespace CapeMay_more_than_twice_Daytona_l155_155142

def Daytona_sharks : ℕ := 12
def CapeMay_sharks : ℕ := 32

theorem CapeMay_more_than_twice_Daytona : CapeMay_sharks - 2 * Daytona_sharks = 8 := by
  sorry

end CapeMay_more_than_twice_Daytona_l155_155142


namespace eqn_solution_set_l155_155188

theorem eqn_solution_set :
  {x : ℝ | x ^ 2 - 1 = 0} = {-1, 1} := 
sorry

end eqn_solution_set_l155_155188


namespace max_value_of_expression_l155_155969

theorem max_value_of_expression (x y z : ℝ) (h : x^2 + y^2 = z^2) :
  ∃ t, t = (3 * Real.sqrt 2) / 2 ∧ ∀ u, u = (x + 2 * y) / z → u ≤ t := by
  sorry

end max_value_of_expression_l155_155969


namespace solve_for_a_l155_155733

theorem solve_for_a (a : ℚ) (h : 2 * a - 3 = 5 - a) : a = 8 / 3 :=
by
  sorry

end solve_for_a_l155_155733


namespace regular_bike_wheels_eq_two_l155_155678

-- Conditions
def regular_bikes : ℕ := 7
def childrens_bikes : ℕ := 11
def wheels_per_childrens_bike : ℕ := 4
def total_wheels_seen : ℕ := 58

-- Define the problem
theorem regular_bike_wheels_eq_two 
  (w : ℕ)
  (h1 : total_wheels_seen = regular_bikes * w + childrens_bikes * wheels_per_childrens_bike) :
  w = 2 :=
by
  -- Proof steps would go here
  sorry

end regular_bike_wheels_eq_two_l155_155678


namespace total_hours_worked_l155_155058

def hours_day1 : ℝ := 2.5
def increment_day2 : ℝ := 0.5
def hours_day2 : ℝ := hours_day1 + increment_day2
def hours_day3 : ℝ := 3.75

theorem total_hours_worked :
  hours_day1 + hours_day2 + hours_day3 = 9.25 :=
sorry

end total_hours_worked_l155_155058


namespace exists_pos_integers_l155_155779

theorem exists_pos_integers (r : ℚ) (hr : r > 0) : 
  ∃ a b c d : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ r = (a^3 + b^3) / (c^3 + d^3) :=
by sorry

end exists_pos_integers_l155_155779


namespace inequality_transitive_l155_155975

theorem inequality_transitive {a b c d : ℝ} (h1 : a > b) (h2 : c > d) : a - d > b - c := 
by
  sorry

end inequality_transitive_l155_155975


namespace volume_of_orange_concentrate_l155_155763

theorem volume_of_orange_concentrate
  (h_jug : ℝ := 8) -- height of the jug in inches
  (d_jug : ℝ := 3) -- diameter of the jug in inches
  (fraction_full : ℝ := 3 / 4) -- jug is three-quarters full
  (ratio_concentrate_to_water : ℝ := 1 / 5) -- ratio of concentrate to water
  : abs ((fraction_full * π * ((d_jug / 2)^2) * h_jug * (1 / (1 + ratio_concentrate_to_water))) - 2.25) < 0.01 :=
by
  sorry

end volume_of_orange_concentrate_l155_155763


namespace custom_op_evaluation_l155_155705

def custom_op (a b : ℤ) : ℤ := a * b - (a + b)

theorem custom_op_evaluation : custom_op 2 (-3) = -5 :=
by
sorry

end custom_op_evaluation_l155_155705


namespace condo_total_units_l155_155633

-- Definitions from conditions
def total_floors := 23
def regular_units_per_floor := 12
def penthouse_units_per_floor := 2
def penthouse_floors := 2
def regular_floors := total_floors - penthouse_floors

-- Definition for total units
def total_units := (regular_floors * regular_units_per_floor) + (penthouse_floors * penthouse_units_per_floor)

-- Theorem statement: prove total units is 256
theorem condo_total_units : total_units = 256 :=
by
  sorry

end condo_total_units_l155_155633


namespace sunset_duration_l155_155817

theorem sunset_duration (changes : ℕ) (interval : ℕ) (total_changes : ℕ) (h1 : total_changes = 12) (h2 : interval = 10) : ∃ hours : ℕ, hours = 2 :=
by
  sorry

end sunset_duration_l155_155817


namespace subtraction_from_double_result_l155_155537

theorem subtraction_from_double_result (x : ℕ) (h : 2 * x = 18) : x - 4 = 5 :=
sorry

end subtraction_from_double_result_l155_155537


namespace amount_decreased_is_5_l155_155391

noncomputable def x : ℕ := 50
noncomputable def equation (x y : ℕ) : Prop := (1 / 5) * x - y = 5

theorem amount_decreased_is_5 : ∃ y : ℕ, equation x y ∧ y = 5 :=
by
  sorry

end amount_decreased_is_5_l155_155391


namespace fraction_simplification_l155_155825

theorem fraction_simplification :
  (3 / 7 + 4 / 5) / (5 / 12 + 2 / 3) = 516 / 455 := by
  sorry

end fraction_simplification_l155_155825


namespace relationship_between_x_and_y_l155_155132

variable (u : ℝ)

theorem relationship_between_x_and_y (h : u > 0) (hx : x = (u + 1)^(1 / u)) (hy : y = (u + 1)^((u + 1) / u)) :
  y^x = x^y :=
by
  sorry

end relationship_between_x_and_y_l155_155132


namespace find_second_number_l155_155731

theorem find_second_number (x : ℝ) : 217 + x + 0.217 + 2.0017 = 221.2357 → x = 2.017 :=
by
  sorry

end find_second_number_l155_155731


namespace sufficient_but_not_necessary_condition_l155_155383

variable (x : ℝ)

theorem sufficient_but_not_necessary_condition (h : x > 1) : x > 0 :=
by
  sorry

end sufficient_but_not_necessary_condition_l155_155383


namespace current_selling_price_is_correct_profit_per_unit_is_correct_l155_155277

variable (a : ℝ)

def original_selling_price (a : ℝ) : ℝ :=
  a * 1.22

def current_selling_price (a : ℝ) : ℝ :=
  original_selling_price a * 0.85

def profit_per_unit (a : ℝ) : ℝ :=
  current_selling_price a - a

theorem current_selling_price_is_correct : current_selling_price a = 1.037 * a :=
by
  unfold current_selling_price original_selling_price
  sorry

theorem profit_per_unit_is_correct : profit_per_unit a = 0.037 * a :=
by
  unfold profit_per_unit current_selling_price original_selling_price
  sorry

end current_selling_price_is_correct_profit_per_unit_is_correct_l155_155277


namespace rectangle_y_value_l155_155060

theorem rectangle_y_value
  (E : (ℝ × ℝ)) (F : (ℝ × ℝ)) (G : (ℝ × ℝ)) (H : (ℝ × ℝ))
  (hE : E = (0, 0)) (hF : F = (0, 5)) (hG : ∃ y : ℝ, G = (y, 5))
  (hH : ∃ y : ℝ, H = (y, 0)) (area : ℝ) (h_area : area = 35)
  (hy_pos : ∃ y : ℝ, y > 0)
  : ∃ y : ℝ, y = 7 :=
by
  sorry

end rectangle_y_value_l155_155060


namespace find_clique_of_size_6_l155_155317

-- Defining the conditions of the graph G
variable (G : SimpleGraph (Fin 12))

-- Condition: For any subset of 9 vertices, there exists a subset of 5 vertices that form a complete subgraph K_5.
def condition (s : Finset (Fin 12)) : Prop :=
  s.card = 9 → ∃ t : Finset (Fin 12), t ⊆ s ∧ t.card = 5 ∧ (∀ u v : Fin 12, u ∈ t → v ∈ t → u ≠ v → G.Adj u v)

-- The theorem to prove given the conditions
theorem find_clique_of_size_6 (h : ∀ s : Finset (Fin 12), condition G s) : 
  ∃ t : Finset (Fin 12), t.card = 6 ∧ (∀ u v : Fin 12, u ∈ t → v ∈ t → u ≠ v → G.Adj u v) :=
sorry

end find_clique_of_size_6_l155_155317


namespace find_f_2018_l155_155023

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom f_zero : f 0 = -1
axiom functional_equation (x : ℝ) : f x = -f (2 - x)

theorem find_f_2018 : f 2018 = 1 := 
by 
  sorry

end find_f_2018_l155_155023


namespace increase_corrosion_with_more_active_metal_rivets_l155_155601

-- Definitions representing conditions
def corrosion_inhibitor (P : Type) : Prop := true
def more_active_metal_rivets (P : Type) : Prop := true
def less_active_metal_rivets (P : Type) : Prop := true
def painted_parts (P : Type) : Prop := true

-- Main theorem statement
theorem increase_corrosion_with_more_active_metal_rivets (P : Type) 
  (h1 : corrosion_inhibitor P)
  (h2 : more_active_metal_rivets P)
  (h3 : less_active_metal_rivets P)
  (h4 : painted_parts P) : 
  more_active_metal_rivets P :=
by {
  -- proof goes here
  sorry
}

end increase_corrosion_with_more_active_metal_rivets_l155_155601


namespace marco_paints_8_15_in_32_minutes_l155_155085

-- Define the rates at which Marco and Carla paint
def marco_rate : ℚ := 1 / 60
def combined_rate : ℚ := 1 / 40

-- Define the function to calculate the fraction of the room painted by Marco alone in a given time
def fraction_painted_by_marco (time: ℚ) : ℚ := time * marco_rate

-- State the theorem to prove
theorem marco_paints_8_15_in_32_minutes :
  (marco_rate + (combined_rate - marco_rate) = combined_rate) →
  fraction_painted_by_marco 32 = 8 / 15 := by
  sorry

end marco_paints_8_15_in_32_minutes_l155_155085


namespace find_largest_n_l155_155286

theorem find_largest_n : ∃ n x y z : ℕ, n > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 
  ∧ n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 6
  ∧ (∀ m x' y' z' : ℕ, m > n → x' > 0 → y' > 0 → z' > 0 
  → m^2 ≠ x'^2 + y'^2 + z'^2 + 2*x'*y' + 2*y'*z' + 2*z'*x' + 3*x' + 3*y' + 3*z' - 6) :=
sorry

end find_largest_n_l155_155286


namespace not_all_zero_implies_at_least_one_nonzero_l155_155143

variable {a b c : ℤ}

theorem not_all_zero_implies_at_least_one_nonzero (h : ¬ (a = 0 ∧ b = 0 ∧ c = 0)) : 
  a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 := 
by 
  sorry

end not_all_zero_implies_at_least_one_nonzero_l155_155143


namespace convex_functions_exist_l155_155246

noncomputable def exponential_function (x : ℝ) : ℝ :=
  4 - 5 * (1 / 2) ^ x

noncomputable def inverse_tangent_function (x : ℝ) : ℝ :=
  (10 / Real.pi) * Real.arctan x - 1

theorem convex_functions_exist :
  ∃ (f1 f2 : ℝ → ℝ),
    (∀ x, 0 < x → f1 x = exponential_function x) ∧
    (∀ x, 0 < x → f2 x = inverse_tangent_function x) ∧
    (∀ x, 0 < x → f1 x ∈ Set.Ioo (-1 : ℝ) 4) ∧
    (∀ x, 0 < x → f2 x ∈ Set.Ioo (-1 : ℝ) 4) ∧
    (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 →
      f1 x1 + f1 x2 < 2 * f1 ((x1 + x2) / 2)) ∧
    (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 →
      f2 x1 + f2 x2 < 2 * f2 ((x1 + x2) / 2)) :=
sorry

end convex_functions_exist_l155_155246


namespace number_of_girls_l155_155199

theorem number_of_girls (n : ℕ) (A : ℝ) 
    (h1 : A = (n * (A + 1) + 55 - 80) / n) : n = 25 :=
by 
  sorry

end number_of_girls_l155_155199


namespace borgnine_lizards_l155_155609

theorem borgnine_lizards (chimps lions tarantulas total_legs : ℕ) (legs_per_chimp legs_per_lion legs_per_tarantula legs_per_lizard lizards : ℕ)
  (H_chimps : chimps = 12)
  (H_lions : lions = 8)
  (H_tarantulas : tarantulas = 125)
  (H_total_legs : total_legs = 1100)
  (H_legs_per_chimp : legs_per_chimp = 4)
  (H_legs_per_lion : legs_per_lion = 4)
  (H_legs_per_tarantula : legs_per_tarantula = 8)
  (H_legs_per_lizard : legs_per_lizard = 4)
  (H_seen_legs : total_legs = (chimps * legs_per_chimp) + (lions * legs_per_lion) + (tarantulas * legs_per_tarantula) + (lizards * legs_per_lizard)) :
  lizards = 5 := 
by
  sorry

end borgnine_lizards_l155_155609


namespace not_square_or_cube_l155_155799

theorem not_square_or_cube (n : ℕ) (h : n > 1) : 
  ¬ (∃ a : ℕ, 2^n - 1 = a^2) ∧ ¬ (∃ a : ℕ, 2^n - 1 = a^3) :=
by
  sorry

end not_square_or_cube_l155_155799


namespace rug_area_correct_l155_155512

def floor_length : ℕ := 10
def floor_width : ℕ := 8
def strip_width : ℕ := 2

def adjusted_length : ℕ := floor_length - 2 * strip_width
def adjusted_width : ℕ := floor_width - 2 * strip_width

def area_floor : ℕ := floor_length * floor_width
def area_rug : ℕ := adjusted_length * adjusted_width

theorem rug_area_correct : area_rug = 24 := by
  sorry

end rug_area_correct_l155_155512


namespace donation_fifth_sixth_l155_155558

-- Conditions definitions
def total_donation := 10000
def first_home := 2750
def second_home := 1945
def third_home := 1275
def fourth_home := 1890

-- Proof statement
theorem donation_fifth_sixth : 
  (total_donation - (first_home + second_home + third_home + fourth_home)) = 2140 := by
  sorry

end donation_fifth_sixth_l155_155558


namespace closest_multiple_of_12_l155_155768

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

-- Define the closest multiple of 4 to 2050 (2048 and 2052)
def closest_multiple_of_4 (n m : ℕ) : ℕ :=
if n % 4 < m % 4 then n - (n % 4)
else m + (4 - (m % 4))

-- Define the conditions for being divisible by both 3 and 4
def is_multiple_of_12 (n : ℕ) : Prop := is_multiple_of n 12

-- Theorem statement
theorem closest_multiple_of_12 (n m : ℕ) (h : n = 2050) (hm : m = 2052) :
  is_multiple_of_12 m :=
sorry

end closest_multiple_of_12_l155_155768


namespace solution_inequality_1_solution_inequality_2_l155_155845

theorem solution_inequality_1 (x : ℝ) : -x^2 + 4*x + 5 < 0 ↔ (x < -1 ∨ x > 5) :=
by sorry

theorem solution_inequality_2 (x : ℝ) : 2*x^2 - 5*x + 2 ≤ 0 ↔ (1/2 ≤ x ∧ x ≤ 2) :=
by sorry

end solution_inequality_1_solution_inequality_2_l155_155845


namespace car_parking_arrangements_l155_155123

theorem car_parking_arrangements : 
  let grid_size := 6
  let group_size := 3
  let red_car_positions := Nat.choose grid_size group_size
  let arrange_black_cars := Nat.factorial grid_size
  (red_car_positions * arrange_black_cars) = 14400 := 
by
  let grid_size := 6
  let group_size := 3
  let red_car_positions := Nat.choose grid_size group_size
  let arrange_black_cars := Nat.factorial grid_size
  sorry

end car_parking_arrangements_l155_155123


namespace rooms_count_l155_155728

theorem rooms_count (total_paintings : ℕ) (paintings_per_room : ℕ) (h1 : total_paintings = 32) (h2 : paintings_per_room = 8) : (total_paintings / paintings_per_room) = 4 := by
  sorry

end rooms_count_l155_155728


namespace total_distance_covered_l155_155504

def teams_data : List (String × Nat × Nat) :=
  [("Green Bay High", 5, 150), 
   ("Blue Ridge Middle", 7, 200),
   ("Sunset Valley Elementary", 4, 100),
   ("Riverbend Prep", 6, 250)]

theorem total_distance_covered (team : String) (members relays : Nat) :
  (team, members, relays) ∈ teams_data →
    (team = "Green Bay High" → members * relays = 750) ∧
    (team = "Blue Ridge Middle" → members * relays = 1400) ∧
    (team = "Sunset Valley Elementary" → members * relays = 400) ∧
    (team = "Riverbend Prep" → members * relays = 1500) :=
  by
    intros; sorry -- Proof omitted

end total_distance_covered_l155_155504


namespace find_x_find_a_l155_155663

-- Definitions based on conditions
def inversely_proportional (p q : ℕ) (k : ℕ) := p * q = k

-- Given conditions for (x, y)
def x1 : ℕ := 36
def y1 : ℕ := 4
def k1 : ℕ := x1 * y1 -- or 144
def y2 : ℕ := 9

-- Given conditions for (a, b)
def a1 : ℕ := 50
def b1 : ℕ := 5
def k2 : ℕ := a1 * b1 -- or 250
def b2 : ℕ := 10

-- Proof statements
theorem find_x (x : ℕ) : inversely_proportional x y2 k1 → x = 16 := by
  sorry

theorem find_a (a : ℕ) : inversely_proportional a b2 k2 → a = 25 := by
  sorry

end find_x_find_a_l155_155663


namespace brian_final_cards_l155_155144

-- Definitions of initial conditions
def initial_cards : ℕ := 76
def cards_taken : ℕ := 59
def packs_bought : ℕ := 3
def cards_per_pack : ℕ := 15

-- The proof problem: Prove that the final number of cards is 62
theorem brian_final_cards : initial_cards - cards_taken + packs_bought * cards_per_pack = 62 :=
by
  -- Proof goes here, 'sorry' used to skip actual proof
  sorry

end brian_final_cards_l155_155144


namespace value_range_of_f_l155_155899

noncomputable def f (x : ℝ) : ℝ := 4 / (x - 2)

theorem value_range_of_f : Set.range (fun x => f x) ∩ Set.Icc 3 6 = Set.Icc 1 4 :=
by
  sorry

end value_range_of_f_l155_155899


namespace A_holds_15_l155_155785

def cards : List (ℕ × ℕ) := [(1, 3), (1, 5), (3, 5)]

variables (A_card B_card C_card : ℕ × ℕ)

-- Conditions from the problem
def C_not_35 : Prop := C_card ≠ (3, 5)
def A_says_not_3 (A_card B_card : ℕ × ℕ) : Prop := ¬(A_card.1 = 3 ∧ B_card.1 = 3 ∨ A_card.2 = 3 ∧ B_card.2 = 3)
def B_says_not_1 (B_card C_card : ℕ × ℕ) : Prop := ¬(B_card.1 = 1 ∧ C_card.1 = 1 ∨ B_card.2 = 1 ∧ C_card.2 = 1)

-- Question to prove
theorem A_holds_15 : 
  ∃ (A_card B_card C_card : ℕ × ℕ),
    A_card ∈ cards ∧ B_card ∈ cards ∧ C_card ∈ cards ∧
    A_card ≠ B_card ∧ B_card ≠ C_card ∧ A_card ≠ C_card ∧
    C_not_35 C_card ∧
    A_says_not_3 A_card B_card ∧
    B_says_not_1 B_card C_card ->
    A_card = (1, 5) :=
sorry

end A_holds_15_l155_155785


namespace arithmetic_progression_common_difference_and_first_terms_l155_155959

def sum (n : ℕ) : ℕ := 5 * n ^ 2
def Sn (a1 d n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_progression_common_difference_and_first_terms:
  ∀ n : ℕ, Sn 5 10 n = sum n :=
by
  sorry

end arithmetic_progression_common_difference_and_first_terms_l155_155959


namespace sufficient_but_not_necessary_condition_ellipse_l155_155574

theorem sufficient_but_not_necessary_condition_ellipse (a : ℝ) :
  (a^2 > 1 → ∀ x y : ℝ, (x^2 / a^2 + y^2 = 1 → a^2 > 1)) ∧
  (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1 → (a^2 > 1 ∨ 0 < a^2 ∧ a^2 < 1)) → ¬ (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1 → a^2 > 1))) :=
by
  sorry

end sufficient_but_not_necessary_condition_ellipse_l155_155574


namespace simplify_expression_l155_155097

theorem simplify_expression : 
  (1 / (64^(1/3))^9) * 8^6 = 1 := by 
  have h1 : 64 = 2^6 := by rfl
  have h2 : 8 = 2^3 := by rfl
  sorry

end simplify_expression_l155_155097


namespace twelfth_term_arithmetic_sequence_l155_155956

theorem twelfth_term_arithmetic_sequence :
  let a := (1 : ℚ) / 2
  let d := (1 : ℚ) / 3
  (a + 11 * d) = (25 : ℚ) / 6 :=
by
  sorry

end twelfth_term_arithmetic_sequence_l155_155956
