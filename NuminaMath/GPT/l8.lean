import Mathlib

namespace simplify_expression_l8_8994

variable (a b : ℝ)

theorem simplify_expression :
  3 * a * (3 * a^3 + 2 * a^2) - 2 * a^2 * (b^2 + 1) = 9 * a^4 + 6 * a^3 - 2 * a^2 * b^2 - 2 * a^2 :=
by
  sorry

end simplify_expression_l8_8994


namespace power_function_decreasing_l8_8765

theorem power_function_decreasing (m : ℝ) (f : ℝ → ℝ)
  (h : ∀ x : ℝ, 0 < x → f x = (m^2 + m - 11) * x^(m - 1))
  (hm : m^2 + m - 11 > 0)
  (hm' : m - 1 < 0)
  (hx : 0 < 1):
  f (-1) = -1 := by 
sorry

end power_function_decreasing_l8_8765


namespace smallest_prime_divisor_of_sum_l8_8890

theorem smallest_prime_divisor_of_sum (a b : ℕ) (h1 : a = 3^19) (h2 : b = 11^13) (h3 : Odd a) (h4 : Odd b) : 
  ∃ p : ℕ, Prime p ∧ p ∣ (a + b) ∧ p = 2 := 
by {
  sorry
}

end smallest_prime_divisor_of_sum_l8_8890


namespace circle_diameter_in_feet_l8_8637

/-- Given: The area of a circle is 25 * pi square inches.
    Prove: The diameter of the circle in feet is 5/6 feet. -/
theorem circle_diameter_in_feet (A : ℝ) (hA : A = 25 * Real.pi) :
  ∃ d : ℝ, d = (5 / 6) :=
by
  -- The proof goes here
  sorry

end circle_diameter_in_feet_l8_8637


namespace sum_of_a_and_b_l8_8457

noncomputable def f (x : Real) : Real := (1 + Real.sin (2 * x)) / 2
noncomputable def a : Real := f (Real.log 5)
noncomputable def b : Real := f (Real.log (1 / 5))

theorem sum_of_a_and_b : a + b = 1 := by
  -- proof to be provided
  sorry

end sum_of_a_and_b_l8_8457


namespace sequence_value_l8_8932

theorem sequence_value (a b c d x : ℕ) (h1 : a = 5) (h2 : b = 9) (h3 : c = 17) (h4 : d = 33)
  (h5 : b - a = 4) (h6 : c - b = 8) (h7 : d - c = 16) (h8 : x - d = 32) : x = 65 := by
  sorry

end sequence_value_l8_8932


namespace fruit_bowl_l8_8638

variable {A P B : ℕ}

theorem fruit_bowl : (P = A + 2) → (B = P + 3) → (A + P + B = 19) → B = 9 :=
by
  intros h1 h2 h3
  sorry

end fruit_bowl_l8_8638


namespace probability_percussion_instruments_l8_8973

-- Define sounds set and classifications
def sounds_set := {"metal", "stone", "wood", "earth", "bamboo", "silk"}
def percussion_set := {"metal", "stone", "wood"}

-- Define a function to calculate binomial coefficient
def binom : ℕ → ℕ → ℕ
| n, 0     := 1
| 0, k     := 0
| n+1, k+1 := binom n k + binom n (k+1)

-- Define the Lean statement for the proof
theorem probability_percussion_instruments :
  let total_combinations := binom 6 2 in
  let favorable_outcomes := binom 3 2 in
  (favorable_outcomes : ℚ) / (total_combinations : ℚ) = 1 / 5 :=
by
  sorry

end probability_percussion_instruments_l8_8973


namespace eggs_per_basket_l8_8201

theorem eggs_per_basket (red_eggs : ℕ) (orange_eggs : ℕ) (min_eggs : ℕ) :
  red_eggs = 30 → orange_eggs = 45 → min_eggs = 5 →
  (∃ k, (30 % k = 0) ∧ (45 % k = 0) ∧ (k ≥ 5) ∧ k = 15) :=
by
  intros h1 h2 h3
  use 15
  sorry

end eggs_per_basket_l8_8201


namespace figure_side_length_l8_8605

theorem figure_side_length (number_of_sides : ℕ) (perimeter : ℝ) (length_of_one_side : ℝ) 
  (h1 : number_of_sides = 8) (h2 : perimeter = 23.6) : length_of_one_side = 2.95 :=
by
  sorry

end figure_side_length_l8_8605


namespace relationship_between_a_and_b_l8_8337

open Real

theorem relationship_between_a_and_b
  (a b x : ℝ)
  (h1 : a ≠ 1)
  (h2 : b ≠ 1)
  (h3 : 4 * (log x / log a)^3 + 5 * (log x / log b)^3 = 7 * (log x)^3) :
  b = a ^ (3 / 5)^(1 / 3) := 
sorry

end relationship_between_a_and_b_l8_8337


namespace original_faculty_is_287_l8_8797

noncomputable def original_faculty (F : ℝ) : Prop :=
  (F * 0.85 * 0.80 = 195)

theorem original_faculty_is_287 : ∃ F : ℝ, original_faculty F ∧ F = 287 := 
by 
  use 287
  sorry

end original_faculty_is_287_l8_8797


namespace find_sum_of_coefficients_l8_8017

theorem find_sum_of_coefficients : 
  (∃ m n p : ℕ, 
    (n.gcd p = 1) ∧ 
    m + 36 = 72 ∧
    n + 33*3 = 103 ∧ 
    p = 3 ∧ 
    (72 + 33 * ℼ + (8 * (1/8 * (4 * π / 3))) + 36) = m + n * π / p) → 
  m + n + p = 430 :=
by {
  sorry
}

end find_sum_of_coefficients_l8_8017


namespace find_a_if_f_even_l8_8353

noncomputable def f (x a : ℝ) : ℝ := (x + a) * Real.log (((2 * x) - 1) / ((2 * x) + 1))

theorem find_a_if_f_even (a : ℝ) :
  (∀ x : ℝ, (x > 1/2 ∨ x < -1/2) → f x a = f (-x) a) → a = 0 :=
by
  intro h1
  -- This is where the mathematical proof would go, but it's omitted as per the requirements.
  sorry

end find_a_if_f_even_l8_8353


namespace toucan_count_correct_l8_8680

def initial_toucans : ℕ := 2
def toucans_joined : ℕ := 1
def total_toucans : ℕ := initial_toucans + toucans_joined

theorem toucan_count_correct : total_toucans = 3 := by
  sorry

end toucan_count_correct_l8_8680


namespace composite_number_N_l8_8525

theorem composite_number_N (y : ℕ) (hy : y > 1) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = (y ^ 125 - 1) / (3 ^ 22 - 1) :=
by
  -- use sorry to skip the proof
  sorry

end composite_number_N_l8_8525


namespace speed_in_still_water_l8_8279

def upstream_speed : ℝ := 20
def downstream_speed : ℝ := 28

theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 24 := by
  sorry

end speed_in_still_water_l8_8279


namespace sugar_theft_problem_l8_8903

-- Define the statements by Gercoginya and the Cook
def gercoginya_statement := "The cook did not steal the sugar"
def cook_statement := "The sugar was stolen by Gercoginya"

-- Define the thief and truth/lie conditions
def thief_lies (x: String) : Prop := x = "The cook stole the sugar"
def other_truth_or_lie (x y: String) : Prop := x = "The sugar was stolen by Gercoginya" ∨ x = "The sugar was not stolen by Gercoginya"

-- The main proof problem to be solved
theorem sugar_theft_problem : 
  ∃ thief : String, 
    (thief = "cook" ∧ thief_lies gercoginya_statement ∧ other_truth_or_lie cook_statement gercoginya_statement) ∨ 
    (thief = "gercoginya" ∧ thief_lies cook_statement ∧ other_truth_or_lie gercoginya_statement cook_statement) :=
sorry

end sugar_theft_problem_l8_8903


namespace min_total_cost_minimize_cost_l8_8277

theorem min_total_cost (x : ℝ) (h₀ : x > 0) :
  (900 / x * 3 + 3 * x) ≥ 180 :=
by sorry

theorem minimize_cost (x : ℝ) (h₀ : x > 0) :
  x = 30 ↔ (900 / x * 3 + 3 * x) = 180 :=
by sorry

end min_total_cost_minimize_cost_l8_8277


namespace daily_calories_burned_l8_8200

def calories_per_pound : ℕ := 3500
def pounds_to_lose : ℕ := 5
def days : ℕ := 35
def total_calories := pounds_to_lose * calories_per_pound

theorem daily_calories_burned :
  (total_calories / days) = 500 := 
  by 
    -- calculation steps
    sorry

end daily_calories_burned_l8_8200


namespace total_players_ground_l8_8368

-- Define the number of players for each type of sport
def c : ℕ := 10
def h : ℕ := 12
def f : ℕ := 16
def s : ℕ := 13

-- Statement of the problem to prove that the total number of players is 51
theorem total_players_ground : c + h + f + s = 51 :=
by
  -- proof will be added later
  sorry

end total_players_ground_l8_8368


namespace cone_lateral_surface_area_l8_8649

theorem cone_lateral_surface_area {r l : ℝ} (hr : r = 3) (hl : l = 5) :
  ∃ A, A = π * r * l ∧ A = 15 * π :=
by
  use π * r * l
  split
  · rw [hr, hl]
    ring
  · sorry

end cone_lateral_surface_area_l8_8649


namespace election_voters_Sobel_percentage_l8_8971

theorem election_voters_Sobel_percentage : 
  ∀ (total_voters : ℚ) (male_percentage : ℚ) (female_voters_Lange_percentage : ℚ) 
    (male_voters_Sobel_percentage : ℚ), 
  male_percentage = 0.6 → 
  female_voters_Lange_percentage = 0.35 →
  male_voters_Sobel_percentage = 0.44 →
  (total_voters * male_percentage * male_voters_Sobel_percentage + 
  total_voters * (1 - male_percentage) * (1 - female_voters_Lange_percentage)) / total_voters * 100 = 52.4 :=
begin
  intros total_voters male_percentage female_voters_Lange_percentage male_voters_Sobel_percentage,
  assume h1 h2 h3,
  calc
  (total_voters * male_percentage * male_voters_Sobel_percentage + 
  total_voters * (1 - male_percentage) * (1 - female_voters_Lange_percentage)) / total_voters * 100
    = ((60 / 100) * 0.44 * total_voters + (40 / 100) * 0.65 * total_voters) / total_voters * 100 : by rw [h1, h2, h3]
  ... = 52.4 : by { sorry }, 
end

end election_voters_Sobel_percentage_l8_8971


namespace polynomial_factor_pair_l8_8159

theorem polynomial_factor_pair (a b : ℝ) :
  (∃ (c d : ℝ), 3 * x^4 + a * x^3 + 48 * x^2 + b * x + 12 = (2 * x^2 - 3 * x + 2) * (c * x^2 + d * x + 6)) →
  (a, b) = (-26.5, -40) :=
by
  sorry

end polynomial_factor_pair_l8_8159


namespace tadpole_catch_l8_8115

variable (T : ℝ) (H1 : T * 0.25 = 45)

theorem tadpole_catch (T : ℝ) (H1 : T * 0.25 = 45) : T = 180 :=
sorry

end tadpole_catch_l8_8115


namespace tens_digit_19_pow_2023_l8_8575

theorem tens_digit_19_pow_2023 :
  ∃ d : ℕ, d = (59 / 10) % 10 ∧ (19 ^ 2023 % 100) / 10 = d :=
by
  have h1 : 19 ^ 10 % 100 = 1 := by sorry
  have h2 : 19 ↔ 0 := by sorry
  have h4 : 2023 % 10 = 3 := by sorry
  have h5 : 19 ^ 10 ↔ 1 := by sorry
  have h6 : 19 ^ 3 % 100 = 59 := by sorry
  have h7 : (19 ^ 2023 % 100) = 59 := by sorry
  exists 5
  split
  repeat { assumption.dump }

end tens_digit_19_pow_2023_l8_8575


namespace right_triangle_unique_value_l8_8269

theorem right_triangle_unique_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
(h1 : a + b + c = (1/2) * a * b) (h2 : c^2 = a^2 + b^2) : a + b - c = 4 :=
by
  sorry

end right_triangle_unique_value_l8_8269


namespace prescription_duration_l8_8625

theorem prescription_duration (D : ℕ) (h1 : (2 * D) * (1 / 5) = 12) : D = 30 :=
by
  sorry

end prescription_duration_l8_8625


namespace sum_of_cubes_l8_8761

theorem sum_of_cubes (x y : ℝ) (hx : x + y = 10) (hxy : x * y = 12) : x^3 + y^3 = 640 := 
by
  sorry

end sum_of_cubes_l8_8761


namespace equation_of_lamps_l8_8780

theorem equation_of_lamps (n k : ℕ) (N M : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : k ≥ n) (h4 : (k - n) % 2 = 0) : 
  N = 2^(k - n) * M := 
sorry

end equation_of_lamps_l8_8780


namespace compare_negative_fractions_l8_8302

theorem compare_negative_fractions : (- (1 / 3 : ℝ)) < (- (1 / 4 : ℝ)) :=
sorry

end compare_negative_fractions_l8_8302


namespace determine_perimeter_of_fourth_shape_l8_8089

theorem determine_perimeter_of_fourth_shape
  (P_1 P_2 P_3 P_4 : ℝ)
  (h1 : P_1 = 8)
  (h2 : P_2 = 11.4)
  (h3 : P_3 = 14.7)
  (h4 : P_1 + P_2 + P_4 = 2 * P_3) :
  P_4 = 10 := 
by
  -- Proof goes here
  sorry

end determine_perimeter_of_fourth_shape_l8_8089


namespace min_value_l8_8232

theorem min_value (x : ℝ) (h : x > 1) : (x + 4 / (x - 1)) ≥ 5 :=
by sorry

end min_value_l8_8232


namespace total_age_proof_l8_8822

noncomputable def total_age : ℕ :=
  let susan := 15
  let arthur := susan + 2
  let bob := 11
  let tom := bob - 3
  let emily := susan / 2
  let david := (arthur + tom + emily) / 3
  susan + arthur + tom + bob + emily + david

theorem total_age_proof : total_age = 70 := by
  unfold total_age
  sorry

end total_age_proof_l8_8822


namespace sum_of_possible_values_of_x_l8_8496

namespace ProofProblem

-- Assume we are working in degrees for angles
def is_scalene_triangle (A B C : ℝ) (a b c : ℝ) :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a

def triangle_angle_sum (A B C : ℝ) : Prop :=
  A + B + C = 180

noncomputable def problem_statement (x : ℝ) (A B C : ℝ) (a b c : ℝ) : Prop :=
  is_scalene_triangle A B C a b c ∧
  B = 45 ∧
  (A = x ∨ C = x) ∧
  (a = b ∨ b = c ∨ c = a) ∧
  triangle_angle_sum A B C

theorem sum_of_possible_values_of_x (x : ℝ) (A B C : ℝ) (a b c : ℝ) :
  problem_statement x A B C a b c →
  x = 45 :=
sorry

end ProofProblem

end sum_of_possible_values_of_x_l8_8496


namespace fraction_division_problem_l8_8220

theorem fraction_division_problem :
  (-1/42 : ℚ) / (1/6 - 3/14 + 2/3 - 2/7) = -1/14 :=
by
  -- Skipping the proof step as per the instructions
  sorry

end fraction_division_problem_l8_8220


namespace total_carrots_l8_8223

theorem total_carrots (carrots_sandy carrots_mary : ℕ) (h1 : carrots_sandy = 8) (h2 : carrots_mary = 6) :
  carrots_sandy + carrots_mary = 14 :=
by
  sorry

end total_carrots_l8_8223


namespace graph_symmetric_about_x_eq_pi_div_8_l8_8341

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x + Real.cos x)

theorem graph_symmetric_about_x_eq_pi_div_8 :
  ∀ x, f (π / 8 - x) = f (π / 8 + x) :=
sorry

end graph_symmetric_about_x_eq_pi_div_8_l8_8341


namespace Emily_GRE_Exam_Date_l8_8727

theorem Emily_GRE_Exam_Date : 
  ∃ (exam_date : ℕ) (exam_month : String), 
  exam_date = 5 ∧ exam_month = "September" ∧
  ∀ study_days break_days start_day_cycles start_break_cycles start_month_june total_days S_june_remaining S_remaining_july S_remaining_august September_start_day, 
    study_days = 15 ∧ 
    break_days = 5 ∧ 
    start_day_cycles = 5 ∧ 
    start_break_cycles = 4 ∧ 
    start_month_june = 1 ∧
    total_days = start_day_cycles * study_days + start_break_cycles * break_days ∧ 
    S_june_remaining = 30 - start_month_june ∧ 
    S_remaining = total_days - S_june_remaining ∧ 
    S_remaining_july = S_remaining - 31 ∧ 
    S_remaining_august = S_remaining_july - 31 ∧ 
    September_start_day = S_remaining_august + 1 ∧
    exam_date = September_start_day ∧ 
    exam_month = "September" := by 
  sorry

end Emily_GRE_Exam_Date_l8_8727


namespace evaluate_expression_l8_8311

variable (b : ℝ) -- assuming b is a real number, (if b should be of different type, modify accordingly)

theorem evaluate_expression (y : ℝ) (h : y = b + 9) : y - b + 5 = 14 :=
by
  sorry

end evaluate_expression_l8_8311


namespace proof_problem_l8_8952

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * Real.sin x + b * x^3 + 4

noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * Real.cos x + 3 * b * x^2

theorem proof_problem (a b : ℝ) :
  f 2016 a b + f (-2016) a b + f' 2017 a b - f' (-2017) a b = 8 := by
  sorry

end proof_problem_l8_8952


namespace freezer_temp_correct_l8_8077

theorem freezer_temp_correct:
  (∃ A B C D : ℝ, 
    A = +18 ∧ 
    B = -18 ∧ 
    C = 0 ∧ 
    D = -5 ∧ 
    (temperature_in_freezer : ℝ) = -18) := 
  sorry

end freezer_temp_correct_l8_8077


namespace exists_x_given_y_l8_8587

theorem exists_x_given_y (y : ℝ) : ∃ x : ℝ, x^2 + y^2 = 10 ∧ x^2 - x * y - 3 * y + 12 = 0 := 
sorry

end exists_x_given_y_l8_8587


namespace ratio_of_teenagers_to_toddlers_l8_8018

theorem ratio_of_teenagers_to_toddlers
  (total_children : ℕ)
  (number_of_toddlers : ℕ)
  (number_of_newborns : ℕ)
  (h1 : total_children = 40)
  (h2 : number_of_toddlers = 6)
  (h3 : number_of_newborns = 4)
  : (total_children - number_of_toddlers - number_of_newborns) / number_of_toddlers = 5 :=
by
  sorry

end ratio_of_teenagers_to_toddlers_l8_8018


namespace solve_for_x_l8_8423

theorem solve_for_x (x : ℤ) (h : 13 * x + 14 * x + 17 * x + 11 = 143) : x = 3 :=
by sorry

end solve_for_x_l8_8423


namespace minimum_k_condition_l8_8172

def is_acute_triangle (a b c : ℕ) : Prop :=
  a * a + b * b > c * c

def any_subset_with_three_numbers_construct_acute_triangle (s : Finset ℕ) : Prop :=
  ∀ t : Finset ℕ, t.card = 3 → 
    (∃ a b c : ℕ, a ∈ t ∧ b ∈ t ∧ c ∈ t ∧ 
      is_acute_triangle a b c ∨
      is_acute_triangle a c b ∨
      is_acute_triangle b c a)

theorem minimum_k_condition (k : ℕ) :
  (∀ s : Finset ℕ, s.card = k → any_subset_with_three_numbers_construct_acute_triangle s) ↔ (k = 29) :=
  sorry

end minimum_k_condition_l8_8172


namespace average_distance_is_600_l8_8790

-- Definitions based on the conditions
def one_lap_distance : ℕ := 200
def johnny_lap_count : ℕ := 4
def mickey_lap_count : ℕ := johnny_lap_count / 2

-- Total distances run by Johnny and Mickey
def johnny_distance : ℕ := johnny_lap_count * one_lap_distance
def mickey_distance : ℕ := mickey_lap_count * one_lap_distance

-- Sum of distances
def total_distance : ℕ := johnny_distance + mickey_distance

-- Number of people
def number_of_people : ℕ := 2

-- Average distance calculation
def average_distance : ℕ := total_distance / number_of_people

-- The theorem to prove: the average distance run by Johnny and Mickey
theorem average_distance_is_600 : average_distance = 600 := by
  sorry

end average_distance_is_600_l8_8790


namespace calculation_correct_l8_8009

theorem calculation_correct : 1984 + 180 / 60 - 284 = 1703 := 
by 
  sorry

end calculation_correct_l8_8009


namespace neon_signs_blink_together_l8_8326

theorem neon_signs_blink_together :
  Nat.lcm (Nat.lcm (Nat.lcm 7 11) 13) 17 = 17017 :=
by
  sorry

end neon_signs_blink_together_l8_8326


namespace max_value_of_x_times_one_minus_2x_l8_8030

theorem max_value_of_x_times_one_minus_2x : 
  ∀ x : ℝ, 0 < x ∧ x < 1 / 2 → x * (1 - 2 * x) ≤ 1 / 8 :=
by
  intro x 
  intro hx
  sorry

end max_value_of_x_times_one_minus_2x_l8_8030


namespace volume_remaining_proof_l8_8171

noncomputable def volume_remaining_part (v_original v_total_small : ℕ) : ℕ := v_original - v_total_small

def original_edge_length := 9
def small_edge_length := 3
def num_edges := 12

def volume_original := original_edge_length ^ 3
def volume_small := small_edge_length ^ 3
def volume_total_small := num_edges * volume_small

theorem volume_remaining_proof : volume_remaining_part volume_original volume_total_small = 405 := by
  sorry

end volume_remaining_proof_l8_8171


namespace percentage_of_class_are_men_proof_l8_8057

/-- Definition of the problem using the conditions provided. -/
def percentage_of_class_are_men (W M : ℝ) : Prop :=
  -- Conditions based on the problem statement
  M + W = 100 ∧
  0.10 * W + 0.85 * M = 40

/-- The proof statement we need to show: Under the given conditions, the percentage of men (M) is 40. -/
theorem percentage_of_class_are_men_proof (W M : ℝ) :
  percentage_of_class_are_men W M → M = 40 :=
by
  sorry

end percentage_of_class_are_men_proof_l8_8057


namespace part1_part2_l8_8186

-- Define A and B according to given expressions
def A (a b : ℚ) : ℚ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℚ) : ℚ := -a^2 + a * b - 1

-- Prove the first statement
theorem part1 (a b : ℚ) : 4 * A a b - (3 * A a b - 2 * B a b) = 5 * a * b - 2 * a - 3 :=
by sorry

-- Prove the second statement
theorem part2 (F : ℚ) (b : ℚ) : (∀ a, A a b + 2 * B a b = F) → b = 2 / 5 :=
by sorry

end part1_part2_l8_8186


namespace ratio_of_areas_of_similar_triangles_l8_8608

theorem ratio_of_areas_of_similar_triangles (m1 m2 : ℝ) (med_ratio : m1 / m2 = 1 / Real.sqrt 2) :
    let area_ratio := (m1 / m2) ^ 2
    area_ratio = 1 / 2 := by
  sorry

end ratio_of_areas_of_similar_triangles_l8_8608


namespace quotient_of_division_l8_8829

theorem quotient_of_division (L : ℕ) (S : ℕ) (Q : ℕ) (h1 : L = 1631) (h2 : L - S = 1365) (h3 : L = S * Q + 35) :
  Q = 6 :=
by
  sorry

end quotient_of_division_l8_8829


namespace regular_polygon_sides_l8_8044

theorem regular_polygon_sides (interior_angle exterior_angle : ℕ)
  (h1 : interior_angle = exterior_angle + 60)
  (h2 : interior_angle + exterior_angle = 180) :
  ∃ n : ℕ, n = 6 :=
by
  have ext_angle_eq : exterior_angle = 60 := sorry
  have ext_angles_sum : exterior_angle * 6 = 360 := sorry
  exact ⟨6, by linarith⟩

end regular_polygon_sides_l8_8044


namespace max_x_add_inv_x_l8_8246

variable (x : ℝ) (y : Fin 2022 → ℝ)

-- Conditions
def sum_condition : Prop := x + (Finset.univ.sum y) = 2024
def reciprocal_sum_condition : Prop := (1/x) + (Finset.univ.sum (λ i => 1 / (y i))) = 2024

-- The statement we need to prove
theorem max_x_add_inv_x (h_sum : sum_condition x y) (h_rec_sum : reciprocal_sum_condition x y) : 
  x + (1/x) ≤ 2 := by
  sorry

end max_x_add_inv_x_l8_8246


namespace smallest_delicious_integer_is_minus_2022_l8_8931

def smallest_delicious_integer (sum_target : ℤ) : ℤ :=
  -2022

theorem smallest_delicious_integer_is_minus_2022
  (B : ℤ)
  (h : ∃ (s : List ℤ), s.sum = 2023 ∧ B ∈ s) :
  B = -2022 :=
sorry

end smallest_delicious_integer_is_minus_2022_l8_8931


namespace remaining_pieces_l8_8007

/-- Define the initial number of pieces on a standard chessboard. -/
def initial_pieces : Nat := 32

/-- Define the number of pieces lost by Audrey. -/
def audrey_lost : Nat := 6

/-- Define the number of pieces lost by Thomas. -/
def thomas_lost : Nat := 5

/-- Proof that the remaining number of pieces on the chessboard is 21. -/
theorem remaining_pieces : initial_pieces - (audrey_lost + thomas_lost) = 21 := by
  -- Mathematical equivalence to 32 - (6 + 5) = 21
  sorry

end remaining_pieces_l8_8007


namespace problem_statement_l8_8209

variable (x y : ℝ)
def t : ℝ := x / y

theorem problem_statement (h : 1 < (x - y) / (x + y) ∧ (x - y) / (x + y) < 3) : t = 1 := by 
  sorry

end problem_statement_l8_8209


namespace problem_l8_8484

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry
noncomputable def z : ℝ := sorry

theorem problem 
  (h1 : 0 < x) 
  (h2 : 0 < y) 
  (h3 : 0 < z) 
  (h4 : x * y = 30) 
  (h5 : x * z = 60) 
  (h6 : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := 
  sorry

end problem_l8_8484


namespace base_conversion_least_sum_l8_8236

theorem base_conversion_least_sum (a b : ℕ) (h : 3 * a + 5 = 5 * b + 3) : a + b = 10 :=
sorry

end base_conversion_least_sum_l8_8236


namespace average_math_score_of_class_l8_8248

theorem average_math_score_of_class (n : ℕ) (jimin_score jung_score avg_others : ℕ) 
  (h1 : n = 40) 
  (h2 : jimin_score = 98) 
  (h3 : jung_score = 100) 
  (h4 : avg_others = 79) : 
  (38 * avg_others + jimin_score + jung_score) / n = 80 :=
by sorry

end average_math_score_of_class_l8_8248


namespace find_other_number_l8_8823

theorem find_other_number (LCM HCF num1 num2 : ℕ) 
  (h1 : LCM = 2310) 
  (h2 : HCF = 30) 
  (h3 : num1 = 330) 
  (h4 : LCM * HCF = num1 * num2) : 
  num2 = 210 := by 
  sorry

end find_other_number_l8_8823


namespace complement_intersection_l8_8955

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4})
variable (hA : A = {1, 2, 3})
variable (hB : B = {2, 3, 4})

theorem complement_intersection :
  (U \ (A ∩ B)) = {1, 4} :=
by
  sorry

end complement_intersection_l8_8955


namespace parallel_vectors_tan_l8_8344

/-- Given vector a and vector b, and given the condition that a is parallel to b,
prove that the value of tan α is 1/4. -/
theorem parallel_vectors_tan (α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (Real.sin α, Real.cos α - 2 * Real.sin α))
  (hb : b = (1, 2))
  (h_parallel : ∃ k : ℝ, a = (k * b.1, k * b.2)) : 
  Real.tan α = 1 / 4 := 
by 
  sorry

end parallel_vectors_tan_l8_8344


namespace sum_of_squares_and_products_l8_8490

theorem sum_of_squares_and_products
  (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z)
  (h4 : x^2 + y^2 + z^2 = 52) (h5 : x * y + y * z + z * x = 24) :
  x + y + z = 10 := 
by
  sorry

end sum_of_squares_and_products_l8_8490


namespace fraction_is_one_fourth_l8_8397

-- Defining the numbers
def num1 : ℕ := 16
def num2 : ℕ := 8

-- Conditions
def difference_correct : Prop := num1 - num2 = 8
def sum_of_numbers : ℕ := num1 + num2
def fraction_of_sum (f : ℚ) : Prop := f * sum_of_numbers = 6

-- Theorem stating the fraction
theorem fraction_is_one_fourth (f : ℚ) (h1 : difference_correct) (h2 : fraction_of_sum f) : f = 1 / 4 :=
by {
  -- This will use the conditions and show that f = 1/4
  sorry
}

end fraction_is_one_fourth_l8_8397


namespace find_a_minus_b_l8_8593

theorem find_a_minus_b (a b x y : ℤ)
  (h_x : x = 1)
  (h_y : y = 1)
  (h1 : a * x + b * y = 2)
  (h2 : x - b * y = 3) :
  a - b = 6 := by
  subst h_x
  subst h_y
  simp at h1 h2
  have h_b: b = -2 := by linarith
  have h_a: a = 4 := by linarith
  rw [h_a, h_b]
  norm_num

end find_a_minus_b_l8_8593


namespace problem1_problem2_l8_8707

theorem problem1 : 4 * Real.sqrt 2 + Real.sqrt 8 - Real.sqrt 18 = 3 * Real.sqrt 2 := by
  sorry

theorem problem2 : Real.sqrt (4 / 3) / Real.sqrt (7 / 3) * Real.sqrt (7 / 5) = 2 * Real.sqrt 5 / 5 := by
  sorry

end problem1_problem2_l8_8707


namespace ethanol_percentage_in_fuel_A_l8_8144

noncomputable def percent_ethanol_in_fuel_A : ℝ := 0.12

theorem ethanol_percentage_in_fuel_A
  (fuel_tank_capacity : ℝ)
  (fuel_A_volume : ℝ)
  (fuel_B_volume : ℝ)
  (fuel_B_ethanol_percent : ℝ)
  (total_ethanol : ℝ) :
  fuel_tank_capacity = 218 → 
  fuel_A_volume = 122 → 
  fuel_B_volume = 96 → 
  fuel_B_ethanol_percent = 0.16 → 
  total_ethanol = 30 → 
  (fuel_A_volume * percent_ethanol_in_fuel_A) + (fuel_B_volume * fuel_B_ethanol_percent) = total_ethanol :=
by
  sorry

end ethanol_percentage_in_fuel_A_l8_8144


namespace probability_red_blue_green_l8_8412

def total_marbles : ℕ := 5 + 4 + 3 + 6
def favorable_marbles : ℕ := 5 + 4 + 3

theorem probability_red_blue_green : 
  (favorable_marbles : ℚ) / total_marbles = 2 / 3 := 
by 
  sorry

end probability_red_blue_green_l8_8412


namespace triangle_angle_determinant_zero_l8_8779

theorem triangle_angle_determinant_zero (θ φ ψ : ℝ) (h : θ + φ + ψ = Real.pi) : 
  Matrix.det !![![Real.cos θ, Real.sin θ, 1], ![Real.cos φ, Real.sin φ, 1], ![Real.cos ψ, Real.sin ψ, 1]] = 0 :=
by 
  sorry

end triangle_angle_determinant_zero_l8_8779


namespace tens_digit_of_19_pow_2023_l8_8579

theorem tens_digit_of_19_pow_2023 : (19 ^ 2023) % 100 = 59 := 
  sorry

end tens_digit_of_19_pow_2023_l8_8579


namespace isosceles_triangle_range_l8_8460

theorem isosceles_triangle_range (x : ℝ) (h1 : 0 < x) (h2 : 2 * x + (10 - 2 * x) = 10):
  (5 / 2) < x ∧ x < 5 :=
by
  sorry

end isosceles_triangle_range_l8_8460


namespace complement_U_A_eq_l8_8954
noncomputable def U := {x : ℝ | x ≥ -2}
noncomputable def A := {x : ℝ | x > -1}
noncomputable def comp_U_A := {x ∈ U | x ∉ A}

theorem complement_U_A_eq : comp_U_A = {x : ℝ | -2 ≤ x ∧ x < -1} :=
by sorry

end complement_U_A_eq_l8_8954


namespace ratio_population_XZ_l8_8897

variable (Population : Type) [Field Population]
variable (Z : Population) -- Population of City Z
variable (Y : Population) -- Population of City Y
variable (X : Population) -- Population of City X

-- Conditions
def population_Y : Y = 2 * Z := sorry
def population_X : X = 7 * Y := sorry

-- Theorem stating the ratio of populations
theorem ratio_population_XZ : (X / Z) = 14 := by
  -- The proof will use the conditions population_Y and population_X
  sorry

end ratio_population_XZ_l8_8897


namespace average_hit_targets_value_average_hit_targets_ge_half_l8_8716

noncomputable def average_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - 1 / n)^n)

theorem average_hit_targets_value (n : ℕ) :
  average_hit_targets n = n * (1 - (1 - 1 / n)^n) :=
by sorry

theorem average_hit_targets_ge_half (n : ℕ) :
  average_hit_targets n >= n / 2 :=
by sorry

end average_hit_targets_value_average_hit_targets_ge_half_l8_8716


namespace tangent_line_at_a_eq_one_range_of_a_for_exactly_one_zero_l8_8476

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := real.log (1 + x) + a * x * real.exp (-x)

theorem tangent_line_at_a_eq_one :
  let a := 1
  in ∀ x, let y := f a x, 
    y = 2 * x :=
by
  intro a x h
  sorry

theorem range_of_a_for_exactly_one_zero :
  (∀ f, f a has_zero_in_each_of (interval -1 0) (interval 0 ∞)) → (a < -1) :=
by
  intro h
  sorry

end tangent_line_at_a_eq_one_range_of_a_for_exactly_one_zero_l8_8476


namespace probability_space_diagonal_l8_8117

theorem probability_space_diagonal : 
  let vertices := 8
  let space_diagonals := 4
  let total_pairs := Nat.choose vertices 2
  4 / total_pairs = 1 / 7 :=
by
  sorry

end probability_space_diagonal_l8_8117


namespace Razorback_tshirt_shop_sales_l8_8532

theorem Razorback_tshirt_shop_sales :
  let tshirt_price := 98
  let hat_price := 45
  let scarf_price := 60
  let tshirts_sold_arkansas := 42
  let hats_sold_arkansas := 32
  let scarves_sold_arkansas := 15
  (tshirts_sold_arkansas * tshirt_price + hats_sold_arkansas * hat_price + scarves_sold_arkansas * scarf_price) = 6456 :=
by
  sorry

end Razorback_tshirt_shop_sales_l8_8532


namespace range_of_a_l8_8651

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (0 ≤ a ∧ a < 3) :=
sorry

end range_of_a_l8_8651


namespace correct_statements_l8_8558

/-
Let A be the event "the first roll is an even number",
B be the event "the second roll is an even number",
C be the event "the sum of the two rolls is an even number",
and both rolls are done with a fair cubic dice.
Prove that the statements A, C, and D are correct.
A: P(A) = 1 - P(B)
C: B and C are independent events
D: P(A ∪ B) = 3 / 4
-/

open ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {P : MeasureTheory.ProbabilityMeasure Ω}

def first_roll_even (ω : Ω) : Prop := ω ∈ {i | i % 2 = 0}
def second_roll_even (ω : Ω) : Prop := ω ∈ {i | i % 2 = 0}
def sum_rolls_even (ω : Ω) : Prop := (ω % 6 % 2 + ω % 6 % 2) % 2 = 0

theorem correct_statements (fair_dice : ∀ ω, P {a | a = first_roll_even ω} = 1 / 2) :
  (P {ω | first_roll_even ω} = 1 - P {ω | second_roll_even ω}) ∧
  (∀ t, P (λ ω, second_roll_even ω ∧ sum_rolls_even ω) = P (λ ω, second_roll_even ω) * P (λ ω, sum_rolls_even ω)) ∧
  (P {ω | first_roll_even ω ∨ second_roll_even ω} = 3 / 4) :=
sorry

end correct_statements_l8_8558


namespace B_share_correct_l8_8094

noncomputable def total_share : ℕ := 120
noncomputable def B_share : ℕ := 20
noncomputable def A_share (x : ℕ) : ℕ := x + 20
noncomputable def C_share (x : ℕ) : ℕ := x + 40

theorem B_share_correct : ∃ x : ℕ, total_share = (A_share x) + x + (C_share x) ∧ x = B_share := by
  sorry

end B_share_correct_l8_8094


namespace factorization_x6_minus_5x4_plus_8x2_minus_4_l8_8730

theorem factorization_x6_minus_5x4_plus_8x2_minus_4 (x : ℝ) :
  x^6 - 5 * x^4 + 8 * x^2 - 4 = (x - 1) * (x + 1) * (x^2 - 2)^2 :=
sorry

end factorization_x6_minus_5x4_plus_8x2_minus_4_l8_8730


namespace sum_first_eight_geom_terms_eq_l8_8586

noncomputable def S8_geom_sum : ℚ :=
  let a := (1 : ℚ) / 3
  let r := (1 : ℚ) / 3
  a * (1 - r^8) / (1 - r)

theorem sum_first_eight_geom_terms_eq :
  S8_geom_sum = 3280 / 6561 :=
by
  sorry

end sum_first_eight_geom_terms_eq_l8_8586


namespace total_keys_needed_l8_8544

-- Definitions based on given conditions
def num_complexes : ℕ := 2
def num_apartments_per_complex : ℕ := 12
def keys_per_lock : ℕ := 3
def num_locks_per_apartment : ℕ := 1

-- Theorem stating the required number of keys
theorem total_keys_needed : 
  (num_complexes * num_apartments_per_complex * keys_per_lock = 72) :=
by
  sorry

end total_keys_needed_l8_8544


namespace find_f1_l8_8048

theorem find_f1 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f (3 * x + 1) = x^2 + 3*x + 2) :
  f 1 = 2 :=
by
  -- Proof is omitted
  sorry

end find_f1_l8_8048


namespace not_perfect_square_of_divisor_l8_8510

theorem not_perfect_square_of_divisor (n d : ℕ) (hn : 0 < n) (hd : d ∣ 2 * n^2) :
  ¬ ∃ x : ℕ, n^2 + d = x^2 :=
by
  sorry

end not_perfect_square_of_divisor_l8_8510


namespace number_of_avocados_l8_8275

-- Constants for the given problem
def banana_cost : ℕ := 1
def apple_cost : ℕ := 2
def strawberry_cost_per_12 : ℕ := 4
def avocado_cost : ℕ := 3
def grape_cost_half_bunch : ℕ := 2
def total_cost : ℤ := 28

-- Quantities of the given fruits
def banana_qty : ℕ := 4
def apple_qty : ℕ := 3
def strawberry_qty : ℕ := 24
def grape_qty_full_bunch_cost : ℕ := 4 -- since half bunch cost $2, full bunch cost $4

-- Definition to calculate the cost of the known fruits
def known_fruit_cost : ℤ :=
  (banana_qty * banana_cost) +
  (apple_qty * apple_cost) +
  (strawberry_qty / 12 * strawberry_cost_per_12) +
  grape_qty_full_bunch_cost

-- The cost of avocados needed to fill the total cost
def avocado_cost_needed : ℤ := total_cost - known_fruit_cost

-- Finally, we need to prove that the number of avocados is 2
theorem number_of_avocados (n : ℕ) : n * avocado_cost = avocado_cost_needed → n = 2 :=
by
  -- Problem data
  have h_banana : ℕ := banana_qty * banana_cost
  have h_apple : ℕ := apple_qty * apple_cost
  have h_strawberry : ℕ := (strawberry_qty / 12) * strawberry_cost_per_12
  have h_grape : ℕ := grape_qty_full_bunch_cost
  have h_known : ℕ := h_banana + h_apple + h_strawberry + h_grape
  
  -- Calculation for number of avocados
  have h_avocado : ℤ := total_cost - h_known
  
  -- Proving number of avocados
  sorry

end number_of_avocados_l8_8275


namespace secret_spread_reaches_3280_on_saturday_l8_8217

theorem secret_spread_reaches_3280_on_saturday :
  (∃ n : ℕ, 4 * ( 3^n - 1) / 2 + 1 = 3280 ) ∧ n = 7  :=
sorry

end secret_spread_reaches_3280_on_saturday_l8_8217


namespace binomial_expansion_a0_a1_a3_a5_l8_8345

theorem binomial_expansion_a0_a1_a3_a5 
    (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
    (h : (1 + 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) :
  a_0 + a_1 + a_3 + a_5 = 123 :=
sorry

end binomial_expansion_a0_a1_a3_a5_l8_8345


namespace factorization_of_difference_of_squares_l8_8312

variable {R : Type} [CommRing R]

theorem factorization_of_difference_of_squares (m : R) : m^2 - 4 = (m + 2) * (m - 2) :=
by sorry

end factorization_of_difference_of_squares_l8_8312


namespace parabola_directrix_y_neg1_l8_8998

-- We define the problem given the conditions.
def parabola_directrix (p : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 = 4 * y → y = -p

-- Now we state what needs to be proved.
theorem parabola_directrix_y_neg1 : parabola_directrix 1 :=
by
  sorry

end parabola_directrix_y_neg1_l8_8998


namespace fuel_tank_capacity_l8_8145

theorem fuel_tank_capacity
  (ethanol_A_fraction : ℝ)
  (ethanol_B_fraction : ℝ)
  (ethanol_total : ℝ)
  (fuel_A_volume : ℝ)
  (C : ℝ)
  (h1 : ethanol_A_fraction = 0.12)
  (h2 : ethanol_B_fraction = 0.16)
  (h3 : ethanol_total = 28)
  (h4 : fuel_A_volume = 99.99999999999999)
  (h5 : 0.12 * 99.99999999999999 + 0.16 * (C - 99.99999999999999) = 28) :
  C = 200 := 
sorry

end fuel_tank_capacity_l8_8145


namespace min_value_arithmetic_sequence_l8_8034

theorem min_value_arithmetic_sequence (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_arith_seq : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_sum : a 1 + a 2014 = 2) :
  (∃ a2 a2013, a 2 = a2 ∧ a 2013 = a2013 ∧ a2 > 0 ∧ a2013 > 0 ∧ ∀ a2 a2013, a 2 = a2 ∧ a 2013 = a2013 → (1/a2 + 1/a2013) ≥ 2) :=
by
  sorry

end min_value_arithmetic_sequence_l8_8034


namespace maria_score_l8_8381

theorem maria_score (m j : ℕ) (h1 : m = j + 50) (h2 : (m + j) / 2 = 112) : m = 137 :=
by
  sorry

end maria_score_l8_8381


namespace retail_price_machine_l8_8287

theorem retail_price_machine (P : ℝ) :
  let wholesale_price := 99
  let discount_rate := 0.10
  let profit_rate := 0.20
  let selling_price := wholesale_price + (profit_rate * wholesale_price)
  0.90 * P = selling_price → P = 132 :=

by
  intro wholesale_price discount_rate profit_rate selling_price h
  sorry -- Proof will be handled here

end retail_price_machine_l8_8287


namespace recurring_decimal_to_fraction_l8_8936

theorem recurring_decimal_to_fraction : ∀ x : ℝ, (x = 7 + (1/3 : ℝ)) → x = (22/3 : ℝ) :=
by
  sorry

end recurring_decimal_to_fraction_l8_8936


namespace difference_between_mean_and_median_l8_8610

def percent_students := {p : ℝ // 0 ≤ p ∧ p ≤ 1}

def students_scores_distribution (p60 p75 p85 p95 : percent_students) : Prop :=
  p60.val + p75.val + p85.val + p95.val = 1 ∧
  p60.val = 0.15 ∧
  p75.val = 0.20 ∧
  p85.val = 0.40 ∧
  p95.val = 0.25

noncomputable def weighted_mean (p60 p75 p85 p95 : percent_students) : ℝ :=
  60 * p60.val + 75 * p75.val + 85 * p85.val + 95 * p95.val

noncomputable def median_score (p60 p75 p85 p95 : percent_students) : ℝ :=
  if p60.val + p75.val < 0.5 then 85 else if p60.val + p75.val < 0.9 then 95 else 60

theorem difference_between_mean_and_median :
  ∀ (p60 p75 p85 p95 : percent_students),
    students_scores_distribution p60 p75 p85 p95 →
    abs (median_score p60 p75 p85 p95 - weighted_mean p60 p75 p85 p95) = 3.25 :=
by
  intro p60 p75 p85 p95
  intro h
  sorry

end difference_between_mean_and_median_l8_8610


namespace factorial_division_l8_8662

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division : (factorial 15) / ((factorial 6) * (factorial 9)) = 5005 :=
by
  sorry

end factorial_division_l8_8662


namespace total_fencing_cost_is_5300_l8_8400

-- Define the conditions
def length_more_than_breadth_condition (l b : ℕ) := l = b + 40
def fencing_cost_per_meter : ℝ := 26.50
def given_length : ℕ := 70

-- Define the perimeter calculation
def perimeter (l b : ℕ) := 2 * l + 2 * b

-- Define the total cost calculation
def total_cost (P : ℕ) (cost_per_meter : ℝ) := P * cost_per_meter

-- State the theorem to be proven
theorem total_fencing_cost_is_5300 (b : ℕ) (l := given_length) :
  length_more_than_breadth_condition l b →
  total_cost (perimeter l b) fencing_cost_per_meter = 5300 :=
by
  sorry

end total_fencing_cost_is_5300_l8_8400


namespace part_one_tangent_line_part_two_range_of_a_l8_8474

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

theorem part_one_tangent_line :
  (∀ x : ℝ, f 1 x = Real.log (1 + x) + x * Real.exp (-x)) →
  f 1 0 = 0 ∧ (deriv (f 1) 0 = 2) →
  ∀ x : ℝ, 2 * x = (deriv (f 1) 0) * x + (f 1 0) :=
sorry

theorem part_two_range_of_a :
  (∀ a : ℝ, a < -1 →
    ∃ x₁ ∈ Ioo (-1 : ℝ) 0, f a x₁ = 0 ∧
    ∃ x₂ ∈ Ioo (0 : ℝ) (+∞ : ℝ), f a x₂ = 0) →
  ∀ a : ℝ, a ∈ Iio (-1) :=
sorry

end part_one_tangent_line_part_two_range_of_a_l8_8474


namespace find_x_for_equation_l8_8899

def f (x : ℝ) : ℝ := 2 * x - 3

theorem find_x_for_equation : (2 * f x - 21 = f (x - 4)) ↔ (x = 8) :=
by
  sorry

end find_x_for_equation_l8_8899


namespace division_remainder_example_l8_8284

theorem division_remainder_example :
  ∃ n, n = 20 * 10 + 10 ∧ n = 210 :=
by
  sorry

end division_remainder_example_l8_8284


namespace Kelly_current_baking_powder_l8_8555

-- Definitions based on conditions
def yesterday_amount : ℝ := 0.4
def difference : ℝ := 0.1
def current_amount : ℝ := yesterday_amount - difference

-- Statement to prove the question == answer given the conditions
theorem Kelly_current_baking_powder : current_amount = 0.3 := 
by
  sorry

end Kelly_current_baking_powder_l8_8555


namespace range_of_w_l8_8749

noncomputable def f (w x : ℝ) : ℝ := Real.sin (w * x) - Real.sqrt 3 * Real.cos (w * x)

theorem range_of_w (w : ℝ) (h_w : 0 < w) :
  (∀ f_zeros : Finset ℝ, ∀ x ∈ f_zeros, (0 < x ∧ x < Real.pi) → f w x = 0 → f_zeros.card = 2) ↔
  (4 / 3 < w ∧ w ≤ 7 / 3) :=
by sorry

end range_of_w_l8_8749


namespace Marcy_120_votes_l8_8966

-- Definitions based on conditions
def votes (name : String) : ℕ := sorry -- placeholder definition

-- Conditions
def Joey_votes := votes "Joey" = 8
def Jill_votes := votes "Jill" = votes "Joey" + 4
def Barry_votes := votes "Barry" = 2 * (votes "Joey" + votes "Jill")
def Marcy_votes := votes "Marcy" = 3 * votes "Barry"
def Tim_votes := votes "Tim" = votes "Marcy" / 2
def Sam_votes := votes "Sam" = votes "Tim" + 10

-- Theorem to prove
theorem Marcy_120_votes : Joey_votes → Jill_votes → Barry_votes → Marcy_votes → Tim_votes → Sam_votes → votes "Marcy" = 120 := by
  intros
  -- Skipping the proof
  sorry

end Marcy_120_votes_l8_8966


namespace cos_tan_values_l8_8330

theorem cos_tan_values (α : ℝ) (h : Real.sin α = -1 / 2) :
  (∃ (quadrant : ℕ), 
    (quadrant = 3 ∧ Real.cos α = -Real.sqrt 3 / 2 ∧ Real.tan α = Real.sqrt 3 / 3) ∨ 
    (quadrant = 4 ∧ Real.cos α = Real.sqrt 3 / 2 ∧ Real.tan α = -Real.sqrt 3 / 3)) :=
sorry

end cos_tan_values_l8_8330


namespace money_distribution_l8_8001

theorem money_distribution (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 360) (h3 : C = 60) : A + B + C = 500 := by
  sorry

end money_distribution_l8_8001


namespace rate_percent_calculation_l8_8902

theorem rate_percent_calculation (SI P T : ℝ) (R : ℝ) : SI = 640 ∧ P = 4000 ∧ T = 2 → SI = P * R * T / 100 → R = 8 :=
by
  intros
  sorry

end rate_percent_calculation_l8_8902


namespace complement_of_M_l8_8479

open Set

def M : Set ℝ := { x | (2 - x) / (x + 3) < 0 }

theorem complement_of_M : (Mᶜ = { x : ℝ | -3 ≤ x ∧ x ≤ 2 }) :=
by
  sorry

end complement_of_M_l8_8479


namespace find_missing_number_l8_8728

theorem find_missing_number (n : ℤ) (h : 1234562 - n * 3 * 2 = 1234490) : 
  n = 12 :=
by
  sorry

end find_missing_number_l8_8728


namespace course_length_l8_8523

noncomputable def timeBicycling := 12 / 60 -- hours
noncomputable def avgRateBicycling := 30 -- miles per hour
noncomputable def timeRunning := (117 - 12) / 60 -- hours
noncomputable def avgRateRunning := 8 -- miles per hour

theorem course_length : avgRateBicycling * timeBicycling + avgRateRunning * timeRunning = 20 := 
by
  sorry

end course_length_l8_8523


namespace solution1_solution2_l8_8010

-- Definition for problem (1)
def problem1 : ℚ :=
  - (1 ^ 4 : ℚ) - (1 / 6) * (2 - (-3 : ℚ) ^ 2) / (-7 : ℚ)

theorem solution1 : problem1 = -7 / 6 :=
by
  sorry

-- Definition for problem (2)
def problem2 : ℚ :=
  ((3 / 2 : ℚ) - (5 / 8) + (7 / 12)) / (-1 / 24) - 8 * ((-1 / 2 : ℚ) ^ 3)

theorem solution2 : problem2 = -34 :=
by
  sorry

end solution1_solution2_l8_8010


namespace sum_of_extreme_values_eq_four_l8_8983

-- Given conditions in problem statement
variables (x y z : ℝ)
variables (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 8)

-- Statement to be proved: sum of smallest and largest possible values of x is 4
theorem sum_of_extreme_values_eq_four : m + M = 4 :=
sorry

end sum_of_extreme_values_eq_four_l8_8983


namespace count_isosceles_triangles_perimeter_25_l8_8481

theorem count_isosceles_triangles_perimeter_25 : 
  ∃ n : ℕ, (
    n = 6 ∧ 
    (∀ x b : ℕ, 
      2 * x + b = 25 → 
      b < 2 * x → 
      b > 0 →
      ∃ m : ℕ, 
        m = (x - 7) / 5
    ) 
  ) := sorry

end count_isosceles_triangles_perimeter_25_l8_8481


namespace count_odd_three_digit_numbers_l8_8328

open Finset

theorem count_odd_three_digit_numbers : 
  (card {n | ∃ (a b c : ℕ), a ∈ {1, 3, 5} ∧ b ∈ {0, 1, 2, 3, 4, 5} ∧ c ∈ {0, 1, 2, 3, 4, 5} ∧ 
       a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ n = 100 * b + 10 * c + a}) = 48 :=
begin
  -- Conditions have been set for the selection of digits to form a number
  sorry
end

end count_odd_three_digit_numbers_l8_8328


namespace pow_of_729_l8_8923

theorem pow_of_729 : (729 : ℝ) ^ (2 / 3) = 81 :=
by sorry

end pow_of_729_l8_8923


namespace sales_tax_difference_l8_8443

noncomputable def price_before_tax : ℝ := 50
noncomputable def sales_tax_rate_7_5_percent : ℝ := 0.075
noncomputable def sales_tax_rate_8_percent : ℝ := 0.08

theorem sales_tax_difference :
  (price_before_tax * sales_tax_rate_8_percent) - (price_before_tax * sales_tax_rate_7_5_percent) = 0.25 :=
by
  sorry

end sales_tax_difference_l8_8443


namespace tangent_line_parallel_coordinates_l8_8403

theorem tangent_line_parallel_coordinates :
  ∃ (x y : ℝ), y = x^3 + x - 2 ∧ (3 * x^2 + 1 = 4) ∧ (x, y) = (-1, -4) :=
by
  sorry

end tangent_line_parallel_coordinates_l8_8403


namespace ratio_of_sums_l8_8758

theorem ratio_of_sums (a b c d : ℚ) (h1 : b / a = 3) (h2 : d / b = 4) (h3 : c = (a + b) / 2) :
  (a + b + c) / (b + c + d) = 8 / 17 :=
by
  sorry

end ratio_of_sums_l8_8758


namespace smallest_prime_divisor_of_3_pow_19_plus_11_pow_13_l8_8884

/-- 
  Problem statement: Prove that the smallest prime divisor of 3^19 + 11^13 is 2, 
  given the conditions:
   - 3^19 is odd
   - 11^13 is odd
   - The sum of two odd numbers is even
-/

theorem smallest_prime_divisor_of_3_pow_19_plus_11_pow_13 :
  ∃ p : ℕ, prime p ∧ p = 2 ∧ p ∣ (3^19 + 11^13) := 
by
  sorry

end smallest_prime_divisor_of_3_pow_19_plus_11_pow_13_l8_8884


namespace compare_neg_rationals_l8_8304

-- Definition and conditions
def abs_neg_one_third : ℚ := |(-1 / 3 : ℚ)|
def abs_neg_one_fourth : ℚ := |(-1 / 4 : ℚ)|

-- Problem statement
theorem compare_neg_rationals : (-1 : ℚ) / 3 < -1 / 4 :=
by
  -- Including the conditions here, even though they are straightforward implications in Lean
  have h1 : abs_neg_one_third = 1 / 3 := abs_neg_one_third
  have h2 : abs_neg_one_fourth = 1 / 4 := abs_neg_one_fourth
  -- We would include steps to show that -1 / 3 < -1 / 4 using the above facts
  sorry

end compare_neg_rationals_l8_8304


namespace seating_arrangements_family_van_correct_l8_8078

noncomputable def num_seating_arrangements (parents : Fin 2) (children : Fin 3) : Nat :=
  let perm3_2 := Nat.factorial 3 / Nat.factorial (3 - 2)
  2 * 1 * perm3_2

theorem seating_arrangements_family_van_correct :
  num_seating_arrangements 2 3 = 12 :=
by
  sorry

end seating_arrangements_family_van_correct_l8_8078


namespace house_value_l8_8322

open Nat

-- Define the conditions
variables (V x : ℕ)
variables (split_amount money_paid : ℕ)
variables (houses_brothers youngest_received : ℕ)
variables (y1 y2 : ℕ)

-- Hypotheses from the conditions
def conditions (V x split_amount money_paid houses_brothers youngest_received y1 y2 : ℕ) :=
  (split_amount = V / 5) ∧
  (houses_brothers = 3) ∧
  (money_paid = 2000) ∧
  (youngest_received = 3000) ∧
  (3 * houses_brothers * money_paid = 6000) ∧
  (y1 = youngest_received) ∧
  (y2 = youngest_received) ∧
  (3 * x + 6000 = V)

-- Main theorem stating the value of one house
theorem house_value (V x : ℕ) (split_amount money_paid houses_brothers youngest_received y1 y2: ℕ) :
  conditions V x split_amount money_paid houses_brothers youngest_received y1 y2 →
  x = 3000 :=
by
  intros
  simp [conditions] at *
  sorry

end house_value_l8_8322


namespace area_circle_l8_8843

-- Define the given condition
def polar_eq (r θ : ℝ) : Prop :=
  r = 3 * Real.cos θ - 4 * Real.sin θ

-- The goal is to prove the area of the circle described by the polar equation
theorem area_circle {r θ : ℝ} (h : polar_eq r θ) :
  ∃ A, A = π * (5 / 2) ^ 2 :=
sorry

end area_circle_l8_8843


namespace highest_total_zits_l8_8626

def zits_per_student_Swanson := 5
def students_Swanson := 25
def total_zits_Swanson := zits_per_student_Swanson * students_Swanson -- should be 125

def zits_per_student_Jones := 6
def students_Jones := 32
def total_zits_Jones := zits_per_student_Jones * students_Jones -- should be 192

def zits_per_student_Smith := 7
def students_Smith := 20
def total_zits_Smith := zits_per_student_Smith * students_Smith -- should be 140

def zits_per_student_Brown := 8
def students_Brown := 16
def total_zits_Brown := zits_per_student_Brown * students_Brown -- should be 128

def zits_per_student_Perez := 4
def students_Perez := 30
def total_zits_Perez := zits_per_student_Perez * students_Perez -- should be 120

theorem highest_total_zits : 
  total_zits_Jones = max total_zits_Swanson (max total_zits_Smith (max total_zits_Brown (max total_zits_Perez total_zits_Jones))) :=
by
  sorry

end highest_total_zits_l8_8626


namespace avg_distance_is_600_l8_8787

-- Assuming Mickey runs half as many times as Johnny
def num_laps_johnny := 4
def num_laps_mickey := num_laps_johnny / 2
def lap_distance := 200

-- Calculating distances
def distance_johnny := num_laps_johnny * lap_distance
def distance_mickey := num_laps_mickey * lap_distance

-- Total distance run by both Mickey and Johnny
def total_distance := distance_johnny + distance_mickey

-- Average distance run by Johnny and Mickey
def avg_distance := total_distance / 2

-- Prove that the average distance run by Johnny and Mickey is 600 meters
theorem avg_distance_is_600 : avg_distance = 600 :=
by
  sorry

end avg_distance_is_600_l8_8787


namespace triangle_inequality_l8_8177

theorem triangle_inequality (a b c : ℝ) (h : a + b + c = 1) : 
  5 * (a^2 + b^2 + c^2) + 18 * a * b * c ≥ 7 / 3 :=
by
  sorry

end triangle_inequality_l8_8177


namespace movie_ticket_final_price_l8_8734

noncomputable def final_ticket_price (initial_price : ℝ) : ℝ :=
  let price_year_1 := initial_price * 1.12
  let price_year_2 := price_year_1 * 0.95
  let price_year_3 := price_year_2 * 1.08
  let price_year_4 := price_year_3 * 0.96
  let price_year_5 := price_year_4 * 1.06
  let price_after_tax := price_year_5 * 1.07
  let final_price := price_after_tax * 0.90
  final_price

theorem movie_ticket_final_price :
  final_ticket_price 100 = 112.61 := by
  sorry

end movie_ticket_final_price_l8_8734


namespace tom_seashells_l8_8874

theorem tom_seashells (days_at_beach : ℕ) (seashells_per_day : ℕ) (total_seashells : ℕ) 
  (h1 : days_at_beach = 5) (h2 : seashells_per_day = 7) : total_seashells = 35 := 
by 
  sorry

end tom_seashells_l8_8874


namespace sale_in_first_month_l8_8693

theorem sale_in_first_month (sale1 sale2 sale3 sale4 sale5 : ℕ) 
  (h1 : sale1 = 5660) (h2 : sale2 = 6200) (h3 : sale3 = 6350) (h4 : sale4 = 6500) 
  (h_avg : (sale1 + sale2 + sale3 + sale4 + sale5) / 5 = 6000) : 
  sale5 = 5290 := 
by
  sorry

end sale_in_first_month_l8_8693


namespace seeds_in_each_flower_bed_l8_8087

theorem seeds_in_each_flower_bed (total_seeds : ℕ) (flower_beds : ℕ) (h1 : total_seeds = 54) (h2 : flower_beds = 9) : total_seeds / flower_beds = 6 :=
by
  sorry

end seeds_in_each_flower_bed_l8_8087


namespace regular_polygon_sides_eq_seven_l8_8537

theorem regular_polygon_sides_eq_seven (n : ℕ) (h1 : D = n * (n-3) / 2) (h2 : D = 2 * n) : n = 7 := 
by
  sorry

end regular_polygon_sides_eq_seven_l8_8537


namespace fractional_eq_solution_range_l8_8242

theorem fractional_eq_solution_range (m : ℝ) : 
  (∃ x : ℝ, (2 * x - m) / (x + 1) = 3 ∧ x > 0) ↔ m < -3 :=
by
  sorry

end fractional_eq_solution_range_l8_8242


namespace chloe_candies_l8_8074

-- Definitions for the conditions
def lindaCandies : ℕ := 34
def totalCandies : ℕ := 62

-- The statement to prove
theorem chloe_candies :
  (totalCandies - lindaCandies) = 28 :=
by
  -- Proof would go here
  sorry

end chloe_candies_l8_8074


namespace factorial_ratio_value_l8_8666

theorem factorial_ratio_value : fact 15 / (fact 6 * fact 9) = 770 := by
  sorry

end factorial_ratio_value_l8_8666


namespace number_of_readers_who_read_both_l8_8369

theorem number_of_readers_who_read_both (S L B total : ℕ) (hS : S = 250) (hL : L = 550) (htotal : total = 650) (h : S + L - B = total) : B = 150 :=
by {
  /-
  Given:
  S = 250 (number of readers who read science fiction)
  L = 550 (number of readers who read literary works)
  total = 650 (total number of readers)
  h : S + L - B = total (relationship between sets)
  We need to prove: B = 150
  -/
  sorry
}

end number_of_readers_who_read_both_l8_8369


namespace total_frogs_in_ponds_l8_8866

def pondA_frogs := 32
def pondB_frogs := pondA_frogs / 2

theorem total_frogs_in_ponds : pondA_frogs + pondB_frogs = 48 := by
  sorry

end total_frogs_in_ponds_l8_8866


namespace deformable_to_triangle_l8_8286

-- We define a planar polygon with n rods connected by hinges
structure PlanarPolygon (n : ℕ) :=
  (rods : Fin n → ℝ)
  (connections : Fin n → Fin n → Prop)

-- Define the conditions for the rods being rigid and connections (hinges)
def rigid_rod (n : ℕ) : PlanarPolygon n → Prop := λ poly => 
  ∀ i j, poly.connections i j → poly.rods i = poly.rods j

-- Defining the theorem for deformation into a triangle
theorem deformable_to_triangle (n : ℕ) (p : PlanarPolygon n) : 
  (n > 4) ↔ ∃ q : PlanarPolygon 3, true :=
by
  sorry

end deformable_to_triangle_l8_8286


namespace number_of_bananas_in_bowl_l8_8640

theorem number_of_bananas_in_bowl (A P B : Nat) (h1 : P = A + 2) (h2 : B = P + 3) (h3 : A + P + B = 19) : B = 9 :=
sorry

end number_of_bananas_in_bowl_l8_8640


namespace average_hit_targets_value_average_hit_targets_ge_half_l8_8715

noncomputable def average_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - 1 / n)^n)

theorem average_hit_targets_value (n : ℕ) :
  average_hit_targets n = n * (1 - (1 - 1 / n)^n) :=
by sorry

theorem average_hit_targets_ge_half (n : ℕ) :
  average_hit_targets n >= n / 2 :=
by sorry

end average_hit_targets_value_average_hit_targets_ge_half_l8_8715


namespace total_profit_is_42000_l8_8893

noncomputable def total_profit (I_B T_B : ℝ) :=
  let I_A := 3 * I_B
  let T_A := 2 * T_B
  let profit_B := I_B * T_B
  let profit_A := I_A * T_A
  profit_A + profit_B

theorem total_profit_is_42000
  (I_B T_B : ℝ)
  (h1 : I_A = 3 * I_B)
  (h2 : T_A = 2 * T_B)
  (h3 : I_B * T_B = 6000) :
  total_profit I_B T_B = 42000 := by
  sorry

end total_profit_is_42000_l8_8893


namespace bus_speed_including_stoppages_l8_8584

theorem bus_speed_including_stoppages
  (speed_excluding_stoppages : ℝ)
  (stoppage_time_per_hour : ℝ) :
  speed_excluding_stoppages = 64 ∧ stoppage_time_per_hour = 15 / 60 →
  (44 / 60) * speed_excluding_stoppages = 48 :=
by
  sorry

end bus_speed_including_stoppages_l8_8584


namespace binary_to_base4_conversion_l8_8711

theorem binary_to_base4_conversion : 
  let binary := (1*2^7 + 1*2^6 + 0*2^5 + 1*2^4 + 1*2^3 + 0*2^2 + 0*2^1 + 1*2^0) 
  let base4 := (3*4^3 + 1*4^2 + 2*4^1 + 1*4^0)
  binary = base4 := by
  sorry

end binary_to_base4_conversion_l8_8711


namespace nathan_final_temperature_l8_8988

theorem nathan_final_temperature : ∃ (final_temp : ℝ), final_temp = 77.4 :=
  let initial_temp : ℝ := 50
  let type_a_increase : ℝ := 2
  let type_b_increase : ℝ := 3.5
  let type_c_increase : ℝ := 4.8
  let type_d_increase : ℝ := 7.2
  let type_a_quantity : ℚ := 6
  let type_b_quantity : ℚ := 5
  let type_c_quantity : ℚ := 9
  let type_d_quantity : ℚ := 3
  let temp_after_a := initial_temp + 3 * type_a_increase
  let temp_after_b := temp_after_a + 2 * type_b_increase
  let temp_after_c := temp_after_b + 3 * type_c_increase
  let final_temp := temp_after_c
  ⟨final_temp, sorry⟩

end nathan_final_temperature_l8_8988


namespace average_distance_is_600_l8_8792

-- Definitions based on the given conditions
def distance_around_block := 200
def johnny_rounds := 4
def mickey_rounds := johnny_rounds / 2

-- The calculated distances
def johnny_distance := johnny_rounds * distance_around_block
def mickey_distance := mickey_rounds * distance_around_block

-- The average distance computation
def average_distance := (johnny_distance + mickey_distance) / 2

-- The theorem to prove that the average distance is 600 meters
theorem average_distance_is_600 : average_distance = 600 := by sorry

end average_distance_is_600_l8_8792


namespace range_of_a_analytical_expression_l8_8181

variables {f : ℝ → ℝ}

-- Problem 1
theorem range_of_a (h_odd : ∀ x, f (-x) = -f x)
  (h_mono : ∀ x y, x < y → f x ≥ f y)
  {a : ℝ} (h_ineq : f (1 - a) + f (1 - 2 * a) < 0) :
  0 < a ∧ a ≤ 2 / 3 :=
sorry

-- Problem 2
theorem analytical_expression 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x, 0 < x ∧ x < 1 → f x = x^2 + x + 1)
  (h_zero : f 0 = 0) :
  ∀ x : ℝ, -1 < x ∧ x < 1 → f x = 
    if x > 0 then x^2 + x + 1
    else if x = 0 then 0
    else -x^2 + x - 1 :=
sorry

end range_of_a_analytical_expression_l8_8181


namespace area_circle_l8_8844

-- Define the given condition
def polar_eq (r θ : ℝ) : Prop :=
  r = 3 * Real.cos θ - 4 * Real.sin θ

-- The goal is to prove the area of the circle described by the polar equation
theorem area_circle {r θ : ℝ} (h : polar_eq r θ) :
  ∃ A, A = π * (5 / 2) ^ 2 :=
sorry

end area_circle_l8_8844


namespace isosceles_triangle_has_perimeter_22_l8_8909

noncomputable def isosceles_triangle_perimeter (a b : ℕ) : ℕ :=
if a + a > b ∧ a + b > a ∧ b + b > a then a + a + b else 0

theorem isosceles_triangle_has_perimeter_22 :
  isosceles_triangle_perimeter 9 4 = 22 :=
by 
  -- Add a note for clarity; this will be completed via 'sorry'
  -- Prove that with side lengths 9 and 4 (with 9 being the equal sides),
  -- they form a valid triangle and its perimeter is 22
  sorry

end isosceles_triangle_has_perimeter_22_l8_8909


namespace find_number_l8_8676

theorem find_number (x : ℝ) (h : 3 * (2 * x + 5) = 129) : x = 19 :=
by
  sorry

end find_number_l8_8676


namespace trigonometric_inequality_l8_8803

theorem trigonometric_inequality (x : ℝ) : 
  -1/3 ≤ (6 * Real.cos x + Real.sin x - 5) / (2 * Real.cos x - 3 * Real.sin x - 5) ∧
  (6 * Real.cos x + Real.sin x - 5) / (2 * Real.cos x - 3 * Real.sin x - 5) ≤ 3 := 
sorry

end trigonometric_inequality_l8_8803


namespace value_of_f_csc_squared_l8_8943

noncomputable def f (x : ℝ) : ℝ := if x ≠ 0 ∧ x ≠ 1 then 1 / x else 0

lemma csc_sq_identity (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) : 
  (f (x / (x - 1)) = 1 / x) := 
  by sorry

theorem value_of_f_csc_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π / 2) :
  f ((1 / (Real.sin t) ^ 2)) = - (Real.cos t) ^ 2 :=
  by sorry

end value_of_f_csc_squared_l8_8943


namespace binary_to_decimal_l8_8571

theorem binary_to_decimal (b : ℕ) (h : b = 2^3 + 2^2 + 0 * 2^1 + 2^0) : b = 13 :=
by {
  -- proof is omitted
  sorry
}

end binary_to_decimal_l8_8571


namespace problem_I_problem_II_l8_8948

theorem problem_I (a b p : ℝ) (F_2 M : ℝ × ℝ)
(h1 : a > b) (h2 : b > 0) (h3 : p > 0)
(h4 : (F_2.1)^2 / a^2 + (F_2.2)^2 / b^2 = 1)
(h5 : M.2^2 = 2 * p * M.1)
(h6 : M.1 = abs (M.2 - F_2.2) - 1)
(h7 : (|F_2.1 - 1|) = 5 / 2) :
    p = 2 ∧ ∃ f : ℝ × ℝ, (f.1)^2 / 9 + (f.2)^2 / 8 = 1 := sorry

theorem problem_II (k m x_0 : ℝ) 
(h8 : k ≠ 0) 
(h9 : m ≠ 0) 
(h10 : km = 1) 
(h11: ∃ A B : ℝ × ℝ, A ≠ B ∧
    (A.2 = k * A.1 + m ∧ B.2 = k * B.1 + m) ∧
    ((A.1)^2 / 9 + (A.2)^2 / 8 = 1) ∧
    ((B.1)^2 / 9 + (B.2)^2 / 8 = 1) ∧
    (x_0 = (A.1 + B.1) / 2)) :
  -1 < x_0 ∧ x_0 < 0 := sorry

end problem_I_problem_II_l8_8948


namespace tallest_giraffe_height_l8_8860

theorem tallest_giraffe_height :
  ∃ (height : ℕ), height = 96 ∧ (height = 68 + 28) := by
  sorry

end tallest_giraffe_height_l8_8860


namespace number_equals_14_l8_8358

theorem number_equals_14 (n : ℕ) (h1 : 2^n - 2^(n-2) = 3 * 2^12) (h2 : n = 14) : n = 14 := 
by 
  sorry

end number_equals_14_l8_8358


namespace fencing_cost_l8_8025

noncomputable def diameter : ℝ := 14
noncomputable def cost_per_meter : ℝ := 2.50
noncomputable def pi := Real.pi

noncomputable def circumference (d : ℝ) : ℝ := pi * d

noncomputable def total_cost (c : ℝ) (r : ℝ) : ℝ := r * c

theorem fencing_cost : total_cost (circumference diameter) cost_per_meter = 109.95 := by
  sorry

end fencing_cost_l8_8025


namespace alyssa_hike_total_distance_l8_8004

theorem alyssa_hike_total_distance
  (e f g h i : ℝ)
  (h1 : e + f + g = 40)
  (h2 : f + g + h = 48)
  (h3 : g + h + i = 54)
  (h4 : e + h = 30) :
  e + f + g + h + i = 118 :=
by
  sorry

end alyssa_hike_total_distance_l8_8004


namespace salary_increase_after_five_years_l8_8987

theorem salary_increase_after_five_years (S : ℝ) : 
  let final_salary := S * (1.12)^5
  let increase := final_salary - S
  let percent_increase := (increase / S) * 100
  percent_increase = 76.23 :=
by
  let final_salary := S * (1.12)^5
  let increase := final_salary - S
  let percent_increase := (increase / S) * 100
  sorry

end salary_increase_after_five_years_l8_8987


namespace sum_of_coefficients_l8_8398

theorem sum_of_coefficients :
  ∃ (a b c d e : ℤ), (512 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ (a + b + c + d + e = 60) :=
by
  sorry

end sum_of_coefficients_l8_8398


namespace test_total_points_l8_8697

def total_points (total_problems comp_problems : ℕ) (points_comp points_word : ℕ) : ℕ :=
  let word_problems := total_problems - comp_problems
  (comp_problems * points_comp) + (word_problems * points_word)

theorem test_total_points :
  total_points 30 20 3 5 = 110 := by
  sorry

end test_total_points_l8_8697


namespace decreasing_f_l8_8031

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≥ 1 then -4 * x + 2 * a else x^2 - a * x + 4

theorem decreasing_f (a : ℝ) : (∀ x, f a x = (if x ≥ 1 then -4 * x + 2 * a else x^2 - a * x + 4)) →
  (∀ x, differentiable_at ℝ (f a) x) →
  (∀ x, deriv (f a) x < 0) ↔ (2 < a ∧ a ≤ 3) :=
sorry

end decreasing_f_l8_8031


namespace max_halls_visited_l8_8261

theorem max_halls_visited (side_len large_tri small_tri: ℕ) 
  (h1 : side_len = 100)
  (h2 : large_tri = 100)
  (h3 : small_tri = 10)
  (div : large_tri = (side_len / small_tri) ^ 2) :
  ∃ m : ℕ, m = 91 → m ≤ large_tri - 9 := 
sorry

end max_halls_visited_l8_8261


namespace hyperbola_asymptote_product_l8_8962

theorem hyperbola_asymptote_product (k1 k2 : ℝ) (h1 : k1 = 1) (h2 : k2 = -1) :
  k1 * k2 = -1 :=
by
  rw [h1, h2]
  norm_num

end hyperbola_asymptote_product_l8_8962


namespace factorial_division_l8_8664

-- Define factorial using the standard library's factorial function
def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- The problem statement
theorem factorial_division :
  (factorial 15) / (factorial 6 * factorial 9) = 834 :=
sorry

end factorial_division_l8_8664


namespace foldable_cube_with_one_face_missing_l8_8636

-- Definitions for the conditions
structure Square where
  -- You can define properties of a square here if necessary

structure Polygon where
  squares : List Square
  congruent : True -- All squares are congruent
  joined_edge_to_edge : True -- The squares are joined edge-to-edge

-- The positions the additional square can be added to
inductive Position
| P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9

-- Define the problem in Lean 4 as a theorem
theorem foldable_cube_with_one_face_missing (base_polygon : Polygon) :
  base_polygon.squares.length = 4 →
  ∃ (positions : List Position), positions.length = 6 ∧
    ∀ pos ∈ positions, 
      let new_polygon := { base_polygon with squares := base_polygon.squares.append [Square.mk] }
      new_polygon.foldable_into_cube_with_one_face_missing pos :=
  sorry

end foldable_cube_with_one_face_missing_l8_8636


namespace polynomial_remainder_l8_8942
-- Importing the broader library needed

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^5 + 2 * x^2 + 3

-- The statement of the theorem
theorem polynomial_remainder :
  p 2 = 43 :=
sorry

end polynomial_remainder_l8_8942


namespace least_integer_x_l8_8320

theorem least_integer_x (x : ℤ) : (2 * |x| + 7 < 17) → x = -4 := by
  sorry

end least_integer_x_l8_8320


namespace find_x_l8_8133

theorem find_x (x : ℝ) (h : 49 / x = 700) : x = 0.07 :=
sorry

end find_x_l8_8133


namespace games_played_by_third_player_l8_8655

theorem games_played_by_third_player
    (games_first : ℕ)
    (games_second : ℕ)
    (games_first_eq : games_first = 10)
    (games_second_eq : games_second = 21) :
    ∃ (games_third : ℕ), games_third = 11 := by
  sorry

end games_played_by_third_player_l8_8655


namespace simplify_expression_l8_8224

-- Define general term for y
variable (y : ℤ)

-- Statement representing the given proof problem
theorem simplify_expression :
  4 * y + 5 * y + 6 * y + 2 = 15 * y + 2 := 
sorry

end simplify_expression_l8_8224


namespace probability_of_closer_to_D_in_triangle_DEF_l8_8974

noncomputable def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  0.5 * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

theorem probability_of_closer_to_D_in_triangle_DEF :
  let D := (0, 0)
  let E := (0, 6)
  let F := (8, 0)
  let M := ((D.1 + F.1) / 2, (D.2 + F.2) / 2)
  let N := ((D.1 + E.1) / 2, (D.2 + E.2) / 2)
  let area_DEF := triangle_area D E F
  let area_DMN := triangle_area D M N
  area_DMN / area_DEF = 1 / 4 := by
    sorry

end probability_of_closer_to_D_in_triangle_DEF_l8_8974


namespace cubics_of_sum_and_product_l8_8762

theorem cubics_of_sum_and_product (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) : 
  x^3 + y^3 = 640 :=
by
  sorry

end cubics_of_sum_and_product_l8_8762


namespace find_A_d_minus_B_d_l8_8811

-- Definitions for the proof problem
variables {d A B : ℕ}
variables {ad bd : ℤ} -- Representing A_d and B_d in ℤ for arithmetic operations

-- Conditions
constants (base_check : d > 7)
constants (digit_check : A < d ∧ B < d)
constants (encoded_check : d^1 * A + d^0 * B + d^2 * A + d^1 * A = 1 * d^2 + 7 * d^1 + 2 * d^0)

-- The theorem to prove
theorem find_A_d_minus_B_d : A - B = 5 :=
good sorry

end find_A_d_minus_B_d_l8_8811


namespace square_side_length_l8_8486

theorem square_side_length (A : ℝ) (side : ℝ) (h₁ : A = side^2) (h₂ : A = 12) : side = 2 * Real.sqrt 3 := 
by
  sorry

end square_side_length_l8_8486


namespace temperature_conversion_l8_8606

theorem temperature_conversion :
  ∀ (k t : ℝ),
    (t = (5 / 9) * (k - 32) ∧ k = 95) →
    t = 35 := by
  sorry

end temperature_conversion_l8_8606


namespace find_other_vertices_l8_8396

theorem find_other_vertices
  (A : ℝ × ℝ) (B C : ℝ × ℝ)
  (S : ℝ × ℝ) (M : ℝ × ℝ)
  (hA : A = (7, 3))
  (hS : S = (5, -5 / 3))
  (hM : M = (3, -1))
  (h_centroid : S = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) 
  (h_orthocenter : ∀ u v : ℝ × ℝ, u ≠ v → u - v = (4, 4) → (u - v) • (C - B) = 0) :
  B = (1, -1) ∧ C = (7, -7) :=
sorry

end find_other_vertices_l8_8396


namespace max_k_value_condition_l8_8362

theorem max_k_value_condition (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  ∃ k, k = 100 ∧ (∀ k < 100, ∃ (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c), 
   (k * a * b * c / (a + b + c) <= (a + b)^2 + (a + b + 4 * c)^2)) :=
sorry

end max_k_value_condition_l8_8362


namespace arcs_intersection_l8_8588

theorem arcs_intersection (k : ℕ) : (1 ≤ k ∧ k ≤ 99) ∧ ¬(∃ m : ℕ, k + 1 = 8 * m) ↔ ∃ n l : ℕ, (2 * l + 1) * 100 = (k + 1) * n ∧ n = 100 ∧ k < 100 := by
  sorry

end arcs_intersection_l8_8588


namespace ratio_of_squares_l8_8240

theorem ratio_of_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a / b = 1 / 3) :
  (4 * a / (4 * b) = 1 / 3) ∧ (a * a / (b * b) = 1 / 9) :=
by
  sorry

end ratio_of_squares_l8_8240


namespace shortest_chord_through_point_on_circle_l8_8161

theorem shortest_chord_through_point_on_circle :
  ∀ (M : ℝ × ℝ) (x y : ℝ),
    M = (3, 0) →
    x^2 + y^2 - 8 * x - 2 * y + 10 = 0 →
    ∃ (a b c : ℝ), a * x + b * y + c = 0 ∧ a = 1 ∧ b = 1 ∧ c = -3 :=
by
  sorry

end shortest_chord_through_point_on_circle_l8_8161


namespace factorial_ratio_value_l8_8667

theorem factorial_ratio_value : fact 15 / (fact 6 * fact 9) = 770 := by
  sorry

end factorial_ratio_value_l8_8667


namespace math_problem_l8_8506

-- Definitions for increasing function and periodic function
def increasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x ≤ f y
def periodic (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x : ℝ, f (x + T) = f x

-- The main theorem statement
theorem math_problem (f g h : ℝ → ℝ) (T : ℝ) :
  (∀ x y : ℝ, x < y → f x + g x ≤ f y + g y) ∧ (∀ x y : ℝ, x < y → f x + h x ≤ f y + h y) ∧ (∀ x y : ℝ, x < y → g x + h x ≤ g y + h y) → 
  ¬(increasing g) ∧
  (∀ x : ℝ, f (x + T) + g (x + T) = f x + g x ∧ f (x + T) + h (x + T) = f x + h x ∧ g (x + T) + h (x + T) = g x + h x) → 
  increasing f ∧ increasing g ∧ increasing h :=
sorry

end math_problem_l8_8506


namespace total_branches_in_pine_tree_l8_8000

-- Definitions based on the conditions
def middle_branch : ℕ := 0 -- arbitrary assignment to represent the middle branch

def jumps_up_5 (b : ℕ) : ℕ := b + 5
def jumps_down_7 (b : ℕ) : ℕ := b - 7
def jumps_up_4 (b : ℕ) : ℕ := b + 4
def jumps_up_9 (b : ℕ) : ℕ := b + 9

-- The statement to be proven
theorem total_branches_in_pine_tree : 
  (jumps_up_9 (jumps_up_4 (jumps_down_7 (jumps_up_5 middle_branch))) = 11) →
  ∃ n, n = 23 :=
by
  sorry

end total_branches_in_pine_tree_l8_8000


namespace average_first_set_eq_3_more_than_second_set_l8_8098

theorem average_first_set_eq_3_more_than_second_set (x : ℤ) :
  let avg_first_set := (14 + 32 + 53) / 3
  let avg_second_set := (x + 47 + 22) / 3
  avg_first_set = avg_second_set + 3 → x = 21 := by
  sorry

end average_first_set_eq_3_more_than_second_set_l8_8098


namespace largest_sum_36_l8_8878

theorem largest_sum_36 : ∃ n : ℕ, ∃ a : ℕ, (n * a + (n * (n - 1)) / 2 = 36) ∧ ∀ m : ℕ, (m * a + (m * (m - 1)) / 2 = 36) → m ≤ 8 :=
by
  sorry

end largest_sum_36_l8_8878


namespace find_f_five_l8_8189

noncomputable def f (x : ℝ) (y : ℝ) : ℝ := 2 * x^2 + y

theorem find_f_five (y : ℝ) (h : f 2 y = 50) : f 5 y = 92 := by
  sorry

end find_f_five_l8_8189


namespace arccos_cos_8_eq_1_point_72_l8_8568

noncomputable def arccos_cos_eight : Real :=
  Real.arccos (Real.cos 8)

theorem arccos_cos_8_eq_1_point_72 : arccos_cos_eight = 1.72 :=
by
  sorry

end arccos_cos_8_eq_1_point_72_l8_8568


namespace zander_construction_cost_l8_8085

noncomputable def cost_of_cement (num_bags : ℕ) (price_per_bag : ℕ) : ℕ :=
  num_bags * price_per_bag

noncomputable def amount_of_sand (num_lorries : ℕ) (tons_per_lorry : ℕ) : ℕ :=
  num_lorries * tons_per_lorry

noncomputable def cost_of_sand (total_tons : ℕ) (price_per_ton : ℕ) : ℕ :=
  total_tons * price_per_ton

noncomputable def total_cost (cost_cement : ℕ) (cost_sand : ℕ) : ℕ :=
  cost_cement + cost_sand

theorem zander_construction_cost :
  total_cost (cost_of_cement 500 10) (cost_of_sand (amount_of_sand 20 10) 40) = 13000 :=
by
  sorry

end zander_construction_cost_l8_8085


namespace even_function_a_zero_l8_8347

noncomputable def f (x a : ℝ) : ℝ := (x + a) * real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) →
  (2 * x - 1) / (2 * x + 1) > 0 → 
  x > 1 / 2 ∨ x < -1 / 2 →
  a = 0 :=
by {
  sorry
}

end even_function_a_zero_l8_8347


namespace find_sin_θ_find_cos_2θ_find_cos_φ_l8_8591

noncomputable def θ : ℝ := sorry
noncomputable def φ : ℝ := sorry

-- Conditions
axiom cos_eq : Real.cos θ = Real.sqrt 5 / 5
axiom θ_in_quadrant_I : 0 < θ ∧ θ < Real.pi / 2
axiom sin_diff_eq : Real.sin (θ - φ) = Real.sqrt 10 / 10
axiom φ_in_quadrant_I : 0 < φ ∧ φ < Real.pi / 2

-- Goals
-- Part (I) Prove the value of sin θ
theorem find_sin_θ : Real.sin θ = 2 * Real.sqrt 5 / 5 :=
by
  sorry

-- Part (II) Prove the value of cos 2θ
theorem find_cos_2θ : Real.cos (2 * θ) = -3 / 5 :=
by
  sorry

-- Part (III) Prove the value of cos φ
theorem find_cos_φ : Real.cos φ = Real.sqrt 2 / 2 :=
by
  sorry

end find_sin_θ_find_cos_2θ_find_cos_φ_l8_8591


namespace abs_neg_two_l8_8825

theorem abs_neg_two : abs (-2) = 2 := by
  sorry

end abs_neg_two_l8_8825


namespace does_not_balance_l8_8253

variables (square odot circ triangle O : ℝ)

-- Conditions represented as hypothesis
def condition1 : Prop := 4 * square = odot + circ
def condition2 : Prop := 2 * circ + odot = 2 * triangle

-- Statement to be proved
theorem does_not_balance (h1 : condition1 square odot circ) (h2 : condition2 circ odot triangle)
 : ¬(2 * triangle + square = triangle + odot + square) := 
sorry

end does_not_balance_l8_8253


namespace elberta_has_22_dollars_l8_8049

theorem elberta_has_22_dollars (granny_smith : ℝ) (anjou : ℝ) (elberta : ℝ) 
  (h1 : granny_smith = 75) 
  (h2 : anjou = granny_smith / 4)
  (h3 : elberta = anjou + 3) : 
  elberta = 22 := 
by
  sorry

end elberta_has_22_dollars_l8_8049


namespace grandmother_age_five_times_lingling_l8_8541

theorem grandmother_age_five_times_lingling (x : ℕ) :
  let lingling_age := 8
  let grandmother_age := 60
  (grandmother_age + x = 5 * (lingling_age + x)) ↔ (x = 5) := by
  sorry

end grandmother_age_five_times_lingling_l8_8541


namespace contradiction_proof_l8_8174

theorem contradiction_proof (x y : ℝ) (h1 : x + y < 2) (h2 : 1 < x) (h3 : 1 < y) : false := 
by 
  sorry

end contradiction_proof_l8_8174


namespace number_of_federal_returns_sold_l8_8531

/-- Given conditions for revenue calculations at the Kwik-e-Tax Center -/
structure TaxCenter where
  price_federal : ℕ
  price_state : ℕ
  price_quarterly : ℕ
  num_state : ℕ
  num_quarterly : ℕ
  total_revenue : ℕ

/-- The specific instance of the TaxCenter for this problem -/
def KwikETaxCenter : TaxCenter :=
{ price_federal := 50,
  price_state := 30,
  price_quarterly := 80,
  num_state := 20,
  num_quarterly := 10,
  total_revenue := 4400 }

/-- Proof statement for the number of federal returns sold -/
theorem number_of_federal_returns_sold (F : ℕ) :
  KwikETaxCenter.price_federal * F + 
  KwikETaxCenter.price_state * KwikETaxCenter.num_state + 
  KwikETaxCenter.price_quarterly * KwikETaxCenter.num_quarterly = 
  KwikETaxCenter.total_revenue → 
  F = 60 :=
by
  intro h
  /- Proof is skipped -/
  sorry

end number_of_federal_returns_sold_l8_8531


namespace farm_needs_12880_ounces_of_horse_food_per_day_l8_8441

-- Define the given conditions
def ratio_sheep_to_horses : ℕ × ℕ := (1, 7)
def food_per_horse_per_day : ℕ := 230
def number_of_sheep : ℕ := 8

-- Define the proof goal
theorem farm_needs_12880_ounces_of_horse_food_per_day :
  let number_of_horses := number_of_sheep * ratio_sheep_to_horses.2
  number_of_horses * food_per_horse_per_day = 12880 :=
by
  sorry

end farm_needs_12880_ounces_of_horse_food_per_day_l8_8441


namespace conic_section_union_l8_8644

theorem conic_section_union : 
  ∀ (y x : ℝ), y^4 - 6*x^4 = 3*y^2 - 2 → 
  ( ( y^2 - 3*x^2 = 1 ∨ y^2 - 2*x^2 = 1 ) ∧ 
    ( y^2 - 2*x^2 = 2 ∨ y^2 - 3*x^2 = 2 ) ) :=
by
  sorry

end conic_section_union_l8_8644


namespace correct_statements_l8_8259

theorem correct_statements (
    A B : Prop)
  (X : ℝ → ℝ)
  (σ : ℝ)
  (h_dist : ∃ (μ = 1), ∀ x, X x = PDF_normal μ σ)
  (h_prob_X_gt_2 : P(X > 2) = 0.2) 
  (r : ℝ) 
  (h_corr_bound : -1 ≤ r ∧ r ≤ 1)
  (h_corr_strong : abs r → ℝ → ℝ ∈ set.Icc -1 1)
  (is_complementary : A ∧ B → ¬ (A ∧ B) ∧ A ∨ B = univ)
  (is_mutually_exclusive : A ∧ B → ¬ (A ∧ B)) :
  (is_complementary → is_mutually_exclusive) ∧
  h_prob_X_gt_2 ∧ P(0 < X < 1) = 0.3 ∧
  (∀ x y, (strong_corr : corr(Pair(x, y)) = r) → ∃ c, abs(r) = c ∧ c ∈ (set.Ico 0 1)) := sorry

end correct_statements_l8_8259


namespace total_visitors_over_two_days_l8_8430

constant visitors_saturday : ℕ := 200
constant additional_visitors_sunday : ℕ := 40

def visitors_sunday : ℕ := visitors_saturday + additional_visitors_sunday
def total_visitors : ℕ := visitors_saturday + visitors_sunday

theorem total_visitors_over_two_days : total_visitors = 440 := by
  -- Proof goes here...
  sorry

end total_visitors_over_two_days_l8_8430


namespace abs_neg_two_l8_8826

def abs (x : ℤ) : ℤ :=
  if x ≥ 0 then x else -x

theorem abs_neg_two : abs (-2) = 2 :=
by
  sorry

end abs_neg_two_l8_8826


namespace farmer_profit_l8_8426

-- Define the conditions and relevant information
def feeding_cost_per_month_per_piglet : ℕ := 12
def number_of_piglets : ℕ := 8

def selling_details : List (ℕ × ℕ × ℕ) :=
[
  (2, 350, 12),
  (3, 400, 15),
  (2, 450, 18),
  (1, 500, 21)
]

-- Calculate total revenue
def total_revenue : ℕ :=
selling_details.foldl (λ acc (piglets, price, _) => acc + piglets * price) 0

-- Calculate total feeding cost
def total_feeding_cost : ℕ :=
selling_details.foldl (λ acc (piglets, _, months) => acc + piglets * feeding_cost_per_month_per_piglet * months) 0

-- Calculate profit
def profit : ℕ := total_revenue - total_feeding_cost

-- Statement of the theorem
theorem farmer_profit : profit = 1788 := by
  sorry

end farmer_profit_l8_8426


namespace kim_money_l8_8980

theorem kim_money (S P K A : ℝ) (h1 : K = 1.40 * S) (h2 : S = 0.80 * P) (h3 : A = 1.25 * (S + K)) (h4 : S + P + A = 3.60) : K = 0.96 :=
by
  sorry

end kim_money_l8_8980


namespace inf_pos_integers_n_sum_two_squares_l8_8093

theorem inf_pos_integers_n_sum_two_squares:
  ∃ (s : ℕ → ℕ), (∀ (k : ℕ), ∃ (a₁ b₁ a₂ b₂ : ℕ),
   a₁ > 0 ∧ b₁ > 0 ∧ a₂ > 0 ∧ b₂ > 0 ∧ s k = n ∧
   n = a₁^2 + b₁^2 ∧ n = a₂^2 + b₂^2 ∧ 
  (a₁ ≠ a₂ ∨ b₁ ≠ b₂)) := sorry

end inf_pos_integers_n_sum_two_squares_l8_8093


namespace expected_value_ξ_l8_8249

noncomputable theory
open_locale big_operators

-- Definitions of the conditions used in the problem
def total_balls := 5
def new_balls := 3
def old_balls := 2
def balls_picked := 2

-- Definition of ξ (ξ is the number of new balls picked in the second match - formal introduction)
def ξ : ℕ → ℝ := λ k, if k = 0 then 0 else if k = 1 then 1 else 2

-- Main theorem
theorem expected_value_ξ : 
  (∑ x in ({0, 1, 2} : finset ℕ), (ξ x * ((choose new_balls x * choose old_balls (balls_picked - x))/choose total_balls balls_picked))) = 18 / 25 :=
sorry

end expected_value_ξ_l8_8249


namespace cos_A_eq_neg_quarter_l8_8964

-- Definitions of angles and sides in the triangle
variables (A B C : ℝ)
variables (a b c : ℝ)

-- Conditions from the math problem
axiom sin_arithmetic_sequence : 2 * Real.sin B = Real.sin A + Real.sin C
axiom side_relation : a = 2 * c

-- Question to be proved as Lean 4 statement
theorem cos_A_eq_neg_quarter (h1 : ∀ {x y z : ℝ}, 2 * y = x + z) 
                              (h2 : ∀ {a b c : ℝ}, a = 2 * c) : 
                              Real.cos A = -1/4 := 
sorry

end cos_A_eq_neg_quarter_l8_8964


namespace winner_vote_count_l8_8768

theorem winner_vote_count (total_votes : ℕ) (x y z w : ℕ) 
  (h1 : total_votes = 5219000) 
  (h2 : y + 22000 = x)
  (h3 : z + 30000 = x)
  (h4 : w + 73000 = x)
  (h5 : x + y + z + w = total_votes) :
  x = 1336000 :=
by
  sorry

end winner_vote_count_l8_8768


namespace circle_area_l8_8839

theorem circle_area (θ : ℝ) : 
  let r := 3 * Real.cos θ - 4 * Real.sin θ in
  let x := r * Real.cos θ in
  let y := r * Real.sin θ in
  let center := (3 / 2 : ℝ, -2 : ℝ) in
  let radius := 5 / 2 in
  (x - center.1)^2 + (y - center.2)^2 = radius^2 →
  ∃ A : ℝ, A = (radius^2) * Real.pi ∧ A = 25 * Real.pi / 4 := 
by sorry

end circle_area_l8_8839


namespace fraction_ratio_l8_8125

theorem fraction_ratio (x : ℚ) (h1 : 2 / 5 / (3 / 7) = x / (1 / 2)) :
  x = 7 / 15 :=
by {
  -- Proof omitted
  sorry
}

end fraction_ratio_l8_8125


namespace area_of_shaded_rectangle_l8_8251

-- Definition of side length of the squares
def side_length : ℕ := 12

-- Definition of the dimensions of the overlapped rectangle
def rectangle_length : ℕ := 20
def rectangle_width : ℕ := side_length

-- Theorem stating the area of the shaded rectangle PBCS
theorem area_of_shaded_rectangle
  (squares_identical : ∀ (a b c d p q r s : ℕ),
    a = side_length → b = side_length →
    p = side_length → q = side_length →
    rectangle_width * (rectangle_length - side_length) = 48) :
  rectangle_width * (rectangle_length - side_length) = 48 :=
by sorry -- Proof omitted

end area_of_shaded_rectangle_l8_8251


namespace exists_infinitely_many_primes_dividing_fib_l8_8204

theorem exists_infinitely_many_primes_dividing_fib (u : ℕ → ℕ) (h0 : u 0 = 0) (h1 : u 1 = 1)
  (hn : ∀ n > 1, u n = u (n-1) + u (n-2)) :
  ∃ᶠ p in Filter.atTop, Prime p ∧ p ∣ u (p - 1) := 
sorry

end exists_infinitely_many_primes_dividing_fib_l8_8204


namespace measure_of_alpha_l8_8879

theorem measure_of_alpha
  (A B D α : ℝ)
  (hA : A = 50)
  (hB : B = 150)
  (hD : D = 140)
  (quadrilateral_sum : A + B + D + α = 360) : α = 20 :=
by
  rw [hA, hB, hD] at quadrilateral_sum
  sorry

end measure_of_alpha_l8_8879


namespace paving_stone_width_l8_8113

theorem paving_stone_width 
    (length_courtyard : ℝ)
    (width_courtyard : ℝ)
    (length_paving_stone : ℝ)
    (num_paving_stones : ℕ)
    (total_area_courtyard : ℝ)
    (total_area_paving_stones : ℝ)
    (width_paving_stone : ℝ)
    (h1 : length_courtyard = 20)
    (h2 : width_courtyard = 16.5)
    (h3 : length_paving_stone = 2.5)
    (h4 : num_paving_stones = 66)
    (h5 : total_area_courtyard = length_courtyard * width_courtyard)
    (h6 : total_area_paving_stones = num_paving_stones * (length_paving_stone * width_paving_stone))
    (h7 : total_area_courtyard = total_area_paving_stones) :
    width_paving_stone = 2 :=
by
  sorry

end paving_stone_width_l8_8113


namespace problem_l8_8813

/-- Given a number d > 7, 
    digits A and B in base d such that the equation AB_d + AA_d = 172_d holds, 
    we want to prove that A_d - B_d = 5. --/

theorem problem (d A B : ℕ) (hd : 7 < d)
  (hAB : d * A + B + d * A + A = 1 * d^2 + 7 * d + 2) : A - B = 5 :=
sorry

end problem_l8_8813


namespace smallest_value_of_reciprocal_sums_l8_8445

theorem smallest_value_of_reciprocal_sums (r1 r2 s p : ℝ) 
  (h1 : r1 + r2 = s)
  (h2 : r1^2 + r2^2 = s)
  (h3 : r1^3 + r2^3 = s)
  (h4 : r1^4 + r2^4 = s)
  (h1004 : r1^1004 + r2^1004 = s)
  (h_r1_r2_roots : ∀ x, x^2 - s * x + p = 0) :
  (1 / r1^1005 + 1 / r2^1005) = 2 :=
by
  sorry

end smallest_value_of_reciprocal_sums_l8_8445


namespace find_smaller_number_l8_8940

-- Define the two numbers such that one is 3 times the other
def numbers (x : ℝ) := (x, 3 * x)

-- Define the condition that the sum of the two numbers is 14
def sum_condition (x y : ℝ) : Prop := x + y = 14

-- The theorem we want to prove
theorem find_smaller_number (x : ℝ) (hx : sum_condition x (3 * x)) : x = 3.5 :=
by
  -- Proof goes here
  sorry

end find_smaller_number_l8_8940


namespace jimmy_change_l8_8616

def cost_of_pens (num_pens : ℕ) (cost_per_pen : ℕ): ℕ := num_pens * cost_per_pen
def cost_of_notebooks (num_notebooks : ℕ) (cost_per_notebook : ℕ): ℕ := num_notebooks * cost_per_notebook
def cost_of_folders (num_folders : ℕ) (cost_per_folder : ℕ): ℕ := num_folders * cost_per_folder

def total_cost : ℕ :=
  cost_of_pens 3 1 + cost_of_notebooks 4 3 + cost_of_folders 2 5

def paid_amount : ℕ := 50

theorem jimmy_change : paid_amount - total_cost = 25 := by
  sorry

end jimmy_change_l8_8616


namespace temperature_below_zero_l8_8613

-- Assume the basic definitions and context needed
def above_zero (temp : Int) := temp > 0
def below_zero (temp : Int) := temp < 0

theorem temperature_below_zero (t1 t2 : Int) (h1 : above_zero t1) (h2 : t2 = -7) :
  below_zero t2 := by 
  -- This is where the proof would go
  sorry

end temperature_below_zero_l8_8613


namespace min_bottles_needed_l8_8138

theorem min_bottles_needed (bottle_size : ℕ) (min_ounces : ℕ) (n : ℕ) 
  (h1 : bottle_size = 15) 
  (h2 : min_ounces = 195) 
  (h3 : 15 * n >= 195) : n = 13 :=
sorry

end min_bottles_needed_l8_8138


namespace divides_p_minus_one_l8_8203

theorem divides_p_minus_one {p a b : ℕ} {n : ℕ} 
  (hp : p ≥ 3) 
  (prime_p : Nat.Prime p )
  (gcd_ab : Nat.gcd a b = 1)
  (hdiv : p ∣ (a ^ (2 ^ n) + b ^ (2 ^ n))) : 
  2 ^ (n + 1) ∣ p - 1 := 
sorry

end divides_p_minus_one_l8_8203


namespace arithmetic_sequence_b1_l8_8944

theorem arithmetic_sequence_b1 
  (b : ℕ → ℝ) 
  (U : ℕ → ℝ)
  (U2023 : ℝ) 
  (b2023 : ℝ)
  (hb2023 : b 2023 = b 1 + 2022 * (b 2 - b 1))
  (hU2023 : U 2023 = 2023 * (b 1 + 1011 * (b 2 - b 1))) 
  (hUn : ∀ n, U n = (n * (2 * b 1 + (n - 1) * (b 2 - b 1)) / 2)) :
  b 1 = (U 2023 - 2023 * b 2023) / 2023 :=
by
  sorry

end arithmetic_sequence_b1_l8_8944


namespace car_speed_kmph_l8_8271

noncomputable def speed_of_car (d : ℝ) (t : ℝ) : ℝ :=
  (d / t) * 3.6

theorem car_speed_kmph : speed_of_car 10 0.9999200063994881 = 36000.29 := by
  sorry

end car_speed_kmph_l8_8271


namespace exists_monomials_l8_8448

theorem exists_monomials (a b : ℕ) :
  ∃ x y : ℕ → ℕ → ℤ,
  (x 2 1 * y 2 1 = -12) ∧
  (∀ m n : ℕ, m ≠ 2 ∨ n ≠ 1 → x m n = 0 ∧ y m n = 0) ∧
  (∃ k l : ℤ, x 2 1 = k * (a ^ 2 * b ^ 1) ∧ y 2 1 = l * (a ^ 2 * b ^ 1) ∧ k + l = 1) :=
by
  sorry

end exists_monomials_l8_8448


namespace delegates_with_at_least_one_female_l8_8739

open Finset

def chooseDelegates (m f : ℕ) : ℕ :=
  (choose f 1 * choose m 2) + (choose f 2 * choose m 1) + (choose f 3)

theorem delegates_with_at_least_one_female :
  ∀ (m f : ℕ), m = 4 → f = 3 → chooseDelegates m f = 31 :=
by
  intros m f hm hf
  rw [hm, hf]
  simp [chooseDelegates, choose]
  sorry

end delegates_with_at_least_one_female_l8_8739


namespace quadratic_prime_roots_l8_8705

theorem quadratic_prime_roots (k : ℕ) (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p + q = 101 → p * q = k → False :=
by
  sorry

end quadratic_prime_roots_l8_8705


namespace find_k_l8_8036

def S (n : ℕ) : ℤ := n^2 - 9 * n

def a (n : ℕ) : ℤ := 
  if n = 1 then S 1
  else S n - S (n - 1)

theorem find_k (k : ℕ) (h1 : 5 < a k) (h2 : a k < 8) : k = 8 := by
  sorry

end find_k_l8_8036


namespace harmonic_mean_ordered_pairs_l8_8737

theorem harmonic_mean_ordered_pairs :
  ∃ n : ℕ, n = 23 ∧ ∀ (a b : ℕ), 
    0 < a ∧ 0 < b ∧ a < b ∧ (2 * a * b = 2 ^ 24 * (a + b)) → n = 23 :=
by sorry

end harmonic_mean_ordered_pairs_l8_8737


namespace probability_at_most_one_incorrect_l8_8363

variable (p : ℝ)

theorem probability_at_most_one_incorrect (h : 0 ≤ p ∧ p ≤ 1) :
  p^9 * (10 - 9*p) = p^10 + 10 * (1 - p) * p^9 := by
  sorry

end probability_at_most_one_incorrect_l8_8363


namespace find_A_d_minus_B_d_l8_8816

-- Definitions of the conditions
def is_digit_in_base (x : ℕ) (d : ℕ) : Prop := x < d

def ab_aa_sum_to_172 (A B d : ℕ) : Prop :=
  is_digit_in_base A d ∧ is_digit_in_base B d ∧ d > 7 ∧ (d * A + B) + (d * A + A) = d^2 + 7 * d + 2

-- The final theorem statement
theorem find_A_d_minus_B_d (A B d : ℕ) (h : ab_aa_sum_to_172 A B d) : A - B = 5 :=
by sorry

end find_A_d_minus_B_d_l8_8816


namespace find_n_l8_8196

theorem find_n (x n : ℝ) (h : x > 0) 
  (h_eq : x / 10 + x / n = 0.14000000000000002 * x) : 
  n = 25 :=
by
  sorry

end find_n_l8_8196


namespace find_A_d_minus_B_d_l8_8817

-- Definitions of the conditions
def is_digit_in_base (x : ℕ) (d : ℕ) : Prop := x < d

def ab_aa_sum_to_172 (A B d : ℕ) : Prop :=
  is_digit_in_base A d ∧ is_digit_in_base B d ∧ d > 7 ∧ (d * A + B) + (d * A + A) = d^2 + 7 * d + 2

-- The final theorem statement
theorem find_A_d_minus_B_d (A B d : ℕ) (h : ab_aa_sum_to_172 A B d) : A - B = 5 :=
by sorry

end find_A_d_minus_B_d_l8_8817


namespace find_original_price_l8_8237

-- Definitions based on Conditions
def original_price (P : ℝ) : Prop :=
  let increased_price := 1.25 * P
  let final_price := increased_price * 0.75
  final_price = 187.5

theorem find_original_price (P : ℝ) (h : original_price P) : P = 200 :=
  by sorry

end find_original_price_l8_8237


namespace fraction_reducible_l8_8585

theorem fraction_reducible (l : ℤ) : ∃ d : ℤ, d ≠ 1 ∧ d > 0 ∧ d = gcd (5 * l + 6) (8 * l + 7) := by 
  use 13
  sorry

end fraction_reducible_l8_8585


namespace main_line_train_probability_l8_8130

noncomputable def probability_catching_main_line (start_main_line start_harbor_line : Nat) (frequency : Nat) : ℝ :=
  if start_main_line % frequency = 0 ∧ start_harbor_line % frequency = 2 then 1 / 2 else 0

theorem main_line_train_probability :
  probability_catching_main_line 0 2 10 = 1 / 2 :=
by
  sorry

end main_line_train_probability_l8_8130


namespace proportion_solution_l8_8263

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 4.5 / (7 / 3)) : x = 0.3888888889 :=
by
  sorry

end proportion_solution_l8_8263


namespace argument_friends_count_l8_8976

-- Define the conditions
def original_friends: ℕ := 20
def current_friends: ℕ := 19
def new_friend: ℕ := 1

-- Define the statement that needs to be proved
theorem argument_friends_count : 
  (original_friends + new_friend - current_friends = 1) :=
by
  -- Placeholder for the proof
  sorry

end argument_friends_count_l8_8976


namespace Billy_current_age_l8_8604

variable (B : ℕ)

theorem Billy_current_age 
  (h1 : ∃ B, 4 * B - B = 12) : B = 4 := by
  sorry

end Billy_current_age_l8_8604


namespace value_of_a_plus_b_minus_c_l8_8032

theorem value_of_a_plus_b_minus_c (a b c : ℝ) 
  (h1 : abs a = 1) 
  (h2 : abs b = 2) 
  (h3 : abs c = 3) 
  (h4 : a > b) 
  (h5 : b > c) : 
  a + b - c = 2 := 
sorry

end value_of_a_plus_b_minus_c_l8_8032


namespace equation1_sol_equation2_sol_equation3_sol_l8_8996

theorem equation1_sol (x : ℝ) : 9 * x^2 - (x - 1)^2 = 0 ↔ (x = -0.5 ∨ x = 0.25) :=
sorry

theorem equation2_sol (x : ℝ) : (x * (x - 3) = 10) ↔ (x = 5 ∨ x = -2) :=
sorry

theorem equation3_sol (x : ℝ) : (x + 3)^2 = 2 * x + 5 ↔ (x = -2) :=
sorry

end equation1_sol_equation2_sol_equation3_sol_l8_8996


namespace sufficient_condition_frac_ineq_inequality_transformation_problem_equivalence_l8_8539

theorem sufficient_condition_frac_ineq (x : ℝ) : (1 < x ∧ x < 2) → ( (x + 1) / (x - 1) > 2) :=
by
  -- Given that 1 < x and x < 2, we need to show (x + 1) / (x - 1) > 2
  sorry

theorem inequality_transformation (x : ℝ) : ( (x + 1) / (x - 1) > 2) ↔ ( (x - 1) * (x - 3) < 0 ) :=
by
  -- Prove that (x + 1) / (x - 1) > 2 is equivalent to (x - 1)(x - 3) < 0
  sorry

theorem problem_equivalence (x : ℝ) : ( (x + 1) / (x - 1) > 2) → (1 < x ∧ x < 3) :=
by
  -- Prove that (x + 1) / (x - 1) > 2 implies 1 < x < 3
  sorry

end sufficient_condition_frac_ineq_inequality_transformation_problem_equivalence_l8_8539


namespace triangle_is_right_triangle_l8_8194

theorem triangle_is_right_triangle
  (a b c : ℝ)
  (h : a^2 + b^2 + c^2 - 10 * a - 6 * b - 8 * c + 50 = 0) :
  a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2 ∨ c^2 = a^2 + b^2 :=
sorry

end triangle_is_right_triangle_l8_8194


namespace find_ordered_triple_l8_8505

theorem find_ordered_triple (a b c : ℝ) (h1 : a > 2) (h2 : b > 2) (h3 : c > 2)
  (h4 : (a + 1)^2 / (b + c - 1) + (b + 3)^2 / (c + a - 3) + (c + 5)^2 / (a + b - 5) = 27) :
  (a, b, c) = (9, 7, 2) :=
by sorry

end find_ordered_triple_l8_8505


namespace production_average_l8_8325

-- Define the conditions and question
theorem production_average (n : ℕ) (P : ℕ) (P_new : ℕ) (h1 : P = n * 70) (h2 : P_new = P + 90) (h3 : P_new = (n + 1) * 75) : n = 3 := 
by sorry

end production_average_l8_8325


namespace max_papers_l8_8294

theorem max_papers (p c r : ℕ) (h1 : p ≥ 2) (h2 : c ≥ 1) (h3 : 3 * p + 5 * c + 9 * r = 72) : r ≤ 6 :=
sorry

end max_papers_l8_8294


namespace greater_number_is_25_l8_8104

theorem greater_number_is_25 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 :=
sorry

end greater_number_is_25_l8_8104


namespace price_of_thermometer_l8_8549

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

end price_of_thermometer_l8_8549


namespace monotonic_increasing_interval_l8_8233

def f (x : ℝ) : ℝ := x^2 - 2

theorem monotonic_increasing_interval :
  ∀ x y: ℝ, 0 <= x -> x <= y -> f x <= f y := 
by
  -- proof would be here
  sorry

end monotonic_increasing_interval_l8_8233


namespace height_at_10inches_l8_8694

theorem height_at_10inches 
  (a : ℚ)
  (h : 20 = (- (4 / 125) * 25 ^ 2 + 20))
  (span_eq : 50 = 50)
  (height_eq : 20 = 20)
  (y_eq : ∀ x : ℚ, - (4 / 125) * x ^ 2 + 20 = 16.8) :
  (- (4 / 125) * 10 ^ 2 + 20) = 16.8 :=
by
  sorry

end height_at_10inches_l8_8694


namespace tree_height_by_time_boy_is_36_inches_l8_8283

noncomputable def final_tree_height : ℕ :=
  let T₀ := 16
  let B₀ := 24
  let Bₓ := 36
  let boy_growth := Bₓ - B₀
  let tree_growth := 2 * boy_growth
  T₀ + tree_growth

theorem tree_height_by_time_boy_is_36_inches :
  final_tree_height = 40 :=
by
  sorry

end tree_height_by_time_boy_is_36_inches_l8_8283


namespace coffee_break_participants_l8_8870

theorem coffee_break_participants (n : ℕ) (h1 : n = 14) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 6 ∧ n - 2 * k > 0 ∧ n - 2 * k < n)
    (h3 : ∀ m, m < n → (∃ k, m = n - 2 * k → m ∈ {6, 8, 10, 12})): 
    ∃ m, n - 2 * m ∈ {6, 8, 10, 12} := 
by
  use 6
  use 8
  use 10
  use 12
  sorry

end coffee_break_participants_l8_8870


namespace min_value_a_b_inv_a_inv_b_l8_8589

theorem min_value_a_b_inv_a_inv_b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b + 1/a + 1/b ≥ 4 :=
sorry

end min_value_a_b_inv_a_inv_b_l8_8589


namespace find_n_l8_8509

variable (a b c n : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hn : n > 0)

theorem find_n (h1 : (a + b) / a = 3)
  (h2 : (b + c) / b = 4)
  (h3 : (c + a) / c = n) :
  n = 7 / 6 := 
sorry

end find_n_l8_8509


namespace valentines_count_l8_8081

theorem valentines_count (x y : ℕ) (h : x * y = x + y + 42) : x * y = 88 := by
  sorry

end valentines_count_l8_8081


namespace circle_chord_length_on_hyperbola_asymptotes_l8_8536

theorem circle_chord_length_on_hyperbola_asymptotes :
  let circle_eq := λ (x y : ℝ), x^2 + y^2 - 6*x - 2*y + 1 = 0 in
  let hyperbola_eq := λ (x y : ℝ), x^2 - y^2 = 1 in
  let asymptote1 := λ (x y : ℝ), y = x in
  let asymptote2 := λ (x y : ℝ), y = -x in
  ∀ p1 p2 p3 p4 : ℝ × ℝ, 
    (circle_eq p1.1 p1.2) ∧ (asymptote1 p1.1 p1.2) ∧ 
    (circle_eq p2.1 p2.2) ∧ (asymptote1 p2.1 p2.2) ∧
    (circle_eq p3.1 p3.2) ∧ (asymptote2 p3.1 p3.2) ∧
    (circle_eq p4.1 p4.2) ∧ (asymptote2 p4.1 p4.2) →
    (dist p1 p2 = sqrt 7) ∧ (dist p3 p4 = sqrt 7) :=
sorry

end circle_chord_length_on_hyperbola_asymptotes_l8_8536


namespace factor_polynomial_l8_8167

theorem factor_polynomial (x y : ℝ) : 
  x^4 + 4 * y^4 = (x^2 - 2 * x * y + 2 * y^2) * (x^2 + 2 * x * y + 2 * y^2) :=
by
  sorry

end factor_polynomial_l8_8167


namespace domain_width_of_g_l8_8187

theorem domain_width_of_g (h : ℝ → ℝ) (domain_h : ∀ x, -8 ≤ x ∧ x ≤ 8 → h x = h x) :
  let g (x : ℝ) := h (x / 2)
  ∃ a b, (∀ x, a ≤ x ∧ x ≤ b → ∃ y, g x = y) ∧ (b - a = 32) := 
sorry

end domain_width_of_g_l8_8187


namespace seeder_path_length_l8_8447

theorem seeder_path_length (initial_grain : ℤ) (decrease_percent : ℝ) (seeding_rate : ℝ) (width : ℝ) 
  (H_initial_grain : initial_grain = 250) 
  (H_decrease_percent : decrease_percent = 14 / 100) 
  (H_seeding_rate : seeding_rate = 175) 
  (H_width : width = 4) :
  (initial_grain * decrease_percent / seeding_rate) * 10000 / width = 500 := 
by 
  sorry

end seeder_path_length_l8_8447


namespace cost_of_fencing_is_289_l8_8941

def side_lengths : List ℕ := [10, 20, 15, 18, 12, 22]

def cost_per_meter : List ℚ := [3, 2, 4, 3.5, 2.5, 3]

def cost_of_side (length : ℕ) (rate : ℚ) : ℚ :=
  (length : ℚ) * rate

def total_cost : ℚ :=
  List.zipWith cost_of_side side_lengths cost_per_meter |>.sum

theorem cost_of_fencing_is_289 : total_cost = 289 := by
  sorry

end cost_of_fencing_is_289_l8_8941


namespace imaginary_part_of_z_l8_8607

-- Define complex numbers and necessary conditions
variable (z : ℂ)

-- The main statement
theorem imaginary_part_of_z (h : z * (1 + 2 * I) = 3 - 4 * I) : 
  (z.im = -2) :=
sorry

end imaginary_part_of_z_l8_8607


namespace shaded_area_l8_8769

theorem shaded_area (PR PV PQ QR : ℝ) (hPR : PR = 20) (hPV : PV = 12) (hPQ_QR : PQ + QR = PR) :
  PR * PV - 1 / 2 * 12 * PR = 120 :=
by
  -- Definitions used earlier
  have h_area_rectangle : PR * PV = 240 := by
    rw [hPR, hPV]
    norm_num
  have h_half_total_unshaded : (1 / 2) * 12 * PR = 120 := by
    rw [hPR]
    norm_num
  rw [h_area_rectangle, h_half_total_unshaded]
  norm_num

end shaded_area_l8_8769


namespace rental_property_key_count_l8_8545

def number_of_keys (complexes apartments_per_complex keys_per_lock locks_per_apartment : ℕ) : ℕ :=
  complexes * apartments_per_complex * keys_per_lock * locks_per_apartment

theorem rental_property_key_count : 
  number_of_keys 2 12 3 1 = 72 := by
  sorry

end rental_property_key_count_l8_8545


namespace quadrilateral_EFGH_inscribed_in_circle_l8_8527

theorem quadrilateral_EFGH_inscribed_in_circle 
  (a b c : ℝ)
  (angle_EFG : ℝ := 60)
  (angle_EHG : ℝ := 50)
  (EH : ℝ := 5)
  (FG : ℝ := 7)
  (EG : ℝ := a)
  (EF : ℝ := b)
  (GH : ℝ := c)
  : EG = 7 * (Real.sin (70 * Real.pi / 180)) / (Real.sin (50 * Real.pi / 180)) :=
by
  sorry

end quadrilateral_EFGH_inscribed_in_circle_l8_8527


namespace complement_union_l8_8752

open Set

variable (U : Set ℝ)
variable (A : Set ℝ)
variable (B : Set ℝ)

theorem complement_union (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) (hU : U = univ) 
(hA : A = {x : ℝ | 0 < x}) 
(hB : B = {x : ℝ | -3 < x ∧ x < 1}) : 
compl (A ∪ B) = {x : ℝ | x ≤ -3} :=
by
  sorry

end complement_union_l8_8752


namespace min_sum_of_dimensions_l8_8997

theorem min_sum_of_dimensions (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 3003) :
  a + b + c = 45 := sorry

end min_sum_of_dimensions_l8_8997


namespace number_of_sides_l8_8148

theorem number_of_sides (n : ℕ) : 
  let a_1 := 6 
  let d := 5
  let a_n := a_1 + (n - 1) * d
  a_n = 5 * n + 1 := 
by
  sorry

end number_of_sides_l8_8148


namespace scrabble_champions_l8_8771

noncomputable def num_champions : Nat := 25
noncomputable def male_percentage : Nat := 40
noncomputable def bearded_percentage : Nat := 40
noncomputable def bearded_bald_percentage : Nat := 60
noncomputable def non_bearded_bald_percentage : Nat := 30

theorem scrabble_champions :
  let male_champions := (male_percentage * num_champions) / 100
  let bearded_champions := (bearded_percentage * male_champions) / 100
  let bearded_bald_champions := (bearded_bald_percentage * bearded_champions) / 100
  let bearded_hair_champions := bearded_champions - bearded_bald_champions
  let non_bearded_champions := male_champions - bearded_champions
  let non_bearded_bald_champions := (non_bearded_bald_percentage * non_bearded_champions) / 100
  let non_bearded_hair_champions := non_bearded_champions - non_bearded_bald_champions
  bearded_bald_champions = 2 ∧ 
  bearded_hair_champions = 2 ∧ 
  non_bearded_bald_champions = 1 ∧ 
  non_bearded_hair_champions = 5 :=
by
  sorry

end scrabble_champions_l8_8771


namespace even_function_a_zero_l8_8351

section

variable (a : ℝ)

def f (x : ℝ) := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero : ∀ x : ℝ, f a x = f a (-x) → a = 0 := by
  sorry

end

end even_function_a_zero_l8_8351


namespace prob_six_largest_is_correct_l8_8906

noncomputable def probability_six_is_largest : ℚ :=
  let total_ways := (Finset.card (Finset.powersetLen 3 (Finset.range 7))) in
  let favorable_ways := (Finset.card (Finset.powersetLen 3 (Finset.range 6))) in
  (favorable_ways : ℚ) / total_ways

theorem prob_six_largest_is_correct : probability_six_is_largest = 4 / 7 := by
  sorry

end prob_six_largest_is_correct_l8_8906


namespace recurring_decimal_to_fraction_l8_8935

theorem recurring_decimal_to_fraction : ∀ x : ℝ, (x = 7 + (1/3 : ℝ)) → x = (22/3 : ℝ) :=
by
  sorry

end recurring_decimal_to_fraction_l8_8935


namespace original_faculty_size_l8_8799

theorem original_faculty_size (F : ℝ) (h1 : F * 0.85 * 0.80 = 195) : F = 287 :=
by
  sorry

end original_faculty_size_l8_8799


namespace cos_plus_2sin_eq_one_l8_8602

theorem cos_plus_2sin_eq_one (α : ℝ) (h : (1 + Real.cos α) / Real.sin α = 1 / 2) : 
  Real.cos α + 2 * Real.sin α = 1 := 
by
  sorry

end cos_plus_2sin_eq_one_l8_8602


namespace sum_le_xyz_plus_two_l8_8466

theorem sum_le_xyz_plus_two (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : 
  x + y + z ≤ xyz + 2 := 
sorry

end sum_le_xyz_plus_two_l8_8466


namespace cars_gain_one_passenger_each_l8_8540

-- Conditions
def initial_people_per_car : ℕ := 3 -- 2 passengers + 1 driver
def total_cars : ℕ := 20
def total_people_at_end : ℕ := 80

-- Question (equivalent to "answer")
theorem cars_gain_one_passenger_each :
  (total_people_at_end = total_cars * initial_people_per_car + total_cars) →
  total_people_at_end - total_cars * initial_people_per_car = total_cars :=
by sorry

end cars_gain_one_passenger_each_l8_8540


namespace ice_palace_steps_l8_8796

theorem ice_palace_steps (time_for_20_steps total_time : ℕ) (h1 : time_for_20_steps = 120) (h2 : total_time = 180) : 
  total_time * 20 / time_for_20_steps = 30 := by
  have time_per_step : ℕ := time_for_20_steps / 20
  have total_steps : ℕ := total_time / time_per_step
  sorry

end ice_palace_steps_l8_8796


namespace find_real_m_of_purely_imaginary_z_l8_8467

theorem find_real_m_of_purely_imaginary_z (m : ℝ) 
  (h1 : m^2 - 8 * m + 15 = 0) 
  (h2 : m^2 - 9 * m + 18 ≠ 0) : 
  m = 5 := 
by 
  sorry

end find_real_m_of_purely_imaginary_z_l8_8467


namespace original_faculty_is_287_l8_8798

noncomputable def original_faculty (F : ℝ) : Prop :=
  (F * 0.85 * 0.80 = 195)

theorem original_faculty_is_287 : ∃ F : ℝ, original_faculty F ∧ F = 287 := 
by 
  use 287
  sorry

end original_faculty_is_287_l8_8798


namespace correct_calculation_l8_8892

-- Define the statements for each option
def option_A (a : ℕ) : Prop := (a^2)^3 = a^5
def option_B (a : ℕ) : Prop := a^3 + a^2 = a^6
def option_C (a : ℕ) : Prop := a^6 / a^3 = a^3
def option_D (a : ℕ) : Prop := a^3 * a^2 = a^6

-- Define the theorem stating that option C is the only correct one
theorem correct_calculation (a : ℕ) : ¬option_A a ∧ ¬option_B a ∧ option_C a ∧ ¬option_D a := by
  sorry

end correct_calculation_l8_8892


namespace trapezium_area_l8_8898

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 17) : 
  (1 / 2 * (a + b) * h) = 323 :=
by
  have ha' : a = 20 := ha
  have hb' : b = 18 := hb
  have hh' : h = 17 := hh
  rw [ha', hb', hh']
  sorry

end trapezium_area_l8_8898


namespace solve_for_x_l8_8740

def δ (x : ℝ) : ℝ := 4 * x + 5
def φ (x : ℝ) : ℝ := 5 * x + 4

theorem solve_for_x (x : ℝ) (h : δ (φ x) = 4) : x = -17 / 20 := by
  sorry

end solve_for_x_l8_8740


namespace problem_l8_8815

/-- Given a number d > 7, 
    digits A and B in base d such that the equation AB_d + AA_d = 172_d holds, 
    we want to prove that A_d - B_d = 5. --/

theorem problem (d A B : ℕ) (hd : 7 < d)
  (hAB : d * A + B + d * A + A = 1 * d^2 + 7 * d + 2) : A - B = 5 :=
sorry

end problem_l8_8815


namespace cube_properties_l8_8670

theorem cube_properties (x : ℝ) (h1 : 6 * (2 * (8 * x)^(1/3))^2 = x) : x = 13824 :=
sorry

end cube_properties_l8_8670


namespace price_increase_percentage_l8_8856

theorem price_increase_percentage (original_price new_price : ℝ) (h₁ : original_price = 300) (h₂ : new_price = 360) : 
  (new_price - original_price) / original_price * 100 = 20 := 
by
  sorry

end price_increase_percentage_l8_8856


namespace digits_difference_l8_8821

theorem digits_difference (d A B : ℕ) (hd : d > 7) (h : d^2 * A + d * B + d^2 * A + d * A = 1 * d^2 + 7 * d + 2) : 
  A - B = 4 := 
sorry

end digits_difference_l8_8821


namespace sqrt_inequality_l8_8332

theorem sqrt_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) (habc : a + b + c = 9) : 
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := 
sorry

end sqrt_inequality_l8_8332


namespace percentage_is_50_l8_8140

theorem percentage_is_50 (P : ℝ) (h1 : P = 0.20 * 15 + 47) : P = 50 := 
by
  -- skip the proof
  sorry

end percentage_is_50_l8_8140


namespace prob_none_given_not_A_l8_8216

-- Definitions based on the conditions
def prob_single (h : ℕ → Prop) : ℝ := 0.2
def prob_double (h1 h2 : ℕ → Prop) : ℝ := 0.1
def prob_triple_given_AB : ℝ := 0.5

-- Assume that h1, h2, and h3 represent the hazards A, B, and C respectively.
variables (A B C : ℕ → Prop)

-- The ultimate theorem we want to prove
theorem prob_none_given_not_A (P : ℕ → Prop) :
  ((1 - (0.2 * 3 + 0.1 * 3) + (prob_triple_given_AB * (prob_single A + prob_double A B))) / (1 - 0.2) = 11 / 9) :=
by
  sorry

end prob_none_given_not_A_l8_8216


namespace knitting_time_total_l8_8782

-- Define knitting times for each item
def hat_knitting_time : ℕ := 2
def scarf_knitting_time : ℕ := 3
def mitten_knitting_time : ℕ := 1
def sock_knitting_time : ℕ := 3 / 2
def sweater_knitting_time : ℕ := 6

-- Define the number of grandchildren
def grandchildren_count : ℕ := 3

-- Total knitting time calculation
theorem knitting_time_total : 
  hat_knitting_time + scarf_knitting_time + 2 * mitten_knitting_time + 2 * sock_knitting_time + sweater_knitting_time = 16 ∧ 
  (hat_knitting_time + scarf_knitting_time + 2 * mitten_knitting_time + 2 * sock_knitting_time + sweater_knitting_time) * grandchildren_count = 48 :=
by 
  sorry

end knitting_time_total_l8_8782


namespace permutation_count_l8_8620

theorem permutation_count :
  let a : Fin 15 → Fin 15 :=
    λ i, if i = 5 then 0 else
         if i < 5 then 14 - i else
         i - 5 in
  (permute_count (a '' univ)) = (choose 14 4) := 
by
  sorry

end permutation_count_l8_8620


namespace cubes_with_odd_red_faces_l8_8437

-- Define the dimensions and conditions of the block
def block_length : ℕ := 6
def block_width: ℕ := 6
def block_height : ℕ := 2

-- The block is painted initially red on all sides
-- Then the bottom face is painted blue
-- The block is cut into 1-inch cubes
-- 

noncomputable def num_cubes_with_odd_red_faces (length width height : ℕ) : ℕ :=
  -- Only edge cubes have odd number of red faces in this configuration
  let corner_count := 8  -- 4 on top + 4 on bottom (each has 4 red faces)
  let edge_count := 40   -- 20 on top + 20 on bottom (each has 3 red faces)
  let face_only_count := 32 -- 16 on top + 16 on bottom (each has 2 red faces)
  -- The resulting total number of cubes with odd red faces
  edge_count

-- The theorem we need to prove
theorem cubes_with_odd_red_faces : num_cubes_with_odd_red_faces block_length block_width block_height = 40 :=
  by 
    -- Proof goes here
    sorry

end cubes_with_odd_red_faces_l8_8437


namespace find_larger_number_l8_8262

-- Definitions based on the conditions
def larger_number (L S : ℕ) : Prop :=
  L - S = 1365 ∧ L = 6 * S + 20

-- The theorem to prove
theorem find_larger_number (L S : ℕ) (h : larger_number L S) : L = 1634 :=
by
  sorry  -- Proof would go here

end find_larger_number_l8_8262


namespace horizontal_asymptote_degree_l8_8446

noncomputable def degree (p : Polynomial ℝ) : ℕ := Polynomial.natDegree p

theorem horizontal_asymptote_degree (p : Polynomial ℝ) :
  (∃ l : ℝ, ∀ ε > 0, ∃ N, ∀ x > N, |(p.eval x / (3 * x^7 - 2 * x^3 + x - 4)) - l| < ε) →
  degree p ≤ 7 :=
sorry

end horizontal_asymptote_degree_l8_8446


namespace expected_hit_targets_expected_hit_targets_not_less_than_half_l8_8721

-- Part (a): The expected number of hit targets
theorem expected_hit_targets (n : ℕ) (h : n ≠ 0) :
  E (number_of_hit_targets n) = n * (1 - (1 - (1 / n)) ^ n) :=
sorry

-- Part (b): The expected number of hit targets cannot be less than n / 2
theorem expected_hit_targets_not_less_than_half (n : ℕ) (h : n ≠ 0) :
  n * (1 - (1 - (1 / n)) ^ n) ≥ n / 2 :=
sorry

end expected_hit_targets_expected_hit_targets_not_less_than_half_l8_8721


namespace find_p5_l8_8777

-- Definitions based on conditions from the problem
def p (x : ℝ) : ℝ :=
  x^4 - 10 * x^3 + 35 * x^2 - 50 * x + 18  -- this construction ensures it's a quartic monic polynomial satisfying provided conditions

-- The main theorem we want to prove
theorem find_p5 :
  p 1 = 3 ∧ p 2 = 7 ∧ p 3 = 13 ∧ p 4 = 21 → p 5 = 51 :=
by
  -- The proof will be inserted here later
  sorry

end find_p5_l8_8777


namespace cookies_per_person_l8_8008

/-- Brenda's mother made cookies for 5 people. She prepared 35 cookies, 
    and each of them had the same number of cookies. 
    We aim to prove that each person had 7 cookies. --/
theorem cookies_per_person (total_cookies : ℕ) (number_of_people : ℕ) 
  (h1 : total_cookies = 35) (h2 : number_of_people = 5) : total_cookies / number_of_people = 7 := 
by
  sorry

end cookies_per_person_l8_8008


namespace avg_speed_last_40_min_is_70_l8_8979

noncomputable def avg_speed_last_interval
  (total_distance : ℝ) (total_time : ℝ)
  (speed_first_40_min : ℝ) (time_first_40_min : ℝ)
  (speed_second_40_min : ℝ) (time_second_40_min : ℝ) : ℝ :=
  let time_last_40_min := total_time - (time_first_40_min + time_second_40_min)
  let distance_first_40_min := speed_first_40_min * time_first_40_min
  let distance_second_40_min := speed_second_40_min * time_second_40_min
  let distance_last_40_min := total_distance - (distance_first_40_min + distance_second_40_min)
  distance_last_40_min / time_last_40_min

theorem avg_speed_last_40_min_is_70
  (h_total_distance : total_distance = 120)
  (h_total_time : total_time = 2)
  (h_speed_first_40_min : speed_first_40_min = 50)
  (h_time_first_40_min : time_first_40_min = 2 / 3)
  (h_speed_second_40_min : speed_second_40_min = 60)
  (h_time_second_40_min : time_second_40_min = 2 / 3) :
  avg_speed_last_interval 120 2 50 (2 / 3) 60 (2 / 3) = 70 :=
by
  sorry

end avg_speed_last_40_min_is_70_l8_8979


namespace sum_powers_of_i_l8_8012

-- Define the conditions
def i : ℂ := Complex.I -- Complex.I is the imaginary unit in ℂ (ℂ is the set of complex numbers)

-- The theorem statement
theorem sum_powers_of_i : (i + i^2 + i^3 + i^4) * 150 + 1 + i + i^2 + i^3 = 0 := by
  sorry

end sum_powers_of_i_l8_8012


namespace line_intersects_circle_l8_8102

/-- The positional relationship between the line y = ax + 1 and the circle x^2 + y^2 - 2x - 3 = 0
    is always intersecting for any real number a. -/
theorem line_intersects_circle (a : ℝ) : 
    ∀ a : ℝ, ∃ x y : ℝ, y = a * x + 1 ∧ x^2 + y^2 - 2 * x - 3 = 0 :=
by
    sorry

end line_intersects_circle_l8_8102


namespace avg_distance_is_600_l8_8785

-- Assuming Mickey runs half as many times as Johnny
def num_laps_johnny := 4
def num_laps_mickey := num_laps_johnny / 2
def lap_distance := 200

-- Calculating distances
def distance_johnny := num_laps_johnny * lap_distance
def distance_mickey := num_laps_mickey * lap_distance

-- Total distance run by both Mickey and Johnny
def total_distance := distance_johnny + distance_mickey

-- Average distance run by Johnny and Mickey
def avg_distance := total_distance / 2

-- Prove that the average distance run by Johnny and Mickey is 600 meters
theorem avg_distance_is_600 : avg_distance = 600 :=
by
  sorry

end avg_distance_is_600_l8_8785


namespace arithmetic_mean_of_fractions_l8_8254

theorem arithmetic_mean_of_fractions :
  (3 : ℚ) / 8 + (5 : ℚ) / 12 / 2 = 19 / 48 := by
  sorry

end arithmetic_mean_of_fractions_l8_8254


namespace range_of_a_l8_8750

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x + x^2 else x - x^2

theorem range_of_a (a : ℝ) : (∀ x, -1/2 ≤ x ∧ x ≤ 1/2 → f (x^2 + 1) > f (a * x)) ↔ -5/2 < a ∧ a < 5/2 := 
sorry

end range_of_a_l8_8750


namespace smallest_prime_after_six_nonprimes_l8_8553

-- Define the set of natural numbers and prime numbers
def is_natural (n : ℕ) : Prop := n ≥ 1
def is_prime (n : ℕ) : Prop := 1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_nonprime (n : ℕ) : Prop := ¬ is_prime n

-- The condition of six consecutive nonprime numbers
def six_consecutive_nonprime (n : ℕ) : Prop := 
  is_nonprime n ∧ 
  is_nonprime (n + 1) ∧ 
  is_nonprime (n + 2) ∧ 
  is_nonprime (n + 3) ∧ 
  is_nonprime (n + 4) ∧ 
  is_nonprime (n + 5)

-- The main theorem stating that 37 is the smallest prime following six consecutive nonprime numbers
theorem smallest_prime_after_six_nonprimes : 
  ∃ (n : ℕ), six_consecutive_nonprime n ∧ is_prime (n + 6) ∧ (∀ m, m < (n + 6) → ¬ is_prime m) :=
sorry

end smallest_prime_after_six_nonprimes_l8_8553


namespace quadratic_roots_sum_l8_8070

theorem quadratic_roots_sum (x₁ x₂ m : ℝ) 
  (eq1 : x₁^2 - (2 * m - 2) * x₁ + (m^2 - 2 * m) = 0) 
  (eq2 : x₂^2 - (2 * m - 2) * x₂ + (m^2 - 2 * m) = 0)
  (h : x₁ + x₂ = 10) : m = 6 :=
sorry

end quadratic_roots_sum_l8_8070


namespace geometric_series_smallest_b_l8_8206

theorem geometric_series_smallest_b (a b c : ℝ) (h_geometric : a * c = b^2) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_product : a * b * c = 216) : b = 6 :=
sorry

end geometric_series_smallest_b_l8_8206


namespace fraction_exponentiation_multiplication_l8_8305

theorem fraction_exponentiation_multiplication :
  (1 / 3) ^ 4 * (1 / 8) = 1 / 648 :=
by
  sorry

end fraction_exponentiation_multiplication_l8_8305


namespace map_distance_l8_8176

theorem map_distance (scale : ℝ) (d_actual_km : ℝ) (d_actual_m : ℝ) (d_actual_cm : ℝ) (d_map : ℝ) :
  scale = 1 / 250000 →
  d_actual_km = 5 →
  d_actual_m = d_actual_km * 1000 →
  d_actual_cm = d_actual_m * 100 →
  d_map = (1 * d_actual_cm) / (1 / scale) →
  d_map = 2 :=
by sorry

end map_distance_l8_8176


namespace shift_down_two_units_l8_8399

def original_function (x : ℝ) : ℝ := 2 * x + 1

def shifted_function (x : ℝ) : ℝ := original_function x - 2

theorem shift_down_two_units :
  ∀ x : ℝ, shifted_function x = 2 * x - 1 :=
by 
  intros x
  simp [shifted_function, original_function]
  sorry

end shift_down_two_units_l8_8399


namespace emilia_strCartons_l8_8582

theorem emilia_strCartons (total_cartons_needed cartons_bought cartons_blueberries : ℕ) (h1 : total_cartons_needed = 42) (h2 : cartons_blueberries = 7) (h3 : cartons_bought = 33) :
  (total_cartons_needed - (cartons_bought + cartons_blueberries)) = 2 :=
by
  sorry

end emilia_strCartons_l8_8582


namespace vacuum_upstairs_more_than_twice_downstairs_l8_8373

theorem vacuum_upstairs_more_than_twice_downstairs 
  (x y : ℕ) 
  (h1 : 27 = 2 * x + y) 
  (h2 : x + 27 = 38) : 
  y = 5 :=
by 
  sorry

end vacuum_upstairs_more_than_twice_downstairs_l8_8373


namespace min_value_of_quadratic_expression_l8_8550

theorem min_value_of_quadratic_expression : ∃ x : ℝ, ∀ y : ℝ, y = x^2 + 12*x + 9 → y ≥ -27 :=
sorry

end min_value_of_quadratic_expression_l8_8550


namespace find_f_at_4_l8_8483

def f (n : ℕ) : ℕ := sorry -- We define the function f.

theorem find_f_at_4 : (∀ x : ℕ, f (2 * x) = 3 * x^2 + 1) → f 4 = 13 :=
by
  sorry

end find_f_at_4_l8_8483


namespace solve_for_b_l8_8318

noncomputable def system_has_solution (b : ℝ) : Prop :=
  ∃ (a : ℝ) (x y : ℝ),
    y = -b - x^2 ∧
    x^2 + y^2 + 8 * a^2 = 4 + 4 * a * (x + y)

theorem solve_for_b (b : ℝ) : system_has_solution b ↔ b ≤ 2 * Real.sqrt 2 + 1 / 4 := 
by 
  sorry

end solve_for_b_l8_8318


namespace coffee_break_l8_8873

variable {n : ℕ}
variables {participants : Finset ℕ} (hparticipants : participants.card = 14)
variables (left : Finset ℕ)
variables (stayed : Finset ℕ) (hstayed : stayed.card = 14 - 2 * left.card)

-- The overall proof problem statement:
theorem coffee_break (h : ∀ x ∈ stayed, ∃! y, (y ∈ left ∧ adjacent x y participants)) :
  left.card = 6 ∨ left.card = 8 ∨ left.card = 10 ∨ left.card = 12 := 
sorry

end coffee_break_l8_8873


namespace range_of_f_l8_8239

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x - 1 else x^2 - 2 * x

theorem range_of_f : set.range f = set.Ioi (-1) := by
  sorry

end range_of_f_l8_8239


namespace area_ratio_triangle_MNO_XYZ_l8_8615

noncomputable def triangle_area_ratio (XY YZ XZ p q r : ℝ) : ℝ := sorry

theorem area_ratio_triangle_MNO_XYZ : 
  ∀ (p q r: ℝ),
  p > 0 → q > 0 → r > 0 →
  p + q + r = 3 / 4 →
  p ^ 2 + q ^ 2 + r ^ 2 = 1 / 2 →
  triangle_area_ratio 12 16 20 p q r = 9 / 32 :=
sorry

end area_ratio_triangle_MNO_XYZ_l8_8615


namespace smallest_positive_debt_pigs_goats_l8_8659

theorem smallest_positive_debt_pigs_goats :
  ∃ p g : ℤ, 350 * p + 240 * g = 10 :=
by
  sorry

end smallest_positive_debt_pigs_goats_l8_8659


namespace xyz_value_l8_8462

-- Define real numbers x, y, z
variables {x y z : ℝ}

-- Define the theorem with the given conditions and conclusion
theorem xyz_value 
  (h1 : (x + y + z) * (xy + xz + yz) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12)
  (h3 : (x + y + z)^2 = x^2 + y^2 + z^2 + 12) :
  x * y * z = 8 := 
sorry

end xyz_value_l8_8462


namespace magnitude_z1_condition_z2_range_condition_l8_8333

-- Define and set up the conditions and problem statements
open Complex

def complex_number_condition (z₁ : ℂ) (m : ℝ) : Prop :=
  z₁ = 1 + m * I ∧ ((z₁ * (1 - I)).re = 0)

def z₂_condition (z₂ z₁ : ℂ) (n : ℝ) : Prop :=
  z₂ = z₁ * (n - I) ∧ z₂.re < 0 ∧ z₂.im < 0

-- Prove that if z₁ = 1 + m * I and z₁ * (1 - I) is pure imaginary, then |z₁| = sqrt 2
theorem magnitude_z1_condition (m : ℝ) (z₁ : ℂ) 
  (h₁ : complex_number_condition z₁ m) : abs z₁ = Real.sqrt 2 :=
by sorry

-- Prove that if z₂ = z₁ * (n + i^3) is in the third quadrant, then n is in the range (-1, 1)
theorem z2_range_condition (n : ℝ) (m : ℝ) (z₁ z₂ : ℂ)
  (h₁ : complex_number_condition z₁ m)
  (h₂ : z₂_condition z₂ z₁ n) : -1 < n ∧ n < 1 :=
by sorry

end magnitude_z1_condition_z2_range_condition_l8_8333


namespace problem_l8_8766

variable (p q : Prop)

theorem problem (h₁ : ¬ p) (h₂ : ¬ (p ∧ q)) : ¬ (p ∨ q) := sorry

end problem_l8_8766


namespace volleyball_team_ways_l8_8629

def num_ways_choose_starers : ℕ :=
  3 * (Nat.choose 12 6 + Nat.choose 12 5)

theorem volleyball_team_ways :
  num_ways_choose_starers = 5148 := by
  sorry

end volleyball_team_ways_l8_8629


namespace average_is_six_l8_8468

-- Define the dataset
def dataset : List ℕ := [5, 9, 9, 3, 4]

-- Define the sum of the dataset values
def datasetSum : ℕ := 5 + 9 + 9 + 3 + 4

-- Define the number of items in the dataset
def datasetCount : ℕ := dataset.length

-- Define the average calculation
def average : ℚ := datasetSum / datasetCount

-- The theorem stating the average value of the given dataset is 6
theorem average_is_six : average = 6 := sorry

end average_is_six_l8_8468


namespace tens_digit_of_19_pow_2023_l8_8578

theorem tens_digit_of_19_pow_2023 :
  (19^2023 % 100) / 10 % 10 = 5 := 
sorry

end tens_digit_of_19_pow_2023_l8_8578


namespace total_amount_paid_l8_8656

def apples_kg := 8
def apples_rate := 70
def mangoes_kg := 9
def mangoes_rate := 65
def oranges_kg := 5
def oranges_rate := 50
def bananas_kg := 3
def bananas_rate := 30

def total_amount := (apples_kg * apples_rate) + (mangoes_kg * mangoes_rate) + (oranges_kg * oranges_rate) + (bananas_kg * bananas_rate)

theorem total_amount_paid : total_amount = 1485 := by
  sorry

end total_amount_paid_l8_8656


namespace max_a_such_that_f_geq_a_min_value_under_constraint_l8_8904

-- Problem (1)
theorem max_a_such_that_f_geq_a :
  ∃ (a : ℝ), (∀ (x : ℝ), |x - (5/2)| + |x - a| ≥ a) ∧ a = 5 / 4 := sorry

-- Problem (2)
theorem min_value_under_constraint :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + 2 * y + 3 * z = 1 ∧
  (3 / x + 2 / y + 1 / z) = 16 + 8 * Real.sqrt 3 := sorry

end max_a_such_that_f_geq_a_min_value_under_constraint_l8_8904


namespace coin_flip_probability_l8_8877

noncomputable def num_favorable_outcomes : ℕ := 
  Nat.choose 15 12 + Nat.choose 15 13 + Nat.choose 15 14 + Nat.choose 15 15

noncomputable def total_outcomes : ℕ := 2 ^ 15

noncomputable def at_least_12_heads_probability : ℚ := 
  num_favorable_outcomes / total_outcomes

theorem coin_flip_probability :
  at_least_12_heads_probability = 9 / 512 := 
by
  simp [at_least_12_heads_probability, num_favorable_outcomes, total_outcomes]
  sorry

end coin_flip_probability_l8_8877


namespace ratio_S15_S5_l8_8458

variable {a : ℕ → ℝ}  -- The geometric sequence
variable {S : ℕ → ℝ}  -- The sum of the first n terms of the geometric sequence

-- Define the conditions:
axiom sum_of_first_n_terms (n : ℕ) : S n = a 0 * (1 - (a 1)^n) / (1 - a 1)
axiom ratio_S10_S5 : S 10 / S 5 = 1 / 2

-- Define the math proof problem:
theorem ratio_S15_S5 : S 15 / S 5 = 3 / 4 :=
  sorry

end ratio_S15_S5_l8_8458


namespace bottles_remaining_after_2_days_l8_8690

theorem bottles_remaining_after_2_days :
  ∀ (initial_bottles : ℕ), initial_bottles = 24 →
  let bottles_first_day := initial_bottles - initial_bottles / 3 in
  let bottles_after_first_day := initial_bottles - bottles_first_day in
  let bottles_second_day := bottles_after_first_day / 2 in
  let bottles_remaining := bottles_after_first_day - bottles_second_day in
  bottles_remaining = 8 :=
by
  intros initial_bottles h_init
  let bottles_first_day := initial_bottles / 3
  let bottles_after_first_day := initial_bottles - bottles_first_day
  let bottles_second_day := bottles_after_first_day / 2
  let bottles_remaining := bottles_after_first_day - bottles_second_day
  have h_init_val : initial_bottles = 24 := h_init
  rw h_init_val at *
  calc
    bottles_first_day = 8 : by sorry
    bottles_after_first_day = 24 - 8 : by sorry
    _ = 16 : by sorry
    bottles_second_day = 16 / 2 : by sorry
    _ = 8 : by sorry
    bottles_remaining = 16 - 8 : by sorry
    _ = 8 : by sorry

end bottles_remaining_after_2_days_l8_8690


namespace montoya_budget_l8_8096

def percentage_food (groceries: ℝ) (eating_out: ℝ) : ℝ :=
  groceries + eating_out

def percentage_transportation_rent_utilities (transportation: ℝ) (rent: ℝ) (utilities: ℝ) : ℝ :=
  transportation + rent + utilities

def total_percentage (food: ℝ) (transportation_rent_utilities: ℝ) : ℝ :=
  food + transportation_rent_utilities

theorem montoya_budget :
  ∀ (groceries : ℝ) (eating_out : ℝ) (transportation : ℝ) (rent : ℝ) (utilities : ℝ),
    groceries = 0.6 → eating_out = 0.2 → transportation = 0.1 → rent = 0.05 → utilities = 0.05 →
    total_percentage (percentage_food groceries eating_out) (percentage_transportation_rent_utilities transportation rent utilities) = 1 :=
by
sorry

end montoya_budget_l8_8096


namespace anna_total_value_l8_8703

theorem anna_total_value (total_bills : ℕ) (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ)
  (h1 : total_bills = 12) (h2 : five_dollar_bills = 4) (h3 : ten_dollar_bills = total_bills - five_dollar_bills) :
  5 * five_dollar_bills + 10 * ten_dollar_bills = 100 := by
  sorry

end anna_total_value_l8_8703


namespace power_of_prime_calculate_729_to_two_thirds_l8_8922

theorem power_of_prime (a : ℝ) (b : ℝ) (c : ℝ) (h : a = b^c) (e : ℝ) :
  a ^ e = b ^ (c * e) := sorry

theorem calculate_729_to_two_thirds : (729 : ℝ) ^ (2 / 3) = 81 := by
  have h : 729 = (3 : ℝ) ^ 6 := by norm_num
  exact power_of_prime (729 : ℝ) (3 : ℝ) (6 : ℝ) h (2 / 3)

end power_of_prime_calculate_729_to_two_thirds_l8_8922


namespace lisa_total_cost_l8_8212

def c_phone := 1000
def c_contract_per_month := 200
def c_case := 0.20 * c_phone
def c_headphones := 0.5 * c_case
def t_year := 12

theorem lisa_total_cost :
  c_phone + (c_case) + (c_headphones) + (c_contract_per_month * t_year) = 3700 :=
by
  sorry

end lisa_total_cost_l8_8212


namespace matchstick_problem_l8_8123

theorem matchstick_problem (n : ℕ) (T : ℕ → ℕ) :
  (∀ n, T n = 4 + 9 * (n - 1)) ∧ n = 15 → T n = 151 :=
by
  sorry

end matchstick_problem_l8_8123


namespace area_circle_l8_8845

-- Define the given condition
def polar_eq (r θ : ℝ) : Prop :=
  r = 3 * Real.cos θ - 4 * Real.sin θ

-- The goal is to prove the area of the circle described by the polar equation
theorem area_circle {r θ : ℝ} (h : polar_eq r θ) :
  ∃ A, A = π * (5 / 2) ^ 2 :=
sorry

end area_circle_l8_8845


namespace coconut_grove_problem_l8_8495

theorem coconut_grove_problem
  (x : ℤ)
  (T40 : ℤ := x + 2)
  (T120 : ℤ := x)
  (T180 : ℤ := x - 2)
  (N_total : ℤ := 40 * (x + 2) + 120 * x + 180 * (x - 2))
  (T_total : ℤ := (x + 2) + x + (x - 2))
  (average_yield : ℤ := 100) :
  (N_total / T_total) = average_yield → x = 7 :=
by
  sorry

end coconut_grove_problem_l8_8495


namespace find_ab_sum_l8_8193

theorem find_ab_sum
  (a b : ℝ)
  (h₁ : a^3 - 3 * a^2 + 5 * a - 1 = 0)
  (h₂ : b^3 - 3 * b^2 + 5 * b - 5 = 0) :
  a + b = 2 := by
  sorry

end find_ab_sum_l8_8193


namespace value_range_f_l8_8244

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * real.exp x * (real.sin x + real.cos x)

theorem value_range_f :
  (set.range (λ (x : ℝ), f x) ∩ set.Icc 0 (real.pi / 2)) = 
  set.Icc (1 / 2) ((1 / 2) * real.exp (real.pi / 2)) :=
sorry

end value_range_f_l8_8244


namespace average_distance_is_600_l8_8793

-- Definitions based on the given conditions
def distance_around_block := 200
def johnny_rounds := 4
def mickey_rounds := johnny_rounds / 2

-- The calculated distances
def johnny_distance := johnny_rounds * distance_around_block
def mickey_distance := mickey_rounds * distance_around_block

-- The average distance computation
def average_distance := (johnny_distance + mickey_distance) / 2

-- The theorem to prove that the average distance is 600 meters
theorem average_distance_is_600 : average_distance = 600 := by sorry

end average_distance_is_600_l8_8793


namespace coffee_break_l8_8872

theorem coffee_break (n k : ℕ) (h1 : n = 14) (h2 : 0 < n - 2 * k) (h3 : n - 2 * k < n) :
  n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12 :=
by
  sorry

end coffee_break_l8_8872


namespace total_frogs_in_both_ponds_l8_8865

noncomputable def total_frogs_combined : Nat :=
let frogs_in_pond_a : Nat := 32
let frogs_in_pond_b : Nat := frogs_in_pond_a / 2
frogs_in_pond_a + frogs_in_pond_b

theorem total_frogs_in_both_ponds :
  total_frogs_combined = 48 := by
  sorry

end total_frogs_in_both_ponds_l8_8865


namespace math_problem_l8_8622

variable {x y z : ℝ}
variable (hx : x > 0) (hy : y > 0) (hz : z > 0)
variable (h : x^2 + y^2 + z^2 = 1)

theorem math_problem : 
  (x / (1 - x^2)) + (y / (1 - y^2)) + (z / (1 - z^2)) ≥ 3 * Real.sqrt 3 / 2 :=
sorry

end math_problem_l8_8622


namespace sampling_probabilities_equal_l8_8735

-- Definitions according to the problem conditions
def population_size := ℕ
def sample_size := ℕ
def simple_random_sampling (N n : ℕ) : Prop := sorry
def systematic_sampling (N n : ℕ) : Prop := sorry
def stratified_sampling (N n : ℕ) : Prop := sorry

-- Probabilities
def P1 : ℝ := sorry -- Probability for simple random sampling
def P2 : ℝ := sorry -- Probability for systematic sampling
def P3 : ℝ := sorry -- Probability for stratified sampling

-- Each definition directly corresponds to a condition in the problem statement.
-- Now, we summarize the equivalent proof problem in Lean.

theorem sampling_probabilities_equal (N n : ℕ) (h1 : simple_random_sampling N n) (h2 : systematic_sampling N n) (h3 : stratified_sampling N n) :
  P1 = P2 ∧ P2 = P3 :=
by sorry

end sampling_probabilities_equal_l8_8735


namespace electronics_weight_is_9_l8_8131

noncomputable def electronics_weight : ℕ :=
  let B : ℕ := sorry -- placeholder for the value of books weight.
  let C : ℕ := 12
  let E : ℕ := 9
  have h1 : (B : ℚ) / (C : ℚ) = 7 / 4 := sorry
  have h2 : (C : ℚ) / (E : ℚ) = 4 / 3 := sorry
  have h3 : (B : ℚ) / (C - 6 : ℚ) = 7 / 2 := sorry
  E

theorem electronics_weight_is_9 : electronics_weight = 9 :=
by
  dsimp [electronics_weight]
  repeat { sorry }

end electronics_weight_is_9_l8_8131


namespace find_X_l8_8972

variable (E X : ℕ)

-- Theorem statement
theorem find_X (hE : E = 9)
              (hSum : E * 100 + E * 10 + E + E * 100 + E * 10 + E = 1798) :
              X = 7 :=
sorry

end find_X_l8_8972


namespace infinite_power_tower_solution_l8_8928

theorem infinite_power_tower_solution (x : ℝ) (y : ℝ) (h1 : y = x ^ x ^ x ^ x ^ x ^ x ^ x ^ x ^ x ^ x) (h2 : y = 4) : x = Real.sqrt 2 :=
by
  sorry

end infinite_power_tower_solution_l8_8928


namespace three_digit_permutations_l8_8051

theorem three_digit_permutations : 
  ∃ n : ℕ, n = 6 ∧ 
    (∀ (p : ℕ), (1 ≤ p / 100 ∧ p / 100 ≤ 3) ∧ 
                   (1 ≤ (p % 100) / 10 ∧ (p % 100) / 10 ≤ 3) ∧ 
                   (1 ≤ p % 10 ∧ p % 10 ≤ 3) ∧ 
                   (digits p = [3-digit number if permutation of [1,2,3]∧ p each digit from {1,2,3} exactly once]) → 
                        n = list.permutations([1, 2, 3]).length) := sorry

end three_digit_permutations_l8_8051


namespace probability_fourth_quadrant_is_one_sixth_l8_8111

def in_fourth_quadrant (x y : ℤ) : Prop :=
  x > 0 ∧ y < 0

def possible_coordinates : List (ℤ × ℤ) :=
  [(0, -1), (0, 2), (0, -3), (-1, 0), (-1, 2), (-1, -3), (2, 0), (2, -1), (2, -3), (-3, 0), (-3, -1), (-3, 2)]

noncomputable def probability_fourth_quadrant : ℚ :=
  (possible_coordinates.count (λ p => in_fourth_quadrant p.fst p.snd)).toNat / (possible_coordinates.length : ℚ)

theorem probability_fourth_quadrant_is_one_sixth :
  probability_fourth_quadrant = 1/6 :=
by
  sorry

end probability_fourth_quadrant_is_one_sixth_l8_8111


namespace synodic_month_is_approx_29_5306_l8_8556

noncomputable def sidereal_month_moon : ℝ := 
27 + 7/24 + 43/1440  -- conversion of 7 hours and 43 minutes to days

noncomputable def sidereal_year_earth : ℝ := 
365 + 6/24 + 9/1440  -- conversion of 6 hours and 9 minutes to days

noncomputable def synodic_month (T_H T_F: ℝ) : ℝ := 
(T_H * T_F) / (T_F - T_H)

theorem synodic_month_is_approx_29_5306 : 
  abs (synodic_month sidereal_month_moon sidereal_year_earth - (29 + 12/24 + 44/1440)) < 0.0001 :=
by 
  sorry

end synodic_month_is_approx_29_5306_l8_8556


namespace power_of_two_l8_8120

theorem power_of_two (Number : ℕ) (h1 : Number = 128) (h2 : Number * (1/4 : ℝ) = 2^5) :
  ∃ power : ℕ, 2^power = 128 := 
by
  use 7
  sorry

end power_of_two_l8_8120


namespace election_result_l8_8654

theorem election_result:
  ∀ (Henry_votes India_votes Jenny_votes Ken_votes Lena_votes : ℕ)
    (counted_percentage : ℕ)
    (counted_votes : ℕ), 
    Henry_votes = 14 → 
    India_votes = 11 → 
    Jenny_votes = 10 → 
    Ken_votes = 8 → 
    Lena_votes = 2 → 
    counted_percentage = 90 → 
    counted_votes = 45 → 
    (counted_percentage * Total_votes / 100 = counted_votes) →
    (Total_votes = counted_votes * 100 / counted_percentage) →
    (Remaining_votes = Total_votes - counted_votes) →
    ((Henry_votes + Max_remaining_Votes >= Max_votes) ∨ 
    (India_votes + Max_remaining_Votes >= Max_votes) ∨ 
    (Jenny_votes + Max_remaining_Votes >= Max_votes)) →
    3 = 
    (if Henry_votes + Remaining_votes > Max_votes then 1 else 0) + 
    (if India_votes + Remaining_votes > Max_votes then 1 else 0) + 
    (if Jenny_votes + Remaining_votes > Max_votes then 1 else 0) := 
  sorry

end election_result_l8_8654


namespace conic_is_pair_of_lines_l8_8162

-- Define the specific conic section equation
def conic_eq (x y : ℝ) : Prop := 9 * x^2 - 36 * y^2 = 0

-- State the theorem
theorem conic_is_pair_of_lines : ∀ x y : ℝ, conic_eq x y ↔ (x = 2 * y ∨ x = -2 * y) :=
by
  -- Sorry is placed to denote that proof steps are omitted in this statement
  sorry

end conic_is_pair_of_lines_l8_8162


namespace negation_equiv_l8_8234

-- Define the proposition that the square of all real numbers is positive
def pos_of_all_squares : Prop := ∀ x : ℝ, x^2 > 0

-- Define the negation of the proposition
def neg_pos_of_all_squares : Prop := ∃ x : ℝ, x^2 ≤ 0

theorem negation_equiv (h : ¬ pos_of_all_squares) : neg_pos_of_all_squares :=
  sorry

end negation_equiv_l8_8234


namespace sprinted_further_than_jogged_l8_8160

def sprint_distance1 := 0.8932
def sprint_distance2 := 0.7773
def sprint_distance3 := 0.9539
def sprint_distance4 := 0.5417
def sprint_distance5 := 0.6843

def jog_distance1 := 0.7683
def jog_distance2 := 0.4231
def jog_distance3 := 0.5733
def jog_distance4 := 0.625
def jog_distance5 := 0.6549

def total_sprint_distance := sprint_distance1 + sprint_distance2 + sprint_distance3 + sprint_distance4 + sprint_distance5
def total_jog_distance := jog_distance1 + jog_distance2 + jog_distance3 + jog_distance4 + jog_distance5

theorem sprinted_further_than_jogged :
  total_sprint_distance - total_jog_distance = 0.8058 :=
by
  sorry

end sprinted_further_than_jogged_l8_8160


namespace abs_neg_two_is_two_l8_8824

def absolute_value (x : ℝ) : ℝ := if x < 0 then -x else x

theorem abs_neg_two_is_two : absolute_value (-2) = 2 :=
by
  sorry

end abs_neg_two_is_two_l8_8824


namespace find_n_l8_8265

noncomputable def f (n : ℝ) : ℝ :=
  n ^ (n / 2)

example : f 2 = 2 := sorry

theorem find_n : ∃ n : ℝ, f n = 12 ∧ abs (n - 3.4641) < 0.0001 := sorry

end find_n_l8_8265


namespace cost_equivalence_min_sets_of_A_l8_8685

noncomputable def cost_of_B := 120
noncomputable def cost_of_A := cost_of_B + 30

theorem cost_equivalence (x : ℕ) :
  (1200 / (x + 30) = 960 / x) → x = 120 :=
by
  sorry

theorem min_sets_of_A :
  ∀ m : ℕ, (150 * m + 120 * (20 - m) ≥ 2800) ↔ m ≥ 14 :=
by
  sorry

end cost_equivalence_min_sets_of_A_l8_8685


namespace circle_area_l8_8847

open Real

def given_equation (r θ : ℝ) : Prop := r = 3 * cos θ - 4 * sin θ

theorem circle_area (r θ : ℝ) (h : given_equation r θ) : 
  ∃ (c : ℝ × ℝ) (R : ℝ), c = (3 / 2, -2) ∧ R = 5 / 2 ∧ π * R^2 = 25 / 4 * π :=
sorry

end circle_area_l8_8847


namespace factorial_division_l8_8663

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division : (factorial 15) / ((factorial 6) * (factorial 9)) = 5005 :=
by
  sorry

end factorial_division_l8_8663


namespace sum_of_nonneg_reals_l8_8489

theorem sum_of_nonneg_reals (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x^2 + y^2 + z^2 = 52) (h5 : x * y + y * z + z * x = 24) :
  x + y + z = 10 :=
sorry

end sum_of_nonneg_reals_l8_8489


namespace product_even_probability_l8_8946

theorem product_even_probability :
  let total_outcomes := 216
  let probability_geoff_odd := (3 / 6) * (3 / 6)
  let probability_geoff_even := 1 - probability_geoff_odd
  probability_geoff_even = 3 / 4 :=
by
  sorry

end product_even_probability_l8_8946


namespace laundry_lcm_l8_8222

theorem laundry_lcm :
  Nat.lcm (Nat.lcm 6 9) (Nat.lcm 12 15) = 180 :=
by
  sorry

end laundry_lcm_l8_8222


namespace normal_level_short_of_capacity_l8_8151

noncomputable def total_capacity (water_amount : ℕ) (percentage : ℝ) : ℝ :=
  water_amount / percentage

noncomputable def normal_level (water_amount : ℕ) : ℕ :=
  water_amount / 2

theorem normal_level_short_of_capacity (water_amount : ℕ) (percentage : ℝ) (capacity : ℝ) (normal : ℕ) : 
  water_amount = 30 ∧ percentage = 0.75 ∧ capacity = total_capacity water_amount percentage ∧ normal = normal_level water_amount →
  (capacity - ↑normal) = 25 :=
by
  intros h
  sorry

end normal_level_short_of_capacity_l8_8151


namespace largest_fraction_l8_8671

def frac_A := (5 : ℚ) / 11
def frac_B := (6 : ℚ) / 13
def frac_C := (19 : ℚ) / 39
def frac_D := (101 : ℚ) / 203
def frac_E := (152 : ℚ) / 303
def frac_F := (80 : ℚ) / 159

theorem largest_fraction : 
  ∀ f ∈ {frac_A, frac_B, frac_C, frac_D, frac_E, frac_F}, f ≤ frac_F :=
by
  sorry

end largest_fraction_l8_8671


namespace repeating_decimal_to_fraction_l8_8938

theorem repeating_decimal_to_fraction : 
  (∃ (x : ℚ), x = 7 + 3 / 9) → 7 + 3 / 9 = 22 / 3 :=
by
  intros h
  sorry

end repeating_decimal_to_fraction_l8_8938


namespace cells_after_one_week_l8_8559

theorem cells_after_one_week : (3 ^ 7) = 2187 :=
by sorry

end cells_after_one_week_l8_8559


namespace incorrect_relation_when_agtb_l8_8054

theorem incorrect_relation_when_agtb (a b : ℝ) (c : ℝ) (h : a > b) : c = 0 → ¬ (a * c^2 > b * c^2) :=
by
  -- Not providing the proof here as specified in the instructions.
  sorry

end incorrect_relation_when_agtb_l8_8054


namespace tangent_line_at_origin_range_of_a_l8_8475

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := real.log (1 + x) + a * x * real.exp (-x)

theorem tangent_line_at_origin (a : ℝ) :
  a = 1 → (∀ x : ℝ, f 1 x = real.log (1 + x) + x * real.exp (-x)) → (0, f 1 0) → 
  ∃ m : ℝ, m = 2 ∧ (∀ x : ℝ, f 1 x = m * x) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x = real.log (1 + x) + a * x * real.exp (-x)) →
  (∃ c₁ ∈ Ioo (-1 : ℝ) 0, f a c₁ = 0) ∧ (∃ c₂ ∈ Ioo 0 (1:ℝ), f a c₂ = 0) → 
  a ∈ Iio (-1) :=
sorry

end tangent_line_at_origin_range_of_a_l8_8475


namespace functional_inequality_solution_l8_8450

theorem functional_inequality_solution {f : ℝ → ℝ} 
  (h : ∀ x y : ℝ, f (x * y) ≤ y * f (x) + f (y)) : 
  ∀ x : ℝ, f x = 0 :=
sorry

end functional_inequality_solution_l8_8450


namespace min_abs_sum_of_x1_x2_l8_8357

open Real

theorem min_abs_sum_of_x1_x2 (x1 x2 : ℝ) (h : 1 / ((2 + sin x1) * (2 + sin (2 * x2))) = 1) : 
  abs (x1 + x2) = π / 4 :=
sorry

end min_abs_sum_of_x1_x2_l8_8357


namespace expression_is_integer_expression_modulo_3_l8_8801

theorem expression_is_integer (n : ℕ) (hn : n > 0) : 
  ∃ (k : ℤ), (n^3 + (3/2) * n^2 + (1/2) * n - 1) = k := 
sorry

theorem expression_modulo_3 (n : ℕ) (hn : n > 0) : 
  (n^3 + (3/2) * n^2 + (1/2) * n - 1) % 3 = 2 :=
sorry

end expression_is_integer_expression_modulo_3_l8_8801


namespace intersection_A_B_l8_8502

-- Define sets A and B
def A : Set ℝ := { x | -2 < x ∧ x < 4 }
def B : Set ℝ := { 2, 3, 4, 5 }

-- State the theorem about the intersection A ∩ B
theorem intersection_A_B : A ∩ B = { 2, 3 } :=
by
  sorry

end intersection_A_B_l8_8502


namespace horner_value_v2_l8_8660

def poly (x : ℤ) : ℤ := 208 + 9 * x^2 + 6 * x^4 + x^6

theorem horner_value_v2 : poly (-4) = ((((0 + -4) * -4 + 6) * -4 + 9) * -4 + 208) :=
by
  sorry

end horner_value_v2_l8_8660


namespace multiply_transformed_l8_8465

theorem multiply_transformed : (268 * 74 = 19832) → (2.68 * 0.74 = 1.9832) :=
by
  intro h
  sorry

end multiply_transformed_l8_8465


namespace find_lambda_l8_8072

variables {K : Type*} [Field K] {V : Type*} [AddCommGroup V] [Module K V]
variables (a b : V) (λ : K)

-- Assuming vectors a and b are not parallel
def not_parallel (a b : V) : Prop := ¬ (∃ (k : K), k ≠ 0 ∧ a = k • b)

-- Given Condition
axiom H1 : not_parallel a b
axiom H2 : ∃ (t : K), λ • a + b = t • (a + 2 • b)

-- Theorem to prove that λ = 1/2
theorem find_lambda (H1 : not_parallel a b) (H2 : ∃ (t : K), λ • a + b = t • (a + 2 • b)) : λ = (1 / 2) := 
  sorry

end find_lambda_l8_8072


namespace circle_area_l8_8849

open Real

def given_equation (r θ : ℝ) : Prop := r = 3 * cos θ - 4 * sin θ

theorem circle_area (r θ : ℝ) (h : given_equation r θ) : 
  ∃ (c : ℝ × ℝ) (R : ℝ), c = (3 / 2, -2) ∧ R = 5 / 2 ∧ π * R^2 = 25 / 4 * π :=
sorry

end circle_area_l8_8849


namespace find_p5_l8_8778

-- Definitions based on conditions from the problem
def p (x : ℝ) : ℝ :=
  x^4 - 10 * x^3 + 35 * x^2 - 50 * x + 18  -- this construction ensures it's a quartic monic polynomial satisfying provided conditions

-- The main theorem we want to prove
theorem find_p5 :
  p 1 = 3 ∧ p 2 = 7 ∧ p 3 = 13 ∧ p 4 = 21 → p 5 = 51 :=
by
  -- The proof will be inserted here later
  sorry

end find_p5_l8_8778


namespace compare_negative_fractions_l8_8301

theorem compare_negative_fractions : (- (1 / 3 : ℝ)) < (- (1 / 4 : ℝ)) :=
sorry

end compare_negative_fractions_l8_8301


namespace sector_area_l8_8035

theorem sector_area (r l : ℝ) (h1 : l + 2 * r = 10) (h2 : l = 2 * r) : 
  (1 / 2) * l * r = 25 / 4 :=
by 
  sorry

end sector_area_l8_8035


namespace sample_capacity_l8_8557

theorem sample_capacity 
  (n : ℕ) 
  (model_A : ℕ) 
  (model_B model_C : ℕ) 
  (ratio_A ratio_B ratio_C : ℕ)
  (r_A : ratio_A = 2)
  (r_B : ratio_B = 3)
  (r_C : ratio_C = 5)
  (total_production_ratio : ratio_A + ratio_B + ratio_C = 10)
  (items_model_A : model_A = 15)
  (proportion : (model_A : ℚ) / (ratio_A : ℚ) = (n : ℚ) / 10) :
  n = 75 :=
by sorry

end sample_capacity_l8_8557


namespace prove_a_zero_l8_8349

noncomputable def f (x a : ℝ) := (x + a) * log ((2 * x - 1) / (2 * x + 1))

theorem prove_a_zero (a : ℝ) : 
  (∀ x, f (-x a) = f (x a)) → a = 0 :=
by 
  sorry

end prove_a_zero_l8_8349


namespace find_minimum_x2_x1_l8_8478

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.log x + 1 / 2

theorem find_minimum_x2_x1 (x1 : ℝ) :
  ∃ x2 : {r : ℝ // 0 < r}, f x1 = g x2 → (x2 - x1) ≥ 1 + Real.log 2 / 2 :=
by
  -- Proof
  sorry

end find_minimum_x2_x1_l8_8478


namespace alex_sandwich_count_l8_8003

theorem alex_sandwich_count :
  (Nat.choose 10 1) * (Nat.choose 12 2) * (Nat.choose 5 1) = 3300 :=
by
  sorry

end alex_sandwich_count_l8_8003


namespace area_of_circle_l8_8852

theorem area_of_circle:
  (∃ (r : ℝ) (θ : ℝ), r = 3 * Real.cos θ - 4 * Real.sin θ) → ∃ area: ℝ, area = (25/4) * Real.pi :=
sorry

end area_of_circle_l8_8852


namespace exists_three_fitting_rectangles_l8_8661

-- Define the fitting condition
def fits_inside (a b c d : ℕ) : Prop :=
  (a ≤ c ∧ b ≤ d) ∨ (a ≤ d ∧ b ≤ c)

-- Set S and its bounds
def S (n : ℕ) : Finset (ℕ × ℕ) := 
  (Finset.product (Finset.range n.succ) (Finset.range n.succ)).filter (λ p, p.1 ≠ 0 ∧ p.2 ≠ 0)

-- Main theorem
theorem exists_three_fitting_rectangles (S : Finset (ℕ × ℕ)) (hS : S.card = 2019 ∧ ∀ r ∈ S, r.1 ≤ 2018 ∧ r.2 ≤ 2018) :
  ∃ A B C, A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ fits_inside A.1 A.2 B.1 B.2 ∧ fits_inside B.1 B.2 C.1 C.2 :=
sorry

end exists_three_fitting_rectangles_l8_8661


namespace union_M_N_inter_complement_M_N_union_complement_M_N_l8_8950

open Set

variable (U : Set ℝ) (M : Set ℝ) (N : Set ℝ)

noncomputable def universal_set := U = univ

def set_M := M = {x : ℝ | x ≤ 3}
def set_N := N = {x : ℝ | x < 1}

theorem union_M_N (hU : universal_set U) (hM : set_M M) (hN : set_N N) :
    M ∪ N = {x : ℝ | x ≤ 3} :=
by sorry

theorem inter_complement_M_N (hU : universal_set U) (hM : set_M M) (hN : set_N N) :
    (U \ M) ∩ N = ∅ :=
by sorry

theorem union_complement_M_N (hU : universal_set U) (hM : set_M M) (hN : set_N N) :
    (U \ M) ∪ (U \ N) = {x : ℝ | x ≥ 1} :=
by sorry

end union_M_N_inter_complement_M_N_union_complement_M_N_l8_8950


namespace bellas_score_l8_8794

theorem bellas_score (sum_19 : ℕ) (sum_20 : ℕ) (avg_19 : ℕ) (avg_20 : ℕ) (n_19 : ℕ) (n_20 : ℕ) :
  avg_19 = 82 → avg_20 = 85 → n_19 = 19 → n_20 = 20 → sum_19 = n_19 * avg_19 → sum_20 = n_20 * avg_20 →
  sum_20 - sum_19 = 142 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end bellas_score_l8_8794


namespace f_monotonic_intervals_f_extreme_values_l8_8751

def f (x : ℝ) : ℝ := x^3 - 12 * x

-- Monotonicity intervals
theorem f_monotonic_intervals (x : ℝ) : 
  (x < -2 → deriv f x > 0) ∧ 
  (-2 < x ∧ x < 2 → deriv f x < 0) ∧ 
  (2 < x → deriv f x > 0) := 
sorry

-- Extreme values
theorem f_extreme_values :
  f (-2) = 16 ∧ f (2) = -16 :=
sorry

end f_monotonic_intervals_f_extreme_values_l8_8751


namespace points_on_fourth_board_l8_8132

-- Definition of the points scored on each dartboard
def points_board_1 : ℕ := 30
def points_board_2 : ℕ := 38
def points_board_3 : ℕ := 41

-- Statement to prove that points on the fourth board are 34
theorem points_on_fourth_board : (points_board_1 + points_board_2) / 2 = 34 :=
by
  -- Given points on first and second boards
  have h1 : points_board_1 + points_board_2 = 68 := by rfl
  sorry

end points_on_fourth_board_l8_8132


namespace lily_typing_speed_l8_8211

-- Define the conditions
def wordsTyped : ℕ := 255
def totalMinutes : ℕ := 19
def breakTime : ℕ := 2
def typingInterval : ℕ := 10
def effectiveMinutes : ℕ := totalMinutes - breakTime

-- Define the number of words typed in effective minutes
def wordsPerMinute (words : ℕ) (minutes : ℕ) : ℕ := words / minutes

-- Statement to be proven
theorem lily_typing_speed : wordsPerMinute wordsTyped effectiveMinutes = 15 :=
by
  -- proof goes here
  sorry

end lily_typing_speed_l8_8211


namespace transfer_people_eq_l8_8687

theorem transfer_people_eq : ∃ x : ℕ, 22 + x = 2 * (26 - x) := 
by 
  -- hypothesis and equation statement
  sorry

end transfer_people_eq_l8_8687


namespace original_volume_of_ice_l8_8896

variable (V : ℝ) 

theorem original_volume_of_ice (h1 : V * (1 / 4) * (1 / 4) = 0.25) : V = 4 :=
  sorry

end original_volume_of_ice_l8_8896


namespace set_intersection_l8_8512

noncomputable def U : Set ℝ := Set.univ
noncomputable def M : Set ℝ := {x | x < 3}
noncomputable def N : Set ℝ := {y | y > 2}
noncomputable def CU_M : Set ℝ := {x | x ≥ 3}

theorem set_intersection :
  (CU_M ∩ N) = {x | x ≥ 3} := by
  sorry

end set_intersection_l8_8512


namespace combined_prism_volume_is_66_l8_8289

noncomputable def volume_of_combined_prisms
  (length_rect : ℝ) (width_rect : ℝ) (height_rect : ℝ)
  (base_tri : ℝ) (height_tri : ℝ) (length_tri : ℝ) : ℝ :=
  let volume_rect := length_rect * width_rect * height_rect
  let area_tri := (1 / 2) * base_tri * height_tri
  let volume_tri := area_tri * length_tri
  volume_rect + volume_tri

theorem combined_prism_volume_is_66 :
  volume_of_combined_prisms 6 4 2 3 3 4 = 66 := by
  sorry

end combined_prism_volume_is_66_l8_8289


namespace sector_angle_l8_8487

-- Define the conditions
def perimeter (r l : ℝ) : ℝ := 2 * r + l
def arc_length (α r : ℝ) : ℝ := α * r

-- Define the problem statement
theorem sector_angle (perimeter_eq : perimeter 1 l = 4) (arc_length_eq : arc_length α 1 = l) : α = 2 := 
by 
  -- remainder of the proof can be added here 
  sorry

end sector_angle_l8_8487


namespace boat_speed_l8_8116

theorem boat_speed (v : ℝ) (h1 : 5 + v = 30) : v = 25 :=
by 
  -- Solve for the speed of the second boat
  sorry

end boat_speed_l8_8116


namespace total_bill_l8_8701

-- Definitions from conditions
def num_people : ℕ := 3
def amount_per_person : ℕ := 45

-- Mathematical proof problem statement
theorem total_bill : num_people * amount_per_person = 135 := by
  sorry

end total_bill_l8_8701


namespace no_real_roots_iff_range_m_l8_8365

open Real

theorem no_real_roots_iff_range_m (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + (m + 3) ≠ 0) ↔ (-2 < m ∧ m < 6) :=
by
  sorry

end no_real_roots_iff_range_m_l8_8365


namespace value_of_expression_l8_8188

theorem value_of_expression (m : ℝ) 
  (h : m^2 - 2 * m - 1 = 0) : 3 * m^2 - 6 * m + 2020 = 2023 := 
by 
  /- Proof is omitted -/
  sorry

end value_of_expression_l8_8188


namespace cos_eq_neg_four_fifths_of_tan_l8_8040

theorem cos_eq_neg_four_fifths_of_tan (α : ℝ) (h_tan : Real.tan α = 3 / 4) (h_interval : α ∈ Set.Ioo Real.pi (3 * Real.pi / 2)) :
  Real.cos α = -4 / 5 :=
sorry

end cos_eq_neg_four_fifths_of_tan_l8_8040


namespace smallest_prime_divisor_of_n_is_2_l8_8885

def a := 3^19
def b := 11^13
def n := a + b

theorem smallest_prime_divisor_of_n_is_2 : nat.prime_divisors n = {2} :=
by
  -- The proof is not needed as per instructions.
  -- We'll add a placeholder sorry to complete the statement.
  sorry

end smallest_prime_divisor_of_n_is_2_l8_8885


namespace smallest_is_C_l8_8250

def A : ℚ := 1/2
def B : ℚ := 9/10
def C : ℚ := 2/5

theorem smallest_is_C : min (min A B) C = C := 
by
  sorry

end smallest_is_C_l8_8250


namespace knitting_time_is_correct_l8_8784

-- Definitions of the conditions
def time_per_hat : ℕ := 2
def time_per_scarf : ℕ := 3
def time_per_mitten : ℕ := 1
def time_per_sock : ℕ := 3 / 2 -- fractional time in hours
def time_per_sweater : ℕ := 6
def number_of_grandchildren : ℕ := 3

-- Compute total time for one complete outfit
def time_per_outfit : ℕ := time_per_hat + time_per_scarf + (time_per_mitten * 2) + (time_per_sock * 2) + time_per_sweater

-- Compute total time for all outfits
def total_knitting_time : ℕ := number_of_grandchildren * time_per_outfit

-- Prove that total knitting time is 48 hours
theorem knitting_time_is_correct : total_knitting_time = 48 := by
  unfold total_knitting_time time_per_outfit
  norm_num
  sorry

end knitting_time_is_correct_l8_8784


namespace total_pay_of_two_employees_l8_8658

theorem total_pay_of_two_employees
  (Y_pay : ℝ)
  (X_pay : ℝ)
  (h1 : Y_pay = 280)
  (h2 : X_pay = 1.2 * Y_pay) :
  X_pay + Y_pay = 616 :=
by
  sorry

end total_pay_of_two_employees_l8_8658


namespace mouse_lives_difference_l8_8422

-- Definitions of variables and conditions
def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := 13

-- Theorem to prove
theorem mouse_lives_difference : mouse_lives - dog_lives = 7 := by
  -- This is where the proof would go, but we use sorry to skip it.
  sorry

end mouse_lives_difference_l8_8422


namespace will_remaining_balance_l8_8416

theorem will_remaining_balance :
  ∀ (initial_money conversion_fee : ℝ) 
    (exchange_rate : ℝ)
    (sweater_cost tshirt_cost shoes_cost hat_cost socks_cost : ℝ)
    (shoes_refund_percentage : ℝ)
    (discount_percentage sales_tax_percentage : ℝ),
  initial_money = 74 →
  conversion_fee = 2 →
  exchange_rate = 1.5 →
  sweater_cost = 13.5 →
  tshirt_cost = 16.5 →
  shoes_cost = 45 →
  hat_cost = 7.5 →
  socks_cost = 6 →
  shoes_refund_percentage = 0.85 →
  discount_percentage = 0.10 →
  sales_tax_percentage = 0.05 →
  (initial_money - conversion_fee) * exchange_rate -
  ((sweater_cost + tshirt_cost + shoes_cost + hat_cost + socks_cost - shoes_cost * shoes_refund_percentage) *
   (1 - discount_percentage) * (1 + sales_tax_percentage)) /
  exchange_rate = 39.87 :=
by
  intros initial_money conversion_fee exchange_rate
        sweater_cost tshirt_cost shoes_cost hat_cost socks_cost
        shoes_refund_percentage discount_percentage sales_tax_percentage
        h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end will_remaining_balance_l8_8416


namespace domain_width_p_l8_8759

variable (f : ℝ → ℝ)
variable (h_dom_f : ∀ x, -12 ≤ x ∧ x ≤ 12 → f x = f x)

noncomputable def p (x : ℝ) : ℝ := f (x / 3)

theorem domain_width_p : (width : ℝ) = 72 :=
by
  let domain_p : Set ℝ := {x | -36 ≤ x ∧ x ≤ 36}
  have : width = 72 := sorry
  exact this

end domain_width_p_l8_8759


namespace problem1_problem2_l8_8708

-- Problem 1
theorem problem1 : 2023^2 - 2024 * 2022 = 1 :=
sorry

-- Problem 2
variables (a b c : ℝ)
theorem problem2 : 5 * a^2 * b^3 * (-1/10 * a * b^3 * c) / (1/2 * a * b^2)^3 = -4 * c :=
sorry

end problem1_problem2_l8_8708


namespace solid_triangle_front_view_l8_8682

def is_triangle_front_view (solid : ℕ) : Prop :=
  solid = 1 ∨ solid = 2 ∨ solid = 3 ∨ solid = 5

theorem solid_triangle_front_view (s : ℕ) (h : s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 4 ∨ s = 5 ∨ s = 6):
  is_triangle_front_view s ↔ (s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 5) :=
by
  sorry

end solid_triangle_front_view_l8_8682


namespace problem_inequality_l8_8741

variable (x y z : ℝ)

theorem problem_inequality (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) :
  2 * (x^3 + y^3 + z^3) ≥ x^2 * y + x^2 * z + y^2 * z + y^2 * x + z^2 * x + z^2 * y := by
  sorry

end problem_inequality_l8_8741


namespace min_width_of_garden_l8_8376

theorem min_width_of_garden (w : ℝ) (h : 0 < w) (h1 : w * (w + 20) ≥ 120) : w ≥ 4 :=
sorry

end min_width_of_garden_l8_8376


namespace red_pencil_count_l8_8101

-- Definitions for provided conditions
def blue_pencils : ℕ := 20
def ratio : ℕ × ℕ := (5, 3)
def red_pencils (blue : ℕ) (rat : ℕ × ℕ) : ℕ := (blue / rat.fst) * rat.snd

-- Theorem statement
theorem red_pencil_count : red_pencils blue_pencils ratio = 12 := 
by
  sorry

end red_pencil_count_l8_8101


namespace remainder_div_150_by_4_eq_2_l8_8169

theorem remainder_div_150_by_4_eq_2 :
  (∃ k : ℕ, k > 0 ∧ 120 % k^2 = 24) → 150 % 4 = 2 :=
by
  intro h
  sorry

end remainder_div_150_by_4_eq_2_l8_8169


namespace total_visitors_over_two_days_l8_8429

-- Definitions of the conditions
def visitors_on_Saturday : ℕ := 200
def additional_visitors_on_Sunday : ℕ := 40

-- Statement of the problem
theorem total_visitors_over_two_days :
  let visitors_on_Sunday := visitors_on_Saturday + additional_visitors_on_Sunday
  let total_visitors := visitors_on_Saturday + visitors_on_Sunday
  total_visitors = 440 :=
by
  let visitors_on_Sunday := visitors_on_Saturday + additional_visitors_on_Sunday
  let total_visitors := visitors_on_Saturday + visitors_on_Sunday
  sorry

end total_visitors_over_two_days_l8_8429


namespace smallest_prime_divisor_of_sum_l8_8888

theorem smallest_prime_divisor_of_sum (h1 : Nat.odd (Nat.pow 3 19))
                                      (h2 : Nat.odd (Nat.pow 11 13))
                                      (h3 : ∀ a b : Nat, Nat.odd a → Nat.odd b → Nat.even (a + b)) :
  Nat.minFac (Nat.pow 3 19 + Nat.pow 11 13) = 2 :=
by
  -- placeholder proof
  sorry

end smallest_prime_divisor_of_sum_l8_8888


namespace perimeter_of_triangle_l8_8060

-- Defining the basic structure of the problem
theorem perimeter_of_triangle (A B C : Type)
  (distance_AB distance_AC distance_BC : ℝ)
  (angle_B : ℝ)
  (h1 : distance_AB = distance_AC)
  (h2 : angle_B = 60)
  (h3 : distance_BC = 4) :
  distance_AB + distance_AC + distance_BC = 12 :=
by 
  sorry

end perimeter_of_triangle_l8_8060


namespace geometric_sequence_a5_eq_2_l8_8175

-- Define geometric sequence and the properties
noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

-- Given conditions
variables {a : ℕ → ℝ} {q : ℝ}

-- Roots of given quadratic equation
variables (h1 : a 3 = 1 ∨ a 3 = 4 / 1) (h2 : a 7 = 4 / a 3)
variables (h3 : q > 0) (h4 : geometric_seq a q)

-- Prove that a5 = 2
theorem geometric_sequence_a5_eq_2 : a 5 = 2 :=
sorry

end geometric_sequence_a5_eq_2_l8_8175


namespace find_k_and_angle_l8_8455

def vector := ℝ × ℝ

def dot_product (u v: vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def orthogonal (u v: vector) : Prop :=
  dot_product u v = 0

theorem find_k_and_angle (k : ℝ) :
  let a : vector := (3, -1)
  let b : vector := (1, k)
  orthogonal a b →
  (k = 3 ∧ dot_product (3+1, -1+3) (3-1, -1-3) = 0) :=
by
  intros
  sorry

end find_k_and_angle_l8_8455


namespace area_of_circle_l8_8851

theorem area_of_circle:
  (∃ (r : ℝ) (θ : ℝ), r = 3 * Real.cos θ - 4 * Real.sin θ) → ∃ area: ℝ, area = (25/4) * Real.pi :=
sorry

end area_of_circle_l8_8851


namespace remainder_of_base12_2563_mod_17_l8_8257

-- Define the base-12 number 2563 in decimal.
def base12_to_decimal : ℕ := 2 * 12^3 + 5 * 12^2 + 6 * 12^1 + 3 * 12^0

-- Define the number 17.
def divisor : ℕ := 17

-- Prove that the remainder when base12_to_decimal is divided by divisor is 1.
theorem remainder_of_base12_2563_mod_17 : base12_to_decimal % divisor = 1 :=
by
  sorry

end remainder_of_base12_2563_mod_17_l8_8257


namespace probability_at_least_one_alarm_on_time_l8_8114

noncomputable def P_alarm_A_on : ℝ := 0.80
noncomputable def P_alarm_B_on : ℝ := 0.90

theorem probability_at_least_one_alarm_on_time :
  (1 - (1 - P_alarm_A_on) * (1 - P_alarm_B_on)) = 0.98 :=
by
  sorry

end probability_at_least_one_alarm_on_time_l8_8114


namespace probability_of_point_A_in_fourth_quadrant_l8_8106

noncomputable def probability_of_fourth_quadrant : ℚ :=
  let cards := {0, -1, 2, -3}
  let total_outcomes := cards.to_finset.powerset.filter (λ s, s.card = 2).card
  let favorable_outcomes := cards.to_finset.powerset.filter (λ s, s.card = 2 ∧ s.contains 2 ∧ s.contains -1 ∨ s.contains -3).card
  favorable_outcomes / total_outcomes

theorem probability_of_point_A_in_fourth_quadrant :
  probability_of_fourth_quadrant = 1 / 6 :=
sorry

end probability_of_point_A_in_fourth_quadrant_l8_8106


namespace shirts_made_today_l8_8146

def shirts_per_minute : ℕ := 6
def minutes_yesterday : ℕ := 12
def total_shirts : ℕ := 156
def shirts_yesterday : ℕ := shirts_per_minute * minutes_yesterday
def shirts_today : ℕ := total_shirts - shirts_yesterday

theorem shirts_made_today :
  shirts_today = 84 :=
by
  sorry

end shirts_made_today_l8_8146


namespace expressions_equality_l8_8526

-- Assumptions that expressions (1) and (2) are well-defined (denominators are non-zero)
variable {a b c m n p : ℝ}
variable (h1 : m ≠ 0)
variable (h2 : bp + cn ≠ 0)
variable (h3 : n ≠ 0)
variable (h4 : ap + cm ≠ 0)

-- Main theorem statement
theorem expressions_equality
  (hS : (a / m) + (bc + np) / (bp + cn) = 0) :
  (b / n) + (ac + mp) / (ap + cm) = 0 :=
  sorry

end expressions_equality_l8_8526


namespace trapezoid_area_l8_8827

variable (a b : ℝ) (h1 : a > b)

theorem trapezoid_area (h2 : ∃ (angle1 angle2 : ℝ), angle1 = 30 ∧ angle2 = 45) : 
  (1/4) * ((a^2 - b^2) * (Real.sqrt 3 - 1)) = 
    ((1/2) * (a + b) * ((b - a) * (Real.sqrt 3 - 1) / 2)) := 
sorry

end trapezoid_area_l8_8827


namespace inverse_of_h_l8_8573

noncomputable def h (x : ℝ) : ℝ := 3 - 7 * x
noncomputable def k (x : ℝ) : ℝ := (3 - x) / 7

theorem inverse_of_h :
  (∀ x : ℝ, h (k x) = x) ∧ (∀ x : ℝ, k (h x) = x) :=
by
  sorry

end inverse_of_h_l8_8573


namespace fenced_area_l8_8832

theorem fenced_area (length_large : ℕ) (width_large : ℕ) 
                    (length_cutout : ℕ) (width_cutout : ℕ) 
                    (h_large : length_large = 20 ∧ width_large = 15)
                    (h_cutout : length_cutout = 4 ∧ width_cutout = 2) : 
                    ((length_large * width_large) - (length_cutout * width_cutout) = 292) := 
by
  sorry

end fenced_area_l8_8832


namespace skylar_current_age_l8_8528

noncomputable def skylar_age_now (donation_start_age : ℕ) (annual_donation total_donation : ℕ) : ℕ :=
  donation_start_age + total_donation / annual_donation

theorem skylar_current_age : skylar_age_now 13 5000 105000 = 34 := by
  -- Proof follows from the conditions
  sorry

end skylar_current_age_l8_8528


namespace union_of_sets_l8_8185

def A : Set ℤ := {1, 2}
def B : Set ℤ := {-1, 0, 1}

theorem union_of_sets : A ∪ B = {-1, 0, 1, 2} :=
by
  sorry

end union_of_sets_l8_8185


namespace largest_number_in_systematic_sample_l8_8738

theorem largest_number_in_systematic_sample (n_products : ℕ) (start : ℕ) (interval : ℕ) (sample_size : ℕ) (largest_number : ℕ)
  (h1 : n_products = 500)
  (h2 : start = 7)
  (h3 : interval = 25)
  (h4 : sample_size = n_products / interval)
  (h5 : sample_size = 20)
  (h6 : largest_number = start + interval * (sample_size - 1))
  (h7 : largest_number = 482) :
  largest_number = 482 := 
  sorry

end largest_number_in_systematic_sample_l8_8738


namespace inverse_f_of_7_l8_8748

def f (x : ℝ) : ℝ := 2 * x^2 + 3

theorem inverse_f_of_7:
  ∀ y : ℝ, f (7) = y ↔ y = 101 :=
by
  sorry

end inverse_f_of_7_l8_8748


namespace infinite_prime_set_exists_l8_8630

noncomputable def P : Set Nat := {p | Prime p ∧ ∃ m : Nat, p ∣ m^2 + 1}

theorem infinite_prime_set_exists :
  ∃ (P : Set Nat), (∀ p ∈ P, Prime p) ∧ (Set.Infinite P) ∧ 
  (∀ (p : Nat) (hp : p ∈ P) (k : ℕ),
    ∃ (m : Nat), p^k ∣ m^2 + 1 ∧ ¬(p^(k+1) ∣ m^2 + 1)) :=
sorry

end infinite_prime_set_exists_l8_8630


namespace circle_area_polar_eq_l8_8836

theorem circle_area_polar_eq (r θ : ℝ) :
    (r = 3 * Real.cos θ - 4 * Real.sin θ) → (π * (5 / 2) ^ 2 = 25 * π / 4) :=
by 
  sorry

end circle_area_polar_eq_l8_8836


namespace complex_sum_identity_l8_8208

theorem complex_sum_identity (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^100 + z^101 + z^102 + z^103 + z^104 = -1 := 
by 
  sorry

end complex_sum_identity_l8_8208


namespace constant_term_expansion_l8_8770

-- Defining the binomial theorem term
noncomputable def binomial_coeff (n k : ℕ) : ℕ := 
  Nat.choose n k

-- The general term of the binomial expansion (2sqrt(x) - 1/x)^6
noncomputable def general_term (r : ℕ) (x : ℝ) : ℝ :=
  binomial_coeff 6 r * (-1)^r * (2^(6-r)) * x^((6 - 3 * r) / 2)

-- Problem statement: Show that the constant term in the expansion is 240
theorem constant_term_expansion :
  (∃ r : ℕ, (6 - 3 * r) / 2 = 0 ∧ 
            general_term r arbitrary = 240) :=
sorry

end constant_term_expansion_l8_8770


namespace original_chairs_count_l8_8278

theorem original_chairs_count (n : ℕ) (m : ℕ) :
  (∀ k : ℕ, (k % 4 = 0 → k * (2 * n / 4) = k * (3 * n / 4) ) ∧ 
  (m = (4 / 2) * 15) ∧ (n = (4 * m / (2 * m)) - ((2 * m) / m)) ∧ 
  n + (n + 9) = 72) → n = 63 :=
by
  sorry

end original_chairs_count_l8_8278


namespace cosine_identity_l8_8180

theorem cosine_identity (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 4) : 
  Real.cos (2 * α - Real.pi / 3) = 7 / 8 := by 
  sorry

end cosine_identity_l8_8180


namespace sequence_a5_l8_8614

theorem sequence_a5 : 
    ∃ (a : ℕ → ℚ), 
    a 1 = 1 / 3 ∧ 
    (∀ (n : ℕ), n ≥ 2 → a n = (-1 : ℚ)^n * 2 * a (n - 1)) ∧ 
    a 5 = -16 / 3 := 
sorry

end sequence_a5_l8_8614


namespace mass_of_added_water_with_temp_conditions_l8_8270

theorem mass_of_added_water_with_temp_conditions
  (m_l : ℝ) (t_pi t_B t : ℝ) (c_B c_l lambda : ℝ) :
  m_l = 0.05 →
  t_pi = -10 →
  t_B = 10 →
  t = 0 →
  c_B = 4200 →
  c_l = 2100 →
  lambda = 3.3 * 10^5 →
  (0.0028 ≤ (2.1 * m_l * 10 + lambda * m_l) / (42 * 10) 
  ∧ (2.1 * m_l * 10) / (42 * 10) ≤ 0.418) :=
by
  sorry

end mass_of_added_water_with_temp_conditions_l8_8270


namespace max_matching_pairs_l8_8214

theorem max_matching_pairs (total_pairs : ℕ) (lost_individual : ℕ) (left_pair : ℕ) : 
  total_pairs = 25 ∧ lost_individual = 9 → left_pair = 20 :=
by
  sorry

end max_matching_pairs_l8_8214


namespace quadratic_root_m_value_l8_8744

theorem quadratic_root_m_value (m : ℝ) (x : ℝ) (h : x = 1) (hx : x^2 + m * x + 2 = 0) : m = -3 :=
by
  sorry

end quadratic_root_m_value_l8_8744


namespace students_range_score_100_140_l8_8273

noncomputable def normal_distribution_condition (X : Type) [norm : NormedAddCommGroup X] [measure_space X]
  (μ : ℝ) (σ : ℝ) (t : ℝ) :=
  ∀ (x : X), (measure_prob (measure_normal μ σ) {y | y > t} = 0.2)

theorem students_range_score_100_140 :
  ∀ (X : Type) [norm : NormedAddCommGroup X] [measure_space X]
    (μ σ : ℝ),
    let students := 50 in
    let P := measure_normal μ σ in
    normal_distribution_condition X μ σ 140 →
    (X = ℝ ∧ μ = 120 ∧ P X = μ ∧ (students * (measure_prob P {y | y > 140} = 0.2)) →
    ∑ in {y | 100 ≤ y ∧ y ≤ 140} = 30) :=
by
  sorry

end students_range_score_100_140_l8_8273


namespace student_weight_l8_8485

theorem student_weight (S R : ℕ) (h1 : S - 5 = 2 * R) (h2 : S + R = 116) : S = 79 :=
sorry

end student_weight_l8_8485


namespace num_distinct_terms_expansion_a_b_c_10_l8_8372

-- Define the expansion of (a+b+c)^10
def num_distinct_terms_expansion (n : ℕ) : ℕ :=
  Nat.choose (n + 3 - 1) (3 - 1)

-- Theorem statement
theorem num_distinct_terms_expansion_a_b_c_10 : num_distinct_terms_expansion 10 = 66 :=
by
  sorry

end num_distinct_terms_expansion_a_b_c_10_l8_8372


namespace paperclips_in_64_volume_box_l8_8288

def volume_16 : ℝ := 16
def volume_32 : ℝ := 32
def volume_64 : ℝ := 64
def paperclips_50 : ℝ := 50
def paperclips_100 : ℝ := 100

theorem paperclips_in_64_volume_box :
  ∃ (k p : ℝ), 
  (paperclips_50 = k * volume_16^p) ∧ 
  (paperclips_100 = k * volume_32^p) ∧ 
  (200 = k * volume_64^p) :=
by
  sorry

end paperclips_in_64_volume_box_l8_8288


namespace foldable_polygons_count_l8_8635

def isValidFolding (base_positions : Finset Nat) (additional_position : Nat) : Prop :=
  ∃ (valid_positions : Finset Nat), valid_positions = {4, 5, 6, 7, 8, 9} ∧ additional_position ∈ valid_positions

theorem foldable_polygons_count : 
  ∃ (valid_additional_positions : Finset Nat), valid_additional_positions = {4, 5, 6, 7, 8, 9} ∧ valid_additional_positions.card = 6 := 
by
  sorry

end foldable_polygons_count_l8_8635


namespace circle_area_and_circumference_changes_l8_8296

noncomputable section

structure Circle :=
  (r : ℝ)

def area (c : Circle) : ℝ := Real.pi * c.r^2

def circumference (c : Circle) : ℝ := 2 * Real.pi * c.r

def percentage_change (original new : ℝ) : ℝ :=
  ((original - new) / original) * 100

theorem circle_area_and_circumference_changes
  (r1 r2 : ℝ) (c1 : Circle := {r := r1}) (c2 : Circle := {r := r2})
  (h1 : r1 = 5) (h2 : r2 = 4) :
  let original_area := area c1
  let new_area := area c2
  let original_circumference := circumference c1
  let new_circumference := circumference c2
  percentage_change original_area new_area = 36 ∧
  new_circumference = 8 * Real.pi ∧
  percentage_change original_circumference new_circumference = 20 :=
by
  sorry

end circle_area_and_circumference_changes_l8_8296


namespace sum_of_cubes_l8_8760

theorem sum_of_cubes (x y : ℝ) (hx : x + y = 10) (hxy : x * y = 12) : x^3 + y^3 = 640 := 
by
  sorry

end sum_of_cubes_l8_8760


namespace tree_height_by_time_boy_is_36_inches_l8_8282

noncomputable def final_tree_height : ℕ :=
  let T₀ := 16
  let B₀ := 24
  let Bₓ := 36
  let boy_growth := Bₓ - B₀
  let tree_growth := 2 * boy_growth
  T₀ + tree_growth

theorem tree_height_by_time_boy_is_36_inches :
  final_tree_height = 40 :=
by
  sorry

end tree_height_by_time_boy_is_36_inches_l8_8282


namespace part_I_part_II_l8_8071

-- Define the function f
def f (x: ℝ) : ℝ := abs (x - 1) - 2 * abs (x + 1)

-- The conditions and questions transformed into Lean statements
theorem part_I : ∃ m, (∀ x: ℝ, f x ≤ m) ∧ (m = f (-1)) ∧ (m = 2) := by
  sorry

theorem part_II (a b c : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c) (h₁ : a^2 + 3 * b^2 + 2 * c^2 = 2) : 
  ∃ n, (∀ a b c : ℝ, (0 < a ∧ 0 < b ∧ 0 < c) ∧ (a^2 + 3 * b^2 + 2 * c^2 = 2) → ab + 2 * bc ≤ n) ∧ (n = 1) := by
  sorry

end part_I_part_II_l8_8071


namespace product_not_perfect_power_l8_8565

theorem product_not_perfect_power (n : ℕ) : ¬∃ (k : ℕ) (a : ℤ), k > 1 ∧ n * (n + 1) = a^k := by
  sorry

end product_not_perfect_power_l8_8565


namespace average_speed_of_car_l8_8266

theorem average_speed_of_car 
  (speed_first_hour : ℕ)
  (speed_second_hour : ℕ)
  (total_time : ℕ)
  (h1 : speed_first_hour = 90)
  (h2 : speed_second_hour = 40)
  (h3 : total_time = 2) : 
  (speed_first_hour + speed_second_hour) / total_time = 65 := 
by
  sorry

end average_speed_of_car_l8_8266


namespace salary_of_A_l8_8103

theorem salary_of_A (A B : ℝ) (h1 : A + B = 7000) (h2 : 0.05 * A = 0.15 * B) : A = 5250 := 
by 
  sorry

end salary_of_A_l8_8103


namespace train_length_correct_l8_8698

noncomputable def train_length (speed_kmph: ℝ) (time_sec: ℝ) : ℝ :=
  let speed_mps := speed_kmph * (5 / 18)
  speed_mps * time_sec

theorem train_length_correct : train_length 250 12 = 833.28 := by
  sorry

end train_length_correct_l8_8698


namespace total_expense_in_decade_l8_8548

/-- Definition of yearly expense on car insurance -/
def yearly_expense : ℕ := 2000

/-- Definition of the number of years in a decade -/
def years_in_decade : ℕ := 10

/-- Proof that the total expense in a decade is 20000 dollars -/
theorem total_expense_in_decade : yearly_expense * years_in_decade = 20000 :=
by
  sorry

end total_expense_in_decade_l8_8548


namespace num_integers_n_with_properties_l8_8482

theorem num_integers_n_with_properties :
  ∃ (N : Finset ℕ), N.card = 50 ∧
  ∀ n ∈ N, n < 150 ∧
    ∃ (m : ℕ), (∃ k, n = 2*k + 1 ∧ m = k*(k+1)) ∧ ¬ (3 ∣ m) :=
sorry

end num_integers_n_with_properties_l8_8482


namespace simon_spending_l8_8433

-- Assume entities and their properties based on the problem
def kabobStickCubes : Nat := 4
def slabCost : Nat := 25
def slabCubes : Nat := 80
def kabobSticksNeeded : Nat := 40

-- Theorem statement based on the problem analysis
theorem simon_spending : 
  (kabobSticksNeeded / (slabCubes / kabobStickCubes)) * slabCost = 50 := by
  sorry

end simon_spending_l8_8433


namespace contrapositive_statement_l8_8643

theorem contrapositive_statement (x : ℝ) : (x ≤ -3 → x < 0) → (x ≥ 0 → x > -3) := 
by
  sorry

end contrapositive_statement_l8_8643


namespace multiply_and_divide_equiv_l8_8672

/-- Defines the operation of first multiplying by 4/5 and then dividing by 4/7 -/
def multiply_and_divide (x : ℚ) : ℚ :=
  (x * (4 / 5)) / (4 / 7)

/-- Statement to prove the operation is equivalent to multiplying by 7/5 -/
theorem multiply_and_divide_equiv (x : ℚ) : 
  multiply_and_divide x = x * (7 / 5) :=
by 
  -- This requires a proof, which we can assume here
  sorry

end multiply_and_divide_equiv_l8_8672


namespace problem_proof_l8_8307

open Matrix

noncomputable def proof_example : Prop :=
  ∃ c d : ℚ, (matrix.mul (λ i j, if i = 0 then if j = 0 then 2 else -2 else if j = 0 then c else d) 
                       (λ i j, if i = 0 then if j = 0 then 2 else -2 else if j = 0 then c else d) 
          = (1 : ℚ) • (1 : matrix (fin 2) (fin 2) ℚ)) ∧ c = 3/2 ∧ d = -2

theorem problem_proof : proof_example := by
  sorry

end problem_proof_l8_8307


namespace find_w_l8_8600

theorem find_w (u v w : ℝ) (h1 : 10 * u + 8 * v + 5 * w = 160)
  (h2 : v = u + 3) (h3 : w = 2 * v) : w = 13.5714 := by
  -- The proof would go here, but we leave it empty as per instructions.
  sorry

end find_w_l8_8600


namespace find_m_from_root_l8_8742

theorem find_m_from_root (m : ℝ) : (x : ℝ) = 1 → x^2 + m * x + 2 = 0 → m = -3 :=
by
  sorry

end find_m_from_root_l8_8742


namespace joan_seashells_l8_8978

/-- Prove that Joan has 36 seashells given the initial conditions. -/
theorem joan_seashells :
  let initial_seashells := 79
  let given_mike := 63
  let found_more := 45
  let traded_seashells := 20
  let lost_seashells := 5
  (initial_seashells - given_mike + found_more - traded_seashells - lost_seashells) = 36 :=
by
  sorry

end joan_seashells_l8_8978


namespace daily_evaporation_rate_l8_8427

/-- A statement that verifies the daily water evaporation rate -/
theorem daily_evaporation_rate
  (initial_water : ℝ)
  (evaporation_percentage : ℝ)
  (evaporation_period : ℕ) :
  initial_water = 15 →
  evaporation_percentage = 0.05 →
  evaporation_period = 15 →
  (evaporation_percentage * initial_water / evaporation_period) = 0.05 :=
by
  intros h_water h_percentage h_period
  sorry

end daily_evaporation_rate_l8_8427


namespace circle_area_polar_eq_l8_8838

theorem circle_area_polar_eq (r θ : ℝ) :
    (r = 3 * Real.cos θ - 4 * Real.sin θ) → (π * (5 / 2) ^ 2 = 25 * π / 4) :=
by 
  sorry

end circle_area_polar_eq_l8_8838


namespace negative_x_y_l8_8356

theorem negative_x_y (x y : ℝ) (h1 : x - y > x) (h2 : x + y < y) : x < 0 ∧ y < 0 :=
by
  sorry

end negative_x_y_l8_8356


namespace distance_between_points_l8_8627

theorem distance_between_points : ∀ (A B : ℤ), A = 5 → B = -3 → |A - B| = 8 :=
by
  intros A B hA hB
  rw [hA, hB]
  norm_num

end distance_between_points_l8_8627


namespace hyperbola_eccentricity_asymptotes_l8_8830

theorem hyperbola_eccentricity_asymptotes :
  (∃ e: ℝ, ∃ m: ℝ, 
    (∀ x y, (x^2 / 8 - y^2 / 4 = 1) → e = Real.sqrt 6 / 2 ∧ y = m * x) ∧ 
    (m = Real.sqrt 2 / 2 ∨ m = -Real.sqrt 2 / 2)) :=
sorry

end hyperbola_eccentricity_asymptotes_l8_8830


namespace repeating_decimal_to_fraction_l8_8937

theorem repeating_decimal_to_fraction : 
  (∃ (x : ℚ), x = 7 + 3 / 9) → 7 + 3 / 9 = 22 / 3 :=
by
  intros h
  sorry

end repeating_decimal_to_fraction_l8_8937


namespace circle_area_polar_eq_l8_8835

theorem circle_area_polar_eq (r θ : ℝ) :
    (r = 3 * Real.cos θ - 4 * Real.sin θ) → (π * (5 / 2) ^ 2 = 25 * π / 4) :=
by 
  sorry

end circle_area_polar_eq_l8_8835


namespace ratio_of_bottles_given_to_first_house_l8_8561

theorem ratio_of_bottles_given_to_first_house 
  (total_bottles : ℕ) 
  (bottles_only_cider : ℕ) 
  (bottles_only_beer : ℕ) 
  (bottles_mixed : ℕ) 
  (first_house_bottles : ℕ) 
  (h1 : total_bottles = 180) 
  (h2 : bottles_only_cider = 40) 
  (h3 : bottles_only_beer = 80) 
  (h4 : bottles_mixed = total_bottles - bottles_only_cider - bottles_only_beer) 
  (h5 : first_house_bottles = 90) : 
  first_house_bottles / total_bottles = 1 / 2 :=
by 
  -- Proof goes here
  sorry

end ratio_of_bottles_given_to_first_house_l8_8561


namespace spent_on_veggies_l8_8404

noncomputable def total_amount : ℕ := 167
noncomputable def spent_on_meat : ℕ := 17
noncomputable def spent_on_chicken : ℕ := 22
noncomputable def spent_on_eggs : ℕ := 5
noncomputable def spent_on_dog_food : ℕ := 45
noncomputable def amount_left : ℕ := 35

theorem spent_on_veggies : 
  total_amount - (spent_on_meat + spent_on_chicken + spent_on_eggs + spent_on_dog_food + amount_left) = 43 := 
by 
  sorry

end spent_on_veggies_l8_8404


namespace calculation_1_calculation_2_calculation_3_calculation_4_l8_8924

theorem calculation_1 : -3 - (-4) = 1 :=
by sorry

theorem calculation_2 : -1/3 + (-4/3) = -5/3 :=
by sorry

theorem calculation_3 : (-2) * (-3) * (-5) = -30 :=
by sorry

theorem calculation_4 : 15 / 4 * (-1/4) = -15/16 :=
by sorry

end calculation_1_calculation_2_calculation_3_calculation_4_l8_8924


namespace average_distance_is_600_l8_8789

-- Definitions based on the conditions
def one_lap_distance : ℕ := 200
def johnny_lap_count : ℕ := 4
def mickey_lap_count : ℕ := johnny_lap_count / 2

-- Total distances run by Johnny and Mickey
def johnny_distance : ℕ := johnny_lap_count * one_lap_distance
def mickey_distance : ℕ := mickey_lap_count * one_lap_distance

-- Sum of distances
def total_distance : ℕ := johnny_distance + mickey_distance

-- Number of people
def number_of_people : ℕ := 2

-- Average distance calculation
def average_distance : ℕ := total_distance / number_of_people

-- The theorem to prove: the average distance run by Johnny and Mickey
theorem average_distance_is_600 : average_distance = 600 := by
  sorry

end average_distance_is_600_l8_8789


namespace abs_sum_bound_l8_8452

theorem abs_sum_bound (k : ℝ) : (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 :=
by {
  sorry
}

end abs_sum_bound_l8_8452


namespace g_range_l8_8380

variable {R : Type*} [LinearOrderedRing R]

-- Let y = f(x) be a function defined on R with a period of 1
def periodic (f : R → R) : Prop :=
  ∀ x, f (x + 1) = f x

-- If g(x) = f(x) + 2x
def g (f : R → R) (x : R) : R := f x + 2 * x

-- If the range of g(x) on the interval [1,2] is [-1,5]
def rangeCondition (f : R → R) : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 2 → -1 ≤ g f x ∧ g f x ≤ 5

-- Then the range of the function g(x) on the interval [-2020,2020] is [-4043,4041]
theorem g_range (f : R → R) 
  (hf_periodic : periodic f) 
  (hf_range : rangeCondition f) : 
  ∀ x, -2020 ≤ x ∧ x ≤ 2020 → -4043 ≤ g f x ∧ g f x ≤ 4041 :=
sorry

end g_range_l8_8380


namespace no_perfect_squares_in_sequence_l8_8736

def tau (a : ℕ) : ℕ := sorry -- Define tau function here

def a_seq (k : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then k else tau (a_seq k (n-1))

theorem no_perfect_squares_in_sequence (k : ℕ) (hk : Prime k) :
  ∀ n : ℕ, ∃ m : ℕ, a_seq k n = m * m → False :=
sorry

end no_perfect_squares_in_sequence_l8_8736


namespace simplify_expression_l8_8808

variable (a : Real)

theorem simplify_expression : (-2 * a) * a - (-2 * a)^2 = -6 * a^2 := by
  sorry

end simplify_expression_l8_8808


namespace gcd_m_n_is_one_l8_8407

def m : ℕ := 122^2 + 234^2 + 344^2

def n : ℕ := 123^2 + 235^2 + 343^2

theorem gcd_m_n_is_one : Nat.gcd m n = 1 :=
by
  sorry

end gcd_m_n_is_one_l8_8407


namespace inverse_function_l8_8574

-- Define the function h
def h (x : ℝ) : ℝ := 3 - 7 * x

-- Define the candidate inverse function k
def k (x : ℝ) : ℝ := (3 - x) / 7

-- State the proof problem
theorem inverse_function (x : ℝ) : h (k x) = x := by
  sorry

end inverse_function_l8_8574


namespace smallest_prime_divisor_of_sum_l8_8887

theorem smallest_prime_divisor_of_sum (h1 : Nat.odd (Nat.pow 3 19))
                                      (h2 : Nat.odd (Nat.pow 11 13))
                                      (h3 : ∀ a b : Nat, Nat.odd a → Nat.odd b → Nat.even (a + b)) :
  Nat.minFac (Nat.pow 3 19 + Nat.pow 11 13) = 2 :=
by
  -- placeholder proof
  sorry

end smallest_prime_divisor_of_sum_l8_8887


namespace weighted_average_is_correct_l8_8524

def bag1_pop_kernels := 60
def bag1_total_kernels := 75
def bag2_pop_kernels := 42
def bag2_total_kernels := 50
def bag3_pop_kernels := 25
def bag3_total_kernels := 100
def bag4_pop_kernels := 77
def bag4_total_kernels := 120
def bag5_pop_kernels := 106
def bag5_total_kernels := 150

noncomputable def weighted_average_percentage : ℚ :=
  ((bag1_pop_kernels / bag1_total_kernels * 100 * bag1_total_kernels) +
   (bag2_pop_kernels / bag2_total_kernels * 100 * bag2_total_kernels) +
   (bag3_pop_kernels / bag3_total_kernels * 100 * bag3_total_kernels) +
   (bag4_pop_kernels / bag4_total_kernels * 100 * bag4_total_kernels) +
   (bag5_pop_kernels / bag5_total_kernels * 100 * bag5_total_kernels)) /
  (bag1_total_kernels + bag2_total_kernels + bag3_total_kernels + bag4_total_kernels + bag5_total_kernels)

theorem weighted_average_is_correct : weighted_average_percentage = 60.61 := 
by
  sorry

end weighted_average_is_correct_l8_8524


namespace solution_set_of_inequality_l8_8402

theorem solution_set_of_inequality :
  { x : ℝ | x ^ 2 - 5 * x + 6 ≤ 0 } = { x : ℝ | 2 ≤ x ∧ x ≤ 3 } :=
by 
  sorry

end solution_set_of_inequality_l8_8402


namespace logs_left_after_3_hours_l8_8424

theorem logs_left_after_3_hours : 
  ∀ (initial_logs : ℕ) (burn_rate : ℕ) (add_rate : ℕ) (time : ℕ),
  initial_logs = 6 →
  burn_rate = 3 →
  add_rate = 2 →
  time = 3 →
  initial_logs + (add_rate * time) - (burn_rate * time) = 3 := 
by
  intros initial_logs burn_rate add_rate time h1 h2 h3 h4
  sorry

end logs_left_after_3_hours_l8_8424


namespace range_of_k_l8_8598

theorem range_of_k (k : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) (h_f : ∀ x, f x = x^3 - 3 * x^2 - k)
  (h_f' : ∀ x, f' x = 3 * x^2 - 6 * x) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0) ↔ -4 < k ∧ k < 0 :=
sorry

end range_of_k_l8_8598


namespace integer_ratio_condition_l8_8210

variable {x y : ℝ}

theorem integer_ratio_condition (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 3) (h3 : ∃ t : ℤ, x = t * y) : 
  ∃ t : ℤ, t = -2 := sorry

end integer_ratio_condition_l8_8210


namespace series_sum_equals_one_fourth_l8_8444

noncomputable def series_term (n : ℕ) : ℝ :=
  3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))

noncomputable def infinite_series_sum : ℝ :=
  ∑' (n : ℕ), series_term (n + 1)

theorem series_sum_equals_one_fourth :
  infinite_series_sum = 1 / 4 :=
by
  -- Proof goes here.
  sorry

end series_sum_equals_one_fourth_l8_8444


namespace sum_of_first_10_terms_l8_8590

noncomputable def a (n : ℕ) : ℕ := sorry
noncomputable def S (n : ℕ) : ℕ := sorry

variable {n : ℕ}

-- Conditions
axiom h1 : ∀ n, S (n + 1) = S n + a n + 3
axiom h2 : a 5 + a 6 = 29

-- Statement to prove
theorem sum_of_first_10_terms : S 10 = 145 := 
sorry

end sum_of_first_10_terms_l8_8590


namespace right_triangle_lengths_l8_8309

theorem right_triangle_lengths (a b c : ℝ) (h1 : c + b = 2 * a) (h2 : c^2 = a^2 + b^2) : 
  b = 3 / 4 * a ∧ c = 5 / 4 * a := 
by
  sorry

end right_triangle_lengths_l8_8309


namespace initial_number_of_men_l8_8112

theorem initial_number_of_men (M : ℕ) (F : ℕ) (h1 : F = M * 20) (h2 : (M - 100) * 10 = M * 15) : 
  M = 200 :=
  sorry

end initial_number_of_men_l8_8112


namespace average_of_remaining_two_numbers_l8_8534

theorem average_of_remaining_two_numbers 
(A B C D E F G H : ℝ) 
(h_avg1 : (A + B + C + D + E + F + G + H) / 8 = 4.5) 
(h_avg2 : (A + B + C) / 3 = 5.2) 
(h_avg3 : (D + E + F) / 3 = 3.6) : 
  ((G + H) / 2 = 4.8) :=
sorry

end average_of_remaining_two_numbers_l8_8534


namespace ratio_of_saute_times_l8_8515

-- Definitions
def time_saute_onions : ℕ := 20
def time_saute_garlic_and_peppers : ℕ := 5
def time_knead_dough : ℕ := 30
def time_rest_dough : ℕ := 2 * time_knead_dough
def combined_knead_rest_time : ℕ := time_knead_dough + time_rest_dough
def time_assemble_calzones : ℕ := combined_knead_rest_time / 10
def total_time : ℕ := 124

-- Conditions
axiom saute_time_condition : time_saute_onions + time_saute_garlic_and_peppers + time_knead_dough + time_rest_dough + time_assemble_calzones = total_time

-- Question to be proved as a theorem
theorem ratio_of_saute_times :
  (time_saute_garlic_and_peppers : ℚ) / time_saute_onions = 1 / 4 :=
by
  -- proof goes here
  sorry

end ratio_of_saute_times_l8_8515


namespace correct_propositions_l8_8379

variable (A : Set ℝ)
variable (oplus : ℝ → ℝ → ℝ)

def condition_a1 : Prop := ∀ a b : ℝ, a ∈ A → b ∈ A → (oplus a b) ∈ A
def condition_a2 : Prop := ∀ a : ℝ, a ∈ A → (oplus a a) = 0
def condition_a3 : Prop := ∀ a b c : ℝ, a ∈ A → b ∈ A → c ∈ A → (oplus (oplus a b) c) = (oplus a c) + (oplus b c) + c

def proposition_1 : Prop := 0 ∈ A
def proposition_2 : Prop := (1 ∈ A) → (oplus (oplus 1 1) 1) = 0
def proposition_3 : Prop := ∀ a : ℝ, a ∈ A → (oplus a 0) = a → a = 0
def proposition_4 : Prop := ∀ a b c : ℝ, a ∈ A → b ∈ A → c ∈ A → (oplus a 0) = a → (oplus a b) = (oplus c b) → a = c

theorem correct_propositions 
  (h1 : condition_a1 A oplus) 
  (h2 : condition_a2 A oplus)
  (h3 : condition_a3 A oplus) : 
  (proposition_1 A) ∧ (¬proposition_2 A oplus) ∧ (proposition_3 A oplus) ∧ (proposition_4 A oplus) := by
  sorry

end correct_propositions_l8_8379


namespace find_a_if_f_even_l8_8352

noncomputable def f (x a : ℝ) : ℝ := (x + a) * Real.log (((2 * x) - 1) / ((2 * x) + 1))

theorem find_a_if_f_even (a : ℝ) :
  (∀ x : ℝ, (x > 1/2 ∨ x < -1/2) → f x a = f (-x) a) → a = 0 :=
by
  intro h1
  -- This is where the mathematical proof would go, but it's omitted as per the requirements.
  sorry

end find_a_if_f_even_l8_8352


namespace area_circle_l8_8846

-- Define the given condition
def polar_eq (r θ : ℝ) : Prop :=
  r = 3 * Real.cos θ - 4 * Real.sin θ

-- The goal is to prove the area of the circle described by the polar equation
theorem area_circle {r θ : ℝ} (h : polar_eq r θ) :
  ∃ A, A = π * (5 / 2) ^ 2 :=
sorry

end area_circle_l8_8846


namespace equation_solution_l8_8632

theorem equation_solution (a b c d : ℝ) :
  a^2 + b^2 + c^2 + d^2 - a * b - b * c - c * d - d + (2 / 5) = 0 ↔ 
  a = 1 / 5 ∧ b = 2 / 5 ∧ c = 3 / 5 ∧ d = 4 / 5 :=
by sorry

end equation_solution_l8_8632


namespace coin_toss_sequences_count_l8_8968

theorem coin_toss_sequences_count :
  (∃ (seq : List Char), 
    seq.length = 15 ∧ 
    (seq == ['H', 'H']) = 5 ∧ 
    (seq == ['H', 'T']) = 3 ∧ 
    (seq == ['T', 'H']) = 2 ∧ 
    (seq == ['T', 'T']) = 4) → 
  (count_sequences == 775360) :=
by
  sorry

end coin_toss_sequences_count_l8_8968


namespace num_ordered_pairs_c_d_l8_8929

def is_solution (c d x y : ℤ) : Prop :=
  c * x + d * y = 2 ∧ x^2 + y^2 = 65

theorem num_ordered_pairs_c_d : 
  ∃ (S : Finset (ℤ × ℤ)), S.card = 136 ∧ 
  ∀ (c d : ℤ), (c, d) ∈ S ↔ ∃ (x y : ℤ), is_solution c d x y :=
sorry

end num_ordered_pairs_c_d_l8_8929


namespace simplify_fraction_l8_8807

theorem simplify_fraction (x : ℝ) : (3*x + 2) / 4 + (x - 4) / 3 = (13*x - 10) / 12 := sorry

end simplify_fraction_l8_8807


namespace angle_relationship_l8_8611

theorem angle_relationship (u x y z w : ℝ)
    (H1 : ∀ (D E : ℝ), x + y + (360 - u - z) = 360)
    (H2 : ∀ (D E : ℝ), z + w + (360 - w - x) = 360) :
    x = (u + 2*z - y - w) / 2 := by
  sorry

end angle_relationship_l8_8611


namespace tree_height_l8_8281

theorem tree_height (boy_initial_height tree_initial_height boy_final_height boy_growth_rate tree_growth_rate : ℝ) 
  (h1 : boy_initial_height = 24) 
  (h2 : tree_initial_height = 16) 
  (h3 : boy_final_height = 36) 
  (h4 : boy_growth_rate = boy_final_height - boy_initial_height) 
  (h5 : tree_growth_rate = 2 * boy_growth_rate) 
  : tree_initial_height + tree_growth_rate = 40 := 
by
  subst h1 h2 h3 h4 h5;
  sorry

end tree_height_l8_8281


namespace avg_distance_is_600_l8_8786

-- Assuming Mickey runs half as many times as Johnny
def num_laps_johnny := 4
def num_laps_mickey := num_laps_johnny / 2
def lap_distance := 200

-- Calculating distances
def distance_johnny := num_laps_johnny * lap_distance
def distance_mickey := num_laps_mickey * lap_distance

-- Total distance run by both Mickey and Johnny
def total_distance := distance_johnny + distance_mickey

-- Average distance run by Johnny and Mickey
def avg_distance := total_distance / 2

-- Prove that the average distance run by Johnny and Mickey is 600 meters
theorem avg_distance_is_600 : avg_distance = 600 :=
by
  sorry

end avg_distance_is_600_l8_8786


namespace cody_games_remaining_l8_8709

-- Definitions based on the conditions
def initial_games : ℕ := 9
def games_given_away : ℕ := 4

-- Theorem statement
theorem cody_games_remaining : initial_games - games_given_away = 5 :=
by sorry

end cody_games_remaining_l8_8709


namespace students_not_taking_french_or_spanish_l8_8128

theorem students_not_taking_french_or_spanish 
  (total_students : ℕ)
  (french_students : ℕ)
  (spanish_students : ℕ)
  (both_languages_students : ℕ) 
  (h_total_students : total_students = 28)
  (h_french_students : french_students = 5)
  (h_spanish_students : spanish_students = 10)
  (h_both_languages_students : both_languages_students = 4) :
  total_students - (french_students + spanish_students - both_languages_students) = 17 := 
by {
  -- Correct answer can be verified with the given conditions
  -- The proof itself is omitted (as instructed)
  sorry
}

end students_not_taking_french_or_spanish_l8_8128


namespace tim_total_spent_l8_8127

-- Define the given conditions
def lunch_cost : ℝ := 50.20
def tip_percentage : ℝ := 0.20

-- Define the total amount spent
def total_amount_spent : ℝ := 60.24

-- Prove the total amount spent given the conditions
theorem tim_total_spent : lunch_cost + (tip_percentage * lunch_cost) = total_amount_spent := by
  -- This is the proof statement corresponding to the problem; the proof itself is not required for this task
  sorry

end tim_total_spent_l8_8127


namespace smallest_value_3a_plus_1_l8_8601

theorem smallest_value_3a_plus_1 
  (a : ℝ)
  (h : 8 * a^2 + 9 * a + 6 = 2) : 
  ∃ (b : ℝ), b = 3 * a + 1 ∧ b = -2 :=
by 
  sorry

end smallest_value_3a_plus_1_l8_8601


namespace triangle_solution_l8_8633

noncomputable def solve_triangle (a : ℝ) (α : ℝ) (t : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let s := 75
  let b := 41
  let c := 58
  let β := 43 + 36 / 60 + 10 / 3600
  let γ := 77 + 19 / 60 + 11 / 3600
  ((b, c), (β, γ))

theorem triangle_solution :
  solve_triangle 51 (59 + 4 / 60 + 39 / 3600) 1020 = ((41, 58), (43 + 36 / 60 + 10 / 3600, 77 + 19 / 60 + 11 / 3600)) :=
sorry  

end triangle_solution_l8_8633


namespace line_does_not_pass_through_second_quadrant_l8_8951

theorem line_does_not_pass_through_second_quadrant (a : ℝ) (h : a ≠ 0) :
  ¬ ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ x - y - a^2 = 0 := 
by
  sorry

end line_does_not_pass_through_second_quadrant_l8_8951


namespace percent_between_20000_and_150000_l8_8141

-- Define the percentages for each group of counties
def less_than_20000 := 30
def between_20000_and_150000 := 45
def more_than_150000 := 25

-- State the theorem using the above definitions
theorem percent_between_20000_and_150000 :
  between_20000_and_150000 = 45 :=
sorry -- Proof placeholder

end percent_between_20000_and_150000_l8_8141


namespace smallest_angle_of_triangle_l8_8855

theorem smallest_angle_of_triangle (y : ℝ) (h : 40 + 70 + y = 180) : 
  ∃ smallest_angle : ℝ, smallest_angle = 40 ∧ smallest_angle = min 40 (min 70 y) := 
by
  use 40
  sorry

end smallest_angle_of_triangle_l8_8855


namespace geom_series_sum_l8_8157

/-- The sum of the first six terms of the geometric series 
    with first term a = 1 and common ratio r = (1 / 4) is 1365 / 1024. -/
theorem geom_series_sum : 
  let a : ℚ := 1
  let r : ℚ := 1 / 4
  let n : ℕ := 6
  (a * (1 - r^n) / (1 - r)) = 1365 / 1024 :=
by
  sorry

end geom_series_sum_l8_8157


namespace tangent_line_at_zero_zero_intervals_l8_8477

-- Define the function f(x) with a parameter a
definition f (a : ℝ) (x : ℝ) : ℝ := Real.ln (1 + x) + a * x * Real.exp (-x)

-- Proof Problem 1: Equation of the tangent line
theorem tangent_line_at_zero (a : ℝ) (x : ℝ) (h_a : a = 1) : 
  let f := f a in
  -- The function with a = 1
  f x = Real.ln (1 + x) + x * Real.exp (-x) →
  -- The tangent line at (0, f(0)) is y = 2x
  ∃ (m : ℝ), m = 2 := sorry

-- Proof Problem 2: Range of values for a
theorem zero_intervals (a : ℝ) :
  -- Condition for f(x) having exactly one zero in each interval (-1,0) and (0, +∞)
  (∃! (x₁ : ℝ), x₁ ∈ (-1,0) ∧ f a x₁ = 0) ∧ (∃! (x₂ : ℝ), x₂ ∈ (0,+∞) ∧ f a x₂ = 0) →
  -- The range of values for a is (-∞, -1)
  a < -1 := sorry

end tangent_line_at_zero_zero_intervals_l8_8477


namespace problem_equivalence_l8_8933

theorem problem_equivalence (n : ℕ) (H₁ : 2 * 2006 = 1) (H₂ : ∀ n : ℕ, (2 * n + 2) * 2006 = 3 * (2 * n * 2006)) :
  2008 * 2006 = 3 ^ 1003 :=
by
  sorry

end problem_equivalence_l8_8933


namespace solve_problem_l8_8859

theorem solve_problem (nabla odot : ℕ) 
  (h1 : 0 < nabla) 
  (h2 : nabla < 20) 
  (h3 : 0 < odot) 
  (h4 : odot < 20) 
  (h5 : nabla ≠ odot) 
  (h6 : nabla * nabla * nabla = nabla) : 
  nabla * nabla = 64 :=
by
  sorry

end solve_problem_l8_8859


namespace scarlet_savings_l8_8390

theorem scarlet_savings : 
  let initial_savings := 80
  let cost_earrings := 23
  let cost_necklace := 48
  let total_spent := cost_earrings + cost_necklace
  initial_savings - total_spent = 9 := 
by 
  sorry

end scarlet_savings_l8_8390


namespace distance_is_twenty_cm_l8_8957

noncomputable def distance_between_pictures_and_board (picture_width: ℕ) (board_width_m: ℕ) (board_width_cm: ℕ) (number_of_pictures: ℕ) : ℕ :=
  let board_total_width := board_width_m * 100 + board_width_cm
  let total_pictures_width := number_of_pictures * picture_width
  let total_distance := board_total_width - total_pictures_width
  let total_gaps := number_of_pictures + 1
  total_distance / total_gaps

theorem distance_is_twenty_cm :
  distance_between_pictures_and_board 30 3 20 6 = 20 :=
by
  sorry

end distance_is_twenty_cm_l8_8957


namespace find_p_l8_8747

theorem find_p (p q : ℚ) (h1 : 3 * p + 4 * q = 15) (h2 : 4 * p + 3 * q = 18) : p = 27 / 7 :=
by
  sorry

end find_p_l8_8747


namespace water_to_add_l8_8688

theorem water_to_add (x : ℚ) (alcohol water : ℚ) (ratio : ℚ) :
  alcohol = 4 → water = 4 →
  (3 : ℚ) / (3 + 5) = (3 : ℚ) / 8 →
  (5 : ℚ) / (3 + 5) = (5 : ℚ) / 8 →
  ratio = 5 / 8 →
  (4 + x) / (8 + x) = ratio →
  x = 8 / 3 :=
by
  intros
  sorry

end water_to_add_l8_8688


namespace sqrt_0_1681_eq_0_41_l8_8594

theorem sqrt_0_1681_eq_0_41 (h : Real.sqrt 16.81 = 4.1) : Real.sqrt 0.1681 = 0.41 := by 
  sorry

end sqrt_0_1681_eq_0_41_l8_8594


namespace minimum_function_value_l8_8026

theorem minimum_function_value :
  ∃ (x y : ℕ), 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 3 ∧
  (∀ x' y', 0 ≤ x' ∧ x' ≤ 2 → 0 ≤ y' ∧ y' ≤ 3 →
  (x^2 * y^2 : ℝ) / ((x^2 + y^2)^2 : ℝ) ≤ (x'^2 * y'^2 : ℝ) / ((x'^2 + y'^2)^2 : ℝ)) ∧
  (x = 0 ∨ y = 0) ∧ ((x^2 * y^2 : ℝ) / ((x^2 + y^2)^2 : ℝ) = 0) :=
by
  --; Implementation of the theorem would follow
  sorry

end minimum_function_value_l8_8026


namespace smallest_positive_angle_l8_8650

theorem smallest_positive_angle (k : ℤ) : ∃ α, α = 400 + k * 360 ∧ α > 0 ∧ α = 40 :=
by
  use 40
  sorry

end smallest_positive_angle_l8_8650


namespace find_larger_number_l8_8652

variable (x y : ℕ)

theorem find_larger_number (h1 : x = 7) (h2 : x + y = 15) : y = 8 := by
  sorry

end find_larger_number_l8_8652


namespace value_of_x_l8_8772

theorem value_of_x {x y z w v : ℝ} 
  (h1 : y * x = 3)
  (h2 : z = 3)
  (h3 : w = z * y)
  (h4 : v = w * z)
  (h5 : v = 18)
  (h6 : w = 6) :
  x = 3 / 2 :=
by
  sorry

end value_of_x_l8_8772


namespace compare_neg_fractions_l8_8299

theorem compare_neg_fractions : - (1 : ℝ) / 3 < - (1 : ℝ) / 4 :=
  sorry

end compare_neg_fractions_l8_8299


namespace equivalent_systems_solution_and_value_l8_8195

-- Definitions for the conditions
def system1 (x y a b : ℝ) : Prop := 
  (2 * (x + 1) - y = 7) ∧ (x + b * y = a)

def system2 (x y a b : ℝ) : Prop := 
  (a * x + y = b) ∧ (3 * x + 2 * (y - 1) = 9)

-- The proof problem as a Lean 4 statement
theorem equivalent_systems_solution_and_value (a b : ℝ) :
  (∃ x y : ℝ, system1 x y a b ∧ system2 x y a b) →
  ((∃ x y : ℝ, x = 3 ∧ y = 1) ∧ (3 * a - b) ^ 2023 = -1) :=
  by sorry

end equivalent_systems_solution_and_value_l8_8195


namespace mosel_fills_315_boxes_per_week_l8_8005

-- Definitions for the conditions given in the problem.
def hens : ℕ := 270
def eggs_per_hen_per_day : ℕ := 1
def boxes_capacity : ℕ := 6
def days_per_week : ℕ := 7

-- Objective: Prove that the number of boxes filled each week is 315
theorem mosel_fills_315_boxes_per_week :
  let eggs_per_day := hens * eggs_per_hen_per_day
  let boxes_per_day := eggs_per_day / boxes_capacity
  let boxes_per_week := boxes_per_day * days_per_week
  boxes_per_week = 315 := by
  sorry

end mosel_fills_315_boxes_per_week_l8_8005


namespace condition_iff_odd_function_l8_8334

theorem condition_iff_odd_function (f : ℝ → ℝ) :
  (∀ x, f x + f (-x) = 0) ↔ (∀ x, f (-x) = -f x) :=
by
  sorry

end condition_iff_odd_function_l8_8334


namespace probability_right_triangle_in_3x3_grid_l8_8454

theorem probability_right_triangle_in_3x3_grid : 
  let vertices := (3 + 1) * (3 + 1)
  let total_combinations := Nat.choose vertices 3
  let right_triangles_on_gridlines := 144
  let right_triangles_off_gridlines := 24 + 32
  let total_right_triangles := right_triangles_on_gridlines + right_triangles_off_gridlines
  (total_right_triangles : ℚ) / total_combinations = 5 / 14 :=
by 
  sorry

end probability_right_triangle_in_3x3_grid_l8_8454


namespace original_mixture_litres_l8_8908

theorem original_mixture_litres 
  (x : ℝ)
  (h1 : 0.20 * x = 0.15 * (x + 5)) :
  x = 15 :=
sorry

end original_mixture_litres_l8_8908


namespace ratio_of_areas_of_two_concentric_circles_l8_8405

theorem ratio_of_areas_of_two_concentric_circles
  (C₁ C₂ : ℝ)
  (h1 : ∀ θ₁ θ₂, θ₁ = 30 ∧ θ₂ = 24 →
      (θ₁ / 360) * C₁ = (θ₂ / 360) * C₂):
  (C₁ / C₂) ^ 2 = (16 / 25) := by
  sorry

end ratio_of_areas_of_two_concentric_circles_l8_8405


namespace fraction_exponentiation_multiplication_l8_8306

theorem fraction_exponentiation_multiplication :
  (1 / 3) ^ 4 * (1 / 8) = 1 / 648 :=
by
  sorry

end fraction_exponentiation_multiplication_l8_8306


namespace circle_touch_externally_circle_one_inside_other_without_touching_circle_completely_outside_l8_8413

-- Definitions encapsulated in theorems with conditions and desired results
theorem circle_touch_externally {d R r : ℝ} (h1 : d = 10) (h2 : R = 8) (h3 : r = 2) : 
  d = R + r :=
by 
  rw [h1, h2, h3]
  sorry

theorem circle_one_inside_other_without_touching {d R r : ℝ} (h1 : d = 4) (h2 : R = 17) (h3 : r = 11) : 
  d < R - r :=
by 
  rw [h1, h2, h3]
  sorry

theorem circle_completely_outside {d R r : ℝ} (h1 : d = 12) (h2 : R = 5) (h3 : r = 3) : 
  d > R + r :=
by 
  rw [h1, h2, h3]
  sorry

end circle_touch_externally_circle_one_inside_other_without_touching_circle_completely_outside_l8_8413


namespace partial_fraction_decomposition_l8_8833

theorem partial_fraction_decomposition :
  ∃ A B : ℚ, (A = 43 / 14) ∧ (B = -31 / 14) ∧
  (3 * A + B = 7) ∧ (-2 * A + 4 * B = -15) :=
by
  use 43 / 14, -31 / 14
  split
  { sorry },
  split
  { sorry },
  split
  { sorry },
  { sorry }

end partial_fraction_decomposition_l8_8833


namespace problem_a_problem_b_l8_8981
-- Import the entire math library to ensure all necessary functionality is included

-- Define the problem context
variables {x y z : ℝ}

-- State the conditions as definitions
def conditions (x y z : ℝ) : Prop :=
  (x ≤ y) ∧ (y ≤ z) ∧ (x + y + z = 12) ∧ (x^2 + y^2 + z^2 = 54)

-- State the formal proof problems
theorem problem_a (h : conditions x y z) : x ≤ 3 ∧ 5 ≤ z :=
sorry

theorem problem_b (h : conditions x y z) : 
  9 ≤ x * y ∧ x * y ≤ 25 ∧
  9 ≤ y * z ∧ y * z ≤ 25 ∧
  9 ≤ z * x ∧ z * x ≤ 25 :=
sorry

end problem_a_problem_b_l8_8981


namespace no_unique_p_l8_8631

-- Define the probabilities P_1 and P_2 given p
def P1 (p : ℝ) : ℝ := 3 * p^2 - 2 * p^3
def P2 (p : ℝ) : ℝ := 3 * p^2 - 3 * p^3

-- Define the expected value E(xi)
def E_xi (p : ℝ) : ℝ := P1 p + P2 p

-- Prove that there does not exist a unique p in (0, 1) such that E(xi) = 1.5
theorem no_unique_p (p : ℝ) (h : 0 < p ∧ p < 1) : E_xi p ≠ 1.5 := by
  sorry

end no_unique_p_l8_8631


namespace total_frogs_in_both_ponds_l8_8864

noncomputable def total_frogs_combined : Nat :=
let frogs_in_pond_a : Nat := 32
let frogs_in_pond_b : Nat := frogs_in_pond_a / 2
frogs_in_pond_a + frogs_in_pond_b

theorem total_frogs_in_both_ponds :
  total_frogs_combined = 48 := by
  sorry

end total_frogs_in_both_ponds_l8_8864


namespace primes_less_or_equal_F_l8_8621

-- Definition of F_n
def F (n : ℕ) : ℕ := 2 ^ (2 ^ n) + 1

-- The main theorem statement
theorem primes_less_or_equal_F (n : ℕ) : ∃ S : Finset ℕ, S.card ≥ n + 1 ∧ ∀ p ∈ S, Nat.Prime p ∧ p ≤ F n := 
sorry

end primes_less_or_equal_F_l8_8621


namespace scarlet_savings_l8_8389

theorem scarlet_savings : 
  let initial_savings := 80
  let cost_earrings := 23
  let cost_necklace := 48
  let total_spent := cost_earrings + cost_necklace
  initial_savings - total_spent = 9 := 
by 
  sorry

end scarlet_savings_l8_8389


namespace closest_point_on_line_l8_8027

structure Point (α : Type) :=
(x : α) (y : α) (z : α)

def line (s : ℚ) : Point ℚ :=
⟨3 + s, 2 - 3 * s, 4 * s⟩

def distance (p1 p2 : Point ℚ) : ℚ :=
(p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

def closestPoint : Point ℚ := ⟨37/17, 74/17, -56/17⟩

def givenPoint : Point ℚ := ⟨1, 4, -2⟩

theorem closest_point_on_line :
  ∃ s : ℚ, line s = closestPoint ∧ 
           ∀ t : ℚ, distance closestPoint givenPoint ≤ distance (line t) givenPoint :=
by
  sorry

end closest_point_on_line_l8_8027


namespace max_cards_from_poster_board_l8_8442

theorem max_cards_from_poster_board (card_length card_width poster_length : ℕ) (h1 : card_length = 2) (h2 : card_width = 3) (h3 : poster_length = 12) : 
  (poster_length / card_length) * (poster_length / card_width) = 24 :=
by
  sorry

end max_cards_from_poster_board_l8_8442


namespace number_of_chairs_borrowed_l8_8965

-- Define the conditions
def red_chairs := 4
def yellow_chairs := 2 * red_chairs
def blue_chairs := yellow_chairs - 2
def total_initial_chairs : Nat := red_chairs + yellow_chairs + blue_chairs
def chairs_left_in_the_afternoon := 15

-- Define the question
def chairs_borrowed_by_Lisa : Nat := total_initial_chairs - chairs_left_in_the_afternoon

-- The theorem to state the proof problem
theorem number_of_chairs_borrowed : chairs_borrowed_by_Lisa = 3 := by
  -- Proof to be added
  sorry

end number_of_chairs_borrowed_l8_8965


namespace factorial_division_l8_8668

theorem factorial_division : (nat.factorial 15) / ((nat.factorial 6) * (nat.factorial 9)) = 5005 := 
by 
    sorry

end factorial_division_l8_8668


namespace three_digit_number_441_or_882_l8_8055

def is_valid_number (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  n = 100 * a + 10 * b + c ∧
  n / (100 * c + 10 * b + a) = 3 ∧
  n % (100 * c + 10 * b + a) = a + b + c

theorem three_digit_number_441_or_882:
  ∀ n : ℕ, is_valid_number n → (n = 441 ∨ n = 882) :=
by
  sorry

end three_digit_number_441_or_882_l8_8055


namespace largest_integer_chosen_l8_8154

-- Define the sequence of operations and establish the resulting constraints
def transformed_value (x : ℤ) : ℤ :=
  2 * (4 * x - 30) - 10

theorem largest_integer_chosen : 
  ∃ (x : ℤ), (10 : ℤ) ≤ transformed_value x ∧ transformed_value x ≤ (99 : ℤ) ∧ x = 21 :=
by
  sorry

end largest_integer_chosen_l8_8154


namespace basic_spatial_data_source_l8_8647

def source_of_basic_spatial_data (s : String) : Prop :=
  s = "Detailed data provided by high-resolution satellite remote sensing technology" ∨
  s = "Data from various databases provided by high-speed networks" ∨
  s = "Various data collected and organized through the information highway" ∨
  s = "Various spatial exchange data provided by GIS"

theorem basic_spatial_data_source :
  source_of_basic_spatial_data "Data from various databases provided by high-speed networks" :=
sorry

end basic_spatial_data_source_l8_8647


namespace probability_of_all_co_captains_l8_8863

def team_sizes : List ℕ := [6, 8, 9, 10]

def captains_per_team : ℕ := 3

noncomputable def probability_all_co_captains (s : ℕ) : ℚ :=
  1 / (Nat.choose s 3 : ℚ)

noncomputable def total_probability : ℚ :=
  (1 / 4 : ℚ) * 
  (probability_all_co_captains 6 + 
   probability_all_co_captains 8 +
   probability_all_co_captains 9 +
   probability_all_co_captains 10)

theorem probability_of_all_co_captains : total_probability = 1 / 84 :=
  sorry

end probability_of_all_co_captains_l8_8863


namespace chord_slope_of_ellipse_bisected_by_point_A_l8_8360

theorem chord_slope_of_ellipse_bisected_by_point_A :
  ∀ (P Q : ℝ × ℝ),
  (P.1^2 / 36 + P.2^2 / 9 = 1) ∧ (Q.1^2 / 36 + Q.2^2 / 9 = 1) ∧ 
  ((P.1 + Q.1) / 2 = 1) ∧ ((P.2 + Q.2) / 2 = 1) →
  (Q.2 - P.2) / (Q.1 - P.1) = -1 / 4 :=
by
  intros
  sorry

end chord_slope_of_ellipse_bisected_by_point_A_l8_8360


namespace min_value_is_1_5_l8_8623

noncomputable def min_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) : ℝ :=
  (1 : ℝ) / (a + b) + 
  (1 : ℝ) / (b + c) + 
  (1 : ℝ) / (c + a)

theorem min_value_is_1_5 {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) :
  min_value a b c h1 h2 h3 h4 = 1.5 :=
sorry

end min_value_is_1_5_l8_8623


namespace gardener_hourly_wage_l8_8295

-- Conditions
def rose_bushes_count : Nat := 20
def cost_per_rose_bush : Nat := 150
def hours_per_day : Nat := 5
def days_worked : Nat := 4
def soil_volume : Nat := 100
def cost_per_cubic_foot_soil : Nat := 5
def total_cost : Nat := 4100

-- Theorem statement
theorem gardener_hourly_wage :
  let cost_of_rose_bushes := rose_bushes_count * cost_per_rose_bush
  let cost_of_soil := soil_volume * cost_per_cubic_foot_soil
  let total_material_cost := cost_of_rose_bushes + cost_of_soil
  let labor_cost := total_cost - total_material_cost
  let total_hours_worked := hours_per_day * days_worked
  (labor_cost / total_hours_worked) = 30 := 
by {
  -- Proof placeholder
  sorry
}

end gardener_hourly_wage_l8_8295


namespace Dorottya_should_go_first_l8_8440

def probability_roll_1_or_2 : ℚ := 2 / 10

def probability_no_roll_1_or_2 : ℚ := 1 - probability_roll_1_or_2

variables {P_1 P_2 P_3 P_4 P_5 P_6 : ℚ}
  (hP1 : P_1 = probability_roll_1_or_2 * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP2 : P_2 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 1) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP3 : P_3 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 2) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP4 : P_4 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 3) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP5 : P_5 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 4) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP6 : P_6 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 5) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))

theorem Dorottya_should_go_first : P_1 > P_2 ∧ P_2 > P_3 ∧ P_3 > P_4 ∧ P_4 > P_5 ∧ P_5 > P_6 :=
by {
  -- Skipping actual proof steps
  sorry
}

end Dorottya_should_go_first_l8_8440


namespace movie_theater_ticket_sales_l8_8917

theorem movie_theater_ticket_sales 
  (A C : ℤ) 
  (h1 : A + C = 900) 
  (h2 : 7 * A + 4 * C = 5100) : 
  A = 500 := 
sorry

end movie_theater_ticket_sales_l8_8917


namespace jinho_remaining_money_l8_8977

def jinho_initial_money : ℕ := 2500
def cost_per_eraser : ℕ := 120
def erasers_bought : ℕ := 5
def cost_per_pencil : ℕ := 350
def pencils_bought : ℕ := 3

theorem jinho_remaining_money :
  jinho_initial_money - (erasers_bought * cost_per_eraser + pencils_bought * cost_per_pencil) = 850 :=
by
  sorry

end jinho_remaining_money_l8_8977


namespace perfect_square_solutions_l8_8939

theorem perfect_square_solutions :
  {n : ℕ | ∃ m : ℕ, n^2 + 77 * n = m^2} = {4, 99, 175, 1444} :=
by
  sorry

end perfect_square_solutions_l8_8939


namespace find_length_of_rectangular_playground_l8_8126

def perimeter (L B : ℕ) : ℕ := 2 * (L + B)

theorem find_length_of_rectangular_playground (P B : ℕ) (hP : P = 1200) (hB : B = 500) : ∃ L, perimeter L B = P ∧ L = 100 :=
by
  sorry

end find_length_of_rectangular_playground_l8_8126


namespace tan_alpha_eq_one_l8_8335

open Real

theorem tan_alpha_eq_one (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (h_cos_sin_eq : cos (α + β) = sin (α - β)) : tan α = 1 :=
by
  sorry

end tan_alpha_eq_one_l8_8335


namespace container_marbles_proportional_l8_8914

theorem container_marbles_proportional (V1 V2 : ℕ) (M1 M2 : ℕ)
(h1 : V1 = 24) (h2 : M1 = 75) (h3 : V2 = 72) (h4 : V1 * M2 = V2 * M1) :
  M2 = 225 :=
by {
  -- Given conditions
  sorry
}

end container_marbles_proportional_l8_8914


namespace base8_base6_positive_integer_l8_8256

theorem base8_base6_positive_integer (C D N : ℕ)
  (base8: N = 8 * C + D)
  (base6: N = 6 * D + C)
  (valid_C_base8: C < 8)
  (valid_D_base6: D < 6)
  (valid_C_D: 7 * C = 5 * D)
: N = 43 := by
  sorry

end base8_base6_positive_integer_l8_8256


namespace average_distance_is_600_l8_8788

-- Definitions based on the conditions
def one_lap_distance : ℕ := 200
def johnny_lap_count : ℕ := 4
def mickey_lap_count : ℕ := johnny_lap_count / 2

-- Total distances run by Johnny and Mickey
def johnny_distance : ℕ := johnny_lap_count * one_lap_distance
def mickey_distance : ℕ := mickey_lap_count * one_lap_distance

-- Sum of distances
def total_distance : ℕ := johnny_distance + mickey_distance

-- Number of people
def number_of_people : ℕ := 2

-- Average distance calculation
def average_distance : ℕ := total_distance / number_of_people

-- The theorem to prove: the average distance run by Johnny and Mickey
theorem average_distance_is_600 : average_distance = 600 := by
  sorry

end average_distance_is_600_l8_8788


namespace number_of_A_not_B_or_C_l8_8970

open Finset

variable {α : Type} (U A B C : Finset α)

-- Variables definition
variables (hU : card U = 300)
variables (hA : card A = 80)
variables (hB : card B = 70)
variables (hC : card C = 60)
variables (hAB : card (A ∩ B) = 30)
variables (hAC : card (A ∩ C) = 25)
variables (hBC : card (B ∩ C) = 20)
variables (hABC : card (A ∩ B ∩ C) = 15)
variables (hNotABC : card (U \ (A ∪ B ∪ C)) = 65)

theorem number_of_A_not_B_or_C : card (A \ (B ∪ C)) = 40 :=
sorry

end number_of_A_not_B_or_C_l8_8970


namespace binary_to_decimal_correct_l8_8421

def binary_to_decimal : ℕ := 110011

theorem binary_to_decimal_correct : 
  binary_to_decimal = 51 := sorry

end binary_to_decimal_correct_l8_8421


namespace profit_percent_is_25_l8_8961

noncomputable def SP : ℝ := sorry
noncomputable def CP : ℝ := 0.80 * SP
noncomputable def Profit : ℝ := SP - CP
noncomputable def ProfitPercent : ℝ := (Profit / CP) * 100

theorem profit_percent_is_25 :
  ProfitPercent = 25 :=
by
  sorry

end profit_percent_is_25_l8_8961


namespace boxes_filled_per_week_l8_8006

theorem boxes_filled_per_week : 
  (let hens := 270
       eggs_per_hen_per_day := 1
       days_in_week := 7
       eggs_per_week := hens * eggs_per_hen_per_day * days_in_week
       boxes_capacity := 6
   in eggs_per_week / boxes_capacity) = 315 :=
by
  let hens := 270
  let eggs_per_hen_per_day := 1
  let days_in_week := 7
  let eggs_per_week := hens * eggs_per_hen_per_day * days_in_week
  let boxes_capacity := 6
  have eggs_per_week_calc: eggs_per_week = 1890 := by 
  { calc 
      eggs_per_week = hens * eggs_per_hen_per_day * days_in_week : rfl
      ... = 270 * 1 * 7 : rfl
      ... = 1890 : by norm_num },
  
  have boxes_filled_calc: eggs_per_week / boxes_capacity = 315 := by 
  { calc 
      eggs_per_week / boxes_capacity = 1890 / boxes_capacity : by rw eggs_per_week_calc
      ... = 1890 / 6 : rfl
      ... = 315 : by norm_num },
  
  exact boxes_filled_calc

end boxes_filled_per_week_l8_8006


namespace relationship_y1_y2_l8_8178

theorem relationship_y1_y2
  (x1 y1 x2 y2 : ℝ)
  (hA : y1 = 3 * x1 + 4)
  (hB : y2 = 3 * x2 + 4)
  (h : x1 < x2) :
  y1 < y2 :=
sorry

end relationship_y1_y2_l8_8178


namespace smallest_prime_divisor_of_sum_l8_8881

theorem smallest_prime_divisor_of_sum (h1 : ∃ k : ℕ, 3^19 = 2*k + 1)
                                      (h2 : ∃ l : ℕ, 11^13 = 2*l + 1) :
  Nat.minFac (3^19 + 11^13) = 2 := 
by
  sorry

end smallest_prime_divisor_of_sum_l8_8881


namespace find_a_n_l8_8953

variable (a : ℕ → ℝ)
axiom a_pos : ∀ n, a n > 0
axiom a1_eq : a 1 = 1
axiom rec_relation : ∀ n, a n * (n * a n - a (n + 1)) = (n + 1) * (a (n + 1)) ^ 2

theorem find_a_n : ∀ n, a n = 1 / n := by
  sorry

end find_a_n_l8_8953


namespace probability_of_point_A_in_fourth_quadrant_l8_8107

noncomputable def probability_of_fourth_quadrant : ℚ :=
  let cards := {0, -1, 2, -3}
  let total_outcomes := cards.to_finset.powerset.filter (λ s, s.card = 2).card
  let favorable_outcomes := cards.to_finset.powerset.filter (λ s, s.card = 2 ∧ s.contains 2 ∧ s.contains -1 ∨ s.contains -3).card
  favorable_outcomes / total_outcomes

theorem probability_of_point_A_in_fourth_quadrant :
  probability_of_fourth_quadrant = 1 / 6 :=
sorry

end probability_of_point_A_in_fourth_quadrant_l8_8107


namespace knitting_time_total_l8_8781

-- Define knitting times for each item
def hat_knitting_time : ℕ := 2
def scarf_knitting_time : ℕ := 3
def mitten_knitting_time : ℕ := 1
def sock_knitting_time : ℕ := 3 / 2
def sweater_knitting_time : ℕ := 6

-- Define the number of grandchildren
def grandchildren_count : ℕ := 3

-- Total knitting time calculation
theorem knitting_time_total : 
  hat_knitting_time + scarf_knitting_time + 2 * mitten_knitting_time + 2 * sock_knitting_time + sweater_knitting_time = 16 ∧ 
  (hat_knitting_time + scarf_knitting_time + 2 * mitten_knitting_time + 2 * sock_knitting_time + sweater_knitting_time) * grandchildren_count = 48 :=
by 
  sorry

end knitting_time_total_l8_8781


namespace probability_fourth_quadrant_is_one_sixth_l8_8110

def in_fourth_quadrant (x y : ℤ) : Prop :=
  x > 0 ∧ y < 0

def possible_coordinates : List (ℤ × ℤ) :=
  [(0, -1), (0, 2), (0, -3), (-1, 0), (-1, 2), (-1, -3), (2, 0), (2, -1), (2, -3), (-3, 0), (-3, -1), (-3, 2)]

noncomputable def probability_fourth_quadrant : ℚ :=
  (possible_coordinates.count (λ p => in_fourth_quadrant p.fst p.snd)).toNat / (possible_coordinates.length : ℚ)

theorem probability_fourth_quadrant_is_one_sixth :
  probability_fourth_quadrant = 1/6 :=
by
  sorry

end probability_fourth_quadrant_is_one_sixth_l8_8110


namespace total_visitors_l8_8431

theorem total_visitors (sat_visitors : ℕ) (sun_visitors_more : ℕ) (h1 : sat_visitors = 200) (h2 : sun_visitors_more = 40) : 
  let sun_visitors := sat_visitors + sun_visitors_more in
  let total_visitors := sat_visitors + sun_visitors in
  total_visitors = 440 :=
by 
  let sun_visitors := sat_visitors + sun_visitors_more;
  let total_visitors := sat_visitors + sun_visitors;
  have h3 : sun_visitors = 240, by {
    rw [h1, h2],
    exact rfl
  };
  have h4 : total_visitors = 440, by {
    rw [h1, h3],
    exact rfl
  };
  exact h4

end total_visitors_l8_8431


namespace pats_stick_covered_l8_8088

/-
Assumptions:
1. Pat's stick is 30 inches long.
2. Jane's stick is 22 inches long.
3. Jane’s stick is two feet (24 inches) shorter than Sarah’s stick.
4. The portion of Pat's stick not covered in dirt is half as long as Sarah’s stick.

Prove that the length of Pat's stick covered in dirt is 7 inches.
-/

theorem pats_stick_covered  (pat_stick_len : ℕ) (jane_stick_len : ℕ) (jane_sarah_diff : ℕ) (pat_not_covered_by_dirt : ℕ) :
  pat_stick_len = 30 → jane_stick_len = 22 → jane_sarah_diff = 24 → pat_not_covered_by_dirt * 2 = jane_stick_len + jane_sarah_diff → 
    (pat_stick_len - pat_not_covered_by_dirt) = 7 :=
by
  intros h1 h2 h3 h4
  sorry

end pats_stick_covered_l8_8088


namespace ralph_final_cost_l8_8805

theorem ralph_final_cost :
  ∀ (total items_with_issue: ℝ) (issue_discount overall_discount: ℝ),
    total = 54.00 → items_with_issue = 20.00 → issue_discount = 0.20 → overall_discount = 0.10 →
    (total - items_with_issue * issue_discount) * (1 - overall_discount) = 45.00 :=
by
  intros total items_with_issue issue_discount overall_discount
  intro h_total h_items h_issue_discount h_overall_discount
  rw [h_total, h_items, h_issue_discount, h_overall_discount]
  norm_num
  sorry

end ralph_final_cost_l8_8805


namespace triangle_inequality_l8_8436

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  3 * (a * b + a * c + b * c) ≤ (a + b + c) ^ 2 ∧ (a + b + c) ^ 2 < 4 * (a * b + a * c + b * c) :=
sorry

end triangle_inequality_l8_8436


namespace remaining_bottles_after_2_days_l8_8689

-- Definitions based on the conditions:
def initial_bottles : ℕ := 24
def fraction_first_day : ℚ := 1 / 3
def fraction_second_day : ℚ := 1 / 2

-- Theorem statement proving the remaining number of bottles after 2 days
theorem remaining_bottles_after_2_days : 
    (initial_bottles - initial_bottles * fraction_first_day) - 
    ((initial_bottles - initial_bottles * fraction_first_day) * fraction_second_day) = 8 := 
by 
    -- Skipping the proof
    sorry

end remaining_bottles_after_2_days_l8_8689


namespace sum_of_first_ten_superb_equals_1399_l8_8712

-- Define a superb number
def is_superb (n : ℕ) : Prop := 
  let divisors := (list.filter (λ x : ℕ, x ≠ n ∧ n % x = 0) (list.range (n+1)))
  list.prod divisors = n

-- Define the sum of the first ten superb numbers
def sum_first_ten_superb : ℕ :=
  (list.filter is_superb (list.range 10000)).take 10).sum

theorem sum_of_first_ten_superb_equals_1399 : sum_first_ten_superb = 1399 := 
  sorry

end sum_of_first_ten_superb_equals_1399_l8_8712


namespace min_value_y_l8_8173

theorem min_value_y (x : ℝ) (hx : x > 2) : 
  ∃ x, x > 2 ∧ (∀ y, y = (x^2 - 4*x + 8) / (x - 2) → y ≥ 4 ∧ y = 4 ↔ x = 4) :=
sorry

end min_value_y_l8_8173


namespace determine_truth_tellers_min_questions_to_determine_truth_tellers_l8_8774

variables (n k : ℕ)
variables (h_n_pos : 0 < n) (h_k_pos : 0 < k) (h_k_le_n : k ≤ n)

theorem determine_truth_tellers (h : k % 2 = 0) : 
  ∃ m : ℕ, m = n :=
  sorry

theorem min_questions_to_determine_truth_tellers :
  ∃ m : ℕ, m = n :=
  sorry

end determine_truth_tellers_min_questions_to_determine_truth_tellers_l8_8774


namespace union_A_B_l8_8329

noncomputable def A : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }
noncomputable def B : Set ℝ := { x | x^3 = x }

theorem union_A_B : A ∪ B = { -1, 0, 1, 2 } := by
  sorry

end union_A_B_l8_8329


namespace parallelogram_area_l8_8428

theorem parallelogram_area (b : ℝ) (h : ℝ) (A : ℝ) 
  (h_b : b = 7) (h_h : h = 2 * b) (h_A : A = b * h) : A = 98 :=
by {
  sorry
}

end parallelogram_area_l8_8428


namespace f_g_minus_g_f_l8_8207

-- Defining the functions f and g
def f (x : ℝ) : ℝ := x^2 + 3
def g (x : ℝ) : ℝ := 3 * x^2 + 5

-- Proving the given math problem
theorem f_g_minus_g_f :
  f (g 2) - g (f 2) = 140 := by
sorry

end f_g_minus_g_f_l8_8207


namespace grasshopper_flea_adjacency_l8_8387

-- We assume that grid cells are indexed by pairs of integers (i.e., positions in ℤ × ℤ)
-- Red cells and white cells are represented as sets of these positions
variable (red_cells : Set (ℤ × ℤ))
variable (white_cells : Set (ℤ × ℤ))

-- We define that the grasshopper can only jump between red cells
def grasshopper_jump (pos : ℤ × ℤ) (new_pos : ℤ × ℤ) : Prop :=
  pos ∈ red_cells ∧ new_pos ∈ red_cells ∧ (pos.1 = new_pos.1 ∨ pos.2 = new_pos.2)

-- We define that the flea can only jump between white cells
def flea_jump (pos : ℤ × ℤ) (new_pos : ℤ × ℤ) : Prop :=
  pos ∈ white_cells ∧ new_pos ∈ white_cells ∧ (pos.1 = new_pos.1 ∨ pos.2 = new_pos.2)

-- Main theorem to be proved
theorem grasshopper_flea_adjacency (g_start : ℤ × ℤ) (f_start : ℤ × ℤ) :
    g_start ∈ red_cells → f_start ∈ white_cells →
    ∃ g1 g2 g3 f1 f2 f3 : ℤ × ℤ,
    (
      grasshopper_jump red_cells g_start g1 ∧
      grasshopper_jump red_cells g1 g2 ∧
      grasshopper_jump red_cells g2 g3
    ) ∧ (
      flea_jump white_cells f_start f1 ∧
      flea_jump white_cells f1 f2 ∧
      flea_jump white_cells f2 f3
    ) ∧
    (abs (g3.1 - f3.1) + abs (g3.2 - f3.2) = 1) :=
  sorry

end grasshopper_flea_adjacency_l8_8387


namespace shirt_to_pants_ratio_l8_8028

noncomputable def cost_uniforms
  (pants_cost shirt_ratio socks_price total_spending : ℕ) : Prop :=
  ∃ (shirt_cost tie_cost : ℕ),
    shirt_cost = shirt_ratio * pants_cost ∧
    tie_cost = shirt_cost / 5 ∧
    5 * (pants_cost + shirt_cost + tie_cost + socks_price) = total_spending

theorem shirt_to_pants_ratio 
  (pants_cost socks_price total_spending : ℕ)
  (h1 : pants_cost = 20)
  (h2 : socks_price = 3)
  (h3 : total_spending = 355)
  (shirt_ratio : ℕ)
  (h4 : cost_uniforms pants_cost shirt_ratio socks_price total_spending) :
  shirt_ratio = 2 := by
  sorry

end shirt_to_pants_ratio_l8_8028


namespace sum_of_nonneg_reals_l8_8488

theorem sum_of_nonneg_reals (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x^2 + y^2 + z^2 = 52) (h5 : x * y + y * z + z * x = 24) :
  x + y + z = 10 :=
sorry

end sum_of_nonneg_reals_l8_8488


namespace smallest_possible_z_l8_8371

theorem smallest_possible_z (w x y z : ℕ) (k : ℕ) (h1 : w = x - 1) (h2 : y = x + 1) (h3 : z = x + 2)
  (h4 : w ≠ x ∧ x ≠ y ∧ y ≠ z ∧ w ≠ y ∧ w ≠ z ∧ x ≠ z) (h5 : k = 2) (h6 : w^3 + x^3 + y^3 = k * z^3) : z = 6 :=
by
  sorry

end smallest_possible_z_l8_8371


namespace factorial_division_l8_8669

theorem factorial_division : (nat.factorial 15) / ((nat.factorial 6) * (nat.factorial 9)) = 5005 := 
by 
    sorry

end factorial_division_l8_8669


namespace snow_fall_time_l8_8191

theorem snow_fall_time (rate_mm_per_minute : ℕ) (time_minute : ℕ) (time_hour : ℕ) (meter_to_cm : ℕ) (cm_to_mm : ℕ) : 
(rate_mm_per_minute = 1 / 6) → 
(time_minute = 60) → 
(meter_to_cm = 100) → 
(cm_to_mm = 10) →
let rate_mm_per_hour := rate_mm_per_minute * time_minute in
let rate_cm_per_hour := rate_mm_per_hour / cm_to_mm in
let time_required_for_1m := meter_to_cm / rate_cm_per_hour in
time_required_for_1m = 100 := 
by
  intros rate_cond time_cond meter_cond cm_cond
  rw [rate_cond, time_cond, meter_cond, cm_cond]
  simp [rate_mm_per_hour, rate_cm_per_hour, time_required_for_1m]
  sorry

end snow_fall_time_l8_8191


namespace general_term_of_sequence_l8_8343

theorem general_term_of_sequence (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ n, a (n + 1) = a n + n + 1) :
  ∀ n, a n = (n^2 + n + 2) / 2 :=
by 
  sorry

end general_term_of_sequence_l8_8343


namespace simplify_expression_l8_8393

variable (a : ℚ)
def expression := ((a + 3) / (a - 1) - 1 / (a - 1)) / ((a^2 + 4 * a + 4) / (a^2 - a))

theorem simplify_expression (h : a = 3) : expression a = 3 / 5 :=
by
  rw [h]
  -- additional simplifications would typically go here if the steps were spelled out
  sorry

end simplify_expression_l8_8393


namespace total_winning_team_points_l8_8293

/-!
# Lean 4 Math Proof Problem

Prove that the total points scored by the winning team at the end of the game is 50 points given the conditions provided.
-/

-- Definitions
def losing_team_points_first_quarter : ℕ := 10
def winning_team_points_first_quarter : ℕ := 2 * losing_team_points_first_quarter
def winning_team_points_second_quarter : ℕ := winning_team_points_first_quarter + 10
def winning_team_points_third_quarter : ℕ := winning_team_points_second_quarter + 20

-- Theorem statement
theorem total_winning_team_points : winning_team_points_third_quarter = 50 :=
by
  sorry

end total_winning_team_points_l8_8293


namespace expected_hit_targets_expected_hit_targets_not_less_than_half_l8_8722

-- Part (a): The expected number of hit targets
theorem expected_hit_targets (n : ℕ) (h : n ≠ 0) :
  E (number_of_hit_targets n) = n * (1 - (1 - (1 / n)) ^ n) :=
sorry

-- Part (b): The expected number of hit targets cannot be less than n / 2
theorem expected_hit_targets_not_less_than_half (n : ℕ) (h : n ≠ 0) :
  n * (1 - (1 - (1 / n)) ^ n) ≥ n / 2 :=
sorry

end expected_hit_targets_expected_hit_targets_not_less_than_half_l8_8722


namespace trapezoid_geometry_proof_l8_8646

theorem trapezoid_geometry_proof
  (midline_length : ℝ)
  (segment_midpoints : ℝ)
  (angle1 angle2 : ℝ)
  (h_midline : midline_length = 5)
  (h_segment_midpoints : segment_midpoints = 3)
  (h_angle1 : angle1 = 30)
  (h_angle2 : angle2 = 60) :
  ∃ (AD BC AB : ℝ), AD = 8 ∧ BC = 2 ∧ AB = 3 :=
by
  sorry

end trapezoid_geometry_proof_l8_8646


namespace sanity_indeterminable_transylvanian_is_upyr_l8_8272

noncomputable def transylvanianClaim := "I have lost my mind."

/-- Proving whether the sanity of the Transylvanian can be determined from the statement -/
theorem sanity_indeterminable (claim : String) : 
  claim = transylvanianClaim → 
  ¬ (∀ (sane : Prop), sane ∨ ¬ sane) := 
by 
  intro h
  rw [transylvanianClaim] at h
  sorry

/-- Proving the nature of whether the Transylvanian is an upyr or human from the statement -/
theorem transylvanian_is_upyr (claim : String) : 
  claim = transylvanianClaim → 
  ∀ (human upyr : Prop), ¬ human ∧ upyr := 
by 
  intro h
  rw [transylvanianClaim] at h
  sorry

end sanity_indeterminable_transylvanian_is_upyr_l8_8272


namespace cosine_of_five_pi_over_three_l8_8414

theorem cosine_of_five_pi_over_three :
  Real.cos (5 * Real.pi / 3) = 1 / 2 :=
sorry

end cosine_of_five_pi_over_three_l8_8414


namespace logs_left_after_3_hours_l8_8425

theorem logs_left_after_3_hours :
  ∀ (burn_rate init_logs added_logs_per_hour hours : ℕ),
    burn_rate = 3 →
    init_logs = 6 →
    added_logs_per_hour = 2 →
    hours = 3 →
    (init_logs + added_logs_per_hour * hours - burn_rate * hours) = 3 :=
by
  intros burn_rate init_logs added_logs_per_hour hours
  intros h_burn_rate h_init_logs h_added_logs_per_hour h_hours
  rw [h_burn_rate, h_init_logs, h_added_logs_per_hour, h_hours]
  simp
  sorry

end logs_left_after_3_hours_l8_8425


namespace glove_selection_l8_8170

theorem glove_selection (pairs_gloves : Finset (Finset (Fin 12))) :
  (pairs_gloves.card = 6) →
  (∀ g ∈ pairs_gloves, g.card = 2) →
  ∃ selected_gloves : Finset (Fin 12),
    (selected_gloves.card = 4) ∧
    (∃ p ∈ selected_gloves.powerset, p.card = 2 ∧ (p ∈ pairs_gloves)) ∧
    (sum (λ p, if p.card = 2 ∧ ∃ g ∈ pairs_gloves, p ∈ g.powerset then 1 else 0) (powerset selected_gloves) = 240) :=
begin
  intro h_pairs_card,
  intro h_each_pair,
  -- Further proof steps should be provided here
  sorry
end

end glove_selection_l8_8170


namespace choose_with_at_least_one_girl_l8_8327

theorem choose_with_at_least_one_girl :
  let boys := 4
  let girls := 2
  let total_students := boys + girls
  let ways_choose_4 := Nat.choose total_students 4
  let ways_all_boys := Nat.choose boys 4
  ways_choose_4 - ways_all_boys = 14 := by
  sorry

end choose_with_at_least_one_girl_l8_8327


namespace find_n_value_l8_8336

theorem find_n_value (n : ℕ) (h : ∃ k : ℤ, n^2 + 5 * n + 13 = k^2) : n = 4 :=
by
  sorry

end find_n_value_l8_8336


namespace fraction_of_new_releases_l8_8292

theorem fraction_of_new_releases (total_books : ℕ) (historical_fiction_percent : ℝ) (historical_new_releases_percent : ℝ) (other_new_releases_percent : ℝ)
  (h1 : total_books = 100)
  (h2 : historical_fiction_percent = 0.4)
  (h3 : historical_new_releases_percent = 0.4)
  (h4 : other_new_releases_percent = 0.2) :
  (historical_fiction_percent * historical_new_releases_percent * total_books) / 
  ((historical_fiction_percent * historical_new_releases_percent * total_books) + ((1 - historical_fiction_percent) * other_new_releases_percent * total_books)) = 4 / 7 :=
by
  have h_books : total_books = 100 := h1
  have h_fiction : historical_fiction_percent = 0.4 := h2
  have h_new_releases : historical_new_releases_percent = 0.4 := h3
  have h_other_new_releases : other_new_releases_percent = 0.2 := h4
  sorry

end fraction_of_new_releases_l8_8292


namespace necessary_but_not_sufficient_condition_l8_8377

variable {M N P : Set α}

theorem necessary_but_not_sufficient_condition (h : M ∩ P = N ∩ P) : 
  (M = N) → (M ∩ P = N ∩ P) :=
sorry

end necessary_but_not_sufficient_condition_l8_8377


namespace find_x_of_orthogonal_vectors_l8_8753

variable (x : ℝ)

def a : ℝ × ℝ × ℝ := (2, -3, 1)
def b : ℝ × ℝ × ℝ := (-4, 2, x)

theorem find_x_of_orthogonal_vectors (h : (2 * -4 + -3 * 2 + 1 * x) = 0) : x = 14 := by
  sorry

end find_x_of_orthogonal_vectors_l8_8753


namespace ganesh_average_speed_l8_8679

noncomputable def averageSpeed (D : ℝ) : ℝ :=
  let time_uphill := D / 60
  let time_downhill := D / 36
  let total_time := time_uphill + time_downhill
  let total_distance := 2 * D
  total_distance / total_time

theorem ganesh_average_speed (D : ℝ) (hD : D > 0) : averageSpeed D = 45 := by
  sorry

end ganesh_average_speed_l8_8679


namespace max_min_value_of_f_l8_8229

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem max_min_value_of_f :
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ f (Real.pi / 6)) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f (Real.pi / 2) ≤ f x) :=
by
  sorry

end max_min_value_of_f_l8_8229


namespace solution_set_for_inequality_l8_8046

def f (x : ℝ) : ℝ := x^3 + x

theorem solution_set_for_inequality {a : ℝ} (h : -2 < a ∧ a < 2) :
  f a + f (a^2 - 2) < 0 ↔ -2 < a ∧ a < 0 ∨ 0 < a ∧ a < 1 := sorry

end solution_set_for_inequality_l8_8046


namespace charlie_paints_140_square_feet_l8_8290

-- Define the conditions
def total_area : ℕ := 320
def ratio_allen : ℕ := 4
def ratio_ben : ℕ := 5
def ratio_charlie : ℕ := 7
def total_parts : ℕ := ratio_allen + ratio_ben + ratio_charlie
def area_per_part := total_area / total_parts
def charlie_parts := 7

-- Prove the main statement
theorem charlie_paints_140_square_feet : charlie_parts * area_per_part = 140 := by
  sorry

end charlie_paints_140_square_feet_l8_8290


namespace time_to_carl_is_28_minutes_l8_8066

variable (distance_to_julia : ℕ := 1) (time_to_julia : ℕ := 4)
variable (distance_to_carl : ℕ := 7)
variable (rate : ℕ := distance_to_julia * time_to_julia) -- Rate as product of distance and time

theorem time_to_carl_is_28_minutes : (distance_to_carl * time_to_julia) = 28 := by
  sorry

end time_to_carl_is_28_minutes_l8_8066


namespace marbles_percentage_l8_8684

def solid_color_other_than_yellow (total_marbles : ℕ) (solid_color_percent solid_yellow_percent : ℚ) : ℚ :=
  solid_color_percent - solid_yellow_percent

theorem marbles_percentage (total_marbles : ℕ) (solid_color_percent solid_yellow_percent : ℚ) :
  solid_color_percent = 90 / 100 →
  solid_yellow_percent = 5 / 100 →
  solid_color_other_than_yellow total_marbles solid_color_percent solid_yellow_percent = 85 / 100 :=
by
  intro h1 h2
  rw [h1, h2]
  norm_num
  sorry

end marbles_percentage_l8_8684


namespace compute_ns_l8_8982

noncomputable def f : ℝ → ℝ :=
sorry

-- Defining the functional equation as a condition
def functional_equation (f : ℝ → ℝ) :=
∀ x y z : ℝ, f (x^2 + y^2 * f z) = x * f x + z * f (y^2)

-- Proving that the number of possible values of f(5) is 2
-- and their sum is 5, thus n * s = 10
theorem compute_ns (f : ℝ → ℝ) (hf : functional_equation f) : 2 * 5 = 10 :=
sorry

end compute_ns_l8_8982


namespace total_time_correct_l8_8925

-- Definitions based on problem conditions
def first_time : ℕ := 15
def time_increment : ℕ := 7
def number_of_flights : ℕ := 7

-- Time taken for a specific flight
def time_for_nth_flight (n : ℕ) : ℕ := first_time + (n - 1) * time_increment

-- Sum of the times for the first seven flights
def total_time : ℕ := (number_of_flights * (first_time + time_for_nth_flight number_of_flights)) / 2

-- Statement to be proven
theorem total_time_correct : total_time = 252 := 
by
  sorry

end total_time_correct_l8_8925


namespace gas_and_maintenance_money_l8_8618

theorem gas_and_maintenance_money
  (income : ℝ := 3200)
  (rent : ℝ := 1250)
  (utilities : ℝ := 150)
  (retirement_savings : ℝ := 400)
  (groceries : ℝ := 300)
  (insurance : ℝ := 200)
  (miscellaneous_expenses : ℝ := 200)
  (car_payment : ℝ := 350) :
  income - (rent + utilities + retirement_savings + groceries + insurance + miscellaneous_expenses + car_payment) = 350 :=
by
  sorry

end gas_and_maintenance_money_l8_8618


namespace quadrilateral_parallelogram_iff_l8_8218

variable (a b c d e f MN : ℝ)

-- Define a quadrilateral as a structure with sides and diagonals 
structure Quadrilateral :=
  (a b c d e f : ℝ)

-- Define the condition: sum of squares of diagonals equals sum of squares of sides
def sum_of_squares_condition (q : Quadrilateral) : Prop :=
  q.e ^ 2 + q.f ^ 2 = q.a ^ 2 + q.b ^ 2 + q.c ^ 2 + q.d ^ 2

-- Define what it means for a quadrilateral to be a parallelogram:
-- Midpoints of the diagonals coincide (MN = 0)
def is_parallelogram (q : Quadrilateral) (MN : ℝ) : Prop :=
  MN = 0

-- Main theorem to prove
theorem quadrilateral_parallelogram_iff (q : Quadrilateral) (MN : ℝ) :
  is_parallelogram q MN ↔ sum_of_squares_condition q :=
sorry

end quadrilateral_parallelogram_iff_l8_8218


namespace smallest_prime_divisor_of_sum_l8_8882

theorem smallest_prime_divisor_of_sum (h1 : ∃ k : ℕ, 3^19 = 2*k + 1)
                                      (h2 : ∃ l : ℕ, 11^13 = 2*l + 1) :
  Nat.minFac (3^19 + 11^13) = 2 := 
by
  sorry

end smallest_prime_divisor_of_sum_l8_8882


namespace r_daily_earning_l8_8129

-- Definitions from conditions in the problem
def earnings_of_all (P Q R : ℕ) : Prop := 9 * (P + Q + R) = 1620
def earnings_p_and_r (P R : ℕ) : Prop := 5 * (P + R) = 600
def earnings_q_and_r (Q R : ℕ) : Prop := 7 * (Q + R) = 910

-- Theorem to prove the daily earnings of r
theorem r_daily_earning (P Q R : ℕ) 
    (h1 : earnings_of_all P Q R)
    (h2 : earnings_p_and_r P R)
    (h3 : earnings_q_and_r Q R) : 
    R = 70 := 
by 
  sorry

end r_daily_earning_l8_8129


namespace coloring_ways_l8_8926

-- Define the function that checks valid coloring
noncomputable def valid_coloring (colors : Fin 6 → Fin 3) : Prop :=
  colors 0 = 0 ∧ -- The central pentagon is colored red
  (colors 1 ≠ colors 0 ∧ colors 2 ≠ colors 1 ∧ 
   colors 3 ≠ colors 2 ∧ colors 4 ≠ colors 3 ∧ 
   colors 5 ≠ colors 4 ∧ colors 1 ≠ colors 5) -- No two adjacent polygons have the same color

-- Define the main theorem
theorem coloring_ways (f : Fin 6 → Fin 3) (h : valid_coloring f) : 
  ∃! (f : Fin 6 → Fin 3), valid_coloring f := by
  sorry

end coloring_ways_l8_8926


namespace intersection_of_M_and_N_l8_8511

def M : Set ℤ := {x : ℤ | -4 < x ∧ x < 2}
def N : Set ℤ := {x : ℤ | x^2 < 4}

theorem intersection_of_M_and_N : (M ∩ N) = { -1, 0, 1 } :=
by
  sorry

end intersection_of_M_and_N_l8_8511


namespace area_of_circle_l8_8853

theorem area_of_circle:
  (∃ (r : ℝ) (θ : ℝ), r = 3 * Real.cos θ - 4 * Real.sin θ) → ∃ area: ℝ, area = (25/4) * Real.pi :=
sorry

end area_of_circle_l8_8853


namespace tens_digit_of_19_pow_2023_l8_8580

theorem tens_digit_of_19_pow_2023 : (19 ^ 2023) % 100 = 59 := 
  sorry

end tens_digit_of_19_pow_2023_l8_8580


namespace part1_tangent_line_eqn_part2_range_of_a_l8_8472

-- Define the function f
def f (x a : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

-- Part (1): Proving the equation of the tangent line at a = 1 and x = 0
theorem part1_tangent_line_eqn :
  (∀ x, f x 1 = Real.log (1 + x) + x * Real.exp (-x)) → 
  (let f' (x : ℝ) := (1 / (1 + x)) + Real.exp (-x) - x * Real.exp (-x) in
    let tangent_line (x : ℝ) := 2 * x in
    tangent_line 0 = 0 ∧ (∀ x, tangent_line x = 2 * x)) :=
by
  sorry

-- Part (2): Finding the range of values for a
theorem part2_range_of_a :
  (∀ x, f x a = Real.log (1 + x) + a * x * Real.exp (-x)) →
  (∀ a, (∃ x ∈ set.Ioo (-1 : ℝ) 0, f x a = 0) ∧ (∃ x ∈ set.Ioi (0 : ℝ), f x a = 0) → a ∈ set.Iio (-1)) :=
by
  sorry

end part1_tangent_line_eqn_part2_range_of_a_l8_8472


namespace MMobile_cheaper_l8_8520

-- Define the given conditions
def TMobile_base_cost : ℕ := 50
def TMobile_additional_cost : ℕ := 16
def MMobile_base_cost : ℕ := 45
def MMobile_additional_cost : ℕ := 14
def additional_lines : ℕ := 3

-- Define functions to calculate total costs
def TMobile_total_cost : ℕ := TMobile_base_cost + TMobile_additional_cost * additional_lines
def MMobile_total_cost : ℕ := MMobile_base_cost + MMobile_additional_cost * additional_lines

-- Statement to be proved
theorem MMobile_cheaper : TMobile_total_cost - MMobile_total_cost = 11 := by
  sorry

end MMobile_cheaper_l8_8520


namespace smallest_prime_divisor_of_sum_l8_8889

theorem smallest_prime_divisor_of_sum (a b : ℕ) (h1 : a = 3^19) (h2 : b = 11^13) (h3 : Odd a) (h4 : Odd b) : 
  ∃ p : ℕ, Prime p ∧ p ∣ (a + b) ∧ p = 2 := 
by {
  sorry
}

end smallest_prime_divisor_of_sum_l8_8889


namespace normal_distribution_calculation_l8_8956

noncomputable def normal_probability : Prop :=
  let μ : ℝ := 5
  let δ : ℝ := 2
  let X := pdf.normal μ δ in
  𝔼[X] = 5 ∧ Var[X] = 4 ∧ Pr (3 < X ≤ 7) = 0.6826

theorem normal_distribution_calculation : normal_probability :=
by sorry

end normal_distribution_calculation_l8_8956


namespace base8_subtraction_l8_8023

theorem base8_subtraction : (7463 - 3154 = 4317) := by sorry

end base8_subtraction_l8_8023


namespace seqD_not_arithmetic_l8_8122

-- Definitions of the sequences
def seqA : List ℤ := [1, 1, 1, 1, 1]
def seqB : List ℤ := [4, 7, 10, 13, 16]
def seqC : List ℚ := [1/3, 2/3, 1, 4/3, 5/3]
def seqD : List ℤ := [-3, -2, -1, 1, 2]

-- Arithmetic sequence definition (for integer sequences)
def is_arithmetic_sequence (seq : List ℤ) : Prop :=
  ∀ (i : ℕ), i < seq.length - 1 → (seq.nth_le (i + 1) (by linarith) - seq.nth_le i (by linarith)) = (seq.nth_le 1 (by linarith) - seq.nth_le 0 (by linarith))

-- The proof problem statement: Prove that seqD is not an arithmetic sequence
theorem seqD_not_arithmetic : ¬ is_arithmetic_sequence seqD :=
by 
  sorry

end seqD_not_arithmetic_l8_8122


namespace min_total_trees_l8_8061

theorem min_total_trees (L X : ℕ) (h1: 13 * L < 100 * X) (h2: 100 * X < 14 * L) : L ≥ 15 :=
  sorry

end min_total_trees_l8_8061


namespace factorization_of_m_squared_minus_4_l8_8315

theorem factorization_of_m_squared_minus_4 (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) :=
by
  sorry

end factorization_of_m_squared_minus_4_l8_8315


namespace probability_of_fourth_quadrant_l8_8108

-- Define the four cards
def cards : List ℤ := [0, -1, 2, -3]

-- Define the fourth quadrant condition for point A(m, n)
def in_fourth_quadrant (m n : ℤ) : Prop := m > 0 ∧ n < 0

-- Calculate the probability of a point being in the fourth quadrant
theorem probability_of_fourth_quadrant :
  let points := (cards.product cards).filter (λ ⟨m, n⟩, m ≠ n)
  let favorable := points.filter (λ ⟨m, n⟩, in_fourth_quadrant m n)
  (favorable.length : ℚ) / (points.length : ℚ) = 1 / 6 := by
    sorry

end probability_of_fourth_quadrant_l8_8108


namespace freezer_temp_is_correct_l8_8076

def freezer_temp (temp: ℤ) := temp

theorem freezer_temp_is_correct (temp: ℤ)
  (freezer_below_zero: temp = -18): freezer_temp temp = -18 := 
by
  -- since freezer_below_zero state that temperature is -18
  exact freezer_below_zero

end freezer_temp_is_correct_l8_8076


namespace area_of_circle_l8_8854

theorem area_of_circle:
  (∃ (r : ℝ) (θ : ℝ), r = 3 * Real.cos θ - 4 * Real.sin θ) → ∃ area: ℝ, area = (25/4) * Real.pi :=
sorry

end area_of_circle_l8_8854


namespace min_max_value_l8_8022

-- Definition of the function to be minimized and maximized
def f (x y : ℝ) : ℝ := |x^3 - x * y^2|

-- Conditions
def x_condition (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def y_condition (y : ℝ) : Prop := true

-- Goal: Prove the minimum of the maximum value
theorem min_max_value :
  ∃ y : ℝ, (∀ x : ℝ, x_condition x → f x y ≤ 8) ∧ (∀ y' : ℝ, (∀ x : ℝ, x_condition x → f x y' ≤ 8) → y' = y) :=
sorry

end min_max_value_l8_8022


namespace simplify_and_evaluate_l8_8995

-- Definitions of given conditions
def a := 1
def b := 2

-- Statement of the theorem
theorem simplify_and_evaluate : (a * b + (a^2 - a * b) - (a^2 - 2 * a * b) = 4) :=
by
  -- Using sorry to indicate the proof is to be completed
  sorry

end simplify_and_evaluate_l8_8995


namespace bus_ride_difference_l8_8628

def oscars_bus_ride : ℝ := 0.75
def charlies_bus_ride : ℝ := 0.25

theorem bus_ride_difference :
  oscars_bus_ride - charlies_bus_ride = 0.50 :=
by
  sorry

end bus_ride_difference_l8_8628


namespace car_travel_distance_l8_8757

theorem car_travel_distance :
  ∀ (train_speed : ℝ) (fraction : ℝ) (time_minutes : ℝ) (car_speed : ℝ) (distance : ℝ),
  train_speed = 90 →
  fraction = 5 / 6 →
  time_minutes = 30 →
  car_speed = fraction * train_speed →
  distance = car_speed * (time_minutes / 60) →
  distance = 37.5 :=
by
  intros train_speed fraction time_minutes car_speed distance
  intros h_train_speed h_fraction h_time_minutes h_car_speed h_distance
  sorry

end car_travel_distance_l8_8757


namespace mr_brown_at_least_five_sons_l8_8382

open ProbabilityTheory

noncomputable def at_least_five_sons_probability (n : ℕ) (p_son : ℝ) (p_daughter : ℝ) :=
  ∑ k in finset.range 4, (nat.choose 8 (5 + k)) * (p_son ^ (5 + k)) * (p_daughter ^ (8 - (5 + k)))

theorem mr_brown_at_least_five_sons : 
  at_least_five_sons_probability 8 0.6 0.4 = 0.594 :=
by
  sorry

end mr_brown_at_least_five_sons_l8_8382


namespace largest_multiple_of_7_less_than_100_l8_8408

theorem largest_multiple_of_7_less_than_100 : ∃ n : ℕ, 7 * n < 100 ∧ ∀ m : ℕ, 7 * m < 100 → 7 * m ≤ 7 * n := by
  sorry

end largest_multiple_of_7_less_than_100_l8_8408


namespace tech_gadgets_components_total_l8_8530

theorem tech_gadgets_components_total (a₁ r n : ℕ) (h₁ : a₁ = 8) (h₂ : r = 3) (h₃ : n = 4) :
  a₁ * (r^n - 1) / (r - 1) = 320 := by
  sorry

end tech_gadgets_components_total_l8_8530


namespace squareInPentagon_l8_8092

-- Definitions pertinent to the problem
structure Pentagon (α : Type) [AddCommGroup α] :=
(A B C D E : α) 

def isRegularPentagon {α : Type} [AddCommGroup α] [LinearOrder α] (P : Pentagon α) : Prop :=
  -- Conditions for a regular pentagon (typically involving equal side lengths and equal angles)
  sorry

def inscribedSquareExists {α : Type} [AddCommGroup α] (P : Pentagon α) : Prop :=
  -- There exists a square inscribed in the pentagon P with vertices on four different sides
  sorry

-- The main theorem to state the proof problem
theorem squareInPentagon {α : Type} [AddCommGroup α] [LinearOrder α] (P : Pentagon α)
  (hP : isRegularPentagon P) : inscribedSquareExists P :=
sorry

end squareInPentagon_l8_8092


namespace length_of_second_platform_is_correct_l8_8919

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

end length_of_second_platform_is_correct_l8_8919


namespace inequality_proof_l8_8388

-- Define the main theorem to be proven.
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (b + c - a) + b^2 * (a + c - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end inequality_proof_l8_8388


namespace algorithm_contains_sequential_structure_l8_8554

theorem algorithm_contains_sequential_structure :
  (∀ algorithm : Type, ∃ seq_struct : Prop, seq_struct) ∧
  (∀ algorithm : Type, ∃ sel_struct : Prop, sel_struct ∨ ¬ sel_struct) ∧
  (∀ algorithm : Type, ∃ loop_struct : Prop, loop_struct) →
  (∀ algorithm : Type, ∃ seq_struct : Prop, seq_struct) := by
  sorry

end algorithm_contains_sequential_structure_l8_8554


namespace cubics_of_sum_and_product_l8_8763

theorem cubics_of_sum_and_product (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) : 
  x^3 + y^3 = 640 :=
by
  sorry

end cubics_of_sum_and_product_l8_8763


namespace domain_of_tan_arcsin_xsq_l8_8713

noncomputable def domain_f (x : ℝ) : Prop :=
  x ≠ 1 ∧ x ≠ -1 ∧ -1 ≤ x ∧ x ≤ 1

theorem domain_of_tan_arcsin_xsq :
  ∀ x : ℝ, -1 < x ∧ x < 1 ↔ domain_f x := 
sorry

end domain_of_tan_arcsin_xsq_l8_8713


namespace factorization_of_m_squared_minus_4_l8_8314

theorem factorization_of_m_squared_minus_4 (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) :=
by
  sorry

end factorization_of_m_squared_minus_4_l8_8314


namespace average_hit_targets_formula_average_hit_targets_ge_half_l8_8719

open Nat Real

noncomputable def average_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - (1 : ℝ) / n) ^ n)

theorem average_hit_targets_formula (n : ℕ) (h : 0 < n) :
  average_hit_targets n = n * (1 - (1 - (1 : ℝ) / n) ^ n) :=
by
  sorry

theorem average_hit_targets_ge_half (n : ℕ) (h : 0 < n) :
  average_hit_targets n ≥ n / 2 :=
by
  sorry

end average_hit_targets_formula_average_hit_targets_ge_half_l8_8719


namespace evaluate_expression_l8_8583

theorem evaluate_expression :
  (2 + 3 / (4 + 5 / (6 + 7 / 8))) = 137 / 52 :=
by
  sorry

end evaluate_expression_l8_8583


namespace function_value_at_6000_l8_8451

theorem function_value_at_6000
  (f : ℝ → ℝ)
  (h0 : f 0 = 1)
  (h1 : ∀ x : ℝ, f (x + 3) = f x + 2 * x + 3) :
  f 6000 = 12000001 :=
by
  sorry

end function_value_at_6000_l8_8451


namespace pat_mark_ratio_l8_8990

theorem pat_mark_ratio :
  ∃ K P M : ℕ, P + K + M = 189 ∧ P = 2 * K ∧ M = K + 105 ∧ P / gcd P M = 1 ∧ M / gcd P M = 3 :=
by
  sorry

end pat_mark_ratio_l8_8990


namespace prove_a_zero_l8_8348

noncomputable def f (x a : ℝ) := (x + a) * log ((2 * x - 1) / (2 * x + 1))

theorem prove_a_zero (a : ℝ) : 
  (∀ x, f (-x a) = f (x a)) → a = 0 :=
by 
  sorry

end prove_a_zero_l8_8348


namespace edric_days_per_week_l8_8725

variable (monthly_salary : ℝ) (hours_per_day : ℝ) (hourly_rate : ℝ) (weeks_per_month : ℝ)
variable (days_per_week : ℝ)

-- Defining the conditions
def monthly_salary_condition : Prop := monthly_salary = 576
def hours_per_day_condition : Prop := hours_per_day = 8
def hourly_rate_condition : Prop := hourly_rate = 3
def weeks_per_month_condition : Prop := weeks_per_month = 4

-- Correct answer
def correct_answer : Prop := days_per_week = 6

-- Proof problem statement
theorem edric_days_per_week :
  monthly_salary_condition monthly_salary ∧
  hours_per_day_condition hours_per_day ∧
  hourly_rate_condition hourly_rate ∧
  weeks_per_month_condition weeks_per_month →
  correct_answer days_per_week :=
by
  sorry

end edric_days_per_week_l8_8725


namespace meal_combinations_l8_8197

theorem meal_combinations :
  let meats := 3
  let vegetables := 5
  let desserts := 5
  let vegetable_combinations := Nat.choose vegetables 3
  meats * vegetable_combinations * desserts = 150 :=
by
  let meats := 3
  let vegetables := 5
  let desserts := 5
  let vegetable_combinations := Nat.choose vegetables 3
  show meats * vegetable_combinations * desserts = 150
  sorry

end meal_combinations_l8_8197


namespace platform_length_259_9584_l8_8692

noncomputable def length_of_platform (speed_kmph time_sec train_length_m : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600  -- conversion from kmph to m/s
  let distance_covered := speed_mps * time_sec
  distance_covered - train_length_m

theorem platform_length_259_9584 :
  length_of_platform 72 26 260.0416 = 259.9584 :=
by sorry

end platform_length_259_9584_l8_8692


namespace fred_added_nine_l8_8993

def onions_in_basket (initial_onions : ℕ) (added_by_sara : ℕ) (taken_by_sally : ℕ) (added_by_fred : ℕ) : ℕ :=
  initial_onions + added_by_sara - taken_by_sally + added_by_fred

theorem fred_added_nine : ∀ (S F : ℕ), onions_in_basket S 4 5 F = S + 8 → F = 9 :=
by
  intros S F h
  sorry

end fred_added_nine_l8_8993


namespace circle_condition_l8_8612

-- Define the center of the circle
def center := ((-3 + 27) / 2, (0 + 0) / 2)

-- Define the radius of the circle
def radius := 15

-- Define the circle's equation
def circle_eq (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the final Lean 4 statement
theorem circle_condition (x : ℝ) : circle_eq x 12 → (x = 21 ∨ x = 3) :=
  by
  intro h
  -- Proof goes here
  sorry

end circle_condition_l8_8612


namespace stickers_on_first_page_l8_8153

theorem stickers_on_first_page :
  ∀ (a b c d e : ℕ), 
    (b = 16) →
    (c = 24) →
    (d = 32) →
    (e = 40) →
    (b - a = 8) →
    (c - b = 8) →
    (d - c = 8) →
    (e - d = 8) →
    a = 8 :=
by
  intros a b c d e hb hc hd he h1 h2 h3 h4
  -- Proof would go here
  sorry

end stickers_on_first_page_l8_8153


namespace find_A_d_minus_B_d_l8_8810

-- Definitions for the proof problem
variables {d A B : ℕ}
variables {ad bd : ℤ} -- Representing A_d and B_d in ℤ for arithmetic operations

-- Conditions
constants (base_check : d > 7)
constants (digit_check : A < d ∧ B < d)
constants (encoded_check : d^1 * A + d^0 * B + d^2 * A + d^1 * A = 1 * d^2 + 7 * d^1 + 2 * d^0)

-- The theorem to prove
theorem find_A_d_minus_B_d : A - B = 5 :=
good sorry

end find_A_d_minus_B_d_l8_8810


namespace independent_sum_of_projections_l8_8508

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem independent_sum_of_projections (A1 A2 A3 P P1 P2 P3 : ℝ × ℝ) 
  (h_eq_triangle : distance A1 A2 = distance A2 A3 ∧ distance A2 A3 = distance A3 A1)
  (h_proj_P1 : P1 = (P.1, A2.2))
  (h_proj_P2 : P2 = (P.1, A3.2))
  (h_proj_P3 : P3 = (P.1, A1.2)) :
  distance A1 P2 + distance A2 P3 + distance A3 P1 = (3 / 2) * distance A1 A2 := 
sorry

end independent_sum_of_projections_l8_8508


namespace container_marbles_volume_l8_8912

theorem container_marbles_volume {V₁ V₂ m₁ m₂ : ℕ} 
  (h₁ : V₁ = 24) (h₂ : m₁ = 75) (h₃ : V₂ = 72) :
  m₂ = 225 :=
by
  have proportion := (m₁ : ℚ) / V₁
  have proportion2 := (m₂ : ℚ) / V₂
  have h4 := proportion = proportion2
  sorry

end container_marbles_volume_l8_8912


namespace option_A_is_incorrect_option_B_is_incorrect_option_C_is_correct_option_D_is_incorrect_l8_8258

variable (x y: ℝ)

theorem option_A_is_incorrect : 5 - 3 * (x + 1) ≠ 5 - 3 * x - 1 := 
by sorry

theorem option_B_is_incorrect : 2 - 4 * (x + 1/4) ≠ 2 - 4 * x + 1 := 
by sorry

theorem option_C_is_correct : 2 - 4 * (1/4 * x + 1) = 2 - x - 4 := 
by sorry

theorem option_D_is_incorrect : 2 * (x - 2) - 3 * (y - 1) ≠ 2 * x - 4 - 3 * y - 3 := 
by sorry

end option_A_is_incorrect_option_B_is_incorrect_option_C_is_correct_option_D_is_incorrect_l8_8258


namespace number_of_special_three_digit_numbers_l8_8052

open Nat

def isThreeDigit (n : ℕ) : Prop := n >= 100 ∧ n <= 999
def hasDigit (d : ℕ) (n : ℕ) : Prop := ∃ (i : ℕ), n / 10 ^ i % 10 = d
def isMultipleOf4 (n : ℕ) : Prop := n % 4 = 0

theorem number_of_special_three_digit_numbers : 
  (Finset.filter (λ n, isThreeDigit n ∧ hasDigit 2 n ∧ hasDigit 5 n ∧ isMultipleOf4 n) 
  (Finset.range 1000)).card = 21 := sorry

end number_of_special_three_digit_numbers_l8_8052


namespace total_frogs_in_ponds_l8_8867

def pondA_frogs := 32
def pondB_frogs := pondA_frogs / 2

theorem total_frogs_in_ponds : pondA_frogs + pondB_frogs = 48 := by
  sorry

end total_frogs_in_ponds_l8_8867


namespace tangent_line_at_origin_range_of_a_l8_8471

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

theorem tangent_line_at_origin :
  tangent_eq_at_origin (λ x, Real.log (1 + x) + x * Real.exp (-x)) (0, 0) (2) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ∃ c, (x ∈ Ioo (-1 : ℝ) 0 → f a x = 0) ∧ (x ∈ Ioo 0 ∞ → f a x = 0)) →
    a ∈ Iio (-1 : ℝ) :=
sorry

end tangent_line_at_origin_range_of_a_l8_8471


namespace net_percentage_error_in_volume_l8_8702

theorem net_percentage_error_in_volume
  (a : ℝ)
  (side_error : ℝ := 0.03)
  (height_error : ℝ := -0.04)
  (depth_error : ℝ := 0.02) :
  ((1 + side_error) * (1 + height_error) * (1 + depth_error) - 1) * 100 = 0.8656 :=
by
  -- Placeholder for the proof
  sorry

end net_percentage_error_in_volume_l8_8702


namespace solve_xyz_integers_l8_8225

theorem solve_xyz_integers (x y z : ℤ) : x^2 + y^2 + z^2 = 2 * x * y * z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end solve_xyz_integers_l8_8225


namespace product_base_8_units_digit_l8_8156

theorem product_base_8_units_digit :
  let sum := 324 + 73
  let product := sum * 27
  product % 8 = 7 :=
by
  let sum := 324 + 73
  let product := sum * 27
  have h : product % 8 = 7 := by
    sorry
  exact h

end product_base_8_units_digit_l8_8156


namespace even_sum_probability_l8_8252

-- Define the probabilities of even and odd outcomes for each wheel
def probability_even_first_wheel : ℚ := 2 / 3
def probability_odd_first_wheel : ℚ := 1 / 3
def probability_even_second_wheel : ℚ := 3 / 5
def probability_odd_second_wheel : ℚ := 2 / 5

-- Define the probabilities of the scenarios that result in an even sum
def probability_both_even : ℚ := probability_even_first_wheel * probability_even_second_wheel
def probability_both_odd : ℚ := probability_odd_first_wheel * probability_odd_second_wheel

-- Define the total probability of an even sum
def probability_even_sum : ℚ := probability_both_even + probability_both_odd

-- The theorem statement to be proven
theorem even_sum_probability :
  probability_even_sum = 8 / 15 :=
by
  sorry

end even_sum_probability_l8_8252


namespace probability_of_fourth_quadrant_l8_8109

-- Define the four cards
def cards : List ℤ := [0, -1, 2, -3]

-- Define the fourth quadrant condition for point A(m, n)
def in_fourth_quadrant (m n : ℤ) : Prop := m > 0 ∧ n < 0

-- Calculate the probability of a point being in the fourth quadrant
theorem probability_of_fourth_quadrant :
  let points := (cards.product cards).filter (λ ⟨m, n⟩, m ≠ n)
  let favorable := points.filter (λ ⟨m, n⟩, in_fourth_quadrant m n)
  (favorable.length : ℚ) / (points.length : ℚ) = 1 / 6 := by
    sorry

end probability_of_fourth_quadrant_l8_8109


namespace student_count_l8_8099

theorem student_count (N : ℕ) (h1 : ∀ W : ℝ, W - 46 = 86 - 40) (h2 : (86 - 46) = 5 * N) : N = 8 :=
sorry

end student_count_l8_8099


namespace carA_travel_time_l8_8566

theorem carA_travel_time 
    (speedA speedB distanceB : ℕ)
    (ratio : ℕ)
    (timeB : ℕ)
    (h_speedA : speedA = 50)
    (h_speedB : speedB = 100)
    (h_distanceB : distanceB = speedB * timeB)
    (h_ratio : distanceA / distanceB = ratio)
    (h_ratio_value : ratio = 3)
    (h_timeB : timeB = 1)
  : distanceA / speedA = 6 :=
by sorry

end carA_travel_time_l8_8566


namespace power_sum_l8_8243

theorem power_sum (n : ℕ) : (-2 : ℤ)^n + (-2 : ℤ)^(n+1) = 2^n := by
  sorry

end power_sum_l8_8243


namespace total_blue_balloons_l8_8374

theorem total_blue_balloons (Joan_balloons : ℕ) (Melanie_balloons : ℕ) (Alex_balloons : ℕ) 
  (hJoan : Joan_balloons = 60) (hMelanie : Melanie_balloons = 85) (hAlex : Alex_balloons = 37) :
  Joan_balloons + Melanie_balloons + Alex_balloons = 182 :=
by
  sorry

end total_blue_balloons_l8_8374


namespace quadratic_b_value_l8_8986

theorem quadratic_b_value {b m : ℝ} (h : ∀ x, x^2 + b * x + 44 = (x + m)^2 + 8) : b = 12 :=
by
  -- hint for proving: expand (x+m)^2 + 8 and equate it with x^2 + bx + 44 to solve for b 
  sorry

end quadratic_b_value_l8_8986


namespace time_to_cover_length_correct_l8_8439

-- Given conditions
def speed_escalator := 20 -- ft/sec
def length_escalator := 210 -- feet
def speed_person := 4 -- ft/sec

-- Time is distance divided by speed
def time_to_cover_length : ℚ :=
  length_escalator / (speed_escalator + speed_person)

theorem time_to_cover_length_correct :
  time_to_cover_length = 8.75 := by
  sorry

end time_to_cover_length_correct_l8_8439


namespace probability_of_positive_l8_8370

-- Definitions based on the conditions
def balls : List ℚ := [-2, 0, 1/4, 3]
def total_balls : ℕ := 4
def positive_filter (x : ℚ) : Bool := x > 0
def positive_balls : List ℚ := balls.filter positive_filter
def positive_count : ℕ := positive_balls.length
def probability : ℚ := positive_count / total_balls

-- Statement to prove
theorem probability_of_positive : probability = 1 / 2 := by
  sorry

end probability_of_positive_l8_8370


namespace probability_one_piece_is_two_probability_both_pieces_longer_than_two_l8_8143

theorem probability_one_piece_is_two (l1 l2 : ℕ) (h_pos : l1 > 0 ∧ l2 > 0) 
    (h_sum : l1 + l2 = 6) (h_cases : {l1, l2} ⊆ {1,2,3,4,5}) :
    (1/5 : ℚ) = 2/5 :=
by
    sorry

theorem probability_both_pieces_longer_than_two (l1 l2 : ℕ) (h_pos : l1 > 0 ∧ l2 > 0) 
    (h_sum : l1 + l2 = 6) (h_cases : {l1, l2} ⊆ {1,2,3,4,5}) :
    (1/3 : ℚ) = 2/6 :=
by
    sorry

end probability_one_piece_is_two_probability_both_pieces_longer_than_two_l8_8143


namespace cos_A_minus_B_l8_8179

theorem cos_A_minus_B (A B : Real) 
  (h1 : Real.sin A + Real.sin B = -1) 
  (h2 : Real.cos A + Real.cos B = 1/2) :
  Real.cos (A - B) = -3/8 :=
by
  sorry

end cos_A_minus_B_l8_8179


namespace linear_equation_m_equals_neg_3_l8_8596

theorem linear_equation_m_equals_neg_3 
  (m : ℤ)
  (h1 : |m| - 2 = 1)
  (h2 : m - 3 ≠ 0) :
  m = -3 :=
sorry

end linear_equation_m_equals_neg_3_l8_8596


namespace root_interval_sum_l8_8342

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x - 8

def has_root_in_interval (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : Prop :=
  a < b ∧ b - a = 1 ∧ f a < 0 ∧ f b > 0

theorem root_interval_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h : has_root_in_interval a b h1 h2) : 
  a + b = 5 :=
sorry

end root_interval_sum_l8_8342


namespace m_mobile_cheaper_than_t_mobile_l8_8517

theorem m_mobile_cheaper_than_t_mobile :
  let t_mobile_cost := 50 + 3 * 16,
      m_mobile_cost := 45 + 3 * 14
  in
  t_mobile_cost - m_mobile_cost = 11 :=
by
  let t_mobile_cost := 50 + 3 * 16,
  let m_mobile_cost := 45 + 3 * 14,
  show t_mobile_cost - m_mobile_cost = 11,
  calc
    50 + 3 * 16 - (45 + 3 * 14) = 98 - 87 : by rfl
    ... = 11 : by rfl

end m_mobile_cheaper_than_t_mobile_l8_8517


namespace lines_intersecting_sum_a_b_l8_8231

theorem lines_intersecting_sum_a_b 
  (a b : ℝ) 
  (hx : ∃ (x y : ℝ), x = 4 ∧ y = 1 ∧ x = 3 * y + a)
  (hy : ∃ (x y : ℝ), x = 4 ∧ y = 1 ∧ y = 3 * x + b)
  : a + b = -10 :=
by
  sorry

end lines_intersecting_sum_a_b_l8_8231


namespace rachels_game_final_configurations_l8_8992

-- Define the number of cells in the grid
def n : ℕ := 2011

-- Define the number of moves needed
def moves_needed : ℕ := n - 3

-- Define a function that counts the number of distinct final configurations
-- based on the number of fights (f) possible in the given moves.
def final_configurations : ℕ := moves_needed + 1

theorem rachels_game_final_configurations : final_configurations = 2009 :=
by
  -- Calculation shows that moves_needed = 2008 and therefore final_configurations = 2008 + 1 = 2009.
  sorry

end rachels_game_final_configurations_l8_8992


namespace exists_a_for_system_solution_l8_8317

theorem exists_a_for_system_solution (b : ℝ) :
  (∃ a x y : ℝ, y = -b - x^2 ∧ x^2 + y^2 + 8*a^2 = 4 + 4*a*(x + y)) ↔ b ≤ 2*sqrt 2 + 1/4 :=
begin
  sorry
end

end exists_a_for_system_solution_l8_8317


namespace case1_DC_correct_case2_DC_correct_l8_8930

-- Case 1
theorem case1_DC_correct (AB AD : ℝ) (HM BD DH MD : ℝ) (hAB : AB = 10) (hAD : AD = 4)
  (hHM : HM = 6 / 5) (hBD : BD = 2 * Real.sqrt 21) (hDH : DH = 4 * Real.sqrt 21 / 5)
  (hMD : MD = 6 * (Real.sqrt 21 - 1) / 5):
  (BD - HM : ℝ) == (8 * Real.sqrt 21 - 12) / 5 :=
by {
  sorry
}

-- Case 2
theorem case2_DC_correct (AB AD : ℝ) (HM BD DH MD : ℝ) (hAB : AB = 8 * Real.sqrt 2) (hAD : AD = 4)
  (hHM : HM = Real.sqrt 2) (hBD : BD = 4 * Real.sqrt 7) (hDH : DH = Real.sqrt 14)
  (hMD : MD = Real.sqrt 14 - Real.sqrt 2):
  (BD - HM : ℝ) == 2 * Real.sqrt 14 - 2 * Real.sqrt 2 :=
by {
  sorry
}

end case1_DC_correct_case2_DC_correct_l8_8930


namespace missed_interior_angle_l8_8435

  theorem missed_interior_angle (n : ℕ) (x : ℝ) 
    (h1 : (n - 2) * 180 = 2750 + x) : x = 130 := 
  by sorry
  
end missed_interior_angle_l8_8435


namespace exists_inscribed_square_in_pentagon_l8_8091

def regular_polygon (n : ℕ) : Prop := ∃ (s : ℝ), s > 0 ∧ ∀ (i j : ℕ), i ≠ j → dist (vertices i) (vertices j) = s

def inscribe_square_in_pentagon (P : Type) [regular_polygon 5 P] : Prop :=
∃ (S : Type) [is_square S], ∀ (v : vertex S), ∃ (e : edge P), v ∈ e

theorem exists_inscribed_square_in_pentagon : 
  ∃ (P : Type) [regular_polygon 5 P], inscribe_square_in_pentagon P :=
begin
  sorry
end

end exists_inscribed_square_in_pentagon_l8_8091


namespace cone_surface_area_and_volume_l8_8535

theorem cone_surface_area_and_volume
  (r l m : ℝ)
  (h_ratio : (π * r * l) / (π * r * l + π * r^2) = 25 / 32)
  (h_height : m = 96) :
  (π * r * l + π * r^2 = 3584 * π) ∧ ((1 / 3) * π * r^2 * m = 25088 * π) :=
by {
  sorry
}

end cone_surface_area_and_volume_l8_8535


namespace original_faculty_size_l8_8800

theorem original_faculty_size (F : ℝ) (h1 : F * 0.85 * 0.80 = 195) : F = 287 :=
by
  sorry

end original_faculty_size_l8_8800


namespace cube_cut_possible_l8_8199

theorem cube_cut_possible (a b : ℝ) (unit_a : a = 1) (unit_b : b = 1) : 
  ∃ (cut : ℝ → ℝ → Prop), (∀ x y, cut x y → (∃ q r : ℝ, q > 0 ∧ r > 0 ∧ q * r > 1)) :=
sorry

end cube_cut_possible_l8_8199


namespace carl_additional_marbles_l8_8011

def initial_marbles := 12
def lost_marbles := initial_marbles / 2
def additional_marbles_from_mom := 25
def marbles_in_jar_after_game := 41

theorem carl_additional_marbles :
  (marbles_in_jar_after_game - additional_marbles_from_mom) + lost_marbles - initial_marbles = 10 :=
by
  sorry

end carl_additional_marbles_l8_8011


namespace Hans_current_age_l8_8617

variable {H : ℕ} -- Hans' current age

-- Conditions
def Josiah_age (H : ℕ) := 3 * H
def Hans_age_in_3_years (H : ℕ) := H + 3
def Josiah_age_in_3_years (H : ℕ) := Josiah_age H + 3
def sum_of_ages_in_3_years (H : ℕ) := Hans_age_in_3_years H + Josiah_age_in_3_years H

-- Theorem to prove
theorem Hans_current_age : sum_of_ages_in_3_years H = 66 → H = 15 :=
by
  sorry

end Hans_current_age_l8_8617


namespace only_n_1_has_integer_solution_l8_8572

theorem only_n_1_has_integer_solution :
  ∀ n : ℕ, (∃ x : ℤ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ n = 1 := 
by 
  sorry

end only_n_1_has_integer_solution_l8_8572


namespace total_bill_l8_8700

-- Definitions from conditions
def num_people : ℕ := 3
def amount_per_person : ℕ := 45

-- Mathematical proof problem statement
theorem total_bill : num_people * amount_per_person = 135 := by
  sorry

end total_bill_l8_8700


namespace digits_difference_l8_8819

theorem digits_difference (d A B : ℕ) (hd : d > 7) (h : d^2 * A + d * B + d^2 * A + d * A = 1 * d^2 + 7 * d + 2) : 
  A - B = 4 := 
sorry

end digits_difference_l8_8819


namespace MMobile_cheaper_l8_8519

-- Define the given conditions
def TMobile_base_cost : ℕ := 50
def TMobile_additional_cost : ℕ := 16
def MMobile_base_cost : ℕ := 45
def MMobile_additional_cost : ℕ := 14
def additional_lines : ℕ := 3

-- Define functions to calculate total costs
def TMobile_total_cost : ℕ := TMobile_base_cost + TMobile_additional_cost * additional_lines
def MMobile_total_cost : ℕ := MMobile_base_cost + MMobile_additional_cost * additional_lines

-- Statement to be proved
theorem MMobile_cheaper : TMobile_total_cost - MMobile_total_cost = 11 := by
  sorry

end MMobile_cheaper_l8_8519


namespace correct_order_shopping_process_l8_8538

/-- Definition of each step --/
def step1 : String := "The buyer logs into the Taobao website to select products."
def step2 : String := "The buyer selects the product, clicks the buy button, and pays through Alipay."
def step3 : String := "Upon receiving the purchase information, the seller ships the goods to the buyer through a logistics company."
def step4 : String := "The buyer receives the goods, inspects them for any issues, and confirms receipt online."
def step5 : String := "After receiving the buyer's confirmation of receipt, the Taobao website transfers the payment from Alipay to the seller."

/-- The correct sequence of steps --/
def correct_sequence : List String := [
  "The buyer logs into the Taobao website to select products.",
  "The buyer selects the product, clicks the buy button, and pays through Alipay.",
  "Upon receiving the purchase information, the seller ships the goods to the buyer through a logistics company.",
  "The buyer receives the goods, inspects them for any issues, and confirms receipt online.",
  "After receiving the buyer's confirmation of receipt, the Taobao website transfers the payment from Alipay to the seller."
]

theorem correct_order_shopping_process :
  [step1, step2, step3, step4, step5] = correct_sequence :=
by
  sorry

end correct_order_shopping_process_l8_8538


namespace bakery_profit_l8_8989

noncomputable def revenue_per_piece : ℝ := 4
noncomputable def pieces_per_pie : ℕ := 3
noncomputable def pies_per_hour : ℕ := 12
noncomputable def cost_per_pie : ℝ := 0.5

theorem bakery_profit (pieces_per_pie_pos : 0 < pieces_per_pie) 
                      (pies_per_hour_pos : 0 < pies_per_hour) 
                      (cost_per_pie_pos : 0 < cost_per_pie) :
  pies_per_hour * (pieces_per_pie * revenue_per_piece) - (pies_per_hour * cost_per_pie) = 138 := 
sorry

end bakery_profit_l8_8989


namespace group_card_exchanges_l8_8058

theorem group_card_exchanges (x : ℕ) (hx : x * (x - 1) = 90) : x = 10 :=
by { sorry }

end group_card_exchanges_l8_8058


namespace exists_list_with_all_players_l8_8969

-- Definitions and assumptions
variable {Player : Type} 

-- Each player plays against every other player exactly once, and there are no ties.
-- Defining defeats relationship
def defeats (p1 p2 : Player) : Prop :=
  sorry -- Assume some ordering or wins relationship

-- Defining the list of defeats
def list_of_defeats (p : Player) : Set Player :=
  { q | defeats p q ∨ (∃ r, defeats p r ∧ defeats r q) }

-- Main theorem to be proven
theorem exists_list_with_all_players (players : Set Player) :
  (∀ p q : Player, p ∈ players → q ∈ players → p ≠ q → (defeats p q ∨ defeats q p)) →
  ∃ p : Player, (list_of_defeats p) = players \ {p} :=
by
  sorry

end exists_list_with_all_players_l8_8969


namespace alec_string_ways_l8_8002

theorem alec_string_ways :
  let letters := ['A', 'C', 'G', 'N']
  let num_ways := 24 * 2 * 2
  num_ways = 96 := 
by
  sorry

end alec_string_ways_l8_8002


namespace g_symmetric_l8_8834

theorem g_symmetric (g : ℝ → ℝ) (h₀ : ∀ x, x ≠ 0 → (g x + 3 * g (1 / x) = 4 * x ^ 2)) : 
  ∀ x : ℝ, x ≠ 0 → g x = g (-x) :=
by 
  sorry

end g_symmetric_l8_8834


namespace first_day_speed_l8_8134

open Real

-- Define conditions
variables (v : ℝ) (t : ℝ)
axiom distance_home_school : 1.5 = v * (t - 7/60)
axiom second_day_condition : 1.5 = 6 * (t - 8/60)

theorem first_day_speed :
  v = 10 :=
by
  -- The proof will be provided here
  sorry

end first_day_speed_l8_8134


namespace routes_from_A_to_B_l8_8755

theorem routes_from_A_to_B (n_r n_d : ℕ) (n_r_eq : n_r = 3) (n_d_eq : n_d = 2) :
  nat.choose (n_r + n_d) n_r = 10 :=
by
  rw [n_r_eq, n_d_eq]
  exact nat.choose_succ_succ 3 2

end routes_from_A_to_B_l8_8755


namespace largest_coefficient_term_in_expansion_l8_8767

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem largest_coefficient_term_in_expansion :
  ∃ (T : ℕ × ℤ × ℕ), 
  (2 : ℤ) ^ (14 - 1) = 8192 ∧ 
  T = (binom 14 4, 2 ^ 10, 4) ∧ 
  ∀ (k : ℕ), 
    (binom 14 k * (2 ^ (14 - k))) ≤ (binom 14 4 * 2 ^ 10) :=
sorry

end largest_coefficient_term_in_expansion_l8_8767


namespace baker_batches_chocolate_chip_l8_8905

noncomputable def number_of_batches (total_cookies : ℕ) (oatmeal_cookies : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  (total_cookies - oatmeal_cookies) / cookies_per_batch

theorem baker_batches_chocolate_chip (total_cookies oatmeal_cookies cookies_per_batch : ℕ) 
  (h_total : total_cookies = 10) 
  (h_oatmeal : oatmeal_cookies = 4) 
  (h_batch : cookies_per_batch = 3) : 
  number_of_batches total_cookies oatmeal_cookies cookies_per_batch = 2 :=
by
  sorry

end baker_batches_chocolate_chip_l8_8905


namespace estimate_number_of_blue_cards_l8_8059

-- Define the given conditions:
def red_cards : ℕ := 8
def frequency_blue_card : ℚ := 0.6

-- Define the statement that needs to be proved:
theorem estimate_number_of_blue_cards (x : ℕ) 
  (h : (x : ℚ) / (x + red_cards) = frequency_blue_card) : 
  x = 12 :=
  sorry

end estimate_number_of_blue_cards_l8_8059


namespace max_b_n_occurs_at_n_l8_8459

def a_n (n : ℕ) (a1 : ℚ) (d : ℚ) : ℚ :=
  a1 + (n-1) * d

def S_n (n : ℕ) (a1 : ℚ) (d : ℚ) : ℚ :=
  n * a1 + (n * (n-1) / 2) * d

def b_n (n : ℕ) (an : ℚ) : ℚ :=
  (1 + an) / an

theorem max_b_n_occurs_at_n :
  ∀ (n : ℕ) (a1 d : ℚ),
  (a1 = -5/2) →
  (S_n 4 a1 d = 2 * S_n 2 a1 d + 4) →
  n = 4 := sorry

end max_b_n_occurs_at_n_l8_8459


namespace circle_area_from_tangency_conditions_l8_8137

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 - 20 * y^2 = 24

-- Tangency to the x-axis implies the circle's lowest point touches the x-axis
def tangent_to_x_axis (circle : ℝ → ℝ → Prop) : Prop :=
  ∃ r y₀, circle 0 y₀ ∧ y₀ = r

-- The circle is given as having tangency conditions to derive from
theorem circle_area_from_tangency_conditions (circle : ℝ → ℝ → Prop) :
  (∀ x y, circle x y → (x = 0 ∨ hyperbola x y)) →
  tangent_to_x_axis circle →
  ∃ area, area = 504 * Real.pi :=
by
  sorry

end circle_area_from_tangency_conditions_l8_8137


namespace cylinder_radius_original_l8_8975

theorem cylinder_radius_original (r : ℝ) (h : ℝ) (h_given : h = 4) 
    (V_increase_radius : π * (r + 4) ^ 2 * h = π * r ^ 2 * (h + 4)) : 
    r = 12 := 
  by
    sorry

end cylinder_radius_original_l8_8975


namespace arccos_cos_eight_l8_8567

-- Define the conditions
def cos_equivalence (x : ℝ) : Prop := cos x = cos (x - 2 * Real.pi)
def range_principal (x : ℝ) : Prop := 0 ≤ x - 2 * Real.pi ∧ x - 2 * Real.pi ≤ Real.pi

-- State the main proposition
theorem arccos_cos_eight :
  cos_equivalence 8 ∧ range_principal 8 → Real.arccos (Real.cos 8) = 8 - 2 * Real.pi :=
by
  sorry

end arccos_cos_eight_l8_8567


namespace num_small_triangles_l8_8385

-- Define the lengths of the legs of the large and small triangles
variables (a h b k : ℕ)

-- Define the areas of the large and small triangles
def area_large_triangle (a h : ℕ) : ℕ := (a * h) / 2
def area_small_triangle (b k : ℕ) : ℕ := (b * k) / 2

-- Define the main theorem
theorem num_small_triangles (ha : a = 6) (hh : h = 4) (hb : b = 2) (hk : k = 1) :
  (area_large_triangle a h) / (area_small_triangle b k) = 12 :=
by
  sorry

end num_small_triangles_l8_8385


namespace more_males_l8_8921

theorem more_males {Total_attendees Male_attendees : ℕ} (h1 : Total_attendees = 120) (h2 : Male_attendees = 62) :
  Male_attendees - (Total_attendees - Male_attendees) = 4 :=
by
  sorry

end more_males_l8_8921


namespace tens_digit_of_19_pow_2023_l8_8577

theorem tens_digit_of_19_pow_2023 :
  (19^2023 % 100) / 10 % 10 = 5 := 
sorry

end tens_digit_of_19_pow_2023_l8_8577


namespace largest_multiple_of_7_less_than_100_l8_8409

theorem largest_multiple_of_7_less_than_100 : ∃ n : ℕ, 7 * n < 100 ∧ ∀ m : ℕ, 7 * m < 100 → 7 * m ≤ 7 * n := by
  sorry

end largest_multiple_of_7_less_than_100_l8_8409


namespace fraction_students_say_like_actually_dislike_l8_8086

theorem fraction_students_say_like_actually_dislike :
  let n := 200
  let p_l := 0.70
  let p_d := 0.30
  let p_ll := 0.85
  let p_ld := 0.15
  let p_dd := 0.80
  let p_dl := 0.20
  let num_like := p_l * n
  let num_dislike := p_d * n
  let num_ll := p_ll * num_like
  let num_ld := p_ld * num_like
  let num_dd := p_dd * num_dislike
  let num_dl := p_dl * num_dislike
  let total_say_like := num_ll + num_dl
  (num_dl / total_say_like) = 12 / 131 := 
by
  sorry

end fraction_students_say_like_actually_dislike_l8_8086


namespace triangle_area_correct_l8_8068

def vector_2d (x y : ℝ) : ℝ × ℝ := (x, y)

def area_of_triangle (a b : ℝ × ℝ) : ℝ :=
  0.5 * abs (a.1 * b.2 - a.2 * b.1)

def a : ℝ × ℝ := vector_2d 3 2
def b : ℝ × ℝ := vector_2d 1 5

theorem triangle_area_correct : area_of_triangle a b = 6.5 :=
by
  sorry

end triangle_area_correct_l8_8068


namespace loan_difference_l8_8202

noncomputable def future_value (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def monthly_compounding : ℝ :=
  future_value 8000 0.10 12 5

noncomputable def semi_annual_compounding : ℝ :=
  future_value 8000 0.10 2 5

noncomputable def interest_difference : ℝ :=
  monthly_compounding - semi_annual_compounding

theorem loan_difference (P : ℝ) (r : ℝ) (n_m n_s t : ℝ) :
    interest_difference = 745.02 := by sorry

end loan_difference_l8_8202


namespace part1_part2_l8_8056

-- Condition definitions
def income2017 : ℝ := 2500
def income2019 : ℝ := 3600
def n : ℕ := 2

-- Part 1: Prove the annual growth rate
theorem part1 (x : ℝ) (hx : income2019 = income2017 * (1 + x) ^ n) : x = 0.2 :=
by sorry

-- Part 2: Prove reaching 4200 yuan with the same growth rate
theorem part2 (hx : income2019 = income2017 * (1 + 0.2) ^ n) : 3600 * (1 + 0.2) ≥ 4200 :=
by sorry

end part1_part2_l8_8056


namespace least_x_value_l8_8338

variable (a b : ℕ)
variable (positive_int_a : 0 < a)
variable (positive_int_b : 0 < b)
variable (h : 2 * a^5 = 3 * b^2)

theorem least_x_value (h : 2 * a^5 = 3 * b^2) (positive_int_a : 0 < a) (positive_int_b : 0 < b) : ∃ x, x = 15552 ∧ x = 2 * a^5 ∧ x = 3 * b^2 :=
sorry

end least_x_value_l8_8338


namespace smallest_positive_difference_l8_8984

theorem smallest_positive_difference (a b : ℤ) (h : 17 * a + 6 * b = 13) : (∃ n : ℤ, n > 0 ∧ n = a - b) → n = 17 :=
by sorry

end smallest_positive_difference_l8_8984


namespace determine_c_square_of_binomial_l8_8308

theorem determine_c_square_of_binomial (c : ℝ) : (∀ x : ℝ, 16 * x^2 + 40 * x + c = (4 * x + 5)^2) → c = 25 :=
by
  intro h
  have key := h 0
  -- By substitution, we skip the expansion steps and immediately conclude the value of c
  sorry

end determine_c_square_of_binomial_l8_8308


namespace inequality_solution_l8_8963

theorem inequality_solution (a b : ℝ)
  (h₁ : ∀ x, - (1 : ℝ) / 2 < x ∧ x < (1 : ℝ) / 3 → ax^2 + bx + (2 : ℝ) > 0)
  (h₂ : - (1 : ℝ) / 2 = -(b / a))
  (h₃ : (- (1 : ℝ) / 6) = 2 / a) :
  a - b = -10 :=
sorry

end inequality_solution_l8_8963


namespace largest_root_l8_8016

theorem largest_root (p q r : ℝ) (h1 : p + q + r = 3) (h2 : p * q + p * r + q * r = -6) (h3 : p * q * r = -8) :
  max (max p q) r = (1 + Real.sqrt 17) / 2 :=
by
  sorry

end largest_root_l8_8016


namespace find_multiple_of_son_age_l8_8238

variable (F S k : ℕ)

theorem find_multiple_of_son_age
  (h1 : F = k * S + 4)
  (h2 : F + 4 = 2 * (S + 4) + 20)
  (h3 : F = 44) :
  k = 4 :=
by
  sorry

end find_multiple_of_son_age_l8_8238


namespace sum_base_6_l8_8297

-- Define base 6 numbers
def n1 : ℕ := 1 * 6^3 + 4 * 6^2 + 5 * 6^1 + 2 * 6^0
def n2 : ℕ := 2 * 6^3 + 3 * 6^2 + 5 * 6^1 + 4 * 6^0

-- Define the expected result in base 6
def expected_sum : ℕ := 4 * 6^3 + 2 * 6^2 + 5 * 6^1 + 0 * 6^0

-- The theorem to prove
theorem sum_base_6 : n1 + n2 = expected_sum := by
    sorry

end sum_base_6_l8_8297


namespace total_boxes_correct_l8_8276

def boxes_chocolate : ℕ := 2
def boxes_sugar : ℕ := 5
def boxes_gum : ℕ := 2
def total_boxes : ℕ := boxes_chocolate + boxes_sugar + boxes_gum

theorem total_boxes_correct : total_boxes = 9 := by
  sorry

end total_boxes_correct_l8_8276


namespace pure_imaginary_z_squared_l8_8642

-- Formalization in Lean 4
theorem pure_imaginary_z_squared (a : ℝ) (h : a + (1 + a) * I = (1 + a) * I) : (a + (1 + a) * I)^2 = -1 :=
by
  sorry

end pure_imaginary_z_squared_l8_8642


namespace stuffed_animals_mom_gift_l8_8674

theorem stuffed_animals_mom_gift (x : ℕ) :
  (10 + x) + 3 * (10 + x) = 48 → x = 2 :=
by {
  sorry
}

end stuffed_animals_mom_gift_l8_8674


namespace sum_of_below_avg_l8_8675

-- Define class averages
def a1 := 75
def a2 := 85
def a3 := 90
def a4 := 65

-- Define the overall average
def avg : ℚ := (a1 + a2 + a3 + a4) / 4

-- Define a predicate indicating if a class average is below the overall average
def below_avg (a : ℚ) : Prop := a < avg

-- The theorem to prove the required sum of averages below the overall average
theorem sum_of_below_avg : a1 < avg ∧ a4 < avg → a1 + a4 = 140 :=
by
  sorry

end sum_of_below_avg_l8_8675


namespace coordinates_of_A_equidistant_BC_l8_8732

theorem coordinates_of_A_equidistant_BC :
  ∃ z : ℚ, (∀ A B C : ℚ × ℚ × ℚ, A = (0, 0, z) ∧ B = (7, 0, -15) ∧ C = (2, 10, -12) →
  (dist A B = dist A C)) ↔ z = -(13/3) :=
by sorry

end coordinates_of_A_equidistant_BC_l8_8732


namespace tangent_line_at_a1_one_zero_per_interval_l8_8470

noncomputable def f (a x : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

theorem tangent_line_at_a1 (a : ℝ) (h : a = 1) : 
  (∃ (m b : ℝ), ∀ x, f a x = m * x + b ∧ m = 2 ∧ b = 0) :=
by
  sorry

theorem one_zero_per_interval (a : ℝ) :
  (∃ x : ℝ, -1 < x ∧ x < 0 ∧ f a x = 0) ∧ (∃ x : ℝ, 0 < x ∧ f a x = 0) ↔ a < -1 :=
by
  sorry

end tangent_line_at_a1_one_zero_per_interval_l8_8470


namespace smallest_prime_divisor_of_n_is_2_l8_8886

def a := 3^19
def b := 11^13
def n := a + b

theorem smallest_prime_divisor_of_n_is_2 : nat.prime_divisors n = {2} :=
by
  -- The proof is not needed as per instructions.
  -- We'll add a placeholder sorry to complete the statement.
  sorry

end smallest_prime_divisor_of_n_is_2_l8_8886


namespace decimal_to_binary_49_l8_8828

theorem decimal_to_binary_49 : ((49:ℕ) = 6 * 2^4 + 3 * 2^3 + 0 * 2^2 + 0 * 2^1 + 0 * 2^0 + 1) ↔ (110001 = 110001) :=
by
  sorry

end decimal_to_binary_49_l8_8828


namespace knitting_time_is_correct_l8_8783

-- Definitions of the conditions
def time_per_hat : ℕ := 2
def time_per_scarf : ℕ := 3
def time_per_mitten : ℕ := 1
def time_per_sock : ℕ := 3 / 2 -- fractional time in hours
def time_per_sweater : ℕ := 6
def number_of_grandchildren : ℕ := 3

-- Compute total time for one complete outfit
def time_per_outfit : ℕ := time_per_hat + time_per_scarf + (time_per_mitten * 2) + (time_per_sock * 2) + time_per_sweater

-- Compute total time for all outfits
def total_knitting_time : ℕ := number_of_grandchildren * time_per_outfit

-- Prove that total knitting time is 48 hours
theorem knitting_time_is_correct : total_knitting_time = 48 := by
  unfold total_knitting_time time_per_outfit
  norm_num
  sorry

end knitting_time_is_correct_l8_8783


namespace tent_ratio_l8_8383

-- Define variables for tents in different parts of the camp
variables (N E C S T : ℕ)

-- Given conditions
def northernmost (N : ℕ) := N = 100
def center (C N : ℕ) := C = 4 * N
def southern (S : ℕ) := S = 200
def total (T N C E S : ℕ) := T = N + C + E + S

-- Main theorem statement for the proof
theorem tent_ratio (N E C S T : ℕ) 
  (hn : northernmost N)
  (hc : center C N) 
  (hs : southern S)
  (ht : total T N C E S) :
  E / N = 2 :=
by sorry

end tent_ratio_l8_8383


namespace zander_construction_cost_l8_8084

noncomputable def cost_of_cement (num_bags : ℕ) (price_per_bag : ℕ) : ℕ :=
  num_bags * price_per_bag

noncomputable def amount_of_sand (num_lorries : ℕ) (tons_per_lorry : ℕ) : ℕ :=
  num_lorries * tons_per_lorry

noncomputable def cost_of_sand (total_tons : ℕ) (price_per_ton : ℕ) : ℕ :=
  total_tons * price_per_ton

noncomputable def total_cost (cost_cement : ℕ) (cost_sand : ℕ) : ℕ :=
  cost_cement + cost_sand

theorem zander_construction_cost :
  total_cost (cost_of_cement 500 10) (cost_of_sand (amount_of_sand 20 10) 40) = 13000 :=
by
  sorry

end zander_construction_cost_l8_8084


namespace original_annual_pension_l8_8920

theorem original_annual_pension (k x c d r s : ℝ) (h1 : k * (x + c) ^ (3/4) = k * x ^ (3/4) + r)
  (h2 : k * (x + d) ^ (3/4) = k * x ^ (3/4) + s) :
  k * x ^ (3/4) = (r - s) / (0.75 * (d - c)) :=
by sorry

end original_annual_pension_l8_8920


namespace binomial_expansion_const_term_l8_8182

theorem binomial_expansion_const_term (a : ℝ) (h : a > 0) 
  (A : ℝ) (B : ℝ) :
  (A = (15 * a ^ 4)) ∧ (B = 15 * a ^ 2) ∧ (A = 4 * B) → B = 60 := 
by 
  -- The actual proof is omitted
  sorry

end binomial_expansion_const_term_l8_8182


namespace find_c_l8_8024

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 + 19 * x - 84
noncomputable def g (x : ℝ) : ℝ := 4 * x ^ 2 - 12 * x + 5

theorem find_c (c : ℝ) 
  (h1 : ∃ x : ℝ, (⌊c⌋ : ℝ) = x ∧ f x = 0)
  (h2 : ∃ x : ℝ, (c - ⌊c⌋) = x ∧ g x = 0) :
  c = -23 / 2 := by
  sorry

end find_c_l8_8024


namespace average_length_one_third_of_strings_l8_8533

theorem average_length_one_third_of_strings (average_six_strings : ℕ → ℕ → ℕ)
    (average_four_strings : ℕ → ℕ → ℕ)
    (total_length : ℕ → ℕ → ℕ)
    (n m : ℕ) :
    (n = 6) →
    (m = 4) →
    (average_six_strings 80 n = 480) →
    (average_four_strings 85 m = 340) →
    (total_length 2 70 = 140) →
    70 = (480 - 340) / 2 :=
by
  intros h_n h_m avg_six avg_four total_len
  sorry

end average_length_one_third_of_strings_l8_8533


namespace lawn_width_l8_8432

variable (W : ℝ)
variable (h₁ : 80 * 15 + 15 * W - 15 * 15 = 1875)
variable (h₂ : 5625 = 3 * 1875)

theorem lawn_width (h₁ : 80 * 15 + 15 * W - 15 * 15 = 1875) (h₂ : 5625 = 3 * 1875) : 
  W = 60 := 
sorry

end lawn_width_l8_8432


namespace factorization_of_difference_of_squares_l8_8313

variable {R : Type} [CommRing R]

theorem factorization_of_difference_of_squares (m : R) : m^2 - 4 = (m + 2) * (m - 2) :=
by sorry

end factorization_of_difference_of_squares_l8_8313


namespace average_hit_targets_formula_average_hit_targets_ge_half_l8_8718

open Nat Real

noncomputable def average_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - (1 : ℝ) / n) ^ n)

theorem average_hit_targets_formula (n : ℕ) (h : 0 < n) :
  average_hit_targets n = n * (1 - (1 - (1 : ℝ) / n) ^ n) :=
by
  sorry

theorem average_hit_targets_ge_half (n : ℕ) (h : 0 < n) :
  average_hit_targets n ≥ n / 2 :=
by
  sorry

end average_hit_targets_formula_average_hit_targets_ge_half_l8_8718


namespace even_function_a_eq_zero_l8_8354

theorem even_function_a_eq_zero :
  ∀ a, (∀ x, (x + a) * log ((2 * x - 1) / (2 * x + 1)) = (a - x) * log ((1 - 2 * x) / (2 * x + 1)) → a = 0) :=
by
  sorry

end even_function_a_eq_zero_l8_8354


namespace geometric_sequence_solution_l8_8339

theorem geometric_sequence_solution (x : ℝ) (h : ∃ r : ℝ, 12 * r = x ∧ x * r = 3) : x = 6 ∨ x = -6 := 
by
  sorry

end geometric_sequence_solution_l8_8339


namespace smallest_sum_3x3_grid_l8_8438

-- Define the given conditions
def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9] -- List of numbers used in the grid
def total_sum : ℕ := 45 -- Total sum of numbers from 1 to 9
def grid_size : ℕ := 3 -- Size of the grid
def corners_ids : List Nat := [0, 2, 6, 8] -- Indices of the corners in the grid
def remaining_sum : ℕ := 25 -- Sum of the remaining 5 numbers (after excluding the corners)

-- Define the goal: Prove that the smallest sum s is achieved
theorem smallest_sum_3x3_grid : ∃ s : ℕ, 
  (∀ (r : Fin grid_size) (c : Fin grid_size),
    r + c = s) → (s = 12) :=
by
  sorry

end smallest_sum_3x3_grid_l8_8438


namespace cuboid_edge_lengths_l8_8831

theorem cuboid_edge_lengths (a b c : ℕ) (S V : ℕ) :
  (S = 2 * (a * b + b * c + c * a)) ∧ (V = a * b * c) ∧ (V = S) ∧ 
  (∃ d : ℕ, d = Int.sqrt (a^2 + b^2 + c^2)) →
  (∃ a b c : ℕ, a = 4 ∧ b = 8 ∧ c = 8) :=
by
  sorry

end cuboid_edge_lengths_l8_8831


namespace region_ratio_l8_8142

theorem region_ratio (side_length : ℝ) (s r : ℝ) 
  (h1 : side_length = 2)
  (h2 : s = (1 / 2) * (1 : ℝ) * (1 : ℝ))
  (h3 : r = (1 / 2) * (Real.sqrt 2) * (Real.sqrt 2)) :
  r / s = 2 :=
by
  sorry

end region_ratio_l8_8142


namespace max_marks_l8_8516

theorem max_marks {M : ℝ} (h : 0.90 * M = 550) : M = 612 :=
sorry

end max_marks_l8_8516


namespace range_of_a_l8_8595

open Real

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 1 → ln x - a * (1 - 1 / x) ≥ 0) → a ≤ 1 :=
by
  sorry

end range_of_a_l8_8595


namespace expand_expression_l8_8729

theorem expand_expression : 
  (5 * x^2 + 2 * x - 3) * (3 * x^3 - x^2) = 15 * x^5 + x^4 - 11 * x^3 + 3 * x^2 := 
by
  sorry

end expand_expression_l8_8729


namespace sum_of_50th_terms_l8_8014

open Nat

-- Definition of arithmetic sequence
def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Definition of geometric sequence
def geometric_sequence (g₁ r n : ℕ) : ℕ := g₁ * r^(n - 1)

-- Prove the sum of the 50th terms of the given sequences
theorem sum_of_50th_terms : 
  arithmetic_sequence 3 6 50 + geometric_sequence 2 3 50 = 297 + 2 * 3^49 :=
by
  sorry

end sum_of_50th_terms_l8_8014


namespace pyramid_base_side_length_l8_8097

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (s : ℝ)
  (h_area_lateral_face : area_lateral_face = 144)
  (h_slant_height : slant_height = 24) :
  (1 / 2) * s * slant_height = area_lateral_face → s = 12 :=
by
  sorry

end pyramid_base_side_length_l8_8097


namespace pyramid_base_edge_length_l8_8228

noncomputable def edge_length_of_pyramid_base : ℝ :=
  let R := 4 -- radius of the hemisphere
  let h := 12 -- height of the pyramid
  let base_length := 6 -- edge-length of the base of the pyramid to be proved
  -- assume necessary geometric configurations of the pyramid and sphere
  base_length

theorem pyramid_base_edge_length :
  ∀ R h base_length, R = 4 → h = 12 → edge_length_of_pyramid_base = base_length → base_length = 6 :=
by
  intros R h base_length hR hH hBaseLength
  have R_spec : R = 4 := hR
  have h_spec : h = 12 := hH
  have base_length_spec : edge_length_of_pyramid_base = base_length := hBaseLength
  sorry

end pyramid_base_edge_length_l8_8228


namespace find_m_from_root_l8_8743

theorem find_m_from_root (m : ℝ) : (x : ℝ) = 1 → x^2 + m * x + 2 = 0 → m = -3 :=
by
  sorry

end find_m_from_root_l8_8743


namespace find_speed_range_l8_8653

noncomputable def runningErrorB (v : ℝ) : ℝ := abs ((300 / v) - 7)
noncomputable def runningErrorC (v : ℝ) : ℝ := abs ((480 / v) - 11)

theorem find_speed_range (v : ℝ) :
  (runningErrorB v + runningErrorC v ≤ 2) →
  33.33 ≤ v ∧ v ≤ 48.75 := sorry

end find_speed_range_l8_8653


namespace part1_average_decrease_rate_part2_unit_price_reduction_l8_8911

-- Part 1: Prove the average decrease rate is 10%
theorem part1_average_decrease_rate (p0 p2 : ℝ) (x : ℝ) 
    (h1 : p0 = 200) 
    (h2 : p2 = 162) 
    (hx : (1 - x)^2 = p2 / p0) : x = 0.1 :=
by {
    sorry
}

-- Part 2: Prove the unit price reduction should be 15 yuan
theorem part2_unit_price_reduction (p_sell p_factory profit : ℝ) (n_initial dn m : ℝ)
    (h3 : p_sell = 200)
    (h4 : p_factory = 162)
    (h5 : n_initial = 20)
    (h6 : dn = 10)
    (h7 : profit = 1150)
    (hx : (38 - m) * (n_initial + 2 * m) = profit) : m = 15 :=
by {
    sorry
}

end part1_average_decrease_rate_part2_unit_price_reduction_l8_8911


namespace math_problem_l8_8504
-- Import the entire mathlib library for necessary mathematical definitions and notations

-- Define the conditions and the statement to prove
theorem math_problem (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 :=
by 
  -- place a sorry as a placeholder for the proof
  sorry

end math_problem_l8_8504


namespace gcd_polynomial_correct_l8_8042

noncomputable def gcd_polynomial (b : ℤ) := 5 * b^3 + b^2 + 8 * b + 38

theorem gcd_polynomial_correct (b : ℤ) (h : 342 ∣ b) : Int.gcd (gcd_polynomial b) b = 38 := by
  sorry

end gcd_polynomial_correct_l8_8042


namespace routes_A_to_B_on_3x2_grid_l8_8756

def routes_from_A_to_B : ℕ := 10

/-- Prove the number of different routes from point A to point B on a 3x2 grid --/
theorem routes_A_to_B_on_3x2_grid : routes_from_A_to_B = (nat.choose 5 2) := by
  sorry

end routes_A_to_B_on_3x2_grid_l8_8756


namespace probability_of_rolling_four_threes_l8_8603
open BigOperators

def probability_four_threes (n : ℕ) (k : ℕ) (p : ℚ) (q : ℚ) : ℚ := 
  (n.choose k) * (p ^ k) * (q ^ (n - k))

theorem probability_of_rolling_four_threes : 
  probability_four_threes 5 4 (1 / 10) (9 / 10) = 9 / 20000 := 
by 
  sorry

end probability_of_rolling_four_threes_l8_8603


namespace bills_difference_l8_8019

noncomputable def Mike_tip : ℝ := 5
noncomputable def Joe_tip : ℝ := 10
noncomputable def Mike_percentage : ℝ := 20
noncomputable def Joe_percentage : ℝ := 25

theorem bills_difference
  (m j : ℝ)
  (Mike_condition : (Mike_percentage / 100) * m = Mike_tip)
  (Joe_condition : (Joe_percentage / 100) * j = Joe_tip) :
  |m - j| = 15 :=
by
  sorry

end bills_difference_l8_8019


namespace remainder_of_4_pow_a_div_10_l8_8895

theorem remainder_of_4_pow_a_div_10 (a : ℕ) (h1 : a > 0) (h2 : a % 2 = 0) :
  (4 ^ a) % 10 = 6 :=
by sorry

end remainder_of_4_pow_a_div_10_l8_8895


namespace albert_earnings_l8_8359

theorem albert_earnings (E E_final : ℝ) : 
  (0.90 * (E * 1.14) = 678) → 
  (E_final = 0.90 * (E * 1.15 * 1.20)) → 
  E_final = 819.72 :=
by
  sorry

end albert_earnings_l8_8359


namespace m_mobile_cheaper_than_t_mobile_l8_8518

theorem m_mobile_cheaper_than_t_mobile :
  let t_mobile_cost := 50 + 3 * 16,
      m_mobile_cost := 45 + 3 * 14
  in
  t_mobile_cost - m_mobile_cost = 11 :=
by
  let t_mobile_cost := 50 + 3 * 16,
  let m_mobile_cost := 45 + 3 * 14,
  show t_mobile_cost - m_mobile_cost = 11,
  calc
    50 + 3 * 16 - (45 + 3 * 14) = 98 - 87 : by rfl
    ... = 11 : by rfl

end m_mobile_cheaper_than_t_mobile_l8_8518


namespace min_value_f_solve_inequality_f_l8_8776

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (2 * x + 4)

-- Proof Problem 1
theorem min_value_f : ∃ x : ℝ, f x = 3 :=
by { sorry }

-- Proof Problem 2
theorem solve_inequality_f : {x : ℝ | abs (f x - 6) ≤ 1} = 
    ({x : ℝ | -10/3 ≤ x ∧ x ≤ -8/3} ∪ 
    {x : ℝ | 0 ≤ x ∧ x ≤ 1} ∪ 
    {x : ℝ | 1 < x ∧ x ≤ 4/3}) :=
by { sorry }

end min_value_f_solve_inequality_f_l8_8776


namespace cook_one_potato_l8_8136

theorem cook_one_potato (total_potatoes cooked_potatoes remaining_potatoes remaining_time : ℕ) 
  (h1 : total_potatoes = 15) 
  (h2 : cooked_potatoes = 6) 
  (h3 : remaining_time = 72)
  (h4 : remaining_potatoes = total_potatoes - cooked_potatoes) :
  (remaining_time / remaining_potatoes) = 8 :=
by
  sorry

end cook_one_potato_l8_8136


namespace right_triangle_area_l8_8731

-- Define the conditions a = 4/3 * b and a = 2/3 * c
variable (a b c : ℝ)
hypothesis h1 : a = (4 / 3) * b
hypothesis h2 : a = (2 / 3) * c

-- Define the theorem stating that the area of the right triangle is 2/3
theorem right_triangle_area : (1 / 2) * a * b = (2 / 3) := by
  sorry

end right_triangle_area_l8_8731


namespace intersection_of_A_and_B_l8_8499

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
  sorry

end intersection_of_A_and_B_l8_8499


namespace passing_percentage_correct_l8_8696

-- The given conditions
def marks_obtained : ℕ := 175
def marks_failed : ℕ := 89
def max_marks : ℕ := 800

-- The theorem to prove
theorem passing_percentage_correct :
  (
    (marks_obtained + marks_failed : ℕ) * 100 / max_marks
  ) = 33 :=
sorry

end passing_percentage_correct_l8_8696


namespace wrap_XL_boxes_per_roll_l8_8599

-- Conditions
def rolls_per_shirt_box : ℕ := 5
def num_shirt_boxes : ℕ := 20
def num_XL_boxes : ℕ := 12
def cost_per_roll : ℕ := 4
def total_cost : ℕ := 32

-- Prove that one roll of wrapping paper can wrap 3 XL boxes
theorem wrap_XL_boxes_per_roll : (num_XL_boxes / ((total_cost / cost_per_roll) - (num_shirt_boxes / rolls_per_shirt_box))) = 3 := 
sorry

end wrap_XL_boxes_per_roll_l8_8599


namespace M_Mobile_cheaper_than_T_Mobile_l8_8521

def T_Mobile_total_cost (lines : ℕ) : ℕ :=
  if lines <= 2 then 50
  else 50 + (lines - 2) * 16

def M_Mobile_total_cost (lines : ℕ) : ℕ :=
  if lines <= 2 then 45
  else 45 + (lines - 2) * 14

theorem M_Mobile_cheaper_than_T_Mobile : 
  T_Mobile_total_cost 5 - M_Mobile_total_cost 5 = 11 :=
by
  sorry

end M_Mobile_cheaper_than_T_Mobile_l8_8521


namespace snow_fall_time_l8_8190

theorem snow_fall_time :
  (∀ rate_per_six_minutes : ℕ, rate_per_six_minutes = 1 →
    (∀ minute : ℕ, minute = 6 →
      (∀ height_in_m : ℕ, height_in_m = 1 →
        ∃ time_in_hours : ℕ, time_in_hours = 100 ))) :=
sorry

end snow_fall_time_l8_8190


namespace inequalities_validity_l8_8079

theorem inequalities_validity (x y a b : ℝ) (hx : x ≤ a) (hy : y ≤ b) (hstrict : x < a ∨ y < b) :
  (x + y ≤ a + b) ∧
  ¬((x - y ≤ a - b) ∧ (x * y ≤ a * b) ∧ (x / y ≤ a / b)) :=
by
  -- Here is where the proof would go.
  sorry

end inequalities_validity_l8_8079


namespace inequality_ac2_geq_bc2_l8_8053

theorem inequality_ac2_geq_bc2 (a b c : ℝ) (h : a > b) : a * c^2 ≥ b * c^2 :=
sorry

end inequality_ac2_geq_bc2_l8_8053


namespace enchilada_taco_cost_l8_8219

theorem enchilada_taco_cost (e t : ℝ) 
  (h1 : 3 * e + 4 * t = 3.50) 
  (h2 : 4 * e + 3 * t = 3.90) : 
  4 * e + 5 * t = 4.56 := 
sorry

end enchilada_taco_cost_l8_8219


namespace total_payment_l8_8082

def cement_bags := 500
def cost_per_bag := 10
def lorries := 20
def tons_per_lorry := 10
def cost_per_ton := 40

theorem total_payment : cement_bags * cost_per_bag + lorries * tons_per_lorry * cost_per_ton = 13000 := by
  sorry

end total_payment_l8_8082


namespace probability_diagonals_intersect_hexagon_l8_8609

theorem probability_diagonals_intersect_hexagon:
  let n : ℕ := 6
  let total_diagonals := (n * (n - 3)) / 2 -- Total number of diagonals in a convex polygon
  let total_pairs := (total_diagonals * (total_diagonals - 1)) / 2 -- Total number of ways to choose 2 diagonals
  let non_principal_intersections := 3 * 6 -- Each of 6 non-principal diagonals intersects 3 others
  let principal_intersections := 4 * 3 -- Each of 3 principal diagonals intersects 4 others
  let total_intersections := (non_principal_intersections + principal_intersections) / 2 -- Correcting for double-counting
  let probability := total_intersections / total_pairs -- Probability of intersection inside the hexagon
  probability = 5 / 12 := by
  let n : ℕ := 6
  let total_diagonals := (n * (n - 3)) / 2
  let total_pairs := (total_diagonals * (total_diagonals - 1)) / 2
  let non_principal_intersections := 3 * 6
  let principal_intersections := 4 * 3
  let total_intersections := (non_principal_intersections + principal_intersections) / 2
  let probability := total_intersections / total_pairs
  have h : total_diagonals = 9 := by norm_num
  have h_pairs : total_pairs = 36 := by norm_num
  have h_intersections : total_intersections = 15 := by norm_num
  have h_prob : probability = 5 / 12 := by norm_num
  exact h_prob

end probability_diagonals_intersect_hexagon_l8_8609


namespace flower_count_l8_8493

variables (o y p : ℕ)

theorem flower_count (h1 : y + p = 7) (h2 : o + p = 10) (h3 : o + y = 5) : o + y + p = 11 := sorry

end flower_count_l8_8493


namespace participants_coffee_l8_8868

theorem participants_coffee (n : ℕ) (h1 : n = 14) (h2 : 0 < (14 - 2 * (n / 2)) < 14) : 
  ∃ (k : ℕ), k ∈ {6, 8, 10, 12} ∧ k = (14 - 2 * (n / 2)) :=
by sorry

end participants_coffee_l8_8868


namespace average_decrease_rate_required_price_reduction_l8_8910

-- Define the conditions
def factory_price_2019 : ℝ := 200
def factory_price_2021 : ℝ := 162
def daily_sold_2019 : ℕ := 20
def price_increase_per_reduction : ℕ := 10
def price_reduction_per_unit : ℝ := 5
def target_daily_profit : ℝ := 1150

-- Part 1: Prove the average decrease rate
theorem average_decrease_rate : 
  ∃ (x : ℝ), (factory_price_2019 * (1 - x)^2 = factory_price_2021) ∧ x = 0.1 :=
begin
  sorry
end

-- Part 2: Prove the required unit price reduction
theorem required_price_reduction :
  ∃ (m : ℝ), ((38 - m) * (daily_sold_2019 + 2 * m / price_reduction_per_unit) = target_daily_profit) ∧ m = 15 :=
begin
  sorry
end

end average_decrease_rate_required_price_reduction_l8_8910


namespace movie_of_the_year_condition_l8_8547

theorem movie_of_the_year_condition (total_lists : ℕ) (fraction : ℚ) (num_lists : ℕ) 
  (h1 : total_lists = 775) (h2 : fraction = 1 / 4) (h3 : num_lists = ⌈fraction * total_lists⌉) : 
  num_lists = 194 :=
by
  -- Using the conditions given,
  -- total_lists = 775,
  -- fraction = 1 / 4,
  -- num_lists = ⌈fraction * total_lists⌉
  -- We need to show num_lists = 194.
  sorry

end movie_of_the_year_condition_l8_8547


namespace point_not_on_graph_l8_8415

theorem point_not_on_graph : ∀ (x y : ℝ), (x, y) = (-1, 1) → ¬ (∃ z : ℝ, z ≠ -1 ∧ y = z / (z + 1)) :=
by {
  sorry
}

end point_not_on_graph_l8_8415


namespace anthony_path_shortest_l8_8147

noncomputable def shortest_distance (A B C D M : ℝ) : ℝ :=
  4 + 2 * Real.sqrt 3

theorem anthony_path_shortest {A B C D : ℝ} (M : ℝ) (side_length : ℝ) (h : side_length = 4) : 
  shortest_distance A B C D M = 4 + 2 * Real.sqrt 3 :=
by 
  sorry

end anthony_path_shortest_l8_8147


namespace triangle_area_is_96_l8_8361

/-- Given a square with side length 8 and an overlapping area that is both three-quarters
    of the area of the square and half of the area of a triangle, prove the triangle's area is 96. -/
theorem triangle_area_is_96 (a : ℕ) (area_of_square : ℕ) (overlapping_area : ℕ) (area_of_triangle : ℕ) 
  (h1 : a = 8) 
  (h2 : area_of_square = a * a) 
  (h3 : overlapping_area = (3 * area_of_square) / 4) 
  (h4 : overlapping_area = area_of_triangle / 2) : 
  area_of_triangle = 96 := 
by 
  sorry

end triangle_area_is_96_l8_8361


namespace intersection_is_ge_negative_one_l8_8985

noncomputable def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 4*x + 3}
noncomputable def N : Set ℝ := {y | ∃ x : ℝ, y = x - 1}

theorem intersection_is_ge_negative_one : M ∩ N = {y | y ≥ -1} := by
  sorry

end intersection_is_ge_negative_one_l8_8985


namespace average_hit_targets_formula_average_hit_targets_ge_half_l8_8720

open Nat Real

noncomputable def average_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - (1 : ℝ) / n) ^ n)

theorem average_hit_targets_formula (n : ℕ) (h : 0 < n) :
  average_hit_targets n = n * (1 - (1 - (1 : ℝ) / n) ^ n) :=
by
  sorry

theorem average_hit_targets_ge_half (n : ℕ) (h : 0 < n) :
  average_hit_targets n ≥ n / 2 :=
by
  sorry

end average_hit_targets_formula_average_hit_targets_ge_half_l8_8720


namespace students_in_circle_l8_8900

theorem students_in_circle (n : ℕ) (h1 : n > 6) (h2 : n > 16) (h3 : n / 2 = 10) : n + 2 = 22 := by
  sorry

end students_in_circle_l8_8900


namespace planting_trees_system_of_equations_l8_8991

/-- This formalizes the problem where we have 20 young pioneers in total, 
each boy planted 3 trees, each girl planted 2 trees,
and together they planted a total of 52 tree seedlings.
We need to formalize proving that the system of linear equations is as follows:
x + y = 20
3x + 2y = 52
-/
theorem planting_trees_system_of_equations (x y : ℕ) (h1 : x + y = 20)
  (h2 : 3 * x + 2 * y = 52) : 
  (x + y = 20 ∧ 3 * x + 2 * y = 52) :=
by
  exact ⟨h1, h2⟩

end planting_trees_system_of_equations_l8_8991


namespace initial_friends_l8_8809

theorem initial_friends (n : ℕ) (h1 : 120 / (n - 4) = 120 / n + 8) : n = 10 := 
by
  sorry

end initial_friends_l8_8809


namespace profit_percentage_is_12_36_l8_8678

noncomputable def calc_profit_percentage (SP CP : ℝ) : ℝ :=
  let Profit := SP - CP
  (Profit / CP) * 100

theorem profit_percentage_is_12_36
  (SP : ℝ) (h1 : SP = 100)
  (CP : ℝ) (h2 : CP = 0.89 * SP) :
  calc_profit_percentage SP CP = 12.36 :=
by
  sorry

end profit_percentage_is_12_36_l8_8678


namespace max_value_f_on_interval_l8_8999

open Real

def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 1

theorem max_value_f_on_interval :
  ∃ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), ∀ y ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f y ≤ f x ∧ f x = 23 := by
  sorry

end max_value_f_on_interval_l8_8999


namespace shells_collected_by_savannah_l8_8064

def num_shells_jillian : ℕ := 29
def num_shells_clayton : ℕ := 8
def total_shells_distributed : ℕ := 54

theorem shells_collected_by_savannah (S : ℕ) :
  num_shells_jillian + S + num_shells_clayton = total_shells_distributed → S = 17 :=
by
  sorry

end shells_collected_by_savannah_l8_8064


namespace find_c1_in_polynomial_q_l8_8507

theorem find_c1_in_polynomial_q
  (m : ℕ)
  (hm : m ≥ 5)
  (hm_odd : m % 2 = 1)
  (D : ℕ → ℕ)
  (hD_q : ∃ (c3 c2 c1 c0 : ℤ), ∀ (m : ℕ), m % 2 = 1 ∧ m ≥ 5 → D m = (c3 * m^3 + c2 * m^2 + c1 * m + c0)) :
  ∃ (c1 : ℤ), c1 = 11 :=
sorry

end find_c1_in_polynomial_q_l8_8507


namespace largest_integral_value_l8_8733

theorem largest_integral_value (x : ℤ) : (1 / 3 : ℚ) < x / 5 ∧ x / 5 < 5 / 8 → x = 3 :=
by
  sorry

end largest_integral_value_l8_8733


namespace julia_initial_money_l8_8773

theorem julia_initial_money 
  (M : ℚ) 
  (h1 : M / 2 - M / 8 = 15) : 
  M = 40 := 
sorry

end julia_initial_money_l8_8773


namespace john_total_skateboarded_miles_l8_8375

-- Definitions
def distance_skateboard_to_park := 16
def distance_walk := 8
def distance_bike := 6
def distance_skateboard_home := distance_skateboard_to_park

-- Statement to prove
theorem john_total_skateboarded_miles : 
  distance_skateboard_to_park + distance_skateboard_home = 32 := 
by
  sorry

end john_total_skateboarded_miles_l8_8375


namespace intersection_complement_is_l8_8624

universe u
variable {α : Type u}

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {1, 3, 5}

theorem intersection_complement_is :
  N ∩ (U \ M) = {3, 5} :=
  sorry

end intersection_complement_is_l8_8624


namespace firm_partners_initial_count_l8_8274

theorem firm_partners_initial_count
  (x : ℕ)
  (h1 : 2*x/(63*x + 35) = 1/34)
  (h2 : 2*x/(20*x + 10) = 1/15) :
  2*x = 14 :=
by
  sorry

end firm_partners_initial_count_l8_8274


namespace find_solutions_l8_8469

theorem find_solutions (x y : Real) :
    (x = 1 ∧ y = 2) ∨
    (x = 1 ∧ y = 0) ∨
    (x = -4 ∧ y = 6) ∨
    (x = -5 ∧ y = 2) ∨
    (x = -3 ∧ y = 0) ↔
    x^2 + x*y + y^2 + 2*x - 3*y - 3 = 0 := by
  sorry

end find_solutions_l8_8469


namespace find_angle_B_find_triangle_area_l8_8366

open Real

theorem find_angle_B (B : ℝ) (h : sqrt 3 * sin (2 * B) = 1 - cos (2 * B)) : B = π / 3 :=
sorry

theorem find_triangle_area (BC A B : ℝ) (hBC : BC = 2) (hA : A = π / 4) (hB : B = π / 3) :
  let AC := BC * (sin B / sin A)
  let C := π - A - B
  let area := (1 / 2) * AC * BC * sin C
  area = (3 + sqrt 3) / 2 :=
sorry


end find_angle_B_find_triangle_area_l8_8366


namespace digits_difference_l8_8820

theorem digits_difference (d A B : ℕ) (hd : d > 7) (h : d^2 * A + d * B + d^2 * A + d * A = 1 * d^2 + 7 * d + 2) : 
  A - B = 4 := 
sorry

end digits_difference_l8_8820


namespace binary_to_decimal_eq_l8_8015

theorem binary_to_decimal_eq :
  (1 * 2^7 + 1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 205 :=
by
  sorry

end binary_to_decimal_eq_l8_8015


namespace total_keys_needed_l8_8543

-- Definitions based on given conditions
def num_complexes : ℕ := 2
def num_apartments_per_complex : ℕ := 12
def keys_per_lock : ℕ := 3
def num_locks_per_apartment : ℕ := 1

-- Theorem stating the required number of keys
theorem total_keys_needed : 
  (num_complexes * num_apartments_per_complex * keys_per_lock = 72) :=
by
  sorry

end total_keys_needed_l8_8543


namespace area_of_rhombus_l8_8901

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 22) (h2 : d2 = 30) : 
  (d1 * d2) / 2 = 330 :=
by
  rw [h1, h2]
  norm_num

-- Here we state the theorem about the area of the rhombus given its diagonal lengths.

end area_of_rhombus_l8_8901


namespace problem_range_of_a_l8_8324

theorem problem_range_of_a (a : ℝ) :
  (∀ x : ℝ, |2 - x| + |3 + x| ≥ a^2 - 4 * a) ↔ -1 ≤ a ∧ a ≤ 5 :=
by
  sorry

end problem_range_of_a_l8_8324


namespace complex_quadratic_solution_l8_8163

theorem complex_quadratic_solution (a b : ℝ) (h₁ : ∀ (x : ℂ), 5 * x ^ 2 - 4 * x + 20 = 0 → x = a + b * Complex.I ∨ x = a - b * Complex.I) :
 a + b ^ 2 = 394 / 25 := 
sorry

end complex_quadratic_solution_l8_8163


namespace maximum_value_l8_8461

theorem maximum_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) :
  (a / (a + 1) + b / (b + 2) ≤ (5 - 2 * Real.sqrt 2) / 4) :=
sorry

end maximum_value_l8_8461


namespace tan_half_angle_l8_8746

-- Definition for the given angle in the third quadrant with a given sine value
def angle_in_third_quadrant_and_sin (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h_sin : Real.sin α = -24 / 25) : Prop :=
  True

-- The main theorem to prove the given condition implies the result
theorem tan_half_angle (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h_sin : Real.sin α = -24 / 25) :
  Real.tan (α / 2) = -4 / 3 :=
by
  sorry

end tan_half_angle_l8_8746


namespace tenth_term_of_sequence_l8_8710

theorem tenth_term_of_sequence : 
  let a_1 := 3
  let d := 6 
  let n := 10 
  (a_1 + (n-1) * d) = 57 := by
  sorry

end tenth_term_of_sequence_l8_8710


namespace molecular_weight_CaCO3_is_100_09_l8_8119

-- Declare the atomic weights
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight constant for calcium carbonate
def molecular_weight_CaCO3 : ℝ :=
  (1 * atomic_weight_Ca) + (1 * atomic_weight_C) + (3 * atomic_weight_O)

-- Prove that the molecular weight of calcium carbonate is 100.09 g/mol
theorem molecular_weight_CaCO3_is_100_09 :
  molecular_weight_CaCO3 = 100.09 :=
by
  -- Proof goes here, placeholder for now
  sorry

end molecular_weight_CaCO3_is_100_09_l8_8119


namespace geometric_sum_six_l8_8858

theorem geometric_sum_six (a r : ℚ) (n : ℕ) 
  (hn₁ : a = 1/4) 
  (hn₂ : r = 1/2) 
  (hS: a * (1 - r^n) / (1 - r) = 63/128) : 
  n = 6 :=
by
  -- Statement to be Proven
  rw [hn₁, hn₂] at hS
  sorry

end geometric_sum_six_l8_8858


namespace intersection_is_correct_l8_8033

-- Define the sets A and B based on the given conditions
def setA : Set ℝ := { x | x > 1/3 }
def setB : Set ℝ := { y | -3 ≤ y ∧ y ≤ 3 }

-- Prove that the intersection of A and B is (1/3, 3]
theorem intersection_is_correct : setA ∩ setB = { x | 1/3 < x ∧ x ≤ 3 } := 
by
  sorry

end intersection_is_correct_l8_8033


namespace expected_hit_targets_expected_hit_targets_not_less_than_half_l8_8723

-- Part (a): The expected number of hit targets
theorem expected_hit_targets (n : ℕ) (h : n ≠ 0) :
  E (number_of_hit_targets n) = n * (1 - (1 - (1 / n)) ^ n) :=
sorry

-- Part (b): The expected number of hit targets cannot be less than n / 2
theorem expected_hit_targets_not_less_than_half (n : ℕ) (h : n ≠ 0) :
  n * (1 - (1 - (1 / n)) ^ n) ≥ n / 2 :=
sorry

end expected_hit_targets_expected_hit_targets_not_less_than_half_l8_8723


namespace negation_of_forall_inequality_l8_8100

theorem negation_of_forall_inequality:
  ¬(∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 - x > 0 :=
by
  sorry

end negation_of_forall_inequality_l8_8100


namespace find_A_d_minus_B_d_l8_8818

-- Definitions of the conditions
def is_digit_in_base (x : ℕ) (d : ℕ) : Prop := x < d

def ab_aa_sum_to_172 (A B d : ℕ) : Prop :=
  is_digit_in_base A d ∧ is_digit_in_base B d ∧ d > 7 ∧ (d * A + B) + (d * A + A) = d^2 + 7 * d + 2

-- The final theorem statement
theorem find_A_d_minus_B_d (A B d : ℕ) (h : ab_aa_sum_to_172 A B d) : A - B = 5 :=
by sorry

end find_A_d_minus_B_d_l8_8818


namespace smallest_prime_divisor_of_3_pow_19_plus_11_pow_13_l8_8883

/-- 
  Problem statement: Prove that the smallest prime divisor of 3^19 + 11^13 is 2, 
  given the conditions:
   - 3^19 is odd
   - 11^13 is odd
   - The sum of two odd numbers is even
-/

theorem smallest_prime_divisor_of_3_pow_19_plus_11_pow_13 :
  ∃ p : ℕ, prime p ∧ p = 2 ∧ p ∣ (3^19 + 11^13) := 
by
  sorry

end smallest_prime_divisor_of_3_pow_19_plus_11_pow_13_l8_8883


namespace compare_neg_fractions_l8_8300

theorem compare_neg_fractions : - (1 : ℝ) / 3 < - (1 : ℝ) / 4 :=
  sorry

end compare_neg_fractions_l8_8300


namespace middle_box_label_l8_8105

/--
Given a sequence of 23 boxes in a row on the table, where each box has a label indicating either
  "There is no prize here" or "The prize is in a neighboring box",
and it is known that exactly one of these statements is true.
Prove that the label on the middle box (the 12th box) says "The prize is in the adjacent box."
-/
theorem middle_box_label :
  ∃ (boxes : Fin 23 → Prop) (labels : Fin 23 → String),
    (∀ i, labels i = "There is no prize here" ∨ labels i = "The prize is in a neighboring box") ∧
    (∃! i : Fin 23, boxes i ∧ (labels i = "The prize is in a neighboring box")) →
    labels ⟨11, sorry⟩ = "The prize is in a neighboring box" :=
sorry

end middle_box_label_l8_8105


namespace rate_of_mixed_oil_l8_8960

/--
If 10 litres of an oil at Rs. 50 per litre is mixed with 5 litres of another oil at Rs. 68 per litre, 
8 litres of a third oil at Rs. 42 per litre, and 7 litres of a fourth oil at Rs. 62 per litre, 
then the rate of the mixed oil per litre is Rs. 53.67.
-/
theorem rate_of_mixed_oil :
  let cost1 := 10 * 50
  let cost2 := 5 * 68
  let cost3 := 8 * 42
  let cost4 := 7 * 62
  let total_cost := cost1 + cost2 + cost3 + cost4
  let total_volume := 10 + 5 + 8 + 7
  let rate_per_litre := total_cost / total_volume
  rate_per_litre = 53.67 :=
by
  intros
  sorry

end rate_of_mixed_oil_l8_8960


namespace number_of_three_digit_numbers_with_123_exactly_once_l8_8050

theorem number_of_three_digit_numbers_with_123_exactly_once : 
  (∃ (l : List ℕ), l = [1, 2, 3] ∧ l.permutations.length = 6) :=
by
  sorry

end number_of_three_digit_numbers_with_123_exactly_once_l8_8050


namespace find_s5_l8_8069

noncomputable def s (a b x y : ℝ) (n : ℕ) : ℝ :=
if n = 1 then (a * x + b * y) else
if n = 2 then (a * x^2 + b * y^2) else
if n = 3 then (a * x^3 + b * y^3) else
if n = 4 then (a * x^4 + b * y^4) else
if n = 5 then (a * x^5 + b * y^5) else 0

theorem find_s5 
  (a b x y : ℝ) :
  s a b x y 1 = 5 →
  s a b x y 2 = 11 →
  s a b x y 3 = 24 →
  s a b x y 4 = 58 →
  s a b x y 5 = 262.88 :=
by
  intros h1 h2 h3 h4
  sorry

end find_s5_l8_8069


namespace premium_rate_l8_8916

theorem premium_rate (P : ℝ) : (14400 / (100 + P)) * 5 = 600 → P = 20 :=
by
  intro h
  sorry

end premium_rate_l8_8916


namespace multiplication_72519_9999_l8_8677

theorem multiplication_72519_9999 :
  72519 * 9999 = 725117481 :=
by
  sorry

end multiplication_72519_9999_l8_8677


namespace find_f_7_5_l8_8043

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom f_range_0_1 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_7_5 : f 7.5 = -0.5 :=
by
  -- The proof goes here
  sorry

end find_f_7_5_l8_8043


namespace circle_area_l8_8840

theorem circle_area (θ : ℝ) : 
  let r := 3 * Real.cos θ - 4 * Real.sin θ in
  let x := r * Real.cos θ in
  let y := r * Real.sin θ in
  let center := (3 / 2 : ℝ, -2 : ℝ) in
  let radius := 5 / 2 in
  (x - center.1)^2 + (y - center.2)^2 = radius^2 →
  ∃ A : ℝ, A = (radius^2) * Real.pi ∧ A = 25 * Real.pi / 4 := 
by sorry

end circle_area_l8_8840


namespace complex_number_identity_l8_8395

theorem complex_number_identity (i : ℂ) (hi : i^2 = -1) : i * (1 + i) = -1 + i :=
by
  sorry

end complex_number_identity_l8_8395


namespace minimum_number_of_different_numbers_l8_8245

theorem minimum_number_of_different_numbers (total_numbers : ℕ) (frequent_count : ℕ) (frequent_occurrences : ℕ) (less_frequent_occurrences : ℕ) (h1 : total_numbers = 2019) (h2 : frequent_count = 10) (h3 : less_frequent_occurrences = 9) : ∃ k : ℕ, k ≥ 225 :=
by {
  sorry
}

end minimum_number_of_different_numbers_l8_8245


namespace smallest_value_N_l8_8434

theorem smallest_value_N (l m n N : ℕ) (h1 : (l - 1) * (m - 1) * (n - 1) = 143) (h2 : N = l * m * n) :
  N = 336 :=
sorry

end smallest_value_N_l8_8434


namespace arithmetic_geometric_means_l8_8221

theorem arithmetic_geometric_means (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 125) : x^2 + y^2 = 1350 :=
by sorry

end arithmetic_geometric_means_l8_8221


namespace abs_neg_two_thirds_l8_8420

-- Conditions: definition of absolute value function
def abs (x : ℚ) : ℚ := if x < 0 then -x else x

-- Main theorem statement: question == answer
theorem abs_neg_two_thirds : abs (-2/3) = 2/3 :=
  by sorry

end abs_neg_two_thirds_l8_8420


namespace find_common_difference_l8_8038

variable {a : ℕ → ℝ} -- Define the sequence as a function from natural numbers to real numbers
variable {d : ℝ} -- Define the common difference as a real number

-- Sequence is arithmetic means there exists a common difference such that a_{n+1} = a_n + d
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Conditions from the problem
variable (h1 : a 3 = 5)
variable (h2 : a 15 = 41)
variable (h3 : is_arithmetic_sequence a d)

-- Theorem statement
theorem find_common_difference : d = 3 :=
by
  sorry

end find_common_difference_l8_8038


namespace dice_probability_l8_8542

def total_outcomes : ℕ := 6 * 6 * 6

def favorable_outcomes : ℕ :=
  -- The number of ways in which one die shows a value and two dice show double that value
  (1 + 3 + 5 + 3 + 1 + 1) * 3

def probability : ℚ :=
  favorable_outcomes / total_outcomes

theorem dice_probability : probability = 7 / 36 :=
by
  -- skipping proof
  sorry

end dice_probability_l8_8542


namespace find_cos_minus_sin_l8_8464

variable (θ : ℝ)
variable (h1 : θ ∈ Set.Ioo (3 * Real.pi / 4) Real.pi)
variable (h2 : Real.sin θ * Real.cos θ = -Real.sqrt 3 / 2)

theorem find_cos_minus_sin : Real.cos θ - Real.sin θ = -Real.sqrt (1 + Real.sqrt 3) := by
  sorry

end find_cos_minus_sin_l8_8464


namespace expand_array_l8_8619

theorem expand_array (n : ℕ) (h₁ : n ≥ 3) 
  (matrix : Fin (n-2) → Fin n → Fin n)
  (h₂ : ∀ i : Fin (n-2), ∀ j: Fin n, ∀ k: Fin n, j ≠ k → matrix i j ≠ matrix i k)
  (h₃ : ∀ j : Fin n, ∀ k: Fin (n-2), ∀ l: Fin (n-2), k ≠ l → matrix k j ≠ matrix l j) :
  ∃ (expanded_matrix : Fin n → Fin n → Fin n), 
    (∀ i : Fin n, ∀ j: Fin n, ∀ k: Fin n, j ≠ k → expanded_matrix i j ≠ expanded_matrix i k) ∧
    (∀ j : Fin n, ∀ k: Fin n, ∀ l: Fin n, k ≠ l → expanded_matrix k j ≠ expanded_matrix l j) :=
sorry

end expand_array_l8_8619


namespace book_pages_total_l8_8934

theorem book_pages_total
  (pages_read_first_day : ℚ) (total_pages : ℚ) (pages_read_second_day : ℚ)
  (rem_read_ratio : ℚ) (read_ratio_mult : ℚ)
  (book_ratio: ℚ) (read_pages_ratio: ℚ)
  (read_second_day_ratio: ℚ):
  pages_read_first_day = 1 / 6 →
  pages_read_second_day = 42 →
  rem_read_ratio = 3 →
  read_ratio_mult = (2 / 6) →
  book_ratio = 3 / 5 →
  read_pages_ratio = 2 / 5 →
  read_second_day_ratio = (2 / 5 - 1 / 6) →
  total_pages = pages_read_second_day / read_second_day_ratio  →
  total_pages = 126 :=
by sorry

end book_pages_total_l8_8934


namespace curves_equiv_and_intersection_l8_8806

theorem curves_equiv_and_intersection (C1 C2 : ℝ → ℝ × ℝ) (ρ θ : ℝ) 
  (hC1 : ∀ t, C1 t = (-t, -1 + √3 * t))
  (hC2 : ∀ θ, (ρ, θ) = (2 * sin θ - 2 * √3 * cos θ, θ))
  (A B : ℝ × ℝ) :
  (∀ x y, y = -1 - √3 * x ↔ (x, y) ∈ set.range (C1)) ∧ 
  (∀ x y, x^2 + 2 * √3 * x + y^2 - 2 * y = 0 ↔ (x, y) ∈ set.range (C2)) ∧ 
  dist A B = 2 * sqrt (4 - (1 / 2)^2) := 
sorry

end curves_equiv_and_intersection_l8_8806


namespace isosceles_triangles_count_l8_8480

theorem isosceles_triangles_count :
  ∃! n, n = 6 ∧
  (∀ (a b : ℕ), 2 * a + b = 25 → 2 * a > b ∧ b > 0 →
  (a = 7 ∧ b = 11) ∨
  (a = 8 ∧ b = 9) ∨
  (a = 9 ∧ b = 7) ∨
  (a = 10 ∧ b = 5) ∨
  (a = 11 ∧ b = 3) ∨
  (a = 12 ∧ b = 1)) :=
begin
  sorry
end

end isosceles_triangles_count_l8_8480


namespace train_passing_through_tunnel_l8_8699

theorem train_passing_through_tunnel :
  let train_length : ℝ := 300
  let tunnel_length : ℝ := 1200
  let speed_in_kmh : ℝ := 54
  let speed_in_mps : ℝ := speed_in_kmh * (1000 / 3600)
  let total_distance : ℝ := train_length + tunnel_length
  let time : ℝ := total_distance / speed_in_mps
  time = 100 :=
by
  sorry

end train_passing_through_tunnel_l8_8699


namespace problem1_solution_problem2_solution_l8_8456

noncomputable def problem1 (α : ℝ) (h : Real.tan α = -2) : Real :=
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α)

theorem problem1_solution (α : ℝ) (h : Real.tan α = -2) : problem1 α h = 5 := by
  sorry

noncomputable def problem2 (α : ℝ) (h : Real.tan α = -2) : Real :=
  1 / (Real.sin α * Real.cos α)

theorem problem2_solution (α : ℝ) (h : Real.tan α = -2) : problem2 α h = -5 / 2 := by
  sorry

end problem1_solution_problem2_solution_l8_8456


namespace part_a_part_b_l8_8386

variables {ABC : Type} [triangle ABC]
variables {A1 B1 C1 : ABC → ABC → ABC} -- Points on the sides BC, CA, and AB respectively
variables {area : ABC → ℝ} -- Function that returns the area of a triangle

theorem part_a :
  ∃ (AB1C1 A1BC1 A1B1C : ABC),
  area AB1C1 ≤ (area ABC) / 4 ∨ area A1BC1 ≤ (area ABC) / 4 ∨ area A1B1C ≤ (area ABC) / 4 :=
sorry

theorem part_b :
  ∃ (AB1C1 A1BC1 A1B1C : ABC),
  area AB1C1 ≤ area A1B1C1 ∨ area A1BC1 ≤ area A1B1C1 ∨ area A1B1C ≤ area A1B1C1 :=
sorry

end part_a_part_b_l8_8386


namespace retailer_received_extra_boxes_l8_8384
-- Necessary import for mathematical proofs

-- Define the conditions
def dozen_boxes := 12
def dozens_ordered := 3
def discount_percent := 25

-- Calculate the total boxes ordered and the discount factor
def total_boxes := dozen_boxes * dozens_ordered
def discount_factor := (100 - discount_percent) / 100

-- Define the number of boxes paid for and the extra boxes received
def paid_boxes := total_boxes * discount_factor
def extra_boxes := total_boxes - paid_boxes

-- Statement of the proof problem
theorem retailer_received_extra_boxes : extra_boxes = 9 :=
by
    -- This is the place where the proof would be written
    sorry

end retailer_received_extra_boxes_l8_8384


namespace factorial_division_l8_8665

-- Define factorial using the standard library's factorial function
def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- The problem statement
theorem factorial_division :
  (factorial 15) / (factorial 6 * factorial 9) = 834 :=
sorry

end factorial_division_l8_8665


namespace calculation_correct_l8_8724

def grid_coloring_probability : ℚ := 591 / 1024

theorem calculation_correct : (m + n = 1615) ↔ (∃ m n : ℕ, m + n = 1615 ∧ gcd m n = 1 ∧ grid_coloring_probability = m / n) := sorry

end calculation_correct_l8_8724


namespace correct_calculation_l8_8121

variable {a : ℝ} (ha : a ≠ 0)

theorem correct_calculation (a : ℝ) (ha : a ≠ 0) : (a^2 * a^3 = a^5) :=
by sorry

end correct_calculation_l8_8121


namespace inequality_solution_l8_8857

theorem inequality_solution (x : ℝ) :
  (x * (x + 2) > x * (3 - x) + 1) ↔ (x < -1/2 ∨ x > 1) :=
by sorry

end inequality_solution_l8_8857


namespace remainder_problem_l8_8267

theorem remainder_problem
  (x : ℕ) (hx : x > 0) (h : 100 % x = 4) : 196 % x = 4 :=
by
  sorry

end remainder_problem_l8_8267


namespace books_remaining_in_special_collection_l8_8285

theorem books_remaining_in_special_collection
  (initial_books : ℕ)
  (loaned_books : ℕ)
  (returned_percentage : ℕ)
  (initial_books_eq : initial_books = 75)
  (loaned_books_eq : loaned_books = 45)
  (returned_percentage_eq : returned_percentage = 80) :
  ∃ final_books : ℕ, final_books = initial_books - (loaned_books - (loaned_books * returned_percentage / 100)) ∧ final_books = 66 :=
by
  sorry

end books_remaining_in_special_collection_l8_8285


namespace rental_property_key_count_l8_8546

def number_of_keys (complexes apartments_per_complex keys_per_lock locks_per_apartment : ℕ) : ℕ :=
  complexes * apartments_per_complex * keys_per_lock * locks_per_apartment

theorem rental_property_key_count : 
  number_of_keys 2 12 3 1 = 72 := by
  sorry

end rental_property_key_count_l8_8546


namespace beavers_still_working_l8_8494

theorem beavers_still_working (total_beavers : ℕ) (wood_beavers dam_beavers lodge_beavers : ℕ)
  (wood_swimming dam_swimming lodge_swimming : ℕ) :
  total_beavers = 12 →
  wood_beavers = 5 →
  dam_beavers = 4 →
  lodge_beavers = 3 →
  wood_swimming = 3 →
  dam_swimming = 2 →
  lodge_swimming = 1 →
  (wood_beavers - wood_swimming) + (dam_beavers - dam_swimming) + (lodge_beavers - lodge_swimming) = 6 :=
by
  intros h_total h_wood h_dam h_lodge h_wood_swim h_dam_swim h_lodge_swim
  sorry

end beavers_still_working_l8_8494


namespace arccos_of_cos_periodic_l8_8569

theorem arccos_of_cos_periodic :
  arccos (cos 8) = 8 - 2 * Real.pi :=
by
  sorry

end arccos_of_cos_periodic_l8_8569


namespace distance_between_towns_l8_8158

variables (x y z : ℝ)

theorem distance_between_towns
  (h1 : x / 24 + y / 16 + z / 12 = 2)
  (h2 : x / 12 + y / 16 + z / 24 = 2.25) :
  x + y + z = 34 :=
sorry

end distance_between_towns_l8_8158


namespace scarlet_savings_l8_8392

theorem scarlet_savings :
  ∀ (initial_savings cost_of_earrings cost_of_necklace amount_left : ℕ),
    initial_savings = 80 →
    cost_of_earrings = 23 →
    cost_of_necklace = 48 →
    amount_left = initial_savings - (cost_of_earrings + cost_of_necklace) →
    amount_left = 9 :=
by
  intros initial_savings cost_of_earrings cost_of_necklace amount_left h_is h_earrings h_necklace h_left
  rw [h_is, h_earrings, h_necklace] at h_left
  exact h_left

end scarlet_savings_l8_8392


namespace tangent_line_at_origin_range_of_a_if_f_has_exactly_one_zero_in_each_interval_l8_8473

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

-- Part 1: Prove that when a = 1, the equation of the tangent line at (0, f(1, 0)) is y = 2x
theorem tangent_line_at_origin (x : ℝ) : 
  let a := 1 in 
  let f' (x : ℝ) := (1 / (1 + x)) + Real.exp (- x) - x * Real.exp (- x) in
  let m := f' 0 in
  let b := f 1 0 in
  m = 2 ∧ b = 0 ∧ (∀ y, y = m * x + b) := 
sorry

-- Part 2: Prove that if f(x) = ln(1+x) + axe^(-x) has exactly one zero in (-1,0) and (0, +∞), 
-- then a ∈ (-∞, -1)
theorem range_of_a_if_f_has_exactly_one_zero_in_each_interval (a : ℝ) :
  (∃! x₁ ∈ Set.Ioo (-1 : ℝ) 0, f a x₁ = 0) ∧ 
  (∃! x₂ ∈ Set.Ioi 0, f a x₂ = 0) → 
  a < -1 :=
sorry

end tangent_line_at_origin_range_of_a_if_f_has_exactly_one_zero_in_each_interval_l8_8473


namespace linear_equation_m_value_l8_8463

theorem linear_equation_m_value (m : ℝ) (x : ℝ) (h : (m - 1) * x ^ |m| - 2 = 0) : m = -1 :=
sorry

end linear_equation_m_value_l8_8463


namespace max_rectangle_area_under_budget_l8_8514

/-- 
Let L and W be the length and width of a rectangle, respectively, where:
1. The length L is made of materials priced at 3 yuan per meter.
2. The width W is made of materials priced at 5 yuan per meter.
3. Both L and W are integers.
4. The total cost 3L + 5W does not exceed 100 yuan.

Prove that the maximum area of the rectangle that can be made under these constraints is 40 square meters.
--/
theorem max_rectangle_area_under_budget :
  ∃ (L W : ℤ), 3 * L + 5 * W ≤ 100 ∧ 0 ≤ L ∧ 0 ≤ W ∧ L * W = 40 :=
sorry

end max_rectangle_area_under_budget_l8_8514


namespace total_pets_remaining_l8_8152

def initial_counts := (7, 6, 4, 5, 3)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)
def morning_sales := (1, 2, 1, 0, 0)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)
def afternoon_sales := (1, 1, 2, 3, 0)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)
def returns := (0, 1, 0, 1, 1)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)

def calculate_remaining (initial_counts morning_sales afternoon_sales returns : Nat × Nat × Nat × Nat × Nat) : Nat :=
  let (p0, k0, r0, g0, c0) := initial_counts
  let (p1, k1, r1, g1, c1) := morning_sales
  let (p2, k2, r2, g2, c2) := afternoon_sales
  let (p3, k3, r3, g3, c3) := returns
  let remaining_puppies := p0 - p1 - p2 + p3
  let remaining_kittens := k0 - k1 - k2 + k3
  let remaining_rabbits := r0 - r1 - r2 + r3
  let remaining_guinea_pigs := g0 - g1 - g2 + g3
  let remaining_chameleons := c0 - c1 - c2 + c3
  remaining_puppies + remaining_kittens + remaining_rabbits + remaining_guinea_pigs + remaining_chameleons

theorem total_pets_remaining : calculate_remaining initial_counts morning_sales afternoon_sales returns = 15 := 
by
  simp [initial_counts, morning_sales, afternoon_sales, returns, calculate_remaining]
  sorry

end total_pets_remaining_l8_8152


namespace sum_of_squares_and_products_l8_8491

theorem sum_of_squares_and_products
  (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z)
  (h4 : x^2 + y^2 + z^2 = 52) (h5 : x * y + y * z + z * x = 24) :
  x + y + z = 10 := 
by
  sorry

end sum_of_squares_and_products_l8_8491


namespace length_of_platform_l8_8907

theorem length_of_platform 
  (speed_train_kmph : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_man : ℝ)
  (conversion_factor : ℝ)
  (speed_train_mps : ℝ)
  (length_train : ℝ)
  (total_distance : ℝ)
  (length_platform : ℝ) :
  speed_train_kmph = 150 →
  time_cross_platform = 45 →
  time_cross_man = 20 →
  conversion_factor = (1000 / 3600) →
  speed_train_mps = speed_train_kmph * conversion_factor →
  length_train = speed_train_mps * time_cross_man →
  total_distance = speed_train_mps * time_cross_platform →
  length_platform = total_distance - length_train →
  length_platform = 1041.75 :=
by sorry

end length_of_platform_l8_8907


namespace number_of_bananas_in_bowl_l8_8641

theorem number_of_bananas_in_bowl (A P B : Nat) (h1 : P = A + 2) (h2 : B = P + 3) (h3 : A + P + B = 19) : B = 9 :=
sorry

end number_of_bananas_in_bowl_l8_8641


namespace find_plot_width_l8_8135

theorem find_plot_width:
  let length : ℝ := 360
  let area_acres : ℝ := 10
  let square_feet_per_acre : ℝ := 43560
  let area_square_feet := area_acres * square_feet_per_acre
  let width := area_square_feet / length
  area_square_feet = 435600 ∧ length = 360 ∧ square_feet_per_acre = 43560
  → width = 1210 :=
by
  intro h
  sorry

end find_plot_width_l8_8135


namespace range_of_k_l8_8192

theorem range_of_k (x : ℝ) (k : ℝ) (h_pos : x > 0) (h_ne : x ≠ 2) :
  (1 / (x - 2) + 3 = (3 - k) / (2 - x)) ↔ (k > -2 ∧ k ≠ 4) :=
by
  sorry

end range_of_k_l8_8192


namespace largest_multiple_of_7_less_than_100_l8_8410

theorem largest_multiple_of_7_less_than_100 : ∃ (n : ℕ), n * 7 < 100 ∧ ∀ (m : ℕ), m * 7 < 100 → m * 7 ≤ n * 7 :=
  by
  sorry

end largest_multiple_of_7_less_than_100_l8_8410


namespace average_weight_of_cats_is_12_l8_8062

noncomputable def cat1 := 12
noncomputable def cat2 := 12
noncomputable def cat3 := 14.7
noncomputable def cat4 := 9.3
def total_weight := cat1 + cat2 + cat3 + cat4
def number_of_cats := 4
def average_weight := total_weight / number_of_cats

theorem average_weight_of_cats_is_12 :
  average_weight = 12 := 
sorry

end average_weight_of_cats_is_12_l8_8062


namespace container_marbles_volume_l8_8913

theorem container_marbles_volume {V₁ V₂ m₁ m₂ : ℕ} 
  (h₁ : V₁ = 24) (h₂ : m₁ = 75) (h₃ : V₂ = 72) :
  m₂ = 225 :=
by
  have proportion := (m₁ : ℚ) / V₁
  have proportion2 := (m₂ : ℚ) / V₂
  have h4 := proportion = proportion2
  sorry

end container_marbles_volume_l8_8913


namespace probability_A_more_heads_than_B_l8_8090

theorem probability_A_more_heads_than_B (n : ℕ) :
  let A_flips_heads := λ (m : ℕ), m > n / 2,
      B_flips_heads := λ (k : ℕ), k > (n - 1) / 2 in
  let event_A := ∃ m, A_flips_heads m,
      event_B := ∃ k, B_flips_heads k in
  probability event_A = 0.5 :=
sorry

end probability_A_more_heads_than_B_l8_8090


namespace find_m_value_l8_8184

def f (x m : ℝ) : ℝ := x^2 - 2*x + m

theorem find_m_value :
  ∃ m : ℝ, (∀ x ∈ (Set.Icc 0 3), f x m ≤ 1) ∧ (∃ x ∈ (Set.Icc 0 3), f x m = 1) ↔ m = -2 :=
by
  sorry

end find_m_value_l8_8184


namespace g_analytical_expression_g_minimum_value_l8_8592

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 1
noncomputable def M (a : ℝ) : ℝ := if (a ≥ 1/3 ∧ a ≤ 1/2) then f a 1 else f a 3
noncomputable def N (a : ℝ) : ℝ := f a (1/a)
noncomputable def g (a : ℝ) : ℝ :=
  if a ≥ 1/3 ∧ a ≤ 1/2 then M a - N a 
  else if a > 1/2 ∧ a ≤ 1 then M a - N a
  else 0 -- outside the given interval, by definition may be kept as 0

theorem g_analytical_expression (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) : 
  g a = if (1/3 ≤ a ∧ a ≤ 1/2) then a + 1/a - 2 else 9 * a + 1/a - 6 := 
sorry

theorem g_minimum_value (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) :
  ∃ (a' : ℝ), 1/3 ≤ a' ∧ a' ≤ 1 ∧ (∀ a, 1/3 ≤ a ∧ a ≤ 1 → g a ≥ g a') ∧ g a' = 1/2 := 
sorry

end g_analytical_expression_g_minimum_value_l8_8592


namespace value_of_7th_term_l8_8691

noncomputable def arithmetic_sequence_a1_d_n (a1 d n a7 : ℝ) : Prop := 
  ((5 * a1 + 10 * d = 68) ∧ 
   (5 * (a1 + (n - 1) * d) - 10 * d = 292) ∧
   (n / 2 * (2 * a1 + (n - 1) * d) = 234) ∧ 
   (a1 + 6 * d = a7))

theorem value_of_7th_term (a1 d n a7 : ℝ) : 
  arithmetic_sequence_a1_d_n a1 d n 18 := 
by
  simp [arithmetic_sequence_a1_d_n]
  sorry

end value_of_7th_term_l8_8691


namespace smallest_square_condition_l8_8321

-- Definition of the conditions
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_last_digit_not_zero (n : ℕ) : Prop := n % 10 ≠ 0

def remove_last_two_digits (n : ℕ) : ℕ :=
  n / 100

-- The statement of the theorem we need to prove
theorem smallest_square_condition : 
  ∃ n : ℕ, is_square n ∧ has_last_digit_not_zero n ∧ is_square (remove_last_two_digits n) ∧ 121 ≤ n :=
sorry

end smallest_square_condition_l8_8321


namespace marie_ends_with_755_l8_8215

def erasers_end (initial lost packs erasers_per_pack : ℕ) : ℕ :=
  initial - lost + packs * erasers_per_pack

theorem marie_ends_with_755 :
  erasers_end 950 420 3 75 = 755 :=
by
  sorry

end marie_ends_with_755_l8_8215


namespace hyperbola_equation_l8_8319

theorem hyperbola_equation {x y : ℝ} (h1 : x ^ 2 / 2 - y ^ 2 = 1) 
  (h2 : x = -2) (h3 : y = 2) : y ^ 2 / 2 - x ^ 2 / 4 = 1 :=
by sorry

end hyperbola_equation_l8_8319


namespace average_hit_targets_value_average_hit_targets_ge_half_l8_8717

noncomputable def average_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - 1 / n)^n)

theorem average_hit_targets_value (n : ℕ) :
  average_hit_targets n = n * (1 - (1 - 1 / n)^n) :=
by sorry

theorem average_hit_targets_ge_half (n : ℕ) :
  average_hit_targets n >= n / 2 :=
by sorry

end average_hit_targets_value_average_hit_targets_ge_half_l8_8717


namespace intersection_A_B_l8_8500

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℕ := {2, 3, 4, 5}

theorem intersection_A_B : A ∩ B = {2, 3} := 
by 
  sorry

end intersection_A_B_l8_8500


namespace divisibility_of_2b_by_a_l8_8949

theorem divisibility_of_2b_by_a (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b)
  (h_cond : ∃ᶠ m in at_top, ∃ᶠ n in at_top, (∃ k₁ : ℕ, m^2 + a * n + b = k₁^2) ∧ (∃ k₂ : ℕ, n^2 + a * m + b = k₂^2)) :
  a ∣ 2 * b :=
sorry

end divisibility_of_2b_by_a_l8_8949


namespace subtraction_verification_l8_8155

theorem subtraction_verification : 888888888888 - 111111111111 = 777777777777 :=
by
  sorry

end subtraction_verification_l8_8155


namespace find_number_l8_8891

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 15) : x = 7.5 :=
sorry

end find_number_l8_8891


namespace virginia_taught_fewer_years_l8_8118

variable (V A : ℕ)

theorem virginia_taught_fewer_years (h1 : V + A + 40 = 93) (h2 : V = A + 9) : 40 - V = 9 := by
  sorry

end virginia_taught_fewer_years_l8_8118


namespace container_marbles_proportional_l8_8915

theorem container_marbles_proportional (V1 V2 : ℕ) (M1 M2 : ℕ)
(h1 : V1 = 24) (h2 : M1 = 75) (h3 : V2 = 72) (h4 : V1 * M2 = V2 * M1) :
  M2 = 225 :=
by {
  -- Given conditions
  sorry
}

end container_marbles_proportional_l8_8915


namespace problem_statements_correct_l8_8673

theorem problem_statements_correct :
    (∀ (select : ℕ) (male female : ℕ), male = 4 → female = 3 → 
      (select = (4 * 3 + 3)) → select ≥ 12 = false) ∧
    (∀ (a1 a2 a3 : ℕ), 
      a2 = 0 ∨ a2 = 1 ∨ a2 = 2 →
      (∃ (cases : ℕ), cases = 14) →
      cases = 14) ∧
    (∀ (ways enter exit : ℕ), enter = 4 → exit = 4 - 1 →
      (ways = enter * exit) → ways = 12 = false) ∧
    (∀ (a b : ℕ),
      a > 0 ∧ a < 10 ∧ b > 0 ∧ b < 10 →
      (∃ (log_val : ℕ), log_val = 54) →
      log_val = 54) := by
  admit

end problem_statements_correct_l8_8673


namespace mixed_gender_selection_count_is_correct_l8_8029

/- Define the given constants -/
def num_male_students : ℕ := 5
def num_female_students : ℕ := 3
def total_students : ℕ := num_male_students + num_female_students
def selection_size : ℕ := 3

/- Define the function to compute binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

/- The Lean 4 statement -/
theorem mixed_gender_selection_count_is_correct
  (num_male_students num_female_students total_students selection_size : ℕ)
  (hc1 : num_male_students = 5)
  (hc2 : num_female_students = 3)
  (hc3 : total_students = num_male_students + num_female_students)
  (hc4 : selection_size = 3) :
  binom total_students selection_size 
  - binom num_male_students selection_size
  - binom num_female_students selection_size = 45 := 
  by 
    -- Only the statement is required
    sorry

end mixed_gender_selection_count_is_correct_l8_8029


namespace even_function_a_eq_zero_l8_8355

theorem even_function_a_eq_zero :
  ∀ a, (∀ x, (x + a) * log ((2 * x - 1) / (2 * x + 1)) = (a - x) * log ((1 - 2 * x) / (2 * x + 1)) → a = 0) :=
by
  sorry

end even_function_a_eq_zero_l8_8355


namespace house_orderings_count_l8_8876

theorem house_orderings_count :
  let houses := Finset (List ℕ)
  let orderings := houses.filter (λ l : List ℕ, 
    ∃ (O R B Y G: ℕ), 
      (O < R) ∧
      (B < Y) ∧
      (abs (l.indexOf B - l.indexOf Y) ≠ 1) ∧
      (G < B ∧ G < Y) ∧
      (l.nodup ∧ 
      (l = [O, R, B, Y, G] ∨
      l = [O, B, R, Y, G] ∨
      l = [O, R, G, B, Y] ∨
      l = [O, G, R, B, Y] ∨
      l = [B, O, R, Y, G] ∨
      l = [B, O, G, R, Y]))
  in orderings.card = 6 := sorry

end house_orderings_count_l8_8876


namespace root_quad_eq_sum_l8_8205

theorem root_quad_eq_sum (a b : ℝ) (h1 : a^2 + a - 2022 = 0) (h2 : b^2 + b - 2022 = 0) (h3 : a + b = -1) : a^2 + 2 * a + b = 2021 :=
by sorry

end root_quad_eq_sum_l8_8205


namespace remainder_of_8673_div_7_l8_8706

theorem remainder_of_8673_div_7 : 8673 % 7 = 3 :=
by
  -- outline structure, proof to be inserted
  sorry

end remainder_of_8673_div_7_l8_8706


namespace mixed_oil_rate_l8_8959

def rate_per_litre_mixed_oil (v1 v2 v3 v4 p1 p2 p3 p4 : ℕ) :=
  (v1 * p1 + v2 * p2 + v3 * p3 + v4 * p4) / (v1 + v2 + v3 + v4)

theorem mixed_oil_rate :
  rate_per_litre_mixed_oil 10 5 8 7 50 68 42 62 = 53.67 :=
by
  sorry

end mixed_oil_rate_l8_8959


namespace at_most_two_zero_points_l8_8045

noncomputable def f (x a : ℝ) := x^3 - 12 * x + a

theorem at_most_two_zero_points (a : ℝ) (h : a ≥ 16) : ∃ l u : ℝ, (∀ x : ℝ, f x a = 0 → x < l ∨ l ≤ x ∧ x ≤ u ∨ u < x) := sorry

end at_most_two_zero_points_l8_8045


namespace janele_cats_average_weight_l8_8063

noncomputable def average_weight_cats (w1 w2 w3 w4 : ℝ) : ℝ :=
  (w1 + w2 + w3 + w4) / 4

theorem janele_cats_average_weight :
  average_weight_cats 12 12 14.7 9.3 = 12 :=
by
  sorry

end janele_cats_average_weight_l8_8063


namespace largest_multiple_of_7_less_than_100_l8_8411

theorem largest_multiple_of_7_less_than_100 : ∃ (n : ℕ), n * 7 < 100 ∧ ∀ (m : ℕ), m * 7 < 100 → m * 7 ≤ n * 7 :=
  by
  sorry

end largest_multiple_of_7_less_than_100_l8_8411


namespace Julie_can_print_complete_newspapers_l8_8065

def sheets_in_box_A : ℕ := 4 * 200
def sheets_in_box_B : ℕ := 3 * 350
def total_sheets : ℕ := sheets_in_box_A + sheets_in_box_B

def front_section_sheets : ℕ := 10
def sports_section_sheets : ℕ := 7
def arts_section_sheets : ℕ := 5
def events_section_sheets : ℕ := 3

def sheets_per_newspaper : ℕ := front_section_sheets + sports_section_sheets + arts_section_sheets + events_section_sheets

theorem Julie_can_print_complete_newspapers : total_sheets / sheets_per_newspaper = 74 := by
  sorry

end Julie_can_print_complete_newspapers_l8_8065


namespace find_f_log_log_3_value_l8_8681

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x - b * Real.logb 3 (Real.sqrt (x*x + 1) - x) + 1

theorem find_f_log_log_3_value
  (a b : ℝ)
  (h1 : f a b (Real.log 10 / Real.log 3) = 5) :
  f a b (-Real.log 10 / Real.log 3) = -3 :=
  sorry

end find_f_log_log_3_value_l8_8681


namespace min_inv_sum_l8_8041

noncomputable def minimum_value_condition (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ 1 = 2*a + b

theorem min_inv_sum (a b : ℝ) (h : minimum_value_condition a b) : 
  ∃ a b : ℝ, (1 / a + 1 / b = 3 + 2 * Real.sqrt 2) := 
by 
  have h1 : a > 0 := h.1;
  have h2 : b > 0 := h.2.1;
  have h3 : 1 = 2 * a + b := h.2.2;
  sorry

end min_inv_sum_l8_8041


namespace distinct_paths_in_grid_l8_8570

def number_of_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

theorem distinct_paths_in_grid :
  number_of_paths 7 8 = 6435 :=
by
  sorry

end distinct_paths_in_grid_l8_8570


namespace intersection_of_A_and_B_l8_8498

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
  sorry

end intersection_of_A_and_B_l8_8498


namespace packages_delivered_by_third_butcher_l8_8513

theorem packages_delivered_by_third_butcher 
  (x y z : ℕ) 
  (h1 : x = 10) 
  (h2 : y = 7) 
  (h3 : 4 * x + 4 * y + 4 * z = 100) : 
  z = 8 :=
by { sorry }

end packages_delivered_by_third_butcher_l8_8513


namespace solve_for_y_l8_8945

theorem solve_for_y (x y : ℝ) : 3 * x + 5 * y = 10 → y = 2 - (3 / 5) * x :=
by 
  -- proof steps would be filled here
  sorry

end solve_for_y_l8_8945


namespace circle_area_polar_eq_l8_8837

theorem circle_area_polar_eq (r θ : ℝ) :
    (r = 3 * Real.cos θ - 4 * Real.sin θ) → (π * (5 / 2) ^ 2 = 25 * π / 4) :=
by 
  sorry

end circle_area_polar_eq_l8_8837


namespace quadratic_root_m_value_l8_8745

theorem quadratic_root_m_value (m : ℝ) (x : ℝ) (h : x = 1) (hx : x^2 + m * x + 2 = 0) : m = -3 :=
by
  sorry

end quadratic_root_m_value_l8_8745


namespace perpendicular_lines_iff_a_eq_1_l8_8419

theorem perpendicular_lines_iff_a_eq_1 :
  ∀ a : ℝ, (∀ x y, (y = a * x + 1) → (y = (a - 2) * x - 1) → (a = 1)) ↔ (a = 1) :=
by sorry

end perpendicular_lines_iff_a_eq_1_l8_8419


namespace infinitely_many_a_l8_8168

theorem infinitely_many_a (n : ℕ) : ∃ (a : ℕ), ∃ (k : ℕ), ∀ n : ℕ, n^6 + 3 * (3 * n^4 * k + 9 * n^2 * k^2 + 9 * k^3) = (n^2 + 3 * k)^3 :=
by
  sorry

end infinitely_many_a_l8_8168


namespace final_cost_l8_8804

-- Definitions of initial conditions
def initial_cart_total : ℝ := 54.00
def discounted_item_original_price : ℝ := 20.00
def discount_rate1 : ℝ := 0.20
def coupon_rate : ℝ := 0.10

-- Prove the final cost after applying discounts
theorem final_cost (initial_cart_total discounted_item_original_price discount_rate1 coupon_rate : ℝ) :
  let discounted_price := discounted_item_original_price * (1 - discount_rate1)
  let total_after_first_discount := initial_cart_total - discounted_price
  let final_total := total_after_first_discount * (1 - coupon_rate)
  final_total = 45.00 :=
by 
  sorry

end final_cost_l8_8804


namespace inverse_function_less_than_zero_l8_8230

theorem inverse_function_less_than_zero (x : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = 2^x + 1) (h₂ : ∀ y, f (f⁻¹ y) = y) (h₃ : ∀ y, f⁻¹ (f y) = y) :
  {x | f⁻¹ x < 0} = {x | 1 < x ∧ x < 2} :=
by
  sorry

end inverse_function_less_than_zero_l8_8230


namespace lisa_phone_spending_l8_8213

variable (cost_phone : ℕ) (cost_contract_per_month : ℕ) (case_percentage : ℕ) (headphones_ratio : ℕ)

/-- Given the cost of the phone, the monthly contract cost, 
    the percentage cost of the case, and ratio cost of headphones,
    prove that the total spending in the first year is correct.
-/ 
theorem lisa_phone_spending 
    (h_cost_phone : cost_phone = 1000) 
    (h_cost_contract_per_month : cost_contract_per_month = 200) 
    (h_case_percentage : case_percentage = 20)
    (h_headphones_ratio : headphones_ratio = 2) :
    cost_phone + (cost_phone * case_percentage / 100) + 
    ((cost_phone * case_percentage / 100) / headphones_ratio) + 
    (cost_contract_per_month * 12) = 3700 :=
by
  sorry

end lisa_phone_spending_l8_8213


namespace remove_two_vertices_eliminate_triangles_l8_8775

variables {V : Type} [Fintype V] (G : SimpleGraph V)

noncomputable def has_no_5_clique (G : SimpleGraph V) : Prop :=
  ∀ (s : Finset V), s.card = 5 → ¬ s.pairwise (G.adj)

noncomputable def every_pair_of_triangles_shares_vertex (G : SimpleGraph V) : Prop :=
  ∀ (T₁ T₂ : Finset V), T₁.card = 3 → T₂.card = 3 → G.is_triangle T₁ → G.is_triangle T₂ → (T₁ ∩ T₂).card ≥ 1

theorem remove_two_vertices_eliminate_triangles :
  (has_no_5_clique G) →
  (every_pair_of_triangles_shares_vertex G) →
  ∃ (X : Finset V), X.card = 2 ∧ ∀ (H : SimpleGraph V), (H = G.delete_vertices X) → ∀ (T : Finset V), T.card = 3 → ¬ H.is_triangle T :=
begin
  intros h1 h2,
  sorry
end

end remove_two_vertices_eliminate_triangles_l8_8775


namespace emily_initial_toys_l8_8165

theorem emily_initial_toys : ∃ (initial_toys : ℕ), initial_toys = 3 + 4 :=
by
  existsi 7
  sorry

end emily_initial_toys_l8_8165


namespace total_payment_l8_8083

def cement_bags := 500
def cost_per_bag := 10
def lorries := 20
def tons_per_lorry := 10
def cost_per_ton := 40

theorem total_payment : cement_bags * cost_per_bag + lorries * tons_per_lorry * cost_per_ton = 13000 := by
  sorry

end total_payment_l8_8083


namespace false_implies_not_all_ripe_l8_8095

def all_ripe (basket : Type) [Nonempty basket] (P : basket → Prop) : Prop :=
  ∀ x : basket, P x

theorem false_implies_not_all_ripe
  (basket : Type)
  [Nonempty basket]
  (P : basket → Prop)
  (h : ¬ all_ripe basket P) :
  (∃ x, ¬ P x) ∧ ¬ all_ripe basket P :=
by
  sorry

end false_implies_not_all_ripe_l8_8095


namespace participants_coffee_l8_8869

theorem participants_coffee (n : ℕ) (h_n : n = 14) (h_not_all : 0 < n - 2 * k) (h_not_all2 : n - 2 * k < n)
(h_neighbors : ∀ (i : ℕ), i < n → ∃ (j : ℕ), j < n ∧ j ≠ i ∧ (remaining i → leaving j) ∧ (leaving i → remaining j)) :
  ∃ (k : ℕ), 1 ≤ k ∧ k ≤ 6 ∧ (n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12) :=
by {
  sorry
}

end participants_coffee_l8_8869


namespace carnival_tickets_l8_8704

theorem carnival_tickets (total_tickets friends : ℕ) (equal_share : ℕ)
  (h1 : friends = 6)
  (h2 : total_tickets = 234)
  (h3 : total_tickets % friends = 0)
  (h4 : equal_share = total_tickets / friends) : 
  equal_share = 39 := 
by
  sorry

end carnival_tickets_l8_8704


namespace find_number_l8_8683

theorem find_number (x : ℝ) (h : 50 + 5 * 12 / (x / 3) = 51) : x = 180 := 
by 
  sorry

end find_number_l8_8683


namespace proportion_of_face_cards_l8_8164

theorem proportion_of_face_cards (p : ℝ) (h : 1 - (1 - p)^3 = 19 / 27) : p = 1 / 3 :=
sorry

end proportion_of_face_cards_l8_8164


namespace common_ratio_is_half_l8_8198

variable {a₁ q : ℝ}

-- Given the conditions of the geometric sequence

-- First condition
axiom h1 : a₁ + a₁ * q ^ 2 = 10

-- Second condition
axiom h2 : a₁ * q ^ 3 + a₁ * q ^ 5 = 5 / 4

-- Proving that the common ratio q is 1/2
theorem common_ratio_is_half : q = 1 / 2 :=
by
  -- The proof details will be filled in here.
  sorry

end common_ratio_is_half_l8_8198


namespace minimum_cost_for_18_oranges_l8_8560

noncomputable def min_cost_oranges (x y : ℕ) : ℕ :=
  10 * x + 30 * y

theorem minimum_cost_for_18_oranges :
  (∃ x y : ℕ, 3 * x + 7 * y = 18 ∧ min_cost_oranges x y = 60) ∧ (60 / 18 = 10 / 3) :=
sorry

end minimum_cost_for_18_oranges_l8_8560


namespace sqrt_six_ineq_l8_8260

theorem sqrt_six_ineq : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 := by
  sorry

end sqrt_six_ineq_l8_8260


namespace M_Mobile_cheaper_than_T_Mobile_l8_8522

def T_Mobile_total_cost (lines : ℕ) : ℕ :=
  if lines <= 2 then 50
  else 50 + (lines - 2) * 16

def M_Mobile_total_cost (lines : ℕ) : ℕ :=
  if lines <= 2 then 45
  else 45 + (lines - 2) * 14

theorem M_Mobile_cheaper_than_T_Mobile : 
  T_Mobile_total_cost 5 - M_Mobile_total_cost 5 = 11 :=
by
  sorry

end M_Mobile_cheaper_than_T_Mobile_l8_8522


namespace work_problem_l8_8529

-- Definition of the conditions and the problem statement
theorem work_problem (P D : ℕ)
  (h1 : ∀ (P : ℕ), ∀ (D : ℕ), (2 * P) * 6 = P * D * 1 / 2) : 
  D = 24 :=
by
  sorry

end work_problem_l8_8529


namespace calculate_expression_l8_8947

theorem calculate_expression (a b c : ℕ) (h1 : a = 2011) (h2 : b = 2012) (h3 : c = 2013) :
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  sorry

end calculate_expression_l8_8947


namespace graph_not_passing_through_origin_l8_8764

theorem graph_not_passing_through_origin (m : ℝ) (h : 3 * m^2 - 2 * m ≠ 0) : m = -(1 / 3) :=
sorry

end graph_not_passing_through_origin_l8_8764


namespace trigonometric_identity_l8_8039

variable (α β : Real) 

theorem trigonometric_identity (h₁ : Real.tan (α + β) = 1) 
                              (h₂ : Real.tan (α - β) = 2) 
                              : (Real.sin (2 * α)) / (Real.cos (2 * β)) = 1 := 
by 
  sorry

end trigonometric_identity_l8_8039


namespace acute_triangle_isosceles_base_angles_l8_8802

theorem acute_triangle_isosceles_base_angles {A B C A1 B1 F : Point} 
  (hABC : acute_triangle A B C)
  (hA1 : perpendicular A1 A C B) 
  (hB1 : perpendicular B1 B C A)
  (hF : midpoint F A B) :
  let γ := angle A C B in 
  isosceles_triangle F A1 B1 ∧ 
  angle F A1 B1 = γ := 
sorry

end acute_triangle_isosceles_base_angles_l8_8802


namespace tan_add_pi_over_4_l8_8331

theorem tan_add_pi_over_4 (α : ℝ) (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 := 
by 
  sorry

end tan_add_pi_over_4_l8_8331


namespace count_dracula_is_alive_l8_8226

variable (P Q : Prop)
variable (h1 : P)          -- I am human
variable (h2 : P → Q)      -- If I am human, then Count Dracula is alive

theorem count_dracula_is_alive : Q :=
by
  sorry

end count_dracula_is_alive_l8_8226


namespace expected_value_winnings_l8_8695

def probability_heads : ℚ := 2 / 5
def probability_tails : ℚ := 3 / 5
def win_amount_heads : ℚ := 5
def lose_amount_tails : ℚ := -4

theorem expected_value_winnings : 
  probability_heads * win_amount_heads + probability_tails * lose_amount_tails = -2 / 5 := 
by 
  sorry

end expected_value_winnings_l8_8695


namespace max_winners_at_least_three_matches_l8_8150

theorem max_winners_at_least_three_matches (n : ℕ) (h : n = 200) :
  (∃ k : ℕ, k ≤ n ∧ ∀ m : ℕ, ((m ≥ 3) → ∃ x : ℕ, x = k → k = 66)) := 
sorry

end max_winners_at_least_three_matches_l8_8150


namespace satisfaction_independence_distribution_and_expectation_l8_8562

-- Conditions
def total_people : ℕ := 200
def both_satisfied : ℕ := 50
def dedication_satisfied : ℕ := 80
def management_satisfied : ℕ := 90
def alpha_value : ℝ := 0.01
def chi_squared_stat : ℝ := (200 * (50 * 80 - 30 * 40)^2) / (80 * 120 * 90 * 110)
def critical_value : ℝ := 6.635
def p_X_0 : ℝ := 27/64
def p_X_1 : ℝ := 27/64
def p_X_2 : ℝ := 9/64
def p_X_3 : ℝ := 1/64
def expected_value_X : ℝ := 3/4

-- Statement for part 1: Testing for Independence
theorem satisfaction_independence : chi_squared_stat > critical_value := by
  sorry

-- Statement for part 2: Distribution table and Mathematical Expectation
theorem distribution_and_expectation :
  ∀ X, (X = 0 → P(X) = p_X_0) ∧
       (X = 1 → P(X) = p_X_1) ∧
       (X = 2 → P(X) = p_X_2) ∧
       (X = 3 → P(X) = p_X_3) ∧
       E(X) = expected_value_X :=
by
  sorry

end satisfaction_independence_distribution_and_expectation_l8_8562


namespace find_p_l8_8264

/-- Given conditions about the coordinates of points on a line, we want to prove p = 3. -/
theorem find_p (m n p : ℝ) 
  (h1 : m = n / 3 - 2 / 5)
  (h2 : m + p = (n + 9) / 3 - 2 / 5) 
  : p = 3 := by 
  sorry

end find_p_l8_8264


namespace tan_add_pi_over_3_l8_8958

variable (y : ℝ)

theorem tan_add_pi_over_3 (h : Real.tan y = 3) : 
  Real.tan (y + π / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) := 
by
  sorry

end tan_add_pi_over_3_l8_8958


namespace determine_k_for_circle_l8_8714

theorem determine_k_for_circle (x y k : ℝ) (h : x^2 + 14*x + y^2 + 8*y - k = 0) (r : ℝ) :
  r = 5 → k = 40 :=
by
  intros radius_eq_five
  sorry

end determine_k_for_circle_l8_8714


namespace evaluate_expression_l8_8310

theorem evaluate_expression (x : ℕ) (h : x = 3) : x + x^2 * (x^(x^2)) = 177150 := by
  sorry

end evaluate_expression_l8_8310


namespace cost_per_bag_l8_8726

theorem cost_per_bag (C : ℝ)
  (total_bags : ℕ := 20)
  (price_per_bag_original : ℝ := 6)
  (sold_original : ℕ := 15)
  (price_per_bag_discounted : ℝ := 4)
  (sold_discounted : ℕ := 5)
  (net_profit : ℝ := 50) :
  sold_original * price_per_bag_original + sold_discounted * price_per_bag_discounted - net_profit = total_bags * C →
  C = 3 :=
by
  intros h
  sorry

end cost_per_bag_l8_8726


namespace compare_neg_rationals_l8_8303

-- Definition and conditions
def abs_neg_one_third : ℚ := |(-1 / 3 : ℚ)|
def abs_neg_one_fourth : ℚ := |(-1 / 4 : ℚ)|

-- Problem statement
theorem compare_neg_rationals : (-1 : ℚ) / 3 < -1 / 4 :=
by
  -- Including the conditions here, even though they are straightforward implications in Lean
  have h1 : abs_neg_one_third = 1 / 3 := abs_neg_one_third
  have h2 : abs_neg_one_fourth = 1 / 4 := abs_neg_one_fourth
  -- We would include steps to show that -1 / 3 < -1 / 4 using the above facts
  sorry

end compare_neg_rationals_l8_8303


namespace inequality_proof_l8_8449

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  (x / (y + z)) + (y / (z + x)) + (z / (x + y)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l8_8449


namespace even_function_a_zero_l8_8350

section

variable (a : ℝ)

def f (x : ℝ) := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero : ∀ x : ℝ, f a x = f a (-x) → a = 0 := by
  sorry

end

end even_function_a_zero_l8_8350


namespace factorize_square_diff_factorize_common_factor_l8_8021

-- Problem 1: Difference of squares
theorem factorize_square_diff (x : ℝ) : 4 * x^2 - 9 = (2 * x + 3) * (2 * x - 3) := 
by
  sorry

-- Problem 2: Factoring out common terms
theorem factorize_common_factor (a b x y : ℝ) (h : y - x = -(x - y)) : 
  2 * a * (x - y) - 3 * b * (y - x) = (x - y) * (2 * a + 3 * b) := 
by
  sorry

end factorize_square_diff_factorize_common_factor_l8_8021


namespace max_area_of_rectangle_l8_8401

-- Question: Prove the largest possible area of a rectangle given the conditions
theorem max_area_of_rectangle :
  ∀ (x : ℝ), (2 * x + 2 * (x + 5) = 60) → x * (x + 5) ≤ 218.75 :=
by
  sorry

end max_area_of_rectangle_l8_8401


namespace intersection_A_B_l8_8501

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℕ := {2, 3, 4, 5}

theorem intersection_A_B : A ∩ B = {2, 3} := 
by 
  sorry

end intersection_A_B_l8_8501


namespace lateral_surface_area_of_cone_l8_8648

theorem lateral_surface_area_of_cone (r l : ℝ) (h₁ : r = 3) (h₂ : l = 5) :
  π * r * l = 15 * π :=
by sorry

end lateral_surface_area_of_cone_l8_8648


namespace circle_area_l8_8848

open Real

def given_equation (r θ : ℝ) : Prop := r = 3 * cos θ - 4 * sin θ

theorem circle_area (r θ : ℝ) (h : given_equation r θ) : 
  ∃ (c : ℝ × ℝ) (R : ℝ), c = (3 / 2, -2) ∧ R = 5 / 2 ∧ π * R^2 = 25 / 4 * π :=
sorry

end circle_area_l8_8848


namespace average_distance_is_600_l8_8791

-- Definitions based on the given conditions
def distance_around_block := 200
def johnny_rounds := 4
def mickey_rounds := johnny_rounds / 2

-- The calculated distances
def johnny_distance := johnny_rounds * distance_around_block
def mickey_distance := mickey_rounds * distance_around_block

-- The average distance computation
def average_distance := (johnny_distance + mickey_distance) / 2

-- The theorem to prove that the average distance is 600 meters
theorem average_distance_is_600 : average_distance = 600 := by sorry

end average_distance_is_600_l8_8791


namespace negation_of_all_honest_l8_8235

-- Define the needed predicates
variable {Man : Type} -- Type for men
variable (man : Man → Prop)
variable (age : Man → ℕ)
variable (honest : Man → Prop)

-- Define the conditions and the statement we want to prove
theorem negation_of_all_honest :
  (∀ x, man x → age x > 30 → honest x) →
  (∃ x, man x ∧ age x > 30 ∧ ¬ honest x) :=
sorry

end negation_of_all_honest_l8_8235


namespace sum_of_first_half_of_numbers_l8_8227

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

end sum_of_first_half_of_numbers_l8_8227


namespace evaluate_expression_l8_8020

theorem evaluate_expression : 3 + (-3)^2 = 12 := by
  sorry

end evaluate_expression_l8_8020


namespace falsity_of_proposition_implies_a_range_l8_8364

theorem falsity_of_proposition_implies_a_range (a : ℝ) : 
  (¬ ∃ x₀ : ℝ, a * Real.sin x₀ + Real.cos x₀ ≥ 2) →
  a ∈ Set.Ioo (-Real.sqrt 3) (Real.sqrt 3) :=
by 
  sorry

end falsity_of_proposition_implies_a_range_l8_8364


namespace integral_2x_ex_l8_8581

open Real IntervalIntegral

theorem integral_2x_ex : ∫ x in 0..1, (2 * x + exp x) = exp 1 := 
begin
  have h : ∀ x, deriv (λ x, x^2 + exp x) x = 2 * x + exp x := by  
  { intro x,
    calc (λ x, x^2 + exp x).deriv x = x.deriv * 1 + (exp x).deriv : 
      by apply funext; intro x; ring_deriv -- Windows addition
                          ... = 2 * x + exp x : by ring },
  rw integral_eq_sub_of_has_deriv_at h (by continuity) (by continuity),
  simp,
  norm_num,
end

end integral_2x_ex_l8_8581


namespace find_A_d_minus_B_d_l8_8812

-- Definitions for the proof problem
variables {d A B : ℕ}
variables {ad bd : ℤ} -- Representing A_d and B_d in ℤ for arithmetic operations

-- Conditions
constants (base_check : d > 7)
constants (digit_check : A < d ∧ B < d)
constants (encoded_check : d^1 * A + d^0 * B + d^2 * A + d^1 * A = 1 * d^2 + 7 * d^1 + 2 * d^0)

-- The theorem to prove
theorem find_A_d_minus_B_d : A - B = 5 :=
good sorry

end find_A_d_minus_B_d_l8_8812


namespace find_x_angle_l8_8551

theorem find_x_angle (x : ℝ) (h : x + x + 140 = 360) : x = 110 :=
by
  sorry

end find_x_angle_l8_8551


namespace prize_difference_l8_8075

def mateo_hourly_rate : ℕ := 20
def sydney_daily_rate : ℕ := 400
def hours_in_a_week : ℕ := 24 * 7
def days_in_a_week : ℕ := 7

def mateo_total : ℕ := mateo_hourly_rate * hours_in_a_week
def sydney_total : ℕ := sydney_daily_rate * days_in_a_week

def difference_amount : ℕ := 560

theorem prize_difference : mateo_total - sydney_total = difference_amount := sorry

end prize_difference_l8_8075


namespace perpendicular_vectors_find_a_l8_8754

theorem perpendicular_vectors_find_a
  (a : ℝ)
  (m : ℝ × ℝ := (1, 2))
  (n : ℝ × ℝ := (a, -1))
  (h : m.1 * n.1 + m.2 * n.2 = 0) :
  a = 2 := 
sorry

end perpendicular_vectors_find_a_l8_8754


namespace tens_digit_19_pow_2023_l8_8576

theorem tens_digit_19_pow_2023 :
  ∃ d : ℕ, d = (59 / 10) % 10 ∧ (19 ^ 2023 % 100) / 10 = d :=
by
  have h1 : 19 ^ 10 % 100 = 1 := by sorry
  have h2 : 19 ↔ 0 := by sorry
  have h4 : 2023 % 10 = 3 := by sorry
  have h5 : 19 ^ 10 ↔ 1 := by sorry
  have h6 : 19 ^ 3 % 100 = 59 := by sorry
  have h7 : (19 ^ 2023 % 100) = 59 := by sorry
  exists 5
  split
  repeat { assumption.dump }

end tens_digit_19_pow_2023_l8_8576


namespace circle_area_l8_8850

open Real

def given_equation (r θ : ℝ) : Prop := r = 3 * cos θ - 4 * sin θ

theorem circle_area (r θ : ℝ) (h : given_equation r θ) : 
  ∃ (c : ℝ × ℝ) (R : ℝ), c = (3 / 2, -2) ∧ R = 5 / 2 ∧ π * R^2 = 25 / 4 * π :=
sorry

end circle_area_l8_8850


namespace cone_volume_l8_8861

-- Define the condition
def cylinder_volume : ℝ := 30

-- Define the statement that needs to be proven
theorem cone_volume (h_cylinder_volume : cylinder_volume = 30) : cylinder_volume / 3 = 10 := 
by 
  -- Proof omitted
  sorry

end cone_volume_l8_8861


namespace smallest_possible_third_term_l8_8918

theorem smallest_possible_third_term :
  ∃ (d : ℝ), (d = -3 + Real.sqrt 134 ∨ d = -3 - Real.sqrt 134) ∧ 
  (7, 7 + d + 3, 7 + 2 * d + 18) = (7, 10 + d, 25 + 2 * d) ∧ 
  min (25 + 2 * (-3 + Real.sqrt 134)) (25 + 2 * (-3 - Real.sqrt 134)) = 19 + 2 * Real.sqrt 134 :=
by
  sorry

end smallest_possible_third_term_l8_8918


namespace find_number_l8_8124

theorem find_number (S Q R N : ℕ) (hS : S = 555 + 445) (hQ : Q = 2 * (555 - 445)) (hR : R = 50) (h_eq : N = S * Q + R) :
  N = 220050 :=
by
  rw [hS, hQ, hR] at h_eq
  norm_num at h_eq
  exact h_eq

end find_number_l8_8124


namespace evaluate_fraction_l8_8166

-- Let's restate the problem in Lean
theorem evaluate_fraction :
  (∃ q, (2024 / 2023 - 2023 / 2024) = 4047 / q) :=
by
  -- Substitute a = 2023
  let a := 2023
  -- Provide the value we expect for q to hold in the reduced fraction.
  use (a * (a + 1)) -- The expected denominator
  -- The proof for the theorem is omitted here
  sorry

end evaluate_fraction_l8_8166


namespace original_profit_percentage_l8_8563

theorem original_profit_percentage {P S : ℝ}
  (h1 : S = 1100)
  (h2 : P ≠ 0)
  (h3 : 1.17 * P = 1170) :
  (S - P) / P * 100 = 10 :=
by
  sorry

end original_profit_percentage_l8_8563


namespace zoe_remaining_pictures_l8_8268

-- Definitions based on the conditions
def total_pictures : Nat := 88
def colored_pictures : Nat := 20

-- Proof statement
theorem zoe_remaining_pictures : total_pictures - colored_pictures = 68 := by
  sorry

end zoe_remaining_pictures_l8_8268


namespace equation_of_line_l8_8645

theorem equation_of_line 
  (a : ℝ) (h : a < 3) 
  (C : ℝ × ℝ) 
  (hC : C = (-2, 3)) 
  (l_intersects_circle : ∃ A B : ℝ × ℝ, 
    (A.1^2 + A.2^2 + 2 * A.1 - 4 * A.2 + a = 0) ∧ 
    (B.1^2 + B.2^2 + 2 * B.1 - 4 * B.2 + a = 0) ∧ 
    (C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))) : 
  ∃ (m b : ℝ), 
    (m = 1) ∧ 
    (b = -5) ∧ 
    (∀ x y, y - 3 = m * (x + 2) ↔ x - y + 5 = 0) :=
by
  sorry

end equation_of_line_l8_8645


namespace remainder_when_dividing_by_y_minus_4_l8_8880

def g (y : ℤ) : ℤ := y^5 - 8 * y^4 + 12 * y^3 + 25 * y^2 - 40 * y + 24

theorem remainder_when_dividing_by_y_minus_4 : g 4 = 8 :=
by
  sorry

end remainder_when_dividing_by_y_minus_4_l8_8880


namespace tasks_to_shower_l8_8080

-- Definitions of the conditions
def tasks_to_clean_house : Nat := 7
def tasks_to_make_dinner : Nat := 4
def minutes_per_task : Nat := 10
def total_minutes : Nat := 2 * 60

-- The theorem we want to prove
theorem tasks_to_shower (x : Nat) :
  total_minutes = (tasks_to_clean_house + tasks_to_make_dinner + x) * minutes_per_task →
  x = 1 := by
  sorry

end tasks_to_shower_l8_8080


namespace circle_area_l8_8841

theorem circle_area (θ : ℝ) : 
  let r := 3 * Real.cos θ - 4 * Real.sin θ in
  let x := r * Real.cos θ in
  let y := r * Real.sin θ in
  let center := (3 / 2 : ℝ, -2 : ℝ) in
  let radius := 5 / 2 in
  (x - center.1)^2 + (y - center.2)^2 = radius^2 →
  ∃ A : ℝ, A = (radius^2) * Real.pi ∧ A = 25 * Real.pi / 4 := 
by sorry

end circle_area_l8_8841


namespace line_through_midpoint_l8_8340

theorem line_through_midpoint (x y : ℝ)
  (ellipse : x^2 / 25 + y^2 / 16 = 1)
  (midpoint : P = (2, 1)) :
  ∃ (A B : ℝ × ℝ),
    A ≠ B ∧
    P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
    (A.1^2 / 25 + A.2^2 / 16 = 1) ∧
    (B.1^2 / 25 + B.2^2 / 16 = 1) ∧
    (x = 32*y - 25*x - 89) :=
sorry

end line_through_midpoint_l8_8340


namespace race_positions_l8_8492

theorem race_positions
  (positions : Fin 15 → String) 
  (h_quinn_lucas : ∃ n : Fin 15, positions n = "Quinn" ∧ positions (n + 4) = "Lucas")
  (h_oliver_quinn : ∃ n : Fin 15, positions (n - 1) = "Oliver" ∧ positions n = "Quinn")
  (h_naomi_oliver : ∃ n : Fin 15, positions n = "Naomi" ∧ positions (n + 3) = "Oliver")
  (h_emma_lucas : ∃ n : Fin 15, positions n = "Lucas" ∧ positions (n + 1) = "Emma")
  (h_sara_naomi : ∃ n : Fin 15, positions n = "Naomi" ∧ positions (n + 1) = "Sara")
  (h_naomi_4th : ∃ n : Fin 15, n = 3 ∧ positions n = "Naomi") :
  positions 6 = "Oliver" :=
by
  sorry

end race_positions_l8_8492


namespace max_winners_3_matches_200_participants_l8_8149

theorem max_winners_3_matches_200_participants (n : ℕ) (h_total : n = 200):
  ∃ (x : ℕ), (∀ y : ℕ, (3 * y ≤ 199) → y ≤ 66) ∧ (3 * 66 ≤ 199) :=
by
  use 66
  split
  · intro y h
    have : (n - 1) = 199 := by linarith
    suffices : 3 * 66 ≤ (n - 1) by linarith
    exact this
    sorry
  · linarith

end max_winners_3_matches_200_participants_l8_8149


namespace complex_number_calculation_l8_8183

theorem complex_number_calculation (z : ℂ) (hz : z = 1 - I) : (z^2 / (z - 1)) = 2 := by
  sorry

end complex_number_calculation_l8_8183


namespace fruit_bowl_l8_8639

variable {A P B : ℕ}

theorem fruit_bowl : (P = A + 2) → (B = P + 3) → (A + P + B = 19) → B = 9 :=
by
  intros h1 h2 h3
  sorry

end fruit_bowl_l8_8639


namespace trig_problem_1_trig_problem_2_l8_8298

noncomputable def trig_expr_1 : ℝ :=
  Real.cos (-11 * Real.pi / 6) + Real.sin (12 * Real.pi / 5) * Real.tan (6 * Real.pi)

noncomputable def trig_expr_2 : ℝ :=
  Real.sin (420 * Real.pi / 180) * Real.cos (750 * Real.pi / 180) +
  Real.sin (-330 * Real.pi / 180) * Real.cos (-660 * Real.pi / 180)

theorem trig_problem_1 : trig_expr_1 = Real.sqrt 3 / 2 :=
by
  sorry

theorem trig_problem_2 : trig_expr_2 = 1 :=
by
  sorry

end trig_problem_1_trig_problem_2_l8_8298


namespace quotient_is_76_l8_8552

def original_number : ℕ := 12401
def divisor : ℕ := 163
def remainder : ℕ := 13

theorem quotient_is_76 : (original_number - remainder) / divisor = 76 :=
by
  sorry

end quotient_is_76_l8_8552


namespace market_value_of_stock_l8_8894

theorem market_value_of_stock (dividend_rate : ℝ) (yield_rate : ℝ) (face_value : ℝ) :
  dividend_rate = 0.12 → yield_rate = 0.08 → face_value = 100 → (dividend_rate * face_value / yield_rate * 100) = 150 :=
by
  intros h1 h2 h3
  sorry

end market_value_of_stock_l8_8894


namespace sum_of_squares_l8_8564

theorem sum_of_squares : 
  let nums := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
  let squares := nums.map (λ x => x * x)
  (squares.sum = 195) := 
by
  let nums := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
  let squares := nums.map (λ x => x * x)
  have h : squares.sum = 195 := sorry
  exact h

end sum_of_squares_l8_8564


namespace intersection_A_B_l8_8503

-- Define sets A and B
def A : Set ℝ := { x | -2 < x ∧ x < 4 }
def B : Set ℝ := { 2, 3, 4, 5 }

-- State the theorem about the intersection A ∩ B
theorem intersection_A_B : A ∩ B = { 2, 3 } :=
by
  sorry

end intersection_A_B_l8_8503


namespace triangle_angle_B_max_sin_A_plus_sin_C_l8_8037

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) (h1 : (a - c) * Real.sin A + c * Real.sin C - b * Real.sin B = 0) 
  (h2 : a / Real.sin A = b / Real.sin B) (h3 : b / Real.sin B = c / Real.sin C) : 
  B = Real.arccos (1/2) := 
sorry

theorem max_sin_A_plus_sin_C (a b c : ℝ) (A B C : ℝ) (h1 : (a - c) * Real.sin A + c * Real.sin C - b * Real.sin B = 0) 
  (h2 : a / Real.sin A = b / Real.sin B) (h3 : b / Real.sin B = c / Real.sin C) 
  (hB : B = Real.arccos (1/2)) : 
  Real.sin A + Real.sin C = Real.sqrt 3 :=
sorry

end triangle_angle_B_max_sin_A_plus_sin_C_l8_8037


namespace outfit_combinations_l8_8634

theorem outfit_combinations (shirts ties hat_choices : ℕ) (h_shirts : shirts = 8) (h_ties : ties = 7) (h_hat_choices : hat_choices = 3) : shirts * ties * hat_choices = 168 := by
  sorry

end outfit_combinations_l8_8634


namespace scarlet_savings_l8_8391

theorem scarlet_savings :
  ∀ (initial_savings cost_of_earrings cost_of_necklace amount_left : ℕ),
    initial_savings = 80 →
    cost_of_earrings = 23 →
    cost_of_necklace = 48 →
    amount_left = initial_savings - (cost_of_earrings + cost_of_necklace) →
    amount_left = 9 :=
by
  intros initial_savings cost_of_earrings cost_of_necklace amount_left h_is h_earrings h_necklace h_left
  rw [h_is, h_earrings, h_necklace] at h_left
  exact h_left

end scarlet_savings_l8_8391


namespace number_of_free_ranging_chickens_is_105_l8_8862

namespace ChickenProblem

-- Conditions as definitions
def coop_chickens : ℕ := 14
def run_chickens : ℕ := 2 * coop_chickens
def free_ranging_chickens : ℕ := 2 * run_chickens - 4
def total_coop_run_chickens : ℕ := coop_chickens + run_chickens

-- The ratio condition
def ratio_condition : Prop :=
  (coop_chickens + run_chickens) * 5 = free_ranging_chickens * 2

-- Proof Statement
theorem number_of_free_ranging_chickens_is_105 :
  free_ranging_chickens = 105 :=
by {
  sorry
}

end ChickenProblem

end number_of_free_ranging_chickens_is_105_l8_8862


namespace calories_per_serving_is_120_l8_8139

-- Define the conditions
def servings : ℕ := 3
def halfCalories : ℕ := 180
def totalCalories : ℕ := 2 * halfCalories

-- Define the target value
def caloriesPerServing : ℕ := totalCalories / servings

-- The proof goal
theorem calories_per_serving_is_120 : caloriesPerServing = 120 :=
by 
  sorry

end calories_per_serving_is_120_l8_8139


namespace probability_YW_correct_l8_8875

noncomputable def probability_YW_greater_than_six_sqrt_three (XY YZ XZ YW : ℝ) : ℝ :=
  if H : XY = 12 ∧ YZ = 6 ∧ XZ = 6 * Real.sqrt 3 then 
    if YW > 6 * Real.sqrt 3 then (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3
    else 0
  else 0

theorem probability_YW_correct : probability_YW_greater_than_six_sqrt_three 12 6 (6 * Real.sqrt 3) (6 * Real.sqrt 3) = (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3 :=
sorry

end probability_YW_correct_l8_8875


namespace domain_of_f_l8_8406

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3)^2 + (x - 6))

theorem domain_of_f :
  ∀ x : ℝ, x ≠ (5 + Real.sqrt 13) / 2 ∧ x ≠ (5 - Real.sqrt 13) / 2 → ∃ y : ℝ, y = f x :=
by
  sorry

end domain_of_f_l8_8406


namespace marble_price_proof_l8_8291

noncomputable def price_per_colored_marble (total_marbles white_percentage black_percentage white_price black_price total_earnings : ℕ) : ℕ :=
  let white_marbles := total_marbles * white_percentage / 100
  let black_marbles := total_marbles * black_percentage / 100
  let colored_marbles := total_marbles - (white_marbles + black_marbles)
  let earnings_from_white := white_marbles * white_price
  let earnings_from_black := black_marbles * black_price
  let earnings_from_colored := total_earnings - (earnings_from_white + earnings_from_black)
  earnings_from_colored / colored_marbles

theorem marble_price_proof : price_per_colored_marble 100 20 30 5 10 1400 = 20 := 
sorry

end marble_price_proof_l8_8291


namespace minimum_value_of_function_l8_8047

theorem minimum_value_of_function :
  ∀ x : ℝ, (x > -2) → (x + (16 / (x + 2)) ≥ 6) :=
by
  intro x hx
  sorry

end minimum_value_of_function_l8_8047


namespace prism_volume_l8_8394

theorem prism_volume 
  (a b c : ℝ)
  (h1 : a * b = 30)
  (h2 : a * c = 50)
  (h3 : b * c = 75) : 
  a * b * c = 150 * Real.sqrt 5 :=
sorry

end prism_volume_l8_8394


namespace tree_height_l8_8280

theorem tree_height (boy_initial_height tree_initial_height boy_final_height boy_growth_rate tree_growth_rate : ℝ) 
  (h1 : boy_initial_height = 24) 
  (h2 : tree_initial_height = 16) 
  (h3 : boy_final_height = 36) 
  (h4 : boy_growth_rate = boy_final_height - boy_initial_height) 
  (h5 : tree_growth_rate = 2 * boy_growth_rate) 
  : tree_initial_height + tree_growth_rate = 40 := 
by
  subst h1 h2 h3 h4 h5;
  sorry

end tree_height_l8_8280


namespace solution1_solution2_l8_8367

open Real

noncomputable def problem1 (a b : ℝ) : Prop :=
a = 2 ∧ b = 2

noncomputable def problem2 (b : ℝ) : Prop :=
b = (2 * (sqrt 3 + sqrt 2)) / 3

theorem solution1 (a b : ℝ) (c : ℝ) (C : ℝ) (area : ℝ)
  (h1 : c = 2)
  (h2 : C = π / 3)
  (h3 : area = sqrt 3)
  (h4 : (1 / 2) * a * b * sin C = area) :
  problem1 a b :=
by sorry

theorem solution2 (a b : ℝ) (c : ℝ) (C : ℝ) (cosA : ℝ)
  (h1 : c = 2)
  (h2 : C = π / 3)
  (h3 : cosA = sqrt 3 / 3)
  (h4 : sin (arccos (sqrt 3 / 3)) = sqrt 6 / 3)
  (h5 : (a / (sqrt 6 / 3)) = (2 / (sqrt 3 / 2)))
  (h6 : ((b / ((3 + sqrt 6) / 6)) = (2 / (sqrt 3 / 2)))) :
  problem2 b :=
by sorry

end solution1_solution2_l8_8367


namespace circle_area_l8_8842

theorem circle_area (θ : ℝ) : 
  let r := 3 * Real.cos θ - 4 * Real.sin θ in
  let x := r * Real.cos θ in
  let y := r * Real.sin θ in
  let center := (3 / 2 : ℝ, -2 : ℝ) in
  let radius := 5 / 2 in
  (x - center.1)^2 + (y - center.2)^2 = radius^2 →
  ∃ A : ℝ, A = (radius^2) * Real.pi ∧ A = 25 * Real.pi / 4 := 
by sorry

end circle_area_l8_8842


namespace green_sweets_count_l8_8247

def total_sweets := 285
def red_sweets := 49
def neither_red_nor_green_sweets := 177

theorem green_sweets_count : 
  (total_sweets - red_sweets - neither_red_nor_green_sweets) = 59 :=
by
  -- The proof will go here
  sorry

end green_sweets_count_l8_8247


namespace minimum_value_of_fraction_plus_variable_l8_8255

theorem minimum_value_of_fraction_plus_variable (a : ℝ) (h : a > 1) : ∃ m, (∀ b, b > 1 → (4 / (b - 1) + b) ≥ m) ∧ m = 5 :=
by
  use 5
  sorry

end minimum_value_of_fraction_plus_variable_l8_8255


namespace coffee_participants_l8_8871

noncomputable def participants_went_for_coffee (total participants : ℕ) (stay : ℕ) : Prop :=
  (0 < stay) ∧ (stay < 14) ∧ (stay % 2 = 0) ∧ ∃ k : ℕ, stay = 2 * k

theorem coffee_participants :
  ∀ (total participants : ℕ), total participants = 14 →
  (∀ participant, ((∃ k : ℕ, stay = 2 * k) → (0 < stay) ∧ (stay < 14))) →
  participants_went_for_coffee total_participants (14 - stay) =
  (stay = 12) ∨ (stay = 10) ∨ (stay = 8) ∨ (stay = 6) :=
by
  sorry

end coffee_participants_l8_8871


namespace no_real_y_for_two_equations_l8_8073

theorem no_real_y_for_two_equations:
  ¬ ∃ (x y : ℝ), x^2 + y^2 = 16 ∧ x^2 + 3 * y + 30 = 0 :=
by
  sorry

end no_real_y_for_two_equations_l8_8073


namespace arithmetic_sequence_general_term_b_sum_formula_l8_8067

theorem arithmetic_sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) :
  (∀ n, a n = a 1 + (n - 1) * d) →
  (a 1 + a 13 = 26) →
  (S 9 = 81) →
  (∀ n, S n = n * (a 1 + a n) / 2) →
  (∀ n, a n = 2 * n - 1) :=
by
sorry

theorem b_sum_formula (b : ℕ → ℝ) (T : ℕ → ℝ) :
  (∀ n, b n = 1 / ((2 * n + 1) * (2 * n + 3))) →
  (∀ n, T n = Σ b i | (i : ℕ), 1 ≤ i ∧ i ≤ n) →
  (∀ n, T n = n / (3 * (2 * n + 3))) :=
by
sorry

end arithmetic_sequence_general_term_b_sum_formula_l8_8067


namespace determine_OP_squared_l8_8686

-- Define the given conditions
variable (O P : Point) -- Points: center O and intersection point P
variable (r : ℝ) (AB CD : ℝ) (E F : Point) -- radius, lengths of chords, midpoints of chords
variable (OE OF : ℝ) -- Distances from center to midpoints of chords
variable (EF : ℝ) -- Distance between midpoints
variable (OP : ℝ) -- Distance from center to intersection point

-- Conditions as given
axiom circle_radius : r = 30
axiom chord_AB_length : AB = 40
axiom chord_CD_length : CD = 14
axiom distance_midpoints : EF = 15
axiom distance_OE : OE = 20
axiom distance_OF : OF = 29

-- The proof problem: determine that OP^2 = 733 given the conditions
theorem determine_OP_squared :
  OP^2 = 733 :=
sorry

end determine_OP_squared_l8_8686


namespace even_function_a_zero_l8_8346

noncomputable def f (x a : ℝ) : ℝ := (x + a) * real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) →
  (2 * x - 1) / (2 * x + 1) > 0 → 
  x > 1 / 2 ∨ x < -1 / 2 →
  a = 0 :=
by {
  sorry
}

end even_function_a_zero_l8_8346


namespace tom_received_20_percent_bonus_l8_8657

-- Define the initial conditions
def tom_spent : ℤ := 250
def gems_per_dollar : ℤ := 100
def total_gems_received : ℤ := 30000

-- Calculate the number of gems received without the bonus
def gems_without_bonus : ℤ := tom_spent * gems_per_dollar
def bonus_gems : ℤ := total_gems_received - gems_without_bonus

-- Calculate the percentage of the bonus
def bonus_percentage : ℚ := (bonus_gems : ℚ) / gems_without_bonus * 100

-- State the theorem
theorem tom_received_20_percent_bonus : bonus_percentage = 20 := by
  sorry

end tom_received_20_percent_bonus_l8_8657


namespace solution_set_inequality_l8_8241

theorem solution_set_inequality (x : ℝ) : 
  (-x^2 + 3 * x - 2 ≥ 0) ↔ (1 ≤ x ∧ x ≤ 2) :=
sorry

end solution_set_inequality_l8_8241


namespace probability_no_adjacent_same_roll_l8_8323

theorem probability_no_adjacent_same_roll :
  let A := 1 -- rolls a six-sided die
  let B := 2 -- rolls a six-sided die
  let C := 3 -- rolls a six-sided die
  let D := 4 -- rolls a six-sided die
  let E := 5 -- rolls a six-sided die
  let people := [A, B, C, D, E]
  -- A and C are required to roll different numbers
  let prob_A_C_diff := 5 / 6
  -- B must roll different from A and C
  let prob_B_diff := 4 / 6
  -- D must roll different from C and A
  let prob_D_diff := 4 / 6
  -- E must roll different from D and A
  let prob_E_diff := 3 / 6
  (prob_A_C_diff * prob_B_diff * prob_D_diff * prob_E_diff) = 10 / 27 :=
by
  sorry

end probability_no_adjacent_same_roll_l8_8323


namespace unique_solution_exists_q_l8_8316

theorem unique_solution_exists_q :
  (∃ q : ℝ, q ≠ 0 ∧ (∀ x y : ℝ, (2 * q * x^2 - 20 * x + 5 = 0) ∧ (2 * q * y^2 - 20 * y + 5 = 0) → x = y)) ↔ q = 10 := 
sorry

end unique_solution_exists_q_l8_8316


namespace total_pieces_ten_row_triangle_l8_8013

-- Definitions based on the conditions
def rods (n : ℕ) : ℕ :=
  (n * (2 * 4 + (n - 1) * 5)) / 2

def connectors (n : ℕ) : ℕ :=
  ((n + 1) * (2 * 1 + n * 1)) / 2

def support_sticks (n : ℕ) : ℕ := 
  if n >= 3 then ((n - 2) * (2 * 2 + (n - 3) * 2)) / 2 else 0

-- The theorem stating the total number of pieces is 395 for a ten-row triangle
theorem total_pieces_ten_row_triangle : rods 10 + connectors 10 + support_sticks 10 = 395 :=
by
  sorry

end total_pieces_ten_row_triangle_l8_8013


namespace jake_hours_of_work_l8_8497

def initialDebt : ℕ := 100
def amountPaid : ℕ := 40
def workRate : ℕ := 15
def remainingDebt : ℕ := initialDebt - amountPaid

theorem jake_hours_of_work : remainingDebt / workRate = 4 := by
  sorry

end jake_hours_of_work_l8_8497


namespace hyperbola_k_range_l8_8597

theorem hyperbola_k_range (k : ℝ) : ((k + 2) * (6 - 2 * k) > 0) ↔ (-2 < k ∧ k < 3) := 
sorry

end hyperbola_k_range_l8_8597


namespace problem_l8_8814

/-- Given a number d > 7, 
    digits A and B in base d such that the equation AB_d + AA_d = 172_d holds, 
    we want to prove that A_d - B_d = 5. --/

theorem problem (d A B : ℕ) (hd : 7 < d)
  (hAB : d * A + B + d * A + A = 1 * d^2 + 7 * d + 2) : A - B = 5 :=
sorry

end problem_l8_8814


namespace polynomial_min_k_eq_l8_8927

theorem polynomial_min_k_eq {k : ℝ} :
  (∃ k : ℝ, ∀ x y : ℝ, 9 * x^2 - 12 * k * x * y + (2 * k^2 + 3) * y^2 - 6 * x - 9 * y + 12 >= 0)
  ↔ k = (Real.sqrt 3) / 4 :=
sorry

end polynomial_min_k_eq_l8_8927


namespace percent_of_x_eq_to_y_l8_8418

variable {x y : ℝ}

theorem percent_of_x_eq_to_y (h: 0.5 * (x - y) = 0.3 * (x + y)) : y = 0.25 * x :=
by
  sorry

end percent_of_x_eq_to_y_l8_8418


namespace original_rent_l8_8417

theorem original_rent {avg_rent_before avg_rent_after : ℝ} (total_before total_after increase_percentage diff_increase : ℝ) :
  avg_rent_before = 800 → 
  avg_rent_after = 880 → 
  total_before = 4 * avg_rent_before → 
  total_after = 4 * avg_rent_after → 
  diff_increase = total_after - total_before → 
  increase_percentage = 0.20 → 
  diff_increase = increase_percentage * R → 
  R = 1600 :=
by sorry

end original_rent_l8_8417


namespace sum_of_discount_rates_l8_8453

theorem sum_of_discount_rates : 
  let fox_price := 15
  let pony_price := 20
  let fox_pairs := 3
  let pony_pairs := 2
  let total_savings := 9
  let pony_discount := 18.000000000000014
  let fox_discount := 4
  let total_discount_rate := fox_discount + pony_discount
  total_discount_rate = 22.000000000000014 := by
sorry

end sum_of_discount_rates_l8_8453


namespace g_constant_term_l8_8378

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := f * g

-- Conditions from the problem
def f_has_constant_term_5 : f.coeff 0 = 5 := sorry
def h_has_constant_term_neg_10 : h.coeff 0 = -10 := sorry
def g_is_quadratic : g.degree ≤ 2 := sorry

-- Statement of the problem
theorem g_constant_term : g.coeff 0 = -2 :=
by
  have h_eq_fg : h = f * g := rfl
  have f_const := f_has_constant_term_5
  have h_const := h_has_constant_term_neg_10
  have g_quad := g_is_quadratic
  sorry

end g_constant_term_l8_8378


namespace scientific_notation_l8_8795

theorem scientific_notation (n : ℝ) (h1 : n = 17600) : ∃ a b, (a = 1.76) ∧ (b = 4) ∧ n = a * 10^b :=
by {
  sorry
}

end scientific_notation_l8_8795


namespace diagonals_of_60_sided_polygon_exterior_angle_of_60_sided_polygon_l8_8967

noncomputable def diagonals_in_regular_polygon (n : ℕ) : ℕ :=
  n * (n - 3) / 2

noncomputable def exterior_angle (n : ℕ) : ℝ :=
  360.0 / n

theorem diagonals_of_60_sided_polygon :
  diagonals_in_regular_polygon 60 = 1710 :=
by
  sorry

theorem exterior_angle_of_60_sided_polygon :
  exterior_angle 60 = 6.0 :=
by
  sorry

end diagonals_of_60_sided_polygon_exterior_angle_of_60_sided_polygon_l8_8967
