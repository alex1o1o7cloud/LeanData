import Mathlib

namespace quadratic_equation_from_absolute_value_l2042_204275

theorem quadratic_equation_from_absolute_value :
  ∃ b c : ℝ, (∀ x : ℝ, |x - 8| = 3 ↔ x^2 + b * x + c = 0) ∧ (b, c) = (-16, 55) :=
sorry

end quadratic_equation_from_absolute_value_l2042_204275


namespace probability_same_color_l2042_204208

-- Definitions for the conditions
def blue_balls : Nat := 8
def yellow_balls : Nat := 5
def total_balls : Nat := blue_balls + yellow_balls

def prob_two_balls_same_color : ℚ :=
  (blue_balls/total_balls) * (blue_balls/total_balls) + (yellow_balls/total_balls) * (yellow_balls/total_balls)

-- Lean statement to be proved
theorem probability_same_color : prob_two_balls_same_color = 89 / 169 :=
by
  -- The proof is omitted as per the instruction
  sorry

end probability_same_color_l2042_204208


namespace no_valid_solutions_l2042_204259

theorem no_valid_solutions (x : ℝ) (h : x ≠ 1) : 
  ¬(3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) :=
sorry

end no_valid_solutions_l2042_204259


namespace opposite_of_2023_is_neg_2023_l2042_204274

theorem opposite_of_2023_is_neg_2023 : (2023 + (-2023) = 0) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l2042_204274


namespace equation_solution_l2042_204243

noncomputable def solve_equation : Set ℝ := {x : ℝ | (3 * x + 2) / (x ^ 2 + 5 * x + 6) = 3 * x / (x - 1)
                                             ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ 1}

theorem equation_solution (r : ℝ) (h : r ∈ solve_equation) : 3 * r ^ 3 + 12 * r ^ 2 + 19 * r + 2 = 0 :=
sorry

end equation_solution_l2042_204243


namespace least_three_digit_divisible_by_2_3_5_7_l2042_204242

theorem least_three_digit_divisible_by_2_3_5_7 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∀ k, 2 ∣ k ∧ 3 ∣ k ∧ 5 ∣ k ∧ 7 ∣ k → n ≤ k) ∧
  (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) ∧ n = 210 :=
by sorry

end least_three_digit_divisible_by_2_3_5_7_l2042_204242


namespace trigonometric_identity_l2042_204214

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 :=
sorry

end trigonometric_identity_l2042_204214


namespace jen_age_proof_l2042_204228

variable (JenAge : ℕ) (SonAge : ℕ)

theorem jen_age_proof (h1 : SonAge = 16) (h2 : JenAge = 3 * SonAge - 7) : JenAge = 41 :=
by
  -- conditions
  rw [h1] at h2
  -- substitution and simplification
  have h3 : JenAge = 3 * 16 - 7 := h2
  norm_num at h3
  exact h3

end jen_age_proof_l2042_204228


namespace exam_correct_answers_l2042_204256

theorem exam_correct_answers (C W : ℕ) 
  (h1 : C + W = 60)
  (h2 : 4 * C - W = 160) : 
  C = 44 :=
sorry

end exam_correct_answers_l2042_204256


namespace greatest_sundays_in_56_days_l2042_204260

theorem greatest_sundays_in_56_days (days_in_first: ℕ) (days_in_week: ℕ) (sundays_in_week: ℕ) : ℕ :=
by 
  -- Given conditions
  have days_in_first := 56
  have days_in_week := 7
  have sundays_in_week := 1

  -- Conclusion
  let num_weeks := days_in_first / days_in_week

  -- Answer
  exact num_weeks * sundays_in_week

-- This theorem establishes that the greatest number of Sundays in 56 days is indeed 8.
-- Proof: The number of Sundays in 56 days is given by the number of weeks (which is 8) times the number of Sundays per week (which is 1).

example : greatest_sundays_in_56_days 56 7 1 = 8 := 
by 
  unfold greatest_sundays_in_56_days
  exact rfl

end greatest_sundays_in_56_days_l2042_204260


namespace add_neg_eq_neg_add_neg_ten_plus_neg_twelve_l2042_204249

theorem add_neg_eq_neg_add (a b : Int) : a + -b = a - b := by
  sorry

theorem neg_ten_plus_neg_twelve : -10 + (-12) = -22 := by
  have h1 : -10 + (-12) = -10 - 12 := add_neg_eq_neg_add _ _
  have h2 : -10 - 12 = -(10 + 12) := by
    sorry -- This step corresponds to recognizing the arithmetic rule for subtraction.
  have h3 : -(10 + 12) = -22 := by
    sorry -- This step is the concrete calculation.
  exact Eq.trans h1 (Eq.trans h2 h3)

end add_neg_eq_neg_add_neg_ten_plus_neg_twelve_l2042_204249


namespace unique_function_satisfying_condition_l2042_204298

theorem unique_function_satisfying_condition (k : ℕ) (hk : 0 < k) :
  ∀ f : ℕ → ℕ, (∀ m n : ℕ, 0 < m → 0 < n → f m + f n ∣ (m + n) ^ k) →
  ∃ c : ℕ, ∀ n : ℕ, f n = n + c :=
by
  sorry

end unique_function_satisfying_condition_l2042_204298


namespace max_c_for_log_inequality_l2042_204239

theorem max_c_for_log_inequality (a b : ℝ) (ha : 1 < a) (hb : 1 < b) : 
  ∃ c : ℝ, c = 1 / 3 ∧ (1 / (3 + Real.log b / Real.log a) + 1 / (3 + Real.log a / Real.log b) ≥ c) :=
by
  use 1 / 3
  sorry

end max_c_for_log_inequality_l2042_204239


namespace longest_side_similar_triangle_l2042_204211

theorem longest_side_similar_triangle (a b c : ℝ) (p : ℝ) (h₀ : a = 8) (h₁ : b = 15) (h₂ : c = 17) (h₃ : a^2 + b^2 = c^2) (h₄ : p = 160) :
  ∃ x : ℝ, (8 * x) + (15 * x) + (17 * x) = p ∧ 17 * x = 68 :=
by
  sorry

end longest_side_similar_triangle_l2042_204211


namespace housewife_oil_cost_l2042_204261

theorem housewife_oil_cost (P R M : ℝ) (hR : R = 45) (hReduction : (P - R) = (15 / 100) * P)
  (hMoreOil : M / P = M / R + 4) : M = 150.61 := 
by
  sorry

end housewife_oil_cost_l2042_204261


namespace find_a_plus_b_l2042_204227

def cubic_function (a b : ℝ) (x : ℝ) := x^3 - x^2 - a * x + b

def tangent_line (x : ℝ) := 2 * x + 1

theorem find_a_plus_b (a b : ℝ) 
  (h1 : tangent_line 0 = 1)
  (h2 : cubic_function a b 0 = 1)
  (h3 : deriv (cubic_function a b) 0 = 2) :
  a + b = -1 :=
by
  sorry

end find_a_plus_b_l2042_204227


namespace simplify_evaluate_expression_l2042_204282

theorem simplify_evaluate_expression (a b : ℤ) (h1 : a = -2) (h2 : b = 4) : 
  (-(3 * a)^2 + 6 * a * b - (a^2 + 3 * (a - 2 * a * b))) = 14 :=
by
  rw [h1, h2]
  sorry

end simplify_evaluate_expression_l2042_204282


namespace find_salary_l2042_204278

def salary_remaining (S : ℝ) (food : ℝ) (house_rent : ℝ) (clothes : ℝ) (remaining : ℝ) : Prop :=
  S - food * S - house_rent * S - clothes * S = remaining

theorem find_salary :
  ∀ S : ℝ, 
  salary_remaining S (1/5) (1/10) (3/5) 15000 → 
  S = 150000 :=
by
  intros S h
  sorry

end find_salary_l2042_204278


namespace magic_square_exists_l2042_204229

theorem magic_square_exists : 
  ∃ (a b c d e f g h : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ 
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ 
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ 
    f ≠ g ∧ f ≠ h ∧
    g ≠ h ∧
    a + b + c = 12 ∧ d + e + f = 12 ∧ g + h + 0 = 12 ∧
    a + d + g = 12 ∧ b + 0 + h = 12 ∧ c + f + 0 = 12 :=
sorry

end magic_square_exists_l2042_204229


namespace number_exceeds_its_part_by_20_l2042_204289

theorem number_exceeds_its_part_by_20 (x : ℝ) (h : x = (3/8) * x + 20) : x = 32 :=
sorry

end number_exceeds_its_part_by_20_l2042_204289


namespace part1_part2_part3_l2042_204286

open Set

variable (x : ℝ)

def A := {x : ℝ | 3 ≤ x ∧ x < 7}
def B := {x : ℝ | 2 < x ∧ x < 10}

theorem part1 : A ∩ B = {x | 3 ≤ x ∧ x < 7} :=
sorry

theorem part2 : (Aᶜ : Set ℝ) = {x | x < 3 ∨ x ≥ 7} :=
sorry

theorem part3 : (A ∪ B)ᶜ = {x | x ≤ 2 ∨ x ≥ 10} :=
sorry

end part1_part2_part3_l2042_204286


namespace sum_of_triangulars_iff_sum_of_squares_l2042_204203

-- Definitions of triangular numbers and sums of squares
def isTriangular (n : ℕ) : Prop := ∃ k, n = k * (k + 1) / 2
def isSumOfTwoTriangulars (m : ℕ) : Prop := ∃ x y, m = (x * (x + 1) / 2) + (y * (y + 1) / 2)
def isSumOfTwoSquares (n : ℕ) : Prop := ∃ a b, n = a * a + b * b

-- Main theorem statement
theorem sum_of_triangulars_iff_sum_of_squares (m : ℕ) (h_pos : 0 < m) : 
  isSumOfTwoTriangulars m ↔ isSumOfTwoSquares (4 * m + 1) :=
sorry

end sum_of_triangulars_iff_sum_of_squares_l2042_204203


namespace slope_of_intersection_line_l2042_204205

theorem slope_of_intersection_line 
    (x y : ℝ)
    (h1 : x^2 + y^2 - 6*x + 4*y - 20 = 0)
    (h2 : x^2 + y^2 - 2*x - 6*y + 10 = 0) :
    ∃ m : ℝ, m = 0.4 := 
sorry

end slope_of_intersection_line_l2042_204205


namespace carousel_seats_count_l2042_204244

theorem carousel_seats_count :
  ∃ (yellow blue red : ℕ), 
  (yellow + blue + red = 100) ∧ 
  (yellow = 34) ∧ 
  (blue = 20) ∧ 
  (red = 46) ∧ 
  (∀ i : ℕ, i < yellow → ∃ j : ℕ, j = yellow.succ * j ∧ (j < 100 ∧ j ≠ yellow.succ * j)) ∧ 
  (∀ k : ℕ, k < blue → ∃ m : ℕ, m = blue.succ * m ∧ (m < 100 ∧ m ≠ blue.succ * m)) ∧ 
  (∀ n : ℕ, n < red → ∃ p : ℕ, p = red.succ * p ∧ (p < 100 ∧ p ≠ red.succ * p)) :=
sorry

end carousel_seats_count_l2042_204244


namespace Matt_jumped_for_10_minutes_l2042_204293

def Matt_skips_per_second : ℕ := 3

def total_skips : ℕ := 1800

def minutes_jumped (m : ℕ) : Prop :=
  m * (Matt_skips_per_second * 60) = total_skips

theorem Matt_jumped_for_10_minutes : minutes_jumped 10 :=
by
  sorry

end Matt_jumped_for_10_minutes_l2042_204293


namespace age_ratio_l2042_204283

variable (A B : ℕ)
variable (k : ℕ)

-- Define the conditions
def sum_of_ages : Prop := A + B = 60
def multiple_of_age : Prop := A = k * B

-- Theorem to prove the ratio of ages
theorem age_ratio (h_sum : sum_of_ages A B) (h_multiple : multiple_of_age A B k) : A = 12 * B :=
by
  sorry

end age_ratio_l2042_204283


namespace upper_left_region_l2042_204200

theorem upper_left_region (t : ℝ) : (2 - 2 * t + 4 ≤ 0) → (t ≤ 3) :=
by
  sorry

end upper_left_region_l2042_204200


namespace fraction_addition_l2042_204218

theorem fraction_addition : (1 / 3) + (5 / 12) = 3 / 4 := 
sorry

end fraction_addition_l2042_204218


namespace compound_oxygen_atoms_l2042_204204

theorem compound_oxygen_atoms (H C O : Nat) (mw : Nat) (H_weight C_weight O_weight : Nat) 
  (h_H : H = 2)
  (h_C : C = 1)
  (h_mw : mw = 62)
  (h_H_weight : H_weight = 1)
  (h_C_weight : C_weight = 12)
  (h_O_weight : O_weight = 16)
  : O = 3 :=
by
  sorry

end compound_oxygen_atoms_l2042_204204


namespace average_age_of_team_is_23_l2042_204220

noncomputable def average_age_team (A : ℝ) : Prop :=
  let captain_age := 27
  let wicket_keeper_age := 28
  let team_size := 11
  let remaining_players := team_size - 2
  let remaining_average_age := A - 1
  11 * A = 55 + 9 * (A - 1)

theorem average_age_of_team_is_23 : average_age_team 23 := by
  sorry

end average_age_of_team_is_23_l2042_204220


namespace arithmetic_sequence_ratio_l2042_204266

variable {a_n b_n : ℕ → ℕ}
variable {S_n T_n : ℕ → ℕ}

-- Given two arithmetic sequences a_n and b_n, their sums of the first n terms are S_n and T_n respectively.
-- Given that S_n / T_n = (2n + 2) / (n + 3).
-- Prove that a_10 / b_10 = 20 / 11.

theorem arithmetic_sequence_ratio (h : ∀ n, S_n n / T_n n = (2 * n + 2) / (n + 3)) : (a_n 10) / (b_n 10) = 20 / 11 := 
by
  sorry

end arithmetic_sequence_ratio_l2042_204266


namespace avg_fish_in_bodies_of_water_l2042_204235

def BoastPoolFish : ℕ := 75
def OnumLakeFish : ℕ := BoastPoolFish + 25
def RiddlePondFish : ℕ := OnumLakeFish / 2
def RippleCreekFish : ℕ := 2 * (OnumLakeFish - BoastPoolFish)
def WhisperingSpringsFish : ℕ := (3 * RiddlePondFish) / 2

def totalFish : ℕ := BoastPoolFish + OnumLakeFish + RiddlePondFish + RippleCreekFish + WhisperingSpringsFish
def averageFish : ℕ := totalFish / 5

theorem avg_fish_in_bodies_of_water : averageFish = 68 :=
by
  sorry

end avg_fish_in_bodies_of_water_l2042_204235


namespace equilateral_triangle_area_l2042_204206

theorem equilateral_triangle_area (perimeter : ℝ) (h1 : perimeter = 120) :
  ∃ A : ℝ, A = 400 * Real.sqrt 3 ∧
    (∃ s : ℝ, s = perimeter / 3 ∧ A = (Real.sqrt 3 / 4) * (s ^ 2)) :=
by
  sorry

end equilateral_triangle_area_l2042_204206


namespace ways_A_to_C_via_B_l2042_204277

def ways_A_to_B : Nat := 2
def ways_B_to_C : Nat := 3

theorem ways_A_to_C_via_B : ways_A_to_B * ways_B_to_C = 6 := by
  sorry

end ways_A_to_C_via_B_l2042_204277


namespace correct_factorization_l2042_204269

-- Definitions from conditions
def A: Prop := ∀ x y: ℝ, x^2 - 4*y^2 = (x + y) * (x - 4*y)
def B: Prop := ∀ x: ℝ, (x + 4) * (x - 4) = x^2 - 16
def C: Prop := ∀ x: ℝ, x^2 - 2*x + 1 = (x - 1)^2
def D: Prop := ∀ x: ℝ, x^2 - 8*x + 9 = (x - 4)^2 - 7

-- Goal is to prove that C is a correct factorization
theorem correct_factorization: C := by
  sorry

end correct_factorization_l2042_204269


namespace relationship_abc_l2042_204232

theorem relationship_abc (a b c : ℝ) 
  (h₁ : a = Real.log 0.5 / Real.log 2) 
  (h₂ : b = Real.sqrt 2) 
  (h₃ : c = 0.5 ^ 2) : 
  a < c ∧ c < b := by
  sorry

end relationship_abc_l2042_204232


namespace instantaneous_velocity_at_2_l2042_204230

def s (t : ℝ) : ℝ := 3 * t^3 - 2 * t^2 + t + 1

theorem instantaneous_velocity_at_2 : 
  (deriv s 2) = 29 :=
by
  -- The proof is skipped by using sorry
  sorry

end instantaneous_velocity_at_2_l2042_204230


namespace sum_of_digits_smallest_N_l2042_204245

theorem sum_of_digits_smallest_N :
  ∃ (N : ℕ), N ≤ 999 ∧ 72 * N < 1000 ∧ (N = 13) ∧ (1 + 3 = 4) := by
  sorry

end sum_of_digits_smallest_N_l2042_204245


namespace equal_constants_l2042_204219

theorem equal_constants (a b : ℝ) :
  (∃ᶠ n in at_top, ⌊a * n + b⌋ ≥ ⌊a + b * n⌋) →
  (∃ᶠ m in at_top, ⌊a + b * m⌋ ≥ ⌊a * m + b⌋) →
  a = b :=
by
  sorry

end equal_constants_l2042_204219


namespace xyz_problem_l2042_204253

variables {x y z : ℝ}

theorem xyz_problem
  (h1 : y + z = 10 - 4 * x)
  (h2 : x + z = -16 - 4 * y)
  (h3 : x + y = 9 - 4 * z) :
  3 * x + 3 * y + 3 * z = 1.5 :=
by 
  sorry

end xyz_problem_l2042_204253


namespace ads_minutes_l2042_204292

-- Definitions and conditions
def videos_per_day : Nat := 2
def minutes_per_video : Nat := 7
def total_time_on_youtube : Nat := 17

-- The theorem to prove
theorem ads_minutes : (total_time_on_youtube - (videos_per_day * minutes_per_video)) = 3 :=
by
  sorry

end ads_minutes_l2042_204292


namespace percentage_difference_l2042_204212

variable (x y : ℝ)
variable (hxy : x = 6 * y)

theorem percentage_difference : ((x - y) / x) * 100 = 83.33 := by
  sorry

end percentage_difference_l2042_204212


namespace tangent_parallel_coordinates_l2042_204210

theorem tangent_parallel_coordinates :
  (∃ (x1 y1 x2 y2 : ℝ), 
    (y1 = x1^3 - 2) ∧ (y2 = x2^3 - 2) ∧ 
    ((3 * x1^2 = 3) ∧ (3 * x2^2 = 3)) ∧ 
    ((x1 = 1 ∧ y1 = -1) ∧ (x2 = -1 ∧ y2 = -3))) :=
sorry

end tangent_parallel_coordinates_l2042_204210


namespace topsoil_cost_proof_l2042_204264

-- Definitions
def cost_per_cubic_foot : ℕ := 8
def cubic_feet_per_cubic_yard : ℕ := 27
def amount_in_cubic_yards : ℕ := 7

-- Theorem
theorem topsoil_cost_proof : cost_per_cubic_foot * cubic_feet_per_cubic_yard * amount_in_cubic_yards = 1512 := by
  -- proof logic goes here
  sorry

end topsoil_cost_proof_l2042_204264


namespace total_distance_is_105_km_l2042_204291

-- Define the boat's speed in still water
def boat_speed_still_water : ℝ := 50

-- Define the current speeds for each hour
def current_speed_first_hour : ℝ := 10
def current_speed_second_hour : ℝ := 20
def current_speed_third_hour : ℝ := 15

-- Calculate the effective speeds for each hour
def effective_speed_first_hour := boat_speed_still_water - current_speed_first_hour
def effective_speed_second_hour := boat_speed_still_water - current_speed_second_hour
def effective_speed_third_hour := boat_speed_still_water - current_speed_third_hour

-- Calculate the distance traveled in each hour
def distance_first_hour := effective_speed_first_hour * 1
def distance_second_hour := effective_speed_second_hour * 1
def distance_third_hour := effective_speed_third_hour * 1

-- Define the total distance
def total_distance_traveled := distance_first_hour + distance_second_hour + distance_third_hour

-- Prove that the total distance traveled is 105 km
theorem total_distance_is_105_km : total_distance_traveled = 105 := by
  sorry

end total_distance_is_105_km_l2042_204291


namespace problem_part1_problem_part2_problem_part3_l2042_204202

noncomputable def given_quadratic (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 - (Real.sqrt 3 + 1) * x + m

noncomputable def sin_cos_eq_quadratic_roots (θ m : ℝ) : Prop := 
  let sinθ := Real.sin θ
  let cosθ := Real.cos θ
  given_quadratic sinθ m = 0 ∧ given_quadratic cosθ m = 0

theorem problem_part1 (θ : ℝ) (h : 0 < θ ∧ θ < 2 * Real.pi) (Hroots : sin_cos_eq_quadratic_roots θ m) : 
  (Real.sin θ / (1 - Real.cos θ) + Real.cos θ / (1 - Real.tan θ)) = (3 + 5 * Real.sqrt 3) / 4 :=
sorry

theorem problem_part2 {θ : ℝ} (h : 0 < θ ∧ θ < 2 * Real.pi) (Hroots : sin_cos_eq_quadratic_roots θ m) : 
  m = Real.sqrt 3 / 4 :=
sorry

theorem problem_part3 (m : ℝ) (sinθ1 cosθ1 sinθ2 cosθ2 : ℝ) (θ1 θ2 : ℝ)
  (H1 : sinθ1 = Real.sqrt 3 / 2 ∧ cosθ1 = 1 / 2 ∧ θ1 = Real.pi / 3)
  (H2 : sinθ2 = 1 / 2 ∧ cosθ2 = Real.sqrt 3 / 2 ∧ θ2 = Real.pi / 6) : 
  ∃ θ, sin_cos_eq_quadratic_roots θ m ∧ 
       (Real.sin θ = sinθ1 ∧ Real.cos θ = cosθ1 ∨ Real.sin θ = sinθ2 ∧ Real.cos θ = cosθ2) :=
sorry

end problem_part1_problem_part2_problem_part3_l2042_204202


namespace min_a_b_l2042_204276

theorem min_a_b : 
  (∀ x : ℝ, 3 * a * (Real.sin x + Real.cos x) + 2 * b * Real.sin (2 * x) ≤ 3) →
  a + b = -2 →
  a = -4 / 5 :=
by
  sorry

end min_a_b_l2042_204276


namespace distribution_of_tickets_l2042_204225

-- Define the number of total people and the number of tickets
def n : ℕ := 10
def k : ℕ := 3

-- Define the permutation function P(n, k)
def P (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Main theorem statement
theorem distribution_of_tickets : P n k = 720 := by
  unfold P
  sorry

end distribution_of_tickets_l2042_204225


namespace batsman_average_l2042_204224

variable (x : ℝ)

theorem batsman_average (h1 : ∀ x, 11 * x + 55 = 12 * (x + 1)) : 
  x = 43 → (x + 1 = 44) :=
by
  sorry

end batsman_average_l2042_204224


namespace fraction_of_shaded_area_l2042_204216

theorem fraction_of_shaded_area (total_length total_width : ℕ) (total_area : ℕ)
  (quarter_fraction half_fraction : ℚ)
  (h1 : total_length = 15) 
  (h2 : total_width = 20)
  (h3 : total_area = total_length * total_width)
  (h4 : quarter_fraction = 1 / 4)
  (h5 : half_fraction = 1 / 2) :
  (half_fraction * quarter_fraction * total_area) / total_area = 1 / 8 :=
by
  sorry

end fraction_of_shaded_area_l2042_204216


namespace least_n_froods_l2042_204234

theorem least_n_froods (n : ℕ) : (∃ n, n ≥ 30 ∧ (n * (n + 1)) / 2 > 15 * n) ∧ (∀ m < 30, (m * (m + 1)) / 2 ≤ 15 * m) :=
sorry

end least_n_froods_l2042_204234


namespace sequence_property_l2042_204213

theorem sequence_property (x : ℝ) (a : ℕ → ℝ) (h : ∀ n, a n = 1 + x ^ (n + 1) + x ^ (n + 2)) (h_given : (a 2) ^ 2 = (a 1) * (a 3)) :
  ∀ n ≥ 3, (a n) ^ 2 = (a (n - 1)) * (a (n + 1)) :=
by
  intros n hn
  sorry

end sequence_property_l2042_204213


namespace tangent_line_at_2_is_12x_minus_y_minus_17_eq_0_range_of_m_for_three_distinct_real_roots_l2042_204257

-- Define the function f
noncomputable def f (x : ℝ) := 2 * x^3 - 3 * x^2 + 3

-- First proof problem: Equation of the tangent line at (2, 7)
theorem tangent_line_at_2_is_12x_minus_y_minus_17_eq_0 :
  ∀ x y : ℝ, y = f x → (x = 2) → y = 7 → (∃ (m b : ℝ), (m = 12) ∧ (b = -17) ∧ (∀ x, 12 * x - y - 17 = 0)) :=
by
  sorry

-- Second proof problem: Range of m for three distinct real roots
theorem range_of_m_for_three_distinct_real_roots :
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) → -3 < m ∧ m < -2 :=
by 
  sorry

end tangent_line_at_2_is_12x_minus_y_minus_17_eq_0_range_of_m_for_three_distinct_real_roots_l2042_204257


namespace initial_notebooks_is_10_l2042_204271

-- Define the conditions
def ordered_notebooks := 6
def lost_notebooks := 2
def current_notebooks := 14

-- Define the initial number of notebooks
def initial_notebooks (N : ℕ) :=
  N + ordered_notebooks - lost_notebooks = current_notebooks

-- The proof statement
theorem initial_notebooks_is_10 : initial_notebooks 10 :=
by
  sorry

end initial_notebooks_is_10_l2042_204271


namespace paper_cups_calculation_l2042_204299

def total_pallets : Nat := 20
def paper_towels : Nat := total_pallets / 2
def tissues : Nat := total_pallets / 4
def paper_plates : Nat := total_pallets / 5
def other_paper_products : Nat := paper_towels + tissues + paper_plates
def paper_cups : Nat := total_pallets - other_paper_products

theorem paper_cups_calculation : paper_cups = 1 := by
  sorry

end paper_cups_calculation_l2042_204299


namespace exists_increasing_triplet_l2042_204263

theorem exists_increasing_triplet (f : ℕ → ℕ) (bij : Function.Bijective f) :
  ∃ (a d : ℕ), 0 < a ∧ 0 < d ∧ f a < f (a + d) ∧ f (a + d) < f (a + 2 * d) :=
by
  sorry

end exists_increasing_triplet_l2042_204263


namespace equation_represents_point_l2042_204217

theorem equation_represents_point (a b x y : ℝ) :
  x^2 + y^2 + 2 * a * x + 2 * b * y + a^2 + b^2 = 0 ↔ x = -a ∧ y = -b := 
by sorry

end equation_represents_point_l2042_204217


namespace Lagrange_interpolation_poly_l2042_204226

noncomputable def Lagrange_interpolation (P : ℝ → ℝ) : Prop :=
  P (-1) = -11 ∧ P (1) = -3 ∧ P (2) = 1 ∧ P (3) = 13

theorem Lagrange_interpolation_poly :
  ∃ P : ℝ → ℝ, Lagrange_interpolation P ∧ ∀ x, P x = x^3 - 2*x^2 + 3*x - 5 :=
by
  sorry

end Lagrange_interpolation_poly_l2042_204226


namespace maximize_take_home_pay_l2042_204209

-- Define the tax system condition
def tax (y : ℝ) : ℝ := y^3

-- Define the take-home pay condition
def take_home_pay (y : ℝ) : ℝ := 100 * y^2 - tax y

-- The theorem to prove the maximum take-home pay is achieved at a specific income level
theorem maximize_take_home_pay : 
  ∃ y : ℝ, take_home_pay y = 100 * 50^2 - 50^3 := sorry

end maximize_take_home_pay_l2042_204209


namespace senior_ticket_cost_is_13_l2042_204270

theorem senior_ticket_cost_is_13
    (adult_ticket_cost : ℕ)
    (child_ticket_cost : ℕ)
    (senior_ticket_cost : ℕ)
    (total_cost : ℕ)
    (num_adults : ℕ)
    (num_children : ℕ)
    (num_senior_citizens : ℕ)
    (age_child1 : ℕ)
    (age_child2 : ℕ)
    (age_child3 : ℕ) :
    adult_ticket_cost = 11 → 
    child_ticket_cost = 8 →
    total_cost = 64 →
    num_adults = 2 →
    num_children = 2 → -- children with discount tickets
    num_senior_citizens = 2 →
    age_child1 = 7 → 
    age_child2 = 10 → 
    age_child3 = 14 → -- this child does not get discount
    senior_ticket_cost * num_senior_citizens = total_cost - (num_adults * adult_ticket_cost + num_children * child_ticket_cost) →
    senior_ticket_cost = 13 :=
by
  intros
  sorry

end senior_ticket_cost_is_13_l2042_204270


namespace passed_in_both_subjects_l2042_204268

theorem passed_in_both_subjects (A B C : ℝ)
  (hA : A = 0.25)
  (hB : B = 0.48)
  (hC : C = 0.27) :
  1 - (A + B - C) = 0.54 := by
  sorry

end passed_in_both_subjects_l2042_204268


namespace sum_R1_R2_eq_19_l2042_204246

-- Definitions for F_1 and F_2 in base R_1 and R_2
def F1_R1 : ℚ := 37 / 99
def F2_R1 : ℚ := 73 / 99
def F1_R2 : ℚ := 25 / 99
def F2_R2 : ℚ := 52 / 99

-- Prove that the sum of R1 and R2 is 19
theorem sum_R1_R2_eq_19 (R1 R2 : ℕ) (hF1R1 : F1_R1 = (3 * R1 + 7) / (R1^2 - 1))
  (hF2R1 : F2_R1 = (7 * R1 + 3) / (R1^2 - 1))
  (hF1R2 : F1_R2 = (2 * R2 + 5) / (R2^2 - 1))
  (hF2R2 : F2_R2 = (5 * R2 + 2) / (R2^2 - 1)) :
  R1 + R2 = 19 :=
  sorry

end sum_R1_R2_eq_19_l2042_204246


namespace part1_part2_l2042_204223

noncomputable def f (a x : ℝ) : ℝ := x^2 + (a+1)*x + a

theorem part1 (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → f a x < 0) → a ≤ -2 := sorry

theorem part2 (a x : ℝ) :
  f a x > 0 ↔
  (a > 1 ∧ (x < -a ∨ x > -1)) ∨
  (a = 1 ∧ x ≠ -1) ∨
  (a < 1 ∧ (x < -1 ∨ x > -a)) := sorry

end part1_part2_l2042_204223


namespace compute_expression_l2042_204296

theorem compute_expression : 1005^2 - 995^2 - 1003^2 + 997^2 = 8000 :=
by
  sorry

end compute_expression_l2042_204296


namespace contractor_realized_work_done_after_20_days_l2042_204290

-- Definitions based on conditions
variable (W w : ℝ)  -- W is total work, w is work per person per day
variable (d : ℝ)  -- d is the number of days we want to find

-- Conditions transformation into Lean definitions
def initial_work_done_in_d_days := 10 * w * d = (1 / 4) * W
def remaining_work_done_in_75_days := 8 * w * 75 = (3 / 4) * W
def total_work := (10 * w * d) + (8 * w * 75) = W

-- Proof statement we need to prove
theorem contractor_realized_work_done_after_20_days :
  initial_work_done_in_d_days W w d ∧ 
  remaining_work_done_in_75_days W w → 
  total_work W w d →
  d = 20 := by
  sorry

end contractor_realized_work_done_after_20_days_l2042_204290


namespace yeast_population_at_1_20_pm_l2042_204297

def yeast_population (initial : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  initial * rate^time

theorem yeast_population_at_1_20_pm : 
  yeast_population 50 3 4 = 4050 :=
by
  -- Proof goes here
  sorry

end yeast_population_at_1_20_pm_l2042_204297


namespace binomial_fermat_l2042_204240

theorem binomial_fermat (p : ℕ) (a b : ℤ) (hp : p.Prime) : 
  ((a + b)^p - a^p - b^p) % p = 0 := by
  sorry

end binomial_fermat_l2042_204240


namespace cylinder_volume_eq_sphere_volume_l2042_204221

theorem cylinder_volume_eq_sphere_volume (a h R x : ℝ) (h_pos : h > 0) (a_pos : a > 0) (R_pos : R > 0)
  (h_volume_eq : (a - h) * x^2 - a * h * x + 2 * h * R^2 = 0) :
  ∃ x : ℝ, a > h ∧ x > 0 ∧ x < h ∧ x = 2 * R^2 / a ∨ 
           h < a ∧ 0 < x ∧ x = (a * h / (a - h)) - h ∧ R^2 < h^2 / 2 :=
sorry

end cylinder_volume_eq_sphere_volume_l2042_204221


namespace costOfBrantsRoyalBananaSplitSundae_l2042_204236

-- Define constants for the prices of the known sundaes
def yvette_sundae_cost : ℝ := 9.00
def alicia_sundae_cost : ℝ := 7.50
def josh_sundae_cost : ℝ := 8.50

-- Define the tip percentage
def tip_percentage : ℝ := 0.20

-- Define the final bill amount
def final_bill : ℝ := 42.00

-- Calculate the total known sundaes cost
def total_known_sundaes_cost : ℝ := yvette_sundae_cost + alicia_sundae_cost + josh_sundae_cost

-- Define a proof to show that the cost of Brant's sundae is $10.00
theorem costOfBrantsRoyalBananaSplitSundae : 
  total_known_sundaes_cost + b = final_bill / (1 + tip_percentage) → b = 10 :=
sorry

end costOfBrantsRoyalBananaSplitSundae_l2042_204236


namespace interval_of_increase_logb_l2042_204233

noncomputable def f (x : ℝ) := Real.logb 5 (2 * x + 1)

-- Define the domain
def domain : Set ℝ := {x | 2 * x + 1 > 0}

-- Define the interval of monotonic increase for the function
def interval_of_increase (f : ℝ → ℝ) : Set ℝ := {x | ∀ y, x < y → f x < f y}

-- Statement of the problem
theorem interval_of_increase_logb :
  interval_of_increase f = {x | x > - (1 / 2)} :=
by
  have h_increase : ∀ x y, x < y → f x < f y := sorry
  exact sorry

end interval_of_increase_logb_l2042_204233


namespace min_a_value_l2042_204207

theorem min_a_value 
  (a x y : ℤ) 
  (h1 : x - y^2 = a) 
  (h2 : y - x^2 = a) 
  (h3 : x ≠ y) 
  (h4 : |x| ≤ 10) : 
  a = -111 :=
sorry

end min_a_value_l2042_204207


namespace number_of_yellow_highlighters_l2042_204215

-- Definitions based on the given conditions
def total_highlighters : Nat := 12
def pink_highlighters : Nat := 6
def blue_highlighters : Nat := 4

-- Statement to prove the question equals the correct answer given the conditions
theorem number_of_yellow_highlighters : 
  ∃ y : Nat, y = total_highlighters - (pink_highlighters + blue_highlighters) := 
by
  -- TODO: The proof will be filled in here
  sorry

end number_of_yellow_highlighters_l2042_204215


namespace fraction_sum_l2042_204273

theorem fraction_sum : ((10 : ℚ) / 9 + (9 : ℚ) / 10 = 2.0 + (0.1 + 0.1 / 9)) :=
by sorry

end fraction_sum_l2042_204273


namespace initial_overs_l2042_204285

theorem initial_overs {x : ℝ} (h1 : 4.2 * x + (83 / 15) * 30 = 250) : x = 20 :=
by
  sorry

end initial_overs_l2042_204285


namespace value_of_k_l2042_204272

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x : ℝ) (k : ℝ) : ℝ := 2 * x^2 - k * x + 7

theorem value_of_k (k : ℝ) : f 5 - g 5 k = 40 → k = 1.4 := by
  sorry

end value_of_k_l2042_204272


namespace find_y_l2042_204251

noncomputable def x : ℝ := (4 / 25)^(1 / 3)

theorem find_y (y : ℝ) (h1 : 0 < y) (h2 : y < x) (h3 : x^x = y^y) : y = (32 / 3125)^(1 / 3) :=
sorry

end find_y_l2042_204251


namespace star_polygon_points_l2042_204222

theorem star_polygon_points (p : ℕ) (ϕ : ℝ) :
  (∀ i : Fin p, ∃ Ci Di : ℝ, Ci = Di + 15) →
  (p * ϕ + p * (ϕ + 15) = 360) →
  p = 24 :=
by
  sorry

end star_polygon_points_l2042_204222


namespace men_with_tv_at_least_11_l2042_204241

-- Definitions for the given conditions
def total_men : ℕ := 100
def married_men : ℕ := 81
def men_with_radio : ℕ := 85
def men_with_ac : ℕ := 70
def men_with_tv_radio_ac_and_married : ℕ := 11

-- The proposition to prove the minimum number of men with TV
theorem men_with_tv_at_least_11 :
  ∃ (T : ℕ), T ≥ men_with_tv_radio_ac_and_married := 
by
  sorry

end men_with_tv_at_least_11_l2042_204241


namespace power_function_properties_l2042_204281

theorem power_function_properties (α : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ α) 
    (h_point : f (1/2) = 2 ) :
    (∀ x : ℝ, f x = 1 / x) ∧ (∀ x : ℝ, 0 < x → (f x) < (f (x / 2))) ∧ (∀ x : ℝ, f (-x) = - (f x)) :=
by
  sorry

end power_function_properties_l2042_204281


namespace sin_cos_sum_eq_one_or_neg_one_l2042_204237

theorem sin_cos_sum_eq_one_or_neg_one (α : ℝ) (h : (Real.sin α)^4 + (Real.cos α)^4 = 1) : (Real.sin α + Real.cos α) = 1 ∨ (Real.sin α + Real.cos α) = -1 :=
sorry

end sin_cos_sum_eq_one_or_neg_one_l2042_204237


namespace greater_number_l2042_204247

theorem greater_number (x y : ℕ) (h1 : x + y = 30) (h2 : x - y = 8) (h3 : x > y) : x = 19 := 
by 
  sorry

end greater_number_l2042_204247


namespace isosceles_triangle_base_angle_l2042_204255

theorem isosceles_triangle_base_angle (α β γ : ℝ) 
  (h_triangle: α + β + γ = 180) 
  (h_isosceles: α = β ∨ α = γ ∨ β = γ) 
  (h_one_angle: α = 80 ∨ β = 80 ∨ γ = 80) : 
  (α = 50 ∨ β = 50 ∨ γ = 50) ∨ (α = 80 ∨ β = 80 ∨ γ = 80) :=
by 
  sorry

end isosceles_triangle_base_angle_l2042_204255


namespace common_divisors_4n_7n_l2042_204201

theorem common_divisors_4n_7n (n : ℕ) (h1 : n < 50) 
    (h2 : (Nat.gcd (4 * n + 5) (7 * n + 6) > 1)) :
    n = 7 ∨ n = 18 ∨ n = 29 ∨ n = 40 := 
  sorry

end common_divisors_4n_7n_l2042_204201


namespace force_of_water_on_lock_wall_l2042_204288

noncomputable def force_on_the_wall (l h γ g : ℝ) : ℝ :=
  γ * g * l * (h^2 / 2)

theorem force_of_water_on_lock_wall :
  force_on_the_wall 20 5 1000 9.81 = 2.45 * 10^6 := by
  sorry

end force_of_water_on_lock_wall_l2042_204288


namespace directrix_of_parabola_l2042_204250

theorem directrix_of_parabola :
  ∀ (x : ℝ), y = x^2 / 4 → y = -1 :=
sorry

end directrix_of_parabola_l2042_204250


namespace modulus_of_z_l2042_204295

-- Define the complex number z
def z : ℂ := -5 + 12 * Complex.I

-- Define a theorem stating the modulus of z is 13
theorem modulus_of_z : Complex.abs z = 13 :=
by
  -- This will be the place to provide proof steps
  sorry

end modulus_of_z_l2042_204295


namespace am_gm_inequality_l2042_204254

theorem am_gm_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a^3 + b^3 + a + b ≥ 4 * a * b :=
by
  sorry

end am_gm_inequality_l2042_204254


namespace jessica_earned_from_washing_l2042_204262

-- Conditions defined as per Problem a)
def weekly_allowance : ℕ := 10
def spent_on_movies : ℕ := weekly_allowance / 2
def remaining_after_movies : ℕ := weekly_allowance - spent_on_movies
def final_amount : ℕ := 11
def earned_from_washing : ℕ := final_amount - remaining_after_movies

-- Lean statement to prove Jessica earned $6 from washing the family car
theorem jessica_earned_from_washing :
  earned_from_washing = 6 := 
by
  -- Proof to be filled in later (skipped here with sorry)
  sorry

end jessica_earned_from_washing_l2042_204262


namespace constant_COG_of_mercury_column_l2042_204294

theorem constant_COG_of_mercury_column (L : ℝ) (A : ℝ) (beta_g : ℝ) (beta_m : ℝ) (alpha_g : ℝ) (x : ℝ) :
  L = 1 ∧ A = 1e-4 ∧ beta_g = 1 / 38700 ∧ beta_m = 1 / 5550 ∧ alpha_g = beta_g / 3 ∧
  x = (2 / (3 * 38700)) / ((1 / 5550) - (2 / 116100)) →
  x = 0.106 :=
by
  sorry

end constant_COG_of_mercury_column_l2042_204294


namespace incenter_correct_l2042_204231

variable (P Q R : Type) [AddCommGroup P] [Module ℝ P]
variable (p q r : ℝ)
variable (P_vec Q_vec R_vec : P)

noncomputable def incenter_coordinates (p q r : ℝ) : ℝ × ℝ × ℝ :=
  (p / (p + q + r), q / (p + q + r), r / (p + q + r))

theorem incenter_correct : 
  incenter_coordinates 8 10 6 = (1/3, 5/12, 1/4) := by
  sorry

end incenter_correct_l2042_204231


namespace andrena_has_more_dolls_than_debelyn_l2042_204279

-- Definitions based on the given conditions
def initial_dolls_debelyn := 20
def initial_gift_debelyn_to_andrena := 2

def initial_dolls_christel := 24
def gift_christel_to_andrena := 5
def gift_christel_to_belissa := 3

def initial_dolls_belissa := 15
def gift_belissa_to_andrena := 4

-- Final number of dolls after exchanges
def final_dolls_debelyn := initial_dolls_debelyn - initial_gift_debelyn_to_andrena
def final_dolls_christel := initial_dolls_christel - gift_christel_to_andrena - gift_christel_to_belissa
def final_dolls_belissa := initial_dolls_belissa - gift_belissa_to_andrena + gift_christel_to_belissa
def final_dolls_andrena := initial_gift_debelyn_to_andrena + gift_christel_to_andrena + gift_belissa_to_andrena

-- Additional conditions
def andrena_more_than_christel := final_dolls_andrena = final_dolls_christel + 2
def belissa_equals_debelyn := final_dolls_belissa = final_dolls_debelyn

-- Proof Statement
theorem andrena_has_more_dolls_than_debelyn :
  andrena_more_than_christel →
  belissa_equals_debelyn →
  final_dolls_andrena - final_dolls_debelyn = 4 :=
by
  sorry

end andrena_has_more_dolls_than_debelyn_l2042_204279


namespace find_m_value_l2042_204238

theorem find_m_value (m : ℚ) :
  (m * 2 / 3 + m * 4 / 9 + m * 8 / 27 = 1) → m = 27 / 38 :=
by 
  intro h
  sorry

end find_m_value_l2042_204238


namespace gym_monthly_revenue_l2042_204265

theorem gym_monthly_revenue (members_per_month_fee : ℕ) (num_members : ℕ) 
  (h1 : members_per_month_fee = 18 * 2) 
  (h2 : num_members = 300) : 
  num_members * members_per_month_fee = 10800 := 
by 
  -- calculation rationale goes here
  sorry

end gym_monthly_revenue_l2042_204265


namespace complement_intersection_l2042_204284

def P : Set ℝ := {y | ∃ x, y = (1 / 2) ^ x ∧ 0 < x}
def Q : Set ℝ := {x | 0 < x ∧ x < 2}

theorem complement_intersection :
  (Set.univ \ P) ∩ Q = {x | 1 ≤ x ∧ x < 2} :=
sorry

end complement_intersection_l2042_204284


namespace scientific_notation_of_284000000_l2042_204287

/--
Given the number 284000000, prove that it can be expressed in scientific notation as 2.84 * 10^8.
-/
theorem scientific_notation_of_284000000 :
  284000000 = 2.84 * 10^8 :=
sorry

end scientific_notation_of_284000000_l2042_204287


namespace space_convex_polyhedron_euler_characteristic_l2042_204252

-- Definition of space convex polyhedron
structure Polyhedron where
  F : ℕ    -- number of faces
  V : ℕ    -- number of vertices
  E : ℕ    -- number of edges

-- Problem statement: Prove that for any space convex polyhedron, F + V - E = 2
theorem space_convex_polyhedron_euler_characteristic (P : Polyhedron) : P.F + P.V - P.E = 2 := by
  sorry

end space_convex_polyhedron_euler_characteristic_l2042_204252


namespace problem1_part1_problem1_part2_problem2_part1_problem2_part2_l2042_204248

open Set

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | -2 < x ∧ x < 5 }
def B : Set ℝ := { x | -1 ≤ x - 1 ∧ x - 1 ≤ 2 }

theorem problem1_part1 : A ∪ B = { x | -2 < x ∧ x < 5 } := sorry
theorem problem1_part2 : A ∩ B = { x | 0 ≤ x ∧ x ≤ 3 } := sorry

def B_c : Set ℝ := { x | x < 0 ∨ 3 < x }

theorem problem2_part1 : A ∪ B_c = U := sorry
theorem problem2_part2 : A ∩ B_c = { x | (-2 < x ∧ x < 0) ∨ (3 < x ∧ x < 5) } := sorry

end problem1_part1_problem1_part2_problem2_part1_problem2_part2_l2042_204248


namespace function_range_y_eq_1_div_x_minus_2_l2042_204267

theorem function_range_y_eq_1_div_x_minus_2 (x : ℝ) : (∀ y : ℝ, y = 1 / (x - 2) ↔ x ∈ {x : ℝ | x ≠ 2}) :=
sorry

end function_range_y_eq_1_div_x_minus_2_l2042_204267


namespace vector2d_propositions_l2042_204258

-- Define the vector structure in ℝ²
structure Vector2D where
  x : ℝ
  y : ℝ

-- Define the relation > on Vector2D
def Vector2D.gt (a1 a2 : Vector2D) : Prop :=
  a1.x > a2.x ∨ (a1.x = a2.x ∧ a1.y > a2.y)

-- Define vectors e1, e2, and 0
def e1 : Vector2D := ⟨ 1, 0 ⟩
def e2 : Vector2D := ⟨ 0, 1 ⟩
def zero : Vector2D := ⟨ 0, 0 ⟩

-- Define propositions
def prop1 : Prop := Vector2D.gt e1 e2 ∧ Vector2D.gt e2 zero
def prop2 (a1 a2 a3 : Vector2D) : Prop := Vector2D.gt a1 a2 → Vector2D.gt a2 a3 → Vector2D.gt a1 a3
def prop3 (a1 a2 a : Vector2D) : Prop := Vector2D.gt a1 a2 → Vector2D.gt (Vector2D.mk (a1.x + a.x) (a1.y + a.y)) (Vector2D.mk (a2.x + a.x) (a2.y + a.y))
def prop4 (a a1 a2 : Vector2D) : Prop := Vector2D.gt a zero → Vector2D.gt a1 a2 → Vector2D.gt (Vector2D.mk (a.x * a1.x + a.y * a1.y) (0)) (Vector2D.mk (a.x * a2.x + a.y * a2.y) 0)

-- Main theorem to prove
theorem vector2d_propositions : prop1 ∧ (∀ a1 a2 a3, prop2 a1 a2 a3) ∧ (∀ a1 a2 a, prop3 a1 a2 a) := 
by
  sorry

end vector2d_propositions_l2042_204258


namespace most_likely_outcome_is_D_l2042_204280

-- Define the basic probability of rolling any specific number with a fair die
def probability_of_specific_roll : ℚ := 1/6

-- Define the probability of each option
def P_A : ℚ := probability_of_specific_roll
def P_B : ℚ := 2 * probability_of_specific_roll
def P_C : ℚ := 3 * probability_of_specific_roll
def P_D : ℚ := 4 * probability_of_specific_roll

-- Define the proof problem statement
theorem most_likely_outcome_is_D : P_D = max P_A (max P_B (max P_C P_D)) :=
sorry

end most_likely_outcome_is_D_l2042_204280
