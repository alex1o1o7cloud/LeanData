import Mathlib

namespace NUMINAMATH_GPT_ratio_d_a_l2168_216851

theorem ratio_d_a (a b c d : ℝ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2) 
  (h3 : c / d = 5) : 
  d / a = 1 / 30 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_d_a_l2168_216851


namespace NUMINAMATH_GPT_sum_of_divisors_2000_l2168_216878

theorem sum_of_divisors_2000 (n : ℕ) (h : n < 2000) :
  ∃ (s : Finset ℕ), (s ⊆ {1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 1000, 2000}) ∧ s.sum id = n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sum_of_divisors_2000_l2168_216878


namespace NUMINAMATH_GPT_total_money_spent_on_clothing_l2168_216886

theorem total_money_spent_on_clothing (cost_shirt cost_jacket : ℝ)
  (h_shirt : cost_shirt = 13.04) (h_jacket : cost_jacket = 12.27) :
  cost_shirt + cost_jacket = 25.31 :=
sorry

end NUMINAMATH_GPT_total_money_spent_on_clothing_l2168_216886


namespace NUMINAMATH_GPT_solution_set_leq_2_l2168_216821

theorem solution_set_leq_2 (x y m n : ℤ)
  (h1 : m * 0 - n = 1)
  (h2 : m * 1 - n = 0)
  (h3 : y = m * x - n) :
  x ≥ -1 ↔ m * x - n ≤ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_set_leq_2_l2168_216821


namespace NUMINAMATH_GPT_largest_reciprocal_l2168_216844

theorem largest_reciprocal :
  let a := -1/2
  let b := 1/4
  let c := 0.5
  let d := 3
  let e := 10
  (1 / b) > (1 / a) ∧ (1 / b) > (1 / c) ∧ (1 / b) > (1 / d) ∧ (1 / b) > (1 / e) :=
by
  let a := -1/2
  let b := 1/4
  let c := 0.5
  let d := 3
  let e := 10
  sorry

end NUMINAMATH_GPT_largest_reciprocal_l2168_216844


namespace NUMINAMATH_GPT_school_children_count_l2168_216874

-- Define the conditions
variable (A P C B G : ℕ)
variable (A_eq : A = 160)
variable (kids_absent : ∀ (present kids absent children : ℕ), present = kids - absent → absent = 160)
variable (bananas_received : ∀ (two_per child kids : ℕ), (2 * kids) + (2 * 160) = 2 * 6400 + (4 * (6400 / 160)))
variable (boys_girls : B = 3 * G)

-- State the theorem
theorem school_children_count (C : ℕ) (A P B G : ℕ) 
  (A_eq : A = 160)
  (kids_absent : P = C - A)
  (bananas_received : (2 * P) + (2 * A) = 2 * P + (4 * (P / A)))
  (boys_girls : B = 3 * G)
  (total_bananas : 2 * P + 4 * (P / A) = 12960) :
  C = 6560 := 
sorry

end NUMINAMATH_GPT_school_children_count_l2168_216874


namespace NUMINAMATH_GPT_triangle_base_length_l2168_216815

theorem triangle_base_length (x : ℝ) :
  (∃ s : ℝ, 4 * s = 64 ∧ s * s = 256) ∧ (32 * x / 2 = 256) → x = 16 := by
  sorry

end NUMINAMATH_GPT_triangle_base_length_l2168_216815


namespace NUMINAMATH_GPT_min_value_of_function_l2168_216870

theorem min_value_of_function : 
  ∃ (c : ℝ), (∀ x : ℝ, (x ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 6)) → (2 * (Real.sin x) ^ 2 + 2 * Real.sin x - 1 / 2) ≥ c) ∧
             (∀ x : ℝ, (x ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 6)) → (2 * (Real.sin x) ^ 2 + 2 * Real.sin x - 1 / 2 = c) → c = 1) := 
sorry

end NUMINAMATH_GPT_min_value_of_function_l2168_216870


namespace NUMINAMATH_GPT_length_of_bridge_l2168_216806

noncomputable def speed_in_m_per_s (v_kmh : ℕ) : ℝ :=
  v_kmh * (1000 / 3600)

noncomputable def total_distance (v : ℝ) (t : ℝ) : ℝ :=
  v * t

theorem length_of_bridge (L_train : ℝ) (v_train_kmh : ℕ) (t : ℝ) (L_bridge : ℝ) :
  L_train = 288 →
  v_train_kmh = 29 →
  t = 48.29 →
  L_bridge = total_distance (speed_in_m_per_s v_train_kmh) t - L_train →
  L_bridge = 100.89 := by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l2168_216806


namespace NUMINAMATH_GPT_average_death_rate_l2168_216893

def birth_rate := 4 -- people every 2 seconds
def net_increase_per_day := 43200 -- people

def seconds_per_day := 86400 -- 24 * 60 * 60

def net_increase_per_second := net_increase_per_day / seconds_per_day -- people per second

def death_rate := (birth_rate / 2) - net_increase_per_second -- people per second

theorem average_death_rate :
  death_rate * 2 = 3 := by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_average_death_rate_l2168_216893


namespace NUMINAMATH_GPT_original_triangle_area_quadrupled_l2168_216866

theorem original_triangle_area_quadrupled {A : ℝ} (h1 : ∀ (a : ℝ), a > 0 → (a * 16 = 64)) : A = 4 :=
by
  have h1 : ∀ (a : ℝ), a > 0 → (a * 16 = 64) := by
    intro a ha
    sorry
  sorry

end NUMINAMATH_GPT_original_triangle_area_quadrupled_l2168_216866


namespace NUMINAMATH_GPT_advertising_department_size_l2168_216828

-- Define the conditions provided in the problem.
def total_employees : Nat := 1000
def sample_size : Nat := 80
def advertising_sample_size : Nat := 4

-- Define the main theorem to prove the given problem.
theorem advertising_department_size :
  ∃ n : Nat, (advertising_sample_size : ℚ) / n = (sample_size : ℚ) / total_employees ∧ n = 50 :=
by
  sorry

end NUMINAMATH_GPT_advertising_department_size_l2168_216828


namespace NUMINAMATH_GPT_abes_total_budget_l2168_216822

theorem abes_total_budget
    (B : ℝ)
    (h1 : B = (1/3) * B + (1/4) * B + 1250) :
    B = 3000 :=
sorry

end NUMINAMATH_GPT_abes_total_budget_l2168_216822


namespace NUMINAMATH_GPT_triangle_is_isosceles_right_l2168_216817

theorem triangle_is_isosceles_right
  (a b c : ℝ)
  (A B C : ℕ)
  (h1 : c = a * Real.cos B)
  (h2 : b = a * Real.sin C) :
  C = 90 ∧ B = 90 ∧ A = 90 :=
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_right_l2168_216817


namespace NUMINAMATH_GPT_phi_range_l2168_216862

noncomputable def f (ω φ x : ℝ) : ℝ :=
  2 * Real.sin (ω * x + φ) + 1

theorem phi_range (ω φ : ℝ) 
  (h₀ : ω > 0)
  (h₁ : |φ| ≤ Real.pi / 2)
  (h₂ : ∃ x₁ x₂, x₁ ≠ x₂ ∧ f ω φ x₁ = 2 ∧ f ω φ x₂ = 2 ∧ |x₂ - x₁| = Real.pi / 3)
  (h₃ : ∀ x, x ∈ Set.Ioo (-Real.pi / 8) (Real.pi / 3) → f ω φ x > 1) :
  φ ∈ Set.Icc (Real.pi / 4) (Real.pi / 3) :=
sorry

end NUMINAMATH_GPT_phi_range_l2168_216862


namespace NUMINAMATH_GPT_technician_round_trip_l2168_216861

theorem technician_round_trip (D : ℝ) (hD : D > 0) :
  let round_trip := 2 * D
  let to_center := D
  let from_center_percent := 0.3 * D
  let traveled_distance := to_center + from_center_percent
  (traveled_distance / round_trip * 100) = 65 := by
  -- Definitions based on the given conditions
  let round_trip := 2 * D
  let to_center := D
  let from_center_percent := 0.3 * D
  let traveled_distance := to_center + from_center_percent
  
  -- Placeholder for the proof to satisfy Lean syntax.
  sorry

end NUMINAMATH_GPT_technician_round_trip_l2168_216861


namespace NUMINAMATH_GPT_ball_cost_l2168_216882

theorem ball_cost (B C : ℝ) (h1 : 7 * B + 6 * C = 3800) (h2 : 3 * B + 5 * C = 1750) (hb : B = 500) : C = 50 :=
by
  sorry

end NUMINAMATH_GPT_ball_cost_l2168_216882


namespace NUMINAMATH_GPT_probability_G_is_one_fourth_l2168_216814

-- Definitions and conditions
variables (p_E p_F p_G p_H : ℚ)
axiom probability_E : p_E = 1/3
axiom probability_F : p_F = 1/6
axiom prob_G_eq_H : p_G = p_H
axiom total_prob_sum : p_E + p_F + p_G + p_G = 1

-- Theorem statement
theorem probability_G_is_one_fourth : p_G = 1/4 :=
by 
  -- Lean proof omitted, only the statement required
  sorry

end NUMINAMATH_GPT_probability_G_is_one_fourth_l2168_216814


namespace NUMINAMATH_GPT_measure_angle_A_l2168_216890

theorem measure_angle_A (a b c : ℝ) (A B C : ℝ)
  (h1 : ∀ (Δ : Type), Δ → Δ → Δ)
  (h2 : a / Real.cos A = b / (2 * Real.cos B) ∧ 
        a / Real.cos A = c / (3 * Real.cos C))
  (h3 : A + B + C = Real.pi) : 
  A = Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_measure_angle_A_l2168_216890


namespace NUMINAMATH_GPT_degrees_for_salaries_l2168_216892

def transportation_percent : ℕ := 15
def research_development_percent : ℕ := 9
def utilities_percent : ℕ := 5
def equipment_percent : ℕ := 4
def supplies_percent : ℕ := 2
def total_percent : ℕ := 100
def total_degrees : ℕ := 360

theorem degrees_for_salaries :
  total_degrees * (total_percent - (transportation_percent + research_development_percent + utilities_percent + equipment_percent + supplies_percent)) / total_percent = 234 := 
by
  sorry

end NUMINAMATH_GPT_degrees_for_salaries_l2168_216892


namespace NUMINAMATH_GPT_commute_proof_l2168_216872

noncomputable def commute_problem : Prop :=
  let d : ℝ := 1.5 -- distance in miles
  let v_w : ℝ := 3 -- walking speed in miles per hour
  let v_t : ℝ := 20 -- train speed in miles per hour
  let walking_minutes : ℝ := (d / v_w) * 60 -- walking time in minutes
  let train_minutes : ℝ := (d / v_t) * 60 -- train time in minutes
  ∃ x : ℝ, walking_minutes = train_minutes + x + 25 ∧ x = 0.5

theorem commute_proof : commute_problem :=
  sorry

end NUMINAMATH_GPT_commute_proof_l2168_216872


namespace NUMINAMATH_GPT_members_playing_both_badminton_and_tennis_l2168_216805

-- Definitions based on conditions
def N : ℕ := 35  -- Total number of members in the sports club
def B : ℕ := 15  -- Number of people who play badminton
def T : ℕ := 18  -- Number of people who play tennis
def Neither : ℕ := 5  -- Number of people who do not play either sport

-- The theorem based on the inclusion-exclusion principle
theorem members_playing_both_badminton_and_tennis :
  (B + T - (N - Neither) = 3) :=
by
  sorry

end NUMINAMATH_GPT_members_playing_both_badminton_and_tennis_l2168_216805


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_S15_l2168_216836

theorem arithmetic_sequence_sum_S15 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hs5 : S 5 = 10) (hs10 : S 10 = 30) 
  (has : ∀ n, S n = n * (2 * a 1 + (n - 1) * a 2) / 2) : 
  S 15 = 60 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_S15_l2168_216836


namespace NUMINAMATH_GPT_journey_time_equality_l2168_216854

variables {v : ℝ} (h : v > 0)

theorem journey_time_equality (v : ℝ) (hv : v > 0) :
  let t1 := 80 / v
  let t2 := 160 / (2 * v)
  t1 = t2 :=
by
  sorry

end NUMINAMATH_GPT_journey_time_equality_l2168_216854


namespace NUMINAMATH_GPT_unattainable_y_value_l2168_216846

theorem unattainable_y_value :
  ∀ (y x : ℝ), (y = (1 - x) / (2 * x^2 + 3 * x + 4)) → (∀ x, 2 * x^2 + 3 * x + 4 ≠ 0) → y ≠ 0 :=
by
  intros y x h1 h2
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_unattainable_y_value_l2168_216846


namespace NUMINAMATH_GPT_michaels_brother_money_end_l2168_216810

theorem michaels_brother_money_end 
  (michael_money : ℕ)
  (brother_money : ℕ)
  (gives_half : ℕ)
  (buys_candy : ℕ) 
  (h1 : michael_money = 42)
  (h2 : brother_money = 17)
  (h3 : gives_half = michael_money / 2)
  (h4 : buys_candy = 3) : 
  brother_money + gives_half - buys_candy = 35 :=
by {
  sorry
}

end NUMINAMATH_GPT_michaels_brother_money_end_l2168_216810


namespace NUMINAMATH_GPT_sequence_a_10_l2168_216881

theorem sequence_a_10 (a : ℕ → ℤ) 
  (H1 : ∀ p q : ℕ, p > 0 → q > 0 → a (p + q) = a p + a q)
  (H2 : a 2 = -6) : 
  a 10 = -30 :=
sorry

end NUMINAMATH_GPT_sequence_a_10_l2168_216881


namespace NUMINAMATH_GPT_clock_chime_time_l2168_216832

theorem clock_chime_time (t_5oclock : ℕ) (n_5chimes : ℕ) (t_10oclock : ℕ) (n_10chimes : ℕ)
  (h1: t_5oclock = 8) (h2: n_5chimes = 5) (h3: n_10chimes = 10) : 
  t_10oclock = 18 :=
by
  sorry

end NUMINAMATH_GPT_clock_chime_time_l2168_216832


namespace NUMINAMATH_GPT_find_number_of_students_l2168_216891

-- Parameters
variable (n : ℕ) (C : ℕ)
def first_and_last_picked_by_sam (n : ℕ) (C : ℕ) : Prop := 
  C + 1 = 2 * n

-- Conditions: number of candies is 120, the bag completes 2 full rounds at the table.
theorem find_number_of_students
  (C : ℕ) (h_C: C = 120) (h_rounds: 2 * n = C):
  n = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_students_l2168_216891


namespace NUMINAMATH_GPT_fraction_division_l2168_216838

theorem fraction_division : (3/4) / (5/8) = (6/5) := by
  sorry

end NUMINAMATH_GPT_fraction_division_l2168_216838


namespace NUMINAMATH_GPT_roots_transformation_l2168_216820

noncomputable def poly_with_roots (r₁ r₂ r₃ : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 3 - 5 * Polynomial.X ^ 2 + 10

noncomputable def transformed_poly_with_roots (r₁ r₂ r₃ : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 3 - 15 * Polynomial.X ^ 2 + 270

theorem roots_transformation (r₁ r₂ r₃ : ℝ) (h : poly_with_roots r₁ r₂ r₃ = 0) :
  transformed_poly_with_roots (3 * r₁) (3 * r₂) (3 * r₃) = Polynomial.X ^ 3 - 15 * Polynomial.X ^ 2 + 270 :=
by
  sorry

end NUMINAMATH_GPT_roots_transformation_l2168_216820


namespace NUMINAMATH_GPT_find_width_of_first_tract_l2168_216800

-- Definitions based on given conditions
noncomputable def area_first_tract (W : ℝ) : ℝ := 300 * W
def area_second_tract : ℝ := 250 * 630
def combined_area : ℝ := 307500

-- The theorem we need to prove: width of the first tract is 500 meters
theorem find_width_of_first_tract (W : ℝ) (h : area_first_tract W + area_second_tract = combined_area) : W = 500 :=
by
  sorry

end NUMINAMATH_GPT_find_width_of_first_tract_l2168_216800


namespace NUMINAMATH_GPT_john_spent_15_dollars_on_soap_l2168_216829

-- Define the number of soap bars John bought
def num_bars : ℕ := 20

-- Define the weight of each bar of soap in pounds
def weight_per_bar : ℝ := 1.5

-- Define the cost per pound of soap in dollars
def cost_per_pound : ℝ := 0.5

-- Total weight of the soap in pounds
def total_weight : ℝ := num_bars * weight_per_bar

-- Total cost of the soap in dollars
def total_cost : ℝ := total_weight * cost_per_pound

-- Statement to prove
theorem john_spent_15_dollars_on_soap : total_cost = 15 :=
by sorry

end NUMINAMATH_GPT_john_spent_15_dollars_on_soap_l2168_216829


namespace NUMINAMATH_GPT_third_median_length_is_9_l2168_216837

noncomputable def length_of_third_median_of_triangle (m₁ m₂ m₃ area : ℝ) : Prop :=
  ∃ median : ℝ, median = m₃

theorem third_median_length_is_9 :
  length_of_third_median_of_triangle 5 7 9 (6 * Real.sqrt 10) :=
by
  sorry

end NUMINAMATH_GPT_third_median_length_is_9_l2168_216837


namespace NUMINAMATH_GPT_prob_first_two_same_color_expected_value_eta_l2168_216813

-- Definitions and conditions
def num_white : ℕ := 4
def num_black : ℕ := 3
def total_pieces : ℕ := num_white + num_black

-- Probability of drawing two pieces of the same color
def prob_same_color : ℚ :=
  (4/7 * 3/6) + (3/7 * 2/6)

-- Expected value of the number of white pieces drawn in the first four draws
def E_eta : ℚ :=
  1 * (4 / 35) + 2 * (18 / 35) + 3 * (12 / 35) + 4 * (1 / 35)

-- Proof statements
theorem prob_first_two_same_color : prob_same_color = 3 / 7 :=
  by sorry

theorem expected_value_eta : E_eta = 16 / 7 :=
  by sorry

end NUMINAMATH_GPT_prob_first_two_same_color_expected_value_eta_l2168_216813


namespace NUMINAMATH_GPT_median_avg_scores_compare_teacher_avg_scores_l2168_216839

-- Definitions of conditions
def class1_students (a : ℕ) := a
def class2_students (b : ℕ) := b
def class3_students (c : ℕ) := c
def class4_students (c : ℕ) := c

def avg_score_1 := 68
def avg_score_2 := 78
def avg_score_3 := 74
def avg_score_4 := 72

-- Part 1: Prove the median of the average scores.
theorem median_avg_scores : 
  let scores := [68, 72, 74, 78]
  ∃ m, m = 73 :=
by 
  sorry

-- Part 2: Prove that the average scores for Teacher Wang and Teacher Li are not necessarily the same.
theorem compare_teacher_avg_scores (a b c : ℕ) (h_ab : a ≠ 0 ∧ b ≠ 0) : 
  let wang_avg := (68 * a + 78 * b) / (a + b)
  let li_avg := 73
  wang_avg ≠ li_avg :=
by
  sorry

end NUMINAMATH_GPT_median_avg_scores_compare_teacher_avg_scores_l2168_216839


namespace NUMINAMATH_GPT_product_of_points_l2168_216808

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 ∧ n % 2 ≠ 0 then 8
  else if n % 2 = 0 ∧ n % 3 ≠ 0 then 3
  else 0

def Chris_rolls : List ℕ := [5, 2, 1, 6]
def Dana_rolls : List ℕ := [6, 2, 3, 3]

def Chris_points : ℕ := (Chris_rolls.map f).sum
def Dana_points : ℕ := (Dana_rolls.map f).sum

theorem product_of_points : Chris_points * Dana_points = 297 := by
  sorry

end NUMINAMATH_GPT_product_of_points_l2168_216808


namespace NUMINAMATH_GPT_planes_parallel_l2168_216895

-- Given definitions and conditions
variables {Line Plane : Type}
variables (a b : Line) (α β γ : Plane)

-- Conditions from the problem
axiom perp_line_plane (line : Line) (plane : Plane) : Prop
axiom parallel_line_plane (line : Line) (plane : Plane) : Prop
axiom parallel_plane_plane (plane1 plane2 : Plane) : Prop

-- Conditions
variable (h1 : parallel_plane_plane γ α)
variable (h2 : parallel_plane_plane γ β)

-- Proof statement
theorem planes_parallel (h1 : parallel_plane_plane γ α) (h2 : parallel_plane_plane γ β) : parallel_plane_plane α β := sorry

end NUMINAMATH_GPT_planes_parallel_l2168_216895


namespace NUMINAMATH_GPT_dalton_movies_l2168_216865

variable (D : ℕ) -- Dalton's movies
variable (Hunter : ℕ := 12) -- Hunter's movies
variable (Alex : ℕ := 15) -- Alex's movies
variable (Together : ℕ := 2) -- Movies watched together
variable (TotalDifferentMovies : ℕ := 30) -- Total different movies

theorem dalton_movies (h : D + Hunter + Alex - Together * 3 = TotalDifferentMovies) : D = 9 := by
  sorry

end NUMINAMATH_GPT_dalton_movies_l2168_216865


namespace NUMINAMATH_GPT_original_salary_l2168_216887

def final_salary_after_changes (S : ℝ) : ℝ :=
  let increased_10 := S * 1.10
  let promoted_8 := increased_10 * 1.08
  let deducted_5 := promoted_8 * 0.95
  let decreased_7 := deducted_5 * 0.93
  decreased_7

theorem original_salary (S : ℝ) (h : final_salary_after_changes S = 6270) : S = 5587.68 :=
by
  -- Proof to be completed here
  sorry

end NUMINAMATH_GPT_original_salary_l2168_216887


namespace NUMINAMATH_GPT_correct_exponent_operation_l2168_216899

theorem correct_exponent_operation (a : ℝ) : a^4 / a^3 = a := 
by
  sorry

end NUMINAMATH_GPT_correct_exponent_operation_l2168_216899


namespace NUMINAMATH_GPT_find_total_stock_worth_l2168_216873

noncomputable def total_stock_worth (X : ℝ) : Prop :=
  let profit := 0.10 * (0.20 * X)
  let loss := 0.05 * (0.80 * X)
  loss - profit = 450

theorem find_total_stock_worth (X : ℝ) (h : total_stock_worth X) : X = 22500 :=
by
  sorry

end NUMINAMATH_GPT_find_total_stock_worth_l2168_216873


namespace NUMINAMATH_GPT_circle_condition_l2168_216858

theorem circle_condition (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2*x - 4*y + m = 0) → m < 5 :=
by
  -- Define constants and equation representation
  let d : ℝ := -2
  let e : ℝ := -4
  let f : ℝ := m
  -- Use the condition for the circle equation
  have h : d^2 + e^2 - 4*f > 0 := sorry
  -- Prove the inequality
  sorry

end NUMINAMATH_GPT_circle_condition_l2168_216858


namespace NUMINAMATH_GPT_range_of_fraction_l2168_216884

theorem range_of_fraction (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 4) (hy : 3 ≤ y ∧ y ≤ 6) :
  ∀ z, z = x / y → (1 / 6 ≤ z ∧ z ≤ 4 / 3) :=
sorry

end NUMINAMATH_GPT_range_of_fraction_l2168_216884


namespace NUMINAMATH_GPT_area_of_rectangle_l2168_216849

def length_fence (x : ℝ) : ℝ := 2 * x + 2 * x

theorem area_of_rectangle (x : ℝ) (h : length_fence x = 150) : x * 2 * x = 2812.5 :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l2168_216849


namespace NUMINAMATH_GPT_average_score_of_class_l2168_216867

theorem average_score_of_class (n : ℕ) (k : ℕ) (jimin_score : ℕ) (jungkook_score : ℕ) (avg_others : ℕ) 
  (total_students : n = 40) (excluding_students : k = 38) 
  (avg_excluding_others : avg_others = 79) 
  (jimin : jimin_score = 98) 
  (jungkook : jungkook_score = 100) : 
  (98 + 100 + (38 * 79)) / 40 = 80 :=
sorry

end NUMINAMATH_GPT_average_score_of_class_l2168_216867


namespace NUMINAMATH_GPT_sum_of_solutions_eq_0_l2168_216818

-- Define the conditions
def y : ℝ := 6
def main_eq (x : ℝ) : Prop := x^2 + y^2 = 145

-- State the theorem
theorem sum_of_solutions_eq_0 : 
  let x1 := Real.sqrt 109
  let x2 := -Real.sqrt 109
  x1 + x2 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_solutions_eq_0_l2168_216818


namespace NUMINAMATH_GPT_find_range_of_x_l2168_216894

-- Conditions
variable (f : ℝ → ℝ)
variable (even_f : ∀ x : ℝ, f x = f (-x))
variable (mono_incr_f : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)

-- Equivalent proof statement
theorem find_range_of_x (x : ℝ) :
  f (Real.log (abs (x + 1)) / Real.log (1 / 2)) < f (-1) ↔ x ∈ Set.Ioo (-3 : ℝ) (-3 / 2) ∪ Set.Ioo (-1 / 2) 1 := by
  sorry

end NUMINAMATH_GPT_find_range_of_x_l2168_216894


namespace NUMINAMATH_GPT_max_value_of_a_l2168_216819

theorem max_value_of_a {a : ℝ} (h : ∀ x ≥ 1, -3 * x^2 + a ≤ 0) : a ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_value_of_a_l2168_216819


namespace NUMINAMATH_GPT_larger_number_is_33_l2168_216841

theorem larger_number_is_33 (x y : ℤ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : max x y = 33 :=
sorry

end NUMINAMATH_GPT_larger_number_is_33_l2168_216841


namespace NUMINAMATH_GPT_landscape_length_l2168_216827

theorem landscape_length (b l : ℕ) (playground_area : ℕ) (total_area : ℕ) 
  (h1 : l = 4 * b) (h2 : playground_area = 1200) (h3 : total_area = 3 * playground_area) (h4 : total_area = l * b) :
  l = 120 := 
by 
  sorry

end NUMINAMATH_GPT_landscape_length_l2168_216827


namespace NUMINAMATH_GPT_find_angle_A_l2168_216869

theorem find_angle_A 
  (a b c A B C : ℝ)
  (h₀ : a = Real.sqrt 2)
  (h₁ : b = 2)
  (h₂ : Real.sin B - Real.cos B = Real.sqrt 2)
  (h₃ : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
  : A = Real.pi / 6 := 
  sorry

end NUMINAMATH_GPT_find_angle_A_l2168_216869


namespace NUMINAMATH_GPT_largest_of_sums_l2168_216876

noncomputable def a1 := (1 / 4 : ℚ) + (1 / 5 : ℚ)
noncomputable def a2 := (1 / 4 : ℚ) + (1 / 6 : ℚ)
noncomputable def a3 := (1 / 4 : ℚ) + (1 / 3 : ℚ)
noncomputable def a4 := (1 / 4 : ℚ) + (1 / 8 : ℚ)
noncomputable def a5 := (1 / 4 : ℚ) + (1 / 7 : ℚ)

theorem largest_of_sums :
  max a1 (max a2 (max a3 (max a4 a5))) = 7 / 12 :=
by sorry

end NUMINAMATH_GPT_largest_of_sums_l2168_216876


namespace NUMINAMATH_GPT_fourth_number_pascal_row_l2168_216830

theorem fourth_number_pascal_row : (Nat.choose 12 3) = 220 := sorry

end NUMINAMATH_GPT_fourth_number_pascal_row_l2168_216830


namespace NUMINAMATH_GPT_mult_closest_l2168_216859

theorem mult_closest :
  0.0004 * 9000000 = 3600 := sorry

end NUMINAMATH_GPT_mult_closest_l2168_216859


namespace NUMINAMATH_GPT_compute_2a_minus_b_l2168_216856

noncomputable def conditions (a b : ℝ) : Prop :=
  a^3 - 12 * a^2 + 47 * a - 60 = 0 ∧
  -b^3 + 12 * b^2 - 47 * b + 180 = 0

theorem compute_2a_minus_b (a b : ℝ) (h : conditions a b) : 2 * a - b = 2 := 
  sorry

end NUMINAMATH_GPT_compute_2a_minus_b_l2168_216856


namespace NUMINAMATH_GPT_central_cell_value_l2168_216834

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end NUMINAMATH_GPT_central_cell_value_l2168_216834


namespace NUMINAMATH_GPT_volume_of_inscribed_cubes_l2168_216868

noncomputable def tetrahedron_cube_volume (a m : ℝ) : ℝ × ℝ :=
  let V1 := (a * m / (a + m))^3
  let V2 := (a * m / (a + (Real.sqrt 2) * m))^3
  (V1, V2)

theorem volume_of_inscribed_cubes (a m : ℝ) (ha : 0 < a) (hm : 0 < m) :
  tetrahedron_cube_volume a m = 
  ( (a * m / (a + m))^3, 
    (a * m / (a + (Real.sqrt 2) * m))^3 ) :=
  by
    sorry

end NUMINAMATH_GPT_volume_of_inscribed_cubes_l2168_216868


namespace NUMINAMATH_GPT_find_valid_triples_l2168_216811

-- Define the theorem to prove the conditions and results
theorem find_valid_triples :
  ∀ (a b c : ℕ), 
    (2^a + 2^b + 1) % (2^c - 1) = 0 ↔ (a = 0 ∧ b = 0 ∧ c = 2) ∨ 
                                      (a = 1 ∧ b = 2 ∧ c = 3) ∨ 
                                      (a = 2 ∧ b = 1 ∧ c = 3) := 
sorry  -- Proof omitted

end NUMINAMATH_GPT_find_valid_triples_l2168_216811


namespace NUMINAMATH_GPT_find_k_l2168_216835

variables (l w : ℝ) (p A k : ℝ)

def rectangle_conditions : Prop :=
  (l / w = 5 / 2) ∧ (p = 2 * (l + w))

theorem find_k (h : rectangle_conditions l w p) :
  A = (5 / 98) * p^2 :=
sorry

end NUMINAMATH_GPT_find_k_l2168_216835


namespace NUMINAMATH_GPT_cost_per_set_l2168_216888

variable {C : ℝ} -- Define the variable cost per set.

theorem cost_per_set
  (initial_outlay : ℝ := 10000) -- Initial outlay for manufacturing.
  (revenue_per_set : ℝ := 50) -- Revenue per set sold.
  (sets_sold : ℝ := 500) -- Sets produced and sold.
  (profit : ℝ := 5000) -- Profit from selling 500 sets.

  (h_profit_eq : profit = (revenue_per_set * sets_sold) - (initial_outlay + C * sets_sold)) :
  C = 20 :=
by
  -- Proof to be filled in later.
  sorry

end NUMINAMATH_GPT_cost_per_set_l2168_216888


namespace NUMINAMATH_GPT_exposed_circular_segment_sum_l2168_216864

theorem exposed_circular_segment_sum (r h : ℕ) (angle : ℕ) (a b c : ℕ) :
    r = 8 ∧ h = 10 ∧ angle = 90 ∧ a = 16 ∧ b = 0 ∧ c = 0 → a + b + c = 16 :=
by
  intros
  sorry

end NUMINAMATH_GPT_exposed_circular_segment_sum_l2168_216864


namespace NUMINAMATH_GPT_share_of_a_l2168_216896

variables {a b c d : ℝ}
variables {total : ℝ}

-- Conditions
def condition1 (a b c d : ℝ) := a = (3/5) * (b + c + d)
def condition2 (a b c d : ℝ) := b = (2/3) * (a + c + d)
def condition3 (a b c d : ℝ) := c = (4/7) * (a + b + d)
def total_distributed (a b c d : ℝ) := a + b + c + d = 1200

-- Theorem to prove
theorem share_of_a (a b c d : ℝ) (h1 : condition1 a b c d) (h2 : condition2 a b c d) (h3 : condition3 a b c d) (h4 : total_distributed a b c d) : 
  a = 247.5 :=
sorry

end NUMINAMATH_GPT_share_of_a_l2168_216896


namespace NUMINAMATH_GPT_tables_made_this_month_l2168_216842

theorem tables_made_this_month (T : ℕ) 
  (h1: ∀ t, t = T → t - 3 < t) 
  (h2 : T + (T - 3) = 17) :
  T = 10 := by
  sorry

end NUMINAMATH_GPT_tables_made_this_month_l2168_216842


namespace NUMINAMATH_GPT_andrea_living_room_area_l2168_216860

/-- Given that 60% of Andrea's living room floor is covered by a carpet 
     which has dimensions 4 feet by 9 feet, prove that the area of 
     Andrea's living room floor is 60 square feet. -/
theorem andrea_living_room_area :
  ∃ A, (0.60 * A = 4 * 9) ∧ A = 60 :=
by
  sorry

end NUMINAMATH_GPT_andrea_living_room_area_l2168_216860


namespace NUMINAMATH_GPT_shaded_area_difference_l2168_216816

theorem shaded_area_difference (A1 A3 A4 : ℚ) (h1 : 4 = 2 * 2) (h2 : A1 + 5 * A1 + 7 * A1 = 6) (h3 : p + q = 49) : 
  ∃ p q : ℕ, p + q = 49 ∧ p = 36 ∧ q = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_shaded_area_difference_l2168_216816


namespace NUMINAMATH_GPT_Walter_allocates_for_school_l2168_216871

open Nat

def Walter_works_5_days_a_week := 5
def Walter_earns_per_hour := 5
def Walter_works_per_day := 4
def Proportion_for_school := 3/4

theorem Walter_allocates_for_school :
  let daily_earnings := Walter_works_per_day * Walter_earns_per_hour
  let weekly_earnings := daily_earnings * Walter_works_5_days_a_week
  let school_allocation := weekly_earnings * Proportion_for_school
  school_allocation = 75 := by
  sorry

end NUMINAMATH_GPT_Walter_allocates_for_school_l2168_216871


namespace NUMINAMATH_GPT_sin_cos_eq_one_sol_set_l2168_216853

-- Define the interval
def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x < 2 * Real.pi

-- Define the condition
def satisfies_eq (x : ℝ) : Prop := Real.sin x + Real.cos x = 1

-- Theorem statement: prove that the solution set is {0, π/2}
theorem sin_cos_eq_one_sol_set :
  ∀ (x : ℝ), in_interval x → satisfies_eq x ↔ x = 0 ∨ x = Real.pi / 2 := by
  sorry

end NUMINAMATH_GPT_sin_cos_eq_one_sol_set_l2168_216853


namespace NUMINAMATH_GPT_smallest_n_for_Tn_gt_2006_over_2016_l2168_216809

-- Definitions from the given problem
def Sn (n : ℕ) : ℚ := n^2 / (n + 1)
def an (n : ℕ) : ℚ := if n = 1 then 1 / 2 else Sn n - Sn (n - 1)
def bn (n : ℕ) : ℚ := an n / (n^2 + n - 1)

-- Definition of Tn sum
def Tn (n : ℕ) : ℚ := (Finset.range n).sum (λ k => bn (k + 1))

-- The main statement
theorem smallest_n_for_Tn_gt_2006_over_2016 : ∃ n : ℕ, Tn n > 2006 / 2016 := by
  sorry

end NUMINAMATH_GPT_smallest_n_for_Tn_gt_2006_over_2016_l2168_216809


namespace NUMINAMATH_GPT_ending_number_of_sequence_divisible_by_11_l2168_216833

theorem ending_number_of_sequence_divisible_by_11 : 
  ∃ (n : ℕ), 19 < n ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 6 → n = 19 + 11 * k) ∧ n = 77 :=
by
  sorry

end NUMINAMATH_GPT_ending_number_of_sequence_divisible_by_11_l2168_216833


namespace NUMINAMATH_GPT_total_poles_needed_l2168_216889

theorem total_poles_needed (longer_side_poles : ℕ) (shorter_side_poles : ℕ) (internal_fence_poles : ℕ) :
  longer_side_poles = 35 → 
  shorter_side_poles = 27 → 
  internal_fence_poles = (shorter_side_poles - 1) → 
  ((longer_side_poles * 2) + (shorter_side_poles * 2) - 4 + internal_fence_poles) = 146 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_total_poles_needed_l2168_216889


namespace NUMINAMATH_GPT_exchange_yen_for_yuan_l2168_216875

-- Define the condition: 100 Japanese yen could be exchanged for 7.2 yuan
def exchange_rate : ℝ := 7.2
def yen_per_100_yuan : ℝ := 100

-- Define the amount in yuan we want to exchange
def yuan_amount : ℝ := 720

-- The mathematical assertion (proof problem)
theorem exchange_yen_for_yuan : 
  (yuan_amount / exchange_rate) * yen_per_100_yuan = 10000 :=
by
  sorry

end NUMINAMATH_GPT_exchange_yen_for_yuan_l2168_216875


namespace NUMINAMATH_GPT_sum_of_powers_l2168_216879

theorem sum_of_powers (a b : ℝ) (h1 : a^2 - b^2 = 8) (h2 : a * b = 2) : a^4 + b^4 = 72 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_powers_l2168_216879


namespace NUMINAMATH_GPT_find_value_of_c_l2168_216804

theorem find_value_of_c (c : ℝ) : (∀ x : ℝ, (-x^2 + c * x + 8 > 0 ↔ x < -2 ∨ x > 4)) → c = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_c_l2168_216804


namespace NUMINAMATH_GPT_sum_of_solutions_l2168_216885

theorem sum_of_solutions (x : ℝ) : 
  (x^2 - 5*x - 26 = 4*x + 21) → 
  (∃ S, S = 9 ∧ ∀ x1 x2, x1 + x2 = S) := by
  intros h
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l2168_216885


namespace NUMINAMATH_GPT_quadratic_function_m_value_l2168_216843

theorem quadratic_function_m_value
  (m : ℝ)
  (h1 : m^2 - 7 = 2)
  (h2 : 3 - m ≠ 0) :
  m = -3 := by
  sorry

end NUMINAMATH_GPT_quadratic_function_m_value_l2168_216843


namespace NUMINAMATH_GPT_problem_statement_l2168_216845

theorem problem_statement : (515 % 1000) = 515 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2168_216845


namespace NUMINAMATH_GPT_circumcircle_incircle_inequality_l2168_216850

theorem circumcircle_incircle_inequality
  (a b : ℝ)
  (h_a : a = 16)
  (h_b : b = 11)
  (R r : ℝ)
  (triangle_inequality : ∀ c : ℝ, 5 < c ∧ c < 27) :
  R ≥ 2.2 * r := sorry

end NUMINAMATH_GPT_circumcircle_incircle_inequality_l2168_216850


namespace NUMINAMATH_GPT_can_all_mushrooms_become_good_l2168_216812

def is_bad (w : Nat) : Prop := w ≥ 10
def is_good (w : Nat) : Prop := w < 10

def mushrooms_initially_bad := 90
def mushrooms_initially_good := 10

def total_mushrooms := mushrooms_initially_bad + mushrooms_initially_good
def total_worms_initial := mushrooms_initially_bad * 10

theorem can_all_mushrooms_become_good :
  ∃ worms_distribution : Fin total_mushrooms → Nat,
  (∀ i : Fin total_mushrooms, is_good (worms_distribution i)) :=
sorry

end NUMINAMATH_GPT_can_all_mushrooms_become_good_l2168_216812


namespace NUMINAMATH_GPT_symmetric_point_x_axis_l2168_216857

variable (P : (ℝ × ℝ)) (x : ℝ) (y : ℝ)

-- Given P is a point (x, y)
def symmetric_about_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

-- Special case for the point (-2, 3)
theorem symmetric_point_x_axis : 
  symmetric_about_x_axis (-2, 3) = (-2, -3) :=
by 
  sorry

end NUMINAMATH_GPT_symmetric_point_x_axis_l2168_216857


namespace NUMINAMATH_GPT_intersection_M_N_l2168_216883

open Set

def M : Set ℝ := { x | -4 < x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l2168_216883


namespace NUMINAMATH_GPT_wifes_raise_l2168_216807

variable (D W : ℝ)
variable (h1 : 0.08 * D = 800)
variable (h2 : 1.08 * D - 1.08 * W = 540)

theorem wifes_raise : 0.08 * W = 760 :=
by
  sorry

end NUMINAMATH_GPT_wifes_raise_l2168_216807


namespace NUMINAMATH_GPT_symmetric_points_y_axis_l2168_216855

theorem symmetric_points_y_axis (a b : ℝ) (h1 : a - b = -3) (h2 : 2 * a + b = 2) :
  a = -1 / 3 ∧ b = 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_points_y_axis_l2168_216855


namespace NUMINAMATH_GPT_find_range_of_a_l2168_216880

variable {f : ℝ → ℝ}
noncomputable def domain_f : Set ℝ := {x | 7 ≤ x ∧ x < 15}
noncomputable def domain_f_2x_plus_1 : Set ℝ := {x | 3 ≤ x ∧ x < 7}
noncomputable def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}
noncomputable def A_or_B_eq_r (a : ℝ) : Prop := domain_f_2x_plus_1 ∪ B a = Set.univ

theorem find_range_of_a (a : ℝ) : 
  A_or_B_eq_r a → 3 ≤ a ∧ a < 6 := 
sorry

end NUMINAMATH_GPT_find_range_of_a_l2168_216880


namespace NUMINAMATH_GPT_probability_hare_killed_l2168_216802

theorem probability_hare_killed (P_hit_1 P_hit_2 P_hit_3 : ℝ)
  (h1 : P_hit_1 = 3 / 5) (h2 : P_hit_2 = 3 / 10) (h3 : P_hit_3 = 1 / 10) :
  (1 - ((1 - P_hit_1) * (1 - P_hit_2) * (1 - P_hit_3))) = 0.748 :=
by
  sorry

end NUMINAMATH_GPT_probability_hare_killed_l2168_216802


namespace NUMINAMATH_GPT_cement_amount_l2168_216825

theorem cement_amount
  (originally_had : ℕ)
  (bought : ℕ)
  (total : ℕ)
  (son_brought : ℕ)
  (h1 : originally_had = 98)
  (h2 : bought = 215)
  (h3 : total = 450)
  (h4 : originally_had + bought + son_brought = total) :
  son_brought = 137 :=
by
  sorry

end NUMINAMATH_GPT_cement_amount_l2168_216825


namespace NUMINAMATH_GPT_num_solutions_abs_x_plus_abs_y_lt_100_l2168_216823

theorem num_solutions_abs_x_plus_abs_y_lt_100 :
  (∃ n : ℕ, n = 338350 ∧ ∀ (x y : ℤ), (|x| + |y| < 100) → True) :=
sorry

end NUMINAMATH_GPT_num_solutions_abs_x_plus_abs_y_lt_100_l2168_216823


namespace NUMINAMATH_GPT_age_difference_l2168_216848

variable (A B C X : ℕ)

theorem age_difference 
  (h1 : C = A - 13)
  (h2 : A + B = B + C + X) 
  : X = 13 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l2168_216848


namespace NUMINAMATH_GPT_third_competitor_eats_l2168_216826

-- Define the conditions based on the problem description
def first_competitor_hot_dogs : ℕ := 12
def second_competitor_hot_dogs := 2 * first_competitor_hot_dogs
def third_competitor_hot_dogs := second_competitor_hot_dogs - (second_competitor_hot_dogs / 4)

-- The theorem we need to prove
theorem third_competitor_eats :
  third_competitor_hot_dogs = 18 := by
  sorry

end NUMINAMATH_GPT_third_competitor_eats_l2168_216826


namespace NUMINAMATH_GPT_greatest_number_of_dimes_l2168_216877

theorem greatest_number_of_dimes (total_value : ℝ) (num_dimes : ℕ) (num_nickels : ℕ) 
  (h_same_num : num_dimes = num_nickels) (h_total_value : total_value = 4.80) 
  (h_value_calculation : 0.10 * num_dimes + 0.05 * num_nickels = total_value) :
  num_dimes = 32 :=
by
  sorry

end NUMINAMATH_GPT_greatest_number_of_dimes_l2168_216877


namespace NUMINAMATH_GPT_range_a_A_intersect_B_empty_range_a_A_union_B_eq_B_l2168_216803

-- Definition of the sets A and B
def A (a : ℝ) (x : ℝ) : Prop := a - 1 < x ∧ x < 2 * a + 1
def B (x : ℝ) : Prop := 0 < x ∧ x < 1

-- Proving range of a for A ∩ B = ∅
theorem range_a_A_intersect_B_empty (a : ℝ) :
  (¬ ∃ x : ℝ, A a x ∧ B x) ↔ (a ≤ -2 ∨ a ≥ 2 ∨ (-2 < a ∧ a ≤ -1/2)) := sorry

-- Proving range of a for A ∪ B = B
theorem range_a_A_union_B_eq_B (a : ℝ) :
  (∀ x : ℝ, A a x ∨ B x → B x) ↔ (a ≤ -2) := sorry

end NUMINAMATH_GPT_range_a_A_intersect_B_empty_range_a_A_union_B_eq_B_l2168_216803


namespace NUMINAMATH_GPT_calculate_expression_l2168_216852

theorem calculate_expression : (1100 * 1100) / ((260 * 260) - (240 * 240)) = 121 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2168_216852


namespace NUMINAMATH_GPT_weight_of_new_person_l2168_216898

theorem weight_of_new_person (A : ℤ) (avg_weight_dec : ℤ) (n : ℤ) (new_avg : ℤ)
  (h1 : A = 102)
  (h2 : avg_weight_dec = 2)
  (h3 : n = 30) 
  (h4 : new_avg = A - avg_weight_dec) : 
  (31 * new_avg) - (30 * A) = 40 := 
by 
  sorry

end NUMINAMATH_GPT_weight_of_new_person_l2168_216898


namespace NUMINAMATH_GPT_inequality_solution_set_l2168_216847

def f (x : ℝ) : ℝ := x^3

theorem inequality_solution_set (x : ℝ) :
  (f (2 * x) + f (x - 1) < 0) ↔ (x < (1 / 3)) := 
sorry

end NUMINAMATH_GPT_inequality_solution_set_l2168_216847


namespace NUMINAMATH_GPT_sawyer_total_octopus_legs_l2168_216897

-- Formalization of the problem conditions
def num_octopuses : Nat := 5
def legs_per_octopus : Nat := 8

-- Formalization of the question and answer
def total_legs : Nat := num_octopuses * legs_per_octopus

-- The proof statement
theorem sawyer_total_octopus_legs : total_legs = 40 :=
by
  sorry

end NUMINAMATH_GPT_sawyer_total_octopus_legs_l2168_216897


namespace NUMINAMATH_GPT_find_x_plus_z_l2168_216831

theorem find_x_plus_z :
  ∃ (x y z : ℝ), 
  (x + y + z = 0) ∧
  (2016 * x + 2017 * y + 2018 * z = 0) ∧
  (2016^2 * x + 2017^2 * y + 2018^2 * z = 2018) ∧
  (x + z = 4036) :=
sorry

end NUMINAMATH_GPT_find_x_plus_z_l2168_216831


namespace NUMINAMATH_GPT_wake_up_time_l2168_216824

-- Definition of the conversion ratio from normal minutes to metric minutes
def conversion_ratio := 36 / 25

-- Definition of normal minutes in a full day
def normal_minutes_in_day := 24 * 60

-- Definition of metric minutes in a full day
def metric_minutes_in_day := 10 * 100

-- Definition to convert normal time (6:36 AM) to normal minutes
def normal_minutes_from_midnight (h m : ℕ) := h * 60 + m

-- Converting normal minutes to metric minutes using the conversion ratio
def metric_minutes (normal_mins : ℕ) := (normal_mins / 36) * 25

-- Definition of the final metric time 2:75
def metric_time := (2 * 100 + 75)

-- Proving the final answer is 275
theorem wake_up_time : 100 * 2 + 10 * 7 + 5 = 275 := by
  sorry

end NUMINAMATH_GPT_wake_up_time_l2168_216824


namespace NUMINAMATH_GPT_exists_rectangle_with_diagonal_zeros_and_ones_l2168_216863

-- Define the problem parameters
def n := 2012
def table := Matrix (Fin n) (Fin n) (Fin 2)

-- Conditions
def row_contains_zero_and_one (m : table) (r : Fin n) : Prop :=
  ∃ c1 c2 : Fin n, m r c1 = 0 ∧ m r c2 = 1

def col_contains_zero_and_one (m : table) (c : Fin n) : Prop :=
  ∃ r1 r2 : Fin n, m r1 c = 0 ∧ m r2 c = 1

-- Problem statement
theorem exists_rectangle_with_diagonal_zeros_and_ones
  (m : table)
  (h_rows : ∀ r : Fin n, row_contains_zero_and_one m r)
  (h_cols : ∀ c : Fin n, col_contains_zero_and_one m c) :
  ∃ (r1 r2 : Fin n) (c1 c2 : Fin n),
    m r1 c1 = 0 ∧ m r2 c2 = 0 ∧ m r1 c2 = 1 ∧ m r2 c1 = 1 :=
sorry

end NUMINAMATH_GPT_exists_rectangle_with_diagonal_zeros_and_ones_l2168_216863


namespace NUMINAMATH_GPT_rectangle_area_l2168_216801

theorem rectangle_area (x : ℝ) (w l : ℝ) (h₁ : l = 3 * w) (h₂ : l^2 + w^2 = x^2) :
    l * w = (3 / 10) * x^2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2168_216801


namespace NUMINAMATH_GPT_smallest_n_19n_congruent_1453_mod_8_l2168_216840

theorem smallest_n_19n_congruent_1453_mod_8 : 
  ∃ (n : ℕ), 19 * n % 8 = 1453 % 8 ∧ ∀ (m : ℕ), (19 * m % 8 = 1453 % 8 → n ≤ m) := 
sorry

end NUMINAMATH_GPT_smallest_n_19n_congruent_1453_mod_8_l2168_216840
