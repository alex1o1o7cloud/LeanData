import Mathlib

namespace NUMINAMATH_GPT_right_triangle_area_l108_10840

theorem right_triangle_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  (1 / 2 : ℝ) * a * b = 24 := by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l108_10840


namespace NUMINAMATH_GPT_number_of_ways_l108_10803

-- Define the conditions
def num_people : ℕ := 3
def num_sports : ℕ := 4

-- Prove the total number of different ways
theorem number_of_ways : num_sports ^ num_people = 64 := by
  sorry

end NUMINAMATH_GPT_number_of_ways_l108_10803


namespace NUMINAMATH_GPT_unique_prime_sum_diff_l108_10809

theorem unique_prime_sum_diff :
  ∀ p : ℕ, Prime p ∧ (∃ p1 p2 p3 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ (p = p1 + 2) ∧ (p = p3 - 2)) → p = 5 :=
by
  sorry

end NUMINAMATH_GPT_unique_prime_sum_diff_l108_10809


namespace NUMINAMATH_GPT_yellow_yellow_pairs_count_l108_10881

def num_blue_students : ℕ := 75
def num_yellow_students : ℕ := 105
def total_pairs : ℕ := 90
def blue_blue_pairs : ℕ := 30

theorem yellow_yellow_pairs_count :
  -- number of pairs where both students are wearing yellow shirts is 45.
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 45 :=
by
  sorry

end NUMINAMATH_GPT_yellow_yellow_pairs_count_l108_10881


namespace NUMINAMATH_GPT_bush_height_l108_10896

theorem bush_height (h : ℕ → ℕ) (h0 : h 5 = 81) (h1 : ∀ n, h (n + 1) = 3 * h n) :
  h 2 = 3 := 
sorry

end NUMINAMATH_GPT_bush_height_l108_10896


namespace NUMINAMATH_GPT_right_isosceles_hypotenuse_angle_l108_10822

theorem right_isosceles_hypotenuse_angle (α β : ℝ) (γ : ℝ)
  (h1 : α = 45) (h2 : β = 45) (h3 : γ = 90)
  (triangle_isosceles : α = β)
  (triangle_right : γ = 90) :
  γ = 90 :=
by
  sorry

end NUMINAMATH_GPT_right_isosceles_hypotenuse_angle_l108_10822


namespace NUMINAMATH_GPT_common_rational_root_l108_10869

-- Definitions for the given conditions
def polynomial1 (a b c : ℤ) (x : ℚ) := 50 * x^4 + a * x^3 + b * x^2 + c * x + 16 = 0
def polynomial2 (d e f g : ℤ) (x : ℚ) := 16 * x^5 + d * x^4 + e * x^3 + f * x^2 + g * x + 50 = 0

-- The proof problem statement: Given the conditions, proving that -1/2 is a common rational root
theorem common_rational_root (a b c d e f g : ℤ) (k : ℚ) 
  (h1 : polynomial1 a b c k)
  (h2 : polynomial2 d e f g k) 
  (h3 : ∃ m n : ℤ, k = -((m : ℚ) / n) ∧ Int.gcd m n = 1) :
  k = -1/2 :=
sorry

end NUMINAMATH_GPT_common_rational_root_l108_10869


namespace NUMINAMATH_GPT_divisible_by_10_l108_10812

theorem divisible_by_10 : (11 * 21 * 31 * 41 * 51 - 1) % 10 = 0 := by
  sorry

end NUMINAMATH_GPT_divisible_by_10_l108_10812


namespace NUMINAMATH_GPT_fraction_value_l108_10829

theorem fraction_value :
  (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400) / 
  ((6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400)) = 244.375 :=
by
  -- The proof would be provided here, but we are skipping it as per the instructions.
  sorry

end NUMINAMATH_GPT_fraction_value_l108_10829


namespace NUMINAMATH_GPT_larger_integer_value_l108_10800

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end NUMINAMATH_GPT_larger_integer_value_l108_10800


namespace NUMINAMATH_GPT_julia_age_after_10_years_l108_10827

-- Define the conditions
def Justin_age : Nat := 26
def Jessica_older_by : Nat := 6
def James_older_by : Nat := 7
def Julia_younger_by : Nat := 8
def years_after : Nat := 10

-- Define the ages now
def Jessica_age_now : Nat := Justin_age + Jessica_older_by
def James_age_now : Nat := Jessica_age_now + James_older_by
def Julia_age_now : Nat := Justin_age - Julia_younger_by

-- Prove that Julia's age after 10 years is 28
theorem julia_age_after_10_years : Julia_age_now + years_after = 28 := by
  sorry

end NUMINAMATH_GPT_julia_age_after_10_years_l108_10827


namespace NUMINAMATH_GPT_choir_members_max_l108_10826

-- Define the conditions and the proof for the equivalent problem.
theorem choir_members_max (c s y : ℕ) (h1 : c < 120) (h2 : s * y + 3 = c) (h3 : (s - 1) * (y + 2) = c) : c = 120 := by
  sorry

end NUMINAMATH_GPT_choir_members_max_l108_10826


namespace NUMINAMATH_GPT_find_range_of_m_l108_10805

def A (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 7
def B (m x : ℝ) : Prop := m + 1 < x ∧ x < 2 * m - 1

theorem find_range_of_m (m : ℝ) : 
  (∀ x, B m x → A x) ∧ (∃ x, B m x) → 2 < m ∧ m ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_find_range_of_m_l108_10805


namespace NUMINAMATH_GPT_find_units_digit_l108_10877

def units_digit (n : ℕ) : ℕ := n % 10

theorem find_units_digit :
  units_digit (3 * 19 * 1933 - 3^4) = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_units_digit_l108_10877


namespace NUMINAMATH_GPT_cos_5_theta_l108_10874

theorem cos_5_theta (θ : ℝ) (h : Real.cos θ = 2 / 5) : Real.cos (5 * θ) = 2762 / 3125 := 
sorry

end NUMINAMATH_GPT_cos_5_theta_l108_10874


namespace NUMINAMATH_GPT_simplify_trig_expression_l108_10853

open Real

theorem simplify_trig_expression (A : ℝ) (h1 : cos A ≠ 0) (h2 : sin A ≠ 0) :
  (1 - (cos A) / (sin A) + 1 / (sin A)) * (1 + (sin A) / (cos A) - 1 / (cos A)) = -2 * (cos (2 * A) / sin (2 * A)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_trig_expression_l108_10853


namespace NUMINAMATH_GPT_find_a_l108_10839

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then x + 3 else 4 / x

theorem find_a (a : ℝ) (h : f a = 2) : a = -1 ∨ a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l108_10839


namespace NUMINAMATH_GPT_min_distance_l108_10878

variables {P Q : ℝ × ℝ}

def line (P : ℝ × ℝ) : Prop := 3 * P.1 + 4 * P.2 + 5 = 0
def circle (Q : ℝ × ℝ) : Prop := (Q.1 - 2) ^ 2 + (Q.2 - 2) ^ 2 = 4

theorem min_distance (P : ℝ × ℝ) (Q : ℝ × ℝ) (hP : line P) (hQ : circle Q) :
  ∃ d : ℝ, d = dist P Q ∧ d = 9 / 5 := sorry

end NUMINAMATH_GPT_min_distance_l108_10878


namespace NUMINAMATH_GPT_brendan_cuts_84_yards_in_week_with_lawnmower_l108_10898

-- Brendan cuts 8 yards per day
def yards_per_day : ℕ := 8

-- The lawnmower increases his efficiency by fifty percent
def efficiency_increase (yards : ℕ) : ℕ :=
  yards + (yards / 2)

-- Calculate total yards cut in 7 days with the lawnmower
def total_yards_in_week (days : ℕ) (daily_yards : ℕ) : ℕ :=
  days * daily_yards

-- Prove the total yards cut in 7 days with the lawnmower is 84
theorem brendan_cuts_84_yards_in_week_with_lawnmower :
  total_yards_in_week 7 (efficiency_increase yards_per_day) = 84 :=
by
  sorry

end NUMINAMATH_GPT_brendan_cuts_84_yards_in_week_with_lawnmower_l108_10898


namespace NUMINAMATH_GPT_candies_left_after_carlos_ate_l108_10845

def num_red_candies : ℕ := 50
def num_yellow_candies : ℕ := 3 * num_red_candies - 35
def num_blue_candies : ℕ := (2 * num_yellow_candies) / 3
def num_green_candies : ℕ := 20
def num_purple_candies : ℕ := num_green_candies / 2
def num_silver_candies : ℕ := 10
def num_candies_eaten_by_carlos : ℕ := num_yellow_candies + num_green_candies / 2

def total_candies : ℕ := num_red_candies + num_yellow_candies + num_blue_candies + num_green_candies + num_purple_candies + num_silver_candies
def candies_remaining : ℕ := total_candies - num_candies_eaten_by_carlos

theorem candies_left_after_carlos_ate : candies_remaining = 156 := by
  sorry

end NUMINAMATH_GPT_candies_left_after_carlos_ate_l108_10845


namespace NUMINAMATH_GPT_washing_machine_capacity_l108_10814

-- Define the conditions:
def shirts : ℕ := 39
def sweaters : ℕ := 33
def loads : ℕ := 9
def total_clothes : ℕ := shirts + sweaters -- which is 72

-- Define the statement to be proved:
theorem washing_machine_capacity : ∃ x : ℕ, loads * x = total_clothes ∧ x = 8 :=
by
  -- proof to be completed
  sorry

end NUMINAMATH_GPT_washing_machine_capacity_l108_10814


namespace NUMINAMATH_GPT_problem1_problem2_l108_10831

theorem problem1 : -20 - (-8) + (-4) = -16 := by
  sorry

theorem problem2 : -1^3 * (-2)^2 / (4 / 3 : ℚ) + |5 - 8| = 0 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l108_10831


namespace NUMINAMATH_GPT_range_a_satisfies_l108_10842

theorem range_a_satisfies (a : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x : ℝ, f x = x^3) (h₂ : f 2 = 8) :
  (f (a - 3) > f (1 - a)) ↔ a > 2 :=
by
  sorry

end NUMINAMATH_GPT_range_a_satisfies_l108_10842


namespace NUMINAMATH_GPT_selling_price_eq_120_l108_10875

-- Definitions based on the conditions
def cost_price : ℝ := 96
def profit_percentage : ℝ := 0.25

-- The proof statement
theorem selling_price_eq_120 (cost_price : ℝ) (profit_percentage : ℝ) : cost_price = 96 → profit_percentage = 0.25 → (cost_price + cost_price * profit_percentage) = 120 :=
by
  intros hcost hprofit
  rw [hcost, hprofit]
  sorry

end NUMINAMATH_GPT_selling_price_eq_120_l108_10875


namespace NUMINAMATH_GPT_basketball_team_heights_l108_10889

theorem basketball_team_heights :
  ∃ (second tallest third fourth shortest : ℝ),
  (tallest = 80.5 ∧
   second = tallest - 6.25 ∧
   third = second - 3.75 ∧
   fourth = third - 5.5 ∧
   shortest = fourth - 4.8 ∧
   second = 74.25 ∧
   third = 70.5 ∧
   fourth = 65 ∧
   shortest = 60.2) := sorry

end NUMINAMATH_GPT_basketball_team_heights_l108_10889


namespace NUMINAMATH_GPT_shortest_side_of_triangle_l108_10817

theorem shortest_side_of_triangle 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b) 
  (h_inequal : a^2 + b^2 > 5 * c^2) :
  c < a ∧ c < b := 
by 
  sorry

end NUMINAMATH_GPT_shortest_side_of_triangle_l108_10817


namespace NUMINAMATH_GPT_fg_equals_seven_l108_10835

def g (x : ℤ) : ℤ := x * x
def f (x : ℤ) : ℤ := 2 * x - 1

theorem fg_equals_seven : f (g 2) = 7 := by
  sorry

end NUMINAMATH_GPT_fg_equals_seven_l108_10835


namespace NUMINAMATH_GPT_rectangle_ratio_l108_10801

theorem rectangle_ratio (w : ℝ) (h : ℝ)
  (hw : h = 10)   -- Length is 10
  (hp : 2 * w + 2 * h = 30) :  -- Perimeter is 30
  w / h = 1 / 2 :=             -- Ratio of width to length is 1/2
by
  -- Pending proof
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l108_10801


namespace NUMINAMATH_GPT_dad_contribution_is_correct_l108_10860

noncomputable def carl_savings_weekly : ℕ := 25
noncomputable def savings_duration_weeks : ℕ := 6
noncomputable def coat_cost : ℕ := 170

-- Total savings after 6 weeks
noncomputable def total_savings : ℕ := carl_savings_weekly * savings_duration_weeks

-- Amount used to pay bills in the seventh week
noncomputable def bills_payment : ℕ := total_savings / 3

-- Money left after paying bills
noncomputable def remaining_savings : ℕ := total_savings - bills_payment

-- Amount needed from Dad
noncomputable def dad_contribution : ℕ := coat_cost - remaining_savings

theorem dad_contribution_is_correct : dad_contribution = 70 := by
  sorry

end NUMINAMATH_GPT_dad_contribution_is_correct_l108_10860


namespace NUMINAMATH_GPT_cosine_angle_is_zero_l108_10825

-- Define the structure of an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  angle_60_deg : Prop

-- Define the structure of a parallelogram built from 6 equilateral triangles
structure Parallelogram where
  composed_of_6_equilateral_triangles : Prop
  folds_into_hexahedral_shape : Prop

-- Define the angle and its cosine computation between two specific directions in the folded hexahedral shape
def cosine_of_angle_between_AB_and_CD (parallelogram : Parallelogram) : ℝ := sorry

-- The condition that needs to be proved
axiom parallelogram_conditions : Parallelogram
axiom cosine_angle_proof : cosine_of_angle_between_AB_and_CD parallelogram_conditions = 0

-- Final proof statement
theorem cosine_angle_is_zero : cosine_of_angle_between_AB_and_CD parallelogram_conditions = 0 :=
cosine_angle_proof

end NUMINAMATH_GPT_cosine_angle_is_zero_l108_10825


namespace NUMINAMATH_GPT_total_annual_interest_l108_10868

theorem total_annual_interest 
    (principal1 principal2 : ℝ)
    (rate1 rate2 : ℝ)
    (time : ℝ)
    (h1 : principal1 = 26000)
    (h2 : rate1 = 0.08)
    (h3 : principal2 = 24000)
    (h4 : rate2 = 0.085)
    (h5 : time = 1) :
    principal1 * rate1 * time + principal2 * rate2 * time = 4120 := 
sorry

end NUMINAMATH_GPT_total_annual_interest_l108_10868


namespace NUMINAMATH_GPT_line_transformation_l108_10858

theorem line_transformation (a b : ℝ)
  (h1 : ∀ x y : ℝ, a * x + y - 7 = 0)
  (A : Matrix (Fin 2) (Fin 2) ℝ) (hA : A = ![![3, 0], ![-1, b]])
  (h2 : ∀ x' y' : ℝ, 9 * x' + y' - 91 = 0) :
  (a = 2) ∧ (b = 13) :=
by
  sorry

end NUMINAMATH_GPT_line_transformation_l108_10858


namespace NUMINAMATH_GPT_positive_integer_solutions_l108_10834

theorem positive_integer_solutions (x : ℕ) (h : 2 * x + 9 ≥ 3 * (x + 2)) : x = 1 ∨ x = 2 ∨ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_l108_10834


namespace NUMINAMATH_GPT_min_value_x1_x2_frac1_x1x2_l108_10865

theorem min_value_x1_x2_frac1_x1x2 (a x1 x2 : ℝ) (ha : a > 2) (h_sum : x1 + x2 = a) (h_prod : x1 * x2 = a - 2) :
  x1 + x2 + 1 / (x1 * x2) ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_x1_x2_frac1_x1x2_l108_10865


namespace NUMINAMATH_GPT_cube_inequality_l108_10844

theorem cube_inequality {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end NUMINAMATH_GPT_cube_inequality_l108_10844


namespace NUMINAMATH_GPT_problem_statement_l108_10832

theorem problem_statement (a b c : ℝ) (h1 : a ∈ Set.Ioi 0) (h2 : b ∈ Set.Ioi 0) (h3 : c ∈ Set.Ioi 0) (h4 : a^2 + b^2 + c^2 = 3) : 
  1 / (2 - a) + 1 / (2 - b) + 1 / (2 - c) ≥ 3 := 
sorry

end NUMINAMATH_GPT_problem_statement_l108_10832


namespace NUMINAMATH_GPT_min_x_y_l108_10897

theorem min_x_y
  (x y : ℝ)
  (h1 : 0 < x)
  (h2 : 0 < y)
  (h3 : x + 2 * y + x * y - 7 = 0) :
  x + y ≥ 3 := by
  sorry

end NUMINAMATH_GPT_min_x_y_l108_10897


namespace NUMINAMATH_GPT_fraction_neg_range_l108_10893

theorem fraction_neg_range (x : ℝ) : (x ≠ 0 ∧ x < 1) ↔ (x - 1 < 0 ∧ x^2 > 0) := by
  sorry

end NUMINAMATH_GPT_fraction_neg_range_l108_10893


namespace NUMINAMATH_GPT_birds_flew_away_l108_10824

-- Define the initial and remaining birds
def original_birds : ℕ := 12
def remaining_birds : ℕ := 4

-- Define the number of birds that flew away
noncomputable def flew_away_birds : ℕ := original_birds - remaining_birds

-- State the theorem that the number of birds that flew away is 8
theorem birds_flew_away : flew_away_birds = 8 := by
  -- Lean expects a proof here. For now, we use sorry to indicate the proof is skipped.
  sorry

end NUMINAMATH_GPT_birds_flew_away_l108_10824


namespace NUMINAMATH_GPT_fraction_from_condition_l108_10890

theorem fraction_from_condition (x f : ℝ) (h : 0.70 * x = f * x + 110) (hx : x = 300) : f = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_from_condition_l108_10890


namespace NUMINAMATH_GPT_verify_condition_C_l108_10894

variable (x y z : ℤ)

-- Given conditions
def condition_C : Prop := x = y ∧ y = z + 1

-- The theorem/proof problem
theorem verify_condition_C (h : condition_C x y z) : (x - y)^2 + (y - z)^2 + (z - x)^2 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_verify_condition_C_l108_10894


namespace NUMINAMATH_GPT_probability_same_color_l108_10815

theorem probability_same_color (pairs : ℕ) (total_shoes : ℕ) (select_shoes : ℕ)
  (h_pairs : pairs = 6) 
  (h_total_shoes : total_shoes = 12) 
  (h_select_shoes : select_shoes = 2) : 
  (Nat.choose total_shoes select_shoes > 0) → 
  (Nat.div (pairs * (Nat.choose 2 2)) (Nat.choose total_shoes select_shoes) = 1/11) :=
by
  sorry

end NUMINAMATH_GPT_probability_same_color_l108_10815


namespace NUMINAMATH_GPT_quadratic_equation_root_condition_l108_10867

theorem quadratic_equation_root_condition (a : ℝ) :
  (∃ x1 x2 : ℝ, (a - 1) * x1^2 - 4 * x1 - 1 = 0 ∧ (a - 1) * x2^2 - 4 * x2 - 1 = 0) ↔ (a ≥ -3 ∧ a ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_root_condition_l108_10867


namespace NUMINAMATH_GPT_can_combine_fig1_can_combine_fig2_l108_10863

-- Given areas for rectangle partitions
variables (S1 S2 S3 S4 : ℝ)
-- Condition: total area of black rectangles equals total area of white rectangles
variable (h1 : S1 + S2 = S3 + S4)

-- Proof problem for Figure 1
theorem can_combine_fig1 : ∃ A : ℝ, S1 + S2 = A ∧ S3 + S4 = A := by
  sorry

-- Proof problem for Figure 2
theorem can_combine_fig2 : ∃ B : ℝ, S1 + S2 = B ∧ S3 + S4 = B := by
  sorry

end NUMINAMATH_GPT_can_combine_fig1_can_combine_fig2_l108_10863


namespace NUMINAMATH_GPT_a_n_is_perfect_square_l108_10883

def seqs (a b : ℕ → ℤ) : Prop :=
  a 0 = 1 ∧ b 0 = 0 ∧ ∀ n, a (n + 1) = 7 * a n + 6 * b n - 3 ∧ b (n + 1) = 8 * a n + 7 * b n - 4

theorem a_n_is_perfect_square (a b : ℕ → ℤ) (h : seqs a b) :
  ∀ n, ∃ k : ℤ, a n = k^2 :=
by
  sorry

end NUMINAMATH_GPT_a_n_is_perfect_square_l108_10883


namespace NUMINAMATH_GPT_min_value_x_y_l108_10862

theorem min_value_x_y {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 4 / y + 1 / x = 4) : x + y ≥ 2 :=
sorry

end NUMINAMATH_GPT_min_value_x_y_l108_10862


namespace NUMINAMATH_GPT_range_of_a_l108_10871

theorem range_of_a (a : ℝ) :  (5 - a > 0) ∧ (a - 2 > 0) ∧ (a - 2 ≠ 1) → (2 < a ∧ a < 3) ∨ (3 < a ∧ a < 5) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l108_10871


namespace NUMINAMATH_GPT_simplify_expression_l108_10807

theorem simplify_expression (m n : ℝ) (h : m^2 + 3 * m * n = 5) : 
  5 * m^2 - 3 * m * n - (-9 * m * n + 3 * m^2) = 10 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l108_10807


namespace NUMINAMATH_GPT_sum_of_brothers_ages_l108_10837

theorem sum_of_brothers_ages (Bill Eric: ℕ) 
  (h1: 4 = Bill - Eric) 
  (h2: Bill = 16) : 
  Bill + Eric = 28 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_brothers_ages_l108_10837


namespace NUMINAMATH_GPT_find_D_l108_10823

-- Definitions
variable (A B C D E F : ℕ)

-- Conditions
axiom sum_AB : A + B = 16
axiom sum_BC : B + C = 12
axiom sum_EF : E + F = 8
axiom total_sum : A + B + C + D + E + F = 18

-- Theorem statement
theorem find_D : D = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_D_l108_10823


namespace NUMINAMATH_GPT_theater_total_revenue_l108_10899

theorem theater_total_revenue :
  let seats := 400
  let capacity := 0.8
  let ticket_price := 30
  let days := 3
  seats * capacity * ticket_price * days = 28800 := by
  sorry

end NUMINAMATH_GPT_theater_total_revenue_l108_10899


namespace NUMINAMATH_GPT_largest_sum_of_distinct_factors_of_1764_l108_10859

theorem largest_sum_of_distinct_factors_of_1764 :
  ∃ (A B C : ℕ), A * B * C = 1764 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A + B + C = 33 :=
by
  sorry

end NUMINAMATH_GPT_largest_sum_of_distinct_factors_of_1764_l108_10859


namespace NUMINAMATH_GPT_six_people_with_A_not_on_ends_l108_10873

-- Define the conditions and the problem statement
def standing_arrangements (n : ℕ) (A : Type) :=
  {l : List A // l.length = n}

theorem six_people_with_A_not_on_ends : 
  (arr : standing_arrangements 6 ℕ) → 
  (∀ a ∈ arr.val, a ≠ 0 ∧ a ≠ 5) → 
  ∃! (total_arrangements : ℕ), total_arrangements = 480 :=
  by
    sorry

end NUMINAMATH_GPT_six_people_with_A_not_on_ends_l108_10873


namespace NUMINAMATH_GPT_minimum_area_rectangle_l108_10821

noncomputable def minimum_rectangle_area (a : ℝ) : ℝ :=
  if a ≤ 0 then (1 - a) * Real.sqrt (1 - a)
  else if a < 1 / 2 then 1 - 2 * a
  else 0

theorem minimum_area_rectangle (a : ℝ) :
  minimum_rectangle_area a =
    if a ≤ 0 then (1 - a) * Real.sqrt (1 - a)
    else if a < 1 / 2 then 1 - 2 * a
    else 0 :=
by
  sorry

end NUMINAMATH_GPT_minimum_area_rectangle_l108_10821


namespace NUMINAMATH_GPT_remainder_ab_cd_l108_10850

theorem remainder_ab_cd (n : ℕ) (hn: n > 0) (a b c d : ℤ) 
  (hac : a * c ≡ 1 [ZMOD n]) (hbd : b * d ≡ 1 [ZMOD n]) : 
  (a * b + c * d) % n = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_ab_cd_l108_10850


namespace NUMINAMATH_GPT_curve_self_intersection_l108_10879

def curve_crosses_itself_at_point (x y : ℝ) : Prop :=
∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ (t₁^2 - 4 = x) ∧ (t₁^3 - 6 * t₁ + 7 = y) ∧ (t₂^2 - 4 = x) ∧ (t₂^3 - 6 * t₂ + 7 = y)

theorem curve_self_intersection : curve_crosses_itself_at_point 2 7 :=
sorry

end NUMINAMATH_GPT_curve_self_intersection_l108_10879


namespace NUMINAMATH_GPT_minimum_a_div_x_l108_10885

theorem minimum_a_div_x (a x y : ℕ) (h1 : 100 < a) (h2 : 100 < x) (h3 : 100 < y) (h4 : y^2 - 1 = a^2 * (x^2 - 1)) :
  2 ≤ a / x :=
by sorry

end NUMINAMATH_GPT_minimum_a_div_x_l108_10885


namespace NUMINAMATH_GPT_sequence_general_formula_l108_10886

theorem sequence_general_formula (a : ℕ → ℕ)
    (h1 : a 1 = 1)
    (h2 : a 2 = 2)
    (h3 : ∀ n, a (n + 2) = a n + 2) :
    ∀ n, a n = n := by
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l108_10886


namespace NUMINAMATH_GPT_books_on_desk_none_useful_l108_10855

theorem books_on_desk_none_useful :
  ∃ (answer : String), answer = "none" ∧ 
  (answer = "nothing" ∨ answer = "no one" ∨ answer = "neither" ∨ answer = "none")
  → answer = "none"
:= by
  sorry

end NUMINAMATH_GPT_books_on_desk_none_useful_l108_10855


namespace NUMINAMATH_GPT_trig_identity_l108_10836

theorem trig_identity :
  let s60 := Real.sin (60 * Real.pi / 180)
  let c1 := Real.cos (1 * Real.pi / 180)
  let c20 := Real.cos (20 * Real.pi / 180)
  let s10 := Real.sin (10 * Real.pi / 180)
  s60 * c1 * c20 - s10 = Real.sqrt 3 / 2 - s10 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l108_10836


namespace NUMINAMATH_GPT_percentage_increase_l108_10848

theorem percentage_increase (L : ℕ) (h : L + 60 = 240) : 
  ((60:ℝ) / (L:ℝ)) * 100 = 33.33 := 
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l108_10848


namespace NUMINAMATH_GPT_geometric_sequence_sixth_term_l108_10882

theorem geometric_sequence_sixth_term (a b : ℚ) (h : a = 3 ∧ b = -1/2) : 
  (a * (b / a) ^ 5) = -1/2592 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sixth_term_l108_10882


namespace NUMINAMATH_GPT_son_age_is_15_l108_10876

theorem son_age_is_15 (S F : ℕ) (h1 : 2 * S + F = 70) (h2 : 2 * F + S = 95) (h3 : F = 40) :
  S = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_son_age_is_15_l108_10876


namespace NUMINAMATH_GPT_simplify_exponents_l108_10864

theorem simplify_exponents : (10^0.5) * (10^0.3) * (10^0.2) * (10^0.1) * (10^0.9) = 100 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_exponents_l108_10864


namespace NUMINAMATH_GPT_find_f_x_l108_10806

theorem find_f_x (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x - 1) = x^2 - x) : ∀ x : ℝ, f x = (1/4) * (x^2 - 1) := 
sorry

end NUMINAMATH_GPT_find_f_x_l108_10806


namespace NUMINAMATH_GPT_mimi_spending_adidas_l108_10828

theorem mimi_spending_adidas
  (total_spending : ℤ)
  (nike_to_adidas_ratio : ℤ)
  (adidas_to_skechers_ratio : ℤ)
  (clothes_spending : ℤ)
  (eq1 : total_spending = 8000)
  (eq2 : nike_to_adidas_ratio = 3)
  (eq3 : adidas_to_skechers_ratio = 5)
  (eq4 : clothes_spending = 2600) :
  ∃ A : ℤ, A + nike_to_adidas_ratio * A + adidas_to_skechers_ratio * A + clothes_spending = total_spending ∧ A = 600 := by
  sorry

end NUMINAMATH_GPT_mimi_spending_adidas_l108_10828


namespace NUMINAMATH_GPT_sqrt_6_between_2_and_3_l108_10887

theorem sqrt_6_between_2_and_3 : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_6_between_2_and_3_l108_10887


namespace NUMINAMATH_GPT_min_large_trucks_needed_l108_10856

-- Define the parameters for the problem
def total_fruit : ℕ := 134
def load_large_truck : ℕ := 15
def load_small_truck : ℕ := 7

-- Define the main theorem to be proved
theorem min_large_trucks_needed :
  ∃ (n : ℕ), n = 8 ∧ (total_fruit = n * load_large_truck + 2 * load_small_truck) :=
by sorry

end NUMINAMATH_GPT_min_large_trucks_needed_l108_10856


namespace NUMINAMATH_GPT_max_height_reached_l108_10895

def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem max_height_reached : ∃ t : ℝ, h t = 161 :=
by
  sorry

end NUMINAMATH_GPT_max_height_reached_l108_10895


namespace NUMINAMATH_GPT_jessica_balloons_l108_10830

-- Defining the number of blue balloons Joan, Sally, and the total number.
def balloons_joan : ℕ := 9
def balloons_sally : ℕ := 5
def balloons_total : ℕ := 16

-- The statement to prove that Jessica has 2 blue balloons
theorem jessica_balloons : balloons_total - (balloons_joan + balloons_sally) = 2 :=
by
  -- Using the given information and arithmetic, we can show the main statement
  sorry

end NUMINAMATH_GPT_jessica_balloons_l108_10830


namespace NUMINAMATH_GPT_solve_inequality_l108_10872

theorem solve_inequality (x : ℝ) (h : x < 4) : (x - 2) / (x - 4) ≥ 3 := sorry

end NUMINAMATH_GPT_solve_inequality_l108_10872


namespace NUMINAMATH_GPT_car_speed_conversion_l108_10866

noncomputable def miles_to_yards : ℕ :=
  1760

theorem car_speed_conversion (speed_mph : ℕ) (time_sec : ℝ) (distance_yards : ℕ) :
  speed_mph = 90 →
  time_sec = 0.5 →
  distance_yards = 22 →
  (1 : ℕ) * miles_to_yards = 1760 := by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_car_speed_conversion_l108_10866


namespace NUMINAMATH_GPT_correct_average_is_15_l108_10802

theorem correct_average_is_15 (n incorrect_avg correct_num wrong_num : ℕ) 
  (h1 : n = 10) (h2 : incorrect_avg = 14) (h3 : correct_num = 36) (h4 : wrong_num = 26) : 
  (incorrect_avg * n + (correct_num - wrong_num)) / n = 15 := 
by 
  sorry

end NUMINAMATH_GPT_correct_average_is_15_l108_10802


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l108_10851

variable {a : ℝ}

theorem sufficient_but_not_necessary (h : a > 1 / a^2) : a^2 > 1 / a ∧ ¬ ∀ a, a^2 > 1 / a → a > 1 / a^2 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l108_10851


namespace NUMINAMATH_GPT_red_marbles_count_l108_10811

theorem red_marbles_count :
  ∀ (total marbles white yellow green red : ℕ),
    total = 50 →
    white = total / 2 →
    yellow = 12 →
    green = yellow / 2 →
    red = total - (white + yellow + green) →
    red = 7 :=
by
  intros total marbles white yellow green red Htotal Hwhite Hyellow Hgreen Hred
  sorry

end NUMINAMATH_GPT_red_marbles_count_l108_10811


namespace NUMINAMATH_GPT_complement_S_union_T_eq_l108_10870

noncomputable def S := {x : ℝ | x > -2}
noncomputable def T := {x : ℝ | x^2 + 3 * x - 4 ≤ 0}
noncomputable def complement_S := {x : ℝ | x ≤ -2}

theorem complement_S_union_T_eq : (complement_S ∪ T) = {x : ℝ | x ≤ 1} := by 
  sorry

end NUMINAMATH_GPT_complement_S_union_T_eq_l108_10870


namespace NUMINAMATH_GPT_solve_pairs_l108_10819

theorem solve_pairs (m n : ℕ) :
  m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ (m, n) = (6, 3) ∨ (m, n) = (9, 3) ∨ (m, n) = (9, 5) ∨ (m, n) = (54, 5) :=
by
  sorry

end NUMINAMATH_GPT_solve_pairs_l108_10819


namespace NUMINAMATH_GPT_find_range_of_r_l108_10810

noncomputable def range_of_r : Set ℝ :=
  {r : ℝ | 3 * Real.sqrt 5 - 3 * Real.sqrt 2 ≤ r ∧ r ≤ 3 * Real.sqrt 5 + 3 * Real.sqrt 2}

theorem find_range_of_r 
  (O : ℝ × ℝ) (A : ℝ × ℝ) (r : ℝ) (h : r > 0)
  (hA : A = (0, 3))
  (C : Set (ℝ × ℝ)) (hC : C = {M : ℝ × ℝ | (M.1 - 3)^2 + (M.2 - 3)^2 = r^2})
  (M : ℝ × ℝ) (hM : M ∈ C)
  (h_cond : (M.1 - 0)^2 + (M.2 - 3)^2 = 2 * ((M.1 - 0)^2 + (M.2 - 0)^2)) :
  r ∈ range_of_r :=
sorry

end NUMINAMATH_GPT_find_range_of_r_l108_10810


namespace NUMINAMATH_GPT_cost_of_one_shirt_l108_10854

theorem cost_of_one_shirt
  (J S : ℝ)
  (h1 : 3 * J + 2 * S = 69)
  (h2 : 2 * J + 3 * S = 76) :
  S = 18 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_one_shirt_l108_10854


namespace NUMINAMATH_GPT_each_tree_takes_one_square_foot_l108_10833

theorem each_tree_takes_one_square_foot (total_length : ℝ) (num_trees : ℕ) (gap_length : ℝ)
    (total_length_eq : total_length = 166) (num_trees_eq : num_trees = 16) (gap_length_eq : gap_length = 10) :
    (total_length - (((num_trees - 1) : ℝ) * gap_length)) / (num_trees : ℝ) = 1 :=
by
  rw [total_length_eq, num_trees_eq, gap_length_eq]
  sorry

end NUMINAMATH_GPT_each_tree_takes_one_square_foot_l108_10833


namespace NUMINAMATH_GPT_infinite_series_sum_l108_10849

theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * (n + 1) * (n + 1) + 2 * (n + 1) + 1) / ((n + 1) * (n + 2) * (n + 3) * (n + 4))) 
  = 7 / 6 := 
by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l108_10849


namespace NUMINAMATH_GPT_line_through_point_equal_distance_l108_10818

noncomputable def line_equation (x0 y0 a b c x1 y1 : ℝ) : Prop :=
  (a * x0 + b * y0 + c = 0) ∧ (a * x1 + b * y1 + c = 0)

theorem line_through_point_equal_distance (A B : ℝ × ℝ) (P : ℝ × ℝ) :
  ∃ (a b c : ℝ), 
    line_equation P.1 P.2 a b c A.1 A.2 ∧ 
    line_equation P.1 P.2 a b c B.1 B.2 ∧
    (a = 2) ∧ (b = 3) ∧ (c = -18) ∨
    (a = 2) ∧ (b = -1) ∧ (c = -2)
:=
sorry

end NUMINAMATH_GPT_line_through_point_equal_distance_l108_10818


namespace NUMINAMATH_GPT_intersection_A_B_l108_10892

-- Define the sets A and B based on given conditions
def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {x | -1 < x}

-- The statement to prove
theorem intersection_A_B : (A ∩ B) = {x | -1 < x ∧ x < 4} :=
  sorry

end NUMINAMATH_GPT_intersection_A_B_l108_10892


namespace NUMINAMATH_GPT_probability_correct_l108_10880

variable (new_balls old_balls total_balls : ℕ)

-- Define initial conditions
def initial_conditions (new_balls old_balls : ℕ) : Prop :=
  new_balls = 4 ∧ old_balls = 2

-- Define total number of balls in the box
def total_balls_condition (new_balls old_balls total_balls : ℕ) : Prop :=
  total_balls = new_balls + old_balls ∧ total_balls = 6

-- Define the combination function
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability of picking one new ball and one old ball
def probability_one_new_one_old (new_balls old_balls total_balls : ℕ) : ℚ :=
  (combination new_balls 1 * combination old_balls 1) / (combination total_balls 2)

-- The theorem to prove the probability
theorem probability_correct (new_balls old_balls total_balls : ℕ)
  (h_initial : initial_conditions new_balls old_balls)
  (h_total : total_balls_condition new_balls old_balls total_balls) :
  probability_one_new_one_old new_balls old_balls total_balls = 8 / 15 := by
  sorry

end NUMINAMATH_GPT_probability_correct_l108_10880


namespace NUMINAMATH_GPT_inequality_solution_l108_10820

theorem inequality_solution (x : ℝ) :
  (x + 2) / (x^2 + 4) > 2 / x + 12 / 5 ↔ x < 0 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l108_10820


namespace NUMINAMATH_GPT_area_ratio_correct_l108_10888

noncomputable def ratio_area_MNO_XYZ (s t u : ℝ) (S_XYZ : ℝ) : ℝ := 
  let S_XMO := s * (1 - u) * S_XYZ
  let S_YNM := t * (1 - s) * S_XYZ
  let S_OZN := u * (1 - t) * S_XYZ
  S_XYZ - S_XMO - S_YNM - S_OZN

theorem area_ratio_correct (s t u : ℝ) (h1 : s + t + u = 3 / 4) 
  (h2 : s^2 + t^2 + u^2 = 3 / 8) : 
  ratio_area_MNO_XYZ s t u 1 = 13 / 32 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_area_ratio_correct_l108_10888


namespace NUMINAMATH_GPT_max_2b_div_a_l108_10813

theorem max_2b_div_a (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) : 
  ∃ max_val, max_val = (2 * b) / a ∧ max_val = (32 / 3) :=
by
  sorry

end NUMINAMATH_GPT_max_2b_div_a_l108_10813


namespace NUMINAMATH_GPT_find_x_eq_neg15_l108_10857

theorem find_x_eq_neg15 :
  ∃ x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) ↔ (x = -15) :=
by
  sorry

end NUMINAMATH_GPT_find_x_eq_neg15_l108_10857


namespace NUMINAMATH_GPT_find_side_length_of_cut_out_square_l108_10861

noncomputable def cardboard_box (x : ℝ) : Prop :=
  let length_initial := 80
  let width_initial := 60
  let area_base := 1500
  let length_final := length_initial - 2 * x
  let width_final := width_initial - 2 * x
  length_final * width_final = area_base

theorem find_side_length_of_cut_out_square : ∃ x : ℝ, cardboard_box x ∧ 0 ≤ x ∧ (80 - 2 * x) > 0 ∧ (60 - 2 * x) > 0 ∧ x = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_side_length_of_cut_out_square_l108_10861


namespace NUMINAMATH_GPT_student_correct_answers_l108_10841

variable (C I : ℕ)

theorem student_correct_answers :
  C + I = 100 ∧ C - 2 * I = 76 → C = 92 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_student_correct_answers_l108_10841


namespace NUMINAMATH_GPT_directrix_of_parabola_l108_10816

theorem directrix_of_parabola : 
  let y := 3 * x^2 - 6 * x + 1
  y = -25 / 12 :=
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l108_10816


namespace NUMINAMATH_GPT_set_equality_l108_10891

theorem set_equality : 
  { x : ℕ | ∃ k : ℕ, 6 - x = k ∧ 8 % k = 0 } = { 2, 4, 5 } :=
by
  sorry

end NUMINAMATH_GPT_set_equality_l108_10891


namespace NUMINAMATH_GPT_magazine_page_height_l108_10808

theorem magazine_page_height
  (charge_per_sq_inch : ℝ := 8)
  (half_page_cost : ℝ := 432)
  (page_width : ℝ := 12) : 
  ∃ h : ℝ, (1/2) * h * page_width * charge_per_sq_inch = half_page_cost :=
by sorry

end NUMINAMATH_GPT_magazine_page_height_l108_10808


namespace NUMINAMATH_GPT_dan_picked_l108_10847

-- Definitions:
def benny_picked : Nat := 2
def total_picked : Nat := 11

-- Problem statement:
theorem dan_picked (b : Nat) (t : Nat) (d : Nat) (h1 : b = benny_picked) (h2 : t = total_picked) (h3 : t = b + d) : d = 9 := by
  sorry

end NUMINAMATH_GPT_dan_picked_l108_10847


namespace NUMINAMATH_GPT_area_of_square_plot_l108_10846

-- Defining the given conditions and question in Lean 4
theorem area_of_square_plot 
  (cost_per_foot : ℕ := 58)
  (total_cost : ℕ := 2784) :
  ∃ (s : ℕ), (4 * s * cost_per_foot = total_cost) ∧ (s * s = 144) :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_plot_l108_10846


namespace NUMINAMATH_GPT_Sara_has_8_balloons_l108_10884

theorem Sara_has_8_balloons (Tom_balloons Sara_balloons total_balloons : ℕ)
  (htom : Tom_balloons = 9)
  (htotal : Tom_balloons + Sara_balloons = 17) :
  Sara_balloons = 8 :=
by
  sorry

end NUMINAMATH_GPT_Sara_has_8_balloons_l108_10884


namespace NUMINAMATH_GPT_triangle_classification_l108_10838

theorem triangle_classification (a b c : ℕ) (h : a + b + c = 12) :
((
  (a = b ∨ b = c ∨ a = c)  -- Isosceles
  ∨ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2)  -- Right-angled
  ∨ (a = b ∧ b = c)  -- Equilateral
)) :=
sorry

end NUMINAMATH_GPT_triangle_classification_l108_10838


namespace NUMINAMATH_GPT_max_value_of_perfect_sequence_l108_10852

def isPerfectSequence (c : ℕ → ℕ) : Prop := ∀ n m : ℕ, 1 ≤ m ∧ m ≤ (Finset.range (n + 1)).sum (fun k => c k) → 
  ∃ (a : ℕ → ℕ), m = (Finset.range (n + 1)).sum (fun k => c k / a k)

theorem max_value_of_perfect_sequence (n : ℕ) : 
  ∃ c : ℕ → ℕ, isPerfectSequence c ∧
    (∀ i, i ≤ n → c i ≤ if i = 1 then 2 else 4 * 3^(i - 2)) ∧
    c n = if n = 1 then 2 else 4 * 3^(n - 2) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_perfect_sequence_l108_10852


namespace NUMINAMATH_GPT_team_total_score_is_correct_l108_10804

-- Define the total number of team members
def total_members : ℕ := 30

-- Define the number of members who didn't show up
def members_absent : ℕ := 8

-- Define the score per member
def score_per_member : ℕ := 4

-- Define the points deducted per incorrect answer
def points_per_incorrect_answer : ℕ := 2

-- Define the total number of incorrect answers
def total_incorrect_answers : ℕ := 6

-- Define the bonus multiplier
def bonus_multiplier : ℝ := 1.5

-- Define the total score calculation
def total_score_calculation (total_members : ℕ) (members_absent : ℕ) (score_per_member : ℕ)
  (points_per_incorrect_answer : ℕ) (total_incorrect_answers : ℕ) (bonus_multiplier : ℝ) : ℝ :=
  let members_present := total_members - members_absent
  let initial_score := members_present * score_per_member
  let total_deductions := total_incorrect_answers * points_per_incorrect_answer
  let final_score := initial_score - total_deductions
  final_score * bonus_multiplier

-- Prove that the total score is 114 points
theorem team_total_score_is_correct : total_score_calculation total_members members_absent score_per_member
  points_per_incorrect_answer total_incorrect_answers bonus_multiplier = 114 :=
by
  sorry

end NUMINAMATH_GPT_team_total_score_is_correct_l108_10804


namespace NUMINAMATH_GPT_total_ants_found_l108_10843

-- Definitions for the number of ants each child finds
def abe_ants : ℕ := 4
def beth_ants : ℕ := abe_ants + (abe_ants / 2)
def cece_ants : ℕ := 2 * abe_ants
def duke_ants : ℕ := abe_ants / 2

-- Statement that needs to be proven
theorem total_ants_found : abe_ants + beth_ants + cece_ants + duke_ants = 20 :=
by sorry

end NUMINAMATH_GPT_total_ants_found_l108_10843
