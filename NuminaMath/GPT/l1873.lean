import Mathlib

namespace NUMINAMATH_GPT_grid_shaded_area_l1873_187388

theorem grid_shaded_area :
  let grid_side := 12
  let grid_area := grid_side^2
  let radius_small := 1.5
  let radius_large := 3
  let area_small := π * radius_small^2
  let area_large := π * radius_large^2
  let total_area_circles := 3 * area_small + area_large
  let visible_area := grid_area - total_area_circles
  let A := 144
  let B := 15.75
  A = 144 ∧ B = 15.75 ∧ (A + B = 159.75) →
  visible_area = 144 - 15.75 * π :=
by
  intros
  sorry

end NUMINAMATH_GPT_grid_shaded_area_l1873_187388


namespace NUMINAMATH_GPT_actual_cost_of_article_l1873_187364

noncomputable def article_actual_cost (x : ℝ) : Prop :=
  (0.58 * x = 1050) → x = 1810.34

theorem actual_cost_of_article : ∃ x : ℝ, article_actual_cost x :=
by
  use 1810.34
  sorry

end NUMINAMATH_GPT_actual_cost_of_article_l1873_187364


namespace NUMINAMATH_GPT_range_of_a_l1873_187344

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, (a * x^2 - 3 * x - 4 = 0) ∧ (a * y^2 - 3 * y - 4 = 0) → x = y) ↔ (a ≤ -9 / 16 ∨ a = 0) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1873_187344


namespace NUMINAMATH_GPT_Tim_cookie_packages_l1873_187345

theorem Tim_cookie_packages 
    (cookies_in_package : ℕ)
    (packets_in_package : ℕ)
    (min_packet_count : ℕ)
    (h1 : cookies_in_package = 5)
    (h2 : packets_in_package = 7)
    (h3 : min_packet_count = 30) :
  ∃ (cookie_packages : ℕ) (packet_packages : ℕ),
    cookie_packages = 7 ∧ packet_packages = 5 ∧
    cookie_packages * cookies_in_package = packet_packages * packets_in_package ∧
    packet_packages * packets_in_package ≥ min_packet_count :=
by
  sorry

end NUMINAMATH_GPT_Tim_cookie_packages_l1873_187345


namespace NUMINAMATH_GPT_correct_operations_result_l1873_187391

-- Define conditions and the problem statement
theorem correct_operations_result (x : ℝ) (h1: x / 8 - 12 = 18) : (x * 8) * 12 = 23040 :=
by
  sorry

end NUMINAMATH_GPT_correct_operations_result_l1873_187391


namespace NUMINAMATH_GPT_number_of_employees_is_five_l1873_187376

theorem number_of_employees_is_five
  (rudy_speed : ℕ)
  (joyce_speed : ℕ)
  (gladys_speed : ℕ)
  (lisa_speed : ℕ)
  (mike_speed : ℕ)
  (average_speed : ℕ)
  (h1 : rudy_speed = 64)
  (h2 : joyce_speed = 76)
  (h3 : gladys_speed = 91)
  (h4 : lisa_speed = 80)
  (h5 : mike_speed = 89)
  (h6 : average_speed = 80) :
  (rudy_speed + joyce_speed + gladys_speed + lisa_speed + mike_speed) / average_speed = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_employees_is_five_l1873_187376


namespace NUMINAMATH_GPT_fish_farm_estimated_mass_l1873_187389

noncomputable def total_fish_mass_in_pond 
  (initial_fry: ℕ) 
  (survival_rate: ℝ) 
  (haul1_count: ℕ) (haul1_avg_weight: ℝ) 
  (haul2_count: ℕ) (haul2_avg_weight: ℝ) 
  (haul3_count: ℕ) (haul3_avg_weight: ℝ) : ℝ :=
  let surviving_fish := initial_fry * survival_rate
  let total_mass_haul1 := haul1_count * haul1_avg_weight
  let total_mass_haul2 := haul2_count * haul2_avg_weight
  let total_mass_haul3 := haul3_count * haul3_avg_weight
  let average_weight_per_fish := (total_mass_haul1 + total_mass_haul2 + total_mass_haul3) / (haul1_count + haul2_count + haul3_count)
  average_weight_per_fish * surviving_fish

theorem fish_farm_estimated_mass :
  total_fish_mass_in_pond 
    80000           -- initial fry
    0.95            -- survival rate
    40 2.5          -- first haul: 40 fish, 2.5 kg each
    25 2.2          -- second haul: 25 fish, 2.2 kg each
    35 2.8          -- third haul: 35 fish, 2.8 kg each
    = 192280 := by
  sorry

end NUMINAMATH_GPT_fish_farm_estimated_mass_l1873_187389


namespace NUMINAMATH_GPT_negative_integer_solutions_l1873_187316

theorem negative_integer_solutions (x : ℤ) : 3 * x + 1 ≥ -5 ↔ x = -2 ∨ x = -1 := 
by
  sorry

end NUMINAMATH_GPT_negative_integer_solutions_l1873_187316


namespace NUMINAMATH_GPT_c_share_l1873_187320

theorem c_share (A B C : ℕ) (h1 : A = B / 2) (h2 : B = C / 2) (h3 : A + B + C = 392) : C = 224 :=
by
  sorry

end NUMINAMATH_GPT_c_share_l1873_187320


namespace NUMINAMATH_GPT_midpoint_coordinate_sum_l1873_187386

theorem midpoint_coordinate_sum
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1 = 10)
  (h2 : y1 = 3)
  (h3 : x2 = 4)
  (h4 : y2 = -3) :
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  xm + ym =  7 := by
  sorry

end NUMINAMATH_GPT_midpoint_coordinate_sum_l1873_187386


namespace NUMINAMATH_GPT_smallest_non_factor_product_of_factors_of_72_l1873_187317

theorem smallest_non_factor_product_of_factors_of_72 : 
  ∃ x y : ℕ, x ≠ y ∧ x * y ∣ 72 ∧ ¬ (x * y ∣ 72) ∧ x * y = 32 := 
by
  sorry

end NUMINAMATH_GPT_smallest_non_factor_product_of_factors_of_72_l1873_187317


namespace NUMINAMATH_GPT_poly_coeff_sum_l1873_187383

theorem poly_coeff_sum (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) (x : ℝ) :
  (2 * x + 3)^8 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + 
                 a_3 * (x + 1)^3 + a_4 * (x + 1)^4 + 
                 a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + 
                 a_7 * (x + 1)^7 + a_8 * (x + 1)^8 →
  a_0 + a_2 + a_4 + a_6 + a_8 = 3281 :=
by
  sorry

end NUMINAMATH_GPT_poly_coeff_sum_l1873_187383


namespace NUMINAMATH_GPT_number_of_even_red_faces_cubes_l1873_187319

def painted_cubes_even_faces : Prop :=
  let block_length := 4
  let block_width := 4
  let block_height := 1
  let edge_cubes_count := 8  -- The count of edge cubes excluding corners
  edge_cubes_count = 8

theorem number_of_even_red_faces_cubes : painted_cubes_even_faces := by
  sorry

end NUMINAMATH_GPT_number_of_even_red_faces_cubes_l1873_187319


namespace NUMINAMATH_GPT_combined_instruments_correct_l1873_187358

-- Definitions of initial conditions
def Charlie_flutes : Nat := 1
def Charlie_horns : Nat := 2
def Charlie_harps : Nat := 1
def Carli_flutes : Nat := 2 * Charlie_flutes
def Carli_horns : Nat := Charlie_horns / 2
def Carli_harps : Nat := 0

-- Calculation of total instruments
def Charlie_total_instruments : Nat := Charlie_flutes + Charlie_horns + Charlie_harps
def Carli_total_instruments : Nat := Carli_flutes + Carli_horns + Carli_harps
def combined_total_instruments : Nat := Charlie_total_instruments + Carli_total_instruments

-- Theorem statement
theorem combined_instruments_correct : combined_total_instruments = 7 := 
by
  sorry

end NUMINAMATH_GPT_combined_instruments_correct_l1873_187358


namespace NUMINAMATH_GPT_method_1_more_cost_effective_l1873_187380

open BigOperators

def racket_price : ℕ := 20
def shuttlecock_price : ℕ := 5
def rackets_bought : ℕ := 4
def shuttlecocks_bought : ℕ := 30
def discount_rate : ℚ := 0.92

def total_price (rackets shuttlecocks : ℕ) := racket_price * rackets + shuttlecock_price * shuttlecocks

def method_1_cost (rackets shuttlecocks : ℕ) := 
  total_price rackets shuttlecocks - shuttlecock_price * rackets

def method_2_cost (total : ℚ) :=
  total * discount_rate

theorem method_1_more_cost_effective :
  method_1_cost rackets_bought shuttlecocks_bought
  <
  method_2_cost (total_price rackets_bought shuttlecocks_bought) :=
by
  sorry

end NUMINAMATH_GPT_method_1_more_cost_effective_l1873_187380


namespace NUMINAMATH_GPT_geo_seq_a6_eight_l1873_187335

-- Definitions based on given conditions
variable (a : ℕ → ℝ) -- the sequence
variable (q : ℝ) -- common ratio
-- Conditions for a_1 * a_3 = 4 and a_4 = 4
def geometric_sequence := ∃ (q : ℝ), ∀ n : ℕ, a (n + 1) = a n * q
def condition1 := a 1 * a 3 = 4
def condition2 := a 4 = 4

-- Proof problem: Prove a_6 = 8 given the conditions above
theorem geo_seq_a6_eight (h1 : condition1 a) (h2 : condition2 a) (hs : geometric_sequence a) : 
  a 6 = 8 :=
sorry

end NUMINAMATH_GPT_geo_seq_a6_eight_l1873_187335


namespace NUMINAMATH_GPT_find_lines_and_intersections_l1873_187361

-- Define the intersection point conditions
def intersection_point (m n : ℝ) : Prop :=
  (2 * m - n + 7 = 0) ∧ (m + n - 1 = 0)

-- Define the perpendicular line to l1 passing through (-2, 3)
def perpendicular_line_through_A (x y : ℝ) : Prop :=
  x + 2 * y - 4 = 0

-- Define the parallel line to l passing through (-2, 3)
def parallel_line_through_A (x y : ℝ) : Prop :=
  2 * x - 3 * y + 13 = 0

-- main theorem
theorem find_lines_and_intersections :
  ∃ m n : ℝ, intersection_point m n ∧ m = -2 ∧ n = 3 ∧
  ∃ l3 : ℝ → ℝ → Prop, l3 = perpendicular_line_through_A ∧
  ∃ l4 : ℝ → ℝ → Prop, l4 = parallel_line_through_A :=
sorry

end NUMINAMATH_GPT_find_lines_and_intersections_l1873_187361


namespace NUMINAMATH_GPT_fixed_real_root_l1873_187334

theorem fixed_real_root (k x : ℝ) (h : x^2 + (k + 3) * x + (k + 2) = 0) : x = -1 :=
sorry

end NUMINAMATH_GPT_fixed_real_root_l1873_187334


namespace NUMINAMATH_GPT_divisor_of_a_l1873_187356

theorem divisor_of_a (a b c d : ℕ) (h1 : Nat.gcd a b = 18) (h2 : Nat.gcd b c = 45) 
  (h3 : Nat.gcd c d = 75) (h4 : 80 < Nat.gcd d a ∧ Nat.gcd d a < 120) : 
  7 ∣ a :=
by
  sorry

end NUMINAMATH_GPT_divisor_of_a_l1873_187356


namespace NUMINAMATH_GPT_initial_average_quiz_score_l1873_187349

theorem initial_average_quiz_score 
  (n : ℕ) (A : ℝ) (dropped_avg : ℝ) (drop_score : ℝ)
  (students_before : n = 16)
  (students_after : n - 1 = 15)
  (dropped_avg_eq : dropped_avg = 64.0)
  (drop_score_eq : drop_score = 8) 
  (total_sum_before_eq : n * A = 16 * A)
  (total_sum_after_eq : (n - 1) * dropped_avg = 15 * 64):
  A = 60.5 := 
by
  sorry

end NUMINAMATH_GPT_initial_average_quiz_score_l1873_187349


namespace NUMINAMATH_GPT_circles_intersect_at_two_points_l1873_187321

noncomputable def point_intersection_count (A B : ℝ × ℝ) (rA rB d : ℝ) : ℕ :=
  let distance := (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2
  if rA + rB >= d ∧ d >= |rA - rB| then 2 else if d = rA + rB ∨ d = |rA - rB| then 1 else 0

theorem circles_intersect_at_two_points :
  point_intersection_count (0, 0) (8, 0) 3 6 8 = 2 :=
by 
  -- Proof for the statement will go here
  sorry

end NUMINAMATH_GPT_circles_intersect_at_two_points_l1873_187321


namespace NUMINAMATH_GPT_tens_digit_of_2023_pow_2024_minus_2025_l1873_187378

theorem tens_digit_of_2023_pow_2024_minus_2025 : 
  ∀ (n : ℕ), n = 2023^2024 - 2025 → ((n % 100) / 10) = 0 :=
by
  intros n h
  sorry

end NUMINAMATH_GPT_tens_digit_of_2023_pow_2024_minus_2025_l1873_187378


namespace NUMINAMATH_GPT_min_value_of_frac_l1873_187314

theorem min_value_of_frac (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : 2 * m + n = 1) (hm : m > 0) (hn : n > 0) :
  (1 / m) + (2 / n) = 8 :=
sorry

end NUMINAMATH_GPT_min_value_of_frac_l1873_187314


namespace NUMINAMATH_GPT_expressions_equal_constant_generalized_identity_l1873_187363

noncomputable def expr1 := (Real.sin (13 * Real.pi / 180))^2 + (Real.cos (17 * Real.pi / 180))^2 - Real.sin (13 * Real.pi / 180) * Real.cos (17 * Real.pi / 180)
noncomputable def expr2 := (Real.sin (15 * Real.pi / 180))^2 + (Real.cos (15 * Real.pi / 180))^2 - Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)
noncomputable def expr3 := (Real.sin (-18 * Real.pi / 180))^2 + (Real.cos (48 * Real.pi / 180))^2 - Real.sin (-18 * Real.pi / 180) * Real.cos (48 * Real.pi / 180)
noncomputable def expr4 := (Real.sin (-25 * Real.pi / 180))^2 + (Real.cos (55 * Real.pi / 180))^2 - Real.sin (-25 * Real.pi / 180) * Real.cos (55 * Real.pi / 180)

theorem expressions_equal_constant :
  expr1 = 3/4 ∧ expr2 = 3/4 ∧ expr3 = 3/4 ∧ expr4 = 3/4 :=
sorry

theorem generalized_identity (α : ℝ) :
  (Real.sin α)^2 + (Real.cos (30 * Real.pi / 180 - α))^2 - Real.sin α * Real.cos (30 * Real.pi / 180 - α) = 3 / 4 :=
sorry

end NUMINAMATH_GPT_expressions_equal_constant_generalized_identity_l1873_187363


namespace NUMINAMATH_GPT_curve_is_line_l1873_187379

theorem curve_is_line (r θ : ℝ) (h : r = 2 / (Real.sin θ + Real.cos θ)) : 
  ∃ m b, ∀ θ, r * Real.cos θ = m * (r * Real.sin θ) + b :=
sorry

end NUMINAMATH_GPT_curve_is_line_l1873_187379


namespace NUMINAMATH_GPT_lottery_probability_l1873_187398

theorem lottery_probability :
  let megaBallProbability := 1 / 30
  let winnerBallCombination := Nat.choose 50 5
  let winnerBallProbability := 1 / winnerBallCombination
  megaBallProbability * winnerBallProbability = 1 / 63562800 :=
by
  let megaBallProbability := 1 / 30
  let winnerBallCombination := Nat.choose 50 5
  have winnerBallCombinationEval: winnerBallCombination = 2118760 := by sorry
  let winnerBallProbability := 1 / winnerBallCombination
  have totalProbability: megaBallProbability * winnerBallProbability = 1 / 63562800 := by sorry
  exact totalProbability

end NUMINAMATH_GPT_lottery_probability_l1873_187398


namespace NUMINAMATH_GPT_part1_part2_l1873_187368

variable (a b : ℝ)

theorem part1 (h : |a - 3| + |b + 6| = 0) : a + b - 2 = -5 := sorry

theorem part2 (h : |a - 3| + |b + 6| = 0) : a - b - 2 = 7 := sorry

end NUMINAMATH_GPT_part1_part2_l1873_187368


namespace NUMINAMATH_GPT_partition_nats_100_subsets_l1873_187343

theorem partition_nats_100_subsets :
  ∃ (S : ℕ → ℕ), (∀ n, 1 ≤ S n ∧ S n ≤ 100) ∧
    (∀ a b c : ℕ, a + 99 * b = c → S a = S c ∨ S a = S b ∨ S b = S c) :=
by
  sorry

end NUMINAMATH_GPT_partition_nats_100_subsets_l1873_187343


namespace NUMINAMATH_GPT_lesser_number_is_14_l1873_187308

theorem lesser_number_is_14 (x y : ℕ) (h₀ : x + y = 60) (h₁ : 4 * y - x = 10) : y = 14 :=
by 
  sorry

end NUMINAMATH_GPT_lesser_number_is_14_l1873_187308


namespace NUMINAMATH_GPT_question1_solution_question2_solution_l1873_187351

-- Define the function f for any value of a
def f (a : ℝ) (x : ℝ) : ℝ :=
  abs (x + 1) - abs (a * x - 1)

-- Definition specifically for question (1) setting a = 1
def f1 (x : ℝ) : ℝ :=
  f 1 x

-- Definition of the set for the inequality in (1)
def solution_set_1 : Set ℝ :=
  { x | f1 x > 1 }

-- Theorem for question (1)
theorem question1_solution :
  solution_set_1 = { x : ℝ | x > 1/2 } :=
sorry

-- Condition for question (2)
def inequality_condition (a : ℝ) (x : ℝ) : Prop :=
  f a x > x

-- Define the interval for x in question (2)
def interval_0_1 (x : ℝ) : Prop :=
  0 < x ∧ x < 1

-- Theorem for question (2)
theorem question2_solution {a : ℝ} :
  (∀ x ∈ {x | interval_0_1 x}, inequality_condition a x) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_GPT_question1_solution_question2_solution_l1873_187351


namespace NUMINAMATH_GPT_value_of_a2019_l1873_187399

noncomputable def a : ℕ → ℝ
| 0 => 3
| (n + 1) => 1 / (1 - a n)

theorem value_of_a2019 : a 2019 = 2 / 3 :=
sorry

end NUMINAMATH_GPT_value_of_a2019_l1873_187399


namespace NUMINAMATH_GPT_king_gvidon_descendants_l1873_187371

def number_of_sons : ℕ := 5
def number_of_descendants_with_sons : ℕ := 100
def number_of_sons_each : ℕ := 3
def number_of_grandsons : ℕ := number_of_descendants_with_sons * number_of_sons_each

def total_descendants : ℕ := number_of_sons + number_of_grandsons

theorem king_gvidon_descendants : total_descendants = 305 :=
by
  sorry

end NUMINAMATH_GPT_king_gvidon_descendants_l1873_187371


namespace NUMINAMATH_GPT_smallest_a_b_sum_l1873_187374

theorem smallest_a_b_sum :
  ∃ (a b : ℕ), 3^6 * 5^3 * 7^2 = a^b ∧ a + b = 317 := 
sorry

end NUMINAMATH_GPT_smallest_a_b_sum_l1873_187374


namespace NUMINAMATH_GPT_value_of_m_l1873_187326

theorem value_of_m (x m : ℝ) (h : 2 * x + m - 6 = 0) (hx : x = 1) : m = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l1873_187326


namespace NUMINAMATH_GPT_total_sharks_l1873_187373

-- Define the number of sharks at each beach.
def N : ℕ := 22
def D : ℕ := 4 * N
def H : ℕ := D / 2

-- Proof that the total number of sharks on the three beaches is 154.
theorem total_sharks : N + D + H = 154 := by
  sorry

end NUMINAMATH_GPT_total_sharks_l1873_187373


namespace NUMINAMATH_GPT_find_difference_l1873_187300

variables (a b c : ℝ)

theorem find_difference (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 50) : c - a = 10 := by
  sorry

end NUMINAMATH_GPT_find_difference_l1873_187300


namespace NUMINAMATH_GPT_matches_played_by_team_B_from_city_A_l1873_187370

-- Define the problem setup, conditions, and the conclusion we need to prove
structure Tournament :=
  (cities : ℕ)
  (teams_per_city : ℕ)

-- Assuming each team except Team A of city A has played a unique number of matches,
-- find the number of matches played by Team B of city A.
theorem matches_played_by_team_B_from_city_A (t : Tournament)
  (unique_match_counts_except_A : ∀ (i j : ℕ), i ≠ j → (i < t.cities → (t.teams_per_city * i ≠ t.teams_per_city * j)) ∧ (i < t.cities - 1 → (t.teams_per_city * i ≠ t.teams_per_city * (t.cities - 1)))) :
  (t.cities = 16) → (t.teams_per_city = 2) → ∃ n, n = 15 :=
by
  sorry

end NUMINAMATH_GPT_matches_played_by_team_B_from_city_A_l1873_187370


namespace NUMINAMATH_GPT_expression_value_l1873_187369

theorem expression_value
  (x y a b : ℤ)
  (h1 : x = 1)
  (h2 : y = 2)
  (h3 : a + 2 * b = 3) :
  2 * a + 4 * b - 5 = 1 := 
by sorry

end NUMINAMATH_GPT_expression_value_l1873_187369


namespace NUMINAMATH_GPT_problems_left_to_grade_l1873_187332

-- Defining all the conditions
def worksheets_total : ℕ := 14
def worksheets_graded : ℕ := 7
def problems_per_worksheet : ℕ := 2

-- Stating the proof problem
theorem problems_left_to_grade : 
  (worksheets_total - worksheets_graded) * problems_per_worksheet = 14 := 
by
  sorry

end NUMINAMATH_GPT_problems_left_to_grade_l1873_187332


namespace NUMINAMATH_GPT_gumball_sharing_l1873_187313

theorem gumball_sharing (init_j : ℕ) (init_jq : ℕ) (mult_j : ℕ) (mult_jq : ℕ) :
  init_j = 40 → init_jq = 60 → mult_j = 5 → mult_jq = 3 →
  (init_j + mult_j * init_j + init_jq + mult_jq * init_jq) / 2 = 240 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_gumball_sharing_l1873_187313


namespace NUMINAMATH_GPT_solve_equation_2021_l1873_187318

theorem solve_equation_2021 (x : ℝ) (hx : 0 ≤ x) : 
  2021 * x = 2022 * (x ^ (2021 : ℕ)) ^ (1 / (2021 : ℕ)) - 1 → x = 1 := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_2021_l1873_187318


namespace NUMINAMATH_GPT_smaller_angle_measure_l1873_187392

theorem smaller_angle_measure (x : ℝ) (h1 : 3 * x + 2 * x = 90) : 2 * x = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_smaller_angle_measure_l1873_187392


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l1873_187342

-- We are proving that the factorization of m^2 - 9 is equal to (m+3)(m-3)
theorem factorize_difference_of_squares (m : ℝ) : m ^ 2 - 9 = (m + 3) * (m - 3) := 
by 
  sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l1873_187342


namespace NUMINAMATH_GPT_least_number_of_trees_l1873_187330

theorem least_number_of_trees (n : ℕ) :
  (∃ k₄ k₅ k₆, n = 4 * k₄ ∧ n = 5 * k₅ ∧ n = 6 * k₆) ↔ n = 60 :=
by 
  sorry

end NUMINAMATH_GPT_least_number_of_trees_l1873_187330


namespace NUMINAMATH_GPT_simplify_expression_l1873_187377

variable (x : ℝ)

theorem simplify_expression : 2 * x - 3 * (2 - x) + 4 * (2 + x) - 5 * (1 - 3 * x) = 24 * x - 3 := 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1873_187377


namespace NUMINAMATH_GPT_sqrt_64_eq_8_l1873_187360

theorem sqrt_64_eq_8 : Real.sqrt 64 = 8 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_64_eq_8_l1873_187360


namespace NUMINAMATH_GPT_find_three_digit_number_l1873_187357

theorem find_three_digit_number : 
  ∀ (c d e : ℕ), 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10 ∧ 0 ≤ e ∧ e < 10 ∧ 
  (10 * c + d) / 99 + (100 * c + 10 * d + e) / 999 = 44 / 99 → 
  100 * c + 10 * d + e = 400 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_three_digit_number_l1873_187357


namespace NUMINAMATH_GPT_find_k_l1873_187375

noncomputable def vector_a : ℝ × ℝ := (3, 1)
noncomputable def vector_b : ℝ × ℝ := (1, 0)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (vector_a.1 + k * vector_b.1, vector_a.2 + k * vector_b.2)

theorem find_k (k : ℝ) (h : vector_a.1 * (vector_a.1 + k * vector_b.1) + vector_a.2 * (vector_a.2 + k * vector_b.2) = 0) : k = -10 / 3 :=
by sorry

end NUMINAMATH_GPT_find_k_l1873_187375


namespace NUMINAMATH_GPT_max_value_sqrt_sum_l1873_187324

theorem max_value_sqrt_sum {x y z : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  ∃ (M : ℝ), M = (Real.sqrt (abs (x - y)) + Real.sqrt (abs (y - z)) + Real.sqrt (abs (z - x))) ∧ M = Real.sqrt 2 + 1 :=
by sorry

end NUMINAMATH_GPT_max_value_sqrt_sum_l1873_187324


namespace NUMINAMATH_GPT_total_weight_of_7_moles_CaO_l1873_187348

/-- Definitions necessary for the problem --/
def atomic_weight_Ca : ℝ := 40.08 -- atomic weight of calcium in g/mol
def atomic_weight_O : ℝ := 16.00 -- atomic weight of oxygen in g/mol
def molecular_weight_CaO : ℝ := atomic_weight_Ca + atomic_weight_O -- molecular weight of CaO in g/mol
def number_of_moles_CaO : ℝ := 7 -- number of moles of CaO

/-- The main theorem statement --/
theorem total_weight_of_7_moles_CaO :
  molecular_weight_CaO * number_of_moles_CaO = 392.56 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_of_7_moles_CaO_l1873_187348


namespace NUMINAMATH_GPT_find_k_l1873_187329

theorem find_k (k a : ℤ)
  (h₁ : 49 + k = a^2)
  (h₂ : 361 + k = (a + 2)^2)
  (h₃ : 784 + k = (a + 4)^2) :
  k = 6035 :=
by sorry

end NUMINAMATH_GPT_find_k_l1873_187329


namespace NUMINAMATH_GPT_remainder_of_98_pow_50_mod_50_l1873_187338

theorem remainder_of_98_pow_50_mod_50 : (98 ^ 50) % 50 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_of_98_pow_50_mod_50_l1873_187338


namespace NUMINAMATH_GPT_simplify_expression_l1873_187312

theorem simplify_expression :
  (4.625 - 13/18 * 9/26) / (9/4) + 2.5 / 1.25 / 6.75 / 1 + 53/68 / ((1/2 - 0.375) / 0.125 + (5/6 - 7/12) / (0.358 - 1.4796 / 13.7)) = 17/27 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1873_187312


namespace NUMINAMATH_GPT_Jill_earnings_l1873_187382

theorem Jill_earnings :
  let earnings_first_month := 10 * 30
  let earnings_second_month := 20 * 30
  let earnings_third_month := 20 * 15
  earnings_first_month + earnings_second_month + earnings_third_month = 1200 :=
by
  sorry

end NUMINAMATH_GPT_Jill_earnings_l1873_187382


namespace NUMINAMATH_GPT_prime_divisors_of_390_l1873_187352

theorem prime_divisors_of_390 : 
  (2 * 195 = 390) → 
  (3 * 65 = 195) → 
  (5 * 13 = 65) → 
  ∃ (S : Finset ℕ), 
    (∀ p ∈ S, Nat.Prime p) ∧ 
    (S.card = 4) ∧ 
    (∀ d ∈ S, d ∣ 390) := 
by
  sorry

end NUMINAMATH_GPT_prime_divisors_of_390_l1873_187352


namespace NUMINAMATH_GPT_students_interested_in_both_l1873_187305

def numberOfStudentsInterestedInBoth (T S M N: ℕ) : ℕ := 
  S + M - (T - N)

theorem students_interested_in_both (T S M N: ℕ) (hT : T = 55) (hS : S = 43) (hM : M = 34) (hN : N = 4) : 
  numberOfStudentsInterestedInBoth T S M N = 26 := 
by 
  rw [hT, hS, hM, hN]
  sorry

end NUMINAMATH_GPT_students_interested_in_both_l1873_187305


namespace NUMINAMATH_GPT_smallest_ducks_l1873_187307

theorem smallest_ducks :
  ∃ D : ℕ, 
  ∃ C : ℕ, 
  ∃ H : ℕ, 
  (13 * D = 17 * C) ∧
  (11 * H = (6 / 5) * 13 * D) ∧
  (17 * C = (3 / 8) * 11 * H) ∧ 
  (13 * D = 520) :=
by 
  sorry

end NUMINAMATH_GPT_smallest_ducks_l1873_187307


namespace NUMINAMATH_GPT_estimate_total_children_l1873_187359

variables (k m n T : ℕ)

/-- There are k children initially given red ribbons. 
    Then m children are randomly selected, 
    and n of them have red ribbons. -/

theorem estimate_total_children (h : n * T = k * m) : T = k * m / n :=
by sorry

end NUMINAMATH_GPT_estimate_total_children_l1873_187359


namespace NUMINAMATH_GPT_sixty_percent_is_240_l1873_187347

variable (x : ℝ)

-- Conditions
def forty_percent_eq_160 : Prop := 0.40 * x = 160

-- Proof problem
theorem sixty_percent_is_240 (h : forty_percent_eq_160 x) : 0.60 * x = 240 :=
sorry

end NUMINAMATH_GPT_sixty_percent_is_240_l1873_187347


namespace NUMINAMATH_GPT_sum_of_first_9_terms_l1873_187302

variable {a : ℕ → ℝ} -- the arithmetic sequence
variable {S : ℕ → ℝ} -- the sum function

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

axiom arithmetic_sequence_condition (h : is_arithmetic_sequence a) : a 5 = 2

theorem sum_of_first_9_terms (h : is_arithmetic_sequence a) (h5: a 5 = 2) : sum_of_first_n_terms a 9 = 18 := by
  sorry

end NUMINAMATH_GPT_sum_of_first_9_terms_l1873_187302


namespace NUMINAMATH_GPT_star_in_S_star_associative_l1873_187322

def S (x : ℕ) : Prop :=
  x > 1 ∧ x % 2 = 1

def f (x : ℕ) : ℕ :=
  Nat.log2 x

def star (a b : ℕ) : ℕ :=
  a + 2 ^ (f a) * (b - 3)

theorem star_in_S (a b : ℕ) (h_a : S a) (h_b : S b) : S (star a b) :=
  sorry

theorem star_associative (a b c : ℕ) (h_a : S a) (h_b : S b) (h_c : S c) :
  star (star a b) c = star a (star b c) :=
  sorry

end NUMINAMATH_GPT_star_in_S_star_associative_l1873_187322


namespace NUMINAMATH_GPT_smallest_divisor_l1873_187397

theorem smallest_divisor (N D : ℕ) (hN : N = D * 7) (hD : D > 0) (hsq : (N / D) = 7) :
  D = 7 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_divisor_l1873_187397


namespace NUMINAMATH_GPT_find_N_l1873_187311

theorem find_N
  (N : ℕ)
  (h : (4 / 10 : ℝ) * (16 / (16 + N : ℝ)) + (6 / 10 : ℝ) * (N / (16 + N : ℝ)) = 0.58) :
  N = 144 :=
sorry

end NUMINAMATH_GPT_find_N_l1873_187311


namespace NUMINAMATH_GPT_max_value_of_p_l1873_187331

theorem max_value_of_p
  (p q r s : ℕ)
  (h1 : p < 3 * q)
  (h2 : q < 4 * r)
  (h3 : r < 5 * s)
  (h4 : s < 90)
  (h5 : 0 < s)
  (h6 : 0 < r)
  (h7 : 0 < q)
  (h8 : 0 < p):
  p ≤ 5324 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_p_l1873_187331


namespace NUMINAMATH_GPT_pencils_in_each_box_l1873_187396

theorem pencils_in_each_box (total_pencils : ℕ) (total_boxes : ℕ) (pencils_per_box : ℕ) 
  (h1 : total_pencils = 648) (h2 : total_boxes = 162) : 
  total_pencils / total_boxes = pencils_per_box := 
by
  sorry

end NUMINAMATH_GPT_pencils_in_each_box_l1873_187396


namespace NUMINAMATH_GPT_bonnets_per_orphanage_l1873_187337

/--
Mrs. Young makes bonnets for kids in the orphanage.
On Monday, she made 10 bonnets.
On Tuesday and Wednesday combined she made twice more than on Monday.
On Thursday she made 5 more than on Monday.
On Friday she made 5 less than on Thursday.
She divided up the bonnets evenly and sent them to 5 orphanages.
Prove that the number of bonnets Mrs. Young sent to each orphanage is 11.
-/
theorem bonnets_per_orphanage :
  let monday := 10
  let tuesday_wednesday := 2 * monday
  let thursday := monday + 5
  let friday := thursday - 5
  let total_bonnets := monday + tuesday_wednesday + thursday + friday
  let orphanages := 5
  total_bonnets / orphanages = 11 :=
by
  sorry

end NUMINAMATH_GPT_bonnets_per_orphanage_l1873_187337


namespace NUMINAMATH_GPT_rainy_days_last_week_l1873_187381

-- All conditions in Lean definitions
def even_integer (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def cups_of_tea_n (n : ℤ) : ℤ := 3
def total_drinks (R NR : ℤ) (m : ℤ) : Prop := 2 * m * R + 3 * NR = 36
def more_tea_than_hot_chocolate (R NR : ℤ) (m : ℤ) : Prop := 3 * NR - 2 * m * R = 12
def odd_number_of_rainy_days (R : ℤ) : Prop := R % 2 = 1
def total_days_in_week (R NR : ℤ) : Prop := R + NR = 7

-- Main statement
theorem rainy_days_last_week : ∃ R m NR : ℤ, 
  odd_number_of_rainy_days R ∧ 
  total_days_in_week R NR ∧ 
  total_drinks R NR m ∧ 
  more_tea_than_hot_chocolate R NR m ∧
  R = 3 :=
by
  sorry

end NUMINAMATH_GPT_rainy_days_last_week_l1873_187381


namespace NUMINAMATH_GPT_solution_set_inequality_l1873_187387

theorem solution_set_inequality (a x : ℝ) (h : a > 0) :
  (∀ x, (a + 1 ≤ x ∧ x ≤ a + 3) ↔ (|((2 * x - 3 - 2 * a) / (x - a))| ≤ 1)) := 
sorry

end NUMINAMATH_GPT_solution_set_inequality_l1873_187387


namespace NUMINAMATH_GPT_pioneer_ages_l1873_187384

def pioneer_data (Burov Gridnev Klimenko Kolya Petya Grisha : String) (Burov_age Gridnev_age Klimenko_age Petya_age Grisha_age : ℕ) :=
  Burov ≠ Kolya ∧
  Petya_age = 12 ∧
  Gridnev_age = Petya_age + 1 ∧
  Grisha_age = Petya_age + 1 ∧
  Burov_age = Grisha_age ∧
-- defining the names corresponding to conditions given in problem
  Burov = Grisha ∧ Gridnev = Kolya ∧ Klimenko = Petya 

theorem pioneer_ages (Burov Gridnev Klimenko Kolya Petya Grisha : String) (Burov_age Gridnev_age Klimenko_age Petya_age Grisha_age : ℕ)
  (h : pioneer_data Burov Gridnev Klimenko Kolya Petya Grisha Burov_age Gridnev_age Klimenko_age Petya_age Grisha_age) :
  (Burov, Burov_age) = (Grisha, 13) ∧ 
  (Gridnev, Gridnev_age) = (Kolya, 13) ∧ 
  (Klimenko, Klimenko_age) = (Petya, 12) :=
by
  sorry

end NUMINAMATH_GPT_pioneer_ages_l1873_187384


namespace NUMINAMATH_GPT_compare_travel_times_l1873_187366

variable (v : ℝ) (t1 t2 : ℝ)

def travel_time_first := t1 = 100 / v
def travel_time_second := t2 = 200 / v

theorem compare_travel_times (h1 : travel_time_first v t1) (h2 : travel_time_second v t2) : 
  t2 = 2 * t1 :=
by
  sorry

end NUMINAMATH_GPT_compare_travel_times_l1873_187366


namespace NUMINAMATH_GPT_simplify_expression_l1873_187325

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
    ((x^2 + 1) / (x - 1)) - (2 * x / (x - 1)) = x - 1 := 
by
    sorry

end NUMINAMATH_GPT_simplify_expression_l1873_187325


namespace NUMINAMATH_GPT_remainder_t100_mod_7_l1873_187340

theorem remainder_t100_mod_7 :
  ∀ T : ℕ → ℕ, (T 1 = 3) →
  (∀ n : ℕ, n > 1 → T n = 3 ^ (T (n - 1))) →
  (T 100 % 7 = 6) :=
by
  intro T h1 h2
  -- sorry to skip the actual proof
  sorry

end NUMINAMATH_GPT_remainder_t100_mod_7_l1873_187340


namespace NUMINAMATH_GPT_sides_of_polygons_l1873_187372

theorem sides_of_polygons (p : ℕ) (γ : ℝ) (n1 n2 : ℕ) (h1 : p = 5) (h2 : γ = 12 / 7) 
    (h3 : n2 = n1 + p) 
    (h4 : 360 / n1 - 360 / n2 = γ) : 
    n1 = 30 ∧ n2 = 35 := 
  sorry

end NUMINAMATH_GPT_sides_of_polygons_l1873_187372


namespace NUMINAMATH_GPT_num_distinct_terms_expansion_a_b_c_10_l1873_187336

-- Define the expansion of (a+b+c)^10
def num_distinct_terms_expansion (n : ℕ) : ℕ :=
  Nat.choose (n + 3 - 1) (3 - 1)

-- Theorem statement
theorem num_distinct_terms_expansion_a_b_c_10 : num_distinct_terms_expansion 10 = 66 :=
by
  sorry

end NUMINAMATH_GPT_num_distinct_terms_expansion_a_b_c_10_l1873_187336


namespace NUMINAMATH_GPT_classroomA_goal_is_200_l1873_187304

def classroomA_fundraising_goal : ℕ :=
  let amount_from_two_families := 2 * 20
  let amount_from_eight_families := 8 * 10
  let amount_from_ten_families := 10 * 5
  let total_raised := amount_from_two_families + amount_from_eight_families + amount_from_ten_families
  let amount_needed := 30
  total_raised + amount_needed

theorem classroomA_goal_is_200 : classroomA_fundraising_goal = 200 := by
  sorry

end NUMINAMATH_GPT_classroomA_goal_is_200_l1873_187304


namespace NUMINAMATH_GPT_number_of_logs_in_stack_l1873_187309

theorem number_of_logs_in_stack :
  let bottom := 15
  let top := 4
  let num_rows := bottom - top + 1
  let total_logs := num_rows * (bottom + top) / 2
  total_logs = 114 := by
{
  let bottom := 15
  let top := 4
  let num_rows := bottom - top + 1
  let total_logs := num_rows * (bottom + top) / 2
  sorry
}

end NUMINAMATH_GPT_number_of_logs_in_stack_l1873_187309


namespace NUMINAMATH_GPT_range_positive_of_odd_increasing_l1873_187393

-- Define f as an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Define f as an increasing function on (-∞,0)
def is_increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → y < 0 → f (x) < f (y)

-- Given an odd function that is increasing on (-∞,0) and f(-1) = 0, prove the range of x for which f(x) > 0 is (-1, 0) ∪ (1, +∞)
theorem range_positive_of_odd_increasing (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_increasing : is_increasing_on_neg f)
  (h_f_neg_one : f (-1) = 0) :
  {x : ℝ | f x > 0} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 1 < x} :=
by
  sorry

end NUMINAMATH_GPT_range_positive_of_odd_increasing_l1873_187393


namespace NUMINAMATH_GPT_min_rows_required_to_seat_students_l1873_187350

-- Definitions based on the conditions
def seats_per_row : ℕ := 168
def total_students : ℕ := 2016
def max_students_per_school : ℕ := 40

def min_number_of_rows : ℕ :=
  -- Given that the minimum number of rows required to seat all students following the conditions is 15
  15

-- Lean statement expressing the proof problem
theorem min_rows_required_to_seat_students :
  ∃ rows : ℕ, rows = min_number_of_rows ∧
  (∀ school_sizes : List ℕ, (∀ size ∈ school_sizes, size ≤ max_students_per_school)
    → (List.sum school_sizes = total_students)
    → ∀ school_arrangement : List (List ℕ), 
        (∀ row_sizes ∈ school_arrangement, List.sum row_sizes ≤ seats_per_row) 
        → List.length school_arrangement ≤ rows) :=
sorry

end NUMINAMATH_GPT_min_rows_required_to_seat_students_l1873_187350


namespace NUMINAMATH_GPT_consumption_increase_l1873_187353

theorem consumption_increase (T C : ℝ) (P : ℝ) (h : 0.82 * (1 + P / 100) = 0.943) :
  P = 15.06 := by
  sorry

end NUMINAMATH_GPT_consumption_increase_l1873_187353


namespace NUMINAMATH_GPT_area_of_region_l1873_187306

theorem area_of_region :
  (∫ x, ∫ y in {y : ℝ | x^4 + y^4 = |x|^3 + |y|^3}, (1 : ℝ)) = 4 :=
sorry

end NUMINAMATH_GPT_area_of_region_l1873_187306


namespace NUMINAMATH_GPT_number_of_boys_in_class_l1873_187394

theorem number_of_boys_in_class (n : ℕ)
  (avg_height : ℕ) (incorrect_height : ℕ) (actual_height : ℕ)
  (actual_avg_height : ℕ)
  (h1 : avg_height = 180)
  (h2 : incorrect_height = 156)
  (h3 : actual_height = 106)
  (h4 : actual_avg_height = 178)
  : n = 25 :=
by 
  -- We have the following conditions:
  -- Incorrect total height = avg_height * n
  -- Difference due to incorrect height = incorrect_height - actual_height
  -- Correct total height = avg_height * n - (incorrect_height - actual_height)
  -- Total height according to actual average = actual_avg_height * n
  -- Equating both, we have:
  -- avg_height * n - (incorrect_height - actual_height) = actual_avg_height * n
  -- We know avg_height, incorrect_height, actual_height, actual_avg_height from h1, h2, h3, h4
  -- Substituting these values and solving:
  -- 180n - (156 - 106) = 178n
  -- 180n - 50 = 178n
  -- 2n = 50
  -- n = 25
  sorry

end NUMINAMATH_GPT_number_of_boys_in_class_l1873_187394


namespace NUMINAMATH_GPT_quadratic_discriminant_l1873_187341

theorem quadratic_discriminant {a b c : ℝ} (h : (a + b + c) * c ≤ 0) : b^2 ≥ 4 * a * c :=
sorry

end NUMINAMATH_GPT_quadratic_discriminant_l1873_187341


namespace NUMINAMATH_GPT_range_of_m_l1873_187395

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y - x * y = 0) : 
  ∀ m : ℝ, (xy ≥ m^2 - 6 * m ↔ -2 ≤ m ∧ m ≤ 8) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1873_187395


namespace NUMINAMATH_GPT_park_will_have_9_oak_trees_l1873_187328

def current_oak_trees : Nat := 5
def additional_oak_trees : Nat := 4
def total_oak_trees : Nat := current_oak_trees + additional_oak_trees

theorem park_will_have_9_oak_trees : total_oak_trees = 9 :=
by
  sorry

end NUMINAMATH_GPT_park_will_have_9_oak_trees_l1873_187328


namespace NUMINAMATH_GPT_cookies_eaten_l1873_187301

theorem cookies_eaten (original remaining : ℕ) (h_original : original = 18) (h_remaining : remaining = 9) :
    original - remaining = 9 := by
  sorry

end NUMINAMATH_GPT_cookies_eaten_l1873_187301


namespace NUMINAMATH_GPT_A_n_eq_B_n_l1873_187315

open Real

noncomputable def A_n (n : ℕ) : ℝ :=
  1408 * (1 - (1 / (2 : ℝ) ^ n))

noncomputable def B_n (n : ℕ) : ℝ :=
  (3968 / 3) * (1 - (1 / (-2 : ℝ) ^ n))

theorem A_n_eq_B_n : A_n 5 = B_n 5 := sorry

end NUMINAMATH_GPT_A_n_eq_B_n_l1873_187315


namespace NUMINAMATH_GPT_minimize_transportation_cost_l1873_187303

noncomputable def transportation_cost (x : ℝ) (distance : ℝ) (k : ℝ) (other_expense : ℝ) : ℝ :=
  k * (x * distance / x^2 + other_expense * distance / x)

theorem minimize_transportation_cost :
  ∀ (distance : ℝ) (max_speed : ℝ) (k : ℝ) (other_expense : ℝ) (x : ℝ),
  0 < x ∧ x ≤ max_speed ∧ max_speed = 50 ∧ distance = 300 ∧ k = 0.5 ∧ other_expense = 800 →
  transportation_cost x distance k other_expense = 150 * (x + 1600 / x) ∧
  (∀ y, (0 < y ∧ y ≤ max_speed) → transportation_cost y distance k other_expense ≥ 12000) ∧
  (transportation_cost 40 distance k other_expense = 12000)
  := 
  by intros distance max_speed k other_expense x H;
     sorry

end NUMINAMATH_GPT_minimize_transportation_cost_l1873_187303


namespace NUMINAMATH_GPT_cos_F_in_triangle_l1873_187390

theorem cos_F_in_triangle (D E F : ℝ) (sin_D : ℝ) (cos_E : ℝ) (cos_F : ℝ) 
  (h1 : sin_D = 4 / 5) 
  (h2 : cos_E = 12 / 13) 
  (D_plus_E_plus_F : D + E + F = π) :
  cos_F = -16 / 65 :=
by
  sorry

end NUMINAMATH_GPT_cos_F_in_triangle_l1873_187390


namespace NUMINAMATH_GPT_number_of_female_democrats_l1873_187310

-- Definitions and conditions
variables (F M D_F D_M D_T : ℕ)
axiom participant_total : F + M = 780
axiom female_democrats : D_F = 1 / 2 * F
axiom male_democrats : D_M = 1 / 4 * M
axiom total_democrats : D_T = 1 / 3 * (F + M)

-- Target statement to be proven
theorem number_of_female_democrats : D_T = 260 → D_F = 130 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_of_female_democrats_l1873_187310


namespace NUMINAMATH_GPT_bill_bathroom_visits_per_day_l1873_187327

theorem bill_bathroom_visits_per_day
  (squares_per_use : ℕ)
  (rolls : ℕ)
  (squares_per_roll : ℕ)
  (days_supply : ℕ)
  (total_uses : squares_per_use = 5)
  (total_rolls : rolls = 1000)
  (squares_from_each_roll : squares_per_roll = 300)
  (total_days : days_supply = 20000) :
  ( (rolls * squares_per_roll) / days_supply / squares_per_use ) = 3 :=
by
  sorry

end NUMINAMATH_GPT_bill_bathroom_visits_per_day_l1873_187327


namespace NUMINAMATH_GPT_GE_eq_GH_l1873_187323

variables (A B C D E F G H : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
          [Inhabited E] [Inhabited F] [Inhabited G] [Inhabited H]
          
variables (AC : Line A C) (AB : Line A B) (BE : Line B E) (DE : Line D E)
          (BG : Line B G) (AF : Line A F) (DE' : Line D E') (angleC : Angle C = 90)

variables (circB : Circle B BC) (tangentDE : Tangent DE circB E) (perpAB : Perpendicular AC AB)
          (intersectionF : Intersect (PerpendicularLine C AB) BE F)
          (intersectionG : Intersect AF DE G) (intersectionH : Intersect (ParallelLine A BG) DE H)

theorem GE_eq_GH : GE = GH := sorry

end NUMINAMATH_GPT_GE_eq_GH_l1873_187323


namespace NUMINAMATH_GPT_sum_first_seven_terms_is_28_l1873_187385

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the arithmetic sequence 
def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n + d

-- Given conditions
axiom a2_a4_a6_sum : a 2 + a 4 + a 6 = 12

-- Prove that the sum of the first seven terms is 28
theorem sum_first_seven_terms_is_28 (h : is_arithmetic_seq a d) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
sorry

end NUMINAMATH_GPT_sum_first_seven_terms_is_28_l1873_187385


namespace NUMINAMATH_GPT_gain_percent_l1873_187346

theorem gain_percent (C S : ℝ) (h : 50 * C = 30 * S) : ((S - C) / C) * 100 = 200 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_gain_percent_l1873_187346


namespace NUMINAMATH_GPT_find_f_neg2_l1873_187333

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem find_f_neg2 : f (-2) = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg2_l1873_187333


namespace NUMINAMATH_GPT_student_chose_124_l1873_187354

theorem student_chose_124 (x : ℤ) (h : 2 * x - 138 = 110) : x = 124 := 
by {
  sorry
}

end NUMINAMATH_GPT_student_chose_124_l1873_187354


namespace NUMINAMATH_GPT_gnome_count_l1873_187365

theorem gnome_count (g_R: ℕ) (g_W: ℕ) (h1: g_R = 4 * g_W) (h2: g_W = 20) : g_R - (40 * g_R / 100) = 48 := by
  sorry

end NUMINAMATH_GPT_gnome_count_l1873_187365


namespace NUMINAMATH_GPT_golden_ticket_problem_l1873_187339

open Real

/-- The golden ratio -/
noncomputable def φ := (1 + sqrt 5) / 2

/-- Assume the proportions and the resulting area -/
theorem golden_ticket_problem
  (a b : ℝ)
  (h : 0 + b * φ = 
        φ - (5 + sqrt 5) / (8 * φ)) :
  b / a = -4 / 3 :=
  sorry

end NUMINAMATH_GPT_golden_ticket_problem_l1873_187339


namespace NUMINAMATH_GPT_loss_of_450_is_negative_450_l1873_187367

-- Define the concept of profit and loss based on given conditions.
def profit (x : Int) := x
def loss (x : Int) := -x

-- The mathematical statement:
theorem loss_of_450_is_negative_450 :
  (profit 1000 = 1000) → (loss 450 = -450) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_loss_of_450_is_negative_450_l1873_187367


namespace NUMINAMATH_GPT_ab_abs_value_l1873_187355

theorem ab_abs_value {a b : ℤ} (ha : a ≠ 0) (hb : b ≠ 0)
  (hroots : ∃ r s : ℤ, (x - r)^2 * (x - s) = x^3 + a * x^2 + b * x + 9 * a) :
  |a * b| = 1344 := 
sorry

end NUMINAMATH_GPT_ab_abs_value_l1873_187355


namespace NUMINAMATH_GPT_total_wheels_l1873_187362

theorem total_wheels (bicycles tricycles : ℕ) (wheels_per_bicycle wheels_per_tricycle : ℕ) 
  (h1 : bicycles = 50) (h2 : tricycles = 20) (h3 : wheels_per_bicycle = 2) (h4 : wheels_per_tricycle = 3) : 
  (bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle = 160) :=
by
  sorry

end NUMINAMATH_GPT_total_wheels_l1873_187362
