import Mathlib

namespace NUMINAMATH_GPT_quadrilateral_angle_cosine_proof_l1795_179531

variable (AB BC CD AD : ℝ)
variable (ϕ B C : ℝ)

theorem quadrilateral_angle_cosine_proof :
  AD^2 = AB^2 + BC^2 + CD^2 - 2 * (AB * BC * Real.cos B + BC * CD * Real.cos C + CD * AB * Real.cos ϕ) :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_angle_cosine_proof_l1795_179531


namespace NUMINAMATH_GPT_find_replaced_weight_l1795_179586

-- Define the conditions and the hypothesis
def replaced_weight (W : ℝ) : Prop :=
  let avg_increase := 2.5
  let num_persons := 8
  let new_weight := 85
  (new_weight - W) = num_persons * avg_increase

-- Define the statement we aim to prove
theorem find_replaced_weight : replaced_weight 65 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_replaced_weight_l1795_179586


namespace NUMINAMATH_GPT_not_support_either_l1795_179529

theorem not_support_either (total_attendance supporters_first supporters_second : ℕ) 
  (h1 : total_attendance = 50) 
  (h2 : supporters_first = 50 * 40 / 100) 
  (h3 : supporters_second = 50 * 34 / 100) : 
  total_attendance - (supporters_first + supporters_second) = 13 :=
by
  sorry

end NUMINAMATH_GPT_not_support_either_l1795_179529


namespace NUMINAMATH_GPT_discount_rate_on_pony_jeans_l1795_179554

theorem discount_rate_on_pony_jeans 
  (F P : ℝ) 
  (H1 : F + P = 22) 
  (H2 : 45 * F + 36 * P = 882) : 
  P = 12 :=
by
  sorry

end NUMINAMATH_GPT_discount_rate_on_pony_jeans_l1795_179554


namespace NUMINAMATH_GPT_angle_A_is_correct_l1795_179582

-- Define the given conditions and the main theorem.
theorem angle_A_is_correct (A : ℝ) (m n : ℝ × ℝ) 
  (h_m : m = (Real.sin (A / 2), Real.cos (A / 2)))
  (h_n : n = (Real.cos (A / 2), -Real.cos (A / 2)))
  (h_eq : 2 * ((Prod.fst m * Prod.fst n) + (Prod.snd m * Prod.snd n)) + (Real.sqrt ((Prod.fst m)^2 + (Prod.snd m)^2)) = Real.sqrt 2 / 2) 
  : A = 5 * Real.pi / 12 := by
  sorry

end NUMINAMATH_GPT_angle_A_is_correct_l1795_179582


namespace NUMINAMATH_GPT_find_radius_l1795_179532

-- Definitions based on conditions
def circle_radius (r : ℝ) : Prop := r = 2

-- Specification based on the question and conditions
theorem find_radius (r : ℝ) : circle_radius r :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_find_radius_l1795_179532


namespace NUMINAMATH_GPT_abc_minus_def_l1795_179599

def f (x y z : ℕ) : ℕ := 5^x * 2^y * 3^z

theorem abc_minus_def {a b c d e f : ℕ} (ha : a = d) (hb : b = e) (hc : c = f + 1) : 
  (100 * a + 10 * b + c) - (100 * d + 10 * e + f) = 1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_abc_minus_def_l1795_179599


namespace NUMINAMATH_GPT_rectangle_area_ratio_l1795_179521

theorem rectangle_area_ratio (x d : ℝ) (h_ratio : 5 * x / (2 * x) = 5 / 2) (h_diag : d = 13) :
  ∃ k : ℝ, 10 * x^2 = k * d^2 ∧ k = 10 / 29 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_ratio_l1795_179521


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1795_179513

variables (V_b V_c V_w : ℝ)

-- Conditions from the problem
def speed_upstream (V_b V_c V_w : ℝ) : ℝ := V_b - V_c - V_w
def water_current_range (V_c : ℝ) : Prop := 2 ≤ V_c ∧ V_c ≤ 4
def wind_resistance_range (V_w : ℝ) : Prop := -1 ≤ V_w ∧ V_w ≤ 1
def upstream_speed : Prop := speed_upstream V_b 4 (2 - (-1)) + (2 - -1) = 4

-- Statement of the proof problem
theorem boat_speed_in_still_water :
  (∀ V_c V_w, water_current_range V_c → wind_resistance_range V_w → speed_upstream V_b V_c V_w = 4) → V_b = 7 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1795_179513


namespace NUMINAMATH_GPT_inequality_solution_l1795_179535

-- Define the condition for the denominator being positive
def denom_positive (x : ℝ) : Prop :=
  x^2 + 2*x + 7 > 0

-- Statement of the problem
theorem inequality_solution (x : ℝ) (h : denom_positive x) :
  (x + 6) / (x^2 + 2*x + 7) ≥ 0 ↔ x ≥ -6 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l1795_179535


namespace NUMINAMATH_GPT_nadia_flower_shop_l1795_179598

theorem nadia_flower_shop :
  let roses := 20
  let lilies := (3 / 4) * roses
  let cost_per_rose := 5
  let cost_per_lily := 2 * cost_per_rose
  let total_cost := roses * cost_per_rose + lilies * cost_per_lily
  total_cost = 250 := by
    sorry

end NUMINAMATH_GPT_nadia_flower_shop_l1795_179598


namespace NUMINAMATH_GPT_number_of_members_in_league_l1795_179581

-- Define the conditions
def pair_of_socks_cost := 4
def t_shirt_cost := pair_of_socks_cost + 6
def cap_cost := t_shirt_cost - 3
def total_cost_per_member := 2 * (pair_of_socks_cost + t_shirt_cost + cap_cost)
def league_total_expenditure := 3144

-- Prove that the number of members in the league is 75
theorem number_of_members_in_league : 
  (∃ (n : ℕ), total_cost_per_member * n = league_total_expenditure) → 
  (∃ (n : ℕ), n = 75) :=
by
  sorry

end NUMINAMATH_GPT_number_of_members_in_league_l1795_179581


namespace NUMINAMATH_GPT_slope_of_tangent_at_1_l1795_179537

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

theorem slope_of_tangent_at_1 : (deriv f 1) = 1 / 2 :=
  by
  sorry

end NUMINAMATH_GPT_slope_of_tangent_at_1_l1795_179537


namespace NUMINAMATH_GPT_find_S15_l1795_179510

-- Define the arithmetic progression series
variable {S : ℕ → ℕ}

-- Given conditions
axiom S5 : S 5 = 3
axiom S10 : S 10 = 12

-- We need to prove the final statement
theorem find_S15 : S 15 = 39 := 
by
  sorry

end NUMINAMATH_GPT_find_S15_l1795_179510


namespace NUMINAMATH_GPT_never_consecutive_again_l1795_179563

theorem never_consecutive_again (n : ℕ) (seq : ℕ → ℕ) :
  (∀ k, seq k = seq 0 + k) → 
  ∀ seq' : ℕ → ℕ,
    (∀ i j, i < j → seq' (2*i) = seq i + seq (j) ∧ seq' (2*i+1) = seq i - seq (j)) →
    ¬ (∀ k, seq' k = seq' 0 + k) :=
by
  sorry

end NUMINAMATH_GPT_never_consecutive_again_l1795_179563


namespace NUMINAMATH_GPT_solve_for_y_l1795_179577

theorem solve_for_y (y : ℝ) (h : 5 * (y ^ (1/3)) - 3 * (y / (y ^ (2/3))) = 10 + (y ^ (1/3))) :
  y = 1000 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_y_l1795_179577


namespace NUMINAMATH_GPT_problem_l1795_179590

theorem problem (a : ℕ → ℝ) (h0 : a 1 = 0) (h9 : a 9 = 0)
  (h2_8 : ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a i > 0) (h_nonneg : ∀ n, 1 ≤ n ∧ n ≤ 9 → a n ≥ 0) : 
  (∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i-1) + a (i+1) < 2 * a i) ∧ (∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i-1) + a (i+1) < 1.9 * a i) := 
sorry

end NUMINAMATH_GPT_problem_l1795_179590


namespace NUMINAMATH_GPT_white_tiles_count_l1795_179500

-- Definitions from conditions
def total_tiles : ℕ := 20
def yellow_tiles : ℕ := 3
def blue_tiles : ℕ := yellow_tiles + 1
def purple_tiles : ℕ := 6

-- We need to prove that number of white tiles is 7
theorem white_tiles_count : total_tiles - (yellow_tiles + blue_tiles + purple_tiles) = 7 := by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_white_tiles_count_l1795_179500


namespace NUMINAMATH_GPT_minimum_value_of_expression_l1795_179550

theorem minimum_value_of_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
    (x^4 / (y - 1)) + (y^4 / (x - 1)) ≥ 12 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l1795_179550


namespace NUMINAMATH_GPT_abby_bridget_adjacent_probability_l1795_179595

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def probability_adjacent : ℚ :=
  let total_seats := 9
  let ab_adj_same_row_pairs := 9
  let ab_adj_diagonal_pairs := 4
  let favorable_outcomes := (ab_adj_same_row_pairs + ab_adj_diagonal_pairs) * 2 * factorial 7
  let total_outcomes := factorial total_seats
  favorable_outcomes / total_outcomes

theorem abby_bridget_adjacent_probability :
  probability_adjacent = 13 / 36 :=
by
  sorry

end NUMINAMATH_GPT_abby_bridget_adjacent_probability_l1795_179595


namespace NUMINAMATH_GPT_john_must_solve_at_least_17_correct_l1795_179517

theorem john_must_solve_at_least_17_correct :
  ∀ (x : ℕ), 25 = 20 + 5 → 7 * x - (20 - x) + 2 * 5 ≥ 120 → x ≥ 17 :=
by
  intros x h1 h2
  -- Remaining steps will be included in the proof
  sorry

end NUMINAMATH_GPT_john_must_solve_at_least_17_correct_l1795_179517


namespace NUMINAMATH_GPT_triangle_area_l1795_179504

namespace MathProof

theorem triangle_area (y_eq_6 y_eq_2_plus_x y_eq_2_minus_x : ℝ → ℝ)
  (h1 : ∀ x, y_eq_6 x = 6)
  (h2 : ∀ x, y_eq_2_plus_x x = 2 + x)
  (h3 : ∀ x, y_eq_2_minus_x x = 2 - x) :
  let a := (4, 6)
  let b := (-4, 6)
  let c := (0, 2)
  let base := dist a b
  let height := (6 - 2:ℝ)
  (1 / 2 * base * height = 16) := by
    sorry

end MathProof

end NUMINAMATH_GPT_triangle_area_l1795_179504


namespace NUMINAMATH_GPT_maria_total_cost_l1795_179544

-- Define the conditions as variables in the Lean environment
def daily_rental_rate : ℝ := 35
def mileage_rate : ℝ := 0.25
def rental_days : ℕ := 3
def miles_driven : ℕ := 500

-- Now, state the theorem that Maria’s total payment should be $230
theorem maria_total_cost : (daily_rental_rate * rental_days) + (mileage_rate * miles_driven) = 230 := 
by
  -- no proof required, just state as sorry
  sorry

end NUMINAMATH_GPT_maria_total_cost_l1795_179544


namespace NUMINAMATH_GPT_problem1_problem2_l1795_179507

def f (x : ℝ) := |x - 1| + |x + 2|

def T (a : ℝ) := -Real.sqrt 3 < a ∧ a < Real.sqrt 3

theorem problem1 (a : ℝ) : (∀ x : ℝ, f x > a^2) ↔ T a :=
by
  sorry

theorem problem2 (m n : ℝ) (h1 : T m) (h2 : T n) : Real.sqrt 3 * |m + n| < |m * n + 3| :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1795_179507


namespace NUMINAMATH_GPT_visual_range_increase_percent_l1795_179587

theorem visual_range_increase_percent :
  let original_visual_range := 100
  let new_visual_range := 150
  ((new_visual_range - original_visual_range) / original_visual_range) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_visual_range_increase_percent_l1795_179587


namespace NUMINAMATH_GPT_sqrt_interval_l1795_179561

theorem sqrt_interval :
  let expr := (Real.sqrt 18) / 3 - (Real.sqrt 2) * (Real.sqrt (1 / 2))
  0 < expr ∧ expr < 1 :=
by
  let expr := (Real.sqrt 18) / 3 - (Real.sqrt 2) * (Real.sqrt (1 / 2))
  sorry

end NUMINAMATH_GPT_sqrt_interval_l1795_179561


namespace NUMINAMATH_GPT_value_of_a_l1795_179524

open Set

theorem value_of_a (a : ℝ) (h : {1, 2} ∪ {x | x^2 - a * x + a - 1 = 0} = {1, 2}) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1795_179524


namespace NUMINAMATH_GPT_knight_probability_sum_l1795_179568

def num_knights := 30
def chosen_knights := 4

-- Calculate valid placements where no knights are adjacent
def valid_placements : ℕ := 26 * 24 * 22 * 20
-- Calculate total unrestricted placements
def total_placements : ℕ := 26 * 27 * 28 * 29
-- Calculate probability
def P : ℚ := 1 - (valid_placements : ℚ) / total_placements

-- Simplify the fraction P to its lowest terms: 553/1079
def simplified_num := 553
def simplified_denom := 1079

-- Sum of the numerator and denominator of simplified P
def sum_numer_denom := simplified_num + simplified_denom

theorem knight_probability_sum :
  sum_numer_denom = 1632 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_knight_probability_sum_l1795_179568


namespace NUMINAMATH_GPT_missing_number_is_twelve_l1795_179589

theorem missing_number_is_twelve
  (x : ℤ)
  (h : 10010 - x * 3 * 2 = 9938) :
  x = 12 :=
sorry

end NUMINAMATH_GPT_missing_number_is_twelve_l1795_179589


namespace NUMINAMATH_GPT_abs_diff_squares_l1795_179594

theorem abs_diff_squares (a b : ℤ) (h_a : a = 105) (h_b : b = 95):
  |a^2 - b^2| = 2000 := by
  sorry

end NUMINAMATH_GPT_abs_diff_squares_l1795_179594


namespace NUMINAMATH_GPT_complement_of_M_l1795_179565

open Set

-- Define the universal set
def U : Set ℝ := univ

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

-- The theorem stating the complement of M in U
theorem complement_of_M : (U \ M) = {y | y < -1} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_l1795_179565


namespace NUMINAMATH_GPT_solve_for_x_l1795_179562

theorem solve_for_x :
  (48 = 5 * x + 3) → x = 9 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1795_179562


namespace NUMINAMATH_GPT_bridge_extension_length_l1795_179596

theorem bridge_extension_length (river_width bridge_length : ℕ) (h_river : river_width = 487) (h_bridge : bridge_length = 295) : river_width - bridge_length = 192 :=
by
  sorry

end NUMINAMATH_GPT_bridge_extension_length_l1795_179596


namespace NUMINAMATH_GPT_find_probability_l1795_179579

noncomputable def probability_distribution (X : ℕ → ℝ) := ∀ k, X k = 1 / (2^k)

theorem find_probability (X : ℕ → ℝ) (h : probability_distribution X) :
  X 3 + X 4 = 3 / 16 :=
by
  sorry

end NUMINAMATH_GPT_find_probability_l1795_179579


namespace NUMINAMATH_GPT_grain_milling_l1795_179518

theorem grain_milling (W : ℝ) (h : 0.9 * W = 100) : W = 111.1 :=
sorry

end NUMINAMATH_GPT_grain_milling_l1795_179518


namespace NUMINAMATH_GPT_classrooms_student_rabbit_difference_l1795_179580

-- Definitions from conditions
def students_per_classroom : Nat := 20
def rabbits_per_classroom : Nat := 3
def number_of_classrooms : Nat := 6

-- Theorem statement
theorem classrooms_student_rabbit_difference :
  (students_per_classroom * number_of_classrooms) - (rabbits_per_classroom * number_of_classrooms) = 102 := by
  sorry

end NUMINAMATH_GPT_classrooms_student_rabbit_difference_l1795_179580


namespace NUMINAMATH_GPT_find_number_of_each_coin_l1795_179558

-- Define the number of coins
variables (n d q : ℕ)

-- Given conditions
axiom twice_as_many_nickels_as_quarters : n = 2 * q
axiom same_number_of_dimes_as_quarters : d = q
axiom total_value_of_coins : 5 * n + 10 * d + 25 * q = 1520

-- Statement to prove
theorem find_number_of_each_coin :
  q = 304 / 9 ∧
  n = 2 * (304 / 9) ∧
  d = 304 / 9 :=
sorry

end NUMINAMATH_GPT_find_number_of_each_coin_l1795_179558


namespace NUMINAMATH_GPT_workers_contribution_l1795_179560

theorem workers_contribution (N C : ℕ) 
(h1 : N * C = 300000) 
(h2 : N * (C + 50) = 360000) : 
N = 1200 :=
sorry

end NUMINAMATH_GPT_workers_contribution_l1795_179560


namespace NUMINAMATH_GPT_total_balls_estimation_l1795_179576

theorem total_balls_estimation 
  (num_red_balls : ℕ)
  (total_trials : ℕ)
  (red_ball_draws : ℕ)
  (red_ball_ratio : ℚ)
  (total_balls_estimate : ℕ)
  (h1 : num_red_balls = 5)
  (h2 : total_trials = 80)
  (h3 : red_ball_draws = 20)
  (h4 : red_ball_ratio = 1 / 4)
  (h5 : red_ball_ratio = red_ball_draws / total_trials)
  (h6 : red_ball_ratio = num_red_balls / total_balls_estimate)
  : total_balls_estimate = 20 := 
sorry

end NUMINAMATH_GPT_total_balls_estimation_l1795_179576


namespace NUMINAMATH_GPT_initial_oranges_per_tree_l1795_179511

theorem initial_oranges_per_tree (x : ℕ) (h1 : 8 * (5 * x - 2 * x) / 5 = 960) : x = 200 :=
sorry

end NUMINAMATH_GPT_initial_oranges_per_tree_l1795_179511


namespace NUMINAMATH_GPT_range_of_a_l1795_179591

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + (a - 1) * x + 1
noncomputable def f' (a x : ℝ) : ℝ := x^2 - a * x + a - 1

theorem range_of_a (a : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → f' a x ≤ 0) ∧ (∀ x, 6 < x → f' a x ≥ 0) ↔ 5 ≤ a ∧ a ≤ 7 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1795_179591


namespace NUMINAMATH_GPT_marie_ends_with_755_l1795_179538

def erasers_end (initial lost packs erasers_per_pack : ℕ) : ℕ :=
  initial - lost + packs * erasers_per_pack

theorem marie_ends_with_755 :
  erasers_end 950 420 3 75 = 755 :=
by
  sorry

end NUMINAMATH_GPT_marie_ends_with_755_l1795_179538


namespace NUMINAMATH_GPT_min_sum_of_squares_of_y_coords_l1795_179564

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line_through_point (m : ℝ) (x y : ℝ) : Prop := x = m * y + 4

theorem min_sum_of_squares_of_y_coords :
  ∃ (m : ℝ), ∀ (x1 y1 x2 y2 : ℝ),
  (line_through_point m x1 y1) →
  (parabola x1 y1) →
  (line_through_point m x2 y2) →
  (parabola x2 y2) →
  x1 ≠ x2 → 
  ((y1 + y2)^2 - 2 * y1 * y2) = 32 :=
sorry

end NUMINAMATH_GPT_min_sum_of_squares_of_y_coords_l1795_179564


namespace NUMINAMATH_GPT_find_number_l1795_179506

theorem find_number : ∃ x, x - 0.16 * x = 126 ↔ x = 150 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l1795_179506


namespace NUMINAMATH_GPT_div_by_7_iff_sum_div_by_7_l1795_179553

theorem div_by_7_iff_sum_div_by_7 (a b : ℕ) : 
  (101 * a + 10 * b) % 7 = 0 ↔ (a + b) % 7 = 0 := 
by
  sorry

end NUMINAMATH_GPT_div_by_7_iff_sum_div_by_7_l1795_179553


namespace NUMINAMATH_GPT_odd_solution_exists_l1795_179583

theorem odd_solution_exists (k m n : ℕ) (h : m * n = k^2 + k + 3) : 
∃ (x y : ℤ), (x^2 + 11 * y^2 = 4 * m ∨ x^2 + 11 * y^2 = 4 * n) ∧ (x % 2 ≠ 0 ∧ y % 2 ≠ 0) :=
sorry

end NUMINAMATH_GPT_odd_solution_exists_l1795_179583


namespace NUMINAMATH_GPT_time_away_is_43point64_minutes_l1795_179536

theorem time_away_is_43point64_minutes :
  ∃ (n1 n2 : ℝ), 
    (195 + n1 / 2 - 6 * n1 = 120 ∨ 195 + n1 / 2 - 6 * n1 = -120) ∧
    (195 + n2 / 2 - 6 * n2 = 120 ∨ 195 + n2 / 2 - 6 * n2 = -120) ∧
    n1 ≠ n2 ∧
    n1 < 60 ∧
    n2 < 60 ∧
    |n2 - n1| = 43.64 :=
sorry

end NUMINAMATH_GPT_time_away_is_43point64_minutes_l1795_179536


namespace NUMINAMATH_GPT_volume_of_region_l1795_179525

theorem volume_of_region :
  ∃ (V : ℝ), V = 9 ∧
  ∀ (x y z : ℝ), |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z| ≤ 6 :=
sorry

end NUMINAMATH_GPT_volume_of_region_l1795_179525


namespace NUMINAMATH_GPT_solve_for_c_l1795_179585

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (c * x) / (2 * x + 3)

theorem solve_for_c {c : ℝ} (hc : ∀ x ≠ (-3/2), f c (f c x) = x) : c = -3 :=
by
  intros
  -- The proof steps will go here
  sorry

end NUMINAMATH_GPT_solve_for_c_l1795_179585


namespace NUMINAMATH_GPT_probability_red_ball_10th_draw_l1795_179520

-- Definitions for conditions in the problem
def total_balls : ℕ := 10
def red_balls : ℕ := 2

-- Probability calculation function
def probability_of_red_ball (total : ℕ) (red : ℕ) : ℚ :=
  red / total

-- Theorem statement: Given the conditions, the probability of drawing a red ball on the 10th attempt is 1/5
theorem probability_red_ball_10th_draw :
  probability_of_red_ball total_balls red_balls = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_red_ball_10th_draw_l1795_179520


namespace NUMINAMATH_GPT_proof_statement_l1795_179548

noncomputable def problem_statement (a b : ℤ) : ℤ :=
  (a^3 + b^3) / (a^2 - a * b + b^2)

theorem proof_statement : problem_statement 5 4 = 9 := by
  sorry

end NUMINAMATH_GPT_proof_statement_l1795_179548


namespace NUMINAMATH_GPT_correct_card_assignment_l1795_179552

theorem correct_card_assignment :
  ∃ (cards : Fin 4 → Fin 4), 
    (¬ (cards 1 = 3 ∨ cards 2 = 3) ∧
     ¬ (cards 0 = 2 ∨ cards 2 = 2) ∧
     ¬ (cards 0 = 1) ∧
     ¬ (cards 0 = 3)) →
    (cards 0 = 4 ∧ cards 1 = 2 ∧ cards 2 = 1 ∧ cards 3 = 3) := 
by {
  sorry
}

end NUMINAMATH_GPT_correct_card_assignment_l1795_179552


namespace NUMINAMATH_GPT_greatest_integer_difference_l1795_179549

theorem greatest_integer_difference (x y : ℤ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 8) : 
  ∃ d, d = y - x ∧ d = 2 := 
by
  sorry

end NUMINAMATH_GPT_greatest_integer_difference_l1795_179549


namespace NUMINAMATH_GPT_mary_initial_blue_crayons_l1795_179542

/-- **Mathematically equivalent proof problem**:
  Given that Mary has 5 green crayons and gives away 3 green crayons and 1 blue crayon,
  and she has 9 crayons left, prove that she initially had 8 blue crayons. 
  -/
theorem mary_initial_blue_crayons (initial_green_crayons : ℕ) (green_given_away : ℕ) (blue_given_away : ℕ)
  (crayons_left : ℕ) (initial_crayons : ℕ) :
  initial_green_crayons = 5 →
  green_given_away = 3 →
  blue_given_away = 1 →
  crayons_left = 9 →
  initial_crayons = crayons_left + (green_given_away + blue_given_away) →
  initial_crayons - initial_green_crayons = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_mary_initial_blue_crayons_l1795_179542


namespace NUMINAMATH_GPT_tangent_line_l1795_179551

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 1 / x

theorem tangent_line (x y : ℝ) (h_inter : y = f x ∧ y = g x) :
  (x - 2 * y + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_l1795_179551


namespace NUMINAMATH_GPT_find_distance_MF_l1795_179540

-- Define the parabola and point conditions
def parabola (x y : ℝ) := y^2 = 8 * x

-- Define the focus of the parabola
def F : ℝ × ℝ := (2, 0)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define the distance squared between two points
def dist_squared (A B : ℝ × ℝ) : ℝ :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2

-- Prove the required statement
theorem find_distance_MF (x y : ℝ) (hM : parabola x y) (h_dist: dist_squared (x, y) O = 3 * (x + 2)) :
  dist_squared (x, y) F = 9 := by
  sorry

end NUMINAMATH_GPT_find_distance_MF_l1795_179540


namespace NUMINAMATH_GPT_train_seat_count_l1795_179578

theorem train_seat_count (t : ℝ)
  (h1 : ∃ (t : ℝ), t = 36 + 0.2 * t + 0.5 * t) :
  t = 120 :=
by
  sorry

end NUMINAMATH_GPT_train_seat_count_l1795_179578


namespace NUMINAMATH_GPT_when_was_p_turned_off_l1795_179528

noncomputable def pipe_p_rate := (1/12 : ℚ)  -- Pipe p rate
noncomputable def pipe_q_rate := (1/15 : ℚ)  -- Pipe q rate
noncomputable def combined_rate := (3/20 : ℚ) -- Combined rate of p and q when both are open
noncomputable def time_after_p_off := (1.5 : ℚ)  -- Time for q to fill alone after p is off
noncomputable def fill_cistern (t : ℚ) := combined_rate * t + pipe_q_rate * time_after_p_off

theorem when_was_p_turned_off (t : ℚ) : fill_cistern t = 1 ↔ t = 6 := sorry

end NUMINAMATH_GPT_when_was_p_turned_off_l1795_179528


namespace NUMINAMATH_GPT_perfect_square_trinomial_l1795_179527

theorem perfect_square_trinomial (k : ℝ) :
  ∃ k, (∀ x, (4 * x^2 - 2 * k * x + 1) = (2 * x + 1)^2 ∨ (4 * x^2 - 2 * k * x + 1) = (2 * x - 1)^2) → 
  (k = 2 ∨ k = -2) := by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l1795_179527


namespace NUMINAMATH_GPT_number_of_solutions_l1795_179509

theorem number_of_solutions :
  ∃ (sols : Finset (ℝ × ℝ × ℝ × ℝ)), 
  (∀ (x y z w : ℝ), ((x, y, z, w) ∈ sols) ↔ (x = z + w + z * w * x ∧ y = w + x + w * x * y ∧ z = x + y + x * y * z ∧ w = y + z + y * z * w ∧ x * y + y * z + z * w + w * x = 2)) ∧ 
  sols.card = 5 :=
sorry

end NUMINAMATH_GPT_number_of_solutions_l1795_179509


namespace NUMINAMATH_GPT_fleas_cannot_reach_final_positions_l1795_179588

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def initial_A : Point2D := ⟨0, 0⟩
def initial_B : Point2D := ⟨1, 0⟩
def initial_C : Point2D := ⟨0, 1⟩

def area (A B C : Point2D) : ℝ :=
  0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def final_A : Point2D := ⟨1, 0⟩
def final_B : Point2D := ⟨-1, 0⟩
def final_C : Point2D := ⟨0, 1⟩

theorem fleas_cannot_reach_final_positions : 
    ¬ (∃ (flea_move_sequence : List (Point2D → Point2D)), 
    area initial_A initial_B initial_C = area final_A final_B final_C) :=
by 
  sorry

end NUMINAMATH_GPT_fleas_cannot_reach_final_positions_l1795_179588


namespace NUMINAMATH_GPT_volume_of_tetrahedron_OABC_l1795_179539

-- Definitions of side lengths and their squared values
def side_length_A_B := 7
def side_length_B_C := 8
def side_length_C_A := 9

-- Squared values of coordinates
def a_sq := 33
def b_sq := 16
def c_sq := 48

-- Main statement to prove the volume
theorem volume_of_tetrahedron_OABC :
  (1/6) * (Real.sqrt a_sq) * (Real.sqrt b_sq) * (Real.sqrt c_sq) = 2 * Real.sqrt 176 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_volume_of_tetrahedron_OABC_l1795_179539


namespace NUMINAMATH_GPT_inequality_correct_l1795_179567

variable {a b c : ℝ}

theorem inequality_correct (h : a * b < 0) : |a - c| ≤ |a - b| + |b - c| :=
sorry

end NUMINAMATH_GPT_inequality_correct_l1795_179567


namespace NUMINAMATH_GPT_log_x_y_eq_sqrt_3_l1795_179584

variable (x y z : ℝ)
variable (hx : 1 < x) (hy : 1 < y) (hz : 1 < z)
variable (h1 : x ^ (Real.log z / Real.log y) = 2)
variable (h2 : y ^ (Real.log x / Real.log y) = 4)
variable (h3 : z ^ (Real.log y / Real.log x) = 8)

theorem log_x_y_eq_sqrt_3 : Real.log y / Real.log x = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_log_x_y_eq_sqrt_3_l1795_179584


namespace NUMINAMATH_GPT_proof_problem_l1795_179512

theorem proof_problem 
  (a1 a2 b2 : ℚ)
  (ha1 : a1 = -9 + (8/3))
  (ha2 : a2 = -9 + 2 * (8/3))
  (hb2 : b2 = -3) :
  b2 * (a1 + a2) = 30 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1795_179512


namespace NUMINAMATH_GPT_Dylan_needs_two_trays_l1795_179546

noncomputable def ice_cubes_glass : ℕ := 8
noncomputable def ice_cubes_pitcher : ℕ := 2 * ice_cubes_glass
noncomputable def tray_capacity : ℕ := 12
noncomputable def total_ice_cubes_used : ℕ := ice_cubes_glass + ice_cubes_pitcher
noncomputable def number_of_trays : ℕ := total_ice_cubes_used / tray_capacity

theorem Dylan_needs_two_trays : number_of_trays = 2 := by
  sorry

end NUMINAMATH_GPT_Dylan_needs_two_trays_l1795_179546


namespace NUMINAMATH_GPT_problem_solution_l1795_179555

noncomputable def inequality_holds (a b : ℝ) (n : ℕ) : Prop :=
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n - 1)

theorem problem_solution (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (1 / a + 1 / b = 1)) (h4 : 0 < n):
  inequality_holds a b n :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1795_179555


namespace NUMINAMATH_GPT_complement_union_eq_complement_l1795_179516

open Set

section ComplementUnion

variable (k : ℤ)

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def UniversalSet : Set ℤ := univ
def ComplementUnion : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}

theorem complement_union_eq_complement :
  UniversalSet \ (SetA ∪ SetB) = ComplementUnion :=
by
  sorry

end ComplementUnion

end NUMINAMATH_GPT_complement_union_eq_complement_l1795_179516


namespace NUMINAMATH_GPT_password_correct_l1795_179523

-- conditions
def poly1 (x y : ℤ) : ℤ := x ^ 4 - y ^ 4
def factor1 (x y : ℤ) : ℤ := (x - y) * (x + y) * (x ^ 2 + y ^ 2)

def poly2 (x y : ℤ) : ℤ := x ^ 3 - x * y ^ 2
def factor2 (x y : ℤ) : ℤ := x * (x - y) * (x + y)

-- given values
def x := 18
def y := 5

-- goal
theorem password_correct : factor2 x y = 18 * 13 * 23 :=
by
  -- We setup the goal with the equivalent sequence of the password generation
  sorry

end NUMINAMATH_GPT_password_correct_l1795_179523


namespace NUMINAMATH_GPT_area_of_inscribed_square_l1795_179543

theorem area_of_inscribed_square (D : ℝ) (h : D = 10) : 
  ∃ A : ℝ, A = 50 :=
by
  sorry

end NUMINAMATH_GPT_area_of_inscribed_square_l1795_179543


namespace NUMINAMATH_GPT_lengths_AC_CB_ratio_GJ_JH_coords_F_on_DE_values_p_q_KL_l1795_179573

-- Problem 1 - Lengths of AC and CB are 15 and 5 respectively.
theorem lengths_AC_CB (x1 y1 x2 y2 x3 y3 : ℝ) :
  (x1, y1) = (1,2) ∧ (x2, y2) = (17,14) ∧ (x3, y3) = (13,11) →
  ∃ (AC CB : ℝ), AC = 15 ∧ CB = 5 :=
by
  sorry

-- Problem 2 - Ratio of GJ and JH is 3:2.
theorem ratio_GJ_JH (x1 y1 x2 y2 x3 y3 : ℝ) :
  (x1, y1) = (11,2) ∧ (x2, y2) = (1,7) ∧ (x3, y3) = (5,5) →
  ∃ (GJ JH : ℝ), GJ / JH = 3 / 2 :=
by
  sorry

-- Problem 3 - Coordinates of point F on DE with ratio 1:2 is (3,7).
theorem coords_F_on_DE (x1 y1 x2 y2 : ℝ) :
  (x1, y1) = (1,6) ∧ (x2, y2) = (7,9) →
  ∃ (x y : ℝ), (x, y) = (3,7) :=
by
  sorry

-- Problem 4 - Values of p and q for point M on KL with ratio 3:4 are p = 15 and q = 2.
theorem values_p_q_KL (x1 y1 x2 y2 x3 y3 : ℝ) :
  (x1, y1) = (1, q) ∧ (x2, y2) = (p, 9) ∧ (x3, y3) = (7,5) →
  ∃ (p q : ℝ), p = 15 ∧ q = 2 :=
by
  sorry

end NUMINAMATH_GPT_lengths_AC_CB_ratio_GJ_JH_coords_F_on_DE_values_p_q_KL_l1795_179573


namespace NUMINAMATH_GPT_price_of_first_variety_l1795_179515

theorem price_of_first_variety
  (P : ℝ)
  (H1 : 1 * P + 1 * 135 + 2 * 175.5 = 4 * 153) :
  P = 126 :=
by
  sorry

end NUMINAMATH_GPT_price_of_first_variety_l1795_179515


namespace NUMINAMATH_GPT_line_eq_l1795_179593

variables {x x1 x2 y y1 y2 : ℝ}

theorem line_eq (h : x2 ≠ x1 ∧ y2 ≠ y1) : 
  (x - x1) / (x2 - x1) = (y - y1) / (y2 - y1) :=
sorry

end NUMINAMATH_GPT_line_eq_l1795_179593


namespace NUMINAMATH_GPT_marthas_bedroom_size_l1795_179505

-- Define the variables and conditions
def total_square_footage := 300
def additional_square_footage := 60
def Martha := 120
def Jenny := Martha + additional_square_footage

-- The main theorem stating the requirement 
theorem marthas_bedroom_size : (Martha + (Martha + additional_square_footage) = total_square_footage) -> Martha = 120 :=
by 
  sorry

end NUMINAMATH_GPT_marthas_bedroom_size_l1795_179505


namespace NUMINAMATH_GPT_bills_are_fake_bart_can_give_exact_amount_l1795_179545

-- Problem (a)
theorem bills_are_fake : 
  (∀ x, x = 17 ∨ x = 19 → false) :=
sorry

-- Problem (b)
theorem bart_can_give_exact_amount (n : ℕ) :
  (∀ m, m = 323  → (n ≥ m → ∃ a b : ℕ, n = 17 * a + 19 * b)) :=
sorry

end NUMINAMATH_GPT_bills_are_fake_bart_can_give_exact_amount_l1795_179545


namespace NUMINAMATH_GPT_ali_initial_money_l1795_179534

theorem ali_initial_money (X : ℝ) (h1 : X / 2 - (1 / 3) * (X / 2) = 160) : X = 480 :=
by sorry

end NUMINAMATH_GPT_ali_initial_money_l1795_179534


namespace NUMINAMATH_GPT_least_beans_l1795_179533

-- Define the conditions 
variables (r b : ℕ)

-- State the theorem 
theorem least_beans (h1 : r ≥ 2 * b + 8) (h2 : r ≤ 3 * b) : b ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_least_beans_l1795_179533


namespace NUMINAMATH_GPT_chris_packed_percentage_l1795_179503

theorem chris_packed_percentage (K C : ℕ) (h : K / (C : ℝ) = 2 / 3) :
  (C / (K + C : ℝ)) * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_chris_packed_percentage_l1795_179503


namespace NUMINAMATH_GPT_find_c_d_l1795_179522

theorem find_c_d (y c d : ℕ) (H1 : y = c + Real.sqrt d) (H2 : y^2 + 4 * y + 4 / y + 1 / (y^2) = 30) :
  c + d = 5 :=
sorry

end NUMINAMATH_GPT_find_c_d_l1795_179522


namespace NUMINAMATH_GPT_fraction_simplification_l1795_179501

-- Definitions based on conditions and question
def lcm_462_42 : ℕ := 462
def prime_factors_462 : List ℕ := [2, 3, 7, 11]
def prime_factors_42 : List ℕ := [2, 3, 7]

-- Main theorem statement
theorem fraction_simplification :
  (1 / 462) + (17 / 42) = 94 / 231 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1795_179501


namespace NUMINAMATH_GPT_quadrilateral_interior_angle_not_greater_90_l1795_179526

-- Definition of the quadrilateral interior angle property
def quadrilateral_interior_angles := ∀ (a b c d : ℝ), (a + b + c + d = 360) → (a > 90 → b > 90 → c > 90 → d > 90 → false)

-- Proposition: There is at least one interior angle in a quadrilateral that is not greater than 90 degrees.
theorem quadrilateral_interior_angle_not_greater_90 :
  (∀ (a b c d : ℝ), (a + b + c + d = 360) → (a > 90 ∧ b > 90 ∧ c > 90 ∧ d > 90) → false) →
  (∃ (a b c d : ℝ), a + b + c + d = 360 ∧ (a ≤ 90 ∨ b ≤ 90 ∨ c ≤ 90 ∨ d ≤ 90)) :=
sorry

end NUMINAMATH_GPT_quadrilateral_interior_angle_not_greater_90_l1795_179526


namespace NUMINAMATH_GPT_read_books_correct_l1795_179570

namespace CrazySillySchool

-- Definitions from conditions
def total_books : Nat := 20
def unread_books : Nat := 5
def read_books : Nat := total_books - unread_books

-- Theorem statement
theorem read_books_correct : read_books = 15 :=
by
  -- Mathematical statement that follows from conditions and correct answer
  sorry

end CrazySillySchool

end NUMINAMATH_GPT_read_books_correct_l1795_179570


namespace NUMINAMATH_GPT_find_angle_C_l1795_179508

-- Given conditions
variable {A B C : ℝ}
variable (h_triangle : A + B + C = π)
variable (h_tanA : Real.tan A = 1/2)
variable (h_cosB : Real.cos B = 3 * Real.sqrt 10 / 10)

-- The proof statement
theorem find_angle_C :
  C = 3 * π / 4 := by
  sorry

end NUMINAMATH_GPT_find_angle_C_l1795_179508


namespace NUMINAMATH_GPT_tank_A_is_60_percent_of_tank_B_capacity_l1795_179541

-- Conditions
def height_A : ℝ := 10
def circumference_A : ℝ := 6
def height_B : ℝ := 6
def circumference_B : ℝ := 10

-- Statement
theorem tank_A_is_60_percent_of_tank_B_capacity (V_A V_B : ℝ) (radius_A radius_B : ℝ)
  (hA : radius_A = circumference_A / (2 * Real.pi))
  (hB : radius_B = circumference_B / (2 * Real.pi))
  (vol_A : V_A = Real.pi * radius_A^2 * height_A)
  (vol_B : V_B = Real.pi * radius_B^2 * height_B) :
  (V_A / V_B) * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_tank_A_is_60_percent_of_tank_B_capacity_l1795_179541


namespace NUMINAMATH_GPT_sum_consecutive_even_integers_l1795_179569

theorem sum_consecutive_even_integers (n : ℕ) (h : 2 * n + 4 = 156) : 
  n + (n + 2) + (n + 4) = 234 := 
by
  sorry

end NUMINAMATH_GPT_sum_consecutive_even_integers_l1795_179569


namespace NUMINAMATH_GPT_martin_waste_time_l1795_179530

theorem martin_waste_time : 
  let waiting_traffic := 2
  let trying_off_freeway := 4 * waiting_traffic
  let detours := 3 * 30 / 60
  let meal := 45 / 60
  let delays := (20 + 40) / 60
  waiting_traffic + trying_off_freeway + detours + meal + delays = 13.25 := 
by
  sorry

end NUMINAMATH_GPT_martin_waste_time_l1795_179530


namespace NUMINAMATH_GPT_alpha_and_2beta_l1795_179571

theorem alpha_and_2beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
  (h_tan_alpha : Real.tan α = 1 / 8) (h_sin_beta : Real.sin β = 1 / 3) :
  α + 2 * β = Real.arctan (15 / 56) := by
  sorry

end NUMINAMATH_GPT_alpha_and_2beta_l1795_179571


namespace NUMINAMATH_GPT_jessica_remaining_time_after_penalties_l1795_179557

-- Definitions for the given conditions
def questions_answered : ℕ := 16
def total_questions : ℕ := 80
def time_used_minutes : ℕ := 12
def exam_duration_minutes : ℕ := 60
def penalty_per_incorrect_answer_minutes : ℕ := 2

-- Define the rate of answering questions
def answering_rate : ℚ := questions_answered / time_used_minutes

-- Define the total time needed to answer all questions
def total_time_needed : ℚ := total_questions / answering_rate

-- Define the remaining time after penalties
def remaining_time_after_penalties (x : ℕ) : ℤ :=
  max 0 (0 - penalty_per_incorrect_answer_minutes * x)

-- The theorem to prove
theorem jessica_remaining_time_after_penalties (x : ℕ) : 
  remaining_time_after_penalties x = max 0 (0 - penalty_per_incorrect_answer_minutes * x) := 
by
  sorry

end NUMINAMATH_GPT_jessica_remaining_time_after_penalties_l1795_179557


namespace NUMINAMATH_GPT_missing_number_unique_l1795_179547

theorem missing_number_unique (x : ℤ) 
  (h : |9 - x * (3 - 12)| - |5 - 11| = 75) : 
  x = 8 :=
sorry

end NUMINAMATH_GPT_missing_number_unique_l1795_179547


namespace NUMINAMATH_GPT_range_of_x_l1795_179597

-- Defining the conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - (f x)

def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y ≤ f x

-- Given conditions in Lean
axiom f : ℝ → ℝ
axiom h_odd : odd_function f
axiom h_decreasing_pos : ∀ x y, 0 < x ∧ x < y → f y ≤ f x
axiom h_f4 : f 4 = 0

-- To prove the range of x for which f(x-3) ≤ 0
theorem range_of_x :
    {x : ℝ | f (x - 3) ≤ 0} = {x : ℝ | -1 ≤ x ∧ x < 3} ∪ {x : ℝ | 7 ≤ x} :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1795_179597


namespace NUMINAMATH_GPT_kevin_exchanges_l1795_179556

variables (x y : ℕ)

def R (x y : ℕ) := 100 - 3 * x + 2 * y
def B (x y : ℕ) := 100 + 2 * x - 4 * y

theorem kevin_exchanges :
  (∃ x y, R x y >= 3 ∧ B x y >= 4 ∧ x + y = 132) :=
sorry

end NUMINAMATH_GPT_kevin_exchanges_l1795_179556


namespace NUMINAMATH_GPT_member_pays_48_percent_of_SRP_l1795_179572

theorem member_pays_48_percent_of_SRP
  (P : ℝ)
  (h₀ : P > 0)
  (basic_discount : ℝ := 0.40)
  (additional_discount : ℝ := 0.20) :
  ((1 - additional_discount) * (1 - basic_discount) * P) / P * 100 = 48 := by
  sorry

end NUMINAMATH_GPT_member_pays_48_percent_of_SRP_l1795_179572


namespace NUMINAMATH_GPT_sequence_is_periodic_l1795_179574

open Nat

def is_periodic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ p > 0, ∀ i, a (i + p) = a i

theorem sequence_is_periodic (a : ℕ → ℕ)
  (h1 : ∀ n, a n < 1988)
  (h2 : ∀ m n, a m + a n ∣ a (m + n)) : is_periodic_sequence a :=
by
  sorry

end NUMINAMATH_GPT_sequence_is_periodic_l1795_179574


namespace NUMINAMATH_GPT_sum_of_series_l1795_179519

noncomputable def seriesSum : ℝ := ∑' n : ℕ, (4 * (n + 1) + 1) / (3 ^ (n + 1))

theorem sum_of_series : seriesSum = 7 / 2 := by
  sorry

end NUMINAMATH_GPT_sum_of_series_l1795_179519


namespace NUMINAMATH_GPT_average_age_condition_l1795_179566

theorem average_age_condition (n : ℕ) 
  (h1 : (↑n * 14) / n = 14) 
  (h2 : ((↑n * 14) + 34) / (n + 1) = 16) : 
  n = 9 := 
by 
-- Proof goes here
sorry

end NUMINAMATH_GPT_average_age_condition_l1795_179566


namespace NUMINAMATH_GPT_determine_sold_cakes_l1795_179502

def initial_cakes := 121
def new_cakes := 170
def remaining_cakes := 186
def sold_cakes (S : ℕ) : Prop := initial_cakes - S + new_cakes = remaining_cakes

theorem determine_sold_cakes : ∃ S, sold_cakes S ∧ S = 105 :=
by
  use 105
  unfold sold_cakes
  simp
  sorry

end NUMINAMATH_GPT_determine_sold_cakes_l1795_179502


namespace NUMINAMATH_GPT_batsman_average_after_11th_inning_l1795_179559

theorem batsman_average_after_11th_inning
  (x : ℝ)  -- the average score of the batsman before the 11th inning
  (h1 : 10 * x + 85 = 11 * (x + 5))  -- given condition from the problem
  : x + 5 = 35 :=   -- goal statement proving the new average
by
  -- We need to prove that new average after the 11th inning is 35
  sorry

end NUMINAMATH_GPT_batsman_average_after_11th_inning_l1795_179559


namespace NUMINAMATH_GPT_ordering_PQR_l1795_179514

noncomputable def P := Real.sqrt 2
noncomputable def Q := Real.sqrt 7 - Real.sqrt 3
noncomputable def R := Real.sqrt 6 - Real.sqrt 2

theorem ordering_PQR : P > R ∧ R > Q := by
  sorry

end NUMINAMATH_GPT_ordering_PQR_l1795_179514


namespace NUMINAMATH_GPT_infection_never_covers_grid_l1795_179575

theorem infection_never_covers_grid (n : ℕ) (H : n > 0) :
  exists (non_infected_cell : ℕ × ℕ), (non_infected_cell.1 < n ∧ non_infected_cell.2 < n) :=
by
  sorry

end NUMINAMATH_GPT_infection_never_covers_grid_l1795_179575


namespace NUMINAMATH_GPT_right_handed_total_l1795_179592

theorem right_handed_total (total_players throwers : Nat) (h1 : total_players = 70) (h2 : throwers = 37) :
  let non_throwers := total_players - throwers
  let left_handed := non_throwers / 3
  let right_handed_non_throwers := non_throwers - left_handed
  let right_handed := right_handed_non_throwers + throwers
  right_handed = 59 :=
by
  sorry

end NUMINAMATH_GPT_right_handed_total_l1795_179592
