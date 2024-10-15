import Mathlib

namespace NUMINAMATH_GPT_collinear_points_sum_l2145_214542

theorem collinear_points_sum (p q : ℝ) 
  (h1 : p = 2) (h2 : q = 4) 
  (collinear : ∃ (s : ℝ), 
     (2, p, q) = (2, s*p, s*q) ∧ 
     (p, 3, q) = (s*p, 3, s*q) ∧ 
     (p, q, 4) = (s*p, s*q, 4)): 
  p + q = 6 := by
  sorry

end NUMINAMATH_GPT_collinear_points_sum_l2145_214542


namespace NUMINAMATH_GPT_minimize_y_at_x_l2145_214510

noncomputable def minimize_y (a b x : ℝ) : ℝ :=
  (x - a)^2 + (x - b)^2 + 2 * (a - b) * x

theorem minimize_y_at_x (a b : ℝ) :
  ∃ x : ℝ, minimize_y a b x = minimize_y a b (b / 2) := by
  sorry

end NUMINAMATH_GPT_minimize_y_at_x_l2145_214510


namespace NUMINAMATH_GPT_Emily_total_cost_l2145_214516

theorem Emily_total_cost :
  let cost_curtains := 2 * 30
  let cost_prints := 9 * 15
  let installation_cost := 50
  let total_cost := cost_curtains + cost_prints + installation_cost
  total_cost = 245 := by
{
 sorry
}

end NUMINAMATH_GPT_Emily_total_cost_l2145_214516


namespace NUMINAMATH_GPT_increasing_sequence_a_range_l2145_214534

theorem increasing_sequence_a_range (a : ℝ) (a_seq : ℕ → ℝ) (h_def : ∀ n, a_seq n = 
  if n ≤ 2 then a * n^2 - ((7 / 8) * a + 17 / 4) * n + 17 / 2
  else a ^ n) : 
  (∀ n, a_seq n < a_seq (n + 1)) → a > 2 :=
by
  sorry

end NUMINAMATH_GPT_increasing_sequence_a_range_l2145_214534


namespace NUMINAMATH_GPT_solution_set_a_range_m_l2145_214582

theorem solution_set_a (a : ℝ) :
  (∀ x : ℝ, |x - a| ≤ 3 ↔ -6 ≤ x ∧ x ≤ 0) ↔ a = -3 :=
by
  sorry

theorem range_m (m : ℝ) :
  (∀ x : ℝ, |x + 3| + |x + 8| ≥ 2 * m) ↔ m ≤ 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_a_range_m_l2145_214582


namespace NUMINAMATH_GPT_proof_by_contradiction_x_gt_y_implies_x3_gt_y3_l2145_214564

theorem proof_by_contradiction_x_gt_y_implies_x3_gt_y3
  (x y: ℝ) (h: x > y) : ¬ (x^3 ≤ y^3) :=
by
  -- We need to show that assuming x^3 <= y^3 leads to a contradiction
  sorry

end NUMINAMATH_GPT_proof_by_contradiction_x_gt_y_implies_x3_gt_y3_l2145_214564


namespace NUMINAMATH_GPT_sum_of_g_35_l2145_214509

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 - 3
noncomputable def g (y : ℝ) : ℝ := y^2 + y + 1

theorem sum_of_g_35 : g 35 = 21 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_g_35_l2145_214509


namespace NUMINAMATH_GPT_inequality_and_equality_conditions_l2145_214588

theorem inequality_and_equality_conditions
    {a b c d : ℝ}
    (ha : 0 < a)
    (hb : 0 < b)
    (hc : 0 < c)
    (hd : 0 < d) :
  (a ^ (1/3) * b ^ (1/3) + c ^ (1/3) * d ^ (1/3) ≤ (a + b + c) ^ (1/3) * (a + c + d) ^ (1/3)) ↔ 
  (b = (a / c) * (a + c) ∧ d = (c / a) * (a + c)) :=
  sorry

end NUMINAMATH_GPT_inequality_and_equality_conditions_l2145_214588


namespace NUMINAMATH_GPT_sum_first_20_terms_arithmetic_seq_l2145_214518

theorem sum_first_20_terms_arithmetic_seq :
  ∃ (a d : ℤ) (S_20 : ℤ), d > 0 ∧
  (a + 2 * d) * (a + 6 * d) = -12 ∧
  (a + 3 * d) + (a + 5 * d) = -4 ∧
  S_20 = 20 * a + (20 * 19 / 2) * d ∧
  S_20 = 180 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_20_terms_arithmetic_seq_l2145_214518


namespace NUMINAMATH_GPT_average_minutes_correct_l2145_214503

variable (s : ℕ)
def sixth_graders := 3 * s
def seventh_graders := s
def eighth_graders := s / 2

def minutes_sixth_graders := 18 * sixth_graders s
def minutes_seventh_graders := 20 * seventh_graders s
def minutes_eighth_graders := 22 * eighth_graders s

def total_minutes := minutes_sixth_graders s + minutes_seventh_graders s + minutes_eighth_graders s
def total_students := sixth_graders s + seventh_graders s + eighth_graders s

def average_minutes := total_minutes s / total_students s

theorem average_minutes_correct : average_minutes s = 170 / 9 := sorry

end NUMINAMATH_GPT_average_minutes_correct_l2145_214503


namespace NUMINAMATH_GPT_find_m_l2145_214589

variable {a_n : ℕ → ℤ}
variable {S : ℕ → ℤ}

def isArithmeticSeq (a_n : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a_n (n + 1) = a_n n + d

def sumSeq (S : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
∀ n, S n = (n * (a_n 1 + a_n n)) / 2

theorem find_m
  (d : ℤ)
  (a_1 : ℤ)
  (a_n : ∀ n, ℤ)
  (S : ℕ → ℤ)
  (h_arith : isArithmeticSeq a_n d)
  (h_sum : sumSeq S a_n)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end NUMINAMATH_GPT_find_m_l2145_214589


namespace NUMINAMATH_GPT_polygon_sides_l2145_214515

theorem polygon_sides {n : ℕ} (h : (n - 2) * 180 = 1080) : n = 8 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l2145_214515


namespace NUMINAMATH_GPT_velocity_zero_at_t_eq_2_l2145_214513

noncomputable def motion_equation (t : ℝ) : ℝ := -4 * t^3 + 48 * t

theorem velocity_zero_at_t_eq_2 :
  (exists t : ℝ, t > 0 ∧ deriv (motion_equation) t = 0) :=
by
  sorry

end NUMINAMATH_GPT_velocity_zero_at_t_eq_2_l2145_214513


namespace NUMINAMATH_GPT_gcd_ab_l2145_214566

def a : ℕ := 130^2 + 215^2 + 310^2
def b : ℕ := 131^2 + 216^2 + 309^2

theorem gcd_ab : Nat.gcd a b = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_ab_l2145_214566


namespace NUMINAMATH_GPT_RachelStillToColor_l2145_214541

def RachelColoringBooks : Prop :=
  let initial_books := 23 + 32
  let colored := 44
  initial_books - colored = 11

theorem RachelStillToColor : RachelColoringBooks := 
  by
    let initial_books := 23 + 32
    let colored := 44
    show initial_books - colored = 11
    sorry

end NUMINAMATH_GPT_RachelStillToColor_l2145_214541


namespace NUMINAMATH_GPT_fraction_identity_l2145_214500

theorem fraction_identity
  (x w y z : ℝ)
  (hxw_pos : x * w > 0)
  (hyz_pos : y * z > 0)
  (hxw_inv_sum : 1 / x + 1 / w = 20)
  (hyz_inv_sum : 1 / y + 1 / z = 25)
  (hxw_inv : 1 / (x * w) = 6)
  (hyz_inv : 1 / (y * z) = 8) :
  (x + y) / (z + w) = 155 / 7 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_fraction_identity_l2145_214500


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l2145_214579

theorem solve_equation_1 (x : ℝ) : x^2 - 3 * x = 4 ↔ x = 4 ∨ x = -1 :=
by
  sorry

theorem solve_equation_2 (x : ℝ) : x * (x - 2) + x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l2145_214579


namespace NUMINAMATH_GPT_transportation_degrees_correct_l2145_214585

-- Define the percentages for the different categories.
def salaries_percent := 0.60
def research_development_percent := 0.09
def utilities_percent := 0.05
def equipment_percent := 0.04
def supplies_percent := 0.02

-- Define the total percentage of non-transportation categories.
def non_transportation_percent := 
  salaries_percent + research_development_percent + utilities_percent + equipment_percent + supplies_percent

-- Define the full circle in degrees.
def full_circle_degrees := 360.0

-- Total percentage which must sum to 1 (i.e., 100%).
def total_budget_percent := 1.0

-- Calculate the percentage for transportation.
def transportation_percent := total_budget_percent - non_transportation_percent

-- Define the result for degrees allocated to transportation.
def transportation_degrees := transportation_percent * full_circle_degrees

-- Prove that the transportation degrees are 72.
theorem transportation_degrees_correct : transportation_degrees = 72.0 :=
by
  unfold transportation_degrees transportation_percent non_transportation_percent
  sorry

end NUMINAMATH_GPT_transportation_degrees_correct_l2145_214585


namespace NUMINAMATH_GPT_james_total_socks_l2145_214508

-- Definitions based on conditions
def red_pairs : ℕ := 20
def black_pairs : ℕ := red_pairs / 2
def white_pairs : ℕ := 2 * (red_pairs + black_pairs)
def green_pairs : ℕ := (red_pairs + black_pairs + white_pairs) + 5

-- Total number of pairs
def total_pairs := red_pairs + black_pairs + white_pairs + green_pairs

-- Total number of socks
def total_socks := total_pairs * 2

-- The main theorem to prove the total number of socks
theorem james_total_socks : total_socks = 370 :=
  by
  -- proof is skipped
  sorry

end NUMINAMATH_GPT_james_total_socks_l2145_214508


namespace NUMINAMATH_GPT_part1_part2_l2145_214590

def f (x : ℝ) : ℝ := abs (2 * x - 4) + abs (x + 1)

theorem part1 (x : ℝ) : f x ≤ 9 → x ∈ Set.Icc (-2 : ℝ) 4 :=
sorry

theorem part2 (a : ℝ) :
  (∃ x ∈ Set.Icc (0 : ℝ) (2 : ℝ), f x = -x^2 + a) →
  (a ∈ Set.Icc (19 / 4) (7 : ℝ)) :=
sorry

end NUMINAMATH_GPT_part1_part2_l2145_214590


namespace NUMINAMATH_GPT_function_form_l2145_214599

def satisfies_condition (f : ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, m > 0 → n > 0 → ⌊ (f (m * n) : ℚ) / n ⌋ = f m

theorem function_form (f : ℕ → ℤ) (h : satisfies_condition f) :
  ∃ r : ℝ, ∀ n : ℕ, 
    (f n = ⌊ (r * n : ℝ) ⌋) ∨ (f n = ⌈ (r * n : ℝ) ⌉ - 1) := 
  sorry

end NUMINAMATH_GPT_function_form_l2145_214599


namespace NUMINAMATH_GPT_Lauryn_employs_80_men_l2145_214522

theorem Lauryn_employs_80_men (W M : ℕ) 
  (h1 : M = W - 20) 
  (h2 : M + W = 180) : 
  M = 80 := 
by 
  sorry

end NUMINAMATH_GPT_Lauryn_employs_80_men_l2145_214522


namespace NUMINAMATH_GPT_part1_part2_l2145_214526

-- Part (1)
theorem part1 (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) 
  (h1 : ∀ n, S_n n = n * a_n 1 + (n-1) * a_n 2 + 2 * a_n (n-1) + a_n n)
  (arithmetic_seq : ∀ n, a_n (n+1) = a_n n + d)
  (S1_eq : S_n 1 = 5)
  (S2_eq : S_n 2 = 18) :
  ∀ n, a_n n = 3 * n + 2 := by
  sorry

-- Part (2)
theorem part2 (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) 
  (h1 : ∀ n, S_n n = n * a_n 1 + (n-1) * a_n 2 + 2 * a_n (n-1) + a_n n)
  (geometric_seq : ∃ q, ∀ n, a_n (n+1) = q * a_n n)
  (S1_eq : S_n 1 = 3)
  (S2_eq : S_n 2 = 15) :
  ∀ n, S_n n = (3^(n+2) - 6 * n - 9) / 4 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l2145_214526


namespace NUMINAMATH_GPT_ann_frosting_time_l2145_214592

theorem ann_frosting_time (time_normal time_sprained n : ℕ) (h1 : time_normal = 5) (h2 : time_sprained = 8) (h3 : n = 10) : 
  ((time_sprained * n) - (time_normal * n)) = 30 := 
by 
  sorry

end NUMINAMATH_GPT_ann_frosting_time_l2145_214592


namespace NUMINAMATH_GPT_hall_area_l2145_214507

theorem hall_area (L W : ℝ) 
  (h1 : W = (1/2) * L)
  (h2 : L - W = 8) : 
  L * W = 128 := 
  sorry

end NUMINAMATH_GPT_hall_area_l2145_214507


namespace NUMINAMATH_GPT_slope_of_line_of_intersections_l2145_214555

theorem slope_of_line_of_intersections : 
  ∀ s : ℝ, let x := (41 * s + 13) / 11
           let y := -((2 * s + 6) / 11)
           ∃ m : ℝ, m = -22 / 451 :=
sorry

end NUMINAMATH_GPT_slope_of_line_of_intersections_l2145_214555


namespace NUMINAMATH_GPT_wooden_easel_cost_l2145_214536

noncomputable def cost_paintbrush : ℝ := 1.5
noncomputable def cost_set_of_paints : ℝ := 4.35
noncomputable def amount_already_have : ℝ := 6.5
noncomputable def additional_amount_needed : ℝ := 12
noncomputable def total_cost_items : ℝ := cost_paintbrush + cost_set_of_paints
noncomputable def total_amount_needed : ℝ := amount_already_have + additional_amount_needed

theorem wooden_easel_cost :
  total_amount_needed - total_cost_items = 12.65 :=
by
  sorry

end NUMINAMATH_GPT_wooden_easel_cost_l2145_214536


namespace NUMINAMATH_GPT_competition_sequences_l2145_214514

-- Define the problem conditions
def team_size : Nat := 7

-- Define the statement to prove
theorem competition_sequences :
  (Nat.choose (2 * team_size) team_size) = 3432 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_competition_sequences_l2145_214514


namespace NUMINAMATH_GPT_points_product_l2145_214539

def f (n : ℕ) : ℕ :=
  if n % 6 == 0 then 6
  else if n % 2 == 0 then 2
  else 0

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

def allie_rolls := [5, 4, 1, 2]
def betty_rolls := [6, 3, 3, 2]

def allie_points := total_points allie_rolls
def betty_points := total_points betty_rolls

theorem points_product : allie_points * betty_points = 32 := by
  sorry

end NUMINAMATH_GPT_points_product_l2145_214539


namespace NUMINAMATH_GPT_tetrahedron_volume_l2145_214535

theorem tetrahedron_volume (a b c : ℝ)
  (h₁ : a + b > c) (h₂ : a + c > b) (h₃ : b + c > a) :
  ∃ V : ℝ, 
    V = (1 / (6 * Real.sqrt 2)) * 
        Real.sqrt ((a^2 + b^2 - c^2) * (a^2 + c^2 - b^2) * (b^2 + c^2 - a^2)) :=
sorry

end NUMINAMATH_GPT_tetrahedron_volume_l2145_214535


namespace NUMINAMATH_GPT_solve_linear_system_l2145_214574

theorem solve_linear_system (m x y : ℝ) 
  (h1 : x + y = 3 * m) 
  (h2 : x - y = 5 * m)
  (h3 : 2 * x + 3 * y = 10) : 
  m = 2 := 
by 
  sorry

end NUMINAMATH_GPT_solve_linear_system_l2145_214574


namespace NUMINAMATH_GPT_smallest_possible_degree_p_l2145_214545

theorem smallest_possible_degree_p (p : Polynomial ℝ) :
  (∀ x, 0 < |x| → ∃ C, |((3 * x^7 + 2 * x^6 - 4 * x^3 + x - 5) / (p.eval x)) - C| < ε)
  → (Polynomial.degree p) ≥ 7 := by
  sorry

end NUMINAMATH_GPT_smallest_possible_degree_p_l2145_214545


namespace NUMINAMATH_GPT_base_four_odd_last_digit_l2145_214587

theorem base_four_odd_last_digit :
  ∃ b : ℕ, b = 4 ∧ (b^4 ≤ 625 ∧ 625 < b^5) ∧ (625 % b % 2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_base_four_odd_last_digit_l2145_214587


namespace NUMINAMATH_GPT_monotonicity_intervals_f_above_g_l2145_214596

noncomputable def f (x m : ℝ) := (Real.exp x) / (x^2 - m * x + 1)

theorem monotonicity_intervals (m : ℝ) (h : m ∈ Set.Ioo (-2 : ℝ) 2) :
  (m = 0 → ∀ x y : ℝ, x ≤ y → f x m ≤ f y m) ∧ 
  (0 < m ∧ m < 2 → ∀ x : ℝ, (x < 1 → f x m < f (x + 1) m) ∧
    (1 < x ∧ x < m + 1 → f x m > f (x + 1) m) ∧
    (x > m + 1 → f x m < f (x + 1) m)) ∧
  (-2 < m ∧ m < 0 → ∀ x : ℝ, (x < m + 1 → f x m < f (x + 1) m) ∧
    (m + 1 < x ∧ x < 1 → f x m > f (x + 1) m) ∧
    (x > 1 → f x m < f (x + 1) m)) :=
sorry

theorem f_above_g (m : ℝ) (hm : m ∈ Set.Ioo (0 : ℝ) (1/2 : ℝ)) (x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) (m + 1)) :
  f x m > x :=
sorry

end NUMINAMATH_GPT_monotonicity_intervals_f_above_g_l2145_214596


namespace NUMINAMATH_GPT_maxwell_walking_speed_l2145_214556

theorem maxwell_walking_speed :
  ∃ v : ℝ, (8 * v + 6 * 7 = 74) ∧ v = 4 :=
by
  exists 4
  constructor
  { norm_num }
  rfl

end NUMINAMATH_GPT_maxwell_walking_speed_l2145_214556


namespace NUMINAMATH_GPT_minimal_circle_intersect_l2145_214543

noncomputable def circle_eq := 
  ∀ (x y : ℝ), 
    (x^2 + y^2 + 4 * x + y + 1 = 0) ∧
    (x^2 + y^2 + 2 * x + 2 * y + 1 = 0) → 
    (x^2 + y^2 + (6/5) * x + (3/5) * y + 1 = 0)

theorem minimal_circle_intersect :
  circle_eq :=
by
  sorry

end NUMINAMATH_GPT_minimal_circle_intersect_l2145_214543


namespace NUMINAMATH_GPT_last_digit_of_sum_edges_l2145_214520

def total_edges (n : ℕ) : ℕ := (n + 1) * n * 2

def internal_edges (n : ℕ) : ℕ := (n - 1) * n * 2

def dominoes (n : ℕ) : ℕ := (n * n) / 2

def perfect_matchings (n : ℕ) : ℕ := if n = 8 then 12988816 else 0  -- specific to 8x8 chessboard

def sum_internal_edges_contribution (n : ℕ) : ℕ := perfect_matchings n * (dominoes n * 2)

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_sum_edges {n : ℕ} (h : n = 8) :
  last_digit (sum_internal_edges_contribution n) = 4 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_last_digit_of_sum_edges_l2145_214520


namespace NUMINAMATH_GPT_balance_three_diamonds_l2145_214568

-- Define the problem conditions
variables (a b c : ℕ)

-- Four Δ's and two ♦'s will balance twelve ●'s
def condition1 : Prop :=
  4 * a + 2 * b = 12 * c

-- One Δ will balance a ♦ and two ●'s
def condition2 : Prop :=
  a = b + 2 * c

-- Theorem to prove how many ●'s will balance three ♦'s
theorem balance_three_diamonds (h1 : condition1 a b c) (h2 : condition2 a b c) : 3 * b = 2 * c :=
by sorry

end NUMINAMATH_GPT_balance_three_diamonds_l2145_214568


namespace NUMINAMATH_GPT_avg_daily_distance_third_dog_summer_l2145_214557

theorem avg_daily_distance_third_dog_summer :
  ∀ (total_days weekends miles_walked_weekday : ℕ), 
    total_days = 30 → weekends = 8 → miles_walked_weekday = 3 →
    (66 / 30 : ℝ) = 2.2 :=
by
  intros total_days weekends miles_walked_weekday h_total h_weekends h_walked
  -- proof goes here
  sorry

end NUMINAMATH_GPT_avg_daily_distance_third_dog_summer_l2145_214557


namespace NUMINAMATH_GPT_cost_of_fencing_irregular_pentagon_l2145_214530

noncomputable def total_cost_fencing (AB BC CD DE AE : ℝ) (cost_per_meter : ℝ) : ℝ := 
  (AB + BC + CD + DE + AE) * cost_per_meter

theorem cost_of_fencing_irregular_pentagon :
  total_cost_fencing 20 25 30 35 40 2 = 300 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_fencing_irregular_pentagon_l2145_214530


namespace NUMINAMATH_GPT_largest_whole_number_lt_150_l2145_214561

theorem largest_whole_number_lt_150 : 
  ∃ x : ℕ, (9 * x < 150) ∧ (∀ y : ℕ, 9 * y < 150 → y ≤ x) :=
  sorry

end NUMINAMATH_GPT_largest_whole_number_lt_150_l2145_214561


namespace NUMINAMATH_GPT_smallest_a_value_l2145_214529

theorem smallest_a_value :
  ∃ (a : ℝ), (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 →
    2 * (Real.sin (Real.pi - (Real.pi * x^2 / 12))) * (Real.cos (Real.pi / 6 * Real.sqrt (9 - x^2))) + 1 = a + 2 * (Real.sin (Real.pi / 6 * Real.sqrt (9 - x^2))) * (Real.cos (Real.pi * x^2 / 12))) ∧
    ∀ a' : ℝ, (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 →
      2 * (Real.sin (Real.pi - (Real.pi * x^2 / 12))) * (Real.cos (Real.pi / 6 * Real.sqrt (9 - x^2))) + 1 = a' + 2 * (Real.sin (Real.pi / 6 * Real.sqrt (9 - x^2))) * (Real.cos (Real.pi * x^2 / 12))) →
      a ≤ a'
  := sorry

end NUMINAMATH_GPT_smallest_a_value_l2145_214529


namespace NUMINAMATH_GPT_lisa_balls_count_l2145_214577

def stepNumber := 1729

def base7DigitsSum(x : Nat) : Nat :=
  x / 7 ^ 3 + (x % 343) / 7 ^ 2 + (x % 49) / 7 + x % 7

theorem lisa_balls_count (h1 : stepNumber = 1729) : base7DigitsSum stepNumber = 11 := by
  sorry

end NUMINAMATH_GPT_lisa_balls_count_l2145_214577


namespace NUMINAMATH_GPT_overlapping_area_zero_l2145_214527

-- Definition of the points and triangles
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

def point0 : Point := { x := 0, y := 0 }
def point1 : Point := { x := 2, y := 2 }
def point2 : Point := { x := 2, y := 0 }
def point3 : Point := { x := 0, y := 2 }
def point4 : Point := { x := 1, y := 1 }

def triangle1 : Triangle := { p1 := point0, p2 := point1, p3 := point2 }
def triangle2 : Triangle := { p1 := point3, p2 := point1, p3 := point0 }

-- Function to calculate the area of a triangle
def area (t : Triangle) : ℝ :=
  0.5 * abs (t.p1.x * (t.p2.y - t.p3.y) + t.p2.x * (t.p3.y - t.p1.y) + t.p3.x * (t.p1.y - t.p2.y))

-- Using collinear points theorem to prove that the area of the overlapping region is zero
theorem overlapping_area_zero : area { p1 := point0, p2 := point1, p3 := point4 } = 0 := 
by 
  -- This follows directly from the fact that the points (0,0), (2,2), and (1,1) are collinear
  -- skipping the actual geometric proof for conciseness
  sorry

end NUMINAMATH_GPT_overlapping_area_zero_l2145_214527


namespace NUMINAMATH_GPT_seven_power_units_digit_l2145_214570

def units_digit (n : ℕ) : ℕ := (7^n) % 10

theorem seven_power_units_digit : units_digit 2023 = 3 := by
  -- Proof omitted for simplification
  sorry

end NUMINAMATH_GPT_seven_power_units_digit_l2145_214570


namespace NUMINAMATH_GPT_workman_problem_l2145_214524

theorem workman_problem
    (total_work : ℝ)
    (B_rate : ℝ)
    (A_rate : ℝ)
    (days_together : ℝ)
    (W : total_work = 8 * (A_rate + B_rate))
    (A_2B : A_rate = 2 * B_rate) :
    total_work = 24 * B_rate :=
by
  sorry

end NUMINAMATH_GPT_workman_problem_l2145_214524


namespace NUMINAMATH_GPT_base6_sum_correct_l2145_214544

theorem base6_sum_correct {S H E : ℕ} (hS : S < 6) (hH : H < 6) (hE : E < 6) 
  (dist : S ≠ H ∧ H ≠ E ∧ S ≠ E) 
  (rightmost : (E + E) % 6 = S) 
  (second_rightmost : (H + H + if E + E < 6 then 0 else 1) % 6 = E) :
  S + H + E = 11 := 
by sorry

end NUMINAMATH_GPT_base6_sum_correct_l2145_214544


namespace NUMINAMATH_GPT_range_of_a_l2145_214505

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 3| + |x + 5| > a) → a < 8 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l2145_214505


namespace NUMINAMATH_GPT_arithmetic_sequence_a2015_l2145_214537

theorem arithmetic_sequence_a2015 :
  ∀ {a : ℕ → ℤ}, (a 1 = 2 ∧ a 5 = 6 ∧ (∀ n, a (n + 1) = a n + a 2 - a 1)) → a 2015 = 2016 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a2015_l2145_214537


namespace NUMINAMATH_GPT_prime_product_sum_91_l2145_214575

theorem prime_product_sum_91 (p1 p2 : ℕ) (h1 : Nat.Prime p1) (h2 : Nat.Prime p2) (h3 : p1 + p2 = 91) : p1 * p2 = 178 :=
sorry

end NUMINAMATH_GPT_prime_product_sum_91_l2145_214575


namespace NUMINAMATH_GPT_inequality_proof_l2145_214551

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2 / 3 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2145_214551


namespace NUMINAMATH_GPT_carries_average_speed_is_approx_34_29_l2145_214538

noncomputable def CarriesActualAverageSpeed : ℝ :=
  let jerry_speed := 40 -- in mph
  let jerry_time := 1/2 -- in hours, 30 minutes = 0.5 hours
  let jerry_distance := jerry_speed * jerry_time

  let beth_distance := jerry_distance + 5
  let beth_time := jerry_time + (20 / 60) -- converting 20 minutes to hours

  let carrie_distance := 2 * jerry_distance
  let carrie_time := 1 + (10 / 60) -- converting 10 minutes to hours

  carrie_distance / carrie_time

theorem carries_average_speed_is_approx_34_29 : 
  |CarriesActualAverageSpeed - 34.29| < 0.01 :=
sorry

end NUMINAMATH_GPT_carries_average_speed_is_approx_34_29_l2145_214538


namespace NUMINAMATH_GPT_train_lengths_l2145_214540

theorem train_lengths (L_A L_P L_B : ℕ) (speed_A_km_hr speed_B_km_hr : ℕ) (time_A_seconds : ℕ)
                      (h1 : L_P = L_A)
                      (h2 : speed_A_km_hr = 72)
                      (h3 : speed_B_km_hr = 80)
                      (h4 : time_A_seconds = 60)
                      (h5 : L_B = L_P / 2)
                      (h6 : L_A + L_P = (speed_A_km_hr * 1000 / 3600) * time_A_seconds) :
  L_A = 600 ∧ L_B = 300 :=
by
  sorry

end NUMINAMATH_GPT_train_lengths_l2145_214540


namespace NUMINAMATH_GPT_geometric_progression_sum_ratio_l2145_214597

theorem geometric_progression_sum_ratio (a : ℝ) (r n : ℕ) (hn : r = 3)
  (h : (a * (1 - r^n) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 28) : n = 6 :=
by
  -- Place the steps of the proof here, which are not required as per instructions.
  sorry

end NUMINAMATH_GPT_geometric_progression_sum_ratio_l2145_214597


namespace NUMINAMATH_GPT_complete_square_transformation_l2145_214573

theorem complete_square_transformation (x : ℝ) : 
  2 * x^2 - 4 * x - 3 = 0 ↔ (x - 1)^2 - (5 / 2) = 0 :=
sorry

end NUMINAMATH_GPT_complete_square_transformation_l2145_214573


namespace NUMINAMATH_GPT_heather_average_balance_l2145_214595

theorem heather_average_balance :
  let balance_J := 150
  let balance_F := 250
  let balance_M := 100
  let balance_A := 200
  let balance_May := 300
  let total_balance := balance_J + balance_F + balance_M + balance_A + balance_May
  let avg_balance := total_balance / 5
  avg_balance = 200 :=
by
  sorry

end NUMINAMATH_GPT_heather_average_balance_l2145_214595


namespace NUMINAMATH_GPT_train_length_is_400_l2145_214565

-- Define the conditions
def time := 40 -- seconds
def speed_kmh := 36 -- km/h

-- Conversion factor from km/h to m/s
def kmh_to_ms (v : ℕ) := (v * 5) / 18

def speed_ms := kmh_to_ms speed_kmh -- convert speed to m/s

-- Definition of length of the train using the given conditions
def train_length := speed_ms * time

-- Theorem to prove the length of the train is 400 meters
theorem train_length_is_400 : train_length = 400 := by
  sorry

end NUMINAMATH_GPT_train_length_is_400_l2145_214565


namespace NUMINAMATH_GPT_sum_of_w_l2145_214523

def g (y : ℝ) : ℝ := (2 * y)^3 - 2 * (2 * y) + 5

theorem sum_of_w (w1 w2 w3 : ℝ)
  (hw1 : g (2 * w1) = 13)
  (hw2 : g (2 * w2) = 13)
  (hw3 : g (2 * w3) = 13) :
  w1 + w2 + w3 = -1 / 4 :=
sorry

end NUMINAMATH_GPT_sum_of_w_l2145_214523


namespace NUMINAMATH_GPT_line_form_x_eq_ky_add_b_perpendicular_y_line_form_x_eq_ky_add_b_perpendicular_x_l2145_214547

theorem line_form_x_eq_ky_add_b_perpendicular_y {k b : ℝ} : 
  ¬ ∃ c : ℝ, x = c ∧ ∀ y : ℝ, x = k*y + b :=
sorry

theorem line_form_x_eq_ky_add_b_perpendicular_x {b : ℝ} : 
  ∃ k : ℝ, k = 0 ∧ ∀ y : ℝ, x = k*y + b :=
sorry

end NUMINAMATH_GPT_line_form_x_eq_ky_add_b_perpendicular_y_line_form_x_eq_ky_add_b_perpendicular_x_l2145_214547


namespace NUMINAMATH_GPT_one_clerk_forms_per_hour_l2145_214572

theorem one_clerk_forms_per_hour
  (total_forms : ℕ)
  (total_hours : ℕ)
  (total_clerks : ℕ) 
  (h1 : total_forms = 2400)
  (h2 : total_hours = 8)
  (h3 : total_clerks = 12) :
  (total_forms / total_hours) / total_clerks = 25 :=
by
  have forms_per_hour := total_forms / total_hours
  have forms_per_clerk_per_hour := forms_per_hour / total_clerks
  sorry

end NUMINAMATH_GPT_one_clerk_forms_per_hour_l2145_214572


namespace NUMINAMATH_GPT_total_birds_remaining_l2145_214594

theorem total_birds_remaining (grey_birds_in_cage : ℕ) (white_birds_next_to_cage : ℕ) :
  (grey_birds_in_cage = 40) →
  (white_birds_next_to_cage = grey_birds_in_cage + 6) →
  (1/2 * grey_birds_in_cage = 20) →
  (1/2 * grey_birds_in_cage + white_birds_next_to_cage = 66) :=
by 
  intros h_grey_birds h_white_birds h_grey_birds_freed
  sorry

end NUMINAMATH_GPT_total_birds_remaining_l2145_214594


namespace NUMINAMATH_GPT_pentagon_perimeter_l2145_214512

noncomputable def perimeter_pentagon (FG GH HI IJ : ℝ) (FH FI FJ : ℝ) : ℝ :=
  FG + GH + HI + IJ + FJ

theorem pentagon_perimeter : 
  ∀ (FG GH HI IJ : ℝ), 
  ∀ (FH FI FJ : ℝ),
  FG = 1 → GH = 1 → HI = 1 → IJ = 1 →
  FH^2 = FG^2 + GH^2 → FI^2 = FH^2 + HI^2 → FJ^2 = FI^2 + IJ^2 →
  perimeter_pentagon FG GH HI IJ FJ = 6 :=
by
  intros FG GH HI IJ FH FI FJ
  intros H_FG H_GH H_HI H_IJ
  intros H1 H2 H3
  sorry

end NUMINAMATH_GPT_pentagon_perimeter_l2145_214512


namespace NUMINAMATH_GPT_inequality_solution_set_l2145_214581

noncomputable def solution_set := { x : ℝ | (x < -1 ∨ 1 < x) ∧ x ≠ 4 }

theorem inequality_solution_set : 
  { x : ℝ | (x^2 - 1) / (4 - x)^2 ≥ 0 } = solution_set :=
  by 
    sorry

end NUMINAMATH_GPT_inequality_solution_set_l2145_214581


namespace NUMINAMATH_GPT_correct_answer_l2145_214576

-- Definitions of the groups
def group_1_well_defined : Prop := false -- Smaller numbers
def group_2_well_defined : Prop := true  -- Non-negative even numbers not greater than 10
def group_3_well_defined : Prop := true  -- All triangles
def group_4_well_defined : Prop := false -- Tall male students

-- Propositions representing the options
def option_A : Prop := group_1_well_defined ∧ group_4_well_defined
def option_B : Prop := group_2_well_defined ∧ group_3_well_defined
def option_C : Prop := group_2_well_defined
def option_D : Prop := group_3_well_defined

-- Theorem stating Option B is the correct answer
theorem correct_answer : option_B ∧ ¬option_A ∧ ¬option_C ∧ ¬option_D := by
  sorry

end NUMINAMATH_GPT_correct_answer_l2145_214576


namespace NUMINAMATH_GPT_remainder_is_nine_l2145_214511

-- Define the dividend and divisor
def n : ℕ := 4039
def d : ℕ := 31

-- Prove that n mod d equals 9
theorem remainder_is_nine : n % d = 9 := by
  sorry

end NUMINAMATH_GPT_remainder_is_nine_l2145_214511


namespace NUMINAMATH_GPT_robotics_club_neither_l2145_214502

theorem robotics_club_neither (total students programming electronics both: ℕ) 
  (h1: total = 120)
  (h2: programming = 80)
  (h3: electronics = 50)
  (h4: both = 15) : 
  total - ((programming - both) + (electronics - both) + both) = 5 :=
by
  sorry

end NUMINAMATH_GPT_robotics_club_neither_l2145_214502


namespace NUMINAMATH_GPT_population_of_town_l2145_214560

theorem population_of_town (F : ℝ) (males : ℕ) (female_glasses : ℝ) (percentage_glasses : ℝ) (total_population : ℝ) 
  (h1 : males = 2000) 
  (h2 : percentage_glasses = 0.30) 
  (h3 : female_glasses = 900) 
  (h4 : percentage_glasses * F = female_glasses) 
  (h5 : total_population = males + F) :
  total_population = 5000 :=
sorry

end NUMINAMATH_GPT_population_of_town_l2145_214560


namespace NUMINAMATH_GPT_paul_diner_total_cost_l2145_214504

/-- At Paul's Diner, sandwiches cost $5 each and sodas cost $3 each. If a customer buys
more than 4 sandwiches, they receive a $10 discount on the total bill. Calculate the total
cost if a customer purchases 6 sandwiches and 3 sodas. -/
def totalCost (num_sandwiches num_sodas : ℕ) : ℕ :=
  let sandwich_cost := 5
  let soda_cost := 3
  let discount := if num_sandwiches > 4 then 10 else 0
  (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost) - discount

theorem paul_diner_total_cost : totalCost 6 3 = 29 :=
by
  sorry

end NUMINAMATH_GPT_paul_diner_total_cost_l2145_214504


namespace NUMINAMATH_GPT_number_of_round_table_arrangements_l2145_214519

theorem number_of_round_table_arrangements : (Nat.factorial 5) / 5 = 24 := 
by
  sorry

end NUMINAMATH_GPT_number_of_round_table_arrangements_l2145_214519


namespace NUMINAMATH_GPT_flowers_bouquets_l2145_214554

theorem flowers_bouquets (tulips: ℕ) (roses: ℕ) (extra: ℕ) (total: ℕ) (used_for_bouquets: ℕ) 
(h1: tulips = 36) 
(h2: roses = 37) 
(h3: extra = 3) 
(h4: total = tulips + roses)
(h5: used_for_bouquets = total - extra) :
used_for_bouquets = 70 := by
  sorry

end NUMINAMATH_GPT_flowers_bouquets_l2145_214554


namespace NUMINAMATH_GPT_jame_weeks_tearing_cards_l2145_214558

def cards_tears_per_time : ℕ := 30
def cards_per_deck : ℕ := 55
def tears_per_week : ℕ := 3
def decks_bought : ℕ := 18

theorem jame_weeks_tearing_cards :
  (cards_tears_per_time * tears_per_week * decks_bought * cards_per_deck) / (cards_tears_per_time * tears_per_week) = 11 := by
  sorry

end NUMINAMATH_GPT_jame_weeks_tearing_cards_l2145_214558


namespace NUMINAMATH_GPT_smallest_rectangles_required_l2145_214546

theorem smallest_rectangles_required :
  ∀ (r h : ℕ) (area_square length_square : ℕ),
  r = 3 → h = 4 →
  (∀ k, (k: ℕ) ∣ (r * h) → (k: ℕ) = r * h) →
  length_square = 12 →
  area_square = length_square * length_square →
  (area_square / (r * h) = 12) :=
by
  intros
  /- The mathematical proof steps will be filled here -/
  sorry

end NUMINAMATH_GPT_smallest_rectangles_required_l2145_214546


namespace NUMINAMATH_GPT_gcd_456_357_l2145_214578

theorem gcd_456_357 : Nat.gcd 456 357 = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_456_357_l2145_214578


namespace NUMINAMATH_GPT_largest_multiple_of_11_less_than_100_l2145_214569

theorem largest_multiple_of_11_less_than_100 : 
  ∀ n, n < 100 → (∃ k, n = k * 11) → n ≤ 99 :=
by
  intro n hn hmul
  sorry

end NUMINAMATH_GPT_largest_multiple_of_11_less_than_100_l2145_214569


namespace NUMINAMATH_GPT_solve_floor_trig_eq_l2145_214549

-- Define the floor function
def floor (x : ℝ) : ℤ := by 
  sorry

-- Define the condition and theorem
theorem solve_floor_trig_eq (x : ℝ) (n : ℤ) : 
  floor (Real.sin x + Real.cos x) = 1 ↔ (∃ n : ℤ, 2 * Real.pi * n ≤ x ∧ x ≤ (2 * Real.pi * n + Real.pi / 2)) := 
  by 
  sorry

end NUMINAMATH_GPT_solve_floor_trig_eq_l2145_214549


namespace NUMINAMATH_GPT_triangle_sides_square_perfect_l2145_214531

theorem triangle_sides_square_perfect (x y z : ℕ) (h : ∃ h_x h_y h_z, 
  h_x = h_y + h_z ∧ 
  2 * h_x * x = 2 * h_y * y ∧ 
  2 * h_x * x = 2 * h_z * z ) :
  ∃ k : ℕ, x^2 + y^2 + z^2 = k^2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_sides_square_perfect_l2145_214531


namespace NUMINAMATH_GPT_odd_function_f_x_pos_l2145_214550

variable (f : ℝ → ℝ)

theorem odd_function_f_x_pos {x : ℝ} (h1 : ∀ x < 0, f x = x^2 + x)
  (h2 : ∀ x, f x = -f (-x)) (hx : 0 < x) :
  f x = -x^2 + x := by
  sorry

end NUMINAMATH_GPT_odd_function_f_x_pos_l2145_214550


namespace NUMINAMATH_GPT_lcm_18_35_is_630_l2145_214506

def lcm_18_35 : ℕ :=
  Nat.lcm 18 35

theorem lcm_18_35_is_630 : lcm_18_35 = 630 := by
  sorry

end NUMINAMATH_GPT_lcm_18_35_is_630_l2145_214506


namespace NUMINAMATH_GPT_seeder_path_length_l2145_214521

theorem seeder_path_length (initial_grain : ℤ) (decrease_percent : ℝ) (seeding_rate : ℝ) (width : ℝ) 
  (H_initial_grain : initial_grain = 250) 
  (H_decrease_percent : decrease_percent = 14 / 100) 
  (H_seeding_rate : seeding_rate = 175) 
  (H_width : width = 4) :
  (initial_grain * decrease_percent / seeding_rate) * 10000 / width = 500 := 
by 
  sorry

end NUMINAMATH_GPT_seeder_path_length_l2145_214521


namespace NUMINAMATH_GPT_radius_of_base_of_cone_is_3_l2145_214593

noncomputable def radius_of_base_of_cone (θ R : ℝ) : ℝ :=
  ((θ / 360) * 2 * Real.pi * R) / (2 * Real.pi)

theorem radius_of_base_of_cone_is_3 :
  radius_of_base_of_cone 120 9 = 3 := 
by 
  simp [radius_of_base_of_cone]
  sorry

end NUMINAMATH_GPT_radius_of_base_of_cone_is_3_l2145_214593


namespace NUMINAMATH_GPT_distance_traveled_l2145_214591

theorem distance_traveled (speed time : ℕ) (h_speed : speed = 20) (h_time : time = 8) : 
  speed * time = 160 := 
by
  -- Solution proof goes here
  sorry

end NUMINAMATH_GPT_distance_traveled_l2145_214591


namespace NUMINAMATH_GPT_perimeter_square_l2145_214501

-- Definition of the side length
def side_length : ℝ := 9

-- Definition of the perimeter calculation
def perimeter (s : ℝ) : ℝ := 4 * s

-- Theorem stating that the perimeter of a square with side length 9 cm is 36 cm
theorem perimeter_square : perimeter side_length = 36 := 
by sorry

end NUMINAMATH_GPT_perimeter_square_l2145_214501


namespace NUMINAMATH_GPT_max_ab_value_l2145_214571

noncomputable def max_ab (a b : ℝ) : ℝ := a * b

theorem max_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 8) : max_ab a b ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_max_ab_value_l2145_214571


namespace NUMINAMATH_GPT_inequality_proof_l2145_214583

open Real

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 1) : 
  (a - b * c) / (a + b * c) + (b - c * a) / (b + c * a) + (c - a * b) / (c + a * b) ≤ 3 / 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2145_214583


namespace NUMINAMATH_GPT_cost_of_one_box_of_paper_clips_l2145_214567

theorem cost_of_one_box_of_paper_clips (p i : ℝ) 
  (h1 : 15 * p + 7 * i = 55.40) 
  (h2 : 12 * p + 10 * i = 61.70) : 
  p = 1.835 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_one_box_of_paper_clips_l2145_214567


namespace NUMINAMATH_GPT_ice_cream_flavors_l2145_214517

theorem ice_cream_flavors : (Nat.choose 8 3) = 56 := 
by {
    sorry
}

end NUMINAMATH_GPT_ice_cream_flavors_l2145_214517


namespace NUMINAMATH_GPT_least_possible_value_of_z_minus_x_l2145_214586

theorem least_possible_value_of_z_minus_x 
  (x y z : ℤ) 
  (h1 : x < y) 
  (h2 : y < z) 
  (h3 : y - x > 5) 
  (h4 : ∃ n : ℤ, x = 2 * n)
  (h5 : ∃ m : ℤ, y = 2 * m + 1) 
  (h6 : ∃ k : ℤ, z = 2 * k + 1) : 
  z - x = 9 := 
sorry

end NUMINAMATH_GPT_least_possible_value_of_z_minus_x_l2145_214586


namespace NUMINAMATH_GPT_number_of_perfect_square_factors_l2145_214584

theorem number_of_perfect_square_factors :
  let n := (2^14) * (3^9) * (5^20)
  ∃ (count : ℕ), 
  (∀ (a : ℕ) (h : a ∣ n), (∃ k, a = k^2) → true) →
  count = 440 :=
by
  sorry

end NUMINAMATH_GPT_number_of_perfect_square_factors_l2145_214584


namespace NUMINAMATH_GPT_stamps_initial_count_l2145_214598

theorem stamps_initial_count (total_stamps stamps_received initial_stamps : ℕ) 
  (h1 : total_stamps = 61)
  (h2 : stamps_received = 27)
  (h3 : initial_stamps = total_stamps - stamps_received) :
  initial_stamps = 34 :=
sorry

end NUMINAMATH_GPT_stamps_initial_count_l2145_214598


namespace NUMINAMATH_GPT_eggs_left_after_taking_l2145_214559

def eggs_in_box_initial : Nat := 47
def eggs_taken_by_Harry : Nat := 5
theorem eggs_left_after_taking : eggs_in_box_initial - eggs_taken_by_Harry = 42 := 
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_eggs_left_after_taking_l2145_214559


namespace NUMINAMATH_GPT_luke_total_score_l2145_214533

theorem luke_total_score (points_per_round : ℕ) (number_of_rounds : ℕ) (total_score : ℕ) : 
  points_per_round = 146 ∧ number_of_rounds = 157 ∧ total_score = points_per_round * number_of_rounds → 
  total_score = 22822 := by 
  sorry

end NUMINAMATH_GPT_luke_total_score_l2145_214533


namespace NUMINAMATH_GPT_polynomial_at_x_neg_four_l2145_214553

noncomputable def f (x : ℝ) : ℝ :=
  12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

theorem polynomial_at_x_neg_four : 
  f (-4) = 220 := by
  sorry

end NUMINAMATH_GPT_polynomial_at_x_neg_four_l2145_214553


namespace NUMINAMATH_GPT_problem_statement_l2145_214528

open Classical

variable (p q : Prop)

theorem problem_statement (h1 : p ∨ q) (h2 : ¬(p ∧ q)) (h3 : ¬ p) : (p = (5 + 2 = 6) ∧ q = (6 > 2)) :=
by
  have hp : p = False := by sorry
  have hq : q = True := by sorry
  exact ⟨by sorry, by sorry⟩

end NUMINAMATH_GPT_problem_statement_l2145_214528


namespace NUMINAMATH_GPT_all_positive_integers_are_nice_l2145_214532

def isNice (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : Fin k → ℕ), (∀ i, ∃ m : ℕ, a i = 2 ^ m) ∧ n = (Finset.univ.sum a) / k

theorem all_positive_integers_are_nice : ∀ n : ℕ, 0 < n → isNice n := sorry

end NUMINAMATH_GPT_all_positive_integers_are_nice_l2145_214532


namespace NUMINAMATH_GPT_number_of_n_divisible_by_prime_lt_20_l2145_214580

theorem number_of_n_divisible_by_prime_lt_20 (N : ℕ) : 
  (N = 69) :=
by
  sorry

end NUMINAMATH_GPT_number_of_n_divisible_by_prime_lt_20_l2145_214580


namespace NUMINAMATH_GPT_quadratic_solution_l2145_214525

theorem quadratic_solution :
  ∀ x : ℝ, (3 * x - 1) * (2 * x + 4) = 1 ↔ x = (-5 + Real.sqrt 55) / 6 ∨ x = (-5 - Real.sqrt 55) / 6 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l2145_214525


namespace NUMINAMATH_GPT_probability_at_most_one_A_B_selected_l2145_214563

def total_employees : ℕ := 36
def ratio_3_2_1 : (ℕ × ℕ × ℕ) := (3, 2, 1)
def sample_size : ℕ := 12
def youth_group_size : ℕ := 6
def total_combinations_youth : ℕ := Nat.choose 6 2
def event_complementary : ℕ := Nat.choose 2 2

theorem probability_at_most_one_A_B_selected :
  let prob := 1 - event_complementary / total_combinations_youth
  prob = (14 : ℚ) / 15 := sorry

end NUMINAMATH_GPT_probability_at_most_one_A_B_selected_l2145_214563


namespace NUMINAMATH_GPT_one_positive_real_solution_l2145_214552

noncomputable def f (x : ℝ) : ℝ := x^4 + 5 * x^3 + 10 * x^2 + 2023 * x - 2021

theorem one_positive_real_solution : 
  ∃! x : ℝ, 0 < x ∧ f x = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_one_positive_real_solution_l2145_214552


namespace NUMINAMATH_GPT_complex_division_evaluation_l2145_214562

open Complex

theorem complex_division_evaluation :
  (2 : ℂ) / (I * (3 - I)) = (1 / 5 : ℂ) - (3 / 5) * I :=
by
  sorry

end NUMINAMATH_GPT_complex_division_evaluation_l2145_214562


namespace NUMINAMATH_GPT_number_of_books_is_10_l2145_214548

def costPerBookBeforeDiscount : ℝ := 5
def discountPerBook : ℝ := 0.5
def totalPayment : ℝ := 45

theorem number_of_books_is_10 (n : ℕ) (h : (costPerBookBeforeDiscount - discountPerBook) * n = totalPayment) : n = 10 := by
  sorry

end NUMINAMATH_GPT_number_of_books_is_10_l2145_214548
