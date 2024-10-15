import Mathlib

namespace NUMINAMATH_GPT_triangles_hyperbola_parallel_l1385_138571

variable (a b c a1 b1 c1 : ℝ)

-- Defining the property that all vertices lie on the hyperbola y = 1/x
def on_hyperbola (x : ℝ) (y : ℝ) : Prop := y = 1 / x

-- Defining the parallelism condition for line segments
def parallel (slope1 slope2 : ℝ) : Prop := slope1 = slope2

theorem triangles_hyperbola_parallel
  (H1A : on_hyperbola a (1 / a))
  (H1B : on_hyperbola b (1 / b))
  (H1C : on_hyperbola c (1 / c))
  (H2A : on_hyperbola a1 (1 / a1))
  (H2B : on_hyperbola b1 (1 / b1))
  (H2C : on_hyperbola c1 (1 / c1))
  (H_AB_parallel_A1B1 : parallel ((b - a) / (a * b * (a - b))) ((b1 - a1) / (a1 * b1 * (a1 - b1))))
  (H_BC_parallel_B1C1 : parallel ((c - b) / (b * c * (b - c))) ((c1 - b1) / (b1 * c1 * (b1 - c1)))) :
  parallel ((c1 - a) / (a * c1 * (a - c1))) ((c - a1) / (a1 * c * (a1 - c))) :=
sorry

end NUMINAMATH_GPT_triangles_hyperbola_parallel_l1385_138571


namespace NUMINAMATH_GPT_circle_occupies_62_8_percent_l1385_138533

noncomputable def largestCirclePercentage (length : ℝ) (width : ℝ) : ℝ :=
  let radius := width / 2
  let circle_area := Real.pi * radius^2
  let rectangle_area := length * width
  (circle_area / rectangle_area) * 100

theorem circle_occupies_62_8_percent : largestCirclePercentage 5 4 = 62.8 := 
by 
  /- Sorry, skipping the proof -/
  sorry

end NUMINAMATH_GPT_circle_occupies_62_8_percent_l1385_138533


namespace NUMINAMATH_GPT_sequence_formula_l1385_138521

theorem sequence_formula (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (h : ∀ n, S_n n = 3 + 2 * a_n n) :
  ∀ n, a_n n = -3 * 2^(n - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l1385_138521


namespace NUMINAMATH_GPT_average_of_remaining_numbers_l1385_138549

theorem average_of_remaining_numbers 
  (numbers : List ℝ) 
  (h_len : numbers.length = 50) 
  (h_avg : (numbers.sum / 50) = 20)
  (h_disc : 45 ∈ numbers ∧ 55 ∈ numbers) 
  (h_count_45_55 : numbers.count 45 = 1 ∧ numbers.count 55 = 1) :
  (numbers.sum - 45 - 55) / (50 - 2) = 18.75 :=
by
  sorry

end NUMINAMATH_GPT_average_of_remaining_numbers_l1385_138549


namespace NUMINAMATH_GPT_range_of_alpha_minus_beta_l1385_138544

variable (α β : ℝ)

theorem range_of_alpha_minus_beta (h1 : -90 < α) (h2 : α < β) (h3 : β < 90) : -180 < α - β ∧ α - β < 0 := 
by
  sorry

end NUMINAMATH_GPT_range_of_alpha_minus_beta_l1385_138544


namespace NUMINAMATH_GPT_complement_union_eq_l1385_138566

open Set

variable (U A B : Set ℤ)

noncomputable def universal_set : Set ℤ := {-2, -1, 0, 1, 2, 3}

noncomputable def setA : Set ℤ := {-1, 0, 3}

noncomputable def setB : Set ℤ := {1, 3}

theorem complement_union_eq :
  A ∪ B = {-1, 0, 1, 3} →
  U = universal_set →
  A = setA →
  B = setB →
  (U \ (A ∪ B)) = {-2, 2} := by
  intros
  sorry

end NUMINAMATH_GPT_complement_union_eq_l1385_138566


namespace NUMINAMATH_GPT_team_points_l1385_138593

theorem team_points (wins losses ties : ℕ) (points_per_win points_per_loss points_per_tie : ℕ) :
  wins = 9 → losses = 3 → ties = 4 → points_per_win = 2 → points_per_loss = 0 → points_per_tie = 1 →
  (points_per_win * wins + points_per_loss * losses + points_per_tie * ties = 22) :=
by
  intro h_wins h_losses h_ties h_points_per_win h_points_per_loss h_points_per_tie
  sorry

end NUMINAMATH_GPT_team_points_l1385_138593


namespace NUMINAMATH_GPT_object_reaches_max_height_at_three_l1385_138562

theorem object_reaches_max_height_at_three :
  ∀ (h : ℝ) (t : ℝ), h = -15 * (t - 3)^2 + 150 → t = 3 :=
by
  sorry

end NUMINAMATH_GPT_object_reaches_max_height_at_three_l1385_138562


namespace NUMINAMATH_GPT_task_completion_time_l1385_138543

theorem task_completion_time :
  let rate_A := 1 / 10
  let rate_B := 1 / 15
  let rate_C := 1 / 15
  let combined_rate := rate_A + rate_B + rate_C
  let working_days_A := 2
  let working_days_B := 1
  let rest_day_A := 1
  let rest_days_B := 2
  let work_done_A := rate_A * working_days_A
  let work_done_B := rate_B * working_days_B
  let work_done_C := rate_C * (working_days_A + rest_day_A)
  let work_done := work_done_A + work_done_B + work_done_C
  let remaining_work := 1 - work_done
  let total_days := (work_done / combined_rate) + rest_day_A + rest_days_B
  total_days = 4 + 1 / 7 := by sorry

end NUMINAMATH_GPT_task_completion_time_l1385_138543


namespace NUMINAMATH_GPT_inequality_problem_l1385_138530

-- Define the problem conditions and goal
theorem inequality_problem (x y : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) : 
  x + y + 1 / (x * y) ≤ 1 / x + 1 / y + x * y := 
sorry

end NUMINAMATH_GPT_inequality_problem_l1385_138530


namespace NUMINAMATH_GPT_probability_gather_info_both_workshops_l1385_138587

theorem probability_gather_info_both_workshops :
  ∃ (p : ℚ), p = 56 / 62 :=
by
  sorry

end NUMINAMATH_GPT_probability_gather_info_both_workshops_l1385_138587


namespace NUMINAMATH_GPT_flowchart_output_is_minus_nine_l1385_138589

-- Given initial state and conditions
def initialState : ℤ := 0

-- Hypothetical function representing the sequence of operations in the flowchart
-- (hiding the exact operations since they are speculative)
noncomputable def flowchartOperations (S : ℤ) : ℤ := S - 9  -- Assuming this operation represents the described flowchart

-- The proof problem
theorem flowchart_output_is_minus_nine : flowchartOperations initialState = -9 :=
by
  sorry

end NUMINAMATH_GPT_flowchart_output_is_minus_nine_l1385_138589


namespace NUMINAMATH_GPT_div_30_div_510_div_66_div_large_l1385_138523

theorem div_30 (a : ℤ) : 30 ∣ (a^5 - a) := 
  sorry  

theorem div_510 (a : ℤ) : 510 ∣ (a^17 - a) := 
  sorry

theorem div_66 (a : ℤ) : 66 ∣ (a^11 - a) := 
  sorry

theorem div_large (a : ℤ) : (2 * 3 * 5 * 7 * 13 * 19 * 37 * 73) ∣ (a^73 - a) := 
  sorry  

end NUMINAMATH_GPT_div_30_div_510_div_66_div_large_l1385_138523


namespace NUMINAMATH_GPT_wall_thickness_is_correct_l1385_138528

-- Define the dimensions of the brick.
def brick_length : ℝ := 80
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the number of required bricks.
def num_bricks : ℝ := 2000

-- Define the dimensions of the wall.
def wall_length : ℝ := 800
def wall_height : ℝ := 600

-- The volume of one brick.
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- The volume of the wall.
def wall_volume (T : ℝ) : ℝ := wall_length * wall_height * T

-- The thickness of the wall to be proved.
theorem wall_thickness_is_correct (T_wall : ℝ) (h : num_bricks * brick_volume = wall_volume T_wall) : 
  T_wall = 22.5 :=
sorry

end NUMINAMATH_GPT_wall_thickness_is_correct_l1385_138528


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1385_138552

variable {R : Type} [LinearOrderedField R]

theorem solution_set_of_inequality (f : R -> R) (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x y, 0 < x ∧ x < y → f x < f y) (h3 : f 1 = 0) :
  { x : R | (f x - f (-x)) / x < 0 } = { x : R | (-1 < x ∧ x < 0) ∨ (0 < x ∧ x < 1) } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1385_138552


namespace NUMINAMATH_GPT_part_a_part_b_l1385_138594

section
-- Definitions based on the conditions
variable (n : ℕ)  -- Variable n representing the number of cities

-- Given a condition function T_n that returns an integer (number of ways to build roads)
def T_n (n : ℕ) : ℕ := sorry  -- Definition placeholder for T_n function

-- Part (a): For all odd n, T_n(n) is divisible by n
theorem part_a (hn : n % 2 = 1) : T_n n % n = 0 := sorry

-- Part (b): For all even n, T_n(n) is divisible by n / 2
theorem part_b (hn : n % 2 = 0) : T_n n % (n / 2) = 0 := sorry

end

end NUMINAMATH_GPT_part_a_part_b_l1385_138594


namespace NUMINAMATH_GPT_bobby_initial_candy_count_l1385_138512

theorem bobby_initial_candy_count (C : ℕ) (h : C + 4 + 14 = 51) : C = 33 :=
by
  sorry

end NUMINAMATH_GPT_bobby_initial_candy_count_l1385_138512


namespace NUMINAMATH_GPT_students_present_l1385_138599

theorem students_present (total_students : ℕ) (absent_percent : ℝ) (total_absent : ℝ) (total_present : ℝ) :
  total_students = 50 → absent_percent = 0.12 → total_absent = total_students * absent_percent →
  total_present = total_students - total_absent →
  total_present = 44 :=
by
  intros _ _ _ _; sorry

end NUMINAMATH_GPT_students_present_l1385_138599


namespace NUMINAMATH_GPT_problem_statement_l1385_138545

theorem problem_statement
  (a b c d : ℕ)
  (h1 : (b + c + d) / 3 + 2 * a = 54)
  (h2 : (a + c + d) / 3 + 2 * b = 50)
  (h3 : (a + b + d) / 3 + 2 * c = 42)
  (h4 : (a + b + c) / 3 + 2 * d = 30) :
  a = 17 ∨ b = 17 ∨ c = 17 ∨ d = 17 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1385_138545


namespace NUMINAMATH_GPT_probability_digits_different_l1385_138584

theorem probability_digits_different : 
  let total_numbers := 490
  let same_digits_numbers := 13
  let different_digits_numbers := total_numbers - same_digits_numbers 
  let probability := different_digits_numbers / total_numbers 
  probability = 477 / 490 :=
by
  sorry

end NUMINAMATH_GPT_probability_digits_different_l1385_138584


namespace NUMINAMATH_GPT_jigi_scored_55_percent_l1385_138534

noncomputable def jigi_percentage (max_score : ℕ) (avg_score : ℕ) (gibi_pct mike_pct lizzy_pct : ℕ) : ℕ := sorry

theorem jigi_scored_55_percent :
  jigi_percentage 700 490 59 99 67 = 55 :=
sorry

end NUMINAMATH_GPT_jigi_scored_55_percent_l1385_138534


namespace NUMINAMATH_GPT_tic_tac_toe_tie_fraction_l1385_138561

theorem tic_tac_toe_tie_fraction :
  let amys_win : ℚ := 5 / 12
  let lilys_win : ℚ := 1 / 4
  1 - (amys_win + lilys_win) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tic_tac_toe_tie_fraction_l1385_138561


namespace NUMINAMATH_GPT_bucket_full_weight_l1385_138578

theorem bucket_full_weight (x y c d : ℝ)
  (h1 : x + 3 / 4 * y = c)
  (h2 : x + 1 / 3 * y = d) :
  x + y = (8 / 5) * c - (7 / 5) * d :=
by
  sorry

end NUMINAMATH_GPT_bucket_full_weight_l1385_138578


namespace NUMINAMATH_GPT_gcd_2024_1728_l1385_138527

theorem gcd_2024_1728 : Int.gcd 2024 1728 = 8 := 
by
  sorry

end NUMINAMATH_GPT_gcd_2024_1728_l1385_138527


namespace NUMINAMATH_GPT_basketball_game_total_points_l1385_138565

theorem basketball_game_total_points :
  ∃ (a d b: ℕ) (r: ℝ), 
      a = b + 2 ∧     -- Eagles lead by 2 points at the end of the first quarter
      (a + d < 100) ∧ -- Points scored by Eagles in each quarter form an increasing arithmetic sequence
      (b * r < 100) ∧ -- Points scored by Lions in each quarter form an increasing geometric sequence
      (a + (a + d) + (a + 2 * d)) = b * (1 + r + r^2) ∧ -- Aggregate score tied at the end of the third quarter
      (a + (a + d) + (a + 2 * d) + (a + 3 * d) + b * (1 + r + r^2 + r^3) = 144) -- Total points scored by both teams 
   :=
sorry

end NUMINAMATH_GPT_basketball_game_total_points_l1385_138565


namespace NUMINAMATH_GPT_total_parents_in_auditorium_l1385_138547

-- Define the conditions.
def girls : Nat := 6
def boys : Nat := 8
def total_kids : Nat := girls + boys
def parents_per_kid : Nat := 2
def total_parents : Nat := total_kids * parents_per_kid

-- The statement to prove.
theorem total_parents_in_auditorium : total_parents = 28 := by
  sorry

end NUMINAMATH_GPT_total_parents_in_auditorium_l1385_138547


namespace NUMINAMATH_GPT_three_number_product_l1385_138526

theorem three_number_product
  (x y z : ℝ)
  (h1 : x + y = 18)
  (h2 : x ^ 2 + y ^ 2 = 220)
  (h3 : z = x - y) :
  x * y * z = 104 * Real.sqrt 29 :=
sorry

end NUMINAMATH_GPT_three_number_product_l1385_138526


namespace NUMINAMATH_GPT_find_a_b_l1385_138550

theorem find_a_b (a b : ℤ) (h : ({a, 0, -1} : Set ℤ) = {4, b, 0}) : a = 4 ∧ b = -1 := by
  sorry

end NUMINAMATH_GPT_find_a_b_l1385_138550


namespace NUMINAMATH_GPT_log9_6_eq_mn_over_2_l1385_138514

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log9_6_eq_mn_over_2
  (m n : ℝ)
  (h1 : log_base 7 4 = m)
  (h2 : log_base 4 6 = n) : 
  log_base 9 6 = (m * n) / 2 := by
  sorry

end NUMINAMATH_GPT_log9_6_eq_mn_over_2_l1385_138514


namespace NUMINAMATH_GPT_intersection_M_complement_N_l1385_138520

open Set Real

def M : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}
def N : Set ℝ := {x | (Real.log 2) ^ (1 - x) < 1}
def complement_N := {x : ℝ | x ≥ 1}

theorem intersection_M_complement_N :
  M ∩ complement_N = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_complement_N_l1385_138520


namespace NUMINAMATH_GPT_negation_of_at_most_four_is_at_least_five_l1385_138551

theorem negation_of_at_most_four_is_at_least_five :
  (∀ n : ℕ, n ≤ 4) ↔ (∃ n : ℕ, n ≥ 5) := 
sorry

end NUMINAMATH_GPT_negation_of_at_most_four_is_at_least_five_l1385_138551


namespace NUMINAMATH_GPT_min_students_in_group_l1385_138583

theorem min_students_in_group 
  (g1 g2 : ℕ) 
  (n1 n2 e1 e2 f1 f2 : ℕ)
  (H_equal_groups : g1 = g2)
  (H_both_languages_g1 : n1 = 5)
  (H_both_languages_g2 : n2 = 5)
  (H_french_students : f1 * 3 = f2)
  (H_english_students : e1 = 4 * e2)
  (H_total_g1 : g1 = f1 + e1 - n1)
  (H_total_g2 : g2 = f2 + e2 - n2) 
: g1 = 28 :=
sorry

end NUMINAMATH_GPT_min_students_in_group_l1385_138583


namespace NUMINAMATH_GPT_find_a_l1385_138537

noncomputable def A (a : ℝ) : Set ℝ := {2^a, 3}
def B : Set ℝ := {2, 3}
def C : Set ℝ := {1, 2, 3}

theorem find_a (a : ℝ) (h : A a ∪ B = C) : a = 0 :=
sorry

end NUMINAMATH_GPT_find_a_l1385_138537


namespace NUMINAMATH_GPT_suitable_high_jump_athlete_l1385_138516

structure Athlete where
  average : ℕ
  variance : ℝ

def A : Athlete := ⟨169, 6.0⟩
def B : Athlete := ⟨168, 17.3⟩
def C : Athlete := ⟨169, 5.0⟩
def D : Athlete := ⟨168, 19.5⟩

def isSuitableCandidate (athlete: Athlete) (average_threshold: ℕ) : Prop :=
  athlete.average = average_threshold

theorem suitable_high_jump_athlete : isSuitableCandidate C 169 ∧
  (∀ a, isSuitableCandidate a 169 → a.variance ≥ C.variance) := by
  sorry

end NUMINAMATH_GPT_suitable_high_jump_athlete_l1385_138516


namespace NUMINAMATH_GPT_max_area_of_triangle_l1385_138590

theorem max_area_of_triangle (a c : ℝ)
    (h1 : a^2 + c^2 = 16 + a * c) : 
    ∃ s : ℝ, s = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_max_area_of_triangle_l1385_138590


namespace NUMINAMATH_GPT_mean_temperature_is_88_75_l1385_138569

def temperatures : List ℕ := [85, 84, 85, 88, 91, 93, 94, 90]

theorem mean_temperature_is_88_75 : (List.sum temperatures : ℚ) / temperatures.length = 88.75 := by
  sorry

end NUMINAMATH_GPT_mean_temperature_is_88_75_l1385_138569


namespace NUMINAMATH_GPT_measure_of_angle_A_values_of_b_and_c_l1385_138567

variable (a b c : ℝ) (A : ℝ)

-- Declare the conditions as hypotheses
def condition1 (a b c : ℝ) := a^2 - c^2 = b^2 - b * c
def condition2 (a : ℝ) := a = 2
def condition3 (b c : ℝ) := b + c = 4

-- Proof that A = 60 degrees when the conditions are satisfied
theorem measure_of_angle_A (h : condition1 a b c) : A = 60 := by
  sorry

-- Proof that b and c are 2 when given conditions are satisfied
theorem values_of_b_and_c (h1 : condition1 2 b c) (h2 : condition3 b c) : b = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_GPT_measure_of_angle_A_values_of_b_and_c_l1385_138567


namespace NUMINAMATH_GPT_percentage_increase_l1385_138500

def old_price : ℝ := 300
def new_price : ℝ := 330

theorem percentage_increase : ((new_price - old_price) / old_price) * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1385_138500


namespace NUMINAMATH_GPT_complement_intersection_l1385_138597

open Set

variable (U A B : Set ℕ)

theorem complement_intersection (U : Set ℕ) (A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {1, 3, 6}) (hB : B = {1, 2}) :
  ((U \ A) ∩ B) = {2} :=
by
  rw [hU, hA, hB]
  sorry

end NUMINAMATH_GPT_complement_intersection_l1385_138597


namespace NUMINAMATH_GPT_part1_solution_set_part2_value_of_t_l1385_138573

open Real

def f (t x : ℝ) : ℝ := x^2 - (t + 1) * x + t

-- Statement for the equivalent proof problem
theorem part1_solution_set (x : ℝ) : 
  (t = 3 → f 3 x > 0 ↔ (x < 1) ∨ (x > 3)) :=
by
  sorry

theorem part2_value_of_t (t : ℝ) :
  (∀ x : ℝ, f t x ≥ 0) → t = 1 :=
by
  sorry

end NUMINAMATH_GPT_part1_solution_set_part2_value_of_t_l1385_138573


namespace NUMINAMATH_GPT_xn_plus_inv_xn_is_integer_l1385_138529

theorem xn_plus_inv_xn_is_integer (x : ℝ) (hx : x ≠ 0) (k : ℤ) (h : x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
by sorry

end NUMINAMATH_GPT_xn_plus_inv_xn_is_integer_l1385_138529


namespace NUMINAMATH_GPT_original_cost_l1385_138557

theorem original_cost (C : ℝ) (h : 550 = 1.35 * C) : C = 550 / 1.35 :=
by
  sorry

end NUMINAMATH_GPT_original_cost_l1385_138557


namespace NUMINAMATH_GPT_lewis_weekly_earning_l1385_138582

def total_amount_earned : ℕ := 178
def number_of_weeks : ℕ := 89
def weekly_earning (total : ℕ) (weeks : ℕ) : ℕ := total / weeks

theorem lewis_weekly_earning : weekly_earning total_amount_earned number_of_weeks = 2 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_lewis_weekly_earning_l1385_138582


namespace NUMINAMATH_GPT_range_of_a_l1385_138522

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + a * x + 1 < 0) → -2 ≤ a ∧ a ≤ 2 := sorry

end NUMINAMATH_GPT_range_of_a_l1385_138522


namespace NUMINAMATH_GPT_functional_equation_solution_l1385_138585

noncomputable def f : ℕ → ℕ := sorry

theorem functional_equation_solution (f : ℕ → ℕ)
    (h : ∀ n : ℕ, f (f (f n)) + f (f n) + f n = 3 * n) :
    ∀ n : ℕ, f n = n := sorry

end NUMINAMATH_GPT_functional_equation_solution_l1385_138585


namespace NUMINAMATH_GPT_tom_age_l1385_138572

theorem tom_age (c : ℕ) (h1 : 2 * c - 1 = tom) (h2 : c + 3 = dave) (h3 : c + (2 * c - 1) + (c + 3) = 30) : tom = 13 :=
  sorry

end NUMINAMATH_GPT_tom_age_l1385_138572


namespace NUMINAMATH_GPT_avg_rate_of_change_interval_1_2_l1385_138577

def f (x : ℝ) : ℝ := 2 * x + 1

theorem avg_rate_of_change_interval_1_2 : 
  (f 2 - f 1) / (2 - 1) = 2 :=
by sorry

end NUMINAMATH_GPT_avg_rate_of_change_interval_1_2_l1385_138577


namespace NUMINAMATH_GPT_integer_solutions_system_l1385_138568

theorem integer_solutions_system :
  {x : ℤ | (4 * (1 + x) / 3 - 1 ≤ (5 + x) / 2) ∧ (x - 5 ≤ (3 * (3 * x - 2)) / 2)} = {0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_system_l1385_138568


namespace NUMINAMATH_GPT_mean_of_set_median_is_128_l1385_138501

theorem mean_of_set_median_is_128 (m : ℝ) (h : m + 7 = 12) : 
  (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5 = 12.8 := by
  sorry

end NUMINAMATH_GPT_mean_of_set_median_is_128_l1385_138501


namespace NUMINAMATH_GPT_B_joined_amount_l1385_138510

theorem B_joined_amount (T : ℝ)
  (A_investment : ℝ := 45000)
  (B_time : ℝ := 2)
  (profit_ratio : ℝ := 2 / 1)
  (investment_ratio_rule : (A_investment * T) / (B_investment_amount * B_time) = profit_ratio) :
  B_investment_amount = 22500 :=
by
  sorry

end NUMINAMATH_GPT_B_joined_amount_l1385_138510


namespace NUMINAMATH_GPT_total_points_scored_l1385_138598

-- Define the variables
def games : ℕ := 10
def points_per_game : ℕ := 12

-- Formulate the proposition to prove
theorem total_points_scored : games * points_per_game = 120 :=
by
  sorry

end NUMINAMATH_GPT_total_points_scored_l1385_138598


namespace NUMINAMATH_GPT_fixed_point_for_any_k_l1385_138507

-- Define the function f representing our quadratic equation
def f (k : ℝ) (x : ℝ) : ℝ :=
  8 * x^2 + 3 * k * x - 5 * k
  
-- The statement representing our proof problem
theorem fixed_point_for_any_k :
  ∀ (a b : ℝ), (∀ (k : ℝ), f k a = b) → (a, b) = (5, 200) :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_for_any_k_l1385_138507


namespace NUMINAMATH_GPT_find_y_l1385_138540

theorem find_y (y : ℕ) (h1 : y % 6 = 5) (h2 : y % 7 = 6) (h3 : y % 8 = 7) : y = 167 := 
by
  sorry  -- Proof is omitted

end NUMINAMATH_GPT_find_y_l1385_138540


namespace NUMINAMATH_GPT_dogs_in_kennel_l1385_138506

variable (C D : ℕ)

-- definition of the ratio condition 
def ratio_condition : Prop :=
  C * 4 = 3 * D

-- definition of the difference condition
def difference_condition : Prop :=
  C = D - 8

theorem dogs_in_kennel (h1 : ratio_condition C D) (h2 : difference_condition C D) : D = 32 :=
by 
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_dogs_in_kennel_l1385_138506


namespace NUMINAMATH_GPT_instantaneous_velocity_at_3_l1385_138539

-- Define the displacement function
def displacement (t : ℝ) : ℝ := t^2 - t

-- State the main theorem that we need to prove
theorem instantaneous_velocity_at_3 : (deriv displacement 3 = 5) := by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_3_l1385_138539


namespace NUMINAMATH_GPT_initial_liquid_X_percentage_is_30_l1385_138517

variable (initial_liquid_X_percentage : ℝ)

theorem initial_liquid_X_percentage_is_30
  (solution_total_weight : ℝ := 8)
  (initial_water_percentage : ℝ := 70)
  (evaporated_water_weight : ℝ := 3)
  (added_solution_weight : ℝ := 3)
  (new_liquid_X_percentage : ℝ := 41.25)
  (total_new_solution_weight : ℝ := 8)
  :
  initial_liquid_X_percentage = 30 :=
sorry

end NUMINAMATH_GPT_initial_liquid_X_percentage_is_30_l1385_138517


namespace NUMINAMATH_GPT_triangle_area_l1385_138505

theorem triangle_area : 
  ∀ (x y: ℝ), (x / 5 + y / 2 = 1) → (x = 5) ∨ (y = 2) → ∃ A : ℝ, A = 5 :=
by
  intros x y h1 h2
  -- Definitions based on the problem conditions
  have hx : x = 5 := sorry
  have hy : y = 2 := sorry
  have base := 5
  have height := 2
  have area := 1 / 2 * base * height
  use area
  sorry

end NUMINAMATH_GPT_triangle_area_l1385_138505


namespace NUMINAMATH_GPT_range_of_m_plus_n_l1385_138554

noncomputable def f (m n x : ℝ) : ℝ := m * 2^x + x^2 + n * x

theorem range_of_m_plus_n (m n : ℝ) :
  (∃ x : ℝ, f m n x = 0 ∧ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_plus_n_l1385_138554


namespace NUMINAMATH_GPT_total_pages_read_l1385_138588

theorem total_pages_read (days : ℕ)
  (deshaun_books deshaun_pages_per_book lilly_percent ben_extra eva_factor sam_pages_per_day : ℕ)
  (lilly_percent_correct : lilly_percent = 75)
  (ben_extra_correct : ben_extra = 25)
  (eva_factor_correct : eva_factor = 2)
  (total_break_days : days = 80)
  (deshaun_books_correct : deshaun_books = 60)
  (deshaun_pages_per_book_correct : deshaun_pages_per_book = 320)
  (sam_pages_per_day_correct : sam_pages_per_day = 150) :
  deshaun_books * deshaun_pages_per_book +
  (lilly_percent * deshaun_books * deshaun_pages_per_book / 100) +
  (deshaun_books * (100 + ben_extra) / 100) * 280 +
  (eva_factor * (deshaun_books * (100 + ben_extra) / 100 * 280)) +
  (sam_pages_per_day * days) = 108450 := 
sorry

end NUMINAMATH_GPT_total_pages_read_l1385_138588


namespace NUMINAMATH_GPT_sqrt_nine_factorial_over_72_eq_l1385_138570

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_nine_factorial_over_72_eq : 
  Real.sqrt ((factorial 9) / 72) = 12 * Real.sqrt 35 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_nine_factorial_over_72_eq_l1385_138570


namespace NUMINAMATH_GPT_count_multiples_of_7_not_14_l1385_138574

theorem count_multiples_of_7_not_14 (n : ℕ) : (n < 500 ∧ n % 7 = 0 ∧ n % 14 ≠ 0) → ∃ (k : ℕ), k = 36 :=
by
  sorry

end NUMINAMATH_GPT_count_multiples_of_7_not_14_l1385_138574


namespace NUMINAMATH_GPT_total_children_on_playground_l1385_138513

theorem total_children_on_playground
  (boys : ℕ) (girls : ℕ)
  (h_boys : boys = 44) (h_girls : girls = 53) :
  boys + girls = 97 :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_children_on_playground_l1385_138513


namespace NUMINAMATH_GPT_af2_plus_bfg_plus_cg2_geq_0_l1385_138546

theorem af2_plus_bfg_plus_cg2_geq_0 (a b c : ℝ) (f g : ℝ) :
  (a * f^2 + b * f * g + c * g^2 ≥ 0) ↔ (a ≥ 0 ∧ c ≥ 0 ∧ 4 * a * c ≥ b^2) := 
sorry

end NUMINAMATH_GPT_af2_plus_bfg_plus_cg2_geq_0_l1385_138546


namespace NUMINAMATH_GPT_solve_equation_l1385_138542

theorem solve_equation (x : ℝ) (h : x ≠ 4) :
  (x - 3) / (4 - x) - 1 = 1 / (x - 4) → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1385_138542


namespace NUMINAMATH_GPT_max_f_l1385_138555

open Real

noncomputable def f (x : ℝ) : ℝ := 3 + log x + 4 / log x

theorem max_f (h : 0 < x ∧ x < 1) : f x ≤ -1 :=
sorry

end NUMINAMATH_GPT_max_f_l1385_138555


namespace NUMINAMATH_GPT_gcd_228_1995_l1385_138538

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_GPT_gcd_228_1995_l1385_138538


namespace NUMINAMATH_GPT_x_can_be_any_sign_l1385_138532

theorem x_can_be_any_sign
  (x y p q : ℝ)
  (h1 : abs (x / y) < abs (p) / q^2)
  (h2 : y ≠ 0) (h3 : q ≠ 0) :
  ∃ (x' : ℝ), True :=
by
  sorry

end NUMINAMATH_GPT_x_can_be_any_sign_l1385_138532


namespace NUMINAMATH_GPT_solve_equation_in_natural_numbers_l1385_138564

theorem solve_equation_in_natural_numbers (x y : ℕ) :
  2 * y^2 - x * y - x^2 + 2 * y + 7 * x - 84 = 0 ↔ (x = 1 ∧ y = 6) ∨ (x = 14 ∧ y = 13) := 
sorry

end NUMINAMATH_GPT_solve_equation_in_natural_numbers_l1385_138564


namespace NUMINAMATH_GPT_total_onions_l1385_138508

theorem total_onions (sara sally fred amy matthew : Nat) 
  (hs : sara = 40) (hl : sally = 55) 
  (hf : fred = 90) (ha : amy = 25) 
  (hm : matthew = 75) :
  sara + sally + fred + amy + matthew = 285 := 
by
  sorry

end NUMINAMATH_GPT_total_onions_l1385_138508


namespace NUMINAMATH_GPT_mnmn_not_cube_in_base_10_and_find_smallest_base_b_l1385_138579

theorem mnmn_not_cube_in_base_10_and_find_smallest_base_b 
    (m n : ℕ) (h1 : m * 10^3 + n * 10^2 + m * 10 + n < 10000) :
    ¬ (∃ k : ℕ, (m * 10^3 + n * 10^2 + m * 10 + n) = k^3) 
    ∧ ∃ b : ℕ, b > 1 ∧ (∃ k : ℕ, (m * b^3 + n * b^2 + m * b + n = k^3)) :=
by sorry

end NUMINAMATH_GPT_mnmn_not_cube_in_base_10_and_find_smallest_base_b_l1385_138579


namespace NUMINAMATH_GPT_discount_price_l1385_138535

theorem discount_price (P P_d : ℝ) 
  (h1 : P_d = 0.85 * P) 
  (P_final : ℝ) 
  (h2 : P_final = 1.25 * P_d) 
  (h3 : P - P_final = 5.25) :
  P_d = 71.4 :=
by
  sorry

end NUMINAMATH_GPT_discount_price_l1385_138535


namespace NUMINAMATH_GPT_find_a_l1385_138511

theorem find_a (a : ℤ) (A : Set ℤ) (B : Set ℤ) :
  A = {-2, 3 * a - 1, a^2 - 3} ∧
  B = {a - 2, a - 1, a + 1} ∧
  A ∩ B = {-2} → a = -3 :=
by
  intro H
  sorry

end NUMINAMATH_GPT_find_a_l1385_138511


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l1385_138553

noncomputable def a : ℚ := 5 / 3
noncomputable def r : ℚ := -1 / 2

theorem infinite_geometric_series_sum : 
  ∑' (n : ℕ), a * r^n = 10 / 9 := 
by sorry

end NUMINAMATH_GPT_infinite_geometric_series_sum_l1385_138553


namespace NUMINAMATH_GPT_mod_calculation_l1385_138559

theorem mod_calculation :
  (3 * 43 + 6 * 37) % 60 = 51 :=
by
  sorry

end NUMINAMATH_GPT_mod_calculation_l1385_138559


namespace NUMINAMATH_GPT_number_of_members_l1385_138560

theorem number_of_members (n : ℕ) (h : n * n = 2025) : n = 45 :=
sorry

end NUMINAMATH_GPT_number_of_members_l1385_138560


namespace NUMINAMATH_GPT_river_joe_collected_money_l1385_138518

theorem river_joe_collected_money :
  let price_catfish : ℤ := 600 -- in cents to avoid floating point issues
  let price_shrimp : ℤ := 350 -- in cents to avoid floating point issues
  let total_orders : ℤ := 26
  let shrimp_orders : ℤ := 9
  let catfish_orders : ℤ := total_orders - shrimp_orders
  let total_catfish_sales : ℤ := catfish_orders * price_catfish
  let total_shrimp_sales : ℤ := shrimp_orders * price_shrimp
  let total_money_collected : ℤ := total_catfish_sales + total_shrimp_sales
  total_money_collected = 13350 := -- in cents, so $133.50 is 13350 cents
by
  sorry

end NUMINAMATH_GPT_river_joe_collected_money_l1385_138518


namespace NUMINAMATH_GPT_find_value_l1385_138548

theorem find_value (x : ℤ) (h : 3 * x - 45 = 159) : (x + 32) * 12 = 1200 :=
by
  sorry

end NUMINAMATH_GPT_find_value_l1385_138548


namespace NUMINAMATH_GPT_eq_abc_gcd_l1385_138524

theorem eq_abc_gcd
  (a b c d : ℕ)
  (h1 : a^a * b^(a + b) = c^c * d^(c + d))
  (h2 : Nat.gcd a b = 1)
  (h3 : Nat.gcd c d = 1) : 
  a = c ∧ b = d := 
sorry

end NUMINAMATH_GPT_eq_abc_gcd_l1385_138524


namespace NUMINAMATH_GPT_maddie_episodes_friday_l1385_138541

theorem maddie_episodes_friday :
  let total_episodes : ℕ := 8
  let episode_duration : ℕ := 44
  let monday_time : ℕ := 138
  let thursday_time : ℕ := 21
  let weekend_time : ℕ := 105
  let total_time : ℕ := total_episodes * episode_duration
  let non_friday_time : ℕ := monday_time + thursday_time + weekend_time
  let friday_time : ℕ := total_time - non_friday_time
  let friday_episodes : ℕ := friday_time / episode_duration
  friday_episodes = 2 :=
by
  sorry

end NUMINAMATH_GPT_maddie_episodes_friday_l1385_138541


namespace NUMINAMATH_GPT_mika_stickers_l1385_138556

def s1 : ℝ := 20.5
def s2 : ℝ := 26.3
def s3 : ℝ := 19.75
def s4 : ℝ := 6.25
def s5 : ℝ := 57.65
def s6 : ℝ := 15.8

theorem mika_stickers 
  (M : ℝ)
  (hM : M = s1 + s2 + s3 + s4 + s5 + s6) 
  : M = 146.25 :=
sorry

end NUMINAMATH_GPT_mika_stickers_l1385_138556


namespace NUMINAMATH_GPT_possible_value_of_a_l1385_138531

theorem possible_value_of_a (a : ℕ) : (5 + 8 > a ∧ a > 3) → (a = 9 → True) :=
by
  intros h ha
  sorry

end NUMINAMATH_GPT_possible_value_of_a_l1385_138531


namespace NUMINAMATH_GPT_max_side_length_l1385_138596

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end NUMINAMATH_GPT_max_side_length_l1385_138596


namespace NUMINAMATH_GPT_problem1_problem2_l1385_138503

noncomputable def op (a b : ℝ) := 2 * a - (3 / 2) * (a + b)

theorem problem1 (x : ℝ) (h : op x 4 = 0) : x = 12 :=
by sorry

theorem problem2 (x m : ℝ) (h : op x m = op (-2) (x + 4)) (hnn : x ≥ 0) : m ≥ 14 / 3 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1385_138503


namespace NUMINAMATH_GPT_system_of_equations_solutions_l1385_138504

theorem system_of_equations_solutions :
  ∃ (sol : Finset (ℝ × ℝ)), sol.card = 3 ∧
    (∀ (x y : ℝ), (x, y) ∈ sol ↔ (x + 3 * y = 3 ∧ abs (abs x - abs y) = 1)) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solutions_l1385_138504


namespace NUMINAMATH_GPT_rem_fraction_l1385_138509

theorem rem_fraction : 
  let rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋;
  rem (5/7) (-3/4) = -1/28 := 
by
  sorry

end NUMINAMATH_GPT_rem_fraction_l1385_138509


namespace NUMINAMATH_GPT_solve_eq_l1385_138592

theorem solve_eq : ∀ x : ℂ, (x^3 + 3*x^2 + 4*x + 6) / (x + 5) = x^2 + 10 ↔
  x = (-3 + Complex.I * Real.sqrt 79) / 2 ∨ x = (-3 - Complex.I * Real.sqrt 79) / 2 :=
by 
  intro x
  sorry

end NUMINAMATH_GPT_solve_eq_l1385_138592


namespace NUMINAMATH_GPT_compute_x_squared_y_plus_xy_squared_l1385_138515

theorem compute_x_squared_y_plus_xy_squared 
  (x y : ℝ)
  (h1 : (1 / x) + (1 / y) = 4)
  (h2 : x * y + x + y = 7) :
  x^2 * y + x * y^2 = 49 := 
  sorry

end NUMINAMATH_GPT_compute_x_squared_y_plus_xy_squared_l1385_138515


namespace NUMINAMATH_GPT_jessies_current_weight_l1385_138519

theorem jessies_current_weight (initial_weight lost_weight : ℝ) (h1 : initial_weight = 69) (h2 : lost_weight = 35) :
  initial_weight - lost_weight = 34 :=
by sorry

end NUMINAMATH_GPT_jessies_current_weight_l1385_138519


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l1385_138563

theorem arithmetic_expression_evaluation :
  (-18) + (-12) - (-33) + 17 = 20 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l1385_138563


namespace NUMINAMATH_GPT_value_less_than_mean_by_std_dev_l1385_138575

theorem value_less_than_mean_by_std_dev :
  ∀ (mean value std_dev : ℝ), mean = 16.2 → std_dev = 2.3 → value = 11.6 → 
  (mean - value) / std_dev = 2 :=
by
  intros mean value std_dev h_mean h_std_dev h_value
  -- The proof goes here, but per instructions, it is skipped
  -- So we put 'sorry' to indicate that the proof is intentionally left incomplete
  sorry

end NUMINAMATH_GPT_value_less_than_mean_by_std_dev_l1385_138575


namespace NUMINAMATH_GPT_contribution_of_eight_families_l1385_138576

/-- Definition of the given conditions --/
def classroom := 200
def two_families := 2 * 20
def ten_families := 10 * 5
def missing_amount := 30

def total_raised (x : ℝ) : ℝ := two_families + ten_families + 8 * x

/-- The main theorem to prove the contribution of each of the eight families --/
theorem contribution_of_eight_families (x : ℝ) (h : total_raised x = classroom - missing_amount) : x = 10 := by
  sorry

end NUMINAMATH_GPT_contribution_of_eight_families_l1385_138576


namespace NUMINAMATH_GPT_arithmetic_sequence_Sn_l1385_138581

noncomputable def S (n : ℕ) : ℕ := sorry -- S is the sequence function

theorem arithmetic_sequence_Sn {n : ℕ} (h1 : S n = 2) (h2 : S (3 * n) = 18) : S (4 * n) = 26 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_Sn_l1385_138581


namespace NUMINAMATH_GPT_probability_rachel_robert_in_picture_l1385_138595

theorem probability_rachel_robert_in_picture :
  let lap_rachel := 120 -- Rachel's lap time in seconds
  let lap_robert := 100 -- Robert's lap time in seconds
  let duration := 900 -- 15 minutes in seconds
  let picture_duration := 60 -- Picture duration in seconds
  let one_third_rachel := lap_rachel / 3 -- One third of Rachel's lap time
  let one_third_robert := lap_robert / 3 -- One third of Robert's lap time
  let rachel_in_window_start := 20 -- Rachel in the window from 20 to 100s
  let rachel_in_window_end := 100
  let robert_in_window_start := 0 -- Robert in the window from 0 to 66.66s
  let robert_in_window_end := 66.66
  let overlap_start := max rachel_in_window_start robert_in_window_start -- The start of overlap
  let overlap_end := min rachel_in_window_end robert_in_window_end -- The end of overlap
  let overlap_duration := overlap_end - overlap_start -- Duration of the overlap
  let probability := overlap_duration / picture_duration -- Probability of both in the picture
  probability = 46.66 / 60 := sorry

end NUMINAMATH_GPT_probability_rachel_robert_in_picture_l1385_138595


namespace NUMINAMATH_GPT_total_golf_balls_l1385_138586

theorem total_golf_balls :
  let dozen := 12
  let dan := 5 * dozen
  let gus := 2 * dozen
  let chris := 48
  dan + gus + chris = 132 :=
by
  let dozen := 12
  let dan := 5 * dozen
  let gus := 2 * dozen
  let chris := 48
  sorry

end NUMINAMATH_GPT_total_golf_balls_l1385_138586


namespace NUMINAMATH_GPT_regular_ticket_cost_l1385_138502

theorem regular_ticket_cost
    (adults : ℕ) (children : ℕ) (cash_given : ℕ) (change_received : ℕ) (adult_cost : ℕ) (child_cost : ℕ) :
    adults = 2 →
    children = 3 →
    cash_given = 40 →
    change_received = 1 →
    child_cost = adult_cost - 2 →
    2 * adult_cost + 3 * child_cost = cash_given - change_received →
    adult_cost = 9 :=
by
  intros h_adults h_children h_cash_given h_change_received h_child_cost h_sum
  sorry

end NUMINAMATH_GPT_regular_ticket_cost_l1385_138502


namespace NUMINAMATH_GPT_greatest_four_digit_number_divisible_by_6_and_12_l1385_138536

theorem greatest_four_digit_number_divisible_by_6_and_12 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n % 6 = 0) ∧ (n % 12 = 0) ∧ 
  (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ (m % 6 = 0) ∧ (m % 12 = 0) → m ≤ n) ∧
  n = 9996 := 
by
  sorry

end NUMINAMATH_GPT_greatest_four_digit_number_divisible_by_6_and_12_l1385_138536


namespace NUMINAMATH_GPT_max_value_of_symmetric_function_l1385_138525

noncomputable def f (x a b : ℝ) : ℝ := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function (a b : ℝ) :
  (∀ x : ℝ, f (-2 - x) a b = f (-2 + x) a b) → ∃ x : ℝ, ∀ y : ℝ, f x a b ≥ f y a b ∧ f x a b = 16 :=
sorry

end NUMINAMATH_GPT_max_value_of_symmetric_function_l1385_138525


namespace NUMINAMATH_GPT_compute_expression_l1385_138580

theorem compute_expression : 12 * (1 / 17) * 34 = 24 := 
by {
  sorry
}

end NUMINAMATH_GPT_compute_expression_l1385_138580


namespace NUMINAMATH_GPT_part_one_part_two_l1385_138558

noncomputable def f (x a : ℝ) : ℝ :=
  |x + a| + 2 * |x - 1|

theorem part_one (a : ℝ) (h : a = 1) : 
  ∃ x : ℝ, f x 1 = 2 :=
sorry

theorem part_two (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hx : ∀ x : ℝ, 1 ≤ x → x ≤ 2 → f x a > x^2 - b + 1) : 
  (a + 1 / 2)^2 + (b + 1 / 2)^2 > 2 :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l1385_138558


namespace NUMINAMATH_GPT_determinant_example_l1385_138591

def determinant_2x2 (a b c d : ℤ) : ℤ :=
  a * d - b * c

theorem determinant_example : determinant_2x2 5 (-4) 2 3 = 23 := 
by 
  sorry

end NUMINAMATH_GPT_determinant_example_l1385_138591
