import Mathlib

namespace NUMINAMATH_GPT_ratio_Polly_to_Pulsar_l2336_233644

theorem ratio_Polly_to_Pulsar (P Po Pe : ℕ) (k : ℕ) (h1 : P = 10) (h2 : Po = k * P) (h3 : Pe = Po / 6) (h4 : P + Po + Pe = 45) : Po / P = 3 :=
by 
  -- Skipping the proof, but this sets up the Lean environment
  sorry

end NUMINAMATH_GPT_ratio_Polly_to_Pulsar_l2336_233644


namespace NUMINAMATH_GPT_min_ab_given_parallel_l2336_233647

-- Define the conditions
def parallel_vectors (a b : ℝ) : Prop :=
  4 * b - a * (b - 1) = 0 ∧ b > 1

-- Prove the main statement
theorem min_ab_given_parallel (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h_parallel : parallel_vectors a b) :
  a + b = 9 :=
sorry  -- Proof is omitted

end NUMINAMATH_GPT_min_ab_given_parallel_l2336_233647


namespace NUMINAMATH_GPT_arccos_sqrt3_div_2_eq_pi_div_6_l2336_233605

theorem arccos_sqrt3_div_2_eq_pi_div_6 : 
  Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 :=
by 
  sorry

end NUMINAMATH_GPT_arccos_sqrt3_div_2_eq_pi_div_6_l2336_233605


namespace NUMINAMATH_GPT_chocolates_sold_l2336_233617

theorem chocolates_sold (C S : ℝ) (n : ℕ) (h1 : 165 * C = n * S) (h2 : ((S - C) / C) * 100 = 10) : n = 150 :=
by
  sorry

end NUMINAMATH_GPT_chocolates_sold_l2336_233617


namespace NUMINAMATH_GPT_f_one_zero_x_range_l2336_233650

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
-- f is defined for x > 0
variable (f : ℝ → ℝ)
variables (h_domain : ∀ x, x > 0 → ∃ y, f x = y)
variables (h1 : f 2 = 1)
variables (h2 : ∀ x y, x > 0 → y > 0 → f (x * y) = f x + f y)
variables (h3 : ∀ x y, x > y → f x > f y)

-- Question 1
theorem f_one_zero (hf1 : f 1 = 0) : True := 
  by trivial
  
-- Question 2
theorem x_range (x: ℝ) (hx: f 3 + f (4 - 8 * x) > 2) : x ≤ 1/3 := sorry

end NUMINAMATH_GPT_f_one_zero_x_range_l2336_233650


namespace NUMINAMATH_GPT_percentage_of_students_who_own_cats_l2336_233628

theorem percentage_of_students_who_own_cats (total_students cats_owned : ℕ) (h_total: total_students = 500) (h_cats: cats_owned = 75) :
  (cats_owned : ℚ) / total_students * 100 = 15 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_students_who_own_cats_l2336_233628


namespace NUMINAMATH_GPT_C_eq_D_iff_n_eq_3_l2336_233684

noncomputable def C (n : ℕ) : ℝ :=
  1000 * (1 - (1 / 3^n)) / (1 - 1 / 3)

noncomputable def D (n : ℕ) : ℝ :=
  2700 * (1 - (1 / (-3)^n)) / (1 + 1 / 3)

theorem C_eq_D_iff_n_eq_3 (n : ℕ) (h : 1 ≤ n) : C n = D n ↔ n = 3 :=
by
  unfold C D
  sorry

end NUMINAMATH_GPT_C_eq_D_iff_n_eq_3_l2336_233684


namespace NUMINAMATH_GPT_sum_first_five_even_numbers_l2336_233611

theorem sum_first_five_even_numbers : (2 + 4 + 6 + 8 + 10) = 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_five_even_numbers_l2336_233611


namespace NUMINAMATH_GPT_find_a10_l2336_233655

variable {n : ℕ}
variable (a : ℕ → ℝ)
variable (h_pos : ∀ (n : ℕ), 0 < a n)
variable (h_mul : ∀ (p q : ℕ), a (p + q) = a p * a q)
variable (h_a8 : a 8 = 16)

theorem find_a10 : a 10 = 32 :=
by
  sorry

end NUMINAMATH_GPT_find_a10_l2336_233655


namespace NUMINAMATH_GPT_parallelogram_area_150deg_10_20_eq_100sqrt3_l2336_233619

noncomputable def parallelogram_area (angle: ℝ) (side1: ℝ) (side2: ℝ) : ℝ :=
  side1 * side2 * Real.sin angle

theorem parallelogram_area_150deg_10_20_eq_100sqrt3 :
  parallelogram_area (150 * Real.pi / 180) 10 20 = 100 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_parallelogram_area_150deg_10_20_eq_100sqrt3_l2336_233619


namespace NUMINAMATH_GPT_quadratic_vertex_coords_l2336_233601

theorem quadratic_vertex_coords :
  ∀ x : ℝ, (y = (x-2)^2 - 1) → (2, -1) = (2, -1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_vertex_coords_l2336_233601


namespace NUMINAMATH_GPT_sum_of_numerator_and_denominator_of_fraction_equiv_2_52_in_lowest_terms_l2336_233651

theorem sum_of_numerator_and_denominator_of_fraction_equiv_2_52_in_lowest_terms :
    let a := 63
    let b := 25
    a + b = 88 := by
  sorry

end NUMINAMATH_GPT_sum_of_numerator_and_denominator_of_fraction_equiv_2_52_in_lowest_terms_l2336_233651


namespace NUMINAMATH_GPT_value_added_to_075_of_number_l2336_233626

theorem value_added_to_075_of_number (N V : ℝ) (h1 : 0.75 * N + V = 8) (h2 : N = 8) : V = 2 := by
  sorry

end NUMINAMATH_GPT_value_added_to_075_of_number_l2336_233626


namespace NUMINAMATH_GPT_largest_even_not_sum_of_two_composite_odds_l2336_233687

-- Definitions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ k, k > 1 ∧ k < n ∧ n % k = 0

-- Theorem statement
theorem largest_even_not_sum_of_two_composite_odds :
  ∀ n : ℕ, is_even n → n > 0 → (¬ (∃ a b : ℕ, is_odd a ∧ is_odd b ∧ is_composite a ∧ is_composite b ∧ n = a + b)) ↔ n = 38 := 
by
  sorry

end NUMINAMATH_GPT_largest_even_not_sum_of_two_composite_odds_l2336_233687


namespace NUMINAMATH_GPT_part1_part2_l2336_233657

noncomputable def quadratic_eq (m x : ℝ) : Prop := m * x^2 - 2 * x + 1 = 0

theorem part1 (m : ℝ) : 
  (∃ x1 x2 : ℝ, quadratic_eq m x1 ∧ quadratic_eq m x2 ∧ x1 ≠ x2) ↔ (m ≤ 1 ∧ m ≠ 0) :=
by sorry

theorem part2 (m : ℝ) (x1 x2 : ℝ) : 
  (quadratic_eq m x1 ∧ quadratic_eq m x2 ∧ x1 * x2 - x1 - x2 = 1/2) ↔ (m = -2) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l2336_233657


namespace NUMINAMATH_GPT_gcf_2550_7140_l2336_233691

def gcf (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcf_2550_7140 : gcf 2550 7140 = 510 := 
  by 
    sorry

end NUMINAMATH_GPT_gcf_2550_7140_l2336_233691


namespace NUMINAMATH_GPT_binomial_param_exact_l2336_233679

variable (ξ : ℕ → ℝ) (n : ℕ) (p : ℝ)

-- Define the conditions: expectation and variance
axiom expectation_eq : n * p = 3
axiom variance_eq : n * p * (1 - p) = 2

-- Statement to prove
theorem binomial_param_exact (h1 : n * p = 3) (h2 : n * p * (1 - p) = 2) : p = 1 / 3 :=
by
  rw [expectation_eq] at h2
  sorry

end NUMINAMATH_GPT_binomial_param_exact_l2336_233679


namespace NUMINAMATH_GPT_right_triangle_OAB_condition_l2336_233643

theorem right_triangle_OAB_condition
  (a b : ℝ)
  (h1: a ≠ 0) 
  (h2: b ≠ 0) :
  (b - a^3) * (b - a^3 - 1/a) = 0 :=
sorry

end NUMINAMATH_GPT_right_triangle_OAB_condition_l2336_233643


namespace NUMINAMATH_GPT_f_decreasing_on_0_1_l2336_233633

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x⁻¹

theorem f_decreasing_on_0_1 : ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → x₁ < x₂ → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_GPT_f_decreasing_on_0_1_l2336_233633


namespace NUMINAMATH_GPT_range_of_m_l2336_233623

theorem range_of_m (m : ℝ) :
  (¬(∀ x y : ℝ, x^2 / (25 - m) + y^2 / (m - 7) = 1 → 25 - m > 0 ∧ m - 7 > 0 ∧ 25 - m > m - 7) ∨ 
   ¬(∀ x y : ℝ, y^2 / 5 - x^2 / m = 1 → 1 < (5 + m) / 5 ∧ (5 + m) / 5 < 4)) 
  → 7 < m ∧ m < 15 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2336_233623


namespace NUMINAMATH_GPT_yellow_dandelions_day_before_yesterday_l2336_233670

theorem yellow_dandelions_day_before_yesterday :
  ∀ (yellow_yesterday white_yesterday yellow_today white_today : ℕ),
    yellow_yesterday = 20 →
    white_yesterday = 14 →
    yellow_today = 15 →
    white_today = 11 →
    ∃ yellow_day_before_yesterday : ℕ,
      yellow_day_before_yesterday = white_yesterday + white_today :=
by sorry

end NUMINAMATH_GPT_yellow_dandelions_day_before_yesterday_l2336_233670


namespace NUMINAMATH_GPT_regular_polygon_sides_l2336_233693

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l2336_233693


namespace NUMINAMATH_GPT_equation_of_parallel_line_l2336_233645

theorem equation_of_parallel_line (A : ℝ × ℝ) (c : ℝ) : 
  A = (-1, 0) → (∀ x y, 2 * x - y + 1 = 0 → 2 * x - y + c = 0) → 
  2 * (-1) - 0 + c = 0 → c = 2 :=
by
  intros A_coord parallel_line point_on_line
  sorry

end NUMINAMATH_GPT_equation_of_parallel_line_l2336_233645


namespace NUMINAMATH_GPT_xiao_ming_arrival_time_l2336_233606

def left_home (departure_time : String) : Prop :=
  departure_time = "6:55"

def time_spent (duration : Nat) : Prop :=
  duration = 30

def arrival_time (arrival : String) : Prop :=
  arrival = "7:25"

theorem xiao_ming_arrival_time :
  left_home "6:55" → time_spent 30 → arrival_time "7:25" :=
by sorry

end NUMINAMATH_GPT_xiao_ming_arrival_time_l2336_233606


namespace NUMINAMATH_GPT_vertex_of_parabola_l2336_233603

theorem vertex_of_parabola 
  (a b c : ℝ) 
  (h1 : a * 2^2 + b * 2 + c = 5)
  (h2 : -b / (2 * a) = 2) : 
  (2, 4 * a + 2 * b + c) = (2, 5) :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l2336_233603


namespace NUMINAMATH_GPT_ann_older_than_susan_l2336_233631

variables (A S : ℕ)

theorem ann_older_than_susan (h1 : S = 11) (h2 : A + S = 27) : A - S = 5 := by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_ann_older_than_susan_l2336_233631


namespace NUMINAMATH_GPT_valid_sequences_count_l2336_233632

def g (n : ℕ) : ℕ :=
  if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 2
  else if n < 3 then 0
  else g (n - 4) + 3 * g (n - 5) + 3 * g (n - 6)

theorem valid_sequences_count : g 17 = 37 :=
  sorry

end NUMINAMATH_GPT_valid_sequences_count_l2336_233632


namespace NUMINAMATH_GPT_circle_radius_tangent_to_semicircles_and_sides_l2336_233697

noncomputable def side_length_of_square : ℝ := 4
noncomputable def side_length_of_smaller_square : ℝ := side_length_of_square / 2
noncomputable def radius_of_semicircle : ℝ := side_length_of_smaller_square / 2
noncomputable def distance_from_center_to_tangent_point : ℝ := Real.sqrt (side_length_of_smaller_square^2 + radius_of_semicircle^2)

theorem circle_radius_tangent_to_semicircles_and_sides : 
  ∃ (r : ℝ), r = (Real.sqrt 5 - 1) / 2 :=
by
  have r : ℝ := (Real.sqrt 5 - 1) / 2
  use r
  sorry -- Proof omitted

end NUMINAMATH_GPT_circle_radius_tangent_to_semicircles_and_sides_l2336_233697


namespace NUMINAMATH_GPT_solution_for_a_l2336_233624

theorem solution_for_a (x : ℝ) (a : ℝ) (h : 2 * x - a = 0) (hx : x = 1) : a = 2 := by
  rw [hx] at h
  linarith


end NUMINAMATH_GPT_solution_for_a_l2336_233624


namespace NUMINAMATH_GPT_question1_is_random_event_question2_probability_xiuShui_l2336_233635

-- Definitions for projects
inductive Project
| A | B | C | D

-- Definition for the problem context and probability computation
def xiuShuiProjects : List Project := [Project.A, Project.B]
def allProjects : List Project := [Project.A, Project.B, Project.C, Project.D]

-- Question 1
def isRandomEvent (event : Project) : Prop :=
  event = Project.C ∧ event ∈ allProjects

theorem question1_is_random_event : isRandomEvent Project.C := by
sorry

-- Question 2: Probability both visit Xiu Shui projects is 1/4
def favorable_outcomes : List (Project × Project) :=
  [(Project.A, Project.A), (Project.A, Project.B), (Project.B, Project.A), (Project.B, Project.B)]

def total_outcomes : List (Project × Project) :=
  List.product allProjects allProjects

def probability (fav : ℕ) (total : ℕ) : ℚ := fav / total

theorem question2_probability_xiuShui : probability favorable_outcomes.length total_outcomes.length = 1 / 4 := by
sorry

end NUMINAMATH_GPT_question1_is_random_event_question2_probability_xiuShui_l2336_233635


namespace NUMINAMATH_GPT_min_value_expression_l2336_233665

theorem min_value_expression :
  (∀ y : ℝ, abs y ≤ 1 → ∃ x : ℝ, 2 * x + y = 1 ∧ ∀ x : ℝ, (0 ≤ x ∧ x ≤ 1 → 
    (∃ y : ℝ, 2 * x + y = 1 ∧ abs y ≤ 1 ∧ (2 * x ^ 2 + 16 * x + 3 * y ^ 2) = 3))) :=
sorry

end NUMINAMATH_GPT_min_value_expression_l2336_233665


namespace NUMINAMATH_GPT_sq_97_l2336_233602

theorem sq_97 : 97^2 = 9409 :=
by
  sorry

end NUMINAMATH_GPT_sq_97_l2336_233602


namespace NUMINAMATH_GPT_tan_sum_simplification_l2336_233674

theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (Real.pi / 4)) = 2 * Real.sqrt 6 - 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_sum_simplification_l2336_233674


namespace NUMINAMATH_GPT_find_x_l2336_233656

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (hxy : x + y + x * y = 71) : x = 8 :=
sorry

end NUMINAMATH_GPT_find_x_l2336_233656


namespace NUMINAMATH_GPT_min_value_frac_expr_l2336_233600

theorem min_value_frac_expr (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : a < 1) (h₃ : 0 ≤ b) (h₄ : b < 1) (h₅ : 0 ≤ c) (h₆ : c < 1) :
  (1 / ((2 - a) * (2 - b) * (2 - c)) + 1 / ((2 + a) * (2 + b) * (2 + c))) ≥ 1 / 8 :=
sorry

end NUMINAMATH_GPT_min_value_frac_expr_l2336_233600


namespace NUMINAMATH_GPT_opposite_neg_2023_l2336_233678

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end NUMINAMATH_GPT_opposite_neg_2023_l2336_233678


namespace NUMINAMATH_GPT_multiplication_addition_l2336_233672

theorem multiplication_addition :
  108 * 108 + 92 * 92 = 20128 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_addition_l2336_233672


namespace NUMINAMATH_GPT_logs_per_tree_is_75_l2336_233641

-- Definitions
def logsPerDay : Nat := 5

def totalDays : Nat := 30 + 31 + 31 + 28

def totalLogs (burnRate : Nat) (days : Nat) : Nat :=
  burnRate * days

def treesNeeded : Nat := 8

def logsPerTree (totalLogs : Nat) (numTrees : Nat) : Nat :=
  totalLogs / numTrees

-- Theorem statement to prove the number of logs per tree
theorem logs_per_tree_is_75 : logsPerTree (totalLogs logsPerDay totalDays) treesNeeded = 75 :=
  by
  sorry

end NUMINAMATH_GPT_logs_per_tree_is_75_l2336_233641


namespace NUMINAMATH_GPT_difference_between_numbers_l2336_233646

theorem difference_between_numbers 
  (A B : ℝ)
  (h1 : 0.075 * A = 0.125 * B)
  (h2 : A = 2430 ∨ B = 2430) :
  A - B = 972 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_numbers_l2336_233646


namespace NUMINAMATH_GPT_men_count_eq_eight_l2336_233621

theorem men_count_eq_eight (M W B : ℕ) (total_earnings : ℝ) (men_wages : ℝ)
  (H1 : M = W) (H2 : W = B) (H3 : B = 8)
  (H4 : total_earnings = 105) (H5 : men_wages = 7) :
  M = 8 := 
by 
  -- We need to show M = 8 given conditions
  sorry

end NUMINAMATH_GPT_men_count_eq_eight_l2336_233621


namespace NUMINAMATH_GPT_least_number_of_cans_l2336_233640

theorem least_number_of_cans (maaza : ℕ) (pepsi : ℕ) (sprite : ℕ) (gcd_val : ℕ) (total_cans : ℕ)
  (h1 : maaza = 50) (h2 : pepsi = 144) (h3 : sprite = 368) (h_gcd : gcd maaza (gcd pepsi sprite) = gcd_val)
  (h_total_cans : total_cans = maaza / gcd_val + pepsi / gcd_val + sprite / gcd_val) :
  total_cans = 281 :=
sorry

end NUMINAMATH_GPT_least_number_of_cans_l2336_233640


namespace NUMINAMATH_GPT_significant_digits_of_square_side_l2336_233695

theorem significant_digits_of_square_side (A : ℝ) (s : ℝ) (h : A = 0.6400) (hs : s^2 = A) : 
  s = 0.8000 :=
sorry

end NUMINAMATH_GPT_significant_digits_of_square_side_l2336_233695


namespace NUMINAMATH_GPT_time_difference_l2336_233634

theorem time_difference (dist1 dist2 : ℕ) (speed : ℕ) (h_dist : dist1 = 600) (h_dist2 : dist2 = 550) (h_speed : speed = 40) :
  (dist1 - dist2) / speed * 60 = 75 := by
  sorry

end NUMINAMATH_GPT_time_difference_l2336_233634


namespace NUMINAMATH_GPT_jenny_boxes_sold_l2336_233675

/--
Jenny sold some boxes of Trefoils. Each box has 8.0 packs. She sold 192 packs in total.
Prove that Jenny sold 24 boxes.
-/
theorem jenny_boxes_sold (packs_per_box : Real) (total_packs_sold : Real) (num_boxes_sold : Real) 
  (h1 : packs_per_box = 8.0) (h2 : total_packs_sold = 192) : num_boxes_sold = 24 :=
by
  have h3 : num_boxes_sold = total_packs_sold / packs_per_box :=
    by sorry
  sorry

end NUMINAMATH_GPT_jenny_boxes_sold_l2336_233675


namespace NUMINAMATH_GPT_probability_of_6_heads_in_10_flips_l2336_233664

theorem probability_of_6_heads_in_10_flips :
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := Nat.choose 10 6
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 210 / 1024 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_6_heads_in_10_flips_l2336_233664


namespace NUMINAMATH_GPT_problems_per_worksheet_l2336_233692

theorem problems_per_worksheet (P : ℕ) (graded : ℕ) (remaining : ℕ) (total_worksheets : ℕ) (total_problems_remaining : ℕ) :
    graded = 5 →
    total_worksheets = 9 →
    total_problems_remaining = 16 →
    remaining = total_worksheets - graded →
    4 * P = total_problems_remaining →
    P = 4 :=
by
  intros h_graded h_worksheets h_problems h_remaining h_equation
  sorry

end NUMINAMATH_GPT_problems_per_worksheet_l2336_233692


namespace NUMINAMATH_GPT_train_speed_l2336_233629

theorem train_speed
  (train_length : ℝ)
  (cross_time : ℝ)
  (man_speed_kmh : ℝ)
  (train_speed_kmh : ℝ) :
  (train_length = 150) →
  (cross_time = 6) →
  (man_speed_kmh = 5) →
  (man_speed_kmh * 1000 / 3600 + (train_speed_kmh * 1000 / 3600)) * cross_time = train_length →
  train_speed_kmh = 85 :=
by
  intros htl hct hmk hs
  sorry

end NUMINAMATH_GPT_train_speed_l2336_233629


namespace NUMINAMATH_GPT_total_cost_of_shoes_before_discount_l2336_233607

theorem total_cost_of_shoes_before_discount (S J H : ℝ) (D : ℝ) (shoes jerseys hats : ℝ) :
  jerseys = 1/4 * shoes ∧
  hats = 2 * jerseys ∧
  D = 0.9 * (6 * shoes + 4 * jerseys + 3 * hats) ∧
  D = 620 →
  6 * shoes = 486.30 := by
  sorry

end NUMINAMATH_GPT_total_cost_of_shoes_before_discount_l2336_233607


namespace NUMINAMATH_GPT_vectors_parallel_implies_fraction_l2336_233615

theorem vectors_parallel_implies_fraction (α : ℝ) :
  let a := (Real.sin α, 3)
  let b := (Real.cos α, 1)
  (a.1 / b.1 = 3) → (Real.sin (2 * α) / (Real.cos α) ^ 2 = 6) :=
by
  sorry

end NUMINAMATH_GPT_vectors_parallel_implies_fraction_l2336_233615


namespace NUMINAMATH_GPT_swimming_time_per_style_l2336_233609

theorem swimming_time_per_style (d v1 v2 v3 v4 t: ℝ) 
    (h1: d = 600) 
    (h2: v1 = 45) 
    (h3: v2 = 35) 
    (h4: v3 = 40) 
    (h5: v4 = 30)
    (h6: t = 15) 
    (h7: d / 4 = 150) 
    : (t / 4 = 3.75) :=
by
  sorry

end NUMINAMATH_GPT_swimming_time_per_style_l2336_233609


namespace NUMINAMATH_GPT_simplify_evaluate_l2336_233688

theorem simplify_evaluate :
  ∀ (x : ℝ), x = Real.sqrt 2 - 1 →
  ((1 + 4 / (x - 3)) / ((x^2 + 2*x + 1) / (2*x - 6))) = Real.sqrt 2 :=
by
  intros x hx
  sorry

end NUMINAMATH_GPT_simplify_evaluate_l2336_233688


namespace NUMINAMATH_GPT_geometric_seq_common_ratio_l2336_233682

theorem geometric_seq_common_ratio 
  (a : ℕ → ℝ) -- a_n is the sequence
  (S : ℕ → ℝ) -- S_n is the partial sum of the sequence
  (h1 : a 3 = 2 * S 2 + 1) -- condition a_3 = 2S_2 + 1
  (h2 : a 4 = 2 * S 3 + 1) -- condition a_4 = 2S_3 + 1
  (h3 : S 2 = a 1 / (1 / q) * (1 - q^3) / (1 - q)) -- sum of first 2 terms
  (h4 : S 3 = a 1 / (1 / q) * (1 - q^4) / (1 - q)) -- sum of first 3 terms
  : q = 3 := -- conclusion
by sorry

end NUMINAMATH_GPT_geometric_seq_common_ratio_l2336_233682


namespace NUMINAMATH_GPT_basic_computer_price_l2336_233604

theorem basic_computer_price (C P : ℝ) 
(h1 : C + P = 2500) 
(h2 : P = (1 / 6) * (C + 500 + P)) : 
  C = 2000 :=
by
  sorry

end NUMINAMATH_GPT_basic_computer_price_l2336_233604


namespace NUMINAMATH_GPT_number_is_correct_l2336_233699

theorem number_is_correct (x : ℝ) (h : 0.35 * x = 0.25 * 50) : x = 35.7143 :=
by 
  sorry

end NUMINAMATH_GPT_number_is_correct_l2336_233699


namespace NUMINAMATH_GPT_minimum_order_amount_to_get_discount_l2336_233681

theorem minimum_order_amount_to_get_discount 
  (cost_quiche : ℝ) (cost_croissant : ℝ) (cost_biscuit : ℝ) (n_quiches : ℝ) (n_croissants : ℝ) (n_biscuits : ℝ)
  (discount_percent : ℝ) (total_with_discount : ℝ) (min_order_amount : ℝ) :
  cost_quiche = 15.0 → cost_croissant = 3.0 → cost_biscuit = 2.0 →
  n_quiches = 2 → n_croissants = 6 → n_biscuits = 6 →
  discount_percent = 0.10 → total_with_discount = 54.0 →
  (n_quiches * cost_quiche + n_croissants * cost_croissant + n_biscuits * cost_biscuit) * (1 - discount_percent) = total_with_discount →
  min_order_amount = 60.0 :=
by
  sorry

end NUMINAMATH_GPT_minimum_order_amount_to_get_discount_l2336_233681


namespace NUMINAMATH_GPT_no_solution_for_equation_l2336_233662

theorem no_solution_for_equation :
  ¬ (∃ x : ℝ, 
    4 * x * (10 * x - (-10 - (3 * x - 8 * (x + 1)))) + 5 * (12 - (4 * (x + 1) - 3 * x)) = 
    18 * x^2 - (6 * x^2 - (7 * x + 4 * (2 * x^2 - x + 11)))) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_equation_l2336_233662


namespace NUMINAMATH_GPT_no_integer_solutions_exist_l2336_233614

theorem no_integer_solutions_exist (n m : ℤ) : 
  (n ^ 2 - m ^ 2 = 250) → false := 
sorry 

end NUMINAMATH_GPT_no_integer_solutions_exist_l2336_233614


namespace NUMINAMATH_GPT_schedule_courses_l2336_233690

/-- Definition of valid schedule count where at most one pair of courses is consecutive. -/
def count_valid_schedules : ℕ := 180

/-- Given 7 periods and 3 courses, determine the number of valid schedules 
    where at most one pair of these courses is consecutive. -/
theorem schedule_courses (periods : ℕ) (courses : ℕ) (valid_schedules : ℕ) :
  periods = 7 → courses = 3 → valid_schedules = count_valid_schedules →
  valid_schedules = 180 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_schedule_courses_l2336_233690


namespace NUMINAMATH_GPT_percentage_increase_l2336_233659

theorem percentage_increase 
  (P : ℝ)
  (bought_price : ℝ := 0.80 * P) 
  (original_profit : ℝ := 0.3600000000000001 * P) :
  ∃ X : ℝ, X = 70.00000000000002 ∧ (1.3600000000000001 * P = bought_price * (1 + X / 100)) :=
sorry

end NUMINAMATH_GPT_percentage_increase_l2336_233659


namespace NUMINAMATH_GPT_jaylene_saves_fraction_l2336_233673

-- Statement of the problem
theorem jaylene_saves_fraction (r_saves : ℝ) (j_saves : ℝ) (m_saves : ℝ) 
    (r_salary_fraction : r_saves = 2 / 5) 
    (m_salary_fraction : m_saves = 1 / 2) 
    (total_savings : 4 * (r_saves * 500 + j_saves * 500 + m_saves * 500) = 3000) : 
    j_saves = 3 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_jaylene_saves_fraction_l2336_233673


namespace NUMINAMATH_GPT_james_carrot_sticks_l2336_233652

def carrots_eaten_after_dinner (total_carrots : ℕ) (carrots_before_dinner : ℕ) : ℕ :=
  total_carrots - carrots_before_dinner

theorem james_carrot_sticks : carrots_eaten_after_dinner 37 22 = 15 := by
  sorry

end NUMINAMATH_GPT_james_carrot_sticks_l2336_233652


namespace NUMINAMATH_GPT_minimum_value_l2336_233694

open Real

theorem minimum_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eqn : 2 * x + y = 2) :
    ∃ x y, (0 < x) ∧ (0 < y) ∧ (2 * x + y = 2) ∧ (x + sqrt (x^2 + y^2) = 8 / 5) :=
sorry

end NUMINAMATH_GPT_minimum_value_l2336_233694


namespace NUMINAMATH_GPT_number_of_good_card_groups_l2336_233683

noncomputable def card_value (k : ℕ) : ℕ := 2 ^ k

def is_good_card_group (cards : Finset ℕ) : Prop :=
  (cards.sum card_value = 2004)

theorem number_of_good_card_groups : 
  ∃ n : ℕ, n = 1006009 ∧ ∃ (cards : Finset ℕ), is_good_card_group cards :=
sorry

end NUMINAMATH_GPT_number_of_good_card_groups_l2336_233683


namespace NUMINAMATH_GPT_subset_div_chain_l2336_233612

theorem subset_div_chain (m n : ℕ) (h_m : m > 0) (h_n : n > 0) (S : Finset ℕ) (hS : S.card = (2^m - 1) * n + 1) (hS_subset : S ⊆ Finset.range (2^(m) * n + 1)) :
  ∃ (a : Fin (m+1) → ℕ), (∀ i, a i ∈ S) ∧ (∀ k : ℕ, k < m → a k ∣ a (k + 1)) :=
sorry

end NUMINAMATH_GPT_subset_div_chain_l2336_233612


namespace NUMINAMATH_GPT_salt_concentration_solution_l2336_233618

theorem salt_concentration_solution
  (x y z : ℕ)
  (h1 : x + y + z = 30)
  (h2 : 2 * x + 3 * y = 35)
  (h3 : 3 * y + 2 * z = 45) :
  x = 10 ∧ y = 5 ∧ z = 15 := by
  sorry

end NUMINAMATH_GPT_salt_concentration_solution_l2336_233618


namespace NUMINAMATH_GPT_quadratic_root_range_l2336_233676

noncomputable def quadratic_function (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 9 * a

theorem quadratic_root_range (a : ℝ) (h : a ≠ 0) (h_distinct_roots : ∃ x1 x2 : ℝ, quadratic_function a x1 = 0 ∧ quadratic_function a x2 = 0 ∧ x1 ≠ x2 ∧ x1 < 1 ∧ x2 > 1) :
    -(2 / 11) < a ∧ a < 0 :=
sorry

end NUMINAMATH_GPT_quadratic_root_range_l2336_233676


namespace NUMINAMATH_GPT_cody_final_tickets_l2336_233620

def initial_tickets : ℝ := 56.5
def lost_tickets : ℝ := 6.3
def spent_tickets : ℝ := 25.75
def won_tickets : ℝ := 10.25
def dropped_tickets : ℝ := 3.1

theorem cody_final_tickets : 
  initial_tickets - lost_tickets - spent_tickets + won_tickets - dropped_tickets = 31.6 :=
by
  sorry

end NUMINAMATH_GPT_cody_final_tickets_l2336_233620


namespace NUMINAMATH_GPT_find_common_difference_l2336_233638

def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
∀ n m : ℕ, n < m → (a m - a n) = (m - n) * (a 1 - a 0)

def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
∀ n : ℕ, S n = n * a 1 + (n * (n - 1)) / 2 * (a 1 - a 0)

noncomputable def quadratic_roots (c : ℚ) (x1 x2 : ℚ) : Prop :=
2 * x1^2 - 12 * x1 + c = 0 ∧ 2 * x2^2 - 12 * x2 + c = 0

theorem find_common_difference
  (a : ℕ → ℚ) (S : ℕ → ℚ) (c : ℚ)
  (h_arith_seq: is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_roots : quadratic_roots c (a 3) (a 7))
  (h_S13 : S 13 = c) :
  (a 1 - a 0 = -3/2) ∨ (a 1 - a 0 = -7/4) :=
sorry

end NUMINAMATH_GPT_find_common_difference_l2336_233638


namespace NUMINAMATH_GPT_brick_length_l2336_233654

theorem brick_length (x : ℝ) (brick_width : ℝ) (brick_height : ℝ) (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ) (number_of_bricks : ℕ)
  (h_brick : brick_width = 11.25) (h_brick_height : brick_height = 6)
  (h_wall : wall_length = 800) (h_wall_width : wall_width = 600) 
  (h_wall_height : wall_height = 22.5) (h_bricks_number : number_of_bricks = 1280)
  (h_eq : (wall_length * wall_width * wall_height) = (x * brick_width * brick_height) * number_of_bricks) : 
  x = 125 := by
  sorry

end NUMINAMATH_GPT_brick_length_l2336_233654


namespace NUMINAMATH_GPT_find_principal_amount_l2336_233671

variable (P : ℝ)
variable (R : ℝ := 5)
variable (T : ℝ := 13)
variable (SI : ℝ := 1300)

theorem find_principal_amount (h1 : SI = (P * R * T) / 100) : P = 2000 :=
sorry

end NUMINAMATH_GPT_find_principal_amount_l2336_233671


namespace NUMINAMATH_GPT_probability_of_MATHEMATICS_letter_l2336_233639

def unique_letters_in_mathematics : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

theorem probability_of_MATHEMATICS_letter :
  let total_letters := 26
  let unique_letters_count := unique_letters_in_mathematics.card
  (unique_letters_count / total_letters : ℝ) = 8 / 26 := by
  sorry

end NUMINAMATH_GPT_probability_of_MATHEMATICS_letter_l2336_233639


namespace NUMINAMATH_GPT_rational_range_l2336_233668

theorem rational_range (a : ℚ) (h : a - |a| = 2 * a) : a ≤ 0 := 
sorry

end NUMINAMATH_GPT_rational_range_l2336_233668


namespace NUMINAMATH_GPT_g_7_eq_98_l2336_233613

noncomputable def g : ℕ → ℝ := sorry

axiom g_0 : g 0 = 0
axiom g_1 : g 1 = 2
axiom functional_equation (m n : ℕ) (h : m ≥ n) : g (m + n) + g (m - n) = (g (2 * m) + g (2 * n)) / 2

theorem g_7_eq_98 : g 7 = 98 :=
sorry

end NUMINAMATH_GPT_g_7_eq_98_l2336_233613


namespace NUMINAMATH_GPT_rectangle_symmetry_l2336_233608

-- Definitions of symmetry properties
def isAxisymmetric (shape : Type) : Prop := sorry
def isCentrallySymmetric (shape : Type) : Prop := sorry

-- Specific shapes
def EquilateralTriangle : Type := sorry
def Parallelogram : Type := sorry
def Rectangle : Type := sorry
def RegularPentagon : Type := sorry

-- The theorem we want to prove
theorem rectangle_symmetry : 
  isAxisymmetric Rectangle ∧ isCentrallySymmetric Rectangle := sorry

end NUMINAMATH_GPT_rectangle_symmetry_l2336_233608


namespace NUMINAMATH_GPT_mary_unanswered_questions_l2336_233689

theorem mary_unanswered_questions :
  ∃ (c w u : ℕ), 150 = 6 * c + 3 * u ∧ 118 = 40 + 5 * c - 2 * w ∧ 50 = c + w + u ∧ u = 16 :=
by
  sorry

end NUMINAMATH_GPT_mary_unanswered_questions_l2336_233689


namespace NUMINAMATH_GPT_probability_of_one_radio_operator_per_group_l2336_233622

def total_ways_to_assign_soldiers_to_groups : ℕ := 27720
def ways_to_assign_radio_operators_to_groups : ℕ := 7560

theorem probability_of_one_radio_operator_per_group :
  (ways_to_assign_radio_operators_to_groups : ℚ) / (total_ways_to_assign_soldiers_to_groups : ℚ) = 3 / 11 := 
sorry

end NUMINAMATH_GPT_probability_of_one_radio_operator_per_group_l2336_233622


namespace NUMINAMATH_GPT_arithmetic_sequences_ratio_l2336_233648

theorem arithmetic_sequences_ratio
  (a b : ℕ → ℕ)
  (S T : ℕ → ℕ)
  (h1 : ∀ n, S n = (n * (2 * (a 1) + (n - 1) * (a 2 - a 1))) / 2)
  (h2 : ∀ n, T n = (n * (2 * (b 1) + (n - 1) * (b 2 - b 1))) / 2)
  (h3 : ∀ n, (S n) / (T n) = (2 * n + 2) / (n + 3)) :
  (a 10) / (b 9) = 2 := sorry

end NUMINAMATH_GPT_arithmetic_sequences_ratio_l2336_233648


namespace NUMINAMATH_GPT_sqrt_expression_l2336_233636

noncomputable def a : ℝ := 5 - 3 * Real.sqrt 2
noncomputable def b : ℝ := 5 + 3 * Real.sqrt 2

theorem sqrt_expression : 
  Real.sqrt (a^2) + Real.sqrt (b^2) + 2 = 12 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_l2336_233636


namespace NUMINAMATH_GPT_jason_daily_charge_l2336_233625

theorem jason_daily_charge 
  (total_cost_eric : ℕ) (days_eric : ℕ) (daily_charge : ℕ)
  (h1 : total_cost_eric = 800) (h2 : days_eric = 20)
  (h3 : daily_charge = total_cost_eric / days_eric) :
  daily_charge = 40 := 
by
  sorry

end NUMINAMATH_GPT_jason_daily_charge_l2336_233625


namespace NUMINAMATH_GPT_temperature_lower_than_minus_three_l2336_233696

theorem temperature_lower_than_minus_three (a b : ℤ) (hx : a = -3) (hy : b = -6) : a + b = -9 :=
by
  sorry

end NUMINAMATH_GPT_temperature_lower_than_minus_three_l2336_233696


namespace NUMINAMATH_GPT_max_students_exam_l2336_233685

/--
An exam contains 4 multiple-choice questions, each with three options (A, B, C). Several students take the exam.
For any group of 3 students, there is at least one question where their answers are all different.
Each student answers all questions. Prove that the maximum number of students who can take the exam is 9.
-/
theorem max_students_exam (n : ℕ) (A B C : ℕ → ℕ → ℕ) (q : ℕ) :
  (∀ (s1 s2 s3 : ℕ), ∃ (q : ℕ), (1 ≤ q ∧ q ≤ 4) ∧ (A s1 q ≠ A s2 q ∧ A s1 q ≠ A s3 q ∧ A s2 q ≠ A s3 q)) →
  q = 4 ∧ (∀ s, 1 ≤ s → s ≤ n) → n ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_max_students_exam_l2336_233685


namespace NUMINAMATH_GPT_find_black_balls_l2336_233653

-- Define the conditions given in the problem.
def initial_balls : ℕ := 10
def all_red_balls (p_red : ℝ) : Prop := p_red = 1
def equal_red_black (p_red : ℝ) (p_black : ℝ) : Prop := p_red = 0.5 ∧ p_black = 0.5
def with_green_balls (p_red : ℝ) (green_balls : ℕ) : Prop := green_balls = 2 ∧ p_red = 0.7

-- Define the total probability condition
def total_probability (p_red : ℝ) (p_green : ℝ) (p_black : ℝ) : Prop :=
  p_red + p_green + p_black = 1

-- The final statement to prove
theorem find_black_balls :
  ∃ black_balls : ℕ,
    initial_balls = 10 ∧
    (∃ p_red : ℝ, all_red_balls p_red) ∧
    (∃ p_red p_black : ℝ, equal_red_black p_red p_black) ∧
    (∃ p_red : ℝ, ∃ green_balls : ℕ, with_green_balls p_red green_balls) ∧
    (∃ p_red p_green p_black : ℝ, total_probability p_red p_green p_black) ∧
    black_balls = 1 :=
sorry

end NUMINAMATH_GPT_find_black_balls_l2336_233653


namespace NUMINAMATH_GPT_total_people_who_eat_vegetarian_l2336_233680

def people_who_eat_only_vegetarian := 16
def people_who_eat_both_vegetarian_and_non_vegetarian := 12

-- We want to prove that the total number of people who eat vegetarian is 28
theorem total_people_who_eat_vegetarian : 
  people_who_eat_only_vegetarian + people_who_eat_both_vegetarian_and_non_vegetarian = 28 :=
by 
  sorry

end NUMINAMATH_GPT_total_people_who_eat_vegetarian_l2336_233680


namespace NUMINAMATH_GPT_part1_part2_l2336_233686

theorem part1 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) : k > 3 / 4 :=
sorry

theorem part2 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) (hx1x2 : ∀ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0 → x1 * x2 = 5) : k = 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2336_233686


namespace NUMINAMATH_GPT_smallest_solution_of_quartic_l2336_233698

theorem smallest_solution_of_quartic :
  ∃ x : ℝ, x^4 - 40*x^2 + 144 = 0 ∧ ∀ y : ℝ, (y^4 - 40*y^2 + 144 = 0) → x ≤ y :=
sorry

end NUMINAMATH_GPT_smallest_solution_of_quartic_l2336_233698


namespace NUMINAMATH_GPT_compute_expression_l2336_233663

theorem compute_expression : 7 * (1 / 21) * 42 = 14 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l2336_233663


namespace NUMINAMATH_GPT_minimizes_G_at_7_over_12_l2336_233637

def F (p q : ℝ) : ℝ :=
  -2 * p * q + 3 * p * (1 - q) + 3 * (1 - p) * q - 4 * (1 - p) * (1 - q)

noncomputable def G (p : ℝ) : ℝ :=
  max (3 * p - 4) (3 - 5 * p)

theorem minimizes_G_at_7_over_12 :
  ∀ p : ℝ, 0 ≤ p ∧ p ≤ 1 → (∀ p, G p ≥ G (7 / 12)) ↔ p = 7 / 12 :=
by
  sorry

end NUMINAMATH_GPT_minimizes_G_at_7_over_12_l2336_233637


namespace NUMINAMATH_GPT_kameron_kangaroos_l2336_233616

theorem kameron_kangaroos (K : ℕ) (B_now : ℕ) (rate : ℕ) (days : ℕ)
    (h1 : B_now = 20)
    (h2 : rate = 2)
    (h3 : days = 40)
    (h4 : B_now + rate * days = K) : K = 100 := by
  sorry

end NUMINAMATH_GPT_kameron_kangaroos_l2336_233616


namespace NUMINAMATH_GPT_gcd_228_1995_l2336_233610

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := 
by
  sorry

end NUMINAMATH_GPT_gcd_228_1995_l2336_233610


namespace NUMINAMATH_GPT_consecutive_integers_sum_l2336_233649

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_GPT_consecutive_integers_sum_l2336_233649


namespace NUMINAMATH_GPT_rope_cut_number_not_8_l2336_233667

theorem rope_cut_number_not_8 (l : ℝ) (h1 : (1 : ℝ) % l = 0) (h2 : (2 : ℝ) % l = 0) (h3 : (3 / l) ≠ 8) : False :=
by
  sorry

end NUMINAMATH_GPT_rope_cut_number_not_8_l2336_233667


namespace NUMINAMATH_GPT_smallest_possible_perimeter_l2336_233669

open Real

theorem smallest_possible_perimeter
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a^2 + b^2 = 2016) :
  a + b + 2^3 * 3 * sqrt 14 = 48 + 2^3 * 3 * sqrt 14 :=
sorry

end NUMINAMATH_GPT_smallest_possible_perimeter_l2336_233669


namespace NUMINAMATH_GPT_factorization_x6_minus_5x4_plus_8x2_minus_4_l2336_233627

theorem factorization_x6_minus_5x4_plus_8x2_minus_4 (x : ℝ) :
  x^6 - 5 * x^4 + 8 * x^2 - 4 = (x - 1) * (x + 1) * (x^2 - 2)^2 :=
sorry

end NUMINAMATH_GPT_factorization_x6_minus_5x4_plus_8x2_minus_4_l2336_233627


namespace NUMINAMATH_GPT_mother_l2336_233660

def age_relations (P M : ℕ) : Prop :=
  P = (2 * M) / 5 ∧ P + 10 = (M + 10) / 2

theorem mother's_present_age (P M : ℕ) (h : age_relations P M) : M = 50 :=
by
  sorry

end NUMINAMATH_GPT_mother_l2336_233660


namespace NUMINAMATH_GPT_cartons_in_case_l2336_233666

theorem cartons_in_case (b : ℕ) (hb : b ≥ 1) (h : 2 * c * b * 500 = 1000) : c = 1 :=
by
  -- sorry is used to indicate where the proof would go
  sorry

end NUMINAMATH_GPT_cartons_in_case_l2336_233666


namespace NUMINAMATH_GPT_variance_scaled_l2336_233642

theorem variance_scaled (s1 : ℝ) (c : ℝ) (h1 : s1 = 3) (h2 : c = 3) :
  s1 * (c^2) = 27 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_variance_scaled_l2336_233642


namespace NUMINAMATH_GPT_find_a_l2336_233661

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 2 then x^2 - 4 else |x - 3| + a

theorem find_a (a : ℝ) (h : f (f (Real.sqrt 6) a) a = 3) : a = 2 := by
  sorry

end NUMINAMATH_GPT_find_a_l2336_233661


namespace NUMINAMATH_GPT_find_30_cent_items_l2336_233677

-- Define the parameters and their constraints
variables (a d b c : ℕ)

-- Define the conditions
def total_items : Prop := a + d + b + c = 50
def total_cost : Prop := 30 * a + 150 * d + 200 * b + 300 * c = 6000

-- The theorem to prove the number of 30-cent items purchased
theorem find_30_cent_items (h1 : total_items a d b c) (h2 : total_cost a d b c) : 
  ∃ a, a + d + b + c = 50 ∧ 30 * a + 150 * d + 200 * b + 300 * c = 6000 := 
sorry

end NUMINAMATH_GPT_find_30_cent_items_l2336_233677


namespace NUMINAMATH_GPT_intersection_is_correct_l2336_233630

def A : Set ℤ := {0, 3, 4}
def B : Set ℤ := {-1, 0, 2, 3}

theorem intersection_is_correct : A ∩ B = {0, 3} := by
  sorry

end NUMINAMATH_GPT_intersection_is_correct_l2336_233630


namespace NUMINAMATH_GPT_fraction_equation_l2336_233658

theorem fraction_equation (a : ℕ) (h : a > 0) (eq : (a : ℚ) / (a + 35) = 0.875) : a = 245 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equation_l2336_233658
