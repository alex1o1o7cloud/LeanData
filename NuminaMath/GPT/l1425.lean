import Mathlib

namespace NUMINAMATH_GPT_bottles_in_cups_l1425_142574

-- Defining the given conditions
variables (BOTTLE GLASS CUP JUG : ℕ)

axiom h1 : JUG = BOTTLE + GLASS
axiom h2 : 2 * JUG = 7 * GLASS
axiom h3 : BOTTLE = CUP + 2 * GLASS

theorem bottles_in_cups : BOTTLE = 5 * CUP :=
sorry

end NUMINAMATH_GPT_bottles_in_cups_l1425_142574


namespace NUMINAMATH_GPT_line_slope_intercept_l1425_142561

theorem line_slope_intercept (x y : ℝ) (k b : ℝ) (h : 3 * x + 4 * y + 5 = 0) :
  k = -3 / 4 ∧ b = -5 / 4 :=
by sorry

end NUMINAMATH_GPT_line_slope_intercept_l1425_142561


namespace NUMINAMATH_GPT_all_positive_rationals_are_red_l1425_142525

-- Define the property of being red for rational numbers
def is_red (x : ℚ) : Prop :=
  ∃ n : ℕ, ∃ (f : ℕ → ℚ), f 0 = 1 ∧ (∀ m : ℕ, f (m + 1) = f m + 1 ∨ f (m + 1) = f m / (f m + 1)) ∧ f n = x

-- Proposition stating that all positive rational numbers are red
theorem all_positive_rationals_are_red :
  ∀ x : ℚ, 0 < x → is_red x :=
  by sorry

end NUMINAMATH_GPT_all_positive_rationals_are_red_l1425_142525


namespace NUMINAMATH_GPT_cannot_reach_eighth_vertex_l1425_142517

def Point := ℕ × ℕ × ℕ

def symmetry (p1 p2 : Point) : Point :=
  let (a, b, c) := p1
  let (a', b', c') := p2
  (2 * a' - a, 2 * b' - b, 2 * c' - c)

def vertices : List Point :=
  [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

theorem cannot_reach_eighth_vertex : ∀ (p : Point), p ∈ vertices → ∀ (q : Point), q ∈ vertices → 
  ¬(symmetry p q = (1, 1, 1)) :=
by
  sorry

end NUMINAMATH_GPT_cannot_reach_eighth_vertex_l1425_142517


namespace NUMINAMATH_GPT_divide_one_meter_into_100_parts_l1425_142531

theorem divide_one_meter_into_100_parts :
  (1 / 100 : ℝ) = 1 / 100 := 
by
  sorry

end NUMINAMATH_GPT_divide_one_meter_into_100_parts_l1425_142531


namespace NUMINAMATH_GPT_arithmetic_sequence_30th_term_l1425_142516

theorem arithmetic_sequence_30th_term (a1 a2 a3 d a30 : ℤ) 
 (h1 : a1 = 3) (h2 : a2 = 12) (h3 : a3 = 21) 
 (h4 : d = a2 - a1) (h5 : a3 = a1 + 2 * d) 
 (h6 : a30 = a1 + 29 * d) : 
 a30 = 264 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_30th_term_l1425_142516


namespace NUMINAMATH_GPT_birds_in_marsh_end_of_day_l1425_142535

def geese_initial : Nat := 58
def ducks : Nat := 37
def geese_flew_away : Nat := 15
def swans : Nat := 22
def herons : Nat := 2

theorem birds_in_marsh_end_of_day : 
  58 - 15 + 37 + 22 + 2 = 104 := by
  sorry

end NUMINAMATH_GPT_birds_in_marsh_end_of_day_l1425_142535


namespace NUMINAMATH_GPT_distance_between_stations_is_correct_l1425_142597

noncomputable def distance_between_stations : ℕ := 200

theorem distance_between_stations_is_correct 
  (start_hour_p : ℕ := 7) 
  (speed_p : ℕ := 20) 
  (start_hour_q : ℕ := 8) 
  (speed_q : ℕ := 25) 
  (meeting_hour : ℕ := 12)
  (time_travel_p := meeting_hour - start_hour_p) -- Time traveled by train from P
  (time_travel_q := meeting_hour - start_hour_q) -- Time traveled by train from Q 
  (distance_travel_p := speed_p * time_travel_p) 
  (distance_travel_q := speed_q * time_travel_q) : 
  distance_travel_p + distance_travel_q = distance_between_stations :=
by 
  sorry

end NUMINAMATH_GPT_distance_between_stations_is_correct_l1425_142597


namespace NUMINAMATH_GPT_number_of_mismatching_socks_l1425_142580

def SteveTotalSocks := 48
def StevePairsMatchingSocks := 11

theorem number_of_mismatching_socks :
  SteveTotalSocks - (StevePairsMatchingSocks * 2) = 26 := by
  sorry

end NUMINAMATH_GPT_number_of_mismatching_socks_l1425_142580


namespace NUMINAMATH_GPT_g_minus_6_eq_neg_20_l1425_142588

noncomputable def g : ℤ → ℤ := sorry

axiom condition1 : g 1 - 1 > 0
axiom condition2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom condition3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

theorem g_minus_6_eq_neg_20 : g (-6) = -20 := sorry

end NUMINAMATH_GPT_g_minus_6_eq_neg_20_l1425_142588


namespace NUMINAMATH_GPT_problem_solution_l1425_142584

noncomputable def sqrt_3_simplest : Prop :=
  let A := Real.sqrt 3
  let B := Real.sqrt 0.5
  let C := Real.sqrt 8
  let D := Real.sqrt (1 / 3)
  ∀ (x : ℝ), x = A ∨ x = B ∨ x = C ∨ x = D → x = A → 
    (x = Real.sqrt 0.5 ∨ x = Real.sqrt 8 ∨ x = Real.sqrt (1 / 3)) ∧ 
    ¬(x = Real.sqrt 0.5 ∨ x = 2 * Real.sqrt 2 ∨ x = Real.sqrt (1 / 3))

theorem problem_solution : sqrt_3_simplest :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1425_142584


namespace NUMINAMATH_GPT_regression_slope_interpretation_l1425_142508

-- Define the variables and their meanings
variable {x y : ℝ}

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 0.8 * x + 4.6

-- Define the proof statement
theorem regression_slope_interpretation (hx : ∀ x, y = regression_line x) :
  ∀ delta_x : ℝ, delta_x = 1 → (regression_line (x + delta_x) - regression_line x) = 0.8 :=
by
  intros delta_x h_delta_x
  rw [h_delta_x, regression_line, regression_line]
  simp
  sorry

end NUMINAMATH_GPT_regression_slope_interpretation_l1425_142508


namespace NUMINAMATH_GPT_company_employee_count_l1425_142564

theorem company_employee_count (E : ℝ) (H1 : E > 0) (H2 : 0.60 * E = 0.55 * (E + 30)) : E + 30 = 360 :=
by
  -- The proof steps would go here, but that is not required.
  sorry

end NUMINAMATH_GPT_company_employee_count_l1425_142564


namespace NUMINAMATH_GPT_select_p_elements_with_integer_mean_l1425_142571

theorem select_p_elements_with_integer_mean {p : ℕ} (hp : Nat.Prime p) (p_odd : p % 2 = 1) :
  ∃ (M : Finset ℕ), (M.card = (p^2 + 1) / 2) ∧ ∃ (S : Finset ℕ), (S.card = p) ∧ ((S.sum id) % p = 0) :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_select_p_elements_with_integer_mean_l1425_142571


namespace NUMINAMATH_GPT_ratio_of_small_rectangle_length_to_width_l1425_142546

-- Define the problem conditions
variables (s : ℝ)

-- Define the length and width of the small rectangle
def length_of_small_rectangle := 3 * s
def width_of_small_rectangle := s

-- Prove that the ratio of the length to the width of the small rectangle is 3
theorem ratio_of_small_rectangle_length_to_width : 
  length_of_small_rectangle s / width_of_small_rectangle s = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_small_rectangle_length_to_width_l1425_142546


namespace NUMINAMATH_GPT_least_xy_value_l1425_142555

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/8) : x * y = 96 :=
sorry

end NUMINAMATH_GPT_least_xy_value_l1425_142555


namespace NUMINAMATH_GPT_evaluate_expression_l1425_142527

theorem evaluate_expression : (6^6) * (12^6) * (6^12) * (12^12) = 72^18 := 
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1425_142527


namespace NUMINAMATH_GPT_track_circumference_is_720_l1425_142560

variable (P Q : Type) -- Define the types of P and Q, e.g., as points or runners.

noncomputable def circumference_of_the_track (C : ℝ) : Prop :=
  ∃ y : ℝ, 
  (∃ first_meeting_condition : Prop, first_meeting_condition = (150 = y - 150) ∧
  ∃ second_meeting_condition : Prop, second_meeting_condition = (2*y - 90 = y + 90) ∧
  C = 2 * y)

theorem track_circumference_is_720 :
  circumference_of_the_track 720 :=
by
  sorry

end NUMINAMATH_GPT_track_circumference_is_720_l1425_142560


namespace NUMINAMATH_GPT_numberOfZeros_l1425_142520

noncomputable def g (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem numberOfZeros :
  ∃ x ∈ Set.Ioo 1 (Real.exp Real.pi), g x = 0 ∧ ∀ y ∈ Set.Ioo 1 (Real.exp Real.pi), g y = 0 → y = x := 
sorry

end NUMINAMATH_GPT_numberOfZeros_l1425_142520


namespace NUMINAMATH_GPT_wall_building_time_l1425_142537

theorem wall_building_time (n t : ℕ) (h1 : n * t = 48) (h2 : n = 4) : t = 12 :=
by
  -- appropriate proof steps would go here
  sorry

end NUMINAMATH_GPT_wall_building_time_l1425_142537


namespace NUMINAMATH_GPT_right_triangle_area_l1425_142510

theorem right_triangle_area :
  ∃ (a b c : ℕ), (c^2 = a^2 + b^2) ∧ (2 * b^2 - 23 * b + 11 = 0) ∧ (a * b / 2 = 330) :=
sorry

end NUMINAMATH_GPT_right_triangle_area_l1425_142510


namespace NUMINAMATH_GPT_rate_of_current_l1425_142523

/-- The speed of a boat in still water is 20 km/hr, and the rate of current is c km/hr.
    The distance travelled downstream in 24 minutes is 9.2 km. What is the rate of the current? -/
theorem rate_of_current (c : ℝ) (h : 24/60 = 0.4 ∧ 9.2 = (20 + c) * 0.4) : c = 3 :=
by
  sorry  -- Proof is not required, only the statement is necessary.

end NUMINAMATH_GPT_rate_of_current_l1425_142523


namespace NUMINAMATH_GPT_B_subset_A_implies_m_values_l1425_142553

noncomputable def A : Set ℝ := { x | x^2 + x - 6 = 0 }
noncomputable def B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }
def possible_m_values : Set ℝ := {1/3, -1/2}

theorem B_subset_A_implies_m_values (m : ℝ) : B m ⊆ A → m ∈ possible_m_values := by
  sorry

end NUMINAMATH_GPT_B_subset_A_implies_m_values_l1425_142553


namespace NUMINAMATH_GPT_correct_statements_l1425_142515

-- Definitions based on the conditions and question
def S (n : ℕ) : ℤ := -n^2 + 7 * n + 1

-- Definition of the sequence an
def a (n : ℕ) : ℤ := 
  if n = 1 then 7 
  else S n - S (n - 1)

-- Theorem statements based on the correct answers derived from solution
theorem correct_statements :
  (∀ n : ℕ, n > 4 → a n < 0) ∧ (S 3 = S 4 ∧ (∀ m : ℕ, S m ≤ S 3)) :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_statements_l1425_142515


namespace NUMINAMATH_GPT_ordered_pair_for_quadratic_with_same_roots_l1425_142562

theorem ordered_pair_for_quadratic_with_same_roots (b c : ℝ) :
  (∀ x : ℝ, |x - 4| = 3 ↔ (x = 7 ∨ x = 1)) →
  (∀ x : ℝ, x^2 + b * x + c = 0 ↔ (x = 7 ∨ x = 1)) →
  (b, c) = (-8, 7) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_ordered_pair_for_quadratic_with_same_roots_l1425_142562


namespace NUMINAMATH_GPT_magic_square_l1425_142551

variable (a b c d e s: ℕ)

axiom h1 : 30 + e + 18 = s
axiom h2 : 15 + c + d = s
axiom h3 : a + 27 + b = s
axiom h4 : 30 + 15 + a = s
axiom h5 : e + c + 27 = s
axiom h6 : 18 + d + b = s
axiom h7 : 30 + c + b = s
axiom h8 : a + c + 18 = s

theorem magic_square : d + e = 47 :=
by
  sorry

end NUMINAMATH_GPT_magic_square_l1425_142551


namespace NUMINAMATH_GPT_root_not_less_than_a_l1425_142573

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - x^3

theorem root_not_less_than_a (a b c x0 : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c)
  (h4 : f a * f b * f c < 0) (hx : f x0 = 0) : ¬ (x0 < a) :=
sorry

end NUMINAMATH_GPT_root_not_less_than_a_l1425_142573


namespace NUMINAMATH_GPT_min_expression_n_12_l1425_142578

theorem min_expression_n_12 : ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (n = 12 → (n / 3 + 50 / n ≤ 
                        m / 3 + 50 / m))) :=
by
  sorry

end NUMINAMATH_GPT_min_expression_n_12_l1425_142578


namespace NUMINAMATH_GPT_dynaco_shares_sold_l1425_142599

-- Define the conditions
def MicrotronPrice : ℝ := 36
def DynacoPrice : ℝ := 44
def TotalShares : ℕ := 300
def AvgPrice : ℝ := 40
def TotalValue : ℝ := TotalShares * AvgPrice

-- Define unknown variables
variables (M D : ℕ)

-- Express conditions in Lean
def total_shares_eq : Prop := M + D = TotalShares
def total_value_eq : Prop := MicrotronPrice * M + DynacoPrice * D = TotalValue

-- Define the problem statement
theorem dynaco_shares_sold : ∃ D : ℕ, 
  (∃ M : ℕ, total_shares_eq M D ∧ total_value_eq M D) ∧ D = 150 :=
by
  sorry

end NUMINAMATH_GPT_dynaco_shares_sold_l1425_142599


namespace NUMINAMATH_GPT_bahs_from_yahs_l1425_142581

theorem bahs_from_yahs (b r y : ℝ) 
  (h1 : 18 * b = 30 * r) 
  (h2 : 10 * r = 25 * y) : 
  1250 * y = 300 * b := 
by
  sorry

end NUMINAMATH_GPT_bahs_from_yahs_l1425_142581


namespace NUMINAMATH_GPT_sum_of_extreme_values_of_g_l1425_142567

def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 5) - 2 * abs (x - 3)

theorem sum_of_extreme_values_of_g :
  ∃ (min_val max_val : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 6 → g x ≥ min_val) ∧ 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 6 → g x ≤ max_val) ∧ 
    (min_val = -8) ∧ 
    (max_val = 0) ∧ 
    (min_val + max_val = -8) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_extreme_values_of_g_l1425_142567


namespace NUMINAMATH_GPT_gcf_120_180_240_is_60_l1425_142504

theorem gcf_120_180_240_is_60 : Nat.gcd (Nat.gcd 120 180) 240 = 60 := by
  sorry

end NUMINAMATH_GPT_gcf_120_180_240_is_60_l1425_142504


namespace NUMINAMATH_GPT_triangle_sides_ratios_l1425_142548

theorem triangle_sides_ratios (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b > c) (h₄ : a + c > b) (h₅ : b + c > a) :
  a / (b + c) = b / (a + c) + c / (a + b) :=
sorry

end NUMINAMATH_GPT_triangle_sides_ratios_l1425_142548


namespace NUMINAMATH_GPT_prove_market_demand_prove_tax_revenue_prove_per_unit_tax_rate_prove_tax_revenue_specified_l1425_142529

noncomputable def market_supply_function (P : ℝ) : ℝ := 6 * P - 312

noncomputable def market_demand_function (a b P : ℝ) : ℝ := a - b * P

noncomputable def price_elasticity_supply (P_e Q_e : ℝ) : ℝ := 6 * (P_e / Q_e)

noncomputable def price_elasticity_demand (b P_e Q_e : ℝ) : ℝ := -b * (P_e / Q_e)

noncomputable def tax_rate := 30

noncomputable def consumer_price_after_tax := 118

theorem prove_market_demand (a P_e Q_e : ℝ) :
  1.5 * |price_elasticity_demand 4 P_e Q_e| = price_elasticity_supply P_e Q_e →
  market_demand_function a 4 P_e = a - 4 * P_e := sorry

theorem prove_tax_revenue (Q_d : ℝ) :
  Q_d = 216 →
  Q_d * tax_rate = 6480 := sorry

theorem prove_per_unit_tax_rate (t : ℝ) :
  t = 60 → 4 * t = 240 := sorry

theorem prove_tax_revenue_specified (t : ℝ) :
  t = 60 →
  (288 * t - 2.4 * t^2) = 8640 := sorry

end NUMINAMATH_GPT_prove_market_demand_prove_tax_revenue_prove_per_unit_tax_rate_prove_tax_revenue_specified_l1425_142529


namespace NUMINAMATH_GPT_find_k_and_slope_l1425_142569

theorem find_k_and_slope : 
  ∃ k : ℝ, (∃ y : ℝ, (3 + y = 8) ∧ (k = -3 * 3 + y)) ∧ (k = -4) ∧ 
  (∀ x y : ℝ, (x + y = 8) → (∃ m b : ℝ, y = m * x + b ∧ m = -1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_k_and_slope_l1425_142569


namespace NUMINAMATH_GPT_intersection_reciprocal_sum_l1425_142511

open Real

theorem intersection_reciprocal_sum :
    ∀ (a b : ℝ),
    (∃ x : ℝ, x - 1 = a ∧ 3 / x = b) ∧
    (a * b = 3) →
    ∃ s : ℝ, (s = (a + b) / 3 ∨ s = -(a + b) / 3) ∧ (1 / a + 1 / b = s) := by
  sorry

end NUMINAMATH_GPT_intersection_reciprocal_sum_l1425_142511


namespace NUMINAMATH_GPT_simon_gift_bags_l1425_142503

theorem simon_gift_bags (rate_per_day : ℕ) (days : ℕ) (total_bags : ℕ) :
  rate_per_day = 42 → days = 13 → total_bags = rate_per_day * days → total_bags = 546 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_simon_gift_bags_l1425_142503


namespace NUMINAMATH_GPT_graph_of_4x2_minus_9y2_is_pair_of_straight_lines_l1425_142558

theorem graph_of_4x2_minus_9y2_is_pair_of_straight_lines :
  (∀ x y : ℝ, (4 * x^2 - 9 * y^2 = 0) → (x / y = 3 / 2 ∨ x / y = -3 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_graph_of_4x2_minus_9y2_is_pair_of_straight_lines_l1425_142558


namespace NUMINAMATH_GPT_license_plate_combinations_l1425_142552

-- Definition for the conditions of the problem
def num_license_plate_combinations : ℕ :=
  let num_letters := 26
  let num_digits := 10
  let choose_two_distinct_letters := (num_letters * (num_letters - 1)) / 2
  let arrange_pairs := 2
  let choose_positions := 6
  let digit_permutations := num_digits ^ 2
  choose_two_distinct_letters * arrange_pairs * choose_positions * digit_permutations

-- The theorem we are proving
theorem license_plate_combinations :
  num_license_plate_combinations = 390000 :=
by
  -- The proof would be provided here.
  sorry

end NUMINAMATH_GPT_license_plate_combinations_l1425_142552


namespace NUMINAMATH_GPT_area_difference_l1425_142556

-- Define the areas of individual components
def area_of_square : ℕ := 1
def area_of_small_triangle : ℚ := (1 / 2) * area_of_square
def area_of_large_triangle : ℚ := (1 / 2) * (1 * 2 * area_of_square)

-- Define the total area of the first figure
def first_figure_area : ℚ := 
    8 * area_of_square +
    6 * area_of_small_triangle +
    2 * area_of_large_triangle

-- Define the total area of the second figure
def second_figure_area : ℚ := 
    4 * area_of_square +
    6 * area_of_small_triangle +
    8 * area_of_large_triangle

-- Define the statement to prove the difference in areas
theorem area_difference : second_figure_area - first_figure_area = 2 := by
    -- sorry is used to indicate that the proof is omitted
    sorry

end NUMINAMATH_GPT_area_difference_l1425_142556


namespace NUMINAMATH_GPT_find_cd_l1425_142549

def g (c d x : ℝ) := c * x^3 - 7 * x^2 + d * x - 4

theorem find_cd : ∃ c d : ℝ, (g c d 2 = -4) ∧ (g c d (-1) = -22) ∧ (c = 19/3) ∧ (d = -8/3) := 
by
  sorry

end NUMINAMATH_GPT_find_cd_l1425_142549


namespace NUMINAMATH_GPT_find_mode_l1425_142575

def scores : List ℕ :=
  [105, 107, 111, 111, 112, 112, 115, 118, 123, 124, 124, 126, 127, 129, 129, 129, 130, 130, 130, 130, 131, 140, 140, 140, 140]

def mode (ls : List ℕ) : ℕ :=
  ls.foldl (λmodeScore score => if ls.count score > ls.count modeScore then score else modeScore) 0

theorem find_mode :
  mode scores = 130 :=
by
  sorry

end NUMINAMATH_GPT_find_mode_l1425_142575


namespace NUMINAMATH_GPT_spider_paths_l1425_142582

-- Define the grid points and the binomial coefficient calculation.
def grid_paths (n m : ℕ) : ℕ := Nat.choose (n + m) n

-- The problem statement
theorem spider_paths : grid_paths 4 3 = 35 := by
  sorry

end NUMINAMATH_GPT_spider_paths_l1425_142582


namespace NUMINAMATH_GPT_exists_integers_for_S_geq_100_l1425_142533

theorem exists_integers_for_S_geq_100 (S : ℤ) (hS : S ≥ 100) :
  ∃ (T C B : ℤ) (P : ℤ),
    T > 0 ∧ C > 0 ∧ B > 0 ∧
    T > C ∧ C > B ∧
    T + C + B = S ∧
    T * C * B = P ∧
    (∀ (T₁ C₁ B₁ T₂ C₂ B₂ : ℤ), 
      T₁ > 0 ∧ C₁ > 0 ∧ B₁ > 0 ∧ 
      T₂ > 0 ∧ C₂ > 0 ∧ B₂ > 0 ∧ 
      T₁ > C₁ ∧ C₁ > B₁ ∧ 
      T₂ > C₂ ∧ C₂ > B₂ ∧ 
      T₁ + C₁ + B₁ = S ∧ 
      T₂ + C₂ + B₂ = S ∧ 
      T₁ * C₁ * B₁ = T₂ * C₂ * B₂ → 
      (T₁ = T₂) ∧ (C₁ = C₂) ∧ (B₁ = B₂) → false) :=
sorry

end NUMINAMATH_GPT_exists_integers_for_S_geq_100_l1425_142533


namespace NUMINAMATH_GPT_circle_equation_l1425_142532

theorem circle_equation :
  ∃ x y : ℝ, x = 2 ∧ y = 0 ∧ ∀ (p q : ℝ), ((p - x)^2 + q^2 = 4) ↔ (p^2 + q^2 - 4 * p = 0) :=
sorry

end NUMINAMATH_GPT_circle_equation_l1425_142532


namespace NUMINAMATH_GPT_shaded_area_between_circles_l1425_142541

theorem shaded_area_between_circles (r1 r2 : ℝ) (h1 : r1 = 4) (h2 : r2 = 5)
  (tangent : True) -- This represents that the circles are externally tangent
  (circumscribed : True) -- This represents the third circle circumscribing the two circles
  : ∃ r3 : ℝ, r3 = 9 ∧ π * r3^2 - (π * r1^2 + π * r2^2) = 40 * π :=
  sorry

end NUMINAMATH_GPT_shaded_area_between_circles_l1425_142541


namespace NUMINAMATH_GPT_probability_all_qualified_probability_two_qualified_probability_at_least_one_qualified_l1425_142545

namespace Sprinters

def P_A : ℚ := 2 / 5
def P_B : ℚ := 3 / 4
def P_C : ℚ := 1 / 3

def P_all_qualified := P_A * P_B * P_C
def P_two_qualified := P_A * P_B * (1 - P_C) + P_A * (1 - P_B) * P_C + (1 - P_A) * P_B * P_C
def P_at_least_one_qualified := 1 - (1 - P_A) * (1 - P_B) * (1 - P_C)

theorem probability_all_qualified : P_all_qualified = 1 / 10 :=
by 
  -- proof here
  sorry

theorem probability_two_qualified : P_two_qualified = 23 / 60 :=
by 
  -- proof here
  sorry

theorem probability_at_least_one_qualified : P_at_least_one_qualified = 9 / 10 :=
by 
  -- proof here
  sorry

end Sprinters

end NUMINAMATH_GPT_probability_all_qualified_probability_two_qualified_probability_at_least_one_qualified_l1425_142545


namespace NUMINAMATH_GPT_triangle_side_cube_l1425_142577

theorem triangle_side_cube 
  (a b c : ℕ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_gcd : Nat.gcd a (Nat.gcd b c) = 1)
  (angle_condition : ∃ A B : ℝ, A = 3 * B) 
  : ∃ n m : ℕ, (a = n ^ 3 ∨ b = n ^ 3 ∨ c = n ^ 3) :=
sorry

end NUMINAMATH_GPT_triangle_side_cube_l1425_142577


namespace NUMINAMATH_GPT_license_plate_count_l1425_142583

/-- Number of vowels available for the license plate -/
def num_vowels := 6

/-- Number of consonants available for the license plate -/
def num_consonants := 20

/-- Number of possible digits for the license plate -/
def num_digits := 10

/-- Number of special characters available for the license plate -/
def num_special_chars := 2

/-- Calculate the total number of possible license plates -/
def total_license_plates : Nat :=
  num_vowels * num_consonants * num_digits * num_consonants * num_special_chars

/- Prove that the total number of possible license plates is 48000 -/
theorem license_plate_count : total_license_plates = 48000 :=
  by
    unfold total_license_plates
    sorry

end NUMINAMATH_GPT_license_plate_count_l1425_142583


namespace NUMINAMATH_GPT_number_of_integers_l1425_142595

theorem number_of_integers (n : ℤ) : (200 < n ∧ n < 300 ∧ ∃ r : ℤ, n % 7 = r ∧ n % 9 = r) ↔ 
  n = 252 ∨ n = 253 ∨ n = 254 ∨ n = 255 ∨ n = 256 ∨ n = 257 ∨ n = 258 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_integers_l1425_142595


namespace NUMINAMATH_GPT_temp_on_Monday_l1425_142536

variable (M T W Th F : ℤ)

-- Given conditions
axiom sum_MTWT : M + T + W + Th = 192
axiom sum_TWTF : T + W + Th + F = 184
axiom temp_F : F = 34
axiom exists_day_temp_42 : ∃ (day : String), 
  (day = "Monday" ∨ day = "Tuesday" ∨ day = "Wednesday" ∨ day = "Thursday" ∨ day = "Friday") ∧
  (if day = "Monday" then M else if day = "Tuesday" then T else if day = "Wednesday" then W else if day = "Thursday" then Th else F) = 42

-- Prove temperature of Monday is 42
theorem temp_on_Monday : M = 42 := 
by
  sorry

end NUMINAMATH_GPT_temp_on_Monday_l1425_142536


namespace NUMINAMATH_GPT_bee_honeycomb_path_l1425_142512

theorem bee_honeycomb_path (x1 x2 x3 : ℕ) (honeycomb_grid : Prop)
  (shortest_path : ℕ) (honeycomb_property : shortest_path = 100)
  (path_decomposition : x1 + x2 + x3 = 100) : x1 = 50 ∧ x2 + x3 = 50 := 
sorry

end NUMINAMATH_GPT_bee_honeycomb_path_l1425_142512


namespace NUMINAMATH_GPT_union_of_sets_l1425_142570

-- Define the sets and conditions
variables (a b : ℝ)
variables (A : Set ℝ) (B : Set ℝ)
variables (log2 : ℝ → ℝ)

-- State the assumptions and final proof goal
theorem union_of_sets (h_inter : A ∩ B = {2}) 
                      (h_A : A = {3, log2 a}) 
                      (h_B : B = {a, b}) 
                      (h_log2 : log2 4 = 2) :
  A ∪ B = {2, 3, 4} :=
by {
    sorry
}

end NUMINAMATH_GPT_union_of_sets_l1425_142570


namespace NUMINAMATH_GPT_triangle_final_position_after_rotation_l1425_142559

-- Definitions for the initial conditions
def square_rolls_clockwise_around_octagon : Prop := 
  true -- placeholder definition, assume this defines the motion correctly

def triangle_initial_position : ℕ := 0 -- representing bottom as 0

-- Defining the proof problem
theorem triangle_final_position_after_rotation :
  square_rolls_clockwise_around_octagon →
  triangle_initial_position = 0 →
  triangle_initial_position = 0 :=
by
  intros
  sorry

end NUMINAMATH_GPT_triangle_final_position_after_rotation_l1425_142559


namespace NUMINAMATH_GPT_coordinates_of_F_double_prime_l1425_142538

-- Definitions of transformations
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Definition of initial point F
def F : ℝ × ℝ := (1, 1)

-- Definition of the transformations applied to point F
def F_prime : ℝ × ℝ := reflect_x F
def F_double_prime : ℝ × ℝ := reflect_y_eq_x F_prime

-- Theorem statement
theorem coordinates_of_F_double_prime : F_double_prime = (-1, 1) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_F_double_prime_l1425_142538


namespace NUMINAMATH_GPT_monotone_increasing_interval_l1425_142542

noncomputable def f (x : ℝ) : ℝ := (x / (x^2 + 1)) + 1

theorem monotone_increasing_interval :
  ∀ x : ℝ, (-1 < x ∧ x < 1) ↔ ∀ ε > 0, ∃ δ > 0, ∀ x₁ x₂, (-1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 ∧ |x₁ - x₂| < δ) → f x₁ ≤ f x₂ + ε := 
sorry

end NUMINAMATH_GPT_monotone_increasing_interval_l1425_142542


namespace NUMINAMATH_GPT_even_perfect_square_factors_l1425_142518

theorem even_perfect_square_factors :
  let factors := 2^6 * 5^4 * 7^3
  ∃ (count : ℕ), count = (3 * 3 * 2) ∧
  ∀ (a b c : ℕ), (0 ≤ a ∧ a ≤ 6 ∧ 0 ≤ c ∧ c ≤ 4 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 
  a % 2 = 0 ∧ 2 ≤ a ∧ c % 2 = 0 ∧ b % 2 = 0) → 
  a * b * c < count :=
by
  sorry

end NUMINAMATH_GPT_even_perfect_square_factors_l1425_142518


namespace NUMINAMATH_GPT_problem1_problem2_l1425_142563

-- Problem 1 Lean Statement
theorem problem1 (m n : ℕ) (h1 : 3 ^ m = 6) (h2 : 9 ^ n = 2) : 3 ^ (m - 2 * n) = 3 :=
by
  sorry

-- Problem 2 Lean Statement
theorem problem2 (x : ℝ) (n : ℕ) (h : x ^ (2 * n) = 3) : (x ^ (3 * n)) ^ 2 - (x ^ 2) ^ (2 * n) = 18 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1425_142563


namespace NUMINAMATH_GPT_total_calories_box_l1425_142501

-- Definitions from the conditions
def bags := 6
def cookies_per_bag := 25
def calories_per_cookie := 18

-- Given the conditions, prove the total calories equals 2700
theorem total_calories_box : bags * cookies_per_bag * calories_per_cookie = 2700 := by
  sorry

end NUMINAMATH_GPT_total_calories_box_l1425_142501


namespace NUMINAMATH_GPT_entree_cost_l1425_142519

theorem entree_cost (E : ℝ) :
  let appetizer := 9
  let dessert := 11
  let tip_rate := 0.30
  let total_cost_with_tip := 78
  let total_cost_before_tip := appetizer + 2 * E + dessert
  total_cost_with_tip = total_cost_before_tip + (total_cost_before_tip * tip_rate) →
  E = 20 :=
by
  intros appetizer dessert tip_rate total_cost_with_tip total_cost_before_tip h
  sorry

end NUMINAMATH_GPT_entree_cost_l1425_142519


namespace NUMINAMATH_GPT_possible_values_x_l1425_142565

-- Define the conditions
def gold_coin_worth (x y : ℕ) (g s : ℝ) : Prop :=
  g = (1 + x / 100.0) * s ∧ s = (1 - y / 100.0) * g

-- Define the main theorem statement
theorem possible_values_x : ∀ (x y : ℕ) (g s : ℝ), gold_coin_worth x y g s → 
  (∃ (n : ℕ), n = 12) :=
by
  -- Definitions based on given conditions
  intro x y g s h
  obtain ⟨hx, hy⟩ := h

  -- Placeholder for proof; skip with sorry
  sorry

end NUMINAMATH_GPT_possible_values_x_l1425_142565


namespace NUMINAMATH_GPT_intersection_complement_l1425_142506

open Set

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hA : A = {1, 3, 4})
variable (hB : B = {4, 5})

theorem intersection_complement :
  A ∩ (U \ B) = {1, 3} :=
by
  rw [hU, hA, hB]
  ext
  simp
  sorry

end NUMINAMATH_GPT_intersection_complement_l1425_142506


namespace NUMINAMATH_GPT_max_value_l1425_142598

theorem max_value (a b c : ℕ) (h1 : a = 2^35) (h2 : b = 26) (h3 : c = 1) : max a (max b c) = 2^35 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_max_value_l1425_142598


namespace NUMINAMATH_GPT_problem_value_l1425_142547

theorem problem_value :
  1 - (-2) - 3 - (-4) - 5 - (-6) = 5 :=
by sorry

end NUMINAMATH_GPT_problem_value_l1425_142547


namespace NUMINAMATH_GPT_m_minus_n_is_perfect_square_l1425_142596

theorem m_minus_n_is_perfect_square (m n : ℕ) (h : 0 < m) (h1 : 0 < n) (h2 : 2001 * m^2 + m = 2002 * n^2 + n) : ∃ k : ℕ, m = n + k^2 :=
by
    sorry

end NUMINAMATH_GPT_m_minus_n_is_perfect_square_l1425_142596


namespace NUMINAMATH_GPT_greatest_integer_gcd_30_is_125_l1425_142594

theorem greatest_integer_gcd_30_is_125 : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ ∀ k : ℕ, k < 150 ∧ Nat.gcd k 30 = 5 → k ≤ n := 
sorry

end NUMINAMATH_GPT_greatest_integer_gcd_30_is_125_l1425_142594


namespace NUMINAMATH_GPT_jeremy_sticker_distribution_l1425_142521

def number_of_ways_to_distribute_stickers (total_stickers sheets : ℕ) : ℕ :=
  (Nat.choose (total_stickers - 1) (sheets - 1))

theorem jeremy_sticker_distribution : number_of_ways_to_distribute_stickers 10 3 = 36 :=
by
  sorry

end NUMINAMATH_GPT_jeremy_sticker_distribution_l1425_142521


namespace NUMINAMATH_GPT_compare_a_b_l1425_142589

theorem compare_a_b (a b : ℝ) (h : 5 * (a - 1) = b + a ^ 2) : a > b :=
sorry

end NUMINAMATH_GPT_compare_a_b_l1425_142589


namespace NUMINAMATH_GPT_jake_weight_l1425_142554

theorem jake_weight:
  ∃ (J S : ℝ), (J - 8 = 2 * S) ∧ (J + S = 290) ∧ (J = 196) :=
by
  sorry

end NUMINAMATH_GPT_jake_weight_l1425_142554


namespace NUMINAMATH_GPT_fractional_inspection_l1425_142524

theorem fractional_inspection:
  ∃ (J E A : ℝ),
  J + E + A = 1 ∧
  0.005 * J + 0.007 * E + 0.012 * A = 0.01 :=
by
  sorry

end NUMINAMATH_GPT_fractional_inspection_l1425_142524


namespace NUMINAMATH_GPT_pet_center_final_count_l1425_142534

/-!
# Problem: Count the total number of pets in a pet center after a series of adoption and collection events.
-/

def initialDogs : Nat := 36
def initialCats : Nat := 29
def initialRabbits : Nat := 15
def initialBirds : Nat := 10

def dogsAdopted1 : Nat := 20
def rabbitsAdopted1 : Nat := 5

def catsCollected : Nat := 12
def rabbitsCollected : Nat := 8
def birdsCollected : Nat := 5

def catsAdopted2 : Nat := 10
def birdsAdopted2 : Nat := 4

def finalDogs : Nat :=
  initialDogs - dogsAdopted1

def finalCats : Nat :=
  initialCats + catsCollected - catsAdopted2

def finalRabbits : Nat :=
  initialRabbits - rabbitsAdopted1 + rabbitsCollected

def finalBirds : Nat :=
  initialBirds + birdsCollected - birdsAdopted2

def totalPets (d c r b : Nat) : Nat :=
  d + c + r + b

theorem pet_center_final_count : 
  totalPets finalDogs finalCats finalRabbits finalBirds = 76 := by
  -- This is where we would provide the proof, but it's skipped as per the instructions.
  sorry

end NUMINAMATH_GPT_pet_center_final_count_l1425_142534


namespace NUMINAMATH_GPT_polynomial_equality_l1425_142505

theorem polynomial_equality (x y : ℝ) (h₁ : 3 * x + 2 * y = 6) (h₂ : 2 * x + 3 * y = 7) : 
  14 * x^2 + 25 * x * y + 14 * y^2 = 85 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_equality_l1425_142505


namespace NUMINAMATH_GPT_yellow_balloons_ratio_l1425_142585

theorem yellow_balloons_ratio 
  (total_balloons : ℕ) 
  (colors : ℕ) 
  (yellow_balloons_taken : ℕ) 
  (h_total_balloons : total_balloons = 672)
  (h_colors : colors = 4)
  (h_yellow_balloons_taken : yellow_balloons_taken = 84) :
  yellow_balloons_taken / (total_balloons / colors) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_yellow_balloons_ratio_l1425_142585


namespace NUMINAMATH_GPT_inf_solutions_l1425_142539

theorem inf_solutions (x y z : ℤ) : 
  ∃ (infinitely many relatively prime solutions : ℕ), x^2 + y^2 = z^5 + z :=
sorry

end NUMINAMATH_GPT_inf_solutions_l1425_142539


namespace NUMINAMATH_GPT_sum_of_digits_of_N_l1425_142514

theorem sum_of_digits_of_N :
  ∃ N : ℕ, 
    10 ≤ N ∧ N < 100 ∧
    5655 % N = 11 ∧ 
    5879 % N = 14 ∧ 
    ((N / 10) + (N % 10)) = 8 := 
sorry

end NUMINAMATH_GPT_sum_of_digits_of_N_l1425_142514


namespace NUMINAMATH_GPT_greatest_number_of_bouquets_l1425_142544

def cherry_lollipops := 4
def orange_lollipops := 6
def raspberry_lollipops := 8
def lemon_lollipops := 10
def candy_canes := 12
def chocolate_coins := 14

theorem greatest_number_of_bouquets : 
  Nat.gcd cherry_lollipops (Nat.gcd orange_lollipops (Nat.gcd raspberry_lollipops (Nat.gcd lemon_lollipops (Nat.gcd candy_canes chocolate_coins)))) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_greatest_number_of_bouquets_l1425_142544


namespace NUMINAMATH_GPT_monotonic_increasing_f_C_l1425_142586

noncomputable def f_A (x : ℝ) : ℝ := -Real.log x
noncomputable def f_B (x : ℝ) : ℝ := 1 / (2^x)
noncomputable def f_C (x : ℝ) : ℝ := -(1 / x)
noncomputable def f_D (x : ℝ) : ℝ := 3^(abs (x - 1))

theorem monotonic_increasing_f_C : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f_C x < f_C y :=
sorry

end NUMINAMATH_GPT_monotonic_increasing_f_C_l1425_142586


namespace NUMINAMATH_GPT_tony_schooling_years_l1425_142509

theorem tony_schooling_years:
  let first_degree := 4
  let additional_degrees := 2 * 4
  let graduate_degree := 2
  first_degree + additional_degrees + graduate_degree = 14 :=
by {
  let first_degree := 4
  let additional_degrees := 2 * 4
  let graduate_degree := 2
  show first_degree + additional_degrees + graduate_degree = 14
  sorry
}

end NUMINAMATH_GPT_tony_schooling_years_l1425_142509


namespace NUMINAMATH_GPT_total_volume_of_mixed_solutions_l1425_142576

theorem total_volume_of_mixed_solutions :
  let v1 := 3.6
  let v2 := 1.4
  v1 + v2 = 5.0 := by
  sorry

end NUMINAMATH_GPT_total_volume_of_mixed_solutions_l1425_142576


namespace NUMINAMATH_GPT_brown_ball_weight_l1425_142550

def total_weight : ℝ := 9.12
def weight_blue : ℝ := 6
def weight_brown : ℝ := 3.12

theorem brown_ball_weight : total_weight - weight_blue = weight_brown :=
by 
  sorry

end NUMINAMATH_GPT_brown_ball_weight_l1425_142550


namespace NUMINAMATH_GPT_solve_special_sequence_l1425_142590

noncomputable def special_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1010 ∧ a 2 = 1015 ∧ ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = 2 * n + 1

theorem solve_special_sequence :
  ∃ a : ℕ → ℕ, special_sequence a ∧ a 1000 = 1676 :=
by
  sorry

end NUMINAMATH_GPT_solve_special_sequence_l1425_142590


namespace NUMINAMATH_GPT_percentage_not_pens_pencils_erasers_l1425_142502

-- Define the given percentages
def percentPens : ℝ := 42
def percentPencils : ℝ := 25
def percentErasers : ℝ := 12
def totalPercent : ℝ := 100

-- The goal is to prove that the percentage of sales that were not pens, pencils, or erasers is 21%
theorem percentage_not_pens_pencils_erasers :
  totalPercent - (percentPens + percentPencils + percentErasers) = 21 := by
  sorry

end NUMINAMATH_GPT_percentage_not_pens_pencils_erasers_l1425_142502


namespace NUMINAMATH_GPT_sequence_x_sequence_y_sequence_z_sequence_t_l1425_142591

theorem sequence_x (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (n^2 + n = 2) else 
   if n = 2 then (n^2 + n = 6) else 
   if n = 3 then (n^2 + n = 12) else 
   if n = 4 then (n^2 + n = 20) else true) := 
by sorry

theorem sequence_y (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (2 * n^2 = 2) else 
   if n = 2 then (2 * n^2 = 8) else 
   if n = 3 then (2 * n^2 = 18) else 
   if n = 4 then (2 * n^2 = 32) else true) := 
by sorry

theorem sequence_z (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (n^3 = 1) else 
   if n = 2 then (n^3 = 8) else 
   if n = 3 then (n^3 = 27) else 
   if n = 4 then (n^3 = 64) else true) := 
by sorry

theorem sequence_t (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (2^n = 2) else 
   if n = 2 then (2^n = 4) else 
   if n = 3 then (2^n = 8) else 
   if n = 4 then (2^n = 16) else true) := 
by sorry

end NUMINAMATH_GPT_sequence_x_sequence_y_sequence_z_sequence_t_l1425_142591


namespace NUMINAMATH_GPT_largest_n_satisfying_ineq_l1425_142528
  
theorem largest_n_satisfying_ineq : ∃ n : ℕ, (n < 10) ∧ ∀ m : ℕ, (m < 10) → m ≤ n ∧ (n < 10) ∧ (m < 10) → n = 9 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_satisfying_ineq_l1425_142528


namespace NUMINAMATH_GPT_running_speed_proof_l1425_142592

-- Definitions used in the conditions
def num_people : ℕ := 4
def stretch_km : ℕ := 300
def bike_speed_kmph : ℕ := 50
def total_time_hours : ℚ := 19 + (1/3)

-- The running speed to be proven
def running_speed_kmph : ℚ := 15.52

-- The main statement
theorem running_speed_proof
  (num_people_eq : num_people = 4)
  (stretch_eq : stretch_km = 300)
  (bike_speed_eq : bike_speed_kmph = 50)
  (total_time_eq : total_time_hours = 19.333333333333332) :
  running_speed_kmph = 15.52 :=
sorry

end NUMINAMATH_GPT_running_speed_proof_l1425_142592


namespace NUMINAMATH_GPT_triangle_area_formula_l1425_142513

theorem triangle_area_formula (a b c R : ℝ) (α β γ : ℝ) 
    (h1 : a / (Real.sin α) = 2 * R) 
    (h2 : b / (Real.sin β) = 2 * R) 
    (h3 : c / (Real.sin γ) = 2 * R) :
    let S := (1 / 2) * a * b * (Real.sin γ)
    S = a * b * c / (4 * R) := 
by 
  sorry

end NUMINAMATH_GPT_triangle_area_formula_l1425_142513


namespace NUMINAMATH_GPT_carol_packs_l1425_142500

theorem carol_packs (n_invites n_per_pack : ℕ) (h1 : n_invites = 12) (h2 : n_per_pack = 4) : n_invites / n_per_pack = 3 :=
by
  sorry

end NUMINAMATH_GPT_carol_packs_l1425_142500


namespace NUMINAMATH_GPT_fruit_basket_cost_is_28_l1425_142587

def basket_total_cost : ℕ := 4 * 1 + 3 * 2 + (24 / 12) * 4 + 2 * 3 + 2 * 2

theorem fruit_basket_cost_is_28 : basket_total_cost = 28 := by
  sorry

end NUMINAMATH_GPT_fruit_basket_cost_is_28_l1425_142587


namespace NUMINAMATH_GPT_expression_evaluation_l1425_142557

noncomputable def evaluate_expression (a b c : ℚ) : ℚ :=
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 9) / (c + 7)

theorem expression_evaluation : 
  ∀ (a b c : ℚ), c = b - 11 → b = a + 3 → a = 5 → 
  (a + 2) ≠ 0 → (b - 3) ≠ 0 → (c + 7) ≠ 0 → 
  evaluate_expression a b c = 72 / 35 :=
by
  intros a b c hc hb ha h1 h2 h3
  rw [ha, hb, hc, evaluate_expression]
  -- The proof is not required.
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1425_142557


namespace NUMINAMATH_GPT_sequence_of_8_numbers_l1425_142530

theorem sequence_of_8_numbers :
  ∃ (a b c d e f g h : ℤ), 
    a + b + c = 100 ∧ b + c + d = 100 ∧ c + d + e = 100 ∧ 
    d + e + f = 100 ∧ e + f + g = 100 ∧ f + g + h = 100 ∧ 
    a = 20 ∧ h = 16 ∧ 
    (a, b, c, d, e, f, g, h) = (20, 16, 64, 20, 16, 64, 20, 16) :=
by
  sorry

end NUMINAMATH_GPT_sequence_of_8_numbers_l1425_142530


namespace NUMINAMATH_GPT_expanded_form_correct_l1425_142543

theorem expanded_form_correct :
  (∃ a b c : ℤ, (∀ x : ℚ, 2 * (x - 3)^2 - 12 = a * x^2 + b * x + c) ∧ (10 * a - b - 4 * c = 8)) :=
by
  sorry

end NUMINAMATH_GPT_expanded_form_correct_l1425_142543


namespace NUMINAMATH_GPT_find_y_intercept_of_second_parabola_l1425_142507

theorem find_y_intercept_of_second_parabola :
  ∃ D : ℝ × ℝ, D = (0, 9) ∧ 
    (∃ A : ℝ × ℝ, A = (10, 4) ∧ 
     ∃ B : ℝ × ℝ, B = (6, 0) ∧ 
     (∀ x y : ℝ, y = (-1/4) * x ^ 2 + 5 * x - 21 → A = (10, 4)) ∧ 
     (∀ x y : ℝ, y = (1/4) * (x - B.1) ^ 2 + B.2 ∧ y = 4 ∧ B = (6, 0) → A = (10, 4))) :=
  sorry

end NUMINAMATH_GPT_find_y_intercept_of_second_parabola_l1425_142507


namespace NUMINAMATH_GPT_max_discount_rate_l1425_142526

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end NUMINAMATH_GPT_max_discount_rate_l1425_142526


namespace NUMINAMATH_GPT_min_abs_difference_on_hyperbola_l1425_142566

theorem min_abs_difference_on_hyperbola : 
  ∀ (x y : ℝ), (x^2 / 8 - y^2 / 4 = 1) → abs (x - y) ≥ 2 := 
by
  intros x y hxy
  sorry

end NUMINAMATH_GPT_min_abs_difference_on_hyperbola_l1425_142566


namespace NUMINAMATH_GPT_sum_n_k_eq_eight_l1425_142579

theorem sum_n_k_eq_eight (n k : Nat) (h1 : 4 * k = n - 3) (h2 : 8 * k + 13 = 3 * n) : n + k = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_n_k_eq_eight_l1425_142579


namespace NUMINAMATH_GPT_transaction_gain_per_year_l1425_142540

theorem transaction_gain_per_year
  (principal : ℝ) (borrow_rate : ℝ) (lend_rate : ℝ) (time : ℕ)
  (principal_eq : principal = 5000)
  (borrow_rate_eq : borrow_rate = 0.04)
  (lend_rate_eq : lend_rate = 0.06)
  (time_eq : time = 2) :
  (principal * lend_rate * time - principal * borrow_rate * time) / time = 100 := by
  sorry

end NUMINAMATH_GPT_transaction_gain_per_year_l1425_142540


namespace NUMINAMATH_GPT_time_to_fill_pool_l1425_142572

theorem time_to_fill_pool :
  ∀ (total_volume : ℝ) (filling_rate : ℝ) (leaking_rate : ℝ),
  total_volume = 60 →
  filling_rate = 1.6 →
  leaking_rate = 0.1 →
  (total_volume / (filling_rate - leaking_rate)) = 40 :=
by
  intros total_volume filling_rate leaking_rate hv hf hl
  rw [hv, hf, hl]
  sorry

end NUMINAMATH_GPT_time_to_fill_pool_l1425_142572


namespace NUMINAMATH_GPT_max_regions_by_five_lines_l1425_142568

theorem max_regions_by_five_lines : 
  ∀ (R : ℕ → ℕ), R 1 = 2 → R 2 = 4 → (∀ n, R (n + 1) = R n + (n + 1)) → R 5 = 16 :=
by
  intros R hR1 hR2 hRec
  sorry

end NUMINAMATH_GPT_max_regions_by_five_lines_l1425_142568


namespace NUMINAMATH_GPT_zero_of_f_l1425_142593

noncomputable def f (x : ℝ) : ℝ := Real.logb 5 (x - 1)

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = 2 :=
by
  use 2
  unfold f
  sorry -- Skip the proof steps, as instructed.

end NUMINAMATH_GPT_zero_of_f_l1425_142593


namespace NUMINAMATH_GPT_sum_of_cubes_l1425_142522

theorem sum_of_cubes (k : ℤ) : 
  24 * k = (k + 2)^3 + (-k)^3 + (-k)^3 + (k - 2)^3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l1425_142522
