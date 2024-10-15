import Mathlib

namespace NUMINAMATH_GPT_width_of_cistern_is_6_l579_57974

-- Length of the cistern
def length : ℝ := 8

-- Breadth of the water surface
def breadth : ℝ := 1.85

-- Total wet surface area
def total_wet_surface_area : ℝ := 99.8

-- Let w be the width of the cistern
def width (w : ℝ) : Prop :=
  total_wet_surface_area = (length * w) + 2 * (length * breadth) + 2 * (w * breadth)

theorem width_of_cistern_is_6 : width 6 :=
  by
    -- This proof is omitted. The statement asserts that the width is 6 meters.
    sorry

end NUMINAMATH_GPT_width_of_cistern_is_6_l579_57974


namespace NUMINAMATH_GPT_sin_C_value_l579_57965

theorem sin_C_value (A B C : Real) (AC BC : Real) (h_AC : AC = 3) (h_BC : BC = 2 * Real.sqrt 3) (h_A : A = 2 * B) :
    let C : Real := Real.pi - A - B
    Real.sin C = Real.sqrt 6 / 9 :=
  sorry

end NUMINAMATH_GPT_sin_C_value_l579_57965


namespace NUMINAMATH_GPT_largest_even_number_in_sequence_of_six_l579_57900

-- Definitions and conditions
def smallest_even_number (x : ℤ) : Prop :=
  x + (x + 2) + (x+4) + (x+6) + (x + 8) + (x + 10) = 540

def sum_of_squares_of_sequence (x : ℤ) : Prop :=
  x^2 + (x + 2)^2 + (x + 4)^2 + (x + 6)^2 + (x + 8)^2 + (x + 10)^2 = 97920

-- Statement to prove
theorem largest_even_number_in_sequence_of_six (x : ℤ) (h1 : smallest_even_number x) (h2 : sum_of_squares_of_sequence x) : x + 10 = 95 :=
  sorry

end NUMINAMATH_GPT_largest_even_number_in_sequence_of_six_l579_57900


namespace NUMINAMATH_GPT_sum_of_a_b_l579_57956

def symmetric_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

theorem sum_of_a_b (a b : ℝ) (h : symmetric_x_axis (3, a) (b, 4)) : a + b = -1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_a_b_l579_57956


namespace NUMINAMATH_GPT_max_nine_multiple_l579_57942

theorem max_nine_multiple {a b c n : ℕ} (h1 : Prime a) (h2 : Prime b) (h3 : Prime c) (h4 : 3 < a) (h5 : 3 < b) (h6 : 3 < c) (h7 : 2 * a + 5 * b = c) : 9 ∣ (a + b + c) :=
sorry

end NUMINAMATH_GPT_max_nine_multiple_l579_57942


namespace NUMINAMATH_GPT_average_and_variance_of_new_data_set_l579_57941

theorem average_and_variance_of_new_data_set
  (avg : ℝ) (var : ℝ) (constant : ℝ)
  (h_avg : avg = 2.8)
  (h_var : var = 3.6)
  (h_const : constant = 60) :
  (avg + constant = 62.8) ∧ (var = 3.6) :=
sorry

end NUMINAMATH_GPT_average_and_variance_of_new_data_set_l579_57941


namespace NUMINAMATH_GPT_mark_old_bills_l579_57922

noncomputable def old_hourly_wage : ℝ := 40
noncomputable def new_hourly_wage : ℝ := 42
noncomputable def work_hours_per_week : ℝ := 8 * 5
noncomputable def personal_trainer_cost_per_week : ℝ := 100
noncomputable def leftover_after_expenses : ℝ := 980

noncomputable def new_weekly_earnings := new_hourly_wage * work_hours_per_week
noncomputable def total_weekly_spending_after_raise := leftover_after_expenses + personal_trainer_cost_per_week
noncomputable def old_bills_per_week := new_weekly_earnings - total_weekly_spending_after_raise

theorem mark_old_bills : old_bills_per_week = 600 := by
  sorry

end NUMINAMATH_GPT_mark_old_bills_l579_57922


namespace NUMINAMATH_GPT_value_of_r_l579_57950

theorem value_of_r (n : ℕ) (h : n = 3) : 
  let s := 2^n - 1
  let r := 4^s - s
  r = 16377 := by
  let s := 2^3 - 1
  let r := 4^s - s
  sorry

end NUMINAMATH_GPT_value_of_r_l579_57950


namespace NUMINAMATH_GPT_basket_weight_l579_57937

variables 
  (B : ℕ) -- Weight of the basket
  (L : ℕ) -- Lifting capacity of one balloon

-- Condition: One balloon can lift a basket with contents weighing not more than 80 kg
axiom one_balloon_lifts (h1 : B + L ≤ 80) : Prop

-- Condition: Two balloons can lift a basket with contents weighing not more than 180 kg
axiom two_balloons_lift (h2 : B + 2 * L ≤ 180) : Prop

-- The proof problem: Determine B under the given conditions
theorem basket_weight (B : ℕ) (L : ℕ) (h1 : B + L ≤ 80) (h2 : B + 2 * L ≤ 180) : B = 20 :=
  sorry

end NUMINAMATH_GPT_basket_weight_l579_57937


namespace NUMINAMATH_GPT_part1_part2_part3_l579_57924

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem part1 : determinant (-3) (-2) 4 5 = -7 := by
  sorry

theorem part2 (x: ℝ) (h: determinant 2 (-2 * x) 3 (-5 * x) = 2) : x = -1/2 := by
  sorry

theorem part3 (m n x: ℝ) 
  (h1: determinant (8 * m * x - 1) (-8/3 + 2 * x) (3/2) (-3) = 
        determinant 6 (-1) (-n) x) : 
    m = -3/8 ∧ n = -7 := by
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l579_57924


namespace NUMINAMATH_GPT_floor_neg_sqrt_eval_l579_57972

theorem floor_neg_sqrt_eval :
  ⌊-(Real.sqrt (64 / 9))⌋ = -3 :=
by
  sorry

end NUMINAMATH_GPT_floor_neg_sqrt_eval_l579_57972


namespace NUMINAMATH_GPT_sum_of_remainders_l579_57970

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 9) : (n % 4 + n % 5) = 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l579_57970


namespace NUMINAMATH_GPT_find_n_value_l579_57969

theorem find_n_value (n : ℕ) (h : 2^6 * 3^3 * n = Nat.factorial 9) : n = 210 := sorry

end NUMINAMATH_GPT_find_n_value_l579_57969


namespace NUMINAMATH_GPT_largest_r_l579_57994

theorem largest_r (a : ℕ → ℕ) (h : ∀ n, 0 < a n ∧ a n ≤ a (n + 2) ∧ a (n + 2) ≤ Int.sqrt (a n ^ 2 + 2 * a (n + 1))) :
  ∃ M, ∀ n ≥ M, a (n + 2) = a n :=
sorry

end NUMINAMATH_GPT_largest_r_l579_57994


namespace NUMINAMATH_GPT_abs_condition_l579_57952

theorem abs_condition (x : ℝ) : |2 * x - 7| ≤ 0 ↔ x = 7 / 2 := 
by
  sorry

end NUMINAMATH_GPT_abs_condition_l579_57952


namespace NUMINAMATH_GPT_jesse_stamps_l579_57947

variable (A E : Nat)

theorem jesse_stamps :
  E = 3 * A ∧ E + A = 444 → E = 333 :=
by
  sorry

end NUMINAMATH_GPT_jesse_stamps_l579_57947


namespace NUMINAMATH_GPT_carl_weight_l579_57978

variable (Al Ben Carl Ed : ℝ)

axiom h1 : Ed = 146
axiom h2 : Ed + 38 = Al
axiom h3 : Al = Ben + 25
axiom h4 : Ben = Carl - 16

theorem carl_weight : Carl = 175 :=
by
  sorry

end NUMINAMATH_GPT_carl_weight_l579_57978


namespace NUMINAMATH_GPT_parabola_circle_intersection_radius_squared_l579_57993

theorem parabola_circle_intersection_radius_squared :
  (∀ x y, y = (x - 2)^2 → x + 1 = (y + 2)^2 → (x - 1)^2 + (y + 1)^2 = 1) :=
sorry

end NUMINAMATH_GPT_parabola_circle_intersection_radius_squared_l579_57993


namespace NUMINAMATH_GPT_truck_gasoline_rate_l579_57935

theorem truck_gasoline_rate (gas_initial gas_final : ℕ) (dist_supermarket dist_farm_turn dist_farm_final : ℕ) 
    (total_miles gas_used : ℕ) : 
  gas_initial = 12 →
  gas_final = 2 →
  dist_supermarket = 10 →
  dist_farm_turn = 4 →
  dist_farm_final = 6 →
  total_miles = dist_supermarket + dist_farm_turn + dist_farm_final →
  gas_used = gas_initial - gas_final →
  total_miles / gas_used = 2 :=
by sorry

end NUMINAMATH_GPT_truck_gasoline_rate_l579_57935


namespace NUMINAMATH_GPT_product_of_ratios_l579_57929

theorem product_of_ratios:
  ∀ (x1 y1 x2 y2 x3 y3 : ℝ),
    (x1^3 - 3 * x1 * y1^2 = 2023) ∧ (y1^3 - 3 * x1^2 * y1 = 2022) →
    (x2^3 - 3 * x2 * y2^2 = 2023) ∧ (y2^3 - 3 * x2^2 * y2 = 2022) →
    (x3^3 - 3 * x3 * y3^2 = 2023) ∧ (y3^3 - 3 * x3^2 * y3 = 2022) →
    (1 - x1/y1) * (1 - x2/y2) * (1 - x3/y3) = 1 / 2023 :=
by
  intros x1 y1 x2 y2 x3 y3
  sorry

end NUMINAMATH_GPT_product_of_ratios_l579_57929


namespace NUMINAMATH_GPT_mod_17_residue_l579_57917

theorem mod_17_residue : (255 + 7 * 51 + 9 * 187 + 5 * 34) % 17 = 0 := 
  by sorry

end NUMINAMATH_GPT_mod_17_residue_l579_57917


namespace NUMINAMATH_GPT_exists_sequence_l579_57989

theorem exists_sequence (n : ℕ) : ∃ (a : ℕ → ℕ), 
  (∀ i, 1 ≤ i → i < n → (a i > a (i + 1))) ∧
  (∀ i, 1 ≤ i → i < n → (a i ∣ a (i + 1)^2)) ∧
  (∀ i j, 1 ≤ i → 1 ≤ j → i < n → j < n → (i ≠ j → ¬(a i ∣ a j))) :=
sorry

end NUMINAMATH_GPT_exists_sequence_l579_57989


namespace NUMINAMATH_GPT_exists_positive_integer_m_l579_57925

theorem exists_positive_integer_m (n : ℕ) (hn : 0 < n) : ∃ m : ℕ, 0 < m ∧ 7^n ∣ (3^m + 5^m - 1) :=
sorry

end NUMINAMATH_GPT_exists_positive_integer_m_l579_57925


namespace NUMINAMATH_GPT_correct_calculation_l579_57995

theorem correct_calculation (m n : ℝ) :
  3 * m^2 * n - 3 * m^2 * n = 0 ∧
  ¬ (3 * m^2 - 2 * m^2 = 1) ∧
  ¬ (3 * m^2 + 2 * m^2 = 5 * m^4) ∧
  ¬ (3 * m + 2 * n = 5 * m * n) := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l579_57995


namespace NUMINAMATH_GPT_smallest_sum_of_squares_l579_57963

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 175) : 
  ∃ (x y : ℤ), x^2 - y^2 = 175 ∧ x^2 + y^2 = 625 :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_squares_l579_57963


namespace NUMINAMATH_GPT_converse_and_inverse_l579_57954

-- Definitions
def is_circle (s : Type) : Prop := sorry
def has_no_corners (s : Type) : Prop := sorry

-- Converse Statement
def converse_false (s : Type) : Prop :=
  has_no_corners s → is_circle s → False

-- Inverse Statement
def inverse_true (s : Type) : Prop :=
  ¬ is_circle s → ¬ has_no_corners s

-- Main Proof Problem
theorem converse_and_inverse (s : Type) :
  (converse_false s) ∧ (inverse_true s) := sorry

end NUMINAMATH_GPT_converse_and_inverse_l579_57954


namespace NUMINAMATH_GPT_length_of_AE_l579_57920

variable (A B C D E : Type) [AddGroup A]
variable (AB CD AC AE EC : ℝ)
variable 
  (hAB : AB = 8)
  (hCD : CD = 18)
  (hAC : AC = 20)
  (hEqualAreas : ∀ (AED BEC : Type), (area AED = area BEC) → (AED = BEC))

theorem length_of_AE (hRatio : AE / EC = 4 / 9) (hSum : AC = AE + EC) : AE = 80 / 13 :=
by
  sorry

end NUMINAMATH_GPT_length_of_AE_l579_57920


namespace NUMINAMATH_GPT_gcd_a_b_eq_one_l579_57975

def a : ℕ := 130^2 + 240^2 + 350^2
def b : ℕ := 131^2 + 241^2 + 349^2

theorem gcd_a_b_eq_one : Nat.gcd a b = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_a_b_eq_one_l579_57975


namespace NUMINAMATH_GPT_b_share_is_approx_1885_71_l579_57909

noncomputable def investment_problem (x : ℝ) : ℝ := 
  let c_investment := x
  let b_investment := (2 / 3) * c_investment
  let a_investment := 3 * b_investment
  let total_investment := a_investment + b_investment + c_investment
  let b_share := (b_investment / total_investment) * 6600
  b_share

theorem b_share_is_approx_1885_71 (x : ℝ) : abs (investment_problem x - 1885.71) < 0.01 := sorry

end NUMINAMATH_GPT_b_share_is_approx_1885_71_l579_57909


namespace NUMINAMATH_GPT_FourConsecIntsSum34Unique_l579_57998

theorem FourConsecIntsSum34Unique :
  ∃! (a b c d : ℕ), (a < b) ∧ (b < c) ∧ (c < d) ∧ (a + b + c + d = 34) ∧ (d = a + 3) :=
by
  -- The proof will be placed here
  sorry

end NUMINAMATH_GPT_FourConsecIntsSum34Unique_l579_57998


namespace NUMINAMATH_GPT_parallel_lines_a_value_l579_57907

theorem parallel_lines_a_value (a : ℝ) 
  (h1 : ∀ x y : ℝ, x + a * y - 1 = 0 → x = a * (-4 * y - 2)) 
  : a = 2 :=
sorry

end NUMINAMATH_GPT_parallel_lines_a_value_l579_57907


namespace NUMINAMATH_GPT_sum_of_squares_eq_1850_l579_57930

-- Assuming definitions for the rates
variables (b j s h : ℕ)

-- Condition from Ed's activity
axiom ed_condition : 3 * b + 4 * j + 2 * s + 3 * h = 120

-- Condition from Sue's activity
axiom sue_condition : 2 * b + 3 * j + 4 * s + 3 * h = 150

-- Sum of squares of biking, jogging, swimming, and hiking rates
def sum_of_squares (b j s h : ℕ) : ℕ := b^2 + j^2 + s^2 + h^2

-- Assertion we want to prove
theorem sum_of_squares_eq_1850 :
  ∃ b j s h : ℕ, 3 * b + 4 * j + 2 * s + 3 * h = 120 ∧ 2 * b + 3 * j + 4 * s + 3 * h = 150 ∧ sum_of_squares b j s h = 1850 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_eq_1850_l579_57930


namespace NUMINAMATH_GPT_smallest_positive_cube_ends_in_112_l579_57908

theorem smallest_positive_cube_ends_in_112 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 112 ∧ (∀ m : ℕ, (m > 0 ∧ m^3 % 1000 = 112) → n ≤ m) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_cube_ends_in_112_l579_57908


namespace NUMINAMATH_GPT_irreducible_fraction_l579_57964

theorem irreducible_fraction (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 :=
by
  sorry

end NUMINAMATH_GPT_irreducible_fraction_l579_57964


namespace NUMINAMATH_GPT_walter_time_spent_at_seals_l579_57923

theorem walter_time_spent_at_seals (S : ℕ) 
(h1 : 8 * S + S + 13 = 130) : S = 13 :=
sorry

end NUMINAMATH_GPT_walter_time_spent_at_seals_l579_57923


namespace NUMINAMATH_GPT_percent_problem_l579_57905

theorem percent_problem (x : ℝ) (h : 0.20 * x = 1000) : 1.20 * x = 6000 := by
  sorry

end NUMINAMATH_GPT_percent_problem_l579_57905


namespace NUMINAMATH_GPT_eva_total_marks_correct_l579_57957

-- Definitions based on conditions
def math_marks_second_sem : ℕ := 80
def arts_marks_second_sem : ℕ := 90
def science_marks_second_sem : ℕ := 90

def math_marks_first_sem : ℕ := math_marks_second_sem + 10
def arts_marks_first_sem : ℕ := arts_marks_second_sem - 15
def science_marks_first_sem : ℕ := science_marks_second_sem - (science_marks_second_sem / 3)

def total_marks_first_sem : ℕ := math_marks_first_sem + arts_marks_first_sem + science_marks_first_sem
def total_marks_second_sem : ℕ := math_marks_second_sem + arts_marks_second_sem + science_marks_second_sem

def total_marks_both_sems : ℕ := total_marks_first_sem + total_marks_second_sem

-- Theorem to be proved
theorem eva_total_marks_correct : total_marks_both_sems = 485 := by
  -- Here, we state that we need to prove the total marks sum up to 485
  sorry

end NUMINAMATH_GPT_eva_total_marks_correct_l579_57957


namespace NUMINAMATH_GPT_max_sinA_cosB_cosC_l579_57945

theorem max_sinA_cosB_cosC (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 0 < A ∧ A < 180) (h3 : 0 < B ∧ B < 180) (h4 : 0 < C ∧ C < 180) : 
  ∃ M : ℝ, M = (1 + Real.sqrt 5) / 2 ∧ ∀ a b c : ℝ, a + b + c = 180 → 0 < a ∧ a < 180 → 0 < b ∧ b < 180 → 0 < c ∧ c < 180 → (Real.sin a + Real.cos b * Real.cos c) ≤ M :=
by sorry

end NUMINAMATH_GPT_max_sinA_cosB_cosC_l579_57945


namespace NUMINAMATH_GPT_range_of_m_l579_57960

-- Definitions and conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def is_eccentricity (e a b : ℝ) : Prop :=
  e = Real.sqrt (1 - (b^2 / a^2))

def is_semi_latus_rectum (d a b : ℝ) : Prop :=
  d = 2 * b^2 / a

-- Main theorem statement
theorem range_of_m (a b m : ℝ) (x y : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0)
  (h3 : is_eccentricity (Real.sqrt (3) / 2) a b)
  (h4 : is_semi_latus_rectum 1 a b)
  (h_ellipse : ellipse a b x y) : 
  m ∈ Set.Ioo (-3 / 2 : ℝ) (3 / 2 : ℝ) := 
sorry

end NUMINAMATH_GPT_range_of_m_l579_57960


namespace NUMINAMATH_GPT_triangle_angle_contradiction_l579_57968

theorem triangle_angle_contradiction (A B C : ℝ) (hA : 60 < A) (hB : 60 < B) (hC : 60 < C) (h_sum : A + B + C = 180) : false :=
by {
  -- This would be the proof part, which we don't need to detail according to the instructions.
  sorry
}

end NUMINAMATH_GPT_triangle_angle_contradiction_l579_57968


namespace NUMINAMATH_GPT_hh3_eq_2943_l579_57933

-- Define the function h
def h (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 2

-- Prove that h(h(3)) = 2943
theorem hh3_eq_2943 : h (h 3) = 2943 :=
by
  sorry

end NUMINAMATH_GPT_hh3_eq_2943_l579_57933


namespace NUMINAMATH_GPT_sport_formulation_water_quantity_l579_57987

theorem sport_formulation_water_quantity (flavoring : ℝ) (corn_syrup : ℝ) (water : ℝ)
    (hs : flavoring / corn_syrup = 1 / 12) 
    (hw : flavoring / water = 1 / 30) 
    (sport_fs_ratio : flavoring / corn_syrup = 3 * (1 / 12)) 
    (sport_fw_ratio : flavoring / water = (1 / 2) * (1 / 30)) 
    (cs_sport : corn_syrup = 1) : 
    water = 15 :=
by
  sorry

end NUMINAMATH_GPT_sport_formulation_water_quantity_l579_57987


namespace NUMINAMATH_GPT_city_renumbering_not_possible_l579_57991

-- Defining the problem conditions
def city_renumbering_invalid (city_graph : Type) (connected : city_graph → city_graph → Prop) : Prop :=
  ∃ (M N : city_graph), ∀ (renumber : city_graph → city_graph),
  (renumber M = N ∧ renumber N = M) → ¬(
    ∀ x y : city_graph,
    connected x y ↔ connected (renumber x) (renumber y)
  )

-- Statement of the problem
theorem city_renumbering_not_possible (city_graph : Type) (connected : city_graph → city_graph → Prop) :
  city_renumbering_invalid city_graph connected :=
sorry

end NUMINAMATH_GPT_city_renumbering_not_possible_l579_57991


namespace NUMINAMATH_GPT_fewerCansCollected_l579_57939

-- Definitions for conditions
def cansCollectedYesterdaySarah := 50
def cansCollectedMoreYesterdayLara := 30
def cansCollectedTodaySarah := 40
def cansCollectedTodayLara := 70

-- Total cans collected yesterday
def totalCansCollectedYesterday := cansCollectedYesterdaySarah + (cansCollectedYesterdaySarah + cansCollectedMoreYesterdayLara)

-- Total cans collected today
def totalCansCollectedToday := cansCollectedTodaySarah + cansCollectedTodayLara

-- Proving the difference
theorem fewerCansCollected :
  totalCansCollectedYesterday - totalCansCollectedToday = 20 := by
  sorry

end NUMINAMATH_GPT_fewerCansCollected_l579_57939


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l579_57904

theorem boat_speed_in_still_water (V_s : ℝ) (D : ℝ) (t_down : ℝ) (t_up : ℝ) (V_b : ℝ) :
  V_s = 3 → t_down = 1 → t_up = 3 / 2 →
  (V_b + V_s) * t_down = D → (V_b - V_s) * t_up = D → V_b = 15 :=
by
  -- sorry is used to skip the actual proof steps
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l579_57904


namespace NUMINAMATH_GPT_range_x_sub_cos_y_l579_57984

theorem range_x_sub_cos_y (x y : ℝ) (h : x^2 + 2 * Real.cos y = 1) : 
  -1 ≤ x - Real.cos y ∧ x - Real.cos y ≤ Real.sqrt 3 + 1 :=
sorry

end NUMINAMATH_GPT_range_x_sub_cos_y_l579_57984


namespace NUMINAMATH_GPT_ScarlettsDishCost_l579_57951

theorem ScarlettsDishCost (L P : ℝ) (tip_rate tip_amount : ℝ) (x : ℝ) 
  (hL : L = 10) (hP : P = 17) (htip_rate : tip_rate = 0.10) (htip_amount : tip_amount = 4) 
  (h : tip_rate * (L + P + x) = tip_amount) : x = 13 :=
by
  sorry

end NUMINAMATH_GPT_ScarlettsDishCost_l579_57951


namespace NUMINAMATH_GPT_tom_catches_48_trout_l579_57918

variable (melanie_tom_catch_ratio : ℕ := 3)
variable (melanie_catch : ℕ := 16)

theorem tom_catches_48_trout (h1 : melanie_catch = 16) (h2 : melanie_tom_catch_ratio = 3) : (melanie_tom_catch_ratio * melanie_catch) = 48 :=
by
  sorry

end NUMINAMATH_GPT_tom_catches_48_trout_l579_57918


namespace NUMINAMATH_GPT_schoolchildren_initial_speed_l579_57901

theorem schoolchildren_initial_speed (v : ℝ) (t t_1 t_2 : ℝ) 
  (h1 : t_1 = (6 * v) / (v + 60) + (400 - 3 * v) / (v + 60)) 
  (h2 : t_2 = (400 - 3 * v) / v) 
  (h3 : t_1 = t_2) : v = 63.24 :=
by sorry

end NUMINAMATH_GPT_schoolchildren_initial_speed_l579_57901


namespace NUMINAMATH_GPT_range_of_a_l579_57996

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x > 3 → x > a)) ↔ (a ≤ 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l579_57996


namespace NUMINAMATH_GPT_fraction_complex_z_l579_57967

theorem fraction_complex_z (z : ℂ) (hz : z = 1 - I) : 2 / z = 1 + I := by
    sorry

end NUMINAMATH_GPT_fraction_complex_z_l579_57967


namespace NUMINAMATH_GPT_polygon_area_is_14_l579_57976

def vertices : List (ℕ × ℕ) :=
  [(1, 2), (2, 2), (3, 3), (3, 4), (4, 5), (5, 5), (6, 5), (6, 4), (5, 3),
   (4, 3), (4, 2), (3, 1), (2, 1), (1, 1)]

noncomputable def area_of_polygon (vs : List (ℕ × ℕ)) : ℝ := sorry

theorem polygon_area_is_14 :
  area_of_polygon vertices = 14 := sorry

end NUMINAMATH_GPT_polygon_area_is_14_l579_57976


namespace NUMINAMATH_GPT_MaryAddedCandy_l579_57981

-- Definitions based on the conditions
def MaryInitialCandyCount (MeganCandyCount : ℕ) : ℕ :=
  3 * MeganCandyCount

-- Given conditions
def MeganCandyCount : ℕ := 5
def MaryTotalCandyCount : ℕ := 25

-- Proof statement
theorem MaryAddedCandy : 
  let MaryInitialCandy := MaryInitialCandyCount MeganCandyCount
  MaryTotalCandyCount - MaryInitialCandy = 10 :=
by 
  sorry

end NUMINAMATH_GPT_MaryAddedCandy_l579_57981


namespace NUMINAMATH_GPT_max_marks_mike_could_have_got_l579_57932

theorem max_marks_mike_could_have_got (p : ℝ) (m_s : ℝ) (d : ℝ) (M : ℝ) :
  p = 0.30 → m_s = 212 → d = 13 → 0.30 * M = (212 + 13) → M = 750 :=
by
  intros hp hms hd heq
  sorry

end NUMINAMATH_GPT_max_marks_mike_could_have_got_l579_57932


namespace NUMINAMATH_GPT_area_of_isosceles_trapezoid_l579_57944

theorem area_of_isosceles_trapezoid (R α : ℝ) (hR : R > 0) (hα1 : 0 < α) (hα2 : α < π) :
  let a := 2 * R
  let b := 2 * R * Real.sin (α / 2)
  let h := R * Real.cos (α / 2)
  (1 / 2) * (a + b) * h = R^2 * (1 + Real.sin (α / 2)) * Real.cos (α / 2) :=
by
  sorry

end NUMINAMATH_GPT_area_of_isosceles_trapezoid_l579_57944


namespace NUMINAMATH_GPT_gcd_yz_min_value_l579_57921

theorem gcd_yz_min_value (x y z : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) 
  (hxy_gcd : Nat.gcd x y = 224) (hxz_gcd : Nat.gcd x z = 546) : 
  Nat.gcd y z = 14 := 
sorry

end NUMINAMATH_GPT_gcd_yz_min_value_l579_57921


namespace NUMINAMATH_GPT_geometric_sequence_eighth_term_is_correct_l579_57913

noncomputable def geometric_sequence_eighth_term : ℚ :=
  let a1 := 2187
  let a5 := 960
  let r := (960 / 2187)^(1/4)
  let a8 := a1 * r^7
  a8

theorem geometric_sequence_eighth_term_is_correct :
  let a1 := 2187
  let a5 := 960
  let r := (960 / 2187)^(1/4)
  let a8 := a1 * r^7
  a8 = 35651584 / 4782969 := by
    sorry

end NUMINAMATH_GPT_geometric_sequence_eighth_term_is_correct_l579_57913


namespace NUMINAMATH_GPT_number_of_stickers_after_losing_page_l579_57940

theorem number_of_stickers_after_losing_page (stickers_per_page : ℕ) (initial_pages : ℕ) (pages_lost : ℕ) :
  stickers_per_page = 20 → initial_pages = 12 → pages_lost = 1 → (initial_pages - pages_lost) * stickers_per_page = 220 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_stickers_after_losing_page_l579_57940


namespace NUMINAMATH_GPT_quad_in_vertex_form_addition_l579_57931

theorem quad_in_vertex_form_addition (a h k : ℝ) (x : ℝ) :
  (∃ a h k, (4 * x^2 - 8 * x + 3) = a * (x - h) ^ 2 + k) →
  a + h + k = 4 :=
by
  sorry

end NUMINAMATH_GPT_quad_in_vertex_form_addition_l579_57931


namespace NUMINAMATH_GPT_find_x_l579_57990

/-- Let x be a real number such that the square roots of a positive number are given by x - 4 and 3. 
    Prove that x equals 1. -/
theorem find_x (x : ℝ) 
  (h₁ : ∃ n : ℝ, n > 0 ∧ n.sqrt = x - 4 ∧ n.sqrt = 3) : 
  x = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l579_57990


namespace NUMINAMATH_GPT_no_nonnegative_integral_solutions_l579_57902

theorem no_nonnegative_integral_solutions :
  ¬ ∃ (x y : ℕ), (x^4 * y^4 - 14 * x^2 * y^2 + 49 = 0) ∧ (x + y = 10) :=
by
  sorry

end NUMINAMATH_GPT_no_nonnegative_integral_solutions_l579_57902


namespace NUMINAMATH_GPT_probability_all_digits_distinct_probability_all_digits_odd_l579_57983

-- Definitions to be used in the proof
def total_possibilities : ℕ := 10^5
def all_distinct_possibilities : ℕ := 10 * 9 * 8 * 7 * 6
def all_odd_possibilities : ℕ := 5^5

-- Probabilities
def prob_all_distinct : ℚ := all_distinct_possibilities / total_possibilities
def prob_all_odd : ℚ := all_odd_possibilities / total_possibilities

-- Lean 4 Statements to Prove
theorem probability_all_digits_distinct :
  prob_all_distinct = 30240 / 100000 := by
  sorry

theorem probability_all_digits_odd :
  prob_all_odd = 3125 / 100000 := by
  sorry

end NUMINAMATH_GPT_probability_all_digits_distinct_probability_all_digits_odd_l579_57983


namespace NUMINAMATH_GPT_polynomial_problem_l579_57926

theorem polynomial_problem 
  (d_1 d_2 d_3 d_4 e_1 e_2 e_3 e_4 : ℝ)
  (h : ∀ (x : ℝ),
    x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 =
    (x^2 + d_1 * x + e_1) * (x^2 + d_2 * x + e_2) * (x^2 + d_3 * x + e_3) * (x^2 + d_4 * x + e_4)) :
  d_1 * e_1 + d_2 * e_2 + d_3 * e_3 + d_4 * e_4 = -1 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_problem_l579_57926


namespace NUMINAMATH_GPT_find_T_l579_57912

theorem find_T (T : ℝ) 
  (h : (1/3) * (1/8) * T = (1/4) * (1/6) * 150) : 
  T = 150 :=
sorry

end NUMINAMATH_GPT_find_T_l579_57912


namespace NUMINAMATH_GPT_paper_folding_possible_layers_l579_57911

theorem paper_folding_possible_layers (n : ℕ) : 16 = 2 ^ n :=
by
  sorry

end NUMINAMATH_GPT_paper_folding_possible_layers_l579_57911


namespace NUMINAMATH_GPT_smallest_Y_74_l579_57992

def isDigitBin (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d = 0 ∨ d = 1

def smallest_Y (Y : ℕ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ isDigitBin T ∧ T % 15 = 0 ∧ Y = T / 15

theorem smallest_Y_74 : smallest_Y 74 := by
  sorry

end NUMINAMATH_GPT_smallest_Y_74_l579_57992


namespace NUMINAMATH_GPT_difference_of_triangular_2010_2009_l579_57948

def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_of_triangular_2010_2009 :
  triangular 2010 - triangular 2009 = 2010 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_triangular_2010_2009_l579_57948


namespace NUMINAMATH_GPT_word_count_proof_l579_57959

def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def consonants : List Char := ['B', 'C', 'D', 'F']
def vowels : List Char := ['A', 'E']

def count_unrestricted_words : ℕ := 6 ^ 5
def count_all_vowel_words : ℕ := 2 ^ 5
def count_one_consonant_words : ℕ := 5 * 4 * (2 ^ 4)
def count_fewer_than_two_consonant_words : ℕ := count_all_vowel_words + count_one_consonant_words

def count_words_with_at_least_two_consonants : ℕ :=
  count_unrestricted_words - count_fewer_than_two_consonant_words

theorem word_count_proof :
  count_words_with_at_least_two_consonants = 7424 := 
by
  -- Proof will be provided here. For now we skip it.
  sorry

end NUMINAMATH_GPT_word_count_proof_l579_57959


namespace NUMINAMATH_GPT_min_n_coloring_property_l579_57936

theorem min_n_coloring_property : ∃ n : ℕ, (∀ (coloring : ℕ → Bool), 
  (∀ (a b c : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n ∧ coloring a = coloring b ∧ coloring b = coloring c → 2 * a + b = c)) ∧ n = 15 := 
sorry

end NUMINAMATH_GPT_min_n_coloring_property_l579_57936


namespace NUMINAMATH_GPT_total_distance_is_correct_l579_57966

noncomputable def magic_ball_total_distance : ℕ := sorry

theorem total_distance_is_correct : magic_ball_total_distance = 80 := sorry

end NUMINAMATH_GPT_total_distance_is_correct_l579_57966


namespace NUMINAMATH_GPT_cos_theta_of_triangle_median_l579_57938

theorem cos_theta_of_triangle_median
  (A : ℝ) (a : ℝ) (m : ℝ) (theta : ℝ)
  (area_eq : A = 24)
  (side_eq : a = 12)
  (median_eq : m = 5)
  (area_formula : A = (1/2) * a * m * Real.sin theta) :
  Real.cos theta = 3 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_cos_theta_of_triangle_median_l579_57938


namespace NUMINAMATH_GPT_arthur_walking_distance_l579_57999

/-- Arthur walks 8 blocks west and 10 blocks south, 
    each block being 1/4 mile -/
theorem arthur_walking_distance 
  (blocks_west : ℕ) (blocks_south : ℕ) (block_distance : ℚ)
  (h1 : blocks_west = 8) (h2 : blocks_south = 10) (h3 : block_distance = 1/4) :
  (blocks_west + blocks_south) * block_distance = 4.5 := 
by
  sorry

end NUMINAMATH_GPT_arthur_walking_distance_l579_57999


namespace NUMINAMATH_GPT_circle_tangent_radii_l579_57910

theorem circle_tangent_radii (a b c : ℝ) (A : ℝ) (p : ℝ)
  (r r_a r_b r_c : ℝ)
  (h1 : p = (a + b + c) / 2)
  (h2 : r = A / p)
  (h3 : r_a = A / (p - a))
  (h4 : r_b = A / (p - b))
  (h5 : r_c = A / (p - c))
  : 1 / r = 1 / r_a + 1 / r_b + 1 / r_c := 
  sorry

end NUMINAMATH_GPT_circle_tangent_radii_l579_57910


namespace NUMINAMATH_GPT_picnic_total_persons_l579_57906

-- Definitions based on given conditions
variables (W M A C : ℕ)
axiom cond1 : M = W + 80
axiom cond2 : A = C + 80
axiom cond3 : M = 120

-- Proof problem: Total persons = 240
theorem picnic_total_persons : W + M + A + C = 240 :=
by
  -- Proof will be filled here
  sorry

end NUMINAMATH_GPT_picnic_total_persons_l579_57906


namespace NUMINAMATH_GPT_fish_original_count_l579_57914

theorem fish_original_count (F : ℕ) (h : F / 2 - F / 6 = 12) : F = 36 := 
by 
  sorry

end NUMINAMATH_GPT_fish_original_count_l579_57914


namespace NUMINAMATH_GPT_jed_correct_speed_l579_57958

def fine_per_mph := 16
def jed_fine := 256
def speed_limit := 50

def jed_speed : Nat := speed_limit + jed_fine / fine_per_mph

theorem jed_correct_speed : jed_speed = 66 := by
  sorry

end NUMINAMATH_GPT_jed_correct_speed_l579_57958


namespace NUMINAMATH_GPT_black_and_white_films_l579_57919

theorem black_and_white_films (y x B : ℕ) 
  (h1 : ∀ B, B = 40 * x)
  (h2 : (4 * y : ℚ) / (((y / x : ℚ) * B / 100) + 4 * y) = 10 / 11) :
  B = 40 * x :=
by sorry

end NUMINAMATH_GPT_black_and_white_films_l579_57919


namespace NUMINAMATH_GPT_projection_of_a_onto_b_l579_57986

namespace VectorProjection

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def scalar_projection (a b : ℝ × ℝ) : ℝ :=
  (dot_product a b) / (magnitude b)

theorem projection_of_a_onto_b :
  scalar_projection (1, -2) (3, 4) = -1 := by
    sorry

end VectorProjection

end NUMINAMATH_GPT_projection_of_a_onto_b_l579_57986


namespace NUMINAMATH_GPT_condition_sufficiency_l579_57953

theorem condition_sufficiency (x₁ x₂ : ℝ) :
  (x₁ > 4 ∧ x₂ > 4) → (x₁ + x₂ > 8 ∧ x₁ * x₂ > 16) ∧ ¬ ((x₁ + x₂ > 8 ∧ x₁ * x₂ > 16) → (x₁ > 4 ∧ x₂ > 4)) :=
by 
  sorry

end NUMINAMATH_GPT_condition_sufficiency_l579_57953


namespace NUMINAMATH_GPT_Caitlin_Sara_weight_l579_57903

variable (A C S : ℕ)

theorem Caitlin_Sara_weight 
  (h1 : A + C = 95) 
  (h2 : A = S + 8) : 
  C + S = 87 := by
  sorry

end NUMINAMATH_GPT_Caitlin_Sara_weight_l579_57903


namespace NUMINAMATH_GPT_angle_F_calculation_l579_57946

theorem angle_F_calculation (D E F : ℝ) :
  D = 80 ∧ E = 2 * F + 30 ∧ D + E + F = 180 → F = 70 / 3 :=
by
  intro h
  cases' h with hD h_remaining
  cases' h_remaining with hE h_sum
  sorry

end NUMINAMATH_GPT_angle_F_calculation_l579_57946


namespace NUMINAMATH_GPT_smallest_n_in_range_l579_57980

theorem smallest_n_in_range : ∃ n : ℕ, n > 1 ∧ (n % 3 = 2) ∧ (n % 7 = 2) ∧ (n % 8 = 2) ∧ 120 ≤ n ∧ n ≤ 149 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_in_range_l579_57980


namespace NUMINAMATH_GPT_find_prime_number_between_50_and_60_l579_57928

theorem find_prime_number_between_50_and_60 (n : ℕ) :
  (50 < n ∧ n < 60) ∧ Prime n ∧ n % 7 = 3 ↔ n = 59 :=
by
  sorry

end NUMINAMATH_GPT_find_prime_number_between_50_and_60_l579_57928


namespace NUMINAMATH_GPT_inequality_proof_equality_case_l579_57979

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 / (b^3 * c) - a / (b^2) ≥ c / b - (c^2) / a) :=
sorry

theorem equality_case (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 / (b^3 * c) - a / (b^2) = c / b - c^2 / a) ↔ (a = b * c) :=
sorry

end NUMINAMATH_GPT_inequality_proof_equality_case_l579_57979


namespace NUMINAMATH_GPT_L_shape_area_correct_l579_57915

noncomputable def large_rectangle_area : ℕ := 12 * 7
noncomputable def small_rectangle_area : ℕ := 4 * 3
noncomputable def L_shape_area := large_rectangle_area - small_rectangle_area

theorem L_shape_area_correct : L_shape_area = 72 := by
  -- here goes your solution
  sorry

end NUMINAMATH_GPT_L_shape_area_correct_l579_57915


namespace NUMINAMATH_GPT_interest_rate_is_five_percent_l579_57961

-- Define the problem parameters
def principal : ℝ := 1200
def amount_after_period : ℝ := 1344
def time_period : ℝ := 2.4

-- Define the simple interest formula
def interest (P R T : ℝ) : ℝ := P * R * T

-- The goal is to prove that the rate of interest is 5% per year
theorem interest_rate_is_five_percent :
  ∃ R, interest principal R time_period = amount_after_period - principal ∧ R = 0.05 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_is_five_percent_l579_57961


namespace NUMINAMATH_GPT_alcohol_percentage_new_mixture_l579_57982

theorem alcohol_percentage_new_mixture (initial_volume new_volume alcohol_initial : ℝ)
  (h1 : initial_volume = 15)
  (h2 : alcohol_initial = 0.20 * initial_volume)
  (h3 : new_volume = initial_volume + 5) :
  (alcohol_initial / new_volume) * 100 = 15 := by
  sorry

end NUMINAMATH_GPT_alcohol_percentage_new_mixture_l579_57982


namespace NUMINAMATH_GPT_find_r_l579_57985

theorem find_r 
  (r RB QC : ℝ)
  (angleA : ℝ)
  (h0 : RB = 6)
  (h1 : QC = 4)
  (h2 : angleA = 90) :
  (r + 6) ^ 2 + (r + 4) ^ 2 = 10 ^ 2 → r = 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_r_l579_57985


namespace NUMINAMATH_GPT_max_median_value_l579_57955

theorem max_median_value (x : ℕ) (h : 198 + x ≤ 392) : x ≤ 194 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_median_value_l579_57955


namespace NUMINAMATH_GPT_big_cows_fewer_than_small_cows_l579_57971

theorem big_cows_fewer_than_small_cows (b s : ℕ) (h1 : b = 6) (h2 : s = 7) : 
  (s - b) / s = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_big_cows_fewer_than_small_cows_l579_57971


namespace NUMINAMATH_GPT_henry_games_l579_57934

theorem henry_games {N H : ℕ} (hN : N = 7) (hH : H = 4 * N) 
    (h_final: H - 6 = 4 * (N + 6)) : H = 58 :=
by
  -- Proof would be inserted here, but skipped using sorry
  sorry

end NUMINAMATH_GPT_henry_games_l579_57934


namespace NUMINAMATH_GPT_sin_2x_from_tan_pi_minus_x_l579_57927

theorem sin_2x_from_tan_pi_minus_x (x : ℝ) (h : Real.tan (Real.pi - x) = 3) : Real.sin (2 * x) = -3 / 5 := by
  sorry

end NUMINAMATH_GPT_sin_2x_from_tan_pi_minus_x_l579_57927


namespace NUMINAMATH_GPT_fraction_to_decimal_l579_57949

theorem fraction_to_decimal : (7 / 50 : ℝ) = 0.14 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l579_57949


namespace NUMINAMATH_GPT_problem_A_problem_B_problem_C_problem_D_l579_57977

theorem problem_A (a b: ℝ) (h : b > 0 ∧ a > b) : ¬(1/a > 1/b) := 
by {
  sorry
}

theorem problem_B (a b: ℝ) (h : a < b ∧ b < 0): (a^2 > a*b) := 
by {
  sorry
}

theorem problem_C (a b: ℝ) (h : a > b): ¬(|a| > |b|) := 
by {
  sorry
}

theorem problem_D (a: ℝ) (h : a > 2): (a + 4/(a-2) ≥ 6) := 
by {
  sorry
}

end NUMINAMATH_GPT_problem_A_problem_B_problem_C_problem_D_l579_57977


namespace NUMINAMATH_GPT_value_of_f_2012_1_l579_57943

noncomputable def f : ℝ → ℝ :=
sorry

-- Condition 1: f is even
axiom even_f : ∀ x : ℝ, f x = f (-x)

-- Condition 2: f(x + 3) = -f(x)
axiom periodicity_f : ∀ x : ℝ, f (x + 3) = -f x

-- Condition 3: f(x) = 2x + 3 for -3 ≤ x ≤ 0
axiom defined_f_on_interval : ∀ x : ℝ, -3 ≤ x ∧ x ≤ 0 → f x = 2 * x + 3

-- Assertion to prove
theorem value_of_f_2012_1 : f 2012.1 = -1.2 :=
by sorry

end NUMINAMATH_GPT_value_of_f_2012_1_l579_57943


namespace NUMINAMATH_GPT_evaluate_expression_l579_57988

theorem evaluate_expression 
    (a b c : ℕ) 
    (ha : a = 7)
    (hb : b = 11)
    (hc : c = 13) :
  let numerator := a^3 * (1 / b - 1 / c) + b^3 * (1 / c - 1 / a) + c^3 * (1 / a - 1 / b)
  let denominator := a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)
  numerator / denominator = 31 := 
by {
  sorry
}

end NUMINAMATH_GPT_evaluate_expression_l579_57988


namespace NUMINAMATH_GPT_keith_apples_correct_l579_57997

def mike_apples : ℕ := 7
def nancy_apples : ℕ := 3
def total_apples : ℕ := 16
def keith_apples : ℕ := total_apples - (mike_apples + nancy_apples)

theorem keith_apples_correct : keith_apples = 6 := by
  -- the actual proof would go here
  sorry

end NUMINAMATH_GPT_keith_apples_correct_l579_57997


namespace NUMINAMATH_GPT_calculate_product_N1_N2_l579_57916

theorem calculate_product_N1_N2 : 
  (∃ (N1 N2 : ℝ), 
    (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → 
      (60 * x - 46) / (x^2 - 5 * x + 6) = N1 / (x - 2) + N2 / (x - 3)) ∧
      N1 * N2 = -1036) :=
  sorry

end NUMINAMATH_GPT_calculate_product_N1_N2_l579_57916


namespace NUMINAMATH_GPT_amount_saved_l579_57962

theorem amount_saved (list_price : ℝ) (tech_deals_discount : ℝ) (electro_bargains_discount : ℝ)
    (tech_deals_price : ℝ) (electro_bargains_price : ℝ) (amount_saved : ℝ) :
  tech_deals_discount = 0.15 →
  list_price = 120 →
  tech_deals_price = list_price * (1 - tech_deals_discount) →
  electro_bargains_discount = 20 →
  electro_bargains_price = list_price - electro_bargains_discount →
  amount_saved = tech_deals_price - electro_bargains_price →
  amount_saved = 2 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_amount_saved_l579_57962


namespace NUMINAMATH_GPT_decrease_in_profit_due_to_looms_breakdown_l579_57973

theorem decrease_in_profit_due_to_looms_breakdown :
  let num_looms := 70
  let month_days := 30
  let total_sales := 1000000
  let total_expenses := 150000
  let daily_sales_per_loom := total_sales / (num_looms * month_days)
  let daily_expenses_per_loom := total_expenses / (num_looms * month_days)
  let loom1_days := 10
  let loom2_days := 5
  let loom3_days := 15
  let loom_repair_cost := 2000
  let loom1_loss := daily_sales_per_loom * loom1_days
  let loom2_loss := daily_sales_per_loom * loom2_days
  let loom3_loss := daily_sales_per_loom * loom3_days
  let total_loss_sales := loom1_loss + loom2_loss + loom3_loss
  let total_repair_cost := loom_repair_cost * 3
  let decrease_in_profit := total_loss_sales + total_repair_cost
  decrease_in_profit = 20285.70 := by
  sorry

end NUMINAMATH_GPT_decrease_in_profit_due_to_looms_breakdown_l579_57973
