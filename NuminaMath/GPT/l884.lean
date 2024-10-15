import Mathlib

namespace NUMINAMATH_GPT_intersection_is_1_l884_88401

def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {y | ∃ x ∈ M, y = x ^ 2}
theorem intersection_is_1 : M ∩ N = {1} := by
  sorry

end NUMINAMATH_GPT_intersection_is_1_l884_88401


namespace NUMINAMATH_GPT_square_area_l884_88499

theorem square_area (x : ℝ) (s1 s2 area : ℝ) 
  (h1 : s1 = 5 * x - 21) 
  (h2 : s2 = 36 - 4 * x) 
  (hs : s1 = s2)
  (ha : area = s1 * s1) : 
  area = 113.4225 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_square_area_l884_88499


namespace NUMINAMATH_GPT_lines_are_parallel_l884_88414

def line1 (x : ℝ) : ℝ := 2 * x + 1
def line2 (x : ℝ) : ℝ := 2 * x + 5

theorem lines_are_parallel : ∀ x y : ℝ, line1 x = y → line2 x = y → false :=
by
  sorry

end NUMINAMATH_GPT_lines_are_parallel_l884_88414


namespace NUMINAMATH_GPT_domain_of_sqrt_tan_minus_one_l884_88449

open Real
open Set

def domain_sqrt_tan_minus_one : Set ℝ := 
  ⋃ k : ℤ, Ico (π/4 + k * π) (π/2 + k * π)

theorem domain_of_sqrt_tan_minus_one :
  {x : ℝ | ∃ y : ℝ, y = sqrt (tan x - 1)} = domain_sqrt_tan_minus_one :=
sorry

end NUMINAMATH_GPT_domain_of_sqrt_tan_minus_one_l884_88449


namespace NUMINAMATH_GPT_initial_percentage_correct_l884_88428

noncomputable def percentInitiallyFull (initialWater: ℕ) (waterAdded: ℕ) (fractionFull: ℚ) (capacity: ℕ) : ℚ :=
  (initialWater : ℚ) / (capacity : ℚ) * 100

theorem initial_percentage_correct (initialWater waterAdded capacity: ℕ) (fractionFull: ℚ) :
  waterAdded = 14 →
  fractionFull = 3/4 →
  capacity = 40 →
  initialWater + waterAdded = fractionFull * capacity →
  percentInitiallyFull initialWater waterAdded fractionFull capacity = 40 :=
by
  intros h1 h2 h3 h4
  unfold percentInitiallyFull
  sorry

end NUMINAMATH_GPT_initial_percentage_correct_l884_88428


namespace NUMINAMATH_GPT_division_quotient_is_correct_l884_88427

noncomputable def polynomial_division_quotient : Polynomial ℚ :=
  Polynomial.div (Polynomial.C 8 * Polynomial.X ^ 3 + 
                  Polynomial.C 16 * Polynomial.X ^ 2 + 
                  Polynomial.C (-7) * Polynomial.X + 
                  Polynomial.C 4) 
                 (Polynomial.C 2 * Polynomial.X + Polynomial.C 5)

theorem division_quotient_is_correct :
  polynomial_division_quotient =
    Polynomial.C 4 * Polynomial.X ^ 2 +
    Polynomial.C (-2) * Polynomial.X +
    Polynomial.C (3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_division_quotient_is_correct_l884_88427


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_abs_values_l884_88443

theorem arithmetic_sequence_sum_abs_values (n : ℕ) (a : ℕ → ℤ)
  (h₁ : a 1 = 13)
  (h₂ : ∀ k, a (k + 1) = a k + (-4)) :
  T_n = if n ≤ 4 then 15 * n - 2 * n^2 else 2 * n^2 - 15 * n + 56 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_abs_values_l884_88443


namespace NUMINAMATH_GPT_total_count_pens_pencils_markers_l884_88406

-- Define the conditions
def ratio_pens_pencils (pens pencils : ℕ) : Prop :=
  6 * pens = 5 * pencils

def nine_more_pencils (pens pencils : ℕ) : Prop :=
  pencils = pens + 9

def ratio_markers_pencils (markers pencils : ℕ) : Prop :=
  3 * markers = 4 * pencils

-- Theorem statement to be proved 
theorem total_count_pens_pencils_markers 
  (pens pencils markers : ℕ) 
  (h1 : ratio_pens_pencils pens pencils)
  (h2 : nine_more_pencils pens pencils)
  (h3 : ratio_markers_pencils markers pencils) : 
  pens + pencils + markers = 171 :=
sorry

end NUMINAMATH_GPT_total_count_pens_pencils_markers_l884_88406


namespace NUMINAMATH_GPT_total_notebooks_distributed_l884_88412

theorem total_notebooks_distributed :
  ∀ (N C : ℕ), 
    (N / C = C / 8) →
    (N = 16 * (C / 2)) →
    N = 512 := 
by
  sorry

end NUMINAMATH_GPT_total_notebooks_distributed_l884_88412


namespace NUMINAMATH_GPT_correct_average_marks_l884_88461

theorem correct_average_marks 
  (n : ℕ) (wrong_avg : ℕ) (wrong_mark : ℕ) (correct_mark : ℕ)
  (h1 : n = 10)
  (h2 : wrong_avg = 100)
  (h3 : wrong_mark = 90)
  (h4 : correct_mark = 10) :
  (n * wrong_avg - wrong_mark + correct_mark) / n = 92 :=
by
  sorry

end NUMINAMATH_GPT_correct_average_marks_l884_88461


namespace NUMINAMATH_GPT_crates_of_mangoes_sold_l884_88455

def total_crates_sold := 50
def crates_grapes_sold := 13
def crates_passion_fruits_sold := 17

theorem crates_of_mangoes_sold : 
  (total_crates_sold - (crates_grapes_sold + crates_passion_fruits_sold) = 20) :=
by 
  sorry

end NUMINAMATH_GPT_crates_of_mangoes_sold_l884_88455


namespace NUMINAMATH_GPT_slope_range_l884_88425

variables (x y k : ℝ)

theorem slope_range :
  (2 ≤ x ∧ x ≤ 3) ∧ (y = -2 * x + 8) ∧ (k = -3 * y / (2 * x)) →
  -3 ≤ k ∧ k ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_slope_range_l884_88425


namespace NUMINAMATH_GPT_problem_statement_l884_88484

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := 2⁻¹
noncomputable def c : ℝ := Real.log 6 / Real.log 5

theorem problem_statement : b < a ∧ a < c := by
  sorry

end NUMINAMATH_GPT_problem_statement_l884_88484


namespace NUMINAMATH_GPT_arrange_singing_begin_end_arrange_singing_not_adjacent_arrange_singing_adjacent_dance_not_adjacent_l884_88474

-- Definitions based on conditions
def performances : Nat := 8
def singing : Nat := 2
def dance : Nat := 3
def variety : Nat := 3

-- Problem 1: Prove arrangement with a singing program at the beginning and end
theorem arrange_singing_begin_end : 1440 = sorry :=
by
  -- proof goes here
  sorry

-- Problem 2: Prove arrangement with singing programs not adjacent
theorem arrange_singing_not_adjacent : 30240 = sorry :=
by
  -- proof goes here
  sorry

-- Problem 3: Prove arrangement with singing programs adjacent and dance not adjacent
theorem arrange_singing_adjacent_dance_not_adjacent : 2880 = sorry :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_arrange_singing_begin_end_arrange_singing_not_adjacent_arrange_singing_adjacent_dance_not_adjacent_l884_88474


namespace NUMINAMATH_GPT_min_2a_plus_3b_l884_88457

theorem min_2a_plus_3b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_parallel : (a * (b - 3) - 2 * b = 0)) :
  (2 * a + 3 * b) = 25 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_min_2a_plus_3b_l884_88457


namespace NUMINAMATH_GPT_solution_set_ineq_l884_88467

theorem solution_set_ineq (x : ℝ) : 
  (x - 1) / (2 * x + 3) > 1 ↔ -4 < x ∧ x < -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_ineq_l884_88467


namespace NUMINAMATH_GPT_union_A_B_intersection_complements_l884_88478
open Set

noncomputable def A : Set ℤ := {x | x^2 - 5 * x + 6 = 0}
noncomputable def B : Set ℤ := {x | x^2 - x - 2 = 0}
def U : Set ℤ := {x | abs x ≤ 3}

theorem union_A_B :
  A ∪ B = { -1, 2, 3 } :=
by sorry

theorem intersection_complements :
  (U \ A) ∩ (U \ B) = { -3, -2, 0, 1 } :=
by sorry

end NUMINAMATH_GPT_union_A_B_intersection_complements_l884_88478


namespace NUMINAMATH_GPT_no_perfect_squares_l884_88431

theorem no_perfect_squares (x y z t : ℕ) (h1 : xy - zt = k) (h2 : x + y = k) (h3 : z + t = k) :
  ¬ (∃ m n : ℕ, x * y = m^2 ∧ z * t = n^2) := by
  sorry

end NUMINAMATH_GPT_no_perfect_squares_l884_88431


namespace NUMINAMATH_GPT_no_20_digit_number_starting_with_11111111111_is_a_perfect_square_l884_88472

theorem no_20_digit_number_starting_with_11111111111_is_a_perfect_square :
  ¬ ∃ (n : ℤ), (10^19 ≤ n ∧ n < 10^20 ∧ (11111111111 * 10^9 ≤ n ∧ n < 11111111112 * 10^9) ∧ (∃ k : ℤ, n = k^2)) :=
by
  sorry

end NUMINAMATH_GPT_no_20_digit_number_starting_with_11111111111_is_a_perfect_square_l884_88472


namespace NUMINAMATH_GPT_rectangle_area_l884_88452

namespace RectangleAreaProof

theorem rectangle_area (SqrArea : ℝ) (SqrSide : ℝ) (RectWidth : ℝ) (RectLength : ℝ) (RectArea : ℝ) :
  SqrArea = 36 →
  SqrSide = Real.sqrt SqrArea →
  RectWidth = SqrSide →
  RectLength = 3 * RectWidth →
  RectArea = RectWidth * RectLength →
  RectArea = 108 := by
  sorry

end RectangleAreaProof

end NUMINAMATH_GPT_rectangle_area_l884_88452


namespace NUMINAMATH_GPT_susans_coins_worth_l884_88480

theorem susans_coins_worth :
  ∃ n d : ℕ, n + d = 40 ∧ (5 * n + 10 * d) = 230 ∧ (10 * n + 5 * d) = 370 :=
sorry

end NUMINAMATH_GPT_susans_coins_worth_l884_88480


namespace NUMINAMATH_GPT_babjis_height_less_by_20_percent_l884_88430

variable (B A : ℝ) (h : A = 1.25 * B)

theorem babjis_height_less_by_20_percent : ((A - B) / A) * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_babjis_height_less_by_20_percent_l884_88430


namespace NUMINAMATH_GPT_solution_is_x_l884_88419

def find_x (x : ℝ) : Prop :=
  64 * (x + 1)^3 - 27 = 0

theorem solution_is_x : ∃ x : ℝ, find_x x ∧ x = -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_solution_is_x_l884_88419


namespace NUMINAMATH_GPT_fenced_area_l884_88465

theorem fenced_area (L W : ℝ) (square_side triangle_leg : ℝ) :
  L = 20 ∧ W = 18 ∧ square_side = 4 ∧ triangle_leg = 3 →
  (L * W - square_side^2 - (1 / 2) * triangle_leg^2 = 339.5) := by
  intros h
  rcases h with ⟨hL, hW, hs, ht⟩
  rw [hL, hW, hs, ht]
  simp
  sorry

end NUMINAMATH_GPT_fenced_area_l884_88465


namespace NUMINAMATH_GPT_jim_age_in_2_years_l884_88492

theorem jim_age_in_2_years (c1 : ∀ t : ℕ, t = 37) (c2 : ∀ j : ℕ, j = 27) : ∀ j2 : ℕ, j2 = 29 :=
by
  sorry

end NUMINAMATH_GPT_jim_age_in_2_years_l884_88492


namespace NUMINAMATH_GPT_other_person_age_l884_88490

variable {x : ℕ} -- age of the other person
variable {y : ℕ} -- Marco's age

-- Conditions given in the problem.
axiom marco_age : y = 2 * x + 1
axiom sum_ages : x + y = 37

-- Goal: Prove that the age of the other person is 12.
theorem other_person_age : x = 12 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_other_person_age_l884_88490


namespace NUMINAMATH_GPT_total_books_gwen_has_l884_88436

-- Definitions based on conditions in part a
def mystery_shelves : ℕ := 5
def picture_shelves : ℕ := 3
def books_per_shelf : ℕ := 4

-- Problem statement in Lean 4
theorem total_books_gwen_has : 
  mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = 32 := by
  -- This is where the proof would go, but we include sorry to skip for now
  sorry

end NUMINAMATH_GPT_total_books_gwen_has_l884_88436


namespace NUMINAMATH_GPT_rattlesnake_tail_percentage_difference_l884_88407

-- Definitions for the problem
def eastern_segments : Nat := 6
def western_segments : Nat := 8

-- The statement to prove
theorem rattlesnake_tail_percentage_difference :
  100 * (western_segments - eastern_segments) / western_segments = 25 := by
  sorry

end NUMINAMATH_GPT_rattlesnake_tail_percentage_difference_l884_88407


namespace NUMINAMATH_GPT_common_ratio_geometric_series_l884_88447

theorem common_ratio_geometric_series (a r S : ℝ) (h₁ : S = a / (1 - r))
  (h₂ : r ≠ 1)
  (h₃ : r^4 * S = S / 81) :
  r = 1/3 :=
by 
  sorry

end NUMINAMATH_GPT_common_ratio_geometric_series_l884_88447


namespace NUMINAMATH_GPT_find_x_approx_l884_88475

theorem find_x_approx :
  ∀ (x : ℝ), 3639 + 11.95 - x^2 = 3054 → abs (x - 24.43) < 0.01 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_find_x_approx_l884_88475


namespace NUMINAMATH_GPT_speed_last_segment_l884_88463

-- Definitions corresponding to conditions
def drove_total_distance : ℝ := 150
def total_time_minutes : ℝ := 120
def time_first_segment_minutes : ℝ := 40
def speed_first_segment_mph : ℝ := 70
def speed_second_segment_mph : ℝ := 75

-- The statement of the problem
theorem speed_last_segment :
  let total_distance : ℝ := drove_total_distance
  let total_time : ℝ := total_time_minutes / 60
  let time_first_segment : ℝ := time_first_segment_minutes / 60
  let time_second_segment : ℝ := time_first_segment
  let time_last_segment : ℝ := time_first_segment
  let distance_first_segment : ℝ := speed_first_segment_mph * time_first_segment
  let distance_second_segment : ℝ := speed_second_segment_mph * time_second_segment
  let distance_two_segments : ℝ := distance_first_segment + distance_second_segment
  let distance_last_segment : ℝ := total_distance - distance_two_segments
  let speed_last_segment := distance_last_segment / time_last_segment
  speed_last_segment = 80 := 
  sorry

end NUMINAMATH_GPT_speed_last_segment_l884_88463


namespace NUMINAMATH_GPT_problem_solution_l884_88420

theorem problem_solution (a d e : ℕ) (ha : 0 < a ∧ a < 10) (hd : 0 < d ∧ d < 10) (he : 0 < e ∧ e < 10) :
  ((10 * a + d) * (10 * a + e) = 100 * a ^ 2 + 110 * a + d * e) ↔ (d + e = 11) := by
  sorry

end NUMINAMATH_GPT_problem_solution_l884_88420


namespace NUMINAMATH_GPT_platform_length_l884_88408

theorem platform_length (length_train : ℝ) (speed_train_kmph : ℝ) (time_sec : ℝ) (length_platform : ℝ) :
  length_train = 1020 → speed_train_kmph = 102 → time_sec = 50 →
  length_platform = (speed_train_kmph * 1000 / 3600) * time_sec - length_train :=
by
  intros
  sorry

end NUMINAMATH_GPT_platform_length_l884_88408


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l884_88426

open Set

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_of_A_and_B :
  A ∩ B = {3, 9} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l884_88426


namespace NUMINAMATH_GPT_equal_angles_proof_l884_88476

/-- Proof Problem: After how many minutes will the hour and minute hands form equal angles with their positions at 12 o'clock? -/
noncomputable def equal_angle_time (x : ℝ) : Prop :=
  -- Defining the conditions for the problem
  let minute_hand_speed := 6 -- degrees per minute
  let hour_hand_speed := 0.5 -- degrees per minute
  let total_degrees := 360 * x -- total degrees of minute hand till time x
  let hour_hand_degrees := 30 * (x / 60) -- total degrees of hour hand till time x

  -- Equation for equal angles formed with respect to 12 o'clock
  30 * (x / 60) = 360 - 360 * (x / 60)

theorem equal_angles_proof :
  ∃ (x : ℝ), equal_angle_time x ∧ x = 55 + 5/13 :=
sorry

end NUMINAMATH_GPT_equal_angles_proof_l884_88476


namespace NUMINAMATH_GPT_marble_weights_total_l884_88418

theorem marble_weights_total:
  0.33 + 0.33 + 0.08 + 0.25 + 0.02 + 0.12 + 0.15 = 1.28 :=
by {
  sorry
}

end NUMINAMATH_GPT_marble_weights_total_l884_88418


namespace NUMINAMATH_GPT_simplify_expr_l884_88460

variable (a b : ℝ)

theorem simplify_expr (h : a + b ≠ 0) : 
  a - b + 2 * b^2 / (a + b) = (a^2 + b^2) / (a + b) :=
sorry

end NUMINAMATH_GPT_simplify_expr_l884_88460


namespace NUMINAMATH_GPT_problem_statement_l884_88456

variable {x : ℝ}
noncomputable def A : ℝ := 39
noncomputable def B : ℝ := -5

theorem problem_statement (h : ∀ x ≠ 3, (A / (x - 3) + B * (x + 2)) = (-5 * x ^ 2 + 18 * x + 30) / (x - 3)) : A + B = 34 := 
sorry

end NUMINAMATH_GPT_problem_statement_l884_88456


namespace NUMINAMATH_GPT_find_a_l884_88470

theorem find_a (a : ℤ) (h : |a + 1| = 3) : a = 2 ∨ a = -4 :=
sorry

end NUMINAMATH_GPT_find_a_l884_88470


namespace NUMINAMATH_GPT_num_integers_between_700_and_900_with_sum_of_digits_18_l884_88450

def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem num_integers_between_700_and_900_with_sum_of_digits_18 : 
  ∃ k, k = 17 ∧ ∀ n, 700 ≤ n ∧ n ≤ 900 ∧ sum_of_digits n = 18 ↔ (1 ≤ k) := 
sorry

end NUMINAMATH_GPT_num_integers_between_700_and_900_with_sum_of_digits_18_l884_88450


namespace NUMINAMATH_GPT_set_intersection_l884_88448

-- Define set A
def A := {x : ℝ | x^2 - 4 * x < 0}

-- Define set B
def B := {x : ℤ | -2 < x ∧ x ≤ 2}

-- Define the intersection of A and B in ℝ
def A_inter_B := {x : ℝ | (x ∈ A) ∧ (∃ (z : ℤ), (x = z) ∧ (z ∈ B))}

-- Proof statement
theorem set_intersection : A_inter_B = {1, 2} :=
by sorry

end NUMINAMATH_GPT_set_intersection_l884_88448


namespace NUMINAMATH_GPT_journey_total_time_l884_88402

theorem journey_total_time (speed1 time1 speed2 total_distance : ℕ) 
  (h1 : speed1 = 40) 
  (h2 : time1 = 3) 
  (h3 : speed2 = 60) 
  (h4 : total_distance = 240) : 
  time1 + (total_distance - speed1 * time1) / speed2 = 5 := 
by 
  sorry

end NUMINAMATH_GPT_journey_total_time_l884_88402


namespace NUMINAMATH_GPT_roots_of_equations_l884_88413

theorem roots_of_equations (a : ℝ) :
  (∃ x : ℝ, x^2 + 4 * a * x - 4 * a + 3 = 0) ∨
  (∃ x : ℝ, x^2 + (a - 1) * x + a^2 = 0) ∨
  (∃ x : ℝ, x^2 + 2 * a * x - 2 * a = 0) ↔ 
  a ≤ -3 / 2 ∨ a ≥ -1 :=
sorry

end NUMINAMATH_GPT_roots_of_equations_l884_88413


namespace NUMINAMATH_GPT_fractions_equal_l884_88497

theorem fractions_equal (a b c d : ℚ) (h1 : a = 2/7) (h2 : b = 3) (h3 : c = 3/7) (h4 : d = 2) :
  a * b = c * d := 
sorry

end NUMINAMATH_GPT_fractions_equal_l884_88497


namespace NUMINAMATH_GPT_students_growth_rate_l884_88438

theorem students_growth_rate (x : ℝ) 
  (h_total : 728 = 200 + 200 * (1+x) + 200 * (1+x)^2) : 
  200 + 200 * (1+x) + 200*(1+x)^2 = 728 := 
  by
  sorry

end NUMINAMATH_GPT_students_growth_rate_l884_88438


namespace NUMINAMATH_GPT_profit_per_piece_correct_sales_volume_correct_maximum_monthly_profit_optimum_selling_price_is_130_l884_88437

-- Define the selling price and cost price
def cost_price : ℝ := 60
def sales_price (x : ℝ) := x

-- 1. Prove the profit per piece
def profit_per_piece (x : ℝ) : ℝ := sales_price x - cost_price

theorem profit_per_piece_correct (x : ℝ) : profit_per_piece x = x - 60 :=
by 
  -- it follows directly from the definition of profit_per_piece
  sorry

-- 2. Define the linear function relationship between monthly sales volume and selling price
def sales_volume (x : ℝ) : ℝ := -2 * x + 400

theorem sales_volume_correct (x : ℝ) : sales_volume x = -2 * x + 400 :=
by 
  -- it follows directly from the definition of sales_volume
  sorry

-- 3. Define the monthly profit and prove the maximized profit
def monthly_profit (x : ℝ) : ℝ := profit_per_piece x * sales_volume x

theorem maximum_monthly_profit (x : ℝ) : 
  monthly_profit x = -2 * x^2 + 520 * x - 24000 :=
by 
  -- it follows directly from the definition of monthly_profit
  sorry

theorem optimum_selling_price_is_130 : ∃ (x : ℝ), (monthly_profit x = 9800) ∧ (x = 130) :=
by
  -- solve this using the properties of quadratic functions
  sorry

end NUMINAMATH_GPT_profit_per_piece_correct_sales_volume_correct_maximum_monthly_profit_optimum_selling_price_is_130_l884_88437


namespace NUMINAMATH_GPT_minimum_value_of_f_div_f_l884_88491

noncomputable def quadratic_function_min_value (a b c : ℝ) (h : 0 < b) (h₀ : 0 < a) (h₁ : 0 < c) (h₂ : b^2 ≤ 4*a*c) : ℝ :=
  (a + b + c) / b

theorem minimum_value_of_f_div_f' (a b c : ℝ) (h : 0 < b)
  (h₀ : 0 < a) (h₁ : 0 < c) (h₂ : b^2 ≤ 4*a*c) :
  quadratic_function_min_value a b c h h₀ h₁ h₂ = 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_f_div_f_l884_88491


namespace NUMINAMATH_GPT_prove_bounds_l884_88464

variable (a b : ℝ)

-- Conditions
def condition1 : Prop := 6 * a - b = 45
def condition2 : Prop := 4 * a + b > 60

-- Proof problem statement
theorem prove_bounds (h1 : condition1 a b) (h2 : condition2 a b) : a > 10.5 ∧ b > 18 :=
sorry

end NUMINAMATH_GPT_prove_bounds_l884_88464


namespace NUMINAMATH_GPT_magnet_cost_is_three_l884_88469

noncomputable def stuffed_animal_cost : ℕ := 6
noncomputable def combined_stuffed_animals_cost : ℕ := 2 * stuffed_animal_cost
noncomputable def magnet_cost : ℕ := combined_stuffed_animals_cost / 4

theorem magnet_cost_is_three : magnet_cost = 3 :=
by
  sorry

end NUMINAMATH_GPT_magnet_cost_is_three_l884_88469


namespace NUMINAMATH_GPT_sum_of_sides_eq_13_or_15_l884_88432

noncomputable def squares_side_lengths (b d : ℕ) : Prop :=
  15^2 = b^2 + 10^2 + d^2

theorem sum_of_sides_eq_13_or_15 :
  ∃ b d : ℕ, squares_side_lengths b d ∧ (b + d = 13 ∨ b + d = 15) :=
sorry

end NUMINAMATH_GPT_sum_of_sides_eq_13_or_15_l884_88432


namespace NUMINAMATH_GPT_question1_question2_case1_question2_case2_question2_case3_l884_88494

def f (x a : ℝ) : ℝ := x^2 + (1 - a) * x - a

theorem question1 (x : ℝ) (h : (-1 < x) ∧ (x < 3)) : f x 3 < 0 := sorry

theorem question2_case1 (x : ℝ) : f x (-1) > 0 ↔ x ≠ -1 := sorry

theorem question2_case2 (x a : ℝ) (h : a > -1) : f x a > 0 ↔ (x < -1 ∨ x > a) := sorry

theorem question2_case3 (x a : ℝ) (h : a < -1) : f x a > 0 ↔ (x < a ∨ x > -1) := sorry

end NUMINAMATH_GPT_question1_question2_case1_question2_case2_question2_case3_l884_88494


namespace NUMINAMATH_GPT_triangle_side_lengths_inequality_l884_88473

theorem triangle_side_lengths_inequality
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := 
sorry

end NUMINAMATH_GPT_triangle_side_lengths_inequality_l884_88473


namespace NUMINAMATH_GPT_value_of_a_minus_b_l884_88433

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 6) (h2 : |b| = 2) (h3 : a + b > 0) :
  a - b = 4 ∨ a - b = 8 :=
  sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l884_88433


namespace NUMINAMATH_GPT_find_angle_complement_supplement_l884_88440

theorem find_angle_complement_supplement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_complement_supplement_l884_88440


namespace NUMINAMATH_GPT_total_earmuffs_l884_88403

theorem total_earmuffs {a b c : ℕ} (h1 : a = 1346) (h2 : b = 6444) (h3 : c = a + b) : c = 7790 := by
  sorry

end NUMINAMATH_GPT_total_earmuffs_l884_88403


namespace NUMINAMATH_GPT_find_n_l884_88482

-- Defining the parameters and conditions
def large_block_positions (n : ℕ) : ℕ := 199 * n + 110 * (n - 1)

-- Theorem statement
theorem find_n (h : large_block_positions n = 2362) : n = 8 :=
sorry

end NUMINAMATH_GPT_find_n_l884_88482


namespace NUMINAMATH_GPT_probability_of_successful_meeting_l884_88493

noncomputable def meeting_probability : ℚ := 7 / 64

theorem probability_of_successful_meeting :
  (∃ x y z : ℝ,
     0 ≤ x ∧ x ≤ 2 ∧
     0 ≤ y ∧ y ≤ 2 ∧
     0 ≤ z ∧ z ≤ 2 ∧
     abs (x - z) ≤ 0.75 ∧
     abs (y - z) ≤ 1.5 ∧
     z ≥ x ∧
     z ≥ y) →
  meeting_probability = 7 / 64 := by
  sorry

end NUMINAMATH_GPT_probability_of_successful_meeting_l884_88493


namespace NUMINAMATH_GPT_find_m_range_l884_88417

theorem find_m_range
  (m y1 y2 y0 x0 : ℝ)
  (a c : ℝ) (h1 : a ≠ 0)
  (h2 : x0 = -2)
  (h3 : ∀ x, (x, ax^2 + 4*a*x + c) = (m, y1) ∨ (x, ax^2 + 4*a*x + c) = (m + 2, y2) ∨ (x, ax^2 + 4*a*x + c) = (x0, y0))
  (h4 : y0 ≥ y2) (h5 : y2 > y1) :
  m < -3 :=
sorry

end NUMINAMATH_GPT_find_m_range_l884_88417


namespace NUMINAMATH_GPT_max_quadratic_in_interval_l884_88444

-- Define the quadratic function
noncomputable def quadratic_fun (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the closed interval
def interval (a b : ℝ) (x : ℝ) : Prop := a ≤ x ∧ x ≤ b

-- Define the maximum value property
def is_max_value (f : ℝ → ℝ) (a b max_val : ℝ) : Prop :=
  ∀ x, interval a b x → f x ≤ max_val

-- State the problem in Lean 4
theorem max_quadratic_in_interval : 
  is_max_value quadratic_fun (-5) 3 36 := 
sorry

end NUMINAMATH_GPT_max_quadratic_in_interval_l884_88444


namespace NUMINAMATH_GPT_sum_local_values_of_digits_l884_88423

theorem sum_local_values_of_digits :
  let d2 := 2000
  let d3 := 300
  let d4 := 40
  let d5 := 5
  d2 + d3 + d4 + d5 = 2345 :=
by
  sorry

end NUMINAMATH_GPT_sum_local_values_of_digits_l884_88423


namespace NUMINAMATH_GPT_solve_for_x_l884_88468

theorem solve_for_x (x : ℚ) (h₁ : (7 * x + 2) / (x - 4) = -6 / (x - 4)) (h₂ : x ≠ 4) :
  x = -8 / 7 := 
  sorry

end NUMINAMATH_GPT_solve_for_x_l884_88468


namespace NUMINAMATH_GPT_sum_reciprocal_l884_88405

-- Definition of the problem
theorem sum_reciprocal (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + y^2 = 4 * x * y) : 
  (1 / x) + (1 / y) = 1 :=
sorry

end NUMINAMATH_GPT_sum_reciprocal_l884_88405


namespace NUMINAMATH_GPT_numbers_of_form_xy9z_div_by_132_l884_88445

theorem numbers_of_form_xy9z_div_by_132 (x y z : ℕ) :
  let N := 1000 * x + 100 * y + 90 + z
  (N % 4 = 0) ∧ ((x + y + 9 + z) % 3 = 0) ∧ ((x + 9 - y - z) % 11 = 0) ↔ 
  (N = 3696) ∨ (N = 4092) ∨ (N = 6996) ∨ (N = 7392) :=
by
  intros
  let N := 1000 * x + 100 * y + 90 + z
  sorry

end NUMINAMATH_GPT_numbers_of_form_xy9z_div_by_132_l884_88445


namespace NUMINAMATH_GPT_circle_center_sum_l884_88422

theorem circle_center_sum (x y : ℝ) (h : (x - 5)^2 + (y - 2)^2 = 38) : x + y = 7 := 
  sorry

end NUMINAMATH_GPT_circle_center_sum_l884_88422


namespace NUMINAMATH_GPT_solve_for_x_l884_88435

theorem solve_for_x (x : ℚ) (h : 1/4 + 7/x = 13/x + 1/9) : x = 216/5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l884_88435


namespace NUMINAMATH_GPT_silvia_savings_l884_88400

def retail_price : ℝ := 1000
def guitar_center_discount_rate : ℝ := 0.15
def sweetwater_discount_rate : ℝ := 0.10
def guitar_center_shipping_fee : ℝ := 100
def sweetwater_shipping_fee : ℝ := 0

def guitar_center_cost : ℝ := retail_price * (1 - guitar_center_discount_rate) + guitar_center_shipping_fee
def sweetwater_cost : ℝ := retail_price * (1 - sweetwater_discount_rate) + sweetwater_shipping_fee

theorem silvia_savings : guitar_center_cost - sweetwater_cost = 50 := by
  sorry

end NUMINAMATH_GPT_silvia_savings_l884_88400


namespace NUMINAMATH_GPT_percent_of_a_is_4b_l884_88458

variable (a b : ℝ)

theorem percent_of_a_is_4b (hab : a = 1.8 * b) :
  (4 * b / a) * 100 = 222.22 := by
  sorry

end NUMINAMATH_GPT_percent_of_a_is_4b_l884_88458


namespace NUMINAMATH_GPT_angle_A_area_of_triangle_l884_88421

open Real

theorem angle_A (a : ℝ) (A B C : ℝ) 
  (h_a : a = 2 * sqrt 3)
  (h_condition1 : 4 * cos A ^ 2 + 4 * cos B * cos C + 1 = 4 * sin B * sin C) :
  A = π / 3 := 
sorry

theorem area_of_triangle (a b c A : ℝ) 
  (h_A : A = π / 3)
  (h_a : a = 2 * sqrt 3)
  (h_b : b = 3 * c) :
  (1 / 2) * b * c * sin A = 9 * sqrt 3 / 7 := 
sorry

end NUMINAMATH_GPT_angle_A_area_of_triangle_l884_88421


namespace NUMINAMATH_GPT_tank_depth_l884_88471

open Real

theorem tank_depth :
  ∃ d : ℝ, (0.75 * (2 * 25 * d + 2 * 12 * d + 25 * 12) = 558) ∧ d = 6 :=
sorry

end NUMINAMATH_GPT_tank_depth_l884_88471


namespace NUMINAMATH_GPT_aaron_age_l884_88481

variable (A : ℕ)
variable (henry_sister_age : ℕ)
variable (henry_age : ℕ)
variable (combined_age : ℕ)

theorem aaron_age (h1 : henry_sister_age = 3 * A)
                 (h2 : henry_age = 4 * henry_sister_age)
                 (h3 : combined_age = henry_sister_age + henry_age)
                 (h4 : combined_age = 240) : A = 16 := by
  sorry

end NUMINAMATH_GPT_aaron_age_l884_88481


namespace NUMINAMATH_GPT_first_car_made_earlier_l884_88486

def year_first_car : ℕ := 1970
def year_third_car : ℕ := 2000
def diff_third_second : ℕ := 20

theorem first_car_made_earlier : (year_third_car - diff_third_second) - year_first_car = 10 := by
  sorry

end NUMINAMATH_GPT_first_car_made_earlier_l884_88486


namespace NUMINAMATH_GPT_mirror_tweet_rate_is_45_l884_88415

-- Defining the conditions given in the problem
def happy_tweet_rate : ℕ := 18
def hungry_tweet_rate : ℕ := 4
def mirror_tweet_rate (x : ℕ) : ℕ := x
def happy_minutes : ℕ := 20
def hungry_minutes : ℕ := 20
def mirror_minutes : ℕ := 20
def total_tweets : ℕ := 1340

-- Proving the rate of tweets when Polly watches herself in the mirror
theorem mirror_tweet_rate_is_45 : mirror_tweet_rate 45 * mirror_minutes = total_tweets - (happy_tweet_rate * happy_minutes + hungry_tweet_rate * hungry_minutes) :=
by 
  sorry

end NUMINAMATH_GPT_mirror_tweet_rate_is_45_l884_88415


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l884_88498

variable {a : ℝ}

theorem sufficient_but_not_necessary_condition :
  (∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ↔ (a ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l884_88498


namespace NUMINAMATH_GPT_solution_of_system_l884_88429

theorem solution_of_system (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : x^3 - 6 * x * y - 8 * y^3 = 1) :
  y = (x - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_of_system_l884_88429


namespace NUMINAMATH_GPT_cellini_inscription_l884_88409

noncomputable def famous_master_engravings (x: Type) : String :=
  "Эту шкатулку изготовил сын Челлини"

theorem cellini_inscription (x: Type) (created_by_cellini : x) :
  famous_master_engravings x = "Эту шкатулку изготовил сын Челлини" :=
by
  sorry

end NUMINAMATH_GPT_cellini_inscription_l884_88409


namespace NUMINAMATH_GPT_Maddie_bought_palettes_l884_88496

-- Defining constants and conditions as per the problem statement.
def cost_per_palette : ℝ := 15
def number_of_lipsticks : ℝ := 4
def cost_per_lipstick : ℝ := 2.50
def number_of_hair_boxes : ℝ := 3
def cost_per_hair_box : ℝ := 4
def total_paid : ℝ := 67

-- Defining the condition which we need to prove for number of makeup palettes bought.
theorem Maddie_bought_palettes (P : ℝ) :
  (number_of_lipsticks * cost_per_lipstick) +
  (number_of_hair_boxes * cost_per_hair_box) +
  (cost_per_palette * P) = total_paid →
  P = 3 :=
sorry

end NUMINAMATH_GPT_Maddie_bought_palettes_l884_88496


namespace NUMINAMATH_GPT_total_area_pool_and_deck_l884_88479

theorem total_area_pool_and_deck (pool_length pool_width deck_width : ℕ) 
  (h1 : pool_length = 12) 
  (h2 : pool_width = 10) 
  (h3 : deck_width = 4) : 
  (pool_length + 2 * deck_width) * (pool_width + 2 * deck_width) = 360 := 
by sorry

end NUMINAMATH_GPT_total_area_pool_and_deck_l884_88479


namespace NUMINAMATH_GPT_same_number_of_friends_l884_88477

theorem same_number_of_friends (n : ℕ) (friends : Fin n → Fin n) :
  (∃ i j : Fin n, i ≠ j ∧ friends i = friends j) :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_same_number_of_friends_l884_88477


namespace NUMINAMATH_GPT_problem_statement_l884_88462

variable (x1 x2 x3 x4 x5 x6 x7 : ℝ)

theorem problem_statement
  (h1 : x1 + 4*x2 + 9*x3 + 16*x4 + 25*x5 + 36*x6 + 49*x7 = 5)
  (h2 : 4*x1 + 9*x2 + 16*x3 + 25*x4 + 36*x5 + 49*x6 + 64*x7 = 20)
  (h3 : 9*x1 + 16*x2 + 25*x3 + 36*x4 + 49*x5 + 64*x6 + 81*x7 = 145) :
  16*x1 + 25*x2 + 36*x3 + 49*x4 + 64*x5 + 81*x6 + 100*x7 = 380 :=
sorry

end NUMINAMATH_GPT_problem_statement_l884_88462


namespace NUMINAMATH_GPT_math_problem_l884_88487

theorem math_problem : 33333 * 33334 = 1111122222 := 
by sorry

end NUMINAMATH_GPT_math_problem_l884_88487


namespace NUMINAMATH_GPT_mean_equality_l884_88454

theorem mean_equality (z : ℚ) :
  (8 + 12 + 24) / 3 = (16 + z) / 2 ↔ z = 40 / 3 :=
by
  sorry

end NUMINAMATH_GPT_mean_equality_l884_88454


namespace NUMINAMATH_GPT_garden_ratio_l884_88495

-- Define the given conditions
def garden_length : ℕ := 100
def garden_perimeter : ℕ := 300

-- Problem statement: Prove the ratio of the length to the width is 2:1
theorem garden_ratio : 
  ∃ (W L : ℕ), 
    L = garden_length ∧ 
    2 * L + 2 * W = garden_perimeter ∧ 
    L / W = 2 :=
by 
  sorry

end NUMINAMATH_GPT_garden_ratio_l884_88495


namespace NUMINAMATH_GPT_increase_productivity_RnD_l884_88410

theorem increase_productivity_RnD :
  let RnD_t := 2640.92
  let ΔAPL_t2 := 0.81
  RnD_t / ΔAPL_t2 = 3260 :=
by
  let RnD_t := 2640.92
  let ΔAPL_t2 := 0.81
  have h : RnD_t / ΔAPL_t2 = 3260 := sorry
  exact h

end NUMINAMATH_GPT_increase_productivity_RnD_l884_88410


namespace NUMINAMATH_GPT_count_n_satisfies_conditions_l884_88404

theorem count_n_satisfies_conditions :
  ∃ (count : ℕ), count = 36 ∧ ∀ (n : ℕ), 
    0 < n ∧ n < 150 →
    ∃ (k : ℕ), 
    (n = 2*k + 2) ∧ 
    (k*(k + 2) % 4 = 0) :=
by
  sorry

end NUMINAMATH_GPT_count_n_satisfies_conditions_l884_88404


namespace NUMINAMATH_GPT_ratio_of_volumes_l884_88453

-- Definitions based on given conditions
def V1 : ℝ := sorry -- Volume of the first vessel
def V2 : ℝ := sorry -- Volume of the second vessel

-- Given condition
def condition : Prop := (3 / 4) * V1 = (5 / 8) * V2

-- The theorem to prove the ratio V1 / V2 is 5 / 6
theorem ratio_of_volumes (h : condition) : V1 / V2 = 5 / 6 :=
sorry

end NUMINAMATH_GPT_ratio_of_volumes_l884_88453


namespace NUMINAMATH_GPT_complementary_event_equivalence_l884_88451

-- Define the event E: hitting the target at least once in two shots.
-- Event E complementary: missing the target both times.

def eventE := "hitting the target at least once"
def complementaryEvent := "missing the target both times"

theorem complementary_event_equivalence :
  (complementaryEvent = "missing the target both times") ↔ (eventE = "hitting the target at least once") :=
by
  sorry

end NUMINAMATH_GPT_complementary_event_equivalence_l884_88451


namespace NUMINAMATH_GPT_range_of_x_l884_88411

theorem range_of_x (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : x + y + z = 1) (h4 : x^2 + y^2 + z^2 = 3) : 1 ≤ x ∧ x ≤ 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l884_88411


namespace NUMINAMATH_GPT_second_term_of_series_l884_88489

noncomputable def geometric_series_second_term (a r S : ℝ) := r * a

theorem second_term_of_series (a r : ℝ) (S : ℝ) (hr : r = 1/4) (hs : S = 16) 
  (hS_formula : S = a / (1 - r)) : geometric_series_second_term a r S = 3 :=
by
  -- Definitions are in place, applying algebraic manipulation steps here would follow
  sorry

end NUMINAMATH_GPT_second_term_of_series_l884_88489


namespace NUMINAMATH_GPT_yogurt_combinations_l884_88466

theorem yogurt_combinations (flavors toppings : ℕ) (h_flavors : flavors = 5) (h_toppings : toppings = 7) :
  (flavors * Nat.choose toppings 3) = 175 := by
  sorry

end NUMINAMATH_GPT_yogurt_combinations_l884_88466


namespace NUMINAMATH_GPT_problem_l884_88434

namespace MathProof

variable {p a b : ℕ}

theorem problem (h1 : Nat.Prime p) (h2 : p % 2 = 1) (h3 : a > 0) (h4 : b > 0) (h5 : (p + 1)^a - p^b = 1) : a = 1 ∧ b = 1 := 
sorry

end MathProof

end NUMINAMATH_GPT_problem_l884_88434


namespace NUMINAMATH_GPT_olly_needs_24_shoes_l884_88416

-- Define the number of paws for different types of pets
def dogs : ℕ := 3
def cats : ℕ := 2
def ferret : ℕ := 1

def paws_per_dog : ℕ := 4
def paws_per_cat : ℕ := 4
def paws_per_ferret : ℕ := 4

-- The theorem we want to prove
theorem olly_needs_24_shoes : 
  dogs * paws_per_dog + cats * paws_per_cat + ferret * paws_per_ferret = 24 :=
by
  sorry

end NUMINAMATH_GPT_olly_needs_24_shoes_l884_88416


namespace NUMINAMATH_GPT_sample_group_b_correct_l884_88442

noncomputable def stratified_sample_group_b (total_cities: ℕ) (group_b_cities: ℕ) (sample_size: ℕ) : ℕ :=
  (sample_size * group_b_cities) / total_cities

theorem sample_group_b_correct : stratified_sample_group_b 36 12 12 = 4 := by
  sorry

end NUMINAMATH_GPT_sample_group_b_correct_l884_88442


namespace NUMINAMATH_GPT_triangle_inequality_iff_inequality_l884_88441

theorem triangle_inequality_iff_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 * (a^4 + b^4 + c^4) < (a^2 + b^2 + c^2)^2) ↔ (a + b > c ∧ b + c > a ∧ c + a > b) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_iff_inequality_l884_88441


namespace NUMINAMATH_GPT_principal_amount_l884_88483

variable (P : ℝ)

/-- Prove the principal amount P given that the simple interest at 4% for 5 years is Rs. 2400 less than the principal --/
theorem principal_amount : 
  (4/100) * P * 5 = P - 2400 → 
  P = 3000 := 
by 
  sorry

end NUMINAMATH_GPT_principal_amount_l884_88483


namespace NUMINAMATH_GPT_no_four_primes_exist_l884_88459

theorem no_four_primes_exist (a b c d : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b)
  (hc : Nat.Prime c) (hd : Nat.Prime d) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : (1 / a : ℚ) + (1 / d) = (1 / b) + (1 / c)) : False := sorry

end NUMINAMATH_GPT_no_four_primes_exist_l884_88459


namespace NUMINAMATH_GPT_find_b_l884_88488

theorem find_b (b n : ℝ) (h_neg : b < 0) :
  (∀ x, x^2 + b * x + 1 / 4 = (x + n)^2 + 1 / 18) → b = - (Real.sqrt 7) / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l884_88488


namespace NUMINAMATH_GPT_max_integer_value_of_f_l884_88446

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 9 * x + 13) / (3 * x^2 + 9 * x + 5)

theorem max_integer_value_of_f : ∀ x : ℝ, ∃ n : ℤ, f x ≤ n ∧ n = 2 :=
by 
  sorry

end NUMINAMATH_GPT_max_integer_value_of_f_l884_88446


namespace NUMINAMATH_GPT_ott_fractional_part_l884_88485

theorem ott_fractional_part (x : ℝ) :
  let moe_initial := 6 * x
  let loki_initial := 5 * x
  let nick_initial := 4 * x
  let ott_initial := 1
  
  let moe_given := (x : ℝ)
  let loki_given := (x : ℝ)
  let nick_given := (x : ℝ)
  
  let ott_returned_each := (1 / 10) * x
  
  let moe_effective := moe_given - ott_returned_each
  let loki_effective := loki_given - ott_returned_each
  let nick_effective := nick_given - ott_returned_each
  
  let ott_received := moe_effective + loki_effective + nick_effective
  let ott_final_money := ott_initial + ott_received
  
  let total_money_original := moe_initial + loki_initial + nick_initial + ott_initial
  let fraction_ott_final := ott_final_money / total_money_original
  
  ott_final_money / total_money_original = (10 + 27 * x) / (150 * x + 10) :=
by
  sorry

end NUMINAMATH_GPT_ott_fractional_part_l884_88485


namespace NUMINAMATH_GPT_total_pencils_l884_88424

-- Define the initial conditions
def initial_pencils : ℕ := 41
def added_pencils : ℕ := 30

-- Define the statement to be proven
theorem total_pencils :
  initial_pencils + added_pencils = 71 :=
by
  sorry

end NUMINAMATH_GPT_total_pencils_l884_88424


namespace NUMINAMATH_GPT_arithmetic_sequence_fourth_term_l884_88439

theorem arithmetic_sequence_fourth_term (b d : ℝ) (h : 2 * b + 2 * d = 10) : b + d = 5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_fourth_term_l884_88439
